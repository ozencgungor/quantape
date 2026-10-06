#ifndef QUANTAPE_MATH_OPTIMIZATION_CALIBRATION_PROBLEM_H
#define QUANTAPE_MATH_OPTIMIZATION_CALIBRATION_PROBLEM_H

//
// CalibrationProblem.h -- model-generic calibration + IFT/KKT layer
//
// The generic contract between models and the calibration/risk machinery:
//
//   b : model parameters (nB)
//   a : market data shared by all quotes (nA)
//   i : quote index (nQ)
//
// A model provides one residual block per quote (value, derivatives wrt b and
// a, second derivatives) and optionally inequality constraints g_k(b) <= 0
// (v1: constraints do not depend on a, matching ImplicitFunction.h). The
// residual is expected to carry its own sqrt(weight); the objective is
//
//   f(b, a) = 1/2 sum_i r_i(b, a)^2.
//
// Everything downstream is assembled here, once, for every model:
//
//   * `assembleCalibration`  -> objective value/gradient, residual Jacobians,
//                               exact LS Hessian (J'WJ + sum w r d2P) and the
//                               mixed block d2f/db da
//   * `calibrationIft`       -> db/da (and dlambda/da) by implicit
//                               differentiation of grad_b f = 0 or the KKT
//                               system; ridge-regularized solve with
//                               conditioning diagnostics
//   * `CalibrationValueGrad` / `CalibrationInequalityConstraint` -> plug the
//     same problem into any optimizer in Math/Optimization (LBFGS<double>,
//     SLSQP<double>, AugLag<double>, ...)
//
// The layer is scalar-generic: `residual<S>`/`inequalities<S>` are templates,
// so the same model adapter serves
//
//   * the analytic double path (`CalibrationValueGrad`, the assembled
//     Hessian/mixed blocks, `calibrationIft`) -- fastest, no tape;
//   * the Stan AD stack (`var`, `fvar<var>`): `CalibrationObjective` /
//     `CalibrationConstraintWriter` plug straight into LBFGS<var>,
//     TNewton<var>, SLSQP<var>/AugLag<var> (OptimizerStanPrimitives.h) and
//     into the implicit-function layer (ImplicitFunction.h f2/g2 shape).
//     Models are expected to price through `make_callback_var`
//     (models/HestonStanPrimitives.h) so the tape holds one node per quote.
//
// The core header has no hard StanMath.h include (the repo's usual core/AD
// split); the AD instantiations are exercised by tests and by the model
// adapters, which include the Stan glue themselves.
//

#include <Eigen/Dense>

#include <algorithm>
#include <cmath>
#include <concepts>
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <utility>
#include <vector>

namespace quantape::math {

/// Derivative order requested from a model: skip work that is not needed.
enum class CalibrationOrder { Value = 0, Gradient = 1, Hessian = 2 };

constexpr bool operator>=(CalibrationOrder lhs, CalibrationOrder rhs) {
    return static_cast<int>(lhs) >= static_cast<int>(rhs);
}
constexpr bool operator<(CalibrationOrder lhs, CalibrationOrder rhs) {
    return !(lhs >= rhs);
}

/// Per-quote residual block filled by the model (`r = sqrt(w)(P_i - target)`).
struct CalibrationResidualBlock {
    double value = 0.0;        ///< residual r_i
    Eigen::VectorXd gradientB; ///< dr_i/db (nB)
    Eigen::VectorXd gradientA; ///< dr_i/da (nA)
    Eigen::MatrixXd hessianBB; ///< d2r_i/db2 (nB x nB)
    Eigen::MatrixXd hessianBA; ///< d2r_i/db da (nB x nA)
};

/// Per-constraint block filled by the model (`g_k(b) <= 0`).
struct CalibrationInequalityBlock {
    double value = 0.0;        ///< g_k
    Eigen::VectorXd gradientB; ///< dg_k/db (nB)
    Eigen::MatrixXd hessianBB; ///< d2g_k/db2 (nB x nB)
};

/**
 * @brief Model adapter concept for calibration and IFT/KKT risk propagation
 *
 * A model fills, for quote `i` and parameter point `(b, a)`, the requested
 * derivative order. Sizes must match `numModelParams()`/`numMarketParams()`;
 * `fillResidual` must at least set `value` for `CalibrationOrder::Value`,
 * additionally both gradients for `Gradient`, and both Hessians for
 * `Hessian`. `fillInequality` is only called for indices below
 * `numInequalities()`.
 */
template <typename P>
concept CalibrationProblem =
    requires(const P& p, std::size_t i, const Eigen::VectorXd& b, const Eigen::VectorXd& a,
             CalibrationOrder order, CalibrationResidualBlock& residual,
             CalibrationInequalityBlock& inequality) {
        { p.numModelParams() } -> std::convertible_to<std::size_t>;
        { p.numMarketParams() } -> std::convertible_to<std::size_t>;
        { p.numQuotes() } -> std::convertible_to<std::size_t>;
        { p.numInequalities() } -> std::convertible_to<std::size_t>;
        p.fillResidual(i, b, a, order, residual);
        p.fillInequality(i, b, inequality);
    };

/**
 * @brief Scalar-generic model view: `residual<S>` (and `inequalities<S>` for
 *        constrained problems) usable at any scalar the model supports
 *        (double, `stan::math::var`, `stan::math::fvar<...>`).
 *
 * The objective convention is `f = 1/2 sum_i r_i^2`; the residual carries its
 * own sqrt(weight). AD scalars are expected to price through
 * `make_callback_var` so no tape is built through the quadrature.
 */
template <typename P, typename S>
concept ScalarCalibrationProblem = requires(const P& p, std::size_t i, const std::vector<S>& b,
                                            const std::vector<S>& a, std::vector<S>& out) {
    { p.numModelParams() } -> std::convertible_to<std::size_t>;
    { p.numMarketParams() } -> std::convertible_to<std::size_t>;
    { p.numQuotes() } -> std::convertible_to<std::size_t>;
    { p.template residual<S>(i, b, a) } -> std::same_as<S>;
};

/// The double instantiation is the minimum every adapter must provide.
template <typename P>
concept ScalarCalibrationModel = ScalarCalibrationProblem<P, double>;

/**
 * @brief Scalar-generic weighted-LS objective
 *
 * Supports both call shapes used across the stack:
 *   - `f(b, a)`  -- ImplicitFunction.h `f2` (data explicit, mixed Hessians)
 *   - `f(b)`     -- optimizer objectives (market captured as constants;
 *                   depends only on `b`, so gradient backends see exactly
 *                   the model-parameter leaves)
 */
template <typename P>
    requires ScalarCalibrationModel<P>
class CalibrationObjective {
public:
    CalibrationObjective(const P& problem, std::vector<double> market)
        : m_problem(&problem), m_market(std::move(market)) {}

    /// ImplicitFunction.h `f2` shape: the model and market scalars are
    /// deduced independently and promoted (`fvar<var>` x with double data,
    /// or both `fvar<var>` in the mixed passes).
    template <typename Sx, typename Sm>
    auto operator()(const std::vector<Sx>& b, const std::vector<Sm>& a) const {
        using S = decltype(Sx(0.0) + Sm(0.0));
        std::vector<S> bPromoted(b.begin(), b.end());
        std::vector<S> aPromoted(a.begin(), a.end());
        const std::size_t nQ = m_problem->numQuotes();
        S value = S(0.0);
        for (std::size_t i = 0; i < nQ; ++i) {
            const S r = m_problem->template residual<S>(i, bPromoted, aPromoted);
            value += S(0.5) * r * r;
        }
        return value;
    }

    /// Optimizer shape: market captured as scalar constants.
    template <typename S>
    S operator()(const std::vector<S>& b) const {
        std::vector<S> a(m_market.size());
        for (std::size_t j = 0; j < m_market.size(); ++j) {
            a[j] = S(m_market[j]);
        }
        return (*this)(b, a);
    }

private:
    const P* m_problem;
    std::vector<double> m_market;
};

/**
 * @brief Scalar-generic inequality writer for AD optimizers / ImplicitFunction
 *
 * Forwards to `problem.inequalities<S>(b, out)` (g <= 0, all rows at once);
 * `AugLag<var>`/`SLSQP<var>` take it as the scalar-generic constraint and
 * ImplicitFunction.h as `g2`.
 */
template <typename P>
    requires ScalarCalibrationModel<P>
struct CalibrationConstraintWriter {
    const P* problem = nullptr;

    /// Optimizer writer shape (AugLag/SLSQP scalar-generic constraints).
    template <typename S>
    void operator()(const std::vector<S>& b, std::vector<S>& out) const {
        problem->template inequalities<S>(b, out);
    }

    /// ImplicitFunction.h `g2` shape (x, m, out).
    template <typename Sx, typename Sm>
    void operator()(const std::vector<Sx>& b, const std::vector<Sm>&, std::vector<Sx>& out) const {
        problem->template inequalities<Sx>(b, out);
    }
};

/// Assembled first/second derivatives of the weighted LS objective at (b, a).
struct CalibrationDerivatives {
    std::size_t nB = 0;
    std::size_t nA = 0;
    std::size_t nQuotes = 0;
    double value = 0.0;        ///< 1/2 sum r_i^2
    Eigen::VectorXd residuals; ///< r (nQ)
    Eigen::VectorXd gradientB; ///< df/db (nB)
    Eigen::VectorXd gradientA; ///< df/da (nA)
    Eigen::MatrixXd jacobianB; ///< dr/db (nQ x nB)
    Eigen::MatrixXd jacobianA; ///< dr/da (nQ x nA)
    Eigen::MatrixXd hessianBB; ///< d2f/db2 (nB x nB)
    Eigen::MatrixXd mixedBA;   ///< d2f/db da (nB x nA)
};

/// Assemble the objective derivatives; `Hessian` order adds the O(r) curvature
/// terms (exact LS Hessian) and the mixed block. `Value` order computes only
/// the residuals and value.
template <CalibrationProblem P>
CalibrationDerivatives assembleCalibration(const P& problem, const Eigen::VectorXd& b,
                                           const Eigen::VectorXd& a,
                                           CalibrationOrder order = CalibrationOrder::Hessian) {
    const std::size_t nB = problem.numModelParams();
    const std::size_t nA = problem.numMarketParams();
    const std::size_t nQ = problem.numQuotes();
    if (static_cast<std::size_t>(b.size()) != nB || static_cast<std::size_t>(a.size()) != nA) {
        throw std::invalid_argument("assembleCalibration: parameter size mismatch");
    }

    CalibrationDerivatives out;
    out.nB = nB;
    out.nA = nA;
    out.nQuotes = nQ;
    out.residuals.resize(static_cast<Eigen::Index>(nQ));
    const bool withGradient =
        static_cast<int>(order) >= static_cast<int>(CalibrationOrder::Gradient);
    const bool withHessian = static_cast<int>(order) >= static_cast<int>(CalibrationOrder::Hessian);
    if (withGradient) {
        out.jacobianB.resize(static_cast<Eigen::Index>(nQ), static_cast<Eigen::Index>(nB));
        out.jacobianA.resize(static_cast<Eigen::Index>(nQ), static_cast<Eigen::Index>(nA));
        out.gradientB.setZero(static_cast<Eigen::Index>(nB));
        out.gradientA.setZero(static_cast<Eigen::Index>(nA));
    }
    if (withHessian) {
        out.hessianBB.setZero(static_cast<Eigen::Index>(nB), static_cast<Eigen::Index>(nB));
        out.mixedBA.setZero(static_cast<Eigen::Index>(nB), static_cast<Eigen::Index>(nA));
    }

    CalibrationResidualBlock block;
    for (std::size_t i = 0; i < nQ; ++i) {
        problem.fillResidual(i, b, a, order, block);
        const double r = block.value;
        out.residuals(static_cast<Eigen::Index>(i)) = r;
        out.value += 0.5 * r * r;
        if (!withGradient) {
            continue;
        }
        if (static_cast<std::size_t>(block.gradientB.size()) != nB ||
            static_cast<std::size_t>(block.gradientA.size()) != nA) {
            throw std::invalid_argument("assembleCalibration: gradient size mismatch");
        }
        out.jacobianB.row(static_cast<Eigen::Index>(i)) = block.gradientB.transpose();
        out.jacobianA.row(static_cast<Eigen::Index>(i)) = block.gradientA.transpose();
        out.gradientB += r * block.gradientB;
        out.gradientA += r * block.gradientA;
        if (withHessian) {
            out.hessianBB.noalias() +=
                block.gradientB * block.gradientB.transpose() + r * block.hessianBB;
            out.mixedBA.noalias() +=
                block.gradientB * block.gradientA.transpose() + r * block.hessianBA;
        }
    }
    return out;
}

/// Diagnostics of a calibration IFT solve.
struct CalibrationIftDiagnostics {
    double condition = 0.0;
    double ridge = 0.0;
    bool regularized = false;
    bool pseudoInverse = false;
    std::size_t rank = 0;
    std::vector<std::size_t> activeInequalities;
};

struct CalibrationIftOptions {
    double feasibilityTol = 1e-6; ///< |g| <= tol counts as active (covers both
                                  ///< solvers' feasibility exits: SLSQP ~1e-7,
                                  ///< AUGLAG ~1e-9, on constraints of scale O(1))
    double lambdaTol = 1e-10;     ///< multiplier > tol counts as strictly active
    double ridge = 0.0;           ///< explicit ridge (0 = automatic minimum)
    int maxRidgeTries = 30;
};

/// db/da and dlambda/da of a calibrated (KKT) optimum.
struct CalibrationIftResult {
    Eigen::MatrixXd dbda;      ///< nB x nA
    Eigen::MatrixXd dlambdaDa; ///< active multipliers x nA (row-major)
    CalibrationIftDiagnostics diagnostics;
};

namespace detail {

/// Solve H X = B for symmetric H; minimal ridge escalation when H is not
/// positive definite (same policy as ImplicitFunction.h).
inline void solveSymmetricRidge(const Eigen::MatrixXd& H, const Eigen::MatrixXd& B,
                                Eigen::MatrixXd& X, const CalibrationIftOptions& options,
                                CalibrationIftDiagnostics& out) {
    const Eigen::Index n = H.rows();
    Eigen::MatrixXd Hr = H;
    double ridge = options.ridge;
    int tries = 0;
    Eigen::LLT<Eigen::MatrixXd> llt(Hr);
    while (llt.info() != Eigen::Success && tries < options.maxRidgeTries) {
        ++tries;
        if (ridge <= 0.0) {
            Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es(Hr);
            const double lmin = es.eigenvalues().minCoeff();
            double scale = 0.0;
            for (Eigen::Index i = 0; i < n; ++i) {
                scale = std::max(scale, std::fabs(H(i, i)));
            }
            ridge = std::max(1e-14 * std::max(1.0, scale), -lmin * (1.0 + 1e-6));
        } else {
            ridge *= 10.0;
        }
        Hr = H;
        Hr.diagonal().array() += ridge;
        llt.compute(Hr);
    }
    if (llt.info() != Eigen::Success) {
        throw std::runtime_error(
            "calibrationIft: Hessian stayed indefinite after ridge escalation");
    }
    X = llt.solve(B);
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es(Hr);
    out.condition = es.eigenvalues().maxCoeff() / std::max(1e-300, es.eigenvalues().minCoeff());
    out.regularized = ridge > 0.0;
    out.ridge = ridge;
}

} // namespace detail

/**
 * @brief Recover active inequality multipliers from stationarity at a
 *        calibrated point (calibrator-agnostic)
 *
 * Solves `min_{lambda >= 0} || grad_b f + sum_k lambda_k grad g_k ||^2` over
 * the constraints detected active (`|g_k| <= feasibilityTol`), dropping
 * negative rows (Lawson-Hanson style). This removes the dependence on which
 * calibrator produced the point (AUGLAG and SLSQP differ in when they export
 * their multipliers, and both stop at slightly different interior/boundary
 * points).
 */
template <CalibrationProblem P>
std::vector<double> recoverActiveMultipliers(const P& problem, const Eigen::VectorXd& bHat,
                                             const Eigen::VectorXd& gradientB,
                                             const CalibrationIftOptions& options = {}) {
    const std::size_t nIneq = problem.numInequalities();
    const std::size_t nB = problem.numModelParams();
    std::vector<double> lambda(nIneq, 0.0);
    if (nIneq == 0) {
        return lambda;
    }
    CalibrationInequalityBlock block;
    std::vector<std::size_t> candidates;
    for (std::size_t k = 0; k < nIneq; ++k) {
        problem.fillInequality(k, bHat, block);
        if (std::fabs(block.value) <= options.feasibilityTol) {
            candidates.push_back(k);
        }
    }
    if (candidates.empty()) {
        return lambda;
    }
    std::vector<Eigen::VectorXd> rows;
    for (std::size_t k : candidates) {
        problem.fillInequality(k, bHat, block);
        rows.push_back(block.gradientB);
    }
    // Active-set refinement: solve the least-squares stationarity system,
    // drop the most negative multiplier, repeat.
    while (!rows.empty()) {
        Eigen::MatrixXd A(static_cast<Eigen::Index>(rows.size()), static_cast<Eigen::Index>(nB));
        for (std::size_t r = 0; r < rows.size(); ++r) {
            A.row(static_cast<Eigen::Index>(r)) = rows[r].transpose();
        }
        const Eigen::VectorXd rhs = -(A * gradientB);
        const Eigen::MatrixXd gram = A * A.transpose(); // normal equations
        const Eigen::VectorXd sol = gram.ldlt().solve(rhs);
        if (!sol.allFinite() || sol.size() != static_cast<Eigen::Index>(rows.size())) {
            break;
        }
        Eigen::Index worst = -1;
        double worstValue = -1e-12;
        for (Eigen::Index r = 0; r < sol.size(); ++r) {
            if (sol(r) < worstValue) {
                worst = r;
                worstValue = sol(r);
            }
        }
        if (worst < 0) {
            for (std::size_t r = 0; r < rows.size(); ++r) {
                lambda[candidates[r]] = sol(static_cast<Eigen::Index>(r));
            }
            return lambda;
        }
        candidates.erase(candidates.begin() + static_cast<std::ptrdiff_t>(worst));
        rows.erase(rows.begin() + static_cast<std::ptrdiff_t>(worst));
    }
    return lambda;
}

/**
 * @brief db/da (and dlambda/da) at a calibrated optimum by IFT of the KKT system
 *
 * Unconstrained (no active inequalities): db/da = -H^{-1} (d2f/db da) with
 * H = d2f/db2 at the optimum. Active inequalities (|g_k| <= feasibilityTol
 * and lambda_k > lambdaTol) extend the system with J_k db = 0 and
 * H += sum lambda_k d2g_k/db2, exactly as ImplicitFunction.h does for AD
 * objectives. Constraints are assumed a-independent (v1).
 *
 * @param ineqMultipliers final multipliers from the calibrator; empty means
 *        "recover from stationarity at bHat" (calibrator-agnostic), and a
 *        non-empty vector must match numInequalities().
 */
template <CalibrationProblem P>
CalibrationIftResult calibrationIft(const P& problem, const Eigen::VectorXd& bHat,
                                    const Eigen::VectorXd& a,
                                    const std::vector<double>& ineqMultipliers = {},
                                    const CalibrationIftOptions& options = {}) {
    const std::size_t nB = problem.numModelParams();
    const std::size_t nA = problem.numMarketParams();
    const std::size_t nIneq = problem.numInequalities();
    if (ineqMultipliers.size() != 0 && ineqMultipliers.size() != nIneq) {
        throw std::invalid_argument("calibrationIft: multiplier size != numInequalities()");
    }
    const CalibrationDerivatives d =
        assembleCalibration(problem, bHat, a, CalibrationOrder::Hessian);
    const std::vector<double> lambda =
        ineqMultipliers.size() == nIneq
            ? ineqMultipliers
            : recoverActiveMultipliers(problem, bHat, d.gradientB, options);

    CalibrationIftResult result;
    result.dbda.resize(static_cast<Eigen::Index>(nB), static_cast<Eigen::Index>(nA));
    result.dlambdaDa.resize(0, static_cast<Eigen::Index>(nA));

    // Active set + Lagrangian Hessian
    Eigen::MatrixXd lagrangianH = d.hessianBB;
    std::vector<Eigen::VectorXd> activeGradients;
    CalibrationInequalityBlock block;
    for (std::size_t k = 0; k < nIneq; ++k) {
        problem.fillInequality(k, bHat, block);
        if (std::fabs(block.value) <= options.feasibilityTol && lambda[k] > options.lambdaTol) {
            result.diagnostics.activeInequalities.push_back(k);
            activeGradients.push_back(block.gradientB);
            lagrangianH.noalias() += lambda[k] * block.hessianBB;
        }
    }
    const std::size_t mA = activeGradients.size();

    if (mA == 0) {
        Eigen::MatrixXd X;
        detail::solveSymmetricRidge(d.hessianBB, -d.mixedBA, X, options, result.diagnostics);
        result.dbda = X;
        return result;
    }

    // KKT system [ H_L  J^T ] [ db    ]   [ -G ]
    //            [ diag(lambda) J ] [ dlam ] = [  0 ]
    const Eigen::Index s = static_cast<Eigen::Index>(nB + mA);
    Eigen::MatrixXd K = Eigen::MatrixXd::Zero(s, s);
    Eigen::MatrixXd R = Eigen::MatrixXd::Zero(s, static_cast<Eigen::Index>(nA));
    K.topLeftCorner(static_cast<Eigen::Index>(nB), static_cast<Eigen::Index>(nB)) = lagrangianH;
    for (std::size_t k = 0; k < mA; ++k) {
        const std::size_t row = result.diagnostics.activeInequalities[k];
        K.topRightCorner(static_cast<Eigen::Index>(nB), 1).col(static_cast<Eigen::Index>(k)) =
            activeGradients[k];
        K.bottomLeftCorner(1, static_cast<Eigen::Index>(nB)).row(static_cast<Eigen::Index>(k)) =
            (lambda[row] * activeGradients[k]).transpose();
    }
    R.topRows(static_cast<Eigen::Index>(nB)) = -d.mixedBA;

    Eigen::FullPivLU<Eigen::MatrixXd> lu(K);
    const Eigen::MatrixXd Y = lu.solve(R);
    if (lu.isInvertible()) {
        result.diagnostics.condition = 1.0 / lu.rcond();
    } else {
        result.diagnostics.pseudoInverse = true;
        result.diagnostics.rank = lu.rank();
        result.diagnostics.condition = std::numeric_limits<double>::max();
    }
    result.dbda = Y.topRows(static_cast<Eigen::Index>(nB));
    result.dlambdaDa = Y.bottomRows(static_cast<Eigen::Index>(mA));
    return result;
}

/// Objective adapter for the double optimizers (LBFGS<double>, AugLag<double>).
template <CalibrationProblem P>
class CalibrationValueGrad {
public:
    CalibrationValueGrad(const P& problem, Eigen::VectorXd market)
        : m_problem(&problem), m_market(std::move(market)) {}

    double operator()(const std::vector<double>& x, std::vector<double>& grad) const {
        const Eigen::Map<const Eigen::VectorXd> b(x.data(), static_cast<Eigen::Index>(x.size()));
        const CalibrationDerivatives d =
            assembleCalibration(*m_problem, b, m_market, CalibrationOrder::Gradient);
        grad.resize(x.size());
        for (Eigen::Index i = 0; i < static_cast<Eigen::Index>(x.size()); ++i) {
            grad[static_cast<std::size_t>(i)] = d.gradientB(i);
        }
        return d.value;
    }

    const Eigen::VectorXd& market() const { return m_market; }

private:
    const P* m_problem;
    Eigen::VectorXd m_market;
};

/// Inequality adapter for AugLag&lt;double&gt;/SLSQP&lt;double&gt; (`c_k = g_k <= 0`,
/// row-major Jacobian).
template <CalibrationProblem P>
struct CalibrationInequalityConstraint {
    const P* problem = nullptr;

    void operator()(const std::vector<double>& x, std::vector<double>& c,
                    std::vector<double>& jacobian) const {
        const Eigen::Map<const Eigen::VectorXd> b(x.data(), static_cast<Eigen::Index>(x.size()));
        const std::size_t nIneq = problem->numInequalities();
        const std::size_t nB = problem->numModelParams();
        c.resize(nIneq);
        jacobian.assign(nIneq * nB, 0.0);
        CalibrationInequalityBlock block;
        for (std::size_t k = 0; k < nIneq; ++k) {
            problem->fillInequality(k, b, block);
            c[k] = block.value;
            for (std::size_t j = 0; j < nB; ++j) {
                jacobian[k * nB + j] = block.gradientB(static_cast<Eigen::Index>(j));
            }
        }
    }
};

/// Constraint values only (for feasibility checks/reporting).
template <CalibrationProblem P>
Eigen::VectorXd calibrationInequalityValues(const P& problem, const Eigen::VectorXd& b) {
    const std::size_t nIneq = problem.numInequalities();
    Eigen::VectorXd out(static_cast<Eigen::Index>(nIneq));
    CalibrationInequalityBlock block;
    for (std::size_t k = 0; k < nIneq; ++k) {
        problem.fillInequality(k, b, block);
        out(static_cast<Eigen::Index>(k)) = block.value;
    }
    return out;
}

} // namespace quantape::math

#endif // QUANTAPE_MATH_OPTIMIZATION_CALIBRATION_PROBLEM_H
