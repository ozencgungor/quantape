#ifndef IMPLICIT_FUNCTION_H
#define IMPLICIT_FUNCTION_H

//
// ImplicitFunction.h -- first-order sensitivities of optima (Stan-dependent)
//
// The IFT layer (docs/ad_optimizers.md §6.1/§6.2): given a converged
// optimizer solution, compute dp/dm -- the Jacobian of the optimal
// parameters w.r.t. the market data -- by implicit differentiation of
// either grad L = 0 (unconstrained) or the KKT system (constrained).
//
// All derivatives are EXACT (dense Hessians from n HVPs, mixed Hessians
// from forward-over-reverse seeds on m); finite differences appear only in
// the tests as cross-checks.
//
// Callables take the data explicitly (m is a separate argument, so the AD
// layer can seed it):
//
//   f2 : S f(const std::vector<Sx>& x, const std::vector<Sm>& m)
//   g2 : void g(const std::vector<Sx>& x, const std::vector<Sm>& m, std::vector<Sx>& out)
//   h2 : void h(const std::vector<Sx>& x, const std::vector<Sm>& m, std::vector<Sx>& out)
//
// Sx/Sm are deduced independently: x must support var and fvar<var> (for
// HVPs and the mixed Hessian), m only needs double for the solve and
// fvar<var> for the mixed passes. g2/h2 must accept Sx == var.
//
// v1 scope (documented in §6.2): the constraints are assumed NOT to depend
// on m directly (dg/dm = 0, dh/dm = 0) -- true for no-arbitrage/structural
// constraints on the model surface, which are functions of the parameters
// only. Bounds are folded in as exact linear rows (x_j = b_j => dp_j = 0).
//
// Tape hygiene: every internal AD pass runs in its own
// nested_rev_autodiff scope; minimizeDifferentialVar builds one small graph
// on the CALLER's tape (callback vars whose adjoints land on m).
//

#include "Math/StanMath.h"

#include <Eigen/Dense>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <utility>
#include <vector>

#include "Constraint.h"
#include "LBFGS.h"
#include "OptimizerPrimitives.h"
#include "OptimizerStanPrimitives.h"
#include "SLSQP.h"

namespace Math {

/// Tolerances for active-set detection and regularization
struct IftOptions {
    // Active-set detection: |g| <= feasibility_tol AND multiplier > lambda_tol.
    // 1e-8 comfortably covers both solvers' own feasibility tolerances
    // (SLSQP's is exact-zero in practice, AUGLAG's is 1e-9) while staying far
    // below any genuinely-inactive |g| ~ O(1).
    double feasibility_tol = 1e-8;
    double lambda_tol = 1e-10; ///< multiplier > tol => strictly active
    double bound_tol = 1e-8;   ///< |x - bound| <= tol*scale => active bound
    double ridge = 0.0;        ///< explicit ridge for H (0 = automatic)
    int max_ridge_tries = 30;  ///< ridge escalation attempts
};

/// Diagnostics of an IFT solve
struct IftResult {
    double condition_number = 0.0;          ///< lambda_max/lambda_min or 1/rcond(K)
    double ridge_used = 0.0;                ///< regularization added to H (0 = none)
    bool regularized = false;               ///< H was not positive definite
    bool pseudo_inverse = false;            ///< KKT system used the pseudo-inverse path
    std::size_t rank = 0;                   ///< rank used by the pseudo-inverse solve
    std::vector<std::size_t> active_ineq;   ///< indices of active inequality rows
    std::vector<std::size_t> active_bounds; ///< parameters pinned by a bound
};

namespace detail {

using fvar = stan::math::fvar<stan::math::var>;
using var = stan::math::var;

/// Dense Hessian of an x-only callable: n exact HVPs, buffers reused
template <typename F>
void denseHessian(const F& f, const std::vector<double>& x, std::vector<double>& H,
                  std::vector<fvar>& theta, std::vector<double>& e, std::vector<double>& col) {
    const std::size_t n = x.size();
    H.assign(n * n, 0.0);
    e.assign(n, 0.0);
    for (std::size_t j = 0; j < n; ++j) {
        e[j] = 1.0;
        hvpInto(f, x, e, theta, col);
        e[j] = 0.0;
        for (std::size_t i = 0; i < n; ++i) {
            H[i * n + j] = col[i];
        }
    }
}

/// Mixed Hessian d^2 f / (dx_i dm_j), row-major n x M.
/// Column j: seed m_j with a unit tangent (forward), differentiate the
/// resulting directional derivative df/dm_j back to x (reverse).
template <typename F2>
void mixedHessian(const F2& f2, const std::vector<double>& x, const std::vector<double>& m,
                  std::vector<double>& G, std::vector<fvar>& xv, std::vector<fvar>& mv) {
    const std::size_t n = x.size();
    const std::size_t M = m.size();
    G.assign(n * M, 0.0);
    xv.resize(n);
    mv.resize(M);
    for (std::size_t j = 0; j < M; ++j) {
        stan::math::nested_rev_autodiff nested;
        for (std::size_t i = 0; i < n; ++i) {
            xv[i] = fvar(var(x[i]), var(0.0));
        }
        for (std::size_t k = 0; k < M; ++k) {
            mv[k] = fvar(var(m[k]), var(k == j ? 1.0 : 0.0));
        }
        fvar y = f2(xv, mv);
        y.d_.grad();
        for (std::size_t i = 0; i < n; ++i) {
            G[i * M + j] = xv[i].val_.adj();
        }
    }
}

/// Solve H X = B for a symmetric H; LLT (Cholesky) fails unless H is
/// positive definite, so a failure triggers the automatic ridge escalation.
/// The ridge is chosen to be MINIMAL: the eigen gap below zero, times
/// (1 + 1e-6) — enough to flip the smallest eigenvalue positive without
/// distorting the well-conditioned part. The eigendecomposition (computed
/// once, after the solve) supplies the condition number.
/// Returns the ridge actually used (0 = none).
inline double solvePositiveDefinite(const Eigen::MatrixXd& H, const Eigen::MatrixXd& B,
                                    Eigen::MatrixXd& X, const IftOptions& options, IftResult& out) {
    const std::size_t n = H.rows();
    Eigen::MatrixXd Hr = H;
    double ridge = options.ridge;

    int tries = 0;
    Eigen::LLT<Eigen::MatrixXd> llt(Hr);
    while (llt.info() != Eigen::Success && tries < options.max_ridge_tries) {
        ++tries;
        if (ridge <= 0.0) {
            // Minimal distortion: flip the most negative eigenvalue positive
            Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es(Hr);
            const double lmin = es.eigenvalues().minCoeff();
            double scale = 0.0;
            for (std::size_t i = 0; i < n; ++i) {
                scale = std::max(scale, std::fabs(H(i, i)));
            }
            ridge = std::max(1e-14 * std::max(1.0, scale), -lmin * (1.0 + 1e-6));
        } else {
            ridge *= 10.0;
        }
        Hr = H;
        for (std::size_t i = 0; i < n; ++i) {
            Hr(i, i) += ridge;
        }
        llt.compute(Hr);
    }
    if (llt.info() != Eigen::Success) {
        throw std::runtime_error("iftUnconstrained: Hessian stayed indefinite "
                                 "after ridge escalation");
    }
    X = llt.solve(B);

    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es(Hr);
    const double lmin = es.eigenvalues().minCoeff();
    const double lmax = es.eigenvalues().maxCoeff();
    out.condition_number = lmax / std::max(lmin, 1e-300);
    out.regularized = ridge > 0.0;
    out.ridge_used = ridge;
    return ridge;
}

} // namespace detail

/**
 * @brief Unconstrained first-order IFT: dp/dm = -H^{-1} G
 *
 * H = d^2 f / dx^2 at (x_hat, m) via n exact HVPs; G = d^2 f / dx dm via
 * forward-over-reverse seeds on m. If H is not positive definite an
 * automatic ridge escalation regularizes it (reported via `out`).
 *
 * @param[out] dp_dm row-major n x M Jacobian of the optimum
 */
template <typename F2>
void iftUnconstrained(const F2& f2, const std::vector<double>& x_hat, const std::vector<double>& m,
                      std::vector<double>& dp_dm, IftResult& out, const IftOptions& options = {}) {
    const std::size_t n = x_hat.size();
    const std::size_t M = m.size();
    if (n == 0) {
        throw std::invalid_argument("iftUnconstrained: empty parameter vector");
    }
    out = IftResult{};
    dp_dm.resize(n * M);
    if (M == 0) {
        return;
    }

    const auto fx = [&](const auto& theta) { return f2(theta, m); };

    std::vector<detail::fvar> theta, xv, mv;
    std::vector<double> e, col, H, G;
    detail::denseHessian(fx, x_hat, H, theta, e, col);
    detail::mixedHessian(f2, x_hat, m, G, xv, mv);

    Eigen::Map<const Eigen::MatrixXd> Hm(H.data(), static_cast<Eigen::Index>(n),
                                         static_cast<Eigen::Index>(n));
    Eigen::MatrixXd RHS(n, M);
    for (std::size_t i = 0; i < n; ++i) {
        for (std::size_t j = 0; j < M; ++j) {
            RHS(static_cast<Eigen::Index>(i), static_cast<Eigen::Index>(j)) = -G[i * M + j];
        }
    }
    Eigen::MatrixXd X;
    detail::solvePositiveDefinite(Hm, RHS, X, options, out);
    for (std::size_t i = 0; i < n; ++i) {
        for (std::size_t j = 0; j < M; ++j) {
            dp_dm[i * M + j] = X(static_cast<Eigen::Index>(i), static_cast<Eigen::Index>(j));
        }
    }
}

/**
 * @brief KKT first-order IFT (docs/ad_optimizers.md §6.2)
 *
 * Active inequalities: |g_i| <= feasibility_tol AND lambda_i > lambda_tol
 * (strict complementarity; degenerate rows fall to the pseudo-inverse path).
 * Active bounds are folded in as exact linear rows. The square system
 *
 *   [ H_L   J_A^T  J_h^T  J_b^T ] [ dp  ]     [ -L_xm ]
 *   [ diag(lA) J_A  0     0    ] [ dlA ]  =  [  0    ]
 *   [ J_h     0     0     0    ] [ dnu ]     [  0    ]
 *   [ J_b     0     0     0    ] [ dmu ]     [  0    ]
 *
 * is solved with PartialPivLU, falling back to a complete orthogonal
 * decomposition (minimum-norm solution) when rcond < pseudo_inverse_tol.
 * H_L is the full Lagrangian Hessian (all multipliers; inactive rows with
 * lambda = 0 contribute nothing).
 *
 * @param[in]  ineq_lambda inequality multipliers (size = #rows of g2)
 * @param[in]  eq_nu       equality multipliers (size = #rows of h2)
 * @param[out] dp_dm       row-major n x M
 * @param[out] dlambda_dm  row-major |A| x M (active inequalities)
 * @param[out] dnu_dm      row-major (me + |active bounds|) x M
 */
template <typename F2, typename G2, typename H2>
void iftKkt(const F2& f2, const G2& g2, const H2& h2, const Bounds& bounds,
            const std::vector<double>& x_hat, const std::vector<double>& m,
            const std::vector<double>& ineq_lambda, const std::vector<double>& eq_nu,
            std::vector<double>& dp_dm, std::vector<double>& dlambda_dm,
            std::vector<double>& dnu_dm, IftResult& out, const IftOptions& options = {}) {
    constexpr bool has_g = !std::is_same_v<std::decay_t<G2>, NoConstraint>;
    constexpr bool has_h = !std::is_same_v<std::decay_t<H2>, NoConstraint>;

    const std::size_t n = x_hat.size();
    const std::size_t M = m.size();
    out = IftResult{};
    dp_dm.resize(n * M);
    dlambda_dm.clear();
    dnu_dm.clear();
    if (n == 0) {
        throw std::invalid_argument("iftKkt: empty parameter vector");
    }
    if (M == 0) {
        return;
    }

    // ── Constraint values + Jacobians at x_hat (one AD evaluation each)
    std::vector<double> c, Jg, ch, Jh;
    if constexpr (has_g) {
        const auto gx = [&](const auto& theta, auto& out_v) { g2(theta, m, out_v); };
        detail::constraintValueJacobian<detail::var>(gx, x_hat, c, Jg);
        if (ineq_lambda.size() != c.size()) {
            throw std::invalid_argument("iftKkt: ineq_lambda size != #inequalities");
        }
    }
    if constexpr (has_h) {
        const auto hx = [&](const auto& theta, auto& out_v) { h2(theta, m, out_v); };
        detail::constraintValueJacobian<detail::var>(hx, x_hat, ch, Jh);
        if (eq_nu.size() != ch.size()) {
            throw std::invalid_argument("iftKkt: eq_nu size != #equalities");
        }
    }

    // ── Active set
    const std::size_t mi_all = c.size();
    const std::size_t me = ch.size();
    for (std::size_t i = 0; i < mi_all; ++i) {
        if (std::fabs(c[i]) <= options.feasibility_tol && ineq_lambda[i] > options.lambda_tol) {
            out.active_ineq.push_back(i);
        }
    }
    if (!bounds.empty()) {
        for (std::size_t j = 0; j < n; ++j) {
            const double scale = std::max(1.0, std::fabs(x_hat[j]));
            if (bounds.hasLower(j) && x_hat[j] - bounds.lower[j] <= options.bound_tol * scale) {
                out.active_bounds.push_back(j);
            }
            if (bounds.hasUpper(j) && bounds.upper[j] - x_hat[j] <= options.bound_tol * scale) {
                out.active_bounds.push_back(j);
            }
        }
    }
    const std::size_t mA = out.active_ineq.size();
    const std::size_t mB = out.active_bounds.size();

    if (mA == 0 && me == 0 && mB == 0) {
        // No binding constraints: degenerate to the unconstrained IFT
        iftUnconstrained(f2, x_hat, m, dp_dm, out, options);
        return;
    }

    // ── Lagrangian Hessian (full multiplier set; lambda = 0 rows add 0)
    const auto lagr = [&](const auto& theta) {
        using S = typename std::decay_t<decltype(theta)>::value_type;
        S L = f2(theta, m);
        if constexpr (has_g) {
            std::vector<S> gv;
            g2(theta, m, gv);
            for (std::size_t i = 0; i < mi_all; ++i) {
                L += S(ineq_lambda[i]) * gv[i];
            }
        }
        if constexpr (has_h) {
            std::vector<S> hv;
            h2(theta, m, hv);
            for (std::size_t k = 0; k < me; ++k) {
                L += S(eq_nu[k]) * hv[k];
            }
        }
        return L;
    };

    std::vector<detail::fvar> theta, xv, mv;
    std::vector<double> e, col, HL, G;
    detail::denseHessian(lagr, x_hat, HL, theta, e, col);
    detail::mixedHessian(f2, x_hat, m, G, xv, mv);

    // ── Assemble the square KKT system
    const std::size_t s = n + mA + me + mB;
    Eigen::MatrixXd K =
        Eigen::MatrixXd::Zero(static_cast<Eigen::Index>(s), static_cast<Eigen::Index>(s));
    Eigen::MatrixXd R =
        Eigen::MatrixXd::Zero(static_cast<Eigen::Index>(s), static_cast<Eigen::Index>(M));

    // Stationarity rows: H_L dp + J_A^T dlA + J_h^T dnu + J_b^T dmu = -L_xm
    for (std::size_t i = 0; i < n; ++i) {
        for (std::size_t jj = 0; jj < n; ++jj) {
            K(static_cast<Eigen::Index>(i), static_cast<Eigen::Index>(jj)) = HL[i * n + jj];
        }
        for (std::size_t a = 0; a < mA; ++a) {
            K(static_cast<Eigen::Index>(i), static_cast<Eigen::Index>(n + a)) =
                Jg[out.active_ineq[a] * n + i];
        }
        for (std::size_t k = 0; k < me; ++k) {
            K(static_cast<Eigen::Index>(i), static_cast<Eigen::Index>(n + mA + k)) = Jh[k * n + i];
        }
        for (std::size_t b = 0; b < mB; ++b) {
            const std::size_t j = out.active_bounds[b];
            const double side =
                x_hat[j] - bounds.lower[j] <= options.bound_tol * std::max(1.0, std::fabs(x_hat[j]))
                    ? -1.0 // x = lower: row  -x_j + lower = 0
                    : 1.0; // x = upper: row  x_j - upper = 0
            // J_b^T(b, i) = side * delta(i, j): only row i == j is nonzero
            K(static_cast<Eigen::Index>(i), static_cast<Eigen::Index>(n + mA + me + b)) =
                (i == j) ? side : 0.0;
        }
        for (std::size_t jm = 0; jm < M; ++jm) {
            R(static_cast<Eigen::Index>(i), static_cast<Eigen::Index>(jm)) = -G[i * M + jm];
        }
    }
    // Complementarity rows: diag(lA) J_A dp = 0
    for (std::size_t a = 0; a < mA; ++a) {
        const std::size_t row = out.active_ineq[a];
        for (std::size_t i = 0; i < n; ++i) {
            K(static_cast<Eigen::Index>(n + a), static_cast<Eigen::Index>(i)) =
                ineq_lambda[row] * Jg[row * n + i];
        }
    }
    // Equality rows: J_h dp = 0
    for (std::size_t k = 0; k < me; ++k) {
        for (std::size_t i = 0; i < n; ++i) {
            K(static_cast<Eigen::Index>(n + mA + k), static_cast<Eigen::Index>(i)) = Jh[k * n + i];
        }
    }
    // Bound rows: J_b dp = 0
    for (std::size_t b = 0; b < mB; ++b) {
        const std::size_t j = out.active_bounds[b];
        const double side =
            x_hat[j] - bounds.lower[j] <= options.bound_tol * std::max(1.0, std::fabs(x_hat[j]))
                ? -1.0
                : 1.0;
        K(static_cast<Eigen::Index>(n + mA + me + b), static_cast<Eigen::Index>(j)) = side;
    }

    // ── Solve: FullPivLU (robust rank/singularity detection; the free
    //    pivots of a rank-deficient system are zeroed = minimum-norm-ish
    //    solution, matching the documented pseudo-inverse fallback)
    Eigen::FullPivLU<Eigen::MatrixXd> lu(K);
    const bool invertible = lu.isInvertible();
    Eigen::MatrixXd Y = lu.solve(R);
    if (invertible) {
        out.condition_number = 1.0 / lu.rcond();
    } else {
        out.pseudo_inverse = true;
        out.rank = lu.rank();
        out.condition_number = std::numeric_limits<double>::max();
    }

    // ── Split the solution
    dlambda_dm.resize(mA * M);
    dnu_dm.resize((me + mB) * M);
    for (std::size_t i = 0; i < n; ++i) {
        for (std::size_t jm = 0; jm < M; ++jm) {
            dp_dm[i * M + jm] = Y(static_cast<Eigen::Index>(i), static_cast<Eigen::Index>(jm));
        }
    }
    for (std::size_t a = 0; a < mA; ++a) {
        for (std::size_t jm = 0; jm < M; ++jm) {
            dlambda_dm[a * M + jm] =
                Y(static_cast<Eigen::Index>(n + a), static_cast<Eigen::Index>(jm));
        }
    }
    for (std::size_t k = 0; k < me + mB; ++k) {
        for (std::size_t jm = 0; jm < M; ++jm) {
            dnu_dm[k * M + jm] =
                Y(static_cast<Eigen::Index>(n + mA + k), static_cast<Eigen::Index>(jm));
        }
    }
}

/**
 * @brief Solve + IFT in one call (docs §4.11 interface sketch)
 *
 * Runs LBFGS<var> (unconstrained) or SLSQP<var> (constrained), which
 * exports the final multipliers into `state`, then computes dp/dm (and
 * multiplier sensitivities when constrained).
 *
 * @param[in,out] x starting point; returns the optimum
 */
template <typename F2, typename G2 = NoConstraint, typename H2 = NoConstraint>
OptimizeResult
minimizeDifferential(const F2& f2, const G2& g2, const H2& h2, const Bounds& bounds,
                     const std::vector<double>& m, std::vector<double>& x, OptimizerState& state,
                     IftResult& ift_out, std::vector<double>* dp_dm = nullptr,
                     std::vector<double>* dlambda_dm = nullptr,
                     std::vector<double>* dnu_dm = nullptr, const StopCriteria& criteria = {},
                     const IftOptions& options = {}) {
    constexpr bool has_g = !std::is_same_v<std::decay_t<G2>, NoConstraint>;
    constexpr bool has_h = !std::is_same_v<std::decay_t<H2>, NoConstraint>;
    bool has_bounds = false;
    if (!bounds.empty()) {
        for (std::size_t j = 0; j < bounds.size(); ++j) {
            if (bounds.hasLower(j) || bounds.hasUpper(j)) {
                has_bounds = true;
                break;
            }
        }
    }
    constexpr bool constrained = has_g || has_h;
    const bool any_constrained = has_g || has_h || has_bounds;

    OptimizeResult result;
    const auto fx = [&](const auto& theta) { return f2(theta, m); };
    if constexpr (!constrained) {
        if (!has_bounds) {
            LBFGS<stan::math::var> solver(criteria, 10);
            result = solver.minimize(fx, x, state);
        } else {
            SLSQP<stan::math::var> solver(criteria);
            result = solver.minimize(fx, NoConstraint{}, NoConstraint{}, bounds, x, state);
        }
    } else if constexpr (has_g && has_h) {
        const auto gx = [&](const auto& theta, auto& out) { g2(theta, m, out); };
        const auto hx = [&](const auto& theta, auto& out) { h2(theta, m, out); };
        SLSQP<stan::math::var> solver(criteria);
        result = solver.minimize(fx, gx, hx, bounds, x, state);
    } else if constexpr (has_g) {
        const auto gx = [&](const auto& theta, auto& out) { g2(theta, m, out); };
        SLSQP<stan::math::var> solver(criteria);
        result = solver.minimize(fx, gx, bounds, x, state);
    } else {
        const auto hx = [&](const auto& theta, auto& out) { h2(theta, m, out); };
        SLSQP<stan::math::var> solver(criteria);
        result = solver.minimize(fx, NoConstraint{}, hx, bounds, x, state);
    }
    if (result != OptimizeResult::Success && result != OptimizeResult::GradientTolReached &&
        result != OptimizeResult::FtolReached && result != OptimizeResult::XtolReached) {
        return result;
    }

    std::vector<double> d1, d2, d3;
    if (!any_constrained) {
        iftUnconstrained(f2, x, m, d1, ift_out, options);
        if (dp_dm) {
            *dp_dm = std::move(d1);
        }
        if (dlambda_dm) {
            dlambda_dm->clear();
        }
        if (dnu_dm) {
            dnu_dm->clear();
        }
    } else {
        iftKkt(f2, g2, h2, bounds, x, m, state.ineq_multipliers, state.eq_multipliers, d1, d2, d3,
               ift_out, options);
        if (dp_dm) {
            *dp_dm = std::move(d1);
        }
        if (dlambda_dm) {
            *dlambda_dm = std::move(d2);
        }
        if (dnu_dm) {
            *dnu_dm = std::move(d3);
        }
    }
    return result;
}

/**
 * @brief Solve + IFT, with the optimum attached to the CALLER's tape
 *
 * p_hat[k] is a callback var whose adjoint flows into the m leaves through
 * dp/dm: differentiating any downstream scalar of p_hat w.r.t. m returns
 * exactly (d scalar / d p_hat) . (dp/dm). The internal solve and IFT runs
 * in nested scopes; only the callback graph stays on the caller's tape.
 *
 * @param[in]  m_var data leaves (already on the caller's tape)
 * @param[in]  x0    starting point (double)
 * @param[out] p_hat callback vars for the optimum
 */
template <typename F2, typename G2 = NoConstraint, typename H2 = NoConstraint>
OptimizeResult
minimizeDifferentialVar(const F2& f2, const G2& g2, const H2& h2, const Bounds& bounds,
                        const std::vector<stan::math::var>& m_var, const std::vector<double>& x0,
                        std::vector<stan::math::var>& p_hat, IftResult* ift_out = nullptr,
                        OptimizerState* state_out = nullptr, const StopCriteria& criteria = {},
                        const IftOptions& options = {}) {
    const std::size_t M = m_var.size();
    const std::size_t n = x0.size();
    std::vector<double> m(M);
    for (std::size_t j = 0; j < M; ++j) {
        m[j] = m_var[j].val();
    }
    std::vector<double> x = x0;
    OptimizerState state;
    IftResult ift;
    std::vector<double> dp_dm;
    const OptimizeResult result = minimizeDifferential(f2, g2, h2, bounds, m, x, state, ift, &dp_dm,
                                                       nullptr, nullptr, criteria, options);
    if (result != OptimizeResult::Success && result != OptimizeResult::GradientTolReached &&
        result != OptimizeResult::FtolReached && result != OptimizeResult::XtolReached) {
        return result;
    }

    p_hat.resize(n);
    for (std::size_t k = 0; k < n; ++k) {
        std::vector<double> row(M);
        for (std::size_t j = 0; j < M; ++j) {
            row[j] = dp_dm[k * M + j];
        }
        std::vector<stan::math::var> m_cap = m_var; // copies share the varis
        p_hat[k] = stan::math::make_callback_var(
            x[k], [m_cap = std::move(m_cap), row = std::move(row)](auto& vi) mutable {
                const double adj = vi.adj();
                for (std::size_t j = 0; j < m_cap.size(); ++j) {
                    m_cap[j].adj() += adj * row[j];
                }
            });
    }
    if (ift_out) {
        *ift_out = ift;
    }
    if (state_out) {
        *state_out = state;
    }
    return result;
}

} // namespace Math

#endif // IMPLICIT_FUNCTION_H
