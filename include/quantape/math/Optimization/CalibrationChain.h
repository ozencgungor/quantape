#ifndef QUANTAPE_MATH_OPTIMIZATION_CALIBRATION_CHAIN_H
#define QUANTAPE_MATH_OPTIMIZATION_CALIBRATION_CHAIN_H

#include "quantape/math/Optimization/Constraint.h"
#include "quantape/math/Optimization/ImplicitFunction.h"
#include "quantape/math/Optimization/OptimizerPrimitives.h"

#include <Eigen/Dense>

#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <string>
#include <vector>

namespace quantape::math {
/**
 * @file CalibrationChain.h
 * @brief Market-risk propagation through calibrations (S5d)
 *
 * Converts model-parameter risks `dV/db` into market risks `dV/da` without
 * re-calibrating, via the implicit function theorem. Two interchangeable
 * routes (both kept; benchmarked in the S5d tests):
 *
 * - **KKT route** (`kktCalibrationJacobian`): calibrate `b` to the quotes
 *   `a` with the project's optimizer stack and differentiate the optimum
 *   with `ImplicitFunction.h` (`dp/dm`, exact dense-Hessian IFT; handles
 *   constraints, regularisation and best fit).
 * - **Instrument-Jacobian route** (`instrumentCalibrationJacobian`, Savine
 *   *IFT Demystified*): `db/da = (dI/db)^+ (dI/da − dI/dc·dc/da)` with the
 *   weighted SVD pseudo-inverse of the model Jacobian `dI/db`. Best-fit,
 *   non-square and near-singular calibrations are safe by construction;
 *   rank/condition are reported instead of silently inverting.
 *
 * The two coincide at the optimum for unconstrained full-rank least-squares
 * with exact fit; for imperfect fit the KKT route is the exact derivative of
 * the (weighted) LS optimum while the instrument route is its Gauss–Newton
 * form (residual-curvature terms dropped) — both converge to the
 * bump-and-recalibrate answer as residuals vanish.
 *
 * Chain conventions
 * -----------------
 *   a: market quotes (nA)
 *   b: calibrated model parameters (nB)
 *   c: parameters derived directly from the market, `c = f(a)` (nC)
 *   I: calibration instruments (nI)
 *
 * Final propagation (`propagateMarketRisks`):
 *
 *     dV/da = (dV/db)(db/da) + (dV/dc)(dc/da)
 *
 * Consumers: SDE/reduced-model gradients (`mc/Gradients.h`) for `dV/db`,
 * and the callable chain (`callable_ad_design.md` §6) for portfolio values.
 */

/// Diagonal calibration weights and instrument Jacobians (all evaluated at
/// the calibration optimum).
struct CalibrationJacobians {
    std::vector<double> weights; ///< nI diagonal of Omega (all positive)
    Eigen::MatrixXd dIdb;        ///< nI x nB  model Jacobian (model vegas)
    Eigen::MatrixXd dIda;        ///< nI x nA  market Jacobian (market vegas)
    Eigen::MatrixXd dIdc;        ///< nI x nC  derived-parameter Jacobian (may be empty)
    Eigen::MatrixXd dcda;        ///< nC x nA  derived parameters, c = f(a) (may be empty)
};

/// Result of the instrument-Jacobian route, with conditioning diagnostics.
struct InstrumentIftResult {
    Eigen::MatrixXd dbda; ///< nB x nA
    std::size_t rank = 0;
    double singularMin = 0.0;
    double singularMax = 0.0;
    double condition = 0.0; ///< singularMax / singularMin (inf if rank-deficient)
};

struct InstrumentIftOptions {
    double svdCut = 1e-12; ///< singular values <= cut are dropped
};

/// Instrument-Jacobian route: weighted SVD pseudo-inverse (Savine).
///
///   db/da = (dI/db)^+ (dI/da - dI/dc dc/da),
///   (dI/db)^+ computed on sqrt(Omega) dI/db (thin SVD, cut at svdCut).
inline InstrumentIftResult instrumentCalibrationJacobian(const CalibrationJacobians& jac,
                                                         const InstrumentIftOptions& options = {}) {
    const Eigen::Index nI = jac.dIdb.rows();
    const Eigen::Index nB = jac.dIdb.cols();
    const Eigen::Index nA = jac.dIda.cols();
    if (static_cast<Eigen::Index>(jac.weights.size()) != nI) {
        throw std::invalid_argument("instrumentCalibrationJacobian: weights size != dI/db rows");
    }
    if (jac.dIda.rows() != nI) {
        throw std::invalid_argument("instrumentCalibrationJacobian: dI/da row mismatch");
    }
    for (double w : jac.weights) {
        if (!(w > 0.0)) {
            throw std::invalid_argument("instrumentCalibrationJacobian: weights must be positive");
        }
    }

    Eigen::MatrixXd rhs = jac.dIda;
    if (jac.dIdc.size() > 0) { // empty (0 x 0) = no derived parameters
        if (jac.dIdc.rows() != nI) {
            throw std::invalid_argument("instrumentCalibrationJacobian: dI/dc row mismatch");
        }
        if (jac.dcda.rows() != jac.dIdc.cols() || jac.dcda.cols() != nA) {
            throw std::invalid_argument("instrumentCalibrationJacobian: dc/da shape mismatch");
        }
        rhs -= jac.dIdc * jac.dcda;
    }

    Eigen::VectorXd sqrtW(nI);
    for (Eigen::Index i = 0; i < nI; ++i) {
        sqrtW(i) = std::sqrt(jac.weights[static_cast<std::size_t>(i)]);
    }
    const Eigen::MatrixXd jw = jac.dIdb.array().colwise() * sqrtW.array();
    const Eigen::MatrixXd rw = rhs.array().colwise() * sqrtW.array();

    Eigen::JacobiSVD<Eigen::MatrixXd> svd(jw, Eigen::ComputeThinU | Eigen::ComputeThinV);
    const Eigen::VectorXd& s = svd.singularValues();

    InstrumentIftResult result;
    result.rank = 0;
    Eigen::VectorXd invS(s.size());
    for (Eigen::Index i = 0; i < s.size(); ++i) {
        if (s(i) > options.svdCut) {
            invS(i) = 1.0 / s(i);
            ++result.rank;
        } else {
            invS(i) = 0.0;
        }
    }
    result.singularMin = s.size() > 0 ? s.minCoeff() : 0.0;
    result.singularMax = s.size() > 0 ? s.maxCoeff() : 0.0;
    result.condition = result.singularMin > 0.0 ? result.singularMax / result.singularMin
                                                : std::numeric_limits<double>::infinity();
    result.dbda = svd.matrixV() * invS.asDiagonal() * svd.matrixU().transpose() * rw;
    return result;
}

/// dV/da = (dV/db)(db/da) + (dV/dc)(dc/da) (either derived block may be empty).
inline Eigen::VectorXd propagateMarketRisks(const Eigen::VectorXd& dVdb,
                                            const Eigen::VectorXd& dVdc,
                                            const Eigen::MatrixXd& dbda,
                                            const Eigen::MatrixXd& dcda) {
    if (dVdb.size() != dbda.rows()) {
        throw std::invalid_argument("propagateMarketRisks: dV/db size != db/da rows");
    }
    Eigen::VectorXd out = dVdb.transpose() * dbda;
    if (dVdc.size() > 0) {
        if (dcda.rows() != dVdc.size() || dcda.cols() != dbda.cols()) {
            throw std::invalid_argument("propagateMarketRisks: dV/dc or dc/da shape mismatch");
        }
        out += dVdc.transpose() * dcda;
    }
    return out;
}

namespace detail {
inline bool iftCalibrationConverged(OptimizeResult result) {
    // RoundoffLimited is accepted: at realistic quote scales the solver
    // cannot improve the objective further in floating point and the
    // iterate is a valid LS optimum (the IFT is then exact to roundoff, as
    // the bump-recalibrate gates verify). Genuine failures (infeasible,
    // max eval/time, failure) still throw.
    return result == OptimizeResult::Success || result == OptimizeResult::GradientTolReached ||
           result == OptimizeResult::FtolReached || result == OptimizeResult::XtolReached ||
           result == OptimizeResult::RoundoffLimited;
}
} // namespace detail

/**
 * @brief KKT route: calibrate `b` to quotes `a`, then `db/da` by IFT
 *
 * `objective(x, m)` is the fitting objective (`m` = quotes `a`, `x` = model
 * parameters `b`), same callable contract as the optimizer/IFT layer.
 * Constraints and bounds are supported (the returned `info` reports
 * active sets and conditioning; final multipliers land in `state`).
 */
template <typename F2, typename G2 = NoConstraint, typename H2 = NoConstraint>
Eigen::MatrixXd
kktCalibrationJacobian(const F2& objective, const std::vector<double>& quotes,
                       std::vector<double>& b, OptimizerState& state, IftResult& info,
                       const G2& g = G2{}, const H2& h = H2{}, const Bounds& bounds = Bounds{},
                       const StopCriteria& criteria = {}, const IftOptions& options = {}) {
    std::vector<double> dpdm;
    const OptimizeResult result =
        minimizeDifferential(objective, g, h, bounds, quotes, b, state, info, &dpdm, nullptr,
                             nullptr, criteria, options);
    if (!detail::iftCalibrationConverged(result)) {
        throw std::runtime_error("kktCalibrationJacobian: calibration did not converge (code " +
                                 std::to_string(static_cast<int>(result)) + ")");
    }
    if (dpdm.size() != b.size() * quotes.size()) {
        throw std::runtime_error("kktCalibrationJacobian: IFT returned no Jacobian (code " +
                                 std::to_string(static_cast<int>(result)) + ")");
    }
    Eigen::MatrixXd out(static_cast<Eigen::Index>(b.size()),
                        static_cast<Eigen::Index>(quotes.size()));
    for (std::size_t i = 0; i < b.size(); ++i) {
        for (std::size_t j = 0; j < quotes.size(); ++j) {
            out(static_cast<Eigen::Index>(i), static_cast<Eigen::Index>(j)) =
                dpdm[i * quotes.size() + j];
        }
    }
    return out;
}

} // namespace quantape::math

#endif // QUANTAPE_MATH_OPTIMIZATION_CALIBRATION_CHAIN_H
