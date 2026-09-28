#ifndef QUANTAPE_MATH_OPTIMIZATION_CALIBRATION_CHAIN_KKT_H
#define QUANTAPE_MATH_OPTIMIZATION_CALIBRATION_CHAIN_KKT_H

//
// CalibrationChainKkt.h -- AD-based KKT calibration route (Stan-dependent)
//
// Solve + IFT in one call for scalar-generic objectives: the optimizers and
// the implicit-function layer run on the AD tape (LBFGS<var>/SLSQP<var> for
// the solve, exact HVPs/mixed Hessians for dp/dm). Kept separate from
// CalibrationChain.h so the generic chain (instrument route, risk
// propagation, assembled-problem IFT) stays Stan-free.
//

#include "quantape/calibration/CalibrationChain.h"
#include "quantape/calibration/ImplicitFunction.h"

#include <string>
#include <vector>

namespace quantape::math {

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

#endif // QUANTAPE_MATH_OPTIMIZATION_CALIBRATION_CHAIN_KKT_H
