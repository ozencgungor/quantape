#ifndef QUANTAPE_CALIBRATION_LEVENBERG_MARQUARDT_IFT_H
#define QUANTAPE_CALIBRATION_LEVENBERG_MARQUARDT_IFT_H

#include "quantape/math/StanMath.h"

#include "quantape/calibration/ImplicitFunction.h"
#include "quantape/math/Optimization/LevenbergMarquardtStanPrimitives.h"

#include <cstddef>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

namespace quantape::math {
/**
 * @file LevenbergMarquardtIft.h
 * @brief Least-squares optimum + IFT sensitivities (LM twin of
 *        `minimizeDifferential`/`minimizeDifferentialVar`)
 *
 * Internal derivatives: the LM solve uses the exact Stan-AD Jacobian
 * (`levenbergMarquardtAd`), or the caller may supply one. External
 * derivatives: the converged optimum's Jacobian w.r.t. the market vector is
 * computed by `iftUnconstrained` (exact HVPs + forward-over-reverse mixed
 * Hessians) and, in the `Var` variant, attached to the caller's tape through
 * one `make_callback_var` per parameter. The solver itself is never
 * differentiated through.
 *
 * Residual callable contract: `Sx` (parameter scalar) and `Sm` (market
 * scalar) are deduced independently, so the residual must be template-generic
 * in both:
 *
 *   `void r(const std::vector<Sx>& x, const std::vector<Sm>& m,
 *           std::vector<Sx>& out)`
 *
 * The LM value/Jacobian passes invoke it with `(double, double)` and
 * `(var, double)`; the IFT Hessian passes with `(fvar<var>, double)` (each
 * HVP column) and `(fvar<var>, fvar<var>)` (mixed Hessian). A model that
 * demands one shared scalar type for both arguments does not satisfy this
 * contract. Fixed market data is always passed as `double` during the solve.
 *
 * The returned optimum must be stationary (`LevenbergMarquardtResult::
 * stationary`); otherwise the IFT derivatives would be silently biased and
 * the entry points throw `std::runtime_error`.
 */

namespace detail {

template <typename Residual>
auto leastSquaresCost(const Residual& residual, const auto& x, const auto& m) {
    using S = std::decay_t<decltype(x[0] * m[0] * 0.5 + m[0] * 0.0)>;
    std::vector<S> residuals;
    residual(x, m, residuals);
    S cost = 0;
    for (const S& value : residuals) {
        cost += 0.5 * value * value;
    }
    return cost;
}

template <typename Residual>
LevenbergMarquardtResult
solveLeastSquares(const Residual& residual, const std::vector<double>& market,
                  std::vector<double>& x, const LevenbergMarquardtOptions& options) {
    const auto wrapped = [&](const auto& point, auto& out) { residual(point, market, out); };
    return levenbergMarquardtAd(wrapped, x, options);
}

/// Statuses that leave `x` at an acceptable optimum for the IFT layer.
inline bool iftAcceptableStatus(OptimizeResult status) {
    return status == OptimizeResult::Success || status == OptimizeResult::GradientTolReached ||
           status == OptimizeResult::FtolReached || status == OptimizeResult::XtolReached ||
           status == OptimizeResult::RoundoffLimited;
}

/// IFT derivatives are exact only at a stationary point; refuse to build
/// biased derivative graphs at a non-stationary stop.
inline void requireStationary(const LevenbergMarquardtResult& lm) {
    if (lm.stationary) {
        return;
    }
    throw std::runtime_error(
        "levenbergMarquardtDifferential: solver stopped at a non-stationary point "
        "(||J^T r|| = " +
        std::to_string(lm.gradientNorm) +
        "); IFT derivatives would be biased. Increase maxIterations or rescale the problem.");
}

} // namespace detail

/// Solve + exact IFT Jacobian `dp/dm` (row-major `n x M`).
///
/// The residual follows the separately-generic contract documented at the top
/// of this file (`std::vector<Sx>` parameters, `std::vector<Sm>` market).
///
/// @throws std::runtime_error when the LM optimum is not stationary (the IFT
///         Jacobian would be silently biased)
/// @param[out] lmOut optional solver diagnostics (`status`, `cost`,
///             `gradient`, `gradientNorm`, `stationary`, `iterations`)
template <typename Residual>
OptimizeResult levenbergMarquardtDifferential(
    const Residual& residual, const std::vector<double>& x0, const std::vector<double>& market,
    std::vector<double>& pHat, IftResult& ift, std::vector<double>* dpDm = nullptr,
    const LevenbergMarquardtOptions& lmOptions = {}, const IftOptions& iftOptions = {},
    LevenbergMarquardtResult* lmOut = nullptr) {
    pHat = x0;
    const LevenbergMarquardtResult lm =
        detail::solveLeastSquares(residual, market, pHat, lmOptions);
    if (lmOut != nullptr) {
        *lmOut = lm;
    }
    if (!detail::iftAcceptableStatus(lm.status)) {
        return lm.status;
    }
    detail::requireStationary(lm);
    std::vector<double> jacobian;
    const auto cost = [&](const auto& x, const auto& m) {
        return detail::leastSquaresCost(residual, x, m);
    };
    iftUnconstrained(cost, pHat, market, jacobian, ift, iftOptions);
    if (dpDm) {
        *dpDm = std::move(jacobian);
    }
    return lm.status;
}

/// Solve + IFT with the optimum attached to the caller's tape: each `p_hat[k]`
/// is a callback var whose adjoint flows into the `m_var` leaves through
/// `dp/dm` (same pattern as `minimizeDifferentialVar`).
///
/// The residual follows the separately-generic contract documented at the top
/// of this file (`std::vector<Sx>` parameters, `std::vector<Sm>` market).
///
/// If the IFT had to regularize the Hessian (`IftResult::regularized`),
/// `dp/dm` is a documented distortion and the derivative graph is refused
/// unless @p allowRidge is set: an arena-only callback destroys the
/// diagnostics, so silently encoding them would be unsafe.
///
/// @throws std::runtime_error when the LM optimum is not stationary, or when
///         the IFT used a ridge without `allowRidge`
/// @param[out] iftOut   optional IFT diagnostics
/// @param[out] stateOut optional final `OptimizerState`, filled exactly like
///              the double path (`LevenbergMarquardt::fillState`)
/// @param[in]  allowRidge accept a ridge-regularized IFT (default: throw)
template <typename Residual>
OptimizeResult levenbergMarquardtDifferentialVar(
    const Residual& residual, const std::vector<stan::math::var>& mVar,
    const std::vector<double>& x0, std::vector<stan::math::var>& pHat, IftResult* iftOut = nullptr,
    OptimizerState* stateOut = nullptr, const LevenbergMarquardtOptions& lmOptions = {},
    const IftOptions& iftOptions = {}, bool allowRidge = false) {
    const std::size_t marketSize = mVar.size();
    std::vector<double> market(marketSize);
    for (std::size_t j = 0; j < marketSize; ++j) {
        market[j] = mVar[j].val();
    }
    std::vector<double> x = x0;
    IftResult ift;
    std::vector<double> dpDm;
    LevenbergMarquardtResult lm;
    const OptimizeResult result = levenbergMarquardtDifferential(residual, x0, market, x, ift,
                                                                 &dpDm, lmOptions, iftOptions, &lm);
    if (!detail::iftAcceptableStatus(result)) {
        return result;
    }
    if (ift.regularized && !allowRidge) {
        throw std::runtime_error(
            "levenbergMarquardtDifferentialVar: IFT used ridge regularization (ridge = " +
            std::to_string(ift.ridgeUsed) +
            "); dp/dm is biased. Pass allowRidge = true to accept it explicitly.");
    }
    const std::size_t n = x.size();
    pHat.resize(n);
    // Arena-allocated captures: callback_vari destructors never run, so
    // std::vector captures would leak their heap blocks.
    stan::math::vari** marketVaris =
        stan::math::ChainableStack::instance_->memalloc_.alloc_array<stan::math::vari*>(marketSize);
    for (std::size_t j = 0; j < marketSize; ++j) {
        marketVaris[j] = mVar[j].vi_;
    }
    for (std::size_t k = 0; k < n; ++k) {
        double* row =
            stan::math::ChainableStack::instance_->memalloc_.alloc_array<double>(marketSize);
        for (std::size_t j = 0; j < marketSize; ++j) {
            row[j] = dpDm[k * marketSize + j];
        }
        pHat[k] = stan::math::make_callback_var(x[k], [marketVaris, row, marketSize](auto& vi) {
            const double adjoint = vi.adj();
            for (std::size_t j = 0; j < marketSize; ++j) {
                marketVaris[j]->adj_ += adjoint * row[j];
            }
        });
    }
    if (iftOut) {
        *iftOut = ift;
    }
    if (stateOut) {
        LevenbergMarquardt::fillState(lm, x, *stateOut);
    }
    return result;
}

} // namespace quantape::math

#endif // QUANTAPE_CALIBRATION_LEVENBERG_MARQUARDT_IFT_H
