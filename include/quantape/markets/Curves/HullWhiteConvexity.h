#pragma once

// Hull-White one-factor futures convexity: the futures rate over an accrual
// period equals the FRA forward plus the convexity adjustment computed here.

#include "quantape/math/Optimization/LevenbergMarquardt.h"

#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <vector>

namespace quantape::markets {
/**
 * @file HullWhiteConvexity.h
 * @brief Hull-White futures/FRA convexity adjustment
 *
 * Under the one-factor Hull-White model with short-rate volatility `sigma` and
 * mean reversion `meanReversion`, the futures rate on the accrual period
 * `[T1, T2]` equals the FRA forward plus
 *
 *   `C = (P(0,T1) / P(0,T2)) * (exp(D) - 1) / accrual`,
 *   `D = sigma^2/(2 a^2) (1 - e^{-a T1})^2 b
 *        + sigma^2/(2 a) (1 - e^{-2 a T1}) b^2`,  `b = (1 - e^{-a accrual}) / a`,
 *
 * the exact expectation of the bond reciprocal `E[e^{int r}]` under the model;
 * the `a -> 0` limit is evaluated with its series. The caller supplies the
 * bond ratio `P(0,T1)/P(0,T2)` from the curve being built, so a futures pillar
 * stores only the resulting constant adjustment. The volatility, mean reversion
 * and bond ratio may be AD scalars: the evaluation uses ADL `exp`/`expm1` and
 * primal-safe branching. The expiry and accrual times are plain doubles.
 */
template <typename SigmaT, typename MeanReversionT, typename RatioT>
auto hullWhiteFuturesAdjustment(const SigmaT& sigma, const MeanReversionT& meanReversion,
                                double expiryTime, double accrual,
                                const RatioT& startOverEndDiscount) {
    if (!(sigma >= 0.0)) {
        throw std::invalid_argument("hullWhiteFuturesAdjustment: negative volatility");
    }
    if (!(meanReversion >= 0.0)) {
        throw std::invalid_argument("hullWhiteFuturesAdjustment: negative mean reversion");
    }
    if (!(accrual > 0.0)) {
        throw std::invalid_argument("hullWhiteFuturesAdjustment: non-positive accrual");
    }
    if (!(expiryTime >= 0.0)) {
        throw std::invalid_argument("hullWhiteFuturesAdjustment: negative expiry time");
    }
    using std::expm1;
    using DoubleT = std::decay_t<decltype(startOverEndDiscount / accrual * sigma * sigma)>;
    const DoubleT scale = startOverEndDiscount / accrual;
    if (!(sigma > 0.0)) {
        return scale * DoubleT(0.0);
    }
    DoubleT exponent;
    if (meanReversion < 1e-8) {
        // Limit of the exact branch as the mean reversion vanishes.
        exponent = sigma * sigma * accrual *
                   (0.5 * expiryTime * expiryTime + expiryTime * accrual);
    } else {
        const auto a = meanReversion;
        const DoubleT b = -expm1(-a * accrual) / a;
        const DoubleT oneMinusExpiry = -expm1(-a * expiryTime);
        const DoubleT oneMinusDoubleExpiry = -expm1(-2.0 * a * expiryTime);
        exponent = sigma * sigma / (2.0 * a * a) * oneMinusExpiry * oneMinusExpiry * b +
                   sigma * sigma / (2.0 * a) * oneMinusDoubleExpiry * b * b;
    }
    return scale * expm1(exponent);
}

/// One observed futures/reference-forward pair: `impliedAdjustment` is
/// `R_fut - FRA_reference` over the accrual period `[T1, T1 + accrual]`.
struct HullWhiteObservation {
    double expiryTime = 0.0;
    double accrual = 0.0;
    double startOverEndDiscount = 1.0;
    double impliedAdjustment = 0.0;
};

struct HullWhiteFitOptions {
    double pinnedMeanReversion = 0.05; ///< Fit of sigma: mean reversion held fixed
    double pinnedSigma = 0.01;         ///< Fit of mean reversion: sigma held fixed
    double initialValue = 0.0;         ///< 0 selects a small default start
    double tolerance = 1e-14;          ///< Absolute residual/step tolerance
    int maxIterations = 200;
};

struct HullWhiteFitResult {
    double sigma = 0.0;
    double meanReversion = 0.0;
    double rmsResidual = 0.0;
    double maxResidual = 0.0;
    int iterations = 0;
    bool converged = false;
};

namespace detail {

inline void hullWhiteResiduals(const std::vector<HullWhiteObservation>& observations,
                               double sigma, double meanReversion,
                               std::vector<double>& out) {
    out.resize(observations.size());
    for (std::size_t i = 0; i < observations.size(); ++i) {
        const HullWhiteObservation& observation = observations[i];
        out[i] = hullWhiteFuturesAdjustment(sigma, meanReversion, observation.expiryTime,
                                            observation.accrual,
                                            observation.startOverEndDiscount) -
                 observation.impliedAdjustment;
    }
}

inline HullWhiteFitResult summarize(const std::vector<HullWhiteObservation>& observations,
                                    double sigma, double meanReversion,
                                    const math::LevenbergMarquardtResult& lm) {
    HullWhiteFitResult result;
    result.sigma = sigma;
    result.meanReversion = meanReversion;
    result.iterations = lm.iterations;
    result.converged = lm.stationary;
    std::vector<double> residuals;
    hullWhiteResiduals(observations, sigma, meanReversion, residuals);
    double sumSquares = 0.0;
    for (const double residual : residuals) {
        sumSquares += residual * residual;
        result.maxResidual = std::max(result.maxResidual, std::abs(residual));
    }
    result.rmsResidual = std::sqrt(sumSquares / static_cast<double>(residuals.size()));
    return result;
}

}  // namespace detail

/// Fit the short-rate volatility with the mean reversion pinned. A joint fit of
/// both parameters is not identified from a futures strip alone (the two
/// profiles are nearly collinear), so one parameter is always an input.
inline HullWhiteFitResult fitHullWhiteSigma(
    const std::vector<HullWhiteObservation>& observations,
    const HullWhiteFitOptions& options = {}) {
    if (observations.empty()) {
        throw std::invalid_argument("fitHullWhiteSigma: no observations");
    }
    if (!std::isfinite(options.pinnedMeanReversion) || options.pinnedMeanReversion < 0.0) {
        throw std::invalid_argument(
            "fitHullWhiteSigma: pinned mean reversion must be finite and non-negative");
    }
    if (!std::isfinite(options.initialValue) || options.initialValue < 0.0) {
        throw std::invalid_argument(
            "fitHullWhiteSigma: initial value must be finite and non-negative");
    }
    math::LevenbergMarquardtOptions lmOptions;
    lmOptions.maxIterations = options.maxIterations;
    lmOptions.gradientTol = 0.0; // the profiles are flat: iterate on steps/cost
    lmOptions.xtol = 0.0;
    lmOptions.ftol = 0.0;
    lmOptions.xtolAbs = options.tolerance;
    lmOptions.ftolAbs = options.tolerance;
    std::vector<double> x{
        std::log(options.initialValue > 0.0 ? options.initialValue : 0.01)};
    const auto residuals = [&](const std::vector<double>& point, std::vector<double>& out) {
        detail::hullWhiteResiduals(observations, std::exp(point[0]), options.pinnedMeanReversion,
                                   out);
    };
    const math::LevenbergMarquardtResult lm = math::levenbergMarquardt(residuals, x, lmOptions);
    return detail::summarize(observations, std::exp(x[0]), options.pinnedMeanReversion, lm);
}

/// Fit the mean reversion with the volatility pinned.
inline HullWhiteFitResult fitHullWhiteMeanReversion(
    const std::vector<HullWhiteObservation>& observations,
    const HullWhiteFitOptions& options = {}) {
    if (observations.empty()) {
        throw std::invalid_argument("fitHullWhiteMeanReversion: no observations");
    }
    if (!std::isfinite(options.pinnedSigma) || options.pinnedSigma < 0.0) {
        throw std::invalid_argument(
            "fitHullWhiteMeanReversion: pinned sigma must be finite and non-negative");
    }
    if (!std::isfinite(options.initialValue) || options.initialValue < 0.0) {
        throw std::invalid_argument(
            "fitHullWhiteMeanReversion: initial value must be finite and non-negative");
    }
    math::LevenbergMarquardtOptions lmOptions;
    lmOptions.maxIterations = options.maxIterations;
    lmOptions.gradientTol = 0.0; // the profiles are flat: iterate on steps/cost
    lmOptions.xtol = 0.0;
    lmOptions.ftol = 0.0;
    lmOptions.xtolAbs = options.tolerance;
    lmOptions.ftolAbs = options.tolerance;
    std::vector<double> x{
        std::log(options.initialValue > 0.0 ? options.initialValue : 0.05)};
    const auto residuals = [&](const std::vector<double>& point, std::vector<double>& out) {
        detail::hullWhiteResiduals(observations, options.pinnedSigma, std::exp(point[0]), out);
    };
    const math::LevenbergMarquardtResult lm = math::levenbergMarquardt(residuals, x, lmOptions);
    return detail::summarize(observations, options.pinnedSigma, std::exp(x[0]), lm);
}

}  // namespace quantape::markets
