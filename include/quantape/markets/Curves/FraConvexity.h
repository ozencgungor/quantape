#pragma once

// Shifted-lognormal FRA convexity: the market FRA rate versus the curve FRA
// forward, as derived from the multiple-curve shifted-lognormal model.

#include <cmath>
#include <type_traits>

namespace quantape::markets {
/**
 * @file FraConvexity.h
 * @brief Shifted-lognormal FRA convexity adjustment
 *
 * The market FRA rate relates to the curve FRA forward by
 *
 *   `R = ((1 + F * accrual) * exp(C) - 1) / accrual`,
 *   `C = (sigmaIndex^2 - sigmaIndex * sigmaDiscount * correlation) * timeToFixing`,
 *
 * with `C` vanishing at the fixing date. The size is below a basis point for
 * typical volatilities; the adjustment is optional and applied per FRA pillar
 * in the bootstrap (and mirrored in the risk rows). Every argument may be an
 * AD scalar: the evaluation uses heterogeneous scalar promotion and ADL `exp`.
 */
template <typename SigmaIndexT, typename SigmaDiscountT, typename CorrelationT>
auto fraConvexityExponent(const SigmaIndexT& sigmaIndex, const SigmaDiscountT& sigmaDiscount,
                          const CorrelationT& correlation, double timeToFixing) {
    using DoubleT = std::decay_t<decltype(sigmaIndex * sigmaIndex * timeToFixing)>;
    const DoubleT sigma = sigmaIndex;
    return (sigma * sigma - sigma * sigmaDiscount * correlation) * timeToFixing;
}

/// Market FRA rate from the curve forward and the convexity exponent.
template <typename ForwardT>
auto impliedFraMarketRate(const ForwardT& forward, double convexityExponent, double accrual) {
    using std::exp;
    return ((1.0 + forward * accrual) * exp(convexityExponent) - 1.0) / accrual;
}

}  // namespace quantape::markets
