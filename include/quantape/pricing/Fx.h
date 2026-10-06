#pragma once

#include "quantape/datetime/Date.h"
#include "quantape/datetime/DayCounter.h"
#include "quantape/markets/Curves/DiscountCurve.h"
#include "quantape/markets/Data/FXRate.h"
#include "quantape/markets/Data/FxQuote.h"

#include <cstddef>
#include <stdexcept>
#include <vector>

namespace quantape::markets {
/**
 * @file Fx.h
 * @brief AD-templated FX spot, forward and swap pricing and spot greeks
 *
 * Conventions: the spot rate is quote-currency units per one base-currency
 * unit (EURUSD ~ 1.10 USD per EUR), base discounts are on the base curve and
 * quote discounts on the quote curve, and every value is expressed in the
 * quote currency. A forward struck at `K` on `N` base units is
 *
 *   `V = N (S D_base(T) - K D_quote(T))`,
 *
 * so covered interest parity `F = S D_base(T)/D_quote(T)` makes the forward
 * worth zero. A swap is the difference of a far and a near forward leg.
 *
 * Every entry point is templated on the scalar type and uses only arithmetic
 * and curve calls, with no value-dependent branching, so `double`, `var` and
 * `fvar<var>` tape one identical path. That makes the pricers their own AD
 * hook: instantiate with `var` for spot delta and with `fvar<var>` for spot
 * gamma. Spot delta and gamma of a discounted cashflow portfolio are also
 * available analytically from the base notional and discount factors, and
 * `fxGreeksFiniteDifference` is the fallback for evaluators that cannot be
 * instantiated with AD scalars.
 */

/// PV in the quote currency of `baseNotional` base-currency units at spot:
/// `V = N S`, affine in the spot.
template <typename ScalarT>
ScalarT fxSpotPv(const ScalarT& spot, const ScalarT& baseNotional) {
    return baseNotional * spot;
}

/// PV in the quote currency of one long base-currency forward of
/// `baseNotional` units struck at `strike`, paying at the discount times:
/// `V = N (S D_base - K D_quote)`.
template <typename ScalarT>
ScalarT fxForwardPv(const ScalarT& spot, const ScalarT& strike, const ScalarT& baseDiscount,
                    const ScalarT& quoteDiscount, const ScalarT& baseNotional) {
    return baseNotional * (spot * baseDiscount - strike * quoteDiscount);
}

/// Forward PV reading the discount factors from an `FXRate` at `maturity`.
/// The curve discounts are read directly so the same entry point serves every
/// AD scalar (the `FXRate` accessors take their time as the scalar type).
template <typename ScalarT>
ScalarT fxForwardPv(const FXRate<ScalarT>& rate, double maturity, const ScalarT& strike,
                    const ScalarT& baseNotional) {
    if (rate.baseCurve() == nullptr || rate.quoteCurve() == nullptr) {
        throw std::invalid_argument("fxForwardPv: FXRate has no discount curves");
    }
    return fxForwardPv(rate.spot(), strike, rate.baseCurve()->discount(maturity),
                       rate.quoteCurve()->discount(maturity), baseNotional);
}

/// PV in the quote currency of an FX swap: long `baseNotional` base units at
/// the near leg and short the same amount at the far leg, each leg priced as a
/// forward: `V = N [(S D_base_far - K_far D_quote_far) -
/// (S D_base_near - K_near D_quote_near)]`.
template <typename ScalarT>
ScalarT fxSwapPv(const ScalarT& spot, const ScalarT& nearStrike, const ScalarT& nearBaseDiscount,
                 const ScalarT& nearQuoteDiscount, const ScalarT& farStrike,
                 const ScalarT& farBaseDiscount, const ScalarT& farQuoteDiscount,
                 const ScalarT& baseNotional) {
    return baseNotional * ((spot * farBaseDiscount - farStrike * farQuoteDiscount) -
                           (spot * nearBaseDiscount - nearStrike * nearQuoteDiscount));
}

/// Swap PV from an `FXRate` and a quoted swap schedule: the near leg exchanges
/// at spot and the far leg at the quote's outright, with both legs discounted
/// on the rate's curves at the quote's dates.
template <typename ScalarT>
ScalarT fxSwapPv(const FXRate<ScalarT>& rate, const FxSwapQuote& quote,
                 const datetime::Date& referenceDate, const datetime::DayCounter& dayCounter,
                 const ScalarT& baseNotional) {
    if (rate.baseCurve() == nullptr || rate.quoteCurve() == nullptr) {
        throw std::invalid_argument("fxSwapPv: FXRate has no discount curves");
    }
    const double nearTime = dayCounter.yearFraction(referenceDate, quote.startDate());
    const double farTime = dayCounter.yearFraction(referenceDate, quote.maturityDate());
    const ScalarT farStrike = quote.outright();
    return fxSwapPv(rate.spot(), rate.spot(), rate.baseCurve()->discount(nearTime),
                    rate.quoteCurve()->discount(nearTime), farStrike,
                    rate.baseCurve()->discount(farTime), rate.quoteCurve()->discount(farTime),
                    baseNotional);
}

/// PV in the quote currency of discounted FX cashflows:
/// `V = sum_i N_i S D_base_i + sum_j Q_j D_quote_j`, where the base notionals
/// and quote cashflows are paired with their own discount factors.
template <typename ScalarT>
ScalarT fxCashflowPv(const ScalarT& spot, const std::vector<ScalarT>& baseNotionals,
                     const std::vector<ScalarT>& baseDiscounts,
                     const std::vector<ScalarT>& quoteCashflows,
                     const std::vector<ScalarT>& quoteDiscounts) {
    if (baseNotionals.size() != baseDiscounts.size() ||
        quoteCashflows.size() != quoteDiscounts.size()) {
        throw std::invalid_argument("fxCashflowPv: cashflow and discount sizes differ");
    }
    ScalarT value = 0.0;
    for (std::size_t i = 0; i < baseNotionals.size(); ++i) {
        value += baseNotionals[i] * spot * baseDiscounts[i];
    }
    for (std::size_t j = 0; j < quoteCashflows.size(); ++j) {
        value += quoteCashflows[j] * quoteDiscounts[j];
    }
    return value;
}

/// FX spot greeks of a scalar value: the first and second derivative with
/// respect to the spot rate.
template <typename ScalarT>
struct FxGreeks {
    ScalarT delta{};
    ScalarT gamma{};
};

/// Analytic spot greeks of discounted FX cashflows: the value is affine in
/// the spot, so the delta is `sum_i N_i D_base_i` and the gamma is exactly
/// zero. The explicit zero keeps gamma reporting uniform across product types.
template <typename ScalarT>
FxGreeks<ScalarT> fxCashflowGreeks(const std::vector<ScalarT>& baseNotionals,
                                   const std::vector<ScalarT>& baseDiscounts) {
    if (baseNotionals.size() != baseDiscounts.size()) {
        throw std::invalid_argument("fxCashflowGreeks: notional and discount sizes differ");
    }
    ScalarT delta = 0.0;
    for (std::size_t i = 0; i < baseNotionals.size(); ++i) {
        delta += baseNotionals[i] * baseDiscounts[i];
    }
    const ScalarT gamma = 0.0;
    return FxGreeks<ScalarT>{delta, gamma};
}

/// Fallback spot greeks by central differences of a scalar evaluator that
/// cannot be instantiated with AD scalars. The default step balances the
/// `O(step^2)` truncation and `O(eps/step)` round-off of the central formulas.
template <typename ValueFn>
FxGreeks<double> fxGreeksFiniteDifference(const ValueFn& value, double spot, double step = 1e-5) {
    if (!(step > 0.0)) {
        throw std::invalid_argument("fxGreeksFiniteDifference: step must be positive");
    }
    const double up = value(spot + step);
    const double down = value(spot - step);
    const double center = value(spot);
    return FxGreeks<double>{(up - down) / (2.0 * step), (up - 2.0 * center + down) / (step * step)};
}

} // namespace quantape::markets
