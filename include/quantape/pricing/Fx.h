#pragma once

#include "quantape/datetime/Date.h"
#include "quantape/datetime/DayCounter.h"
#include "quantape/instruments/BootstrapInstrument.h"
#include "quantape/instruments/FxInstruments.h"
#include "quantape/markets/Data/FXRate.h"
#include "quantape/markets/Data/FxQuote.h"

#include <cstddef>
#include <stdexcept>
#include <type_traits>
#include <utility>
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

namespace detail {

/// Scalar carried by a base/quote discount provider pair.
template <typename BaseT, typename QuoteT>
using FxDiscountScalar = std::decay_t<decltype(std::declval<const BaseT&>().discount(0.0))>;

/// Common scalar of a base/quote provider pair and an explicit value.
template <typename BaseT, typename QuoteT, typename ValueT>
using FxScalar = std::common_type_t<FxDiscountScalar<BaseT, QuoteT>, std::decay_t<ValueT>>;

} // namespace detail

/// Covered-interest-parity outright of a payout at `maturityDate`: the base
/// and quote discount providers are read at the payout date and combined as
/// `F = S D_base / D_quote`. This is the AD-generic core behind the instrument
/// quote helpers.
template <typename ScalarT, typename BaseT, typename QuoteT>
ScalarT cipOutright(const ScalarT& spot, const datetime::Date& maturityDate,
                    const datetime::Date& referenceDate, const datetime::DayCounter& zeroDayCounter,
                    const FxSet<BaseT, QuoteT>& curves) {
    const double t = zeroDayCounter.yearFraction(referenceDate, maturityDate);
    return spot * curves.base(t) / curves.quote(t);
}

/// CIP outright of a dated forward, from an explicit spot.
template <typename ScalarT, typename BaseT, typename QuoteT>
ScalarT impliedFxOutright(const instruments::FxForward& forward, const FxSet<BaseT, QuoteT>& curves,
                          const ScalarT& spot, const datetime::Date& referenceDate,
                          const datetime::DayCounter& zeroDayCounter) {
    return cipOutright(spot, forward.valueDate, referenceDate, zeroDayCounter, curves);
}

/// CIP outright of an FX swap's far leg, from an explicit spot.
template <typename ScalarT, typename BaseT, typename QuoteT>
ScalarT impliedFxOutright(const instruments::FxSwap& swap, const FxSet<BaseT, QuoteT>& curves,
                          const ScalarT& spot, const datetime::Date& referenceDate,
                          const datetime::DayCounter& zeroDayCounter) {
    return cipOutright(spot, swap.maturity, referenceDate, zeroDayCounter, curves);
}

/// CIP outright of a dated forward, at the instrument's own spot.
template <typename BaseT, typename QuoteT>
auto impliedFxOutright(const instruments::FxForward& forward, const FxSet<BaseT, QuoteT>& curves,
                       const datetime::Date& referenceDate,
                       const datetime::DayCounter& zeroDayCounter) {
    using ScalarT = detail::FxDiscountScalar<BaseT, QuoteT>;
    return impliedFxOutright(forward, curves, ScalarT(forward.spot), referenceDate, zeroDayCounter);
}

/// CIP outright of an FX swap's far leg, at the instrument's own spot.
template <typename BaseT, typename QuoteT>
auto impliedFxOutright(const instruments::FxSwap& swap, const FxSet<BaseT, QuoteT>& curves,
                       const datetime::Date& referenceDate,
                       const datetime::DayCounter& zeroDayCounter) {
    using ScalarT = detail::FxDiscountScalar<BaseT, QuoteT>;
    return impliedFxOutright(swap, curves, ScalarT(swap.spot), referenceDate, zeroDayCounter);
}

/// CIP forward points of a dated forward: the model outright minus spot.
template <typename ScalarT, typename BaseT, typename QuoteT>
ScalarT impliedFxForwardPoints(const instruments::FxForward& forward,
                               const FxSet<BaseT, QuoteT>& curves, const ScalarT& spot,
                               const datetime::Date& referenceDate,
                               const datetime::DayCounter& zeroDayCounter) {
    return impliedFxOutright(forward, curves, spot, referenceDate, zeroDayCounter) - spot;
}

/// CIP forward points of an FX swap's far leg in point units: the model
/// outright minus spot, divided by the price value of one point.
template <typename ScalarT, typename BaseT, typename QuoteT>
ScalarT impliedFxForwardPoints(const instruments::FxSwap& swap, const FxSet<BaseT, QuoteT>& curves,
                               const ScalarT& spot, const datetime::Date& referenceDate,
                               const datetime::DayCounter& zeroDayCounter) {
    return (impliedFxOutright(swap, curves, spot, referenceDate, zeroDayCounter) - spot) /
           ScalarT(swap.pointsScale);
}

/// CIP forward points of a dated forward, at the instrument's own spot.
template <typename BaseT, typename QuoteT>
auto impliedFxForwardPoints(const instruments::FxForward& forward,
                            const FxSet<BaseT, QuoteT>& curves, const datetime::Date& referenceDate,
                            const datetime::DayCounter& zeroDayCounter) {
    using ScalarT = detail::FxDiscountScalar<BaseT, QuoteT>;
    return impliedFxForwardPoints(forward, curves, ScalarT(forward.spot), referenceDate,
                                  zeroDayCounter);
}

/// CIP forward points of an FX swap's far leg, at the instrument's own spot.
template <typename BaseT, typename QuoteT>
auto impliedFxForwardPoints(const instruments::FxSwap& swap, const FxSet<BaseT, QuoteT>& curves,
                            const datetime::Date& referenceDate,
                            const datetime::DayCounter& zeroDayCounter) {
    using ScalarT = detail::FxDiscountScalar<BaseT, QuoteT>;
    return impliedFxForwardPoints(swap, curves, ScalarT(swap.spot), referenceDate, zeroDayCounter);
}

/// PV of a dated spot trade: the base notional marked at the market spot and
/// discounted on the base curve at the value date. Reads the stored value date
/// or rolls it from `referenceDate` on the instrument's calendars.
template <typename BaseT, typename QuoteT, typename ValueT>
auto fxSpotPv(const instruments::FxSpot& trade, const FxSet<BaseT, QuoteT>& curves,
              const datetime::Date& referenceDate, const datetime::DayCounter& zeroDayCounter,
              const ValueT& marketSpot, const ValueT& baseNotional) {
    using ScalarT = detail::FxScalar<BaseT, QuoteT, ValueT>;
    const datetime::Date payDate =
        trade.valueDate.serial() != 0
            ? trade.valueDate
            : trade.resolveValueDate(referenceDate, trade.baseCalendar, trade.quoteCalendar);
    const double t = zeroDayCounter.yearFraction(referenceDate, payDate);
    return fxSpotPv(ScalarT(marketSpot), ScalarT(baseNotional)) * ScalarT(curves.base(t));
}

/// PV of a dated forward: the formula-level forward value with base and quote
/// discounts read at the forward's value date.
template <typename BaseT, typename QuoteT, typename ValueT>
auto fxForwardPv(const instruments::FxForward& forward, const FxSet<BaseT, QuoteT>& curves,
                 const datetime::Date& referenceDate, const datetime::DayCounter& zeroDayCounter,
                 const ValueT& marketSpot, const ValueT& baseNotional) {
    using ScalarT = detail::FxScalar<BaseT, QuoteT, ValueT>;
    const datetime::Date payDate =
        forward.valueDate.serial() != 0
            ? forward.valueDate
            : forward.resolveValueDate(referenceDate, forward.baseCalendar, forward.quoteCalendar);
    const double t = zeroDayCounter.yearFraction(referenceDate, payDate);
    return fxForwardPv(ScalarT(marketSpot), ScalarT(forward.strike), ScalarT(curves.base(t)),
                       ScalarT(curves.quote(t)), ScalarT(baseNotional));
}

/// PV of a dated FX swap: the formula-level swap value with both legs
/// discounted at their actual settlement dates.
template <typename BaseT, typename QuoteT, typename ValueT>
auto fxSwapPv(const instruments::FxSwap& swap, const FxSet<BaseT, QuoteT>& curves,
              const datetime::Date& referenceDate, const datetime::DayCounter& zeroDayCounter,
              const ValueT& marketSpot, const ValueT& baseNotional) {
    using ScalarT = detail::FxScalar<BaseT, QuoteT, ValueT>;
    const datetime::Date nearDate =
        swap.start.serial() != 0
            ? swap.start
            : swap.resolveStart(referenceDate, swap.baseCalendar, swap.quoteCalendar);
    const datetime::Date farDate =
        swap.maturity.serial() != 0
            ? swap.maturity
            : swap.resolveMaturity(referenceDate, swap.baseCalendar, swap.quoteCalendar);
    const double nearTime = zeroDayCounter.yearFraction(referenceDate, nearDate);
    const double farTime = zeroDayCounter.yearFraction(referenceDate, farDate);
    return fxSwapPv(ScalarT(marketSpot), ScalarT(swap.spot), ScalarT(curves.base(nearTime)),
                    ScalarT(curves.quote(nearTime)), ScalarT(swap.quotedOutright()),
                    ScalarT(curves.base(farTime)), ScalarT(curves.quote(farTime)),
                    ScalarT(baseNotional));
}

/// Settlement cashflows of a spot trade with scalar-templated amounts.
template <typename ScalarT>
std::vector<instruments::CashflowT<ScalarT>>
fxCashflows(const instruments::FxSpot& trade, const ScalarT& baseNotional,
            const datetime::Date& referenceDate, const datetime::Calendar& baseCalendar,
            const datetime::Calendar& quoteCalendar) {
    return trade.template cashflowsT<ScalarT>(referenceDate, baseCalendar, quoteCalendar,
                                              baseNotional);
}

/// Settlement cashflows of a forward with scalar-templated amounts.
template <typename ScalarT>
std::vector<instruments::CashflowT<ScalarT>>
fxCashflows(const instruments::FxForward& forward, const ScalarT& baseNotional,
            const datetime::Date& referenceDate, const datetime::Calendar& baseCalendar,
            const datetime::Calendar& quoteCalendar) {
    return forward.template cashflowsT<ScalarT>(referenceDate, baseCalendar, quoteCalendar,
                                                baseNotional);
}

/// Settlement cashflows of an FX swap with scalar-templated amounts.
template <typename ScalarT>
std::vector<instruments::CashflowT<ScalarT>>
fxCashflows(const instruments::FxSwap& swap, const ScalarT& baseNotional,
            const datetime::Date& referenceDate, const datetime::Calendar& baseCalendar,
            const datetime::Calendar& quoteCalendar) {
    return swap.template cashflowsT<ScalarT>(referenceDate, baseCalendar, quoteCalendar,
                                             baseNotional);
}

/// PV of materialized settlement cashflows over base/quote discount providers:
/// base-currency flows are marked at spot and discounted on the base curve,
/// quote-currency flows are discounted on the quote curve. The pair identifies
/// which cashflow currency is the base one.
template <typename ScalarT, typename BaseT, typename QuoteT>
ScalarT fxCashflowPv(const FXDescriptor& pair,
                     const std::vector<instruments::CashflowT<ScalarT>>& cashflows,
                     const ScalarT& spot, const datetime::Date& referenceDate,
                     const datetime::DayCounter& zeroDayCounter,
                     const FxSet<BaseT, QuoteT>& curves) {
    const instruments::Currency baseCurrency = instruments::currencyFromCode(pair.baseCcy);
    ScalarT value = 0.0;
    for (const instruments::CashflowT<ScalarT>& cashflow : cashflows) {
        const double t = zeroDayCounter.yearFraction(referenceDate, cashflow.payDate);
        if (cashflow.currency.code == baseCurrency.code) {
            value += cashflow.amount * spot * ScalarT(curves.base(t));
        } else {
            value += cashflow.amount * ScalarT(curves.quote(t));
        }
    }
    return value;
}

} // namespace quantape::markets
