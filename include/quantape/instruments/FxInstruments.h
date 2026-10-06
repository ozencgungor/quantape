#pragma once

#include "quantape/datetime/BusinessDayConvention.h"
#include "quantape/datetime/Calendar.h"
#include "quantape/datetime/Date.h"
#include "quantape/datetime/DayCounter.h"
#include "quantape/datetime/Period.h"
#include "quantape/datetime/TimeConversion.h"
#include "quantape/instruments/Cashflow.h"
#include "quantape/markets/Data/FxQuote.h"
#include "quantape/markets/Descriptors/FXDescriptor.h"

#include <stdexcept>
#include <vector>

namespace quantape::instruments {
/**
 * @file FxInstruments.h
 * @brief Dated spot, forward and swap instruments with settlement lags
 *
 * Each instrument is plain data: the base/quote pair, the trade and settlement
 * dates, the signed base notional and the settlement conventions (per-currency
 * spot lags, joint calendar rule, business-day convention). A positive base
 * notional is long base at the horizon leg (the spot, forward value or swap
 * far date); a negative one is short it.
 *
 * Dates are resolved on the joint base/quote calendar: the spot date is the
 * trade date advanced by the larger of the two currency lags, and a forward or
 * swap far date advances the spot date by the tenor. Factories resolve and
 * store the dates; cashflow materialization can also roll them from a supplied
 * reference date when the instrument was left undated.
 */

/// Joint spot date of a pair: the reference date advanced by
/// `max(baseSpotLag, quoteSpotLag)` days and adjusted on the joint calendar.
inline datetime::Date fxSpotDate(const datetime::Date& referenceDate,
                                 const datetime::Calendar& baseCalendar,
                                 const datetime::Calendar& quoteCalendar, int baseSpotLag,
                                 int quoteSpotLag,
                                 datetime::BusinessDayConvention convention =
                                     datetime::BusinessDayConvention::ModifiedFollowing) {
    return datetime::spotDate(referenceDate, baseCalendar, quoteCalendar, baseSpotLag, quoteSpotLag,
                              convention);
}

/// Forward date of a pair: the joint spot date advanced by the tenor and
/// adjusted on the joint calendar. A zero-length tenor keeps the spot date.
inline datetime::Date fxForwardDate(const datetime::Date& referenceDate,
                                    const datetime::Calendar& baseCalendar,
                                    const datetime::Calendar& quoteCalendar, int baseSpotLag,
                                    int quoteSpotLag, const datetime::Period& tenor,
                                    datetime::BusinessDayConvention convention =
                                        datetime::BusinessDayConvention::ModifiedFollowing) {
    const datetime::Date spot = fxSpotDate(referenceDate, baseCalendar, quoteCalendar, baseSpotLag,
                                           quoteSpotLag, convention);
    if (tenor.length() == 0) {
        return spot;
    }
    return datetime::Calendar::joint(baseCalendar, quoteCalendar).advance(spot, tenor, convention);
}

/// Spot trade: exchanges `baseNotional` base units against quote units at the
/// value date. Both legs settle on the same date, so the trade carries the
/// trade date, the value date and the pair's settlement conventions.
struct FxSpot {
    markets::FXDescriptor pair;       ///< Base/quote currency pair
    datetime::Date tradeDate;         ///< Trade date
    datetime::Date valueDate;         ///< Settlement (value) date
    double spot = 0.0;                ///< Quote units per base unit
    double baseNotional = 0.0;        ///< Signed base amount, positive buys base
    datetime::Calendar baseCalendar;  ///< Base-currency settlement calendar
    datetime::Calendar quoteCalendar; ///< Quote-currency settlement calendar
    int baseSpotLag = 2;              ///< Base-currency settlement lag in days
    int quoteSpotLag = 2;             ///< Quote-currency settlement lag in days
    datetime::BusinessDayConvention convention =
        datetime::BusinessDayConvention::ModifiedFollowing; ///< Settlement roll rule

    /// Risk date: the settlement date.
    datetime::Date date() const { return valueDate; }

    /// Quoted target: the spot rate.
    double target() const { return spot; }

    /// Near settlement date.
    datetime::Date nearDate() const { return valueDate; }

    /// Far settlement date; identical to the near date for a spot trade.
    datetime::Date farDate() const { return valueDate; }

    /// Model quote over a base/quote curve set: the spot rate itself.
    template <typename ScalarT, typename CurveSetT>
    ScalarT impliedQuote(const CurveSetT&) const {
        return ScalarT(spot);
    }

    /// Value date: the stored date when set, otherwise rolled from
    /// `referenceDate` and the supplied calendars.
    datetime::Date resolveValueDate(const datetime::Date& referenceDate,
                                    const datetime::Calendar& baseCalendar,
                                    const datetime::Calendar& quoteCalendar) const {
        if (valueDate.serial() != 0) {
            return valueDate;
        }
        return fxSpotDate(referenceDate, baseCalendar, quoteCalendar, baseSpotLag, quoteSpotLag,
                          convention);
    }

    /// Settlement cashflows with scalar-templated amounts: the base leg and the
    /// quote counter-leg at the traded spot.
    template <typename ScalarT>
    std::vector<CashflowT<ScalarT>>
    cashflowsT(const datetime::Date& referenceDate, const datetime::Calendar& baseCalendar,
               const datetime::Calendar& quoteCalendar, const ScalarT& baseNotional) const {
        const datetime::Date payDate = resolveValueDate(referenceDate, baseCalendar, quoteCalendar);
        return {CashflowT<ScalarT>{payDate, currencyFromCode(pair.baseCcy), baseNotional},
                CashflowT<ScalarT>{payDate, currencyFromCode(pair.quoteCcy),
                                   -baseNotional * ScalarT(spot)}};
    }

    /// Settlement cashflows in the instrument's own quote units.
    std::vector<Cashflow> cashflows(const datetime::Date& referenceDate,
                                    const datetime::Calendar& baseCalendar,
                                    const datetime::Calendar& quoteCalendar) const {
        return cashflowsT<double>(referenceDate, baseCalendar, quoteCalendar, baseNotional);
    }
};

/// Forward trade: exchanges `baseNotional` base units at the value date. The
/// value date is either stored or derived from the tenor and the pair's
/// settlement lags.
struct FxForward {
    markets::FXDescriptor pair; ///< Base/quote currency pair
    datetime::Date tradeDate;   ///< Trade date
    datetime::Date valueDate;   ///< Settlement (value) date
    datetime::Period tenor;     ///< Tenor from the spot date when the value date is unset
    double spot = 0.0;          ///< Quote units per base unit
    double strike = 0.0;        ///< Contracted forward outright
    double baseNotional = 0.0;  ///< Signed base amount, positive buys base
    /// True when the base currency is the collateral currency, so the pair's
    /// non-collateral curve is the quote currency.
    bool isBaseCollateral = false;
    datetime::Calendar baseCalendar;  ///< Base-currency settlement calendar
    datetime::Calendar quoteCalendar; ///< Quote-currency settlement calendar
    int baseSpotLag = 2;              ///< Base-currency settlement lag in days
    int quoteSpotLag = 2;             ///< Quote-currency settlement lag in days
    datetime::BusinessDayConvention convention =
        datetime::BusinessDayConvention::ModifiedFollowing;         ///< Settlement roll rule
    datetime::DayCounter dayCounter{datetime::DayCount::Actual360}; ///< Quote day count
    /// Zero clock used to turn dates into discount times.
    datetime::DayCounter zeroDayCounter{datetime::DayCount::Actual365Fixed};

    /// Risk date: the settlement date.
    datetime::Date date() const { return valueDate; }

    /// Quoted target: the contracted outright.
    double target() const { return strike; }

    /// Contracted outright.
    double outright() const { return strike; }

    /// Near settlement date.
    datetime::Date nearDate() const { return valueDate; }

    /// Far settlement date; identical to the near date for a forward.
    datetime::Date farDate() const { return valueDate; }

    /// Model outright over a base/quote curve set under covered interest
    /// parity, evaluated at the forward's value date.
    template <typename ScalarT, typename CurveSetT>
    ScalarT impliedQuote(const CurveSetT& curves) const {
        const double t = zeroDayCounter.yearFraction(tradeDate, valueDate);
        return ScalarT(spot) * curves.base(t) / curves.quote(t);
    }

    /// Value date: stored when set, otherwise rolled from the reference date
    /// through the spot date and the tenor.
    datetime::Date resolveValueDate(const datetime::Date& referenceDate,
                                    const datetime::Calendar& baseCalendar,
                                    const datetime::Calendar& quoteCalendar) const {
        if (valueDate.serial() != 0) {
            return valueDate;
        }
        if (tenor.length() != 0) {
            return fxForwardDate(referenceDate, baseCalendar, quoteCalendar, baseSpotLag,
                                 quoteSpotLag, tenor, convention);
        }
        return fxSpotDate(referenceDate, baseCalendar, quoteCalendar, baseSpotLag, quoteSpotLag,
                          convention);
    }

    /// Settlement cashflows with scalar-templated amounts: the base leg and the
    /// quote counter-leg at the contracted strike.
    template <typename ScalarT>
    std::vector<CashflowT<ScalarT>>
    cashflowsT(const datetime::Date& referenceDate, const datetime::Calendar& baseCalendar,
               const datetime::Calendar& quoteCalendar, const ScalarT& baseNotional) const {
        const datetime::Date payDate = resolveValueDate(referenceDate, baseCalendar, quoteCalendar);
        return {CashflowT<ScalarT>{payDate, currencyFromCode(pair.baseCcy), baseNotional},
                CashflowT<ScalarT>{payDate, currencyFromCode(pair.quoteCcy),
                                   -baseNotional * ScalarT(strike)}};
    }

    /// Settlement cashflows in the instrument's own quote units.
    std::vector<Cashflow> cashflows(const datetime::Date& referenceDate,
                                    const datetime::Calendar& baseCalendar,
                                    const datetime::Calendar& quoteCalendar) const {
        return cashflowsT<double>(referenceDate, baseCalendar, quoteCalendar, baseNotional);
    }
};

/// FX swap: a funded forward. Positive `baseNotional` sells base at the near
/// leg and buys it back at the far leg, so the value is the far forward minus
/// the near forward and the near leg settles at the pair's spot rate.
struct FxSwap {
    datetime::Date start;    ///< Near leg settlement date
    datetime::Date maturity; ///< Far leg settlement date
    double spot = 0.0;       ///< Spot rate (quote units per base unit)
    double points = 0.0;     ///< Forward points quoted from spot
    double outright = 0.0;   ///< Far outright, used with `QuoteConvention::Outright`
    markets::QuoteConvention convention = markets::QuoteConvention::Points; ///< Quote style
    /// True when the base currency is the collateral currency, so the pair's
    /// non-collateral (foreign) curve is the quote currency.
    bool isFxBaseCollateral = false;
    datetime::DayCounter dayCounter{datetime::DayCount::Actual360}; ///< Quote day count

    markets::FXDescriptor pair;       ///< Base/quote currency pair
    datetime::Date tradeDate;         ///< Trade date (also the implied-quote reference)
    datetime::Date spotDate;          ///< Pair spot date
    datetime::Period farTenor;        ///< Far tenor from spot when `maturity` is unset
    double baseNotional = 0.0;        ///< Signed base amount, positive buys at the far leg
    double pointsScale = 1.0;         ///< Price value of one point
    datetime::Calendar baseCalendar;  ///< Base-currency settlement calendar
    datetime::Calendar quoteCalendar; ///< Quote-currency settlement calendar
    int baseSpotLag = 2;              ///< Base-currency settlement lag in days
    int quoteSpotLag = 2;             ///< Quote-currency settlement lag in days
    datetime::BusinessDayConvention businessDayConvention =
        datetime::BusinessDayConvention::ModifiedFollowing; ///< Settlement roll rule
    /// Zero clock used to turn dates into discount times.
    datetime::DayCounter zeroDayCounter{datetime::DayCount::Actual365Fixed};

    /// Risk date: the far leg settlement date.
    datetime::Date date() const { return maturity; }

    /// Quoted target in the instrument's own convention.
    double target() const {
        return convention == markets::QuoteConvention::Points ? points : outright;
    }

    /// Near settlement date.
    datetime::Date nearDate() const { return start; }

    /// Far settlement date.
    datetime::Date farDate() const { return maturity; }

    /// Far outright from the quote convention and the point scale.
    double quotedOutright() const {
        return convention == markets::QuoteConvention::Points ? spot + points * pointsScale
                                                              : outright;
    }

    /// Forward points from the quote convention and the point scale.
    double quotedPoints() const {
        return convention == markets::QuoteConvention::Points ? points
                                                              : (outright - spot) / pointsScale;
    }

    /// Model quote in the instrument's own convention over a base/quote curve
    /// set under covered interest parity, evaluated at the far date.
    template <typename ScalarT, typename CurveSetT>
    ScalarT impliedQuote(const CurveSetT& curves) const {
        const double t = zeroDayCounter.yearFraction(tradeDate, maturity);
        const ScalarT model = ScalarT(spot) * curves.base(t) / curves.quote(t);
        return convention == markets::QuoteConvention::Points
                   ? (model - ScalarT(spot)) / ScalarT(pointsScale)
                   : model;
    }

    /// Near leg settlement date: stored when set, otherwise the joint spot date
    /// rolled from `referenceDate`.
    datetime::Date resolveStart(const datetime::Date& referenceDate,
                                const datetime::Calendar& baseCalendar,
                                const datetime::Calendar& quoteCalendar) const {
        if (start.serial() != 0) {
            return start;
        }
        return fxSpotDate(referenceDate, baseCalendar, quoteCalendar, baseSpotLag, quoteSpotLag,
                          businessDayConvention);
    }

    /// Far leg settlement date: stored when set, otherwise the spot date
    /// advanced by `farTenor`.
    datetime::Date resolveMaturity(const datetime::Date& referenceDate,
                                   const datetime::Calendar& baseCalendar,
                                   const datetime::Calendar& quoteCalendar) const {
        if (maturity.serial() != 0) {
            return maturity;
        }
        return fxForwardDate(referenceDate, baseCalendar, quoteCalendar, baseSpotLag, quoteSpotLag,
                             farTenor, businessDayConvention);
    }

    /// Settlement cashflows with scalar-templated amounts: the near leg sells
    /// base at spot, the far leg buys it at the quoted outright.
    template <typename ScalarT>
    std::vector<CashflowT<ScalarT>>
    cashflowsT(const datetime::Date& referenceDate, const datetime::Calendar& baseCalendar,
               const datetime::Calendar& quoteCalendar, const ScalarT& baseNotional) const {
        const datetime::Date nearDate = resolveStart(referenceDate, baseCalendar, quoteCalendar);
        const datetime::Date farDate = resolveMaturity(referenceDate, baseCalendar, quoteCalendar);
        return {CashflowT<ScalarT>{nearDate, currencyFromCode(pair.baseCcy), -baseNotional},
                CashflowT<ScalarT>{nearDate, currencyFromCode(pair.quoteCcy),
                                   baseNotional * ScalarT(spot)},
                CashflowT<ScalarT>{farDate, currencyFromCode(pair.baseCcy), baseNotional},
                CashflowT<ScalarT>{farDate, currencyFromCode(pair.quoteCcy),
                                   -baseNotional * ScalarT(quotedOutright())}};
    }

    /// Settlement cashflows in the instrument's own quote units.
    std::vector<Cashflow> cashflows(const datetime::Date& referenceDate,
                                    const datetime::Calendar& baseCalendar,
                                    const datetime::Calendar& quoteCalendar) const {
        return cashflowsT<double>(referenceDate, baseCalendar, quoteCalendar, baseNotional);
    }
};

/// Spot trade factory: resolves the value date from the trade date and the
/// pair's settlement lags on the joint calendar.
inline FxSpot makeFxSpot(const markets::FXDescriptor& pair, const datetime::Date& tradeDate,
                         double spot, double baseNotional, const datetime::Calendar& baseCalendar,
                         const datetime::Calendar& quoteCalendar, int baseSpotLag = 2,
                         int quoteSpotLag = 2,
                         datetime::BusinessDayConvention convention =
                             datetime::BusinessDayConvention::ModifiedFollowing) {
    if (tradeDate.serial() == 0) {
        throw std::invalid_argument("makeFxSpot: trade date is required");
    }
    if (!(spot > 0.0)) {
        throw std::invalid_argument("makeFxSpot: spot must be positive");
    }
    FxSpot out;
    out.pair = pair;
    out.tradeDate = tradeDate;
    out.valueDate =
        fxSpotDate(tradeDate, baseCalendar, quoteCalendar, baseSpotLag, quoteSpotLag, convention);
    out.spot = spot;
    out.baseNotional = baseNotional;
    out.baseCalendar = baseCalendar;
    out.quoteCalendar = quoteCalendar;
    out.baseSpotLag = baseSpotLag;
    out.quoteSpotLag = quoteSpotLag;
    out.convention = convention;
    return out;
}

/// Forward factory from an explicit value date.
inline FxForward makeFxForward(const markets::FXDescriptor& pair, const datetime::Date& tradeDate,
                               const datetime::Date& valueDate, double spot, double strike,
                               double baseNotional, const datetime::Calendar& baseCalendar,
                               const datetime::Calendar& quoteCalendar, int baseSpotLag = 2,
                               int quoteSpotLag = 2, bool isBaseCollateral = false,
                               datetime::BusinessDayConvention convention =
                                   datetime::BusinessDayConvention::ModifiedFollowing) {
    if (tradeDate.serial() == 0 || valueDate.serial() == 0) {
        throw std::invalid_argument("makeFxForward: trade and value dates are required");
    }
    if (!(valueDate >= tradeDate)) {
        throw std::invalid_argument("makeFxForward: value date must not precede the trade date");
    }
    FxForward out;
    out.pair = pair;
    out.tradeDate = tradeDate;
    out.valueDate = valueDate;
    out.spot = spot;
    out.strike = strike;
    out.baseNotional = baseNotional;
    out.isBaseCollateral = isBaseCollateral;
    out.baseCalendar = baseCalendar;
    out.quoteCalendar = quoteCalendar;
    out.baseSpotLag = baseSpotLag;
    out.quoteSpotLag = quoteSpotLag;
    out.convention = convention;
    return out;
}

/// Forward factory from a tenor: the value date is the joint spot date
/// advanced by `tenor` on the joint calendar.
inline FxForward makeFxForward(const markets::FXDescriptor& pair, const datetime::Date& tradeDate,
                               const datetime::Period& tenor, double spot, double strike,
                               double baseNotional, const datetime::Calendar& baseCalendar,
                               const datetime::Calendar& quoteCalendar, int baseSpotLag = 2,
                               int quoteSpotLag = 2, bool isBaseCollateral = false,
                               datetime::BusinessDayConvention convention =
                                   datetime::BusinessDayConvention::ModifiedFollowing) {
    const datetime::Date valueDate = fxForwardDate(tradeDate, baseCalendar, quoteCalendar,
                                                   baseSpotLag, quoteSpotLag, tenor, convention);
    FxForward out =
        makeFxForward(pair, tradeDate, valueDate, spot, strike, baseNotional, baseCalendar,
                      quoteCalendar, baseSpotLag, quoteSpotLag, isBaseCollateral, convention);
    out.tenor = tenor;
    return out;
}

/// Swap factory from explicit near and far dates. An unset near date settles
/// at the joint spot date; the spot date is recorded for both.
inline FxSwap makeFxSwap(const markets::FXDescriptor& pair, const datetime::Date& tradeDate,
                         const datetime::Date& start, const datetime::Date& maturity, double spot,
                         double quoteValue, markets::QuoteConvention convention,
                         bool isBaseCollateral, double pointsScale,
                         const datetime::Calendar& baseCalendar,
                         const datetime::Calendar& quoteCalendar, int baseSpotLag = 2,
                         int quoteSpotLag = 2,
                         datetime::BusinessDayConvention businessDayConvention =
                             datetime::BusinessDayConvention::ModifiedFollowing) {
    if (tradeDate.serial() == 0) {
        throw std::invalid_argument("makeFxSwap: trade date is required");
    }
    if (!(spot > 0.0)) {
        throw std::invalid_argument("makeFxSwap: spot must be positive");
    }
    if (!(pointsScale > 0.0)) {
        throw std::invalid_argument("makeFxSwap: points scale must be positive");
    }
    if (maturity.serial() == 0) {
        throw std::invalid_argument("makeFxSwap: far date is required");
    }
    FxSwap out;
    out.pair = pair;
    out.tradeDate = tradeDate;
    out.spotDate = fxSpotDate(tradeDate, baseCalendar, quoteCalendar, baseSpotLag, quoteSpotLag,
                              businessDayConvention);
    out.start = start.serial() != 0 ? start : out.spotDate;
    out.maturity = maturity;
    if (maturity.serial() != 0 && !(out.start < maturity)) {
        throw std::invalid_argument("makeFxSwap: far date must be after the near date");
    }
    out.spot = spot;
    if (convention == markets::QuoteConvention::Points) {
        out.points = quoteValue;
        out.outright = spot + quoteValue * pointsScale;
    } else {
        out.outright = quoteValue;
        out.points = (quoteValue - spot) / pointsScale;
    }
    out.convention = convention;
    out.isFxBaseCollateral = isBaseCollateral;
    out.pointsScale = pointsScale;
    out.baseCalendar = baseCalendar;
    out.quoteCalendar = quoteCalendar;
    out.baseSpotLag = baseSpotLag;
    out.quoteSpotLag = quoteSpotLag;
    out.businessDayConvention = businessDayConvention;
    return out;
}

/// Swap factory from a far tenor: a spot-starting swap whose near leg settles
/// at the joint spot date and whose far leg settles at spot + `farTenor`.
inline FxSwap makeFxSwap(const markets::FXDescriptor& pair, const datetime::Date& tradeDate,
                         const datetime::Period& farTenor, double spot, double quoteValue,
                         markets::QuoteConvention convention, bool isBaseCollateral,
                         double pointsScale, const datetime::Calendar& baseCalendar,
                         const datetime::Calendar& quoteCalendar, int baseSpotLag = 2,
                         int quoteSpotLag = 2,
                         datetime::BusinessDayConvention businessDayConvention =
                             datetime::BusinessDayConvention::ModifiedFollowing) {
    const datetime::Date spotDate = fxSpotDate(tradeDate, baseCalendar, quoteCalendar, baseSpotLag,
                                               quoteSpotLag, businessDayConvention);
    const datetime::Date maturity =
        fxForwardDate(tradeDate, baseCalendar, quoteCalendar, baseSpotLag, quoteSpotLag, farTenor,
                      businessDayConvention);
    FxSwap out = makeFxSwap(pair, tradeDate, spotDate, maturity, spot, quoteValue, convention,
                            isBaseCollateral, pointsScale, baseCalendar, quoteCalendar, baseSpotLag,
                            quoteSpotLag, businessDayConvention);
    out.farTenor = farTenor;
    return out;
}

} // namespace quantape::instruments
