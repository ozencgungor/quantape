#pragma once

#include "quantape/datetime/BusinessDayConvention.h"
#include "quantape/datetime/Calendar.h"
#include "quantape/datetime/Date.h"
#include "quantape/datetime/DayCounter.h"
#include "quantape/instruments/Cashflow.h"
#include "quantape/pricing/IrMath.h"

#include <stdexcept>
#include <string_view>
#include <type_traits>

namespace quantape::instruments {
/**
 * @file IrInstruments.h
 * @brief Plain-data interest-rate instruments: deposit, repo, FRA, future,
 *        IRS and basis swap
 *
 * Each struct carries only its own convention fields plus the precomputed
 * quote times (`prepare*`) that turn dates into zero times once, so a Brent
 * trial only reads discount factors. The member `impliedQuote<ScalarT>`
 * forwards to the scalar algebra in `pricing/IrMath.h` and reads either a
 * discount curve set or a forecast curve set, so one type serves the discount
 * and forecast bootstraps. Every member is scalar generic, including the swap
 * members, so a quote residual can be evaluated on a `var` curve.
 *
 * `date()` is the instrument's risk maturity (calendar-adjusted where the
 * convention rolls dates, the quoted end for exchange futures), `target()` the
 * quoted rate and `priority()` the overlap-filter rank
 * (Future > Fra > Deposit > Repo).
 */

/// Money-market zero coupon / repo: `D(T) = 1 / (1 + r tau)`.
struct Deposit {
    datetime::Date maturity{};
    datetime::Date start{}; ///< Forecast-side accrual start; unset means the reference date
    datetime::Calendar calendar{};
    datetime::BusinessDayConvention businessDayConvention =
        datetime::BusinessDayConvention::ModifiedFollowing;
    datetime::DayCounter quoteDayCounter{datetime::DayCount::Actual360};
    double quote = 0.0;
    Currency currency{};
    pricing::SimpleForwardQuoteTimes times{};

    /// Risk maturity: quoted date rolled on the calendar.
    datetime::Date date() const { return calendar.adjust(maturity, businessDayConvention); }

    /// Quoted simple rate.
    double target() const { return quote; }

    /// Short kind label.
    static constexpr std::string_view kindName() noexcept { return "Deposit"; }

    /// Overlap-filter rank.
    static constexpr int priority() noexcept { return 2; }

    /// Precomputes the discount-curve quote times at `referenceDate`.
    Deposit& prepareDiscount(const datetime::Date& referenceDate,
                             const datetime::DayCounter& zeroDayCounter) {
        times =
            pricing::prepareDepositTimes(date(), referenceDate, quoteDayCounter, zeroDayCounter);
        return *this;
    }

    /// Precomputes the forecast-curve quote times with an optional explicit
    /// accrual start (the reference date when unset).
    Deposit& prepareForecast(const datetime::Date& startDate, const datetime::Date& referenceDate,
                             const datetime::DayCounter& zeroDayCounter) {
        times = pricing::prepareSimpleForwardTimes(startDate, date(), referenceDate,
                                                   quoteDayCounter, zeroDayCounter);
        return *this;
    }

    /// Model quote over a discount or forecast curve set.
    template <typename ScalarT, typename CurveSetT>
    ScalarT impliedQuote(const CurveSetT& curves) const {
        return pricing::impliedDepositQuote<ScalarT>(times, pricing::discountProvider(curves));
    }
};

/// Repo / buy-sell-back: the deposit algebra with a separate risk bucket.
struct Repo {
    datetime::Date maturity{};
    datetime::Date start{}; ///< Forecast-side accrual start; unset means the reference date
    datetime::Calendar calendar{};
    datetime::BusinessDayConvention businessDayConvention =
        datetime::BusinessDayConvention::ModifiedFollowing;
    datetime::DayCounter quoteDayCounter{datetime::DayCount::Actual360};
    double quote = 0.0;
    Currency currency{};
    pricing::SimpleForwardQuoteTimes times{};

    /// Risk maturity: quoted date rolled on the calendar.
    datetime::Date date() const { return calendar.adjust(maturity, businessDayConvention); }

    /// Quoted simple rate.
    double target() const { return quote; }

    /// Short kind label.
    static constexpr std::string_view kindName() noexcept { return "Repo"; }

    /// Overlap-filter rank.
    static constexpr int priority() noexcept { return 3; }

    /// Precomputes the discount-curve quote times at `referenceDate`.
    Repo& prepareDiscount(const datetime::Date& referenceDate,
                          const datetime::DayCounter& zeroDayCounter) {
        times =
            pricing::prepareDepositTimes(date(), referenceDate, quoteDayCounter, zeroDayCounter);
        return *this;
    }

    /// Precomputes the forecast-curve quote times with an optional explicit
    /// accrual start (the reference date when unset).
    Repo& prepareForecast(const datetime::Date& startDate, const datetime::Date& referenceDate,
                          const datetime::DayCounter& zeroDayCounter) {
        times = pricing::prepareSimpleForwardTimes(startDate, date(), referenceDate,
                                                   quoteDayCounter, zeroDayCounter);
        return *this;
    }

    /// Model quote over a discount or forecast curve set.
    template <typename ScalarT, typename CurveSetT>
    ScalarT impliedQuote(const CurveSetT& curves) const {
        return pricing::impliedRepoQuote<ScalarT>(times, pricing::discountProvider(curves));
    }
};

/// Forward rate agreement: `r = (D(t1)/D(t2) - 1) / tau` with an optional
/// shifted-lognormal convexity exponent.
struct Fra {
    datetime::Date start{};
    datetime::Date maturity{};
    datetime::Calendar calendar{};
    datetime::BusinessDayConvention businessDayConvention =
        datetime::BusinessDayConvention::ModifiedFollowing;
    datetime::DayCounter quoteDayCounter{datetime::DayCount::Actual360};
    double fraConvexityExponent = 0.0;
    double quote = 0.0;
    Currency currency{};
    pricing::SimpleForwardQuoteTimes times{};

    /// Risk maturity: quoted date rolled on the calendar.
    datetime::Date date() const { return calendar.adjust(maturity, businessDayConvention); }

    /// Quoted FRA rate.
    double target() const { return quote; }

    /// Short kind label.
    static constexpr std::string_view kindName() noexcept { return "Fra"; }

    /// Overlap-filter rank.
    static constexpr int priority() noexcept { return 1; }

    /// Precomputes the discount-curve quote times, rolling both accrual ends.
    Fra& prepareDiscount(const datetime::Date& referenceDate,
                         const datetime::DayCounter& zeroDayCounter) {
        times = pricing::prepareFraTimes(calendar.adjust(start, businessDayConvention), date(),
                                         referenceDate, quoteDayCounter, zeroDayCounter);
        return *this;
    }

    /// Precomputes the forecast-curve quote times; the accrual start is
    /// quoted directly (no calendar roll) and `startDate` is the reference
    /// date when the instrument start is unset.
    Fra& prepareForecast(const datetime::Date& startDate, const datetime::Date& referenceDate,
                         const datetime::DayCounter& zeroDayCounter) {
        times = pricing::prepareSimpleForwardTimes(startDate, date(), referenceDate,
                                                   quoteDayCounter, zeroDayCounter);
        return *this;
    }

    /// Model quote over a discount or forecast curve set.
    template <typename ScalarT, typename CurveSetT>
    ScalarT impliedQuote(const CurveSetT& curves) const {
        return pricing::impliedFraQuote<ScalarT>(times, fraConvexityExponent,
                                                 pricing::discountProvider(curves));
    }
};

/// Exchange-traded rate future: the period forward plus a convexity
/// adjustment under every underlying style.
struct Future {
    datetime::Date start{};
    datetime::Date maturity{};
    datetime::Calendar calendar{};
    datetime::DayCounter quoteDayCounter{datetime::DayCount::Actual360};
    pricing::FutureStyle futureStyle = pricing::FutureStyle::Simple;
    pricing::AveragingStyle averagingStyle = pricing::AveragingStyle::Arithmetic;
    double convexityAdjustment = 0.0;
    double quote = 0.0;
    Currency currency{};
    pricing::FutureQuoteTimes times{};

    /// Risk maturity: exchange dates are quoted directly, not rolled.
    datetime::Date date() const { return maturity; }

    /// Quoted futures rate.
    double target() const { return quote; }

    /// Short kind label.
    static constexpr std::string_view kindName() noexcept { return "Future"; }

    /// Overlap-filter rank.
    static constexpr int priority() noexcept { return 0; }

    /// Precomputes the quote times for the instrument's style; `context`
    /// prefixes thrown messages so each caller keeps its own error text.
    Future& prepare(const datetime::Date& referenceDate, const datetime::DayCounter& zeroDayCounter,
                    std::string_view context) {
        times = pricing::prepareFutureTimes(start, maturity, calendar, quoteDayCounter, futureStyle,
                                            averagingStyle, convexityAdjustment, referenceDate,
                                            zeroDayCounter, context);
        return *this;
    }

    /// Model quote over a discount or forecast curve set.
    template <typename ScalarT, typename CurveSetT>
    ScalarT impliedQuote(const CurveSetT& curves) const {
        return pricing::impliedFutureQuote<ScalarT>(times, pricing::discountProvider(curves));
    }
};

/// Fixed-vs-floating par interest-rate swap on a forecast curve: the floating
/// leg fixes on the forecast curve, the fixed leg and all discounting come from
/// the exogenous discount curve (the parent in a standard stack). Fixed and
/// floating schedules may differ.
struct Irs {
    datetime::Date maturity{};
    datetime::Date start{};        ///< Effective date (past for seasoned swaps)
    bool firstCouponFixed = false; ///< First floating coupon already fixed
    double firstCouponRate = 0.0;  ///< Known first floating coupon rate
    double quote = 0.0;            ///< Par rate (decimal)
    datetime::Period floatTenor{3, datetime::TimeUnit::Months};
    datetime::Calendar floatCalendar{};
    datetime::DayCounter floatDayCounter{datetime::DayCount::Actual360};
    datetime::Period fixedTenor{1, datetime::TimeUnit::Years};
    datetime::Calendar fixedCalendar{};
    datetime::DayCounter fixedDayCounter{datetime::DayCount::Thirty360BondBasis};
    datetime::BusinessDayConvention businessDayConvention =
        datetime::BusinessDayConvention::ModifiedFollowing;
    int paymentLag = 0; ///< Calendar-day payment lag, then business-day adjusted
    pricing::IrsQuoteTimes times{};

    /// Risk maturity: the fixed-leg termination rolled on the fixed calendar.
    datetime::Date date() const { return fixedCalendar.adjust(maturity, businessDayConvention); }

    /// Quoted par rate.
    double target() const { return quote; }

    /// Short kind label.
    static constexpr std::string_view kindName() noexcept { return "Irs"; }

    /// Precomputes both legs' coupon times with the caller's zero clock.
    Irs& prepareForecast(const datetime::Date& referenceDate,
                         const datetime::DayCounter& zeroDayCounter) {
        times = pricing::makeIrsQuoteTimes(*this, referenceDate, zeroDayCounter);
        return *this;
    }

    /// Model par rate over a forecast curve set; scalar generic end to end.
    template <typename ScalarT, typename CurveSetT>
    ScalarT impliedQuote(const CurveSetT& curves) const {
        return pricing::impliedIrsParRate<ScalarT>(times, curves.forecastCurve,
                                                   curves.discountCurve);
    }
};

/// Par float-vs-float basis swap on a forecast curve (IBOR vs OIS or IBOR vs
/// IBOR): the spread sits on the parent or child leg per `spreadOnParentLeg`.
struct BasisSwap {
    datetime::Date maturity{};
    double spread = 0.0;           ///< Quoted basis spread (decimal)
    bool spreadOnParentLeg = true; ///< Quote convention: spread on the OIS parent leg
    datetime::Period floatTenor{3, datetime::TimeUnit::Months};
    datetime::DayCounter quoteDayCounter{datetime::DayCount::Actual360};
    datetime::Calendar calendar{};
    datetime::BusinessDayConvention businessDayConvention =
        datetime::BusinessDayConvention::ModifiedFollowing;
    pricing::BasisQuoteTimes times{};

    /// Risk maturity: quoted date rolled on the calendar.
    datetime::Date date() const { return calendar.adjust(maturity, businessDayConvention); }

    /// Quoted basis spread.
    double target() const { return spread; }

    /// Short kind label.
    static constexpr std::string_view kindName() noexcept { return "Basis"; }

    /// Precomputes the coupon times with the caller's zero clock.
    BasisSwap& prepareForecast(const datetime::Date& referenceDate,
                               const datetime::DayCounter& zeroDayCounter) {
        times = pricing::makeBasisQuoteTimes(*this, referenceDate, zeroDayCounter);
        return *this;
    }

    /// Model spread over a forecast curve set; scalar generic end to end.
    template <typename ScalarT, typename CurveSetT>
    ScalarT impliedQuote(const CurveSetT& curves) const {
        const auto& child = curves.forecastCurve;
        const auto discountAt = [&curves](double t) { return curves.discountCurve.discount(t); };
        return pricing::impliedBasisSpread<ScalarT>(times, child, child.parent(), discountAt,
                                                    spreadOnParentLeg);
    }
};

/// Dated deposit factory: validates the maturity, resolves the calendar roll
/// and precomputes the discount-curve times.
inline Deposit makeDeposit(const datetime::Date& maturity, double quote,
                           const datetime::Calendar& calendar,
                           datetime::BusinessDayConvention businessDayConvention,
                           const datetime::DayCounter& quoteDayCounter,
                           const datetime::Date& referenceDate,
                           const datetime::DayCounter& zeroDayCounter) {
    if (maturity.serial() == 0) {
        throw std::invalid_argument("makeDeposit: maturity is required");
    }
    Deposit out;
    out.maturity = maturity;
    out.calendar = calendar;
    out.businessDayConvention = businessDayConvention;
    out.quoteDayCounter = quoteDayCounter;
    out.quote = quote;
    out.prepareDiscount(referenceDate, zeroDayCounter);
    return out;
}

/// Dated repo factory: validates the maturity, resolves the calendar roll and
/// precomputes the discount-curve times.
inline Repo makeRepo(const datetime::Date& maturity, double quote,
                     const datetime::Calendar& calendar,
                     datetime::BusinessDayConvention businessDayConvention,
                     const datetime::DayCounter& quoteDayCounter,
                     const datetime::Date& referenceDate,
                     const datetime::DayCounter& zeroDayCounter) {
    if (maturity.serial() == 0) {
        throw std::invalid_argument("makeRepo: maturity is required");
    }
    Repo out;
    out.maturity = maturity;
    out.calendar = calendar;
    out.businessDayConvention = businessDayConvention;
    out.quoteDayCounter = quoteDayCounter;
    out.quote = quote;
    out.prepareDiscount(referenceDate, zeroDayCounter);
    return out;
}

/// Dated FRA factory: validates both accrual ends and precomputes the
/// discount-curve times.
inline Fra makeFra(const datetime::Date& start, const datetime::Date& maturity, double quote,
                   const datetime::Calendar& calendar,
                   datetime::BusinessDayConvention businessDayConvention,
                   const datetime::DayCounter& quoteDayCounter, const datetime::Date& referenceDate,
                   const datetime::DayCounter& zeroDayCounter) {
    if (start.serial() == 0 || maturity.serial() == 0) {
        throw std::invalid_argument("makeFra: start and maturity are required");
    }
    Fra out;
    out.start = start;
    out.maturity = maturity;
    out.calendar = calendar;
    out.businessDayConvention = businessDayConvention;
    out.quoteDayCounter = quoteDayCounter;
    out.quote = quote;
    out.prepareDiscount(referenceDate, zeroDayCounter);
    return out;
}

/// Dated future factory: validates the quoted reference period and precomputes
/// the fixing grid for the styled quote.
inline Future makeFuture(const datetime::Date& start, const datetime::Date& maturity, double quote,
                         pricing::FutureStyle futureStyle, pricing::AveragingStyle averagingStyle,
                         double convexityAdjustment, const datetime::Calendar& calendar,
                         const datetime::DayCounter& quoteDayCounter,
                         const datetime::Date& referenceDate,
                         const datetime::DayCounter& zeroDayCounter) {
    if (start.serial() == 0 || maturity.serial() == 0) {
        throw std::invalid_argument("makeFuture: start and maturity are required");
    }
    if (!(maturity > start)) {
        throw std::invalid_argument("makeFuture: maturity must be after the start");
    }
    Future out;
    out.start = start;
    out.maturity = maturity;
    out.futureStyle = futureStyle;
    out.averagingStyle = averagingStyle;
    out.convexityAdjustment = convexityAdjustment;
    out.calendar = calendar;
    out.quoteDayCounter = quoteDayCounter;
    out.quote = quote;
    out.prepare(referenceDate, zeroDayCounter, "makeFuture");
    return out;
}

} // namespace quantape::instruments
