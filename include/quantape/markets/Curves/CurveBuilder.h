#pragma once

#include "quantape/datetime/Calendar.h"
#include "quantape/datetime/Date.h"
#include "quantape/datetime/DayCounter.h"
#include "quantape/datetime/Period.h"
#include "quantape/datetime/Schedule.h"
#include "quantape/datetime/TimeConversion.h"
#include "quantape/instruments/IrInstruments.h"
#include "quantape/markets/Curves/BootstrapInstrument.h"
#include "quantape/markets/Curves/DiscountCurve.h"
#include "quantape/markets/Curves/SpreadCurve.h"
#include "quantape/math/Solvers/BrentSolver.h"
#include "quantape/pricing/Ir.h"
#include "quantape/pricing/IrMath.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <numeric>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>
#include <vector>

namespace quantape::markets {
/**
 * @file CurveBuilder.h
 * @brief Sequential exact-fit bootstrap of a discount curve
 *
 * Pillars are solved in maturity order with a bracketed 1-D root find
 * (`math::BrentSolver`) on the curve's zero-rate nodes: deposits and repos,
 * FRAs with a start date and par OIS swaps (compounded overnight leg versus
 * fixed leg, per-coupon payment lags). Basis pillars build a spread curve over
 * a frozen parent. Every instrument carries its own calendar, business-day
 * convention and day count.
 */

enum class PillarKind : std::uint8_t {
    Deposit, ///< Money-market zero coupon: D(T) = 1 / (1 + r tau)
    Repo,    ///< Repo/buy-sell-back: same math as a deposit, separate risk bucket
    Fra,     ///< Forward rate agreement: r = (D(t1)/D(t2) - 1) / tau
    Future,  ///< Exchange-traded rate future: FRA forward plus convexity
    OisSwap, ///< Par OIS: r = (D(start) - D(end)) / annuity (telescoping float)
};

constexpr std::string_view pillarKindName(PillarKind kind) noexcept {
    switch (kind) {
        case PillarKind::Deposit:
            return "Deposit";
        case PillarKind::Repo:
            return "Repo";
        case PillarKind::Fra:
            return "Fra";
        case PillarKind::Future:
            return "Future";
        case PillarKind::OisSwap:
            return "OisSwap";
    }
    return "Unknown";
}

/// Rate-future underlying style, defined with the scalar quote algebra.
using pricing::AveragingStyle;
using pricing::FutureStyle;

/// One bootstrap instrument with its market convention.
struct CurvePillar {
    datetime::Date maturity;
    datetime::Date start; ///< FRA start / future fixing / OIS-IRS effective date
    PillarKind kind = PillarKind::Deposit;
    double quote = 0.0; ///< Simple rate (deposit), par rate (OIS) or futures rate, decimal
    double convexityAdjustment = 0.0;              ///< Futures: added to the fitted forward rate
    FutureStyle futureStyle = FutureStyle::Simple; ///< Futures: underlying style
    AveragingStyle averagingStyle = AveragingStyle::Arithmetic; ///< Averaged futures convention
    double fraConvexityExponent = 0.0; ///< FRA: exponent C, R = ((1+f tau) e^C - 1)/tau
    bool firstCouponFixed = false;     ///< OIS/IRS: first floating coupon already fixed
    double firstCouponRate = 0.0;      ///< OIS/IRS: known first floating coupon rate
    datetime::DayCounter quoteDayCounter{datetime::DayCount::Actual360};
    datetime::Calendar calendar{};
    datetime::Period fixedTenor{1, datetime::TimeUnit::Years};
    int paymentLag = 0; ///< Calendar-day payment lag, then business-day adjusted
    datetime::BusinessDayConvention businessDayConvention =
        datetime::BusinessDayConvention::ModifiedFollowing;
};

/// One par basis-swap pillar (IBOR vs OIS or IBOR vs IBOR, spread on the
/// parent or child leg per `spreadOnParentLeg`) for building a spread curve
/// over a parent.
struct BasisPillar {
    datetime::Date maturity;
    double spread = 0.0;           ///< Quoted basis spread (decimal)
    bool spreadOnParentLeg = true; ///< Quote convention: spread on the OIS parent leg
    datetime::Period floatTenor{3, datetime::TimeUnit::Months};
    datetime::DayCounter quoteDayCounter{datetime::DayCount::Actual360};
    datetime::Calendar calendar{};
    datetime::BusinessDayConvention businessDayConvention =
        datetime::BusinessDayConvention::ModifiedFollowing;
};

/// Consecutive business-day fixing dates over `[effective, maturity]`, used by
/// averaged overnight futures; re-exported from the scalar math header.
using pricing::businessDayFixings;

/// One par fixed-vs-floating IRS pillar on a forecast curve: the floating leg
/// fixes on the forecast curve, the fixed leg and all discounting use the
/// exogenous discount curve (the parent in a standard stack). Fixed and
/// floating schedules may differ.
struct IrsPillar {
    datetime::Date maturity;
    datetime::Date start;          ///< Effective date (past for seasoned swaps)
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
};

/// Pillar maturity adjusted on its calendar and business-day convention.
inline datetime::Date adjustedMaturity(const CurvePillar& pillar) {
    return pillar.calendar.adjust(pillar.maturity, pillar.businessDayConvention);
}

/// FRA start adjusted on the pillar's calendar and business-day convention.
inline datetime::Date adjustedStart(const CurvePillar& pillar) {
    return pillar.calendar.adjust(pillar.start, pillar.businessDayConvention);
}

/// Maturity date the instrument actually references in pricing: deposits,
/// repos, FRAs and OIS schedules roll on their calendar and business-day
/// convention, exchange futures use the quoted end date directly.
inline datetime::Date pillarRiskMaturity(const CurvePillar& pillar) {
    switch (pillar.kind) {
        case PillarKind::Repo:
        case PillarKind::Deposit:
        case PillarKind::Fra:
        case PillarKind::OisSwap:
            return adjustedMaturity(pillar);
        case PillarKind::Future:
            return pillar.maturity;
    }
    return pillar.maturity;
}

/// Compact `ddMonyy` tag of a date with a 3-letter English month, e.g.
/// `14Jan13`.
inline std::string compactDateTag(const datetime::Date& date) {
    static constexpr const char* kMonths[12] = {"Jan", "Feb", "Mar", "Apr", "May", "Jun",
                                                "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"};
    std::string tag;
    const unsigned day = date.dayOfMonth();
    const int year = date.year() % 100;
    if (day < 10) {
        tag += '0';
    }
    tag += std::to_string(day);
    tag += kMonths[date.month() - 1];
    if (year < 10) {
        tag += '0';
    }
    tag += std::to_string(year);
    return tag;
}

/// Risk maturity tag of a pillar: the exact compact maturity date below one
/// year, an integer year for whole-year pillars (IMM dates within about three
/// weeks included) and rounded months otherwise.
inline std::string riskMaturityTag(const datetime::Date& adjustedMaturity, double t) {
    if (t < 1.0) {
        return compactDateTag(adjustedMaturity);
    }
    if (std::abs(t - std::round(t)) <= 20.0 / 365.0) {
        return std::to_string(std::lround(t)) + "Y";
    }
    return std::to_string(std::lround(t * 12.0)) + "M";
}

namespace detail {

/// Adapter that presents one OIS pillar as a plain bootstrap instrument: risk
/// maturity and target from the pillar, the par rate from its prepared coupon
/// times through the shared money-market algebra. OIS keeps its tagged-pillar
/// adapter until it lands as a plain instrument.
class OisPillarInstrument {
public:
    OisPillarInstrument(datetime::Date date, double target,
                        std::vector<pricing::OisCouponTimes> coupons)
        : m_date(date), m_target(target), m_coupons(std::move(coupons)) {}

    datetime::Date date() const { return m_date; }
    double target() const { return m_target; }

    template <typename ScalarT, typename CurveSetT>
    ScalarT impliedQuote(const CurveSetT& curves) const {
        return pricing::impliedOisParRate<ScalarT>(m_coupons, curves);
    }

private:
    datetime::Date m_date;
    double m_target = 0.0;
    std::vector<pricing::OisCouponTimes> m_coupons;
};

} // namespace detail

/// Closed-set discount-curve bootstrap instrument: the four money-market
/// families as plain-data instruments plus the OIS adapter.
using DiscountInstrument = Ladder<instruments::Deposit, instruments::Repo, instruments::Fra,
                                  instruments::Future, detail::OisPillarInstrument>;

/// Adapter from the tagged `CurvePillar` POD to the concept instrument: quote
/// times are prepared once per instrument (never per Brent step) with the
/// caller's zero clock, and `date()` is the instrument's risk maturity.
inline DiscountInstrument toInstrument(const CurvePillar& pillar,
                                       const datetime::Date& referenceDate,
                                       const datetime::DayCounter& zeroDayCounter) {
    using Instrument = DiscountInstrument::Instrument;
    switch (pillar.kind) {
        case PillarKind::Deposit: {
            instruments::Deposit out;
            out.maturity = pillar.maturity;
            out.calendar = pillar.calendar;
            out.businessDayConvention = pillar.businessDayConvention;
            out.quoteDayCounter = pillar.quoteDayCounter;
            out.quote = pillar.quote;
            out.prepareDiscount(referenceDate, zeroDayCounter);
            return DiscountInstrument{Instrument{std::move(out)}};
        }
        case PillarKind::Repo: {
            instruments::Repo out;
            out.maturity = pillar.maturity;
            out.calendar = pillar.calendar;
            out.businessDayConvention = pillar.businessDayConvention;
            out.quoteDayCounter = pillar.quoteDayCounter;
            out.quote = pillar.quote;
            out.prepareDiscount(referenceDate, zeroDayCounter);
            return DiscountInstrument{Instrument{std::move(out)}};
        }
        case PillarKind::Fra: {
            instruments::Fra out;
            out.start = pillar.start;
            out.maturity = pillar.maturity;
            out.calendar = pillar.calendar;
            out.businessDayConvention = pillar.businessDayConvention;
            out.quoteDayCounter = pillar.quoteDayCounter;
            out.fraConvexityExponent = pillar.fraConvexityExponent;
            out.quote = pillar.quote;
            out.prepareDiscount(referenceDate, zeroDayCounter);
            return DiscountInstrument{Instrument{std::move(out)}};
        }
        case PillarKind::Future: {
            instruments::Future out;
            out.start = pillar.start;
            out.maturity = pillar.maturity;
            out.calendar = pillar.calendar;
            out.quoteDayCounter = pillar.quoteDayCounter;
            out.futureStyle = pillar.futureStyle;
            out.averagingStyle = pillar.averagingStyle;
            out.convexityAdjustment = pillar.convexityAdjustment;
            out.quote = pillar.quote;
            out.prepare(referenceDate, zeroDayCounter, "impliedQuote");
            return DiscountInstrument{Instrument{std::move(out)}};
        }
        case PillarKind::OisSwap: {
            const datetime::Date effective =
                pillar.start.serial() != 0 ? pillar.start : referenceDate;
            std::vector<pricing::OisCouponTimes> coupons = pricing::prepareOisCouponTimes(
                effective, pillar.maturity, pillar.fixedTenor, pillar.calendar,
                pillar.businessDayConvention, pillar.paymentLag, pillar.quoteDayCounter,
                pillar.firstCouponFixed, pillar.firstCouponRate, referenceDate, zeroDayCounter);
            return DiscountInstrument{Instrument{detail::OisPillarInstrument(
                pillarRiskMaturity(pillar), pillar.quote, std::move(coupons))}};
        }
    }
    throw std::invalid_argument("impliedQuote: unknown pillar kind");
}

/// Model-implied quote of `pillar` against `curve` (exact-fit residual target):
/// maps the tagged pillar to its concept instrument and evaluates it on the
/// curve. Error texts and validation order match the per-kind implementation.
template <typename DoubleT>
inline DoubleT impliedQuote(const CurvePillar& pillar, const datetime::Date& referenceDate,
                            const DiscountCurve<DoubleT>& curve) {
    const DiscountInstrument instrument =
        toInstrument(pillar, referenceDate, curve.zeroDayCounter());
    return instrument.template impliedQuote<DoubleT>(DiscountSet<DiscountCurve<DoubleT>>{curve});
}

/// Annuity of one leg schedule over the discount curve at its payment dates:
/// the schedule starts at `effective` while discount times are measured from
/// `valuationDate` (they differ for seasoned instruments).
template <typename DiscountT>
    requires CurveProvider<DiscountT, double>
inline double legAnnuity(const DiscountT& discount, const datetime::Date& effective,
                         const datetime::Date& valuationDate, const datetime::Date& maturity,
                         const datetime::Period& tenor, const datetime::Calendar& calendar,
                         const datetime::DayCounter& dayCounter,
                         datetime::BusinessDayConvention businessDayConvention, int paymentLag,
                         const datetime::DayCounter& zeroDayCounter) {
    const datetime::Schedule schedule(effective, maturity, tenor, calendar, businessDayConvention,
                                      datetime::DateGeneration::Forward, false,
                                      datetime::BusinessDayConvention::Unadjusted);
    const std::vector<datetime::Date>& dates = schedule.dates();
    double annuity = 0.0;
    for (std::size_t k = 1; k < dates.size(); ++k) {
        const double tau = datetime::yearFraction(dates[k - 1], dates[k], dayCounter);
        const datetime::Date payDate =
            calendar.advance(dates[k], datetime::Period(paymentLag, datetime::TimeUnit::Days),
                             businessDayConvention);
        const double tPay = datetime::yearFraction(valuationDate, payDate, zeroDayCounter);
        annuity += tau * discount.discount(tPay);
    }
    return annuity;
}

/// Fixed-leg annuity of an IRS pillar over the discount curve.
template <typename DiscountT>
    requires CurveProvider<DiscountT, double>
inline double fixedAnnuity(const DiscountT& discount, const IrsPillar& pillar,
                           const datetime::Date& referenceDate,
                           const datetime::DayCounter& zeroDayCounter) {
    const datetime::Date effective = pillar.start.serial() != 0 ? pillar.start : referenceDate;
    return legAnnuity(discount, effective, referenceDate, pillar.maturity, pillar.fixedTenor,
                      pillar.fixedCalendar, pillar.fixedDayCounter, pillar.businessDayConvention,
                      pillar.paymentLag, zeroDayCounter);
}

/// Par rate of a fixed-vs-floating IRS: the floating forwards come from
/// `forecast`, every discount factor and the fixed annuity from `discount`
/// (exogenous collateral discounting).
template <typename ForecastT, typename DiscountT>
    requires CurveProvider<ForecastT, double> && CurveProvider<DiscountT, double>
inline double impliedIrsRate(const ForecastT& forecast, const DiscountT& discount,
                             const IrsPillar& pillar, const datetime::Date& referenceDate,
                             const datetime::DayCounter& zeroDayCounter) {
    const datetime::Date effective = pillar.start.serial() != 0 ? pillar.start : referenceDate;
    const datetime::Schedule floatSchedule(effective, pillar.maturity, pillar.floatTenor,
                                           pillar.floatCalendar, pillar.businessDayConvention,
                                           datetime::DateGeneration::Forward, false,
                                           datetime::BusinessDayConvention::Unadjusted);
    const std::vector<datetime::Date>& dates = floatSchedule.dates();
    double floatPv = 0.0;
    for (std::size_t k = 1; k < dates.size(); ++k) {
        const double tau = datetime::yearFraction(dates[k - 1], dates[k], pillar.floatDayCounter);
        if (!(tau > 0.0)) {
            throw std::invalid_argument("impliedIrsRate: non-positive float accrual");
        }
        const datetime::Date payDate = pillar.floatCalendar.advance(
            dates[k], datetime::Period(pillar.paymentLag, datetime::TimeUnit::Days),
            pillar.businessDayConvention);
        const double tPay = datetime::yearFraction(referenceDate, payDate, zeroDayCounter);
        const double discountPay = discount.discount(tPay);
        if (k == 1 && pillar.firstCouponFixed) {
            floatPv += tau * discountPay * pillar.firstCouponRate;
            continue;
        }
        const double tPrevious =
            datetime::yearFraction(referenceDate, dates[k - 1], zeroDayCounter);
        const double tAccrual = datetime::yearFraction(referenceDate, dates[k], zeroDayCounter);
        if (!(tPrevious >= 0.0) || !(tAccrual > 0.0)) {
            throw std::invalid_argument(
                "impliedIrsRate: coupons before the reference date must be fixed");
        }
        const double forward =
            (forecast.discount(tPrevious) / forecast.discount(tAccrual) - 1.0) / tau;
        floatPv += tau * discountPay * forward;
    }
    const double annuity = fixedAnnuity(discount, pillar, referenceDate, zeroDayCounter);
    if (!(annuity > 0.0)) {
        throw std::invalid_argument("impliedIrsRate: non-positive fixed annuity");
    }
    return floatPv / annuity;
}

/// Two-IRS convention: the basis spread is the difference of two par IRS
/// rates sharing the same fixed schedule.
inline double basisSpreadAsIrsRateDifference(double firstIrsRate, double secondIrsRate) {
    return firstIrsRate - secondIrsRate;
}

/// Convert a single-IRS basis spread into the two-IRS convention:
/// `Delta_twoIrs = Delta_singleIrs * annuitySecondLeg / annuityFixedSchedule`.
inline double basisSpreadConventionSwitch(double singleSpread, double secondLegAnnuity,
                                          double fixedScheduleAnnuity) {
    if (!(fixedScheduleAnnuity > 0.0)) {
        throw std::invalid_argument(
            "basisSpreadConventionSwitch: non-positive fixed schedule annuity");
    }
    if (!(secondLegAnnuity > 0.0)) {
        throw std::invalid_argument("basisSpreadConventionSwitch: non-positive second leg annuity");
    }
    return singleSpread * secondLegAnnuity / fixedScheduleAnnuity;
}

/// Implied par basis spread of `pillar` against a child spread curve: the
/// parent leg forecasts on `parentForecast` (the child's own parent by
/// default), discounting on `discountCurve` when given (exogenous OIS).
template <typename ParentT, typename ParentForecastT, typename DiscountT = DiscountCurve<double>>
    requires CurveNodeProvider<ParentT> && CurveProvider<ParentForecastT, double> &&
             CurveProvider<DiscountT, double>
inline double impliedBasisSpread(const SpreadCurve<double, ParentT>& child,
                                 const ParentForecastT& parentForecast, const BasisPillar& pillar,
                                 const datetime::Date& referenceDate,
                                 const datetime::DayCounter& zeroDayCounter,
                                 const DiscountT* discountCurve = nullptr) {
    const datetime::Schedule schedule(referenceDate, pillar.maturity, pillar.floatTenor,
                                      pillar.calendar, pillar.businessDayConvention,
                                      datetime::DateGeneration::Forward, false,
                                      datetime::BusinessDayConvention::Unadjusted);
    const std::vector<datetime::Date>& dates = schedule.dates();
    const auto discountAt = [&](double t) -> double {
        return discountCurve != nullptr ? discountCurve->discount(t) : child.parent().discount(t);
    };
    double weightedChild = 0.0;
    double weightedParent = 0.0;
    double annuity = 0.0;
    for (std::size_t k = 1; k < dates.size(); ++k) {
        const double tau = datetime::yearFraction(dates[k - 1], dates[k], pillar.quoteDayCounter);
        if (!(tau > 0.0)) {
            throw std::invalid_argument("impliedBasisSpread: non-positive float accrual");
        }
        const double tPrev = datetime::yearFraction(referenceDate, dates[k - 1], zeroDayCounter);
        const double t = datetime::yearFraction(referenceDate, dates[k], zeroDayCounter);
        if (!(tPrev >= 0.0) || !(t > 0.0)) {
            throw std::invalid_argument(
                "impliedBasisSpread: coupons before the reference date must be fixed");
        }
        const double df = discountAt(t);
        const double forwardChild = (child.discount(tPrev) / child.discount(t) - 1.0) / tau;
        const double forwardParent =
            (parentForecast.discount(tPrev) / parentForecast.discount(t) - 1.0) / tau;
        weightedChild += tau * df * forwardChild;
        weightedParent += tau * df * forwardParent;
        annuity += tau * df;
    }
    if (!(annuity > 0.0)) {
        throw std::invalid_argument("impliedBasisSpread: non-positive annuity");
    }
    const double level = (weightedChild - weightedParent) / annuity;
    return pillar.spreadOnParentLeg ? level : -level;
}

/// Overload using the child's own parent for the parent forecast.
template <typename ParentT, typename DiscountT = DiscountCurve<double>>
    requires CurveNodeProvider<ParentT> && CurveProvider<DiscountT, double>
inline double impliedBasisSpread(const SpreadCurve<double, ParentT>& child,
                                 const BasisPillar& pillar, const datetime::Date& referenceDate,
                                 const datetime::DayCounter& zeroDayCounter,
                                 const DiscountT* discountCurve = nullptr) {
    return impliedBasisSpread(child, child.parent(), pillar, referenceDate, zeroDayCounter,
                              discountCurve);
}

/// One instrument on a forecast curve: a float-vs-float basis swap with a
/// spread, a par fixed-vs-float IRS, a synthetic money-market quote
/// (spot-starting deposit or forward-starting FRA) whose simple forward over
/// its accrual period is the quote, or an exchange-traded future with the same
/// style/convexity semantics as the discount-curve future.
struct ForecastPillar {
    enum class Kind : std::uint8_t { BasisSwap, Irs, Deposit, Fra, Future };
    Kind kind = Kind::BasisSwap;
    BasisPillar basis; ///< Used when `kind == BasisSwap`
    IrsPillar irs;     ///< Used when `kind == Irs`
    // Simple money-market quote fields, used when `kind == Deposit`,
    // `kind == Fra` or `kind == Future`; an unset `start` means the reference
    // date for Deposit/Fra. Futures carry exchange dates directly.
    datetime::Date start;
    datetime::Date maturity;
    double quote = 0.0;
    double convexityAdjustment = 0.0;              ///< Futures: added to the fitted forward rate
    FutureStyle futureStyle = FutureStyle::Simple; ///< Futures: underlying style
    AveragingStyle averagingStyle = AveragingStyle::Arithmetic; ///< Averaged futures convention
    datetime::DayCounter quoteDayCounter{datetime::DayCount::Actual360};
    datetime::Calendar calendar{};
    datetime::BusinessDayConvention businessDayConvention =
        datetime::BusinessDayConvention::ModifiedFollowing;
    /// Direct turn knot: the quote is the funding-window rate and the pillar
    /// reports as turn risk rather than a smooth forecast pillar. It stays an
    /// ordinary bootstrap quote in the same Jacobian and column order.
    bool turnPillar = false;
};

/// Quoted maturity of a forecast pillar's kind, before any calendar roll.
/// Throws when the kind's target maturity is unset: a default (zero-serial)
/// date never comes from a market quote.
inline datetime::Date forecastPillarQuotedMaturity(const ForecastPillar& pillar) {
    switch (pillar.kind) {
        case ForecastPillar::Kind::Irs:
            if (pillar.irs.maturity.serial() == 0) {
                throw std::invalid_argument(
                    "forecastPillarQuotedMaturity: IRS pillar is missing its maturity");
            }
            return pillar.irs.maturity;
        case ForecastPillar::Kind::BasisSwap:
            if (pillar.basis.maturity.serial() == 0) {
                throw std::invalid_argument(
                    "forecastPillarQuotedMaturity: basis pillar is missing its maturity");
            }
            return pillar.basis.maturity;
        case ForecastPillar::Kind::Deposit:
        case ForecastPillar::Kind::Fra:
        case ForecastPillar::Kind::Future:
            if (pillar.maturity.serial() == 0) {
                throw std::invalid_argument(
                    "forecastPillarQuotedMaturity: money-market pillar is missing its maturity");
            }
            return pillar.maturity;
    }
    throw std::invalid_argument("forecastPillarQuotedMaturity: unknown forecast pillar kind");
}

/// Maturity date a forecast pillar actually references in pricing and at its
/// bootstrap node: swap termination dates and money-market quotes roll on their
/// calendar and business-day convention, exchange futures use the quoted end
/// date directly.
inline datetime::Date forecastPillarRiskMaturity(const ForecastPillar& pillar) {
    const datetime::Date quoted = forecastPillarQuotedMaturity(pillar);
    switch (pillar.kind) {
        case ForecastPillar::Kind::Irs:
            return pillar.irs.fixedCalendar.adjust(quoted, pillar.irs.businessDayConvention);
        case ForecastPillar::Kind::Deposit:
        case ForecastPillar::Kind::Fra:
            return pillar.calendar.adjust(quoted, pillar.businessDayConvention);
        case ForecastPillar::Kind::Future:
            return quoted;
        case ForecastPillar::Kind::BasisSwap:
            return pillar.basis.calendar.adjust(quoted, pillar.basis.businessDayConvention);
    }
    throw std::invalid_argument("forecastPillarRiskMaturity: unknown forecast pillar kind");
}

/// Quote a forecast pillar targets in the bootstrap, drawn from the field set
/// of its kind: the IRS par rate, the basis spread, or the money-market /
/// futures quote. Throws when the kind's target fields (maturity, quote,
/// tenors) are unset, so a pillar filled with the wrong field set cannot
/// bootstrap to a zero target.
inline double forecastPillarTarget(const ForecastPillar& pillar) {
    switch (pillar.kind) {
        case ForecastPillar::Kind::Irs:
            (void)forecastPillarQuotedMaturity(pillar);
            if (!(pillar.irs.floatTenor.length() > 0) || !(pillar.irs.fixedTenor.length() > 0)) {
                throw std::invalid_argument(
                    "forecastPillarTarget: IRS pillar is missing a leg tenor");
            }
            if (pillar.irs.quote == 0.0 && pillar.quote != 0.0) {
                throw std::invalid_argument(
                    "forecastPillarTarget: quote is set on the top-level field of an IRS pillar");
            }
            return pillar.irs.quote;
        case ForecastPillar::Kind::BasisSwap:
            (void)forecastPillarQuotedMaturity(pillar);
            if (!(pillar.basis.floatTenor.length() > 0)) {
                throw std::invalid_argument(
                    "forecastPillarTarget: basis pillar is missing its float tenor");
            }
            if (pillar.basis.spread == 0.0 && pillar.quote != 0.0) {
                throw std::invalid_argument(
                    "forecastPillarTarget: quote is set on the top-level field of a basis pillar");
            }
            return pillar.basis.spread;
        case ForecastPillar::Kind::Deposit:
        case ForecastPillar::Kind::Fra:
        case ForecastPillar::Kind::Future:
            (void)forecastPillarQuotedMaturity(pillar);
            return pillar.quote;
    }
    throw std::invalid_argument("forecastPillarTarget: unknown forecast pillar kind");
}

/// Short label of a forecast pillar kind.
constexpr std::string_view forecastPillarKindName(ForecastPillar::Kind kind) noexcept {
    switch (kind) {
        case ForecastPillar::Kind::BasisSwap:
            return "Basis";
        case ForecastPillar::Kind::Irs:
            return "Irs";
        case ForecastPillar::Kind::Deposit:
            return "Deposit";
        case ForecastPillar::Kind::Fra:
            return "Fra";
        case ForecastPillar::Kind::Future:
            return "Future";
    }
    return "Unknown";
}

namespace detail {

/// Precomputed money-market quote times, aliased from the scalar math header.
using SimpleForwardQuoteTimes = pricing::SimpleForwardQuoteTimes;
using FutureQuoteTimes = pricing::FutureQuoteTimes;

/// Everything a discount-curve bootstrap residual needs for one pillar, kept
/// as a thin forwarder to the scalar algebra for the frozen legacy oracle.
/// The schedule and date arithmetic are resolved once when the bootstrap
/// starts; each root-find step only evaluates the trial curve at the times.
struct PillarQuoteTimes {
    PillarKind kind = PillarKind::Deposit;
    pricing::SimpleForwardQuoteTimes deposit;
    pricing::SimpleForwardQuoteTimes fra;
    double fraConvexityExponent = 0.0;
    pricing::FutureQuoteTimes future;
    std::vector<pricing::OisCouponTimes> oisCoupons;
};

/// Precomputed quote inputs of one IRS float coupon.
struct IrsFloatCouponTimes {
    double tau = 0.0;
    double tPay = 0.0;
    double tPrevious = 0.0;
    double tAccrual = 0.0;
    bool firstFixed = false;
    double firstRate = 0.0;
};

/// Precomputed quote inputs of one fixed-leg annuity coupon.
struct IrsAnnuityCouponTimes {
    double tau = 0.0;
    double tPay = 0.0;
};

/// Precomputed quote inputs of one basis-swap coupon.
struct BasisCouponTimes {
    double tau = 0.0;
    double tPrev = 0.0;
    double t = 0.0;
};

/// Everything a forecast-curve bootstrap residual needs for one pillar.
struct ForecastQuoteTimes {
    ForecastPillar::Kind kind = ForecastPillar::Kind::Deposit;
    SimpleForwardQuoteTimes simple;
    FutureQuoteTimes future;
    std::vector<IrsFloatCouponTimes> floatCoupons;
    std::vector<IrsAnnuityCouponTimes> annuityCoupons;
    std::vector<BasisCouponTimes> basisCoupons;
    bool spreadOnParentLeg = true;
};

inline PillarQuoteTimes makePillarQuoteTimes(const CurvePillar& pillar,
                                             const datetime::Date& referenceDate,
                                             const datetime::DayCounter& zeroDayCounter) {
    PillarQuoteTimes times;
    times.kind = pillar.kind;
    switch (pillar.kind) {
        case PillarKind::Repo:
        case PillarKind::Deposit:
            times.deposit = pricing::prepareDepositTimes(adjustedMaturity(pillar), referenceDate,
                                                         pillar.quoteDayCounter, zeroDayCounter);
            break;
        case PillarKind::Fra:
            times.fra =
                pricing::prepareFraTimes(adjustedStart(pillar), adjustedMaturity(pillar),
                                         referenceDate, pillar.quoteDayCounter, zeroDayCounter);
            times.fraConvexityExponent = pillar.fraConvexityExponent;
            break;
        case PillarKind::Future:
            times.future = pricing::prepareFutureTimes(
                pillar.start, pillar.maturity, pillar.calendar, pillar.quoteDayCounter,
                pillar.futureStyle, pillar.averagingStyle, pillar.convexityAdjustment,
                referenceDate, zeroDayCounter, "impliedQuote");
            break;
        case PillarKind::OisSwap:
            times.oisCoupons = pricing::prepareOisCouponTimes(
                pillar.start.serial() != 0 ? pillar.start : referenceDate, pillar.maturity,
                pillar.fixedTenor, pillar.calendar, pillar.businessDayConvention, pillar.paymentLag,
                pillar.quoteDayCounter, pillar.firstCouponFixed, pillar.firstCouponRate,
                referenceDate, zeroDayCounter);
            break;
    }
    return times;
}

/// Discount-curve quote from precomputed times: a thin forwarder to the scalar
/// algebra, kept for the frozen old-vs-new oracle.
template <typename DoubleT>
inline DoubleT evaluatePillarQuote(const PillarQuoteTimes& times,
                                   const DiscountCurve<DoubleT>& curve) {
    switch (times.kind) {
        case PillarKind::Repo:
            return pricing::impliedRepoQuote<DoubleT>(times.deposit, curve);
        case PillarKind::Deposit:
            return pricing::impliedDepositQuote<DoubleT>(times.deposit, curve);
        case PillarKind::Fra:
            return pricing::impliedFraQuote<DoubleT>(times.fra, times.fraConvexityExponent, curve);
        case PillarKind::Future:
            return pricing::impliedFutureQuote<DoubleT>(times.future, curve);
        case PillarKind::OisSwap:
            return pricing::impliedOisParRate<DoubleT>(times.oisCoupons, curve);
    }
    throw std::invalid_argument("impliedQuote: unknown pillar kind");
}

inline SimpleForwardQuoteTimes
makeSimpleForwardQuoteTimes(const ForecastPillar& pillar, const datetime::Date& referenceDate,
                            const datetime::DayCounter& zeroDayCounter) {
    const datetime::Date start = pillar.start.serial() != 0 ? pillar.start : referenceDate;
    const datetime::Date maturity =
        pillar.calendar.adjust(pillar.maturity, pillar.businessDayConvention);
    return pricing::prepareSimpleForwardTimes(start, maturity, referenceDate,
                                              pillar.quoteDayCounter, zeroDayCounter);
}

inline std::vector<IrsFloatCouponTimes>
makeIrsFloatCouponTimes(const IrsPillar& pillar, const datetime::Date& referenceDate,
                        const datetime::DayCounter& zeroDayCounter) {
    const datetime::Date effective = pillar.start.serial() != 0 ? pillar.start : referenceDate;
    const datetime::Schedule floatSchedule(effective, pillar.maturity, pillar.floatTenor,
                                           pillar.floatCalendar, pillar.businessDayConvention,
                                           datetime::DateGeneration::Forward, false,
                                           datetime::BusinessDayConvention::Unadjusted);
    const std::vector<datetime::Date>& dates = floatSchedule.dates();
    std::vector<IrsFloatCouponTimes> coupons;
    coupons.reserve(dates.size() - 1);
    for (std::size_t k = 1; k < dates.size(); ++k) {
        IrsFloatCouponTimes coupon;
        coupon.tau = datetime::yearFraction(dates[k - 1], dates[k], pillar.floatDayCounter);
        if (!(coupon.tau > 0.0)) {
            throw std::invalid_argument("impliedIrsRate: non-positive float accrual");
        }
        const datetime::Date payDate = pillar.floatCalendar.advance(
            dates[k], datetime::Period(pillar.paymentLag, datetime::TimeUnit::Days),
            pillar.businessDayConvention);
        coupon.tPay = datetime::yearFraction(referenceDate, payDate, zeroDayCounter);
        if (k == 1 && pillar.firstCouponFixed) {
            coupon.firstFixed = true;
            coupon.firstRate = pillar.firstCouponRate;
        } else {
            coupon.tPrevious = datetime::yearFraction(referenceDate, dates[k - 1], zeroDayCounter);
            coupon.tAccrual = datetime::yearFraction(referenceDate, dates[k], zeroDayCounter);
            if (!(coupon.tPrevious >= 0.0) || !(coupon.tAccrual > 0.0)) {
                throw std::invalid_argument(
                    "impliedIrsRate: coupons before the reference date must be fixed");
            }
        }
        coupons.push_back(coupon);
    }
    return coupons;
}

inline std::vector<IrsAnnuityCouponTimes>
makeIrsAnnuityCouponTimes(const IrsPillar& pillar, const datetime::Date& referenceDate,
                          const datetime::DayCounter& zeroDayCounter) {
    const datetime::Date effective = pillar.start.serial() != 0 ? pillar.start : referenceDate;
    const datetime::Schedule schedule(effective, pillar.maturity, pillar.fixedTenor,
                                      pillar.fixedCalendar, pillar.businessDayConvention,
                                      datetime::DateGeneration::Forward, false,
                                      datetime::BusinessDayConvention::Unadjusted);
    const std::vector<datetime::Date>& dates = schedule.dates();
    std::vector<IrsAnnuityCouponTimes> coupons;
    coupons.reserve(dates.size() - 1);
    for (std::size_t k = 1; k < dates.size(); ++k) {
        IrsAnnuityCouponTimes coupon;
        coupon.tau = datetime::yearFraction(dates[k - 1], dates[k], pillar.fixedDayCounter);
        const datetime::Date payDate = pillar.fixedCalendar.advance(
            dates[k], datetime::Period(pillar.paymentLag, datetime::TimeUnit::Days),
            pillar.businessDayConvention);
        coupon.tPay = datetime::yearFraction(referenceDate, payDate, zeroDayCounter);
        coupons.push_back(coupon);
    }
    return coupons;
}

inline std::vector<BasisCouponTimes>
makeBasisCouponTimes(const BasisPillar& pillar, const datetime::Date& referenceDate,
                     const datetime::DayCounter& zeroDayCounter) {
    const datetime::Schedule schedule(referenceDate, pillar.maturity, pillar.floatTenor,
                                      pillar.calendar, pillar.businessDayConvention,
                                      datetime::DateGeneration::Forward, false,
                                      datetime::BusinessDayConvention::Unadjusted);
    const std::vector<datetime::Date>& dates = schedule.dates();
    std::vector<BasisCouponTimes> coupons;
    coupons.reserve(dates.size() - 1);
    for (std::size_t k = 1; k < dates.size(); ++k) {
        BasisCouponTimes coupon;
        coupon.tau = datetime::yearFraction(dates[k - 1], dates[k], pillar.quoteDayCounter);
        if (!(coupon.tau > 0.0)) {
            throw std::invalid_argument("impliedBasisSpread: non-positive float accrual");
        }
        coupon.tPrev = datetime::yearFraction(referenceDate, dates[k - 1], zeroDayCounter);
        coupon.t = datetime::yearFraction(referenceDate, dates[k], zeroDayCounter);
        if (!(coupon.tPrev >= 0.0) || !(coupon.t > 0.0)) {
            throw std::invalid_argument(
                "impliedBasisSpread: coupons before the reference date must be fixed");
        }
        coupons.push_back(coupon);
    }
    return coupons;
}

inline ForecastQuoteTimes makeForecastQuoteTimes(const ForecastPillar& pillar,
                                                 const datetime::Date& referenceDate,
                                                 const datetime::DayCounter& zeroDayCounter) {
    ForecastQuoteTimes times;
    times.kind = pillar.kind;
    switch (pillar.kind) {
        case ForecastPillar::Kind::Deposit:
        case ForecastPillar::Kind::Fra:
            times.simple = makeSimpleForwardQuoteTimes(pillar, referenceDate, zeroDayCounter);
            break;
        case ForecastPillar::Kind::Future:
            times.future = pricing::prepareFutureTimes(
                pillar.start, pillar.maturity, pillar.calendar, pillar.quoteDayCounter,
                pillar.futureStyle, pillar.averagingStyle, pillar.convexityAdjustment,
                referenceDate, zeroDayCounter, "impliedForecastFuture");
            break;
        case ForecastPillar::Kind::Irs:
            times.floatCoupons = makeIrsFloatCouponTimes(pillar.irs, referenceDate, zeroDayCounter);
            times.annuityCoupons =
                makeIrsAnnuityCouponTimes(pillar.irs, referenceDate, zeroDayCounter);
            break;
        case ForecastPillar::Kind::BasisSwap:
            times.basisCoupons = makeBasisCouponTimes(pillar.basis, referenceDate, zeroDayCounter);
            times.spreadOnParentLeg = pillar.basis.spreadOnParentLeg;
            break;
    }
    return times;
}

/// Forecast-curve quote from precomputed times; the arithmetic mirrors
/// `impliedForecastQuote` and its swap helpers exactly.
template <typename ForecastT, typename DiscountT>
inline double evaluateForecastQuote(const ForecastQuoteTimes& times, const ForecastT& forecast,
                                    const DiscountT& discounting) {
    switch (times.kind) {
        case ForecastPillar::Kind::Deposit:
        case ForecastPillar::Kind::Fra: {
            const SimpleForwardQuoteTimes& simple = times.simple;
            return (forecast.discount(simple.t1) / forecast.discount(simple.t2) - 1.0) / simple.tau;
        }
        case ForecastPillar::Kind::Future: {
            const FutureQuoteTimes& future = times.future;
            if (future.style == FutureQuoteTimes::Style::AveragedCompounded) {
                double accumulated = 1.0;
                double previousDiscount = 0.0;
                bool hasPrevious = false;
                for (std::size_t k = 0; k < future.previousTimes.size(); ++k) {
                    if (!hasPrevious) {
                        previousDiscount = forecast.discount(future.previousTimes[k]);
                    }
                    const double currentDiscount = forecast.discount(future.currentTimes[k]);
                    accumulated *= previousDiscount / currentDiscount;
                    previousDiscount = currentDiscount;
                    hasPrevious = true;
                }
                return (accumulated - 1.0) / future.tau + future.convexity;
            }
            if (future.style == FutureQuoteTimes::Style::AveragedArithmetic) {
                double sum = 0.0;
                double previousDiscount = 0.0;
                bool hasPrevious = false;
                for (std::size_t k = 0; k < future.previousTimes.size(); ++k) {
                    if (!hasPrevious) {
                        previousDiscount = forecast.discount(future.previousTimes[k]);
                    }
                    const double currentDiscount = forecast.discount(future.currentTimes[k]);
                    sum += (previousDiscount / currentDiscount - 1.0) / future.accrualTaus[k];
                    previousDiscount = currentDiscount;
                    hasPrevious = true;
                }
                return sum / static_cast<double>(future.previousTimes.size()) + future.convexity;
            }
            return (forecast.discount(future.t1) / forecast.discount(future.t2) - 1.0) /
                       future.tau +
                   future.convexity;
        }
        case ForecastPillar::Kind::Irs: {
            double floatPv = 0.0;
            double previousAccrualDiscount = 0.0;
            bool hasPrevious = false;
            for (const IrsFloatCouponTimes& coupon : times.floatCoupons) {
                const double discountPay = discounting.discount(coupon.tPay);
                if (coupon.firstFixed) {
                    floatPv += coupon.tau * discountPay * coupon.firstRate;
                    hasPrevious = false;
                    continue;
                }
                const double previousDiscount =
                    hasPrevious ? previousAccrualDiscount : forecast.discount(coupon.tPrevious);
                const double accrualDiscount = forecast.discount(coupon.tAccrual);
                const double forward = (previousDiscount / accrualDiscount - 1.0) / coupon.tau;
                floatPv += coupon.tau * discountPay * forward;
                previousAccrualDiscount = accrualDiscount;
                hasPrevious = true;
            }
            double annuity = 0.0;
            for (const IrsAnnuityCouponTimes& coupon : times.annuityCoupons) {
                annuity += coupon.tau * discounting.discount(coupon.tPay);
            }
            if (!(annuity > 0.0)) {
                throw std::invalid_argument("impliedIrsRate: non-positive fixed annuity");
            }
            return floatPv / annuity;
        }
        case ForecastPillar::Kind::BasisSwap: {
            const auto& parentForecast = forecast.parent();
            double weightedChild = 0.0;
            double weightedParent = 0.0;
            double annuity = 0.0;
            double previousChild = 0.0;
            double previousParent = 0.0;
            bool hasPrevious = false;
            for (const BasisCouponTimes& coupon : times.basisCoupons) {
                const double df = discounting.discount(coupon.t);
                const double childAtPrevious =
                    hasPrevious ? previousChild : forecast.discount(coupon.tPrev);
                const double childAtCoupon = forecast.discount(coupon.t);
                const double parentAtPrevious =
                    hasPrevious ? previousParent : parentForecast.discount(coupon.tPrev);
                const double parentAtCoupon = parentForecast.discount(coupon.t);
                const double forwardChild = (childAtPrevious / childAtCoupon - 1.0) / coupon.tau;
                const double forwardParent = (parentAtPrevious / parentAtCoupon - 1.0) / coupon.tau;
                weightedChild += coupon.tau * df * forwardChild;
                weightedParent += coupon.tau * df * forwardParent;
                annuity += coupon.tau * df;
                previousChild = childAtCoupon;
                previousParent = parentAtCoupon;
                hasPrevious = true;
            }
            if (!(annuity > 0.0)) {
                throw std::invalid_argument("impliedBasisSpread: non-positive annuity");
            }
            const double level = (weightedChild - weightedParent) / annuity;
            return times.spreadOnParentLeg ? level : -level;
        }
    }
    throw std::invalid_argument("impliedForecastQuote: unknown forecast pillar kind");
}

/// Adapter that presents an IRS or basis-swap forecast pillar as a bootstrap
/// instrument; the swap legs still need a double forecast curve.
class ForecastSwapInstrument {
public:
    ForecastSwapInstrument(const ForecastPillar& pillar, const datetime::Date& referenceDate,
                           const datetime::DayCounter& zeroDayCounter)
        : m_date(forecastPillarRiskMaturity(pillar)), m_target(forecastPillarTarget(pillar)),
          m_times(makeForecastQuoteTimes(pillar, referenceDate, zeroDayCounter)) {}

    datetime::Date date() const { return m_date; }
    double target() const { return m_target; }

    template <typename ScalarT, typename CurveSetT>
    ScalarT impliedQuote(const CurveSetT& curves) const {
        if constexpr (std::is_same_v<ScalarT, double>) {
            return evaluateForecastQuote(m_times, curves.forecastCurve, curves.discountCurve);
        } else {
            (void)curves;
            throw std::invalid_argument(
                "impliedForecastQuote: basis and IRS pillars need a double forecast curve");
        }
    }

private:
    datetime::Date m_date;
    double m_target = 0.0;
    ForecastQuoteTimes m_times;
};

/// Forecast-curve bootstrap instrument: money-market pillars evaluate through
/// the plain-data instruments; IRS and basis pillars keep the swap adapter.
using ForecastInstrument =
    Ladder<instruments::Deposit, instruments::Fra, instruments::Future, ForecastSwapInstrument>;

/// Adapter from the tagged `ForecastPillar` POD to the concept instrument:
/// quote times are prepared once per instrument with the caller's zero clock.
inline ForecastInstrument forecastToInstrument(const ForecastPillar& pillar,
                                               const datetime::Date& referenceDate,
                                               const datetime::DayCounter& zeroDayCounter) {
    using Instrument = ForecastInstrument::Instrument;
    switch (pillar.kind) {
        case ForecastPillar::Kind::Deposit: {
            instruments::Deposit out;
            out.maturity = pillar.maturity;
            out.calendar = pillar.calendar;
            out.businessDayConvention = pillar.businessDayConvention;
            out.quoteDayCounter = pillar.quoteDayCounter;
            out.quote = pillar.quote;
            out.prepareForecast(pillar.start.serial() != 0 ? pillar.start : referenceDate,
                                referenceDate, zeroDayCounter);
            return ForecastInstrument{Instrument{std::move(out)}};
        }
        case ForecastPillar::Kind::Fra: {
            instruments::Fra out;
            out.start = pillar.start;
            out.maturity = pillar.maturity;
            out.calendar = pillar.calendar;
            out.businessDayConvention = pillar.businessDayConvention;
            out.quoteDayCounter = pillar.quoteDayCounter;
            out.quote = pillar.quote;
            out.prepareForecast(pillar.start.serial() != 0 ? pillar.start : referenceDate,
                                referenceDate, zeroDayCounter);
            return ForecastInstrument{Instrument{std::move(out)}};
        }
        case ForecastPillar::Kind::Future: {
            instruments::Future out;
            out.start = pillar.start;
            out.maturity = pillar.maturity;
            out.calendar = pillar.calendar;
            out.quoteDayCounter = pillar.quoteDayCounter;
            out.futureStyle = pillar.futureStyle;
            out.averagingStyle = pillar.averagingStyle;
            out.convexityAdjustment = pillar.convexityAdjustment;
            out.quote = pillar.quote;
            out.prepare(referenceDate, zeroDayCounter, "impliedForecastFuture");
            return ForecastInstrument{Instrument{std::move(out)}};
        }
        case ForecastPillar::Kind::Irs:
        case ForecastPillar::Kind::BasisSwap:
            return ForecastInstrument{
                Instrument{ForecastSwapInstrument(pillar, referenceDate, zeroDayCounter)}};
    }
    throw std::invalid_argument("impliedForecastQuote: unknown forecast pillar kind");
}

} // namespace detail

/// Simple forward of the forecast curve over the pillar's accrual period:
/// `r = (D_f(t1) / D_f(t2) - 1) / tau` with zero times measured from the
/// reference date.
template <typename DoubleT, typename ParentT>
    requires CurveProvider<SpreadCurve<DoubleT, ParentT>, DoubleT>
inline DoubleT impliedSimpleForward(const SpreadCurve<DoubleT, ParentT>& forecast,
                                    const ForecastPillar& pillar,
                                    const datetime::Date& referenceDate,
                                    const datetime::DayCounter& zeroDayCounter) {
    const datetime::Date start = pillar.start.serial() != 0 ? pillar.start : referenceDate;
    const datetime::Date maturity =
        pillar.calendar.adjust(pillar.maturity, pillar.businessDayConvention);
    const pricing::SimpleForwardQuoteTimes times = pricing::prepareSimpleForwardTimes(
        start, maturity, referenceDate, pillar.quoteDayCounter, zeroDayCounter);
    return pricing::impliedSimpleForwardQuote<DoubleT>(times, forecast);
}

/// Model quote of an exchange-traded future on a forecast curve: the forward
/// over the quoted reference period plus the stored convexity adjustment,
/// mirroring the discount-curve future semantics. `Simple` and `Compounded`
/// are the period forward; `Averaged` is the arithmetic mean of the daily
/// simple overnight forwards on the business-day fixing grid or the explicit
/// fixing-grid compounded product, per `averagingStyle`. Exchange fixings and
/// accrual ends are quoted dates, not rolled on a calendar.
template <typename DoubleT, typename ParentT>
    requires CurveProvider<SpreadCurve<DoubleT, ParentT>, DoubleT>
inline DoubleT impliedForecastFuture(const SpreadCurve<DoubleT, ParentT>& forecast,
                                     const ForecastPillar& pillar,
                                     const datetime::Date& referenceDate,
                                     const datetime::DayCounter& zeroDayCounter) {
    instruments::Future instrument;
    instrument.start = pillar.start;
    instrument.maturity = pillar.maturity;
    instrument.calendar = pillar.calendar;
    instrument.quoteDayCounter = pillar.quoteDayCounter;
    instrument.futureStyle = pillar.futureStyle;
    instrument.averagingStyle = pillar.averagingStyle;
    instrument.convexityAdjustment = pillar.convexityAdjustment;
    instrument.prepare(referenceDate, zeroDayCounter, "impliedForecastFuture");
    return instrument.template impliedQuote<DoubleT>(
        DiscountSet<SpreadCurve<DoubleT, ParentT>>{forecast});
}

/// Model quote of a swap-style forecast instrument: the basis spread or the
/// par IRS rate, whichever the pillar targets.
template <typename ParentT, typename DiscountT>
    requires CurveNodeProvider<ParentT> && CurveProvider<DiscountT, double>
inline double impliedForecastSwapQuote(const SpreadCurve<double, ParentT>& forecast,
                                       const DiscountT& discounting, const ForecastPillar& pillar,
                                       const datetime::Date& referenceDate,
                                       const datetime::DayCounter& zeroDayCounter) {
    switch (pillar.kind) {
        case ForecastPillar::Kind::Irs:
            return impliedIrsRate(forecast, discounting, pillar.irs, referenceDate, zeroDayCounter);
        case ForecastPillar::Kind::BasisSwap:
            return impliedBasisSpread(forecast, pillar.basis, referenceDate, zeroDayCounter,
                                      &discounting);
        case ForecastPillar::Kind::Deposit:
        case ForecastPillar::Kind::Fra:
        case ForecastPillar::Kind::Future:
            break;
    }
    throw std::invalid_argument(
        "impliedForecastSwapQuote: deposit, FRA and future pillars are not swap quotes");
}

/// Model quote of a forecast-curve instrument: the simple forward (deposit or
/// FRA), the basis spread or the par IRS rate, whichever the pillar targets.
/// The simple branches are scalar generic and drive the reverse-mode gradient;
/// the swap branches require `double` curves.
template <typename DoubleT, typename ParentT, typename DiscountT>
    requires CurveProvider<SpreadCurve<DoubleT, ParentT>, DoubleT> &&
             CurveProvider<DiscountT, DoubleT>
inline DoubleT impliedForecastQuote(const SpreadCurve<DoubleT, ParentT>& forecast,
                                    const DiscountT& discounting, const ForecastPillar& pillar,
                                    const datetime::Date& referenceDate,
                                    const datetime::DayCounter& zeroDayCounter) {
    switch (pillar.kind) {
        case ForecastPillar::Kind::Deposit:
        case ForecastPillar::Kind::Fra:
            return impliedSimpleForward(forecast, pillar, referenceDate, zeroDayCounter);
        case ForecastPillar::Kind::Future:
            return impliedForecastFuture(forecast, pillar, referenceDate, zeroDayCounter);
        case ForecastPillar::Kind::BasisSwap:
        case ForecastPillar::Kind::Irs:
            if constexpr (std::is_same_v<DoubleT, double>) {
                return impliedForecastSwapQuote(forecast, discounting, pillar, referenceDate,
                                                zeroDayCounter);
            } else {
                throw std::invalid_argument(
                    "impliedForecastQuote: basis and IRS pillars need a double forecast curve");
            }
    }
    throw std::invalid_argument("impliedForecastQuote: unknown forecast pillar kind");
}

/**
 * @brief Sequential exact-fit bootstrap of a spread curve over a parent.
 *
 * Each pillar solves one new spread node so the implied par basis matches the
 * quote; the parent is frozen (exogenous discounting). Pillars are solved in
 * maturity order; off-node cashflows and stencil schemes escalate to the
 * whole-grid fixed point, as in the discount bootstrap.
 */
template <typename ParentT>
    requires CurveNodeProvider<ParentT>
inline SpreadCurve<double, ParentT> bootstrapForecastCurve(
    std::shared_ptr<const ParentT> parent, const DiscountCurve<double>* discountCurve,
    const datetime::Date& referenceDate, const datetime::DayCounter& zeroDayCounter,
    InterpolationScheme scheme, const std::vector<ForecastPillar>& pillars, double accuracy = 1e-14,
    double tension = 0.0) {
    if (!parent) {
        throw std::invalid_argument("bootstrapForecastCurve: null parent");
    }
    if (pillars.empty()) {
        throw std::invalid_argument("bootstrapForecastCurve: no pillars");
    }
    const std::size_t count = pillars.size();
    const auto nodeDate = [](const ForecastPillar& pillar) {
        return forecastPillarRiskMaturity(pillar);
    };
    std::vector<double> nodeTimes(count);
    for (std::size_t i = 0; i < count; ++i) {
        nodeTimes[i] = datetime::yearFraction(referenceDate, nodeDate(pillars[i]), zeroDayCounter);
        if (!(nodeTimes[i] > (i == 0 ? 0.0 : nodeTimes[i - 1]))) {
            throw std::invalid_argument(
                "bootstrapForecastCurve: maturities must be strictly increasing");
        }
    }
    std::vector<detail::ForecastInstrument> instruments;
    instruments.reserve(count);
    for (std::size_t i = 0; i < count; ++i) {
        instruments.push_back(
            detail::forecastToInstrument(pillars[i], referenceDate, zeroDayCounter));
    }
    const auto quoteTrial = [&](const auto& trial, std::size_t i) {
        using ForecastT = std::decay_t<decltype(trial)>;
        if (discountCurve != nullptr) {
            return instruments[i].template impliedQuote<double>(
                ForecastSet<ForecastT, DiscountCurve<double>>{trial, *discountCurve});
        }
        return instruments[i].template impliedQuote<double>(
            ForecastSet<ForecastT, ParentT>{trial, *parent});
    };

    const quantape::math::BrentSolver<double> solver;
    std::vector<double> spreads(count, 0.0);
    std::vector<double> trialTimes;
    std::vector<double> trialSpreads;
    const auto solveNodes = [&](bool multiPass) {
        for (int pass = 0; pass < (multiPass ? 50 : 1); ++pass) {
            const std::vector<double> previous = spreads;
            double lastMove = 0.0;
            for (std::size_t i = 0; i < count; ++i) {
                const std::size_t lastNode = multiPass && pass == 0 ? i : count - 1;
                bool cacheValid = false;
                double cachedX = 0.0;
                double cachedF = 0.0;
                const auto residual = [&](double trialSpread) {
                    if (cacheValid && trialSpread == cachedX) {
                        return cachedF;
                    }
                    trialTimes.assign(1, 0.0);
                    trialSpreads.assign(1, 0.0);
                    if (trialTimes.capacity() < lastNode + 2) {
                        trialTimes.reserve(lastNode + 2);
                        trialSpreads.reserve(lastNode + 2);
                    }
                    for (std::size_t j = 0; j <= lastNode; ++j) {
                        trialTimes.push_back(nodeTimes[j]);
                        trialSpreads.push_back(j == i ? trialSpread : spreads[j]);
                    }
                    const SpreadCurve<double, ParentT> trial(parent, trialTimes, trialSpreads,
                                                             scheme, tension);
                    cachedF = quoteTrial(trial, i) - instruments[i].target();
                    cachedX = trialSpread;
                    cacheValid = true;
                    return cachedF;
                };
                const double guess = i == 0 ? 0.0 : spreads[i - 1];
                double halfWidth = 0.005;
                double lower = guess - halfWidth;
                double upper = guess + halfWidth;
                double fLower = residual(lower);
                double fUpper = residual(upper);
                int widen = 0;
                while (fLower * fUpper > 0.0 && widen < 20) {
                    halfWidth *= 4.0;
                    lower = guess - halfWidth;
                    upper = guess + halfWidth;
                    fLower = residual(lower);
                    fUpper = residual(upper);
                    ++widen;
                }
                if (!(fLower * fUpper <= 0.0) || !std::isfinite(fLower) || !std::isfinite(fUpper)) {
                    throw std::runtime_error("bootstrapForecastCurve: failed to bracket pillar " +
                                             std::to_string(i));
                }
                const double root = solver.solve(residual, accuracy, guess, lower, upper);
                const double move = std::abs(root - previous[i]);
                if (move > lastMove) {
                    lastMove = move;
                }
                spreads[i] = root;
            }
            if (!multiPass || lastMove < 1e-15) {
                break;
            }
        }
    };
    const auto worstResidual = [&]() {
        std::vector<double> times{0.0};
        std::vector<double> values{0.0};
        times.reserve(count + 1);
        values.reserve(count + 1);
        for (std::size_t i = 0; i < count; ++i) {
            times.push_back(nodeTimes[i]);
            values.push_back(spreads[i]);
        }
        const SpreadCurve<double, ParentT> curve(parent, times, values, scheme, tension);
        double worst = 0.0;
        for (std::size_t i = 0; i < count; ++i) {
            const double check = quoteTrial(curve, i) - instruments[i].target();
            if (!std::isfinite(check)) {
                worst = 1e300;
            } else if (std::abs(check) > worst) {
                worst = std::abs(check);
            }
        }
        return worst;
    };
    solveNodes(false);
    if (!(worstResidual() < 1e-9)) {
        solveNodes(true);
    }
    if (!(worstResidual() < 1e-9)) {
        throw std::runtime_error("bootstrapForecastCurve: fixed point did not converge");
    }
    std::vector<double> times{0.0};
    std::vector<double> values{0.0};
    times.reserve(count + 1);
    values.reserve(count + 1);
    for (std::size_t i = 0; i < count; ++i) {
        times.push_back(nodeTimes[i]);
        values.push_back(spreads[i]);
    }
    return SpreadCurve<double, ParentT>(std::move(parent), times, values, scheme, tension);
}

/// Overload taking a mutable parent pointer.
template <typename ParentT>
    requires CurveNodeProvider<ParentT>
inline SpreadCurve<double, ParentT>
bootstrapForecastCurve(std::shared_ptr<ParentT> parent, const DiscountCurve<double>* discountCurve,
                       const datetime::Date& referenceDate,
                       const datetime::DayCounter& zeroDayCounter, InterpolationScheme scheme,
                       const std::vector<ForecastPillar>& pillars, double accuracy = 1e-14,
                       double tension = 0.0) {
    return bootstrapForecastCurve(std::shared_ptr<const ParentT>(std::move(parent)), discountCurve,
                                  referenceDate, zeroDayCounter, scheme, pillars, accuracy,
                                  tension);
}

/// Convenience wrapper over `bootstrapForecastCurve` for basis swaps only.
template <typename ParentT>
    requires CurveNodeProvider<ParentT>
inline SpreadCurve<double, ParentT>
bootstrapSpreadCurve(std::shared_ptr<const ParentT> parent, const datetime::Date& referenceDate,
                     const datetime::DayCounter& zeroDayCounter, InterpolationScheme scheme,
                     const std::vector<BasisPillar>& pillars, double accuracy = 1e-14,
                     double tension = 0.0, const DiscountCurve<double>* discountCurve = nullptr) {
    std::vector<ForecastPillar> forecastPillars;
    forecastPillars.reserve(pillars.size());
    for (const BasisPillar& pillar : pillars) {
        ForecastPillar out;
        out.kind = ForecastPillar::Kind::BasisSwap;
        out.basis = pillar;
        forecastPillars.push_back(out);
    }
    return bootstrapForecastCurve(std::move(parent), discountCurve, referenceDate, zeroDayCounter,
                                  scheme, forecastPillars, accuracy, tension);
}

/// Overload taking a mutable parent pointer.
template <typename ParentT>
    requires CurveNodeProvider<ParentT>
inline SpreadCurve<double, ParentT>
bootstrapSpreadCurve(std::shared_ptr<ParentT> parent, const datetime::Date& referenceDate,
                     const datetime::DayCounter& zeroDayCounter, InterpolationScheme scheme,
                     const std::vector<BasisPillar>& pillars, double accuracy = 1e-14,
                     double tension = 0.0, const DiscountCurve<double>* discountCurve = nullptr) {
    return bootstrapSpreadCurve(std::shared_ptr<const ParentT>(std::move(parent)), referenceDate,
                                zeroDayCounter, scheme, pillars, accuracy, tension, discountCurve);
}

/**
 * @brief Sequential exact-fit bootstrap of a single discount curve.
 *
 * Pillars must be sorted by strictly increasing maturity. Each new zero-rate
 * node is solved with a bracketed Brent root find on the pillar's implied
 * quote, parents/higher nodes being absent by construction. Throws
 * `std::invalid_argument` on malformed input and `std::runtime_error` when a
 * pillar cannot be bracketed or does not converge.
 */
inline DiscountCurve<double>
bootstrapDiscountCurve(const datetime::Date& referenceDate,
                       const datetime::DayCounter& zeroDayCounter, InterpolationSpace space,
                       InterpolationScheme scheme, const std::vector<CurvePillar>& pillars,
                       double accuracy = 1e-14, double tension = 0.0, int switchIndex = 1) {
    if (pillars.empty()) {
        throw std::invalid_argument("bootstrapDiscountCurve: no pillars");
    }
    const std::size_t count = pillars.size();
    std::vector<double> nodeTimes(count);
    std::vector<datetime::Date> maturityDates(count);
    for (std::size_t i = 0; i < count; ++i) {
        const datetime::Date maturity = pillarRiskMaturity(pillars[i]);
        nodeTimes[i] = datetime::yearFraction(referenceDate, maturity, zeroDayCounter);
        if (!(nodeTimes[i] > (i == 0 ? 0.0 : nodeTimes[i - 1]))) {
            throw std::invalid_argument(
                "bootstrapDiscountCurve: maturities must be strictly increasing");
        }
        maturityDates[i] = maturity;
    }
    // The trial curves are constructed from raw times, so their zero clock is
    // the times-constructor default (ACT/365F); the whole-grid check runs on a
    // date-constructed curve and uses the caller's clock. Precomputing both
    // keeps the solve and the check on their original clocks.
    const datetime::DayCounter trialZeroDayCounter(datetime::DayCount::Actual365Fixed);
    std::vector<DiscountInstrument> solveInstruments;
    std::vector<DiscountInstrument> checkInstruments;
    solveInstruments.reserve(count);
    checkInstruments.reserve(count);
    for (std::size_t i = 0; i < count; ++i) {
        solveInstruments.push_back(toInstrument(pillars[i], referenceDate, trialZeroDayCounter));
        checkInstruments.push_back(toInstrument(pillars[i], referenceDate, zeroDayCounter));
    }

    const quantape::math::BrentSolver<double> solver;
    std::vector<double> zeros(count, 0.0);
    std::vector<double> trialTimes;
    std::vector<double> trialZeros;
    const auto solveNodes = [&](bool multiPass) {
        for (int pass = 0; pass < (multiPass ? 50 : 1); ++pass) {
            const std::vector<double> previous = zeros;
            double lastMove = 0.0;
            for (std::size_t i = 0; i < count; ++i) {
                const CurvePillar& pillar = pillars[i];
                const std::size_t lastNode = multiPass && pass == 0 ? i : count - 1;
                bool cacheValid = false;
                double cachedX = 0.0;
                double cachedF = 0.0;
                std::optional<DiscountCurve<double>> trialCurve;
                const auto residual = [&](double trialZero) {
                    if (cacheValid && trialZero == cachedX) {
                        return cachedF;
                    }
                    if (!trialCurve.has_value()) {
                        trialTimes.assign(1, 0.0);
                        trialZeros.assign(1, 0.0);
                        if (trialTimes.capacity() < lastNode + 2) {
                            trialTimes.reserve(lastNode + 2);
                            trialZeros.reserve(lastNode + 2);
                        }
                        for (std::size_t j = 0; j <= lastNode; ++j) {
                            trialTimes.push_back(nodeTimes[j]);
                            trialZeros.push_back(zeros[j]);
                        }
                        trialCurve.emplace(trialTimes, trialZeros, space, scheme, tension,
                                           switchIndex);
                    }
                    detail::CurveTrialUpdater::setNode(*trialCurve, i + 1, trialZero);
                    cachedF = solveInstruments[i].template impliedQuote<double>(
                                  DiscountSet<DiscountCurve<double>>{*trialCurve}) -
                              solveInstruments[i].target();
                    cachedX = trialZero;
                    cacheValid = true;
                    return cachedF;
                };

                const double guess = i == 0 ? 0.0 : zeros[i - 1];
                double lower = guess - 0.5;
                double upper = guess + 0.5;
                double fLower = residual(lower);
                double fUpper = residual(upper);
                int widen = 0;
                while (fLower * fUpper > 0.0 && widen < 12) {
                    lower -= 0.5;
                    upper += 0.5;
                    fLower = residual(lower);
                    fUpper = residual(upper);
                    ++widen;
                }
                if (!(fLower * fUpper <= 0.0) || !std::isfinite(fLower) || !std::isfinite(fUpper)) {
                    throw std::runtime_error("bootstrapDiscountCurve: failed to bracket pillar " +
                                             std::to_string(i) + " (" +
                                             std::string(pillarKindName(pillar.kind)) + ")");
                }
                const double root = solver.solve(residual, accuracy, guess, lower, upper);
                const double move = std::abs(root - previous[i]);
                if (move > lastMove) {
                    lastMove = move;
                }
                zeros[i] = root;
            }
            if (!multiPass || lastMove < 1e-15) {
                break;
            }
        }
    };
    const auto worstResidual = [&]() {
        const DiscountCurve<double> curve(referenceDate, maturityDates, zeroDayCounter, zeros,
                                          space, scheme, tension, switchIndex);
        double worst = 0.0;
        for (std::size_t i = 0; i < count; ++i) {
            const double check = checkInstruments[i].template impliedQuote<double>(
                                     DiscountSet<DiscountCurve<double>>{curve}) -
                                 checkInstruments[i].target();
            if (!std::isfinite(check)) {
                worst = 1e300;
            } else if (std::abs(check) > worst) {
                worst = std::abs(check);
            }
        }
        return worst;
    };
    solveNodes(false);
    if (!(worstResidual() < 1e-9)) {
        // Off-node cashflows under stencil schemes (Akima, mixed, tension) and
        // payment lags need the whole-grid fixed point: pass 0 solves nodes in
        // maturity order, later passes re-solve every node against the full
        // curve until the node moves fall below tolerance.
        solveNodes(true);
    }
    if (!(worstResidual() < 1e-9)) {
        throw std::runtime_error("bootstrapDiscountCurve: fixed point did not converge");
    }
    return DiscountCurve<double>(referenceDate, maturityDates, zeroDayCounter, zeros, space, scheme,
                                 tension, switchIndex);
}

/// Priority filter for overlapping bootstrap instruments: pillars whose
/// maturities are within `overlapDays` calendar days of a cluster's first
/// maturity compete, and the highest-priority kind wins (ties keep the earlier
/// maturity). The default order prefers front-end futures over deposits and
/// FRAs, which come before swaps; the result is maturity-sorted for
/// bootstrapping.
inline std::vector<CurvePillar>
filterOverlappingPillars(const std::vector<CurvePillar>& pillars, int overlapDays = 2,
                         const std::vector<PillarKind>& priority = {
                             PillarKind::Future, PillarKind::Fra, PillarKind::Deposit,
                             PillarKind::Repo, PillarKind::OisSwap}) {
    if (overlapDays < 0) {
        throw std::invalid_argument("filterOverlappingPillars: negative overlap window");
    }
    const auto rank = [&](PillarKind kind) {
        for (std::size_t i = 0; i < priority.size(); ++i) {
            if (priority[i] == kind) {
                return i;
            }
        }
        return priority.size();
    };
    std::vector<std::size_t> order(pillars.size());
    std::iota(order.begin(), order.end(), 0);
    std::stable_sort(order.begin(), order.end(), [&](std::size_t left, std::size_t right) {
        return pillars[left].maturity.serial() < pillars[right].maturity.serial();
    });
    std::vector<CurvePillar> filtered;
    filtered.reserve(pillars.size());
    std::size_t i = 0;
    while (i < order.size()) {
        std::size_t best = order[i];
        std::int32_t clusterEnd = pillars[order[i]].maturity.serial();
        std::size_t j = i + 1;
        while (j < order.size() &&
               pillars[order[j]].maturity.serial() - clusterEnd <= overlapDays) {
            if (rank(pillars[order[j]].kind) < rank(pillars[best].kind)) {
                best = order[j];
            }
            clusterEnd = pillars[order[j]].maturity.serial();
            ++j;
        }
        filtered.push_back(pillars[best]);
        i = j;
    }
    return filtered;
}

} // namespace quantape::markets
