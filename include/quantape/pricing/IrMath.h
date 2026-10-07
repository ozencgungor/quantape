#pragma once

#include "quantape/datetime/Calendar.h"
#include "quantape/datetime/Date.h"
#include "quantape/datetime/DayCounter.h"
#include "quantape/datetime/Period.h"
#include "quantape/datetime/Schedule.h"
#include "quantape/datetime/TimeConversion.h"

#include <cmath>
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace quantape::pricing {
/**
 * @file IrMath.h
 * @brief Scalar-templated money-market quote arithmetic on prepared times
 *
 * This header holds the interest-rate quote algebra with no instrument
 * dependency: schedule arithmetic resolves dates into precomputed time grids
 * once, and the evaluators only read discount factors from a curve provider
 * through `discount(t)`. The same `ScalarT` template serves the `double`
 * bootstrap, reverse-mode `var` gradients and nested `fvar<var>` second-order
 * tests; branching is on convention flags only, never on curve values.
 *
 * Prepared-times preparation throws the exact error text of the original
 * per-instrument entry points, because callers forward those messages
 * verbatim. The instrument wrappers in `pricing/Ir.h` compose this layer with
 * the plain-data instrument types.
 */

/// Rate-future underlying style. `Simple` and `Compounded` are numerically
/// identical on a curve whose overnight index is the curve itself (the
/// compounded rate telescopes to the simple forward); `Averaged` averages the
/// overnight index over the reference period, either arithmetically over the
/// business-day fixing grid (SR1-style daily-rate mean) or compounded
/// (SR3-style: `(D(start)/D(maturity) - 1) / yearFraction`; the compounded rate
/// telescopes, so no fixing grid is needed) per `averagingStyle`.
enum class FutureStyle : std::uint8_t { Simple, Compounded, Averaged };

/// Averaging convention for `FutureStyle::Averaged`: `Arithmetic` is the mean
/// of the daily simple overnight fixings on the business-day grid;
/// `Compounded` is the period compounded overnight rate.
enum class AveragingStyle : std::uint8_t { Arithmetic, Compounded };

/// Consecutive business-day fixing dates over `[effective, maturity]`, used by
/// averaged overnight futures (weekends and holidays are skipped; the fixing
/// preceding a gap carries the multi-day accrual). The grid ends exactly at
/// `maturity`, even when that is not a business day, so the final fixing spans
/// the end of the quoted reference period.
inline std::vector<datetime::Date> businessDayFixings(const datetime::Calendar& calendar,
                                                      const datetime::Date& effective,
                                                      const datetime::Date& maturity) {
    if (maturity < effective) {
        throw std::invalid_argument("businessDayFixings: maturity before effective");
    }
    std::vector<datetime::Date> dates{effective};
    while (dates.back() < maturity) {
        const datetime::Date next =
            calendar.advance(dates.back(), datetime::Period(1, datetime::TimeUnit::Days),
                             datetime::BusinessDayConvention::Following);
        if (!(next > dates.back())) {
            throw std::invalid_argument("businessDayFixings: calendar is not advancing");
        }
        dates.push_back(next >= maturity ? maturity : next);
    }
    return dates;
}

/// Simple forward over a prepared `[t1, t2]` accrual: the inputs are the zero
/// times and the quote day-count year fraction.
struct SimpleForwardQuoteTimes {
    double t1 = 0.0;
    double t2 = 0.0;
    double tau = 0.0;
};

namespace detail {

/// Reads the provided curve for the evaluators: a forecast curve set routes
/// through `forecast(t)`, a discount curve set through `discount(t)`.
template <typename CurveSetT>
struct ForecastDiscountProxy {
    const CurveSetT& curves;

    decltype(auto) discount(double t) const { return curves.forecast(t); }
};

/// Cache key of a generated schedule: the constructor arguments that
/// determine the dates, with the calendar identified by the address of its
/// rule-set name (stable for each rule set, distinct between them).
struct ScheduleKey {
    std::int32_t effective = 0;
    std::int32_t maturity = 0;
    int tenorLength = 0;
    datetime::TimeUnit tenorUnit = datetime::TimeUnit::Days;
    const std::string* calendarName = nullptr;
    datetime::BusinessDayConvention convention = datetime::BusinessDayConvention::Following;
};

/// Per-thread cache of generated forward schedules. Quote-time preparation and
/// the risk-row builders rebuild the same schedule for a pillar on every
/// materialization; the cache keeps one generated schedule per distinct key, so
/// repeated preparations reuse the dates.
class ScheduleCache {
public:
    const datetime::Schedule& get(const datetime::Date& effective, const datetime::Date& maturity,
                                  const datetime::Period& tenor, const datetime::Calendar& calendar,
                                  datetime::BusinessDayConvention convention) {
        // A bootstrap run cycles through a bounded set of schedules, so the cap
        // only trips for a long-lived caller that keeps inventing new reference
        // dates; dropping the entries then bounds the footprint.
        if (m_entries.size() >= kMaxEntries) {
            m_entries.clear();
        }
        const ScheduleKey key{effective.serial(), maturity.serial(), tenor.length(),
                              tenor.unit(),       &calendar.name(),  convention};
        const auto generate = [&] {
            return datetime::Schedule(effective, maturity, tenor, calendar, convention,
                                      datetime::DateGeneration::Forward, false,
                                      datetime::BusinessDayConvention::Unadjusted);
        };
        for (CachedSchedule& entry : m_entries) {
            if (entry.key.effective != key.effective || entry.key.maturity != key.maturity ||
                entry.key.tenorLength != key.tenorLength || entry.key.tenorUnit != key.tenorUnit ||
                entry.key.calendarName != key.calendarName ||
                entry.key.convention != key.convention) {
                continue;
            }
            if (entry.calendarAddress == &calendar) {
                return entry.schedule;
            }
            if (sameRolledDates(entry, calendar)) {
                entry.calendarAddress = &calendar;
                return entry.schedule;
            }
            entry.schedule = generate();
            entry.calendar = calendar;
            entry.calendarAddress = &calendar;
            return entry.schedule;
        }
        CachedSchedule entry;
        entry.key = key;
        entry.calendar = calendar;
        entry.calendarAddress = &calendar;
        entry.schedule = generate();
        m_entries.push_back(std::move(entry));
        return m_entries.back().schedule;
    }

private:
    static constexpr std::size_t kMaxEntries = 1024;

    struct CachedSchedule {
        ScheduleKey key;
        datetime::Calendar calendar;
        const datetime::Calendar* calendarAddress = nullptr;
        datetime::Schedule schedule;
    };

    static bool sameRolledDates(const CachedSchedule& entry, const datetime::Calendar& calendar) {
        const datetime::Schedule& schedule = entry.schedule;
        const std::vector<datetime::Date>& unadjusted = schedule.unadjustedDates();
        for (std::size_t i = 0; i < unadjusted.size(); ++i) {
            const datetime::BusinessDayConvention convention =
                i == 0 ? schedule.effectiveConvention()
                       : (i + 1 == unadjusted.size() ? schedule.terminationConvention()
                                                     : schedule.businessDayConvention());
            if (entry.calendar.adjust(unadjusted[i], convention) !=
                calendar.adjust(unadjusted[i], convention)) {
                return false;
            }
        }
        return true;
    }

    // A deque keeps references to cached schedules stable while it grows, so
    // two lookups in one row cannot dangle each other.
    std::deque<CachedSchedule> m_entries;
};

/// Per-thread schedule cache of the quote-time builders.
inline ScheduleCache& quoteScheduleCache() {
    static thread_local ScheduleCache cache;
    return cache;
}

} // namespace detail

/// True when the curve set exposes a forecast provider (`ForecastSet`).
template <typename CurveSetT>
concept HasForecastProvider = requires(const CurveSetT& curves, double t) { curves.forecast(t); };

/// True when the provider returns a `double` discount factor at a time: the
/// curve-side constraint used by the annuity and par-rate algebra.
template <typename ProviderT>
concept DiscountProvider = requires(const ProviderT& provider, double t) {
    { provider.discount(t) } -> std::convertible_to<double>;
};

/// Routes `curves` to the forecast side when present and to the discount side
/// otherwise, so one instrument type serves both bootstrap entry points.
template <typename CurveSetT>
decltype(auto) discountProvider(const CurveSetT& curves) {
    if constexpr (HasForecastProvider<CurveSetT>) {
        return detail::ForecastDiscountProxy<CurveSetT>{curves};
    } else {
        return curves;
    }
}

/// Precomputed future quote inputs for every underlying style. Simple and
/// compounded futures use `t1`/`t2`/`tau`; the averaged styles carry the
/// fixing grid times (with per-fixing accruals for the arithmetic mean).
struct FutureQuoteTimes {
    enum class Style : std::uint8_t { Simple, AveragedArithmetic, AveragedCompounded };
    Style style = Style::Simple;
    double t1 = 0.0;
    double t2 = 0.0;
    double tau = 0.0;
    double convexity = 0.0;
    std::vector<double> previousTimes;
    std::vector<double> currentTimes;
    std::vector<double> accrualTaus;
};

/// Precomputed quote inputs of one OIS coupon.
struct OisCouponTimes {
    double tau = 0.0;
    double tPay = 0.0;
    double tStart = 0.0;
    double tEnd = 0.0;
    bool firstFixed = false;
    double firstRate = 0.0;
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

/// Prepared coupon grids of one IRS quote.
struct IrsQuoteTimes {
    std::vector<IrsFloatCouponTimes> floatCoupons;
    std::vector<IrsAnnuityCouponTimes> annuityCoupons;
};

/// Prepared coupon grid of one basis-swap quote.
struct BasisQuoteTimes {
    std::vector<BasisCouponTimes> coupons;
};

/// Averaged arithmetic overnight future quote: the mean of the daily simple
/// overnight forwards over the business-day fixing grid, excluding the
/// convexity adjustment.
template <typename ScalarT, typename CurveT>
ScalarT averagedArithmeticFuturesQuote(const CurveT& curve, const datetime::Date& referenceDate,
                                       const datetime::Calendar& calendar,
                                       const datetime::Date& start, const datetime::Date& maturity,
                                       const datetime::DayCounter& quoteDayCounter,
                                       const datetime::DayCounter& zeroDayCounter,
                                       std::string_view context) {
    const std::vector<datetime::Date> fixings = businessDayFixings(calendar, start, maturity);
    if (fixings.size() < 2) {
        throw std::invalid_argument(std::string(context) +
                                    ": empty averaged futures reference period");
    }
    const double t1 = datetime::yearFraction(referenceDate, start, zeroDayCounter);
    if (!(t1 >= 0.0)) {
        throw std::invalid_argument(std::string(context) +
                                    ": futures fixing before the reference date");
    }
    ScalarT sum = 0.0;
    for (std::size_t k = 1; k < fixings.size(); ++k) {
        const double tau = datetime::yearFraction(fixings[k - 1], fixings[k], quoteDayCounter);
        if (!(tau > 0.0)) {
            throw std::invalid_argument(std::string(context) +
                                        ": non-positive averaged futures accrual");
        }
        const double previous =
            datetime::yearFraction(referenceDate, fixings[k - 1], zeroDayCounter);
        const double current = datetime::yearFraction(referenceDate, fixings[k], zeroDayCounter);
        sum += (curve.discount(previous) / curve.discount(current) - 1.0) / tau;
    }
    return sum / static_cast<double>(fixings.size() - 1);
}

/// Precomputes a money-market deposit/repo quote at `maturity` (already rolled
/// on the instrument calendar) into simple-forward times.
inline SimpleForwardQuoteTimes prepareDepositTimes(const datetime::Date& maturity,
                                                   const datetime::Date& referenceDate,
                                                   const datetime::DayCounter& quoteDayCounter,
                                                   const datetime::DayCounter& zeroDayCounter) {
    SimpleForwardQuoteTimes times;
    times.tau = datetime::yearFraction(referenceDate, maturity, quoteDayCounter);
    if (!(times.tau > 0.0)) {
        throw std::invalid_argument("impliedQuote: non-positive deposit accrual");
    }
    times.t1 = 0.0;
    times.t2 = datetime::yearFraction(referenceDate, maturity, zeroDayCounter);
    return times;
}

/// Precomputes a discount-curve FRA quote over `[start, maturity]` (both
/// already rolled on the instrument calendar) into simple-forward times.
inline SimpleForwardQuoteTimes prepareFraTimes(const datetime::Date& start,
                                               const datetime::Date& maturity,
                                               const datetime::Date& referenceDate,
                                               const datetime::DayCounter& quoteDayCounter,
                                               const datetime::DayCounter& zeroDayCounter) {
    SimpleForwardQuoteTimes times;
    times.t1 = datetime::yearFraction(referenceDate, start, zeroDayCounter);
    times.t2 = datetime::yearFraction(referenceDate, maturity, zeroDayCounter);
    times.tau = datetime::yearFraction(start, maturity, quoteDayCounter);
    if (!(times.tau > 0.0)) {
        throw std::invalid_argument("impliedQuote: non-positive FRA accrual");
    }
    if (!(times.t1 >= 0.0)) {
        throw std::invalid_argument("impliedQuote: FRA start before the reference date");
    }
    return times;
}

/// Precomputes a forecast-curve synthetic money-market quote (deposit or
/// forward-starting FRA) over `[start, maturity]`.
inline SimpleForwardQuoteTimes
prepareSimpleForwardTimes(const datetime::Date& start, const datetime::Date& maturity,
                          const datetime::Date& referenceDate,
                          const datetime::DayCounter& quoteDayCounter,
                          const datetime::DayCounter& zeroDayCounter) {
    SimpleForwardQuoteTimes times;
    times.t1 = datetime::yearFraction(referenceDate, start, zeroDayCounter);
    times.t2 = datetime::yearFraction(referenceDate, maturity, zeroDayCounter);
    times.tau = datetime::yearFraction(start, maturity, quoteDayCounter);
    if (!(times.tau > 0.0)) {
        throw std::invalid_argument("impliedSimpleForward: non-positive accrual");
    }
    if (!(times.t1 >= 0.0)) {
        throw std::invalid_argument("impliedSimpleForward: start before the reference date");
    }
    return times;
}

/// Precomputes an exchange-traded future quote for every underlying style.
/// `context` prefixes thrown messages so each caller keeps its own error text.
inline FutureQuoteTimes
prepareFutureTimes(const datetime::Date& start, const datetime::Date& maturity,
                   const datetime::Calendar& calendar, const datetime::DayCounter& quoteDayCounter,
                   FutureStyle futureStyle, AveragingStyle averagingStyle,
                   double convexityAdjustment, const datetime::Date& referenceDate,
                   const datetime::DayCounter& zeroDayCounter, std::string_view context) {
    FutureQuoteTimes times;
    times.convexity = convexityAdjustment;
    if (futureStyle == FutureStyle::Averaged && averagingStyle == AveragingStyle::Compounded) {
        times.style = FutureQuoteTimes::Style::AveragedCompounded;
        const std::vector<datetime::Date> fixings = businessDayFixings(calendar, start, maturity);
        if (fixings.size() < 2) {
            throw std::invalid_argument(std::string(context) +
                                        ": empty averaged futures reference period");
        }
        times.t1 = datetime::yearFraction(referenceDate, start, zeroDayCounter);
        times.tau = datetime::yearFraction(start, maturity, quoteDayCounter);
        if (!(times.tau > 0.0)) {
            throw std::invalid_argument(std::string(context) + ": non-positive futures accrual");
        }
        if (!(times.t1 >= 0.0)) {
            throw std::invalid_argument(std::string(context) +
                                        ": futures fixing before the reference date");
        }
        times.previousTimes.reserve(fixings.size() - 1);
        times.currentTimes.reserve(fixings.size() - 1);
        for (std::size_t k = 1; k < fixings.size(); ++k) {
            times.previousTimes.push_back(
                datetime::yearFraction(referenceDate, fixings[k - 1], zeroDayCounter));
            times.currentTimes.push_back(
                datetime::yearFraction(referenceDate, fixings[k], zeroDayCounter));
        }
        return times;
    }
    if (futureStyle == FutureStyle::Averaged) {
        times.style = FutureQuoteTimes::Style::AveragedArithmetic;
        const std::vector<datetime::Date> fixings = businessDayFixings(calendar, start, maturity);
        if (fixings.size() < 2) {
            throw std::invalid_argument(std::string(context) +
                                        ": empty averaged futures reference period");
        }
        times.t1 = datetime::yearFraction(referenceDate, start, zeroDayCounter);
        if (!(times.t1 >= 0.0)) {
            throw std::invalid_argument(std::string(context) +
                                        ": futures fixing before the reference date");
        }
        times.previousTimes.reserve(fixings.size() - 1);
        times.currentTimes.reserve(fixings.size() - 1);
        times.accrualTaus.reserve(fixings.size() - 1);
        for (std::size_t k = 1; k < fixings.size(); ++k) {
            const double tau = datetime::yearFraction(fixings[k - 1], fixings[k], quoteDayCounter);
            if (!(tau > 0.0)) {
                throw std::invalid_argument(std::string(context) +
                                            ": non-positive averaged futures accrual");
            }
            times.previousTimes.push_back(
                datetime::yearFraction(referenceDate, fixings[k - 1], zeroDayCounter));
            times.currentTimes.push_back(
                datetime::yearFraction(referenceDate, fixings[k], zeroDayCounter));
            times.accrualTaus.push_back(tau);
        }
        return times;
    }
    times.t1 = datetime::yearFraction(referenceDate, start, zeroDayCounter);
    times.t2 = datetime::yearFraction(referenceDate, maturity, zeroDayCounter);
    times.tau = datetime::yearFraction(start, maturity, quoteDayCounter);
    if (!(times.tau > 0.0)) {
        throw std::invalid_argument(std::string(context) + ": non-positive futures accrual");
    }
    if (!(times.t1 >= 0.0)) {
        throw std::invalid_argument(std::string(context) +
                                    ": futures fixing before the reference date");
    }
    return times;
}

/// Precomputes an OIS fixed/floating schedule's coupon times. `effective` is
/// the schedule anchor (the past start date for seasoned swaps), `maturity`
/// the quoted termination.
inline std::vector<OisCouponTimes>
prepareOisCouponTimes(const datetime::Date& effective, const datetime::Date& maturity,
                      const datetime::Period& fixedTenor, const datetime::Calendar& calendar,
                      datetime::BusinessDayConvention businessDayConvention, int paymentLag,
                      const datetime::DayCounter& quoteDayCounter, bool firstCouponFixed,
                      double firstCouponRate, const datetime::Date& referenceDate,
                      const datetime::DayCounter& zeroDayCounter) {
    const datetime::Schedule& schedule = detail::quoteScheduleCache().get(
        effective, maturity, fixedTenor, calendar, businessDayConvention);
    const std::vector<datetime::Date>& dates = schedule.dates();
    std::vector<OisCouponTimes> coupons;
    coupons.reserve(dates.size() - 1);
    for (std::size_t k = 1; k < dates.size(); ++k) {
        OisCouponTimes coupon;
        coupon.tau = datetime::yearFraction(dates[k - 1], dates[k], quoteDayCounter);
        if (!(coupon.tau > 0.0)) {
            throw std::invalid_argument("impliedQuote: non-positive OIS accrual");
        }
        const datetime::Date payDate =
            calendar.advance(dates[k], datetime::Period(paymentLag, datetime::TimeUnit::Days),
                             businessDayConvention);
        coupon.tPay = datetime::yearFraction(referenceDate, payDate, zeroDayCounter);
        if (k == 1 && firstCouponFixed) {
            coupon.firstFixed = true;
            coupon.firstRate = firstCouponRate;
        } else {
            coupon.tStart = datetime::yearFraction(referenceDate, dates[k - 1], zeroDayCounter);
            coupon.tEnd = datetime::yearFraction(referenceDate, dates[k], zeroDayCounter);
            if (!(coupon.tStart >= 0.0) || !(coupon.tEnd > 0.0)) {
                throw std::invalid_argument(
                    "impliedQuote: coupons before the reference date must be fixed");
            }
        }
        coupons.push_back(coupon);
    }
    return coupons;
}

/// Simple-forward quote over the prepared times: `(D(t1)/D(t2) - 1) / tau`.
template <typename ScalarT, typename CurveT>
ScalarT impliedSimpleForwardQuote(const SimpleForwardQuoteTimes& times, const CurveT& curve) {
    return (curve.discount(times.t1) / curve.discount(times.t2) - 1.0) / times.tau;
}

/// Deposit/repo quote of a curve at the prepared maturity: the simple forward
/// from the reference date (t1 = 0).
template <typename ScalarT, typename CurveT>
ScalarT impliedDepositQuote(const SimpleForwardQuoteTimes& times, const CurveT& curve) {
    return impliedSimpleForwardQuote<ScalarT>(times, curve);
}

/// Repo quote: the same money-market algebra as a deposit.
template <typename ScalarT, typename CurveT>
ScalarT impliedRepoQuote(const SimpleForwardQuoteTimes& times, const CurveT& curve) {
    return impliedSimpleForwardQuote<ScalarT>(times, curve);
}

/// FRA quote with the shifted-lognormal convexity convention:
/// `R = ((1 + f tau) exp(C) - 1) / tau`.
template <typename ScalarT, typename CurveT>
ScalarT impliedFraQuote(const SimpleForwardQuoteTimes& times, double convexityExponent,
                        const CurveT& curve) {
    const ScalarT forward = impliedSimpleForwardQuote<ScalarT>(times, curve);
    if (convexityExponent == 0.0) {
        return forward;
    }
    using std::exp;
    return ((1.0 + forward * times.tau) * exp(convexityExponent) - 1.0) / times.tau;
}

/// Future quote from prepared times: simple/compounded is the period forward,
/// averaged styles use the stored fixing grid, plus the convexity adjustment.
template <typename ScalarT, typename CurveT>
ScalarT impliedFutureQuote(const FutureQuoteTimes& future, const CurveT& curve) {
    if (future.style == FutureQuoteTimes::Style::AveragedCompounded) {
        ScalarT accumulated = 1.0;
        ScalarT previousDiscount = 0.0;
        bool hasPrevious = false;
        for (std::size_t k = 0; k < future.previousTimes.size(); ++k) {
            if (!hasPrevious) {
                previousDiscount = curve.discount(future.previousTimes[k]);
            }
            const ScalarT currentDiscount = curve.discount(future.currentTimes[k]);
            accumulated *= previousDiscount / currentDiscount;
            previousDiscount = currentDiscount;
            hasPrevious = true;
        }
        return (accumulated - 1.0) / future.tau + future.convexity;
    }
    if (future.style == FutureQuoteTimes::Style::AveragedArithmetic) {
        ScalarT sum = 0.0;
        ScalarT previousDiscount = 0.0;
        bool hasPrevious = false;
        for (std::size_t k = 0; k < future.previousTimes.size(); ++k) {
            if (!hasPrevious) {
                previousDiscount = curve.discount(future.previousTimes[k]);
            }
            const ScalarT currentDiscount = curve.discount(future.currentTimes[k]);
            sum += (previousDiscount / currentDiscount - 1.0) / future.accrualTaus[k];
            previousDiscount = currentDiscount;
            hasPrevious = true;
        }
        return sum / static_cast<double>(future.previousTimes.size()) + future.convexity;
    }
    return (curve.discount(future.t1) / curve.discount(future.t2) - 1.0) / future.tau +
           future.convexity;
}

/// Par rate of an OIS from prepared coupon times: the compounded floating leg
/// telescopes over the schedule, so the result is the fixed rate that prices
/// the swap at par.
template <typename ScalarT, typename CurveT>
ScalarT impliedOisParRate(const std::vector<OisCouponTimes>& coupons, const CurveT& curve) {
    ScalarT annuity = 0.0;
    ScalarT floating = 0.0;
    ScalarT startDiscount = 0.0;
    bool hasStart = false;
    for (const OisCouponTimes& coupon : coupons) {
        const ScalarT discountPay = curve.discount(coupon.tPay);
        annuity += coupon.tau * discountPay;
        if (coupon.firstFixed) {
            floating += discountPay * coupon.tau * coupon.firstRate;
            hasStart = false;
            continue;
        }
        const ScalarT couponStart = hasStart ? startDiscount : curve.discount(coupon.tStart);
        const ScalarT discountEnd = curve.discount(coupon.tEnd);
        floating += discountPay * (couponStart / discountEnd - 1.0);
        startDiscount = discountEnd;
        hasStart = true;
    }
    if (!(annuity > 0.0)) {
        throw std::invalid_argument("impliedQuote: non-positive OIS annuity");
    }
    return floating / annuity;
}

/// Precomputes an IRS floating leg's coupon times over its own schedule:
/// `start` is the effective date when set and the reference date otherwise.
template <typename IrsLike>
inline std::vector<IrsFloatCouponTimes>
makeIrsFloatCouponTimes(const IrsLike& pillar, const datetime::Date& referenceDate,
                        const datetime::DayCounter& zeroDayCounter) {
    const datetime::Date effective = pillar.start.serial() != 0 ? pillar.start : referenceDate;
    const datetime::Schedule& floatSchedule =
        detail::quoteScheduleCache().get(effective, pillar.maturity, pillar.floatTenor,
                                         pillar.floatCalendar, pillar.businessDayConvention);
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

/// Precomputes an IRS fixed leg's annuity coupon times over its own schedule.
template <typename IrsLike>
inline std::vector<IrsAnnuityCouponTimes>
makeIrsAnnuityCouponTimes(const IrsLike& pillar, const datetime::Date& referenceDate,
                          const datetime::DayCounter& zeroDayCounter) {
    const datetime::Date effective = pillar.start.serial() != 0 ? pillar.start : referenceDate;
    const datetime::Schedule& schedule =
        detail::quoteScheduleCache().get(effective, pillar.maturity, pillar.fixedTenor,
                                         pillar.fixedCalendar, pillar.businessDayConvention);
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

/// Precomputes a basis swap's coupon times; the schedule anchors at the
/// reference date (basis pillars quote spot-starting).
template <typename BasisLike>
inline std::vector<BasisCouponTimes>
makeBasisCouponTimes(const BasisLike& pillar, const datetime::Date& referenceDate,
                     const datetime::DayCounter& zeroDayCounter) {
    const datetime::Schedule& schedule =
        detail::quoteScheduleCache().get(referenceDate, pillar.maturity, pillar.floatTenor,
                                         pillar.calendar, pillar.businessDayConvention);
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

/// Both prepared legs of one IRS quote.
template <typename IrsLike>
inline IrsQuoteTimes makeIrsQuoteTimes(const IrsLike& pillar, const datetime::Date& referenceDate,
                                       const datetime::DayCounter& zeroDayCounter) {
    IrsQuoteTimes times;
    times.floatCoupons = makeIrsFloatCouponTimes(pillar, referenceDate, zeroDayCounter);
    times.annuityCoupons = makeIrsAnnuityCouponTimes(pillar, referenceDate, zeroDayCounter);
    return times;
}

/// Prepared coupons of one basis quote.
template <typename BasisLike>
inline BasisQuoteTimes makeBasisQuoteTimes(const BasisLike& pillar,
                                           const datetime::Date& referenceDate,
                                           const datetime::DayCounter& zeroDayCounter) {
    BasisQuoteTimes times;
    times.coupons = makeBasisCouponTimes(pillar, referenceDate, zeroDayCounter);
    return times;
}

/// Annuity of one leg schedule over the discount curve at its payment dates:
/// the schedule starts at `effective` while discount times are measured from
/// `valuationDate` (they differ for seasoned instruments).
template <typename DiscountT>
    requires DiscountProvider<DiscountT>
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

/// Fixed-leg annuity of an IRS quote over the discount curve.
template <typename DiscountT, typename IrsLike>
    requires DiscountProvider<DiscountT>
inline double fixedAnnuity(const DiscountT& discount, const IrsLike& pillar,
                           const datetime::Date& referenceDate,
                           const datetime::DayCounter& zeroDayCounter) {
    const datetime::Date effective = pillar.start.serial() != 0 ? pillar.start : referenceDate;
    return pricing::legAnnuity(discount, effective, referenceDate, pillar.maturity,
                               pillar.fixedTenor, pillar.fixedCalendar, pillar.fixedDayCounter,
                               pillar.businessDayConvention, pillar.paymentLag, zeroDayCounter);
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

/// Par rate of a fixed-vs-floating IRS from prepared coupon times: the
/// floating forwards come from `forecast`, every discount factor and the
/// fixed annuity from `discounting` (exogenous collateral discounting).
template <typename ScalarT, typename ForecastT, typename DiscountT>
ScalarT impliedIrsParRate(const IrsQuoteTimes& times, const ForecastT& forecast,
                          const DiscountT& discounting) {
    ScalarT floatPv = 0.0;
    ScalarT previousAccrualDiscount = 0.0;
    bool hasPrevious = false;
    for (const IrsFloatCouponTimes& coupon : times.floatCoupons) {
        const ScalarT discountPay = discounting.discount(coupon.tPay);
        if (coupon.firstFixed) {
            floatPv += coupon.tau * discountPay * coupon.firstRate;
            hasPrevious = false;
            continue;
        }
        const ScalarT previousDiscount =
            hasPrevious ? previousAccrualDiscount : forecast.discount(coupon.tPrevious);
        const ScalarT accrualDiscount = forecast.discount(coupon.tAccrual);
        const ScalarT forward = (previousDiscount / accrualDiscount - 1.0) / coupon.tau;
        floatPv += coupon.tau * discountPay * forward;
        previousAccrualDiscount = accrualDiscount;
        hasPrevious = true;
    }
    ScalarT annuity = 0.0;
    for (const IrsAnnuityCouponTimes& coupon : times.annuityCoupons) {
        annuity += coupon.tau * discounting.discount(coupon.tPay);
    }
    if (!(annuity > 0.0)) {
        throw std::invalid_argument("impliedIrsRate: non-positive fixed annuity");
    }
    return floatPv / annuity;
}

/// Implied par basis spread from prepared coupon times: the child leg
/// forecasts on `child`, the parent leg on `parentForecast`, discounting on
/// `discountAt`. The spread sign follows `spreadOnParentLeg`.
template <typename ScalarT, typename ChildT, typename ParentT, typename DiscountFn>
ScalarT impliedBasisSpread(const BasisQuoteTimes& times, const ChildT& child,
                           const ParentT& parentForecast, const DiscountFn& discountAt,
                           bool spreadOnParentLeg) {
    ScalarT weightedChild = 0.0;
    ScalarT weightedParent = 0.0;
    ScalarT annuity = 0.0;
    ScalarT previousChild = 0.0;
    ScalarT previousParent = 0.0;
    bool hasPrevious = false;
    for (const BasisCouponTimes& coupon : times.coupons) {
        const ScalarT df = discountAt(coupon.t);
        const ScalarT childAtPrevious = hasPrevious ? previousChild : child.discount(coupon.tPrev);
        const ScalarT childAtCoupon = child.discount(coupon.t);
        const ScalarT parentAtPrevious =
            hasPrevious ? previousParent : parentForecast.discount(coupon.tPrev);
        const ScalarT parentAtCoupon = parentForecast.discount(coupon.t);
        const ScalarT forwardChild = (childAtPrevious / childAtCoupon - 1.0) / coupon.tau;
        const ScalarT forwardParent = (parentAtPrevious / parentAtCoupon - 1.0) / coupon.tau;
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
    const ScalarT level = (weightedChild - weightedParent) / annuity;
    return spreadOnParentLeg ? level : -level;
}

} // namespace quantape::pricing
