#pragma once

#include "quantape/datetime/Calendar.h"
#include "quantape/datetime/Date.h"
#include "quantape/datetime/DayCounter.h"
#include "quantape/datetime/Period.h"
#include "quantape/datetime/Schedule.h"
#include "quantape/datetime/TimeConversion.h"

#include <cmath>
#include <cstddef>
#include <cstdint>
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

} // namespace detail

/// True when the curve set exposes a forecast provider (`ForecastSet`).
template <typename CurveSetT>
concept HasForecastProvider = requires(const CurveSetT& curves, double t) { curves.forecast(t); };

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
    const datetime::Schedule schedule(effective, maturity, fixedTenor, calendar,
                                      businessDayConvention, datetime::DateGeneration::Forward,
                                      false, datetime::BusinessDayConvention::Unadjusted);
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

} // namespace quantape::pricing
