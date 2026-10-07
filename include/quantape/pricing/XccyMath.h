#pragma once

#include "quantape/datetime/BusinessDayConvention.h"
#include "quantape/datetime/Calendar.h"
#include "quantape/datetime/Date.h"
#include "quantape/datetime/DayCounter.h"
#include "quantape/datetime/Period.h"
#include "quantape/datetime/Schedule.h"
#include "quantape/datetime/TimeConversion.h"

#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <vector>

namespace quantape::pricing {
/**
 * @file XccyMath.h
 * @brief Scalar-templated cross-currency leg and par-spread arithmetic
 *
 * This header holds the cross-currency basis algebra with no instrument
 * dependency: schedule arithmetic resolves both legs into precomputed coupon
 * time grids once, and the evaluators only read discount factors from curve
 * providers through `discount(t)`. The same `ScalarT` template serves the
 * `double` bootstrap, reverse-mode gradients and nested second-order tests;
 * branching is on convention flags only, never on curve values.
 *
 * A constant-notional leg pays coupons on the accrual ends and exchanges the
 * notional at the leg boundaries; a resetting leg's notional resets at every
 * accrual start to the opposite leg's forward-implied level, so its netted
 * cashflows telescope to
 *
 *   `sum_k adjN_k (D_own(t_k) (1 + f_k tau_k) - D_own(t_{k-1}))`,
 *   `adjN_k = D_other(t_{k-1}) / D_own(t_{k-1})`.
 *
 * The instrument wrappers in `pricing/Xccy.h` compose this layer with the
 * plain-data `instruments::XccyBasisSwap`. Prepared-times construction throws
 * the exact error text of the original per-pillar entry points, because
 * callers forward those messages verbatim.
 */

/// Notional convention of the cross-currency basis legs.
enum class XccyNotionalMode : std::uint8_t {
    Const, ///< Both legs keep their inception notional until maturity
    MtM    ///< The selected leg's notional resets at every accrual start
};

/// Schedule-fixed coupon data of one cross-currency leg: accrual, the accrual
/// boundaries and the payment time, all on the zero clock.
struct XccyCouponTimes {
    double tau = 0.0;
    double tPrevious = 0.0;
    double tAccrual = 0.0;
    double tPay = 0.0;
};

/// Schedule-fixed data of one cross-currency leg, resolved once per pillar.
struct XccyLegTimes {
    std::vector<XccyCouponTimes> coupons;
    double tStart = 0.0;
    double tEnd = 0.0;
};

/// Two resolved leg schedules of one pillar plus its notional flags.
struct XccyPillarTimes {
    XccyLegTimes foreign;
    XccyLegTimes domestic;
    bool foreignResets = false;
    bool domesticResets = false;
};

/// Resolve one leg's schedule and coupon times. The accrual check matches the
/// inline evaluation it replaces.
inline XccyLegTimes makeXccyLegTimes(const datetime::Schedule& schedule,
                                     const datetime::DayCounter& accrualDayCounter, int paymentLag,
                                     datetime::BusinessDayConvention businessDayConvention,
                                     const datetime::Date& referenceDate,
                                     const datetime::DayCounter& zeroDayCounter) {
    XccyLegTimes times;
    const std::vector<datetime::Date>& dates = schedule.dates();
    times.coupons.reserve(dates.size() - 1);
    for (std::size_t k = 1; k < dates.size(); ++k) {
        XccyCouponTimes coupon;
        coupon.tau = datetime::yearFraction(dates[k - 1], dates[k], accrualDayCounter);
        if (!(coupon.tau > 0.0)) {
            throw std::invalid_argument("impliedXccyBasisSpread: non-positive accrual");
        }
        coupon.tPrevious = datetime::yearFraction(referenceDate, dates[k - 1], zeroDayCounter);
        coupon.tAccrual = datetime::yearFraction(referenceDate, dates[k], zeroDayCounter);
        const datetime::Date payDate = schedule.calendar().advance(
            dates[k], datetime::Period(paymentLag, datetime::TimeUnit::Days),
            businessDayConvention);
        coupon.tPay = datetime::yearFraction(referenceDate, payDate, zeroDayCounter);
        times.coupons.push_back(coupon);
    }
    times.tStart = datetime::yearFraction(referenceDate, dates.front(), zeroDayCounter);
    times.tEnd = datetime::yearFraction(referenceDate, dates.back(), zeroDayCounter);
    return times;
}

/// Resolve both leg schedules and coupon times of `pillar` on the reference
/// date's zero clock.
template <typename XccyLike>
inline XccyPillarTimes makeXccyPillarTimes(const XccyLike& pillar,
                                           const datetime::Date& referenceDate,
                                           const datetime::DayCounter& zeroDayCounter) {
    const datetime::Schedule foreignSchedule(
        referenceDate, pillar.maturity, pillar.foreignTenor, pillar.foreignCalendar,
        pillar.foreignBusinessDayConvention, datetime::DateGeneration::Forward, false,
        datetime::BusinessDayConvention::Unadjusted);
    const datetime::Schedule domesticSchedule(
        referenceDate, pillar.maturity, pillar.domesticTenor, pillar.domesticCalendar,
        pillar.domesticBusinessDayConvention, datetime::DateGeneration::Forward, false,
        datetime::BusinessDayConvention::Unadjusted);
    XccyPillarTimes times;
    times.foreign =
        makeXccyLegTimes(foreignSchedule, pillar.foreignDayCounter, pillar.foreignPaymentLag,
                         pillar.foreignBusinessDayConvention, referenceDate, zeroDayCounter);
    times.domestic =
        makeXccyLegTimes(domesticSchedule, pillar.domesticDayCounter, pillar.domesticPaymentLag,
                         pillar.domesticBusinessDayConvention, referenceDate, zeroDayCounter);
    times.foreignResets = pillar.notional == XccyNotionalMode::MtM && pillar.resetForeignLeg;
    times.domesticResets = pillar.notional == XccyNotionalMode::MtM && !pillar.resetForeignLeg;
    return times;
}

/// Value and annuity of one leg from precomputed coupon times. The arithmetic
/// follows the original inline loop step for step; a resetting leg returns the
/// telescoped reset value instead of the constant-notional leg value.
template <typename ScalarT, typename DiscountT, typename ForecastT, typename OtherDiscountT>
ScalarT xccyLegValue(const XccyLegTimes& times, const DiscountT& discount,
                     const ForecastT& forecast, const OtherDiscountT& otherDiscount, bool resets,
                     ScalarT& annuity) {
    ScalarT coupons = 0.0;
    ScalarT resetValue = 0.0;
    annuity = 0.0;
    for (const XccyCouponTimes& coupon : times.coupons) {
        const ScalarT df = discount.discount(coupon.tPay);
        const ScalarT forward =
            (forecast.discount(coupon.tPrevious) / forecast.discount(coupon.tAccrual) - 1.0) /
            coupon.tau;
        coupons += coupon.tau * df * forward;
        annuity += coupon.tau * df;
        if (resets) {
            const ScalarT ownStart = discount.discount(coupon.tPrevious);
            const ScalarT adjustment = otherDiscount.discount(coupon.tPrevious) / ownStart;
            resetValue +=
                adjustment *
                (discount.discount(coupon.tAccrual) * (1.0 + forward * coupon.tau) - ownStart);
        }
    }
    if (resets) {
        return resetValue;
    }
    const ScalarT notional = discount.discount(times.tStart) - discount.discount(times.tEnd);
    return coupons + notional;
}

/// Model basis spread from precomputed pillar times: both leg values and the
/// quote-leg par division. A resetting leg's reset adjustment reads the
/// opposite leg's discount curve.
template <typename ScalarT, typename ForeignDiscountT, typename ForeignForecastT,
          typename DomesticDiscountT, typename DomesticForecastT>
ScalarT xccyParSpread(const XccyPillarTimes& times, bool spreadOnForeignLeg,
                      const ForeignDiscountT& foreignDiscount,
                      const ForeignForecastT& foreignForecast,
                      const DomesticDiscountT& domesticDiscount,
                      const DomesticForecastT& domesticForecast) {
    ScalarT foreignAnnuity = 0.0;
    const ScalarT foreignValue =
        xccyLegValue<ScalarT>(times.foreign, foreignDiscount, foreignForecast, domesticDiscount,
                              times.foreignResets, foreignAnnuity);
    ScalarT domesticAnnuity = 0.0;
    const ScalarT domesticValue =
        xccyLegValue<ScalarT>(times.domestic, domesticDiscount, domesticForecast, foreignDiscount,
                              times.domesticResets, domesticAnnuity);
    if (!(foreignAnnuity > 0.0) || !(domesticAnnuity > 0.0)) {
        throw std::invalid_argument("impliedXccyBasisSpread: non-positive annuity");
    }
    if (spreadOnForeignLeg) {
        return (domesticValue - foreignValue) / foreignAnnuity;
    }
    return (foreignValue - domesticValue) / domesticAnnuity;
}

} // namespace quantape::pricing
