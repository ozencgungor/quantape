#pragma once

#include "quantape/datetime/Calendar.h"
#include "quantape/datetime/Date.h"
#include "quantape/datetime/Period.h"
#include "quantape/datetime/Schedule.h"
#include "quantape/datetime/TimeConversion.h"
#include "quantape/markets/Curves/CurveBuilder.h"
#include "quantape/markets/Curves/DiscountCurve.h"
#include "quantape/markets/Curves/FxSwapBuilder.h"
#include "quantape/math/Optimization/LevenbergMarquardt.h"
#include "quantape/math/Solvers/BrentSolver.h"
#include "quantape/math/Solvers/FixedPointIterator.h"

#include <cmath>
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <type_traits>
#include <variant>
#include <vector>

namespace quantape::markets {
/**
 * @file XccyBasisBuilder.h
 * @brief Cross-currency basis bootstrap with constant or resetting notionals
 *
 * Bootstraps the *foreign discount curve collateralized in the domestic
 * currency* from cross-currency basis swap quotes, given the domestic discount
 * and both forecast curves. With constant notionals both legs exchange
 * notional at start and maturity and the FX spot cancels, so the par condition
 * is
 *
 *   `sum_f tau D_f (f_f + b) + (D_f(0) - D_f(T)) = sum_d tau D_d f_d + (D_d(0) - D_d(T))`
 *
 * with the basis spread `b` quoted on the foreign leg. Each pillar solves one
 * foreign zero node (foreign discounts at intermediate coupon dates are
 * interpolated, so arbitrary schedules are supported). Both legs support
 * stubs, payment lags and per-leg business-day conventions.
 *
 * In `XccyNotionalMode::MtM` the selected leg's notional resets at every
 * accrual start to the other leg's forward-implied level. Netted against the
 * periodic notional exchanges, its cashflows telescope to
 *
 *   `sum_k adjN_k (D_own(t_k) (1 + f_k tau_k) - D_own(t_{k-1}))`,
 *   `adjN_k = D_other(t_{k-1}) / D_own(t_{k-1})`,
 *
 * where `D_own`/`D_other` are the resetting and opposite legs' discount
 * curves, `f_k` the resetting leg's simple forward and `tau_k` its accrual.
 * The reset and the telescoping discounts are read on accrual boundaries; the
 * non-resetting leg keeps the constant-notional value above and the spread
 * annuity keeps its `tau * D(pay)` definition on both legs.
 */

/// Solved node times of a forecast provider: `DiscountCurve` exposes `times()`
/// directly, `SpreadCurve` through its spread-node curve.
template <typename T>
concept XccyNodeGrid = requires(const T& curve) {
    { curve.times() } -> std::convertible_to<const std::vector<double>&>;
} || requires(const T& curve) {
    { curve.spreadNodes().times() } -> std::convertible_to<const std::vector<double>&>;
};

/// Anything usable as a forecast curve (DiscountCurve, SpreadCurve, ...):
/// discounting plus the solved node grid that the cross-currency risk rows
/// read (`size()`, the node times and `zeroDayCounter()`).
template <typename T>
concept XccyForecastCurve = requires(const T& curve, double t) {
    { curve.discount(t) } -> std::convertible_to<double>;
    { curve.size() } -> std::convertible_to<std::size_t>;
    curve.zeroDayCounter();
} && XccyNodeGrid<T>;

/// Notional convention of the cross-currency basis legs.
enum class XccyNotionalMode : std::uint8_t {
    Const, ///< Both legs keep their inception notional until maturity
    MtM    ///< The selected leg's notional resets at every accrual start
};

/// Cross-currency basis pillar with both leg conventions and a constant or
/// resetting notional.
struct XccyPillar {
    datetime::Date maturity;
    double spread = 0.0;            ///< Basis spread (decimal)
    bool spreadOnForeignLeg = true; ///< Which leg carries the quoted basis
    /// Notional convention. Analytic stack risk rows cover `Const` pillars
    /// only; resetting pillars need reset-aware rows.
    XccyNotionalMode notional = XccyNotionalMode::Const;
    /// `MtM` only: which leg's notional resets (the domestic leg when false).
    /// The reset notional at an accrual start is the opposite leg's
    /// forward-implied level `D_other(tStart)/D_own(tStart)`. Ignored for
    /// `XccyNotionalMode::Const`.
    bool resetForeignLeg = true;
    datetime::Period foreignTenor{3, datetime::TimeUnit::Months};
    datetime::Calendar foreignCalendar{};
    datetime::DayCounter foreignDayCounter{datetime::DayCount::Actual360};
    datetime::Period domesticTenor{3, datetime::TimeUnit::Months};
    datetime::Calendar domesticCalendar{};
    datetime::DayCounter domesticDayCounter{datetime::DayCount::Actual360};
    int foreignPaymentLag = 0; ///< Coupon payment lag in business days
    int domesticPaymentLag = 0;
    datetime::BusinessDayConvention foreignBusinessDayConvention =
        datetime::BusinessDayConvention::ModifiedFollowing;
    datetime::BusinessDayConvention domesticBusinessDayConvention =
        datetime::BusinessDayConvention::ModifiedFollowing;
};

namespace detail {

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

/// Value and annuity of one leg from precomputed coupon times. The arithmetic
/// follows the original inline loop step for step.
template <typename DiscountT, typename ForecastT, typename OtherDiscountT>
double evaluateXccyLegTimes(const XccyLegTimes& times, const DiscountT& discount,
                            const ForecastT& forecast, const OtherDiscountT& otherDiscount,
                            bool resets, double& annuity) {
    double coupons = 0.0;
    double resetValue = 0.0;
    annuity = 0.0;
    for (const XccyCouponTimes& coupon : times.coupons) {
        const double df = discount.discount(coupon.tPay);
        const double forward =
            (forecast.discount(coupon.tPrevious) / forecast.discount(coupon.tAccrual) - 1.0) /
            coupon.tau;
        coupons += coupon.tau * df * forward;
        annuity += coupon.tau * df;
        if (resets) {
            const double ownStart = discount.discount(coupon.tPrevious);
            const double adjustment = otherDiscount.discount(coupon.tPrevious) / ownStart;
            resetValue +=
                adjustment *
                (discount.discount(coupon.tAccrual) * (1.0 + forward * coupon.tau) - ownStart);
        }
    }
    if (resets) {
        return resetValue;
    }
    const double notional = discount.discount(times.tStart) - discount.discount(times.tEnd);
    return coupons + notional;
}

/// Two resolved leg schedules of one pillar plus its notional flags.
struct XccyPillarTimes {
    XccyLegTimes foreign;
    XccyLegTimes domestic;
    bool foreignResets = false;
    bool domesticResets = false;
};

/// Resolve both leg schedules and coupon times of `pillar` on the reference
/// date's zero clock.
inline XccyPillarTimes makeXccyPillarTimes(const XccyPillar& pillar,
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

/// Model basis spread from precomputed pillar times: both leg values and the
/// quote-leg par division.
template <typename ForeignDiscountT, typename ForeignForecastT, typename DomesticDiscountT,
          typename DomesticForecastT>
double evaluateXccyPillarTimes(const XccyPillarTimes& times, const XccyPillar& pillar,
                               const ForeignDiscountT& foreignDiscount,
                               const ForeignForecastT& foreignForecast,
                               const DomesticDiscountT& domesticDiscount,
                               const DomesticForecastT& domesticForecast) {
    double foreignAnnuity = 0.0;
    const double foreignValue =
        evaluateXccyLegTimes(times.foreign, foreignDiscount, foreignForecast, domesticDiscount,
                             times.foreignResets, foreignAnnuity);
    double domesticAnnuity = 0.0;
    const double domesticValue =
        evaluateXccyLegTimes(times.domestic, domesticDiscount, domesticForecast, foreignDiscount,
                             times.domesticResets, domesticAnnuity);
    if (!(foreignAnnuity > 0.0) || !(domesticAnnuity > 0.0)) {
        throw std::invalid_argument("impliedXccyBasisSpread: non-positive annuity");
    }
    if (pillar.spreadOnForeignLeg) {
        return (domesticValue - foreignValue) / foreignAnnuity;
    }
    return (foreignValue - domesticValue) / domesticAnnuity;
}

} // namespace detail

/// Model-implied basis spread (foreign leg spread) against a foreign discount
/// curve, with both forecast curves given. Handles constant and resetting
/// notionals; the spread annuity is always `sum tau D(pay)` on the quoted leg.
template <XccyForecastCurve ForeignForecastT, XccyForecastCurve DomesticForecastT>
double impliedXccyBasisSpread(const DiscountCurve<double>& foreignDiscount,
                              const ForeignForecastT& foreignForecast,
                              const DiscountCurve<double>& domesticDiscount,
                              const DomesticForecastT& domesticForecast, const XccyPillar& pillar,
                              const datetime::Date& referenceDate,
                              const datetime::DayCounter& zeroDayCounter) {
    const detail::XccyPillarTimes times =
        detail::makeXccyPillarTimes(pillar, referenceDate, zeroDayCounter);
    return detail::evaluateXccyPillarTimes(times, pillar, foreignDiscount, foreignForecast,
                                           domesticDiscount, domesticForecast);
}

/// Sequential exact-fit bootstrap of the foreign discount curve (USD-collateral)
/// from constant- or resetting-notional xccy basis pillars; domestic curves
/// and the foreign forecast curve stay frozen.
template <XccyForecastCurve ForeignForecastT, XccyForecastCurve DomesticForecastT>
DiscountCurve<double> bootstrapXccyDiscountCurve(
    const DiscountCurve<double>& domesticDiscount, const DomesticForecastT& domesticForecast,
    const ForeignForecastT& foreignForecast, const datetime::Date& referenceDate,
    const datetime::DayCounter& zeroDayCounter, InterpolationSpace space,
    InterpolationScheme scheme, const std::vector<XccyPillar>& pillars, double accuracy = 1e-14,
    double tension = 0.0, int switchIndex = 1) {
    if (pillars.empty()) {
        throw std::invalid_argument("bootstrapXccyDiscountCurve: no pillars");
    }
    const std::size_t count = pillars.size();
    std::vector<double> nodeTimes(count);
    std::vector<datetime::Date> maturityDates(count);
    for (std::size_t i = 0; i < count; ++i) {
        const datetime::Date maturity = pillars[i].foreignCalendar.adjust(
            pillars[i].maturity, pillars[i].foreignBusinessDayConvention);
        nodeTimes[i] = datetime::yearFraction(referenceDate, maturity, zeroDayCounter);
        if (!(nodeTimes[i] > (i == 0 ? 0.0 : nodeTimes[i - 1]))) {
            throw std::invalid_argument(
                "bootstrapXccyDiscountCurve: maturities must be strictly increasing");
        }
        maturityDates[i] = maturity;
    }

    // The schedules and day counts are schedule-fixed, so resolve them once
    // per pillar instead of rebuilding them at every Brent step.
    std::vector<detail::XccyPillarTimes> pillarTimes(count);
    for (std::size_t i = 0; i < count; ++i) {
        pillarTimes[i] = detail::makeXccyPillarTimes(pillars[i], referenceDate, zeroDayCounter);
    }
    const auto residual = [&](std::size_t i, const DiscountCurve<double>& trial) {
        return detail::evaluateXccyPillarTimes(pillarTimes[i], pillars[i], trial, foreignForecast,
                                               domesticDiscount, domesticForecast) -
               pillars[i].spread;
    };
    const std::vector<double> zeros =
        detail::bootstrapNodesByBrent(space, scheme, tension, switchIndex, accuracy, nodeTimes,
                                      residual, "bootstrapXccyDiscountCurve");
    return DiscountCurve<double>(referenceDate, maturityDates, zeroDayCounter, zeros, space, scheme,
                                 tension, switchIndex);
}

/// One pillar of a mixed `XccyBasis`-role ladder: FX forward points at the
/// short end and cross-currency basis swaps at the long end.
using XccyMixedPillar = std::variant<FxSwapPillar, XccyPillar>;

namespace detail {

/// Node date of a mixed pillar: the FX far date or the xccy maturity adjusted
/// on the foreign leg's calendar.
inline datetime::Date mixedPillarMaturity(const XccyMixedPillar& pillar) {
    return std::visit(
        [](const auto& value) -> datetime::Date {
            using PillarT = std::decay_t<decltype(value)>;
            if constexpr (std::is_same_v<PillarT, FxSwapPillar>) {
                return value.maturity;
            } else {
                return value.foreignCalendar.adjust(value.maturity,
                                                    value.foreignBusinessDayConvention);
            }
        },
        pillar);
}

/// Shared body of the mixed-ladder bootstraps. `foreignForecastAt(trial)`
/// returns the foreign floating-leg forecast to use for xccy pillars, either
/// the trial curve itself or a frozen forecast curve.
template <typename ForeignForecastProvider, typename DomesticForecastT>
DiscountCurve<double> bootstrapMixedXccyDiscountCurveImpl(
    const DiscountCurve<double>& domesticDiscount, const DomesticForecastT& domesticForecast,
    const datetime::Date& referenceDate, const datetime::DayCounter& zeroDayCounter,
    InterpolationSpace space, InterpolationScheme scheme,
    const std::vector<XccyMixedPillar>& pillars, const ForeignForecastProvider& foreignForecastAt,
    double accuracy, double tension, int switchIndex) {
    if (pillars.empty()) {
        throw std::invalid_argument("bootstrapMixedXccyDiscountCurve: no pillars");
    }
    const std::size_t count = pillars.size();
    std::vector<double> nodeTimes(count);
    std::vector<datetime::Date> maturityDates(count);
    for (std::size_t i = 0; i < count; ++i) {
        maturityDates[i] = mixedPillarMaturity(pillars[i]);
        nodeTimes[i] = datetime::yearFraction(referenceDate, maturityDates[i], zeroDayCounter);
        if (!(nodeTimes[i] > (i == 0 ? 0.0 : nodeTimes[i - 1]))) {
            throw std::invalid_argument(
                "bootstrapMixedXccyDiscountCurve: maturities must be strictly increasing");
        }
    }
    // Resolve the xccy leg schedules once per pillar; the FX pillars are
    // closed-form on their precomputed node time.
    std::vector<detail::XccyPillarTimes> xccyTimes(count);
    for (std::size_t i = 0; i < count; ++i) {
        std::visit(
            [&](const auto& value) {
                using PillarT = std::decay_t<decltype(value)>;
                if constexpr (std::is_same_v<PillarT, XccyPillar>) {
                    xccyTimes[i] =
                        detail::makeXccyPillarTimes(value, referenceDate, zeroDayCounter);
                }
            },
            pillars[i]);
    }
    const auto residual = [&](std::size_t i, const DiscountCurve<double>& foreignDiscount) {
        return std::visit(
            [&](const auto& value) -> double {
                using PillarT = std::decay_t<decltype(value)>;
                if constexpr (std::is_same_v<PillarT, FxSwapPillar>) {
                    const double t = nodeTimes[i];
                    const double model = value.isFxBaseCollateral
                                             ? value.spot * domesticDiscount.discount(t) /
                                                   foreignDiscount.discount(t)
                                             : value.spot * foreignDiscount.discount(t) /
                                                   domesticDiscount.discount(t);
                    const double implied =
                        value.convention == QuoteConvention::Points ? model - value.spot : model;
                    return implied - fxPillarTargetQuote(value);
                } else {
                    return detail::evaluateXccyPillarTimes(xccyTimes[i], value, foreignDiscount,
                                                           foreignForecastAt(foreignDiscount),
                                                           domesticDiscount, domesticForecast) -
                           value.spread;
                }
            },
            pillars[i]);
    };
    const std::vector<double> zeros =
        detail::bootstrapNodesByBrent(space, scheme, tension, switchIndex, accuracy, nodeTimes,
                                      residual, "bootstrapMixedXccyDiscountCurve");
    return DiscountCurve<double>(referenceDate, maturityDates, zeroDayCounter, zeros, space, scheme,
                                 tension, switchIndex);
}

} // namespace detail

/// Self-forecast mixed bootstrap: xccy pillars forecast their floating legs on
/// the trial foreign discount curve (OIS-style swaps), while FX swap pillars
/// use the trial curve as the foreign side of the CIP ratio.
template <XccyForecastCurve DomesticForecastT>
DiscountCurve<double> bootstrapMixedXccyDiscountCurve(
    const DiscountCurve<double>& domesticDiscount, const DomesticForecastT& domesticForecast,
    const datetime::Date& referenceDate, const datetime::DayCounter& zeroDayCounter,
    InterpolationSpace space, InterpolationScheme scheme,
    const std::vector<XccyMixedPillar>& pillars, double accuracy = 1e-14, double tension = 0.0,
    int switchIndex = 1) {
    const auto foreignForecastAt =
        [](const DiscountCurve<double>& trial) -> const DiscountCurve<double>& { return trial; };
    return detail::bootstrapMixedXccyDiscountCurveImpl(
        domesticDiscount, domesticForecast, referenceDate, zeroDayCounter, space, scheme, pillars,
        foreignForecastAt, accuracy, tension, switchIndex);
}

/// Explicit-forecast mixed bootstrap: xccy pillars use `foreignForecast` for
/// their foreign floating leg, while FX swap pillars still reference the trial
/// foreign discount curve through the CIP ratio.
template <XccyForecastCurve ForeignForecastT, XccyForecastCurve DomesticForecastT>
DiscountCurve<double> bootstrapMixedXccyDiscountCurve(
    const DiscountCurve<double>& domesticDiscount, const DomesticForecastT& domesticForecast,
    const ForeignForecastT& foreignForecast, const datetime::Date& referenceDate,
    const datetime::DayCounter& zeroDayCounter, InterpolationSpace space,
    InterpolationScheme scheme, const std::vector<XccyMixedPillar>& pillars,
    double accuracy = 1e-14, double tension = 0.0, int switchIndex = 1) {
    const auto foreignForecastAt =
        [&foreignForecast](const DiscountCurve<double>&) -> const ForeignForecastT& {
        return foreignForecast;
    };
    return detail::bootstrapMixedXccyDiscountCurveImpl(
        domesticDiscount, domesticForecast, referenceDate, zeroDayCounter, space, scheme, pillars,
        foreignForecastAt, accuracy, tension, switchIndex);
}

/// One coupled pillar: the foreign par basis quote (discounted on the xccy
/// curve) and the xccy basis quote (forecast on the foreign spread curve).
struct XccyCoupledPillar {
    XccyPillar xccy;
    BasisPillar basis; ///< Foreign IBOR-vs-OIS par basis, xccy-collateral discounting
};

struct XccyCoupledResult {
    DiscountCurve<double> foreignDiscount;
    SpreadCurve<double> foreignSpread;
    int passes = 0;
    double update = 0.0;
    bool converged = false;         ///< Fixed-point iteration met its update tolerance
    bool usedJointFallback = false; ///< LM fallback ran after a failed fixed point
};

/// Coupled bootstrap of the foreign discount curve (domestic collateral) and
/// the foreign IBOR-vs-OIS spread curve. Each pass re-bootstraps the spread
/// curve discounting on the current discount curve, then the discount curve
/// forecasting on the new spread curve; iterate until the max move of both
/// curves (measured on discount factors at the pillar maturities) is below the
/// tolerance. Xccy pillars may carry either notional convention; the
/// resetting-notional decomposition enters only through the discount-curve
/// residual.
template <XccyForecastCurve DomesticForecastT>
XccyCoupledResult bootstrapXccyCoupled(const DiscountCurve<double>& domesticDiscount,
                                       const DomesticForecastT& domesticForecast,
                                       std::shared_ptr<const DiscountCurve<double>> foreignBase,
                                       const datetime::Date& referenceDate,
                                       const datetime::DayCounter& zeroDayCounter,
                                       InterpolationSpace space, InterpolationScheme scheme,
                                       const std::vector<XccyCoupledPillar>& pillars,
                                       double accuracy = 1e-14, double tension = 0.0,
                                       int switchIndex = 1, math::FixedPointOptions options = {}) {
    if (pillars.empty()) {
        throw std::invalid_argument("bootstrapXccyCoupled: no pillars");
    }
    if (!foreignBase) {
        throw std::invalid_argument("bootstrapXccyCoupled: null foreign base");
    }
    std::vector<XccyPillar> xccyPillars;
    std::vector<BasisPillar> basisPillars;
    std::vector<double> xccyTimes{0.0};
    std::vector<double> basisTimes{0.0};
    for (const XccyCoupledPillar& pillar : pillars) {
        xccyPillars.push_back(pillar.xccy);
        basisPillars.push_back(pillar.basis);
        xccyTimes.push_back(datetime::yearFraction(
            referenceDate,
            pillar.xccy.foreignCalendar.adjust(pillar.xccy.maturity,
                                               pillar.xccy.foreignBusinessDayConvention),
            zeroDayCounter));
        basisTimes.push_back(datetime::yearFraction(
            referenceDate,
            pillar.basis.calendar.adjust(pillar.basis.maturity, pillar.basis.businessDayConvention),
            zeroDayCounter));
    }

    struct State {
        SpreadCurve<double> spread;
        DiscountCurve<double> discount;
    };
    SpreadCurve<double> spreadInit = bootstrapSpreadCurve(
        foreignBase, referenceDate, zeroDayCounter, scheme, basisPillars, accuracy, tension);
    DiscountCurve<double> discountInit = bootstrapXccyDiscountCurve(
        domesticDiscount, domesticForecast, spreadInit, referenceDate, zeroDayCounter, space,
        scheme, xccyPillars, accuracy, tension, switchIndex);
    State state{std::move(spreadInit), std::move(discountInit)};

    const auto pass = [&](State& current) {
        current.spread = bootstrapSpreadCurve(foreignBase, referenceDate, zeroDayCounter, scheme,
                                              basisPillars, accuracy, tension, &current.discount);
        current.discount = bootstrapXccyDiscountCurve(
            domesticDiscount, domesticForecast, current.spread, referenceDate, zeroDayCounter,
            space, scheme, xccyPillars, accuracy, tension, switchIndex);
    };
    const auto updateNorm = [&](const State& before, const State& after) {
        double norm = 0.0;
        for (const double t : basisTimes) {
            const double spreadMove =
                std::abs(before.spread.discount(t) - after.spread.discount(t));
            if (spreadMove > norm) {
                norm = spreadMove;
            }
        }
        for (const double t : xccyTimes) {
            const double discountMove =
                std::abs(before.discount.discount(t) - after.discount.discount(t));
            if (discountMove > norm) {
                norm = discountMove;
            }
        }
        return norm;
    };

    const math::FixedPointResult iteration =
        math::fixedPointIterate(pass, state, updateNorm, options);
    const auto maxResidualAt = [&](const State& current) {
        double maxResidual = 0.0;
        for (std::size_t i = 0; i < xccyPillars.size(); ++i) {
            const double basisResidual =
                impliedBasisSpread(current.spread, basisPillars[i], referenceDate, zeroDayCounter,
                                   &current.discount) -
                basisPillars[i].spread;
            if (std::abs(basisResidual) > maxResidual) {
                maxResidual = std::abs(basisResidual);
            }
            const double xccyResidual =
                impliedXccyBasisSpread(current.discount, current.spread, domesticDiscount,
                                       domesticForecast, xccyPillars[i], referenceDate,
                                       zeroDayCounter) -
                xccyPillars[i].spread;
            if (std::abs(xccyResidual) > maxResidual) {
                maxResidual = std::abs(xccyResidual);
            }
        }
        return maxResidual;
    };

    bool usedFallback = false;
    if (!iteration.converged || !(maxResidualAt(state) < 1e-9)) {
        usedFallback = true;
        const std::size_t unknowns = xccyPillars.size();
        std::vector<double> point(2 * unknowns);
        for (std::size_t i = 0; i < unknowns; ++i) {
            point[i] = state.spread.spread(basisTimes[i + 1]);
            point[unknowns + i] = state.discount.zeros()[i + 1];
        }
        const auto residuals = [&](const std::vector<double>& x, std::vector<double>& out) {
            std::vector<double> times{0.0};
            std::vector<double> spreads{0.0};
            std::vector<double> discountTimes{0.0};
            std::vector<double> discountZeros{0.0};
            for (std::size_t i = 0; i < unknowns; ++i) {
                times.push_back(basisTimes[i + 1]);
                spreads.push_back(x[i]);
                discountTimes.push_back(xccyTimes[i + 1]);
                discountZeros.push_back(x[unknowns + i]);
            }
            const SpreadCurve<double> spread(foreignBase, times, spreads, scheme, tension);
            const DiscountCurve<double> discount(discountTimes, discountZeros, space, scheme,
                                                 tension, switchIndex);
            out.resize(2 * unknowns);
            for (std::size_t i = 0; i < unknowns; ++i) {
                out[i] = impliedBasisSpread(spread, basisPillars[i], referenceDate, zeroDayCounter,
                                            &discount) -
                         basisPillars[i].spread;
                out[unknowns + i] =
                    impliedXccyBasisSpread(discount, spread, domesticDiscount, domesticForecast,
                                           xccyPillars[i], referenceDate, zeroDayCounter) -
                    xccyPillars[i].spread;
            }
        };
        math::levenbergMarquardt(residuals, point);
        std::vector<double> check;
        residuals(point, check);
        double maxResidual = 0.0;
        for (const double value : check) {
            if (std::abs(value) > maxResidual) {
                maxResidual = std::abs(value);
            }
        }
        if (!(maxResidual < 1e-9)) {
            throw std::runtime_error(
                "bootstrapXccyCoupled: joint LM fallback did not converge (max residual " +
                std::to_string(maxResidual) + ")");
        }
        std::vector<double> times{0.0};
        std::vector<double> spreads{0.0};
        std::vector<double> discountTimes{0.0};
        std::vector<double> discountZeros{0.0};
        for (std::size_t i = 0; i < unknowns; ++i) {
            times.push_back(basisTimes[i + 1]);
            spreads.push_back(point[i]);
            discountTimes.push_back(xccyTimes[i + 1]);
            discountZeros.push_back(point[unknowns + i]);
        }
        state.spread = SpreadCurve<double>(foreignBase, times, spreads, scheme, tension);
        state.discount = DiscountCurve<double>(discountTimes, discountZeros, space, scheme, tension,
                                               switchIndex);
    }
    return XccyCoupledResult{std::move(state.discount), std::move(state.spread), iteration.passes,
                             iteration.update,          iteration.converged,     usedFallback};
}

} // namespace quantape::markets
