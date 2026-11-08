#pragma once

#include "quantape/datetime/Date.h"
#include "quantape/datetime/DayCounter.h"
#include "quantape/datetime/TimeConversion.h"
#include "quantape/instruments/BootstrapInstrument.h"
#include "quantape/instruments/XccyInstruments.h"
#include "quantape/markets/Curves/BootstrapInstrument.h"
#include "quantape/markets/Curves/CurveBuilder.h"
#include "quantape/markets/Curves/DiscountCurve.h"
#include "quantape/markets/Curves/FxSwapBuilder.h"
#include "quantape/math/Optimization/LevenbergMarquardt.h"
#include "quantape/math/Solvers/FixedPointIterator.h"
#include "quantape/pricing/Xccy.h"

#include <cmath>
#include <concepts>
#include <cstddef>
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
 *   `sum_f tau D_f (f_f + b) - (D_f(0) - D_f(T)) = sum_d tau D_d f_d - (D_d(0) - D_d(T))`
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
 * annuity keeps its `tau * D(pay)` definition on both legs. The leg algebra
 * lives once in `quantape/pricing/XccyMath.h`; the pillar is the plain-data
 * `instruments::XccyBasisSwap` and evaluates its own `impliedQuote` over the
 * curve sets.
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

/// Cross-currency basis pillar: the plain-data `instruments::XccyBasisSwap`,
/// kept under its original name for the curve-side consumers.
using XccyPillar = instruments::XccyBasisSwap;

/// Notional convention of the cross-currency basis legs.
using XccyNotionalMode = pricing::XccyNotionalMode;

/// Model-implied basis spread (foreign leg spread) against a foreign discount
/// curve, with both forecast curves given. Handles constant and resetting
/// notionals; on a quoted leg that resets, the spread accrues on the reset
/// notional, so its annuity is `sum tau D(pay) adjN` (adjN = D_other/D_own).
template <XccyForecastCurve ForeignForecastT, XccyForecastCurve DomesticForecastT>
double impliedXccyBasisSpread(const DiscountCurve<double>& foreignDiscount,
                              const ForeignForecastT& foreignForecast,
                              const DiscountCurve<double>& domesticDiscount,
                              const DomesticForecastT& domesticForecast, const XccyPillar& pillar,
                              const datetime::Date& referenceDate,
                              const datetime::DayCounter& zeroDayCounter) {
    XccyPillar instrument = pillar;
    instrument.prepare(referenceDate, zeroDayCounter);
    const XccySet<DiscountCurve<double>, ForeignForecastT, DiscountCurve<double>, DomesticForecastT>
        curves{foreignDiscount, foreignForecast, domesticDiscount, domesticForecast};
    return pricing::impliedXccyBasisSpread<double>(instrument, curves);
}

/// Sequential exact-fit bootstrap of the foreign discount curve (USD-collateral)
/// from constant- or resetting-notional xccy basis pillars; domestic curves
/// and the foreign forecast curve stay frozen. Each pillar evaluates its own
/// `impliedQuote` over the trial foreign discount curve, and risk dates come
/// from the instrument's calendar-adjusted maturity.
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
    std::vector<XccyPillar> instruments(count);
    for (std::size_t i = 0; i < count; ++i) {
        instruments[i] = pillars[i];
        maturityDates[i] = instruments[i].date();
        nodeTimes[i] = datetime::yearFraction(referenceDate, maturityDates[i], zeroDayCounter);
        if (!(nodeTimes[i] > (i == 0 ? 0.0 : nodeTimes[i - 1]))) {
            throw std::invalid_argument(
                "bootstrapXccyDiscountCurve: maturities must be strictly increasing");
        }
    }
    // The schedules are schedule-fixed, so resolve them once per pillar
    // instead of rebuilding them at every Brent step.
    for (std::size_t i = 0; i < count; ++i) {
        instruments[i].prepare(referenceDate, zeroDayCounter);
    }
    const auto residual = [&](std::size_t i, const DiscountCurve<double>& trial) {
        const XccySet<DiscountCurve<double>, ForeignForecastT, DiscountCurve<double>,
                      DomesticForecastT>
            curves{trial, foreignForecast, domesticDiscount, domesticForecast};
        return pricing::impliedXccyBasisSpread<double>(instruments[i], curves) -
               instruments[i].target();
    };
    const std::vector<double> zeros =
        detail::bootstrapNodesByBrent(space, scheme, tension, switchIndex, accuracy, nodeTimes,
                                      residual, "bootstrapXccyDiscountCurve");
    return DiscountCurve<double>(referenceDate, maturityDates, zeroDayCounter, zeros, space, scheme,
                                 tension, switchIndex);
}

/// One pillar of a mixed `XccyBasis`-role ladder: FX forward points at the
/// short end and cross-currency basis swaps at the long end. The `Ladder`
/// exposes the alternative's risk date, target and model quote.
using XccyMixedPillar = Ladder<FxSwapPillar, XccyPillar>;

namespace detail {

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
        maturityDates[i] = pillars[i].date();
        nodeTimes[i] = datetime::yearFraction(referenceDate, maturityDates[i], zeroDayCounter);
        if (!(nodeTimes[i] > (i == 0 ? 0.0 : nodeTimes[i - 1]))) {
            throw std::invalid_argument(
                "bootstrapMixedXccyDiscountCurve: maturities must be strictly increasing");
        }
    }
    // Resolve the xccy leg schedules once per pillar; the FX pillars are
    // closed-form on their precomputed node time, measured from the reference
    // date at unit point scale like the dedicated FX bootstrap.
    std::vector<FxSwapPillar> fxInstruments(count);
    std::vector<XccyPillar> xccyInstruments(count);
    for (std::size_t i = 0; i < count; ++i) {
        std::visit(
            [&](const auto& value) {
                using PillarT = std::decay_t<decltype(value)>;
                if constexpr (std::is_same_v<PillarT, FxSwapPillar>) {
                    fxInstruments[i] = value;
                    fxInstruments[i].tradeDate = referenceDate;
                    fxInstruments[i].zeroDayCounter = zeroDayCounter;
                    fxInstruments[i].pointsScale = 1.0;
                } else {
                    xccyInstruments[i] = value;
                    xccyInstruments[i].prepare(referenceDate, zeroDayCounter);
                }
            },
            pillars[i].instrument());
    }
    const auto residual = [&](std::size_t i, const DiscountCurve<double>& foreignDiscount) {
        return std::visit(
            [&](const auto& value) -> double {
                using PillarT = std::decay_t<decltype(value)>;
                if constexpr (std::is_same_v<PillarT, FxSwapPillar>) {
                    if (value.isFxBaseCollateral) {
                        const FxSet<DiscountCurve<double>, DiscountCurve<double>> curves{
                            domesticDiscount, foreignDiscount};
                        return fxInstruments[i].template impliedQuote<double>(curves) -
                               fxInstruments[i].target();
                    }
                    const FxSet<DiscountCurve<double>, DiscountCurve<double>> curves{
                        foreignDiscount, domesticDiscount};
                    return fxInstruments[i].template impliedQuote<double>(curves) -
                           fxInstruments[i].target();
                } else {
                    using ForeignForecastAtT =
                        std::decay_t<decltype(foreignForecastAt(foreignDiscount))>;
                    const ForeignForecastAtT& foreignForecast = foreignForecastAt(foreignDiscount);
                    const XccySet<DiscountCurve<double>, ForeignForecastAtT, DiscountCurve<double>,
                                  DomesticForecastT>
                        curves{foreignDiscount, foreignForecast, domesticDiscount,
                               domesticForecast};
                    return xccyInstruments[i].template impliedQuote<double>(curves) -
                           xccyInstruments[i].target();
                }
            },
            pillars[i].instrument());
    };
    const std::vector<double> zeros =
        detail::bootstrapNodesByBrent(space, scheme, tension, switchIndex, accuracy, nodeTimes,
                                      residual, "bootstrapMixedXccyDiscountCurve");
    return DiscountCurve<double>(referenceDate, maturityDates, zeroDayCounter, zeros, space, scheme,
                                 tension, switchIndex);
}

} // namespace detail

/// Self-forecast mixed bootstrap: FX swap pillars use the trial curve as the
/// foreign side of the CIP ratio.
///
/// A basis-quoted `XccySwap` pillar is only admissible here when the foreign
/// forecast curve is distinct from the foreign discount curve. With forecast
/// == discount the floating leg's coupons telescope to the notional term for
/// every curve, so the leg is worth zero and the par basis measures the foreign
/// curve through the annuity alone -- far too weakly to solve a pillar. Use the
/// explicit-forecast overload, or `bootstrapXccyCoupled` (which bootstraps the
/// foreign discount curve together with an IBOR-vs-OIS spread curve).
template <XccyForecastCurve DomesticForecastT>
DiscountCurve<double> bootstrapMixedXccyDiscountCurve(
    const DiscountCurve<double>& domesticDiscount, const DomesticForecastT& domesticForecast,
    const datetime::Date& referenceDate, const datetime::DayCounter& zeroDayCounter,
    InterpolationSpace space, InterpolationScheme scheme,
    const std::vector<XccyMixedPillar>& pillars, double accuracy = 1e-14, double tension = 0.0,
    int switchIndex = 1) {
    for (const XccyMixedPillar& pillar : pillars) {
        if (!std::holds_alternative<FxSwapPillar>(pillar.instrument())) {
            throw std::invalid_argument(
                "bootstrapMixedXccyDiscountCurve: a basis-quoted XccySwap pillar requires a "
                "foreign forecast curve distinct from the foreign discount curve; use the "
                "explicit-forecast overload or bootstrapXccyCoupled");
        }
    }
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
using XccyCoupledPillar = instruments::XccyCoupledPillar;

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
        xccyTimes.push_back(
            datetime::yearFraction(referenceDate, pillar.xccy.date(), zeroDayCounter));
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
