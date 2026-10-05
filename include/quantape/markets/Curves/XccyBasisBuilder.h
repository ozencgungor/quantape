#pragma once

#include "quantape/datetime/Calendar.h"
#include "quantape/datetime/Date.h"
#include "quantape/datetime/Period.h"
#include "quantape/datetime/Schedule.h"
#include "quantape/datetime/TimeConversion.h"
#include "quantape/markets/Curves/CurveBuilder.h"
#include "quantape/markets/Curves/DiscountCurve.h"
#include "quantape/math/Optimization/LevenbergMarquardt.h"
#include "quantape/math/Solvers/BrentSolver.h"
#include "quantape/math/Solvers/FixedPointIterator.h"

#include <cmath>
#include <concepts>
#include <cstddef>
#include <memory>
#include <stdexcept>
#include <vector>

namespace quantape::markets {
/**
 * @file XccyBasisBuilder.h
 * @brief Const-notional cross-currency basis bootstrap
 *
 * Bootstraps the *foreign discount curve collateralized in the domestic
 * currency* from const-notional cross-currency basis swap quotes, given the
 * domestic discount and both forecast curves. Both legs exchange notional at
 * start and maturity and the FX spot cancels, so the par condition is
 *
 *   `sum_f tau D_f (f_f + b) + (D_f(0) - D_f(T)) = sum_d tau D_d f_d + (D_d(0) - D_d(T))`
 *
 * with the basis spread `b` quoted on the foreign leg. Each pillar solves one
 * foreign zero node (foreign discounts at intermediate coupon dates are
 * interpolated, so arbitrary schedules are supported). Both legs support
 * stubs, payment lags and per-leg business-day conventions.
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

/// Const-notional xccy basis pillar with both leg conventions.
struct XccyPillar {
    datetime::Date maturity;
    double spread = 0.0;            ///< Basis spread (decimal)
    bool spreadOnForeignLeg = true; ///< Which leg carries the quoted basis
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

/// Model-implied const-notional basis spread (foreign leg spread) against a
/// foreign discount curve, with both forecast curves given.
template <XccyForecastCurve ForeignForecastT, XccyForecastCurve DomesticForecastT>
double impliedXccyBasisSpread(const DiscountCurve<double>& foreignDiscount,
                              const ForeignForecastT& foreignForecast,
                              const DiscountCurve<double>& domesticDiscount,
                              const DomesticForecastT& domesticForecast, const XccyPillar& pillar,
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

    const auto legValue = [&](const datetime::Schedule& schedule,
                              const datetime::DayCounter& accrualDayCounter,
                              const DiscountCurve<double>& discount, const auto& forecast,
                              int paymentLag, datetime::BusinessDayConvention businessDayConvention,
                              double& annuity) {
        const std::vector<datetime::Date>& dates = schedule.dates();
        double coupons = 0.0;
        annuity = 0.0;
        for (std::size_t k = 1; k < dates.size(); ++k) {
            const double tau = datetime::yearFraction(dates[k - 1], dates[k], accrualDayCounter);
            if (!(tau > 0.0)) {
                throw std::invalid_argument("impliedXccyBasisSpread: non-positive accrual");
            }
            const double tPrevious =
                datetime::yearFraction(referenceDate, dates[k - 1], zeroDayCounter);
            const double tAccrual = datetime::yearFraction(referenceDate, dates[k], zeroDayCounter);
            const datetime::Date payDate = schedule.calendar().advance(
                dates[k], datetime::Period(paymentLag, datetime::TimeUnit::Days),
                businessDayConvention);
            const double t = datetime::yearFraction(referenceDate, payDate, zeroDayCounter);
            const double df = discount.discount(t);
            const double forward =
                (forecast.discount(tPrevious) / forecast.discount(tAccrual) - 1.0) / tau;
            coupons += tau * df * forward;
            annuity += tau * df;
        }
        const double tStart = datetime::yearFraction(referenceDate, dates.front(), zeroDayCounter);
        const double tEnd = datetime::yearFraction(referenceDate, dates.back(), zeroDayCounter);
        const double notional = discount.discount(tStart) - discount.discount(tEnd);
        return coupons + notional;
    };

    double foreignAnnuity = 0.0;
    const double foreignValue =
        legValue(foreignSchedule, pillar.foreignDayCounter, foreignDiscount, foreignForecast,
                 pillar.foreignPaymentLag, pillar.foreignBusinessDayConvention, foreignAnnuity);
    double domesticAnnuity = 0.0;
    const double domesticValue =
        legValue(domesticSchedule, pillar.domesticDayCounter, domesticDiscount, domesticForecast,
                 pillar.domesticPaymentLag, pillar.domesticBusinessDayConvention, domesticAnnuity);
    if (!(foreignAnnuity > 0.0) || !(domesticAnnuity > 0.0)) {
        throw std::invalid_argument("impliedXccyBasisSpread: non-positive annuity");
    }
    if (pillar.spreadOnForeignLeg) {
        return (domesticValue - foreignValue) / foreignAnnuity;
    }
    return (foreignValue - domesticValue) / domesticAnnuity;
}

/// Sequential exact-fit bootstrap of the foreign discount curve (USD-collateral)
/// from const-notional xccy basis pillars; domestic curves and the foreign
/// forecast curve stay frozen.
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

    const quantape::math::BrentSolver<double> solver;
    std::vector<double> zeros(count, 0.0);
    const auto solveNodes = [&](bool multiPass) {
        for (int pass = 0; pass < (multiPass ? 50 : 1); ++pass) {
            const std::vector<double> previous = zeros;
            double lastMove = 0.0;
            for (std::size_t i = 0; i < count; ++i) {
                const XccyPillar& pillar = pillars[i];
                const std::size_t lastNode = multiPass && pass == 0 ? i : count - 1;
                const auto residual = [&](double trialZero) {
                    std::vector<double> trialTimes{0.0};
                    std::vector<double> trialZeros{0.0};
                    for (std::size_t j = 0; j <= lastNode; ++j) {
                        trialTimes.push_back(nodeTimes[j]);
                        trialZeros.push_back(j == i ? trialZero : zeros[j]);
                    }
                    const DiscountCurve<double> trial(trialTimes, trialZeros, space, scheme,
                                                      tension, switchIndex);
                    return impliedXccyBasisSpread(trial, foreignForecast, domesticDiscount,
                                                  domesticForecast, pillar, referenceDate,
                                                  zeroDayCounter) -
                           pillar.spread;
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
                    throw std::runtime_error(
                        "bootstrapXccyDiscountCurve: failed to bracket pillar " +
                        std::to_string(i));
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
            const double check =
                impliedXccyBasisSpread(curve, foreignForecast, domesticDiscount, domesticForecast,
                                       pillars[i], referenceDate, zeroDayCounter) -
                pillars[i].spread;
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
        throw std::runtime_error("bootstrapXccyDiscountCurve: fixed point did not converge");
    }
    return DiscountCurve<double>(referenceDate, maturityDates, zeroDayCounter, zeros, space, scheme,
                                 tension, switchIndex);
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
/// tolerance.
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
