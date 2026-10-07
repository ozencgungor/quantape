/**
 * @file test_xccy.cpp
 * @brief Cross-currency basis, FX swap and unified stack-risk gates
 *
 * Double-only cross-currency risk: the const-notional, spread-forecast,
 * coupled and resetting-notional bootstraps, the analytic row/stack engines,
 * the resetting-notional goldens and the FX swap point/outright CIP branches.
 */

#include "quantape/markets/Curves/CurveRisk.h"
#include "quantape/markets/Curves/FxSwapBuilder.h"
#include "quantape/markets/Curves/SpreadCurve.h"
#include "quantape/markets/Curves/StackRisk.h"
#include "quantape/markets/Curves/XccyBasisBuilder.h"
#include "quantape/markets/Data/FXRate.h"
#include "quantape/markets/Descriptors/FXDescriptor.h"
#include "quantape/math/LinearAlgebra/DenseSolve.h"
#include "quantape/math/Optimization/LevenbergMarquardt.h"
#include "quantape/math/Solvers/FixedPointIterator.h"

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <variant>
#include <vector>

#include "support/GtestSupport.h"

using namespace quantape;
using markets::DiscountCurve;
using markets::InterpolationScheme;
using markets::InterpolationSpace;

namespace {

/// Frozen copy of the pre-migration cross-currency evaluators and bootstrap
/// drivers, used as the old-vs-new bitwise oracle. Kept independent of the
/// library so the gate compares arithmetic rather than re-running the migrated
/// implementation.
namespace legacy {

enum class NotionalMode : std::uint8_t { Const, MtM };

struct XccyPillar {
    datetime::Date maturity;
    double spread = 0.0;
    bool spreadOnForeignLeg = true;
    NotionalMode notional = NotionalMode::Const;
    bool resetForeignLeg = true;
    datetime::Period foreignTenor{3, datetime::TimeUnit::Months};
    datetime::Calendar foreignCalendar{};
    datetime::DayCounter foreignDayCounter{datetime::DayCount::Actual360};
    datetime::Period domesticTenor{3, datetime::TimeUnit::Months};
    datetime::Calendar domesticCalendar{};
    datetime::DayCounter domesticDayCounter{datetime::DayCount::Actual360};
    int foreignPaymentLag = 0;
    int domesticPaymentLag = 0;
    datetime::BusinessDayConvention foreignBusinessDayConvention =
        datetime::BusinessDayConvention::ModifiedFollowing;
    datetime::BusinessDayConvention domesticBusinessDayConvention =
        datetime::BusinessDayConvention::ModifiedFollowing;
};

struct XccyCouponTimes {
    double tau = 0.0;
    double tPrevious = 0.0;
    double tAccrual = 0.0;
    double tPay = 0.0;
};

struct XccyLegTimes {
    std::vector<XccyCouponTimes> coupons;
    double tStart = 0.0;
    double tEnd = 0.0;
};

struct XccyPillarTimes {
    XccyLegTimes foreign;
    XccyLegTimes domestic;
    bool foreignResets = false;
    bool domesticResets = false;
};

XccyLegTimes makeXccyLegTimes(const datetime::Schedule& schedule,
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

XccyPillarTimes makeXccyPillarTimes(const XccyPillar& pillar, const datetime::Date& referenceDate,
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
    times.foreignResets = pillar.notional == NotionalMode::MtM && pillar.resetForeignLeg;
    times.domesticResets = pillar.notional == NotionalMode::MtM && !pillar.resetForeignLeg;
    return times;
}

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

template <typename ForeignForecastT, typename DomesticForecastT>
double impliedXccyBasisSpread(const DiscountCurve<double>& foreignDiscount,
                              const ForeignForecastT& foreignForecast,
                              const DiscountCurve<double>& domesticDiscount,
                              const DomesticForecastT& domesticForecast, const XccyPillar& pillar,
                              const datetime::Date& referenceDate,
                              const datetime::DayCounter& zeroDayCounter) {
    const XccyPillarTimes times = makeXccyPillarTimes(pillar, referenceDate, zeroDayCounter);
    return evaluateXccyPillarTimes(times, pillar, foreignDiscount, foreignForecast,
                                   domesticDiscount, domesticForecast);
}

template <typename ForeignForecastT, typename DomesticForecastT>
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
    std::vector<XccyPillarTimes> pillarTimes(count);
    for (std::size_t i = 0; i < count; ++i) {
        pillarTimes[i] = makeXccyPillarTimes(pillars[i], referenceDate, zeroDayCounter);
    }
    const auto residual = [&](std::size_t i, const DiscountCurve<double>& trial) {
        return evaluateXccyPillarTimes(pillarTimes[i], pillars[i], trial, foreignForecast,
                                       domesticDiscount, domesticForecast) -
               pillars[i].spread;
    };
    const std::vector<double> zeros =
        markets::detail::bootstrapNodesByBrent(space, scheme, tension, switchIndex, accuracy,
                                               nodeTimes, residual, "bootstrapXccyDiscountCurve");
    return DiscountCurve<double>(referenceDate, maturityDates, zeroDayCounter, zeros, space, scheme,
                                 tension, switchIndex);
}

using MixedPillar = std::variant<markets::FxSwapPillar, XccyPillar>;

datetime::Date mixedPillarMaturity(const MixedPillar& pillar) {
    return std::visit(
        [](const auto& value) -> datetime::Date {
            using PillarT = std::decay_t<decltype(value)>;
            if constexpr (std::is_same_v<PillarT, markets::FxSwapPillar>) {
                return value.maturity;
            } else {
                return value.foreignCalendar.adjust(value.maturity,
                                                    value.foreignBusinessDayConvention);
            }
        },
        pillar);
}

template <typename ForeignForecastProvider>
DiscountCurve<double> bootstrapMixedXccyDiscountCurveImpl(
    const DiscountCurve<double>& domesticDiscount, const DiscountCurve<double>& domesticForecast,
    const datetime::Date& referenceDate, const datetime::DayCounter& zeroDayCounter,
    InterpolationSpace space, InterpolationScheme scheme, const std::vector<MixedPillar>& pillars,
    const ForeignForecastProvider& foreignForecastAt, double accuracy, double tension,
    int switchIndex) {
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
    std::vector<XccyPillarTimes> xccyTimes(count);
    for (std::size_t i = 0; i < count; ++i) {
        std::visit(
            [&](const auto& value) {
                using PillarT = std::decay_t<decltype(value)>;
                if constexpr (std::is_same_v<PillarT, XccyPillar>) {
                    xccyTimes[i] = makeXccyPillarTimes(value, referenceDate, zeroDayCounter);
                }
            },
            pillars[i]);
    }
    const auto residual = [&](std::size_t i, const DiscountCurve<double>& foreignDiscount) {
        return std::visit(
            [&](const auto& value) -> double {
                using PillarT = std::decay_t<decltype(value)>;
                if constexpr (std::is_same_v<PillarT, markets::FxSwapPillar>) {
                    const double t = nodeTimes[i];
                    const double model = value.isFxBaseCollateral
                                             ? value.spot * domesticDiscount.discount(t) /
                                                   foreignDiscount.discount(t)
                                             : value.spot * foreignDiscount.discount(t) /
                                                   domesticDiscount.discount(t);
                    const double implied = value.convention == markets::QuoteConvention::Points
                                               ? model - value.spot
                                               : model;
                    return implied - markets::fxPillarTargetQuote(value);
                } else {
                    return evaluateXccyPillarTimes(xccyTimes[i], value, foreignDiscount,
                                                   foreignForecastAt(foreignDiscount),
                                                   domesticDiscount, domesticForecast) -
                           value.spread;
                }
            },
            pillars[i]);
    };
    const std::vector<double> zeros = markets::detail::bootstrapNodesByBrent(
        space, scheme, tension, switchIndex, accuracy, nodeTimes, residual,
        "bootstrapMixedXccyDiscountCurve");
    return DiscountCurve<double>(referenceDate, maturityDates, zeroDayCounter, zeros, space, scheme,
                                 tension, switchIndex);
}

DiscountCurve<double> bootstrapMixedXccyDiscountCurve(
    const DiscountCurve<double>& domesticDiscount, const DiscountCurve<double>& domesticForecast,
    const datetime::Date& referenceDate, const datetime::DayCounter& zeroDayCounter,
    InterpolationSpace space, InterpolationScheme scheme, const std::vector<MixedPillar>& pillars,
    double accuracy = 1e-14, double tension = 0.0, int switchIndex = 1) {
    const auto foreignForecastAt =
        [](const DiscountCurve<double>& trial) -> const DiscountCurve<double>& { return trial; };
    return bootstrapMixedXccyDiscountCurveImpl(domesticDiscount, domesticForecast, referenceDate,
                                               zeroDayCounter, space, scheme, pillars,
                                               foreignForecastAt, accuracy, tension, switchIndex);
}

DiscountCurve<double> bootstrapMixedXccyDiscountCurve(
    const DiscountCurve<double>& domesticDiscount, const DiscountCurve<double>& domesticForecast,
    const DiscountCurve<double>& foreignForecast, const datetime::Date& referenceDate,
    const datetime::DayCounter& zeroDayCounter, InterpolationSpace space,
    InterpolationScheme scheme, const std::vector<MixedPillar>& pillars, double accuracy = 1e-14,
    double tension = 0.0, int switchIndex = 1) {
    const auto foreignForecastAt =
        [&foreignForecast](const DiscountCurve<double>&) -> const DiscountCurve<double>& {
        return foreignForecast;
    };
    return bootstrapMixedXccyDiscountCurveImpl(domesticDiscount, domesticForecast, referenceDate,
                                               zeroDayCounter, space, scheme, pillars,
                                               foreignForecastAt, accuracy, tension, switchIndex);
}

struct CoupledPillar {
    XccyPillar xccy;
    markets::BasisPillar basis;
};

markets::XccyCoupledResult bootstrapXccyCoupled(
    const DiscountCurve<double>& domesticDiscount, const DiscountCurve<double>& domesticForecast,
    std::shared_ptr<const DiscountCurve<double>> foreignBase, const datetime::Date& referenceDate,
    const datetime::DayCounter& zeroDayCounter, InterpolationSpace space,
    InterpolationScheme scheme, const std::vector<CoupledPillar>& pillars, double accuracy = 1e-14,
    double tension = 0.0, int switchIndex = 1, math::FixedPointOptions options = {}) {
    if (pillars.empty()) {
        throw std::invalid_argument("bootstrapXccyCoupled: no pillars");
    }
    if (!foreignBase) {
        throw std::invalid_argument("bootstrapXccyCoupled: null foreign base");
    }
    std::vector<XccyPillar> xccyPillars;
    std::vector<markets::BasisPillar> basisPillars;
    std::vector<double> xccyTimes{0.0};
    std::vector<double> basisTimes{0.0};
    for (const CoupledPillar& pillar : pillars) {
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
        markets::SpreadCurve<double> spread;
        DiscountCurve<double> discount;
    };
    markets::SpreadCurve<double> spreadInit = markets::bootstrapSpreadCurve(
        foreignBase, referenceDate, zeroDayCounter, scheme, basisPillars, accuracy, tension);
    DiscountCurve<double> discountInit = bootstrapXccyDiscountCurve(
        domesticDiscount, domesticForecast, spreadInit, referenceDate, zeroDayCounter, space,
        scheme, xccyPillars, accuracy, tension, switchIndex);
    State state{std::move(spreadInit), std::move(discountInit)};

    const auto pass = [&](State& current) {
        current.spread =
            markets::bootstrapSpreadCurve(foreignBase, referenceDate, zeroDayCounter, scheme,
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
                markets::impliedBasisSpread(current.spread, basisPillars[i], referenceDate,
                                            zeroDayCounter, &current.discount) -
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
            const markets::SpreadCurve<double> spread(foreignBase, times, spreads, scheme, tension);
            const DiscountCurve<double> discount(discountTimes, discountZeros, space, scheme,
                                                 tension, switchIndex);
            out.resize(2 * unknowns);
            for (std::size_t i = 0; i < unknowns; ++i) {
                out[i] = markets::impliedBasisSpread(spread, basisPillars[i], referenceDate,
                                                     zeroDayCounter, &discount) -
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
        state.spread = markets::SpreadCurve<double>(foreignBase, times, spreads, scheme, tension);
        state.discount = DiscountCurve<double>(discountTimes, discountZeros, space, scheme, tension,
                                               switchIndex);
    }
    return markets::XccyCoupledResult{std::move(state.discount), std::move(state.spread),
                                      iteration.passes,          iteration.update,
                                      iteration.converged,       usedFallback};
}

} // namespace legacy

bool sameBits(double left, double right) {
    return std::bit_cast<std::uint64_t>(left) == std::bit_cast<std::uint64_t>(right);
}

void testXccyConstNotionalBootstrap() {
    const datetime::Date reference(2026, 9, 29);
    const datetime::DayCounter zeroDc(datetime::DayCount::Actual365Fixed);
    const datetime::Calendar calendar = datetime::Calendar::noHolidays();
    std::vector<datetime::Date> dates{reference};
    for (int years = 1; years <= 5; ++years) {
        dates.push_back(reference.plusYears(years));
    }
    const auto buildCurve = [&](double base, double slope) {
        std::vector<double> zeros;
        for (std::size_t i = 1; i < dates.size(); ++i) {
            const double t = datetime::yearFraction(reference, dates[i], zeroDc);
            zeros.push_back(base + slope * t);
        }
        return DiscountCurve<double>(
            reference, std::vector<datetime::Date>(dates.begin() + 1, dates.end()), zeroDc, zeros,
            InterpolationSpace::LogDiscount, InterpolationScheme::Linear);
    };
    const DiscountCurve<double> domesticDiscount = buildCurve(0.040, 0.0005);
    const DiscountCurve<double> domesticForecast = buildCurve(0.043, 0.0004);
    const DiscountCurve<double> foreignForecast = buildCurve(0.025, 0.0008);
    const DiscountCurve<double> foreignTarget = buildCurve(0.028, 0.0006);

    std::vector<markets::XccyPillar> pillars;
    for (std::size_t i = 1; i < dates.size(); ++i) {
        markets::XccyPillar pillar;
        pillar.maturity = dates[i];
        pillar.foreignTenor = datetime::Period(3, datetime::TimeUnit::Months);
        pillar.foreignCalendar = calendar;
        pillar.foreignDayCounter = datetime::DayCounter(datetime::DayCount::Actual360);
        pillar.domesticTenor = datetime::Period(3, datetime::TimeUnit::Months);
        pillar.domesticCalendar = calendar;
        pillar.domesticDayCounter = datetime::DayCounter(datetime::DayCount::Actual360);
        pillar.spread =
            markets::impliedXccyBasisSpread(foreignTarget, foreignForecast, domesticDiscount,
                                            domesticForecast, pillar, reference, zeroDc);
        EXPECT_TRUE(std::isfinite(pillar.spread));
        EXPECT_LT(std::abs(pillar.spread), 0.05);
        pillars.push_back(pillar);
    }

    const DiscountCurve<double> curve = markets::bootstrapXccyDiscountCurve(
        domesticDiscount, domesticForecast, foreignForecast, reference, zeroDc,
        InterpolationSpace::LogDiscount, InterpolationScheme::Linear, pillars);
    EXPECT_EQ(curve.size(), pillars.size() + 1);
    for (std::size_t i = 0; i < pillars.size(); ++i) {
        CHECK_CLOSE("xccy recovered zero", curve.zeros()[i + 1], foreignTarget.zeros()[i + 1],
                    1e-10);
        CHECK_CLOSE("xccy reprice",
                    markets::impliedXccyBasisSpread(curve, foreignForecast, domesticDiscount,
                                                    domesticForecast, pillars[i], reference,
                                                    zeroDc),
                    pillars[i].spread, 1e-10);
    }

    const std::vector<markets::XccyPillar> unsorted{pillars[2], pillars[1]};
    EXPECT_THROW((void)(markets::bootstrapXccyDiscountCurve(
                     domesticDiscount, domesticForecast, foreignForecast, reference, zeroDc,
                     InterpolationSpace::LogDiscount, InterpolationScheme::Linear, unsorted)),
                 std::invalid_argument);
}

void testXccySpreadForecastsAndSpreadSide() {
    const datetime::Date reference(2026, 9, 29);
    const datetime::DayCounter zeroDc(datetime::DayCount::Actual365Fixed);
    const datetime::Calendar calendar = datetime::Calendar::noHolidays();
    std::vector<datetime::Date> dates{reference};
    for (int years = 1; years <= 5; ++years) {
        dates.push_back(reference.plusYears(years));
    }
    const auto buildCurve = [&](double base, double slope) {
        std::vector<double> zeros;
        for (std::size_t i = 1; i < dates.size(); ++i) {
            zeros.push_back(base + slope * datetime::yearFraction(reference, dates[i], zeroDc));
        }
        return DiscountCurve<double>(
            reference, std::vector<datetime::Date>(dates.begin() + 1, dates.end()), zeroDc, zeros,
            InterpolationSpace::LogDiscount, InterpolationScheme::Linear);
    };
    const DiscountCurve<double> domesticDiscount = buildCurve(0.040, 0.0005);
    const DiscountCurve<double> foreignTarget = buildCurve(0.028, 0.0006);
    auto domesticBase = std::make_shared<DiscountCurve<double>>(buildCurve(0.041, 0.0004));
    auto foreignBase = std::make_shared<DiscountCurve<double>>(buildCurve(0.024, 0.0007));
    std::vector<double> spreadTimes{0.0};
    std::vector<double> domesticSpreads{0.0};
    std::vector<double> foreignSpreads{0.0};
    for (std::size_t i = 1; i < dates.size(); ++i) {
        spreadTimes.push_back(datetime::yearFraction(reference, dates[i], zeroDc));
        domesticSpreads.push_back(0.0005 + 0.0001 * static_cast<double>(i));
        foreignSpreads.push_back(0.0002 + 0.0001 * static_cast<double>(i));
    }
    const markets::SpreadCurve<double> domesticForecast(domesticBase, spreadTimes, domesticSpreads);
    const markets::SpreadCurve<double> foreignForecast(foreignBase, spreadTimes, foreignSpreads);

    // Bench vs spread and spread vs spread forecasts; spread on either leg.
    for (const bool spreadOnForeign : {true, false}) {
        SCOPED_TRACE(spreadOnForeign ? "spread on foreign leg" : "spread on domestic leg");
        std::vector<markets::XccyPillar> pillars;
        for (std::size_t i = 1; i < dates.size(); ++i) {
            markets::XccyPillar pillar;
            pillar.maturity = dates[i];
            pillar.foreignTenor = datetime::Period(7, datetime::TimeUnit::Months);
            pillar.foreignPaymentLag = 2;
            pillar.domesticPaymentLag = 1;
            pillar.foreignBusinessDayConvention = datetime::BusinessDayConvention::Following;
            pillar.foreignCalendar = calendar;
            pillar.foreignDayCounter = datetime::DayCounter(datetime::DayCount::Actual360);
            pillar.domesticTenor = datetime::Period(3, datetime::TimeUnit::Months);
            pillar.domesticCalendar = calendar;
            pillar.domesticDayCounter = datetime::DayCounter(datetime::DayCount::Actual360);
            pillar.spreadOnForeignLeg = spreadOnForeign;
            pillar.spread =
                markets::impliedXccyBasisSpread(foreignTarget, foreignForecast, domesticDiscount,
                                                domesticForecast, pillar, reference, zeroDc);
            EXPECT_TRUE(std::isfinite(pillar.spread));
            pillars.push_back(pillar);
        }
        const DiscountCurve<double> curve = markets::bootstrapXccyDiscountCurve(
            domesticDiscount, domesticForecast, foreignForecast, reference, zeroDc,
            InterpolationSpace::LogDiscount, InterpolationScheme::Linear, pillars);
        for (std::size_t i = 0; i < pillars.size(); ++i) {
            CHECK_CLOSE("xccy spread-forecast recovered", curve.zeros()[i + 1],
                        foreignTarget.zeros()[i + 1], 1e-10);
            CHECK_CLOSE("xccy spread-forecast reprice",
                        markets::impliedXccyBasisSpread(curve, foreignForecast, domesticDiscount,
                                                        domesticForecast, pillars[i], reference,
                                                        zeroDc),
                        pillars[i].spread, 1e-10);
        }
    }
}

void testXccyCoupledBootstrap() {
    const datetime::Date reference(2026, 9, 29);
    const datetime::DayCounter zeroDayCounter(datetime::DayCount::Actual365Fixed);
    const datetime::Calendar calendar = datetime::Calendar::noHolidays();
    std::vector<datetime::Date> dates{reference};
    for (int years = 1; years <= 5; ++years) {
        dates.push_back(reference.plusYears(years));
    }
    const auto buildCurve = [&](double base, double slope, int maxYears) {
        std::vector<datetime::Date> pillarDates;
        std::vector<double> zeros;
        for (int years = 1; years <= maxYears; ++years) {
            const datetime::Date date = reference.plusYears(years);
            pillarDates.push_back(date);
            zeros.push_back(base + slope * datetime::yearFraction(reference, date, zeroDayCounter));
        }
        return DiscountCurve<double>(reference, pillarDates, zeroDayCounter, zeros,
                                     InterpolationSpace::LogDiscount, InterpolationScheme::Linear);
    };
    const DiscountCurve<double> domesticDiscount = buildCurve(0.040, 0.0005, 5);
    const DiscountCurve<double> domesticForecast = domesticDiscount;
    const DiscountCurve<double> xccyTarget = buildCurve(0.030, 0.0005, 3);
    auto foreignBase = std::make_shared<DiscountCurve<double>>(buildCurve(0.025, 0.0005, 3));
    const std::vector<double> spreadTimes{0.0, 1.0, 2.0, 3.0};
    const std::vector<double> spreadNodes{0.0, 0.0010, 0.0013, 0.0016};
    const markets::SpreadCurve<double> spreadTarget(foreignBase, spreadTimes, spreadNodes,
                                                    InterpolationScheme::Linear);

    std::vector<markets::XccyCoupledPillar> pillars;
    for (int years = 1; years <= 3; ++years) {
        const datetime::Date maturity = reference.plusYears(years);
        markets::XccyCoupledPillar coupled;
        coupled.xccy.maturity = maturity;
        coupled.xccy.foreignCalendar = calendar;
        coupled.xccy.domesticCalendar = calendar;
        coupled.xccy.spread = markets::impliedXccyBasisSpread(
            xccyTarget, spreadTarget, domesticDiscount, domesticForecast, coupled.xccy, reference,
            zeroDayCounter);
        coupled.basis.maturity = maturity;
        coupled.basis.calendar = calendar;
        coupled.basis.spread = markets::impliedBasisSpread(spreadTarget, coupled.basis, reference,
                                                           zeroDayCounter, &xccyTarget);
        pillars.push_back(coupled);
    }

    const markets::XccyCoupledResult coupled = markets::bootstrapXccyCoupled(
        domesticDiscount, domesticForecast, foreignBase, reference, zeroDayCounter,
        InterpolationSpace::LogDiscount, InterpolationScheme::Linear, pillars);
    EXPECT_TRUE(coupled.converged);
    EXPECT_GE(coupled.passes, 1);
    EXPECT_TRUE(!coupled.usedJointFallback);
    for (std::size_t i = 0; i < pillars.size(); ++i) {
        CHECK_CLOSE("coupled xccy node", coupled.foreignDiscount.zeros()[i + 1],
                    xccyTarget.zeros()[i + 1], 1e-9);
        CHECK_CLOSE("coupled xccy reprice",
                    markets::impliedXccyBasisSpread(coupled.foreignDiscount, coupled.foreignSpread,
                                                    domesticDiscount, domesticForecast,
                                                    pillars[i].xccy, reference, zeroDayCounter),
                    pillars[i].xccy.spread, 1e-9);
        CHECK_CLOSE("coupled basis reprice",
                    markets::impliedBasisSpread(coupled.foreignSpread, pillars[i].basis, reference,
                                                zeroDayCounter, &coupled.foreignDiscount),
                    pillars[i].basis.spread, 1e-9);
    }
}

void testForecastNodeDateConversion() {
    // Native node times are inverted on the zero clock, so an ACT/360 node is
    // not misread on the ACT/365F offset (and leap-year ACT/365F nodes stay
    // exact).
    const datetime::Date reference(2026, 9, 29);
    const datetime::DayCounter act360(datetime::DayCount::Actual360);
    const datetime::DayCounter act365(datetime::DayCount::Actual365Fixed);
    EXPECT_TRUE(markets::forecastNodeDate(act360, reference, 30.0 / 360.0) ==
                reference.plusDays(30));
    EXPECT_TRUE(markets::forecastNodeDate(act365, reference, 366.0 / 365.0) ==
                reference.plusDays(366));
    EXPECT_TRUE(markets::forecastNodeDate(act365, reference, 0.0) == reference);
}

void testXccyJointFallbackAndRisk() {
    const datetime::Date reference(2026, 9, 29);
    const datetime::DayCounter zeroDayCounter(datetime::DayCount::Actual365Fixed);
    const datetime::Calendar calendar = datetime::Calendar::noHolidays();
    const auto buildCurve = [&](double base, double slope, int maxYears) {
        std::vector<datetime::Date> pillarDates;
        std::vector<double> zeros;
        for (int years = 1; years <= maxYears; ++years) {
            const datetime::Date date = reference.plusYears(years);
            pillarDates.push_back(date);
            zeros.push_back(base + slope * datetime::yearFraction(reference, date, zeroDayCounter));
        }
        return DiscountCurve<double>(reference, pillarDates, zeroDayCounter, zeros,
                                     InterpolationSpace::LogDiscount, InterpolationScheme::Linear);
    };
    const DiscountCurve<double> domesticDiscount = buildCurve(0.040, 0.0005, 5);
    const DiscountCurve<double> xccyTarget = buildCurve(0.030, 0.0005, 3);
    const DiscountCurve<double> foreignForecast = buildCurve(0.024, 0.0007, 3);
    const DiscountCurve<double> domesticForecast = buildCurve(0.043, 0.0004, 5);
    auto foreignBase = std::make_shared<DiscountCurve<double>>(buildCurve(0.025, 0.0005, 3));
    const markets::SpreadCurve<double> spreadTarget(
        foreignBase, std::vector<double>{0.0, 1.0, 2.0, 3.0},
        std::vector<double>{0.0, 0.0010, 0.0013, 0.0016}, InterpolationScheme::Linear);

    std::vector<markets::XccyCoupledPillar> coupledPillars;
    std::vector<markets::XccyPillar> xccyPillars;
    for (int years = 1; years <= 3; ++years) {
        const datetime::Date maturity = reference.plusYears(years);
        markets::XccyCoupledPillar cp;
        cp.xccy.maturity = maturity;
        cp.xccy.foreignCalendar = calendar;
        cp.xccy.domesticCalendar = calendar;
        cp.basis.maturity = maturity;
        cp.basis.calendar = calendar;
        cp.xccy.spread =
            markets::impliedXccyBasisSpread(xccyTarget, spreadTarget, domesticDiscount,
                                            domesticForecast, cp.xccy, reference, zeroDayCounter);
        cp.basis.spread = markets::impliedBasisSpread(spreadTarget, cp.basis, reference,
                                                      zeroDayCounter, &xccyTarget);
        coupledPillars.push_back(cp);
        xccyPillars.push_back(cp.xccy);
    }

    // Joint-LM fallback: one fixed-point pass is not enough, so LM takes over.
    math::FixedPointOptions options;
    options.maxPasses = 1;
    const markets::XccyCoupledResult fallback = markets::bootstrapXccyCoupled(
        domesticDiscount, domesticForecast, foreignBase, reference, zeroDayCounter,
        InterpolationSpace::LogDiscount, InterpolationScheme::Linear, coupledPillars, 1e-14, 0.0, 1,
        options);
    EXPECT_TRUE(fallback.usedJointFallback);
    EXPECT_TRUE(!fallback.converged);
    EXPECT_EQ(fallback.passes, 1);
    for (std::size_t i = 0; i < xccyPillars.size(); ++i) {
        CHECK_CLOSE("joint fallback xccy node", fallback.foreignDiscount.zeros()[i + 1],
                    xccyTarget.zeros()[i + 1], 1e-8);
    }

    // Jacobian rows vs central FD.
    std::vector<double> f;
    std::vector<double> c;
    markets::assembleXccyJacobian(xccyTarget, spreadTarget, domesticDiscount, domesticForecast,
                                  xccyPillars, reference, f, c);
    const std::size_t n = xccyPillars.size();
    const std::size_t m = domesticDiscount.size() - 1;
    const double h = 1e-6;
    const auto foreignBumped = [&](std::size_t node, double delta) {
        std::vector<double> z = xccyTarget.zeros();
        z[node] += delta;
        return DiscountCurve<double>(xccyTarget.times(), z, xccyTarget.space(), xccyTarget.scheme(),
                                     xccyTarget.tension(), xccyTarget.switchIndex());
    };
    const auto domesticBumped = [&](std::size_t node, double delta) {
        std::vector<double> z = domesticDiscount.zeros();
        z[node] += delta;
        return DiscountCurve<double>(domesticDiscount.times(), z, domesticDiscount.space(),
                                     domesticDiscount.scheme(), domesticDiscount.tension(),
                                     domesticDiscount.switchIndex());
    };
    const auto quote = [&](const DiscountCurve<double>& fc, const DiscountCurve<double>& dc) {
        return markets::impliedXccyBasisSpread(fc, spreadTarget, dc, domesticForecast,
                                               xccyPillars[0], reference, zeroDayCounter);
    };
    CHECK_CLOSE("xccy foreign row FD", f[0],
                (quote(foreignBumped(1, h), domesticDiscount) -
                 quote(foreignBumped(1, -h), domesticDiscount)) /
                    (2.0 * h),
                1e-8);
    CHECK_CLOSE(
        "xccy domestic row FD", c[0],
        (quote(xccyTarget, domesticBumped(1, h)) - quote(xccyTarget, domesticBumped(1, -h))) /
            (2.0 * h),
        1e-8);

    // Chain-aware root rows and forecast factor rows.
    auto domesticBase = std::make_shared<DiscountCurve<double>>(domesticDiscount);
    const markets::SpreadCurve<double> domesticForecastSpread(
        domesticBase, std::vector<double>{0.0, 1.0, 2.0, 3.0, 4.0, 5.0},
        std::vector<double>{0.0, 0.0005, 0.0005, 0.0005, 0.0005, 0.0005},
        InterpolationScheme::Linear);
    std::vector<double> fFull;
    std::vector<double> cFull;
    std::vector<double> gFull;
    std::vector<double> hFull;
    markets::xccySwapJacobianRows(xccyTarget, spreadTarget, domesticDiscount,
                                  domesticForecastSpread, xccyPillars[0], reference, zeroDayCounter,
                                  fFull, cFull, gFull, hFull);
    const auto quoteChain = [&](const DiscountCurve<double>& rootCurve) {
        auto parent = std::make_shared<DiscountCurve<double>>(rootCurve);
        const auto& nodes = domesticForecastSpread.spreadNodes();
        const markets::SpreadCurve<double> forecast(parent, nodes.times(), nodes.zeros(),
                                                    nodes.scheme(), nodes.tension());
        return markets::impliedXccyBasisSpread(xccyTarget, spreadTarget, rootCurve, forecast,
                                               xccyPillars[0], reference, zeroDayCounter);
    };
    CHECK_CLOSE("xccy chain root row FD", cFull[0],
                (quoteChain(domesticBumped(1, h)) - quoteChain(domesticBumped(1, -h))) / (2.0 * h),
                1e-8);
    const auto quoteForward = [&](std::size_t node, double delta) {
        const auto& nodes = spreadTarget.spreadNodes();
        std::vector<double> spreads = nodes.zeros();
        spreads[node] += delta;
        const markets::SpreadCurve<double> forecast(spreadTarget.parentPointer(), nodes.times(),
                                                    spreads, nodes.scheme(), nodes.tension());
        return markets::impliedXccyBasisSpread(xccyTarget, forecast, domesticDiscount,
                                               domesticDiscount, xccyPillars[0], reference,
                                               zeroDayCounter);
    };
    CHECK_CLOSE("xccy foreign forecast row FD", gFull[0],
                (quoteForward(1, h) - quoteForward(1, -h)) / (2.0 * h), 1e-8);

    // Implicit cross block dz/dp = -F^{-1} C vs a refit with a bumped root node.
    std::vector<double> rhs(n);
    for (std::size_t i = 0; i < n; ++i) {
        rhs[i] = -c[i * m];
    }
    const std::vector<double> crossColumn = quantape::math::solveDense(f, n, rhs);
    const auto refit = [&](const DiscountCurve<double>& root) {
        return markets::bootstrapXccyDiscountCurve(root, domesticForecast, spreadTarget, reference,
                                                   zeroDayCounter, InterpolationSpace::LogDiscount,
                                                   InterpolationScheme::Linear, xccyPillars);
    };
    const DiscountCurve<double> zPlus = refit(domesticBumped(1, h));
    const DiscountCurve<double> zMinus = refit(domesticBumped(1, -h));
    for (std::size_t i = 0; i < n; ++i) {
        CHECK_CLOSE("xccy cross implicit", crossColumn[i],
                    (zPlus.zeros()[i + 1] - zMinus.zeros()[i + 1]) / (2.0 * h), 1e-6);
    }

    // Stack tree pass smoke test.
    std::vector<markets::CurvePillar> rootPillars;
    for (int years = 1; years <= 5; ++years) {
        markets::CurvePillar pillar;
        pillar.maturity = reference.plusYears(years);
        pillar.kind = markets::PillarKind::OisSwap;
        pillar.quoteDayCounter = zeroDayCounter;
        pillar.calendar = calendar;
        pillar.quote = markets::impliedQuote(pillar, reference, domesticDiscount);
        rootPillars.push_back(pillar);
    }
    markets::XccyChildInput<DiscountCurve<double>, DiscountCurve<double>> child;
    child.foreignForecastRole = markets::CurveRole::TenorBasis;
    child.domesticForecastRole = markets::CurveRole::IborOisBasis;
    child.foreignDiscount = &fallback.foreignDiscount;
    child.pillars = xccyPillars;
    child.dVdForeignZeros.assign(fallback.foreignDiscount.size(), 0.0);
    child.dVdForeignZeros[1] = 1.0;
    child.foreignForecast = &foreignForecast;
    child.domesticForecast = &domesticForecast;
    std::vector<std::variant<decltype(child)>> children{child};
    std::vector<double> dVdRoot(domesticDiscount.size(), 0.0);
    dVdRoot[1] = 1.0;
    const std::vector<markets::StackRiskEntry> entries =
        markets::stackQuoteRiskXccy(domesticDiscount, rootPillars, dVdRoot, children, reference);
    EXPECT_EQ(entries.size(), 4u);
    EXPECT_EQ(entries[0].points.size(), rootPillars.size());
    EXPECT_EQ(entries[1].points.size(), n);
    EXPECT_EQ(entries[2].points.size(), 3u); // funder: foreign forecast spread nodes
    EXPECT_EQ(entries[3].points.size(), domesticDiscount.size() - 1);
    for (const markets::StackRiskEntry& entry : entries) {
        entry.validate();
    }

    // Every xccy path reports under its own role, with maturity labels and
    // buckets in pillar order; the role buckets add back to the full table.
    EXPECT_TRUE(entries[0].points[0].role == markets::CurveRole::Discount);
    EXPECT_TRUE(entries[1].points[0].role == markets::CurveRole::XccyBasis);
    EXPECT_TRUE(entries[2].points[0].role == markets::CurveRole::TenorBasis);
    EXPECT_TRUE(entries[3].points[0].role == markets::CurveRole::IborOisBasis);
    for (const markets::QuotePoint& point : entries[1].points) {
        EXPECT_TRUE(point.role == markets::CurveRole::XccyBasis);
    }
    for (const markets::QuotePoint& point : entries[2].points) {
        EXPECT_TRUE(point.role == markets::CurveRole::TenorBasis);
    }
    for (const markets::QuotePoint& point : entries[3].points) {
        EXPECT_TRUE(point.role == markets::CurveRole::IborOisBasis);
    }
    EXPECT_EQ(entries[1].points[0].bucket, "1Y");
    EXPECT_EQ(entries[1].points[0].label, "Xccy 1Y");
    EXPECT_EQ(entries[2].points[0].bucket, "1Y");
    EXPECT_EQ(entries[2].points[0].label, "XccyFwd 1Y");
    EXPECT_EQ(entries[3].points[0].bucket, "1Y");
    EXPECT_EQ(entries[3].points[0].label, "XccyDom 1Y");
    double tableTotal = 0.0;
    for (const markets::StackRiskEntry& entry : entries) {
        for (const markets::QuotePoint& point : entry.points) {
            tableTotal += point.delta;
        }
    }
    double roleTotal = 0.0;
    bool sawDiscount = false;
    bool sawXccyBasis = false;
    bool sawTenorBasis = false;
    bool sawIborOisBasis = false;
    for (const markets::RiskBucket& bucket : markets::stackRoleBuckets(entries)) {
        roleTotal += bucket.delta;
        sawDiscount = sawDiscount || bucket.label == "Discount";
        sawXccyBasis = sawXccyBasis || bucket.label == "XccyBasis";
        sawTenorBasis = sawTenorBasis || bucket.label == "TenorBasis";
        sawIborOisBasis = sawIborOisBasis || bucket.label == "IborOisBasis";
    }
    CHECK_CLOSE("xccy role bucket total", roleTotal, tableTotal, 1e-12);
    EXPECT_TRUE(sawDiscount);
    EXPECT_TRUE(sawXccyBasis);
    EXPECT_TRUE(sawTenorBasis);
    EXPECT_TRUE(sawIborOisBasis);

    // The wrapper is the general engine: a hand-built unified input list (the
    // root, the xccy child, and the two forecast factor blocks) must produce
    // the same entry metadata and deltas to solver precision.
    {
        const markets::StackCurveView::Ptr rootView =
            markets::StackCurveView::make(domesticDiscount);
        const markets::StackCurveView::Ptr foreignForecastView =
            markets::StackCurveView::make(*child.foreignForecast);
        const markets::StackCurveView::Ptr domesticForecastView =
            markets::StackCurveView::make(*child.domesticForecast);
        std::vector<markets::StackCurveInput> unified(4);
        unified[0].curve = rootView;
        unified[0].role = markets::CurveRole::Discount;
        unified[0].discountPillars = rootPillars;
        unified[0].dVdNodes = dVdRoot;
        unified[1].curve = markets::StackCurveView::make(*child.foreignDiscount);
        unified[1].role = child.role;
        unified[1].xccyPillars = child.pillars;
        unified[1].xccy =
            markets::XccyRowInput{foreignForecastView, domesticForecastView, rootView};
        unified[1].dVdNodes = child.dVdForeignZeros;
        unified[2].curve = foreignForecastView;
        unified[2].role = child.foreignForecastRole;
        unified[2].factor = markets::StackFactorInput{
            "XccyFwd", markets::forecastNodeDayCounter(*child.foreignForecast)};
        unified[2].dVdNodes.assign(foreignForecastView->size(), 0.0);
        unified[3].curve = domesticForecastView;
        unified[3].role = child.domesticForecastRole;
        unified[3].factor = markets::StackFactorInput{
            "XccyDom", markets::forecastNodeDayCounter(*child.domesticForecast)};
        unified[3].dVdNodes.assign(domesticForecastView->size(), 0.0);
        const std::vector<markets::StackRiskEntry> general =
            markets::stackQuoteRisk(unified, reference);
        EXPECT_EQ(general.size(), entries.size());
        for (std::size_t k = 0; k < general.size(); ++k) {
            EXPECT_TRUE(general[k].role == entries[k].role);
            EXPECT_EQ(general[k].points.size(), entries[k].points.size());
            for (std::size_t j = 0; j < general[k].points.size(); ++j) {
                EXPECT_EQ(general[k].points[j].label, entries[k].points[j].label);
                EXPECT_EQ(general[k].points[j].bucket, entries[k].points[j].bucket);
                EXPECT_EQ(general[k].points[j].year, entries[k].points[j].year);
                EXPECT_TRUE(general[k].points[j].role == entries[k].points[j].role);
                CHECK_CLOSE("xccy wrapper vs general engine", general[k].points[j].delta,
                            entries[k].points[j].delta, 1e-10);
            }
        }

        // Cross-currency gamma now runs on the analytic rows: the general
        // curvature path differentiates the assembled Jacobian once. A zero
        // HZeta still leaves the bootstrap-curvature correction, and the
        // result must be a finite, symmetric matrix.
        const std::size_t unifiedDim = unified[0].curve->size() + unified[1].curve->size() +
                                       unified[2].curve->size() + unified[3].curve->size() - 4;
        const std::vector<double> zeroHessian(unifiedDim * unifiedDim, 0.0);
        const markets::StackQuoteGamma xccyGamma =
            markets::stackQuoteGamma(unified, zeroHessian, reference);
        EXPECT_EQ(xccyGamma.dim, unifiedDim);
        for (std::size_t a = 0; a < unifiedDim; ++a) {
            for (std::size_t b = a + 1; b < unifiedDim; ++b) {
                CHECK_CLOSE("xccy gamma symmetry", xccyGamma.at(a, b), xccyGamma.at(b, a), 1e-12);
            }
        }

        // An xccy row reference that is not a block of the input list is
        // rejected instead of silently binding to an equal-valued curve.
        {
            std::vector<markets::StackCurveInput> detached = unified;
            detached[1].xccy->foreignForecast =
                markets::StackCurveView::make(fallback.foreignSpread);
            EXPECT_THROW((void)(markets::stackQuoteRisk(detached, reference)),
                         std::invalid_argument);
        }

        // Delta regression: the block-elimination order the wrapper used
        // historically (child solve, root absorption, forecast residuals)
        // recovers the same root, child and factor deltas.
        const std::size_t nRef = xccyPillars.size();
        const std::size_t mRef = rootPillars.size();
        std::vector<double> fRootRef;
        std::vector<double> scratchRef;
        markets::assembleQuoteJacobian(domesticDiscount, rootPillars, reference, fRootRef,
                                       scratchRef);
        std::vector<double> fRef;
        std::vector<double> crossRef;
        std::vector<double> fwdRef;
        std::vector<double> domRef;
        markets::assembleXccyJacobianFull(*child.foreignDiscount, *child.foreignForecast,
                                          domesticDiscount, *child.domesticForecast, xccyPillars,
                                          reference, fRef, crossRef, fwdRef, domRef);
        std::vector<double> fChildTranspose(nRef * nRef);
        for (std::size_t j = 0; j < nRef; ++j) {
            for (std::size_t i = 0; i < nRef; ++i) {
                fChildTranspose[i * nRef + j] = fRef[j * nRef + i];
            }
        }
        std::vector<double> childG(nRef);
        for (std::size_t i = 0; i < nRef; ++i) {
            childG[i] = child.dVdForeignZeros[i + 1];
        }
        const std::vector<double> xChildRef = math::solveDense(fChildTranspose, nRef, childG);
        std::vector<double> rootG(mRef);
        for (std::size_t j = 0; j < mRef; ++j) {
            rootG[j] = dVdRoot[j + 1];
        }
        for (std::size_t i = 0; i < nRef; ++i) {
            for (std::size_t j = 0; j < mRef; ++j) {
                rootG[j] -= crossRef[i * mRef + j] * xChildRef[i];
            }
        }
        std::vector<double> fRootTranspose(mRef * mRef);
        for (std::size_t j = 0; j < mRef; ++j) {
            for (std::size_t i = 0; i < mRef; ++i) {
                fRootTranspose[i * mRef + j] = fRootRef[j * mRef + i];
            }
        }
        const std::vector<double> rootRef = math::solveDense(fRootTranspose, mRef, rootG);
        for (std::size_t j = 0; j < mRef; ++j) {
            CHECK_CLOSE("xccy root delta regression", entries[0].points[j].delta, rootRef[j],
                        1e-10);
        }
        for (std::size_t i = 0; i < nRef; ++i) {
            CHECK_CLOSE("xccy child delta regression", entries[1].points[i].delta, xChildRef[i],
                        1e-10);
        }
        for (std::size_t k = 0; k < entries[2].points.size(); ++k) {
            double referenceValue = 0.0;
            for (std::size_t i = 0; i < nRef; ++i) {
                referenceValue -= fwdRef[i * entries[2].points.size() + k] * xChildRef[i];
            }
            CHECK_CLOSE("xccy fwd factor delta regression", entries[2].points[k].delta,
                        referenceValue, 1e-10);
        }
        for (std::size_t k = 0; k < entries[3].points.size(); ++k) {
            double referenceValue = 0.0;
            for (std::size_t i = 0; i < nRef; ++i) {
                referenceValue -= domRef[i * entries[3].points.size() + k] * xChildRef[i];
            }
            CHECK_CLOSE("xccy dom factor delta regression", entries[3].points[k].delta,
                        referenceValue, 1e-10);
        }
    }

    // Wrapper validation keeps the general engine's input contract: a wrong
    // root sensitivity size, a root pillar count that misses the nodes, or a
    // child without a foreign discount curve are all rejected.
    {
        const std::vector<std::variant<decltype(child)>> single{child};
        std::vector<double> wrongRoot(dVdRoot.size() + 1, 0.0);
        EXPECT_THROW((void)(markets::stackQuoteRiskXccy(domesticDiscount, rootPillars, wrongRoot,
                                                        single, reference)),
                     std::invalid_argument);
        const std::vector<markets::CurvePillar> shortPillars(rootPillars.begin(),
                                                             rootPillars.end() - 1);
        EXPECT_THROW((void)(markets::stackQuoteRiskXccy(domesticDiscount, shortPillars, dVdRoot,
                                                        single, reference)),
                     std::invalid_argument);
        auto broken = child;
        broken.foreignDiscount = nullptr;
        const std::vector<std::variant<decltype(child)>> brokenChildren{broken};
        EXPECT_THROW((void)(markets::stackQuoteRiskXccy(domesticDiscount, rootPillars, dVdRoot,
                                                        brokenChildren, reference)),
                     std::invalid_argument);
    }
}

/// Analytic cross-currency rows against the finite-difference reference, for
/// every referenced block: the leg discount curves, the forecast native nodes,
/// a depth-2 foreign forecast chain and a domestic forecast parented on the
/// domestic discount itself, with per-leg payment lags and stub schedules.
void testXccyAnalyticRowsAgainstFiniteDifference() {
    const datetime::Date reference(2026, 9, 29);
    const datetime::DayCounter zeroDc(datetime::DayCount::Actual365Fixed);
    const datetime::Calendar calendar = datetime::Calendar::noHolidays();
    const auto buildCurve = [&](double base, double slope) {
        const std::vector<datetime::Date> dates{reference.plusYears(1), reference.plusYears(2),
                                                reference.plusYears(3), reference.plusYears(4)};
        std::vector<double> zeros;
        for (const datetime::Date& date : dates) {
            zeros.push_back(base + slope * datetime::yearFraction(reference, date, zeroDc));
        }
        return DiscountCurve<double>(reference, dates, zeroDc, zeros,
                                     InterpolationSpace::LogDiscount, InterpolationScheme::Linear);
    };
    const DiscountCurve<double> foreignDiscount = buildCurve(0.030, 0.0006);
    // The domestic discount object also parents the domestic forecast, so its
    // block row carries both the discount and the forecast-parent sensitivity.
    auto domesticParent = std::make_shared<DiscountCurve<double>>(buildCurve(0.040, 0.0005));
    const std::vector<double> nodeTimes{0.0, 1.0, 2.0, 3.0, 4.0};
    const markets::SpreadCurve<double> domesticForecast(
        domesticParent, nodeTimes, std::vector<double>{0.0, 0.0003, 0.0004, 0.0005, 0.0006},
        InterpolationScheme::Linear);
    auto foreignBase = std::make_shared<DiscountCurve<double>>(buildCurve(0.025, 0.0007));
    auto foreignParent = std::make_shared<markets::SpreadCurve<double>>(
        foreignBase, nodeTimes, std::vector<double>{0.0, 0.0004, 0.0005, 0.0006, 0.0007},
        InterpolationScheme::Linear);
    const markets::SpreadCurve<double, markets::SpreadCurve<double>> foreignForecast(
        foreignParent, nodeTimes, std::vector<double>{0.0, 0.0002, 0.0003, 0.0004, 0.0005},
        InterpolationScheme::Linear);

    const markets::StackCurveView::Ptr foreignDiscountView =
        markets::StackCurveView::make(foreignDiscount);
    const markets::StackCurveView::Ptr foreignForecastView =
        markets::StackCurveView::make(foreignForecast);
    const markets::StackCurveView::Ptr domesticDiscountView =
        markets::StackCurveView::make(*domesticParent);
    const markets::StackCurveView::Ptr domesticForecastView =
        markets::StackCurveView::make(domesticForecast);

    for (const bool spreadOnForeign : {true, false}) {
        SCOPED_TRACE(spreadOnForeign ? "spread on foreign leg" : "spread on domestic leg");
        markets::XccyPillar pillar;
        pillar.maturity = reference.plusYears(3);
        pillar.foreignTenor = datetime::Period(7, datetime::TimeUnit::Months);
        pillar.domesticTenor = datetime::Period(4, datetime::TimeUnit::Months);
        pillar.foreignPaymentLag = 2;
        pillar.domesticPaymentLag = 1;
        pillar.foreignBusinessDayConvention = datetime::BusinessDayConvention::Following;
        pillar.domesticBusinessDayConvention = datetime::BusinessDayConvention::ModifiedFollowing;
        pillar.foreignCalendar = calendar;
        pillar.domesticCalendar = calendar;
        pillar.foreignDayCounter = datetime::DayCounter(datetime::DayCount::Actual360);
        pillar.domesticDayCounter = datetime::DayCounter(datetime::DayCount::Actual360);
        pillar.spreadOnForeignLeg = spreadOnForeign;
        const std::vector<markets::XccyRowBlock> analytic = markets::xccySwapJacobianRowsView(
            *foreignDiscountView, *foreignForecastView, *domesticDiscountView,
            *domesticForecastView, pillar, reference, zeroDc);
        const std::vector<markets::XccyRowBlock> finiteDifference =
            markets::xccySwapJacobianRowsViewFiniteDifference(
                *foreignDiscountView, *foreignForecastView, *domesticDiscountView,
                *domesticForecastView, pillar, reference, zeroDc);
        // Foreign discount, foreign forecast, two foreign ancestors, domestic
        // discount (merged with the domestic forecast parent) and domestic
        // forecast.
        EXPECT_EQ(analytic.size(), 6u);
        EXPECT_EQ(finiteDifference.size(), analytic.size());
        for (std::size_t b = 0; b < analytic.size(); ++b) {
            EXPECT_TRUE(analytic[b].curve->identity() == finiteDifference[b].curve->identity());
            EXPECT_EQ(analytic[b].row.size(), finiteDifference[b].row.size());
            for (std::size_t i = 0; i < analytic[b].row.size(); ++i) {
                CHECK_CLOSE("xccy analytic row vs FD", analytic[b].row[i],
                            finiteDifference[b].row[i], 1e-6);
            }
        }
        double foreignBaseRisk = 0.0;
        double domesticDiscountRisk = 0.0;
        for (const markets::XccyRowBlock& block : analytic) {
            if (block.curve->identity() == foreignBase.get()) {
                for (const double value : block.row) {
                    foreignBaseRisk += std::abs(value);
                }
            }
            if (block.curve->identity() == domesticDiscountView->identity()) {
                for (const double value : block.row) {
                    domesticDiscountRisk += std::abs(value);
                }
            }
        }
        EXPECT_GT(foreignBaseRisk, 1e-8);
        EXPECT_GT(domesticDiscountRisk, 1e-8);
    }
}

/// Xccy gamma against an independent finite difference of the deltas: the
/// quote-space Hessian is compared with the central difference of
/// `stackQuoteRisk` after bumping each root and xccy quote and re-bootstrapping
/// the curve, with the quadratic value's node gradient re-evaluated at the
/// rebuilt nodes.
void testXccyGammaFiniteDifference() {
    const datetime::Date reference(2026, 9, 29);
    const datetime::DayCounter zeroDc(datetime::DayCount::Actual365Fixed);
    const datetime::Calendar calendar = datetime::Calendar::noHolidays();
    const auto buildCurve = [&](double base, double slope) {
        const std::vector<datetime::Date> dates{reference.plusYears(1), reference.plusYears(2)};
        std::vector<double> zeros;
        for (const datetime::Date& date : dates) {
            zeros.push_back(base + slope * datetime::yearFraction(reference, date, zeroDc));
        }
        return DiscountCurve<double>(reference, dates, zeroDc, zeros,
                                     InterpolationSpace::LogDiscount, InterpolationScheme::Linear);
    };
    const DiscountCurve<double> rootTarget = buildCurve(0.030, 0.0004);
    std::vector<markets::CurvePillar> rootPillars;
    for (int year = 1; year <= 2; ++year) {
        markets::CurvePillar pillar;
        pillar.maturity = reference.plusYears(year);
        pillar.kind = markets::PillarKind::OisSwap;
        pillar.quoteDayCounter = zeroDc;
        pillar.calendar = calendar;
        pillar.quote = markets::impliedQuote(pillar, reference, rootTarget);
        rootPillars.push_back(pillar);
    }
    const auto bootstrapRoot = [&](const std::vector<markets::CurvePillar>& quotes) {
        return std::make_shared<DiscountCurve<double>>(
            markets::bootstrapDiscountCurve(reference, zeroDc, InterpolationSpace::LogDiscount,
                                            InterpolationScheme::Linear, quotes));
    };
    std::shared_ptr<DiscountCurve<double>> rootPtr = bootstrapRoot(rootPillars);
    const std::vector<double> forecastTimes{0.0, 1.0, 2.0};
    const std::vector<double> forecastSpreads{0.0, 0.0008, 0.0011};
    const auto makeForecast = [&](const std::shared_ptr<DiscountCurve<double>>& parent) {
        return markets::SpreadCurve<double>(parent, forecastTimes, forecastSpreads,
                                            InterpolationScheme::Linear);
    };
    // The foreign forecast is parented on the root object, so a root bump
    // reaches the xccy rows through the forecast parent chain.
    const markets::SpreadCurve<double> foreignForecast = makeForecast(rootPtr);
    const DiscountCurve<double> domesticForecast = buildCurve(0.041, 0.0003);
    const DiscountCurve<double> foreignTarget = buildCurve(0.028, 0.0005);
    std::vector<markets::XccyPillar> xccyPillars;
    for (int year = 1; year <= 2; ++year) {
        markets::XccyPillar pillar;
        pillar.maturity = reference.plusYears(year);
        pillar.foreignTenor = datetime::Period(3, datetime::TimeUnit::Months);
        pillar.domesticTenor = datetime::Period(3, datetime::TimeUnit::Months);
        pillar.foreignCalendar = calendar;
        pillar.domesticCalendar = calendar;
        pillar.foreignDayCounter = datetime::DayCounter(datetime::DayCount::Actual360);
        pillar.domesticDayCounter = datetime::DayCounter(datetime::DayCount::Actual360);
        pillar.spread = markets::impliedXccyBasisSpread(
            foreignTarget, foreignForecast, *rootPtr, domesticForecast, pillar, reference, zeroDc);
        xccyPillars.push_back(pillar);
    }
    const auto bootstrapChild = [&](const DiscountCurve<double>& rootCurve,
                                    const markets::SpreadCurve<double>& forecast,
                                    const std::vector<markets::XccyPillar>& quotes) {
        return markets::bootstrapXccyDiscountCurve(rootCurve, domesticForecast, forecast, reference,
                                                   zeroDc, InterpolationSpace::LogDiscount,
                                                   InterpolationScheme::Linear, quotes);
    };
    const DiscountCurve<double> childBase = bootstrapChild(*rootPtr, foreignForecast, xccyPillars);

    const std::size_t m = rootPillars.size();
    const std::size_t n = xccyPillars.size();
    const std::size_t p = foreignForecast.size() - 1;
    const std::size_t q = domesticForecast.size() - 1;
    const std::size_t dim = m + n + p + q;
    std::vector<double> g(dim, 0.0);
    for (std::size_t i = 0; i < dim; ++i) {
        g[i] = 0.01 + 0.003 * static_cast<double>(i);
    }
    std::vector<double> HZeta(dim * dim, 0.0);
    for (std::size_t i = 0; i < dim; ++i) {
        for (std::size_t j = 0; j < dim; ++j) {
            HZeta[i * dim + j] =
                0.02 * static_cast<double>(std::min(i, j) + 1) + (i == j ? 0.05 : 0.0);
        }
    }
    const auto zetaAt =
        [&](const DiscountCurve<double>& rootCurve, const DiscountCurve<double>& childCurve,
            const markets::SpreadCurve<double>& fwdCurve, const DiscountCurve<double>& domCurve) {
            std::vector<double> zeta;
            zeta.reserve(dim);
            for (std::size_t i = 1; i <= m; ++i) {
                zeta.push_back(rootCurve.zeros()[i]);
            }
            for (std::size_t i = 1; i <= n; ++i) {
                zeta.push_back(childCurve.zeros()[i]);
            }
            for (std::size_t i = 1; i <= p; ++i) {
                zeta.push_back(fwdCurve.spreadNodes().zeros()[i]);
            }
            for (std::size_t i = 1; i <= q; ++i) {
                zeta.push_back(domCurve.zeros()[i]);
            }
            return zeta;
        };
    const auto gradientAt = [&](const std::vector<double>& zeta) {
        std::vector<double> nodeGradient(dim, 0.0);
        for (std::size_t i = 0; i < dim; ++i) {
            double sum = g[i];
            for (std::size_t j = 0; j < dim; ++j) {
                sum += HZeta[i * dim + j] * zeta[j];
            }
            nodeGradient[i] = sum;
        }
        return nodeGradient;
    };
    const auto buildInputs = [&](const std::shared_ptr<DiscountCurve<double>>& rootCurve,
                                 const DiscountCurve<double>& childCurve,
                                 const markets::SpreadCurve<double>& fwdCurve,
                                 const DiscountCurve<double>& domCurve,
                                 const std::vector<double>& nodeGradient) {
        const markets::StackCurveView::Ptr rootView = markets::StackCurveView::make(*rootCurve);
        const markets::StackCurveView::Ptr childView = markets::StackCurveView::make(childCurve);
        const markets::StackCurveView::Ptr fwdView = markets::StackCurveView::make(fwdCurve);
        const markets::StackCurveView::Ptr domView = markets::StackCurveView::make(domCurve);
        std::vector<markets::StackCurveInput> inputs(4);
        inputs[0].curve = rootView;
        inputs[0].role = markets::CurveRole::Discount;
        inputs[0].discountPillars = rootPillars;
        inputs[0].dVdNodes.assign(rootView->size(), 0.0);
        inputs[1].curve = childView;
        inputs[1].role = markets::CurveRole::XccyBasis;
        inputs[1].xccyPillars = xccyPillars;
        inputs[1].xccy = markets::XccyRowInput{fwdView, domView, rootView};
        inputs[1].dVdNodes.assign(childView->size(), 0.0);
        inputs[2].curve = fwdView;
        inputs[2].role = markets::CurveRole::Forecast;
        inputs[2].factor =
            markets::StackFactorInput{"XccyFwd", markets::forecastNodeDayCounter(fwdCurve)};
        inputs[2].dVdNodes.assign(fwdView->size(), 0.0);
        inputs[3].curve = domView;
        inputs[3].role = markets::CurveRole::Forecast;
        inputs[3].factor =
            markets::StackFactorInput{"XccyDom", markets::forecastNodeDayCounter(domCurve)};
        inputs[3].dVdNodes.assign(domView->size(), 0.0);
        for (std::size_t i = 0; i < m; ++i) {
            inputs[0].dVdNodes[i + 1] = nodeGradient[i];
        }
        for (std::size_t i = 0; i < n; ++i) {
            inputs[1].dVdNodes[i + 1] = nodeGradient[m + i];
        }
        for (std::size_t i = 0; i < p; ++i) {
            inputs[2].dVdNodes[i + 1] = nodeGradient[m + n + i];
        }
        for (std::size_t i = 0; i < q; ++i) {
            inputs[3].dVdNodes[i + 1] = nodeGradient[m + n + p + i];
        }
        return inputs;
    };
    const auto flatten = [](const std::vector<markets::StackRiskEntry>& entries) {
        std::vector<double> flat;
        for (const markets::StackRiskEntry& entry : entries) {
            for (const markets::QuotePoint& point : entry.points) {
                flat.push_back(point.delta);
            }
        }
        return flat;
    };
    const std::vector<double> baseZeta =
        zetaAt(*rootPtr, childBase, foreignForecast, domesticForecast);
    const std::vector<double> baseGradient = gradientAt(baseZeta);
    const markets::StackQuoteGamma gamma = markets::stackQuoteGamma(
        buildInputs(rootPtr, childBase, foreignForecast, domesticForecast, baseGradient), HZeta,
        reference);
    EXPECT_EQ(gamma.dim, dim);
    for (std::size_t i = 0; i < dim; ++i) {
        for (std::size_t j = i + 1; j < dim; ++j) {
            CHECK_CLOSE("xccy gamma symmetry", gamma.at(i, j), gamma.at(j, i), 1e-12);
        }
    }
    const double epsilon = 1e-6;
    for (std::size_t r = 0; r < m; ++r) {
        std::vector<markets::CurvePillar> up = rootPillars;
        std::vector<markets::CurvePillar> down = rootPillars;
        up[r].quote += epsilon;
        down[r].quote -= epsilon;
        const std::shared_ptr<DiscountCurve<double>> rootUp = bootstrapRoot(up);
        const std::shared_ptr<DiscountCurve<double>> rootDown = bootstrapRoot(down);
        const markets::SpreadCurve<double> fwdUp = makeForecast(rootUp);
        const markets::SpreadCurve<double> fwdDown = makeForecast(rootDown);
        const DiscountCurve<double> childUp = bootstrapChild(*rootUp, fwdUp, xccyPillars);
        const DiscountCurve<double> childDown = bootstrapChild(*rootDown, fwdDown, xccyPillars);
        const std::vector<double> plus = flatten(markets::stackQuoteRisk(
            buildInputs(rootUp, childUp, fwdUp, domesticForecast,
                        gradientAt(zetaAt(*rootUp, childUp, fwdUp, domesticForecast))),
            reference));
        const std::vector<double> minus = flatten(markets::stackQuoteRisk(
            buildInputs(rootDown, childDown, fwdDown, domesticForecast,
                        gradientAt(zetaAt(*rootDown, childDown, fwdDown, domesticForecast))),
            reference));
        for (std::size_t a = 0; a < dim; ++a) {
            const double fd = (plus[a] - minus[a]) / (2.0 * epsilon);
            CHECK_CLOSE("xccy gamma root quote FD", gamma.hessian[a * dim + r], fd, 1e-6);
        }
    }
    for (std::size_t r = 0; r < n; ++r) {
        std::vector<markets::XccyPillar> up = xccyPillars;
        std::vector<markets::XccyPillar> down = xccyPillars;
        up[r].spread += epsilon;
        down[r].spread -= epsilon;
        const DiscountCurve<double> childUp = bootstrapChild(*rootPtr, foreignForecast, up);
        const DiscountCurve<double> childDown = bootstrapChild(*rootPtr, foreignForecast, down);
        const std::vector<double> plus = flatten(markets::stackQuoteRisk(
            buildInputs(rootPtr, childUp, foreignForecast, domesticForecast,
                        gradientAt(zetaAt(*rootPtr, childUp, foreignForecast, domesticForecast))),
            reference));
        const std::vector<double> minus = flatten(markets::stackQuoteRisk(
            buildInputs(rootPtr, childDown, foreignForecast, domesticForecast,
                        gradientAt(zetaAt(*rootPtr, childDown, foreignForecast, domesticForecast))),
            reference));
        for (std::size_t a = 0; a < dim; ++a) {
            const double fd = (plus[a] - minus[a]) / (2.0 * epsilon);
            CHECK_CLOSE("xccy gamma xccy quote FD", gamma.hessian[a * dim + m + r], fd, 1e-6);
        }
    }
}

/// Two children sharing one forecast object keep the shared factor block but
/// emit one `XccyFwd`/`XccyDom` entry each; the per-child residuals are
/// non-zero and sum to the shared block total.
void testXccySharedForecastSplit() {
    const datetime::Date reference(2026, 9, 29);
    const datetime::DayCounter zeroDc(datetime::DayCount::Actual365Fixed);
    const datetime::Calendar calendar = datetime::Calendar::noHolidays();
    const auto buildCurve = [&](double base, double slope) {
        const std::vector<datetime::Date> dates{reference.plusYears(1), reference.plusYears(2),
                                                reference.plusYears(3)};
        std::vector<double> zeros;
        for (const datetime::Date& date : dates) {
            zeros.push_back(base + slope * datetime::yearFraction(reference, date, zeroDc));
        }
        return DiscountCurve<double>(reference, dates, zeroDc, zeros,
                                     InterpolationSpace::LogDiscount, InterpolationScheme::Linear);
    };
    const DiscountCurve<double> rootTarget = buildCurve(0.030, 0.0004);
    std::vector<markets::CurvePillar> rootPillars;
    for (int year = 1; year <= 3; ++year) {
        markets::CurvePillar pillar;
        pillar.maturity = reference.plusYears(year);
        pillar.kind = markets::PillarKind::OisSwap;
        pillar.quoteDayCounter = zeroDc;
        pillar.calendar = calendar;
        pillar.quote = markets::impliedQuote(pillar, reference, rootTarget);
        rootPillars.push_back(pillar);
    }
    const auto rootPtr = std::make_shared<DiscountCurve<double>>(
        markets::bootstrapDiscountCurve(reference, zeroDc, InterpolationSpace::LogDiscount,
                                        InterpolationScheme::Linear, rootPillars));
    const DiscountCurve<double> domesticForecast = buildCurve(0.041, 0.0003);
    // The shared forecast is parented on the root object itself, so its parent
    // chain block is the root already present in the stack.
    const std::vector<double> forecastTimes{0.0, 1.0, 2.0, 3.0};
    const markets::SpreadCurve<double> foreignForecast(
        rootPtr, forecastTimes, std::vector<double>{0.0, 0.0005, 0.0007, 0.0009},
        InterpolationScheme::Linear);
    const DiscountCurve<double> targetA = buildCurve(0.028, 0.0005);
    const DiscountCurve<double> targetB = buildCurve(0.032, 0.0004);
    const auto makePillars = [&](const DiscountCurve<double>& target) {
        std::vector<markets::XccyPillar> pillars;
        for (int year = 1; year <= 3; ++year) {
            markets::XccyPillar pillar;
            pillar.maturity = reference.plusYears(year);
            pillar.foreignTenor = datetime::Period(3, datetime::TimeUnit::Months);
            pillar.domesticTenor = datetime::Period(3, datetime::TimeUnit::Months);
            pillar.foreignCalendar = calendar;
            pillar.domesticCalendar = calendar;
            pillar.foreignDayCounter = datetime::DayCounter(datetime::DayCount::Actual360);
            pillar.domesticDayCounter = datetime::DayCounter(datetime::DayCount::Actual360);
            pillar.spread = markets::impliedXccyBasisSpread(
                target, foreignForecast, *rootPtr, domesticForecast, pillar, reference, zeroDc);
            pillars.push_back(pillar);
        }
        return pillars;
    };
    const std::vector<markets::XccyPillar> pillarsA = makePillars(targetA);
    const std::vector<markets::XccyPillar> pillarsB = makePillars(targetB);
    const DiscountCurve<double> discountA = markets::bootstrapXccyDiscountCurve(
        *rootPtr, domesticForecast, foreignForecast, reference, zeroDc,
        InterpolationSpace::LogDiscount, InterpolationScheme::Linear, pillarsA);
    const DiscountCurve<double> discountB = markets::bootstrapXccyDiscountCurve(
        *rootPtr, domesticForecast, foreignForecast, reference, zeroDc,
        InterpolationSpace::LogDiscount, InterpolationScheme::Linear, pillarsB);

    using ChildT = markets::XccyChildInput<markets::SpreadCurve<double>, DiscountCurve<double>>;
    ChildT childA;
    childA.foreignForecastRole = markets::CurveRole::TenorBasis;
    childA.domesticForecastRole = markets::CurveRole::Forecast;
    childA.foreignDiscount = &discountA;
    childA.pillars = pillarsA;
    childA.dVdForeignZeros.assign(discountA.size(), 0.0);
    childA.dVdForeignZeros[1] = 1.0;
    childA.foreignForecast = &foreignForecast;
    childA.domesticForecast = &domesticForecast;
    childA.dVdForeignForecast = std::vector<double>(foreignForecast.size(), 0.0);
    (*childA.dVdForeignForecast)[1] = 0.25;
    childA.dVdDomesticForecast = std::vector<double>(domesticForecast.size(), 0.0);
    (*childA.dVdDomesticForecast)[1] = 0.10;
    ChildT childB = childA;
    childB.foreignForecastRole = markets::CurveRole::IborOisBasis;
    childB.domesticForecastRole = markets::CurveRole::TenorBasis;
    childB.foreignDiscount = &discountB;
    childB.pillars = pillarsB;
    childB.dVdForeignZeros.assign(discountB.size(), 0.0);
    childB.dVdForeignZeros[2] = 1.0;
    (*childB.dVdForeignForecast)[1] = 0.40;
    (*childB.dVdDomesticForecast)[1] = 0.20;

    std::vector<double> dVdRoot(rootPtr->size(), 0.0);
    dVdRoot[1] = 1.0;
    std::vector<std::variant<ChildT>> children;
    children.emplace_back(childA);
    children.emplace_back(childB);
    const std::vector<markets::StackRiskEntry> entries =
        markets::stackQuoteRiskXccy(*rootPtr, rootPillars, dVdRoot, children, reference);
    // Root, then each child's quote, foreign forecast and domestic forecast.
    EXPECT_EQ(entries.size(), 7u);
    EXPECT_EQ(entries[0].points.size(), rootPillars.size());
    EXPECT_EQ(entries[1].points.size(), pillarsA.size());
    EXPECT_EQ(entries[2].points.size(), foreignForecast.size() - 1);
    EXPECT_EQ(entries[3].points.size(), domesticForecast.size() - 1);
    EXPECT_EQ(entries[4].points.size(), pillarsB.size());
    EXPECT_EQ(entries[5].points.size(), foreignForecast.size() - 1);
    EXPECT_EQ(entries[6].points.size(), domesticForecast.size() - 1);
    EXPECT_TRUE(entries[2].role == markets::CurveRole::TenorBasis);
    EXPECT_TRUE(entries[5].role == markets::CurveRole::IborOisBasis);
    EXPECT_TRUE(entries[3].role == markets::CurveRole::Forecast);
    EXPECT_TRUE(entries[6].role == markets::CurveRole::TenorBasis);

    // Shared-block reference: one unified input list with a single foreign and
    // domestic factor block holding the summed direct sensitivities.
    const markets::StackCurveView::Ptr rootView = markets::StackCurveView::make(*rootPtr);
    const markets::StackCurveView::Ptr fwdView = markets::StackCurveView::make(foreignForecast);
    const markets::StackCurveView::Ptr domView = markets::StackCurveView::make(domesticForecast);
    std::vector<markets::StackCurveInput> unified(5);
    unified[0].curve = rootView;
    unified[0].role = markets::CurveRole::Discount;
    unified[0].discountPillars = rootPillars;
    unified[0].dVdNodes = dVdRoot;
    for (std::size_t c = 0; c < 2; ++c) {
        const ChildT& child = c == 0 ? childA : childB;
        unified[1 + c].curve = markets::StackCurveView::make(*child.foreignDiscount);
        unified[1 + c].role = child.role;
        unified[1 + c].xccyPillars = child.pillars;
        unified[1 + c].xccy = markets::XccyRowInput{fwdView, domView, rootView};
        unified[1 + c].dVdNodes = child.dVdForeignZeros;
    }
    unified[3].curve = fwdView;
    unified[3].role = markets::CurveRole::Forecast;
    unified[3].factor =
        markets::StackFactorInput{"XccyFwd", markets::forecastNodeDayCounter(foreignForecast)};
    unified[3].dVdNodes.assign(fwdView->size(), 0.0);
    unified[4].curve = domView;
    unified[4].role = markets::CurveRole::Forecast;
    unified[4].factor =
        markets::StackFactorInput{"XccyDom", markets::forecastNodeDayCounter(domesticForecast)};
    unified[4].dVdNodes.assign(domView->size(), 0.0);
    for (std::size_t i = 1; i < childA.dVdForeignForecast->size(); ++i) {
        unified[3].dVdNodes[i] += (*childA.dVdForeignForecast)[i] + (*childB.dVdForeignForecast)[i];
    }
    for (std::size_t i = 1; i < childA.dVdDomesticForecast->size(); ++i) {
        unified[4].dVdNodes[i] +=
            (*childA.dVdDomesticForecast)[i] + (*childB.dVdDomesticForecast)[i];
    }
    const std::vector<markets::StackRiskEntry> shared = markets::stackQuoteRisk(unified, reference);
    EXPECT_EQ(shared.size(), 5u);
    double childAForward = 0.0;
    double childBForward = 0.0;
    double childADomestic = 0.0;
    double childBDomestic = 0.0;
    for (std::size_t k = 0; k < entries[2].points.size(); ++k) {
        childAForward += std::abs(entries[2].points[k].delta);
        childBForward += std::abs(entries[5].points[k].delta);
        CHECK_CLOSE("shared forecast split",
                    entries[2].points[k].delta + entries[5].points[k].delta,
                    shared[3].points[k].delta, 1e-10);
    }
    for (std::size_t k = 0; k < entries[3].points.size(); ++k) {
        childADomestic += std::abs(entries[3].points[k].delta);
        childBDomestic += std::abs(entries[6].points[k].delta);
        CHECK_CLOSE("shared domestic split",
                    entries[3].points[k].delta + entries[6].points[k].delta,
                    shared[4].points[k].delta, 1e-10);
    }
    EXPECT_GT(childAForward, 1e-8);
    EXPECT_GT(childBForward, 1e-8);
    EXPECT_GT(childADomestic, 1e-8);
    EXPECT_GT(childBDomestic, 1e-8);
}

/// Three-coupon resetting-notional leg with distinct curves: with the
/// resetting leg's forecast equal to its own discount curve the telescoped
/// value `sum_k adjN_k (D(t_k)(1 + f_k tau_k) - D(t_{k-1}))` is zero, so the
/// implied spread reduces to the opposite constant-notional leg value over the
/// quoted leg's annuity. Golden values from an independent closed-form
/// reference.
void testXccyMtMTelescoping() {
    const datetime::Date reference(2026, 9, 29);
    const datetime::DayCounter zeroDc(datetime::DayCount::Actual365Fixed);
    const datetime::Calendar calendar = datetime::Calendar::noHolidays();
    const std::vector<datetime::Date> dates{reference.plusMonths(3), reference.plusMonths(6),
                                            reference.plusMonths(9)};
    const auto buildCurve = [&](double base, double slope) {
        std::vector<double> zeros;
        for (const datetime::Date& date : dates) {
            zeros.push_back(base + slope * datetime::yearFraction(reference, date, zeroDc));
        }
        return DiscountCurve<double>(reference, dates, zeroDc, zeros,
                                     InterpolationSpace::LogDiscount, InterpolationScheme::Linear);
    };
    const DiscountCurve<double> domesticDiscount = buildCurve(0.040, 0.0005);
    const DiscountCurve<double> domesticForecast = buildCurve(0.043, 0.0004);
    const DiscountCurve<double> foreignTarget = buildCurve(0.030, 0.0006);
    const DiscountCurve<double> foreignForecast = buildCurve(0.025, 0.0008);

    const auto makePillar = [&](bool resetForeign, bool spreadOnForeign) {
        markets::XccyPillar pillar;
        pillar.maturity = dates.back();
        pillar.foreignCalendar = calendar;
        pillar.domesticCalendar = calendar;
        pillar.foreignDayCounter = datetime::DayCounter(datetime::DayCount::Actual360);
        pillar.domesticDayCounter = datetime::DayCounter(datetime::DayCount::Actual360);
        pillar.notional = markets::XccyNotionalMode::MtM;
        pillar.resetForeignLeg = resetForeign;
        pillar.spreadOnForeignLeg = spreadOnForeign;
        return pillar;
    };

    // Foreign leg resets and forecasts on itself.
    for (const bool spreadOnForeign : {true, false}) {
        const double expected = spreadOnForeign ? 0.08254738125917123 : -0.08295639871603087;
        SCOPED_TRACE(spreadOnForeign ? "spread on foreign leg" : "spread on domestic leg");
        CHECK_CLOSE("mtm telescoping foreign reset",
                    markets::impliedXccyBasisSpread(
                        foreignTarget, foreignTarget, domesticDiscount, domesticForecast,
                        makePillar(true, spreadOnForeign), reference, zeroDc),
                    expected, 1e-12);
    }
    // Domestic leg resets and forecasts on itself.
    for (const bool spreadOnForeign : {true, false}) {
        const double expected = spreadOnForeign ? -0.055470884346166476 : 0.055745739340945656;
        SCOPED_TRACE(spreadOnForeign ? "spread on foreign leg" : "spread on domestic leg");
        CHECK_CLOSE("mtm telescoping domestic reset",
                    markets::impliedXccyBasisSpread(
                        foreignTarget, foreignForecast, domesticDiscount, domesticDiscount,
                        makePillar(false, spreadOnForeign), reference, zeroDc),
                    expected, 1e-12);
    }
}

/// Golden spreads of a three-coupon resetting-notional pillar with distinct
/// domestic/foreign curves, covering both reset-leg choices and both
/// `spreadOnForeignLeg` values. Values are produced by an independent
/// reference implementation of the telescoped reset and constant-notional
/// formulas.
void testXccyMtMGoldenSpreads() {
    const datetime::Date reference(2026, 9, 29);
    const datetime::DayCounter zeroDc(datetime::DayCount::Actual365Fixed);
    const datetime::Calendar calendar = datetime::Calendar::noHolidays();
    const std::vector<datetime::Date> dates{reference.plusMonths(3), reference.plusMonths(6),
                                            reference.plusMonths(9)};
    const auto buildCurve = [&](double base, double slope) {
        std::vector<double> zeros;
        for (const datetime::Date& date : dates) {
            zeros.push_back(base + slope * datetime::yearFraction(reference, date, zeroDc));
        }
        return DiscountCurve<double>(reference, dates, zeroDc, zeros,
                                     InterpolationSpace::LogDiscount, InterpolationScheme::Linear);
    };
    const DiscountCurve<double> domesticDiscount = buildCurve(0.040, 0.0005);
    const DiscountCurve<double> domesticForecast = buildCurve(0.043, 0.0004);
    const DiscountCurve<double> foreignTarget = buildCurve(0.030, 0.0006);
    const DiscountCurve<double> foreignForecast = buildCurve(0.025, 0.0008);

    struct GoldenCase {
        bool resetForeign;
        bool spreadOnForeign;
        double spread;
    };
    const std::vector<GoldenCase> cases{
        {true, true, 0.08735367316126888},
        {true, false, -0.087786505514019036},
        {false, true, -0.052562454453813891},
        {false, false, 0.052822898348205888},
    };
    for (const GoldenCase& golden : cases) {
        SCOPED_TRACE(::testing::Message() << "resetForeign=" << golden.resetForeign
                                          << " spreadOnForeign=" << golden.spreadOnForeign);
        markets::XccyPillar pillar;
        pillar.maturity = dates.back();
        pillar.foreignCalendar = calendar;
        pillar.domesticCalendar = calendar;
        pillar.foreignDayCounter = datetime::DayCounter(datetime::DayCount::Actual360);
        pillar.domesticDayCounter = datetime::DayCounter(datetime::DayCount::Actual360);
        pillar.notional = markets::XccyNotionalMode::MtM;
        pillar.resetForeignLeg = golden.resetForeign;
        pillar.spreadOnForeignLeg = golden.spreadOnForeign;
        const double spread =
            markets::impliedXccyBasisSpread(foreignTarget, foreignForecast, domesticDiscount,
                                            domesticForecast, pillar, reference, zeroDc);
        CHECK_CLOSE("mtm golden spread", spread, golden.spread, 1e-12);
    }
}

/// A bootstrap ladder mixing one resetting-notional and one
/// constant-notional pillar recovers the synthetic foreign curve at both nodes
/// and reprices both quotes.
void testXccyMixedNotionalBootstrap() {
    const datetime::Date reference(2026, 9, 29);
    const datetime::DayCounter zeroDc(datetime::DayCount::Actual365Fixed);
    const datetime::Calendar calendar = datetime::Calendar::noHolidays();
    const auto buildCurve = [&](double base, double slope, int maxYears) {
        std::vector<datetime::Date> pillarDates;
        std::vector<double> zeros;
        for (int years = 1; years <= maxYears; ++years) {
            pillarDates.push_back(reference.plusYears(years));
            zeros.push_back(base +
                            slope * datetime::yearFraction(reference, pillarDates.back(), zeroDc));
        }
        return DiscountCurve<double>(reference, pillarDates, zeroDc, zeros,
                                     InterpolationSpace::LogDiscount, InterpolationScheme::Linear);
    };
    const DiscountCurve<double> domesticDiscount = buildCurve(0.040, 0.0005, 2);
    const DiscountCurve<double> domesticForecast = buildCurve(0.043, 0.0004, 2);
    const DiscountCurve<double> foreignForecast = buildCurve(0.025, 0.0008, 2);
    const DiscountCurve<double> foreignTarget = buildCurve(0.028, 0.0006, 2);

    std::vector<markets::XccyPillar> pillars;
    for (int years = 1; years <= 2; ++years) {
        markets::XccyPillar pillar;
        pillar.maturity = reference.plusYears(years);
        pillar.foreignCalendar = calendar;
        pillar.domesticCalendar = calendar;
        pillar.foreignTenor = datetime::Period(3, datetime::TimeUnit::Months);
        pillar.domesticTenor = datetime::Period(3, datetime::TimeUnit::Months);
        pillar.foreignDayCounter = datetime::DayCounter(datetime::DayCount::Actual360);
        pillar.domesticDayCounter = datetime::DayCounter(datetime::DayCount::Actual360);
        pillar.notional =
            years == 1 ? markets::XccyNotionalMode::MtM : markets::XccyNotionalMode::Const;
        pillar.spread =
            markets::impliedXccyBasisSpread(foreignTarget, foreignForecast, domesticDiscount,
                                            domesticForecast, pillar, reference, zeroDc);
        pillars.push_back(pillar);
    }

    const DiscountCurve<double> curve = markets::bootstrapXccyDiscountCurve(
        domesticDiscount, domesticForecast, foreignForecast, reference, zeroDc,
        InterpolationSpace::LogDiscount, InterpolationScheme::Linear, pillars);
    for (std::size_t i = 0; i < pillars.size(); ++i) {
        CHECK_CLOSE("mixed notionals recovered zero", curve.zeros()[i + 1],
                    foreignTarget.zeros()[i + 1], 1e-10);
        CHECK_CLOSE("mixed notionals reprice",
                    markets::impliedXccyBasisSpread(curve, foreignForecast, domesticDiscount,
                                                    domesticForecast, pillars[i], reference,
                                                    zeroDc),
                    pillars[i].spread, 1e-10);
    }
}

/// Coupled bootstrap with resetting-notional xccy pillars: the reset enters
/// only the discount-curve residual, so the fixed point must still recover the
/// synthetic discount and spread curves.
void testXccyCoupledMtM() {
    const datetime::Date reference(2026, 9, 29);
    const datetime::DayCounter zeroDc(datetime::DayCount::Actual365Fixed);
    const datetime::Calendar calendar = datetime::Calendar::noHolidays();
    const auto buildCurve = [&](double base, double slope, int maxYears) {
        std::vector<datetime::Date> pillarDates;
        std::vector<double> zeros;
        for (int years = 1; years <= maxYears; ++years) {
            const datetime::Date date = reference.plusYears(years);
            pillarDates.push_back(date);
            zeros.push_back(base + slope * datetime::yearFraction(reference, date, zeroDc));
        }
        return DiscountCurve<double>(reference, pillarDates, zeroDc, zeros,
                                     InterpolationSpace::LogDiscount, InterpolationScheme::Linear);
    };
    const DiscountCurve<double> domesticDiscount = buildCurve(0.040, 0.0005, 5);
    const DiscountCurve<double> domesticForecast = domesticDiscount;
    const DiscountCurve<double> xccyTarget = buildCurve(0.030, 0.0005, 3);
    auto foreignBase = std::make_shared<DiscountCurve<double>>(buildCurve(0.025, 0.0005, 3));
    const std::vector<double> spreadTimes{0.0, 1.0, 2.0, 3.0};
    const std::vector<double> spreadNodes{0.0, 0.0010, 0.0013, 0.0016};
    const markets::SpreadCurve<double> spreadTarget(foreignBase, spreadTimes, spreadNodes,
                                                    InterpolationScheme::Linear);

    std::vector<markets::XccyCoupledPillar> pillars;
    for (int years = 1; years <= 3; ++years) {
        const datetime::Date maturity = reference.plusYears(years);
        markets::XccyCoupledPillar coupled;
        coupled.xccy.maturity = maturity;
        coupled.xccy.foreignCalendar = calendar;
        coupled.xccy.domesticCalendar = calendar;
        coupled.xccy.notional = markets::XccyNotionalMode::MtM;
        coupled.xccy.spread =
            markets::impliedXccyBasisSpread(xccyTarget, spreadTarget, domesticDiscount,
                                            domesticForecast, coupled.xccy, reference, zeroDc);
        coupled.basis.maturity = maturity;
        coupled.basis.calendar = calendar;
        coupled.basis.spread = markets::impliedBasisSpread(spreadTarget, coupled.basis, reference,
                                                           zeroDc, &xccyTarget);
        pillars.push_back(coupled);
    }

    const markets::XccyCoupledResult coupled = markets::bootstrapXccyCoupled(
        domesticDiscount, domesticForecast, foreignBase, reference, zeroDc,
        InterpolationSpace::LogDiscount, InterpolationScheme::Linear, pillars);
    EXPECT_TRUE(coupled.converged);
    EXPECT_TRUE(!coupled.usedJointFallback);
    for (std::size_t i = 0; i < pillars.size(); ++i) {
        CHECK_CLOSE("coupled mtm xccy node", coupled.foreignDiscount.zeros()[i + 1],
                    xccyTarget.zeros()[i + 1], 1e-9);
        CHECK_CLOSE("coupled mtm xccy reprice",
                    markets::impliedXccyBasisSpread(coupled.foreignDiscount, coupled.foreignSpread,
                                                    domesticDiscount, domesticForecast,
                                                    pillars[i].xccy, reference, zeroDc),
                    pillars[i].xccy.spread, 1e-9);
        CHECK_CLOSE("coupled mtm basis reprice",
                    markets::impliedBasisSpread(coupled.foreignSpread, pillars[i].basis, reference,
                                                zeroDc, &coupled.foreignDiscount),
                    pillars[i].basis.spread, 1e-9);
    }
}

/// FX swap points with the quote currency as collateral recover a synthetic
/// foreign (base) discount curve and reprice every pillar exactly.
void testFxSwapPointRecovery() {
    const datetime::Date reference(2026, 9, 29);
    const datetime::DayCounter zeroDc(datetime::DayCount::Actual365Fixed);
    const auto buildCurve = [&](double base, double slope) {
        std::vector<datetime::Date> dates;
        std::vector<double> zeros;
        for (int years = 1; years <= 5; ++years) {
            const datetime::Date date = reference.plusYears(years);
            dates.push_back(date);
            zeros.push_back(base + slope * datetime::yearFraction(reference, date, zeroDc));
        }
        return DiscountCurve<double>(reference, dates, zeroDc, zeros,
                                     InterpolationSpace::LogDiscount, InterpolationScheme::Linear);
    };
    const DiscountCurve<double> domesticDiscount = buildCurve(0.040, 0.0005);
    const DiscountCurve<double> foreignTarget = buildCurve(0.028, 0.0006);
    const double spot = 1.10;

    std::vector<markets::FxSwapPillar> pillars;
    for (int years = 1; years <= 5; ++years) {
        markets::FxSwapPillar pillar;
        pillar.start = reference.plusDays(2);
        pillar.maturity = reference.plusYears(years);
        pillar.spot = spot;
        pillar.isFxBaseCollateral = false;
        pillar.points = markets::impliedFxForwardPoints(foreignTarget, domesticDiscount, spot,
                                                        pillar, reference, zeroDc);
        EXPECT_TRUE(std::isfinite(pillar.points));
        EXPECT_LT(std::abs(pillar.points), 0.5);
        pillars.push_back(pillar);
    }

    const DiscountCurve<double> curve = markets::bootstrapFxDiscountCurve(
        domesticDiscount, reference, zeroDc, InterpolationSpace::LogDiscount,
        InterpolationScheme::Linear, pillars);
    EXPECT_EQ(curve.size(), pillars.size() + 1);
    for (std::size_t i = 0; i < pillars.size(); ++i) {
        CHECK_CLOSE("fx points recovered zero", curve.zeros()[i + 1], foreignTarget.zeros()[i + 1],
                    1e-10);
        CHECK_CLOSE(
            "fx points reprice",
            markets::fxPillarImpliedQuote(curve, domesticDiscount, pillars[i], reference, zeroDc),
            pillars[i].points, 1e-10);
    }

    const std::vector<markets::FxSwapPillar> unsorted{pillars[2], pillars[1]};
    EXPECT_THROW((void)(markets::bootstrapFxDiscountCurve(domesticDiscount, reference, zeroDc,
                                                          InterpolationSpace::LogDiscount,
                                                          InterpolationScheme::Linear, unsorted)),
                 std::invalid_argument);
}

/// Base-currency collateral and outright-quoted pillars solve the mirror CIP
/// branch, where the unknown foreign (quote) curve sits in the denominator.
void testFxSwapBaseCollateralOutright() {
    const datetime::Date reference(2026, 9, 29);
    const datetime::DayCounter zeroDc(datetime::DayCount::Actual365Fixed);
    const auto buildCurve = [&](double base, double slope) {
        std::vector<datetime::Date> dates;
        std::vector<double> zeros;
        for (int years = 1; years <= 5; ++years) {
            const datetime::Date date = reference.plusYears(years);
            dates.push_back(date);
            zeros.push_back(base + slope * datetime::yearFraction(reference, date, zeroDc));
        }
        return DiscountCurve<double>(reference, dates, zeroDc, zeros,
                                     InterpolationSpace::LogDiscount, InterpolationScheme::Linear);
    };
    const DiscountCurve<double> domesticDiscount = buildCurve(0.040, 0.0005);
    const DiscountCurve<double> foreignTarget = buildCurve(0.001, 0.0002);
    const double spot = 157.25;

    std::vector<markets::FxSwapPillar> pillars;
    for (int years = 1; years <= 5; ++years) {
        markets::FxSwapPillar pillar;
        pillar.start = reference.plusDays(2);
        pillar.maturity = reference.plusYears(years);
        pillar.spot = spot;
        pillar.isFxBaseCollateral = true;
        pillar.convention = markets::QuoteConvention::Outright;
        pillar.outright = markets::impliedFxOutright(foreignTarget, domesticDiscount, spot, pillar,
                                                     reference, zeroDc);
        EXPECT_TRUE(std::isfinite(pillar.outright));
        EXPECT_GT(pillar.outright, 0.0);
        pillars.push_back(pillar);
    }

    const DiscountCurve<double> curve = markets::bootstrapFxDiscountCurve(
        domesticDiscount, reference, zeroDc, InterpolationSpace::LogDiscount,
        InterpolationScheme::Linear, pillars);
    for (std::size_t i = 0; i < pillars.size(); ++i) {
        CHECK_CLOSE("fx outright recovered zero", curve.zeros()[i + 1],
                    foreignTarget.zeros()[i + 1], 1e-10);
        CHECK_CLOSE(
            "fx outright reprice",
            markets::fxPillarImpliedQuote(curve, domesticDiscount, pillars[i], reference, zeroDc),
            pillars[i].outright, 1e-10);
    }
}

/// After the points bootstrap the model forward equals `FXRate::forward`, for
/// both collateral directions.
void testFxSwapCipAgainstFXRateForward() {
    const datetime::Date reference(2026, 9, 29);
    const datetime::DayCounter zeroDc(datetime::DayCount::Actual365Fixed);
    const auto buildCurve = [&](double base, double slope) {
        std::vector<datetime::Date> dates;
        std::vector<double> zeros;
        for (int years = 1; years <= 3; ++years) {
            const datetime::Date date = reference.plusYears(years);
            dates.push_back(date);
            zeros.push_back(base + slope * datetime::yearFraction(reference, date, zeroDc));
        }
        return DiscountCurve<double>(reference, dates, zeroDc, zeros,
                                     InterpolationSpace::LogDiscount, InterpolationScheme::Linear);
    };
    const DiscountCurve<double> domesticDiscount = buildCurve(0.040, 0.0005);
    const DiscountCurve<double> foreignTarget = buildCurve(0.028, 0.0006);

    // Quote-currency collateral (EURUSD): base EUR is the unknown foreign curve.
    {
        SCOPED_TRACE("quote collateral");
        const double spot = 1.10;
        std::vector<markets::FxSwapPillar> pillars;
        for (int years = 1; years <= 3; ++years) {
            markets::FxSwapPillar pillar;
            pillar.maturity = reference.plusYears(years);
            pillar.spot = spot;
            pillar.points = markets::impliedFxForwardPoints(foreignTarget, domesticDiscount, spot,
                                                            pillar, reference, zeroDc);
            pillars.push_back(pillar);
        }
        const DiscountCurve<double> foreign = markets::bootstrapFxDiscountCurve(
            domesticDiscount, reference, zeroDc, InterpolationSpace::LogDiscount,
            InterpolationScheme::Linear, pillars);
        const markets::FXRate<double> fx(spot, foreign, domesticDiscount,
                                         markets::FXDescriptor("EUR", "USD", reference.toIso()));
        for (const markets::FxSwapPillar& pillar : pillars) {
            const double t = datetime::yearFraction(reference, pillar.maturity, zeroDc);
            CHECK_CLOSE("fx cip quote collateral", fx.forward(t), spot + pillar.points, 1e-12);
            CHECK_CLOSE("fx cip quote collateral model", fx.forward(t),
                        markets::impliedFxOutright(foreign, domesticDiscount, spot, pillar,
                                                   reference, zeroDc),
                        1e-10);
        }
    }

    // Base-currency collateral (USDJPY): base USD is the frozen collateral
    // curve and the quote JPY is the unknown foreign curve.
    {
        SCOPED_TRACE("base collateral");
        const double spot = 157.25;
        std::vector<markets::FxSwapPillar> pillars;
        for (int years = 1; years <= 3; ++years) {
            markets::FxSwapPillar pillar;
            pillar.maturity = reference.plusYears(years);
            pillar.spot = spot;
            pillar.isFxBaseCollateral = true;
            pillar.convention = markets::QuoteConvention::Outright;
            pillar.outright = markets::impliedFxOutright(foreignTarget, domesticDiscount, spot,
                                                         pillar, reference, zeroDc);
            pillars.push_back(pillar);
        }
        const DiscountCurve<double> foreign = markets::bootstrapFxDiscountCurve(
            domesticDiscount, reference, zeroDc, InterpolationSpace::LogDiscount,
            InterpolationScheme::Linear, pillars);
        const markets::FXRate<double> fx(spot, domesticDiscount, foreign,
                                         markets::FXDescriptor("USD", "JPY", reference.toIso()));
        for (const markets::FxSwapPillar& pillar : pillars) {
            const double t = datetime::yearFraction(reference, pillar.maturity, zeroDc);
            CHECK_CLOSE("fx cip base collateral", fx.forward(t), pillar.outright, 1e-10);
        }
    }
}

/// One maturity-ordered ladder mixing FX forward points at the short end with
/// cross-currency basis swaps long end recovers the synthetic foreign curve at
/// every node and reprices every quote, with both forecast modes.
void testMixedFxXccyLadderRecovery() {
    const datetime::Date reference(2026, 9, 29);
    const datetime::DayCounter zeroDc(datetime::DayCount::Actual365Fixed);
    const datetime::Calendar calendar = datetime::Calendar::noHolidays();
    const auto buildCurve = [&](double base, double slope) {
        std::vector<datetime::Date> dates;
        std::vector<double> zeros;
        for (int years = 1; years <= 3; ++years) {
            const datetime::Date date = reference.plusYears(years);
            dates.push_back(date);
            zeros.push_back(base + slope * datetime::yearFraction(reference, date, zeroDc));
        }
        return DiscountCurve<double>(reference, dates, zeroDc, zeros,
                                     InterpolationSpace::LogDiscount, InterpolationScheme::Linear);
    };
    const DiscountCurve<double> domesticDiscount = buildCurve(0.040, 0.0005);
    const DiscountCurve<double> domesticForecast = buildCurve(0.043, 0.0004);
    const DiscountCurve<double> foreignForecast = buildCurve(0.025, 0.0008);
    const DiscountCurve<double> foreignTarget = buildCurve(0.028, 0.0006);
    const double spot = 1.10;

    std::vector<markets::XccyMixedPillar> pillars;
    markets::FxSwapPillar fxPillar;
    fxPillar.start = reference.plusDays(2);
    fxPillar.maturity = reference.plusYears(1);
    fxPillar.spot = spot;
    fxPillar.points = markets::impliedFxForwardPoints(foreignTarget, domesticDiscount, spot,
                                                      fxPillar, reference, zeroDc);
    pillars.emplace_back(fxPillar);
    std::vector<markets::XccyPillar> xccyPillars;
    for (int years = 2; years <= 3; ++years) {
        markets::XccyPillar xccy;
        xccy.maturity = reference.plusYears(years);
        xccy.foreignTenor = datetime::Period(3, datetime::TimeUnit::Months);
        xccy.domesticTenor = datetime::Period(3, datetime::TimeUnit::Months);
        xccy.foreignCalendar = calendar;
        xccy.domesticCalendar = calendar;
        xccy.foreignDayCounter = datetime::DayCounter(datetime::DayCount::Actual360);
        xccy.domesticDayCounter = datetime::DayCounter(datetime::DayCount::Actual360);
        xccy.spread =
            markets::impliedXccyBasisSpread(foreignTarget, foreignForecast, domesticDiscount,
                                            domesticForecast, xccy, reference, zeroDc);
        EXPECT_TRUE(std::isfinite(xccy.spread));
        pillars.emplace_back(xccy);
        xccyPillars.push_back(xccy);
    }

    const DiscountCurve<double> curve = markets::bootstrapMixedXccyDiscountCurve(
        domesticDiscount, domesticForecast, foreignForecast, reference, zeroDc,
        InterpolationSpace::LogDiscount, InterpolationScheme::Linear, pillars);
    EXPECT_EQ(curve.size(), pillars.size() + 1);
    for (std::size_t i = 0; i < pillars.size(); ++i) {
        CHECK_CLOSE("mixed ladder recovered zero", curve.zeros()[i + 1],
                    foreignTarget.zeros()[i + 1], 1e-10);
    }
    CHECK_CLOSE("mixed ladder fx reprice",
                markets::fxPillarImpliedQuote(curve, domesticDiscount, fxPillar, reference, zeroDc),
                fxPillar.points, 1e-10);
    for (const markets::XccyPillar& xccy : xccyPillars) {
        CHECK_CLOSE("mixed ladder xccy reprice",
                    markets::impliedXccyBasisSpread(curve, foreignForecast, domesticDiscount,
                                                    domesticForecast, xccy, reference, zeroDc),
                    xccy.spread, 1e-10);
    }

    // Self-forecast mode solves the same residual with the trial curve as the
    // foreign floating forecast, so its own residuals must vanish too.
    const DiscountCurve<double> selfCurve = markets::bootstrapMixedXccyDiscountCurve(
        domesticDiscount, domesticForecast, reference, zeroDc, InterpolationSpace::LogDiscount,
        InterpolationScheme::Linear, pillars);
    for (const markets::XccyPillar& xccy : xccyPillars) {
        CHECK_CLOSE("mixed ladder self-forecast reprice",
                    markets::impliedXccyBasisSpread(selfCurve, selfCurve, domesticDiscount,
                                                    domesticForecast, xccy, reference, zeroDc),
                    xccy.spread, 1e-10);
    }

    const std::vector<markets::XccyMixedPillar> unsorted{pillars[2], pillars[1]};
    EXPECT_THROW((void)(markets::bootstrapMixedXccyDiscountCurve(
                     domesticDiscount, domesticForecast, foreignForecast, reference, zeroDc,
                     InterpolationSpace::LogDiscount, InterpolationScheme::Linear, unsorted)),
                 std::invalid_argument);
}

/// Copies a frozen legacy pillar into the migrated instrument; every field is
/// mirrored so the compare isolates the evaluator arithmetic.
markets::XccyPillar toInstrument(const legacy::XccyPillar& from) {
    markets::XccyPillar out;
    out.maturity = from.maturity;
    out.spread = from.spread;
    out.spreadOnForeignLeg = from.spreadOnForeignLeg;
    out.notional = from.notional == legacy::NotionalMode::MtM ? markets::XccyNotionalMode::MtM
                                                              : markets::XccyNotionalMode::Const;
    out.resetForeignLeg = from.resetForeignLeg;
    out.foreignTenor = from.foreignTenor;
    out.foreignCalendar = from.foreignCalendar;
    out.foreignDayCounter = from.foreignDayCounter;
    out.domesticTenor = from.domesticTenor;
    out.domesticCalendar = from.domesticCalendar;
    out.domesticDayCounter = from.domesticDayCounter;
    out.foreignPaymentLag = from.foreignPaymentLag;
    out.domesticPaymentLag = from.domesticPaymentLag;
    out.foreignBusinessDayConvention = from.foreignBusinessDayConvention;
    out.domesticBusinessDayConvention = from.domesticBusinessDayConvention;
    return out;
}

/// Old-vs-new bitwise oracle of the standalone evaluator and bootstrap: the
/// instrument's prepared `impliedQuote` and the bootstrap's node times and
/// zeros must match the frozen pre-migration arithmetic bit for bit, across
/// constant and resetting notionals and both spread sides.
void testXccyStandaloneBitwiseAgainstLegacy() {
    const datetime::Date reference(2026, 9, 29);
    const datetime::DayCounter zeroDc(datetime::DayCount::Actual365Fixed);
    const datetime::Calendar calendar = datetime::Calendar::noHolidays();
    std::vector<datetime::Date> dates{reference};
    for (int years = 1; years <= 5; ++years) {
        dates.push_back(reference.plusYears(years));
    }
    const auto buildCurve = [&](double base, double slope) {
        std::vector<double> zeros;
        for (std::size_t i = 1; i < dates.size(); ++i) {
            zeros.push_back(base + slope * datetime::yearFraction(reference, dates[i], zeroDc));
        }
        return DiscountCurve<double>(
            reference, std::vector<datetime::Date>(dates.begin() + 1, dates.end()), zeroDc, zeros,
            InterpolationSpace::LogDiscount, InterpolationScheme::Linear);
    };
    const DiscountCurve<double> domesticDiscount = buildCurve(0.040, 0.0005);
    const DiscountCurve<double> domesticForecast = buildCurve(0.043, 0.0004);
    const DiscountCurve<double> foreignForecast = buildCurve(0.025, 0.0008);
    const DiscountCurve<double> foreignTarget = buildCurve(0.028, 0.0006);
    const markets::XccySet<DiscountCurve<double>, DiscountCurve<double>, DiscountCurve<double>,
                           DiscountCurve<double>>
        curves{foreignTarget, foreignForecast, domesticDiscount, domesticForecast};

    std::vector<legacy::XccyPillar> legacyPillars;
    std::vector<markets::XccyPillar> pillars;
    for (std::size_t i = 1; i < dates.size(); ++i) {
        legacy::XccyPillar legacyPillar;
        legacyPillar.maturity = dates[i];
        legacyPillar.foreignTenor = datetime::Period(7, datetime::TimeUnit::Months);
        legacyPillar.domesticTenor = datetime::Period(3, datetime::TimeUnit::Months);
        legacyPillar.foreignPaymentLag = i % 2 == 0 ? 2 : 0;
        legacyPillar.domesticPaymentLag = i % 3 == 0 ? 1 : 0;
        legacyPillar.foreignBusinessDayConvention = datetime::BusinessDayConvention::Following;
        legacyPillar.foreignCalendar = calendar;
        legacyPillar.foreignDayCounter = datetime::DayCounter(datetime::DayCount::Actual360);
        legacyPillar.domesticCalendar = calendar;
        legacyPillar.domesticDayCounter = datetime::DayCounter(datetime::DayCount::Actual360);
        legacyPillar.spreadOnForeignLeg = i % 2 == 0;
        if (i == 4) {
            legacyPillar.notional = legacy::NotionalMode::MtM;
            legacyPillar.resetForeignLeg = false;
        }
        legacyPillar.spread =
            legacy::impliedXccyBasisSpread(foreignTarget, foreignForecast, domesticDiscount,
                                           domesticForecast, legacyPillar, reference, zeroDc);

        markets::XccyPillar pillar = toInstrument(legacyPillar);
        const double wrapper =
            markets::impliedXccyBasisSpread(foreignTarget, foreignForecast, domesticDiscount,
                                            domesticForecast, pillar, reference, zeroDc);
        EXPECT_TRUE(sameBits(wrapper, legacyPillar.spread));
        markets::XccyPillar prepared = pillar;
        prepared.prepare(reference, zeroDc);
        EXPECT_TRUE(sameBits(prepared.template impliedQuote<double>(curves), legacyPillar.spread));
        EXPECT_TRUE(prepared.date() ==
                    legacyPillar.foreignCalendar.adjust(legacyPillar.maturity,
                                                        legacyPillar.foreignBusinessDayConvention));
        EXPECT_EQ(prepared.kindName(), "Xccy");
        pillars.push_back(pillar);
        legacyPillars.push_back(legacyPillar);
    }

    const DiscountCurve<double> legacyCurve = legacy::bootstrapXccyDiscountCurve(
        domesticDiscount, domesticForecast, foreignForecast, reference, zeroDc,
        InterpolationSpace::LogDiscount, InterpolationScheme::Linear, legacyPillars);
    const DiscountCurve<double> curve = markets::bootstrapXccyDiscountCurve(
        domesticDiscount, domesticForecast, foreignForecast, reference, zeroDc,
        InterpolationSpace::LogDiscount, InterpolationScheme::Linear, pillars);
    ASSERT_EQ(legacyCurve.size(), curve.size());
    for (std::size_t i = 0; i < curve.size(); ++i) {
        EXPECT_TRUE(sameBits(legacyCurve.times()[i], curve.times()[i]));
        EXPECT_TRUE(sameBits(legacyCurve.zeros()[i], curve.zeros()[i]));
        EXPECT_TRUE(sameBits(legacyCurve.discount(legacyCurve.times()[i]),
                             curve.discount(curve.times()[i])));
    }
}

/// Old-vs-new bitwise oracle of the mixed FX/xccy ladder in explicit- and
/// self-forecast modes: node vectors and zeros match the frozen pre-migration
/// drive, including the FX branch measured from the reference date.
void testXccyMixedBitwiseAgainstLegacy() {
    const datetime::Date reference(2026, 9, 29);
    const datetime::DayCounter zeroDc(datetime::DayCount::Actual365Fixed);
    const datetime::Calendar calendar = datetime::Calendar::noHolidays();
    const auto buildCurve = [&](double base, double slope) {
        const std::vector<datetime::Date> dates{reference.plusMonths(6), reference.plusYears(1),
                                                reference.plusYears(2), reference.plusYears(3),
                                                reference.plusYears(4)};
        std::vector<double> zeros;
        for (const datetime::Date& date : dates) {
            zeros.push_back(base + slope * datetime::yearFraction(reference, date, zeroDc));
        }
        return DiscountCurve<double>(reference, dates, zeroDc, zeros,
                                     InterpolationSpace::LogDiscount, InterpolationScheme::Linear);
    };
    const DiscountCurve<double> domesticDiscount = buildCurve(0.040, 0.0005);
    const DiscountCurve<double> domesticForecast = buildCurve(0.043, 0.0004);
    const DiscountCurve<double> foreignForecast = buildCurve(0.025, 0.0008);
    const DiscountCurve<double> foreignTarget = buildCurve(0.028, 0.0006);
    const double spot = 1.10;

    std::vector<legacy::MixedPillar> legacyPillars;
    std::vector<markets::XccyMixedPillar> pillars;
    markets::FxSwapPillar fxPillar;
    fxPillar.start = reference.plusDays(2);
    fxPillar.maturity = reference.plusMonths(6);
    fxPillar.spot = spot;
    fxPillar.points = markets::impliedFxForwardPoints(foreignTarget, domesticDiscount, spot,
                                                      fxPillar, reference, zeroDc);
    legacyPillars.emplace_back(fxPillar);
    pillars.emplace_back(fxPillar);

    for (int years = 1; years <= 4; ++years) {
        legacy::XccyPillar legacyPillar;
        legacyPillar.maturity = reference.plusYears(years);
        legacyPillar.foreignTenor = datetime::Period(3, datetime::TimeUnit::Months);
        legacyPillar.domesticTenor = datetime::Period(3, datetime::TimeUnit::Months);
        legacyPillar.foreignPaymentLag = 2;
        legacyPillar.domesticPaymentLag = 1;
        legacyPillar.foreignCalendar = calendar;
        legacyPillar.domesticCalendar = calendar;
        legacyPillar.foreignDayCounter = datetime::DayCounter(datetime::DayCount::Actual360);
        legacyPillar.domesticDayCounter = datetime::DayCounter(datetime::DayCount::Actual360);
        legacyPillar.spread =
            legacy::impliedXccyBasisSpread(foreignTarget, foreignForecast, domesticDiscount,
                                           domesticForecast, legacyPillar, reference, zeroDc);
        legacyPillars.emplace_back(legacyPillar);
        pillars.emplace_back(toInstrument(legacyPillar));
    }

    const auto compare = [](const DiscountCurve<double>& expected,
                            const DiscountCurve<double>& actual) {
        ASSERT_EQ(expected.size(), actual.size());
        for (std::size_t i = 0; i < actual.size(); ++i) {
            EXPECT_TRUE(sameBits(expected.times()[i], actual.times()[i]));
            EXPECT_TRUE(sameBits(expected.zeros()[i], actual.zeros()[i]));
        }
    };
    const DiscountCurve<double> legacyExplicit = legacy::bootstrapMixedXccyDiscountCurve(
        domesticDiscount, domesticForecast, foreignForecast, reference, zeroDc,
        InterpolationSpace::LogDiscount, InterpolationScheme::Linear, legacyPillars);
    const DiscountCurve<double> explicitCurve = markets::bootstrapMixedXccyDiscountCurve(
        domesticDiscount, domesticForecast, foreignForecast, reference, zeroDc,
        InterpolationSpace::LogDiscount, InterpolationScheme::Linear, pillars);
    SCOPED_TRACE("explicit forecast");
    compare(legacyExplicit, explicitCurve);

    const DiscountCurve<double> legacySelf = legacy::bootstrapMixedXccyDiscountCurve(
        domesticDiscount, domesticForecast, reference, zeroDc, InterpolationSpace::LogDiscount,
        InterpolationScheme::Linear, legacyPillars);
    const DiscountCurve<double> selfCurve = markets::bootstrapMixedXccyDiscountCurve(
        domesticDiscount, domesticForecast, reference, zeroDc, InterpolationSpace::LogDiscount,
        InterpolationScheme::Linear, pillars);
    SCOPED_TRACE("self forecast");
    compare(legacySelf, selfCurve);
}

/// Old-vs-new bitwise oracle of the coupled bootstrap: the fixed-point
/// metadata, the discount nodes and the spread nodes match the frozen
/// pre-migration drive for both the converged pass and the joint-LM fallback.
void testXccyCoupledBitwiseAgainstLegacy() {
    const datetime::Date reference(2026, 9, 29);
    const datetime::DayCounter zeroDayCounter(datetime::DayCount::Actual365Fixed);
    const datetime::Calendar calendar = datetime::Calendar::noHolidays();
    const auto buildCurve = [&](double base, double slope, int maxYears) {
        std::vector<datetime::Date> pillarDates;
        std::vector<double> zeros;
        for (int years = 1; years <= maxYears; ++years) {
            const datetime::Date date = reference.plusYears(years);
            pillarDates.push_back(date);
            zeros.push_back(base + slope * datetime::yearFraction(reference, date, zeroDayCounter));
        }
        return DiscountCurve<double>(reference, pillarDates, zeroDayCounter, zeros,
                                     InterpolationSpace::LogDiscount, InterpolationScheme::Linear);
    };
    const DiscountCurve<double> domesticDiscount = buildCurve(0.040, 0.0005, 5);
    const DiscountCurve<double> domesticForecast = domesticDiscount;
    const DiscountCurve<double> xccyTarget = buildCurve(0.030, 0.0005, 3);
    auto foreignBase = std::make_shared<DiscountCurve<double>>(buildCurve(0.025, 0.0005, 3));
    const std::vector<double> spreadTimes{0.0, 1.0, 2.0, 3.0};
    const std::vector<double> spreadNodes{0.0, 0.0010, 0.0013, 0.0016};
    const markets::SpreadCurve<double> spreadTarget(foreignBase, spreadTimes, spreadNodes,
                                                    InterpolationScheme::Linear);

    std::vector<legacy::CoupledPillar> legacyPillars;
    std::vector<markets::XccyCoupledPillar> pillars;
    for (int years = 1; years <= 3; ++years) {
        const datetime::Date maturity = reference.plusYears(years);
        legacy::CoupledPillar legacyPillar;
        legacyPillar.xccy.maturity = maturity;
        legacyPillar.xccy.foreignCalendar = calendar;
        legacyPillar.xccy.domesticCalendar = calendar;
        legacyPillar.xccy.spread = legacy::impliedXccyBasisSpread(
            xccyTarget, spreadTarget, domesticDiscount, domesticForecast, legacyPillar.xccy,
            reference, zeroDayCounter);
        legacyPillar.basis.maturity = maturity;
        legacyPillar.basis.calendar = calendar;
        legacyPillar.basis.spread = markets::impliedBasisSpread(
            spreadTarget, legacyPillar.basis, reference, zeroDayCounter, &xccyTarget);
        markets::XccyCoupledPillar pillar;
        pillar.xccy = toInstrument(legacyPillar.xccy);
        pillar.basis = legacyPillar.basis;
        legacyPillars.push_back(legacyPillar);
        pillars.push_back(pillar);
    }

    const auto compare = [](const markets::XccyCoupledResult& expected,
                            const markets::XccyCoupledResult& actual) {
        EXPECT_EQ(expected.passes, actual.passes);
        EXPECT_TRUE(sameBits(expected.update, actual.update));
        EXPECT_EQ(expected.converged, actual.converged);
        EXPECT_EQ(expected.usedJointFallback, actual.usedJointFallback);
        ASSERT_EQ(expected.foreignDiscount.size(), actual.foreignDiscount.size());
        for (std::size_t i = 0; i < actual.foreignDiscount.size(); ++i) {
            EXPECT_TRUE(
                sameBits(expected.foreignDiscount.times()[i], actual.foreignDiscount.times()[i]));
            EXPECT_TRUE(
                sameBits(expected.foreignDiscount.zeros()[i], actual.foreignDiscount.zeros()[i]));
        }
        const auto& expectedSpread = expected.foreignSpread.spreadNodes();
        const auto& actualSpread = actual.foreignSpread.spreadNodes();
        ASSERT_EQ(expectedSpread.size(), actualSpread.size());
        for (std::size_t i = 0; i < actualSpread.size(); ++i) {
            EXPECT_TRUE(sameBits(expectedSpread.times()[i], actualSpread.times()[i]));
            EXPECT_TRUE(sameBits(expectedSpread.zeros()[i], actualSpread.zeros()[i]));
        }
    };

    const markets::XccyCoupledResult legacyResult = legacy::bootstrapXccyCoupled(
        domesticDiscount, domesticForecast, foreignBase, reference, zeroDayCounter,
        InterpolationSpace::LogDiscount, InterpolationScheme::Linear, legacyPillars);
    const markets::XccyCoupledResult result = markets::bootstrapXccyCoupled(
        domesticDiscount, domesticForecast, foreignBase, reference, zeroDayCounter,
        InterpolationSpace::LogDiscount, InterpolationScheme::Linear, pillars);
    EXPECT_TRUE(result.converged);
    EXPECT_TRUE(!result.usedJointFallback);
    SCOPED_TRACE("fixed point");
    compare(legacyResult, result);

    math::FixedPointOptions options;
    options.maxPasses = 1;
    const markets::XccyCoupledResult legacyFallback = legacy::bootstrapXccyCoupled(
        domesticDiscount, domesticForecast, foreignBase, reference, zeroDayCounter,
        InterpolationSpace::LogDiscount, InterpolationScheme::Linear, legacyPillars, 1e-14, 0.0, 1,
        options);
    const markets::XccyCoupledResult fallback =
        markets::bootstrapXccyCoupled(domesticDiscount, domesticForecast, foreignBase, reference,
                                      zeroDayCounter, InterpolationSpace::LogDiscount,
                                      InterpolationScheme::Linear, pillars, 1e-14, 0.0, 1, options);
    EXPECT_TRUE(fallback.usedJointFallback);
    EXPECT_EQ(fallback.passes, 1);
    SCOPED_TRACE("joint LM fallback");
    compare(legacyFallback, fallback);
}

} // namespace

class XccyBootstrapTest : public ::testing::Test {};

TEST_F(XccyBootstrapTest, constNotionalBootstrapRecoversAndRejectsUnsorted) {
    testXccyConstNotionalBootstrap();
}

TEST_F(XccyBootstrapTest, spreadForecastsAndSpreadSide) {
    testXccySpreadForecastsAndSpreadSide();
}

TEST_F(XccyBootstrapTest, coupledBootstrapRecoversDiscountAndSpread) {
    testXccyCoupledBootstrap();
}

TEST_F(XccyBootstrapTest, forecastNodeDateConversion) {
    testForecastNodeDateConversion();
}

TEST_F(XccyBootstrapTest, jointFallbackAndRisk) {
    testXccyJointFallbackAndRisk();
}

TEST_F(XccyBootstrapTest, analyticRowsAgainstFiniteDifference) {
    testXccyAnalyticRowsAgainstFiniteDifference();
}

TEST_F(XccyBootstrapTest, gammaFiniteDifference) {
    testXccyGammaFiniteDifference();
}

TEST_F(XccyBootstrapTest, sharedForecastSplit) {
    testXccySharedForecastSplit();
}

TEST_F(XccyBootstrapTest, mtmTelescopingGolden) {
    testXccyMtMTelescoping();
}

TEST_F(XccyBootstrapTest, mtmGoldenSpreads) {
    testXccyMtMGoldenSpreads();
}

TEST_F(XccyBootstrapTest, mixedNotionalAndCoupledMtM) {
    SCOPED_TRACE("mixed notionals");
    testXccyMixedNotionalBootstrap();
    SCOPED_TRACE("coupled mtm");
    testXccyCoupledMtM();
}

/// The known-flow materializer emits the quoted spread leg's coupons in the
/// leg's currency at its payment schedule, and rejects resetting notionals.
void testXccyCashflowsMaterialization() {
    const datetime::Date reference(2026, 9, 29);
    const datetime::Calendar calendar = datetime::Calendar::noHolidays();
    const instruments::Currency eur = instruments::currencyFromCode("EUR");
    const instruments::Currency usd = instruments::currencyFromCode("USD");
    markets::XccyPillar pillar;
    pillar.maturity = reference.plusYears(1);
    pillar.spread = 0.0011;
    pillar.spreadOnForeignLeg = true;
    pillar.foreignTenor = datetime::Period(3, datetime::TimeUnit::Months);
    pillar.domesticTenor = datetime::Period(3, datetime::TimeUnit::Months);
    pillar.foreignCalendar = calendar;
    pillar.domesticCalendar = calendar;
    pillar.foreignDayCounter = datetime::DayCounter(datetime::DayCount::Actual360);
    pillar.domesticDayCounter = datetime::DayCounter(datetime::DayCount::Actual360);

    const std::vector<instruments::Cashflow> flows =
        pricing::xccyCashflows(pillar, reference, 1.0e6, eur, usd);
    const datetime::Schedule schedule(reference, pillar.maturity, pillar.foreignTenor,
                                      pillar.foreignCalendar, pillar.foreignBusinessDayConvention,
                                      datetime::DateGeneration::Forward, false,
                                      datetime::BusinessDayConvention::Unadjusted);
    ASSERT_EQ(flows.size(), schedule.dates().size() - 1);
    for (std::size_t k = 1; k < schedule.dates().size(); ++k) {
        const double tau = datetime::yearFraction(schedule.dates()[k - 1], schedule.dates()[k],
                                                  pillar.foreignDayCounter);
        const datetime::Date payDate = pillar.foreignCalendar.advance(
            schedule.dates()[k],
            datetime::Period(pillar.foreignPaymentLag, datetime::TimeUnit::Days),
            pillar.foreignBusinessDayConvention);
        EXPECT_EQ(instruments::currencyCode(flows[k - 1].currency), "EUR");
        EXPECT_TRUE(flows[k - 1].payDate == payDate);
        CHECK_CLOSE("xccy cashflow amount", flows[k - 1].amount, 1.0e6 * pillar.spread * tau,
                    1e-12);
    }

    markets::XccyPillar mtm = pillar;
    mtm.notional = markets::XccyNotionalMode::MtM;
    EXPECT_THROW((void)(pricing::xccyCashflows(mtm, reference, 1.0e6, eur, usd)),
                 std::invalid_argument);
}

TEST_F(XccyBootstrapTest, cashflowsMaterialization) {
    testXccyCashflowsMaterialization();
}

TEST_F(XccyBootstrapTest, standaloneBitwiseAgainstLegacy) {
    testXccyStandaloneBitwiseAgainstLegacy();
}

TEST_F(XccyBootstrapTest, mixedBitwiseAgainstLegacy) {
    testXccyMixedBitwiseAgainstLegacy();
}

TEST_F(XccyBootstrapTest, coupledBitwiseAgainstLegacy) {
    testXccyCoupledBitwiseAgainstLegacy();
}

TEST_F(XccyBootstrapTest, fxSwapAndMixedLadderGates) {
    SCOPED_TRACE("fx points");
    testFxSwapPointRecovery();
    SCOPED_TRACE("fx base collateral outright");
    testFxSwapBaseCollateralOutright();
    SCOPED_TRACE("fx cip");
    testFxSwapCipAgainstFXRateForward();
    SCOPED_TRACE("mixed fx xccy ladder");
    testMixedFxXccyLadderRecovery();
}
