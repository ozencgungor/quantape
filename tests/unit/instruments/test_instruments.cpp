/**
 * @file test_instruments.cpp
 * @brief Instruments vocabulary and the bitwise discount-bootstrap gate
 *
 * Foundation and phase-2 gate for the plain-data instruments layer:
 *  - `Cashflow` layout and the `Currency` string round-trip,
 *  - `BootstrapInstrument` concept checks including a `std::variant` `Ladder`,
 *    for `double`, `var` and `fvar<var>`,
 *  - `CurvePillar` -> concept instrument adapter checks,
 *  - bitwise equality of `bootstrapDiscountCurve` against a frozen copy of the
 *    pre-migration solver on the bootstrap-validation, futures and EUR fixture
 *    strips (node times, node zeros, every pillar quote and discount samples at
 *    1Y/5Y/10Y/30Y), across all interpolation schemes and every future style.
 */

#include "quantape/math/StanMath.h"

#include "quantape/datetime/Imm.h"
#include "quantape/instruments/BootstrapInstrument.h"
#include "quantape/instruments/Cashflow.h"
#include "quantape/instruments/IrInstruments.h"
#include "quantape/markets/Curves/BootstrapInstrument.h"
#include "quantape/markets/Curves/CurveBuilder.h"
#include "quantape/markets/Curves/HullWhiteConvexity.h"
#include "quantape/math/Solvers/BrentSolver.h"
#include "quantape/pricing/Ir.h"

#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>
#include <variant>
#include <vector>

#include "support/GtestSupport.h"
#include "support/StanTapeFixture.h"
#include "support/fixtures/PaperEurCurves.h"

using namespace quantape;

namespace {

namespace dt = quantape::datetime;
namespace mk = quantape::markets;
namespace paper = quantape::tests::paper_eur;

const dt::Date kReference(2026, 9, 29);
const dt::DayCounter kZeroDc(dt::DayCount::Actual365Fixed);

/// Frozen copy of the pre-migration discount-curve evaluators, used as the
/// old-vs-new bitwise oracle. Kept independent of the library so the gate
/// compares arithmetic rather than re-running the migrated implementation.
namespace legacy {

/// Consecutive business-day fixing dates over `[effective, maturity]`.
std::vector<dt::Date> fixings(const dt::Calendar& calendar, const dt::Date& effective,
                              const dt::Date& maturity) {
    if (maturity < effective) {
        throw std::invalid_argument("businessDayFixings: maturity before effective");
    }
    std::vector<dt::Date> dates{effective};
    while (dates.back() < maturity) {
        const dt::Date next = calendar.advance(dates.back(), dt::Period(1, dt::TimeUnit::Days),
                                               dt::BusinessDayConvention::Following);
        if (!(next > dates.back())) {
            throw std::invalid_argument("businessDayFixings: calendar is not advancing");
        }
        dates.push_back(next >= maturity ? maturity : next);
    }
    return dates;
}

dt::Date legacyAdjustedMaturity(const mk::CurvePillar& pillar) {
    return pillar.calendar.adjust(pillar.maturity, pillar.businessDayConvention);
}

dt::Date legacyAdjustedStart(const mk::CurvePillar& pillar) {
    return pillar.calendar.adjust(pillar.start, pillar.businessDayConvention);
}

dt::Date riskMaturity(const mk::CurvePillar& pillar) {
    switch (pillar.kind) {
        case mk::PillarKind::Repo:
        case mk::PillarKind::Deposit:
        case mk::PillarKind::Fra:
        case mk::PillarKind::OisSwap:
            return legacyAdjustedMaturity(pillar);
        case mk::PillarKind::Future:
            return pillar.maturity;
    }
    return pillar.maturity;
}

std::string_view kindName(mk::PillarKind kind) noexcept {
    switch (kind) {
        case mk::PillarKind::Deposit:
            return "Deposit";
        case mk::PillarKind::Repo:
            return "Repo";
        case mk::PillarKind::Fra:
            return "Fra";
        case mk::PillarKind::Future:
            return "Future";
        case mk::PillarKind::OisSwap:
            return "OisSwap";
    }
    return "Unknown";
}

struct DepositTimes {
    double tau = 0.0;
    double t = 0.0;
};

struct FraTimes {
    double t1 = 0.0;
    double t2 = 0.0;
    double tau = 0.0;
    double convexityExponent = 0.0;
};

struct FutureTimes {
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

struct OisCouponTimes {
    double tau = 0.0;
    double tPay = 0.0;
    double tStart = 0.0;
    double tEnd = 0.0;
    bool firstFixed = false;
    double firstRate = 0.0;
};

struct QuoteTimes {
    mk::PillarKind kind = mk::PillarKind::Deposit;
    DepositTimes deposit;
    FraTimes fra;
    FutureTimes future;
    std::vector<OisCouponTimes> oisCoupons;
};

FutureTimes makeFutureTimes(const dt::Date& start, const dt::Date& maturity,
                            const dt::Calendar& calendar, const dt::DayCounter& quoteDayCounter,
                            mk::FutureStyle futureStyle, mk::AveragingStyle averagingStyle,
                            double convexityAdjustment, const dt::Date& referenceDate,
                            const dt::DayCounter& zeroDayCounter, std::string_view context) {
    FutureTimes times;
    times.convexity = convexityAdjustment;
    if (futureStyle == mk::FutureStyle::Averaged &&
        averagingStyle == mk::AveragingStyle::Compounded) {
        times.style = FutureTimes::Style::AveragedCompounded;
        const std::vector<dt::Date> grid = fixings(calendar, start, maturity);
        if (grid.size() < 2) {
            throw std::invalid_argument(std::string(context) +
                                        ": empty averaged futures reference period");
        }
        times.t1 = dt::yearFraction(referenceDate, start, zeroDayCounter);
        times.tau = dt::yearFraction(start, maturity, quoteDayCounter);
        if (!(times.tau > 0.0)) {
            throw std::invalid_argument(std::string(context) + ": non-positive futures accrual");
        }
        if (!(times.t1 >= 0.0)) {
            throw std::invalid_argument(std::string(context) +
                                        ": futures fixing before the reference date");
        }
        times.previousTimes.reserve(grid.size() - 1);
        times.currentTimes.reserve(grid.size() - 1);
        for (std::size_t k = 1; k < grid.size(); ++k) {
            times.previousTimes.push_back(
                dt::yearFraction(referenceDate, grid[k - 1], zeroDayCounter));
            times.currentTimes.push_back(dt::yearFraction(referenceDate, grid[k], zeroDayCounter));
        }
        return times;
    }
    if (futureStyle == mk::FutureStyle::Averaged) {
        times.style = FutureTimes::Style::AveragedArithmetic;
        const std::vector<dt::Date> grid = fixings(calendar, start, maturity);
        if (grid.size() < 2) {
            throw std::invalid_argument(std::string(context) +
                                        ": empty averaged futures reference period");
        }
        times.t1 = dt::yearFraction(referenceDate, start, zeroDayCounter);
        if (!(times.t1 >= 0.0)) {
            throw std::invalid_argument(std::string(context) +
                                        ": futures fixing before the reference date");
        }
        times.previousTimes.reserve(grid.size() - 1);
        times.currentTimes.reserve(grid.size() - 1);
        times.accrualTaus.reserve(grid.size() - 1);
        for (std::size_t k = 1; k < grid.size(); ++k) {
            const double tau = dt::yearFraction(grid[k - 1], grid[k], quoteDayCounter);
            if (!(tau > 0.0)) {
                throw std::invalid_argument(std::string(context) +
                                            ": non-positive averaged futures accrual");
            }
            times.previousTimes.push_back(
                dt::yearFraction(referenceDate, grid[k - 1], zeroDayCounter));
            times.currentTimes.push_back(dt::yearFraction(referenceDate, grid[k], zeroDayCounter));
            times.accrualTaus.push_back(tau);
        }
        return times;
    }
    times.t1 = dt::yearFraction(referenceDate, start, zeroDayCounter);
    times.t2 = dt::yearFraction(referenceDate, maturity, zeroDayCounter);
    times.tau = dt::yearFraction(start, maturity, quoteDayCounter);
    if (!(times.tau > 0.0)) {
        throw std::invalid_argument(std::string(context) + ": non-positive futures accrual");
    }
    if (!(times.t1 >= 0.0)) {
        throw std::invalid_argument(std::string(context) +
                                    ": futures fixing before the reference date");
    }
    return times;
}

QuoteTimes makeQuoteTimes(const mk::CurvePillar& pillar, const dt::Date& referenceDate,
                          const dt::DayCounter& zeroDayCounter) {
    QuoteTimes times;
    times.kind = pillar.kind;
    switch (pillar.kind) {
        case mk::PillarKind::Repo:
        case mk::PillarKind::Deposit: {
            const dt::Date maturity = legacyAdjustedMaturity(pillar);
            times.deposit.tau = dt::yearFraction(referenceDate, maturity, pillar.quoteDayCounter);
            if (!(times.deposit.tau > 0.0)) {
                throw std::invalid_argument("impliedQuote: non-positive deposit accrual");
            }
            times.deposit.t = dt::yearFraction(referenceDate, maturity, zeroDayCounter);
            break;
        }
        case mk::PillarKind::Fra: {
            const dt::Date start = legacyAdjustedStart(pillar);
            const dt::Date maturity = legacyAdjustedMaturity(pillar);
            times.fra.t1 = dt::yearFraction(referenceDate, start, zeroDayCounter);
            times.fra.t2 = dt::yearFraction(referenceDate, maturity, zeroDayCounter);
            times.fra.tau = dt::yearFraction(start, maturity, pillar.quoteDayCounter);
            times.fra.convexityExponent = pillar.fraConvexityExponent;
            if (!(times.fra.tau > 0.0)) {
                throw std::invalid_argument("impliedQuote: non-positive FRA accrual");
            }
            if (!(times.fra.t1 >= 0.0)) {
                throw std::invalid_argument("impliedQuote: FRA start before the reference date");
            }
            break;
        }
        case mk::PillarKind::Future:
            times.future = makeFutureTimes(pillar.start, pillar.maturity, pillar.calendar,
                                           pillar.quoteDayCounter, pillar.futureStyle,
                                           pillar.averagingStyle, pillar.convexityAdjustment,
                                           referenceDate, zeroDayCounter, "impliedQuote");
            break;
        case mk::PillarKind::OisSwap: {
            const dt::Date effective = pillar.start.serial() != 0 ? pillar.start : referenceDate;
            const dt::Schedule schedule(effective, pillar.maturity, pillar.fixedTenor,
                                        pillar.calendar, pillar.businessDayConvention,
                                        dt::DateGeneration::Forward, false,
                                        dt::BusinessDayConvention::Unadjusted);
            const std::vector<dt::Date>& dates = schedule.dates();
            times.oisCoupons.reserve(dates.size() - 1);
            for (std::size_t k = 1; k < dates.size(); ++k) {
                OisCouponTimes coupon;
                coupon.tau = dt::yearFraction(dates[k - 1], dates[k], pillar.quoteDayCounter);
                if (!(coupon.tau > 0.0)) {
                    throw std::invalid_argument("impliedQuote: non-positive OIS accrual");
                }
                const dt::Date payDate = pillar.calendar.advance(
                    dates[k], dt::Period(pillar.paymentLag, dt::TimeUnit::Days),
                    pillar.businessDayConvention);
                coupon.tPay = dt::yearFraction(referenceDate, payDate, zeroDayCounter);
                if (k == 1 && pillar.firstCouponFixed) {
                    coupon.firstFixed = true;
                    coupon.firstRate = pillar.firstCouponRate;
                } else {
                    coupon.tStart = dt::yearFraction(referenceDate, dates[k - 1], zeroDayCounter);
                    coupon.tEnd = dt::yearFraction(referenceDate, dates[k], zeroDayCounter);
                    if (!(coupon.tStart >= 0.0) || !(coupon.tEnd > 0.0)) {
                        throw std::invalid_argument(
                            "impliedQuote: coupons before the reference date must be fixed");
                    }
                }
                times.oisCoupons.push_back(coupon);
            }
            break;
        }
    }
    return times;
}

template <typename DoubleT>
DoubleT evaluate(const QuoteTimes& times, const mk::DiscountCurve<DoubleT>& curve) {
    switch (times.kind) {
        case mk::PillarKind::Repo:
        case mk::PillarKind::Deposit:
            return (1.0 / curve.discount(times.deposit.t) - 1.0) / times.deposit.tau;
        case mk::PillarKind::Fra: {
            const FraTimes& fra = times.fra;
            const DoubleT forward =
                (curve.discount(fra.t1) / curve.discount(fra.t2) - 1.0) / fra.tau;
            if (fra.convexityExponent == 0.0) {
                return forward;
            }
            using std::exp;
            return ((1.0 + forward * fra.tau) * exp(fra.convexityExponent) - 1.0) / fra.tau;
        }
        case mk::PillarKind::Future: {
            const FutureTimes& future = times.future;
            if (future.style == FutureTimes::Style::AveragedCompounded) {
                DoubleT accumulated = 1.0;
                DoubleT previousDiscount = 0.0;
                bool hasPrevious = false;
                for (std::size_t k = 0; k < future.previousTimes.size(); ++k) {
                    if (!hasPrevious) {
                        previousDiscount = curve.discount(future.previousTimes[k]);
                    }
                    const DoubleT currentDiscount = curve.discount(future.currentTimes[k]);
                    accumulated *= previousDiscount / currentDiscount;
                    previousDiscount = currentDiscount;
                    hasPrevious = true;
                }
                return (accumulated - 1.0) / future.tau + future.convexity;
            }
            if (future.style == FutureTimes::Style::AveragedArithmetic) {
                DoubleT sum = 0.0;
                DoubleT previousDiscount = 0.0;
                bool hasPrevious = false;
                for (std::size_t k = 0; k < future.previousTimes.size(); ++k) {
                    if (!hasPrevious) {
                        previousDiscount = curve.discount(future.previousTimes[k]);
                    }
                    const DoubleT currentDiscount = curve.discount(future.currentTimes[k]);
                    sum += (previousDiscount / currentDiscount - 1.0) / future.accrualTaus[k];
                    previousDiscount = currentDiscount;
                    hasPrevious = true;
                }
                return sum / static_cast<double>(future.previousTimes.size()) + future.convexity;
            }
            return (curve.discount(future.t1) / curve.discount(future.t2) - 1.0) / future.tau +
                   future.convexity;
        }
        case mk::PillarKind::OisSwap: {
            DoubleT annuity = 0.0;
            DoubleT floating = 0.0;
            DoubleT startDiscount = 0.0;
            bool hasStart = false;
            for (const OisCouponTimes& coupon : times.oisCoupons) {
                const DoubleT discountPay = curve.discount(coupon.tPay);
                annuity += coupon.tau * discountPay;
                if (coupon.firstFixed) {
                    floating += discountPay * coupon.tau * coupon.firstRate;
                    hasStart = false;
                    continue;
                }
                const DoubleT couponStart =
                    hasStart ? startDiscount : curve.discount(coupon.tStart);
                const DoubleT discountEnd = curve.discount(coupon.tEnd);
                floating += discountPay * (couponStart / discountEnd - 1.0);
                startDiscount = discountEnd;
                hasStart = true;
            }
            if (!(annuity > 0.0)) {
                throw std::invalid_argument("impliedQuote: non-positive OIS annuity");
            }
            return floating / annuity;
        }
    }
    throw std::invalid_argument("impliedQuote: unknown pillar kind");
}

struct ForecastSimpleTimes {
    double t1 = 0.0;
    double t2 = 0.0;
    double tau = 0.0;
};

struct ForecastIrsFloatTimes {
    double tau = 0.0;
    double tPay = 0.0;
    double tPrevious = 0.0;
    double tAccrual = 0.0;
    bool firstFixed = false;
    double firstRate = 0.0;
};

struct ForecastIrsAnnuityTimes {
    double tau = 0.0;
    double tPay = 0.0;
};

struct ForecastBasisTimes {
    double tau = 0.0;
    double tPrev = 0.0;
    double t = 0.0;
};

struct ForecastQuoteTimes {
    mk::ForecastPillar::Kind kind = mk::ForecastPillar::Kind::Deposit;
    ForecastSimpleTimes simple;
    FutureTimes future;
    std::vector<ForecastIrsFloatTimes> floatCoupons;
    std::vector<ForecastIrsAnnuityTimes> annuityCoupons;
    std::vector<ForecastBasisTimes> basisCoupons;
    bool spreadOnParentLeg = true;
};

dt::Date forecastQuotedMaturity(const mk::ForecastPillar& pillar) {
    switch (pillar.kind) {
        case mk::ForecastPillar::Kind::Irs:
            if (pillar.irs.maturity.serial() == 0) {
                throw std::invalid_argument(
                    "forecastPillarQuotedMaturity: IRS pillar is missing its maturity");
            }
            return pillar.irs.maturity;
        case mk::ForecastPillar::Kind::BasisSwap:
            if (pillar.basis.maturity.serial() == 0) {
                throw std::invalid_argument(
                    "forecastPillarQuotedMaturity: basis pillar is missing its maturity");
            }
            return pillar.basis.maturity;
        case mk::ForecastPillar::Kind::Deposit:
        case mk::ForecastPillar::Kind::Fra:
        case mk::ForecastPillar::Kind::Future:
            if (pillar.maturity.serial() == 0) {
                throw std::invalid_argument(
                    "forecastPillarQuotedMaturity: money-market pillar is missing its maturity");
            }
            return pillar.maturity;
    }
    throw std::invalid_argument("forecastPillarQuotedMaturity: unknown forecast pillar kind");
}

dt::Date forecastRiskMaturity(const mk::ForecastPillar& pillar) {
    const dt::Date quoted = forecastQuotedMaturity(pillar);
    switch (pillar.kind) {
        case mk::ForecastPillar::Kind::Irs:
            return pillar.irs.fixedCalendar.adjust(quoted, pillar.irs.businessDayConvention);
        case mk::ForecastPillar::Kind::Deposit:
        case mk::ForecastPillar::Kind::Fra:
            return pillar.calendar.adjust(quoted, pillar.businessDayConvention);
        case mk::ForecastPillar::Kind::Future:
            return quoted;
        case mk::ForecastPillar::Kind::BasisSwap:
            return pillar.basis.calendar.adjust(quoted, pillar.basis.businessDayConvention);
    }
    throw std::invalid_argument("forecastPillarRiskMaturity: unknown forecast pillar kind");
}

double forecastTarget(const mk::ForecastPillar& pillar) {
    switch (pillar.kind) {
        case mk::ForecastPillar::Kind::Irs:
            (void)forecastQuotedMaturity(pillar);
            if (!(pillar.irs.floatTenor.length() > 0) || !(pillar.irs.fixedTenor.length() > 0)) {
                throw std::invalid_argument(
                    "forecastPillarTarget: IRS pillar is missing a leg tenor");
            }
            if (pillar.irs.quote == 0.0 && pillar.quote != 0.0) {
                throw std::invalid_argument(
                    "forecastPillarTarget: quote is set on the top-level field of an IRS pillar");
            }
            return pillar.irs.quote;
        case mk::ForecastPillar::Kind::BasisSwap:
            (void)forecastQuotedMaturity(pillar);
            if (!(pillar.basis.floatTenor.length() > 0)) {
                throw std::invalid_argument(
                    "forecastPillarTarget: basis pillar is missing its float tenor");
            }
            if (pillar.basis.spread == 0.0 && pillar.quote != 0.0) {
                throw std::invalid_argument(
                    "forecastPillarTarget: quote is set on the top-level field of a basis pillar");
            }
            return pillar.basis.spread;
        case mk::ForecastPillar::Kind::Deposit:
        case mk::ForecastPillar::Kind::Fra:
        case mk::ForecastPillar::Kind::Future:
            (void)forecastQuotedMaturity(pillar);
            return pillar.quote;
    }
    throw std::invalid_argument("forecastPillarTarget: unknown forecast pillar kind");
}

ForecastQuoteTimes makeForecastQuoteTimes(const mk::ForecastPillar& pillar,
                                          const dt::Date& referenceDate,
                                          const dt::DayCounter& zeroDayCounter) {
    ForecastQuoteTimes times;
    times.kind = pillar.kind;
    switch (pillar.kind) {
        case mk::ForecastPillar::Kind::Deposit:
        case mk::ForecastPillar::Kind::Fra: {
            const dt::Date start = pillar.start.serial() != 0 ? pillar.start : referenceDate;
            const dt::Date maturity =
                pillar.calendar.adjust(pillar.maturity, pillar.businessDayConvention);
            times.simple.t1 = dt::yearFraction(referenceDate, start, zeroDayCounter);
            times.simple.t2 = dt::yearFraction(referenceDate, maturity, zeroDayCounter);
            times.simple.tau = dt::yearFraction(start, maturity, pillar.quoteDayCounter);
            if (!(times.simple.tau > 0.0)) {
                throw std::invalid_argument("impliedSimpleForward: non-positive accrual");
            }
            if (!(times.simple.t1 >= 0.0)) {
                throw std::invalid_argument(
                    "impliedSimpleForward: start before the reference date");
            }
            break;
        }
        case mk::ForecastPillar::Kind::Future:
            times.future = makeFutureTimes(pillar.start, pillar.maturity, pillar.calendar,
                                           pillar.quoteDayCounter, pillar.futureStyle,
                                           pillar.averagingStyle, pillar.convexityAdjustment,
                                           referenceDate, zeroDayCounter, "impliedForecastFuture");
            break;
        case mk::ForecastPillar::Kind::Irs: {
            const dt::Date effective =
                pillar.irs.start.serial() != 0 ? pillar.irs.start : referenceDate;
            const dt::Schedule floatSchedule(
                effective, pillar.irs.maturity, pillar.irs.floatTenor, pillar.irs.floatCalendar,
                pillar.irs.businessDayConvention, dt::DateGeneration::Forward, false,
                dt::BusinessDayConvention::Unadjusted);
            const std::vector<dt::Date>& dates = floatSchedule.dates();
            times.floatCoupons.reserve(dates.size() - 1);
            for (std::size_t k = 1; k < dates.size(); ++k) {
                ForecastIrsFloatTimes coupon;
                coupon.tau = dt::yearFraction(dates[k - 1], dates[k], pillar.irs.floatDayCounter);
                if (!(coupon.tau > 0.0)) {
                    throw std::invalid_argument("impliedIrsRate: non-positive float accrual");
                }
                const dt::Date payDate = pillar.irs.floatCalendar.advance(
                    dates[k], dt::Period(pillar.irs.paymentLag, dt::TimeUnit::Days),
                    pillar.irs.businessDayConvention);
                coupon.tPay = dt::yearFraction(referenceDate, payDate, zeroDayCounter);
                if (k == 1 && pillar.irs.firstCouponFixed) {
                    coupon.firstFixed = true;
                    coupon.firstRate = pillar.irs.firstCouponRate;
                } else {
                    coupon.tPrevious =
                        dt::yearFraction(referenceDate, dates[k - 1], zeroDayCounter);
                    coupon.tAccrual = dt::yearFraction(referenceDate, dates[k], zeroDayCounter);
                    if (!(coupon.tPrevious >= 0.0) || !(coupon.tAccrual > 0.0)) {
                        throw std::invalid_argument(
                            "impliedIrsRate: coupons before the reference date must be fixed");
                    }
                }
                times.floatCoupons.push_back(coupon);
            }
            const dt::Schedule fixedSchedule(
                effective, pillar.irs.maturity, pillar.irs.fixedTenor, pillar.irs.fixedCalendar,
                pillar.irs.businessDayConvention, dt::DateGeneration::Forward, false,
                dt::BusinessDayConvention::Unadjusted);
            const std::vector<dt::Date>& fixedDates = fixedSchedule.dates();
            times.annuityCoupons.reserve(fixedDates.size() - 1);
            for (std::size_t k = 1; k < fixedDates.size(); ++k) {
                ForecastIrsAnnuityTimes coupon;
                coupon.tau =
                    dt::yearFraction(fixedDates[k - 1], fixedDates[k], pillar.irs.fixedDayCounter);
                const dt::Date payDate = pillar.irs.fixedCalendar.advance(
                    fixedDates[k], dt::Period(pillar.irs.paymentLag, dt::TimeUnit::Days),
                    pillar.irs.businessDayConvention);
                coupon.tPay = dt::yearFraction(referenceDate, payDate, zeroDayCounter);
                times.annuityCoupons.push_back(coupon);
            }
            break;
        }
        case mk::ForecastPillar::Kind::BasisSwap: {
            const dt::Schedule schedule(
                referenceDate, pillar.basis.maturity, pillar.basis.floatTenor,
                pillar.basis.calendar, pillar.basis.businessDayConvention,
                dt::DateGeneration::Forward, false, dt::BusinessDayConvention::Unadjusted);
            const std::vector<dt::Date>& dates = schedule.dates();
            times.basisCoupons.reserve(dates.size() - 1);
            for (std::size_t k = 1; k < dates.size(); ++k) {
                ForecastBasisTimes coupon;
                coupon.tau = dt::yearFraction(dates[k - 1], dates[k], pillar.basis.quoteDayCounter);
                if (!(coupon.tau > 0.0)) {
                    throw std::invalid_argument("impliedBasisSpread: non-positive float accrual");
                }
                coupon.tPrev = dt::yearFraction(referenceDate, dates[k - 1], zeroDayCounter);
                coupon.t = dt::yearFraction(referenceDate, dates[k], zeroDayCounter);
                if (!(coupon.tPrev >= 0.0) || !(coupon.t > 0.0)) {
                    throw std::invalid_argument(
                        "impliedBasisSpread: coupons before the reference date must be fixed");
                }
                times.basisCoupons.push_back(coupon);
            }
            times.spreadOnParentLeg = pillar.basis.spreadOnParentLeg;
            break;
        }
    }
    return times;
}

template <typename ForecastT, typename DiscountT>
double evaluateForecastQuote(const ForecastQuoteTimes& times, const ForecastT& forecast,
                             const DiscountT& discounting) {
    switch (times.kind) {
        case mk::ForecastPillar::Kind::Deposit:
        case mk::ForecastPillar::Kind::Fra: {
            const ForecastSimpleTimes& simple = times.simple;
            return (forecast.discount(simple.t1) / forecast.discount(simple.t2) - 1.0) / simple.tau;
        }
        case mk::ForecastPillar::Kind::Future: {
            const FutureTimes& future = times.future;
            if (future.style == FutureTimes::Style::AveragedCompounded) {
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
            if (future.style == FutureTimes::Style::AveragedArithmetic) {
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
        case mk::ForecastPillar::Kind::Irs: {
            double floatPv = 0.0;
            double previousAccrualDiscount = 0.0;
            bool hasPrevious = false;
            for (const ForecastIrsFloatTimes& coupon : times.floatCoupons) {
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
            for (const ForecastIrsAnnuityTimes& coupon : times.annuityCoupons) {
                annuity += coupon.tau * discounting.discount(coupon.tPay);
            }
            if (!(annuity > 0.0)) {
                throw std::invalid_argument("impliedIrsRate: non-positive fixed annuity");
            }
            return floatPv / annuity;
        }
        case mk::ForecastPillar::Kind::BasisSwap: {
            const auto& parentForecast = forecast.parent();
            double weightedChild = 0.0;
            double weightedParent = 0.0;
            double annuity = 0.0;
            double previousChild = 0.0;
            double previousParent = 0.0;
            bool hasPrevious = false;
            for (const ForecastBasisTimes& coupon : times.basisCoupons) {
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

} // namespace legacy

/// Frozen copy of the pre-migration discount-curve solver, used as the
/// old-vs-new bitwise oracle until the owning phase migrates the evaluators.
mk::DiscountCurve<double>
legacyBootstrapDiscountCurve(const dt::Date& referenceDate, const dt::DayCounter& zeroDayCounter,
                             mk::InterpolationSpace space, mk::InterpolationScheme scheme,
                             const std::vector<mk::CurvePillar>& pillars, double accuracy = 1e-14,
                             double tension = 0.0, int switchIndex = 1) {
    if (pillars.empty()) {
        throw std::invalid_argument("bootstrapDiscountCurve: no pillars");
    }
    const std::size_t count = pillars.size();
    std::vector<double> nodeTimes(count);
    std::vector<dt::Date> maturityDates(count);
    for (std::size_t i = 0; i < count; ++i) {
        const dt::Date maturity = legacy::riskMaturity(pillars[i]);
        nodeTimes[i] = dt::yearFraction(referenceDate, maturity, zeroDayCounter);
        if (!(nodeTimes[i] > (i == 0 ? 0.0 : nodeTimes[i - 1]))) {
            throw std::invalid_argument(
                "bootstrapDiscountCurve: maturities must be strictly increasing");
        }
        maturityDates[i] = maturity;
    }
    const dt::DayCounter trialZeroDayCounter(dt::DayCount::Actual365Fixed);
    std::vector<legacy::QuoteTimes> solveQuoteTimes(count);
    std::vector<legacy::QuoteTimes> checkQuoteTimes(count);
    for (std::size_t i = 0; i < count; ++i) {
        solveQuoteTimes[i] = legacy::makeQuoteTimes(pillars[i], referenceDate, trialZeroDayCounter);
        checkQuoteTimes[i] = legacy::makeQuoteTimes(pillars[i], referenceDate, zeroDayCounter);
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
                const mk::CurvePillar& pillar = pillars[i];
                const std::size_t lastNode = multiPass && pass == 0 ? i : count - 1;
                bool cacheValid = false;
                double cachedX = 0.0;
                double cachedF = 0.0;
                std::optional<mk::DiscountCurve<double>> trialCurve;
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
                    mk::detail::CurveTrialUpdater::setNode(*trialCurve, i + 1, trialZero);
                    cachedF = legacy::evaluate(solveQuoteTimes[i], *trialCurve) - pillar.quote;
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
                                             std::string(legacy::kindName(pillar.kind)) + ")");
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
        const mk::DiscountCurve<double> curve(referenceDate, maturityDates, zeroDayCounter, zeros,
                                              space, scheme, tension, switchIndex);
        double worst = 0.0;
        for (std::size_t i = 0; i < count; ++i) {
            const double check = legacy::evaluate(checkQuoteTimes[i], curve) - pillars[i].quote;
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
        throw std::runtime_error("bootstrapDiscountCurve: fixed point did not converge");
    }
    return mk::DiscountCurve<double>(referenceDate, maturityDates, zeroDayCounter, zeros, space,
                                     scheme, tension, switchIndex);
}

/// Frozen copy of the pre-migration forecast-curve solver, used as the
/// old-vs-new bitwise oracle for the forecast bootstrap.
template <typename ParentT>
mk::SpreadCurve<double, ParentT> legacyBootstrapForecastCurve(
    std::shared_ptr<const ParentT> parent, const mk::DiscountCurve<double>* discountCurve,
    const dt::Date& referenceDate, const dt::DayCounter& zeroDayCounter,
    mk::InterpolationScheme scheme, const std::vector<mk::ForecastPillar>& pillars,
    double accuracy = 1e-14, double tension = 0.0) {
    if (!parent) {
        throw std::invalid_argument("bootstrapForecastCurve: null parent");
    }
    if (pillars.empty()) {
        throw std::invalid_argument("bootstrapForecastCurve: no pillars");
    }
    const std::size_t count = pillars.size();
    const auto nodeDate = [](const mk::ForecastPillar& pillar) {
        return legacy::forecastRiskMaturity(pillar);
    };
    std::vector<double> nodeTimes(count);
    for (std::size_t i = 0; i < count; ++i) {
        nodeTimes[i] = dt::yearFraction(referenceDate, nodeDate(pillars[i]), zeroDayCounter);
        if (!(nodeTimes[i] > (i == 0 ? 0.0 : nodeTimes[i - 1]))) {
            throw std::invalid_argument(
                "bootstrapForecastCurve: maturities must be strictly increasing");
        }
    }
    std::vector<legacy::ForecastQuoteTimes> times(count);
    std::vector<double> targets(count);
    for (std::size_t i = 0; i < count; ++i) {
        times[i] = legacy::makeForecastQuoteTimes(pillars[i], referenceDate, zeroDayCounter);
        targets[i] = legacy::forecastTarget(pillars[i]);
    }
    const auto quoteTrial = [&](const auto& trial, std::size_t i) {
        if (discountCurve != nullptr) {
            return legacy::evaluateForecastQuote(times[i], trial, *discountCurve);
        }
        return legacy::evaluateForecastQuote(times[i], trial, *parent);
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
                    const mk::SpreadCurve<double, ParentT> trial(parent, trialTimes, trialSpreads,
                                                                 scheme, tension);
                    cachedF = quoteTrial(trial, i) - targets[i];
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
        std::vector<double> timesFull{0.0};
        std::vector<double> values{0.0};
        timesFull.reserve(count + 1);
        values.reserve(count + 1);
        for (std::size_t i = 0; i < count; ++i) {
            timesFull.push_back(nodeTimes[i]);
            values.push_back(spreads[i]);
        }
        const mk::SpreadCurve<double, ParentT> curve(parent, timesFull, values, scheme, tension);
        double worst = 0.0;
        for (std::size_t i = 0; i < count; ++i) {
            const double check = quoteTrial(curve, i) - targets[i];
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
    std::vector<double> timesFull{0.0};
    std::vector<double> values{0.0};
    timesFull.reserve(count + 1);
    values.reserve(count + 1);
    for (std::size_t i = 0; i < count; ++i) {
        timesFull.push_back(nodeTimes[i]);
        values.push_back(spreads[i]);
    }
    return mk::SpreadCurve<double, ParentT>(std::move(parent), timesFull, values, scheme, tension);
}

bool sameBits(double left, double right) {
    return std::bit_cast<std::uint64_t>(left) == std::bit_cast<std::uint64_t>(right);
}

/// Bitwise old-vs-new gate: node grid, all pillar model quotes and discount
/// samples at the standard tenors.
void checkBootstrapSameBits(const std::vector<mk::CurvePillar>& pillars,
                            const mk::DiscountCurve<double>& legacy,
                            const mk::DiscountCurve<double>& current) {
    EXPECT_EQ(legacy.times().size(), current.times().size());
    EXPECT_EQ(legacy.zeros().size(), current.zeros().size());
    for (std::size_t i = 0; i < legacy.times().size(); ++i) {
        EXPECT_TRUE(sameBits(legacy.times()[i], current.times()[i]));
    }
    for (std::size_t i = 0; i < legacy.zeros().size(); ++i) {
        EXPECT_TRUE(sameBits(legacy.zeros()[i], current.zeros()[i]));
    }
    for (const mk::CurvePillar& pillar : pillars) {
        EXPECT_TRUE(sameBits(mk::impliedQuote(pillar, legacy.referenceDate(), legacy),
                             mk::impliedQuote(pillar, current.referenceDate(), current)));
        EXPECT_TRUE(sameBits(legacy::evaluate(legacy::makeQuoteTimes(pillar, legacy.referenceDate(),
                                                                     legacy.zeroDayCounter()),
                                              legacy),
                             mk::impliedQuote(pillar, legacy.referenceDate(), legacy)));
    }
    for (const double t : {1.0, 5.0, 10.0, 30.0}) {
        EXPECT_TRUE(sameBits(legacy.discount(t), current.discount(t)));
    }
}

struct SchemeCase {
    mk::InterpolationSpace space;
    mk::InterpolationScheme scheme;
    double tension;
};

const std::vector<SchemeCase>& schemeMatrix() {
    static const std::vector<SchemeCase> matrix{
        {mk::InterpolationSpace::LogDiscount, mk::InterpolationScheme::Linear, 0.0},
        {mk::InterpolationSpace::Zero, mk::InterpolationScheme::Linear, 0.0},
        {mk::InterpolationSpace::LogDiscount, mk::InterpolationScheme::Akima, 0.0},
        {mk::InterpolationSpace::LogDiscount, mk::InterpolationScheme::TensionSpline, 8.0},
        {mk::InterpolationSpace::LogDiscount, mk::InterpolationScheme::MonotoneCubic, 0.0},
        {mk::InterpolationSpace::LogDiscount, mk::InterpolationScheme::MixedLinearCubic, 0.0},
    };
    return matrix;
}

std::string schemeLabel(const SchemeCase& item) {
    return "space=" + std::to_string(static_cast<int>(item.space)) +
           " scheme=" + std::to_string(static_cast<int>(item.scheme)) +
           " tension=" + std::to_string(item.tension);
}

mk::DiscountCurve<double> makeTarget(const SchemeCase& item, const std::vector<dt::Date>& dates,
                                     double base, double slope, double curvature) {
    std::vector<double> zeros;
    zeros.reserve(dates.size());
    for (const dt::Date& date : dates) {
        const double t = dt::yearFraction(kReference, date, kZeroDc);
        zeros.push_back(base + slope * t + curvature * t * t);
    }
    return mk::DiscountCurve<double>(kReference, dates, kZeroDc, zeros, item.space, item.scheme,
                                     item.tension);
}

std::vector<mk::CurvePillar> annualOisPillars(const mk::DiscountCurve<double>& target,
                                              const std::vector<dt::Date>& dates,
                                              const dt::Calendar& calendar,
                                              const dt::Period& fixedTenor, int paymentLag) {
    std::vector<mk::CurvePillar> pillars;
    mk::CurvePillar deposit;
    deposit.maturity = dates.front();
    deposit.kind = mk::PillarKind::Deposit;
    deposit.quoteDayCounter = dt::DayCounter(dt::DayCount::Actual360);
    deposit.calendar = calendar;
    deposit.quote = mk::impliedQuote(deposit, kReference, target);
    pillars.push_back(deposit);
    for (std::size_t i = 1; i < dates.size(); ++i) {
        mk::CurvePillar swap;
        swap.maturity = dates[i];
        swap.kind = mk::PillarKind::OisSwap;
        swap.fixedTenor = fixedTenor;
        swap.paymentLag = paymentLag;
        swap.quoteDayCounter = kZeroDc;
        swap.calendar = calendar;
        swap.quote = mk::impliedQuote(swap, kReference, target);
        pillars.push_back(swap);
    }
    return pillars;
}

void checkBootstrapPair(const SchemeCase& item, const std::vector<mk::CurvePillar>& pillars) {
    SCOPED_TRACE(schemeLabel(item));
    const mk::DiscountCurve<double> legacy = legacyBootstrapDiscountCurve(
        kReference, kZeroDc, item.space, item.scheme, pillars, 1e-14, item.tension);
    const mk::DiscountCurve<double> current = mk::bootstrapDiscountCurve(
        kReference, kZeroDc, item.space, item.scheme, pillars, 1e-14, item.tension);
    checkBootstrapSameBits(pillars, legacy, current);
}

/// Bitwise old-vs-new forecast gate: node times and spreads, every pillar's
/// model quote through both the frozen and the migrated evaluator, and
/// discount samples at the standard tenors.
void checkForecastSameBits(const std::vector<mk::ForecastPillar>& pillars,
                           const dt::Date& referenceDate, const dt::DayCounter& zeroDayCounter,
                           const mk::DiscountCurve<double>& discount,
                           const mk::SpreadCurve<double>& legacyCurve,
                           const mk::SpreadCurve<double>& currentCurve) {
    const mk::DiscountCurve<double>& legacyNodes = legacyCurve.spreadNodes();
    const mk::DiscountCurve<double>& currentNodes = currentCurve.spreadNodes();
    EXPECT_EQ(legacyNodes.times().size(), currentNodes.times().size());
    EXPECT_EQ(legacyNodes.zeros().size(), currentNodes.zeros().size());
    for (std::size_t i = 0; i < legacyNodes.times().size(); ++i) {
        EXPECT_TRUE(sameBits(legacyNodes.times()[i], currentNodes.times()[i]));
    }
    for (std::size_t i = 0; i < legacyNodes.zeros().size(); ++i) {
        EXPECT_TRUE(sameBits(legacyNodes.zeros()[i], currentNodes.zeros()[i]));
    }
    for (const mk::ForecastPillar& pillar : pillars) {
        const legacy::ForecastQuoteTimes times =
            legacy::makeForecastQuoteTimes(pillar, referenceDate, zeroDayCounter);
        const double frozen = legacy::evaluateForecastQuote(times, legacyCurve, discount);
        EXPECT_TRUE(sameBits(frozen, mk::impliedForecastQuote(legacyCurve, discount, pillar,
                                                              referenceDate, zeroDayCounter)));
        EXPECT_TRUE(sameBits(frozen, mk::impliedForecastQuote(currentCurve, discount, pillar,
                                                              referenceDate, zeroDayCounter)));
    }
    for (const double t : {1.0, 5.0, 10.0, 30.0}) {
        EXPECT_TRUE(sameBits(legacyCurve.discount(t), currentCurve.discount(t)));
    }
}

double worstForecastReprice(const std::vector<mk::ForecastPillar>& pillars,
                            const mk::SpreadCurve<double>& curve,
                            const mk::DiscountCurve<double>& discount,
                            const dt::Date& referenceDate, const dt::DayCounter& zeroDayCounter) {
    double worst = 0.0;
    for (const mk::ForecastPillar& pillar : pillars) {
        worst = std::max(worst, std::abs(mk::impliedForecastQuote(curve, discount, pillar,
                                                                  referenceDate, zeroDayCounter) -
                                         mk::forecastPillarTarget(pillar)));
    }
    return worst;
}

void checkForecastPair(const SchemeCase& item, const std::vector<mk::ForecastPillar>& pillars,
                       const std::shared_ptr<const mk::DiscountCurve<double>>& parent,
                       const mk::DiscountCurve<double>& discount, const dt::Date& referenceDate,
                       const dt::DayCounter& zeroDayCounter) {
    SCOPED_TRACE(schemeLabel(item));
    const mk::SpreadCurve<double> legacy =
        legacyBootstrapForecastCurve(parent, &discount, referenceDate, zeroDayCounter, item.scheme,
                                     pillars, 1e-14, item.tension);
    const mk::SpreadCurve<double> current =
        mk::bootstrapForecastCurve(parent, &discount, referenceDate, zeroDayCounter, item.scheme,
                                   pillars, 1e-14, item.tension);
    checkForecastSameBits(pillars, referenceDate, zeroDayCounter, discount, legacy, current);
    CHECK_CLOSE("forecast reprice",
                worstForecastReprice(pillars, current, discount, referenceDate, zeroDayCounter),
                0.0, 1e-10);
}

std::shared_ptr<mk::DiscountCurve<double>> syntheticForecastParent(int years, double base) {
    std::vector<dt::Date> dates;
    std::vector<double> zeros;
    for (int year = 1; year <= years; ++year) {
        dates.push_back(kReference.plusYears(year));
        zeros.push_back(base + 0.0005 * static_cast<double>(year));
    }
    return std::make_shared<mk::DiscountCurve<double>>(kReference, dates, kZeroDc, zeros,
                                                       mk::InterpolationSpace::LogDiscount,
                                                       mk::InterpolationScheme::Linear);
}

std::shared_ptr<mk::SpreadCurve<double>>
syntheticForecastTarget(const std::shared_ptr<mk::DiscountCurve<double>>& parent, int years,
                        double spreadBase, double spreadSlope) {
    std::vector<double> times{0.0};
    std::vector<double> spreads{0.0};
    for (int year = 1; year <= years; ++year) {
        times.push_back(dt::yearFraction(kReference, kReference.plusYears(year), kZeroDc));
        spreads.push_back(spreadBase + spreadSlope * static_cast<double>(year));
    }
    return std::make_shared<mk::SpreadCurve<double>>(parent, times, spreads,
                                                     mk::InterpolationScheme::Linear);
}

mk::ForecastPillar syntheticIrsPillar(const dt::Date& maturity) {
    mk::ForecastPillar out;
    out.kind = mk::ForecastPillar::Kind::Irs;
    out.irs.maturity = maturity;
    out.irs.floatTenor = dt::Period(3, dt::TimeUnit::Months);
    out.irs.floatCalendar = dt::Calendar::noHolidays();
    out.irs.floatDayCounter = dt::DayCounter(dt::DayCount::Actual360);
    out.irs.fixedTenor = dt::Period(1, dt::TimeUnit::Years);
    out.irs.fixedCalendar = dt::Calendar::noHolidays();
    out.irs.fixedDayCounter = dt::DayCounter(dt::DayCount::Thirty360BondBasis);
    out.irs.paymentLag = 2;
    return out;
}

mk::ForecastPillar syntheticBasisPillar(const dt::Date& maturity, bool spreadOnParentLeg) {
    mk::ForecastPillar out;
    out.kind = mk::ForecastPillar::Kind::BasisSwap;
    out.basis.maturity = maturity;
    out.basis.spreadOnParentLeg = spreadOnParentLeg;
    out.basis.floatTenor = dt::Period(3, dt::TimeUnit::Months);
    out.basis.quoteDayCounter = dt::DayCounter(dt::DayCount::Actual360);
    out.basis.calendar = dt::Calendar::noHolidays();
    return out;
}

std::vector<mk::ForecastPillar>
syntheticMixedForecastPillars(const mk::SpreadCurve<double>& target,
                              const mk::DiscountCurve<double>& parent) {
    std::vector<mk::ForecastPillar> pillars;
    for (int year : {1, 2}) {
        mk::ForecastPillar out = syntheticBasisPillar(kReference.plusYears(year), year % 2 == 0);
        out.basis.spread = mk::impliedBasisSpread(target, out.basis, kReference, kZeroDc);
        pillars.push_back(out);
    }
    for (int year : {3, 5, 7, 10}) {
        mk::ForecastPillar out = syntheticIrsPillar(kReference.plusYears(year));
        out.irs.quote = mk::impliedIrsRate(target, parent, out.irs, kReference, kZeroDc);
        pillars.push_back(out);
    }
    return pillars;
}

struct FutureStrip {
    std::vector<dt::Date> starts;
    std::vector<dt::Date> ends;
};

FutureStrip immStrip(std::size_t contracts) {
    FutureStrip strip;
    dt::Date start = dt::nextIMMDate(kReference);
    for (std::size_t i = 0; i < contracts; ++i) {
        const dt::Date end = dt::nextIMMDate(start);
        strip.starts.push_back(start);
        strip.ends.push_back(end);
        start = end;
    }
    return strip;
}

std::vector<mk::CurvePillar> futurePillars(const FutureStrip& strip, double convexity,
                                           mk::FutureStyle style, mk::AveragingStyle averaging) {
    std::vector<mk::CurvePillar> pillars;
    mk::CurvePillar deposit;
    deposit.maturity = dt::Period(3, dt::TimeUnit::Months).advance(kReference);
    deposit.kind = mk::PillarKind::Deposit;
    deposit.quoteDayCounter = dt::DayCounter(dt::DayCount::Actual360);
    deposit.calendar = dt::Calendar::noHolidays();
    pillars.push_back(deposit);
    for (std::size_t i = 0; i < strip.starts.size(); ++i) {
        mk::CurvePillar future;
        future.kind = mk::PillarKind::Future;
        future.start = strip.starts[i];
        future.maturity = strip.ends[i];
        future.quoteDayCounter = dt::DayCounter(dt::DayCount::Actual360);
        future.calendar = dt::Calendar::noHolidays();
        future.convexityAdjustment = convexity;
        future.futureStyle = style;
        future.averagingStyle = averaging;
        pillars.push_back(future);
    }
    return pillars;
}

mk::DiscountCurve<double> futureTarget(const FutureStrip& strip) {
    std::vector<dt::Date> targetDates{dt::Period(3, dt::TimeUnit::Months).advance(kReference)};
    targetDates.insert(targetDates.end(), strip.ends.begin(), strip.ends.end());
    std::vector<double> zeros;
    zeros.reserve(targetDates.size());
    for (const dt::Date& date : targetDates) {
        zeros.push_back(0.035 + 0.001 * dt::yearFraction(kReference, date, kZeroDc));
    }
    return mk::DiscountCurve<double>(kReference, targetDates, kZeroDc, zeros,
                                     mk::InterpolationSpace::LogDiscount,
                                     mk::InterpolationScheme::Linear);
}

using DoubleSet = mk::DiscountSet<mk::DiscountCurve<double>>;
using VarSet = mk::DiscountSet<mk::DiscountCurve<stan::math::var>>;
using NestedSet = mk::DiscountSet<mk::DiscountCurve<stan::math::fvar<stan::math::var>>>;
using ForecastDoubleSet = mk::ForecastSet<mk::SpreadCurve<double>, mk::DiscountCurve<double>>;
using InstrumentLadder = mk::Ladder<instruments::Deposit, instruments::Repo>;

static_assert(mk::DiscountCurveSet<DoubleSet, double>);
static_assert(mk::DiscountCurveSet<VarSet, stan::math::var>);
static_assert(mk::DiscountCurveSet<NestedSet, stan::math::fvar<stan::math::var>>);
static_assert(mk::BootstrapInstrument<mk::DiscountInstrument, DoubleSet, double>);
static_assert(mk::BootstrapInstrument<mk::DiscountInstrument, VarSet, stan::math::var>);
static_assert(
    mk::BootstrapInstrument<mk::DiscountInstrument, NestedSet, stan::math::fvar<stan::math::var>>);
static_assert(mk::BootstrapInstrument<instruments::Deposit, DoubleSet, double>);
static_assert(mk::BootstrapInstrument<instruments::Repo, VarSet, stan::math::var>);
static_assert(
    mk::BootstrapInstrument<instruments::Fra, NestedSet, stan::math::fvar<stan::math::var>>);
static_assert(mk::BootstrapInstrument<instruments::Future, VarSet, stan::math::var>);
static_assert(mk::BootstrapInstrument<instruments::Irs, ForecastDoubleSet, double>);
static_assert(mk::BootstrapInstrument<instruments::BasisSwap, ForecastDoubleSet, double>);
static_assert(mk::BootstrapInstrument<mk::detail::ForecastInstrument, ForecastDoubleSet, double>);
static_assert(mk::BootstrapInstrument<InstrumentLadder, DoubleSet, double>);
static_assert(mk::BootstrapInstrument<InstrumentLadder, VarSet, stan::math::var>);
static_assert(
    mk::BootstrapInstrument<InstrumentLadder, NestedSet, stan::math::fvar<stan::math::var>>);

template <typename ScalarT>
double scalarValue(const ScalarT& value) {
    if constexpr (std::is_same_v<ScalarT, double>) {
        return value;
    } else if constexpr (std::is_same_v<ScalarT, stan::math::var>) {
        return value.val();
    } else {
        return value.val().val();
    }
}

template <typename ScalarT>
void checkConceptScalar() {
    const std::vector<double> times{0.0, 1.0};
    const std::vector<ScalarT> zeros{0.0, 0.02};
    const mk::DiscountCurve<ScalarT> curve(times, zeros, mk::InterpolationSpace::LogDiscount,
                                           mk::InterpolationScheme::Linear);
    const mk::DiscountSet<mk::DiscountCurve<ScalarT>> curves{curve};

    mk::CurvePillar pillar;
    pillar.kind = mk::PillarKind::Deposit;
    pillar.maturity = kReference.plusYears(1);
    pillar.quoteDayCounter = kZeroDc;
    pillar.calendar = dt::Calendar::noHolidays();
    pillar.quote = 0.02;
    const mk::DiscountInstrument instrument = mk::toInstrument(pillar, kReference, kZeroDc);

    const ScalarT adapterQuote = instrument.template impliedQuote<ScalarT>(curves);
    const double expected = 1.0 / std::exp(-0.02) - 1.0;
    CHECK_CLOSE("concept adapter quote", scalarValue(adapterQuote), expected, 1e-12);
    EXPECT_TRUE(instrument.date() == kReference.plusYears(1));
    EXPECT_EQ(instrument.target(), 0.02);
}

} // namespace

class InstrumentsVocabularyTest : public StanTapeTest {};
class BootstrapBitwiseTest : public StanTapeTest {};

TEST_F(InstrumentsVocabularyTest, cashflowLayoutAndTrivialCopy) {
    static_assert(std::is_trivially_copyable_v<instruments::Currency>);
    static_assert(std::is_trivially_copyable_v<instruments::Cashflow>);
    static_assert(sizeof(instruments::Currency) == 3);
    static_assert(offsetof(instruments::Cashflow, payDate) <
                  offsetof(instruments::Cashflow, currency));
    static_assert(offsetof(instruments::Cashflow, currency) <
                  offsetof(instruments::Cashflow, amount));
    static_assert(std::is_same_v<decltype(std::declval<instruments::Cashflow>().amount), double>);

    instruments::Cashflow cashflow;
    cashflow.payDate = kReference.plusYears(2);
    cashflow.currency = instruments::Currency{{'E', 'U', 'R'}};
    cashflow.amount = 1234.5;

    const instruments::Cashflow copy = cashflow;
    EXPECT_TRUE(copy.payDate == cashflow.payDate);
    EXPECT_TRUE(copy.currency.code == cashflow.currency.code);
    EXPECT_EQ(copy.amount, cashflow.amount);
    EXPECT_EQ(instruments::currencyCode(copy.currency), "EUR");
}

TEST_F(InstrumentsVocabularyTest, currencyCodeRoundTripShortLong) {
    const instruments::Currency eur = instruments::currencyFromCode("EUR");
    EXPECT_EQ(instruments::currencyCode(eur), "EUR");
    EXPECT_EQ(eur.code[0], 'E');
    EXPECT_EQ(eur.code[1], 'U');
    EXPECT_EQ(eur.code[2], 'R');

    // Short codes pad with spaces; long codes keep the first three characters.
    EXPECT_EQ(instruments::currencyCode(instruments::currencyFromCode("US")), "US ");
    EXPECT_EQ(instruments::currencyCode(instruments::currencyFromCode("USDX")), "USD");
}

TEST_F(InstrumentsVocabularyTest, prioritiesKindNamesFactories) {
    EXPECT_TRUE(instruments::Future::priority() < instruments::Fra::priority());
    EXPECT_TRUE(instruments::Fra::priority() < instruments::Deposit::priority());
    EXPECT_TRUE(instruments::Deposit::priority() < instruments::Repo::priority());
    EXPECT_EQ(instruments::Deposit::kindName(), "Deposit");
    EXPECT_EQ(instruments::Repo::kindName(), "Repo");
    EXPECT_EQ(instruments::Fra::kindName(), "Fra");
    EXPECT_EQ(instruments::Future::kindName(), "Future");

    const instruments::Deposit deposit =
        instruments::makeDeposit(kReference.plusYears(1), 0.02, dt::Calendar::noHolidays(),
                                 dt::BusinessDayConvention::ModifiedFollowing,
                                 dt::DayCounter(dt::DayCount::Actual360), kReference, kZeroDc);
    EXPECT_TRUE(deposit.date() == kReference.plusYears(1));
    EXPECT_EQ(deposit.target(), 0.02);

    const instruments::Future future = instruments::makeFuture(
        kReference.plusMonths(3), kReference.plusMonths(6), 0.041, pricing::FutureStyle::Averaged,
        pricing::AveragingStyle::Compounded, 1e-4, dt::Calendar::noHolidays(),
        dt::DayCounter(dt::DayCount::Actual360), kReference, kZeroDc);
    EXPECT_TRUE(future.date() == kReference.plusMonths(6));
    EXPECT_EQ(future.target(), 0.041);
}

TEST_F(InstrumentsVocabularyTest, conceptQuotesAdapterDateTarget) {
    SCOPED_TRACE("double");
    checkConceptScalar<double>();
    SCOPED_TRACE("var");
    checkConceptScalar<stan::math::var>();
    SCOPED_TRACE("fvar<var>");
    checkConceptScalar<stan::math::fvar<stan::math::var>>();
}

TEST_F(InstrumentsVocabularyTest, variantLadderRepoQuote) {
    const std::vector<double> times{0.0, 1.0};
    const std::vector<double> zeros{0.0, 0.02};
    const mk::DiscountCurve<double> curve(times, zeros, mk::InterpolationSpace::LogDiscount,
                                          mk::InterpolationScheme::Linear);
    const DoubleSet curves{curve};

    instruments::Repo repo;
    repo.maturity = kReference.plusYears(2);
    repo.calendar = dt::Calendar::noHolidays();
    repo.quoteDayCounter = kZeroDc;
    repo.quote = 0.03;
    repo.prepareDiscount(kReference, kZeroDc);
    const InstrumentLadder ladder{InstrumentLadder::Instrument{repo}};
    const double quote = ladder.template impliedQuote<double>(curves);
    CHECK_CLOSE("variant ladder repo quote", quote,
                (1.0 / curve.discount(kZeroDc.yearFraction(kReference, repo.date())) - 1.0) /
                    repo.quoteDayCounter.yearFraction(kReference, repo.date()),
                1e-14);
    EXPECT_TRUE(ladder.date() == repo.date());
    EXPECT_EQ(ladder.target(), 0.03);
    EXPECT_TRUE(std::holds_alternative<instruments::Repo>(ladder.instrument()));
}

TEST_F(InstrumentsVocabularyTest, toInstrumentAdapterMatchesPillar) {
    const std::vector<double> times{0.0, 1.0, 2.0};
    const std::vector<double> zeros{0.0, 0.02, 0.025};
    const mk::DiscountCurve<double> curve(times, zeros, mk::InterpolationSpace::LogDiscount,
                                          mk::InterpolationScheme::Linear);

    const std::vector<mk::PillarKind> kinds{mk::PillarKind::Deposit, mk::PillarKind::Repo,
                                            mk::PillarKind::Fra, mk::PillarKind::Future,
                                            mk::PillarKind::OisSwap};
    for (const mk::PillarKind kind : kinds) {
        SCOPED_TRACE(std::string(legacy::kindName(kind)));
        mk::CurvePillar pillar;
        pillar.kind = kind;
        pillar.maturity = kReference.plusYears(1);
        pillar.start = kReference.plusMonths(3);
        pillar.quote = 0.02;
        pillar.quoteDayCounter = kZeroDc;
        pillar.calendar = dt::Calendar::noHolidays();
        pillar.fixedTenor = dt::Period(6, dt::TimeUnit::Months);
        const mk::DiscountInstrument instrument = mk::toInstrument(pillar, kReference, kZeroDc);
        EXPECT_TRUE(instrument.date() == mk::pillarRiskMaturity(pillar));
        EXPECT_EQ(instrument.target(), pillar.quote);
        EXPECT_TRUE(sameBits(instrument.template impliedQuote<double>(DoubleSet{curve}),
                             mk::impliedQuote(pillar, kReference, curve)));
    }
}

TEST_F(BootstrapBitwiseTest, validationFixtureStrips) {
    const dt::Calendar calendar = dt::Calendar::noHolidays();
    std::vector<dt::Date> nodeDates{dt::Period(6, dt::TimeUnit::Months).advance(kReference)};
    for (int year = 1; year <= 10; ++year) {
        nodeDates.push_back(kReference.plusYears(year));
    }
    for (const SchemeCase& item : schemeMatrix()) {
        const mk::DiscountCurve<double> target =
            makeTarget(item, nodeDates, 0.025, 0.0015, -0.00008);
        checkBootstrapPair(item, annualOisPillars(target, nodeDates, calendar,
                                                  dt::Period(1, dt::TimeUnit::Years), 0));
    }

    std::vector<dt::Date> annualDates;
    for (int year = 1; year <= 6; ++year) {
        annualDates.push_back(kReference.plusYears(year));
    }
    for (const SchemeCase& item : schemeMatrix()) {
        const mk::DiscountCurve<double> target =
            makeTarget(item, annualDates, 0.03, -0.001, 0.0001);
        checkBootstrapPair(item, annualOisPillars(target, annualDates, calendar,
                                                  dt::Period(6, dt::TimeUnit::Months), 2));
    }

    std::vector<dt::Date> stubDates{dt::Period(4, dt::TimeUnit::Months).advance(kReference),
                                    dt::Period(15, dt::TimeUnit::Months).advance(kReference)};
    for (int year = 2; year <= 5; ++year) {
        stubDates.push_back(kReference.plusYears(year));
    }
    for (const SchemeCase& item : schemeMatrix()) {
        const mk::DiscountCurve<double> target = makeTarget(item, stubDates, -0.002, 0.0, 0.0);
        checkBootstrapPair(item, annualOisPillars(target, stubDates, calendar,
                                                  dt::Period(1, dt::TimeUnit::Years), 0));
    }
}

TEST_F(BootstrapBitwiseTest, futuresFixtureStrips) {
    const FutureStrip strip = immStrip(8);
    const mk::DiscountCurve<double> target = futureTarget(strip);
    const double convexity = mk::hullWhiteFuturesAdjustment(0.01, 0.05, 1.0, 0.25, 1.0);

    struct StyleCase {
        mk::FutureStyle style;
        mk::AveragingStyle averaging;
    };
    const std::vector<StyleCase> styles{
        {mk::FutureStyle::Simple, mk::AveragingStyle::Arithmetic},
        {mk::FutureStyle::Compounded, mk::AveragingStyle::Arithmetic},
        {mk::FutureStyle::Averaged, mk::AveragingStyle::Arithmetic},
        {mk::FutureStyle::Averaged, mk::AveragingStyle::Compounded},
    };
    for (const SchemeCase& item : schemeMatrix()) {
        for (const StyleCase& style : styles) {
            SCOPED_TRACE(std::to_string(static_cast<int>(style.style)) + "/" +
                         std::to_string(static_cast<int>(style.averaging)));
            std::vector<mk::CurvePillar> pillars =
                futurePillars(strip, convexity, style.style, style.averaging);
            for (mk::CurvePillar& pillar : pillars) {
                pillar.quote = mk::impliedQuote(pillar, kReference, target);
            }
            const mk::DiscountCurve<double> legacy = legacyBootstrapDiscountCurve(
                kReference, kZeroDc, item.space, item.scheme, pillars, 1e-14, item.tension);
            const mk::DiscountCurve<double> current = mk::bootstrapDiscountCurve(
                kReference, kZeroDc, item.space, item.scheme, pillars, 1e-14, item.tension);
            checkBootstrapSameBits(pillars, legacy, current);
        }
    }
}

TEST_F(BootstrapBitwiseTest, eurSpotAndEcbFixtures) {
    {
        const std::vector<mk::CurvePillar> pillars = paper::oisSpotPillars();
        for (const SchemeCase& item : schemeMatrix()) {
            const mk::DiscountCurve<double> legacy =
                legacyBootstrapDiscountCurve(paper::kReference, paper::kZeroDc, item.space,
                                             item.scheme, pillars, 1e-14, item.tension);
            const mk::DiscountCurve<double> current =
                mk::bootstrapDiscountCurve(paper::kReference, paper::kZeroDc, item.space,
                                           item.scheme, pillars, 1e-14, item.tension);
            checkBootstrapSameBits(pillars, legacy, current);
        }
    }
    for (const SchemeCase& item : schemeMatrix()) {
        const std::vector<mk::CurvePillar> pillars = paper::oisEcbPillars();
        const mk::DiscountCurve<double> legacy =
            legacyBootstrapDiscountCurve(paper::kReference, paper::kZeroDc, item.space, item.scheme,
                                         pillars, 1e-14, item.tension);
        const mk::DiscountCurve<double> current =
            mk::bootstrapDiscountCurve(paper::kReference, paper::kZeroDc, item.space, item.scheme,
                                       pillars, 1e-14, item.tension);
        checkBootstrapSameBits(pillars, legacy, current);
    }
    {
        const std::vector<mk::CurvePillar> pillars = paper::oisEcbPillars();
        const mk::DiscountCurve<double> legacy = legacyBootstrapDiscountCurve(
            paper::kReference, paper::kZeroDc, mk::InterpolationSpace::LogDiscount,
            mk::InterpolationScheme::HymanSpline, pillars);
        const mk::DiscountCurve<double> current = mk::bootstrapDiscountCurve(
            paper::kReference, paper::kZeroDc, mk::InterpolationSpace::LogDiscount,
            mk::InterpolationScheme::HymanSpline, pillars);
        checkBootstrapSameBits(pillars, legacy, current);
    }
}

const std::vector<SchemeCase>& forecastSchemeMatrix() {
    static const std::vector<SchemeCase> matrix{
        {mk::InterpolationSpace::LogDiscount, mk::InterpolationScheme::Linear, 0.0},
        {mk::InterpolationSpace::LogDiscount, mk::InterpolationScheme::Akima, 0.0},
        {mk::InterpolationSpace::LogDiscount, mk::InterpolationScheme::TensionSpline, 8.0},
        {mk::InterpolationSpace::LogDiscount, mk::InterpolationScheme::MonotoneCubic, 0.0},
        {mk::InterpolationSpace::LogDiscount, mk::InterpolationScheme::MixedLinearCubic, 0.0},
    };
    return matrix;
}

/// Akima overshoots on the sparse three-future strip and stalls the whole-grid
/// fixed point in the pre-migration solver too, so the future-style gate runs
/// the schemes the strip can carry; the IRS/basis strips cover Akima.
const std::vector<SchemeCase>& futureSchemeMatrix() {
    static const std::vector<SchemeCase> matrix{
        {mk::InterpolationSpace::LogDiscount, mk::InterpolationScheme::Linear, 0.0},
        {mk::InterpolationSpace::LogDiscount, mk::InterpolationScheme::TensionSpline, 8.0},
        {mk::InterpolationSpace::LogDiscount, mk::InterpolationScheme::MonotoneCubic, 0.0},
        {mk::InterpolationSpace::LogDiscount, mk::InterpolationScheme::MixedLinearCubic, 0.0},
    };
    return matrix;
}

TEST_F(BootstrapBitwiseTest, forecastIrsBasisStrips) {
    const auto parent = syntheticForecastParent(12, 0.032);
    const auto target = syntheticForecastTarget(parent, 12, 0.0008, 0.00008);
    const std::vector<mk::ForecastPillar> pillars = syntheticMixedForecastPillars(*target, *parent);
    for (const SchemeCase& item : forecastSchemeMatrix()) {
        checkForecastPair(item, pillars, parent, *parent, kReference, kZeroDc);
    }

    // Seasoned IRS: the first floating coupon is already fixed, so the first
    // coupon contributes its known rate while later coupons still telescope.
    std::vector<mk::ForecastPillar> seasoned = pillars;
    mk::ForecastPillar& seasonedIrs = seasoned[2];
    seasonedIrs.irs.start = kReference.plusMonths(-3);
    seasonedIrs.irs.firstCouponFixed = true;
    seasonedIrs.irs.firstCouponRate = 0.034;
    seasonedIrs.irs.quote =
        mk::impliedForecastQuote(*target, *parent, seasonedIrs, kReference, kZeroDc);
    for (const SchemeCase& item : forecastSchemeMatrix()) {
        checkForecastPair(item, seasoned, parent, *parent, kReference, kZeroDc);
    }
}

TEST_F(BootstrapBitwiseTest, forecastFutureStyles) {
    const auto parent = syntheticForecastParent(5, 0.03);
    const dt::Date starts[] = {dt::Date(2027, 3, 17), dt::Date(2027, 6, 16), dt::Date(2027, 9, 15)};
    const dt::Date maturities[] = {dt::Date(2027, 6, 16), dt::Date(2027, 9, 15),
                                   dt::Date(2027, 12, 15)};
    std::vector<double> times{0.0};
    std::vector<double> spreads{0.0};
    for (int i = 0; i < 3; ++i) {
        times.push_back(dt::yearFraction(kReference, maturities[i], kZeroDc));
        spreads.push_back(0.0008 + 0.0002 * static_cast<double>(i));
    }
    const auto target = std::make_shared<mk::SpreadCurve<double>>(parent, times, spreads,
                                                                  mk::InterpolationScheme::Linear);
    struct StyleCase {
        mk::FutureStyle style;
        mk::AveragingStyle averaging;
    };
    const std::vector<StyleCase> styles{
        {mk::FutureStyle::Simple, mk::AveragingStyle::Arithmetic},
        {mk::FutureStyle::Compounded, mk::AveragingStyle::Arithmetic},
        {mk::FutureStyle::Averaged, mk::AveragingStyle::Arithmetic},
        {mk::FutureStyle::Averaged, mk::AveragingStyle::Compounded},
    };
    for (const SchemeCase& item : futureSchemeMatrix()) {
        for (const StyleCase& style : styles) {
            SCOPED_TRACE(std::to_string(static_cast<int>(style.style)) + "/" +
                         std::to_string(static_cast<int>(style.averaging)));
            std::vector<mk::ForecastPillar> pillars;
            for (int i = 0; i < 3; ++i) {
                mk::ForecastPillar out;
                out.kind = mk::ForecastPillar::Kind::Future;
                out.start = starts[i];
                out.maturity = maturities[i];
                out.futureStyle = style.style;
                out.averagingStyle = style.averaging;
                out.convexityAdjustment = 1e-5 * static_cast<double>(i + 1);
                out.quoteDayCounter = dt::DayCounter(dt::DayCount::Actual360);
                out.calendar = dt::Calendar::noHolidays();
                out.quote = mk::impliedForecastQuote(*target, *parent, out, kReference, kZeroDc);
                pillars.push_back(out);
            }
            checkForecastPair(item, pillars, parent, *parent, kReference, kZeroDc);
        }
    }
}

TEST_F(BootstrapBitwiseTest, paperEuriborForecastStrips) {
    const mk::DiscountCurve<double> ois = paper::buildOisEcb();
    const auto parent = std::make_shared<mk::DiscountCurve<double>>(ois);
    const std::vector<std::vector<mk::ForecastPillar>> strips{
        paper::euribor3mPillars(), paper::euribor1mPillars(), paper::euribor6mPillars(),
        paper::euribor12mPillars(), paper::euribor3mSyntheticPillars()};
    for (const std::vector<mk::ForecastPillar>& pillars : strips) {
        for (const SchemeCase& item : forecastSchemeMatrix()) {
            checkForecastPair(item, pillars, parent, ois, paper::kReference, paper::kZeroDc);
        }
    }
}

TEST_F(BootstrapBitwiseTest, paperEuriborSyntheticAndDirectTurn) {
    const mk::DiscountCurve<double> ois = paper::buildOisEcb(mk::InterpolationScheme::HymanSpline);
    const auto parent = std::make_shared<mk::DiscountCurve<double>>(ois);
    const std::vector<mk::ForecastPillar> pillars = paper::euribor3mSyntheticPillars();
    const mk::SpreadCurve<double> smooth = mk::bootstrapForecastCurve(
        parent, &ois, paper::kReference, paper::kZeroDc, mk::InterpolationScheme::Linear, pillars);
    const dt::Date postTurnEnd(2014, 3, 13);
    const std::vector<mk::ForecastPillar> direct = paper::directTurnPillars(
        pillars, smooth, paper::kTurnThreeMonthAmplitude,
        {paper::exactFraPillar(paper::kTurnEnd, postTurnEnd,
                               paper::simpleForwardOver(smooth, paper::kTurnEnd, postTurnEnd))});
    for (const SchemeCase& item : forecastSchemeMatrix()) {
        checkForecastPair(item, direct, parent, ois, paper::kReference, paper::kZeroDc);
    }
}
