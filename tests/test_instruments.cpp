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
#include "quantape/log/Log.h"
#include "quantape/markets/Curves/BootstrapInstrument.h"
#include "quantape/markets/Curves/CurveBuilder.h"
#include "quantape/markets/Curves/HullWhiteConvexity.h"
#include "quantape/math/Solvers/BrentSolver.h"
#include "quantape/pricing/Ir.h"
#include "quantape/util/Check.h"

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

#include "paper_eur_curves_fixture.h"

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

bool sameBits(double left, double right) {
    return std::bit_cast<std::uint64_t>(left) == std::bit_cast<std::uint64_t>(right);
}

/// Bitwise old-vs-new gate: node grid, all pillar model quotes and discount
/// samples at the standard tenors.
void checkBootstrapSameBits(const std::vector<mk::CurvePillar>& pillars,
                            const mk::DiscountCurve<double>& legacy,
                            const mk::DiscountCurve<double>& current) {
    CHECK(legacy.times().size() == current.times().size());
    CHECK(legacy.zeros().size() == current.zeros().size());
    for (std::size_t i = 0; i < legacy.times().size(); ++i) {
        CHECK(sameBits(legacy.times()[i], current.times()[i]));
    }
    for (std::size_t i = 0; i < legacy.zeros().size(); ++i) {
        CHECK(sameBits(legacy.zeros()[i], current.zeros()[i]));
    }
    for (const mk::CurvePillar& pillar : pillars) {
        CHECK(sameBits(mk::impliedQuote(pillar, legacy.referenceDate(), legacy),
                       mk::impliedQuote(pillar, current.referenceDate(), current)));
        CHECK(sameBits(legacy::evaluate(legacy::makeQuoteTimes(pillar, legacy.referenceDate(),
                                                               legacy.zeroDayCounter()),
                                        legacy),
                       mk::impliedQuote(pillar, legacy.referenceDate(), legacy)));
    }
    for (const double t : {1.0, 5.0, 10.0, 30.0}) {
        CHECK(sameBits(legacy.discount(t), current.discount(t)));
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
    const mk::DiscountCurve<double> legacy = legacyBootstrapDiscountCurve(
        kReference, kZeroDc, item.space, item.scheme, pillars, 1e-14, item.tension);
    const mk::DiscountCurve<double> current = mk::bootstrapDiscountCurve(
        kReference, kZeroDc, item.space, item.scheme, pillars, 1e-14, item.tension);
    checkBootstrapSameBits(pillars, legacy, current);
}

void testValidationFixtureBits() {
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

void testFuturesFixtureBits() {
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

void testEurFixtureBits() {
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

using DoubleSet = mk::DiscountSet<mk::DiscountCurve<double>>;
using VarSet = mk::DiscountSet<mk::DiscountCurve<stan::math::var>>;
using NestedSet = mk::DiscountSet<mk::DiscountCurve<stan::math::fvar<stan::math::var>>>;
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
    util::checkClose("concept adapter quote", scalarValue(adapterQuote), expected, 1e-12);
    CHECK(instrument.date() == kReference.plusYears(1));
    CHECK(instrument.target() == 0.02);
}

void testConceptQuotes() {
    checkConceptScalar<double>();
    checkConceptScalar<stan::math::var>();
    checkConceptScalar<stan::math::fvar<stan::math::var>>();
}

void testVariantLadder() {
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
    util::checkClose("variant ladder repo quote", quote,
                     (1.0 / curve.discount(kZeroDc.yearFraction(kReference, repo.date())) - 1.0) /
                         repo.quoteDayCounter.yearFraction(kReference, repo.date()),
                     1e-14);
    CHECK(ladder.date() == repo.date());
    CHECK(ladder.target() == 0.03);
    CHECK(std::holds_alternative<instruments::Repo>(ladder.instrument()));
}

void testToInstrumentAdapter() {
    const std::vector<double> times{0.0, 1.0, 2.0};
    const std::vector<double> zeros{0.0, 0.02, 0.025};
    const mk::DiscountCurve<double> curve(times, zeros, mk::InterpolationSpace::LogDiscount,
                                          mk::InterpolationScheme::Linear);

    const std::vector<mk::PillarKind> kinds{mk::PillarKind::Deposit, mk::PillarKind::Repo,
                                            mk::PillarKind::Fra, mk::PillarKind::Future,
                                            mk::PillarKind::OisSwap};
    for (const mk::PillarKind kind : kinds) {
        mk::CurvePillar pillar;
        pillar.kind = kind;
        pillar.maturity = kReference.plusYears(1);
        pillar.start = kReference.plusMonths(3);
        pillar.quote = 0.02;
        pillar.quoteDayCounter = kZeroDc;
        pillar.calendar = dt::Calendar::noHolidays();
        pillar.fixedTenor = dt::Period(6, dt::TimeUnit::Months);
        const mk::DiscountInstrument instrument = mk::toInstrument(pillar, kReference, kZeroDc);
        CHECK(instrument.date() == mk::pillarRiskMaturity(pillar));
        CHECK(instrument.target() == pillar.quote);
        CHECK(sameBits(instrument.template impliedQuote<double>(DoubleSet{curve}),
                       mk::impliedQuote(pillar, kReference, curve)));
    }
}

void testCurrencyCode() {
    const instruments::Currency eur = instruments::currencyFromCode("EUR");
    CHECK(instruments::currencyCode(eur) == "EUR");
    CHECK(eur.code[0] == 'E');
    CHECK(eur.code[1] == 'U');
    CHECK(eur.code[2] == 'R');

    // Short codes pad with spaces; long codes keep the first three characters.
    CHECK(instruments::currencyCode(instruments::currencyFromCode("US")) == "US ");
    CHECK(instruments::currencyCode(instruments::currencyFromCode("USDX")) == "USD");
}

void testInstrumentMetadata() {
    CHECK(instruments::Future::priority() < instruments::Fra::priority());
    CHECK(instruments::Fra::priority() < instruments::Deposit::priority());
    CHECK(instruments::Deposit::priority() < instruments::Repo::priority());
    CHECK(instruments::Deposit::kindName() == "Deposit");
    CHECK(instruments::Repo::kindName() == "Repo");
    CHECK(instruments::Fra::kindName() == "Fra");
    CHECK(instruments::Future::kindName() == "Future");

    const instruments::Deposit deposit =
        instruments::makeDeposit(kReference.plusYears(1), 0.02, dt::Calendar::noHolidays(),
                                 dt::BusinessDayConvention::ModifiedFollowing,
                                 dt::DayCounter(dt::DayCount::Actual360), kReference, kZeroDc);
    CHECK(deposit.date() == kReference.plusYears(1));
    CHECK(deposit.target() == 0.02);

    const instruments::Future future = instruments::makeFuture(
        kReference.plusMonths(3), kReference.plusMonths(6), 0.041, pricing::FutureStyle::Averaged,
        pricing::AveragingStyle::Compounded, 1e-4, dt::Calendar::noHolidays(),
        dt::DayCounter(dt::DayCount::Actual360), kReference, kZeroDc);
    CHECK(future.date() == kReference.plusMonths(6));
    CHECK(future.target() == 0.041);
}

void testCashflowLayout() {
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
    CHECK(copy.payDate == cashflow.payDate);
    CHECK(copy.currency.code == cashflow.currency.code);
    CHECK(copy.amount == cashflow.amount);
    CHECK(instruments::currencyCode(copy.currency) == "EUR");
}

} // namespace

int main() {
    testCashflowLayout();
    testCurrencyCode();
    testInstrumentMetadata();
    testConceptQuotes();
    testVariantLadder();
    testToInstrumentAdapter();
    testValidationFixtureBits();
    testFuturesFixtureBits();
    testEurFixtureBits();
    QTA_LOG_INFO("test", "test_instruments: ok");
    return 0;
}
