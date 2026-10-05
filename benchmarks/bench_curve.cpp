#include "quantape/log/Log.h"
#include "quantape/markets/Curves/CurveBuilder.h"
#include "quantape/markets/Curves/CurveOnGrid.h"
#include "quantape/markets/Curves/CurveRisk.h"
#include "quantape/markets/Curves/DiscountCurve.h"
#include "quantape/markets/Curves/SpreadCurve.h"
#include "quantape/markets/Curves/TurnOverlay.h"
#include "quantape/util/Check.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <memory>
#include <random>
#include <string>
#include <utility>
#include <vector>

using namespace quantape;

namespace {

using markets::DiscountCurve;

volatile double g_sink = 0.0;

struct Timing {
    double p50Ns = 0.0;
    double p99Ns = 0.0;
};

/// Time one repetition per sample and reduce to p50/p99 of the total duration.
template <typename F>
Timing repeatTiming(int repetitions, F&& runOnce) {
    std::vector<double> samples;
    samples.reserve(static_cast<std::size_t>(repetitions));
    for (int r = 0; r < repetitions; ++r) {
        const auto start = std::chrono::steady_clock::now();
        runOnce();
        const auto stop = std::chrono::steady_clock::now();
        samples.push_back(std::chrono::duration<double, std::nano>(stop - start).count());
    }
    std::sort(samples.begin(), samples.end());
    const auto quantile = [&](double q) {
        const std::size_t index =
            static_cast<std::size_t>(std::ceil(q * static_cast<double>(samples.size()))) - 1;
        return samples[std::min(index, samples.size() - 1)];
    };
    return {quantile(0.5), quantile(0.99)};
}

DiscountCurve<double> makeCurve(markets::InterpolationSpace space,
                                markets::InterpolationScheme scheme, double tension = 0.0) {
    std::vector<double> times;
    std::vector<double> zeros;
    for (int i = 0; i <= 20; ++i) {
        const double t = 1.5 * static_cast<double>(i);
        times.push_back(t);
        zeros.push_back(0.04 + 0.001 * t);
    }
    return DiscountCurve<double>(std::move(times), std::move(zeros), space, scheme, tension);
}

void benchCurveEval() {
    constexpr std::size_t kIterations = 200'000;
    constexpr int kRepetitions = 9;
    std::mt19937 rng(42);
    std::uniform_real_distribution<double> dist(1e-6, 29.0);
    std::vector<double> queryTimes(kIterations);
    for (double& t : queryTimes) {
        t = dist(rng);
    }

    const struct {
        markets::InterpolationSpace space;
        markets::InterpolationScheme scheme;
        double tension;
    } cases[] = {
        {markets::InterpolationSpace::LogDiscount, markets::InterpolationScheme::Linear, 0.0},
        {markets::InterpolationSpace::LogDiscount, markets::InterpolationScheme::Akima, 0.0},
        {markets::InterpolationSpace::LogDiscount, markets::InterpolationScheme::TensionSpline,
         8.0},
        {markets::InterpolationSpace::LogDiscount, markets::InterpolationScheme::HymanSpline, 0.0},
        {markets::InterpolationSpace::LogDiscount, markets::InterpolationScheme::MonotoneCubic,
         0.0},
        {markets::InterpolationSpace::LogDiscount, markets::InterpolationScheme::MixedLinearCubic,
         0.0},
        {markets::InterpolationSpace::Zero, markets::InterpolationScheme::Linear, 0.0},
        {markets::InterpolationSpace::Zero, markets::InterpolationScheme::Akima, 0.0},
        {markets::InterpolationSpace::Zero, markets::InterpolationScheme::TensionSpline, 8.0},
        {markets::InterpolationSpace::Zero, markets::InterpolationScheme::HymanSpline, 0.0},
        {markets::InterpolationSpace::Zero, markets::InterpolationScheme::MonotoneCubic, 0.0},
        {markets::InterpolationSpace::Zero, markets::InterpolationScheme::MixedLinearCubic, 0.0},
    };

    for (const auto& item : cases) {
        const DiscountCurve<double> curve = makeCurve(item.space, item.scheme, item.tension);
        const std::string label = std::string(markets::interpolationSpaceName(item.space)) + " x " +
                                  std::string(markets::interpolationSchemeName(item.scheme));

        const Timing discount = repeatTiming(kRepetitions, [&] {
            double sum = 0.0;
            for (double t : queryTimes) {
                sum += curve.discount(t);
            }
            g_sink = sum;
        });
        const Timing zero = repeatTiming(kRepetitions, [&] {
            double sum = 0.0;
            for (double t : queryTimes) {
                sum += curve.zero(t);
            }
            g_sink = sum;
        });
        const Timing forward = repeatTiming(kRepetitions, [&] {
            double sum = 0.0;
            for (double t : queryTimes) {
                sum += curve.forward(t, t + 0.25);
            }
            g_sink = sum;
        });
        const double calls = static_cast<double>(kIterations);
        QTA_LOG_INFO("bench", "{}  discount p50 {} p99 {} ns/call", label,
                     util::num(discount.p50Ns / calls, 2), util::num(discount.p99Ns / calls, 2));
        QTA_LOG_INFO("bench", "{}  zero     p50 {} p99 {} ns/call", label,
                     util::num(zero.p50Ns / calls, 2), util::num(zero.p99Ns / calls, 2));
        QTA_LOG_INFO("bench", "{}  forward  p50 {} p99 {} ns/call", label,
                     util::num(forward.p50Ns / calls, 2), util::num(forward.p99Ns / calls, 2));
    }
}

void benchMaterialize() {
    constexpr std::size_t kPoints = 1 << 20;
    constexpr int kRepetitions = 9;
    std::vector<double> times(kPoints);
    for (std::size_t k = 0; k < kPoints; ++k) {
        times[k] = 30.0 * static_cast<double>(k) / static_cast<double>(kPoints - 1);
    }
    const struct {
        markets::InterpolationSpace space;
        markets::InterpolationScheme scheme;
    } cases[] = {
        {markets::InterpolationSpace::LogDiscount, markets::InterpolationScheme::Linear},
        {markets::InterpolationSpace::Zero, markets::InterpolationScheme::Akima},
    };
    for (const auto& item : cases) {
        const DiscountCurve<double> curve = makeCurve(item.space, item.scheme);
        const Timing timing = repeatTiming(kRepetitions, [&] {
            const auto grid = markets::materialize(curve, times);
            g_sink = static_cast<double>(grid.discount.back()) +
                     static_cast<double>(grid.forward.back());
        });
        const std::string label = std::string(markets::interpolationSpaceName(item.space)) + " x " +
                                  std::string(markets::interpolationSchemeName(item.scheme));
        QTA_LOG_INFO("bench", "materialize {} p50 {} p99 {} ns/point ({} points)", label,
                     util::num(timing.p50Ns / static_cast<double>(kPoints), 2),
                     util::num(timing.p99Ns / static_cast<double>(kPoints), 2), kPoints);
    }
}

struct DiscountFixture {
    datetime::Date reference;
    datetime::DayCounter zeroDayCounter;
    datetime::Calendar calendar;
    std::vector<markets::CurvePillar> pillars;
};

DiscountFixture makeDiscountFixture(int pillarCount) {
    DiscountFixture fixture;
    fixture.reference = datetime::Date(2026, 9, 29);
    fixture.zeroDayCounter = datetime::DayCounter(datetime::DayCount::Actual365Fixed);
    fixture.calendar = datetime::Calendar::noHolidays();

    std::vector<datetime::Date> pillarDates;
    pillarDates.push_back(
        datetime::Period(6, datetime::TimeUnit::Months).advance(fixture.reference));
    for (int years = 1; years <= pillarCount - 1; ++years) {
        pillarDates.push_back(fixture.reference.plusYears(years));
    }
    std::vector<double> targetZeros;
    targetZeros.reserve(pillarDates.size());
    for (const auto& date : pillarDates) {
        const double t = datetime::yearFraction(fixture.reference, date, fixture.zeroDayCounter);
        targetZeros.push_back(0.04 - 0.002 * std::exp(-0.5 * t));
    }
    const DiscountCurve<double> target(fixture.reference, pillarDates, fixture.zeroDayCounter,
                                       targetZeros, markets::InterpolationSpace::LogDiscount,
                                       markets::InterpolationScheme::Linear);
    markets::CurvePillar deposit;
    deposit.maturity = pillarDates.front();
    deposit.kind = markets::PillarKind::Deposit;
    deposit.quoteDayCounter = datetime::DayCounter(datetime::DayCount::Actual360);
    deposit.calendar = fixture.calendar;
    deposit.quote = markets::impliedQuote(deposit, fixture.reference, target);
    fixture.pillars.push_back(deposit);
    for (std::size_t i = 1; i < pillarDates.size(); ++i) {
        markets::CurvePillar swap;
        swap.maturity = pillarDates[i];
        swap.kind = markets::PillarKind::OisSwap;
        swap.quoteDayCounter = fixture.zeroDayCounter;
        swap.calendar = fixture.calendar;
        swap.quote = markets::impliedQuote(swap, fixture.reference, target);
        fixture.pillars.push_back(swap);
    }
    return fixture;
}

void benchBootstrapDiscount() {
    const struct {
        int pillars;
        int repetitions;
    } cases[] = {{10, 300}, {21, 100}, {42, 30}};
    for (const auto& item : cases) {
        const DiscountFixture fixture = makeDiscountFixture(item.pillars);
        const Timing timing = repeatTiming(item.repetitions, [&] {
            const markets::DiscountCurve<double> curve = markets::bootstrapDiscountCurve(
                fixture.reference, fixture.zeroDayCounter, markets::InterpolationSpace::LogDiscount,
                markets::InterpolationScheme::Linear, fixture.pillars);
            g_sink = curve.zeros().back();
        });
        QTA_LOG_INFO("bench", "bootstrapDiscountCurve {} pillars p50 {} p99 {} us/curve ({} reps)",
                     item.pillars, util::num(timing.p50Ns / 1000.0, 1),
                     util::num(timing.p99Ns / 1000.0, 1), item.repetitions);
    }
}

struct ForecastFixture {
    std::shared_ptr<const DiscountCurve<double>> parent;
    std::vector<markets::ForecastPillar> pillars;
};

ForecastFixture makeForecastFixture(int pillarCount) {
    ForecastFixture fixture;
    const datetime::Date reference(2026, 9, 29);
    const datetime::DayCounter zeroDayCounter(datetime::DayCount::Actual365Fixed);
    const datetime::Calendar calendar = datetime::Calendar::noHolidays();

    std::vector<datetime::Date> pillarDates;
    pillarDates.push_back(datetime::Period(6, datetime::TimeUnit::Months).advance(reference));
    for (int years = 1; years <= pillarCount - 1; ++years) {
        pillarDates.push_back(reference.plusYears(years));
    }
    std::vector<double> parentZeros;
    parentZeros.reserve(pillarDates.size());
    for (const auto& date : pillarDates) {
        const double t = datetime::yearFraction(reference, date, zeroDayCounter);
        parentZeros.push_back(0.04 - 0.002 * std::exp(-0.5 * t));
    }
    auto parent = std::make_shared<DiscountCurve<double>>(
        reference, pillarDates, zeroDayCounter, parentZeros,
        markets::InterpolationSpace::LogDiscount, markets::InterpolationScheme::Linear);

    std::vector<double> times{0.0};
    std::vector<double> spreads{0.0};
    std::vector<double> targetSpread;
    for (std::size_t i = 0; i < pillarDates.size(); ++i) {
        const double t = datetime::yearFraction(reference, pillarDates[i], zeroDayCounter);
        times.push_back(t);
        targetSpread.push_back(0.0005 + 0.00002 * t);
        spreads.push_back(targetSpread.back());
    }
    const markets::SpreadCurve<double> target(parent, times, spreads,
                                              markets::InterpolationScheme::Linear);

    fixture.parent = parent;
    for (std::size_t i = 1; i < pillarDates.size(); ++i) {
        markets::ForecastPillar out;
        out.kind = markets::ForecastPillar::Kind::Irs;
        out.irs.maturity = pillarDates[i];
        out.irs.floatTenor = datetime::Period(3, datetime::TimeUnit::Months);
        out.irs.floatCalendar = calendar;
        out.irs.floatDayCounter = datetime::DayCounter(datetime::DayCount::Actual360);
        out.irs.fixedTenor = datetime::Period(1, datetime::TimeUnit::Years);
        out.irs.fixedCalendar = calendar;
        out.irs.fixedDayCounter = datetime::DayCounter(datetime::DayCount::Thirty360BondBasis);
        out.irs.quote =
            markets::impliedIrsRate(target, *parent, out.irs, reference, zeroDayCounter);
        fixture.pillars.push_back(out);
    }
    return fixture;
}

void benchBootstrapForecast() {
    const struct {
        int pillars;
        int repetitions;
    } cases[] = {{10, 60}, {21, 20}, {42, 8}};
    const datetime::Date reference(2026, 9, 29);
    const datetime::DayCounter zeroDayCounter(datetime::DayCount::Actual365Fixed);
    for (const auto& item : cases) {
        const ForecastFixture fixture = makeForecastFixture(item.pillars);
        const Timing timing = repeatTiming(item.repetitions, [&] {
            const markets::SpreadCurve<double> curve = markets::bootstrapForecastCurve(
                fixture.parent, nullptr, reference, zeroDayCounter,
                markets::InterpolationScheme::Linear, fixture.pillars);
            g_sink = curve.spreadNodes().zeros().back();
        });
        QTA_LOG_INFO("bench", "bootstrapForecastCurve {} pillars p50 {} p99 {} us/curve ({} reps)",
                     item.pillars, util::num(timing.p50Ns / 1000.0, 1),
                     util::num(timing.p99Ns / 1000.0, 1), item.repetitions);
    }
}

void benchTurnOverlay() {
    constexpr std::size_t kIterations = 200'000;
    constexpr int kRepetitions = 9;
    std::mt19937 rng(7);
    std::uniform_real_distribution<double> dist(1e-6, 29.0);
    std::vector<double> queryTimes(kIterations);
    for (double& t : queryTimes) {
        t = dist(rng);
    }
    auto base = std::make_shared<const DiscountCurve<double>>(
        makeCurve(markets::InterpolationSpace::LogDiscount, markets::InterpolationScheme::Linear));
    const std::vector<markets::TurnOverlay<double>::Turn> turns{
        {0.5, 0.0002}, {1.5, 0.0003}, {2.5, 0.00025}, {3.5, 0.00015}};
    const std::vector<markets::TurnOverlay<double>::Bump> bumps{{0.9, 1.1, 0.0004},
                                                                {1.9, 2.2, 0.0003}};
    const markets::TurnOverlay<double> overlay(base, turns, bumps);

    const Timing discount = repeatTiming(kRepetitions, [&] {
        double sum = 0.0;
        for (double t : queryTimes) {
            sum += overlay.discount(t);
        }
        g_sink = sum;
    });
    const Timing zero = repeatTiming(kRepetitions, [&] {
        double sum = 0.0;
        for (double t : queryTimes) {
            sum += overlay.zero(t);
        }
        g_sink = sum;
    });
    const Timing forward = repeatTiming(kRepetitions, [&] {
        double sum = 0.0;
        for (double t : queryTimes) {
            sum += overlay.forward(t, t + 0.25);
        }
        g_sink = sum;
    });
    const double calls = static_cast<double>(kIterations);
    QTA_LOG_INFO("bench", "TurnOverlay discount p50 {} p99 {} ns/call",
                 util::num(discount.p50Ns / calls, 2), util::num(discount.p99Ns / calls, 2));
    QTA_LOG_INFO("bench", "TurnOverlay zero     p50 {} p99 {} ns/call",
                 util::num(zero.p50Ns / calls, 2), util::num(zero.p99Ns / calls, 2));
    QTA_LOG_INFO("bench", "TurnOverlay forward  p50 {} p99 {} ns/call",
                 util::num(forward.p50Ns / calls, 2), util::num(forward.p99Ns / calls, 2));
}

void benchQuoteRisk() {
    const DiscountFixture fixture = makeDiscountFixture(21);
    const markets::DiscountCurve<double> curve = markets::bootstrapDiscountCurve(
        fixture.reference, fixture.zeroDayCounter, markets::InterpolationSpace::LogDiscount,
        markets::InterpolationScheme::Linear, fixture.pillars);
    const double t = 10.0;
    std::vector<double> weights;
    curve.zeroNodeWeights(t, weights);
    const double df = curve.discount(t);
    std::vector<double> dVdZeros(weights.size());
    for (std::size_t i = 0; i < weights.size(); ++i) {
        dVdZeros[i] = -t * df * weights[i];
    }
    constexpr int kRepetitions = 200;
    const Timing timing = repeatTiming(kRepetitions, [&] {
        const markets::QuoteRisk risk =
            markets::transformQuoteRisk(curve, fixture.pillars, fixture.reference, dVdZeros);
        g_sink = risk.quoteDeltas.back();
    });
    QTA_LOG_INFO("bench", "quote risk 21 pillars p50 {} p99 {} us/transform",
                 util::num(timing.p50Ns / 1000.0, 1), util::num(timing.p99Ns / 1000.0, 1));
}

} // namespace

int main() {
    benchCurveEval();
    benchMaterialize();
    benchBootstrapDiscount();
    benchBootstrapForecast();
    benchTurnOverlay();
    benchQuoteRisk();
    QTA_LOG_INFO("bench", "done");
    return 0;
}
