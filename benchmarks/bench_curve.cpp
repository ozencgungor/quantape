#include "quantape/log/Log.h"
#include "quantape/markets/Curves/CurveBuilder.h"
#include "quantape/markets/Curves/CurveOnGrid.h"
#include "quantape/markets/Curves/CurveRisk.h"
#include "quantape/markets/Curves/DiscountCurve.h"
#include "quantape/util/Check.h"

#include <chrono>
#include <cmath>
#include <cstddef>
#include <random>
#include <string>
#include <vector>

using namespace quantape;

namespace {

using markets::DiscountCurve;

volatile double g_sink = 0.0;

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

template <typename F>
double nsPerCall(std::size_t iterations, F&& fn) {
    const auto start = std::chrono::steady_clock::now();
    fn();
    const auto stop = std::chrono::steady_clock::now();
    return std::chrono::duration<double, std::nano>(stop - start).count() /
           static_cast<double>(iterations);
}

void benchDiscount() {
    constexpr std::size_t kIterations = 2'000'000;
    std::mt19937 rng(42);
    std::uniform_real_distribution<double> dist(1e-6, 30.0);
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
        {markets::InterpolationSpace::Zero, markets::InterpolationScheme::Linear, 0.0},
        {markets::InterpolationSpace::Zero, markets::InterpolationScheme::Akima, 0.0},
        {markets::InterpolationSpace::LogDiscount, markets::InterpolationScheme::TensionSpline,
         8.0},
        {markets::InterpolationSpace::Zero, markets::InterpolationScheme::MonotoneCubic, 0.0},
        {markets::InterpolationSpace::Zero, markets::InterpolationScheme::MixedLinearCubic, 0.0},
    };

    for (const auto& item : cases) {
        const DiscountCurve<double> curve = makeCurve(item.space, item.scheme, item.tension);
        double sum = 0.0;
        const double ns = nsPerCall(kIterations, [&] {
            for (double t : queryTimes) {
                sum += curve.discount(t);
            }
        });
        g_sink = sum;
        QTA_LOG_INFO("bench", "{} x {}  discount {} ns/call",
                     markets::interpolationSpaceName(item.space),
                     markets::interpolationSchemeName(item.scheme), util::num(ns, 4));
    }
}

void benchBootstrap() {
    const datetime::Date reference(2026, 9, 29);
    const datetime::DayCounter zeroDayCounter(datetime::DayCount::Actual365Fixed);
    const datetime::Calendar calendar = datetime::Calendar::noHolidays();

    std::vector<datetime::Date> pillarDates;
    pillarDates.push_back(datetime::Period(6, datetime::TimeUnit::Months).advance(reference));
    for (int years = 1; years <= 20; ++years) {
        pillarDates.push_back(reference.plusYears(years));
    }
    std::vector<double> targetZeros;
    targetZeros.reserve(pillarDates.size());
    for (const auto& date : pillarDates) {
        const double t = datetime::yearFraction(reference, date, zeroDayCounter);
        targetZeros.push_back(0.04 - 0.002 * std::exp(-0.5 * t));
    }
    const markets::DiscountCurve<double> target(reference, pillarDates, zeroDayCounter, targetZeros,
                                                markets::InterpolationSpace::LogDiscount,
                                                markets::InterpolationScheme::Linear);
    std::vector<markets::CurvePillar> pillars;
    markets::CurvePillar deposit;
    deposit.maturity = pillarDates.front();
    deposit.kind = markets::PillarKind::Deposit;
    deposit.quoteDayCounter = datetime::DayCounter(datetime::DayCount::Actual360);
    deposit.calendar = calendar;
    deposit.quote = markets::impliedQuote(deposit, reference, target);
    pillars.push_back(deposit);
    for (std::size_t i = 1; i < pillarDates.size(); ++i) {
        markets::CurvePillar swap;
        swap.maturity = pillarDates[i];
        swap.kind = markets::PillarKind::OisSwap;
        swap.quoteDayCounter = zeroDayCounter;
        swap.calendar = calendar;
        swap.quote = markets::impliedQuote(swap, reference, target);
        pillars.push_back(swap);
    }

    constexpr int kRepetitions = 200;
    const auto start = std::chrono::steady_clock::now();
    double sink = 0.0;
    markets::DiscountCurve<double> curve = markets::bootstrapDiscountCurve(
        reference, zeroDayCounter, markets::InterpolationSpace::LogDiscount,
        markets::InterpolationScheme::Linear, pillars);
    for (int i = 0; i < kRepetitions; ++i) {
        curve = markets::bootstrapDiscountCurve(reference, zeroDayCounter,
                                                markets::InterpolationSpace::LogDiscount,
                                                markets::InterpolationScheme::Linear, pillars);
        sink += curve.zeros().back();
    }
    const auto stop = std::chrono::steady_clock::now();
    g_sink = sink;
    const double microseconds =
        std::chrono::duration<double, std::micro>(stop - start).count() / kRepetitions;
    QTA_LOG_INFO("bench", "bootstrap 21 pillars  {} us/curve", util::num(microseconds, 3));

    // Quote-space risk: assemble the 21x21 instrument Jacobian and solve.
    const double t = 10.0;
    std::vector<double> weights;
    curve.zeroNodeWeights(t, weights);
    const double df = curve.discount(t);
    std::vector<double> dVdZeros(weights.size());
    for (std::size_t i = 0; i < weights.size(); ++i) {
        dVdZeros[i] = -t * df * weights[i];
    }
    constexpr int kRiskRepetitions = 2000;
    double riskSink = 0.0;
    const auto riskStart = std::chrono::steady_clock::now();
    for (int i = 0; i < kRiskRepetitions; ++i) {
        const markets::QuoteRisk risk =
            markets::transformQuoteRisk(curve, pillars, reference, dVdZeros);
        riskSink += risk.quoteDeltas.back();
    }
    const auto riskStop = std::chrono::steady_clock::now();
    g_sink = riskSink;
    const double riskMicros =
        std::chrono::duration<double, std::micro>(riskStop - riskStart).count() / kRiskRepetitions;
    QTA_LOG_INFO("bench", "quote risk 21 pillars  {} us/transform", util::num(riskMicros, 3));
}

void benchMaterialize() {
    const DiscountCurve<double> curve =
        makeCurve(markets::InterpolationSpace::LogDiscount, markets::InterpolationScheme::Linear);
    constexpr std::size_t kPoints = 1 << 20;
    std::vector<double> times(kPoints);
    for (std::size_t k = 0; k < kPoints; ++k) {
        times[k] = 30.0 * static_cast<double>(k) / static_cast<double>(kPoints - 1);
    }
    const auto start = std::chrono::steady_clock::now();
    const auto grid = markets::materialize(curve, times);
    const auto stop = std::chrono::steady_clock::now();
    g_sink = static_cast<double>(grid.discount.back());
    const double seconds = std::chrono::duration<double>(stop - start).count();
    QTA_LOG_INFO("bench", "materialize {} points  {} M points/s  ({} ns/point)", kPoints,
                 util::num(static_cast<double>(kPoints) / seconds / 1e6, 2),
                 util::num(seconds * 1e9 / static_cast<double>(kPoints), 2));
}

} // namespace

int main() {
    benchDiscount();
    benchMaterialize();
    benchBootstrap();
    QTA_LOG_INFO("bench", "done");
    return 0;
}
