#include "quantape/log/Log.h"
#include "quantape/markets/Curves/CurveBuilder.h"
#include "quantape/markets/Curves/CurveOnGrid.h"
#include "quantape/markets/Curves/DiscountCurve.h"
#include "quantape/markets/Curves/MultiCurveSet.h"
#include "quantape/markets/Curves/ParametricCurve.h"
#include "quantape/markets/Curves/SpreadCurve.h"
#include "quantape/markets/Curves/TurnOverlay.h"
#include "quantape/util/Check.h"

#include <cmath>
#include <memory>
#include <stdexcept>
#include <vector>

using namespace quantape;

namespace {

using markets::CurvePillar;
using markets::DiscountCurve;
using markets::InterpolationScheme;
using markets::InterpolationSpace;
using markets::PillarKind;

DiscountCurve<double> makeLogDiscountCurve() {
    return DiscountCurve<double>({0.0, 1.0, 2.0, 3.0}, {0.0, 0.03, 0.035, 0.04},
                                 InterpolationSpace::LogDiscount, InterpolationScheme::Linear);
}

void testLogDiscountLinear() {
    const DiscountCurve<double> curve = makeLogDiscountCurve();

    CHECK(curve.size() == 4);
    CHECK(curve.discount(0.0) == 1.0);
    CHECK(curve.discount(-1.0) == 1.0);

    // Nodes are exact.
    for (std::size_t i = 0; i < curve.size(); ++i) {
        const double t = curve.times()[i];
        util::checkClose("node discount", curve.discount(t), std::exp(-curve.zeros()[i] * t),
                         1e-15);
    }

    // Between nodes: linear in log discount (y = z t) => constant forward.
    // y nodes: (0, 0), (1, 0.03), (2, 0.07), (3, 0.12).
    util::checkClose("mid discount", curve.discount(1.5), std::exp(-0.05), 1e-15);
    util::checkClose("mid zero", curve.zero(1.5), 0.05 / 1.5, 1e-15);
    util::checkClose("segment forward", curve.forward(1.0, 2.0), 0.04, 1e-15);
    util::checkClose("cross forward", curve.forward(0.5, 1.5), 0.035, 1e-15);

    // Flat instantaneous forward beyond the last node.
    util::checkClose("extrapolated forward", curve.forward(2.5, 3.5), curve.forward(2.0, 3.0),
                     1e-15);

    bool threw = false;
    try {
        (void)curve.forward(2.0, 2.0);
    } catch (const std::invalid_argument&) {
        threw = true;
    }
    CHECK(threw);
}

void testLinearZero() {
    const DiscountCurve<double> curve({0.0, 1.0, 2.0}, {0.0, 0.03, 0.05}, InterpolationSpace::Zero,
                                      InterpolationScheme::Linear);
    util::checkClose("linear zero mid", curve.zero(1.5), 0.04, 1e-15);
    util::checkClose("linear zero discount", curve.discount(1.5), std::exp(-0.04 * 1.5), 1e-15);
}

void testAkimaZero() {
    const DiscountCurve<double> curve({0.0, 1.0, 2.0, 3.0}, {0.0, 0.03, 0.033, 0.04},
                                      InterpolationSpace::Zero, InterpolationScheme::Akima);
    for (std::size_t i = 0; i < curve.size(); ++i) {
        util::checkClose("akima node", curve.zero(curve.times()[i]), curve.zeros()[i], 1e-14);
    }
    const double left = curve.zero(1.0 - 1e-9);
    const double right = curve.zero(1.0 + 1e-9);
    util::checkClose("akima continuity", left, right, 1e-8);

    // Akima is available in LogDiscount space as well (cubic on log-DF).
    const DiscountCurve<double> logAkima({0.0, 1.0, 2.0, 3.0}, {0.0, 0.03, 0.033, 0.04},
                                         InterpolationSpace::LogDiscount,
                                         InterpolationScheme::Akima);
    for (std::size_t i = 0; i < logAkima.size(); ++i) {
        util::checkClose("akima log node", logAkima.discount(logAkima.times()[i]),
                         std::exp(-logAkima.zeros()[i] * logAkima.times()[i]), 1e-14);
    }
}

void testTensionSpline() {
    const std::vector<double> times{0.0, 1.0, 2.0, 3.0, 4.0};
    const std::vector<double> zeros{0.0, 0.03, 0.035, 0.038, 0.04};
    const DiscountCurve<double> curve(times, zeros, InterpolationSpace::LogDiscount,
                                      InterpolationScheme::TensionSpline, 8.0);
    for (std::size_t i = 0; i < curve.size(); ++i) {
        const double t = curve.times()[i];
        util::checkClose("tension node discount", curve.discount(t),
                         std::exp(-curve.zeros()[i] * t), 1e-14);
    }
    const double left = curve.zero(2.0 - 1e-9);
    const double right = curve.zero(2.0 + 1e-9);
    util::checkClose("tension continuity", left, right, 1e-7);

    bool threw = false;
    try {
        (void)DiscountCurve<double>(times, zeros, InterpolationSpace::LogDiscount,
                                    InterpolationScheme::TensionSpline, 0.0);
    } catch (const std::invalid_argument&) {
        threw = true;
    }
    CHECK(threw);

    // Larger tension localizes the influence of a bumped node.
    const auto farImpact = [&](double tension) {
        const DiscountCurve<double> base(times, zeros, InterpolationSpace::LogDiscount,
                                         InterpolationScheme::TensionSpline, tension);
        std::vector<double> bumped = zeros;
        bumped[2] += 1e-4;
        const DiscountCurve<double> up(times, bumped, InterpolationSpace::LogDiscount,
                                       InterpolationScheme::TensionSpline, tension);
        return std::abs(up.discount(3.5) - base.discount(3.5));
    };
    CHECK(farImpact(20.0) < farImpact(2.0));
}

void testMaterialize() {
    const DiscountCurve<double> curve = makeLogDiscountCurve();
    const std::vector<double> times{0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0};
    const auto grid = markets::materialize(curve, times);
    CHECK(grid.size() == times.size());
    CHECK(grid.nSteps() == times.size() - 1);
    for (std::size_t k = 0; k < times.size(); ++k) {
        util::checkClose("grid discount", grid.discountAt(k), curve.discount(times[k]), 1e-15);
        util::checkClose("grid zero", grid.zero[k], curve.zero(times[k]), 1e-15);
        if (k + 1 < times.size()) {
            util::checkClose("grid forward", grid.forwardAt(k),
                             curve.forward(times[k], times[k + 1]), 1e-14);
            const double dt = times[k + 1] - times[k];
            util::checkClose("grid consistency", static_cast<double>(grid.discountAt(k + 1)),
                             static_cast<double>(grid.discountAt(k)) *
                                 std::exp(-static_cast<double>(grid.forwardAt(k)) * dt),
                             1e-14);
        }
    }
}

void testSpreadCurve() {
    static_assert(markets::CurveProvider<DiscountCurve<double>, double>);
    static_assert(markets::CurveProvider<markets::SpreadCurve<double>, double>);

    auto parent = std::make_shared<DiscountCurve<double>>(makeLogDiscountCurve());
    const markets::SpreadCurve<double> spread(parent, {0.0, 1.0, 2.0, 3.0},
                                              {0.0, 0.001, 0.0015, 0.002});
    for (const double t : {0.5, 1.0, 1.5, 2.0, 2.5}) {
        util::checkClose("spread zero", spread.zero(t),
                         static_cast<double>(parent->zero(t)) + spread.spread(t), 1e-15);
        util::checkClose("spread discount", spread.discount(t),
                         std::exp(-static_cast<double>(spread.zero(t)) * t), 1e-15);
    }
    util::checkClose(
        "spread forward", spread.forward(1.0, 2.0),
        (static_cast<double>(spread.zero(2.0)) * 2.0 - static_cast<double>(spread.zero(1.0))) / 1.0,
        1e-14);
}

void testMultiCurveSet() {
    auto usd = std::make_shared<DiscountCurve<double>>(makeLogDiscountCurve());
    auto eur = std::make_shared<DiscountCurve<double>>(
        std::vector<double>{0.0, 1.0, 2.0}, std::vector<double>{0.0, 0.02, 0.025},
        InterpolationSpace::LogDiscount, InterpolationScheme::Linear);
    markets::MultiCurveSet<double> set;
    const markets::CurveKey usdKey{"USD", markets::CurveRole::Discount,
                                   datetime::Period(1, datetime::TimeUnit::Days), "USD"};
    set.add(usdKey, usd);
    set.add(markets::CurveKey{"EUR", markets::CurveRole::Discount,
                              datetime::Period(1, datetime::TimeUnit::Days), "EUR"},
            eur);
    CHECK(set.size() == 2);
    CHECK(set.discountCurve("USD").get() == usd.get());
    CHECK(set.discountCurve("EUR").get() == eur.get());

    // Lookups hand out references into stable storage: hold them across adds.
    const markets::MultiCurveSet<double>::CurveEntry& usdEntry = set.find(usdKey);
    const auto& usdCurveRef = set.discountCurve("USD");

    auto usdEurCollateral = std::make_shared<DiscountCurve<double>>(
        std::vector<double>{0.0, 1.0, 2.0}, std::vector<double>{0.0, 0.015, 0.02},
        InterpolationSpace::LogDiscount, InterpolationScheme::Linear);
    set.add(markets::CurveKey{"USD", markets::CurveRole::Discount,
                              datetime::Period(1, datetime::TimeUnit::Days), "EUR"},
            usdEurCollateral);
    CHECK(set.size() == 3);

    // References captured before the insert still address the same entries.
    CHECK(&set.find(usdKey) == &usdEntry);
    CHECK(&set.discountCurve("USD", "USD") == &usdCurveRef);
    CHECK(usdEntry.discount.get() == usd.get());

    // The one-argument lookup is ambiguous now that USD has two discount curves.
    bool threw = false;
    try {
        (void)set.discountCurve("USD");
    } catch (const std::invalid_argument&) {
        threw = true;
    }
    CHECK(threw);

    // Collateral selects the exact curve.
    CHECK(set.discountCurve("USD", "USD").get() == usd.get());
    CHECK(set.discountCurve("USD", "EUR").get() == usdEurCollateral.get());
    CHECK(set.discountCurve("EUR", "EUR").get() == eur.get());

    threw = false;
    try {
        set.add(usdKey, usd);
    } catch (const std::invalid_argument&) {
        threw = true;
    }
    CHECK(threw);

    threw = false;
    try {
        (void)set.discountCurve("JPY");
    } catch (const std::out_of_range&) {
        threw = true;
    }
    CHECK(threw);

    threw = false;
    try {
        (void)set.discountCurve("USD", "JPY");
    } catch (const std::out_of_range&) {
        threw = true;
    }
    CHECK(threw);
}

void testCompounding() {
    const DiscountCurve<double> curve = makeLogDiscountCurve();
    const double tau = 1.0;
    util::checkClose("compound factor", curve.compoundingFactor(1.0, 2.0),
                     curve.discount(1.0) / curve.discount(2.0), 1e-15);
    util::checkClose("compounded rate", curve.compoundedRate(1.0, 2.0, tau), (std::exp(0.04) - 1.0),
                     1e-14);

    const std::vector<double> boundaries{0.0, 0.25, 0.5, 0.75, 1.0};
    double manual = 0.0;
    for (std::size_t k = 1; k < boundaries.size(); ++k) {
        manual += (curve.discount(boundaries[k - 1]) / curve.discount(boundaries[k]) - 1.0) / 0.25;
    }
    util::checkClose("averaged rate", curve.averagedRate(boundaries), manual / 4.0, 1e-15);

    // Daily compounding telescopes to the compound factor.
    double product = 1.0;
    for (std::size_t k = 1; k < boundaries.size(); ++k) {
        const double simple =
            (curve.discount(boundaries[k - 1]) / curve.discount(boundaries[k]) - 1.0) / 0.25;
        product *= 1.0 + simple * 0.25;
    }
    util::checkClose("compounding telescopes", product, curve.compoundingFactor(0.0, 1.0), 1e-15);
}

void testBasisBootstrap() {
    const datetime::Date reference(2026, 9, 29);
    const datetime::DayCounter zeroDayCounter(datetime::DayCount::Actual365Fixed);
    const datetime::Calendar calendar = datetime::Calendar::noHolidays();

    std::vector<datetime::Date> dates;
    for (int years = 1; years <= 5; ++years) {
        dates.push_back(reference.plusYears(years));
    }
    std::vector<double> parentZeros;
    for (const auto& date : dates) {
        const double t = datetime::yearFraction(reference, date, zeroDayCounter);
        parentZeros.push_back(0.04 - 0.002 * std::exp(-0.5 * t));
    }
    auto parent = std::make_shared<DiscountCurve<double>>(
        reference, dates, zeroDayCounter, parentZeros, InterpolationSpace::LogDiscount,
        InterpolationScheme::Linear);
    const std::vector<double> targetSpreads{0.0005, 0.0007, 0.0009, 0.001, 0.0011};
    std::vector<markets::BasisPillar> pillars;
    std::vector<double> spreadTimes{0.0};
    std::vector<double> spreads{0.0};
    for (std::size_t i = 0; i < dates.size(); ++i) {
        spreadTimes.push_back(datetime::yearFraction(reference, dates[i], zeroDayCounter));
        spreads.push_back(targetSpreads[i]);
    }
    const markets::SpreadCurve<double> target(parent, spreadTimes, spreads);
    for (std::size_t i = 0; i < dates.size(); ++i) {
        markets::BasisPillar pillar;
        pillar.maturity = dates[i];
        pillar.floatTenor = datetime::Period(3, datetime::TimeUnit::Months);
        pillar.quoteDayCounter = zeroDayCounter;
        pillar.calendar = calendar;
        pillar.spread = markets::impliedBasisSpread(target, pillar, reference, zeroDayCounter);
        pillars.push_back(pillar);
    }

    const markets::SpreadCurve<double> curve = markets::bootstrapSpreadCurve(
        parent, reference, zeroDayCounter, InterpolationScheme::Linear, pillars);
    for (std::size_t i = 0; i < pillars.size(); ++i) {
        util::checkClose("recovered basis spread", curve.spreadNodes().zeros()[i + 1],
                         targetSpreads[i], 1e-11);
        util::checkClose("basis reprice",
                         markets::impliedBasisSpread(curve, pillars[i], reference, zeroDayCounter),
                         pillars[i].spread, 1e-11);
    }
}

void testMonotoneCubic() {
    const std::vector<double> times{0.0, 1.0, 2.0, 3.0, 4.0};
    const std::vector<double> zeros{0.0, 0.03, 0.033, 0.036, 0.04};
    const DiscountCurve<double> curve(times, zeros, InterpolationSpace::Zero,
                                      InterpolationScheme::MonotoneCubic);
    for (std::size_t i = 0; i < curve.size(); ++i) {
        util::checkClose("monotone node", curve.zero(times[i]), zeros[i], 1e-14);
    }
    // No overshoot on monotone data, and the derivative keeps the sign.
    double previous = curve.zero(0.0);
    for (double t = 0.05; t <= 4.0; t += 0.05) {
        const double current = curve.zero(t);
        CHECK(current >= previous - 1e-14);
        previous = current;
    }
    bool threw = false;
    try {
        std::vector<double> weights;
        curve.zeroNodeWeights(2.5, weights);
    } catch (const std::invalid_argument&) {
        threw = true;
    }
    CHECK(threw);
}

void testParametricCurve() {
    const auto sampleTimes = [] {
        std::vector<double> times;
        for (double t = 0.25; t <= 30.0; t += 0.25) {
            times.push_back(t);
        }
        return times;
    }();
    {
        markets::ParametricCurve exact;
        exact.form = markets::ParametricForm::NelsonSiegel;
        exact.parameters = {0.03, -0.02, -0.01, 2.0, 0.0, 5.0};
        std::vector<double> zeros;
        for (const double t : sampleTimes) {
            zeros.push_back(exact.zero(t));
        }
        const markets::ParametricCurve fitted = markets::ParametricCurve::fit(
            markets::ParametricForm::NelsonSiegel, sampleTimes, zeros);
        for (const double t : sampleTimes) {
            util::checkClose("NS fit", fitted.zero(t), exact.zero(t), 1e-5);
        }
        util::checkClose("NS discount", fitted.discount(5.0), exact.discount(5.0), 1e-5);
    }
    {
        markets::ParametricCurve exact;
        exact.form = markets::ParametricForm::Svensson;
        exact.parameters = {0.03, -0.02, -0.01, 2.0, 0.005, 5.0};
        std::vector<double> zeros;
        for (const double t : sampleTimes) {
            zeros.push_back(exact.zero(t));
        }
        const markets::ParametricCurve fitted =
            markets::ParametricCurve::fit(markets::ParametricForm::Svensson, sampleTimes, zeros);
        for (const double t : {0.5, 2.0, 5.0, 10.0, 20.0, 30.0}) {
            util::checkClose("Svensson fit", fitted.zero(t), exact.zero(t), 1e-5);
        }
    }
}

void testMixedLinearCubic() {
    const std::vector<double> times{0.0, 1.0, 2.0, 3.0, 4.0};
    const std::vector<double> zeros{0.0, 0.03, 0.033, 0.036, 0.04};
    const DiscountCurve<double> mixed(times, zeros, InterpolationSpace::Zero,
                                      InterpolationScheme::MixedLinearCubic, 0.0, 2);
    const DiscountCurve<double> akima(times, zeros, InterpolationSpace::Zero,
                                      InterpolationScheme::Akima);
    // Left of the switch: piecewise linear.
    util::checkClose("mixed linear segment", mixed.zero(0.5), 0.5 * (zeros[0] + zeros[1]), 1e-14);
    util::checkClose("mixed linear segment 2", mixed.zero(1.5), 0.5 * (zeros[1] + zeros[2]), 1e-14);
    // Right of the switch: identical to Akima.
    for (const double t : {2.2, 2.7, 3.5, 3.9}) {
        util::checkClose("mixed cubic segment", mixed.zero(t), akima.zero(t), 1e-14);
    }
}

void testTurnOverlay() {
    static_assert(markets::CurveProvider<markets::TurnOverlay<double>, double>);
    auto base = std::make_shared<DiscountCurve<double>>(
        std::vector<double>{0.0, 1.0, 2.0, 3.0}, std::vector<double>{0.0, 0.03, 0.03, 0.03},
        InterpolationSpace::LogDiscount, InterpolationScheme::Linear);
    const double turnTime = 2.0;
    const double amplitude = 0.001;
    const markets::TurnOverlay<double> overlay(base, {{turnTime, amplitude}});

    // Flat base: the difference across the turn is the jump alone.
    const double before = overlay.zero(1.5);
    const double after = overlay.zero(2.5);
    util::checkClose("turn zero jump", after - before, amplitude * turnTime / 2.5, 1e-15);
    // DF ratio across the turn is the base carry times exp(-d tTurn).
    util::checkClose("turn DF multiplier", overlay.discount(2.5) / overlay.discount(1.5),
                     std::exp(-0.03 * 1.0) * std::exp(-amplitude * turnTime), 1e-14);
    // Before the turn the overlay equals the base.
    util::checkClose("turn before base", overlay.discount(1.5), base->discount(1.5), 1e-15);
    bool threw = false;
    try {
        (void)markets::TurnOverlay<double>(nullptr, {});
    } catch (const std::invalid_argument&) {
        threw = true;
    }
    CHECK(threw);
}

void testNonAct365ZeroClock() {
    const datetime::Date reference(2026, 9, 29);
    const datetime::Date pillar = reference.plusYears(1);
    const datetime::DayCounter act360(datetime::DayCount::Actual360);
    const DiscountCurve<double> curve(reference, {pillar}, act360, {0.04},
                                      InterpolationSpace::LogDiscount, InterpolationScheme::Linear);
    const double expected = datetime::yearFraction(reference, pillar, act360);
    util::checkClose("ACT/360 zero clock", curve.times()[1], expected, 1e-15);
    CHECK(std::abs(expected - datetime::yearFraction(
                                  reference, pillar,
                                  datetime::DayCounter(datetime::DayCount::Actual365Fixed))) >
          1e-4);
}

void testSchemeSpaceMatrix() {
    const datetime::Date reference(2026, 9, 29);
    const datetime::DayCounter zeroDc(datetime::DayCount::Actual365Fixed);
    const datetime::Calendar calendar = datetime::Calendar::noHolidays();
    std::vector<datetime::Date> dates{reference};
    for (int years = 1; years <= 5; ++years) {
        dates.push_back(reference.plusYears(years));
    }
    const struct {
        InterpolationSpace space;
        InterpolationScheme scheme;
        double tension;
        int switchIndex;
        bool weights;
    } cases[] = {
        {InterpolationSpace::Zero, InterpolationScheme::Linear, 0.0, 1, true},
        {InterpolationSpace::Zero, InterpolationScheme::Akima, 0.0, 1, false},
        {InterpolationSpace::Zero, InterpolationScheme::TensionSpline, 8.0, 1, true},
        {InterpolationSpace::Zero, InterpolationScheme::MonotoneCubic, 0.0, 1, false},
        {InterpolationSpace::Zero, InterpolationScheme::MixedLinearCubic, 0.0, 2, false},
        {InterpolationSpace::LogDiscount, InterpolationScheme::Linear, 0.0, 1, true},
        {InterpolationSpace::LogDiscount, InterpolationScheme::Akima, 0.0, 1, false},
        {InterpolationSpace::LogDiscount, InterpolationScheme::TensionSpline, 8.0, 1, true},
        {InterpolationSpace::LogDiscount, InterpolationScheme::MonotoneCubic, 0.0, 1, false},
        {InterpolationSpace::LogDiscount, InterpolationScheme::MixedLinearCubic, 0.0, 2, false},
    };
    for (const auto& c : cases) {
        std::vector<double> targetZeros;
        for (const auto& date : dates) {
            const double t = datetime::yearFraction(reference, date, zeroDc);
            targetZeros.push_back(0.04 + 0.001 * t - 0.002 * std::exp(-0.7 * t));
        }
        const DiscountCurve<double> target(reference, dates, zeroDc, targetZeros, c.space, c.scheme,
                                           c.tension, c.switchIndex);

        // Bootstrap the same node grid from deposit/OIS quotes of the target.
        std::vector<CurvePillar> pillars;
        CurvePillar deposit;
        deposit.maturity = dates[1];
        deposit.kind = PillarKind::Deposit;
        deposit.quoteDayCounter = datetime::DayCounter(datetime::DayCount::Actual360);
        deposit.calendar = calendar;
        deposit.quote = markets::impliedQuote(deposit, reference, target);
        pillars.push_back(deposit);
        for (std::size_t i = 2; i < dates.size(); ++i) {
            CurvePillar swap;
            swap.maturity = dates[i];
            swap.kind = PillarKind::OisSwap;
            swap.quoteDayCounter = zeroDc;
            swap.calendar = calendar;
            swap.quote = markets::impliedQuote(swap, reference, target);
            pillars.push_back(swap);
        }
        const DiscountCurve<double> curve = markets::bootstrapDiscountCurve(
            reference, zeroDc, c.space, c.scheme, pillars, 1e-14, c.tension, c.switchIndex);

        // Exact reprice and node recovery in every space/scheme combination.
        for (std::size_t i = 0; i < pillars.size(); ++i) {
            util::checkClose("matrix reprice", markets::impliedQuote(pillars[i], reference, curve),
                             pillars[i].quote, 1e-10);
            util::checkClose("matrix recovery", curve.zeros()[i + 1], targetZeros[i + 1], 5e-9);
        }
        // Continuity of zeros across knot boundaries, finite forwards, and
        // monotone decreasing discounts.
        double previousDf = curve.discount(0.0);
        std::vector<double> weights;
        for (double t = 0.05; t <= 5.0; t += 0.05) {
            const double df = curve.discount(t);
            CHECK(df > 0.0);
            CHECK(df <= previousDf + 1e-12);
            previousDf = df;
            const double left = curve.zero(t - 1e-9);
            const double right = curve.zero(t + 1e-9);
            util::checkClose("matrix continuity", left, right, 1e-5);
        }
        if (c.weights) {
            // Partition holds over ALL nodes (including the fixed t=0 node)
            // for global schemes; zeroNodeWeights drops the fixed node.
            curve.spaceValueWeights(2.5, weights);
            double total = 0.0;
            for (const double w : weights) {
                total += w;
            }
            util::checkClose("matrix weights partition", total, 1.0, 1e-10);
        }
        if (c.space == InterpolationSpace::LogDiscount && c.scheme == InterpolationScheme::Linear) {
            util::checkClose("log-linear flat forward", curve.forward(1.1, 1.2),
                             curve.forward(1.3, 1.4), 1e-14);
        }
        // Extrapolation-slope consistency at the ACTUAL last knot (ACT/365F
        // year fractions are not integers, so checks must use `times().back()`).
        if (c.space == InterpolationSpace::Zero) {
            // In Zero space z(t) equals the interpolated value, so a finite
            // extension step is exactly proportional to the end slope. In
            // LogDiscount space z = y/t adds a 1/t curvature, which is covered
            // by the flat-forward checks below instead.
            const double knot = curve.times().back();
            const double delta = 0.4;
            const double extension = curve.zero(knot + delta) - curve.zero(knot);
            const double inside = curve.zero(knot) - curve.zero(knot - 1e-6);
            util::checkClose("zero extrapolation slope", extension, delta * inside / 1e-6, 5e-6);
        }
        if (c.space == InterpolationSpace::LogDiscount) {
            const double knot = curve.times().back();
            const double beyond = curve.forward(knot, knot + 0.5);
            util::checkClose("flat forward extension", curve.forward(knot + 0.5, knot + 1.0),
                             beyond, 1e-12);
            util::checkClose("flat forward slope", curve.forward(knot - 1e-6, knot), beyond, 5e-6);
        }
        // Forwards finite and free of NaN across the grid.
        for (double t = 0.1; t < 5.0; t += 0.1) {
            const double f = curve.forward(t, t + 0.1);
            CHECK(quantape::util::isFiniteBitwise(f));
        }
    }
}

void testErrors() {
    bool threw = false;
    try {
        (void)DiscountCurve<double>(std::vector<double>{}, std::vector<double>{});
    } catch (const std::invalid_argument&) {
        threw = true;
    }
    CHECK(threw);

    threw = false;
    try {
        (void)DiscountCurve<double>(std::vector<double>{0.0, 2.0, 1.0},
                                    std::vector<double>{0.0, 0.03, 0.04});
    } catch (const std::invalid_argument&) {
        threw = true;
    }
    CHECK(threw);
}

void testBootstrapRecovery() {
    const datetime::Date reference(2026, 9, 29);
    const datetime::DayCounter zeroDayCounter(datetime::DayCount::Actual365Fixed);
    const datetime::Calendar calendar = datetime::Calendar::noHolidays();

    std::vector<datetime::Date> pillarDates;
    pillarDates.push_back(datetime::Period(6, datetime::TimeUnit::Months).advance(reference));
    for (int years = 1; years <= 5; ++years) {
        pillarDates.push_back(reference.plusYears(years));
    }

    const auto targetZero = [](double t) { return 0.04 - 0.002 * std::exp(-0.5 * t); };
    std::vector<double> targetZeros;
    targetZeros.reserve(pillarDates.size());
    for (const auto& date : pillarDates) {
        targetZeros.push_back(targetZero(datetime::yearFraction(reference, date, zeroDayCounter)));
    }
    const DiscountCurve<double> target(reference, pillarDates, zeroDayCounter, targetZeros,
                                       InterpolationSpace::LogDiscount,
                                       InterpolationScheme::Linear);

    std::vector<CurvePillar> pillars;
    CurvePillar deposit;
    deposit.maturity = pillarDates.front();
    deposit.kind = PillarKind::Deposit;
    deposit.quoteDayCounter = datetime::DayCounter(datetime::DayCount::Actual360);
    deposit.calendar = calendar;
    deposit.quote = markets::impliedQuote(deposit, reference, target);
    pillars.push_back(deposit);
    for (std::size_t i = 1; i < pillarDates.size(); ++i) {
        CurvePillar swap;
        swap.maturity = pillarDates[i];
        swap.kind = PillarKind::OisSwap;
        swap.quoteDayCounter = zeroDayCounter;
        swap.calendar = calendar;
        swap.quote = markets::impliedQuote(swap, reference, target);
        pillars.push_back(swap);
    }

    const DiscountCurve<double> curve =
        markets::bootstrapDiscountCurve(reference, zeroDayCounter, InterpolationSpace::LogDiscount,
                                        InterpolationScheme::Linear, pillars);
    CHECK(curve.size() == pillars.size() + 1);
    CHECK(curve.discount(0.0) == 1.0);

    for (std::size_t i = 0; i < pillars.size(); ++i) {
        util::checkClose("recovered zero", curve.zeros()[i + 1], targetZeros[i], 1e-11);
        util::checkClose("reprice", markets::impliedQuote(pillars[i], reference, curve),
                         pillars[i].quote, 1e-11);
    }
}

void testOisPaymentLag() {
    const datetime::Date reference(2026, 9, 29);
    const datetime::DayCounter zeroDayCounter(datetime::DayCount::Actual365Fixed);
    const datetime::Calendar calendar = datetime::Calendar::noHolidays();
    std::vector<datetime::Date> pillarDates;
    for (int years = 1; years <= 5; ++years) {
        pillarDates.push_back(reference.plusYears(years));
    }
    const auto targetZero = [](double t) { return 0.035 + 0.001 * t; };
    std::vector<double> targetZeros;
    for (const auto& date : pillarDates) {
        targetZeros.push_back(targetZero(datetime::yearFraction(reference, date, zeroDayCounter)));
    }
    const DiscountCurve<double> target(reference, pillarDates, zeroDayCounter, targetZeros,
                                       InterpolationSpace::LogDiscount,
                                       InterpolationScheme::Linear);
    std::vector<CurvePillar> pillars;
    for (const auto& date : pillarDates) {
        CurvePillar swap;
        swap.maturity = date;
        swap.kind = PillarKind::OisSwap;
        swap.quoteDayCounter = zeroDayCounter;
        swap.calendar = calendar;
        swap.paymentLag = 2;
        swap.quote = markets::impliedQuote(swap, reference, target);
        pillars.push_back(swap);
    }
    const DiscountCurve<double> curve =
        markets::bootstrapDiscountCurve(reference, zeroDayCounter, InterpolationSpace::LogDiscount,
                                        InterpolationScheme::Linear, pillars);
    for (const CurvePillar& pillar : pillars) {
        util::checkClose("OIS payment-lag reprice", markets::impliedQuote(pillar, reference, curve),
                         pillar.quote, 1e-11);
    }
}

void testOisBusinessDayConvention() {
    const datetime::Date reference(2026, 9, 29);
    const datetime::DayCounter zeroDayCounter(datetime::DayCount::Actual365Fixed);
    const datetime::Calendar calendar = datetime::Calendar::weekendsOnly();
    const datetime::Date maturity(2027, 7, 31); // Saturday
    CHECK(maturity.weekday() == datetime::Weekday::Saturday);
    const datetime::Schedule following(
        reference, maturity, datetime::Period(1, datetime::TimeUnit::Years), calendar,
        datetime::BusinessDayConvention::Following, datetime::DateGeneration::Forward, false);
    const datetime::Schedule modified(reference, maturity,
                                      datetime::Period(1, datetime::TimeUnit::Years), calendar,
                                      datetime::BusinessDayConvention::ModifiedFollowing,
                                      datetime::DateGeneration::Forward, false);
    CHECK(following.endDate() == datetime::Date(2027, 8, 2));
    CHECK(modified.endDate() == datetime::Date(2027, 7, 30));

    const std::vector<datetime::Date> pillarDates{maturity};
    const std::vector<double> targetZeros{0.04};
    const DiscountCurve<double> target(reference, pillarDates, zeroDayCounter, targetZeros,
                                       InterpolationSpace::LogDiscount,
                                       InterpolationScheme::Linear);
    for (const auto convention : {datetime::BusinessDayConvention::Following,
                                  datetime::BusinessDayConvention::ModifiedFollowing}) {
        CurvePillar swap;
        swap.maturity = maturity;
        swap.kind = PillarKind::OisSwap;
        swap.quoteDayCounter = zeroDayCounter;
        swap.calendar = calendar;
        swap.businessDayConvention = convention;
        swap.quote = markets::impliedQuote(swap, reference, target);
        const DiscountCurve<double> curve = markets::bootstrapDiscountCurve(
            reference, zeroDayCounter, InterpolationSpace::LogDiscount, InterpolationScheme::Linear,
            std::vector<CurvePillar>{swap});
        util::checkClose("OIS convention reprice", markets::impliedQuote(swap, reference, curve),
                         swap.quote, 1e-11);
    }
}

void testPillarMaturityAdjustment() {
    const datetime::Date reference(2026, 9, 29);
    const datetime::DayCounter zeroDayCounter(datetime::DayCount::Actual365Fixed);
    const datetime::Calendar calendar = datetime::Calendar::weekendsOnly();
    const datetime::Date saturday(2027, 7, 31);
    const datetime::Date friday(2027, 7, 30);

    const DiscountCurve<double> target(
        reference, std::vector<datetime::Date>{friday}, zeroDayCounter, std::vector<double>{0.043},
        InterpolationSpace::LogDiscount, InterpolationScheme::Linear);

    CurvePillar deposit;
    deposit.maturity = saturday;
    deposit.kind = PillarKind::Deposit;
    deposit.quoteDayCounter = datetime::DayCounter(datetime::DayCount::Actual360);
    deposit.calendar = calendar;
    deposit.quote = markets::impliedQuote(deposit, reference, target);
    const DiscountCurve<double> curve = markets::bootstrapDiscountCurve(
        reference, zeroDayCounter, InterpolationSpace::LogDiscount, InterpolationScheme::Linear,
        std::vector<CurvePillar>{deposit});
    util::checkClose("adjusted deposit node time", curve.times().back(),
                     datetime::yearFraction(reference, friday, zeroDayCounter), 1e-15);
    util::checkClose("adjusted deposit reprice", markets::impliedQuote(deposit, reference, curve),
                     deposit.quote, 1e-11);

    CurvePillar raw = deposit;
    raw.businessDayConvention = datetime::BusinessDayConvention::Unadjusted;
    raw.quote = markets::impliedQuote(raw, reference, target);
    const DiscountCurve<double> rawCurve =
        markets::bootstrapDiscountCurve(reference, zeroDayCounter, InterpolationSpace::LogDiscount,
                                        InterpolationScheme::Linear, std::vector<CurvePillar>{raw});
    util::checkClose("unadjusted deposit node time", rawCurve.times().back(),
                     datetime::yearFraction(reference, saturday, zeroDayCounter), 1e-15);

    CurvePillar fra;
    fra.kind = PillarKind::Fra;
    fra.start = datetime::Date(2027, 1, 31); // Sunday
    fra.maturity = saturday;
    fra.quoteDayCounter = datetime::DayCounter(datetime::DayCount::Actual360);
    fra.calendar = calendar;
    fra.quote = markets::impliedQuote(fra, reference, target);
    const DiscountCurve<double> fraCurve =
        markets::bootstrapDiscountCurve(reference, zeroDayCounter, InterpolationSpace::LogDiscount,
                                        InterpolationScheme::Linear, std::vector<CurvePillar>{fra});
    util::checkClose("adjusted FRA reprice", markets::impliedQuote(fra, reference, fraCurve),
                     fra.quote, 1e-11);
}

void testImpliedQuoteUnknownKind() {
    const DiscountCurve<double> curve = makeLogDiscountCurve();
    const datetime::Date reference(2026, 9, 29);
    CurvePillar pillar;
    pillar.kind = static_cast<PillarKind>(42);
    pillar.maturity = datetime::Date(2027, 9, 29);
    bool threw = false;
    try {
        (void)markets::impliedQuote(pillar, reference, curve);
    } catch (const std::invalid_argument&) {
        threw = true;
    }
    CHECK(threw);
}

} // namespace

int main() {
    testLogDiscountLinear();
    testLinearZero();
    testAkimaZero();
    testTensionSpline();
    testMaterialize();
    testSpreadCurve();
    testMultiCurveSet();
    testCompounding();
    testBasisBootstrap();
    testMonotoneCubic();
    testParametricCurve();
    testMixedLinearCubic();
    testTurnOverlay();
    testNonAct365ZeroClock();
    testSchemeSpaceMatrix();
    testErrors();
    testBootstrapRecovery();
    testOisPaymentLag();
    testOisBusinessDayConvention();
    testPillarMaturityAdjustment();
    testImpliedQuoteUnknownKind();
    QTA_LOG_INFO("test", "test_curve: ok");
    return 0;
}
