#include "quantape/markets/Curves/CurveBuilder.h"
#include "quantape/markets/Curves/CurveOnGrid.h"
#include "quantape/markets/Curves/DiscountCurve.h"
#include "quantape/markets/Curves/MultiCurveSet.h"
#include "quantape/markets/Curves/ParametricCurve.h"
#include "quantape/markets/Curves/SpreadCurve.h"
#include "quantape/markets/Curves/TurnOverlay.h"

#include <cmath>
#include <memory>
#include <stdexcept>
#include <vector>

#include "support/GtestSupport.h"

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

} // namespace

TEST(DiscountCurve, logDiscountLinear) {
    const DiscountCurve<double> curve = makeLogDiscountCurve();

    EXPECT_TRUE(curve.size() == 4);
    EXPECT_TRUE(curve.discount(0.0) == 1.0);
    EXPECT_TRUE(curve.discount(-1.0) == 1.0);

    // Nodes are exact.
    for (std::size_t i = 0; i < curve.size(); ++i) {
        const double t = curve.times()[i];
        CHECK_CLOSE("node discount", curve.discount(t), std::exp(-curve.zeros()[i] * t), 1e-15);
    }

    // Between nodes: linear in log discount (y = z t) => constant forward.
    // y nodes: (0, 0), (1, 0.03), (2, 0.07), (3, 0.12).
    CHECK_CLOSE("mid discount", curve.discount(1.5), std::exp(-0.05), 1e-15);
    CHECK_CLOSE("mid zero", curve.zero(1.5), 0.05 / 1.5, 1e-15);
    CHECK_CLOSE("segment forward", curve.forward(1.0, 2.0), 0.04, 1e-15);
    CHECK_CLOSE("cross forward", curve.forward(0.5, 1.5), 0.035, 1e-15);

    // Flat instantaneous forward beyond the last node.
    CHECK_CLOSE("extrapolated forward", curve.forward(2.5, 3.5), curve.forward(2.0, 3.0), 1e-15);

    EXPECT_THROW((void)curve.forward(2.0, 2.0), std::invalid_argument);
}

TEST(DiscountCurve, linearZero) {
    const DiscountCurve<double> curve({0.0, 1.0, 2.0}, {0.0, 0.03, 0.05}, InterpolationSpace::Zero,
                                      InterpolationScheme::Linear);
    CHECK_CLOSE("linear zero mid", curve.zero(1.5), 0.04, 1e-15);
    CHECK_CLOSE("linear zero discount", curve.discount(1.5), std::exp(-0.04 * 1.5), 1e-15);
}

TEST(DiscountCurve, akimaZero) {
    const DiscountCurve<double> curve({0.0, 1.0, 2.0, 3.0}, {0.0, 0.03, 0.033, 0.04},
                                      InterpolationSpace::Zero, InterpolationScheme::Akima);
    for (std::size_t i = 0; i < curve.size(); ++i) {
        CHECK_CLOSE("akima node", curve.zero(curve.times()[i]), curve.zeros()[i], 1e-14);
    }
    const double left = curve.zero(1.0 - 1e-9);
    const double right = curve.zero(1.0 + 1e-9);
    CHECK_CLOSE("akima continuity", left, right, 1e-8);

    // Akima is available in LogDiscount space as well (cubic on log-DF).
    const DiscountCurve<double> logAkima({0.0, 1.0, 2.0, 3.0}, {0.0, 0.03, 0.033, 0.04},
                                         InterpolationSpace::LogDiscount,
                                         InterpolationScheme::Akima);
    for (std::size_t i = 0; i < logAkima.size(); ++i) {
        CHECK_CLOSE("akima log node", logAkima.discount(logAkima.times()[i]),
                    std::exp(-logAkima.zeros()[i] * logAkima.times()[i]), 1e-14);
    }
}

TEST(DiscountCurve, tensionSpline) {
    const std::vector<double> times{0.0, 1.0, 2.0, 3.0, 4.0};
    const std::vector<double> zeros{0.0, 0.03, 0.035, 0.038, 0.04};
    const DiscountCurve<double> curve(times, zeros, InterpolationSpace::LogDiscount,
                                      InterpolationScheme::TensionSpline, 8.0);
    for (std::size_t i = 0; i < curve.size(); ++i) {
        const double t = curve.times()[i];
        CHECK_CLOSE("tension node discount", curve.discount(t), std::exp(-curve.zeros()[i] * t),
                    1e-14);
    }
    const double left = curve.zero(2.0 - 1e-9);
    const double right = curve.zero(2.0 + 1e-9);
    CHECK_CLOSE("tension continuity", left, right, 1e-7);

    EXPECT_THROW((void)DiscountCurve<double>(times, zeros, InterpolationSpace::LogDiscount,
                                             InterpolationScheme::TensionSpline, 0.0),
                 std::invalid_argument);

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
    EXPECT_TRUE(farImpact(20.0) < farImpact(2.0));
}

TEST(DiscountCurve, materialize) {
    const DiscountCurve<double> curve = makeLogDiscountCurve();
    const std::vector<double> times{0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0};
    const auto grid = markets::materialize(curve, times);
    EXPECT_TRUE(grid.size() == times.size());
    EXPECT_TRUE(grid.nSteps() == times.size() - 1);
    for (std::size_t k = 0; k < times.size(); ++k) {
        CHECK_CLOSE("grid discount", grid.discountAt(k), curve.discount(times[k]), 1e-15);
        CHECK_CLOSE("grid zero", grid.zero[k], curve.zero(times[k]), 1e-15);
        if (k + 1 < times.size()) {
            CHECK_CLOSE("grid forward", grid.forwardAt(k), curve.forward(times[k], times[k + 1]),
                        1e-14);
            const double dt = times[k + 1] - times[k];
            CHECK_CLOSE("grid consistency", static_cast<double>(grid.discountAt(k + 1)),
                        static_cast<double>(grid.discountAt(k)) *
                            std::exp(-static_cast<double>(grid.forwardAt(k)) * dt),
                        1e-14);
        }
    }
}

TEST(CurveComposition, spreadCurve) {
    static_assert(markets::CurveProvider<DiscountCurve<double>, double>);
    static_assert(markets::CurveProvider<markets::SpreadCurve<double>, double>);

    auto parent = std::make_shared<DiscountCurve<double>>(makeLogDiscountCurve());
    const markets::SpreadCurve<double> spread(parent, {0.0, 1.0, 2.0, 3.0},
                                              {0.0, 0.001, 0.0015, 0.002});
    for (const double t : {0.5, 1.0, 1.5, 2.0, 2.5}) {
        CHECK_CLOSE("spread zero", spread.zero(t),
                    static_cast<double>(parent->zero(t)) + spread.spread(t), 1e-15);
        CHECK_CLOSE("spread discount", spread.discount(t),
                    std::exp(-static_cast<double>(spread.zero(t)) * t), 1e-15);
    }
    CHECK_CLOSE(
        "spread forward", spread.forward(1.0, 2.0),
        (static_cast<double>(spread.zero(2.0)) * 2.0 - static_cast<double>(spread.zero(1.0))) / 1.0,
        1e-14);
}

TEST(CurveComposition, multiCurveSet) {
    auto usd = std::make_shared<const DiscountCurve<double>>(makeLogDiscountCurve());
    auto eur = std::make_shared<const DiscountCurve<double>>(
        std::vector<double>{0.0, 1.0, 2.0}, std::vector<double>{0.0, 0.02, 0.025},
        InterpolationSpace::LogDiscount, InterpolationScheme::Linear);
    const markets::CurveKey usdKey{"USD", markets::CurveRole::Discount,
                                   datetime::Period(1, datetime::TimeUnit::Days), "USD"};
    const markets::CurveKey eurKey{"EUR", markets::CurveRole::Discount,
                                   datetime::Period(1, datetime::TimeUnit::Days), "EUR"};
    const markets::CurveHandle::Ptr usdHandle = markets::CurveHandle::make(usd);
    const markets::CurveHandle::Ptr eurHandle = markets::CurveHandle::make(eur);
    markets::MultiCurveSet set = markets::MultiCurveSet::fromBuiltCurves(
        {markets::BuiltCurve{usdKey, markets::CurveRole::Discount, usdHandle},
         markets::BuiltCurve{eurKey, markets::CurveRole::Discount, eurHandle}});
    EXPECT_TRUE(set.size() == 2);
    EXPECT_TRUE(set.discountCurve("USD").get() == usdHandle.get());
    EXPECT_TRUE(set.discountCurve("EUR").get() == eurHandle.get());

    // The full key, including role and collateral, resolves the entry.
    const markets::BuiltCurve& usdEntry = set.find(usdKey);
    EXPECT_TRUE(usdEntry.key == usdKey);
    EXPECT_TRUE(usdEntry.role == markets::CurveRole::Discount);
    EXPECT_TRUE(usdEntry.curve.get() == usdHandle.get());

    // Lookups hand out references into stable storage: hold them across adds.
    const auto& usdCurveRef = set.discountCurve("USD");

    auto usdEurCollateral = std::make_shared<const DiscountCurve<double>>(
        std::vector<double>{0.0, 1.0, 2.0}, std::vector<double>{0.0, 0.015, 0.02},
        InterpolationSpace::LogDiscount, InterpolationScheme::Linear);
    const markets::CurveHandle::Ptr usdEurHandle = markets::CurveHandle::make(usdEurCollateral);
    set.add(
        markets::BuiltCurve{markets::CurveKey{"USD", markets::CurveRole::Discount,
                                              datetime::Period(1, datetime::TimeUnit::Days), "EUR"},
                            markets::CurveRole::Discount, usdEurHandle});
    EXPECT_TRUE(set.size() == 3);

    // References captured before the insert still address the same entries.
    EXPECT_TRUE(&set.find(usdKey) == &usdEntry);
    EXPECT_TRUE(&set.discountCurve("USD", "USD") == &usdCurveRef);
    EXPECT_TRUE(usdEntry.curve.get() == usdHandle.get());

    // The one-argument lookup is ambiguous now that USD has two discount curves.
    EXPECT_THROW((void)set.discountCurve("USD"), std::invalid_argument);

    // Collateral selects the exact curve.
    EXPECT_TRUE(set.discountCurve("USD", "USD").get() == usdHandle.get());
    EXPECT_TRUE(set.discountCurve("USD", "EUR").get() == usdEurHandle.get());
    EXPECT_TRUE(set.discountCurve("EUR", "EUR").get() == eurHandle.get());

    EXPECT_THROW(set.add(markets::BuiltCurve{usdKey, markets::CurveRole::Discount, usdHandle}),
                 std::invalid_argument);
    EXPECT_THROW((void)set.discountCurve("JPY"), std::out_of_range);
    EXPECT_THROW((void)set.discountCurve("USD", "JPY"), std::out_of_range);

    // A forecast entry with the same currency is not mistaken for a discount
    // curve because the role is part of the lookup.
    auto forecast = std::make_shared<const markets::SpreadCurve<double>>(
        usd, std::vector<double>{0.0, 1.0, 2.0, 3.0}, std::vector<double>{0.0, 1e-4, 2e-4, 3e-4});
    const markets::CurveKey forecastKey{"USD", markets::CurveRole::Forecast,
                                        datetime::Period(3, datetime::TimeUnit::Months), "USD"};
    set.add(markets::BuiltCurve{forecastKey, markets::CurveRole::Forecast,
                                markets::CurveHandle::make(forecast)});
    EXPECT_TRUE(set.find(forecastKey).role == markets::CurveRole::Forecast);
    EXPECT_TRUE(set.discountCurve("USD", "USD").get() == usdHandle.get());
    EXPECT_TRUE(set.discountCurve("USD", "EUR").get() == usdEurHandle.get());

    EXPECT_THROW(set.add(markets::BuiltCurve{forecastKey, markets::CurveRole::Forecast, nullptr}),
                 std::invalid_argument);
}

TEST(CurveComposition, compounding) {
    const DiscountCurve<double> curve = makeLogDiscountCurve();
    const double tau = 1.0;
    CHECK_CLOSE("compound factor", curve.compoundingFactor(1.0, 2.0),
                curve.discount(1.0) / curve.discount(2.0), 1e-15);
    CHECK_CLOSE("compounded rate", curve.compoundedRate(1.0, 2.0, tau), (std::exp(0.04) - 1.0),
                1e-14);

    const std::vector<double> boundaries{0.0, 0.25, 0.5, 0.75, 1.0};
    double manual = 0.0;
    for (std::size_t k = 1; k < boundaries.size(); ++k) {
        manual += (curve.discount(boundaries[k - 1]) / curve.discount(boundaries[k]) - 1.0) / 0.25;
    }
    CHECK_CLOSE("averaged rate", curve.averagedRate(boundaries), manual / 4.0, 1e-15);

    // Daily compounding telescopes to the compound factor.
    double product = 1.0;
    for (std::size_t k = 1; k < boundaries.size(); ++k) {
        const double simple =
            (curve.discount(boundaries[k - 1]) / curve.discount(boundaries[k]) - 1.0) / 0.25;
        product *= 1.0 + simple * 0.25;
    }
    CHECK_CLOSE("compounding telescopes", product, curve.compoundingFactor(0.0, 1.0), 1e-15);
}

TEST(CurveBootstrap, basisSpreadRecovery) {
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
        CHECK_CLOSE("recovered basis spread", curve.spreadNodes().zeros()[i + 1], targetSpreads[i],
                    1e-11);
        CHECK_CLOSE("basis reprice",
                    markets::impliedBasisSpread(curve, pillars[i], reference, zeroDayCounter),
                    pillars[i].spread, 1e-11);
    }
}

TEST(CurveInterpolation, monotoneCubic) {
    const std::vector<double> times{0.0, 1.0, 2.0, 3.0, 4.0};
    const std::vector<double> zeros{0.0, 0.03, 0.033, 0.036, 0.04};
    const DiscountCurve<double> curve(times, zeros, InterpolationSpace::Zero,
                                      InterpolationScheme::MonotoneCubic);
    for (std::size_t i = 0; i < curve.size(); ++i) {
        CHECK_CLOSE("monotone node", curve.zero(times[i]), zeros[i], 1e-14);
    }
    // No overshoot on monotone data, and the derivative keeps the sign.
    double previous = curve.zero(0.0);
    for (double t = 0.05; t <= 4.0; t += 0.05) {
        const double current = curve.zero(t);
        EXPECT_TRUE(current >= previous - 1e-14);
        previous = current;
    }
    std::vector<double> weights;
    EXPECT_THROW(curve.zeroNodeWeights(2.5, weights), std::invalid_argument);
}

TEST(CurveInterpolation, parametricCurve) {
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
            CHECK_CLOSE("NS fit", fitted.zero(t), exact.zero(t), 1e-5);
        }
        CHECK_CLOSE("NS discount", fitted.discount(5.0), exact.discount(5.0), 1e-5);
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
            CHECK_CLOSE("Svensson fit", fitted.zero(t), exact.zero(t), 1e-5);
        }
    }
}

TEST(CurveInterpolation, mixedLinearCubic) {
    const std::vector<double> times{0.0, 1.0, 2.0, 3.0, 4.0};
    const std::vector<double> zeros{0.0, 0.03, 0.033, 0.036, 0.04};
    const DiscountCurve<double> mixed(times, zeros, InterpolationSpace::Zero,
                                      InterpolationScheme::MixedLinearCubic, 0.0, 2);
    const DiscountCurve<double> akima(times, zeros, InterpolationSpace::Zero,
                                      InterpolationScheme::Akima);
    // Left of the switch: piecewise linear.
    CHECK_CLOSE("mixed linear segment", mixed.zero(0.5), 0.5 * (zeros[0] + zeros[1]), 1e-14);
    CHECK_CLOSE("mixed linear segment 2", mixed.zero(1.5), 0.5 * (zeros[1] + zeros[2]), 1e-14);
    // Right of the switch: identical to Akima.
    for (const double t : {2.2, 2.7, 3.5, 3.9}) {
        CHECK_CLOSE("mixed cubic segment", mixed.zero(t), akima.zero(t), 1e-14);
    }
}

/// Extrapolated node weights must be the scheme-consistent terminal-slope
/// derivative, checked against a central-difference reconstruction for every
/// scheme (including the pinned Hermite schemes).
TEST(CurveRiskWeights, extrapolatedRiskWeights) {
    const std::vector<double> times{0.0, 1.0, 2.0, 3.0, 4.0};
    const std::vector<double> zeros{0.0, 0.03, 0.0335, 0.036, 0.04};
    const double t = 4.0 + 2.0 / 365.0;
    const struct {
        InterpolationSpace space;
        InterpolationScheme scheme;
        double tension;
        int switchIndex;
    } cases[] = {
        {InterpolationSpace::Zero, InterpolationScheme::Linear, 0.0, 1},
        {InterpolationSpace::Zero, InterpolationScheme::Akima, 0.0, 1},
        {InterpolationSpace::Zero, InterpolationScheme::TensionSpline, 8.0, 1},
        {InterpolationSpace::Zero, InterpolationScheme::MonotoneCubic, 0.0, 1},
        {InterpolationSpace::Zero, InterpolationScheme::HymanSpline, 0.0, 1},
        {InterpolationSpace::Zero, InterpolationScheme::MixedLinearCubic, 0.0, 2},
        {InterpolationSpace::LogDiscount, InterpolationScheme::Linear, 0.0, 1},
        {InterpolationSpace::LogDiscount, InterpolationScheme::Akima, 0.0, 1},
        {InterpolationSpace::LogDiscount, InterpolationScheme::TensionSpline, 8.0, 1},
        {InterpolationSpace::LogDiscount, InterpolationScheme::MonotoneCubic, 0.0, 1},
        {InterpolationSpace::LogDiscount, InterpolationScheme::HymanSpline, 0.0, 1},
        {InterpolationSpace::LogDiscount, InterpolationScheme::MixedLinearCubic, 0.0, 3},
    };
    for (const auto& c : cases) {
        const DiscountCurve<double> curve(times, zeros, c.space, c.scheme, c.tension,
                                          c.switchIndex);
        std::vector<double> weights;
        curve.zeroNodeWeights(t, weights);
        const double epsilon = 1e-6;
        for (std::size_t j = 1; j < zeros.size(); ++j) {
            std::vector<double> plus = zeros;
            std::vector<double> minus = zeros;
            plus[j] += epsilon;
            minus[j] -= epsilon;
            const DiscountCurve<double> up(times, plus, c.space, c.scheme, c.tension,
                                           c.switchIndex);
            const DiscountCurve<double> down(times, minus, c.space, c.scheme, c.tension,
                                             c.switchIndex);
            const double fd = (up.zero(t) - down.zero(t)) / (2.0 * epsilon);
            CHECK_CLOSE("extrapolated risk weight", weights[j], fd, 1e-7);
        }
    }
}

/// Tied adjacent secants must not inflate the analytic in-range risk weights:
/// the coefficient Jacobian uses the same standard Akima weights as the
/// primal, so `zeroNodeWeights` matches central differences on a near-flat
/// Zero-space node set. The difference step stays small because it straddles
/// the tie kink, which adds an O(step) term.
TEST(CurveRiskWeights, tiedSecantRiskWeights) {
    const std::vector<double> times{0.0, 1.0, 2.0, 3.0};
    const std::vector<double> zeros{0.0, 0.030, 0.033, 0.036};
    for (const auto scheme : {InterpolationScheme::Akima, InterpolationScheme::MixedLinearCubic}) {
        const DiscountCurve<double> curve(times, zeros, InterpolationSpace::Zero, scheme);
        const double epsilon = 1e-8;
        for (const double t : {0.5, 1.5, 2.5}) {
            std::vector<double> weights;
            curve.zeroNodeWeights(t, weights);
            for (std::size_t j = 1; j < zeros.size(); ++j) {
                std::vector<double> plus = zeros;
                std::vector<double> minus = zeros;
                plus[j] += epsilon;
                minus[j] -= epsilon;
                const DiscountCurve<double> up(times, plus, InterpolationSpace::Zero, scheme);
                const DiscountCurve<double> down(times, minus, InterpolationSpace::Zero, scheme);
                const double fd = (up.zero(t) - down.zero(t)) / (2.0 * epsilon);
                CHECK_CLOSE("tied-secant risk weight", weights[j], fd, 1e-6);
            }
        }
    }
}

TEST(TurnOverlay, turnOverlay) {
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
    CHECK_CLOSE("turn zero jump", after - before, amplitude * turnTime / 2.5, 1e-15);
    // DF ratio across the turn is the base carry times exp(-d tTurn).
    CHECK_CLOSE("turn DF multiplier", overlay.discount(2.5) / overlay.discount(1.5),
                std::exp(-0.03 * 1.0) * std::exp(-amplitude * turnTime), 1e-14);
    // Before the turn the overlay equals the base.
    CHECK_CLOSE("turn before base", overlay.discount(1.5), base->discount(1.5), 1e-15);
    EXPECT_THROW((void)markets::TurnOverlay<double>(nullptr, {}), std::invalid_argument);
}

TEST(TurnOverlay, turnOverlayNodeProvider) {
    static_assert(markets::CurveNodeProvider<markets::TurnOverlay<double>>);
    static_assert(
        markets::CurveNodeProvider<markets::TurnOverlay<double, markets::SpreadCurve<double>>>);

    auto base = std::make_shared<const DiscountCurve<double>>(makeLogDiscountCurve());
    const std::vector<markets::TurnOverlay<double>::Turn> turns{{0.5, 4.0e-4}, {2.0, 1.0e-3}};
    const std::vector<markets::TurnOverlay<double>::Bump> bumps{{1.0, 1.25, 5.0e-4}};
    const markets::TurnOverlay<double> overlay(base, turns, bumps);

    // Native grid: reserved fixed origin, then turn amplitudes, then bumps.
    EXPECT_TRUE(overlay.size() == 4);
    EXPECT_TRUE(overlay.times() == std::vector<double>({0.0, 0.5, 2.0, 1.0}));
    EXPECT_TRUE(overlay.zeroDayCounter().convention() == datetime::DayCount::Actual365Fixed);

    // The analytic weights are the amplitude sensitivities: check them against
    // central differences at points before, inside and after every parameter.
    const auto shiftedZero = [&](std::size_t node, double delta, double t) {
        std::vector<markets::TurnOverlay<double>::Turn> shiftedTurns = turns;
        std::vector<markets::TurnOverlay<double>::Bump> shiftedBumps = bumps;
        if (node < 1 + turns.size()) {
            shiftedTurns[node - 1].second += delta;
        } else {
            shiftedBumps[node - 1 - turns.size()].amplitude += delta;
        }
        const markets::TurnOverlay<double> shifted(base, shiftedTurns, shiftedBumps);
        return shifted.zero(t);
    };
    const double step = 1e-7;
    for (const double t : {-0.5, 0.0, 0.25, 0.5, 0.75, 1.0, 1.1, 1.25, 1.5, 2.0, 2.5, 4.0}) {
        std::vector<double> weights;
        overlay.zeroNodeWeights(t, weights);
        EXPECT_TRUE(weights.size() == overlay.size());
        EXPECT_TRUE(weights[0] == 0.0);
        for (std::size_t node = 1; node < weights.size(); ++node) {
            const double fd =
                (shiftedZero(node, step, t) - shiftedZero(node, -step, t)) / (2.0 * step);
            CHECK_CLOSE("turn overlay amplitude weight", weights[node], fd, 1e-6);
        }
    }

    // The type-erased handle path (`buildStack`'s parent resolution) exposes
    // the same parameter grid and its own pricing surface.
    const markets::CurveHandle::Ptr handle =
        markets::CurveHandle::make(std::make_shared<const markets::TurnOverlay<double>>(overlay));
    EXPECT_TRUE(handle->size() == overlay.size());
    EXPECT_TRUE(handle->times() == overlay.times());
    EXPECT_TRUE(handle->zeroDayCounter().convention() == datetime::DayCount::Actual365Fixed);
    std::vector<double> handleWeights;
    std::vector<double> overlayWeights;
    handle->zeroNodeWeights(1.1, handleWeights);
    overlay.zeroNodeWeights(1.1, overlayWeights);
    EXPECT_TRUE(handleWeights == overlayWeights);

    // A forecast curve bootstrapped over the overlay as its parent provider
    // reprices its synthetic FRA exactly.
    const datetime::Date reference(2026, 9, 29);
    const datetime::DayCounter zeroDc(datetime::DayCount::Actual365Fixed);
    const datetime::DayCounter act360(datetime::DayCount::Actual360);
    const datetime::Calendar calendar = datetime::Calendar::noHolidays();
    const DiscountCurve<double> dateBase(reference,
                                         {reference.plusMonths(6), reference.plusYears(1),
                                          reference.plusYears(2), reference.plusYears(3)},
                                         zeroDc, {0.02, 0.025, 0.03, 0.035},
                                         InterpolationSpace::LogDiscount,
                                         InterpolationScheme::Linear);
    const auto dateBaseShared = std::make_shared<const DiscountCurve<double>>(dateBase);
    const double turnTime = datetime::yearFraction(reference, reference.plusYears(1), zeroDc);
    const double windowBegin = datetime::yearFraction(reference, reference.plusMonths(11), zeroDc);
    const double windowEnd = datetime::yearFraction(reference, reference.plusMonths(13), zeroDc);
    const auto pricedOverlay = std::make_shared<const markets::TurnOverlay<double>>(
        dateBaseShared, std::vector<markets::TurnOverlay<double>::Turn>{{turnTime, 1.0e-3}},
        std::vector<markets::TurnOverlay<double>::Bump>{{windowBegin, windowEnd, 5.0e-4}});

    markets::ForecastPillar pillar;
    pillar.kind = markets::ForecastPillar::Kind::Fra;
    pillar.start = reference.plusMonths(10);
    pillar.maturity = reference.plusMonths(14);
    pillar.calendar = calendar;
    pillar.quoteDayCounter = act360;
    const double startTime = datetime::yearFraction(reference, pillar.start, zeroDc);
    const double endTime = datetime::yearFraction(reference, pillar.maturity, zeroDc);
    const double tau = datetime::yearFraction(pillar.start, pillar.maturity, act360);
    pillar.quote =
        (pricedOverlay->discount(startTime) / pricedOverlay->discount(endTime) - 1.0) / tau;
    const markets::SpreadCurve<double, markets::TurnOverlay<double>> forecast =
        markets::bootstrapForecastCurve(pricedOverlay, nullptr, reference, zeroDc,
                                        markets::InterpolationScheme::Linear, {pillar});
    CHECK_CLOSE("overlay parent bootstrap reprice",
                markets::impliedSimpleForward(forecast, pillar, reference, zeroDc), pillar.quote,
                1e-10);
}

TEST(DiscountCurve, nonAct365ZeroClock) {
    const datetime::Date reference(2026, 9, 29);
    const datetime::Date pillar = reference.plusYears(1);
    const datetime::DayCounter act360(datetime::DayCount::Actual360);
    const DiscountCurve<double> curve(reference, {pillar}, act360, {0.04},
                                      InterpolationSpace::LogDiscount, InterpolationScheme::Linear);
    const double expected = datetime::yearFraction(reference, pillar, act360);
    CHECK_CLOSE("ACT/360 zero clock", curve.times()[1], expected, 1e-15);
    EXPECT_TRUE(std::abs(expected - datetime::yearFraction(
                                        reference, pillar,
                                        datetime::DayCounter(datetime::DayCount::Actual365Fixed))) >
                1e-4);
}

TEST(CurveBootstrap, schemeSpaceMatrix) {
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
            CHECK_CLOSE("matrix reprice", markets::impliedQuote(pillars[i], reference, curve),
                        pillars[i].quote, 1e-10);
            CHECK_CLOSE("matrix recovery", curve.zeros()[i + 1], targetZeros[i + 1], 5e-9);
        }
        // Continuity of zeros across knot boundaries, finite forwards, and
        // monotone decreasing discounts.
        double previousDf = curve.discount(0.0);
        std::vector<double> weights;
        for (double t = 0.05; t <= 5.0; t += 0.05) {
            const double df = curve.discount(t);
            EXPECT_TRUE(df > 0.0);
            EXPECT_TRUE(df <= previousDf + 1e-12);
            previousDf = df;
            const double left = curve.zero(t - 1e-9);
            const double right = curve.zero(t + 1e-9);
            CHECK_CLOSE("matrix continuity", left, right, 1e-5);
        }
        if (c.weights) {
            // Partition holds over ALL nodes (including the fixed t=0 node)
            // for global schemes; zeroNodeWeights drops the fixed node.
            curve.spaceValueWeights(2.5, weights);
            double total = 0.0;
            for (const double w : weights) {
                total += w;
            }
            CHECK_CLOSE("matrix weights partition", total, 1.0, 1e-10);
        }
        if (c.space == InterpolationSpace::LogDiscount && c.scheme == InterpolationScheme::Linear) {
            CHECK_CLOSE("log-linear flat forward", curve.forward(1.1, 1.2), curve.forward(1.3, 1.4),
                        1e-14);
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
            CHECK_CLOSE("zero extrapolation slope", extension, delta * inside / 1e-6, 5e-6);
        }
        if (c.space == InterpolationSpace::LogDiscount) {
            const double knot = curve.times().back();
            const double beyond = curve.forward(knot, knot + 0.5);
            CHECK_CLOSE("flat forward extension", curve.forward(knot + 0.5, knot + 1.0), beyond,
                        1e-12);
            CHECK_CLOSE("flat forward slope", curve.forward(knot - 1e-6, knot), beyond, 5e-6);
        }
        // Forwards finite and free of NaN across the grid.
        for (double t = 0.1; t < 5.0; t += 0.1) {
            const double f = curve.forward(t, t + 0.1);
            EXPECT_TRUE(quantape::util::isFiniteBitwise(f));
        }
    }
}

TEST(DiscountCurve, errors) {
    EXPECT_THROW((void)DiscountCurve<double>(std::vector<double>{}, std::vector<double>{}),
                 std::invalid_argument);
    EXPECT_THROW((void)DiscountCurve<double>(std::vector<double>{0.0, 2.0, 1.0},
                                             std::vector<double>{0.0, 0.03, 0.04}),
                 std::invalid_argument);
}

TEST(CurveBootstrap, bootstrapRecovery) {
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
    EXPECT_TRUE(curve.size() == pillars.size() + 1);
    EXPECT_TRUE(curve.discount(0.0) == 1.0);

    for (std::size_t i = 0; i < pillars.size(); ++i) {
        CHECK_CLOSE("recovered zero", curve.zeros()[i + 1], targetZeros[i], 1e-11);
        CHECK_CLOSE("reprice", markets::impliedQuote(pillars[i], reference, curve),
                    pillars[i].quote, 1e-11);
    }
}

TEST(CurveBootstrap, oisPaymentLag) {
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
        CHECK_CLOSE("OIS payment-lag reprice", markets::impliedQuote(pillar, reference, curve),
                    pillar.quote, 1e-11);
    }
}

TEST(CurveBootstrap, oisBusinessDayConvention) {
    const datetime::Date reference(2026, 9, 29);
    const datetime::DayCounter zeroDayCounter(datetime::DayCount::Actual365Fixed);
    const datetime::Calendar calendar = datetime::Calendar::weekendsOnly();
    const datetime::Date maturity(2027, 7, 31); // Saturday
    EXPECT_TRUE(maturity.weekday() == datetime::Weekday::Saturday);
    const datetime::Schedule following(
        reference, maturity, datetime::Period(1, datetime::TimeUnit::Years), calendar,
        datetime::BusinessDayConvention::Following, datetime::DateGeneration::Forward, false);
    const datetime::Schedule modified(reference, maturity,
                                      datetime::Period(1, datetime::TimeUnit::Years), calendar,
                                      datetime::BusinessDayConvention::ModifiedFollowing,
                                      datetime::DateGeneration::Forward, false);
    EXPECT_TRUE(following.endDate() == datetime::Date(2027, 8, 2));
    EXPECT_TRUE(modified.endDate() == datetime::Date(2027, 7, 30));

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
        CHECK_CLOSE("OIS convention reprice", markets::impliedQuote(swap, reference, curve),
                    swap.quote, 1e-11);
    }
}

TEST(CurveBootstrap, pillarMaturityAdjustment) {
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
    CHECK_CLOSE("adjusted deposit node time", curve.times().back(),
                datetime::yearFraction(reference, friday, zeroDayCounter), 1e-15);
    CHECK_CLOSE("adjusted deposit reprice", markets::impliedQuote(deposit, reference, curve),
                deposit.quote, 1e-11);

    CurvePillar raw = deposit;
    raw.businessDayConvention = datetime::BusinessDayConvention::Unadjusted;
    raw.quote = markets::impliedQuote(raw, reference, target);
    const DiscountCurve<double> rawCurve =
        markets::bootstrapDiscountCurve(reference, zeroDayCounter, InterpolationSpace::LogDiscount,
                                        InterpolationScheme::Linear, std::vector<CurvePillar>{raw});
    CHECK_CLOSE("unadjusted deposit node time", rawCurve.times().back(),
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
    CHECK_CLOSE("adjusted FRA reprice", markets::impliedQuote(fra, reference, fraCurve), fra.quote,
                1e-11);
}

TEST(CurveBootstrap, impliedQuoteUnknownKind) {
    const DiscountCurve<double> curve = makeLogDiscountCurve();
    const datetime::Date reference(2026, 9, 29);
    CurvePillar pillar;
    pillar.kind = static_cast<PillarKind>(42);
    pillar.maturity = datetime::Date(2027, 9, 29);
    EXPECT_THROW((void)markets::impliedQuote(pillar, reference, curve), std::invalid_argument);
}
