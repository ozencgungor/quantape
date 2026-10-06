/**
 * @file test_curve_ad.cpp
 * @brief Curve-stack automatic differentiation gates
 *
 * Curve evaluation is templated on the scalar type, so `DiscountCurve<var>`
 * and friends must produce exact reverse-mode gradients. Bootstrap
 * derivatives flow through the solver layer (`BrentSolver<var>` with the
 * implicit-function-theorem polish), which is gated here on a one-pillar
 * deposit refit. The risk-table case is the only writer of the portfolio CSV:
 * with no env override it lands under the per-test scratch directory and is
 * never a shared path.
 */

#include "quantape/math/StanMath.h"

#include "quantape/markets/Curves/CurveBuilder.h"
#include "quantape/markets/Curves/CurveOnGrid.h"
#include "quantape/markets/Curves/CurveRiskReport.h"
#include "quantape/markets/Curves/DiscountCurve.h"
#include "quantape/markets/Curves/FraConvexity.h"
#include "quantape/markets/Curves/HullWhiteConvexity.h"
#include "quantape/markets/Curves/SpreadCurve.h"
#include "quantape/markets/Curves/StackRisk.h"
#include "quantape/markets/Curves/TurnOverlay.h"
#include "quantape/markets/Curves/TurnOverlayRisk.h"
#include "quantape/math/Interpolations/HymanSplineInterpolation.h"
#include "quantape/math/Interpolations/MonotoneCubicInterpolation.h"
#include "quantape/math/Solvers/BrentSolver.h"
#include "quantape/math/Solvers/SolverStanPrimitives.h"

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <memory>
#include <string>
#include <vector>

#include "support/GtestSupport.h"
#include "support/StanTapeFixture.h"
#include "support/fixtures/PaperEurCurves.h"

using namespace quantape;

namespace {

using namespace quantape::tests::paper_eur;

using markets::DiscountCurve;
using markets::InterpolationScheme;
using markets::InterpolationSpace;
using stan::math::var;

// Mixed-scalar spread parents are refused by the class guard; the predicate
// behind that static_assert must report the mismatch at compile time.
static_assert(markets::detail::SpreadParentScalar<var, DiscountCurve<var>>);
static_assert(markets::detail::SpreadParentScalar<double, DiscountCurve<double>>);
static_assert(!markets::detail::SpreadParentScalar<double, DiscountCurve<var>>);

/// Deterministic net quote delta of the portfolio risk table, asserted after
/// the table is written. The 1M curve now carries the 2Y-60Y long end and the
/// portfolio its five-year 1M leg, which lifts the total from the
/// short-end-only 4.96787843279.
constexpr double kRiskTableTotal = 4.97251004992;

void testDiscountCurveVarGradient(InterpolationSpace space, InterpolationScheme scheme,
                                  double tension = 0.0) {
    const std::vector<double> times{0.0, 1.0, 2.0, 3.0};
    const std::vector<double> baseZeros{0.0, 0.03, 0.035, 0.04};
    std::vector<var> zeros(baseZeros.begin(), baseZeros.end());
    const double horizon = 1.7;
    var objective = 0.0;
    {
        const DiscountCurve<var> curve(times, zeros, space, scheme, tension);
        objective = curve.discount(horizon) + curve.zero(horizon) + curve.forward(0.5, horizon);
    }
    objective.grad();

    const auto primal = [&](const std::vector<double>& nodeZeros) {
        const DiscountCurve<double> curve(times, nodeZeros, space, scheme, tension);
        return curve.discount(horizon) + curve.zero(horizon) + curve.forward(0.5, horizon);
    };
    const double epsilon = 1e-6;
    for (std::size_t j = 1; j < baseZeros.size(); ++j) {
        std::vector<double> plus = baseZeros;
        std::vector<double> minus = baseZeros;
        plus[j] += epsilon;
        minus[j] -= epsilon;
        const double fd = (primal(plus) - primal(minus)) / (2.0 * epsilon);
        CHECK_CLOSE("curve var gradient", zeros[j].adj(), fd, 1e-6);
    }
}

void testMonotoneCubicVarValue() {
    const std::vector<double> times{0.0, 1.0, 2.0, 3.0};
    const std::vector<double> nodeZeros{0.0, 0.03, 0.045, 0.04};
    std::vector<var> zeros(nodeZeros.begin(), nodeZeros.end());
    const DiscountCurve<var> curve(times, zeros, InterpolationSpace::LogDiscount,
                                   InterpolationScheme::MonotoneCubic);
    const DiscountCurve<double> reference(times, nodeZeros, InterpolationSpace::LogDiscount,
                                          InterpolationScheme::MonotoneCubic);
    CHECK_CLOSE("monotone var value", curve.discount(1.5).val(), reference.discount(1.5), 1e-12);
}

/// The risk weights of an AD curve stay primal-pinned: the coefficient
/// Jacobian is built once from the primal node values, so the AD curve returns
/// the same weights as the double curve.
void testVarRiskWeightsMatchDouble() {
    const std::vector<double> times{0.0, 1.0, 2.0, 3.0, 4.0};
    const std::vector<double> nodeZeros{0.0, 0.03, 0.0335, 0.036, 0.04};
    const struct {
        InterpolationScheme scheme;
        double tension;
        int switchIndex;
    } cases[] = {
        {InterpolationScheme::Linear, 0.0, 1},
        {InterpolationScheme::Akima, 0.0, 1},
        {InterpolationScheme::TensionSpline, 8.0, 1},
        {InterpolationScheme::MixedLinearCubic, 0.0, 2},
    };
    for (const auto& c : cases) {
        std::vector<var> zeros(nodeZeros.begin(), nodeZeros.end());
        const DiscountCurve<var> adCurve(times, zeros, InterpolationSpace::LogDiscount, c.scheme,
                                         c.tension, c.switchIndex);
        const DiscountCurve<double> doubleCurve(times, nodeZeros, InterpolationSpace::LogDiscount,
                                                c.scheme, c.tension, c.switchIndex);
        for (const double t : {0.37, 1.5, 3.6}) {
            std::vector<double> adWeights;
            std::vector<double> doubleWeights;
            adCurve.spaceValueWeights(t, adWeights);
            doubleCurve.spaceValueWeights(t, doubleWeights);
            for (std::size_t i = 0; i < adWeights.size(); ++i) {
                CHECK_CLOSE("ad/double space weights", adWeights[i], doubleWeights[i], 1e-15);
            }
        }
        std::vector<double> adWeights;
        std::vector<double> doubleWeights;
        adCurve.zeroNodeWeights(5.3, adWeights);
        doubleCurve.zeroNodeWeights(5.3, doubleWeights);
        for (std::size_t i = 0; i < adWeights.size(); ++i) {
            CHECK_CLOSE("ad/double extrapolated weights", adWeights[i], doubleWeights[i], 1e-15);
        }
    }
}

void testHymanSplineCurveVar() {
    const std::vector<double> times{0.0, 1.0, 2.0, 3.0, 4.0};
    const std::vector<double> baseZeros{0.0, 0.028, 0.033, 0.036, 0.04};
    const double horizon = 2.3;

    std::vector<var> zeros(baseZeros.begin(), baseZeros.end());
    var objective = 0.0;
    {
        const DiscountCurve<var> curve(times, zeros, InterpolationSpace::LogDiscount,
                                       InterpolationScheme::HymanSpline);
        objective = curve.discount(horizon) + curve.zero(horizon) + curve.forward(1.0, horizon);
    }
    objective.grad();

    const DiscountCurve<double> reference(times, baseZeros, InterpolationSpace::LogDiscount,
                                          InterpolationScheme::HymanSpline);
    CHECK_CLOSE("hyman curve var value", objective.val(),
                reference.discount(horizon) + reference.zero(horizon) +
                    reference.forward(1.0, horizon),
                1e-12);

    // The AD branch holds the Hyman slopes fixed, so the central-difference
    // reference evaluates the Hermite basis with the primal slopes and
    // perturbed node values.
    std::vector<double> baseValues(times.size());
    for (std::size_t i = 0; i < times.size(); ++i) {
        baseValues[i] = baseZeros[i] * times[i];
    }
    const math::HymanSplineInterpolation<double> pinned(times, baseValues);
    const std::vector<double>& slopes = pinned.slopes();
    const auto pinnedSpaceValue = [&](const std::vector<double>& values, double t) {
        std::size_t i = 0;
        while (i + 2 < times.size() && t >= times[i + 1]) {
            ++i;
        }
        const double h = times[i + 1] - times[i];
        const double u = (t - times[i]) / h;
        const double u2 = u * u;
        const double u3 = u2 * u;
        const double h00 = 2.0 * u3 - 3.0 * u2 + 1.0;
        const double h10 = u3 - 2.0 * u2 + u;
        const double h01 = -2.0 * u3 + 3.0 * u2;
        const double h11 = u3 - u2;
        return h00 * values[i] + h10 * h * slopes[i] + h01 * values[i + 1] +
               h11 * h * slopes[i + 1];
    };
    const auto pinnedObjective = [&](const std::vector<double>& nodeZeros) {
        std::vector<double> values(times.size());
        for (std::size_t i = 0; i < times.size(); ++i) {
            values[i] = nodeZeros[i] * times[i];
        }
        const double atHorizon = pinnedSpaceValue(values, horizon);
        const double atOne = pinnedSpaceValue(values, 1.0);
        return std::exp(-atHorizon) + atHorizon / horizon + (atHorizon - atOne) / (horizon - 1.0);
    };
    const double epsilon = 1e-6;
    for (std::size_t j = 1; j < baseZeros.size(); ++j) {
        std::vector<double> plus = baseZeros;
        std::vector<double> minus = baseZeros;
        plus[j] += epsilon;
        minus[j] -= epsilon;
        const double fd = (pinnedObjective(plus) - pinnedObjective(minus)) / (2.0 * epsilon);
        CHECK_CLOSE("hyman curve var gradient", zeros[j].adj(), fd, 1e-6);
    }

    // Date-based curve: impliedQuote agrees with the double path.
    const datetime::Date referenceDate(2026, 9, 29);
    const std::vector<datetime::Date> pillarDates{
        referenceDate.plusYears(1), referenceDate.plusYears(2), referenceDate.plusYears(3)};
    const std::vector<double> depositZeros{0.028, 0.033, 0.038};
    const datetime::DayCounter zeroDc(datetime::DayCount::Actual365Fixed);
    std::vector<var> depositAd(depositZeros.begin(), depositZeros.end());
    markets::CurvePillar deposit;
    deposit.kind = markets::PillarKind::Deposit;
    deposit.maturity = referenceDate.plusYears(2);
    deposit.quoteDayCounter = datetime::DayCounter(datetime::DayCount::Actual360);
    deposit.calendar = datetime::Calendar::noHolidays();
    var quote = 0.0;
    {
        const DiscountCurve<var> curve(referenceDate, pillarDates, zeroDc, depositAd,
                                       InterpolationSpace::LogDiscount,
                                       InterpolationScheme::HymanSpline);
        quote = markets::impliedQuote(deposit, referenceDate, curve);
    }
    quote.grad();
    const DiscountCurve<double> dateReference(referenceDate, pillarDates, zeroDc, depositZeros,
                                              InterpolationSpace::LogDiscount,
                                              InterpolationScheme::HymanSpline);
    CHECK_CLOSE("hyman curve var impliedQuote", quote.val(),
                markets::impliedQuote(deposit, referenceDate, dateReference), 1e-12);
}

void testSpreadCurveVarGradient() {
    const std::vector<double> times{0.0, 1.0, 2.0};
    const std::vector<double> baseParent{0.0, 0.03, 0.035};
    const std::vector<double> baseSpreads{0.0, 0.001, 0.0015};
    std::vector<var> parentZeros(baseParent.begin(), baseParent.end());
    std::vector<var> spreads(baseSpreads.begin(), baseSpreads.end());
    auto parent = std::make_shared<const DiscountCurve<var>>(
        times, parentZeros, InterpolationSpace::LogDiscount, InterpolationScheme::Linear);
    var objective = 0.0;
    {
        const markets::SpreadCurve<var> curve(parent, times, spreads, InterpolationScheme::Linear);
        objective = curve.discount(1.5) + curve.zero(1.5);
    }
    objective.grad();

    const auto primal = [&](const std::vector<double>& parentNodes,
                            const std::vector<double>& spreadNodes) {
        auto base = std::make_shared<const DiscountCurve<double>>(
            times, parentNodes, InterpolationSpace::LogDiscount, InterpolationScheme::Linear);
        const markets::SpreadCurve<double> curve(base, times, spreadNodes,
                                                 InterpolationScheme::Linear);
        return curve.discount(1.5) + curve.zero(1.5);
    };
    const double epsilon = 1e-6;
    for (std::size_t j = 1; j < baseParent.size(); ++j) {
        std::vector<double> plus = baseParent;
        std::vector<double> minus = baseParent;
        plus[j] += epsilon;
        minus[j] -= epsilon;
        const double fd =
            (primal(plus, baseSpreads) - primal(minus, baseSpreads)) / (2.0 * epsilon);
        CHECK_CLOSE("spread parent var gradient", parentZeros[j].adj(), fd, 1e-6);
    }
    for (std::size_t j = 1; j < baseSpreads.size(); ++j) {
        std::vector<double> plus = baseSpreads;
        std::vector<double> minus = baseSpreads;
        plus[j] += epsilon;
        minus[j] -= epsilon;
        const double fd = (primal(baseParent, plus) - primal(baseParent, minus)) / (2.0 * epsilon);
        CHECK_CLOSE("spread node var gradient", spreads[j].adj(), fd, 1e-6);
    }
}

/// The Steffen node slopes are pinned from the primal values, so the
/// reverse-mode gradient must match central differences of the Hermite basis
/// evaluated with those pinned slopes.
void testMonotoneCubicPinnedGradient() {
    const std::vector<double> times{0.0, 1.0, 2.0, 3.0, 4.0};
    const std::vector<double> baseZeros{0.0, 0.028, 0.033, 0.036, 0.04};
    const double horizon = 2.3;
    std::vector<var> zeros(baseZeros.begin(), baseZeros.end());
    var objective = 0.0;
    {
        const DiscountCurve<var> curve(times, zeros, InterpolationSpace::LogDiscount,
                                       InterpolationScheme::MonotoneCubic);
        objective = curve.discount(horizon) + curve.zero(horizon) + curve.forward(1.0, horizon);
    }
    objective.grad();

    std::vector<double> baseValues(times.size());
    for (std::size_t i = 0; i < times.size(); ++i) {
        baseValues[i] = baseZeros[i] * times[i];
    }
    const math::MonotoneCubicInterpolation<double> pinned(times, baseValues);
    const std::vector<double>& slopes = pinned.slopes();
    const auto pinnedSpaceValue = [&](const std::vector<double>& values, double t) {
        std::size_t i = 0;
        while (i + 2 < times.size() && t >= times[i + 1]) {
            ++i;
        }
        const double h = times[i + 1] - times[i];
        const double u = (t - times[i]) / h;
        const double u2 = u * u;
        const double u3 = u2 * u;
        const double h00 = 2.0 * u3 - 3.0 * u2 + 1.0;
        const double h10 = u3 - 2.0 * u2 + u;
        const double h01 = -2.0 * u3 + 3.0 * u2;
        const double h11 = u3 - u2;
        return h00 * values[i] + h10 * h * slopes[i] + h01 * values[i + 1] +
               h11 * h * slopes[i + 1];
    };
    const auto pinnedObjective = [&](const std::vector<double>& nodeZeros) {
        std::vector<double> values(times.size());
        for (std::size_t i = 0; i < times.size(); ++i) {
            values[i] = nodeZeros[i] * times[i];
        }
        const double atHorizon = pinnedSpaceValue(values, horizon);
        const double atOne = pinnedSpaceValue(values, 1.0);
        return std::exp(-atHorizon) + atHorizon / horizon + (atHorizon - atOne) / (horizon - 1.0);
    };
    const double epsilon = 1e-6;
    for (std::size_t j = 1; j < baseZeros.size(); ++j) {
        std::vector<double> plus = baseZeros;
        std::vector<double> minus = baseZeros;
        plus[j] += epsilon;
        minus[j] -= epsilon;
        const double fd = (pinnedObjective(plus) - pinnedObjective(minus)) / (2.0 * epsilon);
        CHECK_CLOSE("monotone cubic pinned gradient", zeros[j].adj(), fd, 1e-6);
    }
}

/// Tied adjacent secants (a flat segment beside a steep one) make the
/// standard Akima weights vanish; the tie fallback must keep the reverse-mode
/// gradient finite and consistent with central differences. The difference
/// step is kept small because the finite difference straddles the tie kink,
/// which adds an O(step) term.
void testTiedSecantGradient(InterpolationScheme scheme) {
    const std::vector<double> times{0.0, 1.0, 2.0, 3.0};
    const std::vector<double> baseZeros{0.0, 0.030, 0.033, 0.036};
    std::vector<var> zeros(baseZeros.begin(), baseZeros.end());
    var objective = 0.0;
    {
        const DiscountCurve<var> curve(times, zeros, InterpolationSpace::Zero, scheme);
        objective =
            curve.zero(1.5) + curve.zero(3.6) + curve.discount(1.5) + curve.forward(0.7, 2.4);
    }
    objective.grad();
    const auto primal = [&](const std::vector<double>& nodeZeros) {
        const DiscountCurve<double> curve(times, nodeZeros, InterpolationSpace::Zero, scheme);
        return curve.zero(1.5) + curve.zero(3.6) + curve.discount(1.5) + curve.forward(0.7, 2.4);
    };
    const double epsilon = 1e-8;
    for (std::size_t j = 1; j < baseZeros.size(); ++j) {
        std::vector<double> plus = baseZeros;
        std::vector<double> minus = baseZeros;
        plus[j] += epsilon;
        minus[j] -= epsilon;
        const double fd = (primal(plus) - primal(minus)) / (2.0 * epsilon);
        CHECK_CLOSE("tied-secant curve gradient", zeros[j].adj(), fd, 1e-6);
    }
}

/// The spread curve interpolates its nodes in Zero space, so a near-flat
/// spread node set hits the same tied-secant Akima weights.
void testSpreadCurveTiedSecantGradient() {
    const std::vector<double> times{0.0, 1.0, 2.0, 3.0};
    const std::vector<double> parentZeros{0.0, 0.02, 0.025, 0.03};
    const std::vector<double> baseSpreads{0.0, 0.030, 0.033, 0.036};
    std::vector<var> parentNodes(parentZeros.begin(), parentZeros.end());
    std::vector<var> spreads(baseSpreads.begin(), baseSpreads.end());
    auto parent = std::make_shared<const DiscountCurve<var>>(
        times, parentNodes, InterpolationSpace::LogDiscount, InterpolationScheme::Linear);
    var objective = 0.0;
    {
        const markets::SpreadCurve<var> curve(parent, times, spreads, InterpolationScheme::Akima);
        objective =
            curve.zero(1.5) + curve.zero(3.6) + curve.discount(1.5) + curve.forward(0.7, 2.4);
    }
    objective.grad();
    const auto primal = [&](const std::vector<double>& spreadNodes) {
        auto base = std::make_shared<const DiscountCurve<double>>(
            times, parentZeros, InterpolationSpace::LogDiscount, InterpolationScheme::Linear);
        const markets::SpreadCurve<double> curve(base, times, spreadNodes,
                                                 InterpolationScheme::Akima);
        return curve.zero(1.5) + curve.zero(3.6) + curve.discount(1.5) + curve.forward(0.7, 2.4);
    };
    const double epsilon = 1e-8;
    for (std::size_t j = 1; j < baseSpreads.size(); ++j) {
        std::vector<double> plus = baseSpreads;
        std::vector<double> minus = baseSpreads;
        plus[j] += epsilon;
        minus[j] -= epsilon;
        const double fd = (primal(plus) - primal(minus)) / (2.0 * epsilon);
        CHECK_CLOSE("tied-secant spread gradient", spreads[j].adj(), fd, 1e-6);
    }
}

void testTurnOverlayVarGradient() {
    static_assert(markets::CurveProvider<markets::TurnOverlay<var>, var>);
    static_assert(
        markets::CurveProvider<markets::TurnOverlay<var, markets::SpreadCurve<var>>, var>);
    const std::vector<double> times{0.0, 1.0, 2.0, 3.0};
    const std::vector<double> baseZeros{0.0, 0.02, 0.025, 0.03};
    const std::vector<double> baseSpreads{0.0, 0.001, 0.002, 0.003};
    const std::vector<std::pair<double, double>> turns{{1.5, 1.0e-5}};
    std::vector<var> zeros(baseZeros.begin(), baseZeros.end());
    std::vector<var> spreads(baseSpreads.begin(), baseSpreads.end());
    std::vector<std::pair<double, var>> adTurns;
    adTurns.emplace_back(turns[0].first, turns[0].second);

    var objective = 0.0;
    {
        auto parent = std::make_shared<const DiscountCurve<var>>(
            times, zeros, InterpolationSpace::LogDiscount, InterpolationScheme::Linear);
        auto spread = std::make_shared<const markets::SpreadCurve<var>>(
            parent, times, spreads, InterpolationScheme::Linear);
        const markets::TurnOverlay<var, markets::SpreadCurve<var>> overlay(spread, adTurns,
                                                                           {{1.2, 1.4, 2.0e-5}});
        objective = overlay.zero(2.5) + overlay.discount(2.5) + overlay.forward(1.0, 2.0);
    }
    objective.grad();

    const auto primal = [&](const std::vector<double>& parentNodes,
                            const std::vector<double>& spreadNodes) {
        auto parent = std::make_shared<const DiscountCurve<double>>(
            times, parentNodes, InterpolationSpace::LogDiscount, InterpolationScheme::Linear);
        auto spread = std::make_shared<const markets::SpreadCurve<double>>(
            parent, times, spreadNodes, InterpolationScheme::Linear);
        const markets::TurnOverlay<double, markets::SpreadCurve<double>> overlay(
            spread, turns, {{1.2, 1.4, 2.0e-5}});
        return overlay.zero(2.5) + overlay.discount(2.5) + overlay.forward(1.0, 2.0);
    };
    const double epsilon = 1e-6;
    for (std::size_t j = 1; j < baseZeros.size(); ++j) {
        std::vector<double> plus = baseZeros;
        std::vector<double> minus = baseZeros;
        plus[j] += epsilon;
        minus[j] -= epsilon;
        const double fd =
            (primal(plus, baseSpreads) - primal(minus, baseSpreads)) / (2.0 * epsilon);
        CHECK_CLOSE("turn overlay parent var gradient", zeros[j].adj(), fd, 1e-6);
    }
    for (std::size_t j = 1; j < baseSpreads.size(); ++j) {
        std::vector<double> plus = baseSpreads;
        std::vector<double> minus = baseSpreads;
        plus[j] += epsilon;
        minus[j] -= epsilon;
        const double fd = (primal(baseZeros, plus) - primal(baseZeros, minus)) / (2.0 * epsilon);
        CHECK_CLOSE("turn overlay spread var gradient", spreads[j].adj(), fd, 1e-6);
    }
}

/// Native overlay node weights are the exogenous amplitude sensitivities, so
/// reverse-mode adjoints of `zero(t)` with respect to `var` amplitudes must
/// equal `zeroNodeWeights` on the matching double overlay.
void testTurnOverlayNodeWeightsAd() {
    static_assert(markets::CurveNodeProvider<markets::TurnOverlay<double>>);
    static_assert(markets::CurveProvider<markets::TurnOverlay<var>, var>);
    const std::vector<double> times{0.0, 1.0, 2.0, 3.0};
    const std::vector<double> baseZeros{0.0, 0.02, 0.025, 0.03};
    const double turnTime = 0.5;
    const double bumpBegin = 1.0;
    const double bumpEnd = 1.25;
    const double probe = 2.5;
    var turnAmplitude = 4.0e-4;
    var bumpAmplitude = 5.0e-4;
    var objective = 0.0;
    {
        const auto base = std::make_shared<const DiscountCurve<var>>(
            times, std::vector<var>(baseZeros.begin(), baseZeros.end()),
            InterpolationSpace::LogDiscount, InterpolationScheme::Linear);
        const markets::TurnOverlay<var> overlay(
            base, std::vector<std::pair<double, var>>{{turnTime, turnAmplitude}},
            std::vector<markets::TurnOverlay<var>::Bump>{{bumpBegin, bumpEnd, bumpAmplitude}});
        objective = overlay.zero(probe);
    }
    objective.grad();

    const auto baseDouble = std::make_shared<const DiscountCurve<double>>(
        times, baseZeros, InterpolationSpace::LogDiscount, InterpolationScheme::Linear);
    const markets::TurnOverlay<double> overlayDouble(
        baseDouble, std::vector<std::pair<double, double>>{{turnTime, 4.0e-4}},
        std::vector<markets::TurnOverlay<double>::Bump>{{bumpBegin, bumpEnd, 5.0e-4}});
    std::vector<double> weights;
    overlayDouble.zeroNodeWeights(probe, weights);
    EXPECT_EQ(weights.size(), 3u);
    EXPECT_EQ(weights[0], 0.0);
    CHECK_CLOSE("turn node weight AD", turnAmplitude.adj(), weights[1], 1e-12);
    CHECK_CLOSE("bump node weight AD", bumpAmplitude.adj(), weights[2], 1e-12);
}

/// Exogenous turn risk through the overlay amplitudes: the overlay keeps the
/// amplitudes as AD parameters, so reverse mode returns `dV/d(amplitude)`
/// directly. The gate checks the flat-forward `Bump` (the risk-table
/// representation) and the persistent turn point against closed forms and
/// central differences at the calibrated amplitude.
void testTurnOverlayRiskAdjoint() {
    static_assert(markets::CurveProvider<markets::TurnOverlay<var>, var>);
    const datetime::Date reference(2012, 12, 11);
    const datetime::DayCounter zeroDc(datetime::DayCount::Actual365Fixed);
    const datetime::DayCounter act360(datetime::DayCount::Actual360);
    const datetime::Date turnBegin(2013, 12, 27);
    const datetime::Date turnEnd(2014, 1, 5);
    const datetime::Date crossingStart(2013, 12, 2);
    const datetime::Date crossingEnd(2014, 1, 31);
    const datetime::Date afterStart(2014, 3, 14);
    const datetime::Date afterEnd(2014, 4, 14);
    const double amplitude = 4.0e-4;
    const double tBegin = datetime::yearFraction(reference, turnBegin, zeroDc);
    const double tEnd = datetime::yearFraction(reference, turnEnd, zeroDc);

    const std::vector<double> times{0.0, 1.0, 2.0, 3.0};
    const std::vector<double> baseZeros{0.0, 0.02, 0.025, 0.03};
    const DiscountCurve<double> base(times, baseZeros, InterpolationSpace::LogDiscount,
                                     InterpolationScheme::Linear);
    const DiscountCurve<double> discount(times, baseZeros, InterpolationSpace::LogDiscount,
                                         InterpolationScheme::Linear);
    // One simple floating coupon with the fixed discounting frozen: only the
    // forward references the overlaid curve.
    const auto couponValue = [&](const auto& overlay, const datetime::Date& start,
                                 const datetime::Date& end) {
        const double t1 = datetime::yearFraction(reference, start, zeroDc);
        const double t2 = datetime::yearFraction(reference, end, zeroDc);
        const double tau = datetime::yearFraction(start, end, act360);
        const auto forward = (overlay.discount(t1) / overlay.discount(t2) - 1.0) / tau;
        return discount.discount(t2) * tau * forward;
    };

    const std::vector<std::string> labels{"Turn 2013-12-27"};
    const std::vector<std::pair<double, double>> noTurns{};
    const std::vector<markets::TurnOverlay<double, DiscountCurve<double>>::Bump> bumps{
        {tBegin, tEnd, amplitude}};
    const auto bumpRisk =
        markets::turnOverlayRisk<var>(base, noTurns, bumps, labels, [&](const auto& overlay) {
            return couponValue(overlay, crossingStart, crossingEnd);
        });
    EXPECT_EQ(bumpRisk.size(), 1u);
    EXPECT_EQ(bumpRisk[0].label, "Turn 2013-12-27");

    const double window = datetime::yearFraction(turnBegin, turnEnd, zeroDc);
    const double crossingT1 = datetime::yearFraction(reference, crossingStart, zeroDc);
    const double crossingT2 = datetime::yearFraction(reference, crossingEnd, zeroDc);
    const double ratio = base.discount(crossingT1) / base.discount(crossingT2);
    const double payDiscount = discount.discount(crossingT2);
    // A coupon that fully contains the window: the forward is scaled by
    // exp(amplitude * window), so dV/dA = D_disc(pay) * baseRatio * window *
    // exp(A window).
    const double analytic = payDiscount * ratio * window * std::exp(amplitude * window);
    CHECK_CLOSE("turn bump risk analytic", bumpRisk[0].delta, analytic, 1e-8);

    const auto valueAt = [&](double value) {
        const auto shared = std::make_shared<const DiscountCurve<double>>(base);
        const markets::TurnOverlay<double, DiscountCurve<double>> overlay(shared, {},
                                                                          {{tBegin, tEnd, value}});
        return couponValue(overlay, crossingStart, crossingEnd);
    };
    const double step = 1e-6;
    const double fd = (valueAt(amplitude + step) - valueAt(amplitude - step)) / (2.0 * step);
    CHECK_CLOSE("turn bump risk FD", bumpRisk[0].delta, fd, 1e-6);

    // A coupon entirely after the window: both discount factors carry the same
    // persistent factor, so the forward ratio is unaffected.
    const auto nonSpanning =
        markets::turnOverlayRisk<var>(base, noTurns, bumps, labels, [&](const auto& overlay) {
            return couponValue(overlay, afterStart, afterEnd);
        });
    CHECK_CLOSE("turn bump risk non-spanning", nonSpanning[0].delta, 0.0, 1e-12);

    // Persistent turn point: the factor is exp(-amplitude * tBegin) on every
    // discount factor beyond the turn, so only the coupon end contributes.
    const std::vector<std::pair<double, double>> turns{{tBegin, amplitude}};
    const auto turnRisk =
        markets::turnOverlayRisk<var>(base, turns, {}, labels, [&](const auto& overlay) {
            return couponValue(overlay, crossingStart, crossingEnd);
        });
    const double analyticTurn = payDiscount * ratio * tBegin * std::exp(amplitude * tBegin);
    CHECK_CLOSE("turn point risk analytic", turnRisk[0].delta, analyticTurn, 1e-8);
}

/// The overlay-risk rebound rebuilds the base curve on AD scalars; a date-based
/// double curve must keep its reference date and zero clock so risk functors
/// that read the curve context see the same values, for a plain discount curve
/// and for a spread curve parent chain.
void testTurnOverlayRiskMetadata() {
    const datetime::Date reference(2012, 12, 11);
    const datetime::DayCounter zeroDc(datetime::DayCount::Actual360);
    const std::vector<datetime::Date> pillarDates{reference.plusMonths(6), reference.plusYears(1),
                                                  reference.plusYears(2), reference.plusYears(3)};
    const std::vector<double> zeros{0.01, 0.02, 0.025, 0.03};
    const DiscountCurve<double> base(reference, pillarDates, zeroDc, zeros,
                                     InterpolationSpace::LogDiscount, InterpolationScheme::Linear);
    const auto parent = std::make_shared<const DiscountCurve<double>>(base);
    const markets::SpreadCurve<double> spread(
        parent, base.times(), std::vector<double>{0.0, 0.001, 0.0015, 0.002, 0.0025},
        InterpolationScheme::Linear);
    const std::vector<markets::TurnOverlay<double, DiscountCurve<double>>::Bump> discountBumps{
        {0.5, 0.7, 1.0e-4}};
    const std::vector<markets::TurnOverlay<double, markets::SpreadCurve<double>>::Bump> spreadBumps{
        {0.5, 0.7, 1.0e-4}};
    const std::vector<std::pair<double, double>> noTurns{};
    const std::vector<std::string> labels{"Bump"};

    bool checkedDiscount = false;
    const auto discountRisk = markets::turnOverlayRisk<var>(
        base, noTurns, discountBumps, labels, [&](const auto& overlay) {
            EXPECT_TRUE(overlay.base().referenceDate() == reference);
            EXPECT_TRUE(overlay.base().zeroDayCounter().convention() ==
                        datetime::DayCount::Actual360);
            EXPECT_TRUE(overlay.base().times() == base.times());
            checkedDiscount = true;
            return overlay.discount(1.5);
        });
    EXPECT_TRUE(checkedDiscount);
    EXPECT_EQ(discountRisk.size(), 1u);

    bool checkedSpread = false;
    const auto spreadRisk = markets::turnOverlayRisk<var>(
        spread, noTurns, spreadBumps, labels, [&](const auto& overlay) {
            EXPECT_TRUE(overlay.base().parent().referenceDate() == reference);
            EXPECT_TRUE(overlay.base().zeroDayCounter().convention() ==
                        datetime::DayCount::Actual360);
            EXPECT_TRUE(overlay.base().parent().times() == base.times());
            EXPECT_TRUE(overlay.base().spreadNodes().times() == base.times());
            checkedSpread = true;
            return overlay.discount(1.5);
        });
    EXPECT_TRUE(checkedSpread);
    EXPECT_EQ(spreadRisk.size(), 1u);
}

void testMaterializeVarGradient() {
    const std::vector<double> times{0.0, 1.0, 2.0, 3.0};
    const std::vector<double> baseZeros{0.0, 0.03, 0.035, 0.04};
    const std::vector<double> grid{0.5, 1.0, 1.5, 2.5};
    std::vector<var> zeros(baseZeros.begin(), baseZeros.end());
    var objective = 0.0;
    {
        const DiscountCurve<var> curve(times, zeros, InterpolationSpace::LogDiscount,
                                       InterpolationScheme::Linear);
        const markets::CurveOnGrid<var> materialized = markets::materialize(curve, grid);
        for (std::size_t k = 0; k < materialized.nSteps(); ++k) {
            objective += materialized.discount[k + 1] + materialized.forward[k];
        }
    }
    objective.grad();

    const auto primal = [&](const std::vector<double>& nodeZeros) {
        const DiscountCurve<double> curve(times, nodeZeros, InterpolationSpace::LogDiscount,
                                          InterpolationScheme::Linear);
        const markets::CurveOnGrid<double> materialized = markets::materialize(curve, grid);
        double total = 0.0;
        for (std::size_t k = 0; k < materialized.nSteps(); ++k) {
            total += materialized.discount[k + 1] + materialized.forward[k];
        }
        return total;
    };
    const double epsilon = 1e-6;
    for (std::size_t j = 1; j < baseZeros.size(); ++j) {
        std::vector<double> plus = baseZeros;
        std::vector<double> minus = baseZeros;
        plus[j] += epsilon;
        minus[j] -= epsilon;
        const double fd = (primal(plus) - primal(minus)) / (2.0 * epsilon);
        CHECK_CLOSE("materialize var gradient", zeros[j].adj(), fd, 1e-6);
    }
}

void testDepositBootstrapSolverAd() {
    const double maturity = 2.0;
    const double tau = 2.0;
    const double quoteRoot = 0.031;
    const double zeroRoot = std::log(1.0 + quoteRoot * tau) / maturity;
    var quote = quoteRoot;

    const auto residual = [&](const auto& z) {
        using Scalar = std::decay_t<decltype(z)>;
        const std::vector<Scalar> zeros{Scalar(0.0), z};
        const DiscountCurve<Scalar> curve(std::vector<double>{0.0, maturity}, zeros,
                                          InterpolationSpace::LogDiscount,
                                          InterpolationScheme::Linear);
        const Scalar discount = curve.discount(maturity);
        return (Scalar(1.0) / discount - Scalar(1.0)) / tau - quote;
    };

    math::BrentSolver<var> solver;
    solver.setMaxEvaluations(300);
    var zHat = solver.solve(residual, 1e-12, var(zeroRoot), var(0.0), var(0.1));
    zHat.grad();

    CHECK_CLOSE("bootstrap solver root", zHat.val(), zeroRoot, 1e-9);
    const double dzDq = tau / (maturity * (1.0 + quoteRoot * tau));
    CHECK_CLOSE("bootstrap solver d z/d quote", quote.adj(), dzDq, 1e-9);
}

void testHullWhiteConvexityVarGradient() {
    var sigma = 0.011;
    var meanReversion = 0.045;
    const double expiry = 1.25;
    const double accrual = 0.25;
    var ratio = 1.008;
    var adjustment =
        markets::hullWhiteFuturesAdjustment(sigma, meanReversion, expiry, accrual, ratio);
    adjustment.grad();

    const auto primal = [&](double sigmaValue, double ratioValue) {
        return markets::hullWhiteFuturesAdjustment(sigmaValue, meanReversion.val(), expiry, accrual,
                                                   ratioValue);
    };
    const double epsilon = 1e-6;
    CHECK_CLOSE(
        "hw convexity d/d sigma", sigma.adj(),
        (primal(sigma.val() + epsilon, ratio.val()) - primal(sigma.val() - epsilon, ratio.val())) /
            (2.0 * epsilon),
        1e-9);
    CHECK_CLOSE(
        "hw convexity d/d ratio", ratio.adj(),
        (primal(sigma.val(), ratio.val() + epsilon) - primal(sigma.val(), ratio.val() - epsilon)) /
            (2.0 * epsilon),
        1e-9);
}

void testFraConvexityVarGradient() {
    var sigmaIndex = 0.011;
    var sigmaDiscount = 0.009;
    var correlation = 0.4;
    const double timeToFixing = 1.5;
    var exponent =
        markets::fraConvexityExponent(sigmaIndex, sigmaDiscount, correlation, timeToFixing);
    exponent.grad();
    const double sigma = sigmaIndex.val();
    CHECK_CLOSE("fra exponent d/d sigmaIndex", sigmaIndex.adj(),
                2.0 * sigma * timeToFixing - sigmaDiscount.val() * correlation.val() * timeToFixing,
                1e-12);
    CHECK_CLOSE("fra exponent d/d sigmaDiscount", sigmaDiscount.adj(),
                -sigma * correlation.val() * timeToFixing, 1e-12);
    CHECK_CLOSE("fra exponent d/d correlation", correlation.adj(),
                -sigma * sigmaDiscount.val() * timeToFixing, 1e-12);
}

void testAveragedCompoundedFutureVarValue() {
    const datetime::Date reference(2026, 9, 29);
    const datetime::Date start(2027, 3, 17);
    const datetime::Date maturity(2027, 6, 16);
    const std::vector<double> times{0.0, 0.5, 1.0, 2.0};
    const std::vector<double> baseZeros{0.0, 0.03, 0.033, 0.036};
    std::vector<var> zeros(baseZeros.begin(), baseZeros.end());

    markets::CurvePillar pillar;
    pillar.kind = markets::PillarKind::Future;
    pillar.futureStyle = markets::FutureStyle::Averaged;
    pillar.averagingStyle = markets::AveragingStyle::Compounded;
    pillar.start = start;
    pillar.maturity = maturity;
    pillar.quoteDayCounter = datetime::DayCounter(datetime::DayCount::Actual360);
    pillar.convexityAdjustment = 0.0002;

    var quote = 0.0;
    {
        const DiscountCurve<var> curve(times, zeros, InterpolationSpace::LogDiscount,
                                       InterpolationScheme::Linear);
        quote = markets::impliedQuote(pillar, reference, curve);
    }
    quote.grad();

    const DiscountCurve<double> primalCurve(times, baseZeros, InterpolationSpace::LogDiscount,
                                            InterpolationScheme::Linear);
    CHECK_CLOSE("compounded averaged var value", quote.val(),
                markets::impliedQuote(pillar, reference, primalCurve), 1e-15);

    const double epsilon = 1e-6;
    for (std::size_t j = 1; j < baseZeros.size(); ++j) {
        std::vector<double> plus = baseZeros;
        std::vector<double> minus = baseZeros;
        plus[j] += epsilon;
        minus[j] -= epsilon;
        const DiscountCurve<double> plusCurve(times, plus, InterpolationSpace::LogDiscount,
                                              InterpolationScheme::Linear);
        const DiscountCurve<double> minusCurve(times, minus, InterpolationSpace::LogDiscount,
                                               InterpolationScheme::Linear);
        const double fd = (markets::impliedQuote(pillar, reference, plusCurve) -
                           markets::impliedQuote(pillar, reference, minusCurve)) /
                          (2.0 * epsilon);
        CHECK_CLOSE("compounded averaged var gradient", zeros[j].adj(), fd, 1e-6);
    }
}

void testTurnKnotCurveVarGradient() {
    const datetime::Date reference(2012, 12, 11);
    const datetime::Date anchor(2013, 12, 13);
    const datetime::Date turnBegin(2013, 12, 27);
    const datetime::Date turnEnd(2014, 1, 5);
    const datetime::Date post(2014, 3, 13);
    const datetime::DayCounter zeroDc(datetime::DayCount::Actual365Fixed);
    const datetime::DayCounter act360(datetime::DayCount::Actual360);
    const std::vector<double> nodeTimes{0.0, datetime::yearFraction(reference, anchor, zeroDc),
                                        datetime::yearFraction(reference, turnBegin, zeroDc),
                                        datetime::yearFraction(reference, turnEnd, zeroDc),
                                        datetime::yearFraction(reference, post, zeroDc)};
    const std::vector<double> baseSpreads{0.0, 0.0010, 0.0042, 0.0028, 0.0020};
    const std::vector<double> parentTimes{0.0, 1.0, 2.0};
    const std::vector<double> parentZeros{0.0, 0.0030, 0.0035};

    markets::ForecastPillar pillar;
    pillar.kind = markets::ForecastPillar::Kind::Fra;
    pillar.start = turnBegin;
    pillar.maturity = turnEnd;
    pillar.quoteDayCounter = act360;
    pillar.calendar = datetime::Calendar::noHolidays();

    std::vector<var> spreads(baseSpreads.begin(), baseSpreads.end());
    std::vector<var> parentNodes(parentZeros.begin(), parentZeros.end());
    auto parent = std::make_shared<const DiscountCurve<var>>(
        parentTimes, parentNodes, InterpolationSpace::LogDiscount, InterpolationScheme::Linear);
    var quote = 0.0;
    {
        const markets::SpreadCurve<var> curve(parent, nodeTimes, spreads,
                                              InterpolationScheme::Linear);
        quote = markets::impliedForecastQuote(curve, *parent, pillar, reference, zeroDc);
    }
    quote.grad();

    const auto primal = [&](const std::vector<double>& nodes) {
        auto base = std::make_shared<const DiscountCurve<double>>(
            parentTimes, parentZeros, InterpolationSpace::LogDiscount, InterpolationScheme::Linear);
        const markets::SpreadCurve<double> curve(base, nodeTimes, nodes,
                                                 InterpolationScheme::Linear);
        return markets::impliedForecastQuote(curve, *base, pillar, reference, zeroDc);
    };
    const double epsilon = 1e-6;
    for (std::size_t j = 1; j < baseSpreads.size(); ++j) {
        std::vector<double> plus = baseSpreads;
        std::vector<double> minus = baseSpreads;
        plus[j] += epsilon;
        minus[j] -= epsilon;
        const double fd = (primal(plus) - primal(minus)) / (2.0 * epsilon);
        CHECK_CLOSE("turn knot curve var gradient", spreads[j].adj(), fd, 1e-6);
    }

    // Nested second-order scalar: the turn-knotted curve and its span-straddling
    // quote evaluate through fvar<var>.
    using stan::math::fvar;
    std::vector<fvar<var>> nestedSpreads(baseSpreads.begin(), baseSpreads.end());
    std::vector<fvar<var>> nestedParent(parentZeros.begin(), parentZeros.end());
    auto nestedParentCurve = std::make_shared<const DiscountCurve<fvar<var>>>(
        parentTimes, nestedParent, InterpolationSpace::LogDiscount, InterpolationScheme::Linear);
    const markets::SpreadCurve<fvar<var>> nestedCurve(nestedParentCurve, nodeTimes, nestedSpreads,
                                                      InterpolationScheme::Linear);
    const fvar<var> nested =
        markets::impliedForecastQuote(nestedCurve, *nestedParentCurve, pillar, reference, zeroDc);
    CHECK_CLOSE("turn knot fvar value", nested.val().val(), primal(baseSpreads), 1e-12);
}

void testForecastDepositVarQuote() {
    const std::vector<double> times{0.0, 1.0, 2.0};
    const std::vector<double> baseParent{0.0, 0.03, 0.035};
    const std::vector<double> baseSpreads{0.0, 0.001, 0.0015};
    std::vector<var> parentZeros(baseParent.begin(), baseParent.end());
    std::vector<var> spreads(baseSpreads.begin(), baseSpreads.end());
    const datetime::Date reference(2026, 9, 29);
    const datetime::Date start = reference.plusMonths(6);
    const datetime::Date maturity = reference.plusMonths(18);
    const datetime::DayCounter zeroDc(datetime::DayCount::Actual365Fixed);

    markets::ForecastPillar pillar;
    pillar.kind = markets::ForecastPillar::Kind::Deposit;
    pillar.start = start;
    pillar.maturity = maturity;
    pillar.quoteDayCounter = datetime::DayCounter(datetime::DayCount::Actual360);
    pillar.calendar = datetime::Calendar::noHolidays();

    auto parent = std::make_shared<const DiscountCurve<var>>(
        times, parentZeros, InterpolationSpace::LogDiscount, InterpolationScheme::Linear);
    var quote = 0.0;
    {
        const markets::SpreadCurve<var> curve(parent, times, spreads, InterpolationScheme::Linear);
        quote = markets::impliedForecastQuote(curve, *parent, pillar, reference, zeroDc);
    }
    quote.grad();

    const auto primal = [&](const std::vector<double>& parentNodes,
                            const std::vector<double>& spreadNodes) {
        auto base = std::make_shared<const DiscountCurve<double>>(
            times, parentNodes, InterpolationSpace::LogDiscount, InterpolationScheme::Linear);
        const markets::SpreadCurve<double> curve(base, times, spreadNodes,
                                                 InterpolationScheme::Linear);
        return markets::impliedForecastQuote(curve, *base, pillar, reference, zeroDc);
    };
    const double epsilon = 1e-6;
    for (std::size_t j = 1; j < baseParent.size(); ++j) {
        std::vector<double> plus = baseParent;
        std::vector<double> minus = baseParent;
        plus[j] += epsilon;
        minus[j] -= epsilon;
        const double fd =
            (primal(plus, baseSpreads) - primal(minus, baseSpreads)) / (2.0 * epsilon);
        CHECK_CLOSE("forecast deposit parent var gradient", parentZeros[j].adj(), fd, 1e-6);
    }
    for (std::size_t j = 1; j < baseSpreads.size(); ++j) {
        std::vector<double> plus = baseSpreads;
        std::vector<double> minus = baseSpreads;
        plus[j] += epsilon;
        minus[j] -= epsilon;
        const double fd = (primal(baseParent, plus) - primal(baseParent, minus)) / (2.0 * epsilon);
        CHECK_CLOSE("forecast deposit spread var gradient", spreads[j].adj(), fd, 1e-6);
    }

    // Nested second-order scalar: the same quote evaluates through fvar<var>.
    using stan::math::fvar;
    std::vector<fvar<var>> nestedParent(baseParent.begin(), baseParent.end());
    std::vector<fvar<var>> nestedSpreads(baseSpreads.begin(), baseSpreads.end());
    auto nestedParentCurve = std::make_shared<const DiscountCurve<fvar<var>>>(
        times, nestedParent, InterpolationSpace::LogDiscount, InterpolationScheme::Linear);
    const markets::SpreadCurve<fvar<var>> nestedCurve(nestedParentCurve, times, nestedSpreads,
                                                      InterpolationScheme::Linear);
    const fvar<var> nested =
        markets::impliedForecastQuote(nestedCurve, *nestedParentCurve, pillar, reference, zeroDc);
    CHECK_CLOSE("forecast deposit fvar value", nested.val().val(), primal(baseParent, baseSpreads),
                1e-12);
}

void testForecastFutureVarQuoteCase(markets::FutureStyle style, markets::AveragingStyle averaging,
                                    const char* tag) {
    const std::vector<double> times{0.0, 1.0, 2.0};
    const std::vector<double> baseParent{0.0, 0.03, 0.035};
    const std::vector<double> baseSpreads{0.0, 0.001, 0.0015};
    std::vector<var> parentZeros(baseParent.begin(), baseParent.end());
    std::vector<var> spreads(baseSpreads.begin(), baseSpreads.end());
    const datetime::Date reference(2026, 9, 29);
    const datetime::Date start = reference.plusMonths(6);
    const datetime::Date maturity = reference.plusMonths(18);
    const datetime::DayCounter zeroDc(datetime::DayCount::Actual365Fixed);

    markets::ForecastPillar pillar;
    pillar.kind = markets::ForecastPillar::Kind::Future;
    pillar.start = start;
    pillar.maturity = maturity;
    pillar.futureStyle = style;
    pillar.averagingStyle = averaging;
    pillar.convexityAdjustment = 2.3e-4;
    pillar.quoteDayCounter = datetime::DayCounter(datetime::DayCount::Actual360);
    pillar.calendar = datetime::Calendar::noHolidays();

    auto parent = std::make_shared<const DiscountCurve<var>>(
        times, parentZeros, InterpolationSpace::LogDiscount, InterpolationScheme::Linear);
    var quote = 0.0;
    {
        const markets::SpreadCurve<var> curve(parent, times, spreads, InterpolationScheme::Linear);
        quote = markets::impliedForecastQuote(curve, *parent, pillar, reference, zeroDc);
    }
    quote.grad();

    const auto primal = [&](const std::vector<double>& parentNodes,
                            const std::vector<double>& spreadNodes) {
        auto base = std::make_shared<const DiscountCurve<double>>(
            times, parentNodes, InterpolationSpace::LogDiscount, InterpolationScheme::Linear);
        const markets::SpreadCurve<double> curve(base, times, spreadNodes,
                                                 InterpolationScheme::Linear);
        return markets::impliedForecastQuote(curve, *base, pillar, reference, zeroDc);
    };
    CHECK_CLOSE(tag, quote.val(), primal(baseParent, baseSpreads), 1e-15);

    const double epsilon = 1e-6;
    for (std::size_t j = 1; j < baseParent.size(); ++j) {
        std::vector<double> plus = baseParent;
        std::vector<double> minus = baseParent;
        plus[j] += epsilon;
        minus[j] -= epsilon;
        const double fd =
            (primal(plus, baseSpreads) - primal(minus, baseSpreads)) / (2.0 * epsilon);
        CHECK_CLOSE("forecast future parent var gradient", parentZeros[j].adj(), fd, 1e-6);
    }
    for (std::size_t j = 1; j < baseSpreads.size(); ++j) {
        std::vector<double> plus = baseSpreads;
        std::vector<double> minus = baseSpreads;
        plus[j] += epsilon;
        minus[j] -= epsilon;
        const double fd = (primal(baseParent, plus) - primal(baseParent, minus)) / (2.0 * epsilon);
        CHECK_CLOSE("forecast future spread var gradient", spreads[j].adj(), fd, 1e-6);
    }

    // Nested second-order scalar: the same quote evaluates through fvar<var>.
    using stan::math::fvar;
    std::vector<fvar<var>> nestedParent(baseParent.begin(), baseParent.end());
    std::vector<fvar<var>> nestedSpreads(baseSpreads.begin(), baseSpreads.end());
    auto nestedParentCurve = std::make_shared<const DiscountCurve<fvar<var>>>(
        times, nestedParent, InterpolationSpace::LogDiscount, InterpolationScheme::Linear);
    const markets::SpreadCurve<fvar<var>> nestedCurve(nestedParentCurve, times, nestedSpreads,
                                                      InterpolationScheme::Linear);
    const fvar<var> nested =
        markets::impliedForecastQuote(nestedCurve, *nestedParentCurve, pillar, reference, zeroDc);
    CHECK_CLOSE("forecast future fvar value", nested.val().val(), primal(baseParent, baseSpreads),
                1e-12);
}

/// Portfolio risk table over the replicated curve set. The direct turn knots
/// are ordinary forecast quotes tagged `turnPillar`: they stay in the same
/// quote Jacobian and column order, but report under `CurveRole::TurnOverlay`
/// with a `Turn <date>` label. The table is the stack quote-risk transform of
/// the portfolio's frozen-curve node gradients, proving that discount,
/// forecast and turn buckets compose in one solve. This test is the only
/// writer of the risk table CSV; with no env override it lands under the
/// per-test scratch directory, so CTest order and parallelism cannot race on
/// the file.
void testPortfolioRiskTable(const std::filesystem::path& outputPath) {
    const mk::DiscountCurve<double> ois = buildOisEcb(mk::InterpolationScheme::Linear);
    const auto parent = std::make_shared<mk::DiscountCurve<double>>(ois);
    const auto bootstrapChild = [&](const std::vector<mk::ForecastPillar>& pillars) {
        return std::make_shared<mk::SpreadCurve<double>>(mk::bootstrapForecastCurve(
            parent, &ois, kReference, kZeroDc, mk::InterpolationScheme::Linear, pillars));
    };
    const auto smooth1m = bootstrapChild(euribor1mSyntheticPillars());
    const auto smooth3m = bootstrapChild(euribor3mSyntheticPillars());
    const auto smooth6m = bootstrapChild(euribor6mPillars());
    const auto smooth12m = bootstrapChild(euribor12mPillars());

    const dt::Date postTurnEnd(2014, 3, 13);
    std::vector<std::vector<mk::ForecastPillar>> directPillars(4);
    directPillars[0] =
        directTurnPillars(euribor1mSyntheticPillars(), *smooth1m, kTurnOneMonthAmplitude,
                          {exactFraPillar(kTurnEnd, postTurnEnd,
                                          simpleForwardOver(*smooth1m, kTurnEnd, postTurnEnd))});
    directPillars[1] =
        directTurnPillars(euribor3mSyntheticPillars(), *smooth3m, kTurnThreeMonthAmplitude, {});
    directPillars[2] =
        directTurnPillars(euribor6mPillars(), *smooth6m, kTurnThreeMonthAmplitude, {});
    directPillars[3] =
        directTurnPillars(euribor12mPillars(), *smooth12m, kTurnThreeMonthAmplitude, {});
    for (std::vector<mk::ForecastPillar>& pillars : directPillars) {
        for (mk::ForecastPillar& pillar : pillars) {
            pillar.turnPillar = pillar.kind == mk::ForecastPillar::Kind::Fra &&
                                pillar.start == kTurnBegin && pillar.maturity == kTurnEnd;
        }
    }
    std::vector<std::shared_ptr<mk::SpreadCurve<double>>> directCurves(4);
    for (std::size_t i = 0; i < directPillars.size(); ++i) {
        directCurves[i] = bootstrapChild(directPillars[i]);
        CHECK_CLOSE("risk direct turn reprice",
                    worstRepriceError(directPillars[i], *directCurves[i], ois), 0.0, 1e-10);
    }

    const std::vector<std::string> curveNames{"OIS", "1M", "3M", "6M", "12M"};
    const dt::Period floatTenors[4] = {
        dt::Period(1, dt::TimeUnit::Months), dt::Period(3, dt::TimeUnit::Months),
        dt::Period(6, dt::TimeUnit::Months), dt::Period(12, dt::TimeUnit::Months)};
    const dt::Date fiveYear = dt::Period(5, dt::TimeUnit::Years).advance(kSpot);

    std::vector<RiskSwap> swaps;
    RiskSwap oisSwap;
    oisSwap.curve = 0;
    oisSwap.irs = riskIrsPillar(fiveYear, dt::Period(1, dt::TimeUnit::Years));
    swaps.push_back(oisSwap);
    // Every curve carries a five-year par leg; the 1M curve also keeps the
    // crossing short swap below, so its new 2Y-60Y long end enters the table.
    for (std::size_t i = 0; i < 4; ++i) {
        RiskSwap swap;
        swap.curve = i + 1;
        swap.irs = riskIrsPillar(fiveYear, floatTenors[i]);
        swaps.push_back(swap);
    }
    // One 1M and one 3M swap whose single accrual crosses the funding window.
    RiskSwap crossing1m;
    crossing1m.curve = 1;
    crossing1m.irs = riskIrsPillar(date(2014, 1, 2), dt::Period(1, dt::TimeUnit::Months));
    crossing1m.irs.start = date(2013, 12, 2);
    swaps.push_back(crossing1m);
    RiskSwap crossing3m;
    crossing3m.curve = 2;
    crossing3m.irs = riskIrsPillar(date(2014, 1, 2), dt::Period(3, dt::TimeUnit::Months));
    crossing3m.irs.start = date(2013, 10, 1);
    swaps.push_back(crossing3m);
    // Control: a 3M swap after the window and after the first post-turn
    // curve pillar, so none of its discount factors interpolate the turn
    // nodes. Its turn risk must vanish.
    RiskSwap control;
    control.curve = 2;
    control.irs = riskIrsPillar(date(2014, 6, 20), dt::Period(3, dt::TimeUnit::Months));
    control.irs.start = date(2014, 3, 20);
    swaps.push_back(control);

    for (RiskSwap& swap : swaps) {
        if (swap.curve == 0) {
            swap.fixedRate = mk::impliedIrsRate(ois, ois, swap.irs, kReference, kZeroDc);
        } else {
            swap.fixedRate = mk::impliedIrsRate(*directCurves[swap.curve - 1], ois, swap.irs,
                                                kReference, kZeroDc);
        }
    }

    std::vector<std::vector<double>> dVdNodes(5);
    dVdNodes[0].assign(ois.size(), 0.0);
    for (std::size_t k = 1; k < 5; ++k) {
        dVdNodes[k].assign(directCurves[k - 1]->size(), 0.0);
    }
    std::vector<std::vector<double>> controlNodes(5);
    controlNodes[0].assign(ois.size(), 0.0);
    for (std::size_t k = 1; k < 5; ++k) {
        controlNodes[k].assign(directCurves[k - 1]->size(), 0.0);
    }
    for (const RiskSwap& swap : swaps) {
        if (swap.curve == 0) {
            std::vector<double> forecastPart(ois.size(), 0.0);
            std::vector<double> discountPart(ois.size(), 0.0);
            accumulateSwapNodeSensitivity(ois, ois, swap.irs, swap.fixedRate, swap.notional,
                                          forecastPart, discountPart);
            for (std::size_t i = 0; i < dVdNodes[0].size(); ++i) {
                dVdNodes[0][i] += forecastPart[i] + discountPart[i];
            }
            continue;
        }
        const mk::SpreadCurve<double>& forecast = *directCurves[swap.curve - 1];
        std::vector<double> forecastPart(forecast.size(), 0.0);
        std::vector<double> discountPart(ois.size(), 0.0);
        accumulateSwapNodeSensitivity(forecast, ois, swap.irs, swap.fixedRate, swap.notional,
                                      forecastPart, discountPart);
        for (std::size_t i = 0; i < forecastPart.size(); ++i) {
            dVdNodes[swap.curve][i] += forecastPart[i];
        }
        for (std::size_t i = 0; i < discountPart.size(); ++i) {
            dVdNodes[0][i] += discountPart[i];
        }
    }
    {
        std::vector<double> forecastPart(directCurves[1]->size(), 0.0);
        std::vector<double> discountPart(ois.size(), 0.0);
        accumulateSwapNodeSensitivity(*directCurves[1], ois, control.irs, control.fixedRate,
                                      control.notional, forecastPart, discountPart);
        for (std::size_t i = 0; i < forecastPart.size(); ++i) {
            controlNodes[2][i] = forecastPart[i];
        }
        for (std::size_t i = 0; i < discountPart.size(); ++i) {
            controlNodes[0][i] = discountPart[i];
        }
    }

    // Overlay-mode turn risk: the same portfolio valued through a flat-forward
    // overlay on each smooth forecast curve, with the amplitudes as AD
    // parameters. The overlay is an exogenous rate factor, so this is the
    // amplitude route rather than the direct-knot quote route above.
    const std::vector<mk::TurnOverlay<double, mk::SpreadCurve<double>>::Bump> overlayBumps{
        {kTurnBeginTime, kTurnEndTime, kTurnThreeMonthAmplitude}};
    const std::array<std::shared_ptr<mk::SpreadCurve<double>>, 4> smoothCurves{smooth1m, smooth3m,
                                                                               smooth6m, smooth12m};
    std::vector<double> overlayTurn(4, 0.0);
    std::vector<mk::TurnRiskEntry> overlayEntries;
    for (std::size_t k = 0; k < smoothCurves.size(); ++k) {
        const auto risk = mk::turnOverlayRisk<stan::math::var>(
            *smoothCurves[k], {}, overlayBumps, {"Turn 2013-12-27"}, [&](const auto& overlay) {
                stan::math::var total = 0.0;
                for (const RiskSwap& swap : swaps) {
                    if (swap.curve == k + 1) {
                        total += swapValue(overlay, ois, swap.irs, swap.fixedRate, swap.notional);
                    }
                }
                return total;
            });
        overlayTurn[k] = risk[0].delta;
        overlayEntries.insert(overlayEntries.end(), risk.begin(), risk.end());
    }

    std::vector<mk::StackCurveInput> inputs(5);
    inputs[0].curve = mk::StackCurveView::make(ois);
    inputs[0].role = mk::CurveRole::Discount;
    inputs[0].discountPillars = oisEcbPillars();
    inputs[0].dVdNodes = dVdNodes[0];
    for (std::size_t k = 1; k < 5; ++k) {
        inputs[k].curve = mk::StackCurveView::make(*directCurves[k - 1]);
        inputs[k].role = mk::CurveRole::Forecast;
        inputs[k].forecastPillars = directPillars[k - 1];
        inputs[k].dVdNodes = dVdNodes[k];
    }

    const mk::StackQuoteSystem system = mk::assembleStackQuoteSystem(inputs, kReference);
    const std::vector<mk::StackRiskEntry> entries = mk::stackQuoteRisk(inputs, kReference);

    struct TableRow {
        std::string role;
        std::string label;
        double delta;
    };
    std::vector<TableRow> rows;
    for (std::size_t k = 0; k < entries.size(); ++k) {
        const mk::StackRiskEntry& entry = entries[k];
        for (const mk::QuotePoint& point : entry.points) {
            rows.push_back(TableRow{std::string(mk::curveRoleName(point.role)),
                                    curveNames[k] + " " + point.label, point.delta});
        }
    }
    for (std::size_t k = 0; k < overlayTurn.size(); ++k) {
        rows.push_back(TableRow{"TurnOverlay", curveNames[k + 1] + " Turn overlay 2013-12-27",
                                overlayTurn[k]});
    }
    const std::filesystem::path output(outputPath);
    if (output.has_parent_path()) {
        std::filesystem::create_directories(output.parent_path());
    }
    std::ofstream out(output);
    out << std::setprecision(15);
    out << "role,label,delta\n";
    for (const TableRow& row : rows) {
        out << row.role << ',' << row.label << ',' << row.delta << '\n';
    }

    // Compact view: role totals plus every five-year and turn row. The
    // exogenous overlay turn entries fold into the same role buckets.
    const std::vector<mk::RiskBucket> stackBuckets = mk::stackRoleBuckets(entries);
    const std::vector<mk::RiskBucket> roleBuckets =
        mk::addTurnRiskBuckets(stackBuckets, overlayEntries);

    // Self-quote identity: the transform of one pillar's own Jacobian row must
    // be exactly that quote's unit vector on the exact-fit bootstrap.
    std::vector<std::size_t> curveStartRow(5, 0);
    std::size_t rowCount = 0;
    for (std::size_t k = 0; k < inputs.size(); ++k) {
        curveStartRow[k] = rowCount;
        rowCount += inputs[k].curve->size() - 1;
    }
    EXPECT_EQ(system.dim, rowCount);
    std::vector<std::size_t> selectedRows;
    std::vector<int> selectedYears{5, 1, 5, 5, 5};
    for (std::size_t k = 0; k < inputs.size(); ++k) {
        if (k == 0) {
            const std::vector<mk::CurvePillar> pillars = oisEcbPillars();
            for (std::size_t j = 0; j < pillars.size(); ++j) {
                const double t =
                    dt::yearFraction(kReference, mk::pillarRiskMaturity(pillars[j]), kZeroDc);
                if (static_cast<int>(std::lround(t)) == selectedYears[k]) {
                    selectedRows.push_back(curveStartRow[k] + j);
                }
            }
        } else {
            for (std::size_t j = 0; j < directPillars[k - 1].size(); ++j) {
                const double t = dt::yearFraction(
                    kReference, mk::forecastPillarQuotedMaturity(directPillars[k - 1][j]), kZeroDc);
                if (static_cast<int>(std::lround(t)) == selectedYears[k]) {
                    selectedRows.push_back(curveStartRow[k] + j);
                }
                if (directPillars[k - 1][j].turnPillar) {
                    selectedRows.push_back(curveStartRow[k] + j);
                }
            }
        }
    }
    const auto transformNodeGradient = [&](const std::vector<std::vector<double>>& gradients) {
        std::vector<mk::StackCurveInput> graded = inputs;
        for (std::size_t k = 0; k < graded.size(); ++k) {
            graded[k].dVdNodes = gradients[k];
        }
        const std::vector<mk::StackRiskEntry> transformed = mk::stackQuoteRisk(graded, kReference);
        std::vector<double> flat;
        for (const mk::StackRiskEntry& entry : transformed) {
            for (const mk::QuotePoint& point : entry.points) {
                flat.push_back(point.delta);
            }
        }
        return flat;
    };
    double worstSelf = 0.0;
    for (const std::size_t row : selectedRows) {
        std::vector<std::vector<double>> rowGradient(5);
        for (std::size_t k = 0; k < rowGradient.size(); ++k) {
            rowGradient[k].assign(inputs[k].curve->size(), 0.0);
            for (std::size_t i = 0; i + 1 < inputs[k].curve->size(); ++i) {
                rowGradient[k][i + 1] = system.jacobian[row * system.dim + system.offsets[k] + i];
            }
        }
        const std::vector<double> identity = transformNodeGradient(rowGradient);
        for (std::size_t j = 0; j < identity.size(); ++j) {
            const double expected = j == row ? 1.0 : 0.0;
            worstSelf = std::max(worstSelf, std::abs(identity[j] - expected));
        }
    }
    EXPECT_LT(worstSelf, 1e-8);

    // Turn buckets against the analytic bump weight. The direct quote is a
    // simple rate over the window; the continuously compounded bump weight
    // scales by 1 / (1 + quote * tau).
    const double tauWindow = dt::yearFraction(kTurnBegin, kTurnEnd, kAct360);
    for (std::size_t k = 1; k < 5; ++k) {
        double directTurn = 0.0;
        double turnQuote = 0.0;
        for (std::size_t j = 0; j < directPillars[k - 1].size(); ++j) {
            if (!directPillars[k - 1][j].turnPillar) {
                continue;
            }
            directTurn += entries[k].points[j].delta;
            turnQuote = directPillars[k - 1][j].quote;
        }
        double weight = 0.0;
        for (const RiskSwap& swap : swaps) {
            if (swap.curve == k) {
                weight += turnWindowWeight(*smoothCurves[k - 1], ois, swap.irs,
                                           kTurnThreeMonthAmplitude, swap.notional);
            }
        }
        // The overlay amplitude derivative equals the exact bump weight by
        // construction. The direct quote is a simple rate, so its scale is
        // weight / (1 + quote * tau) before bootstrap re-solve effects. The
        // 5Y curves that re-quote their turn-spanning coupon on the next
        // pillar (6M/12M) show a structurally zero direct bucket while the
        // overlay factor keeps the full exposure; the 1M/3M buckets are the
        // crossing single-coupon swaps below.
        EXPECT_LT(std::abs(overlayTurn[k - 1] - weight), 1e-9);
        if (k == 1 || k == 2) {
            EXPECT_GT(directTurn, 0.0);
        }
    }

    // The control swap has no accrual inside the window: its turn bucket must
    // be structurally zero.
    const std::vector<double> controlIdentity = transformNodeGradient(controlNodes);
    double controlTurn = 0.0;
    for (std::size_t k = 1; k < 5; ++k) {
        for (std::size_t j = 0; j < entries[k].points.size(); ++j) {
            if (entries[k].points[j].role == mk::CurveRole::TurnOverlay) {
                const std::size_t global = curveStartRow[k] + j;
                controlTurn += controlIdentity[global];
            }
        }
    }
    EXPECT_LT(std::abs(controlTurn), 1e-8);

    // Role totals re-sum to the full table: the stack buckets cover the quote
    // side, and the overlay amplitude rows add the exogenous turn factors.
    double stackTotal = 0.0;
    for (const mk::StackRiskEntry& entry : entries) {
        for (const mk::QuotePoint& point : entry.points) {
            stackTotal += point.delta;
        }
    }
    double bucketTotal = 0.0;
    for (const mk::RiskBucket& bucket : stackBuckets) {
        bucketTotal += bucket.delta;
    }
    CHECK_CLOSE("stack role totals re-sum", bucketTotal, stackTotal, 1e-12);
    double rowTotal = 0.0;
    std::vector<mk::RiskBucket> rowBuckets;
    const auto addRowBucket = [&](std::string_view label, double delta) {
        for (mk::RiskBucket& bucket : rowBuckets) {
            if (bucket.label == label) {
                bucket.delta += delta;
                return;
            }
        }
        rowBuckets.push_back(mk::RiskBucket{std::string(label), delta});
    };
    for (const TableRow& row : rows) {
        rowTotal += row.delta;
        addRowBucket(row.role, row.delta);
    }
    double overlayTotal = 0.0;
    for (const double delta : overlayTurn) {
        overlayTotal += delta;
    }
    CHECK_CLOSE("table rows sum to stack plus overlay", rowTotal, stackTotal + overlayTotal, 1e-12);
    double rowBucketTotal = 0.0;
    for (const mk::RiskBucket& bucket : rowBuckets) {
        rowBucketTotal += bucket.delta;
    }
    CHECK_CLOSE("role totals sum to the portfolio delta", rowBucketTotal, rowTotal, 1e-12);
    double turnTotal = 0.0;
    for (const mk::RiskBucket& bucket : rowBuckets) {
        if (bucket.label == "TurnOverlay") {
            turnTotal += bucket.delta;
        }
    }
    double stackTurnTotal = 0.0;
    for (const mk::RiskBucket& bucket : stackBuckets) {
        if (bucket.label == "TurnOverlay") {
            stackTurnTotal += bucket.delta;
        }
    }
    double mergedBucketTotal = 0.0;
    double mergedTurnTotal = 0.0;
    for (const mk::RiskBucket& bucket : roleBuckets) {
        mergedBucketTotal += bucket.delta;
        if (bucket.label == "TurnOverlay") {
            mergedTurnTotal += bucket.delta;
        }
    }
    // The turn-merged role buckets are the full table: quote-side turn knots
    // plus the exogenous overlay amplitudes under one TurnOverlay role.
    CHECK_CLOSE("turn-merged role totals", mergedBucketTotal, rowTotal, 1e-12);
    CHECK_CLOSE("turn-merged TurnOverlay bucket", mergedTurnTotal, turnTotal, 1e-12);
    CHECK_CLOSE("overlay turn risk folded into role buckets", mergedTurnTotal,
                stackTurnTotal + overlayTotal, 1e-12);
    // Deterministic table total: the portfolio's net quote delta across the
    // discount, forecast and turn buckets, including the overlay rows.
    CHECK_CLOSE("risk table total", rowTotal, kRiskTableTotal, 1e-9);
    // Table reading: the five-year forecast self-quotes carry the bulk of the
    // risk (about one fixed annuity each), the discount bucket nets negative
    // because every IBOR leg also discounts on the OIS root, and the turn
    // bucket is small and positive; the overlay rows grow with coupon level
    // and exactly match the sum of payment-discount weights times the funding
    // window overlaps.
}

struct CurveParam {
    InterpolationSpace space;
    InterpolationScheme scheme;
    double tension;
    const char* label;
};

const CurveParam kCurveParams[] = {
    {InterpolationSpace::LogDiscount, InterpolationScheme::Linear, 0.0, "log-discount linear"},
    {InterpolationSpace::Zero, InterpolationScheme::Linear, 0.0, "zero linear"},
    {InterpolationSpace::LogDiscount, InterpolationScheme::Akima, 0.0, "log-discount akima"},
    {InterpolationSpace::LogDiscount, InterpolationScheme::TensionSpline, 8.0,
     "log-discount tension 8"},
};

} // namespace

class CurveAdTest : public StanTapeTest {};

TEST_F(CurveAdTest, discountCurveVarGradientMatchesFiniteDifference) {
    for (const CurveParam& param : kCurveParams) {
        SCOPED_TRACE(param.label);
        testDiscountCurveVarGradient(param.space, param.scheme, param.tension);
    }
}

TEST_F(CurveAdTest, monotoneCubicValueAndPinnedGradient) {
    testMonotoneCubicVarValue();
    SCOPED_TRACE("pinned gradient");
    testMonotoneCubicPinnedGradient();
}

TEST_F(CurveAdTest, varRiskWeightsMatchDouble) {
    testVarRiskWeightsMatchDouble();
}

TEST_F(CurveAdTest, hymanSplineValueGradientAndImpliedQuote) {
    testHymanSplineCurveVar();
}

TEST_F(CurveAdTest, spreadAndTiedSecantVarGradients) {
    SCOPED_TRACE("spread parent and node gradients");
    testSpreadCurveVarGradient();
    for (const InterpolationScheme scheme :
         {InterpolationScheme::Akima, InterpolationScheme::MixedLinearCubic}) {
        SCOPED_TRACE(scheme == InterpolationScheme::Akima ? "tied secant akima"
                                                          : "tied secant mixed linear cubic");
        testTiedSecantGradient(scheme);
    }
    SCOPED_TRACE("spread tied secant gradient");
    testSpreadCurveTiedSecantGradient();
}

TEST_F(CurveAdTest, turnOverlayVarGradientsAndNodeWeights) {
    SCOPED_TRACE("turn overlay gradients");
    testTurnOverlayVarGradient();
    SCOPED_TRACE("turn overlay node weights");
    testTurnOverlayNodeWeightsAd();
}

TEST_F(CurveAdTest, turnOverlayRiskAdjointsAndMetadata) {
    SCOPED_TRACE("risk adjoints");
    testTurnOverlayRiskAdjoint();
    SCOPED_TRACE("risk metadata");
    testTurnOverlayRiskMetadata();
}

TEST_F(CurveAdTest, turnKnotCurveVarGradientAndFvarValue) {
    testTurnKnotCurveVarGradient();
}

TEST_F(CurveAdTest, forecastQuoteVarGradients) {
    SCOPED_TRACE("forecast deposit");
    testForecastDepositVarQuote();
    struct FutureCase {
        markets::FutureStyle style;
        markets::AveragingStyle averaging;
        const char* tag;
    };
    const FutureCase cases[] = {
        {markets::FutureStyle::Simple, markets::AveragingStyle::Arithmetic, "future simple var"},
        {markets::FutureStyle::Averaged, markets::AveragingStyle::Arithmetic,
         "future averaged var"},
        {markets::FutureStyle::Averaged, markets::AveragingStyle::Compounded,
         "future compounded-average var"},
    };
    for (const FutureCase& futureCase : cases) {
        SCOPED_TRACE(futureCase.tag);
        testForecastFutureVarQuoteCase(futureCase.style, futureCase.averaging, futureCase.tag);
    }
}

TEST_F(CurveAdTest, materializeSolverAndFutureVarValue) {
    SCOPED_TRACE("materialize");
    testMaterializeVarGradient();
    SCOPED_TRACE("bootstrap solver");
    testDepositBootstrapSolverAd();
    SCOPED_TRACE("averaged compounded future");
    testAveragedCompoundedFutureVarValue();
}

TEST_F(CurveAdTest, hullWhiteAndFraConvexityAdjoints) {
    SCOPED_TRACE("hull-white");
    testHullWhiteConvexityVarGradient();
    SCOPED_TRACE("fra");
    testFraConvexityVarGradient();
}

/// The risk-table writer owns a per-test scratch directory (TempDirTest) and
/// recovers the tape in the fixture hooks.
class PortfolioRiskTableTest : public TempDirTest {
protected:
    void SetUp() override {
        TempDirTest::SetUp();
        stan::math::recover_memory();
        stan::math::set_zero_all_adjoints();
    }

    void TearDown() override {
        stan::math::set_zero_all_adjoints();
        stan::math::recover_memory();
        TempDirTest::TearDown();
    }
};

TEST_F(PortfolioRiskTableTest, totalsIdentityTurnBucketsAndControls) {
    // Honour the env override: an absolute path is used verbatim, a relative
    // one lands under the scratch directory, and the default is the scratch
    // directory itself (never a shared checkout path).
    std::filesystem::path outputPath = dir() / "risk_table.csv";
    if (const char* overridePath = std::getenv("QTA_PAPER_EUR_RISK")) {
        outputPath = overridePath;
        if (outputPath.is_relative()) {
            outputPath = dir() / outputPath;
        }
    }
    testPortfolioRiskTable(outputPath);
}
