// test_curve_config.cpp — JSON curve-stack configuration gates.
#include "quantape/markets/Curves/CurveConfig.h"
#include "quantape/markets/Curves/FxSwapBuilder.h"
#include "quantape/markets/Curves/XccyBasisBuilder.h"

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <memory>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

#include "support/GtestSupport.h"

using namespace quantape;

namespace {

std::string curveConfigFixture() {
    return RequireDataFile(CURVE_CONFIG_FIXTURE).string();
}

std::string fxConfigFixture() {
    return RequireDataFile(FX_CONFIG_FIXTURE).string();
}

void expectParseReject(const std::string& json, const std::string& needle) {
    bool threw = false;
    std::string message;
    try {
        (void)markets::parseCurveStackSpec(json);
    } catch (const std::invalid_argument& error) {
        threw = true;
        message = error.what();
    }
    EXPECT_TRUE(threw);
    if (threw) {
        EXPECT_NE(message.find(needle), std::string::npos) << message;
    }
}

const markets::BuiltCurve* findBuilt(const std::vector<markets::BuiltCurve>& built,
                                     const markets::CurveKey& key) {
    for (const markets::BuiltCurve& entry : built) {
        if (entry.key == key) {
            return &entry;
        }
    }
    return nullptr;
}

void expectBuildReject(const markets::CurveStackSpec& stack, const std::string& needle) {
    bool threw = false;
    std::string message;
    try {
        (void)markets::buildStack(stack);
    } catch (const std::invalid_argument& error) {
        threw = true;
        message = error.what();
    }
    EXPECT_TRUE(threw);
    if (threw) {
        EXPECT_NE(message.find(needle), std::string::npos) << message;
    }
}

} // namespace

TEST(CurveConfig, loadsUsdOisFixtureAndReprices) {
    const std::string fixture = curveConfigFixture();
    const markets::CurveStackSpec stack = markets::loadCurveStackSpec(fixture);
    EXPECT_TRUE(stack.asOf == datetime::Date::parse("2026-09-29"));
    ASSERT_EQ(stack.curves.size(), 1u);

    const markets::CurveSpec& spec = stack.curves.front();
    EXPECT_EQ(spec.key.currency, "USD");
    EXPECT_TRUE(spec.key.role == markets::CurveRole::Discount);
    EXPECT_EQ(spec.key.collateral, "USD");
    EXPECT_TRUE(spec.space == markets::InterpolationSpace::LogDiscount);
    EXPECT_TRUE(spec.scheme == markets::InterpolationScheme::Linear);
    ASSERT_EQ(spec.pillars.size(), 6u);
    EXPECT_TRUE(spec.pillars.front().kind == markets::PillarSpec::Kind::Deposit);
    EXPECT_TRUE(spec.pillars.back().kind == markets::PillarSpec::Kind::OisSwap);

    std::vector<markets::CurvePillar> filled;
    const markets::DiscountCurve<double> curve = markets::buildCurve(stack, spec, {}, &filled);
    EXPECT_EQ(curve.size(), spec.pillars.size() + 1);
    EXPECT_EQ(curve.discount(0.0), 1.0);

    EXPECT_EQ(filled.size(), spec.pillars.size());
    for (const markets::CurvePillar& pillar : filled) {
        CHECK_CLOSE("config reprice", markets::impliedQuote(pillar, stack.asOf, curve),
                    pillar.quote, 1e-10);
    }

    double previous = curve.discount(0.0);
    for (double t = 0.25; t <= 10.0; t += 0.25) {
        const double current = curve.discount(t);
        EXPECT_TRUE(current < previous);
        previous = current;
    }

    // A flat stack (every spec parentless) builds through the stack builder and
    // matches the direct build discount for discount.
    const markets::CurveStackSpec flat = markets::loadCurveStackSpec(fixture);
    for (const markets::CurveSpec& flatSpec : flat.curves) {
        EXPECT_FALSE(flatSpec.hasParent);
        EXPECT_FALSE(flatSpec.hasDiscount);
    }
    const std::vector<markets::BuiltCurve> built = markets::buildStack(flat);
    ASSERT_EQ(built.size(), flat.curves.size());
    EXPECT_TRUE(built[0].role == markets::CurveRole::Discount);
    const markets::DiscountCurve<double> directly = markets::buildCurve(flat, flat.curves[0]);
    for (double t = 0.25; t <= 10.0; t += 0.25) {
        CHECK_CLOSE("fixture stack builder discount", built[0].curve->discount(t),
                    directly.discount(t), 1e-15);
    }
}

TEST(CurveConfig, parseValidationAndEnumErrors) {
    SCOPED_TRACE("malformed json");
    EXPECT_THROW((void)markets::parseCurveStackSpec("{ not json "), std::invalid_argument);

    SCOPED_TRACE("unknown interpolation scheme");
    {
        bool threw = false;
        try {
            (void)markets::parseCurveStackSpec(
                R"({"asOf": "2026-09-29", "curves": [{"key": {"currency": "USD", "role": "Discount"},
            "interpolation": {"scheme": "RatSpline"}, "pillars": [{"maturity": "2027-09-29",
            "kind": "Deposit", "quote": 0.04}]}]})");
        } catch (const std::invalid_argument& error) {
            threw = true;
            EXPECT_NE(std::string(error.what()).find("HymanSpline"), std::string::npos)
                << error.what();
        }
        EXPECT_TRUE(threw);
    }

    SCOPED_TRACE("empty curves");
    EXPECT_THROW((void)markets::parseCurveStackSpec(R"({"asOf": "2026-09-29", "curves": []})"),
                 std::invalid_argument);

    SCOPED_TRACE("day counter name and role name");
    CHECK_CLOSE(
        "day counter name",
        markets::dayCounterFromName("ACT/365F")
            .yearFraction(datetime::Date::parse("2026-01-01"), datetime::Date::parse("2027-01-01")),
        1.0, 1e-12);
    EXPECT_EQ(markets::curveRoleToName(markets::CurveRole::XccyBasis), "XccyBasis");

    SCOPED_TRACE("both pillar lists");
    const std::string bothLists = R"({
        "asOf": "2026-09-29",
        "curves": [{
            "key": {"currency": "USD", "role": "Discount"},
            "pillars": [{"maturity": "2027-09-29", "kind": "Deposit", "quote": 0.04,
                         "calendar": "NoHolidays"}],
            "forecastPillars": [{"maturity": "2027-09-29", "kind": "Deposit",
                                 "quote": 0.041, "calendar": "NoHolidays"}]
        }]
    })";
    expectParseReject(bothLists, "both");

    SCOPED_TRACE("forecast with discount pillars");
    const std::string forecastWithPillars = R"({
        "asOf": "2026-09-29",
        "curves": [{
            "key": {"currency": "USD", "role": "Forecast"},
            "pillars": [{"maturity": "2027-09-29", "kind": "Deposit", "quote": 0.04,
                         "calendar": "NoHolidays"}]
        }]
    })";
    expectParseReject(forecastWithPillars, "forecastPillars");

    SCOPED_TRACE("discount with forecast pillars");
    const std::string discountWithForecast = R"({
        "asOf": "2026-09-29",
        "curves": [{
            "key": {"currency": "USD", "role": "Discount"},
            "forecastPillars": [{"maturity": "2027-09-29", "kind": "Deposit",
                                 "quote": 0.041, "calendar": "NoHolidays"}]
        }]
    })";
    expectParseReject(discountWithForecast, "pillars");

    // XccyBasis curves are configurable, but only with cross-currency pillar
    // kinds (a plain Deposit pillar is refused with the accepted kinds).
    SCOPED_TRACE("xccy role pillar kinds");
    const std::string xccyRole = R"({
        "asOf": "2026-09-29",
        "curves": [{
            "key": {"currency": "USD", "role": "XccyBasis"},
            "pillars": [{"maturity": "2027-09-29", "kind": "Deposit", "quote": 0.04,
                         "calendar": "NoHolidays"}]
        }]
    })";
    expectParseReject(xccyRole, "FxSwap");

    SCOPED_TRACE("turn overlay role");
    const std::string turnRole = R"({
        "asOf": "2026-09-29",
        "curves": [{
            "key": {"currency": "USD", "role": "TurnOverlay"},
            "pillars": [{"maturity": "2027-09-29", "kind": "Deposit", "quote": 0.04,
                         "calendar": "NoHolidays"}]
        }]
    })";
    expectParseReject(turnRole, "not configurable");

    SCOPED_TRACE("duplicate curve key");
    const std::string duplicate = R"({
        "asOf": "2026-09-29",
        "curves": [
            {
                "key": {"currency": "USD", "role": "Discount", "collateral": "USD"},
                "pillars": [{"maturity": "2027-09-29", "kind": "Deposit", "quote": 0.04,
                             "calendar": "NoHolidays"}]
            },
            {
                "key": {"currency": "USD", "role": "Discount", "collateral": "USD"},
                "pillars": [{"maturity": "2028-09-29", "kind": "Deposit", "quote": 0.041,
                             "calendar": "NoHolidays"}]
            }
        ]
    })";
    expectParseReject(duplicate, "duplicate curve key");

    SCOPED_TRACE("typed field errors");
    const std::string badCollateral = R"({
        "asOf": "2026-09-29",
        "curves": [{
            "key": {"currency": "USD", "role": "Discount", "collateral": 7},
            "pillars": [{"maturity": "2027-09-29", "kind": "Deposit", "quote": 0.04}]
        }]
    })";
    expectParseReject(badCollateral, "string");

    const std::string badZeroDayCounter = R"({
        "asOf": "2026-09-29",
        "curves": [{
            "key": {"currency": "USD", "role": "Discount"},
            "zeroDayCounter": 5,
            "pillars": [{"maturity": "2027-09-29", "kind": "Deposit", "quote": 0.04}]
        }]
    })";
    expectParseReject(badZeroDayCounter, "string");

    const std::string badSwitchIndex = R"({
        "asOf": "2026-09-29",
        "curves": [{
            "key": {"currency": "USD", "role": "Discount"},
            "interpolation": {"switchIndex": 1.5},
            "pillars": [{"maturity": "2027-09-29", "kind": "Deposit", "quote": 0.04}]
        }]
    })";
    expectParseReject(badSwitchIndex, "integer");

    const std::string badBootstrap = R"({
        "asOf": "2026-09-29",
        "curves": [{
            "key": {"currency": "USD", "role": "Discount"},
            "bootstrap": {"method": 3, "accuracy": "tight"},
            "pillars": [{"maturity": "2027-09-29", "kind": "Deposit", "quote": 0.04}]
        }]
    })";
    expectParseReject(badBootstrap, "string");

    SCOPED_TRACE("unknown enum suggestions");
    expectParseReject(
        R"({"asOf": "2026-09-29", "curves": [{
            "key": {"currency": "USD", "role": "Discount"},
            "pillars": [{"maturity": "2027-09-29", "kind": "Deposit", "quote": 0.04,
                         "quoteDayCounter": "ACT/364"}]}]})",
        "ACT/360");
    expectParseReject(
        R"({"asOf": "2026-09-29", "curves": [{
            "key": {"currency": "USD", "role": "Discount"},
            "pillars": [{"maturity": "2027-09-29", "kind": "Deposit", "quote": 0.04,
                         "quoteDayCounter": "30/360 Foo"}]}]})",
        "30E/360");
    expectParseReject(
        R"({"asOf": "2026-09-29", "curves": [{
            "key": {"currency": "USD", "role": "Discount"},
            "pillars": [{"maturity": "2027-09-29", "kind": "Deposit", "quote": 0.04,
                         "calendar": "Mars"}]}]})",
        "TARGET");
    expectParseReject(
        R"({"asOf": "2026-09-29", "curves": [{
            "key": {"currency": "USD", "role": "Discount"},
            "pillars": [{"maturity": "2027-09-29", "kind": "Deposit", "quote": 0.04,
                         "businessDayConvention": "Sometimes"}]}]})",
        "ModifiedFollowing");
    expectParseReject(
        R"({"asOf": "2026-09-29", "curves": [{
            "key": {"currency": "USD", "role": "Discount"},
            "pillars": [{"maturity": "2027-09-29", "kind": "OisSwap", "quote": 0.04,
                         "fixedTenor": {"length": 1, "unit": "Fortnights"}}]}]})",
        "Months");
    expectParseReject(
        R"({"asOf": "2026-09-29", "curves": [{
            "key": {"currency": "USD", "role": "Discount"},
            "fraConvexity": {"model": "SABR", "sigmaIndex": 0.01, "sigmaDiscount": 0.01,
                             "correlation": 0.0},
            "pillars": [{"maturity": "2027-09-29", "kind": "Deposit", "quote": 0.04}]}]})",
        "ShiftedLognormal");
    expectParseReject(
        R"({"asOf": "2026-09-29", "curves": [{
            "key": {"currency": "USD", "role": "Discount"},
            "convexity": {"model": "SABR", "sigma": 0.01, "meanReversion": 0.05},
            "pillars": [{"maturity": "2027-09-29", "kind": "Deposit", "quote": 0.04}]}]})",
        "HullWhite");

    SCOPED_TRACE("forecast interpolation rejects");
    const std::string nonZeroSpace = R"({
        "asOf": "2026-09-29",
        "curves": [{
            "key": {"currency": "USD", "role": "Forecast"},
            "interpolation": {"space": "LogDiscount"},
            "forecastPillars": [{"maturity": "2027-03-29", "kind": "Deposit", "quote": 0.001}]
        }]
    })";
    expectParseReject(nonZeroSpace, "Zero");

    const std::string switchIndex = R"({
        "asOf": "2026-09-29",
        "curves": [{
            "key": {"currency": "USD", "role": "Forecast"},
            "interpolation": {"scheme": "Linear", "switchIndex": 2},
            "forecastPillars": [{"maturity": "2027-03-29", "kind": "Deposit", "quote": 0.001}]
        }]
    })";
    expectParseReject(switchIndex, "switchIndex");

    // An omitted 'space' defaults to the Zero space the spread curve actually
    // uses, rather than silently keeping the discount-curve default.
    SCOPED_TRACE("omitted forecast space defaults to zero");
    const std::string omittedSpace = R"({
        "asOf": "2026-09-29",
        "curves": [{
            "key": {"currency": "USD", "role": "Forecast"},
            "forecastPillars": [{"maturity": "2027-03-29", "kind": "Deposit", "quote": 0.001}]
        }]
    })";
    const markets::CurveStackSpec omittedStack = markets::parseCurveStackSpec(omittedSpace);
    EXPECT_TRUE(omittedStack.curves.front().space == markets::InterpolationSpace::Zero);
    EXPECT_EQ(omittedStack.curves.front().switchIndex, 1);
}

TEST(CurveConfig, defaultIrsFixedDayCounterIsBondBasis) {
    const datetime::DayCounter bondBasis = markets::dayCounterFromName("30/360 BondBasis");
    EXPECT_TRUE(bondBasis.convention() == datetime::DayCount::Thirty360BondBasis);
    EXPECT_TRUE(markets::PillarSpec{}.fixedDayCounter.convention() == bondBasis.convention());

    // The name emitted by the day-count catalogue parses back to the default.
    const std::string emitted(
        datetime::dayCountName(markets::PillarSpec{}.fixedDayCounter.convention()));
    EXPECT_TRUE(markets::dayCounterFromName(emitted).convention() == bondBasis.convention());

    // 30E/360 is a distinct convention with its own config name.
    const datetime::DayCounter euroBond = markets::dayCounterFromName("30E/360");
    EXPECT_TRUE(euroBond.convention() == datetime::DayCount::ThirtyE360);
    EXPECT_TRUE(euroBond.convention() != bondBasis.convention());

    // The default is what an Irs pillar gets when `fixedDayCounter` is omitted,
    // and the explicit name resolves to the same convention.
    const std::string json = R"({
        "asOf": "2026-09-29",
        "curves": [{
            "key": {"currency": "USD", "role": "Forecast"},
            "forecastPillars": [
                {"maturity": "2028-03-29", "kind": "Irs", "quote": 0.043,
                 "floatTenor": {"length": 3, "unit": "Months"},
                 "fixedTenor": {"length": 1, "unit": "Years"},
                 "floatCalendar": "NoHolidays", "fixedCalendar": "NoHolidays"},
                {"maturity": "2029-03-29", "kind": "Irs", "quote": 0.044,
                 "floatTenor": {"length": 3, "unit": "Months"},
                 "fixedTenor": {"length": 1, "unit": "Years"},
                 "floatCalendar": "NoHolidays", "fixedCalendar": "NoHolidays",
                 "fixedDayCounter": "30/360 BondBasis"}
            ]
        }]
    })";
    const markets::CurveStackSpec stack = markets::parseCurveStackSpec(json);
    ASSERT_EQ(stack.curves.front().forecastPillars.size(), 2u);
    EXPECT_TRUE(stack.curves.front().forecastPillars[0].fixedDayCounter.convention() ==
                datetime::DayCount::Thirty360BondBasis);
    EXPECT_TRUE(stack.curves.front().forecastPillars[1].fixedDayCounter.convention() ==
                stack.curves.front().forecastPillars[0].fixedDayCounter.convention());
}

TEST(CurveConfig, hymanSplineAndAveragedCompoundedConfigs) {
    SCOPED_TRACE("hyman spline scheme");
    const std::string json = R"({
        "asOf": "2026-09-29",
        "curves": [{
            "key": {"currency": "USD", "role": "Discount", "collateral": "USD"},
            "zeroDayCounter": "ACT/365F",
            "interpolation": {"space": "LogDiscount", "scheme": "HymanSpline"},
            "bootstrap": {"method": "IterativeSequential", "accuracy": 1e-14},
            "pillars": [
                {"maturity": "2027-03-29", "kind": "Deposit", "quote": 0.0432,
                 "quoteDayCounter": "ACT/360", "calendar": "NoHolidays"},
                {"maturity": "2027-09-29", "kind": "OisSwap", "quote": 0.0405,
                 "quoteDayCounter": "ACT/365F", "calendar": "NoHolidays",
                 "fixedTenor": {"length": 1, "unit": "Years"}},
                {"maturity": "2028-09-29", "kind": "OisSwap", "quote": 0.0395,
                 "quoteDayCounter": "ACT/365F", "calendar": "NoHolidays",
                 "fixedTenor": {"length": 1, "unit": "Years"}},
                {"maturity": "2029-09-29", "kind": "OisSwap", "quote": 0.039,
                 "quoteDayCounter": "ACT/365F", "calendar": "NoHolidays",
                 "fixedTenor": {"length": 1, "unit": "Years"}}
            ]
        }]
    })";
    const markets::CurveStackSpec stack = markets::parseCurveStackSpec(json);
    const markets::CurveSpec& spec = stack.curves.front();
    EXPECT_TRUE(spec.scheme == markets::InterpolationScheme::HymanSpline);
    EXPECT_EQ(markets::interpolationSchemeName(spec.scheme), "HymanSpline");

    std::vector<markets::CurvePillar> filled;
    const markets::DiscountCurve<double> curve = markets::buildCurve(stack, spec, {}, &filled);
    EXPECT_TRUE(curve.scheme() == markets::InterpolationScheme::HymanSpline);
    for (const markets::CurvePillar& pillar : filled) {
        CHECK_CLOSE("hyman config reprice", markets::impliedQuote(pillar, stack.asOf, curve),
                    pillar.quote, 1e-10);
    }
    // The filtered log-discounts keep the discount factors monotone.
    double previous = curve.discount(0.0);
    for (double t = 0.25; t <= 3.0; t += 0.25) {
        const double current = curve.discount(t);
        EXPECT_TRUE(current < previous + 1e-14);
        previous = current;
    }

    SCOPED_TRACE("averaged compounded future");
    const std::string averagedJson = R"({
        "asOf": "2026-09-29",
        "curves": [{
            "key": {"currency": "USD", "role": "Discount", "collateral": "USD"},
            "zeroDayCounter": "ACT/365F",
            "interpolation": {"space": "LogDiscount", "scheme": "Linear"},
            "bootstrap": {"method": "IterativeSequential", "accuracy": 1e-14},
            "pillars": [
                {"start": "2026-12-16", "maturity": "2027-01-16", "kind": "Future",
                 "style": "Averaged", "averaging": "Compounded", "quote": 0.041,
                 "convexityAdjustment": 0.0, "quoteDayCounter": "ACT/360",
                 "calendar": "NoHolidays"}
            ]
        }]
    })";
    const markets::CurveStackSpec averagedStack = markets::parseCurveStackSpec(averagedJson);
    const markets::CurveSpec& averagedSpec = averagedStack.curves.front();
    ASSERT_EQ(averagedSpec.pillars.size(), 1u);
    EXPECT_TRUE(averagedSpec.pillars.front().futureStyle == markets::FutureStyle::Averaged);
    EXPECT_TRUE(averagedSpec.pillars.front().averagingStyle == markets::AveragingStyle::Compounded);
    std::vector<markets::CurvePillar> averagedFilled;
    const markets::DiscountCurve<double> averagedCurve =
        markets::buildCurve(averagedStack, averagedSpec, {}, &averagedFilled);
    const markets::PillarSpec& averagedPillar = averagedSpec.pillars.front();
    ASSERT_EQ(averagedFilled.size(), 1u);
    CHECK_CLOSE("averaged compounded config reprice",
                markets::impliedQuote(averagedFilled.front(), averagedStack.asOf, averagedCurve),
                averagedPillar.quote, 1e-9);

    SCOPED_TRACE("averaged future rejects");
    const std::string wrongStyle = R"({
        "asOf": "2026-09-29",
        "curves": [{
            "key": {"currency": "USD", "role": "Discount", "collateral": "USD"},
            "pillars": [
                {"start": "2026-12-16", "maturity": "2027-01-16", "kind": "Future",
                 "style": "Simple", "averaging": "Compounded", "quote": 0.041,
                 "convexityAdjustment": 0.0, "quoteDayCounter": "ACT/360",
                 "calendar": "NoHolidays"}
            ]
        }]
    })";
    EXPECT_THROW((void)markets::parseCurveStackSpec(wrongStyle), std::invalid_argument);

    const std::string badValue = R"({
        "asOf": "2026-09-29",
        "curves": [{
            "key": {"currency": "USD", "role": "Discount", "collateral": "USD"},
            "pillars": [
                {"start": "2026-12-16", "maturity": "2027-01-16", "kind": "Future",
                 "style": "Averaged", "averaging": "Geometric", "quote": 0.041,
                 "convexityAdjustment": 0.0, "quoteDayCounter": "ACT/360",
                 "calendar": "NoHolidays"}
            ]
        }]
    })";
    EXPECT_THROW((void)markets::parseCurveStackSpec(badValue), std::invalid_argument);
}

TEST(ForecastConfig, mixedPillarReprices) {
    const std::string json = R"({
        "asOf": "2026-09-29",
        "curves": [
            {
                "key": {"currency": "USD", "role": "Discount", "collateral": "USD"},
                "zeroDayCounter": "ACT/365F",
                "interpolation": {"space": "LogDiscount", "scheme": "Linear"},
                "bootstrap": {"method": "IterativeSequential", "accuracy": 1e-14},
                "pillars": [
                    {"maturity": "2027-03-29", "kind": "Deposit", "quote": 0.0432,
                     "quoteDayCounter": "ACT/360", "calendar": "NoHolidays"},
                    {"maturity": "2027-09-29", "kind": "OisSwap", "quote": 0.0405,
                     "quoteDayCounter": "ACT/365F", "calendar": "NoHolidays",
                     "fixedTenor": {"length": 1, "unit": "Years"}},
                    {"maturity": "2028-09-29", "kind": "OisSwap", "quote": 0.0395,
                     "quoteDayCounter": "ACT/365F", "calendar": "NoHolidays",
                     "fixedTenor": {"length": 1, "unit": "Years"}},
                    {"maturity": "2029-09-29", "kind": "OisSwap", "quote": 0.039,
                     "quoteDayCounter": "ACT/365F", "calendar": "NoHolidays",
                     "fixedTenor": {"length": 1, "unit": "Years"}}
                ]
            },
            {
                "key": {"currency": "USD", "role": "Forecast",
                        "indexTenor": {"length": 3, "unit": "Months"}, "collateral": "USD"},
                "zeroDayCounter": "ACT/365F",
                "interpolation": {"space": "Zero", "scheme": "Linear"},
                "bootstrap": {"method": "IterativeSequential", "accuracy": 1e-14},
                "forecastPillars": [
                    {"maturity": "2027-03-29", "kind": "Deposit", "quote": 0.044,
                     "quoteDayCounter": "ACT/360", "calendar": "NoHolidays"},
                    {"start": "2027-03-29", "maturity": "2027-06-29", "kind": "Fra",
                     "quote": 0.0435, "quoteDayCounter": "ACT/360", "calendar": "NoHolidays"},
                    {"maturity": "2028-03-29", "kind": "Irs", "quote": 0.043,
                     "floatTenor": {"length": 3, "unit": "Months"},
                     "fixedTenor": {"length": 1, "unit": "Years"},
                     "floatCalendar": "NoHolidays", "fixedCalendar": "NoHolidays",
                     "floatDayCounter": "ACT/360", "fixedDayCounter": "30/360",
                     "businessDayConvention": "Following", "paymentLag": 2},
                    {"maturity": "2029-03-29", "kind": "BasisSwap", "quote": 0.0012,
                     "floatTenor": {"length": 3, "unit": "Months"},
                     "quoteDayCounter": "ACT/360", "calendar": "NoHolidays",
                     "businessDayConvention": "Following", "spreadOnParentLeg": true}
                ]
            }
        ]
    })";
    const markets::CurveStackSpec stack = markets::parseCurveStackSpec(json);
    ASSERT_EQ(stack.curves.size(), 2u);
    const markets::CurveSpec& discountSpec = stack.curves[0];
    const markets::CurveSpec& forecastSpec = stack.curves[1];
    EXPECT_FALSE(forecastSpec.hasParent);
    EXPECT_FALSE(forecastSpec.hasDiscount);
    ASSERT_EQ(forecastSpec.forecastPillars.size(), 4u);
    EXPECT_TRUE(forecastSpec.forecastPillars[0].kind == markets::PillarSpec::Kind::Deposit);
    EXPECT_TRUE(forecastSpec.forecastPillars[1].kind == markets::PillarSpec::Kind::Fra);
    EXPECT_TRUE(forecastSpec.forecastPillars[2].kind == markets::PillarSpec::Kind::Irs);
    EXPECT_TRUE(forecastSpec.forecastPillars[3].kind == markets::PillarSpec::Kind::BasisSwap);
    EXPECT_TRUE(forecastSpec.forecastPillars[1].start == datetime::Date::parse("2027-03-29"));
    EXPECT_TRUE(forecastSpec.forecastPillars[2].fixedTenor ==
                datetime::Period(1, datetime::TimeUnit::Years));
    EXPECT_TRUE(forecastSpec.forecastPillars[3].spreadOnParentLeg);

    const markets::DiscountCurve<double> discount = markets::buildCurve(stack, discountSpec);
    const auto parent = std::make_shared<const markets::DiscountCurve<double>>(discount);
    std::vector<markets::ForecastPillar> filled;
    const markets::SpreadCurve<double> forecast =
        markets::buildForecastCurve(stack, forecastSpec, parent, nullptr, {}, &filled);
    EXPECT_TRUE(forecast.parentPointer() == parent);
    EXPECT_EQ(filled.size(), forecastSpec.forecastPillars.size());
    EXPECT_TRUE(filled[2].irs.businessDayConvention == datetime::BusinessDayConvention::Following);
    EXPECT_TRUE(filled[3].basis.businessDayConvention ==
                datetime::BusinessDayConvention::Following);
    EXPECT_TRUE(filled[2].irs.floatTenor == datetime::Period(3, datetime::TimeUnit::Months));
    EXPECT_TRUE(filled[2].irs.fixedTenor == datetime::Period(1, datetime::TimeUnit::Years));
    EXPECT_EQ(filled[2].irs.paymentLag, 2);
    EXPECT_TRUE(filled[3].basis.spreadOnParentLeg);
    CHECK_CLOSE("forecast basis spread mapping", filled[3].basis.spread, 0.0012, 1e-15);

    const auto target = [](const markets::ForecastPillar& pillar) {
        switch (pillar.kind) {
            case markets::ForecastPillar::Kind::Irs:
                return pillar.irs.quote;
            case markets::ForecastPillar::Kind::BasisSwap:
                return pillar.basis.spread;
            default:
                return pillar.quote;
        }
    };
    double worstResidual = 0.0;
    for (const markets::ForecastPillar& pillar : filled) {
        const double implied = markets::impliedForecastQuote(forecast, *parent, pillar, stack.asOf,
                                                             forecastSpec.zeroDayCounter);
        CHECK_CLOSE("forecast config reprice", implied, target(pillar), 1e-10);
        const double residual = std::abs(implied - target(pillar));
        if (residual > worstResidual) {
            worstResidual = residual;
        }
    }
    ::testing::Test::RecordProperty("forecast_worst_reprice_residual",
                                    quantape::util::num(worstResidual, 6));
}

TEST(ForecastConfig, validationRejectsAndKeyResolvedCopy) {
    SCOPED_TRACE("malformed forecast pillar kind");
    const std::string malformedKind = R"({
        "asOf": "2026-09-29",
        "curves": [{
            "key": {"currency": "USD", "role": "Forecast"},
            "forecastPillars": [
                {"maturity": "2027-03-29", "kind": "Swap", "quote": 0.001}
            ]
        }]
    })";
    {
        bool threw = false;
        try {
            (void)markets::parseCurveStackSpec(malformedKind);
        } catch (const std::invalid_argument& error) {
            threw = true;
            const std::string message = error.what();
            EXPECT_NE(message.find("Deposit"), std::string::npos) << message;
            EXPECT_NE(message.find("Fra"), std::string::npos) << message;
            EXPECT_NE(message.find("Irs"), std::string::npos) << message;
            EXPECT_NE(message.find("BasisSwap"), std::string::npos) << message;
        }
        EXPECT_TRUE(threw);
    }

    SCOPED_TRACE("missing quote");
    const std::string missingQuote = R"({
        "asOf": "2026-09-29",
        "curves": [{
            "key": {"currency": "USD", "role": "Forecast"},
            "forecastPillars": [{"maturity": "2027-03-29", "kind": "Deposit"}]
        }]
    })";
    EXPECT_THROW((void)markets::parseCurveStackSpec(missingQuote), std::invalid_argument);

    SCOPED_TRACE("missing tenors");
    const std::string missingTenors = R"({
        "asOf": "2026-09-29",
        "curves": [{
            "key": {"currency": "USD", "role": "Forecast"},
            "forecastPillars": [{"maturity": "2027-03-29", "kind": "Irs", "quote": 0.043}]
        }]
    })";
    EXPECT_THROW((void)markets::parseCurveStackSpec(missingTenors), std::invalid_argument);

    const std::string buildConfig = R"({
        "asOf": "2026-09-29",
        "curves": [
            {
                "key": {"currency": "USD", "role": "Discount"},
                "pillars": [{"maturity": "2027-09-29", "kind": "Deposit", "quote": 0.04,
                             "calendar": "NoHolidays"}]
            },
            {
                "key": {"currency": "USD", "role": "Forecast"},
                "forecastPillars": [{"maturity": "2027-09-29", "kind": "Deposit",
                                     "quote": 0.041, "calendar": "NoHolidays"}]
            }
        ]
    })";
    const markets::CurveStackSpec stack = markets::parseCurveStackSpec(buildConfig);
    const auto parent = std::make_shared<const markets::DiscountCurve<double>>(
        markets::buildCurve(stack, stack.curves[0]));

    // A discount spec without forecast pillars cannot be built as a forecast curve.
    SCOPED_TRACE("discount spec as forecast");
    EXPECT_THROW((void)markets::buildForecastCurve(stack, stack.curves[0], parent),
                 std::invalid_argument);

    // A null parent is refused.
    SCOPED_TRACE("null parent");
    EXPECT_THROW((void)markets::buildForecastCurve(stack, stack.curves[1], nullptr),
                 std::invalid_argument);

    // Membership resolves by key, so a copy of the stack's forecast spec builds
    // instead of being mistaken for a colliding sibling through its address.
    SCOPED_TRACE("key-resolved copy");
    markets::CurveSpec copy = stack.curves[1];
    std::vector<markets::ForecastPillar> filled;
    const markets::SpreadCurve<double> forecast =
        markets::buildForecastCurve(stack, copy, parent, nullptr, {}, &filled);
    ASSERT_EQ(filled.size(), 1u);
    EXPECT_TRUE(forecast.parentPointer() == parent);

    SCOPED_TRACE("spread alias");
    const std::string spreadAlias = R"({
        "asOf": "2026-09-29",
        "curves": [{
            "key": {"currency": "USD", "role": "Forecast"},
            "forecastPillars": [{"maturity": "2027-03-29", "kind": "BasisSwap", "spread": 0.002,
                                 "floatTenor": {"length": 3, "unit": "Months"}}]
        }]
    })";
    const markets::CurveStackSpec aliasStack = markets::parseCurveStackSpec(spreadAlias);
    CHECK_CLOSE("forecast spread alias", aliasStack.curves.front().forecastPillars.front().quote,
                0.002, 1e-15);
}

TEST(CurveConfig, referenceCurveIdentityAndForecastFuture) {
    SCOPED_TRACE("self reference");
    const std::string selfReference = R"({
        "asOf": "2026-09-29",
        "curves": [{
            "key": {"currency": "USD", "role": "Discount"},
            "convexity": {"model": "HullWhite", "sigma": 0.01, "meanReversion": 0.05,
                          "referenceCurve": {"currency": "USD", "role": "Discount"}},
            "pillars": [{"maturity": "2027-09-29", "kind": "Deposit", "quote": 0.04}]
        }]
    })";
    expectParseReject(selfReference, "must differ");

    SCOPED_TRACE("missing reference");
    const std::string missingReference = R"({
        "asOf": "2026-09-29",
        "curves": [{
            "key": {"currency": "USD", "role": "Discount"},
            "convexity": {"model": "HullWhite", "sigma": 0.01, "meanReversion": 0.05,
                          "referenceCurve": {"currency": "EUR", "role": "Discount"}},
            "pillars": [
                {"maturity": "2027-03-29", "kind": "Deposit", "quote": 0.04,
                 "calendar": "NoHolidays"},
                {"start": "2027-06-16", "maturity": "2027-09-15", "kind": "Future",
                 "quote": 0.041, "calendar": "NoHolidays"}
            ]
        }]
    })";
    const markets::CurveStackSpec referenceStack = markets::parseCurveStackSpec(missingReference);
    {
        bool threw = false;
        std::string message;
        try {
            (void)markets::buildCurve(referenceStack, referenceStack.curves.front());
        } catch (const std::invalid_argument& error) {
            threw = true;
            message = error.what();
        }
        EXPECT_TRUE(threw);
        if (threw) {
            EXPECT_NE(message.find("EUR"), std::string::npos) << message;
        }
    }

    SCOPED_TRACE("forecast future config");
    const std::string json = R"({
        "asOf": "2026-09-29",
        "curves": [
            {
                "key": {"currency": "USD", "role": "Discount", "collateral": "USD"},
                "zeroDayCounter": "ACT/365F",
                "interpolation": {"space": "LogDiscount", "scheme": "Linear"},
                "pillars": [
                    {"maturity": "2027-03-29", "kind": "Deposit", "quote": 0.0432,
                     "quoteDayCounter": "ACT/360", "calendar": "NoHolidays"},
                    {"maturity": "2027-09-29", "kind": "OisSwap", "quote": 0.0405,
                     "quoteDayCounter": "ACT/365F", "calendar": "NoHolidays",
                     "fixedTenor": {"length": 1, "unit": "Years"}},
                    {"maturity": "2028-09-29", "kind": "OisSwap", "quote": 0.0395,
                     "quoteDayCounter": "ACT/365F", "calendar": "NoHolidays",
                     "fixedTenor": {"length": 1, "unit": "Years"}}
                ]
            },
            {
                "key": {"currency": "EUR", "role": "Forecast",
                        "indexTenor": {"length": 3, "unit": "Months"}, "collateral": "USD"},
                "zeroDayCounter": "ACT/365F",
                "interpolation": {"space": "Zero", "scheme": "Linear"},
                "forecastPillars": [
                    {"start": "2027-03-17", "maturity": "2027-06-16", "kind": "Future",
                     "style": "Averaged", "averaging": "Arithmetic", "quote": 0.041,
                     "convexityAdjustment": 0.0002, "quoteDayCounter": "ACT/360",
                     "calendar": "NoHolidays"},
                    {"start": "2027-06-16", "maturity": "2027-09-15", "kind": "Future",
                     "style": "Simple", "quote": 0.042,
                     "convexityAdjustment": 0.0003, "quoteDayCounter": "ACT/360",
                     "calendar": "NoHolidays"}
                ]
            }
        ]
    })";
    const markets::CurveStackSpec stack = markets::parseCurveStackSpec(json);
    ASSERT_EQ(stack.curves.size(), 2u);
    const markets::CurveSpec& forecastSpec = stack.curves[1];
    ASSERT_EQ(forecastSpec.forecastPillars.size(), 2u);
    const markets::PillarSpec& futureSpec = forecastSpec.forecastPillars[0];
    EXPECT_TRUE(futureSpec.kind == markets::PillarSpec::Kind::Future);
    EXPECT_TRUE(futureSpec.futureStyle == markets::FutureStyle::Averaged);
    EXPECT_TRUE(futureSpec.averagingStyle == markets::AveragingStyle::Arithmetic);
    EXPECT_TRUE(futureSpec.convexityAdjustmentSet);
    CHECK_CLOSE("future config adjustment", futureSpec.convexityAdjustment, 0.0002, 1e-15);

    const markets::DiscountCurve<double> discount = markets::buildCurve(stack, stack.curves[0]);
    const auto parent = std::make_shared<const markets::DiscountCurve<double>>(discount);
    std::vector<markets::ForecastPillar> filled;
    const markets::SpreadCurve<double> forecast =
        markets::buildForecastCurve(stack, forecastSpec, parent, nullptr, {}, &filled);
    ASSERT_EQ(filled.size(), 2u);
    EXPECT_TRUE(filled[0].futureStyle == markets::FutureStyle::Averaged);
    EXPECT_TRUE(filled[0].averagingStyle == markets::AveragingStyle::Arithmetic);
    EXPECT_TRUE(filled[1].futureStyle == markets::FutureStyle::Simple);
    for (const markets::ForecastPillar& pillar : filled) {
        CHECK_CLOSE("future config reprice",
                    markets::impliedForecastQuote(forecast, *parent, pillar, stack.asOf,
                                                  forecastSpec.zeroDayCounter),
                    pillar.quote, 1e-10);
    }

    // A future without an explicit adjustment and no curve-level model must be
    // refused instead of silently pricing with zero convexity.
    SCOPED_TRACE("missing future adjustment");
    const std::string missingAdjustment = R"({
        "asOf": "2026-09-29",
        "curves": [
            {
                "key": {"currency": "USD", "role": "Discount"},
                "pillars": [{"maturity": "2027-09-29", "kind": "Deposit", "quote": 0.04,
                             "calendar": "NoHolidays"}]
            },
            {
                "key": {"currency": "EUR", "role": "Forecast"},
                "forecastPillars": [
                    {"start": "2027-03-17", "maturity": "2027-06-16", "kind": "Future",
                     "style": "Simple", "quote": 0.041, "quoteDayCounter": "ACT/360",
                     "calendar": "NoHolidays"}
                ]
            }
        ]
    })";
    const markets::CurveStackSpec missingStack = markets::parseCurveStackSpec(missingAdjustment);
    const auto missingParent = std::make_shared<const markets::DiscountCurve<double>>(
        markets::buildCurve(missingStack, missingStack.curves[0]));
    {
        bool threw = false;
        try {
            (void)markets::buildForecastCurve(missingStack, missingStack.curves[1], missingParent);
        } catch (const std::invalid_argument& error) {
            threw = true;
            EXPECT_NE(std::string(error.what()).find("convexity"), std::string::npos)
                << error.what();
        }
        EXPECT_TRUE(threw);
    }

    // Unknown future style is rejected at parse time.
    SCOPED_TRACE("unknown future style");
    const std::string unknownStyle = R"({
        "asOf": "2026-09-29",
        "curves": [{
            "key": {"currency": "EUR", "role": "Forecast"},
            "forecastPillars": [
                {"start": "2027-03-17", "maturity": "2027-06-16", "kind": "Future",
                 "style": "Geometric", "quote": 0.041, "convexityAdjustment": 0.0,
                 "quoteDayCounter": "ACT/360", "calendar": "NoHolidays"}
            ]
        }]
    })";
    EXPECT_THROW((void)markets::parseCurveStackSpec(unknownStyle), std::invalid_argument);

    // Curve-level Hull-White model: an unadjusted future takes the
    // self-referential fixed-point adjustment and reprices.
    SCOPED_TRACE("modeled future adjustment");
    const std::string modeled = R"({
        "asOf": "2026-09-29",
        "curves": [
            {
                "key": {"currency": "USD", "role": "Discount"},
                "zeroDayCounter": "ACT/365F",
                "pillars": [{"maturity": "2027-09-29", "kind": "Deposit", "quote": 0.04,
                             "quoteDayCounter": "ACT/365F", "calendar": "NoHolidays"}]
            },
            {
                "key": {"currency": "EUR", "role": "Forecast"},
                "zeroDayCounter": "ACT/365F",
                "interpolation": {"space": "Zero", "scheme": "Linear"},
                "convexity": {"model": "HullWhite", "sigma": 0.01, "meanReversion": 0.05},
                "forecastPillars": [
                    {"start": "2027-03-17", "maturity": "2027-06-16", "kind": "Future",
                     "style": "Simple", "quote": 0.041, "quoteDayCounter": "ACT/360",
                     "calendar": "NoHolidays"}
                ]
            }
        ]
    })";
    const markets::CurveStackSpec modeledStack = markets::parseCurveStackSpec(modeled);
    EXPECT_TRUE(modeledStack.curves[1].convexity.enabled);
    const auto modeledParent = std::make_shared<const markets::DiscountCurve<double>>(
        markets::buildCurve(modeledStack, modeledStack.curves[0]));
    std::vector<markets::ForecastPillar> modeledFilled;
    const markets::SpreadCurve<double> modeledForecast = markets::buildForecastCurve(
        modeledStack, modeledStack.curves[1], modeledParent, nullptr, {}, &modeledFilled);
    ASSERT_EQ(modeledFilled.size(), 1u);
    EXPECT_GT(modeledFilled[0].convexityAdjustment, 0.0);
    CHECK_CLOSE("modeled future config reprice",
                markets::impliedForecastQuote(modeledForecast, *modeledParent, modeledFilled[0],
                                              modeledStack.asOf,
                                              modeledStack.curves[1].zeroDayCounter),
                modeledFilled[0].quote, 1e-9);
}

TEST(CurveConfig, unifiedPillarSpecRoundTrip) {
    static_assert(std::is_same_v<decltype(markets::CurveSpec::pillars),
                                 decltype(markets::CurveSpec::forecastPillars)>);

    const std::string json = R"({
        "asOf": "2026-09-29",
        "curves": [
            {
                "key": {"currency": "USD", "role": "Discount", "collateral": "USD"},
                "zeroDayCounter": "ACT/365F",
                "interpolation": {"space": "LogDiscount", "scheme": "Linear"},
                "pillars": [
                    {"maturity": "2027-03-29", "kind": "Deposit", "quote": 0.0432,
                     "quoteDayCounter": "ACT/360", "calendar": "NoHolidays"},
                    {"start": "2027-06-16", "maturity": "2027-09-15", "kind": "Future",
                     "style": "Averaged", "quote": 0.041, "convexityAdjustment": 0.0,
                     "quoteDayCounter": "ACT/360", "calendar": "NoHolidays"},
                    {"maturity": "2028-03-29", "kind": "OisSwap", "quote": 0.0405,
                     "quoteDayCounter": "ACT/365F", "calendar": "NoHolidays",
                     "fixedTenor": {"length": 1, "unit": "Years"}, "paymentLag": 2,
                     "businessDayConvention": "Following"}
                ]
            },
            {
                "key": {"currency": "EUR", "role": "Forecast",
                        "indexTenor": {"length": 3, "unit": "Months"}, "collateral": "USD"},
                "zeroDayCounter": "ACT/365F",
                "interpolation": {"space": "Zero", "scheme": "Linear"},
                "forecastPillars": [
                    {"maturity": "2027-03-29", "kind": "Deposit", "quote": 0.044,
                     "quoteDayCounter": "ACT/360", "calendar": "NoHolidays"},
                    {"start": "2027-06-16", "maturity": "2027-09-15", "kind": "Future",
                     "style": "Averaged", "averaging": "Compounded", "quote": 0.0415,
                     "convexityAdjustment": 0.0002, "quoteDayCounter": "ACT/360",
                     "calendar": "NoHolidays"},
                    {"maturity": "2028-03-29", "kind": "Irs", "quote": 0.043,
                     "floatTenor": {"length": 3, "unit": "Months"},
                     "fixedTenor": {"length": 1, "unit": "Years"},
                     "floatCalendar": "NoHolidays", "fixedCalendar": "NoHolidays",
                     "floatDayCounter": "ACT/360", "fixedDayCounter": "30/360",
                     "businessDayConvention": "Following", "paymentLag": 2,
                     "firstFixing": 0.0425},
                    {"maturity": "2029-03-29", "kind": "BasisSwap", "spread": 0.0012,
                     "floatTenor": {"length": 3, "unit": "Months"},
                     "quoteDayCounter": "ACT/360", "calendar": "NoHolidays",
                     "businessDayConvention": "Following", "spreadOnParentLeg": false}
                ]
            }
        ]
    })";
    const markets::CurveStackSpec stack = markets::parseCurveStackSpec(json);
    ASSERT_EQ(stack.curves.size(), 2u);
    const markets::CurveSpec& discountSpec = stack.curves[0];
    const markets::CurveSpec& forecastSpec = stack.curves[1];

    // The same spec type serves both sides; the kind selects the concrete
    // instrument and the side its valid values.
    ASSERT_EQ(discountSpec.pillars.size(), 3u);
    EXPECT_TRUE(discountSpec.pillars[0].kind == markets::PillarSpec::Kind::Deposit);
    EXPECT_TRUE(discountSpec.pillars[1].kind == markets::PillarSpec::Kind::Future);
    EXPECT_TRUE(discountSpec.pillars[1].convexityAdjustmentSet);
    EXPECT_TRUE(discountSpec.pillars[2].kind == markets::PillarSpec::Kind::OisSwap);
    ASSERT_EQ(forecastSpec.forecastPillars.size(), 4u);
    EXPECT_TRUE(forecastSpec.forecastPillars[0].kind == markets::PillarSpec::Kind::Deposit);
    EXPECT_TRUE(forecastSpec.forecastPillars[1].kind == markets::PillarSpec::Kind::Future);
    EXPECT_TRUE(forecastSpec.forecastPillars[1].averagingStyle ==
                markets::AveragingStyle::Compounded);
    EXPECT_TRUE(forecastSpec.forecastPillars[2].kind == markets::PillarSpec::Kind::Irs);
    EXPECT_TRUE(forecastSpec.forecastPillars[2].firstCouponFixed);
    CHECK_CLOSE("unified spec first fixing", forecastSpec.forecastPillars[2].firstCouponRate,
                0.0425, 1e-15);
    EXPECT_TRUE(forecastSpec.forecastPillars[3].kind == markets::PillarSpec::Kind::BasisSwap);
    EXPECT_FALSE(forecastSpec.forecastPillars[3].spreadOnParentLeg);
    CHECK_CLOSE("unified spec spread alias", forecastSpec.forecastPillars[3].quote, 0.0012, 1e-15);

    std::vector<markets::CurvePillar> filledDiscount;
    const markets::DiscountCurve<double> discount =
        markets::buildCurve(stack, discountSpec, {}, &filledDiscount);
    EXPECT_EQ(filledDiscount.size(), discountSpec.pillars.size());
    for (const markets::CurvePillar& pillar : filledDiscount) {
        CHECK_CLOSE("unified discount reprice", markets::impliedQuote(pillar, stack.asOf, discount),
                    pillar.quote, 1e-10);
    }

    const auto parent = std::make_shared<const markets::DiscountCurve<double>>(discount);
    std::vector<markets::ForecastPillar> filledForecast;
    const markets::SpreadCurve<double> forecast =
        markets::buildForecastCurve(stack, forecastSpec, parent, nullptr, {}, &filledForecast);
    EXPECT_EQ(filledForecast.size(), forecastSpec.forecastPillars.size());
    for (const markets::ForecastPillar& pillar : filledForecast) {
        CHECK_CLOSE("unified forecast reprice",
                    markets::impliedForecastQuote(forecast, *parent, pillar, stack.asOf,
                                                  forecastSpec.zeroDayCounter),
                    markets::forecastPillarTarget(pillar), 1e-10);
    }
}

/// Configured FX spots and settlement lags feed an XccyBasis mixed ladder
/// (FX points short end, xccy swaps long end) that builds through `buildStack`
/// and reprices every configured quote.
TEST(FxConfig, loadsFixtureBuildsStackAndReprices) {
    const markets::CurveStackSpec stack = markets::loadCurveStackSpec(fxConfigFixture());
    EXPECT_TRUE(stack.asOf == datetime::Date::parse("2026-09-29"));
    ASSERT_EQ(stack.spotLag.size(), 2u);
    EXPECT_EQ(stack.spotLag.at("USD"), 2);
    EXPECT_EQ(stack.spotLag.at("EUR"), 2);
    ASSERT_EQ(stack.fxSpots.size(), 1u);
    EXPECT_EQ(stack.fxSpots[0].descriptor.pair(), "EURUSD");
    CHECK_CLOSE("fx config spot", stack.fxSpots[0].spot, 1.10, 1e-15);
    ASSERT_EQ(stack.curves.size(), 2u);

    const markets::CurveSpec& xccySpec = stack.curves[1];
    EXPECT_TRUE(xccySpec.key.role == markets::CurveRole::XccyBasis);
    EXPECT_EQ(xccySpec.key.collateral, "USD");
    EXPECT_TRUE(xccySpec.hasParent);
    EXPECT_EQ(xccySpec.spotLag, 2);
    EXPECT_EQ(xccySpec.xccy.pair, "EURUSD");
    EXPECT_TRUE(xccySpec.xccy.notional == markets::XccyNotionalMode::Const);
    EXPECT_EQ(xccySpec.xccy.basisLeg, "Base");
    EXPECT_FALSE(xccySpec.xccy.isFxBaseCollateral);
    ASSERT_EQ(xccySpec.pillars.size(), 3u);
    EXPECT_TRUE(xccySpec.pillars[0].kind == markets::PillarSpec::Kind::FxSwap);
    EXPECT_EQ(xccySpec.pillars[0].start.serial(), 0); // resolved from spot lags at build
    EXPECT_TRUE(xccySpec.pillars[0].fxConvention == markets::QuoteConvention::Points);
    CHECK_CLOSE("fx config points", xccySpec.pillars[0].fxPoints, 0.0125, 1e-15);
    EXPECT_TRUE(xccySpec.pillars[1].kind == markets::PillarSpec::Kind::XccySwap);
    CHECK_CLOSE("xccy config spread", xccySpec.pillars[1].quote, 0.0011, 1e-15);

    const std::vector<markets::BuiltCurve> built = markets::buildStack(stack);
    ASSERT_EQ(built.size(), 2u);
    EXPECT_TRUE(built[0].key == stack.curves[0].key);
    EXPECT_TRUE(built[1].key == xccySpec.key);
    EXPECT_TRUE(built[1].role == markets::CurveRole::XccyBasis);

    const markets::DiscountCurve<double> usd = markets::buildCurve(stack, stack.curves[0]);
    // Reconstruct the built foreign curve from its node grid and zeros so the
    // pillar repricing helpers can consume it.
    const std::vector<double>& times = built[1].curve->times();
    std::vector<double> zeros(times.size(), 0.0);
    for (std::size_t i = 1; i < times.size(); ++i) {
        zeros[i] = built[1].curve->zero(times[i]);
    }
    const markets::DiscountCurve<double> eur(times, zeros, markets::InterpolationSpace::LogDiscount,
                                             markets::InterpolationScheme::Linear);
    for (std::size_t i = 0; i < times.size(); ++i) {
        CHECK_CLOSE("fx config rebuilt node", eur.discount(times[i]),
                    built[1].curve->discount(times[i]), 1e-14);
    }

    // FX pillar: the built foreign/domestic CIP ratio reprices the configured
    // points (spot-lag settlement only moves the near date, not the far CIP
    // node).
    const markets::PillarSpec& fxSpec = xccySpec.pillars[0];
    const double tFx = datetime::yearFraction(stack.asOf, fxSpec.maturity, xccySpec.zeroDayCounter);
    CHECK_CLOSE("fx config reprice", stack.fxSpots[0].spot * eur.discount(tFx) / usd.discount(tFx),
                stack.fxSpots[0].spot + fxSpec.fxPoints, 1e-10);

    // Xccy pillars reprice with the built curve as their own floating forecast.
    for (std::size_t k = 1; k < xccySpec.pillars.size(); ++k) {
        const markets::PillarSpec& pillar = xccySpec.pillars[k];
        markets::XccyPillar resolved;
        resolved.maturity = pillar.maturity;
        resolved.spread = pillar.quote;
        resolved.spreadOnForeignLeg = true;
        resolved.notional = markets::XccyNotionalMode::Const;
        resolved.foreignTenor = pillar.foreignTenor;
        resolved.domesticTenor = pillar.domesticTenor;
        resolved.foreignCalendar = pillar.foreignCalendar;
        resolved.domesticCalendar = pillar.domesticCalendar;
        resolved.foreignDayCounter = pillar.foreignDayCounter;
        resolved.domesticDayCounter = pillar.domesticDayCounter;
        resolved.foreignBusinessDayConvention = pillar.businessDayConvention;
        resolved.domesticBusinessDayConvention = pillar.businessDayConvention;
        CHECK_CLOSE("xccy config reprice",
                    markets::impliedXccyBasisSpread(eur, eur, usd, usd, resolved, stack.asOf,
                                                    xccySpec.zeroDayCounter),
                    resolved.spread, 1e-10);
    }
}

/// FX/Xccy config error cases: a missing spot is a build error, while a bad
/// quote convention or collateral leg is refused at parse time; an XccyBasis
/// curve without a parent is a build error.
TEST(FxConfig, errors) {
    SCOPED_TRACE("missing spot");
    const std::string missingSpot = R"({
        "asOf": "2026-09-29",
        "spotLag": { "USD": 2, "EUR": 2 },
        "curves": [
            { "key": {"currency": "USD", "role": "Discount"},
              "pillars": [{"maturity": "2027-09-29", "kind": "Deposit", "quote": 0.04,
                           "calendar": "NoHolidays"}] },
            { "key": {"currency": "EUR", "role": "XccyBasis", "collateral": "USD"},
              "parent": {"currency": "USD", "role": "Discount"},
              "xccy": {"pair": "EURUSD"},
              "pillars": [{"maturity": "2027-09-29", "kind": "FxSwap", "points": 0.01}] }
        ]
    })";
    const markets::CurveStackSpec stack = markets::parseCurveStackSpec(missingSpot);
    {
        bool threw = false;
        std::string message;
        try {
            (void)markets::buildStack(stack);
        } catch (const std::invalid_argument& error) {
            threw = true;
            message = error.what();
        }
        EXPECT_TRUE(threw);
        EXPECT_NE(message.find("spot"), std::string::npos) << message;
    }

    expectParseReject(
        R"({"asOf": "2026-09-29", "curves": [{
            "key": {"currency": "EUR", "role": "XccyBasis", "collateral": "USD"},
            "pillars": [{"maturity": "2027-09-29", "kind": "FxSwap", "points": 0.01,
                         "convention": "Pips"}]}]})",
        "Points");

    expectParseReject(
        R"({"asOf": "2026-09-29", "curves": [{
            "key": {"currency": "EUR", "role": "XccyBasis", "collateral": "USD"},
            "xccy": {"basisLeg": "Both"},
            "pillars": [{"maturity": "2027-09-29", "kind": "XccySwap", "spread": 0.001}]}]})",
        "Base");

    expectParseReject(
        R"({"asOf": "2026-09-29", "curves": [{
            "key": {"currency": "EUR", "role": "XccyBasis", "collateral": "USD"},
            "xccy": {"isFxBaseCollateral": "yes"},
            "pillars": [{"maturity": "2027-09-29", "kind": "FxSwap", "points": 0.01}]}]})",
        "boolean");

    SCOPED_TRACE("missing parent");
    const std::string noParent = R"({
        "asOf": "2026-09-29",
        "curves": [{
            "key": {"currency": "EUR", "role": "XccyBasis", "collateral": "USD"},
            "xccy": {"pair": "EURUSD"},
            "pillars": [{"maturity": "2027-09-29", "kind": "FxSwap", "points": 0.01}]}]})";
    const markets::CurveStackSpec parentless = markets::parseCurveStackSpec(noParent);
    {
        bool threw = false;
        std::string message;
        try {
            (void)markets::buildStack(parentless);
        } catch (const std::invalid_argument& error) {
            threw = true;
            message = error.what();
        }
        EXPECT_TRUE(threw);
        EXPECT_NE(message.find("missing parent"), std::string::npos) << message;
    }
}

TEST(StackBuilder, depthTwoParentResolutionAndReorder) {
    const std::string json = R"({
        "asOf": "2026-09-29",
        "curves": [
            {
                "key": {"currency": "USD", "role": "Discount", "collateral": "USD"},
                "zeroDayCounter": "ACT/365F",
                "interpolation": {"space": "LogDiscount", "scheme": "Linear"},
                "bootstrap": {"method": "IterativeSequential", "accuracy": 1e-14},
                "pillars": [
                    {"maturity": "2027-03-29", "kind": "Deposit", "quote": 0.0432,
                     "quoteDayCounter": "ACT/360", "calendar": "NoHolidays"},
                    {"maturity": "2027-09-29", "kind": "OisSwap", "quote": 0.0405,
                     "quoteDayCounter": "ACT/365F", "calendar": "NoHolidays",
                     "fixedTenor": {"length": 1, "unit": "Years"}},
                    {"maturity": "2028-09-29", "kind": "OisSwap", "quote": 0.0395,
                     "quoteDayCounter": "ACT/365F", "calendar": "NoHolidays",
                     "fixedTenor": {"length": 1, "unit": "Years"}},
                    {"maturity": "2029-09-29", "kind": "OisSwap", "quote": 0.039,
                     "quoteDayCounter": "ACT/365F", "calendar": "NoHolidays",
                     "fixedTenor": {"length": 1, "unit": "Years"}}
                ]
            },
            {
                "key": {"currency": "USD", "role": "Forecast",
                        "indexTenor": {"length": 3, "unit": "Months"}, "collateral": "USD"},
                "parent": {"currency": "USD", "role": "Discount", "collateral": "USD"},
                "zeroDayCounter": "ACT/365F",
                "interpolation": {"space": "Zero", "scheme": "Linear"},
                "bootstrap": {"method": "IterativeSequential", "accuracy": 1e-14},
                "forecastPillars": [
                    {"maturity": "2027-03-29", "kind": "Deposit", "quote": 0.044,
                     "quoteDayCounter": "ACT/360", "calendar": "NoHolidays"},
                    {"start": "2027-03-29", "maturity": "2027-06-29", "kind": "Fra",
                     "quote": 0.0435, "quoteDayCounter": "ACT/360", "calendar": "NoHolidays"},
                    {"maturity": "2028-03-29", "kind": "Irs", "quote": 0.043,
                     "floatTenor": {"length": 3, "unit": "Months"},
                     "fixedTenor": {"length": 1, "unit": "Years"},
                     "floatCalendar": "NoHolidays", "fixedCalendar": "NoHolidays",
                     "floatDayCounter": "ACT/360", "fixedDayCounter": "30/360",
                     "businessDayConvention": "Following", "paymentLag": 2}
                ]
            },
            {
                "key": {"currency": "USD", "role": "Forecast",
                        "indexTenor": {"length": 6, "unit": "Months"}, "collateral": "USD"},
                "parent": {"currency": "USD", "role": "Forecast",
                           "indexTenor": {"length": 3, "unit": "Months"}, "collateral": "USD"},
                "discount": {"currency": "USD", "role": "Discount", "collateral": "USD"},
                "zeroDayCounter": "ACT/365F",
                "interpolation": {"space": "Zero", "scheme": "Linear"},
                "bootstrap": {"method": "IterativeSequential", "accuracy": 1e-14},
                "forecastPillars": [
                    {"maturity": "2027-03-29", "kind": "Deposit", "quote": 0.0455,
                     "quoteDayCounter": "ACT/360", "calendar": "NoHolidays"},
                    {"maturity": "2028-03-29", "kind": "Irs", "quote": 0.045,
                     "floatTenor": {"length": 6, "unit": "Months"},
                     "fixedTenor": {"length": 1, "unit": "Years"},
                     "floatCalendar": "NoHolidays", "fixedCalendar": "NoHolidays",
                     "floatDayCounter": "ACT/360", "fixedDayCounter": "30/360",
                     "businessDayConvention": "Following", "paymentLag": 2}
                ]
            }
        ]
    })";
    const markets::CurveStackSpec stack = markets::parseCurveStackSpec(json);
    ASSERT_EQ(stack.curves.size(), 3u);
    EXPECT_FALSE(stack.curves[0].hasParent);
    EXPECT_FALSE(stack.curves[0].hasDiscount);
    EXPECT_TRUE(stack.curves[1].hasParent);
    EXPECT_TRUE(stack.curves[1].parent == stack.curves[0].key);
    EXPECT_FALSE(stack.curves[1].hasDiscount);
    EXPECT_TRUE(stack.curves[2].hasParent);
    EXPECT_TRUE(stack.curves[2].parent == stack.curves[1].key);
    EXPECT_TRUE(stack.curves[2].hasDiscount);
    EXPECT_TRUE(stack.curves[2].discount == stack.curves[0].key);

    const std::vector<markets::BuiltCurve> built = markets::buildStack(stack);
    ASSERT_EQ(built.size(), 3u);
    EXPECT_TRUE(built[0].key == stack.curves[0].key);
    EXPECT_TRUE(built[0].role == markets::CurveRole::Discount);
    EXPECT_TRUE(built[1].key == stack.curves[1].key);
    EXPECT_TRUE(built[1].role == markets::CurveRole::Forecast);
    EXPECT_TRUE(built[2].key == stack.curves[2].key);
    EXPECT_TRUE(built[2].role == markets::CurveRole::Forecast);
    const markets::BuiltCurve* rootBuilt = findBuilt(built, stack.curves[0].key);
    const markets::BuiltCurve* curve3mBuilt = findBuilt(built, stack.curves[1].key);
    const markets::BuiltCurve* curve6mBuilt = findBuilt(built, stack.curves[2].key);
    ASSERT_TRUE(rootBuilt != nullptr);
    ASSERT_TRUE(curve3mBuilt != nullptr);
    ASSERT_TRUE(curve6mBuilt != nullptr);

    std::vector<markets::CurvePillar> rootFilled;
    const markets::DiscountCurve<double> rootCurve =
        markets::buildCurve(stack, stack.curves[0], {}, &rootFilled);
    for (const markets::CurvePillar& pillar : rootFilled) {
        CHECK_CLOSE("stack root reprice", markets::impliedQuote(pillar, stack.asOf, rootCurve),
                    pillar.quote, 1e-10);
    }
    EXPECT_EQ(rootBuilt->curve->size(), rootCurve.size());
    EXPECT_TRUE(rootBuilt->curve->times() == rootCurve.times());
    EXPECT_EQ(rootBuilt->curve->zeroDayCounter().name(), "ACT/365F");
    for (double t = 0.25; t <= 3.0; t += 0.25) {
        CHECK_CLOSE("stack root handle discount", rootBuilt->curve->discount(t),
                    rootCurve.discount(t), 1e-15);
        CHECK_CLOSE("stack root handle zero", rootBuilt->curve->zero(t), rootCurve.zero(t), 1e-15);
    }
    CHECK_CLOSE("stack root handle forward", rootBuilt->curve->forward(1.0, 2.0),
                rootCurve.forward(1.0, 2.0), 1e-15);
    std::vector<double> rootWeights;
    rootBuilt->curve->zeroNodeWeights(1.5, rootWeights);
    EXPECT_EQ(rootWeights.size(), rootCurve.size());
    const markets::CurveHandle::Ptr rootCopy = markets::CurveHandle::make(rootCurve);
    CHECK_CLOSE("handle copy discount", rootCopy->discount(2.0), rootCurve.discount(2.0), 1e-15);

    // 3M: direct build over the root handle matches the stack builder output.
    std::vector<markets::ForecastPillar> filled3m;
    const markets::SpreadCurve<double, markets::CurveHandle> curve3m = markets::buildForecastCurve(
        stack, stack.curves[1], rootBuilt->curve, nullptr, {}, &filled3m);
    EXPECT_EQ(filled3m.size(), stack.curves[1].forecastPillars.size());
    for (const markets::ForecastPillar& pillar : filled3m) {
        CHECK_CLOSE("stack 3M reprice",
                    markets::impliedForecastQuote(curve3m, curve3m.parent(), pillar, stack.asOf,
                                                  stack.curves[1].zeroDayCounter),
                    markets::forecastPillarTarget(pillar), 1e-10);
    }
    for (double t = 0.25; t <= 3.0; t += 0.25) {
        CHECK_CLOSE("stack 3M handle", curve3mBuilt->curve->discount(t), curve3m.discount(t),
                    1e-15);
    }

    // 6M: parent is the 3M handle, exogenous discount is the OIS root.
    const auto rootShared = std::make_shared<const markets::DiscountCurve<double>>(rootCurve);
    std::vector<markets::ForecastPillar> filled6m;
    const markets::SpreadCurve<double, markets::CurveHandle> curve6m = markets::buildForecastCurve(
        stack, stack.curves[2], curve3mBuilt->curve, rootShared.get(), {}, &filled6m);
    EXPECT_TRUE(curve6m.parentPointer() == curve3mBuilt->curve);
    EXPECT_EQ(filled6m.size(), stack.curves[2].forecastPillars.size());
    for (const markets::ForecastPillar& pillar : filled6m) {
        CHECK_CLOSE("stack 6M reprice",
                    markets::impliedForecastQuote(curve6m, *rootBuilt->curve, pillar, stack.asOf,
                                                  stack.curves[2].zeroDayCounter),
                    markets::forecastPillarTarget(pillar), 1e-10);
    }
    for (double t = 0.25; t <= 3.0; t += 0.25) {
        CHECK_CLOSE("stack 6M handle", curve6mBuilt->curve->discount(t), curve6m.discount(t),
                    1e-15);
    }

    // The parent resolution is material: parenting the same 6M curve on the
    // root instead of the 3M curve changes the bootstrapped total curve.
    std::vector<markets::ForecastPillar> alternativeFilled;
    const markets::SpreadCurve<double, markets::CurveHandle> alternative =
        markets::buildForecastCurve(stack, stack.curves[2], rootBuilt->curve, rootShared.get(), {},
                                    &alternativeFilled);
    double maxParentDifference = 0.0;
    for (double t = 0.25; t <= 3.0; t += 0.25) {
        maxParentDifference =
            std::max(maxParentDifference, std::abs(curve6m.discount(t) - alternative.discount(t)));
    }
    EXPECT_GT(maxParentDifference, 1e-6);
    ::testing::Test::RecordProperty("stack_parent_difference",
                                    quantape::util::num(maxParentDifference, 6));

    // The topological build order is input-order independent: reversing the
    // document still yields root, 3M, 6M.
    markets::CurveStackSpec reordered = stack;
    std::reverse(reordered.curves.begin(), reordered.curves.end());
    const std::vector<markets::BuiltCurve> reorderedBuilt = markets::buildStack(reordered);
    ASSERT_EQ(reorderedBuilt.size(), 3u);
    EXPECT_TRUE(reorderedBuilt[0].key == stack.curves[0].key);
    EXPECT_TRUE(reorderedBuilt[1].key == stack.curves[1].key);
    EXPECT_TRUE(reorderedBuilt[2].key == stack.curves[2].key);
    for (double t = 0.25; t <= 3.0; t += 0.25) {
        CHECK_CLOSE("reordered stack 6M handle", reorderedBuilt[2].curve->discount(t),
                    curve6m.discount(t), 1e-15);
    }

    const markets::CurveHandle::Ptr nullHandle;
    EXPECT_THROW((void)markets::CurveHandle::make(nullHandle), std::invalid_argument);
}

TEST(StackBuilder, errorsAndConvexityReference) {
    SCOPED_TRACE("missing parent");
    expectBuildReject(markets::parseCurveStackSpec(R"({
            "asOf": "2026-09-29",
            "curves": [{
                "key": {"currency": "USD", "role": "Forecast",
                        "indexTenor": {"length": 3, "unit": "Months"}},
                "forecastPillars": [{"maturity": "2027-03-29", "kind": "Deposit",
                                     "quote": 0.044, "quoteDayCounter": "ACT/360",
                                     "calendar": "NoHolidays"}]
            }]
        })"),
                      "missing parent");

    SCOPED_TRACE("discount with parent");
    expectBuildReject(markets::parseCurveStackSpec(R"({
            "asOf": "2026-09-29",
            "curves": [{
                "key": {"currency": "USD", "role": "Discount"},
                "parent": {"currency": "USD", "role": "Discount"},
                "pillars": [{"maturity": "2027-03-29", "kind": "Deposit", "quote": 0.04,
                             "calendar": "NoHolidays"}]
            }]
        })"),
                      "must not declare a parent");

    SCOPED_TRACE("self-reference");
    expectBuildReject(markets::parseCurveStackSpec(R"({
            "asOf": "2026-09-29",
            "curves": [{
                "key": {"currency": "USD", "role": "Forecast",
                        "indexTenor": {"length": 3, "unit": "Months"}},
                "parent": {"currency": "USD", "role": "Forecast",
                           "indexTenor": {"length": 3, "unit": "Months"}},
                "forecastPillars": [{"maturity": "2027-03-29", "kind": "Deposit",
                                     "quote": 0.044, "quoteDayCounter": "ACT/360",
                                     "calendar": "NoHolidays"}]
            }]
        })"),
                      "self-reference");

    SCOPED_TRACE("parent not in stack");
    expectBuildReject(markets::parseCurveStackSpec(R"({
            "asOf": "2026-09-29",
            "curves": [{
                "key": {"currency": "USD", "role": "Forecast",
                        "indexTenor": {"length": 3, "unit": "Months"}},
                "parent": {"currency": "EUR", "role": "Discount"},
                "forecastPillars": [{"maturity": "2027-03-29", "kind": "Deposit",
                                     "quote": 0.044, "quoteDayCounter": "ACT/360",
                                     "calendar": "NoHolidays"}]
            }]
        })"),
                      "not found in the stack");

    SCOPED_TRACE("parent cycle");
    expectBuildReject(markets::parseCurveStackSpec(R"({
            "asOf": "2026-09-29",
            "curves": [
                {
                    "key": {"currency": "USD", "role": "Forecast",
                            "indexTenor": {"length": 3, "unit": "Months"}},
                    "parent": {"currency": "USD", "role": "Forecast",
                               "indexTenor": {"length": 6, "unit": "Months"}},
                    "forecastPillars": [{"maturity": "2027-03-29", "kind": "Deposit",
                                         "quote": 0.044, "quoteDayCounter": "ACT/360",
                                         "calendar": "NoHolidays"}]
                },
                {
                    "key": {"currency": "USD", "role": "Forecast",
                            "indexTenor": {"length": 6, "unit": "Months"}},
                    "parent": {"currency": "USD", "role": "Forecast",
                               "indexTenor": {"length": 3, "unit": "Months"}},
                    "forecastPillars": [{"maturity": "2027-03-29", "kind": "Deposit",
                                         "quote": 0.045, "quoteDayCounter": "ACT/360",
                                         "calendar": "NoHolidays"}]
                }
            ]
        })"),
                      "cycle");

    SCOPED_TRACE("reference must be a discount curve");
    expectBuildReject(markets::parseCurveStackSpec(R"({
            "asOf": "2026-09-29",
            "curves": [
                {
                    "key": {"currency": "USD", "role": "Discount"},
                    "pillars": [{"maturity": "2027-03-29", "kind": "Deposit", "quote": 0.04,
                                 "calendar": "NoHolidays"}]
                },
                {
                    "key": {"currency": "USD", "role": "Forecast",
                            "indexTenor": {"length": 3, "unit": "Months"}},
                    "parent": {"currency": "USD", "role": "Discount"},
                    "forecastPillars": [{"maturity": "2027-03-29", "kind": "Deposit",
                                         "quote": 0.044, "quoteDayCounter": "ACT/360",
                                         "calendar": "NoHolidays"}]
                },
                {
                    "key": {"currency": "EUR", "role": "Forecast",
                            "indexTenor": {"length": 3, "unit": "Months"}},
                    "parent": {"currency": "USD", "role": "Discount"},
                    "convexity": {"model": "HullWhite", "sigma": 0.01, "meanReversion": 0.05,
                                  "referenceCurve": {"currency": "USD", "role": "Forecast",
                                                     "indexTenor": {"length": 3,
                                                                    "unit": "Months"}}},
                    "forecastPillars": [{"start": "2027-03-17", "maturity": "2027-06-16",
                                         "kind": "Future", "style": "Simple", "quote": 0.041,
                                         "quoteDayCounter": "ACT/360", "calendar": "NoHolidays"}]
                }
            ]
        })"),
                      "reference must be a discount curve");

    SCOPED_TRACE("exogenous discount must be a discount curve");
    expectBuildReject(markets::parseCurveStackSpec(R"({
            "asOf": "2026-09-29",
            "curves": [
                {
                    "key": {"currency": "USD", "role": "Discount"},
                    "pillars": [{"maturity": "2027-03-29", "kind": "Deposit", "quote": 0.04,
                                 "calendar": "NoHolidays"}]
                },
                {
                    "key": {"currency": "USD", "role": "Forecast",
                            "indexTenor": {"length": 3, "unit": "Months"}},
                    "parent": {"currency": "USD", "role": "Discount"},
                    "forecastPillars": [{"maturity": "2027-03-29", "kind": "Deposit",
                                         "quote": 0.044, "quoteDayCounter": "ACT/360",
                                         "calendar": "NoHolidays"}]
                },
                {
                    "key": {"currency": "USD", "role": "Forecast",
                            "indexTenor": {"length": 6, "unit": "Months"}},
                    "parent": {"currency": "USD", "role": "Forecast",
                               "indexTenor": {"length": 3, "unit": "Months"}},
                    "discount": {"currency": "USD", "role": "Forecast",
                                 "indexTenor": {"length": 3, "unit": "Months"}},
                    "forecastPillars": [{"maturity": "2027-03-29", "kind": "Deposit",
                                         "quote": 0.045, "quoteDayCounter": "ACT/360",
                                         "calendar": "NoHolidays"}]
                }
            ]
        })"),
                      "exogenous discount must be a discount curve");

    SCOPED_TRACE("convexity reference ordering");
    const std::string json = R"({
        "asOf": "2026-09-29",
        "curves": [
            {
                "key": {"currency": "EUR", "role": "Discount", "collateral": "EUR"},
                "zeroDayCounter": "ACT/365F",
                "convexity": {"model": "HullWhite", "sigma": 0.01, "meanReversion": 0.05,
                              "referenceCurve": {"currency": "USD", "role": "Discount",
                                                 "collateral": "USD"}},
                "pillars": [
                    {"maturity": "2027-03-29", "kind": "Deposit", "quote": 0.041,
                     "quoteDayCounter": "ACT/360", "calendar": "NoHolidays"},
                    {"start": "2027-06-16", "maturity": "2027-09-15", "kind": "Future",
                     "style": "Simple", "quote": 0.0415, "quoteDayCounter": "ACT/360",
                     "calendar": "NoHolidays"}
                ]
            },
            {
                "key": {"currency": "USD", "role": "Discount", "collateral": "USD"},
                "zeroDayCounter": "ACT/365F",
                "pillars": [
                    {"maturity": "2027-03-29", "kind": "Deposit", "quote": 0.0432,
                     "quoteDayCounter": "ACT/360", "calendar": "NoHolidays"},
                    {"maturity": "2027-09-29", "kind": "OisSwap", "quote": 0.0405,
                     "quoteDayCounter": "ACT/365F", "calendar": "NoHolidays",
                     "fixedTenor": {"length": 1, "unit": "Years"}}
                ]
            }
        ]
    })";
    const markets::CurveStackSpec stack = markets::parseCurveStackSpec(json);
    const std::vector<markets::BuiltCurve> built = markets::buildStack(stack);
    ASSERT_EQ(built.size(), 2u);
    // The convexity reference forces the USD curve to build first even though
    // the EUR spec leads the document.
    EXPECT_TRUE(built[0].key == stack.curves[1].key);
    EXPECT_TRUE(built[1].key == stack.curves[0].key);

    const markets::DiscountCurve<double> usd = markets::buildCurve(stack, stack.curves[1]);
    const std::vector<markets::CurveReference> references{{stack.curves[1].key, &usd}};
    std::vector<markets::CurvePillar> eurFilled;
    const markets::DiscountCurve<double> eur =
        markets::buildCurve(stack, stack.curves[0], references, &eurFilled);
    ASSERT_EQ(eurFilled.size(), 2u);
    EXPECT_GT(eurFilled[1].convexityAdjustment, 0.0);
    for (const markets::CurvePillar& pillar : eurFilled) {
        CHECK_CLOSE("stack reference reprice", markets::impliedQuote(pillar, stack.asOf, eur),
                    pillar.quote, 1e-10);
    }
    for (double t = 0.25; t <= 2.0; t += 0.25) {
        CHECK_CLOSE("stack reference curve", built[1].curve->discount(t), eur.discount(t), 1e-14);
    }
}
