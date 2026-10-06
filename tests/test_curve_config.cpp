#include "quantape/log/Log.h"
#include "quantape/markets/Curves/CurveConfig.h"
#include "quantape/markets/Curves/FxSwapBuilder.h"
#include "quantape/markets/Curves/XccyBasisBuilder.h"
#include "quantape/util/Check.h"

#include <algorithm>
#include <cmath>
#include <memory>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

using namespace quantape;

namespace {

void testLoadAndBootstrap() {
    const markets::CurveStackSpec stack = markets::loadCurveStackSpec(CURVE_CONFIG_FIXTURE);
    CHECK(stack.asOf == datetime::Date::parse("2026-09-29"));
    CHECK(stack.curves.size() == 1);

    const markets::CurveSpec& spec = stack.curves.front();
    CHECK(spec.key.currency == "USD");
    CHECK(spec.key.role == markets::CurveRole::Discount);
    CHECK(spec.key.collateral == "USD");
    CHECK(spec.space == markets::InterpolationSpace::LogDiscount);
    CHECK(spec.scheme == markets::InterpolationScheme::Linear);
    CHECK(spec.pillars.size() == 6);
    CHECK(spec.pillars.front().kind == markets::PillarSpec::Kind::Deposit);
    CHECK(spec.pillars.back().kind == markets::PillarSpec::Kind::OisSwap);

    std::vector<markets::CurvePillar> filled;
    const markets::DiscountCurve<double> curve = markets::buildCurve(stack, spec, {}, &filled);
    CHECK(curve.size() == spec.pillars.size() + 1);
    CHECK(curve.discount(0.0) == 1.0);

    CHECK(filled.size() == spec.pillars.size());
    for (const markets::CurvePillar& pillar : filled) {
        util::checkClose("config reprice", markets::impliedQuote(pillar, stack.asOf, curve),
                         pillar.quote, 1e-10);
    }

    double previous = curve.discount(0.0);
    for (double t = 0.25; t <= 10.0; t += 0.25) {
        const double current = curve.discount(t);
        CHECK(current < previous);
        previous = current;
    }
}

void testValidation() {
    bool threw = false;
    try {
        (void)markets::parseCurveStackSpec("{ not json ");
    } catch (const std::invalid_argument&) {
        threw = true;
    }
    CHECK(threw);

    threw = false;
    try {
        (void)markets::parseCurveStackSpec(
            R"({"asOf": "2026-09-29", "curves": [{"key": {"currency": "USD", "role": "Discount"},
            "interpolation": {"scheme": "RatSpline"}, "pillars": [{"maturity": "2027-09-29",
            "kind": "Deposit", "quote": 0.04}]}]})");
    } catch (const std::invalid_argument& error) {
        threw = true;
        CHECK(std::string(error.what()).find("HymanSpline") != std::string::npos);
    }
    CHECK(threw);

    threw = false;
    try {
        (void)markets::parseCurveStackSpec(R"({"asOf": "2026-09-29", "curves": []})");
    } catch (const std::invalid_argument&) {
        threw = true;
    }
    CHECK(threw);

    util::checkClose(
        "day counter name",
        markets::dayCounterFromName("ACT/365F")
            .yearFraction(datetime::Date::parse("2026-01-01"), datetime::Date::parse("2027-01-01")),
        1.0, 1e-12);
    CHECK(markets::curveRoleToName(markets::CurveRole::XccyBasis) == "XccyBasis");
}

void testDefaultFixedDayCounter() {
    const datetime::DayCounter bondBasis = markets::dayCounterFromName("30/360 BondBasis");
    CHECK(bondBasis.convention() == datetime::DayCount::Thirty360BondBasis);
    CHECK(markets::PillarSpec{}.fixedDayCounter.convention() == bondBasis.convention());

    // The name emitted by the day-count catalogue parses back to the default.
    const std::string emitted(
        datetime::dayCountName(markets::PillarSpec{}.fixedDayCounter.convention()));
    CHECK(markets::dayCounterFromName(emitted).convention() == bondBasis.convention());

    // 30E/360 is a distinct convention with its own config name.
    const datetime::DayCounter euroBond = markets::dayCounterFromName("30E/360");
    CHECK(euroBond.convention() == datetime::DayCount::ThirtyE360);
    CHECK(euroBond.convention() != bondBasis.convention());

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
    CHECK(stack.curves.front().forecastPillars.size() == 2);
    CHECK(stack.curves.front().forecastPillars[0].fixedDayCounter.convention() ==
          datetime::DayCount::Thirty360BondBasis);
    CHECK(stack.curves.front().forecastPillars[1].fixedDayCounter.convention() ==
          stack.curves.front().forecastPillars[0].fixedDayCounter.convention());
}

void testHymanSplineScheme() {
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
    CHECK(spec.scheme == markets::InterpolationScheme::HymanSpline);
    CHECK(markets::interpolationSchemeName(spec.scheme) == "HymanSpline");

    std::vector<markets::CurvePillar> filled;
    const markets::DiscountCurve<double> curve = markets::buildCurve(stack, spec, {}, &filled);
    CHECK(curve.scheme() == markets::InterpolationScheme::HymanSpline);
    for (const markets::CurvePillar& pillar : filled) {
        util::checkClose("hyman config reprice", markets::impliedQuote(pillar, stack.asOf, curve),
                         pillar.quote, 1e-10);
    }
    // The filtered log-discounts keep the discount factors monotone.
    double previous = curve.discount(0.0);
    for (double t = 0.25; t <= 3.0; t += 0.25) {
        const double current = curve.discount(t);
        CHECK(current < previous + 1e-14);
        previous = current;
    }
}

void testAveragedCompoundedConfig() {
    const std::string json = R"({
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
    const markets::CurveStackSpec stack = markets::parseCurveStackSpec(json);
    const markets::CurveSpec& spec = stack.curves.front();
    CHECK(spec.pillars.size() == 1);
    CHECK(spec.pillars.front().futureStyle == markets::FutureStyle::Averaged);
    CHECK(spec.pillars.front().averagingStyle == markets::AveragingStyle::Compounded);
    std::vector<markets::CurvePillar> filled;
    const markets::DiscountCurve<double> curve = markets::buildCurve(stack, spec, {}, &filled);
    const markets::PillarSpec& pillar = spec.pillars.front();
    CHECK(filled.size() == 1);
    util::checkClose("averaged compounded config reprice",
                     markets::impliedQuote(filled.front(), stack.asOf, curve), pillar.quote, 1e-9);

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
    bool threw = false;
    try {
        (void)markets::parseCurveStackSpec(wrongStyle);
    } catch (const std::invalid_argument&) {
        threw = true;
    }
    CHECK(threw);

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
    threw = false;
    try {
        (void)markets::parseCurveStackSpec(badValue);
    } catch (const std::invalid_argument&) {
        threw = true;
    }
    CHECK(threw);
}

void testForecastCurveBootstrap() {
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
    CHECK(stack.curves.size() == 2);
    const markets::CurveSpec& discountSpec = stack.curves[0];
    const markets::CurveSpec& forecastSpec = stack.curves[1];
    CHECK(!forecastSpec.hasParent);
    CHECK(!forecastSpec.hasDiscount);
    CHECK(forecastSpec.forecastPillars.size() == 4);
    CHECK(forecastSpec.forecastPillars[0].kind == markets::PillarSpec::Kind::Deposit);
    CHECK(forecastSpec.forecastPillars[1].kind == markets::PillarSpec::Kind::Fra);
    CHECK(forecastSpec.forecastPillars[2].kind == markets::PillarSpec::Kind::Irs);
    CHECK(forecastSpec.forecastPillars[3].kind == markets::PillarSpec::Kind::BasisSwap);
    CHECK(forecastSpec.forecastPillars[1].start == datetime::Date::parse("2027-03-29"));
    CHECK(forecastSpec.forecastPillars[2].fixedTenor ==
          datetime::Period(1, datetime::TimeUnit::Years));
    CHECK(forecastSpec.forecastPillars[3].spreadOnParentLeg);

    const markets::DiscountCurve<double> discount = markets::buildCurve(stack, discountSpec);
    const auto parent = std::make_shared<const markets::DiscountCurve<double>>(discount);
    std::vector<markets::ForecastPillar> filled;
    const markets::SpreadCurve<double> forecast =
        markets::buildForecastCurve(stack, forecastSpec, parent, nullptr, {}, &filled);
    CHECK(forecast.parentPointer() == parent);
    CHECK(filled.size() == forecastSpec.forecastPillars.size());
    CHECK(filled[2].irs.businessDayConvention == datetime::BusinessDayConvention::Following);
    CHECK(filled[3].basis.businessDayConvention == datetime::BusinessDayConvention::Following);
    CHECK(filled[2].irs.floatTenor == datetime::Period(3, datetime::TimeUnit::Months));
    CHECK(filled[2].irs.fixedTenor == datetime::Period(1, datetime::TimeUnit::Years));
    CHECK(filled[2].irs.paymentLag == 2);
    CHECK(filled[3].basis.spreadOnParentLeg);
    util::checkClose("forecast basis spread mapping", filled[3].basis.spread, 0.0012, 1e-15);

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
        util::checkClose("forecast config reprice", implied, target(pillar), 1e-10);
        const double residual = std::abs(implied - target(pillar));
        if (residual > worstResidual) {
            worstResidual = residual;
        }
    }
    QTA_LOG_INFO("test", "forecast config mixed-pillar worst reprice residual {:.3e}",
                 worstResidual);
}

void testForecastCurveValidation() {
    const std::string malformedKind = R"({
        "asOf": "2026-09-29",
        "curves": [{
            "key": {"currency": "USD", "role": "Forecast"},
            "forecastPillars": [
                {"maturity": "2027-03-29", "kind": "Swap", "quote": 0.001}
            ]
        }]
    })";
    bool threw = false;
    try {
        (void)markets::parseCurveStackSpec(malformedKind);
    } catch (const std::invalid_argument& error) {
        threw = true;
        const std::string message = error.what();
        CHECK(message.find("Deposit") != std::string::npos);
        CHECK(message.find("Fra") != std::string::npos);
        CHECK(message.find("Irs") != std::string::npos);
        CHECK(message.find("BasisSwap") != std::string::npos);
    }
    CHECK(threw);

    const std::string missingQuote = R"({
        "asOf": "2026-09-29",
        "curves": [{
            "key": {"currency": "USD", "role": "Forecast"},
            "forecastPillars": [{"maturity": "2027-03-29", "kind": "Deposit"}]
        }]
    })";
    threw = false;
    try {
        (void)markets::parseCurveStackSpec(missingQuote);
    } catch (const std::invalid_argument&) {
        threw = true;
    }
    CHECK(threw);

    const std::string missingTenors = R"({
        "asOf": "2026-09-29",
        "curves": [{
            "key": {"currency": "USD", "role": "Forecast"},
            "forecastPillars": [{"maturity": "2027-03-29", "kind": "Irs", "quote": 0.043}]
        }]
    })";
    threw = false;
    try {
        (void)markets::parseCurveStackSpec(missingTenors);
    } catch (const std::invalid_argument&) {
        threw = true;
    }
    CHECK(threw);

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
    threw = false;
    try {
        (void)markets::buildForecastCurve(stack, stack.curves[0], parent);
    } catch (const std::invalid_argument&) {
        threw = true;
    }
    CHECK(threw);

    // A null parent is refused.
    threw = false;
    try {
        (void)markets::buildForecastCurve(stack, stack.curves[1], nullptr);
    } catch (const std::invalid_argument&) {
        threw = true;
    }
    CHECK(threw);

    // Membership resolves by key, so a copy of the stack's forecast spec builds
    // instead of being mistaken for a colliding sibling through its address.
    markets::CurveSpec copy = stack.curves[1];
    std::vector<markets::ForecastPillar> filled;
    const markets::SpreadCurve<double> forecast =
        markets::buildForecastCurve(stack, copy, parent, nullptr, {}, &filled);
    CHECK(filled.size() == 1);
    CHECK(forecast.parentPointer() == parent);

    const std::string spreadAlias = R"({
        "asOf": "2026-09-29",
        "curves": [{
            "key": {"currency": "USD", "role": "Forecast"},
            "forecastPillars": [{"maturity": "2027-03-29", "kind": "BasisSwap", "spread": 0.002,
                                 "floatTenor": {"length": 3, "unit": "Months"}}]
        }]
    })";
    const markets::CurveStackSpec aliasStack = markets::parseCurveStackSpec(spreadAlias);
    util::checkClose("forecast spread alias",
                     aliasStack.curves.front().forecastPillars.front().quote, 0.002, 1e-15);
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
    CHECK(threw);
    if (threw) {
        CHECK(message.find(needle) != std::string::npos);
    }
}

void testRoleListInvariant() {
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

    const std::string forecastWithPillars = R"({
        "asOf": "2026-09-29",
        "curves": [{
            "key": {"currency": "USD", "role": "Forecast"},
            "pillars": [{"maturity": "2027-09-29", "kind": "Deposit", "quote": 0.04,
                         "calendar": "NoHolidays"}]
        }]
    })";
    expectParseReject(forecastWithPillars, "forecastPillars");

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
    const std::string xccyRole = R"({
        "asOf": "2026-09-29",
        "curves": [{
            "key": {"currency": "USD", "role": "XccyBasis"},
            "pillars": [{"maturity": "2027-09-29", "kind": "Deposit", "quote": 0.04,
                         "calendar": "NoHolidays"}]
        }]
    })";
    expectParseReject(xccyRole, "FxSwap");

    const std::string turnRole = R"({
        "asOf": "2026-09-29",
        "curves": [{
            "key": {"currency": "USD", "role": "TurnOverlay"},
            "pillars": [{"maturity": "2027-09-29", "kind": "Deposit", "quote": 0.04,
                         "calendar": "NoHolidays"}]
        }]
    })";
    expectParseReject(turnRole, "not configurable");
}

void testDuplicateCurveKeys() {
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
}

void testTypedFieldErrors() {
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
}

void testUnknownEnumMessages() {
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
}

void testForecastInterpolationRejects() {
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
    const std::string omittedSpace = R"({
        "asOf": "2026-09-29",
        "curves": [{
            "key": {"currency": "USD", "role": "Forecast"},
            "forecastPillars": [{"maturity": "2027-03-29", "kind": "Deposit", "quote": 0.001}]
        }]
    })";
    const markets::CurveStackSpec stack = markets::parseCurveStackSpec(omittedSpace);
    CHECK(stack.curves.front().space == markets::InterpolationSpace::Zero);
    CHECK(stack.curves.front().switchIndex == 1);
}

void testReferenceCurveIdentity() {
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
    const markets::CurveStackSpec stack = markets::parseCurveStackSpec(missingReference);
    bool threw = false;
    std::string message;
    try {
        (void)markets::buildCurve(stack, stack.curves.front());
    } catch (const std::invalid_argument& error) {
        threw = true;
        message = error.what();
    }
    CHECK(threw);
    if (threw) {
        CHECK(message.find("EUR") != std::string::npos);
    }
}

void testForecastFutureConfig() {
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
    CHECK(stack.curves.size() == 2);
    const markets::CurveSpec& forecastSpec = stack.curves[1];
    CHECK(forecastSpec.forecastPillars.size() == 2);
    const markets::PillarSpec& futureSpec = forecastSpec.forecastPillars[0];
    CHECK(futureSpec.kind == markets::PillarSpec::Kind::Future);
    CHECK(futureSpec.futureStyle == markets::FutureStyle::Averaged);
    CHECK(futureSpec.averagingStyle == markets::AveragingStyle::Arithmetic);
    CHECK(futureSpec.convexityAdjustmentSet);
    util::checkClose("future config adjustment", futureSpec.convexityAdjustment, 0.0002, 1e-15);

    const markets::DiscountCurve<double> discount = markets::buildCurve(stack, stack.curves[0]);
    const auto parent = std::make_shared<const markets::DiscountCurve<double>>(discount);
    std::vector<markets::ForecastPillar> filled;
    const markets::SpreadCurve<double> forecast =
        markets::buildForecastCurve(stack, forecastSpec, parent, nullptr, {}, &filled);
    CHECK(filled.size() == 2);
    CHECK(filled[0].futureStyle == markets::FutureStyle::Averaged);
    CHECK(filled[0].averagingStyle == markets::AveragingStyle::Arithmetic);
    CHECK(filled[1].futureStyle == markets::FutureStyle::Simple);
    for (const markets::ForecastPillar& pillar : filled) {
        util::checkClose("future config reprice",
                         markets::impliedForecastQuote(forecast, *parent, pillar, stack.asOf,
                                                       forecastSpec.zeroDayCounter),
                         pillar.quote, 1e-10);
    }

    // A future without an explicit adjustment and no curve-level model must be
    // refused instead of silently pricing with zero convexity.
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
    bool threw = false;
    try {
        (void)markets::buildForecastCurve(missingStack, missingStack.curves[1], missingParent);
    } catch (const std::invalid_argument& error) {
        threw = true;
        CHECK(std::string(error.what()).find("convexity") != std::string::npos);
    }
    CHECK(threw);

    // Unknown future style is rejected at parse time.
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
    threw = false;
    try {
        (void)markets::parseCurveStackSpec(unknownStyle);
    } catch (const std::invalid_argument&) {
        threw = true;
    }
    CHECK(threw);

    // Curve-level Hull-White model: an unadjusted future takes the
    // self-referential fixed-point adjustment and reprices.
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
    CHECK(modeledStack.curves[1].convexity.enabled);
    const auto modeledParent = std::make_shared<const markets::DiscountCurve<double>>(
        markets::buildCurve(modeledStack, modeledStack.curves[0]));
    std::vector<markets::ForecastPillar> modeledFilled;
    const markets::SpreadCurve<double> modeledForecast = markets::buildForecastCurve(
        modeledStack, modeledStack.curves[1], modeledParent, nullptr, {}, &modeledFilled);
    CHECK(modeledFilled.size() == 1);
    CHECK(modeledFilled[0].convexityAdjustment > 0.0);
    util::checkClose("modeled future config reprice",
                     markets::impliedForecastQuote(modeledForecast, *modeledParent,
                                                   modeledFilled[0], modeledStack.asOf,
                                                   modeledStack.curves[1].zeroDayCounter),
                     modeledFilled[0].quote, 1e-9);
}

void testUnifiedPillarSpec() {
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
    CHECK(stack.curves.size() == 2);
    const markets::CurveSpec& discountSpec = stack.curves[0];
    const markets::CurveSpec& forecastSpec = stack.curves[1];

    // The same spec type serves both sides; the kind selects the concrete
    // instrument and the side its valid values.
    CHECK(discountSpec.pillars.size() == 3);
    CHECK(discountSpec.pillars[0].kind == markets::PillarSpec::Kind::Deposit);
    CHECK(discountSpec.pillars[1].kind == markets::PillarSpec::Kind::Future);
    CHECK(discountSpec.pillars[1].convexityAdjustmentSet);
    CHECK(discountSpec.pillars[2].kind == markets::PillarSpec::Kind::OisSwap);
    CHECK(forecastSpec.forecastPillars.size() == 4);
    CHECK(forecastSpec.forecastPillars[0].kind == markets::PillarSpec::Kind::Deposit);
    CHECK(forecastSpec.forecastPillars[1].kind == markets::PillarSpec::Kind::Future);
    CHECK(forecastSpec.forecastPillars[1].averagingStyle == markets::AveragingStyle::Compounded);
    CHECK(forecastSpec.forecastPillars[2].kind == markets::PillarSpec::Kind::Irs);
    CHECK(forecastSpec.forecastPillars[2].firstCouponFixed);
    util::checkClose("unified spec first fixing", forecastSpec.forecastPillars[2].firstCouponRate,
                     0.0425, 1e-15);
    CHECK(forecastSpec.forecastPillars[3].kind == markets::PillarSpec::Kind::BasisSwap);
    CHECK(!forecastSpec.forecastPillars[3].spreadOnParentLeg);
    util::checkClose("unified spec spread alias", forecastSpec.forecastPillars[3].quote, 0.0012,
                     1e-15);

    std::vector<markets::CurvePillar> filledDiscount;
    const markets::DiscountCurve<double> discount =
        markets::buildCurve(stack, discountSpec, {}, &filledDiscount);
    CHECK(filledDiscount.size() == discountSpec.pillars.size());
    for (const markets::CurvePillar& pillar : filledDiscount) {
        util::checkClose("unified discount reprice",
                         markets::impliedQuote(pillar, stack.asOf, discount), pillar.quote, 1e-10);
    }

    const auto parent = std::make_shared<const markets::DiscountCurve<double>>(discount);
    std::vector<markets::ForecastPillar> filledForecast;
    const markets::SpreadCurve<double> forecast =
        markets::buildForecastCurve(stack, forecastSpec, parent, nullptr, {}, &filledForecast);
    CHECK(filledForecast.size() == forecastSpec.forecastPillars.size());
    for (const markets::ForecastPillar& pillar : filledForecast) {
        util::checkClose("unified forecast reprice",
                         markets::impliedForecastQuote(forecast, *parent, pillar, stack.asOf,
                                                       forecastSpec.zeroDayCounter),
                         markets::forecastPillarTarget(pillar), 1e-10);
    }
}

/// Configured FX spots and settlement lags feed an XccyBasis mixed ladder
/// (FX points short end, xccy swaps long end) that builds through `buildStack`
/// and reprices every configured quote.
void testFxCurveConfig() {
    const markets::CurveStackSpec stack = markets::loadCurveStackSpec(FX_CONFIG_FIXTURE);
    CHECK(stack.asOf == datetime::Date::parse("2026-09-29"));
    CHECK(stack.spotLag.size() == 2);
    CHECK(stack.spotLag.at("USD") == 2);
    CHECK(stack.spotLag.at("EUR") == 2);
    CHECK(stack.fxSpots.size() == 1);
    CHECK(stack.fxSpots[0].descriptor.pair() == "EURUSD");
    util::checkClose("fx config spot", stack.fxSpots[0].spot, 1.10, 1e-15);
    CHECK(stack.curves.size() == 2);

    const markets::CurveSpec& xccySpec = stack.curves[1];
    CHECK(xccySpec.key.role == markets::CurveRole::XccyBasis);
    CHECK(xccySpec.key.collateral == "USD");
    CHECK(xccySpec.hasParent);
    CHECK(xccySpec.spotLag == 2);
    CHECK(xccySpec.xccy.pair == "EURUSD");
    CHECK(xccySpec.xccy.notional == markets::XccyNotionalMode::Const);
    CHECK(xccySpec.xccy.basisLeg == "Base");
    CHECK(!xccySpec.xccy.isFxBaseCollateral);
    CHECK(xccySpec.pillars.size() == 3);
    CHECK(xccySpec.pillars[0].kind == markets::PillarSpec::Kind::FxSwap);
    CHECK(xccySpec.pillars[0].start.serial() == 0); // resolved from spot lags at build
    CHECK(xccySpec.pillars[0].fxConvention == markets::QuoteConvention::Points);
    util::checkClose("fx config points", xccySpec.pillars[0].fxPoints, 0.0125, 1e-15);
    CHECK(xccySpec.pillars[1].kind == markets::PillarSpec::Kind::XccySwap);
    util::checkClose("xccy config spread", xccySpec.pillars[1].quote, 0.0011, 1e-15);

    const std::vector<markets::BuiltCurve> built = markets::buildStack(stack);
    CHECK(built.size() == 2);
    CHECK(built[0].key == stack.curves[0].key);
    CHECK(built[1].key == xccySpec.key);
    CHECK(built[1].role == markets::CurveRole::XccyBasis);

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
        util::checkClose("fx config rebuilt node", eur.discount(times[i]),
                         built[1].curve->discount(times[i]), 1e-14);
    }

    // FX pillar: the built foreign/domestic CIP ratio reprices the configured
    // points (spot-lag settlement only moves the near date, not the far CIP
    // node).
    const markets::PillarSpec& fxSpec = xccySpec.pillars[0];
    const double tFx = datetime::yearFraction(stack.asOf, fxSpec.maturity, xccySpec.zeroDayCounter);
    util::checkClose("fx config reprice",
                     stack.fxSpots[0].spot * eur.discount(tFx) / usd.discount(tFx),
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
        util::checkClose("xccy config reprice",
                         markets::impliedXccyBasisSpread(eur, eur, usd, usd, resolved, stack.asOf,
                                                         xccySpec.zeroDayCounter),
                         resolved.spread, 1e-10);
    }
}

/// FX/Xccy config error cases: a missing spot is a build error, while a bad
/// quote convention or collateral leg is refused at parse time; an XccyBasis
/// curve without a parent is a build error.
void testFxCurveConfigErrors() {
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
    bool threw = false;
    std::string message;
    try {
        (void)markets::buildStack(stack);
    } catch (const std::invalid_argument& error) {
        threw = true;
        message = error.what();
    }
    CHECK(threw);
    CHECK(message.find("spot") != std::string::npos);

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

    const std::string noParent = R"({
        "asOf": "2026-09-29",
        "curves": [{
            "key": {"currency": "EUR", "role": "XccyBasis", "collateral": "USD"},
            "xccy": {"pair": "EURUSD"},
            "pillars": [{"maturity": "2027-09-29", "kind": "FxSwap", "points": 0.01}]}]})";
    const markets::CurveStackSpec parentless = markets::parseCurveStackSpec(noParent);
    threw = false;
    try {
        (void)markets::buildStack(parentless);
    } catch (const std::invalid_argument& error) {
        threw = true;
        message = error.what();
    }
    CHECK(threw);
    CHECK(message.find("missing parent") != std::string::npos);
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
    CHECK(threw);
    if (threw) {
        CHECK(message.find(needle) != std::string::npos);
    }
}

void testStackBuilderDepthTwo() {
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
    CHECK(stack.curves.size() == 3);
    CHECK(!stack.curves[0].hasParent);
    CHECK(!stack.curves[0].hasDiscount);
    CHECK(stack.curves[1].hasParent);
    CHECK(stack.curves[1].parent == stack.curves[0].key);
    CHECK(!stack.curves[1].hasDiscount);
    CHECK(stack.curves[2].hasParent);
    CHECK(stack.curves[2].parent == stack.curves[1].key);
    CHECK(stack.curves[2].hasDiscount);
    CHECK(stack.curves[2].discount == stack.curves[0].key);

    const std::vector<markets::BuiltCurve> built = markets::buildStack(stack);
    CHECK(built.size() == 3);
    CHECK(built[0].key == stack.curves[0].key);
    CHECK(built[0].role == markets::CurveRole::Discount);
    CHECK(built[1].key == stack.curves[1].key);
    CHECK(built[1].role == markets::CurveRole::Forecast);
    CHECK(built[2].key == stack.curves[2].key);
    CHECK(built[2].role == markets::CurveRole::Forecast);
    const markets::BuiltCurve* rootBuilt = findBuilt(built, stack.curves[0].key);
    const markets::BuiltCurve* curve3mBuilt = findBuilt(built, stack.curves[1].key);
    const markets::BuiltCurve* curve6mBuilt = findBuilt(built, stack.curves[2].key);
    CHECK(rootBuilt != nullptr);
    CHECK(curve3mBuilt != nullptr);
    CHECK(curve6mBuilt != nullptr);

    std::vector<markets::CurvePillar> rootFilled;
    const markets::DiscountCurve<double> rootCurve =
        markets::buildCurve(stack, stack.curves[0], {}, &rootFilled);
    for (const markets::CurvePillar& pillar : rootFilled) {
        util::checkClose("stack root reprice", markets::impliedQuote(pillar, stack.asOf, rootCurve),
                         pillar.quote, 1e-10);
    }
    CHECK(rootBuilt->curve->size() == rootCurve.size());
    CHECK(rootBuilt->curve->times() == rootCurve.times());
    CHECK(rootBuilt->curve->zeroDayCounter().name() == "ACT/365F");
    for (double t = 0.25; t <= 3.0; t += 0.25) {
        util::checkClose("stack root handle discount", rootBuilt->curve->discount(t),
                         rootCurve.discount(t), 1e-15);
        util::checkClose("stack root handle zero", rootBuilt->curve->zero(t), rootCurve.zero(t),
                         1e-15);
    }
    util::checkClose("stack root handle forward", rootBuilt->curve->forward(1.0, 2.0),
                     rootCurve.forward(1.0, 2.0), 1e-15);
    std::vector<double> rootWeights;
    rootBuilt->curve->zeroNodeWeights(1.5, rootWeights);
    CHECK(rootWeights.size() == rootCurve.size());
    const markets::CurveHandle::Ptr rootCopy = markets::CurveHandle::make(rootCurve);
    util::checkClose("handle copy discount", rootCopy->discount(2.0), rootCurve.discount(2.0),
                     1e-15);

    // 3M: direct build over the root handle matches the stack builder output.
    std::vector<markets::ForecastPillar> filled3m;
    const markets::SpreadCurve<double, markets::CurveHandle> curve3m = markets::buildForecastCurve(
        stack, stack.curves[1], rootBuilt->curve, nullptr, {}, &filled3m);
    CHECK(filled3m.size() == stack.curves[1].forecastPillars.size());
    for (const markets::ForecastPillar& pillar : filled3m) {
        util::checkClose("stack 3M reprice",
                         markets::impliedForecastQuote(curve3m, curve3m.parent(), pillar,
                                                       stack.asOf, stack.curves[1].zeroDayCounter),
                         markets::forecastPillarTarget(pillar), 1e-10);
    }
    for (double t = 0.25; t <= 3.0; t += 0.25) {
        util::checkClose("stack 3M handle", curve3mBuilt->curve->discount(t), curve3m.discount(t),
                         1e-15);
    }

    // 6M: parent is the 3M handle, exogenous discount is the OIS root.
    const auto rootShared = std::make_shared<const markets::DiscountCurve<double>>(rootCurve);
    std::vector<markets::ForecastPillar> filled6m;
    const markets::SpreadCurve<double, markets::CurveHandle> curve6m = markets::buildForecastCurve(
        stack, stack.curves[2], curve3mBuilt->curve, rootShared.get(), {}, &filled6m);
    CHECK(curve6m.parentPointer() == curve3mBuilt->curve);
    CHECK(filled6m.size() == stack.curves[2].forecastPillars.size());
    for (const markets::ForecastPillar& pillar : filled6m) {
        util::checkClose("stack 6M reprice",
                         markets::impliedForecastQuote(curve6m, *rootBuilt->curve, pillar,
                                                       stack.asOf, stack.curves[2].zeroDayCounter),
                         markets::forecastPillarTarget(pillar), 1e-10);
    }
    for (double t = 0.25; t <= 3.0; t += 0.25) {
        util::checkClose("stack 6M handle", curve6mBuilt->curve->discount(t), curve6m.discount(t),
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
    CHECK(maxParentDifference > 1e-6);
    QTA_LOG_INFO("test", "stack depth-2 parent resolution difference {:.3e}", maxParentDifference);

    // The topological build order is input-order independent: reversing the
    // document still yields root, 3M, 6M.
    markets::CurveStackSpec reordered = stack;
    std::reverse(reordered.curves.begin(), reordered.curves.end());
    const std::vector<markets::BuiltCurve> reorderedBuilt = markets::buildStack(reordered);
    CHECK(reorderedBuilt.size() == 3);
    CHECK(reorderedBuilt[0].key == stack.curves[0].key);
    CHECK(reorderedBuilt[1].key == stack.curves[1].key);
    CHECK(reorderedBuilt[2].key == stack.curves[2].key);
    for (double t = 0.25; t <= 3.0; t += 0.25) {
        util::checkClose("reordered stack 6M handle", reorderedBuilt[2].curve->discount(t),
                         curve6m.discount(t), 1e-15);
    }

    const markets::CurveHandle::Ptr nullHandle;
    bool threw = false;
    try {
        (void)markets::CurveHandle::make(nullHandle);
    } catch (const std::invalid_argument&) {
        threw = true;
    }
    CHECK(threw);
}

void testStackBuilderErrors() {
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
}

void testStackBuilderConvexityReference() {
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
    CHECK(built.size() == 2);
    // The convexity reference forces the USD curve to build first even though
    // the EUR spec leads the document.
    CHECK(built[0].key == stack.curves[1].key);
    CHECK(built[1].key == stack.curves[0].key);

    const markets::DiscountCurve<double> usd = markets::buildCurve(stack, stack.curves[1]);
    const std::vector<markets::CurveReference> references{{stack.curves[1].key, &usd}};
    std::vector<markets::CurvePillar> eurFilled;
    const markets::DiscountCurve<double> eur =
        markets::buildCurve(stack, stack.curves[0], references, &eurFilled);
    CHECK(eurFilled.size() == 2);
    CHECK(eurFilled[1].convexityAdjustment > 0.0);
    for (const markets::CurvePillar& pillar : eurFilled) {
        util::checkClose("stack reference reprice", markets::impliedQuote(pillar, stack.asOf, eur),
                         pillar.quote, 1e-10);
    }
    for (double t = 0.25; t <= 2.0; t += 0.25) {
        util::checkClose("stack reference curve", built[1].curve->discount(t), eur.discount(t),
                         1e-14);
    }
}

void testStackBuilderBackwardCompatible() {
    const markets::CurveStackSpec stack = markets::loadCurveStackSpec(CURVE_CONFIG_FIXTURE);
    for (const markets::CurveSpec& spec : stack.curves) {
        CHECK(!spec.hasParent);
        CHECK(!spec.hasDiscount);
    }
    const std::vector<markets::BuiltCurve> built = markets::buildStack(stack);
    CHECK(built.size() == stack.curves.size());
    CHECK(built[0].role == markets::CurveRole::Discount);
    const markets::DiscountCurve<double> directly = markets::buildCurve(stack, stack.curves[0]);
    for (double t = 0.25; t <= 10.0; t += 0.25) {
        util::checkClose("fixture stack builder discount", built[0].curve->discount(t),
                         directly.discount(t), 1e-15);
    }
}

} // namespace

int main() {
    testLoadAndBootstrap();
    testValidation();
    testDefaultFixedDayCounter();
    testHymanSplineScheme();
    testAveragedCompoundedConfig();
    testForecastCurveBootstrap();
    testForecastCurveValidation();
    testRoleListInvariant();
    testDuplicateCurveKeys();
    testTypedFieldErrors();
    testUnknownEnumMessages();
    testForecastInterpolationRejects();
    testReferenceCurveIdentity();
    testForecastFutureConfig();
    testUnifiedPillarSpec();
    testFxCurveConfig();
    testFxCurveConfigErrors();
    testStackBuilderDepthTwo();
    testStackBuilderErrors();
    testStackBuilderConvexityReference();
    testStackBuilderBackwardCompatible();
    QTA_LOG_INFO("test", "test_curve_config: ok");
    return 0;
}
