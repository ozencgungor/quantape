#include "quantape/markets/Curves/CurveConfig.h"

#include "quantape/log/Log.h"
#include "quantape/util/Check.h"

#include <cmath>
#include <memory>
#include <stdexcept>
#include <string>
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
    CHECK(spec.pillars.front().kind == markets::PillarKind::Deposit);
    CHECK(spec.pillars.back().kind == markets::PillarKind::OisSwap);

    const markets::DiscountCurve<double> curve = markets::buildCurve(stack, spec);
    CHECK(curve.size() == spec.pillars.size() + 1);
    CHECK(curve.discount(0.0) == 1.0);

    for (const markets::PillarSpec& pillar : spec.pillars) {
        markets::CurvePillar out;
        out.maturity = pillar.maturity;
        out.kind = pillar.kind;
        out.quote = pillar.quote;
        out.quoteDayCounter = pillar.quoteDayCounter;
        out.calendar = pillar.calendar;
        out.fixedTenor = pillar.fixedTenor;
        out.paymentLag = pillar.paymentLag;
        out.businessDayConvention = pillar.businessDayConvention;
        util::checkClose("config reprice", markets::impliedQuote(out, stack.asOf, curve),
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

    util::checkClose("day counter name", markets::dayCounterFromName("ACT/365F").yearFraction(
                                             datetime::Date::parse("2026-01-01"),
                                             datetime::Date::parse("2027-01-01")),
                     1.0, 1e-12);
    CHECK(markets::curveRoleToName(markets::CurveRole::XccyBasis) == "XccyBasis");
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

    const markets::DiscountCurve<double> curve = markets::buildCurve(stack, spec);
    CHECK(curve.scheme() == markets::InterpolationScheme::HymanSpline);
    for (const markets::PillarSpec& pillar : spec.pillars) {
        markets::CurvePillar out;
        out.maturity = pillar.maturity;
        out.kind = pillar.kind;
        out.quote = pillar.quote;
        out.quoteDayCounter = pillar.quoteDayCounter;
        out.calendar = pillar.calendar;
        out.fixedTenor = pillar.fixedTenor;
        out.businessDayConvention = pillar.businessDayConvention;
        out.paymentLag = pillar.paymentLag;
        util::checkClose("hyman config reprice", markets::impliedQuote(out, stack.asOf, curve),
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
    const markets::DiscountCurve<double> curve = markets::buildCurve(stack, spec);
    const markets::PillarSpec& pillar = spec.pillars.front();
    markets::CurvePillar out;
    out.maturity = pillar.maturity;
    out.start = pillar.start;
    out.kind = pillar.kind;
    out.quote = pillar.quote;
    out.convexityAdjustment = pillar.convexityAdjustment;
    out.futureStyle = pillar.futureStyle;
    out.averagingStyle = pillar.averagingStyle;
    out.quoteDayCounter = pillar.quoteDayCounter;
    out.calendar = pillar.calendar;
    util::checkClose("averaged compounded config reprice",
                     markets::impliedQuote(out, stack.asOf, curve), pillar.quote, 1e-9);

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
    CHECK(forecastSpec.forecastPillars.size() == 4);
    CHECK(forecastSpec.forecastPillars[0].kind == markets::ForecastPillar::Kind::Deposit);
    CHECK(forecastSpec.forecastPillars[1].kind == markets::ForecastPillar::Kind::Fra);
    CHECK(forecastSpec.forecastPillars[2].kind == markets::ForecastPillar::Kind::Irs);
    CHECK(forecastSpec.forecastPillars[3].kind == markets::ForecastPillar::Kind::BasisSwap);
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
    CHECK(filled[2].irs.businessDayConvention ==
          datetime::BusinessDayConvention::Following);
    CHECK(filled[3].basis.businessDayConvention ==
          datetime::BusinessDayConvention::Following);
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
        const double implied =
            markets::impliedForecastQuote(forecast, *parent, pillar, stack.asOf,
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

    const std::string xccyRole = R"({
        "asOf": "2026-09-29",
        "curves": [{
            "key": {"currency": "USD", "role": "XccyBasis"},
            "pillars": [{"maturity": "2027-09-29", "kind": "Deposit", "quote": 0.04,
                         "calendar": "NoHolidays"}]
        }]
    })";
    expectParseReject(xccyRole, "not configurable");

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
    const markets::ForecastPillarSpec& futureSpec = forecastSpec.forecastPillars[0];
    CHECK(futureSpec.kind == markets::ForecastPillar::Kind::Future);
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
    const markets::CurveStackSpec missingStack =
        markets::parseCurveStackSpec(missingAdjustment);
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

} // namespace

int main() {
    testLoadAndBootstrap();
    testValidation();
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
    QTA_LOG_INFO("test", "test_curve_config: ok");
    return 0;
}
