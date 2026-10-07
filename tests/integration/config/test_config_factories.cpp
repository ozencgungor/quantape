// test_config_factories.cpp — config PillarSpec -> family instrument factories.
#include "quantape/datetime/Calendar.h"
#include "quantape/datetime/Date.h"
#include "quantape/datetime/Period.h"
#include "quantape/instruments/PillarVariants.h"
#include "quantape/markets/Curves/CurveConfig.h"
#include "quantape/markets/Descriptors/FXDescriptor.h"

#include <stdexcept>
#include <string>
#include <variant>

#include "support/GtestSupport.h"

using namespace quantape;

namespace {

std::string errorOf(auto&& callable) {
    try {
        callable();
    } catch (const std::invalid_argument& error) {
        return error.what();
    }
    return {};
}

} // namespace

TEST(ConfigFactories, discountKindsMapToFamilyInstruments) {
    const std::string json = R"({
        "asOf": "2026-09-29",
        "curves": [{
            "key": {"currency": "USD", "role": "Discount"},
            "zeroDayCounter": "ACT/365F",
            "pillars": [
                {"maturity": "2027-03-29", "kind": "Deposit", "quote": 0.0432,
                 "quoteDayCounter": "ACT/360", "calendar": "NoHolidays"},
                {"maturity": "2027-06-29", "kind": "Repo", "quote": 0.0435,
                 "quoteDayCounter": "ACT/360", "calendar": "NoHolidays"},
                {"start": "2027-06-29", "maturity": "2027-09-29", "kind": "Fra",
                 "quote": 0.0436, "fraConvexityExponent": 0.001,
                 "quoteDayCounter": "ACT/360", "calendar": "NoHolidays"},
                {"start": "2027-12-16", "maturity": "2028-03-15", "kind": "Future",
                 "style": "Simple", "quote": 0.0438, "convexityAdjustment": 0.0002,
                 "quoteDayCounter": "ACT/360", "calendar": "NoHolidays"},
                {"start": "2027-09-29", "maturity": "2028-09-29", "kind": "OisSwap",
                 "quote": 0.0405, "quoteDayCounter": "ACT/365F", "calendar": "NoHolidays",
                 "fixedTenor": {"length": 1, "unit": "Years"}, "paymentLag": 2,
                 "businessDayConvention": "Following"}
            ]
        }]
    })";
    const markets::CurveStackSpec stack = markets::parseCurveStackSpec(json);
    const markets::CurveSpec& spec = stack.curves.front();
    ASSERT_EQ(spec.pillars.size(), 5u);

    const markets::DiscountInstrument deposit =
        markets::toDiscountInstrument(spec.pillars[0], stack.asOf, spec.zeroDayCounter);
    EXPECT_EQ(markets::instrumentKindName(deposit), "Deposit");
    EXPECT_TRUE(deposit.date() == datetime::Date::parse("2027-03-29"));
    CHECK_CLOSE("deposit target", deposit.target(), 0.0432, 1e-15);
    EXPECT_TRUE(std::get<instruments::Deposit>(deposit.instrument()).calendar.name() ==
                "NoHolidays");

    const markets::DiscountInstrument repo =
        markets::toDiscountInstrument(spec.pillars[1], stack.asOf, spec.zeroDayCounter);
    EXPECT_EQ(markets::instrumentKindName(repo), "Repo");
    EXPECT_TRUE(repo.date() == datetime::Date::parse("2027-06-29"));
    CHECK_CLOSE("repo target", repo.target(), 0.0435, 1e-15);

    const markets::DiscountInstrument fra =
        markets::toDiscountInstrument(spec.pillars[2], stack.asOf, spec.zeroDayCounter);
    EXPECT_EQ(markets::instrumentKindName(fra), "Fra");
    EXPECT_TRUE(fra.date() == datetime::Date::parse("2027-09-29"));
    CHECK_CLOSE("fra target", fra.target(), 0.0436, 1e-15);
    CHECK_CLOSE("fra exponent", std::get<instruments::Fra>(fra.instrument()).fraConvexityExponent,
                0.001, 1e-15);

    const markets::DiscountInstrument future =
        markets::toDiscountInstrument(spec.pillars[3], stack.asOf, spec.zeroDayCounter);
    EXPECT_EQ(markets::instrumentKindName(future), "Future");
    EXPECT_TRUE(future.date() == datetime::Date::parse("2028-03-15"));
    const instruments::Future& futureInstrument =
        std::get<instruments::Future>(future.instrument());
    CHECK_CLOSE("future adjustment", futureInstrument.convexityAdjustment, 0.0002, 1e-15);
    EXPECT_TRUE(futureInstrument.futureStyle == markets::FutureStyle::Simple);

    const markets::DiscountInstrument ois =
        markets::toDiscountInstrument(spec.pillars[4], stack.asOf, spec.zeroDayCounter);
    EXPECT_EQ(markets::instrumentKindName(ois), "OisSwap");
    EXPECT_TRUE(ois.date() == datetime::Date::parse("2028-09-29"));
    CHECK_CLOSE("ois target", ois.target(), 0.0405, 1e-15);
    const auto& oisInstrument = std::get<markets::detail::OisPillarInstrument>(ois.instrument());
    EXPECT_TRUE(oisInstrument.fixedTenor == datetime::Period(1, datetime::TimeUnit::Years));
    EXPECT_EQ(oisInstrument.paymentLag, 2);
    EXPECT_TRUE(oisInstrument.businessDayConvention == datetime::BusinessDayConvention::Following);

    // Role mismatches keep the build-curve error texts.
    markets::PillarSpec forecastKind;
    forecastKind.kind = markets::PillarSpec::Kind::Irs;
    forecastKind.maturity = datetime::Date::parse("2028-03-29");
    forecastKind.quote = 0.043;
    const std::string wrongSide = errorOf([&] {
        (void)markets::toDiscountInstrument(forecastKind, stack.asOf, spec.zeroDayCounter);
    });
    EXPECT_EQ(wrongSide, "CurveConfig: buildCurve: forecast pillar kind on a discount curve");
}

TEST(ConfigFactories, forecastKindsMapToFamilyInstruments) {
    const std::string json = R"({
        "asOf": "2026-09-29",
        "curves": [{
            "key": {"currency": "USD", "role": "Forecast",
                    "indexTenor": {"length": 3, "unit": "Months"}},
            "zeroDayCounter": "ACT/365F",
            "forecastPillars": [
                {"maturity": "2027-03-29", "kind": "Deposit", "quote": 0.044,
                 "quoteDayCounter": "ACT/360", "calendar": "NoHolidays"},
                {"start": "2027-03-29", "maturity": "2027-06-29", "kind": "Fra", "quote": 0.0435,
                 "quoteDayCounter": "ACT/360", "calendar": "NoHolidays"},
                {"maturity": "2028-03-29", "kind": "Irs", "quote": 0.043,
                 "floatTenor": {"length": 3, "unit": "Months"},
                 "fixedTenor": {"length": 1, "unit": "Years"},
                 "floatCalendar": "NoHolidays", "fixedCalendar": "NoHolidays"},
                {"maturity": "2029-03-29", "kind": "BasisSwap", "quote": 0.0012,
                 "floatTenor": {"length": 3, "unit": "Months"},
                 "quoteDayCounter": "ACT/360", "calendar": "NoHolidays",
                 "spreadOnParentLeg": false}
            ]
        }]
    })";
    const markets::CurveStackSpec stack = markets::parseCurveStackSpec(json);
    const markets::CurveSpec& spec = stack.curves.front();
    ASSERT_EQ(spec.forecastPillars.size(), 4u);

    const markets::ForecastInstrument deposit =
        markets::toForecastInstrument(spec.forecastPillars[0], stack.asOf, spec.zeroDayCounter);
    EXPECT_EQ(markets::instrumentKindName(deposit), "Deposit");
    CHECK_CLOSE("forecast deposit target", deposit.target(), 0.044, 1e-15);

    const markets::ForecastInstrument fra =
        markets::toForecastInstrument(spec.forecastPillars[1], stack.asOf, spec.zeroDayCounter);
    EXPECT_EQ(markets::instrumentKindName(fra), "Fra");
    EXPECT_TRUE(std::get<instruments::Fra>(fra.instrument()).start ==
                datetime::Date::parse("2027-03-29"));

    const markets::ForecastInstrument irs =
        markets::toForecastInstrument(spec.forecastPillars[2], stack.asOf, spec.zeroDayCounter);
    EXPECT_EQ(markets::instrumentKindName(irs), "Irs");
    CHECK_CLOSE("irs target", irs.target(), 0.043, 1e-15);
    EXPECT_TRUE(irs.date() == datetime::Date::parse("2028-03-29"));

    const markets::ForecastInstrument basis =
        markets::toForecastInstrument(spec.forecastPillars[3], stack.asOf, spec.zeroDayCounter);
    EXPECT_EQ(markets::instrumentKindName(basis), "Basis");
    CHECK_CLOSE("basis target", basis.target(), 0.0012, 1e-15);
    EXPECT_FALSE(std::get<instruments::BasisSwap>(basis.instrument()).spreadOnParentLeg);

    // A discount-only kind on the forecast factory keeps its literal error.
    const std::string repoJson = R"({
        "asOf": "2026-09-29",
        "curves": [{
            "key": {"currency": "USD", "role": "Discount"},
            "pillars": [{"maturity": "2027-03-29", "kind": "Repo", "quote": 0.043,
                         "calendar": "NoHolidays"}]
        }]
    })";
    const markets::CurveStackSpec repoStack = markets::parseCurveStackSpec(repoJson);
    const markets::PillarSpec& repoPillar = repoStack.curves.front().pillars.front();
    const std::string message = errorOf([&] {
        (void)markets::toDiscountInstrument(repoPillar, repoStack.asOf,
                                            repoStack.curves.front().zeroDayCounter);
    });
    EXPECT_TRUE(message.empty());

    const std::string forecastMessage = errorOf([&] {
        (void)markets::toForecastInstrument(repoPillar, repoStack.asOf,
                                            repoStack.curves.front().zeroDayCounter);
    });
    EXPECT_EQ(forecastMessage,
              "CurveConfig: buildForecastCurve: discount pillar kind on a forecast curve");
}

TEST(ConfigFactories, fxValidationRunsBeforeMakeFxSwap) {
    markets::CurveStackSpec stack;
    stack.asOf = datetime::Date::parse("2026-09-29");
    markets::XccyCurveSpec xccy;
    markets::PillarSpec pillar;
    pillar.kind = markets::PillarSpec::Kind::FxSwap;
    pillar.start = datetime::Date::parse("2026-10-01");
    pillar.maturity = datetime::Date::parse("2027-09-29");
    pillar.fxPoints = 0.01;
    const std::string where = "buildStack: curve 'EUR/XccyBasis'";

    xccy.pair.clear();
    const std::string missingPair =
        errorOf([&] { (void)markets::toFxInstrument(stack, pillar, xccy, where); });
    EXPECT_EQ(missingPair,
              "CurveConfig: buildStack: curve 'EUR/XccyBasis': FxSwap pillars require a 'pair' "
              "or a curve-level 'xccy.pair'");

    xccy.pair = "EURUSD";
    const std::string missingSpot =
        errorOf([&] { (void)markets::toFxInstrument(stack, pillar, xccy, where); });
    EXPECT_EQ(missingSpot,
              "CurveConfig: buildStack: curve 'EUR/XccyBasis': missing FX spot for pair 'EURUSD'");

    stack.fxSpots.push_back(markets::FxSpotSpec{markets::FXDescriptor("EUR", "USD"), 1.10});
    pillar.start = pillar.maturity;
    const std::string badOrder =
        errorOf([&] { (void)markets::toFxInstrument(stack, pillar, xccy, where); });
    EXPECT_EQ(badOrder,
              "CurveConfig: buildStack: curve 'EUR/XccyBasis': FxSwap pillar maturity must be "
              "after the near date");

    // The factory's own makeFxSwap texts never leak on a valid pillar.
    pillar.start = datetime::Date::parse("2026-10-01");
    const markets::FxInstrument instrument = markets::toFxInstrument(stack, pillar, xccy, where);
    CHECK_CLOSE("fx instrument target", instrument.target(), 0.01, 1e-15);
    EXPECT_TRUE(instrument.maturity == pillar.maturity);
    EXPECT_TRUE(instrument.start == pillar.start);
    EXPECT_FALSE(instrument.isFxBaseCollateral);
}

TEST(ConfigFactories, xccyKindsMapCurveLevelLegConventions) {
    markets::XccyCurveSpec xccy;
    xccy.pair = "EURUSD";
    xccy.notional = markets::XccyNotionalMode::MtM;
    xccy.basisLeg = "Quote";
    xccy.resetLeg = "Base";
    xccy.isFxBaseCollateral = false;
    markets::PillarSpec pillar;
    pillar.kind = markets::PillarSpec::Kind::XccySwap;
    pillar.maturity = datetime::Date::parse("2031-09-29");
    pillar.quote = 0.0011;
    pillar.foreignCalendar = datetime::Calendar::noHolidays();
    pillar.domesticCalendar = datetime::Calendar::noHolidays();
    pillar.foreignTenor = datetime::Period(3, datetime::TimeUnit::Months);
    pillar.domesticTenor = datetime::Period(6, datetime::TimeUnit::Months);
    pillar.businessDayConvention = datetime::BusinessDayConvention::Following;

    const markets::XccyInstrument instrument = markets::toXccyInstrument(pillar, xccy);
    const auto& xccyInstrument = std::get<instruments::XccyBasisSwap>(instrument.instrument());
    CHECK_CLOSE("xccy target", instrument.target(), 0.0011, 1e-15);
    EXPECT_TRUE(instrument.date() == datetime::Date::parse("2031-09-29"));
    EXPECT_TRUE(xccyInstrument.notional == markets::XccyNotionalMode::MtM);
    EXPECT_TRUE(xccyInstrument.spreadOnForeignLeg == false);
    EXPECT_TRUE(xccyInstrument.resetForeignLeg == true);
    EXPECT_TRUE(xccyInstrument.foreignTenor == datetime::Period(3, datetime::TimeUnit::Months));
    EXPECT_TRUE(xccyInstrument.domesticTenor == datetime::Period(6, datetime::TimeUnit::Months));

    // The per-pillar collateral override flips the basis to the foreign leg.
    markets::PillarSpec overrideSpec = pillar;
    overrideSpec.isFxBaseCollateral = true;
    overrideSpec.isFxBaseCollateralSet = true;
    const markets::XccyInstrument overridden = markets::toXccyInstrument(overrideSpec, xccy);
    const auto& overriddenInstrument =
        std::get<instruments::XccyBasisSwap>(overridden.instrument());
    EXPECT_TRUE(overriddenInstrument.spreadOnForeignLeg == true);
    EXPECT_TRUE(overriddenInstrument.resetForeignLeg == false);
}
