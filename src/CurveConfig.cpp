#include "quantape/markets/Curves/CurveConfig.h"

#include "quantape/markets/Curves/FraConvexity.h"
#include "quantape/markets/Curves/FxSwapBuilder.h"
#include "quantape/markets/Curves/HullWhiteConvexity.h"
#include "quantape/markets/Curves/XccyBasisBuilder.h"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>

#include <nlohmann/json.hpp>

namespace quantape::markets {

namespace {

using Json = nlohmann::json;

[[noreturn]] void fail(const std::string& where, const std::string& message) {
    throw std::invalid_argument("CurveConfig: " + where + ": " + message);
}

std::string keyLabel(const CurveKey& key) {
    return key.currency + "/" + std::string(curveRoleToName(key.role));
}

const Json& require(const Json& node, const char* key, const std::string& where) {
    const auto it = node.find(key);
    if (it == node.end()) {
        fail(where, std::string("missing field '") + key + "'");
    }
    return *it;
}

template <typename T>
T asType(const Json& node, const char* key, const std::string& where, std::string_view expected) {
    const Json& value = require(node, key, where);
    try {
        return value.get<T>();
    } catch (const Json::exception&) {
        fail(where, std::string("field '") + key + "' must be " + std::string(expected) +
                        " (found " + value.type_name() + ")");
    }
}

std::string asString(const Json& node, const char* key, const std::string& where) {
    return asType<std::string>(node, key, where, "a string");
}

double asDouble(const Json& node, const char* key, const std::string& where) {
    return asType<double>(node, key, where, "a number");
}

int asInt(const Json& node, const char* key, const std::string& where) {
    const Json& value = require(node, key, where);
    if (!value.is_number_integer()) {
        fail(where, std::string("field '") + key + "' must be an integer (found " +
                        value.type_name() + ")");
    }
    try {
        return value.get<int>();
    } catch (const Json::exception&) {
        fail(where, std::string("field '") + key + "' must fit in a 32-bit integer");
    }
}

bool asBool(const Json& node, const char* key, const std::string& where) {
    return asType<bool>(node, key, where, "a boolean");
}

datetime::Date asDate(const Json& node, const char* key, const std::string& where) {
    const std::string text = asString(node, key, where);
    try {
        return datetime::Date::parse(text);
    } catch (const std::exception&) {
        fail(where, std::string("field '") + key + "' is not a YYYY-MM-DD date: " + text);
    }
}

datetime::Period asPeriod(const Json& node, const std::string& where) {
    if (!node.is_object()) {
        fail(where, "period must be an object {length, unit}");
    }
    const double rawLength = asDouble(node, "length", where);
    if (static_cast<double>(static_cast<int>(rawLength)) != rawLength) {
        fail(where, "'length' must be an integer");
    }
    const int length = static_cast<int>(rawLength);
    if (length <= 0) {
        fail(where, "'length' must be positive");
    }
    const std::string unit = asString(node, "unit", where);
    if (unit == "Days") {
        return datetime::Period(length, datetime::TimeUnit::Days);
    }
    if (unit == "Weeks") {
        return datetime::Period(length, datetime::TimeUnit::Weeks);
    }
    if (unit == "Months") {
        return datetime::Period(length, datetime::TimeUnit::Months);
    }
    if (unit == "Years") {
        return datetime::Period(length, datetime::TimeUnit::Years);
    }
    fail(where, "unknown period unit '" + unit + "' (expected Days, Weeks, Months or Years)");
}

CurveKey asKey(const Json& node, const std::string& where) {
    CurveKey key;
    key.currency = asString(node, "currency", where);
    key.role = curveRoleFromName(asString(node, "role", where));
    if (node.contains("collateral")) {
        key.collateral = asString(node, "collateral", where);
    } else {
        key.collateral = key.currency;
    }
    const auto tenor = node.find("indexTenor");
    if (tenor != node.end()) {
        key.indexTenor = asPeriod(*tenor, where + ".indexTenor");
    }
    return key;
}

/// Shared kind map for both curve sides; `forecast` selects the forecast-only
/// kinds and the error text, while the shared names always map the same way.
/// `xccy` selects the cross-currency kinds for `XccyBasis`-role curves.
PillarSpec::Kind pillarKindFromName(std::string_view name, bool forecast, bool xccy,
                                    const std::string& where) {
    if (xccy) {
        if (name == "FxSwap") {
            return PillarSpec::Kind::FxSwap;
        }
        if (name == "XccySwap") {
            return PillarSpec::Kind::XccySwap;
        }
        fail(where, "unknown cross-currency pillar kind '" + std::string(name) +
                        "' (expected FxSwap or XccySwap)");
    }
    if (name == "Deposit") {
        return PillarSpec::Kind::Deposit;
    }
    if (name == "Fra") {
        return PillarSpec::Kind::Fra;
    }
    if (name == "Future") {
        return PillarSpec::Kind::Future;
    }
    if (forecast) {
        if (name == "Irs") {
            return PillarSpec::Kind::Irs;
        }
        if (name == "BasisSwap") {
            return PillarSpec::Kind::BasisSwap;
        }
        fail(where, "unknown forecast pillar kind '" + std::string(name) +
                        "' (expected Deposit, Fra, Future, Irs or BasisSwap)");
    }
    if (name == "Repo") {
        return PillarSpec::Kind::Repo;
    }
    if (name == "OisSwap") {
        return PillarSpec::Kind::OisSwap;
    }
    fail(where, "unknown pillar kind '" + std::string(name) +
                    "' (expected Deposit, Repo, Fra, Future or OisSwap)");
}

QuoteConvention quoteConventionFromName(const std::string& name, const std::string& where) {
    if (name == "Points") {
        return QuoteConvention::Points;
    }
    if (name == "Outright") {
        return QuoteConvention::Outright;
    }
    fail(where, "unknown quote convention '" + name + "' (expected Points or Outright)");
}

XccyNotionalMode notionalModeFromName(const std::string& name, const std::string& where) {
    if (name == "Const") {
        return XccyNotionalMode::Const;
    }
    if (name == "MtM") {
        return XccyNotionalMode::MtM;
    }
    fail(where, "unknown notional mode '" + name + "' (expected Const or MtM)");
}

std::string legFromName(const std::string& name, const std::string& where) {
    if (name == "Base" || name == "Quote") {
        return name;
    }
    fail(where, "unknown leg '" + name + "' (expected Base or Quote)");
}

FutureStyle futureStyleFromName(const std::string& style, const std::string& where) {
    if (style == "Simple") {
        return FutureStyle::Simple;
    }
    if (style == "Compounded") {
        return FutureStyle::Compounded;
    }
    if (style == "Averaged") {
        return FutureStyle::Averaged;
    }
    fail(where, "unknown future style '" + style + "' (expected Simple, Compounded or Averaged)");
}

AveragingStyle averagingStyleFromName(const std::string& name, const std::string& where) {
    if (name == "Arithmetic") {
        return AveragingStyle::Arithmetic;
    }
    if (name == "Compounded") {
        return AveragingStyle::Compounded;
    }
    fail(where, "unknown averaging style '" + name + "' (expected Arithmetic or Compounded)");
}

/// Parse one pillar object shared by the `pillars` and `forecastPillars`
/// arrays; `forecast` selects the accepted kinds, field aliases and
/// kind-specific checks of the forecast side; `xccy` selects the
/// cross-currency kinds of an `XccyBasis`-role curve.
PillarSpec parsePillarSpec(const Json& node, const std::string& where, bool forecast, bool xccy) {
    PillarSpec pillar;
    pillar.maturity = asDate(node, "maturity", where);
    if (node.contains("start")) {
        pillar.start = asDate(node, "start", where);
    }
    pillar.kind = pillarKindFromName(asString(node, "kind", where), forecast, xccy, where);
    const PillarSpec::Kind kind = pillar.kind;
    if (kind == PillarSpec::Kind::FxSwap) {
        if (node.contains("convention")) {
            pillar.fxConvention =
                quoteConventionFromName(asString(node, "convention", where), where);
        } else if (node.contains("outright") && !node.contains("points")) {
            pillar.fxConvention = QuoteConvention::Outright;
        }
        if (pillar.fxConvention == QuoteConvention::Points) {
            pillar.fxPoints = asDouble(node, "points", where);
        } else {
            pillar.fxOutright = asDouble(node, "outright", where);
        }
        if (node.contains("pair")) {
            pillar.fxPair = asString(node, "pair", where);
            if (pillar.fxPair.size() != 6) {
                fail(where, "'pair' must be a six-letter base+quote string (e.g. \"EURUSD\")");
            }
        }
        if (node.contains("isFxBaseCollateral")) {
            pillar.isFxBaseCollateral = asBool(node, "isFxBaseCollateral", where);
            pillar.isFxBaseCollateralSet = true;
        }
        if (node.contains("quoteDayCounter")) {
            pillar.quoteDayCounter = dayCounterFromName(asString(node, "quoteDayCounter", where));
        } else if (node.contains("dayCounter")) {
            pillar.quoteDayCounter = dayCounterFromName(asString(node, "dayCounter", where));
        }
        return pillar;
    }
    if (kind == PillarSpec::Kind::XccySwap) {
        if (node.contains("quote")) {
            pillar.quote = asDouble(node, "quote", where);
        } else if (node.contains("spread")) {
            pillar.quote = asDouble(node, "spread", where);
        } else {
            fail(where, "missing field 'spread'");
        }
        if (node.contains("notional")) {
            pillar.xccyNotional = notionalModeFromName(asString(node, "notional", where), where);
            pillar.xccyNotionalSet = true;
        }
        if (node.contains("isFxBaseCollateral")) {
            pillar.isFxBaseCollateral = asBool(node, "isFxBaseCollateral", where);
            pillar.isFxBaseCollateralSet = true;
        }
        if (node.contains("foreignTenor")) {
            pillar.foreignTenor = asPeriod(node.at("foreignTenor"), where + ".foreignTenor");
        }
        if (node.contains("domesticTenor")) {
            pillar.domesticTenor = asPeriod(node.at("domesticTenor"), where + ".domesticTenor");
        }
        if (node.contains("foreignCalendar")) {
            pillar.foreignCalendar = calendarFromName(asString(node, "foreignCalendar", where));
        }
        if (node.contains("domesticCalendar")) {
            pillar.domesticCalendar = calendarFromName(asString(node, "domesticCalendar", where));
        }
        if (node.contains("foreignDayCounter")) {
            pillar.foreignDayCounter =
                dayCounterFromName(asString(node, "foreignDayCounter", where));
        }
        if (node.contains("domesticDayCounter")) {
            pillar.domesticDayCounter =
                dayCounterFromName(asString(node, "domesticDayCounter", where));
        }
        if (node.contains("businessDayConvention")) {
            pillar.businessDayConvention =
                conventionFromName(asString(node, "businessDayConvention", where));
        }
        if (node.contains("foreignPaymentLag")) {
            pillar.foreignPaymentLag = asInt(node, "foreignPaymentLag", where);
            if (pillar.foreignPaymentLag < 0) {
                fail(where, "'foreignPaymentLag' must be non-negative");
            }
        }
        if (node.contains("domesticPaymentLag")) {
            pillar.domesticPaymentLag = asInt(node, "domesticPaymentLag", where);
            if (pillar.domesticPaymentLag < 0) {
                fail(where, "'domesticPaymentLag' must be non-negative");
            }
        }
        return pillar;
    }
    if (!forecast && (kind == PillarSpec::Kind::Fra || kind == PillarSpec::Kind::Future) &&
        !node.contains("start")) {
        fail(where, "Fra/Future pillars require a 'start' date");
    }
    if (forecast && kind == PillarSpec::Kind::Future && !node.contains("start")) {
        fail(where, "Future forecast pillars require a 'start' date");
    }
    if (node.contains("quote")) {
        pillar.quote = asDouble(node, "quote", where);
    } else if (forecast && kind == PillarSpec::Kind::BasisSwap && node.contains("spread")) {
        pillar.quote = asDouble(node, "spread", where);
    } else {
        fail(where, "missing field 'quote'");
    }
    if (node.contains("quoteDayCounter")) {
        pillar.quoteDayCounter = dayCounterFromName(asString(node, "quoteDayCounter", where));
    } else if (forecast && node.contains("dayCounter")) {
        pillar.quoteDayCounter = dayCounterFromName(asString(node, "dayCounter", where));
    } else if (forecast && node.contains("floatDayCounter")) {
        pillar.quoteDayCounter = dayCounterFromName(asString(node, "floatDayCounter", where));
    }
    if (node.contains("calendar")) {
        pillar.calendar = calendarFromName(asString(node, "calendar", where));
    } else if (forecast && node.contains("floatCalendar")) {
        pillar.calendar = calendarFromName(asString(node, "floatCalendar", where));
    }
    if (node.contains("businessDayConvention")) {
        pillar.businessDayConvention =
            conventionFromName(asString(node, "businessDayConvention", where));
    }
    if (node.contains("fixedTenor")) {
        pillar.fixedTenor = asPeriod(node.at("fixedTenor"), where + ".fixedTenor");
    }
    if (forecast) {
        if (node.contains("floatTenor")) {
            pillar.floatTenor = asPeriod(node.at("floatTenor"), where + ".floatTenor");
        }
        if (node.contains("floatCalendar")) {
            pillar.floatCalendar = calendarFromName(asString(node, "floatCalendar", where));
        } else {
            pillar.floatCalendar = pillar.calendar;
        }
        if (node.contains("fixedCalendar")) {
            pillar.fixedCalendar = calendarFromName(asString(node, "fixedCalendar", where));
        } else {
            pillar.fixedCalendar = pillar.calendar;
        }
        if (node.contains("floatDayCounter")) {
            pillar.floatDayCounter = dayCounterFromName(asString(node, "floatDayCounter", where));
        } else {
            pillar.floatDayCounter = pillar.quoteDayCounter;
        }
        if (node.contains("fixedDayCounter")) {
            pillar.fixedDayCounter = dayCounterFromName(asString(node, "fixedDayCounter", where));
        }
    }
    if (node.contains("firstFixing")) {
        if (forecast) {
            if (kind != PillarSpec::Kind::Irs) {
                fail(where, "'firstFixing' is only valid for Irs forecast pillars");
            }
        } else if (kind != PillarSpec::Kind::OisSwap) {
            fail(where, "'firstFixing' is only valid for OisSwap pillars");
        }
        pillar.firstCouponRate = asDouble(node, "firstFixing", where);
        pillar.firstCouponFixed = true;
    } else if (forecast && node.contains("firstCouponRate")) {
        if (kind != PillarSpec::Kind::Irs) {
            fail(where, "'firstCouponRate' is only valid for Irs forecast pillars");
        }
        pillar.firstCouponRate = asDouble(node, "firstCouponRate", where);
        pillar.firstCouponFixed = true;
    }
    if (forecast && node.contains("firstCouponFixed")) {
        if (kind != PillarSpec::Kind::Irs) {
            fail(where, "'firstCouponFixed' is only valid for Irs forecast pillars");
        }
        pillar.firstCouponFixed = asBool(node, "firstCouponFixed", where);
    }
    if (!forecast && node.contains("fraConvexityExponent")) {
        if (kind != PillarSpec::Kind::Fra) {
            fail(where, "'fraConvexityExponent' is only valid for Fra pillars");
        }
        pillar.fraConvexityExponent = asDouble(node, "fraConvexityExponent", where);
        pillar.fraConvexityExponentSet = true;
    }
    if (node.contains("convexityAdjustment")) {
        if (kind != PillarSpec::Kind::Future) {
            fail(where, forecast ? "'convexityAdjustment' is only valid for Future forecast pillars"
                                 : "'convexityAdjustment' is only valid for Future pillars");
        }
        pillar.convexityAdjustment = asDouble(node, "convexityAdjustment", where);
        pillar.convexityAdjustmentSet = true;
    }
    if (node.contains("style")) {
        if (kind != PillarSpec::Kind::Future) {
            fail(where, forecast ? "'style' is only valid for Future forecast pillars"
                                 : "'style' is only valid for Future pillars");
        }
        pillar.futureStyle = futureStyleFromName(asString(node, "style", where), where);
    }
    if (node.contains("averaging")) {
        if (kind != PillarSpec::Kind::Future) {
            fail(where, forecast ? "'averaging' is only valid for Future forecast pillars"
                                 : "'averaging' is only valid for Future pillars");
        }
        if (pillar.futureStyle != FutureStyle::Averaged) {
            fail(where, forecast ? "'averaging' is only valid for Future forecast pillars with "
                                   "'style': 'Averaged'"
                                 : "'averaging' is only valid for Future pillars with 'style': "
                                   "'Averaged'");
        }
        pillar.averagingStyle = averagingStyleFromName(asString(node, "averaging", where), where);
    }
    if (node.contains("paymentLag")) {
        if (forecast) {
            if (kind != PillarSpec::Kind::Irs) {
                fail(where, "'paymentLag' is only valid for Irs forecast pillars");
            }
        } else if (kind != PillarSpec::Kind::OisSwap) {
            fail(where, "'paymentLag' is only valid for OisSwap pillars");
        }
        pillar.paymentLag = asInt(node, "paymentLag", where);
        if (pillar.paymentLag < 0) {
            fail(where, "'paymentLag' must be non-negative");
        }
    }
    if (forecast && node.contains("spreadOnParentLeg")) {
        if (kind != PillarSpec::Kind::BasisSwap) {
            fail(where, "'spreadOnParentLeg' is only valid for BasisSwap forecast pillars");
        }
        pillar.spreadOnParentLeg = asBool(node, "spreadOnParentLeg", where);
    }
    if (forecast && kind == PillarSpec::Kind::Irs &&
        (!node.contains("floatTenor") || !node.contains("fixedTenor"))) {
        fail(where, "Irs forecast pillars require 'floatTenor' and 'fixedTenor'");
    }
    return pillar;
}

InterpolationSpace spaceFromName(std::string_view name, const std::string& where) {
    if (name == "Zero") {
        return InterpolationSpace::Zero;
    }
    if (name == "LogDiscount") {
        return InterpolationSpace::LogDiscount;
    }
    fail(where,
         "unknown interpolation space '" + std::string(name) + "' (expected Zero or LogDiscount)");
}

InterpolationScheme schemeFromName(const std::string& name, const std::string& where) {
    if (name == "Linear") {
        return InterpolationScheme::Linear;
    }
    if (name == "Akima") {
        return InterpolationScheme::Akima;
    }
    if (name == "TensionSpline") {
        return InterpolationScheme::TensionSpline;
    }
    if (name == "HymanSpline") {
        return InterpolationScheme::HymanSpline;
    }
    if (name == "MonotoneCubic") {
        return InterpolationScheme::MonotoneCubic;
    }
    if (name == "MixedLinearCubic") {
        return InterpolationScheme::MixedLinearCubic;
    }
    fail(where, "unknown interpolation scheme '" + name +
                    "' (expected Linear, Akima, TensionSpline, HymanSpline, MonotoneCubic or "
                    "MixedLinearCubic)");
}

} // namespace

datetime::DayCounter dayCounterFromName(std::string_view name) {
    using datetime::DayCount;
    if (name == "ACT/360") {
        return datetime::DayCounter(DayCount::Actual360);
    }
    if (name == "ACT/365F") {
        return datetime::DayCounter(DayCount::Actual365Fixed);
    }
    if (name == "ACT/365.25") {
        return datetime::DayCounter(DayCount::Actual365_25);
    }
    if (name == "30/360") {
        return datetime::DayCounter(DayCount::Thirty360US);
    }
    if (name == "30/360 BondBasis" || name == "30/360 Bond Basis") {
        return datetime::DayCounter(DayCount::Thirty360BondBasis);
    }
    if (name == "30E/360") {
        return datetime::DayCounter(DayCount::ThirtyE360);
    }
    if (name == "ACT/ACT") {
        return datetime::DayCounter(DayCount::ActualActualISDA);
    }
    throw std::invalid_argument(
        "CurveConfig: unknown day counter '" + std::string(name) +
        "' (expected ACT/360, ACT/365F, ACT/365.25, 30/360, 30/360 BondBasis, 30E/360 or "
        "ACT/ACT)");
}

std::string conventionName(datetime::BusinessDayConvention convention) {
    switch (convention) {
        case datetime::BusinessDayConvention::Unadjusted:
            return "Unadjusted";
        case datetime::BusinessDayConvention::Following:
            return "Following";
        case datetime::BusinessDayConvention::ModifiedFollowing:
            return "ModifiedFollowing";
        case datetime::BusinessDayConvention::HalfMonthModifiedFollowing:
            return "HalfMonthModifiedFollowing";
        case datetime::BusinessDayConvention::Preceding:
            return "Preceding";
        case datetime::BusinessDayConvention::ModifiedPreceding:
            return "ModifiedPreceding";
        case datetime::BusinessDayConvention::Nearest:
            return "Nearest";
    }
    return "Unknown";
}

datetime::BusinessDayConvention conventionFromName(std::string_view name) {
    if (name == "Unadjusted")
        return datetime::BusinessDayConvention::Unadjusted;
    if (name == "Following")
        return datetime::BusinessDayConvention::Following;
    if (name == "ModifiedFollowing")
        return datetime::BusinessDayConvention::ModifiedFollowing;
    if (name == "HalfMonthModifiedFollowing") {
        return datetime::BusinessDayConvention::HalfMonthModifiedFollowing;
    }
    if (name == "Preceding")
        return datetime::BusinessDayConvention::Preceding;
    if (name == "ModifiedPreceding")
        return datetime::BusinessDayConvention::ModifiedPreceding;
    if (name == "Nearest")
        return datetime::BusinessDayConvention::Nearest;
    throw std::invalid_argument(
        "CurveConfig: unknown convention name '" + std::string(name) +
        "' (expected Unadjusted, Following, ModifiedFollowing, HalfMonthModifiedFollowing, "
        "Preceding, ModifiedPreceding or Nearest)");
}

datetime::Calendar calendarFromName(std::string_view name) {
    if (name == "WeekendsOnly") {
        return datetime::Calendar::weekendsOnly();
    }
    if (name == "NoHolidays") {
        return datetime::Calendar::noHolidays();
    }
    if (name == "SIFMA") {
        return datetime::Calendar::sifma();
    }
    if (name == "FederalReserve") {
        return datetime::Calendar::federalReserve();
    }
    if (name == "TARGET") {
        return datetime::Calendar::target();
    }
    if (name == "UnitedKingdom") {
        return datetime::Calendar::unitedKingdom();
    }
    if (name == "Japan") {
        return datetime::Calendar::japan();
    }
    throw std::invalid_argument(
        "CurveConfig: unknown calendar '" + std::string(name) +
        "' (expected WeekendsOnly, NoHolidays, SIFMA, FederalReserve, TARGET, UnitedKingdom or "
        "Japan)");
}

CurveRole curveRoleFromName(std::string_view name) {
    if (name == "Discount") {
        return CurveRole::Discount;
    }
    if (name == "Forecast") {
        return CurveRole::Forecast;
    }
    if (name == "TenorBasis") {
        return CurveRole::TenorBasis;
    }
    if (name == "IborOisBasis") {
        return CurveRole::IborOisBasis;
    }
    if (name == "XccyBasis") {
        return CurveRole::XccyBasis;
    }
    if (name == "TurnOverlay") {
        return CurveRole::TurnOverlay;
    }
    throw std::invalid_argument(
        "CurveConfig: unknown curve role '" + std::string(name) +
        "' (expected Discount, Forecast, TenorBasis, IborOisBasis, XccyBasis or TurnOverlay)");
}

std::string_view curveRoleToName(CurveRole role) {
    return curveRoleName(role);
}

CurveStackSpec parseCurveStackSpec(const std::string& jsonText) {
    Json document;
    try {
        document = Json::parse(jsonText);
    } catch (const Json::parse_error& error) {
        throw std::invalid_argument(std::string("CurveConfig: JSON parse error: ") + error.what());
    }
    if (!document.is_object()) {
        fail("root", "document must be an object");
    }

    CurveStackSpec stack;
    stack.asOf = asDate(document, "asOf", "root");
    if (document.contains("spotLag")) {
        const Json& lags = document.at("spotLag");
        if (!lags.is_object()) {
            fail("root.spotLag", "must be an object mapping currency to a settlement lag");
        }
        for (auto it = lags.begin(); it != lags.end(); ++it) {
            if (!it.value().is_number_integer()) {
                fail("root.spotLag." + it.key(), "settlement lag must be an integer");
            }
            const int lag = it.value().get<int>();
            if (lag < 0) {
                fail("root.spotLag." + it.key(), "settlement lag must be non-negative");
            }
            stack.spotLag[it.key()] = lag;
        }
    }
    if (document.contains("fx")) {
        const Json& fx = document.at("fx");
        if (!fx.is_object()) {
            fail("root.fx", "must be an object with a 'spots' array");
        }
        if (fx.contains("spots")) {
            const Json& spots = fx.at("spots");
            if (!spots.is_array()) {
                fail("root.fx.spots", "must be an array");
            }
            for (std::size_t i = 0; i < spots.size(); ++i) {
                const std::string where = "root.fx.spots[" + std::to_string(i) + "]";
                const Json& spotNode = spots[i];
                if (!spotNode.is_object()) {
                    fail(where, "spot must be an object");
                }
                std::string base;
                std::string quote;
                if (spotNode.contains("base") || spotNode.contains("quote")) {
                    base = asString(spotNode, "base", where);
                    quote = asString(spotNode, "quote", where);
                } else {
                    const std::string pair = asString(spotNode, "pair", where);
                    if (pair.size() != 6) {
                        fail(where, "'pair' must be a six-letter base+quote string");
                    }
                    base = pair.substr(0, 3);
                    quote = pair.substr(3, 3);
                }
                FXDescriptor descriptor(base, quote, stack.asOf.toIso());
                if (!descriptor.hasKnownCurrencies()) {
                    fail(where, "unknown base or quote currency '" + descriptor.pair() + "'");
                }
                const double spot = asDouble(spotNode, "spot", where);
                if (!(spot > 0.0)) {
                    fail(where, "'spot' must be positive");
                }
                for (const FxSpotSpec& existing : stack.fxSpots) {
                    if (existing.descriptor.pair() == descriptor.pair()) {
                        fail(where, "duplicate FX spot for pair '" + descriptor.pair() + "'");
                    }
                }
                stack.fxSpots.push_back(FxSpotSpec{std::move(descriptor), spot});
            }
        }
    }
    const Json& curves = require(document, "curves", "root");
    if (!curves.is_array() || curves.empty()) {
        fail("root", "field 'curves' must be a non-empty array");
    }
    for (std::size_t i = 0; i < curves.size(); ++i) {
        const std::string where = "curves[" + std::to_string(i) + "]";
        const Json& node = curves[i];
        if (!node.is_object()) {
            fail(where, "curve spec must be an object");
        }
        CurveSpec spec;
        spec.key = asKey(require(node, "key", where), where + ".key");
        const std::string roleName = std::string(curveRoleToName(spec.key.role));
        if (spec.key.role != CurveRole::Discount && spec.key.role != CurveRole::Forecast &&
            spec.key.role != CurveRole::TenorBasis && spec.key.role != CurveRole::IborOisBasis &&
            spec.key.role != CurveRole::XccyBasis) {
            fail(where, "curve role '" + roleName + "' is not configurable");
        }
        const auto spotLagIt = stack.spotLag.find(spec.key.currency);
        if (spotLagIt != stack.spotLag.end()) {
            spec.spotLag = spotLagIt->second;
        }
        if (node.contains("xccy")) {
            if (spec.key.role != CurveRole::XccyBasis) {
                fail(where, "'xccy' is only valid for XccyBasis curves");
            }
            const Json& xccyNode = node.at("xccy");
            if (!xccyNode.is_object()) {
                fail(where, "'xccy' must be an object");
            }
            if (xccyNode.contains("pair")) {
                spec.xccy.pair = asString(xccyNode, "pair", where + ".xccy");
                if (spec.xccy.pair.size() != 6) {
                    fail(where + ".xccy",
                         "'pair' must be a six-letter base+quote string (e.g. \"EURUSD\")");
                }
            }
            if (xccyNode.contains("notional")) {
                spec.xccy.notional = notionalModeFromName(
                    asString(xccyNode, "notional", where + ".xccy"), where + ".xccy");
            }
            if (xccyNode.contains("basisLeg")) {
                spec.xccy.basisLeg =
                    legFromName(asString(xccyNode, "basisLeg", where + ".xccy"), where + ".xccy");
            }
            if (xccyNode.contains("resetLeg")) {
                spec.xccy.resetLeg =
                    legFromName(asString(xccyNode, "resetLeg", where + ".xccy"), where + ".xccy");
            }
            if (xccyNode.contains("isFxBaseCollateral")) {
                spec.xccy.isFxBaseCollateral =
                    asBool(xccyNode, "isFxBaseCollateral", where + ".xccy");
            }
        }
        if (node.contains("parent")) {
            spec.hasParent = true;
            spec.parent = asKey(node.at("parent"), where + ".parent");
        }
        if (node.contains("discount")) {
            spec.hasDiscount = true;
            spec.discount = asKey(node.at("discount"), where + ".discount");
        }
        for (const CurveSpec& existing : stack.curves) {
            if (existing.key == spec.key) {
                fail(where, "duplicate curve key '" + keyLabel(spec.key) + "'");
            }
        }
        const bool hasPillars = node.contains("pillars");
        const bool hasForecastPillars = node.contains("forecastPillars");
        if (hasPillars && hasForecastPillars) {
            fail(where, "specify either 'pillars' or 'forecastPillars', not both");
        }
        const bool forecastRole = spec.key.role == CurveRole::Forecast;
        if (forecastRole && hasPillars) {
            fail(where, "curve role 'Forecast' requires 'forecastPillars', not 'pillars'");
        }
        if (!forecastRole && hasForecastPillars) {
            fail(where, "curve role '" + roleName + "' requires 'pillars', not 'forecastPillars'");
        }
        if (forecastRole && !hasForecastPillars) {
            fail(where, "missing field 'forecastPillars'");
        }
        if (!forecastRole && !hasPillars) {
            fail(where, "missing field 'pillars'");
        }
        if (node.contains("fraConvexity")) {
            const Json& fraNode = node.at("fraConvexity");
            if (!fraNode.is_object()) {
                fail(where, "'fraConvexity' must be an object");
            }
            const std::string model = asString(fraNode, "model", where + ".fraConvexity");
            if (model != "ShiftedLognormal") {
                fail(where + ".fraConvexity",
                     "unknown FRA convexity model '" + model + "' (expected ShiftedLognormal)");
            }
            spec.fraConvexity.enabled = true;
            spec.fraConvexity.sigmaIndex = asDouble(fraNode, "sigmaIndex", where + ".fraConvexity");
            spec.fraConvexity.sigmaDiscount =
                asDouble(fraNode, "sigmaDiscount", where + ".fraConvexity");
            spec.fraConvexity.correlation =
                asDouble(fraNode, "correlation", where + ".fraConvexity");
            if (spec.fraConvexity.sigmaIndex < 0.0 || spec.fraConvexity.sigmaDiscount < 0.0) {
                fail(where + ".fraConvexity", "volatilities must be non-negative");
            }
            if (std::abs(spec.fraConvexity.correlation) > 1.0) {
                fail(where + ".fraConvexity", "'correlation' must be in [-1, 1]");
            }
        }
        if (node.contains("convexity")) {
            const Json& convexityNode = node.at("convexity");
            if (!convexityNode.is_object()) {
                fail(where, "'convexity' must be an object");
            }
            const std::string model = asString(convexityNode, "model", where + ".convexity");
            if (model != "HullWhite") {
                fail(where + ".convexity",
                     "unknown convexity model '" + model + "' (expected HullWhite)");
            }
            spec.convexity.enabled = true;
            spec.convexity.sigma = asDouble(convexityNode, "sigma", where + ".convexity");
            spec.convexity.meanReversion =
                asDouble(convexityNode, "meanReversion", where + ".convexity");
            if (!(spec.convexity.sigma > 0.0)) {
                fail(where + ".convexity", "'sigma' must be positive");
            }
            if (spec.convexity.meanReversion < 0.0) {
                fail(where + ".convexity", "'meanReversion' must be non-negative");
            }
            if (convexityNode.contains("referenceCurve")) {
                spec.convexity.hasReference = true;
                spec.convexity.referenceCurve =
                    asKey(convexityNode.at("referenceCurve"), where + ".convexity.referenceCurve");
                if (spec.convexity.referenceCurve == spec.key) {
                    fail(where + ".convexity", "'referenceCurve' must differ from the curve key '" +
                                                   keyLabel(spec.key) + "'");
                }
            }
        }
        if (node.contains("zeroDayCounter")) {
            spec.zeroDayCounter = dayCounterFromName(asString(node, "zeroDayCounter", where));
        }
        bool spaceExplicit = false;
        if (node.contains("interpolation")) {
            const Json& interpolation = node.at("interpolation");
            if (!interpolation.is_object()) {
                fail(where, "'interpolation' must be an object");
            }
            if (interpolation.contains("space")) {
                spec.space =
                    spaceFromName(asString(interpolation, "space", where + ".interpolation"),
                                  where + ".interpolation");
                spaceExplicit = true;
            }
            if (interpolation.contains("scheme")) {
                spec.scheme =
                    schemeFromName(asString(interpolation, "scheme", where + ".interpolation"),
                                   where + ".interpolation");
            }
            if (interpolation.contains("tension")) {
                spec.tension = asDouble(interpolation, "tension", where + ".interpolation");
            }
            if (interpolation.contains("switchIndex")) {
                spec.switchIndex = asInt(interpolation, "switchIndex", where + ".interpolation");
            }
        }
        if (hasForecastPillars) {
            if (!spaceExplicit) {
                spec.space = InterpolationSpace::Zero;
            } else if (spec.space != InterpolationSpace::Zero) {
                fail(where + ".interpolation", "forecast curves require 'space': 'Zero' (got '" +
                                                   std::string(interpolationSpaceName(spec.space)) +
                                                   "')");
            }
            if (spec.switchIndex != 1) {
                fail(where + ".interpolation", "'switchIndex' must be 1 for forecast curves (got " +
                                                   std::to_string(spec.switchIndex) + ")");
            }
        }
        if (node.contains("bootstrap")) {
            const Json& bootstrap = node.at("bootstrap");
            if (!bootstrap.is_object()) {
                fail(where, "'bootstrap' must be an object");
            }
            if (bootstrap.contains("method")) {
                spec.bootstrapMethod = asString(bootstrap, "method", where + ".bootstrap");
            }
            if (bootstrap.contains("accuracy")) {
                spec.accuracy = asDouble(bootstrap, "accuracy", where + ".bootstrap");
            }
        }
        if (spec.bootstrapMethod != "IterativeSequential") {
            fail(where, "unsupported bootstrap method '" + spec.bootstrapMethod + "'");
        }
        const bool forecast = spec.key.role == CurveRole::Forecast;
        const bool xccy = spec.key.role == CurveRole::XccyBasis;
        const char* listName = forecast ? "forecastPillars" : "pillars";
        const Json& pillarNodes = require(node, listName, where);
        if (!pillarNodes.is_array() || pillarNodes.empty()) {
            fail(where, std::string("field '") + listName + "' must be a non-empty array");
        }
        for (std::size_t k = 0; k < pillarNodes.size(); ++k) {
            const std::string pillarWhere = where + "." + listName + "[" + std::to_string(k) + "]";
            PillarSpec pillar = parsePillarSpec(pillarNodes[k], pillarWhere, forecast, xccy);
            if (forecast) {
                spec.forecastPillars.push_back(std::move(pillar));
            } else {
                spec.pillars.push_back(std::move(pillar));
            }
        }
        stack.curves.push_back(std::move(spec));
    }
    return stack;
}

CurveStackSpec loadCurveStackSpec(const std::string& path) {
    std::ifstream stream(path);
    if (!stream) {
        throw std::runtime_error("CurveConfig: cannot open file '" + path + "'");
    }
    std::ostringstream buffer;
    buffer << stream.rdbuf();
    return parseCurveStackSpec(buffer.str());
}

namespace {

const CurveSpec* findCurveSpec(const CurveStackSpec& stack, const CurveKey& key) {
    for (const CurveSpec& spec : stack.curves) {
        if (spec.key == key) {
            return &spec;
        }
    }
    return nullptr;
}

const DiscountCurve<double>* resolveReference(const std::vector<CurveReference>& references,
                                              const CurveKey& key) {
    for (const CurveReference& reference : references) {
        if (reference.key == key) {
            return reference.curve;
        }
    }
    return nullptr;
}

PillarKind toDiscountKind(PillarSpec::Kind kind) {
    switch (kind) {
        case PillarSpec::Kind::Deposit:
            return PillarKind::Deposit;
        case PillarSpec::Kind::Repo:
            return PillarKind::Repo;
        case PillarSpec::Kind::Fra:
            return PillarKind::Fra;
        case PillarSpec::Kind::Future:
            return PillarKind::Future;
        case PillarSpec::Kind::OisSwap:
            return PillarKind::OisSwap;
        case PillarSpec::Kind::Irs:
        case PillarSpec::Kind::BasisSwap:
        case PillarSpec::Kind::FxSwap:
        case PillarSpec::Kind::XccySwap:
            break;
    }
    throw std::invalid_argument(
        "CurveConfig: buildCurve: forecast pillar kind on a discount curve");
}

ForecastPillar::Kind toForecastKind(PillarSpec::Kind kind) {
    switch (kind) {
        case PillarSpec::Kind::Deposit:
            return ForecastPillar::Kind::Deposit;
        case PillarSpec::Kind::Fra:
            return ForecastPillar::Kind::Fra;
        case PillarSpec::Kind::Future:
            return ForecastPillar::Kind::Future;
        case PillarSpec::Kind::Irs:
            return ForecastPillar::Kind::Irs;
        case PillarSpec::Kind::BasisSwap:
            return ForecastPillar::Kind::BasisSwap;
        case PillarSpec::Kind::Repo:
        case PillarSpec::Kind::OisSwap:
        case PillarSpec::Kind::FxSwap:
        case PillarSpec::Kind::XccySwap:
            break;
    }
    throw std::invalid_argument(
        "CurveConfig: buildForecastCurve: discount pillar kind on a forecast curve");
}

/// Discount-side mapping: the shared market fields plus the future options.
CurvePillar toCurvePillar(const PillarSpec& pillar) {
    CurvePillar out;
    out.maturity = pillar.maturity;
    out.start = pillar.start;
    out.kind = toDiscountKind(pillar.kind);
    out.quote = pillar.quote;
    out.quoteDayCounter = pillar.quoteDayCounter;
    out.calendar = pillar.calendar;
    out.fixedTenor = pillar.fixedTenor;
    out.paymentLag = pillar.paymentLag;
    out.convexityAdjustment = pillar.convexityAdjustment;
    out.futureStyle = pillar.futureStyle;
    out.averagingStyle = pillar.averagingStyle;
    out.firstCouponFixed = pillar.firstCouponFixed;
    out.firstCouponRate = pillar.firstCouponRate;
    out.businessDayConvention = pillar.businessDayConvention;
    return out;
}

/// Forecast-side mapping: the kind selects which sub-struct receives the
/// market fields; future options stay on the top-level pillar.
ForecastPillar toForecastPillar(const PillarSpec& pillar) {
    ForecastPillar out;
    out.kind = toForecastKind(pillar.kind);
    const auto copyMoneyMarketFields = [&] {
        out.start = pillar.start;
        out.maturity = pillar.maturity;
        out.quote = pillar.quote;
        out.quoteDayCounter = pillar.quoteDayCounter;
        out.calendar = pillar.calendar;
        out.businessDayConvention = pillar.businessDayConvention;
    };
    switch (out.kind) {
        case ForecastPillar::Kind::Deposit:
        case ForecastPillar::Kind::Fra:
            copyMoneyMarketFields();
            break;
        case ForecastPillar::Kind::Future:
            copyMoneyMarketFields();
            out.convexityAdjustment = pillar.convexityAdjustment;
            out.futureStyle = pillar.futureStyle;
            out.averagingStyle = pillar.averagingStyle;
            break;
        case ForecastPillar::Kind::Irs:
            out.irs.start = pillar.start;
            out.irs.maturity = pillar.maturity;
            out.irs.quote = pillar.quote;
            out.irs.floatTenor = pillar.floatTenor;
            out.irs.floatCalendar = pillar.floatCalendar;
            out.irs.floatDayCounter = pillar.floatDayCounter;
            out.irs.fixedTenor = pillar.fixedTenor;
            out.irs.fixedCalendar = pillar.fixedCalendar;
            out.irs.fixedDayCounter = pillar.fixedDayCounter;
            out.irs.businessDayConvention = pillar.businessDayConvention;
            out.irs.paymentLag = pillar.paymentLag;
            out.irs.firstCouponFixed = pillar.firstCouponFixed;
            out.irs.firstCouponRate = pillar.firstCouponRate;
            break;
        case ForecastPillar::Kind::BasisSwap:
            out.basis.maturity = pillar.maturity;
            out.basis.spread = pillar.quote;
            out.basis.spreadOnParentLeg = pillar.spreadOnParentLeg;
            out.basis.floatTenor = pillar.floatTenor;
            out.basis.quoteDayCounter = pillar.quoteDayCounter;
            out.basis.calendar = pillar.calendar;
            out.basis.businessDayConvention = pillar.businessDayConvention;
            break;
    }
    return out;
}

/// Settlement lag of a currency, defaulting to the T+2 majors convention.
int currencySpotLag(const CurveStackSpec& stack, const std::string& currency) {
    const auto it = stack.spotLag.find(currency);
    return it != stack.spotLag.end() ? it->second : 2;
}

/// FX-side mapping: resolves the pair spot and both currencies' settlement
/// lags; an omitted near date settles at the joint spot date.
FxSwapPillar toFxSwapPillar(const CurveStackSpec& stack, const PillarSpec& pillar,
                            const XccyCurveSpec& xccy, const std::string& where) {
    FxSwapPillar out;
    out.start = pillar.start;
    out.maturity = pillar.maturity;
    out.points = pillar.fxPoints;
    out.outright = pillar.fxOutright;
    out.convention = pillar.fxConvention;
    out.dayCounter = pillar.quoteDayCounter;
    out.isFxBaseCollateral =
        pillar.isFxBaseCollateralSet ? pillar.isFxBaseCollateral : xccy.isFxBaseCollateral;
    const std::string pair = pillar.fxPair.empty() ? xccy.pair : pillar.fxPair;
    if (pair.size() != 6) {
        fail(where, "FxSwap pillars require a 'pair' or a curve-level 'xccy.pair'");
    }
    const std::string base = pair.substr(0, 3);
    const std::string quote = pair.substr(3, 3);
    for (const FxSpotSpec& spot : stack.fxSpots) {
        if (spot.descriptor.pair() == pair) {
            out.spot = spot.spot;
            break;
        }
    }
    if (!(out.spot > 0.0)) {
        fail(where, "missing FX spot for pair '" + pair + "'");
    }
    if (out.start.serial() == 0) {
        const int baseLag = currencySpotLag(stack, base);
        const int quoteLag = currencySpotLag(stack, quote);
        out.start =
            datetime::spotDate(stack.asOf, datetime::Calendar::weekendsOnly(), baseLag, quoteLag);
    }
    if (!(out.start < out.maturity)) {
        fail(where, "FxSwap pillar maturity must be after the near date");
    }
    return out;
}

/// Xccy-side mapping: curve-level pair conventions plus the per-pillar leg
/// tenor, calendar, day count and payment lag.
XccyPillar toXccyPillar(const PillarSpec& pillar, const XccyCurveSpec& xccy) {
    XccyPillar out;
    out.maturity = pillar.maturity;
    out.spread = pillar.quote;
    const bool baseCollateral =
        pillar.isFxBaseCollateralSet ? pillar.isFxBaseCollateral : xccy.isFxBaseCollateral;
    const bool basisOnBase = xccy.basisLeg == "Base";
    out.spreadOnForeignLeg = basisOnBase ? !baseCollateral : baseCollateral;
    const bool resetOnBase = xccy.resetLeg == "Base";
    out.resetForeignLeg = resetOnBase ? !baseCollateral : baseCollateral;
    out.notional = pillar.xccyNotionalSet ? pillar.xccyNotional : xccy.notional;
    out.foreignTenor = pillar.foreignTenor;
    out.domesticTenor = pillar.domesticTenor;
    out.foreignCalendar = pillar.foreignCalendar;
    out.domesticCalendar = pillar.domesticCalendar;
    out.foreignDayCounter = pillar.foreignDayCounter;
    out.domesticDayCounter = pillar.domesticDayCounter;
    out.foreignPaymentLag = pillar.foreignPaymentLag;
    out.domesticPaymentLag = pillar.domesticPaymentLag;
    out.foreignBusinessDayConvention = pillar.businessDayConvention;
    out.domesticBusinessDayConvention = pillar.businessDayConvention;
    return out;
}

} // namespace

namespace detail {

void validateConvexityReference(const CurveSpec& spec) {
    if (spec.convexity.hasReference && spec.convexity.referenceCurve == spec.key) {
        throw std::invalid_argument(
            "CurveConfig: convexity reference curve must differ from the curve itself");
    }
}

[[noreturn]] void failMissingFutureConvexity(std::string_view errorPrefix) {
    throw std::invalid_argument(std::string(errorPrefix) +
                                "Future pillars need 'convexityAdjustment' or a curve-level "
                                "'convexity' model");
}

template <typename Pillar, typename Curve>
void applyModelConvexityToPillars(const CurveStackSpec& stack, const CurveSpec& spec,
                                  std::vector<Pillar>& pillars,
                                  const std::vector<std::size_t>& modelFutures,
                                  const Curve& referenceCurve) {
    for (const std::size_t index : modelFutures) {
        const Pillar& pillar = pillars[index];
        const double t1 = datetime::yearFraction(stack.asOf, pillar.start, spec.zeroDayCounter);
        const double t2 = datetime::yearFraction(stack.asOf, pillar.maturity, spec.zeroDayCounter);
        const double accrual =
            datetime::yearFraction(pillar.start, pillar.maturity, pillar.quoteDayCounter);
        const double ratio = referenceCurve.discount(t1) / referenceCurve.discount(t2);
        pillars[index].convexityAdjustment = hullWhiteFuturesAdjustment(
            spec.convexity.sigma, spec.convexity.meanReversion, t1, accrual, ratio);
    }
}

template <typename Pillar, typename Bootstrap, typename Verify>
auto applyModelConvexity(const CurveStackSpec& stack, const CurveSpec& spec,
                         std::vector<Pillar>& pillars,
                         const std::vector<CurveReference>& references,
                         const std::vector<std::size_t>& modelFutures, std::string_view errorPrefix,
                         const char* convergenceMessage, Bootstrap bootstrap, Verify verify)
    -> decltype(bootstrap()) {
    using Curve = decltype(bootstrap());
    if (modelFutures.empty()) {
        return bootstrap();
    }
    if (spec.convexity.hasReference) {
        const DiscountCurve<double>* reference =
            resolveReference(references, spec.convexity.referenceCurve);
        if (reference == nullptr) {
            throw std::invalid_argument(std::string(errorPrefix) + "convexity reference curve '" +
                                        keyLabel(spec.convexity.referenceCurve) +
                                        "' not found in the supplied references");
        }
        applyModelConvexityToPillars(stack, spec, pillars, modelFutures, *reference);
        return bootstrap();
    }
    // Self-referential fixed point: the adjustment is bp-scale, so the bond
    // ratio stabilizes after the first re-bootstrap.
    Curve curve = bootstrap();
    for (int pass = 0; pass < 4; ++pass) {
        std::vector<double> previous;
        previous.reserve(modelFutures.size());
        for (const std::size_t index : modelFutures) {
            previous.push_back(pillars[index].convexityAdjustment);
        }
        applyModelConvexityToPillars(stack, spec, pillars, modelFutures, curve);
        double maxMove = 0.0;
        for (std::size_t i = 0; i < modelFutures.size(); ++i) {
            const double move =
                std::abs(pillars[modelFutures[i]].convexityAdjustment - previous[i]);
            if (move > maxMove) {
                maxMove = move;
            }
        }
        if (maxMove < 1e-15) {
            break;
        }
        curve = bootstrap();
    }
    // Verify the converged adjustments actually reprice the futures.
    applyModelConvexityToPillars(stack, spec, pillars, modelFutures, curve);
    double worst = 0.0;
    for (const std::size_t index : modelFutures) {
        const double check = verify(curve, index);
        if (!std::isfinite(check)) {
            worst = 1e300;
        } else if (std::abs(check) > worst) {
            worst = std::abs(check);
        }
    }
    if (!(worst < 1e-9)) {
        throw std::runtime_error(convergenceMessage);
    }
    return curve;
}

} // namespace detail

DiscountCurve<double> buildCurve(const CurveStackSpec& stack, const CurveSpec& spec,
                                 const std::vector<CurveReference>& references,
                                 std::vector<CurvePillar>* filledPillars) {
    if (spec.pillars.empty()) {
        throw std::invalid_argument("CurveConfig: buildCurve: curve '" + keyLabel(spec.key) +
                                    "' has no pillars");
    }
    if (!spec.forecastPillars.empty()) {
        throw std::invalid_argument("CurveConfig: buildCurve: curve '" + keyLabel(spec.key) +
                                    "' carries forecast pillars; use buildForecastCurve");
    }
    detail::validateConvexityReference(spec);
    std::vector<CurvePillar> pillars;
    std::vector<std::size_t> modelFutures;
    pillars.reserve(spec.pillars.size());
    for (const PillarSpec& pillar : spec.pillars) {
        CurvePillar out = toCurvePillar(pillar);
        if (out.kind == PillarKind::Fra) {
            if (pillar.fraConvexityExponentSet) {
                out.fraConvexityExponent = pillar.fraConvexityExponent;
            } else if (spec.fraConvexity.enabled) {
                if (pillar.start.serial() == 0) {
                    fail("pillar", "FRA convexity requires a 'start' date");
                }
                const datetime::Date adjustedStart =
                    pillar.calendar.adjust(pillar.start, pillar.businessDayConvention);
                const double timeToFixing =
                    datetime::yearFraction(stack.asOf, adjustedStart, spec.zeroDayCounter);
                out.fraConvexityExponent = fraConvexityExponent(
                    spec.fraConvexity.sigmaIndex, spec.fraConvexity.sigmaDiscount,
                    spec.fraConvexity.correlation, timeToFixing);
            }
        }
        if (out.kind == PillarKind::Future) {
            if (pillar.convexityAdjustmentSet) {
                // Explicit market-marked adjustment.
            } else if (spec.convexity.enabled) {
                modelFutures.push_back(pillars.size());
            } else {
                detail::failMissingFutureConvexity("CurveConfig: ");
            }
        }
        pillars.push_back(out);
    }
    const auto bootstrap = [&]() {
        return bootstrapDiscountCurve(stack.asOf, spec.zeroDayCounter, spec.space, spec.scheme,
                                      pillars, spec.accuracy, spec.tension, spec.switchIndex);
    };
    const auto finish = [&](DiscountCurve<double> curve) {
        if (filledPillars != nullptr) {
            *filledPillars = pillars;
        }
        return curve;
    };
    const auto verify = [&](const DiscountCurve<double>& curve, const std::size_t index) {
        return impliedQuote(pillars[index], stack.asOf, curve) - pillars[index].quote;
    };
    return finish(detail::applyModelConvexity(
        stack, spec, pillars, references, modelFutures, "CurveConfig: ",
        "CurveConfig: futures convexity fixed point did not converge", bootstrap, verify));
}

namespace detail {

/// Shared forecast-curve body: validation, pillar mapping, convexity and the
/// parent-generic bootstrap. Both public overloads forward here.
template <typename ParentT>
SpreadCurve<double, ParentT>
buildForecastCurveImpl(const CurveStackSpec& stack, const CurveSpec& spec,
                       std::shared_ptr<const ParentT> parent, const DiscountCurve<double>* discount,
                       const std::vector<CurveReference>& references,
                       std::vector<ForecastPillar>* filledPillars) {
    if (!parent) {
        throw std::invalid_argument("CurveConfig: buildForecastCurve: null parent");
    }
    if (spec.forecastPillars.empty()) {
        throw std::invalid_argument("CurveConfig: buildForecastCurve: curve '" +
                                    keyLabel(spec.key) + "' has no forecast pillars");
    }
    if (!spec.pillars.empty()) {
        throw std::invalid_argument("CurveConfig: buildForecastCurve: curve '" +
                                    keyLabel(spec.key) +
                                    "' carries discount pillars; use buildCurve");
    }
    if (spec.key.role != CurveRole::Forecast) {
        throw std::invalid_argument("CurveConfig: buildForecastCurve: curve role '" +
                                    std::string(curveRoleToName(spec.key.role)) +
                                    "' cannot use 'forecastPillars'");
    }
    if (const CurveSpec* sibling = findCurveSpec(stack, spec.key);
        sibling != nullptr && !sibling->pillars.empty()) {
        throw std::invalid_argument("CurveConfig: buildForecastCurve: forecast curve key '" +
                                    keyLabel(spec.key) +
                                    "' is also a parent curve key in the stack");
    }
    detail::validateConvexityReference(spec);
    std::vector<ForecastPillar> pillars;
    std::vector<std::size_t> modelFutures;
    pillars.reserve(spec.forecastPillars.size());
    for (const PillarSpec& pillar : spec.forecastPillars) {
        ForecastPillar out = toForecastPillar(pillar);
        if (out.kind == ForecastPillar::Kind::Future) {
            if (pillar.convexityAdjustmentSet) {
                // Explicit market-marked adjustment.
            } else if (spec.convexity.enabled) {
                modelFutures.push_back(pillars.size());
            } else {
                detail::failMissingFutureConvexity("CurveConfig: buildForecastCurve: ");
            }
        }
        pillars.push_back(out);
    }
    const auto bootstrap = [&]() {
        return bootstrapForecastCurve<ParentT>(parent, discount, stack.asOf, spec.zeroDayCounter,
                                               spec.scheme, pillars, spec.accuracy, spec.tension);
    };
    const auto finish = [&](SpreadCurve<double, ParentT> curve) {
        if (filledPillars != nullptr) {
            *filledPillars = pillars;
        }
        return curve;
    };
    const auto verify = [&](const SpreadCurve<double, ParentT>& curve, const std::size_t index) {
        return impliedForecastQuote(curve, curve.parent(), pillars[index], stack.asOf,
                                    spec.zeroDayCounter) -
               pillars[index].quote;
    };
    return finish(detail::applyModelConvexity(
        stack, spec, pillars, references, modelFutures, "CurveConfig: buildForecastCurve: ",
        "CurveConfig: forecast futures convexity fixed point did not converge", bootstrap, verify));
}

} // namespace detail

SpreadCurve<double> buildForecastCurve(const CurveStackSpec& stack, const CurveSpec& spec,
                                       std::shared_ptr<const DiscountCurve<double>> parent,
                                       const DiscountCurve<double>* discount,
                                       const std::vector<CurveReference>& references,
                                       std::vector<ForecastPillar>* filledPillars) {
    return detail::buildForecastCurveImpl<DiscountCurve<double>>(
        stack, spec, std::move(parent), discount, references, filledPillars);
}

template <typename ParentT>
    requires CurveNodeProvider<ParentT>
SpreadCurve<double, ParentT> buildForecastCurve(const CurveStackSpec& stack, const CurveSpec& spec,
                                                std::shared_ptr<const ParentT> parent,
                                                const DiscountCurve<double>* discount,
                                                const std::vector<CurveReference>& references,
                                                std::vector<ForecastPillar>* filledPillars) {
    return detail::buildForecastCurveImpl<ParentT>(stack, spec, std::move(parent), discount,
                                                   references, filledPillars);
}

template SpreadCurve<double> buildForecastCurve<DiscountCurve<double>>(
    const CurveStackSpec& stack, const CurveSpec& spec,
    std::shared_ptr<const DiscountCurve<double>> parent, const DiscountCurve<double>* discount,
    const std::vector<CurveReference>& references, std::vector<ForecastPillar>* filledPillars);

template SpreadCurve<double, CurveHandle> buildForecastCurve<CurveHandle>(
    const CurveStackSpec& stack, const CurveSpec& spec, std::shared_ptr<const CurveHandle> parent,
    const DiscountCurve<double>* discount, const std::vector<CurveReference>& references,
    std::vector<ForecastPillar>* filledPillars);

template SpreadCurve<double, SpreadCurve<double, DiscountCurve<double>>>
buildForecastCurve<SpreadCurve<double, DiscountCurve<double>>>(
    const CurveStackSpec& stack, const CurveSpec& spec,
    std::shared_ptr<const SpreadCurve<double, DiscountCurve<double>>> parent,
    const DiscountCurve<double>* discount, const std::vector<CurveReference>& references,
    std::vector<ForecastPillar>* filledPillars);

template SpreadCurve<double, SpreadCurve<double, CurveHandle>>
buildForecastCurve<SpreadCurve<double, CurveHandle>>(
    const CurveStackSpec& stack, const CurveSpec& spec,
    std::shared_ptr<const SpreadCurve<double, CurveHandle>> parent,
    const DiscountCurve<double>* discount, const std::vector<CurveReference>& references,
    std::vector<ForecastPillar>* filledPillars);

namespace {

constexpr std::size_t kNoCurve = static_cast<std::size_t>(-1);

std::size_t findCurveIndex(const CurveStackSpec& stack, const CurveKey& key) {
    for (std::size_t i = 0; i < stack.curves.size(); ++i) {
        if (stack.curves[i].key == key) {
            return i;
        }
    }
    return kNoCurve;
}

std::string curvePathLabel(const CurveStackSpec& stack, const std::vector<std::size_t>& path) {
    std::string out;
    for (std::size_t i = 0; i < path.size(); ++i) {
        if (i != 0) {
            out += " -> ";
        }
        out += keyLabel(stack.curves[path[i]].key);
    }
    return out;
}

} // namespace

std::vector<BuiltCurve> buildStack(const CurveStackSpec& stack) {
    const std::size_t count = stack.curves.size();
    if (count == 0) {
        throw std::invalid_argument("CurveConfig: buildStack: stack has no curves");
    }
    for (std::size_t i = 0; i < count; ++i) {
        for (std::size_t j = i + 1; j < count; ++j) {
            if (stack.curves[i].key == stack.curves[j].key) {
                throw std::invalid_argument("CurveConfig: buildStack: duplicate curve key '" +
                                            keyLabel(stack.curves[i].key) + "'");
            }
        }
    }

    std::vector<std::size_t> parentIndex(count, kNoCurve);
    std::vector<std::size_t> discountIndex(count, kNoCurve);
    std::vector<std::size_t> referenceIndex(count, kNoCurve);
    for (std::size_t i = 0; i < count; ++i) {
        const CurveSpec& spec = stack.curves[i];
        const std::string where = "buildStack: curve '" + keyLabel(spec.key) + "'";
        const bool forecast = spec.key.role == CurveRole::Forecast;
        const bool xccyBasis = spec.key.role == CurveRole::XccyBasis;
        const bool usesParent = forecast || xccyBasis;
        if (forecast && !spec.hasParent) {
            throw std::invalid_argument("CurveConfig: " + where +
                                        " is missing parent: forecast curves must declare a "
                                        "'parent' curve");
        }
        if (xccyBasis && !spec.hasParent) {
            throw std::invalid_argument("CurveConfig: " + where +
                                        " is missing parent: XccyBasis curves must declare a "
                                        "'parent' discount curve");
        }
        if (!usesParent && spec.hasParent) {
            throw std::invalid_argument("CurveConfig: " + where +
                                        " is not a forecast curve and must not declare a parent");
        }
        if (spec.hasParent) {
            parentIndex[i] = findCurveIndex(stack, spec.parent);
            if (parentIndex[i] == kNoCurve) {
                throw std::invalid_argument("CurveConfig: " + where + " references parent curve '" +
                                            keyLabel(spec.parent) + "' not found in the stack");
            }
            if (parentIndex[i] == i) {
                throw std::invalid_argument("CurveConfig: " + where +
                                            " cannot be its own parent (self-reference)");
            }
            if (xccyBasis && stack.curves[parentIndex[i]].key.role != CurveRole::Discount) {
                throw std::invalid_argument("CurveConfig: " + where + " references parent curve '" +
                                            keyLabel(spec.parent) +
                                            "' that is not a discount curve");
            }
        }
        if (spec.hasDiscount) {
            if (!usesParent) {
                throw std::invalid_argument(
                    "CurveConfig: " + where +
                    " is not a forecast curve and must not declare a discount curve");
            }
            discountIndex[i] = findCurveIndex(stack, spec.discount);
            if (discountIndex[i] == kNoCurve) {
                throw std::invalid_argument("CurveConfig: " + where +
                                            " references discount curve '" +
                                            keyLabel(spec.discount) + "' not found in the stack");
            }
            if (discountIndex[i] == i) {
                throw std::invalid_argument("CurveConfig: " + where +
                                            " cannot discount on itself (self-reference)");
            }
            if (stack.curves[discountIndex[i]].key.role != CurveRole::Discount) {
                throw std::invalid_argument(
                    "CurveConfig: " + where + " references curve '" + keyLabel(spec.discount) +
                    "' as exogenous discount curve: exogenous discount must be a discount curve");
            }
        }
        if (spec.convexity.hasReference) {
            referenceIndex[i] = findCurveIndex(stack, spec.convexity.referenceCurve);
            if (referenceIndex[i] == kNoCurve) {
                throw std::invalid_argument(
                    "CurveConfig: " + where + " references convexity curve '" +
                    keyLabel(spec.convexity.referenceCurve) + "' not found in the stack");
            }
            if (referenceIndex[i] == i) {
                throw std::invalid_argument("CurveConfig: " + where +
                                            " cannot be its own convexity reference "
                                            "(self-reference)");
            }
            if (stack.curves[referenceIndex[i]].key.role != CurveRole::Discount) {
                throw std::invalid_argument("CurveConfig: " + where + " references curve '" +
                                            keyLabel(spec.convexity.referenceCurve) +
                                            "': reference must be a discount curve");
            }
        }
    }

    std::vector<unsigned char> state(count, 0); // 0 unvisited, 1 on the path, 2 built
    std::vector<std::shared_ptr<const DiscountCurve<double>>> discountCurves(count);
    std::vector<std::shared_ptr<const CurveHandle>> handles(count);
    std::vector<std::size_t> path;
    std::vector<BuiltCurve> built;
    built.reserve(count);

    const auto buildOne = [&](auto&& self, const std::size_t index, const bool parentEdge) -> void {
        if (state[index] == 2) {
            return;
        }
        if (state[index] == 1) {
            path.push_back(index);
            const std::string where = parentEdge ? "parent graph contains a cycle: "
                                                 : "dependency graph contains a cycle: ";
            throw std::invalid_argument("CurveConfig: buildStack: " + where +
                                        curvePathLabel(stack, path));
        }
        state[index] = 1;
        path.push_back(index);
        if (parentIndex[index] != kNoCurve) {
            self(self, parentIndex[index], true);
        }
        if (discountIndex[index] != kNoCurve) {
            self(self, discountIndex[index], false);
        }
        if (referenceIndex[index] != kNoCurve) {
            self(self, referenceIndex[index], false);
        }
        std::vector<CurveReference> references;
        references.reserve(built.size());
        for (std::size_t i = 0; i < count; ++i) {
            if (state[i] == 2 && stack.curves[i].key.role == CurveRole::Discount) {
                references.push_back(CurveReference{stack.curves[i].key, discountCurves[i].get()});
            }
        }
        const CurveSpec& spec = stack.curves[index];
        if (spec.key.role == CurveRole::Forecast) {
            const std::shared_ptr<const CurveHandle>& parentHandle = handles[parentIndex[index]];
            const DiscountCurve<double>* discount = discountIndex[index] != kNoCurve
                                                        ? discountCurves[discountIndex[index]].get()
                                                        : nullptr;
            SpreadCurve<double, CurveHandle> forecast =
                buildForecastCurve(stack, spec, parentHandle, discount, references);
            handles[index] = CurveHandle::make(
                std::make_shared<const SpreadCurve<double, CurveHandle>>(std::move(forecast)));
        } else if (spec.key.role == CurveRole::XccyBasis) {
            const std::string where = "buildStack: curve '" + keyLabel(spec.key) + "'";
            const DiscountCurve<double>* domesticDiscount =
                discountCurves[parentIndex[index]].get();
            const DiscountCurve<double>* domesticForecast =
                discountIndex[index] != kNoCurve ? discountCurves[discountIndex[index]].get()
                                                 : domesticDiscount;
            std::vector<XccyMixedPillar> pillars;
            pillars.reserve(spec.pillars.size());
            for (const PillarSpec& pillar : spec.pillars) {
                if (pillar.kind == PillarSpec::Kind::FxSwap) {
                    pillars.emplace_back(toFxSwapPillar(stack, pillar, spec.xccy, where));
                } else if (pillar.kind == PillarSpec::Kind::XccySwap) {
                    pillars.emplace_back(toXccyPillar(pillar, spec.xccy));
                } else {
                    throw std::invalid_argument("CurveConfig: " + where +
                                                " pillars must be FxSwap or XccySwap");
                }
            }
            auto discountCurve =
                std::make_shared<const DiscountCurve<double>>(bootstrapMixedXccyDiscountCurve(
                    *domesticDiscount, *domesticForecast, stack.asOf, spec.zeroDayCounter,
                    spec.space, spec.scheme, pillars, spec.accuracy, spec.tension,
                    spec.switchIndex));
            discountCurves[index] = discountCurve;
            handles[index] = CurveHandle::make(std::move(discountCurve));
        } else {
            auto discountCurve =
                std::make_shared<const DiscountCurve<double>>(buildCurve(stack, spec, references));
            discountCurves[index] = discountCurve;
            handles[index] = CurveHandle::make(std::move(discountCurve));
        }
        state[index] = 2;
        path.pop_back();
        built.push_back(BuiltCurve{spec.key, spec.key.role, handles[index]});
    };

    for (std::size_t i = 0; i < count; ++i) {
        buildOne(buildOne, i, false);
    }
    return built;
}

} // namespace quantape::markets
