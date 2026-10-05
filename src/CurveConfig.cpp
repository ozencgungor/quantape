#include <cmath>
#include "quantape/markets/Curves/CurveConfig.h"
#include "quantape/markets/Curves/FraConvexity.h"
#include "quantape/markets/Curves/HullWhiteConvexity.h"

#include <nlohmann/json.hpp>

#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>

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
    fail(where, "unknown period unit '" + unit +
                    "' (expected Days, Weeks, Months or Years)");
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

PillarKind pillarKindFromName(std::string_view name, const std::string& where) {
    if (name == "Deposit") {
        return PillarKind::Deposit;
    }
    if (name == "Repo") {
        return PillarKind::Repo;
    }
    if (name == "Fra") {
        return PillarKind::Fra;
    }
    if (name == "Future") {
        return PillarKind::Future;
    }
    if (name == "OisSwap") {
        return PillarKind::OisSwap;
    }
    fail(where, "unknown pillar kind '" + std::string(name) +
                    "' (expected Deposit, Repo, Fra, Future or OisSwap)");
}

ForecastPillar::Kind forecastPillarKindFromName(std::string_view name, const std::string& where) {
    if (name == "Deposit") {
        return ForecastPillar::Kind::Deposit;
    }
    if (name == "Fra") {
        return ForecastPillar::Kind::Fra;
    }
    if (name == "Future") {
        return ForecastPillar::Kind::Future;
    }
    if (name == "Irs") {
        return ForecastPillar::Kind::Irs;
    }
    if (name == "BasisSwap") {
        return ForecastPillar::Kind::BasisSwap;
    }
    fail(where, "unknown forecast pillar kind '" + std::string(name) +
                    "' (expected Deposit, Fra, Future, Irs or BasisSwap)");
}

InterpolationSpace spaceFromName(std::string_view name, const std::string& where) {
    if (name == "Zero") {
        return InterpolationSpace::Zero;
    }
    if (name == "LogDiscount") {
        return InterpolationSpace::LogDiscount;
    }
    fail(where, "unknown interpolation space '" + std::string(name) +
                    "' (expected Zero or LogDiscount)");
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
    if (name == "ACT/ACT") {
        return datetime::DayCounter(DayCount::ActualActualISDA);
    }
    throw std::invalid_argument("CurveConfig: unknown day counter '" + std::string(name) +
                                "' (expected ACT/360, ACT/365F, ACT/365.25, 30/360 or ACT/ACT)");
}

std::string conventionName(datetime::BusinessDayConvention convention) {
    switch (convention) {
        case datetime::BusinessDayConvention::Unadjusted: return "Unadjusted";
        case datetime::BusinessDayConvention::Following: return "Following";
        case datetime::BusinessDayConvention::ModifiedFollowing: return "ModifiedFollowing";
        case datetime::BusinessDayConvention::HalfMonthModifiedFollowing:
            return "HalfMonthModifiedFollowing";
        case datetime::BusinessDayConvention::Preceding: return "Preceding";
        case datetime::BusinessDayConvention::ModifiedPreceding: return "ModifiedPreceding";
        case datetime::BusinessDayConvention::Nearest: return "Nearest";
    }
    return "Unknown";
}

datetime::BusinessDayConvention conventionFromName(std::string_view name) {
    if (name == "Unadjusted") return datetime::BusinessDayConvention::Unadjusted;
    if (name == "Following") return datetime::BusinessDayConvention::Following;
    if (name == "ModifiedFollowing") return datetime::BusinessDayConvention::ModifiedFollowing;
    if (name == "HalfMonthModifiedFollowing") {
        return datetime::BusinessDayConvention::HalfMonthModifiedFollowing;
    }
    if (name == "Preceding") return datetime::BusinessDayConvention::Preceding;
    if (name == "ModifiedPreceding") return datetime::BusinessDayConvention::ModifiedPreceding;
    if (name == "Nearest") return datetime::BusinessDayConvention::Nearest;
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
            spec.key.role != CurveRole::TenorBasis && spec.key.role != CurveRole::IborOisBasis) {
            fail(where, "curve role '" + roleName + "' is not configurable");
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
                fail(where + ".fraConvexity", "unknown FRA convexity model '" + model +
                                                  "' (expected ShiftedLognormal)");
            }
            spec.fraConvexity.enabled = true;
            spec.fraConvexity.sigmaIndex =
                asDouble(fraNode, "sigmaIndex", where + ".fraConvexity");
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
                fail(where + ".convexity", "unknown convexity model '" + model +
                                                "' (expected HullWhite)");
            }
            spec.convexity.enabled = true;
            spec.convexity.sigma =
                asDouble(convexityNode, "sigma", where + ".convexity");
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
                spec.convexity.referenceCurve = asKey(
                    convexityNode.at("referenceCurve"), where + ".convexity.referenceCurve");
                if (spec.convexity.referenceCurve == spec.key) {
                    fail(where + ".convexity",
                         "'referenceCurve' must differ from the curve key '" +
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
                fail(where + ".interpolation",
                     "forecast curves require 'space': 'Zero' (got '" +
                         std::string(interpolationSpaceName(spec.space)) + "')");
            }
            if (spec.switchIndex != 1) {
                fail(where + ".interpolation",
                     "'switchIndex' must be 1 for forecast curves (got " +
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
        const Json emptyPillars = Json::array();
        const Json& pillars = hasPillars ? node.at("pillars") : emptyPillars;
        if (hasPillars && (!pillars.is_array() || pillars.empty())) {
            fail(where, "field 'pillars' must be a non-empty array");
        }
        for (std::size_t k = 0; k < pillars.size(); ++k) {
            const std::string pillarWhere = where + ".pillars[" + std::to_string(k) + "]";
            const Json& pillarNode = pillars[k];
            PillarSpec pillar;
            pillar.maturity = asDate(pillarNode, "maturity", pillarWhere);
            if (pillarNode.contains("start")) {
                pillar.start = asDate(pillarNode, "start", pillarWhere);
            }
            pillar.kind = pillarKindFromName(asString(pillarNode, "kind", pillarWhere), pillarWhere);
            if ((pillar.kind == PillarKind::Fra || pillar.kind == PillarKind::Future) &&
                !pillarNode.contains("start")) {
                fail(pillarWhere, "Fra/Future pillars require a 'start' date");
            }
            pillar.quote = asDouble(pillarNode, "quote", pillarWhere);
            if (pillarNode.contains("quoteDayCounter")) {
                pillar.quoteDayCounter =
                    dayCounterFromName(asString(pillarNode, "quoteDayCounter", pillarWhere));
            }
            if (pillarNode.contains("calendar")) {
                pillar.calendar = calendarFromName(asString(pillarNode, "calendar", pillarWhere));
            }
            if (pillarNode.contains("businessDayConvention")) {
                pillar.businessDayConvention = conventionFromName(
                    asString(pillarNode, "businessDayConvention", pillarWhere));
            }
            if (pillarNode.contains("fixedTenor")) {
                pillar.fixedTenor = asPeriod(pillarNode.at("fixedTenor"), pillarWhere + ".fixedTenor");
            }
            if (pillarNode.contains("firstFixing")) {
                if (pillar.kind != PillarKind::OisSwap) {
                    fail(pillarWhere, "'firstFixing' is only valid for OisSwap pillars");
                }
                pillar.firstCouponRate = asDouble(pillarNode, "firstFixing", pillarWhere);
                pillar.firstCouponFixed = true;
            }
            if (pillarNode.contains("fraConvexityExponent")) {
                if (pillar.kind != PillarKind::Fra) {
                    fail(pillarWhere, "'fraConvexityExponent' is only valid for Fra pillars");
                }
                pillar.fraConvexityExponent =
                    asDouble(pillarNode, "fraConvexityExponent", pillarWhere);
                pillar.fraConvexityExponentSet = true;
            }
            if (pillarNode.contains("convexityAdjustment")) {
                if (pillar.kind != PillarKind::Future) {
                    fail(pillarWhere, "'convexityAdjustment' is only valid for Future pillars");
                }
                pillar.convexityAdjustment =
                    asDouble(pillarNode, "convexityAdjustment", pillarWhere);
                pillar.convexityAdjustmentSet = true;
            }
            if (pillarNode.contains("style")) {
                if (pillar.kind != PillarKind::Future) {
                    fail(pillarWhere, "'style' is only valid for Future pillars");
                }
                const std::string style = asString(pillarNode, "style", pillarWhere);
                if (style == "Simple") {
                    pillar.futureStyle = FutureStyle::Simple;
                } else if (style == "Compounded") {
                    pillar.futureStyle = FutureStyle::Compounded;
                } else if (style == "Averaged") {
                    pillar.futureStyle = FutureStyle::Averaged;
                } else {
                    fail(pillarWhere, "unknown future style '" + style +
                                          "' (expected Simple, Compounded or Averaged)");
                }
            }
            if (pillarNode.contains("averaging")) {
                if (pillar.kind != PillarKind::Future) {
                    fail(pillarWhere, "'averaging' is only valid for Future pillars");
                }
                if (pillar.futureStyle != FutureStyle::Averaged) {
                    fail(pillarWhere,
                         "'averaging' is only valid for Future pillars with 'style': 'Averaged'");
                }
                const std::string averaging = asString(pillarNode, "averaging", pillarWhere);
                if (averaging == "Arithmetic") {
                    pillar.averagingStyle = AveragingStyle::Arithmetic;
                } else if (averaging == "Compounded") {
                    pillar.averagingStyle = AveragingStyle::Compounded;
                } else {
                    fail(pillarWhere, "unknown averaging style '" + averaging +
                                          "' (expected Arithmetic or Compounded)");
                }
            }
            if (pillarNode.contains("paymentLag")) {
                if (pillar.kind != PillarKind::OisSwap) {
                    fail(pillarWhere, "'paymentLag' is only valid for OisSwap pillars");
                }
                pillar.paymentLag = asInt(pillarNode, "paymentLag", pillarWhere);
                if (pillar.paymentLag < 0) {
                    fail(pillarWhere, "'paymentLag' must be non-negative");
                }
            }
            spec.pillars.push_back(pillar);
        }
        if (node.contains("forecastPillars")) {
            const Json& forecastPillars = node.at("forecastPillars");
            if (!forecastPillars.is_array() || forecastPillars.empty()) {
                fail(where, "field 'forecastPillars' must be a non-empty array");
            }
            for (std::size_t k = 0; k < forecastPillars.size(); ++k) {
                const std::string pillarWhere =
                    where + ".forecastPillars[" + std::to_string(k) + "]";
                const Json& pillarNode = forecastPillars[k];
                ForecastPillarSpec pillar;
                pillar.maturity = asDate(pillarNode, "maturity", pillarWhere);
                if (pillarNode.contains("start")) {
                    pillar.start = asDate(pillarNode, "start", pillarWhere);
                }
                pillar.kind = forecastPillarKindFromName(
                    asString(pillarNode, "kind", pillarWhere), pillarWhere);
                if (pillar.kind == ForecastPillar::Kind::Future &&
                    !pillarNode.contains("start")) {
                    fail(pillarWhere, "Future forecast pillars require a 'start' date");
                }
                if (pillarNode.contains("quote")) {
                    pillar.quote = asDouble(pillarNode, "quote", pillarWhere);
                } else if (pillar.kind == ForecastPillar::Kind::BasisSwap &&
                           pillarNode.contains("spread")) {
                    pillar.quote = asDouble(pillarNode, "spread", pillarWhere);
                } else {
                    fail(pillarWhere, "missing field 'quote'");
                }
                if (pillarNode.contains("style")) {
                    if (pillar.kind != ForecastPillar::Kind::Future) {
                        fail(pillarWhere, "'style' is only valid for Future forecast pillars");
                    }
                    const std::string style = asString(pillarNode, "style", pillarWhere);
                    if (style == "Simple") {
                        pillar.futureStyle = FutureStyle::Simple;
                    } else if (style == "Compounded") {
                        pillar.futureStyle = FutureStyle::Compounded;
                    } else if (style == "Averaged") {
                        pillar.futureStyle = FutureStyle::Averaged;
                    } else {
                        fail(pillarWhere, "unknown future style '" + style +
                                              "' (expected Simple, Compounded or Averaged)");
                    }
                }
                if (pillarNode.contains("averaging")) {
                    if (pillar.kind != ForecastPillar::Kind::Future) {
                        fail(pillarWhere,
                             "'averaging' is only valid for Future forecast pillars");
                    }
                    if (pillar.futureStyle != FutureStyle::Averaged) {
                        fail(pillarWhere,
                             "'averaging' is only valid for Future forecast pillars with "
                             "'style': 'Averaged'");
                    }
                    const std::string averaging = asString(pillarNode, "averaging", pillarWhere);
                    if (averaging == "Arithmetic") {
                        pillar.averagingStyle = AveragingStyle::Arithmetic;
                    } else if (averaging == "Compounded") {
                        pillar.averagingStyle = AveragingStyle::Compounded;
                    } else {
                        fail(pillarWhere, "unknown averaging style '" + averaging +
                                              "' (expected Arithmetic or Compounded)");
                    }
                }
                if (pillarNode.contains("convexityAdjustment")) {
                    if (pillar.kind != ForecastPillar::Kind::Future) {
                        fail(pillarWhere,
                             "'convexityAdjustment' is only valid for Future forecast pillars");
                    }
                    pillar.convexityAdjustment =
                        asDouble(pillarNode, "convexityAdjustment", pillarWhere);
                    pillar.convexityAdjustmentSet = true;
                }
                if (pillarNode.contains("calendar")) {
                    pillar.calendar = calendarFromName(asString(pillarNode, "calendar", pillarWhere));
                } else if (pillarNode.contains("floatCalendar")) {
                    pillar.calendar =
                        calendarFromName(asString(pillarNode, "floatCalendar", pillarWhere));
                }
                if (pillarNode.contains("quoteDayCounter")) {
                    pillar.quoteDayCounter = dayCounterFromName(
                        asString(pillarNode, "quoteDayCounter", pillarWhere));
                } else if (pillarNode.contains("dayCounter")) {
                    pillar.quoteDayCounter =
                        dayCounterFromName(asString(pillarNode, "dayCounter", pillarWhere));
                } else if (pillarNode.contains("floatDayCounter")) {
                    pillar.quoteDayCounter = dayCounterFromName(
                        asString(pillarNode, "floatDayCounter", pillarWhere));
                }
                if (pillarNode.contains("businessDayConvention")) {
                    pillar.businessDayConvention = conventionFromName(
                        asString(pillarNode, "businessDayConvention", pillarWhere));
                }
                if (pillarNode.contains("floatTenor")) {
                    pillar.floatTenor =
                        asPeriod(pillarNode.at("floatTenor"), pillarWhere + ".floatTenor");
                }
                if (pillarNode.contains("fixedTenor")) {
                    pillar.fixedTenor =
                        asPeriod(pillarNode.at("fixedTenor"), pillarWhere + ".fixedTenor");
                }
                if (pillarNode.contains("floatCalendar")) {
                    pillar.floatCalendar =
                        calendarFromName(asString(pillarNode, "floatCalendar", pillarWhere));
                } else {
                    pillar.floatCalendar = pillar.calendar;
                }
                if (pillarNode.contains("fixedCalendar")) {
                    pillar.fixedCalendar =
                        calendarFromName(asString(pillarNode, "fixedCalendar", pillarWhere));
                } else {
                    pillar.fixedCalendar = pillar.calendar;
                }
                if (pillarNode.contains("floatDayCounter")) {
                    pillar.floatDayCounter =
                        dayCounterFromName(asString(pillarNode, "floatDayCounter", pillarWhere));
                } else {
                    pillar.floatDayCounter = pillar.quoteDayCounter;
                }
                if (pillarNode.contains("fixedDayCounter")) {
                    pillar.fixedDayCounter =
                        dayCounterFromName(asString(pillarNode, "fixedDayCounter", pillarWhere));
                }
                if (pillarNode.contains("paymentLag")) {
                    if (pillar.kind != ForecastPillar::Kind::Irs) {
                        fail(pillarWhere, "'paymentLag' is only valid for Irs forecast pillars");
                    }
                    pillar.paymentLag = asInt(pillarNode, "paymentLag", pillarWhere);
                    if (pillar.paymentLag < 0) {
                        fail(pillarWhere, "'paymentLag' must be non-negative");
                    }
                }
                if (pillarNode.contains("firstFixing")) {
                    if (pillar.kind != ForecastPillar::Kind::Irs) {
                        fail(pillarWhere, "'firstFixing' is only valid for Irs forecast pillars");
                    }
                    pillar.firstCouponRate = asDouble(pillarNode, "firstFixing", pillarWhere);
                    pillar.firstCouponFixed = true;
                } else if (pillarNode.contains("firstCouponRate")) {
                    if (pillar.kind != ForecastPillar::Kind::Irs) {
                        fail(pillarWhere,
                             "'firstCouponRate' is only valid for Irs forecast pillars");
                    }
                    pillar.firstCouponRate =
                        asDouble(pillarNode, "firstCouponRate", pillarWhere);
                    pillar.firstCouponFixed = true;
                }
                if (pillarNode.contains("firstCouponFixed")) {
                    if (pillar.kind != ForecastPillar::Kind::Irs) {
                        fail(pillarWhere,
                             "'firstCouponFixed' is only valid for Irs forecast pillars");
                    }
                    pillar.firstCouponFixed = asBool(pillarNode, "firstCouponFixed", pillarWhere);
                }
                if (pillarNode.contains("spreadOnParentLeg")) {
                    if (pillar.kind != ForecastPillar::Kind::BasisSwap) {
                        fail(pillarWhere,
                             "'spreadOnParentLeg' is only valid for BasisSwap forecast pillars");
                    }
                    pillar.spreadOnParentLeg =
                        asBool(pillarNode, "spreadOnParentLeg", pillarWhere);
                }
                if (pillar.kind == ForecastPillar::Kind::Irs &&
                    (!pillarNode.contains("floatTenor") || !pillarNode.contains("fixedTenor"))) {
                    fail(pillarWhere,
                         "Irs forecast pillars require 'floatTenor' and 'fixedTenor'");
                }
                spec.forecastPillars.push_back(pillar);
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
                         const std::vector<std::size_t>& modelFutures,
                         std::string_view errorPrefix, const char* convergenceMessage,
                         Bootstrap bootstrap, Verify verify) -> decltype(bootstrap()) {
    using Curve = decltype(bootstrap());
    if (modelFutures.empty()) {
        return bootstrap();
    }
    if (spec.convexity.hasReference) {
        const DiscountCurve<double>* reference =
            resolveReference(references, spec.convexity.referenceCurve);
        if (reference == nullptr) {
            throw std::invalid_argument(std::string(errorPrefix) +
                                        "convexity reference curve '" +
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
        CurvePillar out;
        out.maturity = pillar.maturity;
        out.start = pillar.start;
        out.kind = pillar.kind;
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
                out.fraConvexityExponent =
                    fraConvexityExponent(spec.fraConvexity.sigmaIndex,
                                         spec.fraConvexity.sigmaDiscount,
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

SpreadCurve<double> buildForecastCurve(const CurveStackSpec& stack, const CurveSpec& spec,
                                       std::shared_ptr<const DiscountCurve<double>> parent,
                                       const DiscountCurve<double>* discount,
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
        throw std::invalid_argument("CurveConfig: buildForecastCurve: curve '" + keyLabel(spec.key) +
                                    "' carries discount pillars; use buildCurve");
    }
    if (spec.key.role != CurveRole::Forecast) {
        throw std::invalid_argument(
            "CurveConfig: buildForecastCurve: curve role '" +
            std::string(curveRoleToName(spec.key.role)) + "' cannot use 'forecastPillars'");
    }
    if (const CurveSpec* sibling = findCurveSpec(stack, spec.key);
        sibling != nullptr && !sibling->pillars.empty()) {
        throw std::invalid_argument(
            "CurveConfig: buildForecastCurve: forecast curve key '" + keyLabel(spec.key) +
            "' is also a parent curve key in the stack");
    }
    detail::validateConvexityReference(spec);
    std::vector<ForecastPillar> pillars;
    std::vector<std::size_t> modelFutures;
    pillars.reserve(spec.forecastPillars.size());
    for (const ForecastPillarSpec& pillar : spec.forecastPillars) {
        ForecastPillar out;
        out.kind = pillar.kind;
        switch (pillar.kind) {
            case ForecastPillar::Kind::Deposit:
            case ForecastPillar::Kind::Fra:
                out.start = pillar.start;
                out.maturity = pillar.maturity;
                out.quote = pillar.quote;
                out.quoteDayCounter = pillar.quoteDayCounter;
                out.calendar = pillar.calendar;
                out.businessDayConvention = pillar.businessDayConvention;
                break;
            case ForecastPillar::Kind::Future:
                out.start = pillar.start;
                out.maturity = pillar.maturity;
                out.quote = pillar.quote;
                out.convexityAdjustment = pillar.convexityAdjustment;
                out.futureStyle = pillar.futureStyle;
                out.averagingStyle = pillar.averagingStyle;
                out.quoteDayCounter = pillar.quoteDayCounter;
                out.calendar = pillar.calendar;
                out.businessDayConvention = pillar.businessDayConvention;
                if (pillar.convexityAdjustmentSet) {
                    // Explicit market-marked adjustment.
                } else if (spec.convexity.enabled) {
                    modelFutures.push_back(pillars.size());
                } else {
                    detail::failMissingFutureConvexity("CurveConfig: buildForecastCurve: ");
                }
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
        pillars.push_back(out);
    }
    const auto bootstrap = [&]() {
        return bootstrapForecastCurve<DiscountCurve<double>>(
            parent, discount, stack.asOf, spec.zeroDayCounter, spec.scheme, pillars,
            spec.accuracy, spec.tension);
    };
    const auto finish = [&](SpreadCurve<double> curve) {
        if (filledPillars != nullptr) {
            *filledPillars = pillars;
        }
        return curve;
    };
    const auto verify = [&](const SpreadCurve<double>& curve, const std::size_t index) {
        return impliedForecastQuote(curve, curve.parent(), pillars[index], stack.asOf,
                                    spec.zeroDayCounter) -
               pillars[index].quote;
    };
    return finish(detail::applyModelConvexity(
        stack, spec, pillars, references, modelFutures, "CurveConfig: buildForecastCurve: ",
        "CurveConfig: forecast futures convexity fixed point did not converge", bootstrap,
        verify));
}

} // namespace quantape::markets
