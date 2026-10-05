#pragma once

#include "quantape/datetime/Calendar.h"
#include "quantape/datetime/Date.h"
#include "quantape/datetime/DayCounter.h"
#include "quantape/datetime/Period.h"
#include "quantape/markets/Curves/Curve.h"
#include "quantape/markets/Curves/CurveBuilder.h"
#include "quantape/markets/Curves/DiscountCurve.h"
#include "quantape/markets/Curves/SpreadCurve.h"

#include <memory>
#include <string>
#include <string_view>
#include <vector>

namespace quantape::markets {
/**
 * @file CurveConfig.h
 * @brief External JSON curve/quote configuration
 *
 * Code carries mechanics, config carries market data. The loader produces
 * plain schema structs; `nlohmann/json` is included only by the loader
 * translation unit (`src/CurveConfig.cpp`), never by consumers of this header.
 */

/// One bootstrap instrument as configured.
struct PillarSpec {
    datetime::Date maturity;
    datetime::Date start; ///< FRA start / future fixing
    PillarKind kind = PillarKind::Deposit;
    double quote = 0.0;
    double convexityAdjustment = 0.0; ///< Futures: added to the fitted forward rate
    FutureStyle futureStyle = FutureStyle::Simple; ///< Futures: underlying style
    AveragingStyle averagingStyle = AveragingStyle::Arithmetic; ///< Averaged futures convention
    double fraConvexityExponent = 0.0; ///< FRA: explicit convexity exponent
    bool fraConvexityExponentSet = false;
    bool firstCouponFixed = false; ///< OIS: first floating coupon already fixed
    double firstCouponRate = 0.0;
    datetime::DayCounter quoteDayCounter{datetime::DayCount::Actual360};
    datetime::Calendar calendar{};
    datetime::Period fixedTenor{1, datetime::TimeUnit::Years};
    int paymentLag = 0; ///< Coupon payment lag in business days
    datetime::BusinessDayConvention businessDayConvention =
        datetime::BusinessDayConvention::ModifiedFollowing;
    bool convexityAdjustmentSet = false; ///< Futures: explicit adjustment present
};

/// One forecast (spread) curve bootstrap instrument as configured. Simple
/// money-market quotes (`Deposit`, `Fra`) use the generic calendar and quote
/// day counter; `Irs` uses the float/fixed conventions; `BasisSwap` uses the
/// float tenor, generic calendar and quote day counter, with `quote` carrying
/// the basis spread; `Future` uses the exchange start/maturity, the generic
/// calendar (the averaged fixing grid) and quote day counter.
struct ForecastPillarSpec {
    datetime::Date start; ///< Required for Future; optional for Deposit/Fra/Irs
    datetime::Date maturity;
    ForecastPillar::Kind kind = ForecastPillar::Kind::Deposit;
    double quote = 0.0; ///< Simple rate, par IRS rate, basis spread or futures rate (decimal)
    double convexityAdjustment = 0.0; ///< Futures: added to the fitted forward rate
    bool convexityAdjustmentSet = false; ///< Futures: explicit adjustment present
    FutureStyle futureStyle = FutureStyle::Simple; ///< Futures: underlying style
    AveragingStyle averagingStyle = AveragingStyle::Arithmetic; ///< Averaged futures convention
    datetime::Period floatTenor{3, datetime::TimeUnit::Months};
    datetime::Period fixedTenor{1, datetime::TimeUnit::Years};
    datetime::Calendar floatCalendar{};
    datetime::Calendar fixedCalendar{};
    datetime::Calendar calendar{};
    datetime::DayCounter quoteDayCounter{datetime::DayCount::Actual360};
    datetime::DayCounter floatDayCounter{datetime::DayCount::Actual360};
    datetime::DayCounter fixedDayCounter{datetime::DayCount::Thirty360BondBasis};
    int paymentLag = 0;
    datetime::BusinessDayConvention businessDayConvention =
        datetime::BusinessDayConvention::ModifiedFollowing;
    bool firstCouponFixed = false;
    double firstCouponRate = 0.0;
    bool spreadOnParentLeg = true;
};

/// Curve-level shifted-lognormal FRA convexity (optional).
struct FraConvexitySpec {
    bool enabled = false;
    double sigmaIndex = 0.0;
    double sigmaDiscount = 0.0;
    double correlation = 0.0;
};

/// Curve-level futures convexity model with an optional explicit reference
/// curve for the bond ratio (resolved by the caller when building).
struct ConvexitySpec {
    bool enabled = false;
    double sigma = 0.0;
    double meanReversion = 0.0;
    bool hasReference = false;
    CurveKey referenceCurve;
};

/// One curve specification: identity, interpolation, bootstrap, pillars.
struct CurveSpec {
    CurveKey key;
    datetime::DayCounter zeroDayCounter{datetime::DayCount::Actual365Fixed};
    InterpolationSpace space = InterpolationSpace::LogDiscount;
    InterpolationScheme scheme = InterpolationScheme::Linear;
    double tension = 0.0;
    int switchIndex = 1;
    double accuracy = 1e-14;
    std::string bootstrapMethod = "IterativeSequential";
    std::vector<PillarSpec> pillars;
    std::vector<ForecastPillarSpec> forecastPillars;
    ConvexitySpec convexity;
    FraConvexitySpec fraConvexity;
};

/// Whole stack snapshot (as-of date plus curve specs).
struct CurveStackSpec {
    datetime::Date asOf;
    std::vector<CurveSpec> curves;
};

/// Parse a JSON document (throws `std::invalid_argument` with context).
CurveStackSpec parseCurveStackSpec(const std::string& jsonText);

/// Read a file and parse it (throws `std::runtime_error` when unreadable).
CurveStackSpec loadCurveStackSpec(const std::string& path);

/// A built curve available when computing reference-based convexity.
struct CurveReference {
    CurveKey key;
    const DiscountCurve<double>* curve = nullptr;
};

/// Build (bootstrap) one configured curve. Future pillars without an explicit
/// per-pillar adjustment take their convexity from the curve-level `convexity`
/// model, using the referenced curve when given and the self-referential
/// fixed point otherwise. `filledPillars`, when given, receives the pillars
/// with the model-derived adjustments applied (for risk use).
DiscountCurve<double> buildCurve(const CurveStackSpec& stack, const CurveSpec& spec,
                                 const std::vector<CurveReference>& references = {},
                                 std::vector<CurvePillar>* filledPillars = nullptr);

/// Build (bootstrap) the forecast spread curve configured by
/// `spec.forecastPillars` over `parent`; `discount`, when given, is the
/// exogenous collateral curve used for discounting swap cashflows (the parent
/// otherwise). `filledPillars`, when given, receives the mapped pillars (for
/// risk use). Forecast futures without an explicit per-pillar
/// `convexityAdjustment` take their convexity from the curve-level `convexity`
/// model, using the referenced curve when given and the self-referential fixed
/// point on the forecast curve otherwise. Throws `std::invalid_argument` on a
/// null parent, an empty forecast pillar list, a curve key that collides with a
/// parent in `stack`, or a future pillar with neither an explicit adjustment
/// nor a curve-level model.
SpreadCurve<double> buildForecastCurve(const CurveStackSpec& stack, const CurveSpec& spec,
                                       std::shared_ptr<const DiscountCurve<double>> parent,
                                       const DiscountCurve<double>* discount = nullptr,
                                       const std::vector<CurveReference>& references = {},
                                       std::vector<ForecastPillar>* filledPillars = nullptr);

/// Config name maps (also used to validate the document).
datetime::DayCounter dayCounterFromName(std::string_view name);
datetime::Calendar calendarFromName(std::string_view name);
std::string conventionName(datetime::BusinessDayConvention convention);
datetime::BusinessDayConvention conventionFromName(std::string_view name);
CurveRole curveRoleFromName(std::string_view name);
std::string_view curveRoleToName(CurveRole role);

} // namespace quantape::markets
