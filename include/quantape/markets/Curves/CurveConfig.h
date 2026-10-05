#pragma once

#include "quantape/datetime/Calendar.h"
#include "quantape/datetime/Date.h"
#include "quantape/datetime/DayCounter.h"
#include "quantape/datetime/Period.h"
#include "quantape/markets/Curves/Curve.h"
#include "quantape/markets/Curves/CurveBuilder.h"
#include "quantape/markets/Curves/DiscountCurve.h"
#include "quantape/markets/Curves/SpreadCurve.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
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

/// One configured bootstrap instrument for either curve side. `kind` selects
/// the concrete instrument and which convention fields apply: discount curves
/// use `Deposit`, `Repo`, `Fra`, `Future` and `OisSwap`; forecast curves use
/// `Deposit`, `Fra`, `Future`, `Irs` and `BasisSwap`. Simple money-market
/// quotes (`Deposit`, `Fra`, `Future`) use `calendar` and `quoteDayCounter`;
/// `Irs` uses the float/fixed conventions; `BasisSwap` uses `floatTenor`,
/// `calendar` and `quoteDayCounter`, with `quote` carrying the basis spread;
/// `OisSwap` uses `calendar`, `quoteDayCounter`, `fixedTenor` and `paymentLag`.
/// `start` is required for discount `Fra`/`Future` and forecast `Future`.
/// `convexityAdjustmentSet` and `fraConvexityExponentSet` record whether the
/// optional fields were present, since an explicit zero adjustment differs
/// from an omitted one.
struct PillarSpec {
    enum class Kind : std::uint8_t {
        Deposit,
        Repo,
        Fra,
        Future,
        OisSwap,
        Irs,
        BasisSwap,
    };

    Kind kind = Kind::Deposit;
    datetime::Date maturity;
    datetime::Date start; ///< FRA start / future fixing / OIS effective date
    double quote = 0.0;   ///< Simple rate, par swap rate, basis spread or futures rate (decimal)
    double convexityAdjustment = 0.0;              ///< Futures: added to the fitted forward rate
    bool convexityAdjustmentSet = false;           ///< Futures: explicit adjustment present
    FutureStyle futureStyle = FutureStyle::Simple; ///< Futures: underlying style
    AveragingStyle averagingStyle = AveragingStyle::Arithmetic; ///< Averaged futures convention
    double fraConvexityExponent = 0.0; ///< FRA: explicit convexity exponent
    bool fraConvexityExponentSet = false;
    bool firstCouponFixed = false; ///< OIS/IRS: first floating coupon already fixed
    double firstCouponRate = 0.0;
    datetime::DayCounter quoteDayCounter{datetime::DayCount::Actual360};
    datetime::Calendar calendar{};
    datetime::Period fixedTenor{1, datetime::TimeUnit::Years};
    int paymentLag = 0; ///< Coupon payment lag in business days
    datetime::BusinessDayConvention businessDayConvention =
        datetime::BusinessDayConvention::ModifiedFollowing;
    datetime::Period floatTenor{3, datetime::TimeUnit::Months};
    datetime::Calendar floatCalendar{};
    datetime::Calendar fixedCalendar{};
    datetime::DayCounter floatDayCounter{datetime::DayCount::Actual360};
    datetime::DayCounter fixedDayCounter{datetime::DayCount::Thirty360BondBasis};
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

/// One curve specification: identity, dependencies, interpolation, bootstrap,
/// pillars. `parent` is the curve a forecast index spreads over; `discount` is
/// the exogenous collateral curve used to discount forecast swap cashflows.
/// The `has*` flags distinguish an omitted key from a default-valued key. Both
/// keys are optional in the document and validated by `buildStack`.
struct CurveSpec {
    CurveKey key;
    CurveKey parent;
    CurveKey discount;
    bool hasParent = false;
    bool hasDiscount = false;
    datetime::DayCounter zeroDayCounter{datetime::DayCount::Actual365Fixed};
    InterpolationSpace space = InterpolationSpace::LogDiscount;
    InterpolationScheme scheme = InterpolationScheme::Linear;
    double tension = 0.0;
    int switchIndex = 1;
    double accuracy = 1e-14;
    std::string bootstrapMethod = "IterativeSequential";
    std::vector<PillarSpec> pillars;
    std::vector<PillarSpec> forecastPillars;
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

/// Overload taking any node provider as parent, so a forecast curve can parent
/// on a previously built forecast curve (directly or through a `CurveHandle`).
template <typename ParentT>
    requires CurveNodeProvider<ParentT>
SpreadCurve<double, ParentT>
buildForecastCurve(const CurveStackSpec& stack, const CurveSpec& spec,
                   std::shared_ptr<const ParentT> parent,
                   const DiscountCurve<double>* discount = nullptr,
                   const std::vector<CurveReference>& references = {},
                   std::vector<ForecastPillar>* filledPillars = nullptr);

namespace detail {
template <typename Provider>
class CurveHandleOf;
} // namespace detail

/// Type-erased double provider over a curve node grid, so forecast curves can
/// parent on forecast curves at any depth. `make` accepts any
/// `CurveNodeProvider`; the wrapped curve must outlive the handle.
class CurveHandle {
public:
    using Ptr = std::shared_ptr<const CurveHandle>;

    virtual ~CurveHandle() = default;

    virtual double discount(double t) const = 0;
    virtual double zero(double t) const = 0;
    virtual double forward(double t1, double t2) const = 0;
    virtual std::size_t size() const = 0;
    virtual void zeroNodeWeights(double t, std::vector<double>& out) const = 0;
    virtual const datetime::DayCounter& zeroDayCounter() const = 0;
    virtual const std::vector<double>& times() const = 0;

    /// Wrap a shared curve provider.
    template <typename Provider>
        requires CurveNodeProvider<Provider>
    static Ptr make(std::shared_ptr<const Provider> provider);

    /// Wrap a copy of a curve provider.
    template <typename Provider>
        requires CurveNodeProvider<Provider>
    static Ptr make(const Provider& provider);
};

/// One built stack curve: identity, role and type-erased provider.
struct BuiltCurve {
    CurveKey key;
    CurveRole role = CurveRole::Discount;
    std::shared_ptr<const CurveHandle> curve;
};

/// Build every curve in `spec` in dependency order (parents, discounts and
/// convexity references first) and return them in build order. Throws
/// `std::invalid_argument` on duplicate keys, a forecast curve without a
/// parent, a non-forecast curve declaring one, missing or self-referencing
/// keys, a cyclic parent graph or a non-discount convexity reference.
std::vector<BuiltCurve> buildStack(const CurveStackSpec& spec);

template <typename Provider>
    requires CurveNodeProvider<Provider>
CurveHandle::Ptr CurveHandle::make(std::shared_ptr<const Provider> provider) {
    if (!provider) {
        throw std::invalid_argument("CurveHandle: null provider");
    }
    return std::make_shared<const detail::CurveHandleOf<Provider>>(std::move(provider));
}

template <typename Provider>
    requires CurveNodeProvider<Provider>
CurveHandle::Ptr CurveHandle::make(const Provider& provider) {
    return make(std::make_shared<const Provider>(provider));
}

namespace detail {

/// Concrete `CurveHandle` over a node-provider curve.
template <typename Provider>
class CurveHandleOf final : public CurveHandle {
public:
    explicit CurveHandleOf(std::shared_ptr<const Provider> provider)
        : m_provider(std::move(provider)) {}

    double discount(double t) const override { return m_provider->discount(t); }
    double zero(double t) const override { return m_provider->zero(t); }
    double forward(double t1, double t2) const override { return m_provider->forward(t1, t2); }
    std::size_t size() const override { return m_provider->size(); }
    void zeroNodeWeights(double t, std::vector<double>& out) const override {
        m_provider->zeroNodeWeights(t, out);
    }
    const datetime::DayCounter& zeroDayCounter() const override {
        return m_provider->zeroDayCounter();
    }
    const std::vector<double>& times() const override {
        if constexpr (requires { m_provider->times(); }) {
            return m_provider->times();
        } else if constexpr (requires { m_provider->spreadNodes().times(); }) {
            return m_provider->spreadNodes().times();
        } else {
            static const std::vector<double> empty;
            return empty;
        }
    }

private:
    std::shared_ptr<const Provider> m_provider;
};

} // namespace detail

/// Config name maps (also used to validate the document).
datetime::DayCounter dayCounterFromName(std::string_view name);
datetime::Calendar calendarFromName(std::string_view name);
std::string conventionName(datetime::BusinessDayConvention convention);
datetime::BusinessDayConvention conventionFromName(std::string_view name);
CurveRole curveRoleFromName(std::string_view name);
std::string_view curveRoleToName(CurveRole role);

} // namespace quantape::markets
