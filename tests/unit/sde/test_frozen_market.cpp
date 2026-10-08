// test_frozen_market.cpp — multi-curve snapshot, freeze and FX view gates
#include "quantape/datetime/Period.h"
#include "quantape/markets/Curves/Curve.h"
#include "quantape/markets/Curves/CurveConfig.h"
#include "quantape/markets/Curves/CurveOnGrid.h"
#include "quantape/markets/Curves/DiscountCurve.h"
#include "quantape/markets/Curves/MultiCurveSet.h"
#include "quantape/markets/Curves/SpreadCurve.h"
#include "quantape/sde/FrozenMarket.h"
#include "quantape/sde/StateCurveMap.h"

#include <cstddef>
#include <memory>
#include <span>
#include <stdexcept>
#include <utility>
#include <vector>

#include "support/GtestSupport.h"

using namespace quantape;

namespace {

using markets::BuiltCurve;
using markets::CurveHandle;
using markets::CurveKey;
using markets::CurveRole;
using markets::DiscountCurve;
using markets::InterpolationScheme;
using markets::InterpolationSpace;
using markets::MultiCurveSet;
using markets::SpreadCurve;

DiscountCurve<double> makeOisCurve() {
    return DiscountCurve<double>({0.0, 1.0, 2.0, 3.0, 5.0}, {0.0, 0.02, 0.025, 0.03, 0.035},
                                 InterpolationSpace::LogDiscount, InterpolationScheme::Linear);
}

CurveKey oisKey() {
    return CurveKey{"OIS", CurveRole::Discount, datetime::Period(1, datetime::TimeUnit::Days),
                    "OIS"};
}

CurveKey spreadKey() {
    return CurveKey{"OIS", CurveRole::Forecast, datetime::Period(3, datetime::TimeUnit::Months),
                    "OIS"};
}

struct TestStack {
    std::shared_ptr<const DiscountCurve<double>> ois;
    std::shared_ptr<const SpreadCurve<double>> spread;
    CurveHandle::Ptr oisHandle;
    CurveHandle::Ptr spreadHandle;
};

TestStack makeTestStack() {
    TestStack stack;
    stack.ois = std::make_shared<const DiscountCurve<double>>(makeOisCurve());
    stack.spread = std::make_shared<const SpreadCurve<double>>(SpreadCurve<double>(
        stack.ois, {0.0, 1.0, 2.0, 3.0, 5.0}, {0.0, 0.001, 0.0012, 0.0015, 0.002}));
    stack.oisHandle = CurveHandle::make(stack.ois);
    stack.spreadHandle = CurveHandle::make(stack.spread);
    return stack;
}

MultiCurveSet makeSet(const TestStack& stack) {
    return MultiCurveSet::fromBuiltCurves(
        {BuiltCurve{oisKey(), CurveRole::Discount, stack.oisHandle},
         BuiltCurve{spreadKey(), CurveRole::Forecast, stack.spreadHandle}});
}

std::vector<double> freezeTimes() {
    return {0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0};
}

} // namespace

TEST(FrozenMarket, resolvesCurvesByKeyAndRole) {
    const TestStack stack = makeTestStack();
    const MultiCurveSet set = makeSet(stack);
    const sde::FrozenMarket market = sde::FrozenMarket::fromStack(set);

    EXPECT_TRUE(market.size() == 2);
    EXPECT_TRUE(market.curve(oisKey()) == stack.oisHandle.get());
    EXPECT_TRUE(market.curve(spreadKey()) == stack.spreadHandle.get());
    EXPECT_TRUE(market.discountCurve("OIS") == stack.oisHandle.get());
    EXPECT_TRUE(market.discountCurve("OIS", "OIS") == stack.oisHandle.get());
    EXPECT_TRUE(market.forecastCurve("OIS", datetime::Period(3, datetime::TimeUnit::Months)) ==
                stack.spreadHandle.get());

    // Unknown identities resolve to nullptr, never to a throw.
    const CurveKey unknown{"USD", CurveRole::Discount,
                           datetime::Period(1, datetime::TimeUnit::Days), "USD"};
    EXPECT_TRUE(market.curve(unknown) == nullptr);
    EXPECT_TRUE(market.discountCurve("USD") == nullptr);
    EXPECT_TRUE(market.forecastCurve("OIS", datetime::Period(6, datetime::TimeUnit::Months)) ==
                nullptr);
    EXPECT_TRUE(market.fxSpot("EURUSD") == nullptr);
}

TEST(FrozenMarket, snapshotOutlivesTheSourceSet) {
    const TestStack stack = makeTestStack();
    const CurveHandle* ois = nullptr;
    {
        const MultiCurveSet set = makeSet(stack);
        const sde::FrozenMarket snapshot = sde::FrozenMarket::fromStack(set);
        ois = snapshot.curve(oisKey());
    }
    ASSERT_TRUE(ois != nullptr);
    CHECK_CLOSE_REL("snapshot keeps the provider alive", ois->discount(2.0),
                    stack.ois->discount(2.0), 1e-15);
}

TEST(FrozenMarket, freezeRoundTripsProviderValues) {
    const TestStack stack = makeTestStack();
    const sde::FrozenMarket market = sde::FrozenMarket::fromStack(makeSet(stack));
    const std::vector<double> times = freezeTimes();

    for (const CurveKey& key : {oisKey(), spreadKey()}) {
        const CurveHandle* handle = market.curve(key);
        ASSERT_TRUE(handle != nullptr);
        const markets::CurveOnGrid<double> grid = market.freeze(key, times);
        EXPECT_TRUE(grid.size() == times.size());
        EXPECT_TRUE(grid.nSteps() == times.size() - 1);
        for (std::size_t k = 0; k < times.size(); ++k) {
            EXPECT_EQ(grid.discountAt(k), handle->discount(times[k]));
            EXPECT_EQ(grid.zero[k], handle->zero(times[k]));
            if (k + 1 < times.size()) {
                EXPECT_EQ(grid.forwardAt(k), handle->forward(times[k], times[k + 1]));
            }
        }
    }
    const CurveKey unknown{"USD", CurveRole::Discount,
                           datetime::Period(1, datetime::TimeUnit::Days), "USD"};
    EXPECT_THROW((void)market.freeze(unknown, times), std::out_of_range);
}

TEST(CurveOnGrid, genericMaterializeFreezesSpreadCurve) {
    const TestStack stack = makeTestStack();
    const std::vector<double> times = freezeTimes();
    const markets::CurveOnGrid<double> grid = markets::materialize(*stack.spread, times);
    EXPECT_TRUE(grid.size() == times.size());
    for (std::size_t k = 0; k < times.size(); ++k) {
        EXPECT_EQ(grid.discountAt(k), stack.spread->discount(times[k]));
        EXPECT_EQ(grid.zero[k], stack.spread->zero(times[k]));
        if (k + 1 < times.size()) {
            EXPECT_EQ(grid.forwardAt(k), stack.spread->forward(times[k], times[k + 1]));
        }
    }
}

TEST(FrozenMarket, fxSpotViewForwardsOnBothCurves) {
    const TestStack stack = makeTestStack();
    sde::FrozenMarket::FxSpotView eurusd;
    eurusd.pair = "EURUSD";
    eurusd.spot = 1.08;
    eurusd.base = stack.oisHandle;
    eurusd.quote = stack.spreadHandle;
    const sde::FrozenMarket market(std::vector<sde::FrozenMarket::CurveEntry>{},
                                   {std::move(eurusd)});

    const sde::FrozenMarket::FxSpotView* view = market.fxSpot("EURUSD");
    ASSERT_TRUE(view != nullptr);
    CHECK_CLOSE_REL("fx forward", view->forward(2.0),
                    1.08 * stack.oisHandle->discount(2.0) / stack.spreadHandle->discount(2.0),
                    1e-15);

    sde::FrozenMarket::FxSpotView incomplete;
    incomplete.pair = "GBPUSD";
    incomplete.spot = 1.25;
    EXPECT_THROW((void)incomplete.forward(1.0), std::runtime_error);
}

TEST(FrozenMarket, stateCurveSeamAcceptsFrozenHandle) {
    const TestStack stack = makeTestStack();
    const sde::FrozenMarket market = sde::FrozenMarket::fromStack(makeSet(stack));
    const CurveHandle* handle = market.curve(oisKey());
    ASSERT_TRUE(handle != nullptr);

    const std::vector<double> obs{0.0, 1.0, 2.0};
    const std::vector<double> pay{1.0, 2.0, 3.0, 4.0};
    const sde::GaussianStateCurve map = sde::GaussianStateCurve::fromCurve(*handle, obs, pay, {});
    const std::span<const double> zero;
    for (std::size_t k = 0; k < map.obsCount(); ++k) {
        for (std::size_t j = 0; j < map.payCount(); ++j) {
            EXPECT_EQ(map.bond(k, j, zero), handle->discount(pay[j]) / handle->discount(obs[k]));
        }
    }
}

TEST(FrozenMarket, rejectsNullAndDuplicateCurves) {
    const TestStack stack = makeTestStack();
    const sde::FrozenMarket::CurveEntry ois{oisKey(), CurveRole::Discount, stack.oisHandle};
    const sde::FrozenMarket::CurveEntry broken{spreadKey(), CurveRole::Forecast, nullptr};
    EXPECT_THROW(sde::FrozenMarket({broken}), std::invalid_argument);
    EXPECT_THROW(sde::FrozenMarket({ois, ois}), std::invalid_argument);
}
