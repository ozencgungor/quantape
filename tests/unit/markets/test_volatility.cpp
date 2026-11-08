#include "quantape/markets/Volatility/EQDVolatility.h"
#include "quantape/markets/Volatility/FXVolatility.h"

#include <vector>

#include "support/GtestSupport.h"

namespace {

/// A non-separable surface: transposing it changes every off-diagonal node.
double surfaceValue(double expiry, double delta) {
    return 0.10 + 0.02 * expiry + 0.05 * delta + 0.30 * expiry * delta;
}

std::vector<std::vector<double>> makeSurface(const std::vector<double>& expiries,
                                             const std::vector<double>& deltas) {
    std::vector<std::vector<double>> surface(expiries.size(),
                                             std::vector<double>(deltas.size(), 0.0));
    for (std::size_t i = 0; i < expiries.size(); ++i) {
        for (std::size_t j = 0; j < deltas.size(); ++j) {
            surface[i][j] = surfaceValue(expiries[i], deltas[j]);
        }
    }
    return surface;
}

} // namespace

// The surface is stored as z[expiry][delta] while BicubicInterpolation indexes
// z[y][x], so the axes must be handed over as (deltas, expiries) and queried at
// (delta, expiry). Every node of a non-square surface must be reproduced.
TEST(FXVolatilityTest, reproducesEveryNodeOfANonSquareSurface) {
    const std::vector<double> expiries{0.25, 0.5, 1.0, 2.0, 5.0};
    const std::vector<double> deltas{0.10, 0.25, 0.50, 0.75};
    const auto surface = makeSurface(expiries, deltas);
    const quantape::markets::FXVolatility<double> fx(expiries, deltas, surface, 1.10);
    for (std::size_t i = 0; i < expiries.size(); ++i) {
        for (std::size_t j = 0; j < deltas.size(); ++j) {
            SCOPED_TRACE(::testing::Message()
                         << "expiry " << expiries[i] << " delta " << deltas[j]);
            CHECK_CLOSE("surface node", fx.vol(expiries[i], deltas[j]), surface[i][j], 1e-12);
        }
    }
    // And a point in between must be close to the same smooth function.
    CHECK_CLOSE_REL("interpolated point", fx.vol(0.75, 0.30), surfaceValue(0.75, 0.30), 2e-2);
}

TEST(FXVolatilityTest, squareSurfaceIsNotTransposed) {
    const std::vector<double> expiries{0.5, 1.0, 2.0};
    const std::vector<double> deltas{0.10, 0.25, 0.50};
    const auto surface = makeSurface(expiries, deltas);
    const quantape::markets::FXVolatility<double> fx(expiries, deltas, surface, 1.10);
    // A transposed construction would swap these two nodes.
    CHECK_CLOSE("expiry 0.5, delta 0.1", fx.vol(0.5, 0.10), surface[0][0], 1e-12);
    CHECK_CLOSE("expiry 2.0, delta 0.1", fx.vol(2.0, 0.10), surface[2][0], 1e-12);
    CHECK_CLOSE("expiry 0.5, delta 0.5", fx.vol(0.5, 0.50), surface[0][2], 1e-12);
    EXPECT_GT(std::fabs(surface[0][2] - surface[2][0]), 1e-3)
        << "the transposed nodes must differ for this test to mean anything";
}

// EQDVolatility already passed its axes correctly; keep it pinned so a future
// refactor cannot make the two classes disagree.
TEST(EQDVolatilityTest, reproducesEveryNode) {
    const std::vector<double> strikes{80.0, 100.0, 120.0};
    const std::vector<double> expiries{0.5, 1.0};
    std::vector<std::vector<double>> surface(expiries.size(),
                                             std::vector<double>(strikes.size(), 0.0));
    for (std::size_t i = 0; i < expiries.size(); ++i) {
        for (std::size_t j = 0; j < strikes.size(); ++j) {
            surface[i][j] = 0.20 + 0.01 * expiries[i] * (strikes[j] - 100.0) / 20.0;
        }
    }
    const quantape::markets::EQDVolatility<double> eq(expiries, strikes, surface, 100.0);
    for (std::size_t i = 0; i < expiries.size(); ++i) {
        for (std::size_t j = 0; j < strikes.size(); ++j) {
            SCOPED_TRACE(::testing::Message()
                         << "expiry " << expiries[i] << " strike " << strikes[j]);
            CHECK_CLOSE("eqd surface node", eq.vol(expiries[i], strikes[j]), surface[i][j], 1e-12);
        }
    }
}
