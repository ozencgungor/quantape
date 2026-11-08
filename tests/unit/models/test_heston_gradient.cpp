// test_heston_gradient.cpp — full-point complex-step gradients on the price leg.
//
// The strike and spot components use relative imaginary steps: both enter
// mu = log(S/K) + ..., so an absolute step comparable to a small strike makes
// Im(mu) O(1) and the Lewis integrand overflows on a wide truncation range.

#include "quantape/models/HestonModel.h"

#include <cmath>

#include "support/GtestSupport.h"

using namespace quantape::models;

TEST(HestonGradientTest, strikeGradientIsFiniteAndMatchesFiniteDifference) {
    const HestonModel model; // EFGL default, Auto control variate
    const HestonParams params;
    const double tMax = 1.0;
    for (const double strike : {1.0, 1e-2, 1e-6, 1e-8}) {
        SCOPED_TRACE(::testing::Message() << "strike=" << strike);
        const HestonFullPoint point{params.v0, params.kappa, params.theta, params.sigma, params.rho,
                                    1.0,       strike,       0.0,          0.0};
        const auto gradient = model.fullGradient(point, tMax);
        ASSERT_TRUE(std::isfinite(gradient(0, HESTON_STRIKE)));

        // Log-strike central difference: dC/dK = (C(K e^h) - C(K e^-h)) / (2 h K).
        const double h = 1e-4;
        HestonMarket up;
        up.strike = strike * std::exp(h);
        up.tMax = tMax;
        HestonMarket down;
        down.strike = strike * std::exp(-h);
        down.tMax = tMax;
        const double fd = (model.call(params, up) - model.call(params, down)) / (2.0 * h * strike);
        CHECK_CLOSE_REL("strike complex-step gradient vs log-strike FD", gradient(0, HESTON_STRIKE),
                        fd, 1e-3);
    }
}

// A wide Gauss-Legendre truncation range exercises the integrand damping with
// u up to ~1000: the strike step must stay relative so Im(mu) remains tiny
// and the gradient finite at a strike far below the absolute step size.
TEST(HestonGradientTest, tinyStrikeStaysFiniteOnWideTruncation) {
    const HestonModel model(HestonConfig{1024, 1000.0, HestonControlVariate::CfMatched,
                                         HestonQuadrature::GaussLegendre});
    const HestonParams params;
    const HestonFullPoint point{params.v0, params.kappa, params.theta, params.sigma, params.rho,
                                1.0,       1e-8,         0.0,          0.0};
    const auto gradient = model.fullGradient(point, 1.0);
    EXPECT_TRUE(std::isfinite(gradient(0, HESTON_STRIKE)));
    EXPECT_LT(std::fabs(gradient(0, HESTON_STRIKE)), 1e3);
}
