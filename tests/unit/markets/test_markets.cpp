/**
 * @file test_markets.cpp
 * @brief Test market data objects
 *
 * Demonstrates usage of IR curves, yield curves, volatility surfaces,
 * and market data objects with both double and AD types.
 */

#include "quantape/math/StanMath.h"

#include "quantape/markets/MarketData.h"
#include "quantape/math/Autodiff/PrimalExtraction.h"

#include <cmath>
#include <utility>
#include <vector>

#include "support/GtestSupport.h"
#include "support/StanTapeFixture.h"

using namespace quantape::markets;

namespace {

class MarketsTest : public StanTapeTest {};

} // namespace

TEST_F(MarketsTest, irCurveDiscountAndForward) {
    std::vector<double> tenors = {0.5, 1.0, 2.0, 5.0, 10.0};
    std::vector<double> rates = {0.01, 0.015, 0.02, 0.025, 0.03};
    IRCurveDescriptor desc("USD", "OIS", "2024-01-01");
    DiscountCurve<double> curve(tenors, rates);
    EXPECT_TRUE(desc.identifier() == "USD.OIS.2024-01-01");

    // Discount factors and zero rates are finite and positive on the grid.
    for (double t : {0.5, 1.0, 2.5, 5.0, 10.0}) {
        double df = curve.discountFactor(t);
        double r = curve.zeroRate(t);
        EXPECT_TRUE(std::isfinite(df));
        EXPECT_TRUE(df > 0.0);
        EXPECT_TRUE(std::isfinite(r));
    }

    double fwd = curve.forwardRate(1.0, 2.0);
    EXPECT_TRUE(std::isfinite(fwd));
}

TEST_F(MarketsTest, yieldCurveZeroSpace) {
    std::vector<double> tenors = {0.5, 1.0, 2.0, 5.0};
    std::vector<double> yields = {0.015, 0.018, 0.02, 0.022};
    YieldCurveDescriptor desc("USD", "SPX_DIV", "2024-01-01", "DIVIDEND");
    DiscountCurve<double> divCurve(tenors, yields, InterpolationSpace::Zero,
                                   InterpolationScheme::Linear);
    EXPECT_TRUE(desc.identifier() == "USD.SPX_DIV.2024-01-01");

    for (std::size_t i = 0; i < tenors.size(); ++i) {
        CHECK_CLOSE("yield node", divCurve.yield(tenors[i]), yields[i], 1e-15);
        CHECK_CLOSE("yield discount", divCurve.discountFactor(tenors[i]),
                    std::exp(-yields[i] * tenors[i]), 1e-15);
    }
}

TEST_F(MarketsTest, survivalProbabilityFvarVarValue) {
    std::vector<double> tenors = {1.0, 2.0, 5.0, 10.0};
    std::vector<double> survProbs = {0.98, 0.95, 0.88, 0.75};
    CreditDescriptor desc("AAPL", "USD", "SENIOR", "2024-01-01");
    SurvivalProbabilityCurve<double> spCurve(tenors, survProbs, desc);

    // AD compile/value check: the curve instantiates and evaluates for the
    // nested `fvar<var>` scalar.
    using FvarVar = stan::math::fvar<stan::math::var>;
    SurvivalProbabilityCurve<FvarVar> adCurve(tenors, survProbs, desc);
    const FvarVar adSp = adCurve.survivalProb(FvarVar(2.0));
    CHECK_CLOSE("fvar<var> survival prob", quantape::math::detail::primalValue(adSp),
                spCurve.survivalProb(2.0), 1e-12);
}

TEST_F(MarketsTest, survivalProbabilityMovedCurve) {
    std::vector<double> tenors = {1.0, 2.0, 5.0, 10.0};
    std::vector<double> survProbs = {0.98, 0.95, 0.88, 0.75};
    CreditDescriptor desc("AAPL", "USD", "SENIOR", "2024-01-01");
    SurvivalProbabilityCurve<double> spCurve(tenors, survProbs, desc);

    // The `fvar<var>` curve evaluates identically after a move.
    using FvarVar = stan::math::fvar<stan::math::var>;
    SurvivalProbabilityCurve<FvarVar> adCurve(tenors, survProbs, desc);
    SurvivalProbabilityCurve<FvarVar> movedCurve(std::move(adCurve));
    const FvarVar movedSp = movedCurve.survivalProb(FvarVar(2.0));
    CHECK_CLOSE("moved fvar<var> survival prob", quantape::math::detail::primalValue(movedSp),
                spCurve.survivalProb(2.0), 1e-12);
}

TEST_F(MarketsTest, eqdForwardCip) {
    std::vector<double> tenors = {0.5, 1.0, 2.0, 5.0};
    std::vector<double> divYields = {0.015, 0.018, 0.02, 0.022};
    std::vector<double> rates = {0.01, 0.015, 0.02, 0.025};

    YieldCurveDescriptor divDesc("USD", "SPX_DIV", "2024-01-01", "DIVIDEND");
    IRCurveDescriptor rateDesc("USD", "OIS", "2024-01-01");

    DiscountCurve<double> divCurve(tenors, divYields, InterpolationSpace::Zero,
                                   InterpolationScheme::Linear);
    DiscountCurve<double> rateCurve(tenors, rates);

    double spot = 4500.0;
    EQDDescriptor eqdDesc("SPX", "INDEX", "USD");
    EQDData<double> spxData(spot, divCurve, rateCurve, eqdDesc);
    EXPECT_TRUE(eqdDesc.identifier() == "SPX.INDEX.USD");

    for (double t : {0.5, 1.0, 2.0, 5.0}) {
        CHECK_CLOSE("eqd forward", spxData.forward(t),
                    spot * rateCurve.discountFactor(t) / divCurve.discountFactor(t), 1e-15);
        CHECK_CLOSE("eqd discount", spxData.discountFactor(t), rateCurve.discountFactor(t), 1e-15);
    }
}

TEST_F(MarketsTest, eqdVolatilitySurfaceSamples) {
    std::vector<double> expiries = {0.25, 0.5, 1.0, 2.0};
    std::vector<double> strikes = {4000, 4250, 4500, 4750, 5000};
    std::vector<std::vector<double>> vols = {
        {0.25, 0.22, 0.20, 0.21, 0.23}, // 0.25y
        {0.24, 0.21, 0.19, 0.20, 0.22}, // 0.5y
        {0.23, 0.20, 0.18, 0.19, 0.21}, // 1y
        {0.22, 0.19, 0.17, 0.18, 0.20}  // 2y
    };

    EQDDescriptor desc("SPX", "INDEX", "USD");
    EQDVolatility<double> volSurf(expiries, strikes, vols, 4500.0, desc);

    EXPECT_TRUE(volSurf.referenceSpot() == 4500.0);
    CHECK_CLOSE("eqd vol 0.5y 4500", volSurf.vol(0.5, 4500.0), 0.19, 1e-12);
    CHECK_CLOSE("eqd vol 1y 4250", volSurf.vol(1.0, 4250.0), 0.20, 1e-12);
    CHECK_CLOSE("eqd vol 1y 4750", volSurf.vol(1.0, 4750.0), 0.19, 1e-12);
}

TEST_F(MarketsTest, fxDescriptorAndCipForward) {
    std::vector<double> tenors = {0.5, 1.0, 2.0, 5.0};
    std::vector<double> usdRates = {0.01, 0.015, 0.02, 0.025};
    std::vector<double> eurRates = {0.005, 0.008, 0.012, 0.015};

    IRCurveDescriptor usdDesc("USD", "OIS", "2024-01-01");
    IRCurveDescriptor eurDesc("EUR", "ESTR", "2024-01-01");

    DiscountCurve<double> usdCurve(tenors, usdRates);
    DiscountCurve<double> eurCurve(tenors, eurRates);

    double spot = 1.10; // EURUSD spot: USD per 1 EUR
    FXDescriptor fxDesc("EUR", "USD", "2024-01-01");
    FXRate<double> fx(spot, eurCurve, usdCurve, fxDesc);

    EXPECT_TRUE(fxDesc.pair() == "EURUSD");
    EXPECT_TRUE(fxDesc.baseCcy == "EUR");
    EXPECT_TRUE(fxDesc.quoteCcy == "USD");
    EXPECT_TRUE(fxDesc.hasKnownCurrencies());

    // CIP: F(1y) = S * DF_base / DF_quote = 1.10 * exp(-0.008 + 0.015).
    CHECK_CLOSE("EURUSD 1y forward", fx.forward(1.0), spot * std::exp(-0.008 + 0.015), 1e-12);
}

TEST_F(MarketsTest, irVolTypes) {
    std::vector<double> expiries = {1.0, 2.0, 5.0};
    std::vector<double> tenors = {1.0, 5.0, 10.0};
    std::vector<std::vector<double>> swaptionVols = {
        {0.50, 0.45, 0.42}, // 1y expiry
        {0.48, 0.43, 0.40}, // 2y expiry
        {0.45, 0.40, 0.38}  // 5y expiry
    };

    IRVolDescriptor swaptionDesc("USD", "SWAPTION", "", "SOFR");
    SwaptionVolatility<double> swaptionSurf(expiries, tenors, swaptionVols, swaptionDesc);

    EXPECT_TRUE(swaptionSurf.volType() == IRVolType::Swaption);
    CHECK_CLOSE("swaption atm vol", swaptionSurf.atmVol(2.0, 5.0), 0.43, 1e-12);

    std::vector<double> capExpiries = {1.0, 2.0, 5.0, 10.0};
    std::vector<double> capTenors = {0.25, 0.5}; // 3M and 6M forward rates
    std::vector<std::vector<double>> capVols = {
        {0.35, 0.34}, // 1y
        {0.33, 0.32}, // 2y
        {0.30, 0.29}, // 5y
        {0.28, 0.27}  // 10y
    };

    IRVolDescriptor capDesc("USD", "CAPFLOOR", "", "SOFR");
    CapVolatility<double> capSurf(capExpiries, capTenors, capVols, capDesc);

    EXPECT_TRUE(capSurf.volType() == IRVolType::Cap);
    CHECK_CLOSE("cap atm vol", capSurf.atmVol(2.0, 0.25), 0.33, 1e-12);
}

TEST_F(MarketsTest, volSurfaceOperationsMatchClosedForm) {
    std::vector<double> expiries = {0.5, 1.0, 2.0};
    std::vector<double> strikes = {90, 100, 110};
    std::vector<std::vector<double>> vols = {
        {0.25, 0.20, 0.22}, // 0.5y
        {0.23, 0.18, 0.20}, // 1y
        {0.21, 0.16, 0.18}  // 2y
    };

    EQDDescriptor desc("TEST", "EQUITY", "USD");
    EQDVolatility<double> volSurf(expiries, strikes, vols, 100.0, desc);

    // Scale by 1.1 (10% vol increase).
    EQDVolatility<double> volSurf1 = volSurf;
    volSurf1.scale(1.1);
    CHECK_CLOSE("scale 1.1", volSurf1.vol(1.0, 100.0), 0.18 * 1.1, 1e-12);

    // Shift by +0.01 (100 bp vol increase).
    EQDVolatility<double> volSurf2 = volSurf;
    volSurf2.shift(0.01);
    CHECK_CLOSE("shift by 0.01", volSurf2.vol(1.0, 100.0), 0.18 + 0.01, 1e-12);

    // Apply function (square root of each vol).
    EQDVolatility<double> volSurf3 = volSurf;
    volSurf3.applyFunction([](double v) { return std::sqrt(v); });
    CHECK_CLOSE("apply sqrt", volSurf3.vol(1.0, 100.0), std::sqrt(0.18), 1e-12);

    // Apply function with coordinates (moneyness smile adjustment).
    EQDVolatility<double> volSurf4 = volSurf;
    volSurf4.applyFunctionWithCoords([](double v, double expiry, double strike) {
        // Add smile: increase vol for out-of-money options
        double moneyness = strike / 100.0;                  // spot = 100
        double smileAdj = 0.01 * std::abs(moneyness - 1.0); // OTM adjustment
        return v + smileAdj;
    });
    CHECK_CLOSE("smile atm", volSurf4.vol(1.0, 100.0), 0.18, 1e-12);
    CHECK_CLOSE("smile otm put", volSurf4.vol(1.0, 90.0), 0.23 + 0.01 * 0.1, 1e-12);
    CHECK_CLOSE("smile otm call", volSurf4.vol(1.0, 110.0), 0.20 + 0.01 * 0.1, 1e-12);

    // Bump specific point (1y, 100 strike) by 0.05.
    EQDVolatility<double> volSurf5 = volSurf;
    volSurf5.bump(1, 1, 0.05); // expiry index 1 (1y), strike index 1 (100)
    CHECK_CLOSE("bump point", volSurf5.vol(1.0, 100.0), 0.18 + 0.05, 1e-12);

    // Operator overloads: *= 1.2 then += 0.005.
    EQDVolatility<double> volSurf6 = volSurf;
    volSurf6 *= 1.2;
    volSurf6 += 0.005;
    CHECK_CLOSE("operators", volSurf6.vol(1.0, 100.0), 0.18 * 1.2 + 0.005, 1e-12);
}
