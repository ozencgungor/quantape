/**
 * @file test_markets.cpp
 * @brief Test market data objects
 *
 * Demonstrates usage of IR curves, yield curves, volatility surfaces,
 * and market data objects with both double and AD types.
 */

#include "quantape/markets/MarketData.h"

#include <iomanip>
#include <iostream>
#include <vector>

#include "TestSupport.h"

using namespace quantape::markets;

void testIRCurve() {
    QTA_LOG_INFO("test", "=== IR Curve Test ===");

    // Create IR curve
    std::vector<double> tenors = {0.5, 1.0, 2.0, 5.0, 10.0};
    std::vector<double> rates = {0.01, 0.015, 0.02, 0.025, 0.03};
    IRCurveDescriptor desc("USD", "OIS", "2024-01-01");
    IRCurve<double> curve(tenors, rates, desc);

    QTA_LOG_INFO("test", "Curve: {}", desc.identifier());

    // Test discount factors
    QTA_LOG_INFO("test", "Discount Factors:");
    for (double t : {0.5, 1.0, 2.5, 5.0, 10.0}) {
        double df = curve.discountFactor(t);
        double r = curve.zeroRate(t);
        QTA_LOG_INFO("test", "  t={}: DF={}, r={}", quantape_test::num(t),
                     quantape_test::num(df), quantape_test::num(r));
    }

    // Test forward rate
    double fwd = curve.forwardRate(1.0, 2.0);
    QTA_LOG_INFO("test", "Forward rate (1y1y): {}", quantape_test::num(fwd));
}

void testYieldCurve() {
    QTA_LOG_INFO("test", "=== Yield Curve Test ===");

    // Create dividend yield curve
    std::vector<double> tenors = {0.5, 1.0, 2.0, 5.0};
    std::vector<double> yields = {0.015, 0.018, 0.02, 0.022};
    YieldCurveDescriptor desc("USD", "SPX_DIV", "2024-01-01", "DIVIDEND");
    YieldCurve<double> divCurve(tenors, yields, desc);

    QTA_LOG_INFO("test", "Curve: {}", desc.identifier());

    QTA_LOG_INFO("test", "Dividend Yields:");
    for (double t : {0.5, 1.0, 2.0, 5.0}) {
        double y = divCurve.yield(t);
        double df = divCurve.discountFactor(t);
        QTA_LOG_INFO("test", "  t={}: yield={}, DF={}", quantape_test::num(t),
                     quantape_test::num(y), quantape_test::num(df));
    }
}

void testSurvivalProbabilityCurve() {
    QTA_LOG_INFO("test", "=== Survival Probability Curve Test ===");

    // Create credit curve
    std::vector<double> tenors = {1.0, 2.0, 5.0, 10.0};
    std::vector<double> survProbs = {0.98, 0.95, 0.88, 0.75};
    CreditDescriptor desc("AAPL", "USD", "SENIOR", "2024-01-01");
    SurvivalProbabilityCurve<double> spCurve(tenors, survProbs, desc);

    QTA_LOG_INFO("test", "Credit: {}", desc.identifier());

    QTA_LOG_INFO("test", "Credit Metrics:");
    for (double t : {1.0, 2.0, 5.0, 10.0}) {
        double sp = spCurve.survivalProb(t);
        double pd = spCurve.defaultProb(t);
        double avgHazard = spCurve.avgHazardRate(t);
        QTA_LOG_INFO("test", "  t={}: SP={}, PD={}, avg hazard={}", quantape_test::num(t),
                     quantape_test::num(sp), quantape_test::num(pd), quantape_test::num(avgHazard));
    }
}

void testEQDData() {
    QTA_LOG_INFO("test", "=== Equity Data Test ===");

    // Create curves
    std::vector<double> tenors = {0.5, 1.0, 2.0, 5.0};
    std::vector<double> divYields = {0.015, 0.018, 0.02, 0.022};
    std::vector<double> rates = {0.01, 0.015, 0.02, 0.025};

    YieldCurveDescriptor divDesc("USD", "SPX_DIV", "2024-01-01", "DIVIDEND");
    IRCurveDescriptor rateDesc("USD", "OIS", "2024-01-01");

    YieldCurve<double> divCurve(tenors, divYields, divDesc);
    IRCurve<double> rateCurve(tenors, rates, rateDesc);

    // Create equity data
    double spot = 4500.0;
    EQDDescriptor eqdDesc("SPX", "INDEX", "USD");
    EQDData<double> spxData(spot, divCurve, rateCurve, eqdDesc);

    QTA_LOG_INFO("test", "Equity: {}", eqdDesc.identifier());
    QTA_LOG_INFO("test", "Spot: {}", quantape_test::num(spxData.spot()));

    QTA_LOG_INFO("test", "Forward Prices:");
    for (double t : {0.5, 1.0, 2.0, 5.0}) {
        double fwd = spxData.forward(t);
        double df = spxData.discountFactor(t);
        QTA_LOG_INFO("test", "  t={}: Forward={}, DF={}", quantape_test::num(t),
                     quantape_test::num(fwd), quantape_test::num(df));
    }
}

void testVolatilitySurfaces() {
    QTA_LOG_INFO("test", "=== Volatility Surfaces Test ===");

    try {
        // Create equity vol surface
        std::vector<double> expiries = {0.25, 0.5, 1.0, 2.0};
        std::vector<double> strikes = {4000, 4250, 4500, 4750, 5000};

        // Simple vol surface (ATM around 0.20, smile)
        std::vector<std::vector<double>> vols = {
            {0.25, 0.22, 0.20, 0.21, 0.23}, // 0.25y
            {0.24, 0.21, 0.19, 0.20, 0.22}, // 0.5y
            {0.23, 0.20, 0.18, 0.19, 0.21}, // 1y
            {0.22, 0.19, 0.17, 0.18, 0.20}  // 2y
        };

        QTA_LOG_INFO("test", "Creating vol surface with {} expiries and {} strikes",
                     expiries.size(), strikes.size());
        QTA_LOG_INFO("test", "Vol matrix size: {} x {}", vols.size(), vols[0].size());

        EQDDescriptor desc("SPX", "INDEX", "USD");
        EQDVolatility<double> volSurf(expiries, strikes, vols, 4500.0, desc);

        QTA_LOG_INFO("test", "EQD Vol Surface: {}", desc.identifier());
        QTA_LOG_INFO("test", "Ref Spot: {}", quantape_test::num(volSurf.referenceSpot()));

        QTA_LOG_INFO("test", "Sample Volatilities:");
        QTA_LOG_INFO("test", "  vol(0.5y, 4500) = {}",
                     quantape_test::num(volSurf.vol(0.5, 4500.0)));
        QTA_LOG_INFO("test", "  vol(1.0y, 4250) = {}",
                     quantape_test::num(volSurf.vol(1.0, 4250.0)));
        QTA_LOG_INFO("test", "  vol(1.0y, 4750) = {}",
                     quantape_test::num(volSurf.vol(1.0, 4750.0)));
    } catch (const std::exception& e) {
        QTA_LOG_ERROR("test", "Vol surface error: {}", e.what());
    }
}

void testFXRate() {
    QTA_LOG_INFO("test", "=== FX Rate Test ===");

    // Create FX rate with curves
    std::vector<double> tenors = {0.5, 1.0, 2.0, 5.0};
    std::vector<double> usdRates = {0.01, 0.015, 0.02, 0.025};
    std::vector<double> eurRates = {0.005, 0.008, 0.012, 0.015};

    IRCurveDescriptor usdDesc("USD", "OIS", "2024-01-01");
    IRCurveDescriptor eurDesc("EUR", "ESTR", "2024-01-01");

    IRCurve<double> usdCurve(tenors, usdRates, usdDesc);
    IRCurve<double> eurCurve(tenors, eurRates, eurDesc);

    double spot = 1.10; // USDEUR spot
    FXDescriptor fxDesc("USD", "EUR", "2024-01-01");
    FXRate<double> fx(spot, usdCurve, eurCurve, fxDesc);

    QTA_LOG_INFO("test", "FX Pair: {}", fxDesc.pair());
    QTA_LOG_INFO("test", "Spot: {}", quantape_test::num(fx.spot()));

    QTA_LOG_INFO("test", "Forward Rates (covered IRP):");
    for (double t : {0.5, 1.0, 2.0, 5.0}) {
        double fwd = fx.forward(t);
        QTA_LOG_INFO("test", "  t={}: Forward={}", quantape_test::num(t),
                     quantape_test::num(fwd));
    }
}

void testIRVolTypes() {
    QTA_LOG_INFO("test", "=== IR Volatility Types Test ===");

    // Create a swaption volatility surface
    std::vector<double> expiries = {1.0, 2.0, 5.0};
    std::vector<double> tenors = {1.0, 5.0, 10.0};
    std::vector<std::vector<double>> swaptionVols = {
        {0.50, 0.45, 0.42}, // 1y expiry
        {0.48, 0.43, 0.40}, // 2y expiry
        {0.45, 0.40, 0.38}  // 5y expiry
    };

    IRVolDescriptor swaptionDesc("USD", "SWAPTION", "", "SOFR");
    SwaptionVolatility<double> swaptionSurf(expiries, tenors, swaptionVols, swaptionDesc);

    QTA_LOG_INFO("test", "Swaption Vol Surface: {}", swaptionDesc.identifier());
    QTA_LOG_INFO("test", "  Vol type: {}",
                 (swaptionSurf.volType() == IRVolType::Swaption ? "Swaption" : "Cap"));
    QTA_LOG_INFO("test", "  ATM vol (2y expiry, 5y tenor): {}",
                 quantape_test::num(swaptionSurf.atmVol(2.0, 5.0)));

    // Create a cap volatility surface
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

    QTA_LOG_INFO("test", "Cap Vol Surface: {}", capDesc.identifier());
    QTA_LOG_INFO("test", "  Vol type: {}",
                 (capSurf.volType() == IRVolType::Cap ? "Cap" : "Swaption"));
    QTA_LOG_INFO("test", "  ATM vol (2y expiry, 3M tenor): {}",
                 quantape_test::num(capSurf.atmVol(2.0, 0.25)));
}

void testVolSurfaceOperations() {
    QTA_LOG_INFO("test", "=== Volatility Surface Element-Wise Operations Test ===");

    // Create a simple equity vol surface
    std::vector<double> expiries = {0.5, 1.0, 2.0};
    std::vector<double> strikes = {90, 100, 110};
    std::vector<std::vector<double>> vols = {
        {0.25, 0.20, 0.22}, // 0.5y
        {0.23, 0.18, 0.20}, // 1y
        {0.21, 0.16, 0.18}  // 2y
    };

    EQDDescriptor desc("TEST", "EQUITY", "USD");
    EQDVolatility<double> volSurf(expiries, strikes, vols, 100.0, desc);

    QTA_LOG_INFO("test", "Original ATM vol (1y, 100): {}",
                 quantape_test::num(volSurf.vol(1.0, 100.0)));

    // Test 1: Scale by 1.1 (10% vol increase)
    QTA_LOG_INFO("test", "Test 1: Scale by 1.1");
    EQDVolatility<double> volSurf1 = volSurf;
    volSurf1.scale(1.1);
    QTA_LOG_INFO("test", "  After scaling: {}",
                 quantape_test::num(volSurf1.vol(1.0, 100.0)));
    QTA_LOG_INFO("test", "  Expected: {}", quantape_test::num(0.18 * 1.1));

    // Test 2: Shift by +0.01 (100 bp vol increase)
    QTA_LOG_INFO("test", "Test 2: Shift by +0.01");
    EQDVolatility<double> volSurf2 = volSurf;
    volSurf2.shift(0.01);
    QTA_LOG_INFO("test", "  After shifting: {}",
                 quantape_test::num(volSurf2.vol(1.0, 100.0)));
    QTA_LOG_INFO("test", "  Expected: {}", quantape_test::num(0.18 + 0.01));

    // Test 3: Apply function (square each vol)
    QTA_LOG_INFO("test", "Test 3: Apply function (square root)");
    EQDVolatility<double> volSurf3 = volSurf;
    volSurf3.applyFunction([](double v) { return std::sqrt(v); });
    QTA_LOG_INFO("test", "  After sqrt: {}", quantape_test::num(volSurf3.vol(1.0, 100.0)));
    QTA_LOG_INFO("test", "  Expected: {}", quantape_test::num(std::sqrt(0.18)));

    // Test 4: Apply function with coordinates (vol smile adjustment)
    QTA_LOG_INFO("test", "Test 4: Apply function with coordinates (moneyness adjustment)");
    EQDVolatility<double> volSurf4 = volSurf;
    volSurf4.applyFunctionWithCoords([](double v, double expiry, double strike) {
        // Add smile: increase vol for out-of-money options
        double moneyness = strike / 100.0;                  // spot = 100
        double smileAdj = 0.01 * std::abs(moneyness - 1.0); // OTM adjustment
        return v + smileAdj;
    });
    QTA_LOG_INFO("test", "  ATM (100): {}", quantape_test::num(volSurf4.vol(1.0, 100.0)));
    QTA_LOG_INFO("test", "  OTM Put (90): {}",
                 quantape_test::num(volSurf4.vol(1.0, 90.0)));
    QTA_LOG_INFO("test", "  OTM Call (110): {}",
                 quantape_test::num(volSurf4.vol(1.0, 110.0)));

    // Test 5: Bump specific point
    QTA_LOG_INFO("test", "Test 5: Bump specific point (1y, 100 strike)");
    EQDVolatility<double> volSurf5 = volSurf;
    volSurf5.bump(1, 1, 0.05); // expiry index 1 (1y), strike index 1 (100)
    QTA_LOG_INFO("test", "  After bumping by 0.05: {}",
                 quantape_test::num(volSurf5.vol(1.0, 100.0)));
    QTA_LOG_INFO("test", "  Expected: {}", quantape_test::num(0.18 + 0.05));
    QTA_LOG_INFO("test", "  Nearby point (1y, 90): {} (should be similar to original)",
                 quantape_test::num(volSurf5.vol(1.0, 90.0)));

    // Test 6: Operator overloads
    QTA_LOG_INFO("test", "Test 6: Operator overloads");
    EQDVolatility<double> volSurf6 = volSurf;
    volSurf6 *= 1.2;   // Scale by 1.2
    volSurf6 += 0.005; // Shift by 50 bp
    QTA_LOG_INFO("test", "  After *= 1.2 and += 0.005: {}",
                 quantape_test::num(volSurf6.vol(1.0, 100.0)));
    QTA_LOG_INFO("test", "  Expected: {}", quantape_test::num(0.18 * 1.2 + 0.005));
}

int main() {
    try {
        testIRCurve();
        testYieldCurve();
        testSurvivalProbabilityCurve();
        testEQDData();
        testVolatilitySurfaces();
        testFXRate();
        testIRVolTypes();
        testVolSurfaceOperations();

        QTA_LOG_INFO("test", "========================================");
        QTA_LOG_INFO("test", "All market data tests completed successfully!");
        QTA_LOG_INFO("test", "========================================");

        return 0;
    } catch (const std::exception& e) {
        QTA_LOG_ERROR("test", "Error: {}", e.what());
        return 1;
    }
}
