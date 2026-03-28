/**
 * @file test_markets.cpp
 * @brief Test market data objects
 *
 * Demonstrates usage of IR curves, yield curves, volatility surfaces,
 * and market data objects with both double and AD types.
 */

#include <iomanip>
#include <iostream>
#include <vector>

#include "Markets/MarketData.h"

using namespace Markets;

void testIRCurve() {
    std::cout << "=== IR Curve Test ===\n\n";

    // Create IR curve
    std::vector<double> tenors = {0.5, 1.0, 2.0, 5.0, 10.0};
    std::vector<double> rates = {0.01, 0.015, 0.02, 0.025, 0.03};
    IRCurveDescriptor desc("USD", "OIS", "2024-01-01");
    IRCurve<double> curve(tenors, rates, desc);

    std::cout << "Curve: " << desc.identifier() << "\n\n";

    // Test discount factors
    std::cout << "Discount Factors:\n";
    for (double t : {0.5, 1.0, 2.5, 5.0, 10.0}) {
        double df = curve.discountFactor(t);
        double r = curve.zeroRate(t);
        std::cout << "  t=" << t << ": DF=" << df << ", r=" << r << "\n";
    }

    // Test forward rate
    double fwd = curve.forwardRate(1.0, 2.0);
    std::cout << "\nForward rate (1y1y): " << fwd << "\n\n";
}

void testYieldCurve() {
    std::cout << "=== Yield Curve Test ===\n\n";

    // Create dividend yield curve
    std::vector<double> tenors = {0.5, 1.0, 2.0, 5.0};
    std::vector<double> yields = {0.015, 0.018, 0.02, 0.022};
    YieldCurveDescriptor desc("USD", "SPX_DIV", "2024-01-01", "DIVIDEND");
    YieldCurve<double> divCurve(tenors, yields, desc);

    std::cout << "Curve: " << desc.identifier() << "\n\n";

    std::cout << "Dividend Yields:\n";
    for (double t : {0.5, 1.0, 2.0, 5.0}) {
        double y = divCurve.yield(t);
        double df = divCurve.discountFactor(t);
        std::cout << "  t=" << t << ": yield=" << y << ", DF=" << df << "\n";
    }
    std::cout << "\n";
}

void testSurvivalProbabilityCurve() {
    std::cout << "=== Survival Probability Curve Test ===\n\n";

    // Create credit curve
    std::vector<double> tenors = {1.0, 2.0, 5.0, 10.0};
    std::vector<double> survProbs = {0.98, 0.95, 0.88, 0.75};
    CreditDescriptor desc("AAPL", "USD", "SENIOR", "2024-01-01");
    SurvivalProbabilityCurve<double> spCurve(tenors, survProbs, desc);

    std::cout << "Credit: " << desc.identifier() << "\n\n";

    std::cout << "Credit Metrics:\n";
    for (double t : {1.0, 2.0, 5.0, 10.0}) {
        double sp = spCurve.survivalProb(t);
        double pd = spCurve.defaultProb(t);
        double avgHazard = spCurve.avgHazardRate(t);
        std::cout << "  t=" << t << ": SP=" << sp << ", PD=" << pd << ", avg hazard=" << avgHazard
                  << "\n";
    }
    std::cout << "\n";
}

void testEQDData() {
    std::cout << "=== Equity Data Test ===\n\n";

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

    std::cout << "Equity: " << eqdDesc.identifier() << "\n";
    std::cout << "Spot: " << spxData.spot() << "\n\n";

    std::cout << "Forward Prices:\n";
    for (double t : {0.5, 1.0, 2.0, 5.0}) {
        double fwd = spxData.forward(t);
        double df = spxData.discountFactor(t);
        std::cout << "  t=" << t << ": Forward=" << fwd << ", DF=" << df << "\n";
    }
    std::cout << "\n";
}

void testVolatilitySurfaces() {
    std::cout << "=== Volatility Surfaces Test ===\n\n";

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

        std::cout << "Creating vol surface with " << expiries.size() << " expiries and "
                  << strikes.size() << " strikes\n";
        std::cout << "Vol matrix size: " << vols.size() << " x " << vols[0].size() << "\n\n";

        EQDDescriptor desc("SPX", "INDEX", "USD");
        EQDVolatility<double> volSurf(expiries, strikes, vols, 4500.0, desc);

        std::cout << "EQD Vol Surface: " << desc.identifier() << "\n";
        std::cout << "Ref Spot: " << volSurf.referenceSpot() << "\n\n";

        std::cout << "Sample Volatilities:\n";
        std::cout << "  vol(0.5y, 4500) = " << volSurf.vol(0.5, 4500.0) << "\n";
        std::cout << "  vol(1.0y, 4250) = " << volSurf.vol(1.0, 4250.0) << "\n";
        std::cout << "  vol(1.0y, 4750) = " << volSurf.vol(1.0, 4750.0) << "\n";
        std::cout << "\n";
    } catch (const std::exception& e) {
        std::cerr << "Vol surface error: " << e.what() << "\n\n";
    }
}

void testFXRate() {
    std::cout << "=== FX Rate Test ===\n\n";

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

    std::cout << "FX Pair: " << fxDesc.pair() << "\n";
    std::cout << "Spot: " << fx.spot() << "\n\n";

    std::cout << "Forward Rates (covered IRP):\n";
    for (double t : {0.5, 1.0, 2.0, 5.0}) {
        double fwd = fx.forward(t);
        std::cout << "  t=" << t << ": Forward=" << fwd << "\n";
    }
    std::cout << "\n";
}

void testIRVolTypes() {
    std::cout << "=== IR Volatility Types Test ===\n\n";

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

    std::cout << "Swaption Vol Surface: " << swaptionDesc.identifier() << "\n";
    std::cout << "  Vol type: "
              << (swaptionSurf.volType() == IRVolType::Swaption ? "Swaption" : "Cap") << "\n";
    std::cout << "  ATM vol (2y expiry, 5y tenor): " << swaptionSurf.atmVol(2.0, 5.0) << "\n\n";

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

    std::cout << "Cap Vol Surface: " << capDesc.identifier() << "\n";
    std::cout << "  Vol type: " << (capSurf.volType() == IRVolType::Cap ? "Cap" : "Swaption")
              << "\n";
    std::cout << "  ATM vol (2y expiry, 3M tenor): " << capSurf.atmVol(2.0, 0.25) << "\n\n";
}

void testVolSurfaceOperations() {
    std::cout << "=== Volatility Surface Element-Wise Operations Test ===\n\n";

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

    std::cout << "Original ATM vol (1y, 100): " << volSurf.vol(1.0, 100.0) << "\n\n";

    // Test 1: Scale by 1.1 (10% vol increase)
    std::cout << "Test 1: Scale by 1.1\n";
    EQDVolatility<double> volSurf1 = volSurf;
    volSurf1.scale(1.1);
    std::cout << "  After scaling: " << volSurf1.vol(1.0, 100.0) << "\n";
    std::cout << "  Expected: " << 0.18 * 1.1 << "\n\n";

    // Test 2: Shift by +0.01 (100 bp vol increase)
    std::cout << "Test 2: Shift by +0.01\n";
    EQDVolatility<double> volSurf2 = volSurf;
    volSurf2.shift(0.01);
    std::cout << "  After shifting: " << volSurf2.vol(1.0, 100.0) << "\n";
    std::cout << "  Expected: " << 0.18 + 0.01 << "\n\n";

    // Test 3: Apply function (square each vol)
    std::cout << "Test 3: Apply function (square root)\n";
    EQDVolatility<double> volSurf3 = volSurf;
    volSurf3.applyFunction([](double v) { return std::sqrt(v); });
    std::cout << "  After sqrt: " << volSurf3.vol(1.0, 100.0) << "\n";
    std::cout << "  Expected: " << std::sqrt(0.18) << "\n\n";

    // Test 4: Apply function with coordinates (vol smile adjustment)
    std::cout << "Test 4: Apply function with coordinates (moneyness adjustment)\n";
    EQDVolatility<double> volSurf4 = volSurf;
    volSurf4.applyFunctionWithCoords([](double v, double expiry, double strike) {
        // Add smile: increase vol for out-of-money options
        double moneyness = strike / 100.0;                  // spot = 100
        double smileAdj = 0.01 * std::abs(moneyness - 1.0); // OTM adjustment
        return v + smileAdj;
    });
    std::cout << "  ATM (100): " << volSurf4.vol(1.0, 100.0) << "\n";
    std::cout << "  OTM Put (90): " << volSurf4.vol(1.0, 90.0) << "\n";
    std::cout << "  OTM Call (110): " << volSurf4.vol(1.0, 110.0) << "\n\n";

    // Test 5: Bump specific point
    std::cout << "Test 5: Bump specific point (1y, 100 strike)\n";
    EQDVolatility<double> volSurf5 = volSurf;
    volSurf5.bump(1, 1, 0.05); // expiry index 1 (1y), strike index 1 (100)
    std::cout << "  After bumping by 0.05: " << volSurf5.vol(1.0, 100.0) << "\n";
    std::cout << "  Expected: " << 0.18 + 0.05 << "\n";
    std::cout << "  Nearby point (1y, 90): " << volSurf5.vol(1.0, 90.0)
              << " (should be similar to original)\n\n";

    // Test 6: Operator overloads
    std::cout << "Test 6: Operator overloads\n";
    EQDVolatility<double> volSurf6 = volSurf;
    volSurf6 *= 1.2;   // Scale by 1.2
    volSurf6 += 0.005; // Shift by 50 bp
    std::cout << "  After *= 1.2 and += 0.005: " << volSurf6.vol(1.0, 100.0) << "\n";
    std::cout << "  Expected: " << 0.18 * 1.2 + 0.005 << "\n\n";
}

int main() {
    std::cout << std::setprecision(6);
    std::cout << std::fixed;

    try {
        testIRCurve();
        testYieldCurve();
        testSurvivalProbabilityCurve();
        testEQDData();
        testVolatilitySurfaces();
        testFXRate();
        testIRVolTypes();
        testVolSurfaceOperations();

        std::cout << "========================================\n";
        std::cout << "All market data tests completed successfully!\n";
        std::cout << "========================================\n";

        return 0;
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }
}
