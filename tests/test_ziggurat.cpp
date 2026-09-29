/**
 * @file test_ziggurat.cpp
 * @brief Tests for the Ziggurat normal RNG: table verification, statistical tests, benchmarks
 */

#include "quantape/math/Random/ZigguratNormal.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <random>
#include <vector>

#include "TestSupport.h"

// ═══════════════════════════════════════════════════════════════════════════
// HELPERS
// ═══════════════════════════════════════════════════════════════════════════

template <typename F>
double timeNs(F&& fn, int reps = 1'000'000) {
    // Warmup
    volatile double sink = 0;
    for (int i = 0; i < 1000; ++i)
        sink += fn();

    auto t0 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < reps; ++i)
        sink += fn();
    auto t1 = std::chrono::high_resolution_clock::now();

    return std::chrono::duration<double, std::nano>(t1 - t0).count() / reps;
}

// ═══════════════════════════════════════════════════════════════════════════
// TABLE VERIFICATION
// ═══════════════════════════════════════════════════════════════════════════

void testTableGeneration() {
    QTA_LOG_INFO("test", "\n═══════════════════════════════════════════════════════════\n"
                                  " ZIGGURAT TABLE VERIFICATION\n"
                                  "═══════════════════════════════════════════════════════════\n");

    auto tab = quantape::math::mc::generateZigguratTables();
    auto v = quantape::math::mc::verifyZigguratTables(tab);

    QTA_LOG_INFO("test", "  r (tail cutoff)      = {}", quantape_test::num(tab.r));
    QTA_LOG_INFO("test", "  v (layer area)       = {}", quantape_test::num(tab.A));
    QTA_LOG_INFO("test", "  base width v/y[0]    = {}",
                 quantape_test::num(tab.A / tab.y[0]));
    QTA_LOG_INFO("test", "  x[0] = r             = {}", quantape_test::num(tab.x[0]));
    QTA_LOG_INFO("test", "  x[1]                 = {}", quantape_test::num(tab.x[1]));
    QTA_LOG_INFO("test", "  x[N-2] = x[254]      = {}", quantape_test::num(tab.x[254]));
    QTA_LOG_INFO("test", "  x[N-1] = x[255]      = {} (sentinel, should be 0)",
                 quantape_test::num(tab.x[255]));
    QTA_LOG_INFO("test", "  y[0]                 = {}", quantape_test::num(tab.y[0]));
    QTA_LOG_INFO("test", "  y[N-2] = y[254]      = {}", quantape_test::num(tab.y[254]));
    QTA_LOG_INFO("test", "  y[N-1] = y[255]      = {} (should be ~1)",
                 quantape_test::num(tab.y[255]));

    QTA_LOG_INFO("test", "  Verification:");
    QTA_LOG_INFO("test", "    max area rel error = {}",
                 quantape_test::num(v.max_area_error));
    QTA_LOG_INFO("test", "    closure error      = {}  (|y[N] - 1|)",
                 quantape_test::num(v.closure_error));
    QTA_LOG_INFO("test", "    max f(x) error     = {}", quantape_test::num(v.max_f_error));
    QTA_LOG_INFO("test", "    max f^{{-1}} error   = {}",
                 quantape_test::num(v.max_finv_error));
    QTA_LOG_INFO("test", "    monotone x?        = {}", (v.monotone_x ? "YES" : "NO"));
    QTA_LOG_INFO("test", "    monotone y?        = {}", (v.monotone_y ? "YES" : "NO"));

    // Pass/fail checks
    bool ok = true;
    if (v.max_area_error > 1e-10) {
        QTA_LOG_ERROR("test", "  ** FAIL: area error too large");
        ok = false;
    }
    if (v.closure_error > 1e-10) {
        QTA_LOG_ERROR("test", "  ** FAIL: closure error too large");
        ok = false;
    }
    if (v.max_f_error > 1e-15) {
        QTA_LOG_ERROR("test", "  ** FAIL: f consistency error too large");
        ok = false;
    }
    if (!v.monotone_x || !v.monotone_y) {
        QTA_LOG_ERROR("test", "  ** FAIL: monotonicity violated");
        ok = false;
    }
    if (ok) {
        QTA_LOG_INFO("test", "  All table checks PASSED");
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// STATISTICAL TESTS
// ═══════════════════════════════════════════════════════════════════════════

void testStatistics() {
    QTA_LOG_INFO("test", "\n═══════════════════════════════════════════════════════════\n"
                                  " STATISTICAL TESTS  (N = 10,000,000)\n"
                                  "═══════════════════════════════════════════════════════════\n");

    constexpr int N = 10'000'000;
    quantape::math::mc::ZigguratNormal zig(12345);

    // Accumulate moments using compensated summation
    double sum1 = 0, sum2 = 0, sum3 = 0, sum4 = 0;
    double c1 = 0, c2 = 0, c3 = 0, c4 = 0;

    for (int i = 0; i < N; ++i) {
        double x = zig();
        double x2 = x * x;
        double x3 = x2 * x;
        double x4 = x2 * x2;

        // Kahan summation for each moment
        auto kahan = [](double& sum, double& comp, double term) {
            double y = term - comp;
            double t = sum + y;
            comp = (t - sum) - y;
            sum = t;
        };
        kahan(sum1, c1, x);
        kahan(sum2, c2, x2);
        kahan(sum3, c3, x3);
        kahan(sum4, c4, x4);
    }

    double mean = sum1 / N;
    double var = sum2 / N - mean * mean;
    double skew =
        (sum3 / N - 3.0 * mean * sum2 / N + 2.0 * mean * mean * mean) / std::pow(var, 1.5);
    double kurt = (sum4 / N - 4.0 * mean * sum3 / N + 6.0 * mean * mean * sum2 / N -
                   3.0 * mean * mean * mean * mean) /
                  (var * var);

    QTA_LOG_INFO("test", "  Mean       = {}   (expected: 0)", quantape_test::num(mean, 8));
    QTA_LOG_INFO("test", "  Variance   = {}   (expected: 1)", quantape_test::num(var, 8));
    QTA_LOG_INFO("test", "  Skewness   = {}   (expected: 0)", quantape_test::num(skew, 8));
    QTA_LOG_INFO("test", "  Kurtosis   = {}   (expected: 3)", quantape_test::num(kurt, 8));

    // Standard errors (for N=10M)
    double se_mean = 1.0 / std::sqrt(N);
    double se_var = std::sqrt(2.0 / N);
    double se_skew = std::sqrt(6.0 / N);
    double se_kurt = std::sqrt(24.0 / N);

    auto check = [](const char* name, double val, double expected, double se) {
        double z = std::abs(val - expected) / se;
        bool pass = z < 4.0; // 4-sigma tolerance
        QTA_LOG_INFO("test", "  {}: z = {} sigma  {}", name, quantape_test::num(z, 2),
                     pass ? "PASS" : "** FAIL **");
        return pass;
    };

    bool ok = true;
    ok &= check("Mean    ", mean, 0.0, se_mean);
    ok &= check("Variance", var, 1.0, se_var);
    ok &= check("Skewness", skew, 0.0, se_skew);
    ok &= check("Kurtosis", kurt, 3.0, se_kurt);

    if (ok)
        QTA_LOG_INFO("test", "\n  All statistical tests PASSED");
}

// ═══════════════════════════════════════════════════════════════════════════
// TAIL TEST: Check CDF at several sigma levels
// ═══════════════════════════════════════════════════════════════════════════

void testTails() {
    QTA_LOG_INFO("test", "\n═══════════════════════════════════════════════════════════\n"
                                  " TAIL DISTRIBUTION TEST  (N = 50,000,000)\n"
                                  "═══════════════════════════════════════════════════════════\n");

    constexpr long N = 50'000'000;
    quantape::math::mc::ZigguratNormal zig(67890);

    // Count samples beyond various thresholds
    double thresholds[] = {1.0, 2.0, 3.0, 4.0, 5.0};
    long counts[5] = {};

    for (long i = 0; i < N; ++i) {
        double x = std::abs(zig());
        for (int j = 0; j < 5; ++j) {
            if (x > thresholds[j])
                counts[j]++;
        }
    }

    QTA_LOG_INFO("test", "  Threshold  Observed     Expected     Ratio");
    QTA_LOG_INFO("test", "  ─────────  ──────────   ──────────   ─────");
    for (int j = 0; j < 5; ++j) {
        double expected_frac = std::erfc(thresholds[j] / std::sqrt(2.0));
        double observed_frac = static_cast<double>(counts[j]) / N;
        double ratio = observed_frac / expected_frac;
        QTA_LOG_INFO("test", "  {} sigma  {}   {}   {}",
                     quantape_test::num(thresholds[j], 6), quantape_test::num(observed_frac, 6),
                     quantape_test::num(expected_frac, 6), quantape_test::num(ratio, 6));
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// BENCHMARK: Ziggurat vs std::normal_distribution
// ═══════════════════════════════════════════════════════════════════════════

void benchmark() {
    QTA_LOG_INFO("test", "\n═══════════════════════════════════════════════════════════\n"
                                  " BENCHMARK  (1,000,000 samples each)\n"
                                  "═══════════════════════════════════════════════════════════\n");

    quantape::math::mc::ZigguratNormal zig(42);
    std::mt19937_64 mt(42);
    std::normal_distribution<double> std_normal(0.0, 1.0);

    double zig_ns = timeNs([&]() { return zig(); });
    double std_ns = timeNs([&]() { return std_normal(mt); });

    QTA_LOG_INFO("test", "  Ziggurat              : {} ns/sample",
                 quantape_test::num(zig_ns, 2));
    QTA_LOG_INFO("test", "  std::normal_distribution: {} ns/sample",
                 quantape_test::num(std_ns, 2));
    QTA_LOG_INFO("test", "  Speedup               : {}x",
                 quantape_test::num(std_ns / zig_ns, 2));
}

// ═══════════════════════════════════════════════════════════════════════════

int main() {
    testTableGeneration();
    testStatistics();
    testTails();
    benchmark();
    return 0;
}
