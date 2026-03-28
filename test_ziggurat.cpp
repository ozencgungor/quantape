/**
 * @file test_ziggurat.cpp
 * @brief Tests for the Ziggurat normal RNG: table verification, statistical tests, benchmarks
 */

#include "Math/ZigguratNormal.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <random>
#include <vector>

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
    std::cout << "═══════════════════════════════════════════════════════════\n";
    std::cout << " ZIGGURAT TABLE VERIFICATION\n";
    std::cout << "═══════════════════════════════════════════════════════════\n\n";

    auto tab = mc::generateZigguratTables();
    auto v = mc::verifyZigguratTables(tab);

    std::cout << std::setprecision(16);
    std::cout << "  r (tail cutoff)      = " << tab.r << "\n";
    std::cout << "  v (layer area)       = " << tab.A << "\n";
    std::cout << "  base width v/y[0]    = " << tab.A / tab.y[0] << "\n";
    std::cout << "  x[0] = r             = " << tab.x[0] << "\n";
    std::cout << "  x[1]                 = " << tab.x[1] << "\n";
    std::cout << "  x[N-2] = x[254]      = " << tab.x[254] << "\n";
    std::cout << "  x[N-1] = x[255]      = " << tab.x[255] << " (sentinel, should be 0)\n";
    std::cout << "  y[0]                 = " << tab.y[0] << "\n";
    std::cout << "  y[N-2] = y[254]      = " << tab.y[254] << "\n";
    std::cout << "  y[N-1] = y[255]      = " << tab.y[255] << " (should be ~1)\n\n";

    std::cout << "  Verification:\n";
    std::cout << "    max area rel error = " << v.max_area_error << "\n";
    std::cout << "    closure error      = " << v.closure_error << "  (|y[N] - 1|)\n";
    std::cout << "    max f(x) error     = " << v.max_f_error << "\n";
    std::cout << "    max f^{-1} error   = " << v.max_finv_error << "\n";
    std::cout << "    monotone x?        = " << (v.monotone_x ? "YES" : "NO") << "\n";
    std::cout << "    monotone y?        = " << (v.monotone_y ? "YES" : "NO") << "\n\n";

    // Pass/fail checks
    bool ok = true;
    if (v.max_area_error > 1e-10) {
        std::cout << "  ** FAIL: area error too large\n";
        ok = false;
    }
    if (v.closure_error > 1e-10) {
        std::cout << "  ** FAIL: closure error too large\n";
        ok = false;
    }
    if (v.max_f_error > 1e-15) {
        std::cout << "  ** FAIL: f consistency error too large\n";
        ok = false;
    }
    if (!v.monotone_x || !v.monotone_y) {
        std::cout << "  ** FAIL: monotonicity violated\n";
        ok = false;
    }
    if (ok) {
        std::cout << "  All table checks PASSED\n";
    }
    std::cout << "\n";
}

// ═══════════════════════════════════════════════════════════════════════════
// STATISTICAL TESTS
// ═══════════════════════════════════════════════════════════════════════════

void testStatistics() {
    std::cout << "═══════════════════════════════════════════════════════════\n";
    std::cout << " STATISTICAL TESTS  (N = 10,000,000)\n";
    std::cout << "═══════════════════════════════════════════════════════════\n\n";

    constexpr int N = 10'000'000;
    mc::ZigguratNormal zig(12345);

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

    std::cout << std::setprecision(8);
    std::cout << "  Mean       = " << std::setw(14) << mean << "   (expected: 0)\n";
    std::cout << "  Variance   = " << std::setw(14) << var << "   (expected: 1)\n";
    std::cout << "  Skewness   = " << std::setw(14) << skew << "   (expected: 0)\n";
    std::cout << "  Kurtosis   = " << std::setw(14) << kurt << "   (expected: 3)\n\n";

    // Standard errors (for N=10M)
    double se_mean = 1.0 / std::sqrt(N);
    double se_var = std::sqrt(2.0 / N);
    double se_skew = std::sqrt(6.0 / N);
    double se_kurt = std::sqrt(24.0 / N);

    auto check = [](const char* name, double val, double expected, double se) {
        double z = std::abs(val - expected) / se;
        bool pass = z < 4.0; // 4-sigma tolerance
        std::cout << "  " << name << ": z = " << std::setprecision(2) << z << " sigma  "
                  << (pass ? "PASS" : "** FAIL **") << "\n";
        return pass;
    };

    bool ok = true;
    ok &= check("Mean    ", mean, 0.0, se_mean);
    ok &= check("Variance", var, 1.0, se_var);
    ok &= check("Skewness", skew, 0.0, se_skew);
    ok &= check("Kurtosis", kurt, 3.0, se_kurt);

    if (ok)
        std::cout << "\n  All statistical tests PASSED\n";
    std::cout << "\n";
}

// ═══════════════════════════════════════════════════════════════════════════
// TAIL TEST: Check CDF at several sigma levels
// ═══════════════════════════════════════════════════════════════════════════

void testTails() {
    std::cout << "═══════════════════════════════════════════════════════════\n";
    std::cout << " TAIL DISTRIBUTION TEST  (N = 50,000,000)\n";
    std::cout << "═══════════════════════════════════════════════════════════\n\n";

    constexpr long N = 50'000'000;
    mc::ZigguratNormal zig(67890);

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

    std::cout << std::setprecision(6);
    std::cout << "  Threshold  Observed     Expected     Ratio\n";
    std::cout << "  ─────────  ──────────   ──────────   ─────\n";
    for (int j = 0; j < 5; ++j) {
        double expected_frac = std::erfc(thresholds[j] / std::sqrt(2.0));
        double observed_frac = static_cast<double>(counts[j]) / N;
        double ratio = observed_frac / expected_frac;
        std::cout << "  " << std::setw(5) << thresholds[j] << " sigma"
                  << "  " << std::setw(10) << observed_frac << "   " << std::setw(10)
                  << expected_frac << "   " << std::setw(8) << ratio << "\n";
    }
    std::cout << "\n";
}

// ═══════════════════════════════════════════════════════════════════════════
// BENCHMARK: Ziggurat vs std::normal_distribution
// ═══════════════════════════════════════════════════════════════════════════

void benchmark() {
    std::cout << "═══════════════════════════════════════════════════════════\n";
    std::cout << " BENCHMARK  (1,000,000 samples each)\n";
    std::cout << "═══════════════════════════════════════════════════════════\n\n";

    mc::ZigguratNormal zig(42);
    std::mt19937_64 mt(42);
    std::normal_distribution<double> std_normal(0.0, 1.0);

    double zig_ns = timeNs([&]() { return zig(); });
    double std_ns = timeNs([&]() { return std_normal(mt); });

    std::cout << std::setprecision(2) << std::fixed;
    std::cout << "  Ziggurat              : " << zig_ns << " ns/sample\n";
    std::cout << "  std::normal_distribution: " << std_ns << " ns/sample\n";
    std::cout << "  Speedup               : " << std_ns / zig_ns << "x\n\n";
}

// ═══════════════════════════════════════════════════════════════════════════

int main() {
    std::cout << "\n";
    testTableGeneration();
    testStatistics();
    testTails();
    benchmark();
    return 0;
}
