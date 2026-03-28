/**
 * @file test_mcfarland.cpp
 * @brief Tests for the McFarland modified ziggurat: statistical tests, benchmarks
 */

#include "Math/McFarlandNormal.h"
#include "Math/PCGRandom.hpp"
#include "Math/ZigguratNormal.h"

#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <random>

// ═══════════════════════════════════════════════════════════════════════════
// HELPERS
// ═══════════════════════════════════════════════════════════════════════════

template <typename F>
double timeNs(F&& fn, int reps = 1'000'000) {
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
// STATISTICAL TESTS
// ═══════════════════════════════════════════════════════════════════════════

void testStatistics() {
    std::cout << "═══════════════════════════════════════════════════════════\n";
    std::cout << " STATISTICAL TESTS  (N = 1,000,000,000)\n";
    std::cout << "═══════════════════════════════════════════════════════════\n\n";

    constexpr int N = 1'000'000'000;
    pcg64 rng(12345);
    mc::McFarlandNormal<pcg64> gen(rng);

    double sum1 = 0, sum2 = 0, sum3 = 0, sum4 = 0;
    double c1 = 0, c2 = 0, c3 = 0, c4 = 0;

    for (int i = 0; i < N; ++i) {
        double x = gen();
        double x2 = x * x;
        double x3 = x2 * x;
        double x4 = x2 * x2;

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

    double se_mean = 1.0 / std::sqrt(N);
    double se_var = std::sqrt(2.0 / N);
    double se_skew = std::sqrt(6.0 / N);
    double se_kurt = std::sqrt(24.0 / N);

    auto check = [](const char* name, double val, double expected, double se) {
        double z = std::abs(val - expected) / se;
        bool pass = z < 4.0;
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
// TAIL TEST
// ═══════════════════════════════════════════════════════════════════════════

void testTails() {
    std::cout << "═══════════════════════════════════════════════════════════\n";
    std::cout << " TAIL DISTRIBUTION TEST  (N = 10,000,000,000)\n";
    std::cout << "═══════════════════════════════════════════════════════════\n\n";

    constexpr long N = 10'000'000'000;
    pcg64 rng(67890);
    mc::McFarlandNormal<pcg64> gen(rng);

    double thresholds[] = {1.0, 2.0, 3.0, 4.0, 5.0};
    long counts[5] = {};

    for (long i = 0; i < N; ++i) {
        double x = std::abs(gen());
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
// BENCHMARK
// ═══════════════════════════════════════════════════════════════════════════

void benchmark() {
    std::cout << "═══════════════════════════════════════════════════════════\n";
    std::cout << " BENCHMARK  (1,000,000 samples each)\n";
    std::cout << "═══════════════════════════════════════════════════════════\n\n";

    // McFarland + PCG64
    pcg64 rng_pcg(42);
    mc::McFarlandNormal<pcg64> mcf_pcg(rng_pcg);

    // McFarland + Xoshiro
    mc::Xoshiro256ss rng_xo(42);
    mc::McFarlandNormal<mc::Xoshiro256ss> mcf_xo(rng_xo);

    // Marsaglia Ziggurat + Xoshiro
    mc::ZigguratNormal zig(42);

    // std::normal_distribution + mt19937_64
    std::mt19937_64 mt(42);
    std::normal_distribution<double> std_normal(0.0, 1.0);

    double mcf_pcg_ns = timeNs([&]() { return mcf_pcg(); });
    double mcf_xo_ns = timeNs([&]() { return mcf_xo(); });
    double zig_ns = timeNs([&]() { return zig(); });
    double std_ns = timeNs([&]() { return std_normal(mt); });

    std::cout << std::setprecision(2) << std::fixed;
    std::cout << "  McFarland + PCG64       : " << mcf_pcg_ns << " ns/sample\n";
    std::cout << "  McFarland + Xoshiro256  : " << mcf_xo_ns << " ns/sample\n";
    std::cout << "  Marsaglia + Xoshiro256  : " << zig_ns << " ns/sample\n";
    std::cout << "  std::normal_distribution: " << std_ns << " ns/sample\n";
    std::cout << "  Speedup vs std (McF+PCG): " << std_ns / mcf_pcg_ns << "x\n";
    std::cout << "  Speedup vs std (Zig+Xo) : " << std_ns / zig_ns << "x\n\n";
}

// ═══════════════════════════════════════════════════════════════════════════

int main() {
    std::cout << "\n";
    testStatistics();
    testTails();
    benchmark();
    return 0;
}
