/**
 * @file test_mcfarland.cpp
 * @brief Tests for the McFarland modified ziggurat: statistical tests, benchmarks
 */

#include "quantape/math/Random/McFarlandNormal.h"
#include "quantape/math/Random/PCGRandom.hpp"
#include "quantape/math/Random/ZigguratNormal.h"

#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <random>

#include "TestSupport.h"

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
    QTA_LOG_INFO("quantape.test", "\n═══════════════════════════════════════════════════════════\n"
                                  " STATISTICAL TESTS  (N = 1,000,000,000)\n"
                                  "═══════════════════════════════════════════════════════════\n");

    constexpr int N = 1'000'000'000;
    pcg64 rng(12345);
    quantape::math::mc::McFarlandNormal<pcg64> gen(rng);

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

    QTA_LOG_INFO("quantape.test", "  Mean       = {}   (expected: 0)", quantape_test::num(mean, 8));
    QTA_LOG_INFO("quantape.test", "  Variance   = {}   (expected: 1)", quantape_test::num(var, 8));
    QTA_LOG_INFO("quantape.test", "  Skewness   = {}   (expected: 0)", quantape_test::num(skew, 8));
    QTA_LOG_INFO("quantape.test", "  Kurtosis   = {}   (expected: 3)", quantape_test::num(kurt, 8));

    double se_mean = 1.0 / std::sqrt(N);
    double se_var = std::sqrt(2.0 / N);
    double se_skew = std::sqrt(6.0 / N);
    double se_kurt = std::sqrt(24.0 / N);

    auto check = [](const char* name, double val, double expected, double se) {
        double z = std::abs(val - expected) / se;
        bool pass = z < 4.0;
        QTA_LOG_INFO("quantape.test", "  {}: z = {} sigma  {}", name, quantape_test::num(z, 2),
                     pass ? "PASS" : "** FAIL **");
        return pass;
    };

    bool ok = true;
    ok &= check("Mean    ", mean, 0.0, se_mean);
    ok &= check("Variance", var, 1.0, se_var);
    ok &= check("Skewness", skew, 0.0, se_skew);
    ok &= check("Kurtosis", kurt, 3.0, se_kurt);

    if (ok)
        QTA_LOG_INFO("quantape.test", "\n  All statistical tests PASSED");
}

// ═══════════════════════════════════════════════════════════════════════════
// TAIL TEST
// ═══════════════════════════════════════════════════════════════════════════

void testTails() {
    QTA_LOG_INFO("quantape.test", "\n═══════════════════════════════════════════════════════════\n"
                                  " TAIL DISTRIBUTION TEST  (N = 10,000,000,000)\n"
                                  "═══════════════════════════════════════════════════════════\n");

    constexpr long N = 10'000'000'000;
    pcg64 rng(67890);
    quantape::math::mc::McFarlandNormal<pcg64> gen(rng);

    double thresholds[] = {1.0, 2.0, 3.0, 4.0, 5.0};
    long counts[5] = {};

    for (long i = 0; i < N; ++i) {
        double x = std::abs(gen());
        for (int j = 0; j < 5; ++j) {
            if (x > thresholds[j])
                counts[j]++;
        }
    }

    QTA_LOG_INFO("quantape.test", "  Threshold  Observed     Expected     Ratio");
    QTA_LOG_INFO("quantape.test", "  ─────────  ──────────   ──────────   ─────");
    for (int j = 0; j < 5; ++j) {
        double expected_frac = std::erfc(thresholds[j] / std::sqrt(2.0));
        double observed_frac = static_cast<double>(counts[j]) / N;
        double ratio = observed_frac / expected_frac;
        QTA_LOG_INFO("quantape.test", "  {} sigma  {}   {}   {}",
                     quantape_test::num(thresholds[j], 6), quantape_test::num(observed_frac, 6),
                     quantape_test::num(expected_frac, 6), quantape_test::num(ratio, 6));
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// BENCHMARK
// ═══════════════════════════════════════════════════════════════════════════

void benchmark() {
    QTA_LOG_INFO("quantape.test", "\n═══════════════════════════════════════════════════════════\n"
                                  " BENCHMARK  (1,000,000 samples each)\n"
                                  "═══════════════════════════════════════════════════════════\n");

    // McFarland + PCG64
    pcg64 rng_pcg(42);
    quantape::math::mc::McFarlandNormal<pcg64> mcf_pcg(rng_pcg);

    // McFarland + Xoshiro
    quantape::math::mc::Xoshiro256ss rng_xo(42);
    quantape::math::mc::McFarlandNormal<quantape::math::mc::Xoshiro256ss> mcf_xo(rng_xo);

    // Marsaglia Ziggurat + Xoshiro
    quantape::math::mc::ZigguratNormal zig(42);

    // std::normal_distribution + mt19937_64
    std::mt19937_64 mt(42);
    std::normal_distribution<double> std_normal(0.0, 1.0);

    double mcf_pcg_ns = timeNs([&]() { return mcf_pcg(); });
    double mcf_xo_ns = timeNs([&]() { return mcf_xo(); });
    double zig_ns = timeNs([&]() { return zig(); });
    double std_ns = timeNs([&]() { return std_normal(mt); });

    QTA_LOG_INFO("quantape.test", "  McFarland + PCG64       : {} ns/sample",
                 quantape_test::num(mcf_pcg_ns, 2));
    QTA_LOG_INFO("quantape.test", "  McFarland + Xoshiro256  : {} ns/sample",
                 quantape_test::num(mcf_xo_ns, 2));
    QTA_LOG_INFO("quantape.test", "  Marsaglia + Xoshiro256  : {} ns/sample",
                 quantape_test::num(zig_ns, 2));
    QTA_LOG_INFO("quantape.test", "  std::normal_distribution: {} ns/sample",
                 quantape_test::num(std_ns, 2));
    QTA_LOG_INFO("quantape.test", "  Speedup vs std (McF+PCG): {}x",
                 quantape_test::num(std_ns / mcf_pcg_ns, 2));
    QTA_LOG_INFO("quantape.test", "  Speedup vs std (Zig+Xo) : {}x",
                 quantape_test::num(std_ns / zig_ns, 2));
}

// ═══════════════════════════════════════════════════════════════════════════

int main() {
    testStatistics();
    testTails();
    benchmark();
    return 0;
}
