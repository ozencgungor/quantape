/**
 * @file test_mcfarland.cpp
 * @brief Gates for the McFarland modified ziggurat RNG.
 *
 * The sample loops are heavy (1e9 moments, 1e10 tails); this file is the
 * `qta_test_math_slow` target and must never join the fast tier.
 */

#include "quantape/math/Random/McFarlandNormal.h"
#include "quantape/math/Random/PCGRandom.hpp"
#include "quantape/math/Random/ZigguratNormal.h"

#include <chrono>
#include <cmath>
#include <random>
#include <string>

#include "support/GtestSupport.h"

using quantape::math::mc::McFarlandNormal;
using quantape::math::mc::Xoshiro256ss;
using quantape::math::mc::ZigguratNormal;

namespace {

void recordNumber(const std::string& name, double value) {
    ::testing::Test::RecordProperty(name, quantape::util::num(value));
}

template <typename F>
double timeNs(F&& fn, int reps = 1'000'000) {
    // Warmup
    volatile double sink = 0;
    for (int i = 0; i < 1000; ++i) {
        sink += fn();
    }

    const auto t0 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < reps; ++i) {
        sink += fn();
    }
    const auto t1 = std::chrono::high_resolution_clock::now();

    return std::chrono::duration<double, std::nano>(t1 - t0).count() / reps;
}

} // namespace

TEST(McFarland, momentsMatchNormalWithinFourSigma) {
    constexpr int N = 1'000'000'000;
    pcg64 rng(12345);
    McFarlandNormal<pcg64> gen(rng);

    // Accumulate moments using compensated summation
    double sum1 = 0, sum2 = 0, sum3 = 0, sum4 = 0;
    double c1 = 0, c2 = 0, c3 = 0, c4 = 0;

    for (int i = 0; i < N; ++i) {
        const double x = gen();
        const double x2 = x * x;
        const double x3 = x2 * x;
        const double x4 = x2 * x2;

        // Kahan summation for each moment
        auto kahan = [](double& sum, double& comp, double term) {
            const double y = term - comp;
            const double t = sum + y;
            comp = (t - sum) - y;
            sum = t;
        };
        kahan(sum1, c1, x);
        kahan(sum2, c2, x2);
        kahan(sum3, c3, x3);
        kahan(sum4, c4, x4);
    }

    const double mean = sum1 / N;
    const double var = sum2 / N - mean * mean;
    const double skew =
        (sum3 / N - 3.0 * mean * sum2 / N + 2.0 * mean * mean * mean) / std::pow(var, 1.5);
    const double kurt = (sum4 / N - 4.0 * mean * sum3 / N + 6.0 * mean * mean * sum2 / N -
                         3.0 * mean * mean * mean * mean) /
                        (var * var);

    // Standard errors, 4-sigma tolerance
    const double se_mean = 1.0 / std::sqrt(N);
    const double se_var = std::sqrt(2.0 / N);
    const double se_skew = std::sqrt(6.0 / N);
    const double se_kurt = std::sqrt(24.0 / N);

    const auto expectWithinFourSigma = [](const char* name, double z) {
        SCOPED_TRACE(name);
        EXPECT_LT(z, 4.0);
    };
    expectWithinFourSigma("Mean", std::abs(mean - 0.0) / se_mean);
    expectWithinFourSigma("Variance", std::abs(var - 1.0) / se_var);
    expectWithinFourSigma("Skewness", std::abs(skew - 0.0) / se_skew);
    expectWithinFourSigma("Kurtosis", std::abs(kurt - 3.0) / se_kurt);
}

TEST(McFarland, tailFractionsMatchErfc) {
    constexpr long N = 10'000'000'000;
    pcg64 rng(67890);
    McFarlandNormal<pcg64> gen(rng);

    const double thresholds[] = {1.0, 2.0, 3.0, 4.0, 5.0};
    long counts[5] = {};

    for (long i = 0; i < N; ++i) {
        const double x = std::abs(gen());
        for (int j = 0; j < 5; ++j) {
            if (x > thresholds[j]) {
                counts[j]++;
            }
        }
    }

    for (int j = 0; j < 5; ++j) {
        const double expected_frac = std::erfc(thresholds[j] / std::sqrt(2.0));
        const double observed_frac = static_cast<double>(counts[j]) / N;
        const std::string prefix =
            "tail_" + std::to_string(static_cast<int>(thresholds[j])) + "_sigma";
        recordNumber(prefix + "_observed", observed_frac);
        recordNumber(prefix + "_expected", expected_frac);
        recordNumber(prefix + "_ratio", observed_frac / expected_frac);
    }
}

TEST(McFarland, benchmarkAgainstStdNormal) {
    // McFarland + PCG64
    pcg64 rng_pcg(42);
    McFarlandNormal<pcg64> mcf_pcg(rng_pcg);

    // McFarland + Xoshiro
    Xoshiro256ss rng_xo(42);
    McFarlandNormal<Xoshiro256ss> mcf_xo(rng_xo);

    // Marsaglia Ziggurat + Xoshiro
    ZigguratNormal zig(42);

    // std::normal_distribution + mt19937_64
    std::mt19937_64 mt(42);
    std::normal_distribution<double> std_normal(0.0, 1.0);

    const double mcf_pcg_ns = timeNs([&]() { return mcf_pcg(); });
    const double mcf_xo_ns = timeNs([&]() { return mcf_xo(); });
    const double zig_ns = timeNs([&]() { return zig(); });
    const double std_ns = timeNs([&]() { return std_normal(mt); });

    recordNumber("mcfarland_pcg64_ns_per_sample", mcf_pcg_ns);
    recordNumber("mcfarland_xoshiro_ns_per_sample", mcf_xo_ns);
    recordNumber("marsaglia_xoshiro_ns_per_sample", zig_ns);
    recordNumber("std_normal_ns_per_sample", std_ns);
    recordNumber("speedup_mcfarland_pcg64", std_ns / mcf_pcg_ns);
    recordNumber("speedup_ziggurat", std_ns / zig_ns);
}
