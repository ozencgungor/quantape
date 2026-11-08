/**
 * @file test_mcfarland.cpp
 * @brief Gates for the McFarland modified ziggurat RNG.
 *
 * The sample loops are heavy (1e9 moments, 1e10 tails); this file is the
 * `qta_test_math_slow` target and must never join the fast tier.
 *
 * What is gated here:
 *   - exact table identity (`McFarlandTables.*`): FNV-1a checksums over the
 *     raw 64-bit representation of every entry of both table structs, plus
 *     literal pins of the two entries a previous transcription had dropped.
 *     A single wrong number fails, which a purely statistical suite cannot do
 *     (a shifted table still looks normal-ish).
 *   - the exponential overhang threshold (`exponentialTailMass...`): only the
 *     tail branch can return x > X_0, so P(x > X_0) = e^{-X_0} exactly; the
 *     corrupted map[] made that mass ~5% too heavy (z = +7.9 at 5e7 draws).
 *   - the normal path: 4-sigma moments and 4-sigma tail fractions. The tail
 *     rows are real assertions now; the previous version only recorded
 *     properties and therefore could never fail.
 */

#include "quantape/math/Random/McFarlandNormal.h"
#include "quantape/math/Random/PCGRandom.hpp"
#include "quantape/math/Random/ZigguratNormal.h"

#include <bit>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iterator>
#include <random>
#include <string>
#include <type_traits>

#include "support/GtestSupport.h"

using quantape::math::mc::McFarlandExponential;
using quantape::math::mc::McFarlandNormal;
using quantape::math::mc::Xoshiro256ss;
using quantape::math::mc::ZigguratNormal;
using quantape::math::mc::detail::ExpTables;
using quantape::math::mc::detail::NormTables;

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

// ── exact table identity ──────────────────────────────────────────────────
// FNV-1a over the raw 64-bit word of every entry. Doubles are hashed
// bit-exactly (`bit_cast`), integers by value, low byte first, so the digest
// is independent of host endianness and of how the initializer is wrapped.

std::uint64_t fnv1aWord(std::uint64_t h, std::uint64_t v) {
    for (int b = 0; b < 8; ++b) {
        h ^= (v >> (8 * b)) & 0xFFU;
        h *= 0x100000001B3ULL;
    }
    return h;
}

template <typename T, std::size_t N>
std::uint64_t tableChecksum(const T (&table)[N]) {
    static_assert(sizeof(T) <= 8, "table entries are at most 64 bits");
    std::uint64_t h = 0xCBF29CE484222325ULL;
    for (std::size_t i = 0; i < N; ++i) {
        std::uint64_t word = 0;
        if constexpr (std::is_same_v<T, double>) {
            word = std::bit_cast<std::uint64_t>(table[i]);
        } else {
            word = static_cast<std::uint64_t>(table[i]);
        }
        h = fnv1aWord(h, word);
    }
    return h;
}

constexpr double kPow2_63 = 9223372036854775808.0;

} // namespace

// ── tables ────────────────────────────────────────────────────────────────

TEST(McFarlandTables, exponentialTablesMatchUpstreamReference) {
    // Reference: cd-mcfarland/fast_prng, revision
    // fd75d779b715b6957b6727e9c079f8eef8b58f6f (master, 2025-05-21),
    // exponential.h (__exp_X__, Y, map, ipmf). Recompute these constants only
    // from a verified upstream fetch -- never from the table under test.
    EXPECT_EQ(tableChecksum(ExpTables::X), 0xbe82cf58fc38c3a1ULL);
    EXPECT_EQ(tableChecksum(ExpTables::Y), 0x1e56ada5b7af45c9ULL);
    EXPECT_EQ(tableChecksum(ExpTables::map), 0x9094d3b1de3c33beULL);
    EXPECT_EQ(tableChecksum(ExpTables::ipmf), 0xfdbe85f404a82814ULL);

    // The two rows a previous transcription dropped, pinned literally (an
    // exact literal round-trips through both TUs). Y carries N_BINS + 1 = 253
    // entries; a short initializer leaves Y[252] == 0.0 and shifts every band
    // from index 110 up, which moments tests cannot see.
    EXPECT_EQ(ExpTables::Y[110], 1.63902456195e-20);
    EXPECT_EQ(ExpTables::Y[252], 1.08420217249e-19);
    EXPECT_GT(ExpTables::Y[252], ExpTables::Y[251]);

    // map[176] was the dropped entry; map[255] must still be the reference's
    // final 0, not the stray 0 that an off-by-one appended.
    EXPECT_EQ(ExpTables::map[176], 252);
    EXPECT_EQ(ExpTables::map[255], 0);

    // Shape: X strictly decreasing from X[0] = X_0 / 2^63 down to the 0
    // sentinel, Y strictly increasing, every alias target inside the bins.
    EXPECT_NEAR(ExpTables::X[0] * kPow2_63, ExpTables::X_0, 1e-9);
    EXPECT_EQ(ExpTables::X[ExpTables::N_BINS], 0.0);
    for (int i = 0; i < ExpTables::N_BINS; ++i) {
        EXPECT_GT(ExpTables::X[i], ExpTables::X[i + 1]) << "X not decreasing at " << i;
        EXPECT_LT(ExpTables::Y[i], ExpTables::Y[i + 1]) << "Y not increasing at " << i;
    }
    for (int i = 0; i < 256; ++i) {
        EXPECT_LE(ExpTables::map[i], ExpTables::N_BINS) << "alias target out of range at " << i;
    }
}

TEST(McFarlandTables, normalTablesMatchUpstreamReference) {
    // Reference: cd-mcfarland/fast_prng, revision
    // fd75d779b715b6957b6727e9c079f8eef8b58f6f, normal.h.
    EXPECT_EQ(tableChecksum(NormTables::X), 0x0c9e5ca9dd6f0e99ULL);
    EXPECT_EQ(tableChecksum(NormTables::Y), 0x4f0d127389f58dc6ULL);
    EXPECT_EQ(tableChecksum(NormTables::map), 0x430aa0885bdbb027ULL);
    EXPECT_EQ(tableChecksum(NormTables::ipmf), 0x60c3b0e747035028ULL);

    // N_BINS + 1 boundaries each; X ends with the 0.0 sentinel, Y with the
    // non-zero upper edge, so a short initializer shows up as a moved digest
    // and as a failed monotonicity check below.
    EXPECT_NEAR(NormTables::X[0] * kPow2_63, NormTables::X_0, 1e-9);
    EXPECT_EQ(NormTables::X[NormTables::N_BINS], 0.0);
    EXPECT_GT(NormTables::Y[NormTables::N_BINS], 0.0);
    EXPECT_EQ(NormTables::j_inflection, 204);
    for (int i = 0; i < NormTables::N_BINS; ++i) {
        EXPECT_GT(NormTables::X[i], NormTables::X[i + 1]) << "X not decreasing at " << i;
        EXPECT_LT(NormTables::Y[i], NormTables::Y[i + 1]) << "Y not increasing at " << i;
    }
    for (int i = 0; i < 256; ++i) {
        EXPECT_LE(NormTables::map[i], NormTables::N_BINS) << "alias target out of range at " << i;
    }
}

// ── exponential path ──────────────────────────────────────────────────────

TEST(McFarland, exponentialTailMassMatchesExactTailProbability) {
    // X_0 is the overhang threshold: every early-exit value is X[i] * int63
    // with X[i] * 2^63 <= X_0, so x > X_0 is produced only by the tail branch
    // (j == 0) and must occur with exactly e^{-X_0}. The corrupted map[] gave
    // the tail one column too many: +5% mass (z = +7.9 at 5e7 draws).
    constexpr long N = 100'000'000;
    pcg64 rng(67890);
    McFarlandExponential<pcg64> gen(rng);

    long count = 0;
    long double sum = 0.0L;
    for (long i = 0; i < N; ++i) {
        const double x = gen();
        sum += x;
        if (x > ExpTables::X_0) {
            ++count;
        }
    }

    const double p = std::exp(-ExpTables::X_0);
    const double expected = static_cast<double>(N) * p;
    const double sd = std::sqrt(expected * (1.0 - p));
    const double z = (static_cast<double>(count) - expected) / sd;
    recordNumber("exp_tail_observed", static_cast<double>(count) / static_cast<double>(N));
    recordNumber("exp_tail_expected", p);
    recordNumber("exp_tail_ratio", static_cast<double>(count) / expected);
    recordNumber("exp_tail_z", z);
    EXPECT_LT(std::abs(z), 4.0) << "P(x > X_0) must be e^-X_0 = " << p << ", got " << count
                                << " of " << N;

    // E[x] = 1 for the unit exponential; the same defect biased the mean by
    // +2.6e-4 (1.000264), which this 4-sigma band catches at N = 1e8.
    const double mean = static_cast<double>(sum / static_cast<long double>(N));
    const double meanZ = (mean - 1.0) * std::sqrt(static_cast<double>(N));
    recordNumber("exp_mean", mean);
    recordNumber("exp_mean_z", meanZ);
    EXPECT_LT(std::abs(meanZ), 4.0) << "E[x] must be 1, got " << mean;
}

// ── normal path ───────────────────────────────────────────────────────────

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

TEST(McFarland, tailFractionsMatchErfcWithinFourSigma) {
    constexpr long N = 10'000'000'000;
    pcg64 rng(67890);
    McFarlandNormal<pcg64> gen(rng);

    // The last row is NormTables::X_0, the start of the normal tail branch --
    // the only branch that consults the exponential sampler -- so it also
    // gates that hand-off, not just the density.
    const double thresholds[] = {1.0, 2.0, 3.0, 4.0, 5.0, NormTables::X_0};
    const char* labels[] = {"1_sigma", "2_sigma", "3_sigma", "4_sigma", "5_sigma", "X_0"};
    constexpr int kThresholds = static_cast<int>(std::size(thresholds));
    long counts[kThresholds] = {};

    for (long i = 0; i < N; ++i) {
        const double x = std::abs(gen());
        for (int j = 0; j < kThresholds; ++j) {
            if (x > thresholds[j]) {
                ++counts[j];
            }
        }
    }

    for (int j = 0; j < kThresholds; ++j) {
        const double expected_frac = std::erfc(thresholds[j] / std::sqrt(2.0));
        const double expected = static_cast<double>(N) * expected_frac;
        const double sd = std::sqrt(expected * (1.0 - expected_frac));
        const double z = (static_cast<double>(counts[j]) - expected) / sd;
        const std::string prefix = std::string("tail_") + labels[j];
        recordNumber(prefix + "_observed", static_cast<double>(counts[j]) / static_cast<double>(N));
        recordNumber(prefix + "_expected", expected_frac);
        recordNumber(prefix + "_ratio", static_cast<double>(counts[j]) / expected);
        recordNumber(prefix + "_z", z);
        SCOPED_TRACE(prefix);
        EXPECT_LT(std::abs(z), 4.0) << "threshold " << thresholds[j] << ": got " << counts[j]
                                    << " of " << N << ", expected " << expected;
    }
}

// ── benchmark ─────────────────────────────────────────────────────────────

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

    // Timing is informational; only that a sample is produced per call is a gate.
    EXPECT_GT(mcf_pcg_ns, 0.0);
    EXPECT_GT(mcf_xo_ns, 0.0);
}
