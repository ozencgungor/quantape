// test_sobol_generator.cpp — Sobol generator + SDE engine source
//
// Gates:
//   - dimension 1 known values (original-order semantics)
//   - 1D bin exactness and 2D digital-net exactness (vs tValue2D) for fixtures
//   - digital shifts: reproducibility, seed separation, unbiased normal
//     integration E[exp(sigma Z)] over shift replicas
//   - engine integration: SdeSimulator GBM terminal call with
//     SobolGaussianSource, bitwise schedule/block invariance, BS comparison
//   - layout validation (table too small throws), dimension mapping
//
// Stan-free: Sobol headers + mc/ + processes/.
#include "quantape/math/Random/Sobol/GF2.h"
#include "quantape/math/Random/Sobol/SobolGenerator.h"
#include "quantape/math/Random/Sobol/SobolQuality.h"
#include "quantape/mc/Estimator.h"
#include "quantape/mc/RandomSource.h"
#include "quantape/mc/SdeSimulator.h"
#include "quantape/mc/SobolSource.h"
#include "quantape/mc/TimeGrid.h"
#include "quantape/mc/processes/SdeProcesses.h"

#include <Eigen/Dense>

#include <cmath>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <ostream>
#include <random>
#include <string>
#include <thread>
#include <vector>

#include "support/GtestSupport.h"

using namespace quantape::math::mc;
using namespace quantape::math::mc::sobol;
using namespace quantape::mc;
using namespace quantape::processes;

namespace {

std::vector<Entry> makeFixture(int nDims, std::uint64_t seed) {
    std::mt19937_64 rng(seed);
    std::vector<Entry> out;
    std::uint32_t dim = 2;
    for (int degree = 1; degree <= 10 && static_cast<int>(out.size()) < nDims; ++degree) {
        for (const std::uint64_t poly : gf2::enumerate_primitive(degree)) {
            if (static_cast<int>(out.size()) >= nDims) {
                break;
            }
            Entry e;
            e.dim = dim++;
            e.s = static_cast<std::uint32_t>(degree);
            e.a = gf2::encode_a(poly, degree);
            e.m.resize(static_cast<std::size_t>(degree));
            for (int k = 1; k <= degree; ++k) {
                e.m[static_cast<std::size_t>(k - 1)] = ((rng() % (1ULL << (k - 1))) << 1) | 1;
            }
            out.push_back(std::move(e));
        }
    }
    return out;
}

double normalCdf(double x) {
    return 0.5 * std::erfc(-x / std::sqrt(2.0));
}

struct DimParam {
    std::uint32_t dim;
};

void PrintTo(const DimParam& p, std::ostream* os) {
    *os << "Dim" << p.dim;
}

} // namespace

class SobolGeneratorTest : public ::testing::Test {
protected:
    std::vector<Entry> fixture = makeFixture(100, 4);
};

class SobolBin1DTest : public SobolGeneratorTest, public ::testing::WithParamInterface<DimParam> {};

class SobolBin2DTest : public SobolGeneratorTest, public ::testing::WithParamInterface<DimParam> {};

TEST_F(SobolGeneratorTest, dimensionOneKnownValuesAndDimensionCount) {
    const SobolGenerator gen(fixture, SobolOptions{0, 0, true});
    EXPECT_EQ(gen.uniform(0, 1), 0.0);
    EXPECT_EQ(gen.uniform(1, 1), 0.5);
    EXPECT_EQ(gen.uniform(2, 1), 0.25);
    EXPECT_EQ(gen.uniform(3, 1), 0.75);
    EXPECT_EQ(gen.uniform(4, 1), 0.125);
    EXPECT_EQ(gen.dimensionCount(), 101u);
}

TEST_P(SobolBin1DTest, oneDimension) {
    const std::uint32_t dim = GetParam().dim;
    const SobolGenerator gen(fixture, SobolOptions{0, 0, true});
    constexpr int kM = 10;
    const std::uint32_t N = 1u << kM;

    // 1D: 64 bins, 16 points each, for several fixture dims.
    std::vector<int> counts(64, 0);
    for (std::uint32_t p = 0; p < N; ++p) {
        counts[static_cast<std::size_t>(gen.uniformBits(p, dim) >> (64 - 6))]++;
    }
    int bin = 0;
    for (int c : counts) {
        SCOPED_TRACE("bin " + std::to_string(bin++));
        EXPECT_EQ(c, 16);
    }
}

TEST_P(SobolBin2DTest, twoDimensionalDigitalNet) {
    const std::uint32_t d = GetParam().dim;
    const SobolGenerator gen(fixture, SobolOptions{0, 0, true});
    constexpr int kM = 10;
    const std::uint32_t N = 1u << kM;

    // 2D: t-value from the validated SobolQuality implementation.
    std::vector<DirectionMatrix> mats(fixture.size() + 2);
    mats[0] = identityMatrix();
    for (std::size_t i = 0; i < fixture.size(); ++i) {
        mats[i + 1] = directionMatrix(fixture[i]);
    }
    const int t = tValue2D(mats[d - 2], mats[d - 1], kM);
    const int budget = kM - t;
    const int b1 = budget / 2, b2 = budget - b1;
    std::vector<int> counts(std::size_t(1) << budget, 0);
    for (std::uint32_t p = 0; p < N; ++p) {
        const std::uint32_t x = static_cast<std::uint32_t>(gen.uniformBits(p, d - 1) >> (64 - b1));
        const std::uint32_t y = static_cast<std::uint32_t>(gen.uniformBits(p, d) >> (64 - b2));
        counts[(std::size_t(x) << b2) | y]++;
    }
    int bin = 0;
    for (int c : counts) {
        SCOPED_TRACE("bin " + std::to_string(bin++));
        EXPECT_EQ(c, (1 << t));
    }
}

INSTANTIATE_TEST_SUITE_P(Table, SobolBin1DTest,
                         ::testing::Values(DimParam{1u}, DimParam{2u}, DimParam{3u}, DimParam{10u},
                                           DimParam{50u}, DimParam{100u}),
                         [](const ::testing::TestParamInfo<DimParam>& info) {
                             return "Dim" + std::to_string(info.param.dim);
                         });

INSTANTIATE_TEST_SUITE_P(Table, SobolBin2DTest,
                         ::testing::Values(DimParam{3u}, DimParam{9u}, DimParam{27u}, DimParam{55u},
                                           DimParam{99u}),
                         [](const ::testing::TestParamInfo<DimParam>& info) {
                             return "Dim" + std::to_string(info.param.dim);
                         });

TEST_F(SobolGeneratorTest, digitalShiftReproducibilityAndPointOffset) {
    const SobolGenerator plain(fixture, SobolOptions{0, 1, true});
    const SobolGenerator alt(fixture, SobolOptions{0, 0, true});
    const SobolGenerator shifted(fixture, SobolOptions{12345, 1, true});
    EXPECT_NE(shifted.uniformBits(7, 5), alt.uniformBits(7, 5));
    EXPECT_EQ(shifted.uniformBits(7, 5),
              SobolGenerator(fixture, SobolOptions{12345, 1, true}).uniformBits(7, 5));
    EXPECT_EQ(plain.uniformBits(0, 3), alt.uniformBits(1, 3)); // point offset
}

TEST_F(SobolGeneratorTest, digitalShiftUnbiasedNormalIntegral) {
    // Unbiased normal integration with shift replicas: E[exp(sigma Z)].
    const double sigma = 0.5;
    const double reference = std::exp(0.5 * sigma * sigma);
    const std::uint32_t N = 1u << 12;
    double sum = 0.0;
    const int reps = 16;
    for (int r = 1; r <= reps; ++r) {
        const SobolGenerator g(fixture,
                               SobolOptions{static_cast<std::uint64_t>(r) * 7919, 0, true});
        double est = 0.0;
        for (std::uint32_t p = 0; p < N; ++p) {
            est += std::exp(sigma * g.normal(p, 1));
        }
        sum += est / N;
    }
    const double mean = sum / reps;
    EXPECT_LT(std::fabs(mean - reference), 2e-3);
}

TEST_F(SobolGeneratorTest, engineScheduleAndBlockBitwiseInvariance) {
    const std::size_t steps = 64;
    const TimeGrid grid(1.0, steps);
    std::vector<std::vector<double>> theta(steps);
    const auto x0 = Eigen::VectorXd::Constant(1, 100.0);
    SdeSimulator<double> simulator(grid, theta);
    const GbmProcess model{0.03, 0.2};

    const std::size_t nPaths = 4096;
    const auto makeSource = [this](std::uint64_t seed) {
        return SobolGaussianSource(SobolGenerator(fixture, SobolOptions{seed, 1, true}), 1, steps,
                                   1, 0);
    };
    const auto seqSource = makeSource(0);
    const auto parSource = makeSource(0);

    const auto blocksSeq = simulator.simulate(x0, driftOf(model), diffusionOf(model), seqSource,
                                              nPaths, 1024, Schedule::Sequential);
    const auto blocksPar = simulator.simulate(x0, driftOf(model), diffusionOf(model), parSource,
                                              nPaths, 1024, Schedule::Parallel);

    double maxDiff = 0.0;
    for (std::size_t b = 0; b < blocksSeq.size(); ++b) {
        maxDiff = std::max(
            maxDiff,
            (blocksSeq[b].states.back() - blocksPar[b].states.back()).cwiseAbs().maxCoeff());
    }
    EXPECT_EQ(maxDiff, 0.0);

    // Block-size invariance (same total paths, different blocking).
    const auto blocksTight = simulator.simulate(x0, driftOf(model), diffusionOf(model),
                                                makeSource(0), nPaths, 512, Schedule::Sequential);
    maxDiff = 0.0;
    for (std::size_t b = 0; b < blocksSeq.size(); ++b) {
        const std::size_t lo = b * 1024, hi = std::min<std::size_t>(lo + 1024, nPaths);
        // compare path-wise through flat indexing of the tight blocking
        for (std::size_t p = lo; p < hi; ++p) {
            const std::size_t tb = p / 512, tp = p % 512;
            maxDiff = std::max(
                maxDiff,
                std::fabs(blocksSeq[b].states.back()(0, static_cast<Eigen::Index>(p - lo)) -
                          blocksTight[tb].states.back()(0, static_cast<Eigen::Index>(tp))));
        }
    }
    EXPECT_EQ(maxDiff, 0.0);
}

TEST_F(SobolGeneratorTest, qmcVsIidVsBlackScholes) {
    const std::size_t steps = 64;
    const TimeGrid grid(1.0, steps);
    std::vector<std::vector<double>> theta(steps);
    const auto x0 = Eigen::VectorXd::Constant(1, 100.0);
    SdeSimulator<double> simulator(grid, theta);
    const GbmProcess model{0.03, 0.2};

    const std::size_t nPaths = 4096;
    const auto makeSource = [this](std::uint64_t seed) {
        return SobolGaussianSource(SobolGenerator(fixture, SobolOptions{seed, 1, true}), 1, steps,
                                   1, 0);
    };

    const auto payoff = [](const PathBlock<double>& b, Eigen::VectorXd& out) {
        out = (b.states.back().row(0).array() - 100.0).max(0.0).matrix().transpose();
    };

    // QMC error bars from digital-shift replicas (as in practice).
    const int reps = 8;
    double sum = 0.0, sumSq = 0.0;
    for (int rep = 1; rep <= reps; ++rep) {
        const auto blocks = simulator.simulate(x0, driftOf(model), diffusionOf(model),
                                               makeSource(static_cast<std::uint64_t>(rep) * 7919),
                                               nPaths, 1024, Schedule::Sequential);
        const auto est = quantape::mc::estimate(blocks, payoff, Schedule::Sequential);
        sum += est.mean;
        sumSq += est.mean * est.mean;
    }
    const double qmc = sum / reps;
    const double qmcSe = std::sqrt(std::max(sumSq / reps - qmc * qmc, 0.0) / (reps - 1));

    // iid control at identical settings: isolates discretisation bias.
    double iSum = 0.0, iSumSq = 0.0;
    for (int rep = 1; rep <= reps; ++rep) {
        IidGaussianSource<> src(1, 1000 + rep);
        const auto blocks = simulator.simulate(x0, driftOf(model), diffusionOf(model), src, nPaths,
                                               1024, Schedule::Sequential);
        const auto est = quantape::mc::estimate(blocks, payoff, Schedule::Sequential);
        iSum += est.mean;
        iSumSq += est.mean * est.mean;
    }
    const double iid = iSum / reps;
    const double iidSe = std::sqrt(std::max(iSumSq / reps - iid * iid, 0.0) / (reps - 1));

    const double r = 0.03, sigma = 0.2;
    const double d1 = (r + 0.5 * sigma * sigma) / sigma;
    const double d2 = d1 - sigma;
    const double bs = 100.0 * normalCdf(d1) - 100.0 * std::exp(-r) * normalCdf(d2);
    EXPECT_LT(std::fabs(qmc - iid), 5.0 * (qmcSe + iidSe) + 0.02);
    EXPECT_LT(std::fabs(qmc - bs), 0.5);
}

TEST_F(SobolGeneratorTest, layoutValidationAndDimensionMapping) {
    auto shared = std::make_shared<const SobolGenerator>(fixture);
    EXPECT_THROW(SobolGaussianSource(shared, 1, 102, 1, 0), std::invalid_argument);

    const SobolGaussianSource src(shared, 2, 10, 1, 2);
    EXPECT_EQ(src.normalDimension(0, 0), 1u);
    EXPECT_EQ(src.normalDimension(9, 1), 20u);
    EXPECT_EQ(src.uniformDimension(0, 0), 21u);
    EXPECT_EQ(src.uniformDimension(9, 1), 40u);

    Eigen::MatrixXd out(2, 1);
    src.fillPath(3, 5, out, 0);
    EXPECT_EQ(out(0, 0), shared->normal(3, src.normalDimension(5, 0)));
    EXPECT_EQ(out(1, 0), shared->normal(3, src.normalDimension(5, 1)));
    Eigen::MatrixXd u(2, 3);
    src.fillUniform(2, 4, 3, 0, 2, u);
    EXPECT_EQ(u(1, 2), shared->uniform(6, src.uniformDimension(2, 1)));
}

TEST_F(SobolGeneratorTest, optimizationApiReplicasMaxBitsAndPreparedDimension) {
    const SobolGenerator base(fixture, SobolOptions{777, 1, true});
    EXPECT_EQ(base.maxBits(), 32u);
    EXPECT_EQ(base.preparedDimension(), 101u);
    EXPECT_EQ(base.uniformBits(11, 7, 777), base.uniformBits(11, 7));
    EXPECT_EQ(base.normal(11, 7, 777), base.normal(11, 7));
    EXPECT_NE(base.uniformBits(11, 7, 888), base.uniformBits(11, 7, 999));

    // 32-bit storage vs 64-bit: identical top 32 bits, near-identical normals.
    const SobolGenerator wide(fixture, SobolOptions{777, 1, true, 64});
    EXPECT_EQ(wide.maxBits(), 64u);
    for (std::uint64_t p : {0ULL, 1ULL, 12345ULL, 99999ULL}) {
        for (std::uint32_t d : {1u, 2u, 55u}) {
            EXPECT_EQ((base.uniformBits(p, d) >> 32), (wide.uniformBits(p, d) >> 32));
        }
    }
    EXPECT_LT(std::fabs(base.normal(12345, 55) - wide.normal(12345, 55)), 1e-6);

    // maxDimension restricts the build; access beyond prepared throws.
    const SobolGenerator small(fixture, SobolOptions{0, 1, true, 32, 10});
    EXPECT_EQ(small.preparedDimension(), 10u);
}

TEST_F(SobolGeneratorTest, optimizationThrowsOutOfRange) {
    const SobolGenerator base(fixture, SobolOptions{777, 1, true});
    const SobolGenerator small(fixture, SobolOptions{0, 1, true, 32, 10});
    EXPECT_THROW(small.uniformBits(1, 11), std::out_of_range);
    EXPECT_THROW(base.uniformBits(1ULL << 32, 1), std::out_of_range);
}

TEST_F(SobolGeneratorTest, rngJumpAheadAndSeedReproducibility) {
    // Independent gray-code generation from the fixture's words: direct
    // point access must equal stepping to that (gray) index.
    const auto grayIndex = [](std::uint64_t g) {
        std::uint64_t i = g;
        for (std::uint64_t b = g >> 1; b != 0; b >>= 1) {
            i ^= b;
        }
        return i;
    };
    const SobolGenerator jump(fixture, SobolOptions{0, 0, true});
    const int bits = 12;
    std::vector<std::uint32_t> V(static_cast<std::size_t>(bits) + 1, 0);
    for (std::uint32_t d : {2u, 17u, 64u}) {
        const auto m = integerDirectionNumbers(fixture[d - 2], bits);
        for (int k = 1; k <= bits; ++k) {
            V[static_cast<std::size_t>(k)] = static_cast<std::uint32_t>(m[k] << (32 - k));
        }
        const std::uint32_t N = 1u << bits;
        std::vector<std::uint32_t> X(N, 0);
        std::vector<std::uint32_t> C(N, 1);
        for (std::uint32_t i = 1; i < N; ++i) {
            std::uint32_t v = i, c = 1;
            while (v & 1u) {
                v >>= 1;
                ++c;
            }
            C[i] = c;
        }
        for (std::uint32_t i = 1; i < N; ++i) {
            X[i] = X[i - 1] ^ V[C[i - 1]];
        }
        for (std::uint64_t p : {1ULL, 2ULL, 3ULL, 100ULL, 4095ULL}) {
            const std::uint64_t bitValue = jump.uniformBits(p, d, 0) >> 32;
            EXPECT_EQ(bitValue, X[grayIndex(p)]);
        }
    }

    // Seed reproducibility: independent instances agree bitwise.
    const SobolGenerator g1(fixture, SobolOptions{2024, 3, true});
    const SobolGenerator g2(fixture, SobolOptions{2024, 3, true});
    for (std::uint64_t p : {0ULL, 5ULL, 7777ULL}) {
        for (std::uint32_t d : {1u, 9u, 80u}) {
            EXPECT_EQ(g1.uniformBits(p, d), g2.uniformBits(p, d));
            EXPECT_EQ(g1.normal(p, d), g2.normal(p, d));
        }
    }
}

TEST_F(SobolGeneratorTest, parallelFillMatchesSequential) {
    // Thread-safety: parallel fills are bitwise identical to sequential.
    const auto shared =
        std::make_shared<const SobolGenerator>(fixture, SobolOptions{31415, 1, true});
    const SobolGaussianSource src(shared, 2, 8, 1, 0, 999);
    const std::size_t nPaths = 4096, threads = 4;
    Eigen::MatrixXd seqOut(2, static_cast<Eigen::Index>(nPaths));
    Eigen::MatrixXd parOut(2, static_cast<Eigen::Index>(nPaths));
    for (std::size_t p = 0; p < nPaths; ++p) {
        for (std::size_t k = 0; k < 8; ++k) {
            src.fillPath(p, k, seqOut, p); // overwritten per step; compare final
        }
    }
    std::vector<std::thread> workers;
    for (std::size_t t = 0; t < threads; ++t) {
        workers.emplace_back([&, t] {
            const std::size_t lo = t * nPaths / threads, hi = (t + 1) * nPaths / threads;
            for (std::size_t p = lo; p < hi; ++p) {
                for (std::size_t k = 0; k < 8; ++k) {
                    src.fillPath(p, k, parOut, p);
                }
            }
        });
    }
    for (auto& w : workers) {
        w.join();
    }
    EXPECT_TRUE(seqOut == parOut);
}

TEST_F(SobolGeneratorTest, realRefinedTableIsOptional) {
    const char* candidates[] = {"joe-kuo-65536-refined-w256.txt",
                                "../joe-kuo-65536-refined-w256.txt"};
    const char* found = nullptr;
    for (const char* path : candidates) {
        if (std::filesystem::exists(path)) {
            found = path;
            break;
        }
    }
    SKIP_UNLESS_ASSET(found != nullptr, "joe-kuo-65536-refined-w256.txt absent");

    const auto gen = SobolGenerator::fromFile(found, SobolOptions{999, 0, true});
    EXPECT_EQ(gen.dimensionCount(), 65536u);
    for (std::uint32_t dim : {21202u, 40000u, 65536u}) {
        const double z = gen.normal(12345, dim);
        EXPECT_TRUE(std::isfinite(z) && std::fabs(z) < 10.0);
    }
}

TEST_F(SobolGeneratorTest, defaultTableIsMmapSharedAndMatchesText) {
    const std::string def = SobolGenerator::defaultTablePath();
    SKIP_UNLESS_ASSET(!def.empty() && std::filesystem::exists(def),
                      "no compile-time Sobol table configured");

    const auto bin = SobolGenerator::fromDefaultTable(SobolOptions{0, 1, false});
    const std::uint32_t dims = bin.dimensionCount();
    EXPECT_TRUE(bin.mapped());
    EXPECT_GE(dims, 4096u); // committed prebuilt asset is 8192 dims; raw tables reach 65536
    EXPECT_EQ(bin.preparedDimension(), dims);

    const char* txtCandidates[] = {"joe-kuo-65536-refined-w256.txt",
                                   "../joe-kuo-65536-refined-w256.txt"};
    for (const char* txt : txtCandidates) {
        if (!std::filesystem::exists(txt)) {
            continue;
        }
        const auto text = SobolGenerator::fromFile(txt, SobolOptions{0, 1, false});
        for (std::uint64_t p : {1ULL, 999ULL, 123456ULL, (1ULL << 20) + 7}) {
            for (std::uint32_t d : {1u, 2u, std::min(dims, 21202u), dims}) {
                EXPECT_EQ(bin.uniformBits(p, d), text.uniformBits(p, d));
                EXPECT_EQ(bin.normal(p, d), text.normal(p, d));
            }
        }
        break;
    }

    auto a = SobolGenerator::sharedFromDefaultTable(SobolOptions{0, 1, false});
    auto b = SobolGenerator::sharedFromDefaultTable(SobolOptions{0, 1, false});
    auto c = SobolGenerator::sharedFromDefaultTable(SobolOptions{7, 1, false});
    EXPECT_EQ(a.get(), b.get());
    EXPECT_NE(a.get(), c.get());
}
