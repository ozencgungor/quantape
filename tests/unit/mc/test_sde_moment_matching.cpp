// test_sde_moment_matching.cpp — S4: generic QE moment matching
//
// Gates:
//   - qeSample: exact first two moments + non-negativity across psi regimes
//     (quadratic branch, boundary, exponential branch)
//   - Uniform stream contract: block == per-path, values in [0,1),
//     independent from the Gaussian factors
//   - CIR with MomentMatching1D: terminal mean/variance equal the exact CIR
//     moments at ANY grid resolution (conditional moment matching), zero
//     negatives, reproducible across runs
//
// Stan-free.
#include "quantape/math/Random/ZigguratNormal.h"
#include "quantape/mc/MomentMatching.h"
#include "quantape/mc/RandomSource.h"
#include "quantape/mc/SdePrimitives.h"
#include "quantape/mc/SdeSimulator.h"
#include "quantape/mc/TimeGrid.h"

#include <Eigen/Dense>

#include <cmath>
#include <cstdint>
#include <cstring>
#include <ostream>
#include <set>
#include <stdexcept>
#include <vector>

#include "support/GtestSupport.h"

using quantape::mc::IidGaussianSource;
using quantape::mc::MomentMatching1D;
using quantape::mc::qeSample;
using quantape::mc::SdeSimulator;
using quantape::mc::StateMatrix;
using quantape::mc::TimeGrid;

namespace {

// ── qeSample moment checks across psi regimes ──

struct QeCase {
    double m;
    double s2;
    const char* tag;
    const char* id;
};

void PrintTo(const QeCase& c, std::ostream* os) {
    *os << c.id;
}

struct StepsParam {
    std::size_t steps;
};

void PrintTo(const StepsParam& p, std::ostream* os) {
    *os << "Steps" << p.steps;
}

// psi = s2 / m^2
const QeCase kQeCases[] = {
    {0.04, 0.00016, "quadratic psi=0.1", "QuadraticPsi01"},
    {0.04, 0.0016, "quadratic psi=1.0", "QuadraticPsi10"},
    {0.04, 0.0024, "boundary psi=1.5", "BoundaryPsi15"},
    {0.04, 0.0064, "exponential psi=4", "ExponentialPsi4"},
    {0.04, 0.04, "exponential psi=25", "ExponentialPsi25"},
};

// ── CIR with moment matching: exact terminal moments at any resolution ──

struct CirQeMoments {
    template <typename Scalar>
    void mean(const StateMatrix<Scalar>& x, double, double dt, const std::vector<Scalar>& th,
              StateMatrix<Scalar>& out) const {
        using std::exp;
        const Scalar decay = exp(-th[0] * Scalar(dt));
        out = x * decay;
        out.array() += (th[1] * (Scalar(1.0) - decay));
    }
    template <typename Scalar>
    void variance(const StateMatrix<Scalar>& x, double, double dt, const std::vector<Scalar>& th,
                  StateMatrix<Scalar>& out) const {
        using std::exp;
        const Scalar decay = exp(-th[0] * Scalar(dt));
        const Scalar oneMinus = Scalar(1.0) - decay;
        out = x * (th[2] * th[2] * decay * oneMinus / th[0]);
        out.array() += (th[1] * th[2] * th[2] * oneMinus * oneMinus / (Scalar(2.0) * th[0]));
    }
};

struct CirModel {
    template <typename Scalar>
    void drift(const StateMatrix<Scalar>& x, double, const std::vector<Scalar>& th,
               StateMatrix<Scalar>& out) const {
        out = (th[0] * (th[1] - x.array())).matrix();
    }
    template <typename Scalar>
    void diffusion(const StateMatrix<Scalar>& x, double, const std::vector<Scalar>& th, std::size_t,
                   StateMatrix<Scalar>& out) const {
        out = (th[2] * x.array().cwiseMax(Scalar(0.0)).sqrt()).matrix();
    }
};

constexpr double kKappa = 2.0;
constexpr double kLevel = 0.04;
constexpr double kSigma = 0.2;
constexpr double kV0 = 0.04;
constexpr std::size_t kCirPaths = 500000;

} // namespace

class MomentMatchingTest : public ::testing::TestWithParam<QeCase> {};

TEST_P(MomentMatchingTest, qeSampleMomentsAndPositivity) {
    const QeCase& c = GetParam();
    const std::size_t n = 2000000;
    IidGaussianSource<> source(1, 4242);
    Eigen::MatrixXd z;
    Eigen::MatrixXd u;
    // fill the whole stream once (blocks of n), reuse for all cases
    source.fill(0, 0, n, z);
    source.fillUniform(0, 0, n, 0, 1, u);

    double sum = 0.0;
    double sumSq = 0.0;
    std::size_t negatives = 0;
    for (std::size_t i = 0; i < n; ++i) {
        const double v = qeSample(c.m, c.s2, z(0, static_cast<Eigen::Index>(i)),
                                  u(0, static_cast<Eigen::Index>(i)));
        sum += v;
        sumSq += v * v;
        if (v < 0.0) {
            ++negatives;
        }
    }
    const double mean = sum / static_cast<double>(n);
    const double var = sumSq / static_cast<double>(n) - mean * mean;
    CHECK_CLOSE(c.tag, mean, c.m, 5.0 * std::sqrt(c.s2 / static_cast<double>(n)));
    CHECK_CLOSE(c.tag, var, c.s2, 6.0 * c.s2 * std::sqrt(2.0 / static_cast<double>(n)));
    EXPECT_EQ(negatives, 0u);
}

INSTANTIATE_TEST_SUITE_P(Table, MomentMatchingTest, ::testing::ValuesIn(kQeCases),
                         [](const ::testing::TestParamInfo<QeCase>& info) {
                             return info.param.id;
                         });

// The uniform stream contract, stated so that a collision would fail it.
//
// `qeSample` reads its uniform only on the exponential branch, so a uniform
// that aliases a Gaussian key is invisible in most parameter regimes and shows
// up as a wrong price elsewhere. The invariant is therefore asserted
// structurally (disjoint index sets, not "the values differ at one step") and
// pinned reproducibly (block == per-path, fixed golden).
TEST(MomentMatching, uniformKeysAreDisjointFromGaussianKeys) {
    // Sweep the factor count and the stream count: a layout that is disjoint for
    // one (q, U) but not another is still wrong.
    for (std::size_t q : {1u, 2u, 3u, 5u}) {
        for (std::size_t streams : {1u, 2u, 3u, 5u}) {
            SCOPED_TRACE(::testing::Message() << "q=" << q << " streams=" << streams);
            std::set<std::uint64_t> gaussian;
            for (std::size_t step = 0; step < 4096; ++step) {
                for (std::size_t j = 0; j < q; ++j) {
                    gaussian.insert(quantape::mc::keyFor(2024, 3, step, j, q));
                }
            }
            for (std::size_t step = 0; step < 4096; ++step) {
                for (std::size_t s = 0; s < streams; ++s) {
                    EXPECT_EQ(gaussian.count(quantape::mc::uniformKey(2024, 3, step, s)), 0u);
                }
            }
        }
    }
    // Uniform streams must also be disjoint from each other across steps: a
    // stride taken from the call's stream count would alias stream s of step k
    // with stream s' of step k' once the counts differ.
    std::set<std::uint64_t> uniforms;
    for (std::size_t step = 0; step < 4096; ++step) {
        for (std::size_t s = 0; s < 2; ++s) {
            EXPECT_TRUE(uniforms.insert(quantape::mc::uniformKey(2024, 3, step, s)).second);
        }
    }
}

TEST(MomentMatching, uniformStreamIsReproducible) {
    const IidGaussianSource<> source(2, 2024);

    // Values in [0,1), and a block fill is bitwise equal to per-path fills.
    Eigen::MatrixXd block;
    source.fillUniform(3, 5, 4, 0, 1, block);
    ASSERT_TRUE(block.rows() == 1 && block.cols() == 4);
    for (Eigen::Index p = 0; p < 4; ++p) {
        EXPECT_GE(block(0, p), 0.0);
        EXPECT_LT(block(0, p), 1.0);
        Eigen::MatrixXd single;
        source.fillUniform(3, 5 + static_cast<std::size_t>(p), 1, 0, 1, single);
        EXPECT_EQ(single(0, 0), block(0, p));
    }

    // Reject a stream request the fixed stride cannot serve.
    Eigen::MatrixXd tooMany;
    EXPECT_THROW(source.fillUniform(0, 0, 1, 0, quantape::mc::kUniformStreamStride + 1, tooMany),
                 std::out_of_range);
}

// The reproducibility contract for BOTH streams: a fixed (seed, path, step,
// index) must keep producing the same double forever. These constants change
// only if the key map or the sampler changes, and that changes every simulated
// price — so the test is deliberately a hard pin, not a tolerance.
TEST(MomentMatching, keyedStreamsMatchTheirGoldens) {
    auto fnv = [](std::uint64_t h, const double v) {
        std::uint64_t bits = 0;
        std::memcpy(&bits, &v, sizeof bits);
        for (int b = 0; b < 8; ++b) {
            h ^= (bits >> (8 * b)) & 0xFF;
            h *= 0x100000001B3ULL;
        }
        return h;
    };
    const IidGaussianSource<> source(2, 2024);
    std::uint64_t hNormal = 0xcbf29ce484222325ULL;
    std::uint64_t hUniform = 0xd6e8feb86659fd93ULL;
    Eigen::MatrixXd z, u;
    for (std::size_t k = 0; k < 8; ++k) {
        for (std::size_t p : {0u, 1u, 2u}) {
            source.fill(k, p, 1, z);
            hNormal = fnv(hNormal, z(0, 0));
            hNormal = fnv(hNormal, z(1, 0));
            source.fillUniform(k, p, 1, 0, 2, u);
            hUniform = fnv(hUniform, u(0, 0));
            hUniform = fnv(hUniform, u(1, 0));
        }
    }
    EXPECT_EQ(hNormal, 0xe95dca772fa07cfbULL) << "gaussian key map / sampler changed";
    EXPECT_EQ(hUniform, 0xa5f4a8912e55d72dULL) << "uniform key map changed";
}

// ── Ziggurat policy: one shared table, unchanged keyed stream ──
//
// `ZigguratSampler::sample` used to construct a `ZigguratNormal` per draw,
// which regenerated the 256-layer table every time (~0.4 ms/draw). It now
// reads one process-wide table; the draw itself must stay bit-for-bit the old
// table-per-draw stream, so this compares the two paths directly and pins a
// hard golden hash of the keyed stream.
TEST(MomentMatching, zigguratDrawsMatchPerDrawTablePath) {
    std::size_t mismatches = 0;
    for (std::size_t i = 0; i < 5000; ++i) {
        const std::uint64_t key = quantape::mc::splitmix64(
            0x1234ULL + static_cast<std::uint64_t>(i) * 0x9E3779B97F4A7C15ULL);
        const double perDrawTable = quantape::math::mc::ZigguratNormal(key)();
        const double sharedTable = quantape::mc::ZigguratSampler::sample(key);
        if (std::memcmp(&perDrawTable, &sharedTable, sizeof(double)) != 0) {
            ++mismatches;
            if (mismatches == 1) {
                ADD_FAILURE() << "key " << key << ": per-draw table " << perDrawTable
                              << " vs shared table " << sharedTable;
            }
        }
    }
    EXPECT_EQ(mismatches, 0u) << "the shared-table ziggurat changed the keyed stream";
}

TEST(MomentMatching, zigguratStreamMatchesItsGolden) {
    auto fnv = [](std::uint64_t h, const double v) {
        std::uint64_t bits = 0;
        std::memcpy(&bits, &v, sizeof bits);
        for (int b = 0; b < 8; ++b) {
            h ^= (bits >> (8 * b)) & 0xFF;
            h *= 0x100000001B3ULL;
        }
        return h;
    };
    std::uint64_t h = 0xcbf29ce484222325ULL;
    for (std::size_t i = 0; i < 1000; ++i) {
        h = fnv(h, quantape::mc::ZigguratSampler::sample(quantape::mc::keyForIndex(2024, 0, i)));
    }
    EXPECT_EQ(h, 0x656e51ef4b274afcULL) << "ziggurat key map / shared table changed";
}

TEST(MomentMatching, zigguratTableIsBuiltOnce) {
    // Function-local `static const`: every call must hand out the same table
    // object, and that object must be the canonical generated table (so the
    // draw matches a fresh `ZigguratNormal`'s). A per-draw rebuild would fail
    // neither check on its own, but together with the two gates above it pins
    // "shared table, unchanged values".
    EXPECT_EQ(&quantape::mc::zigguratTables(), &quantape::mc::zigguratTables());
    const quantape::math::mc::ZigguratTables fresh = quantape::math::mc::generateZigguratTables();
    EXPECT_EQ(quantape::mc::zigguratTables().r, fresh.r);
    EXPECT_EQ(quantape::mc::zigguratTables().A, fresh.A);
    EXPECT_EQ(quantape::mc::zigguratTables().ktab[0], fresh.ktab[0]);
    EXPECT_EQ(quantape::mc::zigguratTables().wtab[quantape::math::mc::ZigguratTables::N - 1],
              fresh.wtab[quantape::math::mc::ZigguratTables::N - 1]);
}

TEST(MomentMatching, zigguratSourceFillsMatchDirectDraws) {
    quantape::mc::IidGaussianSource<quantape::mc::ZigguratSampler> source(2, 2024);
    Eigen::MatrixXd block;
    source.fill(3, 5, 4, block);
    ASSERT_TRUE(block.rows() == 2 && block.cols() == 4);
    for (std::size_t p = 0; p < 4; ++p) {
        Eigen::MatrixXd single;
        source.fill(3, 5 + p, 1, single);
        for (std::size_t j = 0; j < 2; ++j) {
            EXPECT_EQ(single(static_cast<Eigen::Index>(j), 0),
                      block(static_cast<Eigen::Index>(j), static_cast<Eigen::Index>(p)));
            EXPECT_EQ(
                block(static_cast<Eigen::Index>(j), static_cast<Eigen::Index>(p)),
                quantape::mc::ZigguratSampler::sample(quantape::mc::keyFor(2024, 5 + p, 3, j, 2)));
        }
    }
}

class CirMomentMatchingTest : public ::testing::TestWithParam<StepsParam> {};

TEST_P(CirMomentMatchingTest, cirTerminalMomentsExact) {
    const std::size_t nSteps = GetParam().steps;
    const std::size_t nPaths = kCirPaths;

    // Exact CIR terminal moments (continuous):
    const double exactMean = kV0 * std::exp(-kKappa) + kLevel * (1.0 - std::exp(-kKappa));
    const double exactVar =
        kV0 * kSigma * kSigma * (std::exp(-kKappa) - std::exp(-2.0 * kKappa)) / kKappa +
        kLevel * kSigma * kSigma * (1.0 - std::exp(-kKappa)) * (1.0 - std::exp(-kKappa)) /
            (2.0 * kKappa);

    const TimeGrid grid(1.0, nSteps);
    std::vector<std::vector<double>> theta(nSteps, {kKappa, kLevel, kSigma});
    const SdeSimulator<double, MomentMatching1D<CirQeMoments>> simulator(grid, theta);
    const IidGaussianSource<> source(1, 909);

    double sum = 0.0;
    double sumSq = 0.0;
    std::size_t n = 0;
    std::size_t negatives = 0;
    const auto blocks = simulator.simulate(
        Eigen::VectorXd::Constant(1, kV0),
        [](const auto& x, double, const auto& th, auto& out) { CirModel{}.drift(x, 0.0, th, out); },
        [](const auto& x, double, const auto& th, std::size_t j, auto& out) {
            CirModel{}.diffusion(x, 0.0, th, j, out);
        },
        source, nPaths, 8192);
    for (const auto& b : blocks) {
        const auto& x = b.states.back();
        for (Eigen::Index p = 0; p < x.cols(); ++p) {
            sum += x(0, p);
            sumSq += x(0, p) * x(0, p);
            if (x(0, p) < 0.0) {
                ++negatives;
            }
        }
        n += b.nPaths;
    }
    const double mean = sum / static_cast<double>(n);
    const double var = sumSq / static_cast<double>(n) - mean * mean;
    const double seMean = std::sqrt(exactVar / static_cast<double>(n));
    CHECK_CLOSE("CIR QE mean", mean, exactMean, 5.0 * seMean);
    CHECK_CLOSE("CIR QE variance", var, exactVar,
                6.0 * exactVar * std::sqrt(2.0 / static_cast<double>(n)));
    EXPECT_EQ(negatives, 0u);
}

INSTANTIATE_TEST_SUITE_P(Table, CirMomentMatchingTest,
                         ::testing::Values(StepsParam{4}, StepsParam{52}),
                         [](const ::testing::TestParamInfo<StepsParam>& info) {
                             return "Steps" + std::to_string(info.param.steps);
                         });

TEST(MomentMatching, cirReproducibleAcrossRuns) {
    for (std::size_t nSteps : {std::size_t(4), std::size_t(52)}) {
        SCOPED_TRACE("nSteps=" + std::to_string(nSteps));
        const TimeGrid grid(1.0, nSteps);
        std::vector<std::vector<double>> theta(nSteps, {kKappa, kLevel, kSigma});
        const SdeSimulator<double, MomentMatching1D<CirQeMoments>> simulator(grid, theta);
        const IidGaussianSource<> source(1, 909);
        const auto blocks = simulator.simulate(
            Eigen::VectorXd::Constant(1, kV0),
            [](const auto& x, double, const auto& th, auto& out) {
                CirModel{}.drift(x, 0.0, th, out);
            },
            [](const auto& x, double, const auto& th, std::size_t j, auto& out) {
                CirModel{}.diffusion(x, 0.0, th, j, out);
            },
            source, kCirPaths, 8192);

        // Reproducibility: identical run -> bitwise identical terminal
        const IidGaussianSource<> source2(1, 909);
        const auto blocks2 = simulator.simulate(
            Eigen::VectorXd::Constant(1, kV0),
            [](const auto& x, double, const auto& th, auto& out) {
                CirModel{}.drift(x, 0.0, th, out);
            },
            [](const auto& x, double, const auto& th, std::size_t j, auto& out) {
                CirModel{}.diffusion(x, 0.0, th, j, out);
            },
            source2, 1024, 256);
        double diff = 0.0;
        std::size_t globalPath = 0;
        for (std::size_t bi = 0; bi < blocks2.size(); ++bi) {
            for (std::size_t p = 0; p < blocks2[bi].nPaths; ++p, ++globalPath) {
                const std::size_t blockIndex = globalPath / 8192;
                const std::size_t pathInBlock = globalPath % 8192;
                diff = std::max(diff, std::fabs(blocks2[bi].states.back()(0, p) -
                                                blocks[blockIndex].states.back()(0, pathInBlock)));
            }
        }
        EXPECT_EQ(diff, 0.0);
    }
}
