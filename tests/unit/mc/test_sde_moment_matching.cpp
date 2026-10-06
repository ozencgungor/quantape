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
#include "quantape/mc/MomentMatching.h"
#include "quantape/mc/RandomSource.h"
#include "quantape/mc/SdePrimitives.h"
#include "quantape/mc/SdeSimulator.h"
#include "quantape/mc/TimeGrid.h"

#include <Eigen/Dense>

#include <cmath>
#include <cstdint>
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

TEST(MomentMatching, uniformStreamContract) {
    IidGaussianSource<> source(2, 77);
    Eigen::MatrixXd block;
    source.fillUniform(3, 5, 4, 0, 1, block);
    EXPECT_TRUE(block.rows() == 1 && block.cols() == 4);
    for (Eigen::Index p = 0; p < 4; ++p) {
        EXPECT_TRUE(block(0, p) >= 0.0 && block(0, p) < 1.0);
        Eigen::MatrixXd single;
        source.fillUniform(3, 5 + static_cast<std::size_t>(p), 1, 0, 1, single);
        EXPECT_EQ(single(0, 0), block(0, p));
    }
    // normals and uniforms occupy disjoint key spaces
    Eigen::MatrixXd z;
    source.fill(3, 5, 4, z);
    bool allDifferent = true;
    for (Eigen::Index p = 0; p < 4; ++p) {
        if (z(0, p) == block(0, p)) {
            allDifferent = false;
        }
    }
    EXPECT_TRUE(allDifferent);
}

class CirMomentMatchingTest : public ::testing::TestWithParam<std::size_t> {};

TEST_P(CirMomentMatchingTest, cirTerminalMomentsExact) {
    const std::size_t nSteps = GetParam();
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
                         ::testing::Values(std::size_t(4), std::size_t(52)),
                         [](const ::testing::TestParamInfo<std::size_t>& info) {
                             return "Steps" + std::to_string(info.param);
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
