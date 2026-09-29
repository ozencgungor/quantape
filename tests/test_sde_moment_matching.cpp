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
#include "quantape/mc/Estimator.h"
#include "quantape/mc/MomentMatching.h"
#include "quantape/mc/RandomSource.h"
#include "quantape/mc/SdePrimitives.h"
#include "quantape/mc/SdeSimulator.h"
#include "quantape/mc/TimeGrid.h"

#include <Eigen/Dense>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "TestSupport.h"

using quantape::mc::IidGaussianSource;
using quantape::mc::MomentMatching1D;
using quantape::mc::qeSample;
using quantape::mc::SdeSimulator;
using quantape::mc::StateMatrix;
using quantape::mc::TimeGrid;

namespace {

bool isFiniteBitwise(double x) {
    std::uint64_t bits = 0;
    std::memcpy(&bits, &x, sizeof(double));
    return ((bits >> 52) & 0x7FFULL) != 0x7FFULL;
}

void checkClose(const char* label, double got, double expected, double tol) {
    if (!isFiniteBitwise(got) || !isFiniteBitwise(expected) ||
        !(std::fabs(got - expected) <= tol)) {
        QTA_LOG_ERROR("test", "FAIL: {} got={} expected={} tol={}", label,
                      quantape_test::num(got, 12), quantape_test::num(expected, 12),
                      quantape_test::num(tol, 3));
        std::exit(1);
    }
}

// ── qeSample moment checks across psi regimes ──

void testQeSamplerMoments() {
    struct Case {
        double m;
        double s2;
        const char* tag;
    };
    // psi = s2 / m^2
    const std::vector<Case> cases{{0.04, 0.00016, "quadratic psi=0.1"},
                                  {0.04, 0.0016, "quadratic psi=1.0"},
                                  {0.04, 0.0024, "boundary psi=1.5"},
                                  {0.04, 0.0064, "exponential psi=4"},
                                  {0.04, 0.04, "exponential psi=25"}};
    const std::size_t n = 2000000;
    IidGaussianSource<> source(1, 4242);
    Eigen::MatrixXd z;
    Eigen::MatrixXd u;
    // fill the whole stream once (blocks of n), reuse for all cases
    source.fill(0, 0, n, z);
    source.fillUniform(0, 0, n, 0, 1, u);

    for (const Case& c : cases) {
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
        checkClose(c.tag, mean, c.m, 5.0 * std::sqrt(c.s2 / static_cast<double>(n)));
        checkClose(c.tag, var, c.s2, 6.0 * c.s2 * std::sqrt(2.0 / static_cast<double>(n)));
        CHECK(negatives == 0);
    }
    QTA_LOG_INFO("test", "  [ok] qeSample: exact moments + positivity across psi regimes");
}

void testUniformStreamContract() {
    IidGaussianSource<> source(2, 77);
    Eigen::MatrixXd block;
    source.fillUniform(3, 5, 4, 0, 1, block);
    CHECK(block.rows() == 1 && block.cols() == 4);
    for (Eigen::Index p = 0; p < 4; ++p) {
        CHECK(block(0, p) >= 0.0 && block(0, p) < 1.0);
        Eigen::MatrixXd single;
        source.fillUniform(3, 5 + static_cast<std::size_t>(p), 1, 0, 1, single);
        CHECK(single(0, 0) == block(0, p));
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
    CHECK(allDifferent);
    QTA_LOG_INFO("test", "  [ok] fillUniform: block contract, [0,1), disjoint streams");
}

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

void testCirMomentMatching() {
    const double kappa = 2.0, level = 0.04, sigma = 0.2, v0 = 0.04;
    const std::size_t nPaths = 500000;

    // Exact CIR terminal moments (continuous):
    const double exactMean = v0 * std::exp(-kappa) + level * (1.0 - std::exp(-kappa));
    const double exactVar =
        v0 * sigma * sigma * (std::exp(-kappa) - std::exp(-2.0 * kappa)) / kappa +
        level * sigma * sigma * (1.0 - std::exp(-kappa)) * (1.0 - std::exp(-kappa)) / (2.0 * kappa);

    // Coarse and fine grids: QE matches the conditional moments exactly at
    // any dt, so the terminal moments must equal the exact ones on both.
    for (std::size_t nSteps : {std::size_t(4), std::size_t(52)}) {
        const TimeGrid grid(1.0, nSteps);
        std::vector<std::vector<double>> theta(nSteps, {kappa, level, sigma});
        const SdeSimulator<double, MomentMatching1D<CirQeMoments>> simulator(grid, theta);
        const IidGaussianSource<> source(1, 909);

        double sum = 0.0;
        double sumSq = 0.0;
        std::size_t n = 0;
        std::size_t negatives = 0;
        const auto blocks = simulator.simulate(
            Eigen::VectorXd::Constant(1, v0),
            [](const auto& x, double, const auto& th, auto& out) {
                CirModel{}.drift(x, 0.0, th, out);
            },
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
        checkClose("CIR QE mean", mean, exactMean, 5.0 * seMean);
        checkClose("CIR QE variance", var, exactVar,
                   6.0 * exactVar * std::sqrt(2.0 / static_cast<double>(n)));
        CHECK(negatives == 0);

        // Reproducibility: identical run -> bitwise identical terminal
        const IidGaussianSource<> source2(1, 909);
        const auto blocks2 = simulator.simulate(
            Eigen::VectorXd::Constant(1, v0),
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
        CHECK(diff == 0.0);
        QTA_LOG_INFO("test",
                     "  [ok] CIR QE nSteps={}: mean={} var={} (exact {} / {}), neg=0", nSteps,
                     quantape_test::num(mean, 6), quantape_test::num(var, 3),
                     quantape_test::num(exactMean, 6), quantape_test::num(exactVar, 3));
    }
}

} // namespace

int main() {
    testQeSamplerMoments();
    testUniformStreamContract();
    testCirMomentMatching();
    QTA_LOG_INFO("test", "ALL SDE MOMENT-MATCHING TESTS PASSED");
    return 0;
}
