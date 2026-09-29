// test_sde_simulator.cpp — SDE simulation engine (S0/S1)
//
// Gates:
//   - TimeGrid: explicit/irregular grids, dt, validation
//   - RandomSource contract: block fill == per-path columns, path/step
//     independence, keyed reproducibility
//   - Euler core: OU discrete moments (uniform + irregular grids), GBM
//     (Euler mean exact, terminal call vs Black-Scholes), CIR mean
//   - Reproducibility: simulatePath(i) == block path i bitwise; block-size
//     invariance (bitwise)
//   - Estimator: mean/variance/SE
//
// Stan-free: includes only mc/ headers + Eigen.
#include "quantape/mc/Estimator.h"
#include "quantape/mc/RandomSource.h"
#include "quantape/mc/Schemes.h"
#include "quantape/mc/SdeSimulator.h"
#include "quantape/mc/TimeGrid.h"

#include <Eigen/Dense>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <vector>

#include "quantape/log/Log.h"
#include "quantape/util/Check.h"
using quantape::util::checkClose;
using quantape::util::isFiniteBitwise;

using quantape::mc::diffusionOf;
using quantape::mc::driftOf;
using quantape::mc::Euler;
using quantape::mc::IidGaussianSource;
using quantape::mc::PathBlock;
using quantape::mc::SdeSimulator;
using quantape::mc::StateMatrix;
using quantape::mc::TimeGrid;

// Both sampler policies must satisfy the source concept
static_assert(quantape::mc::RandomSource<IidGaussianSource<quantape::mc::McFarlandSampler>>);
static_assert(quantape::mc::RandomSource<IidGaussianSource<quantape::mc::ZigguratSampler>>);

namespace {

bool bitwiseEqual(const Eigen::MatrixXd& a, const Eigen::MatrixXd& b) {
    if (a.rows() != b.rows() || a.cols() != b.cols()) {
        return false;
    }
    return (a.array() == b.array()).all();
}

// Bitwise finiteness guard: under -ffast-math the compiler folds
// isnan/isfinite to constants, so NaN/Inf silently pass `<=` comparisons.


double bsCall(double s0, double k, double r, double sigma, double t) {
    const double sqrtT = std::sqrt(t);
    const double d1 = (std::log(s0 / k) + (r + 0.5 * sigma * sigma) * t) / (sigma * sqrtT);
    const double d2 = d1 - sigma * sqrtT;
    const double phi = 0.5 * std::erfc(-d1 * M_SQRT1_2);
    const double phi2 = 0.5 * std::erfc(-d2 * M_SQRT1_2);
    return s0 * phi - k * std::exp(-r * t) * phi2;
}

// ── Models (batch functors; theta = piecewise-constant per interval) ──

// dX = kappa (level - X) dt + sigma dW ; theta = {kappa, level, sigma}
struct OuModel {
    template <typename Scalar>
    void drift(const StateMatrix<Scalar>& x, double, const std::vector<Scalar>& th,
               StateMatrix<Scalar>& out) const {
        out = (th[0] * (th[1] - x.array())).matrix();
    }
    template <typename Scalar>
    void diffusion(const StateMatrix<Scalar>& x, double, const std::vector<Scalar>& th, std::size_t,
                   StateMatrix<Scalar>& out) const {
        out.resize(x.rows(), x.cols());
        out.setConstant(th[2]);
    }
};

// dS = mu S dt + sigma S dW ; theta = {mu, sigma}
struct GbmModel {
    template <typename Scalar>
    void drift(const StateMatrix<Scalar>& x, double, const std::vector<Scalar>& th,
               StateMatrix<Scalar>& out) const {
        out = (th[0] * x.array()).matrix();
    }
    template <typename Scalar>
    void diffusion(const StateMatrix<Scalar>& x, double, const std::vector<Scalar>& th, std::size_t,
                   StateMatrix<Scalar>& out) const {
        out = (th[1] * x.array()).matrix();
    }
};

// dV = kappa (level - V) dt + sigma sqrt(max(V,0)) dW ; theta = {kappa, level, sigma}
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

// ── Tests ──

void testTimeGrid() {
    const TimeGrid uniform(1.0, 4);
    CHECK(uniform.nSteps() == 4);
    checkClose("uniform dt", uniform.dt(0), 0.25, 0.0);
    checkClose("uniform tMax", uniform.tMax(), 1.0, 0.0);

    const TimeGrid irregular(std::vector<double>{0.0, 0.1, 0.4, 1.0});
    CHECK(irregular.nSteps() == 3);
    checkClose("irregular dt0", irregular.dt(0), 0.1, 1e-15);
    checkClose("irregular dt2", irregular.dt(2), 0.6, 1e-15);

    bool threw = false;
    try {
        TimeGrid bad(std::vector<double>{0.0, 0.5, 0.5});
    } catch (const std::invalid_argument&) {
        threw = true;
    }
    CHECK(threw);
    QTA_LOG_INFO("test", "  [ok] TimeGrid: uniform/irregular/validation");
}

void testSourceContract() {
    IidGaussianSource<> source(3, 42);
    CHECK(source.factorCount() == 3);

    Eigen::MatrixXd block;
    source.fill(5, 10, 4, block);
    CHECK(block.rows() == 3 && block.cols() == 4);

    // block fill == per-path fills for the same (path, step)
    for (std::size_t j = 0; j < 4; ++j) {
        Eigen::MatrixXd single;
        source.fill(5, 10 + j, 1, single);
        for (Eigen::Index i = 0; i < 3; ++i) {
            CHECK(single(i, 0) == block(i, static_cast<Eigen::Index>(j)));
        }
    }
    // different (path, step, factor) => different draws
    Eigen::MatrixXd other;
    source.fill(6, 10, 4, other);
    CHECK(!bitwiseEqual(block, other));
    Eigen::MatrixXd otherPath;
    source.fill(5, 11, 4, otherPath);
    CHECK(!bitwiseEqual(block, otherPath));
    // valid normal draws: sample mean/variance sanity over many draws
    IidGaussianSource<> many(1, 7);
    const std::size_t n = 200000;
    double sum = 0.0;
    double sumSq = 0.0;
    Eigen::MatrixXd draws;
    for (std::size_t k = 0; k < n / 1000; ++k) {
        many.fill(k, 0, 1000, draws);
        for (Eigen::Index i = 0; i < draws.size(); ++i) {
            sum += draws(i);
            sumSq += draws(i) * draws(i);
        }
    }
    const double mean = sum / static_cast<double>(n);
    const double var = sumSq / static_cast<double>(n) - mean * mean;
    checkClose("keyed normal mean", mean, 0.0, 0.02);
    checkClose("keyed normal var", var, 1.0, 0.05);
    QTA_LOG_INFO("test", "  [ok] RandomSource: block contract, independence, moments");
}

void testOuMoments() {
    const double kappa = 1.5, level = 0.8, sigma = 0.4, x0 = 2.0;
    const std::size_t nSteps = 32;
    const TimeGrid grid(1.0, nSteps);
    const std::size_t nPaths = 200000;

    std::vector<std::vector<double>> theta(nSteps, {kappa, level, sigma});
    const SdeSimulator<double> simulator(grid, theta);
    const IidGaussianSource<> source(1, 123);

    // Exact discrete Euler moments for a linear recursion:
    //   E_{k+1} = (1 - k dt) E_k + k l dt,  V_{k+1} = (1 - k dt)^2 V_k + s^2 dt
    double e = x0, v = 0.0;
    for (std::size_t k = 0; k < nSteps; ++k) {
        const double dt = grid.dt(k);
        v = (1.0 - kappa * dt) * (1.0 - kappa * dt) * v + sigma * sigma * dt;
        e = (1.0 - kappa * dt) * e + kappa * level * dt;
    }

    const auto blocks = simulator.simulate(Eigen::VectorXd::Constant(1, x0), driftOf(OuModel{}),
                                           diffusionOf(OuModel{}), source, nPaths, 8192);
    double sum = 0.0, sumSq = 0.0;
    std::size_t n = 0;
    for (const auto& b : blocks) {
        const auto& x = b.states.back();
        for (Eigen::Index p = 0; p < x.cols(); ++p) {
            sum += x(0, p);
            sumSq += x(0, p) * x(0, p);
        }
        n += static_cast<std::size_t>(x.cols());
    }
    const double mean = sum / n;
    const double var = sumSq / n - mean * mean;
    const double seMean = std::sqrt(v / n);
    const double seVar = v * std::sqrt(2.0 / n);
    checkClose("OU mean", mean, e, 5.0 * seMean);
    checkClose("OU var", var, v, 6.0 * seVar);

    // Irregular grid: same recursion with per-step dt
    const TimeGrid irregular(std::vector<double>{0.0, 0.1, 0.25, 0.6, 1.0});
    std::vector<std::vector<double>> thetaIrr(irregular.nSteps(), {kappa, level, sigma});
    const SdeSimulator<double> simulatorIrr(irregular, thetaIrr);
    e = x0;
    v = 0.0;
    for (std::size_t k = 0; k < irregular.nSteps(); ++k) {
        const double dt = irregular.dt(k);
        v = (1.0 - kappa * dt) * (1.0 - kappa * dt) * v + sigma * sigma * dt;
        e = (1.0 - kappa * dt) * e + kappa * level * dt;
    }
    const auto blocksIrr =
        simulatorIrr.simulate(Eigen::VectorXd::Constant(1, x0), driftOf(OuModel{}),
                              diffusionOf(OuModel{}), source, nPaths, 8192);
    sum = sumSq = 0.0;
    for (const auto& b : blocksIrr) {
        const auto& x = b.states.back();
        for (Eigen::Index p = 0; p < x.cols(); ++p) {
            sum += x(0, p);
            sumSq += x(0, p) * x(0, p);
        }
    }
    const double meanIrr = sum / nPaths;
    const double varIrr = sumSq / nPaths - meanIrr * meanIrr;
    checkClose("OU irregular mean", meanIrr, e, 5.0 * std::sqrt(v / nPaths));
    checkClose("OU irregular var", varIrr, v, 6.0 * v * std::sqrt(2.0 / nPaths));
    QTA_LOG_INFO("test", "  [ok] OU discrete moments: uniform + irregular grids");
}

void testGbmPrice() {
    const double mu = 0.05, sigma = 0.2, s0 = 100.0, k = 100.0, t = 1.0;
    const std::size_t nSteps = 250;
    const TimeGrid grid(t, nSteps);
    const std::size_t nPaths = 200000;

    std::vector<std::vector<double>> theta(nSteps, {mu, sigma});
    const SdeSimulator<double> simulator(grid, theta);
    const IidGaussianSource<> source(1, 99);

    const auto blocks = simulator.simulate(Eigen::VectorXd::Constant(1, s0), driftOf(GbmModel{}),
                                           diffusionOf(GbmModel{}), source, nPaths, 8192);
    // Euler mean is exact for the multiplicative recursion: (1 + mu dt)^n
    double eulerMean = s0;
    for (std::size_t k = 0; k < nSteps; ++k) {
        eulerMean *= (1.0 + mu * grid.dt(k));
    }
    double sum = 0.0;
    for (const auto& b : blocks) {
        const auto& x = b.states.back();
        for (Eigen::Index p = 0; p < x.cols(); ++p) {
            sum += x(0, p);
        }
    }
    const double mean = sum / nPaths;
    checkClose("GBM Euler mean", mean, eulerMean,
               5.0 * eulerMean * sigma * std::sqrt(1.0 / nPaths));

    // Terminal call vs Black-Scholes (Euler bias allowed; 1.5% relative)
    double callSum = 0.0;
    for (const auto& b : blocks) {
        const auto& x = b.states.back();
        for (Eigen::Index p = 0; p < x.cols(); ++p) {
            callSum += std::max(x(0, p) - k, 0.0);
        }
    }
    const double call = callSum / nPaths;
    // The engine returns the UNDISCOUNTED expectation E[(S_T - K)+]
    // (discounting belongs to the payoff layer): under drift mu this is
    // e^{mu T} * BS(r = mu).
    const double reference = std::exp(mu * t) * bsCall(s0, k, mu, sigma, t);
    checkClose("GBM undiscounted call", call, reference, 0.01 * reference);
    QTA_LOG_INFO("test", "  [ok] GBM: Euler mean exact, E[call] = {} vs analytic {}",
                 quantape::util::num(call, 4), quantape::util::num(reference, 4));
}

void testCirMean() {
    const double kappa = 2.0, level = 0.04, sigma = 0.2, v0 = 0.04;
    const std::size_t nSteps = 64;
    const TimeGrid grid(0.25, nSteps);
    const std::size_t nPaths = 200000;

    std::vector<std::vector<double>> theta(nSteps, {kappa, level, sigma});
    const SdeSimulator<double> simulator(grid, theta);
    const IidGaussianSource<> source(1, 5);

    // Exact discrete mean recursion; variance recursion with the E[V_k] term
    double e = v0, v = 0.0;
    for (std::size_t k = 0; k < nSteps; ++k) {
        const double dt = grid.dt(k);
        v = (1.0 - kappa * dt) * (1.0 - kappa * dt) * v + sigma * sigma * dt * e;
        e = (1.0 - kappa * dt) * e + kappa * level * dt;
    }
    const auto blocks = simulator.simulate(Eigen::VectorXd::Constant(1, v0), driftOf(CirModel{}),
                                           diffusionOf(CirModel{}), source, nPaths, 8192);
    double sum = 0.0, sumSq = 0.0;
    std::size_t negatives = 0;
    for (const auto& b : blocks) {
        const auto& x = b.states.back();
        for (Eigen::Index p = 0; p < x.cols(); ++p) {
            sum += x(0, p);
            sumSq += x(0, p) * x(0, p);
            if (x(0, p) < 0.0) {
                ++negatives;
            }
        }
    }
    const double mean = sum / nPaths;
    const double var = sumSq / nPaths - mean * mean;
    checkClose("CIR mean", mean, e, 5.0 * std::sqrt(v / nPaths));
    checkClose("CIR var (approx)", var, v, 0.1 * v);
    CHECK(negatives * 10 < nPaths); // Euler may dip negative rarely; not the norm
    QTA_LOG_INFO("test", "  [ok] CIR: mean exact recursion, var close, negatives={}",
                 negatives);
}

void testReproducibility() {
    const std::size_t nSteps = 16;
    const TimeGrid grid(0.5, nSteps);
    std::vector<std::vector<double>> theta(nSteps, {1.0, 0.5, 0.3});
    const SdeSimulator<double> simulator(grid, theta);

    // Block-size invariance: same seed/source, different block sizes ->
    // bitwise-identical trajectories for the same path index.
    double maxDiff = 0.0;
    for (std::size_t blockSize : {std::size_t(1), std::size_t(7), std::size_t(64)}) {
        const IidGaussianSource<> source(1, 2024);
        const auto blocks =
            simulator.simulate(Eigen::VectorXd::Constant(1, 1.0), driftOf(OuModel{}),
                               diffusionOf(OuModel{}), source, 64, blockSize);
        std::size_t globalPath = 0;
        for (const auto& b : blocks) {
            for (std::size_t p = 0; p < b.nPaths; ++p, ++globalPath) {
                const PathBlock<double> single =
                    simulator.simulatePath(Eigen::VectorXd::Constant(1, 1.0), driftOf(OuModel{}),
                                           diffusionOf(OuModel{}), source, globalPath);
                for (std::size_t k = 0; k <= nSteps; ++k) {
                    maxDiff =
                        std::max(maxDiff, std::abs(b.states[k](0, p) - single.states[k](0, 0)));
                }
            }
        }
    }
    CHECK(maxDiff == 0.0);
    QTA_LOG_INFO("test",
                 "  [ok] reproducibility: simulatePath == block path, block-size invariant");
}

struct TerminalCall {
    double strike = 100.0;
    void operator()(const PathBlock<double>& block, Eigen::VectorXd& out) const {
        const auto& xT = block.states.back();
        out.resize(static_cast<Eigen::Index>(block.nPaths));
        for (Eigen::Index p = 0; p < xT.cols(); ++p) {
            out(p) = std::max(xT(0, p) - strike, 0.0);
        }
    }
};

struct ConstantPayoff {
    double value = 1.0;
    void operator()(const PathBlock<double>& block, Eigen::VectorXd& out) const {
        out = Eigen::VectorXd::Constant(static_cast<Eigen::Index>(block.nPaths), value);
    }
};

void testEstimator() {
    const double sigma = 0.2, s0 = 100.0, k = 100.0, t = 1.0;
    const std::size_t nSteps = 250;
    const TimeGrid grid(t, nSteps);
    std::vector<std::vector<double>> theta(nSteps, {0.0, sigma});
    const SdeSimulator<double> simulator(grid, theta);
    const IidGaussianSource<> source(1, 77);
    const std::size_t nPaths = 100000;
    const auto blocks = simulator.simulate(Eigen::VectorXd::Constant(1, s0), driftOf(GbmModel{}),
                                           diffusionOf(GbmModel{}), source, nPaths, 4096);

    // Constant payoff: mean/SE sanity
    const auto constant = quantape::mc::estimate(blocks, ConstantPayoff{2.5});
    checkClose("estimator constant mean", constant.mean, 2.5, 0.0);
    checkClose("estimator constant var", constant.variance, 0.0, 0.0);
    CHECK(constant.nPaths == nPaths);

    // European call: SE consistency (|mean - BS| within ~5 SE)
    const auto call = quantape::mc::estimate(blocks, TerminalCall{k});
    const double reference = bsCall(s0, k, 0.0, sigma, t); // drift 0 => undiscounted
    CHECK(std::fabs(call.mean - reference) <= 5.0 * call.stdError + 0.01 * reference);
    QTA_LOG_INFO("test", "  [ok] estimator: call = {} +/- {} vs BS {}",
                 quantape::util::num(call.mean, 4), quantape::util::num(call.stdError, 4),
                 quantape::util::num(reference, 4));
}

} // namespace

// Parallel schedule must be deterministic: identical blocks bitwise, and
// the estimator combines block partials in order (same result as sequential
// up to floating-point summation order).
void testParallelSchedule() {
    const std::size_t nSteps = 64;
    const TimeGrid grid(1.0, nSteps);
    std::vector<std::vector<double>> theta(nSteps, {1.0, 0.5, 0.3});
    const SdeSimulator<double> simulator(grid, theta);
    const auto x0 = Eigen::VectorXd::Constant(1, 1.0);

    const IidGaussianSource<> seqSource(1, 4242);
    const auto seqBlocks =
        simulator.simulate(x0, driftOf(OuModel{}), diffusionOf(OuModel{}), seqSource, 20000, 1024,
                           quantape::mc::Schedule::Sequential);
    const IidGaussianSource<> parSource(1, 4242);
    const auto parBlocks =
        simulator.simulate(x0, driftOf(OuModel{}), diffusionOf(OuModel{}), parSource, 20000, 1024,
                           quantape::mc::Schedule::Parallel);
    CHECK(seqBlocks.size() == parBlocks.size());
    double maxDiff = 0.0;
    for (std::size_t bi = 0; bi < seqBlocks.size(); ++bi) {
        for (std::size_t k = 0; k < seqBlocks[bi].states.size(); ++k) {
            maxDiff = std::max(
                maxDiff, (seqBlocks[bi].states[k] - parBlocks[bi].states[k]).cwiseAbs().maxCoeff());
        }
    }
    CHECK(maxDiff == 0.0);

    const auto seqEstimate =
        quantape::mc::estimate(seqBlocks, TerminalCall{100.0}, quantape::mc::Schedule::Sequential);
    const auto parEstimate =
        quantape::mc::estimate(parBlocks, TerminalCall{100.0}, quantape::mc::Schedule::Parallel);
    checkClose("parallel estimator mean", parEstimate.mean, seqEstimate.mean,
               1e-12 * std::fabs(seqEstimate.mean) + 1e-14);
    checkClose("parallel estimator var", parEstimate.variance, seqEstimate.variance,
               1e-12 * seqEstimate.variance + 1e-14);

    // Streaming parallel: identical block values, arbitrary call order
    const IidGaussianSource<> streamSeqSource(1, 4242);
    std::vector<double> seqLast;
    simulator.simulateBlocks(
        x0, driftOf(OuModel{}), diffusionOf(OuModel{}), streamSeqSource, 20000, 1024,
        [&seqLast](const quantape::mc::PathBlock<double>& block, std::size_t) {
            seqLast.push_back(block.states.back()(0, 0));
        },
        quantape::mc::Schedule::Sequential);
    const IidGaussianSource<> streamParSource(1, 4242);
    std::vector<double> parLast(seqLast.size(), 0.0);
    simulator.simulateBlocks(
        x0, driftOf(OuModel{}), diffusionOf(OuModel{}), streamParSource, 20000, 1024,
        [&parLast](const quantape::mc::PathBlock<double>& block, std::size_t blockIndex) {
            parLast[blockIndex] = block.states.back()(0, 0);
        },
        quantape::mc::Schedule::Parallel);
    CHECK(seqLast == parLast);

    QTA_LOG_INFO("test",
                 "  [ok] parallel schedule: bitwise block equality, estimator order-stable");
}

int main() {
    testTimeGrid();
    testSourceContract();
    testOuMoments();
    testGbmPrice();
    testCirMean();
    testReproducibility();
    testEstimator();
    testParallelSchedule();
    QTA_LOG_INFO("test", "ALL SDE SIMULATOR TESTS PASSED");
    return 0;
}
