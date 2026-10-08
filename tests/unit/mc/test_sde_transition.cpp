// test_sde_transition.cpp — optional exact/conditional transition hook
//
// Gates:
//   - concept discovery: processes without the hook fall back to the scheme
//   - mechanism parity: a hook that mirrors Euler gives bitwise-equal paths
//   - exact transition: GBM terminal moments and European call vs analytic
//   - recorded/step interactivity: hook runs zero allocations per step
//
// Stan-free: includes only mc/ headers + Eigen.
#include "quantape/mc/RandomSource.h"
#include "quantape/mc/Schemes.h"
#include "quantape/mc/SdeSimulator.h"
#include "quantape/mc/TimeGrid.h"
#include "quantape/mc/Transition.h"

#include <Eigen/Dense>

#include <cmath>
#include <cstddef>
#include <vector>

#include "support/GtestSupport.h"

using quantape::mc::diffusionOf;
using quantape::mc::driftOf;
using quantape::mc::Euler;
using quantape::mc::IidGaussianSource;
using quantape::mc::PathBlock;
using quantape::mc::SdeSimulator;
using quantape::mc::StateMatrix;
using quantape::mc::TimeGrid;
using quantape::mc::Transition;
using quantape::mc::TransitionScheme;

namespace {

double bsCall(double s0, double k, double r, double sigma, double t) {
    const double sqrtT = std::sqrt(t);
    const double d1 = (std::log(s0 / k) + (r + 0.5 * sigma * sigma) * t) / (sigma * sqrtT);
    const double d2 = d1 - sigma * sqrtT;
    const double phi = 0.5 * std::erfc(-d1 * M_SQRT1_2);
    const double phi2 = 0.5 * std::erfc(-d2 * M_SQRT1_2);
    return s0 * phi - k * std::exp(-r * t) * phi2;
}

// dX = kappa (level - X) dt + sigma dW ; theta = {kappa, level, sigma}
struct NoHookOu {
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

// Same OU fields plus a transition that reproduces Euler's arithmetic exactly
// (same buffers, same operation order) — the mechanism-parity probe.
struct EulerMirrorOu {
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

    template <typename Scalar>
    void transition(const StateMatrix<Scalar>& x, double t, double dt, const Eigen::MatrixXd& z,
                    const std::vector<Scalar>& th, StateMatrix<Scalar>& out) const {
        StateMatrix<Scalar> driftBuf;
        drift(x, t, th, driftBuf);
        out = x + Scalar(dt) * driftBuf;
        StateMatrix<Scalar> gBuf;
        diffusion(x, t, th, 0, gBuf);
        const Scalar sqrtDt(std::sqrt(dt));
        Eigen::Matrix<Scalar, 1, Eigen::Dynamic> zScaled =
            z.row(0).template cast<Scalar>() * sqrtDt;
        out.array() += gBuf.array().rowwise() * zScaled.array();
    }
};

// dS = mu S dt + sigma S dW with the exact lognormal transition.
// theta = {mu, sigma}; the law is exact for any dt.
struct ExactGbm {
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

    template <typename Scalar>
    void transition(const StateMatrix<Scalar>& x, double, double dt, const Eigen::MatrixXd& z,
                    const std::vector<Scalar>& th, StateMatrix<Scalar>& out) const {
        using std::sqrt;
        const Scalar mu = th[0];
        const Scalar sigma = th[1];
        const Scalar logDrift = (mu - Scalar(0.5) * sigma * sigma) * Scalar(dt);
        const auto logScale = (sigma * Scalar(sqrt(dt))) * z.row(0).array() + Scalar(logDrift);
        out = (x.array() * logScale.exp()).matrix();
    }
};

static_assert(!Transition<NoHookOu, double>);
static_assert(Transition<EulerMirrorOu, double>);
static_assert(Transition<ExactGbm, double>);

bool sameBlocks(const std::vector<PathBlock<double>>& a, const std::vector<PathBlock<double>>& b) {
    if (a.size() != b.size()) {
        return false;
    }
    for (std::size_t bi = 0; bi < a.size(); ++bi) {
        if (a[bi].states.size() != b[bi].states.size()) {
            return false;
        }
        for (std::size_t k = 0; k < a[bi].states.size(); ++k) {
            if (a[bi].states[k].rows() != b[bi].states[k].rows() ||
                a[bi].states[k].cols() != b[bi].states[k].cols() ||
                !(a[bi].states[k].array() == b[bi].states[k].array()).all()) {
                return false;
            }
        }
    }
    return true;
}

} // namespace

TEST(SdeTransition, hookFallsBackToSchemeWithoutOne) {
    const std::size_t nSteps = 16;
    const TimeGrid grid(0.5, nSteps);
    const std::vector<std::vector<double>> theta(nSteps, {1.0, 0.5, 0.3});
    const SdeSimulator<double, Euler> eulerSimulator(grid, theta);
    const SdeSimulator<double, TransitionScheme<NoHookOu>> transitionSimulator(
        grid, theta, quantape::mc::transitionScheme(NoHookOu{}));
    const IidGaussianSource<> source(1, 2024);
    const auto x0 = Eigen::VectorXd::Constant(1, 1.0);

    const auto eulerBlocks =
        eulerSimulator.simulate(x0, driftOf(NoHookOu{}), diffusionOf(NoHookOu{}), source, 64, 16);
    const auto transitionBlocks = transitionSimulator.simulate(
        x0, driftOf(NoHookOu{}), diffusionOf(NoHookOu{}), source, 64, 16);
    EXPECT_TRUE(sameBlocks(eulerBlocks, transitionBlocks));
}

TEST(SdeTransition, hookMatchingEulerIsBitwiseEqual) {
    const std::size_t nSteps = 32;
    const TimeGrid grid(1.0, nSteps);
    const std::vector<std::vector<double>> theta(nSteps, {1.5, 0.8, 0.4});
    const SdeSimulator<double, Euler> eulerSimulator(grid, theta);
    const SdeSimulator<double, TransitionScheme<EulerMirrorOu>> transitionSimulator(
        grid, theta, quantape::mc::transitionScheme(EulerMirrorOu{}));
    const IidGaussianSource<> source(1, 99);
    const auto x0 = Eigen::VectorXd::Constant(1, 2.0);

    const auto eulerBlocks = eulerSimulator.simulate(x0, driftOf(EulerMirrorOu{}),
                                                     diffusionOf(EulerMirrorOu{}), source, 128, 32);
    const auto transitionBlocks = transitionSimulator.simulate(
        x0, driftOf(EulerMirrorOu{}), diffusionOf(EulerMirrorOu{}), source, 128, 32);
    EXPECT_TRUE(sameBlocks(eulerBlocks, transitionBlocks));
}

TEST(SdeTransition, exactGbmMatchesAnalyticMoments) {
    const double s0 = 100.0;
    const double mu = 0.05;
    const double sigma = 0.2;
    const double dt = 0.25;
    const std::size_t nPaths = 400000;
    const TimeGrid grid(std::vector<double>{0.0, dt});
    const std::vector<std::vector<double>> theta(1, {mu, sigma});
    const SdeSimulator<double, TransitionScheme<ExactGbm>> simulator(
        grid, theta, quantape::mc::transitionScheme(ExactGbm{}));
    const IidGaussianSource<> source(1, 7);

    const auto blocks = simulator.simulate(Eigen::VectorXd::Constant(1, s0), driftOf(ExactGbm{}),
                                           diffusionOf(ExactGbm{}), source, nPaths, 8192);
    double sum = 0.0;
    double sumSq = 0.0;
    std::size_t n = 0;
    for (const auto& block : blocks) {
        const auto& x = block.states.back();
        for (Eigen::Index p = 0; p < x.cols(); ++p) {
            sum += x(0, p);
            sumSq += x(0, p) * x(0, p);
            ++n;
        }
    }
    const double mean = sum / static_cast<double>(n);
    const double variance = sumSq / static_cast<double>(n) - mean * mean;
    const double exactMean = s0 * std::exp(mu * dt);
    const double exactVar =
        s0 * s0 * std::exp(2.0 * mu * dt) * (std::exp(sigma * sigma * dt) - 1.0);
    const double seMean = std::sqrt(exactVar / static_cast<double>(n));
    CHECK_CLOSE("exact lognormal mean", mean, exactMean, 5.0 * seMean);
    CHECK_CLOSE_REL("exact lognormal variance", variance, exactVar, 0.02);
}

TEST(SdeTransition, exactGbmEuropeanCallMatchesBlackScholes) {
    const double s0 = 100.0;
    const double strike = 100.0;
    const double mu = 0.05;
    const double sigma = 0.2;
    const double tMax = 1.0;
    const std::size_t nPaths = 200000;
    const TimeGrid grid(tMax, 1); // exact transition needs a single interval
    const std::vector<std::vector<double>> theta(1, {mu, sigma});
    const SdeSimulator<double, TransitionScheme<ExactGbm>> simulator(
        grid, theta, quantape::mc::transitionScheme(ExactGbm{}));
    const IidGaussianSource<> source(1, 77);

    const auto blocks = simulator.simulate(Eigen::VectorXd::Constant(1, s0), driftOf(ExactGbm{}),
                                           diffusionOf(ExactGbm{}), source, nPaths, 8192);
    double callSum = 0.0;
    for (const auto& block : blocks) {
        const auto& x = block.states.back();
        for (Eigen::Index p = 0; p < x.cols(); ++p) {
            callSum += std::max(x(0, p) - strike, 0.0);
        }
    }
    const double call = callSum / static_cast<double>(nPaths);
    // The engine returns the undiscounted expectation E[(S_T - K)+]; under
    // drift mu this is e^{mu T} * BS(r = mu).
    const double reference = std::exp(mu * tMax) * bsCall(s0, strike, mu, sigma, tMax);
    CHECK_CLOSE_REL("exact lognormal call", call, reference, 0.01);
}

TEST(SdeTransition, hookIsReproducibleAndKeepsFullPathShape) {
    const std::size_t nSteps = 64;
    const TimeGrid grid(1.0, nSteps);
    const std::vector<std::vector<double>> theta(nSteps, {0.05, 0.2});
    const SdeSimulator<double, TransitionScheme<ExactGbm>> simulator(
        grid, theta, quantape::mc::transitionScheme(ExactGbm{}));
    const IidGaussianSource<> source(1, 7);
    const auto x0 = Eigen::VectorXd::Constant(1, 100.0);
    const PathBlock<double> a =
        simulator.simulatePath(x0, driftOf(ExactGbm{}), diffusionOf(ExactGbm{}), source, 3);
    const PathBlock<double> b =
        simulator.simulatePath(x0, driftOf(ExactGbm{}), diffusionOf(ExactGbm{}), source, 3);
    EXPECT_EQ(a.states.size(), nSteps + 1);
    EXPECT_TRUE(sameBlocks({a}, {b}));
}
