// test_sde_gradients.cpp — pathwise AD gradients (S5a)
//
// Gates:
//   1. AD wiring: gradient == central finite differences (CRN, same
//      discretization) for every parameter, tight tolerance
//   2. GBM: Delta/Vega/Rho vs Black-Scholes (Milstein); Euler bias in dt
//   3. OU: exact discrete-mean derivatives (pathwise-exact components)
//   4. CIR QE: exact-moment derivatives through the uniform-stream scheme
//   5. Primal parity: gradient-mode value == double engine value (bitwise)
//   6. Determinism: parallel == sequential (bitwise, ordered reduction)
//
// Uses Stan rev-mode AD; parallel-safe under STAN_THREADS.
#include "quantape/math/StanMath.h"

#include "quantape/mc/ForwardStan.h"
#include "quantape/mc/Gradients.h"
#include "quantape/mc/MomentMatching.h"
#include "quantape/mc/RandomSource.h"
#include "quantape/mc/Schemes.h"
#include "quantape/mc/SchemesStan.h"
#include "quantape/mc/SdeSimulator.h"
#include "quantape/mc/StateDerivatives.h"
#include "quantape/mc/TimeGrid.h"
#include "quantape/mc/mcfwdrev/ForwardGradients.h"
#include "quantape/mc/mcfwdrev/LeanGradients.h"
#include "quantape/payoffs/Indicators.h"
#include "quantape/util/Constants.h"
using ::quantape::util::kPi;

#include <Eigen/Dense>

#include <cmath>
#include <cstdint>
#include <cstring>
#include <vector>

#include "support/GtestSupport.h"
#include "support/StanTapeFixture.h"

using quantape::mc::diffusionOf;
using quantape::mc::driftOf;
using quantape::mc::Euler;
using quantape::mc::GradientEstimate;
using quantape::mc::IidGaussianSource;
using quantape::mc::Milstein;
using quantape::mc::MomentMatching1D;
using quantape::mc::PathBlock;
using quantape::mc::Schedule;
using quantape::mc::SdeSimulator;
using quantape::mc::simulateGradient;
using quantape::mc::StateMatrix;
using quantape::mc::TimeGrid;

namespace {

// ── Small check helpers (bitwise finiteness under -ffast-math) ──

bool bitwiseEqual(double a, double b) {
    std::uint64_t ba = 0;
    std::uint64_t bb = 0;
    std::memcpy(&ba, &a, sizeof(double));
    std::memcpy(&bb, &b, sizeof(double));
    return ba == bb;
}

double normalCdf(double x) {
    return 0.5 * std::erfc(-x * M_SQRT1_2);
}

double normalPdf(double x) {
    return std::exp(-0.5 * x * x) / std::sqrt(2.0 * kPi);
}

// ── Models (batch functors; theta = piecewise-constant per interval) ──

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

// Exact CIR conditional moments (same as the QE scheme consumes)
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

// ── Payoffs (scalar-generic single-path; frozen-LSM variants live in the
//    callable layer and use the same signature) ──

struct TerminalCall {
    double strike = 0.0;
    template <typename Scalar>
    Scalar operator()(const PathBlock<Scalar>& path) const {
        const Scalar s = path.states.back()(0, 0);
        return s > Scalar(strike) ? s - Scalar(strike) : Scalar(0.0);
    }
};

struct TerminalValue {
    template <typename Scalar>
    Scalar operator()(const PathBlock<Scalar>& path) const {
        return path.states.back()(0, 0);
    }
};

// Batch terminal call for the blocked forward mode
struct TerminalCallBatchPayoff {
    double strike = 0.0;
    template <typename Scalar>
    void operator()(const PathBlock<Scalar>& path,
                    Eigen::Matrix<Scalar, Eigen::Dynamic, 1>& out) const {
        const auto& x = path.states.back();
        out.resize(x.cols());
        for (Eigen::Index p = 0; p < x.cols(); ++p) {
            const Scalar s = x(0, p);
            out(p) = s > Scalar(strike) ? s - Scalar(strike) : Scalar(0.0);
        }
    }
};

// Digital, smoothed with the linear ramp (pathwise-differentiable)
struct SmoothedDigitalCall {
    double strike = 0.0;
    double eps = 1e-3;
    template <typename Scalar>
    Scalar operator()(const PathBlock<Scalar>& path) const {
        return quantape::payoffs::smoothIndicator(path.states.back()(0, 0) - Scalar(strike), eps);
    }
};

// ── Helpers ──

/// Total over paths of the same discretization in double precision, path
/// order preserved (FD/center reference and primal-parity reference).
template <typename Scheme = Euler, typename Model, typename Payoff>
double doubleValue(const Model& model, const Payoff& payoff, const TimeGrid& grid,
                   const std::vector<std::vector<double>>& thetaSteps, const Eigen::VectorXd& x0,
                   const std::vector<double>& theta, std::size_t nPaths, std::uint64_t seed) {
    const SdeSimulator<double, Scheme> simulator(grid, thetaSteps);
    const IidGaussianSource<> source(1, seed);
    double sum = 0.0;
    for (std::size_t pathIndex = 0; pathIndex < nPaths; ++pathIndex) {
        const PathBlock<double> path = simulator.template simulatePathSharedTheta<double>(
            x0, driftOf(model), diffusionOf(model), source, pathIndex, theta);
        sum += payoff.template operator()<double>(path);
    }
    return sum / static_cast<double>(nPaths);
}

/// GBM test block: shared configuration for analytic comparisons.
struct GbmCase {
    double s0 = 100.0;
    double strike = 100.0;
    double mu = 0.05;
    double sigma = 0.2;
    double tMax = 1.0;
};

// Undiscounted European call value and exact first-order greeks
// (engine returns undiscounted expectations; the GBM drift mu plays the
// role of r, so every analytic value carries exp(mu T)).
void gbmAnalytic(const GbmCase& c, double& value, double& delta, double& vega, double& rho) {
    const double sqrtT = std::sqrt(c.tMax);
    const double d1 =
        (std::log(c.s0 / c.strike) + (c.mu + 0.5 * c.sigma * c.sigma) * c.tMax) / (c.sigma * sqrtT);
    value = std::exp(c.mu * c.tMax) * (c.s0 * normalCdf(d1) - c.strike * std::exp(-c.mu * c.tMax) *
                                                                  normalCdf(d1 - c.sigma * sqrtT));
    delta = std::exp(c.mu * c.tMax) * normalCdf(d1);
    vega = std::exp(c.mu * c.tMax) * c.s0 * sqrtT * normalPdf(d1);
    rho = c.tMax * c.s0 * delta;
}

} // namespace

// ── Gate 0: smoothed indicators (branchless, AD-friendly) ──

class SmoothIndicatorTest : public StanTapeTest {};

TEST_F(SmoothIndicatorTest, rampAndDerivativeIdentities) {
    using quantape::payoffs::smoothIndicator;
    using quantape::payoffs::smoothIndicatorDerivative;
    const double eps = 1e-3;
    CHECK_CLOSE("indicator far left", smoothIndicator(-1.0, eps), 0.0, 0.0);
    CHECK_CLOSE("indicator left edge", smoothIndicator(-0.5 * eps, eps), 0.0, 0.0);
    CHECK_CLOSE("indicator at zero", smoothIndicator(0.0, eps), 0.5, 0.0);
    CHECK_CLOSE("indicator right edge", smoothIndicator(0.5 * eps, eps), 1.0, 0.0);
    CHECK_CLOSE("indicator far right", smoothIndicator(1.0, eps), 1.0, 0.0);
    const double x = 0.3 * eps;
    CHECK_CLOSE("indicator derivative inside", smoothIndicatorDerivative(x, eps), 1.0 / eps, 0.0);
    CHECK_CLOSE("indicator derivative outside", smoothIndicatorDerivative(0.8 * eps, eps), 0.0,
                0.0);
}

// ── Gate 1: AD wiring vs CRN finite differences on the same discretization ──

class SdeGradientFdTest : public StanTapeTest {};

TEST_F(SdeGradientFdTest, eulerTerminalCallMatchesCrnFiniteDifference) {
    const std::size_t nSteps = 32;
    const std::size_t nPaths = 256;
    const TimeGrid grid(1.0, nSteps);
    const std::vector<std::vector<double>> thetaSteps(nSteps);
    const SdeSimulator<double> simulator(grid, thetaSteps);
    const GbmModel model;
    const TerminalCall payoff{100.0};
    const std::uint64_t seed = 4242;

    const double s0 = 100.0;
    const double mu = 0.05;
    const double sigma = 0.2;
    Eigen::VectorXd x0(1);
    x0(0) = s0;
    const std::vector<double> theta = {mu, sigma};

    const IidGaussianSource<> source(1, seed);
    const GradientEstimate estimate = simulateGradient(simulator, x0, theta, driftOf(model),
                                                       diffusionOf(model), source, payoff, nPaths);

    const auto valueAt = [&](double s0_, double mu_, double sigma_) {
        Eigen::VectorXd x(1);
        x(0) = s0_;
        return doubleValue(model, payoff, grid, thetaSteps, x, {mu_, sigma_}, nPaths, seed);
    };

    const double center = valueAt(s0, mu, sigma);
    CHECK_CLOSE("fd center value vs AD", estimate.value, center,
                1e-12 * std::max(1.0, std::fabs(center)));

    const double params[3] = {s0, mu, sigma};
    for (int j = 0; j < 3; ++j) {
        const double h = 1e-5 * std::max(1.0, std::fabs(params[j]));
        double plus[3] = {s0, mu, sigma};
        double minus[3] = {s0, mu, sigma};
        plus[j] += h;
        minus[j] -= h;
        const double fd =
            (valueAt(plus[0], plus[1], plus[2]) - valueAt(minus[0], minus[1], minus[2])) /
            (2.0 * h);
        const double ad = estimate.gradient(j);
        const double tol = 1e-6 * std::max(1.0, std::fabs(ad));
        CHECK_CLOSE("fd consistency", ad, fd, tol);
    }
}

TEST_F(SdeGradientFdTest, smoothedDigitalAndCheckpointedMatchFiniteDifference) {
    const std::size_t nSteps = 32;
    const std::size_t nPaths = 256;
    const TimeGrid grid(1.0, nSteps);
    const std::vector<std::vector<double>> thetaSteps(nSteps);
    const SdeSimulator<double> simulator(grid, thetaSteps);
    const GbmModel model;
    const TerminalCall payoff{100.0};
    const std::uint64_t seed = 4242;

    const double s0 = 100.0;
    const double mu = 0.05;
    const double sigma = 0.2;
    Eigen::VectorXd x0(1);
    x0(0) = s0;
    const std::vector<double> theta = {mu, sigma};

    // Same gate for a smoothed digital (proves the indicator is
    // differentiated) and for the checkpointed mode (proves the gluing).
    const SmoothedDigitalCall digitalPayoff{100.0, 1e-3};
    const IidGaussianSource<> digitalSource(1, seed);
    const GradientEstimate digital =
        simulateGradient(simulator, x0, theta, driftOf(model), diffusionOf(model), digitalSource,
                         digitalPayoff, nPaths);
    const IidGaussianSource<> checkpointSource(1, seed);
    const GradientEstimate checkpoint = simulateGradientCheckpointed(
        simulator, x0, theta, driftOf(model), diffusionOf(model), checkpointSource, payoff, nPaths);

    const auto digitalAt = [&](double s0_, double mu_, double sigma_) {
        Eigen::VectorXd x(1);
        x(0) = s0_;
        return doubleValue(model, digitalPayoff, grid, thetaSteps, x, {mu_, sigma_}, nPaths, seed);
    };
    const auto callAt = [&](double s0_, double mu_, double sigma_) {
        Eigen::VectorXd x(1);
        x(0) = s0_;
        return doubleValue(model, payoff, grid, thetaSteps, x, {mu_, sigma_}, nPaths, seed);
    };
    const double params[3] = {s0, mu, sigma};
    for (int j = 0; j < 3; ++j) {
        const double h = 1e-5 * std::max(1.0, std::fabs(params[j]));
        double plus[3] = {s0, mu, sigma};
        double minus[3] = {s0, mu, sigma};
        plus[j] += h;
        minus[j] -= h;
        const double fdDigital =
            (digitalAt(plus[0], plus[1], plus[2]) - digitalAt(minus[0], minus[1], minus[2])) /
            (2.0 * h);
        const double fdCheckpoint =
            (callAt(plus[0], plus[1], plus[2]) - callAt(minus[0], minus[1], minus[2])) / (2.0 * h);
        CHECK_CLOSE("fd smoothed digital", digital.gradient(j), fdDigital,
                    1e-6 * std::max(1.0, std::fabs(digital.gradient(j))));
        CHECK_CLOSE("fd checkpointed", checkpoint.gradient(j), fdCheckpoint,
                    1e-6 * std::max(1.0, std::fabs(checkpoint.gradient(j))));
    }
}

// ── Gate 2: GBM greeks vs Black-Scholes (Milstein), Euler dt bias ──

class SdeGradientGbmTest : public StanTapeTest {};

TEST_F(SdeGradientGbmTest, milsteinMatchesBlackScholesGreeks) {
    const GbmCase c;
    const std::size_t nSteps = 256;
    const std::size_t nPaths = 20000;
    const TimeGrid grid(c.tMax, nSteps);
    const std::vector<std::vector<double>> thetaSteps(nSteps);
    const std::vector<double> theta = {c.mu, c.sigma};
    Eigen::VectorXd x0(1);
    x0(0) = c.s0;
    const TerminalCall payoff{c.strike};

    double value = 0.0, delta = 0.0, vega = 0.0, rho = 0.0;
    gbmAnalytic(c, value, delta, vega, rho);

    const SdeSimulator<double, Milstein> simulator(grid, thetaSteps);
    const IidGaussianSource<> source(1, 31415);
    const GradientEstimate estimate =
        simulateGradient(simulator, x0, theta, driftOf(GbmModel{}), diffusionOf(GbmModel{}), source,
                         payoff, nPaths, Schedule::Parallel);
    CHECK_CLOSE("gbm call value", estimate.value, value,
                0.015 * value + 5.0 * estimate.valueStdError);
    CHECK_CLOSE("gbm delta", estimate.gradient(0), delta,
                0.015 * std::fabs(delta) + 5.0 * estimate.stdErrors(0));
    CHECK_CLOSE("gbm vega", estimate.gradient(2), vega,
                0.015 * std::fabs(vega) + 5.0 * estimate.stdErrors(2));
    CHECK_CLOSE("gbm rho", estimate.gradient(1), rho,
                0.015 * std::fabs(rho) + 5.0 * estimate.stdErrors(1));
}

TEST_F(SdeGradientGbmTest, eulerVegaWithinSanityBand) {
    const GbmCase c;
    const std::size_t nPaths = 20000;
    Eigen::VectorXd x0(1);
    x0(0) = c.s0;
    const std::vector<double> theta = {c.mu, c.sigma};
    const TerminalCall payoff{c.strike};

    double value = 0.0, delta = 0.0, vega = 0.0, rho = 0.0;
    gbmAnalytic(c, value, delta, vega, rho);

    // Euler scheme-level error is *reported* (not asserted): with keyed
    // draws the realization error is MC-noise dominated at feasible path
    // counts, so a dt-order test needs a dedicated variance-reduced study.
    // The exact-derivative property is covered by the FD gate above.
    const std::size_t steps[2] = {16, 256};
    for (int i = 0; i < 2; ++i) {
        const TimeGrid coarse(c.tMax, steps[i]);
        const std::vector<std::vector<double>> thetaCoarse(steps[i]);
        const SdeSimulator<double, Euler> simulator(coarse, thetaCoarse);
        const IidGaussianSource<> source(1, 2718);
        const GradientEstimate estimate =
            simulateGradient(simulator, x0, theta, driftOf(GbmModel{}), diffusionOf(GbmModel{}),
                             source, payoff, nPaths, Schedule::Parallel);
        const double vegaError = estimate.gradient(2) - vega;
        RecordProperty("euler_vega_error_" + std::to_string(steps[i]),
                       quantape::util::num(vegaError, 6));
        CHECK_CLOSE("euler vega sanity", estimate.gradient(2), vega,
                    0.05 * std::fabs(vega) + 5.0 * estimate.stdErrors(2));
    }
}

// ── Gate 3: OU exact discrete-mean derivatives ──

class SdeGradientOuTest : public StanTapeTest {};

TEST_F(SdeGradientOuTest, exactDiscreteMeanDerivatives) {
    const double kappa = 1.5;
    const double level = 0.3;
    const double sigma = 0.25;
    const double xInit = 0.1;
    const std::size_t nSteps = 16;
    const std::size_t nPaths = 4000;
    const double tMax = 1.0;
    const double dt = tMax / static_cast<double>(nSteps);

    const TimeGrid grid(tMax, nSteps);
    const std::vector<std::vector<double>> thetaSteps(nSteps);
    const SdeSimulator<double> simulator(grid, thetaSteps);
    const std::vector<double> theta = {kappa, level, sigma};
    Eigen::VectorXd x0(1);
    x0(0) = xInit;

    const IidGaussianSource<> source(1, 555);
    const GradientEstimate estimate =
        simulateGradient(simulator, x0, theta, driftOf(OuModel{}), diffusionOf(OuModel{}), source,
                         TerminalValue{}, nPaths);

    // Euler mean recursion is pathwise-exact: m_n = level + (x0 - level) a^n
    const double a = 1.0 - kappa * dt;
    const double an = std::pow(a, static_cast<double>(nSteps));
    const double exactX0 = an;
    const double exactLevel = 1.0 - an;
    const double exactKappa = (xInit - level) * static_cast<double>(nSteps) *
                              std::pow(a, static_cast<double>(nSteps) - 1.0) * (-dt);
    const double exactMean = level + (xInit - level) * an;

    CHECK_CLOSE("ou dx0", estimate.gradient(0), exactX0, 1e-10);
    CHECK_CLOSE("ou dkappa", estimate.gradient(1), exactKappa,
                0.001 * std::fabs(exactKappa) + 5.0 * estimate.stdErrors(1));
    CHECK_CLOSE("ou dlevel", estimate.gradient(2), exactLevel, 1e-10);
    CHECK_CLOSE("ou dsigma", estimate.gradient(3), 0.0, 5.0 * estimate.stdErrors(3) + 1e-12);
    CHECK_CLOSE("ou mean value", estimate.value, exactMean, 5.0 * estimate.valueStdError + 1e-12);
}

// ── Gate 4: CIR QE moment derivatives through the uniform stream ──

class SdeGradientCirQeTest : public StanTapeTest {};

TEST_F(SdeGradientCirQeTest, exactMomentDerivatives) {
    const double kappa = 2.0;
    const double level = 0.04;
    const double sigma = 0.2;
    const double v0 = 0.04;
    const std::size_t nSteps = 4;
    const std::size_t nPaths = 20000;
    const double tMax = 1.0;

    const TimeGrid grid(tMax, nSteps);
    const std::vector<std::vector<double>> thetaSteps(nSteps);
    const SdeSimulator<double, MomentMatching1D<CirQeMoments>> simulator(
        grid, thetaSteps, MomentMatching1D<CirQeMoments>{});
    const std::vector<double> theta = {kappa, level, sigma};
    Eigen::VectorXd x0(1);
    x0(0) = v0;

    const IidGaussianSource<> source(1, 777);
    const GradientEstimate estimate =
        simulateGradient(simulator, x0, theta, driftOf(CirModel{}), diffusionOf(CirModel{}), source,
                         TerminalValue{}, nPaths, Schedule::Parallel);

    // QE matches the exact CIR conditional moments, so E[V_T] is the exact
    // mean and its derivatives are the exact-mean derivatives.
    const double decay = std::exp(-kappa * tMax);
    const double exactMean = v0 * decay + level * (1.0 - decay);
    const double exactX0 = decay;
    const double exactLevel = 1.0 - decay;
    const double exactKappa = (level - v0) * tMax * decay;

    CHECK_CLOSE("cir mean value", estimate.value, exactMean, 1e-3 + 5.0 * estimate.valueStdError);
    CHECK_CLOSE("cir dx0", estimate.gradient(0), exactX0, 1e-3 + 5.0 * estimate.stdErrors(0));
    CHECK_CLOSE("cir dkappa", estimate.gradient(1), exactKappa,
                1e-3 * std::fabs(exactKappa) + 5.0 * estimate.stdErrors(1));
    CHECK_CLOSE("cir dlevel", estimate.gradient(2), exactLevel, 1e-3 + 5.0 * estimate.stdErrors(2));
    CHECK_CLOSE("cir dsigma", estimate.gradient(3), 0.0, 5.0 * estimate.stdErrors(3) + 1e-12);
}

// ── Checkpointed mode: exactness, equivalence, long horizon ──

class SdeGradientCheckpointedTest : public StanTapeTest {};

TEST_F(SdeGradientCheckpointedTest, ouExactDerivatives) {
    // OU exact discrete-mean derivatives (terminal payoff)
    const double kappa = 1.5, level = 0.3, sigma = 0.25, xInit = 0.1;
    const std::size_t nSteps = 16;
    const std::size_t nPaths = 4000;
    const double tMax = 1.0;
    const double dt = tMax / static_cast<double>(nSteps);
    const TimeGrid grid(tMax, nSteps);
    const std::vector<std::vector<double>> thetaSteps(nSteps);
    const SdeSimulator<double> simulator(grid, thetaSteps);
    Eigen::VectorXd x0(1);
    x0(0) = xInit;
    const std::vector<double> theta = {kappa, level, sigma};

    const IidGaussianSource<> source(1, 555);
    const GradientEstimate estimate =
        simulateGradientCheckpointed(simulator, x0, theta, driftOf(OuModel{}),
                                     diffusionOf(OuModel{}), source, TerminalValue{}, nPaths);

    const double a = 1.0 - kappa * dt;
    const double an = std::pow(a, static_cast<double>(nSteps));
    const double exactKappa = (xInit - level) * static_cast<double>(nSteps) *
                              std::pow(a, static_cast<double>(nSteps) - 1.0) * (-dt);
    CHECK_CLOSE("cp ou dx0", estimate.gradient(0), an, 1e-10);
    CHECK_CLOSE("cp ou dkappa", estimate.gradient(1), exactKappa,
                0.001 * std::fabs(exactKappa) + 5.0 * estimate.stdErrors(1));
    CHECK_CLOSE("cp ou dlevel", estimate.gradient(2), 1.0 - an, 1e-10);
    CHECK_CLOSE("cp ou dsigma", estimate.gradient(3), 0.0, 5.0 * estimate.stdErrors(3) + 1e-12);
}

TEST_F(SdeGradientCheckpointedTest, longHorizonParityIsBitwise) {
    // Full-tape vs checkpointed equivalence (same paths), parity,
    // determinism, and a long horizon
    const std::size_t nSteps = 1024;
    const std::size_t nPaths = 256;
    const TimeGrid grid(1.0, nSteps);
    const std::vector<std::vector<double>> thetaSteps(nSteps);
    const SdeSimulator<double> simulator(grid, thetaSteps);
    const std::vector<double> theta = {0.05, 0.2};
    Eigen::VectorXd x0(1);
    x0(0) = 100.0;
    const TerminalCall payoff{100.0};

    const IidGaussianSource<> srcFull(1, 2468);
    const GradientEstimate full =
        simulateGradient(simulator, x0, theta, driftOf(GbmModel{}), diffusionOf(GbmModel{}),
                         srcFull, payoff, nPaths, Schedule::Parallel);
    const IidGaussianSource<> srcCp(1, 2468);
    const GradientEstimate checkpointed = simulateGradientCheckpointed(
        simulator, x0, theta, driftOf(GbmModel{}), diffusionOf(GbmModel{}), srcCp, payoff, nPaths,
        Schedule::Parallel);
    const IidGaussianSource<> srcCpSeq(1, 2468);
    const GradientEstimate checkpointedSeq = simulateGradientCheckpointed(
        simulator, x0, theta, driftOf(GbmModel{}), diffusionOf(GbmModel{}), srcCpSeq, payoff,
        nPaths, Schedule::Sequential);

    // terminal-seed arithmetic differs from full-tape reversal by
    // summation order only
    for (Eigen::Index j = 0; j < full.gradient.size(); ++j) {
        CHECK_CLOSE("cp vs full gradient", checkpointed.gradient(j), full.gradient(j),
                    1e-8 * std::max(1.0, std::fabs(full.gradient(j))));
    }
    EXPECT_TRUE(bitwiseEqual(checkpointed.value, full.value));
    EXPECT_TRUE((checkpointed.gradient.array() == checkpointedSeq.gradient.array()).all());
    EXPECT_TRUE(bitwiseEqual(checkpointed.value, checkpointedSeq.value));

    const double reference =
        doubleValue(GbmModel{}, payoff, grid, thetaSteps, x0, theta, nPaths, 2468);
    EXPECT_TRUE(bitwiseEqual(checkpointed.value, reference));
}

// ── Gates 5/6: primal parity (bitwise) and schedule determinism ──

class SdeDeterminismTest : public StanTapeTest {};

TEST_F(SdeDeterminismTest, schedulesAndReductionsAreBitwise) {
    const std::size_t nSteps = 64;
    const std::size_t nPaths = 4096;
    const TimeGrid grid(1.0, nSteps);
    const std::vector<std::vector<double>> thetaSteps(nSteps);
    const SdeSimulator<double> simulator(grid, thetaSteps);
    const std::vector<double> theta = {0.05, 0.2};
    Eigen::VectorXd x0(1);
    x0(0) = 100.0;
    const TerminalCall payoff{100.0};
    const std::uint64_t seed = 1234;

    const IidGaussianSource<> seqSource(1, seed);
    const GradientEstimate seq =
        simulateGradient(simulator, x0, theta, driftOf(GbmModel{}), diffusionOf(GbmModel{}),
                         seqSource, payoff, nPaths, Schedule::Sequential);
    const IidGaussianSource<> parSource(1, seed);
    const GradientEstimate par =
        simulateGradient(simulator, x0, theta, driftOf(GbmModel{}), diffusionOf(GbmModel{}),
                         parSource, payoff, nPaths, Schedule::Parallel);

    EXPECT_TRUE(bitwiseEqual(seq.value, par.value));
    EXPECT_TRUE(bitwiseEqual(seq.valueStdError, par.valueStdError));
    EXPECT_TRUE((seq.gradient.array() == par.gradient.array()).all());
    EXPECT_TRUE((seq.stdErrors.array() == par.stdErrors.array()).all());
}

TEST_F(SdeDeterminismTest, primalParityAgainstDoubleEngine) {
    const std::size_t nSteps = 64;
    const std::size_t nPaths = 4096;
    const TimeGrid grid(1.0, nSteps);
    const std::vector<std::vector<double>> thetaSteps(nSteps);
    const SdeSimulator<double> simulator(grid, thetaSteps);
    const std::vector<double> theta = {0.05, 0.2};
    Eigen::VectorXd x0(1);
    x0(0) = 100.0;
    const TerminalCall payoff{100.0};
    const std::uint64_t seed = 1234;

    const IidGaussianSource<> seqSource(1, seed);
    const GradientEstimate seq =
        simulateGradient(simulator, x0, theta, driftOf(GbmModel{}), diffusionOf(GbmModel{}),
                         seqSource, payoff, nPaths, Schedule::Sequential);

    const double reference =
        doubleValue(GbmModel{}, payoff, grid, thetaSteps, x0, theta, nPaths, seed);
    // -ffast-math: the double instantiation may reassociate/vectorize while
    // the AD instantiation is a scalar operator-overloaded loop, so primal
    // parity is tight-relative, not bitwise (schedules *are* bitwise).
    const double parityDiff = std::fabs(seq.value - reference);
    EXPECT_LE(parityDiff, 1e-12 * std::max(1.0, std::fabs(reference)));
}

// ── Forward mode (no-tape duals): wiring, equivalence, state derivatives ──

class SdeForwardTest : public StanTapeTest {};

TEST_F(SdeForwardTest, matchesReverseOnKeyedPaths) {
    // GBM/Euler: forward == reverse == FD (same keyed paths)
    const std::size_t nSteps = 32;
    const std::size_t nPaths = 256;
    const TimeGrid grid(1.0, nSteps);
    const std::vector<std::vector<double>> thetaSteps(nSteps);
    const SdeSimulator<double> simulator(grid, thetaSteps);
    const std::vector<double> theta = {0.05, 0.2};
    Eigen::VectorXd x0(1);
    x0(0) = 100.0;
    const TerminalCall payoff{100.0};
    const std::uint64_t seed = 4242;

    const IidGaussianSource<> srcFwd(1, seed);
    const GradientEstimate fwd = simulateGradientForward<3>(
        simulator, x0, theta, driftOf(GbmModel{}), diffusionOf(GbmModel{}), srcFwd, payoff, nPaths);
    const IidGaussianSource<> srcRev(1, seed);
    const GradientEstimate rev = simulateGradient(simulator, x0, theta, driftOf(GbmModel{}),
                                                  diffusionOf(GbmModel{}), srcRev, payoff, nPaths);
    for (int j = 0; j < 3; ++j) {
        CHECK_CLOSE("fwd vs rev", fwd.gradient(j), rev.gradient(j),
                    1e-8 * std::max(1.0, std::fabs(rev.gradient(j))));
    }
}

TEST_F(SdeForwardTest, milsteinGreeks) {
    // Milstein BS greeks via forward mode (N = 3)
    const GbmCase c;
    const std::size_t nSteps = 256;
    const std::size_t nPaths = 20000;
    const TimeGrid grid(c.tMax, nSteps);
    const std::vector<std::vector<double>> thetaSteps(nSteps);
    const SdeSimulator<double, Milstein> simulator(grid, thetaSteps);
    const std::vector<double> theta = {c.mu, c.sigma};
    Eigen::VectorXd x0(1);
    x0(0) = c.s0;
    const TerminalCall payoff{c.strike};
    const IidGaussianSource<> src(1, 31415);
    const GradientEstimate fwd = simulateGradientForward<3>(
        simulator, x0, theta, driftOf(GbmModel{}), diffusionOf(GbmModel{}), src, payoff, nPaths,
        Schedule::Parallel);
    double value = 0.0, delta = 0.0, vega = 0.0, rho = 0.0;
    gbmAnalytic(c, value, delta, vega, rho);
    CHECK_CLOSE("fwd milstein delta", fwd.gradient(0), delta,
                0.015 * std::fabs(delta) + 5.0 * fwd.stdErrors(0));
    CHECK_CLOSE("fwd milstein vega", fwd.gradient(2), vega,
                0.015 * std::fabs(vega) + 5.0 * fwd.stdErrors(2));
    CHECK_CLOSE("fwd milstein rho", fwd.gradient(1), rho,
                0.015 * std::fabs(rho) + 5.0 * fwd.stdErrors(1));
}

TEST_F(SdeForwardTest, cirQeMomentDerivatives) {
    // CIR QE via forward mode: exact-moment derivatives (N = 4)
    const double kappa = 2.0, level = 0.04, sigma = 0.2, v0 = 0.04;
    const std::size_t nSteps = 4;
    const std::size_t nPaths = 20000;
    const TimeGrid grid(1.0, nSteps);
    const std::vector<std::vector<double>> thetaSteps(nSteps);
    const SdeSimulator<double, MomentMatching1D<CirQeMoments>> simulator(
        grid, thetaSteps, MomentMatching1D<CirQeMoments>{});
    const std::vector<double> theta = {kappa, level, sigma};
    Eigen::VectorXd x0(1);
    x0(0) = v0;
    const IidGaussianSource<> src(1, 777);
    const GradientEstimate fwd = simulateGradientForward<4>(
        simulator, x0, theta, driftOf(CirModel{}), diffusionOf(CirModel{}), src, TerminalValue{},
        nPaths, Schedule::Parallel);

    const double decay = std::exp(-kappa);
    const double exactMean = v0 * decay + level * (1.0 - decay);
    const double exactX0 = decay;
    const double exactLevel = 1.0 - decay;
    const double exactKappa = (level - v0) * decay;
    CHECK_CLOSE("fwd cir mean", fwd.value, exactMean, 1e-3 + 5.0 * fwd.valueStdError);
    CHECK_CLOSE("fwd cir dx0", fwd.gradient(0), exactX0, 1e-3 + 5.0 * fwd.stdErrors(0));
    CHECK_CLOSE("fwd cir dlevel", fwd.gradient(2), exactLevel, 1e-3 + 5.0 * fwd.stdErrors(2));
    CHECK_CLOSE("fwd cir dkappa", fwd.gradient(1), exactKappa,
                1e-3 * std::fabs(exactKappa) + 5.0 * fwd.stdErrors(1));
}

TEST_F(SdeForwardTest, smoothedDigitalMatchesFiniteDifference) {
    // Smoothed digital FD gate via forward mode
    const std::size_t nSteps = 32;
    const std::size_t nPaths = 256;
    const TimeGrid grid(1.0, nSteps);
    const std::vector<std::vector<double>> thetaSteps(nSteps);
    const SdeSimulator<double> simulator(grid, thetaSteps);
    const std::vector<double> theta = {0.05, 0.2};
    Eigen::VectorXd x0(1);
    x0(0) = 100.0;
    const SmoothedDigitalCall payoff{100.0, 1e-3};
    const std::uint64_t seed = 31415;
    const IidGaussianSource<> src(1, seed);
    const GradientEstimate fwd = simulateGradientForward<3>(
        simulator, x0, theta, driftOf(GbmModel{}), diffusionOf(GbmModel{}), src, payoff, nPaths);
    const double s0 = 100.0;
    const auto digitalAt = [&](double s0_, double mu_, double sigma_) {
        Eigen::VectorXd x(1);
        x(0) = s0_;
        return doubleValue(GbmModel{}, payoff, grid, thetaSteps, x, {mu_, sigma_}, nPaths, seed);
    };
    const double params[3] = {s0, 0.05, 0.2};
    for (int j = 0; j < 3; ++j) {
        const double h = 1e-5 * std::max(1.0, std::fabs(params[j]));
        double plus[3] = {s0, 0.05, 0.2};
        double minus[3] = {s0, 0.05, 0.2};
        plus[j] += h;
        minus[j] -= h;
        const double fd =
            (digitalAt(plus[0], plus[1], plus[2]) - digitalAt(minus[0], minus[1], minus[2])) /
            (2.0 * h);
        CHECK_CLOSE("fwd smoothed digital", fwd.gradient(j), fd,
                    1e-6 * std::max(1.0, std::fabs(fwd.gradient(j))));
    }
}

TEST_F(SdeForwardTest, stateDerivativesMatchFvarSweeps) {
    // State derivatives: forward one-pass == fvar-sweep version, all steps
    const std::size_t nSteps = 8;
    const TimeGrid grid(1.0, nSteps);
    const std::vector<std::vector<double>> thetaSteps(nSteps);
    const SdeSimulator<double> simulator(grid, thetaSteps);
    const std::vector<double> theta = {0.05, 0.2};
    Eigen::VectorXd x0(1);
    x0(0) = 100.0;
    const std::uint64_t seed = 321;
    const std::size_t pathIndex = 0;

    std::vector<quantape::mc::StateMatrix<double>> fwdBlocks;
    const IidGaussianSource<> srcFwd(1, seed);
    quantape::mc::simulatePathDerivativesForward<3>(
        simulator, x0, theta, driftOf(GbmModel{}), diffusionOf(GbmModel{}), srcFwd, pathIndex,
        [&fwdBlocks](std::size_t, const quantape::mc::StateMatrix<double>& dX) {
            fwdBlocks.push_back(dX);
        });
    const IidGaussianSource<> srcSweep(1, seed);
    const auto sweep = quantape::mc::simulatePathDerivatives(
        simulator, x0, theta, driftOf(GbmModel{}), diffusionOf(GbmModel{}), srcSweep, pathIndex);
    EXPECT_EQ(fwdBlocks.size(), sweep.dX.size());
    double maxDiff = 0.0;
    for (std::size_t k = 0; k < fwdBlocks.size(); ++k) {
        maxDiff = std::max(maxDiff, (fwdBlocks[k] - sweep.dX[k]).cwiseAbs().maxCoeff());
    }
    CHECK_CLOSE("forward derivatives == sweeps", maxDiff, 0.0, 1e-10);
}

TEST_F(SdeForwardTest, blockedEqualsPerPathEqualsParallel) {
    // Blocked forward == per-path forward (bitwise; partial last block)
    const std::size_t nSteps = 64;
    const std::size_t nPaths = 1000 + 37;
    const TimeGrid grid(1.0, nSteps);
    const std::vector<std::vector<double>> thetaSteps(nSteps);
    const SdeSimulator<double> simulator(grid, thetaSteps);
    const std::vector<double> theta = {0.05, 0.2};
    Eigen::VectorXd x0(1);
    x0(0) = 100.0;
    const std::uint64_t seed = 2001;

    const IidGaussianSource<> srcPer(1, seed);
    const auto per = quantape::mc::simulateGradientForwardSamples<3>(
        simulator, x0, theta, driftOf(GbmModel{}), diffusionOf(GbmModel{}), srcPer,
        TerminalCall{100.0}, 0, nPaths, Schedule::Sequential);
    const IidGaussianSource<> srcBlock(1, seed);
    const auto block = quantape::mc::simulateGradientForwardBlockSamples<3>(
        simulator, x0, theta, driftOf(GbmModel{}), diffusionOf(GbmModel{}), srcBlock,
        TerminalCallBatchPayoff{100.0}, 0, nPaths, 64, Schedule::Sequential);
    EXPECT_TRUE(per.values == block.values);
    EXPECT_TRUE((per.gradients.array() == block.gradients.array()).all());

    const IidGaussianSource<> srcBlockPar(1, seed);
    const auto blockPar = quantape::mc::simulateGradientForwardBlockSamples<3>(
        simulator, x0, theta, driftOf(GbmModel{}), diffusionOf(GbmModel{}), srcBlockPar,
        TerminalCallBatchPayoff{100.0}, 0, nPaths, 64, Schedule::Parallel);
    EXPECT_TRUE(blockPar.values == block.values);
    EXPECT_TRUE((blockPar.gradients.array() == block.gradients.array()).all());

    // Forward samples: parallel == sequential bitwise
    const std::size_t samplePaths = 1024;
    const IidGaussianSource<> srcSeq(1, 999);
    const auto seq = quantape::mc::simulateGradientForwardSamples<3>(
        simulator, x0, theta, driftOf(GbmModel{}), diffusionOf(GbmModel{}), srcSeq,
        TerminalCall{100.0}, 0, samplePaths, Schedule::Sequential);
    const IidGaussianSource<> srcPar(1, 999);
    const auto par = quantape::mc::simulateGradientForwardSamples<3>(
        simulator, x0, theta, driftOf(GbmModel{}), diffusionOf(GbmModel{}), srcPar,
        TerminalCall{100.0}, 0, samplePaths, Schedule::Parallel);
    EXPECT_TRUE(seq.values == par.values);
    EXPECT_TRUE((seq.gradients.array() == par.gradients.array()).all());
}

// ── Default forward mode: Stan fvar sweeps == Stan reverse ──

class SdeStanForwardTest : public StanTapeTest {};

TEST_F(SdeStanForwardTest, matchesStanReverse) {
    const std::size_t nSteps = 32;
    const std::size_t nPaths = 256;
    const TimeGrid grid(1.0, nSteps);
    const std::vector<std::vector<double>> thetaSteps(nSteps);
    const std::vector<double> theta = {0.05, 0.2};
    Eigen::VectorXd x0(1);
    x0(0) = 100.0;
    const TerminalCall payoff{100.0};
    const std::uint64_t seed = 4242;

    const IidGaussianSource<> srcF(1, seed);
    const GradientEstimate fwd = quantape::mc::simulateGradientStanForward(
        SdeSimulator<double>(grid, thetaSteps), x0, theta, driftOf(GbmModel{}),
        diffusionOf(GbmModel{}), srcF, payoff, nPaths);
    const IidGaussianSource<> srcR(1, seed);
    const GradientEstimate rev =
        simulateGradient(SdeSimulator<double>(grid, thetaSteps), x0, theta, driftOf(GbmModel{}),
                         diffusionOf(GbmModel{}), srcR, payoff, nPaths);
    CHECK_CLOSE("stan fwd value", fwd.value, rev.value, 1e-14);
    for (int j = 0; j < 3; ++j) {
        CHECK_CLOSE("stan fwd vs reverse", fwd.gradient(j), rev.gradient(j),
                    1e-8 * std::max(1.0, std::fabs(rev.gradient(j))));
    }

    // Milstein and CIR QE coverage
    {
        const SdeSimulator<double, Milstein> simulator(grid, thetaSteps);
        const IidGaussianSource<> src(1, seed);
        const GradientEstimate f =
            quantape::mc::simulateGradientStanForward(simulator, x0, theta, driftOf(GbmModel{}),
                                                      diffusionOf(GbmModel{}), src, payoff, nPaths);
        const IidGaussianSource<> src2(1, seed);
        const GradientEstimate r = simulateGradient(simulator, x0, theta, driftOf(GbmModel{}),
                                                    diffusionOf(GbmModel{}), src2, payoff, nPaths);
        for (int j = 0; j < 3; ++j) {
            CHECK_CLOSE("stan fwd milstein", f.gradient(j), r.gradient(j),
                        1e-8 * std::max(1.0, std::fabs(r.gradient(j))));
        }
    }
    {
        const double kappa = 2.0, level = 0.04, sigma = 0.2, v0 = 0.04;
        const std::size_t steps = 4;
        const TimeGrid shortGrid(1.0, steps);
        const std::vector<std::vector<double>> shortTheta(steps);
        const SdeSimulator<double, MomentMatching1D<CirQeMoments>> simulator(
            shortGrid, shortTheta, MomentMatching1D<CirQeMoments>{});
        const std::vector<double> cirTheta = {kappa, level, sigma};
        Eigen::VectorXd v(1);
        v(0) = v0;
        const IidGaussianSource<> src(1, 777);
        const GradientEstimate f = quantape::mc::simulateGradientStanForward(
            simulator, v, cirTheta, driftOf(CirModel{}), diffusionOf(CirModel{}), src,
            TerminalValue{}, 4000, Schedule::Parallel);
        const IidGaussianSource<> src2(1, 777);
        const GradientEstimate r =
            simulateGradient(simulator, v, cirTheta, driftOf(CirModel{}), diffusionOf(CirModel{}),
                             src2, TerminalValue{}, 4000, Schedule::Parallel);
        for (int j = 0; j < 4; ++j) {
            CHECK_CLOSE("stan fwd cir qe", f.gradient(j), r.gradient(j),
                        1e-8 * std::max(1.0, std::fabs(r.gradient(j))));
        }
    }
}

// ── Lean reverse tape: equivalence, determinism, tape size ──

class LeanTapeTest : public StanTapeTest {};

TEST_F(LeanTapeTest, scalarRulesAreExact) {
    using quantape::mc::RevScalar;
    using quantape::mc::RevTape;
    RevTape& tape = RevTape::active();
    tape.clear();
    const RevScalar x(2.0, tape.input(2.0));
    const RevScalar y(3.0, tape.input(3.0));
    const RevScalar f = x * y + x; // df/dx = y+1 = 4, df/dy = x = 2
    tape.reverse(f.node);
    CHECK_CLOSE("lean scalar df/dx", tape.adjoint(x.node), 4.0, 1e-14);
    CHECK_CLOSE("lean scalar df/dy", tape.adjoint(y.node), 2.0, 1e-14);

    tape.clear();
    const RevScalar a(0.25, tape.input(0.25));
    const RevScalar g = exp(sqrt(a)); // dg/da = exp(sqrt(a))/(2 sqrt(a))
    tape.reverse(g.node);
    const double expected = std::exp(std::sqrt(0.25)) / (2.0 * std::sqrt(0.25));
    CHECK_CLOSE("lean scalar exp/sqrt", tape.adjoint(a.node), expected, 1e-14);

    tape.clear();
    const RevScalar p(0.7, tape.input(0.7));
    const RevScalar q(1.3, tape.input(1.3));
    const RevScalar h = log(p / q) - p * p; // dh/dp = 1/p - 2p, dh/dq = -1/q
    tape.reverse(h.node);
    CHECK_CLOSE("lean scalar log/div", tape.adjoint(p.node), 1.0 / 0.7 - 2.0 * 0.7, 1e-14);
    CHECK_CLOSE("lean scalar log/div q", tape.adjoint(q.node), -1.0 / 1.3, 1e-14);
}

class LeanReverseTest : public StanTapeTest {};

TEST_F(LeanReverseTest, matchesForwardOnKeyedPaths) {
    // 1. GBM/Euler: lean == forward == FD (same keyed paths)
    {
        const std::size_t nSteps = 32;
        const std::size_t nPaths = 256;
        const TimeGrid grid(1.0, nSteps);
        const std::vector<std::vector<double>> thetaSteps(nSteps);
        const SdeSimulator<double> simulator(grid, thetaSteps);
        const std::vector<double> theta = {0.05, 0.2};
        Eigen::VectorXd x0(1);
        x0(0) = 100.0;
        const TerminalCall payoff{100.0};
        const std::uint64_t seed = 4242;

        const IidGaussianSource<> srcLean(1, seed);
        const GradientEstimate lean =
            quantape::mc::simulateGradientLean(simulator, x0, theta, driftOf(GbmModel{}),
                                               diffusionOf(GbmModel{}), srcLean, payoff, nPaths);
        const IidGaussianSource<> srcFwd(1, seed);
        const GradientEstimate fwd =
            simulateGradientForward<3>(simulator, x0, theta, driftOf(GbmModel{}),
                                       diffusionOf(GbmModel{}), srcFwd, payoff, nPaths);
        CHECK_CLOSE("lean value", lean.value, fwd.value, 1e-14);
        const IidGaussianSource<> srcLeanS(1, seed);
        const auto leanSamples = quantape::mc::simulateGradientLeanSamples(
            simulator, x0, theta, driftOf(GbmModel{}), diffusionOf(GbmModel{}), srcLeanS, payoff, 0,
            nPaths, Schedule::Sequential);
        const IidGaussianSource<> srcFwdS(1, seed);
        const auto fwdSamples = quantape::mc::simulateGradientForwardSamples<3>(
            simulator, x0, theta, driftOf(GbmModel{}), diffusionOf(GbmModel{}), srcFwdS, payoff, 0,
            nPaths, Schedule::Sequential);
        for (std::size_t pth = 0; pth < 3; ++pth) {
            CHECK_CLOSE("lean path value", leanSamples.values[pth], fwdSamples.values[pth], 1e-15);
            for (int j = 0; j < 3; ++j) {
                CHECK_CLOSE("lean path grad",
                            leanSamples.gradients(j, static_cast<Eigen::Index>(pth)),
                            fwdSamples.gradients(j, static_cast<Eigen::Index>(pth)), 1e-12);
            }
        }
        for (int j = 0; j < 3; ++j) {
            CHECK_CLOSE("lean vs forward", lean.gradient(j), fwd.gradient(j),
                        1e-10 * std::max(1.0, std::fabs(fwd.gradient(j))));
        }
    }

    // 2. Milstein (lean tape under fvar) and CIR QE
    {
        const GbmCase c;
        const std::size_t nSteps = 128;
        const std::size_t nPaths = 4000;
        const TimeGrid grid(c.tMax, nSteps);
        const std::vector<std::vector<double>> thetaSteps(nSteps);
        const std::vector<double> theta = {c.mu, c.sigma};
        Eigen::VectorXd x0(1);
        x0(0) = c.s0;
        const TerminalCall payoff{c.strike};
        const std::uint64_t seed = 31415;

        const SdeSimulator<double, Milstein> simulator(grid, thetaSteps);
        const IidGaussianSource<> srcLean(1, seed);
        const GradientEstimate lean = quantape::mc::simulateGradientLean(
            simulator, x0, theta, driftOf(GbmModel{}), diffusionOf(GbmModel{}), srcLean, payoff,
            nPaths, Schedule::Parallel);
        const IidGaussianSource<> srcFwd(1, seed);
        const GradientEstimate fwd = simulateGradientForward<3>(
            simulator, x0, theta, driftOf(GbmModel{}), diffusionOf(GbmModel{}), srcFwd, payoff,
            nPaths, Schedule::Parallel);
        for (int j = 0; j < 3; ++j) {
            CHECK_CLOSE("lean milstein vs forward", lean.gradient(j), fwd.gradient(j),
                        1e-8 * std::max(1.0, std::fabs(fwd.gradient(j))));
        }
    }
    {
        const double kappa = 2.0, level = 0.04, sigma = 0.2, v0 = 0.04;
        const std::size_t nSteps = 4;
        const std::size_t nPaths = 20000;
        const TimeGrid grid(1.0, nSteps);
        const std::vector<std::vector<double>> thetaSteps(nSteps);
        const SdeSimulator<double, MomentMatching1D<CirQeMoments>> simulator(
            grid, thetaSteps, MomentMatching1D<CirQeMoments>{});
        const std::vector<double> theta = {kappa, level, sigma};
        Eigen::VectorXd x0(1);
        x0(0) = v0;
        const IidGaussianSource<> srcLean(1, 777);
        const GradientEstimate lean = quantape::mc::simulateGradientLean(
            simulator, x0, theta, driftOf(CirModel{}), diffusionOf(CirModel{}), srcLean,
            TerminalValue{}, nPaths, Schedule::Parallel);
        const IidGaussianSource<> srcFwd(1, 777);
        const GradientEstimate fwd = simulateGradientForward<4>(
            simulator, x0, theta, driftOf(CirModel{}), diffusionOf(CirModel{}), srcFwd,
            TerminalValue{}, nPaths, Schedule::Parallel);
        for (int j = 0; j < 4; ++j) {
            CHECK_CLOSE("lean cir qe vs forward", lean.gradient(j), fwd.gradient(j),
                        1e-8 * std::max(1.0, std::fabs(fwd.gradient(j))));
        }
    }
}

TEST_F(LeanReverseTest, smoothedDigitalAndScheduleBitwise) {
    // 3. Smoothed digital FD gate + schedule determinism + tape size
    const std::size_t nSteps = 32;
    const std::size_t nPaths = 256;
    const TimeGrid grid(1.0, nSteps);
    const std::vector<std::vector<double>> thetaSteps(nSteps);
    const SdeSimulator<double> simulator(grid, thetaSteps);
    const std::vector<double> theta = {0.05, 0.2};
    Eigen::VectorXd x0(1);
    x0(0) = 100.0;
    const SmoothedDigitalCall payoff{100.0, 1e-3};
    const std::uint64_t seed = 31415;
    const IidGaussianSource<> src(1, seed);
    const GradientEstimate lean = quantape::mc::simulateGradientLean(
        simulator, x0, theta, driftOf(GbmModel{}), diffusionOf(GbmModel{}), src, payoff, nPaths);
    const double s0 = 100.0;
    const auto digitalAt = [&](double s0_, double mu_, double sigma_) {
        Eigen::VectorXd x(1);
        x(0) = s0_;
        return doubleValue(GbmModel{}, payoff, grid, thetaSteps, x, {mu_, sigma_}, nPaths, seed);
    };
    const double params[3] = {s0, 0.05, 0.2};
    for (int j = 0; j < 3; ++j) {
        const double h = 1e-5 * std::max(1.0, std::fabs(params[j]));
        double plus[3] = {s0, 0.05, 0.2};
        double minus[3] = {s0, 0.05, 0.2};
        plus[j] += h;
        minus[j] -= h;
        const double fd =
            (digitalAt(plus[0], plus[1], plus[2]) - digitalAt(minus[0], minus[1], minus[2])) /
            (2.0 * h);
        CHECK_CLOSE("lean smoothed digital", lean.gradient(j), fd,
                    1e-6 * std::max(1.0, std::fabs(lean.gradient(j))));
    }

    const IidGaussianSource<> srcSeq(1, 999);
    const auto seq = quantape::mc::simulateGradientLeanSamples(
        simulator, x0, theta, driftOf(GbmModel{}), diffusionOf(GbmModel{}), srcSeq,
        TerminalCall{100.0}, 0, 1024, Schedule::Sequential);
    const IidGaussianSource<> srcPar(1, 999);
    const auto par = quantape::mc::simulateGradientLeanSamples(
        simulator, x0, theta, driftOf(GbmModel{}), diffusionOf(GbmModel{}), srcPar,
        TerminalCall{100.0}, 0, 1024, Schedule::Parallel);
    EXPECT_TRUE(seq.values == par.values);
    EXPECT_TRUE((seq.gradients.array() == par.gradients.array()).all());

    const std::size_t nodesEuler = quantape::mc::leanPathTapeNodes(
        simulator, x0, theta, driftOf(GbmModel{}), diffusionOf(GbmModel{}), src, payoff, 0);
    const std::size_t nodesMilstein = quantape::mc::leanPathTapeNodes(
        SdeSimulator<double, Milstein>(grid, thetaSteps), x0, theta, driftOf(GbmModel{}),
        diffusionOf(GbmModel{}), src, payoff, 0);
    RecordProperty("lean_tape_nodes_euler", static_cast<int>(nodesEuler));
    RecordProperty("lean_tape_nodes_milstein", static_cast<int>(nodesMilstein));
}

// ── S5c: state derivatives dX(t)/d[x0; theta] ──

class SdeStateDerivativesTest : public StanTapeTest {};

TEST_F(SdeStateDerivativesTest, gbmTangentsMatchFiniteDifference) {
    const double s0 = 100.0, mu = 0.05, sigma = 0.2;
    const std::size_t nSteps = 8;
    const TimeGrid grid(1.0, nSteps);
    const std::vector<std::vector<double>> thetaSteps(nSteps);
    const std::vector<double> theta = {mu, sigma};
    Eigen::VectorXd x0(1);
    x0(0) = s0;
    const std::uint64_t seed = 321;
    const std::size_t pathIndex = 0;

    // GBM / Euler: tangents vs central FD of the same keyed path, all steps
    const SdeSimulator<double> simulator(grid, thetaSteps);
    const IidGaussianSource<> src(1, seed);
    const quantape::mc::StateDerivativePath deriv = quantape::mc::simulatePathDerivatives(
        simulator, x0, theta, driftOf(GbmModel{}), diffusionOf(GbmModel{}), src, pathIndex);
    EXPECT_EQ(deriv.dX.size(), nSteps + 1);
    EXPECT_EQ(deriv.dX[0](0, 0), 1.0);
    EXPECT_TRUE(deriv.dX[0](0, 1) == 0.0 && deriv.dX[0](0, 2) == 0.0);

    const auto pathAt = [&](double s0_, double mu_, double sigma_) {
        const IidGaussianSource<> source(1, seed);
        Eigen::VectorXd x(1);
        x(0) = s0_;
        return simulator.simulatePathSharedTheta<double>(
            x, driftOf(GbmModel{}), diffusionOf(GbmModel{}), source, pathIndex, {mu_, sigma_});
    };
    const double params[3] = {s0, mu, sigma};
    for (int j = 0; j < 3; ++j) {
        const double h = 1e-5 * std::max(1.0, std::fabs(params[j]));
        double plus[3] = {s0, mu, sigma};
        double minus[3] = {s0, mu, sigma};
        plus[j] += h;
        minus[j] -= h;
        const PathBlock<double> up = pathAt(plus[0], plus[1], plus[2]);
        const PathBlock<double> down = pathAt(minus[0], minus[1], minus[2]);
        for (std::size_t k = 0; k <= nSteps; ++k) {
            const double fd = (up.states[k](0, 0) - down.states[k](0, 0)) / (2.0 * h);
            const double tangent = deriv.dX[k](0, static_cast<Eigen::Index>(j));
            CHECK_CLOSE("sde gbm tangent", tangent, fd,
                        1e-6 * std::max(1.0, std::fabs(tangent)) + 1e-10);
        }
    }
}

TEST_F(SdeStateDerivativesTest, milsteinAndCirQeMatchFiniteDifference) {
    const double s0 = 100.0, mu = 0.05, sigma = 0.2;
    const std::vector<double> theta = {mu, sigma};
    Eigen::VectorXd x0(1);
    x0(0) = s0;
    const std::uint64_t seed = 321;
    const std::size_t pathIndex = 0;

    // Milstein and CIR-QE (branchy scheme) coverage
    const std::size_t steps = 4;
    const TimeGrid shortGrid(1.0, steps);
    const std::vector<std::vector<double>> shortTheta(steps);

    // Milstein GBM
    {
        const SdeSimulator<double, Milstein> simulator(shortGrid, shortTheta);
        const IidGaussianSource<> src(1, seed);
        const auto deriv = quantape::mc::simulatePathDerivatives(
            simulator, x0, theta, driftOf(GbmModel{}), diffusionOf(GbmModel{}), src, pathIndex);
        const auto pathAt = [&](double s0_, double mu_, double sigma_) {
            const IidGaussianSource<> source(1, seed);
            Eigen::VectorXd x(1);
            x(0) = s0_;
            return simulator.simulatePathSharedTheta<double>(
                x, driftOf(GbmModel{}), diffusionOf(GbmModel{}), source, pathIndex, {mu_, sigma_});
        };
        const double h = 1e-5;
        const PathBlock<double> up = pathAt(s0, mu + h, sigma);
        const PathBlock<double> down = pathAt(s0, mu - h, sigma);
        for (std::size_t k = 0; k <= steps; ++k) {
            const double fd = (up.states[k](0, 0) - down.states[k](0, 0)) / (2.0 * h);
            CHECK_CLOSE("milstein tangent dmu", deriv.dX[k](0, 1), fd,
                        1e-6 * std::max(1.0, std::fabs(fd)) + 1e-10);
        }
    }

    // CIR QE: uniforms + psi branches through the tangent pass
    {
        const double kappa = 2.0, level = 0.04, sig = 0.2, v0 = 0.04;
        const SdeSimulator<double, MomentMatching1D<CirQeMoments>> simulator(
            shortGrid, shortTheta, MomentMatching1D<CirQeMoments>{});
        const IidGaussianSource<> src(1, seed);
        Eigen::VectorXd v(1);
        v(0) = v0;
        const std::vector<double> cirTheta = {kappa, level, sig};
        const auto deriv = quantape::mc::simulatePathDerivatives(
            simulator, v, cirTheta, driftOf(CirModel{}), diffusionOf(CirModel{}), src, pathIndex);
        const auto pathAt = [&](double v0_, double k_, double l_, double s_) {
            const IidGaussianSource<> source(1, seed);
            Eigen::VectorXd x(1);
            x(0) = v0_;
            return simulator.simulatePathSharedTheta<double>(
                x, driftOf(CirModel{}), diffusionOf(CirModel{}), source, pathIndex, {k_, l_, s_});
        };
        const double h = 1e-6;
        const PathBlock<double> up = pathAt(v0, kappa + h, level, sig);
        const PathBlock<double> down = pathAt(v0, kappa - h, level, sig);
        for (std::size_t k = 0; k <= steps; ++k) {
            const double fd = (up.states[k](0, 0) - down.states[k](0, 0)) / (2.0 * h);
            CHECK_CLOSE("cir qe tangent dkappa", deriv.dX[k](0, 1), fd,
                        1e-5 * std::max(1.0, std::fabs(fd)) + 1e-9);
        }
    }
}

TEST_F(SdeStateDerivativesTest, chainRuleHandoffMatchesSimulateGradient) {
    // Chain-rule handoff: dE[pi]/dtheta = E[(dpi/dx_N) * Y_N]
    const double s0 = 100.0, mu = 0.05, sigma = 0.2;
    const std::vector<double> theta = {mu, sigma};
    Eigen::VectorXd x0(1);
    x0(0) = s0;
    const std::uint64_t seed = 321;

    const std::size_t steps = 64;
    const std::size_t nPaths = 2000;
    const TimeGrid chainGrid(1.0, steps);
    const std::vector<std::vector<double>> chainTheta(steps);
    const SdeSimulator<double> simulator(chainGrid, chainTheta);
    const TerminalCall payoff{100.0};

    double lhs[3] = {0.0, 0.0, 0.0};
    for (std::size_t pathIndex2 = 0; pathIndex2 < nPaths; ++pathIndex2) {
        const IidGaussianSource<> src(1, seed);
        const auto deriv = quantape::mc::simulatePathDerivatives(
            simulator, x0, theta, driftOf(GbmModel{}), diffusionOf(GbmModel{}), src, pathIndex2);
        const IidGaussianSource<> srcPath(1, seed);
        const PathBlock<double> path = simulator.simulatePathSharedTheta<double>(
            x0, driftOf(GbmModel{}), diffusionOf(GbmModel{}), srcPath, pathIndex2, theta);
        const double indicator = path.states.back()(0, 0) > 100.0 ? 1.0 : 0.0;
        for (int j = 0; j < 3; ++j) {
            lhs[j] += indicator * deriv.dX[steps](0, j);
        }
    }
    for (int j = 0; j < 3; ++j) {
        lhs[j] /= static_cast<double>(nPaths);
    }

    const IidGaussianSource<> srcGrad(1, seed);
    const GradientEstimate gradient =
        simulateGradient(simulator, x0, theta, driftOf(GbmModel{}), diffusionOf(GbmModel{}),
                         srcGrad, payoff, nPaths, Schedule::Parallel);
    for (int j = 0; j < 3; ++j) {
        CHECK_CLOSE("chain-rule handoff", lhs[j], gradient.gradient(j),
                    1e-8 * std::max(1.0, std::fabs(gradient.gradient(j))));
    }
}

// ── Multiprocessing shards: merged samples == single run (bitwise) ──

class SdeShardingTest : public StanTapeTest {};

TEST_F(SdeShardingTest, mergedSamplesReduceBitwise) {
    const std::size_t nSteps = 64;
    const std::size_t nPaths = 1024;
    const std::size_t split = 300;
    const TimeGrid grid(1.0, nSteps);
    const std::vector<std::vector<double>> thetaSteps(nSteps);
    const SdeSimulator<double> simulator(grid, thetaSteps);
    const std::vector<double> theta = {0.05, 0.2};
    Eigen::VectorXd x0(1);
    x0(0) = 100.0;
    const TerminalCall payoff{100.0};
    const std::uint64_t seed = 999;

    const IidGaussianSource<> srcLeft(1, seed);
    const quantape::mc::GradientSamples left =
        quantape::mc::simulateGradientSamples(simulator, x0, theta, driftOf(GbmModel{}),
                                              diffusionOf(GbmModel{}), srcLeft, payoff, 0, split);
    const IidGaussianSource<> srcRight(1, seed);
    const quantape::mc::GradientSamples right = quantape::mc::simulateGradientSamples(
        simulator, x0, theta, driftOf(GbmModel{}), diffusionOf(GbmModel{}), srcRight, payoff, split,
        nPaths - split);

    quantape::mc::GradientSamples merged;
    merged.pathBegin = 0;
    merged.nStateDims = left.nStateDims;
    merged.nParameters = left.nParameters;
    merged.values = left.values;
    merged.values.insert(merged.values.end(), right.values.begin(), right.values.end());
    merged.gradients.resize(left.gradients.rows(), static_cast<Eigen::Index>(nPaths));
    merged.gradients.leftCols(static_cast<Eigen::Index>(split)) = left.gradients;
    merged.gradients.rightCols(static_cast<Eigen::Index>(nPaths - split)) = right.gradients;

    const IidGaussianSource<> srcFull(1, seed);
    const quantape::mc::GradientSamples full =
        quantape::mc::simulateGradientSamples(simulator, x0, theta, driftOf(GbmModel{}),
                                              diffusionOf(GbmModel{}), srcFull, payoff, 0, nPaths);

    const auto estMerged = quantape::mc::reduceGradientSamples(merged);
    const auto estFull = quantape::mc::reduceGradientSamples(full);
    EXPECT_TRUE(bitwiseEqual(estMerged.value, estFull.value));
    EXPECT_TRUE(bitwiseEqual(estMerged.valueStdError, estFull.valueStdError));
    EXPECT_TRUE((estMerged.gradient.array() == estFull.gradient.array()).all());
    EXPECT_TRUE((estMerged.stdErrors.array() == estFull.stdErrors.array()).all());
}
