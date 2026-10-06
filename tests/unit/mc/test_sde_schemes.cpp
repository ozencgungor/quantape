// test_sde_schemes.cpp — SDE schemes (S2)
//
// Gates:
//   - Strong convergence on GBM (same Brownian path via the keyed source):
//     Euler order ~1/2, Milstein order ~1 (commutative noise)
//   - Milstein with constant diffusion == Euler (correction vanishes)
//   - Predictor-Corrector deterministic order 2 vs Euler 1 (cubic drift ODE)
//   - CIR positivity with a model-specific exact-variance scheme
//
// Stan-dependent (Milstein uses forward-mode fvar); include StanMath.h first.
#include "quantape/math/StanMath.h"

#include "quantape/mc/RandomSource.h"
#include "quantape/mc/Schemes.h"
#include "quantape/mc/SchemesStan.h"
#include "quantape/mc/SdePrimitives.h"
#include "quantape/mc/SdeSimulator.h"
#include "quantape/mc/TimeGrid.h"

#include <Eigen/Dense>

#include <cmath>
#include <cstdint>
#include <ostream>
#include <vector>

#include "support/GtestSupport.h"
#include "support/StanTapeFixture.h"

using quantape::util::isFiniteBitwise;

using quantape::mc::diffusionOf;
using quantape::mc::driftOf;
using quantape::mc::Euler;
using quantape::mc::IidGaussianSource;
using quantape::mc::Milstein;
using quantape::mc::PathBlock;
using quantape::mc::PredictorCorrector;
using quantape::mc::SdeSimulator;
using quantape::mc::StateMatrix;
using quantape::mc::TimeGrid;

namespace {

// Bitwise finiteness guard: under -ffast-math the compiler folds
// isnan/isfinite to constants, so NaN/Inf silently pass `<=` comparisons.
// Inspecting the exponent bits cannot be optimized away.

// ── Models ──

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

// dX = -X^3 dt (deterministic; sigma = 0) ; theta = {}
struct CubicDriftModel {
    template <typename Scalar>
    void drift(const StateMatrix<Scalar>& x, double, const std::vector<Scalar>&,
               StateMatrix<Scalar>& out) const {
        out = (-x.array() * x.array() * x.array()).matrix();
    }
    template <typename Scalar>
    void diffusion(const StateMatrix<Scalar>& x, double, const std::vector<Scalar>&, std::size_t,
                   StateMatrix<Scalar>& out) const {
        out.resize(x.rows(), x.cols());
        out.setConstant(Scalar(0.0));
    }
};

// ── Strong convergence on GBM (same Brownian path for scheme and exact) ──

struct GbmStrongResult {
    double error = 0.0;
};

template <typename Scheme>
GbmStrongResult gbmStrongError(std::size_t nSteps, std::size_t nPaths, double mu, double sigma,
                               double s0, double tMax, std::uint64_t seed) {
    const TimeGrid grid(tMax, nSteps);
    std::vector<std::vector<double>> theta(nSteps, {mu, sigma});
    const SdeSimulator<double, Scheme> simulator(grid, theta);
    const IidGaussianSource<> source(1, seed);

    const auto blocks = simulator.simulate(Eigen::VectorXd::Constant(1, s0), driftOf(GbmModel{}),
                                           diffusionOf(GbmModel{}), source, nPaths, 4096);

    // Exact terminal from the SAME keyed Brownian path:
    //   W_T = sum_k sqrt(dt_k) z_k,   S_T = s0 exp((mu - sigma^2/2) T + sigma W_T)
    IidGaussianSource<> noise(1, seed);
    Eigen::MatrixXd z;
    Eigen::VectorXd w = Eigen::VectorXd::Zero(static_cast<Eigen::Index>(nPaths));
    for (std::size_t k = 0; k < nSteps; ++k) {
        noise.fill(k, 0, nPaths, z);
        w.array() += std::sqrt(grid.dt(k)) * z.row(0).transpose().array();
    }
    Eigen::VectorXd exact(static_cast<Eigen::Index>(nPaths));
    for (Eigen::Index p = 0; p < exact.size(); ++p) {
        exact(p) = s0 * std::exp((mu - 0.5 * sigma * sigma) * tMax + sigma * w(p));
    }

    double sum = 0.0;
    std::size_t globalPath = 0;
    for (const PathBlock<double>& b : blocks) {
        const auto& x = b.states.back();
        for (Eigen::Index p = 0; p < x.cols(); ++p, ++globalPath) {
            sum += std::fabs(x(0, p) - exact(static_cast<Eigen::Index>(globalPath)));
        }
    }
    return {sum / static_cast<double>(nPaths)};
}

double strongOrder(std::size_t nSteps, const std::vector<double>& errors) {
    // consecutive ratios: dt halves, error ~ dt^order
    double orderSum = 0.0;
    std::size_t count = 0;
    for (std::size_t i = 1; i < errors.size(); ++i) {
        orderSum += std::log(errors[i - 1] / errors[i]) / std::log(2.0);
        ++count;
    }
    (void)nSteps;
    return orderSum / static_cast<double>(count);
}

// ── Predictor-Corrector deterministic order (cubic drift ODE) ──

template <typename Scheme>
double odeError(std::size_t nSteps) {
    const TimeGrid grid(1.0, nSteps);
    std::vector<std::vector<double>> theta(nSteps);
    const SdeSimulator<double, Scheme> simulator(grid, theta);
    const IidGaussianSource<> source(1, 0);
    const auto blocks =
        simulator.simulate(Eigen::VectorXd::Constant(1, 1.0), driftOf(CubicDriftModel{}),
                           diffusionOf(CubicDriftModel{}), source, 1, 1);
    const double xT = blocks[0].states.back()(0, 0);
    const double exact = 1.0 / std::sqrt(1.0 + 2.0 * 1.0); // x0/sqrt(1 + 2 x0^2 T)
    return std::fabs(xT - exact);
}

// ── CIR positivity with full-truncation Euler ──
//
// Euler on the variance can step slightly negative (then sqrt -> NaN), so
// the standard practical treatment is full truncation: clamp the variance
// at zero after every step. This is a model-specific scheme policy built
// on top of Euler (the extension point); variance-optimal
// positivity-preserving schemes (Andersen QE) arrive in S4.
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

struct CirFullTruncation {
    static constexpr std::size_t uniformStreams = 0;

    template <typename Scalar>
    using Scratch = typename Euler::template Scratch<Scalar>;

    template <typename Scalar, typename DF, typename GF>
        requires quantape::mc::Drift<DF, Scalar> && quantape::mc::Diffusion<GF, Scalar>
    void step(const StateMatrix<Scalar>& x, StateMatrix<Scalar>& xNext, double t, double dt,
              const Eigen::MatrixXd& z, const Eigen::MatrixXd& uniforms, const DF& drift,
              const GF& diffusion, const std::vector<Scalar>& theta,
              Scratch<Scalar>& scratch) const {
        Euler{}.step(x, xNext, t, dt, z, uniforms, drift, diffusion, theta, scratch);
        xNext = xNext.cwiseMax(Scalar(0.0)); // full truncation
    }
};

struct StrongOrderCase {
    const char* name;
    bool milstein;
    double lower;
    double upper;
};

void PrintTo(const StrongOrderCase& c, std::ostream* os) {
    *os << c.name;
}

} // namespace

class SdeSchemesTest : public StanTapeTest {};

class SdeStrongOrderTest : public StanTapeTest,
                           public ::testing::WithParamInterface<StrongOrderCase> {};

TEST_P(SdeStrongOrderTest, matchesExpectedOrder) {
    const double mu = 0.05, sigma = 0.3, s0 = 1.0, tMax = 1.0;
    const std::size_t nPaths = 40000;
    const std::vector<std::size_t> levels{16, 32, 64, 128};

    std::vector<double> errors;
    for (std::size_t n : levels) {
        if (GetParam().milstein) {
            errors.push_back(gbmStrongError<Milstein>(n, nPaths, mu, sigma, s0, tMax, 321).error);
        } else {
            errors.push_back(gbmStrongError<Euler>(n, nPaths, mu, sigma, s0, tMax, 321).error);
        }
    }
    const double order = strongOrder(levels.front(), errors);
    EXPECT_GT(order, GetParam().lower);
    EXPECT_LT(order, GetParam().upper);
}

INSTANTIATE_TEST_SUITE_P(Table, SdeStrongOrderTest,
                         ::testing::Values(StrongOrderCase{"Euler", false, 0.35, 0.65},
                                           StrongOrderCase{"Milstein", true, 0.80, 1.15}),
                         [](const ::testing::TestParamInfo<StrongOrderCase>& info) {
                             return info.param.name;
                         });

// Milstein correction vanishes for constant diffusion => bitwise Euler
TEST_F(SdeSchemesTest, milsteinConstantDiffusionIsBitwiseEuler) {
    const TimeGrid grid(1.0, 32);
    std::vector<std::vector<double>> theta(32, {1.5, 0.8, 0.4});
    const IidGaussianSource<> source(1, 11);
    const SdeSimulator<double, Euler> eulerSim(grid, theta);
    const SdeSimulator<double, Milstein> milsteinSim(grid, theta);

    const auto x0 = Eigen::VectorXd::Constant(1, 2.0);
    const auto a =
        eulerSim.simulate(x0, driftOf(OuModel{}), diffusionOf(OuModel{}), source, 64, 16);
    const auto b =
        milsteinSim.simulate(x0, driftOf(OuModel{}), diffusionOf(OuModel{}), source, 64, 16);
    double maxDiff = 0.0;
    for (std::size_t bi = 0; bi < a.size(); ++bi) {
        for (std::size_t k = 0; k < a[bi].states.size(); ++k) {
            maxDiff = std::max(maxDiff, (a[bi].states[k] - b[bi].states[k]).cwiseAbs().maxCoeff());
        }
    }
    EXPECT_EQ(maxDiff, 0.0);
}

TEST_F(SdeSchemesTest, predictorCorrectorSecondOrder) {
    const std::vector<std::size_t> levels{8, 16, 32, 64};
    double eulerOrderSum = 0.0, pcOrderSum = 0.0;
    std::vector<double> eulerErr, pcErr;
    for (std::size_t n : levels) {
        eulerErr.push_back(odeError<Euler>(n));
        pcErr.push_back(odeError<PredictorCorrector>(n));
    }
    for (std::size_t i = 1; i < levels.size(); ++i) {
        eulerOrderSum += std::log(eulerErr[i - 1] / eulerErr[i]) / std::log(2.0);
        pcOrderSum += std::log(pcErr[i - 1] / pcErr[i]) / std::log(2.0);
    }
    const double eulerOrder = eulerOrderSum / 3.0;
    const double pcOrder = pcOrderSum / 3.0;
    EXPECT_GT(eulerOrder, 0.85);
    EXPECT_LT(eulerOrder, 1.15);
    EXPECT_GT(pcOrder, 1.75);
    EXPECT_LT(pcOrder, 2.25);
}

TEST_F(SdeSchemesTest, cirFullTruncationPositivity) {
    const double kappa = 2.0, level = 0.04, sigma = 0.2, v0 = 0.04;
    const std::size_t nSteps = 52;
    const TimeGrid grid(1.0, nSteps);
    std::vector<std::vector<double>> theta(nSteps, {kappa, level, sigma});
    const SdeSimulator<double, CirFullTruncation> simulator(grid, theta);
    const IidGaussianSource<> source(1, 8);
    const std::size_t nPaths = 200000;
    const auto blocks = simulator.simulate(Eigen::VectorXd::Constant(1, v0), driftOf(CirModel{}),
                                           diffusionOf(CirModel{}), source, nPaths, 8192);
    double sum = 0.0;
    std::size_t negatives = 0;
    for (const auto& b : blocks) {
        const auto& x = b.states.back();
        for (Eigen::Index p = 0; p < x.cols(); ++p) {
            sum += x(0, p);
            if (x(0, p) < 0.0) {
                ++negatives;
            }
        }
    }
    const double mean = sum / static_cast<double>(nPaths);
    // Continuous CIR mean (exact for the untruncated recursion); truncation
    // adds a small positive bias (~3e-4 here), allowed for explicitly.
    const double exactMean = v0 * std::exp(-kappa * 1.0) + level * (1.0 - std::exp(-kappa * 1.0));
    EXPECT_EQ(negatives, 0u);
    CHECK_CLOSE("CIR full-truncation mean", mean, exactMean, 5.0 * 4.2e-5 + 3e-4);
}
