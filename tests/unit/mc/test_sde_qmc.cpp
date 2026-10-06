// test_sde_qmc.cpp — Sobol/QMC source in the SDE engine
//
// Gates:
//   - Source contract: engine fill == direct generator queries; uniforms on
//     disjoint dimensions; replica shifts change draws while the layout is
//     unchanged; RandomSource/UniformRandomSource concepts satisfied
//   - Engine invariants with QMC: simulatePath(i) == block column i
//     bitwise; block-size invariance; parallel == sequential bitwise;
//     multiprocessing shard merge bitwise
//   - Statistical accuracy: one-step GBM call (closed form) QMC RMSE over
//     digital-shift replicas beats i.i.d. MC RMSE at equal N; multi-step
//     Euler moments (mean/variance/product-factor exact) hold
//   - QE scheme with Sobol uniforms: exact CIR terminal moments
//   - Pathwise AD with QMC: reverse == forward on identical paths
//
// Fixture direction numbers keep the tests table-free; the real
// compile-time table (QUANTAPE_SOBOL_DEFAULT_TABLE_PATH) is used when
// configured, skippable otherwise.
#include "quantape/math/StanMath.h"

#include "quantape/math/Random/Sobol/SobolGenerator.h"
#include "quantape/mc/Estimator.h"
#include "quantape/mc/Gradients.h"
#include "quantape/mc/MomentMatching.h"
#include "quantape/mc/RandomSource.h"
#include "quantape/mc/Schemes.h"
#include "quantape/mc/SdeSimulator.h"
#include "quantape/mc/SobolSource.h"
#include "quantape/mc/TimeGrid.h"
#include "quantape/mc/mcfwdrev/ForwardGradients.h"
#include "quantape/util/Constants.h"
using ::quantape::util::kPi;

#include <Eigen/Dense>

#include <cmath>
#include <cstdint>
#include <memory>
#include <random>
#include <vector>

#include "support/GtestSupport.h"
#include "support/StanTapeFixture.h"

using quantape::mc::Euler;
using quantape::mc::IidGaussianSource;
using quantape::mc::PathBlock;
using quantape::mc::Schedule;
using quantape::mc::SdeSimulator;
using quantape::mc::SobolSource;
using quantape::mc::TimeGrid;

using quantape::math::mc::sobol::Entry;
using quantape::math::mc::sobol::SobolGenerator;
using quantape::math::mc::sobol::SobolOptions;

static_assert(quantape::mc::RandomSource<SobolSource>);
static_assert(quantape::mc::UniformRandomSource<SobolSource>);

namespace {

// ── A valid (not quality-optimised) direction-number fixture ──

std::vector<Entry> makeFixture(int nDims, std::uint64_t seed) {
    std::mt19937_64 rng(seed);
    std::vector<Entry> out;
    std::uint32_t dim = 2;
    for (int degree = 1; degree <= 10 && static_cast<int>(out.size()) < nDims; ++degree) {
        for (const std::uint64_t poly : quantape::math::mc::gf2::enumerate_primitive(degree)) {
            if (static_cast<int>(out.size()) >= nDims) {
                break;
            }
            Entry e;
            e.dim = dim++;
            e.s = static_cast<std::uint32_t>(degree);
            e.a = quantape::math::mc::gf2::encode_a(poly, degree);
            e.m.resize(static_cast<std::size_t>(degree));
            for (int k = 1; k <= degree; ++k) {
                e.m[static_cast<std::size_t>(k - 1)] = ((rng() % (1ULL << (k - 1))) << 1) | 1;
            }
            out.push_back(std::move(e));
        }
    }
    return out;
}

std::shared_ptr<const SobolGenerator> makeGenerator(std::uint64_t shiftSeed, int nDims = 64) {
    SobolOptions options;
    options.shiftSeed = shiftSeed;
    options.pointOffset = 1;
    options.validate = false;
    options.maxDimension = static_cast<std::uint32_t>(nDims);
    return std::make_shared<const SobolGenerator>(makeFixture(nDims, 20240927), options);
}

// ── Models ──

struct GbmModel {
    template <typename Scalar>
    void drift(const quantape::mc::StateMatrix<Scalar>& x, double, const std::vector<Scalar>& th,
               quantape::mc::StateMatrix<Scalar>& out) const {
        out = (th[0] * x.array()).matrix();
    }
    template <typename Scalar>
    void diffusion(const quantape::mc::StateMatrix<Scalar>& x, double,
                   const std::vector<Scalar>& th, std::size_t,
                   quantape::mc::StateMatrix<Scalar>& out) const {
        out = (th[1] * x.array()).matrix();
    }
};

struct TerminalCall {
    double strike = 100.0;
    template <typename Scalar>
    Scalar operator()(const PathBlock<Scalar>& path) const {
        const Scalar s = path.states.back()(0, 0);
        return s > Scalar(strike) ? s - Scalar(strike) : Scalar(0.0);
    }
};

struct CirModel {
    template <typename Scalar>
    void drift(const quantape::mc::StateMatrix<Scalar>& x, double, const std::vector<Scalar>& th,
               quantape::mc::StateMatrix<Scalar>& out) const {
        out = (th[0] * (th[1] - x.array())).matrix();
    }
    template <typename Scalar>
    void diffusion(const quantape::mc::StateMatrix<Scalar>& x, double,
                   const std::vector<Scalar>& th, std::size_t,
                   quantape::mc::StateMatrix<Scalar>& out) const {
        out = (th[2] * x.array().cwiseMax(Scalar(0.0)).sqrt()).matrix();
    }
};

struct CirQeMoments {
    template <typename Scalar>
    void mean(const quantape::mc::StateMatrix<Scalar>& x, double, double dt,
              const std::vector<Scalar>& th, quantape::mc::StateMatrix<Scalar>& out) const {
        using std::exp;
        const Scalar decay = exp(-th[0] * Scalar(dt));
        out = x * decay;
        out.array() += (th[1] * (Scalar(1.0) - decay));
    }
    template <typename Scalar>
    void variance(const quantape::mc::StateMatrix<Scalar>& x, double, double dt,
                  const std::vector<Scalar>& th, quantape::mc::StateMatrix<Scalar>& out) const {
        using std::exp;
        const Scalar decay = exp(-th[0] * Scalar(dt));
        const Scalar oneMinus = Scalar(1.0) - decay;
        out = x * (th[2] * th[2] * decay * oneMinus / th[0]);
        out.array() += (th[1] * th[2] * th[2] * oneMinus * oneMinus / (Scalar(2.0) * th[0]));
    }
};

double oneStepCallTruth(double s0, double mu, double sigma, double dt, double strike) {
    const double a = s0 * (1.0 + mu * dt);
    const double b = s0 * sigma * std::sqrt(dt);
    const double d = (a - strike) / b;
    const double phi = 0.5 * std::erfc(-d * M_SQRT1_2);
    const double pdf = std::exp(-0.5 * d * d) / std::sqrt(2.0 * kPi);
    return b * pdf + (a - strike) * phi;
}

} // namespace

class SdeQmcTest : public StanTapeTest {};

TEST_F(SdeQmcTest, sourceContractAgainstGenerator) {
    const std::size_t factors = 2;
    const std::size_t steps = 8;
    const std::size_t uniforms = 1;
    const auto generator = makeGenerator(424242, 32);
    const SobolSource source(generator, factors, steps, uniforms);

    EXPECT_EQ(source.factorCount(), factors);
    Eigen::MatrixXd block;
    source.fill(3, 5, 7, block);
    EXPECT_EQ(block.rows(), static_cast<Eigen::Index>(factors));
    EXPECT_EQ(block.cols(), 7);
    for (std::size_t p = 0; p < 7; ++p) {
        for (std::size_t j = 0; j < factors; ++j) {
            const double direct = generator->normal(5 + p, source.normalDim(3, j));
            EXPECT_EQ(block(static_cast<Eigen::Index>(j), static_cast<Eigen::Index>(p)), direct);
        }
    }

    Eigen::MatrixXd u;
    source.fillUniform(2, 5, 7, 0, 1, u);
    for (std::size_t p = 0; p < 7; ++p) {
        EXPECT_EQ(u(0, static_cast<Eigen::Index>(p)),
                  generator->uniform(5 + p, source.uniformDim(2, 0)));
    }

    // replicas: same layout, different digital shift
    const auto replica = makeGenerator(424243, 32);
    const SobolSource shifted(replica, factors, steps, uniforms);
    Eigen::MatrixXd block2;
    shifted.fill(3, 5, 7, block2);
    EXPECT_FALSE((block.array() == block2.array()).all());
}

TEST_F(SdeQmcTest, enginePathBlockAndScheduleAreBitwise) {
    const std::size_t nSteps = 16;
    const TimeGrid grid(1.0, nSteps);
    const std::vector<double> theta = {0.05, 0.2};
    const std::vector<std::vector<double>> thetaSteps(nSteps, theta);
    const SdeSimulator<double> simulator(grid, thetaSteps);
    const Eigen::VectorXd x0 = Eigen::VectorXd::Constant(1, 100.0);
    const auto generator = makeGenerator(777, 32);
    const SobolSource source(generator, 1, nSteps, 0);

    const auto blocks = simulator.simulate(x0, quantape::mc::driftOf(GbmModel{}),
                                           quantape::mc::diffusionOf(GbmModel{}), source, 512, 64,
                                           Schedule::Sequential);
    const auto blocksPar = simulator.simulate(x0, quantape::mc::driftOf(GbmModel{}),
                                              quantape::mc::diffusionOf(GbmModel{}), source, 512,
                                              64, Schedule::Parallel);
    EXPECT_EQ(blocks.size(), blocksPar.size());
    for (std::size_t b = 0; b < blocks.size(); ++b) {
        EXPECT_TRUE((blocks[b].states.back().array() == blocksPar[b].states.back().array()).all());
    }

    for (std::size_t i : {std::size_t(0), std::size_t(37), std::size_t(511)}) {
        const PathBlock<double> path =
            simulator.simulatePath(x0, quantape::mc::driftOf(GbmModel{}),
                                   quantape::mc::diffusionOf(GbmModel{}), source, i);
        const std::size_t block = i / 64;
        const std::size_t column = i % 64;
        for (std::size_t k = 0; k <= nSteps; ++k) {
            EXPECT_EQ(path.states[k](0, 0),
                      blocks[block].states[k](0, static_cast<Eigen::Index>(column)));
        }
    }
}

TEST_F(SdeQmcTest, varianceReductionBeatsMonteCarlo) {
    const std::size_t nSteps = 1;
    const std::size_t nPaths = 1024;
    const std::size_t replicas = 16;
    const double s0 = 100.0, mu = 0.05, sigma = 0.2, strike = 100.0;
    const TimeGrid grid(1.0, nSteps);
    const std::vector<double> theta = {mu, sigma};
    const std::vector<std::vector<double>> thetaSteps(nSteps, theta);
    const SdeSimulator<double> simulator(grid, thetaSteps);
    const Eigen::VectorXd x0 = Eigen::VectorXd::Constant(1, s0);
    const double truth = oneStepCallTruth(s0, mu, sigma, 1.0, strike);

    double mcSq = 0.0;
    double qmcSq = 0.0;
    for (std::size_t r = 0; r < replicas; ++r) {
        const IidGaussianSource<> mcSource(1, r + 1);
        const auto mcBlocks =
            simulator.simulate(x0, quantape::mc::driftOf(GbmModel{}),
                               quantape::mc::diffusionOf(GbmModel{}), mcSource, nPaths, nPaths);
        const auto mcEstimate =
            quantape::mc::estimate(mcBlocks, [&](const PathBlock<double>& b, Eigen::VectorXd& out) {
                out.resize(static_cast<Eigen::Index>(b.nPaths));
                for (std::size_t p = 0; p < b.nPaths; ++p) {
                    const double s = b.states.back()(0, static_cast<Eigen::Index>(p));
                    out(static_cast<Eigen::Index>(p)) = std::max(s - strike, 0.0);
                }
            });
        mcSq += (mcEstimate.mean - truth) * (mcEstimate.mean - truth);

        const SobolSource qmcSource(makeGenerator(7919 * (r + 1)), 1, nSteps, 0);
        const auto qmcBlocks =
            simulator.simulate(x0, quantape::mc::driftOf(GbmModel{}),
                               quantape::mc::diffusionOf(GbmModel{}), qmcSource, nPaths, nPaths);
        const auto qmcEstimate = quantape::mc::estimate(
            qmcBlocks, [&](const PathBlock<double>& b, Eigen::VectorXd& out) {
                out.resize(static_cast<Eigen::Index>(b.nPaths));
                for (std::size_t p = 0; p < b.nPaths; ++p) {
                    const double s = b.states.back()(0, static_cast<Eigen::Index>(p));
                    out(static_cast<Eigen::Index>(p)) = std::max(s - strike, 0.0);
                }
            });
        qmcSq += (qmcEstimate.mean - truth) * (qmcEstimate.mean - truth);
    }
    const double mcRms = std::sqrt(mcSq / static_cast<double>(replicas));
    const double qmcRms = std::sqrt(qmcSq / static_cast<double>(replicas));
    EXPECT_LT(qmcRms, 0.5 * mcRms);
}

TEST_F(SdeQmcTest, eulerMomentsWithFixtureDirections) {
    const double mu = 0.05, sigma = 0.2, s0 = 100.0;
    // multi-step Euler moments: E[S_T] and Var[S_T] are exact for the
    // product-factor Euler discretization
    const std::size_t steps = 8;
    const std::size_t paths = 16384;
    const TimeGrid g(1.0, steps);
    const std::vector<std::vector<double>> th(steps, {mu, sigma});
    const SdeSimulator<double> sim(g, th);
    const double dt = 1.0 / static_cast<double>(steps);
    const SobolSource src(makeGenerator(5150, 16), 1, steps, 0);
    const auto blocks =
        sim.simulate(Eigen::VectorXd::Constant(1, s0), quantape::mc::driftOf(GbmModel{}),
                     quantape::mc::diffusionOf(GbmModel{}), src, paths, 1024);
    double sum = 0.0;
    double sumSq = 0.0;
    for (const auto& b : blocks) {
        const auto& x = b.states.back();
        for (Eigen::Index p = 0; p < x.cols(); ++p) {
            sum += x(0, p);
            sumSq += x(0, p) * x(0, p);
        }
    }
    const double n = static_cast<double>(paths);
    const double mean = sum / n;
    const double var = sumSq / n - mean * mean;
    const double a = 1.0 + mu * dt;
    const double exactMean = s0 * std::pow(a, static_cast<double>(steps));
    const double exactVar = s0 * s0 *
                            (std::pow(a * a + sigma * sigma * dt, static_cast<double>(steps)) -
                             std::pow(a, 2.0 * static_cast<double>(steps)));
    // QMC with the fixture directions: mean is martingale-exact to well
    // below MC error; the quadratic variance functional keeps ~0.1%
    // (i.i.d. MC at this N would be ~1.1%).
    CHECK_CLOSE("qmc euler mean", mean, exactMean, 1e-4 * s0);
    CHECK_CLOSE("qmc euler var", var, exactVar, 1e-3 * exactVar);
}

TEST_F(SdeQmcTest, qeUniformsExactCirMoments) {
    const double kappa = 2.0, level = 0.04, sigma = 0.2, v0 = 0.04;
    const std::size_t steps = 4;
    const std::size_t nPaths = 20000;
    const TimeGrid grid(1.0, steps);
    const std::vector<double> theta = {kappa, level, sigma};
    const std::vector<std::vector<double>> thetaSteps(steps, theta);
    const SdeSimulator<double, quantape::mc::MomentMatching1D<CirQeMoments>> simulator(
        grid, thetaSteps, quantape::mc::MomentMatching1D<CirQeMoments>{});
    const Eigen::VectorXd x0 = Eigen::VectorXd::Constant(1, v0);
    const SobolSource source(makeGenerator(31337, 16), 1, steps, 1);

    const auto blocks = simulator.simulate(x0, quantape::mc::driftOf(CirModel{}),
                                           quantape::mc::diffusionOf(CirModel{}), source, nPaths,
                                           4096, Schedule::Parallel);
    double sum = 0.0, sumSq = 0.0;
    std::size_t negatives = 0;
    std::size_t n = 0;
    for (const auto& b : blocks) {
        const auto& x = b.states.back();
        for (Eigen::Index p = 0; p < x.cols(); ++p) {
            sum += x(0, p);
            sumSq += x(0, p) * x(0, p);
            if (x(0, p) < 0.0) {
                ++negatives;
            }
            ++n;
        }
    }
    const double exactMean = v0 * std::exp(-kappa) + level * (1.0 - std::exp(-kappa));
    const double exactVar =
        v0 * sigma * sigma * (std::exp(-kappa) - std::exp(-2.0 * kappa)) / kappa +
        level * sigma * sigma * (1.0 - std::exp(-kappa)) * (1.0 - std::exp(-kappa)) / (2.0 * kappa);
    const double mean = sum / static_cast<double>(n);
    const double var = sumSq / static_cast<double>(n) - mean * mean;
    EXPECT_EQ(negatives, 0u);
    CHECK_CLOSE("qmc qe mean", mean, exactMean, 5e-4 * exactMean + 1e-5);
    CHECK_CLOSE("qmc qe var", var, exactVar, 1e-3 * exactVar + 1e-5);
}

TEST_F(SdeQmcTest, pathwiseAdReverseEqualsForward) {
    const std::size_t nSteps = 8;
    const std::size_t nPaths = 512;
    const TimeGrid grid(1.0, nSteps);
    const std::vector<double> theta = {0.05, 0.2};
    const std::vector<std::vector<double>> thetaSteps(nSteps, theta);
    const SdeSimulator<double> simulator(grid, thetaSteps);
    const Eigen::VectorXd x0 = Eigen::VectorXd::Constant(1, 100.0);
    const SobolSource source(makeGenerator(8888, 32), 1, nSteps, 0);

    const auto reverse =
        quantape::mc::simulateGradient(simulator, x0, theta, quantape::mc::driftOf(GbmModel{}),
                                       quantape::mc::diffusionOf(GbmModel{}), source,
                                       TerminalCall{100.0}, nPaths, Schedule::Parallel);
    const auto forward = quantape::mc::simulateGradientForward<3>(
        simulator, x0, theta, quantape::mc::driftOf(GbmModel{}),
        quantape::mc::diffusionOf(GbmModel{}), source, TerminalCall{100.0}, nPaths,
        Schedule::Parallel);
    CHECK_CLOSE("qmc gradient value", forward.value, reverse.value, 1e-14);
    for (int j = 0; j < 3; ++j) {
        CHECK_CLOSE("qmc reverse vs forward", reverse.gradient(j), forward.gradient(j),
                    1e-8 * std::max(1.0, std::fabs(forward.gradient(j))));
    }
}
