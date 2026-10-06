// test_sde_processes.cpp — S3: process bundles + Heston QE
//
// Gates:
//   - Process drift/diffusion: GBM/OU/CIR reproduce the discrete moment
//     recursions through the bundles (not inline models)
//   - CIR bundle with MomentMatching1D: exact terminal moments (as S4, but
//     through the process bundle)
//   - Heston QE: variance terminal moments equal the exact CIR moments;
//     no negatives/NaN in the Feller-violating regime; call prices
//     consistent with a fine-grid Euler reference
//
// Stan-free.
#include "quantape/mc/MomentMatching.h"
#include "quantape/mc/RandomSource.h"
#include "quantape/mc/Schemes.h"
#include "quantape/mc/SdePrimitives.h"
#include "quantape/mc/SdeSimulator.h"
#include "quantape/mc/TimeGrid.h"
#include "quantape/mc/processes/HestonQeProcess.h"
#include "quantape/mc/processes/SdeProcesses.h"

#include <Eigen/Dense>

#include <cmath>
#include <cstdint>
#include <vector>

#include "support/GtestSupport.h"

using quantape::util::isFiniteBitwise;

using quantape::mc::diffusionOf;
using quantape::mc::driftOf;
using quantape::mc::Euler;
using quantape::mc::IidGaussianSource;
using quantape::mc::MomentMatching1D;
using quantape::mc::SdeSimulator;
using quantape::mc::TimeGrid;
using quantape::processes::CirProcess;
using quantape::processes::GbmProcess;
using quantape::processes::HestonProcess;
using quantape::processes::HestonQeProcess;

namespace {

// ── Heston QE helper ──

struct HestonStats {
    double vMean = 0.0;
    double vVar = 0.0;
    double callPrice = 0.0;
    double callStdError = 0.0;
    std::size_t negatives = 0;
    std::size_t nonFinite = 0;
};

template <typename Scheme>
HestonStats runHeston(const Scheme& scheme, const HestonProcess& model, double strike,
                      std::size_t nSteps, std::size_t nPaths, std::uint64_t seed) {
    const double s0 = 100.0, v0 = 0.04, tMax = 1.0;
    const Eigen::VectorXd x0 = (Eigen::Vector2d() << std::log(s0), v0).finished();
    const TimeGrid grid(tMax, nSteps);
    std::vector<std::vector<double>> theta(nSteps);
    const SdeSimulator<double, Scheme> simulator(grid, theta, scheme);
    const IidGaussianSource<> source(2, seed);
    const auto blocks =
        simulator.simulate(x0, driftOf(model), diffusionOf(model), source, nPaths, 8192);
    HestonStats stats;
    double vSum = 0.0, vSumSq = 0.0, callSum = 0.0, callSumSq = 0.0;
    std::size_t n = 0;
    for (const auto& b : blocks) {
        const auto& x = b.states.back();
        for (Eigen::Index p = 0; p < x.cols(); ++p) {
            const double v = x(1, p);
            if (!isFiniteBitwise(v) || !isFiniteBitwise(x(0, p))) {
                ++stats.nonFinite;
                continue;
            }
            if (v < 0.0) {
                ++stats.negatives;
            }
            const double price = std::exp(x(0, p));
            const double payoff = std::max(price - strike, 0.0);
            vSum += v;
            vSumSq += v * v;
            callSum += payoff;
            callSumSq += payoff * payoff;
            ++n;
        }
    }
    stats.vMean = vSum / static_cast<double>(n);
    stats.vVar = vSumSq / static_cast<double>(n) - stats.vMean * stats.vMean;
    stats.callPrice = callSum / static_cast<double>(n);
    const double callVar = callSumSq / static_cast<double>(n) - stats.callPrice * stats.callPrice;
    stats.callStdError = std::sqrt(callVar / static_cast<double>(n));
    return stats;
}

} // namespace

TEST(SdeProcessGbm, bundleEulerMeanAndCall) {
    const GbmProcess model{0.05, 0.2};
    const std::size_t nSteps = 250;
    const std::size_t nPaths = 200000;
    const double s0 = 100.0, tMax = 1.0;
    const TimeGrid grid(tMax, nSteps);
    std::vector<std::vector<double>> theta(nSteps); // bundle holds its params
    const SdeSimulator<double> simulator(grid, theta);
    const IidGaussianSource<> source(1, 31);
    const auto blocks = simulator.simulate(Eigen::VectorXd::Constant(1, s0), driftOf(model),
                                           diffusionOf(model), source, nPaths, 8192);
    double mean = 0.0;
    double call = 0.0;
    std::size_t n = 0;
    for (const auto& b : blocks) {
        const auto& x = b.states.back();
        for (Eigen::Index p = 0; p < x.cols(); ++p) {
            mean += x(0, p);
            call += std::max(x(0, p) - s0, 0.0);
        }
        n += b.nPaths;
    }
    mean /= static_cast<double>(n);
    call /= static_cast<double>(n);
    double eulerMean = s0;
    for (std::size_t k = 0; k < nSteps; ++k) {
        eulerMean *= (1.0 + model.mu * grid.dt(k));
    }
    CHECK_CLOSE("GBM bundle Euler mean", mean, eulerMean,
                5.0 * eulerMean * model.sigma / std::sqrt(static_cast<double>(n)));
    // undiscounted call vs e^{mu T} BS(mu)
    const double k = s0;
    const double d1 = (std::log(s0 / k) + (model.mu + 0.5 * model.sigma * model.sigma) * tMax) /
                      (model.sigma * std::sqrt(tMax));
    const double d2 = d1 - model.sigma * std::sqrt(tMax);
    const double callRef = std::exp(model.mu * tMax) *
                           (s0 * 0.5 * std::erfc(-d1 * M_SQRT1_2) -
                            k * std::exp(-model.mu * tMax) * 0.5 * std::erfc(-d2 * M_SQRT1_2));
    CHECK_CLOSE("GBM bundle call", call, callRef, 0.01 * callRef);
}

TEST(SdeProcessCir, bundleQeMomentsExact) {
    const CirProcess model{2.0, 0.04, 0.2};
    const std::size_t nSteps = 52;
    const std::size_t nPaths = 400000;
    const double v0 = 0.04;
    const TimeGrid grid(1.0, nSteps);
    std::vector<std::vector<double>> theta(nSteps);
    const SdeSimulator<double, MomentMatching1D<CirProcess>> simulator(grid, theta);
    const IidGaussianSource<> source(1, 55);
    const auto blocks = simulator.simulate(Eigen::VectorXd::Constant(1, v0), driftOf(model),
                                           diffusionOf(model), source, nPaths, 8192);
    const double exactMean =
        v0 * std::exp(-model.kappa) + model.level * (1.0 - std::exp(-model.kappa));
    const double exactVar =
        v0 * model.sigma * model.sigma * (std::exp(-model.kappa) - std::exp(-2.0 * model.kappa)) /
            model.kappa +
        model.level * model.sigma * model.sigma * (1.0 - std::exp(-model.kappa)) *
            (1.0 - std::exp(-model.kappa)) / (2.0 * model.kappa);
    double sum = 0.0;
    double sumSq = 0.0;
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
    const double mean = sum / static_cast<double>(nPaths);
    const double var = sumSq / static_cast<double>(nPaths) - mean * mean;
    CHECK_CLOSE("CIR bundle QE mean", mean, exactMean,
                5.0 * std::sqrt(exactVar / static_cast<double>(nPaths)));
    CHECK_CLOSE("CIR bundle QE var", var, exactVar,
                6.0 * exactVar * std::sqrt(2.0 / static_cast<double>(nPaths)));
    EXPECT_EQ(negatives, 0u);
}

TEST(SdeProcessHeston, qeVarianceMomentsExact) {
    const std::size_t nPaths = 200000;
    const double v0 = 0.04;
    const HestonProcess model{/*mu=*/0.02, /*kappa=*/2.0, /*level=*/0.04, /*eta=*/0.3,
                              /*rho=*/-0.7};

    // Variance moments are the exact CIR conditional-moment result
    const HestonStats qe =
        runHeston(HestonQeProcess<HestonProcess>{model}, model, 100.0, 64, nPaths, 606);
    const double exactMean =
        v0 * std::exp(-model.kappa) + model.level * (1.0 - std::exp(-model.kappa));
    const double exactVar = v0 * model.eta * model.eta *
                                (std::exp(-model.kappa) - std::exp(-2.0 * model.kappa)) /
                                model.kappa +
                            model.level * model.eta * model.eta * (1.0 - std::exp(-model.kappa)) *
                                (1.0 - std::exp(-model.kappa)) / (2.0 * model.kappa);
    CHECK_CLOSE("Heston QE V mean", qe.vMean, exactMean,
                5.0 * std::sqrt(exactVar / static_cast<double>(nPaths)));
    CHECK_CLOSE("Heston QE V var", qe.vVar, exactVar,
                8.0 * exactVar * std::sqrt(2.0 / static_cast<double>(nPaths)));
    EXPECT_TRUE(qe.negatives == 0 && qe.nonFinite == 0);
}

TEST(SdeProcessHeston, callMatchesFineEulerReference) {
    const std::size_t nPaths = 200000;
    const HestonProcess model{/*mu=*/0.02, /*kappa=*/2.0, /*level=*/0.04, /*eta=*/0.3,
                              /*rho=*/-0.7};

    // Call price: QE (N=64) vs a fine-grid Euler reference (N=2048)
    const HestonStats qe =
        runHeston(HestonQeProcess<HestonProcess>{model}, model, 100.0, 64, nPaths, 606);
    const HestonStats reference = runHeston(Euler{}, model, 100.0, 2048, nPaths, 606);
    const HestonStats coarseEuler = runHeston(Euler{}, model, 100.0, 64, nPaths, 606);
    const double qeErr = std::fabs(qe.callPrice - reference.callPrice);
    const double eulerErr = std::fabs(coarseEuler.callPrice - reference.callPrice);
    EXPECT_LE(qeErr, 4.0 * qe.callStdError + 4.0 * reference.callStdError + 0.01);
    EXPECT_LE(qeErr, eulerErr + 4.0 * qe.callStdError);
}

TEST(SdeProcessHeston, deepOtmAndFellerViolatingSafe) {
    const std::size_t nPaths = 200000;
    const HestonProcess model{/*mu=*/0.02, /*kappa=*/2.0, /*level=*/0.04, /*eta=*/0.3,
                              /*rho=*/-0.7};

    // Deep OTM
    const HestonStats qeOtm =
        runHeston(HestonQeProcess<HestonProcess>{model}, model, 140.0, 64, nPaths, 707);
    const HestonStats refOtm = runHeston(Euler{}, model, 140.0, 2048, nPaths, 707);
    EXPECT_LE(std::fabs(qeOtm.callPrice - refOtm.callPrice),
              4.0 * qeOtm.callStdError + 4.0 * refOtm.callStdError + 0.005);

    // Feller-violating regime: 2 kappa level < eta^2 (0.08 < 0.36)
    const HestonProcess violating{/*mu=*/0.02, /*kappa=*/1.0, /*level=*/0.04, /*eta=*/0.6,
                                  /*rho=*/-0.7};
    const HestonStats hard =
        runHeston(HestonQeProcess<HestonProcess>{violating}, violating, 100.0, 64, nPaths, 808);
    EXPECT_TRUE(hard.negatives == 0 && hard.nonFinite == 0);
}
