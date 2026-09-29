// test_heston_mc.cpp — H5: theta-driven Heston QE (prices + pathwise gradients)
//
// The Heston bundle and the QE scheme read their coefficients from the
// engine's per-interval `theta = {mu, kappa, level, eta, rho}` (v0 travels in
// the initial state), which makes the parameters AD leaves for pathwise
// gradients. Gates:
//
//   1. theta-driven == member-driven paths (bitwise, QE and Euler)
//   2. catalog: QE + Sobol QMC call prices vs the analytic pricer
//   3. pathwise QE gradients (`simulateGradient`) vs the analytic full
//      gradient: [d/d lnS0, d/d v0, d/d mu, d/d kappa, d/d theta, d/d sigma,
//      d/d rho], plus an engine-FD cross-check

#include "quantape/math/StanMath.h"

#include "quantape/calibration/CalibrationProblem.h"
#include "quantape/calibration/HestonCalibration.h"
#include "quantape/math/Integrals/DoubleExponentialIntegrator.h"
#include "quantape/math/Optimization/LBFGS.h"
#include "quantape/mc/Gradients.h"
#include "quantape/mc/RandomSource.h"
#include "quantape/mc/SobolSource.h"
#include "quantape/mc/TimeGrid.h"
#include "quantape/mc/processes/HestonQeProcess.h"
#include "quantape/mc/processes/SdeProcesses.h"
#include "quantape/models/HestonModel.h"

#include <Eigen/Dense>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <type_traits>
#include <vector>

#include "TestSupport.h"

using quantape::mc::diffusionOf;
using quantape::mc::driftOf;
using quantape::mc::Euler;
using quantape::mc::GradientEstimate;
using quantape::mc::IidGaussianSource;
using quantape::mc::PathBlock;
using quantape::mc::Schedule;
using quantape::mc::SdeSimulator;
using quantape::mc::simulateGradient;
using quantape::mc::SobolSource;
using quantape::mc::TimeGrid;
using quantape::processes::HestonProcess;
using quantape::processes::HestonQeProcess;

namespace {

bool isFiniteBitwise(double x) {
    std::uint64_t bits = 0;
    std::memcpy(&bits, &x, sizeof(double));
    return ((bits >> 52) & 0x7FFULL) != 0x7FFULL;
}

bool bitwiseEqual(double a, double b) {
    std::uint64_t ba = 0;
    std::uint64_t bb = 0;
    std::memcpy(&ba, &a, sizeof(double));
    std::memcpy(&bb, &b, sizeof(double));
    return ba == bb;
}

void checkClose(const char* label, double got, double expected, double tol) {
    if (!isFiniteBitwise(got) || !isFiniteBitwise(expected) ||
        !(std::fabs(got - expected) <= tol)) {
        QTA_LOG_ERROR("quantape.test", "FAIL: {} got={} expected={} err={} tol={}", label,
                      quantape_test::num(got, 12), quantape_test::num(expected, 12),
                      quantape_test::num(std::fabs(got - expected), 3), quantape_test::num(tol, 3));
        std::exit(1);
    }
}

/// Terminal call on the Heston (lnS, V) state.
struct HestonCallPayoff {
    double strike = 1.0;
    template <typename Scalar>
    Scalar operator()(const PathBlock<Scalar>& path) const {
        using std::exp;
        const Scalar s = exp(path.states.back()(0, 0));
        return s > Scalar(strike) ? s - Scalar(strike) : Scalar(0.0);
    }
};

Eigen::VectorXd owning(const std::vector<double>& v) {
    return Eigen::Map<const Eigen::VectorXd>(v.data(), static_cast<Eigen::Index>(v.size()));
}

std::vector<double> thetaOf(const HestonProcess& p) {
    return {p.mu, p.kappa, p.level, p.eta, p.rho};
}

// ── 1. theta-driven == member-driven, bitwise ──

void testThetaEquivalence() {
    const HestonProcess model{0.02, 2.0, 0.04, 0.5, -0.6};
    const Eigen::VectorXd x0 = (Eigen::Vector2d() << 0.0, 0.04).finished();
    const std::size_t nSteps = 16;
    const std::size_t nPaths = 512;
    const TimeGrid grid(1.0, nSteps);

    const std::vector<std::vector<double>> emptyTheta(nSteps);
    const std::vector<std::vector<double>> filledTheta(nSteps, thetaOf(model));

    auto terminal = [&](const std::vector<std::vector<double>>& theta, auto schemeTag) {
        Eigen::MatrixXd last;
        if constexpr (std::is_same_v<decltype(schemeTag), HestonQeProcess<HestonProcess>>) {
            const SdeSimulator<double, HestonQeProcess<HestonProcess>> simulator(grid, theta,
                                                                                 schemeTag);
            const IidGaussianSource<> source(2, 4242);
            const auto blocks =
                simulator.simulate(x0, driftOf(model), diffusionOf(model), source, nPaths, 8192);
            last = blocks.back().states.back();
        } else {
            const SdeSimulator<double, Euler> simulator(grid, theta);
            const IidGaussianSource<> source(2, 4242);
            const auto blocks =
                simulator.simulate(x0, driftOf(model), diffusionOf(model), source, nPaths, 8192);
            last = blocks.back().states.back();
        }
        return last;
    };

    const Eigen::MatrixXd qeEmpty = terminal(emptyTheta, HestonQeProcess<HestonProcess>{model});
    const Eigen::MatrixXd qeFilled = terminal(filledTheta, HestonQeProcess<HestonProcess>{model});
    const Eigen::MatrixXd euEmpty = terminal(emptyTheta, Euler{});
    const Eigen::MatrixXd euFilled = terminal(filledTheta, Euler{});
    CHECK(qeEmpty.rows() == qeFilled.rows() && qeEmpty.cols() == qeFilled.cols());
    int diffs = 0;
    for (Eigen::Index i = 0; i < qeEmpty.size(); ++i) {
        diffs += bitwiseEqual(qeEmpty(i), qeFilled(i)) ? 0 : 1;
        diffs += bitwiseEqual(euEmpty(i), euFilled(i)) ? 0 : 1;
    }
    checkClose("theta equivalence (QE + Euler)", static_cast<double>(diffs), 0.0, 0.0);
    QTA_LOG_INFO("quantape.test",
                 "  [ok] theta-driven paths bitwise equal to member-driven (QE + Euler)");
}

// ── 2. catalog: QE + Sobol QMC vs analytic ──

void testQmcPricesVsAnalytic() {
    if (quantape::math::mc::sobol::SobolGenerator::defaultTablePath().empty()) {
        QTA_LOG_WARN("quantape.test", "  [skip] no compile-time Sobol table configured");
        return;
    }
    const auto generator = quantape::math::mc::sobol::SobolGenerator::sharedFromDefaultTable();
    const quantape::models::HestonModel analytic;

    struct Case {
        quantape::models::HestonParams params;
        std::size_t steps;
    };
    const Case cases[3] = {
        {{0.04, 2.5, 0.06, 0.30, -0.1}, 64},  // moderate vol-of-vol
        {{0.01, 0.5, 0.01, 1.00, -0.5}, 128}, // high vol-of-vol
        {{0.09, 1.0, 0.09, 0.30, 0.5}, 64},   // positive correlation
    };
    const double strikes[3] = {0.9, 1.0, 1.1};
    const std::size_t nPaths = 1u << 16;

    double worst = 0.0;
    for (const Case& c : cases) {
        const HestonProcess process{0.0, c.params.kappa, c.params.theta, c.params.sigma,
                                    c.params.rho};
        const Eigen::VectorXd x0 = (Eigen::Vector2d() << 0.0, c.params.v0).finished();
        const TimeGrid grid(1.0, c.steps);
        const std::vector<std::vector<double>> thetaSteps(c.steps, thetaOf(process));
        const SdeSimulator<double, HestonQeProcess<HestonProcess>> simulator(
            grid, thetaSteps, HestonQeProcess<HestonProcess>{process, true});
        const SobolSource source(generator, 2, c.steps, 1);
        const auto blocks = simulator.simulate(x0, driftOf(process), diffusionOf(process), source,
                                               nPaths, 8192, Schedule::Parallel);
        for (double k : strikes) {
            const HestonCallPayoff payoff{k};
            double sum = 0.0;
            std::size_t n = 0;
            for (const auto& b : blocks) {
                const auto& x = b.states.back();
                for (Eigen::Index p = 0; p < x.cols(); ++p) {
                    sum += std::max(std::exp(x(0, p)) - k, 0.0);
                    ++n;
                }
            }
            const double qmc = sum / static_cast<double>(n);
            const quantape::models::HestonMarket market{1.0, k, 0.0, 0.0, 1.0};
            const double exact = analytic.call(c.params, market);
            worst = std::max(worst, std::fabs(qmc - exact));
        }
    }
    // QMC noise + the documented QE scheme discretization bias.
    checkClose("catalog QE vs analytic", worst, 0.0, 5e-3);
    QTA_LOG_INFO("quantape.test", "  [ok] catalog QE+Sobol vs analytic (worst |err| {})",
                 quantape_test::num(worst, 2));
}

// ── 3. pathwise QE gradients vs analytic ──

/// Maps the engine sensitivity layout [lnS0, v0, mu, kappa, level, eta, rho]
/// to the analytic full gradient at market {S=1, K, r=q=0, T=1}.
Eigen::VectorXd analyticEngineGradient(const quantape::models::HestonModel& model,
                                       const quantape::models::HestonParams& params,
                                       double strike) {
    const quantape::models::HestonMarket market{1.0, strike, 0.0, 0.0, 1.0};
    const quantape::models::HestonFullPoint point = quantape::models::toFullPoint(params, market);
    const auto g = model.fullGradient(point, market.tMax);
    const double price = model.call(params, market);
    Eigen::VectorXd out(7);
    out(0) = g(quantape::models::HESTON_SPOT); // d/d lnS0 at S0 = 1
    out(1) = g(quantape::models::HESTON_V0);
    out(2) = market.tMax * price + g(quantape::models::HESTON_RATE); // d/d mu
    out(3) = g(quantape::models::HESTON_KAPPA);
    out(4) = g(quantape::models::HESTON_THETA);
    out(5) = g(quantape::models::HESTON_SIGMA);
    out(6) = g(quantape::models::HESTON_RHO);
    return out;
}

void testPathwiseGradientsVsAnalytic() {
    const quantape::models::HestonParams params{0.04, 2.5, 0.06, 0.30, -0.1};
    const HestonProcess process{0.0, params.kappa, params.theta, params.sigma, params.rho};
    const Eigen::VectorXd x0 = (Eigen::Vector2d() << 0.0, params.v0).finished();
    const std::size_t nSteps = 64;
    const std::size_t nPaths = 1u << 12;
    const double strike = 1.0;
    const TimeGrid grid(1.0, nSteps);
    const std::vector<std::vector<double>> thetaSteps(nSteps, thetaOf(process));
    const SdeSimulator<double, HestonQeProcess<HestonProcess>> simulator(
        grid, thetaSteps, HestonQeProcess<HestonProcess>{process, true});
    const HestonCallPayoff payoff{strike};
    const std::vector<double> theta = thetaOf(process);

    const IidGaussianSource<> iid(2, 777);
    const GradientEstimate ad = simulateGradient(simulator, x0, theta, driftOf(process),
                                                 diffusionOf(process), iid, payoff, nPaths);

    const quantape::models::HestonModel model;
    const Eigen::VectorXd exact = analyticEngineGradient(model, params, strike);
    const quantape::models::HestonMarket market{1.0, strike, 0.0, 0.0, 1.0};
    const double exactPrice = model.call(params, market);

    QTA_LOG_INFO("quantape.test", "  pathwise QE value {} vs analytic {} (err {}, mc se {})",
                 quantape_test::num(ad.value, 6), quantape_test::num(exactPrice, 6),
                 quantape_test::num(ad.value - exactPrice, 2),
                 quantape_test::num(ad.valueStdError, 2));
    checkClose("pathwise value vs analytic", ad.value, exactPrice, 5e-3 + 4.0 * ad.valueStdError);

    const char* names[7] = {"lnS0", "v0", "mu", "kappa", "level", "eta", "rho"};
    double worstRel = 0.0;
    for (int i = 0; i < 7; ++i) {
        const double tol = 8e-3 * std::max(0.5, std::fabs(exact(i))) + 6.0 * ad.stdErrors(i);
        worstRel = std::max(worstRel, std::fabs(ad.gradient(i) - exact(i)) /
                                          std::max(1e-12, std::fabs(exact(i))));
        checkClose(names[i], ad.gradient(i), exact(i), tol);
        QTA_LOG_INFO("quantape.test", "    d/d {} pathwise {}  analytic {}  err {}", names[i],
                     quantape_test::num(ad.gradient(i), 6), quantape_test::num(exact(i), 6),
                     quantape_test::num(ad.gradient(i) - exact(i), 2));
    }
    QTA_LOG_INFO("quantape.test", "  [ok] pathwise QE gradients vs analytic (worst rel {})",
                 quantape_test::num(worstRel, 2));

    // Engine central-FD cross-check on the same discretization (tight).
    auto engineValue = [&](const std::vector<double>& th, const Eigen::VectorXd& x) {
        const SdeSimulator<double, HestonQeProcess<HestonProcess>> sim(
            grid, std::vector<std::vector<double>>(nSteps, th),
            HestonQeProcess<HestonProcess>{process, true});
        const IidGaussianSource<> src(2, 777);
        double sum = 0.0;
        for (std::size_t p = 0; p < nPaths; ++p) {
            const auto path = sim.template simulatePathSharedTheta<double>(
                x, driftOf(process), diffusionOf(process), src, p, th);
            sum += payoff.template operator()<double>(path);
        }
        return sum / static_cast<double>(nPaths);
    };
    for (int j = 0; j < 7; ++j) {
        Eigen::VectorXd xp = x0;
        Eigen::VectorXd xm = x0;
        std::vector<double> tp = theta;
        std::vector<double> tm = theta;
        double h = 1e-5;
        if (j < 2) {
            h *= std::max(1.0, std::fabs(x0(j)));
            xp(j) += h;
            xm(j) -= h;
        } else {
            h *= std::max(1.0, std::fabs(theta[static_cast<std::size_t>(j - 2)]));
            tp[static_cast<std::size_t>(j - 2)] += h;
            tm[static_cast<std::size_t>(j - 2)] -= h;
        }
        const double fd = (engineValue(tp, xp) - engineValue(tm, xm)) / (2.0 * h);
        checkClose(names[j], ad.gradient(j), fd, 1e-4 * std::max(1.0, std::fabs(fd)));
    }
    QTA_LOG_INFO("quantape.test", "  [ok] pathwise AD == engine central FD (7 sensitivities)");
}

// ── 4. flagship: SDE simulation -> calibration -> IFT market risk ──
//
// The MC pathwise greeks at the truth (dV/db and the direct market
// partials) propagated through the analytic calibration IFT must match the
// analytic total dV/da of a product outside the quote set.

void testFlagshipSdeToIft() {
    const quantape::models::HestonParams params{0.04, 2.5, 0.06, 0.30, -0.1};
    const HestonProcess process{0.0, params.kappa, params.theta, params.sigma, params.rho};
    const Eigen::VectorXd x0 = (Eigen::Vector2d() << 0.0, params.v0).finished();
    const std::size_t nSteps = 64;
    const std::size_t nPaths = 1u << 13;
    // (bias study below uses these; see the step-refinement block)
    const TimeGrid grid(1.0, nSteps);
    const std::vector<std::vector<double>> thetaSteps(nSteps, thetaOf(process));
    const SdeSimulator<double, HestonQeProcess<HestonProcess>> simulator(
        grid, thetaSteps, HestonQeProcess<HestonProcess>{process, true});
    const std::vector<double> theta = thetaOf(process);

    // QMC pathwise greeks of the product (K=1, T=1, outside the quote set).
    // The calibration chain amplifies gradient noise through the kappa
    // column (~ -900), so Sobol common random numbers keep the composed
    // risk usable.
    const double productStrike = 1.0;
    const HestonCallPayoff payoff{productStrike};
    if (quantape::math::mc::sobol::SobolGenerator::defaultTablePath().empty()) {
        QTA_LOG_WARN("quantape.test",
                     "  [skip] flagship chain: no compile-time Sobol table configured");
        return;
    }
    const auto generator = quantape::math::mc::sobol::SobolGenerator::sharedFromDefaultTable();
    const SobolSource qmc(generator, 2, nSteps, 1);
    const GradientEstimate mc = simulateGradient(simulator, x0, theta, driftOf(process),
                                                 diffusionOf(process), qmc, payoff, nPaths);

    // Calibration quote set at the same market, maturities excluding T=1.
    // Four maturities x five strikes: the kappa/sigma directions are well
    // identified, so the chain does not amplify the MC gradient noise.
    std::vector<quantape::models::HestonCalibrationQuote> quotes;
    for (double t : {0.25, 0.5, 1.25, 1.75}) {
        for (double k : {0.85, 0.95, 1.0, 1.05, 1.15}) {
            const quantape::models::HestonMarket m{1.0, k, 0.0, 0.0, t};
            const quantape::models::HestonModel model;
            quotes.push_back({k, t, model.call(params, m), 1.0});
        }
    }
    const quantape::models::HestonModel analytic;
    const quantape::models::HestonModelCalibrationProblem fixed(analytic, quotes, 1.0, 0.0, 0.0);
    quantape::math::StopCriteria criteria;
    criteria.ftol_rel = 1e-14;
    criteria.xtol_rel = 1e-12;
    criteria.grad_tol = 1e-12;
    criteria.maxeval = 500;
    quantape::math::LBFGS<double> solver(criteria, 10);
    std::vector<double> bv = {0.05, 1.5, 0.05, 0.5, -0.3};
    const Eigen::VectorXd noMarket(0);
    solver.minimize(quantape::math::CalibrationValueGrad{fixed, noMarket}, bv);
    const Eigen::VectorXd bHat = owning(bv);

    // Quote-price chain: a = the quote prices (the standard market-risk
    // setup), so db/da is well conditioned and the MC gradient noise is not
    // amplified.
    quantape::models::HestonQuotePriceCalibrationProblem chain(analytic, quotes, 1.0, 0.0, 0.0);
    Eigen::VectorXd targets(static_cast<Eigen::Index>(quotes.size()));
    for (std::size_t i = 0; i < quotes.size(); ++i) {
        targets(static_cast<Eigen::Index>(i)) = quotes[i].target;
    }
    const auto ift = quantape::math::calibrationIft(chain, bHat, targets);

    Eigen::VectorXd dVdb(5); // [v0, kappa, theta, sigma, rho]
    dVdb << mc.gradient(1), mc.gradient(3), mc.gradient(4), mc.gradient(5), mc.gradient(6);
    // ---- step-refinement study of the composed risk -------------------
    {
        const quantape::models::HestonMarket productMarket{1.0, productStrike, 0.0, 0.0, 1.0};
        const quantape::models::HestonParams bParams{bHat(0), bHat(1), bHat(2), bHat(3), bHat(4)};
        const auto gex = analytic.fullGradient(
            quantape::models::toFullPoint(bParams, productMarket), productMarket.tMax);
        const char* names[5] = {"v0", "kappa", "theta", "sigma", "rho"};
        const int idx[5] = {quantape::models::HESTON_V0, quantape::models::HESTON_KAPPA,
                            quantape::models::HESTON_THETA, quantape::models::HESTON_SIGMA,
                            quantape::models::HESTON_RHO};
        Eigen::VectorXd g64, g128;
        for (std::size_t steps : {std::size_t(64), std::size_t(128), std::size_t(256)}) {
            const TimeGrid g2(1.0, steps);
            const SdeSimulator<double, HestonQeProcess<HestonProcess>> sim(
                g2, std::vector<std::vector<double>>(steps, theta),
                HestonQeProcess<HestonProcess>{process, true});
            const SobolSource src(generator, 2, steps, 1);
            const GradientEstimate e = simulateGradient(sim, x0, theta, driftOf(process),
                                                        diffusionOf(process), src, payoff, nPaths);
            Eigen::VectorXd block(5);
            block << e.gradient(1), e.gradient(3), e.gradient(4), e.gradient(5), e.gradient(6);
            std::string stepLine = "    steps=" + std::to_string(steps) + ":";
            for (int i = 0; i < 5; ++i) {
                stepLine += " " + std::string(names[i]) + " err " +
                            quantape_test::num(block(i) - gex(0, idx[i]), 2);
            }
            QTA_LOG_INFO("quantape.test", "{}", stepLine);
            if (steps == 64) {
                g64 = block;
            } else if (steps == 128) {
                g128 = block;
            }
        }
        const Eigen::VectorXd gExtrap = 2.0 * g128 - g64;
        Eigen::VectorXd ge(5);
        for (int i = 0; i < 5; ++i) {
            ge(i) = gex(0, idx[i]);
        }
        QTA_LOG_INFO("quantape.test", "    Richardson(64,128) err: {} (max)",
                     quantape_test::num((gExtrap - ge).cwiseAbs().maxCoeff(), 2));
    }
    // -------------------------------------------------------------------
    const Eigen::VectorXd mcRisk = dVdb.transpose() * ift.dbda;

    const quantape::models::HestonMarket productMarket{1.0, productStrike, 0.0, 0.0, 1.0};
    const quantape::models::HestonParams bParams{bHat(0), bHat(1), bHat(2), bHat(3), bHat(4)};
    const auto g = analytic.fullGradient(quantape::models::toFullPoint(bParams, productMarket),
                                         productMarket.tMax);
    Eigen::VectorXd modelGrad(5);
    modelGrad << g(quantape::models::HESTON_V0), g(quantape::models::HESTON_KAPPA),
        g(quantape::models::HESTON_THETA), g(quantape::models::HESTON_SIGMA),
        g(quantape::models::HESTON_RHO);
    const Eigen::VectorXd analyticRisk = modelGrad.transpose() * ift.dbda;

    // Propagated MC uncertainty (diagonal): sigma_j = sqrt(sum_i (dbda_ij se_i)^2)
    Eigen::VectorXd sigma = Eigen::VectorXd::Zero(ift.dbda.cols());
    for (Eigen::Index j = 0; j < ift.dbda.cols(); ++j) {
        double var = 0.0;
        for (int i = 0; i < 5; ++i) {
            const double d = ift.dbda(i, j) * mc.stdErrors(i + 1);
            var += d * d;
        }
        sigma(j) = std::sqrt(var);
    }
    const double worstRatio =
        ((mcRisk - analyticRisk).array().abs() / (0.02 + 4.0 * sigma.array())).maxCoeff();
    QTA_LOG_INFO("quantape.test",
                 "  flagship: MC dV/db through db/da ({}x{}); worst |diff|/(2% + 4 sigma) = {}",
                 static_cast<long>(ift.dbda.rows()), static_cast<long>(ift.dbda.cols()),
                 quantape_test::num(worstRatio, 2));
    checkClose("flagship MC+IFT risk vs analytic", worstRatio, 0.0, 1.0);
    QTA_LOG_INFO("quantape.test",
                 "  [ok] SDE -> calibration IFT -> market risk chain (worst ratio {})",
                 quantape_test::num(worstRatio, 2));
}

// QE step-convergence: the scheme (Andersen 2008, verified branch-for-branch)
// is first-order accurate; the constant is large only in the extreme
// Feller-violating / long-maturity corner (sigma = 0.75, T = 2).
void testQeStepConvergence() {
    const quantape::models::HestonModel model;
    const std::size_t nPaths = 1u << 14;

    // Independent certification of the analytic reference for the extreme
    // configuration (CF-matched Lewis integrand via the double-exponential
    // rule).
    const quantape::models::HestonParams extreme{0.04, 2.5, 0.06, 0.75, -0.1};
    const quantape::models::HestonMarket market2{1.0, 1.0, 0.0, 0.0, 2.0};
    const double exact = model.call(extreme, market2);
    {
        const auto p5 = quantape::models::HestonModel::toComplex(extreme);
        const std::complex<double> phiHalf =
            quantape::models::HestonModel::characteristic(std::complex<double>(0.0, -0.5), p5, 2.0);
        const double sigmaBs = std::sqrt(-4.0 * std::log(phiHalf.real()));
        const double base =
            quantape::pricing::GBS<double>{
                1.0, 1.0, 0.0, 0.0, sigmaBs, 2.0, quantape::pricing::OptionType::Call}
                .price();
        const double de = quantape::math::integrateDoubleExponential<double>([&](double u) {
            const std::complex<double> phi =
                quantape::models::HestonModel::characteristicOnContour(u, p5.data(), 2.0);
            const double w = u * u + 0.25;
            const double phiCv = std::exp(-0.5 * sigmaBs * sigmaBs * 2.0 * w);
            return ((std::complex<double>(phiCv, 0.0) - phi) / w).real();
        });
        checkClose("DE-certified analytic reference", base + de / M_PI, exact, 1e-12);
    }

    // Extreme corner: bias decreases with refinement.
    const HestonProcess process{0.0, extreme.kappa, extreme.theta, extreme.sigma, extreme.rho};
    const Eigen::VectorXd x0 = (Eigen::Vector2d() << 0.0, extreme.v0).finished();
    const std::vector<double> theta = thetaOf(process);
    auto priceAt = [&](const HestonProcess& pr, const Eigen::VectorXd& y0,
                       const std::vector<double>& th, double tMax, std::size_t steps, double strike,
                       std::uint64_t seed) {
        const TimeGrid grid(tMax, steps);
        const SdeSimulator<double, HestonQeProcess<HestonProcess>> sim(
            grid, std::vector<std::vector<double>>(steps, th),
            HestonQeProcess<HestonProcess>{pr, true});
        const IidGaussianSource<> src(2, seed);
        const auto blocks =
            sim.simulate(y0, driftOf(pr), diffusionOf(pr), src, nPaths, 8192, Schedule::Parallel);
        double sum = 0.0;
        std::size_t n = 0;
        for (const auto& b : blocks) {
            const auto& x = b.states.back();
            for (Eigen::Index p = 0; p < x.cols(); ++p) {
                sum += std::max(std::exp(x(0, p)) - strike, 0.0);
                ++n;
            }
        }
        return sum / static_cast<double>(n);
    };
    const double bias64 = priceAt(process, x0, theta, 2.0, 64, 1.0, 999) - exact;
    const double bias256 = priceAt(process, x0, theta, 2.0, 256, 1.0, 999) - exact;
    QTA_LOG_INFO("quantape.test",
                 "  extreme corner (sigma=0.75, T=2): bias 64 steps {}, 256 steps {}",
                 quantape_test::num(bias64, 2), quantape_test::num(bias256, 2));
    CHECK(bias64 < 0.0 && bias256 < 0.0);
    CHECK(std::fabs(bias256) < 0.75 * std::fabs(bias64));
    CHECK(std::fabs(bias256) < 2e-2);

    // Moderate vol-of-vol: sub-basis-point accuracy already at 64 steps.
    const quantape::models::HestonParams moderate{0.04, 2.5, 0.06, 0.30, -0.1};
    const quantape::models::HestonMarket market1{1.0, 1.0, 0.0, 0.0, 1.0};
    const HestonProcess procM{0.0, moderate.kappa, moderate.theta, moderate.sigma, moderate.rho};
    const Eigen::VectorXd y0 = (Eigen::Vector2d() << 0.0, moderate.v0).finished();
    const double biasModerate =
        priceAt(procM, y0, thetaOf(procM), 1.0, 64, 1.0, 555) - model.call(moderate, market1);
    QTA_LOG_INFO("quantape.test", "  moderate (sigma=0.30, T=1): bias at 64 steps {}",
                 quantape_test::num(biasModerate, 2));
    CHECK(std::fabs(biasModerate) < 2e-3);
    QTA_LOG_INFO("quantape.test",
                 "  [ok] QE step convergence (first order; large constant only in the "
                 "extreme Feller-violating corner)");
}

} // namespace

int main() {
    QTA_LOG_INFO("quantape.test", "Heston MC gates (H5)");
    testQeStepConvergence();
    testThetaEquivalence();
    testQmcPricesVsAnalytic();
    testPathwiseGradientsVsAnalytic();
    testFlagshipSdeToIft();
    QTA_LOG_INFO("quantape.test", "ALL HESTON MC TESTS PASSED");
    return 0;
}
