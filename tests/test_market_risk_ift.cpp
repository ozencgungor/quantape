// test_market_risk_ift.cpp — market-risk propagation through calibrations (S5d)
//
// Gates:
//   11. KKT route (optimizer + ImplicitFunction.h) == instrument-Jacobian
//       route (weighted SVD pseudo-inverse) on exact-fit full-rank LS
//   12. synthetic quotes: bump quote -> recalibrate -> reprice vs
//       IFT-propagated greeks (single/two/three-quote chains)
//   13. calibration-instrument exactness: pricing the instrument through
//       the chain reproduces its market greeks exactly
//   14. route benchmark: instrument vs KKT timing (reported)
//   plus: end-to-end SDE gradient (mc/Gradients.h) -> IFT market risk
#include "quantape/math/StanMath.h"

#include "quantape/calibration/CalibrationChainKkt.h"
#include "quantape/log/Log.h"
#include "quantape/mc/Gradients.h"
#include "quantape/mc/RandomSource.h"
#include "quantape/mc/SchemesStan.h"
#include "quantape/mc/SdeSimulator.h"
#include "quantape/mc/TimeGrid.h"
#include "quantape/util/Check.h"

#include <Eigen/Dense>

#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <vector>
using quantape::util::checkClose;
using quantape::util::isFiniteBitwise;

using quantape::math::Bounds;
using quantape::math::CalibrationJacobians;
using quantape::math::IftResult;
using quantape::math::OptimizerState;
using quantape::math::StopCriteria;

namespace {

// ── Scalar-generic undiscounted Black-Scholes call (drift mu as r) ──

template <typename Scalar>
Scalar bsCallS(const Scalar& s0, double strike, const Scalar& mu, const Scalar& sigma,
               double tMax) {
    using std::erfc;
    using std::exp;
    using std::log;
    using std::sqrt;
    const Scalar sqrtT = sqrt(Scalar(tMax));
    const Scalar d1 =
        (log(s0 / Scalar(strike)) + (mu + Scalar(0.5) * sigma * sigma) * Scalar(tMax)) /
        (sigma * sqrtT);
    const Scalar d2 = d1 - sigma * sqrtT;
    const Scalar nd1 = Scalar(0.5) * erfc(-d1 * Scalar(M_SQRT1_2));
    const Scalar nd2 = Scalar(0.5) * erfc(-d2 * Scalar(M_SQRT1_2));
    // Undiscounted expectation E[(S_T - K)^+] = e^{mu T} * C_BS(r = mu),
    // matching the mc engine convention (discounting is a payoff concern).
    return s0 * exp(mu * Scalar(tMax)) * nd1 - Scalar(strike) * nd2;
}

double bsCallPrice(double s0, double strike, double mu, double sigma, double tMax) {
    return bsCallS<double>(s0, strike, mu, sigma, tMax);
}

double bsCallDelta(double s0, double strike, double mu, double sigma, double tMax) {
    const double sqrtT = std::sqrt(tMax);
    const double d1 = (std::log(s0 / strike) + (mu + 0.5 * sigma * sigma) * tMax) / (sigma * sqrtT);
    return std::exp(mu * tMax) * 0.5 * std::erfc(-d1 * M_SQRT1_2);
}

double bsCallVega(double s0, double strike, double mu, double sigma, double tMax) {
    const double sqrtT = std::sqrt(tMax);
    const double d1 = (std::log(s0 / strike) + (mu + 0.5 * sigma * sigma) * tMax) / (sigma * sqrtT);
    const double phi = std::exp(-0.5 * d1 * d1) / std::sqrt(2.0 * M_PI);
    return std::exp(mu * tMax) * s0 * sqrtT * phi;
}

// ── Fitting objectives (callable contract: f2(x, m), x = model params) ──

struct SingleQuoteObjective {
    double strike = 100.0;
    double mu = 0.05;
    double tMax = 1.0;

    template <typename Sx, typename Sm>
    auto operator()(const std::vector<Sx>& x, const std::vector<Sm>& m) const {
        using S = decltype(x[0] * Sx(0.0) + m[0] * Sm(0.0));
        const S model = bsCallS<S>(S(100.0), strike, S(mu), x[0], tMax);
        const S r = model - S(m[0]);
        return S(0.5) * r * r;
    }
};

struct TwoQuoteObjective {
    double strike1 = 100.0;
    double strike2 = 110.0;
    double mu = 0.05;
    double tMax = 1.0;

    template <typename Sx, typename Sm>
    auto operator()(const std::vector<Sx>& x, const std::vector<Sm>& m) const {
        using S = decltype(x[0] * Sx(0.0) + m[0] * Sm(0.0));
        const S model1 = bsCallS<S>(x[0], strike1, S(mu), x[1], tMax);
        const S model2 = bsCallS<S>(x[0], strike2, S(mu), x[1], tMax);
        const S r1 = model1 - S(m[0]);
        const S r2 = model2 - S(m[1]);
        return S(0.5) * (r1 * r1 + r2 * r2);
    }
};

struct ThreeQuoteObjective {
    double strikes[3] = {90.0, 100.0, 110.0};
    double weights[3] = {1.0, 1.0, 0.5};
    double mu = 0.05;
    double tMax = 1.0;

    template <typename Sx, typename Sm>
    auto operator()(const std::vector<Sx>& x, const std::vector<Sm>& m) const {
        using S = decltype(x[0] * Sx(0.0) + m[0] * Sm(0.0));
        S sum = S(0.0);
        for (int k = 0; k < 3; ++k) {
            const S model = bsCallS<S>(x[0], strikes[k], S(mu), x[1], tMax);
            const S r = model - S(m[k]);
            sum += S(0.5 * weights[k]) * r * r;
        }
        return sum;
    }
};

// ── Instrument Jacobian rows for the single-asset chain ──

/// dI/db for a sigma-only calibration (one column).
CalibrationJacobians sigmaOnlyJacobians(const std::vector<double>& strikes, double s0, double mu,
                                        double sigma, double tMax,
                                        const std::vector<double>& weights) {
    const Eigen::Index nI = static_cast<Eigen::Index>(strikes.size());
    CalibrationJacobians jac;
    jac.weights = weights;
    jac.dIdb.resize(nI, 1);
    jac.dIda = -Eigen::MatrixXd::Identity(nI, nI); // I_i = P_i - a_i (price data)
    for (Eigen::Index i = 0; i < nI; ++i) {
        jac.dIdb(i, 0) = bsCallVega(s0, strikes[static_cast<std::size_t>(i)], mu, sigma, tMax);
    }
    return jac;
}

/// dI/db for an (S0, sigma) calibration (two columns).
CalibrationJacobians singleAssetJacobians(const std::vector<double>& strikes, double s0, double mu,
                                          double sigma, double tMax,
                                          const std::vector<double>& weights) {
    const Eigen::Index nI = static_cast<Eigen::Index>(strikes.size());
    CalibrationJacobians jac;
    jac.weights = weights;
    jac.dIdb.resize(nI, 2);
    jac.dIda = -Eigen::MatrixXd::Identity(nI, nI); // I_i = P_i - a_i (price data)
    for (Eigen::Index i = 0; i < nI; ++i) {
        jac.dIdb(i, 0) = bsCallDelta(s0, strikes[static_cast<std::size_t>(i)], mu, sigma, tMax);
        jac.dIdb(i, 1) = bsCallVega(s0, strikes[static_cast<std::size_t>(i)], mu, sigma, tMax);
    }
    return jac;
}

// ── Gate 11/12/13: single-quote chain ──

void testSingleQuoteChain() {
    const double s0 = 100.0, mu = 0.05, tMax = 1.0, sigma = 0.25, k1 = 100.0, k2 = 110.0;
    const double a1 = bsCallPrice(s0, k1, mu, sigma, tMax);
    const double vega1 = bsCallVega(s0, k1, mu, sigma, tMax);

    // Instrument route (sigma-only chain)
    const auto jac = sigmaOnlyJacobians({k1}, s0, mu, sigma, tMax, {1.0});
    const auto inst = quantape::math::instrumentCalibrationJacobian(jac);
    CHECK(inst.rank == 1);
    checkClose("inst db/da", inst.dbda(0, 0), 1.0 / vega1, 1e-10);

    // KKT route
    std::vector<double> quotes = {a1};
    std::vector<double> b = {0.20};
    OptimizerState state;
    IftResult info;
    const Eigen::MatrixXd kkt = quantape::math::kktCalibrationJacobian(
        SingleQuoteObjective{k1, mu, tMax}, quotes, b, state, info);
    checkClose("kkt calibrated sigma", b[0], sigma, 1e-10);
    checkClose("kkt db/da", kkt(0, 0), 1.0 / vega1, 1e-8);

    // Gate 11: routes agree
    checkClose("routes agree", kkt(0, 0), inst.dbda(0, 0), 1e-8);

    // Gate 13: instrument exactness (V = the calibration instrument)
    Eigen::VectorXd dVdb(1);
    dVdb(0) = vega1;
    const Eigen::VectorXd selfRisk =
        quantape::math::propagateMarketRisks(dVdb, Eigen::VectorXd{}, inst.dbda, Eigen::MatrixXd{});
    checkClose("instrument exactness", selfRisk(0), 1.0, 1e-12);

    // Gate 12: downstream product (K2), IFT vs bump-recalibrate
    Eigen::VectorXd dVdb2(1);
    dVdb2(0) = bsCallVega(s0, k2, mu, sigma, tMax);
    const Eigen::VectorXd risk = quantape::math::propagateMarketRisks(dVdb2, Eigen::VectorXd{},
                                                                      inst.dbda, Eigen::MatrixXd{});

    const double h = 1e-4;
    std::vector<double> bp = {sigma}, bm = {sigma};
    std::vector<double> ap = {a1 + h}, am = {a1 - h};
    OptimizerState sp, sm;
    IftResult ip, im;
    quantape::math::kktCalibrationJacobian(SingleQuoteObjective{k1, mu, tMax}, ap, bp, sp, ip);
    quantape::math::kktCalibrationJacobian(SingleQuoteObjective{k1, mu, tMax}, am, bm, sm, im);
    const double fd =
        (bsCallPrice(s0, k2, mu, bp[0], tMax) - bsCallPrice(s0, k2, mu, bm[0], tMax)) / (2.0 * h);
    checkClose("single-quote IFT vs bump-recalibrate", risk(0), fd, 1e-6);
    QTA_LOG_INFO("test", "  [ok] single quote: KKT/inst agree, instrument exact, vs FD ({} vs {})",
                 quantape::util::num(risk(0), 9), quantape::util::num(fd, 9));
}

// ── Gate 12: two-quote chain (S0 and sigma) ──

void testTwoQuoteChain() {
    const double mu = 0.05, tMax = 1.0, sigma = 0.25, s0 = 100.0;
    const double k1 = 100.0, k2 = 110.0, k3 = 105.0;
    const std::vector<double> quotes = {bsCallPrice(s0, k1, mu, sigma, tMax),
                                        bsCallPrice(s0, k2, mu, sigma, tMax)};

    // KKT route
    std::vector<double> b = {99.0, 0.3};
    OptimizerState state;
    IftResult info;
    const Eigen::MatrixXd kkt = quantape::math::kktCalibrationJacobian(
        TwoQuoteObjective{k1, k2, mu, tMax}, quotes, b, state, info);
    checkClose("two-quote S0", b[0], s0, 1e-9);
    checkClose("two-quote sigma", b[1], sigma, 1e-9);

    // Instrument route and analytic Jacobian inverse
    const auto jac = singleAssetJacobians({k1, k2}, s0, mu, sigma, tMax, {1.0, 1.0});
    const auto inst = quantape::math::instrumentCalibrationJacobian(jac);
    const Eigen::MatrixXd jinv = jac.dIdb.inverse();
    for (Eigen::Index i = 0; i < 2; ++i) {
        for (Eigen::Index j = 0; j < 2; ++j) {
            checkClose("two-quote inst == J^-1", inst.dbda(i, j), jinv(i, j), 1e-10);
            checkClose("two-quote kkt == J^-1", kkt(i, j), jinv(i, j), 1e-8);
        }
    }

    // Downstream product K3: risk to each quote vs bump-recalibrate
    Eigen::VectorXd dVdb(2);
    dVdb(0) = bsCallDelta(s0, k3, mu, sigma, tMax);
    dVdb(1) = bsCallVega(s0, k3, mu, sigma, tMax);
    const Eigen::VectorXd risk =
        quantape::math::propagateMarketRisks(dVdb, Eigen::VectorXd{}, inst.dbda, Eigen::MatrixXd{});
    const double h = 1e-4;
    for (int j = 0; j < 2; ++j) {
        std::vector<double> ap = quotes, am = quotes;
        ap[j] += h;
        am[j] -= h;
        std::vector<double> bp = {s0, sigma}, bm = {s0, sigma};
        OptimizerState sp, sm;
        IftResult ip, im;
        quantape::math::kktCalibrationJacobian(TwoQuoteObjective{k1, k2, mu, tMax}, ap, bp, sp, ip);
        quantape::math::kktCalibrationJacobian(TwoQuoteObjective{k1, k2, mu, tMax}, am, bm, sm, im);
        const double vp = bsCallS<double>(bp[0], k3, mu, bp[1], tMax);
        const double vm = bsCallS<double>(bm[0], k3, mu, bm[1], tMax);
        const double fd = (vp - vm) / (2.0 * h);
        checkClose("two-quote IFT vs bump-recalibrate", risk(j), fd, 1e-5);
    }
    QTA_LOG_INFO("test", "  [ok] two quotes: routes == J^-1, both components vs FD");
}

// ── Gates 12/14: weighted best fit (3 quotes, 2 params) ──

void testBestFitWeighted() {
    const double mu = 0.05, tMax = 1.0, sigma = 0.25, s0 = 100.0;
    const ThreeQuoteObjective objective{};
    const std::vector<double> quotes = {bsCallPrice(s0, objective.strikes[0], mu, sigma, tMax),
                                        bsCallPrice(s0, objective.strikes[1], mu, sigma, tMax),
                                        bsCallPrice(s0, objective.strikes[2], mu, sigma, tMax)};

    // Calibrate (best fit; S0 can fit exactly, sigma only approximately)
    std::vector<double> b = {100.0, 0.2};
    OptimizerState state;
    IftResult info;
    const Eigen::MatrixXd kkt =
        quantape::math::kktCalibrationJacobian(objective, quotes, b, state, info);
    const auto kal = std::chrono::steady_clock::now();
    const Eigen::MatrixXd kkt2 = quantape::math::kktCalibrationJacobian(
        objective, quotes, b, state, info); // warm restart (timing gate 14)
    const auto kal2 = std::chrono::steady_clock::now();
    (void)kkt2;

    // Instrument route at the optimum
    const auto jac = singleAssetJacobians(
        {objective.strikes[0], objective.strikes[1], objective.strikes[2]}, b[0], mu, b[1], tMax,
        {objective.weights[0], objective.weights[1], objective.weights[2]});
    const auto t0 = std::chrono::steady_clock::now();
    const auto inst = quantape::math::instrumentCalibrationJacobian(jac);
    const auto t1 = std::chrono::steady_clock::now();
    CHECK(inst.rank == 2);

    // Downstream product K=105: both routes vs bump-recalibrate
    const double k3 = 105.0;
    Eigen::VectorXd dVdb(2);
    dVdb(0) = bsCallDelta(b[0], k3, mu, b[1], tMax);
    dVdb(1) = bsCallVega(b[0], k3, mu, b[1], tMax);
    const Eigen::VectorXd riskInst =
        quantape::math::propagateMarketRisks(dVdb, Eigen::VectorXd{}, inst.dbda, Eigen::MatrixXd{});
    const Eigen::VectorXd riskKkt =
        quantape::math::propagateMarketRisks(dVdb, Eigen::VectorXd{}, kkt, Eigen::MatrixXd{});

    const double h = 1e-4;
    double maxKktErr = 0.0;
    double maxInstErr = 0.0;
    for (int j = 0; j < 3; ++j) {
        std::vector<double> ap = quotes, am = quotes;
        ap[j] += h;
        am[j] -= h;
        std::vector<double> bp = b, bm = b;
        OptimizerState sp, sm;
        IftResult ip, im;
        quantape::math::kktCalibrationJacobian(objective, ap, bp, sp, ip);
        quantape::math::kktCalibrationJacobian(objective, am, bm, sm, im);
        const double vp = bsCallS<double>(bp[0], k3, mu, bp[1], tMax);
        const double vm = bsCallS<double>(bm[0], k3, mu, bm[1], tMax);
        const double fd = (vp - vm) / (2.0 * h);
        maxKktErr = std::max(maxKktErr, std::fabs(riskKkt(j) - fd));
        maxInstErr = std::max(maxInstErr, std::fabs(riskInst(j) - fd));
        checkClose("best-fit KKT vs FD", riskKkt(j), fd, 1e-5);
        checkClose("best-fit inst vs FD", riskInst(j), fd, 1e-3);
    }
    const double kktUs = std::chrono::duration<double, std::micro>(kal2 - kal).count();
    const double instUs = std::chrono::duration<double, std::micro>(t1 - t0).count();
    QTA_LOG_INFO("test",
                 "  [ok] best fit (3 quotes, weights): max |KKT-FD|={}, |inst-FD|={}; "
                 "timing warm KKT {} us vs instrument {} us (m=2, nI=3)",
                 quantape::util::num(maxKktErr, 3), quantape::util::num(maxInstErr, 3),
                 quantape::util::num(kktUs), quantape::util::num(instUs));
}

// ── End-to-end: SDE gradient (S5) -> IFT chain (S5d) ──

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

struct TerminalCallPayoff {
    double strike = 100.0;
    template <typename Scalar>
    Scalar operator()(const quantape::mc::PathBlock<Scalar>& path) const {
        const Scalar s = path.states.back()(0, 0);
        return s > Scalar(strike) ? s - Scalar(strike) : Scalar(0.0);
    }
};

void testSdeChain() {
    using quantape::mc::IidGaussianSource;
    using quantape::mc::Milstein;
    using quantape::mc::Schedule;
    using quantape::mc::SdeSimulator;
    using quantape::mc::simulateGradient;
    using quantape::mc::TimeGrid;

    const double s0 = 100.0, mu = 0.05, sigma = 0.25, tMax = 1.0;
    const double k1 = 100.0, k2 = 110.0;
    const std::size_t nSteps = 128;
    const std::size_t nPaths = 20000;

    const TimeGrid grid(tMax, nSteps);
    const std::vector<std::vector<double>> thetaSteps(nSteps);
    const SdeSimulator<double, Milstein> simulator(grid, thetaSteps);
    const std::vector<double> theta = {mu, sigma};
    Eigen::VectorXd x0(1);
    x0(0) = s0;
    const IidGaussianSource<> source(1, 4242);
    const auto estimate = simulateGradient(simulator, x0, theta, quantape::mc::driftOf(GbmModel{}),
                                           quantape::mc::diffusionOf(GbmModel{}), source,
                                           TerminalCallPayoff{k2}, nPaths, Schedule::Parallel);

    // Chain: single-quote calibration on K1 -> market risk of the K2 call
    const auto jac = sigmaOnlyJacobians({k1}, s0, mu, sigma, tMax, {1.0});
    const auto inst = quantape::math::instrumentCalibrationJacobian(jac);
    Eigen::VectorXd dVdb(1);
    dVdb(0) = estimate.gradient(2); // dV/dsigma from the SDE engine
    const Eigen::VectorXd risk =
        quantape::math::propagateMarketRisks(dVdb, Eigen::VectorXd{}, inst.dbda, Eigen::MatrixXd{});

    const double vega1 = bsCallVega(s0, k1, mu, sigma, tMax);
    const double analytic = bsCallVega(s0, k2, mu, sigma, tMax) / vega1;
    const double tol = 0.02 * std::fabs(analytic) + 5.0 * estimate.stdErrors(2) / vega1;
    checkClose("SDE -> IFT chain", risk(0), analytic, tol);
    QTA_LOG_INFO("test", "  [ok] SDE gradient -> IFT: dV/da={} (analytic {}, se {})",
                 quantape::util::num(risk(0), 6), quantape::util::num(analytic, 6),
                 quantape::util::num(estimate.stdErrors(2) / vega1, 2));
}

} // namespace

int main() {
    QTA_LOG_INFO("test", "Market-risk IFT tests (S5d)");
    testSingleQuoteChain();
    testTwoQuoteChain();
    testBestFitWeighted();
    testSdeChain();
    QTA_LOG_INFO("test", "ALL MARKET-RISK IFT TESTS PASSED");
    return 0;
}
