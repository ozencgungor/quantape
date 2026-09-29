// test_heston_analytic.cpp — Heston model H0: CF, Lewis+BS-CV pricer, CS greeks
//
// Gates:
//   - In-house quadrature: Gauss–Laguerre moments (k! for k <= 2n-1),
//     Gauss–Legendre exactness (2/(m+1) for m <= 2n-1)
//   - Characteristic function: phi(0) = 1, phi(-i/2) in (0,1)
//   - Deterministic-variance limit: sigma -> 0 prices the BS call with the
//     integrated-variance volatility
//   - Truncation/order self-convergence (order, uMax)
//   - QE scheme, noise-free Sobol cross-check: with the Andersen martingale
//     correction the one-step E[S_T] equals the forward to ~1e-4 relative
//     (uncorrected it is ~1.2e-3 high); the 252-step call matches the
//     analytic price within the documented QE discretization bias
//     (high vol-of-vol; ~0.7% at 252 steps, converging as dt -> 0)
//   - Complex-step parameter gradients vs central finite differences
#include "quantape/math/Integrals/DoubleExponentialIntegrator.h"
#include "quantape/math/Integrals/GaussLaguerre.h"
#include "quantape/math/Integrals/GaussLegendre.h"
#include "quantape/math/SpecialFunctions/TrigIntegrals.h"
#include "quantape/mc/SdeSimulator.h"
#include "quantape/mc/SobolSource.h"
#include "quantape/mc/TimeGrid.h"
#include "quantape/mc/processes/HestonQeProcess.h"
#include "quantape/mc/processes/SdeProcesses.h"
#include "quantape/models/HestonModel.h"
#include "quantape/pricing/BlackScholes.h"

#include <Eigen/Dense>

#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#include "TestSupport.h"

namespace {

bool isFiniteBitwise(double x) {
    std::uint64_t bits = 0;
    std::memcpy(&bits, &x, sizeof(double));
    return ((bits >> 52) & 0x7FFULL) != 0x7FFULL;
}

void checkClose(const char* label, double got, double expected, double tol) {
    if (!isFiniteBitwise(got) || !isFiniteBitwise(expected) ||
        !(std::fabs(got - expected) <= tol)) {
        QTA_LOG_ERROR("test", "FAIL: {} got={} expected={} err={} tol={}", label,
                      quantape_test::num(got, 15), quantape_test::num(expected, 15),
                      quantape_test::num(std::fabs(got - expected), 3), quantape_test::num(tol, 3));
        std::exit(1);
    }
}

double factorial(int n) {
    double out = 1.0;
    for (int i = 2; i <= n; ++i) {
        out *= static_cast<double>(i);
    }
    return out;
}

void testQuadratures() {
    const quantape::math::GaussLaguerre laguerre(16);
    for (int k = 0; k <= 31; ++k) {
        const double moment = laguerre.evaluate<double>(
            [k](double x) { return std::pow(x, static_cast<double>(k)); });
        checkClose("laguerre moment", moment, factorial(k), 1e-9 * factorial(k) + 1e-12);
    }

    const quantape::math::GaussLegendre legendre(16);
    for (int m = 0; m <= 31; ++m) {
        const double integral = legendre.integrate<double>(
            [m](double x) { return std::pow(x, static_cast<double>(m)); }, -1.0, 1.0);
        const double expected = (m % 2 == 0) ? 2.0 / static_cast<double>(m + 1) : 0.0;
        checkClose("legendre moment", integral, expected, 1e-12);
    }
    QTA_LOG_INFO("test",
                 "  [ok] quadrature moments exact (Laguerre k!, Legendre 2/(m+1))");
}

void testCharacteristicIdentities() {
    const quantape::models::HestonParams params;
    const auto p = quantape::models::HestonModel::toComplex(params);
    const auto one =
        quantape::models::HestonModel::characteristic(std::complex<double>(0.0, 0.0), p, 1.5);
    checkClose("phi(0) real", one.real(), 1.0, 1e-14);
    checkClose("phi(0) imag", one.imag(), 0.0, 1e-14);

    const auto half =
        quantape::models::HestonModel::characteristic(std::complex<double>(0.0, -0.5), p, 1.5);
    CHECK(isFiniteBitwise(half.real()) && isFiniteBitwise(half.imag()));
    CHECK(half.real() > 0.0 && half.real() < 1.0);
    QTA_LOG_INFO("test", "  [ok] characteristic function: phi(0)=1, phi(-i/2) in (0,1)");
}

void testDeterministicVarianceLimit() {
    const quantape::models::HestonMarket market{1.0, 1.0, 0.01, 0.0, 1.25};
    for (double v0 : {0.04, 0.09}) {
        quantape::models::HestonParams params;
        params.v0 = v0;
        params.kappa = 2.0;
        params.theta = 0.06;
        params.sigma = 1e-4;
        params.rho = -0.5;
        const double iv = params.theta * market.tMax +
                          (params.v0 - params.theta) *
                              (1.0 - std::exp(-params.kappa * market.tMax)) / params.kappa;
        const double vol = std::sqrt(iv / market.tMax);
        const quantape::pricing::GBS<double> bs{market.spot,
                                                market.strike,
                                                market.rate,
                                                market.rate - market.dividend,
                                                vol,
                                                market.tMax,
                                                quantape::pricing::OptionType::Call};
        const quantape::models::HestonModel model;
        checkClose("BS limit", model.call(params, market), bs.price(), 1e-6);
    }
    QTA_LOG_INFO("test", "  [ok] deterministic-variance limit -> BS(integrated variance)");
}

void testSelfConvergence() {
    const quantape::models::HestonParams params{0.04, 2.5, 0.06, 0.75, -0.1};
    const quantape::models::HestonMarket market{1.0, 1.0, 0.0, 0.0, 1.0};
    const quantape::models::HestonModel coarse(quantape::models::HestonConfig{
        128, 100.0, quantape::models::HestonControlVariate::CfMatched,
        quantape::models::HestonQuadrature::GaussLegendre});
    const quantape::models::HestonModel fine(quantape::models::HestonConfig{
        1024, 400.0, quantape::models::HestonControlVariate::CfMatched,
        quantape::models::HestonQuadrature::GaussLegendre});
    checkClose("order/uMax convergence", coarse.call(params, market), fine.call(params, market),
               1e-9);
    QTA_LOG_INFO("test", "  [ok] pricer self-convergence in order and truncation");
}

void testQeSobolMartingaleAndCrossCheck() {
    using quantape::mc::Schedule;
    using quantape::mc::SdeSimulator;
    using quantape::mc::SobolSource;
    using quantape::mc::TimeGrid;
    using quantape::processes::HestonProcess;
    using quantape::processes::HestonQeProcess;

    const double mu = 0.01;
    const quantape::models::HestonParams params{0.04, 2.5, 0.06, 0.75, -0.1};
    const HestonProcess process{mu, params.kappa, params.theta, params.sigma, params.rho};
    const quantape::models::HestonMarket market{1.0, 1.0, mu, 0.0, 1.0};
    const quantape::models::HestonModel model(quantape::models::HestonConfig{512, 200.0});
    if (quantape::math::mc::sobol::SobolGenerator::defaultTablePath().empty()) {
        QTA_LOG_WARN("test",
                     "  [skip] QE/Sobol cross-check: no compile-time Sobol table configured");
        return;
    }
    const auto generator = quantape::math::mc::sobol::SobolGenerator::sharedFromDefaultTable();
    const Eigen::VectorXd x0 = (Eigen::Vector2d() << 0.0, params.v0).finished();
    const std::size_t nPaths = 1u << 18;

    // One step: with the Andersen martingale correction, E[S_T] = forward.
    {
        const TimeGrid grid(1.0, 1);
        const std::vector<std::vector<double>> thetaSteps(1);
        const SdeSimulator<double, HestonQeProcess<HestonProcess>> simulator(
            grid, thetaSteps, HestonQeProcess<HestonProcess>{process, true});
        const SobolSource source(generator, 2, 1, 1);
        const auto blocks = simulator.simulate(x0, quantape::mc::driftOf(process),
                                               quantape::mc::diffusionOf(process), source, nPaths,
                                               16384, Schedule::Parallel);
        double sum = 0.0;
        std::size_t n = 0;
        for (const auto& b : blocks) {
            const auto& x = b.states.back();
            for (Eigen::Index p = 0; p < x.cols(); ++p) {
                sum += std::exp(x(0, p));
                ++n;
            }
        }
        const double mean = sum / static_cast<double>(n);
        checkClose("corrected one-step martingale", mean / std::exp(mu) - 1.0, 0.0, 2e-5);
    }

    // 252 steps: the QE call matches the analytic price within the scheme's
    // discretization bias for this high vol-of-vol set (~0.7%).
    {
        const std::size_t nSteps = 252;
        const TimeGrid grid(1.0, nSteps);
        const std::vector<std::vector<double>> thetaSteps(nSteps);
        const SdeSimulator<double, HestonQeProcess<HestonProcess>> simulator(
            grid, thetaSteps, HestonQeProcess<HestonProcess>{process, true});
        const SobolSource source(generator, 2, nSteps, 1);
        const auto blocks = simulator.simulate(x0, quantape::mc::driftOf(process),
                                               quantape::mc::diffusionOf(process), source, nPaths,
                                               16384, Schedule::Parallel);
        double call = 0.0;
        std::size_t n = 0;
        for (const auto& b : blocks) {
            const auto& x = b.states.back();
            for (Eigen::Index p = 0; p < x.cols(); ++p) {
                call += std::max(std::exp(x(0, p)) - market.strike, 0.0);
                ++n;
            }
        }
        const double mean = call / static_cast<double>(n);
        const double analytic = model.undiscountedCall(params, market);
        checkClose("QE 252-step call", mean, analytic, 0.015 * analytic);
        QTA_LOG_INFO("test",
                     "  [ok] QE/Sobol: corrected one-step martingale, 252-step call {} vs {} "
                     "(scheme bias {}%)",
                     quantape_test::num(mean, 5), quantape_test::num(analytic, 5),
                     quantape_test::num(100.0 * (mean / analytic - 1.0), 2));
    }
}

void testControlVariates() {
    using quantape::math::cosIntegral;
    using quantape::math::sinIntegral;

    // Si/Ci anchor values (Abramowitz–Stegun 5.2)
    checkClose("Si(1)", sinIntegral(1.0), 0.946083070367183, 1e-12);
    checkClose("Ci(1)", cosIntegral(1.0), 0.337403922900968, 1e-12);
    checkClose("Si(10)", sinIntegral(10.0), 1.658347594218874, 1e-11);
    checkClose("Ci(10)", cosIntegral(10.0), -0.045456433004456, 1e-11);

    // CV identity: every reference prices the same call. Each CV is tested
    // in the regime it is designed for (the asymptotic CV's qualifying
    // condition is c_inf < 0.15; outside it the EFGL rule degrades to the
    // omega = 0 row and is only ~1e-7 accurate).
    const quantape::models::HestonParams standard{0.04, 2.5, 0.06, 0.75, -0.1};
    const quantape::models::HestonMarket standardMarket{1.0, 1.06, 0.02, 0.0, 2.0};
    const quantape::models::HestonParams slowDecay{0.01, 0.5, 0.01, 1.0, -0.5};
    const quantape::models::HestonMarket slowDecayMarket{1.0, 1.0, 0.0, 0.0, 1.0};
    const quantape::models::HestonParams* paramSets[3] = {&standard, &standard, &slowDecay};
    const quantape::models::HestonMarket* marketSets[3] = {&standardMarket, &standardMarket,
                                                           &slowDecayMarket};
    const char* names[3] = {"CfMatched", "VarianceMatched", "Asymptotic"};
    int idx = 0;
    for (auto cv : {quantape::models::HestonControlVariate::CfMatched,
                    quantape::models::HestonControlVariate::VarianceMatched,
                    quantape::models::HestonControlVariate::Asymptotic}) {
        const quantape::models::HestonModel referenceModel(quantape::models::HestonConfig{
            1024, 400.0, cv, quantape::models::HestonQuadrature::GaussLegendre});
        const double reference = referenceModel.call(*paramSets[idx], *marketSets[idx]);
        const quantape::models::HestonModel model(quantape::models::HestonConfig{512, 200.0, cv});
        const double price = model.call(*paramSets[idx], *marketSets[idx]);
        QTA_LOG_INFO("test", "    CV {} price={}  err={} (EFGL vs Gauss-Legendre)",
                     names[idx], quantape_test::num(price, 12),
                     quantape_test::num(price - reference, 2));
        checkClose(names[idx], price, reference, 1e-9);
        ++idx;
    }

    // Node stability in the slow-decay regime where Auto picks Asymptotic
    const quantape::models::HestonParams slow{0.01, 0.5, 0.01, 1.0, -0.5};
    const quantape::models::HestonMarket slowMarket{1.0, 1.0, 0.0, 0.0, 1.0};
    const quantape::models::HestonModel autoModel(
        quantape::models::HestonConfig{512, 200.0, quantape::models::HestonControlVariate::Auto});
    const double truth = autoModel.call(slow, slowMarket);
    const quantape::models::HestonModel lowBs(
        quantape::models::HestonConfig{24, 60.0, quantape::models::HestonControlVariate::CfMatched,
                                       quantape::models::HestonQuadrature::GaussLegendre});
    const quantape::models::HestonModel lowAsym(
        quantape::models::HestonConfig{24, 60.0, quantape::models::HestonControlVariate::Asymptotic,
                                       quantape::models::HestonQuadrature::GaussLegendre});
    const double errBs = std::fabs(lowBs.call(slow, slowMarket) - truth);
    const double errAsym = std::fabs(lowAsym.call(slow, slowMarket) - truth);
    QTA_LOG_INFO("test",
                 "  low-order (24 nodes) errors: BS-CV {}, asymptotic-CV {} (auto CV chosen)",
                 quantape_test::num(errBs, 3), quantape_test::num(errAsym, 3));
    CHECK(errAsym < errBs);
    QTA_LOG_INFO("test",
                 "  [ok] control variates: Si/Ci anchors, CV identity, node stability");
}

void testSurrogate() {
    const quantape::models::HestonParams params{0.04, 2.5, 0.06, 0.75, -0.1};
    const quantape::models::HestonMarket market{1.0, 1.06, 0.02, 0.0, 2.0};
    const quantape::models::HestonModel model(quantape::models::HestonConfig{512, 200.0});
    const auto surrogate = quantape::models::makeQuoteSurrogate(model, params, market);
    // Taylor consistency at a nearby point
    const Eigen::Matrix<double, 5, 1> dtheta =
        (Eigen::Matrix<double, 5, 1>() << 0.001, 0.005, -0.0015, 0.005, 0.01).finished();
    quantape::models::HestonParams shifted = params;
    shifted.v0 += dtheta(0);
    shifted.kappa += dtheta(1);
    shifted.theta += dtheta(2);
    shifted.sigma += dtheta(3);
    shifted.rho += dtheta(4);
    const double exact = model.call(shifted, market);
    const double taylor = surrogate.eval(dtheta);
    checkClose("surrogate Taylor", taylor, exact, 2e-7);

    // timing (reported): full pricer vs precomputed surrogate
    const auto now = [] { return std::chrono::steady_clock::now(); };
    const auto ns = [](auto a, auto b) {
        return std::chrono::duration<double, std::nano>(b - a).count();
    };
    {
        auto t0 = now();
        double sink = 0.0;
        for (int i = 0; i < 200; ++i)
            sink += model.call(params, market);
        auto t1 = now();
        const double priceNs = ns(t0, t1) / 200.0;
        t0 = now();
        for (int i = 0; i < 50; ++i)
            sink += model.callGradient(params, market)(0, 0);
        t1 = now();
        const double gradNs = ns(t0, t1) / 50.0;
        t0 = now();
        for (int i = 0; i < 20; ++i)
            sink += model.callHessian(params, market)(0, 0);
        t1 = now();
        const double hessNs = ns(t0, t1) / 20.0;
        const auto surrogate2 = quantape::models::makeQuoteSurrogate(model, params, market);
        t0 = now();
        Eigen::Matrix<double, 5, 1> varying = dtheta;
        for (int i = 0; i < 2000000; ++i) {
            varying(0) = dtheta(0) * (1.0 + 1e-9 * static_cast<double>(i % 7));
            sink += surrogate2.eval(varying);
        }
        t1 = now();
        const double evalNs = ns(t0, t1) / 2000000.0;
        QTA_LOG_INFO("test",
                     "  timings: price {} us, CS gradient {} us, Hessian {} us, "
                     "surrogate build {} us, surrogate eval {} ns ({}x cheaper/eval, sink {})",
                     quantape_test::num(priceNs / 1000.0, 1),
                     quantape_test::num(gradNs / 1000.0, 1), quantape_test::num(hessNs / 1000.0, 1),
                     quantape_test::num((priceNs + gradNs + hessNs) / 1000.0, 1),
                     quantape_test::num(evalNs, 0), quantape_test::num(priceNs / evalNs, 0),
                     quantape_test::num(sink, 3));
    }
    QTA_LOG_INFO("test", "  [ok] quote surrogate: value/gradient/Hessian, Taylor err {}",
                 quantape_test::num(std::fabs(taylor - exact), 2));
}

void testSmallSigmaAndPolicyBoundaries() {
    using quantape::models::HestonConfig;
    using quantape::models::HestonControlVariate;
    using quantape::models::HestonModel;
    using quantape::models::HestonParams;
    using quantape::models::HestonQuadrature;
    const HestonModel efgl; // EFGL default, Auto CV

    // Small sigma: the residual cancellation the expansion would guard
    // against must stay below the quadrature accuracy.
    {
        const HestonParams params{0.04, 2.5, 0.06, 0.01, -0.1};
        const quantape::models::HestonMarket market{1.0, 1.0, 0.0, 0.0, 1.0};
        const HestonModel reference(HestonConfig{1024, 1000.0, HestonControlVariate::CfMatched,
                                                 HestonQuadrature::GaussLegendre});
        const double price = efgl.call(params, market);
        const double ref = reference.call(params, market);
        checkClose("small-sigma price (sigma=0.01)", price, ref, 1e-8 * std::max(1.0, ref));
        const auto g = efgl.fullGradient(quantape::models::toFullPoint(params, market), 1.0);
        CHECK(std::isfinite(g(0, quantape::models::HESTON_SIGMA)));
        QTA_LOG_INFO("test", "  [ok] small sigma (0.01): price={} ref={} dsigma={}",
                     quantape_test::num(price, 10), quantape_test::num(ref, 10),
                     quantape_test::num(g(0, quantape::models::HESTON_SIGMA), 6));
    }

    // Auto CV threshold (c_inf = 0.15): the two branches must agree at the
    // switch point (the CV identity holds for any fixed reference), and Auto
    // must pick each side of the threshold.
    {
        const double v0 = 0.04, kappa = 2.5, theta = 0.06, rho = -0.1, t = 0.5;
        const double r1 = std::sqrt(1.0 - rho * rho);
        const double sigmaThreshold = r1 * (v0 + t * kappa * theta) / 0.15;
        const quantape::models::HestonMarket market{1.0, 1.0, 0.0, 0.0, t};
        const HestonParams params{v0, kappa, theta, sigmaThreshold, rho};
        const auto point = quantape::models::toFullPoint(params, market);

        const HestonModel cf(HestonConfig{512, 200.0, HestonControlVariate::CfMatched});
        const HestonModel as(HestonConfig{512, 200.0, HestonControlVariate::Asymptotic});
        const HestonModel automatic(HestonConfig{512, 200.0, HestonControlVariate::Auto});
        const double cfValue = cf.call(params, market);
        const double asValue = as.call(params, market);
        const double autoValue = automatic.call(params, market);
        const auto cfGrad = cf.fullGradient(point, t);
        const auto asGrad = as.fullGradient(point, t);
        double worstGrad = 0.0;
        for (int j = 0; j < 9; ++j) {
            worstGrad = std::max(worstGrad, std::fabs(cfGrad(0, j) - asGrad(0, j)));
        }
        checkClose("CV branch value identity at threshold", cfValue, asValue,
                   1e-9 * std::max(1.0, std::fabs(cfValue)));
        checkClose("CV branch gradient identity at threshold", worstGrad, 0.0, 1e-6);
        checkClose("Auto picks a branch at the threshold", autoValue, std::min(cfValue, asValue),
                   1e-9 + std::fabs(cfValue - asValue));
        QTA_LOG_INFO("test", "  [ok] Auto-CV threshold: branch identity dV={} dgrad={}",
                     quantape_test::num(std::fabs(cfValue - asValue), 2),
                     quantape_test::num(worstGrad, 2));
    }

    // EFGL vs Gauss-Legendre across an oscillation-frequency sweep (the
    // EFGL row/scale selection is piecewise; the values must stay within the
    // rules' shared accuracy).
    {
        const HestonParams params{0.04, 2.5, 0.06, 0.75, -0.1};
        const HestonModel reference(HestonConfig{1024, 1000.0, HestonControlVariate::CfMatched,
                                                 HestonQuadrature::GaussLegendre});
        double worst = 0.0;
        for (int i = 0; i < 40; ++i) {
            const double spot = 0.7 + 0.02 * static_cast<double>(i); // ln(F/K) sweeps rows
            const quantape::models::HestonMarket market{spot, 1.0, 0.0, 0.0, 1.0};
            worst = std::max(worst,
                             std::fabs(efgl.call(params, market) - reference.call(params, market)));
        }
        QTA_LOG_INFO("test", "  [ok] EFGL vs GL across frequency rows: max |dC| = {}",
                     quantape_test::num(worst, 2));
        checkClose("EFGL row-switch continuity", worst, 0.0, 1e-9);
    }
}

void testOscillatoryValidator() {
    // Independent cross-method check: reconstruct the CF-matched Lewis
    // integrand (the pricer's default branch) and integrate it with the
    // Ooura double-exponential rule; it must reproduce the EFGL price.
    using quantape::models::HestonModel;
    using quantape::models::HestonParams;
    const HestonParams params{0.04, 2.5, 0.06, 0.75, -0.1};
    const quantape::models::HestonMarket market{1.0, 1.06, 0.02, 0.0, 2.0};
    const HestonModel model; // EFGL, Auto -> CfMatched for this configuration
    const auto p = HestonModel::toComplex(params);
    const double t = market.tMax;
    const double fwd = market.spot * std::exp((market.rate - market.dividend) * t);
    const double mu = std::log(market.spot / market.strike) + (market.rate - market.dividend) * t;
    const double prefactor = std::sqrt(fwd * market.strike) * std::exp(-market.rate * t) / M_PI;

    // CF-matched control volatility: sigma^2 = -(8/T) ln Re phi(-i/2)
    const std::complex<double> phiHalf =
        HestonModel::characteristic(std::complex<double>(0.0, -0.5), p, t);
    const double sigmaBs = std::sqrt(-8.0 / t * std::log(phiHalf.real()));
    const double base = quantape::pricing::GBS<double>{market.spot,
                                                       market.strike,
                                                       market.rate,
                                                       market.rate - market.dividend,
                                                       sigmaBs,
                                                       t,
                                                       quantape::pricing::OptionType::Call}
                            .price();

    const double quadrature = quantape::math::integrateDoubleExponential<double>(
        [&](double u) {
            const std::complex<double> phi = HestonModel::characteristicOnContour(u, p.data(), t);
            const double w = u * u + 0.25;
            const double phiCv = std::exp(-0.5 * sigmaBs * sigmaBs * t * w);
            const std::complex<double> osc = std::exp(std::complex<double>(0.0, u * mu));
            return ((std::complex<double>(phiCv, 0.0) - phi) / w * osc).real();
        },
        0.05, 2000);
    const double price = base + prefactor * quadrature;
    const double reference = model.call(params, market);
    QTA_LOG_INFO("test", "  [ok] DE validator: price {} vs EFGL {} (err {})",
                 quantape_test::num(price, 10), quantape_test::num(reference, 10),
                 quantape_test::num(price - reference, 2));
    checkClose("double-exponential Lewis integral vs EFGL", price, reference, 1e-8);
}

void testFullJacobians() {
    using quantape::models::HestonFullPoint;
    using quantape::models::HestonModel;
    const quantape::models::HestonParams params{0.04, 2.5, 0.06, 0.75, -0.1};
    const quantape::models::HestonMarket market{1.0, 1.06, 0.02, 0.0, 2.0};
    const HestonModel model;
    const HestonFullPoint x = quantape::models::toFullPoint(params, market);

    checkClose("callFull == call", model.callFull(x, market.tMax), model.call(params, market),
               1e-15);

    const auto g = model.fullGradient(x, market.tMax);
    const auto g5 = model.callGradient(params, market);
    double modelBlock = 0.0;
    for (int j = 0; j < 5; ++j) {
        modelBlock = std::max(modelBlock, std::fabs(g(0, j) - g5(0, j)));
    }
    checkClose("fullGradient model block == callGradient", modelBlock, 0.0, 1e-14);

    // full gradient vs central FD, all nine
    const double base[9] = {params.v0,   params.kappa,  params.theta, params.sigma,   params.rho,
                            market.spot, market.strike, market.rate,  market.dividend};
    const char* names[9] = {"v0", "kappa", "theta", "sigma", "rho", "S", "K", "r", "q"};
    for (int j = 0; j < 9; ++j) {
        const double h = 1e-6 * std::max(1.0, std::fabs(base[j]));
        HestonFullPoint up = x;
        HestonFullPoint down = x;
        up[static_cast<std::size_t>(j)] += h;
        down[static_cast<std::size_t>(j)] -= h;
        const double fd =
            (model.callFull(up, market.tMax) - model.callFull(down, market.tMax)) / (2.0 * h);
        checkClose(names[j], g(0, j), fd, 1e-6 * std::max(1.0, std::fabs(fd)));
    }
    QTA_LOG_INFO("test", "  [ok] full 9-parameter gradient vs central FD");

    const auto H = model.fullHessian(x, market.tMax);
    const auto H5 = model.callHessian(params, market);
    double hBlock = 0.0;
    for (int i = 0; i < 5; ++i) {
        for (int j = 0; j < 5; ++j) {
            hBlock = std::max(hBlock, std::fabs(H(i, j) - H5(i, j)));
        }
    }
    checkClose("fullHessian model block == callHessian", hBlock, 0.0, 1e-12);

    // market Hessian vs FD of the analytic gradient
    for (int j = 5; j < 9; ++j) {
        const double h = 1e-4 * std::max(1.0, std::fabs(base[j]));
        HestonFullPoint up = x;
        HestonFullPoint down = x;
        up[static_cast<std::size_t>(j)] += h;
        down[static_cast<std::size_t>(j)] -= h;
        const auto gu = model.fullGradient(up, market.tMax);
        const auto gd = model.fullGradient(down, market.tMax);
        for (int i = 0; i < 9; ++i) {
            const double fd = (gu(0, i) - gd(0, i)) / (2.0 * h);
            checkClose("fullHessian market column vs FD(gradient)", H(i, j), fd,
                       1e-4 * std::max(1.0, std::fabs(fd)));
        }
    }
    QTA_LOG_INFO("test", "  [ok] fullHessian market block vs FD of fullGradient");

    // Euler homogeneity of the call: S C_SS + K C_SK = 0, S C_SK + K C_KK = 0
    const double e1 = market.spot * H(5, 5) + market.strike * H(5, 6);
    const double e2 = market.spot * H(5, 6) + market.strike * H(6, 6);
    checkClose("Euler 1", e1, 0.0, 1e-5);
    checkClose("Euler 2", e2, 0.0, 1e-5);
    QTA_LOG_INFO("test", "  [ok] market-Hessian Euler homogeneity (spot/strike scaling)");

    // Full 9-parameter surrogate Taylor consistency
    const auto surrogate = quantape::models::makeFullSurrogate(model, x, market.tMax);
    Eigen::Matrix<double, 9, 1> dx;
    dx << 0.001, 0.005, -0.0015, 0.005, 0.01, 0.002, -0.001, 0.0001, -0.0002;
    HestonFullPoint shifted = x;
    for (int i = 0; i < 9; ++i) {
        shifted[static_cast<std::size_t>(i)] += dx(i);
    }
    const double exact = model.callFull(shifted, market.tMax);
    checkClose("full surrogate Taylor", surrogate.eval(dx), exact, 2e-6);
    QTA_LOG_INFO("test",
                 "  [ok] full 9-parameter surrogate: value/gradient/Hessian, Taylor err {}",
                 quantape_test::num(std::fabs(surrogate.eval(dx) - exact), 2));

    const auto now = [] { return std::chrono::steady_clock::now(); };
    const auto ns = [](auto a, auto b) {
        return std::chrono::duration<double, std::nano>(b - a).count();
    };
    double sink = 0.0;
    auto t0 = now();
    for (int i = 0; i < 50; ++i) {
        sink += model.fullGradient(x, market.tMax)(0, 0);
    }
    auto t1 = now();
    const double gradUs = ns(t0, t1) / 50.0 / 1000.0;
    t0 = now();
    for (int i = 0; i < 20; ++i) {
        sink += model.fullHessian(x, market.tMax)(0, 0);
    }
    t1 = now();
    const double hessUs = ns(t0, t1) / 20.0 / 1000.0;
    t0 = now();
    for (int i = 0; i < 20; ++i) {
        sink += quantape::models::makeFullSurrogate(model, x, market.tMax).value;
    }
    t1 = now();
    const double buildUs = ns(t0, t1) / 20.0 / 1000.0;
    QTA_LOG_INFO("test",
                 "  full-Jacobian timings: gradient {} us, Hessian {} us, "
                 "surrogate build {} us (sink {})",
                 quantape_test::num(gradUs, 1), quantape_test::num(hessUs, 1),
                 quantape_test::num(buildUs, 1), quantape_test::num(sink, 3));
}

void testComplexStepGradients() {
    const quantape::models::HestonParams params{0.04, 2.5, 0.06, 0.75, -0.1};
    const quantape::models::HestonMarket market{1.0, 1.06, 0.02, 0.0, 2.0};
    const quantape::models::HestonModel model(quantape::models::HestonConfig{512, 200.0});
    const Eigen::Matrix<double, 1, 5> cs = model.callGradient(params, market);

    const double values[5] = {params.v0, params.kappa, params.theta, params.sigma, params.rho};
    for (int j = 0; j < 5; ++j) {
        const double h = 1e-6 * std::max(1.0, std::fabs(values[j]));
        quantape::models::HestonParams up = params;
        quantape::models::HestonParams down = params;
        double* fieldsUp[5] = {&up.v0, &up.kappa, &up.theta, &up.sigma, &up.rho};
        double* fieldsDown[5] = {&down.v0, &down.kappa, &down.theta, &down.sigma, &down.rho};
        *fieldsUp[j] += h;
        *fieldsDown[j] -= h;
        const double fd = (model.call(up, market) - model.call(down, market)) / (2.0 * h);
        checkClose("complex-step vs FD", cs(0, j), fd, 1e-6 * std::max(1e-8, std::fabs(fd)) + 1e-9);
    }
    QTA_LOG_INFO("test", "  [ok] complex-step gradients vs central FD (all 5 parameters)");
}

} // namespace

int main() {
    QTA_LOG_INFO("test", "Heston model (H0) tests");
    testQuadratures();
    testCharacteristicIdentities();
    testDeterministicVarianceLimit();
    testSelfConvergence();
    testQeSobolMartingaleAndCrossCheck();
    testControlVariates();
    testSurrogate();
    testComplexStepGradients();
    testFullJacobians();
    testSmallSigmaAndPolicyBoundaries();
    testOscillatoryValidator();
    QTA_LOG_INFO("test", "ALL HESTON MODEL TESTS PASSED");
    return 0;
}
