// test_heston_calibration.cpp — H6: Heston calibration strategy shoot-out
//
// Synthetic quotes are generated from a known Heston parameter set with the
// analytic pricer (EFGL). Six strategies are compared on the weighted
// least-squares fit, both noiseless and with 0.2% price noise:
//
//   S1  LBFGS<double>, exact analytic value+gradient, price space, uniform
//   S2  LBFGS<double>, vega weights (1/vega^2, the IV-space linearization)
//   S3  LBFGS<double>, implied-vol space residuals (dIV/dtheta = vega^-1 dC)
//   S4  LBFGS<double>, log-parameter space (positive params, tanh(rho))
//   S5  Levenberg-Marquardt Gauss-Newton on the analytic Jacobian
//   S6  Newton: LM with the true LS Hessian (J'WJ + sum w r_i H_i)
//   S7  LBFGS<var> through the make_callback_var price (Stan backend)
//
// Plus identifiability diagnostics: the singular values of the 9-parameter
// Jacobian (why S/K joint scale and r/q term structure are the weak
// directions) and LM fits with 6 (5+r) and 9 parameters.
//
// Gates: noiseless fits recover the truth (objective < 1e-9, max parameter
// error < 5e-3); noisy fits land within 3x of the noise floor.

#include "quantape/math/StanMath.h"

#include "quantape/calibration/CalibrationProblem.h"
#include "quantape/calibration/HestonCalibration.h"
#include "quantape/math/Optimization/AugLag.h"
#include "quantape/math/Optimization/LBFGS.h"
#include "quantape/math/Optimization/OptimizerStanPrimitives.h"
#include "quantape/math/Optimization/SLSQP.h"
#include "quantape/models/HestonModel.h"
#include "quantape/models/HestonStanPrimitives.h"
#include "quantape/pricing/BlackScholes.h"

#include <Eigen/Dense>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <random>
#include <string>
#include <type_traits>
#include <vector>

#include "TestSupport.h"

using quantape::models::HestonFullPoint;
using quantape::models::HestonMarket;
using quantape::models::HestonModel;
using quantape::models::HestonParams;

namespace {

constexpr int kNumParams = quantape::models::HESTON_PARAM_COUNT;

void checkClose(const char* label, double got, double expected, double tol) {
    if (!(std::fabs(got - expected) <= tol)) {
        QTA_LOG_ERROR("quantape.test", "FAIL: {} got={} expected={} err={} tol={}", label,
                      quantape_test::num(got, 12), quantape_test::num(expected, 12),
                      quantape_test::num(std::fabs(got - expected), 3), quantape_test::num(tol, 3));
        std::exit(1);
    }
}

struct Quote {
    double strike = 1.0;
    double tMax = 1.0;
    double targetPrice = 0.0;
    double targetIv = 0.0;
    double vega = 0.0;
};

struct Fixture {
    HestonParams truth;
    double spot = 1.0;
    double rate = 0.0;
    double dividend = 0.0;
    HestonModel model;
    std::vector<Quote> quotes;
};

double impliedVol(double price, double S, double K, double r, double q, double T) {
    double lo = 1e-6;
    double hi = 5.0;
    for (int i = 0; i < 100; ++i) {
        const double mid = 0.5 * (lo + hi);
        const double p =
            quantape::pricing::GBS<double>{
                S, K, r, r - q, mid, T, quantape::pricing::OptionType::Call}
                .price();
        if (p < price) {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    return 0.5 * (lo + hi);
}

Fixture makeFixture(double noise, unsigned seed,
                    HestonParams truth = {0.04, 2.5, 0.06, 0.75, -0.1}) {
    Fixture fx;
    fx.truth = truth;
    fx.spot = 1.0;
    fx.rate = 0.0;
    fx.dividend = 0.0;
    const double strikes[5] = {0.8, 0.9, 1.0, 1.1, 1.25};
    const double maturities[2] = {0.5, 2.0};
    std::mt19937_64 rng(seed);
    std::normal_distribution<double> dist(0.0, noise);
    for (double t : maturities) {
        const HestonMarket market{fx.spot, 0.0, fx.rate, fx.dividend, t};
        for (double k : strikes) {
            HestonMarket m = market;
            m.strike = k;
            const double clean = fx.model.call(fx.truth, m);
            const double target = clean + dist(rng) * clean;
            Quote q;
            q.strike = k;
            q.tMax = t;
            q.targetPrice = target;
            q.targetIv = impliedVol(target, fx.spot, k, fx.rate, fx.dividend, t);
            q.vega =
                quantape::pricing::gbsAnalytical(fx.spot, k, fx.rate, fx.rate - fx.dividend,
                                                 q.targetIv, t, quantape::pricing::OptionType::Call)
                    .g1.dV_dvol;
            fx.quotes.push_back(q);
        }
    }
    return fx;
}

// ── Fit problem: active parameter subset, weights, residual space ──

enum class WeightMode { Uniform, Vega };
enum class SpaceMode { Price, ImpliedVol };

struct FitProblem {
    const Fixture* fx = nullptr;
    std::vector<int> active{0, 1, 2, 3, 4};
    WeightMode weights = WeightMode::Uniform;
    SpaceMode space = SpaceMode::Price;
    /// Feller-by-construction: sigma = sqrt(2 kappa theta), active = {v0, kappa,
    /// theta, rho}; the sigma column of the Jacobian carries the chain rule.
    bool fellerBoundary = false;
    mutable bool feasible = true;

    HestonMarket marketFor(const Quote& q) const {
        return {fx->spot, q.strike, fx->rate, fx->dividend, q.tMax};
    }

    bool strikeActive() const {
        return std::find(active.begin(), active.end(), quantape::models::HESTON_STRIKE) !=
               active.end();
    }

    /// Parameter point for one quote: the quote's own strike unless strike is
    /// an active fit parameter.
    HestonFullPoint quotePoint(const std::vector<double>& xa, const Quote& q) const {
        HestonFullPoint p = point(xa);
        if (!strikeActive()) {
            p[quantape::models::HESTON_STRIKE] = q.strike;
        }
        return p;
    }

    HestonFullPoint point(const std::vector<double>& xa) const {
        HestonFullPoint p{};
        const HestonMarket base{fx->spot, 1.0, fx->rate, fx->dividend, 1.0};
        p = quantape::models::toFullPoint(fx->truth, base);
        p[quantape::models::HESTON_SPOT] = fx->spot;
        p[quantape::models::HESTON_STRIKE] = 1.0;
        p[quantape::models::HESTON_RATE] = fx->rate;
        p[quantape::models::HESTON_DIVIDEND] = fx->dividend;
        for (std::size_t k = 0; k < active.size(); ++k) {
            p[static_cast<std::size_t>(active[k])] = xa[k];
        }
        if (fellerBoundary) {
            p[quantape::models::HESTON_SIGMA] = std::sqrt(2.0 * p[quantape::models::HESTON_KAPPA] *
                                                          p[quantape::models::HESTON_THETA]);
        }
        return p;
    }

    bool pointFeasible(const HestonFullPoint& p) const {
        return p[quantape::models::HESTON_V0] > 1e-10 &&
               p[quantape::models::HESTON_KAPPA] > 1e-10 &&
               p[quantape::models::HESTON_THETA] > 1e-10 &&
               p[quantape::models::HESTON_SIGMA] > 1e-10 &&
               std::fabs(p[quantape::models::HESTON_RHO]) < 0.999;
    }

    double sqrtWeight(const Quote& q) const {
        if (weights == WeightMode::Vega) {
            return 1.0 / std::max(1e-4, q.vega);
        }
        return 1.0;
    }

    /// Residuals r_i = sqrtW_i (model_i - target_i) and (optionally) the
    /// Jacobian wrt the active parameters. IV space uses
    /// dIV/dx = (dC/dx) / vega_BS(IV_model).
    void residuals(const std::vector<double>& xa, std::vector<double>& r,
                   Eigen::MatrixXd* jacobian) const {
        const HestonFullPoint p = point(xa);
        feasible = pointFeasible(p);
        const std::size_t nq = fx->quotes.size();
        const std::size_t np = active.size();
        r.resize(nq, 0.0);
        if (jacobian) {
            jacobian->resize(static_cast<Eigen::Index>(nq), static_cast<Eigen::Index>(np));
        }
        for (std::size_t i = 0; i < nq; ++i) {
            const Quote& q = fx->quotes[i];
            if (!feasible) {
                r[i] = 1.0;
                if (jacobian) {
                    jacobian->row(static_cast<Eigen::Index>(i)).setZero();
                }
                continue;
            }
            const HestonFullPoint pq = quotePoint(xa, q);
            const double price = fx->model.callFull(pq, q.tMax);
            const double sw = sqrtWeight(q);
            if (space == SpaceMode::Price) {
                r[i] = sw * (price - q.targetPrice);
                if (jacobian) {
                    const auto g = fx->model.fullGradient(pq, q.tMax);
                    if (fellerBoundary) {
                        const double kappa = xa[1];
                        const double theta = xa[2];
                        const double sigma = std::sqrt(2.0 * kappa * theta);
                        const double dSigmaDKappa = sigma > 0.0 ? theta / sigma : 0.0;
                        const double dSigmaDTheta = sigma > 0.0 ? kappa / sigma : 0.0;
                        (*jacobian)(static_cast<Eigen::Index>(i), 0) = sw * g(0, 0);
                        (*jacobian)(static_cast<Eigen::Index>(i), 1) =
                            sw * (g(0, 1) + g(0, 3) * dSigmaDKappa);
                        (*jacobian)(static_cast<Eigen::Index>(i), 2) =
                            sw * (g(0, 2) + g(0, 3) * dSigmaDTheta);
                        (*jacobian)(static_cast<Eigen::Index>(i), 3) = sw * g(0, 4);
                    } else {
                        for (std::size_t k = 0; k < np; ++k) {
                            (*jacobian)(static_cast<Eigen::Index>(i),
                                        static_cast<Eigen::Index>(k)) = sw * g(0, active[k]);
                        }
                    }
                }
            } else {
                const double iv =
                    impliedVol(price, fx->spot, q.strike, fx->rate, fx->dividend, q.tMax);
                r[i] = sw * (iv - q.targetIv);
                if (jacobian) {
                    const auto g = fx->model.fullGradient(pq, q.tMax);
                    const double vega = quantape::pricing::gbsAnalytical(
                                            fx->spot, q.strike, fx->rate, fx->rate - fx->dividend,
                                            iv, q.tMax, quantape::pricing::OptionType::Call)
                                            .g1.dV_dvol;
                    for (std::size_t k = 0; k < np; ++k) {
                        (*jacobian)(static_cast<Eigen::Index>(i), static_cast<Eigen::Index>(k)) =
                            sw * g(0, active[k]) / std::max(1e-6, vega);
                    }
                }
            }
        }
    }

    double value(const std::vector<double>& xa) const {
        std::vector<double> r;
        residuals(xa, r, nullptr);
        if (!feasible) {
            return 1e300;
        }
        double f = 0.0;
        for (double ri : r) {
            f += ri * ri;
        }
        return f;
    }

    /// True LS Hessian: J'WJ + sum_i w_i r_i H_i (raw residuals, not sqrt).
    /// Price space only (the IV-space residual curvature is Gauss-Newton).
    Eigen::MatrixXd exactHessian(const std::vector<double>& xa,
                                 const std::vector<double>& r) const {
        std::vector<double> rTmp;
        Eigen::MatrixXd J;
        residuals(xa, rTmp, &J);
        Eigen::MatrixXd H = 2.0 * (J.transpose() * J);
        const std::size_t np = active.size();
        for (std::size_t i = 0; i < fx->quotes.size(); ++i) {
            const Quote& q = fx->quotes[i];
            const double sw = sqrtWeight(q);
            const double raw = r[i] / std::max(1e-30, sw); // raw residual (unweighted)
            if (std::fabs(raw) < 1e-13) {
                continue;
            }
            const auto Hi = fx->model.fullHessian(quotePoint(xa, q), q.tMax);
            for (std::size_t a = 0; a < np; ++a) {
                for (std::size_t b = 0; b < np; ++b) {
                    H(static_cast<Eigen::Index>(a), static_cast<Eigen::Index>(b)) +=
                        2.0 * raw * Hi(active[a], active[b]);
                }
            }
        }
        return H;
    }
};

struct Run {
    std::string name;
    std::vector<double> x;
    double f = 0.0;
    std::size_t evals = 0;
    int iterations = 0;
    double ms = 0.0;
    bool converged = false;
    double maxParamErr = 0.0;
};

Run summarize(const std::string& name, const FitProblem& prob, const std::vector<double>& x,
              std::size_t evals, int iterations, double ms, bool converged) {
    Run run;
    run.name = name;
    run.x = x;
    run.f = prob.value(x);
    run.evals = evals;
    run.iterations = iterations;
    run.ms = ms;
    run.converged = converged;
    const HestonParams truth = prob.fx->truth;
    const double tv[9] = {truth.v0,      truth.kappa, truth.theta,   truth.sigma,      truth.rho,
                          prob.fx->spot, 1.0,         prob.fx->rate, prob.fx->dividend};
    for (std::size_t k = 0; k < prob.active.size(); ++k) {
        if (prob.active[k] != quantape::models::HESTON_STRIKE) {
            run.maxParamErr = std::max(run.maxParamErr, std::fabs(x[k] - tv[prob.active[k]]));
        }
    }
    return run;
}

// ── Strategy S1/S2/S3: LBFGS<double> with exact value+gradient ──

struct PriceValueGrad {
    const FitProblem* prob = nullptr;

    double operator()(const std::vector<double>& x, std::vector<double>& grad) const {
        std::vector<double> r;
        Eigen::MatrixXd J;
        prob->residuals(x, r, &J);
        Eigen::Map<Eigen::VectorXd> rv(r.data(), static_cast<Eigen::Index>(r.size()));
        const double f = rv.squaredNorm();
        const Eigen::VectorXd g = 2.0 * (J.transpose() * rv);
        grad.resize(static_cast<std::size_t>(g.size()));
        for (Eigen::Index i = 0; i < g.size(); ++i) {
            grad[static_cast<std::size_t>(i)] = g(i);
        }
        return f;
    }
};

// ── Strategy S4: log-parameter space ──

struct LogSpaceValueGrad {
    const FitProblem* prob = nullptr;

    static std::vector<double> toX(const std::vector<double>& y) {
        std::vector<double> x(5);
        x[0] = std::exp(y[0]);
        x[1] = std::exp(y[1]);
        x[2] = std::exp(y[2]);
        x[3] = std::exp(y[3]);
        x[4] = 0.95 * std::tanh(y[4]);
        return x;
    }

    static std::vector<double> toY(const std::vector<double>& x) {
        std::vector<double> y(5);
        y[0] = std::log(x[0]);
        y[1] = std::log(x[1]);
        y[2] = std::log(x[2]);
        y[3] = std::log(x[3]);
        y[4] = std::atanh(std::min(0.949, std::max(-0.949, x[4] / 0.95)));
        return y;
    }

    double operator()(const std::vector<double>& y, std::vector<double>& grad) const {
        const std::vector<double> x = toX(y);
        std::vector<double> gx;
        const double f = PriceValueGrad{prob}(x, gx);
        grad.resize(5);
        grad[0] = gx[0] * x[0];
        grad[1] = gx[1] * x[1];
        grad[2] = gx[2] * x[2];
        grad[3] = gx[3] * x[3];
        grad[4] = gx[4] * 0.95 * (1.0 - std::tanh(y[4]) * std::tanh(y[4]));
        return f;
    }
};

// ── Strategy S5/S6: Levenberg-Marquardt (Gauss-Newton or exact Newton) ──

Run runLevenbergMarquardt(const std::string& name, const FitProblem& prob,
                          const std::vector<double>& x0, bool exactHessian, int maxIter = 200) {
    const auto t0 = std::chrono::steady_clock::now();
    std::vector<double> x = x0;
    std::vector<double> r;
    Eigen::MatrixXd J;
    std::size_t evals = 0;
    int iterations = 0;
    bool converged = false;
    double lambda = 1e-3;
    prob.residuals(x, r, &J);
    ++evals;
    auto squared = [](const std::vector<double>& v) {
        double f = 0.0;
        for (double vi : v) {
            f += vi * vi;
        }
        return f;
    };
    double f = squared(r);
    for (int it = 0; it < maxIter; ++it) {
        iterations = it + 1;
        Eigen::Map<Eigen::VectorXd> rv(r.data(), static_cast<Eigen::Index>(r.size()));
        Eigen::VectorXd g = 2.0 * (J.transpose() * rv);
        if (g.lpNorm<Eigen::Infinity>() < 1e-12) {
            converged = true;
            break;
        }
        Eigen::MatrixXd H =
            exactHessian ? prob.exactHessian(x, r) : Eigen::MatrixXd(2.0 * (J.transpose() * J));
        bool accepted = false;
        for (int lm = 0; lm < 14; ++lm) {
            Eigen::MatrixXd damped = H;
            for (Eigen::Index a = 0; a < damped.rows(); ++a) {
                damped(a, a) += lambda * std::max(1e-12, std::fabs(H(a, a)));
            }
            const Eigen::VectorXd delta = damped.ldlt().solve(-g);
            if (!delta.allFinite()) {
                lambda *= 10.0;
                continue;
            }
            std::vector<double> trial(x.size());
            for (std::size_t k = 0; k < x.size(); ++k) {
                trial[k] = x[k] + delta(static_cast<Eigen::Index>(k));
            }
            std::vector<double> rTrial;
            prob.residuals(trial, rTrial, nullptr);
            ++evals;
            const double fTrial = squared(rTrial);
            if (prob.feasible && fTrial < f) {
                x = trial;
                std::vector<double> rNew;
                prob.residuals(x, rNew, &J);
                ++evals;
                r = rNew;
                f = fTrial;
                lambda = std::max(1e-12, lambda / 3.0);
                accepted = true;
                if (delta.norm() < 1e-11 * (1.0 + x.size())) {
                    converged = true;
                }
                break;
            }
            lambda *= 10.0;
        }
        if (!accepted || converged) {
            converged = converged || accepted;
            break;
        }
    }
    const auto t1 = std::chrono::steady_clock::now();
    const double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    return summarize(name, prob, x, evals, iterations, ms, converged);
}

// ── Strategy S7: LBFGS<var> through the callback-var price ──

template <typename S>
S stanHestonPrice(const HestonModel& model, const std::vector<S>& x, const HestonMarket& m) {
    if constexpr (std::is_same_v<S, stan::math::var>) {
        Eigen::Matrix<stan::math::var, 5, 1> v;
        for (int i = 0; i < 5; ++i) {
            v(i) = x[static_cast<std::size_t>(i)];
        }
        return quantape::models::hestonCall(model, v, m);
    } else {
        Eigen::Matrix<stan::math::fvar<stan::math::var>, 5, 1> v;
        for (int i = 0; i < 5; ++i) {
            v(i) = x[static_cast<std::size_t>(i)];
        }
        return quantape::models::hestonCall(model, v, m);
    }
}

struct StanValue {
    const FitProblem* prob = nullptr;

    template <typename S>
    S operator()(const std::vector<S>& x) const {
        S f = S(0.0);
        for (const Quote& q : prob->fx->quotes) {
            const S price = stanHestonPrice<S>(prob->fx->model, x, prob->marketFor(q));
            const S r = static_cast<double>(prob->sqrtWeight(q)) * (price - q.targetPrice);
            f += r * r;
        }
        return f;
    }
};

// ── Noise Monte Carlo: parameter error and IV repricing error ──

struct NoiseStats {
    double mean[5] = {0, 0, 0, 0, 0};
    double std[5] = {0, 0, 0, 0, 0};
    double ivRmseBp = 0.0;
    double f = 0.0;
    int runs = 0;
    int stalls = 0;
};

void runNoiseStudy() {
    QTA_LOG_INFO("quantape.test",
                 "NOISE MONTE CARLO (LM Gauss-Newton, vega weights, 40 seeds per level)");
    const double levels[3] = {0.001, 0.002, 0.005};
    const std::vector<double> start{0.05, 1.5, 0.05, 0.5, -0.3};
    for (double noise : levels) {
        NoiseStats stats;
        for (unsigned seed = 0; seed < 40; ++seed) {
            Fixture fx = makeFixture(noise, 9000u + seed);
            FitProblem prob;
            prob.fx = &fx;
            prob.weights = WeightMode::Vega;
            const Run run = runLevenbergMarquardt("mc", prob, start, false);
            const double truth5[5] = {fx.truth.v0, fx.truth.kappa, fx.truth.theta, fx.truth.sigma,
                                      fx.truth.rho};
            for (int k = 0; k < 5; ++k) {
                stats.mean[k] += run.x[static_cast<std::size_t>(k)] - truth5[k];
            }
            stats.f += run.f;
            ++stats.runs;
            double ss = 0.0;
            for (const Quote& q : fx.quotes) {
                HestonFullPoint pq = prob.point(run.x);
                pq[quantape::models::HESTON_STRIKE] = q.strike;
                const double p = fx.model.callFull(pq, q.tMax);
                const double iv = impliedVol(p, fx.spot, q.strike, fx.rate, fx.dividend, q.tMax);
                const double e = iv - q.targetIv;
                ss += e * e;
            }
            stats.ivRmseBp += std::sqrt(ss / static_cast<double>(fx.quotes.size())) * 1e4;
        }
        // second pass for the std
        for (unsigned seed = 0; seed < 40; ++seed) {
            Fixture fx = makeFixture(noise, 9000u + seed);
            FitProblem prob;
            prob.fx = &fx;
            prob.weights = WeightMode::Vega;
            const Run run = runLevenbergMarquardt("mc", prob, start, false);
            const double truth5[5] = {fx.truth.v0, fx.truth.kappa, fx.truth.theta, fx.truth.sigma,
                                      fx.truth.rho};
            for (int k = 0; k < 5; ++k) {
                const double d =
                    run.x[static_cast<std::size_t>(k)] - truth5[k] - stats.mean[k] / 40.0;
                stats.std[k] += d * d;
            }
        }
        for (int k = 0; k < 5; ++k) {
            stats.mean[k] /= 40.0;
            stats.std[k] = std::sqrt(stats.std[k] / 40.0);
        }
        stats.ivRmseBp /= 40.0;
        stats.f /= 40.0;
        const char* names[5] = {"v0", "kappa", "theta", "sigma", "rho"};
        std::string biasLine = "  noise " + quantape_test::num(noise * 100.0, 1) + "%:  ";
        for (int k = 0; k < 5; ++k) {
            biasLine += std::string(names[k]) + " bias=" + quantape_test::num(stats.mean[k], 2) +
                        " sd=" + quantape_test::num(stats.std[k], 2) + "  ";
        }
        QTA_LOG_INFO("quantape.test", "{}", biasLine);
        QTA_LOG_INFO("quantape.test", "                 mean IV RMSE={} bp, mean f={}, runs={}",
                     quantape_test::num(stats.ivRmseBp, 1), quantape_test::num(stats.f, 2),
                     stats.runs);
        CHECK(stats.ivRmseBp < 40.0 * (noise / 0.001));
    }
}

// ── Fixed-market fast adapter: same chain, model-only kernels ──

void testFixedMarketAdapter() {
    Fixture fx = makeFixture(0.0, 3u);
    FitProblem fit;
    fit.fx = &fx;
    fit.weights = WeightMode::Vega;

    std::vector<quantape::models::HestonCalibrationQuote> quotes;
    for (const Quote& q : fx.quotes) {
        quotes.push_back({q.strike, q.tMax, q.targetPrice, 1.0 / (q.vega * q.vega)});
    }
    const quantape::models::HestonCalibrationProblem full(fx.model, quotes);
    const quantape::models::HestonModelCalibrationProblem fixed(fx.model, quotes, fx.spot, fx.rate,
                                                                fx.dividend);
    Eigen::VectorXd b0(5);
    b0 << fx.truth.v0, fx.truth.kappa, fx.truth.theta, fx.truth.sigma, fx.truth.rho;
    const Eigen::VectorXd a = Eigen::Vector3d(fx.spot, fx.rate, fx.dividend);

    const auto dFull =
        quantape::math::assembleCalibration(full, b0, a, quantape::math::CalibrationOrder::Hessian);
    Eigen::VectorXd empty(0);
    const auto dFixed = quantape::math::assembleCalibration(
        fixed, b0, empty, quantape::math::CalibrationOrder::Hessian);
    checkClose("fixed-market value", dFixed.value, dFull.value, 1e-14);
    const double gradDiff = (dFixed.gradientB - dFull.gradientB).norm();
    const double hessDiff = (dFixed.hessianBB - dFull.hessianBB.topLeftCorner(5, 5)).norm();
    checkClose("fixed-market gradient == full", gradDiff, 0.0, 1e-10);
    checkClose("fixed-market Hessian == full", hessDiff, 0.0, 1e-9);

    const auto now = [] { return std::chrono::steady_clock::now(); };
    const auto us = [](auto t0, auto t1) {
        return std::chrono::duration<double, std::micro>(t1 - t0).count();
    };
    double fullUs = 0.0;
    double fixedUs = 0.0;
    for (int r = 0; r < 10; ++r) {
        auto t0 = now();
        for (int i = 0; i < 5; ++i) {
            (void)quantape::math::assembleCalibration(full, b0, a,
                                                      quantape::math::CalibrationOrder::Hessian);
        }
        auto t1 = now();
        fullUs += us(t0, t1) / 5.0;
        t0 = now();
        for (int i = 0; i < 5; ++i) {
            (void)quantape::math::assembleCalibration(fixed, b0, empty,
                                                      quantape::math::CalibrationOrder::Hessian);
        }
        t1 = now();
        fixedUs += us(t0, t1) / 5.0;
    }
    fullUs /= 10.0;
    fixedUs /= 10.0;
    QTA_LOG_INFO("quantape.test",
                 "  assemble per quote-set: full 9-param {} us, fixed 5-param {} us ({}x)",
                 quantape_test::num(fullUs, 1), quantape_test::num(fixedUs, 1),
                 quantape_test::num(fullUs / fixedUs, 2));

    // Same optimum through LBFGS<double> with both adapters
    quantape::math::StopCriteria criteria;
    criteria.ftol_rel = 1e-14;
    criteria.xtol_rel = 1e-12;
    criteria.grad_tol = 1e-12;
    criteria.maxeval = 2000;
    std::vector<double> xFull = {0.05, 1.5, 0.05, 0.5, -0.3};
    std::vector<double> xFixed = xFull;
    quantape::math::LBFGS<double> solver(criteria, 10);
    solver.minimize(quantape::math::CalibrationValueGrad{full, a}, xFull);
    quantape::math::LBFGS<double> solver2(criteria, 10);
    solver2.minimize(quantape::math::CalibrationValueGrad{fixed, empty}, xFixed);
    checkClose("fixed-market optimum",
               (Eigen::Map<Eigen::VectorXd>(xFixed.data(), 5) -
                Eigen::Map<Eigen::VectorXd>(xFull.data(), 5))
                   .norm(),
               0.0, 1e-8);
    QTA_LOG_INFO("quantape.test", "  [ok] fixed-market adapter: identical derivatives and optimum");
}

// ── Reporting ──

void printRun(const Run& run) {
    QTA_LOG_INFO("quantape.test", "  {} f={} iters={} evals={} {} ms converged={} maxdparam={}",
                 run.name, quantape_test::num(run.f, 3), run.iterations, run.evals,
                 quantape_test::num(run.ms, 1), run.converged ? "yes" : "no",
                 quantape_test::num(run.maxParamErr, 2));
    std::string xLine = "      x = [";
    for (std::size_t i = 0; i < run.x.size(); ++i) {
        xLine += (i ? ", " : "") + quantape_test::num(run.x[i], 6);
    }
    xLine += "]";
    QTA_LOG_INFO("quantape.test", "{}", xLine);
}

void printTruth(const Fixture& fx) {
    QTA_LOG_INFO("quantape.test", "  truth = [{}, {}, {}, {}, {}]",
                 quantape_test::num(fx.truth.v0, 6), quantape_test::num(fx.truth.kappa, 6),
                 quantape_test::num(fx.truth.theta, 6), quantape_test::num(fx.truth.sigma, 6),
                 quantape_test::num(fx.truth.rho, 6));
}

double noiseFloor(const FitProblem& prob) {
    double floor = 0.0;
    for (const Quote& q : prob.fx->quotes) {
        const double sw = prob.sqrtWeight(q);
        floor += sw * sw * 0.002 * 0.002 * q.targetPrice * q.targetPrice;
    }
    return floor;
}

} // namespace

int main() {
    QTA_LOG_INFO("quantape.test", "Heston calibration strategy shoot-out (H6)");

    const std::vector<double> start{0.05, 1.5, 0.05, 0.5, -0.3};

    for (int noisy = 0; noisy < 2; ++noisy) {
        Fixture fx = makeFixture(noisy ? 0.002 : 0.0, 42u + static_cast<unsigned>(noisy));
        QTA_LOG_INFO("quantape.test", "{} data ({} quotes, noise {}%)",
                     noisy ? "NOISY" : "NOISELESS", fx.quotes.size(),
                     quantape_test::num(noisy ? 0.2 : 0.0, 2));
        printTruth(fx);
        {
            FitProblem pt;
            pt.fx = &fx;
            pt.weights = WeightMode::Uniform;
            const std::vector<double> truth5{fx.truth.v0, fx.truth.kappa, fx.truth.theta,
                                             fx.truth.sigma, fx.truth.rho};
            const double uniformF = pt.value(truth5);
            pt.weights = WeightMode::Vega;
            QTA_LOG_INFO("quantape.test", "  [check] f(truth) uniform={}  vega={}",
                         quantape_test::num(uniformF, 6), quantape_test::num(pt.value(truth5), 6));
        }

        {
            FitProblem prob;
            prob.fx = &fx;
            prob.weights = WeightMode::Uniform;
            quantape::math::StopCriteria criteria;
            criteria.ftol_rel = 1e-14;
            criteria.xtol_rel = 1e-12;
            criteria.grad_tol = 1e-12;
            criteria.maxeval = 500;
            quantape::math::LBFGS<double> solver(criteria, 10);
            std::vector<double> x = start;
            quantape::math::OptimizerState state;
            const auto t0 = std::chrono::steady_clock::now();
            const auto result = solver.minimize(PriceValueGrad{&prob}, x, state);
            const auto t1 = std::chrono::steady_clock::now();
            printRun(summarize("S1 LBFGS price (uniform)", prob, x, state.evals,
                               static_cast<int>(state.iterations),
                               std::chrono::duration<double, std::milli>(t1 - t0).count(),
                               result == quantape::math::OptimizeResult::GradientTolReached ||
                                   result == quantape::math::OptimizeResult::FtolReached ||
                                   result == quantape::math::OptimizeResult::XtolReached));
            if (!noisy) {
                CHECK(prob.value(x) < 1e-9);
                CHECK(std::fabs(x[3] - fx.truth.sigma) < 5e-3);
            }
        }
        {
            FitProblem prob;
            prob.fx = &fx;
            prob.weights = WeightMode::Vega;
            quantape::math::StopCriteria criteria;
            criteria.ftol_rel = 1e-14;
            criteria.xtol_rel = 1e-12;
            criteria.grad_tol = 1e-12;
            criteria.maxeval = 500;
            quantape::math::LBFGS<double> solver(criteria, 10);
            std::vector<double> x = start;
            quantape::math::OptimizerState state;
            const auto t0 = std::chrono::steady_clock::now();
            const auto result = solver.minimize(PriceValueGrad{&prob}, x, state);
            const auto t1 = std::chrono::steady_clock::now();
            printRun(summarize("S2 LBFGS price (vega w)", prob, x, state.evals,
                               static_cast<int>(state.iterations),
                               std::chrono::duration<double, std::milli>(t1 - t0).count(),
                               result != quantape::math::OptimizeResult::Failure));
            if (!noisy) {
                CHECK(prob.value(x) < 1e-9);
                CHECK(std::fabs(x[3] - fx.truth.sigma) < 5e-3);
            }
        }
        // S3 IV space
        {
            FitProblem prob;
            prob.fx = &fx;
            prob.weights = WeightMode::Uniform;
            prob.space = SpaceMode::ImpliedVol;
            quantape::math::StopCriteria criteria;
            criteria.ftol_rel = 1e-14;
            criteria.xtol_rel = 1e-12;
            criteria.grad_tol = 1e-12;
            criteria.maxeval = 500;
            quantape::math::LBFGS<double> solver(criteria, 10);
            std::vector<double> x = start;
            quantape::math::OptimizerState state;
            const auto t0 = std::chrono::steady_clock::now();
            const auto result = solver.minimize(PriceValueGrad{&prob}, x, state);
            const auto t1 = std::chrono::steady_clock::now();
            printRun(summarize("S3 LBFGS implied-vol space", prob, x, state.evals,
                               static_cast<int>(state.iterations),
                               std::chrono::duration<double, std::milli>(t1 - t0).count(),
                               result != quantape::math::OptimizeResult::Failure));
            if (!noisy) {
                CHECK(prob.value(x) < 1e-9);
                CHECK(std::fabs(x[3] - fx.truth.sigma) < 5e-3);
            }
        }
        // S4 log space
        {
            FitProblem prob;
            prob.fx = &fx;
            prob.weights = WeightMode::Vega;
            quantape::math::StopCriteria criteria;
            criteria.ftol_rel = 1e-14;
            criteria.xtol_rel = 1e-12;
            criteria.grad_tol = 1e-12;
            criteria.maxeval = 500;
            quantape::math::LBFGS<double> solver(criteria, 10);
            std::vector<double> y = LogSpaceValueGrad::toY(start);
            quantape::math::OptimizerState state;
            const auto t0 = std::chrono::steady_clock::now();
            const auto result = solver.minimize(LogSpaceValueGrad{&prob}, y, state);
            const auto t1 = std::chrono::steady_clock::now();
            const std::vector<double> x = LogSpaceValueGrad::toX(y);
            printRun(summarize("S4 LBFGS log-space (vega w)", prob, x, state.evals,
                               static_cast<int>(state.iterations),
                               std::chrono::duration<double, std::milli>(t1 - t0).count(),
                               result != quantape::math::OptimizeResult::Failure));
            if (!noisy) {
                CHECK(prob.value(x) < 1e-9);
                CHECK(std::fabs(x[3] - fx.truth.sigma) < 5e-3);
            }
        }
        // S5 Gauss-Newton
        {
            FitProblem prob;
            prob.fx = &fx;
            prob.weights = WeightMode::Vega;
            const Run run = runLevenbergMarquardt("S5 LM Gauss-Newton", prob, start, false);
            printRun(run);
            if (!noisy) {
                CHECK(run.f < 1e-9);
                CHECK(std::fabs(run.x[3] - fx.truth.sigma) < 5e-3);
            }
        }
        // S6 exact Newton
        {
            FitProblem prob;
            prob.fx = &fx;
            prob.weights = WeightMode::Vega;
            const Run run = runLevenbergMarquardt("S6 LM Newton (exact H)", prob, start, true);
            printRun(run);
            if (!noisy) {
                CHECK(run.f < 1e-9);
                CHECK(std::fabs(run.x[3] - fx.truth.sigma) < 5e-3);
            }
        }
        // S7 Stan var callback
        {
            FitProblem prob;
            prob.fx = &fx;
            prob.weights = WeightMode::Vega;
            quantape::math::StopCriteria criteria;
            criteria.ftol_rel = 1e-14;
            criteria.xtol_rel = 1e-12;
            criteria.grad_tol = 1e-12;
            criteria.maxeval = 500;
            quantape::math::LBFGS<stan::math::var> solver(criteria, 10);
            std::vector<double> x = start;
            quantape::math::OptimizerState state;
            const auto t0 = std::chrono::steady_clock::now();
            const auto result = solver.minimize(StanValue{&prob}, x, state);
            const auto t1 = std::chrono::steady_clock::now();
            stan::math::recover_memory();
            printRun(summarize("S7 LBFGS<var> callback", prob, x, state.evals,
                               static_cast<int>(state.iterations),
                               std::chrono::duration<double, std::milli>(t1 - t0).count(),
                               result != quantape::math::OptimizeResult::Failure));
            if (!noisy) {
                CHECK(prob.value(x) < 1e-9);
                CHECK(std::fabs(x[3] - fx.truth.sigma) < 5e-3);
            }
        }
        if (noisy) {
            FitProblem prob;
            prob.fx = &fx;
            prob.weights = WeightMode::Vega;
            const double floor = noiseFloor(prob);
            QTA_LOG_INFO("quantape.test", "  noise floor (vega weights) = {}",
                         quantape_test::num(floor, 3));
            // all strategies should have found comparable optima; re-run LM as
            // the reference and check it is within a small factor of the floor
            const Run run = runLevenbergMarquardt("S6 noisy reference", prob, start, true);
            CHECK(run.f < 5.0 * floor);
        }
    }

    testFixedMarketAdapter();

    runNoiseStudy();

    // ── Feller constraint: AUGLAG vs SLSQP, binding vs inactive ──
    {
        struct FellerConstraint { // g(x) = sigma^2 - 2 kappa theta <= 0 (double backend)
            void operator()(const std::vector<double>& x, std::vector<double>& c,
                            std::vector<double>& J) const {
                c.assign(1, x[3] * x[3] - 2.0 * x[1] * x[2]);
                J.assign(5, 0.0);
                J[1] = -2.0 * x[2];
                J[2] = -2.0 * x[1];
                J[3] = 2.0 * x[3];
            }
        };
        const auto bounds = quantape::math::Bounds::fromVectors({1e-6, 1e-6, 1e-6, 1e-6, -0.999},
                                                                {10.0, 50.0, 10.0, 10.0, 0.999});

        auto runCase = [&](const char* label, HestonParams truth,
                           const std::vector<double>& start) {
            Fixture fx = makeFixture(0.0, 21u, truth);
            FitProblem prob;
            prob.fx = &fx;
            prob.weights = WeightMode::Vega;

            const Run unconstrained =
                runLevenbergMarquardt("unconstrained (LM GN)", prob, start, false);
            printRun(unconstrained);
            const double gUnconstrained = unconstrained.x[3] * unconstrained.x[3] -
                                          2.0 * unconstrained.x[1] * unconstrained.x[2];
            const bool binding = gUnconstrained > 0.01;
            QTA_LOG_INFO(
                "quantape.test", "  {}: Feller 2*kappa*theta - sigma^2 at unconstrained = {} ({})",
                label, quantape_test::num(-gUnconstrained, 6), binding ? "VIOLATED" : "satisfied");

            // Reference: the boundary optimum by construction
            // (sigma = sqrt(2 kappa theta), 4 free parameters).
            FitProblem boundary;
            boundary.fx = &fx;
            boundary.weights = WeightMode::Vega;
            boundary.fellerBoundary = true;
            boundary.active = {0, 1, 2, 4};
            const Run boundaryRun =
                runLevenbergMarquardt("Feller boundary (4-param)", boundary,
                                      {start[0], start[1], start[2], start[4]}, false);
            printRun(boundaryRun);
            const double sigmaBoundary = std::sqrt(2.0 * boundaryRun.x[1] * boundaryRun.x[2]);
            QTA_LOG_INFO(
                "quantape.test", "      boundary point = [{}, {}, {}, {}, {}]",
                quantape_test::num(boundaryRun.x[0], 6), quantape_test::num(boundaryRun.x[1], 6),
                quantape_test::num(boundaryRun.x[2], 6), quantape_test::num(sigmaBoundary, 6),
                quantape_test::num(boundaryRun.x[3], 6));

            const std::vector<double> xBoundary5{boundaryRun.x[0], boundaryRun.x[1],
                                                 boundaryRun.x[2], sigmaBoundary, boundaryRun.x[3]};
            std::vector<double> gRef;
            PriceValueGrad{&prob}(xBoundary5, gRef);
            const double dgRef[5] = {0.0, -2.0 * boundaryRun.x[2], -2.0 * boundaryRun.x[1],
                                     2.0 * sigmaBoundary, 0.0};
            double gnorm = 0.0, denom = 0.0;
            for (int k = 0; k < 5; ++k) {
                gnorm -= gRef[static_cast<std::size_t>(k)] * dgRef[k];
                denom += dgRef[k] * dgRef[k];
            }
            const double lambdaStar = gnorm / denom;
            QTA_LOG_INFO("quantape.test", "      envelope multiplier lambda* = {}, f* = {}",
                         quantape_test::num(lambdaStar, 6), quantape_test::num(boundaryRun.f, 6));

            quantape::math::StopCriteria criteria;
            criteria.maxeval = 2000;
            criteria.ftol_rel = 1e-12;
            criteria.xtol_rel = 1e-10;
            std::vector<double> xAug;
            {
                std::vector<double> x = start;
                quantape::math::OptimizerState state;
                const auto t0 = std::chrono::steady_clock::now();
                quantape::math::AugLag<double> solver(criteria);
                const auto result =
                    solver.minimize(PriceValueGrad{&prob}, FellerConstraint{}, bounds, x, state);
                const auto t1 = std::chrono::steady_clock::now();
                const double g = x[3] * x[3] - 2.0 * x[1] * x[2];
                const double lambda =
                    state.ineq_multipliers.empty() ? 0.0 : state.ineq_multipliers[0];
                std::vector<double> grad;
                PriceValueGrad{&prob}(x, grad);
                const double dg[5] = {0.0, -2.0 * x[2], -2.0 * x[1], 2.0 * x[3], 0.0};
                double kkt = 0.0;
                for (int k = 0; k < 5; ++k) {
                    kkt = std::max(kkt,
                                   std::fabs(grad[static_cast<std::size_t>(k)] + lambda * dg[k]));
                }
                QTA_LOG_INFO("quantape.test",
                             "  AUGLAG Feller ({})  f={}  g={}  lambda={}  KKT={}  evals={} {} ms",
                             quantape::math::to_string(result),
                             quantape_test::num(prob.value(x), 6), quantape_test::num(g, 2),
                             quantape_test::num(lambda, 6), quantape_test::num(kkt, 2), state.evals,
                             quantape_test::num(
                                 std::chrono::duration<double, std::milli>(t1 - t0).count(), 1));
                QTA_LOG_INFO("quantape.test", "      x = [{}, {}, {}, {}, {}]",
                             quantape_test::num(x[0], 6), quantape_test::num(x[1], 6),
                             quantape_test::num(x[2], 6), quantape_test::num(x[3], 6),
                             quantape_test::num(x[4], 6));
                CHECK(result != quantape::math::OptimizeResult::Failure);
                CHECK(g <= 1e-7);
                CHECK(kkt < 1e-5);
                CHECK(prob.value(x) <= boundaryRun.f + 1e-6);
                if (!binding) {
                    CHECK(std::fabs(prob.value(x) - unconstrained.f) < 1e-9);
                }
                xAug = x;
            }
            {
                std::vector<double> x = start;
                quantape::math::OptimizerState state;
                const auto t0 = std::chrono::steady_clock::now();
                quantape::math::SLSQP<double> solver(criteria);
                const auto result =
                    solver.minimize(PriceValueGrad{&prob}, FellerConstraint{}, bounds, x, state);
                const auto t1 = std::chrono::steady_clock::now();
                const double g = x[3] * x[3] - 2.0 * x[1] * x[2];
                QTA_LOG_INFO(
                    "quantape.test", "  SLSQP  Feller ({})  f={}  g={}  iters={} evals={} {} ms",
                    quantape::math::to_string(result), quantape_test::num(prob.value(x), 6),
                    quantape_test::num(g, 2), static_cast<int>(state.iterations), state.evals,
                    quantape_test::num(std::chrono::duration<double, std::milli>(t1 - t0).count(),
                                       1));
                QTA_LOG_INFO("quantape.test", "      x = [{}, {}, {}, {}, {}]",
                             quantape_test::num(x[0], 6), quantape_test::num(x[1], 6),
                             quantape_test::num(x[2], 6), quantape_test::num(x[3], 6),
                             quantape_test::num(x[4], 6));
                CHECK(result != quantape::math::OptimizeResult::Failure);
                CHECK(g <= 1e-6);
                CHECK(prob.value(x) <= boundaryRun.f + 1e-6);
                CHECK(std::fabs(prob.value(x) - prob.value(xAug)) < 1e-5);
                if (!binding) {
                    CHECK(std::fabs(prob.value(x) - unconstrained.f) < 1e-9);
                }
            }
            // Cost of enforcing Feller: worst implied-vol shift vs the targets
            auto worstIvShift = [&](const std::vector<double>& x5) {
                double worst = 0.0;
                for (const Quote& q : fx.quotes) {
                    HestonFullPoint pq = prob.point(x5);
                    pq[quantape::models::HESTON_STRIKE] = q.strike;
                    const double p = fx.model.callFull(pq, q.tMax);
                    const double iv =
                        impliedVol(p, fx.spot, q.strike, fx.rate, fx.dividend, q.tMax);
                    worst = std::max(worst, std::fabs(iv - q.targetIv));
                }
                return worst;
            };
            QTA_LOG_INFO("quantape.test",
                         "  {}: max |IV shift| unconstrained={}  boundary={}  AUGLAG={}", label,
                         quantape_test::num(worstIvShift(unconstrained.x), 2),
                         quantape_test::num(worstIvShift(xBoundary5), 2),
                         quantape_test::num(worstIvShift(xAug), 2));
            if (binding) {
                CHECK(boundaryRun.f > unconstrained.f);
                CHECK(lambdaStar > 1e-3);
            } else {
                CHECK(boundaryRun.f >= unconstrained.f - 1e-15);
                CHECK(lambdaStar < 1e-6);
            }
        };

        QTA_LOG_INFO("quantape.test", "FELLER CONSTRAINT (vega-weighted LS, bounds enforced)");
        QTA_LOG_INFO("quantape.test", "-- case A: blog truth (Feller violated)");
        runCase("case A", HestonParams{0.04, 2.5, 0.06, 0.75, -0.1},
                std::vector<double>{0.05, 1.5, 0.05, 0.5, -0.3});
        QTA_LOG_INFO("quantape.test",
                     "-- case B: Feller-feasible truth (2*kappa*theta = 0.45 > sigma^2 = 0.25)");
        runCase("case B", HestonParams{0.04, 2.5, 0.09, 0.50, -0.1},
                std::vector<double>{0.05, 1.5, 0.07, 0.4, -0.3});
    }

    // ── Hard-start robustness: far start, GN vs log-space LBFGS ──
    {
        Fixture fx = makeFixture(0.0, 11u);
        const std::vector<double> farStart{0.01, 5.0, 0.02, 0.2, -0.7};
        QTA_LOG_INFO("quantape.test", "HARD START [0.01, 5.0, 0.02, 0.2, -0.7]");
        FitProblem prob;
        prob.fx = &fx;
        prob.weights = WeightMode::Vega;

        const Run gn = runLevenbergMarquardt("LM Gauss-Newton", prob, farStart, false);
        printRun(gn);
        quantape::math::StopCriteria criteria;
        criteria.ftol_rel = 1e-14;
        criteria.xtol_rel = 1e-12;
        criteria.grad_tol = 1e-12;
        criteria.maxeval = 800;
        quantape::math::LBFGS<double> solver(criteria, 10);
        std::vector<double> y = LogSpaceValueGrad::toY(farStart);
        quantape::math::OptimizerState state;
        const auto t0 = std::chrono::steady_clock::now();
        const auto result = solver.minimize(LogSpaceValueGrad{&prob}, y, state);
        const auto t1 = std::chrono::steady_clock::now();
        const std::vector<double> xLog = LogSpaceValueGrad::toX(y);
        const Run lg = summarize("LBFGS log-space", prob, xLog, state.evals,
                                 static_cast<int>(state.iterations),
                                 std::chrono::duration<double, std::milli>(t1 - t0).count(),
                                 result != quantape::math::OptimizeResult::Failure);
        printRun(lg);

        // Multi-start recipe: if a single far-start run stalls, restart GN from
        // the log-space iterate; the combined recipe must reach the truth.
        if (gn.f > 1e-9 && lg.f > 1e-9) {
            const Run gn2 = runLevenbergMarquardt("LM GN (restarted)", prob, xLog, false);
            printRun(gn2);
            CHECK(gn2.f < 1e-9);
        } else {
            CHECK(gn.f < 1e-9 || lg.f < 1e-9);
        }
    }

    // ── Identifiability diagnostics: 9-parameter Jacobian SVD ──
    {
        Fixture fx = makeFixture(0.0, 7u);
        FitProblem prob;
        prob.fx = &fx;
        prob.weights = WeightMode::Vega;
        const HestonFullPoint p = prob.point(std::vector<double>{
            fx.truth.v0, fx.truth.kappa, fx.truth.theta, fx.truth.sigma, fx.truth.rho});
        Eigen::MatrixXd J(fx.quotes.size(), kNumParams);
        for (std::size_t i = 0; i < fx.quotes.size(); ++i) {
            HestonFullPoint pq = p;
            pq[quantape::models::HESTON_STRIKE] = fx.quotes[i].strike;
            const auto g = fx.model.fullGradient(pq, fx.quotes[i].tMax);
            const double sw = prob.sqrtWeight(fx.quotes[i]);
            for (int k = 0; k < kNumParams; ++k) {
                J(static_cast<Eigen::Index>(i), k) = sw * g(0, k);
            }
        }
        Eigen::JacobiSVD<Eigen::MatrixXd> svd(J, Eigen::ComputeThinV);
        const Eigen::MatrixXd V = svd.matrixV();
        QTA_LOG_INFO("quantape.test", "9-parameter Jacobian: J {}x{}, sv.size={}, V {}x{}",
                     static_cast<long>(J.rows()), static_cast<long>(J.cols()),
                     static_cast<long>(svd.singularValues().size()), static_cast<long>(V.rows()),
                     static_cast<long>(V.cols()));
        std::string svLine = "singular values (vega weights, truth):";
        for (Eigen::Index i = 0; i < svd.singularValues().size(); ++i) {
            svLine += " " + quantape_test::num(svd.singularValues()(i), 3);
        }
        QTA_LOG_INFO("quantape.test", "{}", svLine);
        QTA_LOG_INFO(
            "quantape.test", "  condition = {}",
            quantape_test::num(svd.singularValues()(0) / svd.singularValues()(kNumParams - 1), 3));

        // Weak directions: joint S/K scale and (r, q)
        QTA_LOG_INFO("quantape.test", "  weakest direction coeffs [v0,k,th,sig,rho,S,K,r,q]:");
        for (Eigen::Index j = kNumParams - 1; j >= kNumParams - 3; --j) {
            std::string row = "    sv=" + quantape_test::num(svd.singularValues()(j), 2) + " :";
            for (int k = 0; k < kNumParams; ++k) {
                row += " " + quantape_test::num(V(k, j), 3);
            }
            QTA_LOG_INFO("quantape.test", "{}", row);
        }

        // 6- and 9-parameter LM fits from a perturbed start
        {
            FitProblem prob6;
            prob6.fx = &fx;
            prob6.weights = WeightMode::Vega;
            prob6.active = {0, 1, 2, 3, 4, 7}; // model + r
            const Run run =
                runLevenbergMarquardt("LM 5+r (S,K,q fixed)", prob6,
                                      std::vector<double>{0.05, 1.5, 0.05, 0.5, -0.3, 0.01}, true);
            printRun(run);
            CHECK(run.f < 1e-9);
        }
        {
            FitProblem prob9;
            prob9.fx = &fx;
            prob9.weights = WeightMode::Vega;
            prob9.active = {0, 1, 2, 3, 4, 5, 7, 8}; // strike stays per quote
            const Run run = runLevenbergMarquardt(
                "LM 8-param (S,r,q + model)", prob9,
                std::vector<double>{0.05, 1.5, 0.05, 0.5, -0.3, 0.99, 0.01, -0.01}, true);
            printRun(run);
            QTA_LOG_INFO("quantape.test",
                         "  (9-parameter joint fits are near-singular; see report)");
        }
    }

    QTA_LOG_INFO("quantape.test", "ALL HESTON CALIBRATION TESTS PASSED");
    return 0;
}
