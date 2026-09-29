// test_heston_ift.cpp — H6: IFT/KKT risk hookup for the Heston calibration
//
// The generic layer (`calibration/CalibrationProblem.h`) is exercised
// end to end for two models through the same concept:
//
//   Heston: b = [v0, kappa, theta, sigma, rho], a = [spot, rate, dividend],
//           Feller inequality exposed;
//   Black-Scholes (vol): b = [sigma], a = [spot, rate, dividend], no
//           constraints.
//
// Gates (all against bump-and-recalibrate finite differences):
//   1. assembleCalibration/`calibrationIft`: db/da at the unconstrained LS
//      optimum from two calibrators (LBFGS<double> and SLSQP<double>)
//   2. Feller-constrained optimum (AugLag<double>): db/da and dlambda/da
//      from the KKT system of the generic layer
//   3. downstream product risk dV/da = (dV/db)(db/da) vs reprice-after-bump
//   4. KKT route vs instrument route on the same assembled Jacobians
//   5. the BS adapter through the identical generic calls (model genericity)

#include "quantape/math/StanMath.h"

#include "quantape/calibration/CalibrationChain.h"
#include "quantape/calibration/CalibrationProblem.h"
#include "quantape/calibration/HestonCalibration.h"
#include "quantape/calibration/ImplicitFunction.h"
#include "quantape/math/Optimization/AugLag.h"
#include "quantape/math/Optimization/LBFGS.h"
#include "quantape/math/Optimization/OptimizerStanPrimitives.h"
#include "quantape/math/Optimization/SLSQP.h"
#include "quantape/models/HestonModel.h"
#include "quantape/pricing/BlackScholes.h"

#include <Eigen/Dense>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

#include "quantape/log/Log.h"
#include "quantape/util/Check.h"
using quantape::util::checkClose;

using quantape::math::Bounds;
using quantape::math::CalibrationInequalityConstraint;
using quantape::math::CalibrationValueGrad;
using quantape::math::OptimizerState;
using quantape::math::StopCriteria;
using quantape::models::HestonCalibrationProblem;
using quantape::models::HestonCalibrationQuote;
using quantape::models::HestonMarket;
using quantape::models::HestonModel;

namespace {


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

// ── Heston fixture ──

constexpr int kNB = 5;
constexpr int kNA = 3;

struct HestonFixture {
    HestonModel model;
    HestonCalibrationProblem problem{model, {}};
    Eigen::VectorXd a;
    std::vector<HestonCalibrationQuote> quotes;
};

HestonFixture makeHestonFixture() {
    HestonFixture fx;
    fx.problem = HestonCalibrationProblem(fx.model, {});
    const quantape::models::HestonParams truth{0.04, 2.5, 0.06, 0.75, -0.1};
    const double strikes[5] = {0.8, 0.9, 1.0, 1.1, 1.25};
    const double maturities[2] = {0.5, 2.0};
    for (double t : maturities) {
        for (double k : strikes) {
            const HestonMarket market{1.0, k, 0.0, 0.0, t};
            const double target = fx.model.call(truth, market);
            const double iv = impliedVol(target, 1.0, k, 0.0, 0.0, t);
            const double vega = quantape::pricing::gbsAnalytical(
                                    1.0, k, 0.0, 0.0, iv, t, quantape::pricing::OptionType::Call)
                                    .g1.dV_dvol;
            fx.quotes.push_back(HestonCalibrationQuote{k, t, target, 1.0 / (vega * vega)});
        }
    }
    fx.problem = HestonCalibrationProblem(fx.model, fx.quotes);
    fx.a.resize(kNA);
    fx.a << 1.0, 0.0, 0.0;
    return fx;
}

std::vector<double> asVector(const Eigen::VectorXd& v) {
    return std::vector<double>(v.data(), v.data() + v.size());
}

Eigen::VectorXd owning(const std::vector<double>& v) {
    return Eigen::Map<const Eigen::VectorXd>(v.data(), static_cast<Eigen::Index>(v.size()));
}

StopCriteria tightCriteria(int maxeval = 2000) {
    StopCriteria criteria;
    criteria.ftol_rel = 1e-14;
    criteria.xtol_rel = 1e-12;
    criteria.grad_tol = 1e-12;
    criteria.maxeval = maxeval;
    return criteria;
}

Eigen::VectorXd calibrateUnconstrained(const HestonCalibrationProblem& prob,
                                       const Eigen::VectorXd& a, const std::vector<double>& start) {
    quantape::math::LBFGS<double> solver(tightCriteria(), 10);
    std::vector<double> x = start;
    const auto result = solver.minimize(CalibrationValueGrad{prob, a}, x);
    CHECK(result != quantape::math::OptimizeResult::Failure);
    return owning(x);
}

Eigen::VectorXd calibrateConstrained(const HestonCalibrationProblem& prob, const Eigen::VectorXd& a,
                                     const std::vector<double>& start,
                                     std::vector<double>* lambda = nullptr) {
    const auto bounds =
        Bounds::fromVectors({1e-6, 1e-6, 1e-6, 1e-6, -0.999}, {10.0, 50.0, 10.0, 10.0, 0.999});
    // Start from the unconstrained optimum of the bumped problem so the
    // constrained solver must project onto the active boundary (a warm start
    // on the boundary can satisfy SLSQP's stationarity test immediately and
    // skip the small optimum shift).
    std::vector<double> x = start;
    {
        quantape::math::LBFGS<double> inner(tightCriteria(), 10);
        inner.minimize(CalibrationValueGrad{prob, a}, x);
    }
    quantape::math::SLSQP<double> solver(tightCriteria());
    OptimizerState state;
    const auto result =
        solver.minimize(CalibrationValueGrad{prob, a},
                        CalibrationInequalityConstraint<HestonCalibrationProblem>{&prob},
                        quantape::math::NoConstraint{}, bounds, x, state);
    CHECK(result != quantape::math::OptimizeResult::Failure);
    if (lambda) {
        *lambda =
            state.ineq_multipliers.empty() ? std::vector<double>{0.0} : state.ineq_multipliers;
    }
    return owning(x);
}

// ── Black-Scholes adapter (same generic concept, no constraints) ──

class BsCalibrationProblem {
public:
    struct Quote {
        double strike;
        double tMax;
        double target;
        double weight;
    };

    BsCalibrationProblem(std::vector<Quote> quotes) : quotes_(std::move(quotes)) {}

    std::size_t numModelParams() const { return 1; }
    std::size_t numMarketParams() const { return 3; }
    std::size_t numQuotes() const { return quotes_.size(); }
    std::size_t numInequalities() const { return 0; }

    void fillResidual(std::size_t i, const Eigen::VectorXd& b, const Eigen::VectorXd& a,
                      quantape::math::CalibrationOrder order,
                      quantape::math::CalibrationResidualBlock& out) const {
        const Quote& q = quotes_[i];
        const double vol = b(0);
        const double S = a(0), r = a(1), div = a(2);
        const double sw = std::sqrt(q.weight);
        const auto res = quantape::pricing::gbsAnalytical(S, q.strike, r, r - div, vol, q.tMax,
                                                          quantape::pricing::OptionType::Call);
        out.value = sw * (res.price - q.target);
        if (order >= quantape::math::CalibrationOrder::Gradient) {
            out.gradientB.resize(1);
            out.gradientA.resize(3);
            out.gradientB(0) = sw * res.g1.dV_dvol;
            out.gradientA(0) = sw * res.g1.dV_dS;
            out.gradientA(1) = sw * (res.g1.dV_drDisc + res.g1.dV_db);
            out.gradientA(2) = sw * (-res.g1.dV_db);
        }
        if (order >= quantape::math::CalibrationOrder::Hessian) {
            out.hessianBB.resize(1, 1);
            out.hessianBA.resize(1, 3);
            out.hessianBB(0, 0) = sw * res.g2.d2V_dvol2;
            out.hessianBA(0, 0) = sw * res.g2.d2V_dS_dvol;
            out.hessianBA(0, 1) = sw * (res.g2.d2V_drDisc_dvol + res.g2.d2V_db_dvol);
            out.hessianBA(0, 2) = sw * (-res.g2.d2V_db_dvol);
        }
    }

    void fillInequality(std::size_t, const Eigen::VectorXd&,
                        quantape::math::CalibrationInequalityBlock& out) const {
        out.value = 0.0;
        out.gradientB.resize(0);
        out.hessianBB.resize(0, 0);
    }

private:
    std::vector<Quote> quotes_;
};

// ── Feller-boundary parameterization (sigma = sqrt(2 kappa theta)) ──
//
// Independent, well-conditioned reference for the constrained IFT: the
// boundary surface has 4 free parameters, so bump-recalibrate finite
// differences avoid the cond(J) ~ 1e7 sensitivity of the constrained
// interior solve.
class FellerBoundaryProblem {
public:
    FellerBoundaryProblem(const quantape::models::HestonModelCalibrationProblem& inner)
        : m_inner(&inner) {}

    std::size_t numModelParams() const { return 4; }
    std::size_t numMarketParams() const { return 0; }
    std::size_t numQuotes() const { return m_inner->numQuotes(); }
    std::size_t numInequalities() const { return 0; }

    static std::vector<double> expand(const Eigen::VectorXd& b4) {
        const double kappa = b4(1);
        const double theta = b4(2);
        const double sigma = std::sqrt(2.0 * kappa * theta);
        return {b4(0), kappa, theta, sigma, b4(3)};
    }

    void fillResidual(std::size_t i, const Eigen::VectorXd& b4, const Eigen::VectorXd&,
                      quantape::math::CalibrationOrder order,
                      quantape::math::CalibrationResidualBlock& out) const {
        const std::vector<double> b5 = expand(b4);
        const Eigen::VectorXd b5v = Eigen::Map<const Eigen::VectorXd>(b5.data(), 5);
        Eigen::VectorXd empty(0);
        m_inner->fillResidual(i, b5v, empty, order, m_block);
        out.value = m_block.value;
        if (order >= quantape::math::CalibrationOrder::Gradient) {
            const double kappa = b4(1);
            const double theta = b4(2);
            const double sigma = b5[3];
            const double sk = theta / sigma;
            const double st = kappa / sigma;
            out.gradientB.resize(4);
            out.gradientB(0) = m_block.gradientB(0);
            out.gradientB(1) = m_block.gradientB(1) + m_block.gradientB(3) * sk;
            out.gradientB(2) = m_block.gradientB(2) + m_block.gradientB(3) * st;
            out.gradientB(3) = m_block.gradientB(4);
            out.gradientA.resize(0);
        }
        if (order >= quantape::math::CalibrationOrder::Hessian) {
            const double kappa = b4(1);
            const double theta = b4(2);
            const double sigma = b5[3];
            const double sk = theta / sigma;
            const double st = kappa / sigma;
            const double skk = -theta * theta / (sigma * sigma * sigma);
            const double stt = -kappa * kappa / (sigma * sigma * sigma);
            const double skt = 0.5 / sigma;
            const std::array<double, 4> s1 = {0.0, sk, st, 0.0};
            const std::array<double, 4> s2 = {0.0, 0.0, 0.0, 0.0}; // rho independent
            (void)s2;
            out.hessianBB.resize(4, 4);
            for (int a = 0; a < 4; ++a) {
                for (int b = 0; b < 4; ++b) {
                    const double h5ab = m_block.hessianBB(a, b);
                    const double h5a3 = m_block.hessianBB(a, 3);
                    const double h53b = m_block.hessianBB(3, b);
                    const double h533 = m_block.hessianBB(3, 3);
                    double value =
                        h5ab + h5a3 * s1[static_cast<std::size_t>(b)] +
                        h53b * s1[static_cast<std::size_t>(a)] +
                        h533 * s1[static_cast<std::size_t>(a)] * s1[static_cast<std::size_t>(b)];
                    if ((a == 1 && b == 1)) {
                        value += m_block.gradientB(3) * skk;
                    } else if (a == 2 && b == 2) {
                        value += m_block.gradientB(3) * stt;
                    } else if ((a == 1 && b == 2) || (a == 2 && b == 1)) {
                        value += m_block.gradientB(3) * skt;
                    }
                    out.hessianBB(a, b) = value;
                }
            }
            out.hessianBA.resize(4, 0);
        }
    }

    void fillInequality(std::size_t, const Eigen::VectorXd&,
                        quantape::math::CalibrationInequalityBlock& out) const {
        out.value = 0.0;
        out.gradientB.resize(0);
        out.hessianBB.resize(0, 0);
    }

private:
    const quantape::models::HestonModelCalibrationProblem* m_inner;
    mutable quantape::math::CalibrationResidualBlock m_block;
};

// ── Generic IFT-vs-FD harness ──

/// Central-difference Jacobian of any recalibration output w.r.t. market
/// parameters: f(a) must return a fixed-size vector; result is nOut x nA.
template <typename Fn>
Eigen::MatrixXd bumpJacobian(const Eigen::VectorXd& a, Fn f, double hScale) {
    const Eigen::VectorXd f0 = f(a);
    Eigen::MatrixXd out(f0.size(), a.size());
    for (Eigen::Index j = 0; j < a.size(); ++j) {
        const double h = hScale * std::max(1.0, std::fabs(a(j)));
        Eigen::VectorXd ap = a;
        Eigen::VectorXd am = a;
        ap(j) += h;
        am(j) -= h;
        out.col(j) = (f(ap) - f(am)) / (2.0 * h);
    }
    return out;
}

/// Second-order accurate reference: Richardson over two bump sizes
/// ((4 F(h) - F(2h)) / 3 for central differences = O(h^4) truncation).
template <typename Fn>
Eigen::MatrixXd richardsonJacobian(const Eigen::VectorXd& a, Fn f) {
    const Eigen::MatrixXd coarse = bumpJacobian(a, f, 2e-4);
    const Eigen::MatrixXd fine = bumpJacobian(a, f, 1e-4);
    return (4.0 * fine - coarse) / 3.0;
}

void testHestonUnconstrained() {
    QTA_LOG_INFO("test", "Heston IFT: unconstrained LS");
    const HestonFixture fx = makeHestonFixture();
    const std::vector<double> start{0.05, 1.5, 0.05, 0.5, -0.3};

    const Eigen::VectorXd bHat = calibrateUnconstrained(fx.problem, fx.a, start);
    const auto assembled = quantape::math::assembleCalibration(
        fx.problem, bHat, fx.a, quantape::math::CalibrationOrder::Hessian);
    checkClose("stationarity grad_b", assembled.gradientB.norm(), 0.0, 1e-9);

    // Same chain at the SLSQP optimum (a second calibrator, same interface)
    {
        quantape::math::SLSQP<double> solver(tightCriteria());
        std::vector<double> x = asVector(bHat);
        const auto result =
            solver.minimize(CalibrationValueGrad{fx.problem, fx.a}, quantape::math::NoConstraint{},
                            Bounds::unbounded(5), x);
        CHECK(result != quantape::math::OptimizeResult::Failure);
        const Eigen::VectorXd bSl = owning(x);
        const auto iftLbfgs = quantape::math::calibrationIft(fx.problem, bHat, fx.a);
        const auto iftSl = quantape::math::calibrationIft(fx.problem, bSl, fx.a);
        checkClose("LBFGS vs SLSQP db/da", (iftLbfgs.dbda - iftSl.dbda).norm(), 0.0, 1e-6);
    }

    quantape::math::CalibrationIftOptions options;
    const auto ift = quantape::math::calibrationIft(fx.problem, bHat, fx.a, {}, options);
    QTA_LOG_INFO("test", "  condition = {}, ridge = {}",
                 quantape::util::num(ift.diagnostics.condition, 3),
                 quantape::util::num(ift.diagnostics.ridge, 2));

    const auto recalibrate = [&](const Eigen::VectorXd& a, const std::vector<double>& s) {
        return calibrateUnconstrained(fx.problem, a, s);
    };
    const Eigen::MatrixXd fd = richardsonJacobian(
        fx.a, [&](const Eigen::VectorXd& a) { return recalibrate(a, asVector(bHat)); });
    QTA_LOG_INFO("test", "  db/da (KKT):");
    for (Eigen::Index i = 0; i < ift.dbda.rows(); ++i) {
        std::string row = "    [";
        for (Eigen::Index j = 0; j < ift.dbda.cols(); ++j) {
            row += quantape::util::num(ift.dbda(i, j), 6) + " ";
        }
        row += "] err vs FD = " + quantape::util::num((ift.dbda.row(i) - fd.row(i)).norm(), 2);
        QTA_LOG_INFO("test", "{}", row);
    }
    const double rel = (ift.dbda - fd).norm() / std::max(1e-12, fd.norm());
    checkClose("db/da KKT vs bump-recalibrate", rel, 0.0, 1e-4);

    // Downstream product: Heston call K=1.0, T=1.0 (not a calibration quote)
    const HestonMarket productMarket{1.0, 1.0, 0.0, 0.0, 1.0};
    const quantape::models::HestonParams bParams{bHat(0), bHat(1), bHat(2), bHat(3), bHat(4)};
    const double productValue = fx.model.call(bParams, productMarket);
    const quantape::models::HestonFullPoint productPoint =
        quantape::models::toFullPoint(bParams, productMarket);
    const auto productGradient = fx.model.fullGradient(productPoint, productMarket.tMax);
    Eigen::VectorXd dVdb(kNB);
    for (int k = 0; k < kNB; ++k) {
        dVdb(k) = productGradient(0, k);
    }
    const Eigen::VectorXd dVda = dVdb.transpose() * ift.dbda;
    const Eigen::MatrixXd dVdaFd = richardsonJacobian(fx.a, [&](const Eigen::VectorXd& a) {
        const Eigen::VectorXd bb = recalibrate(a, asVector(bHat));
        const quantape::models::HestonParams bp{bb(0), bb(1), bb(2), bb(3), bb(4)};
        Eigen::VectorXd out(1);
        out(0) = fx.model.call(bp, productMarket);
        return out;
    });
    QTA_LOG_INFO("test", "  product dV/da (IFT) = [{}, {}, {}], FD = [{}, {}, {}]",
                 quantape::util::num(dVda(0), 6), quantape::util::num(dVda(1), 6),
                 quantape::util::num(dVda(2), 6), quantape::util::num(dVdaFd(0, 0), 6),
                 quantape::util::num(dVdaFd(0, 1), 6), quantape::util::num(dVdaFd(0, 2), 6));
    checkClose("product dV/da vs bump-recalibrate", (dVda - dVdaFd.row(0).transpose()).norm(), 0.0,
               1e-4 * std::max(1.0, dVdaFd.norm()));

    // Instrument route on the same assembled Jacobians (noisy-free data:
    // the residual-curvature terms vanish, both routes coincide)
    const auto jac = quantape::math::calibrationJacobiansFromAssembled(assembled);
    const auto inst = quantape::math::instrumentCalibrationJacobian(jac);
    QTA_LOG_INFO("test", "  instrument route: rank={} cond={}", inst.rank,
                 quantape::util::num(inst.condition, 3));
    for (Eigen::Index i = 0; i < inst.dbda.rows(); ++i) {
        std::string row = "    [";
        for (Eigen::Index j = 0; j < inst.dbda.cols(); ++j) {
            row += quantape::util::num(inst.dbda(i, j), 6) + " ";
        }
        row += "]";
        QTA_LOG_INFO("test", "{}", row);
    }
    checkClose("KKT vs instrument route", (ift.dbda - inst.dbda).norm(), 0.0,
               1e-4 * std::max(1.0, ift.dbda.norm()));
    QTA_LOG_INFO("test",
                 "  [ok] unconstrained: KKT vs FD, product risk, instrument route");
}

void testHestonFellerConstrained() {
    QTA_LOG_INFO("test", "Heston IFT: Feller-constrained (binding)");
    const HestonFixture fx = makeHestonFixture();
    const std::vector<double> start{0.05, 1.5, 0.05, 0.5, -0.3};

    std::vector<double> exported;
    const Eigen::VectorXd bHat = calibrateConstrained(fx.problem, fx.a, start, &exported);
    const auto g = quantape::math::calibrationInequalityValues(fx.problem, bHat);
    const auto assembled = quantape::math::assembleCalibration(
        fx.problem, bHat, fx.a, quantape::math::CalibrationOrder::Gradient);
    const auto lambda =
        quantape::math::recoverActiveMultipliers(fx.problem, bHat, assembled.gradientB);
    QTA_LOG_INFO("test", "  g = {}, recovered lambda = {} (calibrator exported {})",
                 quantape::util::num(g(0), 3), quantape::util::num(lambda[0], 6),
                 quantape::util::num(exported.empty() ? 0.0 : exported[0], 6));
    CHECK(g(0) <= 1e-7);
    CHECK(lambda[0] > 1e-6);

    // Auto-recovery inside the chain (no multipliers passed)
    const auto ift = quantape::math::calibrationIft(fx.problem, bHat, fx.a);
    CHECK(ift.diagnostics.activeInequalities.size() == 1);

    const Eigen::MatrixXd fd = richardsonJacobian(fx.a, [&](const Eigen::VectorXd& a) {
        return calibrateConstrained(fx.problem, a, asVector(bHat));
    });
    const double rel = (ift.dbda - fd).norm() / std::max(1e-12, fd.norm());
    // The constrained optimum is ill-conditioned (cond(J) ~ 1.2e7 for this
    // 2-maturity quote set): the recalibrated reference carries SLSQP's
    // feasibility tolerance into the flat kappa/sigma directions, so the
    // agreement is limited to a few percent. The chain itself is exact given
    // the point and multipliers.
    QTA_LOG_INFO("test", "  db/da KKT vs FD: rel err = {} (reference precision limited)",
                 quantape::util::num(rel, 3));
    checkClose("constrained db/da vs bump-recalibrate", rel, 0.0, 5e-2);

    // dlambda/da vs FD of the AUGLAG multiplier
    const Eigen::MatrixXd dlambdaFd = richardsonJacobian(fx.a, [&](const Eigen::VectorXd& a) {
        const Eigen::VectorXd bb = calibrateConstrained(fx.problem, a, asVector(bHat));
        const auto da = quantape::math::assembleCalibration(
            fx.problem, bb, a, quantape::math::CalibrationOrder::Gradient);
        const auto ll = quantape::math::recoverActiveMultipliers(fx.problem, bb, da.gradientB);
        Eigen::VectorXd out(1);
        out(0) = ll[0];
        return out;
    });
    const double dlRel = (ift.dlambdaDa - dlambdaFd).norm() / std::max(1e-12, dlambdaFd.norm());
    QTA_LOG_INFO("test", "  dlambda/da (IFT) = [{}, {}, {}], FD = [{}, {}, {}]",
                 quantape::util::num(ift.dlambdaDa(0, 0), 6),
                 quantape::util::num(ift.dlambdaDa(0, 1), 6),
                 quantape::util::num(ift.dlambdaDa(0, 2), 6), quantape::util::num(dlambdaFd(0, 0), 6),
                 quantape::util::num(dlambdaFd(0, 1), 6), quantape::util::num(dlambdaFd(0, 2), 6));
    checkClose("dlambda/da vs bump-recalibrate", dlRel, 0.0, 5e-2);
    // Independent boundary reference: sigma = sqrt(2 kappa theta), 4 free
    // parameters; Richardson FD of that well-conditioned optimum avoids the
    // cond(J) ~ 1e7 sensitivity of the constrained interior solve.
    {
        const auto fixedQuotes = [&]() {
            std::vector<quantape::models::HestonCalibrationQuote> quotes;
            for (const quantape::models::HestonCalibrationQuote& q : fx.problem.quotes()) {
                quotes.push_back(q);
            }
            return quotes;
        }();
        auto boundaryFit = [&](const Eigen::VectorXd& a) {
            const quantape::models::HestonModelCalibrationProblem inner(fx.model, fixedQuotes, a(0),
                                                                        a(1), a(2));
            const FellerBoundaryProblem boundary(inner);
            Eigen::VectorXd start(4);
            start << bHat(0), bHat(1), bHat(2), bHat(4);
            quantape::math::LBFGS<double> solver(tightCriteria(), 10);
            std::vector<double> x4 = asVector(start);
            const Eigen::VectorXd noMarket(0);
            const auto result =
                solver.minimize(quantape::math::CalibrationValueGrad{boundary, noMarket}, x4);
            CHECK(result != quantape::math::OptimizeResult::Failure);
            const Eigen::VectorXd b4 = owning(x4);
            const std::vector<double> b5 = FellerBoundaryProblem::expand(b4);
            Eigen::VectorXd out(5);
            for (int k = 0; k < 5; ++k) {
                out(k) = b5[static_cast<std::size_t>(k)];
            }
            return out;
        };
        const Eigen::MatrixXd fdBoundary = richardsonJacobian(fx.a, boundaryFit);
        const double relBoundary =
            (ift.dbda - fdBoundary).norm() / std::max(1e-12, fdBoundary.norm());
        QTA_LOG_INFO("test", "  boundary-reference db/da: rel err = {} (cond ~ 1e3)",
                     quantape::util::num(relBoundary, 3));
        checkClose("constrained db/da vs boundary FD", relBoundary, 0.0, 1e-3);
    }

    QTA_LOG_INFO("test", "  [ok] Feller-constrained: db/da and dlambda/da vs FD");
}

void testBlackScholesGenericity() {
    QTA_LOG_INFO("test", "Generic layer with a second model (Black-Scholes vol)");
    const double S = 1.0, r = 0.02, q = 0.01;
    const double truthVol = 0.24;
    std::vector<BsCalibrationProblem::Quote> quotes;
    const double strikes[4] = {0.9, 1.0, 1.05, 1.1};
    for (double k : strikes) {
        const double target =
            quantape::pricing::GBS<double>{
                S, k, r, r - q, truthVol, 1.0, quantape::pricing::OptionType::Call}
                .price();
        quotes.push_back({k, 1.0, target, 1.0});
    }
    const BsCalibrationProblem prob(quotes);
    Eigen::VectorXd a(3);
    a << S, r, q;

    quantape::math::LBFGS<double> solver(tightCriteria(), 4);
    std::vector<double> x{0.4};
    const auto result = solver.minimize(CalibrationValueGrad{prob, a}, x);
    CHECK(result != quantape::math::OptimizeResult::Failure);
    const Eigen::VectorXd bHat = Eigen::Map<Eigen::VectorXd>(x.data(), x.size());
    checkClose("BS recovered vol", bHat(0), truthVol, 1e-8);

    const auto ift = quantape::math::calibrationIft(prob, bHat, a);
    auto fit = [&](const Eigen::VectorXd& aa) {
        quantape::math::LBFGS<double> s(tightCriteria(), 4);
        std::vector<double> xx = asVector(bHat);
        s.minimize(CalibrationValueGrad{prob, aa}, xx);
        return owning(xx);
    };
    const Eigen::MatrixXd fd = richardsonJacobian(a, fit);
    QTA_LOG_INFO("test", "  dvol/da (IFT) = [{}, {}, {}], FD = [{}, {}, {}]",
                 quantape::util::num(ift.dbda(0, 0), 6), quantape::util::num(ift.dbda(0, 1), 6),
                 quantape::util::num(ift.dbda(0, 2), 6), quantape::util::num(fd(0, 0), 6),
                 quantape::util::num(fd(0, 1), 6), quantape::util::num(fd(0, 2), 6));
    const double rel = (ift.dbda - fd).norm() / std::max(1e-12, fd.norm());
    checkClose("BS dvol/da vs bump-recalibrate", rel, 0.0, 1e-4);
    QTA_LOG_INFO("test", "  [ok] same assemble/IFT calls, second model");
}

// ── Stan interop: var/fvar<var> through the same adapter ──
//
// The scalar-generic `CalibrationObjective`/`CalibrationConstraintWriter`
// plug into the existing AD stack: LBFGS<var> for the solve, and
// ImplicitFunction.h (minimizeDifferential / iftKkt) for the IFT. Prices
// come from the `make_callback_var` overloads, so the tape holds one node
// per quote regardless of the quadrature.

void testStanInterop() {
    QTA_LOG_INFO("test", "Stan interop: var / fvar<var> through the same adapter");
    const HestonFixture fx = makeHestonFixture();
    const std::vector<double> start{0.05, 1.5, 0.05, 0.5, -0.3};
    const std::vector<double> market = asVector(fx.a);

    const Eigen::VectorXd bHat = calibrateUnconstrained(fx.problem, fx.a, start);
    const quantape::math::CalibrationObjective<HestonCalibrationProblem> objective(fx.problem,
                                                                                   market);

    // 1. var solve: same optimum as the double path
    {
        stan::math::recover_memory();
        quantape::math::StopCriteria criteria = tightCriteria();
        quantape::math::LBFGS<stan::math::var> solver(criteria, 10);
        std::vector<double> x = start;
        OptimizerState state;
        const auto result = solver.minimize(objective, x, state);
        stan::math::recover_memory();
        CHECK(result != quantape::math::OptimizeResult::Failure);
        const Eigen::VectorXd bVar = owning(x);
        QTA_LOG_INFO("test", "  LBFGS<var> optimum vs double path: {} (evals={})",
                     quantape::util::num((bVar - bHat).norm(), 2), state.evals);
        checkClose("var optimum == double optimum", (bVar - bHat).norm(), 0.0, 1e-6);

        // analytic chain at the var optimum agrees with the double chain
        const auto iftVar = quantape::math::calibrationIft(fx.problem, bVar, fx.a);
        const auto iftDouble = quantape::math::calibrationIft(fx.problem, bHat, fx.a);
        // The chain is ill-conditioned (cond ~1.2e7), so a 1e-10 optimum
        // difference maps into ~1e-3 of db/da; compare relative to the norm.
        const double relChain =
            (iftVar.dbda - iftDouble.dbda).norm() / std::max(1e-12, iftDouble.dbda.norm());
        checkClose("chain at var optimum", relChain, 0.0, 1e-4);
    }

    // 1b. Fixed-market adapter (5-parameter kernels) through the SAME AD
    // objective contract: the calibration solve is ~2.5x cheaper and reaches
    // the same optimum. The nine-parameter adapter stays available for the
    // market-risk IFT below.
    {
        std::vector<quantape::models::HestonCalibrationQuote> quotes;
        for (const quantape::models::HestonCalibrationQuote& q : fx.problem.quotes()) {
            quotes.push_back(q);
        }
        quantape::models::HestonModelCalibrationProblem fixedProb(fx.model, quotes, fx.a(0),
                                                                  fx.a(1), fx.a(2));
        const quantape::math::CalibrationObjective<quantape::models::HestonModelCalibrationProblem>
            fixedObjective(fixedProb, {});
        quantape::math::StopCriteria criteria = tightCriteria();
        auto solve = [&](const auto& objective, const std::vector<double>& x0) {
            stan::math::recover_memory();
            quantape::math::LBFGS<stan::math::var> solver(criteria, 10);
            std::vector<double> xx = x0;
            OptimizerState st;
            const auto t0 = std::chrono::steady_clock::now();
            solver.minimize(objective, xx, st);
            const auto t1 = std::chrono::steady_clock::now();
            stan::math::recover_memory();
            return std::make_tuple(owning(xx),
                                   std::chrono::duration<double, std::micro>(t1 - t0).count());
        };
        const auto [xF, usF] = solve(fixedObjective, start);
        const auto [x9, us9] = solve(objective, start);
        QTA_LOG_INFO("test",
                     "  AD solve: 5-param {} us, 9-param {} us ({}x), optimum diff {}",
                     quantape::util::num(usF, 1), quantape::util::num(us9, 1),
                     quantape::util::num(us9 / usF, 2), quantape::util::num((xF - x9).norm(), 2));
        checkClose("fixed-market AD optimum == 9-param", (xF - x9).norm(), 0.0, 1e-6);
    }

    // 2. ImplicitFunction.h IFT with the same objective/constraints
    {
        std::vector<double> x = asVector(bHat);
        OptimizerState state;
        quantape::math::IftResult info;
        std::vector<double> dpdm;
        const auto none = quantape::math::Bounds::unbounded(5);
        const auto t0 = std::chrono::steady_clock::now();
        const auto result = quantape::math::minimizeDifferential(
            objective, quantape::math::NoConstraint{}, quantape::math::NoConstraint{}, none, market,
            x, state, info, &dpdm);
        const auto t1 = std::chrono::steady_clock::now();
        QTA_LOG_INFO(
            "test", "  AD IFT wall: {} us (primal cache)",
            quantape::util::num(std::chrono::duration<double, std::micro>(t1 - t0).count(), 1));
        CHECK(result != quantape::math::OptimizeResult::Failure);
        Eigen::MatrixXd adDbda(5, 3);
        for (int i = 0; i < 5; ++i) {
            for (int j = 0; j < 3; ++j) {
                adDbda(i, j) = dpdm[static_cast<std::size_t>(i) * 3 + static_cast<std::size_t>(j)];
            }
        }
        const Eigen::VectorXd xAd = owning(x);
        const auto ift = quantape::math::calibrationIft(fx.problem, xAd, fx.a);
        const double rel = (adDbda - ift.dbda).norm() / std::max(1.0, ift.dbda.norm());
        QTA_LOG_INFO("test",
                     "  AD IFT (minimizeDifferential) vs analytic chain: rel err = {}",
                     quantape::util::num(rel, 2));
        checkClose("AD IFT == analytic chain", rel, 0.0, 1e-6);

        // 3. constrained AD IFT: Feller via the scalar-generic writer
        const std::vector<double> cStart = asVector(calibrateConstrained(fx.problem, fx.a, start));
        std::vector<double> xc = cStart;
        OptimizerState stateC;
        const auto bounds =
            Bounds::fromVectors({1e-6, 1e-6, 1e-6, 1e-6, -0.999}, {10.0, 50.0, 10.0, 10.0, 0.999});
        const quantape::math::CalibrationConstraintWriter<HestonCalibrationProblem> constraints{
            &fx.problem};
        quantape::math::SLSQP<stan::math::var> solverC(tightCriteria());
        const auto resultC = solverC.minimize(objective, constraints,
                                              quantape::math::NoConstraint{}, bounds, xc, stateC);
        CHECK(resultC != quantape::math::OptimizeResult::Failure);
        const Eigen::VectorXd xOpt = owning(xc);

        // Multipliers: use the solver's when exported, otherwise recover from
        // stationarity (calibrator-agnostic, same convention).
        std::vector<double> lam = stateC.ineq_multipliers;
        if (lam.size() != fx.problem.numInequalities()) {
            const auto grad = quantape::math::assembleCalibration(
                fx.problem, xOpt, fx.a, quantape::math::CalibrationOrder::Gradient);
            lam = quantape::math::recoverActiveMultipliers(fx.problem, xOpt, grad.gradientB);
        }
        std::vector<double> dpdmC;
        std::vector<double> dlamC;
        std::vector<double> dnuC;
        quantape::math::IftResult infoC;
        quantape::math::iftKkt([&](const auto& x, const auto& m) { return objective(x, m); },
                               constraints, quantape::math::NoConstraint{}, bounds, xc, market, lam,
                               {}, dpdmC, dlamC, dnuC, infoC);
        Eigen::MatrixXd adC(5, 3);
        for (int i = 0; i < 5; ++i) {
            for (int j = 0; j < 3; ++j) {
                adC(i, j) = dpdmC[static_cast<std::size_t>(i) * 3 + static_cast<std::size_t>(j)];
            }
        }
        const auto iftC = quantape::math::calibrationIft(fx.problem, xOpt, fx.a, lam);
        const double relC = (adC - iftC.dbda).norm() / std::max(1.0, iftC.dbda.norm());
        QTA_LOG_INFO("test",
                     "  constrained AD IFT vs analytic chain: rel err = {} (active rows={})",
                     quantape::util::num(relC, 2), iftC.diagnostics.activeInequalities.size());
        checkClose("constrained AD IFT == analytic chain", relC, 0.0, 1e-5);
        QTA_LOG_INFO("test", "  dlambda/dm (AD) = [{}, {}, {}], analytic = [{}, {}, {}]",
                     quantape::util::num(dlamC[0], 6), quantape::util::num(dlamC[1], 6),
                     quantape::util::num(dlamC[2], 6), quantape::util::num(iftC.dlambdaDa(0, 0), 6),
                     quantape::util::num(iftC.dlambdaDa(0, 1), 6),
                     quantape::util::num(iftC.dlambdaDa(0, 2), 6));
        for (int j = 0; j < 3; ++j) {
            checkClose("dlambda/dm AD == analytic", dlamC[static_cast<std::size_t>(j)],
                       iftC.dlambdaDa(0, j), 1e-3 * std::max(1.0, std::fabs(iftC.dlambdaDa(0, j))));
        }
    }
    QTA_LOG_INFO("test", "  [ok] var solve + callback-var pricing + AD IFT/KKT chains");
}

} // namespace

int main() {
    QTA_LOG_INFO("test", "Generic calibration/IFT layer (H6)");
    testHestonUnconstrained();
    testHestonFellerConstrained();
    testBlackScholesGenericity();
    testStanInterop();
    QTA_LOG_INFO("test", "ALL CALIBRATION IFT TESTS PASSED");
    return 0;
}
