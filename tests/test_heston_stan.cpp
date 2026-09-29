//
// test_heston_stan.cpp -- Stan `var`/`fvar<var>` wiring of the Heston pricer
//
// Gates the `make_callback_var` instantiation against the double Jacobians:
//  - var    : single-node tape; adjoint == `HestonModel::fullGradient`
//  - fvar<var>: single tangent node; `stan::math::hessian` == `fullHessian`
//  - value parity with `callFull`
//

#include "quantape/math/StanMath.h"

#include "quantape/models/HestonStanPrimitives.h"

#include <Eigen/Dense>

#include <chrono>
#include <cmath>
#include <cstdio>
#include <type_traits>

#include "TestSupport.h"

namespace {

constexpr int P = quantape::models::HESTON_PARAM_COUNT;

int failures = 0;

void check(bool ok, const char* name, double got = 0.0, double tol = 0.0) {
    if (!ok) {
        ++failures;
        QTA_LOG_ERROR("test", "FAIL: {} (got={} tol={})", name, quantape_test::num(got, 3),
                      quantape_test::num(tol, 3));
    }
}

using quantape::models::HestonFullPoint;
using quantape::models::HestonModel;

struct HestonStanFunctor {
    const HestonModel* model = nullptr;
    double tMax = 1.0;

    template <typename TT>
    TT operator()(const Eigen::Matrix<TT, Eigen::Dynamic, 1>& theta) const {
        if constexpr (std::is_same_v<TT, double>) {
            Eigen::Matrix<double, P, 1> x = theta;
            return quantape::models::hestonCall(*model, x, tMax);
        } else if constexpr (std::is_same_v<TT, stan::math::var>) {
            Eigen::Matrix<stan::math::var, P, 1> x = theta;
            return quantape::models::hestonCall(*model, x, tMax);
        } else {
            Eigen::Matrix<stan::math::fvar<stan::math::var>, P, 1> x = theta;
            return quantape::models::hestonCall(*model, x, tMax);
        }
    }
};

Eigen::VectorXd makePoint() {
    const quantape::models::HestonFullPoint x{0.04, 2.5,  0.06, 0.75,
                                              -0.1, // v0, kappa, theta, sigma, rho
                                              1.0,  1.06, 0.02, 0.0}; // S, K, r, q
    Eigen::VectorXd out(P);
    for (int i = 0; i < P; ++i) {
        out(i) = x[static_cast<std::size_t>(i)];
    }
    return out;
}

void testStanGradientAndHessian() {
    using quantape::models::HestonModel;
    const HestonModel model;
    const Eigen::VectorXd x = makePoint();
    HestonFullPoint point{};
    for (int i = 0; i < P; ++i) {
        point[static_cast<std::size_t>(i)] = x(i);
    }

    const double price = model.callFull(point, 1.0);
    const Eigen::Matrix<double, 1, P> g = model.fullGradient(point, 1.0);
    const Eigen::Matrix<double, P, P> H = model.fullHessian(point, 1.0);

    HestonStanFunctor functor{&model, 1.0};

    double fx = 0.0;
    Eigen::VectorXd grad(P);
    stan::math::gradient(functor, x, fx, grad);
    check(std::fabs(fx - price) < 1e-12, "stan var value == callFull", std::fabs(fx - price),
          1e-12);

    double gradErr = 0.0;
    for (int i = 0; i < P; ++i) {
        gradErr = std::max(gradErr, std::fabs(grad(i) - g(i)));
    }
    check(gradErr < 1e-9, "stan adjoint == fullGradient", gradErr, 1e-9);
    QTA_LOG_INFO("test", "  [ok] var adjoint == fullGradient (max err {}, 1-node tape)",
                 quantape_test::num(gradErr, 2));

    double fx2 = 0.0;
    Eigen::VectorXd grad2(P);
    Eigen::MatrixXd hess(P, P);
    stan::math::hessian(functor, x, fx2, grad2, hess);
    double hessErr = 0.0;
    for (int i = 0; i < P; ++i) {
        for (int j = 0; j < P; ++j) {
            hessErr = std::max(hessErr, std::fabs(hess(i, j) - H(i, j)));
        }
    }
    check(hessErr < 1e-9, "stan fvar<var> Hessian == fullHessian", hessErr, 1e-9);
    QTA_LOG_INFO("test",
                 "  [ok] fvar<var> Hessian == fullHessian (max err {}, 2-node tape)",
                 quantape_test::num(hessErr, 2));

    // Independent FD of the re-evaluated gradient for three representative entries
    const double h = 1e-4;
    const int idx[3] = {quantape::models::HESTON_SIGMA, quantape::models::HESTON_SPOT,
                        quantape::models::HESTON_RATE};
    for (int k = 0; k < 3; ++k) {
        HestonFullPoint up = point, down = point;
        up[static_cast<std::size_t>(idx[k])] += h;
        down[static_cast<std::size_t>(idx[k])] -= h;
        const Eigen::Matrix<double, 1, P> gu = model.fullGradient(up, 1.0);
        const Eigen::Matrix<double, 1, P> gd = model.fullGradient(down, 1.0);
        for (int i = 0; i < P; ++i) {
            const double fd = (gu(i) - gd(i)) / (2.0 * h);
            check(std::fabs(fd - H(i, idx[k])) < 1e-5 * std::max(1.0, std::fabs(fd)),
                  "fullHessian column vs FD(gradient)", std::fabs(fd - H(i, idx[k])), 1e-5);
        }
    }
    QTA_LOG_INFO("test", "  [ok] fullHessian columns vs central FD of fullGradient");
}

void testPrimalCache() {
    const HestonModel model;
    const Eigen::VectorXd x = makePoint();
    HestonFullPoint point{};
    for (int i = 0; i < P; ++i) {
        point[static_cast<std::size_t>(i)] = x(i);
    }
    const Eigen::Matrix<double, P, P> H = model.fullHessian(point, 1.0);
    const auto g = model.fullGradient(point, 1.0);

    quantape::models::HestonSourceCache cache(model);
    auto cachedFunctor = [&](const auto& z) {
        using T = typename std::decay_t<decltype(z)>::Scalar;
        if constexpr (std::is_same_v<T, double>) {
            Eigen::Matrix<double, P, 1> xv = z;
            return quantape::models::hestonCall(model, xv, 1.0);
        } else if constexpr (std::is_same_v<T, stan::math::var>) {
            Eigen::Matrix<stan::math::var, P, 1> xv = z;
            return cache.price(xv, 1.0);
        } else {
            Eigen::Matrix<stan::math::fvar<stan::math::var>, P, 1> xv = z;
            return cache.price(xv, 1.0);
        }
    };
    double fx = 0.0;
    Eigen::VectorXd grad(P);
    Eigen::MatrixXd Hc(P, P);
    stan::math::hessian(cachedFunctor, x, fx, grad, Hc);
    double hessErr = 0.0;
    for (int i = 0; i < P; ++i) {
        hessErr = std::max(hessErr, std::fabs(Hc(i, i) - H(i, i)));
        for (int j = 0; j < P; ++j) {
            hessErr = std::max(hessErr, std::fabs(Hc(i, j) - H(i, j)));
        }
    }
    check(hessErr < 1e-12, "cached fvar<var> Hessian == fullHessian", hessErr, 1e-12);
    QTA_LOG_INFO("test", "  [ok] cached price: hessian err {}, rebuilds={}",
                 quantape_test::num(hessErr, 2), cache.rebuilds());

    // timing: cached vs uncached hessian
    HestonStanFunctor functor{&model, 1.0};
    const auto now = [] { return std::chrono::steady_clock::now(); };
    const auto us = [](auto a, auto b) {
        return std::chrono::duration<double, std::micro>(b - a).count();
    };
    const int reps = 3;
    double plain = 0.0;
    double cached = 0.0;
    for (int r = 0; r < reps; ++r) {
        Eigen::MatrixXd Ht(P, P);
        auto t0 = now();
        stan::math::hessian(functor, x, fx, grad, Ht);
        auto t1 = now();
        plain += us(t0, t1);
        t0 = now();
        stan::math::hessian(cachedFunctor, x, fx, grad, Ht);
        t1 = now();
        cached += us(t0, t1);
    }
    QTA_LOG_INFO("test", "  timings: stan hessian plain {} us, primal-cached {} us ({}x)",
                 quantape_test::num(plain / reps, 1), quantape_test::num(cached / reps, 1),
                 quantape_test::num(plain / cached, 1));
}

void testTiming() {
    const HestonModel model;
    const Eigen::VectorXd x = makePoint();
    HestonFullPoint point{};
    for (int i = 0; i < P; ++i) {
        point[static_cast<std::size_t>(i)] = x(i);
    }
    HestonStanFunctor functor{&model, 1.0};

    const auto now = [] { return std::chrono::steady_clock::now(); };
    const auto us = [](auto a, auto b) {
        return std::chrono::duration<double, std::micro>(b - a).count();
    };

    const int reps = 5;
    double gradUs = 0.0;
    double hessUs = 0.0;
    for (int r = 0; r < reps; ++r) {
        double fx = 0.0;
        Eigen::VectorXd grad(P);
        auto t0 = now();
        stan::math::gradient(functor, x, fx, grad);
        auto t1 = now();
        gradUs += us(t0, t1);
        Eigen::MatrixXd H(P, P);
        t0 = now();
        stan::math::hessian(functor, x, fx, grad, H);
        t1 = now();
        hessUs += us(t0, t1);
    }
    QTA_LOG_INFO("test",
                 "  timings: stan gradient (var, value+grad build) {} us, "
                 "stan hessian (fvar<var>, value+grad+hess build) {} us",
                 quantape_test::num(gradUs / reps, 1), quantape_test::num(hessUs / reps, 1));
}

} // namespace

int main() {
    QTA_LOG_INFO("test", "Heston Stan callback-var tests");
    testStanGradientAndHessian();
    testPrimalCache();
    testTiming();
    if (failures == 0) {
        QTA_LOG_INFO("test", "ALL HESTON STAN TESTS PASSED");
        return 0;
    }
    QTA_LOG_ERROR("test", "{} FAILURES", failures);
    return 1;
}
