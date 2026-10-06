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
#include <cstdint>
#include <type_traits>

#include "support/GtestSupport.h"
#include "support/StanTapeFixture.h"

namespace {

constexpr int P = quantape::models::HESTON_PARAM_COUNT;

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

HestonFullPoint asFullPoint(const Eigen::VectorXd& x) {
    HestonFullPoint point{};
    for (int i = 0; i < P; ++i) {
        point[static_cast<std::size_t>(i)] = x(i);
    }
    return point;
}

class HestonStanTest : public StanTapeTest {};

} // namespace

TEST_F(HestonStanTest, varAdjointMatchesFullGradient) {
    const HestonModel model;
    const Eigen::VectorXd x = makePoint();
    const HestonFullPoint point = asFullPoint(x);
    const double price = model.callFull(point, 1.0);
    const Eigen::Matrix<double, 1, P> g = model.fullGradient(point, 1.0);

    HestonStanFunctor functor{&model, 1.0};
    double fx = 0.0;
    Eigen::VectorXd grad(P);
    stan::math::gradient(functor, x, fx, grad);
    CHECK_CLOSE("stan var value == callFull", fx, price, 1e-12);

    double gradErr = 0.0;
    for (int i = 0; i < P; ++i) {
        gradErr = std::max(gradErr, std::fabs(grad(i) - g(i)));
    }
    CHECK_CLOSE("stan adjoint == fullGradient", gradErr, 0.0, 1e-9);
    ::testing::Test::RecordProperty("stan_var_value_err",
                                    quantape::util::num(std::fabs(fx - price), 3));
    ::testing::Test::RecordProperty("stan_adjoint_err", quantape::util::num(gradErr, 3));
}

TEST_F(HestonStanTest, fvarHessianMatchesFullHessian) {
    const HestonModel model;
    const Eigen::VectorXd x = makePoint();
    const HestonFullPoint point = asFullPoint(x);
    const Eigen::Matrix<double, P, P> H = model.fullHessian(point, 1.0);

    HestonStanFunctor functor{&model, 1.0};
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
    CHECK_CLOSE("stan fvar<var> Hessian == fullHessian", hessErr, 0.0, 1e-9);
    ::testing::Test::RecordProperty("stan_hessian_err", quantape::util::num(hessErr, 3));
}

TEST_F(HestonStanTest, hessianColumnsMatchFiniteDifference) {
    const HestonModel model;
    const quantape::models::HestonFullPoint point = asFullPoint(makePoint());
    const Eigen::Matrix<double, P, P> H = model.fullHessian(point, 1.0);

    // Independent FD of the re-evaluated gradient for three representative entries
    const double h = 1e-4;
    const int idx[3] = {quantape::models::HESTON_SIGMA, quantape::models::HESTON_SPOT,
                        quantape::models::HESTON_RATE};
    for (int k = 0; k < 3; ++k) {
        SCOPED_TRACE(::testing::Message() << "column " << k);
        HestonFullPoint up = point, down = point;
        up[static_cast<std::size_t>(idx[k])] += h;
        down[static_cast<std::size_t>(idx[k])] -= h;
        const Eigen::Matrix<double, 1, P> gu = model.fullGradient(up, 1.0);
        const Eigen::Matrix<double, 1, P> gd = model.fullGradient(down, 1.0);
        for (int i = 0; i < P; ++i) {
            const double fd = (gu(i) - gd(i)) / (2.0 * h);
            CHECK_CLOSE("fullHessian column vs FD(gradient)", H(i, idx[k]), fd,
                        1e-5 * std::max(1.0, std::fabs(fd)));
        }
    }
}

TEST_F(HestonStanTest, cachedHessianMatchesDirect) {
    const HestonModel model;
    const Eigen::VectorXd x = makePoint();
    const HestonFullPoint point = asFullPoint(x);
    const Eigen::Matrix<double, P, P> H = model.fullHessian(point, 1.0);

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
    CHECK_CLOSE("cached fvar<var> Hessian == fullHessian", hessErr, 0.0, 1e-12);
    ::testing::Test::RecordProperty("cached_hessian_err", quantape::util::num(hessErr, 3));
    ::testing::Test::RecordProperty("rebuilds", static_cast<int64_t>(cache.rebuilds()));

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
    ::testing::Test::RecordProperty("stan_hessian_plain_us", quantape::util::num(plain / reps, 1));
    ::testing::Test::RecordProperty("stan_hessian_cached_us",
                                    quantape::util::num(cached / reps, 1));
}
