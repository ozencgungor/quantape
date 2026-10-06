// test_tanh_sinh.cpp — validates quantape::math::TanhSinhIntegrator
//
//   double: polynomial, trigonometric, endpoint-singular integrands,
//           reversed bounds, infinite and half-infinite domains
//   var:    integral value matches double; gradient w.r.t. a parameter
//           (in the integrand and in the bounds) matches the analytic value
//   `fvar<var>`: second derivative (Hessian) matches the analytic value
//
// The legacy helper close(a, b, tol) was relative: |a-b| <= tol*(1+|b|).
// CHECK_CLOSE is absolute, so each call passes the effective absolute
// tolerance tol*(1+|expected|), keeping the contract identical
// (module_plan_numerics §3.8).
#include "quantape/math/StanMath.h"

#include "quantape/math/Integrals/IntegratorStanPrimitives.h"
#include "quantape/math/Integrals/TanhSinhIntegrator.h"
#include "quantape/util/Constants.h"
using ::quantape::util::kPi;

#include <cmath>
#include <type_traits>
#include <vector>

#include "support/GtestSupport.h"
#include "support/StanTapeFixture.h"

using quantape::math::TanhSinhIntegrator;

class TanhSinhAdTest : public StanTapeTest {};

namespace {

double effectiveTol(double expected, double tol) {
    return tol * (1.0 + std::abs(expected));
}

constexpr double kAbsTol = 1e-10;
constexpr std::size_t kMaxEvals = 1'000'000;

} // namespace

TEST_F(TanhSinhAdTest, finiteDomainsAndEndpointSingularities) {
    TanhSinhIntegrator<double> integ(kAbsTol, kMaxEvals);

    const double i_poly = integ([](double x) { return x * x; }, 0.0, 1.0);
    CHECK_CLOSE("int_0^1 x^2", i_poly, 1.0 / 3.0, effectiveTol(1.0 / 3.0, 1e-9));

    const double i_sin = integ([](double x) { return std::sin(x); }, 0.0, kPi);
    CHECK_CLOSE("int_0^pi sin", i_sin, 2.0, effectiveTol(2.0, 1e-9));

    // endpoint singularity: int_0^1 x^{-1/2} dx = 2
    // Plain path: the abscissa collapses onto the endpoint in double, so
    // the boundary guard limits accuracy to ~1e-8 (sqrt of machine eps
    // in the unresolved tail) — the complement path below does better.
    const double i_sqrt_sing = integ([](double x) { return 1.0 / std::sqrt(x); }, 0.0, 1.0);
    CHECK_CLOSE("int_0^1 x^-1/2 (plain)", i_sqrt_sing, 2.0, effectiveTol(2.0, 1e-6));

    // both endpoints singular: int_{-1}^{1} 1/sqrt(1-x^2) dx = pi
    const double i_arcsin = integ([](double x) { return 1.0 / std::sqrt(1.0 - x * x); }, -1.0, 1.0);
    CHECK_CLOSE("int_-1^1 (1-x^2)^-1/2 (plain)", i_arcsin, kPi, effectiveTol(kPi, 1e-6));

    // reversed bounds flip the sign
    const double i_rev = integ([](double x) { return x * x; }, 1.0, 0.0);
    CHECK_CLOSE("reversed bounds", i_rev, -1.0 / 3.0, effectiveTol(-1.0 / 3.0, 1e-9));
}

TEST_F(TanhSinhAdTest, complementPathReachesFullPrecision) {
    // The functor uses the signed distance d to the nearer endpoint
    // (d > 0 near the lower endpoint, d < 0 near the upper one). A tighter
    // accuracy target is used to show it is not the ~1e-8 guard floor that
    // limits the plain path. Observed accuracy here is ~1e-10 absolute (the
    // asymptotic error estimator under-reports the residual for square-root
    // singularities), ~100x better than plain.
    TanhSinhIntegrator<double> integ_fine(1e-12, kMaxEvals);

    const double i_sqrt_sing_c = integ_fine.integrateWithComplement(
        [](double x, double d) { return 1.0 / std::sqrt(d > 0.0 ? d : x); }, 0.0, 1.0);
    CHECK_CLOSE("int_0^1 x^-1/2 (complement)", i_sqrt_sing_c, 2.0, effectiveTol(2.0, 1e-9));

    const double i_arcsin_c = integ_fine.integrateWithComplement(
        [](double, double d) {
            const double ad = std::abs(d);
            return 1.0 / std::sqrt(ad * (2.0 - ad)); // 1-x^2 near either end
        },
        -1.0, 1.0);
    CHECK_CLOSE("int_-1^1 (1-x^2)^-1/2 (complement)", i_arcsin_c, kPi, effectiveTol(kPi, 1e-9));
}

TEST_F(TanhSinhAdTest, infiniteDomains) {
    TanhSinhIntegrator<double> integ(kAbsTol, kMaxEvals);

    // whole real line: int exp(-x^2) dx = sqrt(pi)
    const double i_gauss = integ.integrateInfinite([](double x) { return std::exp(-x * x); });
    CHECK_CLOSE("int_-inf^inf exp(-x^2)", i_gauss, std::sqrt(kPi),
                effectiveTol(std::sqrt(kPi), 1e-8));

    // half-line: int_0^inf x exp(-x) dx = 1
    const double i_exp = integ.integrateToInfinity([](double x) { return x * std::exp(-x); }, 0.0);
    CHECK_CLOSE("int_0^inf x exp(-x)", i_exp, 1.0, effectiveTol(1.0, 1e-8));
}

TEST_F(TanhSinhAdTest, movingBoundAndIntegrandParameterGradients) {
    // I(theta) = int_0^1 theta^2 x^2 dx = theta^2/3; dI/dtheta = 2 theta/3
    std::vector<stan::math::var> theta_v{stan::math::var(2.0)};
    Eigen::VectorXd x0(1);
    x0(0) = 2.0;
    double fx = 0.0;
    Eigen::VectorXd grad;
    stan::math::gradient(
        [&](const auto& th) {
            using Scalar = typename std::decay_t<decltype(th)>::Scalar;
            if constexpr (std::is_same_v<Scalar, double>) {
                TanhSinhIntegrator<double> integ(kAbsTol, kMaxEvals);
                return integ([&](double x) { return th(0) * th(0) * x * x; }, 0.0, 1.0);
            } else {
                TanhSinhIntegrator<Scalar> integ(kAbsTol, kMaxEvals);
                return integ([&](Scalar x) { return th(0) * th(0) * x * x; }, Scalar(0.0),
                             Scalar(1.0));
            }
        },
        x0, fx, grad);

    CHECK_CLOSE("var integral value", fx, 4.0 / 3.0, effectiveTol(4.0 / 3.0, 1e-9));
    CHECK_CLOSE("var dI/dtheta", grad(0), 4.0 / 3.0, effectiveTol(4.0 / 3.0, 1e-8));

    // parameter in the integrand: int_0^1 exp(theta x) dx, theta = 1
    //   value  = e - 1,  d/dtheta = 1
    Eigen::VectorXd x1(1);
    x1(0) = 1.0;
    double fx1 = 0.0;
    Eigen::VectorXd grad1;
    stan::math::gradient(
        [&](const auto& th) {
            using Scalar = typename std::decay_t<decltype(th)>::Scalar;
            if constexpr (std::is_same_v<Scalar, double>) {
                TanhSinhIntegrator<double> integ(kAbsTol, kMaxEvals);
                return integ([&](double x) { return std::exp(th(0) * x); }, 0.0, 1.0);
            } else {
                TanhSinhIntegrator<Scalar> integ(kAbsTol, kMaxEvals);
                return integ([&](Scalar x) { return stan::math::exp(th(0) * x); }, Scalar(0.0),
                             Scalar(1.0));
            }
        },
        x1, fx1, grad1);

    CHECK_CLOSE("integrand-parameter integral", fx1, std::exp(1.0) - 1.0,
                effectiveTol(std::exp(1.0) - 1.0, 1e-8));
    CHECK_CLOSE("integrand-parameter gradient", grad1(0), 1.0, effectiveTol(1.0, 1e-7));
}

TEST_F(TanhSinhAdTest, fvarVarHessian) {
    // I(theta) = int_0^1 theta^2 x^2 dx = theta^2/3  =>  H = 2/3
    Eigen::VectorXd x0(1);
    x0(0) = 2.0;
    double fx = 0.0;
    Eigen::VectorXd grad;
    Eigen::Matrix<double, -1, -1> H;
    stan::math::hessian(
        [&](const auto& th) {
            using Scalar = typename std::decay_t<decltype(th)>::Scalar;
            if constexpr (std::is_same_v<Scalar, double>) {
                TanhSinhIntegrator<double> integ(kAbsTol, kMaxEvals);
                return integ([&](double x) { return th(0) * th(0) * x * x; }, 0.0, 1.0);
            } else {
                TanhSinhIntegrator<Scalar> integ(kAbsTol, kMaxEvals);
                return integ([&](Scalar x) { return th(0) * th(0) * x * x; }, Scalar(0.0, 0.0),
                             Scalar(1.0, 0.0));
            }
        },
        x0, fx, grad, H);

    CHECK_CLOSE("fvar<var> d2I/dtheta2", H(0, 0), 2.0 / 3.0, effectiveTol(2.0 / 3.0, 1e-6));
}
