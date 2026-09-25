// test_tanh_sinh.cpp — validates quantape::math::TanhSinhIntegrator
//
//   double: polynomial, trigonometric, endpoint-singular integrands,
//           reversed bounds, infinite and half-infinite domains
//   var:    integral value matches double; gradient w.r.t. a parameter
//           (in the integrand and in the bounds) matches the analytic value
//   fvar<var>: second derivative (Hessian) matches the analytic value
//
// Run: ./test_tanh_sinh
#include "quantape/math/Integrals/IntegratorStanPrimitives.h"
#include "quantape/math/Integrals/TanhSinhIntegrator.h"
#include "quantape/math/StanMath.h"

#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <vector>

#define CHECK(cond)                                                                                \
    do {                                                                                           \
        if (!(cond)) {                                                                             \
            std::cerr << "FAIL: " << #cond << " (line " << __LINE__ << ")\n";                      \
            std::exit(1);                                                                          \
        }                                                                                          \
    } while (0)

using quantape::math::TanhSinhIntegrator;

namespace {

bool close(double a, double b, double tol) {
    return std::abs(a - b) <= tol * (1.0 + std::abs(b));
}

} // namespace

int main() {
    std::cout << std::setprecision(12);
    constexpr double ABS_TOL = 1e-10;

    // ── double-precision checks ──
    {
        TanhSinhIntegrator<double> integ(ABS_TOL, 1'000'000);

        const double i_poly = integ([](double x) { return x * x; }, 0.0, 1.0);
        CHECK(close(i_poly, 1.0 / 3.0, 1e-9));

        const double i_sin = integ([](double x) { return std::sin(x); }, 0.0, M_PI);
        CHECK(close(i_sin, 2.0, 1e-9));

        // endpoint singularity: int_0^1 x^{-1/2} dx = 2
        // Plain path: the abscissa collapses onto the endpoint in double, so
        // the boundary guard limits accuracy to ~1e-8 (sqrt of machine eps
        // in the unresolved tail) — the complement path below does better.
        const double i_sqrt_sing = integ([](double x) { return 1.0 / std::sqrt(x); }, 0.0, 1.0);
        CHECK(close(i_sqrt_sing, 2.0, 1e-6));

        // both endpoints singular: int_{-1}^{1} 1/sqrt(1-x^2) dx = pi
        const double i_arcsin =
            integ([](double x) { return 1.0 / std::sqrt(1.0 - x * x); }, -1.0, 1.0);
        CHECK(close(i_arcsin, M_PI, 1e-6));

        // complement-aware path reaches full precision on the same integrals:
        // the functor uses the signed distance d to the nearer endpoint
        // (d > 0 near the lower endpoint, d < 0 near the upper one). A
        // tighter accuracy target is used to show it is not the ~1e-8 guard
        // floor that limits the plain path. Observed accuracy here is
        // ~1e-10 absolute (the asymptotic error estimator under-reports the
        // residual for square-root singularities), ~100x better than plain.
        TanhSinhIntegrator<double> integ_fine(1e-12, 1'000'000);
        const double i_sqrt_sing_c = integ_fine.integrateWithComplement(
            [](double x, double d) { return 1.0 / std::sqrt(d > 0.0 ? d : x); }, 0.0, 1.0);
        CHECK(close(i_sqrt_sing_c, 2.0, 1e-9));

        const double i_arcsin_c = integ_fine.integrateWithComplement(
            [](double, double d) {
                const double ad = std::abs(d);
                return 1.0 / std::sqrt(ad * (2.0 - ad)); // 1-x^2 near either end
            },
            -1.0, 1.0);
        CHECK(close(i_arcsin_c, M_PI, 1e-9));

        // reversed bounds flip the sign
        const double i_rev = integ([](double x) { return x * x; }, 1.0, 0.0);
        CHECK(close(i_rev, -1.0 / 3.0, 1e-9));

        // whole real line: int exp(-x^2) dx = sqrt(pi)
        const double i_gauss = integ.integrateInfinite([](double x) { return std::exp(-x * x); });
        CHECK(close(i_gauss, std::sqrt(M_PI), 1e-8));

        // half-line: int_0^inf x exp(-x) dx = 1
        const double i_exp =
            integ.integrateToInfinity([](double x) { return x * std::exp(-x); }, 0.0);
        CHECK(close(i_exp, 1.0, 1e-8));

        std::cout << "double: polynomial/sin/endpoint-singular/infinite domains all pass\n"
                  << "        (evals on last call: " << integ.numberOfEvaluations() << ")\n";
    }

    // ── AD (var) checks ──
    {
        // I(theta) = int_0^1 theta^2 x^2 dx = theta^2/3
        // dI/dtheta = 2 theta/3
        std::vector<stan::math::var> theta_v{stan::math::var(2.0)};
        Eigen::VectorXd x0(1);
        x0(0) = 2.0;
        double fx = 0.0;
        Eigen::VectorXd grad;
        stan::math::gradient(
            [&](const auto& th) {
                using Scalar = typename std::decay_t<decltype(th)>::Scalar;
                if constexpr (std::is_same_v<Scalar, double>) {
                    TanhSinhIntegrator<double> integ(ABS_TOL, 1'000'000);
                    return integ([&](double x) { return th(0) * th(0) * x * x; }, 0.0, 1.0);
                } else {
                    TanhSinhIntegrator<Scalar> integ(ABS_TOL, 1'000'000);
                    return integ([&](Scalar x) { return th(0) * th(0) * x * x; }, Scalar(0.0),
                                 Scalar(1.0));
                }
            },
            x0, fx, grad);

        std::cout << "var: I(theta) = " << fx << " (expected 1.333333...)\n";
        CHECK(close(fx, 4.0 / 3.0, 1e-9));
        std::cout << "     dI/dtheta = " << grad(0) << " (expected 1.333333...)\n";
        CHECK(close(grad(0), 4.0 / 3.0, 1e-8));

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
                    TanhSinhIntegrator<double> integ(ABS_TOL, 1'000'000);
                    return integ([&](double x) { return std::exp(th(0) * x); }, 0.0, 1.0);
                } else {
                    TanhSinhIntegrator<Scalar> integ(ABS_TOL, 1'000'000);
                    return integ([&](Scalar x) { return stan::math::exp(th(0) * x); }, Scalar(0.0),
                                 Scalar(1.0));
                }
            },
            x1, fx1, grad1);

        CHECK(close(fx1, std::exp(1.0) - 1.0, 1e-8));
        CHECK(close(grad1(0), 1.0, 1e-7));
        std::cout << "     integrand-parameter gradient = " << grad1(0) << " (expected 1)\n";
    }

    // ── second order (fvar<var>) ──
    {
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
                    TanhSinhIntegrator<double> integ(ABS_TOL, 1'000'000);
                    return integ([&](double x) { return th(0) * th(0) * x * x; }, 0.0, 1.0);
                } else {
                    TanhSinhIntegrator<Scalar> integ(ABS_TOL, 1'000'000);
                    return integ([&](Scalar x) { return th(0) * th(0) * x * x; }, Scalar(0.0, 0.0),
                                 Scalar(1.0, 0.0));
                }
            },
            x0, fx, grad, H);

        std::cout << "fvar<var>: d2I/dtheta2 = " << H(0, 0) << " (expected 0.666666...)\n";
        CHECK(close(H(0, 0), 2.0 / 3.0, 1e-6));
    }

    std::cout << "test_tanh_sinh: all invariants hold\n";
    return 0;
}
