// test_integrator_primitives.cpp — validates the automatic AD dispatch of the
// quantape::math::Integrals classes (Math/Integrals/IntegratorStanPrimitives.h)
//
// Tested: TrapezoidIntegratorDefault, TrapezoidIntegratorMidPoint,
// SimpsonIntegrator, GaussLobattoIntegrator, GaussLegendreIntegrator(20),
// TanhSinhIntegrator. For each, with DoubleT = double / var / `fvar<var>`:
//
//   double:    value matches the analytic result (evaluation-count reference)
//   var:       value + gradient match, and the converged rule performs the
//              SAME number of integrand evaluations as the double path —
//              proving the one-pass rule extraction (no re-dispatch, no
//              per-node accumulation tape)
//   `fvar<var>`: value + gradient + Hessian (via stan::math::hessian) match
//   bounds:    dI/dtheta for I(theta) = int_0^theta x^2 dx matches theta^2
//
// Integrands:
//   theta0^2 x^2 + theta1 x on [0,1]: I = theta0^2/3 + theta1/2,
//       dI = (2 theta0/3, 1/2), H00 = 2/3 (exactly integrated by all methods)
//   exp(theta x) on [0,1]:            I = (e^theta - 1)/theta,
//       dI = (e^theta (theta-1) + 1)/theta^2,
//       H  = (e^theta (theta^2 - 2 theta + 2) - 2)/theta^3
//       (non-polynomial; all methods accurate to << 1e-6). x^theta is avoided
//       deliberately: forward-mode d/dtheta x^theta = x^theta ln x evaluates
//       to 0 * -inf = NaN at x = 0, which Stan's fvar produces while rev-mode
//       special-cases the same point.
//
// The legacy helper close(a, b, tol) was relative: |a-b| <= tol*(1+|b|).
// CHECK_CLOSE is absolute, so each call passes the effective absolute
// tolerance tol*(1+|expected|), keeping the contract identical.
#include "quantape/math/StanMath.h"

#include "quantape/math/Integrals/IntegratorStanPrimitives.h"

#include <cmath>
#include <cstddef>
#include <type_traits>
#include <vector>

#include "support/GtestSupport.h"
#include "support/StanTapeFixture.h"

using stan::math::var;

class IntegratorPrimitivesTest : public StanTapeTest {};

namespace {

double effectiveTol(double expected, double tol) {
    return tol * (1.0 + std::abs(expected));
}

constexpr double A = 0.0;
constexpr double B = 1.0;
constexpr size_t MAX_EVALS = 1'000'000;

// ── quadratic integrand ──
constexpr double TH0 = 2.0;
constexpr double TH1 = 3.0;
constexpr double I_REF = 17.0 / 6.0; // 4/3 + 3/2
constexpr double G0_REF = 4.0 / 3.0;
constexpr double G1_REF = 0.5;
constexpr double H00_REF = 2.0 / 3.0;

auto integrand = [](auto x, const auto& th) { return th[0] * th[0] * x * x + th[1] * x; };

// ── exp(theta x) integrand ──
constexpr double THETA2 = 2.5;
const double I2_REF = (std::exp(THETA2) - 1.0) / THETA2;
const double G2_REF = (std::exp(THETA2) * (THETA2 - 1.0) + 1.0) / (THETA2 * THETA2);
const double H2_REF =
    (std::exp(THETA2) * (THETA2 * THETA2 - 2.0 * THETA2 + 2.0) - 2.0) / (THETA2 * THETA2 * THETA2);

auto integrand2 = [](auto x, const auto& th) { return stan::math::exp(th[0] * x); };

/**
 * @brief Check one integrand against analytic value/gradient/Hessian for all
 *        three scalar types, including double/AD evaluation-count parity.
 *
 * @param hess_ref Flattened (row-major) Hessian reference, n x n.
 */
template <typename Factory, typename Fn>
void checkIntegrand(const Factory& factory, const Fn& f, const std::vector<double>& theta_d,
                    double value_ref, const std::vector<double>& grad_ref,
                    const std::vector<double>& hess_ref, double tol) {
    const size_t n = theta_d.size();

    // ── double: value + converged-rule evaluation count ──
    size_t n_evals = 0;
    {
        auto integ = factory(quantape::math::ScalarTag<double>{});
        const double I = integ([&](double x) { return f(x, theta_d); }, A, B);
        CHECK_CLOSE("double value", I, value_ref, effectiveTol(value_ref, tol));
        n_evals = integ.numberOfEvaluations();
        EXPECT_GT(n_evals, 0U);
    }

    // ── var: value + gradient + rule parity with double ──
    {
        std::vector<var> th;
        th.reserve(n);
        for (double t : theta_d)
            th.push_back(var(t));

        auto integ = factory(quantape::math::ScalarTag<var>{});
        var I = integ([&](auto x) { return f(x, th); }, var(A), var(B));
        I.grad();

        CHECK_CLOSE("var value", I.val(), value_ref, effectiveTol(value_ref, tol));
        for (size_t i = 0; i < n; ++i)
            CHECK_CLOSE("var gradient", th[i].adj(), grad_ref[i], effectiveTol(grad_ref[i], tol));
        EXPECT_EQ(integ.numberOfEvaluations(), n_evals);
    }

    // ── `fvar<var>`: value + gradient + Hessian ──
    {
        Eigen::VectorXd x0(n);
        for (size_t i = 0; i < n; ++i)
            x0(static_cast<Eigen::Index>(i)) = theta_d[i];

        const auto runner = [&](const auto& th_eig) {
            using S = typename std::decay_t<decltype(th_eig)>::Scalar;
            std::vector<S> th(th_eig.data(), th_eig.data() + th_eig.size());
            auto integ = factory(quantape::math::ScalarTag<S>{});
            return integ([&](auto x) { return f(x, th); }, S(A), S(B));
        };

        double fx = 0.0;
        Eigen::VectorXd grad;
        Eigen::Matrix<double, -1, -1> H;
        stan::math::hessian(runner, x0, fx, grad, H);

        CHECK_CLOSE("fvar value", fx, value_ref, effectiveTol(value_ref, tol));
        for (size_t i = 0; i < n; ++i)
            CHECK_CLOSE("fvar gradient", grad(static_cast<Eigen::Index>(i)), grad_ref[i],
                        effectiveTol(grad_ref[i], tol));
        for (size_t i = 0; i < n; ++i)
            for (size_t j = 0; j < n; ++j)
                CHECK_CLOSE("fvar hessian",
                            H(static_cast<Eigen::Index>(i), static_cast<Eigen::Index>(j)),
                            hess_ref[i * n + j], effectiveTol(hess_ref[i * n + j], tol));
    }
}

template <typename Factory>
void runIntegrator(const char* name, const Factory& factory, double tol_quad, double tol_pow) {
    SCOPED_TRACE(name);

    // polynomial: exactly integrated by the polynomial rules
    checkIntegrand(factory, integrand, {TH0, TH1}, I_REF, {G0_REF, G1_REF},
                   {H00_REF, 0.0, 0.0, 0.0}, tol_quad);
    // non-polynomial: all methods accurate to ~1e-9 here (GL20: ~2e-9)
    checkIntegrand(factory, integrand2, {THETA2}, I2_REF, {G2_REF}, {H2_REF}, tol_pow);

    // bounds carrying parameters: I(theta) = int_0^theta x^2 dx = theta^3/3,
    // dI/dtheta = theta^2 (frozen rule, affine nodes + interval-scaled weights)
    var theta = 2.0;
    auto integ = factory(quantape::math::ScalarTag<var>{});
    var I = integ([](auto x) { return x * x; }, var(0.0), theta);
    I.grad();
    CHECK_CLOSE("moving-bound value", I.val(), 8.0 / 3.0, effectiveTol(8.0 / 3.0, tol_quad));
    CHECK_CLOSE("moving-bound gradient", theta.adj(), 4.0, effectiveTol(4.0, tol_quad));
}

auto trapezoid = [](auto tag) {
    using S = typename decltype(tag)::type;
    return quantape::math::TrapezoidIntegratorDefault<S>(1e-9, MAX_EVALS);
};
// MidPointPolicy's nodes do not nest across the 3x refinement, so its
// recurrence converges only linearly (~1/3 error decay per level). The
// values are correct but ~1e-9 would need ~1e9 evaluations; use a
// tolerance the policy reaches in ~2e4 evaluations instead.
auto trapezoid_mid = [](auto tag) {
    using S = typename decltype(tag)::type;
    return quantape::math::TrapezoidIntegratorMidPoint<S>(1e-3, MAX_EVALS);
};
auto simpson = [](auto tag) {
    using S = typename decltype(tag)::type;
    return quantape::math::SimpsonIntegrator<S>(1e-10, MAX_EVALS);
};
auto lobatto = [](auto tag) {
    using S = typename decltype(tag)::type;
    return quantape::math::GaussLobattoIntegrator<S>(1e-9, MAX_EVALS);
};
auto legendre = [](auto tag) {
    using S = typename decltype(tag)::type;
    return quantape::math::GaussLegendreIntegrator<S>(20);
};
auto tanh_sinh = [](auto tag) {
    using S = typename decltype(tag)::type;
    return quantape::math::TanhSinhIntegrator<S>(1e-10, MAX_EVALS);
};

} // namespace

TEST_F(IntegratorPrimitivesTest, trapezoidDefault) {
    runIntegrator("trapezoid (default)", trapezoid, 1e-8, 1e-6);
}

TEST_F(IntegratorPrimitivesTest, trapezoidMidpoint) {
    runIntegrator("trapezoid (midpoint)", trapezoid_mid, 1e-3, 1e-3);
}

TEST_F(IntegratorPrimitivesTest, simpson) {
    runIntegrator("simpson", simpson, 1e-8, 1e-6);
}

TEST_F(IntegratorPrimitivesTest, gaussLobatto) {
    runIntegrator("gauss-lobatto", lobatto, 1e-8, 1e-6);
}

TEST_F(IntegratorPrimitivesTest, gaussLegendre20) {
    runIntegrator("gauss-legendre (20)", legendre, 1e-8, 1e-6);
}

TEST_F(IntegratorPrimitivesTest, tanhSinh) {
    runIntegrator("tanh-sinh", tanh_sinh, 1e-8, 1e-6);
}
