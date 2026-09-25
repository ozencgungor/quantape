// test_integrator_primitives.cpp — validates the automatic AD dispatch of the
// Math::Integrals classes (Math/Integrals/IntegratorStanPrimitives.h)
//
// Tested: TrapezoidIntegratorDefault, TrapezoidIntegratorMidPoint,
// SimpsonIntegrator, GaussLobattoIntegrator, GaussLegendreIntegrator(20),
// TanhSinhIntegrator. For each, with DoubleT = double / var / fvar<var>:
//
//   double:    value matches the analytic result (evaluation-count reference)
//   var:       value + gradient match, and the converged rule performs the
//              SAME number of integrand evaluations as the double path —
//              proving the one-pass rule extraction (no re-dispatch, no
//              per-node accumulation tape)
//   fvar<var>: value + gradient + Hessian (via stan::math::hessian) match
//   bounds:    dI/dtheta for I(theta) = int_0^theta x^2 dx matches theta^2
//
// Integrands:
//   theta0^2 x^2 + theta1 x on [0,1]: I = theta0^2/3 + theta1/2,
//       dI = (2 theta0/3, 1/2), H00 = 2/3 (exactly integrated by all methods)
//   x^theta on [0,1]:                 I = 1/(1+theta), dI = -1/(1+theta)^2,
//       H = 2/(1+theta)^3 (non-polynomial; all methods accurate to << 1e-6)
//
// Run: ./test_integrator_primitives
#include "Math/Integrals/IntegratorStanPrimitives.h"
#include "Math/StanMath.h"

#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <type_traits>
#include <vector>

#define CHECK(cond)                                                                                \
    do {                                                                                           \
        if (!(cond)) {                                                                             \
            std::cerr << "FAIL: " << #cond << " (line " << __LINE__ << ")\n";                      \
            std::exit(1);                                                                          \
        }                                                                                          \
    } while (0)

using stan::math::fvar;
using stan::math::var;

namespace {

bool close(double a, double b, double tol) {
    return std::abs(a - b) <= tol * (1.0 + std::abs(b));
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

// ── x^theta integrand ──
constexpr double THETA2 = 2.5;
constexpr double I2_REF = 1.0 / 3.5;
constexpr double G2_REF = -1.0 / (3.5 * 3.5);
constexpr double H2_REF = 2.0 / (3.5 * 3.5 * 3.5);

auto integrand2 = [](auto x, const auto& th) { return stan::math::pow(x, th[0]); };

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
        auto integ = factory(Math::ScalarTag<double>{});
        const double I = integ([&](double x) { return f(x, theta_d); }, A, B);
        CHECK(close(I, value_ref, tol));
        n_evals = integ.numberOfEvaluations();
        CHECK(n_evals > 0);
    }

    // ── var: value + gradient + rule parity with double ──
    {
        stan::math::recover_memory();
        std::vector<var> th;
        th.reserve(n);
        for (double t : theta_d)
            th.push_back(var(t));

        auto integ = factory(Math::ScalarTag<var>{});
        var I = integ([&](auto x) { return f(x, th); }, var(A), var(B));
        I.grad();

        CHECK(close(I.val(), value_ref, tol));
        for (size_t i = 0; i < n; ++i)
            CHECK(close(th[i].adj(), grad_ref[i], tol));
        CHECK(integ.numberOfEvaluations() == n_evals);
    }

    // ── fvar<var>: value + gradient + Hessian ──
    {
        stan::math::recover_memory();

        Eigen::VectorXd x0(n);
        for (size_t i = 0; i < n; ++i)
            x0(static_cast<Eigen::Index>(i)) = theta_d[i];

        const auto runner = [&](const auto& th_eig) {
            using S = typename std::decay_t<decltype(th_eig)>::Scalar;
            std::vector<S> th(th_eig.data(), th_eig.data() + th_eig.size());
            auto integ = factory(Math::ScalarTag<S>{});
            return integ([&](auto x) { return f(x, th); }, S(A), S(B));
        };

        double fx = 0.0;
        Eigen::VectorXd grad;
        Eigen::Matrix<double, -1, -1> H;
        stan::math::hessian(runner, x0, fx, grad, H);

        CHECK(close(fx, value_ref, tol));
        for (size_t i = 0; i < n; ++i)
            CHECK(close(grad(static_cast<Eigen::Index>(i)), grad_ref[i], tol));
        for (size_t i = 0; i < n; ++i)
            for (size_t j = 0; j < n; ++j)
                CHECK(close(H(static_cast<Eigen::Index>(i), static_cast<Eigen::Index>(j)),
                            hess_ref[i * n + j], tol));
    }
}

template <typename Factory>
void runIntegrator(const char* name, const Factory& factory, double tol_quad, double tol_pow) {
    // polynomial: exactly integrated by the polynomial rules
    checkIntegrand(factory, integrand, {TH0, TH1}, I_REF, {G0_REF, G1_REF},
                   {H00_REF, 0.0, 0.0, 0.0}, tol_quad);
    // non-polynomial: all methods accurate to ~1e-9 here (GL20: ~2e-9)
    checkIntegrand(factory, integrand2, {THETA2}, I2_REF, {G2_REF}, {H2_REF}, tol_pow);

    // bounds carrying parameters: I(theta) = int_0^theta x^2 dx = theta^3/3,
    // dI/dtheta = theta^2 (frozen rule, affine nodes + interval-scaled weights)
    {
        stan::math::recover_memory();
        var theta = 2.0;
        auto integ = factory(Math::ScalarTag<var>{});
        var I = integ([](auto x) { return x * x; }, var(0.0), theta);
        I.grad();
        CHECK(close(I.val(), 8.0 / 3.0, tol_quad));
        CHECK(close(theta.adj(), 4.0, tol_quad));
    }

    std::cout << "  " << name << ": double/var/fvar<var> vs analytic OK\n";
}

} // namespace

int main() {
    std::cout << std::setprecision(12);

    auto trapezoid = [](auto tag) {
        using S = typename decltype(tag)::type;
        return Math::TrapezoidIntegratorDefault<S>(1e-9, MAX_EVALS);
    };
    // MidPointPolicy's nodes do not nest across the 3x refinement, so its
    // recurrence converges only linearly (~1/3 error decay per level). The
    // values are correct but ~1e-9 would need ~1e9 evaluations; use a
    // tolerance the policy reaches in ~2e4 evaluations instead.
    auto trapezoid_mid = [](auto tag) {
        using S = typename decltype(tag)::type;
        return Math::TrapezoidIntegratorMidPoint<S>(1e-3, MAX_EVALS);
    };
    auto simpson = [](auto tag) {
        using S = typename decltype(tag)::type;
        return Math::SimpsonIntegrator<S>(1e-10, MAX_EVALS);
    };
    auto lobatto = [](auto tag) {
        using S = typename decltype(tag)::type;
        return Math::GaussLobattoIntegrator<S>(1e-9, MAX_EVALS);
    };
    auto legendre = [](auto tag) {
        using S = typename decltype(tag)::type;
        return Math::GaussLegendreIntegrator<S>(20);
    };
    auto tanh_sinh = [](auto tag) {
        using S = typename decltype(tag)::type;
        return Math::TanhSinhIntegrator<S>(1e-10, MAX_EVALS);
    };

    std::cout << "── analytic value/gradient/Hessian, all scalar types ──\n";
    runIntegrator("trapezoid (default) ", trapezoid, 1e-8, 1e-6);
    runIntegrator("trapezoid (midpoint)", trapezoid_mid, 1e-3, 1e-3);
    runIntegrator("simpson             ", simpson, 1e-8, 1e-6);
    runIntegrator("gauss-lobatto       ", lobatto, 1e-8, 1e-6);
    runIntegrator("gauss-legendre (20) ", legendre, 1e-8, 1e-6);
    runIntegrator("tanh-sinh           ", tanh_sinh, 1e-8, 1e-6);

    std::cout << "test_integrator_primitives: all invariants hold\n";
    return 0;
}
