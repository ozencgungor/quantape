// test_math.cpp — NumericalMethods integrator, quadrature and solver demos.
//
// The legacy TU was NOGATE (it printed computed errors without gating them);
// the original values are preserved here as RecordProperty diagnostics. Gate
// backfill is inventory §3.3 Math.IntegratorConvergenceOrders (P2, separate
// coverage workstream).

#include "quantape/math/NumericalMethods.h"
#include "quantape/util/Constants.h"

#include <cmath>
#include <cstddef>
#include <string>

#include "support/GtestSupport.h"

using namespace quantape::math;
using quantape::util::kPi;

namespace {

void recordNumber(const std::string& name, double value) {
    ::testing::Test::RecordProperty(name, quantape::util::num(value, 12));
}

} // namespace

TEST(MathIntegrators, trapezoidAndSimpsonPolynomial) {
    // Test function: f(x) = x^2, integral from 0 to 1 should be 1/3
    auto f_square = [](double x) { return x * x; };
    const double exact = 1.0 / 3.0;

    TrapezoidIntegratorDefault<double> trapezoid_default(1e-8, 1000);
    recordNumber("trapezoid_default_error",
                 std::fabs(trapezoid_default(f_square, 0.0, 1.0) - exact));

    TrapezoidIntegratorMidPoint<double> trapezoid_midpoint(1e-8, 1000);
    recordNumber("trapezoid_midpoint_error",
                 std::fabs(trapezoid_midpoint(f_square, 0.0, 1.0) - exact));

    SimpsonIntegrator<double> simpson(1e-8, 1000);
    recordNumber("simpson_error", std::fabs(simpson(f_square, 0.0, 1.0) - exact));
}

TEST(MathIntegrators, gaussLegendreOrders) {
    auto f_square = [](double x) { return x * x; };
    const double exact = 1.0 / 3.0;

    for (std::size_t order : {2, 3, 5, 10, 20}) {
        GaussLegendreIntegrator<double> integrator(order);
        recordNumber("gauss_legendre_order_" + std::to_string(order) + "_error",
                     std::fabs(integrator(f_square, 0.0, 1.0) - exact));
    }
}

TEST(MathIntegrators, oscillatorySineIntegral) {
    auto f_sin = [](double x) { return std::sin(x); };
    const double exact = 2.0; // ∫₀^π sin(x) dx = 2

    TrapezoidIntegratorDefault<double> trap(1e-8, 1000);
    recordNumber("sine_trapezoid_error", std::fabs(trap(f_sin, 0.0, kPi) - exact));

    SimpsonIntegrator<double> simpson(1e-8, 1000);
    recordNumber("sine_simpson_error", std::fabs(simpson(f_sin, 0.0, kPi) - exact));

    GaussLegendreIntegrator<double> gauss(20);
    recordNumber("sine_gauss20_error", std::fabs(gauss(f_sin, 0.0, kPi) - exact));
}

TEST(MathQuadrature, polynomialExactnessOnStandardDomain) {
    // Test polynomial integration (Gauss quadrature is exact for polynomials):
    // for order n, exact for polynomials up to degree 2n-1.
    auto poly2 = [](double x) { return 1.0 + 2.0 * x + 3.0 * x * x; };                 // degree 2
    auto poly4 = [](double x) { return 1.0 + x + x * x + x * x * x + x * x * x * x; }; // degree 4

    // Exact integral of poly2 from -1 to 1: ∫(1 + 2x + 3x²)dx = [x + x² + x³]_{-1}^{1} = 4
    // Exact integral of poly4 from -1 to 1: ∫(1 + x + x² + x³ + x⁴)dx = 46/15
    // (the legacy file printed 16/5 here, which is wrong)
    const double exact_poly2 = 4.0;
    const double exact_poly4 = 46.0 / 15.0;

    for (std::size_t order : {2, 3, 5}) {
        GaussLegendreQuadrature<double> quad(order);
        recordNumber("poly2_order_" + std::to_string(order) + "_error",
                     std::fabs(quad.integrate(poly2, -1.0, 1.0) - exact_poly2));
    }

    for (std::size_t order : {2, 3, 5}) {
        GaussLegendreQuadrature<double> quad(order);
        recordNumber("poly4_order_" + std::to_string(order) + "_error",
                     std::fabs(quad.integrate(poly4, -1.0, 1.0) - exact_poly4));
    }
}

TEST(MathSolvers, rootFindersMatchAnalytic) {
    // f(x) = x^2 - 2, root at x = sqrt(2)
    auto f = [](double x) { return x * x - 2.0; };
    const double exact = std::sqrt(2.0);

    BisectionSolver<double> bisection;
    bisection.setMaxEvaluations(100);
    recordNumber("bisection_x2_minus_2_error",
                 std::fabs(bisection.solve(f, 1e-10, 1.5, 0.0, 3.0) - exact));

    SecantSolver<double> secant;
    secant.setMaxEvaluations(100);
    recordNumber("secant_x2_minus_2_error",
                 std::fabs(secant.solve(f, 1e-10, 1.5, 0.0, 3.0) - exact));

    NewtonSolver<double> newton;
    newton.setMaxEvaluations(100);
    recordNumber("newton_x2_minus_2_error",
                 std::fabs(newton.solve(f, 1e-10, 1.5, 0.0, 3.0) - exact));

    BrentSolver<double> brent;
    brent.setMaxEvaluations(100);
    recordNumber("brent_x2_minus_2_error", std::fabs(brent.solve(f, 1e-10, 1.5, 0.0, 3.0) - exact));

    RidderSolver<double> ridder;
    ridder.setMaxEvaluations(100);
    recordNumber("ridder_x2_minus_2_error",
                 std::fabs(ridder.solve(f, 1e-10, 1.5, 0.0, 3.0) - exact));

    FalsePositionSolver<double> falsepos;
    falsepos.setMaxEvaluations(100);
    recordNumber("falsepos_x2_minus_2_error",
                 std::fabs(falsepos.solve(f, 1e-10, 1.5, 0.0, 3.0) - exact));
}

TEST(MathSolvers, bracketingAndExplicitDerivative) {
    // f(x) = exp(x) - 3, root at x = ln(3)
    auto f_exp = [](double x) { return std::exp(x) - 3.0; };
    const double exact_exp = std::log(3.0);

    BrentSolver<double> brent;
    recordNumber("brent_exp_minus_3_error",
                 std::fabs(brent.solve(f_exp, 1e-10, 1.0, 0.0, 2.0) - exact_exp));

    // Automatic bracketing: f(x) = x^3 - x - 2 (root at x ~ 1.521)
    auto f_cubic = [](double x) { return x * x * x - x - 2.0; };
    BrentSolver<double> auto_bracket;
    recordNumber("brent_auto_bracket_root", auto_bracket.solve(f_cubic, 1e-10, 1.5, 0.1));

    // Newton with explicit derivative (template parameter, no std::function)
    auto f_square = [](double x) { return x * x - 2.0; };
    auto df_square = [](double x) { return 2.0 * x; };
    NewtonSolverWithDerivative<double, decltype(df_square)> newton_explicit(df_square);
    newton_explicit.setMaxEvaluations(100);
    recordNumber("newton_explicit_x2_minus_2_error",
                 std::fabs(newton_explicit.solve(f_square, 1e-10, 1.5, 0.0, 3.0) - std::sqrt(2.0)));
}
