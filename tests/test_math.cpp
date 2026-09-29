#include "quantape/math/NumericalMethods.h"

#include <cmath>
#include <iomanip>
#include <iostream>

#include "quantape/log/Log.h"
#include "quantape/util/Check.h"

using namespace quantape::math;

void testIntegration() {
    QTA_LOG_INFO("test", "=== Integration Tests ===");

    // Test function: f(x) = x^2, integral from 0 to 1 should be 1/3
    auto f_square = [](double x) { return x * x; };

    // Test 1: Trapezoid integrator with default policy
    {
        TrapezoidIntegratorDefault<double> integrator(1e-8, 1000);
        double result = integrator(f_square, 0.0, 1.0);
        double exact = 1.0 / 3.0;
        QTA_LOG_INFO("test", "Trapezoid (Default Policy):");
        QTA_LOG_INFO("test", "  ∫₀¹ x² dx = {}", quantape::util::num(result));
        QTA_LOG_INFO("test", "  Exact     = {}", quantape::util::num(exact));
        QTA_LOG_INFO("test", "  Error     = {}",
                     quantape::util::num(std::fabs(result - exact)));
        QTA_LOG_INFO("test", "  Evals     = {}", integrator.numberOfEvaluations());
    }

    // Test 2: Trapezoid integrator with midpoint policy
    {
        TrapezoidIntegratorMidPoint<double> integrator(1e-8, 1000);
        double result = integrator(f_square, 0.0, 1.0);
        double exact = 1.0 / 3.0;
        QTA_LOG_INFO("test", "Trapezoid (MidPoint Policy):");
        QTA_LOG_INFO("test", "  ∫₀¹ x² dx = {}", quantape::util::num(result));
        QTA_LOG_INFO("test", "  Exact     = {}", quantape::util::num(exact));
        QTA_LOG_INFO("test", "  Error     = {}",
                     quantape::util::num(std::fabs(result - exact)));
        QTA_LOG_INFO("test", "  Evals     = {}", integrator.numberOfEvaluations());
    }

    // Test 3: Simpson integrator
    {
        SimpsonIntegrator<double> integrator(1e-8, 1000);
        double result = integrator(f_square, 0.0, 1.0);
        double exact = 1.0 / 3.0;
        QTA_LOG_INFO("test", "Simpson's Rule:");
        QTA_LOG_INFO("test", "  ∫₀¹ x² dx = {}", quantape::util::num(result));
        QTA_LOG_INFO("test", "  Exact     = {}", quantape::util::num(exact));
        QTA_LOG_INFO("test", "  Error     = {}",
                     quantape::util::num(std::fabs(result - exact)));
        QTA_LOG_INFO("test", "  Evals     = {}", integrator.numberOfEvaluations());
    }

    // Test 4: Gauss-Legendre integrator (various orders)
    {
        double exact = 1.0 / 3.0;
        for (size_t order : {2, 3, 5, 10, 20}) {
            GaussLegendreIntegrator<double> integrator(order);
            double result = integrator(f_square, 0.0, 1.0);
            QTA_LOG_INFO("test", "Gauss-Legendre (order {}):", order);
            QTA_LOG_INFO("test", "  ∫₀¹ x² dx = {}", quantape::util::num(result));
            QTA_LOG_INFO("test", "  Error     = {}",
                         quantape::util::num(std::fabs(result - exact)));
            QTA_LOG_INFO("test", "  Evals     = {}", integrator.numberOfEvaluations());
        }
    }

    // Test 5: More challenging integral - sin(x) from 0 to π
    {
        auto f_sin = [](double x) { return std::sin(x); };
        double exact = 2.0; // ∫₀^π sin(x) dx = 2

        TrapezoidIntegratorDefault<double> trap(1e-8, 1000);
        double result_trap = trap(f_sin, 0.0, M_PI);

        SimpsonIntegrator<double> simpson(1e-8, 1000);
        double result_simpson = simpson(f_sin, 0.0, M_PI);

        GaussLegendreIntegrator<double> gauss(20);
        double result_gauss = gauss(f_sin, 0.0, M_PI);

        QTA_LOG_INFO("test", "Integral of sin(x) from 0 to π:");
        QTA_LOG_INFO("test", "  Trapezoid  = {} (error: {})",
                     quantape::util::num(result_trap),
                     quantape::util::num(std::fabs(result_trap - exact)));
        QTA_LOG_INFO("test", "  Simpson    = {} (error: {})",
                     quantape::util::num(result_simpson),
                     quantape::util::num(std::fabs(result_simpson - exact)));
        QTA_LOG_INFO("test", "  Gauss-20   = {} (error: {})",
                     quantape::util::num(result_gauss),
                     quantape::util::num(std::fabs(result_gauss - exact)));
        QTA_LOG_INFO("test", "  Exact      = {}", quantape::util::num(exact));
    }
}

void testSolvers() {
    QTA_LOG_INFO("test", "=== Root Finding Tests ===");

    // Test 1: f(x) = x^2 - 2, root at x = sqrt(2)
    {
        auto f = [](double x) { return x * x - 2.0; };
        double exact = std::sqrt(2.0);

        QTA_LOG_INFO("test",
                     "Finding root of x² - 2 = 0 (exact: √2 = {}):", quantape::util::num(exact));

        // Bisection
        {
            BisectionSolver<double> solver;
            solver.setMaxEvaluations(100);
            double root = solver.solve(f, 1e-10, 1.5, 0.0, 3.0);
            QTA_LOG_INFO("test", "  Bisection:  root = {} (error: {})",
                         quantape::util::num(root), quantape::util::num(std::fabs(root - exact)));
        }

        // Secant
        {
            SecantSolver<double> solver;
            solver.setMaxEvaluations(100);
            double root = solver.solve(f, 1e-10, 1.5, 0.0, 3.0);
            QTA_LOG_INFO("test", "  Secant:     root = {} (error: {})",
                         quantape::util::num(root), quantape::util::num(std::fabs(root - exact)));
        }

        // Newton (with finite differences)
        {
            NewtonSolver<double> solver;
            solver.setMaxEvaluations(100);
            double root = solver.solve(f, 1e-10, 1.5, 0.0, 3.0);
            QTA_LOG_INFO("test", "  Newton:     root = {} (error: {})",
                         quantape::util::num(root), quantape::util::num(std::fabs(root - exact)));
        }

        // Brent
        {
            BrentSolver<double> solver;
            solver.setMaxEvaluations(100);
            double root = solver.solve(f, 1e-10, 1.5, 0.0, 3.0);
            QTA_LOG_INFO("test", "  Brent:      root = {} (error: {})",
                         quantape::util::num(root), quantape::util::num(std::fabs(root - exact)));
        }

        // Ridder
        {
            RidderSolver<double> solver;
            solver.setMaxEvaluations(100);
            double root = solver.solve(f, 1e-10, 1.5, 0.0, 3.0);
            QTA_LOG_INFO("test", "  Ridder:     root = {} (error: {})",
                         quantape::util::num(root), quantape::util::num(std::fabs(root - exact)));
        }

        // False Position
        {
            FalsePositionSolver<double> solver;
            solver.setMaxEvaluations(100);
            double root = solver.solve(f, 1e-10, 1.5, 0.0, 3.0);
            QTA_LOG_INFO("test", "  FalsePos:   root = {} (error: {})",
                         quantape::util::num(root), quantape::util::num(std::fabs(root - exact)));
        }
    }

    // Test 2: f(x) = exp(x) - 3, root at x = ln(3)
    {
        auto f = [](double x) { return std::exp(x) - 3.0; };
        double exact = std::log(3.0);

        QTA_LOG_INFO("test", "Finding root of exp(x) - 3 = 0 (exact: ln(3) = {}):",
                     quantape::util::num(exact));

        BrentSolver<double> solver;
        double root = solver.solve(f, 1e-10, 1.0, 0.0, 2.0);

        QTA_LOG_INFO("test", "  Brent:      root = {} (error: {})",
                     quantape::util::num(root), quantape::util::num(std::fabs(root - exact)));
        QTA_LOG_INFO("test", "  f(root)     = {}", quantape::util::num(f(root)));
    }

    // Test 3: Automatic bracketing
    {
        auto f = [](double x) { return x * x * x - x - 2.0; }; // Root at x ≈ 1.521

        BrentSolver<double> solver;
        double root = solver.solve(f, 1e-10, 1.5, 0.1); // Auto-bracket from guess with step

        QTA_LOG_INFO("test", "Finding root of x³ - x - 2 = 0 with auto-bracketing:");
        QTA_LOG_INFO("test", "  Brent:      root = {}", quantape::util::num(root));
        QTA_LOG_INFO("test", "  f(root)     = {}", quantape::util::num(f(root)));
    }

    // Test 4: Newton with explicit derivative
    {
        auto f = [](double x) { return x * x - 2.0; };
        auto df = [](double x) { return 2.0 * x; };
        double exact = std::sqrt(2.0);

        // Derivative is a template parameter (no std::function / setDerivative)
        NewtonSolverWithDerivative<double, decltype(df)> solver(df);
        solver.setMaxEvaluations(100);
        double root = solver.solve(f, 1e-10, 1.5, 0.0, 3.0);

        QTA_LOG_INFO("test", "Newton with explicit derivative for x² - 2 = 0:");
        QTA_LOG_INFO("test", "  Root        = {} (error: {})", quantape::util::num(root),
                     quantape::util::num(std::fabs(root - exact)));
    }
}

void testQuadratureOnStandardDomain() {
    QTA_LOG_INFO("test", "=== Quadrature on Standard Domain [-1, 1] ===");

    // Test polynomial integration (Gauss quadrature is exact for polynomials)
    // For order n, exact for polynomials up to degree 2n-1

    auto poly2 = [](double x) { return 1.0 + 2.0 * x + 3.0 * x * x; };                 // degree 2
    auto poly4 = [](double x) { return 1.0 + x + x * x + x * x * x + x * x * x * x; }; // degree 4

    // Exact integral of poly2 from -1 to 1: ∫(1 + 2x + 3x²)dx = [x + x² + x³]_{-1}^{1} = 4
    // Exact integral of poly4 from -1 to 1: ∫(1 + x + x² + x³ + x⁴)dx = 16/5

    double exact_poly2 = 4.0;
    double exact_poly4 = 16.0 / 5.0;

    QTA_LOG_INFO("test", "Polynomial degree 2: 1 + 2x + 3x²");
    QTA_LOG_INFO("test", "Exact integral [-1, 1]: {}", quantape::util::num(exact_poly2));

    for (size_t order : {2, 3, 5}) {
        GaussLegendreQuadrature<double> quad(order);
        double result = quad.integrate(poly2, -1.0, 1.0);
        QTA_LOG_INFO("test", "  Order {}: {} (error: {})", order,
                     quantape::util::num(result),
                     quantape::util::num(std::fabs(result - exact_poly2)));
    }

    QTA_LOG_INFO("test", "Polynomial degree 4: 1 + x + x² + x³ + x⁴");
    QTA_LOG_INFO("test", "Exact integral [-1, 1]: {}", quantape::util::num(exact_poly4));

    for (size_t order : {2, 3, 5}) {
        GaussLegendreQuadrature<double> quad(order);
        double result = quad.integrate(poly4, -1.0, 1.0);
        QTA_LOG_INFO("test", "  Order {}: {} (error: {})", order,
                     quantape::util::num(result),
                     quantape::util::num(std::fabs(result - exact_poly4)));
    }
}

int main() {
    try {
        testIntegration();
        testQuadratureOnStandardDomain();
        testSolvers();

        QTA_LOG_INFO("test", "All tests completed successfully!");
        return 0;
    } catch (const std::exception& e) {
        QTA_LOG_ERROR("test", "Error: {}", e.what());
        return 1;
    }
}
