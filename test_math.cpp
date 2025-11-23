#include "Math/NumericalMethods.h"
#include <iostream>
#include <iomanip>
#include <cmath>

using namespace Math;

void testIntegration() {
    std::cout << "=== Integration Tests ===\n\n";

    // Test function: f(x) = x^2, integral from 0 to 1 should be 1/3
    auto f_square = [](double x) { return x * x; };

    // Test 1: Trapezoid integrator with default policy
    {
        TrapezoidIntegratorDefault<double> integrator(1e-8, 1000);
        double result = integrator(f_square, 0.0, 1.0);
        double exact = 1.0 / 3.0;
        std::cout << "Trapezoid (Default Policy):\n";
        std::cout << "  ∫₀¹ x² dx = " << std::setprecision(10) << result << "\n";
        std::cout << "  Exact     = " << exact << "\n";
        std::cout << "  Error     = " << std::fabs(result - exact) << "\n";
        std::cout << "  Evals     = " << integrator.numberOfEvaluations() << "\n\n";
    }

    // Test 2: Trapezoid integrator with midpoint policy
    {
        TrapezoidIntegratorMidPoint<double> integrator(1e-8, 1000);
        double result = integrator(f_square, 0.0, 1.0);
        double exact = 1.0 / 3.0;
        std::cout << "Trapezoid (MidPoint Policy):\n";
        std::cout << "  ∫₀¹ x² dx = " << std::setprecision(10) << result << "\n";
        std::cout << "  Exact     = " << exact << "\n";
        std::cout << "  Error     = " << std::fabs(result - exact) << "\n";
        std::cout << "  Evals     = " << integrator.numberOfEvaluations() << "\n\n";
    }

    // Test 3: Simpson integrator
    {
        SimpsonIntegrator<double> integrator(1e-8, 1000);
        double result = integrator(f_square, 0.0, 1.0);
        double exact = 1.0 / 3.0;
        std::cout << "Simpson's Rule:\n";
        std::cout << "  ∫₀¹ x² dx = " << std::setprecision(10) << result << "\n";
        std::cout << "  Exact     = " << exact << "\n";
        std::cout << "  Error     = " << std::fabs(result - exact) << "\n";
        std::cout << "  Evals     = " << integrator.numberOfEvaluations() << "\n\n";
    }

    // Test 4: Gauss-Legendre integrator (various orders)
    {
        double exact = 1.0 / 3.0;
        for (size_t order : {2, 3, 5, 10, 20}) {
            GaussLegendreIntegrator<double> integrator(order);
            double result = integrator(f_square, 0.0, 1.0);
            std::cout << "Gauss-Legendre (order " << order << "):\n";
            std::cout << "  ∫₀¹ x² dx = " << std::setprecision(10) << result << "\n";
            std::cout << "  Error     = " << std::fabs(result - exact) << "\n";
            std::cout << "  Evals     = " << integrator.numberOfEvaluations() << "\n\n";
        }
    }

    // Test 5: More challenging integral - sin(x) from 0 to π
    {
        auto f_sin = [](double x) { return std::sin(x); };
        double exact = 2.0;  // ∫₀^π sin(x) dx = 2

        TrapezoidIntegratorDefault<double> trap(1e-8, 1000);
        double result_trap = trap(f_sin, 0.0, M_PI);

        SimpsonIntegrator<double> simpson(1e-8, 1000);
        double result_simpson = simpson(f_sin, 0.0, M_PI);

        GaussLegendreIntegrator<double> gauss(20);
        double result_gauss = gauss(f_sin, 0.0, M_PI);

        std::cout << "Integral of sin(x) from 0 to π:\n";
        std::cout << "  Trapezoid  = " << std::setprecision(10) << result_trap
                  << " (error: " << std::fabs(result_trap - exact) << ")\n";
        std::cout << "  Simpson    = " << result_simpson
                  << " (error: " << std::fabs(result_simpson - exact) << ")\n";
        std::cout << "  Gauss-20   = " << result_gauss
                  << " (error: " << std::fabs(result_gauss - exact) << ")\n";
        std::cout << "  Exact      = " << exact << "\n\n";
    }
}

void testSolvers() {
    std::cout << "=== Root Finding Tests ===\n\n";

    // Test 1: f(x) = x^2 - 2, root at x = sqrt(2)
    {
        auto f = [](double x) { return x * x - 2.0; };
        double exact = std::sqrt(2.0);

        std::cout << "Finding root of x² - 2 = 0 (exact: √2 = " << exact << "):\n\n";

        // Bisection
        {
            BisectionSolver<double> solver;
            solver.setMaxEvaluations(100);
            double root = solver.solve(f, 1e-10, 1.5, 0.0, 3.0);
            std::cout << "  Bisection:  root = " << std::setprecision(15) << root
                      << " (error: " << std::fabs(root - exact) << ")\n";
        }

        // Secant
        {
            SecantSolver<double> solver;
            solver.setMaxEvaluations(100);
            double root = solver.solve(f, 1e-10, 1.5, 0.0, 3.0);
            std::cout << "  Secant:     root = " << std::setprecision(15) << root
                      << " (error: " << std::fabs(root - exact) << ")\n";
        }

        // Newton (with finite differences)
        {
            NewtonSolver<double> solver;
            solver.setMaxEvaluations(100);
            double root = solver.solve(f, 1e-10, 1.5, 0.0, 3.0);
            std::cout << "  Newton:     root = " << std::setprecision(15) << root
                      << " (error: " << std::fabs(root - exact) << ")\n";
        }

        // Brent
        {
            BrentSolver<double> solver;
            solver.setMaxEvaluations(100);
            double root = solver.solve(f, 1e-10, 1.5, 0.0, 3.0);
            std::cout << "  Brent:      root = " << std::setprecision(15) << root
                      << " (error: " << std::fabs(root - exact) << ")\n";
        }

        // Ridder
        {
            RidderSolver<double> solver;
            solver.setMaxEvaluations(100);
            double root = solver.solve(f, 1e-10, 1.5, 0.0, 3.0);
            std::cout << "  Ridder:     root = " << std::setprecision(15) << root
                      << " (error: " << std::fabs(root - exact) << ")\n";
        }

        // False Position
        {
            FalsePositionSolver<double> solver;
            solver.setMaxEvaluations(100);
            double root = solver.solve(f, 1e-10, 1.5, 0.0, 3.0);
            std::cout << "  FalsePos:   root = " << std::setprecision(15) << root
                      << " (error: " << std::fabs(root - exact) << ")\n\n";
        }
    }

    // Test 2: f(x) = exp(x) - 3, root at x = ln(3)
    {
        auto f = [](double x) { return std::exp(x) - 3.0; };
        double exact = std::log(3.0);

        std::cout << "Finding root of exp(x) - 3 = 0 (exact: ln(3) = " << exact << "):\n\n";

        BrentSolver<double> solver;
        double root = solver.solve(f, 1e-10, 1.0, 0.0, 2.0);

        std::cout << "  Brent:      root = " << std::setprecision(15) << root
                  << " (error: " << std::fabs(root - exact) << ")\n";
        std::cout << "  f(root)     = " << f(root) << "\n\n";
    }

    // Test 3: Automatic bracketing
    {
        auto f = [](double x) { return x * x * x - x - 2.0; };  // Root at x ≈ 1.521

        BrentSolver<double> solver;
        double root = solver.solve(f, 1e-10, 1.5, 0.1);  // Auto-bracket from guess with step

        std::cout << "Finding root of x³ - x - 2 = 0 with auto-bracketing:\n";
        std::cout << "  Brent:      root = " << std::setprecision(15) << root << "\n";
        std::cout << "  f(root)     = " << f(root) << "\n\n";
    }

    // Test 4: Newton with explicit derivative
    {
        auto f = [](double x) { return x * x - 2.0; };
        auto df = [](double x) { return 2.0 * x; };
        double exact = std::sqrt(2.0);

        NewtonSolverWithDerivative<double> solver;
        solver.setDerivative(df);
        solver.setMaxEvaluations(100);
        double root = solver.solve(f, 1e-10, 1.5, 0.0, 3.0);

        std::cout << "Newton with explicit derivative for x² - 2 = 0:\n";
        std::cout << "  Root        = " << std::setprecision(15) << root
                  << " (error: " << std::fabs(root - exact) << ")\n\n";
    }
}

void testQuadratureOnStandardDomain() {
    std::cout << "=== Quadrature on Standard Domain [-1, 1] ===\n\n";

    // Test polynomial integration (Gauss quadrature is exact for polynomials)
    // For order n, exact for polynomials up to degree 2n-1

    auto poly2 = [](double x) { return 1.0 + 2.0*x + 3.0*x*x; };  // degree 2
    auto poly4 = [](double x) { return 1.0 + x + x*x + x*x*x + x*x*x*x; };  // degree 4

    // Exact integral of poly2 from -1 to 1: ∫(1 + 2x + 3x²)dx = [x + x² + x³]_{-1}^{1} = 4
    // Exact integral of poly4 from -1 to 1: ∫(1 + x + x² + x³ + x⁴)dx = 16/5

    double exact_poly2 = 4.0;
    double exact_poly4 = 16.0 / 5.0;

    std::cout << "Polynomial degree 2: 1 + 2x + 3x²\n";
    std::cout << "Exact integral [-1, 1]: " << exact_poly2 << "\n";

    for (size_t order : {2, 3, 5}) {
        GaussLegendreQuadrature<double> quad(order);
        double result = quad.integrate(poly2, -1.0, 1.0);
        std::cout << "  Order " << order << ": " << std::setprecision(15) << result
                  << " (error: " << std::fabs(result - exact_poly2) << ")\n";
    }

    std::cout << "\nPolynomial degree 4: 1 + x + x² + x³ + x⁴\n";
    std::cout << "Exact integral [-1, 1]: " << exact_poly4 << "\n";

    for (size_t order : {2, 3, 5}) {
        GaussLegendreQuadrature<double> quad(order);
        double result = quad.integrate(poly4, -1.0, 1.0);
        std::cout << "  Order " << order << ": " << std::setprecision(15) << result
                  << " (error: " << std::fabs(result - exact_poly4) << ")\n";
    }
    std::cout << "\n";
}

int main() {
    std::cout << std::setprecision(15);

    try {
        testIntegration();
        testQuadratureOnStandardDomain();
        testSolvers();

        std::cout << "All tests completed successfully!\n";
        return 0;
    }
    catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }
}
