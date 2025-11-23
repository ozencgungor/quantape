/**
 * @file test_ad.cpp
 * @brief Test automatic differentiation with Stan Math
 *
 * This test demonstrates full AD capabilities of the interpolation and integration library
 * using Stan Math's reverse-mode automatic differentiation.
 */

#include "Math/Interpolations.h"
#include "Math/NumericalMethods.h"

#include <stan/math.hpp>

#include <algorithm> // For std::copy
#include <cmath>
#include <iomanip>
#include <iostream>
#include <iterator> // For std::back_inserter
#include <vector>

using ADVariableT = stan::math::var;
using namespace Math;

void testLinearInterpolationAD() {
    std::cout << "=== Linear Interpolation with AD ===\n\n";

    // Create interpolation data with AD types
    std::vector<ADVariableT> x = {0.0, 1.0, 2.0, 3.0};
    std::vector<ADVariableT> y = {0.0, 1.0, 4.0, 9.0};

    LinearInterpolation<ADVariableT> interp(x, y);

    // Test 1: Interpolate at x=1.5
    ADVariableT x_eval = 1.5;
    ADVariableT y_interp = interp(x_eval);

    std::cout << "Interpolated value at x=1.5: " << y_interp.val() << "\n";
    std::cout << "Expected (linear): 2.5\n\n";

    // Test 2: Compute derivative with respect to x_eval
    stan::math::grad(y_interp.vi_);
    double dy_dx = x_eval.adj();
    std::cout << "Derivative d/dx (via AD): " << dy_dx << "\n";
    std::cout << "Expected slope: 3.0 (slope between (1,1) and (2,4))\n\n";

    // Test 3: Sensitivity to knot values
    stan::math::recover_memory();
    std::vector<ADVariableT> x2 = {0.0, 1.0, 2.0, 3.0};
    std::vector<ADVariableT> y2 = {0.0, 1.0, 4.0, 9.0};
    LinearInterpolation<ADVariableT> interp2(x2, y2);

    ADVariableT result = interp2(1.5);
    stan::math::grad(result.vi_);

    std::cout << "Sensitivity to knot values:\n";
    std::cout << "  ∂result/∂y[1] = " << y2[1].adj() << " (expected: 0.5)\n";
    std::cout << "  ∂result/∂y[2] = " << y2[2].adj() << " (expected: 0.5)\n\n";
}

void testCubicSplineInterpolationAD() {
    std::cout << "=== Cubic Spline Interpolation with AD ===\n\n";

    // Create data for x^2 function
    std::vector<ADVariableT> x = {0.0, 1.0, 2.0, 3.0};
    std::vector<ADVariableT> y = {0.0, 1.0, 4.0, 9.0};

    CubicInterpolation<ADVariableT> spline(x, y, CubicInterpolation<ADVariableT>::Spline);

    // Test interpolation value
    ADVariableT x_eval = 1.5;
    ADVariableT y_interp = spline(x_eval);

    std::cout << "Interpolated value at x=1.5: " << y_interp.val() << "\n";
    std::cout << "Expected (x^2): 2.25\n";
    std::cout << "Spline approximation error: " << std::abs(y_interp.val() - 2.25) << "\n\n";

    // Compute derivative
    stan::math::grad(y_interp.vi_);
    double dy_dx = x_eval.adj();
    std::cout << "Derivative d/dx at x=1.5 (via AD): " << dy_dx << "\n";
    std::cout << "Expected (2*1.5): 3.0\n";
    std::cout << "Derivative error: " << std::abs(dy_dx - 3.0) << "\n\n";

    stan::math::recover_memory();
}

void testAllCubicMethodsAD() {
    std::cout << "=== All Cubic Interpolation Methods with AD ===\n\n";

    // Test smooth function: sin(x) at several points
    std::vector<ADVariableT> x;
    std::vector<ADVariableT> y;
    for (int i = 0; i <= 10; ++i) {
        double xi = i * M_PI / 10.0;
        x.push_back(ADVariableT(xi));
        y.push_back(ADVariableT(std::sin(xi)));
    }

    ADVariableT x_eval = M_PI / 4.0;
    double expected_value = std::sin(M_PI / 4.0);
    double expected_deriv = std::cos(M_PI / 4.0);

    std::vector<std::pair<std::string, typename CubicInterpolation<ADVariableT>::DerivativeApprox>>
        methods = {{"Spline", CubicInterpolation<ADVariableT>::Spline},
                   {"Parabolic", CubicInterpolation<ADVariableT>::Parabolic},
                   {"Akima", CubicInterpolation<ADVariableT>::Akima},
                   {"Kruger", CubicInterpolation<ADVariableT>::Kruger},
                   {"Harmonic", CubicInterpolation<ADVariableT>::Harmonic}};

    std::cout << "Testing sin(x) at x = π/4\n";
    std::cout << "Expected value: " << expected_value << "\n";
    std::cout << "Expected derivative: " << expected_deriv << "\n\n";

    for (const auto& [name, method] : methods) {
        stan::math::recover_memory();

        // Recreate vectors with fresh AD variables for each test
        std::vector<ADVariableT> x_copy;
        std::vector<ADVariableT> y_copy;
        for (int i = 0; i <= 10; ++i) {
            double xi = i * M_PI / 10.0;
            x_copy.push_back(ADVariableT(xi));
            y_copy.push_back(ADVariableT(std::sin(xi)));
        }

        CubicInterpolation<ADVariableT> interp(x_copy, y_copy, method);
        // Recreate x_test after recover_memory to avoid using invalid AD variable
        ADVariableT x_test = M_PI / 4.0;
        ADVariableT result = interp(x_test);

        stan::math::grad(result.vi_);

        std::cout << std::setw(12) << name << ": ";
        std::cout << "value = " << std::setw(10) << result.val();
        std::cout << " (err: " << std::setw(10) << std::abs(result.val() - expected_value) << "), ";
        std::cout << "deriv = " << std::setw(10) << x_test.adj();
        std::cout << " (err: " << std::setw(10) << std::abs(x_test.adj() - expected_deriv) << ")\n";
    }
    std::cout << "\n";
    stan::math::recover_memory();
}

void testBilinearInterpolationAD() {
    std::cout << "=== Bilinear Interpolation with AD ===\n\n";

    // Create 2D grid: f(x,y) = x*y
    std::vector<ADVariableT> x = {0.0, 1.0, 2.0};
    std::vector<ADVariableT> y = {0.0, 1.0, 2.0};
    std::vector<std::vector<ADVariableT>> z(3, std::vector<ADVariableT>(3));

    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            z[i][j] = x[j] * y[i];
        }
    }

    BilinearInterpolation<ADVariableT> interp(x, y, z);

    // Evaluate at (1.5, 1.5)
    ADVariableT x_eval = 1.5;
    ADVariableT y_eval = 1.5;
    ADVariableT result = interp(x_eval, y_eval);

    std::cout << "f(1.5, 1.5) = " << result.val() << "\n";
    std::cout << "Expected: 2.25 (1.5 * 1.5)\n\n";

    // Compute partial derivatives
    stan::math::grad(result.vi_);
    std::cout << "∂f/∂x at (1.5, 1.5) = " << x_eval.adj() << " (expected: 1.5)\n";
    std::cout << "∂f/∂y at (1.5, 1.5) = " << y_eval.adj() << " (expected: 1.5)\n\n";
    stan::math::recover_memory();
}

void testIntegrationAD() {
    std::cout << "=== Integration with AD ===\n\n";

    // Test 1: Integrate x^2 from 0 to θ, derivative should be θ^2
    std::cout << "Test: I(θ) = ∫₀^θ x² dx\n";
    std::cout << "Expected: I(θ) = θ³/3, dI/dθ = θ²\n\n";

    ADVariableT theta = 2.0;

    auto f = [](ADVariableT x) { return x * x; };

    TrapezoidIntegratorDefault<ADVariableT> integrator(1e-6, 1000);
    ADVariableT integral = integrator(f, ADVariableT(0.0), theta);

    double expected_integral = std::pow(2.0, 3) / 3.0;
    double expected_derivative = 2.0 * 2.0; // θ^2 at θ=2

    std::cout << "Integral value: " << integral.val() << " (expected: " << expected_integral
              << ")\n";

    stan::math::grad(integral.vi_);
    std::cout << "Derivative dI/dθ: " << theta.adj() << " (expected: " << expected_derivative
              << ")\n";
    std::cout << "Error in derivative: " << std::abs(theta.adj() - expected_derivative) << "\n\n";

    // Test 2: Parametric integral - ∫₀¹ θ*x² dx = θ/3
    std::cout << "Test: I(θ) = ∫₀¹ θ*x² dx\n";
    std::cout << "Expected: I(θ) = θ/3, dI/dθ = 1/3\n\n";

    stan::math::recover_memory();
    ADVariableT param = 3.0;

    auto f_param = [param](ADVariableT x) { return param * x * x; };

    TrapezoidIntegratorDefault<ADVariableT> integrator2(1e-6, 1000);
    ADVariableT integral2 = integrator2(f_param, ADVariableT(0.0), ADVariableT(1.0));

    double expected_integral2 = 3.0 / 3.0;
    double expected_deriv2 = 1.0 / 3.0;

    std::cout << "Integral value: " << integral2.val() << " (expected: " << expected_integral2
              << ")\n";

    stan::math::grad(integral2.vi_);
    std::cout << "Derivative dI/dθ: " << param.adj() << " (expected: " << expected_deriv2 << ")\n";
    std::cout << "Error in derivative: " << std::abs(param.adj() - expected_deriv2) << "\n\n";
    stan::math::recover_memory();
}

void testIntegrateInterpolatedFunctionAD() {
    std::cout << "=== Integrate Interpolated Function with AD ===\n\n";

    // Create interpolation for exp(x) using AD types
    std::vector<ADVariableT> x_data;
    std::vector<ADVariableT> y_data;

    for (int i = 0; i <= 10; ++i) {
        double xi = i * 0.2;
        x_data.push_back(ADVariableT(xi));
        y_data.push_back(ADVariableT(std::exp(xi)));
    }

    CubicInterpolation<ADVariableT> spline(x_data, y_data, CubicInterpolation<ADVariableT>::Spline);

    // Integrate the interpolated function from 0 to upper_limit
    ADVariableT upper_limit = 1.0;

    auto interpolated_func = [&spline](ADVariableT x) {
        return spline(x, true); // Allow extrapolation
    };

    SimpsonIntegrator<ADVariableT> integrator(1e-6, 1000);
    ADVariableT integral = integrator(interpolated_func, ADVariableT(0.0), upper_limit);

    // For exp(x), ∫₀^b exp(x)dx = exp(b) - 1
    double expected_value = std::exp(1.0) - 1.0;
    double expected_deriv = std::exp(1.0); // d/db[exp(b)-1] = exp(b)

    std::cout << "Integrating interpolated exp(x) from 0 to 1\n";
    std::cout << "Integral value: " << integral.val() << " (expected: " << expected_value << ")\n";
    std::cout << "Error: " << std::abs(integral.val() - expected_value) << "\n\n";

    stan::math::grad(integral.vi_);
    std::cout << "Derivative d/d(upper_limit): " << upper_limit.adj()
              << " (expected: " << expected_deriv << ")\n";
    std::cout << "Error in derivative: " << std::abs(upper_limit.adj() - expected_deriv) << "\n\n";

    // Test sensitivity to knot values
    std::cout << "Sensitivity to interpolation knot values:\n";
    std::cout << "  ∂Integral/∂y_data[5] (at x=1.0): " << y_data[5].adj() << "\n";
    std::cout << "  (This shows how integral changes with knot value adjustments)\n\n";
    stan::math::recover_memory();
}

void testFinancialSensitivityExample() {
    std::cout << "=== Financial Example: Price Sensitivity ===\n\n";
    std::cout << "Scenario: Discount curve interpolation and present value calculation\n\n";

    // Discount factors at different maturities (time in years)
    std::vector<ADVariableT> maturities = {0.0, 0.5, 1.0, 2.0, 3.0, 5.0};
    std::vector<ADVariableT> discount_factors = {1.0, 0.98, 0.96, 0.92, 0.88, 0.80};

    // Create interpolation for discount curve
    CubicInterpolation<ADVariableT> discount_curve(maturities, discount_factors,
                                                   CubicInterpolation<ADVariableT>::Spline);

    // Cash flows: 100 at t=1.5 years
    ADVariableT cash_flow_time = 1.5;
    ADVariableT cash_flow_amount = 100.0;

    ADVariableT discount_factor = discount_curve(cash_flow_time);
    ADVariableT present_value = cash_flow_amount * discount_factor;

    std::cout << "Cash flow: $100 at t=1.5 years\n";
    std::cout << "Discount factor at t=1.5: " << discount_factor.val() << "\n";
    std::cout << "Present value: $" << present_value.val() << "\n\n";

    // Compute sensitivities (Greeks)
    stan::math::grad(present_value.vi_);

    std::cout << "Price sensitivities to discount curve knots:\n";
    for (size_t i = 0; i < discount_factors.size(); ++i) {
        std::cout << "  ∂PV/∂DF[t=" << maturities[i].val() << "] = " << discount_factors[i].adj()
                  << "\n";
    }
    std::cout << "\n(These are the risk sensitivities - how PV changes with curve movements)\n\n";
    stan::math::recover_memory();
}

int main() {
    std::cout << std::setprecision(8);
    std::cout << std::fixed;

    try {
        testLinearInterpolationAD();
        testCubicSplineInterpolationAD();
        testAllCubicMethodsAD();
        testBilinearInterpolationAD();
        testIntegrationAD();
        testIntegrateInterpolatedFunctionAD();
        testFinancialSensitivityExample();

        std::cout << "========================================\n";
        std::cout << "All AD tests completed successfully!\n";
        std::cout << "========================================\n";

        return 0;
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }
}
