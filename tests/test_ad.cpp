/**
 * @file test_ad.cpp
 * @brief Test automatic differentiation with Stan Math
 *
 * This test demonstrates full AD capabilities of the interpolation and integration library
 * using Stan Math's reverse-mode automatic differentiation.
 */

#include "quantape/math/StanMath.h"

#include "quantape/log/Log.h"
#include "quantape/math/Integrals/IntegratorStanPrimitives.h"
#include "quantape/math/Interpolations.h"
#include "quantape/math/Interpolations/InterpolationStanPrimitives.h"
#include "quantape/math/NumericalMethods.h"
#include "quantape/util/Check.h"

#include <algorithm> // For std::copy
#include <cmath>
#include <iomanip>
#include <iostream>
#include <iterator> // For std::back_inserter
#include <vector>

using ADVariableT = stan::math::var;
using namespace quantape::math;

void testLinearInterpolationAD() {
    QTA_LOG_INFO("test", "=== Linear Interpolation with AD ===");

    // Create interpolation data with AD types
    std::vector<ADVariableT> x = {0.0, 1.0, 2.0, 3.0};
    std::vector<ADVariableT> y = {0.0, 1.0, 4.0, 9.0};

    LinearInterpolation<ADVariableT> interp(x, y);

    // Test 1: Interpolate at x=1.5
    ADVariableT x_eval = 1.5;
    ADVariableT y_interp = interp(x_eval);

    QTA_LOG_INFO("test", "Interpolated value at x=1.5: {}", quantape::util::num(y_interp.val()));
    QTA_LOG_INFO("test", "Expected (linear): 2.5");

    // Test 2: Compute derivative with respect to x_eval
    stan::math::grad(y_interp.vi_);
    double dy_dx = x_eval.adj();
    QTA_LOG_INFO("test", "Derivative d/dx (via AD): {}", quantape::util::num(dy_dx));
    QTA_LOG_INFO("test", "Expected slope: 3.0 (slope between (1,1) and (2,4))");

    // Test 3: Sensitivity to knot values
    stan::math::recover_memory();
    std::vector<ADVariableT> x2 = {0.0, 1.0, 2.0, 3.0};
    std::vector<ADVariableT> y2 = {0.0, 1.0, 4.0, 9.0};
    LinearInterpolation<ADVariableT> interp2(x2, y2);

    ADVariableT result = interp2(1.5);
    stan::math::grad(result.vi_);

    QTA_LOG_INFO("test", "Sensitivity to knot values:");
    QTA_LOG_INFO("test", "  ∂result/∂y[1] = {} (expected: 0.5)", quantape::util::num(y2[1].adj()));
    QTA_LOG_INFO("test", "  ∂result/∂y[2] = {} (expected: 0.5)", quantape::util::num(y2[2].adj()));
}

void testCubicSplineInterpolationAD() {
    QTA_LOG_INFO("test", "=== Cubic Spline Interpolation with AD ===");

    // Create data for x^2 function
    std::vector<ADVariableT> x = {0.0, 1.0, 2.0, 3.0};
    std::vector<ADVariableT> y = {0.0, 1.0, 4.0, 9.0};

    CubicInterpolation<ADVariableT> spline(x, y, CubicDerivativeApprox::Spline);

    // Test interpolation value
    ADVariableT x_eval = 1.5;
    ADVariableT y_interp = spline(x_eval);

    QTA_LOG_INFO("test", "Interpolated value at x=1.5: {}", quantape::util::num(y_interp.val()));
    QTA_LOG_INFO("test", "Expected (x^2): 2.25");
    QTA_LOG_INFO("test", "Spline approximation error: {}",
                 quantape::util::num(std::abs(y_interp.val() - 2.25)));

    // Compute derivative
    stan::math::grad(y_interp.vi_);
    double dy_dx = x_eval.adj();
    QTA_LOG_INFO("test", "Derivative d/dx at x=1.5 (via AD): {}", quantape::util::num(dy_dx));
    QTA_LOG_INFO("test", "Expected (2*1.5): 3.0");
    QTA_LOG_INFO("test", "Derivative error: {}", quantape::util::num(std::abs(dy_dx - 3.0)));

    stan::math::recover_memory();
}

void testAllCubicMethodsAD() {
    QTA_LOG_INFO("test", "=== All Cubic Interpolation Methods with AD ===");

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
        methods = {{"Spline", CubicDerivativeApprox::Spline},
                   {"Parabolic", CubicDerivativeApprox::Parabolic},
                   {"Akima", CubicDerivativeApprox::Akima},
                   {"Kruger", CubicDerivativeApprox::Kruger},
                   {"Harmonic", CubicDerivativeApprox::Harmonic}};

    QTA_LOG_INFO("test", "Testing sin(x) at x = π/4");
    QTA_LOG_INFO("test", "Expected value: {}", quantape::util::num(expected_value));
    QTA_LOG_INFO("test", "Expected derivative: {}", quantape::util::num(expected_deriv));

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

        QTA_LOG_INFO("test", "{}: value = {} (err: {}), deriv = {} (err: {})", name,
                     quantape::util::num(result.val()),
                     quantape::util::num(std::abs(result.val() - expected_value)),
                     quantape::util::num(x_test.adj()),
                     quantape::util::num(std::abs(x_test.adj() - expected_deriv)));
    }
    stan::math::recover_memory();
}

void testBilinearInterpolationAD() {
    QTA_LOG_INFO("test", "=== Bilinear Interpolation with AD ===");

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

    QTA_LOG_INFO("test", "f(1.5, 1.5) = {}", quantape::util::num(result.val()));
    QTA_LOG_INFO("test", "Expected: 2.25 (1.5 * 1.5)");

    // Compute partial derivatives
    stan::math::grad(result.vi_);
    QTA_LOG_INFO("test", "∂f/∂x at (1.5, 1.5) = {} (expected: 1.5)",
                 quantape::util::num(x_eval.adj()));
    QTA_LOG_INFO("test", "∂f/∂y at (1.5, 1.5) = {} (expected: 1.5)",
                 quantape::util::num(y_eval.adj()));
    stan::math::recover_memory();
}

void testIntegrationAD() {
    QTA_LOG_INFO("test", "=== Integration with AD ===");

    // Test 1: Integrate x^2 from 0 to θ, derivative should be θ^2
    QTA_LOG_INFO("test", "Test: I(θ) = ∫₀^θ x² dx");
    QTA_LOG_INFO("test", "Expected: I(θ) = θ³/3, dI/dθ = θ²");

    ADVariableT theta = 2.0;

    auto f = [](ADVariableT x) { return x * x; };

    TrapezoidIntegratorDefault<ADVariableT> integrator(1e-6, 1000);
    ADVariableT integral = integrator(f, ADVariableT(0.0), theta);

    double expected_integral = std::pow(2.0, 3) / 3.0;
    double expected_derivative = 2.0 * 2.0; // θ^2 at θ=2

    QTA_LOG_INFO("test", "Integral value: {} (expected: {})", quantape::util::num(integral.val()),
                 quantape::util::num(expected_integral));

    stan::math::grad(integral.vi_);
    QTA_LOG_INFO("test", "Derivative dI/dθ: {} (expected: {})", quantape::util::num(theta.adj()),
                 quantape::util::num(expected_derivative));
    QTA_LOG_INFO("test", "Error in derivative: {}",
                 quantape::util::num(std::abs(theta.adj() - expected_derivative)));

    // Test 2: Parametric integral - ∫₀¹ θ*x² dx = θ/3
    QTA_LOG_INFO("test", "Test: I(θ) = ∫₀¹ θ*x² dx");
    QTA_LOG_INFO("test", "Expected: I(θ) = θ/3, dI/dθ = 1/3");

    stan::math::recover_memory();
    ADVariableT param = 3.0;

    auto f_param = [param](ADVariableT x) { return param * x * x; };

    TrapezoidIntegratorDefault<ADVariableT> integrator2(1e-6, 1000);
    ADVariableT integral2 = integrator2(f_param, ADVariableT(0.0), ADVariableT(1.0));

    double expected_integral2 = 3.0 / 3.0;
    double expected_deriv2 = 1.0 / 3.0;

    QTA_LOG_INFO("test", "Integral value: {} (expected: {})", quantape::util::num(integral2.val()),
                 quantape::util::num(expected_integral2));

    stan::math::grad(integral2.vi_);
    QTA_LOG_INFO("test", "Derivative dI/dθ: {} (expected: {})", quantape::util::num(param.adj()),
                 quantape::util::num(expected_deriv2));
    QTA_LOG_INFO("test", "Error in derivative: {}",
                 quantape::util::num(std::abs(param.adj() - expected_deriv2)));
    stan::math::recover_memory();
}

void testIntegrateInterpolatedFunctionAD() {
    QTA_LOG_INFO("test", "=== Integrate Interpolated Function with AD ===");

    // Create interpolation for exp(x) using AD types
    std::vector<ADVariableT> x_data;
    std::vector<ADVariableT> y_data;

    for (int i = 0; i <= 10; ++i) {
        double xi = i * 0.2;
        x_data.push_back(ADVariableT(xi));
        y_data.push_back(ADVariableT(std::exp(xi)));
    }

    CubicInterpolation<ADVariableT> spline(x_data, y_data, CubicDerivativeApprox::Spline);

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

    QTA_LOG_INFO("test", "Integrating interpolated exp(x) from 0 to 1");
    QTA_LOG_INFO("test", "Integral value: {} (expected: {})", quantape::util::num(integral.val()),
                 quantape::util::num(expected_value));
    QTA_LOG_INFO("test", "Error: {}",
                 quantape::util::num(std::abs(integral.val() - expected_value)));

    stan::math::grad(integral.vi_);
    QTA_LOG_INFO("test", "Derivative d/d(upper_limit): {} (expected: {})",
                 quantape::util::num(upper_limit.adj()), quantape::util::num(expected_deriv));
    QTA_LOG_INFO("test", "Error in derivative: {}",
                 quantape::util::num(std::abs(upper_limit.adj() - expected_deriv)));

    // Test sensitivity to knot values
    QTA_LOG_INFO("test", "Sensitivity to interpolation knot values:");
    QTA_LOG_INFO("test", "  ∂Integral/∂y_data[5] (at x=1.0): {}",
                 quantape::util::num(y_data[5].adj()));
    QTA_LOG_INFO("test", "  (This shows how integral changes with knot value adjustments)");
    stan::math::recover_memory();
}

void testFinancialSensitivityExample() {
    QTA_LOG_INFO("test", "=== Financial Example: Price Sensitivity ===");
    QTA_LOG_INFO("test", "Scenario: Discount curve interpolation and present value calculation");

    // Discount factors at different maturities (time in years)
    std::vector<ADVariableT> maturities = {0.0, 0.5, 1.0, 2.0, 3.0, 5.0};
    std::vector<ADVariableT> discount_factors = {1.0, 0.98, 0.96, 0.92, 0.88, 0.80};

    // Create interpolation for discount curve
    CubicInterpolation<ADVariableT> discount_curve(maturities, discount_factors,
                                                   CubicDerivativeApprox::Spline);

    // Cash flows: 100 at t=1.5 years
    ADVariableT cash_flow_time = 1.5;
    ADVariableT cash_flow_amount = 100.0;

    ADVariableT discount_factor = discount_curve(cash_flow_time);
    ADVariableT present_value = cash_flow_amount * discount_factor;

    QTA_LOG_INFO("test", "Cash flow: $100 at t=1.5 years");
    QTA_LOG_INFO("test", "Discount factor at t=1.5: {}",
                 quantape::util::num(discount_factor.val()));
    QTA_LOG_INFO("test", "Present value: ${}", quantape::util::num(present_value.val()));

    // Compute sensitivities (Greeks)
    stan::math::grad(present_value.vi_);

    QTA_LOG_INFO("test", "Price sensitivities to discount curve knots:");
    for (size_t i = 0; i < discount_factors.size(); ++i) {
        QTA_LOG_INFO("test", "  ∂PV/∂DF[t={}] = {}", quantape::util::num(maturities[i].val()),
                     quantape::util::num(discount_factors[i].adj()));
    }
    QTA_LOG_INFO("test",
                 "(These are the risk sensitivities - how PV changes with curve movements)");
    stan::math::recover_memory();
}

int main() {
    try {
        testLinearInterpolationAD();
        testCubicSplineInterpolationAD();
        testAllCubicMethodsAD();
        testBilinearInterpolationAD();
        testIntegrationAD();
        testIntegrateInterpolatedFunctionAD();
        testFinancialSensitivityExample();

        QTA_LOG_INFO("test", "========================================");
        QTA_LOG_INFO("test", "All AD tests completed successfully!");
        QTA_LOG_INFO("test", "========================================");

        return 0;
    } catch (const std::exception& e) {
        QTA_LOG_ERROR("test", "Error: {}", e.what());
        return 1;
    }
}
