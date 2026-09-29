#include "quantape/math/Interpolations/BicubicInterpolation.h"
#include "quantape/math/Interpolations/BilinearInterpolation.h"
#include "quantape/math/Interpolations/CubicInterpolation.h"
#include "quantape/math/Interpolations/LogLinearInterpolation.h"

#include <cassert>
#include <cmath>
#include <iostream>
#include <vector>

#include "quantape/log/Log.h"
#include "quantape/util/Check.h"

void testLogLinearInterpolation() {
    QTA_LOG_INFO("test", "=== Log-Linear Interpolation Tests ===");

    std::vector<double> x = {0.0, 1.0, 2.0, 3.0, 4.0};
    std::vector<double> y = {1.0, 2.718, 7.389, 20.086, 54.598}; // approx exp(x)

    quantape::math::LogLinearInterpolation<double> interp(x, y);

    double val1 = interp(1.5);
    double expected1 = exp(1.5);
    QTA_LOG_INFO("test", "Value at 1.5: {} (expected: {})", quantape::util::num(val1),
                 quantape::util::num(expected1));
    assert(std::abs(val1 - expected1) < 1e-3);

    QTA_LOG_INFO("test", "Log-Linear interpolation tests passed!");
}

void testCubicSplineInterpolation() {
    QTA_LOG_INFO("test", "=== Cubic Spline Interpolation Tests ===");

    // Test case 1: Simple 3-point natural cubic spline
    std::vector<double> x_simple = {0.0, 1.0, 2.0};
    std::vector<double> y_simple = {0.0, 1.0, 0.0};

    quantape::math::CubicInterpolation<double> interp_simple(
        x_simple, y_simple, quantape::math::CubicDerivativeApprox::Spline);
    // For natural cubic spline through (0,0), (1,1), (2,0):
    // First derivatives: d0=1.5, d1=0, d2=-1.5
    // Coefficients: a[0]=1.5, b[0]=0, c[0]=-0.5
    // At x=0.5: P(0.5) = 0 + 1.5*0.5 + 0*0.25 + (-0.5)*0.125 = 0.6875

    double val_simple = interp_simple(0.5);
    double expected_val_simple = 0.6875; // Correct value for natural cubic spline
    QTA_LOG_INFO("test", "Value at 0.5 (Simple 3-point): {} (expected: {})",
                 quantape::util::num(val_simple), quantape::util::num(expected_val_simple));
    assert(std::abs(val_simple - expected_val_simple) < 1e-9);

    // Test case 2: Polynomial function (x^2)
    std::vector<double> x_poly = {0.0, 1.0, 2.0, 3.0};
    std::vector<double> y_poly = {0.0, 1.0, 4.0, 9.0}; // x^2

    quantape::math::CubicInterpolation<double> interp_poly(
        x_poly, y_poly, quantape::math::CubicDerivativeApprox::Spline);
    double val_poly = interp_poly(1.5);
    double expected_poly = 2.25; // 1.5^2
    QTA_LOG_INFO("test", "Value at 1.5 (Polynomial x^2): {} (expected: {})",
                 quantape::util::num(val_poly), quantape::util::num(expected_poly));
    // Natural spline won't match quadratic exactly due to boundary conditions
    // (natural BC has f''=0 at ends, but x^2 has f''=2 everywhere)
    assert(std::abs(val_poly - expected_poly) < 0.1);

    // Test case 3: Smooth function (sine wave) - relaxed tolerance
    std::vector<double> x_sin = {0.0, 1.0, 2.0, 3.0, 4.0};
    std::vector<double> y_sin = {0.0, 0.84147098, 0.90929743, 0.14112001, -0.7568025}; // sin(x)

    quantape::math::CubicInterpolation<double> interp_natural_sin(
        x_sin, y_sin, quantape::math::CubicDerivativeApprox::Spline);
    double val_natural_sin = interp_natural_sin(1.5);
    double expected_sin = sin(1.5);
    QTA_LOG_INFO("test", "Value at 1.5 (Natural, sin(x)): {} (expected: {})",
                 quantape::util::num(val_natural_sin), quantape::util::num(expected_sin));
    // Relaxed tolerance for sparse knots on transcendental function
    assert(std::abs(val_natural_sin - expected_sin) < 1e-3);

    QTA_LOG_INFO("test", "Cubic Spline interpolation tests passed!");
}

void testBilinearInterpolation() {
    QTA_LOG_INFO("test", "=== Bilinear Interpolation Tests ===");
    std::vector<double> x = {0, 1};
    std::vector<double> y = {0, 1};
    std::vector<std::vector<double>> z = {{0, 1}, {1, 2}};
    quantape::math::BilinearInterpolation<double> interp(x, y, z);
    double val = interp(0.5, 0.5);
    QTA_LOG_INFO("test", "Value at (0.5, 0.5): {} (expected: 1.0)",
                 quantape::util::num(val));
    assert(std::abs(val - 1.0) < 1e-9);
    QTA_LOG_INFO("test", "Bilinear interpolation tests passed!");
}

void testBicubicInterpolation() {
    QTA_LOG_INFO("test", "=== Bicubic Interpolation Tests ===");
    std::vector<double> x = {0, 1, 2};
    std::vector<double> y = {0, 1, 2};
    auto f = [](double x, double y) { return x * y + x + y; };
    std::vector<std::vector<double>> z(3, std::vector<double>(3));
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            z[i][j] = f(x[j], y[i]);
        }
    }

    quantape::math::BicubicInterpolation<double> interp(x, y, z);
    double val = interp(0.5, 0.5);
    double expected = f(0.5, 0.5);
    QTA_LOG_INFO("test", "Value at (0.5, 0.5): {} (expected: {})", quantape::util::num(val),
                 quantape::util::num(expected));
    assert(std::abs(val - expected) < 1e-9);
    QTA_LOG_INFO("test", "Bicubic interpolation tests passed!");
}

int main() {
    try {
        testLogLinearInterpolation();
        testCubicSplineInterpolation();
        testBilinearInterpolation();
        testBicubicInterpolation();
        QTA_LOG_INFO("test", "All interpolation tests completed successfully!");
        return 0;
    } catch (const std::exception& e) {
        QTA_LOG_ERROR("test", "Error: {}", e.what());
        return 1;
    }
}
