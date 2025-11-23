#include "Math/Interpolations/LogLinearInterpolation.h"
#include "Math/Interpolations/CubicSplineInterpolation.h"
#include "Math/Interpolations/BilinearInterpolation.h"
#include "Math/Interpolations/BicubicInterpolation.h"
#include <iostream>
#include <vector>
#include <cassert>
#include <cmath>

void testLogLinearInterpolation() {
    std::cout << "=== Log-Linear Interpolation Tests ===\n\n";

    std::vector<double> x = {0.0, 1.0, 2.0, 3.0, 4.0};
    std::vector<double> y = {1.0, 2.718, 7.389, 20.086, 54.598}; // approx exp(x)

    Math::LogLinearInterpolation<double> interp(x, y);

    double val1 = interp(1.5);
    double expected1 = exp(1.5);
    std::cout << "Value at 1.5: " << val1 << " (expected: " << expected1 << ")" << std::endl;
    assert(std::abs(val1 - expected1) < 1e-3);
    
    std::cout << "\nLog-Linear interpolation tests passed!\n\n";
}

void testCubicSplineInterpolation() {
    std::cout << "=== Cubic Spline Interpolation Tests ===\n\n";

    // Test case 1: Simple 3-point natural cubic spline
    std::vector<double> x_simple = {0.0, 1.0, 2.0};
    std::vector<double> y_simple = {0.0, 1.0, 0.0};

    Math::CubicSplineInterpolation<double> interp_simple(x_simple, y_simple, Math::CubicSplineInterpolation<double>::Natural);
    // For natural cubic spline through (0,0), (1,1), (2,0):
    // First derivatives: d0=1.5, d1=0, d2=-1.5
    // Coefficients: a[0]=1.5, b[0]=0, c[0]=-0.5
    // At x=0.5: P(0.5) = 0 + 1.5*0.5 + 0*0.25 + (-0.5)*0.125 = 0.6875

    double val_simple = interp_simple(0.5);
    double expected_val_simple = 0.6875; // Correct value for natural cubic spline
    std::cout << "Value at 0.5 (Simple 3-point): " << val_simple << " (expected: " << expected_val_simple << ")" << std::endl;
    assert(std::abs(val_simple - expected_val_simple) < 1e-9);

    // Test case 2: Polynomial function (x^2)
    std::vector<double> x_poly = {0.0, 1.0, 2.0, 3.0};
    std::vector<double> y_poly = {0.0, 1.0, 4.0, 9.0}; // x^2

    Math::CubicSplineInterpolation<double> interp_poly(x_poly, y_poly, Math::CubicSplineInterpolation<double>::Natural);
    double val_poly = interp_poly(1.5);
    double expected_poly = 2.25; // 1.5^2
    std::cout << "Value at 1.5 (Polynomial x^2): " << val_poly << " (expected: " << expected_poly << ")" << std::endl;
    // Natural spline won't match quadratic exactly due to boundary conditions
    // (natural BC has f''=0 at ends, but x^2 has f''=2 everywhere)
    assert(std::abs(val_poly - expected_poly) < 0.1);

    // Test case 3: Smooth function (sine wave) - relaxed tolerance
    std::vector<double> x_sin = {0.0, 1.0, 2.0, 3.0, 4.0};
    std::vector<double> y_sin = {0.0, 0.84147098, 0.90929743, 0.14112001, -0.7568025}; // sin(x)

    Math::CubicSplineInterpolation<double> interp_natural_sin(x_sin, y_sin, Math::CubicSplineInterpolation<double>::Natural);
    double val_natural_sin = interp_natural_sin(1.5);
    double expected_sin = sin(1.5);
    std::cout << "Value at 1.5 (Natural, sin(x)): " << val_natural_sin << " (expected: " << expected_sin << ")" << std::endl;
    // Relaxed tolerance for sparse knots on transcendental function
    assert(std::abs(val_natural_sin - expected_sin) < 1e-3);


    std::cout << "\nCubic Spline interpolation tests passed!\n\n";
}

void testBilinearInterpolation() {
    std::cout << "=== Bilinear Interpolation Tests ===\n\n";
    std::vector<double> x = {0, 1};
    std::vector<double> y = {0, 1};
    std::vector<std::vector<double>> z = {{0, 1}, {1, 2}};
    Math::BilinearInterpolation<double> interp(x, y, z);
    double val = interp(0.5, 0.5);
    std::cout << "Value at (0.5, 0.5): " << val << " (expected: 1.0)" << std::endl;
    assert(std::abs(val - 1.0) < 1e-9);
    std::cout << "\nBilinear interpolation tests passed!\n\n";
}

void testBicubicInterpolation() {
    std::cout << "=== Bicubic Interpolation Tests ===\n\n";
    std::vector<double> x = {0, 1, 2};
    std::vector<double> y = {0, 1, 2};
    auto f = [](double x, double y) { return x * y + x + y; };
    std::vector<std::vector<double>> z(3, std::vector<double>(3));
    for(int i=0; i<3; ++i) {
        for(int j=0; j<3; ++j) {
            z[i][j] = f(x[j], y[i]);
        }
    }

    Math::BicubicInterpolation<double> interp(x, y, z);
    double val = interp(0.5, 0.5);
    double expected = f(0.5, 0.5);
    std::cout << "Value at (0.5, 0.5): " << val << " (expected: " << expected << ")" << std::endl;
    assert(std::abs(val - expected) < 1e-9);
    std::cout << "\nBicubic interpolation tests passed!\n\n";
}

int main() {
    try {
        testLogLinearInterpolation();
        testCubicSplineInterpolation();
        testBilinearInterpolation();
        testBicubicInterpolation();
        std::cout << "All interpolation tests completed successfully!\n";
        return 0;
    }
    catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }
}