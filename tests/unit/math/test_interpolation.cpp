// test_interpolation.cpp — log-linear, cubic, bilinear and bicubic gates.
//
// The legacy TU used C assert(), which compiles out under NDEBUG; the gates
// below are live in every configuration.

#include "quantape/math/Interpolations/BicubicInterpolation.h"
#include "quantape/math/Interpolations/BilinearInterpolation.h"
#include "quantape/math/Interpolations/CubicInterpolation.h"
#include "quantape/math/Interpolations/LogLinearInterpolation.h"

#include <cmath>
#include <vector>

#include "support/GtestSupport.h"

TEST(Interpolation, logLinearValue) {
    std::vector<double> x = {0.0, 1.0, 2.0, 3.0, 4.0};
    std::vector<double> y = {1.0, 2.718, 7.389, 20.086, 54.598}; // approx exp(x)

    const quantape::math::LogLinearInterpolation<double> interp(x, y);

    double val1 = interp(1.5);
    double expected1 = exp(1.5);
    CHECK_CLOSE("log-linear at 1.5", val1, expected1, 1e-3);
}

TEST(Interpolation, naturalCubicThreePointExact) {
    // Natural cubic spline through (0,0), (1,1), (2,0):
    // First derivatives: d0=1.5, d1=0, d2=-1.5
    // Coefficients: a[0]=1.5, b[0]=0, c[0]=-0.5
    // At x=0.5: P(0.5) = 0 + 1.5*0.5 + 0*0.25 + (-0.5)*0.125 = 0.6875
    std::vector<double> x_simple = {0.0, 1.0, 2.0};
    std::vector<double> y_simple = {0.0, 1.0, 0.0};

    const quantape::math::CubicInterpolation<double> interp_simple(
        x_simple, y_simple, quantape::math::CubicDerivativeApprox::Spline);

    double val_simple = interp_simple(0.5);
    double expected_val_simple = 0.6875; // Correct value for natural cubic spline
    CHECK_CLOSE("natural cubic 3-point at 0.5", val_simple, expected_val_simple, 1e-9);
}

TEST(Interpolation, cubicQuadraticRelaxedTolerance) {
    // Natural spline won't match quadratic exactly due to boundary conditions
    // (natural BC has f''=0 at ends, but x^2 has f''=2 everywhere)
    std::vector<double> x_poly = {0.0, 1.0, 2.0, 3.0};
    std::vector<double> y_poly = {0.0, 1.0, 4.0, 9.0}; // x^2

    const quantape::math::CubicInterpolation<double> interp_poly(
        x_poly, y_poly, quantape::math::CubicDerivativeApprox::Spline);

    double val_poly = interp_poly(1.5);
    double expected_poly = 2.25; // 1.5^2
    CHECK_CLOSE("natural cubic x^2 at 1.5", val_poly, expected_poly, 0.1);
}

TEST(Interpolation, cubicSineRelaxedTolerance) {
    // Relaxed tolerance for sparse knots on a transcendental function.
    std::vector<double> x_sin = {0.0, 1.0, 2.0, 3.0, 4.0};
    std::vector<double> y_sin = {0.0, 0.84147098, 0.90929743, 0.14112001, -0.7568025}; // sin(x)

    const quantape::math::CubicInterpolation<double> interp_natural_sin(
        x_sin, y_sin, quantape::math::CubicDerivativeApprox::Spline);

    double val_natural_sin = interp_natural_sin(1.5);
    double expected_sin = sin(1.5);
    CHECK_CLOSE("natural cubic sin at 1.5", val_natural_sin, expected_sin, 1e-3);
}

TEST(Interpolation, bilinearMidpoint) {
    std::vector<double> x = {0, 1};
    std::vector<double> y = {0, 1};
    std::vector<std::vector<double>> z = {{0, 1}, {1, 2}};

    const quantape::math::BilinearInterpolation<double> interp(x, y, z);

    double val = interp(0.5, 0.5);
    CHECK_CLOSE("bilinear at (0.5, 0.5)", val, 1.0, 1e-9);
}

TEST(Interpolation, bicubicReproducesBilinearPlane) {
    std::vector<double> x = {0, 1, 2};
    std::vector<double> y = {0, 1, 2};
    auto f = [](double xv, double yv) { return xv * yv + xv + yv; };
    std::vector<std::vector<double>> z(3, std::vector<double>(3));
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            z[i][j] = f(x[j], y[i]);
        }
    }

    const quantape::math::BicubicInterpolation<double> interp(x, y, z);

    double val = interp(0.5, 0.5);
    double expected = f(0.5, 0.5);
    CHECK_CLOSE("bicubic plane at (0.5, 0.5)", val, expected, 1e-9);
}
