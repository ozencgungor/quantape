// test_interpolation.cpp — log-linear, cubic, bilinear and bicubic gates.
//
// The gates below use the support assertions, so they are live in every
// configuration (C assert() would compile out under NDEBUG).

#include "quantape/math/Interpolations/BicubicInterpolation.h"
#include "quantape/math/Interpolations/BilinearInterpolation.h"
#include "quantape/math/Interpolations/CubicInterpolation.h"
#include "quantape/math/Interpolations/LogLinearInterpolation.h"
#include "quantape/math/Interpolations/TensionSplineInterpolation.h"

#include <algorithm>
#include <cmath>
#include <limits>
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

// The natural cubic spline must be C^2 on a NON-UNIFORM grid: the C^2 system's
// off-diagonals are crossed relative to the M-form system, so they only
// coincide with the symmetric ones when dx[i] == dx[i-1]. A uniform grid cannot
// distinguish the two.
TEST(InterpolationTest, naturalSplineIsC2OnNonUniformGrid) {
    const std::vector<double> x{0.0, 1.0, 3.0};
    const std::vector<double> y{0.0, 1.0, 0.0};
    const quantape::math::CubicInterpolation<double> spline(
        x, y, quantape::math::CubicDerivativeApprox::Spline);
    // Reference natural spline (solved independently): d = {1.25, 0.5, -1.0}.
    CHECK_CLOSE("natural spline value at 0.5", spline(0.5), 0.59375, 1e-12);
    // C^2 at the interior knot x = 1: the second derivative must not jump.
    const double eps = 1e-5;
    const auto secondDerivative = [&](double t) {
        return (spline(t + eps) - 2.0 * spline(t) + spline(t - eps)) / (eps * eps);
    };
    CHECK_CLOSE_REL("natural spline C2 at the interior knot", secondDerivative(1.0), -1.5, 1e-4);
    // A uniform grid exercises the same symmetric system.
    const std::vector<double> xu{0.0, 1.0, 2.0};
    const std::vector<double> yu{0.0, 1.0, 0.0};
    const quantape::math::CubicInterpolation<double> uniform(
        xu, yu, quantape::math::CubicDerivativeApprox::Spline);
    CHECK_CLOSE("uniform-grid spline value at 0.5", uniform(0.5), 0.6875, 1e-12);
}

// As tension -> 0 the tension spline tends to the natural cubic spline, and as
// tension -> infinity to the piecewise linear interpolant. The limits are
// numerically stable down to sigma ~ 1e-8 and up to sigma*h ~ 1000: the
// 1/(sigma^2 sinh p) factor does not cancel catastrophically at small sigma
// and sinh does not overflow at large sigma.
TEST(InterpolationTest, tensionSplineLimitsAreStable) {
    const std::vector<double> x{0.0, 1.0, 2.5, 4.0, 6.0};
    const std::vector<double> y{0.0, 1.5, 1.2, 2.5, 2.0};
    const quantape::math::CubicInterpolation<double> natural(
        x, y, quantape::math::CubicDerivativeApprox::Spline);
    const auto worstGap = [&](double sigma) {
        const quantape::math::TensionSplineInterpolation<double> ts(x, y, sigma);
        double worst = 0.0;
        for (std::size_t i = 0; i + 1 < x.size(); ++i) {
            const double mid = 0.5 * (x[i] + x[i + 1]);
            worst = std::max(worst, std::fabs(ts(mid) - natural(mid)));
        }
        return worst;
    };
    // Quadratic convergence towards the natural spline.
    const double gap1e2 = worstGap(1e-2);
    const double gap1e4 = worstGap(1e-4);
    const double gap1e6 = worstGap(1e-6);
    const double gap1e8 = worstGap(1e-8);
    CHECK_CLOSE_REL("tension 1e-2 vs natural", gap1e2, 9.2e-6, 0.1);
    EXPECT_LT(gap1e4, 1e-8);
    EXPECT_LT(gap1e6, 1e-12);
    EXPECT_LT(gap1e8, 1e-14);
    // Large tension: finite, and close to the linear interpolant.
    for (double sigma : {500.0, 1000.0}) {
        SCOPED_TRACE(::testing::Message() << "sigma=" << sigma);
        const quantape::math::TensionSplineInterpolation<double> ts(x, y, sigma);
        for (std::size_t i = 0; i + 1 < x.size(); ++i) {
            const double mid = 0.5 * (x[i] + x[i + 1]);
            const double linear = 0.5 * (y[i] + y[i + 1]);
            ASSERT_TRUE(std::isfinite(ts(mid))) << "large tension must stay finite";
            CHECK_CLOSE("large-tension limit is linear", ts(mid), linear, 1e-6);
        }
    }
}

namespace {

/// Standard Akima node slopes (one-sided end stencils included) with the
/// unclamped tie rule; valid on grids whose secants are well separated.
std::vector<double> standardAkimaSlopes(const std::vector<double>& x,
                                        const std::vector<double>& y) {
    const std::size_t n = x.size();
    std::vector<double> s(n - 1);
    for (std::size_t i = 0; i + 1 < n; ++i) {
        s[i] = (y[i + 1] - y[i]) / (x[i + 1] - x[i]);
    }
    const auto mix = [](double w1, double a, double w2, double b) {
        return (w1 * a + w2 * b) / (w1 + w2);
    };
    std::vector<double> d(n);
    const double sLeft = 2.0 * s[0] - s[1];
    const double sLeftLeft = 3.0 * s[0] - 2.0 * s[1];
    d[0] = mix(std::fabs(s[1] - s[0]), sLeft, std::fabs(sLeft - sLeftLeft), s[0]);
    d[1] = mix(std::fabs(s[2] - s[1]), s[0], std::fabs(s[0] - sLeft), s[1]);
    for (std::size_t i = 2; i + 2 < n; ++i) {
        d[i] = mix(std::fabs(s[i + 1] - s[i]), s[i - 1], std::fabs(s[i - 1] - s[i - 2]), s[i]);
    }
    const double sLast = s[n - 2];
    const double sPrev = s[n - 3];
    const double sRight = 2.0 * sLast - sPrev;
    const double sRightRight = 3.0 * sLast - 2.0 * sPrev;
    d[n - 2] = mix(std::fabs(sRight - sLast), sPrev, std::fabs(sPrev - s[n - 4]), sLast);
    d[n - 1] = mix(std::fabs(sRightRight - sRight), sLast, std::fabs(sLast - sPrev), sRight);
    return d;
}

} // namespace

// The Akima tie fallback must obey the cubic-Hermite monotonicity bound: on
// data that is non-negative and flat up to a point, the interpolant over the
// flat segment stays at the data value instead of dipping below it.
TEST(InterpolationTest, akimaTieFallbackRespectsMonotoneBound) {
    const std::vector<double> x{0.0, 1.0, 2.0, 3.0, 4.0};
    const std::vector<double> y{0.0, 0.0, 0.0, 10.0, 20.0};
    const quantape::math::CubicInterpolation<double> interp(
        x, y, quantape::math::CubicDerivativeApprox::Akima);

    double minValue = std::numeric_limits<double>::infinity();
    for (int k = 0; k <= 40000; ++k) {
        minValue = std::min(minValue, interp(4.0 * static_cast<double>(k) / 40000.0));
    }
    EXPECT_GE(minValue, 0.0) << "tie fallback must keep the interpolant in the data range";
    CHECK_CLOSE("tied node slope", interp.aCoeffs()[2], 0.0, 1e-15);

    // C1 continuity at every knot: the left-segment end derivative equals the
    // right-segment start derivative.
    const auto& a = interp.aCoeffs();
    const auto& b = interp.bCoeffs();
    const auto& c = interp.cCoeffs();
    for (std::size_t i = 0; i + 1 < a.size(); ++i) {
        const double h = x[i + 1] - x[i];
        const double leftDerivative = a[i] + 2.0 * b[i] * h + 3.0 * c[i] * h * h;
        CHECK_CLOSE("Akima C1 continuity", leftDerivative, a[i + 1], 1e-12);
    }
}

// On well-separated secants the monotonicity clamp must not alter the standard
// Akima slopes: every aCoeff is the reference-formula node derivative.
TEST(InterpolationTest, akimaWellSeparatedMatchesReferenceFormula) {
    const std::vector<double> x{0.0, 0.7, 1.9, 2.4, 3.8, 5.0, 6.1};
    const std::vector<double> y{1.3, -0.6, 2.4, 1.1, 5.3, 2.2, 4.8};
    const quantape::math::CubicInterpolation<double> interp(
        x, y, quantape::math::CubicDerivativeApprox::Akima);
    std::vector<double> reference = standardAkimaSlopes(x, y);
    reference.pop_back(); // aCoeffs stores the segment-start node derivatives
    CHECK_CLOSE_SEQ("Akima well-separated slopes", interp.aCoeffs(), reference, 1e-14);
}
