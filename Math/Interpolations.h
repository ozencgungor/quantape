#ifndef INTERPOLATIONS_H
#define INTERPOLATIONS_H

/**
 * @file Interpolations.h
 * @brief Main header for AD-compatible interpolation library
 *
 * This library provides template-based interpolation methods that work
 * with both regular floating point (double) and automatic differentiation
 * types (stan::math::var).
 *
 * Available 1D interpolations:
 * - LinearInterpolation: Simple linear interpolation
 * - LogLinearInterpolation: Log-linear interpolation
 * - CubicInterpolation: Cubic interpolation with multiple methods (Spline, Akima, etc.)
 *
 * Available 2D interpolations:
 * - BilinearInterpolation: 2D linear interpolation
 * - BicubicInterpolation: 2D cubic interpolation (supports Spline, Akima, etc.)
 *
 * Usage:
 *   #include "Math/Interpolations.h"
 *
 *   // 1D Linear interpolation
 *   std::vector<double> x = {0.0, 1.0, 2.0, 3.0};
 *   std::vector<double> y = {0.0, 1.0, 4.0, 9.0};
 *   Math::LinearInterpolation<double> interp(x, y);
 *   double value = interp(1.5);  // Interpolate at x=1.5
 *
 *   // Cubic spline interpolation (natural spline)
 *   Math::CubicInterpolation<double> spline(x, y);  // Defaults to Spline method
 *   double spline_value = spline(1.5);
 *   double derivative = spline.derivative(1.5);
 *
 *   // 2D Bicubic interpolation
 *   std::vector<std::vector<double>> z = {{...}, {...}, {...}};
 *   Math::BicubicInterpolation<double> bicubic(x, y, z);  // Defaults to Spline
 *   double val_2d = bicubic(1.5, 1.5);
 *
 *   // For automatic differentiation (requires Stan Math)
 *   using ADVariableT = stan::math::var;
 *   std::vector<ADVariableT> x_ad = {0.0, 1.0, 2.0, 3.0};
 *   std::vector<ADVariableT> y_ad = {0.0, 1.0, 4.0, 9.0};
 *   Math::CubicInterpolation<ADVariableT> ad_spline(x_ad, y_ad);
 *   ADVariableT result_ad = ad_spline(ADVariableT(1.5));
 */

// Base classes
#include "Interpolations/Interpolation.h"
#include "Interpolations/Interpolation2D.h"

// 1D interpolations
#include "Interpolations/CubicInterpolation.h"
#include "Interpolations/LinearInterpolation.h"
#include "Interpolations/LogLinearInterpolation.h"

// 2D interpolations
#include "Interpolations/BicubicInterpolation.h"
#include "Interpolations/BilinearInterpolation.h"

// Type aliases for convenience
namespace Math {
// Stan Math AD type alias (when Stan Math is available)
// Uncomment when linking with Stan Math:
// #include <stan/math.hpp>
// using ADVariableT = stan::math::var;
}

#endif // INTERPOLATIONS_H
