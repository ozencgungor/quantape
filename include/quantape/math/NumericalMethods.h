#ifndef NUMERICAL_METHODS_H
#define NUMERICAL_METHODS_H

/**
 * @file NumericalMethods.h
 * @brief Main convenience header for the numerical components (Stan-free)
 *
 * Covers integration, root finding, interpolation and optimization through
 * the component umbrellas:
 *
 *   `#include "quantape/math/NumericalMethods.h"`   // double-only consumers
 *
 *   quantape::math::TrapezoidIntegratorDefault<double> integ(1e-6, 1000);
 *   double I = integ([](double x) { return x * x; }, 0.0, 1.0);
 *
 *   quantape::math::BrentSolver<double> solver;
 *   double root = solver.solve(f, 1e-12, guess, 0.0, 1.0);
 *
 * For automatic differentiation (stan::math::var / fvar<...>) include the
 * single AD umbrella as well:
 *
 *   `#include "quantape/math/StanMath.h"`
 *   `#include "quantape/math/StanPrimitives.h"`
 *
 * Random-number utilities (Math/Random/) are intentionally not included;
 * include them directly, e.g. "quantape/math/Random/PCGRandom.hpp".
 */

// Component umbrellas (all Stan-free)
#include "Integrals.h"
#include "Interpolations.h"
#include "Optimization.h"
#include "Solvers.h"

#endif // NUMERICAL_METHODS_H
