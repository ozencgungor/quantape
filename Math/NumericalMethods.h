#ifndef NUMERICAL_METHODS_H
#define NUMERICAL_METHODS_H

/**
 * @file NumericalMethods.h
 * @brief Main header for AD-compatible numerical methods library
 *
 * This library provides template-based numerical integration and
 * root finding methods that work with both regular floating point
 * (double) and automatic differentiation types (stan::math::var).
 *
 * Usage:
 *   #include "Math/NumericalMethods.h"
 *
 *   // For regular computation
 *   Math::TrapezoidIntegratorDefault<double> integrator(1e-6, 1000);
 *   double result = integrator([](double x) { return x*x; }, 0.0, 1.0);
 *
 *   // For automatic differentiation (requires Stan Math)
 *   using ADVariableT = stan::math::var;
 *   Math::TrapezoidIntegratorDefault<ADVariableT> ad_integrator(1e-6, 1000);
 *   ADVariableT result_ad = ad_integrator([](ADVariableT x) { return x*x; },
 *                                          ADVariableT(0.0), ADVariableT(1.0));
 */

// Integration methods
#include "Integrals/Integrator.h"
#include "Integrals/TrapezoidIntegrator.h"
#include "Integrals/SimpsonIntegrator.h"
#include "Integrals/GaussianQuadrature.h"

// Solver methods
#include "Solvers/Solver1DBase.h"
#include "Solvers/BisectionSolver.h"
#include "Solvers/BrentSolver.h"
#include "Solvers/SecantSolver.h"
#include "Solvers/NewtonSolver.h"
#include "Solvers/RidderSolver.h"
#include "Solvers/FalsePositionSolver.h"

// Type aliases for convenience
namespace Math {
    // Stan Math AD type alias (when Stan Math is available)
    // Uncomment when linking with Stan Math:
    // #include <stan/math.hpp>
    // using ADVariableT = stan::math::var;
}

#endif // NUMERICAL_METHODS_H
