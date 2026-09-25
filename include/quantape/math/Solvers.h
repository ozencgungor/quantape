#ifndef SOLVERS_H
#define SOLVERS_H

/**
 * @file Solvers.h
 * @brief Convenience header for the 1-D root finders (Stan-free)
 *
 * Solvers: Bisection, Brent, Secant, Ridder, False Position, Newton
 * (finite-difference / objective-provided derivative) and Newton with an
 * explicit derivative functor, plus the Thomas tridiagonal solver.
 *
 * Usage:
 *   #include "Math/Solvers.h"
 *   quantape::math::BrentSolver<double> solver;
 *   double root = solver.solve(f, 1e-12, guess, xMin, xMax);
 *
 * For automatic differentiation (var / fvar<var>) also include
 * "Math/Solvers/SolverStanPrimitives.h" — or "Math/StanPrimitives.h" — for
 * exact implicit-function-theorem gradients. This header stays Stan-free.
 */

#include "Solvers/BisectionSolver.h"
#include "Solvers/BrentSolver.h"
#include "Solvers/FalsePositionSolver.h"
#include "Solvers/NewtonSolver.h"
#include "Solvers/RidderSolver.h"
#include "Solvers/SecantSolver.h"
#include "Solvers/Solver1DBase.h"
#include "Solvers/SolverPrimitives.h"
#include "Solvers/TridiagonalSolver.h"

#endif // SOLVERS_H
