#ifndef OPTIMIZATION_H
#define OPTIMIZATION_H

/**
 * @file Optimization.h
 * @brief Convenience header for the optimizers (Stan-free)
 *
 * Primitives (result codes, stop criteria, state, multiplier export), the
 * `Optimizer<DoubleT, Impl>` base, the shared strong-Wolfe line search
 * and the derivative-based algorithms (L-BFGS, truncated Newton, SLSQP,
 * AUGLAG) plus constraints/bounds and the dense active-set QP solver.
 *
 * For automatic differentiation also include
 * "quantape/math/Optimization/OptimizerStanPrimitives.h" — or "quantape/math/StanPrimitives.h"
 * — for exact gradients, HVPs and constraint Jacobians; the first-order IFT
 * layer (dp/dm, KKT sensitivities, var composition) lives in
 * "quantape/math/Optimization/ImplicitFunction.h" (Stan-dependent, like
 * OptimizerStanPrimitives.h). This header stays Stan-free.
 */

#include "Optimization/AugLag.h"
#include "Optimization/Constraint.h"
#include "Optimization/LBFGS.h"
#include "Optimization/LineSearch.h"
#include "Optimization/OptimizerPrimitives.h"
#include "Optimization/QpSolver.h"
#include "Optimization/SLSQP.h"
#include "Optimization/TNewton.h"

#endif // OPTIMIZATION_H
