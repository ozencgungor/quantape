#ifndef STAN_PRIMITIVES_H
#define STAN_PRIMITIVES_H

/**
 * @file StanPrimitives.h
 * @brief Convenience header for all AD dispatch layers (Stan-only)
 *
 * Include this together with a Stan Math header (`<stan/math.hpp>`, and
 * `<stan/math/mix.hpp>` for fvar). It pulls in every AD specialization:
 *
 *   - Integrals/IntegratorStanPrimitives.h   one-pass rule extraction
 *   - Solvers/SolverStanPrimitives.h         IFT gradients / pathwise Hessians
 *   - Interpolations/InterpolationStanPrimitives.h  passive-abscissa fast paths
 *   - Optimization/OptimizerStanPrimitives.h valueGrad / exact HVP / constraint Jacobians
 *   - Optimization/ImplicitFunction.h        dp/dm, KKT sensitivities, var composition
 *   - Autodiff/Hvp.h                         forward-over-reverse HVP
 *
 * The component umbrellas (NumericalMethods.h, Integrals.h, Solvers.h,
 * Interpolations.h, Optimization.h) stay Stan-free; this header is the single
 * include for AD users.
 */

#include "Autodiff/Hvp.h"
#include "Integrals/IntegratorStanPrimitives.h"
#include "Interpolations/InterpolationStanPrimitives.h"
#include "Optimization/ImplicitFunction.h"
#include "Optimization/OptimizerStanPrimitives.h"
#include "Solvers/SolverStanPrimitives.h"

#endif // STAN_PRIMITIVES_H
