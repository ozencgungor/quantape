#ifndef INTEGRALS_H
#define INTEGRALS_H

/**
 * @file Integrals.h
 * @brief Convenience header for the integration methods (Stan-free)
 *
 * Integrators: Trapezoid (Default/MidPoint policies), Simpson, Gauss-Lobatto,
 * Gauss-Legendre, Tanh-Sinh.
 *
 * Usage:
 *   #include "Math/Integrals.h"
 *   quantape::math::TrapezoidIntegratorDefault<double> integ(1e-8, 1000);
 *   double I = integ([](double x) { return x * x; }, 0.0, 1.0);
 *
 * For automatic differentiation (var / fvar<var>) also include
 * "Math/Integrals/IntegratorStanPrimitives.h" — or the single AD umbrella
 * "Math/StanPrimitives.h" — and the integrators use the optimized one-pass
 * rule-extraction path automatically. This header stays Stan-free.
 */

#include "Integrals/GaussLobattoIntegrator.h"
#include "Integrals/GaussianQuadrature.h"
#include "Integrals/Integrator.h"
#include "Integrals/SimpsonIntegrator.h"
#include "Integrals/TanhSinhIntegrator.h"
#include "Integrals/TrapezoidIntegrator.h"

#endif // INTEGRALS_H
