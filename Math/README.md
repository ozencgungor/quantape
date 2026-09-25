# Math Library - AD-Compatible Numerical Methods

A template-based numerical methods library supporting regular floating-point
computation (`double`) and automatic differentiation (`stan::math::var` for
gradients, `stan::math::fvar<...>` for Hessians/HVPs), selected automatically
by the scalar template parameter.

## Overview

This library provides comprehensive numerical integration, root finding,
interpolation, optimization and random-number utilities, all designed to work
seamlessly with automatic differentiation (AD) frameworks. The implementation
is inspired by QuantLib's design patterns but fully templated for AD
compatibility.

## Library Structure

```
Math/
├── Autodiff/
│   ├── Hvp.h                     # Hessian-vector products (forward-over-reverse)
│   └── PrimalExtraction.h        # shared recursive primal extraction
│
├── Integrals/
│   ├── Integrator.h              # Base integrator class template
│   ├── TrapezoidIntegrator.h     # Trapezoid rule with adaptive refinement
│   ├── SimpsonIntegrator.h       # Simpson's rule (Richardson extrapolation)
│   ├── GaussLobattoIntegrator.h  # Adaptive Gauss-Lobatto
│   ├── GaussianQuadrature.h      # Gauss-Legendre quadrature
│   ├── TanhSinhIntegrator.h      # Double-exponential quadrature
│   └── IntegratorStanPrimitives.h # var/fvar<var> one-pass rule extraction
│
├── Interpolations/
│   ├── Interpolation.h           # CRTP base (1D); Interpolation2D.h for 2D
│   ├── LinearInterpolation.h
│   ├── LogLinearInterpolation.h
│   ├── CubicInterpolation.h      # Spline/Parabolic/Akima/Kruger/Harmonic
│   ├── BilinearInterpolation.h / BicubicInterpolation.h
│   └── InterpolationStanPrimitives.h # passive-abscissa fast paths
│
├── Optimization/
│   ├── OptimizerPrimitives.h     # result codes, StopCriteria/State, stop tests, CRTP base
│   ├── OptimizerStanPrimitives.h # valueGrad, exact HVP, constraint Jacobians
│   ├── ImplicitFunction.h        # first-order IFT: dp/dm, KKT sensitivities, var composition
│   ├── LineSearch.h              # strong-Wolfe bracket/zoom (PS1L01 constants)
│   ├── LBFGS.h                   # L-BFGS (PLIS semantics), CRTP method
│   ├── TNewton.h                 # truncated Newton (PNET semantics), exact HVP
│   ├── Constraint.h              # bounds, constraint concepts, feasibility helpers
│   ├── QpSolver.h                # dense active-set QP (equalities + inequalities)
│   ├── SLSQP.h                   # SQP with damped BFGS + L1 merit line search
│   └── AugLag.h                  # augmented Lagrangian over L-BFGS
│
├── Random/
│   ├── PCGRandom.hpp / PCGExtras.hpp / PCGUint128.hpp  # PCG generators
│   ├── ZigguratNormal.h          # Marsaglia ziggurat normal sampler
│   ├── McFarlandNormal.h         # McFarland modified ziggurat
│   └── Sobol/                    # Sobol' low-discrepancy sequences
│
├── Solvers/
│   ├── SolverPrimitives.h        # concepts, SolverState, SolverConfig, rebind
│   ├── Solver1DBase.h            # Base solver class with CRTP pattern
│   ├── BisectionSolver.h         # Bisection method (most robust)
│   ├── SecantSolver.h            # Secant method (no derivatives)
│   ├── NewtonSolver.h            # Newton-Raphson (+ explicit-derivative variant)
│   ├── BrentSolver.h             # Brent's method (hybrid approach)
│   ├── RidderSolver.h            # Ridder's exponential formula
│   ├── FalsePositionSolver.h     # False position (regula falsi)
│   ├── TridiagonalSolver.h       # Thomas algorithm (double + AD scalars)
│   └── SolverStanPrimitives.h    # var IFT gradients, fvar pathwise Hessians
│
├── Integrals.h                   # umbrella: all integrators (Stan-free)
├── Interpolations.h              # umbrella: all interpolators (Stan-free)
├── Optimization.h                # umbrella: optimizer building blocks (Stan-free)
├── Solvers.h                     # umbrella: all 1-D solvers (Stan-free)
├── StanPrimitives.h              # umbrella: all AD dispatch layers (Stan-only)
└── NumericalMethods.h            # umbrella: Integrals + Interpolations + Optimization + Solvers
```

## Features

### Integration Methods

| Method | Order | Convergence | Best For |
|--------|-------|-------------|----------|
| **Trapezoid** | O(h²) | Adaptive | General purpose |
| **Simpson** | O(h⁴) | Adaptive | Smooth functions |
| **Gauss-Lobatto** | O(h⁷) | Adaptive | Smooth functions over closed intervals |
| **Gauss-Legendre** | 2n-1 polynomial | Fixed order | High accuracy, few evaluations |
| **Tanh-Sinh** | ~exponential | Adaptive | Endpoint singularities, infinite domains |

**Performance from tests:**
- Trapezoid: ~10⁻⁹ error with ~8K evaluations
- Simpson: Machine precision with ~65 evaluations
- Gauss-Legendre (order 20): ~10⁻¹⁶ error with 20 evaluations

### Root Finding Methods

| Method | Convergence | Derivatives | Robustness | Best For |
|--------|-------------|-------------|------------|----------|
| **Bisection** | Linear | No | Highest | Guaranteed convergence |
| **Secant** | ~1.618 | No | Good | Fast without derivatives |
| **Newton** | Quadratic | Yes | Medium | When derivatives available |
| **Brent** | Super-linear | No | High | Best general-purpose choice |
| **Ridder** | ~1.839 | No | High | Faster than bisection |
| **False Position** | Super-linear | No | High | Alternative to bisection |

**Performance from tests (finding √2, accuracy 1e-10):**
- Bisection: ~10⁻¹¹ error
- Secant: ~10⁻¹⁵ error
- Newton: Exact (0 error)
- Brent: Exact (0 error), ~7 evaluations vs bisection's ~42 on `cos(x)-x`
- Ridder: ~10⁻¹⁶ error
- False Position: ~10⁻¹⁵ error

## Usage

### Basic Integration

```cpp
#include "Math/NumericalMethods.h"

using namespace Math;

// Define function to integrate
auto f = [](double x) { return x * x; };

// Method 1: Trapezoid integration
TrapezoidIntegratorDefault<double> trap(1e-8, 1000);
double result = trap(f, 0.0, 1.0);  // ∫₀¹ x² dx = 1/3

// Method 2: Simpson's rule (higher order)
SimpsonIntegrator<double> simpson(1e-8, 1000);
result = simpson(f, 0.0, 1.0);

// Method 3: Gauss-Legendre (best accuracy)
GaussLegendreIntegrator<double> gauss(20);  // order 20
result = gauss(f, 0.0, 1.0);
```

### Root Finding

```cpp
auto f = [](double x) { return x * x - 2.0; };  // Find √2

// Simple bisection (most robust)
BisectionSolver<double> bisection;
double root = bisection.solve(f, 1e-10, 1.5, 0.0, 3.0);

// Brent's method (best general-purpose)
BrentSolver<double> brent;
root = brent.solve(f, 1e-10, 1.5, 0.0, 3.0);

// Newton with explicit derivative (fastest when available)
auto df = [](double x) { return 2.0 * x; };
NewtonSolverWithDerivative<double, decltype(df)> newton(df);
root = newton.solve(f, 1e-10, 1.5, 0.0, 3.0);

// Automatic bracketing
root = brent.solve(f, 1e-10, 1.5, 0.1);  // Auto-bracket from guess
```

### Automatic Differentiation Support

AD dispatch is automatic by the `DoubleT` template parameter. Include the
primitives header for each primitive you use (plus a Stan Math header) and
call the classes as usual — there is no separate AD entry point:

```cpp
#include <stan/math.hpp>
#include <stan/math/mix.hpp>
#include "Math/NumericalMethods.h"
#include "Math/StanPrimitives.h"

using stan::math::var;
using namespace Math;

// ── var: exact gradients ──────────────────────────────────────────────
// roots: solved at double precision, gradients by the implicit function
// theorem (exact for every solver, including bisection)
var theta = 4.0;
auto g_ad = [&theta](const var& x) { return x * x - theta; };
BrentSolver<var> solver;
var root = solver.solve(g_ad, 1e-10, var(1.5), var(0.0), var(3.0));
root.grad();                        // theta.adj() = 1/(2 sqrt(theta))

// integrals: converged rule extracted once, exact weighted gradients
TrapezoidIntegratorDefault<var> integ(1e-8, 1000);
var I = integ([](const var& x) { return x * x; }, var(0.0), var(1.0));

// interpolation: weights are AD expressions, so the evaluation point is
// differentiated too — interp(solve(...)) composes through both primitives
std::vector<var> xs{0.0, 1.0, 2.0}, ys{0.0, 1.0, 4.0};
LinearInterpolation<var> interp(xs, ys);
var z = interp(root);

// ── fvar<...>: Hessians and Hessian-vector products ───────────────────
using std::sqrt;
Eigen::VectorXd x0(1);
x0 << 4.0;
double fx;
Eigen::VectorXd grad;
Eigen::Matrix<double, -1, -1> H;
stan::math::hessian(
    [&](const auto& th) {
        using S = typename std::decay_t<decltype(th)>::Scalar;
        BrentSolver<S> s;
        return s.solve([&](const S& x) { return x * x - th(0); }, 1e-12, S(1.5), S(0.0),
                       S(3.0));
    },
    x0, fx, grad, H);

#include "Math/Autodiff/Hvp.h"
auto quartic = [](const auto& x) { return x * x * x * x - 3.0 * x * x + 2.0 * x; };
double hv = Math::hvp(quartic, 0.7, 2.3); // f''(0.7) * 2.3

// ── passive-abscissa fast path ────────────────────────────────────────
// when the query coordinate is known to be constant, restore the
// node-minimal AD paths (the x adjoint is intentionally not pushed):
var y = interp.evaluateFixed(root); // 2D: bilinear.evaluateFixed(x, y)
```

Contracts and traps are documented on the primitives headers:
`var` solver gradients are taken w.r.t. parameters inside the objective only
(brackets/guesses are used by value); `fvar` pathwise derivatives are exact
for smooth solvers, and bisection's pathwise derivative is zero (use `var`);
control flow is primal-pinned, so derivatives at branches are the one-sided
values of the selected branch.

**Gotcha — Eigen include order.** Stan injects `MatrixBase::val()/adj()` into
Eigen through `EIGEN_MATRIXBASE_PLUGIN` (`prim/fun/Eigen.hpp`), and the plugin
only applies if Eigen is first parsed AFTER the define. A translation unit
that includes `<Eigen/Dense>` before `<stan/math.hpp>` sets Eigen's include
guards first, silently drops the plugin, and every matrix-var path
(`mdivide_left_spd`, GP covariances, `grad()`) fails with "no member named
'adj'". `format_code.sh` (clang-format) re-sorts such include blocks, so
this can regress silently.

**The fix is `"Math/StanMath.h"`** — the single Stan include. It fixes the
order internally (stan → mix → Eigen) behind `// clang-format off` guards,
provides `<Eigen/Dense>` transitively, and `.clang-format`'s
`IncludeCategories` sort it before any third-party include. Project code
includes it instead of raw `<stan/math.hpp>` + `<Eigen/Dense>` — the ordering
problem disappears entirely.

**Optimizer sensitivities (IFT).** Once an optimum is in hand,
`Math/Optimization/ImplicitFunction.h` differentiates it:

```cpp
auto f2 = [](const auto& x, const auto& m) { /* S-typed objective of x and data m */ };
std::vector<double> dp_dm;
Math::IftResult ift;
Math::iftUnconstrained(f2, x_hat, m, dp_dm, ift);   // dp/dm = -H^{-1} G, exact AD
// constrained: iftKkt(f2, g2, h2, bounds, x_hat, m, lam, nu, dp_dm, dlam_dm, dnu_dm, ift)
// one-call:  minimizeDifferential(f2, g2, h2, bounds, m, x, state, ift, &dp_dm, ...)
// on-tape:   minimizeDifferentialVar(f2, ..., m_var, x0, p_hat_var, ...)
```

### Hessian-Vector Products

`Math/Autodiff/Hvp.h` provides forward-over-reverse HVPs for scalar-generic
objectives, the primitive for Newton-CG / trust-region layers:

```cpp
#include "Math/Autodiff/Hvp.h"
auto f = [](const auto& x) { return x * x * x * x - 3.0 * x * x + 2.0 * x; };
double hvp = Math::hvp(f, 0.7, 2.3); // f''(x) * v, one forward-over-reverse pass
```

## Design Patterns

### Template Parameter `DoubleT`

All classes are templated on the numeric type `DoubleT`:

- `DoubleT = double` for regular computation
- `DoubleT = stan::math::var` for automatic differentiation
- Can be extended to other numeric types (e.g., `mpfr::mpreal`, `boost::multiprecision`)

### Policy-Based Integration

The `TrapezoidIntegrator` uses policy classes to define refinement strategies:

```cpp
// Default policy: N → 2N (doubles intervals)
TrapezoidIntegrator<double, DefaultPolicy> default_integrator(1e-8, 1000);

// MidPoint policy: N → 3N (triples intervals)
TrapezoidIntegrator<double, MidPointPolicy> midpoint_integrator(1e-8, 1000);

// Convenient typedefs
TrapezoidIntegratorDefault<double> trap1(1e-8, 1000);
TrapezoidIntegratorMidPoint<double> trap2(1e-8, 1000);
```

### CRTP for Solvers

The `Solver1D` base class uses the Curiously Recurring Template Pattern (CRTP) for static polymorphism:

```cpp
template<typename DoubleT, typename Impl>
class Solver1D { ... };

template<typename DoubleT>
class BrentSolver : public Solver1D<DoubleT, BrentSolver<DoubleT>> { ... };
```

This provides zero-overhead abstraction - as efficient as hand-coded implementations.

## Components

### Integrals

**Integrator.h**
- Base class for all integrators
- Accuracy and max evaluations control
- Evaluation counting and error reporting

**TrapezoidIntegrator.h**
- Adaptive trapezoid integration
- Policy-based refinement strategies (Default, MidPoint)
- Convergence detection
- Second-order accuracy (O(h²))

**SimpsonIntegrator.h**
- Simpson's 1/3 rule via Richardson extrapolation
- Fourth-order accuracy (O(h⁴))
- Formula: `(4*Trap_fine - Trap_coarse) / 3`
- Excellent for smooth functions

**GaussianQuadrature.h**
- Gauss-Legendre quadrature (orders 2, 3, 4, 5, 6, 10, 20)
- Exact for polynomials up to degree 2n-1
- Tabulated weights and abscissas
- Domain transformation for arbitrary intervals
- Best accuracy with minimal evaluations

### Solvers

**Solver1DBase.h**
- CRTP base class for all 1D solvers
- Automatic bracketing support
- Bounded search domains
- Common interface for all derived solvers

**BisectionSolver.h**
- Simplest and most robust method
- Guaranteed convergence for continuous functions
- Linear convergence (slow but sure)
- Use when robustness is critical

**SecantSolver.h**
- No derivatives required
- Super-linear convergence (~1.618)
- Faster than bisection
- May fail to converge in pathological cases

**NewtonSolver.h**
- Quadratic convergence near root
- Two variants:
  - `NewtonSolver`: uses the objective's `derivative(x)` when present,
    otherwise scaled central finite differences
  - `NewtonSolverWithDerivative<DoubleT, Derivative>`: derivative functor is
    a template parameter (stored by value, no `std::function`)
- Falls back to bisection if the Newton step jumps outside the brackets

**BrentSolver.h**
- Hybrid: bisection + secant + inverse quadratic interpolation
- Zeroin (Forsythe–Malcolm–Moler) formulation; interpolation branches active
- Guaranteed convergence (like bisection)
- Super-linear convergence (like secant): ~7 evaluations vs bisection's ~42
  on `cos(x) - x` to 1e-12
- **Recommended general-purpose solver**

**RidderSolver.h**
- Exponential formula method
- Convergence order ~1.839
- More robust than secant
- Good alternative to Brent

**FalsePositionSolver.h**
- Linear interpolation (regula falsi)
- Super-linear convergence
- Alternative to bisection
- Can be slow if one endpoint becomes "stuck"

**SolverPrimitives.h / SolverStanPrimitives.h**
- `SolverPrimitives.h` (Stan-free): `SolverScalar` / `SolverFunction`
  concepts, `SolverState` (all solve working state is local, so solvers are
  thread-safe and re-entrant), `SolverConfig`, `SolverRebind`
- `SolverStanPrimitives.h`: `var` = double-precision solve + one
  implicit-function-theorem Newton polish (`dx/dθ = −f_θ/f_x`, exact for
  every solver incl. bisection); `fvar<...>` = pathwise route (exact for
  smooth solvers)

**TridiagonalSolver.h**
- Thomas algorithm, generic over `double` / AD scalars
- Zero-pivot and size validation; used by cubic-spline coefficient construction

### Interpolations

**Interpolation.h / Interpolation2D.h**
- CRTP bases: grid coordinates are `double`, node values are `DoubleT`
- Default evaluation builds the weights as `DoubleT`, so the query
  coordinate is on the tape (`dI/dx`, mixed `d²I/dxdy` blocks included)
- `evaluateFixed` / `derivativeFixed` restore the node-minimal callback
  paths (weights as doubles, x adjoint not pushed) when the coordinate is
  constant; 2D variant: `evaluateFixed(x, y)`

**LinearInterpolation.h / LogLinearInterpolation.h / CubicInterpolation.h**
- Linear and log-linear (exponential of the log-linear form) interpolation
- Cubic with Spline (default), Parabolic, Akima, Kruger, Harmonic derivatives
- Adaptive cubic methods expose true active-branch curvature through the
  default AD path; the fast path uses the branch-pinned probe

**BilinearInterpolation.h / BicubicInterpolation.h**
- Tensor-product 2D interpolation over AD node values, both query
  coordinates differentiated by default

**InterpolationStanPrimitives.h**
- Passive-abscissa specializations used by `evaluateFixed` / fast paths
  (var: one tape node + adjoint pushes; fvar<var>: callback vars for the
  knot Hessian blocks)

### Optimization

**OptimizerPrimitives.h**
- `OptimizeResult` codes, `StopCriteria` (NLopt semantics: ftol/xtol
  rel/abs, gradient tolerance, stop value, eval/time budgets, weighted
  x-norms), `OptimizerState`
- Stop tests: `stopFtol`, `stopX`, `stopDx`, `stopGrad`, `stopEvals`,
  `stopTime`
- Concepts: `OptimizationScalar`, `VectorObjective`, `VectorConstraint`
- CRTP base `Optimizer<DoubleT, Impl>`: configuration only, all working
  state local to each `minimize()` call (thread-safe, re-entrant); `DoubleT`
  selects the derivative backend (`double` value-only, `var` exact
  gradients, `fvar<...>` gradients/HVPs)

**OptimizerStanPrimitives.h**
- `valueGrad<DoubleT>`: value + exact gradient (one reverse pass inside a
  nested scope)
- `hvp`: exact Hessian-vector products by forward-over-reverse, the
  replacement for finite-difference HVPs in truncated Newton
- `constraintValueJacobian<DoubleT>`: constraint values + row-major Jacobians

**LineSearch.h**
- Strong-Wolfe bracket/zoom (Nocedal & Wright 3.5/3.6) with NLopt PS1L01
  constants (`c1 = 1e-4`, `c2 = 0.9`), bisection zoom, step clamps

**LBFGS.h**
- `LBFGS<DoubleT>` CRTP method (PLIS semantics): two-loop recursion with
  `gamma = s.y / y.y` scaling, restart to steepest descent on nonpositive
  curvature or failed uniform-descent test, Wolfe line search
- AD backends (`var` / `fvar<...>`): scalar-generic objective, exact gradient
  from one reverse pass; **double mode**: AD-free value+gradient callback
  `double f(const std::vector<double>&, std::vector<double>&)` (NLopt-style,
  analytic or caller-side Stan)
- Memory default `max(10, n)` capped by `maxeval`; all work state local to
  `minimize()` (re-entrant)

**TNewton.h**
- `TNewton<DoubleT>` truncated Newton (PNET semantics, NLopt
  `NLOPT_LD_TNEWTON*`): inner CG for `H d = -g` with exact AD Hessian-vector
  products (forward-over-reverse, one fvar tape per inner step), forcing term
  `eta = min(0.8, sqrt(||g||_2), 1/nit)^2`, max `n + 3` inner steps
- `restart` flag mirrors NLopt's mos1 split: negative curvature on the first
  CG step falls back to steepest descent in BOTH modes; a failed PNET uniform
  descent test (`g.s + 1e-4 * ||g|| * ||s|| <= 0`) uses `s = -g` when
  `restart = true`, or recomputes the CG and fails (`iterm = -10`) otherwise
- PNET's absolute guards are scale-relative for exact AD: curvature
  `pAp > 1e-14 * ||p|| * ||Ap||` (vs `alf <= 1e-120`) and the descent test
  above (vs an absolute `gd < -1e-10`, which rejects the legitimate ~1e-16
  Newton directional derivative near the minimum)
- Line search with PS1L01-scaled step bounds
  (`rmin = 1e-10*||g||/||s||`, `rmax = min(1e10*||g||/||s||, 1e16/||s||)`);
  identity preconditioner v1 (L-BFGS preconditioner is a follow-up)

**Constraint.h / QpSolver.h**
- `Bounds` (unbounded sentinels, feasibility, violation, projection);
  `NoConstraint`; `ValueJacobianConstraint` (double `con(x, c, J)`);
  `ConstraintEvaluator`; `maxViolation`, `l1Penalty`
- `solveActiveSetQp`: dense primal active-set QP with equality flags and
  equalities/violated-inequalities working set, Gaussian-elimination KKT
  solves, multiplier-based dropping; box bounds enter as ordinary rows;
  linearly dependent active rows (flat arbitrage curves) are dropped
  automatically

**SLSQP.h**
- `SLSQP<DoubleT>`: damped BFGS Lagrangian Hessian, QP subproblem over
  linearized constraints and bounds, L1 exact-penalty merit search with
  NLopt's per-row multiplier update `mu <- max(|lam|, (mu+|lam|)/2)`,
  convergence when the QP step vanishes at a feasible point (KKT)
- NLopt contract: the start point must satisfy the box bounds (throws
  otherwise); line-search failure maps to `RoundoffLimited` (NLopt mode 8)
- Exact gradients/Jacobians per backend; `minimize(f, ineq[, eq], bounds, x)`
  plus the inherited unconstrained entry

**AugLag.h**
- `AugLag<DoubleT>`: Rockafellar augmented Lagrangian over an inner L-BFGS
- Explicit duals for constraints and box bounds, smoothed plus-function
  (`O(1e-12)` bias), multiplier updates, Birgin-Martinez rho0 heuristic and
  NLopt's complementarity-aware ICM escalation (`ICM > 0.5 * prev`)
- Convergence at `violation <= 1e-9` and projected KKT `<= 1e-6`, with an
  SLSQP feasibility polish to tighten the point exactly (the AL converges
  only linearly near the boundary)

**ImplicitFunction.h (IFT layer, P4.8)**
- First-order sensitivities of optima w.r.t. market data: `iftUnconstrained`
  (`dp/dm = -H^{-1} G`, dense H from n exact HVPs, mixed `L_xm` from
  forward-over-reverse seeds on m) and `iftKkt` (the §6.2 square system with
  FullPivLU and a minimum-norm fallback for degenerate active sets, rank and
  condition reported)
- SLSQP/AUGLAG export final multipliers in `OptimizerState`;
  active set = `|g| <= 1e-8` AND `lambda > 1e-10`, active bounds as exact
  linear rows; v1 scopes `dg/dm = dh/dm = 0` (constraints on the model
  surface only)
- `minimizeDifferential` (solve + IFT in one call) and
  `minimizeDifferentialVar` (optimum as callback vars on the caller's tape,
  adjoints flow into m through `dp/dm`)
- Unconstrained solves use LLT with a minimal ridge (eigen gap below zero);
  the eigendecomposition reports the condition number

Truncated Newton (`TNewton.h`), the constrained stack (P4.5–P4.7) and the
first-order IFT layer (P4.8) are implemented; second-order IFT (P4.10) is
staged for v1.5.

### Autodiff

**Hvp.h**
- `Math::hvp(f, x, v)`: forward-over-reverse Hessian-vector product
  `f''(x)·v` for scalar-generic objectives

**PrimalExtraction.h** (`Autodiff/`)
- Shared recursive primal extraction used by the integrator and solver AD
  dispatch (`double → var → fvar<var> → ...`)

### Random

**PCGRandom.hpp / PCGExtras.hpp / PCGUint128.hpp**
- PCG family generators (pcg32/pcg64 and friends, `mc::` namespace)

**ZigguratNormal.h / McFarlandNormal.h**
- Marsaglia ziggurat and McFarland modified-ziggurat normal samplers,
  templated on the underlying uniform generator

**Sobol/**
- Sobol' low-discrepancy sequences (`DirectionNumbers`, `GF2`, `CBCSearch`)

## Performance Comparison

From `test_math.cpp` results:

### Integration: ∫₀¹ x² dx = 1/3

| Method | Error | Evaluations |
|--------|-------|-------------|
| Trapezoid (Default) | 2.5×10⁻⁹ | 8,193 |
| Trapezoid (MidPoint) | 2.1×10⁻⁹ | 129,140,164 |
| **Simpson** | **0 (exact)** | **65** |
| Gauss-Legendre (order 2) | 5.6×10⁻¹⁷ | 2 |
| Gauss-Legendre (order 20) | 5.6×10⁻¹⁷ | 20 |

**Winner:** Simpson's rule provides exact results with minimal evaluations for polynomial integrands.

### Root Finding: x² - 2 = 0 (√2)

| Method | Error | Convergence |
|--------|-------|-------------|
| Bisection | 7.6×10⁻¹¹ | Linear |
| Secant | 2.2×10⁻¹⁵ | ~1.618 |
| **Newton** | **0 (exact)** | **Quadratic** |
| Brent | 1.1×10⁻¹¹ | Super-linear |
| Ridder | 8.9×10⁻¹⁶ | ~1.839 |
| False Position | 2.2×10⁻¹⁵ | Super-linear |

**Winner:** Newton with exact derivatives achieves machine precision instantly.

## Testing

All test targets must exit `0`:

```bash
cmake --build build --target test_math test_solvers test_interpolation test_markets \
    test_ad test_integrator_primitives test_solvers_ad test_interpolation_xad \
    test_autodiff_primitives test_cubic_weights test_bicubic_weights test_tanh_sinh \
    test_optimization test_optimizers_stress test_optimizers_bench \
    test_ziggurat test_mcfarland -j 8

for t in test_math test_solvers test_interpolation test_markets test_ad \
         test_integrator_primitives test_solvers_ad test_interpolation_xad \
         test_autodiff_primitives test_cubic_weights test_bicubic_weights test_tanh_sinh \
         test_optimization test_optimizers_stress test_ift test_ziggurat test_mcfarland; do
    ./build/$t || echo "FAILED: $t"
done
```

| Target | Validates |
|--------|-----------|
| `test_math` | Integrators + solvers at double precision |
| `test_solvers` | Golden roots, error handling, Brent interpolation regression (evaluation counts) |
| `test_interpolation` | Interpolation values |
| `test_markets` | Curves / volatility construction |
| `test_ad` | End-to-end AD (interpolation, integration, markets) at `var` |
| `test_integrator_primitives` | All integrators at double/`var`/`fvar<var>`: value, grad, Hessian, eval-count parity, AD bounds |
| `test_solvers_ad` | `var` IFT gradients (all solvers), `fvar` pathwise Hessians, auto-bracketing, composites |
| `test_interpolation_xad` | Evaluation-point AD: `dI/dx`, mixed `d²I/dxdy`, cubic mixed Hessians vs FD, `evaluateFixed` contract |
| `test_cubic_weights` / `test_bicubic_weights` | Cubic/bicubic AD dispatch: gradients vs FD, true Hessians vs second-order FD, weight-matrix fast paths |
| `test_autodiff_primitives` | HVP, `solve(interp)`, `integrate(interp)`, nested solves — analytic + FD cross-checks |
| `test_optimization` | Optimizer stack: stop criteria, CRTP base, exact `valueGrad`/`hvp`/constraint Jacobians, Wolfe line search, L-BFGS, QP solver, SLSQP/AUGLAG on equality/inequality/bounds/arbitrage fixtures |
| `test_ift` | IFT layer: unconstrained/KKT `dp/dm` vs analytic and bump-and-recalibrate FD, multiplier sensitivities, degenerate active sets, ridge escalation, condition reporting, var composition |
| `test_optimizers_stress` | Hard problems (Rosenbrock n=2/10/50, Beale, Himmelblau, 1e8/1e16-conditioned quadratics), degenerate/redundant constraints, 100-bounds, 20-point arbitrage curve, NLopt-parity checks |
| `test_optimizers_bench` | Timing harness (µs/run per method) for the sampling profiler |
| `test_ziggurat` / `test_mcfarland` | Ziggurat and McFarland normal samplers: correctness and throughput |
| `test_tanh_sinh` | Tanh-Sinh quadrature value + AD gradients/Hessians |

## Integration with Stan Math

1. Include `<stan/math.hpp>` (plus `<stan/math/mix.hpp>` for `fvar`) and the
   single AD umbrella:
   ```cpp
   #include <stan/math.hpp>
   #include <stan/math/mix.hpp>
   #include "Math/StanPrimitives.h"
   ```
   It pulls in every dispatch layer:
   - `Math/Integrals/IntegratorStanPrimitives.h`
   - `Math/Solvers/SolverStanPrimitives.h`
   - `Math/Interpolations/InterpolationStanPrimitives.h`
   - `Math/Optimization/OptimizerStanPrimitives.h`
   - `Math/Autodiff/Hvp.h` (Hessian-vector products)
   Individual primitives headers remain available for fine-grained includes.
2. `Math/NumericalMethods.h` (and the component umbrellas `Integrals.h`,
   `Solvers.h`, `Interpolations.h`, `Optimization.h`) stay Stan-free, so
   double-only translation units never pull the Stan include paths. Random
   utilities (`Math/Random/`) are also separate.
3. Link TBB + SUNDIALS and set the Stan defines
   (`STAN_THREADS=true`, `STAN_NO_RANGE_CHECKS`, `_REENTRANT`,
   `TBB_INTERFACE_NEW`) — see the CMake targets `test_solvers_ad`,
   `test_interpolation_xad` and `test_autodiff_primitives` for a working
   template.

## Future Extensions

Roadmap for the AD engine (grow on the existing primitives):

### Optimizers
- L-BFGS with AD gradients (one reverse pass per objective evaluation)
- Newton-CG / trust-region using `Math::hvp`
- Differentiable optimization: implicit sensitivity layer
  `dp̂/dm = −H⁻¹ ∂²L/∂p∂m` for calibrated parameters

### Pricing / calibration
- Heston COS/Fourier pricer on the integrator primitives + implied-vol
  inversion through the solver IFT
- Calibration risk: `d(calibrated params)/d(market vols)` in one graph
- Sparse AAD sensitivity assembly across curves → surfaces → payoffs

### Integrals / Solvers
- Gauss-Kronrod, Romberg, Filon (oscillatory) integration
- Halley, Muller, Steffensen methods
- Exact second-order IFT solver sensitivities (today: `var` first order,
  `fvar` pathwise for smooth solvers)

## References

- [QuantLib](https://www.quantlib.org/) - Design inspiration
- [Stan Math](https://mc-stan.org/math/) - Automatic differentiation framework
- Press et al., "Numerical Recipes in C" (2nd ed.) - Algorithm implementations
- Golub & Welsch (1986), "Gauss quadratures and orthogonal polynomials"
- Brent (1973), "Algorithms for Minimization Without Derivatives"

## License

Part of the generic_MC project. See top-level LICENSE file.
