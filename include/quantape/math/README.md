# quantape/math — numerical primitives

The mathematical core of Quantape: integrators, 1-D root finders, interpolators,
optimizers and RNGs. Every primitive is templated on the scalar type, so the
same code runs in `double`, `stan::math::var` (exact gradients) and
`stan::math::fvar<...>` (Hessians / Hessian-vector products).

Full reference: the headers themselves are the documentation (comment-dense),
rendered by Doxygen — `cmake --build build --target docs`, then open
`docs/index.html`.

## Includes

| Header | What |
|---|---|
| `quantape/math/NumericalMethods.h` | Stan-free umbrella (integrals, solvers, interpolation, optimization) |
| `quantape/math/StanMath.h` | Single Stan include (plugin-safe stan + Eigen ordering) |
| `quantape/math/StanPrimitives.h` | AD dispatch layers for all primitives |

## Examples

### Integrate

```cpp
#include "quantape/math/NumericalMethods.h"
using namespace quantape::math;

auto f = [](double x) { return x * x; };
TrapezoidIntegratorDefault<double> trap(1e-8, 1000);
double I = trap(f, 0.0, 1.0);                       // 1/3
GaussLegendreIntegrator<double> gauss(20);          // fixed-order alternative
```

### Solve a root

```cpp
BrentSolver<double> brent;
double root = brent.solve([](double x) { return x * x - 2.0; }, 1e-10, 1.5, 0.0, 3.0);
// or with automatic bracketing: brent.solve(f, 1e-10, guess, step)
```

### Interpolate

```cpp
LinearInterpolation<double> lin({0.0, 1.0, 2.0}, {0.0, 1.0, 4.0});
double y = lin(0.5);
CubicInterpolation<double> cub(xs, ys);             // Spline (default), Akima, ...
BilinearInterpolation<double> bil(xs, ys, z_2d);    // 2-D tensor product
```

### Optimize

```cpp
LBFGS<stan::math::var> solver(StopCriteria{}, /*memory=*/10);
auto rosen = [](const std::vector<var>& x) { /* ... */ };
solver.minimize(rosen, x);                          // in-place, exact AD gradients
// constrained: SLSQP<var>, AugLag<var>; sensitivities: ImplicitFunction.h
```

### Differentiate through anything

```cpp
#include <stan/math.hpp>
#include "quantape/math/StanMath.h"
#include "quantape/math/StanPrimitives.h"
using stan::math::var;

var theta = 4.0;
BrentSolver<var> s;
var root = s.solve([&](const var& x) { return x * x - theta; }, 1e-10, var(1.5), var(0.0), var(3.0));
root.grad();                                        // theta.adj() = 1/(2*sqrt(theta))
```

`var` primitives attach exact derivatives without differentiating the
iteration (integrals: frozen-rule weighted callbacks; roots: implicit function
theorem; interpolators: AD weights; optimizers: exact gradients/HVPs and
IFT sensitivities). `fvar<...>` keeps the pathwise route for Hessians.
Control flow is primal-pinned — derivatives at branches are one-sided values
of the selected branch.

### Distributions

```cpp
#include "quantape/math/Distributions.h"
using namespace quantape::math;

Normal<double> n{0.3, 1.4};
double p = n.cdf(0.8);                       // pdf/cdf/ccdf/log forms/inv
auto gh = n.cdfGradHess(0.8);                // analytic g1/g2 (double path)

BivariateNormal<double> bvn{-0.5};           // Genz-style 1-D quadrature
double c = bvn.cdf(a, b);                    // + analytic bvnCdfGradHess

Gamma<double> g{2.3, 1.7};                   // Gamma, InverseGamma, ChiSquare,
                                             // NoncentralChiSquare, Beta,
                                             // StudentT, Lognormal, Poisson,
                                             // Exponential, InverseGaussian
```

Also: `NoncentralChiSquare` / `NoncentralBeta` / `NoncentralF`, and
`TrivariateNormal` (Genz method, with analytic gradient/Hessian packs).

All types follow the `DoubleT` (Black76) pattern: `double` uses the optimized
kernels and the analytic derivative packs; other scalars (Stan `var` today,
the in-house AD scalars later) run the generic scalar path. Every family with
complete first/second parameter partials also has Stan callback bridges: one
node for `var` and two nodes for `fvar<var>` (`stan::math::hessian`), built on
the uniform `makeVjpVar` / `makeVhpFvar` convention in
`Distributions/DistributionsStanPrimitives.h`; quantile derivative packs
(`invGradHess`) cover the inverse-quantile chain. Coverage, algorithm sources,
derivative tiers and accuracy gates: `internal_docs/math_distributions_plan.md`.

## Tests

Run `./build/<target>`; exit code 0 = pass. `test_math`, `test_solvers`,
`test_interpolation`, `test_ad`, `test_integrator_primitives`,
`test_solvers_ad`, `test_interpolation_xad`, `test_autodiff_primitives`,
`test_cubic_weights`, `test_bicubic_weights`, `test_tanh_sinh`,
`test_optimization`, `test_optimizers_stress`, `test_ift`, `test_ziggurat`,
`test_mcfarland`. Benchmarks live in `benchmarks/`.
