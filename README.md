# generic_MC

C++20 library for Monte Carlo and derivatives pricing, built around one idea:
**every numerical primitive is AD-compatible and composes**. Integrators,
solvers and interpolators work with `double`, `stan::math::var` (exact
gradients) and `stan::math::fvar<...>` (Hessians / Hessian-vector products),
selected automatically by the scalar template parameter.

## Components

| Directory | Contents |
|---|---|
| `Math/` | Integrators (Trapezoid, Simpson, Gauss-Lobatto, Gauss-Legendre, Tanh-Sinh), 1-D solvers (Bisection, Brent, Secant, Ridder, False Position, Newton), tridiagonal solver, interpolators (Linear, Log-Linear, Cubic, Bilinear, Bicubic), HVP utility, RNG (`Random/`: PCG, ziggurat, McFarland, Sobol) |
| `Math/Optimization/` | Optimizer stack: result codes, stop criteria, CRTP `Optimizer<DoubleT, Impl>` base, exact AD gradient / HVP / constraint-Jacobian helpers, strong-Wolfe line search, L-BFGS (AD or AD-free double `f(x, grad)` mode), truncated Newton (exact HVP), dense active-set QP, SLSQP and augmented Lagrangian with final-multiplier export, and the first-order IFT layer (dp/dm, KKT sensitivities, var composition — `docs/ad_optimizers.md`) |
| `Math/*/*StanPrimitives.h` | AD dispatch layers: one-pass weighted rule extraction for integrals, implicit-function-theorem gradients for roots, AD-weight/fast-path interpolation, forward-over-reverse HVP (single include: `Math/StanPrimitives.h`; `Math/NumericalMethods.h` is the Stan-free umbrella for the rest) |
| `Markets/` | Curves (yield, IR, survival), volatility (IR, EQ, FX) |
| `Models/` | Stochastic models and bridge samplers |
| `ScenarioGeneration/` | Monte Carlo scenario machinery |
| `Pricing/` | Pricers (e.g. Black-Scholes) and AD primitives |

## Build

```bash
cmake -S . -B build
cmake --build build --target <target> -j 8
```

Requires C++20 (tested with Homebrew LLVM 20). Third-party dependencies
(Stan Math 4.9, Eigen, Boost, oneTBB, SUNDIALS) are fetched/configured by
CMake; see the top-level `CMakeLists.txt` for the exact versions.

## Tests

Run a target and check the exit code; all of the following must exit `0`.

```bash
./build/<target>
```

| Target | Validates |
|---|---|
| `test_math` | Integrators + solvers at double precision |
| `test_solvers` | Solver golden roots, error handling, Brent interpolation regression (evaluation counts) |
| `test_interpolation` | Interpolation values at double precision |
| `test_markets` | Curves / volatility construction and shapes |
| `test_ad` | End-to-end AD: interpolation, integration, markets at `var` |
| `test_integrator_primitives` | All integrators at double/`var`/`fvar<var>`: value, grad, Hessian, evaluation-count parity, AD bounds |
| `test_solvers_ad` | `var` IFT gradients for every solver (incl. bisection), `fvar` pathwise Hessians, var-only objectives, auto-bracketing, composites with interpolation |
| `test_interpolation_xad` | Evaluation-point AD: analytic `dI/dx`, mixed `d²I/dxdy` blocks, cubic mixed Hessians vs finite differences, `evaluateFixed` contract |
| `test_cubic_weights` / `test_bicubic_weights` | Cubic/bicubic AD dispatch: gradients vs FD, true active-branch Hessians vs second-order FD, weight-matrix fast paths |
| `test_autodiff_primitives` | Hessian-vector products, `solve(interp)`, `integrate(interp)`, nested solves — all with analytic references and FD cross-checks |
| `test_optimization` | Optimizer stack: stop criteria, CRTP base, exact gradients/HVPs/constraint Jacobians, Wolfe line search, L-BFGS, QP, SLSQP/AUGLAG constrained fixtures |
| `test_optimizers_stress` | Hard/edge problems: Rosenbrock n=50, Beale, Himmelblau, 1e8/1e16 conditioning, degenerate constraints, 20-point arbitrage curve, NLopt parity |
| `test_optimizers_bench` | Timing harness for the sampling profiler |
| `test_ift` | IFT sensitivities: `dp/dm` vs analytic + bump-and-recalibrate FD, KKT multiplier sensitivities, degenerate active sets, ridge escalation, condition reporting, var composition |
| `test_ziggurat` / `test_mcfarland` | Ziggurat / McFarland normal samplers: correctness and throughput |
| `test_tanh_sinh` | Tanh-Sinh quadrature value + AD gradients/Hessians |

## AD architecture in one paragraph

Primitives are templated on the scalar and route by type at compile time.
`var` paths avoid differentiating algorithms where it is fragile: integrals
extract the converged quadrature rule in a double pass and attach exact
weighted gradients (one callback per output); roots solve at double precision
and attach `dx/dθ = −f_θ/f_x` via the implicit function theorem; interpolators
build weights as `AD` expressions so the query coordinate is differentiated by
default. `fvar<...>` keeps the generic pathwise route (exact for smooth
functions, all orders) and powers Hessians and HVPs. Control flow is always
primal-pinned; derivative semantics at branches are the one-sided values of
the selected branch. See `Math/README.md` for details and contracts.

The optimizer building blocks follow the same pattern: iteration state in
double, exact AD gradients and HVPs per step (no finite differences), final
multipliers exported by the constrained solvers, and first-order IFT/KKT
sensitivities of the optimum w.r.t. market data (`dp/dm`, multiplier
sensitivities, `var` composition on the caller's tape) — see
`docs/ad_optimizers.md`.
