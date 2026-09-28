# Quantape

C++20 library for Monte Carlo and derivatives pricing, built around one idea:
**every numerical primitive is AD-compatible and composes**. Integrators,
solvers and interpolators work with `double`, `stan::math::var` (exact
gradients) and `stan::math::fvar<...>` (Hessians / Hessian-vector products),
selected automatically by the scalar template parameter.

## Layout

```
include/quantape/      public headers (namespace quantape::)
    math/              integrators, solvers, interpolation, optimization, AD, RNG
    markets/           curves (yield, IR, survival), volatility (IR, EQ, FX), mostly a stub for now
    mc/                SDE simulation engine + pathwise AD (gradients, sensitivities); mc/processes/ = SDE process bundles
    models/            models (process + pricing/calibration mathematics; e.g. HestonModel)
    payoffs/           payoff helpers (smoothed indicators)
    pricing/           pricers (e.g. Black-Scholes) and AD primitives
    scenario/          Monte Carlo scenario machinery
src/                   implementation files (model/simulator .cpp)
tests/                 correctness gates (exit code 0)
benchmarks/            timing harnesses (not correctness gates); `bench_sde` covers the SDE engine and every AD mode
examples/              usage demos
docs/                  Doxyfile + docs/index.html entry point (generated HTML gitignored)
scripts/               format_code.sh
```

All headers live under one include root: consumers include
`"quantape/math/..."`, `"quantape/pricing/..."`, etc. The AD single-include is
`quantape/math/StanMath.h` (plugin-safe stan + Eigen ordering); the Stan-free
umbrella is `quantape/math/NumericalMethods.h`, and the AD umbrella is
`quantape/math/StanPrimitives.h`.

## Components

| Module | Contents                                                                                                                                                                                                                                                                                                                                                                                             |
|---|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `quantape/math/` | Integrators (Trapezoid, Simpson, Gauss-Lobatto, Gauss-Legendre, Tanh-Sinh), 1-D solvers (Bisection, Brent, Secant, Ridder, False Position, Newton), tridiagonal solver, interpolators (Linear, Log-Linear, Cubic, Bilinear, Bicubic), HVP utility, RNG (`Random/`: PCG, ziggurat, McFarland, Sobol)                                                                                                  |
| `quantape/math/Optimization/` | Optimizers: result codes, stop criteria, `Optimizer<DoubleT, Impl>` base, exact AD gradient / HVP / constraint-Jacobian helpers, strong-Wolfe line search, L-BFGS (AD or AD-free double `f(x, grad)` mode), truncated Newton (exact HVP), dense active-set QP, SLSQP and augmented Lagrangian with final-multiplier export |
| `quantape/calibration/` | Calibration use of the optimizer stack: scalar-generic `CalibrationProblem` concept (double / `var` / `fvar<var>` residuals through `make_callback_var` pricing), assembled LS derivatives + `calibrationIft` (db/da, dlambda/da, calibrator-agnostic multiplier recovery), `CalibrationChain` (instrument-Jacobian route, `propagateMarketRisks`) and the AD twins `CalibrationChainKkt` / `ImplicitFunction` (minimizeDifferential, iftKkt), plus the Heston adapter (`HestonCalibration.h`) |
| `quantape/math/*/StanPrimitives.h` | AD dispatch layers: one-pass weighted rule extraction for integrals, implicit-function-theorem gradients for roots, AD-weight/fast-path interpolation, forward-over-reverse HVP                                                                                                                                                                                                                      |
| `quantape/markets/` | Curves (yield, IR, survival), volatility (IR, EQ, FX), currently a stub                                                                                                                                                                                                                                                                                                                              |
| `quantape/processes/` | Stochastic models and bridge samplers                                                                                                                                                                                                                                                                                                                                                                |
| `quantape/models/` | Model layer: characteristic functions and analytic pricers with instrument gradients (`HestonModel.h`: Gatheral CF, Lewis + control variate, complex-step Jacobians) |
| `quantape/mc/` | SDE simulation engine: time grids, piecewise-constant parameters, blocked/streamed simulation, TBB schedules, keyed reproducible Gaussian/uniform draws, **Sobol/QMC source** (jump-ahead points, digital-shift replicas, layouts documented in `SobolSource.h`), schemes (Euler, predictor-corrector, Milstein, moment-matching QE), Monte Carlo estimator; pathwise AD: **Stan-backed defaults** (reverse `Gradients.h`, forward `ForwardStan.h`, state derivatives `StateDerivatives.h`), opt-in accelerated scalar types under `mc/mcfwdrev/` (fixed-size forward dual, lean reverse tape), checkpointed gluing for long horizons, multiprocessing sharding (bitwise-mergeable samples) |
| `quantape/payoffs/` | Payoff helpers: branchless smoothed indicators for pathwise-differentiable digital/barrier payoffs |
| `quantape/scenario/` | Monte Carlo scenario machinery                                                                                                                                                                                                                                                                                                                                                                       |
| `quantape/pricing/` | Pricers (e.g. Black-Scholes) and AD primitives                                                                                                                                                                                                                                                                                                                                                       |

## Build

```bash
cmake -S . -B build
cmake --build build --target <target> -j 8

# API documentation (Doxygen) -> build/docs/doxygen/html
cmake --build build --target doc
```

Requires C++20 (tested with Homebrew LLVM 20). Third-party dependencies
(Stan Math 4.9, Eigen, Boost, oneTBB, SUNDIALS) are fetched/configured by
CMake; see the top-level `CMakeLists.txt` for the exact versions. The static
library target is `quantape`; tests, benchmarks and examples link it
transitively.

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
| `test_optimization` | Optimizer stack: stop criteria, base class, exact gradients/HVPs/constraint Jacobians, Wolfe line search, L-BFGS, QP, SLSQP/AUGLAG constrained fixtures |
| `test_optimizers_stress` | Hard/edge problems: Rosenbrock n=50, Beale, Himmelblau, 1e8/1e16 conditioning, degenerate constraints, 20-point arbitrage curve, NLopt parity |
| `test_ift` | IFT sensitivities: `dp/dm` vs analytic + bump-and-recalibrate FD, KKT multiplier sensitivities, degenerate active sets, ridge escalation, condition reporting, var composition |
| `test_sde_simulator` | SDE core: grids, keyed reproducible draws, Euler moments (OU exact, GBM vs Black-Scholes), CIR, bitwise path/block equivalence, estimator |
| `test_sde_schemes` | Scheme convergence orders (Euler 1/2, Milstein 1, predictor-corrector 2), CIR positivity |
| `test_sde_moment_matching` | QE sampler: exact conditional moments, positivity across regimes, uniform-stream contract, CIR terminal moments |
| `test_sde_processes` | Process bundles (GBM/OU/CIR/Heston) and Heston QE: moments, reference prices, Feller-violating safety |
| `test_heston_analytic` | Heston model: trap-free CF, Lewis+BS/asymptotic CV pricer, EFGL vs Gauss-Legendre identity, low-order node stability, deterministic-variance limit, QE/Sobol martingale + cross-check, full 9-parameter complex-step gradient/Hessian vs FD, surrogate Taylor, Euler homogeneity |
| `test_heston_stan` | Heston Stan wiring: `make_callback_var` `var` adjoint == analytic gradient (1-node tape), `fvar<var>` Hessian == analytic Hessian (2-node tape), FD cross-checks |
| `test_heston_mc` | Theta-driven Heston QE (`theta = {mu, kappa, level, eta, rho}`): bitwise theta-vs-member path equivalence, QE+Sobol catalog vs analytic, pathwise `simulateGradient` greeks vs analytic + engine-FD cross-check |
| `test_heston_ift` | Generic calibration/IFT layer: KKT vs bump-recalibrate Richardsons, product-risk propagation, instrument-route cross-check, calibrator-agnostic multiplier recovery, Feller-active dlambda/da, a second model (Black-Scholes) through the same concept, and the Stan interop (`LBFGS<var>`, `minimizeDifferential`/`iftKkt` == analytic chain to 1e-13) |
| `test_heston_calibration` | 10-quote synthetic Heston calibration shoot-out: exact-gradient L-BFGS (price/vega/IV/log space), LM Gauss-Newton, exact-Newton, `LBFGS<var>` callback — noiseless recovery to 1e-9, noisy agreement, hard-start robustness, 9-parameter Jacobian condition/identifiability, model+r and model+S+r+q joint fits, and Feller-constrained AUGLAG vs SLSQP vs a sigma=sqrt(2 kappa theta) boundary reference (binding/inactive KKT multipliers) |
| `test_sde_qmc` | Sobol source contract and engine invariants (bitwise), QMC vs i.i.d. RMSE at equal N (one-step call, closed form), multi-step Euler moment gates, QE with Sobol uniforms, AD modes agreeing under QMC |
| `test_sde_gradients` | Pathwise AD: finite-difference wiring, GBM greeks vs Black-Scholes, exact-moment derivatives, forward/checkpointed/lean modes all gate-equal, state derivatives, smoothed digitals, shard merge, schedule determinism |
| `test_market_risk_ift` | Calibration chains: KKT vs instrument-Jacobian routes, bump-recalibrate, instrument exactness, end-to-end SDE gradient -> IFT |
| `test_ziggurat` / `test_mcfarland` | Ziggurat / McFarland normal samplers: correctness and throughput |
| `test_tanh_sinh` | Tanh-Sinh quadrature value + AD gradients/Hessians |

Benchmarks live in `benchmarks/` (`test_optimizers_bench`,
`test_bs_hessian_bench`, AD benches, exp/log bit-trick bench) — timing
harnesses for the sampling profiler, not correctness gates.

## AD architecture

Primitives are templated on the scalar and route by type at compile time.
`var` paths avoid differentiating algorithms where it is fragile: integrals
extract the converged quadrature rule in a double pass and attach exact
weighted gradients (one callback per output); roots solve at double precision
and attach `dx/dθ = −f_θ/f_x` via the implicit function theorem; interpolators
build weights as `AD` expressions so the query coordinate is differentiated by
default. `fvar<...>` keeps the generic pathwise route (exact for smooth
functions, all orders) and powers Hessians and HVPs. Control flow is always
primal-pinned; derivative semantics at branches are the one-sided values of
the selected branch. See `include/quantape/math/README.md` for details and
contracts.

The optimizer building blocks follow the same pattern: iteration state in
double, exact AD gradients and HVPs per step (no finite differences), final
multipliers exported by the constrained solvers, and first-order IFT/KKT
sensitivities of the optimum w.r.t. market data (`dp/dm`, multiplier
sensitivities, `var` composition on the caller's tape) — see
`internal_docs/ad_optimizers.md`.

The SDE engine extends the same discipline to simulation: keyed draws are
parameter-independent and never dualized (common random numbers are
bitwise), schemes and functors are scalar-generic, and every AD mode
differentiates the scheme exactly as coded. Stan-backed implementations are
the defaults and the reference — reverse (`mc/Gradients.h`), forward
(`mc/ForwardStan.h`), state derivatives (`mc/StateDerivatives.h`) — while
the accelerated self-contained scalar types under `mc/mcfwdrev/` (fixed-size
forward dual, lean reverse tape) are opt-in and gated in the tests against
the Stan modes. Estimators compose as data (per-path samples, ordered
reduction, bitwise parallel == sequential, shard-mergeable across
processes), and market risks propagate through the calibration via the IFT
layer — KKT route or weighted instrument-Jacobian pseudo-inverse
(`calibration/CalibrationChain.h`) — without re-calibrating.

## Documentation

Headers are comment-dense (`/** ... */` file/class/method docs). Generate the
API reference with `cmake --build build --target doc` (Doxygen + Graphviz,
`docs/Doxyfile`); the output lands in `docs/html` and is reachable through the
single entry point `docs/index.html` (open it in a browser). The generated
HTML is gitignored — regenerate after changes. Class graphs, collaboration
diagrams and call/caller graphs are enabled.
