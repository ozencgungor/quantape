# Quantape

C++20 library for Monte Carlo and derivatives pricing, built around one idea:
**every numerical primitive is AD-compatible and composes**. Integrators,
solvers and interpolators work with `double`, `stan::math::var` (exact
gradients) and `stan::math::fvar<...>` (Hessians / Hessian-vector products),
selected automatically by the scalar template parameter.

## Layout

```
include/quantape/      public headers (namespace quantape::)
    calibration/       calibration problems, IFT/KKT risk chains, model adapters
    datetime/          dates, calendars, schedules, day counts, time conversion
    instruments/       cashflow vocabulary and dated FX spot/forward/swap records
    math/              integrators, solvers, interpolation, optimization, AD, RNG, high-precision arithmetic, special functions
    markets/           multi-curve IR/FX stack: discount/forecast/spread/xccy curves, bootstrap, risk, config; volatility still a stub
    mc/                SDE simulation engine + pathwise AD (gradients, sensitivities); mc/processes/ = SDE process bundles
    models/            models (process + pricing/calibration mathematics; e.g. HestonModel)
    payoffs/           payoff helpers (smoothed indicators)
    pricing/           pricers (Black-Scholes, FX spot/forward/swap) and AD primitives
    scenario/          Monte Carlo scenario machinery
src/                   implementation files (model/simulator/curve/risk .cpp)
tests/                 correctness gates (exit code 0); 62 registered targets
benchmarks/            timing harnesses (not correctness gates); `bench_curve`, `bench_sde`, `bench_risk` and `bench_fx`
examples/              usage demos
docs/                  Doxyfile + docs/index.html entry point (generated HTML gitignored)
repo_tools/            repo-level scripts (format_code.sh, policy/commit-msg checks)
tools/                 offline tooling: Sobol extenders/refiners/verifiers, analyzers, table packer, market-data fetchers, AWS spot runner
```

All headers live under one include root: consumers include
`"quantape/math/..."`, `"quantape/pricing/..."`, etc. The AD single-include is
`quantape/math/StanMath.h` (plugin-safe stan + Eigen ordering); the Stan-free
umbrella is `quantape/math/NumericalMethods.h`, and the AD umbrella is
`quantape/math/StanPrimitives.h`.

## Components

| Module | Contents                                                                                                                                                                                                                                                                                                                                                                                             |
|---|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `quantape/math/` | Integrators (Trapezoid, Simpson, Gauss-Lobatto, Gauss-Legendre, Tanh-Sinh, Ooura double-exponential, exponentially fitted Gauss–Laguerre (tabular rule)), in-house double-double arithmetic (`Precision/`) and oscillatory special functions Si/Ci (`SpecialFunctions/`), 1-D solvers (Bisection, Brent, Secant, Ridder, False Position, Newton), tridiagonal solver, interpolators (Linear, Log-Linear, Cubic, Monotone cubic, Hyman-filtered spline, tension spline, Bilinear, Bicubic), HVP utility, RNG (`Random/`: PCG, ziggurat, McFarland, Sobol generator + quality)                                                                                                  |
| `quantape/math/Optimization/` | Optimizers: result codes, stop criteria, `Optimizer<DoubleT, Impl>` base, exact AD gradient / HVP / constraint-Jacobian helpers, strong-Wolfe line search, L-BFGS (AD or AD-free double `f(x, grad)` mode), truncated Newton (exact HVP), dense active-set QP, SLSQP and augmented Lagrangian with final-multiplier export |
| `quantape/calibration/` | Calibration use of the optimizer stack: scalar-generic `CalibrationProblem` concept (double / `var` / `fvar<var>` residuals through `make_callback_var` pricing), assembled LS derivatives + `calibrationIft` (db/da, dlambda/da, calibrator-agnostic multiplier recovery), `CalibrationChain` (instrument-Jacobian route, `propagateMarketRisks`) and the AD twins `CalibrationChainKkt` / `ImplicitFunction` (minimizeDifferential, iftKkt), plus the Heston adapter (`HestonCalibration.h`) |
| `quantape/math/*/StanPrimitives.h` | AD dispatch layers: one-pass weighted rule extraction for integrals, implicit-function-theorem gradients for roots, AD-weight/fast-path interpolation, forward-over-reverse HVP                                                                                                                                                                                                                      |
| `quantape/instruments/` | Plain-data instrument vocabulary: scalar-templated `CashflowT<ScalarT>` with an inline currency, and dated FX `FxSpot` / `FxForward` / `FxSwap` records (`FxInstruments.h`) with settlement-lag factories, date resolution and cashflow materialization |
| `quantape/markets/` | Multi-curve IR/FX stack: `DiscountCurve` / `SpreadCurve` zero-node curves in zero/log-discount spaces with linear, Akima, tension-spline, monotone-cubic, Hyman-filtered and mixed schemes; sequential exact-fit bootstrap of deposits/repos/FRAs/exchange-traded futures (simple, compounded and averaged RFR styles)/par OIS/IRS/basis pillars, escalating to a whole-grid fixed point for off-node cashflows and stencil schemes; shifted-lognormal FRA and Hull-White futures convexity; synthetic money-market index quotes from the overnight curve plus integrated basis (cross-tenor strips) and quote-strip extrapolation; turn-of-year overlay (persistent turns and funding bumps); xccy basis bootstrap (constant and MtM-resetting notionals) and FX forward-points/collateral bootstrap; external JSON config (document schema in `CurveConfig.h`) with `buildStack` parent/discount/forecast trees; analytic quote deltas, Gauss-Newton quote gamma and exact (bootstrap-curvature corrected) stack gamma, stack/tree risk with xccy rows, factor and turn buckets, risk reports and hedge solve; volatility (IR, EQ, FX) still a stub |
| `quantape/mc/processes/` | SDE process bundles (GBM, OU, CIR, Heston; Heston QE moment-matching)                                                                                                                                                                                                                                                                                                                                                                |
| `quantape/models/` | Model layer: characteristic functions and analytic pricers with instrument gradients (`HestonModel.h`: Gatheral CF, Lewis + Black–Scholes/asymptotic control variates, EFGL quadrature, full 9-parameter complex-step gradient/Hessian, quote surrogates, single-node callback-var AD prices in `HestonStanPrimitives.h`); no FX models/vol yet |
| `quantape/mc/` | SDE simulation engine: time grids, piecewise-constant parameters, blocked/streamed simulation, TBB schedules, keyed reproducible Gaussian/uniform draws, **Sobol/QMC source** (jump-ahead points, digital-shift replicas, layouts documented in `SobolSource.h`), schemes (Euler, predictor-corrector, Milstein, moment-matching QE), Monte Carlo estimator; pathwise AD: **Stan-backed defaults** (reverse `Gradients.h`, forward `ForwardStan.h`, state derivatives `StateDerivatives.h`), opt-in accelerated scalar types under `mc/mcfwdrev/` (fixed-size forward dual, lean reverse tape), checkpointed gluing for long horizons, multiprocessing sharding (bitwise-mergeable samples) |
| `quantape/payoffs/` | Payoff helpers: branchless smoothed indicators for pathwise-differentiable digital/barrier payoffs |
| `quantape/scenario/` | Monte Carlo scenario machinery                                                                                                                                                                                                                                                                                                                                                                       |
| `quantape/pricing/` | Pricers: Black-Scholes and FX spot/forward/swap (`Fx.h`, scalar-templated with analytic or AD spot greeks) plus AD primitives                                                                                                                                                                                                                                                                        |

FX is implemented for delta-one products: spot/forward/swap quotes and
conventions, dated instruments with materialized settlement cashflows,
date-aware AD pricers and spot greeks, forward-point/collateral discount
bootstrap, MtM-resetting cross-currency basis and the FX risk/greeks side
table. FX volatility and option models are not implemented yet; the
`CurveRole::FxVol` role and the volatility modules are reserved for that later
work.

## Build

```bash
# Configured presets: debug (-O0 -g3, benchmarks off), release (-O3, benchmarks on)
cmake --preset release
cmake --build --preset release

# Plain configure also works
cmake -S . -B build
cmake --build build --target <target> -j 2

# Optional: compile-time Sobol table (text or .qsb); packed to a binary asset
# automatically and exposed via SobolGenerator::fromDefaultTable
cmake -S . -B build -DQUANTAPE_SOBOL_TABLE=/path/to/joe-kuo-table.txt

# clang-format the tree, or verify it (wrapper: repo_tools/format_code.sh)
cmake --build build --target format
cmake --build build --target format-check

# API documentation (Doxygen) -> build/docs/doxygen/html
cmake --build build --target doc
```

Requires C++20 (tested with Homebrew LLVM 20). Third-party dependencies
(Stan Math 4.9, Eigen, Boost, oneTBB, SUNDIALS) are fetched/configured by
CMake; see the top-level `CMakeLists.txt` for the exact versions. The static
library target is `quantape`; tests, benchmarks and examples link it
transitively.

## Tests

62 correctness gates are registered with CTest; every gate must exit `0`. Run
one target directly, or the whole suite through CTest (presets build into
`build/debug` and `build/release`).

```bash
./build/release/<target>  # single correctness gate
ctest --preset release    # all registered tests
ctest --test-dir build    # plain configure tree
```

| Target | Validates |
|---|---|
| `test_math` | Integrators + solvers at double precision |
| `test_solvers` | Solver golden roots, error handling, Brent interpolation regression (evaluation counts) |
| `test_interpolation` | Interpolation values at double precision |
| `test_math_utils` | Shared math utilities and interpolation/tridiagonal AD paths (`var` / `fvar<var>`) |
| `test_markets` | Discount/survival curve construction and shapes |
| `test_curve` | Curve construction/evaluation: interpolation spaces × schemes, analytic node weights vs FD, extrapolation, exact fit |
| `test_bootstrap_validation` | Cross-scheme bootstrap invariants: exact fit, off-node fixed point, positivity/arbitrage, stubs and negative rates |
| `test_curve_config` | JSON curve config: parsing/validation, parent/discount/forecast trees, xccy and FX spots, error cases |
| `test_instrument_options` | Instrument options: shifted-lognormal FRA convexity, seasoned first-fixed coupons, futures adjustment edges |
| `test_forecast_pillars` | Forecast curves: par IRS pillars with exogenous discounting, IRBS conventions, stubs and overlapping fixings |
| `test_futures` | Rate futures: pricing, simple/compounded/averaged RFR styles, convexity, bootstrap and risk |
| `test_synthetic_quotes` | Synthetic index deposit/FRA quotes from the OIS curve plus integrated basis, and quote extrapolation |
| `test_instruments` | Instruments vocabulary: `Cashflow`, bootstrap-instrument concept on double/`var`/`fvar<var>`, bitwise bootstrap gate |
| `test_curve_ad` | Curve-stack AD: evaluation, solver-derived bootstrap sensitivities (`var`/`fvar<var>`), turn overlays |
| `test_curve_risk` | Analytic quote deltas/gamma, stack/tree risk, turn overlay risk, dual pricing/risk scheme, buckets vs FD |
| `test_curve_risk_report` | Risk reports: kind/end/year buckets, hedge QR conditioning and the hedge solve |
| `test_xccy` | Xccy basis bootstrap (constant and MtM resetting notionals), analytic rows vs FD, coupled re-solve |
| `test_hull_white_convexity` | Hull-White futures convexity against the affine bond-algebra reference |
| `test_fx_quote` | FX quote envelopes, point/outright conversion, annualized premium, joint-calendar spot settlement |
| `test_fx_pricing` | FX pricers (double/`var`/`fvar<var>`), reset-aware xccy rows, FX stack risk and greeks |
| `test_fx_instruments` | Dated FX instruments: settlement lags, cashflow materialization, CIP and PV gates |
| `test_levenberg_marquardt_ad` | Levenberg-Marquardt optimizer with AD residuals and IFT polish |
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
| `test_black76` | Black-76 / GBS analytical Greeks (1st and 2nd order) vs finite differences and Stan AD |
| `test_mixed_hessian` | Mixed 1st/2nd-order AD through a pricing chain (analytical Greeks where available, AD fallback): mixed Hessian blocks vs FD |
| `test_pricer_hessian` | Pricer-hierarchy Hessian architecture: nested analytical AD paths per product vs FD |
| `test_precision` | In-house double-double (~32 digits): error-free arithmetic, sqrt/exp/log/sin/cos, Si anchors, Ooura double-exponential oscillatory integrals vs closed forms |
| `test_sobol_generator` | Sobol generator: dimension-1 semantics, digital-net exactness, digital shifts/replicas, 32/64-bit storage parity, jump-ahead (gray order), thread equivalence, mmap asset vs text table, engine integration |
| `test_sobol_quality` | Sobol table quality battery: exactness, discrepancy and integration criteria |
| `test_sobol_tools` | Table tooling: extend/refine/verify round-trips and failure modes |
| `test_date` / `test_daycount` / `test_calendar` / `test_schedule` | Datetime layer: dates, day counts, calendars and schedule generation |
| `test_logging` / `test_number_format` / `test_util` | Logging facade, number formatting and shared check/numeric helpers |

Benchmarks live in `benchmarks/` (`bench_curve`, `bench_sde`, `bench_risk`,
`bench_fx`, `bench_sobol`, `bench_daycount`, `test_optimizers_bench`,
`test_bs_hessian_bench`, and the Stan/AD micro-benchmarks `test_bs_ad_bench`,
`test_bvn_ad_bench`, `test_interp_ad_bench`, `test_fast_exp_log`) — timing
harnesses for the sampling profiler, not correctness gates. `bench_curve`,
`bench_sde`, `bench_risk` and `bench_fx` report p50/p99 wall time and have
allocation-tracking twins (`bench_sde_alloc`, `bench_risk_alloc`,
`bench_fx_alloc`).

## Tools

`tools/` holds offline utilities (not correctness gates): the Sobol table
pipeline (`extend_sobol`, `refine_sobol`, `verify_sobol`, `analyze_sobol`,
`sobol_to_binary`), market-data collection under `tools/market_data/` (CBOE /
Tradier / ORATS option chains, swaption volumes, offline replay) and the AWS
spot runner `tools/aws/run_spot.sh` with its bundle builder. Run logs,
collected data and python bytecode are gitignored.

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

Curve evaluation follows the same rule: `DiscountCurve` and `SpreadCurve`
instantiate at `double`, `var` and `fvar<var>`, and every bootstrap instrument
models the scalar-generic `BootstrapInstrument` concept
(`impliedQuote<ScalarT>`), so curve values, quote residuals and solver-derived
sensitivities (through the scalar-generic Brent solver and its
implicit-function-theorem polish) compose on the caller's tape. The production
bootstrap, quote-Jacobian transforms, stack/tree risk, xccy rows, risk reports
and hedge solve are double-only paths, gated at double precision.

The optimizer building blocks follow the same pattern: iteration state in
double, exact AD gradients and HVPs per step (no finite differences), final
multipliers exported by the constrained solvers, and first-order IFT/KKT
sensitivities of the optimum w.r.t. market data (`dp/dm`, multiplier
sensitivities, `var` composition on the caller's tape); the headers under
`calibration/` and `math/Optimization/` carry the contracts.

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
