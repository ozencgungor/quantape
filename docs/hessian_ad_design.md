# Second-Order Sensitivities via `stan::math::hessian` — Design Document

## 1. Problem Statement

We have a pricing library with 100+ pricers, all templated on `DoubleT`. We want full second-order sensitivities (gamma, vanna, volga, cross-gammas) of any trade price with respect to all market data nodes — without modifying any pricer.

The market data nodes are the raw values that define curves, surfaces, and spot prices: each vol surface grid point, each discount factor node, each spot price. For a typical trade, this is $N \sim 100$–$500$ nodes. The Hessian $H_{ij} = \partial^2 P / \partial m_i \partial m_j$ is $N \times N$.

### 1.1 Why `stan::math::hessian`

| Approach | Changes to pricers | Cost | Accuracy | Hessian symmetry |
|----------|-------------------|------|----------|-----------------|
| Central FD | None | $O(N^2)$ pricings | $O(h^2)$, step-size sensitive | Exact by construction |
| Analytical Greeks + `grad()` | Every pricer returns `var` Greeks | $O(N)$ pricings | Exact | Exact |
| `stan::math::hessian` | **None** | $O(N)$ pricings | **Exact** | **Exact** |

`stan::math::hessian` uses forward-over-reverse mode (internally `fvar<var>`) to compute the full Hessian in $N$ reverse-mode sweeps. Since all pricers are already templated on `DoubleT`, they compile with `fvar<var>` without modification.

### 1.2 How `stan::math::hessian` Works Internally

```cpp
// Stan Math signature
template <typename F>
void hessian(const F& f,
             const Eigen::VectorXd& x,   // input: N market data values
             double& fx,                   // output: price
             Eigen::VectorXd& grad,        // output: N first-order sensitivities
             Eigen::MatrixXd& H);          // output: N×N Hessian

// f must implement:
//   template<typename T>
//   T operator()(const Eigen::Matrix<T, Dynamic, 1>& x) const;
```

**Internal mechanism**: `hessian` computes the Hessian one column at a time. For column $i$:

1. Constructs an input vector of type `fvar<var>`:
   - Each element $x_j$ becomes `fvar<var>(var(x_j), (j == i) ? 1.0 : 0.0)`
   - The `fvar` tangent is seeded in direction $e_i$ (forward mode)
   - The `var` part enables reverse mode

2. Calls `f(x_fvar_var)`, getting a result of type `fvar<var>`:
   - The `.val()` (a `var`) encodes the price and the full reverse-mode tape
   - The `.d_` (a `var`) encodes $\partial f / \partial x_i$ and its reverse-mode tape

3. Calls `grad()` on the `.d_` component:
   - This backpropagates through the tape of $\partial f / \partial x_i$
   - The adjoints give $\partial^2 f / \partial x_i \partial x_j$ for all $j$ — one Hessian column

4. From iteration $i = 0$ (the first column), also extracts:
   - $f(x)$ from `.val().val()`
   - $\nabla f$ from the `grad()` of `.val()`

**Total cost**: $N$ forward evaluations of `f`, each with one reverse pass. Each evaluation costs roughly $3$–$5\times$ a single `var`-only evaluation (the `fvar` wrapper adds overhead to every arithmetic operation).

### 1.3 `fvar<var>` Type Anatomy

Understanding the type helps debug issues:

```
fvar<var>
├── .val()     → var    (the function value, with reverse-mode tape)
│   ├── .val() → double (the actual numeric value)
│   └── .adj() → double (adjoint, populated after grad())
└── .d_        → var    (the directional derivative, with its own tape)
    ├── .val() → double (numeric value of the derivative)
    └── .adj() → double (second-order adjoint, populated after grad())
```

Every arithmetic operation on `fvar<var>` simultaneously:
- Computes the value (double)
- Records the operation on the `var` tape (for reverse mode)
- Propagates the forward-mode tangent (via `fvar` chain rule)
- Records the tangent propagation on the `var` tape (for second-order reverse)

This is why `fvar<var>` supports exact second derivatives: the forward mode differentiates once, the reverse mode differentiates again.

---

## 2. Architecture Overview

### 2.1 Existing Components

```
Generators (shared, double only)
│   Produce raw simulated values for each path/timepoint.
│   Owned centrally (unique_ptr), shared across trades via raw pointers.
│   Examples: EQDVolGenerator, EQDSpotGenerator, YieldCurveGenerator
│   Interface: getRawValues(path, timepoint) → vector<double>
│   These are NEVER templated on DoubleT. They produce doubles only.
│
ScenarioData<DoubleT> (per trade, lightweight)
│   Each trade has its own ScenarioData instance.
│   Holds pointers to only the generators this trade needs.
│   On query (getEQDVol, getRate, etc.):
│     1. Reads raw doubles from the generator for current path/timepoint
│     2. Constructs an ADTemplate instance (EQDVol<DoubleT>, IRCurve<DoubleT>, etc.)
│     3. Returns the ADTemplate to the pricer
│   The pricer never sees the generator — only the ADTemplate.
│
ADTemplates (market data primitives, templated on DoubleT)
│   EQDVol<DoubleT>:     2D vol surface, bicubic interpolation, returns DoubleT
│   EQDSpot<DoubleT>:    spot price, forward/discount calculations, returns DoubleT
│   IRCurve<DoubleT>:    yield curve, cubic spline interpolation, returns DoubleT
│   YieldCurve<DoubleT>: dividend/repo curve
│   FXRate<DoubleT>:     FX spot with optional curves
│   SurvivalCurve<DoubleT>: credit curve
│
│   Each ADTemplate:
│     - Stores market data values as DoubleT (the AD-sensitive quantities)
│     - Stores grid coordinates as double (strikes, maturities — not differentiated)
│     - Provides interpolation, forward rate, discount factor methods — all returning DoubleT
│     - Knows its sensitivity names (e.g., "risk_eqdvol_AAPL_1Y_100")
│     - Knows the layout of its nodes (for pack/unpack)
│
Pricers (templated on DoubleT, 100+ implementations)
│   template<typename DoubleT>
│   class PricerBase {
│       virtual DoubleT priceTrade(ScenarioData<DoubleT>& sd) = 0;
│   };
│
│   Pricers receive DoubleT values from ScenarioData (via ADTemplates) and
│   from helper functions that operate on those ADTemplates.
│   All arithmetic is in DoubleT. Return DoubleT price.
│   No pricer ever sees a generator, a raw double array, or AD internals.
```

### 2.2 Data Flow — Normal Pricing (double)

```
Generator::getRawValues(path, tp)          → std::vector<double>
    ↓
ScenarioData<double>::getEQDVol("AAPL")
    ↓ queries generator, gets doubles, constructs ADTemplate
EQDVol<double>(maturities, strikes, volSurf_as_double)
    ↓ returned to pricer
PricerA<double>::priceTrade(sd)
    ↓ calls sd.getEQDVol("AAPL").vol(T, K) → double
    ↓ calls sd.getEQDSpot("AAPL").forward(T) → double
    ↓ calls sd.getIRCurve("USD").discountFactor(T) → double
    ↓ performs pricing arithmetic in double
    → double price
```

### 2.3 Data Flow — Hessian (fvar\<var\>)

The pricer and `ScenarioData<double>` already exist and have been stepped to the current (path, timestep). The Hessian functor references the live pricer to capture its accumulated state via `rebind<T>()`, and uses the live ScenarioData's node mappings to construct a fresh `ScenarioData<T>`.

**Important**: `ScenarioData` has deleted copy constructors — we never copy a ScenarioData. The AD-typed `ScenarioData<T>` is constructed from scratch using the node mappings (structural metadata) from the live `ScenarioData<double>`.

```
sd.packMarketData()                        → Eigen::VectorXd x  (snapshot from generators)
    ↓
stan::math::hessian(functor, x, ...)       [internally, for each column i:]
    ↓ constructs fvar<var> vector from x, seeding tangent in direction e_i
    ↓ calls functor(x_fvar_var)
        ↓
        ScenarioData<fvar<var>> sd_ad(node_mappings);  // new, from mappings (NOT copied)
        sd_ad.loadFromVector(x_fvar_var);
            ↓ builds EQDVol<fvar<var>>, EQDSpot<fvar<var>>, IRCurve<fvar<var>>
            ↓ from the fvar<var> vector elements (these are the AD leaves)
        pricer.rebind<fvar<var>>()
            ↓ creates PricerA<fvar<var>> with state copied from live PricerA<double>
            ↓ accumulated DoubleT state (e.g., running averages) → value_of → T constant
            ↓ plain state (bools, ints, doubles) → copied directly
        pricer_ad.loadScenarioData(sd_ad)
        pricer_ad.priceTrade(sd_ad)
            ↓ calls sd_ad.getEQDVol("AAPL").vol(T, K) → fvar<var>
            ↓ calls sd_ad.getEQDSpot("AAPL").forward(T) → fvar<var>
            ↓ calls sd_ad.getIRCurve("USD").discountFactor(T) → fvar<var>
            ↓ all pricing arithmetic in fvar<var>
            → fvar<var> price
    ↓ stan::math::hessian extracts column i of the Hessian via grad()
    ↓ repeats for i = 0, ..., N-1
→ double price, Eigen::VectorXd grad, Eigen::MatrixXd Hessian
```

The pricer code is identical in both flows. The only difference is the type `DoubleT`.

### 2.4 Ownership and Lifetime Diagram

```
                  ┌─────────────────────────────┐
                  │     Generator Registry       │
                  │  (owns generators via        │
                  │   unique_ptr, long-lived)    │
                  │                              │
                  │  EQDVolGen("AAPL")  ─────┐  │
                  │  EQDSpotGen("AAPL") ────┐│  │
                  │  YCGen("USD")       ───┐││  │
                  └────────────────────────┼┼┼──┘
                                           │││
         ┌─────────────────────────────────┼┼┼──┐
         │  ScenarioData<double> (Trade A) │││  │
         │  (per-trade, lightweight)       │││  │
         │                                 │││  │
         │  m_generators: [               │││  │
         │    (EQDVolDesc, ptr) ───────────┘││  │
         │    (EQDSpotDesc, ptr) ────────────┘│  │
         │    (YCDesc, ptr) ──────────────────┘  │
         │  ]                                    │
         │  m_path = 42, m_timepoint = 7         │
         │                                       │
         │  getEQDVol("AAPL"):                   │
         │    gen->getRawValues(42, 7)            │
         │    → construct EQDVol<double>(...)     │
         │    → return to pricer                  │
         └───────────────────────────────────────┘

         ┌─────────────────────────────────────────────┐
         │  ScenarioData<fvar<var>> (for Hessian)       │
         │  (temporary, created inside functor from     │
         │   node_mappings — NOT copied from sd<double>)│
         │                                              │
         │  Copy ctor: DELETED                          │
         │  Constructed via: ScenarioData<T>(mappings)  │
         │  m_loadedFromVector = true                   │
         │  m_generators: [] (empty, not used)          │
         │                                              │
         │  m_eqdVols: {"AAPL" → EQDVol<fvar<var>>}    │
         │  m_eqdSpots: {"AAPL" → EQDSpot<fvar<var>>}  │
         │  m_irCurves: {"USD" → IRCurve<fvar<var>>}    │
         │                                              │
         │  getEQDVol("AAPL"):                          │
         │    return m_eqdVols.at("AAPL")               │
         └──────────────────────────────────────────────┘

         ┌──────────────────────────────────────────────┐
         │  PricerA<fvar<var>> (for Hessian)             │
         │  (created via pricer_double.rebind<T>())      │
         │                                               │
         │  Config: copied as-is (pure doubles/enums)    │
         │  DoubleT state: value_of → promoted to T      │
         │    e.g., running_avg: double 103.5 → T(103.5) │
         │    (constant on AD tape, zero tangent)         │
         │  Plain state: copied directly                  │
         │    e.g., barrier_breached: false                │
         │    e.g., fixing_count: 5                       │
         └───────────────────────────────────────────────┘
```

---

## 3. New Infrastructure

Only three pieces of new code are needed. No existing pricers, generators, or ADTemplates are modified.

### 3.1 Sensitivity Name Registry

Each ADTemplate descriptor knows its sensitivity names and node count. ScenarioData aggregates them into an ordered flat layout:

```cpp
struct NodeMapping {
    Descriptor descriptor;    // variant: EQDVolDescriptor | EQDSpotDescriptor | ...
    int offset;               // start index in flat vector
    int count;                // number of nodes for this descriptor
};

template<typename DoubleT>
class ScenarioData {
    // Existing
    std::vector<std::pair<Descriptor, Generator*>> m_generators;
    int m_path, m_timepoint;

    // New: ordered registry of all market data nodes for this trade
    std::vector<std::string> m_sensitivityNames;
    std::vector<NodeMapping> m_nodeMappings;

    // Build the registry. Called once after all generators are added.
    void buildSensitivityRegistry() {
        m_sensitivityNames.clear();
        m_nodeMappings.clear();
        int offset = 0;

        for (auto& [desc, gen] : m_generators) {
            NodeMapping nm;
            nm.descriptor = desc;
            nm.offset = offset;

            // Dispatch on descriptor type to get names
            auto names = std::visit([](const auto& d) {
                return d.sensitivityNames();
            }, desc);

            nm.count = static_cast<int>(names.size());
            m_sensitivityNames.insert(m_sensitivityNames.end(),
                                       names.begin(), names.end());
            m_nodeMappings.push_back(nm);
            offset += nm.count;
        }
    }

    const std::vector<std::string>& sensitivityNames() const {
        return m_sensitivityNames;
    }

    const std::vector<NodeMapping>& nodeMappings() const {
        return m_nodeMappings;
    }

    int numNodes() const {
        return static_cast<int>(m_sensitivityNames.size());
    }
};
```

**Sensitivity name format** (convention for each ADTemplate type):

| ADTemplate | Name format | Example | Node count |
|-----------|------------|---------|------------|
| `EQDVol` | `risk_eqdvol_{ticker}_{maturity}_{strike}` | `risk_eqdvol_AAPL_1Y_100` | $N_{\text{mat}} \times N_{\text{strike}}$ |
| `EQDSpot` | `risk_eqdspot_{ticker}` | `risk_eqdspot_AAPL` | 1 |
| `IRCurve` | `risk_ir_{name}_{tenor}` | `risk_ir_USD_2Y` | $N_{\text{tenor}}$ |
| `YieldCurve` | `risk_div_{ticker}_{tenor}` | `risk_div_AAPL_1Y` | $N_{\text{tenor}}$ |
| `FXRate` | `risk_fx_{pair}` | `risk_fx_EURUSD` | 1 |
| `SurvivalCurve` | `risk_credit_{name}_{tenor}` | `risk_credit_GS_5Y` | $N_{\text{tenor}}$ |

**Ordering**: nodes are laid out in the order generators were added to ScenarioData. Within each generator, the ADTemplate's own `sensitivityNames()` method defines the sub-ordering (e.g., row-major for vol surfaces: maturity outer, strike inner).

### 3.2 Pack: Snapshot Generator State to Flat Vector

```cpp
template<typename DoubleT>
class ScenarioData {

    Eigen::VectorXd packMarketData() const {
        Eigen::VectorXd x(numNodes());

        for (const auto& nm : m_nodeMappings) {
            std::visit([&](const auto& desc) {
                auto* gen = findGenerator(desc);
                packDescriptor(desc, gen, x, nm.offset);
            }, nm.descriptor);
        }

        return x;
    }

private:
    // Pack EQDVol: row-major (maturity outer, strike inner)
    void packDescriptor(const EQDVolDescriptor& desc,
                         const EQDVolGenerator* gen,
                         Eigen::VectorXd& x, int offset) const {
        auto volSurf = gen->getVolSurface(m_path, m_timepoint, desc.ticker);
        int idx = 0;
        for (int i = 0; i < desc.maturities.size(); ++i)
            for (int j = 0; j < desc.strikes.size(); ++j)
                x(offset + idx++) = volSurf(i, j);
    }

    // Pack EQDSpot: single value
    void packDescriptor(const EQDSpotDescriptor& desc,
                         const EQDSpotGenerator* gen,
                         Eigen::VectorXd& x, int offset) const {
        x(offset) = gen->getSpot(m_path, m_timepoint, desc.ticker);
    }

    // Pack IRCurve: tenor nodes in order
    void packDescriptor(const IRCurveDescriptor& desc,
                         const IRCurveGenerator* gen,
                         Eigen::VectorXd& x, int offset) const {
        auto rates = gen->getRates(m_path, m_timepoint, desc.name);
        for (int i = 0; i < desc.tenors.size(); ++i)
            x(offset + i) = rates[i];
    }

    // Pack YieldCurve: same as IRCurve
    void packDescriptor(const YieldCurveDescriptor& desc,
                         const YieldCurveGenerator* gen,
                         Eigen::VectorXd& x, int offset) const {
        auto yields = gen->getYields(m_path, m_timepoint, desc.ticker);
        for (int i = 0; i < desc.tenors.size(); ++i)
            x(offset + i) = yields[i];
    }

    // Pack FXRate: single value
    void packDescriptor(const FXDescriptor& desc,
                         const FXGenerator* gen,
                         Eigen::VectorXd& x, int offset) const {
        x(offset) = gen->getRate(m_path, m_timepoint, desc.pair);
    }

    // Pack SurvivalCurve: tenor nodes
    void packDescriptor(const CreditDescriptor& desc,
                         const CreditGenerator* gen,
                         Eigen::VectorXd& x, int offset) const {
        auto probs = gen->getSurvivalProbs(m_path, m_timepoint, desc.name);
        for (int i = 0; i < desc.tenors.size(); ++i)
            x(offset + i) = probs[i];
    }
};
```

The ordering of nodes in the flat vector matches `m_sensitivityNames` exactly. This is the contract that makes gradient and Hessian entries interpretable.

### 3.3 Unpack: Populate from Flat AD Vector

```cpp
template<typename DoubleT>
class ScenarioData {
    // Storage for AD-loaded data (used when m_loadedFromVector = true)
    std::unordered_map<std::string, EQDVol<DoubleT>> m_eqdVols;
    std::unordered_map<std::string, EQDSpot<DoubleT>> m_eqdSpots;
    std::unordered_map<std::string, IRCurve<DoubleT>> m_irCurves;
    std::unordered_map<std::string, YieldCurve<DoubleT>> m_yieldCurves;
    std::unordered_map<std::string, FXRate<DoubleT>> m_fxRates;
    std::unordered_map<std::string, SurvivalCurve<DoubleT>> m_survCurves;
    bool m_loadedFromVector = false;

    // Populate all ADTemplates from a flat vector of type T.
    // T can be double, var, or fvar<var>.
    // The vector layout must match the order from buildSensitivityRegistry().
    void loadFromVector(const Eigen::Matrix<DoubleT, Eigen::Dynamic, 1>& x) {
        m_loadedFromVector = true;

        for (const auto& nm : m_nodeMappings) {
            std::visit([&](const auto& desc) {
                unpackDescriptor(desc, x, nm.offset);
            }, nm.descriptor);
        }
    }

private:
    // Unpack EQDVol
    void unpackDescriptor(const EQDVolDescriptor& desc,
                           const Eigen::Matrix<DoubleT, -1, 1>& x,
                           int offset) {
        int nMat = static_cast<int>(desc.maturities.size());
        int nStr = static_cast<int>(desc.strikes.size());

        // Build vol surface matrix with DoubleT values from AD vector
        Eigen::Matrix<DoubleT, Eigen::Dynamic, Eigen::Dynamic> volSurf(nMat, nStr);
        int idx = 0;
        for (int i = 0; i < nMat; ++i)
            for (int j = 0; j < nStr; ++j)
                volSurf(i, j) = x(offset + idx++);

        // Construct EQDVol<DoubleT> with double grid coordinates, DoubleT values
        // Grid coordinates (maturities, strikes) are double — not differentiated
        // Values (vol surface entries) are DoubleT — these are the AD leaves
        m_eqdVols[desc.ticker] = EQDVol<DoubleT>(
            desc.maturities,   // vector<double> — grid, not AD
            desc.strikes,      // vector<double> — grid, not AD
            volSurf);          // Matrix<DoubleT> — values, AD-active
    }

    // Unpack EQDSpot
    void unpackDescriptor(const EQDSpotDescriptor& desc,
                           const Eigen::Matrix<DoubleT, -1, 1>& x,
                           int offset) {
        m_eqdSpots[desc.ticker] = EQDSpot<DoubleT>(x(offset));
    }

    // Unpack IRCurve
    void unpackDescriptor(const IRCurveDescriptor& desc,
                           const Eigen::Matrix<DoubleT, -1, 1>& x,
                           int offset) {
        int n = static_cast<int>(desc.tenors.size());
        Eigen::Matrix<DoubleT, Eigen::Dynamic, 1> rates = x.segment(offset, n);
        m_irCurves[desc.name] = IRCurve<DoubleT>(
            desc.tenors,  // vector<double> — grid, not AD
            rates);       // Vector<DoubleT> — values, AD-active
    }

    // Unpack YieldCurve
    void unpackDescriptor(const YieldCurveDescriptor& desc,
                           const Eigen::Matrix<DoubleT, -1, 1>& x,
                           int offset) {
        int n = static_cast<int>(desc.tenors.size());
        Eigen::Matrix<DoubleT, Eigen::Dynamic, 1> yields = x.segment(offset, n);
        m_yieldCurves[desc.ticker] = YieldCurve<DoubleT>(desc.tenors, yields);
    }

    // Unpack FXRate
    void unpackDescriptor(const FXDescriptor& desc,
                           const Eigen::Matrix<DoubleT, -1, 1>& x,
                           int offset) {
        m_fxRates[desc.pair] = FXRate<DoubleT>(x(offset));
    }

    // Unpack SurvivalCurve
    void unpackDescriptor(const CreditDescriptor& desc,
                           const Eigen::Matrix<DoubleT, -1, 1>& x,
                           int offset) {
        int n = static_cast<int>(desc.tenors.size());
        Eigen::Matrix<DoubleT, Eigen::Dynamic, 1> probs = x.segment(offset, n);
        m_survCurves[desc.name] = SurvivalCurve<DoubleT>(desc.tenors, probs);
    }
};
```

**Key invariant**: The `loadFromVector` layout is the **exact inverse** of `packMarketData`. Same descriptor order, same node order within each descriptor.

### 3.4 Dual-Source Getter Methods

ScenarioData needs to serve data from either generators (normal path) or loaded AD vector (hessian path):

```cpp
template<typename DoubleT>
class ScenarioData {

    EQDVol<DoubleT> getEQDVol(const std::string& ticker) {
        if (m_loadedFromVector) {
            auto it = m_eqdVols.find(ticker);
            if (it == m_eqdVols.end())
                throw std::runtime_error("EQDVol not loaded for " + ticker);
            return it->second;
        }
        // Normal path: query generator, construct ADTemplate from doubles
        auto* gen = findGenerator<EQDVolGenerator>(EQDVolDescriptor{ticker});
        auto rawSurf = gen->getVolSurface(m_path, m_timepoint, ticker);
        auto& desc = getDescriptor<EQDVolDescriptor>(ticker);
        return EQDVol<DoubleT>(desc.maturities, desc.strikes, rawSurf);
    }

    EQDSpot<DoubleT> getEQDSpot(const std::string& ticker) {
        if (m_loadedFromVector)
            return m_eqdSpots.at(ticker);
        auto* gen = findGenerator<EQDSpotGenerator>(EQDSpotDescriptor{ticker});
        return EQDSpot<DoubleT>(gen->getSpot(m_path, m_timepoint, ticker));
    }

    IRCurve<DoubleT> getIRCurve(const std::string& name) {
        if (m_loadedFromVector)
            return m_irCurves.at(name);
        auto* gen = findGenerator<IRCurveGenerator>(IRCurveDescriptor{name});
        auto& desc = getDescriptor<IRCurveDescriptor>(name);
        auto rates = gen->getRates(m_path, m_timepoint, name);
        return IRCurve<DoubleT>(desc.tenors, rates);
    }

    YieldCurve<DoubleT> getYieldCurve(const std::string& ticker) {
        if (m_loadedFromVector)
            return m_yieldCurves.at(ticker);
        auto* gen = findGenerator<YieldCurveGenerator>(YieldCurveDescriptor{ticker});
        auto& desc = getDescriptor<YieldCurveDescriptor>(ticker);
        auto yields = gen->getYields(m_path, m_timepoint, ticker);
        return YieldCurve<DoubleT>(desc.tenors, yields);
    }

    FXRate<DoubleT> getFXRate(const std::string& pair) {
        if (m_loadedFromVector)
            return m_fxRates.at(pair);
        auto* gen = findGenerator<FXGenerator>(FXDescriptor{pair});
        return FXRate<DoubleT>(gen->getRate(m_path, m_timepoint, pair));
    }

    SurvivalCurve<DoubleT> getSurvivalCurve(const std::string& name) {
        if (m_loadedFromVector)
            return m_survCurves.at(name);
        auto* gen = findGenerator<CreditGenerator>(CreditDescriptor{name});
        auto& desc = getDescriptor<CreditDescriptor>(name);
        auto probs = gen->getSurvivalProbs(m_path, m_timepoint, name);
        return SurvivalCurve<DoubleT>(desc.tenors, probs);
    }
};
```

When `loadFromVector` has been called, `m_loadedFromVector = true` and all getters return from the maps. No generators are touched. This is the only behavioral change in ScenarioData, and it is purely additive — the normal pricing path is unaffected.

### 3.5 Constructor for AD ScenarioData

The functor creates a `ScenarioData<fvar<var>>` from scratch. It needs the structural information (which descriptors, what grid sizes) but not the generator pointers:

```cpp
template<typename DoubleT>
class ScenarioData {
    // Construct from blueprint: copies descriptor structure, no generators
    explicit ScenarioData(const std::vector<NodeMapping>& mappings)
        : m_nodeMappings(mappings)
        , m_loadedFromVector(false) {
        // Rebuild sensitivity names from descriptors
        for (const auto& nm : m_nodeMappings) {
            auto names = std::visit([](const auto& d) {
                return d.sensitivityNames();
            }, nm.descriptor);
            m_sensitivityNames.insert(m_sensitivityNames.end(),
                                       names.begin(), names.end());
        }
    }
};
```

---

## 4. The Hessian Functor

### 4.1 Design Principle

The pricer and `ScenarioData<double>` are instantiated **outside** the functor, in the normal pricing loop. By the time the Hessian is computed at a given (path, timestep), the pricer already holds whatever accumulated state it needs (barrier flags, running averages, exercise history). The functor's job is narrow:

1. Build a `ScenarioData<T>` from the AD vector (using node mappings, **not** copying the live ScenarioData — copy constructors are deleted)
2. Create a `Pricer<T>` via `rebind<T>()` from the live `Pricer<double>`, which copies all accumulated state
3. Call `loadScenarioData` + `priceTrade` on the AD-typed objects

The functor does **not** own the pricer or the ScenarioData. It holds references to them.

### 4.2 Implementation

```cpp
template<template<typename> class PricerType>
struct HessianFunctor {
    // References to live objects — the functor does not own these.
    const PricerType<double>& live_pricer;
    const std::vector<NodeMapping>& node_mappings;  // from sd.nodeMappings()

    template<typename T>
    T operator()(const Eigen::Matrix<T, Eigen::Dynamic, 1>& x) const {
        // 1. Construct ScenarioData<T> from node mappings (NOT copied from sd<double>)
        ScenarioData<T> sd_ad(node_mappings);

        // 2. Populate ADTemplates from the AD vector — these are the AD leaves
        sd_ad.loadFromVector(x);

        // 3. Rebind pricer: creates PricerType<T> with accumulated state from live pricer
        //    - Config (strings, doubles, enums): copied as-is
        //    - Plain state (bools, ints): copied as-is
        //    - DoubleT state (running sums, cached values): value_of → T(constant)
        auto pricer_ad = live_pricer.template rebind<T>();

        // 4. Load market data and price — identical to the normal pricing path
        pricer_ad.loadScenarioData(sd_ad);
        return pricer_ad.priceTrade(sd_ad);
    }
};

// Factory: captures references to the live pricer and ScenarioData structure
template<template<typename> class PricerType>
HessianFunctor<PricerType> makeHessianFunctor(
    const PricerType<double>& pricer,
    const ScenarioData<double>& sd)
{
    return HessianFunctor<PricerType>{pricer, sd.nodeMappings()};
}
```

### 4.3 Why This Works — Step by Step

1. **`stan::math::hessian`** calls `functor(x)` with `T = fvar<var>`.
2. The functor constructs `ScenarioData<fvar<var>>` from node mappings — same descriptor structure as the live `ScenarioData<double>` (same tickers, same grid sizes, same ordering). **No copy** of the live ScenarioData occurs.
3. `loadFromVector` builds `EQDVol<fvar<var>>`, `IRCurve<fvar<var>>`, etc. from the AD vector. The vol surface entries, spot prices, and curve node values are `fvar<var>` — these are the AD leaves.
4. `rebind<fvar<var>>()` creates a `PricerA<fvar<var>>` carrying the live pricer's accumulated state. Any `DoubleT`-typed state (e.g., `running_avg`) is extracted via `stan::math::value_of` and promoted to a `fvar<var>` constant — on the tape but with zero tangent, so it does not contribute to derivatives.
5. `loadScenarioData` and `priceTrade` run exactly as in the normal pricing path, but with `fvar<var>` arithmetic.
6. The pricer returns `fvar<var>`, which `stan::math::hessian` unpacks into the price, gradient, and one Hessian column.
7. Steps 1–6 repeat $N$ times (once per market data node), each time with a different forward-mode seed direction.

### 4.4 The `rebind<T>()` Requirement

Each pricer must provide a method to create a differently-typed copy of itself. This is the **only new requirement** on pricers for Hessian support.

```cpp
template<typename DoubleT>
class PricerBase {
public:
    virtual DoubleT priceTrade(ScenarioData<DoubleT>& sd) = 0;
    virtual void loadScenarioData(ScenarioData<DoubleT>& sd) = 0;

    // New: create a PricerType<T> carrying this pricer's accumulated state.
    // Default implementation works for stateless pricers.
    template<typename T>
    PricerBase<T> rebind() const;  // see below for how this is implemented
};
```

**Why not a virtual method?** `rebind<T>()` is a template — it cannot be virtual. Instead, each concrete pricer provides a converting constructor:

```cpp
template<typename DoubleT>
class AsianPricer : public PricerBase<DoubleT> {
    // Config (plain data — never DoubleT)
    std::string m_underlying;
    std::vector<double> m_fixing_dates;
    double m_strike;

    // Accumulated state — may include DoubleT members
    DoubleT m_running_sum = DoubleT(0.0);
    int m_fixing_count = 0;
    DoubleT m_max_spot = DoubleT(0.0);       // for lookback-asian hybrids
    bool m_knocked_in = false;

public:
    // Normal constructor
    AsianPricer(const std::string& underlying,
                const std::vector<double>& fixing_dates,
                double strike)
        : m_underlying(underlying), m_fixing_dates(fixing_dates), m_strike(strike) {}

    // Converting constructor — creates AsianPricer<T> from AsianPricer<OtherT>
    template<typename OtherT>
    explicit AsianPricer(const AsianPricer<OtherT>& other)
        : m_underlying(other.underlying())
        , m_fixing_dates(other.fixingDates())
        , m_strike(other.strike())
        // DoubleT state: extract double via value_of, promote to T as constant
        , m_running_sum(DoubleT(stan::math::value_of(other.runningSum())))
        , m_fixing_count(other.fixingCount())
        , m_max_spot(DoubleT(stan::math::value_of(other.maxSpot())))
        , m_knocked_in(other.knockedIn())
    {}

    // rebind: convenience wrapper around the converting constructor
    template<typename T>
    AsianPricer<T> rebind() const {
        return AsianPricer<T>(*this);
    }

    // ... priceTrade, loadScenarioData as before ...
};
```

**What `stan::math::value_of` does**: Extracts the plain `double` from any AD type:
- `value_of(double d)` → `d`
- `value_of(var v)` → `v.val()` (the double)
- `value_of(fvar<var> fv)` → `value_of(fv.val())` → double

The extracted double, when promoted to `T(double_value)`, becomes a **constant** on the AD tape — it has zero tangent and will not produce adjoint contributions. This is exactly right: past fixings are realized history, not current market data.

#### 4.4.1 State Categories in `rebind`

| State type | Example | How to copy |
|------------|---------|-------------|
| Config (plain doubles, strings, enums) | strike, dates, currency | Copy directly |
| Counter/flag state (int, bool) | fixing_count, barrier_breached | Copy directly |
| `DoubleT` state (accumulated values) | running_sum, max_spot | `value_of` → `T(constant)` |
| Cached market data (`DoubleT`) | m_currentSpot, m_currentVol | **Skip** — `loadScenarioData` will overwrite these from `sd_ad` |

The last category is important: if a pricer caches market data from `loadScenarioData` into member variables, those caches will be overwritten when `loadScenarioData(sd_ad)` is called with the AD-typed ScenarioData. So they don't need special handling in `rebind`.

#### 4.4.2 Stateless Pricers — Zero Effort

For stateless pricers (the majority — vanilla options, swaps, swaptions), `rebind` is trivial because there's no accumulated state:

```cpp
template<typename DoubleT>
class EquityOptionPricer : public PricerBase<DoubleT> {
    std::string m_underlying;
    double m_strike, m_maturity;
    OptionType m_type;

public:
    // Converting constructor — just copies config
    template<typename OtherT>
    explicit EquityOptionPricer(const EquityOptionPricer<OtherT>& other)
        : m_underlying(other.underlying())
        , m_strike(other.strike())
        , m_maturity(other.maturity())
        , m_type(other.type())
    {}

    template<typename T>
    EquityOptionPricer<T> rebind() const { return EquityOptionPricer<T>(*this); }

    // ...
};
```

For these pricers, `rebind` is mechanical — a CRTP base or macro could generate it. But since each pricer already knows its own members, a simple converting constructor is the most explicit and debuggable approach.

### 4.5 What if the Pricer Creates Helpers?

Many pricers delegate to helper functions or objects. These must also be templated:

```cpp
// Helper that calculates forward rates — templated on DoubleT
template<typename DoubleT>
DoubleT calculateForwardRate(const IRCurve<DoubleT>& curve,
                              double t1, double t2) {
    DoubleT df1 = curve.discountFactor(t1);
    DoubleT df2 = curve.discountFactor(t2);
    return (df1 / df2 - 1.0) / (t2 - t1);
}

// Helper struct — also templated
template<typename DoubleT>
struct BlackScholesCalculator {
    DoubleT forward, strike, vol, df, maturity;

    DoubleT callPrice() const {
        using stan::math::exp;
        using stan::math::log;
        using stan::math::sqrt;
        using stan::math::Phi;

        DoubleT d1 = (log(forward / strike) + 0.5 * vol * vol * maturity)
                      / (vol * sqrt(maturity));
        DoubleT d2 = d1 - vol * sqrt(maturity);
        return df * (forward * Phi(d1) - strike * Phi(d2));
    }
};
```

As long as these helpers use `DoubleT` consistently, they compile with `fvar<var>` automatically.

### 4.6 Path-Dependent State: What the Hessian Measures

When the Hessian is computed at timestep $t$ on path $m$, the pricer has already accumulated state from timesteps $0, \ldots, t-1$. The `rebind` copies this state as constants. The Hessian therefore measures:

$$H_{ij} = \frac{\partial^2}{\partial m_i \partial m_j} \text{Price}(\text{market data at } t \mid \text{path history } 0..t\!-\!1)$$

This is the economically meaningful quantity: *"given what has already happened on this path, how does the price respond to perturbations in today's market data?"* Past fixings, barrier events, and exercise decisions are sunk — they are not market data you have sensitivity to.

If you need sensitivity to a past fixing (e.g., for a reset-in-advance swap), that fixing's value should be part of the current ScenarioData's packed vector, not accumulated pricer state.

---

## 5. Call Site Integration

### 5.1 Basic Usage

```cpp
// ═══════════════════════════════════════════════════════════════
// Setup (once per trade, before MC loop)
// ═══════════════════════════════════════════════════════════════
ScenarioData<double> sd;
sd.addEQDVol("AAPL", vol_gen);      // non-owning pointer to shared generator
sd.addEQDSpot("AAPL", spot_gen);
sd.addIRCurve("USD", curve_gen);
sd.buildSensitivityRegistry();       // builds name list and node mappings

EquityOptionPricer<double> pricer("AAPL", 100.0, 1.0, "USD", OptionType::Call);

// ═══════════════════════════════════════════════════════════════
// In the MC loop (pricer and sd already exist, pricer has accumulated state)
// ═══════════════════════════════════════════════════════════════
sd.setScenarioTimepoint(path, timepoint);
pricer.loadScenarioData(sd);
double pv = pricer.priceTrade(sd);

// ═══════════════════════════════════════════════════════════════
// Hessian computation at this (path, timepoint)
// ═══════════════════════════════════════════════════════════════

// Step 1: Snapshot current market data from generators to flat vector
Eigen::VectorXd x = sd.packMarketData();

// Step 2: Build functor (references the live pricer + sd's node mappings)
auto functor = makeHessianFunctor<EquityOptionPricer>(pricer, sd);

// Step 3: Compute
double fx;
Eigen::VectorXd grad;   // N first-order sensitivities
Eigen::MatrixXd H;       // N×N Hessian
stan::math::hessian(functor, x, fx, grad, H);

// Step 4: Interpret results
const auto& names = sd.sensitivityNames();
for (int i = 0; i < names.size(); ++i) {
    std::cout << names[i] << ": delta=" << grad(i);
    for (int j = 0; j <= i; ++j)
        if (std::abs(H(i,j)) > 1e-12)
            std::cout << "  cross[" << names[j] << "]=" << H(i,j);
    std::cout << "\n";
}
```

### 5.2 First-Order Only (Faster)

When only first-order sensitivities are needed, use `stan::math::gradient` instead. It performs a single reverse pass using `var` (not `fvar<var>`) — roughly $5\times$ the cost of a single double pricing, regardless of $N$.

```cpp
// Same functor works for both gradient and hessian.
// gradient() instantiates with T = var.
// hessian() instantiates with T = fvar<var>.
auto functor = makeHessianFunctor<EquityOptionPricer>(pricer, sd);
Eigen::VectorXd x = sd.packMarketData();

double fx;
Eigen::VectorXd grad;
stan::math::gradient(functor, x, fx, grad);
// Cost: ~5× single pricing, regardless of N
```

### 5.3 Unified Sensitivity Interface

```cpp
enum class SensitivityOrder { FIRST, SECOND };

struct SensitivityResult {
    double price;
    std::vector<std::string> names;     // N sensitivity names
    Eigen::VectorXd gradient;            // N first-order
    Eigen::MatrixXd hessian;             // N×N second-order (empty if FIRST)
};

template<template<typename> class PricerType>
SensitivityResult calculateSensitivities(
    const PricerType<double>& pricer,
    ScenarioData<double>& sd,
    SensitivityOrder order = SensitivityOrder::FIRST)
{
    SensitivityResult result;
    result.names = sd.sensitivityNames();

    Eigen::VectorXd x = sd.packMarketData();
    auto functor = makeHessianFunctor<PricerType>(pricer, sd);

    if (order == SensitivityOrder::FIRST) {
        stan::math::gradient(functor, x, result.price, result.gradient);
    } else {
        stan::math::hessian(functor, x,
                             result.price, result.gradient, result.hessian);
    }

    return result;
}
```

### 5.4 Integration with the MC Loop

The pricer and ScenarioData are instantiated once before the double loop. The pricer accumulates state across timesteps naturally. The Hessian functor is built fresh at each (path, timestep) where sensitivities are needed, referencing the live pricer at that point.

```cpp
// Setup (once per trade)
ScenarioData<double> sd;
// ... add generators ...
sd.buildSensitivityRegistry();

EquityOptionPricer<double> pricer("AAPL", 100.0, 1.0, "USD", OptionType::Call);

for (int path = 0; path < N_paths; ++path) {
    // Reset pricer state at start of each path (if path-dependent)
    pricer.resetPathState();

    for (int tp = 0; tp < N_timepoints; ++tp) {
        sd.setScenarioTimepoint(path, tp);
        pricer.loadScenarioData(sd);

        // Always: price (fast, double only)
        // Pricer updates its internal state (barriers, fixings, etc.)
        double pv = pricer.priceTrade(sd);
        accumulate_exposure(path, tp, pv);

        // Conditionally: first-order Greeks
        if (need_greeks(path, tp)) {
            auto sens = calculateSensitivities<EquityOptionPricer>(
                pricer, sd, SensitivityOrder::FIRST);
            accumulate_greeks(path, tp, sens);
        }

        // Rarely: full Hessian (e.g., t=0 only, or for PnL explain)
        if (need_hessian(path, tp)) {
            auto sens = calculateSensitivities<EquityOptionPricer>(
                pricer, sd, SensitivityOrder::SECOND);
            accumulate_hessian(path, tp, sens);
        }
    }
}
```

**Note on ordering**: The Hessian call uses `rebind` to create a separate `Pricer<fvar<var>>` — it does **not** mutate the live `Pricer<double>`. The live pricer's state is unaffected by the Hessian computation, so the normal pricing loop continues correctly.

---

## 6. Per-ADTemplate Sensitivity Extraction

The Hessian output is a flat $N \times N$ matrix. Extract structured blocks for risk reporting:

```cpp
struct StructuredSensitivities {
    // ─── First order ───
    std::map<std::string, double> delta;                    // ∂P/∂S per equity
    std::map<std::string, Eigen::MatrixXd> vega;            // ∂P/∂σ per vol surface
    std::map<std::string, Eigen::VectorXd> rho;             // ∂P/∂r per curve
    std::map<std::string, Eigen::VectorXd> div_sens;        // ∂P/∂q per div curve
    std::map<std::string, double> fx_delta;                  // ∂P/∂FX per pair
    std::map<std::string, Eigen::VectorXd> credit_sens;     // ∂P/∂λ per credit curve

    // ─── Second order: same-type ───
    std::map<std::string, double> gamma;                     // ∂²P/∂S²
    std::map<std::string, Eigen::MatrixXd> volga;            // ∂²P/∂σ_i∂σ_j
    std::map<std::string, Eigen::MatrixXd> curve_gamma;      // ∂²P/∂r_i∂r_j

    // ─── Second order: cross-type ───
    std::map<std::string, Eigen::VectorXd> vanna;            // ∂²P/∂S∂σ_i
    std::map<std::string, Eigen::VectorXd> spot_rate;        // ∂²P/∂S∂r_i
    std::map<std::string, Eigen::MatrixXd> vega_rho_cross;   // ∂²P/∂σ_i∂r_j
    std::map<std::string, Eigen::VectorXd> spot_div;         // ∂²P/∂S∂q_i
    std::map<std::string, double> spot_fx;                    // ∂²P/∂S∂FX
};

StructuredSensitivities extractStructured(
    const SensitivityResult& result,
    const std::vector<NodeMapping>& mappings)
{
    StructuredSensitivities out;

    // ─── First order and diagonal second-order blocks ───
    for (const auto& nm : mappings) {
        std::visit([&](const auto& desc) {
            using D = std::decay_t<decltype(desc)>;

            if constexpr (std::is_same_v<D, EQDVolDescriptor>) {
                int nMat = desc.maturities.size();
                int nStr = desc.strikes.size();

                // Vega: gradient segment reshaped to (mat × strike)
                Eigen::MatrixXd vega(nMat, nStr);
                int idx = 0;
                for (int i = 0; i < nMat; ++i)
                    for (int j = 0; j < nStr; ++j)
                        vega(i, j) = result.gradient(nm.offset + idx++);
                out.vega[desc.ticker] = vega;

                // Volga: H sub-matrix (nm.count × nm.count)
                if (result.hessian.size() > 0) {
                    out.volga[desc.ticker] = result.hessian.block(
                        nm.offset, nm.offset, nm.count, nm.count);
                }
            }
            else if constexpr (std::is_same_v<D, EQDSpotDescriptor>) {
                out.delta[desc.ticker] = result.gradient(nm.offset);
                if (result.hessian.size() > 0)
                    out.gamma[desc.ticker] = result.hessian(nm.offset, nm.offset);
            }
            else if constexpr (std::is_same_v<D, IRCurveDescriptor>) {
                out.rho[desc.name] = result.gradient.segment(nm.offset, nm.count);
                if (result.hessian.size() > 0)
                    out.curve_gamma[desc.name] = result.hessian.block(
                        nm.offset, nm.offset, nm.count, nm.count);
            }
            else if constexpr (std::is_same_v<D, YieldCurveDescriptor>) {
                out.div_sens[desc.ticker] = result.gradient.segment(nm.offset, nm.count);
            }
            else if constexpr (std::is_same_v<D, FXDescriptor>) {
                out.fx_delta[desc.pair] = result.gradient(nm.offset);
            }
            else if constexpr (std::is_same_v<D, CreditDescriptor>) {
                out.credit_sens[desc.name] = result.gradient.segment(nm.offset, nm.count);
            }
        }, nm.descriptor);
    }

    // ─── Cross-type blocks ───
    if (result.hessian.size() == 0) return out;

    for (const auto& nm_a : mappings) {
        for (const auto& nm_b : mappings) {
            if (nm_a.offset >= nm_b.offset) continue;  // upper triangle only

            auto extract_block = [&]() {
                return result.hessian.block(
                    nm_a.offset, nm_b.offset, nm_a.count, nm_b.count);
            };

            // Spot × Vol → vanna
            if (std::holds_alternative<EQDSpotDescriptor>(nm_a.descriptor) &&
                std::holds_alternative<EQDVolDescriptor>(nm_b.descriptor)) {
                auto& ticker_a = std::get<EQDSpotDescriptor>(nm_a.descriptor).ticker;
                auto& ticker_b = std::get<EQDVolDescriptor>(nm_b.descriptor).ticker;
                if (ticker_a == ticker_b) {
                    // vanna is a vector: ∂²P/∂S∂σ_i for each vol node
                    out.vanna[ticker_a] = result.hessian.block(
                        nm_a.offset, nm_b.offset, 1, nm_b.count).transpose();
                }
            }

            // Spot × Curve → spot-rate cross
            if (std::holds_alternative<EQDSpotDescriptor>(nm_a.descriptor) &&
                std::holds_alternative<IRCurveDescriptor>(nm_b.descriptor)) {
                auto& ticker = std::get<EQDSpotDescriptor>(nm_a.descriptor).ticker;
                auto& curve = std::get<IRCurveDescriptor>(nm_b.descriptor).name;
                out.spot_rate[ticker + "_" + curve] = result.hessian.block(
                    nm_a.offset, nm_b.offset, 1, nm_b.count).transpose();
            }

            // Vol × Curve → vega-rho cross
            if (std::holds_alternative<EQDVolDescriptor>(nm_a.descriptor) &&
                std::holds_alternative<IRCurveDescriptor>(nm_b.descriptor)) {
                auto& ticker = std::get<EQDVolDescriptor>(nm_a.descriptor).ticker;
                auto& curve = std::get<IRCurveDescriptor>(nm_b.descriptor).name;
                out.vega_rho_cross[ticker + "_" + curve] = extract_block();
            }
        }
    }

    return out;
}
```

---

## 7. Performance

### 7.1 Cost Model

`stan::math::hessian` performs $N$ forward-over-reverse sweeps. Each sweep costs approximately $3$–$5\times$ a single reverse-mode (`var`-only) evaluation. The reverse-mode evaluation itself is typically $3$–$5\times$ the double evaluation. So total Hessian cost is roughly $10$–$25 \cdot N \times$ double cost.

| Pricer complexity | Double | `var` (gradient) | `fvar<var>` (1 sweep) | Full Hessian ($N = 200$) |
|-------------------|--------|-------------------|----------------------|--------------------------|
| Closed-form (BS) | ~1 μs | ~5 μs | ~15 μs | ~3 ms |
| Lattice (100 steps) | ~100 μs | ~500 μs | ~1.5 ms | ~300 ms |
| Simple MC (1K paths) | ~1 ms | ~5 ms | ~15 ms | ~3 s |
| Complex MC (10K paths) | ~10 ms | ~50 ms | ~150 ms | ~30 s |

### 7.2 Memory

Each `fvar<var>` value occupies ~48 bytes on the Stan arena (two `var` nodes). The tape grows proportionally to the number of operations. For a BS pricer with ~50 operations and $N = 200$ inputs, the tape per sweep is ~$200 \times 48 + 50 \times 48 \approx 12$ KB. Negligible.

For MC pricers with millions of operations, the tape can grow to tens of MB per sweep. Stan's arena allocator handles this efficiently, but monitor peak memory if running many sweeps without `recover_memory()`.

### 7.3 Sparse Hessian for Large $N$

When $N > 500$, the full Hessian is $N^2$ doubles (e.g., $500^2 \times 8 = 2$ MB) and takes $O(N)$ sweeps. For many risk applications, the full Hessian is unnecessary — most cross-gammas are negligible.

**Block-diagonal Hessian**: Only compute within-group and selected cross-group blocks.

```cpp
struct SparseHessianConfig {
    // Which descriptor pairs to compute cross-Hessians for
    // If empty, compute only within-descriptor blocks (diagonal)
    std::vector<std::pair<Descriptor, Descriptor>> cross_blocks;
};
```

Implementation uses a **SubsetFunctor** that holds the full vector at fixed double values and only varies a subset:

```cpp
template<typename BaseFunctor>
struct SubsetFunctor {
    const BaseFunctor& base;
    Eigen::VectorXd x_fixed;        // full vector (double, held fixed)
    std::vector<int> active_indices; // which elements to vary

    template<typename T>
    T operator()(const Eigen::Matrix<T, Eigen::Dynamic, 1>& x_active) const {
        // Start with all elements as T(fixed_value)
        int N = static_cast<int>(x_fixed.size());
        Eigen::Matrix<T, Eigen::Dynamic, 1> x_full(N);
        for (int i = 0; i < N; ++i)
            x_full(i) = T(x_fixed(i));

        // Overwrite the active subset with AD values
        for (int i = 0; i < static_cast<int>(active_indices.size()); ++i)
            x_full(active_indices[i]) = x_active(i);

        return base(x_full);
    }
};

template<typename BaseFunctor>
auto makeSubsetFunctor(const BaseFunctor& base,
                        const Eigen::VectorXd& x_full,
                        const std::vector<int>& indices) {
    return SubsetFunctor<BaseFunctor>{base, x_full, indices};
}
```

Usage for block-diagonal Hessian:

```cpp
void computeSparseHessian(ScenarioData<double>& sd,
                           const auto& functor,
                           const SparseHessianConfig& config,
                           SensitivityResult& result)
{
    Eigen::VectorXd x_full = sd.packMarketData();
    int N = static_cast<int>(x_full.size());
    result.hessian = Eigen::MatrixXd::Zero(N, N);

    // Full gradient (always needed, cheap — single reverse pass)
    stan::math::gradient(functor, x_full, result.price, result.gradient);

    // ─── Diagonal blocks: Hessian within each descriptor ───
    for (const auto& nm : sd.nodeMappings()) {
        std::vector<int> indices(nm.count);
        std::iota(indices.begin(), indices.end(), nm.offset);

        auto sub = makeSubsetFunctor(functor, x_full, indices);
        Eigen::VectorXd x_sub = x_full.segment(nm.offset, nm.count);

        double fx_sub;
        Eigen::VectorXd grad_sub;
        Eigen::MatrixXd H_sub;
        stan::math::hessian(sub, x_sub, fx_sub, grad_sub, H_sub);

        result.hessian.block(nm.offset, nm.offset, nm.count, nm.count) = H_sub;
    }

    // ─── Cross blocks: Hessian between selected descriptor pairs ───
    for (auto& [desc_a, desc_b] : config.cross_blocks) {
        auto nm_a = sd.findMapping(desc_a);
        auto nm_b = sd.findMapping(desc_b);

        // Combine both subsets
        std::vector<int> indices;
        for (int i = 0; i < nm_a.count; ++i) indices.push_back(nm_a.offset + i);
        for (int i = 0; i < nm_b.count; ++i) indices.push_back(nm_b.offset + i);

        auto sub = makeSubsetFunctor(functor, x_full, indices);
        int n_ab = nm_a.count + nm_b.count;
        Eigen::VectorXd x_sub(n_ab);
        x_sub.head(nm_a.count) = x_full.segment(nm_a.offset, nm_a.count);
        x_sub.tail(nm_b.count) = x_full.segment(nm_b.offset, nm_b.count);

        double fx_sub;
        Eigen::VectorXd grad_sub;
        Eigen::MatrixXd H_sub;
        stan::math::hessian(sub, x_sub, fx_sub, grad_sub, H_sub);

        // Extract and place the off-diagonal cross block
        auto cross = H_sub.topRightCorner(nm_a.count, nm_b.count);
        result.hessian.block(nm_a.offset, nm_b.offset, nm_a.count, nm_b.count) = cross;
        result.hessian.block(nm_b.offset, nm_a.offset, nm_b.count, nm_a.count) = cross.transpose();
    }
}
```

### 7.4 Parallelization

The $N$ forward-over-reverse sweeps inside `stan::math::hessian` are sequential (single-threaded). For parallelism:

**Option A: Parallelize across trades** (simplest, recommended). Each trade's Hessian is independent. With 100 trades on 10 threads: $10\times$ speedup.

**Option B: Parallelize across timepoints**. Each path/timepoint evaluation is independent.

**Option C: Parallel Hessian sweeps** (custom implementation). Each of the $N$ forward-over-reverse sweeps is independent — they can run on separate threads with separate tapes. Implement using `stan::math::nested_rev_autodiff`:

```cpp
template<typename F>
void parallel_hessian(const F& f, const Eigen::VectorXd& x,
                       double& fx, Eigen::VectorXd& grad, Eigen::MatrixXd& H,
                       int num_threads = 0) {
    using stan::math::fvar;
    using stan::math::var;
    int N = static_cast<int>(x.size());
    H.resize(N, N);

    if (num_threads <= 0)
        num_threads = std::max(1u, std::thread::hardware_concurrency());

    // Base evaluation for price and gradient (single thread)
    stan::math::gradient(f, x, fx, grad);

    // Each thread computes a subset of Hessian columns
    std::vector<std::thread> threads;
    std::atomic<int> next_col{0};

    auto worker = [&]() {
        for (;;) {
            int i = next_col.fetch_add(1, std::memory_order_relaxed);
            if (i >= N) break;

            // Each column computation has its own tape (nested_rev_autodiff)
            stan::math::nested_rev_autodiff nested;

            // Build fvar<var> input with tangent seeded in direction e_i
            Eigen::Matrix<fvar<var>, Eigen::Dynamic, 1> x_fv(N);
            for (int j = 0; j < N; ++j)
                x_fv(j) = fvar<var>(var(x(j)), (j == i) ? 1.0 : 0.0);

            // Evaluate
            fvar<var> result = f(x_fv);

            // Extract Hessian column i from the derivative part
            stan::math::grad(result.d_.vi_);
            for (int j = 0; j < N; ++j)
                H(i, j) = x_fv(j).val().adj();
        }
    };

    threads.reserve(num_threads);
    for (int t = 0; t < num_threads; ++t)
        threads.emplace_back(worker);
    for (auto& t : threads)
        t.join();
}
```

**Note**: Stan's `var` uses a global arena with thread-local storage. Ensure each thread gets its own arena via `nested_rev_autodiff` or by calling `stan::math::ChainableStack::instance_` initialization per thread. Check Stan Math threading documentation for your version.

---

## 8. Analytical Primitives for Hot-Path Functions

For frequently called functions whose analytical derivatives are known (Black-Scholes, interpolations, SABR approximation), we can register them as Stan Math primitives with hand-coded partials. This short-circuits the AD tape: instead of recording every intermediate operation, Stan records a **single node** with pre-supplied adjoints.

**No design changes required.** The pricer calls `bs_call(F, K, sigma, df, T)` as before — the compiler dispatches to the right overload based on `DoubleT`. The functor, `rebind`, `packMarketData`, `loadFromVector` — all unchanged.

### 8.1 Mechanism: How Stan Math Primitives Work

#### 8.1.1 First Order — `var` overload via `precomputed_gradients`

For a function $f(x_1, \ldots, x_n)$ where we know $\partial f / \partial x_i$ analytically, `precomputed_gradients` registers a single `var` node whose adjoints are the supplied partials:

```cpp
inline stan::math::var bs_call(const stan::math::var& F,
                                const stan::math::var& K,
                                const stan::math::var& sigma,
                                const stan::math::var& df,
                                double T) {
    // Extract doubles
    double f = F.val(), k = K.val(), s = sigma.val(), d = df.val();
    double sqrtT = std::sqrt(T);
    double d1 = (std::log(f / k) + 0.5 * s * s * T) / (s * sqrtT);
    double d2 = d1 - s * sqrtT;
    double Nd1 = 0.5 * std::erfc(-d1 * M_SQRT1_2);
    double Nd2 = 0.5 * std::erfc(-d2 * M_SQRT1_2);
    double nd1 = std::exp(-0.5 * d1 * d1) / std::sqrt(2.0 * M_PI);

    // Value
    double price = d * (f * Nd1 - k * Nd2);

    // Analytical first-order partials
    double dP_dF     = d * Nd1;                  // delta
    double dP_dK     = -d * Nd2;                 // dual delta
    double dP_dsigma = d * f * nd1 * sqrtT;      // vega
    double dP_ddf    = f * Nd1 - k * Nd2;        // undiscounted price

    return stan::math::precomputed_gradients(
        price,
        {F, K, sigma, df},
        {dP_dF, dP_dK, dP_dsigma, dP_ddf}
    );
}
```

**Tape cost**: 1 node (the `precomputed_gradients` node) vs ~30 nodes for the full formula. Reverse pass reads 4 pre-stored doubles instead of backpropagating through `log`, `erfc`, `exp`, `sqrt`, etc.

This overload is selected when the pricer runs with `DoubleT = var` (i.e., `stan::math::gradient` calls).

#### 8.1.2 Second Order — `fvar<var>` overload

For `stan::math::hessian`, the inputs are `fvar<var>`. The strategy: compute value and first-order partials as **`var` expressions** (not doubles), then assemble the `fvar<var>` return using the chain rule. Reverse mode differentiates through the partials to get second derivatives.

```cpp
inline stan::math::fvar<stan::math::var> bs_call(
    const stan::math::fvar<stan::math::var>& F,
    const stan::math::fvar<stan::math::var>& K,
    const stan::math::fvar<stan::math::var>& sigma,
    const stan::math::fvar<stan::math::var>& df,
    double T)
{
    using stan::math::var;
    using stan::math::fvar;

    // Extract var components — these go on the reverse tape
    var f = F.val(), k = K.val(), s = sigma.val(), d = df.val();

    // Compute intermediates in var arithmetic (small tape)
    var sqrtT_v(std::sqrt(T));
    var d1 = (log(f / k) + 0.5 * s * s * T) / (s * sqrtT_v);
    var d2 = d1 - s * sqrtT_v;
    var Nd1 = stan::math::Phi(d1);
    var Nd2 = stan::math::Phi(d2);
    var nd1 = exp(-0.5 * d1 * d1) / sqrt(2.0 * M_PI);

    // Value (var)
    var price = d * (f * Nd1 - k * Nd2);

    // Partials as var — each is a small expression (~2-3 tape nodes)
    var dP_dF     = d * Nd1;
    var dP_dK     = -d * Nd2;
    var dP_dsigma = d * f * nd1 * sqrtT_v;
    var dP_ddf    = f * Nd1 - k * Nd2;

    // Forward tangent via chain rule: df/dx_i * dx_i/dt (where t is the seed direction)
    var tangent = dP_dF * F.d_ + dP_dK * K.d_
                + dP_dsigma * sigma.d_ + dP_ddf * df.d_;

    return fvar<var>(price, tangent);
}
```

When `hessian()` calls `grad()` on `tangent`, it differentiates through `Nd1`, `nd1`, `d1`, etc. — these are simple `var` expressions, so the second-order tape is small.

**Why this gives correct second derivatives**: The `tangent` is $\sum_i (\partial f / \partial x_i) \cdot \dot{x}_i$, where each partial is a `var` that depends on the inputs. Reverse-mode differentiating `tangent` w.r.t. all inputs gives $\sum_i (\partial^2 f / \partial x_j \partial x_i) \cdot \dot{x}_i$ — exactly column $j$ of the Hessian times the seed direction. This is mathematically identical to what happens when the full formula is expanded, but with a much smaller tape.

#### 8.1.3 Double overload (normal pricing)

```cpp
inline double bs_call(double F, double K, double sigma, double df, double T) {
    double sqrtT = std::sqrt(T);
    double d1 = (std::log(F / K) + 0.5 * sigma * sigma * T) / (sigma * sqrtT);
    double d2 = d1 - sigma * sqrtT;
    return df * (F * 0.5 * std::erfc(-d1 * M_SQRT1_2)
               - K * 0.5 * std::erfc(-d2 * M_SQRT1_2));
}
```

All three overloads coexist. The compiler picks the right one based on `DoubleT`.

### 8.2 Tape Size Comparison

| Function | Naive tape nodes | Analytical primitive tape nodes | Speedup factor |
|----------|-----------------|-------------------------------|---------------|
| BS call/put | ~60 | ~15 | ~4× |
| BS digital (call/put) | ~40 | ~8 | ~5× |
| Normal (Bachelier) call | ~30 | ~8 | ~4× |
| SABR implied vol (Hagan) | ~120 | ~20 | ~6× |
| Barrier correction (Broadie-Glasserman) | ~80 | ~15 | ~5× |
| Cubic spline eval (given coefficients) | ~15 | ~5 | ~3× |

The speedup factor applies to **each `fvar<var>` sweep** — so for $N = 200$ Hessian columns, a 4× reduction per sweep compounds to 4× overall Hessian speedup for that subexpression.

### 8.3 Which Functions to Prioritize

Prioritize by **call frequency × tape node count**:

1. **Black-Scholes family** (call, put, digital call/put, straddle) — called at every (path, timestep) in MC, ~60 nodes each. Highest impact.
2. **Normal (Bachelier) model** — same call pattern, simpler partials.
3. **Cubic spline evaluation** — called for every interpolation lookup (vol surface, yield curve). Each lookup is ~15 nodes but called dozens of times per pricing. Aggregate impact is large.
4. **SABR/SVI implied vol** — ~120 nodes, called during vol surface construction. High per-call impact.
5. **Barrier corrections** — Broadie-Glasserman continuity correction, moderately complex.
6. **CDF/PDF compositions** — `Phi(x)`, `exp(-x²)` chains that appear in many closed-form pricers.

### 8.4 Implementation Pattern for Other Functions

The pattern is always the same three overloads:

```cpp
// 1. double — fast, no tape
inline double my_func(double x1, double x2, ...) {
    // direct computation
}

// 2. var — precomputed_gradients with analytical partials
inline stan::math::var my_func(const stan::math::var& x1,
                                const stan::math::var& x2, ...) {
    double v1 = x1.val(), v2 = x2.val();
    double value = /* analytical */;
    double df_dx1 = /* analytical */;
    double df_dx2 = /* analytical */;
    return stan::math::precomputed_gradients(value, {x1, x2}, {df_dx1, df_dx2});
}

// 3. fvar<var> — partials as var, forward tangent via chain rule
inline stan::math::fvar<stan::math::var> my_func(
    const stan::math::fvar<stan::math::var>& x1,
    const stan::math::fvar<stan::math::var>& x2, ...)
{
    using stan::math::var;
    var v1 = x1.val(), v2 = x2.val();

    // Compute value and partials in var arithmetic
    var value = /* same formula, but in var */;
    var df_dx1 = /* partial w.r.t. x1, in var */;
    var df_dx2 = /* partial w.r.t. x2, in var */;

    // Forward tangent
    var tangent = df_dx1 * x1.d_ + df_dx2 * x2.d_;

    return stan::math::fvar<var>(value, tangent);
}
```

**Key detail for `fvar<var>` overload**: The partials must be computed in `var` arithmetic (not double), because reverse mode needs to differentiate through them. If you used `double` partials, the Hessian would be zero (no tape to differentiate).

### 8.5 Example: Digital Call

```cpp
// Digital call: pays 1 if S > K at expiry
// Value = df * N(d2)
// dP/dF = df * n(d2) / (F * sigma * sqrt(T))
// dP/dsigma = -df * n(d2) * d1 / sigma

inline stan::math::fvar<stan::math::var> digital_call(
    const stan::math::fvar<stan::math::var>& F,
    const stan::math::fvar<stan::math::var>& K,
    const stan::math::fvar<stan::math::var>& sigma,
    const stan::math::fvar<stan::math::var>& df,
    double T)
{
    using stan::math::var;
    using stan::math::fvar;

    var f = F.val(), k = K.val(), s = sigma.val(), d = df.val();
    var sqrtT_v(std::sqrt(T));
    var d1 = (log(f / k) + 0.5 * s * s * T) / (s * sqrtT_v);
    var d2 = d1 - s * sqrtT_v;
    var Nd2 = stan::math::Phi(d2);
    var nd2 = exp(-0.5 * d2 * d2) / sqrt(2.0 * M_PI);

    var price = d * Nd2;

    // Partials
    var dP_dF     = d * nd2 / (f * s * sqrtT_v);
    var dP_dK     = -d * nd2 / (k * s * sqrtT_v);
    var dP_dsigma = -d * nd2 * d1 / s;
    var dP_ddf    = Nd2;

    var tangent = dP_dF * F.d_ + dP_dK * K.d_
                + dP_dsigma * sigma.d_ + dP_ddf * df.d_;

    return fvar<var>(price, tangent);
}
```

### 8.6 Cubic Spline Evaluation Primitive

Spline evaluation is called very frequently (every vol/rate lookup). Given precomputed coefficients $a_i, b_i, c_i, d_i$ for the interval containing $x$:

$$f(x) = a_i + b_i h + c_i h^2 + d_i h^3, \quad h = x - x_i$$

The coefficients depend on the curve node values (AD leaves). For the primitive, we treat the coefficients as `var` inputs:

```cpp
inline stan::math::fvar<stan::math::var> spline_eval(
    const stan::math::fvar<stan::math::var>& a,
    const stan::math::fvar<stan::math::var>& b,
    const stan::math::fvar<stan::math::var>& c,
    const stan::math::fvar<stan::math::var>& d_coeff,
    double h)   // h = x - x_i, typically a non-AD query point
{
    using stan::math::var;
    using stan::math::fvar;

    var av = a.val(), bv = b.val(), cv = c.val(), dv = d_coeff.val();

    // Value: a + b*h + c*h² + d*h³
    var value = av + h * (bv + h * (cv + h * dv));

    // Partials w.r.t. coefficients (trivial)
    // df/da = 1, df/db = h, df/dc = h², df/dd = h³
    var tangent = a.d_ + h * (b.d_ + h * (c.d_ + h * d_coeff.d_));

    return fvar<var>(value, tangent);
}
```

This is particularly efficient because the partials w.r.t. coefficients are just powers of $h$ — no var operations needed for the tangent computation itself. The second derivatives come from reverse-mode differentiating the coefficients (which depend on the curve nodes through the tridiagonal solve).

### 8.7 When NOT to Use Analytical Primitives

- **One-off or rare functions**: The implementation cost (three overloads, deriving partials, testing) isn't worth it for functions called once per pricing.
- **Functions with many inputs**: If $n > 10$ inputs, the tangent computation ($n$ multiplications) starts to rival the naive tape cost. The crossover depends on the formula complexity.
- **Functions where partials are as complex as the function**: For some functions (e.g., multivariate copulas), the analytical partials are harder to implement than the function itself. Let AD handle those.
- **Prototyping phase**: Get correctness first with naive AD. Profile. Then add analytical primitives for the hot spots.

### 8.8 Verification

Each analytical primitive must be verified against naive AD:

```cpp
// Verify bs_call primitive matches naive implementation
template<typename T>
T bs_call_naive(T F, T K, T sigma, T df, double T_mat) {
    using stan::math::log; using stan::math::sqrt;
    using stan::math::exp; using stan::math::Phi;
    T sqrtT = sqrt(T_mat);
    T d1 = (log(F / K) + 0.5 * sigma * sigma * T_mat) / (sigma * sqrtT);
    T d2 = d1 - sigma * sqrtT;
    return df * (F * Phi(d1) - K * Phi(d2));
}

void verify_bs_primitive() {
    auto make_functor = [](auto bs_fn) {
        return [bs_fn](const auto& x) {
            using T = typename std::decay_t<decltype(x)>::Scalar;
            return bs_fn(T(x(0)), T(x(1)), T(x(2)), T(x(3)), 1.0);
        };
    };

    Eigen::VectorXd x(4);
    x << 100.0, 100.0, 0.2, 0.95;  // F, K, sigma, df

    double fx1, fx2;
    Eigen::VectorXd g1, g2;
    Eigen::MatrixXd H1, H2;

    stan::math::hessian(make_functor([](auto... args){ return bs_call(args...); }),
                         x, fx1, g1, H1);
    stan::math::hessian(make_functor([](auto... args){ return bs_call_naive(args...); }),
                         x, fx2, g2, H2);

    assert(std::abs(fx1 - fx2) < 1e-12);
    assert((g1 - g2).norm() < 1e-10);
    assert((H1 - H2).norm() < 1e-8);
}
```

Run this for each primitive with a range of inputs (ATM, deep ITM, deep OTM, short/long maturity, high/low vol) to catch edge cases.

---

## 9. AD-Safe Math Functions

Many pricing operations use `min`, `max`, `abs`, `clamp`, `heaviside`, and payoff functions that have discontinuous derivatives. AD produces correct first derivatives everywhere except at the kink, but the **second derivative is zero almost everywhere** (Hessian entries involving these operations will be zero or undefined at the kink).

For smooth, AD-compatible second derivatives, replace these with differentiable approximations.

### 9.1 Core Smooth Approximations

All functions below are templated on `DoubleT` and work with `double`, `var`, and `fvar<var>`.

```cpp
namespace ad_math {

// ═══════════════════════════════════════════════════════════════
// smooth_max(a, b, eps)
// ═══════════════════════════════════════════════════════════════
// Approximates max(a, b) with continuous second derivative.
//
// Uses the LogSumExp trick:
//   smooth_max(a, b) = eps * log(exp(a/eps) + exp(b/eps))
//
// Properties:
//   - Converges to max(a, b) as eps → 0
//   - C∞ (infinitely differentiable)
//   - smooth_max(a, b) ≥ max(a, b) (always overestimates by ≤ eps·ln(2))
//   - ∂/∂a = sigmoid((a-b)/eps) = exp(a/eps) / (exp(a/eps) + exp(b/eps))
//   - ∂²/∂a² = sigmoid'((a-b)/eps) / eps
//
// Numerically stable form (avoids overflow):
//   smooth_max(a, b) = max(a,b) + eps * log(1 + exp(-|a-b|/eps))
//
// Default eps = 1e-4 gives max error of ~7e-5 and smooth transition
// over a window of ~4*eps around a=b.

template<typename T>
T smooth_max(const T& a, const T& b, double eps = 1e-4) {
    using stan::math::exp;
    using stan::math::log;
    using stan::math::fabs;

    T diff = (a - b) / eps;
    // Numerically stable LogSumExp
    T max_ab = stan::math::fmax(a, b);  // for the stable form
    T abs_diff = fabs(diff);
    return max_ab + eps * log(1.0 + exp(-abs_diff));
}

// ═══════════════════════════════════════════════════════════════
// smooth_min(a, b, eps)
// ═══════════════════════════════════════════════════════════════
// smooth_min(a, b) = -smooth_max(-a, -b)

template<typename T>
T smooth_min(const T& a, const T& b, double eps = 1e-4) {
    return -smooth_max(-a, -b, eps);
}

// ═══════════════════════════════════════════════════════════════
// smooth_abs(x, eps)
// ═══════════════════════════════════════════════════════════════
// Approximates |x| with continuous second derivative.
//
// Uses: smooth_abs(x) = sqrt(x² + eps²)
//
// Properties:
//   - smooth_abs(x) ≥ |x| (overestimates by ≤ eps)
//   - smooth_abs(0) = eps (not exactly 0)
//   - ∂/∂x = x / sqrt(x² + eps²)
//   - ∂²/∂x² = eps² / (x² + eps²)^{3/2}  (bell-shaped, peaks at x=0)

template<typename T>
T smooth_abs(const T& x, double eps = 1e-6) {
    using stan::math::sqrt;
    return sqrt(x * x + eps * eps);
}

// ═══════════════════════════════════════════════════════════════
// smooth_relu(x, eps)  — smooth max(x, 0)
// ═══════════════════════════════════════════════════════════════
// Also known as "softplus":
//   smooth_relu(x) = eps * log(1 + exp(x/eps))
//
// Properties:
//   - Converges to max(x, 0) as eps → 0
//   - ∂/∂x = sigmoid(x/eps) = 1 / (1 + exp(-x/eps))
//   - ∂²/∂x² = sigmoid(x/eps) * (1 - sigmoid(x/eps)) / eps

template<typename T>
T smooth_relu(const T& x, double eps = 1e-4) {
    using stan::math::exp;
    using stan::math::log;

    T z = x / eps;
    // Numerically stable softplus
    T zv = stan::math::value_of_rec(z);
    if (zv > 20.0)  return x;                          // exp(z) dominates
    if (zv < -20.0) return eps * exp(z);                // exp(z) ≈ 0
    return eps * log(1.0 + exp(z));
}

// ═══════════════════════════════════════════════════════════════
// smooth_heaviside(x, eps)  — smooth step function
// ═══════════════════════════════════════════════════════════════
// Approximates H(x) = {0 if x < 0, 1 if x > 0} using sigmoid.
//
//   smooth_heaviside(x) = 1 / (1 + exp(-x/eps))
//
// Properties:
//   - ∂/∂x = smooth_heaviside(x) * (1 - smooth_heaviside(x)) / eps
//   - Useful for digital payoffs, barrier indicators

template<typename T>
T smooth_heaviside(const T& x, double eps = 1e-4) {
    using stan::math::exp;
    return 1.0 / (1.0 + exp(-x / eps));
}

// ═══════════════════════════════════════════════════════════════
// smooth_indicator(x, lo, hi, eps)  — smooth 1_{lo < x < hi}
// ═══════════════════════════════════════════════════════════════
// Approximates the indicator function for x ∈ (lo, hi).
//
//   smooth_indicator(x) = smooth_heaviside(x - lo) * smooth_heaviside(hi - x)

template<typename T>
T smooth_indicator(const T& x, double lo, double hi, double eps = 1e-4) {
    return smooth_heaviside(x - lo, eps) * smooth_heaviside(hi - x, eps);
}

// ═══════════════════════════════════════════════════════════════
// smooth_clamp(x, lo, hi, eps)
// ═══════════════════════════════════════════════════════════════
// Approximates clamp(x, lo, hi) = min(max(x, lo), hi)
//
// Uses nested smooth_max / smooth_min:
//   smooth_clamp(x) = smooth_min(smooth_max(x, lo, eps), hi, eps)

template<typename T>
T smooth_clamp(const T& x, double lo, double hi, double eps = 1e-4) {
    return smooth_min(smooth_max(x, T(lo), eps), T(hi), eps);
}

// ═══════════════════════════════════════════════════════════════
// smooth_sign(x, eps)
// ═══════════════════════════════════════════════════════════════
// Approximates sign(x) = {-1, 0, +1} smoothly.
//
//   smooth_sign(x) = tanh(x / eps)
//
// Or equivalently: 2 * smooth_heaviside(x, eps) - 1

template<typename T>
T smooth_sign(const T& x, double eps = 1e-4) {
    using stan::math::tanh;
    return tanh(x / eps);
}

// ═══════════════════════════════════════════════════════════════
// smooth_power(x, n, eps)  — smooth |x|^n * sign(x) for non-integer n
// ═══════════════════════════════════════════════════════════════
// For cases where pow(x, n) is needed but x can be near zero.
// Uses smooth_abs to avoid the kink.

template<typename T>
T smooth_power(const T& x, double n, double eps = 1e-8) {
    using stan::math::pow;
    return pow(smooth_abs(x, eps), n) * smooth_sign(x, eps);
}

} // namespace ad_math
```

### 9.2 Smooth Payoff Functions

Common derivative payoffs have kinks. Replace with smooth versions for AD:

```cpp
namespace ad_payoffs {

// ═══════════════════════════════════════════════════════════════
// smooth_call(S, K, eps)  — smooth max(S - K, 0)
// ═══════════════════════════════════════════════════════════════
// European call payoff. Uses softplus.
//
// The kink at S = K causes ∂²payoff/∂S² = δ(S-K) (Dirac delta),
// which AD represents as 0. Smoothing gives a finite bell-shaped gamma.

template<typename T>
T smooth_call(const T& S, double K, double eps = 1e-4) {
    return ad_math::smooth_relu(S - K, eps);
}

// ═══════════════════════════════════════════════════════════════
// smooth_put(S, K, eps)  — smooth max(K - S, 0)
// ═══════════════════════════════════════════════════════════════

template<typename T>
T smooth_put(const T& S, double K, double eps = 1e-4) {
    return ad_math::smooth_relu(K - S, eps);
}

// ═══════════════════════════════════════════════════════════════
// smooth_digital_call(S, K, eps)  — smooth 1_{S > K}
// ═══════════════════════════════════════════════════════════════
// Digital (binary) call payoff. Exact payoff is Heaviside(S - K).
// AD gives ∂/∂S = δ(S-K), ∂²/∂S² = δ'(S-K) — both useless.
// Smooth version gives finite, meaningful Greeks.

template<typename T>
T smooth_digital_call(const T& S, double K, double eps = 1e-4) {
    return ad_math::smooth_heaviside(S - K, eps);
}

// ═══════════════════════════════════════════════════════════════
// smooth_digital_put(S, K, eps)  — smooth 1_{S < K}
// ═══════════════════════════════════════════════════════════════

template<typename T>
T smooth_digital_put(const T& S, double K, double eps = 1e-4) {
    return ad_math::smooth_heaviside(K - S, eps);
}

// ═══════════════════════════════════════════════════════════════
// smooth_call_spread(S, K1, K2, eps)
// ═══════════════════════════════════════════════════════════════
// Approximates a tight call spread (K2 > K1), often used to
// approximate a digital payoff: [max(S-K1,0) - max(S-K2,0)] / (K2-K1)
//
// Using smooth calls, the spread is automatically smooth.

template<typename T>
T smooth_call_spread(const T& S, double K1, double K2, double eps = 1e-4) {
    return (smooth_call(S, K1, eps) - smooth_call(S, K2, eps)) / (K2 - K1);
}

// ═══════════════════════════════════════════════════════════════
// smooth_barrier(S, B, type, eps)
// ═══════════════════════════════════════════════════════════════
// Smooth barrier indicator.
//   UP_AND_IN:  smooth_heaviside(S - B)
//   UP_AND_OUT: smooth_heaviside(B - S)
//   DOWN_AND_IN:  smooth_heaviside(B - S)
//   DOWN_AND_OUT: smooth_heaviside(S - B)

enum class BarrierType { UP_AND_IN, UP_AND_OUT, DOWN_AND_IN, DOWN_AND_OUT };

template<typename T>
T smooth_barrier(const T& S, double B, BarrierType type, double eps = 1e-4) {
    switch (type) {
        case BarrierType::UP_AND_IN:
        case BarrierType::DOWN_AND_OUT:
            return ad_math::smooth_heaviside(S - B, eps);
        case BarrierType::UP_AND_OUT:
        case BarrierType::DOWN_AND_IN:
            return ad_math::smooth_heaviside(B - S, eps);
    }
    return T(0);
}

// ═══════════════════════════════════════════════════════════════
// smooth_range_accrual(S, lo, hi, eps)
// ═══════════════════════════════════════════════════════════════
// Smooth indicator for range accrual: 1 if lo < S < hi, 0 otherwise.

template<typename T>
T smooth_range_accrual(const T& S, double lo, double hi, double eps = 1e-4) {
    return ad_math::smooth_indicator(S, lo, hi, eps);
}

// ═══════════════════════════════════════════════════════════════
// smooth_collar(S, K_put, K_call, eps)
// ═══════════════════════════════════════════════════════════════
// Collar payoff: max(S - K_call, 0) - max(K_put - S, 0)
// Has kinks at both K_put and K_call.

template<typename T>
T smooth_collar(const T& S, double K_put, double K_call, double eps = 1e-4) {
    return smooth_call(S, K_call, eps) - smooth_put(S, K_put, eps);
}

// ═══════════════════════════════════════════════════════════════
// smooth_straddle(S, K, eps)
// ═══════════════════════════════════════════════════════════════
// |S - K| payoff. Kink at S = K.

template<typename T>
T smooth_straddle(const T& S, double K, double eps = 1e-4) {
    return ad_math::smooth_abs(S - K, eps);
}

} // namespace ad_payoffs
```

### 9.3 Choosing Epsilon

The smoothing parameter `eps` controls the trade-off between accuracy and differentiability:

| eps | Smoothing window | Max error vs exact | Hessian magnitude near kink |
|-----|------------------|--------------------|----------------------------|
| $10^{-2}$ | ~4 cents (if S in dollars) | ~0.7 cents | moderate |
| $10^{-3}$ | ~0.4 cents | ~0.07 cents | large |
| $10^{-4}$ | ~0.04 cents | ~0.007 cents | very large |
| $10^{-6}$ | negligible | negligible | extreme (may cause numerics) |

**Guidelines**:
- For payoff smoothing (call, put, digital): `eps` relative to notional. For a $100 underlying, `eps = 0.01` (1 cent) is typical.
- For barrier smoothing: `eps` relative to monitoring frequency. Daily monitoring ~ `eps = 0.001 * S`.
- For mathematical operations (abs, max in intermediate calcs): `eps = 1e-6` to `1e-8`.
- For Hessian stability: don't make `eps` too small — the second derivative scales as $1/\epsilon$, which can cause overflow or numerical noise.

### 9.4 When to Use Smooth vs Exact

| Operation | Use exact (Stan's built-in) | Use smooth |
|-----------|----------------------------|------------|
| `max(a, b)` in intermediate calc | If only gradient needed | If Hessian needed and $a \approx b$ is possible |
| `max(S - K, 0)` payoff | Never for Hessian | Always for Hessian |
| `abs(x)` for residual/error | Yes (not differentiated) | N/A |
| `abs(x)` in pricing formula | If only gradient needed | If Hessian needed |
| Barrier indicator `1_{S > B}` | Never for any Greeks | Always |
| `clamp(x, lo, hi)` | If only gradient needed | If Hessian needed |

**Stan Math's built-in `fmax`, `fmin`, `fabs`** are first-order differentiable (they select the correct subgradient) but their second derivatives are zero everywhere except at the kink (where they are undefined). This means the Hessian will have zero entries where you'd expect nonzero gamma/volga. Smooth versions fix this.

### 9.5 Conditional Logic in Pricers

Pricers often have conditional logic that depends on market data values:

```cpp
// PROBLEMATIC: branching on DoubleT value
template<typename DoubleT>
DoubleT priceTrade(ScenarioData<DoubleT>& sd) {
    DoubleT S = sd.getEQDSpot("AAPL").spot();
    DoubleT vol = sd.getEQDVol("AAPL").vol(T, K);

    // This branch depends on S — AD can't differentiate through the branch
    if (S > barrier) {          // BUG: comparison with DoubleT
        return payoff_above();
    } else {
        return payoff_below();
    }
}
```

**Problem**: When `DoubleT = fvar<var>`, the comparison `S > barrier` uses `value_of_rec(S) > barrier` — it branches on the current numeric value. The derivative is computed for whichever branch is taken, but the **switch between branches is not differentiated**. This means the gradient and Hessian are correct within each region but miss the jump at the boundary.

**Solution 1: smooth_heaviside blending**

```cpp
template<typename DoubleT>
DoubleT priceTrade(ScenarioData<DoubleT>& sd) {
    DoubleT S = sd.getEQDSpot("AAPL").spot();

    DoubleT above = payoff_above();
    DoubleT below = payoff_below();

    // Smooth blend — both branches are always evaluated
    DoubleT weight = ad_math::smooth_heaviside(S - barrier, eps);
    return weight * above + (1.0 - weight) * below;
}
```

Both branches are evaluated (costs 2× the compute), but the derivatives correctly capture the transition.

**Solution 2: accept the discontinuity**

For many pricers, the branch is far from the kink during normal market conditions (e.g., knock-in barrier is far from current spot). In this case, the AD Greeks are correct and the discontinuity is irrelevant. Only smooth if the branch point could be near current market data values.

### 9.6 Flooring and Capping in Rate Calculations

Interest rate pricers frequently floor rates at zero or cap them:

```cpp
// Common in rate pricers — kink at rate = 0
DoubleT floored_rate = stan::math::fmax(rate, 0.0);
DoubleT libor = stan::math::fmax(forward_rate, 0.0);  // LIBOR floor
DoubleT coupon = stan::math::fmin(rate, cap_rate);     // cap
```

For Hessian computation, replace with:

```cpp
DoubleT floored_rate = ad_math::smooth_max(rate, T(0.0), 1e-6);  // 0.1bp smoothing
DoubleT libor = ad_math::smooth_relu(forward_rate, 1e-6);
DoubleT coupon = ad_math::smooth_min(rate, T(cap_rate), 1e-6);
```

The `eps` should be small relative to typical rate values (1e-6 = 0.01bp).

---

## 10. `fvar<var>` Compatibility Requirements

### 10.1 What Works Automatically

Any code that uses standard C++ arithmetic and Stan-overloaded math functions:

```cpp
// Arithmetic: all work with fvar<var>
DoubleT a = x + y;
DoubleT b = x * y;
DoubleT c = x / y;
DoubleT d = x - y;
DoubleT e = -x;

// Math functions: Stan provides overloads for all of these
DoubleT f01 = stan::math::exp(x);
DoubleT f02 = stan::math::log(x);
DoubleT f03 = stan::math::sqrt(x);
DoubleT f04 = stan::math::pow(x, 2.0);
DoubleT f05 = stan::math::pow(x, y);       // both AD
DoubleT f06 = stan::math::Phi(x);          // normal CDF
DoubleT f07 = stan::math::inv_Phi(x);      // inverse normal CDF
DoubleT f08 = stan::math::erf(x);
DoubleT f09 = stan::math::erfc(x);
DoubleT f10 = stan::math::fabs(x);         // |x|, first-order differentiable
DoubleT f11 = stan::math::fmax(x, y);      // max, first-order differentiable
DoubleT f12 = stan::math::fmin(x, y);      // min, first-order differentiable
DoubleT f13 = stan::math::sin(x);
DoubleT f14 = stan::math::cos(x);
DoubleT f15 = stan::math::tan(x);
DoubleT f16 = stan::math::asin(x);
DoubleT f17 = stan::math::acos(x);
DoubleT f18 = stan::math::atan(x);
DoubleT f19 = stan::math::atan2(y, x);
DoubleT f20 = stan::math::sinh(x);
DoubleT f21 = stan::math::cosh(x);
DoubleT f22 = stan::math::tanh(x);
DoubleT f23 = stan::math::log1p(x);        // log(1+x), numerically stable
DoubleT f24 = stan::math::expm1(x);        // exp(x)-1, numerically stable
DoubleT f25 = stan::math::cbrt(x);         // cube root
DoubleT f26 = stan::math::square(x);       // x²
DoubleT f27 = stan::math::inv(x);          // 1/x
DoubleT f28 = stan::math::inv_sqrt(x);     // 1/√x
DoubleT f29 = stan::math::lgamma(x);       // log-gamma
DoubleT f30 = stan::math::digamma(x);      // ψ(x)
DoubleT f31 = stan::math::beta(x, y);      // beta function
DoubleT f32 = stan::math::lbeta(x, y);     // log-beta
DoubleT f33 = stan::math::gamma_p(a, x);   // regularized incomplete gamma
DoubleT f34 = stan::math::gamma_q(a, x);   // upper regularized incomplete gamma

// Comparisons: use value_of_rec to extract double for branching
if (stan::math::value_of_rec(x) > 0.0) { ... }
if (stan::math::value_of_rec(x) > stan::math::value_of_rec(y)) { ... }

// Eigen operations: work if scalar type is fvar<var>
Eigen::Matrix<DoubleT, Eigen::Dynamic, 1> v(3);
v(0) = x; v(1) = y; v(2) = z;
DoubleT norm = v.norm();
DoubleT dot = v.dot(w);
Eigen::Matrix<DoubleT, Eigen::Dynamic, Eigen::Dynamic> M;
Eigen::Matrix<DoubleT, Eigen::Dynamic, 1> result = M * v;
```

### 10.2 What Breaks and How to Fix

| Pattern | Problem | Fix |
|---------|---------|-----|
| `double d = someVar;` | Implicit conversion strips AD info | `DoubleT d = someVar;` |
| `std::exp(x)` | `std::exp` not overloaded for `fvar<var>` | `using stan::math::exp; exp(x);` |
| `std::max(x, y)` | `std::max` uses `<` which may not work | `stan::math::fmax(x, y)` |
| `std::min(x, y)` | Same issue | `stan::math::fmin(x, y)` |
| `std::abs(x)` | Not overloaded for `fvar<var>` | `stan::math::fabs(x)` |
| `(double)x` | Strips AD | `stan::math::value_of_rec(x)` for branch only |
| `static_cast<double>(x)` | Strips AD | Same as above |
| `x > 0 ? a : b` | May not compile for `fvar<var>` | `stan::math::value_of_rec(x) > 0 ? a : b` |
| `for (int i = 0; i < x; ++i)` | Loop bound can't be `fvar<var>` | Use `value_of_rec(x)` for loop bound |
| `pow(x, n)` with `int n` | May call wrong overload | `stan::math::pow(x, static_cast<double>(n))` |
| `x == 0.0` | Exact equality undefined for AD | `stan::math::value_of_rec(x) == 0.0` |
| Assignment to `double&` | Can't assign `fvar<var>` to `double` ref | Template the output type |
| `printf("%f", x)` | Can't print `fvar<var>` | `printf("%f", stan::math::value_of_rec(x))` |
| External library (LAPACK, etc.) | Expects `double*` arrays | See Section 10.4 |

### 10.3 ADTemplate Checklist

Each ADTemplate class should verify these properties for `fvar<var>` compatibility:

**Data storage:**
- [ ] Vol surface / curve values stored as `DoubleT` (not `double`)
- [ ] Grid coordinates (strikes, maturities, tenors) stored as `double` (correct — not AD'd)
- [ ] Spot price stored as `DoubleT`
- [ ] No `double` member that should be `DoubleT`

**Computation:**
- [ ] All interpolation methods return `DoubleT`
- [ ] All forward/discount calculations return `DoubleT`
- [ ] `exp`, `log`, `sqrt` etc. use `stan::math::` or `using` declarations
- [ ] No `std::max`, `std::min`, `std::abs` on `DoubleT` values
- [ ] No implicit conversion to `double` in computation path
- [ ] No external library calls in computation path (or properly wrapped)

**Control flow:**
- [ ] All branches use `value_of_rec()` for the comparison, not raw `DoubleT`
- [ ] Loop bounds are `int` or `double`, never `DoubleT`
- [ ] Index calculations use `int`, not `DoubleT`

**Construction:**
- [ ] Can be constructed from `DoubleT` values + `double` grid coordinates
- [ ] Constructor does not perform arithmetic that requires `double` (e.g., computing spline coefficients from `DoubleT` values — the coefficients should be `DoubleT`)

### 10.4 Wrapping External Libraries

If a pricer calls an external library (e.g., a PDE solver, a numerical integrator, or a special function library written in C/Fortran), the library expects `double*`. You cannot pass `fvar<var>*`.

**Strategy**: Extract numeric values, call the library, then use `precomputed_gradients` to create a `var` node with analytically known derivatives:

```cpp
template<typename DoubleT>
DoubleT call_external_solver(const DoubleT& S, const DoubleT& sigma,
                              const DoubleT& r, double K, double T) {
    // Extract numeric values
    double S_val = stan::math::value_of_rec(S);
    double sig_val = stan::math::value_of_rec(sigma);
    double r_val = stan::math::value_of_rec(r);

    // Call external library (double only)
    double price = external_pde_pricer(S_val, sig_val, r_val, K, T);

    // If DoubleT == double, just return
    if constexpr (std::is_same_v<DoubleT, double>) {
        return price;
    } else {
        // Compute analytical or FD first-order derivatives
        double dP_dS = fd_derivative([&](double s) {
            return external_pde_pricer(s, sig_val, r_val, K, T);
        }, S_val);
        double dP_dsig = fd_derivative([&](double s) {
            return external_pde_pricer(S_val, s, r_val, K, T);
        }, sig_val);
        double dP_dr = fd_derivative([&](double rr) {
            return external_pde_pricer(S_val, sig_val, rr, K, T);
        }, r_val);

        // Create AD node with known derivatives
        return stan::math::precomputed_gradients(
            price, {S, sigma, r}, {dP_dS, dP_dsig, dP_dr});
    }
}
```

**Important limitation**: This approach gives correct first-order derivatives but the **Hessian entries involving this external call will be zero** (because the precomputed gradients are `double`, not `var` — they don't have their own AD tape). For full Hessian support through an external library call, you would need to also compute second-order derivatives externally and use a more complex wrapping strategy (e.g., returning `fvar<var>` with manually set tangent).

### 10.5 Handling `using` Declarations Systematically

To make all 100+ pricers AD-compatible without modifying each one individually, add `using` declarations to a central header that all pricers include:

```cpp
// ad_using.h — include this in all pricer/ADTemplate headers
// Brings Stan math functions into scope so that unqualified calls
// (exp, log, sqrt, etc.) resolve to Stan overloads for any DoubleT.

#pragma once

// Only bring these into the ad_math namespace, not globally
namespace ad_using {
    using stan::math::exp;
    using stan::math::log;
    using stan::math::log2;
    using stan::math::log10;
    using stan::math::log1p;
    using stan::math::sqrt;
    using stan::math::cbrt;
    using stan::math::pow;
    using stan::math::fabs;
    using stan::math::fmax;
    using stan::math::fmin;
    using stan::math::sin;
    using stan::math::cos;
    using stan::math::tan;
    using stan::math::asin;
    using stan::math::acos;
    using stan::math::atan;
    using stan::math::atan2;
    using stan::math::sinh;
    using stan::math::cosh;
    using stan::math::tanh;
    using stan::math::erf;
    using stan::math::erfc;
    using stan::math::Phi;
    using stan::math::inv_Phi;
    using stan::math::expm1;
    using stan::math::square;
} // namespace ad_using
```

Pricers can then do `using namespace ad_using;` inside their methods. This is cleaner than modifying every `exp()` call across 100+ files.

**Alternative**: If pricers already use `std::exp` etc., and you don't want to touch them, a compile-time approach is to create an `ADDouble<T>` wrapper that provides implicit conversion from `T` and overloads all operators/functions. But this is more complex.

---

## 11. Interpolation Under AD

Interpolation is critical — it's how vol surfaces and yield curves produce `DoubleT` values from `DoubleT` node data. The interpolation must be differentiable.

### 11.1 Linear Interpolation

For linear interpolation between nodes $(x_i, y_i)$ and $(x_{i+1}, y_{i+1})$ at point $x$:

$$y(x) = y_i \cdot \frac{x_{i+1} - x}{x_{i+1} - x_i} + y_{i+1} \cdot \frac{x - x_i}{x_{i+1} - x_i}$$

- $x_i, x_{i+1}$ are grid coordinates (`double`) — not differentiated
- $y_i, y_{i+1}$ are node values (`DoubleT`) — the AD leaves
- $x$ is the query point (`double` for vol lookup by maturity/strike)
- Result is `DoubleT`

**First derivative** $\partial y / \partial y_i$: the interpolation weight $(x_{i+1} - x) / (x_{i+1} - x_i)$ — a `double` constant. Correct and exact under AD.

**Second derivative** $\partial^2 y / \partial y_i \partial y_j$: zero (linear interpolation is linear in the node values). This is correct — linear interpolation has no second-order cross terms between nodes.

**AD compatibility**: Perfect. No issues.

### 11.2 Cubic Spline Interpolation

Cubic splines produce values as:

$$y(x) = a_i + b_i(x - x_i) + c_i(x - x_i)^2 + d_i(x - x_i)^3$$

where the coefficients $a_i, b_i, c_i, d_i$ are computed from the node values $y_0, \ldots, y_n$ by solving a tridiagonal system.

**Key question**: Are the spline coefficients `DoubleT` or `double`?

They **must be `DoubleT`**. The coefficients depend on the node values (which are `DoubleT` AD leaves), so the coefficient computation (tridiagonal solve) must operate in `DoubleT` arithmetic. If coefficients are computed as `double`, the AD tape is broken.

**Tridiagonal solve in DoubleT**: Thomas algorithm works with any scalar type:

```cpp
template<typename T>
void tridiagonal_solve(const std::vector<double>& a,    // sub-diagonal (double — from grid)
                        const std::vector<double>& b,    // diagonal (double — from grid)
                        const std::vector<double>& c,    // super-diagonal (double — from grid)
                        std::vector<T>& d,                // RHS (DoubleT — from node values)
                        std::vector<T>& x)               // solution (DoubleT)
{
    int n = d.size();
    std::vector<double> c_prime(n);    // modified coefficients (double)
    std::vector<T> d_prime(n);          // modified RHS (DoubleT)

    c_prime[0] = c[0] / b[0];
    d_prime[0] = d[0] / b[0];

    for (int i = 1; i < n; ++i) {
        double m = b[i] - a[i] * c_prime[i-1];  // double (grid-only)
        c_prime[i] = c[i] / m;
        d_prime[i] = (d[i] - a[i] * d_prime[i-1]) / m;
    }

    x[n-1] = d_prime[n-1];
    for (int i = n-2; i >= 0; --i)
        x[i] = d_prime[i] - c_prime[i] * x[i+1];
}
```

Note: The tridiagonal matrix coefficients (`a`, `b`, `c`) come from grid spacing (double), but the RHS (`d`) and solution (`x`) are `DoubleT` because they depend on node values. The division `d[i] / b[i]` is `DoubleT / double`, which works correctly.

### 11.3 Bicubic Interpolation for Vol Surfaces

For 2D bicubic interpolation on a vol surface:

1. For each maturity row, interpolate along the strike axis → `DoubleT` intermediate values
2. Interpolate along the maturity axis using the intermediate values → `DoubleT` result

Each step is a 1D cubic spline in `DoubleT`. The cascade correctly propagates AD through both dimensions.

**Hessian implications**: The vol at a given (T, K) depends on ~16 surrounding grid nodes (4×4 stencil for bicubic). The gradient has 16 nonzero entries. The Hessian has up to 16×16 = 256 nonzero entries in the vol-vol block — but most are small.

---

## 12. Summary of Changes

### New Code (additive, no existing code modified)

| Component | What | Lines (est.) |
|-----------|------|-------------|
| `ScenarioData` | `buildSensitivityRegistry()` | ~30 |
| `ScenarioData` | `packMarketData()` | ~60 |
| `ScenarioData` | `loadFromVector()` | ~80 |
| `ScenarioData` | Dual-source getters | ~15 per type × 6 types |
| `ScenarioData` | Node-mappings constructor (non-copy) | ~20 |
| `HessianFunctor` | Functor template + factory | ~30 |
| Pricers | `rebind<T>()` + converting constructor | ~10–20 per pricer |
| `calculateSensitivities` | Unified interface | ~30 |
| `extractStructured` | Result unpacking + cross blocks | ~120 |
| `SubsetFunctor` | Sparse Hessian support | ~40 |
| `ad_math` namespace | Smooth approximations | ~150 |
| `ad_payoffs` namespace | Smooth payoff functions | ~100 |
| `ad_using.h` | Centralized `using` declarations | ~30 |
| `parallel_hessian` | Optional parallel implementation | ~50 |
| Analytical primitives | BS, digital, Bachelier, spline (3 overloads each) | ~80 per function |
| **Total** | | **~850 lines + primitives** |

### Modified (Minimal, Per-Pricer)

- Each pricer needs a **converting constructor** and **`rebind<T>()`** method (Section 4.4). For stateless pricers this is ~5 lines; for path-dependent pricers with `DoubleT` state, ~10–20 lines. This is the only pricer-level change.

### Not Modified

- Pricer business logic (`priceTrade`, `loadScenarioData`) — unchanged
- Any generator
- Any ADTemplate (assuming they already use `DoubleT` correctly — if not, fixes are one-time per class)
- The MC simulation loop structure (Hessian calls are inserted alongside existing pricing calls)
- The interpolation library (assuming it's already templated on `DoubleT`)

### Template Instantiations Required

The following must compile with `T = fvar<var>`:

- `ScenarioData<fvar<var>>`
- All ADTemplates: `EQDVol<fvar<var>>`, `EQDSpot<fvar<var>>`, `IRCurve<fvar<var>>`, `YieldCurve<fvar<var>>`, `FXRate<fvar<var>>`, `SurvivalCurve<fvar<var>>`
- All pricers: `PricerA<fvar<var>>`, `PricerB<fvar<var>>`, ...
- All interpolation classes with `fvar<var>` scalars
- All helper functions used by pricers

If compilation fails, the fix is always in that specific class — typically one of:
1. `std::exp` → `stan::math::exp` (or add `using`)
2. `double d = someVar` → `DoubleT d = someVar`
3. `std::max` → `stan::math::fmax`
4. Comparison `x > 0` → `value_of_rec(x) > 0`

These are one-time fixes. Once a class compiles with `fvar<var>`, it works for all future uses.

---

## Appendix A: Stan Math Include Requirements

```cpp
// Core AD types
#include <stan/math/rev/core.hpp>              // var, grad(), nested_rev_autodiff
#include <stan/math/fwd/core.hpp>              // fvar

// Hessian and gradient functors
#include <stan/math/mix/functor/hessian.hpp>   // stan::math::hessian
#include <stan/math/rev/functor/gradient.hpp>  // stan::math::gradient

// Math function overloads (include what your pricers use)
#include <stan/math/rev/fun/exp.hpp>
#include <stan/math/rev/fun/log.hpp>
#include <stan/math/rev/fun/sqrt.hpp>
#include <stan/math/rev/fun/pow.hpp>
#include <stan/math/rev/fun/Phi.hpp>           // normal CDF
#include <stan/math/rev/fun/inv_Phi.hpp>       // inverse normal CDF
#include <stan/math/rev/fun/erf.hpp>
#include <stan/math/rev/fun/erfc.hpp>
#include <stan/math/rev/fun/fabs.hpp>
#include <stan/math/rev/fun/fmax.hpp>
#include <stan/math/rev/fun/fmin.hpp>
#include <stan/math/rev/fun/tanh.hpp>
#include <stan/math/rev/fun/log1p.hpp>
#include <stan/math/rev/fun/expm1.hpp>

// Eigen integration
#include <stan/math/prim/fun/Eigen.hpp>

// Utility
#include <stan/math/prim/fun/value_of_rec.hpp>
```

Note: Stan Math 4.9.0 with Apple Clang 17 has build issues with `#include <stan/math.hpp>`. Use targeted includes as above.

## Appendix B: Relationship Between Gradient, Hessian, and Greek Names

| Greek | Mathematical | Gradient/Hessian location | Description |
|-------|-------------|--------------------------|-------------|
| Delta | $\partial P / \partial S$ | `grad[spot_idx]` | First-order spot |
| Gamma | $\partial^2 P / \partial S^2$ | `H[spot_idx][spot_idx]` | Second-order spot |
| Vega | $\partial P / \partial \sigma_{i,j}$ | `grad[vol_idx]` | First-order per vol node |
| Volga | $\partial^2 P / \partial \sigma_i \partial \sigma_j$ | `H[vol_idx_i][vol_idx_j]` | Vol-vol cross |
| Vanna | $\partial^2 P / \partial S \partial \sigma_i$ | `H[spot_idx][vol_idx_i]` | Spot-vol cross |
| Rho | $\partial P / \partial r_i$ | `grad[curve_idx]` | First-order per rate node |
| Curve Gamma | $\partial^2 P / \partial r_i \partial r_j$ | `H[curve_i][curve_j]` | Rate-rate cross |
| Spot-Rate | $\partial^2 P / \partial S \partial r_i$ | `H[spot_idx][curve_i]` | Spot-rate cross |
| Vega-Rho | $\partial^2 P / \partial \sigma_i \partial r_j$ | `H[vol_i][curve_j]` | Vol-rate cross |
| FX Delta | $\partial P / \partial \text{FX}$ | `grad[fx_idx]` | First-order FX |
| FX Gamma | $\partial^2 P / \partial \text{FX}^2$ | `H[fx_idx][fx_idx]` | Second-order FX |
| Quanto | $\partial^2 P / \partial S \partial \text{FX}$ | `H[spot_idx][fx_idx]` | Spot-FX cross |

## Appendix C: Verifying Hessian Correctness

### Symmetry Check

```cpp
double max_asymmetry = (H - H.transpose()).cwiseAbs().maxCoeff();
double rel_asymmetry = max_asymmetry / H.cwiseAbs().maxCoeff();
assert(rel_asymmetry < 1e-10);  // should be ~1e-14 for exact AD
```

### Finite-Difference Cross-Check

```cpp
void verify_hessian(const auto& functor, const Eigen::VectorXd& x,
                     const Eigen::VectorXd& grad, const Eigen::MatrixXd& H,
                     int n_checks = 10) {
    int N = x.size();
    double h = 1e-5;

    std::mt19937 rng(42);
    std::uniform_int_distribution<int> dist(0, N - 1);

    for (int check = 0; check < n_checks; ++check) {
        int i = dist(rng);
        int j = dist(rng);

        // Gradient check: central FD
        Eigen::VectorXd x_p = x, x_m = x;
        x_p(i) += h; x_m(i) -= h;
        double fd_grad = (functor(x_p) - functor(x_m)) / (2 * h);
        double grad_err = std::abs(fd_grad - grad(i)) / std::max(1.0, std::abs(grad(i)));
        assert(grad_err < 1e-5);

        // Hessian diagonal check
        double fd_H_ii = (functor(x_p) - 2 * functor(x) + functor(x_m)) / (h * h);
        double hess_err = std::abs(fd_H_ii - H(i, i)) / std::max(1.0, std::abs(H(i, i)));
        assert(hess_err < 1e-3);

        // Hessian off-diagonal check
        if (i != j) {
            Eigen::VectorXd x_pp = x, x_pm = x, x_mp = x, x_mm = x;
            x_pp(i) += h; x_pp(j) += h;
            x_pm(i) += h; x_pm(j) -= h;
            x_mp(i) -= h; x_mp(j) += h;
            x_mm(i) -= h; x_mm(j) -= h;
            double fd_H_ij = (functor(x_pp) - functor(x_pm) - functor(x_mp) + functor(x_mm))
                              / (4 * h * h);
            double cross_err = std::abs(fd_H_ij - H(i, j)) / std::max(1.0, std::abs(H(i, j)));
            assert(cross_err < 1e-3);
        }
    }
}
```

### Common Failure Modes

| Symptom | Likely cause | Fix |
|---------|-------------|-----|
| Hessian is all zeros | `double` cast somewhere breaking tape | Find and fix the cast |
| Hessian is non-symmetric | Bug in AD (unlikely) or numerical issue | Check for very large values, reduce eps |
| FD and AD disagree at specific node | Non-differentiable operation near that node | Use smooth approximation |
| FD and AD agree for gradient but not Hessian | `fmax`/`fmin` zeroing second derivative | Use `smooth_max`/`smooth_min` |
| Hessian has NaN/Inf | Division by zero, log of zero, sqrt of negative | Add floors: `log(fmax(x, 1e-15))` |
| Gradient correct, Hessian row all zero for node $i$ | Node $i$ enters only linearly (correct!) or tape is broken at that node | Verify: if linear, zero Hessian row is correct |
