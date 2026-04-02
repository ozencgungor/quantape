# Multi-Factor Markovian Bergomi: Calibration & Pricing

## 1. Model Specification

### 1.1 Forward Variance Dynamics

The instantaneous forward variance curve evolves as:

$$\xi_t(T) = \xi_0(T) \cdot \exp\!\Bigl(\sum_{k=1}^{K} w_k\, e^{-\kappa_k(T-t)}\, X_k(t) \;-\; \tfrac{1}{2}\,\Sigma(t, T)\Bigr)$$

where $X_k$ are Ornstein-Uhlenbeck factors driven by a **shared** vol Brownian $W^v$:

$$dX_k(t) = -\kappa_k\, X_k(t)\, dt + dW^v(t), \qquad X_k(0) = 0$$

The variance correction $\Sigma(t, T)$ ensures $E[\xi_t(T)] = \xi_0(T)$ (martingale condition). Since all factors share the same Brownian, they are correlated:

$$\text{Cov}[X_j(t),\, X_k(t)] = \frac{1 - e^{-(\kappa_j + \kappa_k)t}}{\kappa_j + \kappa_k}$$

The correction is the variance of the full exponent, **including cross terms**:

$$\Sigma(t, T) = \sum_{j,k} w_j\, w_k\, e^{-(\kappa_j + \kappa_k)(T-t)}\,\text{Cov}[X_j(t),\, X_k(t)]$$

Note: The diagonal terms ($j = k$) recover $w_k^2\, e^{-2\kappa_k(T-t)}\, \frac{1 - e^{-2\kappa_k t}}{2\kappa_k}$, but the off-diagonal terms are essential for correctness when $K \geq 2$.

### 1.2 Spot Variance

Setting $T = t$ in the forward variance gives the instantaneous spot variance:

$$V(t) = \xi_0(t) \cdot \exp\!\Bigl(\sum_k w_k\, X_k(t) - \tfrac{1}{2}\sum_k w_k^2\, \sigma_k^2(t)\Bigr)$$

where $\sigma_k^2(t) = \text{Var}[X_k(t)]$.

### 1.3 Stock Price

$$d\ln S = (r - q - \tfrac{1}{2}V)\,dt + \sqrt{V}\bigl(\rho\, dW^v + \sqrt{1-\rho^2}\, dW^\perp\bigr)$$

Integrated over $[0, T]$:

$$\ln\frac{S(T)}{F(T)} = -\tfrac{1}{2}\,\mathcal{I} + \rho\,\mathcal{M} + \sqrt{1-\rho^2}\;\mathcal{G}$$

where $F(T) = S_0\, e^{(r-q)T}$ is the forward, and:

| Symbol | Definition | Nature |
|--------|-----------|--------|
| $\mathcal{I}$ | $\int_0^T V(t)\,dt$ | Integrated variance (random) |
| $\mathcal{M}$ | $\int_0^T \sqrt{V(t)}\,dW^v(t)$ | Stochastic integral (random) |
| $\mathcal{G}$ | $\int_0^T \sqrt{V(t)}\,dW^\perp(t)$ | Independent noise (random) |

**Key property**: Conditional on $(\mathcal{I}, \mathcal{M})$, the term $\mathcal{G}$ is Gaussian with variance $\mathcal{I}$, independent of $\mathcal{M}$.

### 1.4 Model Parameters

| Parameter | Role | Typical range | Count |
|-----------|------|---------------|-------|
| $\xi_0(T)$ | Forward variance curve | From market | Curve (not optimized) |
| $K$ | Number of factors | 2-3 | Fixed |
| $\kappa_k$ | Mean-reversion speeds | $[0.1, 10]$ | $K$ |
| $w_k$ | Factor weights | $[0.1, 5]$ | $K$ |
| $\eta_k$ | Vol-of-vol per factor | Constrained (see 1.5) | 0 (derived) |
| $\rho$ | Spot-vol correlation | $[-0.95, -0.3]$ | 1 |

**Total free parameters**: $2K + 1$ (for $K = 2$: **5 parameters**).

### 1.5 Reparametrization: $\eta_k = w_k$

In the standard multi-factor Bergomi, the factors are **not** independently scaled. Since all $X_k$ share the same $W^v$ and the model is specified by how they enter the forward variance exponent, we set $\eta_k = w_k$ (absorbed into the weight). The vol-of-vol of the forward variance at maturity $T$ observed from time $t$ is then:

$$\text{vol-of-vol}^2(t, T) = \sum_k w_k^2\, e^{-2\kappa_k(T-t)}$$

This matches the Bergomi (2005) convention and avoids parameter redundancy.

With this convention, the OU dynamics become:

$$dX_k = -\kappa_k X_k\,dt + w_k\,dW^v$$

and $\sigma_k^2(t) = \frac{w_k^2}{2\kappa_k}(1 - e^{-2\kappa_k t})$.

---

## 2. Forward Variance Curve Extraction

### 2.1 From Variance Swap Rates

The model-free variance swap rate for maturity $T$ is:

$$\sigma_{VS}^2(T) = \frac{2}{T}\int_0^\infty \frac{C(K,T) - (F-K)^+}{K^2}\,dK$$

The forward variance is:

$$\xi_0(T) = \frac{d}{dT}\bigl[T\,\sigma_{VS}^2(T)\bigr]$$

### 2.2 Numerical Procedure

1. **For each maturity $T_j$** on the market grid:
   - Convert market implied vols $\sigma_{\text{mkt}}(K_i, T_j)$ to call prices $C(K_i, T_j)$ via Black-Scholes.
   - Integrate $\int [C(K) - (F-K)^+] / K^2\,dK$ using **adaptive Simpson** on the available strike range.
   - **Wing extrapolation**: Beyond the last quoted strike, use Roger Lee's moment formula:
     $$\sigma^2(k) \to \beta_R |k| / T \quad \text{as } k \to +\infty$$
     $$\sigma^2(k) \to \beta_L |k| / T \quad \text{as } k \to -\infty$$
     where $k = \ln(K/F)$ and $\beta_{R/L} \leq 2$ (no-arbitrage bound). Fit $\beta$ from the last 2-3 quoted strikes.

2. **Build the total variance function** $w(T) = T \cdot \sigma_{VS}^2(T)$:
   - Fit a **monotone cubic Hermite spline** (Fritsch-Carlson) through $(T_j, w_j)$.
   - Monotonicity ensures $\xi_0(T) \geq 0$ (calendar spread arbitrage-free).

3. **Differentiate analytically**:
   $$\xi_0(T) = w'(T)$$
   from the spline's piecewise polynomial derivative. No finite-difference noise.

### 2.3 From Option Prices Directly (if no variance swap quotes)

If only European options are available, use **Dupire's formula** in implied vol space:

$$\xi_0(T) \approx \sigma_{\text{ATM}}^2(T) + 2T\,\sigma_{\text{ATM}}(T)\,\frac{\partial\sigma_{\text{ATM}}}{\partial T}$$

evaluated at $K = F(T)$ (ATM forward). Compute $\partial\sigma/\partial T$ from the splined implied vol surface.

---

## 3. European Option Pricing Engine

This is the inner loop of calibration. We need **sub-vol-bp accuracy** and **< 5ms per option**.

### 3.1 Conditional Lognormal Decomposition

Conditional on $(\mathcal{I}, \mathcal{M})$, the payoff expectation is analytic:

$$E\bigl[(S(T)-K)^+\,\big|\,\mathcal{I},\mathcal{M}\bigr] = \text{BS}\!\bigl(F\,e^{\rho\mathcal{M} - \frac{1}{2}\rho^2\mathcal{I}},\; K,\; (1-\rho^2)\mathcal{I}\bigr)$$

where $\text{BS}(F', K, w)$ is the undiscounted Black-Scholes formula with total variance $w$:

$$\text{BS}(F', K, w) = F'\,\Phi(d_+) - K\,\Phi(d_-), \qquad d_\pm = \frac{\ln(F'/K) \pm \frac{1}{2}w}{\sqrt{w}}$$

Therefore:

$$C(K, T) = e^{-rT}\,E\!\Big[\text{BS}\!\Big(F\,e^{\rho\mathcal{M}-\frac{1}{2}\rho^2\mathcal{I}},\; K,\; (1-\rho^2)\mathcal{I}\Big)\Big]$$

The expectation is over the joint distribution of $(\mathcal{I}, \mathcal{M})$ induced by the factor dynamics.

### 3.2 Characteristic Function Approach

Define the log-stock characteristic function:

$$\varphi(u) = E\!\bigl[e^{iu\ln(S(T)/F)}\bigr] = E\!\bigl[e^{-\alpha\,\mathcal{I} + \beta\,\mathcal{M}}\bigr]$$

where:

$$\alpha = \tfrac{1}{2}\bigl(iu + u^2(1-\rho^2)\bigr), \qquad \beta = iu\rho$$

The **independent noise** $\mathcal{G}$ has been integrated out analytically (it contributes the $u^2(1-\rho^2)$ term).

### 3.3 Recursive Gauss-Hermite Computation of $\varphi(u)$

Discretize $[0, T]$ into $N$ steps at times $0 = t_0 < t_1 < \cdots < t_N = T$.

Define the backward value function on the factor state $\mathbf{x} = (x_1, \ldots, x_K)$:

$$\Psi_n(\mathbf{x}) = E\!\bigl[e^{-\alpha\,\mathcal{I}_{n:N} + \beta\,\mathcal{M}_{n:N}}\,\big|\,\mathbf{X}(t_n) = \mathbf{x}\bigr]$$

**Terminal condition**: $\Psi_N(\mathbf{x}) = 1$.

**Backward step**: From $\Psi_{n+1}$ to $\Psi_n$:

$$\Psi_n(\mathbf{x}) = E_{Z_v}\!\Big[\exp\!\bigl(-\alpha\,\Delta\mathcal{I}_n(Z_v, \mathbf{x}) + \beta\,\Delta\mathcal{M}_n(Z_v, \mathbf{x})\bigr)\;\Psi_{n+1}(\mathbf{x}'(Z_v))\Big]$$

where $Z_v \sim N(0,1)$ is the single Gaussian driving all factors over $[t_n, t_{n+1}]$:

$$x'_k = x_k\,e^{-\kappa_k\Delta_n} + w_k\,\sqrt{\frac{1 - e^{-2\kappa_k\Delta_n}}{2\kappa_k}}\;Z_v$$

The expectation is a **1D Gauss-Hermite quadrature** (regardless of $K$):

$$\Psi_n(\mathbf{x}) \approx \frac{1}{\sqrt{\pi}} \sum_{q=1}^{N_{GH}} w_q^{GH}\;\exp\!\bigl(-\alpha\,\Delta\mathcal{I}_n + \beta\,\Delta\mathcal{M}_n\bigr)\;\Psi_{n+1}(\mathbf{x}')$$

with $Z_v = \sqrt{2}\,z_q^{GH}$ (standard Gauss-Hermite nodes/weights).

#### 3.3.1 Within-Step Integrated Variance $\Delta\mathcal{I}_n$

Given factor endpoints $\mathbf{x}$ (at $t_n$) and $\mathbf{x}'$ (at $t_{n+1}$), compute:

$$\Delta\mathcal{I}_n = \int_{t_n}^{t_{n+1}} V(t)\,dt$$

using **5-point Gauss-Legendre quadrature**. At each GL node $u \in [t_n, t_{n+1}]$:

1. **OU bridge conditional mean** for each factor:
   $$\bar{X}_k(u) = x_k\,\frac{\sinh(\kappa_k(t_{n+1}-u))}{\sinh(\kappa_k\Delta_n)} + x'_k\,\frac{\sinh(\kappa_k(u-t_n))}{\sinh(\kappa_k\Delta_n)}$$

2. **Marginal variance** at time $u$:
   $$\sigma_k^2(u) = \frac{w_k^2}{2\kappa_k}\bigl(1 - e^{-2\kappa_k u}\bigr)$$

3. **Spot variance** at the bridge mean:
   $$V(u) = \xi_0(u)\,\exp\!\Bigl(\sum_k w_k\,\bar{X}_k(u) - \tfrac{1}{2}\sum_k w_k^2\,\sigma_k^2(u)\Bigr)$$

4. **Quadrature**:
   $$\Delta\mathcal{I}_n \approx \frac{\Delta_n}{2}\sum_{g=1}^{5} w_g^{GL}\;V(u_g)$$

**Bridge variance correction**: The OU bridge has residual variance at intermediate points:

$$\text{Var}_{\text{bridge}}[X_k(u)] = \frac{w_k^2}{2\kappa_k}\,\frac{\sinh(\kappa_k(u - t_n))\,\sinh(\kappa_k(t_{n+1} - u))}{\sinh(\kappa_k\Delta_n)}$$

Since $V$ is the exponential of a Gaussian, the conditional expectation $E[V(u)\,|\,\mathbf{x}, \mathbf{x}']$ includes a moment-generating correction:

$$E[V(u)\,|\,\mathbf{x}, \mathbf{x}'] = \xi_0(u)\,\exp\!\Bigl(\sum_k w_k\,\bar{X}_k(u) - \tfrac{1}{2}\sum_k w_k^2\,\sigma_k^2(u) + \tfrac{1}{2}\sum_k w_k^2\,\text{Var}_{\text{bridge},k}(u)\Bigr)$$

The correction is subtle: $-\frac{1}{2}\sigma_k^2(u)$ is the marginal variance correction (ensures $E[V(u)] = \xi_0(u)$), while $+\frac{1}{2}\text{Var}_{\text{bridge},k}(u)$ accounts for the residual bridge uncertainty.

**Simplification**: Define $\gamma_k(u) = \sigma_k^2(u) - \text{Var}_{\text{bridge},k}(u)$. Then:

$$E[V(u)\,|\,\mathbf{x}, \mathbf{x}'] = \xi_0(u)\,\exp\!\Bigl(\sum_k w_k\,\bar{X}_k(u) - \tfrac{1}{2}\sum_k w_k^2\,\gamma_k(u)\Bigr)$$

Note: $\gamma_k(u)$ depends only on $\kappa_k$, $w_k$, $t_n$, $t_{n+1}$, $u$ — it can be **precomputed** for each GL node.

#### 3.3.2 Within-Step Stochastic Integral $\Delta\mathcal{M}_n$

The stochastic integral $\Delta\mathcal{M}_n = \int_{t_n}^{t_{n+1}} \sqrt{V(t)}\,dW^v$ is correlated with $Z_v$.

**Decomposition**: Split into the part explained by the step's Brownian increment and a residual:

$$\Delta\mathcal{M}_n = \mu_{\mathcal{M}}(Z_v)\;+\;\epsilon_{\mathcal{M}}$$

where $\epsilon_{\mathcal{M}}$ is independent of $Z_v$ (and of the factor endpoints).

**Leading-order approximation** (trapezoidal, error $O(\Delta^2)$ per step):

$$\Delta\mathcal{M}_n \approx \tfrac{1}{2}\bigl(\sqrt{V(t_n)} + \sqrt{V(t_{n+1})}\bigr)\;\sqrt{\Delta_n}\;Z_v$$

**Higher-accuracy approximation** (GL-based):

Using the same bridge interpolation as for $\Delta\mathcal{I}_n$:

$$\mu_{\mathcal{M}}(Z_v) = \Bigl(\frac{\Delta_n}{2}\sum_{g=1}^{5} w_g^{GL}\;\sqrt{E[V(u_g)\,|\,\mathbf{x},\mathbf{x}']}\Bigr)\;\frac{Z_v}{\sqrt{\Delta_n}}$$

This uses the identity that $\int f(t)\,dW(t) \approx \bar{f} \cdot \Delta W$ where $\bar{f}$ is the path average and $\Delta W = \sqrt{\Delta}\,Z_v$.

**Residual variance**:

$$\text{Var}[\epsilon_{\mathcal{M}}] = E[\Delta\mathcal{I}_n\,|\,\mathbf{x},\mathbf{x}'] - \mu_{\mathcal{M}}^2/\Delta_n$$

For the characteristic function, the residual contributes an additional factor:

$$E[e^{\beta\epsilon_\mathcal{M}}\,|\,\mathbf{x},\mathbf{x}'] = \exp\!\bigl(\tfrac{1}{2}\beta^2\,\text{Var}[\epsilon_{\mathcal{M}}]\bigr)$$

since $\epsilon_{\mathcal{M}}$ is approximately Gaussian. This correction is small (second order in $\Delta$) but included for maximum accuracy.

#### 3.3.3 State Grid for $\Psi_n$

Represent $\Psi_n$ on a tensor-product grid in factor space $(x_1, \ldots, x_K)$.

**Grid construction** for factor $k$ at time step $n$:

- Center: $\mu_k = 0$ (OU process has zero long-run mean)
- Spread: $\pm 4\sigma_k(t_n)$ where $\sigma_k(t_n) = w_k\sqrt{(1 - e^{-2\kappa_k t_n})/(2\kappa_k)}$
- Points: $N_x = 30$ per factor (cubic interpolation between grid points)

For $K = 2$: $30 \times 30 = 900$ grid points.
For $K = 3$: $30 \times 30 \times 30 = 27{,}000$ grid points (still fast).

**Interpolation**: When $\mathbf{x}'(Z_v)$ falls between grid points, use **cubic interpolation** (in each dimension) on $\Psi_{n+1}$. For complex-valued $\Psi$ (since $\alpha, \beta$ are complex), interpolate real and imaginary parts separately.

#### 3.3.4 Algorithm Summary

```
Input:  complex u (Fourier variable), model params, ξ₀(·), time grid {t_n}
Output: φ(u) = characteristic function value

1. Precompute for each step n = 0,...,N-1:
   - OU transition parameters: decay_k = e^{-κ_k Δ_n}, diffusion_k = w_k √((1-e^{-2κ_k Δ_n})/(2κ_k))
   - GL nodes u_g mapped to [t_n, t_{n+1}]
   - Bridge coefficients and variance corrections γ_k(u_g) at each GL node
   - ξ₀(u_g) values

2. Build factor grids: for each n, compute x_k grid centered at 0, width ±4σ_k(t_n)

3. Initialize: Ψ_N(x) = 1 + 0i  for all grid points

4. For n = N-1 down to 0:
   For each grid point x = (x₁,...,x_K):
     Ψ_n(x) = 0
     For q = 1 to N_GH:
       z_v = √2 · gh_node[q]
       x'_k = x_k · decay_k + diffusion_k · z_v        (for each k)
       Compute ΔI_n via 5-point GL (bridge-interpolated V)
       Compute ΔM_n via GL-based stochastic integral approximation
       Compute residual variance correction
       exponent = -α·ΔI_n + β·ΔM_n + ½β²·Var[ε_M]
       Ψ_next = cubic_interpolate(Ψ_{n+1} grid, x')
       Ψ_n(x) += gh_weight[q] · exp(exponent) · Ψ_next
     Ψ_n(x) /= √π

5. Return φ(u) = Ψ_0(0, ..., 0)
```

**Complexity per $\varphi(u)$ evaluation**: $N \times N_x^K \times N_{GH} \times 5_{GL}$

| $K$ | $N$ | $N_x$ | $N_{GH}$ | Total evals | Time (est.) |
|-----|-----|--------|-----------|-------------|-------------|
| 2   | 40  | 30     | 20        | 3.6M        | ~2ms        |
| 3   | 40  | 20     | 20        | 6.4M        | ~5ms        |

### 3.4 Lewis Option Pricing Formula

Given $\varphi(u)$, the undiscounted call price is (Lewis 2001):

$$C = F - \frac{\sqrt{KF}}{\pi}\int_0^\infty \text{Re}\!\left[\frac{\varphi(u - \tfrac{i}{2})\;e^{-iu\ln(K/F)}}{u^2 + \tfrac{1}{4}}\right]du$$

This formula is numerically stable (no cancellation issues) and converges rapidly.

**Numerical integration**: 64-point Gauss-Legendre on $[0, u_{\max}]$ where:

$$u_{\max} = \frac{50}{\sqrt{\mathcal{I}_0 \cdot T}}, \qquad \mathcal{I}_0 = \int_0^T \xi_0(t)\,dt$$

The integrand decays as $\sim e^{-\frac{1}{2}u^2 \mathcal{I}_0}$, so 64 points gives relative accuracy $\sim 10^{-12}$.

**Total cost per option**: 64 evaluations of $\varphi(u)$ $\times$ ~2ms each $\approx$ **130ms**.

To bring this within the 1-minute budget for a full surface, we use the optimizations in Section 3.5.

### 3.5 Performance Optimizations

#### 3.5.1 Share the Backward Pass Across Strikes

For a fixed maturity $T$, $\varphi(u)$ **does not depend on $K$**. The strike only enters the Lewis integral:

$$C(K, T) = F - \frac{\sqrt{KF}}{\pi}\int_0^\infty \text{Re}\!\left[\frac{\varphi(u - \tfrac{i}{2})\;e^{-iu\ln(K/F)}}{u^2 + \tfrac{1}{4}}\right]du$$

So: **compute $\varphi(u)$ once per maturity at all 64 $u$-nodes, then evaluate the Lewis integral for each strike.** The per-strike cost is just a 64-point dot product.

**Cost per maturity**: 64 $\times$ ~2ms = **~130ms** (independent of number of strikes).

#### 3.5.2 Reduce Time Steps

Use **non-uniform time stepping** adapted to the maturity:
- 10 steps for $T \leq 0.5$y
- 20 steps for $T \leq 2$y
- 30 steps for $T \leq 5$y

Fewer steps = proportionally faster. GL quadrature within each step compensates for larger $\Delta$.

#### 3.5.3 Reduce Gauss-Hermite Nodes

For $K = 2$, $N_{GH} = 15$ typically suffices (instead of 20). Verify by comparing $N_{GH} = 15$ vs $N_{GH} = 25$ on a test case.

#### 3.5.4 Precompute Factor Grid Transition Matrices

For each time step $n$ and each GH node $q$, the map $\mathbf{x} \to \mathbf{x}'(Z_v)$ is **affine**. Precompute the interpolation weights mapping $\Psi_{n+1}$ grid values to $\Psi_n$ grid values. This turns the backward pass into matrix-vector multiplies:

$$\vec{\Psi}_n = \sum_q \text{diag}(\vec{f}_q) \cdot A_q \cdot \vec{\Psi}_{n+1}$$

where $A_q$ is the interpolation matrix and $\vec{f}_q$ contains the exponential factors.

#### 3.5.5 Parallelize Over Maturities

Each maturity is **independent** (different $T$, different backward pass). With 10 maturities on 10 threads:

**Estimated total**: 10 maturities $\times$ 130ms / 10 threads $\approx$ **13 seconds per surface**.

Within budget, and 64 Lewis points may be reduced to 32 for another 2$\times$ speedup.

#### 3.5.6 Cache $\xi_0$ and Bridge Coefficients

The forward variance curve and all bridge-related coefficients ($\gamma_k$, sinh ratios, GL nodes) depend only on the time grid, not on the optimization parameters. **Precompute once** before the LM loop.

Wait — the bridge coefficients depend on $\kappa_k$, which **is** being optimized. So recompute them only when $\kappa$ changes. If $\kappa$ is fixed (see Section 4.2), they are truly precomputable.

---

## 4. Calibration Algorithm

### 4.1 Objective Function

Given $N_T$ maturities and $N_K^{(j)}$ strikes per maturity, minimize:

$$\mathcal{L}(\theta) = \sum_{j=1}^{N_T}\sum_{i=1}^{N_K^{(j)}} \omega_{ij}\;\bigl(\sigma_{\text{model}}(K_i, T_j;\,\theta) - \sigma_{\text{mkt}}(K_i, T_j)\bigr)^2$$

where $\theta = (w_1, w_2, \kappa_1, \kappa_2, \rho)$ and $\omega_{ij}$ are user-specified weights.

### 4.2 Weighting Scheme

The weights $\omega_{ij}$ control which parts of the surface are fit most accurately. Default scheme:

$$\omega_{ij} = \omega_j^{(\text{mat})} \cdot \omega_i^{(\text{strike})} \cdot \omega_{ij}^{(\text{vega})} \cdot \omega_{ij}^{(\text{bid-ask})}$$

Each factor:

| Component | Formula | Purpose |
|-----------|---------|---------|
| $\omega_j^{(\text{mat})}$ | User-specified per maturity | Prioritize liquid tenors or hedging-critical maturities |
| $\omega_i^{(\text{strike})}$ | User-specified per strike | Upweight ATM, downweight deep OTM wings |
| $\omega_{ij}^{(\text{vega})}$ | $\text{vega}_{ij}^2 / \text{vega}_{\text{ATM},j}^2$ | Vega-weight: errors in high-vega options matter more for hedging |
| $\omega_{ij}^{(\text{bid-ask})}$ | $1 / \max(\sigma_{ij}^{\text{ask}} - \sigma_{ij}^{\text{bid}},\, \epsilon)^2$ | Tight markets get more weight; wide markets tolerate more model error |

**Preset weight profiles**:

```
UNIFORM:     ω_ij = 1
ATM_FOCUSED: ω_strike = exp(-2k²),  k = ln(K/F)
VEGA:        ω_ij = vega²_ij / vega²_ATM,j
HEDGING:     ω_ij = vega²_ij · ω_mat_j  (user-specified maturity priorities)
BID_ASK:     ω_ij = 1 / (σ_ask - σ_bid)²
```

**Custom overrides**: For exotic hedging or regulatory scenarios, allow per-instrument overrides:

```cpp
struct CalibrationPoint {
    double strike;
    double maturity;
    double market_vol;
    double weight;        // total weight (user can override)
    double bid_vol;       // optional: for bid-ask weighting
    double ask_vol;       // optional: for bid-ask weighting
    bool   is_excluded;   // exclude from calibration entirely
};
```

### 4.3 Initialization via Analytical Formulas

Good starting parameters are critical for LM convergence. Use closed-form approximations.

#### 4.3.1 ATM Implied Variance

By construction:

$$\sigma_{\text{ATM}}^2(T)\,T \approx \int_0^T \xi_0(T)\,dt$$

This is exact to leading order and matched automatically.

#### 4.3.2 ATM Skew

The ATM implied vol skew at maturity $T$ is (Bergomi 2005, to first order in vol-of-vol):

$$\mathcal{S}(T) \equiv \left.\frac{\partial\sigma_{\text{imp}}}{\partial k}\right|_{k=0} = \frac{\rho}{2\,\sigma_{\text{ATM}}(T)\,T}\;\sum_k w_k\;\int_0^T \xi_0(t)\,\frac{1 - e^{-\kappa_k(T-t)}}{\kappa_k}\,dt$$

Define the **skew kernel integral** per factor:

$$\mathcal{K}_k(T) = \int_0^T \xi_0(t)\,\frac{1 - e^{-\kappa_k(T-t)}}{\kappa_k}\,dt = \frac{1}{\kappa_k}\left[\mathcal{I}_0(T) - e^{-\kappa_k T}\int_0^T \xi_0(t)\,e^{\kappa_k t}\,dt\right]$$

The remaining integral $\int_0^T \xi_0(t)\,e^{\kappa_k t}\,dt$ is evaluated numerically (20-point GL on the $\xi_0$ spline).

Then:

$$\mathcal{S}(T) = \frac{\rho}{2\,\sigma_{\text{ATM}}(T)\,T}\;\sum_k w_k\,\mathcal{K}_k(T)$$

**Matching**: Given market skews $\mathcal{S}_{\text{mkt}}(T_j)$, this gives linear constraints on $\rho \cdot w_k$.

#### 4.3.3 ATM Curvature (Smile Convexity)

The ATM curvature (second derivative of implied vol in log-moneyness) is:

$$\mathcal{C}(T) \equiv \left.\frac{\partial^2\sigma_{\text{imp}}}{\partial k^2}\right|_{k=0}$$

To second order in vol-of-vol:

$$\mathcal{C}(T) = \frac{1}{\sigma_{\text{ATM}}(T)\,T}\left[\frac{1+2\rho^2}{4}\;\sum_{k,l} w_k\,w_l\;\mathcal{J}_{kl}(T) - \mathcal{S}(T)^2\right]$$

where the **curvature kernel** is:

$$\mathcal{J}_{kl}(T) = \int_0^T \xi_0(t)\int_0^t\int_0^t e^{-\kappa_k(t-u)}\,e^{-\kappa_l(t-v)}\;\min(u,v)\;du\,dv\,dt$$

This triple integral simplifies (by Fubini) to expressions involving exponential integrals of $\xi_0$, evaluated numerically.

For $K = 2$, we have 3 terms: $\mathcal{J}_{11}$, $\mathcal{J}_{12}$, $\mathcal{J}_{22}$.

**Matching**: Given market curvatures at two maturities, solve for $(w_1, w_2)$ given $\rho$.

#### 4.3.4 Initialization Procedure

```
1. Fix κ₁ = 1/T_short (e.g., 4.0 for 3-month), κ₂ = 1/T_long (e.g., 0.3 for 3-year)

2. Compute market ATM skews S_mkt(T_j) for j = 1,...,N_T
   (finite difference: S ≈ [σ(K+, T) - σ(K-, T)] / [ln(K+/F) - ln(K-/F)] at K± = F·e^{±0.05})

3. Compute skew kernels K₁(T_j), K₂(T_j)

4. Linear regression: S_mkt(T_j) · 2σ_ATM(T_j)T_j = ρ·[w₁K₁(T_j) + w₂K₂(T_j)]
   Two unknowns: ρw₁, ρw₂. Least-squares with N_T equations.

5. From market curvatures C_mkt(T_j), solve for |ρ| using the curvature formula.
   Then: w_k = (ρw_k) / ρ.

6. Starting point: θ₀ = (w₁, w₂, κ₁, κ₂, ρ)
```

### 4.4 Levenberg-Marquardt Optimization

#### 4.4.1 Residual Vector

$$r_{ij}(\theta) = \sqrt{\omega_{ij}}\;\bigl(\sigma_{\text{model}}(K_i, T_j;\,\theta) - \sigma_{\text{mkt}}(K_i, T_j)\bigr)$$

so that $\mathcal{L}(\theta) = \|r(\theta)\|^2$.

Total residuals: $M = \sum_j N_K^{(j)}$ (typically 100-200).

#### 4.4.2 Jacobian Computation

The Jacobian $J_{m,p} = \partial r_m / \partial \theta_p$ is computed by **central finite differences**:

$$J_{m,p} \approx \frac{r_m(\theta + h_p\,e_p) - r_m(\theta - h_p\,e_p)}{2h_p}$$

Step sizes:
- $h_w = 10^{-4}$ (weights)
- $h_\kappa = 10^{-3}$ (mean-reversions)
- $h_\rho = 10^{-5}$ (correlation)

This requires $2 \times 5 = 10$ additional surface evaluations per LM iteration. Since each surface evaluation is ~13s (parallelized), the Jacobian takes ~26s.

**Alternative**: Adjoint AD through the pricing engine (see Section 5.3).

#### 4.4.3 LM Update

Standard damped Gauss-Newton:

$$\theta_{i+1} = \theta_i - (J^T J + \lambda\,\text{diag}(J^T J))^{-1}\,J^T\,r$$

with adaptive $\lambda$ (increase if step worsens, decrease if improves). The $5 \times 5$ system is solved directly.

#### 4.4.4 Parameter Bounds

Enforce via projected LM (clamp after each step):

| Parameter | Lower | Upper |
|-----------|-------|-------|
| $w_k$ | 0.01 | 10.0 |
| $\kappa_k$ | 0.05 | 20.0 |
| $\rho$ | -0.99 | -0.01 |

Additionally, enforce ordering: $\kappa_1 > \kappa_2$ (factor 1 is "fast", factor 2 is "slow").

#### 4.4.5 Convergence

Terminate when:
- $\|\Delta\theta\| / \|\theta\| < 10^{-6}$, **or**
- $\|\nabla\mathcal{L}\| < 10^{-8}$, **or**
- Max 30 iterations reached.

Typical convergence: **5-10 iterations** with good initialization.

### 4.5 Calibration Budget

For 10 maturities $\times$ 15 strikes = 150 calibration points, $K = 2$ factors:

| Component | Cost | Notes |
|-----------|------|-------|
| Initialization (analytical) | ~100ms | One-time skew/curvature integrals |
| One surface evaluation | ~13s | 10 maturities $\times$ 130ms, parallelized over 10 threads |
| Jacobian (finite diff) | ~26s | 10 perturbations $\times$ 13s / 5 threads |
| Per LM iteration | ~39s | 1 evaluation + 1 Jacobian |
| Total (8 iterations) | **~5 minutes** | Exceeds 1-minute budget |

**To meet the 1-minute target**, use the following acceleration:

1. **Reduce $N_{GH}$ to 12**: ~1.5$\times$ speedup (negligible accuracy loss)
2. **Reduce $N$ (time steps) to 15-20**: ~2$\times$ speedup
3. **Reduce Lewis points to 32**: ~2$\times$ speedup
4. **Fix $\kappa_1, \kappa_2$**: Jacobian is now $3 \times M$ instead of $5 \times M$ → 6 perturbations instead of 10
5. **One-sided finite differences**: $J_{m,p} \approx [r_m(\theta + h_p e_p) - r_m(\theta)] / h_p$. 3 perturbation evaluations + 1 base = 4 total.

With these: ~13s / (1.5 $\times$ 2 $\times$ 2) = ~2s per surface evaluation. 4 evaluations per iteration $\times$ 8 iterations = **~64 seconds**. Within budget.

For even tighter budgets, use the analytical formulas (Section 4.3) directly in a Nelder-Mead simplex optimizer, checking against the full pricing engine only at the end.

---

## 5. Implementation Architecture

### 5.1 Class Hierarchy

```
BergomiCalibrator
├── ForwardVarianceCurve         ξ₀(T) spline, extraction from market
├── BergomiPricingEngine         Characteristic function + Lewis pricing
│   ├── FactorGrid               Tensor-product grid for Ψ
│   ├── BackwardSolver           Recursive GH + GL quadrature
│   └── LewisIntegrator          Fourier inversion for option prices
├── CalibrationObjective         Residuals, weights, Jacobian
│   ├── WeightScheme             Weight computation per point
│   └── CalibrationPoint[]       Market data + weights
└── LevenbergMarquardt           Optimizer with bounds
```

### 5.2 Key Interfaces

```cpp
// Forward variance curve
struct ForwardVarianceCurve {
    // Interpolated ξ₀(t) from monotone cubic spline
    double xi0(double t) const;
    // Integrated: ∫₀ᵗ ξ₀(u) du
    double integrated_xi0(double t) const;
    // Weighted integral: ∫₀ᵗ ξ₀(u) e^{κu} du (for skew kernel)
    double weighted_integral(double t, double kappa) const;

    // Build from variance swap rates
    static ForwardVarianceCurve from_vs_rates(
        const std::vector<double>& T,
        const std::vector<double>& vs_rate_sq);

    // Build from option surface
    static ForwardVarianceCurve from_options(
        const std::vector<double>& T,
        const std::vector<std::vector<double>>& K,
        const std::vector<std::vector<double>>& implied_vol,
        const std::vector<double>& forwards);
};

// Model parameters
struct BergomiParams {
    int K;                          // number of factors
    std::vector<double> w;          // factor weights w_k
    std::vector<double> kappa;      // mean-reversion speeds κ_k
    double rho;                     // spot-vol correlation

    // Derived
    double vol_of_vol_sq(double T) const;       // Σ w_k² e^{-2κ_k T}
    double skew_kernel(double T,
        const ForwardVarianceCurve& xi) const;  // for initialization
};

// Pricing engine
struct BergomiPricingEngine {
    // Price a full set of strikes at one maturity
    // Returns implied vols
    std::vector<double> price_smile(
        double T,
        const std::vector<double>& strikes,
        double forward,
        const BergomiParams& params,
        const ForwardVarianceCurve& xi,
        const PricingConfig& config) const;

    // Characteristic function at one u value (complex)
    std::complex<double> char_func(
        std::complex<double> u,
        double T,
        const BergomiParams& params,
        const ForwardVarianceCurve& xi,
        const PricingConfig& config) const;
};

struct PricingConfig {
    int n_time_steps    = 25;   // backward recursion steps
    int n_gauss_hermite = 15;   // GH nodes for factor integration
    int n_gauss_legendre_var = 5;  // GL nodes for variance integral
    int n_lewis_points  = 48;   // Fourier inversion nodes
    int n_grid_per_factor = 30; // grid points per factor dimension
    double grid_stdev   = 4.0;  // grid extends to ±4σ
};

// Calibration
struct CalibrationConfig {
    // Optimizer settings
    int max_iterations      = 30;
    double tol_params       = 1e-6;
    double tol_gradient     = 1e-8;
    double lambda_init      = 1e-3;

    // Parameter bounds
    double w_min = 0.01, w_max = 10.0;
    double kappa_min = 0.05, kappa_max = 20.0;
    double rho_min = -0.99, rho_max = -0.01;

    // Fixed parameters (if set, not optimized)
    std::optional<double> fixed_kappa1;
    std::optional<double> fixed_kappa2;

    // Finite difference step sizes
    double fd_step_w     = 1e-4;
    double fd_step_kappa = 1e-3;
    double fd_step_rho   = 1e-5;

    // Weighting
    WeightScheme weight_scheme = WeightScheme::VEGA;

    // Threading
    int num_threads = 0;  // 0 = hardware_concurrency
};

enum class WeightScheme {
    UNIFORM,
    ATM_FOCUSED,
    VEGA,
    HEDGING,
    BID_ASK,
    CUSTOM
};

struct CalibrationPoint {
    double strike;
    double maturity;
    double forward;       // forward price at this maturity
    double market_vol;    // market implied vol
    double weight;        // user override (used when scheme = CUSTOM, or as multiplier)
    double bid_vol;       // optional: for BID_ASK scheme
    double ask_vol;       // optional: for BID_ASK scheme
    bool   excluded;      // if true, skip this point
};

struct CalibrationResult {
    BergomiParams params;
    double objective_value;              // final L value
    int iterations;
    std::vector<double> model_vols;      // model vols at calibration points
    std::vector<double> residuals;       // weighted residuals
    double max_abs_error_vol;            // worst-case vol error (bps)
    double rms_error_vol;                // RMS vol error (bps)
    bool converged;
};

// Top-level calibrator
struct BergomiCalibrator {
    CalibrationResult calibrate(
        const std::vector<CalibrationPoint>& market_data,
        const ForwardVarianceCurve& xi,
        const CalibrationConfig& config,
        const BergomiParams* initial_guess = nullptr  // optional warm start
    ) const;
};
```

### 5.3 Adjoint Jacobian (Optional, for Speed)

Instead of finite-difference Jacobians, propagate adjoints through the backward recursion. Since the recursion is:

$$\Psi_n = G_n(\Psi_{n+1};\,\theta)$$

the adjoint satisfies:

$$\bar{\theta} \mathrel{+}= \frac{\partial G_n}{\partial\theta}^T \bar{\Psi}_n$$

$$\bar{\Psi}_{n+1} \mathrel{+}= \frac{\partial G_n}{\partial\Psi_{n+1}}^T \bar{\Psi}_n$$

This gives exact gradients $\partial\mathcal{L}/\partial\theta$ in **one forward + one backward pass** through the recursion — same cost as a single surface evaluation, replacing 10 finite-difference evaluations.

This is a significant optimization but adds implementation complexity. Worth doing if calibrating many surfaces (e.g., daily recalibration across hundreds of underlyings).

---

## 6. Validation

### 6.1 Internal Consistency Checks

| Test | Method | Expected |
|------|--------|----------|
| $E[V(T)] = \xi_0(T)$ | MC with 1M paths | Relative error $< 10^{-3}$ |
| ATM vol matches $\xi_0$ | Price ATM option, extract vol | $< 0.1$ bp difference |
| Put-call parity | Price put and call at same $(K, T)$ | Machine precision |
| $\varphi(0) = 1$ | Evaluate char function | $|1 - \varphi(0)| < 10^{-12}$ |
| $\varphi(-i) = 1$ (forward martingale) | Evaluate char function | $|1 - \varphi(-i)| < 10^{-10}$ |

### 6.2 Convergence Studies

For a reference parameter set, vary numerical parameters and measure option price changes:

1. **Time steps $N$**: Run $N = 10, 20, 40, 80$. Verify Richardson extrapolation gives $O(\Delta^2)$ convergence.
2. **GH nodes**: Run $N_{GH} = 10, 15, 20, 25$. Should saturate by 15-20.
3. **GL nodes**: Run $N_{GL} = 3, 5, 7$. Should saturate by 5.
4. **Factor grid $N_x$**: Run $N_x = 15, 20, 25, 30, 40$. Should saturate by 25-30.
5. **Lewis integration points**: Run 16, 32, 48, 64. Should saturate by 32-48.

### 6.3 Benchmark Against MC

After calibration, run a high-fidelity MC simulation (Section 7 reference) with $10^6$ paths and compare implied vols. Agreement within 0.1 vol bp confirms the pricing engine accuracy.

---

## 7. Simulation Post-Calibration (Reference)

After calibration, use the exact simulation scheme for path generation:

```
For each path:
  X_k(0) = 0,  ln S(0) = ln S_spot

  For each step [t_n, t_{n+1}]:
    1. Draw Z_v, Z_perp ~ N(0,1)  [from Sobol + inverse CDF]

    2. Exact OU update:
       X_k(t_{n+1}) = X_k(t_n) e^{-κ_k Δ} + w_k √((1-e^{-2κ_k Δ})/(2κ_k)) · Z_v

    3. V(t_{n+1}) = ξ₀(t_{n+1}) exp(Σ w_k X_k(t_{n+1}) - ½ Σ w_k² σ_k²(t_{n+1}))

    4. IV = GL-quadrature of V over [t_n, t_{n+1}] using OU bridge

    5. ln S(t_{n+1}) = ln S(t_n) + (r-q)Δ - ½IV + ρ√IV·Z_v + √(1-ρ²)√IV·Z_perp
```

See the previous discussion for full details on bridge interpolation and GL quadrature.

---

## Appendix A: Gauss-Hermite Nodes and Weights

For $N_{GH} = 15$, the nodes $z_q$ and weights $w_q$ satisfy:

$$\int_{-\infty}^{\infty} f(x)\,e^{-x^2}\,dx \approx \sum_{q=1}^{N} w_q\,f(z_q)$$

To integrate against $N(0,1)$: substitute $x = z/\sqrt{2}$, so nodes are $\sqrt{2}\,z_q$ and weights are $w_q/\sqrt{\pi}$.

Precomputed tables for $N_{GH} = 5, 10, 15, 20, 25, 30$ should be embedded as `constexpr` arrays.

## Appendix B: Gauss-Legendre Nodes and Weights

For 5-point GL on $[-1, 1]$:

| Node | Weight |
|------|--------|
| $\pm 0.906179845938664$ | $0.236926885056189$ |
| $\pm 0.538469310105683$ | $0.478628670499366$ |
| $0$ | $0.568888888888889$ |

Map to $[a, b]$: $u = \tfrac{a+b}{2} + \tfrac{b-a}{2}\,z$, multiply weights by $\tfrac{b-a}{2}$.

## Appendix C: Lewis Integration Derivation

Starting from the risk-neutral call price:

$$C = e^{-rT}\,E[(S(T) - K)^+] = e^{-rT}\,\frac{1}{2\pi}\int_{-\infty+iv_0}^{\infty+iv_0} \frac{\hat{\varphi}(z)\,K^{1-iz}}{z(z-i)}\,dz$$

Setting $v_0 = 1/2$ (midway between poles at $z = 0$ and $z = i$) and $z = u + i/2$:

$$C = e^{-rT}\left[F - \frac{\sqrt{KF}}{\pi}\int_0^\infty \text{Re}\!\left(\frac{\varphi(u - i/2)\,e^{-iu\ln(K/F)}}{u^2 + 1/4}\right)du\right]$$

where $\varphi(u) = E[e^{iu\ln(S(T)/F)}]$ is the demeaned characteristic function.
