# American Option Pricing with Brownian Bridge Exercise Correction

## 1. Problem Statement

In a Monte Carlo exposure simulation (XVA, PFE, EPE), we step forward in time through discrete timepoints $t_0, t_1, \ldots, t_N = T$ on each path. At every $(path, timepoint)$ we need the mark-to-market (MTM) of each trade.

For American options, we use an analytical approximation (Bjerksund-Stensland 2002) to price the option at each timepoint given the current spot and volatility. This handles the *continuation value* correctly — BS2002 accounts for the early exercise premium over the remaining life.

**The problem**: Between consecutive timepoints (e.g., 7 days apart), the underlying may cross the optimal exercise boundary without us observing it. If we only check at discrete dates, we undercount early exercises and overstate the option's survival probability.

**The solution**: Use a Brownian bridge to compute the probability that the underlying crossed the exercise boundary between observation dates, and weight the MTM accordingly.

### 1.1 Why This Matters

For a deep ITM American put with spot near the exercise boundary:
- Discrete checking at 7-day intervals can miss 10–30% of early exercises
- This leads to overestimation of exposure (the option appears alive when it should have been exercised)
- For CVA/DVA calculations, this bias directly affects credit exposure profiles

### 1.2 Scope

This document covers:
- American put and call options (explicit formulas for both — Sections 4.3 and 4.4)
- Heston stochastic volatility (we observe both $S$ and $v$ at each timepoint)
- Bjerksund-Stensland 2002 for the analytical exercise boundary and option price
- All combinations of $r$ and $q$, including $q < 0$ and edge cases (Section 4.5)
- Forward-stepping MC simulation (exposure-style, not Longstaff-Schwartz backward induction)

---

## 2. Model Setup

### 2.1 Heston Dynamics

Under the risk-neutral measure:

$$dS = (r - q)\,S\,dt + \sqrt{v}\,S\,dW_S$$

$$dv = \kappa(\theta - v)\,dt + \xi\sqrt{v}\,dW_v$$

$$dW_S\,dW_v = \rho\,dt$$

where:
- $S$: equity spot price
- $v$: instantaneous variance
- $r$: risk-free rate
- $q$: continuous dividend yield
- $\kappa$: variance mean-reversion speed
- $\theta$: long-run variance
- $\xi$: vol-of-vol
- $\rho$: spot-vol correlation

### 2.2 Simulation Grid

- Timepoints: $t_0 = 0 < t_1 < t_2 < \cdots < t_N = T$
- Interval length: $\Delta t_i = t_i - t_{i-1}$ (typically 7 days = 7/365)
- At each timepoint we observe: $S_i = S(t_i)$ and $v_i = v(t_i)$
- Number of MC paths: $M$ (typically $10^3$–$10^5$ for XVA)

### 2.3 What We Compute at Each $(path, t_i)$

1. **Effective volatility** $\sigma_{\text{eff}}(t_i)$ — for feeding into BS2002
2. **Exercise boundary** $B_i$ — critical stock price below which immediate exercise is optimal
3. **Crossing probability** $p_i$ — probability that spot crossed $B_i$ between $t_{i-1}$ and $t_i$
4. **Survival probability** $Q_i$ — probability that the option is still alive at $t_i$
5. **PV** at $t_i$ — the mark-to-market of the option position

---

## 3. Effective Volatility Under Heston

### 3.1 The Problem

BS2002 assumes flat (constant) Black-Scholes volatility. Under Heston, volatility is stochastic. We need to map the current Heston state $(S_i, v_i)$ to a single effective volatility $\sigma_{\text{eff}}$ that BS2002 can use.

### 3.2 Expected Average Variance

The natural choice is the expected average variance over the remaining life $[t_i, T]$, conditional on the current variance $v_i$:

$$\sigma^2_{\text{eff}}(t_i) = \frac{1}{T - t_i} \int_{t_i}^{T} \mathbb{E}[v(u) \mid v(t_i) = v_i]\,du$$

Under Heston, the conditional expectation of the variance process is:

$$\mathbb{E}[v(u) \mid v_i] = v_i\,e^{-\kappa(u - t_i)} + \theta\left(1 - e^{-\kappa(u - t_i)}\right)$$

Integrating over $[t_i, T]$:

$$\int_{t_i}^{T} \mathbb{E}[v(u) \mid v_i]\,du = \theta(T - t_i) + (v_i - \theta)\,\frac{1 - e^{-\kappa(T - t_i)}}{\kappa}$$

Therefore:

$$\boxed{\sigma^2_{\text{eff}}(t_i) = \theta + (v_i - \theta)\,\frac{1 - e^{-\kappa(T - t_i)}}{\kappa(T - t_i)}}$$

$$\sigma_{\text{eff}}(t_i) = \sqrt{\sigma^2_{\text{eff}}(t_i)}$$

### 3.3 Properties

- When $v_i = \theta$: $\sigma^2_{\text{eff}} = \theta$ (variance at long-run level, no correction needed)
- When $\kappa(T - t_i) \to 0$ (near expiry or slow mean-reversion): $\sigma^2_{\text{eff}} \to v_i$ (instantaneous variance dominates)
- When $\kappa(T - t_i) \to \infty$ (long-dated or fast mean-reversion): $\sigma^2_{\text{eff}} \to \theta$ (reverts to long-run)

### 3.4 Why Not Just Use $\sqrt{v_i}$?

Using the instantaneous vol $\sqrt{v_i}$ ignores mean reversion. If $v_i$ is temporarily elevated (e.g., during a vol spike), $\sqrt{v_i}$ overstates the vol over the remaining life, leading to:
- Overstated option values
- Exercise boundary $B_i$ that is too low (for a put), causing undercounting of exercises
- Incorrect exposure profiles

The effective vol correctly blends the current variance toward the long-run level over the option's remaining life.

### 3.5 Alternative: Moment-Matched Vol

A more sophisticated approach matches the first two moments of the integrated variance distribution. Under Heston, the variance of the integrated variance is known:

$$\text{Var}\left[\int_{t_i}^{T} v(u)\,du \;\middle|\; v_i\right] = \frac{\xi^2}{\kappa^2}\left[\frac{v_i - \theta}{\kappa}\left(1 - e^{-\kappa\tau}\right)^2 + \frac{\theta\tau}{2}\left(1 - e^{-\kappa\tau}\right)^2 \cdot \frac{2}{\kappa\tau}\right]$$

where $\tau = T - t_i$. This allows constructing a distribution for the effective vol, but for BS2002 input the first-moment (expected average) is sufficient and much simpler.

---

## 4. Exercise Boundary from Bjerksund-Stensland 2002

### 4.1 Overview

BS2002 approximates the American option price using a **piecewise-flat exercise boundary**. The boundary formulas are fully closed-form (no root-finding, no iteration), producing explicit values that plug directly into the crossing probability.

This section gives direct formulas for both American puts and calls, covering all combinations of $r$ and $q$, including negative dividend yields and edge cases.

### 4.2 The BS Quadratic

Both put and call boundaries rest on the perpetual American option solution. The characteristic equation is:

$$\frac{1}{2}\sigma^2\beta(\beta - 1) + b\,\beta - r = 0$$

with cost of carry $b = r - q$. The two roots are:

$$\beta_{\pm} = \frac{1}{2} - \frac{b}{\sigma^2} \pm \sqrt{\left(\frac{1}{2} - \frac{b}{\sigma^2}\right)^2 + \frac{2r}{\sigma^2}}$$

The discriminant is $\left(\frac{1}{2} - \frac{b}{\sigma^2}\right)^2 + \frac{2r}{\sigma^2}$. Since $r > 0$ (required for put early exercise) and the squared term is non-negative, the discriminant is always positive, regardless of $q$. Both roots are always real.

**Properties** (for $r > 0$):
- $\beta_+ > 1$ always (the positive root)
- $\beta_- < 0$ always (the negative root)
- $\beta_+ \cdot \beta_- = -2r/\sigma^2 < 0$ (Vieta's formula confirms opposite signs)

### 4.3 American Put Boundary

**Input**: $K$ (strike), $\tau = T - t_i$ (time to expiry), $r$ (risk-free rate), $q$ (dividend yield), $\sigma$ (volatility)

**Early exercise condition**: A put is optimally exercised early only when $r > 0$ (interest earned on strike proceeds exceeds option time value). If $r \leq 0$, set $B^{\text{put}} = 0$ (no early exercise).

For $r > 0$, the put boundary uses the **negative root** $\beta_-$ of the BS quadratic:

**Step 1 — Put exponent**

$$\boxed{\beta_p = \frac{1}{2} - \frac{r - q}{\sigma^2} - \sqrt{\left(\frac{1}{2} - \frac{r-q}{\sigma^2}\right)^2 + \frac{2r}{\sigma^2}}}$$

This is always real and satisfies $\beta_p < 0$ for $r > 0$, regardless of the sign of $q$.

**Step 2 — Perpetual boundary scaling**

The perpetual American put boundary is $B_\infty^{\text{put}} = \frac{\beta_p}{\beta_p - 1} \cdot K$. Define the scaling factor:

$$\boxed{\hat{c}_\infty = \frac{\beta_p - 1}{\beta_p} = 1 + \frac{1}{|\beta_p|}}$$

Since $\beta_p < 0$: $\hat{c}_\infty > 1$. The perpetual put boundary is $K / \hat{c}_\infty < K$. ✓

**Step 3 — Floor scaling (at-expiry boundary)**

$$\boxed{\hat{c}_0 = \begin{cases} \max\!\left(1,\;\dfrac{q}{r}\right) & \text{if } q > 0 \\[6pt] 1 & \text{if } q \leq 0 \end{cases}}$$

Interpretation of $B_0 = K / \hat{c}_0$:
- $q \leq 0$: $\hat{c}_0 = 1$, $B_0 = K$ (boundary reaches strike at expiry) ✓
- $0 < q < r$: $\hat{c}_0 = 1$, $B_0 = K$ ✓
- $q = r$: $\hat{c}_0 = 1$, $B_0 = K$ ✓
- $q > r$: $\hat{c}_0 = q/r > 1$, $B_0 = rK/q < K$ (high dividends reduce exercise incentive) ✓

**Step 4 — BS2002 time split** (golden ratio)

$$t_1 = \frac{\sqrt{5} - 1}{2}\,\tau \approx 0.6180\,\tau, \qquad \tau_1 = \tau - t_1 = \frac{3 - \sqrt{5}}{2}\,\tau \approx 0.3820\,\tau$$

**Step 5 — Time-decay parameter $h$**

The $h$ parameter controls how quickly the boundary transitions from its at-expiry value to the perpetual value. For the put, the general form is:

$$h = -\bigl((r - q)\,\tau' + 2\sigma\sqrt{\tau'}\bigr)\;\frac{\hat{c}_{\text{floor}}}{\hat{c}_\infty - \hat{c}_{\text{floor}}}$$

**Guard**: $h$ can become positive when $(r-q)\sqrt{\tau'} > 2\sigma$ (very high interest rate differential relative to vol). If $h > 0$, the formula would give $\hat{c} > \hat{c}_\infty$ (boundary below the perpetual level), which is non-physical. Enforce:

$$h = \min(h,\; 0)$$

When $h = 0$: $\hat{c} = \hat{c}_{\text{floor}}$, i.e., $B = B_0$ (immediate exercise is optimal when deeply ITM). This is the correct behavior for very high $r$ or very low $\sigma$.

**Step 6 — Near-expiry boundary** (second segment, length $\tau_1$)

$$h_1 = \min\!\left(-\bigl((r - q)\,\tau_1 + 2\sigma\sqrt{\tau_1}\bigr)\;\frac{\hat{c}_0}{\hat{c}_\infty - \hat{c}_0},\;\; 0\right)$$

$$\boxed{\hat{c}_1 = \hat{c}_0 + (\hat{c}_\infty - \hat{c}_0)\bigl(1 - e^{h_1}\bigr)}$$

**Step 7 — Far-from-expiry boundary** (first segment, length $\tau$)

$$h_2 = \min\!\left(-\bigl((r - q)\,\tau + 2\sigma\sqrt{\tau}\bigr)\;\frac{\hat{c}_1}{\hat{c}_\infty - \hat{c}_1},\;\; 0\right)$$

$$\boxed{\hat{c}_2 = \hat{c}_1 + (\hat{c}_\infty - \hat{c}_1)\bigl(1 - e^{h_2}\bigr)}$$

**Step 8 — Put exercise boundaries**

$$\boxed{B_{\text{near}}^{\text{put}} = \frac{K}{\hat{c}_1}, \qquad B_{\text{far}}^{\text{put}} = \frac{K}{\hat{c}_2}}$$

Ordering: $B_\infty \leq B_{\text{far}} \leq B_{\text{near}} \leq K$.

The piecewise boundary for remaining life $\tau' \in [0, \tau]$:

$$B^{\text{put}}(\tau') = \begin{cases} B_{\text{far}} & \tau' \in [0, t_1) \\[4pt] B_{\text{near}} & \tau' \in [t_1, \tau] \end{cases}$$

### 4.4 American Call Boundary

**Early exercise condition**: A call is optimally exercised early only when $q > 0$ (the holder foregoes dividends by not owning the stock). If $q \leq 0$, set $B^{\text{call}} = +\infty$ (no early exercise).

For $q > 0$, the call boundary uses the **positive root** $\beta_+$:

**Step 1 — Call exponent**

$$\boxed{\beta_c = \frac{1}{2} - \frac{r - q}{\sigma^2} + \sqrt{\left(\frac{1}{2} - \frac{r-q}{\sigma^2}\right)^2 + \frac{2r}{\sigma^2}}}$$

$\beta_c > 1$ for $r > 0$.

**Step 2 — Perpetual boundary scaling**

$$\boxed{c_\infty = \frac{\beta_c}{\beta_c - 1}}$$

Since $\beta_c > 1$: $c_\infty > 1$. The perpetual call boundary is $c_\infty \cdot K > K$. ✓

**Step 3 — Floor scaling (at-expiry boundary)**

$$\boxed{c_0 = \begin{cases} \max\!\left(1,\;\dfrac{r}{q}\right) & \text{if } q > 0 \\[6pt] +\infty & \text{if } q \leq 0 \text{ (no early exercise)} \end{cases}}$$

For $q > 0$:
- $r < q$: $c_0 = 1$, $I_0 = K$ (boundary drops to strike at expiry) ✓
- $r = q$: $c_0 = 1$, $I_0 = K$ ✓
- $r > q > 0$: $c_0 = r/q > 1$, $I_0 = (r/q)K > K$ ✓

**Step 4 — Time split**: Same golden ratio as for puts.

**Step 5 — Time-decay parameter**

$$h = -\bigl((r - q)\,\tau' + 2\sigma\sqrt{\tau'}\bigr)\;\frac{c_{\text{floor}}}{c_\infty - c_{\text{floor}}}$$

Note: for calls, the sign convention on the carry term is $(r-q)$, the same as for puts. This is because both call and put boundaries share the same BS quadratic; the difference is only which root is used.

**Guard**: Same as for puts — enforce $h = \min(h, 0)$.

**Step 6 — Near-expiry call boundary** (segment length $\tau_1$)

$$h_1 = \min\!\left(-\bigl((r - q)\,\tau_1 + 2\sigma\sqrt{\tau_1}\bigr)\;\frac{c_0}{c_\infty - c_0},\;\; 0\right)$$

$$\boxed{c_1 = c_0 + (c_\infty - c_0)\bigl(1 - e^{h_1}\bigr)}$$

**Step 7 — Far-from-expiry call boundary** (segment length $\tau$)

$$h_2 = \min\!\left(-\bigl((r - q)\,\tau + 2\sigma\sqrt{\tau}\bigr)\;\frac{c_1}{c_\infty - c_1},\;\; 0\right)$$

$$\boxed{c_2 = c_1 + (c_\infty - c_1)\bigl(1 - e^{h_2}\bigr)}$$

**Step 8 — Call exercise boundaries**

$$\boxed{I_{\text{near}}^{\text{call}} = c_1 \cdot K, \qquad I_{\text{far}}^{\text{call}} = c_2 \cdot K}$$

Ordering: $K \leq I_{\text{near}} \leq I_{\text{far}} \leq I_\infty = c_\infty K$.

The piecewise boundary:

$$I^{\text{call}}(\tau') = \begin{cases} I_{\text{far}} & \tau' \in [0, t_1) \\[4pt] I_{\text{near}} & \tau' \in [t_1, \tau] \end{cases}$$

### 4.5 Edge Case Table

The following table summarizes early exercise behavior for all $(r, q)$ combinations:

| $r$ | $q$ | Put early exercise? | Call early exercise? | Notes |
|-----|-----|--------------------|--------------------|-------|
| $r > 0$ | $q > 0$ | Yes | Yes | Standard case. Both boundaries are finite. |
| $r > 0$ | $q = 0$ | Yes | No | No dividends → call never exercises. $B^{\text{call}} = +\infty$. |
| $r > 0$ | $q < 0$ | Yes | No | Negative "dividend" (stock lending fee, etc.) makes holding even more attractive for calls. Put boundary still well-defined since discriminant uses $2r/\sigma^2 > 0$. |
| $r = 0$ | $q > 0$ | No | Yes | Zero rates → no incentive to exercise put (no interest on $K$). Call exercises to capture dividends. |
| $r = 0$ | $q = 0$ | No | No | Neither exercises early. American = European. |
| $r = 0$ | $q < 0$ | No | No | Neither exercises early. |
| $r < 0$ | $q > 0$ | No | Yes | Negative rates → put holder penalized for receiving $K$. Call still exercises for dividends. |
| $r < 0$ | $q \leq 0$ | No | No | Neither exercises early. |

**Implementation rule**: Before computing boundary formulas, check:
- Put: if $r \leq 0$, return $B = 0$
- Call: if $q \leq 0$, return $I = +\infty$

### 4.6 Why the Boundary Is Closed-Form

The boundary $B = K / \hat{c}$ (put) or $I = c \cdot K$ (call) depends on $(r, q, \sigma, \tau)$ only — not on the current spot $S$.

This is because the BS quadratic exponents $\beta_\pm$ and the scaling factors $c_\infty, c_0$ involve only market parameters. The $h$-formulas involve ratios like $c_0 / (c_\infty - c_0)$ where any spot dependence would cancel. The boundary is therefore:

- **Independent of $S$** — depends only on contractual/market parameters
- **Precomputable** per timepoint for all paths (except for the $\sigma_{\text{eff}}$ dependence on $v_i$)
- **Explicit** — only elementary operations (exp, sqrt, max, division)

**Caveat on $\sigma_{\text{eff}}$**: Since $\sigma_{\text{eff}}(t_i)$ depends on $v_i$ (Section 3.2), the boundary varies across paths. The computation is cheap ($\sim 20$ flops).

### 4.7 Numerical Examples

**Example 1 — Standard put** ($K = 100$, $\tau = 1$, $r = 0.05$, $q = 0.02$, $\sigma = 0.25$):

$$b = r - q = 0.03, \quad \sigma^2 = 0.0625$$

$$\beta_p = 0.5 - \frac{0.03}{0.0625} - \sqrt{\left(0.5 - 0.48\right)^2 + \frac{0.10}{0.0625}} = 0.02 - \sqrt{0.0004 + 1.6} = 0.02 - 1.265 = -1.245$$

$$\hat{c}_\infty = 1 + 1/1.245 = 1.803, \quad \hat{c}_0 = \max(1, 0.02/0.05) = 1$$

For the BS2002 two-piece ($t_1 = 0.618$, $\tau_1 = 0.382$):

$$h_1 = -\bigl(0.03 \times 0.382 + 2 \times 0.25 \times \sqrt{0.382}\bigr) \cdot \frac{1}{0.803} = -(0.0115 + 0.309) \cdot 1.245 = -0.399$$

$$\hat{c}_1 = 1 + 0.803(1 - e^{-0.399}) = 1 + 0.803 \times 0.329 = 1.264$$

$$h_2 = -(0.03 + 0.5) \cdot \frac{1.264}{1.803 - 1.264} = -0.530 \cdot 2.345 = -1.243$$

$$\hat{c}_2 = 1.264 + 0.539(1 - e^{-1.243}) = 1.264 + 0.539 \times 0.711 = 1.647$$

$$B_{\text{near}} = 100/1.264 = 79.1, \quad B_{\text{far}} = 100/1.647 = 60.7$$

**Example 2 — Put with negative dividend yield** ($K = 100$, $\tau = 1$, $r = 0.05$, $q = -0.05$, $\sigma = 0.25$):

$$b = r - q = 0.10, \quad \hat{c}_0 = 1 \text{ (since } q < 0\text{)}$$

$$\beta_p = 0.5 - \frac{0.10}{0.0625} - \sqrt{(0.5 - 1.6)^2 + 1.6} = -1.1 - \sqrt{1.21 + 1.6} = -1.1 - 1.676 = -2.776$$

$$\hat{c}_\infty = 1 + 1/2.776 = 1.360$$

The boundary is lower ($B_\infty = 73.5$) than in Example 1 ($B_\infty = 55.5$). With negative $q$, the stock grows faster (no dividend drag), making the put less likely to be exercised, which pushes the boundary down. ✓

**Example 3 — Standard call** ($K = 100$, $\tau = 1$, $r = 0.05$, $q = 0.03$, $\sigma = 0.25$):

$$\beta_c = 0.5 - \frac{0.02}{0.0625} + \sqrt{(0.5 - 0.32)^2 + 1.6} = 0.18 + \sqrt{0.0324 + 1.6} = 0.18 + 1.278 = 1.458$$

$$c_\infty = \frac{1.458}{0.458} = 3.183, \quad c_0 = \max(1, 0.05/0.03) = 1.667$$

$$h_1 = -(0.02 \times 0.382 + 0.309) \cdot \frac{1.667}{3.183 - 1.667} = -0.317 \cdot 1.099 = -0.348$$

$$c_1 = 1.667 + 1.516(1 - e^{-0.348}) = 1.667 + 1.516 \times 0.294 = 2.113$$

$$h_2 = -(0.02 + 0.5) \cdot \frac{2.113}{3.183 - 2.113} = -0.520 \cdot 1.975 = -1.027$$

$$c_2 = 2.113 + 1.070(1 - e^{-1.027}) = 2.113 + 1.070 \times 0.641 = 2.799$$

$$I_{\text{near}} = 2.113 \times 100 = 211.3, \quad I_{\text{far}} = 2.799 \times 100 = 279.9$$

### 4.8 Verification of Boundary Properties

**Put boundary**:

| Property | Required | Formula gives |
|----------|----------|---------------|
| $B < K$ | Yes (ITM exercise) | $\hat{c} > 1 \Rightarrow K/\hat{c} < K$ ✓ |
| $B \to K$ as $\tau \to 0$ (when $q \leq r$) | Yes | $h \to 0$, $\hat{c} \to \hat{c}_0 = 1$ ✓ |
| $B \to rK/q$ as $\tau \to 0$ (when $q > r$) | Yes | $\hat{c}_0 = q/r$ ✓ |
| $B \to K/\hat{c}_\infty$ as $\tau \to \infty$ | Yes | $h \to -\infty$, $\hat{c} \to \hat{c}_\infty$ ✓ |
| $B$ decreasing in $\sigma$ | Yes | $|\beta_p| \downarrow \Rightarrow \hat{c}_\infty \uparrow \Rightarrow B \downarrow$ ✓ |
| $B$ increasing in $r$ | Yes | $|\beta_p| \uparrow \Rightarrow \hat{c}_\infty \downarrow \Rightarrow B \uparrow$ ✓ |
| $B_{\text{near}} \geq B_{\text{far}}$ | Yes | $\hat{c}_2 \geq \hat{c}_1$ ✓ |

**Call boundary**:

| Property | Required | Formula gives |
|----------|----------|---------------|
| $I > K$ | Yes (ITM exercise) | $c > 1 \Rightarrow cK > K$ ✓ |
| $I \to K$ as $\tau \to 0$ (when $r \leq q$) | Yes | $c_0 = 1$ ✓ |
| $I \to (r/q)K$ as $\tau \to 0$ (when $r > q$) | Yes | $c_0 = r/q$ ✓ |
| $I \to c_\infty K$ as $\tau \to \infty$ | Yes | ✓ |
| $I$ increasing in $\sigma$ | Yes | Larger $c_\infty$ ✓ |
| $I_{\text{far}} \geq I_{\text{near}}$ | Yes | $c_2 \geq c_1$ ✓ |

### 4.9 Which Boundary to Use for the Crossing Check

At time $t_i$ with remaining life $\tau_i$, BS2002 splits the remaining life into two segments. The interval $[t_i, t_{i+1}]$ (next 7 days) falls in the **first segment** (far from expiry), so:

$$B_i^{\text{put}} = B_{\text{far}} = \frac{K}{\hat{c}_2}, \qquad I_i^{\text{call}} = I_{\text{far}} = c_2 \cdot K$$

| Choice | Put formula | Call formula | When to use |
|--------|------------|-------------|-------------|
| Far boundary | $K/\hat{c}_2$ | $c_2 K$ | **Default** — current time segment |
| Near boundary | $K/\hat{c}_1$ | $c_1 K$ | Conservative (risk management) |

**Recommendation**: Use the far boundary for consistency with BS2002's own pricing formula.

### 4.10 BS2002 Pricing Formulas

The American option price is needed for the $V_i$ term in the PV formula. The BS2002 price is computed using auxiliary functions $\phi$ and $\psi$.

**Auxiliary function $\phi$** (univariate normal terms):

$$\phi(S, \tau, \gamma, H, I) = e^{\lambda}\,S^{\gamma}\left[\Phi(d_1) - \left(\frac{I}{S}\right)^{\!\kappa}\Phi(d_2)\right]$$

where:

$$\lambda = \bigl(-r + \gamma\,b + \tfrac{1}{2}\gamma(\gamma - 1)\sigma^2\bigr)\,\tau, \qquad \kappa = \frac{2b}{\sigma^2} + (2\gamma - 1)$$

$$d_1 = -\frac{\ln(S/H) + \bigl(b + (\gamma - \tfrac{1}{2})\sigma^2\bigr)\,\tau}{\sigma\sqrt{\tau}}, \qquad d_2 = -\frac{\ln(I^2/(S H)) + \bigl(b + (\gamma - \tfrac{1}{2})\sigma^2\bigr)\,\tau}{\sigma\sqrt{\tau}}$$

with $b = r - q$.

**Auxiliary function $\psi$** (bivariate normal terms):

$$\psi(S, \tau, \gamma, H, I_2, I_1, t_1) = e^{\lambda}\,S^{\gamma}\Bigl[\Phi_2(e_1, f_1;\;\rho) - \left(\frac{I_1}{S}\right)^{\!\kappa}\Phi_2(e_2, f_2;\;\rho)$$
$$\quad- \left(\frac{I_2}{S}\right)^{\!\kappa}\Phi_2(e_3, f_3;\;\rho) + \left(\frac{I_1 I_2}{S^2}\right)^{\!\kappa/2}\Phi_2(e_4, f_4;\;\rho)\Bigr]$$

where $\Phi_2(a, b;\;\rho)$ is the standard bivariate normal CDF with correlation $\rho = -\sqrt{t_1/\tau}$, and:

$$e_1 = -\frac{\ln(S/I_1) + \bigl(b + (\gamma - \tfrac{1}{2})\sigma^2\bigr)\,t_1}{\sigma\sqrt{t_1}}, \qquad f_1 = -\frac{\ln(S/I_2) + \bigl(b + (\gamma - \tfrac{1}{2})\sigma^2\bigr)\,\tau}{\sigma\sqrt{\tau}}$$

$$e_2 = -\frac{\ln(I_2^2/(S I_1)) + \bigl(b + (\gamma - \tfrac{1}{2})\sigma^2\bigr)\,t_1}{\sigma\sqrt{t_1}}, \qquad f_2 = -\frac{\ln(I_1^2/(S I_2)) + \bigl(b + (\gamma - \tfrac{1}{2})\sigma^2\bigr)\,\tau}{\sigma\sqrt{\tau}}$$

$$e_3 = -\frac{\ln(S/I_2) + \bigl(b + (\gamma - \tfrac{1}{2})\sigma^2\bigr)\,t_1}{\sigma\sqrt{t_1}}, \qquad f_3 = f_1$$

$$e_4 = -\frac{\ln(I_1 I_2 / S^2) + \bigl(b + (\gamma - \tfrac{1}{2})\sigma^2\bigr)\,t_1}{\sigma\sqrt{t_1}}, \qquad f_4 = f_2$$

**American Call Price** (direct, when $q > 0$):

Define call boundaries $I_1 = c_1 K$, $I_2 = c_2 K$ (Section 4.4).

If $S \geq I_2$: immediate exercise, $C_{\text{Am}} = S - K$.

Otherwise:

$$\alpha_1 = (I_1 - K)\,I_1^{-\beta_c}, \qquad \alpha_2 = (I_2 - K)\,I_2^{-\beta_c}$$

$$C_{\text{Am}} = \alpha_2\,S^{\beta_c} - \alpha_2\,\phi(S,\tau,\beta_c,I_2,I_2) + \phi(S,\tau,1,I_2,I_2) - \phi(S,\tau,1,I_1,I_2)$$
$$\quad - K\,\phi(S,\tau,0,I_2,I_2) + K\,\phi(S,\tau,0,I_1,I_2)$$
$$\quad + \alpha_1\,\phi(S,\tau,\beta_c,I_1,I_2) - \alpha_1\,\psi(S,\tau,\beta_c,I_1,I_2,I_1,t_1)$$
$$\quad + \psi(S,\tau,1,I_1,I_2,I_1,t_1) - \psi(S,\tau,1,K,I_2,I_1,t_1)$$
$$\quad - K\,\psi(S,\tau,0,I_1,I_2,I_1,t_1) + K\,\psi(S,\tau,0,K,I_2,I_1,t_1)$$

If $q \leq 0$: $C_{\text{Am}} = C_{\text{BS}}(S, K, \tau, r, q, \sigma)$ (European call, no early exercise premium).

**American Put Price** (via put-call symmetry):

$$P_{\text{Am}}(S, K, T, r, q, \sigma) = C_{\text{Am}}(K, S, T, q, r, \sigma)$$

That is, evaluate the call formula above with the substitutions $S \to K$, $K \to S$, $r \to q$, $q \to r$, $b \to q - r$, and use the put boundaries: $I_1 = \hat{c}_1 \cdot S$, $I_2 = \hat{c}_2 \cdot S$.

If $r \leq 0$: $P_{\text{Am}} = P_{\text{BS}}(S, K, \tau, r, q, \sigma)$ (European put).

### 4.11 Boundary Behavior

**Put** ($r > 0$):
- $\tau \to 0$: $B \to K$ (when $q \leq r$) or $B \to rK/q$ (when $q > r$)
- $\tau \to \infty$: $B \to K/\hat{c}_\infty$ (perpetual boundary)
- High $\sigma$: $|\beta_p| \downarrow$, $\hat{c}_\infty \uparrow$, $B \downarrow$ (more value in waiting)
- High $r$: $|\beta_p| \uparrow$, $\hat{c}_\infty \downarrow$, $B \uparrow$ (exercise sooner for interest on $K$)
- Negative $q$: stock grows faster → put less likely to be exercised → $B$ lower

**Call** ($q > 0$):
- $\tau \to 0$: $I \to K$ (when $r \leq q$) or $I \to (r/q)K$ (when $r > q$)
- $\tau \to \infty$: $I \to c_\infty K$ (perpetual boundary)
- High $\sigma$: $\beta_c \downarrow$, $c_\infty \uparrow$, $I \uparrow$ (more value in waiting)
- High $q$: $\beta_c \uparrow$, $c_\infty \downarrow$, $I \downarrow$ (exercise sooner for dividends)
- Within 7-day intervals: boundaries are nearly constant except very near expiry

### 4.12 Numerical Guards

```cpp
// Put: no early exercise if r <= 0
if (r <= 1e-12) { B_near = 0; B_far = 0; return; }

// Call: no early exercise if q <= 0
if (q <= 1e-12) { I_near = INF; I_far = INF; return; }

// Degenerate case: c_inf ≈ c_floor (σ → 0 and r ≈ q)
if (c_inf - c_floor < 1e-10) {
    // Boundary at floor value (immediate exercise optimal)
    B_near = K / c_floor;  // put
    // I_near = c_floor * K;  // call
    B_far = B_near;
    return;
}

// h guard: enforce non-positive
h = std::min(h, 0.0);

// Similarly for second step
if (c_inf - c_1 < 1e-10) {
    B_far = B_near;
    return;
}
```

---

## 5. Brownian Bridge Crossing Probability

### 5.1 Setup

At $t_{i-1}$ the option is alive (survival probability $Q_{i-1} > 0$). We observe $S_{i-1}$ and $S_i$ at the two endpoints. We want:

$$p_i = P\!\left(\min_{u \in [t_{i-1}, t_i]} S(u) \leq B_i \;\middle|\; S_{i-1},\; S_i,\; \text{variance path}\right)$$

### 5.2 Log-Space Formulation

Under Heston, the log-spot evolves as:

$$d\ln S = \left(r - q - \frac{v}{2}\right)dt + \sqrt{v}\,dW_S$$

Conditional on the full variance path $\{v(u)\}_{u \in [t_{i-1}, t_i]}$, the log-spot is a Gaussian process with time-varying volatility. The Brownian bridge for such a process uses the **integrated variance**:

$$\bar{\sigma}^2_i = \int_{t_{i-1}}^{t_i} v(u)\,du$$

### 5.3 Integrated Variance Approximation

We observe $v_{i-1}$ and $v_i$ but not the full variance path between them. We approximate the integrated variance.

**Trapezoidal rule** (simplest):

$$\bar{\sigma}^2_i \approx \frac{v_{i-1} + v_i}{2}\,\Delta t_i$$

**Conditional expectation** (accounts for mean reversion):

$$\bar{\sigma}^2_i \approx \mathbb{E}\left[\int_{t_{i-1}}^{t_i} v(u)\,du \;\middle|\; v_{i-1}\right] = \theta\,\Delta t_i + (v_{i-1} - \theta)\,\frac{1 - e^{-\kappa\,\Delta t_i}}{\kappa}$$

**Conditional on both endpoints** (bridge expectation, most accurate):

$$\bar{\sigma}^2_i \approx \mathbb{E}\left[\int_{t_{i-1}}^{t_i} v(u)\,du \;\middle|\; v_{i-1},\, v_i\right]$$

Under the CIR process, the bridge expectation is:

$$\bar{\sigma}^2_i \approx \frac{v_{i-1} + v_i}{2}\,\Delta t_i + \frac{\theta - \frac{v_{i-1} + v_i}{2}}{12}\,\kappa\,\Delta t_i^2$$

For 7-day intervals with typical equity parameters ($\kappa \sim 2$–$5$, $\Delta t \sim 0.019$), the $O(\Delta t^2)$ correction term is negligible. **The trapezoidal rule is sufficient.**

### 5.4 Crossing Probability Formula

For a put with exercise boundary $B_i$ (exercise if $S \leq B_i$), define $b_i = \ln B_i$.

**Case 1**: Both endpoints above the boundary ($S_{i-1} > B_i$ and $S_i > B_i$):

$$\boxed{p_i = \exp\!\left(-\frac{2\,\ln\!\left(\frac{S_{i-1}}{B_i}\right)\,\ln\!\left(\frac{S_i}{B_i}\right)}{\bar{\sigma}^2_i}\right)}$$

This is the exact Brownian bridge minimum crossing probability for a process with variance $\bar{\sigma}^2_i$, conditional on the endpoints. Both log-ratios are positive (since both spots are above $B_i$), making the exponent negative, giving $p_i \in (0, 1)$.

**Case 2**: At least one endpoint below or at the boundary ($S_{i-1} \leq B_i$ or $S_i \leq B_i$):

$$p_i = 1$$

If $S_i \leq B_i$, the spot is in the exercise region — the path must have crossed the boundary (since spot paths are continuous). If $S_{i-1} \leq B_i$, the option should have been exercised at $t_{i-1}$ (handled by $Q_{i-1} = 0$ in practice).

### 5.5 Derivation of the Crossing Formula

For completeness, here is the derivation. Let $X(t) = \ln S(t)$ and consider the bridge $X(t)$ conditional on $X(t_{i-1}) = x_0$ and $X(t_i) = x_1$, with total variance $\bar{\sigma}^2$.

The Brownian bridge $X(t)$ for $t \in [t_{i-1}, t_i]$ has the representation:

$$X(t) = x_0 + \frac{t - t_{i-1}}{\Delta t_i}(x_1 - x_0) + \bar{\sigma}\sqrt{\frac{(t - t_{i-1})(t_i - t)}{\Delta t_i}}\,Z$$

where $Z$ is standard normal, and the variance of the bridge at time $t$ is:

$$\text{Var}[X(t) \mid X(t_{i-1}), X(t_i)] = \bar{\sigma}^2 \cdot \frac{(t - t_{i-1})(t_i - t)}{\Delta t_i^2}$$

For a time-changed Brownian motion with integrated variance $\bar{\sigma}^2$, the standard Brownian bridge result applies in "variance time." The key identity for the minimum of a Brownian bridge:

$$P\!\left(\min_{t \in [t_{i-1}, t_i]} X(t) \leq b \;\middle|\; X(t_{i-1}) = x_0,\, X(t_i) = x_1\right) = \exp\!\left(-\frac{2(x_0 - b)(x_1 - b)}{\bar{\sigma}^2}\right)$$

for $x_0 > b$ and $x_1 > b$. This follows from the reflection principle applied to the Brownian bridge.

Substituting $x_0 = \ln S_{i-1}$, $x_1 = \ln S_i$, $b = \ln B_i$:

$$p_i = \exp\!\left(-\frac{2(\ln S_{i-1} - \ln B_i)(\ln S_i - \ln B_i)}{\bar{\sigma}^2_i}\right) = \exp\!\left(-\frac{2\,\ln\!\left(\frac{S_{i-1}}{B_i}\right)\,\ln\!\left(\frac{S_i}{B_i}\right)}{\bar{\sigma}^2_i}\right)$$

### 5.6 Sensitivity of $p_i$ to Inputs

Understanding the sensitivities helps calibrate the grid spacing:

$$\frac{\partial p_i}{\partial S_{i-1}} = -\frac{2\ln(S_i / B_i)}{S_{i-1}\,\bar{\sigma}^2_i}\,p_i$$

$$\frac{\partial p_i}{\partial S_i} = -\frac{2\ln(S_{i-1} / B_i)}{S_i\,\bar{\sigma}^2_i}\,p_i$$

$$\frac{\partial p_i}{\partial \bar{\sigma}^2_i} = \frac{2\ln(S_{i-1}/B_i)\,\ln(S_i/B_i)}{(\bar{\sigma}^2_i)^2}\,p_i$$

**Implications**:
- $p_i$ increases as either endpoint gets closer to $B_i$ (smaller log-ratio)
- $p_i$ increases with variance (wider bridge, more likely to touch barrier)
- $p_i$ approaches 1 as $\bar{\sigma}^2_i \to \infty$ or either $S \to B_i^+$
- $p_i$ approaches 0 as $\bar{\sigma}^2_i \to 0$ or either $S \to \infty$

### 5.7 Numerical Stability

The exponent $-2\ln(S_{i-1}/B_i)\ln(S_i/B_i)/\bar\sigma^2_i$ can underflow to $-\infty$ when both endpoints are far from the boundary (deep OTM). This is fine — $\exp(-\infty) = 0$, meaning no crossing. But avoid computing $\ln(S/B)$ when $S/B$ is exactly 1 (would give 0, and the formula gives $p = 1$, which is Case 2).

Guard:
```cpp
if (S_prev <= B || S_curr <= B) {
    p = 1.0;
} else {
    double log_ratio_prev = std::log(S_prev / B);
    double log_ratio_curr = std::log(S_curr / B);
    double exponent = -2.0 * log_ratio_prev * log_ratio_curr / sigma2_bar;
    p = std::exp(exponent);
}
```

---

## 6. Full PV Calculation

### 6.1 State Variables

On each path, we maintain:

| Variable | Type | Description |
|----------|------|-------------|
| $S_i$ | from simulation | Equity spot at $t_i$ |
| $v_i$ | from simulation | Heston variance at $t_i$ |
| $Q_i$ | accumulated | Survival probability at $t_i$ |
| $\text{PV}_i$ | output | Mark-to-market at $t_i$ |

### 6.2 Initialization

$$Q_0 = 1 \quad \text{(option is alive at inception)}$$

$$\text{PV}_0 = V_{\text{BS2002}}(S_0,\; \sigma_{\text{eff}}(t_0),\; K,\; r,\; q,\; T)$$

### 6.3 Recursive Update at $t_i$ (for $i = 1, 2, \ldots, N$)

**Step 1**: Compute effective volatility for BS2002:

$$\sigma^2_{\text{eff}}(t_i) = \theta + (v_i - \theta)\,\frac{1 - e^{-\kappa(T - t_i)}}{\kappa(T - t_i)}$$

**Step 2**: Compute exercise boundary from BS2002 (Section 4.3, using $\sigma = \sigma_{\text{eff}}(t_i)$ and $\tau = T - t_i$):

$$\beta_i = \frac{1}{2} - \frac{q-r}{\sigma^2} + \sqrt{\left(\frac{1}{2} - \frac{q-r}{\sigma^2}\right)^2 + \frac{2q}{\sigma^2}}, \qquad c_{\infty,i} = \frac{\beta_i}{\beta_i - 1}, \qquad c_{0} = \max\!\left(1, \frac{q}{r}\right)$$

$$h_i = -\bigl((q-r)\tau + 2\sigma\sqrt{\tau}\bigr)\,\frac{c_0}{c_{\infty,i} - c_0}, \qquad c_i = c_0 + (c_{\infty,i} - c_0)(1 - e^{h_i})$$

$$B_i = K / c_i$$

(This is the BS1993 single-boundary version. For the BS2002 two-piece version, see Section 4.3, Steps 5–8.)

**Step 3**: Compute integrated variance over $[t_{i-1}, t_i]$:

$$\bar{\sigma}^2_i = \frac{v_{i-1} + v_i}{2}\,\Delta t_i$$

**Step 4**: Compute crossing probability:

$$p_i = \begin{cases} 1 & \text{if } S_{i-1} \leq B_i \text{ or } S_i \leq B_i \\[6pt] \exp\!\left(-\dfrac{2\,\ln\!\left(\frac{S_{i-1}}{B_i}\right)\ln\!\left(\frac{S_i}{B_i}\right)}{\bar{\sigma}^2_i}\right) & \text{otherwise} \end{cases}$$

**Step 5**: Update survival probability:

$$Q_i = Q_{i-1}\,(1 - p_i)$$

**Step 6**: Compute BS2002 American option price (if option might be alive):

$$V_i = V_{\text{BS2002}}(S_i,\;\sigma_{\text{eff}}(t_i),\;K,\;r,\;q,\;T - t_i)$$

**Step 7**: Compute PV:

$$\boxed{\text{PV}(t_i) = Q_i \cdot V_i}$$

### 6.4 Interpretation

- **$Q_i = 1$**: No exercise has occurred (or is very unlikely) — full BS2002 value
- **$Q_i = 0$**: Option was definitely exercised — PV is zero (position closed)
- **$0 < Q_i < 1$**: Probabilistic mixture — option may or may not have been exercised, PV reflects the expected MTM

The PV correctly decays as the option accumulates exercise probability over time. This is the expected exposure profile that feeds into CVA/DVA calculations.

### 6.5 Properties of the PV

1. **Monotonicity of $Q_i$**: $Q_i \leq Q_{i-1}$ always (survival can only decrease). Once $Q_i = 0$, it stays zero.

2. **At expiry**: $Q_N \cdot V_N = Q_N \cdot (K - S_N)^+$ (the European payoff weighted by survival).

3. **Consistency check**: On paths where spot stays far above the boundary (deep OTM put), $p_i \approx 0$ for all $i$, $Q_i \approx 1$, and PV $\approx V_{\text{BS2002}}$ — correct, as the option was never near exercise.

4. **Consistency check**: On paths where spot crashes well below $B_i$ at some $t_j$, $p_j = 1$, $Q_j = 0$, and PV $= 0$ for all subsequent $t_i$ — correct, as the option was exercised.

---

## 7. Exercise Cash Flows

### 7.1 When Exercise Cash Flows Matter

For exposure calculations (CVA/PFE), the PV formula in Section 6 is typically sufficient — we care about the value of the outstanding derivative.

However, for:
- **PnL attribution**: Need to record when and at what level the option was exercised
- **Cash flow projection**: Need to forecast settlement amounts
- **Funding valuation (FVA)**: Need to know when cash changes hands

We need the exercise cash flows.

### 7.2 Exercise Probability Per Interval

The probability of exercise occurring specifically in interval $[t_{i-1}, t_i]$ is:

$$E_i = Q_{i-1} \cdot p_i$$

This is the probability of being alive at $t_{i-1}$ times the probability of crossing in the interval. Note:

$$\sum_{i=1}^{N} E_i + Q_N = 1$$

The option either exercised in some interval (total probability $1 - Q_N$) or survived to expiry (probability $Q_N$).

### 7.3 Exercise Value

At the optimal exercise boundary, the holder receives the intrinsic value. For a put exercised at $S = B_i$:

$$\phi_i = (K - B_i)^+$$

The exact crossing level is $B_i$ by construction — that's the boundary where exercise is optimal.

### 7.4 Expected Exercise Cash Flow

The expected (undiscounted) exercise cash flow in interval $i$:

$$\text{CF}_i = E_i \cdot (K - B_i)^+$$

Discounted to $t_0$:

$$\text{CF}_i^{\text{disc}} = E_i \cdot D(t_0, \bar{\tau}_i) \cdot (K - B_i)^+$$

where $\bar{\tau}_i$ is the expected exercise time within the interval and $D(t_0, \bar{\tau}_i) = e^{-r\bar{\tau}_i}$ is the discount factor.

### 7.5 Expected Exercise Time

The expected first passage time for a Brownian bridge, conditional on crossing, is:

$$\bar{\tau}_i = \mathbb{E}[\tau \mid \tau \in [t_{i-1}, t_i],\; S(t_{i-1}),\; S(t_i)]$$

For a Brownian bridge with endpoints $x_0 = \ln(S_{i-1}/B_i)$ and $x_1 = \ln(S_i/B_i)$ and variance $\bar\sigma^2_i$, the conditional expected crossing time (measured from $t_{i-1}$) is:

$$\mathbb{E}[\tau - t_{i-1} \mid \text{cross}] = \Delta t_i \cdot \frac{x_0}{x_0 + x_1}$$

This is the **harmonic-mean weighted midpoint**: if $S_{i-1}$ is closer to the boundary ($x_0$ small), the expected crossing time is earlier; if $S_i$ is closer ($x_1$ small), the crossing is expected later.

So:

$$\bar{\tau}_i = t_{i-1} + \Delta t_i \cdot \frac{\ln(S_{i-1}/B_i)}{\ln(S_{i-1}/B_i) + \ln(S_i/B_i)}$$

For practical purposes (7-day intervals), the **midpoint approximation** $\bar{\tau}_i \approx \frac{t_{i-1} + t_i}{2}$ introduces negligible discounting error (the difference is at most 3.5 days of discounting).

### 7.6 Total Option Value Decomposition

The value of the American put at $t_0$ on a given path can be decomposed as:

$$V(t_0) = \underbrace{\sum_{i=1}^{N} E_i \cdot D(t_0, \bar{\tau}_i) \cdot (K - B_i)^+}_{\text{early exercise value}} + \underbrace{Q_N \cdot D(t_0, T) \cdot (K - S_N)^+}_{\text{terminal payoff}}$$

Averaged over all paths, this gives the Monte Carlo estimate of the American option price, which should be close to the BS2002 price at $t_0$ (serving as a consistency check).

---

## 8. Extension to American Calls

The call boundary $I_i$ and pricing formulas are given in Section 4.4. Exercise is optimal when $S \geq I$ (boundary above spot). The key modifications relative to the put case are:

### 8.1 Crossing Probability for Calls

For a call, we need the probability that the **maximum** of the bridge exceeds $I_i$:

$$p_i^{\text{call}} = \begin{cases} 1 & \text{if } S_{i-1} \geq I_i \text{ or } S_i \geq I_i \\[6pt] \exp\!\left(-\dfrac{2\,\ln\!\left(\frac{I_i}{S_{i-1}}\right)\ln\!\left(\frac{I_i}{S_i}\right)}{\bar{\sigma}^2_i}\right) & \text{if } S_{i-1} < I_i \text{ and } S_i < I_i \end{cases}$$

The log-ratios $\ln(I_i/S)$ are positive when both endpoints are below the boundary, ensuring the exponent is negative. The formula follows from the Brownian bridge maximum crossing probability (reflection principle, same derivation as Section 5.5 with "max" instead of "min").

### 8.2 Exercise Value for Calls

$$\phi_i^{\text{call}} = (I_i - K)^+$$

### 8.3 When Calls Exercise Early

American calls exercise early only when $q > 0$ (Section 4.5). If $q \leq 0$, set $I = +\infty$ and skip the bridge correction entirely — the call is equivalent to European.

---

## 9. Boundary Choice for the Crossing Check

### 9.1 The Boundary Is Time-Dependent

BS2002's boundary $B(t)$ varies over the remaining life. Between $t_{i-1}$ and $t_i$, the "true" boundary is a curve, not a constant. The Brownian bridge formula assumes a **flat** barrier. We must choose a single $B_i$ for each interval.

### 9.2 Options

| Choice | Formula | Conservatism | When to use |
|--------|---------|-------------|-------------|
| End-of-interval | $B_i = B(t_i)$ | Slightly conservative near expiry | Default choice |
| Start-of-interval | $B_i = B(t_{i-1})$ | Slightly anti-conservative near expiry | When boundary is falling |
| Maximum | $B_i = \max(B(t_{i-1}), B(t_i))$ | Conservative | For risk management |
| Average | $B_i = \frac{1}{2}(B(t_{i-1}) + B(t_i))$ | Neutral | General purpose |

### 9.3 Recommendation

For a put, $B(t)$ is **increasing** as $t \to T$ (boundary rises toward $K$ near expiry). The maximum over the interval is $B(t_i) = B_i$. Using the end-of-interval boundary:

- Is consistent with the Brownian bridge formula (flat barrier at the highest point)
- Gives a conservative crossing probability (more exercises detected)
- Is the simplest to implement (only need BS2002 at $t_i$, not $t_{i-1}$)

**Near expiry** (last 2–4 weeks), the boundary changes rapidly within a 7-day interval. Consider:
1. Halving the time step (3.5-day intervals)
2. Using the average boundary
3. Computing BS2002 at both endpoints and using the max

### 9.4 Piecewise Linear Boundary (Advanced)

For higher accuracy, linearly interpolate the boundary within each interval:

$$\tilde{B}(u) = B_{i-1} + \frac{u - t_{i-1}}{\Delta t_i}(B_i - B_{i-1}), \quad u \in [t_{i-1}, t_i]$$

The crossing probability for a linearly moving barrier is more complex. An effective approximation is to use the **maximum** of the interpolated boundary:

$$B_i^{\max} = \max(B_{i-1}, B_i)$$

and apply the flat-barrier formula with $B_i^{\max}$. This is conservative but simple.

---

## 10. Accuracy and Error Analysis

### 10.1 Sources of Error

| Source | Magnitude | Can improve by |
|--------|-----------|---------------|
| BS2002 boundary approximation | ~1–3% of option value | Use better boundary (e.g., integral equation solver) |
| Flat barrier within interval | $O(\Delta t^2)$ | Finer time grid, interpolated boundary |
| Trapezoidal integrated variance | $O(\Delta t^2)$ | Bridge-conditional expectation (Section 5.3) |
| GBM Brownian bridge under Heston | $O(\xi\sqrt{\Delta t})$ | Negligible for 7-day intervals |
| $\sigma_{\text{eff}}$ approximation | Model-dependent | Moment-matched vol (Section 3.5) |

### 10.2 Dominant Error

The **BS2002 boundary approximation** is the dominant error. BS2002 is known to be accurate to within 1–3 basis points for typical equity parameters, but the boundary itself can differ from the true optimal boundary by a few percent, particularly:
- Deep ITM (boundary close to $K$)
- Near expiry (boundary changes rapidly)
- High dividend yield (boundary has complex structure)

The Brownian bridge correction itself is **exact** given a flat barrier and GBM dynamics. The approximation error from using it under Heston with the trapezoidal integrated variance is of order $\xi \cdot \Delta t$, which is negligible for weekly intervals.

### 10.3 Convergence Behavior

As $\Delta t \to 0$ (finer time grid), the scheme converges to continuous monitoring:
- $p_i \to 0$ for each interval (less chance to cross in a shorter interval)
- But the cumulative effect $1 - Q_N = 1 - \prod(1 - p_i)$ converges to the true continuous-monitoring exercise probability
- The rate of convergence is $O(\Delta t)$ for the flat-barrier approximation

### 10.4 Validation Strategy

1. **Static test**: For a single timepoint ($N = 1$, European-style), verify that $Q_1 \cdot V_{\text{BS2002}}$ matches a known reference price for a Bermudan option with one exercise date.

2. **Continuous-monitoring limit**: With very fine time steps ($\Delta t = 1$ day), the scheme should converge to the BS2002 price at $t_0$ (since BS2002 assumes continuous monitoring).

3. **Boundary cases**:
   - Deep OTM: $Q_i \approx 1$ for all $i$, PV $\approx$ European price
   - Deep ITM: $Q_i$ drops to 0 quickly, early exercise dominates
   - At the boundary: $p_i$ should be close to 0.5

4. **Monte Carlo consistency**: Average PV at $t_0$ over all paths should approximate the BS2002 price (since BS2002 is our pricing model).

---

## 11. Implementation

### 11.1 Data Structures

```cpp
struct AmericanOptionState {
    double survival_prob;    // Q_i: probability option is still alive
    double prev_spot;        // S_{i-1}: spot at previous timepoint
    double prev_var;         // v_{i-1}: variance at previous timepoint
};

struct AmericanOptionResult {
    double pv;               // Q_i * V_BS2002
    double survival_prob;    // Q_i
    double crossing_prob;    // p_i (this interval only)
    double exercise_prob;    // E_i = Q_{i-1} * p_i
    double exercise_cf;      // E_i * (K - B_i)^+
    double boundary;         // B_i
};
```

### 11.2 Core Algorithm

```cpp
template<typename DoubleT>
AmericanOptionResult priceAtTimepoint(
    DoubleT S_curr,          // current spot
    DoubleT v_curr,          // current Heston variance
    double t_curr,           // current time
    double T,                // option maturity
    double K,                // strike
    double r,                // risk-free rate
    double q,                // dividend yield
    double kappa,            // Heston mean reversion
    double theta,            // Heston long-run variance
    AmericanOptionState& state)
{
    AmericanOptionResult result;
    double dt = t_curr - state.prev_time;
    double tau = T - t_curr;

    // Step 1: Effective vol for BS2002
    DoubleT sigma2_eff = theta + (v_curr - theta)
                         * (1.0 - std::exp(-kappa * tau)) / (kappa * tau);
    DoubleT sigma_eff = std::sqrt(sigma2_eff);

    // Step 2: BS2002 price and exercise boundary
    auto [V_bs, B] = bs2002_price_and_boundary(S_curr, K, tau, r, q, sigma_eff);
    result.boundary = B;

    // Step 3: Integrated variance (trapezoidal)
    DoubleT sigma2_bar = 0.5 * (state.prev_var + v_curr) * dt;

    // Step 4: Crossing probability
    if (state.prev_spot <= B || S_curr <= B) {
        result.crossing_prob = 1.0;
    } else {
        DoubleT log_prev = std::log(state.prev_spot / B);
        DoubleT log_curr = std::log(S_curr / B);
        result.crossing_prob = std::exp(-2.0 * log_prev * log_curr / sigma2_bar);
    }

    // Step 5: Update survival
    result.survival_prob = state.survival_prob * (1.0 - result.crossing_prob);
    result.exercise_prob = state.survival_prob * result.crossing_prob;
    result.exercise_cf = result.exercise_prob * std::max(K - B, 0.0);

    // Step 6: PV
    result.pv = result.survival_prob * V_bs;

    // Update state for next timepoint
    state.survival_prob = result.survival_prob;
    state.prev_spot = S_curr;
    state.prev_var = v_curr;
    state.prev_time = t_curr;

    return result;
}
```

### 11.3 Integration with the Pricing Loop

```cpp
// Setup
AmericanOptionState state;
state.survival_prob = 1.0;
state.prev_spot = S_0;
state.prev_var = v_0;
state.prev_time = 0.0;

for (int path = 0; path < N_paths; ++path) {
    state.survival_prob = 1.0;   // reset per path

    for (int i = 1; i <= N_timepoints; ++i) {
        double S_i = scenarioData.getSpot(path, i);
        double v_i = scenarioData.getVariance(path, i);
        double t_i = timepoints[i];

        auto result = priceAtTimepoint(S_i, v_i, t_i, T, K, r, q,
                                        kappa, theta, state);

        // Store for exposure calculation
        exposure(path, i) = result.pv;
        exercise_cf(path, i) = result.exercise_cf;
    }
}
```

### 11.4 AD Compatibility

The algorithm is fully compatible with the Hessian functor design (see `hessian_ad_design.md`). The `AmericanOptionState` contains only `double` values (survival probability, previous spot/variance) — these are path history, not market data. Under `rebind<T>()`:

- `state.survival_prob` → `T(value)` (constant, zero tangent)
- `state.prev_spot` → `T(value)` (constant — this was the *previous* timepoint's spot, not current market data)
- `state.prev_var` → `T(value)` (same reasoning)

The Hessian at $t_i$ differentiates through `S_curr`, `v_curr`, and the market data in `sigma_eff` and `V_bs` — not through the accumulated state.

The `std::log`, `std::exp`, `std::sqrt` calls should use `stan::math` equivalents (or `using` declarations) for AD compatibility:

```cpp
using stan::math::log;
using stan::math::exp;
using stan::math::sqrt;
```

---

## 12. Performance Considerations

### 12.1 Cost Per (Path, Timepoint)

| Operation | Cost |
|-----------|------|
| $\sigma_{\text{eff}}$ computation | ~5 flops (1 exp, 1 div, 1 mult) |
| BS2002 price + boundary | ~50–100 flops (the dominant cost) |
| Integrated variance | ~3 flops |
| Crossing probability | ~10 flops (2 logs, 1 exp, 1 mult, 1 div) |
| Survival update + PV | ~5 flops |
| **Total** | **~80–120 flops** |

BS2002 is the bottleneck. For comparison, a single BS vanilla price is ~30 flops. BS2002 involves two BS vanilla evaluations plus the early exercise premium calculation.

### 12.2 Amortization

For $M = 10{,}000$ paths and $N = 52$ weekly timepoints (1-year option), the total cost is:
- $10{,}000 \times 52 \times 120 \approx 62M$ flops
- At 10 GFlops (single-core), this is ~6 ms

This is negligible compared to the cost of generating the Heston paths themselves.

### 12.3 Hessian Cost

With $N_{\text{nodes}} = 200$ market data nodes, the Hessian adds $200\times$ the cost of a single evaluation (one forward-over-reverse sweep per column). Using analytical primitives for BS2002 (see `hessian_ad_design.md`, Section 8):

- Naive AD: ~120 tape nodes per BS2002 call → ~24K tape operations per Hessian column
- Analytical BS2002 primitive: ~20 tape nodes → ~4K tape operations per column
- Total Hessian: ~4K × 200 = 800K operations ≈ microseconds

---

## 13. Extensions

### 13.1 Discrete Dividends

For stocks with discrete dividends at known dates $t_{d_1}, t_{d_2}, \ldots$:
- The exercise boundary has jumps at ex-dividend dates
- Before ex-date: boundary accounts for upcoming dividend (spot will drop)
- After ex-date: standard boundary

**Adjustment**: At each ex-dividend date $t_d$ falling in $[t_{i-1}, t_i]$, split the interval into two sub-intervals at $t_d$. Apply the bridge correction separately to each sub-interval, with the appropriate boundary and the spot jump at $t_d$.

### 13.2 Term Structure of Rates

BS2002 accepts a single flat rate $r$. For a term-structure, use the **annualized forward rate** over the remaining life:

$$r_{\text{eff}}(t_i) = -\frac{\ln D(t_i, T)}{T - t_i}$$

where $D(t_i, T)$ is the discount factor from $t_i$ to $T$, read from the yield curve at $t_i$.

### 13.3 Multiple Exercise Boundaries (Bermudan)

For Bermudan options with exercise dates $\{T_1, \ldots, T_M\} \subset \{t_0, \ldots, t_N\}$:
- The Brownian bridge correction only applies between consecutive exercise dates
- Between non-exercise timepoints, $p_i = 0$ (cannot exercise)
- The boundary $B_i$ is only defined at exercise dates

### 13.4 Jump-Diffusion

Under Merton-style jump-diffusion, the Brownian bridge formula underestimates the crossing probability (jumps can cross the barrier without passing through intermediate levels). Corrections:
- Add jump contribution: $p_i^{\text{jump}} = 1 - e^{-\lambda\Delta t_i} + e^{-\lambda\Delta t_i}\,p_i^{\text{bridge}}$
- Or use the full jump-diffusion first passage time distribution (significantly more complex)

### 13.5 Barrier Monitoring Offset (Broadie-Glasserman-Kou)

The Brownian bridge approach can be combined with the Broadie-Glasserman-Kou (1997) barrier shift for additional accuracy. BGK shifts the barrier to account for discrete monitoring bias:

$$B_i^{\text{shifted}} = B_i \cdot e^{-\beta\sqrt{\bar\sigma^2_i / (n\,\Delta t_i)}}$$

where $\beta \approx 0.5826$ (Zeta function correction) and $n$ is the number of monitoring sub-intervals. For the Brownian bridge approach (which already accounts for continuous crossing), this shift is not needed — it would double-count the correction. The BGK shift is for when you do **not** use the bridge.

---

## Appendix A: Complete Formulas Summary

For quick reference, the complete set of formulas under Heston, covering both puts and calls.

**Inputs per timepoint**: $S_i$, $v_i$, $t_i$, and state $(Q_{i-1}, S_{i-1}, v_{i-1})$.

**Parameters**: $K$, $T$, $r$, $q$, $\kappa$, $\theta$.

### A.1 Effective Volatility (Heston → BS2002)

$$\sigma^2_{\text{eff},i} = \theta + (v_i - \theta)\,\frac{1 - e^{-\kappa(T-t_i)}}{\kappa(T-t_i)}, \qquad \sigma_i = \sqrt{\sigma^2_{\text{eff},i}}$$

### A.2 Exercise Boundary ($\tau_i = T - t_i$, $b = r - q$)

**Put** (requires $r > 0$, else $B = 0$):

$$\beta_p = \frac{1}{2} - \frac{b}{\sigma_i^2} - \sqrt{\left(\frac{1}{2} - \frac{b}{\sigma_i^2}\right)^2 + \frac{2r}{\sigma_i^2}}, \qquad \hat{c}_\infty = 1 + \frac{1}{|\beta_p|}$$

$$\hat{c}_0 = \begin{cases} \max(1,\; q/r) & q > 0 \\ 1 & q \leq 0 \end{cases}$$

$$h = \min\!\left(-\bigl(b\,\tau_i + 2\sigma_i\sqrt{\tau_i}\bigr)\;\frac{\hat{c}_0}{\hat{c}_\infty - \hat{c}_0},\;\; 0\right)$$

$$\hat{c} = \hat{c}_0 + (\hat{c}_\infty - \hat{c}_0)(1 - e^{h}), \qquad B_i = K / \hat{c}$$

(For BS2002 two-piece: apply two-step procedure from Section 4.3, Steps 4–8.)

**Call** (requires $q > 0$, else $I = +\infty$):

$$\beta_c = \frac{1}{2} - \frac{b}{\sigma_i^2} + \sqrt{\left(\frac{1}{2} - \frac{b}{\sigma_i^2}\right)^2 + \frac{2r}{\sigma_i^2}}, \qquad c_\infty = \frac{\beta_c}{\beta_c - 1}$$

$$c_0 = \max(1,\; r/q)$$

$$h = \min\!\left(-\bigl(b\,\tau_i + 2\sigma_i\sqrt{\tau_i}\bigr)\;\frac{c_0}{c_\infty - c_0},\;\; 0\right)$$

$$c = c_0 + (c_\infty - c_0)(1 - e^{h}), \qquad I_i = c \cdot K$$

### A.3 Integrated Variance (trapezoidal)

$$\bar{\sigma}^2_i = \tfrac{1}{2}(v_{i-1} + v_i)\,\Delta t_i$$

### A.4 Crossing Probability

**Put** (exercise when $S \leq B$):

$$p_i = \begin{cases} 1 & S_{i-1} \leq B_i \text{ or } S_i \leq B_i \\[4pt] \exp\!\Bigl(-\dfrac{2\ln(S_{i-1}/B_i)\,\ln(S_i/B_i)}{\bar\sigma^2_i}\Bigr) & \text{otherwise} \end{cases}$$

**Call** (exercise when $S \geq I$):

$$p_i = \begin{cases} 1 & S_{i-1} \geq I_i \text{ or } S_i \geq I_i \\[4pt] \exp\!\Bigl(-\dfrac{2\ln(I_i/S_{i-1})\,\ln(I_i/S_i)}{\bar\sigma^2_i}\Bigr) & \text{otherwise} \end{cases}$$

### A.5 Survival, Exercise, and PV

$$Q_i = Q_{i-1}\,(1 - p_i), \qquad E_i = Q_{i-1}\,p_i$$

$$V_i = V_{\text{BS2002}}(S_i,\;\sigma_i,\;K,\;r,\;q,\;\tau_i) \qquad \text{(Section 4.10)}$$

$$\boxed{\text{PV}(t_i) = Q_i \cdot V_i}$$

$$\text{CF}_i^{\text{put}} = E_i \cdot (K - B_i)^+, \qquad \text{CF}_i^{\text{call}} = E_i \cdot (I_i - K)^+$$
