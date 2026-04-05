# Multi-Curve Hierarchical Risk Transformation

## 1. Problem Statement

Our pricing engine computes portfolio value $V$ as a function of rate curve data.
Internally, every curve is stored as either:

- **Base curve (SOFR):** a vector of zero rates $\mathbf{z}^{(0)}$ at standard tenor nodes.
- **Spread curves:** a vector of spreads $\mathbf{s}^{(c)}$ over a parent curve.

At pricing time the engine reconstructs the full zero rate for any curve $c > 0$ via
$z^{(c)}_i = \widetilde{z}^{(\pi(c))}(t^{(c)}_i) + s^{(c)}_i$, but the **inputs
to the AD tape** depend on which parameterisation is used. There are two cases:

**Case A — AD inputs are $\mathbf{\theta} = (z^{(0)}, s^{(1)}, \ldots, s^{(C)})$.**
The engine takes base zeros and spreads as `stan::math::var` inputs and reconstructs
each curve's zeros on the tape. Stan's reverse-mode AD then gives:

$$
\bar{z}^{(0)}_i = \frac{\partial V}{\partial z^{(0)}_i}, \qquad
\bar{s}^{(c)}_i = \frac{\partial V}{\partial s^{(c)}_i} \quad (c > 0)
$$

In this case the spread propagation through the curve tree (Section 4) is handled
**automatically by the AD tape** — the reconstruction $z^{(c)} = \text{interp}(z^{(\pi)}) + s^{(c)}$
is recorded, and the reverse pass chains through it. The adjoint $\bar{z}^{(0)}$
already includes contributions from all descendant curves. The only post-processing
needed is the per-curve triangular solve (Section 5) to convert from
$\partial V/\partial \mathbf{\theta}$ to $\partial V/\partial \mathbf{r}$.

**Case B — AD inputs are $\mathbf{\zeta} = (z^{(0)}, z^{(1)}, \ldots, z^{(C)})$.**
The engine takes the fully resolved zero rates for every curve as independent
`stan::math::var` inputs. Stan's reverse-mode AD gives:

$$
\bar{z}^{(c)}_i = \frac{\partial V}{\partial z^{(c)}_i} \quad \forall\, c
$$

These are partial derivatives treating each curve's zeros as independent — they do
**not** account for the spread structure. Before the triangular solve, we must first
propagate adjoints through the curve tree (Section 4) to recover
$\partial V/\partial \mathbf{\theta}$.

The remainder of this document derives the full transformation for both cases. The
target in either case is the same: sensitivities to the **bootstrap instruments** —
the FRAs, futures, par swaps, and basis swaps that were actually used to construct
the curves.

---

## 2. Curve Hierarchy

### 2.1 Definitions

Let $\mathcal{C} = \{0, 1, \ldots, C\}$ be the set of curves arranged as a rooted
tree. Curve $0$ is the base (e.g. SOFR). Every other curve $c > 0$ has a unique
parent $\pi(c) \in \mathcal{C}$ and is defined as a spread over that parent.

Each curve $c$ has $n_c$ nodes at times $\mathbf{t}^{(c)} = (t^{(c)}_1, \ldots, t^{(c)}_{n_c})$
with zero rates $\mathbf{z}^{(c)} = (z^{(c)}_1, \ldots, z^{(c)}_{n_c})$.

The zero rate of curve $c > 0$ at an arbitrary time $t$ is reconstructed from the
parent curve and a spread. The parent zeros and the spreads are interpolated
**independently** on their own node grids (which need not coincide), then combined.
The exact reconstruction depends on the spread convention used by the engine:

**Additive spread model** (believed to be our engine's convention). The interpolated
spread is added to the interpolated parent zero rate:

$$
\widetilde{z}^{(c)}(t) = \widetilde{z}^{(\pi(c))}(t) + \widetilde{s}^{(c)}(t)
$$

At the spread curve's own node times: $z^{(c)}_i = \widetilde{z}^{(\pi(c))}(t^{(c)}_i) + s^{(c)}_i$.

**Multiplicative (proportional) spread model.** The spread scales the interpolated
parent zero rate:

$$
\widetilde{z}^{(c)}(t) = \widetilde{z}^{(\pi(c))}(t) \cdot (1 + \widetilde{s}^{(c)}(t))
$$

At the spread curve's own node times: $z^{(c)}_i = \widetilde{z}^{(\pi(c))}(t^{(c)}_i) \cdot (1 + s^{(c)}_i)$.

In both cases, $\widetilde{z}^{(\pi(c))}(t)$ denotes the interpolated parent zero
rate at time $t$, and $\widetilde{s}^{(c)}(t)$ denotes the interpolated spread at
time $t$. For the base curve, $\mathbf{z}^{(0)}$ is the independent parameter
directly (no parent, no spread).

The choice of spread model affects the **beta coefficients** — the partial derivatives
that link sensitivities across the curve tree. We define:

- **Spread beta** $\beta^{s} = \partial z^{(c)}_i / \partial s^{(c)}_i$ — scales the
  spread adjoint into a zero-rate adjoint.
- **Propagation beta** $\beta^{\pi} = \partial z^{(c)}_i / \partial \widetilde{z}^{(\pi)}_i$
  — scales the child adjoint before accumulating into the parent (the full partial
  w.r.t. parent node $j$ is $\beta^{\pi} \cdot (P_c)_{ij}$, where $(P_c)_{ij}$ is
  the interpolation weight from Section 4.1).

### 2.1.1 Beta coefficients for common spread conventions

The table below lists all standard spread conventions and their betas. Only the first
two are clean enough for practical use in a zero-rate-based engine; the rest are
included for completeness.

| Spread model | Reconstruction $z^{(c)}$ | $\beta^{s}$ | $\beta^{\pi}$ | Notes |
|---|---|---|---|---|
| **Additive on zeros** | $z_\pi + s$ | $1$ | $1$ | Simplest; believed to be our engine's convention. All betas trivial. |
| **Multiplicative on zeros** | $z_\pi \cdot (1 + s)$ | $z_\pi$ | $1 + s$ | Propagation beta is $1+s$. Spread beta depends on parent level. |
| **Multiplicative on $(1+\text{rate})$** | $(1+z_\pi)(1+s) - 1$ | $1 + z_\pi$ | $1 + s$ | Same propagation beta as multiplicative on zeros, but spread beta depends on parent level differently. |
| **Additive on log-DF** | $z_\pi \cdot t_\pi / t + s$ | $1$ | $t_\pi / t$ | Equivalent to additive on zeros only if $t_\pi = t$ (coincident nodes). The time ratio appears because $\ln D_\pi = -z_\pi t_\pi$. |
| **Multiplicative on DF** | $z_\pi + s$ (via $D = D_\pi e^{-st}$) | $1$ | $1$ | Additive on zeros in disguise — the exponential cancels. |
| **Simple-rate DF spread** | $-\ln(D_\pi/(1+s\tau))/t$ | $\frac{\tau}{t(1+s\tau)}$ | $\frac{z_\pi t_\pi}{z^{(c)} t}$ | Messy; depends on the resulting zero rate. Uncommon for zero-rate engines. |
| **Additive on forwards** | $\frac{1}{t}\int_0^t (f_\pi(u) + s(u))\,du$ | nonlocal | nonlocal | Perturbing a forward spread at one tenor affects all longer zero rates. Not compatible with node-local betas. |

For our purposes, the additive and multiplicative-on-zeros models are the relevant
cases. If your engine's beta is exactly $1$, you are in the additive model. If it
is $1 + s$, you are in the multiplicative model (propagation beta).

### 2.2 Example

```
SOFR (c=0, base zeros)
├── OIS (c=1) :  z^(1)(t) = f(interp_z(z^(0), t), interp_s(s^(1), t))
│   └── Funding (c=2) :  z^(2)(t) = f(interp_z(z^(1), t), interp_s(s^(2), t))
├── LIBOR 3M (c=3) :  z^(3)(t) = f(interp_z(z^(0), t), interp_s(s^(3), t))
└── EUR xccy (c=4) :  z^(4)(t) = f(interp_z(z^(0), t), interp_s(s^(4), t))
    └── EUR credit (c=5) :  z^(5)(t) = f(interp_z(z^(4), t), interp_s(s^(5), t))
```

where $f(z_\pi, s) = z_\pi + s$ (additive) or $f(z_\pi, s) = z_\pi (1 + s)$
(multiplicative). The `interp_z` and `interp_s` operations interpolate on their
respective node grids independently.

### 2.3 Discount factors

Throughout, discount factors are derived from zero rates via:

$$
D^{(c)}(t) = \exp\bigl(-\widetilde{z}^{(c)}(t) \cdot t\bigr)
$$

At node times: $D^{(c)}_i = \exp(-z^{(c)}_i \, t^{(c)}_i)$.

The partial derivative of a discount factor with respect to its own zero rate node is:

$$
\frac{\partial D^{(c)}_i}{\partial z^{(c)}_i} = -t^{(c)}_i \, D^{(c)}_i
$$

This identity appears throughout the instrument Jacobian derivations below.

---

## 3. Three Vector Spaces

We identify three coordinate systems of the same total dimension $N = \sum_c n_c$.

| Symbol            | Name | Components |
|-------------------|------|------------|
| $\mathbf{\zeta}$  | Zero rates | $(z^{(0)}, z^{(1)}, \ldots, z^{(C)})$ — fully resolved zeros per curve |
| $\mathbf{\theta}$ | Independent parameters | $(z^{(0)}, s^{(1)}, s^{(2)}, \ldots, s^{(C)})$ — base zeros concatenated with all spread vectors; one entry per bootstrap DOF (see Section 3.1) |
| $\mathbf{r}$      | Instrument quotes | $(r^{(0)}, r^{(1)}, \ldots, r^{(C)})$ — FRA rates, swap rates, basis spreads |

### 3.1 What $\mathbf{\theta}$ is (and is not)

$\mathbf{\theta}$ is **not** a new quantity — it is simply the concatenation of the
data the engine already stores: base curve zeros and spread vectors. Concretely:

$$
\mathbf{\theta} = \bigl(\underbrace{z^{(0)}_1, \ldots, z^{(0)}_{n_0}}_{\text{SOFR zeros}},\;
\underbrace{s^{(1)}_1, \ldots, s^{(1)}_{n_1}}_{\text{OIS spreads}},\;
\underbrace{s^{(2)}_1, \ldots, s^{(2)}_{n_2}}_{\text{funding spreads}},\;
\ldots\bigr)
$$

The block for the base curve ($c = 0$) is zeros: $\theta^{(0)} = z^{(0)}$.
The block for every other curve ($c > 0$) is spreads: $\theta^{(c)} = s^{(c)}$.

By contrast, $\mathbf{\zeta}$ stacks the **fully resolved** zero rates per curve.
For $c > 0$, $z^{(c)}_i = \widetilde{z}^{(\pi(c))}(t^{(c)}_i) + s^{(c)}_i$
includes the parent contribution, so $\mathbf{\zeta}$ has redundancy — changing
$z^{(0)}$ without adjusting $z^{(c)}$ would break the spread relationship.
$\mathbf{\theta}$ has no such redundancy: every entry is an independent degree of
freedom.

Our engine internally computes **beta coefficients** that link derivatives with
respect to the fully resolved zeros back to derivatives with respect to the
independent parameters (base zeros and spreads). These betas encode the same
relationship as the $L$ matrix in Section 4 — specifically, the interpolation
weights that describe how a parent curve's zeros flow into a child curve's
reconstructed zeros. Whether this conversion is done on the AD tape (Case A) or
as a post-processing step (Case B) depends on how the engine parameterises its
AD inputs. The end result is the same: $\partial V / \partial \mathbf{\theta}$.

### 3.2 Pricing chain

The full pricing chain is:

$$
\mathbf{r} \xrightarrow{\text{bootstrap}} \mathbf{\theta} \xrightarrow{\text{spread addition}} \mathbf{\zeta} \xrightarrow{\text{pricing}} V
$$

The relationship between $\mathbf{\theta}$ and $\mathbf{\zeta}$ is invertible
(Section 4). The distinction matters because the AD engine may differentiate against
either one:

**Case A** ($\mathbf{\theta}$ on tape). The engine reconstructs $\mathbf{\zeta}$
from $\mathbf{\theta}$ internally, so the tape already encodes $\mathbf{\theta} \to V$.
Stan gives $\partial V / \partial \mathbf{\theta}$ directly, and:

$$
\frac{\partial V}{\partial \mathbf{r}}
= \operatorname{diag}\bigl((F^{(c)})^{-\top}\bigr)
  \cdot \frac{\partial V}{\partial \mathbf{\theta}}
$$

This is just the per-curve triangular solve (Section 5). No tree propagation needed.

**Case B** ($\mathbf{\zeta}$ on tape). The engine takes fully resolved zeros as
independent inputs. Stan gives $\partial V / \partial \mathbf{\zeta}$, and:

$$
\frac{\partial V}{\partial \mathbf{r}}
= \operatorname{diag}\bigl((F^{(c)})^{-\top}\bigr) \cdot L^\top
  \cdot \frac{\partial V}{\partial \mathbf{\zeta}}
$$

where $L = \partial\mathbf{\zeta}/\partial\mathbf{\theta}$ is the spread propagation
matrix (Section 4). The product $L^\top \cdot \partial V/\partial\mathbf{\zeta}$
first recovers $\partial V/\partial\mathbf{\theta}$ by accumulating the parent curve
contributions that the AD tape did not see.

In both cases the final step is the same: solve $(F^{(c)})^\top \mathbf{x} = \bar{\mathbf{g}}^{(c)}$
per curve, where $\bar{\mathbf{g}}^{(c)}$ is the sensitivity to $\mathbf{\theta}^{(c)}$.

We derive $L$ in Section 4 and $F^{(c)}$ in Section 5.

---

## 4. Spread Propagation: $L = \partial\mathbf{\zeta}/\partial\mathbf{\theta}$

> **Case A shortcut.** If the engine takes $\mathbf{\theta}$ as AD inputs (base zeros
> and spreads), this entire section is handled automatically by the AD tape. The
> reconstruction $z^{(c)} = f(\text{interp}(z^{(\pi)}),\, s^{(c)})$ is recorded on
> the tape, and Stan's reverse pass chains through the correct betas regardless of
> whether the spread model is additive or multiplicative. The adjoint $\bar{z}^{(0)}$
> already includes all descendant contributions (with the appropriate beta scaling),
> and each $\bar{s}^{(c)}$ is already the correct $\partial V / \partial s^{(c)}$.
> Skip to Section 5.
>
> The material below is needed only for **Case B** (engine takes fully resolved
> zeros $\mathbf{\zeta}$ as independent AD inputs).

### 4.1 Interpolation weight matrix

For each non-root curve $c$, define the **interpolation weight matrix**
$P_c \in \mathbb{R}^{n_c \times n_{\pi(c)}}$:

$$
(P_c)_{ij} = \frac{\partial}{\partial z^{(\pi(c))}_j}
              \widetilde{z}^{(\pi(c))}(t^{(c)}_i)
$$

This encodes how node $j$ of the parent curve affects the interpolated parent zero
at the time of node $i$ of curve $c$. Its structure depends on the interpolation
method:

**Linear interpolation on zero rates.** Suppose $t^{(\pi(c))}_k \le t^{(c)}_i < t^{(\pi(c))}_{k+1}$.
Let $\alpha = (t^{(c)}_i - t^{(\pi(c))}_k) / (t^{(\pi(c))}_{k+1} - t^{(\pi(c))}_k)$. Then:

$$
(P_c)_{i,k} = 1 - \alpha, \qquad (P_c)_{i,k+1} = \alpha
$$

and all other entries in row $i$ are zero. $P_c$ is bidiagonal (at most 2 nonzeros per
row, summing to 1).

**Log-linear interpolation on discount factors.** The interpolated DF is
$D(t) = D_k^{1-\beta} \, D_{k+1}^{\beta}$ where $\beta = (t - t_k)/(t_{k+1} - t_k)$.
Converting to zero rates:

$$
z(t) \cdot t = (1-\beta) \, z_k \, t_k + \beta \, z_{k+1} \, t_{k+1}
$$

$$
(P_c)_{i,k} = \frac{(1-\beta) \, t^{(\pi)}_k}{t^{(c)}_i}, \qquad
(P_c)_{i,k+1} = \frac{\beta \, t^{(\pi)}_{k+1}}{t^{(c)}_i}
$$

Still bidiagonal, but rows no longer sum to 1. They sum to
$(1-\beta) t_k / t_i + \beta \, t_{k+1} / t_i$, which equals 1 only when $t_i$ lies
between $t_k$ and $t_{k+1}$ linearly in time (not in general for log-linear).

**Coincident nodes.** If all child node times appear among the parent node times
(common when curves share the same tenor grid), then $P_c$ reduces to a
row-selection (sub-)identity matrix: $(P_c)_{ij} = \delta_{t^{(c)}_i, t^{(\pi)}_j}$.

### 4.2 Beta coefficients and block structure of $L$

The matrix $L = \partial\mathbf{\zeta}/\partial\mathbf{\theta}$ has block entries that
depend on the spread model. We define two diagonal **beta matrices** for each curve $c$:

$$
(\Beta^{s}_c)_{ii} = \frac{\partial z^{(c)}_i}{\partial s^{(c)}_i}, \qquad
(\Beta^{\pi}_c)_{ii} = \frac{\partial z^{(c)}_i}{\partial \widetilde{z}^{(\pi)}_i}
$$

where $\widetilde{z}^{(\pi)}_i \equiv \widetilde{z}^{(\pi(c))}(t^{(c)}_i)$ is the
interpolated parent zero at node $i$.

| Spread model | $\Beta^{s}_c$ (spread beta) | $\Beta^{\pi}_c$ (propagation beta) |
|---|---|---|
| Additive: $z = z_\pi + s$ | $I$ | $I$ |
| Multiplicative: $z = z_\pi(1+s)$ | $\text{diag}(\widetilde{z}^{(\pi)}_i)$ | $\text{diag}(1 + s^{(c)}_i)$ |

The full partials are:

$$
\frac{\partial z^{(c)}_i}{\partial s^{(c)}_j} = (\Beta^{s}_c)_{ii} \, \delta_{ij}, \qquad
\frac{\partial z^{(c)}_i}{\partial z^{(\pi(c))}_j} = (\Beta^{\pi}_c)_{ii} \, (P_c)_{ij}
$$

Under topological ordering (parents before children), $L$ is **block
lower-triangular**. The diagonal blocks are $\Beta^{s}_c$ (identity for additive,
$\text{diag}(z_\pi)$ for multiplicative). For a chain $0 \to 1 \to 2$:

$$
L = \begin{pmatrix}
I & 0 & 0 \\
\Beta^{\pi}_1 P_1 & \Beta^{s}_1 & 0 \\
\Beta^{\pi}_2 P_2 \Beta^{\pi}_1 P_1 & \Beta^{\pi}_2 P_2 \Beta^{s}_1 & \Beta^{s}_2
\end{pmatrix}
$$

For the **additive** model this simplifies to the familiar form (all betas are $I$):

$$
L_{\text{add}} = \begin{pmatrix}
I & 0 & 0 \\
P_1 & I & 0 \\
P_2 P_1 & P_2 & I
\end{pmatrix}
$$

For a general tree, the $(d, a)$ block for descendant $d$ and ancestor $a$ along
path $a = c_0 \to c_1 \to \cdots \to c_m = d$ is:

$$
L_{d,a} = \Beta^{\pi}_{c_m} P_{c_m} \, \Beta^{\pi}_{c_{m-1}} P_{c_{m-1}} \cdots \Beta^{\pi}_{c_1} P_{c_1}
$$

with $\Beta^{s}_{c_0}$ on the right if $a \ne 0$. If $a$ is not an ancestor of $d$,
the block is zero.

### 4.3 Transpose action (what we actually need)

We never form $L$ explicitly. We need $L^\top \cdot \bar{\mathbf{\zeta}}$, which
is a **leaf-to-root accumulation**:

**Additive model:**

$$
\bar{\theta}^{(c)} = \bar{z}^{(c)} + \sum_{d:\,\pi(d)=c} P_d^\top \, \bar{\theta}^{(d)}
$$

**Multiplicative model:**

$$
\bar{\theta}^{(c)} = \Beta^{s}_c \, \bar{z}^{(c)}
  + \sum_{d:\,\pi(d)=c} P_d^\top \, \Beta^{\pi}_d \, \bar{\theta}^{(d)}
$$

Note that in the multiplicative case, the propagation beta $\Beta^{\pi}_d = \text{diag}(1 + s^{(d)}_i)$
scales the child's adjoint **before** projecting onto the parent's nodes via
$P_d^\top$. The spread beta $\Beta^{s}_c = \text{diag}(\widetilde{z}^{(\pi)}_i)$
scales the local adjoint.

Both are traversed in reverse topological order (leaves first). This is
$O(\sum_c n_c \cdot n_{\pi(c)})$ total — linear in the number of nonzeros of the
$P$ matrices, with only an elementwise multiply overhead for the multiplicative model.

---

## 5. Bootstrap Inversion: $B^{-1} = \partial\mathbf{\theta}/\partial\mathbf{r}$

### 5.1 Block-diagonal structure

Each curve $c$ is bootstrapped independently: holding parent curves fixed, the
spreads $\mathbf{s}^{(c)}$ are determined from quotes $\mathbf{r}^{(c)}$. Therefore
$\partial\mathbf{\theta}/\partial\mathbf{r}$ is **block-diagonal**:

$$
\frac{\partial\mathbf{\theta}}{\partial\mathbf{r}}
= \operatorname{diag}\bigl(
  (F^{(0)})^{-1},\; (F^{(1)})^{-1},\; \ldots,\; (F^{(C)})^{-1}
\bigr)
$$

where $F^{(c)}$ is the **forward Jacobian** of curve $c$:

$$
F^{(c)}_{ji} = \frac{\partial r^{(c)}_j}{\partial z^{(c)}_i}
$$

— the partial of instrument $j$'s par rate with respect to zero rate node $i$,
**with all other curves held fixed**.

Since during bootstrap $\partial z^{(c)}_i / \partial s^{(c)}_i = 1$ and parent is
frozen, we have $\partial r^{(c)}/\partial s^{(c)} = \partial r^{(c)}/\partial z^{(c)} = F^{(c)}$.

### 5.2 Triangular structure of $F^{(c)}$

Instruments are ordered by maturity (short to long). Instrument $j$ with maturity
$T_j$ depends only on zero rates at nodes $t_i \le T_j$. Therefore $F^{(c)}$ is
**lower-triangular** (including the diagonal), and the system
$(F^{(c)})^\top \mathbf{x} = \mathbf{g}$ can be solved by back-substitution in
$O(n_c^2)$ without any matrix inversion.

In practice $F^{(c)}$ is also **banded**: FRAs touch 2 nodes, swaps touch all nodes
up to maturity. A swap at the $k$-th maturity has $k$ nonzero partials per row. The
overall cost of the triangular solve is $O(n_c^2 / 2)$.

### 5.3 Derivation of $F^{(c)}$ for standard instruments

We now derive the entries of $F^{(c)}$ for each instrument type. Throughout this
section we suppress the curve superscript $(c)$ for readability. All discount factors,
zero rates, and times refer to curve $c$.

**Notation.** $D_i = e^{-z_i t_i}$ is the discount factor at node $i$.
$\tau_{ij} = \tau(t_i, t_j)$ is the year fraction (day count) between nodes $i$ and
$j$.

#### 5.3.1 Forward Rate Agreement (FRA)

A FRA between nodes $i_1$ and $i_2$ ($i_1 < i_2$) pays the difference between the
realised rate and the contracted rate over $[t_{i_1}, t_{i_2}]$. Its par rate is the
implied forward:

$$
f_{i_1, i_2} = \frac{1}{\tau_{i_1 i_2}} \left(\frac{D_{i_1}}{D_{i_2}} - 1\right)
$$

Define $R = D_{i_1} / D_{i_2} = \exp(z_{i_2} t_{i_2} - z_{i_1} t_{i_1})$.
Then $f = (R - 1)/\tau$.

$$
\frac{\partial f}{\partial z_{i_1}}
= \frac{1}{\tau_{i_1 i_2}} \cdot R \cdot (-t_{i_1})
= -\frac{t_{i_1}}{\tau_{i_1 i_2}} \,(1 + \tau_{i_1 i_2} \, f_{i_1 i_2})
$$

$$
\frac{\partial f}{\partial z_{i_2}}
= \frac{1}{\tau_{i_1 i_2}} \cdot R \cdot t_{i_2}
= \frac{t_{i_2}}{\tau_{i_1 i_2}} \,(1 + \tau_{i_1 i_2} \, f_{i_1 i_2})
$$

$$
\frac{\partial f}{\partial z_k} = 0 \quad \text{for } k \ne i_1, i_2
$$

**Each FRA contributes exactly 2 nonzero entries to its row of $F$.**

#### 5.3.2 Futures (SOFR futures, no convexity adjustment)

Under the assumption that futures rates equal forward rates (no convexity adjustment),
the partials are identical to the FRA case above.

The quoted price is $P = 100 - 100 \cdot f$, so if $F$ is expressed in rate terms,
the futures row of $F$ has the same entries as the FRA. If expressed in price terms,
negate and scale by 100.

#### 5.3.3 Par swap rate (OIS or fixed-vs-float)

A spot-starting swap with fixed leg payments at nodes $i_1, i_2, \ldots, i_m$ (where
$i_m$ corresponds to the maturity $T$). The par swap rate is:

$$
S = \frac{D_0 - D_{i_m}}{A}, \qquad
A = \sum_{k=1}^{m} \tau_k \, D_{i_k}
$$

where $D_0 = 1$ for a spot-starting swap, $\tau_k = \tau(t_{i_{k-1}}, t_{i_k})$ are
fixed leg year fractions, and $A$ is the annuity (PV01 of the fixed leg).

Differentiating with the quotient rule:

$$
\frac{\partial S}{\partial z_{i_k}}
= \frac{1}{A}\left[-\frac{\partial D_{i_m}}{\partial z_{i_k}} \cdot \mathbb{1}_{k=m} - S \,\frac{\partial A}{\partial z_{i_k}}\right]
$$

The annuity partial is:

$$
\frac{\partial A}{\partial z_{i_k}} = \tau_k \cdot (-t_{i_k}) \cdot D_{i_k}
= -\tau_k \, t_{i_k} \, D_{i_k}
$$

**Interior coupon node** ($k < m$):

$$
\frac{\partial S}{\partial z_{i_k}}
= \frac{S \, \tau_k \, t_{i_k} \, D_{i_k}}{A}
$$

**Final node** ($k = m$), using $\partial D_{i_m}/\partial z_{i_m} = -t_{i_m} D_{i_m}$:

$$
\frac{\partial S}{\partial z_{i_m}}
= \frac{t_{i_m} \, D_{i_m}}{A}\,(1 + S \, \tau_m)
$$

**Node 0** (spot, if $D_0$ depends on the overnight rate): usually $D_0 = 1$ exactly
and $\partial S / \partial z_0 = 0$.

**Nodes not on the fixed leg schedule** ($z_j$ with $t_j \ne t_{i_k}$ for any $k$):
$\partial S / \partial z_j = 0$, assuming cash flows fall exactly on node times.
If a cash flow date falls between nodes and requires interpolation, additional terms
arise through the interpolation weights, but this is avoided by choosing node times
to include all instrument cash flow dates.

#### 5.3.4 Basis swap (spread instruments for child curves)

Consider a basis swap that exchanges:
- **Receive leg**: floating at curve $c$'s rate (the child)
- **Pay leg**: floating at curve $\pi(c)$'s rate plus a quoted spread $b$

Both legs are discounted on a common discounting curve $d$ (often OIS). The
par basis spread $b$ satisfies:

$$
\sum_{k} \tau_k \, D^{(d)}_{i_k} \, f^{(c)}_{k}
= \sum_{k} \tau_k \, D^{(d)}_{i_k} \, \bigl(f^{(\pi(c))}_{k} + b\bigr)
$$

where $f^{(c)}_k$ and $f^{(\pi(c))}_k$ are the forward rates over period $k$ on
curves $c$ and $\pi(c)$ respectively. Rearranging:

$$
b = \frac{\sum_k \tau_k \, D^{(d)}_{i_k} \bigl(f^{(c)}_k - f^{(\pi(c))}_k\bigr)}
         {\sum_k \tau_k \, D^{(d)}_{i_k}}
= \frac{\sum_k \tau_k \, D^{(d)}_{i_k} \, f^{(c)}_k - \sum_k \tau_k \, D^{(d)}_{i_k} \, f^{(\pi)}_k}
       {A_d}
$$

During bootstrap of curve $c$, the parent curve and the discounting curve are **held
fixed**. Only $f^{(c)}_k$ varies. Substituting $f^{(c)}_k = (D^{(c)}_{k-1}/D^{(c)}_k - 1)/\tau_k$:

$$
\sum_k \tau_k \, D^{(d)}_{i_k} \, f^{(c)}_k
= \sum_k D^{(d)}_{i_k} \left(\frac{D^{(c)}_{k-1}}{D^{(c)}_k} - 1\right)
$$

The derivative with respect to $z^{(c)}_j$ (affecting $D^{(c)}_j$):

$$
\frac{\partial b}{\partial z^{(c)}_j}
= \frac{1}{A_d} \sum_{k} D^{(d)}_{i_k}
  \frac{\partial}{\partial z^{(c)}_j}\!\left(\frac{D^{(c)}_{k-1}}{D^{(c)}_k}\right)
$$

Node $j$ appears in at most two consecutive periods: as the end of period $j$ and
the start of period $j+1$. Using $\partial D^{(c)}_j / \partial z^{(c)}_j = -t_j D^{(c)}_j$:

$$
\frac{\partial}{\partial z^{(c)}_j}\!\left(\frac{D^{(c)}_{j-1}}{D^{(c)}_j}\right)
= \frac{t_j \, D^{(c)}_{j-1}}{D^{(c)}_j}
= t_j \,(1 + \tau_j f^{(c)}_j)
$$

$$
\frac{\partial}{\partial z^{(c)}_j}\!\left(\frac{D^{(c)}_j}{D^{(c)}_{j+1}}\right)
= \frac{-t_j \, D^{(c)}_j}{D^{(c)}_{j+1}}
= -t_j \,(1 + \tau_{j+1} f^{(c)}_{j+1})
$$

Combining:

$$
\frac{\partial b}{\partial z^{(c)}_j}
= \frac{t_j}{A_d}\Bigl[D^{(d)}_j \,(1 + \tau_j f^{(c)}_j)- D^{(d)}_{j+1}\,(1 + \tau_{j+1} f^{(c)}_{j+1})
\Bigr]
$$

with boundary adjustments at $j = 0$ (first period only) and $j = m$ (last period
only).

**Each basis swap row of $F^{(c)}$ has at most $m+1$ nonzero entries** (one per
floating period boundary), making $F^{(c)}$ moderately dense but structured.

#### 5.3.5 Handling interpolation at non-node cash flow dates

If an instrument's cash flow at time $t$ does not coincide with a node, the discount
factor $D(t)$ depends on multiple node values through interpolation:

$$
\frac{\partial D(t)}{\partial z_j}
= \frac{\partial D(t)}{\partial \widetilde{z}(t)} \cdot \frac{\partial \widetilde{z}(t)}{\partial z_j}
= -t \, D(t) \cdot w_j(t)
$$

where $w_j(t)$ is the interpolation weight of node $j$ at time $t$ (exactly the
entries of $P_c$ from Section 4, applied within the same curve). All formulas above
generalise by replacing $-t_i D_i \, \delta_{ij}$ with $-t \, D(t) \, w_j(t)$.

In practice, this is avoided by ensuring that node times include all instrument cash
flow dates. This is the standard approach in curve construction.

---

## 6. Full Transformation

### 6.1 Combined Jacobian

The full Jacobian from instruments to zeros is:

$$
\frac{\partial\mathbf{\zeta}}{\partial\mathbf{r}}
= L \cdot \operatorname{diag}\bigl((F^{(c)})^{-1}\bigr)
$$

and the risk transformation is:

$$
\frac{\partial V}{\partial\mathbf{r}}
= \operatorname{diag}\bigl((F^{(c)})^{-\top}\bigr) \cdot L^\top
  \cdot \frac{\partial V}{\partial\mathbf{\zeta}}
$$

### 6.2 Algorithm — Case A ($\mathbf{\theta}$ on tape)

When the engine takes base zeros $z^{(0)}$ and spreads $s^{(c)}$ as AD inputs, Stan
already gives $\partial V / \partial \mathbf{\theta}$ directly. Only one pass is
needed:

```
Input:  ḡ^(0) = ∂V/∂z^(0)        (from Stan, includes descendant contributions)
        ḡ^(c) = ∂V/∂s^(c)        (from Stan, for each spread curve c > 0)

For each curve c:
    Solve (F^(c))^T · x^(c) = ḡ^(c)  by back-substitution
    ∂V/∂r^(c) = x^(c)
```

This is the simplest path. The $L$ propagation is free — it happened on the AD tape.

### 6.3 Algorithm — Case B ($\mathbf{\zeta}$ on tape)

When the engine takes fully resolved zeros for every curve as independent AD inputs,
the spread structure must be recovered manually. Two passes:

**Pass 1 — Propagate adjoints through the curve tree** (compute $L^\top \bar{\mathbf{\zeta}}$):

The propagation depends on the spread model. In both cases, traverse in reverse
topological order (leaves → root):

*Additive model:*

```
Input:  ḡ^(c) = ∂V/∂z^(c)  for all c  (from Stan reverse-mode AD)

For c in reverse topological order (leaves → root):
    ḡ^(π(c)) += P_c^T · ḡ^(c)
```

*Multiplicative model:*

```
Input:  ḡ^(c) = ∂V/∂z^(c)  for all c  (from Stan reverse-mode AD)

For c in reverse topological order (leaves → root):
    ḡ^(π(c)) += P_c^T · diag(1 + s^(c)) · ḡ^(c)    # propagation beta
    ḡ^(c)    *= diag(z_parent_interp^(c))             # spread beta
```

where `z_parent_interp^(c)_i` $= \widetilde{z}^{(\pi(c))}(t^{(c)}_i)$ is the
interpolated parent zero at node $i$ of curve $c$.

After this pass, $\bar{g}^{(c)} = \partial V / \partial \theta^{(c)}$:
sensitivity to the independent bootstrap parameters.

**Pass 2 — Solve the triangular systems** (identical to Case A):

```
For each curve c:
    Solve (F^(c))^T · x^(c) = ḡ^(c)  by back-substitution
    ∂V/∂r^(c) = x^(c)
```

$F^{(c)}$ is lower-triangular (instruments ordered short → long, each instrument
depends only on nodes up to its maturity), so $(F^{(c)})^\top$ is upper-triangular
and the solve is straightforward back-substitution.

### 6.4 Complexity

| Operation | Case A | Case B |
|-----------|--------|--------|
| AD backward pass (Stan) | $O(N)$ | $O(N)$ |
| Pass 1: tree propagation | — (on tape) | $O(\sum_c \text{nnz}(P_c))$ |
| Pass 2: triangular solves | $O(\sum_c n_c^2 / 2)$ | $O(\sum_c n_c^2 / 2)$ |
| **Total post-AD** | **$O(\sum_c n_c^2)$** | **$O(\sum_c \text{nnz}(P_c) + \sum_c n_c^2)$** |

For 6 curves with 60 nodes each: $N = 360$, total $\approx 6 \times 1800 = 10{,}800$
flops. Entirely negligible compared to the pricing itself. Case A is cheaper because
the tree propagation is absorbed into the AD tape at no additional cost — the
reconstruction of zeros from spreads is just a few additions per node, which Stan
handles as part of the normal reverse sweep.

---

## 7. Constructing $F^{(c)}$ in Practice

### 7.1 Analytic assembly

Given the curve's zero rates, discount factors, forward rates, and annuity values
(all of which are available after bootstrapping), the entries of $F^{(c)}$ are computed
directly from the formulas in Section 5.3. No finite differences or AD required.

The assembly loop for a curve bootstrapped from $n_{\text{fra}}$ FRAs followed by
$n_{\text{swap}}$ par swaps:

```
F = zeros(n, n)   # n = n_fra + n_swap

# FRA rows
for j = 1, ..., n_fra:
    i1, i2 = start_node[j], end_node[j]
    R = D[i1] / D[i2]
    F[j, i1] = -t[i1] * R / tau[j]
    F[j, i2] = +t[i2] * R / tau[j]

# Par swap rows
for j = 1, ..., n_swap:
    row = n_fra + j
    nodes = coupon_nodes[j]   # all fixed leg payment nodes
    A = sum(tau[k] * D[nodes[k]] for k in 1..m)
    S = (1 - D[nodes[m]]) / A
    for k = 1, ..., m-1:     # interior coupon nodes
        F[row, nodes[k]] = S * tau[k] * t[nodes[k]] * D[nodes[k]] / A
    # final node
    k = m
    F[row, nodes[k]] = t[nodes[k]] * D[nodes[k]] * (1 + S * tau[k]) / A
```

### 7.2 AD-based assembly (alternative)

If the bootstrap itself is implemented with Stan AD, one can obtain $F^{(c)}$ by
differentiating the instrument rate functions with respect to the zero rates. For
each instrument $j$, evaluate $r_j(\mathbf{z})$ as a `stan::math::var` expression
and read off the adjoints. This is $O(n)$ per instrument, $O(n^2)$ total, with the
advantage of being agnostic to instrument type.

However, since we only need $(F^{(c)})^{-\top} \cdot \bar{g}$, not $F^{(c)}$ itself,
there is a more efficient approach (Section 8).

---

## 8. Avoiding Explicit $F$ Construction via AD

### 8.1 Implicit triangular solve

Instead of forming $F^{(c)}$ and solving $(F^{(c)})^\top \mathbf{x} = \bar{\mathbf{g}}$,
we can solve the system implicitly by exploiting the sequential (triangular) structure
of the bootstrap.

The bootstrap processes instruments in order $j = 1, 2, \ldots, n_c$. Each step
determines one spread/zero node from one instrument quote, given previously determined
nodes. This sequential dependence means:

$$
z_j = z_j(r_1, r_2, \ldots, r_j)
$$

The Jacobian $\partial\mathbf{z}/\partial\mathbf{r}$ (inverse of $F$) is also
lower-triangular. Its diagonal entries are:

$$
\left(\frac{\partial z_j}{\partial r_j}\right)_{\!r_1,\ldots,r_{j-1}\text{ fixed}}
= \frac{1}{F_{jj}}
$$

and the off-diagonal entries encode how bumping an earlier instrument $r_k$ ($k < j$)
propagates through all subsequent bootstrap steps.

### 8.2 Sequential adjoint bootstrap

We can compute $(F^{(c)})^{-\top} \bar{\mathbf{g}}$ without forming any matrix, by
running the bootstrap in reverse:

```
# Forward pass: bootstrap z_1, z_2, ..., z_n from r_1, ..., r_n
# (already done during curve construction)

# Reverse pass: adjoint propagation
x = zeros(n)       # will hold ∂V/∂r
ḡ = copy(ḡ_input)  # ∂V/∂θ^(c) from Pass 1

for j = n, n-1, ..., 1:
    # ∂V/∂r_j = ḡ_j · ∂z_j/∂r_j = ḡ_j / F_jj
    x[j] = ḡ[j] / F_jj

    # Propagate: bumping r_j changes z_j, which affects
    # the bootstrap of z_{j+1}, ..., z_n through the curve state.
    # The effect on ḡ[k] for k < j is:
    for k = 1, ..., j-1:
        ḡ[k] -= F[j,k] * x[j]
```

This is mathematically equivalent to the triangular solve but can be implemented
without storing $F$ if each bootstrap step is AD-enabled: the inner loop
$\bar{g}_k \mathrel{-}= F_{jk} \, x_j$ is exactly the adjoint of computing $r_j$
from $z_1, \ldots, z_j$.

### 8.3 Hybrid approach (recommended)

In practice, the cleanest implementation is:

1. **Form $F^{(c)}$ analytically** using the closed-form partials from Section 5.3.
   These are simple and cheap.
2. **Solve by back-substitution.** $F^{(c)}$ is small ($\le 60 \times 60$) and
   triangular — the solve is sub-microsecond.

Reserve the implicit AD approach (Section 8.2) for exotic instrument types where
deriving $\partial r_j / \partial z_i$ analytically is impractical.

---

## 9. Second-Order Risk (Gamma in Instrument Space)

### 9.1 Hessian transformation

If the engine also provides the Hessian $H_\zeta = \partial^2 V / \partial\mathbf{\zeta}^2$
(via `stan::math::hessian` using `fvar<var>`), the instrument-space Hessian is:

$$
H_r = J^\top H_\zeta \, J + \sum_i \bar{\zeta}_i \, \frac{\partial^2 \zeta_i}{\partial\mathbf{r}^2}
$$

where $J = \partial\mathbf{\zeta}/\partial\mathbf{r} = L \cdot \text{diag}((F^{(c)})^{-1})$.

The second term involves the **curvature of the bootstrap map** — how the zero rates
depend nonlinearly on the instrument quotes. For most instruments (FRAs, swaps) this
curvature is small, and the approximation $H_r \approx J^\top H_\zeta J$ is standard.

### 9.2 Efficient computation

We do not form $J$ or $H_\zeta$ as dense matrices. Instead:

1. Compute $H_\zeta \cdot \mathbf{e}_k$ for each instrument direction $\mathbf{e}_k$
   (mapped to zero-rate space via $J$) using `stan::math::hessian` or directional
   second derivatives.
2. Apply $J^\top$ to each column of the result.

For a portfolio-level gamma report with $M$ key instrument directions, this is
$O(M \times \text{cost of one Hessian-vector product})$.

---

## 10. Verification

### 10.1 Round-trip test (first order)

For any instrument bump $\delta\mathbf{r}$, the following must agree to machine
precision (up to interpolation and bootstrap solver tolerance):

$$
\left(\frac{\partial V}{\partial\mathbf{r}}\right)^{\!\top} \delta\mathbf{r}
\;\stackrel{?}{=}\;
\left(\frac{\partial V}{\partial\mathbf{\zeta}}\right)^{\!\top} \delta\mathbf{\zeta}
$$

where $\delta\mathbf{\zeta}$ is obtained by re-bootstrapping from
$\mathbf{r} + \delta\mathbf{r}$ and differencing the resulting zero rates.

### 10.2 Finite-difference validation

Bump each instrument quote $r^{(c)}_j$ by $\epsilon$ (e.g. 1bp), re-bootstrap the
full curve hierarchy, reprice $V$, and compare:

$$
\frac{\partial V}{\partial r^{(c)}_j}
\approx \frac{V(\mathbf{r} + \epsilon \mathbf{e}_j) - V(\mathbf{r} - \epsilon \mathbf{e}_j)}{2\epsilon}
$$

This is the definitive test. Disagreement indicates an error in either $P_c$
(interpolation weights), $F^{(c)}$ (instrument partials), or the tree propagation.

### 10.3 Partition of unity

The sum of all instrument DV01s (dollar value of 1bp parallel shift in all quotes
of a given curve) must equal the curve's total DV01 from a parallel zero rate shift:

$$
\sum_j \frac{\partial V}{\partial r^{(c)}_j}
= \sum_i \frac{\partial V}{\partial \theta^{(c)}_i} \cdot
  \sum_j \bigl((F^{(c)})^{-1}\bigr)_{ij}
$$

This is a weaker but quick sanity check.

---

## 11. Summary of Notation

| Symbol | Meaning |
|--------|---------|
| $c$ | Curve index, $c = 0$ is base (SOFR) |
| $\pi(c)$ | Parent curve of $c$ |
| $n_c$ | Number of zero rate nodes on curve $c$ |
| $z^{(c)}_i$ | Zero rate at node $i$ of curve $c$ (reconstructed for $c > 0$) |
| $s^{(c)}_i$ | Spread at node $i$ of curve $c$ (for $c > 0$; stored directly) |
| $\mathbf{\theta}$ | Independent parameters: $(z^{(0)}, s^{(1)}, \ldots, s^{(C)})$ |
| $\mathbf{\zeta}$ | Fully resolved zeros: $(z^{(0)}, z^{(1)}, \ldots, z^{(C)})$ |
| $D^{(c)}_i$ | Discount factor $= e^{-z^{(c)}_i t^{(c)}_i}$ |
| $P_c$ | Interpolation weight matrix, $n_c \times n_{\pi(c)}$ (Case B only) |
| $F^{(c)}$ | Forward Jacobian of curve $c$'s instruments w.r.t. its zeros |
| $L$ | Block lower-triangular spread propagation matrix (Case B only) |
| $\bar{z}^{(c)}$ | AD adjoint of zero rates (Case B: $\partial V / \partial z^{(c)}$) |
| $\bar{s}^{(c)}$ | AD adjoint of spreads (Case A: $\partial V / \partial s^{(c)}$) |
| $\bar{g}^{(c)}$ | Sensitivity to independent parameter: $\partial V / \partial \theta^{(c)}$ |
| $\partial V / \partial r^{(c)}$ | Final result: instrument sensitivities |

### Case selection guide

| What the engine stores as `var` | AD gives | Tree propagation | Section |
|------|----------|------------------|---------|
| Base zeros + spreads ($\mathbf{\theta}$) | $\partial V/\partial \mathbf{\theta}$ directly | Automatic (on tape) | 6.2 |
| All zeros ($\mathbf{\zeta}$) | $\partial V/\partial \mathbf{\zeta}$ (independent) | Manual (Pass 1) | 6.3 |
