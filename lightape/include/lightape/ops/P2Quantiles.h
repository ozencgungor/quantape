#ifndef LIGHTAPE_OPS_P2_QUANTILES_H
#define LIGHTAPE_OPS_P2_QUANTILES_H

/**
 * @file P2Quantiles.h
 * @brief P2 quantile functions: inverse regularized incomplete beta
 *        `invIncBeta` for `double`, `RevScalar` and `Tangent`
 *
 * Public API layer over the ported double kernels in
 * `lightape/base/{IncompleteBeta,Digamma}.h`:
 *
 *   double:            invIncBeta(a, b, p[, guess]),
 *   RevScalar:         one `External` block with the exact (a, b, p)
 *                      gradient/Hessian by implicit differentiation,
 *   Tangent<S,N>:      the same first-derivative factors applied tangent-wise.
 *
 * Primal algorithm (`detail::invIncBetaSolve`):
 * - symmetry: for `p > 1/2` the problem is solved in the mirrored orientation
 *   `I_{1-x}(b,a) = 1-p`, so the working tail is always the smaller one;
 * - seed: logit-scale normal approximation in the A&S 26.5.22 family. The
 *   exact logit moments `E[logit X] = psi(a)-psi(b)` and
 *   `Var[logit X] = psi'(a)+psi'(b)` (Digamma.h) replace the asymptotic
 *   coefficients: `u = psi(a)-psi(b) + z sqrt(psi'(a)+psi'(b))` with
 *   `z = Phi^{-1}(p)`, `x0 = 1/(1+e^{-u})`. A caller-supplied `guess` in
 *   `(0,1)` short-circuits the seed (mapped through the symmetry);
 * - iteration: safeguarded Halley on `I_x(a,b) - p` with the exact pdf and
 *   pdf' from `betaPdfD1D2` (`f' = f ((a-1)/x - (b-1)/(1-x))`). The bracket
 *   `[lo,hi]` is maintained by monotonicity of `I_x`; a Halley step is taken
 *   only strictly inside the bracket, otherwise the bracket midpoint is used.
 *   Each step is accepted at `|dx| <= 1e-15 |x|`, at most 100 steps are made.
 * - the solved point is clamped to `(0,1)`; `p = 0` / `p = 1` return
 *   `0` / `1`, `a <= 0`, `b <= 0` or `p` outside `[0,1]` return NaN.
 * - failure mode: assert-free. If the loop does not meet the step tolerance
 *   within 100 iterations it returns the bracket point with the smallest
 *   `|I_x(a,b) - p|` seen; the residual is then governed by the conditioning
 *   of `I_x` and may exceed 1e-15. The seed plus bracketed Halley converges
 *   for every strictly interior `(a,b,p)` in practice (all test triples,
 *   including `a,b < 1` and `p` near 0/1, reach the step tolerance).
 *
 * AD shape (matches the tape's replay contract):
 * - `invIncBeta` records one `External` block over `{a,b,p}`. Implicit
 *   differentiation of `F(x,a,b) = I_x(a,b) - p = 0` at the solved root with
 *   the exact 3-jet pack `betaGradHess(a,b,x)` gives
 *     x_a = -I_a/I_x, x_b = -I_b/I_x, x_p = 1/I_x,
 *     x_aa = -(I_aa + I_ax x_a)/I_x + I_a(I_ax + I_xx x_a)/I_x^2,
 *     x_ab = -(I_ab + I_ax x_b)/I_x + I_a(I_bx + I_xx x_b)/I_x^2,
 *     x_ap = -(I_ax x_p)/I_x + I_a I_xx x_p/I_x^2,
 *     x_bb = -(I_bb + I_bx x_b)/I_x + I_b(I_bx + I_xx x_b)/I_x^2,
 *     x_bp = -(I_bx x_p)/I_x + I_b I_xx x_p/I_x^2,
 *     x_pp = -I_xx/I_x^3.
 *   Every partial is available from the pack, so the block carries `vjp`,
 *   `svp` and `vhp` and `RevTape::hessianVector` replays it exactly; no
 *   entry is ever fabricated, and if a factor were unavailable the code
 *   path would keep `vhp = nullptr` (fail loud) rather than approximate.
 *   At `p = 0`/`1` the implicit division follows the one-sided pack values
 *   and the partials may be non-finite (documented edge; use for `p in (0,1)`).
 * - the `Tangent<S,N>` overload evaluates the same first-derivative factors
 *   at the primal point (`detail::primalValue` strips nesting), exact for
 *   first order; nested levels see the factors as locally constant (use the
 *   `RevScalar` block for the true Hessian).
 */

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>

#include "lightape/base/Digamma.h"
#include "lightape/base/IncompleteBeta.h"
#include "lightape/base/ScalarConstant.h"
#include "lightape/fwd/ForwardScalar.h"
#include "lightape/ops/Callbacks.h"
#include "lightape/ops/P1Functions.h"
#include "lightape/rev/LeanReverse.h"

namespace lightape {

/// `x` with `I_x(a,b) = p`; see the file header for the algorithm, the
/// `guess` contract and the failure mode.
inline double invIncBeta(double a, double b, double p, double guess = -1.0);

namespace detail {

/// Iteration cap for the beta quantile solver; reaching it is a graceful
/// failure (the best bracket point seen is returned), never an error.
inline constexpr int kInvIncBetaMaxIterations = 100;

/// Relative x-step tolerance of the safeguarded Halley iteration.
inline constexpr double kInvIncBetaRelTol = 1e-15;

/// Smallest seed accepted; keeps the seed strictly inside `(0,1)`.
inline constexpr double kInvIncBetaMinSeed = 1e-300;

/// Seed for `I_x(a,b) = p` with `p <= 1/2`: logit-scale normal approximation
/// in the A&S 26.5.22 family (exact logit moments, standard normal quantile).
inline double invIncBetaSeed(double a, double b, double p) {
    const double z = invPhi(p);
    const double u =
        digammaDouble(a) - digammaDouble(b) + z * std::sqrt(trigammaDouble(a) + trigammaDouble(b));
    const double x = 1.0 / (1.0 + std::exp(-u));
    if (!(x > 0.0 && x < 1.0)) {
        return 0.5;
    }
    return x;
}

/// Root approximation plus the number of range evaluations used (probe for
/// the `guess` contract; the public API discards it).
struct InvIncBetaResult {
    double x = 0.0;
    int iterations = 0;
};

/// Safeguarded Halley solve of `I_x(a,b) = p`, `a,b > 0`, `p in [0,1]`.
/// Assumes valid inputs (the public wrapper validates); the result is always
/// in `[0,1]` and monotonically bracketed.
inline InvIncBetaResult invIncBetaSolve(double a, double b, double p, double guess) {
    InvIncBetaResult out;
    // Mirror p > 1/2 onto the smaller tail: I_x(a,b) = p  <=>
    // I_{1-x}(b,a) = 1-p.
    const bool swapped = p > 0.5;
    const double aa = swapped ? b : a;
    const double bb = swapped ? a : b;
    const double pp = swapped ? 1.0 - p : p;

    double x = 0.0;
    if (guess > 0.0 && guess < 1.0) {
        x = swapped ? 1.0 - guess : guess;
    } else {
        x = invIncBetaSeed(aa, bb, pp);
    }
    if (!(x > 0.0 && x < 1.0)) {
        x = 0.5;
    }
    x = std::fmin(std::fmax(x, kInvIncBetaMinSeed), std::nextafter(1.0, 0.0));

    double lo = 0.0;
    double hi = 1.0;
    double bestX = x;
    double bestErr = std::numeric_limits<double>::infinity();
    for (int it = 0; it < kInvIncBetaMaxIterations; ++it) {
        out.iterations = it + 1;
        const double f = incBetaDouble(aa, bb, x);
        const double delta = f - pp;
        if (std::fabs(delta) < bestErr) {
            bestErr = std::fabs(delta);
            bestX = x;
        }
        if (!std::isfinite(delta)) {
            break;
        }
        if (delta == 0.0) {
            out.x = swapped ? 1.0 - x : x;
            return out;
        }
        if (delta < 0.0) {
            lo = x;
        } else {
            hi = x;
        }
        double next = 0.5 * (lo + hi);
        const BetaValueD1D2 pdf = betaPdfD1D2(aa, bb, x);
        if (std::isfinite(pdf.d1) && pdf.d1 > 0.0) {
            const double denom = 2.0 * pdf.d1 * pdf.d1 - delta * pdf.d2;
            if (std::isfinite(denom) && denom != 0.0) {
                const double candidate = x - 2.0 * delta * pdf.d1 / denom;
                if (candidate > lo && candidate < hi) {
                    next = candidate;
                }
            }
        }
        // A step at the rounding level of x is accepted so a sign flip of a
        // rounded residual cannot make the safeguard bisect away from it.
        if (std::fabs(next - x) <= kInvIncBetaRelTol * std::fabs(x)) {
            out.x = swapped ? 1.0 - next : next;
            return out;
        }
        x = next;
    }
    out.x = swapped ? 1.0 - bestX : bestX;
    return out;
}

} // namespace detail

/// `x` with `I_x(a,b) = p`, `a,b > 0`, `p in [0,1]`.
///
/// `guess`, when strictly inside `(0,1)`, is used as the starting point of
/// the safeguarded iteration instead of the built-in seed (the solve is
/// orientation-mirrored, so the guess is too). The default `-1.0` selects
/// the built-in seed. See the file header for the algorithm, the exactness
/// contract and the residual/failure behavior.
inline double invIncBeta(double a, double b, double p, double guess) {
    const double nan = std::numeric_limits<double>::quiet_NaN();
    if (!(a > 0.0) || !(b > 0.0) || std::isnan(p) || p < 0.0 || p > 1.0) {
        return nan;
    }
    if (p == 0.0) {
        return 0.0;
    }
    if (p == 1.0) {
        return 1.0;
    }
    return detail::invIncBetaSolve(a, b, p, guess).x;
}

namespace detail {

/// Value and exact `(a, b, p)` gradient/Hessian of `x = invIncBeta(a,b,p)` by
/// implicit differentiation of `I_x(a,b) - p = 0` (see the file header).
struct InvIncBetaFactors {
    double value = 0.0;
    double ga = 0.0;
    double gb = 0.0;
    double gp = 0.0;
    double haa = 0.0;
    double hab = 0.0;
    double hap = 0.0;
    double hbb = 0.0;
    double hbp = 0.0;
    double hpp = 0.0;
};

inline InvIncBetaFactors invIncBetaFactors(double a, double b, double p) {
    InvIncBetaFactors out;
    out.value = invIncBeta(a, b, p);
    const BetaGradHess pack = betaGradHess(a, b, out.value);
    const double invIx = 1.0 / pack.gX;
    const double invIx2 = invIx * invIx;
    const double xa = -pack.gA * invIx;
    const double xb = -pack.gB * invIx;
    const double xp = invIx;
    out.ga = xa;
    out.gb = xb;
    out.gp = xp;
    out.haa = -(pack.hAA + pack.hAX * xa) * invIx + pack.gA * (pack.hAX + pack.hXX * xa) * invIx2;
    out.hab = -(pack.hAB + pack.hAX * xb) * invIx + pack.gA * (pack.hBX + pack.hXX * xb) * invIx2;
    out.hap = -pack.hAX * xp * invIx + pack.gA * pack.hXX * xp * invIx2;
    out.hbb = -(pack.hBB + pack.hBX * xb) * invIx + pack.gB * (pack.hBX + pack.hXX * xb) * invIx2;
    out.hbp = -pack.hBX * xp * invIx + pack.gB * pack.hXX * xp * invIx2;
    out.hpp = -pack.hXX * xp * invIx2;
    return out;
}

// ── External callbacks (factors precomputed at the solved point) ──

inline void invIncBetaVjp(const void* payload, double seed, const ExtArg* a, std::uint32_t /*n*/,
                          RevTape& tape) noexcept {
    const auto* p = static_cast<const InvIncBetaFactors*>(payload);
    tape.addAdjoint(a[0].node, seed * p->ga);
    tape.addAdjoint(a[1].node, seed * p->gb);
    tape.addAdjoint(a[2].node, seed * p->gp);
}

inline double invIncBetaSvp(const void* payload, const ExtArg* /*a*/, const double* s,
                            std::uint32_t /*n*/) noexcept {
    const auto* p = static_cast<const InvIncBetaFactors*>(payload);
    return p->ga * s[0] + p->gb * s[1] + p->gp * s[2];
}

inline void invIncBetaVhp(const void* payload, double adjoint, const ExtArg* /*a*/, const double* s,
                          std::uint32_t /*n*/, double* b) noexcept {
    const auto* p = static_cast<const InvIncBetaFactors*>(payload);
    b[0] += adjoint * (p->haa * s[0] + p->hab * s[1] + p->hap * s[2]);
    b[1] += adjoint * (p->hab * s[0] + p->hbb * s[1] + p->hbp * s[2]);
    b[2] += adjoint * (p->hap * s[0] + p->hbp * s[1] + p->hpp * s[2]);
}

} // namespace detail

/// `invIncBeta` as one `External` block over `{a,b,p}`: exact first and
/// second order (implicit differentiation of the exact `betaGradHess` pack),
/// so `RevTape::hessianVector` replays it exactly.
inline RevScalar invIncBeta(const RevScalar& a, const RevScalar& b, const RevScalar& p) {
    const detail::InvIncBetaFactors pack = detail::invIncBetaFactors(a.value, b.value, p.value);
    const RevScalar args[3] = {a, b, p};
    return makeVhpFvar(pack.value, args, 3, detail::invIncBetaVjp, detail::invIncBetaSvp,
                       detail::invIncBetaVhp, pack);
}

/// `invIncBeta`: first-derivative factors evaluated at the primal point and
/// treated as locally constant for nested `Tangent` levels (exact first
/// order; use the `RevScalar` block for the true Hessian).
template <typename S, std::size_t N>
Tangent<S, N> invIncBeta(const Tangent<S, N>& a, const Tangent<S, N>& b, const Tangent<S, N>& p) {
    Tangent<S, N> out;
    out.value = invIncBeta(a.value, b.value, p.value);
    const detail::InvIncBetaFactors pack = detail::invIncBetaFactors(
        detail::primalValue(a.value), detail::primalValue(b.value), detail::primalValue(p.value));
    const S ga = detail::constantOf<S>(pack.ga);
    const S gb = detail::constantOf<S>(pack.gb);
    const S gp = detail::constantOf<S>(pack.gp);
    for (std::size_t i = 0; i < N; ++i) {
        out.d[i] = ga * a.d[i] + gb * b.d[i] + gp * p.d[i];
    }
    return out;
}

} // namespace lightape

#endif // LIGHTAPE_OPS_P2_QUANTILES_H
