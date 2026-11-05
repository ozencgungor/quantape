#ifndef LIGHTAPE_OPS_P2_FUNCTIONS_H
#define LIGHTAPE_OPS_P2_FUNCTIONS_H

/**
 * @file P2Functions.h
 * @brief P2 completeness functions: fdim, fmod/remainder, lambertW0 and
 *        Owen's T for `double`, `RevScalar` and `Tangent`
 *
 *   fdim(a,b)        = max(a - b, 0)                  branchy composite
 *   fmod(a,b)        = a - trunc(a/b) b               External (vhp = 0)
 *   remainder(a,b)   = a - nearbyint(a/b) b           External (vhp = 0)
 *   lambertW0(x)     = w with w e^w = x, w >= -1      safeguarded Newton
 *   owensT(h,a)      = (1/2pi) int_0^a e^{-h^2(1+x^2)/2}/(1+x^2) dx
 *
 * AD shape (matches the tape's replay contract):
 * - `fdim` is an ordinary composite: the primal picks the branch and the
 *   winner `a - b` (one Sub node) is shared, so reverse mode sees only that
 *   node and forward mode differentiates the linear expression exactly.
 *   `a - b <= 0` (including NaN) returns a node-less constant zero.
 * - `fmod`/`remainder` are a.e. linear in `(a, b)` — `n = trunc(a/b)` resp.
 *   `n = nearbyint(a/b)` is locally constant — so one `External` block with
 *   `vjp = (1, -n)`, `svp = s_a - n s_b` and a zero `vhp` is exact at every
 *   non-breakpoint; `hessianVector` replays it (zeros). The breakpoints are
 *   the multiples where `a/b` crosses an integer (resp. half-integer); at a
 *   breakpoint the derivative is the branch value chosen by the primal `n`
 *   (the usual a.e. contract, same as libm's sign-of-zero pick).
 * - `lambertW0` records one `External` block with the closed-form first and
 *   second derivatives at the solved point `L = W(x)`:
 *     f'  = L / (x (1 + L)) = e^{-L} / (1 + L),
 *     f'' = -L^2 (L + 2) / (x^2 (1 + L)^3) = -e^{-2L} (L + 2) / (1 + L)^3,
 *   evaluated in the e^{-L} form so `x = 0` (`L = 0`) gives `f'' = -2`
 *   without a 0/0. `vhp` is exact.
 * - `owensT` is clean-room from the defining integral (no Boost/Stan code):
 *   the primal is a fixed 32-point Gauss-Legendre rule on `[0, a]` (nodes and
 *   weights generated with mpmath to 20 digits and embedded below), and the
 *   analytic derivatives are
 *     dT/da = e^{-h^2(1+a^2)/2} / (2 pi (1 + a^2)),
 *     dT/dh = -e^{-h^2/2} erf(h a / sqrt(2)) / (2 sqrt(2 pi)),
 *     d2T/dh2 = -h dT/dh - a e^{-h^2(1+a^2)/2} / (2 pi),
 *     d2T/dhda = -h e^{-h^2(1+a^2)/2} / (2 pi),
 *     d2T/da2 = -(a/(1+a^2)) (dT/da) (h^2 (1+a^2) + 2),
 *   so the `External` block carries exact `vjp`/`svp`/`vhp` and
 *   `hessianVector` replays it. Measured error over 1681 points on
 *   `|h|, |a| <= 6` is < 3e-16 absolute (target 1e-12), degrading gracefully
 *   towards the tail; the `a < 0` antisymmetry `T(h,-a) = -T(h,a)` and the
 *   `h < 0` parity `T(-h,a) = T(h,a)` are built in (the integral depends on
 *   `h^2` only).
 *
 * The `Tangent` overloads follow the P1/Bessel pattern: the first-derivative
 * factors are evaluated at the primal point and treated as locally constant
 * for nested `Tangent` levels (exact first order, zero second order through
 * the factors — use the `RevScalar` block for the true Hessian).
 */

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>

#include "lightape/base/ScalarConstant.h"
#include "lightape/fwd/ForwardScalar.h"
#include "lightape/ops/Callbacks.h"
#include "lightape/ops/Probability.h"
#include "lightape/rev/LeanReverse.h"

namespace lightape {

/// e and 1/e (correctly rounded doubles) for `lambertW0`.
inline constexpr double kP2E = 2.71828182845904523536;
inline constexpr double kP2InvE = 0.36787944117144232160;

/// 1/(2 pi) and 1/(4 pi) (correctly rounded doubles) for `owensT`.
inline constexpr double kP2Inv2Pi = 0.15915494309189533577;
inline constexpr double kP2Inv4Pi = 0.07957747154594766788;

// ── fdim ──

/// `max(a - b, 0)` as a branchy composite: the primal comparison pins the
/// branch and the Sub node `a - b` is shared when positive, so reverse mode
/// records one node and both adjoints flow along it; the flat branch returns
/// a node-less zero. NaN (comparison false) maps to 0.
template <typename T>
T fdim(const T& a, const T& b) {
    const T d = a - b;
    return d > detail::constantOf<T>(0.0) ? d : detail::constantOf<T>(0.0);
}

// ── fmod / remainder (double) ──

/// `std::fmod(a,b)`: truncated quotient remainder; sign follows `a`.
inline double fmod(double a, double b) {
    return std::fmod(a, b);
}

/// `std::remainder(a,b)`: round-half-to-even quotient remainder; the result is
/// in `[-|b|/2, |b|/2]` with the sign chosen by `std::nearbyint`.
inline double remainder(double a, double b) {
    return std::remainder(a, b);
}

namespace detail {

/// `n = trunc(a/b)` resp. `nearbyint(a/b)` at the primal, locally constant.
struct P2QuotientPayload {
    double n = 0.0;
};

inline void p2QuotientVjp(const void* payload, double seed, const ExtArg* a, std::uint32_t /*n*/,
                          RevTape& tape) noexcept {
    const auto* p = static_cast<const P2QuotientPayload*>(payload);
    tape.addAdjoint(a[0].node, seed);
    tape.addAdjoint(a[1].node, -seed * p->n);
}

inline double p2QuotientSvp(const void* payload, const ExtArg* /*a*/, const double* s,
                            std::uint32_t /*n*/) noexcept {
    const auto* p = static_cast<const P2QuotientPayload*>(payload);
    return s[0] - p->n * s[1];
}

/// The quotient is locally constant, so the second derivative is zero a.e.
/// (kept as a real callback so the block still replays exactly).
inline void p2ZeroVhp(const void* /*payload*/, double /*adjoint*/, const ExtArg* /*a*/,
                      const double* /*s*/, std::uint32_t /*n*/, double* /*b*/) noexcept {}

/// Underlying double of a scalar (strips nested `Tangent` levels); the P2
/// `Tangent` overloads evaluate their derivative factors at this primal point.
template <typename T>
double p2PrimalValue(const T& x) {
    if constexpr (requires { x.value; }) {
        return p2PrimalValue(x.value);
    } else {
        return static_cast<double>(x);
    }
}

} // namespace detail

// ── fmod / remainder (RevScalar) ──

/// `fmod(a,b)`: one External block, first order exact a.e. with
/// `n = trunc(a/b)`, second order zero.
inline RevScalar fmod(const RevScalar& a, const RevScalar& b) {
    const double value = std::fmod(a.value, b.value);
    const detail::P2QuotientPayload payload{std::trunc(a.value / b.value)};
    const RevScalar args[2] = {a, b};
    return makeVhpFvar(value, args, 2, detail::p2QuotientVjp, detail::p2QuotientSvp,
                       detail::p2ZeroVhp, payload);
}

/// `remainder(a,b)`: mirror of `fmod` with `n = nearbyint(a/b)`
/// (round half to even, the same rule as `std::remainder`).
inline RevScalar remainder(const RevScalar& a, const RevScalar& b) {
    const double value = std::remainder(a.value, b.value);
    const detail::P2QuotientPayload payload{std::nearbyint(a.value / b.value)};
    const RevScalar args[2] = {a, b};
    return makeVhpFvar(value, args, 2, detail::p2QuotientVjp, detail::p2QuotientSvp,
                       detail::p2ZeroVhp, payload);
}

// ── fmod / remainder (Tangent) ──

/// `fmod(a,b)`: the quotient `n` is locally constant, so the tangent is
/// `d_a - n d_b` (exact first order; nested levels see `n` as a constant).
template <typename S, std::size_t N>
Tangent<S, N> fmod(const Tangent<S, N>& a, const Tangent<S, N>& b) {
    Tangent<S, N> out;
    out.value = fmod(a.value, b.value);
    const double n = std::trunc(detail::p2PrimalValue(a.value) / detail::p2PrimalValue(b.value));
    const S factor = detail::constantOf<S>(n);
    for (std::size_t i = 0; i < N; ++i) {
        out.d[i] = a.d[i] - factor * b.d[i];
    }
    return out;
}

/// `remainder(a,b)`: mirror of the `fmod` tangent overload with
/// `n = nearbyint(a/b)`.
template <typename S, std::size_t N>
Tangent<S, N> remainder(const Tangent<S, N>& a, const Tangent<S, N>& b) {
    Tangent<S, N> out;
    out.value = remainder(a.value, b.value);
    const double n =
        std::nearbyint(detail::p2PrimalValue(a.value) / detail::p2PrimalValue(b.value));
    const S factor = detail::constantOf<S>(n);
    for (std::size_t i = 0; i < N; ++i) {
        out.d[i] = a.d[i] - factor * b.d[i];
    }
    return out;
}

// ── lambertW0 ──

/// Principal branch of the Lambert W function: returns `w >= -1` with
/// `w e^w = x`, for `x >= -1/e`; below the branch point (or NaN) -> NaN,
/// `x = 0 -> 0` and `x = +inf -> +inf`.
///
/// Algorithm: a region seed (branch-point series near `-1/e`, the identity
/// `W(x) ~ x` near zero, the log-asymptotic `L1 - L2 + L2/L1` above `e`)
/// followed by at most 64 safeguarded Newton steps `w <- w - (w e^w - x) /
/// (e^w (1+w))` on the monotone bracket `[-1, 0]` (resp. `[0, x]`): a step
/// outside the bracket (or a non-positive derivative) is replaced by
/// bisection, and a step at rounding level is accepted so the sign flip of a
/// rounded residual cannot bisect away from the root.
///
/// Measured against mpmath at 50 dps on `[-1/e + 10^-14, 1e12]`: `|w - W| <=
/// 7e-15` for `x >= -1/e + 1e-8` (typically ~1e-16) and the round trip holds
/// at `|w e^w - x| <= 1e-14 max(1, |x|)`. Closer than ~1e-8 to the branch
/// point the error grows like `eps/(1 + W)` because `|dW/dx| -> inf` makes
/// the double evaluation of `w e^w - x` intrinsically ill-conditioned (at
/// `x = -1/e + 1e-14`, `|1 + W| ~ 2.5e-7` and the observed error is ~1.5e-10,
/// the double floor).
inline double lambertW0(double x) {
    if (std::isnan(x) || x < -kP2InvE) {
        return std::numeric_limits<double>::quiet_NaN();
    }
    if (x == -kP2InvE) {
        return -1.0;
    }
    if (x == 0.0) {
        return 0.0;
    }
    if (std::isinf(x)) {
        return x; // +inf (x < -1/e was rejected above)
    }
    double w = 0.0; // all branches below assign; initializer keeps the check quiet
    if (x < -0.25) {
        const double p = std::sqrt(2.0 * (kP2E * x + 1.0));
        w = -1.0 + p * (1.0 - p * (1.0 / 3.0 - p * (11.0 / 72.0)));
    } else if (x < 1.0) {
        w = x;
    } else {
        const double l1 = std::log(x);
        if (x <= kP2E) {
            w = 0.5 * x;
        } else {
            const double l2 = std::log(l1);
            w = l1 - l2 + l2 / l1;
        }
    }
    const double lo = x < 0.0 ? -1.0 : 0.0;
    const double hi = x < 0.0 ? 0.0 : x;
    if (!(w > lo && w < hi)) {
        w = 0.5 * (lo + hi);
    }
    double loB = lo;
    double hiB = hi;
    for (int i = 0; i < 64; ++i) {
        const double ew = std::exp(w);
        const double f = w * ew - x;
        if (f == 0.0) {
            break;
        }
        const double fp = ew * (1.0 + w);
        if (fp > 0.0) {
            const double wn = w - f / fp;
            // At rounding level the residual can flip sign between adjacent
            // doubles; accept the step instead of letting the safeguard
            // bisect away from the root.
            const double wTol =
                4.0 * std::numeric_limits<double>::epsilon() * std::fmax(1.0, std::fabs(w));
            if (std::fabs(wn - w) <= wTol) {
                w = wn;
                break;
            }
            if (wn > loB && wn < hiB) {
                if (f > 0.0) {
                    hiB = w;
                } else {
                    loB = w;
                }
                w = wn;
                continue;
            }
        }
        if (f > 0.0) {
            hiB = w;
        } else {
            loB = w;
        }
        const double mid = 0.5 * (loB + hiB);
        if (mid == w) {
            break;
        }
        w = mid;
    }
    return w;
}

namespace detail {

/// Closed-form first/second derivatives of `lambertW0` at the solved `w`.
struct P2LambertWPayload {
    double fp = 0.0;
    double fpp = 0.0;
};

inline void p2LambertWVjp(const void* payload, double seed, const ExtArg* a, std::uint32_t /*n*/,
                          RevTape& tape) noexcept {
    const auto* p = static_cast<const P2LambertWPayload*>(payload);
    tape.addAdjoint(a[0].node, seed * p->fp);
}

inline double p2LambertWSvp(const void* payload, const ExtArg* /*a*/, const double* s,
                            std::uint32_t /*n*/) noexcept {
    const auto* p = static_cast<const P2LambertWPayload*>(payload);
    return p->fp * s[0];
}

inline void p2LambertWVhp(const void* payload, double adjoint, const ExtArg* /*a*/, const double* s,
                          std::uint32_t /*n*/, double* b) noexcept {
    const auto* p = static_cast<const P2LambertWPayload*>(payload);
    b[0] += adjoint * p->fpp * s[0];
}

} // namespace detail

/// `lambertW0` as one `External` block: exact first and second order at the
/// solved point, with `f' = e^{-L}/(1+L)` and `f'' = -e^{-2L}(L+2)/(1+L)^3`
/// (finite at `x = 0`, where they evaluate to 1 and -2).
inline RevScalar lambertW0(const RevScalar& x) {
    const double l = lambertW0(x.value);
    const double den = 1.0 + l;
    const detail::P2LambertWPayload payload{std::exp(-l) / den,
                                            -std::exp(-2.0 * l) * (l + 2.0) / (den * den * den)};
    const RevScalar args[1] = {x};
    return makeVhpFvar(l, args, 1, detail::p2LambertWVjp, detail::p2LambertWSvp,
                       detail::p2LambertWVhp, payload);
}

/// `lambertW0`: the first-derivative factor `e^{-L}/(1+L)` at the primal,
/// treated as locally constant for nested `Tangent` levels.
template <typename S, std::size_t N>
Tangent<S, N> lambertW0(const Tangent<S, N>& x) {
    Tangent<S, N> out;
    out.value = lambertW0(x.value);
    const double l = lambertW0(detail::p2PrimalValue(x.value));
    const S scale = detail::constantOf<S>(std::exp(-l) / (1.0 + l));
    for (std::size_t i = 0; i < N; ++i) {
        out.d[i] = x.d[i] * scale;
    }
    return out;
}

// ── Owen's T ──

namespace detail {

/// 32-point Gauss-Legendre nodes on [-1, 1] (mpmath, 20 digits; the rule is
/// exact for polynomials up to degree 63).
inline constexpr double kOwensTNodes[32] = {
    -0.99726386184948156354,  -0.9856115115452683354,  -0.96476225558750643077,
    -0.93490607593773968917,  -0.89632115576605212397, -0.84936761373256997013,
    -0.79448379596794240696,  -0.73218211874028968039, -0.66304426693021520098,
    -0.58771575724076232904,  -0.50689990893222939002, -0.42135127613063534536,
    -0.33186860228212764978,  -0.23928736225213707454, -0.14447196158279649349,
    -0.048307665687738316235, 0.048307665687738316235, 0.14447196158279649349,
    0.23928736225213707454,   0.33186860228212764978,  0.42135127613063534536,
    0.50689990893222939002,   0.58771575724076232904,  0.66304426693021520098,
    0.73218211874028968039,   0.79448379596794240696,  0.84936761373256997013,
    0.89632115576605212397,   0.93490607593773968917,  0.96476225558750643077,
    0.9856115115452683354,    0.99726386184948156354};

/// Matching Gauss-Legendre weights (sum = 2 to 6e-61).
inline constexpr double kOwensTWeights[32] = {
    0.0070186100094700966004, 0.016274394730905670605, 0.025392065309262059456,
    0.034273862913021433103,  0.042835898022226680657, 0.050998059262376176196,
    0.058684093478535547145,  0.065822222776361846838, 0.072345794108848506225,
    0.078193895787070306472,  0.083311924226946755222, 0.087652093004403811143,
    0.091173878695763884713,  0.093844399080804565639, 0.095638720079274859419,
    0.096540088514727800567,  0.096540088514727800567, 0.095638720079274859419,
    0.093844399080804565639,  0.091173878695763884713, 0.087652093004403811143,
    0.083311924226946755222,  0.078193895787070306472, 0.072345794108848506225,
    0.065822222776361846838,  0.058684093478535547145, 0.050998059262376176196,
    0.042835898022226680657,  0.034273862913021433103, 0.025392065309262059456,
    0.016274394730905670605,  0.0070186100094700966004};

/// `(1/2pi) int_0^a ...` for `a >= 0`: map `[0, a]` onto `[-1, 1]` with
/// `x = a (1 + t)/2` and apply the fixed 32-point rule.
inline double owensTQuadrature(double h, double a) {
    const double h2 = h * h;
    double sum = 0.0;
    for (int i = 0; i < 32; ++i) {
        const double x = 0.5 * a * (1.0 + kOwensTNodes[i]);
        const double t = 1.0 + x * x;
        sum += kOwensTWeights[i] * std::exp(-0.5 * h2 * t) / t;
    }
    return kP2Inv4Pi * a * sum;
}

} // namespace detail

/// Owen's T: `T(h,a) = (1/2pi) int_0^a e^{-h^2(1+x^2)/2}/(1+x^2) dx`, clean-room
/// from the defining integral. `a < 0` uses `T(h,-a) = -T(h,a)`, NaN
/// propagates, `a = 0` gives 0 and `a = +/-inf` the closed form
/// `erfc(|h|/sqrt(2))/4`. See the file header for the quadrature accuracy.
inline double owensT(double h, double a) {
    if (std::isnan(h) || std::isnan(a)) {
        return std::numeric_limits<double>::quiet_NaN();
    }
    if (a == 0.0) {
        return 0.0;
    }
    if (std::isinf(a)) {
        return 0.25 * std::erfc(std::fabs(h) * kInvSqrt2);
    }
    if (a < 0.0) {
        return -owensT(h, -a);
    }
    return detail::owensTQuadrature(h, a);
}

namespace detail {

/// Primal value plus the exact gradient/Hessian of `T(h, a)` (see the file
/// header for the closed forms). Valid for signed `h` and `a`.
struct OwensTPack {
    double value = 0.0;
    double gh = 0.0;
    double ga = 0.0;
    double hhh = 0.0;
    double hha = 0.0;
    double haa = 0.0;
};

inline OwensTPack owensTPack(double h, double a) {
    OwensTPack p;
    p.value = owensT(h, a);
    const double h2 = h * h;
    const double a2 = a * a;
    const double e = std::exp(-0.5 * h2 * (1.0 + a2));
    const double twoPiTerm = e * kP2Inv2Pi;
    p.ga = twoPiTerm / (1.0 + a2);
    p.hha = -h * twoPiTerm;
    p.gh = -0.5 * kInvSqrt2Pi * std::exp(-0.5 * h2) * std::erf(h * a * kInvSqrt2);
    p.hhh = -h * p.gh - a * twoPiTerm;
    p.haa = -(a / (1.0 + a2)) * p.ga * (h2 * (1.0 + a2) + 2.0);
    return p;
}

// ── External callbacks (precomputed primal partials) ──

inline void owensTVjp(const void* payload, double seed, const ExtArg* a, std::uint32_t /*n*/,
                      RevTape& tape) noexcept {
    const auto* p = static_cast<const OwensTPack*>(payload);
    tape.addAdjoint(a[0].node, seed * p->gh);
    tape.addAdjoint(a[1].node, seed * p->ga);
}

inline double owensTSvp(const void* payload, const ExtArg* /*a*/, const double* s,
                        std::uint32_t /*n*/) noexcept {
    const auto* p = static_cast<const OwensTPack*>(payload);
    return p->gh * s[0] + p->ga * s[1];
}

inline void owensTVhp(const void* payload, double adjoint, const ExtArg* /*a*/, const double* s,
                      std::uint32_t /*n*/, double* b) noexcept {
    const auto* p = static_cast<const OwensTPack*>(payload);
    b[0] += adjoint * (p->hhh * s[0] + p->hha * s[1]);
    b[1] += adjoint * (p->hha * s[0] + p->haa * s[1]);
}

} // namespace detail

/// `owensT` as one binary `External` block with the exact gradient/Hessian of
/// the integral (closed-form second derivatives, so `hessianVector` replays
/// it exactly).
inline RevScalar owensT(const RevScalar& h, const RevScalar& a) {
    const detail::OwensTPack pack = detail::owensTPack(h.value, a.value);
    const RevScalar args[2] = {h, a};
    return makeVhpFvar(pack.value, args, 2, detail::owensTVjp, detail::owensTSvp, detail::owensTVhp,
                       pack);
}

/// `owensT`: first-derivative factors `(dT/dh, dT/da)` evaluated at the primal
/// point and treated as locally constant for nested `Tangent` levels.
template <typename S, std::size_t N>
Tangent<S, N> owensT(const Tangent<S, N>& h, const Tangent<S, N>& a) {
    Tangent<S, N> out;
    out.value = owensT(h.value, a.value);
    const detail::OwensTPack pack =
        detail::owensTPack(detail::p2PrimalValue(h.value), detail::p2PrimalValue(a.value));
    const S gh = detail::constantOf<S>(pack.gh);
    const S ga = detail::constantOf<S>(pack.ga);
    for (std::size_t i = 0; i < N; ++i) {
        out.d[i] = gh * h.d[i] + ga * a.d[i];
    }
    return out;
}

} // namespace lightape

#endif // LIGHTAPE_OPS_P2_FUNCTIONS_H
