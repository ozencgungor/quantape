#ifndef LIGHTAPE_OPS_ELEMENTARY_EXTRAS_H
#define LIGHTAPE_OPS_ELEMENTARY_EXTRAS_H

/**
 * @file ElementaryExtras.h
 * @brief P2 elementary extras: log1mexp / logDiffExp / multiplyLog and invErfc
 *
 * Scalar-generic composites for `double`, `RevScalar`, `Tangent<S, N>` and
 * nested `Tangent` (up to the constant-injection depth), plus a Newton
 * double kernel for the inverse complementary error function with a
 * first+second order `RevScalar` External wrapper:
 *
 *   log1mexp(x)      = log(1 - e^x)          (x < 0)
 *   logDiffExp(a, b) = log(e^a - e^b)        (a >= b)
 *   multiplyLog(a,b) = b log(a)              (0 at b = 0, gradient pinned)
 *   invErfc(y)       = x with erfc(x) = y    (y in (0, 2))
 *
 * AD shape: `log1mexp`, `logDiffExp` and `multiplyLog` are ordinary composite
 * expressions, so reverse mode reuses the existing exp/log/log1p opcodes and
 * the tape's exact Hessian replay works without dedicated opcodes. `invErfc`
 * cannot be composed from elementary ops (it is the inverse of erfc), so its
 * `RevScalar` overload records one `External` block carrying the solved point
 * `x`, the first derivative `f' = -sqrt(pi)/2 e^{x^2}` and the exact second
 * derivative `f'' = 2 x (f')^2`, which `RevTape::hessianVector` replays
 * exactly.
 */

#include <cmath>
#include <cstdint>
#include <limits>

#include "lightape/base/ScalarConstant.h"
#include "lightape/fwd/ForwardScalar.h"
#include "lightape/ops/Callbacks.h"
#include "lightape/rev/LeanReverse.h"

namespace lightape {

/// sqrt(pi), sqrt(pi)/2 and (ln pi)/2 (correctly rounded) for `invErfc`.
inline constexpr double kSqrtPi = 1.77245385090551602730;
inline constexpr double kHalfSqrtPi = 0.88622692545275801365;
inline constexpr double kHalfLnPi = 0.57236494292470008707;

/// `log(1 - e^x)` for `x < 0`, evaluated stably as `log1p(-exp(x))`.
/// For `x = 0` this is `-inf`, for `x > 0` NaN; use `logDiffExp` for the
/// two-argument difference form instead.
template <typename T>
T log1mexp(const T& x) {
    using std::exp;
    using std::log1p;
    return log1p(-exp(x));
}

/// `log(e^a - e^b)` for `a >= b`, evaluated as `a + log1p(-exp(b - a))` so the
/// exponential is taken on the non-positive difference. Domain: `a >= b`
/// (`a < b` yields NaN through `log1p` of a negative argument; `a = b` gives
/// `-inf`). Derivatives are the logistic weights of `d = b - a <= 0`:
///   `f_a = 1/(1 - e^d)`, `f_b = -e^d/(1 - e^d)`.
template <typename T>
T logDiffExp(const T& a, const T& b) {
    using std::exp;
    using std::log1p;
    return a + log1p(-exp(b - a));
}

/// `b log(a)` with the product pinned to exactly 0 at `b = 0` (primal-pinned
/// branch, so the 0 * log(a) corner carries zero gradient even for `a <= 0`
/// instead of a NaN). Away from `b = 0` it is the ordinary composite with
/// `f_a = b/a`, `f_b = log(a)`.
template <typename T>
T multiplyLog(const T& a, const T& b) {
    using std::log;
    if (b == detail::constantOf<T>(0.0)) {
        return detail::constantOf<T>(0.0);
    }
    return b * log(a);
}

namespace detail {

/// Precomputed first/second derivative of `erfc^{-1}` at the solved root.
struct InvErfcPayload {
    double fp = 0.0;
    double fpp = 0.0;
};

inline void invErfcVjp(const void* payload, double seed, const ExtArg* a, std::uint32_t /*n*/,
                       RevTape& tape) noexcept {
    const auto* p = static_cast<const InvErfcPayload*>(payload);
    tape.addAdjoint(a[0].node, seed * p->fp);
}

inline double invErfcSvp(const void* payload, const ExtArg* /*a*/, const double* s,
                         std::uint32_t /*n*/) noexcept {
    const auto* p = static_cast<const InvErfcPayload*>(payload);
    return p->fp * s[0];
}

inline void invErfcVhp(const void* payload, double adjoint, const ExtArg* /*a*/, const double* s,
                       std::uint32_t /*n*/, double* b) noexcept {
    const auto* p = static_cast<const InvErfcPayload*>(payload);
    b[0] += adjoint * p->fpp * s[0];
}

/// Seed for `invErfc` on `(0, 1]`. For `y >= 1/2` (i.e. `|t| <= 1/2` with
/// `t = y - 1`) a (5,5) Pade approximant of the odd function
/// `x = -erfinv(t) = t P(t^2)/Q(t^2)`, fit from the in-house series (max
/// error ~2e-13 over the interval, well inside Newton's basin). For
/// `y < 1/2` the one-term asymptotic inversion of `erfc`,
/// `x ~ s - log(sqrt(pi) s)/(2 s)` with `s = sqrt(-log(y))`, whose error
/// falls from ~0.12 at the boundary as `y` decreases. Both feeds are only
/// seeds: Newton polishes to double precision.
inline double invErfcSeed(double y) {
    if (y >= 0.5) {
        static constexpr double p[6] = {-0.88622692545275801365,  2.0404152179484740854,
                                        -1.630379318435826785,    0.52693327921778292165,
                                        -0.059245732835684122524, 0.0011071517904181480892};
        static constexpr double q[6] = {1.0,
                                        -2.5641614119568573832,
                                        2.3670502144056557635,
                                        -0.94287226104410587256,
                                        0.15012760213642570637,
                                        -0.0064398863661304438982};
        const double t = y - 1.0;
        const double z = t * t;
        double num = p[5];
        double den = q[5];
        for (int i = 4; i >= 0; --i) {
            num = num * z + p[i];
            den = den * z + q[i];
        }
        return t * num / den;
    }
    const double s = std::sqrt(-std::log(y));
    return s - (kHalfLnPi + std::log(s)) / (2.0 * s);
}

} // namespace detail

/// Inverse complementary error function: returns `x` with `erfc(x) = y`.
/// The seeded Newton iteration uses `erfc' = -2/sqrt(pi) e^{-x^2}`, i.e.
/// `x_{k+1} = x_k + (erfc(x_k) - y) (sqrt(pi)/2) e^{x_k^2}`, with at most 8
/// steps and an early exit once an update no longer changes `x` (sub-ulp);
/// typically 1-6 steps. Accuracy target: round-trip `erfc(invErfc(y)) = y`
/// at the 1e-14 relative level for `y >= 1e-12` (at the `std::erfc` rounding
/// limit), degrading gracefully towards the deep tail where `erfc` itself
/// underflows. Boundaries: `y = 0 -> +inf`, `y = 2 -> -inf`, outside `[0, 2]`
/// or NaN -> NaN.
inline double invErfc(double y) {
    if (std::isnan(y) || y < 0.0 || y > 2.0) {
        return std::numeric_limits<double>::quiet_NaN();
    }
    if (y == 0.0) {
        return std::numeric_limits<double>::infinity();
    }
    if (y == 2.0) {
        return -std::numeric_limits<double>::infinity();
    }
    if (y > 1.0) {
        return -invErfc(2.0 - y); // erfc(-x) = 2 - erfc(x)
    }
    double x = detail::invErfcSeed(y);
    for (int i = 0; i < 8; ++i) {
        const double f = std::erfc(x) - y;
        const double step = kHalfSqrtPi * f * std::exp(x * x);
        if (!std::isfinite(step)) {
            break;
        }
        const double next = x + step;
        if (next == x) {
            break;
        }
        x = next;
    }
    return x;
}

/// `invErfc` as one `External` block over the solved point: exact first and
/// second order via `makeVhpFvar`, so `RevTape::hessianVector` replays it.
/// With `f' = -sqrt(pi)/2 e^{x^2}` at the root, `f'' = 2 x (f')^2` (one more
/// chain rule through the inverse relation).
inline RevScalar invErfc(const RevScalar& y) {
    const double x = invErfc(y.value);
    const double fp = -kHalfSqrtPi * std::exp(x * x);
    const detail::InvErfcPayload payload{fp, 2.0 * x * fp * fp};
    const RevScalar args[1] = {y};
    return makeVhpFvar(x, args, 1, detail::invErfcVjp, detail::invErfcSvp, detail::invErfcVhp,
                       payload);
}

} // namespace lightape

#endif // LIGHTAPE_OPS_ELEMENTARY_EXTRAS_H
