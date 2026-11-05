#ifndef LIGHTAPE_OPS_GENERIC_H
#define LIGHTAPE_OPS_GENERIC_H

#include <cmath>
#include <type_traits>
#include <utility>

#include "lightape/base/ScalarConstant.h"

namespace lightape {
/**
 * @file Generic.h
 * @brief Scalar-generic composite ops for `double`, `RevScalar` and (nested) `Tangent`
 *
 * Every template below is written against the scalar interface only —
 * arithmetic operators, comparisons on the primal and ADL math — so the same
 * call compiles for `double`, `RevScalar`, `Tangent<S, N>` and nested
 * `Tangent<Tangent<...>, N>` (third order). Branches (`clamp`, the softplus
 * overflow guard, `fmax` in the log-sum-exp forms) are pinned by the primal,
 * the same a.e. contract as the engines, so the selected branch's node is
 * shared and adjoints flow along it.
 *
 * Literal injection goes through `detail::constantOf`: a plain double cannot
 * reach the scalar of a triple-nested `Tangent` in one implicit conversion
 * (chained user-defined conversions do not compose), so constants are built
 * one nesting level at a time through the `value` member. `RevScalar`
 * constants stay node-less (they never touch the tape); `Tangent` constants
 * carry zero tangents.
 */

/// Clamp to `[lo, hi]`, returning by value (no dangling reference for
/// temporaries). The saturated sides return `lo`/`hi` themselves, so the
/// adjoint flows to the bound (or is zero along the flat branch).
template <typename T>
T clamp(const T& x, const T& lo, const T& hi) {
    return x < lo ? lo : (hi < x ? hi : x);
}

/// Clamp to `[0, 1]`.
template <typename T>
T clamp01(const T& x) {
    return clamp(x, detail::constantOf<T>(0.0), detail::constantOf<T>(1.0));
}

/// Exact mathematical sign as a `T`: -1, 0, +1 (NaN yields 0). The result is
/// locally constant, so its own derivative is 0 a.e.; the useful composition
/// is `x * signum(x)` with derivative `signum(x)`. `unitSign` is the AD
/// companion that keeps a non-zero sign at the origin.
template <typename T>
T signum(const T& x) {
    const T zero = detail::constantOf<T>(0.0);
    const T one = detail::constantOf<T>(1.0);
    return x > zero ? one : (x < zero ? -one : zero);
}

/// ±1 with +1 at zero: the sign of `|x|` / of a clamp direction where
/// `signum` would kill the gradient.
template <typename T>
T unitSign(const T& x) {
    return x >= detail::constantOf<T>(0.0) ? detail::constantOf<T>(1.0)
                                           : detail::constantOf<T>(-1.0);
}

/// `x * x` without the intermediate temporary of `std::pow`.
template <typename T>
T sqr(const T& x) {
    return x * x;
}

/// `x * x * x`.
template <typename T>
T cube(const T& x) {
    return x * x * x;
}

/// Linear interpolation `a + t (b - a)`; `t` is a passive double fraction
/// (a call in the house `util::lerp` sense), so only `a` and `b` carry
/// derivatives.
template <typename T>
T lerp(const T& a, const T& b, double t) {
    return a + detail::constantOf<T>(t) * (b - a);
}

/// `a / b`, or 0 when `|b| <= eps`: the dead zone keeps the result finite and
/// zero-gradient instead of exploding (or branching) near a zero denominator.
template <typename T>
T safeDiv(const T& a, const T& b, double eps) {
    using std::fabs;
    return fabs(b) <= detail::constantOf<T>(eps) ? detail::constantOf<T>(0.0) : a / b;
}

/// Softplus `log(1 + exp(x))`, evaluated as `x` for `x > 30` so `exp` never
/// overflows. Derivatives, with `s = sigmoid(x)`:
///   f' = s,  f'' = s(1 - s),  f''' = s(1 - s)(1 - 2s).
template <typename T>
T softplus(const T& x) {
    using std::exp;
    using std::log1p;
    return x > detail::constantOf<T>(30.0) ? x : log1p(exp(x));
}

/// Logistic sigmoid `1 / (1 + exp(-x))`. Derivatives, with `s = sigmoid(x)`:
///   s' = s(1 - s),  s'' = s(1 - s)(1 - 2s),  s''' = s(1 - s)(1 - 6s + 6s^2).
template <typename T>
T sigmoid(const T& x) {
    using std::exp;
    const T one = detail::constantOf<T>(1.0);
    return one / (one + exp(-x));
}

/// Logit `log(p / (1 - p))`, the inverse of `sigmoid`. Derivatives, with
/// `u = p(1 - p)`:  f' = 1/u,  f'' = (2p - 1)/u^2,
/// f''' = (2(2p - 1)^2 + 2u)/u^3.
template <typename T>
T logit(const T& p) {
    using std::log;
    const T one = detail::constantOf<T>(1.0);
    return log(p / (one - p));
}

/// Smooth maximum via log-sum-exp, `m + log1p(exp(-|a - b| k))/k` with
/// `m = fmax(a, b)` and `k > 0` (large `k` tightens towards `max`).
/// Derivatives are the logistic weights, with `w = sigmoid(k (a - b))`:
///   da = w,  db = 1 - w,
///   d2a2 = d2b2 = k w(1 - w),  d2ab = -k w(1 - w),
///   d3a3 = k^2 w(1 - w)(1 - 2w)  (mixed signs alternate in a/b).
template <typename T>
T smoothMax(const T& a, const T& b, double k) {
    using std::exp;
    using std::fabs;
    using std::fmax;
    using std::log1p;
    const T kk = detail::constantOf<T>(k);
    const T m = fmax(a, b);
    return m + log1p(exp(-fabs(a - b) * kk)) / kk;
}

/// Two-argument log-sum-exp `log(exp(a) + exp(b))`, with the max shift for
/// overflow safety: `m + log1p(exp(-|a - b|))`, `m = fmax(a, b)`. This is
/// `smoothMax(a, b, 1)`; derivatives `w = sigmoid(a - b)` as above with k=1.
template <typename T>
T logSumExp2(const T& a, const T& b) {
    using std::exp;
    using std::fabs;
    using std::fmax;
    using std::log1p;
    const T m = fmax(a, b);
    return m + log1p(exp(-fabs(a - b)));
}

/// C^1 smoothing of `|x|`: `sqrt(x^2 + eps^2)`, the house `smoothAbs`.
/// Derivatives, with `M = sqrt(x^2 + eps^2)`:
///   f' = x/M,  f'' = eps^2/M^3,  f''' = -3 eps^2 x/M^5.
template <typename T>
T smoothAbs(const T& x, double eps = 1e-8) {
    using std::sqrt;
    const T e = detail::constantOf<T>(eps);
    return sqrt(x * x + e * e);
}

} // namespace lightape

#endif // LIGHTAPE_OPS_GENERIC_H
