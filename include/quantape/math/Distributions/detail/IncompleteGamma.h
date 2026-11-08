#ifndef QUANTAPE_MATH_DISTRIBUTIONS_DETAIL_INCOMPLETE_GAMMA_H
#define QUANTAPE_MATH_DISTRIBUTIONS_DETAIL_INCOMPLETE_GAMMA_H

#include "quantape/math/Distributions/detail/NormalQuantile.h"
#include "quantape/math/Distributions/detail/SecondOrderJet.h"

#include <cmath>
#include <limits>
#include <type_traits>

namespace quantape::math {
namespace detail {
/**
 * @file IncompleteGamma.h
 * @brief Regularized incomplete gamma functions and shared derivative packs.
 *
 * Kernels:
 *   * `gammaP(a, x) = P(a, x) = gamma(a, x) / Gamma(a)` (lower regularized),
 *   * `gammaQ(a, x) = Q(a, x) = Gamma(a, x) / Gamma(a)` (upper regularized),
 *   * `gammaLogP` / `gammaLogQ`: their logarithms, evaluated so that deep-tail
 *     probabilities stay finite (`log P` is built directly from the series
 *     exponent, `log Q` directly from the continued fraction),
 *   * `gammaPdfD1D2` / `gammaCdfD1D2`: `{value, value', value''}` for the
 *     Gamma(shape, rate) density and CDF; every derivative is analytic,
 *   * `gammaCdfGradHess` / `gammaPdfGradHess`: full (shape, rate, x)
 *     gradient/Hessian packs; the parameter entries are exact derivatives of
 *     the series/continued fraction, obtained by evaluating `gammaP<T>` on a
 *     `SecondOrderJet` local 2-jet in `(shape, y)`, the x entries keep the
 *     closed forms of `gammaCdfD1D2`/`gammaPdfD1D2` bit-for-bit,
 *   * `gammaPDouble` / `gammaQDouble` / `gammaLogPDouble` / `gammaLogQDouble`:
 *     the named canonical `double` entries (explicit `double` instantiations of
 *     the templates); distribution headers dispatch `DoubleT = double` to these
 *     and route other scalars through the templates above.
 *
 * Algorithm (Numerical Recipes style): power series for `x < a + 1`,
 * modified-Lentz continued fraction for `x >= a + 1`, both normalized with the
 * prefactor `x^a e^{-x} / Gamma(a)`. The series/CF iterate until the increment
 * is ~1e-17 relative, well below the rounding of the prefactor.
 *
 * Large shape (`a >= kGammaLargeShape = 1e6`): the iteration budget and the
 * prefactor both scale with the shape so the kernels stay accurate up to the
 * documented cap ~1e8 and beyond.
 *   * Both the power series and the Lentz CF need `O(sqrt(a))` terms near
 *     `x ~ a` (the series term is ~`exp(-n^2 / (2a))`, so the 1e-17 test needs
 *     `n ~ 8.9 sqrt(a)`). The budget is therefore `10000` for moderate shape
 *     and `9 sqrt(a) + 100` above it (`gammaIterationCap`), which is ~9e4
 *     terms at `a = 1e8`.
 *   * The prefactor is evaluated in the Stirling form
 *     `a log x - x - lgamma(a) = a (log(1+u) - u) + L(a)`, `u = (x - a)/a`,
 *     `L(a) = 0.5 log(a / 2 pi) - 1/(12a) + ...` (`gammaLogPrefactor`). The
 *     literal form cancels terms of size `|lgamma(a)|`, so its absolute
 *     exponent error is `~eps |lgamma(a)|` (~4e-7 at a = 1e8, i.e. 4e-7
 *     relative on the probability); the Stirling form is `O(log a)` and, with
 *     `log1pMinusArg` for the small-`u` bracket, holds ~1e-13 relative
 *     accuracy. Below the threshold the literal expression is evaluated
 *     unchanged.
 *   * The budget is hard-capped at 4e6 iterations. If it is exhausted (shape
 *     beyond ~2e11) the kernels return NaN; they never return a truncated
 *     sum.
 *
 * Special values: `x <= 0 -> P = 0, Q = 1`; `x = +inf -> P = 1, Q = 0`;
 * `a <= 0` or NaN input returns NaN. The log variants remain finite for
 * probabilities far below 1e-300 for moderate shapes; the caller's exp() is
 * the only underflow limit.
 */
struct ValueD1D2 {
    double value = 0.0;
    double d1 = 0.0;
    double d2 = 0.0;
};

/// +Infinity for floating-point `T`, `T(1) / T(0)` (natural IEEE propagation)
/// for AD scalars.
template <class T>
inline T positiveInfinity() {
    if constexpr (std::is_floating_point_v<T>) {
        return std::numeric_limits<T>::infinity();
    } else {
        return T(1) / T(0);
    }
}

/// -Infinity, see `positiveInfinity`.
template <class T>
inline T negativeInfinity() {
    if constexpr (std::is_floating_point_v<T>) {
        return -std::numeric_limits<T>::infinity();
    } else {
        return T(-1) / T(0);
    }
}

/// Quiet NaN for floating-point `T`, `T(0) / T(0)` for AD scalars.
template <class T>
inline T nanValue() {
    if constexpr (std::is_floating_point_v<T>) {
        return std::numeric_limits<T>::quiet_NaN();
    } else {
        return T(0) / T(0);
    }
}

/// True only for +Inf (NaN-safe): `x - x` is NaN exactly for the infinities.
template <class T>
inline bool isPositiveInfinity(const T& x) {
    return x > T(0) && x - x != T(0);
}

/// Shapes at or above this use the scaled iteration budget and the Stirling
/// form of the log prefactor (see the file comment); below it the literal
/// prefactor and the 10000-term budget apply.
inline constexpr double kGammaLargeShape = 1.0e6;

/// The fixed series/CF budget used for moderate shape.
inline constexpr int kGammaBaseIterationCap = 10000;

/// Iteration budget for the series/CF at shape `a`.
///
/// `kGammaBaseIterationCap` for `a <= kGammaLargeShape`, and `9 sqrt(a) + 100`
/// above, which covers the 1e-17 term test at `x ~ a` (`n ~ 8.9 sqrt(a)`) up
/// to the documented shape cap ~1e8 (~9e4 terms). Capped at `4e6` so a
/// pathological shape fails deterministically (the callers return NaN) instead
/// of running for minutes.
inline int gammaIterationCap(double a) {
    constexpr int kHardCap = 4000000;
    if (!(a > kGammaLargeShape)) {
        return kGammaBaseIterationCap;
    }
    const double scaled = 9.0 * std::sqrt(a) + 100.0;
    if (!(scaled < static_cast<double>(kHardCap))) {
        return kHardCap;
    }
    const int cap = static_cast<int>(scaled);
    return cap > kGammaBaseIterationCap ? cap : kGammaBaseIterationCap;
}

/**
 * @brief `L(a) = a log a - a - lgamma(a)`, the Stirling remainder.
 *
 * Evaluated by its asymptotic series
 * `0.5 log(a / 2 pi) - 1/(12a) + 1/(360a^3) - 1/(1260a^5) + 1/(1680a^7) - ...`,
 * which is `O(log a)` and therefore free of the cancellation inherent in the
 * literal `a log a - a - lgamma(a)` (absolute error `~eps |lgamma(a)|`). Used
 * only for `a >= kGammaLargeShape`; at `a = 1e6` the first omitted term is
 * `-691/(360360 a^11) ~ 2e-72`.
 */
template <class T>
inline T gammaStirlingRemainder(T a) {
    using std::log;
    constexpr double k2Pi = 6.283185307179586476925286766559005768;
    const T invA = T(1) / a;
    const T invA2 = invA * invA;
    return T(0.5) * log(a / T(k2Pi)) -
           invA * (T(1.0 / 12.0) -
                   invA2 * (T(1.0 / 360.0) - invA2 * (T(1.0 / 1260.0) - invA2 * T(1.0 / 1680.0))));
}

/**
 * @brief `log(1 + u) - u`, evaluated by its Taylor series for `|u| <= 0.25`.
 *
 * The direct form cancels its `O(u)` leading term and leaves only roundoff of
 * the bracket `O(u^2)`; the series `-u^2/2 + u^3/3 - u^4/4 + ...` (geometric at
 * `|u| = 0.25`, 40 terms reach ~1e-25) has no such cancellation. Outside the
 * small-`u` window the direct form is well conditioned and is used instead.
 */
template <class T>
inline T log1pMinusArg(T u) {
    using std::fabs;
    using std::log1p;
    if (!(fabs(u) <= T(0.25))) {
        return log1p(u) - u;
    }
    const T u2 = u * u;
    T power = u2;
    T sum = -T(0.5) * u2;
    for (int m = 3; m <= 40; ++m) {
        power *= u;
        sum += (m % 2 == 1 ? T(1) : T(-1)) * power / T(m);
        if (fabs(power) <= fabs(sum) * T(1e-18)) {
            break;
        }
    }
    return sum;
}

/**
 * @brief `a log x - x - lgamma(a)`, the series/CF log prefactor.
 *
 * Evaluated for `a >= kGammaLargeShape` only (the callers keep the literal
 * expression below the threshold, bitwise). With `u = (x - a)/a` the Stirling
 * remainder separates the `O(a)` cancellation:
 *   `a log x - x - lgamma(a) = a (log(1 + u) - u) + L(a)`.
 * For `x` within a factor of two of `a` the difference `x - a` is exact
 * (Sterbenz) and `log1pMinusArg` keeps the bracket relative-accurate, so the
 * exponent is good to ~1e-13 even when `|lgamma(a)| ~ 1e9`.
 */
template <class T>
inline T gammaLogPrefactor(T a, T x) {
    const T u = (x - a) / a;
    return gammaStirlingRemainder(a) + a * log1pMinusArg(u);
}

/**
 * @brief Convergence test for the series/CF increments.
 *
 * Generic (`double`, AD scalars): the historical value comparison
 * `fabs(u) <= fabs(v) * tol`, so existing scalar results are unchanged.
 *
 * The `SecondOrderJet` overload additionally requires every derivative
 * component of `u` to be below `maxAbsComponent(v) * 1e-15`: a zero value
 * increment alone can hide a not-yet-converged derivative, e.g. in the
 * continued fraction at `shape = 1`, where the first correction coefficient
 * `-1 * (1 - shape)` vanishes and `delta` reaches 1 before its shape
 * derivative converges. The 1e-15 derivative floor is well above the jet
 * roundoff noise (~1e-16) and far below the 1e-9/1e-10 test gates.
 */
template <class T>
inline bool incrementConverged(const T& u, const T& v, const T& tol) {
    using std::fabs;
    return fabs(u) <= fabs(v) * tol;
}

/// Jet specialization of `incrementConverged`, see the generic comment.
inline bool incrementConverged(const SecondOrderJet& u, const SecondOrderJet& v,
                               const SecondOrderJet& tol) noexcept {
    const double derivativeBound = maxAbsComponent(v) * 1e-15;
    return std::fabs(u.value) <= std::fabs(v.value) * tol.value &&
           std::fabs(u.d1) <= derivativeBound && std::fabs(u.d2) <= derivativeBound &&
           std::fabs(u.d11) <= derivativeBound && std::fabs(u.d12) <= derivativeBound &&
           std::fabs(u.d22) <= derivativeBound;
}

/**
 * @brief `log P(a, x)` from the power series (valid for `0 < x < a + 1`).
 *
 * `P(a, x) = x^a e^{-x} / Gamma(a) * sum_n x^n / (a (a+1) ... (a+n))`, so the
 * log is taken term-wise: no intermediate underflow in the prefactor. The
 * iteration budget scales with `sqrt(a)`; if it is exhausted (shape beyond the
 * documented cap) the result is NaN rather than a truncated sum (see the file
 * comment).
 */
template <class T>
inline T gammaSeriesLogP(T a, T x) {
    using std::fabs;
    using std::lgamma;
    using std::log;
    const int cap = gammaIterationCap(scalarValue(a));
    T ap = a;
    T del = T(1) / a;
    T sum = del;
    bool converged = false;
    for (int n = 1; n <= cap; ++n) {
        ap += T(1);
        del *= x / ap;
        sum += del;
        if (incrementConverged(del, sum, T(1e-17))) {
            converged = true;
            break;
        }
    }
    if (!converged && cap > kGammaBaseIterationCap) {
        return nanValue<T>();
    }
    if (scalarValue(a) < kGammaLargeShape) {
        return log(sum) + a * log(x) - x - lgamma(a);
    }
    return log(sum) + gammaLogPrefactor(a, x);
}

/**
 * @brief `log Q(a, x)` from the continued fraction (valid for `x >= a + 1`).
 *
 * Lentz's modified method as in Numerical Recipes `gcf`; returns the log of
 * the continued fraction times the same prefactor, so `Q` can underflow in
 * the caller but `log Q` stays finite. A non-positive converged numerator
 * cannot occur mathematically; if rounding drives it there the result is
 * -Inf (probability below double precision). The iteration budget scales with
 * `sqrt(a)`; an exhausted budget (shape beyond the documented cap) returns NaN.
 */
template <class T>
inline T gammaContinuedFractionLogQ(T a, T x) {
    using std::fabs;
    using std::lgamma;
    using std::log;
    const double tiny = 1e-300;
    const int cap = gammaIterationCap(scalarValue(a));
    T b = x + T(1) - a;
    T c = T(1) / T(tiny);
    T d = T(1) / b;
    T h = d;
    bool converged = false;
    for (int i = 1; i <= cap; ++i) {
        const T an = -T(i) * (T(i) - a);
        b += T(2);
        d = an * d + b;
        if (fabs(d) < T(tiny)) {
            d = T(tiny);
        }
        c = b + an / c;
        if (fabs(c) < T(tiny)) {
            c = T(tiny);
        }
        d = T(1) / d;
        const T delta = d * c;
        h *= delta;
        if (incrementConverged(delta - T(1), T(1), T(1e-17))) {
            converged = true;
            break;
        }
    }
    if (!converged && cap > kGammaBaseIterationCap) {
        return nanValue<T>();
    }
    if (!(h > T(0))) {
        return negativeInfinity<T>();
    }
    if (scalarValue(a) < kGammaLargeShape) {
        return log(h) + a * log(x) - x - lgamma(a);
    }
    return log(h) + gammaLogPrefactor(a, x);
}

/// `log P(a, x)`, finite in the deep lower tail; NaN for invalid `a`.
template <class T>
inline T gammaLogP(T a, T x) {
    using std::expm1;
    using std::log;
    if (!(a > T(0)) || x != x) {
        return nanValue<T>();
    }
    if (x <= T(0)) {
        return negativeInfinity<T>();
    }
    if (isPositiveInfinity(x)) {
        return T(0);
    }
    if (x < a + T(1)) {
        return gammaSeriesLogP(a, x);
    }
    return log(-expm1(gammaContinuedFractionLogQ(a, x)));
}

/// `log Q(a, x)`, finite in the deep upper tail; NaN for invalid `a`.
template <class T>
inline T gammaLogQ(T a, T x) {
    using std::expm1;
    using std::log;
    if (!(a > T(0)) || x != x) {
        return nanValue<T>();
    }
    if (x <= T(0)) {
        return T(0);
    }
    if (isPositiveInfinity(x)) {
        return negativeInfinity<T>();
    }
    if (x < a + T(1)) {
        return log(-expm1(gammaSeriesLogP(a, x)));
    }
    return gammaContinuedFractionLogQ(a, x);
}

/**
 * @brief Lower regularized incomplete gamma `P(a, x)`.
 *
 * `x < a + 1`: power series; otherwise `1 - exp(log Q)` via `-expm1(log Q)`,
 * which preserves accuracy when `Q` is tiny. Returns 0 for `x <= 0`, 1 at
 * `x = +inf` and NaN for `a <= 0` / NaN arguments.
 */
template <class T>
inline T gammaP(T a, T x) {
    using std::exp;
    using std::expm1;
    if (!(a > T(0)) || x != x) {
        return nanValue<T>();
    }
    if (x <= T(0)) {
        return T(0);
    }
    if (isPositiveInfinity(x)) {
        return T(1);
    }
    if (x < a + T(1)) {
        return exp(gammaSeriesLogP(a, x));
    }
    return -expm1(gammaContinuedFractionLogQ(a, x));
}

/**
 * @brief Upper regularized incomplete gamma `Q(a, x) = 1 - P(a, x)`.
 *
 * Mirror of `gammaP`: `-expm1(log P)` in the series branch (accurate for
 * small `P`), the continued fraction otherwise.
 */
template <class T>
inline T gammaQ(T a, T x) {
    using std::exp;
    using std::expm1;
    if (!(a > T(0)) || x != x) {
        return nanValue<T>();
    }
    if (x <= T(0)) {
        return T(1);
    }
    if (isPositiveInfinity(x)) {
        return T(0);
    }
    if (x < a + T(1)) {
        return -expm1(gammaSeriesLogP(a, x));
    }
    return exp(gammaContinuedFractionLogQ(a, x));
}

// ---------------------------------------------------------------------------
// Named double entry points (the `...Double` suffix marks the canonical double
// path). The distribution wrappers dispatch `DoubleT = double` here and other
// scalars to the templates above; each entry is a thin explicit `double`
// instantiation, so there is one canonical double path per kernel and no
// duplicated algorithm body.
// ---------------------------------------------------------------------------

/// `gammaP` at `T = double`.
inline double gammaPDouble(double a, double x) {
    return gammaP<double>(a, x);
}

/// `gammaQ` at `T = double`.
inline double gammaQDouble(double a, double x) {
    return gammaQ<double>(a, x);
}

/// `gammaLogP` at `T = double`.
inline double gammaLogPDouble(double a, double x) {
    return gammaLogP<double>(a, x);
}

/// `gammaLogQ` at `T = double`.
inline double gammaLogQDouble(double a, double x) {
    return gammaLogQ<double>(a, x);
}

/**
 * @brief Gamma(shape, rate) density with its first two x-derivatives at `x = 0`
 * (one-sided limits).
 *
 * With `u = shape - 1` and `f(x) = C x^u e^{-rate x}`, `C =
 * rate^(u+1)/Gamma(u+1)`: `f(0)`/`f'(0)`/`f''(0)` are 0, a finite value or
 * +-Inf depending on `u`; each case is the exact leading term.
 */
inline ValueD1D2 gammaPdfD1D2AtZero(double shape, double rate) {
    const double u = shape - 1.0;
    const double inf = std::numeric_limits<double>::infinity();
    ValueD1D2 out{};
    if (u > 0.0) {
        out.value = 0.0;
    } else if (u < 0.0) {
        out.value = inf;
    } else {
        out.value = rate;
    }
    if (u > 1.0) {
        out.d1 = 0.0;
    } else if (u == 1.0) {
        out.d1 = rate * rate;
    } else if (u > 0.0) {
        out.d1 = inf;
    } else if (u == 0.0) {
        out.d1 = -rate * rate;
    } else {
        out.d1 = -inf;
    }
    if (u > 2.0) {
        out.d2 = 0.0;
    } else if (u == 2.0) {
        out.d2 = rate * rate * rate;
    } else if (u > 1.0) {
        out.d2 = inf;
    } else if (u == 1.0) {
        out.d2 = -2.0 * rate * rate * rate;
    } else if (u > 0.0) {
        out.d2 = -inf;
    } else if (u == 0.0) {
        out.d2 = rate * rate * rate;
    } else {
        out.d2 = inf;
    }
    return out;
}

/// log of `C x^(shape-1) e^{-rate x}`, `C = rate^shape / Gamma(shape)`.
inline double gammaLogPdfValue(double shape, double rate, double x) {
    using std::lgamma;
    using std::log;
    return shape * log(rate) - lgamma(shape) + (shape - 1.0) * log(x) - rate * x;
}

/**
 * @brief `{f, f', f''}` for the Gamma(shape, rate) density, all analytic.
 *
 * `f'/f = (shape - 1)/x - rate =: q` and `f''/f = q^2 - (shape - 1)/x^2`.
 * `x <= 0` uses one-sided limits (`x < 0` is identically zero); `x = +inf`
 * is `{0, 0, 0}`; invalid parameters or NaN give a NaN pack.
 */
inline ValueD1D2 gammaPdfD1D2(double shape, double rate, double x) {
    using std::exp;
    const double nan = std::numeric_limits<double>::quiet_NaN();
    if (!(shape > 0.0) || !(rate > 0.0) || x != x) {
        return ValueD1D2{nan, nan, nan};
    }
    if (x < 0.0) {
        return ValueD1D2{0.0, 0.0, 0.0};
    }
    if (x == 0.0) {
        return gammaPdfD1D2AtZero(shape, rate);
    }
    if (isPositiveInfinity(x)) {
        return ValueD1D2{0.0, 0.0, 0.0};
    }
    const double pdf = exp(gammaLogPdfValue(shape, rate, x));
    const double q = (shape - 1.0) / x - rate;
    ValueD1D2 out{};
    out.value = pdf;
    out.d1 = pdf * q;
    out.d2 = pdf * (q * q - (shape - 1.0) / (x * x));
    return out;
}

/**
 * @brief `{F, F', F''}` for the Gamma(shape, rate) CDF, all analytic.
 *
 * `F = P(shape, rate x)`, hence `F' = f` and `F'' = f' = f q` (analytic
 * density derivatives, no numerical differentiation). Boundaries as in
 * `gammaPdfD1D2`; at `x = 0` the values are the one-sided limits.
 */
inline ValueD1D2 gammaCdfD1D2(double shape, double rate, double x) {
    using std::exp;
    const double nan = std::numeric_limits<double>::quiet_NaN();
    if (!(shape > 0.0) || !(rate > 0.0) || x != x) {
        return ValueD1D2{nan, nan, nan};
    }
    if (x <= 0.0) {
        if (x < 0.0) {
            return ValueD1D2{0.0, 0.0, 0.0};
        }
        const ValueD1D2 atZero = gammaPdfD1D2AtZero(shape, rate);
        return ValueD1D2{0.0, atZero.value, atZero.d1};
    }
    if (isPositiveInfinity(x)) {
        return ValueD1D2{1.0, 0.0, 0.0};
    }
    const double pdf = exp(gammaLogPdfValue(shape, rate, x));
    const double q = (shape - 1.0) / x - rate;
    ValueD1D2 out{};
    out.value = gammaPDouble(shape, rate * x);
    out.d1 = pdf;
    out.d2 = pdf * q;
    return out;
}

// ---------------------------------------------------------------------------
// Full parameter-derivative packs.
//
// Method (documented decision): the shape and mixed shape derivatives are
// exact derivatives of the *evaluated* series/continued-fraction expression,
// produced by instantiating the same `gammaP<T>` kernel at
// `T = SecondOrderJet` with a local 2-jet in `(shape, y)`, `y = rate x`
// (option (b) of the workstream spec; no Stan code is copied). `lgamma` is
// differentiated through `digammaDouble`/`trigammaDouble`; branch selection
// follows the value component (ordinary double iteration), while the
// convergence test additionally requires the derivative components to have
// converged (`incrementConverged`). No finite differences anywhere.
//
// The x entries are kept as the existing closed forms (`exp(gammaLogPdfValue)`
// and `q = (shape - 1)/x - rate`, identical arithmetic to
// `gammaCdfD1D2`/`gammaPdfD1D2`) so the packs reproduce the old x-packs
// bit-for-bit; `F_ay, F_aa` come from the jet and the rate entries use the
// cancellation-free density closed forms (`dy/drate = x d/dy`).
//
// Convention: `pdf(x)` below is the Gamma(shape, rate) density at `x`, so
// `gx = pdf` and `gRate = x * p_y(rate x) = (x / rate) * pdf`.
//
// Edges: invalid shape/rate or NaN x -> all-NaN; `x < 0` -> zero pack (the law
// is identically zero there, every partial vanishes); `x = 0` -> all-NaN (the
// one-sided parameter/mixed derivatives are unbounded in general and the
// shape derivative crosses `shape = 1`); `x = +Inf` -> value/zero pack.
// ---------------------------------------------------------------------------

/// (shape, rate, x) value + gradient + Hessian for the Gamma CDF.
struct GammaCdfGradHess {
    double value = 0.0;
    double gShape = 0.0;
    double gRate = 0.0;
    double gx = 0.0;
    double hShapeShape = 0.0;
    double hShapeRate = 0.0;
    double hShapeX = 0.0;
    double hRateRate = 0.0;
    double hRateX = 0.0;
    double hxx = 0.0;
};

/// All-NaN CDF pack (invalid/edge inputs), see the section comment.
inline GammaCdfGradHess gammaCdfGradHessNaN() noexcept {
    const double nan = std::numeric_limits<double>::quiet_NaN();
    return {nan, nan, nan, nan, nan, nan, nan, nan, nan, nan};
}

/**
 * @brief `{F, F_a, F_r, F_x, F_aa, F_ar, F_ax, F_rr, F_rx, F_xx}` for
 * `F = P(shape, rate x)`, all exact.
 *
 * `F_a, F_aa, F_ar, F_ax` come from the exact 2-jet of the incomplete-gamma
 * kernel at `(shape, y)`, `y = rate x`; `F_x = pdf`, `F_xx = pdf q` are the
 * closed forms of `gammaCdfD1D2`; `F_r = (x/rate) pdf`,
 * `F_rr = pdf x ((shape - 1)/rate - x)/rate` and `F_rx = pdf (shape/rate - x)`
 * are the cancellation-free density-derivative closed forms (the log-space
 * jet's `d22` cancels at `1/y^2` for tiny `y`). See the section comment for
 * the method and edge behavior.
 */
inline GammaCdfGradHess gammaCdfGradHess(double shape, double rate, double x) noexcept {
    using std::exp;
    if (!(shape > 0.0) || !(rate > 0.0) || x != x) {
        return gammaCdfGradHessNaN();
    }
    if (x < 0.0) {
        return GammaCdfGradHess{};
    }
    if (x == 0.0) {
        return gammaCdfGradHessNaN();
    }
    if (isPositiveInfinity(x)) {
        GammaCdfGradHess out{};
        out.value = 1.0;
        return out;
    }

    const double y = rate * x;
    const SecondOrderJet aJet(shape, 1.0, 0.0, 0.0, 0.0, 0.0);
    const SecondOrderJet yJet(y, 0.0, 1.0, 0.0, 0.0, 0.0);
    const SecondOrderJet p = gammaP<SecondOrderJet>(aJet, yJet);

    const double pdf = exp(gammaLogPdfValue(shape, rate, x));
    const double q = (shape - 1.0) / x - rate;

    GammaCdfGradHess out{};
    out.value = gammaPDouble(shape, y);
    out.gShape = p.d1;
    out.gRate = (x / rate) * pdf;
    out.gx = pdf;
    out.hShapeShape = p.d11;
    out.hShapeRate = x * p.d12;
    out.hShapeX = rate * p.d12;
    // The r-Hessian entries are derivatives of the density, where the
    // log-space jet's d22 cancels at 1/y^2 for tiny y (f_yy + f_y^2); the
    // closed forms `F_rr = pdf x ((a - 1)/r - x)/r` and `F_rx = pdf (a/r - x)`
    // are cancellation-free and are used instead.
    out.hRateRate = pdf * x * ((shape - 1.0) / rate - x) / rate;
    out.hRateX = pdf * (shape / rate - x);
    out.hxx = pdf * q;
    return out;
}

/// (shape, rate, x) value + gradient + Hessian for the Gamma density.
struct GammaPdfGradHess {
    double value = 0.0;
    double gShape = 0.0;
    double gRate = 0.0;
    double gx = 0.0;
    double hShapeShape = 0.0;
    double hShapeRate = 0.0;
    double hShapeX = 0.0;
    double hRateRate = 0.0;
    double hRateX = 0.0;
    double hxx = 0.0;
};

/// All-NaN density pack (invalid/edge inputs), see the section comment.
inline GammaPdfGradHess gammaPdfGradHessNaN() noexcept {
    const double nan = std::numeric_limits<double>::quiet_NaN();
    return {nan, nan, nan, nan, nan, nan, nan, nan, nan, nan};
}

/**
 * @brief `{f, f_a, f_r, f_x, f_aa, f_ar, f_ax, f_rr, f_rx, f_xx}` for the
 * Gamma(shape, rate) density, all closed form.
 *
 * With `L = log f = shape log(rate) - lgamma(shape) + (shape - 1) log x -
 * rate x` and `La = log(rate) + log(x) - psi(shape)`, `Lr = shape/rate - x`,
 * `Lx = q = (shape - 1)/x - rate`:
 *   `f_a = f La`, `f_r = f Lr`, `f_x = f q`,
 *   `f_aa = f (La^2 - psi'(shape))`, `f_ar = f (La Lr + 1/rate)`,
 *   `f_ax = f (La q + 1/x)`, `f_rr = f (Lr^2 - shape/rate^2)`,
 *   `f_rx = f (Lr q - 1)`, `f_xx = f (q^2 - (shape - 1)/x^2)`.
 * `f, f_x, f_xx` use the identical arithmetic of `gammaPdfD1D2`, so they
 * reproduce that pack bit-for-bit. See the section comment for edge behavior.
 */
inline GammaPdfGradHess gammaPdfGradHess(double shape, double rate, double x) noexcept {
    using std::exp;
    using std::log;
    if (!(shape > 0.0) || !(rate > 0.0) || x != x) {
        return gammaPdfGradHessNaN();
    }
    if (x < 0.0 || isPositiveInfinity(x)) {
        return GammaPdfGradHess{};
    }
    if (x == 0.0) {
        return gammaPdfGradHessNaN();
    }
    const double pdf = exp(gammaLogPdfValue(shape, rate, x));
    const double q = (shape - 1.0) / x - rate;
    const double psi = digammaDouble(shape);
    const double tri = trigammaDouble(shape);
    const double la = log(rate) + log(x) - psi;
    const double lr = shape / rate - x;

    GammaPdfGradHess out{};
    out.value = pdf;
    out.gShape = pdf * la;
    out.gRate = pdf * lr;
    out.gx = pdf * q;
    out.hShapeShape = pdf * (la * la - tri);
    out.hShapeRate = pdf * (la * lr + 1.0 / rate);
    out.hShapeX = pdf * (la * q + 1.0 / x);
    out.hRateRate = pdf * (lr * lr - shape / (rate * rate));
    out.hRateX = pdf * (lr * q - 1.0);
    out.hxx = pdf * (q * q - (shape - 1.0) / (x * x));
    return out;
}

} // namespace detail
} // namespace quantape::math

#endif // QUANTAPE_MATH_DISTRIBUTIONS_DETAIL_INCOMPLETE_GAMMA_H
