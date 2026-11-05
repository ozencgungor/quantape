#ifndef LIGHTAPE_BASE_INCOMPLETE_GAMMA_H
#define LIGHTAPE_BASE_INCOMPLETE_GAMMA_H

/**
 * @file IncompleteGamma.h
 * @brief Regularized incomplete gamma (double) and its exact (shape, x)
 *        value/gradient/Hessian pack for the lightape AD layer
 *
 * In-house port of quantape::math::detail's incomplete-gamma kernels and the
 * `gammaCdfGradHess` parameter pack restricted to `rate = 1` (same owner; see
 * quantape/math/Distributions/detail/IncompleteGamma.h). Evaluation order is
 * preserved from the source.
 *
 * Kernels:
 *   * `gammaP(a, x) = P(a, x) = gamma(a, x) / Gamma(a)` (lower regularized),
 *   * `gammaQ(a, x) = Q(a, x) = Gamma(a, x) / Gamma(a)` (upper regularized),
 *   * `gammaGradHess(a, x)`: `{P, P_a, P_x, P_aa, P_ax, P_xx}`.
 *
 * Algorithm (Numerical Recipes style): power series for `x < a + 1`,
 * modified-Lentz continued fraction for `x >= a + 1`, both normalized with
 * `lgamma(a)`; the series/CF iterate until the increment is ~1e-17 relative
 * (iteration cap 10000). The a-direction derivatives are exact derivatives of
 * the *evaluated* series/CF expression, produced by instantiating the same
 * `gammaP<T>` kernel at a local two-variable, second-order forward jet whose
 * `lgamma` is differentiated through `digammaDouble`/`trigammaDouble`; the x
 * entries keep the closed forms (`P_x = pdf = exp(log pdf)`, `P_xx = pdf q`).
 * The exact `P_ax` comes from the jet cross term. No finite differences
 * anywhere; branch selection follows the value component and the convergence
 * test additionally requires the derivative components to have converged.
 *
 * Special values (same as the source): `x <= 0 -> P = 0, Q = 1`; `x = +inf ->
 * P = 1, Q = 0`; `a <= 0` or NaN input returns NaN. The gradient pack flags
 * the singular boundaries (`x = 0` gives all-NaN derivatives) instead of
 * inventing finite values. std-only, no allocation (the pack is noexcept).
 */

#include <cmath>
#include <limits>
#include <type_traits>

#include "lightape/base/Digamma.h"

namespace lightape {
namespace detail {

// ---------------------------------------------------------------------------
// Two-variable, second-order forward jet: exact local differentiation of the
// incomplete-gamma kernel without an AD engine. Port of quantape's
// SecondOrderJet (same owner). Components (value, d1, d2, d11, d12, d22).
// ---------------------------------------------------------------------------

struct GammaSecondOrderJet {
    double value = 0.0;
    double d1 = 0.0;
    double d2 = 0.0;
    double d11 = 0.0;
    double d12 = 0.0;
    double d22 = 0.0;

    GammaSecondOrderJet() = default;
    // Implicit on purpose: scalar literals inside the generic kernels.
    GammaSecondOrderJet(double v) : value(v) {} // NOLINT(google-explicit-constructor)
    GammaSecondOrderJet(double v, double firstA, double firstB, double secondAA, double secondAB,
                        double secondBB)
        : value(v), d1(firstA), d2(firstB), d11(secondAA), d12(secondAB), d22(secondBB) {}
};

inline GammaSecondOrderJet operator+(const GammaSecondOrderJet& x,
                                     const GammaSecondOrderJet& y) noexcept {
    return {x.value + y.value, x.d1 + y.d1,   x.d2 + y.d2,
            x.d11 + y.d11,     x.d12 + y.d12, x.d22 + y.d22};
}

inline GammaSecondOrderJet operator-(const GammaSecondOrderJet& x,
                                     const GammaSecondOrderJet& y) noexcept {
    return {x.value - y.value, x.d1 - y.d1,   x.d2 - y.d2,
            x.d11 - y.d11,     x.d12 - y.d12, x.d22 - y.d22};
}

inline GammaSecondOrderJet operator-(const GammaSecondOrderJet& x) noexcept {
    return {-x.value, -x.d1, -x.d2, -x.d11, -x.d12, -x.d22};
}

inline GammaSecondOrderJet operator*(const GammaSecondOrderJet& x,
                                     const GammaSecondOrderJet& y) noexcept {
    return {x.value * y.value,
            x.d1 * y.value + x.value * y.d1,
            x.d2 * y.value + x.value * y.d2,
            x.d11 * y.value + 2.0 * x.d1 * y.d1 + x.value * y.d11,
            x.d12 * y.value + x.d1 * y.d2 + x.d2 * y.d1 + x.value * y.d12,
            x.d22 * y.value + 2.0 * x.d2 * y.d2 + x.value * y.d22};
}

inline GammaSecondOrderJet operator/(const GammaSecondOrderJet& x,
                                     const GammaSecondOrderJet& y) noexcept {
    const double inv = 1.0 / y.value;
    const double w = x.value * inv;
    const double w1 = (x.d1 - w * y.d1) * inv;
    const double w2 = (x.d2 - w * y.d2) * inv;
    return {w,
            w1,
            w2,
            (x.d11 - w * y.d11 - 2.0 * w1 * y.d1) * inv,
            (x.d12 - w * y.d12 - w1 * y.d2 - w2 * y.d1) * inv,
            (x.d22 - w * y.d22 - 2.0 * w2 * y.d2) * inv};
}

inline GammaSecondOrderJet& operator+=(GammaSecondOrderJet& x,
                                       const GammaSecondOrderJet& y) noexcept {
    x = x + y;
    return x;
}

inline GammaSecondOrderJet& operator-=(GammaSecondOrderJet& x,
                                       const GammaSecondOrderJet& y) noexcept {
    x = x - y;
    return x;
}

inline GammaSecondOrderJet& operator*=(GammaSecondOrderJet& x,
                                       const GammaSecondOrderJet& y) noexcept {
    x = x * y;
    return x;
}

inline GammaSecondOrderJet& operator/=(GammaSecondOrderJet& x,
                                       const GammaSecondOrderJet& y) noexcept {
    x = x / y;
    return x;
}

inline bool operator==(const GammaSecondOrderJet& x, const GammaSecondOrderJet& y) noexcept {
    return x.value == y.value;
}

inline bool operator!=(const GammaSecondOrderJet& x, const GammaSecondOrderJet& y) noexcept {
    return x.value != y.value;
}

inline bool operator<(const GammaSecondOrderJet& x, const GammaSecondOrderJet& y) noexcept {
    return x.value < y.value;
}

inline bool operator<=(const GammaSecondOrderJet& x, const GammaSecondOrderJet& y) noexcept {
    return x.value <= y.value;
}

inline bool operator>(const GammaSecondOrderJet& x, const GammaSecondOrderJet& y) noexcept {
    return x.value > y.value;
}

inline bool operator>=(const GammaSecondOrderJet& x, const GammaSecondOrderJet& y) noexcept {
    return x.value >= y.value;
}

/// `fabs`: sign flip on all derivative components (a.e. derivative; `+` at 0).
inline GammaSecondOrderJet fabs(const GammaSecondOrderJet& x) noexcept {
    const double s = x.value < 0.0 ? -1.0 : 1.0;
    return {std::fabs(x.value), s * x.d1, s * x.d2, s * x.d11, s * x.d12, s * x.d22};
}

/// `max(|value|, |d1|, ..., |d22|)`: the convergence scale used by the
/// series/CF iteration test (see `incrementConverged`).
inline double maxAbsComponent(const GammaSecondOrderJet& x) noexcept {
    return std::fmax(std::fmax(std::fmax(std::fabs(x.value), std::fabs(x.d1)),
                               std::fmax(std::fabs(x.d2), std::fabs(x.d11))),
                     std::fmax(std::fabs(x.d12), std::fabs(x.d22)));
}

/// `log`: value exact, derivatives by logarithmic differentiation.
inline GammaSecondOrderJet log(const GammaSecondOrderJet& x) noexcept {
    const double inv = 1.0 / x.value;
    const double l1 = x.d1 * inv;
    const double l2 = x.d2 * inv;
    return {std::log(x.value),    l1, l2, x.d11 * inv - l1 * l1, x.d12 * inv - l1 * l2,
            x.d22 * inv - l2 * l2};
}

/// `exp`: `d^n exp = exp * (chain rule)`, same rule as `expm1`.
inline GammaSecondOrderJet exp(const GammaSecondOrderJet& x) noexcept {
    const double e = std::exp(x.value);
    return {e,
            e * x.d1,
            e * x.d2,
            e * (x.d11 + x.d1 * x.d1),
            e * (x.d12 + x.d1 * x.d2),
            e * (x.d22 + x.d2 * x.d2)};
}

/// `expm1`: derivatives are those of `exp`, the value uses `std::expm1` for
/// accuracy near zero.
inline GammaSecondOrderJet expm1(const GammaSecondOrderJet& x) noexcept {
    const double e = std::exp(x.value);
    return {std::expm1(x.value),
            e * x.d1,
            e * x.d2,
            e * (x.d11 + x.d1 * x.d1),
            e * (x.d12 + x.d1 * x.d2),
            e * (x.d22 + x.d2 * x.d2)};
}

/// `lgamma`: `d/da lgamma = psi(a)`, `d^2/da^2 = psi'(a)` (Digamma.h).
inline GammaSecondOrderJet lgamma(const GammaSecondOrderJet& x) noexcept {
    const double psi = digammaDouble(x.value);
    const double tri = trigammaDouble(x.value);
    return {std::lgamma(x.value),
            psi * x.d1,
            psi * x.d2,
            tri * x.d1 * x.d1 + psi * x.d11,
            tri * x.d1 * x.d2 + psi * x.d12,
            tri * x.d2 * x.d2 + psi * x.d22};
}

// ---------------------------------------------------------------------------
// Generic kernel helpers (double and the local jet).
// ---------------------------------------------------------------------------

/// Quiet NaN for floating-point `T`, `T(0) / T(0)` for the jet.
template <class T>
T nanValue() {
    if constexpr (std::is_floating_point_v<T>) {
        return std::numeric_limits<T>::quiet_NaN();
    } else {
        return T(0) / T(0);
    }
}

/// True only for +Inf (NaN-safe): `x - x` is NaN exactly for the infinities.
template <class T>
bool isPositiveInfinity(const T& x) {
    return x > T(0) && x - x != T(0);
}

/// Convergence test for the series/CF increments: the historical value
/// comparison `fabs(u) <= fabs(v) * tol`.
template <class T>
bool incrementConverged(const T& u, const T& v, const T& tol) {
    using std::fabs;
    return fabs(u) <= fabs(v) * tol;
}

/// Jet specialization: a vanishing value increment alone can hide a
/// not-yet-converged derivative (e.g. the continued fraction at `shape = 1`),
/// so every derivative component must also be below `maxAbsComponent(v) *
/// 1e-15` — well above jet roundoff (~1e-16) and far below the derivative
/// gates.
inline bool incrementConverged(const GammaSecondOrderJet& u, const GammaSecondOrderJet& v,
                               const GammaSecondOrderJet& tol) noexcept {
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
 * log is taken term-wise: no intermediate underflow in the prefactor.
 */
template <class T>
T gammaSeriesLogP(T a, T x) {
    using std::fabs;
    using std::lgamma;
    using std::log;
    T ap = a;
    T del = T(1) / a;
    T sum = del;
    for (int n = 1; n <= 10000; ++n) {
        ap += T(1);
        del *= x / ap;
        sum += del;
        if (incrementConverged(del, sum, T(1e-17))) {
            break;
        }
    }
    return log(sum) + a * log(x) - x - lgamma(a);
}

/**
 * @brief `log Q(a, x)` from the continued fraction (valid for `x >= a + 1`).
 *
 * Lentz's modified method as in Numerical Recipes `gcf`; the log of the
 * continued fraction times the prefactor, so `Q` can underflow in the caller
 * but `log Q` stays finite. A non-positive converged numerator cannot occur
 * mathematically; if rounding drives it there the result is -Inf.
 */
template <class T>
T gammaContinuedFractionLogQ(T a, T x) {
    using std::fabs;
    using std::lgamma;
    using std::log;
    const double tiny = 1e-300;
    T b = x + T(1) - a;
    T c = T(1) / T(tiny);
    T d = T(1) / b;
    T h = d;
    for (int i = 1; i <= 10000; ++i) {
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
            break;
        }
    }
    if (!(h > T(0))) {
        return T(-1) / T(0);
    }
    return log(h) + a * log(x) - x - lgamma(a);
}

/**
 * @brief Lower regularized incomplete gamma `P(a, x)`.
 *
 * `x < a + 1`: power series; otherwise `1 - exp(log Q)` via `-expm1(log Q)`,
 * which preserves accuracy when `Q` is tiny. Returns 0 for `x <= 0`, 1 at
 * `x = +inf` and NaN for `a <= 0` / NaN arguments.
 */
template <class T>
T gammaP(T a, T x) {
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
T gammaQ(T a, T x) {
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

/// Canonical double entries (thin explicit instantiations, one path).
inline double gammaPDouble(double a, double x) {
    return gammaP<double>(a, x);
}

inline double gammaQDouble(double a, double x) {
    return gammaQ<double>(a, x);
}

// ---------------------------------------------------------------------------
// Exact value/gradient/Hessian pack in (shape, x) for the `External` wrappers.
// Mirrors the source `gammaCdfGradHess` at `rate = 1`; the x entries keep the
// closed forms, the shape entries come from the 2-jet (see the file comment).
// Edges: invalid/NaN -> all-NaN; `x < 0` -> zero pack; `x = 0` -> all-NaN
// derivatives (the one-sided parameter/mixed derivatives are unbounded for
// `shape < 1`, so no finite value is invented); `x = +Inf` -> value 1, zero
// derivatives.
// ---------------------------------------------------------------------------

struct GammaGradHess {
    double value = 0.0;
    double gShape = 0.0;
    double gx = 0.0;
    double hShapeShape = 0.0;
    double hShapeX = 0.0;
    double hxx = 0.0;
};

/// All-NaN pack (invalid/edge inputs), see the section comment.
inline GammaGradHess gammaGradHessNaN() noexcept {
    const double nan = std::numeric_limits<double>::quiet_NaN();
    return {nan, nan, nan, nan, nan, nan};
}

/// log of `x^(shape-1) e^{-x}`, `C = 1 / Gamma(shape)`; rate=1 density log.
inline double gammaLogPdfValue(double shape, double x) {
    using std::lgamma;
    using std::log;
    return -lgamma(shape) + (shape - 1.0) * log(x) - x;
}

/// `{P, P_a, P_x, P_aa, P_ax, P_xx}` for `P = P(shape, x)`, all exact.
inline GammaGradHess gammaGradHess(double shape, double x) noexcept {
    using std::exp;
    if (!(shape > 0.0) || x != x) {
        return gammaGradHessNaN();
    }
    if (x < 0.0) {
        return GammaGradHess{};
    }
    if (x == 0.0) {
        return gammaGradHessNaN();
    }
    if (isPositiveInfinity(x)) {
        GammaGradHess out{};
        out.value = 1.0;
        return out;
    }

    const GammaSecondOrderJet aJet(shape, 1.0, 0.0, 0.0, 0.0, 0.0);
    const GammaSecondOrderJet xJet(x, 0.0, 1.0, 0.0, 0.0, 0.0);
    const GammaSecondOrderJet p = gammaP<GammaSecondOrderJet>(aJet, xJet);

    const double pdf = exp(gammaLogPdfValue(shape, x));
    const double q = (shape - 1.0) / x - 1.0;

    GammaGradHess out{};
    out.value = gammaPDouble(shape, x);
    out.gShape = p.d1;
    out.gx = pdf;
    out.hShapeShape = p.d11;
    out.hShapeX = p.d12;
    out.hxx = pdf * q;
    return out;
}

} // namespace detail
} // namespace lightape

#endif // LIGHTAPE_BASE_INCOMPLETE_GAMMA_H
