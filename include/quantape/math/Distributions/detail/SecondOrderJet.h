#ifndef QUANTAPE_MATH_DISTRIBUTIONS_DETAIL_SECOND_ORDER_JET_H
#define QUANTAPE_MATH_DISTRIBUTIONS_DETAIL_SECOND_ORDER_JET_H

/**
 * @file SecondOrderJet.h
 * @brief Minimal two-variable, second-order forward jet: exact local
 * differentiation of the incomplete-gamma kernels without an AD engine.
 *
 * `SecondOrderJet` carries `(value, d1, d2, d11, d12, d22)` — a function value
 * and its first/second partials with respect to two independent variables.
 * Arithmetic and the elementary functions used by the
 * series/continued-fraction kernels (`fabs`, `log`, `log1p`, `exp`, `expm1`,
 * `lgamma`) are overloaded with their exact chain rules; `lgamma` is
 * differentiated through `digammaDouble`/`trigammaDouble`, so an entire
 * derivative pack is produced by running a single templated kernel on one jet.
 *
 * Comparisons act on the value component, so branch selection follows the
 * ordinary `double` iteration. The series/CF convergence checks route through
 * `detail::incrementConverged`, whose jet overload additionally requires the
 * derivative components to have converged: a vanishing value increment alone
 * can hide a not-yet-converged derivative (e.g. the continued fraction at
 * `shape = 1`). This is an internal helper for the parameter-derivative packs;
 * it is std-only, allocation-free and `noexcept`. Treat the derivatives as
 * exact (machine-precision) derivatives of the evaluated series/CF expression;
 * no finite differences are involved anywhere.
 *
 * Naming: `d1`/`d2` are the first partials, `d11`/`d12`/`d22` the second ones.
 * The implicit single-`double` constructor lets the kernels keep writing
 * `T(1)`, `T(1e-17)`, `T(tiny)`, ... unchanged.
 */

#include "quantape/math/Distributions/detail/Digamma.h"

#include <cmath>

namespace quantape::math {
namespace detail {

struct SecondOrderJet {
    double value = 0.0;
    double d1 = 0.0;
    double d2 = 0.0;
    double d11 = 0.0;
    double d12 = 0.0;
    double d22 = 0.0;

    SecondOrderJet() = default;
    // Implicit on purpose: scalar literals inside the generic kernels.
    SecondOrderJet(double v) : value(v) {} // NOLINT(google-explicit-constructor)
    SecondOrderJet(double v, double firstA, double firstB, double secondAA, double secondAB,
                   double secondBB)
        : value(v), d1(firstA), d2(firstB), d11(secondAA), d12(secondAB), d22(secondBB) {}
};

inline SecondOrderJet operator+(const SecondOrderJet& x, const SecondOrderJet& y) noexcept {
    return {x.value + y.value, x.d1 + y.d1,   x.d2 + y.d2,
            x.d11 + y.d11,     x.d12 + y.d12, x.d22 + y.d22};
}

inline SecondOrderJet operator-(const SecondOrderJet& x, const SecondOrderJet& y) noexcept {
    return {x.value - y.value, x.d1 - y.d1,   x.d2 - y.d2,
            x.d11 - y.d11,     x.d12 - y.d12, x.d22 - y.d22};
}

inline SecondOrderJet operator-(const SecondOrderJet& x) noexcept {
    return {-x.value, -x.d1, -x.d2, -x.d11, -x.d12, -x.d22};
}

inline SecondOrderJet operator*(const SecondOrderJet& x, const SecondOrderJet& y) noexcept {
    return {x.value * y.value,
            x.d1 * y.value + x.value * y.d1,
            x.d2 * y.value + x.value * y.d2,
            x.d11 * y.value + 2.0 * x.d1 * y.d1 + x.value * y.d11,
            x.d12 * y.value + x.d1 * y.d2 + x.d2 * y.d1 + x.value * y.d12,
            x.d22 * y.value + 2.0 * x.d2 * y.d2 + x.value * y.d22};
}

inline SecondOrderJet operator/(const SecondOrderJet& x, const SecondOrderJet& y) noexcept {
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

inline SecondOrderJet& operator+=(SecondOrderJet& x, const SecondOrderJet& y) noexcept {
    x = x + y;
    return x;
}

inline SecondOrderJet& operator-=(SecondOrderJet& x, const SecondOrderJet& y) noexcept {
    x = x - y;
    return x;
}

inline SecondOrderJet& operator*=(SecondOrderJet& x, const SecondOrderJet& y) noexcept {
    x = x * y;
    return x;
}

inline SecondOrderJet& operator/=(SecondOrderJet& x, const SecondOrderJet& y) noexcept {
    x = x / y;
    return x;
}

inline bool operator==(const SecondOrderJet& x, const SecondOrderJet& y) noexcept {
    return x.value == y.value;
}

inline bool operator!=(const SecondOrderJet& x, const SecondOrderJet& y) noexcept {
    return x.value != y.value;
}

inline bool operator<(const SecondOrderJet& x, const SecondOrderJet& y) noexcept {
    return x.value < y.value;
}

inline bool operator<=(const SecondOrderJet& x, const SecondOrderJet& y) noexcept {
    return x.value <= y.value;
}

inline bool operator>(const SecondOrderJet& x, const SecondOrderJet& y) noexcept {
    return x.value > y.value;
}

inline bool operator>=(const SecondOrderJet& x, const SecondOrderJet& y) noexcept {
    return x.value >= y.value;
}

/// `fabs`: sign flip on all derivative components (a.e. derivative; `+` at 0).
inline SecondOrderJet fabs(const SecondOrderJet& x) noexcept {
    const double s = x.value < 0.0 ? -1.0 : 1.0;
    return {std::fabs(x.value), s * x.d1, s * x.d2, s * x.d11, s * x.d12, s * x.d22};
}

/// `max(|value|, |d1|, ..., |d22|)`: the convergence scale used by the
/// series/CF iteration test (see `incrementConverged`).
inline double maxAbsComponent(const SecondOrderJet& x) noexcept {
    return std::fmax(std::fmax(std::fmax(std::fabs(x.value), std::fabs(x.d1)),
                               std::fmax(std::fabs(x.d2), std::fabs(x.d11))),
                     std::fmax(std::fabs(x.d12), std::fabs(x.d22)));
}

/// `log`: value exact, derivatives by logarithmic differentiation.
inline SecondOrderJet log(const SecondOrderJet& x) noexcept {
    const double inv = 1.0 / x.value;
    const double l1 = x.d1 * inv;
    const double l2 = x.d2 * inv;
    return {std::log(x.value),    l1, l2, x.d11 * inv - l1 * l1, x.d12 * inv - l1 * l2,
            x.d22 * inv - l2 * l2};
}

/// `exp`: `d^n exp = exp * (chain rule)`, same rule as `expm1`.
inline SecondOrderJet exp(const SecondOrderJet& x) noexcept {
    const double e = std::exp(x.value);
    return {e,
            e * x.d1,
            e * x.d2,
            e * (x.d11 + x.d1 * x.d1),
            e * (x.d12 + x.d1 * x.d2),
            e * (x.d22 + x.d2 * x.d2)};
}

/// `expm1`: mathematically `exp - 1`, derivatives are those of `exp`, the
/// value uses `std::expm1` for accuracy near zero.
inline SecondOrderJet expm1(const SecondOrderJet& x) noexcept {
    const double e = std::exp(x.value);
    return {std::expm1(x.value),
            e * x.d1,
            e * x.d2,
            e * (x.d11 + x.d1 * x.d1),
            e * (x.d12 + x.d1 * x.d2),
            e * (x.d22 + x.d2 * x.d2)};
}

/// `log1p`: derivatives of `log(1 + x)`.
inline SecondOrderJet log1p(const SecondOrderJet& x) noexcept {
    const double inv = 1.0 / (1.0 + x.value);
    const double l1 = x.d1 * inv;
    const double l2 = x.d2 * inv;
    return {std::log1p(x.value),  l1, l2, x.d11 * inv - l1 * l1, x.d12 * inv - l1 * l2,
            x.d22 * inv - l2 * l2};
}

/// `lgamma`: `d/da lgamma = psi(a)`, `d^2/da^2 = psi'(a)` (Digamma.h).
inline SecondOrderJet lgamma(const SecondOrderJet& x) noexcept {
    const double psi = digammaDouble(x.value);
    const double tri = trigammaDouble(x.value);
    return {std::lgamma(x.value),
            psi * x.d1,
            psi * x.d2,
            tri * x.d1 * x.d1 + psi * x.d11,
            tri * x.d1 * x.d2 + psi * x.d12,
            tri * x.d2 * x.d2 + psi * x.d22};
}

} // namespace detail
} // namespace quantape::math

#endif // QUANTAPE_MATH_DISTRIBUTIONS_DETAIL_SECOND_ORDER_JET_H
