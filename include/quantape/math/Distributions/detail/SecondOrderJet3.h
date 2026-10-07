#ifndef QUANTAPE_MATH_DISTRIBUTIONS_DETAIL_SECOND_ORDER_JET3_H
#define QUANTAPE_MATH_DISTRIBUTIONS_DETAIL_SECOND_ORDER_JET3_H

/**
 * @file SecondOrderJet3.h
 * @brief Minimal three-variable, second-order forward jet: exact local
 * differentiation of the incomplete-beta kernel without an AD engine.
 *
 * `SecondOrderJet3` carries a function value and its first/second partials
 * with respect to three independent variables `(A, B, X)`, in the fixed
 * component order
 *
 *     (value, dA, dB, dX, dAA, dAB, dAX, dBB, dBX, dXX)
 *
 * — ten doubles, no allocation, `noexcept`. Arithmetic and the elementary
 * functions used by the series/continued-fraction kernels (`fabs`, `log`,
 * `log1p`, `exp`, `expm1`, `lgamma`) are overloaded with their exact chain
 * rules, so an entire (3-parameter) gradient/Hessian pack is produced by
 * running a single templated kernel on one jet. `lgamma` is differentiated
 * through `digammaDouble`/`trigammaDouble` (Digamma.h), exactly as the
 * two-variable `SecondOrderJet` does.
 *
 * The type is a drop-in generic scalar for the `incBeta<T>` kernel: it is
 * seeded with one unit derivative per parameter and the kernel's arithmetic
 * performs the full chain rule internally. Comparisons act on the value
 * component, so branch selection follows the ordinary `double` iteration.
 * The continued-fraction convergence test is handled in `IncompleteBeta.h`
 * (a vanishing value increment alone can hide a not-yet-converged
 * derivative); the generic jet has no iteration logic of its own.
 *
 * This mirrors `SecondOrderJet.h` component-for-component with a third
 * variable; it is deliberately generic and reusable (e.g. for further
 * three-parameter kernels). Treat the derivatives as exact
 * (machine-precision) derivatives of the evaluated expression; no finite
 * differences are involved anywhere.
 */

#include "quantape/math/Distributions/detail/Digamma.h"

#include <cmath>

namespace quantape::math {
namespace detail {

struct SecondOrderJet3 {
    double value = 0.0;
    double dA = 0.0;
    double dB = 0.0;
    double dX = 0.0;
    double dAA = 0.0;
    double dAB = 0.0;
    double dAX = 0.0;
    double dBB = 0.0;
    double dBX = 0.0;
    double dXX = 0.0;

    SecondOrderJet3() = default;
    // Implicit on purpose: scalar literals inside the generic kernels.
    SecondOrderJet3(double v) : value(v) {} // NOLINT(google-explicit-constructor)
    SecondOrderJet3(double v, double firstA, double firstB, double firstX, double secondAA,
                    double secondAB, double secondAX, double secondBB, double secondBX,
                    double secondXX)
        : value(v), dA(firstA), dB(firstB), dX(firstX), dAA(secondAA), dAB(secondAB), dAX(secondAX),
          dBB(secondBB), dBX(secondBX), dXX(secondXX) {}
};

inline SecondOrderJet3 operator+(const SecondOrderJet3& x, const SecondOrderJet3& y) noexcept {
    return {x.value + y.value, x.dA + y.dA,   x.dB + y.dB,   x.dX + y.dX,   x.dAA + y.dAA,
            x.dAB + y.dAB,     x.dAX + y.dAX, x.dBB + y.dBB, x.dBX + y.dBX, x.dXX + y.dXX};
}

inline SecondOrderJet3 operator-(const SecondOrderJet3& x, const SecondOrderJet3& y) noexcept {
    return {x.value - y.value, x.dA - y.dA,   x.dB - y.dB,   x.dX - y.dX,   x.dAA - y.dAA,
            x.dAB - y.dAB,     x.dAX - y.dAX, x.dBB - y.dBB, x.dBX - y.dBX, x.dXX - y.dXX};
}

inline SecondOrderJet3 operator-(const SecondOrderJet3& x) noexcept {
    return {-x.value, -x.dA, -x.dB, -x.dX, -x.dAA, -x.dAB, -x.dAX, -x.dBB, -x.dBX, -x.dXX};
}

inline SecondOrderJet3 operator*(const SecondOrderJet3& x, const SecondOrderJet3& y) noexcept {
    return {x.value * y.value,
            x.dA * y.value + x.value * y.dA,
            x.dB * y.value + x.value * y.dB,
            x.dX * y.value + x.value * y.dX,
            x.dAA * y.value + 2.0 * x.dA * y.dA + x.value * y.dAA,
            x.dAB * y.value + x.dA * y.dB + x.dB * y.dA + x.value * y.dAB,
            x.dAX * y.value + x.dA * y.dX + x.dX * y.dA + x.value * y.dAX,
            x.dBB * y.value + 2.0 * x.dB * y.dB + x.value * y.dBB,
            x.dBX * y.value + x.dB * y.dX + x.dX * y.dB + x.value * y.dBX,
            x.dXX * y.value + 2.0 * x.dX * y.dX + x.value * y.dXX};
}

inline SecondOrderJet3 operator/(const SecondOrderJet3& x, const SecondOrderJet3& y) noexcept {
    const double inv = 1.0 / y.value;
    const double w = x.value * inv;
    const double wA = (x.dA - w * y.dA) * inv;
    const double wB = (x.dB - w * y.dB) * inv;
    const double wX = (x.dX - w * y.dX) * inv;
    return {w,
            wA,
            wB,
            wX,
            (x.dAA - w * y.dAA - 2.0 * wA * y.dA) * inv,
            (x.dAB - w * y.dAB - wA * y.dB - wB * y.dA) * inv,
            (x.dAX - w * y.dAX - wA * y.dX - wX * y.dA) * inv,
            (x.dBB - w * y.dBB - 2.0 * wB * y.dB) * inv,
            (x.dBX - w * y.dBX - wB * y.dX - wX * y.dB) * inv,
            (x.dXX - w * y.dXX - 2.0 * wX * y.dX) * inv};
}

inline SecondOrderJet3& operator+=(SecondOrderJet3& x, const SecondOrderJet3& y) noexcept {
    x = x + y;
    return x;
}

inline SecondOrderJet3& operator-=(SecondOrderJet3& x, const SecondOrderJet3& y) noexcept {
    x = x - y;
    return x;
}

inline SecondOrderJet3& operator*=(SecondOrderJet3& x, const SecondOrderJet3& y) noexcept {
    x = x * y;
    return x;
}

inline SecondOrderJet3& operator/=(SecondOrderJet3& x, const SecondOrderJet3& y) noexcept {
    x = x / y;
    return x;
}

inline bool operator==(const SecondOrderJet3& x, const SecondOrderJet3& y) noexcept {
    return x.value == y.value;
}

inline bool operator!=(const SecondOrderJet3& x, const SecondOrderJet3& y) noexcept {
    return x.value != y.value;
}

inline bool operator<(const SecondOrderJet3& x, const SecondOrderJet3& y) noexcept {
    return x.value < y.value;
}

inline bool operator<=(const SecondOrderJet3& x, const SecondOrderJet3& y) noexcept {
    return x.value <= y.value;
}

inline bool operator>(const SecondOrderJet3& x, const SecondOrderJet3& y) noexcept {
    return x.value > y.value;
}

inline bool operator>=(const SecondOrderJet3& x, const SecondOrderJet3& y) noexcept {
    return x.value >= y.value;
}

/// `fabs`: sign flip on all derivative components (a.e. derivative; `+` at 0).
inline SecondOrderJet3 fabs(const SecondOrderJet3& x) noexcept {
    const double s = x.value < 0.0 ? -1.0 : 1.0;
    return {std::fabs(x.value), s * x.dA,  s * x.dB,  s * x.dX,  s * x.dAA,
            s * x.dAB,          s * x.dAX, s * x.dBB, s * x.dBX, s * x.dXX};
}

/// `max(|value|, |dA|, ..., |dXX|)`: the convergence scale used by the
/// series/CF iteration test (see `incBetaDeltaConverged`).
inline double maxAbsComponent(const SecondOrderJet3& x) noexcept {
    return std::fmax(std::fmax(std::fmax(std::fabs(x.value), std::fabs(x.dA)),
                               std::fmax(std::fabs(x.dB), std::fabs(x.dX))),
                     std::fmax(std::fmax(std::fabs(x.dAA), std::fabs(x.dAB)),
                               std::fmax(std::fmax(std::fabs(x.dAX), std::fabs(x.dBB)),
                                         std::fmax(std::fabs(x.dBX), std::fabs(x.dXX)))));
}

/// `log`: value exact, derivatives by logarithmic differentiation.
inline SecondOrderJet3 log(const SecondOrderJet3& x) noexcept {
    const double inv = 1.0 / x.value;
    const double lA = x.dA * inv;
    const double lB = x.dB * inv;
    const double lX = x.dX * inv;
    return {std::log(x.value),
            lA,
            lB,
            lX,
            x.dAA * inv - lA * lA,
            x.dAB * inv - lA * lB,
            x.dAX * inv - lA * lX,
            x.dBB * inv - lB * lB,
            x.dBX * inv - lB * lX,
            x.dXX * inv - lX * lX};
}

/// `exp`: `d^n exp = exp * (chain rule)`, same rule as `expm1`.
inline SecondOrderJet3 exp(const SecondOrderJet3& x) noexcept {
    const double e = std::exp(x.value);
    return {e,
            e * x.dA,
            e * x.dB,
            e * x.dX,
            e * (x.dAA + x.dA * x.dA),
            e * (x.dAB + x.dA * x.dB),
            e * (x.dAX + x.dA * x.dX),
            e * (x.dBB + x.dB * x.dB),
            e * (x.dBX + x.dB * x.dX),
            e * (x.dXX + x.dX * x.dX)};
}

/// `expm1`: mathematically `exp - 1`, derivatives are those of `exp`, the
/// value uses `std::expm1` for accuracy near zero.
inline SecondOrderJet3 expm1(const SecondOrderJet3& x) noexcept {
    const double e = std::exp(x.value);
    return {std::expm1(x.value),
            e * x.dA,
            e * x.dB,
            e * x.dX,
            e * (x.dAA + x.dA * x.dA),
            e * (x.dAB + x.dA * x.dB),
            e * (x.dAX + x.dA * x.dX),
            e * (x.dBB + x.dB * x.dB),
            e * (x.dBX + x.dB * x.dX),
            e * (x.dXX + x.dX * x.dX)};
}

/// `log1p`: derivatives of `log(1 + x)`.
inline SecondOrderJet3 log1p(const SecondOrderJet3& x) noexcept {
    const double inv = 1.0 / (1.0 + x.value);
    const double lA = x.dA * inv;
    const double lB = x.dB * inv;
    const double lX = x.dX * inv;
    return {std::log1p(x.value),
            lA,
            lB,
            lX,
            x.dAA * inv - lA * lA,
            x.dAB * inv - lA * lB,
            x.dAX * inv - lA * lX,
            x.dBB * inv - lB * lB,
            x.dBX * inv - lB * lX,
            x.dXX * inv - lX * lX};
}

/// `lgamma`: `d/da lgamma = psi(a)`, `d^2/da^2 = psi'(a)` (Digamma.h).
inline SecondOrderJet3 lgamma(const SecondOrderJet3& x) noexcept {
    const double psi = digammaDouble(x.value);
    const double tri = trigammaDouble(x.value);
    return {std::lgamma(x.value),
            psi * x.dA,
            psi * x.dB,
            psi * x.dX,
            tri * x.dA * x.dA + psi * x.dAA,
            tri * x.dA * x.dB + psi * x.dAB,
            tri * x.dA * x.dX + psi * x.dAX,
            tri * x.dB * x.dB + psi * x.dBB,
            tri * x.dB * x.dX + psi * x.dBX,
            tri * x.dX * x.dX + psi * x.dXX};
}

} // namespace detail
} // namespace quantape::math

#endif // QUANTAPE_MATH_DISTRIBUTIONS_DETAIL_SECOND_ORDER_JET3_H
