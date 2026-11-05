#ifndef LIGHTAPE_BASE_DIGAMMA_H
#define LIGHTAPE_BASE_DIGAMMA_H

/**
 * @file Digamma.h
 * @brief Digamma and polygamma (double) for the lightape AD engine
 *
 * In-house port of quantape::math::detail::digammaDouble/trigammaDouble
 * (same owner). Upward recurrence into the asymptotic regime (`x >= 12`) and
 * Bernoulli-number series there,
 *
 *   psi(x)  ~ ln x - 1/(2x) - 1/(12x^2) + 1/(120x^4) - 1/(252x^6) + ...
 *   psi'(x) ~ 1/x + 1/(2x^2) + 1/(6x^3) - 1/(30x^5) + 1/(42x^7) - ...
 *
 * The same scheme gives every higher order through the general series
 *
 *   psi^(n)(x) ~ (-1)^(n+1) [ (n-1)!/x^n + n!/(2 x^(n+1))
 *                             + sum_{k>=1} B_2k (2k+n-1)! / ((2k)! x^(2k+n)) ],
 *
 * with the recurrence psi^(n)(x) = psi^(n)(x+1) - (-1)^n n! / x^(n+1); the
 * `polygamma` entry point exposes all orders (`digamma` is n = 0,
 * `trigamma` n = 1, `tetragamma` n = 2, `pentagamma` n = 3).
 *
 * Accuracy target <=1e-12 relative for x > 0; x <= 0 returns NaN. On a dense
 * 0.005..1e4 mpmath-gated grid orders 1..4 stay within 8e-16 relative; the
 * digamma relative error peaks at 7.8e-13 right at its zero (x ~ 1.46), where
 * the absolute error is still ~1e-16. The first dropped Bernoulli term at the
 * x = 12 threshold is below 1e-17 relative for n <= 4, so double rounding
 * (~1e-16) dominates there; the recurrence is the only other error source.
 * std-only, no allocation.
 */

#include <cmath>
#include <limits>

namespace lightape {
namespace detail {

/// Digamma psi(x) for x > 0 (NaN otherwise).
inline double digammaDouble(double x) {
    if (!(x > 0.0) || std::isnan(x)) {
        return std::numeric_limits<double>::quiet_NaN();
    }
    double result = 0.0;
    while (x < 12.0) {
        result -= 1.0 / x;
        x += 1.0;
    }
    const double inv = 1.0 / x;
    const double inv2 = inv * inv;
    // -1/(12x^2) + 1/(120x^4) - 1/(252x^6) + 1/(240x^8) - 1/(132x^10)
    const double series =
        1.0 / 12.0 -
        inv2 * (1.0 / 120.0 - inv2 * (1.0 / 252.0 - inv2 * (1.0 / 240.0 - inv2 * (1.0 / 132.0))));
    return result + std::log(x) - 0.5 * inv - inv2 * series;
}

/// Trigamma psi'(x) for x > 0 (NaN otherwise).
inline double trigammaDouble(double x) {
    if (!(x > 0.0) || std::isnan(x)) {
        return std::numeric_limits<double>::quiet_NaN();
    }
    double result = 0.0;
    while (x < 12.0) {
        result += 1.0 / (x * x);
        x += 1.0;
    }
    const double inv = 1.0 / x;
    const double inv2 = inv * inv;
    // 1/(2x^2) + 1/(6x^3) - 1/(30x^5) + 1/(42x^7) - 1/(30x^9) + 5/(66x^11)
    const double tail =
        1.0 / 6.0 +
        inv2 * (-1.0 / 30.0 + inv2 * (1.0 / 42.0 + inv2 * (-1.0 / 30.0 + inv2 * (5.0 / 66.0))));
    return result + inv * (1.0 + inv * (0.5 + inv * tail));
}

/// Bernoulli numbers B_2, B_4, ..., B_20 (B_2k for k = 1..10).
inline constexpr double kBernoulli2To20[10] = {
    1.0 / 6.0,       -1.0 / 30.0, 1.0 / 42.0,      -1.0 / 30.0,     5.0 / 66.0,
    -691.0 / 2730.0, 7.0 / 6.0,   -3617.0 / 510.0, 43867.0 / 798.0, -174611.0 / 330.0,
};

/// Polygamma psi^(n)(x) for n >= 0, x > 0 (NaN otherwise).
///
/// Upward recurrence into `x >= 12` then the Bernoulli series above with
/// B_2..B_20. At the threshold the first dropped term is below 1e-17 relative
/// for n <= 4; the recurrence (one `pow` per step) and the final double
/// rounding are the accuracy limit. `n = 0` defers to `digammaDouble`, whose
/// leading ln x term replaces the (n-1)! term of the general series.
inline double polygammaDouble(int n, double x) {
    if (n < 0 || !(x > 0.0) || std::isnan(x)) {
        return std::numeric_limits<double>::quiet_NaN();
    }
    if (n == 0) {
        return digammaDouble(x);
    }
    double factorial = 1.0; // n!
    for (int i = 2; i <= n; ++i) {
        factorial *= static_cast<double>(i);
    }
    const bool even = (n % 2 == 0);
    // psi^(n)(x) = psi^(n)(x+1) - (-1)^n n!/x^(n+1).
    double result = 0.0;
    while (x < 12.0) {
        const double p = factorial / std::pow(x, n + 1);
        result += even ? -p : p;
        x += 1.0;
    }
    double prevFactorial = 1.0; // (n-1)!
    for (int i = 2; i < n; ++i) {
        prevFactorial *= static_cast<double>(i);
    }
    const double inv = 1.0 / x;
    const double inv2 = inv * inv;
    // ratio_k = (2k+n-1)!/(2k)!, ratio_0 = (n-1)!, grown by the adjacent
    // terms so no factorial is ever formed explicitly.
    double ratio = prevFactorial;
    double invPow = 1.0; // inv^(2k)
    double tail = 0.0;
    for (int k = 1; k <= 10; ++k) {
        const double twoK = 2.0 * static_cast<double>(k);
        ratio *= (twoK + static_cast<double>(n) - 1.0) * (twoK + static_cast<double>(n) - 2.0) /
                 (twoK * (twoK - 1.0));
        invPow *= inv2;
        tail += kBernoulli2To20[k - 1] * ratio * invPow;
    }
    double invN = 1.0; // inv^n
    for (int i = 0; i < n; ++i) {
        invN *= inv;
    }
    const double magnitude = invN * (prevFactorial + 0.5 * factorial * inv + tail);
    return even ? result - magnitude : result + magnitude;
}

/// Tetragamma psi''(x) for x > 0 (NaN otherwise).
inline double tetragammaDouble(double x) {
    return polygammaDouble(2, x);
}

/// Pentagamma psi'''(x) for x > 0 (NaN otherwise).
inline double pentagammaDouble(double x) {
    return polygammaDouble(3, x);
}

/// Fourth derivative psi''''(x) for x > 0 (NaN otherwise).
inline double psi4Double(double x) {
    return polygammaDouble(4, x);
}

} // namespace detail

/// Digamma psi(x) for x > 0 (NaN otherwise).
inline double digamma(double x) {
    return detail::digammaDouble(x);
}

/// Trigamma psi'(x) for x > 0 (NaN otherwise).
inline double trigamma(double x) {
    return detail::trigammaDouble(x);
}

/// Tetragamma psi''(x) for x > 0 (NaN otherwise).
inline double tetragamma(double x) {
    return detail::tetragammaDouble(x);
}

/// Pentagamma psi'''(x) for x > 0 (NaN otherwise).
inline double pentagamma(double x) {
    return detail::pentagammaDouble(x);
}

/// Polygamma psi^(n)(x) for n >= 0, x > 0 (NaN otherwise).
inline double polygamma(int n, double x) {
    return detail::polygammaDouble(n, x);
}

} // namespace lightape

#endif // LIGHTAPE_BASE_DIGAMMA_H
