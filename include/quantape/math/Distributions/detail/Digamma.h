#ifndef QUANTAPE_MATH_DISTRIBUTIONS_DETAIL_DIGAMMA_H
#define QUANTAPE_MATH_DISTRIBUTIONS_DETAIL_DIGAMMA_H

/**
 * @file Digamma.h
 * @brief Digamma and trigamma (double) for the distribution parameter derivatives
 *
 * Shared by the Gamma/Beta/StudentT/NoncentralChiSquare parameter-derivative
 * packs. Implementation: upward recurrence into the asymptotic regime
 * (`x >= 12`) and Bernoulli-number series there,
 *
 *   psi(x)  ~ ln x - 1/(2x) - 1/(12x^2) + 1/(120x^4) - 1/(252x^6) + ...
 *   psi'(x) ~ 1/x + 1/(2x^2) + 1/(6x^3) - 1/(30x^5) + 1/(42x^7) - ...
 *
 * Accuracy target ~1e-14 relative for x > 0; x <= 0 returns NaN.
 * std-only, no allocation.
 */

#include <cmath>
#include <limits>

namespace quantape::math {
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

} // namespace detail
} // namespace quantape::math

#endif // QUANTAPE_MATH_DISTRIBUTIONS_DETAIL_DIGAMMA_H
