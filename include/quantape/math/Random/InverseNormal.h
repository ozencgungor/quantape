#ifndef QUANTAPE_MATH_RANDOM_INVERSENORMAL_H
#define QUANTAPE_MATH_RANDOM_INVERSENORMAL_H

#include <cmath>

namespace quantape::math {
/**
 * @file InverseNormal.h
 * @brief Acklam's inverse normal CDF (Stan-free, standard library only).
 *
 * Maximum relative error ~1.15e-9 over (0,1); adequate for MC/QMC path
 * generation (the transform error is deterministic and far below sampling
 * error). Endpoints are clamped to +-1e10: callers should map uniforms to
 * the open interval, e.g. p = (bits + 0.5) / 2^64.
 */
inline double inverseNormal(double p) {
    static const double a[6] = {-3.969683028665376e+01, 2.209460984245205e+02,
                                -2.759285104469687e+02, 1.383577518672690e+02,
                                -3.066479806614716e+01, 2.506628277459239e+00};
    static const double b[5] = {-5.447609879822406e+01, 1.615858368580409e+02,
                                -1.556989798598866e+02, 6.680131188771972e+01,
                                -1.328068155288572e+01};
    static const double c[6] = {-7.784894002430293e-03, -3.223964580411365e-01,
                                -2.400758277161838e+00, -2.549732539343734e+00,
                                4.374664141464968e+00, 2.938163982698783e+00};
    static const double d[4] = {7.784695709041462e-03, 3.224671290700398e-01,
                                2.445134137142996e+00, 3.754408661907416e+00};
    const double pl = 0.02425;
    if (p <= 0.0) {
        return -1e10;
    }
    if (p >= 1.0) {
        return 1e10;
    }
    if (p < pl) {
        const double q = std::sqrt(-2.0 * std::log(p));
        return (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) /
               ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0);
    }
    if (p > 1.0 - pl) {
        const double q = std::sqrt(-2.0 * std::log(1.0 - p));
        return -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) /
               ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0);
    }
    const double q = p - 0.5;
    const double r = q * q;
    return (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q /
           (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1.0);
}

} // namespace quantape::math
#endif // QUANTAPE_MATH_RANDOM_INVERSENORMAL_H
