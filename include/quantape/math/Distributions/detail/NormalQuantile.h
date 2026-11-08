#ifndef QUANTAPE_MATH_DISTRIBUTIONS_DETAIL_NORMALQUANTILE_H
#define QUANTAPE_MATH_DISTRIBUTIONS_DETAIL_NORMALQUANTILE_H

/**
 * @file NormalQuantile.h
 * @brief Shared AS241 standard normal quantile (double)
 *
 * Wichura's AS241 rational approximations (1988); the coefficient sets are the
 * ones Stan Math ships (BSD-3-Clause; see NOTICE). The `p >= 0.9999` branch
 * evaluates on the complementary probability `(2e9 - 2e9 p) / 2e9` so the
 * rational is fed a well-conditioned argument in the far upper tail.
 *
 * Accuracy ~1e-15 in the central region, degrading gracefully toward the
 * tails. The lower-tail rational is fitted in `r = sqrt(-log p)` from 5 up to
 * ~27.5, which covers every positive double: the smallest subnormal has
 * `r = sqrt(-log 5e-324) = 27.28`. `normalInvDouble` therefore evaluates the
 * finite quantile for every representable `0 < p < 1`, including the whole
 * subnormal range (e.g. `-38.4674` at `p = 5e-324` and `-37.6725` at
 * `p = 7e-311`, each to ~1 ulp); only the degenerate `p = 0` / `p = 1`
 * endpoints map to `-/+infinity`, and NaN propagates.
 *
 * `scalarValue` is the matching off-tape view used by the quantile solvers:
 * the double solve runs on these values and the caller's final Newton step in
 * the scalar type re-introduces the taping.
 *
 * Shared by `Normal<T>` (portfolio quantile) and `Random/InverseNormal.h`
 * (path-generation transform). std-only, no allocation.
 */

#include <cmath>
#include <limits>
#include <type_traits>

namespace quantape::math {
namespace detail {

/**
 * AS241 rational approximation for the standard normal quantile (Wichura 1988).
 * Coefficient sets are the ones Stan Math ships (BSD-3-Clause; see NOTICE).
 * @param p probability in (0,1)
 * @return standard normal quantile
 */
inline double as241(double p) {
    static constexpr double a[8] = {3.3871328727963666080e+00, 1.3314166789178437745e+02,
                                    1.9715909503065514427e+03, 1.3731693765509461125e+04,
                                    4.5921953931549871457e+04, 6.7265770927008700853e+04,
                                    3.3430575583588128105e+04, 2.5090809287301226727e+03};
    static constexpr double b[7] = {4.2313330701600911252e+01, 6.8718700749205790830e+02,
                                    5.3941960214247511077e+03, 2.1213794301586595867e+04,
                                    3.9307895800092710610e+04, 2.8729085735721942674e+04,
                                    5.2264952788528545610e+03};
    static constexpr double c[8] = {1.42343711074968357734e+00, 4.63033784615654529590e+00,
                                    5.76949722146069140550e+00, 3.64784832476320460504e+00,
                                    1.27045825245236838258e+00, 2.41780725177450611770e-01,
                                    2.27238449892691845833e-02, 7.74545014278341407640e-04};
    static constexpr double d[7] = {2.05319162663775882187e+00, 1.67638483018380384940e+00,
                                    6.89767334985100004550e-01, 1.48103976427480074590e-01,
                                    1.51986665636164571966e-02, 5.47593808499534494600e-04,
                                    1.05075007164441684324e-09};
    static constexpr double e[8] = {6.65790464350110377720e+00, 5.46378491116411436990e+00,
                                    1.78482653991729133580e+00, 2.96560571828504891230e-01,
                                    2.65321895265761230930e-02, 1.24266094738807843860e-03,
                                    2.71155556874348757815e-05, 2.01033439929228813265e-07};
    static constexpr double f[7] = {5.99832206555887937690e-01, 1.36929880922735805310e-01,
                                    1.48753612908506148525e-02, 7.86869131145613259100e-04,
                                    1.84631831751005468180e-05, 1.42151175831644588870e-07,
                                    2.04426310338993978564e-15};

    const double q = p - 0.5;
    if (std::fabs(q) <= 0.425) {
        const double r = 0.180625 - q * q;
        return q *
               (((((((a[7] * r + a[6]) * r + a[5]) * r + a[4]) * r + a[3]) * r + a[2]) * r + a[1]) *
                    r +
                a[0]) /
               (((((((b[6] * r + b[5]) * r + b[4]) * r + b[3]) * r + b[2]) * r + b[1]) * r + b[0]) *
                    r +
                1.0);
    }
    double r = q < 0.0 ? p : 1.0 - p;
    if (r <= 0.0) {
        return 0.0;
    }
    r = std::sqrt(-std::log(r));
    double val = 0.0;
    if (r <= 5.0) {
        r -= 1.6;
        val = (((((((c[7] * r + c[6]) * r + c[5]) * r + c[4]) * r + c[3]) * r + c[2]) * r + c[1]) *
                   r +
               c[0]) /
              (((((((d[6] * r + d[5]) * r + d[4]) * r + d[3]) * r + d[2]) * r + d[1]) * r + d[0]) *
                   r +
               1.0);
    } else {
        r -= 5.0;
        val = (((((((e[7] * r + e[6]) * r + e[5]) * r + e[4]) * r + e[3]) * r + e[2]) * r + e[1]) *
                   r +
               e[0]) /
              (((((((f[6] * r + f[5]) * r + f[4]) * r + f[3]) * r + f[2]) * r + f[1]) * r + f[0]) *
                   r +
               1.0);
    }
    return q < 0.0 ? -val : val;
}

/**
 * @brief Double value of a scalar: identity for floating point, `.val()` for
 * AD scalars, `.value` for the `SecondOrderJet` kernels.
 *
 * This is the off-tape view used as the seed of the quantile solvers (the
 * double solve must not see the tape); the caller then performs its final
 * Newton steps in the original scalar type so the gradient flows. Defining it
 * once here keeps every distribution header taped through the same helper.
 */
template <class T>
inline double scalarValue(const T& x) {
    if constexpr (std::is_floating_point_v<T>) {
        return static_cast<double>(x);
    } else if constexpr (requires { x.val(); }) {
        return scalarValue(x.val());
    } else {
        return static_cast<double>(x.value);
    }
}

/// standard normal quantile (double); boundaries -> +/-inf, invalid -> NaN.
inline double normalInvDouble(double p) {
    if (std::isnan(p)) {
        return p;
    }
    if (p < 0.0 || p > 1.0) {
        return std::numeric_limits<double>::quiet_NaN();
    }
    if (p == 0.0) {
        return -std::numeric_limits<double>::infinity();
    }
    if (p == 1.0) {
        return std::numeric_limits<double>::infinity();
    }
    // Upper-tail evaluation on the complementary probability (Stan trick).
    constexpr double kBigInt = 2.0e9;
    if (p >= 0.9999) {
        return -as241((kBigInt - kBigInt * p) / kBigInt);
    }
    return as241(p);
}

} // namespace detail
} // namespace quantape::math

#endif // QUANTAPE_MATH_DISTRIBUTIONS_DETAIL_NORMALQUANTILE_H
