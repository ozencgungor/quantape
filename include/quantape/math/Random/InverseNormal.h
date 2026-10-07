#ifndef QUANTAPE_MATH_RANDOM_INVERSENORMAL_H
#define QUANTAPE_MATH_RANDOM_INVERSENORMAL_H

#include "quantape/math/Distributions/detail/NormalQuantile.h"

namespace quantape::math {
/**
 * @file InverseNormal.h
 * @brief Inverse normal CDF backed by the shared AS241 kernel (std-only).
 *
 * Switch from Acklam's rational (maximum relative error ~1.15e-9) to Wichura's
 * AS241 rational (`detail::normalInvDouble`, ~1e-15 central, degrading
 * gracefully in the tails), the same kernel `Normal<T>::inv` uses. The
 * quantile is deterministic through the seed and far more accurate than MC/QMC
 * sampling error. Endpoints are clamped to +-1e10 as before: callers should
 * map uniforms to the open interval, e.g. p = (bits + 0.5) / 2^64.
 */
inline double inverseNormal(double p) {
    if (p <= 0.0) {
        return -1e10;
    }
    if (p >= 1.0) {
        return 1e10;
    }
    return detail::normalInvDouble(p);
}

} // namespace quantape::math
#endif // QUANTAPE_MATH_RANDOM_INVERSENORMAL_H
