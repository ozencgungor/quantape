#ifndef QUANTAPE_MATH_DISTRIBUTIONS_TRIVARIATE_NORMAL_H
#define QUANTAPE_MATH_DISTRIBUTIONS_TRIVARIATE_NORMAL_H

#include "quantape/math/Distributions/detail/TrivariateNormalKernel.h"

namespace quantape::math {

/**
 * @file TrivariateNormal.h
 * @brief Standard trivariate normal distribution (CDF and analytic
 * derivative packs).
 *
 * Phi3(h1,h2,h3; r12,r13,r23) = P(X1 <= h1, X2 <= h2, X3 <= h3) for standard
 * normal margins.  The CDF follows the published Genz (2004) reduction on top
 * of the in-house bivariate kernel: after sorting the largest-magnitude
 * correlation into r23, the Plackett-style path integral is evaluated with a
 * deterministic composite Gauss-Kronrod 15-point rule (refinement until the
 * embedded error estimate stabilizes), with exact degenerate limits for
 * r = 0, |r| = 1, infinities and the orthant closed form at h = 0.  See
 * `detail/TrivariateNormalKernel.h` for the algorithm and branches.
 *
 * Accuracy against high-precision mpmath references: < 1e-14 for
 * well-conditioned correlation triples, limited by the 1.4e-14
 * near-degenerate bivariate ceiling for |conditional rho| >= 0.925; graceful
 * degradation for near-singular triples.  NaN inputs, |r_ij| > 1 and
 * inconsistent (non-positive-definite) correlations produce NaN; the value is
 * clamped to [0,1].
 *
 * `cdfGradHess` returns the value plus the exact analytic first derivatives
 * g[6] = dPhi3/d(h1,h2,h3,r12,r13,r23) and the full symmetric Hessian h[21]
 * (lower-triangle row-major, see `detail::tvnHessIndex`); `cdfGrad` is the
 * first-order-only form.  The derivative packs require finite h, |r_ij| < 1
 * and a positive-definite correlation triple and return all-NaN otherwise.
 * No heap, no exceptions, no finite differences.
 */
struct TrivariateNormal {
    /// Correlations r12, r13, r23 (symmetric matrix with unit diagonal).
    double r12 = 0.0;
    double r13 = 0.0;
    double r23 = 0.0;

    /// Phi3, the joint probability P(X1 <= h1, X2 <= h2, X3 <= h3).
    double cdf(double h1, double h2, double h3) const {
        return detail::tvnCdf(h1, h2, h3, r12, r13, r23);
    }

    /// Value plus analytic gradient.
    detail::TvnCdfGrad cdfGrad(double h1, double h2, double h3) const {
        return detail::tvnCdfGrad(h1, h2, h3, r12, r13, r23);
    }

    /// Value, analytic gradient and full symmetric Hessian.
    detail::TvnGradHess cdfGradHess(double h1, double h2, double h3) const {
        return detail::tvnCdfGradHess(h1, h2, h3, r12, r13, r23);
    }
};

} // namespace quantape::math

#endif // QUANTAPE_MATH_DISTRIBUTIONS_TRIVARIATE_NORMAL_H
