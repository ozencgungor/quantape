#ifndef QUANTAPE_MATH_DISTRIBUTIONS_BIVARIATE_NORMAL_H
#define QUANTAPE_MATH_DISTRIBUTIONS_BIVARIATE_NORMAL_H

#include "quantape/math/Distributions/detail/BivariateNormalKernel.h"

namespace quantape::math {
/**
 * @file BivariateNormal.h
 * @brief Standard bivariate normal distribution (CDF and PDF).
 *
 * Phi2(a,b;rho) = P(X <= a, Y <= b) for a standard bivariate normal pair with
 * correlation rho. The CDF is evaluated with the fixed-rule Genz (2004)
 * / Drezner-Wesolowsky (1989) approximation: a global Gauss-Legendre
 * quadrature of the Plackett integral for |rho| < 0.925 and the published
 * near-degenerate expansion for |rho| >= 0.925 (see
 * detail/BivariateNormalKernel.h for the algorithm and exact branches).
 *
 * Accuracy (verified against mpmath references at 50 digits): maximum
 * absolute error ~1e-16 for |rho| < 0.925 and <= 1.4e-14 for |rho| >= 0.925,
 * including |rho| = 1 - 1e-15. Exact limits at rho = 0, +-1 and at
 * a,b = +-inf; NaN inputs and |rho| > 1 produce NaN. `pdf` returns the
 * bivariate normal density; at |rho| = 1 the density is degenerate and the
 * expression propagates NaN/Inf.
 *
 * Scalar-generic: instantiate with `double` or an AD scalar exposing
 * arithmetic and ADL exp/erfc/isnan/isinf overloads. No heap, no exceptions.
 */
template <class T>
struct BivariateNormal {
    /// Correlation coefficient, |rho| <= 1.
    T rho;

    /// Phi2(a,b;rho), the joint probability P(X <= a, Y <= b).
    T cdf(T a, T b) const { return detail::bvnCdf(a, b, rho); }

    /// Bivariate normal density at (a,b):
    /// exp(-(a^2 - 2 rho a b + b^2)/(2(1-rho^2))) / (2 pi sqrt(1-rho^2)).
    T pdf(T a, T b) const { return detail::bvnPdf(a, b, rho); }
};

} // namespace quantape::math

#endif // QUANTAPE_MATH_DISTRIBUTIONS_BIVARIATE_NORMAL_H
