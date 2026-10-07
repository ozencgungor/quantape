#ifndef QUANTAPE_MATH_DISTRIBUTIONS_H
#define QUANTAPE_MATH_DISTRIBUTIONS_H

/**
 * @file Distributions.h
 * @brief Convenience header for the distribution types (Stan-free)
 *
 * Finance-first probability distributions with analytic derivative packs for
 * the common cases. Each type follows the Black76 pattern:
 *
 *   `quantape::math::Normal<DoubleT> n{mu, sigma};`
 *   `double c = n.cdf(x);              // DoubleT = double`
 *   `var    v = n.cdf(x);              // generic scalar path`
 *
 * Analysis kernels (double), the `detail::*D1D2`/`*GradHess` packs and the
 * generic scalar path live in each header; the headers are self-contained and
 * std-only (plus quantape/util and the in-house Random utilities).
 *
 * See `internal_docs/math_distributions_plan.md` for the coverage/derivative
 * tiers and accuracy gates.
 */

#include "Distributions/Beta.h"
#include "Distributions/BivariateNormal.h"
#include "Distributions/ChiSquare.h"
#include "Distributions/Exponential.h"
#include "Distributions/Gamma.h"
#include "Distributions/InverseGaussian.h"
#include "Distributions/Lognormal.h"
#include "Distributions/NoncentralBeta.h"
#include "Distributions/NoncentralChiSquare.h"
#include "Distributions/Normal.h"
#include "Distributions/Poisson.h"
#include "Distributions/StudentT.h"
#include "Distributions/TrivariateNormal.h"

#endif // QUANTAPE_MATH_DISTRIBUTIONS_H
