#ifndef QUANTAPE_MATH_DISTRIBUTIONS_POISSON_H
#define QUANTAPE_MATH_DISTRIBUTIONS_POISSON_H

#include "quantape/math/Distributions/detail/IncompleteGamma.h"

#include <cmath>
#include <limits>
#include <type_traits>

namespace quantape::math {
/**
 * @file Poisson.h
 * @brief Poisson law: pmf, log pmf, CDF and survival function.
 *
 * For `k = 0, 1, 2, ...`:
 *   `pmf(k) = lambda^k e^(-lambda) / k!`,
 *   `logPmf(k) = k log(lambda) - lambda - lgamma(k + 1)` (no intermediate
 *   overflow, so the upper tail stays usable),
 *   `cdf(k) = P(X <= k) = Q(k + 1, lambda)`,
 *   `ccdf(k) = P(X > k) = P(k + 1, lambda)`
 * with `P`/`Q` the regularized incomplete gamma functions (series + continued
 * fraction, `detail/IncompleteGamma.h`). `k < 0` gives `pmf = cdf = 0`,
 * `ccdf = 1`; `lambda = 0` is the point mass at 0; invalid parameters give NaN.
 *
 * `Poisson<double>` dispatches each method to the named
 * `detail::poissonPmfDouble`/`poissonLogPmfDouble`/`poissonCdfDouble`/
 * `poissonCcdfDouble` entry (the Gamma double kernels `gammaQDouble`/
 * `gammaPDouble` for the CDF/ccdf); other scalars run the generic
 * `detail::poisson...T<T>` path with ADL math.
 *
 * The double-only derivative packs `detail::poissonPmfD1D2` /
 * `detail::poissonCdfD1D2` use the analytic lambda-derivatives
 *   `d pmf/dlambda = pmf (k - lambda)/lambda`,
 *   `d^2 pmf/dlambda^2 = pmf ((k - lambda)^2 - k)/lambda^2`,
 *   `d cdf/dlambda = -pmf`,
 *   `d^2 cdf/dlambda^2 = -pmf (k - lambda)/lambda`.
 * (The second pmf derivative follows from `f''/f = ((log f)')^2 + (log f)'' =
 * ((k-lambda)^2 - k)/lambda^2`; note the `-k`, not `-lambda`.)
 */
namespace detail {

/// Poisson(lambda) log pmf, generic scalar (see `Poisson::logPmf`).
template <class T>
T poissonLogPmfT(T lambda, int k) {
    using std::lgamma;
    using std::log;
    if (k < 0) {
        return detail::negativeInfinity<T>();
    }
    if (lambda != lambda) {
        return lambda;
    }
    if (!(lambda > T(0))) {
        if (lambda == T(0)) {
            return k == 0 ? T(0) : detail::negativeInfinity<T>();
        }
        return detail::nanValue<T>();
    }
    if (k == 0) {
        return -lambda;
    }
    return T(k) * log(lambda) - lambda - lgamma(T(k) + T(1));
}

/// Poisson(lambda) pmf, generic scalar: `exp(logPmf)`.
template <class T>
T poissonPmfT(T lambda, int k) {
    using std::exp;
    return exp(poissonLogPmfT(lambda, k));
}

/// Poisson(lambda) CDF, generic scalar: `Q(k + 1, lambda)`.
template <class T>
T poissonCdfT(T lambda, int k) {
    if (k < 0) {
        return T(0);
    }
    if (lambda != lambda) {
        return lambda;
    }
    if (!(lambda > T(0))) {
        if (lambda == T(0)) {
            return T(1);
        }
        return detail::nanValue<T>();
    }
    return detail::gammaQ(T(k + 1), lambda);
}

/// Poisson(lambda) ccdf, generic scalar: `P(k + 1, lambda)`.
template <class T>
T poissonCcdfT(T lambda, int k) {
    if (k < 0) {
        return T(1);
    }
    if (lambda != lambda) {
        return lambda;
    }
    if (!(lambda > T(0))) {
        if (lambda == T(0)) {
            return T(0);
        }
        return detail::nanValue<T>();
    }
    return detail::gammaP(T(k + 1), lambda);
}

// ---------------------------------------------------------------------------
// Named double entry points (the canonical double tier).
// ---------------------------------------------------------------------------

/// Poisson(lambda) log pmf at `T = double`.
inline double poissonLogPmfDouble(double lambda, int k) {
    return poissonLogPmfT<double>(lambda, k);
}

/// Poisson(lambda) pmf at `T = double`.
inline double poissonPmfDouble(double lambda, int k) {
    return poissonPmfT<double>(lambda, k);
}

/// Poisson(lambda) CDF at `T = double`: `gammaQDouble(k + 1, lambda)`.
inline double poissonCdfDouble(double lambda, int k) {
    if (k < 0) {
        return 0.0;
    }
    if (lambda != lambda) {
        return lambda;
    }
    if (!(lambda > 0.0)) {
        if (lambda == 0.0) {
            return 1.0;
        }
        return detail::nanValue<double>();
    }
    return detail::gammaQDouble(static_cast<double>(k + 1), lambda);
}

/// Poisson(lambda) ccdf at `T = double`: `gammaPDouble(k + 1, lambda)`.
inline double poissonCcdfDouble(double lambda, int k) {
    if (k < 0) {
        return 1.0;
    }
    if (lambda != lambda) {
        return lambda;
    }
    if (!(lambda > 0.0)) {
        if (lambda == 0.0) {
            return 0.0;
        }
        return detail::nanValue<double>();
    }
    return detail::gammaPDouble(static_cast<double>(k + 1), lambda);
}

} // namespace detail

/**
 * @brief Poisson(lambda) with pmf/log pmf/CDF/ccdf.
 *
 * `DoubleT = double` dispatches to the named `detail::poisson*Double` entries;
 * other scalars run the generic `detail::poisson*T` kernels (ADL math).
 */
template <class T>
struct Poisson {
    T lambda;

    /// Probability mass; underflows to 0 far in the tail without overflow.
    T pmf(int k) const {
        if constexpr (std::is_same_v<T, double>) {
            return detail::poissonPmfDouble(lambda, k);
        } else {
            return detail::poissonPmfT(lambda, k);
        }
    }

    T logPmf(int k) const {
        if constexpr (std::is_same_v<T, double>) {
            return detail::poissonLogPmfDouble(lambda, k);
        } else {
            return detail::poissonLogPmfT(lambda, k);
        }
    }

    /// `P(X <= k) = Q(k + 1, lambda)`; exactly 0 for `k < 0`.
    T cdf(int k) const {
        if constexpr (std::is_same_v<T, double>) {
            return detail::poissonCdfDouble(lambda, k);
        } else {
            return detail::poissonCdfT(lambda, k);
        }
    }

    /// `P(X > k) = P(k + 1, lambda)`; exactly 1 for `k < 0`.
    T ccdf(int k) const {
        if constexpr (std::is_same_v<T, double>) {
            return detail::poissonCcdfDouble(lambda, k);
        } else {
            return detail::poissonCcdfT(lambda, k);
        }
    }
};

namespace detail {
/**
 * @brief `{pmf, d pmf/dlambda, d^2 pmf/dlambda^2}` for Poisson(lambda).
 *
 * All lambda-derivatives analytic. `k < 0` is the zero pack; `lambda = 0` is
 * the point mass (its lambda-derivatives are discontinuous, reported as NaN);
 * `lambda < 0` or NaN gives a NaN pack.
 */
inline ValueD1D2 poissonPmfD1D2(double lambda, int k) {
    using std::exp;
    using std::lgamma;
    using std::log;
    const double nan = std::numeric_limits<double>::quiet_NaN();
    if (k < 0) {
        return ValueD1D2{0.0, 0.0, 0.0};
    }
    if (lambda != lambda || lambda < 0.0) {
        return ValueD1D2{nan, nan, nan};
    }
    if (lambda == 0.0) {
        return ValueD1D2{k == 0 ? 1.0 : 0.0, nan, nan};
    }
    const double logPmf = (k == 0) ? -lambda : k * log(lambda) - lambda - lgamma(k + 1.0);
    const double pmf = exp(logPmf);
    const double dk = static_cast<double>(k) - lambda;
    ValueD1D2 out{};
    out.value = pmf;
    out.d1 = pmf * dk / lambda;
    out.d2 = pmf * (dk * dk - static_cast<double>(k)) / (lambda * lambda);
    return out;
}

/**
 * @brief `{cdf, d cdf/dlambda, d^2 cdf/dlambda^2}` for Poisson(lambda).
 *
 * `cdf = Q(k + 1, lambda)`, `d cdf/dlambda = -pmf(k)`,
 * `d^2 cdf/dlambda^2 = -pmf(k) (k - lambda)/lambda`; the first derivative is
 * just minus the analytic pmf, so only pmf evaluation is shared.
 */
inline ValueD1D2 poissonCdfD1D2(double lambda, int k) {
    const double nan = std::numeric_limits<double>::quiet_NaN();
    if (k < 0) {
        return ValueD1D2{0.0, 0.0, 0.0};
    }
    if (lambda != lambda || lambda < 0.0) {
        return ValueD1D2{nan, nan, nan};
    }
    if (lambda == 0.0) {
        // Point mass: cdf jumps from 1 (k >= 0) to 0 (k < 0); the two-sided
        // lambda-derivatives at 0 are the k >= 0 one-sided limits.
        return ValueD1D2{1.0, k == 0 ? -1.0 : 0.0, k == 0 ? 1.0 : 0.0};
    }
    const double pmf = poissonPmfD1D2(lambda, k).value;
    ValueD1D2 out{};
    out.value = gammaQDouble(static_cast<double>(k + 1), lambda);
    out.d1 = -pmf;
    out.d2 = -pmf * (static_cast<double>(k) - lambda) / lambda;
    return out;
}

} // namespace detail
} // namespace quantape::math

#endif // QUANTAPE_MATH_DISTRIBUTIONS_POISSON_H
