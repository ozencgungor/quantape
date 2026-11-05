#ifndef LIGHTAPE_BASE_NONCENTRAL_CHI_SQUARE_H
#define LIGHTAPE_BASE_NONCENTRAL_CHI_SQUARE_H

/**
 * @file NoncentralChiSquare.h
 * @brief Noncentral chi-square CDF / CCDF / density (double kernels only)
 *
 * First-party port of the double-precision value kernels from
 * `quantape/math/Distributions/NoncentralChiSquare.h` (same owner; evaluation
 * order preserved). The quantape derivative packs (Stan callbacks, GradHess
 * machinery) are deliberately not ported here; the lightape `External` factors
 * for them live in `lightape/ops/NoncentralChiSquare.h`, built on these
 * kernels.
 *
 * Kernels:
 *   * `noncentralChiSquareCdf(nu, lambda, x)`:
 *     `F(x;nu,lambda) = sum_j w_j P(chi^2_{nu+2j} <= x)` with log weights
 *     `log w_j = -lambda/2 + j log(lambda/2) - lgamma(j+1)`. The sum runs
 *     outwards from the mode `j0 = floor(lambda/2)` with weights normalised to
 *     the mode (relative weights), seeded by one `gammaP` and advanced with
 *     the central-CDF recurrence `P(a_{j+1}) = P(a_j) - d_j`,
 *     `d_j = y^{a_j} e^{-y}/Gamma(a_j+1)`, `w_{j+1} = w_j (lambda/2)/(j+1)`;
 *     terms below `w0 * 1e-18` relative weight are dropped. `lambda = 0`
 *     reduces to `gammaP(nu/2, x/2)`.
 *   * `noncentralChiSquareCcdf(nu, lambda, x)`: the same mixture with
 *     `Q(a_{j+1}) = Q(a_j) + d_j` and `gammaQ` seeding; `Ccdf = 1 - Cdf`.
 *   * `noncentralChiSquarePdf(nu, lambda, x)`:
 *     `f = 1/2 e^{-(x+lambda)/2} (x/lambda)^{nu/4-1/2} I_{nu/2-1}(sqrt(lambda x))`
 *     in log space via the ported `logBesselI`; `x = 0` limits and `lambda = 0`
 *     (central density) handled exactly. NaN/Inf propagate.
 *
 * The Bessel-I helpers are a double-only port of the subset of
 * `quantape/math/Distributions/detail/BesselI.h` that the density needs
 * (series, asymptotic expansion, log form, signed ratio); the order-derivative
 * machinery of that header is not ported. std-only, no allocation.
 */

#include <cmath>
#include <limits>

#include "lightape/base/IncompleteGamma.h"

namespace lightape {
namespace detail {

// ---------------------------------------------------------------------------
// Modified Bessel function of the first kind, real order (double subset).
// Port of quantape::math::detail::bessel_detail (same owner); names are
// prefixed to avoid clashing with a future shared base/BesselI.h.
// ---------------------------------------------------------------------------

namespace nccs_bessel_detail {

/// 2 pi (avoids M_PI, which is not in the C++ standard).
inline constexpr double kTwoPi = 6.283185307179586476925286766559005768;

/// Series evaluation of `I_nu(z)`: sum of the `(z/2)^nu`-expanded series.
inline double seriesSum(double nu, double z) {
    using std::exp;
    using std::fabs;
    using std::lgamma;
    using std::log;
    using std::tgamma;
    const double halfZ = z / 2.0;
    // lgamma returns log|Gamma|: recover the sign, which flips I_nu for
    // nu in (-2,-1), (-4,-3), ...
    const double gammaSign = tgamma(nu + 1.0) < 0.0 ? -1.0 : 1.0;
    double term = gammaSign * exp(nu * log(halfZ) - lgamma(nu + 1.0));
    double sum = term;
    const double z2 = halfZ * halfZ;
    for (int m = 1; m <= 100000; ++m) {
        term *= z2 / (static_cast<double>(m) * (static_cast<double>(m) + nu));
        sum += term;
        if (fabs(term) <= fabs(sum) * 1e-17) {
            break;
        }
    }
    return sum;
}

/// Multiplicative asymptotic correction series for `I_nu(z)` at large z:
/// `1 + sum_{m>=1} (-1)^m (mu-1)(mu-9)...(mu-(2m-1)^2) / (m! (8z)^m)`,
/// truncated at the smallest term (the series diverges past its minimum).
inline double asymptoticCorrection(double nu, double z) {
    using std::fabs;
    const double mu = 4.0 * nu * nu;
    double term = 1.0;
    double sum = 1.0;
    for (int m = 1; m <= 100; ++m) {
        const double odd = static_cast<double>(2 * m - 1);
        const double next = term * (-(mu - odd * odd) / (static_cast<double>(m) * 8.0 * z));
        if (fabs(next) >= fabs(term)) {
            break;
        }
        term = next;
        sum += term;
        if (fabs(term) <= fabs(sum) * 1e-17) {
            break;
        }
    }
    return sum;
}

/// True when the asymptotic expansion is accurate enough: past the series
/// crossover and with a small first correction (`z > nu^2/2`).
inline bool useAsymptotic(double nu, double z) {
    return z >= 15.0 + 0.5 * nu && z >= 0.5 * nu * nu;
}

} // namespace nccs_bessel_detail

/// `I_nu(z)` for `z > 0`; negative integer orders are reflected with
/// `I_{-k} = I_k` (other real orders by analytic continuation).
inline double nccsBesselI(double nu, double z) {
    using std::exp;
    using std::sqrt;
    if (nu < 0.0 && nu == std::floor(nu)) {
        return nccsBesselI(-nu, z);
    }
    if (nccs_bessel_detail::useAsymptotic(nu, z)) {
        return exp(z) / sqrt(nccs_bessel_detail::kTwoPi * z) *
               nccs_bessel_detail::asymptoticCorrection(nu, z);
    }
    return nccs_bessel_detail::seriesSum(nu, z);
}

/// `log(I_nu(z))` in log space (no overflow for large z); meaningful where
/// `I_nu > 0` (all `nu > -1`).
inline double nccsLogBesselI(double nu, double z) {
    using std::log;
    if (nu < 0.0 && nu == std::floor(nu)) {
        return nccsLogBesselI(-nu, z);
    }
    if (nccs_bessel_detail::useAsymptotic(nu, z)) {
        return z - 0.5 * log(nccs_bessel_detail::kTwoPi * z) +
               log(nccs_bessel_detail::asymptoticCorrection(nu, z));
    }
    return log(nccs_bessel_detail::seriesSum(nu, z));
}

/// Signed ratio `I_num(z) / I_den(z)` (`z > 0`): direct division while both
/// values are finite, log-magnitude ratio when either overflows.
inline double nccsBesselIRatio(double num, double den, double z) {
    using std::exp;
    const double numerator = nccsBesselI(num, z);
    const double denominator = nccsBesselI(den, z);
    if (denominator != 0.0 && std::isfinite(numerator) && std::isfinite(denominator)) {
        const double quotient = numerator / denominator;
        if (std::isfinite(quotient)) {
            return quotient;
        }
    }
    return exp(nccsLogBesselI(num, z) - nccsLogBesselI(den, z));
}

} // namespace detail

/// Noncentral chi-square CDF (Poisson mixture with the central-CDF recurrence).
inline double noncentralChiSquareCdf(double nu, double lambda, double x) {
    if (!(nu > 0.0) || !(lambda >= 0.0) || std::isnan(x)) {
        return std::numeric_limits<double>::quiet_NaN();
    }
    if (x <= 0.0) {
        return 0.0;
    }
    if (std::isinf(x)) {
        return 1.0;
    }
    if (lambda == 0.0) {
        return detail::gammaPDouble(0.5 * nu, 0.5 * x);
    }
    const double halfL = 0.5 * lambda;
    const double y = 0.5 * x;
    const double a0 = 0.5 * nu;
    const long j0 = static_cast<long>(std::floor(halfL));
    const double lw0 = -halfL + static_cast<double>(j0) * std::log(halfL) -
                       std::lgamma(static_cast<double>(j0) + 1.0);
    const double w0 = std::exp(lw0);
    const double P0 = detail::gammaPDouble(a0 + static_cast<double>(j0), y);
    const double logD0 = (a0 + static_cast<double>(j0)) * std::log(y) - y -
                         std::lgamma(a0 + static_cast<double>(j0) + 1.0);
    const double d0 = std::exp(logD0);
    const double wFloor = w0 * 1e-18;

    double sum = w0 * P0;
    double w = w0;
    double P = P0;
    double d = d0;
    for (long j = j0;; ++j) { // upward
        w *= halfL / (static_cast<double>(j) + 1.0);
        P -= d;
        d *= y / (a0 + static_cast<double>(j) + 1.0);
        sum += w * P;
        if (w < wFloor) {
            break;
        }
    }
    if (j0 > 0) {
        w = w0;
        P = P0;
        d = d0;
        for (long j = j0; j >= 1; --j) { // downward
            d *= (a0 + static_cast<double>(j)) / y;
            P += d;
            w *= static_cast<double>(j) / halfL;
            sum += w * P;
            if (w < wFloor) {
                break;
            }
        }
    }
    return sum;
}

/// Noncentral chi-square upper tail (same mixture with the `Q` recurrence).
inline double noncentralChiSquareCcdf(double nu, double lambda, double x) {
    if (!(nu > 0.0) || !(lambda >= 0.0) || std::isnan(x)) {
        return std::numeric_limits<double>::quiet_NaN();
    }
    if (x <= 0.0) {
        return 1.0;
    }
    if (std::isinf(x)) {
        return 0.0;
    }
    if (lambda == 0.0) {
        return detail::gammaQDouble(0.5 * nu, 0.5 * x);
    }
    const double halfL = 0.5 * lambda;
    const double y = 0.5 * x;
    const double a0 = 0.5 * nu;
    const long j0 = static_cast<long>(std::floor(halfL));
    const double lw0 = -halfL + static_cast<double>(j0) * std::log(halfL) -
                       std::lgamma(static_cast<double>(j0) + 1.0);
    const double w0 = std::exp(lw0);
    const double Q0 = detail::gammaQDouble(a0 + static_cast<double>(j0), y);
    const double logD0 = (a0 + static_cast<double>(j0)) * std::log(y) - y -
                         std::lgamma(a0 + static_cast<double>(j0) + 1.0);
    const double d0 = std::exp(logD0);
    const double wFloor = w0 * 1e-18;

    double sum = w0 * Q0;
    double w = w0;
    double Q = Q0;
    double d = d0;
    for (long j = j0;; ++j) { // upward
        w *= halfL / (static_cast<double>(j) + 1.0);
        Q += d;
        d *= y / (a0 + static_cast<double>(j) + 1.0);
        sum += w * Q;
        if (w < wFloor) {
            break;
        }
    }
    if (j0 > 0) {
        w = w0;
        Q = Q0;
        d = d0;
        for (long j = j0; j >= 1; --j) { // downward
            d *= (a0 + static_cast<double>(j)) / y;
            Q -= d;
            w *= static_cast<double>(j) / halfL;
            sum += w * Q;
            if (w < wFloor) {
                break;
            }
        }
    }
    return sum;
}

/// Noncentral chi-square density (Bessel form, assembled in log space).
inline double noncentralChiSquarePdf(double nu, double lambda, double x) {
    if (!(nu > 0.0) || !(lambda >= 0.0) || std::isnan(x) || x < 0.0) {
        return std::numeric_limits<double>::quiet_NaN();
    }
    if (x == 0.0) {
        if (nu > 2.0) {
            return 0.0;
        }
        if (nu == 2.0) {
            return std::exp(-0.5 * lambda);
        }
        return std::numeric_limits<double>::infinity();
    }
    if (std::isinf(x)) {
        return 0.0;
    }
    const double a = 0.5 * nu - 1.0;
    const double z = std::sqrt(lambda * x);
    if (z == 0.0) {
        // lambda == 0, or lambda*x underflowed: the noncentral correction is
        // below double resolution, so use the central density.
        return std::exp((0.5 * nu - 1.0) * std::log(x) - 0.5 * x - 0.5 * nu * std::log(2.0) -
                        std::lgamma(0.5 * nu));
    }
    const double logFactor = -0.5 * (x + lambda) + (0.5 * a) * std::log(x / lambda) +
                             std::log(0.5) + detail::nccsLogBesselI(a, z);
    return std::exp(logFactor);
}

} // namespace lightape

#endif // LIGHTAPE_BASE_NONCENTRAL_CHI_SQUARE_H
