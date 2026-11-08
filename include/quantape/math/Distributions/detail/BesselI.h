#ifndef QUANTAPE_MATH_DISTRIBUTIONS_DETAIL_BESSEL_I_H
#define QUANTAPE_MATH_DISTRIBUTIONS_DETAIL_BESSEL_I_H

/**
 * @file BesselI.h
 * @brief Modified Bessel function of the first kind, real order (I_nu, log I_nu)
 *
 * Consumers: noncentral chi-square density, CIR/CEV exact densities.
 *
 * Algorithms:
 *  - Power series `I_nu(z) = (z/2)^nu sum_m (z^2/4)^m / (m! Gamma(m+nu+1))`
 *    for moderate z, evaluated with the term recurrence
 *    `t_{m+1} = t_m (z^2/4) / ((m+1)(m+nu+1))`. The sign of `Gamma(nu+1)` is
 *    applied explicitly (`lgamma` alone drops it, which flips `I_nu` for
 *    `nu in (-2,-1)`).
 *  - Asymptotic expansion for large z (`z > 15 + nu/2` and `z > nu^2/2`):
 *    `I_nu(z) ~ e^z / sqrt(2 pi z) * (1 - (mu-1)/(8z) + (mu-1)(mu-9)/(2(8z)^2) - ...)`,
 *    `mu = 4 nu^2`, truncated at its smallest term (the series is asymptotic,
 *    so running past the minimum diverges); the truncation error is ~`e^{-2z}`.
 *  - `logBesselI` returns log(I_nu) in log space (no overflow for large z);
 *    it is only meaningful where `I_nu > 0` (all `nu > -1`).
 *  - `besselIRatio` returns a signed `I_num/I_den`, falling back to the
 *    log-magnitude ratio when the values overflow (large z, both positive).
 *
 * Negative integer orders are reflected with `I_{-k} = I_k` before the series,
 * which would otherwise divide by `Gamma(0)` (the noncentral chi-square
 * derivative ratios at `nu = 2` need orders `-1` and `-2`).
 *
 * `besselK` (second kind) is not implemented yet; noncentral chi-square only
 * needs I. Add K with its own accuracy gates when a CIR/CEV density consumer
 * lands. std-only; no allocation; `z > 0`; series iteration cap 100000 bounds
 * large-order `|nu| >> z` use.
 */

#include "quantape/math/Distributions/detail/Digamma.h"
#include "quantape/math/Distributions/detail/SecondOrderJet.h"

#include <cmath>
#include <limits>

namespace quantape::math {
namespace detail {

namespace bessel_detail {

/// Series evaluation of I_nu(z): returns the sum of the (z/2)^nu-expanded series.
template <class T>
T seriesSum(T nu, T z) {
    using std::exp;
    using std::fabs;
    using std::lgamma;
    using std::log;
    using std::tgamma;
    const T halfZ = z / T(2);
    // lgamma returns log|Gamma|, so recover the sign separately: Gamma(nu+1)
    // is negative for nu in (-2,-1), (-4,-3), ... and would otherwise flip I_nu.
    const T gammaSign = tgamma(nu + T(1)) < T(0) ? T(-1) : T(1);
    T term = gammaSign * exp(nu * log(halfZ) - lgamma(nu + T(1)));
    T sum = term;
    const T z2 = halfZ * halfZ;
    for (int m = 1; m <= 100000; ++m) {
        term *= z2 / (T(m) * (T(m) + nu));
        sum += term;
        if (fabs(term) <= fabs(sum) * T(1e-17)) {
            break;
        }
    }
    return sum;
}

/// Multiplicative asymptotic correction series for I_nu(z) at large z:
/// `1 + sum_{m>=1} (-1)^m (mu-1)(mu-9)...(mu-(2m-1)^2) / (m! (8z)^m)`,
/// evaluated with the recurrence `t_m = -t_{m-1} (mu - (2m-1)^2) / (m * 8z)`
/// and truncated at the smallest term: the series diverges past its minimum,
/// so the 1e-17 test alone is not enough at moderate z (~15..20).
template <class T>
T asymptoticCorrection(T nu, T z) {
    using std::fabs;
    const T mu = T(4) * nu * nu;
    T term = T(1);
    T sum = T(1);
    for (int m = 1; m <= 100; ++m) {
        const T odd = T(2 * m - 1);
        const T next = term * (-(mu - odd * odd) / (T(m) * T(8) * z));
        if (fabs(next) >= fabs(term)) {
            break;
        }
        term = next;
        sum += term;
        if (fabs(term) <= fabs(sum) * T(1e-17)) {
            break;
        }
    }
    return sum;
}

inline double valueOf(double x) {
    return x;
}

/// True when the asymptotic expansion is accurate enough: past the series
/// crossover and with a small first correction (`z > nu^2/2`).
inline bool useAsymptotic(double nuD, double zD) {
    return zD >= 15.0 + 0.5 * nuD && zD >= 0.5 * nuD * nuD;
}

} // namespace bessel_detail

/**
 * @brief Modified Bessel function of the first kind I_nu(z), z > 0.
 *
 * Requires `nu > -1` for the series to stay positive; other real orders are
 * computed by analytic continuation (negative integer orders are reflected).
 */
template <class T>
T besselI(T nu, T z) {
    using std::exp;
    using std::sqrt;
    const double nuD = bessel_detail::valueOf(nu);
    const double zD = bessel_detail::valueOf(z);
    if (nuD < 0.0 && nuD == std::floor(nuD)) {
        return besselI(-nu, z);
    }
    if (bessel_detail::useAsymptotic(nuD, zD)) {
        return exp(z) / sqrt(T(2) * T(M_PI) * z) * bessel_detail::asymptoticCorrection(nu, z);
    }
    return bessel_detail::seriesSum(nu, z);
}

/**
 * @brief log(I_nu(z)) evaluated without overflow for large z.
 *
 * Only meaningful where `I_nu(z) > 0` (guaranteed for `nu > -1`); for
 * negative non-integer orders use `besselIRatio` instead.
 */
template <class T>
T logBesselI(T nu, T z) {
    using std::log;
    using std::sqrt;
    const double nuD = bessel_detail::valueOf(nu);
    const double zD = bessel_detail::valueOf(z);
    if (nuD < 0.0 && nuD == std::floor(nuD)) {
        return logBesselI(-nu, z);
    }
    if (bessel_detail::useAsymptotic(nuD, zD)) {
        return z - T(0.5) * log(T(2) * T(M_PI) * z) +
               log(bessel_detail::asymptoticCorrection(nu, z));
    }
    return log(bessel_detail::seriesSum(nu, z));
}

namespace bessel_detail {

/// Series accumulators for I_nu and its order derivatives:
/// `I = sum t_m`, `S1 = sum t_m psi(m+nu+1)`,
/// `S2 = sum t_m (psi(m+nu+1)^2 - psi'(m+nu+1))`.
inline void orderSeries(double nu, double z, double& I, double& S1, double& S2) {
    const double halfZ = z / 2.0;
    const double lz = std::log(halfZ);
    const double psi0 = digammaDouble(nu + 1.0);
    double term = std::exp(nu * lz - std::lgamma(nu + 1.0));
    I = term;
    S1 = term * psi0;
    S2 = term * (psi0 * psi0 - trigammaDouble(nu + 1.0));
    const double z2 = halfZ * halfZ;
    for (int m = 1; m <= 100000; ++m) {
        term *= z2 / (m * (m + nu));
        const double psi = digammaDouble(m + nu + 1.0);
        I += term;
        S1 += term * psi;
        S2 += term * (psi * psi - trigammaDouble(m + nu + 1.0));
        if (std::fabs(term) <= std::fabs(I) * 1e-17) {
            break;
        }
    }
}

/// Asymptotic correction series and its order derivatives. `t_m` uses the
/// convergent-until-optimal-term recurrence of `asymptoticCorrection`; each
/// term is differentiated in `nu` through `d ln a_m/dnu` and its second
/// derivative (guarded at the removable poles `4 nu^2 = (2m-1)^2`).
inline void orderAsymptotic(double nu, double z, double& S, double& S1, double& S2) {
    // Run the multiplicative correction recurrence on an exact jet in nu and
    // truncate on the largest component of the term vs the sum, so a vanishing
    // value term (e.g. mu = 1 at nu = 1/2) cannot stop the derivative sums
    // prematurely. The jet's second variable is unused.
    SecondOrderJet nuj(nu, 1.0, 0.0, 0.0, 0.0, 0.0);
    const SecondOrderJet muj = SecondOrderJet(4.0) * nuj * nuj;
    SecondOrderJet a(1.0);
    SecondOrderJet acc(1.0);
    SecondOrderJet scale(-1.0 / (8.0 * z));
    double prevMag = std::numeric_limits<double>::infinity();
    for (int m = 1; m <= 400; ++m) {
        const double c = static_cast<double>(2 * m - 1) * static_cast<double>(2 * m - 1);
        a = a * (muj - SecondOrderJet(c)) / SecondOrderJet(static_cast<double>(m));
        const SecondOrderJet term = a * scale;
        const double mag = maxAbsComponent(term);
        if (m > 1 && mag > prevMag) {
            break; // divergent tail: stop at the smallest term, do not add it
        }
        acc = acc + term;
        prevMag = mag;
        scale = scale * SecondOrderJet(-1.0 / (8.0 * z));
        if (mag <= maxAbsComponent(acc) * 1e-17) {
            break;
        }
    }
    S = acc.value;
    S1 = acc.d1;
    S2 = acc.d11;
}

} // namespace bessel_detail

/**
 * @brief Order derivative of I_nu(z): `dI_nu(z)/dnu` (nu > -1, z > 0).
 *
 * Series route (`z` below the `besselI` asymptotic switch): `dI/dnu = I ln(z/2)
 * - sum t_m psi(m+nu+1)`; asymptotic route: the multiplicative correction is
 * differentiated termwise in `mu = 4 nu^2`. The route is selected by the same
 * `useAsymptotic` predicate as `besselI` (both `z >= 15 + nu/2` **and**
 * `z >= nu^2/2`), so the derivative never differentiates a branch the value
 * did not take. Accuracy ~1e-13 relative for nu in [0,20], z in [1e-3, 30].
 */
inline double besselINuDeriv(double nu, double z) {
    if (!(nu > -1.0) || !(z > 0.0) || std::isnan(nu) || std::isnan(z)) {
        return std::numeric_limits<double>::quiet_NaN();
    }
    const double lz = std::log(z / 2.0);
    if (bessel_detail::useAsymptotic(nu, z)) {
        double S = 0.0, S1 = 0.0, S2 = 0.0;
        bessel_detail::orderAsymptotic(nu, z, S, S1, S2);
        const double pref = std::exp(z) / std::sqrt(2.0 * M_PI * z);
        return pref * S1;
    }
    double I = 0.0, S1 = 0.0, S2 = 0.0;
    bessel_detail::orderSeries(nu, z, I, S1, S2);
    return I * lz - S1;
}

/**
 * @brief Second order derivative of I_nu(z): `d2I_nu(z)/dnu2`.
 *
 * `I (ln(z/2))^2 - 2 ln(z/2) S1 + S2` with `S1`, `S2` from the same
 * series/asymptotic differentiation as `besselINuDeriv`, selected by the
 * shared `useAsymptotic` predicate (see there).
 */
inline double besselINuDeriv2(double nu, double z) {
    if (!(nu > -1.0) || !(z > 0.0) || std::isnan(nu) || std::isnan(z)) {
        return std::numeric_limits<double>::quiet_NaN();
    }
    const double lz = std::log(z / 2.0);
    if (bessel_detail::useAsymptotic(nu, z)) {
        double S = 0.0, S1 = 0.0, S2 = 0.0;
        bessel_detail::orderAsymptotic(nu, z, S, S1, S2);
        const double pref = std::exp(z) / std::sqrt(2.0 * M_PI * z);
        return pref * S2;
    }
    double I = 0.0, S1 = 0.0, S2 = 0.0;
    bessel_detail::orderSeries(nu, z, I, S1, S2);
    return I * lz * lz - 2.0 * lz * S1 + S2;
}

/**
 * @brief Signed ratio `I_num(z) / I_den(z)` (`z > 0`).
 *
 * Direct division whenever both values are finite and the denominator is
 * nonzero; this keeps the sign for negative non-integer orders (e.g.
 * `I_{-1.75}(z) < 0` for small `z`). When either value overflows the log
 * magnitude ratio is used (large `z`, where both values are positive).
 */
template <class T>
T besselIRatio(T num, T den, T z) {
    using std::exp;
    const T numerator = besselI(num, z);
    const T denominator = besselI(den, z);
    if (denominator != T(0) && std::isfinite(bessel_detail::valueOf(numerator)) &&
        std::isfinite(bessel_detail::valueOf(denominator))) {
        const T quotient = numerator / denominator;
        if (std::isfinite(bessel_detail::valueOf(quotient))) {
            return quotient;
        }
    }
    return exp(logBesselI(num, z) - logBesselI(den, z));
}

} // namespace detail
} // namespace quantape::math

#endif // QUANTAPE_MATH_DISTRIBUTIONS_DETAIL_BESSEL_I_H
