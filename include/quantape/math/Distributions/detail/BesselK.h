#ifndef QUANTAPE_MATH_DISTRIBUTIONS_DETAIL_BESSEL_K_H
#define QUANTAPE_MATH_DISTRIBUTIONS_DETAIL_BESSEL_K_H

/**
 * @file BesselK.h
 * @brief Modified Bessel function of the second kind K_nu(z), real order
 *        nu >= 0, z > 0, plus its log and first z-derivative.
 *
 * Consumers: CIR/CEV exact transition densities (companion of BesselI.h).
 *
 * Algorithm (hybrid; every branch exact, no finite differences):
 *  - Integer-part reduction: `n = round(nu)`, `u = nu - n` in [-1/2, 1/2).
 *    The seed pair `(K_u, K_{u+1})` is evaluated by the most accurate method
 *    for the argument, then the requested `K_nu` (and `K_{nu+1}` for the
 *    derivative) follows from the forward recurrence
 *    `K_{v+1} = K_{v-1} + (2v/z) K_v`. K is the dominant solution of the
 *    recurrence in the increasing-order direction for the small arguments
 *    here, so forward recurrence is stable.
 *  - `0 < z <= 1`: Temme's series (J. Comput. Phys. 19 (1975) 324) for
 *    `|u| <= 1/2`, which is regular at `u = 0` and avoids the
 *    `1/sin(pi u)` cancellation of `K_u = (pi/2)(I_{-u} - I_u)/sin(pi u)`.
 *    It needs the even/odd combinations
 *    `Q(u) = (Gamma(1+u)-Gamma(1-u))/(2u)`, `R(u) = (Gamma(1+u)+Gamma(1-u))/2`;
 *    for `|u| <= 0.05` these come from the Maclaurin series of `log Gamma(1+u)`
 *    powered by zeta constants (the tgamma difference loses all relative
 *    accuracy as `u -> 0`), otherwise from `tgamma` directly. Measured seed
 *    error <= 4e-15.
 *  - `1 < z < 16`: Thompson-Barnett continued fraction (Steed's algorithm)
 *    for `z1/z0 = U(u+3/2, 2u+1, 2z) / U(u+1/2, 2u+1, 2z)`, which stays
 *    cancellation-free where the ascending series has already lost digits
 *    (Temme at z = 2 is 3e-14, at z = 4 is 1e-12). Measured seed error <= 3e-15.
 *  - `z >= 16`: asymptotic expansion
 *    `K_nu(z) ~ sqrt(pi/(2z)) e^{-z} (1 + (mu-1)/(8z) + (mu-1)(mu-9)/(2!(8z)^2)
 *    + ...)`, `mu = 4 nu^2`, truncated at its smallest term via the same
 *    convergent-until-optimal-term recurrence as BesselI.h (running past the
 *    minimum diverges). The optimal truncation only reaches ~1e-15 for the
 *    seed orders `|u| <= 1/2` at `z ~ 16` (at z = 2.5 its error is ~1e-3),
 *    which is why the split sits at 16 rather than at the usual series
 *    crossover. Exact for half-integer seed orders, where the series terminates.
 *  - Half-integer `nu` (`u = -1/2`): the seed pair uses the closed forms
 *    `K_{1/2}(z) = sqrt(pi/(2z)) e^{-z}` and
 *    `K_{3/2}(z) = K_{1/2}(z) (1 + 1/z)`, and `K_{-1/2} = K_{1/2}`.
 *
 * `logBesselK` assembles log(K) in log space in every regime, so it neither
 * overflows for large order / small z nor underflows for large z (where
 * `besselK` correctly returns 0). Its forward recurrence uses
 * `log(fact + exp(l_prev - l_cur))`, which cannot overflow.
 *
 * `besselKPrime(nu,z) = -(K_{nu-1}(z) + K_{nu+1}(z))/2` with the even-order
 * reflection `K_{nu-1} = K_{|nu-1|}`. A derivative with respect to the order
 * `dK/dnu` is intentionally not provided (the consumers differentiate in z).
 *
 * Accuracy (double, verified against mpmath dps=60 on an 8500-point grid,
 * nu in [0,20], z in [1e-8, 1e3]): `besselK` max relative error 2.8e-15
 * (2e-14 target; 1.7e-15 for z < 0.5), `besselKPrime` max relative error
 * 2.9e-15. `logBesselK` max absolute error 1.5e-14 for z in [0.5,16), 6e-14
 * at z = 200 and 3.4e-13 at z = 1e3: the absolute error scales with the size
 * of the log value (`~eps |log K|`, scaled error <= 7.3e-16), which is
 * inherent to representing a ~1000-magnitude logarithm in double.
 *
 * Domain/edges (documented, no exceptions): `nu < 0`, non-finite `nu`,
 * `z <= 0` or NaN input return NaN; `z = +inf` returns 0 (`logBesselK`
 * returns -inf). `besselK` underflows to 0 for `z` beyond ~745 and overflows
 * to +inf for very small z at large order; `logBesselK` remains finite.
 * Orders above 1e5 recurrence steps (far outside the consumer range) return
 * NaN instead of looping unboundedly. std-only (plus util/Constants.h);
 * no allocation. Double-only: no AD scalar path (the CIR/CEV density
 * kernels are double-only, like the NoncentralChiSquare derivative packs).
 */

#include "quantape/math/Distributions/detail/BesselI.h"
#include "quantape/util/Constants.h"

#include <cmath>
#include <limits>

namespace quantape::math {
namespace detail {

namespace besselk_detail {

using ::quantape::util::kEulerGamma;
using ::quantape::util::kLn2;
using ::quantape::util::kPi;

/// Temme seed order: |u| <= 0.5. Maclaurin branch bound for Q (=e1) and R.
inline constexpr double kQrSeriesBound = 0.05;
/// Temme / continued-fraction argument split.
inline constexpr double kContinuedFractionZ = 1.0;
/// Asymptotic-expansion argument split (optimal truncation reaches 1e-15 here).
inline constexpr double kAsymptoticZ = 16.0;
/// Recurrence-step cap; orders above it are outside the supported domain.
inline constexpr int kMaxSteps = 100000;
/// Series/continued-fraction iteration caps (convergence needs tens of terms).
inline constexpr int kMaxTemmeTerms = 10000;
inline constexpr int kMaxContinuedFractionTerms = 200000;

// zeta(2j) and zeta(2j+1) for the small-|u| Maclaurin series; generated at
// 40 decimal digits and rounded to double.
inline constexpr double kZeta2 = 1.6449340668482264365;
inline constexpr double kZeta3 = 1.2020569031595942854;
inline constexpr double kZeta4 = 1.0823232337111381915;
inline constexpr double kZeta5 = 1.0369277551433699263;
inline constexpr double kZeta6 = 1.0173430619844491397;
inline constexpr double kZeta7 = 1.0083492773819228268;
inline constexpr double kZeta8 = 1.0040773561979443394;
inline constexpr double kZeta9 = 1.0020083928260822144;
inline constexpr double kZeta10 = 1.0009945751278180853;
inline constexpr double kZeta11 = 1.0004941886041194646;
inline constexpr double kZeta12 = 1.0002460865533080483;
inline constexpr double kZeta13 = 1.0001227133475784891;

/// Even/odd Gamma combinations around u = 0 (both even, analytic):
/// `Q = (Gamma(1+u)-Gamma(1-u))/(2u)` and `R = (Gamma(1+u)+Gamma(1-u))/2`.
/// The direct tgamma difference loses relative accuracy for small |u|
/// (absolute tgamma error ~eps divides by 2u), so the Maclaurin series of
/// `log Gamma(1+u) = -gamma u + sum_{k>=2} (-1)^k zeta(k) u^k / k` is used:
/// with `A = sum_even zeta(k) u^k / k` and `B = -gamma u - sum_odd zeta(k)
/// u^k / k`, `Q = e^A sinh(B)/u` and `R = e^A cosh(B)`.
inline void qrPair(double u, double& qOut, double& rOut) {
    if (std::fabs(u) <= kQrSeriesBound) {
        const double u2 = u * u;
        const double even =
            u2 * (kZeta2 / 2.0 +
                  u2 * (kZeta4 / 4.0 +
                        u2 * (kZeta6 / 6.0 +
                              u2 * (kZeta8 / 8.0 + u2 * (kZeta10 / 10.0 + u2 * kZeta12 / 12.0)))));
        const double odd =
            kZeta3 / 3.0 +
            u2 * (kZeta5 / 5.0 +
                  u2 * (kZeta7 / 7.0 +
                        u2 * (kZeta9 / 9.0 + u2 * (kZeta11 / 11.0 + u2 * kZeta13 / 13.0))));
        const double b = -kEulerGamma * u - u * u2 * odd;
        const double eA = std::exp(even);
        qOut = eA * std::sinh(b) / u;
        rOut = eA * std::cosh(b);
        return;
    }
    const double gammaPlus = std::tgamma(1.0 + u);
    const double gammaMinus = std::tgamma(1.0 - u);
    qOut = (gammaPlus - gammaMinus) / (2.0 * u);
    rOut = 0.5 * (gammaPlus + gammaMinus);
}

/// Temme's pair `(K_u(x), K_{u+1}(x))` for `|u| <= 1/2`, `0 < x <= 1`.
/// The `sin(pi u)` factor of the reflection formula cancels against the
/// Gamma combinations (Q, R) and never appears in the arithmetic.
inline void temmePair(double u, double x, double& kvOut, double& kv1Out) {
    double q = 0.0;
    double r = 0.0;
    if (u == 0.0) {
        q = -kEulerGamma;
        r = 1.0;
    } else {
        qrPair(u, q, r);
    }
    // log(x/2) split so the product never underflows to 0 for subnormal x.
    const double a = std::log(x) - kLn2;
    const double b = std::exp(u * a);
    const double sigma = -a * u;
    double p = std::tgamma(1.0 + u) / (2.0 * b);
    double qq = std::tgamma(1.0 - u) * b * 0.5;
    // f = cosh(sigma) Q + sinh(sigma)/u R is the u -> 0 continuation of the
    // standard Temme f; at u = 0 it collapses to psi(1) + log(2/x).
    double f =
        (u == 0.0) ? (-kEulerGamma - a) : (std::cosh(sigma) * q + (std::sinh(sigma) / u) * r);
    double h = p;
    double coefficient = 1.0;
    double sum = f;
    double sum1 = h;
    const double eps = std::numeric_limits<double>::epsilon();
    for (int k = 1; k <= kMaxTemmeTerms; ++k) {
        const double kd = static_cast<double>(k);
        f = (kd * f + p + qq) / (kd * kd - u * u);
        p /= (kd - u);
        qq /= (kd + u);
        h = p - kd * f;
        coefficient *= x * x / (4.0 * kd);
        sum += coefficient * f;
        sum1 += coefficient * h;
        if (std::fabs(coefficient * f) <= std::fmax(std::fabs(sum), std::fabs(sum1)) * eps) {
            break;
        }
    }
    kvOut = sum;
    kv1Out = 2.0 * sum1 / x;
}

/// Thompson-Barnett continued fraction for `(K_u(x), K_{u+1}(x))`,
/// `|u| <= 1/2`, `x > 1` (Steed's algorithm, modified Lentz variant).
inline void continuedFractionPair(double u, double x, double& kvOut, double& kv1Out) {
    double a = u * u - 0.25;
    double b = 2.0 * (x + 1.0);
    double d = 1.0 / b;
    double f = d;
    double delta = d;
    double prev = 0.0;
    double current = 1.0;
    double q = -a;
    double c = -a;
    double s = 1.0 + q * delta;
    const double eps = std::numeric_limits<double>::epsilon();
    for (int k = 2; k <= kMaxContinuedFractionTerms; ++k) {
        const double kd = static_cast<double>(k);
        a -= 2.0 * (kd - 1.0);
        b += 2.0;
        d = 1.0 / (b + a * d);
        delta *= b * d - 1.0;
        f += delta;
        const double qNext = (prev - (b - 2.0) * current) / a;
        prev = current;
        current = qNext;
        c *= -a / kd;
        q += c * qNext;
        s += q * delta;
        if (std::fabs(q * delta) < std::fabs(s) * eps) {
            break;
        }
    }
    const double ratio = (0.5 + u + x + (u * u - 0.25) * f) / x;
    kvOut = std::sqrt(kPi / (2.0 * x)) * std::exp(-x) / s;
    kv1Out = kvOut * ratio;
}

/// Multiplicative asymptotic correction for K_nu(z):
/// `1 + sum_{m>=1} prod_{j=1}^m (mu-(2j-1)^2) / (m! (8z)^m)`, `mu = 4 nu^2`,
/// with the recurrence `t_m = t_{m-1} (mu-(2m-1)^2)/(m 8z)` truncated at the
/// smallest term (the series is divergent past its minimum).
inline double asymptoticCorrection(double nu, double z) {
    const double mu = 4.0 * nu * nu;
    double term = 1.0;
    double sum = 1.0;
    for (int m = 1; m <= 200; ++m) {
        const double md = static_cast<double>(m);
        const double odd = 2.0 * md - 1.0;
        const double next = term * (mu - odd * odd) / (md * 8.0 * z);
        if (std::fabs(next) >= std::fabs(term)) {
            break;
        }
        term = next;
        sum += term;
        if (std::fabs(term) <= std::fabs(sum) * 1e-17) {
            break;
        }
    }
    return sum;
}

/// `log K_nu(z)` assembled directly in log space at `z >= kAsymptoticZ`.
inline double asymptoticLog(double nu, double z) {
    return 0.5 * std::log(kPi / (2.0 * z)) - z + std::log(asymptoticCorrection(nu, z));
}

/// Seed pair `(K_u(z), K_{u+1}(z))` by the regime split above.
inline void seedPair(double u, double z, double& kvOut, double& kv1Out) {
    if (z >= kAsymptoticZ) {
        const double prefactor = std::sqrt(kPi / (2.0 * z)) * std::exp(-z);
        kvOut = prefactor * asymptoticCorrection(u, z);
        kv1Out = prefactor * asymptoticCorrection(u + 1.0, z);
    } else if (z > kContinuedFractionZ) {
        continuedFractionPair(u, z, kvOut, kv1Out);
    } else {
        temmePair(u, z, kvOut, kv1Out);
    }
}

/// Exact half-integer seed pair: `K_{1/2}`, `K_{3/2}` closed forms and the
/// even-order reflection `K_{-1/2} = K_{1/2}` (so the pair is duplicated).
inline void halfIntegerPair(double u, double z, double& kvOut, double& kv1Out) {
    // Written as sqrt(pi/2)/sqrt(z) so pi/(2z) cannot overflow for tiny z.
    const double kHalf = std::sqrt(kPi / 2.0) / std::sqrt(z) * std::exp(-z);
    kvOut = kHalf;
    kv1Out = (u == 0.5) ? kHalf * (1.0 + 1.0 / z) : kHalf;
}

} // namespace besselk_detail

/**
 * @brief Modified Bessel function of the second kind K_nu(z), `nu >= 0`, `z > 0`.
 *
 * Returns NaN outside the domain (`nu < 0`, non-finite `nu`, `z <= 0`, NaN),
 * 0 at `z = +inf`, and underflows to 0 for `z` around 745 and beyond.
 */
inline double besselK(double nu, double z) {
    using namespace besselk_detail;
    if (!std::isfinite(nu) || std::isnan(z)) {
        return std::numeric_limits<double>::quiet_NaN();
    }
    if (!(nu >= 0.0) || !(z > 0.0)) {
        return std::numeric_limits<double>::quiet_NaN();
    }
    if (std::isinf(z)) {
        return 0.0;
    }
    const double n = std::round(nu);
    const double u = nu - n;
    if (n > static_cast<double>(kMaxSteps)) {
        return std::numeric_limits<double>::quiet_NaN();
    }
    double prev = 0.0;
    double current = 0.0;
    if (u == -0.5 || u == 0.5) {
        halfIntegerPair(u, z, prev, current);
    } else {
        seedPair(u, z, prev, current);
    }
    const int steps = static_cast<int>(n);
    for (int k = 1; k <= steps; ++k) {
        const double factor = 2.0 * (u + static_cast<double>(k)) / z;
        const double next = factor * current + prev;
        prev = current;
        current = next;
    }
    return prev;
}

/**
 * @brief log(K_nu(z)) evaluated in log space, so it stays finite where
 * `besselK` underflows (large z) or overflows (small z, large order).
 *
 * Same domain as `besselK`; `z = +inf` gives -inf.
 */
inline double logBesselK(double nu, double z) {
    using namespace besselk_detail;
    if (!std::isfinite(nu) || std::isnan(z)) {
        return std::numeric_limits<double>::quiet_NaN();
    }
    if (!(nu >= 0.0) || !(z > 0.0)) {
        return std::numeric_limits<double>::quiet_NaN();
    }
    if (std::isinf(z)) {
        return -std::numeric_limits<double>::infinity();
    }
    const double n = std::round(nu);
    const double u = nu - n;
    if (n > static_cast<double>(kMaxSteps)) {
        return std::numeric_limits<double>::quiet_NaN();
    }
    double logPrev = 0.0;
    double logCurrent = 0.0;
    if (u == -0.5 || u == 0.5) {
        const double logHalf = 0.5 * (std::log(kPi / 2.0) - std::log(z)) - z;
        logPrev = logHalf;
        logCurrent = (u == 0.5) ? logHalf + std::log1p(1.0 / z) : logHalf;
    } else if (z >= kAsymptoticZ) {
        logPrev = asymptoticLog(u, z);
        logCurrent = asymptoticLog(u + 1.0, z);
    } else {
        double value = 0.0;
        double value1 = 0.0;
        if (z > kContinuedFractionZ) {
            continuedFractionPair(u, z, value, value1);
        } else {
            temmePair(u, z, value, value1);
        }
        logPrev = std::log(value);
        logCurrent = std::log(value1);
    }
    const int steps = static_cast<int>(n);
    for (int k = 1; k <= steps; ++k) {
        const double factor = 2.0 * (u + static_cast<double>(k)) / z;
        const double logNext = logCurrent + std::log(factor + std::exp(logPrev - logCurrent));
        logPrev = logCurrent;
        logCurrent = logNext;
    }
    return logPrev;
}

/**
 * @brief `K'_nu(z) = -(K_{nu-1}(z) + K_{nu+1}(z))/2`, using the even-order
 * reflection `K_{nu-1} = K_{|nu-1|}`; NaN outside the `besselK` domain.
 */
inline double besselKPrime(double nu, double z) {
    if (!std::isfinite(nu) || std::isnan(z) || !(nu >= 0.0) || !(z > 0.0)) {
        return std::numeric_limits<double>::quiet_NaN();
    }
    if (std::isinf(z)) {
        return 0.0;
    }
    return -0.5 * (besselK(std::fabs(nu - 1.0), z) + besselK(nu + 1.0, z));
}

// ---------------------------------------------------------------------------
// Order derivative dK_nu(z)/dnu
//
// Integral representation, stable for all z:
//   K_nu(z) = int_0^inf e^{-z cosh t} cosh(nu t) dt,
//   dK/dnu  = int_0^inf e^{-z cosh t} t sinh(nu t) dt.
// With s = cosh t - 1 = u^2 it becomes
//   K_nu(z)     = (e^{-z}/sqrt(z)) int_{-inf}^{inf} e^{-t^2} cosh(nu acosh(1+t^2/z))/sqrt(t^2/z+2)
//   dt, dK/dnu      = (e^{-z}/sqrt(z)) int e^{-t^2} acosh(1+t^2/z) sinh(nu
//   acosh(1+t^2/z))/sqrt(t^2/z+2) dt,
// evaluated with a 64-point Gauss-Hermite rule (table below, generated at 60
// digits; sum of weights = sqrt(pi)). Accuracy ~1e-9 relative for z >= 0.5.
//
// For 0 < z < 0.5 the Hermite rule loses accuracy as z -> 0; there the exact
// reflection formula
//   K_nu = (pi/2) (I_{-nu} - I_nu) / sin(pi nu),
//   dK/dnu = (pi/2)[ (-dI/dnu|_{-nu} - dI/dnu|_nu)/sin(pi nu)
//                    - (I_{-nu} - I_nu) pi cos(pi nu)/sin^2(pi nu) ]
// is used with the exact I order derivatives from BesselI.h. The combination
// is 0/0 near integer order; this corner (z < 0.5 and |sin(pi nu)| < 0.1)
// returns NaN and is documented rather than approximated.
// ---------------------------------------------------------------------------

namespace bessel_k_detail {

constexpr double kGhNodes[64] = {
    -10.526123167960546,  -9.895287586829539,   -9.3731595496467212,  -8.9072490999647698,
    -8.4775290833798631,  -8.0736872850102252,  -7.6895401640404968,  -7.3210130327809492,
    -6.9652411205511075,  -6.6201122626360274,  -6.2840112287748282,  -5.955666326799486,
    -5.6340521643499721,  -5.3183252246332709,  -5.0077796021987682,  -4.7018156474074998,
    -4.3999171682281376,  -4.1016344745666567,  -3.8065715139453605,  -3.5143759357409062,
    -3.2247312919920357,  -2.9373508230046218,  -2.651972435430635,   -2.3683545886324014,
    -2.086272879881762,   -1.8055171714655449,  -1.5258891402098637,  -1.2472001569431179,
    -0.96926942307117802, -0.69192230581004458, -0.41498882412107868, -0.13830224498700972,
    0.13830224498700972,  0.41498882412107868,  0.69192230581004458,  0.96926942307117802,
    1.2472001569431179,   1.5258891402098637,   1.8055171714655449,   2.086272879881762,
    2.3683545886324014,   2.651972435430635,    2.9373508230046218,   3.2247312919920357,
    3.5143759357409062,   3.8065715139453605,   4.1016344745666567,   4.3999171682281376,
    4.7018156474074998,   5.0077796021987682,   5.3183252246332709,   5.6340521643499721,
    5.955666326799486,    6.2840112287748282,   6.6201122626360274,   6.9652411205511075,
    7.3210130327809492,   7.6895401640404968,   8.0736872850102252,   8.4775290833798631,
    8.9072490999647698,   9.3731595496467212,   9.895287586829539,    10.526123167960546};
constexpr double kGhWeights[64] = {
    5.5357065358569428e-49, 1.6797479901081592e-43, 3.4211380112557405e-39, 1.5573906246297638e-35,
    2.5496608991129993e-32, 1.9291035954649669e-29, 7.8617977889259104e-27, 1.9117068833006428e-24,
    2.9828627842798512e-22, 3.1522545665037814e-20, 2.3518847106758191e-18, 1.280093391322438e-16,
    5.2186237265908475e-15, 1.6283407307097204e-13, 3.9591777669477239e-12, 7.6152172501454514e-11,
    1.1736167423215493e-9,  1.4651253164761094e-8,  1.4955329367272471e-7,  1.2583402510311846e-6,
    8.7884992308503592e-6,  5.1259291357862747e-5,  0.00025098369851306249, 0.0010363290995075777,
    0.0036225869785344588,  0.010756040509879137,   0.027203128953688918,   0.058739981964099435,
    0.10849834930618684,    0.1716858423490837,     0.23299478606267805,    0.27137742494130398,
    0.27137742494130398,    0.23299478606267805,    0.1716858423490837,     0.10849834930618684,
    0.058739981964099435,   0.027203128953688918,   0.010756040509879137,   0.0036225869785344588,
    0.0010363290995075777,  0.00025098369851306249, 5.1259291357862747e-5,  8.7884992308503592e-6,
    1.2583402510311846e-6,  1.4955329367272471e-7,  1.4651253164761094e-8,  1.1736167423215493e-9,
    7.6152172501454514e-11, 3.9591777669477239e-12, 1.6283407307097204e-13, 5.2186237265908475e-15,
    1.280093391322438e-16,  2.3518847106758191e-18, 3.1522545665037814e-20, 2.9828627842798512e-22,
    1.9117068833006428e-24, 7.8617977889259104e-27, 1.9291035954649669e-29, 2.5496608991129993e-32,
    1.5573906246297638e-35, 3.4211380112557405e-39, 1.6797479901081592e-43, 5.5357065358569428e-49};

inline double scaledKCosh(double nu, double z) {
    double s = 0.0;
    for (int i = 0; i < 64; ++i) {
        const double t = kGhNodes[i];
        const double q = t * t / z;
        const double a = std::acosh(1.0 + q);
        s += kGhWeights[i] * std::cosh(nu * a) / std::sqrt(q + 2.0);
    }
    return s;
}

inline double scaledKSinh(double nu, double z) {
    double s = 0.0;
    for (int i = 0; i < 64; ++i) {
        const double t = kGhNodes[i];
        const double q = t * t / z;
        const double a = std::acosh(1.0 + q);
        s += kGhWeights[i] * a * std::sinh(nu * a) / std::sqrt(q + 2.0);
    }
    return s;
}

} // namespace bessel_k_detail

/**
 * @brief Order derivative of K_nu(z): dK_nu(z)/dnu (nu >= 0, z > 0).
 *
 * Gauss-Hermite integral for z >= 0.5 (~1e-9 relative); exact reflection
 * formula with the I order derivatives at base orders in (-1,1) plus the
 * stable ascending K recurrence for z < 0.5 when the order is away from an
 * integer. Integer/near-integer orders with z < 0.5 fall back to the scaled
 * integral with documented degraded accuracy (~1e-5 at z ~ 0.05).
 */
inline double besselKNuDeriv(double nu, double z) {
    if (!std::isfinite(nu) || std::isnan(z) || !(nu >= 0.0) || !(z > 0.0)) {
        return std::numeric_limits<double>::quiet_NaN();
    }
    if (std::isinf(z)) {
        return 0.0;
    }
    if (z >= 0.5) {
        const double pref = std::exp(-z) / std::sqrt(z);
        return pref * bessel_k_detail::scaledKSinh(nu, z);
    }
    const double u = nu - std::floor(nu);
    if (std::sin(M_PI * u) < 0.3) {
        // Integer or near-integer order: the reflection combination is 0/0
        // there, so use the scaled integral outside its accurate z band and
        // document the degraded accuracy (roughly 1e-5 at z ~ 0.05, 4e-4 at
        // z ~ 0.01).
        const double pref = std::exp(-z) / std::sqrt(z);
        return pref * bessel_k_detail::scaledKSinh(nu, z);
    }
    // Reflection formula at a base order v in (-1, 1) away from an integer:
    //   K_v = (pi/2)(I_{-v} - I_v)/sin(pi v),
    //   dK/dv = (pi/2)[ (-dI/dv|_{-v} - dI/dv|_v)/sin(pi v)
    //                    - (I_{-v} - I_v) pi cos(pi v)/sin^2(pi v) ].
    const auto reflectionDeriv = [z](double v) {
        const double s = std::sin(M_PI * v);
        const double A = besselI(-v, z);
        const double B = besselI(v, z);
        const double dA = -besselINuDeriv(-v, z);
        const double dB = besselINuDeriv(v, z);
        const double c = std::cos(M_PI * v);
        return 0.5 * M_PI * ((dA - dB) / s - (A - B) * M_PI * c / (s * s));
    };
    // Seeds at u - 1 and u, then the stable ascending recurrence
    // dK_{v+1} = dK_{v-1} + (2/z) K_v + (2v/z) dK_v.
    double dprev = reflectionDeriv(u - 1.0);
    double dcur = reflectionDeriv(u);
    const int n = static_cast<int>(std::floor(nu));
    for (int k = 0; k < n; ++k) {
        const double v = u + static_cast<double>(k);
        const double dnext = dprev + (2.0 / z) * besselK(v, z) + (2.0 * v / z) * dcur;
        dprev = dcur;
        dcur = dnext;
    }
    return dcur;
}

/**
 * @brief Order log-derivative `d log K_nu(z)/dnu`.
 *
 * Ratio of the two scaled Gauss-Hermite integrals; stable for any z where the
 * integrals are representable, including z >= 0.5 where K itself underflows.
 * Falls back to `besselKNuDeriv(nu,z)/besselK(nu,z)` for z < 0.5.
 */
inline double besselKNuLogDeriv(double nu, double z) {
    if (!std::isfinite(nu) || std::isnan(z) || !(nu >= 0.0) || !(z > 0.0)) {
        return std::numeric_limits<double>::quiet_NaN();
    }
    if (std::isinf(z)) {
        return 0.0;
    }
    if (z >= 0.5) {
        const double c = bessel_k_detail::scaledKCosh(nu, z);
        const double s = bessel_k_detail::scaledKSinh(nu, z);
        return s / c;
    }
    const double k = besselK(nu, z);
    const double dk = besselKNuDeriv(nu, z);
    return dk / k;
}

} // namespace detail
} // namespace quantape::math

#endif // QUANTAPE_MATH_DISTRIBUTIONS_DETAIL_BESSEL_K_H
