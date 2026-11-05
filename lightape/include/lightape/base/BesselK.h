#ifndef LIGHTAPE_BASE_BESSELK_H
#define LIGHTAPE_BASE_BESSELK_H

/**
 * @file BesselK.h
 * @brief Modified Bessel function of the second kind K_nu(z), real order
 *        nu >= 0, z > 0, plus its log, first derivatives and the exact
 *        second-order kernels d2K/dnu2, d2K/dnudz, d2K/dz2
 *
 * In-house port of quantape::math::detail's BesselK.h (same owner; see
 * quantape/math/Distributions/detail/BesselK.h). Evaluation order is preserved
 * from the source; only the namespace changes and the shared constants that
 * quantape reads from util/Constants.h come from base/BesselI.h (kPi/kLn2/
 * kEulerGamma, inlined for the std-only lightape layer).
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
 *  - `0 < z <= 1`: Temme's series for `|u| <= 1/2`, regular at `u = 0`
 *    (`Q(u)`, `R(u)` even/odd Gamma combinations; Maclaurin branch for
 *    `|u| <= 0.05`).
 *  - `1 < z < 16`: Thompson-Barnett continued fraction (Steed's algorithm)
 *    for `z1/z0 = U(u+3/2, 2u+1, 2z) / U(u+1/2, 2u+1, 2z)`.
 *  - `z >= 16`: asymptotic expansion truncated at its smallest term, exact
 *    for half-integer seed orders.
 *  - Half-integer `nu` (`u = -1/2`): closed forms for `K_{1/2}`, `K_{3/2}`.
 *
 * `logBesselK` assembles log(K) in log space in every regime, so it neither
 * overflows for large order / small z nor underflows for large z (where
 * `besselK` correctly returns 0). Its forward recurrence uses
 * `log(fact + exp(l_prev - l_cur))`, which cannot overflow.
 *
 * `besselKPrime(nu,z) = -(K_{nu-1}(z) + K_{nu+1}(z))/2` with the even-order
 * reflection `K_{nu-1} = K_{|nu-1|}`.
 *
 * `besselKNuDeriv` (order derivative `dK_nu(z)/dnu`): Gauss-Hermite integral
 * for z >= 0.5 (~1e-9 relative); exact reflection formula with the I order
 * derivatives at base orders in (-1,1) plus the stable ascending K recurrence
 * for z < 0.5 when the order is away from an integer; integer/near-integer
 * orders with z < 0.5 fall back to the scaled integral with documented
 * degraded accuracy. `besselKNuLogDeriv` is the ratio of the two scaled
 * integrals (`d log K/dnu`).
 *
 * `besselKNuDeriv2` (d2K/dnu2), `besselKNuZDeriv` (d2K/dnudz) and
 * `besselKZZDeriv` (d2K/dz2) are the exact second-order kernels used by the AD
 * layer's VHP replay (decision #26: no finite differences). The two order
 * integrals come from the exact integral representation, evaluated with an
 * adaptive double-exponential rule; `besselKZZDeriv` is assembled from the
 * modified Bessel ODE. Verified <= ~6e-15 relative on the accepted grid
 * (nu in [0,20], z in [0.01,40]) and <= ~1e-14 out to nu = 50, z = 1e-3.
 * Where the transformed integrand itself overflows, roughly nu >= 60 with
 * z <= 0.01 and nu >= 90 with z <= 0.1, the order kernels return NaN instead
 * of an approximation, so the AD wrapper records first order only and
 * `hessianVector` fails loudly.
 *
 * Domain/edges (documented, no exceptions): `nu < 0`, non-finite `nu`,
 * `z <= 0` or NaN input return NaN; `z = +inf` returns 0 (`logBesselK`
 * returns -inf). Orders above 1e5 recurrence steps return NaN instead of
 * looping unboundedly. std-only; no allocation.
 */

#include <cmath>
#include <limits>

#include "lightape/base/BesselI.h"

namespace lightape {
namespace detail {

namespace besselk_detail {

using bessel_detail::kEulerGamma;
using bessel_detail::kLn2;
using bessel_detail::kPi;

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
//   K_nu(z)     = (e^{-z}/sqrt(z)) int e^{-t^2}
//   cosh(nu acosh(1+t^2/z))/sqrt(t^2/z+2) dt, dK/dnu      = (e^{-z}/sqrt(z)) int
//   e^{-t^2} acosh(1+t^2/z) sinh(nu acosh(1+t^2/z))/sqrt(t^2/z+2) dt, evaluated
//   with a 64-point Gauss-Hermite rule (table below, generated at 60 digits;
//   sum of weights = sqrt(pi)). Accuracy ~1e-9 relative for z >= 0.5.
//
// For 0 < z < 0.5 the Hermite rule loses accuracy as z -> 0; there the exact
// reflection formula
//   K_nu = (pi/2) (I_{-nu} - I_nu) / sin(pi nu),
//   dK/dnu = (pi/2)[ (-dI/dnu|_{-nu} - dI/dnu|_nu)/sin(pi nu)
//                    - (I_{-nu} - I_nu) pi cos(pi nu)/sin^2(pi nu) ]
// is used with the exact I order derivatives from BesselI.h. The combination
// is 0/0 near integer order; this corner (z < 0.5 and |sin(pi nu)| < 0.3)
// falls back to the scaled integral with documented degraded accuracy rather
// than failing.
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
    if (std::sin(bessel_detail::kPi * u) < 0.3) {
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
        const double s = std::sin(bessel_detail::kPi * v);
        const double A = besselI(-v, z);
        const double B = besselI(v, z);
        const double dA = -besselINuDeriv(-v, z);
        const double dB = besselINuDeriv(v, z);
        const double c = std::cos(bessel_detail::kPi * v);
        return 0.5 * bessel_detail::kPi *
               ((dA - dB) / s - (A - B) * bessel_detail::kPi * c / (s * s));
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

// ---------------------------------------------------------------------------
// Second order derivatives d2K/dnu2, d2K/dnudz and d2K/dz2
//
// Exact integral representation (z > 0, nu >= 0):
//   K_nu(z) = int_0^inf e^{-z cosh t} cosh(nu t) dt,
//   K_nn    = int_0^inf e^{-z cosh t} t^2 cosh(nu t) dt,
//   K_nz    = -int_0^inf e^{-z cosh t} t sinh(nu t) cosh t dt.
// Substituting u = cosh t - 1 (dt = du/sinh t), then factoring e^{-z},
//   K_nn = e^{-z} int_0^inf e^{-z u} t^2 cosh(nu t) / sqrt(u(u+2)) du,
//   K_nz = -e^{-z} int_0^inf e^{-z u} t sinh(nu t) cosh t / sqrt(u(u+2)) du,
// with t = acosh(1+u). The square-root endpoint is integrable; the
// double-exponential map u = exp(w - exp(-w)) (du/dw = u (1+e^{-w})) turns
// the transformed integrand into an entire function of w that decays doubly
// exponentially at both ends, so the trapezoidal rule converges rapidly.
// The step h = 0.25/(1+sqrt(1+nu)) resolves the peak of the integrand
// (width ~ 1/sqrt(nu) around t* = asinh(nu/z)); the range k = -K..K is
// doubled until two consecutive levels agree to kDeTolerance relative, so
// the same rule covers z ~ 0.01 through large z. No finite differences.
// ---------------------------------------------------------------------------

namespace bessel_k_detail {

/// DE step scale: resolves the order-dependent peak width ~ 1/sqrt(nu).
inline constexpr double kDeStepScale = 0.25;
/// Relative agreement required between two consecutive node-range levels.
inline constexpr double kDeTolerance = 1e-14;
/// Largest node half-range K (2K+1 nodes per level).
inline constexpr int kDeMaxK = 8192;

/// Trapezoidal DE sums for `int_0^inf e^{-z u} t^2 cosh(nu t) / sqrt(u(u+2)) du`
/// and `int_0^inf e^{-z u} t sinh(nu t) cosh(t) / sqrt(u(u+2)) du`
/// (unscaled by e^{-z}); returns false if the range extension has not
/// converged within `kDeMaxK` (the caller reports NaN; fail loud).
inline bool kSecondOrderIntegrals(double nu, double z, double& hNuNu, double& hNuZ) {
    const double h = kDeStepScale / (1.0 + std::sqrt(1.0 + nu));
    double prevNN = 0.0;
    double prevNZ = 0.0;
    for (int halfRange = 16; halfRange <= kDeMaxK; halfRange *= 2) {
        double accNN = 0.0;
        double accNZ = 0.0;
        for (int k = -halfRange; k <= halfRange; ++k) {
            const double w = static_cast<double>(k) * h;
            const double em = std::exp(-w);
            const double u = std::exp(w - em);
            if (u == 0.0) {
                continue; // beyond the doubly-exponentially decaying tail
            }
            const double e = std::exp(-z * u);
            if (e == 0.0) {
                continue; // underflowed tail
            }
            // e^{-z u} du/dw / sqrt(u(u+2)) = e^{-z u} sqrt(u)(1+e^{-w})/sqrt(u+2).
            const double g = h * e * std::sqrt(u) * (1.0 + em) / std::sqrt(u + 2.0);
            const double t = std::acosh(1.0 + u);
            const double c = std::cosh(nu * t);
            const double s = std::sinh(nu * t);
            accNN += g * t * t * c;
            accNZ += g * t * s * std::cosh(t);
        }
        if (halfRange > 16 &&
            std::fabs(accNN - prevNN) <=
                kDeTolerance * std::fmax(std::fabs(accNN), std::fabs(prevNN)) &&
            std::fabs(accNZ - prevNZ) <=
                kDeTolerance * std::fmax(std::fabs(accNZ), std::fabs(prevNZ))) {
            const double pref = std::exp(-z);
            hNuNu = pref * accNN;
            hNuZ = -pref * accNZ;
            return true;
        }
        prevNN = accNN;
        prevNZ = accNZ;
    }
    return false;
}

} // namespace bessel_k_detail

/**
 * @brief Second order derivative `d2K_nu(z)/dnu2` (`nu >= 0`, `z > 0`).
 *
 * Exact integral `int_0^inf e^{-z cosh t} t^2 cosh(nu t) dt` evaluated with
 * the adaptive double-exponential rule above (<= ~6e-15 relative on the
 * accepted grid `nu in [0,20]`, `z in [0.01,40]`, and <= ~1e-14 out to
 * `nu = 50`, `z = 1e-3`). NaN outside the `besselK` domain or when the
 * quadrature overflows / fails to converge (never an approximation).
 */
inline double besselKNuDeriv2(double nu, double z) {
    if (!std::isfinite(nu) || std::isnan(z) || !(nu >= 0.0) || !(z > 0.0)) {
        return std::numeric_limits<double>::quiet_NaN();
    }
    if (std::isinf(z)) {
        return 0.0;
    }
    if (std::round(nu) > static_cast<double>(besselk_detail::kMaxSteps)) {
        return std::numeric_limits<double>::quiet_NaN();
    }
    double hNuNu = 0.0;
    double hNuZ = 0.0;
    if (!bessel_k_detail::kSecondOrderIntegrals(nu, z, hNuNu, hNuZ)) {
        return std::numeric_limits<double>::quiet_NaN();
    }
    return hNuNu;
}

/**
 * @brief Mixed second derivative `d2K_nu(z)/dnudz` (`nu >= 0`, `z > 0`).
 *
 * Exact integral `-int_0^inf e^{-z cosh t} t sinh(nu t) cosh t dt` evaluated
 * with the adaptive double-exponential rule above (<= ~1e-14 relative over
 * `nu in [0,20]`, `z >= 0.01`). NaN outside the `besselK` domain or when the
 * quadrature overflows / fails to converge (never an approximation).
 */
inline double besselKNuZDeriv(double nu, double z) {
    if (!std::isfinite(nu) || std::isnan(z) || !(nu >= 0.0) || !(z > 0.0)) {
        return std::numeric_limits<double>::quiet_NaN();
    }
    if (std::isinf(z)) {
        return 0.0;
    }
    if (std::round(nu) > static_cast<double>(besselk_detail::kMaxSteps)) {
        return std::numeric_limits<double>::quiet_NaN();
    }
    double hNuNu = 0.0;
    double hNuZ = 0.0;
    if (!bessel_k_detail::kSecondOrderIntegrals(nu, z, hNuNu, hNuZ)) {
        return std::numeric_limits<double>::quiet_NaN();
    }
    return hNuZ;
}

/**
 * @brief Second z-derivative `d2K_nu(z)/dz2` (`nu >= 0`, `z > 0`).
 *
 * Assembled exactly from the modified Bessel ODE
 * `z^2 K'' + z K' - (z^2 + nu^2) K = 0`:
 * `K'' = K + (nu^2/z^2) K - K'/z`, with `K` from `besselK` and `K'` from
 * `besselKPrime` (both exact first-order kernels). NaN outside the domain.
 */
inline double besselKZZDeriv(double nu, double z) {
    if (!std::isfinite(nu) || std::isnan(z) || !(nu >= 0.0) || !(z > 0.0)) {
        return std::numeric_limits<double>::quiet_NaN();
    }
    if (std::isinf(z)) {
        return 0.0;
    }
    const double k = besselK(nu, z);
    const double prime = besselKPrime(nu, z);
    return k + (nu * nu / (z * z)) * k - prime / z;
}

} // namespace detail
} // namespace lightape

#endif // LIGHTAPE_BASE_BESSELK_H
