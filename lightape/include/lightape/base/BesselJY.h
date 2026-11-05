#ifndef LIGHTAPE_BASE_BESSELJY_H
#define LIGHTAPE_BASE_BESSELJY_H

/**
 * @file BesselJY.h
 * @brief Ordinary Bessel functions J_nu(z), Y_nu(z), real order, z > 0:
 *        values, z-derivatives (recurrence) and order derivatives
 *
 * Companion of base/BesselI.h (same in-house kernel discipline); std-only.
 *
 * Algorithms:
 *  - Ascending series (the fixed method inside `|z| < 15 + |nu|`):
 *      J_nu(z) = (z/2)^nu sum_{k>=0} (-z^2/4)^k / (k! (nu+1)_k),
 *    with the Pochhammer product iterated (`t_{k+1} = t_k (-z^2/4) /
 *    ((k+1)(nu+k+1))`). The (z/2)^nu / Gamma(nu+1) prefactor is evaluated as
 *    `exp(nu log(z/2) - lgamma(nu+1))` with the sign of Gamma(nu+1) applied
 *    separately, and the final product is guarded in log space (never
 *    `exp(logPref)` alone, so `exp` overflow cannot produce NaN through
 *    `inf * 0`). Negative integer orders reflect with J_{-n} = (-1)^n J_n.
 *  - Asymptotic (DLMF 10.17.3/10.17.4, DLMF 10.17.5 coefficients):
 *      J_nu(z) ~ sqrt(2/(pi z)) [ cos(w) P(nu,z) - sin(w) Q(nu,z) ],
 *      Y_nu(z) ~ sqrt(2/(pi z)) [ sin(w) P(nu,z) + cos(w) Q(nu,z) ],
 *      w = z - nu pi/2 - pi/4,
 *    P = sum_k (-1)^k a_{2k}/z^{2k}, Q = sum_k (-1)^k a_{2k+1}/z^{2k+1},
 *    a_k = prod_{j=1}^k (4 nu^2 - (2j-1)^2) / (k! 8^k).
 *    P and Q are truncated at the term of smallest second-order jet
 *    magnitude (max of |value|, |d/dnu|, |d2/dnu2| over the term, see
 *    base/BesselI.h's BesselSecondOrderJet) over a 400-term scan: the
 *    expansions diverge past their minimum, and the term magnitudes are not
 *    unimodal for large order (they can rise, fall through many orders and
 *    only then diverge, so stopping at the first non-decreasing term was
 *    far too early -- at (|nu|, z) = (15, 60) the first term is ~1.7 while
 *    the smallest is near k = 61). Jet-based truncation keeps the recorded
 *    AD value and its order derivatives derivatives of the same finite
 *    polynomial. Terms used grow like ~z/2 at moderate order (J_0(25) keeps
 *    26 P / 26 Q terms, J_0(15) 16/16, J_0.5(20) 21/21 -- the latter with
 *    exactly zero value corrections, since a_k(1/2) = 0 for k >= 2).
 *  - Method choice beyond the series band: the series loses digits to
 *    cancellation once z exceeds ~|nu| (its largest term is ~e^z/(2 pi z)
 *    while |J_nu| <= 1), and the asymptotic only reaches double precision
 *    once z is well past |nu| (z >= 15 + |nu| suffices for |nu| <= ~6, but
 *    for larger order the crossover moves much farther out). Three methods
 *    are therefore ordered as:
 *      bridge (integrals, see below)   when z > max(8, 0.4|nu| + 6)
 *                                      and z < 30 + 2|nu|, |nu| <= 24;
 *      ascending series              for z < 15 + |nu| below the bridge;
 *      asymptotic otherwise (z >= 15 + |nu| outside the bridge), with the
 *      residual comparison (`exp(logPref) max|t_k| * 4e-16` against
 *      `sqrt(2/(pi z)) (|cos w| errP + |sin w| errQ)`) deciding against the
 *      series there. For |nu| > 24 inside the series band the previous
 *      behaviour is unchanged (the bridge's order limit is 24).
 *    Without the bridge, e.g. the series at (|nu|, z) = (10, 40) would be
 *    O(1) wrong and the previous gap cells (10, 25) ~1e-7, (10, 30) ~2.6e-5,
 *    (20, 50) O(1) were only as good as the better expansion there.
 *  - Integral bridge (DLMF 10.9.2 / 10.9.7, real nu, z > 0):
 *      J_nu(z) = (1/pi) int_0^pi cos(nu th - z sin th) dth
 *                - (sin(nu pi)/pi) int_0^inf e^{-z sinh t - nu t} dt,
 *      Y_nu(z) = (1/pi) int_0^pi sin(z sin th - nu th) dth
 *                - (1/pi) int_0^inf (e^{nu t} + cos(nu pi) e^{-nu t})
 *                  e^{-z sinh t} dt.
 *    The theta integrals use one Gauss-Legendre panel on [0, pi] with
 *    `n = 64 + 8 ceil((z + 2|nu|)/16)` nodes from the cached ladder
 *    {64, 80, 96, 112, 128, 160, 192} (all eight sums share one node loop).
 *    The t integrals are evaluated directly in t with 64-point
 *    Gauss-Legendre on [0, T], T the fixed point of
 *    `z sinh T = 80 + |nu| T` (integrand below e^{-80}; an u = z sinh t
 *    substitution was rejected because the u-integrand's branch point sits
 *    at distance z, which ruins Gauss-Laguerre convergence at small z).
 *    Both integrands are entire in their variables, so the panels converge
 *    spectrally. See the bridge block below for the differentiated kernels.
 *  - Y_nu: integer order n uses DLMF 10.8.1,
 *      Y_n(z) = -(1/pi)(z/2)^{-n} sum_{k=0}^{n-1} (n-k-1)!/k! (z^2/4)^k
 *               + (2/pi) ln(z/2) J_n(z)
 *               - (1/pi)(z/2)^n sum_{k>=0}
 *                 (psi(k+1)+psi(n+k+1)) (-z^2/4)^k / (k! (n+k)!),
 *    with the 1/n! of the third sum's k = 0 term carried in the
 *    `exp(n log(z/2) - lgamma(n+1))` prefactor; negative integers reflect
 *    with Y_{-n} = (-1)^n Y_n. Non-integer order uses
 *    Y_nu = (cos(nu pi) J_nu - J_{-nu}) / sin(nu pi) below the bridge.
 *
 * Order derivatives (nu-active use):
 *  - J series: termwise differentiation of the same series. With
 *    `d log(t_k)/dnu = log(z/2) - psi(nu+k+1)`,
 *      dJ/dnu  = J log(z/2) - sum t_k psi(nu+k+1),
 *      d2J/dnu2 = J log(z/2)^2 - 2 log(z/2) sum t_k psi_k
 *                 + sum t_k (psi_k^2 - psi'_k),
 *    psi/psi' extended to negative non-integer arguments by reflection
 *    (`psi(x) = psi(1-x) - pi cot(pi x)`,
 *    `psi'(x) = pi^2/sin^2(pi x) - psi'(1-x)`). At non-positive integer
 *    orders the prefactored series has a pole; the kernels return NaN
 *    (J_nu itself is entire in nu, but the pole limit is not taken).
 *  - Y non-integer: the reflection formula differentiated analytically with
 *    the J order derivatives at +/- nu.
 *  - Integer orders: the reflection form is 0/0. In the bridge band (and,
 *    for |nu| <= 24, anywhere in the ascending-series region) Y's order
 *    derivatives switch to the differentiated integral kernels, whose
 *    theta/t integrands are smooth in nu at integer orders; the entry
 *    points return the finite limit there. Non-positive integer J order
 *    derivatives remain NaN outside the bridge (the prefactored series has
 *    a pole and the pole limit is not taken), but are finite inside it.
 *  - Asymptotic region (both): P and Q are differentiated exactly as finite
 *    polynomials in nu (the BesselSecondOrderJet above) and the phase
 *    contributes its closed-form nu derivatives (dw/dnu = -pi/2). There the
 *    integer-order Y kernels are finite (the reflection 0/0 is bypassed).
 *
 * Domain / accuracy notes (honest measured limits; double precision):
 *  - `z > 0` required (real branches); z = +inf -> 0; NaN in -> NaN out.
 *  - Series iteration cap 100000 (large |nu| outside the guard can exhaust
 *    it; the partial sum is returned, as in base/BesselI.h).
 *  - The series prefactor overflow guard only covers the leading prefactor;
 *    for extreme |nu| (|nu| >~ 2000 near the series band edge) the iterated
 *    terms themselves may overflow or underflow. Order derivatives use the
 *    unprefactored term accumulation and share that limit.
 *  - Series-band cancellation is intrinsic: on a 90-point grid
 *    (nu in [-3.7, 10], z in [0.1, 40]) the worst relative errors were
 *    ~4e-11 (J values) and ~1e-8 (integer Y values, n = 10 at z = 20); the
 *    bridge band now replaces those cells, so on that grid (and on a dense
 *    nu in [-6, 20] step 0.5, z in [0.05, 60] grid) the worst J/Y values
 *    outside the documented small-z reflection regime are ~6e-13 (J at
 *    (13.5, 50)) and ~9e-13 (Y at (4, 16), where |Y| = 7.3e-5 and the two
 *    integral terms cancel; away from Y's zeros Y stays <= ~4e-13).
 *  - The former transition gap is closed for |nu| <= 24: on the dense grid
 *    the listed cells are (10, 20) <= 6e-15, (10, 25) <= 6e-15,
 *    (10, 30) <= 3e-15 and (20, 50) <= 7e-15 relative (previously ~1e-7,
 *    ~2.6e-5 and O(1) at the last three). The floor near a zero of J or Y
 *    is the double-precision cancellation of the two integral terms; the
 *    bridge is still the best of the three methods there. For |nu| > 24 the
 *    series/asymptotic structure is unchanged (series inside z < 15 + |nu|,
 *    residual comparison beyond), but the asymptotic now truncates at its
 *    true minimum term (see above), which fixes e.g. the (14..16, 60) cells
 *    that the old first-non-decreasing stop truncated after k = 1.
 *  - Integer-order Y order derivatives: finite for |nu| <= 24 (bridge).
 *    Measured against mpmath `diff` at n in {0, 1, 5}, z in {0.5, 3, 20}:
 *    dY/dnu <= 3e-14 and d2Y/dnu2 <= 7e-13 relative (the latter at
 *    (20, 20)). The bridge's t integrand may overflow double range for very
 *    small z with large integer order (the true derivative overflows there
 *    too); the value then is +-inf rather than NaN.
 *  - Non-integer Y loses digits like 1/|sin(nu pi)| near integers below the
 *    bridge (use the integer order there); the bridge removes that loss
 *    inside its band. For negative non-integer orders at small z the
 *    reflection formula's `cos(nu pi) J_nu` term against the large J_{-nu}
 *    limits relative accuracy (e.g. ~1e-9 for Y' at nu = -2.5, z = 0.1);
 *    exact half-integer orders would reduce to Y_nu = (+-1) J_{-nu}, but
 *    the generic reflection branch is kept. The bridge cannot recover that
 *    regime below its band either (the theta integral has the same
 *    cancellation), so it stays documented.
 */

#include <cmath>
#include <cstddef>
#include <limits>
#include <vector>

#include "lightape/base/BesselI.h"

namespace lightape {
namespace besseljy_detail {

using detail::BesselSecondOrderJet;
using detail::digammaDouble;
using detail::maxAbsComponent;
using detail::trigammaDouble;
using detail::bessel_detail::kPi;

/// Series iteration cap (mirrors base/BesselI.h).
inline constexpr int kMaxSeriesTerms = 100000;
/// pi/2 (asymptotic phase derivative).
inline constexpr double kHalfPi = 1.5707963267948966;
/// log(DBL_MAX): the direct-prefactor cutoff in the log-space guard.
inline constexpr double kLogPrefDirect = 700.0;

inline double nan() {
    return std::numeric_limits<double>::quiet_NaN();
}

/// psi(x) for real x; non-integer x <= 0 by reflection, poles -> NaN.
inline double signedDigamma(double x) {
    if (x > 0.0) {
        return digammaDouble(x);
    }
    if (x == std::floor(x)) {
        return nan();
    }
    return digammaDouble(1.0 - x) - kPi / std::tan(kPi * x);
}

/// psi'(x) for real x; non-integer x <= 0 by reflection, poles -> NaN.
inline double signedTrigamma(double x) {
    if (x > 0.0) {
        return trigammaDouble(x);
    }
    if (x == std::floor(x)) {
        return nan();
    }
    const double s = std::sin(kPi * x);
    return kPi * kPi / (s * s) - trigammaDouble(1.0 - x);
}

/// Scaled series sum (no prefactor) plus the largest term magnitude, used
/// both for the value and for the cancellation-error estimate.
struct JYSeriesScaled {
    double sum = 0.0;
    double maxTerm = 0.0;
};

/// Terms `t_k = (-z^2/4)^k / (k! (nu+1)_k)` of the J series, `z > 0`.
inline JYSeriesScaled jSeriesScaled(double nu, double z) {
    const double z24 = 0.25 * z * z;
    double term = 1.0;
    double sum = 1.0;
    double maxTerm = 1.0;
    for (int k = 1; k <= kMaxSeriesTerms; ++k) {
        term *= -z24 / (static_cast<double>(k) * (nu + static_cast<double>(k)));
        sum += term;
        maxTerm = std::fmax(maxTerm, std::fabs(term));
        if (std::fabs(term) <= std::fabs(sum) * 1e-17) {
            break;
        }
    }
    return {sum, maxTerm};
}

/// J_nu(z) series value, `nu` not a non-positive integer, `z > 0`.
inline double jSeriesValue(double nu, double z) {
    const double lz = std::log(0.5 * z);
    const double gammaSign = std::tgamma(nu + 1.0) < 0.0 ? -1.0 : 1.0;
    const double logPref = nu * lz - std::lgamma(nu + 1.0);
    const double sum = jSeriesScaled(nu, z).sum;
    if (logPref <= kLogPrefDirect) {
        return gammaSign * std::exp(logPref) * sum;
    }
    // exp(logPref) alone would overflow; combine magnitudes in log space so a
    // small sum can still bring the product back into range.
    const double sumSign = sum < 0.0 ? -1.0 : 1.0;
    return gammaSign * sumSign * std::exp(logPref + std::log(std::fabs(sum)));
}

/// Series value plus first/second order derivatives of J_nu(z).
struct JYSeriesDerivs {
    double value = 0.0;
    double dnu = 0.0;
    double dnu2 = 0.0;
};

/// `nu` not a non-positive integer, `z > 0`. Accumulates the prefactored
/// terms (base/BesselI.h `orderSeries` pattern) with the psi/psi' weights of
/// the exact termwise differentiation; see the file header for the formulas.
inline JYSeriesDerivs jSeriesDerivs(double nu, double z) {
    const double lz = std::log(0.5 * z);
    const double gammaSign = std::tgamma(nu + 1.0) < 0.0 ? -1.0 : 1.0;
    double term = gammaSign * std::exp(nu * lz - std::lgamma(nu + 1.0));
    const double psi0 = signedDigamma(nu + 1.0);
    JYSeriesDerivs r;
    r.value = term;
    double s1 = term * psi0;
    double s2 = term * (psi0 * psi0 - signedTrigamma(nu + 1.0));
    const double z24 = 0.25 * z * z;
    for (int k = 1; k <= kMaxSeriesTerms; ++k) {
        term *= -z24 / (static_cast<double>(k) * (nu + static_cast<double>(k)));
        const double psi = signedDigamma(nu + static_cast<double>(k) + 1.0);
        r.value += term;
        s1 += term * psi;
        s2 += term * (psi * psi - signedTrigamma(nu + static_cast<double>(k) + 1.0));
        if (std::fabs(term) <= std::fabs(r.value) * 1e-17) {
            break;
        }
    }
    r.dnu = r.value * lz - s1;
    r.dnu2 = r.value * lz * lz - 2.0 * lz * s1 + s2;
    return r;
}

/// Asymptotic accuracy guard: within `|z| < 15 + |nu|` the series is the
/// specified method; beyond that boundary the two residual estimates below
/// pick the expansion with the smaller error.
inline bool inSeriesBand(double nu, double z) {
    return z < 15.0 + std::fabs(nu);
}

/// Truncated DLMF 10.17.3 P and Q with their first/second nu derivatives,
/// carried as second-order forward jets in nu (base/BesselI.h). `errP`/`errQ`
/// are the jet magnitudes of the first term that was *not* added, the
/// standard asymptotic residual proxy (the truncated tail is dominated by
/// that term).
struct JYAsymPQ {
    BesselSecondOrderJet p;
    BesselSecondOrderJet q;
    double errP = 0.0;
    double errQ = 0.0;
};

/// Prefix sum of one asymptotic correction series with jet-based optimal
/// truncation. The term magnitudes are *not* unimodal for large nu: they can
/// rise, fall through many orders, cross a near-zero factor and only then
/// diverge, so stopping at the first non-decreasing term truncates far too
/// early (e.g. (nu, z) = (15, 60), where the first term is already ~1.7 and
/// the smallest term is near k = 61). The scan therefore takes the global
/// minimum of the second-order jet magnitude over the same 400-term budget
/// as before, then re-accumulates the prefix up to it; `err` receives the
/// jet magnitude of the first omitted term. Jet-based truncation keeps the
/// recorded AD value and its order derivatives those of the same finite
/// polynomial.
template <class TermRatio>
BesselSecondOrderJet truncatedAsymSeries(const BesselSecondOrderJet& initial, TermRatio nextTerm,
                                         double& err) {
    int bestIndex = 0;
    double bestMag = maxAbsComponent(initial);
    BesselSecondOrderJet term = initial;
    for (int k = 1; k <= 400; ++k) {
        term = nextTerm(term, k);
        const double mag = maxAbsComponent(term);
        if (mag < bestMag) {
            bestMag = mag;
            bestIndex = k;
        }
    }
    BesselSecondOrderJet sum = initial;
    term = initial;
    for (int k = 1; k <= bestIndex; ++k) {
        term = nextTerm(term, k);
        sum += term;
    }
    err = maxAbsComponent(nextTerm(term, bestIndex + 1));
    return sum;
}

inline JYAsymPQ asymPQ(double nu, double z) {
    const BesselSecondOrderJet nuj(nu, 1.0, 0.0, 0.0, 0.0, 0.0);
    const BesselSecondOrderJet muj = BesselSecondOrderJet(4.0) * nuj * nuj;
    const double z2 = z * z;
    // P = sum_{k>=0} (-1)^k a_{2k}/z^{2k}, p_0 = 1,
    // p_k/p_{k-1} = -(mu-(4k-3)^2)(mu-(4k-1)^2)/((2k-1)(2k) 64 z^2).
    double errP = 0.0;
    double errQ = 0.0;
    const BesselSecondOrderJet pSum = truncatedAsymSeries(
        BesselSecondOrderJet(1.0),
        [&](const BesselSecondOrderJet& prev, int k) {
            const double kd = static_cast<double>(k);
            const double a = 4.0 * kd - 3.0;
            const double b = 4.0 * kd - 1.0;
            return prev *
                   ((muj - BesselSecondOrderJet(a * a)) * (muj - BesselSecondOrderJet(b * b))) *
                   BesselSecondOrderJet(-1.0 / ((2.0 * kd - 1.0) * (2.0 * kd) * 64.0 * z2));
        },
        errP);
    // Q = sum_{k>=0} (-1)^k a_{2k+1}/z^{2k+1}, q_0 = (mu-1)/(8z),
    // q_k/q_{k-1} = -(mu-(4k-1)^2)(mu-(4k+1)^2)/((2k)(2k+1) 64 z^2).
    const BesselSecondOrderJet q0 =
        (muj - BesselSecondOrderJet(1.0)) * BesselSecondOrderJet(1.0 / (8.0 * z));
    const BesselSecondOrderJet qSum = truncatedAsymSeries(
        q0,
        [&](const BesselSecondOrderJet& prev, int k) {
            const double kd = static_cast<double>(k);
            const double a = 4.0 * kd - 1.0;
            const double b = 4.0 * kd + 1.0;
            return prev *
                   ((muj - BesselSecondOrderJet(a * a)) * (muj - BesselSecondOrderJet(b * b))) *
                   BesselSecondOrderJet(-1.0 / ((2.0 * kd) * (2.0 * kd + 1.0) * 64.0 * z2));
        },
        errQ);
    return {pSum, qSum, errP, errQ};
}

/// sqrt(2/(pi z)) times the phase pair (cos w, sin w) at the asymptotic
/// argument `w = z - nu pi/2 - pi/4`.
struct JYAsymPhase {
    double pref = 0.0;
    double c = 0.0;
    double s = 0.0;
};

inline JYAsymPhase asymPhase(double nu, double z) {
    const double omega = z - 0.25 * kPi - 0.5 * nu * kPi;
    return {std::sqrt(2.0 / (kPi * z)), std::cos(omega), std::sin(omega)};
}

/// `true` = evaluate with the ascending series, `false` = asymptotic.
///
/// Within `|z| < 15 + |nu|` the series is the specified method. Beyond that
/// boundary both methods can be poor (the series loses digits to cancellation
/// once z exceeds ~|nu|, the asymptotic only converges as z grows well past
/// |nu|), so the smaller of two residual estimates wins:
///  - series: about `exp(logPref) max|t_k| * 4e-16`, a few ulps of the
///    largest *prefactored* term (the `(z/2)^nu / Gamma(nu+1)` prefactor is
///    carried; log space saturates extreme cases to +inf),
///  - asymptotic: `sqrt(2/(pi z)) (|cos w| errP + |sin w| errQ)` with the
///    first omitted correction at the series' true minimum as the
///    tail-dominating residual.
inline bool preferSeries(double nu, double z) {
    if (inSeriesBand(nu, z)) {
        return true;
    }
    const JYAsymPQ pq = asymPQ(nu, z);
    const JYAsymPhase w = asymPhase(nu, z);
    const double asymResidual = w.pref * (std::fabs(w.c) * pq.errP + std::fabs(w.s) * pq.errQ);
    // The series error is a few ulps of its largest *prefactored* term; the
    // (z/2)^nu / Gamma(nu+1) prefactor does not cancel against the asymptotic
    // residual, so it must be carried here. Log space saturates the extreme
    // cases to +inf (asymptotic wins).
    const double maxTerm = jSeriesScaled(nu, z).maxTerm;
    const double logPref = nu * std::log(0.5 * z) - std::lgamma(nu + 1.0);
    const double logResidual = logPref + std::log(maxTerm) + std::log(4e-16);
    const double seriesResidual =
        logResidual > 700.0 ? std::numeric_limits<double>::infinity() : std::exp(logResidual);
    return seriesResidual <= asymResidual;
}

// ── Integral-representation bridge (DLMF 10.9.2 / 10.9.7) ──
//
// The transition gap between the ascending series and the asymptotic
// expansion is bridged with the integral representations (real nu, z > 0)
//
//   J_nu(z) = (1/pi) int_0^pi cos(nu th - z sin th) dth
//             - (sin(nu pi)/pi) int_0^inf e^{-z sinh t - nu t} dt,
//   Y_nu(z) = (1/pi) int_0^pi sin(z sin th - nu th) dth
//             - (1/pi) int_0^inf (e^{nu t} + cos(nu pi) e^{-nu t})
//               e^{-z sinh t} dt.
//
// The order/z derivatives below are obtained by differentiating under the
// integral sign (no finite differences); the theta and t integrals are
// smooth in nu at integer orders, so the integer-order Y order derivatives
// stay finite (the reflection form is 0/0 there).
//
// Quadrature:
//  - theta integrals: one Gauss-Legendre panel on [0, pi]. The node count is
//    sized from the phase variation `z + 2|nu|`:
//    `n = 64 + 8 ceil((z + 2|nu|)/16)`, rounded up to the cached ladder
//    {64, 80, 96, 112, 128, 160, 192} (at most 128 for the documented band
//    |nu| <= 24, z < 30 + 2|nu|). The rule is built once per node count by
//    Newton iteration on P_n and all eight sums share one node loop.
//  - t integrals: u = z sinh t substitution would put the branch point at
//    distance z from the real axis (slow Gauss-Laguerre convergence at small
//    z), so the integrals are evaluated directly in t with 64-point
//    Gauss-Legendre on [0, T]. T solves `z sinh T = 80 + |nu| T` (so the
//    integrand has decayed below e^{-80}; the fixed point converges in a few
//    iterations). The integrand is entire in t, so the panel converges
//    spectrally; 64 nodes cover the documented band.
//
// Accuracy: in the band the bridge holds <= ~1e-13 relative on dense grids
// for values and order derivatives (closer to 1e-14 outside near-zero
// cells); inherited limits are documented in the file header.

/// One-panel Gauss-Legendre rule on [-1, 1] with ascending nodes.
struct GaussLegendreRule {
    std::vector<double> nodes;
    std::vector<double> weights;
};

/// `n`-node Gauss-Legendre rule on [-1, 1], built with the Newton scheme of
/// Numerical Recipes `gauleg` (initial guess cos(pi (i-1/4)/(n+1/2))), with
/// P'_n recomputed at the converged node so the outermost weights keep full
/// relative accuracy.
inline GaussLegendreRule buildGaussLegendreRule(int n) {
    GaussLegendreRule rule;
    rule.nodes.assign(static_cast<std::size_t>(n), 0.0);
    rule.weights.assign(static_cast<std::size_t>(n), 0.0);
    const int half = (n + 1) / 2;
    const double nd = static_cast<double>(n);
    for (int i = 1; i <= half; ++i) {
        double z = std::cos(kPi * (static_cast<double>(i) - 0.25) / (nd + 0.5));
        for (int iter = 0; iter < 100; ++iter) {
            double p1 = 1.0;
            double p2 = 0.0;
            for (int j = 1; j <= n; ++j) {
                const double p3 = p2;
                p2 = p1;
                p1 = ((2.0 * j - 1.0) * z * p2 - (j - 1.0) * p3) / static_cast<double>(j);
            }
            const double pp = nd * (z * p1 - p2) / (z * z - 1.0);
            const double step = p1 / pp;
            z -= step;
            if (std::fabs(step) <= 4e-16 * (1.0 + std::fabs(z))) {
                break;
            }
        }
        double p1 = 1.0;
        double p2 = 0.0;
        for (int j = 1; j <= n; ++j) {
            const double p3 = p2;
            p2 = p1;
            p1 = ((2.0 * j - 1.0) * z * p2 - (j - 1.0) * p3) / static_cast<double>(j);
        }
        const double pp = nd * (z * p1 - p2) / (z * z - 1.0);
        const double w = 2.0 / ((1.0 - z * z) * pp * pp);
        rule.nodes[static_cast<std::size_t>(i - 1)] = -z;
        rule.nodes[static_cast<std::size_t>(n - i)] = z;
        rule.weights[static_cast<std::size_t>(i - 1)] = w;
        rule.weights[static_cast<std::size_t>(n - i)] = w;
    }
    return rule;
}

/// Cached Gauss-Legendre ladders, built once at first use (magic static,
/// immutable afterwards; concurrent reads are safe). Theta rules are picked
/// per call, the t rule is fixed at 64 nodes.
struct JYQuadratureTables {
    static constexpr int kThetaSizes[7] = {64, 80, 96, 112, 128, 160, 192};
    std::vector<GaussLegendreRule> theta;
    GaussLegendreRule t;
};

inline const JYQuadratureTables& jyQuadratureTables() {
    static const JYQuadratureTables tables = [] {
        JYQuadratureTables built;
        for (const int size : JYQuadratureTables::kThetaSizes) {
            built.theta.push_back(buildGaussLegendreRule(size));
        }
        built.t = buildGaussLegendreRule(64);
        return built;
    }();
    return tables;
}

/// Theta node count sized from the phase variation `z + 2|nu|`, rounded up to
/// the cached ladder. Returns the ladder index.
inline int thetaRuleIndex(double nu, double z) {
    const double need = 64.0 + 8.0 * std::ceil((z + 2.0 * std::fabs(nu)) / 16.0);
    for (int i = 0; i < 7; ++i) {
        if (static_cast<double>(JYQuadratureTables::kThetaSizes[i]) >= need) {
            return i;
        }
    }
    return 6;
}

/// Finishing time of the t integrals: the fixed point of
/// `z sinh T = 80 + |nu| T`, where the integrand has decayed past e^{-80}.
inline double bridgeTMax(double nu, double z) {
    const double target = 80.0;
    double t = std::asinh(target / z) + 1.0;
    for (int iter = 0; iter < 60; ++iter) {
        const double next = std::asinh((target + std::fabs(nu) * t) / z);
        if (next == t) {
            break;
        }
        t = next;
    }
    return t;
}

/// Value, order derivatives and z derivatives of J_nu and Y_nu from the
/// differentiated integral representations above; one quadrature pass fills
/// all eight entries. `z > 0`, real `nu` (the theta/t integrands are smooth
/// through integer orders).
struct JYBridge {
    double j = 0.0;
    double jNu = 0.0;
    double jNu2 = 0.0;
    double jZ = 0.0;
    double y = 0.0;
    double yNu = 0.0;
    double yNu2 = 0.0;
    double yZ = 0.0;
};

inline JYBridge jyBridge(double nu, double z) {
    const JYQuadratureTables& tables = jyQuadratureTables();
    const GaussLegendreRule& thetaRule = tables.theta[thetaRuleIndex(nu, z)];

    // Theta integrals:
    //   ic0 = int cos(phi),   is0 = int sin(phi),   phi = nu th - z sin th,
    //   ic1/is1 = the same with a th weight, ic2/is2 with th^2,
    //   icz/isz = the same with a sin(th) weight (phi - z derivatives).
    double ic0 = 0.0;
    double is0 = 0.0;
    double ic1 = 0.0;
    double is1 = 0.0;
    double ic2 = 0.0;
    double is2 = 0.0;
    double icz = 0.0;
    double isz = 0.0;
    const double halfPi = 0.5 * kPi;
    for (std::size_t k = 0; k < thetaRule.nodes.size(); ++k) {
        const double theta = halfPi * (thetaRule.nodes[k] + 1.0);
        const double wq = halfPi * thetaRule.weights[k];
        const double phase = nu * theta - z * std::sin(theta);
        const double c = std::cos(phase);
        const double s = std::sin(phase);
        const double st = std::sin(theta);
        ic0 += wq * c;
        is0 += wq * s;
        ic1 += wq * theta * c;
        is1 += wq * theta * s;
        ic2 += wq * theta * theta * c;
        is2 += wq * theta * theta * s;
        icz += wq * c * st;
        isz += wq * s * st;
    }

    // t integrals:
    //   tp0/tm0 = int e^{-z sinh t} e^{+-nu t} dt,
    //   tp1/tm1 with a t weight, tp2/tm2 with t^2,
    //   tps/tms with a sinh(t) weight (the z derivatives).
    const double tMax = bridgeTMax(nu, z);
    const GaussLegendreRule& tRule = tables.t;
    double tp0 = 0.0;
    double tm0 = 0.0;
    double tp1 = 0.0;
    double tm1 = 0.0;
    double tp2 = 0.0;
    double tm2 = 0.0;
    double tps = 0.0;
    double tms = 0.0;
    const double halfT = 0.5 * tMax;
    for (std::size_t k = 0; k < tRule.nodes.size(); ++k) {
        const double t = halfT * (tRule.nodes[k] + 1.0);
        const double wq = halfT * tRule.weights[k];
        const double sh = std::sinh(t);
        const double envelope = wq * std::exp(-z * sh);
        const double ep = envelope * std::exp(nu * t);
        const double em = envelope * std::exp(-nu * t);
        tp0 += ep;
        tm0 += em;
        tp1 += ep * t;
        tm1 += em * t;
        tp2 += ep * t * t;
        tm2 += em * t * t;
        tps += ep * sh;
        tms += em * sh;
    }

    // sin(nu pi)/cos(nu pi) are set exactly at integer orders: the floating
    // sin(kPi * n) residue (~1e-16) would otherwise multiply large t
    // integrals and leak into the smooth integer-order limit.
    const bool integerOrder = nu == std::floor(nu);
    const double sinNuPi = integerOrder ? 0.0 : std::sin(kPi * nu);
    const double cosNuPi =
        integerOrder ? (std::fmod(std::fabs(nu), 2.0) == 1.0 ? -1.0 : 1.0) : std::cos(kPi * nu);

    JYBridge r;
    r.j = ic0 / kPi - sinNuPi * tm0 / kPi;
    r.jNu = -is1 / kPi - cosNuPi * tm0 + sinNuPi * tm1 / kPi;
    r.jNu2 = -ic2 / kPi + kPi * sinNuPi * tm0 + 2.0 * cosNuPi * tm1 - sinNuPi * tm2 / kPi;
    r.jZ = isz / kPi + sinNuPi * tms / kPi;
    r.y = -is0 / kPi - (tp0 + cosNuPi * tm0) / kPi;
    r.yNu = -ic1 / kPi - (tp1 - kPi * sinNuPi * tm0 - cosNuPi * tm1) / kPi;
    r.yNu2 = is2 / kPi -
             (tp2 - kPi * kPi * cosNuPi * tm0 + 2.0 * kPi * sinNuPi * tm1 + cosNuPi * tm2) / kPi;
    r.yZ = icz / kPi + (tps + cosNuPi * tms) / kPi;
    return r;
}

/// Documented bridge band: `|nu| <= 24` and
/// `max(8, 0.4|nu| + 6) < z < 30 + 2|nu|`. Below the band the ascending
/// series is accurate (the 0.4|nu| + 6 edge keeps the integer-Y series
/// cancellation under ~1e-12); above it the asymptotic expansion is.
inline bool inBridgeBand(double nu, double z) {
    const double anu = std::fabs(nu);
    return anu <= 24.0 && z > std::fmax(8.0, 0.4 * anu + 6.0) && z < 30.0 + 2.0 * anu;
}

} // namespace besseljy_detail

// ── J ──

/**
 * @brief Ordinary Bessel function of the first kind J_nu(z), real order,
 * `z > 0`; NaN outside the domain, 0 at z = +inf.
 */
inline double besselJ(double nu, double z) {
    if (!std::isfinite(nu) || std::isnan(z) || !(z > 0.0)) {
        return besseljy_detail::nan();
    }
    if (std::isinf(z)) {
        return 0.0;
    }
    if (nu < 0.0 && nu == std::floor(nu)) {
        const double sign = std::fmod(-nu, 2.0) == 0.0 ? 1.0 : -1.0;
        return sign * besselJ(-nu, z);
    }
    if (besseljy_detail::inBridgeBand(nu, z)) {
        return besseljy_detail::jyBridge(nu, z).j;
    }
    if (!besseljy_detail::preferSeries(nu, z)) {
        const besseljy_detail::JYAsymPQ pq = besseljy_detail::asymPQ(nu, z);
        const besseljy_detail::JYAsymPhase w = besseljy_detail::asymPhase(nu, z);
        return w.pref * (w.c * pq.p.value - w.s * pq.q.value);
    }
    return besseljy_detail::jSeriesValue(nu, z);
}

namespace besseljy_detail {

/// DLMF 10.8.1 integer-order series, `nD >= 0` integral, `z > 0`.
inline double yInteger(double nD, double z) {
    const int n = static_cast<int>(nD);
    const double halfZ = 0.5 * z;
    const double lz = std::log(halfZ);
    double value = 0.0;
    if (n > 0) {
        // f_0 = (n-1)! (z/2)^{-n}; f_k = f_{k-1} z^2 / (4 (n-k) k).
        double f = std::exp(std::lgamma(nD) - nD * lz);
        double sum = f;
        const double z24 = 0.25 * z * z;
        for (int k = 1; k <= n - 1; ++k) {
            f *= z24 / ((nD - static_cast<double>(k)) * static_cast<double>(k));
            sum += f;
        }
        value = -sum / kPi;
    }
    value += (2.0 / kPi) * lz * besselJ(nD, z);
    // Third sum with the 1/n! of its k = 0 term pulled into the prefactor:
    // t_k = (-z^2/4)^k / (k! (n+1)_k), t_0 = 1,
    // t_k/t_{k-1} = -z^2/(4 k (n+k)).
    double t = 1.0;
    double third = 0.0;
    const double z24 = 0.25 * z * z;
    for (int k = 0; k <= kMaxSeriesTerms; ++k) {
        third += (signedDigamma(static_cast<double>(k) + 1.0) +
                  signedDigamma(nD + static_cast<double>(k) + 1.0)) *
                 t;
        if (std::fabs(t) <= std::fabs(third) * 1e-17) {
            break;
        }
        t *= -z24 / ((static_cast<double>(k) + 1.0) * (nD + static_cast<double>(k) + 1.0));
    }
    const double logPref = nD * lz - std::lgamma(nD + 1.0);
    value -= third * std::exp(logPref) / kPi;
    return value;
}

/// Non-integer reflection branch, `z > 0`, `sin(nu pi) != 0`.
inline double yReflection(double nu, double z) {
    const double s = std::sin(kPi * nu);
    const double c = std::cos(kPi * nu);
    return (besselJ(nu, z) * c - besselJ(-nu, z)) / s;
}

} // namespace besseljy_detail

/**
 * @brief Ordinary Bessel function of the second kind Y_nu(z), real order,
 * `z > 0`; NaN outside the domain, 0 at z = +inf.
 */
inline double besselY(double nu, double z) {
    if (!std::isfinite(nu) || std::isnan(z) || !(z > 0.0)) {
        return besseljy_detail::nan();
    }
    if (std::isinf(z)) {
        return 0.0;
    }
    if (nu < 0.0 && nu == std::floor(nu)) {
        const double sign = std::fmod(-nu, 2.0) == 0.0 ? 1.0 : -1.0;
        return sign * besselY(-nu, z);
    }
    if (besseljy_detail::inBridgeBand(nu, z)) {
        return besseljy_detail::jyBridge(nu, z).y;
    }
    if (!besseljy_detail::preferSeries(nu, z)) {
        const besseljy_detail::JYAsymPQ pq = besseljy_detail::asymPQ(nu, z);
        const besseljy_detail::JYAsymPhase w = besseljy_detail::asymPhase(nu, z);
        return w.pref * (w.s * pq.p.value + w.c * pq.q.value);
    }
    if (nu == std::floor(nu)) {
        return besseljy_detail::yInteger(nu, z);
    }
    return besseljy_detail::yReflection(nu, z);
}

// ── z-derivatives (integral bridge in the band, recurrence elsewhere) ──

/// `J'_nu(z) = J_{nu-1}(z) - (nu/z) J_nu(z)`; the bridge's differentiated
/// theta/t integrals are used in the band (the recurrence remains the
/// regression cross-check there).
inline double besselJPrime(double nu, double z) {
    if (!std::isfinite(nu) || std::isnan(z) || !(z > 0.0)) {
        return besseljy_detail::nan();
    }
    if (std::isinf(z)) {
        return 0.0;
    }
    if (besseljy_detail::inBridgeBand(nu, z)) {
        return besseljy_detail::jyBridge(nu, z).jZ;
    }
    return besselJ(nu - 1.0, z) - nu * besselJ(nu, z) / z;
}

/// `Y'_nu(z) = Y_{nu-1}(z) - (nu/z) Y_nu(z)` (same recurrence as J).
inline double besselYPrime(double nu, double z) {
    if (!std::isfinite(nu) || std::isnan(z) || !(z > 0.0)) {
        return besseljy_detail::nan();
    }
    if (std::isinf(z)) {
        return 0.0;
    }
    if (besseljy_detail::inBridgeBand(nu, z)) {
        return besseljy_detail::jyBridge(nu, z).yZ;
    }
    return besselY(nu - 1.0, z) - nu * besselY(nu, z) / z;
}

// ── order derivatives ──

/**
 * @brief `dJ_nu(z)/dnu`.
 *
 * Exact termwise differentiation in the series region for every real `nu`
 * except the non-positive integers (poles of the prefactored series -> NaN);
 * in the asymptotic region the truncated P/Q polynomials are differentiated
 * exactly in `nu`; in the bridge band the differentiated integral kernels
 * are smooth for every real order (integer included).
 */
inline double besselJNuDeriv(double nu, double z) {
    if (!std::isfinite(nu) || std::isnan(z) || !(z > 0.0)) {
        return besseljy_detail::nan();
    }
    if (std::isinf(z)) {
        return 0.0;
    }
    if (besseljy_detail::inBridgeBand(nu, z)) {
        return besseljy_detail::jyBridge(nu, z).jNu;
    }
    if (!besseljy_detail::preferSeries(nu, z)) {
        const besseljy_detail::JYAsymPQ pq = besseljy_detail::asymPQ(nu, z);
        const besseljy_detail::JYAsymPhase w = besseljy_detail::asymPhase(nu, z);
        return w.pref * (w.c * pq.p.d1 - w.s * pq.q.d1 +
                         besseljy_detail::kHalfPi * (w.s * pq.p.value + w.c * pq.q.value));
    }
    if (nu < 0.0 && nu == std::floor(nu)) {
        return besseljy_detail::nan();
    }
    return besseljy_detail::jSeriesDerivs(nu, z).dnu;
}

/**
 * @brief `d2J_nu(z)/dnu2` (same domain as `besselJNuDeriv`).
 */
inline double besselJNuDeriv2(double nu, double z) {
    if (!std::isfinite(nu) || std::isnan(z) || !(z > 0.0)) {
        return besseljy_detail::nan();
    }
    if (std::isinf(z)) {
        return 0.0;
    }
    if (besseljy_detail::inBridgeBand(nu, z)) {
        return besseljy_detail::jyBridge(nu, z).jNu2;
    }
    if (!besseljy_detail::preferSeries(nu, z)) {
        const besseljy_detail::JYAsymPQ pq = besseljy_detail::asymPQ(nu, z);
        const besseljy_detail::JYAsymPhase w = besseljy_detail::asymPhase(nu, z);
        const double pi2over4 = 0.25 * besseljy_detail::kPi * besseljy_detail::kPi;
        return w.pref * (-pi2over4 * (w.c * pq.p.value - w.s * pq.q.value) +
                         besseljy_detail::kPi * (w.s * pq.p.d1 + w.c * pq.q.d1) + w.c * pq.p.d11 -
                         w.s * pq.q.d11);
    }
    if (nu < 0.0 && nu == std::floor(nu)) {
        return besseljy_detail::nan();
    }
    return besseljy_detail::jSeriesDerivs(nu, z).dnu2;
}

namespace besseljy_detail {

/// Reflection-formula order derivatives shared by the Y kernels. All J
/// kernels are evaluated at the primal `(nu, z)`; at integer `nu` the caller
/// must not enter this path (0/0).
struct YReflectionDerivs {
    double value = 0.0;
    double dnu = 0.0;
    double dnu2 = 0.0;
};

inline YReflectionDerivs yReflectionDerivs(double nu, double z) {
    const double s = std::sin(kPi * nu);
    const double c = std::cos(kPi * nu);
    const double jn = besselJ(nu, z);
    const double jmn = besselJ(-nu, z);
    const double fn = besselJNuDeriv(nu, z);
    const double fmn = besselJNuDeriv(-nu, z);
    const double gn = besselJNuDeriv2(nu, z);
    const double gmn = besselJNuDeriv2(-nu, z);
    const double a = jn * c - jmn;
    const double ad = fn * c - kPi * s * jn + fmn;
    const double add = gn * c - 2.0 * kPi * s * fn - kPi * kPi * c * jn - gmn;
    YReflectionDerivs r;
    r.value = a / s;
    r.dnu = ad / s - a * kPi * c / (s * s);
    r.dnu2 = add / s - 2.0 * kPi * c * ad / (s * s) + kPi * kPi * a / s +
             2.0 * kPi * kPi * c * c * a / (s * s * s);
    return r;
}

} // namespace besseljy_detail

/**
 * @brief `dY_nu(z)/dnu`.
 *
 * Non-integer order: reflection formula differentiated with the J order
 * derivatives; asymptotic region: the truncated P/Q polynomials
 * differentiated exactly in `nu`; in the bridge band every real order (and,
 * for `|nu| <= 24` in the series region, every integer order) uses the
 * differentiated integral kernels: the theta/t integrands are smooth in
 * `nu` at integer orders, so the 0/0 reflection form is bypassed and the
 * derivative is finite there.
 */
inline double besselYNuDeriv(double nu, double z) {
    if (!std::isfinite(nu) || std::isnan(z) || !(z > 0.0)) {
        return besseljy_detail::nan();
    }
    if (std::isinf(z)) {
        return 0.0;
    }
    if (besseljy_detail::inBridgeBand(nu, z)) {
        return besseljy_detail::jyBridge(nu, z).yNu;
    }
    if (!besseljy_detail::preferSeries(nu, z)) {
        const besseljy_detail::JYAsymPQ pq = besseljy_detail::asymPQ(nu, z);
        const besseljy_detail::JYAsymPhase w = besseljy_detail::asymPhase(nu, z);
        return w.pref * (w.s * pq.p.d1 + w.c * pq.q.d1 +
                         besseljy_detail::kHalfPi * (w.s * pq.q.value - w.c * pq.p.value));
    }
    if (nu == std::floor(nu)) {
        if (std::fabs(nu) <= 24.0) {
            return besseljy_detail::jyBridge(nu, z).yNu;
        }
        return besseljy_detail::nan();
    }
    return besseljy_detail::yReflectionDerivs(nu, z).dnu;
}

/**
 * @brief `d2Y_nu(z)/dnu2` (same domain as `besselYNuDeriv`).
 */
inline double besselYNuDeriv2(double nu, double z) {
    if (!std::isfinite(nu) || std::isnan(z) || !(z > 0.0)) {
        return besseljy_detail::nan();
    }
    if (std::isinf(z)) {
        return 0.0;
    }
    if (besseljy_detail::inBridgeBand(nu, z)) {
        return besseljy_detail::jyBridge(nu, z).yNu2;
    }
    if (!besseljy_detail::preferSeries(nu, z)) {
        const besseljy_detail::JYAsymPQ pq = besseljy_detail::asymPQ(nu, z);
        const besseljy_detail::JYAsymPhase w = besseljy_detail::asymPhase(nu, z);
        const double pi2over4 = 0.25 * besseljy_detail::kPi * besseljy_detail::kPi;
        return w.pref * (w.s * pq.p.d11 + w.c * pq.q.d11 +
                         besseljy_detail::kPi * (w.s * pq.q.d1 - w.c * pq.p.d1) -
                         pi2over4 * (w.s * pq.p.value + w.c * pq.q.value));
    }
    if (nu == std::floor(nu)) {
        if (std::fabs(nu) <= 24.0) {
            return besseljy_detail::jyBridge(nu, z).yNu2;
        }
        return besseljy_detail::nan();
    }
    return besseljy_detail::yReflectionDerivs(nu, z).dnu2;
}

} // namespace lightape

#endif // LIGHTAPE_BASE_BESSELJY_H
