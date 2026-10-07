#ifndef QUANTAPE_MATH_DISTRIBUTIONS_DETAIL_INCOMPLETE_BETA_H
#define QUANTAPE_MATH_DISTRIBUTIONS_DETAIL_INCOMPLETE_BETA_H

#include "quantape/math/Distributions/detail/Digamma.h"
#include "quantape/math/Distributions/detail/SecondOrderJet3.h"

#include <cmath>
#include <limits>

namespace quantape::math {
/**
 * @file IncompleteBeta.h
 * @brief Regularized incomplete beta `I_x(a,b)` and analytic derivative packs.
 *
 * `I_x(a,b) = B_x(a,b)/B(a,b)` with
 * `B_x(a,b) = int_0^x t^{a-1}(1-t)^{b-1} dt` and
 * `B(a,b) = Gamma(a)Gamma(b)/Gamma(a+b)`.
 *
 * Algorithm (Numerical Recipes, 3rd ed., section 6.4, `betacf`): the
 * regularized integral is evaluated through the modified Lentz continued
 * fraction on whichever side converges fastest,
 *
 *     I_x(a,b) = x^a (1-x)^b / (a B(a,b)) * CF(a,b,x),   x <= (a+1)/(a+b+2),
 *     I_x(a,b) = 1 - I_{1-x}(b,a)                         otherwise,
 *
 * and the front factor is carried in log space (`incBetaLogFront`), so the
 * value and its logarithm are produced independently of each other: the
 * `log` path never under/overflows for tiny `x^a (1-x)^b` products, and the
 * value path is `exp(log)`.
 *
 * Accuracy: a few 1e-15 relative for moderate shapes; the continued fraction
 * needs more terms (up to ~50) and loses a few ulp for `a,b ~ 1e3`, and the
 * `1 - I` symmetry branch has the usual cancellation only for results close
 * to 1 (never for the small tail, which is always the direct branch). The
 * `log` variants stay relative-accurate in either tail. `a <= 0`, `b <= 0`
 * or NaN parameters yield NaN; `x <= 0` yields 0 (log: `-inf`), `x >= 1`
 * yields 1 (log: 0).
 *
 * Derivatives (x-direction, parameters held fixed):
 *     d/dx   I_x(a,b) = x^{a-1}(1-x)^{b-1}/B(a,b)          (beta pdf),
 *     d2/dx2 I_x(a,b) = pdf * ((a-1)/x - (b-1)/(1-x)).
 *
 * The `*D1D2` packs return `{value, d1, d2}` with `d1` the density and `d2`
 * its first derivative, for both the pdf and the cdf entry points, matching
 * the convention used by the sibling distribution modules.
 *
 * Full parameter packs: `betaCdfGradHess` / `betaPdfGradHess` return the
 * value, gradient and Hessian in `(a, b, x)`. The parameter entries come from
 * the exact 3-jet chain rule of the continued-fraction expression
 * (`incBeta<SecondOrderJet3>` for the cdf; `lgamma`-derived digamma/trigamma
 * algebra for the density), and the x entries reproduce the existing
 * `*D1D2` arithmetic bit-for-bit (see the pack comments). No finite
 * differences are involved anywhere.
 *
 * std-only (no Boost/Eigen/AD); the `std::` math calls are made visible with
 * `using` declarations so the kernels compile for generic scalar `T`.
 */
namespace detail {

/// Value plus analytic first/second x-derivative of a distribution function.
/// Unique per module to avoid clashing with sibling `*D1D2` pack definitions.
struct BetaValueD1D2 {
    double value = 0.0;
    double d1 = 0.0;
    double d2 = 0.0;
};

/// Continued-fraction term cap. Well above the observed need (<= ~50 for
/// shapes up to 1e3); hitting it is a graceful accuracy degradation, never a
/// non-termination.
inline constexpr int kIncBetaMaxIterations = 2000;

/**
 * @brief Convergence test for the continued-fraction multiplier `delta`.
 *
 * Generic (`double`, AD scalars): the historical value comparison
 * `fabs(delta - 1) <= eps`, so existing scalar results are unchanged.
 *
 * The `SecondOrderJet3` overload additionally requires every derivative
 * component of `delta` to be below a 1e-15 floor: the value multiplier can
 * reach 1 (and the value test pass) one iteration before the parameter
 * derivative components settle — the same failure mode handled by
 * `incrementConverged` in the incomplete-gamma kernel. The floor is well
 * above jet roundoff (~1e-16) and far below the 1e-9/1e-10 reference gates.
 */
template <class T>
inline bool incBetaDeltaConverged(const T& delta, const T& eps) {
    using std::fabs;
    return fabs(delta - T(1)) <= eps;
}

/// Jet specialization of `incBetaDeltaConverged`, see the generic comment.
inline bool incBetaDeltaConverged(const SecondOrderJet3& delta,
                                  const SecondOrderJet3& eps) noexcept {
    constexpr double derivativeFloor = 1e-15;
    return std::fabs(delta.value - 1.0) <= eps.value && std::fabs(delta.dA) <= derivativeFloor &&
           std::fabs(delta.dB) <= derivativeFloor && std::fabs(delta.dX) <= derivativeFloor &&
           std::fabs(delta.dAA) <= derivativeFloor && std::fabs(delta.dAB) <= derivativeFloor &&
           std::fabs(delta.dAX) <= derivativeFloor && std::fabs(delta.dBB) <= derivativeFloor &&
           std::fabs(delta.dBX) <= derivativeFloor && std::fabs(delta.dXX) <= derivativeFloor;
}

/**
 * @brief Modified Lentz continued fraction for the incomplete beta (NR `betacf`).
 *
 * Evaluated only for `x <= (a+1)/(a+b+2)` (callers enforce the split), where
 * the three-term recurrence converges geometrically. `fpmin` floors the
 * denominators so the iteration never divides by a rounded zero; `eps` is a
 * few ulp, so the returned factor is accurate to the working precision.
 */
template <class T>
T incBetaContinuedFraction(T a, T b, T x) {
    using std::fabs;
    const T eps = T(4.0 * std::numeric_limits<double>::epsilon());
    const T fpmin = T(std::numeric_limits<double>::min()) / eps;
    const T qab = a + b;
    const T qap = a + T(1);
    const T qam = a - T(1);
    T c = T(1);
    T d = T(1) - qab * x / qap;
    if (fabs(d) < fpmin) {
        d = fpmin;
    }
    d = T(1) / d;
    T h = d;
    for (int m = 1; m <= kIncBetaMaxIterations; ++m) {
        const T mT = T(m);
        const T m2 = T(2 * m);
        T aa = mT * (b - mT) * x / ((qam + m2) * (a + m2));
        d = T(1) + aa * d;
        if (fabs(d) < fpmin) {
            d = fpmin;
        }
        c = T(1) + aa / c;
        if (fabs(c) < fpmin) {
            c = fpmin;
        }
        d = T(1) / d;
        h *= d * c;
        aa = -(a + mT) * (qab + mT) * x / ((a + m2) * (qap + m2));
        d = T(1) + aa * d;
        if (fabs(d) < fpmin) {
            d = fpmin;
        }
        c = T(1) + aa / c;
        if (fabs(c) < fpmin) {
            c = fpmin;
        }
        d = T(1) / d;
        const T del = d * c;
        h *= del;
        if (incBetaDeltaConverged(del, eps)) {
            break;
        }
    }
    return h;
}

/**
 * @brief Log of the front factor `x^a (1-x)^b / (a B(a,b))` for `I_x(a,b)`.
 *
 * `lgamma` differences and the `a log x + b log1p(-x)` product are evaluated
 * without forming `B(a,b)` or `x^a`, which under/overflow in opposite
 * directions for large shapes.
 */
template <class T>
T incBetaLogFront(T a, T b, T x) {
    using std::lgamma;
    using std::log;
    using std::log1p;
    return lgamma(a + b) - lgamma(a) - lgamma(b) + a * log(x) + b * log1p(-x) - log(a);
}

/// Direct-branch log: valid for `0 < x <= (a+1)/(a+b+2)`, any `a,b > 0`.
template <class T>
T incBetaLowLog(T a, T b, T x) {
    using std::log;
    return incBetaLogFront(a, b, x) + log(incBetaContinuedFraction(a, b, x));
}

/// Direct-branch value (`exp` of `incBetaLowLog`); underflows to 0 in the deep tail.
template <class T>
T incBetaLow(T a, T b, T x) {
    using std::exp;
    return exp(incBetaLowLog(a, b, x));
}

/**
 * @brief `log I_x(a,b)` for `a,b > 0`, evaluated on the fast side.
 *
 * In the symmetry branch the small tail is still the direct continued
 * fraction, and `log1p(-q)` handles the `q -> 0` case without cancellation,
 * so `logCdf`/`logCcdf` keep relative accuracy in both tails.
 */
template <class T>
T incBetaLog(T a, T b, T x) {
    using std::log1p;
    const T nan = T(std::numeric_limits<double>::quiet_NaN());
    const T inf = std::numeric_limits<T>::infinity();
    if (!(a > T(0)) || !(b > T(0))) {
        return nan;
    }
    if (x <= T(0)) {
        return -inf;
    }
    if (x >= T(1)) {
        return T(0);
    }
    if (x < (a + T(1)) / (a + b + T(2))) {
        return incBetaLowLog(a, b, x);
    }
    const T q = incBetaLow(b, a, T(1) - x);
    return q >= T(1) ? -inf : log1p(-q);
}

/**
 * @brief `log(1 - I_x(a,b))` for `a,b > 0`, evaluated on the accurate side.
 *
 * Mirror of `incBetaLog`: in the lower branch the complement is formed with
 * `log1p(-I_x)` so that `x -> 0` keeps `log ccdf ~ -I_x` even when `x` is far
 * below the `1 - x` rounding gap; in the upper branch the small tail
 * `I_{1-x}(b,a)` is the direct continued fraction and its log needs no
 * cancellation handling.
 */
template <class T>
T incBetaLogCcdf(T a, T b, T x) {
    using std::log1p;
    const T nan = T(std::numeric_limits<double>::quiet_NaN());
    const T inf = std::numeric_limits<T>::infinity();
    if (!(a > T(0)) || !(b > T(0))) {
        return nan;
    }
    if (x <= T(0)) {
        return T(0);
    }
    if (x >= T(1)) {
        return -inf;
    }
    if (x < (a + T(1)) / (a + b + T(2))) {
        const T q = incBetaLow(a, b, x);
        return q >= T(1) ? -inf : log1p(-q);
    }
    return incBetaLowLog(b, a, T(1) - x);
}

/**
 * @brief `1 - I_x(a,b)` for `a,b > 0`, evaluated with the small side direct.
 *
 * Mirrors `incBetaLogCcdf` in value space: the lower branch forms `1 - I_x`
 * from the direct tail, the upper branch returns `I_{1-x}(b,a)` directly, so
 * neither `x -> 0` nor `x -> 1` suffers a `1 - (1-x)` argument round trip.
 */
template <class T>
T incBetaComplement(T a, T b, T x) {
    const T nan = T(std::numeric_limits<double>::quiet_NaN());
    if (!(a > T(0)) || !(b > T(0))) {
        return nan;
    }
    if (x <= T(0)) {
        return T(1);
    }
    if (x >= T(1)) {
        return T(0);
    }
    if (x < (a + T(1)) / (a + b + T(2))) {
        const T q = incBetaLow(a, b, x);
        return q >= T(1) ? T(0) : T(1) - q;
    }
    return incBetaLow(b, a, T(1) - x);
}

/**
 * @brief Regularized incomplete beta `I_x(a,b)`.
 *
 * @param a,b positive shape parameters (`a <= 0` or `b <= 0` -> NaN)
 * @param x   evaluation point; `x <= 0` -> 0, `x >= 1` -> 1
 */
template <class T>
T incBeta(T a, T b, T x) {
    using std::exp;
    const T nan = T(std::numeric_limits<double>::quiet_NaN());
    if (!(a > T(0)) || !(b > T(0))) {
        return nan;
    }
    if (x <= T(0)) {
        return T(0);
    }
    if (x >= T(1)) {
        return T(1);
    }
    if (x < (a + T(1)) / (a + b + T(2))) {
        return exp(incBetaLowLog(a, b, x));
    }
    return T(1) - exp(incBetaLowLog(b, a, T(1) - x));
}

/**
 * @brief Log beta density `log[x^{a-1}(1-x)^{b-1}/B(a,b)]`, `x in [0,1]`.
 *
 * Singular edges are handled explicitly (`a<1` -> `+inf` at `x=0`, `a>1` ->
 * `-inf`, `a=1` -> finite); outside the support the density is 0, so the log
 * is `-inf`. Invalid parameters yield NaN.
 */
template <class T>
T betaLogPdf(T a, T b, T x) {
    using std::lgamma;
    using std::log;
    using std::log1p;
    const T nan = T(std::numeric_limits<double>::quiet_NaN());
    const T inf = std::numeric_limits<T>::infinity();
    if (!(a > T(0)) || !(b > T(0))) {
        return nan;
    }
    if (x < T(0) || x > T(1)) {
        return -inf;
    }
    T xTerm = T(0);
    if (x > T(0)) {
        xTerm = (a - T(1)) * log(x);
    } else if (a < T(1)) {
        xTerm = inf;
    } else if (a > T(1)) {
        xTerm = -inf;
    }
    T oneMinusXTerm = T(0);
    if (x < T(1)) {
        oneMinusXTerm = (b - T(1)) * log1p(-x);
    } else if (b < T(1)) {
        oneMinusXTerm = inf;
    } else if (b > T(1)) {
        oneMinusXTerm = -inf;
    }
    return xTerm + oneMinusXTerm + lgamma(a + b) - lgamma(a) - lgamma(b);
}

/// Beta pdf with `d1 = pdf`, `d2 = pdf' = pdf ((a-1)/x - (b-1)/(1-x))`.
inline BetaValueD1D2 betaPdfD1D2(double a, double b, double x) {
    using std::exp;
    const double value = exp(betaLogPdf(a, b, x));
    const double d2 = value * ((a - 1.0) / x - (b - 1.0) / (1.0 - x));
    return {value, value, d2};
}

/// Beta cdf with `value = I_x(a,b)`, `d1 = pdf`, `d2 = pdf'`.
inline BetaValueD1D2 betaCdfD1D2(double a, double b, double x) {
    const BetaValueD1D2 pdf = betaPdfD1D2(a, b, x);
    return {incBeta(a, b, x), pdf.value, pdf.d2};
}

// ---------------------------------------------------------------------------
// Full parameter-derivative packs.
//
// Method (documented decision): the `a`/`b` entries are exact derivatives of
// the *evaluated* continued-fraction expression, produced by instantiating the
// same `incBeta<T>` kernel at `T = SecondOrderJet3` with one unit derivative
// per parameter in `(a, b, x)`; `lgamma` is differentiated through
// digamma/trigamma, so no finite differences are involved anywhere. Branch
// selection follows the value component (ordinary double iteration) and the
// convergence test additionally requires the derivative components to have
// converged (`incBetaDeltaConverged`).
//
// The x entries keep the existing closed forms (`betaCdfD1D2` for the cdf;
// `(a-1)/x - (b-1)/(1-x)` algebra for the density), so `value`/`gx` (and the
// cdf `hxx`) reproduce the old packs bit-for-bit. The density pack is fully
// closed form: `lbeta` derivatives are the digamma/trigamma expressions
// `La = log x - psi(a) + psi(a+b)`, `Lb = log1p(-x) - psi(b) + psi(a+b)`,
// `Laa = psi'(a+b) - psi'(a)`, `Lab = psi'(a+b)`, `Lbb = psi'(a+b) - psi'(b)`,
// and the x/mixed entries are algebra on
// `q = (a-1)/x - (b-1)/(1-x)`.
//
// Edges: invalid `a`/`b` or NaN `x` -> all-NaN pack; outside `[0,1]` the law
// is constant (0 below, 1 above for the cdf) so every partial vanishes; at
// `x = 0` and `x = 1` the `value`/`gx`/`hxx` entries are the one-sided closed
// forms (possibly +-Inf/NaN, exactly as the old packs) while the parameter
// entries are their well-defined limits (zero: `I_x -> 0` or `1` uniformly in
// `a,b`); the mixed parameter-x one-sided limits can be singular there and
// are not represented.
// ---------------------------------------------------------------------------

/// (alpha, beta, x) value + gradient + Hessian for the Beta CDF.
struct BetaCdfGradHess {
    double value = 0.0;
    double gAlpha = 0.0;
    double gBeta = 0.0;
    double gx = 0.0;
    double hAA = 0.0;
    double hAB = 0.0;
    double hAx = 0.0;
    double hBB = 0.0;
    double hBx = 0.0;
    double hxx = 0.0;
};

/// (alpha, beta, x) value + gradient + Hessian for the Beta density.
struct BetaPdfGradHess {
    double value = 0.0;
    double gAlpha = 0.0;
    double gBeta = 0.0;
    double gx = 0.0;
    double hAA = 0.0;
    double hAB = 0.0;
    double hAx = 0.0;
    double hBB = 0.0;
    double hBx = 0.0;
    double hxx = 0.0;
};

/// All-NaN CDF pack (invalid inputs), see the section comment.
inline BetaCdfGradHess betaCdfGradHessNaN() noexcept {
    const double nan = std::numeric_limits<double>::quiet_NaN();
    return {nan, nan, nan, nan, nan, nan, nan, nan, nan, nan};
}

/// All-NaN density pack (invalid inputs), see the section comment.
inline BetaPdfGradHess betaPdfGradHessNaN() noexcept {
    const double nan = std::numeric_limits<double>::quiet_NaN();
    return {nan, nan, nan, nan, nan, nan, nan, nan, nan, nan};
}

/**
 * @brief `{F, F_a, F_b, F_x, F_aa, F_ab, F_ax, F_bb, F_bx, F_xx}` for
 * `F = I_x(a,b)`, all exact.
 *
 * `F_a, F_b` and their second derivatives come from the exact 3-jet of the
 * continued-fraction kernel at `(a, b, x)`; `F_x` and `F_xx` are the closed
 * forms of `betaCdfD1D2` (bitwise). See the section comment for the method
 * and edge behavior.
 */
inline BetaCdfGradHess betaCdfGradHess(double a, double b, double x) noexcept {
    if (!(a > 0.0) || !(b > 0.0) || x != x) {
        return betaCdfGradHessNaN();
    }
    if (x < 0.0) {
        return BetaCdfGradHess{};
    }
    if (x > 1.0) {
        BetaCdfGradHess out{};
        out.value = 1.0;
        return out;
    }
    if (x == 0.0 || x == 1.0) {
        BetaCdfGradHess out{};
        const BetaValueD1D2 old = betaCdfD1D2(a, b, x);
        out.value = old.value;
        out.gx = old.d1;
        out.hxx = old.d2;
        return out;
    }

    const SecondOrderJet3 aJet(a, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
    const SecondOrderJet3 bJet(b, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
    const SecondOrderJet3 xJet(x, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
    const SecondOrderJet3 ipack = incBeta<SecondOrderJet3>(aJet, bJet, xJet);

    const BetaValueD1D2 old = betaCdfD1D2(a, b, x);
    BetaCdfGradHess out{};
    out.value = old.value;
    out.gAlpha = ipack.dA;
    out.gBeta = ipack.dB;
    out.gx = old.d1;
    out.hAA = ipack.dAA;
    out.hAB = ipack.dAB;
    out.hAx = ipack.dAX;
    out.hBB = ipack.dBB;
    out.hBx = ipack.dBX;
    out.hxx = old.d2;
    return out;
}

/**
 * @brief `{f, f_a, f_b, f_x, f_aa, f_ab, f_ax, f_bb, f_bx, f_xx}` for the
 * `Beta(a,b)` density, all closed form.
 *
 * With `L = log f = (a-1) log x + (b-1) log1p(-x) - log B(a,b)` and the
 * digamma/trigamma expressions of the section comment:
 *   `f_a = f La`, `f_b = f Lb`, `f_x = f q`,
 *   `f_aa = f (La^2 + Laa)`, `f_ab = f (La Lb + Lab)`,
 *   `f_ax = f (La q + 1/x)`, `f_bb = f (Lb^2 + Lbb)`,
 *   `f_bx = f (Lb q - 1/(1-x))`,
 *   `f_xx = f (q^2 - (a-1)/x^2 - (b-1)/(1-x)^2)`.
 * `value` and `gx` reproduce `betaPdfD1D2`'s `value` and `d2` bit-for-bit.
 * See the section comment for edge behavior.
 */
inline BetaPdfGradHess betaPdfGradHess(double a, double b, double x) noexcept {
    using std::exp;
    using std::log;
    using std::log1p;
    if (!(a > 0.0) || !(b > 0.0) || x != x) {
        return betaPdfGradHessNaN();
    }
    if (x < 0.0 || x > 1.0) {
        return BetaPdfGradHess{};
    }
    if (x == 0.0 || x == 1.0) {
        BetaPdfGradHess out{};
        const BetaValueD1D2 old = betaPdfD1D2(a, b, x);
        out.value = old.value;
        out.gx = old.d2;
        out.hxx = std::numeric_limits<double>::quiet_NaN();
        return out;
    }

    const double pdf = exp(betaLogPdf(a, b, x));
    const double q = (a - 1.0) / x - (b - 1.0) / (1.0 - x);
    const double psiA = digammaDouble(a);
    const double psiB = digammaDouble(b);
    const double psiSum = digammaDouble(a + b);
    const double triA = trigammaDouble(a);
    const double triB = trigammaDouble(b);
    const double triSum = trigammaDouble(a + b);
    const double la = log(x) - psiA + psiSum;
    const double lb = log1p(-x) - psiB + psiSum;

    BetaPdfGradHess out{};
    out.value = pdf;
    out.gAlpha = pdf * la;
    out.gBeta = pdf * lb;
    out.gx = pdf * q;
    out.hAA = pdf * (la * la + triSum - triA);
    out.hAB = pdf * (la * lb + triSum);
    out.hAx = pdf * (la * q + 1.0 / x);
    out.hBB = pdf * (lb * lb + triSum - triB);
    out.hBx = pdf * (lb * q - 1.0 / (1.0 - x));
    out.hxx = pdf * (q * q - (a - 1.0) / (x * x) - (b - 1.0) / ((1.0 - x) * (1.0 - x)));
    return out;
}

} // namespace detail

/**
 * @brief Convenience alias for the shared incomplete-beta kernel.
 *
 * The implementation lives in `detail`; this forwarder lets callers in
 * `quantape::math` use the NR-style name directly (`incBeta(a,b,x)`).
 */
template <class T>
T incBeta(T a, T b, T x) {
    return detail::incBeta(a, b, x);
}

} // namespace quantape::math

#endif // QUANTAPE_MATH_DISTRIBUTIONS_DETAIL_INCOMPLETE_BETA_H
