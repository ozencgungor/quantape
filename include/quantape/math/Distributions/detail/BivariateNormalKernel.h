#ifndef QUANTAPE_MATH_DISTRIBUTIONS_DETAIL_BIVARIATE_NORMAL_KERNEL_H
#define QUANTAPE_MATH_DISTRIBUTIONS_DETAIL_BIVARIATE_NORMAL_KERNEL_H

#include "quantape/util/Constants.h"

#include <cmath>
#include <limits>

namespace quantape::math::detail {

/**
 * @file BivariateNormalKernel.h
 * @brief Fixed-rule bivariate normal CDF/PDF kernel (standard library only).
 *
 * Phi2(a,b;rho) = P(X <= a, Y <= b) for standard normal margins and
 * correlation rho.  The CDF uses Genz's fixed-rule variant of the
 * Drezner-Wesolowsky (1989) reduction (also Genz 2004, which is the published
 * algorithm this file follows).  For |rho| < 0.925 the lower-left probability
 * comes from the Plackett identity
 *
 *     Phi2(a,b;rho) = Phi(a) Phi(b)
 *         + (1/2pi) Integral_0^{asin(rho)}
 *               exp( (a b sin t - (a^2 + b^2)/2) / cos^2 t ) dt ,
 *
 * whose integrand is carried on [0, asin(rho)] with a global symmetric
 * Gauss-Legendre rule; by |rho| the rule is the 6-, 12- or 20-point rule
 * (stored as its 3, 6 or 10 negative-half nodes and weights).  For
 * |rho| >= 0.925 the near-degenerate expansion of Drezner and Wesolowsky is
 * applied to the complementary upper tail P(X > h, Y > k):
 *
 *     P(X > h, Y > k) = Phi(-max(h,k)) - S/(2pi),
 *
 * where S combines a closed-form leading term built from
 * c = (4 - hk)/8 and d = (12 - hk)/16 (the published arrangement; the
 * d = (12 - hk)/80 convention absorbs the explicit 1/5 below into d), a
 * boundary term containing Phi(-|h-k|/sqrt(1-rho^2)), and a 12-point
 * Gauss-Legendre remainder.  The lower-left CDF is recovered through
 * Phi2(a,b;rho) = P(X > -a, Y > -b).
 *
 * Algorithm: Genz/Drezner-Wesolowsky bivariate normal approximation (see
 * Drezner & Wesolowsky 1989; Genz 2004); clean-room implementation, no source
 * copied.
 *
 * Accuracy (double): verified against 1056 mpmath references computed at 50
 * digits with two independent high-precision quadratures that agreed to
 * < 1e-46.  The observed maximum absolute error is ~1e-16 for |rho| < 0.925
 * and <= 1.4e-14 for |rho| >= 0.925 (the 12-point remainder quadrature),
 * including |rho| = 1 - 1e-15.  rho = 0, rho = +-1 and a or b = +-inf are
 * exact; NaN inputs and |rho| > 1 propagate NaN; the final value is clamped
 * to [0,1].  The header is std-only and scalar-generic: unqualified
 * exp/erfc/sin/asin/sqrt plus `using std::...` let AD scalar types supply
 * their own overloads through ADL.
 */

using ::quantape::util::kInvSqrt2;
using ::quantape::util::kInvSqrt2Pi;
using ::quantape::util::kPi;
using ::quantape::util::kSqrt2Pi;
using ::quantape::util::kTwoPi;

/// Standard normal density, scalar-generic.
template <class T>
T normalPdf(const T& x) {
    using std::exp;
    return kInvSqrt2Pi.hi * exp(-T(0.5) * x * x);
}

/// Standard normal CDF Phi(x) = 0.5 erfc(-x / sqrt(2)), scalar-generic.
template <class T>
T normalCdf(const T& x) {
    using std::erfc;
    return T(0.5) * erfc(-x * kInvSqrt2.hi);
}

/// Double pack of the two helpers above (AD-free, hot-path call sites).
inline double normalPdfDouble(double x) {
    return normalPdf(x);
}
inline double normalCdfDouble(double x) {
    return normalCdf(x);
}

/// Bivariate normal density at (a,b) with correlation rho, scalar-generic.
template <class T>
T bvnPdf(const T& a, const T& b, const T& rho) {
    using std::exp;
    using std::sqrt;
    const T s2 = (T(1) - rho) * (T(1) + rho);
    const T q = a * a - T(2) * rho * a * b + b * b;
    return exp(-q / (T(2) * s2)) / (kTwoPi.hi * sqrt(s2));
}

/// Conditional standardization pair d1 = (a - rho b)/s, d2 = (b - rho a)/s
/// (s = sqrt(1-rho^2)); provided for consumers that need the pack.
struct BvnD1D2 {
    double d1;
    double d2;
};

inline BvnD1D2 bvnD1D2(double a, double b, double rho) {
    const double s = std::sqrt((1.0 - rho) * (1.0 + rho));
    return {(a - rho * b) / s, (b - rho * a) / s};
}

/// Negative halves of the standard Gauss-Legendre rules on [-1,1] with 6, 12
/// and 20 points.  Row 0 holds the three distinct nodes of the 6-point rule,
/// row 1 the six of the 12-point rule, row 2 the ten of the 20-point rule;
/// unused entries are zero.  The values are correctly rounded doubles
/// generated at 50 digits; each row's weights sum to 1 because the mirror
/// (positive) half carries the same weights.
inline constexpr double kBvnGlHalfNodes[3][10] = {
    {-0.93246951420315203, -0.66120938646626451, -0.23861918608319691, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
     0.0},
    {-0.98156063424671925, -0.90411725637047486, -0.76990267419430469, -0.58731795428661745,
     -0.36783149899818019, -0.12523340851146892, 0.0, 0.0, 0.0, 0.0},
    {-0.99312859918509492, -0.96397192727791379, -0.91223442825132591, -0.83911697182221882,
     -0.74633190646015079, -0.63605368072651503, -0.51086700195082710, -0.37370608871541956,
     -0.22778585114164508, -0.076526521133497334},
};

inline constexpr double kBvnGlHalfWeights[3][10] = {
    {0.17132449237917035, 0.36076157304813861, 0.46791393457269105, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
     0.0},
    {0.047175336386511827, 0.10693932599531843, 0.16007832854334623, 0.20316742672306592,
     0.23349253653835481, 0.24914704581340279, 0.0, 0.0, 0.0, 0.0},
    {0.017614007139152118, 0.040601429800386941, 0.062672048334109064, 0.083276741576704749,
     0.10193011981724044, 0.11819453196151842, 0.13168863844917663, 0.14209610931838205,
     0.14917298647260375, 0.15275338713072585},
};

/// Number of stored half-nodes in each rule row.
inline constexpr int kBvnGlHalfCount[3] = {3, 6, 10};

/// Rule row used for the remainder quadrature of the near-degenerate
/// |rho| >= 0.925 branch.  The 12-point rule row keeps the remainder below
/// 1.4e-14 against high-precision references (the 20-point row would reach
/// ~1e-16 at twice the cost); the |rho| >= 0.925 accuracy budget is 1e-10.
inline constexpr int kBvnNearDegenerateRule = 1;

/// Rule row selected by |rho| (the published 6/12/20-point thresholds).
template <class T>
int bvnGlRuleIndex(const T& absRho) {
    return absRho < T(0.3) ? 0 : (absRho < T(0.75) ? 1 : 2);
}

/// Plackett correction
///     C = (1/2pi) Integral_0^{asin(rho)}
///             exp( (h k sin t - (h^2 + k^2)/2) / cos^2 t ) dt ,
/// used for |rho| < 0.925.  The integrand is sampled at the reflected pairs
/// t = (asin(rho)/2) (1 +- x_i) of the selected symmetric Gauss-Legendre
/// rule, with each stored weight used twice; the weights sum to 1 per half so
/// the correction scale is asin(rho)/(4 pi).
template <class T>
T bvnPlackettCorrection(const T& h, const T& k, const T& rho) {
    using std::asin;
    using std::exp;
    using std::sin;
    const T absRho = rho < T(0) ? -rho : rho;
    const int rule = bvnGlRuleIndex(absRho);
    const int count = kBvnGlHalfCount[rule];
    const T alpha = asin(rho);
    const T halfSumSq = (h * h + k * k) / T(2);
    const T hk = h * k;
    T sum = T(0);
    for (int i = 0; i < count; ++i) {
        const T w = T(kBvnGlHalfWeights[rule][i]);
        const T sPos = sin(alpha * (T(1) + T(kBvnGlHalfNodes[rule][i])) / T(2));
        const T sNeg = sin(alpha * (T(1) - T(kBvnGlHalfNodes[rule][i])) / T(2));
        sum += w * (exp((hk * sPos - halfSumSq) / (T(1) - sPos * sPos)) +
                    exp((hk * sNeg - halfSumSq) / (T(1) - sNeg * sNeg)));
    }
    return sum * alpha / (T(4) * T(kPi.hi));
}

/// Upper tail P(X > h, Y > k) for |rho| >= 0.925 with finite h, k and
/// |rho| < 1: the near-degenerate Drezner-Wesolowsky expansion with Genz's
/// double-precision corrections.  For rho < 0 the second margin is reflected
/// (k -> -k, hk -> -hk); this leaves the expansion unchanged and only swaps
/// the boundary term applied at the end.
template <class T>
T bvnUpperTailNearDegenerate(const T& h, const T& k, const T& rho) {
    using std::exp;
    using std::sqrt;
    T kk = k;
    T hk = h * k;
    if (rho < T(0)) {
        kk = -kk;
        hk = -hk;
    }
    const T oneMinusR2 = (T(1) - rho) * (T(1) + rho);
    const T a = sqrt(oneMinusR2);
    const T bs = (h - kk) * (h - kk);
    const T c = (T(4) - hk) / T(8);
    // Published arrangement d = (12 - hk)/16 with an explicit 1/5 factor;
    // identical to the d = (12 - hk)/80 convention with the 1/5 absorbed.
    const T d = (T(12) - hk) / T(16);
    T value = a * exp(-(bs / oneMinusR2 + hk) / T(2)) *
              (T(1) - c * (bs - oneMinusR2) * (T(1) - d * bs / T(5)) / T(3) +
               c * d * oneMinusR2 * oneMinusR2 / T(5));
    if (hk > T(-160)) {
        const T b = sqrt(bs);
        value -= exp(-hk / T(2)) * T(kSqrt2Pi.hi) * normalCdf(-b / a) * b *
                 (T(1) - c * bs * (T(1) - d * bs / T(5)) / T(3));
    }
    const T halfA = a / T(2);
    const int count = kBvnGlHalfCount[kBvnNearDegenerateRule];
    for (int i = 0; i < count; ++i) {
        const T node = T(kBvnGlHalfNodes[kBvnNearDegenerateRule][i]);
        const T w = T(kBvnGlHalfWeights[kBvnNearDegenerateRule][i]);
        const T tPos = halfA * (T(1) + node);
        const T xPos = tPos * tPos;
        const T rPos = sqrt(T(1) - xPos);
        value += halfA * w *
                 (exp(-bs / (T(2) * xPos) - hk / (T(1) + rPos)) / rPos -
                  exp(-(bs / xPos + hk) / T(2)) * (T(1) + c * xPos * (T(1) + d * xPos)));
        const T tNeg = T(1) - node;
        const T xNeg = oneMinusR2 * tNeg * tNeg / T(4);
        const T rNeg = sqrt(T(1) - xNeg);
        value += halfA * w * exp(-(bs / xNeg + hk) / T(2)) *
                 (exp(-hk * (T(1) - rNeg) / (T(2) * (T(1) + rNeg))) / rNeg -
                  (T(1) + c * xNeg * (T(1) + d * xNeg)));
    }
    value = -value / T(kTwoPi.hi);
    if (rho > T(0)) {
        value += normalCdf(-(h > kk ? h : kk));
    } else {
        const T tail = normalCdf(-h) - normalCdf(-kk);
        value = -value + (tail > T(0) ? tail : T(0));
    }
    return value;
}

/// Standard bivariate normal CDF Phi2(a,b;rho), scalar-generic.
template <class T>
T bvnCdf(const T& a, const T& b, const T& rho) {
    using std::isinf;
    using std::isnan;
    const double nan = std::numeric_limits<double>::quiet_NaN();
    if (isnan(a) || isnan(b) || isnan(rho)) {
        return T(nan);
    }
    if (isinf(rho) || rho > T(1) || rho < T(-1)) {
        return T(nan);
    }
    if (isinf(a)) {
        return a < T(0) ? T(0) : normalCdf(b);
    }
    if (isinf(b)) {
        return b < T(0) ? T(0) : normalCdf(a);
    }
    if (rho == T(0)) {
        return normalCdf(a) * normalCdf(b);
    }
    if (rho == T(1)) {
        return normalCdf(a < b ? a : b);
    }
    if (rho == T(-1)) {
        const T sum = normalCdf(a) + normalCdf(b) - T(1);
        return sum > T(0) ? sum : T(0);
    }
    // Extreme finite arguments: |a| or |b| beyond 40 already pins the joint
    // probability to a univariate limit far inside double resolution (the
    // neglected orthant mass is <= Phi(-40) ~ 7e-175), and the squared terms
    // of the near-degenerate expansion could overflow.  These shortcuts keep
    // the kernel total for every finite input.
    constexpr double kExtremeTail = 40.0;
    if (a > T(kExtremeTail)) {
        return b < T(-kExtremeTail) ? T(0) : normalCdf(b);
    }
    if (b > T(kExtremeTail)) {
        return a < T(-kExtremeTail) ? T(0) : normalCdf(a);
    }
    if (a < T(-kExtremeTail) || b < T(-kExtremeTail)) {
        return T(0);
    }
    const T absRho = rho < T(0) ? -rho : rho;
    T result = T(0);
    if (absRho < T(0.925)) {
        result = normalCdf(a) * normalCdf(b) + bvnPlackettCorrection(a, b, rho);
    } else {
        // P(X > -a, Y > -b) is the lower-left probability Phi2(a,b;rho).
        result = bvnUpperTailNearDegenerate(-a, -b, rho);
    }
    if (result < T(0)) {
        return T(0);
    }
    if (result > T(1)) {
        return T(1);
    }
    return result;
}

/// Value plus analytic gradient (ga,gb,gr) and Hessian (haa,hab,har,hbb,hbr,
/// hrr) of Phi2 with respect to (a,b,rho). Finite a,b and |rho| < 1 required.
struct BvnGradHess {
    double value;
    double ga;
    double gb;
    double gr;
    double haa;
    double hab;
    double har;
    double hbb;
    double hbr;
    double hrr;
};

inline BvnGradHess bvnCdfGradHess(double a, double b, double rho) {
    const double nan = std::numeric_limits<double>::quiet_NaN();
    if (!std::isfinite(a) || !std::isfinite(b) || !std::isfinite(rho) || std::abs(rho) >= 1.0) {
        return {nan, nan, nan, nan, nan, nan, nan, nan, nan, nan};
    }
    const double s2 = (1.0 - rho) * (1.0 + rho);
    const double s = std::sqrt(s2);
    const double za = (b - rho * a) / s;
    const double zb = (a - rho * b) / s;
    const double phia = normalPdfDouble(a);
    const double phib = normalPdfDouble(b);
    const double phiza = normalPdfDouble(za);
    const double phizb = normalPdfDouble(zb);
    const double phiPrimeA = -a * phia;
    const double phiPrimeB = -b * phib;
    const double PhiZa = normalCdfDouble(za);
    const double PhiZb = normalCdfDouble(zb);
    const double phi2 = bvnPdf(a, b, rho);

    BvnGradHess out{};
    out.value = bvnCdf(a, b, rho);
    out.ga = phia * PhiZa;
    out.gb = phib * PhiZb;
    out.gr = phi2;
    out.haa = phiPrimeA * PhiZa - phia * phiza * rho / s;
    out.hab = phia * phiza / s;
    out.har = phia * phiza * (rho * b - a) / (s2 * s);
    out.hbb = phiPrimeB * PhiZb - phib * phizb * rho / s;
    out.hbr = phib * phizb * (rho * a - b) / (s2 * s);
    out.hrr = phi2 * (rho / s2 + (a * b * (1.0 + rho * rho) - rho * (a * a + b * b)) / (s2 * s2));
    return out;
}

} // namespace quantape::math::detail

#endif // QUANTAPE_MATH_DISTRIBUTIONS_DETAIL_BIVARIATE_NORMAL_KERNEL_H
