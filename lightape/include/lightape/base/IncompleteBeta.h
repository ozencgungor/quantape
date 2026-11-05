#ifndef LIGHTAPE_BASE_INCOMPLETE_BETA_H
#define LIGHTAPE_BASE_INCOMPLETE_BETA_H

/**
 * @file IncompleteBeta.h
 * @brief Regularized incomplete beta (double) and its exact (a, b, x)
 *        value/gradient/Hessian pack for the lightape AD layer
 *
 * In-house port of quantape::math::detail's incomplete-beta kernel and the
 * `betaCdfGradHess` parameter pack (same owner; see
 * quantape/math/Distributions/detail/IncompleteBeta.h). Evaluation order is
 * preserved from the source.
 *
 * `I_x(a,b) = B_x(a,b)/B(a,b)` with `B_x(a,b) = int_0^x t^{a-1}(1-t)^{b-1} dt`
 * and `B(a,b) = Gamma(a)Gamma(b)/Gamma(a+b)`.
 *
 * Algorithm (Numerical Recipes, 3rd ed., section 6.4, `betacf`): the
 * regularized integral is evaluated through the modified Lentz continued
 * fraction on whichever side converges fastest,
 *
 *     I_x(a,b) = x^a (1-x)^b / (a B(a,b)) * CF(a,b,x),   x <= (a+1)/(a+b+2),
 *     I_x(a,b) = 1 - I_{1-x}(b,a)                         otherwise,
 *
 * with the front factor carried in log space (`incBetaLogFront`) so the value
 * never under/overflows for tiny `x^a (1-x)^b` products.
 *
 * The parameter derivatives are exact derivatives of the *evaluated*
 * continued-fraction expression, produced by instantiating the same
 * `incBeta<T>` kernel at a local three-variable, second-order forward jet
 * whose `lgamma` is differentiated through `digammaDouble`/`trigammaDouble`;
 * the x entries keep the closed forms (`I_x = exp(log pdf)`,
 * `I_xx = pdf ((a-1)/x - (b-1)/(1-x))`). No finite differences anywhere;
 * branch selection follows the value component and the convergence test
 * additionally requires the derivative components to have converged.
 *
 * `a <= 0`, `b <= 0` or NaN parameters yield NaN; `x <= 0` yields 0,
 * `x >= 1` yields 1. `betaLogPdfDouble` handles the singular edges exactly
 * (a<1 -> +inf at x=0, a>1 -> -inf, a=1 -> finite). std-only, no allocation
 * (the pack is noexcept).
 */

#include <cmath>
#include <limits>

#include "lightape/base/Digamma.h"

namespace lightape {
namespace detail {

// ---------------------------------------------------------------------------
// Three-variable, second-order forward jet: exact local differentiation of
// the incomplete-beta kernel without an AD engine. Port of quantape's
// SecondOrderJet3 (same owner). Component order
// (value, dA, dB, dX, dAA, dAB, dAX, dBB, dBX, dXX).
// ---------------------------------------------------------------------------

struct BetaSecondOrderJet3 {
    double value = 0.0;
    double dA = 0.0;
    double dB = 0.0;
    double dX = 0.0;
    double dAA = 0.0;
    double dAB = 0.0;
    double dAX = 0.0;
    double dBB = 0.0;
    double dBX = 0.0;
    double dXX = 0.0;

    BetaSecondOrderJet3() = default;
    // Implicit on purpose: scalar literals inside the generic kernels.
    BetaSecondOrderJet3(double v) : value(v) {} // NOLINT(google-explicit-constructor)
    BetaSecondOrderJet3(double v, double firstA, double firstB, double firstX, double secondAA,
                        double secondAB, double secondAX, double secondBB, double secondBX,
                        double secondXX)
        : value(v), dA(firstA), dB(firstB), dX(firstX), dAA(secondAA), dAB(secondAB), dAX(secondAX),
          dBB(secondBB), dBX(secondBX), dXX(secondXX) {}
};

inline BetaSecondOrderJet3 operator+(const BetaSecondOrderJet3& x,
                                     const BetaSecondOrderJet3& y) noexcept {
    return {x.value + y.value, x.dA + y.dA,   x.dB + y.dB,   x.dX + y.dX,   x.dAA + y.dAA,
            x.dAB + y.dAB,     x.dAX + y.dAX, x.dBB + y.dBB, x.dBX + y.dBX, x.dXX + y.dXX};
}

inline BetaSecondOrderJet3 operator-(const BetaSecondOrderJet3& x,
                                     const BetaSecondOrderJet3& y) noexcept {
    return {x.value - y.value, x.dA - y.dA,   x.dB - y.dB,   x.dX - y.dX,   x.dAA - y.dAA,
            x.dAB - y.dAB,     x.dAX - y.dAX, x.dBB - y.dBB, x.dBX - y.dBX, x.dXX - y.dXX};
}

inline BetaSecondOrderJet3 operator-(const BetaSecondOrderJet3& x) noexcept {
    return {-x.value, -x.dA, -x.dB, -x.dX, -x.dAA, -x.dAB, -x.dAX, -x.dBB, -x.dBX, -x.dXX};
}

inline BetaSecondOrderJet3 operator*(const BetaSecondOrderJet3& x,
                                     const BetaSecondOrderJet3& y) noexcept {
    return {x.value * y.value,
            x.dA * y.value + x.value * y.dA,
            x.dB * y.value + x.value * y.dB,
            x.dX * y.value + x.value * y.dX,
            x.dAA * y.value + 2.0 * x.dA * y.dA + x.value * y.dAA,
            x.dAB * y.value + x.dA * y.dB + x.dB * y.dA + x.value * y.dAB,
            x.dAX * y.value + x.dA * y.dX + x.dX * y.dA + x.value * y.dAX,
            x.dBB * y.value + 2.0 * x.dB * y.dB + x.value * y.dBB,
            x.dBX * y.value + x.dB * y.dX + x.dX * y.dB + x.value * y.dBX,
            x.dXX * y.value + 2.0 * x.dX * y.dX + x.value * y.dXX};
}

inline BetaSecondOrderJet3 operator/(const BetaSecondOrderJet3& x,
                                     const BetaSecondOrderJet3& y) noexcept {
    const double inv = 1.0 / y.value;
    const double w = x.value * inv;
    const double wA = (x.dA - w * y.dA) * inv;
    const double wB = (x.dB - w * y.dB) * inv;
    const double wX = (x.dX - w * y.dX) * inv;
    return {w,
            wA,
            wB,
            wX,
            (x.dAA - w * y.dAA - 2.0 * wA * y.dA) * inv,
            (x.dAB - w * y.dAB - wA * y.dB - wB * y.dA) * inv,
            (x.dAX - w * y.dAX - wA * y.dX - wX * y.dA) * inv,
            (x.dBB - w * y.dBB - 2.0 * wB * y.dB) * inv,
            (x.dBX - w * y.dBX - wB * y.dX - wX * y.dB) * inv,
            (x.dXX - w * y.dXX - 2.0 * wX * y.dX) * inv};
}

inline BetaSecondOrderJet3& operator+=(BetaSecondOrderJet3& x,
                                       const BetaSecondOrderJet3& y) noexcept {
    x = x + y;
    return x;
}

inline BetaSecondOrderJet3& operator-=(BetaSecondOrderJet3& x,
                                       const BetaSecondOrderJet3& y) noexcept {
    x = x - y;
    return x;
}

inline BetaSecondOrderJet3& operator*=(BetaSecondOrderJet3& x,
                                       const BetaSecondOrderJet3& y) noexcept {
    x = x * y;
    return x;
}

inline BetaSecondOrderJet3& operator/=(BetaSecondOrderJet3& x,
                                       const BetaSecondOrderJet3& y) noexcept {
    x = x / y;
    return x;
}

inline bool operator==(const BetaSecondOrderJet3& x, const BetaSecondOrderJet3& y) noexcept {
    return x.value == y.value;
}

inline bool operator!=(const BetaSecondOrderJet3& x, const BetaSecondOrderJet3& y) noexcept {
    return x.value != y.value;
}

inline bool operator<(const BetaSecondOrderJet3& x, const BetaSecondOrderJet3& y) noexcept {
    return x.value < y.value;
}

inline bool operator<=(const BetaSecondOrderJet3& x, const BetaSecondOrderJet3& y) noexcept {
    return x.value <= y.value;
}

inline bool operator>(const BetaSecondOrderJet3& x, const BetaSecondOrderJet3& y) noexcept {
    return x.value > y.value;
}

inline bool operator>=(const BetaSecondOrderJet3& x, const BetaSecondOrderJet3& y) noexcept {
    return x.value >= y.value;
}

/// `fabs`: sign flip on all derivative components (a.e. derivative; `+` at 0).
inline BetaSecondOrderJet3 fabs(const BetaSecondOrderJet3& x) noexcept {
    const double s = x.value < 0.0 ? -1.0 : 1.0;
    return {std::fabs(x.value), s * x.dA,  s * x.dB,  s * x.dX,  s * x.dAA,
            s * x.dAB,          s * x.dAX, s * x.dBB, s * x.dBX, s * x.dXX};
}

/// `max(|value|, |dA|, ..., |dXX|)`: the convergence scale used by the
/// continued-fraction iteration test (see `incBetaDeltaConverged`).
inline double maxAbsComponent(const BetaSecondOrderJet3& x) noexcept {
    return std::fmax(std::fmax(std::fmax(std::fabs(x.value), std::fabs(x.dA)),
                               std::fmax(std::fabs(x.dB), std::fabs(x.dX))),
                     std::fmax(std::fmax(std::fabs(x.dAA), std::fabs(x.dAB)),
                               std::fmax(std::fmax(std::fabs(x.dAX), std::fabs(x.dBB)),
                                         std::fmax(std::fabs(x.dBX), std::fabs(x.dXX)))));
}

/// `log`: value exact, derivatives by logarithmic differentiation.
inline BetaSecondOrderJet3 log(const BetaSecondOrderJet3& x) noexcept {
    const double inv = 1.0 / x.value;
    const double lA = x.dA * inv;
    const double lB = x.dB * inv;
    const double lX = x.dX * inv;
    return {std::log(x.value),
            lA,
            lB,
            lX,
            x.dAA * inv - lA * lA,
            x.dAB * inv - lA * lB,
            x.dAX * inv - lA * lX,
            x.dBB * inv - lB * lB,
            x.dBX * inv - lB * lX,
            x.dXX * inv - lX * lX};
}

/// `exp`: `d^n exp = exp * (chain rule)`, same rule as `expm1`.
inline BetaSecondOrderJet3 exp(const BetaSecondOrderJet3& x) noexcept {
    const double e = std::exp(x.value);
    return {e,
            e * x.dA,
            e * x.dB,
            e * x.dX,
            e * (x.dAA + x.dA * x.dA),
            e * (x.dAB + x.dA * x.dB),
            e * (x.dAX + x.dA * x.dX),
            e * (x.dBB + x.dB * x.dB),
            e * (x.dBX + x.dB * x.dX),
            e * (x.dXX + x.dX * x.dX)};
}

/// `expm1`: derivatives are those of `exp`, the value uses `std::expm1`.
inline BetaSecondOrderJet3 expm1(const BetaSecondOrderJet3& x) noexcept {
    const double e = std::exp(x.value);
    return {std::expm1(x.value),
            e * x.dA,
            e * x.dB,
            e * x.dX,
            e * (x.dAA + x.dA * x.dA),
            e * (x.dAB + x.dA * x.dB),
            e * (x.dAX + x.dA * x.dX),
            e * (x.dBB + x.dB * x.dB),
            e * (x.dBX + x.dB * x.dX),
            e * (x.dXX + x.dX * x.dX)};
}

/// `log1p`: derivatives of `log(1 + x)`.
inline BetaSecondOrderJet3 log1p(const BetaSecondOrderJet3& x) noexcept {
    const double inv = 1.0 / (1.0 + x.value);
    const double lA = x.dA * inv;
    const double lB = x.dB * inv;
    const double lX = x.dX * inv;
    return {std::log1p(x.value),
            lA,
            lB,
            lX,
            x.dAA * inv - lA * lA,
            x.dAB * inv - lA * lB,
            x.dAX * inv - lA * lX,
            x.dBB * inv - lB * lB,
            x.dBX * inv - lB * lX,
            x.dXX * inv - lX * lX};
}

/// `lgamma`: `d/da lgamma = psi(a)`, `d^2/da^2 = psi'(a)` (Digamma.h).
inline BetaSecondOrderJet3 lgamma(const BetaSecondOrderJet3& x) noexcept {
    const double psi = digammaDouble(x.value);
    const double tri = trigammaDouble(x.value);
    return {std::lgamma(x.value),
            psi * x.dA,
            psi * x.dB,
            psi * x.dX,
            tri * x.dA * x.dA + psi * x.dAA,
            tri * x.dA * x.dB + psi * x.dAB,
            tri * x.dA * x.dX + psi * x.dAX,
            tri * x.dB * x.dB + psi * x.dBB,
            tri * x.dB * x.dX + psi * x.dBX,
            tri * x.dX * x.dX + psi * x.dXX};
}

// ---------------------------------------------------------------------------
// Generic kernel helpers (double and the local jet).
// ---------------------------------------------------------------------------

/// Continued-fraction term cap. Well above the observed need (<= ~50 for
/// shapes up to 1e3); hitting it is a graceful accuracy degradation, never a
/// non-termination.
inline constexpr int kIncBetaMaxIterations = 2000;

/// Generic convergence test for the continued-fraction multiplier `delta`:
/// the historical value comparison `fabs(delta - 1) <= eps`.
template <class T>
bool incBetaDeltaConverged(const T& delta, const T& eps) {
    using std::fabs;
    return fabs(delta - T(1)) <= eps;
}

/// Jet specialization: the value multiplier can reach 1 (and the value test
/// pass) one iteration before the parameter derivative components settle, so
/// every derivative component must additionally be below a 1e-15 floor.
inline bool incBetaDeltaConverged(const BetaSecondOrderJet3& delta,
                                  const BetaSecondOrderJet3& eps) noexcept {
    constexpr double derivativeFloor = 1e-15;
    return std::fabs(delta.value - 1.0) <= eps.value && std::fabs(delta.dA) <= derivativeFloor &&
           std::fabs(delta.dB) <= derivativeFloor && std::fabs(delta.dX) <= derivativeFloor &&
           std::fabs(delta.dAA) <= derivativeFloor && std::fabs(delta.dAB) <= derivativeFloor &&
           std::fabs(delta.dAX) <= derivativeFloor && std::fabs(delta.dBB) <= derivativeFloor &&
           std::fabs(delta.dBX) <= derivativeFloor && std::fabs(delta.dXX) <= derivativeFloor;
}

/**
 * @brief Modified Lentz continued fraction for the incomplete beta (NR
 * `betacf`).
 *
 * Evaluated only for `x <= (a+1)/(a+b+2)` (callers enforce the split), where
 * the three-term recurrence converges geometrically. `fpmin` floors the
 * denominators so the iteration never divides by a rounded zero.
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

/// Canonical double entry.
inline double incBetaDouble(double a, double b, double x) {
    return incBeta<double>(a, b, x);
}

/**
 * @brief Log beta density `log[x^{a-1}(1-x)^{b-1}/B(a,b)]`, `x in [0,1]`.
 *
 * Singular edges are handled explicitly (`a<1` -> `+inf` at `x=0`, `a>1` ->
 * `-inf`, `a=1` -> finite); outside the support the density is 0, so the log
 * is `-inf`. Invalid parameters yield NaN.
 */
inline double betaLogPdfDouble(double a, double b, double x) {
    using std::lgamma;
    using std::log;
    using std::log1p;
    const double nan = std::numeric_limits<double>::quiet_NaN();
    const double inf = std::numeric_limits<double>::infinity();
    if (!(a > 0.0) || !(b > 0.0)) {
        return nan;
    }
    if (x < 0.0 || x > 1.0) {
        return -inf;
    }
    double xTerm = 0.0;
    if (x > 0.0) {
        xTerm = (a - 1.0) * log(x);
    } else if (a < 1.0) {
        xTerm = inf;
    } else if (a > 1.0) {
        xTerm = -inf;
    }
    double oneMinusXTerm = 0.0;
    if (x < 1.0) {
        oneMinusXTerm = (b - 1.0) * log1p(-x);
    } else if (b < 1.0) {
        oneMinusXTerm = inf;
    } else if (b > 1.0) {
        oneMinusXTerm = -inf;
    }
    return xTerm + oneMinusXTerm + lgamma(a + b) - lgamma(a) - lgamma(b);
}

/// Beta pdf with `d1 = pdf`, `d2 = pdf' = pdf ((a-1)/x - (b-1)/(1-x))`.
struct BetaValueD1D2 {
    double value = 0.0;
    double d1 = 0.0;
    double d2 = 0.0;
};

inline BetaValueD1D2 betaPdfD1D2(double a, double b, double x) {
    using std::exp;
    const double value = exp(betaLogPdfDouble(a, b, x));
    const double d2 = value * ((a - 1.0) / x - (b - 1.0) / (1.0 - x));
    return {value, value, d2};
}

/// Beta cdf with `value = I_x(a,b)`, `d1 = pdf`, `d2 = pdf'`.
inline BetaValueD1D2 betaCdfD1D2(double a, double b, double x) {
    const BetaValueD1D2 pdf = betaPdfD1D2(a, b, x);
    return {incBetaDouble(a, b, x), pdf.value, pdf.d2};
}

// ---------------------------------------------------------------------------
// Exact value/gradient/Hessian pack in (a, b, x) for the `External` wrapper.
// Mirrors the source `betaCdfGradHess`; the x entries keep the closed forms,
// the parameter entries come from the 3-jet (see the file comment). Edges:
// invalid/NaN -> all-NaN; outside [0,1] the law is constant, every partial
// vanishes; at `x = 0` / `x = 1` the parameter entries are their well-defined
// zero limits and the value/gX/hXX entries are the one-sided closed forms.
// ---------------------------------------------------------------------------

struct BetaGradHess {
    double value = 0.0;
    double gA = 0.0;
    double gB = 0.0;
    double gX = 0.0;
    double hAA = 0.0;
    double hAB = 0.0;
    double hAX = 0.0;
    double hBB = 0.0;
    double hBX = 0.0;
    double hXX = 0.0;
};

/// All-NaN pack (invalid inputs), see the section comment.
inline BetaGradHess betaGradHessNaN() noexcept {
    const double nan = std::numeric_limits<double>::quiet_NaN();
    return {nan, nan, nan, nan, nan, nan, nan, nan, nan, nan};
}

/// `{I, I_a, I_b, I_x, I_aa, I_ab, I_ax, I_bb, I_bx, I_xx}` for `I = I_x(a,b)`.
inline BetaGradHess betaGradHess(double a, double b, double x) noexcept {
    if (!(a > 0.0) || !(b > 0.0) || x != x) {
        return betaGradHessNaN();
    }
    if (x < 0.0) {
        return BetaGradHess{};
    }
    if (x > 1.0) {
        BetaGradHess out{};
        out.value = 1.0;
        return out;
    }
    if (x == 0.0 || x == 1.0) {
        BetaGradHess out{};
        const BetaValueD1D2 old = betaCdfD1D2(a, b, x);
        out.value = old.value;
        out.gX = old.d1;
        out.hXX = old.d2;
        return out;
    }

    const BetaSecondOrderJet3 aJet(a, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
    const BetaSecondOrderJet3 bJet(b, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
    const BetaSecondOrderJet3 xJet(x, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
    const BetaSecondOrderJet3 ipack = incBeta<BetaSecondOrderJet3>(aJet, bJet, xJet);

    const BetaValueD1D2 old = betaCdfD1D2(a, b, x);
    BetaGradHess out{};
    out.value = old.value;
    out.gA = ipack.dA;
    out.gB = ipack.dB;
    out.gX = old.d1;
    out.hAA = ipack.dAA;
    out.hAB = ipack.dAB;
    out.hAX = ipack.dAX;
    out.hBB = ipack.dBB;
    out.hBX = ipack.dBX;
    out.hXX = old.d2;
    return out;
}

} // namespace detail
} // namespace lightape

#endif // LIGHTAPE_BASE_INCOMPLETE_BETA_H
