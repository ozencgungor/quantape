#ifndef QUANTAPE_MATH_DISTRIBUTIONS_STUDENT_T_H
#define QUANTAPE_MATH_DISTRIBUTIONS_STUDENT_T_H

#include "quantape/math/Distributions/detail/Digamma.h"
#include "quantape/math/Distributions/detail/IncompleteBeta.h"
#include "quantape/math/Distributions/detail/SecondOrderJet3.h"
#include "quantape/math/Random/InverseNormal.h"
#include "quantape/util/Constants.h"

#include <cmath>
#include <limits>
#include <type_traits>

namespace quantape::math {
/**
 * @file StudentT.h
 * @brief Student's t distribution: pdf, cdf, log tails and quantile (std-only).
 *
 * `X ~ t_nu`, `nu > 0`:
 *
 *     pdf(x)  = Gamma((nu+1)/2) / (sqrt(nu pi) Gamma(nu/2)) (1 + x^2/nu)^{-(nu+1)/2}
 *             = exp(lgamma((nu+1)/2) - lgamma(nu/2) - 0.5 log(nu pi)
 *                   - 0.5 (nu+1) log1p(x^2/nu)),
 *     cdf(x)  = 1 - (1/2) I_{z}(nu/2, 1/2),  z = nu/(nu+x^2),  x >= 0,
 *             =       (1/2) I_{z}(nu/2, 1/2),                x <  0,
 *     ccdf(x) = cdf(-x),  logCcdf(x) = logCdf(-x),
 *
 * with `I` the regularized incomplete beta (shared continued-fraction kernel
 * in `detail/IncompleteBeta.h`). The kernel forms `w = x^2/nu` and uses the
 * exact complements `z = 1/(1+w)` and `t = w/(1+w)` rather than `nu/(nu+x^2)`
 * and `1 - z`, so the `|x| -> 0` branch keeps full accuracy. The log twins
 * route through `incBetaLog`/`incBetaLogCcdf`, so the positive tail keeps
 * full relative accuracy even when `cdf` rounds to 1 (`logCdf` is computed as
 * `log1p(-ccdf)`), and the negative tail is the direct log integral.
 *
 * `inv(p)` solves the upper tail `ccdf(y) = p` for `p <= 1/2` by safeguarded
 * Newton in `v = log y` (the tail elasticity `y pdf/ccdf -> nu`, so the step
 * is well scaled for all `nu`), seeded by the normal quantile widened by
 * `sqrt(nu/(nu-2))` for `nu > 2`, then mirrors for `p > 1/2`. Relative
 * accuracy is ~1e-12 in the interior; for `nu` very large (>= ~1e5) the
 * `incBeta` argument `1 - z` loses a few digits, leaving ~1e-10 relative in
 * the extreme tails, and quantiles beyond the double range return `+-Inf`.
 *
 * Accuracy note: `lgamma((nu+1)/2) - lgamma(nu/2)` cancels for very large
 * `nu`, giving ~1e-9 relative pdf error around `nu ~ 1e6`; the cdf is
 * unaffected at that level. `nu <= 0` or NaN parameters propagate NaN.
 *
 * Derivatives (x-direction, `nu` fixed):
 *     d/dx   cdf(x) = pdf(x),
 *     d2/dx2 cdf(x) = pdf'(x) = pdf(x) * (-(nu+1) x / (nu + x^2)).
 *
 * Full parameter packs: `studentTCdfGradHess` / `studentTPdfGradHess` return
 * the value, gradient and Hessian in `(nu, x)`. The cdf parameter entries
 * come from the exact 3-jet of the incomplete-beta kernel at
 * `(nu/2, 1/2, z)`, `z = nu/(nu+x^2)`, chained to `nu` with the closed-form
 * derivatives of `z` (the x entries keep the existing closed forms); the
 * density pack is closed form with digamma/trigamma (`log`) derivatives. No
 * finite differences are involved. See the pack comments for the exact
 * expressions and edge behavior (`nu <= 0` NaN, `x = 0` one-sided limits,
 * `|x| -> Inf` flat pack).
 *
 * Two-tier dispatch (the Normal.h pattern): `T = double` accessors call the
 * named `detail::studentT*Double` kernels, which forward to the shared
 * incomplete-beta kernels explicitly instantiated for double; other scalars
 * run the generic kernel path. `inv` keeps its double-seeded solve.
 */
namespace detail {

/// Log t density; NaN for `nu <= 0`, `-inf` in the tails (pdf underflow).
template <class T>
T studentTLogPdf(T nu, T x) {
    using ::quantape::util::kPi;
    using std::lgamma;
    using std::log;
    using std::log1p;
    const T nan = T(std::numeric_limits<double>::quiet_NaN());
    if (!(nu > T(0))) {
        return nan;
    }
    return lgamma((nu + T(1)) / T(2)) - lgamma(nu / T(2)) - T(0.5) * log(nu * T(kPi.hi)) -
           T(0.5) * (nu + T(1)) * log1p(x * x / nu);
}

/**
 * @brief Upper-tail `P(X > x)` for `x >= 0` on the stable side of `z`.
 *
 * Uses `w = x^2/nu` and the exact complements `z = 1/(1+w) = nu/(nu+x^2)`,
 * `t = w/(1+w) = 1 - z`, both formed without subtraction. For `z` in the
 * continued-fraction fast range the tail `(1/2) I_z(nu/2,1/2)` is direct;
 * otherwise it is `(1/2)(1 - I_t(1/2,nu/2))` with `t < (1.5)/(nu/2+2.5)`, the
 * fast range of the mirrored integral. This avoids the catastrophic
 * `1 - z` cancellation at `|x| -> 0` that a `nu/(nu+x^2)` argument would
 * introduce.
 */
template <class T>
T studentTCcdfNonNegative(T nu, T x) {
    const T half = T(0.5);
    const T w = x * x / nu;
    const T z = T(1) / (T(1) + w);
    const T threshold = (nu / T(2) + T(1)) / (nu / T(2) + T(2.5));
    if (z <= threshold) {
        return half * incBetaLow(nu / T(2), half, z);
    }
    const T t = w / (T(1) + w);
    return half * (T(1) - incBeta(half, nu / T(2), t));
}

/// `log P(X > x)` for `x >= 0`, mirroring `studentTCcdfNonNegative`.
template <class T>
T studentTLogCcdfNonNegative(T nu, T x) {
    using std::log;
    const T half = T(0.5);
    const T w = x * x / nu;
    const T z = T(1) / (T(1) + w);
    const T threshold = (nu / T(2) + T(1)) / (nu / T(2) + T(2.5));
    if (z <= threshold) {
        return log(half) + incBetaLowLog(nu / T(2), half, z);
    }
    const T t = w / (T(1) + w);
    return log(half) + incBetaLogCcdf(half, nu / T(2), t);
}

/// t cdf via the incomplete-beta identity; `ccdf(x) = cdf(-x)`.
template <class T>
T studentTCdf(T nu, T x) {
    const T nan = T(std::numeric_limits<double>::quiet_NaN());
    if (!(nu > T(0))) {
        return nan;
    }
    const T magnitude = x >= T(0) ? x : -x;
    if (x >= T(0)) {
        return T(1) - studentTCcdfNonNegative(nu, magnitude);
    }
    return studentTCcdfNonNegative(nu, magnitude);
}

/// Log cdf with the small tail evaluated directly; stable as `cdf -> 0/1`.
template <class T>
T studentTLogCdf(T nu, T x) {
    using std::exp;
    using std::log1p;
    const T nan = T(std::numeric_limits<double>::quiet_NaN());
    if (!(nu > T(0))) {
        return nan;
    }
    const T magnitude = x >= T(0) ? x : -x;
    const T logTail = studentTLogCcdfNonNegative(nu, magnitude);
    if (x >= T(0)) {
        return log1p(-exp(logTail));
    }
    return logTail;
}

/// `log(1 - cdf(x))`; the reflection `logCdf(-x)` avoids the subtraction.
template <class T>
T studentTLogCcdf(T nu, T x) {
    using std::exp;
    using std::log1p;
    const T nan = T(std::numeric_limits<double>::quiet_NaN());
    if (!(nu > T(0))) {
        return nan;
    }
    const T magnitude = x >= T(0) ? x : -x;
    const T logTail = studentTLogCcdfNonNegative(nu, magnitude);
    if (x >= T(0)) {
        return logTail;
    }
    return log1p(-exp(logTail));
}

/// `P(X > x) = cdf(-x)`.
template <class T>
T studentTCcdf(T nu, T x) {
    return studentTCdf(nu, -x);
}

// ---------------------------------------------------------------------------
// Named double kernels (the `T = double` fast path of `StudentT<T>`). Each
// entry forwards to the shared templated kernel explicitly instantiated for
// double, so every method has one canonical double definition and no
// algorithm body is duplicated. `inv` keeps its own double-seeded solve.
// ---------------------------------------------------------------------------

/// log pdf of t_nu at x (double).
inline double studentTLogPdfDouble(double nu, double x) {
    return studentTLogPdf<double>(nu, x);
}

/// pdf of t_nu at x (double).
inline double studentTPdfDouble(double nu, double x) {
    using std::exp;
    return exp(studentTLogPdfDouble(nu, x));
}

/// cdf of t_nu at x (double).
inline double studentTCdfDouble(double nu, double x) {
    return studentTCdf<double>(nu, x);
}

/// log cdf of t_nu at x (double).
inline double studentTLogCdfDouble(double nu, double x) {
    return studentTLogCdf<double>(nu, x);
}

/// complementary cdf of t_nu at x (double).
inline double studentTCcdfDouble(double nu, double x) {
    return studentTCcdf<double>(nu, x);
}

/// log complementary cdf of t_nu at x (double).
inline double studentTLogCcdfDouble(double nu, double x) {
    return studentTLogCcdf<double>(nu, x);
}

/// t pdf with `d1 = pdf`, `d2 = pdf' = pdf * (-(nu+1) x / (nu + x^2))`.
inline BetaValueD1D2 studentTPdfD1D2(double nu, double x) {
    using std::exp;
    const double value = exp(studentTLogPdf(nu, x));
    const double d2 = value * (-(nu + 1.0) * x / (nu + x * x));
    return {value, value, d2};
}

/// t cdf with `value = cdf`, `d1 = pdf`, `d2 = pdf'`.
inline BetaValueD1D2 studentTCdfD1D2(double nu, double x) {
    const BetaValueD1D2 pdf = studentTPdfD1D2(nu, x);
    return {studentTCdf(nu, x), pdf.value, pdf.d2};
}

// ---------------------------------------------------------------------------
// Full parameter-derivative packs in (nu, x).
//
// Method (documented decision): the t cdf is `F = 1 - (1/2) I_z(nu/2, 1/2)`
// for `x >= 0` and `F = (1/2) I_z(nu/2, 1/2)` for `x < 0`, `z = nu/(nu+x^2)`.
// The `(a = nu/2, z)` gradient/Hessian of `I` is obtained exactly from one
// `incBeta<SecondOrderJet3>` evaluation with `b = 1/2` constant, and chained
// to nu through the closed-form derivatives of `z` (no finite differences).
// The x entries keep the existing closed forms (bitwise with `*D1D2`), and
// the density pack's parameter entries are the digamma/trigamma `log f`
// derivatives.
//
// Edge behavior (mirrors the value paths): `nu <= 0`, NaN `nu` or NaN `x`
// -> all-NaN pack; `x = +-Inf` -> flat pack (cdf value 1/0, every partial 0);
// `x = 0` -> one-sided limits (cdf 1/2; `gx = pdf(0)`, `hNux = d pdf/dnu`,
// the remaining partials 0). For `|x|` so small that `z` rounds to 1 the
// jet's constant branch returns zero nu-partials whose true magnitude is
// O(|x|); for `|x|` so large that `z` underflows to 0 the pack is flat.
// Around `nu ~ 1e6` the value-level `lgamma` cancellation noted in the file
// header limits the density accuracy (and hence the nu entries); the cdf
// pack degrades more gracefully.
// ---------------------------------------------------------------------------

/// (nu, x) value + gradient + Hessian for the Student-t density.
struct StudentTPdfGradHess {
    double value = 0.0;
    double gNu = 0.0;
    double gx = 0.0;
    double hNuNu = 0.0;
    double hNux = 0.0;
    double hxx = 0.0;
};

/// (nu, x) value + gradient + Hessian for the Student-t CDF.
struct StudentTCdfGradHess {
    double value = 0.0;
    double gNu = 0.0;
    double gx = 0.0;
    double hNuNu = 0.0;
    double hNux = 0.0;
    double hxx = 0.0;
};

/// All-NaN density pack (invalid inputs), see the section comment.
inline StudentTPdfGradHess studentTPdfGradHessNaN() noexcept {
    const double nan = std::numeric_limits<double>::quiet_NaN();
    return {nan, nan, nan, nan, nan, nan};
}

/// All-NaN CDF pack (invalid inputs), see the section comment.
inline StudentTCdfGradHess studentTCdfGradHessNaN() noexcept {
    const double nan = std::numeric_limits<double>::quiet_NaN();
    return {nan, nan, nan, nan, nan, nan};
}

/**
 * @brief `{f, f_nu, f_x, f_nunu, f_nux, f_xx}` for the `t_nu` density.
 *
 * With `L = log f = lgamma((nu+1)/2) - lgamma(nu/2) - (1/2) log(nu pi) -
 * (1/2)(nu+1) log1p(x^2/nu)`, `D = nu + x^2`, `w = x^2/nu`:
 *   `L_nu  = (psi((nu+1)/2) - psi(nu/2))/2 - 1/(2 nu) - log1p(w)/2
 *            + (nu+1) x^2/(2 nu D)`,
 *   `L_nunu = (psi'((nu+1)/2) - psi'(nu/2))/4 + 1/(2 nu^2) + x^2/(nu D)
 *             - x^2 (nu+1)(2 nu+x^2)/(2 nu^2 D^2)`,
 *   `L_nux = x (1 - x^2)/D^2`, `q = -(nu+1) x/D`,
 * so `f_nu = f L_nu`, `f_nunu = f (L_nunu + L_nu^2)`, `f_nux = f (L_nux +
 * L_nu q)`, `f_x = f q`, `f_xx = f (q^2 - (nu+1)(nu-x^2)/D^2)`. `value` and
 * `gx` reproduce `studentTPdfD1D2`'s `value` and `d2` bit-for-bit.
 */
inline StudentTPdfGradHess studentTPdfGradHess(double nu, double x) noexcept {
    using std::exp;
    using std::log1p;
    if (!(nu > 0.0) || x != x) {
        return studentTPdfGradHessNaN();
    }
    if (std::isinf(x)) {
        return StudentTPdfGradHess{};
    }

    const double D = nu + x * x;
    const double pdf = exp(studentTLogPdf(nu, x));
    const double q = -(nu + 1.0) * x / D;
    const double psiA = digammaDouble(0.5 * (nu + 1.0));
    const double psiB = digammaDouble(0.5 * nu);
    const double triA = trigammaDouble(0.5 * (nu + 1.0));
    const double triB = trigammaDouble(0.5 * nu);
    const double ln = 0.5 * (psiA - psiB) - 0.5 / nu - 0.5 * log1p(x * x / nu) +
                      (nu + 1.0) * x * x / (2.0 * nu * D);
    const double lnn = 0.25 * (triA - triB) + 0.5 / (nu * nu) + x * x / (nu * D) -
                       x * x * (nu + 1.0) * (2.0 * nu + x * x) / (2.0 * nu * nu * D * D);
    const double lnx = x * (1.0 - x * x) / (D * D);

    StudentTPdfGradHess out{};
    out.value = pdf;
    out.gNu = pdf * ln;
    out.gx = pdf * q;
    out.hNuNu = pdf * (lnn + ln * ln);
    out.hNux = pdf * (lnx + ln * q);
    out.hxx = pdf * (q * q - (nu + 1.0) * (nu - x * x) / (D * D));
    return out;
}

/**
 * @brief `{F, F_nu, F_x, F_nunu, F_nux, F_xx}` for the `t_nu` CDF.
 *
 * `F = c -+ (1/2) P`, `P = I_z(nu/2, 1/2)`, `z = nu/(nu+x^2)`; the sign is
 * `-` for `x >= 0` (`c = 1`) and `+` for `x < 0` (`c = 0`). The jet gives
 * `P_a, P_z, P_aa, P_az, P_zz` at `(a, b, z)`, and with
 * `dz/dnu = x^2/D^2`, `dz/dx = -2 nu x/D^2`,
 * `d2z/dnu2 = -2 x^2/D^3`, `d2z/dnu dx = 2 x (nu-x^2)/D^3`,
 * `d2z/dx2 = 2 nu (3x^2-nu)/D^3` (`D = nu+x^2`):
 *   `P_nu = P_a/2 + P_z z_nu`,
 *   `P_nunu = P_aa/4 + P_az z_nu + P_zz z_nu^2 + P_z z_nunu`,
 *   `P_nux = P_az z_x/2 + P_zz z_nu z_x + P_z z_nux`.
 * `value`, `gx` and `hxx` reproduce `studentTCdfD1D2` bit-for-bit. See the
 * section comment for the `x = 0` limits and the near-zero/near-one `z`
 * caveats.
 */
inline StudentTCdfGradHess studentTCdfGradHess(double nu, double x) noexcept {
    if (!(nu > 0.0) || x != x) {
        return studentTCdfGradHessNaN();
    }
    if (std::isinf(x)) {
        StudentTCdfGradHess out{};
        out.value = x > 0.0 ? 1.0 : 0.0;
        return out;
    }
    if (x == 0.0) {
        StudentTCdfGradHess out{};
        out.value = 0.5;
        const StudentTPdfGradHess density = studentTPdfGradHess(nu, 0.0);
        out.gx = density.value;
        out.hNux = density.gNu;
        return out;
    }

    const double D = nu + x * x;
    const double a = 0.5 * nu;
    const double z = nu / D;
    if (!(z > 0.0)) {
        // |x| so large that z underflowed (or x*x overflowed): flat pack.
        StudentTCdfGradHess out{};
        out.value = studentTCdf(nu, x);
        return out;
    }

    const SecondOrderJet3 aJet(a, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
    const SecondOrderJet3 bJet(0.5);
    const SecondOrderJet3 zJet(z, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
    const SecondOrderJet3 ipack = incBeta<SecondOrderJet3>(aJet, bJet, zJet);

    const double zn = x * x / (D * D);
    const double zx = -2.0 * nu * x / (D * D);
    const double znn = -2.0 * x * x / (D * D * D);
    const double znx = 2.0 * x * (nu - x * x) / (D * D * D);
    const double pNu = 0.5 * ipack.dA + ipack.dX * zn;
    const double pNuNu = 0.25 * ipack.dAA + ipack.dAX * zn + ipack.dXX * zn * zn + ipack.dX * znn;
    const double pNuX = 0.5 * ipack.dAX * zx + ipack.dXX * zn * zx + ipack.dX * znx;

    const BetaValueD1D2 old = studentTCdfD1D2(nu, x);
    const double sign = x > 0.0 ? -0.5 : 0.5;
    StudentTCdfGradHess out{};
    out.value = old.value;
    out.gNu = sign * pNu;
    out.gx = old.d1;
    out.hNuNu = sign * pNuNu;
    out.hNux = sign * pNuX;
    out.hxx = old.d2;
    return out;
}

/// Bracket-expansion cap: t quantiles grow at most like `p^{-1/nu}`.
inline constexpr int kStudentTBracketSteps = 2000;

/**
 * @brief Upper-tail t quantile: solves `ccdf(y) = p` for `0 < p <= 1/2`, `y > 0`.
 *
 * Seed from `inverseNormal`; bracket `[0, hi]` grown by doubling until
 * `ccdf(hi) <= p` (giving `+-Inf` when the quantile exceeds the double
 * range). Iteration is Newton in `v = log y` on `log ccdf(y) - log p`, whose
 * slope `y pdf/ccdf` is bounded by `~nu` in the power-law tail; a candidate
 * outside the bracket or non-finite is replaced by the bisection midpoint.
 * Termination is a relative step below 1e-13 (estimated quantile relative
 * error ~1e-12).
 */
template <class T>
T studentTInvLower(T nu, T p) {
    using std::exp;
    using std::fabs;
    using std::log;
    using std::sqrt;
    const T inf = std::numeric_limits<T>::infinity();
    const T logP = log(p);
    T y = T(-inverseNormal(static_cast<double>(p)));
    if (nu > T(2)) {
        y *= sqrt(nu / (nu - T(2)));
    }
    if (!(y > T(0))) {
        y = T(1);
    }
    T lo = T(0);
    T hi = y;
    if (studentTCcdf(nu, y) > p) {
        T t = y;
        bool bracketed = false;
        for (int i = 0; i < kStudentTBracketSteps; ++i) {
            t *= T(2);
            if (!(t < std::numeric_limits<double>::max())) {
                break;
            }
            if (studentTCcdf(nu, t) <= p) {
                hi = t;
                bracketed = true;
                break;
            }
        }
        if (!bracketed) {
            return inf;
        }
    }
    for (int it = 0; it < 200; ++it) {
        const T lcc = studentTLogCcdf(nu, y);
        const T r = lcc - logP;
        if (r > T(0)) {
            lo = y;
        } else {
            hi = y;
        }
        const T slope = exp(log(y) + studentTLogPdf(nu, y) - lcc); // y pdf / ccdf
        T yn = T(-1);
        if (slope > T(0)) {
            const T dv = r / slope;
            if (dv == dv) {
                yn = y * exp(dv);
            }
        }
        if (!(yn > lo) || !(yn < hi)) {
            yn = T(0.5) * (lo + hi);
        }
        const T step = fabs(yn - y);
        y = yn;
        if (step <= 1e-13 * y) {
            break;
        }
    }
    return y;
}

/// t quantile at double: the exact `StudentT<double>::inv` body (same tail
/// mirror and `p = 1/2` shortcut), so the pack `value` is bitwise `inv(p)`.
inline double studentTInvDouble(double nu, double p) {
    const double nan = std::numeric_limits<double>::quiet_NaN();
    const double inf = std::numeric_limits<double>::infinity();
    if (!(nu > 0.0)) {
        return nan;
    }
    if (!(p >= 0.0) || !(p <= 1.0)) {
        return nan;
    }
    if (p <= 0.0) {
        return -inf;
    }
    if (p >= 1.0) {
        return inf;
    }
    if (p == 0.5) {
        return 0.0;
    }
    if (p > 0.5) {
        return studentTInvLower<double>(nu, 1.0 - p);
    }
    return -studentTInvLower<double>(nu, p);
}

// ---------------------------------------------------------------------------
// Student-t quantile parameter-derivative pack.
//
// Same implicit differentiation as in Normal.h: with `f = F_x`, `f' = F_xx`
// at `x = Q`, `Q_p = 1/f`, `Q_nu = -F_nu/f`,
// `Q_pp = -f'/f^3`, `Q_p,nu = -(F_x,nu + f' Q_nu)/f^2`,
// `Q_nunu = -(F_nunu + 2 F_nu,x Q_nu + f' Q_nu^2)/f`, with the nu partials
// from `studentTCdfGradHess` at the quantile. The `value` is bitwise
// `StudentT::inv(p)`; outside the open unit interval the derivative entries
// are NaN (`value` keeps the -Inf/+Inf endpoint convention, NaN for invalid
// nu or p outside [0,1]). No finite differences.
// ---------------------------------------------------------------------------

/// (p, nu) value + gradient + Hessian of the Student-t quantile.
struct StudentTInvGradHess {
    double value = 0.0;
    double gP = 0.0;
    double gNu = 0.0;
    double hPP = 0.0;
    double hPNu = 0.0;
    double hNuNu = 0.0;
};

/// `{Q, Q_p, Q_nu, Q_pp, Q_p,nu, Q_nunu}` for the Student-t quantile; see the
/// section comment.
inline StudentTInvGradHess studentTInvGradHess(double nu, double p) {
    const double nan = std::numeric_limits<double>::quiet_NaN();
    StudentTInvGradHess out;
    out.value = studentTInvDouble(nu, p);
    if (!(p > 0.0) || !(p < 1.0)) {
        out.gP = nan;
        out.gNu = nan;
        out.hPP = nan;
        out.hPNu = nan;
        out.hNuNu = nan;
        return out;
    }
    const StudentTCdfGradHess c = studentTCdfGradHess(nu, out.value);
    const double f = c.gx;
    const double fp = c.hxx;
    out.gP = 1.0 / f;
    out.gNu = -c.gNu / f;
    out.hPP = -fp / (f * f * f);
    out.hPNu = -(c.hNux + fp * out.gNu) / (f * f);
    out.hNuNu = -(c.hNuNu + 2.0 * c.hNux * out.gNu + fp * out.gNu * out.gNu) / f;
    return out;
}

} // namespace detail

template <class T>
struct StudentT {
    T nu;

    T pdf(T x) const;
    T logPdf(T x) const;
    T cdf(T x) const;
    T logCdf(T x) const;
    T ccdf(T x) const;
    T logCcdf(T x) const;
    T inv(T p) const;
    detail::StudentTInvGradHess invGradHess(T p) const;
};

template <class T>
T StudentT<T>::pdf(T x) const {
    if constexpr (std::is_same_v<T, double>) {
        return detail::studentTPdfDouble(nu, x);
    } else {
        using std::exp;
        return exp(logPdf(x));
    }
}

template <class T>
T StudentT<T>::logPdf(T x) const {
    if constexpr (std::is_same_v<T, double>) {
        return detail::studentTLogPdfDouble(nu, x);
    } else {
        return detail::studentTLogPdf(nu, x);
    }
}

template <class T>
T StudentT<T>::cdf(T x) const {
    if constexpr (std::is_same_v<T, double>) {
        return detail::studentTCdfDouble(nu, x);
    } else {
        return detail::studentTCdf(nu, x);
    }
}

template <class T>
T StudentT<T>::logCdf(T x) const {
    if constexpr (std::is_same_v<T, double>) {
        return detail::studentTLogCdfDouble(nu, x);
    } else {
        return detail::studentTLogCdf(nu, x);
    }
}

template <class T>
T StudentT<T>::ccdf(T x) const {
    if constexpr (std::is_same_v<T, double>) {
        return detail::studentTCcdfDouble(nu, x);
    } else {
        return detail::studentTCcdf(nu, x);
    }
}

template <class T>
T StudentT<T>::logCcdf(T x) const {
    if constexpr (std::is_same_v<T, double>) {
        return detail::studentTLogCcdfDouble(nu, x);
    } else {
        return detail::studentTLogCcdf(nu, x);
    }
}

template <class T>
T StudentT<T>::inv(T p) const {
    const T nan = T(std::numeric_limits<double>::quiet_NaN());
    const T inf = std::numeric_limits<T>::infinity();
    if (!(nu > T(0))) {
        return nan;
    }
    if (!(p >= T(0)) || !(p <= T(1))) {
        return nan;
    }
    if (p <= T(0)) {
        return -inf;
    }
    if (p >= T(1)) {
        return inf;
    }
    if (p == T(0.5)) {
        return T(0);
    }
    if (p > T(0.5)) {
        return detail::studentTInvLower(nu, T(1) - p);
    }
    return -detail::studentTInvLower(nu, p);
}

template <class T>
detail::StudentTInvGradHess StudentT<T>::invGradHess(T p) const {
    static_assert(std::is_same_v<T, double>, "invGradHess analytic pack requires T = double");
    return detail::studentTInvGradHess(nu, p);
}

} // namespace quantape::math

#endif // QUANTAPE_MATH_DISTRIBUTIONS_STUDENT_T_H
