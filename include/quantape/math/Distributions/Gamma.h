#ifndef QUANTAPE_MATH_DISTRIBUTIONS_GAMMA_H
#define QUANTAPE_MATH_DISTRIBUTIONS_GAMMA_H

#include "quantape/math/Distributions/detail/IncompleteGamma.h"
#include "quantape/math/Random/InverseNormal.h"

#include <cmath>
#include <limits>
#include <type_traits>

namespace quantape::math {
/**
 * @file Gamma.h
 * @brief Gamma and InverseGamma laws: density, CDF, log tails and quantile.
 *
 * Gamma(shape, rate):
 *   `pdf(x) = rate^shape x^(shape-1) e^(-rate x) / Gamma(shape)`,
 *   `cdf(x) = P(shape, rate x)`, `ccdf(x) = Q(shape, rate x)` with `P`/`Q` the
 *   regularized incomplete gamma functions (series + continued fraction,
 *   `detail/IncompleteGamma.h`).
 *
 * InverseGamma(shape, scale): if `Y ~ Gamma(shape, rate = scale)` then
 * `1 / Y` has this law, so
 *   `pdf(x) = scale^shape / Gamma(shape) x^(-shape-1) e^(-scale/x)`,
 *   `cdf(x) = Q(shape, scale/x)`, `ccdf(x) = P(shape, scale/x)`.
 *
 * `Gamma::inv` uses a Wilson-Hilferty cube-root seed with Acklam's inverse
 * normal, then 30 safeguarded Newton steps on the analytic CDF (derivative =
 * the analytic density, see `detail::gammaPdfD1D2`) and a 60-step bisection
 * fallback on a bracket `[0, expanding upper bound]`. For `p` in the interior
 * `[1e-12, 1 - 1e-12]` the quantile is good to relative ~1e-12..1e-13 (the
 * CDF's own accuracy is the limit); verified down to `p ~ 1e-100` with the
 * small-p surrogate seed. Two graceful degradations:
 *   * lower tail `p < 1e-12`: the seed falls back to
 *     `(p Gamma(a+1))^(1/a) / rate`; for `p` below ~1e-300 the true quantile
 *     may be unrepresentable and a near-zero value is returned;
 *   * upper tail `1 - p < ~1e-10`: the input `p` has already lost the tail
 *     digits (`ulp(1) / (1-p)` relative), so the recovered quantile inherits
 *     an absolute error `~ulp(1) / pdf(x)` that no inversion algorithm can
 *     remove (e.g. at `1 - p = 5e-10` for shape 2.5/rate 1.3 the result is
 *     only good to ~1e-9 relative).
 *
 * Every accessor is `const` and scalar-generic over `T` (double, AD scalars).
 * `Gamma<double>`/`InverseGamma<double>` dispatch each method to its named
 * `detail::...Double` entry (the canonical double tier, built on the named
 * `detail::gammaPDouble`/`gammaQDouble`/`gammaLogPDouble`/`gammaLogQDouble`
 * incomplete-gamma entries); other scalars run the generic `detail::...T<T>`
 * kernels with ADL math. Constants are literals, no allocation, no exceptions.
 */

namespace detail {

// ---------------------------------------------------------------------------
// Scalar-generic method kernels (`...T`). The distribution wrappers use these
// for `DoubleT != double`; the named `...Double` entries below are the
// canonical double path.
// ---------------------------------------------------------------------------

/// Gamma(shape, rate) density, generic scalar (see `Gamma::pdf`).
template <class T>
T gammaPdfT(T shape, T rate, T x) {
    using std::exp;
    using std::lgamma;
    using std::log;
    if (!(shape > T(0)) || !(rate > T(0))) {
        return detail::nanValue<T>();
    }
    if (x != x) {
        return x;
    }
    if (x <= T(0)) {
        if (x < T(0) || shape > T(1)) {
            return T(0);
        }
        if (shape < T(1)) {
            return detail::positiveInfinity<T>();
        }
        return rate;
    }
    if (detail::isPositiveInfinity(x)) {
        return T(0);
    }
    return exp(shape * log(rate) - lgamma(shape) + (shape - T(1)) * log(x) - rate * x);
}

/// Gamma(shape, rate) log density, generic scalar (see `Gamma::logPdf`).
template <class T>
T gammaLogPdfT(T shape, T rate, T x) {
    using std::lgamma;
    using std::log;
    if (!(shape > T(0)) || !(rate > T(0))) {
        return detail::nanValue<T>();
    }
    if (x != x) {
        return x;
    }
    if (x <= T(0)) {
        if (x < T(0) || shape > T(1)) {
            return detail::negativeInfinity<T>();
        }
        if (shape < T(1)) {
            return detail::positiveInfinity<T>();
        }
        return log(rate);
    }
    if (detail::isPositiveInfinity(x)) {
        return detail::negativeInfinity<T>();
    }
    return shape * log(rate) - lgamma(shape) + (shape - T(1)) * log(x) - rate * x;
}

/// Gamma(shape, rate) CDF, generic scalar: `P(shape, rate x)`.
template <class T>
T gammaCdfT(T shape, T rate, T x) {
    if (!(rate > T(0))) {
        return detail::nanValue<T>();
    }
    return detail::gammaP(shape, rate * x);
}

/// Gamma(shape, rate) log CDF, generic scalar: `log P(shape, rate x)`.
template <class T>
T gammaLogCdfT(T shape, T rate, T x) {
    if (!(rate > T(0))) {
        return detail::nanValue<T>();
    }
    return detail::gammaLogP(shape, rate * x);
}

/// Gamma(shape, rate) ccdf, generic scalar: `Q(shape, rate x)`.
template <class T>
T gammaCcdfT(T shape, T rate, T x) {
    if (!(rate > T(0))) {
        return detail::nanValue<T>();
    }
    return detail::gammaQ(shape, rate * x);
}

/// Gamma(shape, rate) log ccdf, generic scalar: `log Q(shape, rate x)`.
template <class T>
T gammaLogCcdfT(T shape, T rate, T x) {
    if (!(rate > T(0))) {
        return detail::nanValue<T>();
    }
    return detail::gammaLogQ(shape, rate * x);
}

/// InverseGamma(shape, scale) density, generic scalar (see `InverseGamma::pdf`).
template <class T>
T invGammaPdfT(T shape, T scale, T x) {
    using std::exp;
    using std::lgamma;
    using std::log;
    if (!(shape > T(0)) || !(scale > T(0))) {
        return detail::nanValue<T>();
    }
    if (x != x) {
        return x;
    }
    if (x <= T(0) || detail::isPositiveInfinity(x)) {
        return T(0);
    }
    return exp(shape * log(scale) - lgamma(shape) - (shape + T(1)) * log(x) - scale / x);
}

/// InverseGamma(shape, scale) log density, generic scalar.
template <class T>
T invGammaLogPdfT(T shape, T scale, T x) {
    using std::lgamma;
    using std::log;
    if (!(shape > T(0)) || !(scale > T(0))) {
        return detail::nanValue<T>();
    }
    if (x != x) {
        return x;
    }
    if (x <= T(0) || detail::isPositiveInfinity(x)) {
        return detail::negativeInfinity<T>();
    }
    return shape * log(scale) - lgamma(shape) - (shape + T(1)) * log(x) - scale / x;
}

/// InverseGamma(shape, scale) CDF, generic scalar: `Q(shape, scale/x)`.
template <class T>
T invGammaCdfT(T shape, T scale, T x) {
    if (!(shape > T(0)) || !(scale > T(0))) {
        return detail::nanValue<T>();
    }
    if (x != x) {
        return x;
    }
    if (x <= T(0)) {
        return T(0);
    }
    if (detail::isPositiveInfinity(x)) {
        return T(1);
    }
    return detail::gammaQ(shape, scale / x);
}

/// InverseGamma(shape, scale) log CDF, generic scalar: `log Q(shape, scale/x)`.
template <class T>
T invGammaLogCdfT(T shape, T scale, T x) {
    if (!(shape > T(0)) || !(scale > T(0))) {
        return detail::nanValue<T>();
    }
    if (x != x) {
        return x;
    }
    if (x <= T(0)) {
        return detail::negativeInfinity<T>();
    }
    if (detail::isPositiveInfinity(x)) {
        return T(0);
    }
    return detail::gammaLogQ(shape, scale / x);
}

/// InverseGamma(shape, scale) ccdf, generic scalar: `P(shape, scale/x)`.
template <class T>
T invGammaCcdfT(T shape, T scale, T x) {
    if (!(shape > T(0)) || !(scale > T(0))) {
        return detail::nanValue<T>();
    }
    if (x != x) {
        return x;
    }
    if (x <= T(0)) {
        return T(1);
    }
    if (detail::isPositiveInfinity(x)) {
        return T(0);
    }
    return detail::gammaP(shape, scale / x);
}

/// InverseGamma(shape, scale) log ccdf, generic scalar: `log P(shape, scale/x)`.
template <class T>
T invGammaLogCcdfT(T shape, T scale, T x) {
    if (!(shape > T(0)) || !(scale > T(0))) {
        return detail::nanValue<T>();
    }
    if (x != x) {
        return x;
    }
    if (x <= T(0)) {
        return T(0);
    }
    if (detail::isPositiveInfinity(x)) {
        return detail::negativeInfinity<T>();
    }
    return detail::gammaLogP(shape, scale / x);
}

/// Gamma(shape, rate) quantile, generic scalar (see `Gamma::inv`).
template <class T>
T gammaInvT(T shape, T rate, T p) {
    using std::exp;
    using std::fabs;
    using std::lgamma;
    using std::log;
    using std::sqrt;
    if (p != p) {
        return p;
    }
    if (!(shape > T(0)) || !(rate > T(0))) {
        return detail::nanValue<T>();
    }
    if (p <= T(0)) {
        return T(0);
    }
    if (p >= T(1)) {
        return detail::positiveInfinity<T>();
    }

    const T a = shape;
    const T r = rate;
    const double z = inverseNormal(detail::scalarValue(p));
    const T w = T(1) - T(1) / (T(9) * a) + T(z) / (T(3) * sqrt(a));
    T x = w * w * w / r;
    if (!(x > T(0)) || detail::isPositiveInfinity(x)) {
        // Small-p surrogate: P(a, x) ~ (rate x)^a / Gamma(a + 1).
        x = exp((log(p) + lgamma(a + T(1))) / a) / r;
    }
    if (!(x > T(0))) {
        x = (a / r) * T(1e-12);
    }

    T lo = T(0);
    T hi = x;
    T cdfHi = gammaCdfT(shape, rate, hi);
    for (int expand = 0; cdfHi < p && expand < 200; ++expand) {
        lo = hi;
        hi = hi * T(2);
        cdfHi = gammaCdfT(shape, rate, hi);
    }

    bool converged = false;
    for (int it = 0; it < 30; ++it) {
        const T cdfX = gammaCdfT(shape, rate, x);
        if (cdfX < p) {
            lo = x;
        } else {
            hi = x;
        }
        const T step = (cdfX - p) / gammaPdfT(shape, rate, x);
        const T next = x - step;
        if (fabs(step) <= T(1e-13) * (x > T(1) ? x : T(1))) {
            x = next;
            converged = true;
            break;
        }
        if (!(next > lo && next < hi)) {
            break;
        }
        x = next;
    }
    if (!converged) {
        for (int it = 0; it < 60; ++it) {
            const T mid = lo + (hi - lo) / T(2);
            if (gammaCdfT(shape, rate, mid) < p) {
                lo = mid;
            } else {
                hi = mid;
            }
        }
        x = lo + (hi - lo) / T(2);
    }
    return x;
}

// ---------------------------------------------------------------------------
// Named double entry points (the canonical double tier). Each wraps the
// shared generic kernel explicitly instantiated at `double`; the incomplete
// gamma calls use the named `...Double` kernels from detail/IncompleteGamma.h.
// ---------------------------------------------------------------------------

/// Gamma(shape, rate) density at `T = double`.
inline double gammaPdfDouble(double shape, double rate, double x) {
    return gammaPdfT<double>(shape, rate, x);
}

/// Gamma(shape, rate) log density at `T = double`.
inline double gammaLogPdfDouble(double shape, double rate, double x) {
    return gammaLogPdfT<double>(shape, rate, x);
}

/// Gamma(shape, rate) CDF at `T = double`.
inline double gammaCdfDouble(double shape, double rate, double x) {
    if (!(rate > 0.0)) {
        return detail::nanValue<double>();
    }
    return detail::gammaPDouble(shape, rate * x);
}

/// Gamma(shape, rate) log CDF at `T = double`.
inline double gammaLogCdfDouble(double shape, double rate, double x) {
    if (!(rate > 0.0)) {
        return detail::nanValue<double>();
    }
    return detail::gammaLogPDouble(shape, rate * x);
}

/// Gamma(shape, rate) ccdf at `T = double`.
inline double gammaCcdfDouble(double shape, double rate, double x) {
    if (!(rate > 0.0)) {
        return detail::nanValue<double>();
    }
    return detail::gammaQDouble(shape, rate * x);
}

/// Gamma(shape, rate) log ccdf at `T = double`.
inline double gammaLogCcdfDouble(double shape, double rate, double x) {
    if (!(rate > 0.0)) {
        return detail::nanValue<double>();
    }
    return detail::gammaLogQDouble(shape, rate * x);
}

/// InverseGamma(shape, scale) density at `T = double`.
inline double invGammaPdfDouble(double shape, double scale, double x) {
    return invGammaPdfT<double>(shape, scale, x);
}

/// InverseGamma(shape, scale) log density at `T = double`.
inline double invGammaLogPdfDouble(double shape, double scale, double x) {
    return invGammaLogPdfT<double>(shape, scale, x);
}

/// InverseGamma(shape, scale) CDF at `T = double`.
inline double invGammaCdfDouble(double shape, double scale, double x) {
    if (!(shape > 0.0) || !(scale > 0.0)) {
        return detail::nanValue<double>();
    }
    if (x != x) {
        return x;
    }
    if (x <= 0.0) {
        return 0.0;
    }
    if (detail::isPositiveInfinity(x)) {
        return 1.0;
    }
    return detail::gammaQDouble(shape, scale / x);
}

/// InverseGamma(shape, scale) log CDF at `T = double`.
inline double invGammaLogCdfDouble(double shape, double scale, double x) {
    if (!(shape > 0.0) || !(scale > 0.0)) {
        return detail::nanValue<double>();
    }
    if (x != x) {
        return x;
    }
    if (x <= 0.0) {
        return detail::negativeInfinity<double>();
    }
    if (detail::isPositiveInfinity(x)) {
        return 0.0;
    }
    return detail::gammaLogQDouble(shape, scale / x);
}

/// InverseGamma(shape, scale) ccdf at `T = double`.
inline double invGammaCcdfDouble(double shape, double scale, double x) {
    if (!(shape > 0.0) || !(scale > 0.0)) {
        return detail::nanValue<double>();
    }
    if (x != x) {
        return x;
    }
    if (x <= 0.0) {
        return 1.0;
    }
    if (detail::isPositiveInfinity(x)) {
        return 0.0;
    }
    return detail::gammaPDouble(shape, scale / x);
}

/// InverseGamma(shape, scale) log ccdf at `T = double`.
inline double invGammaLogCcdfDouble(double shape, double scale, double x) {
    if (!(shape > 0.0) || !(scale > 0.0)) {
        return detail::nanValue<double>();
    }
    if (x != x) {
        return x;
    }
    if (x <= 0.0) {
        return 0.0;
    }
    if (detail::isPositiveInfinity(x)) {
        return detail::negativeInfinity<double>();
    }
    return detail::gammaLogPDouble(shape, scale / x);
}

/// Gamma(shape, rate) quantile at `T = double`.
inline double gammaInvDouble(double shape, double rate, double p) {
    return gammaInvT<double>(shape, rate, p);
}

// ---------------------------------------------------------------------------
// Gamma quantile parameter-derivative pack.
//
// Same implicit differentiation as in Normal.h: with `f = F_x`, `f' = F_xx`
// at `x = Q`, `Q_p = 1/f`, `Q_theta = -F_theta/f`,
// `Q_pp = -f'/f^3`, `Q_p,theta = -(F_x,theta + f' Q_theta)/f^2`,
// `Q_theta,theta = -(F_theta,theta + 2 F_theta,x Q_theta + f' Q_theta^2)/f`,
// with the shape/rate partials taken from `gammaCdfGradHess` at the quantile.
// The `value` is bitwise `Gamma::inv(p)`; outside the open unit interval the
// derivative entries are NaN (`value` keeps the 0/+Inf endpoint convention).
// No finite differences.
// ---------------------------------------------------------------------------

/// Gamma quantile `{Q, Q_p, Q_shape, Q_rate, Q_pp, Q_p,shape, Q_p,rate,
/// Q_shapeshape, Q_shaperate, Q_raterate}`.
struct GammaInvGradHess {
    double value = 0.0;
    double gP = 0.0;
    double gShape = 0.0;
    double gRate = 0.0;
    double hPP = 0.0;
    double hPShape = 0.0;
    double hPRate = 0.0;
    double hShapeShape = 0.0;
    double hShapeRate = 0.0;
    double hRateRate = 0.0;
};

/// `{Q, Q_p, Q_shape, Q_rate, Q_pp, Q_p,shape, Q_p,rate, Q_shapeshape,
/// Q_shaperate, Q_raterate}` for the Gamma quantile; see the section comment.
inline GammaInvGradHess gammaInvGradHess(double shape, double rate, double p) {
    const double nan = std::numeric_limits<double>::quiet_NaN();
    GammaInvGradHess out;
    out.value = gammaInvDouble(shape, rate, p);
    if (!(p > 0.0) || !(p < 1.0)) {
        out.gP = nan;
        out.gShape = nan;
        out.gRate = nan;
        out.hPP = nan;
        out.hPShape = nan;
        out.hPRate = nan;
        out.hShapeShape = nan;
        out.hShapeRate = nan;
        out.hRateRate = nan;
        return out;
    }
    const GammaCdfGradHess c = gammaCdfGradHess(shape, rate, out.value);
    const double f = c.gx;
    const double fp = c.hxx;
    out.gP = 1.0 / f;
    out.gShape = -c.gShape / f;
    out.gRate = -c.gRate / f;
    out.hPP = -fp / (f * f * f);
    out.hPShape = -(c.hShapeX + fp * out.gShape) / (f * f);
    out.hPRate = -(c.hRateX + fp * out.gRate) / (f * f);
    out.hShapeShape =
        -(c.hShapeShape + 2.0 * c.hShapeX * out.gShape + fp * out.gShape * out.gShape) / f;
    out.hShapeRate = -(c.hShapeRate + c.hShapeX * out.gRate + c.hRateX * out.gShape +
                       fp * out.gShape * out.gRate) /
                     f;
    out.hRateRate = -(c.hRateRate + 2.0 * c.hRateX * out.gRate + fp * out.gRate * out.gRate) / f;
    return out;
}

// ---------------------------------------------------------------------------
// InverseGamma(shape, scale) parameter-derivative packs.
//
// The law is the reciprocal Gamma: `cdf(x) = Q(shape, scale/x) =
// 1 - G(1/x)` and `pdf(x) = g(1/x) / x^2`, where `G`/`g` are the
// Gamma(shape, rate = scale) CDF/density. Every entry below is the exact chain
// rule of the Gamma packs under `t = 1/x` (`dt/dx = -1/x^2`); no finite
// differences, no duplicated kernel. Watch the x-signs: the first x
// derivative of the CDF is `+G_t / x^2` and the second is
// `-G_tt / x^4 - 2 G_t / x^3`; for the density
// `p_x = -g_t / x^4 - 2 g / x^3` and `p_xx = g_tt / x^6 + 6 g_t / x^5 +
// 6 g / x^4`.
//
// Edges (same policy as the Gamma packs): invalid shape/scale or NaN x ->
// all-NaN; `x < 0` -> zero pack; `x = 0` -> all-NaN (the one-sided
// derivatives are unbounded); `x = +Inf` -> cdf value 1 / density 0 with all
// partials 0.
// ---------------------------------------------------------------------------

/// (shape, scale, x) value + gradient + Hessian for the InverseGamma CDF.
struct InverseGammaCdfGradHess {
    double value = 0.0;
    double gShape = 0.0;
    double gScale = 0.0;
    double gx = 0.0;
    double hShapeShape = 0.0;
    double hShapeScale = 0.0;
    double hShapeX = 0.0;
    double hScaleScale = 0.0;
    double hScaleX = 0.0;
    double hxx = 0.0;
};

/// All-NaN InverseGamma CDF pack (invalid/edge inputs).
inline InverseGammaCdfGradHess inverseGammaCdfGradHessNaN() noexcept {
    const double nan = std::numeric_limits<double>::quiet_NaN();
    return {nan, nan, nan, nan, nan, nan, nan, nan, nan, nan};
}

/**
 * @brief `{F, F_a, F_s, F_x, F_aa, F_as, F_ax, F_ss, F_sx, F_xx}` for
 * `F(x) = Q(shape, scale/x) = 1 - G(1/x)`, exact chain rule of
 * `gammaCdfGradHess(shape, scale, 1/x)`.
 */
inline InverseGammaCdfGradHess inverseGammaCdfGradHess(double shape, double scale,
                                                       double x) noexcept {
    if (!(shape > 0.0) || !(scale > 0.0) || x != x) {
        return inverseGammaCdfGradHessNaN();
    }
    if (x < 0.0) {
        return InverseGammaCdfGradHess{};
    }
    if (x == 0.0) {
        return inverseGammaCdfGradHessNaN();
    }
    if (isPositiveInfinity(x)) {
        InverseGammaCdfGradHess out{};
        out.value = 1.0;
        return out;
    }

    const GammaCdfGradHess g = gammaCdfGradHess(shape, scale, 1.0 / x);
    const double x2 = x * x;
    const double x3 = x2 * x;
    const double x4 = x2 * x2;

    InverseGammaCdfGradHess out{};
    out.value = 1.0 - g.value;
    out.gShape = -g.gShape;
    out.gScale = -g.gRate;
    out.gx = g.gx / x2;
    out.hShapeShape = -g.hShapeShape;
    out.hShapeScale = -g.hShapeRate;
    out.hShapeX = g.hShapeX / x2;
    out.hScaleScale = -g.hRateRate;
    out.hScaleX = g.hRateX / x2;
    out.hxx = -g.hxx / x4 - 2.0 * g.gx / x3;
    return out;
}

/// (shape, scale, x) value + gradient + Hessian for the InverseGamma density.
struct InverseGammaPdfGradHess {
    double value = 0.0;
    double gShape = 0.0;
    double gScale = 0.0;
    double gx = 0.0;
    double hShapeShape = 0.0;
    double hShapeScale = 0.0;
    double hShapeX = 0.0;
    double hScaleScale = 0.0;
    double hScaleX = 0.0;
    double hxx = 0.0;
};

/// All-NaN InverseGamma density pack (invalid/edge inputs).
inline InverseGammaPdfGradHess inverseGammaPdfGradHessNaN() noexcept {
    const double nan = std::numeric_limits<double>::quiet_NaN();
    return {nan, nan, nan, nan, nan, nan, nan, nan, nan, nan};
}

/**
 * @brief `{p, p_a, p_s, p_x, p_aa, p_as, p_ax, p_ss, p_sx, p_xx}` for the
 * InverseGamma(shape, scale) density `p(x) = g(1/x) / x^2`, exact chain rule
 * of `gammaPdfGradHess(shape, scale, 1/x)`.
 */
inline InverseGammaPdfGradHess inverseGammaPdfGradHess(double shape, double scale,
                                                       double x) noexcept {
    if (!(shape > 0.0) || !(scale > 0.0) || x != x) {
        return inverseGammaPdfGradHessNaN();
    }
    if (x < 0.0 || isPositiveInfinity(x)) {
        return InverseGammaPdfGradHess{};
    }
    if (x == 0.0) {
        return inverseGammaPdfGradHessNaN();
    }

    const GammaPdfGradHess g = gammaPdfGradHess(shape, scale, 1.0 / x);
    const double x2 = x * x;
    const double x3 = x2 * x;
    const double x4 = x2 * x2;
    const double x5 = x4 * x;
    const double x6 = x3 * x3;

    InverseGammaPdfGradHess out{};
    out.value = g.value / x2;
    out.gShape = g.gShape / x2;
    out.gScale = g.gRate / x2;
    out.gx = -g.gx / x4 - 2.0 * g.value / x3;
    out.hShapeShape = g.hShapeShape / x2;
    out.hShapeScale = g.hShapeRate / x2;
    out.hShapeX = -g.hShapeX / x4 - 2.0 * g.gShape / x3;
    out.hScaleScale = g.hRateRate / x2;
    out.hScaleX = -g.hRateX / x4 - 2.0 * g.gRate / x3;
    out.hxx = g.hxx / x6 + 6.0 * g.gx / x5 + 6.0 * g.value / x4;
    return out;
}

} // namespace detail

/**
 * @brief Gamma(shape, rate) with pdf/cdf/ccdf/log forms and quantile.
 *
 * The parameter members are public. Every method is `DoubleT`-generic:
 * `DoubleT = double` dispatches to the named `detail::gamma*Double` entry,
 * other scalars run the generic `detail::gamma*T` kernel (ADL math, hence AD).
 */
template <class T>
struct Gamma {
    T shape;
    T rate;

    /// Density; one-sided limits at `x = 0` (rate for shape = 1, 0/Inf else).
    T pdf(T x) const {
        if constexpr (std::is_same_v<T, double>) {
            return detail::gammaPdfDouble(shape, rate, x);
        } else {
            return detail::gammaPdfT(shape, rate, x);
        }
    }

    /// log density; -Inf at `x <= 0` except the shape = 1 limit `log(rate)`.
    T logPdf(T x) const {
        if constexpr (std::is_same_v<T, double>) {
            return detail::gammaLogPdfDouble(shape, rate, x);
        } else {
            return detail::gammaLogPdfT(shape, rate, x);
        }
    }

    T cdf(T x) const {
        if constexpr (std::is_same_v<T, double>) {
            return detail::gammaCdfDouble(shape, rate, x);
        } else {
            return detail::gammaCdfT(shape, rate, x);
        }
    }

    /// `log P(shape, rate x)`; finite well past the double underflow of cdf.
    T logCdf(T x) const {
        if constexpr (std::is_same_v<T, double>) {
            return detail::gammaLogCdfDouble(shape, rate, x);
        } else {
            return detail::gammaLogCdfT(shape, rate, x);
        }
    }

    T ccdf(T x) const {
        if constexpr (std::is_same_v<T, double>) {
            return detail::gammaCcdfDouble(shape, rate, x);
        } else {
            return detail::gammaCcdfT(shape, rate, x);
        }
    }

    /// `log Q(shape, rate x)`; finite in the deep upper tail.
    T logCcdf(T x) const {
        if constexpr (std::is_same_v<T, double>) {
            return detail::gammaLogCcdfDouble(shape, rate, x);
        } else {
            return detail::gammaLogCcdfT(shape, rate, x);
        }
    }

    /**
     * @brief Quantile: smallest `x` with `cdf(x) >= p`.
     *
     * Wilson-Hilferty seed then safeguarded Newton (`p <= 0 -> 0`,
     * `p >= 1 -> +Inf`, NaN propagates). See the file comment for the
     * interior accuracy target and the documented tail limitations.
     */
    T inv(T p) const {
        if constexpr (std::is_same_v<T, double>) {
            return detail::gammaInvDouble(shape, rate, p);
        } else {
            return detail::gammaInvT(shape, rate, p);
        }
    }

    /// double-only: full gradient/Hessian of the quantile in (p, shape, rate).
    detail::GammaInvGradHess invGradHess(double p) const {
        static_assert(std::is_same_v<T, double>, "invGradHess analytic pack requires T = double");
        return detail::gammaInvGradHess(shape, rate, p);
    }
};

/**
 * @brief InverseGamma(shape, scale): `1 / Gamma(shape, rate = scale)`.
 *
 * `pdf(x) = scale^shape / Gamma(shape) x^(-shape-1) e^(-scale/x)` for
 * `x > 0`; both `pdf(0)` and `pdf(+inf)` are 0. No quantile is provided by
 * the agreed external API. `DoubleT = double` dispatches to the named
 * `detail::invGamma*Double` entries; other scalars run the generic
 * `detail::invGamma*T` kernels.
 */
template <class T>
struct InverseGamma {
    T shape;
    T scale;

    T pdf(T x) const {
        if constexpr (std::is_same_v<T, double>) {
            return detail::invGammaPdfDouble(shape, scale, x);
        } else {
            return detail::invGammaPdfT(shape, scale, x);
        }
    }

    T logPdf(T x) const {
        if constexpr (std::is_same_v<T, double>) {
            return detail::invGammaLogPdfDouble(shape, scale, x);
        } else {
            return detail::invGammaLogPdfT(shape, scale, x);
        }
    }

    T cdf(T x) const {
        if constexpr (std::is_same_v<T, double>) {
            return detail::invGammaCdfDouble(shape, scale, x);
        } else {
            return detail::invGammaCdfT(shape, scale, x);
        }
    }

    T logCdf(T x) const {
        if constexpr (std::is_same_v<T, double>) {
            return detail::invGammaLogCdfDouble(shape, scale, x);
        } else {
            return detail::invGammaLogCdfT(shape, scale, x);
        }
    }

    T ccdf(T x) const {
        if constexpr (std::is_same_v<T, double>) {
            return detail::invGammaCcdfDouble(shape, scale, x);
        } else {
            return detail::invGammaCcdfT(shape, scale, x);
        }
    }

    T logCcdf(T x) const {
        if constexpr (std::is_same_v<T, double>) {
            return detail::invGammaLogCcdfDouble(shape, scale, x);
        } else {
            return detail::invGammaLogCcdfT(shape, scale, x);
        }
    }
};

} // namespace quantape::math

#endif // QUANTAPE_MATH_DISTRIBUTIONS_GAMMA_H
