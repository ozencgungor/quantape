#ifndef QUANTAPE_MATH_DISTRIBUTIONS_LOGNORMAL_H
#define QUANTAPE_MATH_DISTRIBUTIONS_LOGNORMAL_H

#include "quantape/math/Random/InverseNormal.h"
#include "quantape/util/Constants.h"

#include <cmath>
#include <limits>
#include <type_traits>

/**
 * @file Lognormal.h
 * @brief Log-normal distribution: pdf/cdf/tails plus x-direction derivative
 * packs, std-only and scalar-generic.
 *
 * X = exp(mu + sigma Z), Z ~ N(0, 1), support x > 0. With z = (ln x - mu)/sigma:
 *
 *   pdf(x)    = exp(-z^2 / 2) / (x sigma sqrt(2 pi))
 *   cdf(x)    = Phi(z) = erfc(-z / sqrt(2)) / 2
 *   ccdf(x)   = 1 - cdf(x) = Phi(-z)
 *   logPdf(x) = -z^2 / 2 - ln x - ln sigma - ln sqrt(2 pi)
 *   logCdf    = ln Phi(z), logCcdf = ln Phi(-z)
 *
 * Derivatives (x > 0, q = 1 + z / sigma):
 *
 *   pdf'(x)  = -pdf(x) q / x
 *   pdf''(x) = pdf(x) (q^2 - 1 / sigma^2 + q) / x^2
 *   cdf'(x)  = pdf(x),   cdf''(x) = pdf'(x)
 *
 * Accuracy: Phi is std::erfc based, full double accuracy in the body; the
 * far-left tail underflows to cdf = 0 below z ~ -38 (logCdf = -inf there,
 * a stable log-Phi asymptotic is future work). The derivative packs are exact
 * analytic expressions (no finite differences) and are verified against
 * central differences in the unit tests.
 *
 * Parameter derivative packs (double only): `lognormalPdfGradHess` and
 * `lognormalCdfGradHess`, taking (mu, sigma, x), return the full 3x3
 * gradient/Hessian in (mu, sigma, x), all closed form from z, phi(z), Phi(z)
 * and the 1/x factors (z_mu = -1/sigma, z_sigma = -z/sigma, z_x = 1/(x sigma);
 * no finite differences). The value and the x-direction rows are the existing
 * `lognormalPdfD1D2`/`lognormalCdfD1D2` anchors. Boundary behavior: x <= 0
 * returns the zero pack (value 0, all partials 0); x = +inf returns the pdf
 * zero pack and the cdf pack {value 1, all partials 0} (saturating limits);
 * deep-underflow x (pdf or cdf underflows to 0) returns the zero pack, since
 * every partial underflows with the value; NaN propagates through every entry.
 * These feed the double-side Greeks and, later, External payloads; they are
 * independent of any AD scalar type.
 *
 * inv(p) = exp(mu + sigma * inverseNormal(p)) reuses Acklam's rational
 * approximation from Random/InverseNormal.h: maximum relative error ~1.15e-9
 * in the normal quantile (seed-quality; an AS241/Newton refinement is planned).
 * The inverse is double-seeded, so it is exact only to that seed accuracy and
 * carries no derivative information with respect to p. p <= 0 maps to 0 and
 * p >= 1 to +inf via the +-1e10 endpoint clamp.
 *
 * Edge cases: x <= 0 gives pdf = cdf = 0, logPdf = logCdf = -inf, ccdf = 1,
 * logCcdf = 0; x = +inf gives pdf = 0, cdf = 1, ccdf = 0, logPdf = logCcdf =
 * -inf; NaN propagates through every method. Precondition: sigma > 0 (sigma = 0
 * produces division-by-zero infinities; no guard, no throw).
 *
 * Two-tier dispatch (the Normal.h pattern): `T = double` accessors call the
 * named `detail::lognormal*Double` kernels, which forward to the shared
 * templated kernels explicitly instantiated for double; other scalars run the
 * generic kernel path. `inv` keeps its double-seeded solve.
 */
namespace quantape::math {
using ::quantape::util::kInvSqrt2;
using ::quantape::util::kSqrt2Pi;

namespace detail {

/// Value plus first and second x-derivatives. Unique to this header so the
/// distribution headers may be included together in one translation unit.
template <class T>
struct LognormalD1D2 {
    T value;
    T d1;
    T d2;
};

/// Value plus the full gradient/Hessian of the lognormal pdf in (mu, sigma,
/// x), all closed form. Field order follows the (mu, sigma, x) parameter
/// order of `lognormalPdfGradHess` (note Normal.h uses the (x, mu, sigma)
/// convention for its own pack, which is a different family).
struct LognormalPdfGradHess {
    double value = 0.0;
    double gmu = 0.0;
    double gsig = 0.0;
    double gx = 0.0;
    double hmumu = 0.0;
    double hmusig = 0.0;
    double hmux = 0.0;
    double hsigsig = 0.0;
    double hsigx = 0.0;
    double hxx = 0.0;
};

/// Value plus the full gradient/Hessian of the lognormal cdf in (mu, sigma,
/// x); same field layout as `LognormalPdfGradHess`.
struct LognormalCdfGradHess {
    double value = 0.0;
    double gmu = 0.0;
    double gsig = 0.0;
    double gx = 0.0;
    double hmumu = 0.0;
    double hmusig = 0.0;
    double hmux = 0.0;
    double hsigsig = 0.0;
    double hsigx = 0.0;
    double hxx = 0.0;
};

/// Standard normal CDF through std::erfc (local copy: Normal.h is a sibling
/// module and must not be a dependency here).
template <class T>
inline T lognormalPhi(T z) {
    using std::erfc;
    return T(0.5) * erfc(-z * T(kInvSqrt2.hi));
}

/// pdf and its first two x-derivatives; x <= 0 returns the zero pack.
template <class T>
inline LognormalD1D2<T> lognormalPdfD1D2(T x, T mu, T sigma) {
    using std::exp;
    using std::log;
    const T zero(0.0);
    if (x <= zero) {
        return {zero, zero, zero};
    }
    const T z = (log(x) - mu) / sigma;
    const T q = T(1.0) + z / sigma;
    const T pdf = exp(-T(0.5) * z * z) / (x * sigma * T(kSqrt2Pi.hi));
    return {pdf, -pdf * q / x, pdf * (q * q - T(1.0) / (sigma * sigma) + q) / (x * x)};
}

/// cdf plus its first two x-derivatives (cdf' = pdf, cdf'' = pdf').
template <class T>
inline LognormalD1D2<T> lognormalCdfD1D2(T x, T mu, T sigma) {
    using std::log;
    const T zero(0.0);
    if (x <= zero) {
        return {zero, zero, zero};
    }
    const T z = (log(x) - mu) / sigma;
    const LognormalD1D2<T> p = lognormalPdfD1D2(x, mu, sigma);
    return {lognormalPhi(z), p.value, p.d1};
}

/**
 * @brief pdf value, gradient and Hessian in (mu, sigma, x), all closed form.
 *
 * With `z = (ln x - mu) / sigma`, `n = phi(z)`, `p = n / (x sigma)`:
 * `p_mu = z p / sigma`, `p_sigma = p (z^2 - 1) / sigma`,
 * `p_mumu = p (z^2 - 1) / sigma^2`, `p_musigma = z p (z^2 - 3) / sigma^2`,
 * `p_sigsig = p (z^4 - 5 z^2 + 2) / sigma^2`; the x rows carry the 1/x
 * factors of the log transform (`p_x = -p q / x` with `q = 1 + z/sigma`,
 * `p_mux = p (1 - z^2 - z sigma) / (x sigma^2)`, etc.). x <= 0 and x = +inf
 * return the zero pack (the density and every partial vanish).
 */
inline LognormalPdfGradHess lognormalPdfGradHess(double mu, double sigma, double x) {
    LognormalPdfGradHess out;
    if (x <= 0.0 || std::isinf(x)) {
        return out;
    }
    const LognormalD1D2<double> p = lognormalPdfD1D2<double>(x, mu, sigma);
    out.value = p.value;
    if (p.value == 0.0) {
        return out; // density underflowed: every partial underflows with it
    }
    out.gx = p.d1;
    out.hxx = p.d2;
    using std::log;
    const double z = (log(x) - mu) / sigma;
    const double z2 = z * z;
    out.gmu = z * p.value / sigma;
    out.gsig = p.value * (z2 - 1.0) / sigma;
    out.hmumu = p.value * (z2 - 1.0) / (sigma * sigma);
    out.hmusig = z * p.value * (z2 - 3.0) / (sigma * sigma);
    out.hmux = p.value * (1.0 - z2 - z * sigma) / (x * sigma * sigma);
    out.hsigsig = p.value * (z2 * z2 - 5.0 * z2 + 2.0) / (sigma * sigma);
    out.hsigx = p.value * ((1.0 - z2) + z * (3.0 - z2) / sigma) / (x * sigma);
    return out;
}

/**
 * @brief cdf value, gradient and Hessian in (mu, sigma, x), all closed form.
 *
 * With `n = phi(z)` and `p = n / (x sigma)`: `F_mu = -n / sigma`,
 * `F_sigma = -z n / sigma`, `F_x = p`; `F_mumu = -z n / sigma^2`,
 * `F_musigma = n (1 - z^2) / sigma^2`, `F_sigsig = -z n (z^2 - 2) / sigma^2`,
 * `F_mux = z n / (x sigma^2)`, `F_sigmax = n (z^2 - 1) / (x sigma^2)`,
 * `F_xx = -p q / x`. x <= 0 returns the zero pack; x = +inf returns
 * `{value 1, all partials 0}` (the saturating limits); in the deep left tail
 * the anchored `gx`/`hxx` can be 0/0 = NaN while `value` underflows to 0, so
 * those rows are collapsed to their 0 limit there (NaN inputs still propagate).
 */
inline LognormalCdfGradHess lognormalCdfGradHess(double mu, double sigma, double x) {
    LognormalCdfGradHess out;
    if (x <= 0.0) {
        return out;
    }
    if (std::isinf(x)) {
        out.value = 1.0;
        return out;
    }
    const LognormalD1D2<double> c = lognormalCdfD1D2<double>(x, mu, sigma);
    const LognormalD1D2<double> p = lognormalPdfD1D2<double>(x, mu, sigma);
    out.value = c.value;
    out.gx = c.d1;
    out.hxx = c.d2;
    // Deep-tail underflow can turn the anchored x rows into 0/0 = NaN while
    // the cdf itself underflows to 0; their true limit is 0. NaN inputs leave
    // c.value NaN, so they are not collapsed (every entry propagates NaN).
    if (c.value == 0.0 && std::isnan(out.gx)) {
        out.gx = 0.0;
    }
    if (c.value == 0.0 && std::isnan(out.hxx)) {
        out.hxx = 0.0;
    }
    using std::log;
    const double z = (log(x) - mu) / sigma;
    const double z2 = z * z;
    const double n = p.value * x * sigma; // phi(z)
    out.gmu = -n / sigma;
    out.gsig = -z * n / sigma;
    out.hmumu = -z * n / (sigma * sigma);
    out.hmusig = n * (1.0 - z2) / (sigma * sigma);
    out.hmux = z * n / (x * sigma * sigma);
    out.hsigsig = -z * n * (z2 - 2.0) / (sigma * sigma);
    out.hsigx = n * (z2 - 1.0) / (x * sigma * sigma);
    return out;
}

// ---------------------------------------------------------------------------
// Quantile parameter-derivative pack.
//
// Same implicit differentiation as in Normal.h: with `f = F_x`, `f' = F_xx`
// at `x = Q`, `Q_p = 1/f`, `Q_theta = -F_theta/f`,
// `Q_pp = -f'/f^3`, `Q_p,theta = -(F_x,theta + f' Q_theta)/f^2`,
// `Q_theta,theta = -(F_theta,theta + 2 F_theta,x Q_theta + f' Q_theta^2)/f`.
// The `value` is bitwise `Lognormal::inv(p)`; outside the open unit interval
// the derivative entries are NaN (`value` keeps the 0/+Inf endpoint
// convention). No finite differences.
// ---------------------------------------------------------------------------

/// (p, mu, sigma) value + gradient + Hessian of the lognormal quantile.
struct LognormalInvGradHess {
    double value = 0.0;
    double gP = 0.0;
    double gmu = 0.0;
    double gsig = 0.0;
    double hPP = 0.0;
    double hPmu = 0.0;
    double hPsig = 0.0;
    double hmumu = 0.0;
    double hmusig = 0.0;
    double hsigsig = 0.0;
};

/// `{Q, Q_p, Q_mu, Q_sigma, Q_pp, Q_pmu, Q_psigma, Q_mumu, Q_musig,
/// Q_sigmasig}` for the lognormal quantile; see the section comment.
inline LognormalInvGradHess lognormalInvGradHess(double mu, double sigma, double p) {
    using std::exp;
    const double nan = std::numeric_limits<double>::quiet_NaN();
    const double z = inverseNormal(p);
    LognormalInvGradHess out;
    if (z >= 1e10) {
        out.value = std::numeric_limits<double>::infinity();
    } else if (z <= -1e10) {
        out.value = 0.0;
    } else {
        out.value = exp(mu + sigma * z);
    }
    if (!(p > 0.0) || !(p < 1.0)) {
        out.gP = nan;
        out.gmu = nan;
        out.gsig = nan;
        out.hPP = nan;
        out.hPmu = nan;
        out.hPsig = nan;
        out.hmumu = nan;
        out.hmusig = nan;
        out.hsigsig = nan;
        return out;
    }
    const LognormalCdfGradHess c = lognormalCdfGradHess(mu, sigma, out.value);
    const double f = c.gx;
    const double fp = c.hxx;
    out.gP = 1.0 / f;
    out.gmu = -c.gmu / f;
    out.gsig = -c.gsig / f;
    out.hPP = -fp / (f * f * f);
    out.hPmu = -(c.hmux + fp * out.gmu) / (f * f);
    out.hPsig = -(c.hsigx + fp * out.gsig) / (f * f);
    out.hmumu = -(c.hmumu + 2.0 * c.hmux * out.gmu + fp * out.gmu * out.gmu) / f;
    out.hmusig = -(c.hmusig + c.hmux * out.gsig + c.hsigx * out.gmu + fp * out.gmu * out.gsig) / f;
    out.hsigsig = -(c.hsigsig + 2.0 * c.hsigx * out.gsig + fp * out.gsig * out.gsig) / f;
    return out;
}

/// log pdf (x <= 0 -> -inf); the `T` body of `Lognormal::logPdf`.
template <class T>
inline T lognormalLogPdfT(T x, T mu, T sigma) {
    using std::log;
    const T zero(0.0);
    if (x <= zero) {
        return -std::numeric_limits<T>::infinity();
    }
    const T z = (log(x) - mu) / sigma;
    return -T(0.5) * z * z - log(x) - log(sigma) - log(T(kSqrt2Pi.hi));
}

/// log cdf (x <= 0 -> -inf); the `T` body of `Lognormal::logCdf`.
template <class T>
inline T lognormalLogCdfT(T x, T mu, T sigma) {
    using std::log;
    const T zero(0.0);
    if (x <= zero) {
        return -std::numeric_limits<T>::infinity();
    }
    return log(lognormalPhi((log(x) - mu) / sigma));
}

/// complementary cdf (x <= 0 -> 1); the `T` body of `Lognormal::ccdf`.
template <class T>
inline T lognormalCcdfT(T x, T mu, T sigma) {
    using std::log;
    const T zero(0.0);
    if (x <= zero) {
        return T(1.0);
    }
    return lognormalPhi(-(log(x) - mu) / sigma);
}

/// log complementary cdf (x <= 0 -> 0); the `T` body of `Lognormal::logCcdf`.
template <class T>
inline T lognormalLogCcdfT(T x, T mu, T sigma) {
    using std::log;
    const T zero(0.0);
    if (x <= zero) {
        return zero;
    }
    return log(lognormalPhi(-(log(x) - mu) / sigma));
}

// ---------------------------------------------------------------------------
// Named double kernels (the `T = double` fast path of `Lognormal<T>`). Each
// entry forwards to the shared templated kernel explicitly instantiated for
// double, so every method has one canonical double definition and no
// algorithm body is duplicated. `inv` keeps its own double-seeded solve.
// ---------------------------------------------------------------------------

/// pdf of Lognormal(mu, sigma^2) at x (double).
inline double lognormalPdfDouble(double x, double mu, double sigma) {
    return lognormalPdfD1D2<double>(x, mu, sigma).value;
}

/// log pdf (double).
inline double lognormalLogPdfDouble(double x, double mu, double sigma) {
    return lognormalLogPdfT<double>(x, mu, sigma);
}

/// cdf of Lognormal(mu, sigma^2) at x (double).
inline double lognormalCdfDouble(double x, double mu, double sigma) {
    return lognormalCdfD1D2<double>(x, mu, sigma).value;
}

/// log cdf (double).
inline double lognormalLogCdfDouble(double x, double mu, double sigma) {
    return lognormalLogCdfT<double>(x, mu, sigma);
}

/// complementary cdf (double).
inline double lognormalCcdfDouble(double x, double mu, double sigma) {
    return lognormalCcdfT<double>(x, mu, sigma);
}

/// log complementary cdf (double).
inline double lognormalLogCcdfDouble(double x, double mu, double sigma) {
    return lognormalLogCcdfT<double>(x, mu, sigma);
}

} // namespace detail

template <class T>
struct Lognormal {
    T mu;    ///< log-mean.
    T sigma; ///< log-standard-deviation, sigma > 0.

    T pdf(T x) const {
        if constexpr (std::is_same_v<T, double>) {
            return detail::lognormalPdfDouble(x, mu, sigma);
        } else {
            return detail::lognormalPdfD1D2(x, mu, sigma).value;
        }
    }

    T logPdf(T x) const {
        if constexpr (std::is_same_v<T, double>) {
            return detail::lognormalLogPdfDouble(x, mu, sigma);
        } else {
            return detail::lognormalLogPdfT(x, mu, sigma);
        }
    }

    T cdf(T x) const {
        if constexpr (std::is_same_v<T, double>) {
            return detail::lognormalCdfDouble(x, mu, sigma);
        } else {
            return detail::lognormalCdfD1D2(x, mu, sigma).value;
        }
    }

    T logCdf(T x) const {
        if constexpr (std::is_same_v<T, double>) {
            return detail::lognormalLogCdfDouble(x, mu, sigma);
        } else {
            return detail::lognormalLogCdfT(x, mu, sigma);
        }
    }

    T ccdf(T x) const {
        if constexpr (std::is_same_v<T, double>) {
            return detail::lognormalCcdfDouble(x, mu, sigma);
        } else {
            return detail::lognormalCcdfT(x, mu, sigma);
        }
    }

    T logCcdf(T x) const {
        if constexpr (std::is_same_v<T, double>) {
            return detail::lognormalLogCcdfDouble(x, mu, sigma);
        } else {
            return detail::lognormalLogCcdfT(x, mu, sigma);
        }
    }

    /// double-only: full gradient/Hessian of the pdf in (mu, sigma, x).
    detail::LognormalPdfGradHess pdfGradHess(double x) const {
        static_assert(std::is_same_v<T, double>, "pdfGradHess analytic pack requires T = double");
        return detail::lognormalPdfGradHess(mu, sigma, x);
    }

    /// double-only: full gradient/Hessian of the cdf in (mu, sigma, x).
    detail::LognormalCdfGradHess cdfGradHess(double x) const {
        static_assert(std::is_same_v<T, double>, "cdfGradHess analytic pack requires T = double");
        return detail::lognormalCdfGradHess(mu, sigma, x);
    }

    /// double-only: full gradient/Hessian of the quantile in (p, mu, sigma).
    detail::LognormalInvGradHess invGradHess(double p) const {
        static_assert(std::is_same_v<T, double>, "invGradHess analytic pack requires T = double");
        return detail::lognormalInvGradHess(mu, sigma, p);
    }

    /// Quantile exp(mu + sigma * inverseNormal(p)); seed-limited to ~1.15e-9
    /// relative (Acklam), p <= 0 -> 0, p >= 1 -> +inf.
    T inv(T p) const {
        using std::exp;
        const double z = inverseNormal(static_cast<double>(p));
        if (z >= 1e10) {
            return std::numeric_limits<T>::infinity();
        }
        if (z <= -1e10) {
            return T(0.0);
        }
        return exp(mu + sigma * T(z));
    }
};

} // namespace quantape::math

#endif // QUANTAPE_MATH_DISTRIBUTIONS_LOGNORMAL_H
