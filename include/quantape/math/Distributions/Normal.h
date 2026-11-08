#ifndef QUANTAPE_MATH_DISTRIBUTIONS_NORMAL_H
#define QUANTAPE_MATH_DISTRIBUTIONS_NORMAL_H

/**
 * @file Normal.h
 * @brief Normal and standard normal distribution (pdf, cdf, ccdf, log forms, quantile)
 *
 * Double kernels plus the `DoubleT`-templated `Normal<DoubleT>` type (the
 * Black76 pattern): `DoubleT = double` uses the optimized kernels below;
 * other scalars (future in-house AD, Stan during migration) run the generic
 * scalar path with ADL math.
 *
 * Accuracy / algorithms:
 *  - `pdf` uses the closed form; <=2 ulp.
 *  - `cdf` uses `0.5*erfc(-z/sqrt(2))`; absolute error ~1e-16 central.
 *  - `logCdf`/`logCcdf` switch to the Mills-ratio asymptotic expansion below
 *    z = -37.5 (value-space cdf would underflow), so the log survives to
 *    z ~ -1e4.
 *  - `inv` ports Wichura's AS241 rational approximations (the same coefficient
 *    sets Stan Math uses; BSD-3, see NOTICE) with the p >= 0.9999 evaluation
 *    trick for the upper tail.
 *  - Analytic x-direction first/second derivatives (`pdfD1D2`, `cdfD1D2`) and
 *    the full parameter gradient/Hessian (`pdfGradHess`, `cdfGradHess`) are
 *    provided in `detail`; these feed the double-side Greeks and, later,
 *    External payloads.
 *
 * std-only (util/Constants.h plus <cmath>); no allocation; NaN/Inf propagate.
 */

#include "quantape/util/Constants.h"

#include <cmath>
#include <limits>
#include <type_traits>

#include "detail/NormalQuantile.h"

namespace quantape::math {
using ::quantape::util::kInvSqrt2;
using ::quantape::util::kPi;
using ::quantape::util::kSqrt2Pi;

namespace detail {

/// value plus first and second x-derivatives of a double kernel.
struct NormalD1D2 {
    double value = 0.0;
    double d1 = 0.0;
    double d2 = 0.0;
};

/// cdf value, gradient and Hessian in (x, mu, sigma).
struct NormalCdfGradHess {
    double value = 0.0;
    double gx = 0.0;
    double gmu = 0.0;
    double gsig = 0.0;
    double hxx = 0.0;
    double hxmu = 0.0;
    double hxsig = 0.0;
    double hmumu = 0.0;
    double hmusig = 0.0;
    double hsigsig = 0.0;
};

/// pdf value, gradient and Hessian in (x, mu, sigma).
struct NormalPdfGradHess {
    double value = 0.0;
    double gx = 0.0;
    double gmu = 0.0;
    double gsig = 0.0;
    double hxx = 0.0;
    double hxmu = 0.0;
    double hxsig = 0.0;
    double hmumu = 0.0;
    double hmusig = 0.0;
    double hsigsig = 0.0;
};

/// pdf of N(mu, sigma^2) at x (double).
inline double normalPdfDouble(double x, double mu, double sigma) {
    const double z = (x - mu) / sigma;
    return std::exp(-0.5 * z * z) / (sigma * static_cast<double>(kSqrt2Pi));
}

/// cdf of N(mu, sigma^2) at x (double).
inline double normalCdfDouble(double x, double mu, double sigma) {
    const double z = (x - mu) / sigma;
    return 0.5 * std::erfc(-z * static_cast<double>(kInvSqrt2));
}

/// ccdf of N(mu, sigma^2) at x (double).
inline double normalCcdfDouble(double x, double mu, double sigma) {
    const double z = (x - mu) / sigma;
    return 0.5 * std::erfc(z * static_cast<double>(kInvSqrt2));
}

/// log cdf with a Mills-ratio asymptotic below z = -37.5 (log of zero avoided).
inline double normalLogCdfDouble(double x, double mu, double sigma) {
    const double z = (x - mu) / sigma;
    if (z > -37.5) {
        return std::log(normalCdfDouble(x, mu, sigma));
    }
    const double t = -z;
    const double t2 = t * t;
    // Phi(-t) = phi(t)/t * (1 - 1/t^2 + 3/t^4 - 15/t^6 + ...)
    const double corr = 1.0 - 1.0 / t2 + 3.0 / (t2 * t2) - 15.0 / (t2 * t2 * t2);
    return -0.5 * t2 - std::log(t) - 0.5 * std::log(2.0 * static_cast<double>(kPi)) +
           std::log(corr);
}

/// log ccdf with the same asymptotic above z = 37.5.
inline double normalLogCcdfDouble(double x, double mu, double sigma) {
    const double z = (x - mu) / sigma;
    if (z < 37.5) {
        return std::log(normalCcdfDouble(x, mu, sigma));
    }
    const double t = z;
    const double t2 = t * t;
    const double corr = 1.0 - 1.0 / t2 + 3.0 / (t2 * t2) - 15.0 / (t2 * t2 * t2);
    return -0.5 * t2 - std::log(t) - 0.5 * std::log(2.0 * static_cast<double>(kPi)) +
           std::log(corr);
}

// ---------------------------------------------------------------------------
// Generic scalar paths (double uses the kernels above via the struct below).
// ---------------------------------------------------------------------------

template <class T>
T normalPdfT(T x, T mu, T sigma) {
    using std::exp;
    const T z = (x - mu) / sigma;
    return exp(-T(0.5) * z * z) / (sigma * T(static_cast<double>(kSqrt2Pi)));
}

template <class T>
T normalCdfT(T x, T mu, T sigma) {
    using std::erfc;
    const T z = (x - mu) / sigma;
    return T(0.5) * erfc(-z * T(static_cast<double>(kInvSqrt2)));
}

template <class T>
T normalCcdfT(T x, T mu, T sigma) {
    using std::erfc;
    const T z = (x - mu) / sigma;
    return T(0.5) * erfc(z * T(static_cast<double>(kInvSqrt2)));
}

template <class T>
T normalLogCdfT(T x, T mu, T sigma) {
    using std::log;
    return log(normalCdfT(x, mu, sigma));
}

template <class T>
T normalLogCcdfT(T x, T mu, T sigma) {
    using std::log;
    return log(normalCcdfT(x, mu, sigma));
}

template <class T>
T normalInvT(T p) {
    if constexpr (std::is_same_v<T, double>) {
        return normalInvDouble(p);
    } else {
        using std::fabs;
        using std::log;
        using std::sqrt;
        // The AS241 coefficient sets, indexed by increasing power, so Horner
        // starts at the highest index. The first three sets are the numerators
        // of degrees 7 and 6; the denominators are 1 + r*P(r) with P of degree 6.
        static constexpr double kA[8] = {3.3871328727963666080e+00, 1.3314166789178437745e+02,
                                         1.9715909503065514427e+03, 1.3731693765509461125e+04,
                                         4.5921953931549871457e+04, 6.7265770927008700853e+04,
                                         3.3430575583588128105e+04, 2.5090809287301226727e+03};
        static constexpr double kB[7] = {4.2313330701600911252e+01, 6.8718700749205790830e+02,
                                         5.3941960214247511077e+03, 2.1213794301586595867e+04,
                                         3.9307895800092710610e+04, 2.8729085735721942674e+04,
                                         5.2264952788528545610e+03};
        static constexpr double kC[8] = {1.42343711074968357734e+00, 4.63033784615654529590e+00,
                                         5.76949722146069140550e+00, 3.64784832476320460504e+00,
                                         1.27045825245236838258e+00, 2.41780725177450611770e-01,
                                         2.27238449892691845833e-02, 7.74545014278341407640e-04};
        static constexpr double kD[7] = {2.05319162663775882187e+00, 1.67638483018380384940e+00,
                                         6.89767334985100004550e-01, 1.48103976427480074590e-01,
                                         1.51986665636164571966e-02, 5.47593808499534494600e-04,
                                         1.05075007164441684324e-09};
        static constexpr double kE[8] = {6.65790464350110377720e+00, 5.46378491116411436990e+00,
                                         1.78482653991729133580e+00, 2.96560571828504891230e-01,
                                         2.65321895265761230930e-02, 1.24266094738807843860e-03,
                                         2.71155556874348757815e-05, 2.01033439929228813265e-07};
        static constexpr double kF[7] = {5.99832206555887937690e-01, 1.36929880922735805310e-01,
                                         1.48753612908506148525e-02, 7.86869131145613259100e-04,
                                         1.84631831751005468180e-05, 1.42151175831644588870e-07,
                                         2.04426310338993978564e-15};
        const auto horner = [](const double* c, int top, const T& x) {
            T acc = T(c[top]);
            for (int i = top - 1; i >= 0; --i) {
                acc = acc * x + T(c[i]);
            }
            return acc;
        };

        const T q = p - T(0.5);
        if (fabs(q) <= T(0.425)) {
            const T r = T(0.180625) - q * q;
            // AS241's denominators are 1 + r*P(r), not P(r): P is degree 6 and
            // the constant term is 1.
            return q * horner(kA, 7, r) / (horner(kB, 6, r) * r + T(1.0));
        }
        T r = q < T(0.0) ? p : T(1.0) - p;
        if (r <= T(0.0)) {
            return T(0.0);
        }
        r = sqrt(-log(r));
        T val = T(0.0);
        if (r <= T(5.0)) {
            r -= T(1.6);
            val = horner(kC, 7, r) / (horner(kD, 6, r) * r + T(1.0));
        } else {
            r -= T(5.0);
            val = horner(kE, 7, r) / (horner(kF, 6, r) * r + T(1.0));
        }
        const auto quantile = [&](const T& x) -> T {
            const T q = x - T(0.5);
            if (fabs(q) <= T(0.425)) {
                const T r = T(0.180625) - q * q;
                // AS241's denominators are 1 + r*P(r), not P(r): P is degree 6
                // and the constant term is 1.
                return q * horner(kA, 7, r) / (horner(kB, 6, r) * r + T(1.0));
            }
            T r = q < T(0.0) ? x : T(1.0) - x;
            if (r <= T(0.0)) {
                return T(0.0);
            }
            r = sqrt(-log(r));
            T val = T(0.0);
            if (r <= T(5.0)) {
                r -= T(1.6);
                val = horner(kC, 7, r) / (horner(kD, 6, r) * r + T(1.0));
            } else {
                r -= T(5.0);
                val = horner(kE, 7, r) / (horner(kF, 6, r) * r + T(1.0));
            }
            return q < T(0.0) ? -val : val;
        };
        // `normalInvDouble` switches to a (2e9 - 2e9*p)/2e9 complement for
        // p >= 0.9999. Do NOT mirror that here: `2e9 * p` is rounded at ulp
        // 4.8e-7, which is 1e-16 absolute but ~1e-7 *relative* for the
        // complement, costing ~5e-9 in the quantile at p = 1 - 1e-9. Evaluating
        // `1 - p` directly (exact by Sterbenz) is the more accurate route and
        // agrees with `as241(p)` to a few ulp.
        return quantile(p);
    }
}

/// value, d/dx, d^2/dx^2 of the normal pdf.
inline NormalD1D2 normalPdfD1D2(double x, double mu, double sigma) {
    const double z = (x - mu) / sigma;
    const double pdf = normalPdfDouble(x, mu, sigma);
    return {pdf, -z * pdf / sigma, (z * z - 1.0) * pdf / (sigma * sigma)};
}

/// value, d/dx (=pdf), d^2/dx^2 (=pdf') of the normal cdf.
inline NormalD1D2 normalCdfD1D2(double x, double mu, double sigma) {
    const double z = (x - mu) / sigma;
    const double pdf = normalPdfDouble(x, mu, sigma);
    return {normalCdfDouble(x, mu, sigma), pdf, -z * pdf / sigma};
}

/// cdf with the full analytic first/second partials in (x, mu, sigma).
inline NormalCdfGradHess normalCdfGradHess(double x, double mu, double sigma) {
    const double z = (x - mu) / sigma;
    const double p = normalPdfDouble(x, mu, sigma); // phi(z)/sigma
    NormalCdfGradHess out;
    out.value = normalCdfDouble(x, mu, sigma);
    out.gx = p;
    out.gmu = -p;
    out.gsig = -p * z;
    out.hxx = -z * p / sigma;
    out.hxmu = z * p / sigma;
    out.hxsig = (z * z - 1.0) * p / sigma;
    out.hmumu = -z * p / sigma;
    out.hmusig = -(z * z - 1.0) * p / sigma;
    out.hsigsig = -z * (z * z - 2.0) * p / sigma;
    return out;
}

/**
 * @brief pdf with the full analytic first/second partials in (x, mu, sigma).
 *
 * With `z = (x - mu) / sigma` and `p = phi(z) / sigma` the log-density is
 * `L = -z^2/2 - log sigma - const`, so the gradient is `p L'` and the Hessian
 * is `p (L' L'^T + L'')` with
 *   L = (-z, z, z^2 - 1) / sigma,
 *   L'' = [-1, 1, 2z; 1, -1, -2z; 2z, -2z, 1 - 3 z^2] / sigma^2,
 * which is the closed form below.
 */
inline NormalPdfGradHess normalPdfGradHess(double x, double mu, double sigma) {
    const double z = (x - mu) / sigma;
    const double p = normalPdfDouble(x, mu, sigma); // phi(z) / sigma
    const double z2 = z * z;
    const double pOverSigma2 = p / (sigma * sigma);
    NormalPdfGradHess out;
    out.value = p;
    out.gx = -z * p / sigma;
    out.gmu = z * p / sigma;
    out.gsig = (z2 - 1.0) * p / sigma;
    out.hxx = (z2 - 1.0) * pOverSigma2;
    out.hxmu = (1.0 - z2) * pOverSigma2;
    out.hxsig = z * (3.0 - z2) * pOverSigma2;
    out.hmumu = (z2 - 1.0) * pOverSigma2;
    out.hmusig = z * (z2 - 3.0) * pOverSigma2;
    out.hsigsig = (z2 * z2 - 5.0 * z2 + 2.0) * pOverSigma2;
    return out;
}

// ---------------------------------------------------------------------------
// Quantile parameter-derivative pack.
//
// General method (the same in every distribution header): differentiating the
// identity `F(Q(p, theta), theta) = p` implicitly, with `f = F_x` and
// `f' = F_xx` evaluated at `x = Q`, gives
//
//   Q_p        = 1 / f,
//   Q_theta_j  = -F_theta_j / f,
//   Q_pp       = -f' / f^3,
//   Q_p,theta_j = -(F_x,theta_j + f' Q_theta_j) / f^2,
//   Q_theta_j,theta_k = -(F_theta_j,theta_k + F_theta_j,x Q_theta_k
//                         + F_theta_k,x Q_theta_j + f' Q_theta_j Q_theta_k) / f,
//
// where the cdf parameter partials come from the existing cdf pack at the
// quantile. Every entry is exact analytic algebra of that pack; no finite
// differences. The pack `value` is bitwise `Normal::inv(p)`; outside the open
// unit interval the derivatives are NaN (the quantile is not differentiable
// at the endpoints) while `value` keeps the `inv` endpoint convention
// (`-Inf`/`+Inf` at 0/1, NaN outside).
// ---------------------------------------------------------------------------

/// (p, mu, sigma) value + gradient + Hessian of the normal quantile.
struct NormalInvGradHess {
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
/// Q_sigmasig}` for the normal quantile; see the section comment.
inline NormalInvGradHess normalInvGradHess(double p, double mu, double sigma) {
    const double nan = std::numeric_limits<double>::quiet_NaN();
    NormalInvGradHess out;
    out.value = mu + sigma * normalInvDouble(p);
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
    const NormalCdfGradHess c = normalCdfGradHess(out.value, mu, sigma);
    const double f = c.gx;
    const double fp = c.hxx;
    out.gP = 1.0 / f;
    out.gmu = -c.gmu / f;
    out.gsig = -c.gsig / f;
    out.hPP = -fp / (f * f * f);
    out.hPmu = -(c.hxmu + fp * out.gmu) / (f * f);
    out.hPsig = -(c.hxsig + fp * out.gsig) / (f * f);
    out.hmumu = -(c.hmumu + 2.0 * c.hxmu * out.gmu + fp * out.gmu * out.gmu) / f;
    out.hmusig = -(c.hmusig + c.hxmu * out.gsig + c.hxsig * out.gmu + fp * out.gmu * out.gsig) / f;
    out.hsigsig = -(c.hsigsig + 2.0 * c.hxsig * out.gsig + fp * out.gsig * out.gsig) / f;
    return out;
}

} // namespace detail

/**
 * @brief Normal distribution N(mu, sigma^2) with pdf/cdf/ccdf/log forms and quantile.
 *
 * The parameter members are public (Black76 pattern). All methods are
 * `DoubleT`-generic; the double instantiations call the optimized kernels,
 * other scalars use the generic path (ADL math). The `cdfGradHess`/`pdfD1D2`/
 * `cdfD1D2` methods are double-only analytic derivative entry points.
 */
template <class DoubleT>
struct Normal {
    DoubleT mu;
    DoubleT sigma;

    DoubleT pdf(DoubleT x) const {
        if constexpr (std::is_same_v<DoubleT, double>) {
            return detail::normalPdfDouble(x, mu, sigma);
        } else {
            return detail::normalPdfT(x, mu, sigma);
        }
    }

    DoubleT logPdf(DoubleT x) const {
        using std::log;
        if constexpr (std::is_same_v<DoubleT, double>) {
            const double z = (x - mu) / sigma;
            return -0.5 * z * z - log(sigma * static_cast<double>(kSqrt2Pi));
        } else {
            const DoubleT z = (x - mu) / sigma;
            return -DoubleT(0.5) * z * z - log(sigma * DoubleT(static_cast<double>(kSqrt2Pi)));
        }
    }

    DoubleT cdf(DoubleT x) const {
        if constexpr (std::is_same_v<DoubleT, double>) {
            return detail::normalCdfDouble(x, mu, sigma);
        } else {
            return detail::normalCdfT(x, mu, sigma);
        }
    }

    DoubleT logCdf(DoubleT x) const {
        if constexpr (std::is_same_v<DoubleT, double>) {
            return detail::normalLogCdfDouble(x, mu, sigma);
        } else {
            return detail::normalLogCdfT(x, mu, sigma);
        }
    }

    DoubleT ccdf(DoubleT x) const {
        if constexpr (std::is_same_v<DoubleT, double>) {
            return detail::normalCcdfDouble(x, mu, sigma);
        } else {
            return detail::normalCcdfT(x, mu, sigma);
        }
    }

    DoubleT logCcdf(DoubleT x) const {
        if constexpr (std::is_same_v<DoubleT, double>) {
            return detail::normalLogCcdfDouble(x, mu, sigma);
        } else {
            return detail::normalLogCcdfT(x, mu, sigma);
        }
    }

    DoubleT inv(DoubleT p) const {
        using std::fma;
        return mu + sigma * detail::normalInvT(p);
    }

    /// double-only: x-direction value/d1/d2 of the pdf.
    detail::NormalD1D2 pdfD1D2(double x) const {
        static_assert(std::is_same_v<DoubleT, double>,
                      "pdfD1D2 analytic pack requires DoubleT = double");
        return detail::normalPdfD1D2(x, static_cast<double>(mu), static_cast<double>(sigma));
    }

    /// double-only: x-direction value/d1/d2 of the cdf.
    detail::NormalD1D2 cdfD1D2(double x) const {
        static_assert(std::is_same_v<DoubleT, double>,
                      "cdfD1D2 analytic pack requires DoubleT = double");
        return detail::normalCdfD1D2(x, static_cast<double>(mu), static_cast<double>(sigma));
    }

    /// double-only: full gradient/Hessian of the cdf in (x, mu, sigma).
    detail::NormalCdfGradHess cdfGradHess(double x) const {
        static_assert(std::is_same_v<DoubleT, double>,
                      "cdfGradHess analytic pack requires DoubleT = double");
        return detail::normalCdfGradHess(x, static_cast<double>(mu), static_cast<double>(sigma));
    }

    /// double-only: full gradient/Hessian of the quantile in (p, mu, sigma).
    detail::NormalInvGradHess invGradHess(double p) const {
        static_assert(std::is_same_v<DoubleT, double>,
                      "invGradHess analytic pack requires DoubleT = double");
        return detail::normalInvGradHess(p, static_cast<double>(mu), static_cast<double>(sigma));
    }
};

} // namespace quantape::math

#endif // QUANTAPE_MATH_DISTRIBUTIONS_NORMAL_H
