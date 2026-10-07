#ifndef QUANTAPE_MATH_DISTRIBUTIONS_EXPONENTIAL_H
#define QUANTAPE_MATH_DISTRIBUTIONS_EXPONENTIAL_H

#include <cmath>
#include <limits>
#include <type_traits>

/**
 * @file Exponential.h
 * @brief Exponential distribution: pdf/cdf/tails plus x-direction derivative
 * packs, std-only and scalar-generic.
 *
 * Rate lambda > 0, support x >= 0:
 *
 *   pdf(x)    = lambda exp(-lambda x)
 *   cdf(x)    = 1 - exp(-lambda x) = -expm1(-lambda x)
 *   ccdf(x)   = exp(-lambda x)
 *   logPdf(x) = ln lambda - lambda x
 *   logCdf    = ln(-expm1(-lambda x)), logCcdf = -lambda x
 *
 * Derivatives (x >= 0, pdf' = -lambda pdf):
 *
 *   pdf'(x)  = -lambda pdf(x)
 *   pdf''(x) = lambda^2 pdf(x)
 *   cdf'(x)  = pdf(x),   cdf''(x) = pdf'(x)
 *
 * Accuracy: cdf uses std::expm1 and logCdf uses std::log of the expm1 form, so
 * the lower tail is accurate for small lambda x (no 1 - exp cancellation).
 * The packs are exact analytic expressions, verified against central
 * differences in the unit tests.
 *
 * Parameter derivative packs (double only): `exponentialPdfGradHess` and
 * `exponentialCdfGradHess`, taking (lambda, x), return the full 2x2
 * gradient/Hessian in (lambda, x), all closed form (no finite differences):
 * `pdf_lambda = c (1 - lambda x)`, `pdf_lambdalambda = c (lambda x^2 - 2x)`,
 * `pdf_lambdax = lambda c (lambda x - 2)` with `c = exp(-lambda x)`; the cdf
 * entries are `cdf_lambda = x c`, `cdf_lambdalambda = -x^2 c`,
 * `cdf_lambdax = c (1 - lambda x)`. The value and x-direction rows are the
 * existing `exponentialPdfD1D2`/`exponentialCdfD1D2` anchors. x < 0 returns
 * the zero pack; x = +inf returns the pdf zero pack and the cdf pack
 * {value 1, all partials 0}; x = 0 keeps the right-hand branch (pdf' =
 * -lambda^2); NaN propagates.
 *
 * inv(p) = -ln(1 - p) / lambda = -log1p(-p) / lambda; p <= 0 -> 0,
 * p >= 1 -> +inf (documented endpoint convention), NaN propagates.
 *
 * Edge cases: x < 0 gives pdf = cdf = 0, logPdf = logCdf = -inf, ccdf = 1,
 * logCcdf = 0; x = 0 is in the support (closed at the origin per the density
 * convention) with pdf = lambda, cdf = 0, ccdf = 1, logCdf = -inf. The
 * derivative packs report the right-hand branch at x = 0 (pdf' = -lambda^2).
 * x = +inf gives pdf = 0, cdf = 1, ccdf = 0, logCcdf = -inf. Precondition:
 * lambda > 0 (lambda = 0 yields NaNs; no guard, no throw).
 *
 * Two-tier dispatch (the Normal.h pattern): `T = double` accessors call the
 * named `detail::exponential*Double` kernels, which forward to the shared
 * templated kernels explicitly instantiated for double; other scalars run the
 * generic kernel path. `inv` keeps its direct closed-form solve.
 */
namespace quantape::math {
namespace detail {

/// Value plus first and second x-derivatives. Unique to this header so the
/// distribution headers may be included together in one translation unit.
template <class T>
struct ExponentialD1D2 {
    T value;
    T d1;
    T d2;
};

/// Value plus the full gradient/Hessian of the exponential pdf in (lambda, x).
/// The lambda block uses the `L` shorthand (`gLambda`, `hLL`, `hLx`).
struct ExponentialPdfGradHess {
    double value = 0.0;
    double gLambda = 0.0;
    double gx = 0.0;
    double hLL = 0.0;
    double hLx = 0.0;
    double hxx = 0.0;
};

/// Value plus the full gradient/Hessian of the exponential cdf in
/// (lambda, x); same field layout as `ExponentialPdfGradHess`.
struct ExponentialCdfGradHess {
    double value = 0.0;
    double gLambda = 0.0;
    double gx = 0.0;
    double hLL = 0.0;
    double hLx = 0.0;
    double hxx = 0.0;
};

/// pdf and its first two x-derivatives; x < 0 returns the zero pack.
template <class T>
inline ExponentialD1D2<T> exponentialPdfD1D2(T x, T lambda) {
    using std::exp;
    const T zero(0.0);
    if (x < zero) {
        return {zero, zero, zero};
    }
    const T pdf = lambda * exp(-lambda * x);
    return {pdf, -lambda * pdf, lambda * lambda * pdf};
}

/// cdf plus its first two x-derivatives (cdf' = pdf, cdf'' = pdf').
template <class T>
inline ExponentialD1D2<T> exponentialCdfD1D2(T x, T lambda) {
    using std::expm1;
    const T zero(0.0);
    if (x < zero) {
        return {zero, zero, zero};
    }
    const ExponentialD1D2<T> p = exponentialPdfD1D2(x, lambda);
    return {-expm1(-lambda * x), p.value, p.d1};
}

/**
 * @brief pdf value, gradient and Hessian in (lambda, x), all closed form.
 *
 * With `p = lambda c`, `c = exp(-lambda x)`:
 * `p_lambda = c (1 - lambda x) = p (1/lambda - x)`, `p_x = -lambda p`,
 * `p_lambdalambda = c (lambda x^2 - 2x)`, `p_lambdax = lambda c (lambda x - 2)`,
 * `p_xx = lambda^2 p`. x < 0 and x = +inf return the zero pack; x = 0 keeps
 * the right-hand branch (value lambda, `p_lambda = 1`, `p_lambdax = -2 lambda`,
 * `p_xx = lambda^3`).
 */
inline ExponentialPdfGradHess exponentialPdfGradHess(double lambda, double x) {
    ExponentialPdfGradHess out;
    if (x < 0.0 || std::isinf(x)) {
        return out;
    }
    const ExponentialD1D2<double> p = exponentialPdfD1D2<double>(x, lambda);
    out.value = p.value;
    out.gx = p.d1;
    out.hxx = p.d2;
    using std::exp;
    const double c = exp(-lambda * x);
    out.gLambda = c * (1.0 - lambda * x);
    out.hLL = c * (lambda * x * x - 2.0 * x);
    out.hLx = lambda * c * (lambda * x - 2.0);
    return out;
}

/**
 * @brief cdf value, gradient and Hessian in (lambda, x), all closed form.
 *
 * With `p = lambda c`, `c = exp(-lambda x)`: `F_lambda = x c`, `F_x = p`,
 * `F_lambdalambda = -x^2 c`, `F_lambdax = c (1 - lambda x)`,
 * `F_xx = -lambda p`. x < 0 returns the zero pack; x = +inf returns
 * `{value 1, all partials 0}`; x = 0 keeps the right-hand branch
 * (value 0, `F_lambda = 0`, `F_lambdax = 1`, `F_xx = -lambda^2`).
 */
inline ExponentialCdfGradHess exponentialCdfGradHess(double lambda, double x) {
    ExponentialCdfGradHess out;
    if (x < 0.0) {
        return out;
    }
    if (std::isinf(x)) {
        out.value = 1.0;
        return out;
    }
    const ExponentialD1D2<double> c = exponentialCdfD1D2<double>(x, lambda);
    out.value = c.value;
    out.gx = c.d1;
    out.hxx = c.d2;
    using std::exp;
    const double e = exp(-lambda * x);
    out.gLambda = x * e;
    out.hLL = -x * x * e;
    out.hLx = e * (1.0 - lambda * x);
    return out;
}

// ---------------------------------------------------------------------------
// Quantile parameter-derivative pack.
//
// Same implicit differentiation as in Normal.h: with `f = F_x`, `f' = F_xx`
// at `x = Q`, `Q_p = 1/f`, `Q_lambda = -F_lambda/f`,
// `Q_pp = -f'/f^3`, `Q_p,lambda = -(F_x,lambda + f' Q_lambda)/f^2`,
// `Q_lambdalambda = -(F_lambdalambda + 2 F_lambda,x Q_lambda
//                    + f' Q_lambda^2)/f`.
// The `value` is bitwise `Exponential::inv(p)`; outside the open unit
// interval the derivative entries are NaN (`value` keeps the 0/+Inf endpoint
// convention). No finite differences.
// ---------------------------------------------------------------------------

/// (p, lambda) value + gradient + Hessian of the exponential quantile.
struct ExponentialInvGradHess {
    double value = 0.0;
    double gP = 0.0;
    double gLambda = 0.0;
    double hPP = 0.0;
    double hPL = 0.0;
    double hLL = 0.0;
};

/// `{Q, Q_p, Q_lambda, Q_pp, Q_p,lambda, Q_lambdalambda}` for the exponential
/// quantile; see the section comment.
inline ExponentialInvGradHess exponentialInvGradHess(double lambda, double p) {
    using std::log1p;
    const double nan = std::numeric_limits<double>::quiet_NaN();
    ExponentialInvGradHess out;
    if (p <= 0.0) {
        out.value = 0.0;
    } else if (p >= 1.0) {
        out.value = std::numeric_limits<double>::infinity();
    } else {
        out.value = -log1p(-p) / lambda;
    }
    if (!(p > 0.0) || !(p < 1.0)) {
        out.gP = nan;
        out.gLambda = nan;
        out.hPP = nan;
        out.hPL = nan;
        out.hLL = nan;
        return out;
    }
    const ExponentialCdfGradHess c = exponentialCdfGradHess(lambda, out.value);
    const double f = c.gx;
    const double fp = c.hxx;
    out.gP = 1.0 / f;
    out.gLambda = -c.gLambda / f;
    out.hPP = -fp / (f * f * f);
    out.hPL = -(c.hLx + fp * out.gLambda) / (f * f);
    out.hLL = -(c.hLL + 2.0 * c.hLx * out.gLambda + fp * out.gLambda * out.gLambda) / f;
    return out;
}

/// log pdf (x < 0 -> -inf); the `T` body of `Exponential::logPdf`.
template <class T>
inline T exponentialLogPdfT(T x, T lambda) {
    using std::log;
    const T zero(0.0);
    if (x < zero) {
        return -std::numeric_limits<T>::infinity();
    }
    return log(lambda) - lambda * x;
}

/// log cdf (x < 0 -> -inf); the `T` body of `Exponential::logCdf`.
template <class T>
inline T exponentialLogCdfT(T x, T lambda) {
    using std::expm1;
    using std::log;
    const T zero(0.0);
    if (x < zero) {
        return -std::numeric_limits<T>::infinity();
    }
    return log(-expm1(-lambda * x));
}

/// complementary cdf (x < 0 -> 1); the `T` body of `Exponential::ccdf`.
template <class T>
inline T exponentialCcdfT(T x, T lambda) {
    using std::exp;
    const T zero(0.0);
    if (x < zero) {
        return T(1.0);
    }
    return exp(-lambda * x);
}

/// log complementary cdf (x < 0 -> 0); the `T` body of `Exponential::logCcdf`.
template <class T>
inline T exponentialLogCcdfT(T x, T lambda) {
    const T zero(0.0);
    if (x < zero) {
        return zero;
    }
    return -lambda * x;
}

// ---------------------------------------------------------------------------
// Named double kernels (the `T = double` fast path of `Exponential<T>`). Each
// entry forwards to the shared templated kernel explicitly instantiated for
// double, so every method has one canonical double definition and no
// algorithm body is duplicated. `inv` keeps its own direct solve.
// ---------------------------------------------------------------------------

/// pdf of Exponential(lambda) at x (double).
inline double exponentialPdfDouble(double x, double lambda) {
    return exponentialPdfD1D2<double>(x, lambda).value;
}

/// log pdf (double).
inline double exponentialLogPdfDouble(double x, double lambda) {
    return exponentialLogPdfT<double>(x, lambda);
}

/// cdf of Exponential(lambda) at x (double).
inline double exponentialCdfDouble(double x, double lambda) {
    return exponentialCdfD1D2<double>(x, lambda).value;
}

/// log cdf (double).
inline double exponentialLogCdfDouble(double x, double lambda) {
    return exponentialLogCdfT<double>(x, lambda);
}

/// complementary cdf (double).
inline double exponentialCcdfDouble(double x, double lambda) {
    return exponentialCcdfT<double>(x, lambda);
}

/// log complementary cdf (double).
inline double exponentialLogCcdfDouble(double x, double lambda) {
    return exponentialLogCcdfT<double>(x, lambda);
}

} // namespace detail

template <class T>
struct Exponential {
    T lambda; ///< rate, lambda > 0.

    T pdf(T x) const {
        if constexpr (std::is_same_v<T, double>) {
            return detail::exponentialPdfDouble(x, lambda);
        } else {
            return detail::exponentialPdfD1D2(x, lambda).value;
        }
    }

    T logPdf(T x) const {
        if constexpr (std::is_same_v<T, double>) {
            return detail::exponentialLogPdfDouble(x, lambda);
        } else {
            return detail::exponentialLogPdfT(x, lambda);
        }
    }

    T cdf(T x) const {
        if constexpr (std::is_same_v<T, double>) {
            return detail::exponentialCdfDouble(x, lambda);
        } else {
            using std::expm1;
            const T zero(0.0);
            if (x < zero) {
                return zero;
            }
            return -expm1(-lambda * x);
        }
    }

    T logCdf(T x) const {
        if constexpr (std::is_same_v<T, double>) {
            return detail::exponentialLogCdfDouble(x, lambda);
        } else {
            return detail::exponentialLogCdfT(x, lambda);
        }
    }

    T ccdf(T x) const {
        if constexpr (std::is_same_v<T, double>) {
            return detail::exponentialCcdfDouble(x, lambda);
        } else {
            return detail::exponentialCcdfT(x, lambda);
        }
    }

    T logCcdf(T x) const {
        if constexpr (std::is_same_v<T, double>) {
            return detail::exponentialLogCcdfDouble(x, lambda);
        } else {
            return detail::exponentialLogCcdfT(x, lambda);
        }
    }

    /// double-only: full gradient/Hessian of the pdf in (lambda, x).
    detail::ExponentialPdfGradHess pdfGradHess(double x) const {
        static_assert(std::is_same_v<T, double>, "pdfGradHess analytic pack requires T = double");
        return detail::exponentialPdfGradHess(lambda, x);
    }

    /// double-only: full gradient/Hessian of the cdf in (lambda, x).
    detail::ExponentialCdfGradHess cdfGradHess(double x) const {
        static_assert(std::is_same_v<T, double>, "cdfGradHess analytic pack requires T = double");
        return detail::exponentialCdfGradHess(lambda, x);
    }

    /// double-only: full gradient/Hessian of the quantile in (p, lambda).
    detail::ExponentialInvGradHess invGradHess(double p) const {
        static_assert(std::is_same_v<T, double>, "invGradHess analytic pack requires T = double");
        return detail::exponentialInvGradHess(lambda, p);
    }

    /// Quantile -log1p(-p) / lambda; p <= 0 -> 0, p >= 1 -> +inf.
    T inv(T p) const {
        using std::log1p;
        const T zero(0.0);
        if (p <= zero) {
            return zero;
        }
        if (p >= T(1.0)) {
            return std::numeric_limits<T>::infinity();
        }
        return -log1p(-p) / lambda;
    }
};

} // namespace quantape::math

#endif // QUANTAPE_MATH_DISTRIBUTIONS_EXPONENTIAL_H
