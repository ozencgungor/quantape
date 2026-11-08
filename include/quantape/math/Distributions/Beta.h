#ifndef QUANTAPE_MATH_DISTRIBUTIONS_BETA_H
#define QUANTAPE_MATH_DISTRIBUTIONS_BETA_H

#include "quantape/math/Distributions/detail/IncompleteBeta.h"
#include "quantape/math/Random/InverseNormal.h"

#include <cmath>
#include <limits>
#include <type_traits>

namespace quantape::math {
/**
 * @file Beta.h
 * @brief Beta distribution: pdf, cdf, log tails and quantile (std-only).
 *
 * `X ~ Beta(a,b)`, `a,b > 0`, supported on `[0,1]`:
 *
 *     pdf(x)    = x^{a-1}(1-x)^{b-1}/B(a,b),
 *     cdf(x)    = I_x(a,b)                        (regularized incomplete beta),
 *     logCdf    = log I_x(a,b),
 *     logCcdf   = log(1 - I_x(a,b)) = log I_{1-x}(b,a),
 *     ccdf(x)   = I_{1-x}(b,a),
 *
 * with `B` the beta function. The cdf/tails call the shared
 * `detail::incBeta`/`detail::incBetaLog` continued-fraction kernels (see
 * `detail/IncompleteBeta.h` for the algorithm, accuracy and derivative
 * coverage); `ccdf`/`logCcdf` use their small-side-direct complement twins,
 * so neither tail suffers `1 - p` cancellation.
 *
 * `inv(p)` returns the `p`-quantile: for `p > 1/2` it is mirrored to the
 * small tail, `inv(p;a,b) = 1 - inv(1-p;b,a)`, and the lower-tail solve is
 * safeguarded Newton in `log x` seeded from the AS241 inverse normal.
 * Relative accuracy is ~1e-12 for `p` in the interior; near the edges the
 * achieved accuracy is limited by the double representation of `x` (for
 * `p < 1e-12` or `p > 1 - 1e-12` the quantile construction degrades
 * gracefully: the returned value is the representable endpoint/nearest value
 * rather than NaN, and quantiles below the double range underflow to 0 or
 * round to 1).
 *
 * The double tier is that solve. The generic (AD) tier runs the same solve
 * off-tape on `detail::scalarValue` views (so the method instantiates for
 * every scalar type) and then performs two final Newton steps in `T` on
 * `log I_x(a,b) = log p`, in the small-tail variable so the mirror keeps
 * working. Those steps are a no-op in exact arithmetic; they are what put `p`
 * on the tape, so the adjoint is `dQ/dp = 1/pdf(Q)`.
 *
 * Edges and invalid input: `pdf = 0` (`logPdf = -inf`) outside `[0,1]`;
 * singular edges (`a<1` at `x=0`, `b<1` at `x=1`) give `+inf`; `a <= 0` or
 * `b <= 0` or NaN parameters propagate NaN; `inv(p)` is NaN for `p` outside
 * `[0,1]` and returns 0/1 at `p = 0/1`.
 *
 * Two-tier dispatch (the Normal.h pattern): `T = double` accessors call the
 * named `detail::beta*Double` kernels, which forward to the shared
 * `detail::incBeta` family explicitly instantiated for double; other scalars
 * run the generic kernel path. `inv` keeps its double-seeded solve.
 */
namespace detail {
struct BetaInvGradHess;
} // namespace detail

template <class T>
struct Beta {
    T alpha;
    T beta;

    T pdf(T x) const;
    T logPdf(T x) const;
    T cdf(T x) const;
    T logCdf(T x) const;
    T ccdf(T x) const;
    T logCcdf(T x) const;
    T inv(T p) const;
    detail::BetaInvGradHess invGradHess(T p) const;
};

namespace detail {

/**
 * @brief Lower-tail Beta quantile: solves `I_x(a,b) = p` for `0 < p <= 1/2`.
 *
 * Seed: normal approximation with the exact Beta mean/variance,
 * `x0 = mu + sigma inverseNormal(p)`. Iteration: Newton in `v = log x` on
 * `log I_x(a,b) - log p`, whose elasticity `x pdf/cdf` remains O(1) in
 * power-law tails, with a maintained `[lo,hi]` bracket; a candidate leaving
 * the bracket (or overflowing) is replaced by the bisection midpoint.
 * Termination is a relative step below 1e-13 (estimated quantile relative
 * error ~1e-12+; the `p > 1/2` mirror in `Beta::inv` keeps the solved tail
 * the small one, and the seed never needs to be accurate). The seed uses the
 * off-tape `detail::scalarValue(p)`, so the generic scalar path compiles for
 * AD types; `Beta::inv` then adds the taped Newton steps.
 */
template <class T>
T betaInvLower(T a, T b, T p) {
    using std::exp;
    using std::fabs;
    using std::log;
    using std::sqrt;
    const T logP = log(p);
    const T ab = a + b;
    const T mu = a / ab;
    const T var = a * b / (ab * ab * (ab + T(1)));
    T x = mu + sqrt(var) * T(inverseNormal(detail::scalarValue(p)));
    if (!(x > T(0)) || !(x < T(1))) {
        x = mu;
    }
    T lo = T(0);
    T hi = T(1);
    for (int it = 0; it < 200; ++it) {
        const T lc = incBetaLog(a, b, x);
        const T f = lc - logP;
        if (f > T(0)) {
            hi = x;
        } else {
            lo = x;
        }
        const T slope = exp(log(x) + betaLogPdf(a, b, x) - lc); // x pdf / cdf
        T xn = T(-1);
        if (slope > T(0)) {
            const T dv = -f / slope;
            if (dv == dv) {
                xn = x * exp(dv);
            }
        }
        if (!(xn > lo) || !(xn < hi)) {
            xn = T(0.5) * (lo + hi);
        }
        const T step = fabs(xn - x);
        x = xn;
        if (step <= T(1e-13) * x || x <= T(0)) {
            break;
        }
    }
    return x;
}

/**
 * @brief Two taped Newton steps on `log I_x(a,b) = log p`, in place.
 *
 * Called by the generic (non-double) `Beta::inv` after the off-tape double
 * solve. The step is `dv = -(log I_x - log p) / (x pdf / cdf)` in `v = log x`
 * (the same formulation as `betaInvLower`, whose elasticity stays O(1) in the
 * power-law tails); at the solution `dv = 0`, so the value is unchanged to
 * roundoff while `d x/dp = 1/pdf(x)` becomes visible on the tape. A step that
 * leaves `(0,1)` or is NaN ends the refinement, leaving the double quantile.
 */
template <class T>
inline void betaRefineLower(T a, T b, T p, T& x) {
    using std::exp;
    using std::log;
    const T logP = log(p);
    for (int it = 0; it < 2; ++it) {
        const T lc = incBetaLog(a, b, x);
        const T slope = exp(log(x) + betaLogPdf(a, b, x) - lc);
        if (!(slope > T(0))) {
            break;
        }
        const T dv = -(lc - logP) / slope;
        if (!(dv == dv)) {
            break;
        }
        const T next = x * exp(dv);
        if (!(next > T(0)) || !(next < T(1))) {
            break;
        }
        x = next;
    }
}

// ---------------------------------------------------------------------------
// Named double kernels (the `T = double` fast path of `Beta<T>`). Each entry
// forwards to the shared templated kernel explicitly instantiated for double,
// so every method has one canonical double definition and no algorithm body
// is duplicated. `inv` keeps its own double-seeded solve.
// ---------------------------------------------------------------------------

/// log pdf of Beta(a,b) at x (double).
inline double betaLogPdfDouble(double a, double b, double x) {
    return betaLogPdf<double>(a, b, x);
}

/// pdf of Beta(a,b) at x (double).
inline double betaPdfDouble(double a, double b, double x) {
    using std::exp;
    return exp(betaLogPdfDouble(a, b, x));
}

/// cdf of Beta(a,b) at x (double).
inline double betaCdfDouble(double a, double b, double x) {
    return incBeta<double>(a, b, x);
}

/// log cdf of Beta(a,b) at x (double).
inline double betaLogCdfDouble(double a, double b, double x) {
    return incBetaLog<double>(a, b, x);
}

/// complementary cdf of Beta(a,b) at x (double).
inline double betaCcdfDouble(double a, double b, double x) {
    return incBetaComplement<double>(a, b, x);
}

/// log complementary cdf of Beta(a,b) at x (double).
inline double betaLogCcdfDouble(double a, double b, double x) {
    return detail::incBetaLogCcdf<double>(a, b, x);
}

/// Beta(a,b) quantile at double: the exact `Beta<double>::inv` body (same
/// mirror to the small tail), so the pack `value` is bitwise `inv(p)`.
inline double betaInvDouble(double a, double b, double p) {
    const double nan = std::numeric_limits<double>::quiet_NaN();
    if (!(a > 0.0) || !(b > 0.0)) {
        return nan;
    }
    if (!(p >= 0.0) || !(p <= 1.0)) {
        return nan;
    }
    if (p <= 0.0) {
        return 0.0;
    }
    if (p >= 1.0) {
        return 1.0;
    }
    if (p > 0.5) {
        return 1.0 - betaInvLower<double>(b, a, 1.0 - p);
    }
    return betaInvLower<double>(a, b, p);
}

// ---------------------------------------------------------------------------
// Beta quantile parameter-derivative pack.
//
// Same implicit differentiation as in Normal.h: with `f = F_x`, `f' = F_xx`
// at `x = Q`, `Q_p = 1/f`, `Q_theta = -F_theta/f`,
// `Q_pp = -f'/f^3`, `Q_p,theta = -(F_x,theta + f' Q_theta)/f^2`,
// `Q_theta,theta = -(F_theta,theta + 2 F_theta,x Q_theta + f' Q_theta^2)/f`,
// with the alpha/beta partials from `betaCdfGradHess` at the quantile. The
// `value` is bitwise `Beta::inv(p)`; outside the open unit interval the
// derivative entries are NaN (`value` keeps the 0/1 endpoint convention, NaN
// for invalid parameters or p outside [0,1]). No finite differences.
// ---------------------------------------------------------------------------

/// (p, alpha, beta) value + gradient + Hessian of the Beta quantile.
struct BetaInvGradHess {
    double value = 0.0;
    double gP = 0.0;
    double gAlpha = 0.0;
    double gBeta = 0.0;
    double hPP = 0.0;
    double hPAlpha = 0.0;
    double hPBeta = 0.0;
    double hAA = 0.0;
    double hAB = 0.0;
    double hBB = 0.0;
};

/// `{Q, Q_p, Q_a, Q_b, Q_pp, Q_p,a, Q_p,b, Q_aa, Q_ab, Q_bb}` for the Beta
/// quantile; see the section comment.
inline BetaInvGradHess betaInvGradHess(double alpha, double beta, double p) {
    const double nan = std::numeric_limits<double>::quiet_NaN();
    BetaInvGradHess out;
    out.value = betaInvDouble(alpha, beta, p);
    if (!(p > 0.0) || !(p < 1.0)) {
        out.gP = nan;
        out.gAlpha = nan;
        out.gBeta = nan;
        out.hPP = nan;
        out.hPAlpha = nan;
        out.hPBeta = nan;
        out.hAA = nan;
        out.hAB = nan;
        out.hBB = nan;
        return out;
    }
    const BetaCdfGradHess c = betaCdfGradHess(alpha, beta, out.value);
    const double f = c.gx;
    const double fp = c.hxx;
    out.gP = 1.0 / f;
    out.gAlpha = -c.gAlpha / f;
    out.gBeta = -c.gBeta / f;
    out.hPP = -fp / (f * f * f);
    out.hPAlpha = -(c.hAx + fp * out.gAlpha) / (f * f);
    out.hPBeta = -(c.hBx + fp * out.gBeta) / (f * f);
    out.hAA = -(c.hAA + 2.0 * c.hAx * out.gAlpha + fp * out.gAlpha * out.gAlpha) / f;
    out.hAB = -(c.hAB + c.hAx * out.gBeta + c.hBx * out.gAlpha + fp * out.gAlpha * out.gBeta) / f;
    out.hBB = -(c.hBB + 2.0 * c.hBx * out.gBeta + fp * out.gBeta * out.gBeta) / f;
    return out;
}

} // namespace detail

template <class T>
T Beta<T>::pdf(T x) const {
    if constexpr (std::is_same_v<T, double>) {
        return detail::betaPdfDouble(alpha, beta, x);
    } else {
        using std::exp;
        return exp(logPdf(x));
    }
}

template <class T>
T Beta<T>::logPdf(T x) const {
    if constexpr (std::is_same_v<T, double>) {
        return detail::betaLogPdfDouble(alpha, beta, x);
    } else {
        return detail::betaLogPdf(alpha, beta, x);
    }
}

template <class T>
T Beta<T>::cdf(T x) const {
    if constexpr (std::is_same_v<T, double>) {
        return detail::betaCdfDouble(alpha, beta, x);
    } else {
        return detail::incBeta(alpha, beta, x);
    }
}

template <class T>
T Beta<T>::logCdf(T x) const {
    if constexpr (std::is_same_v<T, double>) {
        return detail::betaLogCdfDouble(alpha, beta, x);
    } else {
        return detail::incBetaLog(alpha, beta, x);
    }
}

template <class T>
T Beta<T>::ccdf(T x) const {
    if constexpr (std::is_same_v<T, double>) {
        return detail::betaCcdfDouble(alpha, beta, x);
    } else {
        return detail::incBetaComplement(alpha, beta, x);
    }
}

template <class T>
T Beta<T>::logCcdf(T x) const {
    if constexpr (std::is_same_v<T, double>) {
        return detail::betaLogCcdfDouble(alpha, beta, x);
    } else {
        return detail::incBetaLogCcdf(alpha, beta, x);
    }
}

template <class T>
T Beta<T>::inv(T p) const {
    const T nan = T(std::numeric_limits<double>::quiet_NaN());
    if (!(alpha > T(0)) || !(beta > T(0))) {
        return nan;
    }
    if (!(p >= T(0)) || !(p <= T(1))) {
        return nan;
    }
    if (p <= T(0)) {
        return T(0);
    }
    if (p >= T(1)) {
        return T(1);
    }
    if constexpr (std::is_same_v<T, double>) {
        if (p > T(0.5)) {
            return T(1) - detail::betaInvLower(beta, alpha, T(1) - p);
        }
        return detail::betaInvLower(alpha, beta, p);
    } else {
        // Generic scalar: solve off-tape in double (small tail), then two
        // taped Newton steps. The `p > 1/2` branch keeps the same mirror, so
        // the solve always runs on the small `I_x` tail.
        const double pd = detail::scalarValue(p);
        if (pd > 0.5) {
            const T q = T(1) - p;
            T y = T(detail::betaInvDouble(detail::scalarValue(beta), detail::scalarValue(alpha),
                                          detail::scalarValue(q)));
            detail::betaRefineLower(beta, alpha, q, y);
            return T(1) - y;
        }
        T x = T(detail::betaInvDouble(detail::scalarValue(alpha), detail::scalarValue(beta), pd));
        detail::betaRefineLower(alpha, beta, p, x);
        return x;
    }
}

template <class T>
detail::BetaInvGradHess Beta<T>::invGradHess(T p) const {
    static_assert(std::is_same_v<T, double>, "invGradHess analytic pack requires T = double");
    return detail::betaInvGradHess(alpha, beta, p);
}

} // namespace quantape::math

#endif // QUANTAPE_MATH_DISTRIBUTIONS_BETA_H
