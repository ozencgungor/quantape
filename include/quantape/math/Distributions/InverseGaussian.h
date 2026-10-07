#ifndef QUANTAPE_MATH_DISTRIBUTIONS_INVERSEGAUSSIAN_H
#define QUANTAPE_MATH_DISTRIBUTIONS_INVERSEGAUSSIAN_H

#include "quantape/math/Distributions/detail/NormalQuantile.h"
#include "quantape/util/Constants.h"

#include <cmath>
#include <limits>
#include <type_traits>

/**
 * @file InverseGaussian.h
 * @brief Inverse Gaussian (Wald) distribution: pdf/cdf/tails plus x-direction
 * derivative packs, std-only and scalar-generic.
 *
 * Mean mu > 0, shape lambda > 0, support x > 0 (Schrodinger 1915; Tweedie
 * 1957; see Chhikara & Folks, "The Inverse Gaussian Distribution", 1989):
 *
 *   pdf(x)  = sqrt(lambda / (2 pi x^3)) exp(-lambda (x - mu)^2 / (2 mu^2 x))
 *   logPdf  = ln lambda / 2 - ln(2 pi) / 2 - 3 ln x / 2
 *             - lambda (x - mu)^2 / (2 mu^2 x)
 *   cdf(x)  = Phi(z1) + exp(2 lambda / mu) Phi(z2)
 *   z1      = sqrt(lambda / x) (x / mu - 1)
 *   z2      = -sqrt(lambda / x) (x / mu + 1)
 *   ccdf(x) = 1 - cdf(x)
 *
 * Source/accuracy: the cdf uses the Shuster (1968) normal representation.
 * z1/z2 are evaluated as sqrt(lambda) / mu * (sqrt(x) -+ mu / sqrt(x)) --
 * algebraically identical to the boxed form and overflow-safe as x -> +inf
 * (direct x / mu overflowed and 0 * inf produced NaNs). The second term
 * exp(2 lambda / mu) Phi(z2) overflows when 2 lambda / mu exceeds ~709 and can
 * then degrade to inf * 0 = NaN in the far-right tail; a stable
 * log-Phi-asymptotic version is planned. The x <= 0 branch returns cdf = 0.
 * ccdf is 1 - cdf, so its relative accuracy decays in the far-right tail.
 *
 * Derivatives (x > 0), with u1 = -3 / (2x) - lambda / (2 mu^2) + lambda / (2
 * x^2) and u2 = 3 / (2 x^2) - lambda / x^3:
 *
 *   pdf'(x)  = pdf(x) u1
 *   pdf''(x) = pdf(x) (u1^2 + u2)
 *   cdf'(x)  = pdf(x),   cdf''(x) = pdf'(x)
 *
 * The packs are exact analytic expressions (no finite differences), verified
 * against central differences in the unit tests.
 *
 * Parameter derivative packs (double only): `inverseGaussianPdfGradHess` and
 * `inverseGaussianCdfGradHess`, taking (mu, lambda, x), return the full 3x3
 * gradient/Hessian in (mu, lambda, x). All entries are closed form (no finite
 * differences): the pdf is elementary, and the cdf differentiates the two-Phi
 * representation term by term with `A = sqrt(lambda/x)(x/mu - 1)`,
 * `B = -sqrt(lambda/x)(x/mu + 1)`, `G = exp(2 lambda/mu)` and their chain-rule
 * partials (the bookkeeping is in the function bodies). The value and the
 * x-direction rows are the existing `inverseGaussianPdfD1D2` /
 * `inverseGaussianCdfD1D2` anchors. x <= 0 returns the zero pack; x = +inf
 * returns the pdf zero pack and the cdf pack {value 1, all partials 0}.
 *
 * Overflow regime: the cdf second term `G Phi(B)` overflows when
 * `2 lambda / mu` exceeds ~709 (documented at the value level above). The
 * parameter partials inherit that failure mode unchanged -- `G` and its
 * partials are computed in double, so above the threshold entries can be
 * +-inf or inf*0 = NaN exactly where the value is. No asymptotic
 * repair is attempted here; far-tail stability is future work. NaN
 * propagates.
 *
 * Edge cases: x <= 0 gives pdf = cdf = 0, logPdf = -inf, ccdf = 1; x = +inf
 * gives pdf = 0, cdf = 1, ccdf = 0; NaN propagates. Preconditions: mu > 0,
 * lambda > 0 (non-positive values yield NaN/Inf; no guard, no throw).
 *
 * Two-tier dispatch (the Normal.h pattern): `T = double` accessors call the
 * named `detail::inverseGaussian*Double` kernels, which forward to the shared
 * templated kernels explicitly instantiated for double; other scalars run the
 * generic kernel path.
 */
namespace quantape::math {
using ::quantape::util::kInvSqrt2;
using ::quantape::util::kSqrt2Pi;
using ::quantape::util::kTwoPi;

namespace detail {

/// Value plus first and second x-derivatives. Unique to this header so the
/// distribution headers may be included together in one translation unit.
template <class T>
struct InverseGaussianD1D2 {
    T value;
    T d1;
    T d2;
};

/// Value plus the full gradient/Hessian of the IG pdf in (mu, lambda, x).
/// The lambda block uses the `L` shorthand (`gLambda`, `hLL`, `hLx`,
/// `hmuLambda`).
struct InverseGaussianPdfGradHess {
    double value = 0.0;
    double gmu = 0.0;
    double gLambda = 0.0;
    double gx = 0.0;
    double hmumu = 0.0;
    double hmuLambda = 0.0;
    double hmux = 0.0;
    double hLL = 0.0;
    double hLx = 0.0;
    double hxx = 0.0;
};

/// Value plus the full gradient/Hessian of the IG cdf in (mu, lambda, x);
/// same field layout as `InverseGaussianPdfGradHess`.
struct InverseGaussianCdfGradHess {
    double value = 0.0;
    double gmu = 0.0;
    double gLambda = 0.0;
    double gx = 0.0;
    double hmumu = 0.0;
    double hmuLambda = 0.0;
    double hmux = 0.0;
    double hLL = 0.0;
    double hLx = 0.0;
    double hxx = 0.0;
};

/// Standard normal CDF through std::erfc (local copy: Normal.h is a sibling
/// module and must not be a dependency here).
template <class T>
inline T inverseGaussianPhi(T z) {
    using std::erfc;
    return T(0.5) * erfc(-z * T(kInvSqrt2.hi));
}

/// log pdf for x > 0, written so that the (x - mu)^2 / x exponent cannot
/// overflow to inf / inf for x = +inf.
template <class T>
inline T inverseGaussianLogPdf(T x, T mu, T lambda) {
    using std::log;
    const T q = x - T(2.0) * mu + mu * mu / x;
    return T(0.5) * log(lambda) - T(0.5) * log(T(kTwoPi.hi)) - T(1.5) * log(x) -
           lambda * q / (T(2.0) * mu * mu);
}

/// pdf and its first two x-derivatives; x <= 0 returns the zero pack.
template <class T>
inline InverseGaussianD1D2<T> inverseGaussianPdfD1D2(T x, T mu, T lambda) {
    using std::exp;
    const T zero(0.0);
    if (x <= zero) {
        return {zero, zero, zero};
    }
    const T pdf = exp(inverseGaussianLogPdf(x, mu, lambda));
    const T u1 = -T(1.5) / x - lambda / (T(2.0) * mu * mu) + lambda / (T(2.0) * x * x);
    const T u2 = T(1.5) / (x * x) - lambda / (x * x * x);
    return {pdf, pdf * u1, pdf * (u1 * u1 + u2)};
}

/// cdf plus its first two x-derivatives (cdf' = pdf, cdf'' = pdf').
template <class T>
inline InverseGaussianD1D2<T> inverseGaussianCdfD1D2(T x, T mu, T lambda) {
    using std::exp;
    using std::sqrt;
    const T zero(0.0);
    if (x <= zero) {
        return {zero, zero, zero};
    }
    const T root = sqrt(x);
    const T s = sqrt(lambda) / mu;
    const T z1 = s * (root - mu / root);
    const T z2 = -s * (root + mu / root);
    const T value = inverseGaussianPhi(z1) + exp(T(2.0) * lambda / mu) * inverseGaussianPhi(z2);
    const InverseGaussianD1D2<T> p = inverseGaussianPdfD1D2(x, mu, lambda);
    return {value, p.value, p.d1};
}

/**
 * @brief IG pdf value, gradient and Hessian in (mu, lambda, x), closed form.
 *
 * With `log p = L(x, mu, lambda)` the entries are `p_i = p L_i` and
 * `p_ij = p (L_i L_j + L_ij)`. The log-density derivatives are
 * `L_mu = lambda (x - mu) / mu^3`,
 * `L_lambda = 1/(2 lambda) - (x - mu)^2 / (2 mu^2 x)`,
 * `L_x = -3/(2x) - lambda/(2 mu^2) + lambda/(2 x^2)`,
 * `L_mumu = lambda (2 mu - 3x) / mu^4`, `L_mulambda = (x - mu) / mu^3`,
 * `L_mux = lambda / mu^3`, `L_lambdalambda = -1/(2 lambda^2)`,
 * `L_lambdax = 1/(2 x^2) - 1/(2 mu^2)`, `L_xx = 3/(2 x^2) - lambda/x^3`.
 * x <= 0 and x = +inf return the zero pack; see the file header for the
 * `exp(2 lambda/mu)` overflow caveat (the pdf does not use that factor).
 */
inline InverseGaussianPdfGradHess inverseGaussianPdfGradHess(double mu, double lambda, double x) {
    InverseGaussianPdfGradHess out;
    if (x <= 0.0 || std::isinf(x)) {
        return out;
    }
    const InverseGaussianD1D2<double> d = inverseGaussianPdfD1D2<double>(x, mu, lambda);
    out.value = d.value;
    if (d.value == 0.0) {
        return out; // density underflowed: every partial underflows with it
    }
    out.gx = d.d1;
    out.hxx = d.d2;
    const double p = d.value;
    const double xm = x - mu;
    const double Lmu = lambda * xm / (mu * mu * mu);
    const double Llambda = 0.5 / lambda - xm * xm / (2.0 * mu * mu * x);
    const double Lx = -1.5 / x - lambda / (2.0 * mu * mu) + lambda / (2.0 * x * x);
    out.gmu = p * Lmu;
    out.gLambda = p * Llambda;
    out.hmumu = p * (Lmu * Lmu + lambda * (2.0 * mu - 3.0 * x) / (mu * mu * mu * mu));
    out.hmuLambda = p * (Lmu * Llambda + xm / (mu * mu * mu));
    out.hmux = p * (Lx * Lmu + lambda / (mu * mu * mu));
    out.hLL = p * (Llambda * Llambda - 0.5 / (lambda * lambda));
    out.hLx = p * (Lx * Llambda + (0.5 / (x * x) - 0.5 / (mu * mu)));
    return out;
}

/**
 * @brief IG cdf value, gradient and Hessian in (mu, lambda, x), closed form.
 *
 * `F = Phi(A) + G Phi(B)` with `A = w u`, `B = -w v`, `w = sqrt(lambda/x)`,
 * `u = x/mu - 1`, `v = x/mu + 1`, `G = exp(2 lambda/mu)`. Writing
 * `a = phi(A)`, `b = phi(B)`, `N_B = Phi(B)` and using the chain rule
 * (`F_i = a A_i + G_i N_B + G b B_i`,
 * `F_ij = a(A_ij - A A_i A_j) + G_ij N_B + b(G_i B_j + G_j B_i)
 *          + G b (B_ij - B B_i B_j)`) every mixed entry follows from the A/B/G
 * partials computed below; no finite differences. The value and both x rows
 * (`gx = pdf`, `hxx = pdf'`) are anchored on the existing
 * `inverseGaussianCdfD1D2`/`inverseGaussianPdfD1D2` kernels, which stay finite
 * in the overflow regime where the A/B/G form of `F_xx` would not.
 * Underflow corners return the anchored value with the remaining partials set
 * to zero, because every partial underflows with the value there: a density or
 * cdf that underflows to 0 returns the zero pack, and an infinite `w`, `A` or
 * `B` (saturated normal terms) returns the anchored value with zero partials.
 * In the `2 lambda/mu >~ 709` regime the mixed entries inherit the documented
 * `G` overflow (inf / NaN), while the pdf pack stays finite (it has no `G`).
 */
inline InverseGaussianCdfGradHess inverseGaussianCdfGradHess(double mu, double lambda, double x) {
    InverseGaussianCdfGradHess out;
    if (x <= 0.0) {
        return out;
    }
    if (std::isinf(x)) {
        out.value = 1.0;
        return out;
    }
    const InverseGaussianD1D2<double> c = inverseGaussianCdfD1D2<double>(x, mu, lambda);
    out.value = c.value;
    if (c.value == 0.0) {
        return out; // lower-tail underflow: the value and every partial are 0
    }
    out.gx = c.d1;
    out.hxx = c.d2;

    using std::exp;
    using std::sqrt;
    const double w = sqrt(lambda / x);
    if (std::isinf(w)) {
        return out; // deep lower-tail underflow: the finite entries vanish
    }
    const double u = x / mu - 1.0;
    const double v = x / mu + 1.0;
    const double A = w * u;
    const double B = -w * v;
    if (std::isinf(A) || std::isinf(B)) {
        return out; // saturated tails: a = b = N_B underflow, all partials 0
    }
    const double a = exp(-0.5 * A * A) / static_cast<double>(kSqrt2Pi);
    const double b = exp(-0.5 * B * B) / static_cast<double>(kSqrt2Pi);
    const double nB = inverseGaussianPhi(B);
    const double G = exp(2.0 * lambda / mu);
    if (a == 0.0 && b == 0.0 && std::isfinite(G)) {
        // Both normal terms underflowed (non-overflow G): the value is
        // saturated (0 or 1) and the mixed partials vanish in double. Guards
        // the 0 * inf of the chain-rule products below.
        return out;
    }

    const double A_l = A / (2.0 * lambda);
    const double B_l = B / (2.0 * lambda);
    const double A_m = -w * x / (mu * mu);
    const double B_m = -A_m;
    const double A_x = w * v / (2.0 * x);
    const double B_x = w * (mu - x) / (2.0 * mu * x);
    const double A_ll = -A / (4.0 * lambda * lambda);
    const double B_ll = -B / (4.0 * lambda * lambda);
    const double A_ml = A_m / (2.0 * lambda);
    const double B_ml = B_m / (2.0 * lambda);
    const double A_mm = 2.0 * w * x / (mu * mu * mu);
    const double B_mm = -A_mm;
    const double A_mx = -w / (2.0 * mu * mu);
    const double B_mx = -A_mx;
    const double A_lx = A_x / (2.0 * lambda);
    const double B_lx = B_x / (2.0 * lambda);

    const double G_l = 2.0 * G / mu;
    const double G_m = -2.0 * lambda * G / (mu * mu);
    const double G_ll = 4.0 * G / (mu * mu);
    const double G_ml = -2.0 * G * (2.0 * lambda + mu) / (mu * mu * mu);
    const double G_mm = 4.0 * lambda * G * (lambda + mu) / (mu * mu * mu * mu);

    out.gmu = a * A_m + G_m * nB + G * b * B_m;
    out.gLambda = a * A_l + G_l * nB + G * b * B_l;
    out.hmumu = a * (A_mm - A * A_m * A_m) + G_mm * nB + 2.0 * b * G_m * B_m +
                G * b * (B_mm - B * B_m * B_m);
    out.hmuLambda = a * (A_ml - A * A_m * A_l) + G_ml * nB + b * (G_m * B_l + G_l * B_m) +
                    G * b * (B_ml - B * B_m * B_l);
    out.hmux = a * (A_mx - A * A_m * A_x) + b * G_m * B_x + G * b * (B_mx - B * B_m * B_x);
    out.hLL = a * (A_ll - A * A_l * A_l) + G_ll * nB + 2.0 * b * G_l * B_l +
              G * b * (B_ll - B * B_l * B_l);
    out.hLx = a * (A_lx - A * A_l * A_x) + b * G_l * B_x + G * b * (B_lx - B * B_l * B_x);
    return out;
}

// ---------------------------------------------------------------------------
// Named double kernels (the `T = double` fast path of `InverseGaussian<T>`).
// Each entry forwards to the shared templated kernel explicitly instantiated
// for double, so every method has one canonical double definition and no
// algorithm body is duplicated.
// ---------------------------------------------------------------------------

/// pdf of IG(mu, lambda) at x (double).
inline double inverseGaussianPdfDouble(double x, double mu, double lambda) {
    return inverseGaussianPdfD1D2<double>(x, mu, lambda).value;
}

/// log pdf (double).
inline double inverseGaussianLogPdfDouble(double x, double mu, double lambda) {
    if (x <= 0.0) {
        return -std::numeric_limits<double>::infinity();
    }
    return inverseGaussianLogPdf<double>(x, mu, lambda);
}

/// cdf of IG(mu, lambda) at x (double).
inline double inverseGaussianCdfDouble(double x, double mu, double lambda) {
    return inverseGaussianCdfD1D2<double>(x, mu, lambda).value;
}

/// complementary cdf (double).
inline double inverseGaussianCcdfDouble(double x, double mu, double lambda) {
    return 1.0 - inverseGaussianCdfDouble(x, mu, lambda);
}

/**
 * @brief IG(mu, lambda) quantile (double).
 *
 * Seed: the matching-variance normal approximation
 * `x0 = mu + z_p sqrt(mu^3 / lambda)` (`z_p` from the shared AS241 kernel);
 * when that is non-positive (deep lower tail) the seed falls back to the mean.
 * Solve: safeguarded Newton on the analytic CDF with the analytic density
 * (`inverseGaussianPdfDouble`), maintaining the bracket `[0, hi]` (`hi`
 * doubled until `cdf(hi) >= p`, capped at 2000 expansions); a step that leaves
 * the bracket drops the iteration into 100 bisection sweeps. Deterministic,
 * no finite differences; termination is a relative step below 1e-13, so the
 * interior round-trip error `cdf(inv(p)) - p` is at the CDF's own accuracy
 * (measured ~1e-14..1e-13 relative on the reference grid).
 *
 * Tails/edges: `p <= 0 -> 0`, `p >= 1 -> +Inf`, NaN propagates, invalid
 * `mu`/`lambda` -> NaN. Deep lower tails are limited by the density underflow
 * (`pdf` -> 0 makes the Newton step non-finite and the bisection sweep still
 * resolves the bracket), and the documented `exp(2 lambda/mu)` overflow
 * regime of the CDF (roughly `2 lambda/mu > 709`) is inherited unchanged: in
 * that regime the CDF value is NaN/Inf and the quantile is the seed/fallback,
 * so the inverse is only reliable while `2 lambda/mu` stays below ~700.
 */
inline double inverseGaussianInvDouble(double mu, double lambda, double p) {
    using std::fabs;
    using std::sqrt;
    const double nan = std::numeric_limits<double>::quiet_NaN();
    const double inf = std::numeric_limits<double>::infinity();
    if (p != p) {
        return p;
    }
    if (!(mu > 0.0) || !(lambda > 0.0)) {
        return nan;
    }
    if (p <= 0.0) {
        return 0.0;
    }
    if (p >= 1.0) {
        return inf;
    }

    double x = mu + normalInvDouble(p) * sqrt(mu * mu * mu / lambda);
    if (!(x > 0.0) || std::isinf(x)) {
        x = mu;
    }
    double lo = 0.0;
    double hi = x;
    if (!(inverseGaussianCdfDouble(hi, mu, lambda) >= p)) {
        for (int expand = 0; expand < 2000; ++expand) {
            lo = hi;
            hi *= 2.0;
            if (std::isinf(hi) || inverseGaussianCdfDouble(hi, mu, lambda) >= p) {
                break;
            }
        }
    }
    bool converged = false;
    for (int it = 0; it < 100; ++it) {
        const double cdfX = inverseGaussianCdfDouble(x, mu, lambda);
        if (cdfX < p) {
            lo = x;
        } else {
            hi = x;
        }
        const double step = (cdfX - p) / inverseGaussianPdfDouble(x, mu, lambda);
        const double next = x - step;
        // Relative step (an absolute floor would stall the deep lower tail
        // where the density underflows and the bracket still holds the root).
        if (fabs(step) <= 1e-13 * x) {
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
        for (int it = 0; it < 100; ++it) {
            const double mid = lo + 0.5 * (hi - lo);
            if (inverseGaussianCdfDouble(mid, mu, lambda) < p) {
                lo = mid;
            } else {
                hi = mid;
            }
        }
        x = lo + 0.5 * (hi - lo);
    }
    return x;
}

// ---------------------------------------------------------------------------
// IG quantile parameter-derivative pack.
//
// Same implicit differentiation as in Normal.h: with `f = F_x`, `f' = F_xx`
// at `x = Q`, `Q_p = 1/f`, `Q_mu = -F_mu/f`, `Q_lambda = -F_lambda/f`,
// `Q_pp = -f'/f^3`, `Q_p,theta = -(F_theta,x + f' Q_theta)/f^2`,
// `Q_theta,theta = -(F_theta,theta + 2 F_theta,x Q_theta + f' Q_theta^2)/f`,
// with the (mu, lambda) partials from `inverseGaussianCdfGradHess` at the
// quantile. The `value` is bitwise `InverseGaussian::inv(p)`; outside the
// open unit interval the derivative entries are NaN (`value` keeps the
// 0/+Inf endpoint convention). No finite differences. The pack inherits the
// CDF's `exp(2 lambda/mu)` overflow regime (see `inverseGaussianInvDouble`).
// ---------------------------------------------------------------------------

/// IG quantile `{Q, Q_p, Q_mu, Q_lambda, Q_pp, Q_p,mu, Q_p,lambda, Q_mumu,
/// Q_mulambda, Q_lambdalambda}`.
struct InverseGaussianInvGradHess {
    double value = 0.0;
    double gP = 0.0;
    double gmu = 0.0;
    double gLambda = 0.0;
    double hPP = 0.0;
    double hPmu = 0.0;
    double hPLambda = 0.0;
    double hmumu = 0.0;
    double hmuLambda = 0.0;
    double hLL = 0.0;
};

/// `{Q, Q_p, Q_mu, Q_lambda, Q_pp, Q_p,mu, Q_p,lambda, Q_mumu, Q_mulambda,
/// Q_lambdalambda}` for the IG quantile; see the section comment.
inline InverseGaussianInvGradHess inverseGaussianInvGradHess(double mu, double lambda, double p) {
    const double nan = std::numeric_limits<double>::quiet_NaN();
    InverseGaussianInvGradHess out;
    out.value = inverseGaussianInvDouble(mu, lambda, p);
    if (!(p > 0.0) || !(p < 1.0)) {
        out.gP = nan;
        out.gmu = nan;
        out.gLambda = nan;
        out.hPP = nan;
        out.hPmu = nan;
        out.hPLambda = nan;
        out.hmumu = nan;
        out.hmuLambda = nan;
        out.hLL = nan;
        return out;
    }
    const InverseGaussianCdfGradHess c = inverseGaussianCdfGradHess(mu, lambda, out.value);
    const double f = c.gx;
    const double fp = c.hxx;
    out.gP = 1.0 / f;
    out.gmu = -c.gmu / f;
    out.gLambda = -c.gLambda / f;
    out.hPP = -fp / (f * f * f);
    out.hPmu = -(c.hmux + fp * out.gmu) / (f * f);
    out.hPLambda = -(c.hLx + fp * out.gLambda) / (f * f);
    out.hmumu = -(c.hmumu + 2.0 * c.hmux * out.gmu + fp * out.gmu * out.gmu) / f;
    out.hmuLambda =
        -(c.hmuLambda + c.hmux * out.gLambda + c.hLx * out.gmu + fp * out.gmu * out.gLambda) / f;
    out.hLL = -(c.hLL + 2.0 * c.hLx * out.gLambda + fp * out.gLambda * out.gLambda) / f;
    return out;
}

} // namespace detail

template <class T>
struct InverseGaussian {
    T mu;     ///< mean, mu > 0.
    T lambda; ///< shape, lambda > 0.

    T pdf(T x) const {
        if constexpr (std::is_same_v<T, double>) {
            return detail::inverseGaussianPdfDouble(x, mu, lambda);
        } else {
            return detail::inverseGaussianPdfD1D2(x, mu, lambda).value;
        }
    }

    T logPdf(T x) const {
        if constexpr (std::is_same_v<T, double>) {
            return detail::inverseGaussianLogPdfDouble(x, mu, lambda);
        } else {
            const T zero(0.0);
            if (x <= zero) {
                return -std::numeric_limits<T>::infinity();
            }
            return detail::inverseGaussianLogPdf(x, mu, lambda);
        }
    }

    T cdf(T x) const {
        if constexpr (std::is_same_v<T, double>) {
            return detail::inverseGaussianCdfDouble(x, mu, lambda);
        } else {
            return detail::inverseGaussianCdfD1D2(x, mu, lambda).value;
        }
    }

    T ccdf(T x) const {
        if constexpr (std::is_same_v<T, double>) {
            return detail::inverseGaussianCcdfDouble(x, mu, lambda);
        } else {
            return T(1.0) - cdf(x);
        }
    }

    /**
     * @brief Quantile: safeguarded Newton on the analytic CDF.
     *
     * Double-only (`detail::inverseGaussianInvDouble`); see that function for
     * the seed, bracket, termination and tail/overflow limitations.
     */
    T inv(T p) const {
        static_assert(std::is_same_v<T, double>, "InverseGaussian::inv is currently double-only");
        return detail::inverseGaussianInvDouble(mu, lambda, p);
    }

    /// double-only: full gradient/Hessian of the pdf in (mu, lambda, x).
    detail::InverseGaussianPdfGradHess pdfGradHess(double x) const {
        static_assert(std::is_same_v<T, double>, "pdfGradHess analytic pack requires T = double");
        return detail::inverseGaussianPdfGradHess(mu, lambda, x);
    }

    /// double-only: full gradient/Hessian of the cdf in (mu, lambda, x).
    detail::InverseGaussianCdfGradHess cdfGradHess(double x) const {
        static_assert(std::is_same_v<T, double>, "cdfGradHess analytic pack requires T = double");
        return detail::inverseGaussianCdfGradHess(mu, lambda, x);
    }

    /// double-only: full gradient/Hessian of the quantile in (p, mu, lambda).
    detail::InverseGaussianInvGradHess invGradHess(double p) const {
        static_assert(std::is_same_v<T, double>, "invGradHess analytic pack requires T = double");
        return detail::inverseGaussianInvGradHess(mu, lambda, p);
    }
};

} // namespace quantape::math

#endif // QUANTAPE_MATH_DISTRIBUTIONS_INVERSEGAUSSIAN_H
