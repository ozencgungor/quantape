#ifndef QUANTAPE_MATH_DISTRIBUTIONS_NONCENTRAL_BETA_H
#define QUANTAPE_MATH_DISTRIBUTIONS_NONCENTRAL_BETA_H

/**
 * @file NoncentralBeta.h
 * @brief Noncentral beta and noncentral F distributions (CDF/density + full
 * analytic derivative packs).
 *
 * Consumer: robustness/backtest overlays, optional calibration transforms;
 * noncentral F is the standard power/ANOVA law.
 *
 * Noncentral beta `X ~ NCB(alpha, beta, ncp)`, `alpha,beta > 0`, `ncp >= 0`,
 * supported on `[0,1]`, is the Poisson mixture
 *
 *     F(x)  = sum_j w_j I_x(alpha + j, beta),
 *     f(x)  = sum_j w_j f_{alpha+j,beta}(x),
 *     log w_j = -ncp/2 + j log(ncp/2) - lgamma(j+1),
 *
 * with `I`/`f` the central regularized incomplete beta and its density. The
 * double CDF/ccdf/pdf use the same mode-centred relative-weight window as the
 * noncentral chi-square mixture (`j0 = floor(ncp/2)`, terms below `e^{-45}`
 * of the mode weight dropped, absolute scale restored from `w_{j0}`), so large
 * `ncp` never underflows the weights; `logPdf` sums `log w_j + log f_j` in log
 * space around the mode, so deep-tail densities stay finite. The documented
 * limitation is the same as for the noncentral chi-square CDF: a tail whose
 * probability is dominated by terms outside the window is not resolved.
 * `ncp = 0` reduces exactly to the central beta kernels for every entry.
 *
 * Derivative packs (all double, no finite differences anywhere):
 *  - `NoncentralBetaCdfGradHess`: full `(alpha, beta, ncp, x)` gradient and
 *    Hessian. The `alpha`/`beta`/`x` entries are the term-by-term sums of the
 *    exact `betaCdfGradHess` packs (the same 3-jet continued-fraction
 *    derivatives as workstream B); the `ncp` entries follow the weight shift
 *    `dF/dncp = 1/2 [F(alpha+1) - F]`, so
 *    `d2F/dncp2 = 1/4 [F(alpha+2) - 2F(alpha+1) + F]`,
 *    `d2F/dncp dx = 1/2 [f(alpha+1) - f]`,
 *    `d2F/dncp dalpha = 1/2 [F_alpha(alpha+1) - F_alpha]`, and the beta
 *    entry is analogous. Edges: `x < 0` -> zero pack, `x > 1` -> value 1
 *    (CDF) / 0 (density) with all partials zero; at `x = 0, 1` the value and
 *    the x-entries are the one-sided closed forms and the parameter entries
 *    are their limits (zero), while the mixed parameter-x one-sided limits
 *    can be singular and are not represented (zero), exactly as
 *    `betaCdfGradHess`/`betaPdfGradHess`.
 *  - `NoncentralBetaPdfGradHess`: the same layout for the mixture density,
 *    with `ncp` entries `1/2[f(alpha+1) - f]`, `1/4[f(alpha+2) - 2f(alpha+1) +
 *    f]`, `1/2[f_x(alpha+1) - f_x]`, `1/2[f_alpha(alpha+1) - f_alpha]`.
 *
 * Noncentral F `Y ~ NCF(nu1, nu2, ncp)` is the transform
 * `X = nu1 Y / (nu2 + nu1 Y)`, `Y = (nu2/nu1) X/(1-X)`, so the CDF/density are
 * the noncentral-beta CDF/density at `x = nu1 Y/(nu2 + nu1 Y)` times the
 * Jacobian `nu1 nu2/(nu2 + nu1 Y)^2`. `NoncentralFCdfGradHess` and
 * `NoncentralFPdfGradHess` chain the `(nu1, nu2, ncp, Y)` partials onto the
 * beta packs analytically (same transform derivatives, all closed form).
 *
 * std-only (Beta.h + the sibling detail headers); no allocation; NaN/Inf
 * propagate.
 */

#include "quantape/math/Distributions/Beta.h"
#include "quantape/math/Distributions/detail/IncompleteBeta.h"

#include <cmath>
#include <limits>
#include <type_traits>

namespace quantape::math {
namespace detail {

// ---------------------------------------------------------------------------
// Poisson mixture window. The log-weight machinery is duplicated from
// NoncentralChiSquare.h on purpose: this header stays self-contained (the
// detail kernels are not factored into a new shared header), and the loop
// below is identical to `noncentralChiSquareCdfDouble`'s (mode-centred
// relative weights, `e^{-45}` cutoff, absolute scale from the mode weight).
// ---------------------------------------------------------------------------

/// Relative weight cutoff: `exp(-45) ~ 3e-20`.
inline constexpr double kNcbLogCut = -45.0;

/// Calls `term(j, relativeWeight)` over the window, then `finalize(scale)`
/// with `scale = exp(logWeight(j0))`. Precondition: `ncp > 0`.
template <class Term, class Finalize>
inline void ncbPoissonWindow(double ncp, const Term& term, const Finalize& finalize) {
    const double halfL = 0.5 * ncp;
    const double logHalfL = std::log(halfL);
    auto logWeight = [&](long j) {
        return -halfL + static_cast<double>(j) * logHalfL -
               std::lgamma(static_cast<double>(j) + 1.0);
    };
    const long j0 = static_cast<long>(std::floor(halfL));
    const double lw0 = logWeight(j0);

    for (long j = j0; j <= j0 + 200000; ++j) {
        const double lw = logWeight(j) - lw0;
        if (lw < kNcbLogCut) {
            break;
        }
        term(j, std::exp(lw));
    }
    if (j0 > 0) {
        for (long j = j0 - 1; j >= 0; --j) {
            const double lw = logWeight(j) - lw0;
            if (lw < kNcbLogCut) {
                break;
            }
            term(j, std::exp(lw));
        }
    }
    finalize(std::exp(lw0));
}

/// `exp(log w_{j0}) * sum_j exp(log w_j - log w_{j0}) term(j)`.
template <class Term>
inline double ncbWeightedSum(double ncp, const Term& term) {
    double sum = 0.0;
    ncbPoissonWindow(
        ncp, [&](long j, double w) { sum += w * term(j); }, [&](double scale) { sum *= scale; });
    return sum;
}

// ---------------------------------------------------------------------------
// Double value kernels.
// ---------------------------------------------------------------------------

/// Noncentral beta CDF, double, robust in `ncp` (mode-centred mixture).
inline double noncentralBetaCdfDouble(double alpha, double beta, double ncp, double x) {
    if (!(alpha > 0.0) || !(beta > 0.0) || !(ncp >= 0.0) || std::isnan(x)) {
        return std::numeric_limits<double>::quiet_NaN();
    }
    if (x <= 0.0) {
        return 0.0;
    }
    if (x >= 1.0) {
        return 1.0;
    }
    if (ncp == 0.0) {
        return betaCdfDouble(alpha, beta, x);
    }
    return ncbWeightedSum(
        ncp, [&](long j) { return betaCdfDouble(alpha + static_cast<double>(j), beta, x); });
}

/// Noncentral beta upper tail (central complements keep the small side direct).
inline double noncentralBetaCcdfDouble(double alpha, double beta, double ncp, double x) {
    if (!(alpha > 0.0) || !(beta > 0.0) || !(ncp >= 0.0) || std::isnan(x)) {
        return std::numeric_limits<double>::quiet_NaN();
    }
    if (x <= 0.0) {
        return 1.0;
    }
    if (x >= 1.0) {
        return 0.0;
    }
    if (ncp == 0.0) {
        return betaCcdfDouble(alpha, beta, x);
    }
    return ncbWeightedSum(
        ncp, [&](long j) { return betaCcdfDouble(alpha + static_cast<double>(j), beta, x); });
}

/// Noncentral beta density (mixture of central densities; boundary values are
/// the one-sided limits, possibly +Inf for `alpha < 1`).
inline double noncentralBetaPdfDouble(double alpha, double beta, double ncp, double x) {
    if (!(alpha > 0.0) || !(beta > 0.0) || !(ncp >= 0.0) || std::isnan(x)) {
        return std::numeric_limits<double>::quiet_NaN();
    }
    if (x < 0.0 || x > 1.0) {
        return 0.0;
    }
    if (ncp == 0.0) {
        return betaPdfDouble(alpha, beta, x);
    }
    return ncbWeightedSum(
        ncp, [&](long j) { return betaPdfDouble(alpha + static_cast<double>(j), beta, x); });
}

/**
 * @brief `log` of the noncentral beta density (`double`).
 *
 * `log f = logsumexp_j (log w_j + log f_{alpha+j,beta}(x))` with absolute log
 * weights, so deep-tail densities never underflow midway. The `j` grid runs
 * over the Poisson bulk plus 12 standard deviations (the mixture in `j` is
 * concave, so the omitted terms are far below the mode). At the singular
 * endpoints the value falls back to `log` of the one-sided density.
 */
inline double noncentralBetaLogPdfDouble(double alpha, double beta, double ncp, double x) {
    const double inf = std::numeric_limits<double>::infinity();
    const double nan = std::numeric_limits<double>::quiet_NaN();
    if (!(alpha > 0.0) || !(beta > 0.0) || !(ncp >= 0.0) || std::isnan(x)) {
        return nan;
    }
    if (x < 0.0 || x > 1.0) {
        return -inf;
    }
    if (ncp == 0.0) {
        return betaLogPdfDouble(alpha, beta, x);
    }
    if (x == 0.0 || x == 1.0) {
        return std::log(noncentralBetaPdfDouble(alpha, beta, ncp, x));
    }
    const double halfL = 0.5 * ncp;
    const double logHalfL = std::log(halfL);
    const long jmax = static_cast<long>(halfL + 12.0 * std::sqrt(halfL) + 60.0);
    double lse = -inf;
    for (long j = 0; j <= jmax; ++j) {
        const double jD = static_cast<double>(j);
        const double t =
            -halfL + jD * logHalfL - std::lgamma(jD + 1.0) + betaLogPdfDouble(alpha + jD, beta, x);
        if (t > lse) {
            lse = t + std::log1p(std::exp(lse - t));
        } else {
            lse = lse + std::log1p(std::exp(t - lse));
        }
    }
    return lse;
}

// ---------------------------------------------------------------------------
// Full parameter-derivative packs.
//
// Method: the alpha/beta/x entries are the mode-centred Poisson sums of the
// exact workstream-B packs `betaCdfGradHess`/`betaPdfGradHess` (themselves
// 3-jet derivatives of the continued fraction, no finite differences). The
// ncp entries use the weight-shift identities
//
//   d/dncp sum_j w_j g_j = 1/2 sum_j (w_{j-1} - w_j) g_j,
//
// which turn into shifted-parameter differences of the whole mixture, e.g.
// `dF/dncp = 1/2[F(alpha+1) - F]`, `d2F/dncp2 = 1/4[F(alpha+2) - 2F(alpha+1)
// + F]`, `d2F/dncp dx = 1/2[f(alpha+1) - f]` and
// `d2F/dncp dalpha = 1/2[F_alpha(alpha+1) - F_alpha]`. Every shifted mixture
// is evaluated with the same window/pack machinery (`ncbCdfSums`/
// `ncbPdfSums` at `alpha + 1`, `alpha + 2`), so no entry needs a second
// special function.
//
// Boundary convention mirrors `betaCdfGradHess`/`betaPdfGradHess`: at
// `x = 0, 1` the x-entries are the one-sided closed forms and every
// parameter and mixed entry is set to zero (the parameter limits are zero
// uniformly; the mixed one-sided limits need not exist and are not
// represented).
// ---------------------------------------------------------------------------

/// (alpha, beta, ncp, x) value + gradient + Hessian for the noncentral beta
/// CDF.
struct NoncentralBetaCdfGradHess {
    double value = 0.0;
    double gAlpha = 0.0;
    double gBeta = 0.0;
    double gNcp = 0.0;
    double gx = 0.0;
    double hAA = 0.0;
    double hAB = 0.0;
    double hANcp = 0.0;
    double hAx = 0.0;
    double hBB = 0.0;
    double hBNcp = 0.0;
    double hBx = 0.0;
    double hNcpNcp = 0.0;
    double hNcpX = 0.0;
    double hxx = 0.0;
};

/// (alpha, beta, ncp, x) value + gradient + Hessian for the noncentral beta
/// density.
struct NoncentralBetaPdfGradHess {
    double value = 0.0;
    double gAlpha = 0.0;
    double gBeta = 0.0;
    double gNcp = 0.0;
    double gx = 0.0;
    double hAA = 0.0;
    double hAB = 0.0;
    double hANcp = 0.0;
    double hAx = 0.0;
    double hBB = 0.0;
    double hBNcp = 0.0;
    double hBx = 0.0;
    double hNcpNcp = 0.0;
    double hNcpX = 0.0;
    double hxx = 0.0;
};

/// All-NaN CDF pack (invalid inputs), see the section comment.
inline NoncentralBetaCdfGradHess noncentralBetaCdfGradHessNaN() noexcept {
    const double nan = std::numeric_limits<double>::quiet_NaN();
    return {nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan};
}

/// All-NaN density pack (invalid inputs), see the section comment.
inline NoncentralBetaPdfGradHess noncentralBetaPdfGradHessNaN() noexcept {
    const double nan = std::numeric_limits<double>::quiet_NaN();
    return {nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan};
}

/// Mixed `{value, gAlpha, gBeta, gx, hAA, hAB, hAx, hBB, hBx, hxx}` sums of the
/// central beta CDF packs over the Poisson window. Preconditions:
/// `alpha,beta > 0`, `ncp >= 0`, `0 < x < 1`.
struct NcbCdfSums {
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

/// See `NcbCdfSums`.
inline NcbCdfSums ncbCdfSums(double alpha, double beta, double ncp, double x) {
    NcbCdfSums out{};
    if (ncp == 0.0) {
        const BetaCdfGradHess p = betaCdfGradHess(alpha, beta, x);
        out = {p.value, p.gAlpha, p.gBeta, p.gx, p.hAA, p.hAB, p.hAx, p.hBB, p.hBx, p.hxx};
        return out;
    }
    ncbPoissonWindow(
        ncp,
        [&](long j, double w) {
            const BetaCdfGradHess p = betaCdfGradHess(alpha + static_cast<double>(j), beta, x);
            out.value += w * p.value;
            out.gAlpha += w * p.gAlpha;
            out.gBeta += w * p.gBeta;
            out.gx += w * p.gx;
            out.hAA += w * p.hAA;
            out.hAB += w * p.hAB;
            out.hAx += w * p.hAx;
            out.hBB += w * p.hBB;
            out.hBx += w * p.hBx;
            out.hxx += w * p.hxx;
        },
        [&](double scale) {
            out.value *= scale;
            out.gAlpha *= scale;
            out.gBeta *= scale;
            out.gx *= scale;
            out.hAA *= scale;
            out.hAB *= scale;
            out.hAx *= scale;
            out.hBB *= scale;
            out.hBx *= scale;
            out.hxx *= scale;
        });
    return out;
}

/// Mixed beta-density pack sums, see `NcbCdfSums`.
struct NcbPdfSums {
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

/// See `NcbPdfSums`.
inline NcbPdfSums ncbPdfSums(double alpha, double beta, double ncp, double x) {
    NcbPdfSums out{};
    if (ncp == 0.0) {
        const BetaPdfGradHess p = betaPdfGradHess(alpha, beta, x);
        out = {p.value, p.gAlpha, p.gBeta, p.gx, p.hAA, p.hAB, p.hAx, p.hBB, p.hBx, p.hxx};
        return out;
    }
    ncbPoissonWindow(
        ncp,
        [&](long j, double w) {
            const BetaPdfGradHess p = betaPdfGradHess(alpha + static_cast<double>(j), beta, x);
            out.value += w * p.value;
            out.gAlpha += w * p.gAlpha;
            out.gBeta += w * p.gBeta;
            out.gx += w * p.gx;
            out.hAA += w * p.hAA;
            out.hAB += w * p.hAB;
            out.hAx += w * p.hAx;
            out.hBB += w * p.hBB;
            out.hBx += w * p.hBx;
            out.hxx += w * p.hxx;
        },
        [&](double scale) {
            out.value *= scale;
            out.gAlpha *= scale;
            out.gBeta *= scale;
            out.gx *= scale;
            out.hAA *= scale;
            out.hAB *= scale;
            out.hAx *= scale;
            out.hBB *= scale;
            out.hBx *= scale;
            out.hxx *= scale;
        });
    return out;
}

/// One-sided density curvature at the endpoints: the Poisson sum of the
/// central `betaPdfD1D2(...).d2` values (may be +-Inf/NaN exactly like the
/// central pack at the boundary).
inline double ncbPdfCurvatureAtBoundary(double alpha, double beta, double ncp, double x) {
    if (ncp == 0.0) {
        return betaPdfD1D2(alpha, beta, x).d2;
    }
    return ncbWeightedSum(
        ncp, [&](long j) { return betaPdfD1D2(alpha + static_cast<double>(j), beta, x).d2; });
}

/**
 * @brief `{F, F_a, F_b, F_ncp, F_x, F_aa, F_ab, F_a,ncp, F_ax, F_bb, F_b,ncp,
 * F_bx, F_ncp,ncp, F_ncp,x, F_xx}` for the noncentral beta CDF, all analytic.
 *
 * See the section comment for the method and edge behavior.
 */
inline NoncentralBetaCdfGradHess noncentralBetaCdfGradHess(double alpha, double beta, double ncp,
                                                           double x) {
    if (!(alpha > 0.0) || !(beta > 0.0) || !(ncp >= 0.0) || std::isnan(x)) {
        return noncentralBetaCdfGradHessNaN();
    }
    if (x < 0.0) {
        return NoncentralBetaCdfGradHess{};
    }
    if (x > 1.0) {
        NoncentralBetaCdfGradHess out{};
        out.value = 1.0;
        return out;
    }
    if (x == 0.0 || x == 1.0) {
        NoncentralBetaCdfGradHess out{};
        out.value = x == 0.0 ? 0.0 : 1.0;
        out.gx = noncentralBetaPdfDouble(alpha, beta, ncp, x);
        out.hxx = ncbPdfCurvatureAtBoundary(alpha, beta, ncp, x);
        return out;
    }

    const NcbCdfSums s0 = ncbCdfSums(alpha, beta, ncp, x);
    const NcbCdfSums s1 = ncbCdfSums(alpha + 1.0, beta, ncp, x);
    const NcbCdfSums s2 = ncbCdfSums(alpha + 2.0, beta, ncp, x);

    NoncentralBetaCdfGradHess out{};
    out.value = s0.value;
    out.gAlpha = s0.gAlpha;
    out.gBeta = s0.gBeta;
    out.gNcp = 0.5 * (s1.value - s0.value);
    out.gx = s0.gx;
    out.hAA = s0.hAA;
    out.hAB = s0.hAB;
    out.hANcp = 0.5 * (s1.gAlpha - s0.gAlpha);
    out.hAx = s0.hAx;
    out.hBB = s0.hBB;
    out.hBNcp = 0.5 * (s1.gBeta - s0.gBeta);
    out.hBx = s0.hBx;
    out.hNcpNcp = 0.25 * (s2.value - 2.0 * s1.value + s0.value);
    out.hNcpX = 0.5 * (s1.gx - s0.gx);
    out.hxx = s0.hxx;
    return out;
}

/**
 * @brief `{f, f_a, f_b, f_ncp, f_x, f_aa, f_ab, f_a,ncp, f_ax, f_bb, f_b,ncp,
 * f_bx, f_ncp,ncp, f_ncp,x, f_xx}` for the noncentral beta density, all
 * analytic.
 *
 * The `ncp` entries are the shifted-parameter differences of the density
 * mixture; see the section comment for the method and edge behavior.
 */
inline NoncentralBetaPdfGradHess noncentralBetaPdfGradHess(double alpha, double beta, double ncp,
                                                           double x) {
    if (!(alpha > 0.0) || !(beta > 0.0) || !(ncp >= 0.0) || std::isnan(x)) {
        return noncentralBetaPdfGradHessNaN();
    }
    if (x < 0.0 || x > 1.0) {
        return NoncentralBetaPdfGradHess{};
    }
    if (x == 0.0 || x == 1.0) {
        NoncentralBetaPdfGradHess out{};
        out.value = noncentralBetaPdfDouble(alpha, beta, ncp, x);
        out.gx = ncbPdfCurvatureAtBoundary(alpha, beta, ncp, x);
        out.hxx = std::numeric_limits<double>::quiet_NaN();
        return out;
    }

    const NcbPdfSums s0 = ncbPdfSums(alpha, beta, ncp, x);
    const NcbPdfSums s1 = ncbPdfSums(alpha + 1.0, beta, ncp, x);
    const NcbPdfSums s2 = ncbPdfSums(alpha + 2.0, beta, ncp, x);

    NoncentralBetaPdfGradHess out{};
    out.value = s0.value;
    out.gAlpha = s0.gAlpha;
    out.gBeta = s0.gBeta;
    out.gNcp = 0.5 * (s1.value - s0.value);
    out.gx = s0.gx;
    out.hAA = s0.hAA;
    out.hAB = s0.hAB;
    out.hANcp = 0.5 * (s1.gAlpha - s0.gAlpha);
    out.hAx = s0.hAx;
    out.hBB = s0.hBB;
    out.hBNcp = 0.5 * (s1.gBeta - s0.gBeta);
    out.hBx = s0.hBx;
    out.hNcpNcp = 0.25 * (s2.value - 2.0 * s1.value + s0.value);
    out.hNcpX = 0.5 * (s1.gx - s0.gx);
    out.hxx = s0.hxx;
    return out;
}

// ---------------------------------------------------------------------------
// Noncentral F.
//
// Y ~ NCF(nu1, nu2, ncp) is the transform of X ~ NCB(nu1/2, nu2/2, ncp),
//
//     x(y) = nu1 y / (nu2 + nu1 y),   dx/dy = nu1 nu2 / (nu2 + nu1 y)^2,
//
// so the F CDF/density are the beta CDF/density at x(y) and the Jacobian
// times the latter, respectively. The F packs below chain the (nu1, nu2, ncp,
// y) partials onto the noncentral-beta packs with the exact first and second
// derivatives of x and of the Jacobian (all closed form; the second entries
// are complete, not finite differences).
// ---------------------------------------------------------------------------

/// `(nu1, nu2, ncp, y)` value + gradient + Hessian for the noncentral F CDF.
struct NoncentralFCdfGradHess {
    double value = 0.0;
    double gNu1 = 0.0;
    double gNu2 = 0.0;
    double gNcp = 0.0;
    double gy = 0.0;
    double hNu1Nu1 = 0.0;
    double hNu1Nu2 = 0.0;
    double hNu1Ncp = 0.0;
    double hNu1Y = 0.0;
    double hNu2Nu2 = 0.0;
    double hNu2Ncp = 0.0;
    double hNu2Y = 0.0;
    double hNcpNcp = 0.0;
    double hNcpY = 0.0;
    double hyy = 0.0;
};

/// `(nu1, nu2, ncp, y)` value + gradient + Hessian for the noncentral F
/// density.
struct NoncentralFPdfGradHess {
    double value = 0.0;
    double gNu1 = 0.0;
    double gNu2 = 0.0;
    double gNcp = 0.0;
    double gy = 0.0;
    double hNu1Nu1 = 0.0;
    double hNu1Nu2 = 0.0;
    double hNu1Ncp = 0.0;
    double hNu1Y = 0.0;
    double hNu2Nu2 = 0.0;
    double hNu2Ncp = 0.0;
    double hNu2Y = 0.0;
    double hNcpNcp = 0.0;
    double hNcpY = 0.0;
    double hyy = 0.0;
};

/// All-NaN F CDF pack (invalid inputs).
inline NoncentralFCdfGradHess noncentralFCdfGradHessNaN() noexcept {
    const double nan = std::numeric_limits<double>::quiet_NaN();
    return {nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan};
}

/// All-NaN F density pack (invalid inputs).
inline NoncentralFPdfGradHess noncentralFPdfGradHessNaN() noexcept {
    const double nan = std::numeric_limits<double>::quiet_NaN();
    return {nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan};
}

/// Noncentral F CDF (double), through the beta transform.
inline double noncentralFCdfDouble(double nu1, double nu2, double ncp, double y) {
    if (!(nu1 > 0.0) || !(nu2 > 0.0) || !(ncp >= 0.0) || std::isnan(y)) {
        return std::numeric_limits<double>::quiet_NaN();
    }
    if (y <= 0.0) {
        return 0.0;
    }
    if (std::isinf(y)) {
        return 1.0;
    }
    const double x = nu1 * y / (nu2 + nu1 * y);
    return noncentralBetaCdfDouble(0.5 * nu1, 0.5 * nu2, ncp, x);
}

/// Noncentral F density (double): beta density at `x(y)` times `dx/dy`.
inline double noncentralFPdfDouble(double nu1, double nu2, double ncp, double y) {
    if (!(nu1 > 0.0) || !(nu2 > 0.0) || !(ncp >= 0.0) || std::isnan(y)) {
        return std::numeric_limits<double>::quiet_NaN();
    }
    if (y < 0.0 || std::isinf(y)) {
        return 0.0;
    }
    const double d = nu2 + nu1 * y;
    const double x = nu1 * y / d;
    return noncentralBetaPdfDouble(0.5 * nu1, 0.5 * nu2, ncp, x) * (nu1 * nu2) / (d * d);
}

/// Noncentral F upper tail (double), through the beta complement.
inline double noncentralFCcdfDouble(double nu1, double nu2, double ncp, double y) {
    if (!(nu1 > 0.0) || !(nu2 > 0.0) || !(ncp >= 0.0) || std::isnan(y)) {
        return std::numeric_limits<double>::quiet_NaN();
    }
    if (y <= 0.0) {
        return 1.0;
    }
    if (std::isinf(y)) {
        return 0.0;
    }
    const double x = nu1 * y / (nu2 + nu1 * y);
    return noncentralBetaCcdfDouble(0.5 * nu1, 0.5 * nu2, ncp, x);
}

/// Noncentral F log density (double).
inline double noncentralFLogPdfDouble(double nu1, double nu2, double ncp, double y) {
    const double inf = std::numeric_limits<double>::infinity();
    if (!(nu1 > 0.0) || !(nu2 > 0.0) || !(ncp >= 0.0) || std::isnan(y)) {
        return std::numeric_limits<double>::quiet_NaN();
    }
    if (y < 0.0 || std::isinf(y)) {
        return -inf;
    }
    const double d = nu2 + nu1 * y;
    const double x = nu1 * y / d;
    return noncentralBetaLogPdfDouble(0.5 * nu1, 0.5 * nu2, ncp, x) + std::log(nu1) +
           std::log(nu2) - 2.0 * std::log(d);
}

/**
 * @brief `{F, F_nu1, F_nu2, F_ncp, F_y, ...}` for the noncentral F CDF, all
 * analytic chains of the noncentral-beta pack.
 *
 * With `x = nu1 y/(nu2 + nu1 y)`, `D = nu2 + nu1 y`, `a = nu1/2`, `b = nu2/2`
 * and `Phi` the noncentral-beta CDF pack at `(a, b, ncp, x)`:
 * `F_y = Phi_x x_y`, `F_nu1 = Phi_x x_nu1 + Phi_a/2`, `F_nu2 = Phi_x x_nu2 +
 * Phi_b/2`, `F_ncp = Phi_ncp`, and the Hessian uses the closed-form second
 * derivatives of the transform (`x_yy = -2 nu1^2 nu2/D^3`, ...). Edges as in
 * `noncentralBetaCdfGradHess`: `y < 0` -> zero pack, `y = +inf` -> value 1
 * with zero partials, invalid input -> all-NaN.
 */
inline NoncentralFCdfGradHess noncentralFCdfGradHess(double nu1, double nu2, double ncp, double y) {
    if (!(nu1 > 0.0) || !(nu2 > 0.0) || !(ncp >= 0.0) || std::isnan(y)) {
        return noncentralFCdfGradHessNaN();
    }
    if (y < 0.0) {
        return NoncentralFCdfGradHess{};
    }
    if (std::isinf(y)) {
        NoncentralFCdfGradHess out{};
        out.value = 1.0;
        return out;
    }

    const double d = nu2 + nu1 * y;
    const double invD = 1.0 / d;
    const double x = nu1 * y * invD;
    const NoncentralBetaCdfGradHess p = noncentralBetaCdfGradHess(0.5 * nu1, 0.5 * nu2, ncp, x);

    const double xy = nu1 * nu2 * invD * invD;
    const double xn1 = y * nu2 * invD * invD;
    const double xn2 = -nu1 * y * invD * invD;
    const double xyy = -2.0 * nu1 * nu1 * nu2 * invD * invD * invD;
    const double xn1y = nu2 * (nu2 - nu1 * y) * invD * invD * invD;
    const double xn2y = nu1 * (nu1 * y - nu2) * invD * invD * invD;
    const double xn1n1 = -2.0 * y * y * nu2 * invD * invD * invD;
    const double xn1n2 = y * (nu1 * y - nu2) * invD * invD * invD;
    const double xn2n2 = 2.0 * nu1 * y * invD * invD * invD;

    NoncentralFCdfGradHess out{};
    out.value = p.value;
    out.gNu1 = p.gx * xn1 + 0.5 * p.gAlpha;
    out.gNu2 = p.gx * xn2 + 0.5 * p.gBeta;
    out.gNcp = p.gNcp;
    out.gy = p.gx * xy;
    out.hNu1Nu1 = p.hxx * xn1 * xn1 + p.hAx * xn1 + 0.25 * p.hAA + p.gx * xn1n1;
    out.hNu1Nu2 =
        p.hxx * xn1 * xn2 + 0.5 * p.hAx * xn2 + 0.5 * p.hBx * xn1 + 0.25 * p.hAB + p.gx * xn1n2;
    out.hNu1Ncp = p.hNcpX * xn1 + 0.5 * p.hANcp;
    out.hNu1Y = p.hxx * xy * xn1 + 0.5 * p.hAx * xy + p.gx * xn1y;
    out.hNu2Nu2 = p.hxx * xn2 * xn2 + p.hBx * xn2 + 0.25 * p.hBB + p.gx * xn2n2;
    out.hNu2Ncp = p.hNcpX * xn2 + 0.5 * p.hBNcp;
    out.hNu2Y = p.hxx * xy * xn2 + 0.5 * p.hBx * xy + p.gx * xn2y;
    out.hNcpNcp = p.hNcpNcp;
    out.hNcpY = p.hNcpX * xy;
    out.hyy = p.hxx * xy * xy + p.gx * xyy;
    return out;
}

/**
 * @brief `{f, f_nu1, f_nu2, f_ncp, f_y, ...}` for the noncentral F density,
 * all analytic.
 *
 * With the same transform and `pi = Phi(x)` the noncentral-beta *density*
 * pack, the F density is `J pi` with Jacobian `J = nu1 nu2/D^2`; every first
 * and second partial below is the closed-form product rule on `J` and `pi`
 * (see the code for the `J` derivatives). Edges as in the CDF pack.
 */
inline NoncentralFPdfGradHess noncentralFPdfGradHess(double nu1, double nu2, double ncp, double y) {
    if (!(nu1 > 0.0) || !(nu2 > 0.0) || !(ncp >= 0.0) || std::isnan(y)) {
        return noncentralFPdfGradHessNaN();
    }
    if (y < 0.0 || std::isinf(y)) {
        return NoncentralFPdfGradHess{};
    }

    const double d = nu2 + nu1 * y;
    const double invD = 1.0 / d;
    const double x = nu1 * y * invD;
    const NoncentralBetaPdfGradHess p = noncentralBetaPdfGradHess(0.5 * nu1, 0.5 * nu2, ncp, x);

    // Transform derivatives (same as the CDF pack).
    const double xy = nu1 * nu2 * invD * invD;
    const double xn1 = y * nu2 * invD * invD;
    const double xn2 = -nu1 * y * invD * invD;
    const double xn1y = nu2 * (nu2 - nu1 * y) * invD * invD * invD;
    const double xn2y = nu1 * (nu1 * y - nu2) * invD * invD * invD;
    const double xn1n1 = -2.0 * y * y * nu2 * invD * invD * invD;
    const double xn1n2 = y * (nu1 * y - nu2) * invD * invD * invD;
    const double xn2n2 = 2.0 * nu1 * y * invD * invD * invD;
    // Jacobian J = nu1 nu2/D^2 and its derivatives.
    const double jac = nu1 * nu2 * invD * invD;
    const double jy = -2.0 * nu1 * nu1 * nu2 * invD * invD * invD;
    const double jn1 = nu2 * (nu2 - nu1 * y) * invD * invD * invD;
    const double jn2 = nu1 * (nu1 * y - nu2) * invD * invD * invD;
    const double jyy = 6.0 * nu1 * nu1 * nu1 * nu2 * invD * invD * invD * invD;
    const double jn1y = 2.0 * nu1 * nu2 * (nu1 * y - 2.0 * nu2) * invD * invD * invD * invD;
    const double jn2y = 2.0 * nu1 * nu1 * (2.0 * nu2 - nu1 * y) * invD * invD * invD * invD;
    const double jn1n1 = 2.0 * y * nu2 * (nu1 * y - 2.0 * nu2) * invD * invD * invD * invD;
    const double jn1n2 =
        (-nu2 * nu2 + 4.0 * nu1 * y * nu2 - nu1 * nu1 * y * y) * invD * invD * invD * invD;
    const double jn2n2 = 2.0 * nu1 * (nu2 - 2.0 * nu1 * y) * invD * invD * invD * invD;

    // First derivatives.
    const double px = p.gx;
    const double pa = p.gAlpha;
    const double pb = p.gBeta;
    const double pn = p.gNcp;
    NoncentralFPdfGradHess out{};
    out.value = p.value * jac;
    out.gNu1 = (px * xn1 + 0.5 * pa) * jac + p.value * jn1;
    out.gNu2 = (px * xn2 + 0.5 * pb) * jac + p.value * jn2;
    out.gNcp = pn * jac;
    out.gy = px * xy * jac + p.value * jy;

    // Second derivatives: (phi_x x_d + phi/2) J products plus the transform
    // and Jacobian Hessians.
    out.hNu1Nu1 = (p.hxx * xn1 * xn1 + p.hAx * xn1 + 0.25 * p.hAA + px * xn1n1) * jac +
                  2.0 * (px * xn1 + 0.5 * pa) * jn1 + p.value * jn1n1;
    out.hNu1Nu2 =
        (p.hxx * xn1 * xn2 + 0.5 * p.hAx * xn2 + 0.5 * p.hBx * xn1 + 0.25 * p.hAB + px * xn1n2) *
            jac +
        (px * xn2 + 0.5 * pb) * jn1 + (px * xn1 + 0.5 * pa) * jn2 + p.value * jn1n2;
    out.hNu1Ncp = (p.hNcpX * xn1 + 0.5 * p.hANcp) * jac + pn * jn1;
    out.hNu1Y = (p.hxx * xy * xn1 + 0.5 * p.hAx * xy + px * xn1y) * jac +
                (px * xn1 + 0.5 * pa) * jy + px * xy * jn1 + p.value * jn1y;
    out.hNu2Nu2 = (p.hxx * xn2 * xn2 + p.hBx * xn2 + 0.25 * p.hBB + px * xn2n2) * jac +
                  2.0 * (px * xn2 + 0.5 * pb) * jn2 + p.value * jn2n2;
    out.hNu2Ncp = (p.hNcpX * xn2 + 0.5 * p.hBNcp) * jac + pn * jn2;
    out.hNu2Y = (p.hxx * xy * xn2 + 0.5 * p.hBx * xy + px * xn2y) * jac +
                (px * xn2 + 0.5 * pb) * jy + px * xy * jn2 + p.value * jn2y;
    out.hNcpNcp = p.hNcpNcp * jac;
    out.hNcpY = p.hNcpX * xy * jac + pn * jy;
    out.hyy = (p.hxx * xy * xy + px * (-2.0 * nu1 * nu1 * nu2 * invD * invD * invD)) * jac +
              2.0 * (px * xy) * jy + p.value * jyy;
    return out;
}

} // namespace detail

/**
 * @brief Noncentral beta distribution `X ~ NCB(alpha, beta, ncp)`.
 *
 * All accessors are the documented double path (`static_assert`); the
 * derivative packs are `cdfGradHess`/`pdfGradHess`. See the file comment for
 * the mixture, the edge behavior and the window limitation shared with the
 * noncentral chi-square CDF.
 */
template <class DoubleT>
struct NoncentralBeta {
    DoubleT alpha;
    DoubleT beta;
    DoubleT ncp;

    DoubleT pdf(DoubleT x) const {
        static_assert(std::is_same_v<DoubleT, double>,
                      "NoncentralBeta::pdf is currently double-only");
        return detail::noncentralBetaPdfDouble(alpha, beta, ncp, x);
    }

    DoubleT logPdf(DoubleT x) const {
        static_assert(std::is_same_v<DoubleT, double>,
                      "NoncentralBeta::logPdf is currently double-only");
        return detail::noncentralBetaLogPdfDouble(alpha, beta, ncp, x);
    }

    DoubleT cdf(DoubleT x) const {
        static_assert(std::is_same_v<DoubleT, double>,
                      "NoncentralBeta::cdf is currently double-only");
        return detail::noncentralBetaCdfDouble(alpha, beta, ncp, x);
    }

    DoubleT ccdf(DoubleT x) const {
        static_assert(std::is_same_v<DoubleT, double>,
                      "NoncentralBeta::ccdf is currently double-only");
        return detail::noncentralBetaCcdfDouble(alpha, beta, ncp, x);
    }

    detail::NoncentralBetaCdfGradHess cdfGradHess(double x) const {
        static_assert(std::is_same_v<DoubleT, double>, "pack requires DoubleT = double");
        return detail::noncentralBetaCdfGradHess(alpha, beta, ncp, x);
    }

    detail::NoncentralBetaPdfGradHess pdfGradHess(double x) const {
        static_assert(std::is_same_v<DoubleT, double>, "pack requires DoubleT = double");
        return detail::noncentralBetaPdfGradHess(alpha, beta, ncp, x);
    }
};

/**
 * @brief Noncentral F distribution `Y ~ NCF(nu1, nu2, ncp)`.
 *
 * Thin transform of `NoncentralBeta(nu1/2, nu2/2, ncp)`:
 * `Y = (nu2/nu1) X/(1-X)` with `X = nu1 Y/(nu2 + nu1 Y)`. The derivative
 * packs are `cdfGradHess`/`pdfGradHess` in `(nu1, nu2, ncp, Y)`.
 */
template <class DoubleT>
struct NoncentralF {
    DoubleT nu1;
    DoubleT nu2;
    DoubleT ncp;

    DoubleT pdf(DoubleT y) const {
        static_assert(std::is_same_v<DoubleT, double>, "NoncentralF::pdf is currently double-only");
        return detail::noncentralFPdfDouble(nu1, nu2, ncp, y);
    }

    DoubleT logPdf(DoubleT y) const {
        static_assert(std::is_same_v<DoubleT, double>,
                      "NoncentralF::logPdf is currently double-only");
        return detail::noncentralFLogPdfDouble(nu1, nu2, ncp, y);
    }

    DoubleT cdf(DoubleT y) const {
        static_assert(std::is_same_v<DoubleT, double>, "NoncentralF::cdf is currently double-only");
        return detail::noncentralFCdfDouble(nu1, nu2, ncp, y);
    }

    DoubleT ccdf(DoubleT y) const {
        static_assert(std::is_same_v<DoubleT, double>,
                      "NoncentralF::ccdf is currently double-only");
        return detail::noncentralFCcdfDouble(nu1, nu2, ncp, y);
    }

    detail::NoncentralFCdfGradHess cdfGradHess(double y) const {
        static_assert(std::is_same_v<DoubleT, double>, "pack requires DoubleT = double");
        return detail::noncentralFCdfGradHess(nu1, nu2, ncp, y);
    }

    detail::NoncentralFPdfGradHess pdfGradHess(double y) const {
        static_assert(std::is_same_v<DoubleT, double>, "pack requires DoubleT = double");
        return detail::noncentralFPdfGradHess(nu1, nu2, ncp, y);
    }
};

} // namespace quantape::math

#endif // QUANTAPE_MATH_DISTRIBUTIONS_NONCENTRAL_BETA_H
