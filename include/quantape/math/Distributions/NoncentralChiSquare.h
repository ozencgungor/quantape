#ifndef QUANTAPE_MATH_DISTRIBUTIONS_NONCENTRAL_CHI_SQUARE_H
#define QUANTAPE_MATH_DISTRIBUTIONS_NONCENTRAL_CHI_SQUARE_H

/**
 * @file NoncentralChiSquare.h
 * @brief Noncentral chi-square distribution (CDF/PDF + analytic derivative packs)
 *
 * Consumer: CIR exact transition bridge, CEV densities, MGF checks.
 *
 * Algorithms:
 *  - CDF: Poisson mixture `F(x;nu,lambda) = sum_j w_j P(chi^2_{nu+2j} <= x)`
 *    with log weights `log w_j = -lambda/2 + j log(lambda/2) - lgamma(j+1)`.
 *    The sum runs outwards from the mode `j0 = floor(lambda/2)` with weights
 *    normalised to the mode value (relative weights), so the absolute scale
 *    never underflows even for lambda ~ 1e6. Truncation uses the rigorous
 *    monotone-weight bound `w_j r / (1 - r)` (`r` the next weight ratio,
 *    every term `<= 1`) on the mass still left in the direction of travel,
 *    stopped below `1e-18` of the running sum -- so the summation always
 *    reaches the indices that build a deep tail, in either direction.
 *    Measured against mpmath (dps=45) and Boost (cpp_bin_float_100) on
 *    `nu <= 10, lambda <= 1000`: max rel 1.7e-13 where the result is
 *    >= 1e-12, with deep tails resolved (e.g.
 *    `F(242.674; 2, 1000) = 2.178291e-58` and `ccdf(2900) = 1.179080e-109`).
 *    Remaining caveats: the `lgamma` weight noise is
 *    ~`eps*lambda*log(lambda)` at the largest lambda (~6e-12 relative bulk
 *    error at `lambda = 1e5`); a probability beyond the double range
 *    underflows to 0; and the fixed 4e6-step summation cap bounds the deepest
 *    tails to `lambda` of order 1e11.
 *  - PDF: `f(x) = 1/2 e^{-(x+lambda)/2} (x/lambda)^{nu/4-1/2} I_{nu/2-1}(sqrt(lambda x))`
 *    using `detail::besselI` (log space, `logBesselI`); x=0 limits handled
 *    exactly. `logPdf` is the same expression assembled in log space.
 *  - Analytic derivative packs: x-direction `NccsD1D2` (value/d1/d2) and the
 *    lambda recurrences `dF/dlambda = (F(nu+2)-F(nu))/2`,
 *    `d2F/dlambda2 = (F(nu+4)-2F(nu+2)+F(nu))/4`,
 *    `d2F/dxdlambda = (f(nu+2)-f(nu))/2`.
 *    With `a = nu/2 - 1`, `z = sqrt(lambda x)`, `R = I_{a-1}/I_a` and
 *    `g = f'/f = z R / (2x) - 1/2`, the explicit second derivative is
 *    `f'' = f (g^2 + lambda/(4x) (I_{a-2}/I_a - R^2))`; the Turan bracket is
 *    formed from signed direct ratios while the Bessels are finite and via
 *    `expm1` on log ratios in the overflow regime, so the `I_{a-2} ~ I_{a-1}
 *    ~ I_a` cancellation at large `z` stays harmless. Measured vs mpmath over
 *    216 points with `nu <= 10, lambda <= 1000`: pdf 1.3e-13, d1 2.3e-12,
 *    d2 7.4e-11. `pdfD1D2` at `x = 0` (and `x < 0`) leaves d1/d2 NaN: the
 *    one-sided limits are not provided.
 *  - Full `(nu, lambda, x)` CDF pack `NccsCdfGradHess`. The nu partials are
 *    the exact `a`-partials of the same Poisson mixture,
 *    `dF/dnu = 1/2 sum_j w_j P_a(nu/2+j, x/2)`,
 *    `d2F/dnu2 = 1/4 sum_j w_j P_aa`, evaluated with the by-term
 *    `gammaCdfGradHess(a, 1, x/2)` packs; the mixed `d2F/dnudx` uses
 *    `d/da P_a(a, y) = d/da [y^{a-1}e^{-y}/Gamma(a)] = pdf(a,y)(log y -
 *    psi(a))`, so `d2F/dnudx = 1/4 sum_j w_j pdf(a, x/2)(log(x/2) - psi(a))`
 *    (verified against mp.diff). `d2F/dnudlambda` follows the weight shift
 *    `1/2[dF/dnu(nu+2) - dF/dnu(nu)]`. The nu sum reuses the mode-centred
 *    relative-weight window of the CDF (the gamma packs are exact derivatives
 *    of the evaluated series/CF, no finite differences anywhere).
 *
 *    Accuracy ceiling at large nu: the shape partials inherit the incomplete-
 *    gamma kernel's degradation (relative ~1e-15 at moderate shape, ~1e-11 at
 *    shape ~1e4, practical cap ~1e8), and the second-order nu entries add
 *    mixture cancellation (`hNuNu` sums signed terms). Measured against
 *    mpmath (mp.diff at dps=40) over the committed `nccs_nu_reference.csv`
 *    grid (`nu <= 1000`, `lambda <= 200`): max relative error cdf 1.1e-13,
 *    gNu 2.7e-13, hNuNu 5.1e-13, hNuLambda 8.8e-12, hNuX 2.3e-13; the CSV
 *    gate is 1e-8, a ~1000x envelope over the observed ceiling. A pdf-side
 *    nu pack (`NccsPdfGradHess`) is deliberately NOT provided: the density's
 *    nu partial needs `d I_a(z)/da` at `a = nu/2 - 1`, which the BesselI
 *    kernel does not expose, and the CDF pack (the CIR/CEV consumer) covers
 *    the required surface; x-direction density derivatives remain available
 *    through `pdfD1D2`.
 *  - `logCcdf` sums the log-space mixture with `gammaLogQ` (log-sum-exp)
 *    around the maximum of `log w_j + log Q_j`; unlike the double `ccdf` it
 *    is not limited by the mode-centred weight window, so deep upper tails
 *    stay accurate (verified against mpmath to 2.6e-13 absolute across the
 *    reference grid, 1.4e-12 on deep points where `ccdf` underflows).
 *    Near `Q ~ 1` the absolute log error is bounded by the `lgamma` weight
 *    noise (~`eps*lambda*log(lambda)`), not by `eps*|log Q|`.
 *  - The generic `T` CDF/CCDF path exists for AD composition (moderate
 *    lambda, documented); PDF/logPdf/logCcdf and the derivative packs are
 *    double-only.
 *
 * std-only (plus util/Constants.h and the sibling detail headers); no
 * allocation; NaN/Inf propagate.
 */

#include "quantape/math/Distributions/ChiSquare.h"
#include "quantape/math/Distributions/detail/BesselI.h"
#include "quantape/math/Distributions/detail/IncompleteGamma.h"
#include "quantape/math/Distributions/detail/NormalQuantile.h"

#include <cmath>
#include <limits>
#include <type_traits>

namespace quantape::math {
namespace detail {

/// value plus first/second x-derivatives of a double kernel.
struct NccsD1D2 {
    double value = 0.0;
    double d1 = 0.0;
    double d2 = 0.0;
};

/// (nu, lambda, x) value + gradient + Hessian for the noncentral chi-square
/// CDF. Component order follows the sibling `GammaCdfGradHess`/`BetaCdfGradHess`
/// packs: value, first partials, then the upper triangle of the Hessian.
struct NccsCdfGradHess {
    double value = 0.0;
    double gNu = 0.0;
    double gLambda = 0.0;
    double gx = 0.0;
    double hNuNu = 0.0;
    double hNuLambda = 0.0;
    double hNuX = 0.0;
    double hLambdaLambda = 0.0;
    double hLambdaX = 0.0;
    double hxx = 0.0;
};

/// scalar value helper (keeps the header Stan-free while supporting AD scalars).
template <class T>
double nccsValue(const T& x) {
    if constexpr (requires { x.val(); }) {
        return static_cast<double>(x.val());
    } else {
        return static_cast<double>(x);
    }
}

/// noncentral chi-square CDF, double, robust in lambda (mixture; see @file).
inline double noncentralChiSquareCdfDouble(double nu, double lambda, double x) {
    if (!(nu > 0.0) || !(lambda >= 0.0) || std::isnan(x)) {
        return std::numeric_limits<double>::quiet_NaN();
    }
    if (x <= 0.0) {
        return 0.0;
    }
    if (std::isinf(x)) {
        return 1.0;
    }
    if (lambda == 0.0) {
        return gammaP(0.5 * nu, 0.5 * x);
    }
    // Mode-centred summation with the central-CDF recurrence instead of one
    // gammaP per term: with a_j = nu/2 + j and y = x/2,
    //   P(a_{j+1}, y) = P(a_j, y) - d_j,  d_j = y^{a_j} e^{-y}/Gamma(a_j+1),
    //   w_{j+1} = w_j (lambda/2)/(j+1),
    // so one gammaP seeds the mode and each further term costs a few flops.
    const double halfL = 0.5 * lambda;
    const double y = 0.5 * x;
    const double a0 = 0.5 * nu;
    const long j0 = static_cast<long>(std::floor(halfL));
    const double lw0 = -halfL + static_cast<double>(j0) * std::log(halfL) -
                       std::lgamma(static_cast<double>(j0) + 1.0);
    const double w0 = std::exp(lw0);
    const double P0 = gammaP(a0 + static_cast<double>(j0), y);
    const double logD0 = (a0 + static_cast<double>(j0)) * std::log(y) - y -
                         std::lgamma(a0 + static_cast<double>(j0) + 1.0);
    const double d0 = std::exp(logD0);
    // Truncation bound: away from the mode the Poisson weights are monotone,
    // so the whole mass left in the direction of travel is at most
    // `w * r / (1 - r)` with `r` the next weight ratio (`P <= 1` there and
    // every term is positive). The loops stop only when that bound is below
    // kRelCut of the running sum, so the summation reaches every index that
    // contributes to the requested relative accuracy -- including the indices
    // far from the mode that build a deep tail.
    constexpr double kRelCut = 1e-18;
    constexpr long kMaxSteps = 4000000;

    double sum = w0 * P0;
    double w = w0;
    double P = P0;
    double d = d0;
    for (long j = j0; j - j0 < kMaxSteps; ++j) { // upward
        const double ratio = halfL / (static_cast<double>(j) + 1.0);
        w *= ratio;
        P -= d;
        d *= y / (a0 + static_cast<double>(j) + 1.0);
        sum += w * P;
        const double remainder = ratio < 1.0 ? w * ratio / (1.0 - ratio) : 0.0;
        if (!(w > 0.0) || remainder < sum * kRelCut) {
            break;
        }
    }
    if (j0 > 0) {
        w = w0;
        P = P0;
        d = d0;
        for (long j = j0; j >= 1; --j) { // downward
            d *= (a0 + static_cast<double>(j)) / y;
            P += d;
            w *= static_cast<double>(j) / halfL;
            sum += w * P;
            const double ratio = static_cast<double>(j - 1) / halfL;
            const double remainder = ratio < 1.0 ? w * ratio / (1.0 - ratio) : 0.0;
            if (!(w > 0.0) || remainder < sum * kRelCut) {
                break;
            }
        }
    }
    return sum;
}

/// noncentral chi-square upper tail.
inline double noncentralChiSquareCcdfDouble(double nu, double lambda, double x) {
    if (!(nu > 0.0) || !(lambda >= 0.0) || std::isnan(x)) {
        return std::numeric_limits<double>::quiet_NaN();
    }
    if (x <= 0.0) {
        return 1.0;
    }
    if (std::isinf(x)) {
        return 0.0;
    }
    if (lambda == 0.0) {
        return gammaQ(0.5 * nu, 0.5 * x);
    }
    // Same recurrence as the CDF with Q(a_{j+1}) = Q(a_j) + d_j:
    // one gammaQ seeds the mode, the rest are flops. The truncation bound is
    // the same (`Q <= 1`, monotone weights); it lets the summation climb to
    // the indices above the mode that dominate a deep *upper* tail.
    const double halfL = 0.5 * lambda;
    const double y = 0.5 * x;
    const double a0 = 0.5 * nu;
    const long j0 = static_cast<long>(std::floor(halfL));
    const double lw0 = -halfL + static_cast<double>(j0) * std::log(halfL) -
                       std::lgamma(static_cast<double>(j0) + 1.0);
    const double w0 = std::exp(lw0);
    const double Q0 = gammaQ(a0 + static_cast<double>(j0), y);
    const double logD0 = (a0 + static_cast<double>(j0)) * std::log(y) - y -
                         std::lgamma(a0 + static_cast<double>(j0) + 1.0);
    const double d0 = std::exp(logD0);
    constexpr double kRelCut = 1e-18;
    constexpr long kMaxSteps = 4000000;

    double sum = w0 * Q0;
    double w = w0;
    double Q = Q0;
    double d = d0;
    for (long j = j0; j - j0 < kMaxSteps; ++j) { // upward
        const double ratio = halfL / (static_cast<double>(j) + 1.0);
        w *= ratio;
        Q += d;
        d *= y / (a0 + static_cast<double>(j) + 1.0);
        sum += w * Q;
        const double remainder = ratio < 1.0 ? w * ratio / (1.0 - ratio) : 0.0;
        if (!(w > 0.0) || remainder < sum * kRelCut) {
            break;
        }
    }
    if (j0 > 0) {
        w = w0;
        Q = Q0;
        d = d0;
        for (long j = j0; j >= 1; --j) { // downward
            d *= (a0 + static_cast<double>(j)) / y;
            Q -= d;
            w *= static_cast<double>(j) / halfL;
            sum += w * Q;
            const double ratio = static_cast<double>(j - 1) / halfL;
            const double remainder = ratio < 1.0 ? w * ratio / (1.0 - ratio) : 0.0;
            if (!(w > 0.0) || remainder < sum * kRelCut) {
                break;
            }
        }
    }
    return sum;
}

/**
 * @brief `log` of the upper tail via the Poisson mixture, summed in log space.
 *
 * `log Q = logsumexp_j (log w_j + log Q(nu/2+j, x/2))` with `gammaLogQ` per
 * term (`log w_j = -lambda/2 + j log(lambda/2) - lgamma(j+1)`), so tiny tail
 * probabilities never underflow midway. In a deep upper tail the dominant
 * terms sit far *above* the Poisson mode (the weight is tiny but
 * `Q(nu/2+j, x/2) -> 1`), where the linear-space double mixture underflows;
 * `t(j) = log w_j + log Q(...)` is unimodal with its
 * maximum at or above `j0 = floor(lambda/2)`, and the sum runs over the terms
 * within `kLogCut` of that maximum (found by hill-climbing up from `j0`).
 * Accuracy is the log-sum-exp shift plus `gammaLogQ`; over the reference grid
 * (nu <= 10, lambda <= 500) the absolute log error is < 2.6e-13 versus mpmath,
 * and ~1.4e-12 on deep points where the double `ccdf` has already underflowed.
 * Near `Q ~ 1` the absolute error follows the `lgamma` weight noise
 * (~`eps lambda log lambda`), not `eps |log Q|`. Cost grows with the search
 * range ~`sqrt(lambda x)`, so this is best-effort for extreme tails.
 */
inline double noncentralChiSquareLogCcdfDouble(double nu, double lambda, double x) {
    const double inf = std::numeric_limits<double>::infinity();
    const double nan = std::numeric_limits<double>::quiet_NaN();
    if (!(nu > 0.0) || !(lambda >= 0.0) || std::isnan(x)) {
        return nan;
    }
    if (x <= 0.0) {
        return 0.0;
    }
    if (std::isinf(x)) {
        return -inf;
    }
    if (lambda == 0.0) {
        return gammaLogQ(0.5 * nu, 0.5 * x);
    }
    const double halfL = 0.5 * lambda;
    const double logHalfL = std::log(halfL);
    auto logTerm = [&](long j) {
        const double logW =
            -halfL + static_cast<double>(j) * logHalfL - std::lgamma(static_cast<double>(j) + 1.0);
        return logW + gammaLogQ(0.5 * nu + static_cast<double>(j), 0.5 * x);
    };
    const long j0 = static_cast<long>(std::floor(halfL));
    constexpr double kLogCut = -45.0;

    long jPeak = j0;
    double tPeak = logTerm(j0);
    for (long j = j0 + 1; j <= j0 + 4000000; ++j) {
        const double t = logTerm(j);
        if (!(t > tPeak)) {
            break;
        }
        tPeak = t;
        jPeak = j;
    }

    double lse = -inf;
    auto addTerm = [&](long j) {
        const double t = logTerm(j);
        if (!(t > tPeak + kLogCut)) {
            return false;
        }
        if (t > lse) {
            lse = t + std::log1p(std::exp(lse - t));
        } else {
            lse = lse + std::log1p(std::exp(t - lse));
        }
        return true;
    };
    for (long j = jPeak; j <= jPeak + 4000000; ++j) {
        if (!addTerm(j)) {
            break;
        }
    }
    for (long j = jPeak - 1; j >= 0; --j) {
        if (!addTerm(j)) {
            break;
        }
    }
    return lse;
}

/// noncentral chi-square density (double).
inline double noncentralChiSquarePdfDouble(double nu, double lambda, double x) {
    if (!(nu > 0.0) || !(lambda >= 0.0) || std::isnan(x) || x < 0.0) {
        return std::numeric_limits<double>::quiet_NaN();
    }
    if (x == 0.0) {
        if (nu > 2.0) {
            return 0.0;
        }
        if (nu == 2.0) {
            return std::exp(-0.5 * lambda);
        }
        return std::numeric_limits<double>::infinity();
    }
    if (std::isinf(x)) {
        return 0.0;
    }
    const double a = 0.5 * nu - 1.0;
    const double z = std::sqrt(lambda * x);
    if (z == 0.0) {
        // lambda == 0, or lambda*x underflowed: the noncentral correction is
        // below double resolution, so use the central density.
        return std::exp((0.5 * nu - 1.0) * std::log(x) - 0.5 * x - 0.5 * nu * std::log(2.0) -
                        std::lgamma(0.5 * nu));
    }
    const double logFactor =
        -0.5 * (x + lambda) + (0.5 * a) * std::log(x / lambda) + std::log(0.5) + logBesselI(a, z);
    return std::exp(logFactor);
}

/**
 * @brief log density, assembled in log space with `logBesselI` (double-only).
 *
 * Same expression as `noncentralChiSquarePdfDouble`, so no overflow/underflow
 * of the prefactors: `log f = -(x+lambda)/2 + (a/2) log(x/lambda) + log(1/2)
 * + log I_a(z)`. The only loss of relative precision is the cancellation
 * `-(x+lambda)/2 + z` around `x ~ lambda` (also present in the asymptotic
 * expansion), which bounds the *absolute* log error by ~`eps max(x, lambda)`:
 * ~1e-13 at `lambda = 500`, ~1e-10 at `lambda = 1e6`. Deep tails (pdf below
 * `1e-308`) remain finite in log space.
 */
inline double noncentralChiSquareLogPdfDouble(double nu, double lambda, double x) {
    const double inf = std::numeric_limits<double>::infinity();
    const double nan = std::numeric_limits<double>::quiet_NaN();
    if (!(nu > 0.0) || !(lambda >= 0.0) || std::isnan(x) || x < 0.0) {
        return nan;
    }
    if (x == 0.0) {
        if (nu > 2.0) {
            return -inf;
        }
        if (nu == 2.0) {
            return -0.5 * lambda;
        }
        return inf;
    }
    if (std::isinf(x)) {
        return -inf;
    }
    const double a = 0.5 * nu - 1.0;
    const double z = std::sqrt(lambda * x);
    if (z == 0.0) {
        return (0.5 * nu - 1.0) * std::log(x) - 0.5 * x - 0.5 * nu * std::log(2.0) -
               std::lgamma(0.5 * nu);
    }
    return -0.5 * (x + lambda) + 0.5 * a * std::log(x / lambda) + std::log(0.5) + logBesselI(a, z);
}

/// Shared `{f, f'/f}` for the Bessel form (`lambda > 0`, `x > 0`). The ratio
/// is signed and stays finite where `I_a(z)` overflows (`besselIRatio`); at
/// `z = 0` (lambda*x underflow) `z R / (2x) -> a/x` is taken directly.
inline void nccsPdfValueSlope(double nu, double lambda, double x, double& f, double& g) {
    const double a = 0.5 * nu - 1.0;
    const double z = std::sqrt(lambda * x);
    f = noncentralChiSquarePdfDouble(nu, lambda, x);
    if (z == 0.0) {
        g = a / x - 0.5;
        return;
    }
    const double ratio = besselIRatio(a - 1.0, a, z);
    g = z * ratio / (2.0 * x) - 0.5;
}

/// d/dx of the noncentral density (`I_{a-1}/I_a`; NaN for `x -> 0+`, where the
/// one-sided limit is not provided).
inline double noncentralChiSquarePdfPrime(double nu, double lambda, double x) {
    const double nan = std::numeric_limits<double>::quiet_NaN();
    if (!(nu > 0.0) || !(lambda >= 0.0) || std::isnan(x) || x < 0.0) {
        return nan;
    }
    if (x == 0.0) {
        return nan;
    }
    if (std::isinf(x)) {
        return 0.0;
    }
    if (lambda == 0.0) {
        return noncentralChiSquarePdfDouble(nu, lambda, x) * ((0.5 * nu - 1.0) / x - 0.5);
    }
    double f = 0.0;
    double g = 0.0;
    nccsPdfValueSlope(nu, lambda, x, f, g);
    return f * g;
}

/// Turan bracket `I_{a-2}/I_a - (I_{a-1}/I_a)^2` (`z > 0`), formed by signed
/// direct ratios while the values are finite, and as
/// `exp(2 log R) expm1(log(I_{a-2}/I_a) - 2 log R)` in the overflow regime
/// (the difference of near-equal `O(1)` ratios is `O(1/z)`).
inline double nccsPdfTuranBracket(double a, double z) {
    const double Ia = besselI(a, z);
    const double Ia1 = besselI(a - 1.0, z);
    const double Ia2 = besselI(a - 2.0, z);
    if (Ia != 0.0 && std::isfinite(Ia) && std::isfinite(Ia1) && std::isfinite(Ia2)) {
        const double ratio = Ia1 / Ia;
        const double bracket = Ia2 / Ia - ratio * ratio;
        if (std::isfinite(bracket)) {
            return bracket;
        }
    }
    const double logR = logBesselI(a - 1.0, z) - logBesselI(a, z);
    const double logTuran = logBesselI(a - 2.0, z) - logBesselI(a, z) - 2.0 * logR;
    return std::exp(2.0 * logR) * std::expm1(logTuran);
}

/**
 * @brief `f''(x)` for the noncentral density (double-only).
 *
 * With `a = nu/2 - 1`, `z = sqrt(lambda x)`, `R = I_{a-1}(z)/I_a(z)` and
 * `g = f'/f = z R / (2x) - 1/2`, differentiating `f'/f` once more gives
 * `f'' = f (g^2 + lambda/(4x) (I_{a-2}/I_a - R^2))` (the `R`-terms cancel;
 * see `nccsPdfTuranBracket`). For `lambda = 0` this is the central
 * Gamma(df/2, 1/2) value `f (q^2 - (nu/2-1)/x^2)`, `q = (nu/2-1)/x - 1/2`.
 * Returns NaN for `x <= 0` (one-sided limits not provided).
 */
inline double noncentralChiSquarePdfSecondDerivative(double nu, double lambda, double x) {
    const double nan = std::numeric_limits<double>::quiet_NaN();
    if (!(nu > 0.0) || !(lambda >= 0.0) || std::isnan(x) || x < 0.0) {
        return nan;
    }
    if (x == 0.0) {
        return nan;
    }
    if (std::isinf(x)) {
        return 0.0;
    }
    const double a = 0.5 * nu - 1.0;
    const double z = std::sqrt(lambda * x);
    if (z == 0.0) {
        // lambda == 0 or lambda*x underflow: the lambda/(4x) Turan term has a
        // finite central limit folded into the Gamma pack.
        return gammaPdfD1D2(0.5 * nu, 0.5, x).d2;
    }
    double f = 0.0;
    double g = 0.0;
    nccsPdfValueSlope(nu, lambda, x, f, g);
    const double turan = nccsPdfTuranBracket(a, z);
    return f * (g * g + (lambda / (4.0 * x)) * turan);
}

/// CDF value/d1/d2 in x (d2 = f').
inline NccsD1D2 noncentralChiSquareCdfD1D2(double nu, double lambda, double x) {
    return {noncentralChiSquareCdfDouble(nu, lambda, x),
            noncentralChiSquarePdfDouble(nu, lambda, x),
            noncentralChiSquarePdfPrime(nu, lambda, x)};
}

/// PDF value/d1/d2 in x, all analytic.
inline NccsD1D2 noncentralChiSquarePdfD1D2(double nu, double lambda, double x) {
    const double nan = std::numeric_limits<double>::quiet_NaN();
    if (!(nu > 0.0) || !(lambda >= 0.0) || std::isnan(x)) {
        return {nan, nan, nan};
    }
    if (x < 0.0) {
        return {nan, nan, nan};
    }
    if (x == 0.0) {
        return {noncentralChiSquarePdfDouble(nu, lambda, x), nan, nan};
    }
    if (std::isinf(x)) {
        return {0.0, 0.0, 0.0};
    }
    const double a = 0.5 * nu - 1.0;
    const double z = std::sqrt(lambda * x);
    if (z == 0.0) {
        const ValueD1D2 central = gammaPdfD1D2(0.5 * nu, 0.5, x);
        return {central.value, central.d1, central.d2};
    }
    double f = 0.0;
    double g = 0.0;
    nccsPdfValueSlope(nu, lambda, x, f, g);
    const double turan = nccsPdfTuranBracket(a, z);
    return {f, f * g, f * (g * g + (lambda / (4.0 * x)) * turan)};
}

/// dF/dlambda = (F(nu+2) - F(nu)) / 2.
inline double nccsCdfLambdaD1(double nu, double lambda, double x) {
    return 0.5 * (noncentralChiSquareCdfDouble(nu + 2.0, lambda, x) -
                  noncentralChiSquareCdfDouble(nu, lambda, x));
}

/// d2F/dlambda2 = (F(nu+4) - 2 F(nu+2) + F(nu)) / 4.
inline double nccsCdfLambdaD2(double nu, double lambda, double x) {
    return 0.25 * (noncentralChiSquareCdfDouble(nu + 4.0, lambda, x) -
                   2.0 * noncentralChiSquareCdfDouble(nu + 2.0, lambda, x) +
                   noncentralChiSquareCdfDouble(nu, lambda, x));
}

/// d2F/dxdlambda = (f(nu+2) - f(nu)) / 2.
inline double nccsCdfXLambdaD2(double nu, double lambda, double x) {
    return 0.5 * (noncentralChiSquarePdfDouble(nu + 2.0, lambda, x) -
                  noncentralChiSquarePdfDouble(nu, lambda, x));
}

/// PDF value, gradient and Hessian in (nu, lambda, x).
struct NccsPdfGradHess {
    double value = 0.0;
    double gNu = 0.0;
    double gLambda = 0.0;
    double gx = 0.0;
    double hNuNu = 0.0;
    double hNuLambda = 0.0;
    double hNuX = 0.0;
    double hLambdaLambda = 0.0;
    double hLambdaX = 0.0;
    double hxx = 0.0;
};

/// Absolute-weight shape recurrence for the rate-1 gamma mixture.
///
/// Seeds one `gammaCdfGradHess`/`gammaPdfGradHess` at the mode shape
/// `s0 = nu/2 + floor(lambda/2)` and advances the shape by +-1 with the exact
/// recurrences (all quantities carry the Poisson weight, so nothing
/// overflows):
///   d(s+1) = d(s) y/s,  P(s+1) = P(s) - d(s+1),
///   psi(s+1) = psi(s) + 1/s,  psi'(s+1) = psi'(s) - 1/s^2,
/// plus the induced recurrences for `P_s`, `P_ss`, `P_sy` and for the density
/// derivatives `g_s`, `g_ss`, `g_sy`, `g_yy`. This replaces one gamma-pack
/// call per Poisson term with a handful of flops, which is the dominant cost
/// of the noncentral nu/parameter Hessian packs.
struct NccsNuSums {
    double cdfValue = 0.0;
    double cdf1 = 0.0;  // sum w P_s           (multiply by 0.5 for dF/dnu)
    double cdf2 = 0.0;  // sum w P_ss          (x 0.25 for d2F/dnu2)
    double cdfx = 0.0;  // sum w P_sy          (x 0.25 for d2F/dnudx)
    double pdf0 = 0.0;  // sum w g
    double pdf1 = 0.0;  // sum w g_s           (x 0.5 for df/dnu)
    double pdf2 = 0.0;  // sum w g_ss          (x 0.25 for d2f/dnu2)
    double pdfxy = 0.0; // sum w g_sy          (x 0.5 for d2f/dnudx)
    double pdfyy = 0.0; // sum w g_yy
};

inline NccsNuSums nccsNuSums(double nu, double lambda, double x) {
    const double y = 0.5 * x;
    const double lny = std::log(y);
    const double a0 = 0.5 * nu;
    NccsNuSums out;
    if (lambda == 0.0) {
        const GammaCdfGradHess cp = gammaCdfGradHess(a0, 1.0, y);
        const GammaPdfGradHess dp = gammaPdfGradHess(a0, 1.0, y);
        out.cdfValue = cp.value;
        out.cdf1 = cp.gShape;
        out.cdf2 = cp.hShapeShape;
        out.cdfx = dp.gShape; // P_sy = d_s
        out.pdf0 = dp.value;
        out.pdf1 = dp.gShape;
        out.pdf2 = dp.hShapeShape;
        out.pdfxy = dp.hShapeX;
        out.pdfyy = dp.hxx;
        return out;
    }
    const double halfL = 0.5 * lambda;
    const long j0 = static_cast<long>(std::floor(halfL));
    const double lw0 = -halfL + static_cast<double>(j0) * std::log(halfL) -
                       std::lgamma(static_cast<double>(j0) + 1.0);
    const double w0 = std::exp(lw0);
    double s = a0 + static_cast<double>(j0);
    const GammaCdfGradHess cp = gammaCdfGradHess(s, 1.0, y);
    const GammaPdfGradHess dp = gammaPdfGradHess(s, 1.0, y);
    double psi = digammaDouble(s);
    double dpsi = trigammaDouble(s);
    double uP = w0 * cp.value;
    double uD = w0 * dp.value; // scaled density
    double up = w0 * cp.gShape;
    double uq = w0 * cp.hShapeShape;
    double ur = w0 * dp.gShape; // P_sy = d_s
    auto addDensityDerivs = [&](double dscaled, double shape) {
        const double lm = lny - psi;
        out.pdf0 += dscaled;
        out.pdf1 += dscaled * lm;
        out.pdf2 += dscaled * (lm * lm - dpsi);
        const double q = (shape - 1.0) / y - 1.0;
        out.pdfxy += dscaled * q * lm + dscaled / y;
        out.pdfyy += dscaled * (q * q - (shape - 1.0) / (y * y));
    };
    out.cdfValue += uP;
    out.cdf1 += up;
    out.cdf2 += uq;
    out.cdfx += ur;
    addDensityDerivs(uD, s);

    // Forward: j -> j+1, s -> s+1.
    double wrel = 1.0;
    for (long j = j0;; ++j) {
        const double rw = halfL / (static_cast<double>(j) + 1.0);
        const double dnext = uD * rw * (y / s);
        const double psiNext = psi + 1.0 / s;
        const double dpsiNext = dpsi - 1.0 / (s * s);
        const double lm = lny - psiNext;
        uP = uP * rw - dnext;
        up = up * rw - dnext * lm;
        uq = uq * rw - dnext * (lm * lm - dpsiNext);
        ur = ur * rw - dnext * ((s / y - 1.0) * lm + 1.0 / y);
        uD = dnext;
        psi = psiNext;
        dpsi = dpsiNext;
        s += 1.0;
        out.cdfValue += uP;
        out.cdf1 += up;
        out.cdf2 += uq;
        out.cdfx += ur;
        addDensityDerivs(uD, s);
        wrel *= rw;
        if (wrel < 1e-18) {
            break;
        }
    }

    // Backward: j -> j-1, s -> s-1 (rebuild the mode state).
    s = a0 + static_cast<double>(j0);
    psi = digammaDouble(s);
    dpsi = trigammaDouble(s);
    uP = w0 * cp.value;
    uD = w0 * dp.value;
    up = w0 * cp.gShape;
    uq = w0 * cp.hShapeShape;
    ur = w0 * dp.gShape;
    wrel = 1.0;
    for (long j = j0; j >= 1; --j) {
        const double rwb = static_cast<double>(j) / halfL;
        const double dnext = uD * rwb * ((s - 1.0) / y);
        const double lm = lny - psi;
        uP = rwb * (uP + uD);
        up = rwb * (up + uD * lm);
        uq = rwb * (uq + uD * (lm * lm - dpsi));
        ur = rwb * (ur + uD * (((s - 1.0) / y - 1.0) * lm + 1.0 / y));
        uD = dnext;
        psi -= 1.0 / (s - 1.0);
        dpsi += 1.0 / ((s - 1.0) * (s - 1.0));
        s -= 1.0;
        out.cdfValue += uP;
        out.cdf1 += up;
        out.cdf2 += uq;
        out.cdfx += ur;
        addDensityDerivs(uD, s);
        wrel *= rwb;
        if (wrel < 1e-18) {
            break;
        }
    }
    return out;
}

/**
 * @brief Noncentral chi-square density and its full (nu, lambda, x) Hessian.
 *
 * Poisson-mixture route (same weights/window as the CDF pack): with
 * `f = sum_j w_j(lambda) g_{nu+2j}(x)` and `g_{df}` the central chi-square
 * density,
 *   d/dnu  = sum_j w_j (1/2) dg/dshape at shape = (nu+2j)/2, rate = 1/2,
 *   d2/dnu2 = sum_j w_j (1/4) d2g/dshape2,
 *   d2/dnudx = sum_j w_j (1/2) d2g/dshape dx,
 * all from `gammaPdfGradHess((nu+2j)/2, 1/2, x)`; the lambda partials use the
 * weight-shift recurrences `f_lambda = (f_{nu+2} - f_{nu})/2`,
 * `f_lambda2 = (f_{nu+4} - 2 f_{nu+2} + f_{nu})/4`,
 * `f_lambda x = (f'_{nu+2} - f'_nu)/2`, and `f_{nu,lambda}` differentiates the
 * nu-sum in lambda through the same weight shift. The value and `d/dx` are
 * anchored on the Bessel-based double kernels (`noncentralChiSquarePdfDouble`
 * / `noncentralChiSquarePdfPrime`), so the pack agrees with the accessors.
 *
 * Domain: `nu > 0`, `lambda >= 0`, `x > 0` (all-NaN otherwise; callers fall
 * back to the generic mixture on `x = 0`/invalid edges).
 */
inline NccsPdfGradHess noncentralChiSquarePdfGradHess(double nu, double lambda, double x) {
    NccsPdfGradHess out;
    const double nan = std::numeric_limits<double>::quiet_NaN();
    if (!(nu > 0.0) || !(lambda >= 0.0) || !(x > 0.0) || std::isnan(nu) || std::isnan(lambda) ||
        std::isnan(x)) {
        out.value = out.gNu = out.gLambda = out.gx = nan;
        out.hNuNu = out.hNuLambda = out.hNuX = nan;
        out.hLambdaLambda = out.hLambdaX = out.hxx = nan;
        return out;
    }

    // nu/x entries from the shape recurrence (one seed gamma pack per call);
    // value and d/dx are anchored on the Bessel kernels, the lambda partials
    // on the weight-shift recurrences.
    const NccsNuSums sums = nccsNuSums(nu, lambda, x);
    const NccsNuSums sumsShift = nccsNuSums(nu + 2.0, lambda, x);

    const double value = noncentralChiSquarePdfDouble(nu, lambda, x);
    out.value = value;
    out.gx = noncentralChiSquarePdfPrime(nu, lambda, x);
    out.gNu = 0.25 * sums.pdf1;
    out.hNuNu = 0.125 * sums.pdf2;
    out.hNuX = 0.125 * sums.pdfxy;
    out.hxx = 0.125 * sums.pdfyy;
    out.gLambda = 0.5 * (noncentralChiSquarePdfDouble(nu + 2.0, lambda, x) - value);
    out.hLambdaLambda = 0.25 * (noncentralChiSquarePdfDouble(nu + 4.0, lambda, x) -
                                2.0 * noncentralChiSquarePdfDouble(nu + 2.0, lambda, x) + value);
    out.hLambdaX = 0.5 * (noncentralChiSquarePdfPrime(nu + 2.0, lambda, x) - out.gx);
    out.hNuLambda = 0.125 * (sumsShift.pdf1 - sums.pdf1);
    return out;
}

// ---------------------------------------------------------------------------
// Full (nu, lambda, x) derivative pack.
//
// The nu partials differentiate the Poisson mixture term-by-term. With
// `F = sum_j w_j P(a_j, y)`, `a_j = nu/2 + j`, `y = x/2` (this is exactly the
// rate-1 incomplete-gamma CDF `P(a_j, y)` with the same `y`), the chain rule
// `d/dnu = 1/2 d/da_j` gives
//
//   F_nu    = 1/2 sum_j w_j P_a(a_j, y),
//   F_nu2   = 1/4 sum_j w_j P_aa(a_j, y),
//   F_nu,x  = 1/4 sum_j w_j p(a_j, y) (log y - psi(a_j)),
//
// where `P_a`/`P_aa` come from `gammaCdfGradHess(a_j, 1, y)` and the mixed
// entry from `d/da P_a(a, y) = d/da p(a, y) = p (log y - psi(a))` (the
// rate-1 Gamma density pack's `gShape`). This is an exact derivative of the
// evaluated series/CF expression per term; the mixture uses the identical
// mode-centred relative-weight window as the double CDF (only linear in the
// per-term values). The remaining entries reuse the lambda recurrences and the
// x pack above, so `value`/`gx`/`hxx` reproduce them.
//
// The mixed `F_nu,lambda` uses the weight-shift recurrence
// `dF/dlambda = 1/2 [F(nu+2) - F(nu)]` differentiated in nu:
// `F_nu,lambda = 1/2 [F_nu(nu+2) - F_nu(nu)]`.
//
// Edge behavior mirrors `gammaCdfGradHess`: invalid parameters or NaN x ->
// all-NaN; `x < 0` -> zero pack; `x = 0` -> all-NaN (the one-sided
// parameter/mixed limits are not represented); `x = +inf` -> value 1, all
// partials 0.
// ---------------------------------------------------------------------------

/// All-NaN CDF pack (invalid/edge inputs), see the section comment.
inline NccsCdfGradHess nccsCdfGradHessNaN() noexcept {
    const double nan = std::numeric_limits<double>::quiet_NaN();
    return {nan, nan, nan, nan, nan, nan, nan, nan, nan, nan};
}

/// `{F, dF/dnu, d2F/dnu2, d2F/dnudx}` of the noncentral CDF: the per-term
/// incomplete-gamma packs summed over the same mode-centred Poisson window as
/// `noncentralChiSquareCdfDouble`. Preconditions: `nu > 0`, `lambda >= 0`,
/// `0 < x < inf`.
struct NccsNuMixture {
    double value = 0.0;
    double d1 = 0.0;
    double d2 = 0.0;
    double dx = 0.0;
};

/// See `NccsNuMixture`.
inline NccsNuMixture nccsCdfNuMixture(double nu, double lambda, double x) {
    const NccsNuSums sums = nccsNuSums(nu, lambda, x);
    NccsNuMixture out;
    out.value = sums.cdfValue;
    out.d1 = 0.5 * sums.cdf1;
    out.d2 = 0.25 * sums.cdf2;
    out.dx = 0.25 * sums.cdfx;
    return out;
}

/// `dF/dnu` of the noncentral CDF (single mixture pass; used for the
/// `d2F/dnudlambda` weight-shift recurrence).
inline double nccsCdfNuD1(double nu, double lambda, double x) {
    return nccsCdfNuMixture(nu, lambda, x).d1;
}

/**
 * @brief `{F, F_nu, F_lambda, F_x, F_nunu, F_nulam, F_nux, F_lalam, F_lamx,
 * F_xx}` for the noncentral chi-square CDF, all analytic.
 *
 * `F_nu/F_nunu/F_nux` are the exact shape derivatives of the evaluated
 * incomplete-gamma mixture; `F_lambda/F_lalam/F_lamx` are the weight-shift
 * recurrences; `F_x/F_xx` are the Bessel density pack. No finite differences.
 * See the section comment for the method and edge behavior.
 */
inline NccsCdfGradHess noncentralChiSquareCdfGradHess(double nu, double lambda, double x) {
    if (!(nu > 0.0) || !(lambda >= 0.0) || std::isnan(x)) {
        return nccsCdfGradHessNaN();
    }
    if (x < 0.0) {
        return NccsCdfGradHess{};
    }
    if (x == 0.0) {
        return nccsCdfGradHessNaN();
    }
    if (std::isinf(x)) {
        NccsCdfGradHess out{};
        out.value = 1.0;
        return out;
    }

    const NccsNuMixture mix = nccsCdfNuMixture(nu, lambda, x);
    const NccsD1D2 xPack = noncentralChiSquareCdfD1D2(nu, lambda, x);
    NccsCdfGradHess out{};
    // Anchor the pack value on the production accessor (the nu-mixture's own
    // sum differs by a few ulp from the recurrence-based CDF).
    out.value = noncentralChiSquareCdfDouble(nu, lambda, x);
    out.gNu = mix.d1;
    out.gLambda = nccsCdfLambdaD1(nu, lambda, x);
    out.gx = xPack.d1;
    out.hNuNu = mix.d2;
    out.hNuLambda = 0.5 * (nccsCdfNuD1(nu + 2.0, lambda, x) - mix.d1);
    out.hNuX = mix.dx;
    out.hLambdaLambda = nccsCdfLambdaD2(nu, lambda, x);
    out.hLambdaX = nccsCdfXLambdaD2(nu, lambda, x);
    out.hxx = xPack.d2;
    return out;
}

/**
 * @brief Noncentral chi-square quantile (double).
 *
 * Regimes: `lambda == 0` reduces exactly to `chiSquareInvDouble(nu, p)`
 * (bitwise), matching the central law. For `lambda > 0` the seed is the
 * normal approximation `nu + lambda + z_p sqrt(2(nu + 2 lambda))` (`z_p` from
 * the shared AS241 kernel), falling back to the central chi-square quantile
 * when the seed is non-positive (deep lower tail), then to `nu/2` when even
 * that underflows. Solve: safeguarded Newton on the Poisson-mixture CDF with
 * the Bessel density, maintaining `[0, hi]` (`hi` doubled until
 * `cdf(hi) >= p`, capped at 2000 expansions); a step leaving the bracket drops
 * into 100 bisection sweeps. Deterministic, tolerance ~1e-10 relative: the
 * mixture CDF itself carries a relative error growing with `lambda`
 * (~1e-11 at `lambda = 1e5`) and the density `lgamma` noise
 * ~`eps lambda log(lambda)`, so the recoverable quantile accuracy is
 * regime-limited rather than solver-limited.
 *
 * Tails/edges: `p <= 0 -> 0`, `p >= 1 -> +Inf`, NaN propagates, invalid
 * `nu`/`lambda` -> NaN. The mixture CDF resolves both tails up to the
 * documented relative accuracy, so the quantile tracks them: at
 * `nu = 2, lambda = 1000` the deep lower-tail value is
 * `inv(1e-60) = 232.4947870730993` (Boost `cpp_bin_float_100` reference
 * 232.49478707309926) and `inv(1e-12) = 605.4638154469945`.
 */
inline double noncentralChiSquareInvDouble(double nu, double lambda, double p) {
    using std::fabs;
    using std::sqrt;
    const double nan = std::numeric_limits<double>::quiet_NaN();
    const double inf = std::numeric_limits<double>::infinity();
    if (p != p) {
        return p;
    }
    if (!(nu > 0.0) || !(lambda >= 0.0)) {
        return nan;
    }
    if (p <= 0.0) {
        return 0.0;
    }
    if (p >= 1.0) {
        return inf;
    }
    if (lambda == 0.0) {
        return chiSquareInvDouble(nu, p);
    }

    double x = nu + lambda + normalInvDouble(p) * sqrt(2.0 * (nu + 2.0 * lambda));
    if (!(x > 0.0) || std::isinf(x)) {
        x = chiSquareInvDouble(nu, p);
    }
    if (!(x > 0.0)) {
        x = 0.5 * nu;
    }
    double lo = 0.0;
    double hi = x;
    if (!(noncentralChiSquareCdfDouble(nu, lambda, hi) >= p)) {
        for (int expand = 0; expand < 2000; ++expand) {
            lo = hi;
            hi *= 2.0;
            if (std::isinf(hi) || noncentralChiSquareCdfDouble(nu, lambda, hi) >= p) {
                break;
            }
        }
    }
    bool converged = false;
    for (int it = 0; it < 100; ++it) {
        const double cdfX = noncentralChiSquareCdfDouble(nu, lambda, x);
        if (cdfX < p) {
            lo = x;
        } else {
            hi = x;
        }
        const double step = (cdfX - p) / noncentralChiSquarePdfDouble(nu, lambda, x);
        const double next = x - step;
        // Relative step: an absolute 1e-13 floor is far too coarse in the
        // deep lower tail, where the density ~x^(nu/2-1) diverges for nu < 2
        // and the residual `(F - p) = step f` stays large at a tiny x.
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
            if (noncentralChiSquareCdfDouble(nu, lambda, mid) < p) {
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
// Noncentral chi-square quantile parameter-derivative pack.
//
// Same implicit differentiation as in Normal.h: with `f = F_x`, `f' = F_xx`
// at `x = Q`, `Q_p = 1/f`, `Q_nu = -F_nu/f`, `Q_lambda = -F_lambda/f`,
// `Q_pp = -f'/f^3`, `Q_p,theta = -(F_theta,x + f' Q_theta)/f^2`,
// `Q_theta,theta = -(F_theta,theta + 2 F_theta,x Q_theta + f' Q_theta^2)/f`,
// with the (nu, lambda) partials from `noncentralChiSquareCdfGradHess` at the
// quantile (which supplies `f = gx` and `f' = hxx` from the Bessel density
// pack). The `value` is bitwise `NoncentralChiSquare::inv(p)`; outside the
// open unit interval the derivative entries are NaN (`value` keeps the
// 0/+Inf endpoint convention). No finite differences. The pack inherits the
// CDF mixture's `lambda`-dependent accuracy ceiling documented in the file
// header (large `lambda`, ~1e-11 relative at `lambda = 1e5`).
// ---------------------------------------------------------------------------

/// Nccs quantile `{Q, Q_p, Q_nu, Q_lambda, Q_pp, Q_p,nu, Q_p,lambda, Q_nunu,
/// Q_nulambda, Q_lambdalambda}`.
struct NoncentralChiSquareInvGradHess {
    double value = 0.0;
    double gP = 0.0;
    double gNu = 0.0;
    double gLambda = 0.0;
    double hPP = 0.0;
    double hPNu = 0.0;
    double hPLambda = 0.0;
    double hNuNu = 0.0;
    double hNuLambda = 0.0;
    double hLambdaLambda = 0.0;
};

/// `{Q, Q_p, Q_nu, Q_lambda, Q_pp, Q_p,nu, Q_p,lambda, Q_nunu, Q_nulambda,
/// Q_lambdalambda}` for the noncentral chi-square quantile; see the section
/// comment.
inline NoncentralChiSquareInvGradHess noncentralChiSquareInvGradHess(double nu, double lambda,
                                                                     double p) {
    const double nan = std::numeric_limits<double>::quiet_NaN();
    NoncentralChiSquareInvGradHess out;
    out.value = noncentralChiSquareInvDouble(nu, lambda, p);
    if (!(p > 0.0) || !(p < 1.0)) {
        out.gP = nan;
        out.gNu = nan;
        out.gLambda = nan;
        out.hPP = nan;
        out.hPNu = nan;
        out.hPLambda = nan;
        out.hNuNu = nan;
        out.hNuLambda = nan;
        out.hLambdaLambda = nan;
        return out;
    }
    const NccsCdfGradHess c = noncentralChiSquareCdfGradHess(nu, lambda, out.value);
    const double f = c.gx;
    const double fp = c.hxx;
    out.gP = 1.0 / f;
    out.gNu = -c.gNu / f;
    out.gLambda = -c.gLambda / f;
    out.hPP = -fp / (f * f * f);
    out.hPNu = -(c.hNuX + fp * out.gNu) / (f * f);
    out.hPLambda = -(c.hLambdaX + fp * out.gLambda) / (f * f);
    out.hNuNu = -(c.hNuNu + 2.0 * c.hNuX * out.gNu + fp * out.gNu * out.gNu) / f;
    out.hNuLambda =
        -(c.hNuLambda + c.hNuX * out.gLambda + c.hLambdaX * out.gNu + fp * out.gNu * out.gLambda) /
        f;
    out.hLambdaLambda =
        -(c.hLambdaLambda + 2.0 * c.hLambdaX * out.gLambda + fp * out.gLambda * out.gLambda) / f;
    return out;
}

} // namespace detail

/**
 * @brief Noncentral chi-square distribution with `df` degrees of freedom and
 * noncentrality `ncp`.
 *
 * The double instantiation (and all derivative packs) is the production path;
 * the generic `T` CDF is provided for AD composition and is documented for
 * moderate `ncp` (<= ~600, where the un-centred weight recursion is stable).
 */
template <class DoubleT>
struct NoncentralChiSquare {
    DoubleT df;
    DoubleT ncp;

    DoubleT cdf(DoubleT x) const {
        if constexpr (std::is_same_v<DoubleT, double>) {
            return detail::noncentralChiSquareCdfDouble(df, ncp, x);
        } else {
            using std::exp;
            using std::lgamma;
            using std::log;
            const double lam = detail::nccsValue(ncp);
            const long jmax = static_cast<long>(lam * 0.5 + 12.0 * std::sqrt(lam * 0.5) + 60.0);
            const DoubleT half = DoubleT(0.5);
            DoubleT w = exp(-half * ncp);
            DoubleT sum = DoubleT(0);
            for (long j = 0; j <= jmax; ++j) {
                sum += w * detail::gammaP(half * df + DoubleT(j), half * x);
                w *= (half * ncp) / DoubleT(j + 1);
            }
            return sum;
        }
    }

    DoubleT ccdf(DoubleT x) const {
        if constexpr (std::is_same_v<DoubleT, double>) {
            return detail::noncentralChiSquareCcdfDouble(df, ncp, x);
        } else {
            using std::exp;
            const double lam = detail::nccsValue(ncp);
            const long jmax = static_cast<long>(lam * 0.5 + 12.0 * std::sqrt(lam * 0.5) + 60.0);
            const DoubleT half = DoubleT(0.5);
            DoubleT w = exp(-half * ncp);
            DoubleT sum = DoubleT(0);
            for (long j = 0; j <= jmax; ++j) {
                sum += w * detail::gammaQ(half * df + DoubleT(j), half * x);
                w *= (half * ncp) / DoubleT(j + 1);
            }
            return sum;
        }
    }

    DoubleT pdf(DoubleT x) const {
        static_assert(std::is_same_v<DoubleT, double>,
                      "NoncentralChiSquare::pdf is currently double-only");
        return detail::noncentralChiSquarePdfDouble(df, ncp, x);
    }

    DoubleT logPdf(DoubleT x) const {
        static_assert(std::is_same_v<DoubleT, double>,
                      "NoncentralChiSquare::logPdf is currently double-only");
        return detail::noncentralChiSquareLogPdfDouble(df, ncp, x);
    }

    DoubleT logCcdf(DoubleT x) const {
        static_assert(std::is_same_v<DoubleT, double>,
                      "NoncentralChiSquare::logCcdf is currently double-only");
        return detail::noncentralChiSquareLogCcdfDouble(df, ncp, x);
    }

    /**
     * @brief Quantile: safeguarded Newton on the mixture CDF, double-only.
     *
     * See `detail::noncentralChiSquareInvDouble` for the seed, bracket,
     * termination and the large-`ncp`/tail limitations.
     */
    DoubleT inv(DoubleT p) const {
        static_assert(std::is_same_v<DoubleT, double>,
                      "NoncentralChiSquare::inv is currently double-only");
        return detail::noncentralChiSquareInvDouble(df, ncp, p);
    }

    detail::NccsD1D2 cdfD1D2(double x) const {
        static_assert(std::is_same_v<DoubleT, double>, "pack requires DoubleT = double");
        return detail::noncentralChiSquareCdfD1D2(df, ncp, x);
    }

    detail::NccsD1D2 pdfD1D2(double x) const {
        static_assert(std::is_same_v<DoubleT, double>, "pack requires DoubleT = double");
        return detail::noncentralChiSquarePdfD1D2(df, ncp, x);
    }

    detail::NccsCdfGradHess cdfGradHess(double x) const {
        static_assert(std::is_same_v<DoubleT, double>, "pack requires DoubleT = double");
        return detail::noncentralChiSquareCdfGradHess(df, ncp, x);
    }

    detail::NccsPdfGradHess pdfGradHess(double x) const {
        static_assert(std::is_same_v<DoubleT, double>, "pack requires DoubleT = double");
        return detail::noncentralChiSquarePdfGradHess(df, ncp, x);
    }

    /// double-only: full gradient/Hessian of the quantile in (p, df, ncp).
    detail::NoncentralChiSquareInvGradHess invGradHess(double p) const {
        static_assert(std::is_same_v<DoubleT, double>, "pack requires DoubleT = double");
        return detail::noncentralChiSquareInvGradHess(df, ncp, p);
    }
};

} // namespace quantape::math

#endif // QUANTAPE_MATH_DISTRIBUTIONS_NONCENTRAL_CHI_SQUARE_H
