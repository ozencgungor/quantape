#ifndef LIGHTAPE_OPS_NONCENTRAL_CHI_SQUARE_H
#define LIGHTAPE_OPS_NONCENTRAL_CHI_SQUARE_H

/**
 * @file NoncentralChiSquare.h
 * @brief Noncentral chi-square CDF/CCDF/density for `RevScalar` and `Tangent`
 *
 * Public API layer over the ported double kernels in
 * `lightape/base/NoncentralChiSquare.h`. Each `RevScalar` entry records ONE
 * `External` block over the three parents `{nu, lambda, x}` with the exact
 * primal gradient/Hessian precomputed in the payload, so `RevTape::reverse`
 * gets the gradient and `RevTape::hessianVector` replays the block exactly
 * (no finite differences, no fake entries).
 *
 * Exactness status per partial (all from the same-owner quantape kernels,
 * ported in `detail` below):
 * - x: `F_x = +-pdf` (Bessel form) for the CDF/CCDF; `gx` for the density is
 *   the analytic Bessel-ratio slope `f'/f = z R/(2x) - 1/2`.
 * - lambda: the weight-shift recurrences
 *   `F_lambda = (F(nu+2) - F(nu))/2`,
 *   `F_lambdalambda = (F(nu+4) - 2 F(nu+2) + F(nu))/4`,
 *   `F_lambdax = (f(nu+2) - f(nu))/2`, and for the density the same shifts.
 * - nu: the exact shape partials of the Poisson mixture (`nccsNuSums`): the
 *   per-term incomplete-gamma shape jet (`gammaGradHess`) advanced by the
 *   documented shape/weight recurrences, with `F_nu = 1/2 sum w P_a`,
 *   `F_nunu = 1/4 sum w P_aa`, `F_nux = 1/4 sum w P_sy`; no finite
 *   differences. The pdf nu entries use the gamma density's closed-form
 *   shape derivatives in the same sum.
 * - mixed nu/lambda: `F_nulambda = 1/2 [F_nu(nu+2) - F_nu(nu)]`.
 *
 * CCDF is `1 - CDF`, so it reuses the CDF pack negated (`sign = -1`), which is
 * exact. Domain edges mirror the source packs: CDF/CCDF at `x = 0` keep the
 * one-sided parameter derivatives as NaN (they are not represented); the
 * density pack is all-NaN outside `nu > 0, lambda >= 0, x > 0` (the caller
 * still sees the double kernel's value at the `x = 0` limits, with NaN
 * derivatives).
 *
 * All three RevScalar wrappers record full `vjp + svp + vhp` (every entry is
 * an exact analytic expression), so `hessianVector` returns true. The
 * `Tangent<S,N>` overloads evaluate the same first-derivative factors at the
 * primal point and contract them with the tangents; second-order tangent
 * levels only propagate through the double-kernel value (`out.value`), the
 * factors themselves are locally constant, matching the P1 layer.
 */

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>

#include "lightape/base/Digamma.h"
#include "lightape/base/IncompleteGamma.h"
#include "lightape/base/NoncentralChiSquare.h"
#include "lightape/base/ScalarConstant.h"
#include "lightape/fwd/ForwardScalar.h"
#include "lightape/ops/Callbacks.h"
#include "lightape/rev/LeanReverse.h"

namespace lightape {

namespace detail {

// ---------------------------------------------------------------------------
// Double derivative packs (ports of the quantape noncentral chi-square
// derivative kernels; the value kernels themselves live in the base header).
// ---------------------------------------------------------------------------

/// Strips nested `Tangent` levels down to the double primal.
template <typename T>
double nccsPrimal(const T& x) {
    if constexpr (requires { x.value; }) {
        return nccsPrimal(x.value);
    } else {
        return static_cast<double>(x);
    }
}

/// Absolute-weight shape recurrence accumulators for the rate-1 gamma
/// mixture (port of quantape `NccsNuSums`).
struct NccsNuSums {
    double cdfValue = 0.0;
    double cdf1 = 0.0;  // sum w P_s           (x 0.5 for dF/dnu)
    double cdf2 = 0.0;  // sum w P_ss          (x 0.25 for d2F/dnu2)
    double cdfx = 0.0;  // sum w P_sy          (x 0.25 for d2F/dnudx)
    double pdf0 = 0.0;  // sum w g
    double pdf1 = 0.0;  // sum w g_s           (x 0.5 for df/dnu)
    double pdf2 = 0.0;  // sum w g_ss          (x 0.25 for d2f/dnu2)
    double pdfxy = 0.0; // sum w g_sy          (x 0.5 for d2f/dnudx)
    double pdfyy = 0.0; // sum w g_yy
};

/// Rate-1 gamma density and its shape/x derivatives at `(shape, y)`, closed
/// form (the quantape `gammaPdfGradHess(s, 1, y)` restricted to the entries
/// `NccsNuSums` seeds with): `{g, g_s, g_ss, g_sy, g_yy}`.
struct NccsGammaDensityPack {
    double value = 0.0;
    double gShape = 0.0;
    double hShapeShape = 0.0;
    double hShapeX = 0.0;
    double hxx = 0.0;
};

inline NccsGammaDensityPack nccsGammaDensityPack(double shape, double y) {
    const double lny = std::log(y);
    const double psi = digammaDouble(shape);
    const double dpsi = trigammaDouble(shape);
    const double g = std::exp((shape - 1.0) * lny - y - std::lgamma(shape));
    const double lm = lny - psi;
    const double q = (shape - 1.0) / y - 1.0;
    return {g, g * lm, g * (lm * lm - dpsi), g * q * lm + g / y,
            g * (q * q - (shape - 1.0) / (y * y))};
}

/// Shape-recurrence sums over the mode-centred Poisson window. Same
/// evaluation order as the quantape kernel; `gammaGradHess` supplies the
/// incomplete-gamma entries, the density entries are the closed forms above.
inline NccsNuSums nccsNuSums(double nu, double lambda, double x) {
    const double y = 0.5 * x;
    const double lny = std::log(y);
    const double a0 = 0.5 * nu;
    NccsNuSums out;
    if (lambda == 0.0) {
        const GammaGradHess cp = gammaGradHess(a0, y);
        const NccsGammaDensityPack dp = nccsGammaDensityPack(a0, y);
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
    const GammaGradHess cp = gammaGradHess(s, y);
    const NccsGammaDensityPack dp = nccsGammaDensityPack(s, y);
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

/// `{F, dF/dnu, d2F/dnu2, d2F/dnudx}` from one `nccsNuSums` pass.
struct NccsNuMixture {
    double value = 0.0;
    double d1 = 0.0;
    double d2 = 0.0;
    double dx = 0.0;
};

inline NccsNuMixture nccsCdfNuMixture(double nu, double lambda, double x) {
    const NccsNuSums sums = nccsNuSums(nu, lambda, x);
    NccsNuMixture out;
    out.value = sums.cdfValue;
    out.d1 = 0.5 * sums.cdf1;
    out.d2 = 0.25 * sums.cdf2;
    out.dx = 0.25 * sums.cdfx;
    return out;
}

/// `dF/dnu` (single mixture pass).
inline double nccsCdfNuD1(double nu, double lambda, double x) {
    return nccsCdfNuMixture(nu, lambda, x).d1;
}

/// `dF/dlambda = (F(nu+2) - F(nu)) / 2`.
inline double nccsCdfLambdaD1(double nu, double lambda, double x) {
    return 0.5 *
           (noncentralChiSquareCdf(nu + 2.0, lambda, x) - noncentralChiSquareCdf(nu, lambda, x));
}

/// `d2F/dlambda2 = (F(nu+4) - 2 F(nu+2) + F(nu)) / 4`.
inline double nccsCdfLambdaD2(double nu, double lambda, double x) {
    return 0.25 * (noncentralChiSquareCdf(nu + 4.0, lambda, x) -
                   2.0 * noncentralChiSquareCdf(nu + 2.0, lambda, x) +
                   noncentralChiSquareCdf(nu, lambda, x));
}

/// `d2F/dxdlambda = (f(nu+2) - f(nu)) / 2`.
inline double nccsCdfXLambdaD2(double nu, double lambda, double x) {
    return 0.5 *
           (noncentralChiSquarePdf(nu + 2.0, lambda, x) - noncentralChiSquarePdf(nu, lambda, x));
}

/// Shared `{f, f'/f}` for the Bessel form (`lambda > 0`, `x > 0`); the ratio
/// stays finite where `I_a` overflows, and `z = 0` takes `a/x`.
inline void nccsPdfValueSlope(double nu, double lambda, double x, double& f, double& g) {
    const double a = 0.5 * nu - 1.0;
    const double z = std::sqrt(lambda * x);
    f = noncentralChiSquarePdf(nu, lambda, x);
    if (z == 0.0) {
        g = a / x - 0.5;
        return;
    }
    const double ratio = nccsBesselIRatio(a - 1.0, a, z);
    g = z * ratio / (2.0 * x) - 0.5;
}

/// `d/dx` of the noncentral density (NaN for `x -> 0+`).
inline double nccsPdfPrime(double nu, double lambda, double x) {
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
        return noncentralChiSquarePdf(nu, lambda, x) * ((0.5 * nu - 1.0) / x - 0.5);
    }
    double f = 0.0;
    double g = 0.0;
    nccsPdfValueSlope(nu, lambda, x, f, g);
    return f * g;
}

/// `{F, F_nu, F_lambda, F_x, F_nunu, F_nulam, F_nux, F_lalam, F_lamx, F_xx}`
/// for the noncentral CDF, all analytic (port of the quantape pack).
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

inline NccsCdfGradHess nccsCdfGradHessNaN() noexcept {
    const double nan = std::numeric_limits<double>::quiet_NaN();
    return {nan, nan, nan, nan, nan, nan, nan, nan, nan, nan};
}

inline NccsCdfGradHess nccsCdfGradHess(double nu, double lambda, double x) {
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
    NccsCdfGradHess out{};
    // Anchor the pack value on the value kernel (the nu-mixture's own sum
    // differs by a few ulp from the recurrence-based CDF).
    out.value = noncentralChiSquareCdf(nu, lambda, x);
    out.gNu = mix.d1;
    out.gLambda = nccsCdfLambdaD1(nu, lambda, x);
    out.gx = noncentralChiSquarePdf(nu, lambda, x);
    out.hNuNu = mix.d2;
    out.hNuLambda = 0.5 * (nccsCdfNuD1(nu + 2.0, lambda, x) - mix.d1);
    out.hNuX = mix.dx;
    out.hLambdaLambda = nccsCdfLambdaD2(nu, lambda, x);
    out.hLambdaX = nccsCdfXLambdaD2(nu, lambda, x);
    out.hxx = nccsPdfPrime(nu, lambda, x);
    return out;
}

/// `{f, f_nu, f_lambda, f_x, f_nunu, f_nulam, f_nux, f_lalam, f_lamx,
/// f_xx}` for the noncentral density, all analytic (port of the quantape
/// pack; the value and `f_x` are anchored on the Bessel kernels).
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

inline NccsPdfGradHess nccsPdfGradHess(double nu, double lambda, double x) {
    NccsPdfGradHess out;
    const double nan = std::numeric_limits<double>::quiet_NaN();
    if (!(nu > 0.0) || !(lambda >= 0.0) || !(x > 0.0) || std::isnan(nu) || std::isnan(lambda) ||
        std::isnan(x)) {
        out.value = out.gNu = out.gLambda = out.gx = nan;
        out.hNuNu = out.hNuLambda = out.hNuX = nan;
        out.hLambdaLambda = out.hLambdaX = out.hxx = nan;
        return out;
    }
    const NccsNuSums sums = nccsNuSums(nu, lambda, x);
    const NccsNuSums sumsShift = nccsNuSums(nu + 2.0, lambda, x);

    const double value = noncentralChiSquarePdf(nu, lambda, x);
    out.value = value;
    out.gx = nccsPdfPrime(nu, lambda, x);
    out.gNu = 0.25 * sums.pdf1;
    out.hNuNu = 0.125 * sums.pdf2;
    out.hNuX = 0.125 * sums.pdfxy;
    out.hxx = 0.125 * sums.pdfyy;
    out.gLambda = 0.5 * (noncentralChiSquarePdf(nu + 2.0, lambda, x) - value);
    out.hLambdaLambda = 0.25 * (noncentralChiSquarePdf(nu + 4.0, lambda, x) -
                                2.0 * noncentralChiSquarePdf(nu + 2.0, lambda, x) + value);
    out.hLambdaX = 0.5 * (nccsPdfPrime(nu + 2.0, lambda, x) - out.gx);
    out.hNuLambda = 0.125 * (sumsShift.pdf1 - sums.pdf1);
    return out;
}

// ---------------------------------------------------------------------------
// External payloads and callbacks (precomputed primal partials). The CDF and
// CCDF share the same pack; `sign` is +1 for the CDF and -1 for the CCDF
// (`Ccdf = 1 - Cdf`, so every partial flips sign exactly).
// ---------------------------------------------------------------------------

struct NccsCdfPayload {
    double sign = 1.0;
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

inline void nccsCdfVjp(const void* payload, double seed, const ExtArg* a, std::uint32_t /*n*/,
                       RevTape& tape) noexcept {
    const auto* p = static_cast<const NccsCdfPayload*>(payload);
    const double s = seed * p->sign;
    tape.addAdjoint(a[0].node, s * p->gNu);
    tape.addAdjoint(a[1].node, s * p->gLambda);
    tape.addAdjoint(a[2].node, s * p->gx);
}

inline double nccsCdfSvp(const void* payload, const ExtArg* /*a*/, const double* s,
                         std::uint32_t /*n*/) noexcept {
    const auto* p = static_cast<const NccsCdfPayload*>(payload);
    return p->sign * (p->gNu * s[0] + p->gLambda * s[1] + p->gx * s[2]);
}

inline void nccsCdfVhp(const void* payload, double adjoint, const ExtArg* /*a*/, const double* s,
                       std::uint32_t /*n*/, double* b) noexcept {
    const auto* p = static_cast<const NccsCdfPayload*>(payload);
    const double t = adjoint * p->sign;
    b[0] += t * (p->hNuNu * s[0] + p->hNuLambda * s[1] + p->hNuX * s[2]);
    b[1] += t * (p->hNuLambda * s[0] + p->hLambdaLambda * s[1] + p->hLambdaX * s[2]);
    b[2] += t * (p->hNuX * s[0] + p->hLambdaX * s[1] + p->hxx * s[2]);
}

struct NccsPdfPayload {
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

inline void nccsPdfVjp(const void* payload, double seed, const ExtArg* a, std::uint32_t /*n*/,
                       RevTape& tape) noexcept {
    const auto* p = static_cast<const NccsPdfPayload*>(payload);
    tape.addAdjoint(a[0].node, seed * p->gNu);
    tape.addAdjoint(a[1].node, seed * p->gLambda);
    tape.addAdjoint(a[2].node, seed * p->gx);
}

inline double nccsPdfSvp(const void* payload, const ExtArg* /*a*/, const double* s,
                         std::uint32_t /*n*/) noexcept {
    const auto* p = static_cast<const NccsPdfPayload*>(payload);
    return p->gNu * s[0] + p->gLambda * s[1] + p->gx * s[2];
}

inline void nccsPdfVhp(const void* payload, double adjoint, const ExtArg* /*a*/, const double* s,
                       std::uint32_t /*n*/, double* b) noexcept {
    const auto* p = static_cast<const NccsPdfPayload*>(payload);
    b[0] += adjoint * (p->hNuNu * s[0] + p->hNuLambda * s[1] + p->hNuX * s[2]);
    b[1] += adjoint * (p->hNuLambda * s[0] + p->hLambdaLambda * s[1] + p->hLambdaX * s[2]);
    b[2] += adjoint * (p->hNuX * s[0] + p->hLambdaX * s[1] + p->hxx * s[2]);
}

} // namespace detail

// ── RevScalar API (one External block each, full exact vjp + svp + vhp) ──

/// `F(nu, lambda; x)`: exact `(nu, lambda, x)` gradient/Hessian in the
/// payload, so the block replays exactly under `hessianVector`.
inline RevScalar noncentralChiSquareCdf(const RevScalar& nu, const RevScalar& lambda,
                                        const RevScalar& x) {
    const double value = noncentralChiSquareCdf(nu.value, lambda.value, x.value);
    const detail::NccsCdfGradHess pack = detail::nccsCdfGradHess(nu.value, lambda.value, x.value);
    const detail::NccsCdfPayload payload{
        1.0,       pack.gNu,           pack.gLambda,  pack.gx, pack.hNuNu, pack.hNuLambda,
        pack.hNuX, pack.hLambdaLambda, pack.hLambdaX, pack.hxx};
    const RevScalar args[3] = {nu, lambda, x};
    return makeVhpFvar(value, args, 3, detail::nccsCdfVjp, detail::nccsCdfSvp, detail::nccsCdfVhp,
                       payload);
}

/// `Q(nu, lambda; x) = 1 - F`: same pack as the CDF with every partial
/// negated (exact).
inline RevScalar noncentralChiSquareCcdf(const RevScalar& nu, const RevScalar& lambda,
                                         const RevScalar& x) {
    const double value = noncentralChiSquareCcdf(nu.value, lambda.value, x.value);
    const detail::NccsCdfGradHess pack = detail::nccsCdfGradHess(nu.value, lambda.value, x.value);
    const detail::NccsCdfPayload payload{
        -1.0,           pack.gNu,  pack.gLambda,       pack.gx,       pack.hNuNu,
        pack.hNuLambda, pack.hNuX, pack.hLambdaLambda, pack.hLambdaX, pack.hxx};
    const RevScalar args[3] = {nu, lambda, x};
    return makeVhpFvar(value, args, 3, detail::nccsCdfVjp, detail::nccsCdfSvp, detail::nccsCdfVhp,
                       payload);
}

/// `f(nu, lambda; x)`: exact density gradient/Hessian in the payload.
inline RevScalar noncentralChiSquarePdf(const RevScalar& nu, const RevScalar& lambda,
                                        const RevScalar& x) {
    const double value = noncentralChiSquarePdf(nu.value, lambda.value, x.value);
    const detail::NccsPdfGradHess pack = detail::nccsPdfGradHess(nu.value, lambda.value, x.value);
    const detail::NccsPdfPayload payload{pack.gNu,           pack.gLambda,   pack.gx,
                                         pack.hNuNu,         pack.hNuLambda, pack.hNuX,
                                         pack.hLambdaLambda, pack.hLambdaX,  pack.hxx};
    const RevScalar args[3] = {nu, lambda, x};
    return makeVhpFvar(value, args, 3, detail::nccsPdfVjp, detail::nccsPdfSvp, detail::nccsPdfVhp,
                       payload);
}

// ── Tangent API (forward mode; first-derivative factors at the primal) ──

/// `F(nu, lambda; x)`: `d = gNu * dnu + gLambda * dlambda + gx * dx`.
template <typename S, std::size_t N>
Tangent<S, N> noncentralChiSquareCdf(const Tangent<S, N>& nu, const Tangent<S, N>& lambda,
                                     const Tangent<S, N>& x) {
    Tangent<S, N> out;
    out.value = noncentralChiSquareCdf(nu.value, lambda.value, x.value);
    const detail::NccsCdfGradHess pack =
        detail::nccsCdfGradHess(detail::nccsPrimal(nu.value), detail::nccsPrimal(lambda.value),
                                detail::nccsPrimal(x.value));
    const S gNu = detail::constantOf<S>(pack.gNu);
    const S gLambda = detail::constantOf<S>(pack.gLambda);
    const S gx = detail::constantOf<S>(pack.gx);
    for (std::size_t i = 0; i < N; ++i) {
        out.d[i] = gNu * nu.d[i] + gLambda * lambda.d[i] + gx * x.d[i];
    }
    return out;
}

/// `Q = 1 - F`: mirror of the CDF tangent overload with negated factors.
template <typename S, std::size_t N>
Tangent<S, N> noncentralChiSquareCcdf(const Tangent<S, N>& nu, const Tangent<S, N>& lambda,
                                      const Tangent<S, N>& x) {
    Tangent<S, N> out;
    out.value = noncentralChiSquareCcdf(nu.value, lambda.value, x.value);
    const detail::NccsCdfGradHess pack =
        detail::nccsCdfGradHess(detail::nccsPrimal(nu.value), detail::nccsPrimal(lambda.value),
                                detail::nccsPrimal(x.value));
    const S gNu = detail::constantOf<S>(-pack.gNu);
    const S gLambda = detail::constantOf<S>(-pack.gLambda);
    const S gx = detail::constantOf<S>(-pack.gx);
    for (std::size_t i = 0; i < N; ++i) {
        out.d[i] = gNu * nu.d[i] + gLambda * lambda.d[i] + gx * x.d[i];
    }
    return out;
}

/// `f(nu, lambda; x)`: density tangent overload.
template <typename S, std::size_t N>
Tangent<S, N> noncentralChiSquarePdf(const Tangent<S, N>& nu, const Tangent<S, N>& lambda,
                                     const Tangent<S, N>& x) {
    Tangent<S, N> out;
    out.value = noncentralChiSquarePdf(nu.value, lambda.value, x.value);
    const detail::NccsPdfGradHess pack =
        detail::nccsPdfGradHess(detail::nccsPrimal(nu.value), detail::nccsPrimal(lambda.value),
                                detail::nccsPrimal(x.value));
    const S gNu = detail::constantOf<S>(pack.gNu);
    const S gLambda = detail::constantOf<S>(pack.gLambda);
    const S gx = detail::constantOf<S>(pack.gx);
    for (std::size_t i = 0; i < N; ++i) {
        out.d[i] = gNu * nu.d[i] + gLambda * lambda.d[i] + gx * x.d[i];
    }
    return out;
}

} // namespace lightape

#endif // LIGHTAPE_OPS_NONCENTRAL_CHI_SQUARE_H
