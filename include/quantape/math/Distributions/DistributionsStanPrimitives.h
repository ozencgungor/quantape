#ifndef QUANTAPE_MATH_DISTRIBUTIONS_DISTRIBUTIONS_STAN_PRIMITIVES_H
#define QUANTAPE_MATH_DISTRIBUTIONS_DISTRIBUTIONS_STAN_PRIMITIVES_H

/**
 * @file DistributionsStanPrimitives.h
 * @brief Stan `var`/`fvar<var>` specializations for the distribution methods
 *        whose complete first-order parameter partials exist in the double
 *        packs.
 *
 * Stan-side bridge until the in-house `ad::External` adapters land. The
 * generic `DoubleT` scalar path tapes the kernel internals (z, erfc, the
 * incomplete-gamma series/continued fraction, the bivariate quadrature). Two
 * optimized tiers collapse that:
 *
 *  - `var`: ONE `stan::math::make_callback_var` node whose value is the
 *    canonical double kernel and whose reverse pass pushes the analytic
 *    partials directly into the participating `var`s.
 *  - `fvar<var>` (second order, for `stan::math::hessian`): TWO callback
 *    nodes. The value node repeats the `var` route (same value, same gradient
 *    push). The tangent node's value is the exact directional derivative
 *    `g . w` for the tangent weights `w = (.d_.val())`; its reverse pass
 *    pushes `a H w` into the leaves when its adjoint is `a`, which is exactly
 *    what `hessian()` reads column by column, so the kernel is never taped.
 *    Same pattern as `Black76<fvar<var>>::price()`.
 *
 * Every route is written through two helpers with `make_callback_var`-style
 * signatures (see `detail::makeVjpVar` / `detail::makeVhpFvar`):
 *
 *   `var       makeVjpVar(value, pushGrad)`            // 1 node, pushGrad(adj)
 *   `fvar<var> makeVhpFvar(value, tangent, pushGrad, pushHess)`  // 2 nodes
 *
 * with `pushGrad(adj)` pushing `adj * g` and `pushHess(adj)` pushing
 * `adj * H w`. New families must use the same helpers so the planned in-house
 * `ad::External` VJP/VHP payload callbacks are a backend swap, not a rewrite.
 *
 * Optimized methods (all active-parameter first partials and the full
 * gradient/Hessian packs available; the family rows are 1 node in the `var`
 * tier and 2 nodes in the `fvar<var>` tier, the same shape as the
 * Normal/BVN/Poisson rows):
 *
 *   Normal<var>::cdf/pdf/logCdf/ccdf/logCcdf            (x, mu, sigma)  1 node
 *   Normal<fvar<var>>::cdf/pdf/logCdf/ccdf/logCcdf      (x, mu, sigma)  2 nodes
 *   BivariateNormal<var>::cdf                           (a, b, rho)     1 node
 *   BivariateNormal<fvar<var>>::cdf                     (a, b, rho)     2 nodes
 *   Poisson<var>::pmf(int k) / cdf(int k)               (lambda)        1 node
 *   Poisson<fvar<var>>::pmf(int k) / cdf(int k)         (lambda)        2 nodes
 *   Gamma<var/fvar<var>>::{cdf,pdf}                     (shape, rate, x)
 *   InverseGamma<var/fvar<var>>::{cdf,pdf}              (shape, scale, x)
 *   ChiSquare<var/fvar<var>>::{cdf,pdf}                 (df, x)
 *   Beta<var/fvar<var>>::{cdf,pdf}                      (alpha, beta, x)
 *   StudentT<var/fvar<var>>::{cdf,pdf}                  (nu, x)
 *   Lognormal<var/fvar<var>>::{cdf,pdf}                 (mu, sigma, x)
 *   Exponential<var/fvar<var>>::{cdf,pdf}               (lambda, x)
 *   InverseGaussian<var/fvar<var>>::{cdf,pdf}           (mu, lambda, x)
 *   NoncentralChiSquare<var/fvar<var>>::{cdf,ccdf,logCcdf} (df, ncp, x)
 *   NoncentralBeta<var/fvar<var>>::{cdf,pdf,ccdf,logPdf} (alpha, beta, ncp, x)
 *   NoncentralF<var/fvar<var>>::{cdf,pdf,ccdf,logPdf}   (nu1, nu2, ncp, y)
 *
 * Partial formulas per Normal/BVN/Poisson specialization (z = (x - mu) /
 * sigma, phi = normal density, Phi = normal CDF):
 *
 *   cdf     : g_x = phi/sigma, g_mu = -phi/sigma, g_sigma = -z phi/sigma
 *   pdf     : g_x = -z phi/sigma, g_mu = z phi/sigma,
 *             g_sigma = (z^2 - 1) phi/sigma
 *   logCdf  : g_x = phi/(sigma Phi), g_mu = -g_x, g_sigma = -z g_x
 *   ccdf    : value = 1/2 erfc(z/sqrt(2)); gradients = -(cdf gradients)
 *   logCcdf : mirror of logCdf with G = ccdf, gradG = -(cdf gradients)
 *   BVN cdf : g_a = phi(a) Phi((b - rho a)/s), g_b = phi(b) Phi((a - rho b)/s),
 *             g_rho = phi2(a, b; rho),  s = sqrt(1 - rho^2)
 *   P pmf   : d/dlambda = pmf (k - lambda)/lambda
 *   P cdf   : d/dlambda = -pmf(k)
 *
 * Noncentral tail/log routes (all built from the family packs; the node value
 * is the matching double accessor, never `pack.value`, so the log-space
 * accessors stay exact):
 *
 *   ccdf     : value = ccdf(x); grads = -grad F; Hess = -Hess F
 *   logCcdf  : G = ccdf, g = grad ccdf = -grad F, H = -Hess F;
 *              grads = g/G; Hess_log = H/G - g g^T/G^2
 *   logPdf   : f = density pack value; grads = f_i/f;
 *              Hess_log = f_ij/f - f_i f_j/f^2
 *
 * The family routes carry no formulas here: the value and every partial come
 * from the family's own `*GradHess` double pack (Gamma/InverseGamma in
 * `detail/IncompleteGamma.h`, ChiSquare = Gamma(df/2, 1/2), Beta and StudentT
 * from the exact incomplete-beta `SecondOrderJet3` pack, the closed-form
 * Lognormal/Exponential/InverseGaussian packs, and the noncentral
 * `NccsCdfGradHess`/`NoncentralBeta*GradHess`/`NoncentralF*GradHess` mixtures in
 * `NoncentralChiSquare.h`/`NoncentralBeta.h`). `detail::makeVjpFromPack` /
 * `detail::makeVhpFromPack` map the pack fields onto the active leaves at one
 * place per family; the reverse pass still ends in `makeVjpVar` /
 * `makeVhpFvar`, so every specialization has the same callback shape.
 *
 * The `fvar<var>` tier uses the full analytic Hessian packs:
 * `detail::normalCdfGradHess`, the new `detail::normalPdfGradHess` (all 9
 * entries each), `detail::bvnCdfGradHess` (3x3), `detail::poissonPmfD1D2` /
 * `detail::poissonCdfD1D2` (`d2`) and the family `*GradHess` packs above
 * (including the 3x3 `NccsCdfGradHess` and the 4x4 noncentral-beta/F packs).
 * `logCdf`/`logCcdf` apply the quotient rule to the corresponding pack at the
 * evaluation point: `grad = g/F`, `H_log = H/F - g g^T/F^2` (for logCcdf,
 * `G = 1 - F`, `gradG = -g`, `HessG = -H`). The `var` tier mirrors those
 * ratios for `logCdf`/`logCcdf` (1 node each).
 *
 * Value semantics are IDENTICAL to the generic path: the value is the same
 * double expression the generic scalar path evaluates, so edge cases
 * (sigma <= 0, infinities, NaN) match exactly. In particular `logCdf` is
 * `log(cdf)` (the generic `normalLogCdfT`), NOT the robust asymptotic of
 * `normalLogCdfDouble`: the far-tail underflow behaviour is unchanged.
 * The `fvar<var>` `logCdf`/`logCcdf` fall back to the generic taping path only
 * when the probability underflows to exactly zero (the quotient rule would
 * divide by zero); the `var` ratios mirror the original var `logCdf` route
 * without that guard (log(0) and IEEE inf/NaN behave as in the generic path).
 * Where the double pack cannot represent the boundary the
 * specialization falls back to the generic path so values and adjoints match
 * exactly there too:
 * BivariateNormal for non-finite arguments or |rho| >= 1 (pack needs finite
 * a, b and |rho| < 1; the generic path keeps the exact inf / |rho| = 1 limits
 * and NaN propagation), Poisson for lambda <= 0 or NaN (the pack reports
 * one-sided limits at the lambda = 0 point mass, the generic path keeps the
 * constant / NaN value and zero lambda sensitivity).
 *
 * Family fallbacks (the generic scalar kernel, same value/adjoints): invalid
 * parameters (non-positive or non-finite shape/scale/df/alpha/beta/nu/mu/
 * sigma/lambda), NaN x, and the singular one-sided pack boundaries that the
 * packs cannot represent as a full derivative array: Gamma, InverseGamma and
 * ChiSquare at x = 0 (pack all-NaN, the generic path keeps the finite
 * one-sided limits) and Beta at x = 0 / x = 1. StudentT cdf also falls back
 * at x = 0: the pack reports the right-derivative limit (`gx = pdf(0)`) while
 * the generic path keeps the symmetric zero derivative at the median.
 * NoncentralChiSquare::cdf and ::ccdf share the x = 0 fallback (the
 * `NccsCdfGradHess` pack is all-NaN there; the generic tapped mixture returns
 * 0 / ~1 with zero adjoints) plus invalid/non-finite df-ncp; their generic
 * scalar mixtures are mirrored in `detail::nccsCdfGeneric` /
 * `detail::nccsCcdfGeneric` because the specializations replace the primary
 * members. NoncentralChiSquare::logCcdf has no generic tapped body to fall back
 * to (the header accessor is double-only): invalid/NaN inputs and the x = 0
 * constant branch (log ccdf = 0) return the accessor value with zero pushes,
 * and the deep-tail G = ccdf == 0 branch (double underflow, finite log-space
 * accessor) does the same -- the pack quotient cannot be formed there and no
 * log-space derivative pack exists, so no gradients are pushed rather than
 * wrong ones. Everywhere else -- x outside the support (constant branches),
 * the saturating x = +-Inf limits, Beta outside [0, 1] -- the pack is used,
 * its value and partials agreeing with the generic path there.
 *
 * NoncentralBeta and NoncentralF have no generic scalar `T` body at all (their
 * header accessors are double-only and the packs are the single implementation),
 * so their cdf/pdf/ccdf/logPdf specializations take the pack for every edge:
 * invalid parameters -> all-NaN, x < 0 / x > 1 -> zero-CDF / zero-density
 * packs, x = 0 / 1 -> the one-sided closed forms (with the mixed parameter-x
 * entries zero, exactly as `noncentralBeta*GradHess` documents). `logPdf`
 * pushes `f_i/f` and `f_ij/f - f_i f_j/f^2` from the density pack; when the
 * double density is exactly zero (support edge, or underflow where the
 * log-space accessor is still finite) the quotient is undefined, so it zeroes
 * the pushes and keeps the accessor's (-Inf / finite) value.
 *
 * The remaining methods stay on the generic scalar path because at least one
 * active-parameter first partial has no double pack yet:
 *
 *  - Normal<fvar<var>>::{logPdf, inv}: logPdf has no (mu, sigma) derivative
 *    pack and inv has no closed-form parameter derivative pack.
 *  - BivariateNormal::pdf: no (a, b, rho) gradient pack.
 *  - Poisson::{logPmf, ccdf}: no lambda-derivative pack (ccdf would be the
 *    sign-flip of poissonCdfD1D2 but is out of scope here).
 *  - All methods other than {cdf, pdf} for Gamma, InverseGamma, ChiSquare,
 *    Beta, StudentT, Lognormal, Exponential and InverseGaussian (log forms,
 *    ccdf, inv): no parameter derivative packs yet.
 *  - NoncentralChiSquare::logCdf: not a member of the distribution header,
 *    so there is nothing to specialize. `pdf`/`logPdf` are now optimized
 *    through `NccsPdfGradHess` (Poisson-mixture density pack with the exact
 *    nu/lambda/x first and second partials; see the density section below).
 *  - NoncentralBeta/NoncentralF::logCcdf: not members of the headers (only
 *    ccdf/logPdf exist), so there is nothing to specialize.
 *
 * The callbacks capture the participating `var` handles by value (a `var` is
 * trivially destructible and shares its `vi_`), never `this`: the
 * specialization then stays correct even when the distribution object is a
 * temporary that dies before the reverse pass.
 *
 * Include "quantape/math/StanMath.h" first (this header does it) and only
 * from Stan AD translation units; non-Stan builds include Distributions.h and
 * see the generic templates only.
 */

#include "quantape/math/StanMath.h"

#include "quantape/math/Distributions/Beta.h"
#include "quantape/math/Distributions/BivariateNormal.h"
#include "quantape/math/Distributions/ChiSquare.h"
#include "quantape/math/Distributions/Exponential.h"
#include "quantape/math/Distributions/Gamma.h"
#include "quantape/math/Distributions/InverseGaussian.h"
#include "quantape/math/Distributions/Lognormal.h"
#include "quantape/math/Distributions/NoncentralBeta.h"
#include "quantape/math/Distributions/NoncentralChiSquare.h"
#include "quantape/math/Distributions/Normal.h"
#include "quantape/math/Distributions/Poisson.h"
#include "quantape/math/Distributions/StudentT.h"

#include <array>
#include <cmath>
#include <cstddef>

namespace quantape::math {
namespace detail {

/// One-node gradient callback in the `make_callback_var(value, fn)` shape.
///
/// `pushGrad(adj)` pushes `adj * g_i` into the participating leaves. The inner
/// lambda is invoked on the reverse pass with the node's adjoint, which is the
/// same signature shape the planned in-house `ad::External` VJP payloads use
/// (`void(const P& payload, double value, double adj, RevTape&) noexcept`), so
/// each distribution route is written once and swapped to the in-house tape
/// when it lands.
template <class PushGrad>
inline stan::math::var makeVjpVar(double value, PushGrad pushGrad) {
    return stan::math::make_callback_var(value,
                                         [pushGrad](auto& vi) mutable { pushGrad(vi.adj()); });
}

/// Two-node gradient + Hessian result, the `make_callback_var` shape one order
/// up: the value node is the `makeVjpVar` route; the tangent node's value is
/// the exact directional derivative `g . w`, and its reverse pass pushes
/// `adj * H w` (`pushHess(adj)`). Maps 1:1 to the planned `ad::External` VHP
/// payload callback (same signature with the tangent weights `w`).
///
/// Use this for every second-order distribution route so the callback shape is
/// uniform across families.
template <class PushGrad, class PushHess>
inline stan::math::fvar<stan::math::var> makeVhpFvar(double value, double tangent,
                                                     PushGrad pushGrad, PushHess pushHess) {
    stan::math::var valueNode = makeVjpVar(value, pushGrad);
    stan::math::var tangentNode = stan::math::make_callback_var(
        tangent, [pushHess](auto& vi) mutable { pushHess(vi.adj()); });
    return stan::math::fvar<stan::math::var>(valueNode, tangentNode);
}

// ---------------------------------------------------------------------------
// Generic analytic-pack adapters and the per-family field maps. `Pack` only
// supplies `value`; the leaves and their first partials (plus, for the VHP
// route, the tangent weights and row-major Hessian) are spelled out in one
// place per family. The reverse passes still end in `makeVjpVar` /
// `makeVhpFvar`, so every family route has the same callback shape.
// ---------------------------------------------------------------------------

/// One-node result for `N` active leaves: same callback shape as
/// `makeVjpFromPack`, with an explicit node value. Used by the log/tail
/// transforms whose value is the double accessor, not `pack.value`.
template <std::size_t N>
inline stan::math::var makeVjpFromParts(double value, const std::array<stan::math::var, N>& leaves,
                                        const std::array<double, N>& grads) {
    return makeVjpVar(value, [leaves, grads](double adj) {
        for (std::size_t i = 0; i < N; ++i) {
            leaves[i].adj() += adj * grads[i];
        }
    });
}

/// Two-node counterpart of `makeVjpFromParts` (tangent value `g . w`, reverse
/// pass pushing `adj * H w`, row-major symmetric `hess`).
template <std::size_t N>
inline stan::math::fvar<stan::math::var>
makeVhpFromParts(double value, const std::array<stan::math::var, N>& leaves,
                 const std::array<double, N>& grads, const std::array<double, N>& weights,
                 const std::array<double, N * N>& hess) {
    double tangent = 0.0;
    for (std::size_t i = 0; i < N; ++i) {
        tangent += grads[i] * weights[i];
    }
    return makeVhpFvar(
        value, tangent,
        [leaves, grads](double adj) {
            for (std::size_t i = 0; i < N; ++i) {
                leaves[i].adj() += adj * grads[i];
            }
        },
        [leaves, weights, hess](double adj) {
            for (std::size_t i = 0; i < N; ++i) {
                double hw = 0.0;
                for (std::size_t j = 0; j < N; ++j) {
                    hw += hess[i * N + j] * weights[j];
                }
                leaves[i].adj() += adj * hw;
            }
        });
}

/// One-node result for an analytic pack with `N` active leaves.
template <class Pack, std::size_t N>
inline stan::math::var makeVjpFromPack(const Pack& pack,
                                       const std::array<stan::math::var, N>& leaves,
                                       const std::array<double, N>& grads) {
    return makeVjpFromParts(pack.value, leaves, grads);
}

/// Two-node result for an analytic pack: tangent value `g . w`, reverse pass
/// pushing `adj * H w` (`hess` row-major, symmetric).
template <class Pack, std::size_t N>
inline stan::math::fvar<stan::math::var>
makeVhpFromPack(const Pack& pack, const std::array<stan::math::var, N>& leaves,
                const std::array<double, N>& grads, const std::array<double, N>& weights,
                const std::array<double, N * N>& hess) {
    return makeVhpFromParts(pack.value, leaves, grads, weights, hess);
}

/// Gamma(shape, rate) pack -> (shape, rate, x) route.
template <class Pack>
inline stan::math::var gammaGradHessVjp(const Pack& pack, stan::math::var shape,
                                        stan::math::var rate, stan::math::var x) {
    return makeVjpFromPack(pack, std::array<stan::math::var, 3>{shape, rate, x},
                           std::array<double, 3>{pack.gShape, pack.gRate, pack.gx});
}

template <class Pack>
inline stan::math::fvar<stan::math::var> gammaGradHessVhp(const Pack& pack, stan::math::var shape,
                                                          stan::math::var rate, stan::math::var x,
                                                          double wShape, double wRate, double wx) {
    return makeVhpFromPack(pack, std::array<stan::math::var, 3>{shape, rate, x},
                           std::array<double, 3>{pack.gShape, pack.gRate, pack.gx},
                           std::array<double, 3>{wShape, wRate, wx},
                           std::array<double, 9>{pack.hShapeShape, pack.hShapeRate, pack.hShapeX,
                                                 pack.hShapeRate, pack.hRateRate, pack.hRateX,
                                                 pack.hShapeX, pack.hRateX, pack.hxx});
}

/// InverseGamma(shape, scale) pack -> (shape, scale, x) route.
template <class Pack>
inline stan::math::var invGammaGradHessVjp(const Pack& pack, stan::math::var shape,
                                           stan::math::var scale, stan::math::var x) {
    return makeVjpFromPack(pack, std::array<stan::math::var, 3>{shape, scale, x},
                           std::array<double, 3>{pack.gShape, pack.gScale, pack.gx});
}

template <class Pack>
inline stan::math::fvar<stan::math::var>
invGammaGradHessVhp(const Pack& pack, stan::math::var shape, stan::math::var scale,
                    stan::math::var x, double wShape, double wScale, double wx) {
    return makeVhpFromPack(pack, std::array<stan::math::var, 3>{shape, scale, x},
                           std::array<double, 3>{pack.gShape, pack.gScale, pack.gx},
                           std::array<double, 3>{wShape, wScale, wx},
                           std::array<double, 9>{pack.hShapeShape, pack.hShapeScale, pack.hShapeX,
                                                 pack.hShapeScale, pack.hScaleScale, pack.hScaleX,
                                                 pack.hShapeX, pack.hScaleX, pack.hxx});
}

/// ChiSquare(df) pack -> (df, x) route.
template <class Pack>
inline stan::math::var chiSquareGradHessVjp(const Pack& pack, stan::math::var df,
                                            stan::math::var x) {
    return makeVjpFromPack(pack, std::array<stan::math::var, 2>{df, x},
                           std::array<double, 2>{pack.gDf, pack.gx});
}

template <class Pack>
inline stan::math::fvar<stan::math::var> chiSquareGradHessVhp(const Pack& pack, stan::math::var df,
                                                              stan::math::var x, double wDf,
                                                              double wx) {
    return makeVhpFromPack(pack, std::array<stan::math::var, 2>{df, x},
                           std::array<double, 2>{pack.gDf, pack.gx}, std::array<double, 2>{wDf, wx},
                           std::array<double, 4>{pack.hDfDf, pack.hDfX, pack.hDfX, pack.hxx});
}

/// Beta(alpha, beta) pack -> (alpha, beta, x) route.
template <class Pack>
inline stan::math::var betaGradHessVjp(const Pack& pack, stan::math::var alpha,
                                       stan::math::var beta, stan::math::var x) {
    return makeVjpFromPack(pack, std::array<stan::math::var, 3>{alpha, beta, x},
                           std::array<double, 3>{pack.gAlpha, pack.gBeta, pack.gx});
}

template <class Pack>
inline stan::math::fvar<stan::math::var> betaGradHessVhp(const Pack& pack, stan::math::var alpha,
                                                         stan::math::var beta, stan::math::var x,
                                                         double wAlpha, double wBeta, double wx) {
    return makeVhpFromPack(pack, std::array<stan::math::var, 3>{alpha, beta, x},
                           std::array<double, 3>{pack.gAlpha, pack.gBeta, pack.gx},
                           std::array<double, 3>{wAlpha, wBeta, wx},
                           std::array<double, 9>{pack.hAA, pack.hAB, pack.hAx, pack.hAB, pack.hBB,
                                                 pack.hBx, pack.hAx, pack.hBx, pack.hxx});
}

/// StudentT(nu) pack -> (nu, x) route.
template <class Pack>
inline stan::math::var studentTGradHessVjp(const Pack& pack, stan::math::var nu,
                                           stan::math::var x) {
    return makeVjpFromPack(pack, std::array<stan::math::var, 2>{nu, x},
                           std::array<double, 2>{pack.gNu, pack.gx});
}

template <class Pack>
inline stan::math::fvar<stan::math::var> studentTGradHessVhp(const Pack& pack, stan::math::var nu,
                                                             stan::math::var x, double wNu,
                                                             double wx) {
    return makeVhpFromPack(pack, std::array<stan::math::var, 2>{nu, x},
                           std::array<double, 2>{pack.gNu, pack.gx}, std::array<double, 2>{wNu, wx},
                           std::array<double, 4>{pack.hNuNu, pack.hNux, pack.hNux, pack.hxx});
}

/// Lognormal(mu, sigma) pack -> (mu, sigma, x) route.
template <class Pack>
inline stan::math::var lognormalGradHessVjp(const Pack& pack, stan::math::var mu,
                                            stan::math::var sigma, stan::math::var x) {
    return makeVjpFromPack(pack, std::array<stan::math::var, 3>{mu, sigma, x},
                           std::array<double, 3>{pack.gmu, pack.gsig, pack.gx});
}

template <class Pack>
inline stan::math::fvar<stan::math::var>
lognormalGradHessVhp(const Pack& pack, stan::math::var mu, stan::math::var sigma, stan::math::var x,
                     double wMu, double wSigma, double wx) {
    return makeVhpFromPack(
        pack, std::array<stan::math::var, 3>{mu, sigma, x},
        std::array<double, 3>{pack.gmu, pack.gsig, pack.gx}, std::array<double, 3>{wMu, wSigma, wx},
        std::array<double, 9>{pack.hmumu, pack.hmusig, pack.hmux, pack.hmusig, pack.hsigsig,
                              pack.hsigx, pack.hmux, pack.hsigx, pack.hxx});
}

/// Exponential(lambda) pack -> (lambda, x) route.
template <class Pack>
inline stan::math::var exponentialGradHessVjp(const Pack& pack, stan::math::var lambda,
                                              stan::math::var x) {
    return makeVjpFromPack(pack, std::array<stan::math::var, 2>{lambda, x},
                           std::array<double, 2>{pack.gLambda, pack.gx});
}

template <class Pack>
inline stan::math::fvar<stan::math::var>
exponentialGradHessVhp(const Pack& pack, stan::math::var lambda, stan::math::var x, double wLambda,
                       double wx) {
    return makeVhpFromPack(pack, std::array<stan::math::var, 2>{lambda, x},
                           std::array<double, 2>{pack.gLambda, pack.gx},
                           std::array<double, 2>{wLambda, wx},
                           std::array<double, 4>{pack.hLL, pack.hLx, pack.hLx, pack.hxx});
}

/// InverseGaussian(mu, lambda) pack -> (mu, lambda, x) route.
template <class Pack>
inline stan::math::var inverseGaussianGradHessVjp(const Pack& pack, stan::math::var mu,
                                                  stan::math::var lambda, stan::math::var x) {
    return makeVjpFromPack(pack, std::array<stan::math::var, 3>{mu, lambda, x},
                           std::array<double, 3>{pack.gmu, pack.gLambda, pack.gx});
}

template <class Pack>
inline stan::math::fvar<stan::math::var>
inverseGaussianGradHessVhp(const Pack& pack, stan::math::var mu, stan::math::var lambda,
                           stan::math::var x, double wMu, double wLambda, double wx) {
    return makeVhpFromPack(pack, std::array<stan::math::var, 3>{mu, lambda, x},
                           std::array<double, 3>{pack.gmu, pack.gLambda, pack.gx},
                           std::array<double, 3>{wMu, wLambda, wx},
                           std::array<double, 9>{pack.hmumu, pack.hmuLambda, pack.hmux,
                                                 pack.hmuLambda, pack.hLL, pack.hLx, pack.hmux,
                                                 pack.hLx, pack.hxx});
}

/// NoncentralChiSquare(df=nu, ncp=lambda) pack -> (df, ncp, x) route.
template <class Pack>
inline stan::math::var nccsGradHessVjp(const Pack& pack, stan::math::var df, stan::math::var ncp,
                                       stan::math::var x) {
    return makeVjpFromPack(pack, std::array<stan::math::var, 3>{df, ncp, x},
                           std::array<double, 3>{pack.gNu, pack.gLambda, pack.gx});
}

template <class Pack>
inline stan::math::fvar<stan::math::var> nccsGradHessVhp(const Pack& pack, stan::math::var df,
                                                         stan::math::var ncp, stan::math::var x,
                                                         double wDf, double wNcp, double wx) {
    return makeVhpFromPack(pack, std::array<stan::math::var, 3>{df, ncp, x},
                           std::array<double, 3>{pack.gNu, pack.gLambda, pack.gx},
                           std::array<double, 3>{wDf, wNcp, wx},
                           std::array<double, 9>{pack.hNuNu, pack.hNuLambda, pack.hNuX,
                                                 pack.hNuLambda, pack.hLambdaLambda, pack.hLambdaX,
                                                 pack.hNuX, pack.hLambdaX, pack.hxx});
}

/// NoncentralBeta(alpha, beta, ncp) pack -> (alpha, beta, ncp, x) route (shared
/// by the CDF and density packs, which have the same 15-field layout).
template <class Pack>
inline stan::math::var ncbGradHessVjp(const Pack& pack, stan::math::var alpha, stan::math::var beta,
                                      stan::math::var ncp, stan::math::var x) {
    return makeVjpFromPack(pack, std::array<stan::math::var, 4>{alpha, beta, ncp, x},
                           std::array<double, 4>{pack.gAlpha, pack.gBeta, pack.gNcp, pack.gx});
}

template <class Pack>
inline stan::math::fvar<stan::math::var>
ncbGradHessVhp(const Pack& pack, stan::math::var alpha, stan::math::var beta, stan::math::var ncp,
               stan::math::var x, double wAlpha, double wBeta, double wNcp, double wx) {
    return makeVhpFromPack(pack, std::array<stan::math::var, 4>{alpha, beta, ncp, x},
                           std::array<double, 4>{pack.gAlpha, pack.gBeta, pack.gNcp, pack.gx},
                           std::array<double, 4>{wAlpha, wBeta, wNcp, wx},
                           std::array<double, 16>{pack.hAA, pack.hAB, pack.hANcp, pack.hAx,
                                                  pack.hAB, pack.hBB, pack.hBNcp, pack.hBx,
                                                  pack.hANcp, pack.hBNcp, pack.hNcpNcp, pack.hNcpX,
                                                  pack.hAx, pack.hBx, pack.hNcpX, pack.hxx});
}

/// NoncentralF(nu1, nu2, ncp) pack -> (nu1, nu2, ncp, y) route (shared by the
/// CDF and density packs, which have the same 15-field layout).
template <class Pack>
inline stan::math::var ncfGradHessVjp(const Pack& pack, stan::math::var nu1, stan::math::var nu2,
                                      stan::math::var ncp, stan::math::var y) {
    return makeVjpFromPack(pack, std::array<stan::math::var, 4>{nu1, nu2, ncp, y},
                           std::array<double, 4>{pack.gNu1, pack.gNu2, pack.gNcp, pack.gy});
}

template <class Pack>
inline stan::math::fvar<stan::math::var>
ncfGradHessVhp(const Pack& pack, stan::math::var nu1, stan::math::var nu2, stan::math::var ncp,
               stan::math::var y, double wNu1, double wNu2, double wNcp, double wy) {
    return makeVhpFromPack(pack, std::array<stan::math::var, 4>{nu1, nu2, ncp, y},
                           std::array<double, 4>{pack.gNu1, pack.gNu2, pack.gNcp, pack.gy},
                           std::array<double, 4>{wNu1, wNu2, wNcp, wy},
                           std::array<double, 16>{pack.hNu1Nu1, pack.hNu1Nu2, pack.hNu1Ncp,
                                                  pack.hNu1Y, pack.hNu1Nu2, pack.hNu2Nu2,
                                                  pack.hNu2Ncp, pack.hNu2Y, pack.hNu1Ncp,
                                                  pack.hNu2Ncp, pack.hNcpNcp, pack.hNcpY,
                                                  pack.hNu1Y, pack.hNu2Y, pack.hNcpY, pack.hyy});
}

/// Mirror of the generic `NoncentralChiSquare<DoubleT>::cdf` Poisson-mixture
/// body for the fallback boundaries of the optimized routes (invalid
/// parameters, NaN x and the x = 0 pack all-NaN boundary). The specialization
/// replaces that member for `var`/`fvar<var>`, so this keeps the fallback
/// values and adjoints identical to the pre-specialization path. Keep in sync
/// with the generic branch in NoncentralChiSquare.h.
template <class T>
inline T nccsCdfGeneric(const T& df, const T& ncp, const T& x) {
    using std::exp;
    const double lam = scalarValue(ncp);
    const long jmax = static_cast<long>(lam * 0.5 + 12.0 * std::sqrt(lam * 0.5) + 60.0);
    const T half = T(0.5);
    T w = exp(-half * ncp);
    T sum = T(0);
    for (long j = 0; j <= jmax; ++j) {
        sum += w * gammaP(half * df + T(j), half * x);
        w *= (half * ncp) / T(j + 1);
    }
    return sum;
}

/// Mirror of the generic `NoncentralChiSquare<DoubleT>::ccdf` Poisson-mixture
/// body; same fallback contract as `nccsCdfGeneric`. Keep in sync with the
/// generic branch in NoncentralChiSquare.h.
template <class T>
inline T nccsCcdfGeneric(const T& df, const T& ncp, const T& x) {
    using std::exp;
    const double lam = scalarValue(ncp);
    const long jmax = static_cast<long>(lam * 0.5 + 12.0 * std::sqrt(lam * 0.5) + 60.0);
    const T half = T(0.5);
    T w = exp(-half * ncp);
    T sum = T(0);
    for (long j = 0; j <= jmax; ++j) {
        sum += w * gammaQ(half * df + T(j), half * x);
        w *= (half * ncp) / T(j + 1);
    }
    return sum;
}

// ---------------------------------------------------------------------------
// Transformed (tail / log) routes onto the noncentral packs. The node value is
// always the distribution header's double accessor, NOT `pack.value`: the
// log-space accessors (`logCcdf`, `logPdf`) are accurate where the double
// probability underflows, so the value cannot be reconstructed from the pack.
// The push arrays are the elementary transforms documented per member
// specialization; each route still ends in `makeVjpFromParts` /
// `makeVhpFromParts` (hence `makeVjpVar` / `makeVhpFvar`).
// ---------------------------------------------------------------------------

/// `ccdf = 1 - cdf`: negate every `NccsCdfGradHess` field (df, ncp, x).
template <class Pack>
inline stan::math::var nccsCcdfVjp(const Pack& pack, double value, stan::math::var df,
                                   stan::math::var ncp, stan::math::var x) {
    return makeVjpFromParts(value, std::array<stan::math::var, 3>{df, ncp, x},
                            std::array<double, 3>{-pack.gNu, -pack.gLambda, -pack.gx});
}

template <class Pack>
inline stan::math::fvar<stan::math::var>
nccsCcdfVhp(const Pack& pack, double value, stan::math::var df, stan::math::var ncp,
            stan::math::var x, double wDf, double wNcp, double wx) {
    return makeVhpFromParts(value, std::array<stan::math::var, 3>{df, ncp, x},
                            std::array<double, 3>{-pack.gNu, -pack.gLambda, -pack.gx},
                            std::array<double, 3>{wDf, wNcp, wx},
                            std::array<double, 9>{-pack.hNuNu, -pack.hNuLambda, -pack.hNuX,
                                                  -pack.hNuLambda, -pack.hLambdaLambda,
                                                  -pack.hLambdaX, -pack.hNuX, -pack.hLambdaX,
                                                  -pack.hxx});
}

/// `log(ccdf)` quotient on the negated `NccsCdfGradHess` pack with
/// `G = 1 - F`: `grad = -F_grad/G`, `H_log = -F_Hess/G - grad grad^T`.
template <class Pack>
inline stan::math::var nccsLogCcdfVjp(const Pack& pack, double value, double ccdf,
                                      stan::math::var df, stan::math::var ncp, stan::math::var x) {
    const double invG = 1.0 / ccdf;
    return makeVjpFromParts(
        value, std::array<stan::math::var, 3>{df, ncp, x},
        std::array<double, 3>{-pack.gNu * invG, -pack.gLambda * invG, -pack.gx * invG});
}

template <class Pack>
inline stan::math::fvar<stan::math::var>
nccsLogCcdfVhp(const Pack& pack, double value, double ccdf, stan::math::var df, stan::math::var ncp,
               stan::math::var x, double wDf, double wNcp, double wx) {
    const double invG = 1.0 / ccdf;
    const double gDf = -pack.gNu * invG;
    const double gNcp = -pack.gLambda * invG;
    const double gx = -pack.gx * invG;
    return makeVhpFromParts(
        value, std::array<stan::math::var, 3>{df, ncp, x}, std::array<double, 3>{gDf, gNcp, gx},
        std::array<double, 3>{wDf, wNcp, wx},
        std::array<double, 9>{-pack.hNuNu * invG - gDf * gDf, -pack.hNuLambda * invG - gDf * gNcp,
                              -pack.hNuX * invG - gDf * gx, -pack.hNuLambda * invG - gNcp * gDf,
                              -pack.hLambdaLambda * invG - gNcp * gNcp,
                              -pack.hLambdaX * invG - gNcp * gx, -pack.hNuX * invG - gx * gDf,
                              -pack.hLambdaX * invG - gx * gNcp, -pack.hxx * invG - gx * gx});
}

/// `ccdf = 1 - cdf`: negate every `NoncentralBeta*GradHess` field.
template <class Pack>
inline stan::math::var ncbCcdfVjp(const Pack& pack, double value, stan::math::var alpha,
                                  stan::math::var beta, stan::math::var ncp, stan::math::var x) {
    return makeVjpFromParts(value, std::array<stan::math::var, 4>{alpha, beta, ncp, x},
                            std::array<double, 4>{-pack.gAlpha, -pack.gBeta, -pack.gNcp, -pack.gx});
}

template <class Pack>
inline stan::math::fvar<stan::math::var>
ncbCcdfVhp(const Pack& pack, double value, stan::math::var alpha, stan::math::var beta,
           stan::math::var ncp, stan::math::var x, double wAlpha, double wBeta, double wNcp,
           double wx) {
    return makeVhpFromParts(
        value, std::array<stan::math::var, 4>{alpha, beta, ncp, x},
        std::array<double, 4>{-pack.gAlpha, -pack.gBeta, -pack.gNcp, -pack.gx},
        std::array<double, 4>{wAlpha, wBeta, wNcp, wx},
        std::array<double, 16>{-pack.hAA, -pack.hAB, -pack.hANcp, -pack.hAx, -pack.hAB, -pack.hBB,
                               -pack.hBNcp, -pack.hBx, -pack.hANcp, -pack.hBNcp, -pack.hNcpNcp,
                               -pack.hNcpX, -pack.hAx, -pack.hBx, -pack.hNcpX, -pack.hxx});
}

/// `log(pdf)`: `grad = f_i/f`, `H_log = f_ij/f - f_i f_j/f^2`. A zero pack
/// density (support edge or underflow, where the log-space accessor may still
/// be finite) has no representable quotient: the pushes are zeroed.
template <class Pack>
inline stan::math::var ncbLogPdfVjp(const Pack& pack, double value, stan::math::var alpha,
                                    stan::math::var beta, stan::math::var ncp, stan::math::var x) {
    if (pack.value == 0.0) {
        return makeVjpFromParts(value, std::array<stan::math::var, 4>{alpha, beta, ncp, x},
                                std::array<double, 4>{0.0, 0.0, 0.0, 0.0});
    }
    const double invF = 1.0 / pack.value;
    return makeVjpFromParts(value, std::array<stan::math::var, 4>{alpha, beta, ncp, x},
                            std::array<double, 4>{pack.gAlpha * invF, pack.gBeta * invF,
                                                  pack.gNcp * invF, pack.gx * invF});
}

template <class Pack>
inline stan::math::fvar<stan::math::var>
ncbLogPdfVhp(const Pack& pack, double value, stan::math::var alpha, stan::math::var beta,
             stan::math::var ncp, stan::math::var x, double wAlpha, double wBeta, double wNcp,
             double wx) {
    const std::array<stan::math::var, 4> leaves{alpha, beta, ncp, x};
    const std::array<double, 4> weights{wAlpha, wBeta, wNcp, wx};
    if (pack.value == 0.0) {
        return makeVhpFromParts(value, leaves, std::array<double, 4>{0.0, 0.0, 0.0, 0.0}, weights,
                                std::array<double, 16>{});
    }
    const double invF = 1.0 / pack.value;
    const double gAlpha = pack.gAlpha * invF;
    const double gBeta = pack.gBeta * invF;
    const double gNcp = pack.gNcp * invF;
    const double gx = pack.gx * invF;
    return makeVhpFromParts(
        value, leaves, std::array<double, 4>{gAlpha, gBeta, gNcp, gx}, weights,
        std::array<double, 16>{pack.hAA * invF - gAlpha * gAlpha, pack.hAB * invF - gAlpha * gBeta,
                               pack.hANcp * invF - gAlpha * gNcp, pack.hAx * invF - gAlpha * gx,
                               pack.hAB * invF - gBeta * gAlpha, pack.hBB * invF - gBeta * gBeta,
                               pack.hBNcp * invF - gBeta * gNcp, pack.hBx * invF - gBeta * gx,
                               pack.hANcp * invF - gNcp * gAlpha, pack.hBNcp * invF - gNcp * gBeta,
                               pack.hNcpNcp * invF - gNcp * gNcp, pack.hNcpX * invF - gNcp * gx,
                               pack.hAx * invF - gx * gAlpha, pack.hBx * invF - gx * gBeta,
                               pack.hNcpX * invF - gx * gNcp, pack.hxx * invF - gx * gx});
}

/// `ccdf = 1 - cdf`: negate every `NoncentralF*GradHess` field.
template <class Pack>
inline stan::math::var ncfCcdfVjp(const Pack& pack, double value, stan::math::var nu1,
                                  stan::math::var nu2, stan::math::var ncp, stan::math::var y) {
    return makeVjpFromParts(value, std::array<stan::math::var, 4>{nu1, nu2, ncp, y},
                            std::array<double, 4>{-pack.gNu1, -pack.gNu2, -pack.gNcp, -pack.gy});
}

template <class Pack>
inline stan::math::fvar<stan::math::var>
ncfCcdfVhp(const Pack& pack, double value, stan::math::var nu1, stan::math::var nu2,
           stan::math::var ncp, stan::math::var y, double wNu1, double wNu2, double wNcp,
           double wy) {
    return makeVhpFromParts(
        value, std::array<stan::math::var, 4>{nu1, nu2, ncp, y},
        std::array<double, 4>{-pack.gNu1, -pack.gNu2, -pack.gNcp, -pack.gy},
        std::array<double, 4>{wNu1, wNu2, wNcp, wy},
        std::array<double, 16>{-pack.hNu1Nu1, -pack.hNu1Nu2, -pack.hNu1Ncp, -pack.hNu1Y,
                               -pack.hNu1Nu2, -pack.hNu2Nu2, -pack.hNu2Ncp, -pack.hNu2Y,
                               -pack.hNu1Ncp, -pack.hNu2Ncp, -pack.hNcpNcp, -pack.hNcpY,
                               -pack.hNu1Y, -pack.hNu2Y, -pack.hNcpY, -pack.hyy});
}

/// `log(pdf)` on the noncentral-F density pack; same contract as
/// `ncbLogPdfVjp`.
template <class Pack>
inline stan::math::var ncfLogPdfVjp(const Pack& pack, double value, stan::math::var nu1,
                                    stan::math::var nu2, stan::math::var ncp, stan::math::var y) {
    if (pack.value == 0.0) {
        return makeVjpFromParts(value, std::array<stan::math::var, 4>{nu1, nu2, ncp, y},
                                std::array<double, 4>{0.0, 0.0, 0.0, 0.0});
    }
    const double invF = 1.0 / pack.value;
    return makeVjpFromParts(value, std::array<stan::math::var, 4>{nu1, nu2, ncp, y},
                            std::array<double, 4>{pack.gNu1 * invF, pack.gNu2 * invF,
                                                  pack.gNcp * invF, pack.gy * invF});
}

template <class Pack>
inline stan::math::fvar<stan::math::var>
ncfLogPdfVhp(const Pack& pack, double value, stan::math::var nu1, stan::math::var nu2,
             stan::math::var ncp, stan::math::var y, double wNu1, double wNu2, double wNcp,
             double wy) {
    const std::array<stan::math::var, 4> leaves{nu1, nu2, ncp, y};
    const std::array<double, 4> weights{wNu1, wNu2, wNcp, wy};
    if (pack.value == 0.0) {
        return makeVhpFromParts(value, leaves, std::array<double, 4>{0.0, 0.0, 0.0, 0.0}, weights,
                                std::array<double, 16>{});
    }
    const double invF = 1.0 / pack.value;
    const double gNu1 = pack.gNu1 * invF;
    const double gNu2 = pack.gNu2 * invF;
    const double gNcp = pack.gNcp * invF;
    const double gy = pack.gy * invF;
    return makeVhpFromParts(
        value, leaves, std::array<double, 4>{gNu1, gNu2, gNcp, gy}, weights,
        std::array<double, 16>{pack.hNu1Nu1 * invF - gNu1 * gNu1, pack.hNu1Nu2 * invF - gNu1 * gNu2,
                               pack.hNu1Ncp * invF - gNu1 * gNcp, pack.hNu1Y * invF - gNu1 * gy,
                               pack.hNu1Nu2 * invF - gNu2 * gNu1, pack.hNu2Nu2 * invF - gNu2 * gNu2,
                               pack.hNu2Ncp * invF - gNu2 * gNcp, pack.hNu2Y * invF - gNu2 * gy,
                               pack.hNu1Ncp * invF - gNcp * gNu1, pack.hNu2Ncp * invF - gNcp * gNu2,
                               pack.hNcpNcp * invF - gNcp * gNcp, pack.hNcpY * invF - gNcp * gy,
                               pack.hNu1Y * invF - gy * gNu1, pack.hNu2Y * invF - gy * gNu2,
                               pack.hNcpY * invF - gy * gNcp, pack.hyy * invF - gy * gy});
}

} // namespace detail

// ============================================================================
// Normal<var>::cdf -- analytical (x, mu, sigma) partials, 1 tape node
// ============================================================================

template <>
inline stan::math::var Normal<stan::math::var>::cdf(stan::math::var x) const {
    const auto pack = detail::normalCdfGradHess(x.val(), mu.val(), sigma.val());
    return detail::makeVjpVar(pack.value, [mu = mu, sigma = sigma, x, gx = pack.gx, gmu = pack.gmu,
                                           gsig = pack.gsig](double adj) {
        x.adj() += adj * gx;
        mu.adj() += adj * gmu;
        sigma.adj() += adj * gsig;
    });
}

// ============================================================================
// Normal<var>::pdf -- analytical (x, mu, sigma) partials, 1 tape node
//
// g_x and the value come from the pdf pack; g_mu = z phi/sigma and
// g_sigma = (z^2 - 1) phi/sigma are the closed-form parameter derivatives.
// ============================================================================

template <>
inline stan::math::var Normal<stan::math::var>::pdf(stan::math::var x) const {
    const double x0 = x.val(), mu0 = mu.val(), sigma0 = sigma.val();
    const double z = (x0 - mu0) / sigma0;
    const auto pack = detail::normalPdfD1D2(x0, mu0, sigma0);
    const double gx = pack.d1;
    const double gmu = -pack.d1;
    const double gsig = pack.value * (z * z - 1.0) / sigma0;
    return detail::makeVjpVar(pack.value, [mu = mu, sigma = sigma, x, gx, gmu, gsig](double adj) {
        x.adj() += adj * gx;
        mu.adj() += adj * gmu;
        sigma.adj() += adj * gsig;
    });
}

// ============================================================================
// Normal<var>::logCdf -- pdf/cdf ratios, 1 tape node
//
// Value is log(cdf) exactly as the generic normalLogCdfT path; the ratios are
// the exact derivatives of that value, so no branch/approximation is
// introduced.
// ============================================================================

template <>
inline stan::math::var Normal<stan::math::var>::logCdf(stan::math::var x) const {
    const double x0 = x.val(), mu0 = mu.val(), sigma0 = sigma.val();
    const double z = (x0 - mu0) / sigma0;
    const double cdf = detail::normalCdfDouble(x0, mu0, sigma0);
    const double pdf = detail::normalPdfDouble(x0, mu0, sigma0);
    const double gx = pdf / cdf;
    const double gmu = -gx;
    const double gsig = -gx * z;
    return detail::makeVjpVar(std::log(cdf),
                              [mu = mu, sigma = sigma, x, gx, gmu, gsig](double adj) {
                                  x.adj() += adj * gx;
                                  mu.adj() += adj * gmu;
                                  sigma.adj() += adj * gsig;
                              });
}

// ============================================================================
// Normal<var>::ccdf -- sign-flipped cdf partials, 1 tape node
//
// The value uses normalCcdfDouble (the direct erfc form of the generic
// normalCcdfT, not 1 - cdf) so it matches the generic path bit for bit.
// ============================================================================

template <>
inline stan::math::var Normal<stan::math::var>::ccdf(stan::math::var x) const {
    const double x0 = x.val(), mu0 = mu.val(), sigma0 = sigma.val();
    const auto pack = detail::normalCdfGradHess(x0, mu0, sigma0);
    return detail::makeVjpVar(
        detail::normalCcdfDouble(x0, mu0, sigma0),
        [mu = mu, sigma = sigma, x, gx = -pack.gx, gmu = -pack.gmu, gsig = -pack.gsig](double adj) {
            x.adj() += adj * gx;
            mu.adj() += adj * gmu;
            sigma.adj() += adj * gsig;
        });
}

// ============================================================================
// Normal<var>::logCcdf -- pdf/ccdf ratios, 1 tape node
//
// Mirror of the logCdf route with G = ccdf and gradG = -(cdf gradients):
// value is log(ccdf) exactly as the generic normalLogCcdfT path (no
// asymptotic branch), so the far-tail underflow behaviour is unchanged.
// ============================================================================

template <>
inline stan::math::var Normal<stan::math::var>::logCcdf(stan::math::var x) const {
    const double x0 = x.val(), mu0 = mu.val(), sigma0 = sigma.val();
    const double z = (x0 - mu0) / sigma0;
    const double ccdf = detail::normalCcdfDouble(x0, mu0, sigma0);
    const double pdf = detail::normalPdfDouble(x0, mu0, sigma0);
    const double gx = -pdf / ccdf;
    const double gmu = pdf / ccdf;
    const double gsig = z * pdf / ccdf;
    return detail::makeVjpVar(std::log(ccdf),
                              [mu = mu, sigma = sigma, x, gx, gmu, gsig](double adj) {
                                  x.adj() += adj * gx;
                                  mu.adj() += adj * gmu;
                                  sigma.adj() += adj * gsig;
                              });
}

// ============================================================================
// BivariateNormal<var>::cdf -- analytical (a, b, rho) gradient, 1 tape node
//
// The pack needs finite a, b and |rho| < 1; the non-smooth edges fall back to
// the generic quadrature path so the exact inf / |rho| = 1 limits and NaN
// propagation are preserved (at the cost of taping them, as before).
// ============================================================================

template <>
inline stan::math::var BivariateNormal<stan::math::var>::cdf(stan::math::var a,
                                                             stan::math::var b) const {
    const double a0 = a.val(), b0 = b.val(), rho0 = rho.val();
    if (!std::isfinite(a0) || !std::isfinite(b0) || !std::isfinite(rho0) || std::abs(rho0) >= 1.0) {
        return detail::bvnCdf(a, b, rho);
    }

    const auto pack = detail::bvnCdfGradHess(a0, b0, rho0);
    return detail::makeVjpVar(
        pack.value, [rho = rho, a, b, ga = pack.ga, gb = pack.gb, gr = pack.gr](double adj) {
            a.adj() += adj * ga;
            b.adj() += adj * gb;
            rho.adj() += adj * gr;
        });
}

// ============================================================================
// Poisson<var>::pmf / cdf -- analytical lambda partials, 1 tape node each
//
// Only lambda is active. lambda <= 0 (including NaN) falls back to the
// generic path: the generic kernel returns the exact point-mass constants at
// lambda = 0 / NaN with zero lambda sensitivity, while the double pack reports
// the one-sided limits at the point mass.
// ============================================================================

template <>
inline stan::math::var Poisson<stan::math::var>::pmf(int k) const {
    const double lambda0 = lambda.val();
    if (!(lambda0 > 0.0)) {
        return detail::poissonPmfT<stan::math::var>(lambda, k);
    }
    const auto pack = detail::poissonPmfD1D2(lambda0, k);
    return detail::makeVjpVar(
        pack.value, [lambda = lambda, d1 = pack.d1](double adj) { lambda.adj() += adj * d1; });
}

template <>
inline stan::math::var Poisson<stan::math::var>::cdf(int k) const {
    const double lambda0 = lambda.val();
    if (!(lambda0 > 0.0)) {
        return detail::poissonCdfT<stan::math::var>(lambda, k);
    }
    const auto pack = detail::poissonCdfD1D2(lambda0, k);
    return detail::makeVjpVar(
        pack.value, [lambda = lambda, d1 = pack.d1](double adj) { lambda.adj() += adj * d1; });
}

// ============================================================================
// Gamma<var>::cdf / pdf -- analytic (shape, rate, x) packs, 1 tape node each
//
// Invalid/NaN/non-finite shape-rate, NaN x and the x = 0 boundary (pack
// all-NaN) fall back to the generic scalar kernel; x outside the support and
// the saturating x = +Inf limit take the pack, whose value and partials are
// the generic ones there.
// ============================================================================

template <>
inline stan::math::var Gamma<stan::math::var>::cdf(stan::math::var x) const {
    const double shape0 = shape.val();
    const double rate0 = rate.val();
    const double x0 = x.val();
    if (!(shape0 > 0.0) || !(rate0 > 0.0) || !std::isfinite(shape0) || !std::isfinite(rate0) ||
        x0 != x0 || x0 == 0.0) {
        return detail::gammaCdfT<stan::math::var>(shape, rate, x);
    }
    const auto pack = detail::gammaCdfGradHess(shape0, rate0, x0);
    return detail::gammaGradHessVjp(pack, shape, rate, x);
}

template <>
inline stan::math::var Gamma<stan::math::var>::pdf(stan::math::var x) const {
    const double shape0 = shape.val();
    const double rate0 = rate.val();
    const double x0 = x.val();
    if (!(shape0 > 0.0) || !(rate0 > 0.0) || !std::isfinite(shape0) || !std::isfinite(rate0) ||
        x0 != x0 || x0 == 0.0) {
        return detail::gammaPdfT<stan::math::var>(shape, rate, x);
    }
    const auto pack = detail::gammaPdfGradHess(shape0, rate0, x0);
    return detail::gammaGradHessVjp(pack, shape, rate, x);
}

// ============================================================================
// InverseGamma<var>::cdf / pdf -- analytic (shape, scale, x) packs, 1 node
//
// Same fallback contract as Gamma: the pack is all-NaN at x = 0 while the
// generic path keeps the finite one-sided values/derivatives.
// ============================================================================

template <>
inline stan::math::var InverseGamma<stan::math::var>::cdf(stan::math::var x) const {
    const double shape0 = shape.val();
    const double scale0 = scale.val();
    const double x0 = x.val();
    if (!(shape0 > 0.0) || !(scale0 > 0.0) || !std::isfinite(shape0) || !std::isfinite(scale0) ||
        x0 != x0 || x0 == 0.0) {
        return detail::invGammaCdfT<stan::math::var>(shape, scale, x);
    }
    const auto pack = detail::inverseGammaCdfGradHess(shape0, scale0, x0);
    return detail::invGammaGradHessVjp(pack, shape, scale, x);
}

template <>
inline stan::math::var InverseGamma<stan::math::var>::pdf(stan::math::var x) const {
    const double shape0 = shape.val();
    const double scale0 = scale.val();
    const double x0 = x.val();
    if (!(shape0 > 0.0) || !(scale0 > 0.0) || !std::isfinite(shape0) || !std::isfinite(scale0) ||
        x0 != x0 || x0 == 0.0) {
        return detail::invGammaPdfT<stan::math::var>(shape, scale, x);
    }
    const auto pack = detail::inverseGammaPdfGradHess(shape0, scale0, x0);
    return detail::invGammaGradHessVjp(pack, shape, scale, x);
}

// ============================================================================
// ChiSquare<var>::cdf / pdf -- analytic (df, x) packs, 1 tape node each
//
// The generic ChiSquare path is Gamma(df/2, 1/2); the fallback reproduces it
// with the generic Gamma kernel (the ChiSquare packs are Gamma's at those
// parameters).
// ============================================================================

template <>
inline stan::math::var ChiSquare<stan::math::var>::cdf(stan::math::var x) const {
    const double df0 = df.val();
    const double x0 = x.val();
    if (!(df0 > 0.0) || !std::isfinite(df0) || x0 != x0 || x0 == 0.0) {
        return detail::gammaCdfT<stan::math::var>(df / stan::math::var(2.0), stan::math::var(0.5),
                                                  x);
    }
    const auto pack = detail::chiSquareCdfGradHess(df0, x0);
    return detail::chiSquareGradHessVjp(pack, df, x);
}

template <>
inline stan::math::var ChiSquare<stan::math::var>::pdf(stan::math::var x) const {
    const double df0 = df.val();
    const double x0 = x.val();
    if (!(df0 > 0.0) || !std::isfinite(df0) || x0 != x0 || x0 == 0.0) {
        return detail::gammaPdfT<stan::math::var>(df / stan::math::var(2.0), stan::math::var(0.5),
                                                  x);
    }
    const auto pack = detail::chiSquarePdfGradHess(df0, x0);
    return detail::chiSquareGradHessVjp(pack, df, x);
}

// ============================================================================
// Beta<var>::cdf / pdf -- analytic (alpha, beta, x) packs, 1 tape node each
//
// Invalid alpha/beta, NaN x and the singular edges x = 0 / x = 1 (the pack's
// gx/hxx are one-sided limits there, while the generic path is a constant
// branch) fall back to the generic incBeta/betaLogPdf kernels.
// ============================================================================

template <>
inline stan::math::var Beta<stan::math::var>::cdf(stan::math::var x) const {
    const double alpha0 = alpha.val();
    const double beta0 = beta.val();
    const double x0 = x.val();
    if (!(alpha0 > 0.0) || !(beta0 > 0.0) || !std::isfinite(alpha0) || !std::isfinite(beta0) ||
        x0 != x0 || x0 == 0.0 || x0 == 1.0) {
        return detail::incBeta(alpha, beta, x);
    }
    const auto pack = detail::betaCdfGradHess(alpha0, beta0, x0);
    return detail::betaGradHessVjp(pack, alpha, beta, x);
}

template <>
inline stan::math::var Beta<stan::math::var>::pdf(stan::math::var x) const {
    using std::exp;
    const double alpha0 = alpha.val();
    const double beta0 = beta.val();
    const double x0 = x.val();
    if (!(alpha0 > 0.0) || !(beta0 > 0.0) || !std::isfinite(alpha0) || !std::isfinite(beta0) ||
        x0 != x0 || x0 == 0.0 || x0 == 1.0) {
        return exp(detail::betaLogPdf(alpha, beta, x));
    }
    const auto pack = detail::betaPdfGradHess(alpha0, beta0, x0);
    return detail::betaGradHessVjp(pack, alpha, beta, x);
}

// ============================================================================
// StudentT<var>::cdf / pdf -- analytic (nu, x) packs, 1 tape node each
//
// Invalid nu, NaN/non-finite nu and NaN x fall back. The cdf also falls back
// at x = 0: the pack is the right-derivative limit (gx = pdf(0)) while the
// generic path keeps the symmetric zero derivative at the median. The density
// is smooth at x = 0, so only nu/x validity gates it.
// ============================================================================

template <>
inline stan::math::var StudentT<stan::math::var>::cdf(stan::math::var x) const {
    const double nu0 = nu.val();
    const double x0 = x.val();
    if (!(nu0 > 0.0) || !std::isfinite(nu0) || x0 != x0 || x0 == 0.0) {
        return detail::studentTCdf(nu, x);
    }
    const auto pack = detail::studentTCdfGradHess(nu0, x0);
    return detail::studentTGradHessVjp(pack, nu, x);
}

template <>
inline stan::math::var StudentT<stan::math::var>::pdf(stan::math::var x) const {
    using std::exp;
    const double nu0 = nu.val();
    const double x0 = x.val();
    if (!(nu0 > 0.0) || !std::isfinite(nu0) || x0 != x0) {
        return exp(detail::studentTLogPdf(nu, x));
    }
    const auto pack = detail::studentTPdfGradHess(nu0, x0);
    return detail::studentTGradHessVjp(pack, nu, x);
}

// ============================================================================
// Lognormal<var>::cdf / pdf -- analytic (mu, sigma, x) packs, 1 node each
//
// Invalid/non-finite mu-sigma and NaN x fall back; x <= 0 and x = +Inf take
// the pack (the zero / saturating-limits packs match the generic constants).
// ============================================================================

template <>
inline stan::math::var Lognormal<stan::math::var>::cdf(stan::math::var x) const {
    const double mu0 = mu.val();
    const double sigma0 = sigma.val();
    const double x0 = x.val();
    if (!(sigma0 > 0.0) || !std::isfinite(sigma0) || !std::isfinite(mu0) || x0 != x0) {
        return detail::lognormalCdfD1D2(x, mu, sigma).value;
    }
    const auto pack = detail::lognormalCdfGradHess(mu0, sigma0, x0);
    return detail::lognormalGradHessVjp(pack, mu, sigma, x);
}

template <>
inline stan::math::var Lognormal<stan::math::var>::pdf(stan::math::var x) const {
    const double mu0 = mu.val();
    const double sigma0 = sigma.val();
    const double x0 = x.val();
    if (!(sigma0 > 0.0) || !std::isfinite(sigma0) || !std::isfinite(mu0) || x0 != x0) {
        return detail::lognormalPdfD1D2(x, mu, sigma).value;
    }
    const auto pack = detail::lognormalPdfGradHess(mu0, sigma0, x0);
    return detail::lognormalGradHessVjp(pack, mu, sigma, x);
}

// ============================================================================
// Exponential<var>::cdf / pdf -- analytic (lambda, x) packs, 1 tape node each
//
// The packs reproduce the generic branches at x < 0 / x = 0 / x = +Inf, so
// only invalid/non-finite lambda and NaN x fall back.
// ============================================================================

template <>
inline stan::math::var Exponential<stan::math::var>::cdf(stan::math::var x) const {
    using std::expm1;
    const double lambda0 = lambda.val();
    const double x0 = x.val();
    if (!(lambda0 > 0.0) || !std::isfinite(lambda0) || x0 != x0) {
        if (x < 0.0) {
            return stan::math::var(0.0);
        }
        return -expm1(-lambda * x);
    }
    const auto pack = detail::exponentialCdfGradHess(lambda0, x0);
    return detail::exponentialGradHessVjp(pack, lambda, x);
}

template <>
inline stan::math::var Exponential<stan::math::var>::pdf(stan::math::var x) const {
    const double lambda0 = lambda.val();
    const double x0 = x.val();
    if (!(lambda0 > 0.0) || !std::isfinite(lambda0) || x0 != x0) {
        return detail::exponentialPdfD1D2(x, lambda).value;
    }
    const auto pack = detail::exponentialPdfGradHess(lambda0, x0);
    return detail::exponentialGradHessVjp(pack, lambda, x);
}

// ============================================================================
// InverseGaussian<var>::cdf / pdf -- analytic (mu, lambda, x) packs, 1 node
//
// Invalid/non-finite mu-lambda and NaN x fall back; the x <= 0 zero pack and
// x = +Inf saturating packs match the generic constants (see InverseGaussian.h
// for the documented exp(2 lambda/mu) overflow regime).
// ============================================================================

template <>
inline stan::math::var InverseGaussian<stan::math::var>::cdf(stan::math::var x) const {
    const double mu0 = mu.val();
    const double lambda0 = lambda.val();
    const double x0 = x.val();
    if (!(mu0 > 0.0) || !(lambda0 > 0.0) || !std::isfinite(mu0) || !std::isfinite(lambda0) ||
        x0 != x0) {
        return detail::inverseGaussianCdfD1D2(x, mu, lambda).value;
    }
    const auto pack = detail::inverseGaussianCdfGradHess(mu0, lambda0, x0);
    return detail::inverseGaussianGradHessVjp(pack, mu, lambda, x);
}

template <>
inline stan::math::var InverseGaussian<stan::math::var>::pdf(stan::math::var x) const {
    const double mu0 = mu.val();
    const double lambda0 = lambda.val();
    const double x0 = x.val();
    if (!(mu0 > 0.0) || !(lambda0 > 0.0) || !std::isfinite(mu0) || !std::isfinite(lambda0) ||
        x0 != x0) {
        return detail::inverseGaussianPdfD1D2(x, mu, lambda).value;
    }
    const auto pack = detail::inverseGaussianPdfGradHess(mu0, lambda0, x0);
    return detail::inverseGaussianGradHessVjp(pack, mu, lambda, x);
}

// ============================================================================
// NoncentralChiSquare<var>::cdf -- analytic (df, ncp, x) pack, 1 tape node
//
// The CDF pack covers every active-parameter partial (df = nu, ncp = lambda).
// Invalid/non-finite df-ncp, NaN x and the x = 0 pack singularity (all-NaN; the
// generic tapped mixture returns 0 with zero adjoints there) fall back to
// `detail::nccsCdfGeneric`; x < 0 (zero pack) and x = +Inf (saturating pack)
// take the pack, matching the generic constants there. `pdf`/`logPdf` use the
// density pack (section below); `ccdf`/`logCcdf` have their own routes below.
// ============================================================================

template <>
inline stan::math::var NoncentralChiSquare<stan::math::var>::cdf(stan::math::var x) const {
    using var = stan::math::var;
    const double df0 = df.val();
    const double ncp0 = ncp.val();
    const double x0 = x.val();
    if (!(df0 > 0.0) || !(ncp0 >= 0.0) || !std::isfinite(df0) || !std::isfinite(ncp0) || x0 != x0 ||
        x0 == 0.0) {
        return detail::nccsCdfGeneric<var>(df, ncp, x);
    }
    const auto pack = detail::noncentralChiSquareCdfGradHess(df0, ncp0, x0);
    return detail::nccsGradHessVjp(pack, df, ncp, x);
}

// ============================================================================
// NoncentralChiSquare<var>::ccdf -- sign-flipped cdf pack, 1 tape node
//
// Guard and fallback are exactly the cdf route's (invalid/non-finite df-ncp,
// NaN x and the x = 0 pack all-NaN boundary take `detail::nccsCcdfGeneric`,
// the generic tapped mixture; x = 0 returns the mixture's ~1 with zero
// adjoints). Everywhere else the cdf pack is used with all fields negated;
// x < 0 (zero pack) -> value 1 with zero adjoints, x = +Inf (saturating pack)
// -> value 0 with zero adjoints. The node value is the double accessor
// `noncentralChiSquareCcdfDouble`.
// ============================================================================

template <>
inline stan::math::var NoncentralChiSquare<stan::math::var>::ccdf(stan::math::var x) const {
    using var = stan::math::var;
    const double df0 = df.val();
    const double ncp0 = ncp.val();
    const double x0 = x.val();
    if (!(df0 > 0.0) || !(ncp0 >= 0.0) || !std::isfinite(df0) || !std::isfinite(ncp0) || x0 != x0 ||
        x0 == 0.0) {
        return detail::nccsCcdfGeneric<var>(df, ncp, x);
    }
    const auto pack = detail::noncentralChiSquareCdfGradHess(df0, ncp0, x0);
    return detail::nccsCcdfVjp(pack, detail::noncentralChiSquareCcdfDouble(df0, ncp0, x0), df, ncp,
                               x);
}

// ============================================================================
// NoncentralChiSquare<var>::logCcdf -- quotient rule on the ccdf pack, 1 node
//
// With G = 1 - F, g = -F_grad and H = -F_Hess:
// grad = g/G and Hess_log = H/G - g g^T/G^2.
// The value is the log-space double accessor `noncentralChiSquareLogCcdfDouble`,
// never log of the double ccdf: the accessor stays accurate where the double
// ccdf underflows to zero. There is no generic scalar logCcdf body to fall
// back to, so the boundaries the pack cannot represent take the accessor value
// with zero pushes: invalid/non-finite df-ncp and NaN x (NaN value), the
// x = 0 constant branch (value 0, all partials zero) and the G == 0 underflow
// branch (deep tails and x = +Inf, where the quotient cannot be formed; the
// value there is the finite log-space accessor, -Inf at x = +Inf).
// ============================================================================

template <>
inline stan::math::var NoncentralChiSquare<stan::math::var>::logCcdf(stan::math::var x) const {
    const double df0 = df.val();
    const double ncp0 = ncp.val();
    const double x0 = x.val();
    const double value = detail::noncentralChiSquareLogCcdfDouble(df0, ncp0, x0);
    if (!(df0 > 0.0) || !(ncp0 >= 0.0) || !std::isfinite(df0) || !std::isfinite(ncp0) || x0 != x0 ||
        x0 == 0.0) {
        return detail::makeVjpVar(value, [](double) {});
    }
    const double ccdf = detail::noncentralChiSquareCcdfDouble(df0, ncp0, x0);
    if (ccdf == 0.0) {
        return detail::makeVjpVar(value, [](double) {});
    }
    const auto pack = detail::noncentralChiSquareCdfGradHess(df0, ncp0, x0);
    return detail::nccsLogCcdfVjp(pack, value, ccdf, df, ncp, x);
}

// ============================================================================
// NoncentralChiSquare<var/fvar<var>>::pdf / logPdf -- mixture density pack
//
// The Poisson-mixture density pack carries (df, ncp, x) first/second
// partials. `logPdf` applies `grad = f_i/f` and `H = f_ij/f - f_i f_j/f^2`.
// Boundaries (invalid/non-finite df-ncp, NaN x, x <= 0) push zero adjoints
// with the double accessor value; the pack's all-NaN corners become a NaN
// value node with zero pushes. The generic scalar `pdf`/`logPdf` accessors
// are double-only, so this specialization is the only AD route.
// ============================================================================

template <>
inline stan::math::var NoncentralChiSquare<stan::math::var>::pdf(stan::math::var x) const {
    using var = stan::math::var;
    const double df0 = df.val();
    const double ncp0 = ncp.val();
    const double x0 = x.val();
    if (!(df0 > 0.0) || !(ncp0 >= 0.0) || !std::isfinite(df0) || !std::isfinite(ncp0) ||
        std::isnan(x0) || x0 <= 0.0) {
        return detail::makeVjpVar(detail::noncentralChiSquarePdfDouble(df0, ncp0, x0),
                                  [](double) {});
    }
    const auto pack = detail::noncentralChiSquarePdfGradHess(df0, ncp0, x0);
    if (std::isnan(pack.value)) {
        return detail::makeVjpVar(pack.value, [](double) {});
    }
    return detail::nccsGradHessVjp(pack, df, ncp, x);
}

template <>
inline stan::math::fvar<stan::math::var>
NoncentralChiSquare<stan::math::fvar<stan::math::var>>::pdf(
    stan::math::fvar<stan::math::var> x) const {
    const double df0 = df.val_.val();
    const double ncp0 = ncp.val_.val();
    const double x0 = x.val_.val();
    if (!(df0 > 0.0) || !(ncp0 >= 0.0) || !std::isfinite(df0) || !std::isfinite(ncp0) ||
        std::isnan(x0) || x0 <= 0.0) {
        return detail::makeVhpFvar(
            detail::noncentralChiSquarePdfDouble(df0, ncp0, x0), 0.0, [](double) {}, [](double) {});
    }
    const auto pack = detail::noncentralChiSquarePdfGradHess(df0, ncp0, x0);
    if (std::isnan(pack.value)) {
        return detail::makeVhpFvar(pack.value, 0.0, [](double) {}, [](double) {});
    }
    return detail::nccsGradHessVhp(pack, df.val_, ncp.val_, x.val_, df.d_.val(), ncp.d_.val(),
                                   x.d_.val());
}

template <>
inline stan::math::var NoncentralChiSquare<stan::math::var>::logPdf(stan::math::var x) const {
    using var = stan::math::var;
    const double df0 = df.val();
    const double ncp0 = ncp.val();
    const double x0 = x.val();
    const double kNaN = std::numeric_limits<double>::quiet_NaN();
    if (!(df0 > 0.0) || !(ncp0 >= 0.0) || !std::isfinite(df0) || !std::isfinite(ncp0) ||
        std::isnan(x0) || x0 <= 0.0) {
        const double value = detail::noncentralChiSquarePdfDouble(df0, ncp0, x0);
        return detail::makeVjpVar(value > 0.0 ? std::log(value) : kNaN, [](double) {});
    }
    const auto pack = detail::noncentralChiSquarePdfGradHess(df0, ncp0, x0);
    if (!(pack.value > 0.0)) {
        return detail::makeVjpVar(kNaN, [](double) {});
    }
    const double inv = 1.0 / pack.value;
    return detail::makeVjpVar(std::log(pack.value),
                              [dfv = df, ncpv = ncp, xv = x, inv, gNu = pack.gNu,
                               gLambda = pack.gLambda, gx = pack.gx](double adj) {
                                  dfv.adj() += adj * gNu * inv;
                                  ncpv.adj() += adj * gLambda * inv;
                                  xv.adj() += adj * gx * inv;
                              });
}

template <>
inline stan::math::fvar<stan::math::var>
NoncentralChiSquare<stan::math::fvar<stan::math::var>>::logPdf(
    stan::math::fvar<stan::math::var> x) const {
    const double df0 = df.val_.val();
    const double ncp0 = ncp.val_.val();
    const double x0 = x.val_.val();
    const double kNaN = std::numeric_limits<double>::quiet_NaN();
    if (!(df0 > 0.0) || !(ncp0 >= 0.0) || !std::isfinite(df0) || !std::isfinite(ncp0) ||
        std::isnan(x0) || x0 <= 0.0) {
        const double value = detail::noncentralChiSquarePdfDouble(df0, ncp0, x0);
        return detail::makeVhpFvar(
            value > 0.0 ? std::log(value) : kNaN, 0.0, [](double) {}, [](double) {});
    }
    const auto pack = detail::noncentralChiSquarePdfGradHess(df0, ncp0, x0);
    if (!(pack.value > 0.0)) {
        return detail::makeVhpFvar(kNaN, 0.0, [](double) {}, [](double) {});
    }
    const double inv = 1.0 / pack.value;
    const std::array<double, 3> g = {pack.gNu * inv, pack.gLambda * inv, pack.gx * inv};
    const double hv[3][3] = {{pack.hNuNu, pack.hNuLambda, pack.hNuX},
                             {pack.hNuLambda, pack.hLambdaLambda, pack.hLambdaX},
                             {pack.hNuX, pack.hLambdaX, pack.hxx}};
    const std::array<double, 3> w = {df.d_.val(), ncp.d_.val(), x.d_.val()};
    const double tangent = g[0] * w[0] + g[1] * w[1] + g[2] * w[2];
    return detail::makeVhpFvar(
        std::log(pack.value), tangent,
        [dfv = df.val_, ncpv = ncp.val_, xv = x.val_, g](double adj) {
            dfv.adj() += adj * g[0];
            ncpv.adj() += adj * g[1];
            xv.adj() += adj * g[2];
        },
        [dfv = df.val_, ncpv = ncp.val_, xv = x.val_, g, hv, inv, w](double adj) {
            const std::array<double, 3> hl = {
                (hv[0][0] * inv - g[0] * g[0]) * w[0] + (hv[0][1] * inv - g[0] * g[1]) * w[1] +
                    (hv[0][2] * inv - g[0] * g[2]) * w[2],
                (hv[1][0] * inv - g[1] * g[0]) * w[0] + (hv[1][1] * inv - g[1] * g[1]) * w[1] +
                    (hv[1][2] * inv - g[1] * g[2]) * w[2],
                (hv[2][0] * inv - g[2] * g[0]) * w[0] + (hv[2][1] * inv - g[2] * g[1]) * w[1] +
                    (hv[2][2] * inv - g[2] * g[2]) * w[2]};
            dfv.adj() += adj * hl[0];
            ncpv.adj() += adj * hl[1];
            xv.adj() += adj * hl[2];
        });
}

// ============================================================================
// NoncentralBeta<var>::cdf / pdf / ccdf / logPdf -- analytic
// (alpha, beta, ncp, x) packs, 1 tape node each
//
// The noncentral-beta accessors are double-only in the header (no generic
// scalar mixture body), so every edge takes the pack: invalid parameters ->
// all-NaN, x < 0 / x > 1 -> zero packs, x = 0 / 1 -> the one-sided closed
// forms with zero mixed parameter-x entries (see NoncentralBeta.h).
// `ccdf` negates the cdf pack; `logPdf` applies grad = f_i/f and
// Hess_log = f_ij/f - f_i f_j/f^2 to the density pack, with the pushes zeroed
// where the double density is exactly zero (support edge / underflow), since
// the quotient is undefined there and the log-space accessor can still be
// finite. The node values are the matching double accessors
// (`noncentralBetaCcdfDouble` / `noncentralBetaLogPdfDouble`).
// ============================================================================

template <>
inline stan::math::var NoncentralBeta<stan::math::var>::cdf(stan::math::var x) const {
    const auto pack =
        detail::noncentralBetaCdfGradHess(alpha.val(), beta.val(), ncp.val(), x.val());
    return detail::ncbGradHessVjp(pack, alpha, beta, ncp, x);
}

template <>
inline stan::math::var NoncentralBeta<stan::math::var>::pdf(stan::math::var x) const {
    const auto pack =
        detail::noncentralBetaPdfGradHess(alpha.val(), beta.val(), ncp.val(), x.val());
    return detail::ncbGradHessVjp(pack, alpha, beta, ncp, x);
}

template <>
inline stan::math::var NoncentralBeta<stan::math::var>::ccdf(stan::math::var x) const {
    const auto pack =
        detail::noncentralBetaCdfGradHess(alpha.val(), beta.val(), ncp.val(), x.val());
    return detail::ncbCcdfVjp(
        pack, detail::noncentralBetaCcdfDouble(alpha.val(), beta.val(), ncp.val(), x.val()), alpha,
        beta, ncp, x);
}

template <>
inline stan::math::var NoncentralBeta<stan::math::var>::logPdf(stan::math::var x) const {
    const auto pack =
        detail::noncentralBetaPdfGradHess(alpha.val(), beta.val(), ncp.val(), x.val());
    return detail::ncbLogPdfVjp(
        pack, detail::noncentralBetaLogPdfDouble(alpha.val(), beta.val(), ncp.val(), x.val()),
        alpha, beta, ncp, x);
}

// ============================================================================
// NoncentralF<var>::cdf / pdf / ccdf / logPdf -- analytic
// (nu1, nu2, ncp, y) packs, 1 node each
//
// Same edge contract as NoncentralBeta (the F packs chain the beta packs
// through the x = nu1 y/(nu2 + nu1 y) transform); invalid parameters -> all-NaN,
// y < 0 / y = +Inf -> zero packs. `ccdf` negates the cdf pack; `logPdf` uses
// `f_i/f` and `f_ij/f - f_i f_j/f^2` on the density pack with the zero-density
// guard, and the node values are `noncentralFCcdfDouble` /
// `noncentralFLogPdfDouble`.
// ============================================================================

template <>
inline stan::math::var NoncentralF<stan::math::var>::cdf(stan::math::var y) const {
    const auto pack = detail::noncentralFCdfGradHess(nu1.val(), nu2.val(), ncp.val(), y.val());
    return detail::ncfGradHessVjp(pack, nu1, nu2, ncp, y);
}

template <>
inline stan::math::var NoncentralF<stan::math::var>::pdf(stan::math::var y) const {
    const auto pack = detail::noncentralFPdfGradHess(nu1.val(), nu2.val(), ncp.val(), y.val());
    return detail::ncfGradHessVjp(pack, nu1, nu2, ncp, y);
}

template <>
inline stan::math::var NoncentralF<stan::math::var>::ccdf(stan::math::var y) const {
    const auto pack = detail::noncentralFCdfGradHess(nu1.val(), nu2.val(), ncp.val(), y.val());
    return detail::ncfCcdfVjp(
        pack, detail::noncentralFCcdfDouble(nu1.val(), nu2.val(), ncp.val(), y.val()), nu1, nu2,
        ncp, y);
}

template <>
inline stan::math::var NoncentralF<stan::math::var>::logPdf(stan::math::var y) const {
    const auto pack = detail::noncentralFPdfGradHess(nu1.val(), nu2.val(), ncp.val(), y.val());
    return detail::ncfLogPdfVjp(
        pack, detail::noncentralFLogPdfDouble(nu1.val(), nu2.val(), ncp.val(), y.val()), nu1, nu2,
        ncp, y);
}

// ============================================================================
// Second-order (`fvar<var>`) shared builders
//
// Both builders return `fvar<var>(valueNode, tangentNode)`: the value node is
// the exact `var`-route callback (value + analytic gradient), the tangent node
// carries the directional derivative g . w as its value and pushes `a H w`
// into the (by-value captured) leaf `var`s on the reverse pass. Two tape nodes
// per call; the leaf handles are `.val_` of the incoming `fvar<var>`s.
// ============================================================================

namespace detail {

/// Two-callback `fvar<var>` result from a Normal gradient/Hessian pack (the
/// `NormalCdfGradHess`/`NormalPdfGradHess` 10-field shape).
template <class Pack>
inline stan::math::fvar<stan::math::var>
normalGradHessFvar(const Pack& pack, stan::math::var x, stan::math::var mu, stan::math::var sigma,
                   double wx, double wmu, double wsigma) {
    const double tangent = pack.gx * wx + pack.gmu * wmu + pack.gsig * wsigma;
    return makeVhpFvar(
        pack.value, tangent,
        [x, mu, sigma, gx = pack.gx, gmu = pack.gmu, gsig = pack.gsig](double adj) {
            x.adj() += adj * gx;
            mu.adj() += adj * gmu;
            sigma.adj() += adj * gsig;
        },
        [x, mu, sigma, wx, wmu, wsigma, hxx = pack.hxx, hxmu = pack.hxmu, hxsig = pack.hxsig,
         hmumu = pack.hmumu, hmusig = pack.hmusig, hsigsig = pack.hsigsig](double adj) {
            x.adj() += adj * (hxx * wx + hxmu * wmu + hxsig * wsigma);
            mu.adj() += adj * (hxmu * wx + hmumu * wmu + hmusig * wsigma);
            sigma.adj() += adj * (hxsig * wx + hmusig * wmu + hsigsig * wsigma);
        });
}

/// Two-callback `fvar<var>` result for the Poisson pmf/cdf packs (lambda only:
/// the tangent row collapses to `d2 * w`).
inline stan::math::fvar<stan::math::var> poissonGradHessFvar(const ValueD1D2& pack,
                                                             stan::math::var lambda, double w) {
    return makeVhpFvar(
        pack.value, pack.d1 * w, [lambda, d1 = pack.d1](double adj) { lambda.adj() += adj * d1; },
        [lambda, w, d2 = pack.d2](double adj) { lambda.adj() += adj * d2 * w; });
}

} // namespace detail

// ============================================================================
// Normal<fvar<var>>::cdf -- analytic (x, mu, sigma) Hessian, 2 tape nodes
// ============================================================================

template <>
inline stan::math::fvar<stan::math::var>
Normal<stan::math::fvar<stan::math::var>>::cdf(stan::math::fvar<stan::math::var> x) const {
    const stan::math::var xv = x.val_;
    const stan::math::var muv = mu.val_;
    const stan::math::var sigmav = sigma.val();
    const auto pack = detail::normalCdfGradHess(xv.val(), muv.val(), sigmav.val());
    return detail::normalGradHessFvar(pack, xv, muv, sigmav, x.d_.val(), mu.d_.val(),
                                      sigma.d_.val());
}

// ============================================================================
// Normal<fvar<var>>::pdf -- analytic (x, mu, sigma) Hessian, 2 tape nodes
// ============================================================================

template <>
inline stan::math::fvar<stan::math::var>
Normal<stan::math::fvar<stan::math::var>>::pdf(stan::math::fvar<stan::math::var> x) const {
    const stan::math::var xv = x.val_;
    const stan::math::var muv = mu.val_;
    const stan::math::var sigmav = sigma.val();
    const auto pack = detail::normalPdfGradHess(xv.val(), muv.val(), sigmav.val());
    return detail::normalGradHessFvar(pack, xv, muv, sigmav, x.d_.val(), mu.d_.val(),
                                      sigma.d_.val());
}

// ============================================================================
// Normal<fvar<var>>::logCdf -- quotient rule on the cdf pack, 2 tape nodes
//
// Value is log(cdf) exactly as the generic path. cdf == 0 (deep-tail
// underflow) falls back to the generic tape: the quotient rule divides by F.
// ============================================================================

template <>
inline stan::math::fvar<stan::math::var>
Normal<stan::math::fvar<stan::math::var>>::logCdf(stan::math::fvar<stan::math::var> x) const {
    using fvar = stan::math::fvar<stan::math::var>;
    using var = stan::math::var;

    const double x0 = x.val_.val(), mu0 = mu.val_.val(), sigma0 = sigma.val_.val();
    const double cdf = detail::normalCdfDouble(x0, mu0, sigma0);
    if (cdf == 0.0) {
        return detail::normalLogCdfT<fvar>(x, mu, sigma);
    }
    const var xv = x.val_, muv = mu.val_, sigmav = sigma.val_;
    const auto cdfPack = detail::normalCdfGradHess(x0, mu0, sigma0);
    const double invF = 1.0 / cdf;
    detail::NormalCdfGradHess pack{};
    pack.value = std::log(cdf);
    pack.gx = cdfPack.gx * invF;
    pack.gmu = cdfPack.gmu * invF;
    pack.gsig = cdfPack.gsig * invF;
    pack.hxx = cdfPack.hxx * invF - pack.gx * pack.gx;
    pack.hxmu = cdfPack.hxmu * invF - pack.gx * pack.gmu;
    pack.hxsig = cdfPack.hxsig * invF - pack.gx * pack.gsig;
    pack.hmumu = cdfPack.hmumu * invF - pack.gmu * pack.gmu;
    pack.hmusig = cdfPack.hmusig * invF - pack.gmu * pack.gsig;
    pack.hsigsig = cdfPack.hsigsig * invF - pack.gsig * pack.gsig;
    return detail::normalGradHessFvar(pack, xv, muv, sigmav, x.d_.val(), mu.d_.val(),
                                      sigma.d_.val());
}

// ============================================================================
// Normal<fvar<var>>::ccdf -- sign-flipped cdf Hessian pack, 2 tape nodes
//
// The value uses normalCcdfDouble (the direct erfc form of the generic
// normalCcdfT, not 1 - cdf) so it matches the generic path bit for bit.
// ============================================================================

template <>
inline stan::math::fvar<stan::math::var>
Normal<stan::math::fvar<stan::math::var>>::ccdf(stan::math::fvar<stan::math::var> x) const {
    const stan::math::var xv = x.val_;
    const stan::math::var muv = mu.val_;
    const stan::math::var sigmav = sigma.val();
    const auto cdfPack = detail::normalCdfGradHess(xv.val(), muv.val(), sigmav.val());
    detail::NormalCdfGradHess pack = cdfPack;
    pack.value = detail::normalCcdfDouble(xv.val(), muv.val(), sigmav.val());
    pack.gx = -cdfPack.gx;
    pack.gmu = -cdfPack.gmu;
    pack.gsig = -cdfPack.gsig;
    pack.hxx = -cdfPack.hxx;
    pack.hxmu = -cdfPack.hxmu;
    pack.hxsig = -cdfPack.hxsig;
    pack.hmumu = -cdfPack.hmumu;
    pack.hmusig = -cdfPack.hmusig;
    pack.hsigsig = -cdfPack.hsigsig;
    return detail::normalGradHessFvar(pack, xv, muv, sigmav, x.d_.val(), mu.d_.val(),
                                      sigma.d_.val());
}

// ============================================================================
// Normal<fvar<var>>::logCcdf -- quotient rule on the ccdf pack, 2 tape nodes
//
// Mirror of logCdf with G = ccdf, gradG = -g, HessG = -H; ccdf == 0 falls
// back to the generic tape.
// ============================================================================

template <>
inline stan::math::fvar<stan::math::var>
Normal<stan::math::fvar<stan::math::var>>::logCcdf(stan::math::fvar<stan::math::var> x) const {
    using fvar = stan::math::fvar<stan::math::var>;
    using var = stan::math::var;

    const double x0 = x.val_.val(), mu0 = mu.val_.val(), sigma0 = sigma.val_.val();
    const double ccdf = detail::normalCcdfDouble(x0, mu0, sigma0);
    if (ccdf == 0.0) {
        return detail::normalLogCcdfT<fvar>(x, mu, sigma);
    }
    const var xv = x.val_, muv = mu.val_, sigmav = sigma.val_;
    const auto cdfPack = detail::normalCdfGradHess(x0, mu0, sigma0);
    const double invG = 1.0 / ccdf;
    detail::NormalCdfGradHess pack{};
    pack.value = std::log(ccdf);
    pack.gx = -cdfPack.gx * invG;
    pack.gmu = -cdfPack.gmu * invG;
    pack.gsig = -cdfPack.gsig * invG;
    pack.hxx = -cdfPack.hxx * invG - pack.gx * pack.gx;
    pack.hxmu = -cdfPack.hxmu * invG - pack.gx * pack.gmu;
    pack.hxsig = -cdfPack.hxsig * invG - pack.gx * pack.gsig;
    pack.hmumu = -cdfPack.hmumu * invG - pack.gmu * pack.gmu;
    pack.hmusig = -cdfPack.hmusig * invG - pack.gmu * pack.gsig;
    pack.hsigsig = -cdfPack.hsigsig * invG - pack.gsig * pack.gsig;
    return detail::normalGradHessFvar(pack, xv, muv, sigmav, x.d_.val(), mu.d_.val(),
                                      sigma.d_.val());
}

// ============================================================================
// BivariateNormal<fvar<var>>::cdf -- analytic (a, b, rho) Hessian, 2 tape nodes
//
// Same edge fallback as the var tier: non-finite a, b or rho, or |rho| >= 1,
// runs the generic quadrature path with the fvar<var> arguments.
// ============================================================================

template <>
inline stan::math::fvar<stan::math::var>
BivariateNormal<stan::math::fvar<stan::math::var>>::cdf(stan::math::fvar<stan::math::var> a,
                                                        stan::math::fvar<stan::math::var> b) const {
    const double a0 = a.val_.val(), b0 = b.val_.val(), rho0 = rho.val_.val();
    if (!std::isfinite(a0) || !std::isfinite(b0) || !std::isfinite(rho0) || std::abs(rho0) >= 1.0) {
        return detail::bvnCdf(a, b, rho);
    }

    const auto pack = detail::bvnCdfGradHess(a0, b0, rho0);
    const stan::math::var av = a.val_, bv = b.val_, rhov = rho.val_;
    const double wa = a.d_.val(), wb = b.d_.val(), wr = rho.d_.val();
    const double tangent = pack.ga * wa + pack.gb * wb + pack.gr * wr;

    return detail::makeVhpFvar(
        pack.value, tangent,
        [av, bv, rhov, ga = pack.ga, gb = pack.gb, gr = pack.gr](double adj) {
            av.adj() += adj * ga;
            bv.adj() += adj * gb;
            rhov.adj() += adj * gr;
        },
        [av, bv, rhov, wa, wb, wr, haa = pack.haa, hab = pack.hab, har = pack.har, hbb = pack.hbb,
         hbr = pack.hbr, hrr = pack.hrr](double adj) {
            av.adj() += adj * (haa * wa + hab * wb + har * wr);
            bv.adj() += adj * (hab * wa + hbb * wb + hbr * wr);
            rhov.adj() += adj * (har * wa + hbr * wb + hrr * wr);
        });
}

// ============================================================================
// Poisson<fvar<var>>::pmf / cdf -- analytic lambda Hessian, 2 tape nodes each
//
// Same lambda <= 0 / NaN fallback as the var tier (the generic path keeps the
// exact point-mass constants and zero lambda sensitivity).
// ============================================================================

template <>
inline stan::math::fvar<stan::math::var>
Poisson<stan::math::fvar<stan::math::var>>::pmf(int k) const {
    using fvar = stan::math::fvar<stan::math::var>;

    const double lambda0 = lambda.val_.val();
    if (!(lambda0 > 0.0)) {
        return detail::poissonPmfT<fvar>(lambda, k);
    }
    const auto pack = detail::poissonPmfD1D2(lambda0, k);
    return detail::poissonGradHessFvar(pack, lambda.val_, lambda.d_.val());
}

template <>
inline stan::math::fvar<stan::math::var>
Poisson<stan::math::fvar<stan::math::var>>::cdf(int k) const {
    using fvar = stan::math::fvar<stan::math::var>;

    const double lambda0 = lambda.val_.val();
    if (!(lambda0 > 0.0)) {
        return detail::poissonCdfT<fvar>(lambda, k);
    }
    const auto pack = detail::poissonCdfD1D2(lambda0, k);
    return detail::poissonGradHessFvar(pack, lambda.val_, lambda.d_.val());
}

// ============================================================================
// Family `fvar<var>` routes -- analytic Hessian packs, 2 tape nodes per call
//
// Same guards and fallbacks as the var tier, evaluated at `.val_.val()`; the
// tangent weights are the `.d_.val()` of the incoming fvar parameters. The
// value node mirrors the var route and the tangent node pushes H . w, so
// `stan::math::hessian` reads the analytic Hessian without taping any kernel.
// ============================================================================

template <>
inline stan::math::fvar<stan::math::var>
Gamma<stan::math::fvar<stan::math::var>>::cdf(stan::math::fvar<stan::math::var> x) const {
    using fvar = stan::math::fvar<stan::math::var>;
    const double shape0 = shape.val_.val();
    const double rate0 = rate.val_.val();
    const double x0 = x.val_.val();
    if (!(shape0 > 0.0) || !(rate0 > 0.0) || !std::isfinite(shape0) || !std::isfinite(rate0) ||
        x0 != x0 || x0 == 0.0) {
        return detail::gammaCdfT<fvar>(shape, rate, x);
    }
    const auto pack = detail::gammaCdfGradHess(shape0, rate0, x0);
    return detail::gammaGradHessVhp(pack, shape.val_, rate.val_, x.val_, shape.d_.val(),
                                    rate.d_.val(), x.d_.val());
}

template <>
inline stan::math::fvar<stan::math::var>
Gamma<stan::math::fvar<stan::math::var>>::pdf(stan::math::fvar<stan::math::var> x) const {
    using fvar = stan::math::fvar<stan::math::var>;
    const double shape0 = shape.val_.val();
    const double rate0 = rate.val_.val();
    const double x0 = x.val_.val();
    if (!(shape0 > 0.0) || !(rate0 > 0.0) || !std::isfinite(shape0) || !std::isfinite(rate0) ||
        x0 != x0 || x0 == 0.0) {
        return detail::gammaPdfT<fvar>(shape, rate, x);
    }
    const auto pack = detail::gammaPdfGradHess(shape0, rate0, x0);
    return detail::gammaGradHessVhp(pack, shape.val_, rate.val_, x.val_, shape.d_.val(),
                                    rate.d_.val(), x.d_.val());
}

template <>
inline stan::math::fvar<stan::math::var>
InverseGamma<stan::math::fvar<stan::math::var>>::cdf(stan::math::fvar<stan::math::var> x) const {
    using fvar = stan::math::fvar<stan::math::var>;
    const double shape0 = shape.val_.val();
    const double scale0 = scale.val_.val();
    const double x0 = x.val_.val();
    if (!(shape0 > 0.0) || !(scale0 > 0.0) || !std::isfinite(shape0) || !std::isfinite(scale0) ||
        x0 != x0 || x0 == 0.0) {
        return detail::invGammaCdfT<fvar>(shape, scale, x);
    }
    const auto pack = detail::inverseGammaCdfGradHess(shape0, scale0, x0);
    return detail::invGammaGradHessVhp(pack, shape.val_, scale.val_, x.val_, shape.d_.val(),
                                       scale.d_.val(), x.d_.val());
}

template <>
inline stan::math::fvar<stan::math::var>
InverseGamma<stan::math::fvar<stan::math::var>>::pdf(stan::math::fvar<stan::math::var> x) const {
    using fvar = stan::math::fvar<stan::math::var>;
    const double shape0 = shape.val_.val();
    const double scale0 = scale.val_.val();
    const double x0 = x.val_.val();
    if (!(shape0 > 0.0) || !(scale0 > 0.0) || !std::isfinite(shape0) || !std::isfinite(scale0) ||
        x0 != x0 || x0 == 0.0) {
        return detail::invGammaPdfT<fvar>(shape, scale, x);
    }
    const auto pack = detail::inverseGammaPdfGradHess(shape0, scale0, x0);
    return detail::invGammaGradHessVhp(pack, shape.val_, scale.val_, x.val_, shape.d_.val(),
                                       scale.d_.val(), x.d_.val());
}

template <>
inline stan::math::fvar<stan::math::var>
ChiSquare<stan::math::fvar<stan::math::var>>::cdf(stan::math::fvar<stan::math::var> x) const {
    using fvar = stan::math::fvar<stan::math::var>;
    const double df0 = df.val_.val();
    const double x0 = x.val_.val();
    if (!(df0 > 0.0) || !std::isfinite(df0) || x0 != x0 || x0 == 0.0) {
        return detail::gammaCdfT<fvar>(df / fvar(2.0), fvar(0.5), x);
    }
    const auto pack = detail::chiSquareCdfGradHess(df0, x0);
    return detail::chiSquareGradHessVhp(pack, df.val_, x.val_, df.d_.val(), x.d_.val());
}

template <>
inline stan::math::fvar<stan::math::var>
ChiSquare<stan::math::fvar<stan::math::var>>::pdf(stan::math::fvar<stan::math::var> x) const {
    using fvar = stan::math::fvar<stan::math::var>;
    const double df0 = df.val_.val();
    const double x0 = x.val_.val();
    if (!(df0 > 0.0) || !std::isfinite(df0) || x0 != x0 || x0 == 0.0) {
        return detail::gammaPdfT<fvar>(df / fvar(2.0), fvar(0.5), x);
    }
    const auto pack = detail::chiSquarePdfGradHess(df0, x0);
    return detail::chiSquareGradHessVhp(pack, df.val_, x.val_, df.d_.val(), x.d_.val());
}

template <>
inline stan::math::fvar<stan::math::var>
Beta<stan::math::fvar<stan::math::var>>::cdf(stan::math::fvar<stan::math::var> x) const {
    using fvar = stan::math::fvar<stan::math::var>;
    const double alpha0 = alpha.val_.val();
    const double beta0 = beta.val_.val();
    const double x0 = x.val_.val();
    if (!(alpha0 > 0.0) || !(beta0 > 0.0) || !std::isfinite(alpha0) || !std::isfinite(beta0) ||
        x0 != x0 || x0 == 0.0 || x0 == 1.0) {
        return detail::incBeta<fvar>(alpha, beta, x);
    }
    const auto pack = detail::betaCdfGradHess(alpha0, beta0, x0);
    return detail::betaGradHessVhp(pack, alpha.val_, beta.val_, x.val_, alpha.d_.val(),
                                   beta.d_.val(), x.d_.val());
}

template <>
inline stan::math::fvar<stan::math::var>
Beta<stan::math::fvar<stan::math::var>>::pdf(stan::math::fvar<stan::math::var> x) const {
    using std::exp;
    const double alpha0 = alpha.val_.val();
    const double beta0 = beta.val_.val();
    const double x0 = x.val_.val();
    if (!(alpha0 > 0.0) || !(beta0 > 0.0) || !std::isfinite(alpha0) || !std::isfinite(beta0) ||
        x0 != x0 || x0 == 0.0 || x0 == 1.0) {
        return exp(detail::betaLogPdf(alpha, beta, x));
    }
    const auto pack = detail::betaPdfGradHess(alpha0, beta0, x0);
    return detail::betaGradHessVhp(pack, alpha.val_, beta.val_, x.val_, alpha.d_.val(),
                                   beta.d_.val(), x.d_.val());
}

template <>
inline stan::math::fvar<stan::math::var>
StudentT<stan::math::fvar<stan::math::var>>::cdf(stan::math::fvar<stan::math::var> x) const {
    const double nu0 = nu.val_.val();
    const double x0 = x.val_.val();
    if (!(nu0 > 0.0) || !std::isfinite(nu0) || x0 != x0 || x0 == 0.0) {
        return detail::studentTCdf(nu, x);
    }
    const auto pack = detail::studentTCdfGradHess(nu0, x0);
    return detail::studentTGradHessVhp(pack, nu.val_, x.val_, nu.d_.val(), x.d_.val());
}

template <>
inline stan::math::fvar<stan::math::var>
StudentT<stan::math::fvar<stan::math::var>>::pdf(stan::math::fvar<stan::math::var> x) const {
    using std::exp;
    const double nu0 = nu.val_.val();
    const double x0 = x.val_.val();
    if (!(nu0 > 0.0) || !std::isfinite(nu0) || x0 != x0) {
        return exp(detail::studentTLogPdf(nu, x));
    }
    const auto pack = detail::studentTPdfGradHess(nu0, x0);
    return detail::studentTGradHessVhp(pack, nu.val_, x.val_, nu.d_.val(), x.d_.val());
}

template <>
inline stan::math::fvar<stan::math::var>
Lognormal<stan::math::fvar<stan::math::var>>::cdf(stan::math::fvar<stan::math::var> x) const {
    const double mu0 = mu.val_.val();
    const double sigma0 = sigma.val_.val();
    const double x0 = x.val_.val();
    if (!(sigma0 > 0.0) || !std::isfinite(sigma0) || !std::isfinite(mu0) || x0 != x0) {
        return detail::lognormalCdfD1D2(x, mu, sigma).value;
    }
    const auto pack = detail::lognormalCdfGradHess(mu0, sigma0, x0);
    return detail::lognormalGradHessVhp(pack, mu.val_, sigma.val_, x.val_, mu.d_.val(),
                                        sigma.d_.val(), x.d_.val());
}

template <>
inline stan::math::fvar<stan::math::var>
Lognormal<stan::math::fvar<stan::math::var>>::pdf(stan::math::fvar<stan::math::var> x) const {
    const double mu0 = mu.val_.val();
    const double sigma0 = sigma.val_.val();
    const double x0 = x.val_.val();
    if (!(sigma0 > 0.0) || !std::isfinite(sigma0) || !std::isfinite(mu0) || x0 != x0) {
        return detail::lognormalPdfD1D2(x, mu, sigma).value;
    }
    const auto pack = detail::lognormalPdfGradHess(mu0, sigma0, x0);
    return detail::lognormalGradHessVhp(pack, mu.val_, sigma.val_, x.val_, mu.d_.val(),
                                        sigma.d_.val(), x.d_.val());
}

template <>
inline stan::math::fvar<stan::math::var>
Exponential<stan::math::fvar<stan::math::var>>::cdf(stan::math::fvar<stan::math::var> x) const {
    using fvar = stan::math::fvar<stan::math::var>;
    using std::expm1;
    const double lambda0 = lambda.val_.val();
    const double x0 = x.val_.val();
    if (!(lambda0 > 0.0) || !std::isfinite(lambda0) || x0 != x0) {
        if (x < fvar(0.0)) {
            return fvar(0.0);
        }
        return -expm1(-lambda * x);
    }
    const auto pack = detail::exponentialCdfGradHess(lambda0, x0);
    return detail::exponentialGradHessVhp(pack, lambda.val_, x.val_, lambda.d_.val(), x.d_.val());
}

template <>
inline stan::math::fvar<stan::math::var>
Exponential<stan::math::fvar<stan::math::var>>::pdf(stan::math::fvar<stan::math::var> x) const {
    const double lambda0 = lambda.val_.val();
    const double x0 = x.val_.val();
    if (!(lambda0 > 0.0) || !std::isfinite(lambda0) || x0 != x0) {
        return detail::exponentialPdfD1D2(x, lambda).value;
    }
    const auto pack = detail::exponentialPdfGradHess(lambda0, x0);
    return detail::exponentialGradHessVhp(pack, lambda.val_, x.val_, lambda.d_.val(), x.d_.val());
}

template <>
inline stan::math::fvar<stan::math::var>
InverseGaussian<stan::math::fvar<stan::math::var>>::cdf(stan::math::fvar<stan::math::var> x) const {
    const double mu0 = mu.val_.val();
    const double lambda0 = lambda.val_.val();
    const double x0 = x.val_.val();
    if (!(mu0 > 0.0) || !(lambda0 > 0.0) || !std::isfinite(mu0) || !std::isfinite(lambda0) ||
        x0 != x0) {
        return detail::inverseGaussianCdfD1D2(x, mu, lambda).value;
    }
    const auto pack = detail::inverseGaussianCdfGradHess(mu0, lambda0, x0);
    return detail::inverseGaussianGradHessVhp(pack, mu.val_, lambda.val_, x.val_, mu.d_.val(),
                                              lambda.d_.val(), x.d_.val());
}

template <>
inline stan::math::fvar<stan::math::var>
InverseGaussian<stan::math::fvar<stan::math::var>>::pdf(stan::math::fvar<stan::math::var> x) const {
    const double mu0 = mu.val_.val();
    const double lambda0 = lambda.val_.val();
    const double x0 = x.val_.val();
    if (!(mu0 > 0.0) || !(lambda0 > 0.0) || !std::isfinite(mu0) || !std::isfinite(lambda0) ||
        x0 != x0) {
        return detail::inverseGaussianPdfD1D2(x, mu, lambda).value;
    }
    const auto pack = detail::inverseGaussianPdfGradHess(mu0, lambda0, x0);
    return detail::inverseGaussianGradHessVhp(pack, mu.val_, lambda.val_, x.val_, mu.d_.val(),
                                              lambda.d_.val(), x.d_.val());
}

// ============================================================================
// NoncentralChiSquare<fvar<var>>::cdf -- analytic (df, ncp, x) Hessian, 2 nodes
//
// Same guard and generic fallback as the var tier, evaluated at `.val_.val()`;
// the tangent weights are the `.d_.val()` of the incoming parameters.
// ============================================================================

template <>
inline stan::math::fvar<stan::math::var>
NoncentralChiSquare<stan::math::fvar<stan::math::var>>::cdf(
    stan::math::fvar<stan::math::var> x) const {
    using fvar = stan::math::fvar<stan::math::var>;
    const double df0 = df.val_.val();
    const double ncp0 = ncp.val_.val();
    const double x0 = x.val_.val();
    if (!(df0 > 0.0) || !(ncp0 >= 0.0) || !std::isfinite(df0) || !std::isfinite(ncp0) || x0 != x0 ||
        x0 == 0.0) {
        return detail::nccsCdfGeneric<fvar>(df, ncp, x);
    }
    const auto pack = detail::noncentralChiSquareCdfGradHess(df0, ncp0, x0);
    return detail::nccsGradHessVhp(pack, df.val_, ncp.val_, x.val_, df.d_.val(), ncp.d_.val(),
                                   x.d_.val());
}

// ============================================================================
// NoncentralChiSquare<fvar<var>>::ccdf / logCcdf -- analytic tail Hessians,
// 2 tape nodes each
//
// Same guard and generic fallback as the var tier, evaluated at `.val_.val()`;
// the tangent weights are the `.d_.val()` of the incoming parameters. The
// logCcdf zero-push branches (invalid/NaN, x = 0 and ccdf underflow) carry no
// tangent content, so the shared one-node `makeVjpVar` route is returned
// wrapped as an `fvar<var>` with zero tangent.
// ============================================================================

template <>
inline stan::math::fvar<stan::math::var>
NoncentralChiSquare<stan::math::fvar<stan::math::var>>::ccdf(
    stan::math::fvar<stan::math::var> x) const {
    using fvar = stan::math::fvar<stan::math::var>;
    const double df0 = df.val_.val();
    const double ncp0 = ncp.val_.val();
    const double x0 = x.val_.val();
    if (!(df0 > 0.0) || !(ncp0 >= 0.0) || !std::isfinite(df0) || !std::isfinite(ncp0) || x0 != x0 ||
        x0 == 0.0) {
        return detail::nccsCcdfGeneric<fvar>(df, ncp, x);
    }
    const auto pack = detail::noncentralChiSquareCdfGradHess(df0, ncp0, x0);
    return detail::nccsCcdfVhp(pack, detail::noncentralChiSquareCcdfDouble(df0, ncp0, x0), df.val_,
                               ncp.val_, x.val_, df.d_.val(), ncp.d_.val(), x.d_.val());
}

template <>
inline stan::math::fvar<stan::math::var>
NoncentralChiSquare<stan::math::fvar<stan::math::var>>::logCcdf(
    stan::math::fvar<stan::math::var> x) const {
    using fvar = stan::math::fvar<stan::math::var>;
    const double df0 = df.val_.val();
    const double ncp0 = ncp.val_.val();
    const double x0 = x.val_.val();
    const double value = detail::noncentralChiSquareLogCcdfDouble(df0, ncp0, x0);
    if (!(df0 > 0.0) || !(ncp0 >= 0.0) || !std::isfinite(df0) || !std::isfinite(ncp0) || x0 != x0 ||
        x0 == 0.0) {
        return fvar(detail::makeVjpVar(value, [](double) {}));
    }
    const double ccdf = detail::noncentralChiSquareCcdfDouble(df0, ncp0, x0);
    if (ccdf == 0.0) {
        return fvar(detail::makeVjpVar(value, [](double) {}));
    }
    const auto pack = detail::noncentralChiSquareCdfGradHess(df0, ncp0, x0);
    return detail::nccsLogCcdfVhp(pack, value, ccdf, df.val_, ncp.val_, x.val_, df.d_.val(),
                                  ncp.d_.val(), x.d_.val());
}

// ============================================================================
// NoncentralBeta<fvar<var>>::cdf / pdf / ccdf / logPdf -- analytic 4x4 Hessian
// packs, 2 nodes
//
// Same pack-everywhere edge contract as the var tier (no generic scalar body);
// the logPdf zero-density guard mirrors the var tier with zero tangent content.
// ============================================================================

template <>
inline stan::math::fvar<stan::math::var>
NoncentralBeta<stan::math::fvar<stan::math::var>>::cdf(stan::math::fvar<stan::math::var> x) const {
    const auto pack = detail::noncentralBetaCdfGradHess(alpha.val_.val(), beta.val_.val(),
                                                        ncp.val_.val(), x.val_.val());
    return detail::ncbGradHessVhp(pack, alpha.val_, beta.val_, ncp.val_, x.val_, alpha.d_.val(),
                                  beta.d_.val(), ncp.d_.val(), x.d_.val());
}

template <>
inline stan::math::fvar<stan::math::var>
NoncentralBeta<stan::math::fvar<stan::math::var>>::pdf(stan::math::fvar<stan::math::var> x) const {
    const auto pack = detail::noncentralBetaPdfGradHess(alpha.val_.val(), beta.val_.val(),
                                                        ncp.val_.val(), x.val_.val());
    return detail::ncbGradHessVhp(pack, alpha.val_, beta.val_, ncp.val_, x.val_, alpha.d_.val(),
                                  beta.d_.val(), ncp.d_.val(), x.d_.val());
}

template <>
inline stan::math::fvar<stan::math::var>
NoncentralBeta<stan::math::fvar<stan::math::var>>::ccdf(stan::math::fvar<stan::math::var> x) const {
    const auto pack = detail::noncentralBetaCdfGradHess(alpha.val_.val(), beta.val_.val(),
                                                        ncp.val_.val(), x.val_.val());
    return detail::ncbCcdfVhp(pack,
                              detail::noncentralBetaCcdfDouble(alpha.val_.val(), beta.val_.val(),
                                                               ncp.val_.val(), x.val_.val()),
                              alpha.val_, beta.val_, ncp.val_, x.val_, alpha.d_.val(),
                              beta.d_.val(), ncp.d_.val(), x.d_.val());
}

template <>
inline stan::math::fvar<stan::math::var> NoncentralBeta<stan::math::fvar<stan::math::var>>::logPdf(
    stan::math::fvar<stan::math::var> x) const {
    const auto pack = detail::noncentralBetaPdfGradHess(alpha.val_.val(), beta.val_.val(),
                                                        ncp.val_.val(), x.val_.val());
    return detail::ncbLogPdfVhp(pack,
                                detail::noncentralBetaLogPdfDouble(alpha.val_.val(),
                                                                   beta.val_.val(), ncp.val_.val(),
                                                                   x.val_.val()),
                                alpha.val_, beta.val_, ncp.val_, x.val_, alpha.d_.val(),
                                beta.d_.val(), ncp.d_.val(), x.d_.val());
}

// ============================================================================
// NoncentralF<fvar<var>>::cdf / pdf / ccdf / logPdf -- analytic 4x4 Hessian
// packs, 2 nodes
//
// Same pack-everywhere edge contract as the var tier.
// ============================================================================

template <>
inline stan::math::fvar<stan::math::var>
NoncentralF<stan::math::fvar<stan::math::var>>::cdf(stan::math::fvar<stan::math::var> y) const {
    const auto pack = detail::noncentralFCdfGradHess(nu1.val_.val(), nu2.val_.val(), ncp.val_.val(),
                                                     y.val_.val());
    return detail::ncfGradHessVhp(pack, nu1.val_, nu2.val_, ncp.val_, y.val_, nu1.d_.val(),
                                  nu2.d_.val(), ncp.d_.val(), y.d_.val());
}

template <>
inline stan::math::fvar<stan::math::var>
NoncentralF<stan::math::fvar<stan::math::var>>::pdf(stan::math::fvar<stan::math::var> y) const {
    const auto pack = detail::noncentralFPdfGradHess(nu1.val_.val(), nu2.val_.val(), ncp.val_.val(),
                                                     y.val_.val());
    return detail::ncfGradHessVhp(pack, nu1.val_, nu2.val_, ncp.val_, y.val_, nu1.d_.val(),
                                  nu2.d_.val(), ncp.d_.val(), y.d_.val());
}

template <>
inline stan::math::fvar<stan::math::var>
NoncentralF<stan::math::fvar<stan::math::var>>::ccdf(stan::math::fvar<stan::math::var> y) const {
    const auto pack = detail::noncentralFCdfGradHess(nu1.val_.val(), nu2.val_.val(), ncp.val_.val(),
                                                     y.val_.val());
    return detail::ncfCcdfVhp(
        pack,
        detail::noncentralFCcdfDouble(nu1.val_.val(), nu2.val_.val(), ncp.val_.val(), y.val_.val()),
        nu1.val_, nu2.val_, ncp.val_, y.val_, nu1.d_.val(), nu2.d_.val(), ncp.d_.val(), y.d_.val());
}

template <>
inline stan::math::fvar<stan::math::var>
NoncentralF<stan::math::fvar<stan::math::var>>::logPdf(stan::math::fvar<stan::math::var> y) const {
    const auto pack = detail::noncentralFPdfGradHess(nu1.val_.val(), nu2.val_.val(), ncp.val_.val(),
                                                     y.val_.val());
    return detail::ncfLogPdfVhp(pack,
                                detail::noncentralFLogPdfDouble(nu1.val_.val(), nu2.val_.val(),
                                                                ncp.val_.val(), y.val_.val()),
                                nu1.val_, nu2.val_, ncp.val_, y.val_, nu1.d_.val(), nu2.d_.val(),
                                ncp.d_.val(), y.d_.val());
}

} // namespace quantape::math

#endif // QUANTAPE_MATH_DISTRIBUTIONS_DISTRIBUTIONS_STAN_PRIMITIVES_H
