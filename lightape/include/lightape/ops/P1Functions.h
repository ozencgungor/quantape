#ifndef LIGHTAPE_OPS_P1_FUNCTIONS_H
#define LIGHTAPE_OPS_P1_FUNCTIONS_H

/**
 * @file P1Functions.h
 * @brief P1 special functions: normal quantile, incomplete gamma/beta and the
 *        beta function, for `double`, `RevScalar` and `Tangent`
 *
 * Public API layer over the ported double kernels in
 * `lightape/base/{NormalQuantile,IncompleteGamma,IncompleteBeta}.h`:
 *
 *   doubles:      invPhi, beta, incBeta, gammaP, gammaQ,
 *   RevScalar:    one `External` block each (first order plus exact VHP where
 *                 a closed-form/forward-jet second-order route exists),
 *   Tangent<S,N>: the same first-derivative factors applied tangent-wise.
 *
 * Exactness contract (matches the tape's replay contract):
 * - `invPhi` and `gammaP`/`gammaQ` record `vjp + svp + vhp`, so
 *   `RevTape::hessianVector` replays them exactly (analytic Hessian evaluated
 *   at the primal, not finite differences).
 * - `incBeta` records the full exact (a,b,x) gradient/Hessian from the ported
 *   three-variable second-order jet, so its replay is exact as well. Had a
 *   parameter second derivative not been available, the block would have kept
 *   `vjp + svp` only and `hessianVector` would return false (fail loud, never
 *   approximate); the generic test for that path is
 *   `CallbacksTest.MakeVjpVarHessianFailsLoudly`.
 *
 * `gammaP`/`gammaQ` partials at the primal `(a, z)`:
 *   `P_z = z^{a-1} e^{-z} / Gamma(a)`,
 *   `Q_z = -P_z`,
 *   `P_az = P_z (log z - psi(a))`, `P_aa` from the ported forward jet.
 * `incBeta` at `(a, b, x)`:
 *   `I_x = x^{a-1}(1-x)^{b-1}/B(a,b)`, parameters from the ported 3-jet.
 *
 * The `Tangent` overloads re-use the first-derivative factors of the primal
 * point (`detail::primalValue` strips any nesting), which is exact for
 * first-order tangents; nested `Tangent` levels only propagate the factors
 * that are scalar-generic (`invPhi`), the rest are locally constant.
 */

#include <cmath>
#include <cstddef>
#include <cstdint>

#include "lightape/base/IncompleteBeta.h"
#include "lightape/base/IncompleteGamma.h"
#include "lightape/base/NormalQuantile.h"
#include "lightape/base/ScalarConstant.h"
#include "lightape/fwd/ForwardScalar.h"
#include "lightape/ops/Callbacks.h"
#include "lightape/ops/Probability.h"
#include "lightape/rev/LeanReverse.h"

namespace lightape {

// ── double API ──

/// Standard normal quantile `Phi^{-1}(p)`; boundaries -> +/-inf, invalid ->
/// NaN.
inline double invPhi(double p) {
    return detail::normalInvDouble(p);
}

/// Beta function `B(a,b) = Gamma(a) Gamma(b) / Gamma(a+b)` via `lgamma`.
inline double beta(double a, double b) {
    using std::exp;
    using std::lgamma;
    return exp(lgamma(a) + lgamma(b) - lgamma(a + b));
}

/// Regularized incomplete beta `I_x(a,b)`; `a,b > 0`, `x in [0,1]`.
inline double incBeta(double a, double b, double x) {
    return detail::incBetaDouble(a, b, x);
}

/// Lower regularized incomplete gamma `P(a,z)`; `a > 0`, `z >= 0`.
inline double gammaP(double a, double z) {
    return detail::gammaPDouble(a, z);
}

/// Upper regularized incomplete gamma `Q(a,z) = 1 - P(a,z)`.
inline double gammaQ(double a, double z) {
    return detail::gammaQDouble(a, z);
}

namespace detail {

/// Underlying double of a scalar (strips nested `Tangent` levels); the P1
/// `Tangent` overloads evaluate their derivative factors at this primal point.
template <typename T>
double primalValue(const T& x) {
    if constexpr (requires { x.value; }) {
        return primalValue(x.value);
    } else {
        return static_cast<double>(x);
    }
}

// ── External payloads and callbacks (precomputed primal partials) ──

struct P1InvPhiPayload {
    double fp = 0.0;
    double fpp = 0.0;
};

inline void p1InvPhiVjp(const void* payload, double seed, const ExtArg* a, std::uint32_t /*n*/,
                        RevTape& tape) noexcept {
    const auto* p = static_cast<const P1InvPhiPayload*>(payload);
    tape.addAdjoint(a[0].node, seed * p->fp);
}

inline double p1InvPhiSvp(const void* payload, const ExtArg* /*a*/, const double* s,
                          std::uint32_t /*n*/) noexcept {
    const auto* p = static_cast<const P1InvPhiPayload*>(payload);
    return p->fp * s[0];
}

inline void p1InvPhiVhp(const void* payload, double adjoint, const ExtArg* /*a*/, const double* s,
                        std::uint32_t /*n*/, double* b) noexcept {
    const auto* p = static_cast<const P1InvPhiPayload*>(payload);
    b[0] += adjoint * p->fpp * s[0];
}

/// `sign` is +1 for `gammaP`, -1 for `gammaQ`; the derivative entries are those
/// of `P` (shaped at the primal), the value is the caller's kernel value.
struct P1GammaPayload {
    double sign = 1.0;
    double gShape = 0.0;
    double gx = 0.0;
    double hShapeShape = 0.0;
    double hShapeX = 0.0;
    double hxx = 0.0;
};

inline void p1GammaVjp(const void* payload, double seed, const ExtArg* a, std::uint32_t /*n*/,
                       RevTape& tape) noexcept {
    const auto* p = static_cast<const P1GammaPayload*>(payload);
    const double s = seed * p->sign;
    tape.addAdjoint(a[0].node, s * p->gShape);
    tape.addAdjoint(a[1].node, s * p->gx);
}

inline double p1GammaSvp(const void* payload, const ExtArg* /*a*/, const double* s,
                         std::uint32_t /*n*/) noexcept {
    const auto* p = static_cast<const P1GammaPayload*>(payload);
    return p->sign * (p->gShape * s[0] + p->gx * s[1]);
}

inline void p1GammaVhp(const void* payload, double adjoint, const ExtArg* /*a*/, const double* s,
                       std::uint32_t /*n*/, double* b) noexcept {
    const auto* p = static_cast<const P1GammaPayload*>(payload);
    const double a = adjoint * p->sign;
    b[0] += a * (p->hShapeShape * s[0] + p->hShapeX * s[1]);
    b[1] += a * (p->hShapeX * s[0] + p->hxx * s[1]);
}

struct P1BetaPayload {
    double gA = 0.0;
    double gB = 0.0;
    double gX = 0.0;
    double hAA = 0.0;
    double hAB = 0.0;
    double hAX = 0.0;
    double hBB = 0.0;
    double hBX = 0.0;
    double hXX = 0.0;
};

inline void p1BetaVjp(const void* payload, double seed, const ExtArg* a, std::uint32_t /*n*/,
                      RevTape& tape) noexcept {
    const auto* p = static_cast<const P1BetaPayload*>(payload);
    tape.addAdjoint(a[0].node, seed * p->gA);
    tape.addAdjoint(a[1].node, seed * p->gB);
    tape.addAdjoint(a[2].node, seed * p->gX);
}

inline double p1BetaSvp(const void* payload, const ExtArg* /*a*/, const double* s,
                        std::uint32_t /*n*/) noexcept {
    const auto* p = static_cast<const P1BetaPayload*>(payload);
    return p->gA * s[0] + p->gB * s[1] + p->gX * s[2];
}

inline void p1BetaVhp(const void* payload, double adjoint, const ExtArg* /*a*/, const double* s,
                      std::uint32_t /*n*/, double* b) noexcept {
    const auto* p = static_cast<const P1BetaPayload*>(payload);
    b[0] += adjoint * (p->hAA * s[0] + p->hAB * s[1] + p->hAX * s[2]);
    b[1] += adjoint * (p->hAB * s[0] + p->hBB * s[1] + p->hBX * s[2]);
    b[2] += adjoint * (p->hAX * s[0] + p->hBX * s[1] + p->hXX * s[2]);
}

} // namespace detail

// ── RevScalar API (one External block each) ──

/// `Phi^{-1}(p)`: derivative `1/phi(y)`, second derivative `y / phi(y)^2` at
/// `y = Phi^{-1}(p)`; exact replay via the precomputed factors.
inline RevScalar invPhi(const RevScalar& p) {
    const double y = detail::normalInvDouble(p.value);
    const double phi = normpdf(y);
    const detail::P1InvPhiPayload payload{1.0 / phi, y / (phi * phi)};
    const RevScalar args[1] = {p};
    return makeVhpFvar(y, args, 1, detail::p1InvPhiVjp, detail::p1InvPhiSvp, detail::p1InvPhiVhp,
                       payload);
}

/// `P(a,z)`: one External block, exact `(a,z)` gradient/Hessian.
inline RevScalar gammaP(const RevScalar& a, const RevScalar& z) {
    const detail::GammaGradHess pack = detail::gammaGradHess(a.value, z.value);
    const double value = detail::gammaPDouble(a.value, z.value);
    const detail::P1GammaPayload payload{1.0,          pack.gShape, pack.gx, pack.hShapeShape,
                                         pack.hShapeX, pack.hxx};
    const RevScalar args[2] = {a, z};
    return makeVhpFvar(value, args, 2, detail::p1GammaVjp, detail::p1GammaSvp, detail::p1GammaVhp,
                       payload);
}

/// `Q(a,z) = 1 - P(a,z)`: same partials as `gammaP`, negated.
inline RevScalar gammaQ(const RevScalar& a, const RevScalar& z) {
    const detail::GammaGradHess pack = detail::gammaGradHess(a.value, z.value);
    const double value = detail::gammaQDouble(a.value, z.value);
    const detail::P1GammaPayload payload{-1.0,         pack.gShape, pack.gx, pack.hShapeShape,
                                         pack.hShapeX, pack.hxx};
    const RevScalar args[2] = {a, z};
    return makeVhpFvar(value, args, 2, detail::p1GammaVjp, detail::p1GammaSvp, detail::p1GammaVhp,
                       payload);
}

/// `I_x(a,b)`: one 3-argument External block with the exact `(a,b,x)`
/// gradient/Hessian.
inline RevScalar incBeta(const RevScalar& a, const RevScalar& b, const RevScalar& x) {
    const detail::BetaGradHess pack = detail::betaGradHess(a.value, b.value, x.value);
    const double value = detail::incBetaDouble(a.value, b.value, x.value);
    const detail::P1BetaPayload payload{pack.gA,  pack.gB,  pack.gX,  pack.hAA, pack.hAB,
                                        pack.hAX, pack.hBB, pack.hBX, pack.hXX};
    const RevScalar args[3] = {a, b, x};
    return makeVhpFvar(value, args, 3, detail::p1BetaVjp, detail::p1BetaSvp, detail::p1BetaVhp,
                       payload);
}

// ── Tangent API (forward mode) ──

/// `Phi^{-1}`: the scale `1/phi(out)` is scalar-generic, so nested `Tangent`
/// levels differentiate it through `normpdf` naturally.
template <typename S, std::size_t N>
Tangent<S, N> invPhi(const Tangent<S, N>& p) {
    Tangent<S, N> out;
    out.value = invPhi(p.value);
    const S scale = detail::constantOf<S>(1.0) / normpdf(out.value);
    for (std::size_t i = 0; i < N; ++i) {
        out.d[i] = p.d[i] * scale;
    }
    return out;
}

/// `P(a,z)`: first-derivative factors evaluated at the primal point.
template <typename S, std::size_t N>
Tangent<S, N> gammaP(const Tangent<S, N>& a, const Tangent<S, N>& z) {
    Tangent<S, N> out;
    out.value = gammaP(a.value, z.value);
    const detail::GammaGradHess pack =
        detail::gammaGradHess(detail::primalValue(a.value), detail::primalValue(z.value));
    const S gA = detail::constantOf<S>(pack.gShape);
    const S gZ = detail::constantOf<S>(pack.gx);
    for (std::size_t i = 0; i < N; ++i) {
        out.d[i] = gA * a.d[i] + gZ * z.d[i];
    }
    return out;
}

/// `Q(a,z)`: mirror of the `gammaP` tangent overload.
template <typename S, std::size_t N>
Tangent<S, N> gammaQ(const Tangent<S, N>& a, const Tangent<S, N>& z) {
    Tangent<S, N> out;
    out.value = gammaQ(a.value, z.value);
    const detail::GammaGradHess pack =
        detail::gammaGradHess(detail::primalValue(a.value), detail::primalValue(z.value));
    const S gA = detail::constantOf<S>(-pack.gShape);
    const S gZ = detail::constantOf<S>(-pack.gx);
    for (std::size_t i = 0; i < N; ++i) {
        out.d[i] = gA * a.d[i] + gZ * z.d[i];
    }
    return out;
}

/// `I_x(a,b)`: first-derivative factors evaluated at the primal point.
template <typename S, std::size_t N>
Tangent<S, N> incBeta(const Tangent<S, N>& a, const Tangent<S, N>& b, const Tangent<S, N>& x) {
    Tangent<S, N> out;
    out.value = incBeta(a.value, b.value, x.value);
    const detail::BetaGradHess pack = detail::betaGradHess(
        detail::primalValue(a.value), detail::primalValue(b.value), detail::primalValue(x.value));
    const S gA = detail::constantOf<S>(pack.gA);
    const S gB = detail::constantOf<S>(pack.gB);
    const S gX = detail::constantOf<S>(pack.gX);
    for (std::size_t i = 0; i < N; ++i) {
        out.d[i] = gA * a.d[i] + gB * b.d[i] + gX * x.d[i];
    }
    return out;
}

} // namespace lightape

#endif // LIGHTAPE_OPS_P1_FUNCTIONS_H
