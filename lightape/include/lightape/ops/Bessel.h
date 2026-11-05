#ifndef LIGHTAPE_OPS_BESSEL_H
#define LIGHTAPE_OPS_BESSEL_H

/**
 * @file Bessel.h
 * @brief Real-order modified Bessel functions for `double`, `RevScalar` and
 *        `Tangent`: I_nu(z), K_nu(z), logI_nu(z)
 *
 * Public API layer over the ported double kernels in
 * `lightape/base/{BesselI,BesselK}.h`:
 *
 *   doubles:      besselI, besselK, logBesselI, logBesselK,
 *   RevScalar:    one binary `External` block each (args = {nu, z}),
 *   Tangent<S,N>: the same first-derivative factors applied tangent-wise.
 *
 * Exactness contract (matches the tape's replay contract):
 * - `besselI` records `vjp + svp + vhp`, so `RevTape::hessianVector` replays
 *   the exact analytic Hessian at the primal:
 *     H_nn = besselINuDeriv2(nu,z)                          (ported kernel),
 *     g_n  = besselINuDeriv(nu,z)                           (ported kernel),
 *     g_z  = I_{nu-1} - nu I_nu / z,
 *     H_nz = besselINuDeriv(nu+1,z) + I_nu/z + nu g_n/z,
 *            the exact d/dnu of g_z after substituting the recurrence
 *            I_{nu-1} = I_{nu+1} + (2nu/z) I_nu, so the shifted-order kernel
 *            stays inside its nu > -1 domain (finite at integer orders),
 *     H_zz = I_{nu-2} - ((2nu-1)/z) I_{nu-1}
 *            + (nu^2+nu)/z^2 I_nu                           (d/dz of g_z).
 *   Both mixed and zz entries are exact derivatives of the same z-derivative
 *   recurrence and use only the ported `besselI`/`besselINuDeriv` kernels; no
 *   finite differences, no dropped terms.
 * - `logBesselI` records `vjp + svp + vhp`; the value comes from the ported
 *   log-space kernel and the partials/Hessian are the log chain rule applied
 *   to I's exact pack (`g/I`, `H/I - g g^T/I^2`). Exact wherever I is finite
 *   (near the overflow boundary `besselI` itself returns inf and the factors
 *   degrade to NaN).
 * - `besselK` records `vjp + svp + vhp` with the exact analytic Hessian at the
 *   primal (decision #26: no finite differences):
 *     H_nn = besselKNuDeriv2(nu,z)                     (exact DE integral),
 *     H_nz = besselKNuZDeriv(nu,z)                     (exact DE integral),
 *     H_zz = besselKZZDeriv(nu,z)
 *            = K + (nu^2/z^2) K - K'/z                 (modified Bessel ODE),
 *   so `hessianVector` replays the exact second order. When any pack entry is
 *   non-finite (quadrature overflow or non-convergence) the block is recorded
 *   first-order only with `makeVjpVar` and `hessianVector` returns false
 *   (fail loud, never fake).
 * - `logBesselK` records `vjp + svp + vhp` in the log-space value: the
 *   first-order factors are exact (`besselKNuLogDeriv` for order,
 *   `K'/K` for z) and the Hessian is the log chain rule on K's exact pack
 *   (`H/K - g g^T`). It reuses `besselK`'s finite-pack guard, so it degrades
 *   to a first-order block (and `hessianVector` returns false) exactly where
 *   K's Hessian pack is non-finite.
 *
 * First-order factors (all at the primal `(nu, z)`):
 *   I: g_n = besselINuDeriv, g_z = I_{nu-1} - nu I_nu/z,
 *   K: g_n = besselKNuDeriv, g_z = besselKPrime
 *      (= -(K_{nu-1}+K_{nu+1})/2 with the even-order reflection),
 *   logI: g_n = I_nu'/I, g_z = I_z/I,
 *   logK: g_n = besselKNuLogDeriv, g_z = K'/K.
 *
 * The `Tangent` overloads re-use the first-derivative factors of the primal
 * point (`detail::besselPrimalValue` strips any nesting), which is exact for
 * first-order tangents; nested `Tangent` levels treat the factors as locally
 * constant.
 */

#include <cmath>
#include <cstddef>
#include <cstdint>

#include "lightape/base/BesselI.h"
#include "lightape/base/BesselK.h"
#include "lightape/base/ScalarConstant.h"
#include "lightape/fwd/ForwardScalar.h"
#include "lightape/ops/Callbacks.h"
#include "lightape/rev/LeanReverse.h"

namespace lightape {

// ── double API ──

/// Modified Bessel function of the first kind `I_nu(z)`, `z > 0`.
inline double besselI(double nu, double z) {
    return detail::besselI(nu, z);
}

/// Modified Bessel function of the second kind `K_nu(z)`, `nu >= 0`, `z > 0`.
inline double besselK(double nu, double z) {
    return detail::besselK(nu, z);
}

/// `log(I_nu(z))` evaluated in log space (finite for large z).
inline double logBesselI(double nu, double z) {
    return detail::logBesselI(nu, z);
}

/// `log(K_nu(z))` evaluated in log space (finite where `besselK` underflows
/// for large z or overflows for small z / large order).
inline double logBesselK(double nu, double z) {
    return detail::logBesselK(nu, z);
}

namespace detail {

/// Underlying double of a scalar (strips nested `Tangent` levels); the
/// Bessel `Tangent` overloads evaluate their derivative factors here.
template <typename T>
double besselPrimalValue(const T& x) {
    if constexpr (requires { x.value; }) {
        return besselPrimalValue(x.value);
    } else {
        return static_cast<double>(x);
    }
}

/// Primal value plus first/second partials of I_nu(z); the callbacks read it
/// as an opaque payload (trivially copyable, copied into the tape).
struct BesselIPack {
    double value = 0.0;
    double gNu = 0.0;
    double gZ = 0.0;
    double hNuNu = 0.0;
    double hNuZ = 0.0;
    double hZZ = 0.0;
};

/// Exact value/gradient/Hessian pack of I_nu(z) at the primal `(nu, z)`.
/// `hNuNu` is the ported second order-derivative kernel; `hNuZ`/`hZZ` are the
/// exact derivatives of the z-derivative recurrence (see the file header).
inline BesselIPack besselIPack(double nu, double z) {
    BesselIPack p;
    p.value = besselI(nu, z);
    const double im1 = besselI(nu - 1.0, z);
    const double im2 = besselI(nu - 2.0, z);
    const double dNu = besselINuDeriv(nu, z);
    p.gNu = dNu;
    p.gZ = im1 - nu * p.value / z;
    p.hNuNu = besselINuDeriv2(nu, z);
    p.hNuZ = besselINuDeriv(nu + 1.0, z) + p.value / z + nu * dNu / z;
    p.hZZ = im2 - ((2.0 * nu - 1.0) / z) * im1 + ((nu * nu + nu) / (z * z)) * p.value;
    return p;
}

/// Same pack for logI_nu(z): ported log-space value and the log chain rule
/// applied to I's exact gradient/Hessian.
inline BesselIPack logBesselIPack(double nu, double z) {
    const BesselIPack p = besselIPack(nu, z);
    const double inv = 1.0 / p.value;
    const double gNu = p.gNu * inv;
    const double gZ = p.gZ * inv;
    BesselIPack out;
    out.value = logBesselI(nu, z);
    out.gNu = gNu;
    out.gZ = gZ;
    out.hNuNu = p.hNuNu * inv - gNu * gNu;
    out.hNuZ = p.hNuZ * inv - gNu * gZ;
    out.hZZ = p.hZZ * inv - gZ * gZ;
    return out;
}

/// First-order pack of K_nu(z) at the primal, used by the `Tangent` overload
/// (which never needs the Hessian).
struct BesselKGrad {
    double value = 0.0;
    double gNu = 0.0;
    double gZ = 0.0;
};

inline BesselKGrad besselKGrad(double nu, double z) {
    return {besselK(nu, z), besselKNuDeriv(nu, z), besselKPrime(nu, z)};
}

/// Value/gradient/Hessian pack of K_nu(z) at the primal. `hessianExact` is
/// set only when all six entries are finite; the order-derivative kernels
/// return NaN when the quadrature overflows or fails to converge, and the
/// RevScalar wrapper then records first order only (fail loud).
struct BesselKPack {
    double value = 0.0;
    double gNu = 0.0;
    double gZ = 0.0;
    double hNuNu = 0.0;
    double hNuZ = 0.0;
    double hZZ = 0.0;
    bool hessianExact = false;
};

inline bool besselKPackFinite(const BesselKPack& p) {
    return std::isfinite(p.value) && std::isfinite(p.gNu) && std::isfinite(p.gZ) &&
           std::isfinite(p.hNuNu) && std::isfinite(p.hNuZ) && std::isfinite(p.hZZ);
}

/// Exact value/gradient/Hessian pack of K_nu(z) at the primal `(nu, z)`; the
/// Hessian entries are the exact kernels of base/BesselK.h (see the file
/// header). A NaN from any kernel zeroes `hessianExact`.
inline BesselKPack besselKPack(double nu, double z) {
    BesselKPack p;
    p.value = besselK(nu, z);
    p.gNu = besselKNuDeriv(nu, z);
    p.gZ = besselKPrime(nu, z);
    p.hNuNu = besselKNuDeriv2(nu, z);
    p.hNuZ = besselKNuZDeriv(nu, z);
    p.hZZ = besselKZZDeriv(nu, z);
    p.hessianExact = besselKPackFinite(p);
    return p;
}

/// Value/gradient/Hessian pack of `log K_nu(z)` at the primal `(nu, z)`, in
/// the same layout as `BesselKPack` so the K callbacks can consume it. The
/// value is the ported log-space kernel; the first-order factors are the
/// exact `besselKNuLogDeriv` (order) and `besselKPrime/besselK` (z); the
/// Hessian is the log chain rule on K's exact pack,
///   L_nn = K_nn/K - (K_n/K)^2,
///   L_nz = K_nz/K - K_n K_z/K^2,
///   L_zz = K_zz/K - (K_z/K)^2.
/// `hessianExact` mirrors `besselK`: set only when all six entries are
/// finite, so the wrapper degrades to first order (fail loud) exactly where
/// the underlying K pack does.
inline BesselKPack logBesselKPack(double nu, double z) {
    const BesselKPack p = besselKPack(nu, z);
    BesselKPack out;
    out.value = logBesselK(nu, z);
    const double invK = 1.0 / p.value;
    out.gNu = besselKNuLogDeriv(nu, z);
    out.gZ = p.gZ * invK;
    out.hNuNu = p.hNuNu * invK - out.gNu * out.gNu;
    out.hNuZ = p.hNuZ * invK - out.gNu * out.gZ;
    out.hZZ = p.hZZ * invK - out.gZ * out.gZ;
    out.hessianExact = besselKPackFinite(out);
    return out;
}

// ── External callbacks (precomputed primal partials) ──

inline void besselPackVjp(const void* payload, double seed, const ExtArg* a, std::uint32_t /*n*/,
                          RevTape& tape) noexcept {
    const auto* p = static_cast<const BesselIPack*>(payload);
    tape.addAdjoint(a[0].node, seed * p->gNu);
    tape.addAdjoint(a[1].node, seed * p->gZ);
}

inline double besselPackSvp(const void* payload, const ExtArg* /*a*/, const double* s,
                            std::uint32_t /*n*/) noexcept {
    const auto* p = static_cast<const BesselIPack*>(payload);
    return p->gNu * s[0] + p->gZ * s[1];
}

inline void besselPackVhp(const void* payload, double adjoint, const ExtArg* /*a*/, const double* s,
                          std::uint32_t /*n*/, double* b) noexcept {
    const auto* p = static_cast<const BesselIPack*>(payload);
    b[0] += adjoint * (p->hNuNu * s[0] + p->hNuZ * s[1]);
    b[1] += adjoint * (p->hNuZ * s[0] + p->hZZ * s[1]);
}

inline void besselKVjp(const void* payload, double seed, const ExtArg* a, std::uint32_t /*n*/,
                       RevTape& tape) noexcept {
    const auto* p = static_cast<const BesselKPack*>(payload);
    tape.addAdjoint(a[0].node, seed * p->gNu);
    tape.addAdjoint(a[1].node, seed * p->gZ);
}

inline double besselKSvp(const void* payload, const ExtArg* /*a*/, const double* s,
                         std::uint32_t /*n*/) noexcept {
    const auto* p = static_cast<const BesselKPack*>(payload);
    return p->gNu * s[0] + p->gZ * s[1];
}

inline void besselKVhp(const void* payload, double adjoint, const ExtArg* /*a*/, const double* s,
                       std::uint32_t /*n*/, double* b) noexcept {
    const auto* p = static_cast<const BesselKPack*>(payload);
    b[0] += adjoint * (p->hNuNu * s[0] + p->hNuZ * s[1]);
    b[1] += adjoint * (p->hNuZ * s[0] + p->hZZ * s[1]);
}

} // namespace detail

// ── RevScalar API (one External block each) ──

/// `I_nu(z)`: one binary External block, exact full first+second order.
inline RevScalar besselI(const RevScalar& nu, const RevScalar& z) {
    const detail::BesselIPack pack = detail::besselIPack(nu.value, z.value);
    const RevScalar args[2] = {nu, z};
    return makeVhpFvar(pack.value, args, 2, detail::besselPackVjp, detail::besselPackSvp,
                       detail::besselPackVhp, pack);
}

/// `K_nu(z)`: one binary External block, exact full first+second order
/// (`besselKNuDeriv2`/`besselKNuZDeriv` integrals plus the ODE `hZZ`); falls
/// back to a first-order block with a null VHP where the pack is non-finite,
/// so `hessianVector` returns false there.
inline RevScalar besselK(const RevScalar& nu, const RevScalar& z) {
    const detail::BesselKPack pack = detail::besselKPack(nu.value, z.value);
    const RevScalar args[2] = {nu, z};
    if (pack.hessianExact) {
        return makeVhpFvar(pack.value, args, 2, detail::besselKVjp, detail::besselKSvp,
                           detail::besselKVhp, pack);
    }
    return makeVjpVar(pack.value, args, 2, detail::besselKVjp, pack);
}

/// `log K_nu(z)`: one binary External block over the log-space value and the
/// exact K pack. Full first+second order where the pack is finite; where any
/// entry is non-finite the block is first-order only with a null VHP, so
/// `hessianVector` returns false there (fail loud, never approximate).
inline RevScalar logBesselK(const RevScalar& nu, const RevScalar& z) {
    const detail::BesselKPack pack = detail::logBesselKPack(nu.value, z.value);
    const RevScalar args[2] = {nu, z};
    if (pack.hessianExact) {
        return makeVhpFvar(pack.value, args, 2, detail::besselKVjp, detail::besselKSvp,
                           detail::besselKVhp, pack);
    }
    return makeVjpVar(pack.value, args, 2, detail::besselKVjp, pack);
}

/// `logI_nu(z)`: one binary External block, exact full first+second order
/// from I's pack via the log chain rule.
inline RevScalar logBesselI(const RevScalar& nu, const RevScalar& z) {
    const detail::BesselIPack pack = detail::logBesselIPack(nu.value, z.value);
    const RevScalar args[2] = {nu, z};
    return makeVhpFvar(pack.value, args, 2, detail::besselPackVjp, detail::besselPackSvp,
                       detail::besselPackVhp, pack);
}

// ── Tangent API (forward mode) ──

/// `I_nu(z)`: first-derivative factors evaluated at the primal point.
template <typename S, std::size_t N>
Tangent<S, N> besselI(const Tangent<S, N>& nu, const Tangent<S, N>& z) {
    Tangent<S, N> out;
    out.value = besselI(nu.value, z.value);
    const detail::BesselIPack pack = detail::besselIPack(detail::besselPrimalValue(nu.value),
                                                         detail::besselPrimalValue(z.value));
    const S gNu = detail::constantOf<S>(pack.gNu);
    const S gZ = detail::constantOf<S>(pack.gZ);
    for (std::size_t i = 0; i < N; ++i) {
        out.d[i] = gNu * nu.d[i] + gZ * z.d[i];
    }
    return out;
}

/// `K_nu(z)`: first-derivative factors evaluated at the primal point.
template <typename S, std::size_t N>
Tangent<S, N> besselK(const Tangent<S, N>& nu, const Tangent<S, N>& z) {
    Tangent<S, N> out;
    out.value = besselK(nu.value, z.value);
    const detail::BesselKGrad pack = detail::besselKGrad(detail::besselPrimalValue(nu.value),
                                                         detail::besselPrimalValue(z.value));
    const S gNu = detail::constantOf<S>(pack.gNu);
    const S gZ = detail::constantOf<S>(pack.gZ);
    for (std::size_t i = 0; i < N; ++i) {
        out.d[i] = gNu * nu.d[i] + gZ * z.d[i];
    }
    return out;
}

/// `logK_nu(z)`: first-derivative factors evaluated at the primal point.
template <typename S, std::size_t N>
Tangent<S, N> logBesselK(const Tangent<S, N>& nu, const Tangent<S, N>& z) {
    Tangent<S, N> out;
    out.value = logBesselK(nu.value, z.value);
    const detail::BesselKPack pack = detail::logBesselKPack(detail::besselPrimalValue(nu.value),
                                                            detail::besselPrimalValue(z.value));
    const S gNu = detail::constantOf<S>(pack.gNu);
    const S gZ = detail::constantOf<S>(pack.gZ);
    for (std::size_t i = 0; i < N; ++i) {
        out.d[i] = gNu * nu.d[i] + gZ * z.d[i];
    }
    return out;
}

/// `logI_nu(z)`: first-derivative factors evaluated at the primal point.
template <typename S, std::size_t N>
Tangent<S, N> logBesselI(const Tangent<S, N>& nu, const Tangent<S, N>& z) {
    Tangent<S, N> out;
    out.value = logBesselI(nu.value, z.value);
    const detail::BesselIPack pack = detail::logBesselIPack(detail::besselPrimalValue(nu.value),
                                                            detail::besselPrimalValue(z.value));
    const S gNu = detail::constantOf<S>(pack.gNu);
    const S gZ = detail::constantOf<S>(pack.gZ);
    for (std::size_t i = 0; i < N; ++i) {
        out.d[i] = gNu * nu.d[i] + gZ * z.d[i];
    }
    return out;
}

} // namespace lightape

#endif // LIGHTAPE_OPS_BESSEL_H
