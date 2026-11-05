#ifndef LIGHTAPE_OPS_P2BESSELJY_H
#define LIGHTAPE_OPS_P2BESSELJY_H

/**
 * @file P2BesselJY.h
 * @brief Ordinary Bessel functions for `RevScalar` (one External block each)
 *        and first-order `Tangent`: J_nu(z), Y_nu(z)
 *
 * AD shape (matches the tape's replay contract, same pattern as
 * ops/Bessel.h):
 * - `besselJ` / `besselY` record **one binary `External` block** over
 *   `{nu, z}`. The `vjp` scatters the precomputed exact partials
 *   `(gNu, gZ) = (dJ/dnu, J_{nu-1} - nu J_nu/z)` and the analogous Y pair;
 *   constant `nu`/`z` slots are node-less and ignored by `addAdjoint`, so no
 *   zero seeds are ever emitted.
 * - When all six pack entries (value, gradient, Hessian) are finite, the
 *   block is recorded with `makeVhpFvar` and `hessianVector` replays the
 *   exact analytic Hessian:
 *     H_nn = d2J/dnu2                                     (kernel),
 *     H_nz = -dJ/dnu|_{nu+1} + J_nu/z + nu (dJ/dnu)/z     (d/dnu of gZ
 *            after the recurrence J_{nu-1} = (2nu/z) J_nu - J_{nu+1}),
 *     H_zz = J_{nu-2} - ((2nu-1)/z) J_{nu-1}
 *            + ((nu^2+nu)/z^2) J_nu                       (d/dz of gZ),
 *   and the identical recurrences with Y and `besselYNuDeriv`.
 * - Otherwise (non-positive integer order where the J order derivative
 *   kernel has a pole outside the integral bridge, or integer order with
 *   |nu| > 24 outside it for Y) the block is recorded first-order only with
 *   `makeVjpVar`: `hessianVector` returns false (fail loud, never fake) and
 *   the unavailable d/dnu entry stays NaN rather than being zeroed.
 *   Integer-order Y's d/dnu is finite in the bridge band and, for |nu| <= 24,
 *   also in the whole ascending-series region: the differentiated integral
 *   kernels used there are smooth in nu at integer orders, so the reflection
 *   0/0 is bypassed and the Hessian block is recorded.
 *
 * The `Tangent` overloads follow the P2/P1 pattern: the first-derivative
 * factors are evaluated at the primal point and treated as locally constant
 * for nested `Tangent` levels (exact first order, zero second order through
 * the factors; use the `RevScalar` block for the true Hessian).
 *
 * Bounds: all kernel domain/accuracy caveats of base/BesselJY.h apply,
 * including the documented near-integer Y accuracy loss of the non-integer
 * branch.
 */

#include <cmath>
#include <cstddef>
#include <cstdint>

#include "lightape/base/BesselJY.h"
#include "lightape/base/ScalarConstant.h"
#include "lightape/fwd/ForwardScalar.h"
#include "lightape/ops/Callbacks.h"
#include "lightape/rev/LeanReverse.h"

namespace lightape {
namespace detail {

/// Underlying double of a scalar (strips nested `Tangent` levels); the
/// P2BesselJY `Tangent` overloads evaluate their derivative factors here.
template <typename T>
double jyPrimalValue(const T& x) {
    if constexpr (requires { x.value; }) {
        return jyPrimalValue(x.value);
    } else {
        return static_cast<double>(x);
    }
}

/// Value plus first/second partials of an ordinary Bessel kernel. The
/// callbacks read it as an opaque trivially-copyable payload.
struct BesselJYPack {
    double value = 0.0;
    double gNu = 0.0;
    double gZ = 0.0;
    double hNuNu = 0.0;
    double hNuZ = 0.0;
    double hZZ = 0.0;
    bool hessianExact = false;
};

inline bool besselJYPackFinite(const BesselJYPack& p) {
    return std::isfinite(p.value) && std::isfinite(p.gNu) && std::isfinite(p.gZ) &&
           std::isfinite(p.hNuNu) && std::isfinite(p.hNuZ) && std::isfinite(p.hZZ);
}

/// Exact pack of J_nu(z) at the primal `(nu, z)`; see the file header for the
/// Hessian entries. `hessianExact` is false where the order-derivative
/// kernels return NaN (non-positive integer order in the series region).
inline BesselJYPack besselJPack(double nu, double z) {
    BesselJYPack p;
    p.value = besselJ(nu, z);
    p.gNu = besselJNuDeriv(nu, z);
    p.gZ = besselJ(nu - 1.0, z) - nu * p.value / z;
    p.hNuNu = besselJNuDeriv2(nu, z);
    p.hNuZ = -besselJNuDeriv(nu + 1.0, z) + p.value / z + nu * p.gNu / z;
    p.hZZ = besselJ(nu - 2.0, z) - ((2.0 * nu - 1.0) / z) * besselJ(nu - 1.0, z) +
            ((nu * nu + nu) / (z * z)) * p.value;
    p.hessianExact = besselJYPackFinite(p);
    return p;
}

/// Same pack for Y_nu(z); integer orders use the differentiated integral
/// bridge wherever it is selected (bridge band, and the series region for
/// |nu| <= 24), so `gNu` is finite and `hessianExact` stays true there.
inline BesselJYPack besselYPack(double nu, double z) {
    BesselJYPack p;
    p.value = besselY(nu, z);
    p.gNu = besselYNuDeriv(nu, z);
    p.gZ = besselY(nu - 1.0, z) - nu * p.value / z;
    p.hNuNu = besselYNuDeriv2(nu, z);
    p.hNuZ = -besselYNuDeriv(nu + 1.0, z) + p.value / z + nu * p.gNu / z;
    p.hZZ = besselY(nu - 2.0, z) - ((2.0 * nu - 1.0) / z) * besselY(nu - 1.0, z) +
            ((nu * nu + nu) / (z * z)) * p.value;
    p.hessianExact = besselJYPackFinite(p);
    return p;
}

// ── External callbacks (precomputed primal partials) ──

inline void besselJYVjp(const void* payload, double seed, const ExtArg* a, std::uint32_t /*n*/,
                        RevTape& tape) noexcept {
    const auto* p = static_cast<const BesselJYPack*>(payload);
    tape.addAdjoint(a[0].node, seed * p->gNu);
    tape.addAdjoint(a[1].node, seed * p->gZ);
}

inline double besselJYSvp(const void* payload, const ExtArg* /*a*/, const double* s,
                          std::uint32_t /*n*/) noexcept {
    const auto* p = static_cast<const BesselJYPack*>(payload);
    return p->gNu * s[0] + p->gZ * s[1];
}

inline void besselJYVhp(const void* payload, double adjoint, const ExtArg* /*a*/, const double* s,
                        std::uint32_t /*n*/, double* b) noexcept {
    const auto* p = static_cast<const BesselJYPack*>(payload);
    b[0] += adjoint * (p->hNuNu * s[0] + p->hNuZ * s[1]);
    b[1] += adjoint * (p->hNuZ * s[0] + p->hZZ * s[1]);
}

} // namespace detail

// ── RevScalar API (one External block each) ──

/// `J_nu(z)`: one binary External block; full first+second order where the
/// order-derivative kernels are finite, first order only otherwise.
inline RevScalar besselJ(const RevScalar& nu, const RevScalar& z) {
    const detail::BesselJYPack pack = detail::besselJPack(nu.value, z.value);
    const RevScalar args[2] = {nu, z};
    if (pack.hessianExact) {
        return makeVhpFvar(pack.value, args, 2, detail::besselJYVjp, detail::besselJYSvp,
                           detail::besselJYVhp, pack);
    }
    return makeVjpVar(pack.value, args, 2, detail::besselJYVjp, pack);
}

/// `Y_nu(z)`: one binary External block; full first+second order where the
/// order-derivative kernels are finite, first order only otherwise
/// (`hessianVector` then returns false).
inline RevScalar besselY(const RevScalar& nu, const RevScalar& z) {
    const detail::BesselJYPack pack = detail::besselYPack(nu.value, z.value);
    const RevScalar args[2] = {nu, z};
    if (pack.hessianExact) {
        return makeVhpFvar(pack.value, args, 2, detail::besselJYVjp, detail::besselJYSvp,
                           detail::besselJYVhp, pack);
    }
    return makeVjpVar(pack.value, args, 2, detail::besselJYVjp, pack);
}

// ── Tangent API (forward mode) ──

/// `J_nu(z)`: first-derivative factors evaluated at the primal point.
template <typename S, std::size_t N>
Tangent<S, N> besselJ(const Tangent<S, N>& nu, const Tangent<S, N>& z) {
    Tangent<S, N> out;
    out.value = besselJ(nu.value, z.value);
    const detail::BesselJYPack pack =
        detail::besselJPack(detail::jyPrimalValue(nu.value), detail::jyPrimalValue(z.value));
    const S gNu = detail::constantOf<S>(pack.gNu);
    const S gZ = detail::constantOf<S>(pack.gZ);
    for (std::size_t i = 0; i < N; ++i) {
        out.d[i] = gNu * nu.d[i] + gZ * z.d[i];
    }
    return out;
}

/// `Y_nu(z)`: first-derivative factors evaluated at the primal point.
template <typename S, std::size_t N>
Tangent<S, N> besselY(const Tangent<S, N>& nu, const Tangent<S, N>& z) {
    Tangent<S, N> out;
    out.value = besselY(nu.value, z.value);
    const detail::BesselJYPack pack =
        detail::besselYPack(detail::jyPrimalValue(nu.value), detail::jyPrimalValue(z.value));
    const S gNu = detail::constantOf<S>(pack.gNu);
    const S gZ = detail::constantOf<S>(pack.gZ);
    for (std::size_t i = 0; i < N; ++i) {
        out.d[i] = gNu * nu.d[i] + gZ * z.d[i];
    }
    return out;
}

} // namespace lightape

#endif // LIGHTAPE_OPS_P2BESSELJY_H
