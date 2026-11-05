#ifndef LIGHTAPE_OPS_BIVARIATE_NORMAL_H
#define LIGHTAPE_OPS_BIVARIATE_NORMAL_H

/**
 * @file BivariateNormal.h
 * @brief `Phi2(a,b;rho)` for `double` (base kernel), `RevScalar` and `Tangent`
 *
 * AD wrapper over the ported kernel in `lightape/base/BivariateNormal.h`:
 *
 * - `RevScalar`: ONE `External` block over the three cached parents
 *   `{a,b,rho}` recorded through `makeVhpFvar` (full first + second order, so
 *   `RevTape::hessianVector` replays it exactly). The block carries the
 *   precomputed primal partials of the true `Phi2` (conditional-normal
 *   marginalization) and their analytic second derivatives:
 *
 *     ga  = dPhi2/da = phi(a) Phi((b - rho a)/s),
 *     gb  = dPhi2/db = phi(b) Phi((a - rho b)/s),
 *     gr  = dPhi2/drho = phi2(a,b;rho),
 *     s   = sqrt(1 - rho^2),
 *
 *   with the Hessian entries `haa, hab, har, hbb, hbr, hrr` from
 *   `detail::bvnCdfGradHess` (all closed forms; the VHP is exact analytic,
 *   never finite differences and never a silent fallback). Domain: the pack
 *   is defined for finite `a,b` and `|rho| < 1`; at the exact edges
 *   (`+-inf`, `|rho| = 1`) the kernel value has its documented limits but the
 *   partials are not defined and the pack (and hence the adjoints) is NaN —
 *   fail loud. `rho = 0` stays on the smooth product branch.
 *
 * - `Tangent<S,N>`: the value is the scalar-generic kernel evaluated at `S`
 *   (so nested `Tangent` levels carry the value's own tangents) and the
 *   tangents are the same analytic factors evaluated at `S`:
 *
 *     dPhi2 = ga * da + gb * db + gr * drho.
 *
 *   Because `ga`/`gb`/`gr` are scalar-generic (not stripped to double),
 *   nesting `Tangent<Tangent<double,1>,1>` differentiates them and yields the
 *   exact second derivative of the primal-analytic factor model.
 */

#include <cmath>
#include <cstddef>
#include <cstdint>

#include "lightape/base/BivariateNormal.h"
#include "lightape/fwd/ForwardScalar.h"
#include "lightape/ops/Callbacks.h"
#include "lightape/rev/LeanReverse.h"

namespace lightape {

namespace detail {

/// Precomputed primal partials of Phi2 for one External block.
struct BvnCdfPayload {
    double ga = 0.0;
    double gb = 0.0;
    double gr = 0.0;
    double haa = 0.0;
    double hab = 0.0;
    double har = 0.0;
    double hbb = 0.0;
    double hbr = 0.0;
    double hrr = 0.0;
};

inline void bvnCdfVjp(const void* payload, double seed, const ExtArg* a, std::uint32_t /*n*/,
                      RevTape& tape) noexcept {
    const auto* p = static_cast<const BvnCdfPayload*>(payload);
    tape.addAdjoint(a[0].node, seed * p->ga);
    tape.addAdjoint(a[1].node, seed * p->gb);
    tape.addAdjoint(a[2].node, seed * p->gr);
}

inline double bvnCdfSvp(const void* payload, const ExtArg* /*a*/, const double* s,
                        std::uint32_t /*n*/) noexcept {
    const auto* p = static_cast<const BvnCdfPayload*>(payload);
    return p->ga * s[0] + p->gb * s[1] + p->gr * s[2];
}

inline void bvnCdfVhp(const void* payload, double adjoint, const ExtArg* /*a*/, const double* s,
                      std::uint32_t /*n*/, double* b) noexcept {
    const auto* p = static_cast<const BvnCdfPayload*>(payload);
    b[0] += adjoint * (p->haa * s[0] + p->hab * s[1] + p->har * s[2]);
    b[1] += adjoint * (p->hab * s[0] + p->hbb * s[1] + p->hbr * s[2]);
    b[2] += adjoint * (p->har * s[0] + p->hbr * s[1] + p->hrr * s[2]);
}

} // namespace detail

/// `Phi2(a,b;rho)`: one 3-argument External block with the exact analytic
/// gradient and Hessian of the true bivariate normal CDF.
inline RevScalar bvnCdf(const RevScalar& a, const RevScalar& b, const RevScalar& rho) {
    const detail::BvnGradHess pack = detail::bvnCdfGradHess(a.value, b.value, rho.value);
    const detail::BvnCdfPayload payload{pack.ga,  pack.gb,  pack.gr,  pack.haa, pack.hab,
                                        pack.har, pack.hbb, pack.hbr, pack.hrr};
    const RevScalar args[3] = {a, b, rho};
    return makeVhpFvar(pack.value, args, 3u, detail::bvnCdfVjp, detail::bvnCdfSvp,
                       detail::bvnCdfVhp, payload);
}

/// `Phi2(a,b;rho)` in forward mode: the scalar-generic kernel for the value
/// and the analytic factors for the directional derivatives.
template <typename S, std::size_t N>
Tangent<S, N> bvnCdf(const Tangent<S, N>& a, const Tangent<S, N>& b, const Tangent<S, N>& rho) {
    Tangent<S, N> out;
    out.value = detail::bvnCdfKernel(a.value, b.value, rho.value);
    const S ga = detail::bvnCdfPartialA(a.value, b.value, rho.value);
    const S gb = detail::bvnCdfPartialB(a.value, b.value, rho.value);
    const S gr = detail::bvnCdfPartialRho(a.value, b.value, rho.value);
    for (std::size_t i = 0; i < N; ++i) {
        out.d[i] = ga * a.d[i] + gb * b.d[i] + gr * rho.d[i];
    }
    return out;
}

} // namespace lightape

#endif // LIGHTAPE_OPS_BIVARIATE_NORMAL_H
