#ifndef LIGHTAPE_MIX_FWDREV_H
#define LIGHTAPE_MIX_FWDREV_H

#include <cstddef>
#include <cstdint>

#include "lightape/fwd/ForwardScalar.h"
#include "lightape/rev/LeanReverse.h"

namespace lightape {
/**
 * @file FwdRev.h
 * @brief fwd-over-rev mode mixing: N Hessian-vector products per recording
 *
 * `FwdRevScalar<N>` is `Tangent<RevScalar, N>`: the value and each of the `N`
 * directional tangents are ordinary `RevScalar` nodes on the shared
 * thread-local tape (the clean-room `fvar<var>` pattern; decisions #7/#15).
 * The caller records the function once with `FwdRevScalar<N>` inputs whose
 * `.value` members are tape leaves and whose `.d[k]` components carry the
 * k-th direction (plain `RevScalar` constants, i.e. node-less):
 *
 *     RevTape& tape = RevTape::active();
 *     tape.clear();
 *     FwdRevScalar<2> x(RevScalar(0.7, tape.input(0.7)), {v0x, v1x});
 *     FwdRevScalar<2> y(RevScalar(1.3, tape.input(1.3)), {v0y, v1y});
 *     const FwdRevScalar<2> f = exp(x * y) + log(x) - 3 * sin(y);
 *     // ... value: f.value.value, gradient: reverse f.value ...
 *     double out[2 * 2];
 *     reverseHvps(f, 2, inputs, out);   // out[k * 2 + i] = (H v_k)_i
 *
 * `f.d[k]` is the directional derivative of `f` along direction `k` built
 * from RevScalar arithmetic, so one reverse sweep of its node yields
 * `d/dx (v_k . grad f) = H v_k` exactly. `reverseHvps` runs one sweep per
 * direction and writes the `nInputs x N` HVP matrix.
 *
 * Cost: one recording plus `N` reverse sweeps over the shared tape and no
 * O(N^2) memory — node-less constants keep the tape the size of a plain
 * `RevScalar` recording, unlike re-recording the function once per direction.
 * The naive nesting still costs `1 + 3N` tape nodes per multiply (each
 * tangent component is an independent expression; `mixed_mode_chains.md`
 * §1.4), so this is the compatibility/oracle chain: `RevTape::hessianVector`
 * replay is the default second-order engine and a payload-dual tape the
 * type-level target (decisions #15).
 */

/// `N` directional tangents over a shared reverse tape.
template <std::size_t N>
using FwdRevScalar = Tangent<RevScalar, N>;

/// Reverse each tangent component of a `FwdRevScalar<N>` recording and write
/// the Hessian-vector products: `out[k * nInputs + i] = (H v_k)_i`, where
/// `v_k` is the k-th direction seeded into the input tangents and `H` is the
/// Hessian of `f.value` w.r.t. the `nInputs` input leaves `inputSlots[i]`
/// (the `.value.node` of the recorded inputs). Direction-major layout, so each
/// row is contiguous and directly comparable with
/// `RevTape::hessianVector(root, inputs, nInputs, v_k, out + k * nInputs)`.
///
/// Each sweep is seeded on the k-th tangent node (`f.d[k]`); a node-less
/// tangent (constant directional derivative, e.g. a function linear along
/// `v_k`) has zero curvature and writes a zero row. Adjoints are cleared
/// before every sweep; consume-on-use leaves the input (`Input`) slots
/// readable after each sweep, so `out` is valid when this returns and the last
/// row's leaf adjoints stay on the tape. No nodes are appended by the sweeps;
/// the caller must not record on the tape between recording `f` and calling
/// this.
template <std::size_t N>
void reverseHvps(const FwdRevScalar<N>& f, std::size_t nInputs, const std::uint32_t* inputSlots,
                 double* out) {
    RevTape& tape = RevTape::active();
    for (std::size_t k = 0; k < N; ++k) {
        tape.resetAdjoints();
        double* row = out + k * nInputs;
        const std::uint32_t root = f.d[k].node;
        if (root == RevTape::kNone) {
            for (std::size_t i = 0; i < nInputs; ++i) {
                row[i] = 0.0;
            }
            continue;
        }
        tape.reverse(root);
        for (std::size_t i = 0; i < nInputs; ++i) {
            row[i] = tape.adjoint(inputSlots[i]);
        }
    }
}

} // namespace lightape

#endif // LIGHTAPE_MIX_FWDREV_H
