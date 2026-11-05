#ifndef LIGHTAPE_MIX_THIRDORDER_H
#define LIGHTAPE_MIX_THIRDORDER_H

#include <cstddef>
#include <cstdint>

#include "lightape/fwd/ForwardScalar.h"
#include "lightape/rev/LeanReverse.h"

namespace lightape {
/**
 * @file ThirdOrder.h
 * @brief Third-order directional derivatives by mode nesting on the lean tape
 *
 * `FwdFwdRevScalar<N>` is `Tangent<Tangent<RevScalar, N>, N>`:
 * forward-over-forward-over-reverse over the shared 48 B first-order tape
 * (decisions #7/#15: order 3 is directional/contracted only; no
 * `fvar<fvar<var>>` chain and no second tape). The function is recorded once;
 * every input carries two independent forward direction sets and the reverse
 * sweep supplies the third slot:
 *
 *     RevTape& tape = RevTape::active();
 *     tape.clear();
 *     // x_i = x0_i + sum_k eps_k A_{k,i} + sum_l eta_l B_{l,i} (straight line)
 *     const FwdFwdRevScalar<N> x = Seed(x0, A, B);
 *     const FwdFwdRevScalar<N> f = exp(x * y) + log(x) - 3.0 * sin(y);
 *     const std::uint32_t inputs[2] = {x.value.value.node, y.value.value.node};
 *     double out[N * N * 2];
 *     reverseThird(f, 2, inputs, out);       // N^2 reverse sweeps
 *     // out[(k * N + l) * 2 + i] = T[A_k, B_l, e_i] (see below)
 *
 * Seeding convention (per input `x_i`): `x_i.value.value` is the tape leaf
 * (`RevTape::input`); `x_i.value.d[l]` carries the inner direction `B_l`;
 * `x_i.d[k].value` carries the outer direction `A_k`; the cross slots
 * `x_i.d[k].d[l]` hold the second path derivative (zero for the straight-line
 * parameterization above). With that convention `f.d[k].d[l]` is an ordinary
 * `RevScalar` equal to `d2 f / dA_k dB_l`, so one reverse sweep of its node
 * yields the gradient of that mixed directional derivative; stacking the
 * `N^2` sweeps gives the third-order tensor contracted with both forward
 * directions and left open in the reverse-seed slot:
 *
 *     out[(k * N + l) * nInputs + i]
 *         = T[A_k, B_l, e_i]
 *         = sum_{p,q} A_{k,p} B_{l,q} * d3f / (dx_p dx_q dx_i).
 *
 * This is the three-way contraction (two forward directions, one reverse
 * seed). With the basis seeds `A_k = B_k = e_k` and `N = nInputs` the output
 * is the dense `n x n x n` third-derivative tensor, row-major with the two
 * forward slots outermost. Cost is one recording plus `N^2` sweeps over the
 * tape; the sweeps append no nodes and each clears stale adjoints first, so
 * the tape position is unchanged on return. The caller must not record
 * between recording `f` and calling either helper, exactly as for
 * `reverseHvps` (FwdRev.h).
 *
 * Relation to the replay route: `higher_order.md` §1 derives a four-level
 * reverse replay (`A, C, D, E`) over a plain `RevScalar` recording. This
 * header is the mode-nesting formulation of the same contracted quantity
 * (order-3 tensors are diagnostics/contractions only, decision #7); it reuses
 * `Tangent` and the existing first-order VJPs untouched, at the price of
 * `1 + 3N` tape nodes per multiplication for each tangent component
 * (`mixed_mode_chains.md` §1.4). The tape-free triple-`Tangent<double>`
 * forward remains the independent oracle.
 *
 * Gamma chain: `lgamma`/`tgamma` nest through the polygamma chain
 * (`digamma` -> `trigamma` -> `tetragamma`; base Digamma.h, tape ops in
 * LeanReverse.h). The mixed component's innermost nodes are `RevScalar`
 * `Trigamma` nodes whose VJP factor is `tetragamma`, so third derivatives of
 * the gamma family are exact while the chain reaches that order;
 * `tetragamma(Tangent<RevScalar, N>)` (a pentagamma VJP) is the first
 * unsupported depth (ForwardScalar.h).
 *
 * The functions use `RevTape::active()`, so recording and sweeps must run on
 * the same thread and tape.
 */

/// `N` outer and `N` inner forward directions over a shared reverse tape.
template <std::size_t N>
using FwdFwdRevScalar = Tangent<Tangent<RevScalar, N>, N>;

/// Reverse each mixed tangent component of a `FwdFwdRevScalar<N>` recording
/// and write the contracted third derivatives, `N^2` sweeps total:
///
///     out[(k * N + l) * nInputs + i] = T[A_k, B_l, e_i]
///         = sum_{p,q} A_{k,p} B_{l,q} d3f / (dx_p dx_q dx_i),
///
/// with `A_k` the outer direction seeded in `x_i.d[k].value` and `B_l` the
/// inner direction seeded in `x_i.value.d[l]` (direction-major layout, each
/// `nInputs`-long row contiguous). `inputSlots[i]` is the tape leaf of the
/// i-th input (`x_i.value.value.node`); a `kNone` slot is a pure constant
/// input that contributes a zero column entry. `out` is fully overwritten.
///
/// Each sweep is seeded on the mixed node `f.d[k].d[l]`; a node-less mixed
/// component (a function whose mixed second directional derivative is
/// constant, e.g. any linear function) writes a zero row without a sweep.
/// Adjoints are cleared before every sweep, so input (`Input`) adjoints are
/// read cleanly and the leaves stay readable when this returns. No nodes are
/// appended by the sweeps.
template <std::size_t N>
void reverseThird(const FwdFwdRevScalar<N>& f, std::size_t nInputs, const std::uint32_t* inputSlots,
                  double* out) {
    RevTape& tape = RevTape::active();
    for (std::size_t k = 0; k < N; ++k) {
        for (std::size_t l = 0; l < N; ++l) {
            tape.resetAdjoints();
            double* row = out + (k * N + l) * nInputs;
            const std::uint32_t root = f.d[k].d[l].node;
            if (root == RevTape::kNone) {
                for (std::size_t i = 0; i < nInputs; ++i) {
                    row[i] = 0.0;
                }
                continue;
            }
            tape.reverse(root);
            for (std::size_t i = 0; i < nInputs; ++i) {
                const std::uint32_t slot = inputSlots[i];
                row[i] = (slot == RevTape::kNone) ? 0.0 : tape.adjoint(slot);
            }
        }
    }
}

/// Pure univariate third derivative `d3 / deps3 f(x + eps v)` in one forward
/// recording plus one reverse sweep and one `O(nInputs)` dot. Slot 0 of BOTH
/// forward levels must have been seeded with `direction` (i.e. `A_0 = B_0 =
/// direction`); then `f.d[0].d[0]` is `d2 / deps2 f(x + eps v)` and its
/// gradient contracted with `direction` is precisely
/// `T[direction, direction, direction] = sum_{p,q,r} v_p v_q v_r d3f`.
///
/// Why the reverse seed is not the obstacle: a lean reverse sweep seeds a
/// scalar at a root, not a vector at the leaves, so the third direction
/// cannot be "seeded into" the sweep. It does not need to be — the sweep
/// returns the whole gradient row and the dot contracts it, still only one
/// reverse pass. Re-recording would be required only if the two forward slots
/// had been seeded with other directions; without that precondition this
/// returns the general mixed contraction `T[direction, B_0, A_0]` (and with
/// `B_0 = 0`, e.g. a single-level tangent recording, it is identically zero).
/// Use `reverseThird` when the full `N x N` tensor is wanted.
template <std::size_t N>
double thirdDirectional(const FwdFwdRevScalar<N>& f, const std::uint32_t* inputSlots,
                        std::size_t nInputs, const double* direction) {
    RevTape& tape = RevTape::active();
    tape.resetAdjoints();
    const std::uint32_t root = f.d[0].d[0].node;
    if (root == RevTape::kNone) {
        return 0.0;
    }
    tape.reverse(root);
    double out = 0.0;
    for (std::size_t i = 0; i < nInputs; ++i) {
        const std::uint32_t slot = inputSlots[i];
        if (slot != RevTape::kNone) {
            out += direction[i] * tape.adjoint(slot);
        }
    }
    return out;
}

} // namespace lightape

#endif // LIGHTAPE_MIX_THIRDORDER_H
