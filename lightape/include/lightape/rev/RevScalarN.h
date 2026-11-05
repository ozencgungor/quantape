#ifndef LIGHTAPE_REV_REVSCALARN_H
#define LIGHTAPE_REV_REVSCALARN_H

#include <cstddef>

#include "lightape/batch/BatchTape.h"

namespace lightape {
/**
 * @file RevScalarN.h
 * @brief `RevScalarN<N>` / `RevTapeN<N>`: vector mode as the SoA batch tape
 * (decision #15, decision #9).
 *
 * Vector mode over `N` independent lanes is realized by the SoA
 * `BatchTape<W>` engine, so these are aliases and nothing else — no second
 * tape implementation:
 *
 *     RevScalarN<N> = BatchScalar<N>;
 *     RevTapeN<N>   = BatchTape<N>;
 *
 * Mapping (per decisions #9/#15):
 * - `BatchTape<W>` records one shared straight-line `RevOp` structure for `W`
 *   independent scalar programs ("lanes") and stores each node's lane values,
 *   adjoints and cached parent values contiguously in `std::array<double,W>`
 *   planes, so the sweep's inner width-`W` loops autovectorize (NEON 2, AVX2
 *   4, AVX-512 8 fp64 lanes) with no intrinsics.
 * - The reverse sweep is per-lane: every node update is applied to each lane
 *   with the lane's own values and partials, and no reduction ever crosses
 *   lanes, so lane `l` reproduces the scalar `RevScalar` tape of lane `l`
 *   for the same program (the bitwise contract in `BatchTape.h`).
 * - Gradients are read per lane: `tape.adjoint(slot)[l]` is the adjoint of
 *   lane `l`'s input (and `tape.value(slot)[l]` its primal). Seeding takes
 *   one seed per lane via `reverse(root, std::array<double,W>)`.
 *
 * Naming note: `N` is the lane (independent-evaluation) count, matching the
 * `RevScalarN<N>` entry of the mode-tier table in
 * `mode_mixing_and_requirements.md` §0/#15; it is not XAD's `AReal<T,N>`
 * "N derivative components stored inside every node" layout. The streaming
 * form of that idea — one sweep over a shared structure with a `K`-wide
 * adjoint batch — stays a replay-engine concern for now.
 */

/// Vector-adjoint scalar: `N` independent lanes on the shared SoA tape.
template <std::size_t N>
using RevScalarN = BatchScalar<N>;

/// Vector-mode tape: the `N`-lane SoA reverse tape.
template <std::size_t N>
using RevTapeN = BatchTape<N>;

} // namespace lightape

#endif // LIGHTAPE_REV_REVSCALARN_H
