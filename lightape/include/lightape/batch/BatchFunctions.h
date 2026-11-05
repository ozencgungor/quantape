#ifndef LIGHTAPE_BATCH_BATCHFUNCTIONS_H
#define LIGHTAPE_BATCH_BATCHFUNCTIONS_H

#include <array>
#include <cstddef>
#include <limits>

#include "lightape/batch/BatchTape.h"

namespace lightape {
/**
 * @file BatchFunctions.h
 * @brief Fused lane-wise reductions/activations over `BatchTape<W>` scalars.
 *
 * Every function below is a straight-line composite of the batch opcodes
 * (Input/Add/Sub/Mul/Div/Neg/Sqrt/Exp/Log/Abs/Pow/fmin/fmax), so it records
 * one shared statement stream for all W lanes and needs no file outside
 * `BatchTape.h`. Per lane the statements are emitted and swept in exactly the
 * order of the scalar `lightape` counterpart; sibling lanes share only the
 * instruction stream, never a reduction, so each lane's primal and every
 * input's adjoint are bitwise equal to a scalar tape running that lane's
 * values through the same program (the `BatchTape` bitwise contract).
 *
 * `logSumExp`/`softmax` mirror the scalar `External` implementations
 * statement for statement: the max-shift scan (`>` compare chain), the
 * zero-initialized ordered sum `s += exp(x_i - m)`, the `m + log(s)` root and
 * the VJP term `(seed / s) * exp(x_i - m)` all use the scalar order, so
 * per-lane values and adjoints match the scalar path bitwise whenever the
 * scalar path uses that order (the max scan and the sum are the only ordering
 * choices, and no lane ever crosses a reduction). `softmax` records one shared
 * `lse` chain and one `exp(x_i - lse)` per component, matching the scalar
 * shared-node structure.
 *
 * `softplus`/`smoothMax` mirror `ops/Generic.h`. The batch tape has no
 * `Log1p` opcode, so `log1p(u)` is composed as `log(1 + u)` (the one
 * documented deviation: the primal is the correctly-rounded composition, not
 * libm `log1p`). `softplus` keeps the scalar `x > 30` overflow guard: the
 * `exp` argument is capped with `fmin(x, 30)` (an exact no-op for `x <= 30`)
 * so no lane can overflow, and the guard is a lane-wise `fmin`/`fmax` select
 * that routes the chosen branch's adjoint through the mask machinery untouched
 * (all-lane-uniform recordings collapse to the scalar shared node).
 */
namespace batch {

/// Stable lane-wise log-sum-exp over `n` lane-valued scalars. The per-lane
/// program is `m = max_i x_i` via the scalar `>` scan, `s += exp(x_i - m)`
/// from `s = 0` in index order, then `m + log(s)`: the same operation order
/// (and the same per-lane VJP term `(seed / s) * exp(x_i - m)`) as
/// `lightape::logSumExp`, so each lane's value and every input's adjoint are
/// bitwise equal to the scalar path when it uses that order. An empty range
/// returns the node-less constant 0 like the scalar overload.
template <std::size_t W>
BatchScalar<W> logSumExp(const BatchScalar<W>* xs, std::size_t n) {
    if (n == 0) {
        return BatchScalar<W>(0.0);
    }
    std::array<double, W> peak = xs[0].value;
    for (std::size_t i = 1; i < n; ++i) {
        for (std::size_t l = 0; l < W; ++l) {
            if (xs[i].value[l] > peak[l]) {
                peak[l] = xs[i].value[l];
            }
        }
    }
    const BatchScalar<W> shift(peak);
    BatchScalar<W> sum(0.0);
    for (std::size_t i = 0; i < n; ++i) {
        sum = sum + exp(xs[i] - shift);
    }
    return shift + log(sum);
}

/// Stable lane-wise softmax over `n` lane-valued scalars: one shared
/// `logSumExp` chain plus one `exp(x_i - lse)` per component, exactly the
/// scalar `lightape::softmax` structure (one shared lse node, ordinary tape
/// nodes for every output, so a downstream objective reverses in one sweep).
template <std::size_t W>
void softmax(const BatchScalar<W>* xs, std::size_t n, BatchScalar<W>* out) {
    if (n == 0) {
        return;
    }
    const BatchScalar<W> lse = logSumExp(xs, n);
    for (std::size_t i = 0; i < n; ++i) {
        out[i] = exp(xs[i] - lse);
    }
}

namespace detail {

/// Per-lane select of `a` in the `take` lanes and `b` elsewhere, built from
/// the tape's own `fmin`/`fmax` routing. `fmax(a, take ? -inf : +inf)` pushes
/// every non-`take` lane to +inf, `fmax(b, take ? +inf : -inf)` pushes every
/// `take` lane to +inf, and the final `fmin` picks the wanted side
/// (`fmin(a', +inf) = a'`, `fmin(+inf, b') = b'`). The losing side receives
/// nothing, so the winner's adjoint is the incoming adjoint routed through the
/// mask nodes untouched; an all-`take` or all-drop decision collapses to `a`
/// resp. `b` themselves (uniform `fmin`/`fmax` share the winner's node), which
/// keeps uniform recordings node-for-node equal to the scalar branch.
template <std::size_t W>
BatchScalar<W> selectLanes(const std::array<bool, W>& take, const BatchScalar<W>& a,
                           const BatchScalar<W>& b) {
    std::array<double, W> avoidA{};
    std::array<double, W> avoidB{};
    for (std::size_t l = 0; l < W; ++l) {
        avoidA[l] = take[l] ? -std::numeric_limits<double>::infinity()
                            : std::numeric_limits<double>::infinity();
        avoidB[l] = take[l] ? std::numeric_limits<double>::infinity()
                            : -std::numeric_limits<double>::infinity();
    }
    return fmin(fmax(a, BatchScalar<W>(avoidA)), fmax(b, BatchScalar<W>(avoidB)));
}

} // namespace detail

/// Lane-wise softplus with the scalar `lightape::softplus` overflow guard
/// (`x > 30 ? x : log1p(exp(x))`). `log1p` is composed as `log(1 + u)` and the
/// `exp` argument is capped with `fmin(x, 30)` (exact for `x <= 30`, where the
/// guard picks the composite branch) so no lane can overflow; the guard itself
/// is a lane-wise `fmin`/`fmax` select using the mask routing. Uniform lanes
/// return `x` (all `x > 30`) or the composite node (all `x <= 30`) untouched,
/// matching the scalar branch's node sharing.
template <std::size_t W>
BatchScalar<W> softplus(const BatchScalar<W>& x) {
    const BatchScalar<W> capped = fmin(x, BatchScalar<W>(30.0));
    const BatchScalar<W> y = log(BatchScalar<W>(1.0) + exp(capped));
    std::array<bool, W> big{};
    for (std::size_t l = 0; l < W; ++l) {
        big[l] = x.value[l] > 30.0;
    }
    return detail::selectLanes(big, x, y);
}

/// Lane-wise smooth maximum `m + log1p(exp(-|a - b| k)) / k` with
/// `m = fmax(a, b)`, mirroring `lightape::smoothMax` (with `log1p` composed as
/// `log(1 + u)`; the `exp` argument is `<= 0`, so it never overflows). `fmax`
/// shares the winner's node when every lane picks the same operand and records
/// the mask-routed select otherwise, the same branch-sharing contract as the
/// scalar generic.
template <std::size_t W>
BatchScalar<W> smoothMax(const BatchScalar<W>& a, const BatchScalar<W>& b, double k) {
    const BatchScalar<W> kk(k);
    const BatchScalar<W> m = fmax(a, b);
    return m + log(BatchScalar<W>(1.0) + exp(-fabs(a - b) * kk)) / kk;
}

} // namespace batch
} // namespace lightape

#endif // LIGHTAPE_BATCH_BATCHFUNCTIONS_H
