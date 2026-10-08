#ifndef QUANTAPE_MC_EVALUATE_H
#define QUANTAPE_MC_EVALUATE_H

#include "quantape/mc/SdePrimitives.h"

#include <Eigen/Dense>

#include <concepts>
#include <cstddef>
#include <span>
#include <stdexcept>
#include <type_traits>
#include <utility>

namespace quantape::mc {
/**
 * @file Evaluate.h
 * @brief Block-batched SoA evaluation of per-path functions
 *
 * Simulation and pricing loops evaluate per-path quantities on blocks of
 * states: terminal or schedule PVs, exercise indicators, regression rows. The
 * engine layout is SoA — a state block is `(nDims x nPaths)`, one path per
 * column — and evaluation is batched, allocation-free and AD-templated
 * (`double`, `var`, `fvar<var>`).
 *
 * Two interchangeable callable forms:
 *
 *   - **PathEvaluator**: `f(state, out)` evaluates one path, with `state` a
 *     length-nDims span (one state column) and `out` a length-nOutputs span
 *     (one output column). This is the scalar form pricing/LSMC code writes.
 *   - **BatchEvaluator**: `f(states, out)` evaluates the whole
 *     `(nDims x nPaths)` state block into the `(nOutputs x nPaths)` output
 *     block in one call. This is the vectorized form model code writes.
 *
 * `evaluateBlock` drives a batch callable into a caller-owned output block;
 * `perPath` lifts a path callable to the batch contract by walking the
 * columns. Neither allocates: both blocks are caller-owned and written
 * directly, so a caller that reuses its output block evaluates with zero
 * allocations per call. A simulator sink uses the batch form on
 * `block.states[k]`; a pricing loop uses the same call on its state block.
 */

/// Scalar per-path evaluator: `f(state, out)` with `state` a length-nDims and
/// `out` a length-nOutputs span; writes every output of one path.
template <typename F, typename Scalar>
concept PathEvaluator =
    requires(const F& f, std::span<const Scalar> state, std::span<Scalar> out) { f(state, out); };

/// SoA batch evaluator: `f(states, out)` with `states` (nDims x nPaths) and
/// `out` (nOutputs x nPaths) — one call for the whole block.
template <typename F, typename Scalar>
concept BatchEvaluator = requires(const F& f, const StateMatrix<Scalar>& states,
                                  StateMatrix<Scalar>& out) { f(states, out); };

/// Drive a batch evaluator over one state block into a caller-owned output
/// block that must already have the matching path count. Performs no
/// allocation and no dispatch beyond the inlined callable.
template <typename Scalar, typename F>
    requires BatchEvaluator<F, Scalar>
void evaluateBlock(const F& evaluator, const StateMatrix<Scalar>& states,
                   StateMatrix<Scalar>& out) {
    if (out.cols() != states.cols()) {
        throw std::invalid_argument("evaluateBlock: output and state path counts differ");
    }
    evaluator(states, out);
}

/// Per-path evaluator lifted to the batch contract: each state column is
/// evaluated in place into the matching output column. The evaluator is held
/// by value (cheap, stateless descriptions) and reused across calls.
template <typename Scalar, typename F>
    requires PathEvaluator<F, Scalar>
class PerPathEvaluator {
public:
    explicit PerPathEvaluator(F evaluator) : m_evaluator(std::move(evaluator)) {}

    void operator()(const StateMatrix<Scalar>& states, StateMatrix<Scalar>& out) const {
        if (out.cols() != states.cols()) {
            throw std::invalid_argument("PerPathEvaluator: output and state path counts differ");
        }
        const Eigen::Index nDims = states.rows();
        const Eigen::Index nOutputs = out.rows();
        for (Eigen::Index p = 0; p < states.cols(); ++p) {
            m_evaluator(
                std::span<const Scalar>(states.col(p).data(), static_cast<std::size_t>(nDims)),
                std::span<Scalar>(out.col(p).data(), static_cast<std::size_t>(nOutputs)));
        }
    }

private:
    F m_evaluator;
};

/// Lift a per-path evaluator to the batch contract for `Scalar`.
template <typename Scalar, typename F>
    requires PathEvaluator<F, Scalar>
PerPathEvaluator<Scalar, std::decay_t<F>> perPath(F&& evaluator) {
    return PerPathEvaluator<Scalar, std::decay_t<F>>(std::forward<F>(evaluator));
}

} // namespace quantape::mc

#endif // QUANTAPE_MC_EVALUATE_H
