#ifndef QUANTAPE_MATH_FIXED_POINT_ITERATOR_H
#define QUANTAPE_MATH_FIXED_POINT_ITERATOR_H

#include <algorithm>
#include <cstddef>

namespace quantape::math {
/**
 * @file FixedPointIterator.h
 * @brief Damped fixed-point driver for coupled curve builds
 *
 * Generic over the state and the pass: `pass(state)` rewrites the state in
 * place, `updateNorm(previous, current)` measures the change (e.g. max-norm
 * over node updates). Used by the turn-overlay re-fit and the xccy/turn
 * coupled bootstrap; the joint solver is the fallback when this fails.
 */

struct FixedPointOptions {
    double xtol = 1e-12; ///< Convergence tolerance on the update norm
    int maxPasses = 20;  ///< Hard pass limit
    int stablePasses = 1;///< Consecutive passes below tolerance required
};

struct FixedPointResult {
    bool converged = false;
    int passes = 0;
    double update = 0.0;
};

/// Run `pass` until `updateNorm` stays below tolerance for `stablePasses`
/// consecutive iterations (or `maxPasses` is reached).
template <typename Pass, typename State, typename Norm>
FixedPointResult fixedPointIterate(Pass&& pass, State& state, Norm&& updateNorm,
                                   const FixedPointOptions& options = {}) {
    FixedPointResult result;
    State previous = state;
    int stable = 0;
    const int maxPasses = std::max(1, options.maxPasses);
    for (int iteration = 0; iteration < maxPasses; ++iteration) {
        previous = state;
        pass(state);
        result.passes = iteration + 1;
        result.update = updateNorm(previous, state);
        if (result.update <= options.xtol) {
            ++stable;
            if (stable >= std::max(1, options.stablePasses)) {
                result.converged = true;
                return result;
            }
        } else {
            stable = 0;
        }
    }
    return result;
}

} // namespace quantape::math

#endif // QUANTAPE_MATH_FIXED_POINT_ITERATOR_H
