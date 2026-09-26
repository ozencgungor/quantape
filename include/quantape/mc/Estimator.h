#ifndef QUANTAPE_MC_ESTIMATOR_H
#define QUANTAPE_MC_ESTIMATOR_H

#include "quantape/mc/Parallel.h"
#include "quantape/mc/SdePrimitives.h"

#include <Eigen/Dense>

#include <cmath>
#include <concepts>
#include <cstddef>
#include <vector>

namespace quantape::mc {
/**
 * @file Estimator.h
 * @brief Monte Carlo estimation over path blocks
 *
 * ## Specification
 *
 * For per-path values v^{(p)} = payoff(path p) collected over all blocks
 * (N paths total), the estimator computes
 *
 *     mean      m = (1/N) sum_p v^{(p)}
 *     variance  s^2 = ( sum_p (v^{(p)})^2 - N m^2 ) / (N - 1)
 *     stdError  SE = sqrt(s^2 / N)              (standard error of the mean)
 *
 * Payoffs are *batch* callables filling one value per path of a block, so
 * evaluation is vectorized and blocked (only the current block resident);
 * values are UNDISCOUNTED expectations of the path functional — discounting
 * belongs to the payoff/pricing layer.
 *
 * Pathwise-gradient accumulation (var mode, per-path nested tapes with
 * pinned noise) lands in the AD wrapper, not here.
 */

struct Estimate {
    double mean = 0.0;
    double variance = 0.0;
    double stdError = 0.0; ///< standard error of the mean
    std::size_t nPaths = 0;
};

/// Batch payoff: `out[p] = payoff(path p of the block)`, out.size() == nPaths
template <typename P>
concept BatchPayoff =
    requires(const P& p, const PathBlock<double>& block, Eigen::VectorXd& out) { p(block, out); };

struct BlockPartial {
    double sum = 0.0;
    double sumSq = 0.0;
    std::size_t n = 0;
};

template <typename Payoff>
    requires BatchPayoff<Payoff>
Estimate estimate(const std::vector<PathBlock<double>>& blocks, const Payoff& payoff,
                  Schedule schedule = Schedule::Sequential) {
    std::vector<BlockPartial> partials(blocks.size());
    detail::parallelFor(blocks.size(), schedule, [&](std::size_t blockIndex) {
        Eigen::VectorXd values;
        const PathBlock<double>& block = blocks[blockIndex];
        payoff(block, values);
        BlockPartial& partial = partials[blockIndex];
        for (Eigen::Index i = 0; i < values.size(); ++i) {
            const double v = values(i);
            partial.sum += v;
            partial.sumSq += v * v;
        }
        partial.n = static_cast<std::size_t>(values.size());
    });

    // Combine partials in block order (deterministic for any schedule)
    double sum = 0.0;
    double sumSq = 0.0;
    std::size_t n = 0;
    for (const BlockPartial& partial : partials) {
        sum += partial.sum;
        sumSq += partial.sumSq;
        n += partial.n;
    }

    Estimate result;
    result.nPaths = n;
    if (n == 0) {
        return result;
    }
    result.mean = sum / static_cast<double>(n);
    if (n > 1) {
        const double var = (sumSq - static_cast<double>(n) * result.mean * result.mean) /
                           static_cast<double>(n - 1);
        result.variance = var > 0.0 ? var : 0.0;
        result.stdError = std::sqrt(result.variance / static_cast<double>(n));
    }
    return result;
}

} // namespace quantape::mc

#endif // QUANTAPE_MC_ESTIMATOR_H
