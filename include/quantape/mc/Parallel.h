#ifndef QUANTAPE_MC_PARALLEL_H
#define QUANTAPE_MC_PARALLEL_H

#include <cstddef>

#include <tbb/blocked_range.h>
#include <tbb/parallel_for.h>

namespace quantape::mc {
/**
 * @file Parallel.h
 * @brief Execution schedule for the block loops (TBB task arena)
 *
 * The engine's block loop is embarrassingly parallel: blocks are
 * independent, sources are const/thread-safe (keyed draws), and each block
 * writes its own storage slot, so parallel execution is deterministic —
 * bitwise identical per path regardless of schedule or thread count.
 *
 * `detail::parallelFor` runs over TBB's persistent thread pool (project
 * dependency, also used by Stan), so small jobs pay almost no thread-start
 * cost; exceptions propagate out of the parallel_for.
 */

enum class Schedule {
    Sequential,
    Parallel,
};

namespace detail {

template <typename F>
void parallelFor(std::size_t count, Schedule schedule, const F& fn) {
    if (schedule != Schedule::Parallel || count <= 1) {
        for (std::size_t i = 0; i < count; ++i) {
            fn(i);
        }
        return;
    }
    tbb::parallel_for(tbb::blocked_range<std::size_t>(0, count),
                      [&fn](const tbb::blocked_range<std::size_t>& range) {
                          for (std::size_t i = range.begin(); i < range.end(); ++i) {
                              fn(i);
                          }
                      });
}

} // namespace detail
} // namespace quantape::mc

#endif // QUANTAPE_MC_PARALLEL_H
