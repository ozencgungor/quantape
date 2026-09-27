#ifndef QUANTAPE_MC_SDE_SIMULATOR_H
#define QUANTAPE_MC_SDE_SIMULATOR_H

#include "quantape/mc/Parallel.h"
#include "quantape/mc/Schemes.h"
#include "quantape/mc/SdePrimitives.h"
#include "quantape/mc/TimeGrid.h"

#include <Eigen/Dense>

#include <algorithm>
#include <cstddef>
#include <stdexcept>
#include <utility>
#include <vector>

namespace quantape::mc {
/**
 * @file SdeSimulator.h
 * @brief Generic SDE simulator
 *
 * ## Mathematical specification
 *
 * Simulates dX = f(X,t,theta) dt + sum_{j=1..q} g_j(X,t,theta) dW_j on the
 * grid t_0 < ... < t_N, with a scheme S advancing one interval at a time:
 *
 *     X_0       = x0                              (given initial state)
 *     Z_k       ~ source.fill(k, ...)             standardized (q x B) draws
 *     X_{k+1}   = S.step(X_k, t_k, dt_k, Z_k, f, g, theta_k)
 *
 * for k = 0..N-1 and every path column; theta_k is the piecewise-constant
 * parameter vector of interval [t_k, t_{k+1}]. All dt-dependent scaling is
 * the scheme's (sqrt(dt_k) for Euler); sources stay standardized.
 *
 * ## Blocking and reproducibility
 *
 * `simulateBlocks` streams one (d x blockSize) block at a time to a sink —
 * blocked evaluation is a contract (large path counts / dimensions must
 * not require the full tensor resident; `blockSize` is the knob).
 * `simulate` collects all blocks; `simulatePath(i)` returns exactly the
 * trajectory of path i of a full run, bitwise, because the keyed source
 * makes each draw a pure function of (seed, path, step, factor).
 *
 * Scalar-generic: `double` powers pricing; `var` will power AD through a
 * dedicated wrapper (per-path nested tapes, pinned noise = same draws).
 * All work buffers are local to a call (re-entrant, thread-safe); the
 * block loop is the natural parallelization point later.
 */
template <typename Scalar = double, typename Scheme = Euler>
class SdeSimulator {
public:
    SdeSimulator(TimeGrid timeGrid, std::vector<std::vector<Scalar>> theta,
                 Scheme scheme = Scheme{})
        : timeGrid_(std::move(timeGrid)), theta_(std::move(theta)), scheme_(std::move(scheme)) {
        if (theta_.size() != timeGrid_.nSteps()) {
            throw std::invalid_argument(
                "SdeSimulator: theta must hold one vector per grid interval");
        }
    }

    const TimeGrid& timeGrid() const { return timeGrid_; }
    const std::vector<std::vector<Scalar>>& theta() const { return theta_; }

    /// Stream blocks to a sink: sink(const PathBlock<Scalar>&, std::size_t blockIndex)
    ///
    /// With `Schedule::Parallel` the sink is invoked concurrently from
    /// worker threads and in nondeterministic order: it must be thread-safe
    /// and use `blockIndex` if deterministic accumulation is required (e.g.
    /// write into per-block partials and reduce afterwards). Path values are
    /// identical to the sequential schedule.
    template <typename DriftF, typename DiffusionF, typename Source, typename Sink>
        requires Drift<DriftF, Scalar> && Diffusion<DiffusionF, Scalar> && RandomSource<Source> &&
                 (Scheme::uniformStreams == 0 || UniformRandomSource<Source>)
    void simulateBlocks(const Eigen::Matrix<Scalar, Eigen::Dynamic, 1>& x0, const DriftF& drift,
                        const DiffusionF& diffusion, const Source& source, std::size_t nPaths,
                        std::size_t blockSize, const Sink& sink,
                        Schedule schedule = Schedule::Sequential) const {
        if (nPaths == 0) {
            throw std::invalid_argument("SdeSimulator: nPaths must be positive");
        }
        if (blockSize == 0) {
            throw std::invalid_argument("SdeSimulator: blockSize must be positive");
        }
        const std::size_t nBlocks = (nPaths + blockSize - 1) / blockSize;
        detail::parallelFor(nBlocks, schedule, [&](std::size_t blockIndex) {
            const std::size_t pathBegin = blockIndex * blockSize;
            const std::size_t blockPaths = std::min(blockSize, nPaths - pathBegin);
            PathBlock<Scalar> block =
                simulateBlock(x0, drift, diffusion, source, pathBegin, blockPaths);
            sink(block, blockIndex);
        });
    }

    /// Collecting overload; `Schedule::Parallel` stripes blocks over
    /// std::thread workers (deterministic: blocks are independent and the
    /// keyed source makes every path a pure function of its index).
    template <typename DriftF, typename DiffusionF, typename Source>
        requires Drift<DriftF, Scalar> && Diffusion<DiffusionF, Scalar> && RandomSource<Source> &&
                 (Scheme::uniformStreams == 0 || UniformRandomSource<Source>)
    std::vector<PathBlock<Scalar>>
    simulate(const Eigen::Matrix<Scalar, Eigen::Dynamic, 1>& x0, const DriftF& drift,
             const DiffusionF& diffusion, const Source& source, std::size_t nPaths,
             std::size_t blockSize, Schedule schedule = Schedule::Sequential) const {
        if (nPaths == 0) {
            throw std::invalid_argument("SdeSimulator: nPaths must be positive");
        }
        if (blockSize == 0) {
            throw std::invalid_argument("SdeSimulator: blockSize must be positive");
        }
        const std::size_t nBlocks = (nPaths + blockSize - 1) / blockSize;
        std::vector<PathBlock<Scalar>> blocks(nBlocks);
        detail::parallelFor(nBlocks, schedule, [&](std::size_t blockIndex) {
            const std::size_t pathBegin = blockIndex * blockSize;
            const std::size_t blockPaths = std::min(blockSize, nPaths - pathBegin);
            blocks[blockIndex] = simulateBlock(x0, drift, diffusion, source, pathBegin, blockPaths);
        });
        return blocks;
    }

    /// One path, bitwise identical to path `pathIndex` of a full run
    template <typename DriftF, typename DiffusionF, typename Source>
        requires Drift<DriftF, Scalar> && Diffusion<DiffusionF, Scalar> && RandomSource<Source> &&
                 (Scheme::uniformStreams == 0 || UniformRandomSource<Source>)
    PathBlock<Scalar> simulatePath(const Eigen::Matrix<Scalar, Eigen::Dynamic, 1>& x0,
                                   const DriftF& drift, const DiffusionF& diffusion,
                                   const Source& source, std::size_t pathIndex) const {
        return simulateBlock(x0, drift, diffusion, source, pathIndex, 1);
    }

    /// One path with an explicit scalar type and a shared (per-step
    /// identical) theta — the stepping contract used by the AD layers,
    /// which build fresh `var` leaves per path and thread them through
    /// every step. Stan-free: no AD types appear here; `PathScalar`
    /// instantiates on demand (e.g. `stan::math::var` in `Gradients.h`).
    template <typename PathScalar, typename DriftF, typename DiffusionF, typename Source>
        requires Drift<DriftF, PathScalar> && Diffusion<DiffusionF, PathScalar> &&
                 RandomSource<Source> &&
                 (Scheme::uniformStreams == 0 || UniformRandomSource<Source>)
    PathBlock<PathScalar>
    simulatePathSharedTheta(const Eigen::Matrix<PathScalar, Eigen::Dynamic, 1>& x0,
                            const DriftF& drift, const DiffusionF& diffusion, const Source& source,
                            std::size_t pathIndex, const std::vector<PathScalar>& theta) const {
        const std::size_t nSteps = timeGrid_.nSteps();
        const std::size_t nDims = static_cast<std::size_t>(x0.size());

        PathBlock<PathScalar> path;
        path.resize(nDims, nSteps, 1);
        path.states[0].col(0) = x0;

        typename Scheme::template Scratch<PathScalar> scratch(static_cast<Eigen::Index>(nDims), 1);
        Eigen::MatrixXd z;
        Eigen::MatrixXd uniforms;
        for (std::size_t k = 0; k < nSteps; ++k) {
            stepPath(path.states[k], path.states[k + 1], k, pathIndex, source, drift, diffusion,
                     theta, z, uniforms, scratch);
        }
        return path;
    }

    /// One scheme step with explicit scalar type and caller-owned buffers —
    /// the stepping primitive shared by the block loop and the AD layers.
    /// When the caller reuses `z`, `uniforms` and the scheme scratch, a
    /// whole path costs zero allocations (see `mc/Gradients.h` workspaces).
    template <typename PathScalar, typename DriftF, typename DiffusionF, typename Source>
        requires Drift<DriftF, PathScalar> && Diffusion<DiffusionF, PathScalar> &&
                 RandomSource<Source> &&
                 (Scheme::uniformStreams == 0 || UniformRandomSource<Source>)
    void stepPath(StateMatrix<PathScalar>& x, StateMatrix<PathScalar>& xNext, std::size_t step,
                  std::size_t pathIndex, const Source& source, const DriftF& drift,
                  const DiffusionF& diffusion, const std::vector<PathScalar>& theta,
                  Eigen::MatrixXd& z, Eigen::MatrixXd& uniforms,
                  typename Scheme::template Scratch<PathScalar>& scratch) const {
        const std::size_t nPaths = static_cast<std::size_t>(x.cols());
        source.fill(step, pathIndex, nPaths, z);
        if constexpr (Scheme::uniformStreams > 0) {
            source.fillUniform(step, pathIndex, nPaths, 0, Scheme::uniformStreams, uniforms);
        }
        scheme_.step(x, xNext, timeGrid_.time(step), timeGrid_.dt(step), z, uniforms, drift,
                     diffusion, theta, scratch);
    }

private:
    template <typename DriftF, typename DiffusionF, typename Source>
    PathBlock<Scalar> simulateBlock(const Eigen::Matrix<Scalar, Eigen::Dynamic, 1>& x0,
                                    const DriftF& drift, const DiffusionF& diffusion,
                                    const Source& source, std::size_t pathBegin,
                                    std::size_t blockPaths) const {
        const std::size_t nSteps = timeGrid_.nSteps();
        const std::size_t nDims = static_cast<std::size_t>(x0.size());

        PathBlock<Scalar> block;
        block.resize(nDims, nSteps, blockPaths);
        for (std::size_t p = 0; p < blockPaths; ++p) {
            block.states[0].col(static_cast<Eigen::Index>(p)) = x0;
        }

        // Scheme-owned scratch, constructed once per block and reused
        // across steps (zero per-step allocation).
        typename Scheme::template Scratch<Scalar> scratch(static_cast<Eigen::Index>(nDims),
                                                          static_cast<Eigen::Index>(blockPaths));
        Eigen::MatrixXd z;
        Eigen::MatrixXd uniforms;

        for (std::size_t k = 0; k < nSteps; ++k) {
            stepPath(block.states[k], block.states[k + 1], k, pathBegin, source, drift, diffusion,
                     theta_[k], z, uniforms, scratch);
        }
        return block;
    }

    TimeGrid timeGrid_;
    std::vector<std::vector<Scalar>> theta_;
    Scheme scheme_;
};

} // namespace quantape::mc

#endif // QUANTAPE_MC_SDE_SIMULATOR_H
