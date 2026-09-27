#ifndef QUANTAPE_MC_FORWARD_GRADIENTS_H
#define QUANTAPE_MC_FORWARD_GRADIENTS_H

#include "quantape/mc/Gradients.h"
#include "quantape/mc/Parallel.h"
#include "quantape/mc/SdeSimulator.h"
#include "quantape/mc/mcfwdrev/ForwardScalar.h"

#include <Eigen/Dense>

#include <concepts>
#include <cstddef>
#include <stdexcept>
#include <vector>

namespace quantape::mc {
/**
 * @file ForwardGradients.h
 * @brief Forward-mode pathwise gradients and state derivatives (no tape)
 *
 * The tape-free alternative to `mc/Gradients.h` for small parameter
 * counts: the whole path runs on `Tangent<Scalar, N>` duals, so the value
 * and all `N = d + p` sensitivities `[∂/∂x0; ∂/∂θ]` come out of one pass
 * (no AD nodes, no reverse sweep, no scope setup).
 *
 * - `simulateGradientForward<N>` — same estimator contract as
 *   `simulateGradient`: per-path samples, ordered reduction, bitwise
 *   parallel == sequential, shard-addressable via `pathBegin`.
 * - `simulateGradientForwardSamples` — the shard-addressable core.
 * - `simulatePathDerivativesForward<N>` — the S5c state-derivative pass in
 *   ONE sweep (vs `d + p` fvar sweeps in `mc/StateDerivatives.h`), streamed
 *   per step.
 *
 * Cost model: ~(N + 1)x a plain `double` path, all in registers/L1.
 * Reverse mode (`simulateGradient`) wins when `N` is large (its cost is
 * constant in the number of sensitivities); forward wins for the typical
 * model case (`N <= ~8`).
 *
 * The same fixed-noise contract applies: keyed draws are never dualized.
 */

/// Scalar-generic single-path payoff instantiated at `Tangent<double, N>`.
template <typename P, std::size_t N>
concept TangentPayoff = requires(const P& p, const PathBlock<Tangent<double, N>>& path) {
    { p.template operator()<Tangent<double, N>>(path) } -> std::same_as<Tangent<double, N>>;
};

/// Batch tangent payoff: fills one dual per path of the block
/// (`out[p] = payoff(path p)`), the vectorized counterpart used by the
/// blocked forward mode.
template <typename P, std::size_t N>
concept TangentBatchPayoff = requires(const P& p, const PathBlock<Tangent<double, N>>& path,
                                      Eigen::Matrix<Tangent<double, N>, Eigen::Dynamic, 1>& out) {
    p.template operator()<Tangent<double, N>>(path, out);
};

namespace detail {

template <typename Scheme, std::size_t N>
struct ForwardWorkspace {
    using Dual = Tangent<double, N>;

    PathBlock<Dual> path;
    std::vector<Dual> thetaVars;
    Eigen::Matrix<Dual, Eigen::Dynamic, 1> x0Vars;
    typename Scheme::template Scratch<Dual> scratch;
    Eigen::MatrixXd z;
    Eigen::MatrixXd uniforms;

    ForwardWorkspace(Eigen::Index nDims, std::size_t nSteps, std::size_t nTheta,
                     Eigen::Index nPaths = 1)
        : scratch(nDims, nPaths) {
        path.resize(static_cast<std::size_t>(nDims), nSteps, static_cast<std::size_t>(nPaths));
        thetaVars.resize(nTheta);
        x0Vars.resize(nDims);
    }
};

} // namespace detail

/**
 * @brief Forward-mode pathwise gradient samples (shard-addressable)
 *
 * `N` must equal `x0.size() + theta.size()` (layout `[x0; θ]`); one dual
 * path per sample, streaming the value and the `N` tangents.
 */
template <std::size_t N, typename Scheme, typename DriftF, typename DiffusionF, typename Source,
          typename Payoff>
    requires Drift<DriftF, double> && Diffusion<DiffusionF, double> && RandomSource<Source> &&
             (Scheme::uniformStreams == 0 || UniformRandomSource<Source>) &&
             TangentPayoff<Payoff, N>
GradientSamples
simulateGradientForwardSamples(const SdeSimulator<double, Scheme>& simulator,
                               const Eigen::VectorXd& x0, const std::vector<double>& theta,
                               const DriftF& drift, const DiffusionF& diffusion,
                               const Source& source, const Payoff& payoff, std::size_t pathBegin,
                               std::size_t nPaths, Schedule schedule = Schedule::Sequential) {
    using Dual = Tangent<double, N>;
    if (nPaths == 0) {
        throw std::invalid_argument("simulateGradientForwardSamples: nPaths must be positive");
    }
    const std::size_t d = static_cast<std::size_t>(x0.size());
    const std::size_t p = theta.size();
    if (d + p != N) {
        throw std::invalid_argument("simulateGradientForwardSamples: N must equal d + p");
    }
    const std::size_t nSteps = simulator.timeGrid().nSteps();

    GradientSamples samples;
    samples.pathBegin = pathBegin;
    samples.nStateDims = d;
    samples.nParameters = p;
    samples.values.assign(nPaths, 0.0);
    samples.gradients =
        Eigen::MatrixXd::Zero(static_cast<Eigen::Index>(N), static_cast<Eigen::Index>(nPaths));

    const auto makeWorkspace = [&] {
        return detail::ForwardWorkspace<Scheme, N>(static_cast<Eigen::Index>(d), nSteps, p);
    };
    detail::parallelForWithLocal(
        nPaths, schedule, makeWorkspace,
        [&](std::size_t offset, detail::ForwardWorkspace<Scheme, N>& ws) {
            const std::size_t pathIndex = pathBegin + offset;

            // Seed the parameter directions: dX0/dx0 = I, dX0/dθ = 0.
            for (std::size_t i = 0; i < d; ++i) {
                ws.x0Vars(static_cast<Eigen::Index>(i)) = Dual(x0(static_cast<Eigen::Index>(i)));
                ws.x0Vars(static_cast<Eigen::Index>(i)).tangent(i) = 1.0;
            }
            for (std::size_t j = 0; j < p; ++j) {
                ws.thetaVars[j] = Dual(theta[j]);
                ws.thetaVars[j].tangent(d + j) = 1.0;
            }

            ws.path.states[0].col(0) = ws.x0Vars;
            for (std::size_t k = 0; k < nSteps; ++k) {
                simulator.template stepPath<Dual>(ws.path.states[k], ws.path.states[k + 1], k,
                                                  pathIndex, source, drift, diffusion, ws.thetaVars,
                                                  ws.z, ws.uniforms, ws.scratch);
            }
            const Dual y = payoff.template operator()<Dual>(ws.path);

            samples.values[offset] = y.value;
            for (std::size_t i = 0; i < N; ++i) {
                samples.gradients(static_cast<Eigen::Index>(i), static_cast<Eigen::Index>(offset)) =
                    y.d[i];
            }
        });
    return samples;
}

/**
 * @brief Blocked forward-mode gradient samples (B paths per dual step)
 *
 * Same math as `simulateGradientForwardSamples`, but a task simulates a
 * block of `blockSize` paths with vectorized dual state matrices, so the
 * per-step Eigen expression overhead is amortized over the block (the B=1
 * forward Euler `.rowwise()` artifact). The payoff is the batch variant
 * (`out[p]`), so terminal payoffs vectorize too. Per-path values and
 * gradients are bitwise identical to the per-path mode (same keyed draws,
 * same column-wise arithmetic; the last block may simulate a few padding
 * paths beyond `nPaths` whose samples are discarded — paths are
 * independent, so valid columns are unaffected).
 */
template <std::size_t N, typename Scheme, typename DriftF, typename DiffusionF, typename Source,
          typename Payoff>
    requires Drift<DriftF, double> && Diffusion<DiffusionF, double> && RandomSource<Source> &&
             (Scheme::uniformStreams == 0 || UniformRandomSource<Source>) &&
             TangentBatchPayoff<Payoff, N>
GradientSamples simulateGradientForwardBlockSamples(
    const SdeSimulator<double, Scheme>& simulator, const Eigen::VectorXd& x0,
    const std::vector<double>& theta, const DriftF& drift, const DiffusionF& diffusion,
    const Source& source, const Payoff& payoff, std::size_t pathBegin, std::size_t nPaths,
    std::size_t blockSize = 64, Schedule schedule = Schedule::Sequential) {
    using Dual = Tangent<double, N>;
    if (nPaths == 0) {
        throw std::invalid_argument("simulateGradientForwardBlockSamples: nPaths positive");
    }
    if (blockSize == 0) {
        throw std::invalid_argument("simulateGradientForwardBlockSamples: blockSize positive");
    }
    const std::size_t d = static_cast<std::size_t>(x0.size());
    const std::size_t p = theta.size();
    if (d + p != N) {
        throw std::invalid_argument("simulateGradientForwardBlockSamples: N must equal d + p");
    }
    const std::size_t nSteps = simulator.timeGrid().nSteps();
    const std::size_t nBlocks = (nPaths + blockSize - 1) / blockSize;
    const Eigen::Index B = static_cast<Eigen::Index>(blockSize);

    GradientSamples samples;
    samples.pathBegin = pathBegin;
    samples.nStateDims = d;
    samples.nParameters = p;
    samples.values.assign(nPaths, 0.0);
    samples.gradients =
        Eigen::MatrixXd::Zero(static_cast<Eigen::Index>(N), static_cast<Eigen::Index>(nPaths));

    const auto makeWorkspace = [&] {
        return detail::ForwardWorkspace<Scheme, N>(static_cast<Eigen::Index>(d), nSteps, p, B);
    };
    detail::parallelForWithLocal(
        nBlocks, schedule, makeWorkspace,
        [&](std::size_t blockIndex, detail::ForwardWorkspace<Scheme, N>& ws) {
            const std::size_t blockBegin = blockIndex * blockSize;
            const std::size_t blockValid = std::min(blockSize, nPaths - blockBegin);
            const std::size_t pathBeginBlock = pathBegin + blockBegin;

            // Identical parameter directions on every column; all B columns
            // are simulated (padding columns have keys >= nPaths).
            for (std::size_t i = 0; i < d; ++i) {
                Dual seed(x0(static_cast<Eigen::Index>(i)));
                seed.tangent(i) = 1.0;
                for (Eigen::Index j = 0; j < B; ++j) {
                    ws.path.states[0](static_cast<Eigen::Index>(i), j) = seed;
                }
            }
            for (std::size_t j = 0; j < p; ++j) {
                ws.thetaVars[j] = Dual(theta[j]);
                ws.thetaVars[j].tangent(d + j) = 1.0;
            }

            for (std::size_t k = 0; k < nSteps; ++k) {
                simulator.template stepPath<Dual>(ws.path.states[k], ws.path.states[k + 1], k,
                                                  pathBeginBlock, source, drift, diffusion,
                                                  ws.thetaVars, ws.z, ws.uniforms, ws.scratch);
            }
            Eigen::Matrix<Dual, Eigen::Dynamic, 1> out(B);
            payoff.template operator()<Dual>(ws.path, out);

            for (std::size_t j = 0; j < blockValid; ++j) {
                const std::size_t offset = blockBegin + j;
                samples.values[offset] = out(static_cast<Eigen::Index>(j)).value;
                for (std::size_t i = 0; i < N; ++i) {
                    samples.gradients(static_cast<Eigen::Index>(i),
                                      static_cast<Eigen::Index>(offset)) =
                        out(static_cast<Eigen::Index>(j)).d[i];
                }
            }
        });
    return samples;
}

/// Convenience: full path range, reduced estimate (blocked forward mode).
template <std::size_t N, typename Scheme, typename DriftF, typename DiffusionF, typename Source,
          typename Payoff>
    requires Drift<DriftF, double> && Diffusion<DiffusionF, double> && RandomSource<Source> &&
             (Scheme::uniformStreams == 0 || UniformRandomSource<Source>) &&
             TangentBatchPayoff<Payoff, N>
GradientEstimate
simulateGradientForwardBlock(const SdeSimulator<double, Scheme>& simulator,
                             const Eigen::VectorXd& x0, const std::vector<double>& theta,
                             const DriftF& drift, const DiffusionF& diffusion, const Source& source,
                             const Payoff& payoff, std::size_t nPaths, std::size_t blockSize = 64,
                             Schedule schedule = Schedule::Sequential) {
    return reduceGradientSamples(simulateGradientForwardBlockSamples<N>(
        simulator, x0, theta, drift, diffusion, source, payoff, 0, nPaths, blockSize, schedule));
}

/// Convenience: full path range, reduced estimate (forward mode).
template <std::size_t N, typename Scheme, typename DriftF, typename DiffusionF, typename Source,
          typename Payoff>
    requires Drift<DriftF, double> && Diffusion<DiffusionF, double> && RandomSource<Source> &&
             (Scheme::uniformStreams == 0 || UniformRandomSource<Source>) &&
             TangentPayoff<Payoff, N>
GradientEstimate
simulateGradientForward(const SdeSimulator<double, Scheme>& simulator, const Eigen::VectorXd& x0,
                        const std::vector<double>& theta, const DriftF& drift,
                        const DiffusionF& diffusion, const Source& source, const Payoff& payoff,
                        std::size_t nPaths, Schedule schedule = Schedule::Sequential) {
    return reduceGradientSamples(simulateGradientForwardSamples<N>(
        simulator, x0, theta, drift, diffusion, source, payoff, 0, nPaths, schedule));
}

/**
 * @brief State derivatives `∂X(t)/∂[x0; θ]` in one forward pass
 *
 * Same contract as `mc/StateDerivatives.h` (`sink(step, d × N)`), but one
 * dual sweep instead of `N` nested-fvar sweeps — the callable layer's hot
 * path. `N` must equal `x0.size() + theta.size()`.
 */
template <std::size_t N, typename Scheme, typename DriftF, typename DiffusionF, typename Source,
          typename Sink>
    requires Drift<DriftF, double> && Diffusion<DiffusionF, double> && RandomSource<Source> &&
             (Scheme::uniformStreams == 0 || UniformRandomSource<Source>)
void simulatePathDerivativesForward(const SdeSimulator<double, Scheme>& simulator,
                                    const Eigen::VectorXd& x0, const std::vector<double>& theta,
                                    const DriftF& drift, const DiffusionF& diffusion,
                                    const Source& source, std::size_t pathIndex, const Sink& sink) {
    using Dual = Tangent<double, N>;
    const Eigen::Index d = x0.size();
    const std::size_t p = theta.size();
    if (static_cast<std::size_t>(d) + p != N) {
        throw std::invalid_argument("simulatePathDerivativesForward: N must equal d + p");
    }
    const std::size_t nSteps = simulator.timeGrid().nSteps();

    StateMatrix<Dual> x(d, 1);
    StateMatrix<Dual> xNext(d, 1);
    for (Eigen::Index i = 0; i < d; ++i) {
        x(i, 0) = Dual(x0(i));
        x(i, 0).tangent(static_cast<std::size_t>(i)) = 1.0;
    }
    std::vector<Dual> thetaDual(p);
    for (std::size_t j = 0; j < p; ++j) {
        thetaDual[j] = Dual(theta[j]);
        thetaDual[j].tangent(static_cast<std::size_t>(d) + j) = 1.0;
    }
    typename Scheme::template Scratch<Dual> scratch(d, 1);
    Eigen::MatrixXd z;
    Eigen::MatrixXd uniforms;

    StateMatrix<double> block = StateMatrix<double>::Zero(d, static_cast<Eigen::Index>(N));
    for (Eigen::Index i = 0; i < d; ++i) {
        block(i, i) = 1.0; // ∂x0/∂x0 = I; ∂x0/∂θ = 0
    }
    sink(0, block);

    for (std::size_t k = 0; k < nSteps; ++k) {
        simulator.template stepPath<Dual>(x, xNext, k, pathIndex, source, drift, diffusion,
                                          thetaDual, z, uniforms, scratch);
        for (Eigen::Index i = 0; i < d; ++i) {
            for (std::size_t j = 0; j < N; ++j) {
                block(i, static_cast<Eigen::Index>(j)) = xNext(i, 0).d[j];
            }
        }
        sink(k + 1, block);
        x.swap(xNext);
    }
}

} // namespace quantape::mc

#endif // QUANTAPE_MC_FORWARD_GRADIENTS_H
