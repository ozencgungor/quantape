#ifndef QUANTAPE_MC_STATE_DERIVATIVES_H
#define QUANTAPE_MC_STATE_DERIVATIVES_H

#include "quantape/math/StanMath.h"

#include "quantape/mc/SdePrimitives.h"
#include "quantape/mc/SdeSimulator.h"

#include <Eigen/Dense>

#include <concepts>
#include <cstddef>
#include <stdexcept>
#include <vector>

namespace quantape::mc {
/**
 * @file StateDerivatives.h
 * @brief Pathwise state-derivative pass ∂X(t)/∂[x0; θ] (S5c)
 *
 * Produces the forward tangent process
 *
 *     Y_k(i, j) = ∂ X_k(i) / ∂ param_j,   param = [x0; θ],
 *
 * for every grid time k, by one forward-mode (`fvar<double>`) sweep per
 * parameter over the scheme exactly as coded — the same discretization the
 * value and adjoint passes run. Cost is `(d + p) x` a plain double path;
 * tangents are exact (no step size).
 *
 * ## Streaming contract
 *
 * `simulatePathDerivatives(..., sink)` calls `sink(step, dX)` for
 * `step = 0 .. nSteps` in order, each `dX` a `(d × (d + p))` block whose
 * columns are `[∂X/∂x0 | ∂X/∂θ]` at that grid time. The all-times buffer is
 * `(nSteps + 1) × d × (d + p)` doubles per path (small: e.g. 20 KB for
 * Heston at 252 steps), so paths stay independently processable.
 *
 * ## Consumers
 *
 * - Callable/LSMC backward differentiation: the fold trick
 *   `∂V/∂θ += (∂V/∂X) · ∂X/∂θ|t₀` and the smoothed-δ boundary terms
 *   (`callable_ad_design.md` §2.2, eqs. 48/52/56).
 * - Cross-checks: for a terminal payoff,
 *   `∂E[π]/∂θ = E[(∂π/∂x_N) · Y_N]` (tested against `simulateGradient`).
 *
 * Fixed-noise differentiation as everywhere in `mc/`: keyed draws are
 * independent of the parameters and never dualized.
 */

/// Per-path state derivatives at every grid time.
struct StateDerivativePath {
    std::size_t nDims = 0;
    std::size_t nParameters = 0; ///< d + p, layout [x0; theta]
    std::size_t nSteps = 0;
    std::vector<StateMatrix<double>> dX; ///< dX[k] = d × (d + p), k = 0..nSteps
};

/**
 * @brief State derivatives for one path, streamed in step order
 *
 * Runs `d + p` forward-mode sweeps (one per parameter direction) and calls
 * `sink(step, dX_k)` for every grid time. Uses the absolute `pathIndex`
 * for keyed draws, so results are shard-addressable and reproducible.
 */
template <typename Scheme, typename DriftF, typename DiffusionF, typename Source, typename Sink>
    requires Drift<DriftF, double> && Diffusion<DiffusionF, double> && RandomSource<Source> &&
             (Scheme::uniformStreams == 0 || UniformRandomSource<Source>)
void simulatePathDerivatives(const SdeSimulator<double, Scheme>& simulator,
                             const Eigen::VectorXd& x0, const std::vector<double>& theta,
                             const DriftF& drift, const DiffusionF& diffusion, const Source& source,
                             std::size_t pathIndex, const Sink& sink) {
    using Dual = stan::math::fvar<double>;
    const Eigen::Index d = x0.size();
    const std::size_t p = theta.size();
    const std::size_t nParameters = static_cast<std::size_t>(d) + p;
    const std::size_t nSteps = simulator.timeGrid().nSteps();

    if (d <= 0) {
        throw std::invalid_argument("simulatePathDerivatives: x0 must be non-empty");
    }

    const Eigen::Index nParams = static_cast<Eigen::Index>(nParameters);
    // Tangent path: one contiguous (nSteps + 1) * d x (d + p) buffer (row
    // block k holds dX_k), so the pass allocates once instead of per step.
    Eigen::MatrixXd tangents =
        Eigen::MatrixXd::Zero(static_cast<Eigen::Index>(nSteps + 1) * d, nParams);
    for (Eigen::Index i = 0; i < d; ++i) {
        tangents(i, i) = 1.0; // ∂x0/∂x0 = I; ∂x0/∂θ = 0
    }

    // Sweep buffers constructed once per path (dims fixed across sweeps).
    StateMatrix<Dual> x(d, 1);
    StateMatrix<Dual> xNext(d, 1);
    std::vector<Dual> thetaDual(p, Dual(0.0));
    typename Scheme::template Scratch<Dual> scratch(d, 1);
    Eigen::MatrixXd z;
    Eigen::MatrixXd uniforms;

    for (std::size_t j = 0; j < nParameters; ++j) {
        for (Eigen::Index i = 0; i < d; ++i) {
            const bool seedState =
                (j < static_cast<std::size_t>(d)) && (i == static_cast<Eigen::Index>(j));
            x(i, 0) = Dual(x0(i), seedState ? 1.0 : 0.0);
        }
        for (std::size_t i = 0; i < p; ++i) {
            const bool seedTheta =
                (j >= static_cast<std::size_t>(d)) && (j - static_cast<std::size_t>(d) == i);
            thetaDual[i] = Dual(theta[i], seedTheta ? 1.0 : 0.0);
        }

        for (std::size_t k = 0; k < nSteps; ++k) {
            simulator.template stepPath<Dual>(x, xNext, k, pathIndex, source, drift, diffusion,
                                              thetaDual, z, uniforms, scratch);
            const Eigen::Index rowOffset = (static_cast<Eigen::Index>(k) + 1) * d;
            for (Eigen::Index i = 0; i < d; ++i) {
                tangents(rowOffset + i, static_cast<Eigen::Index>(j)) = xNext(i, 0).d_;
            }
            x.swap(xNext);
        }
    }

    StateMatrix<double> stepBlock(d, nParams);
    for (std::size_t k = 0; k <= nSteps; ++k) {
        const Eigen::Index rowOffset = static_cast<Eigen::Index>(k) * d;
        for (Eigen::Index j = 0; j < nParams; ++j) {
            for (Eigen::Index i = 0; i < d; ++i) {
                stepBlock(i, j) = tangents(rowOffset + i, j);
            }
        }
        sink(k, stepBlock);
    }
}

/// Convenience: store and return the full derivative path for one path.
template <typename Scheme, typename DriftF, typename DiffusionF, typename Source>
    requires Drift<DriftF, double> && Diffusion<DiffusionF, double> && RandomSource<Source> &&
             (Scheme::uniformStreams == 0 || UniformRandomSource<Source>)
StateDerivativePath
simulatePathDerivatives(const SdeSimulator<double, Scheme>& simulator, const Eigen::VectorXd& x0,
                        const std::vector<double>& theta, const DriftF& drift,
                        const DiffusionF& diffusion, const Source& source, std::size_t pathIndex) {
    StateDerivativePath out;
    out.nDims = static_cast<std::size_t>(x0.size());
    out.nParameters = out.nDims + theta.size();
    out.nSteps = simulator.timeGrid().nSteps();
    out.dX.resize(out.nSteps + 1);
    simulatePathDerivatives(
        simulator, x0, theta, drift, diffusion, source, pathIndex,
        [&out](std::size_t step, const StateMatrix<double>& dX) { out.dX[step] = dX; });
    return out;
}

} // namespace quantape::mc

#endif // QUANTAPE_MC_STATE_DERIVATIVES_H
