#ifndef QUANTAPE_MC_LEAN_GRADIENTS_H
#define QUANTAPE_MC_LEAN_GRADIENTS_H

#include "quantape/mc/Gradients.h"
#include "quantape/mc/Parallel.h"
#include "quantape/mc/SdeSimulator.h"
#include "quantape/mc/mcfwdrev/LeanReverse.h"

#include <Eigen/Dense>

#include <concepts>
#include <cstddef>
#include <stdexcept>
#include <vector>

namespace quantape::mc {
/**
 * @file LeanGradients.h
 * @brief Pathwise gradients on the lean reverse tape (no Stan var)
 *
 * **Opt-in accelerated reverse mode.** The *default* and correctness
 * reference is the Stan-backed `simulateGradient` (`mc/Gradients.h`);
 * this header (`mc/mcfwdrev/`) is the lean-tape alternative for many-factor states beyond
 * the forward-mode crossover, where reverse mode's p-independent cost is
 * essential. Its outputs must stay gate-equal to both the Stan reverse
 * mode and the (FD-gated) forward mode — the test suite enforces exact
 * same-path agreement — so it can be enabled per workload, never by
 * default, without weakening the correctness story.
 *
 * Same estimator/contract as `simulateGradient` (per-path tape, ordered
 * reduction, shard-addressable), but stepping runs on `RevScalar`/
 * `RevTape` (`mc/mcfwdrev/LeanReverse.h`): fixed-layout nodes, opcode-switched
 * reverse sweep, node-less constants, thread-local arena reused across
 * paths.
 *
 * Bitwise determinism across schedules holds exactly as in the other
 * modes (per-path independence + ordered reduction).
 */

/// Scalar-generic single-path payoff instantiated at `RevScalar`.
template <typename P>
concept LeanPayoff = requires(const P& p, const PathBlock<RevScalar>& path) {
    { p.template operator()<RevScalar>(path) } -> std::same_as<RevScalar>;
};

namespace detail {

template <typename Scheme>
struct LeanWorkspace {
    PathBlock<RevScalar> path;
    std::vector<RevScalar> thetaVars;
    Eigen::Matrix<RevScalar, Eigen::Dynamic, 1> x0Vars;
    typename Scheme::template Scratch<RevScalar> scratch;
    Eigen::MatrixXd z;
    Eigen::MatrixXd uniforms;

    LeanWorkspace(Eigen::Index nDims, std::size_t nSteps, std::size_t nTheta)
        : thetaVars(nTheta), x0Vars(nDims), scratch(nDims, 1) {
        path.resize(static_cast<std::size_t>(nDims), nSteps, 1);
    }
};

} // namespace detail

/// Tape nodes used by one path (diagnostics / benchmarking).
template <typename Scheme, typename DriftF, typename DiffusionF, typename Source, typename Payoff>
    requires Drift<DriftF, double> && Diffusion<DiffusionF, double> && RandomSource<Source> &&
             (Scheme::uniformStreams == 0 || UniformRandomSource<Source>) && LeanPayoff<Payoff>
std::size_t leanPathTapeNodes(const SdeSimulator<double, Scheme>& simulator,
                              const Eigen::VectorXd& x0, const std::vector<double>& theta,
                              const DriftF& drift, const DiffusionF& diffusion,
                              const Source& source, const Payoff& payoff,
                              std::size_t pathIndex = 0) {
    const std::size_t d = static_cast<std::size_t>(x0.size());
    const std::size_t p = theta.size();
    const std::size_t nSteps = simulator.timeGrid().nSteps();
    RevTape& tape = RevTape::active();
    tape.clear();
    tape.reserve(nSteps * 64);

    PathBlock<RevScalar> path;
    path.resize(d, nSteps, 1);
    std::vector<RevScalar> thetaVars(p);
    for (std::size_t j = 0; j < p; ++j) {
        thetaVars[j] = RevScalar(theta[j], tape.input(theta[j]));
    }
    for (std::size_t i = 0; i < d; ++i) {
        const double v = x0(static_cast<Eigen::Index>(i));
        path.states[0](static_cast<Eigen::Index>(i), 0) = RevScalar(v, tape.input(v));
    }
    typename Scheme::template Scratch<RevScalar> scratch(static_cast<Eigen::Index>(d), 1);
    Eigen::MatrixXd z;
    Eigen::MatrixXd uniforms;
    for (std::size_t k = 0; k < nSteps; ++k) {
        simulator.template stepPath<RevScalar>(path.states[k], path.states[k + 1], k, pathIndex,
                                               source, drift, diffusion, thetaVars, z, uniforms,
                                               scratch);
    }
    const RevScalar y = payoff.template operator()<RevScalar>(path);
    (void)y;
    return tape.size();
}

/**
 * @brief Pathwise gradient samples on the lean reverse tape
 * (shard-addressable; the payoff is any scalar-generic path payoff).
 */
template <typename Scheme, typename DriftF, typename DiffusionF, typename Source, typename Payoff>
    requires Drift<DriftF, double> && Diffusion<DiffusionF, double> && RandomSource<Source> &&
             (Scheme::uniformStreams == 0 || UniformRandomSource<Source>) && LeanPayoff<Payoff>
GradientSamples
simulateGradientLeanSamples(const SdeSimulator<double, Scheme>& simulator,
                            const Eigen::VectorXd& x0, const std::vector<double>& theta,
                            const DriftF& drift, const DiffusionF& diffusion, const Source& source,
                            const Payoff& payoff, std::size_t pathBegin, std::size_t nPaths,
                            Schedule schedule = Schedule::Sequential) {
    if (nPaths == 0) {
        throw std::invalid_argument("simulateGradientLeanSamples: nPaths must be positive");
    }
    const std::size_t d = static_cast<std::size_t>(x0.size());
    const std::size_t p = theta.size();
    const std::size_t nSteps = simulator.timeGrid().nSteps();

    GradientSamples samples;
    samples.pathBegin = pathBegin;
    samples.nStateDims = d;
    samples.nParameters = p;
    samples.values.assign(nPaths, 0.0);
    samples.gradients =
        Eigen::MatrixXd::Zero(static_cast<Eigen::Index>(d + p), static_cast<Eigen::Index>(nPaths));

    const auto makeWorkspace = [&] {
        return detail::LeanWorkspace<Scheme>(static_cast<Eigen::Index>(d), nSteps, p);
    };
    detail::parallelForWithLocal(
        nPaths, schedule, makeWorkspace,
        [&](std::size_t offset, detail::LeanWorkspace<Scheme>& ws) {
            const std::size_t pathIndex = pathBegin + offset;
            RevTape& tape = RevTape::active();
            tape.clear();
            tape.reserve(nSteps * 64);

            for (std::size_t j = 0; j < p; ++j) {
                ws.thetaVars[j] = RevScalar(theta[j], tape.input(theta[j]));
            }
            for (std::size_t i = 0; i < d; ++i) {
                const double v = x0(static_cast<Eigen::Index>(i));
                ws.x0Vars(static_cast<Eigen::Index>(i)) = RevScalar(v, tape.input(v));
            }
            ws.path.states[0].col(0) = ws.x0Vars;

            for (std::size_t k = 0; k < nSteps; ++k) {
                simulator.template stepPath<RevScalar>(ws.path.states[k], ws.path.states[k + 1], k,
                                                       pathIndex, source, drift, diffusion,
                                                       ws.thetaVars, ws.z, ws.uniforms, ws.scratch);
            }
            const RevScalar y = payoff.template operator()<RevScalar>(ws.path);

            samples.values[offset] = y.value;
            if (y.node == RevTape::kNone) {
                return; // constant payoff: all sensitivities zero
            }
            tape.reverse(y.node);
            for (std::size_t i = 0; i < d; ++i) {
                samples.gradients(static_cast<Eigen::Index>(i), static_cast<Eigen::Index>(offset)) =
                    tape.adjoint(ws.x0Vars(static_cast<Eigen::Index>(i)).node);
            }
            for (std::size_t j = 0; j < p; ++j) {
                samples.gradients(static_cast<Eigen::Index>(d + j),
                                  static_cast<Eigen::Index>(offset)) =
                    tape.adjoint(ws.thetaVars[j].node);
            }
        });
    return samples;
}

/// Convenience: full path range, reduced estimate (lean reverse).
template <typename Scheme, typename DriftF, typename DiffusionF, typename Source, typename Payoff>
    requires Drift<DriftF, double> && Diffusion<DiffusionF, double> && RandomSource<Source> &&
             (Scheme::uniformStreams == 0 || UniformRandomSource<Source>) && LeanPayoff<Payoff>
GradientEstimate
simulateGradientLean(const SdeSimulator<double, Scheme>& simulator, const Eigen::VectorXd& x0,
                     const std::vector<double>& theta, const DriftF& drift,
                     const DiffusionF& diffusion, const Source& source, const Payoff& payoff,
                     std::size_t nPaths, Schedule schedule = Schedule::Sequential) {
    return reduceGradientSamples(simulateGradientLeanSamples(simulator, x0, theta, drift, diffusion,
                                                             source, payoff, 0, nPaths, schedule));
}

} // namespace quantape::mc

#endif // QUANTAPE_MC_LEAN_GRADIENTS_H
