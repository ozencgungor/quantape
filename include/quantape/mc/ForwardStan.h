#ifndef QUANTAPE_MC_FORWARD_STAN_H
#define QUANTAPE_MC_FORWARD_STAN_H

#include "quantape/math/StanMath.h"

#include "quantape/mc/Gradients.h"
#include "quantape/mc/Parallel.h"
#include "quantape/mc/SdeSimulator.h"

#include <Eigen/Dense>

#include <concepts>
#include <cstddef>
#include <stdexcept>
#include <vector>

namespace quantape::mc {
/**
 * @file ForwardStan.h
 * @brief Stan-forward-mode pathwise gradients (default forward mode)
 *
 * The **default/reference forward mode**: the path runs on
 * `stan::math::fvar<double>` and one sweep per parameter direction
 * (`d + p` sweeps, seeded with a unit tangent on the component) yields
 * `∂π/∂param_j` for every path. No tape, exact derivatives of the
 * discretization, same estimator/reduction/sharding contract as the other
 * modes.
 *
 * This is the Stan-based forward implementation; the accelerated
 * fixed-size-dual and lean reverse variants live under `mc/mcfwdrev/` and
 * are opt-in, each gated against this and/or the Stan reverse mode
 * (`simulateGradient`). Cost is `(d + p + 0) x` a plain path (one sweep
 * per direction), so for wide parameter vectors prefer the Stan reverse
 * mode; for narrow ones this is simple, Stan-native, and second-order
 * composable (`fvar<var>` if ever needed).
 *
 * Fixed-noise contract as everywhere: keyed draws are never dualized.
 */

/// Scalar-generic payoff instantiated at `fvar<double>` and `double`.
template <typename P>
concept FvarPayoff = requires(const P& p, const PathBlock<stan::math::fvar<double>>& pathFvar,
                              const PathBlock<double>& pathDouble) {
    {
        p.template operator()<stan::math::fvar<double>>(pathFvar)
    } -> std::same_as<stan::math::fvar<double>>;
    { p.template operator()<double>(pathDouble) } -> std::same_as<double>;
};

namespace detail {

template <typename Scheme>
struct StanForwardWorkspace {
    using Dual = stan::math::fvar<double>;

    PathBlock<Dual> path;
    std::vector<Dual> thetaVars;
    Eigen::Matrix<Dual, Eigen::Dynamic, 1> x0Vars;
    typename Scheme::template Scratch<Dual> scratch;
    Eigen::MatrixXd z;
    Eigen::MatrixXd uniforms;

    StanForwardWorkspace(Eigen::Index nDims, std::size_t nSteps, std::size_t nTheta)
        : thetaVars(nTheta), x0Vars(nDims), scratch(nDims, 1) {
        path.resize(static_cast<std::size_t>(nDims), nSteps, 1);
    }
};

} // namespace detail

/**
 * @brief Stan-fvar forward gradient samples (shard-addressable)
 *
 * Runs `d + p` forward sweeps per path — one unit tangent per component of
 * `[x0; theta]` — and records the payoff's tangent in each direction.
 */
template <typename Scheme, typename DriftF, typename DiffusionF, typename Source, typename Payoff>
    requires Drift<DriftF, double> && Diffusion<DiffusionF, double> && RandomSource<Source> &&
             (Scheme::uniformStreams == 0 || UniformRandomSource<Source>) && FvarPayoff<Payoff>
GradientSamples simulateGradientStanForwardSamples(const SdeSimulator<double, Scheme>& simulator,
                                                   const Eigen::VectorXd& x0,
                                                   const std::vector<double>& theta,
                                                   const DriftF& drift, const DiffusionF& diffusion,
                                                   const Source& source, const Payoff& payoff,
                                                   std::size_t pathBegin, std::size_t nPaths,
                                                   Schedule schedule = Schedule::Sequential) {
    using Dual = stan::math::fvar<double>;
    if (nPaths == 0) {
        throw std::invalid_argument("simulateGradientStanForwardSamples: nPaths positive");
    }
    const std::size_t d = static_cast<std::size_t>(x0.size());
    const std::size_t p = theta.size();
    const std::size_t nParameters = d + p;
    const std::size_t nSteps = simulator.timeGrid().nSteps();

    GradientSamples samples;
    samples.pathBegin = pathBegin;
    samples.nStateDims = d;
    samples.nParameters = p;
    samples.values.assign(nPaths, 0.0);
    samples.gradients = Eigen::MatrixXd::Zero(static_cast<Eigen::Index>(nParameters),
                                              static_cast<Eigen::Index>(nPaths));

    const auto makeWorkspace = [&] {
        return detail::StanForwardWorkspace<Scheme>(static_cast<Eigen::Index>(d), nSteps, p);
    };
    detail::parallelForWithLocal(
        nPaths, schedule, makeWorkspace,
        [&](std::size_t offset, detail::StanForwardWorkspace<Scheme>& ws) {
            const std::size_t pathIndex = pathBegin + offset;
            for (std::size_t j = 0; j < nParameters; ++j) {
                for (std::size_t i = 0; i < d; ++i) {
                    const bool seed = (j < d) && (i == j);
                    ws.x0Vars(static_cast<Eigen::Index>(i)) =
                        Dual(x0(static_cast<Eigen::Index>(i)), seed ? 1.0 : 0.0);
                }
                for (std::size_t i = 0; i < p; ++i) {
                    const bool seed = (j >= d) && (j - d == i);
                    ws.thetaVars[i] = Dual(theta[i], seed ? 1.0 : 0.0);
                }
                ws.path.states[0].col(0) = ws.x0Vars;
                for (std::size_t k = 0; k < nSteps; ++k) {
                    simulator.template stepPath<Dual>(ws.path.states[k], ws.path.states[k + 1], k,
                                                      pathIndex, source, drift, diffusion,
                                                      ws.thetaVars, ws.z, ws.uniforms, ws.scratch);
                }
                const Dual y = payoff.template operator()<Dual>(ws.path);
                if (j == 0) {
                    samples.values[offset] = y.val_;
                }
                samples.gradients(static_cast<Eigen::Index>(j), static_cast<Eigen::Index>(offset)) =
                    y.d_;
            }
        });
    return samples;
}

/// Convenience: full path range, reduced estimate (Stan forward mode).
template <typename Scheme, typename DriftF, typename DiffusionF, typename Source, typename Payoff>
    requires Drift<DriftF, double> && Diffusion<DiffusionF, double> && RandomSource<Source> &&
             (Scheme::uniformStreams == 0 || UniformRandomSource<Source>) && FvarPayoff<Payoff>
GradientEstimate simulateGradientStanForward(const SdeSimulator<double, Scheme>& simulator,
                                             const Eigen::VectorXd& x0,
                                             const std::vector<double>& theta, const DriftF& drift,
                                             const DiffusionF& diffusion, const Source& source,
                                             const Payoff& payoff, std::size_t nPaths,
                                             Schedule schedule = Schedule::Sequential) {
    return reduceGradientSamples(simulateGradientStanForwardSamples(
        simulator, x0, theta, drift, diffusion, source, payoff, 0, nPaths, schedule));
}

} // namespace quantape::mc

#endif // QUANTAPE_MC_FORWARD_STAN_H
