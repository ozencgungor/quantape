#ifndef QUANTAPE_MC_GRADIENTS_H
#define QUANTAPE_MC_GRADIENTS_H

#include "quantape/math/StanMath.h"

#include "quantape/mc/Parallel.h"
#include "quantape/mc/SdePrimitives.h"
#include "quantape/mc/SdeSimulator.h"

#include <Eigen/Dense>

#include <concepts>
#include <cstddef>
#include <stdexcept>
#include <vector>

namespace quantape::mc {
/**
 * @file Gradients.h
 * @brief Pathwise (IPA) model gradients via per-path reverse-mode AD (S5)
 *
 * Two interchangeable modes, both following the same math contract:
 *
 * - `simulateGradient` — one nested tape per path (full path on the tape),
 *   reversed once, on Stan `var` (the **default/reference reverse mode**;
 *   the lean-tape alternative lives in `mc/mcfwdrev/LeanGradients.h` and must stay
 *   gate-equal to it). Any `PathPayoff`; the mode to use when the payoff
 *   depends on the whole path.
 * - `simulateGradientCheckpointed` — step-wise tape gluing (Savine §5/§7
 *   checkpointing adapted to time stepping): a plain `double` forward pass
 *   fills one reused path buffer, then each step builds a tiny tape (7
 *   nodes for 1-D Euler) that is reversed immediately with the incoming
 *   state adjoint (`lambda' x_{k+1}` seeding). Peak AD memory is the
 *   one-step tape plus `d x nSteps` doubles, independent of the horizon.
 *   Restriction: the payoff must depend on the terminal state only.
 *
 * Both modes use per-worker workspaces (`parallelForWithLocal`), so the
 * hot path performs zero allocations per path: path storage, scheme
 * scratch, draw buffers and AD leaves are reused across paths.
 *
 * ## Execution models
 *
 * - `Schedule::Parallel` (TBB): in-process threads over path blocks.
 * - `Schedule::Sequential` + the `*Samples` entry points: **multiprocessing
 *   sharding**, one single-threaded process per core over disjoint path
 *   ranges (`pathBegin`, `nPaths`), merged with
 *   `reduceGradientSamples`. Because draws are keyed by *absolute* path
 *   index and the reduction walks samples in path order, merging shards is
 *   bitwise identical to a single-process run — the caller only needs
 *   enough aggregate memory for the samples.
 *
 * ## Estimator
 *
 *     value    = (1/N) sum_p pi(X^{(p)})
 *     gradient = (1/N) sum_p d pi(X^{(p)}) / d (x0, theta)
 *
 * per-path samples are reduced in path order, so `mean ± SE` falls out of
 * the same pass and cost is constant in the number of sensitivities.
 *
 * ## Correctness contract (internal_docs/sde_gradients_design.md)
 *
 * - Fixed-noise differentiation: keyed draws are independent of
 *   (x0, theta) and are never taped -> bitwise common random numbers.
 * - We differentiate the scheme exactly as coded; gradient bias equals
 *   the scheme's own bias.
 * - Regularity: Lipschitz payoffs; control flow (positivity guards, QE
 *   branches) is primal-pinned (a.e. correct derivative). Discontinuous
 *   payoffs must be smoothed (`payoffs/Indicators.h`).
 * - Primal parity with the `double` engine: checkpointed mode is bitwise
 *   (value pass is plain double); full-tape mode agrees to ~1 ulp
 *   (design-doc gate 10).
 *
 * ## LSM / callable switch
 *
 * The payoff is scalar-generic, and its internals decide whether the tape
 * runs through an LSMC regression: computing regression proxies in plain
 * `double` freezes them (cash-flow differentials; Savine / Antonov "noisy"
 * regime), while computing them in `Scalar` differentiates through them
 * (full regime). Both are the same call here — the switch lives in the
 * payoff implementation (`callable_ad_design.md` §6).
 */

/// Value plus per-sensitivity MC errors; gradient layout is [x0; theta].
struct GradientEstimate {
    double value = 0.0;
    double valueStdError = 0.0;
    Eigen::VectorXd gradient;  ///< d + p entries
    Eigen::VectorXd stdErrors; ///< d + p entries
    std::size_t nPaths = 0;
    std::size_t nStateDims = 0;
    std::size_t nParameters = 0;
};

/// Per-path samples over an absolute path range [pathBegin, pathBegin+n).
/// Merging shards = concatenate in path order + `reduceGradientSamples`.
struct GradientSamples {
    std::vector<double> values; ///< v_p, one per path (ordered)
    Eigen::MatrixXd gradients;  ///< (d + p) x nPaths, column per path
    std::size_t pathBegin = 0;
    std::size_t nStateDims = 0;
    std::size_t nParameters = 0;
    std::size_t nPaths() const { return values.size(); }
};

/// Scalar-generic single-path payoff: `Scalar operator()<Scalar>(path)`.
template <typename P>
concept PathPayoff = requires(const P& p, const PathBlock<stan::math::var>& pathVar,
                              const PathBlock<double>& pathDouble) {
    { p.template operator()<stan::math::var>(pathVar) } -> std::same_as<stan::math::var>;
    { p.template operator()<double>(pathDouble) } -> std::same_as<double>;
};

namespace detail {

/// Per-worker gradient workspace: allocated once, reused for every path
/// the worker processes (zero per-path allocation on the hot path).
template <typename Scheme>
struct GradientWorkspace {
    using var = stan::math::var;

    // full-tape mode
    PathBlock<var> varPath;
    std::vector<var> thetaVars;
    Eigen::Matrix<var, Eigen::Dynamic, 1> x0Vars;

    // checkpointed mode
    PathBlock<double> fwdPath;
    PathBlock<var> termPath; ///< one state (nSteps = 0): terminal seed
    StateMatrix<var> xVar;
    StateMatrix<var> xNextVar;
    Eigen::VectorXd lambda;
    std::vector<double> thetaAcc;

    // shared step buffers
    typename Scheme::template Scratch<double> scratchDouble;
    typename Scheme::template Scratch<var> scratchVar;
    Eigen::MatrixXd z;
    Eigen::MatrixXd uniforms;

    GradientWorkspace(Eigen::Index nDims, std::size_t nSteps, std::size_t nTheta,
                      std::size_t nPaths)
        : scratchDouble(nDims, static_cast<Eigen::Index>(1)),
          scratchVar(nDims, static_cast<Eigen::Index>(1)) {
        varPath.resize(static_cast<std::size_t>(nDims), nSteps, nPaths);
        fwdPath.resize(static_cast<std::size_t>(nDims), nSteps, nPaths);
        termPath.resize(static_cast<std::size_t>(nDims), 0, nPaths);
        xVar.resize(nDims, static_cast<Eigen::Index>(nPaths));
        xNextVar.resize(nDims, static_cast<Eigen::Index>(nPaths));
        lambda.resize(nDims);
        thetaVars.resize(nTheta);
        thetaAcc.assign(nTheta, 0.0);
        x0Vars.resize(nDims);
    }
};

} // namespace detail

/// Ordered reduction of per-path samples into the estimate (deterministic;
/// identical for every schedule and for merged multiprocessing shards).
inline GradientEstimate reduceGradientSamples(const GradientSamples& samples) {
    const std::size_t nPaths = samples.nPaths();
    const std::size_t d = samples.nStateDims;
    const std::size_t p = samples.nParameters;
    const std::size_t nSensitivities = d + p;

    GradientEstimate estimate;
    estimate.nPaths = nPaths;
    estimate.nStateDims = d;
    estimate.nParameters = p;
    estimate.gradient = Eigen::VectorXd::Zero(static_cast<Eigen::Index>(nSensitivities));
    estimate.stdErrors = Eigen::VectorXd::Zero(static_cast<Eigen::Index>(nSensitivities));
    if (nPaths == 0) {
        return estimate;
    }

    double sumValue = 0.0;
    Eigen::VectorXd sumGrad = Eigen::VectorXd::Zero(static_cast<Eigen::Index>(nSensitivities));
    for (std::size_t path = 0; path < nPaths; ++path) {
        sumValue += samples.values[path];
        sumGrad += samples.gradients.col(static_cast<Eigen::Index>(path));
    }
    const double n = static_cast<double>(nPaths);
    estimate.value = sumValue / n;
    estimate.gradient = sumGrad / n;

    double sumValueSq = 0.0;
    Eigen::VectorXd sumGradSq = Eigen::VectorXd::Zero(static_cast<Eigen::Index>(nSensitivities));
    for (std::size_t path = 0; path < nPaths; ++path) {
        const double dv = samples.values[path] - estimate.value;
        sumValueSq += dv * dv;
        const Eigen::VectorXd dg =
            samples.gradients.col(static_cast<Eigen::Index>(path)) - estimate.gradient;
        sumGradSq += dg.cwiseProduct(dg);
    }
    if (nPaths > 1) {
        estimate.valueStdError = std::sqrt(sumValueSq / (n - 1.0) / n);
        estimate.stdErrors = (sumGradSq / (n - 1.0) / n).cwiseSqrt();
    }
    return estimate;
}

/**
 * @brief Pathwise gradient samples — full-tape mode (any `PathPayoff`)
 *
 * One nested tape per path on a reused workspace, reversed once, for paths
 * [pathBegin, pathBegin+nPaths). Draws use the absolute path index, so
 * shards are independent and mergeable in path order.
 */
template <typename Scheme, typename DriftF, typename DiffusionF, typename Source, typename Payoff>
    requires Drift<DriftF, double> && Diffusion<DiffusionF, double> && RandomSource<Source> &&
             (Scheme::uniformStreams == 0 || UniformRandomSource<Source>) && PathPayoff<Payoff>
GradientSamples simulateGradientSamples(const SdeSimulator<double, Scheme>& simulator,
                                        const Eigen::VectorXd& x0, const std::vector<double>& theta,
                                        const DriftF& drift, const DiffusionF& diffusion,
                                        const Source& source, const Payoff& payoff,
                                        std::size_t pathBegin, std::size_t nPaths,
                                        Schedule schedule = Schedule::Sequential) {
    using stan::math::var;
    if (nPaths == 0) {
        throw std::invalid_argument("simulateGradientSamples: nPaths must be positive");
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
        return detail::GradientWorkspace<Scheme>(static_cast<Eigen::Index>(d), nSteps, p, 1);
    };
    detail::parallelForWithLocal(
        nPaths, schedule, makeWorkspace,
        [&](std::size_t offset, detail::GradientWorkspace<Scheme>& ws) {
            const std::size_t pathIndex = pathBegin + offset;
            stan::math::nested_rev_autodiff scope;

            // Fresh AD leaves per path: they must live on this worker's
            // nested (thread-local) tape, never shared across workers.
            for (std::size_t j = 0; j < p; ++j) {
                ws.thetaVars[j] = theta[j];
            }
            for (std::size_t i = 0; i < d; ++i) {
                ws.x0Vars(i) = x0(i);
            }
            ws.varPath.states[0].col(0) = ws.x0Vars;
            for (std::size_t k = 0; k < nSteps; ++k) {
                simulator.template stepPath<var>(ws.varPath.states[k], ws.varPath.states[k + 1], k,
                                                 pathIndex, source, drift, diffusion, ws.thetaVars,
                                                 ws.z, ws.uniforms, ws.scratchVar);
            }
            var y = payoff.template operator()<var>(ws.varPath);

            y.grad(); // one reverse sweep; adjoints are readable below
            samples.values[offset] = y.val();
            for (std::size_t i = 0; i < d; ++i) {
                samples.gradients(static_cast<Eigen::Index>(i), static_cast<Eigen::Index>(offset)) =
                    ws.x0Vars(i).adj();
            }
            for (std::size_t j = 0; j < p; ++j) {
                samples.gradients(static_cast<Eigen::Index>(d + j),
                                  static_cast<Eigen::Index>(offset)) = ws.thetaVars[j].adj();
            }
        });
    return samples;
}

/**
 * @brief Pathwise gradient samples — checkpointed (step-wise gluing) mode
 *
 * Forward pass in `double` (one reused `d x (nSteps+1)` buffer; value read
 * from the terminal state), then for `k = nSteps-1 .. 0` a one-step `var`
 * tape is built and reversed immediately with the incoming state adjoint
 * (`y = lambda' x_{k+1}`; `y.grad()` yields the next lambda via `x_k` and
 * this step's theta contribution, accumulated over steps). Peak AD memory
 * is the one-step tape (tens of nodes, cache-resident) plus `d x nSteps`
 * doubles — no tape grows with the horizon.
 *
 * Contract: the payoff may depend on the **terminal state only** (the
 * adjoint seed is built from a one-state path). Path-dependent payoffs
 * must use `simulateGradientSamples`.
 */
template <typename Scheme, typename DriftF, typename DiffusionF, typename Source, typename Payoff>
    requires Drift<DriftF, double> && Diffusion<DiffusionF, double> && RandomSource<Source> &&
             (Scheme::uniformStreams == 0 || UniformRandomSource<Source>) && PathPayoff<Payoff>
GradientSamples simulateGradientCheckpointedSamples(
    const SdeSimulator<double, Scheme>& simulator, const Eigen::VectorXd& x0,
    const std::vector<double>& theta, const DriftF& drift, const DiffusionF& diffusion,
    const Source& source, const Payoff& payoff, std::size_t pathBegin, std::size_t nPaths,
    Schedule schedule = Schedule::Sequential) {
    using stan::math::var;
    if (nPaths == 0) {
        throw std::invalid_argument("simulateGradientCheckpointedSamples: nPaths must be positive");
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
        return detail::GradientWorkspace<Scheme>(static_cast<Eigen::Index>(d), nSteps, p, 1);
    };
    detail::parallelForWithLocal(
        nPaths, schedule, makeWorkspace,
        [&](std::size_t offset, detail::GradientWorkspace<Scheme>& ws) {
            const std::size_t pathIndex = pathBegin + offset;

            // 1. Forward pass in plain double: value + all states (reused
            //    buffer, one contiguous allocation per worker).
            ws.fwdPath.states[0].col(0) = x0;
            for (std::size_t k = 0; k < nSteps; ++k) {
                simulator.template stepPath<double>(ws.fwdPath.states[k], ws.fwdPath.states[k + 1],
                                                    k, pathIndex, source, drift, diffusion, theta,
                                                    ws.z, ws.uniforms, ws.scratchDouble);
            }
            samples.values[offset] = payoff.template operator()<double>(ws.fwdPath);

            // 2. Terminal adjoint seed: lambda = d pi / d x_N (tiny tape).
            {
                stan::math::nested_rev_autodiff scope;
                ws.termPath.states[0] = ws.fwdPath.states.back();
                var pi = payoff.template operator()<var>(ws.termPath);
                pi.grad();
                for (std::size_t i = 0; i < d; ++i) {
                    ws.lambda(static_cast<Eigen::Index>(i)) =
                        ws.termPath.states[0](static_cast<Eigen::Index>(i), 0).adj();
                }
            }

            // 3. Backward gluing: one tiny tape per step, reversed with the
            //    incoming state adjoint.
            for (std::size_t j = 0; j < p; ++j) {
                ws.thetaAcc[j] = 0.0;
            }
            for (std::size_t step = nSteps; step-- > 0;) {
                stan::math::nested_rev_autodiff scope;
                for (std::size_t i = 0; i < d; ++i) {
                    ws.xVar(static_cast<Eigen::Index>(i), 0) =
                        ws.fwdPath.states[step](static_cast<Eigen::Index>(i), 0);
                }
                for (std::size_t j = 0; j < p; ++j) {
                    ws.thetaVars[j] = theta[j];
                }
                simulator.template stepPath<var>(ws.xVar, ws.xNextVar, step, pathIndex, source,
                                                 drift, diffusion, ws.thetaVars, ws.z, ws.uniforms,
                                                 ws.scratchVar);

                // Seeding y = lambda' x_{k+1} costs d branchless multiply-adds.
                var y = 0.0;
                for (std::size_t i = 0; i < d; ++i) {
                    y += ws.lambda(static_cast<Eigen::Index>(i)) *
                         ws.xNextVar(static_cast<Eigen::Index>(i), 0);
                }
                y.grad();
                for (std::size_t i = 0; i < d; ++i) {
                    ws.lambda(static_cast<Eigen::Index>(i)) =
                        ws.xVar(static_cast<Eigen::Index>(i), 0).adj();
                }
                for (std::size_t j = 0; j < p; ++j) {
                    ws.thetaAcc[j] += ws.thetaVars[j].adj();
                }
            }

            // After k = 0, lambda holds dV/dx0.
            for (std::size_t i = 0; i < d; ++i) {
                samples.gradients(static_cast<Eigen::Index>(i), static_cast<Eigen::Index>(offset)) =
                    ws.lambda(static_cast<Eigen::Index>(i));
            }
            for (std::size_t j = 0; j < p; ++j) {
                samples.gradients(static_cast<Eigen::Index>(d + j),
                                  static_cast<Eigen::Index>(offset)) = ws.thetaAcc[j];
            }
        });
    return samples;
}

/// Convenience: full path range, reduced estimate (in-process or TBB).
template <typename Scheme, typename DriftF, typename DiffusionF, typename Source, typename Payoff>
    requires Drift<DriftF, double> && Diffusion<DiffusionF, double> && RandomSource<Source> &&
             (Scheme::uniformStreams == 0 || UniformRandomSource<Source>) && PathPayoff<Payoff>
GradientEstimate simulateGradient(const SdeSimulator<double, Scheme>& simulator,
                                  const Eigen::VectorXd& x0, const std::vector<double>& theta,
                                  const DriftF& drift, const DiffusionF& diffusion,
                                  const Source& source, const Payoff& payoff, std::size_t nPaths,
                                  Schedule schedule = Schedule::Sequential) {
    return reduceGradientSamples(simulateGradientSamples(simulator, x0, theta, drift, diffusion,
                                                         source, payoff, 0, nPaths, schedule));
}

/// Convenience: full path range, reduced estimate (checkpointed mode).
template <typename Scheme, typename DriftF, typename DiffusionF, typename Source, typename Payoff>
    requires Drift<DriftF, double> && Diffusion<DiffusionF, double> && RandomSource<Source> &&
             (Scheme::uniformStreams == 0 || UniformRandomSource<Source>) && PathPayoff<Payoff>
GradientEstimate simulateGradientCheckpointed(const SdeSimulator<double, Scheme>& simulator,
                                              const Eigen::VectorXd& x0,
                                              const std::vector<double>& theta, const DriftF& drift,
                                              const DiffusionF& diffusion, const Source& source,
                                              const Payoff& payoff, std::size_t nPaths,
                                              Schedule schedule = Schedule::Sequential) {
    return reduceGradientSamples(simulateGradientCheckpointedSamples(
        simulator, x0, theta, drift, diffusion, source, payoff, 0, nPaths, schedule));
}

} // namespace quantape::mc

#endif // QUANTAPE_MC_GRADIENTS_H
