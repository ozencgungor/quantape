#ifndef QUANTAPE_MC_GRADIENTS_H
#define QUANTAPE_MC_GRADIENTS_H

#include "quantape/math/StanMath.h"

#include "quantape/mc/Parallel.h"
#include "quantape/mc/SdePrimitives.h"
#include "quantape/mc/SdeSimulator.h"

#include <Eigen/Dense>

#include <algorithm>
#include <cmath>
#include <concepts>
#include <cstddef>
#include <limits>
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
    /// Terminal-seed block. It holds `nSteps + 1` states so that any legal grid
    /// index is defined (a payoff indexing `states[k]` must not be UB); all are
    /// zero except the last, which is overwritten per path. The block is only
    /// used to seed the adjoint, and that is exact solely for a terminal payoff
    /// — see the value check in the stepping loop.
    PathBlock<var> termPath;
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
        termPath.resize(static_cast<std::size_t>(nDims), nSteps, nPaths);
        for (StateMatrix<var>& state : termPath.states) {
            state.setZero();
        }
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
        const Eigen::Index col = static_cast<Eigen::Index>(path);
        // In-place per-sensitivity pass: no (d + p) temporary per path.
        for (Eigen::Index i = 0; i < static_cast<Eigen::Index>(nSensitivities); ++i) {
            const double dg = samples.gradients(i, col) - estimate.gradient(i);
            sumGradSq(i) += dg * dg;
        }
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
    quantape::mc::requireSharedTheta(simulator.theta(), theta);
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
    quantape::mc::requireSharedTheta(simulator.theta(), theta);
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
                ws.termPath.states.back().col(0) = ws.fwdPath.states.back();
                var pi = payoff.template operator()<var>(ws.termPath);
                // Checkpointing seeds the adjoint from the terminal state alone,
                // so it is exact only when the payoff is terminal. Evaluate the
                // payoff on the seed block and compare with the value taken from
                // the full forward path: a path-dependent payoff differs and is
                // rejected here instead of silently returning the terminal-only
                // gradient. (Poisoning the block with NaN would not work: a
                // linear payoff still receives finite adjoints, because adjoints
                // do not depend on intermediate values.)
                if (!(pi.val() == samples.values[offset])) {
                    throw std::invalid_argument(
                        "simulateGradientCheckpointed: the payoff depends on non-terminal "
                        "states; use simulateGradient (full tape) instead");
                }
                pi.grad();
                for (std::size_t i = 0; i < d; ++i) {
                    ws.lambda(static_cast<Eigen::Index>(i)) =
                        ws.termPath.states.back()(static_cast<Eigen::Index>(i), 0).adj();
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

/**
 * @brief Per-interval pathwise gradient samples — time-dependent parameters
 *
 * `dV/dtheta_k` for every grid interval `k`, where `theta_k` is the parameter
 * vector the simulator applies on `[t_k, t_{k+1})` (see `SdeSimulator`). One
 * per-step `var` leaf per parameter is created on each path's tape, so a single
 * reverse sweep yields every interval's adjoint:
 *
 *     dV/dtheta_k = B_k^T lambda_{k+1},     B_k = dPhi_k/dtheta_k
 *
 * which is exact for the discretised model (the adjoint recursion
 * `lambda_k = (dpi/dX_k) + A_k^T lambda_{k+1}` is the chain rule, not an
 * approximation). The parameters are taken from the simulator, so the AD model
 * and the simulated model are the same object by construction; an empty
 * per-step theta is rejected because there would be nothing to differentiate.
 *
 * Output is aggregate rather than per-path: the raw object is `p x nSteps` per
 * path, which is `O(p*nSteps*nPaths)` and infeasible at production shapes. Each
 * block accumulates its own partial sums in a fixed block index, so the
 * reduction is deterministic and shard-mergeable.
 *
 * For any parameterisation `theta_k = f_k(phi)`, the gradient follows by the
 * chain rule: `dV/dphi = sum_k (df_k/dphi)^T (dV/dtheta_k)`. For a genuinely
 * shared parameter the sum over `k` is the shared derivative; for a piecewise
 * model it is the *uniform-shift* derivative, which is a different quantity.
 */
struct IntervalGradientEstimate {
    Eigen::MatrixXd mean;     ///< (p x nSteps): mean dV/dtheta_k
    Eigen::MatrixXd stdError; ///< (p x nSteps)
    double value = 0.0;       ///< mean payoff on the same paths
    double valueStdError = 0.0;
    std::size_t nPaths = 0;
    std::size_t nParameters = 0;
    std::size_t nSteps = 0;
};

namespace detail {

/// Per-worker workspace for the per-interval kernel: step-indexed leaves, one
/// path buffer, and reused draw/scratch buffers (no per-path allocation).
template <typename Scheme>
struct IntervalWorkspace {
    using var = stan::math::var;
    PathBlock<var> path;
    Eigen::Matrix<var, Eigen::Dynamic, 1> x0Vars;
    std::vector<std::vector<var>> thetaVars;
    Eigen::MatrixXd z;
    Eigen::MatrixXd uniforms;
    typename Scheme::template Scratch<var> scratch;
    Eigen::MatrixXd grad; ///< (p x nSteps) per-path interval gradient scratch

    IntervalWorkspace(Eigen::Index nDims, std::size_t nSteps, std::size_t nTheta,
                      Eigen::Index nPaths)
        : thetaVars(nSteps, std::vector<var>(nTheta)), scratch(nDims, nPaths) {
        x0Vars.resize(nDims);
        path.resize(static_cast<std::size_t>(nDims), nSteps, 1);
    }
};

} // namespace detail

template <typename Scheme, typename DriftF, typename DiffusionF, typename Source, typename Payoff>
    requires Drift<DriftF, double> && Diffusion<DiffusionF, double> && RandomSource<Source> &&
             (Scheme::uniformStreams == 0 || UniformRandomSource<Source>) && PathPayoff<Payoff>
IntervalGradientEstimate simulateIntervalGradient(const SdeSimulator<double, Scheme>& simulator,
                                                  const Eigen::VectorXd& x0, const DriftF& drift,
                                                  const DiffusionF& diffusion, const Source& source,
                                                  const Payoff& payoff, std::size_t nPaths,
                                                  Schedule schedule = Schedule::Sequential) {
    using stan::math::var;
    if (nPaths == 0) {
        throw std::invalid_argument("simulateIntervalGradient: nPaths must be positive");
    }
    const std::size_t d = static_cast<std::size_t>(x0.size());
    const std::size_t nSteps = simulator.timeGrid().nSteps();
    const auto& thetaSteps = simulator.theta();
    if (thetaSteps.empty() || thetaSteps[0].empty()) {
        throw std::invalid_argument(
            "simulateIntervalGradient: the simulator must carry a non-empty per-interval theta");
    }
    const std::size_t p = thetaSteps[0].size();
    for (std::size_t k = 0; k < nSteps; ++k) {
        if (thetaSteps[k].size() != p) {
            throw std::invalid_argument(
                "simulateIntervalGradient: per-interval theta must have a uniform width");
        }
    }

    const std::size_t kBlock = 1024;
    const std::size_t nBlocks = (nPaths + kBlock - 1) / kBlock;
    std::vector<Eigen::MatrixXd> blockSum(nBlocks);
    std::vector<Eigen::MatrixXd> blockSumSq(nBlocks);
    std::vector<double> blockValue(nBlocks, 0.0);
    std::vector<double> blockValueSq(nBlocks, 0.0);

    const auto makeWorkspace = [&] {
        return detail::IntervalWorkspace<Scheme>(static_cast<Eigen::Index>(d), nSteps, p, 1);
    };
    detail::parallelForWithLocal(
        nBlocks, schedule, makeWorkspace,
        [&](std::size_t block, detail::IntervalWorkspace<Scheme>& ws) {
            Eigen::MatrixXd sum = Eigen::MatrixXd::Zero(static_cast<Eigen::Index>(p),
                                                        static_cast<Eigen::Index>(nSteps));
            Eigen::MatrixXd sumSq = sum;
            double valueSum = 0.0;
            double valueSq = 0.0;
            const std::size_t begin = block * kBlock;
            const std::size_t end = std::min(begin + kBlock, nPaths);
            for (std::size_t pathIndex = begin; pathIndex < end; ++pathIndex) {
                stan::math::nested_rev_autodiff scope;
                for (std::size_t k = 0; k < nSteps; ++k) {
                    for (std::size_t j = 0; j < p; ++j) {
                        ws.thetaVars[k][j] = thetaSteps[k][j];
                    }
                }
                for (std::size_t i = 0; i < d; ++i) {
                    ws.x0Vars(static_cast<Eigen::Index>(i)) = x0(static_cast<Eigen::Index>(i));
                }
                ws.path.states[0].col(0) = ws.x0Vars;
                for (std::size_t k = 0; k < nSteps; ++k) {
                    simulator.template stepPath<var>(
                        ws.path.states[k], ws.path.states[k + 1], k, pathIndex, source, drift,
                        diffusion, ws.thetaVars[k], ws.z, ws.uniforms, ws.scratch);
                }
                var y = payoff.template operator()<var>(ws.path);
                y.grad();
                const double value = y.val();
                valueSum += value;
                valueSq += value * value;
                for (std::size_t k = 0; k < nSteps; ++k) {
                    for (std::size_t j = 0; j < p; ++j) {
                        const double g = ws.thetaVars[k][j].adj();
                        sum(static_cast<Eigen::Index>(j), static_cast<Eigen::Index>(k)) += g;
                        sumSq(static_cast<Eigen::Index>(j), static_cast<Eigen::Index>(k)) += g * g;
                    }
                }
            }
            blockSum[block] = sum;
            blockSumSq[block] = sumSq;
            blockValue[block] = valueSum;
            blockValueSq[block] = valueSq;
        });

    IntervalGradientEstimate estimate;
    estimate.mean =
        Eigen::MatrixXd::Zero(static_cast<Eigen::Index>(p), static_cast<Eigen::Index>(nSteps));
    estimate.stdError = estimate.mean;
    estimate.nPaths = nPaths;
    estimate.nParameters = p;
    estimate.nSteps = nSteps;
    for (std::size_t b = 0; b < nBlocks; ++b) {
        estimate.mean += blockSum[b];
        estimate.stdError += blockSumSq[b];
        estimate.value += blockValue[b];
        estimate.valueStdError += blockValueSq[b];
    }
    const double n = static_cast<double>(nPaths);
    estimate.mean /= n;
    estimate.value /= n;
    estimate.valueStdError = nPaths > 1
                                 ? std::sqrt(std::max(0.0, estimate.valueStdError / n -
                                                               estimate.value * estimate.value) /
                                             (n - 1.0) / n)
                                 : 0.0;
    for (Eigen::Index j = 0; j < estimate.mean.rows(); ++j) {
        for (Eigen::Index k = 0; k < estimate.mean.cols(); ++k) {
            const double m = estimate.mean(j, k);
            const double second = estimate.stdError(j, k) / n;
            estimate.stdError(j, k) =
                nPaths > 1 ? std::sqrt(std::max(0.0, (second - m * m) / (n - 1.0)) / n) : 0.0;
        }
    }
    return estimate;
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

/**
 * @brief Contracted pathwise gradient samples — time-dependent parameters
 *
 * Mode B of the per-interval design. The raw per-interval gradient is
 * `p x nSteps` *per path*, which is `O(p*nSteps*nPaths)` and infeasible at
 * production shapes, so it is contracted in the kernel instead: given
 * `W_k in R^{p x q}` the sample is
 *
 *     h^(i) = sum_k W_k^T G^(i)[:,k],        G^(i)[:,k] = dV/dtheta_k |path i
 *
 * which is exactly `dV/dphi` for any parameterisation `theta_k = f_k(phi)`
 * with `W_k = df_k/dphi`. Storing the per-path `h` (not the raw `G`) keeps the
 * usual `(d + q) x nPaths` sample layout, so the reduction, the standard error,
 * the CRN/FD gates and shard merging all work unchanged.
 *
 * The contraction must be done *per path*: `Var(sum_k c_k g_k)` is not
 * `sum_k c_k^2 Var(g_k)`, so contracting after reducing would drop the
 * cross-interval covariance and understate the standard error. This entry point
 * is the one to use whenever an SE is wanted on a parameterisation.
 *
 * `contraction(step)` returns the `p x q` weight block for interval `step`.
 */
template <typename Scheme, typename DriftF, typename DiffusionF, typename Source, typename Payoff,
          typename Contraction>
    requires Drift<DriftF, double> && Diffusion<DiffusionF, double> && RandomSource<Source> &&
             (Scheme::uniformStreams == 0 || UniformRandomSource<Source>) && PathPayoff<Payoff>
GradientSamples simulateContractedGradientSamples(const SdeSimulator<double, Scheme>& simulator,
                                                  const Eigen::VectorXd& x0, const DriftF& drift,
                                                  const DiffusionF& diffusion, const Source& source,
                                                  const Payoff& payoff,
                                                  const Contraction& contraction,
                                                  std::size_t pathBegin, std::size_t nPaths,
                                                  Schedule schedule = Schedule::Sequential) {
    using stan::math::var;
    if (nPaths == 0) {
        throw std::invalid_argument("simulateContractedGradientSamples: nPaths must be positive");
    }
    const std::size_t d = static_cast<std::size_t>(x0.size());
    const std::size_t nSteps = simulator.timeGrid().nSteps();
    const auto& thetaSteps = simulator.theta();
    if (thetaSteps.empty() || thetaSteps[0].empty()) {
        throw std::invalid_argument(
            "simulateContractedGradientSamples: the simulator must carry a non-empty "
            "per-interval theta");
    }
    const std::size_t p = thetaSteps[0].size();
    for (std::size_t k = 0; k < nSteps; ++k) {
        if (thetaSteps[k].size() != p) {
            throw std::invalid_argument(
                "simulateContractedGradientSamples: per-interval theta must have a uniform width");
        }
    }
    // Validate the weight blocks before touching the tape.
    std::vector<Eigen::MatrixXd> weights(nSteps);
    std::size_t q = 0;
    for (std::size_t k = 0; k < nSteps; ++k) {
        weights[k] = contraction(k);
        if (weights[k].rows() != static_cast<Eigen::Index>(p) || weights[k].cols() == 0) {
            throw std::invalid_argument(
                "simulateContractedGradientSamples: each weight block must be p x q with q > 0");
        }
        if (k == 0) {
            q = static_cast<std::size_t>(weights[k].cols());
        } else if (static_cast<std::size_t>(weights[k].cols()) != q) {
            throw std::invalid_argument(
                "simulateContractedGradientSamples: the weight blocks must share a column count");
        }
    }

    GradientSamples samples;
    samples.pathBegin = pathBegin;
    samples.nStateDims = d;
    samples.nParameters = q;
    samples.values.assign(nPaths, 0.0);
    samples.gradients =
        Eigen::MatrixXd::Zero(static_cast<Eigen::Index>(d + q), static_cast<Eigen::Index>(nPaths));

    const auto makeWorkspace = [&] {
        detail::IntervalWorkspace<Scheme> ws(static_cast<Eigen::Index>(d), nSteps, p, 1);
        ws.grad.resize(static_cast<Eigen::Index>(p), static_cast<Eigen::Index>(nSteps));
        return ws;
    };
    detail::parallelForWithLocal(
        nPaths, schedule, makeWorkspace,
        [&](std::size_t offset, detail::IntervalWorkspace<Scheme>& ws) {
            const std::size_t pathIndex = pathBegin + offset;
            stan::math::nested_rev_autodiff scope;
            for (std::size_t k = 0; k < nSteps; ++k) {
                for (std::size_t j = 0; j < p; ++j) {
                    ws.thetaVars[k][j] = thetaSteps[k][j];
                }
            }
            for (std::size_t i = 0; i < d; ++i) {
                ws.x0Vars(static_cast<Eigen::Index>(i)) = x0(static_cast<Eigen::Index>(i));
            }
            ws.path.states[0].col(0) = ws.x0Vars;
            for (std::size_t k = 0; k < nSteps; ++k) {
                simulator.template stepPath<var>(ws.path.states[k], ws.path.states[k + 1], k,
                                                 pathIndex, source, drift, diffusion,
                                                 ws.thetaVars[k], ws.z, ws.uniforms, ws.scratch);
            }
            var y = payoff.template operator()<var>(ws.path);
            y.grad();
            samples.values[offset] = y.val();
            for (std::size_t k = 0; k < nSteps; ++k) {
                for (std::size_t j = 0; j < p; ++j) {
                    ws.grad(static_cast<Eigen::Index>(j), static_cast<Eigen::Index>(k)) =
                        ws.thetaVars[k][j].adj();
                }
            }
            for (std::size_t i = 0; i < d; ++i) {
                samples.gradients(static_cast<Eigen::Index>(i), static_cast<Eigen::Index>(offset)) =
                    ws.x0Vars(static_cast<Eigen::Index>(i)).adj();
            }
            for (std::size_t jq = 0; jq < q; ++jq) {
                double acc = 0.0;
                for (std::size_t k = 0; k < nSteps; ++k) {
                    for (std::size_t jp = 0; jp < p; ++jp) {
                        acc += weights[k](static_cast<Eigen::Index>(jp),
                                          static_cast<Eigen::Index>(jq)) *
                               ws.grad(static_cast<Eigen::Index>(jp), static_cast<Eigen::Index>(k));
                    }
                }
                samples.gradients(static_cast<Eigen::Index>(d + jq),
                                  static_cast<Eigen::Index>(offset)) = acc;
            }
        });
    return samples;
}

/// Convenience: full path range, reduced contracted estimate.
template <typename Scheme, typename DriftF, typename DiffusionF, typename Source, typename Payoff,
          typename Contraction>
    requires Drift<DriftF, double> && Diffusion<DiffusionF, double> && RandomSource<Source> &&
             (Scheme::uniformStreams == 0 || UniformRandomSource<Source>) && PathPayoff<Payoff>
GradientEstimate
simulateContractedGradient(const SdeSimulator<double, Scheme>& simulator, const Eigen::VectorXd& x0,
                           const DriftF& drift, const DiffusionF& diffusion, const Source& source,
                           const Payoff& payoff, const Contraction& contraction, std::size_t nPaths,
                           Schedule schedule = Schedule::Sequential) {
    return reduceGradientSamples(simulateContractedGradientSamples(
        simulator, x0, drift, diffusion, source, payoff, contraction, 0, nPaths, schedule));
}

} // namespace quantape::mc

#endif // QUANTAPE_MC_GRADIENTS_H
