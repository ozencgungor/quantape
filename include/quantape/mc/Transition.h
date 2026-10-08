#ifndef QUANTAPE_MC_TRANSITION_H
#define QUANTAPE_MC_TRANSITION_H

#include "quantape/mc/Schemes.h"
#include "quantape/mc/SdePrimitives.h"

#include <Eigen/Dense>

#include <concepts>
#include <cstddef>
#include <utility>
#include <vector>

namespace quantape::mc {
/**
 * @file Transition.h
 * @brief Optional exact / conditional transition hook
 *
 * The default stepping contract is a discretization *scheme* (`Schemes.h`):
 * the process supplies drift/diffusion fields and the scheme advances one grid
 * interval. A process may instead expose
 *
 *     process.transition(x, t, dt, z, theta, out)
 *
 * with `out(:, p)` drawn from the process's transition law for `X(t + dt)`
 * given `x(:, p)`, where `z` is the standardized Gaussian factor block
 * `(q x nPaths)` delivered by the source (pinned noise, exactly as for a
 * scheme; the process casts it to its scalar). Two documented semantics:
 *
 *   - **exact**: `out` is a sample of the exact conditional law of
 *     `X(t + dt) | X(t)` — no discretization bias (e.g. the exponential of a
 *     Gaussian factor model, or a linear SDE with exact Gaussian transition);
 *   - **conditional-on-vol**: `out` is exact only conditionally on the
 *     realized auxiliary state already carried in `x` (typically a stochastic
 *     variance component): given that component the asset transition is the
 *     exact Gaussian law with its integrated variance. The auxiliary
 *     component itself is advanced by the same hook.
 *
 * `TransitionScheme<Process, Fallback>` is the opt-in adapter: it satisfies
 * the scheme interface and calls `Process::transition` for every `Scalar` the
 * process supports, falling back to `Fallback` (default `Euler`) otherwise.
 * The `if constexpr` probe is per scalar type, so processes without a hook —
 * or hooks implemented only for `double` while pricing under `var` — keep the
 * existing scheme behavior unchanged. The hook is scalar-generic (`double`,
 * `var`, `fvar<var>`) and allocation-free: it writes into the caller-owned
 * `out` block, and only consumes Gaussian factor draws (no uniform streams in
 * this version; `uniformStreams` follows the fallback).
 */

/// Exact/conditional transition hook: `out(:, p)` is drawn from the
/// transition law of `X(t + dt)` given `x(:, p)` using standardized draws `z`.
template <typename P, typename Scalar>
concept Transition = requires(const P& p, const StateMatrix<Scalar>& x, double t, double dt,
                              const Eigen::MatrixXd& z, const std::vector<Scalar>& theta,
                              StateMatrix<Scalar>& out) { p.transition(x, t, dt, z, theta, out); };

/// Scheme adapter: use the process's exact/conditional transition for scalars
/// it implements, otherwise delegate to `Fallback`.
template <typename Process, typename Fallback = Euler>
struct TransitionScheme {
    Process process;
    Fallback fallback;

    static constexpr std::size_t uniformStreams = Fallback::uniformStreams;

    template <typename Scalar>
    struct Scratch {
        typename Fallback::template Scratch<Scalar> fallback;
        Scratch(Eigen::Index nDims, Eigen::Index nPaths) : fallback(nDims, nPaths) {}
    };

    template <typename Scalar, typename DriftF, typename DiffusionF>
        requires Drift<DriftF, Scalar> && Diffusion<DiffusionF, Scalar>
    void step(const StateMatrix<Scalar>& x, StateMatrix<Scalar>& xNext, double t, double dt,
              const Eigen::MatrixXd& z, const Eigen::MatrixXd& uniforms, const DriftF& drift,
              const DiffusionF& diffusion, const std::vector<Scalar>& theta,
              Scratch<Scalar>& scratch) const {
        if constexpr (Transition<Process, Scalar>) {
            process.transition(x, t, dt, z, theta, xNext);
        } else {
            fallback.step(x, xNext, t, dt, z, uniforms, drift, diffusion, theta, scratch.fallback);
        }
    }
};

/// Convenience construction with a process and an optional fallback scheme.
template <typename Process, typename Fallback = Euler>
TransitionScheme<Process, Fallback> transitionScheme(Process process,
                                                     Fallback fallback = Fallback{}) {
    return TransitionScheme<Process, Fallback>{std::move(process), std::move(fallback)};
}

} // namespace quantape::mc

#endif // QUANTAPE_MC_TRANSITION_H
