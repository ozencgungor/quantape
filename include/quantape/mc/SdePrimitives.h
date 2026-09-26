#ifndef QUANTAPE_MC_SDE_PRIMITIVES_H
#define QUANTAPE_MC_SDE_PRIMITIVES_H

#include <Eigen/Dense>

#include <concepts>
#include <cstddef>
#include <vector>

namespace quantape::mc {
/**
 * @file SdePrimitives.h
 * @brief Types and concepts for the SDE simulation engine
 *
 * ## Mathematical specification
 *
 * The engine simulates an Ito SDE in factor form
 *
 *     dX(t) = f(X(t), t, theta) dt + sum_{j=1..q} g_j(X(t), t, theta) dW_j(t),
 *     X in R^d, W = (W_1, ..., W_q) independent standard Brownian motions,
 *
 * on a grid t_0 < t_1 < ... < t_N (`TimeGrid`). The drift f and the q
 * factor diffusion fields g_j are user functors; theta is piecewise
 * constant on the grid, one parameter vector per interval
 * [t_k, t_{k+1}] (time-dependent parameters fall out).
 *
 * ## Batch (vectorized) layout
 *
 * State is a batch matrix, one *path per column*:
 *
 *     StateMatrix<Scalar> X,  shape (d x B):  X(i, p) = X_i^{(p)}
 *
 * so a step updates all B paths with Eigen-vectorized column operations.
 * The functors are batch and scalar-generic over the element type:
 *
 *     drift(x, t, theta, out):
 *         out(:, p) = f(x(:, p), t, theta)          for every path p
 *     diffusion(x, t, theta, j, out):
 *         out(:, p) = g_j(x(:, p), t, theta)        for every path p
 *
 * A process provides f and g_j as named members (a "process bundle":
 * `process.drift(...)`, `process.diffusion(...)`); `driftOf(model)` /
 * `diffusionOf(model)` adapt it to the two functors the engine takes.
 *
 * Factor decomposition: a general matrix diffusion g (d x q) enters as the
 * q vector fields g_j = g e_j (the columns), which is what a Cholesky or
 * PCA factorization produces; models with genuinely matrix-valued g can
 * always be written this way, and the batch update stays path-vectorized
 * (X += sum_j g_j * sqrt(dt) z_j, z_j the standardized factor draw).
 *
 * Note: the low-level normal samplers live in `quantape::math::mc`
 * (ZigguratNormal/McFarlandNormal); this namespace (`quantape::mc`) is the
 * simulation engine.
 */

/// Batch state: (nDims x nPaths), one path per column.
template <typename Scalar = double>
using StateMatrix = Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>;

/**
 * @brief Storage for one block of simulated paths
 *
 * `states[k]` is the (nDims x nPaths) state block at grid time t_k, so a
 * block holds `nSteps + 1` matrices:
 *
 *     states[k](i, p) = X_i^{(p)}(t_k),   k = 0..N
 *
 * Path access within a step is a contiguous column (`states[k].col(p)`);
 * blocked evaluation keeps only the current block resident (the storage
 * contract for large path counts / large dimensions).
 */
template <typename Scalar = double>
struct PathBlock {
    std::size_t nDims = 0;
    std::size_t nPaths = 0;
    std::vector<StateMatrix<Scalar>> states; ///< nSteps + 1 matrices

    void resize(std::size_t nDimsIn, std::size_t nSteps, std::size_t nPathsIn) {
        nDims = nDimsIn;
        nPaths = nPathsIn;
        states.resize(nSteps + 1);
        for (StateMatrix<Scalar>& s : states) {
            s.resize(static_cast<Eigen::Index>(nDims), static_cast<Eigen::Index>(nPaths));
        }
    }

    std::size_t nSteps() const { return states.empty() ? 0 : states.size() - 1; }
};

/// Batch drift functor: `out(:, p) = f(x(:, p), t, theta)` for all paths p.
template <typename F, typename Scalar>
concept Drift =
    requires(const F& f, const StateMatrix<Scalar>& x, double t, const std::vector<Scalar>& theta,
             StateMatrix<Scalar>& out) { f(x, t, theta, out); };

/// Conditional moments functor for moment-matching schemes:
/// `out(:, p) = E[X_{k+1}(:,p) | X_k(:,p)]` and the corresponding variance
/// given (t_k, dt_k, theta).
template <typename M, typename Scalar>
concept Moments = requires(const M& m, const StateMatrix<Scalar>& x, double t, double dt,
                           const std::vector<Scalar>& theta, StateMatrix<Scalar>& out) {
    m.mean(x, t, dt, theta, out);
    m.variance(x, t, dt, theta, out);
};

/// Factor-wise diffusion functor: `out(:, p) = g_j(x(:, p), t, theta)` for
/// all paths p, `factor` selects j in {0, ..., q-1}.
template <typename G, typename Scalar>
concept Diffusion =
    requires(const G& g, const StateMatrix<Scalar>& x, double t, const std::vector<Scalar>& theta,
             std::size_t factor, StateMatrix<Scalar>& out) { g(x, t, theta, factor, out); };

/**
 * @brief Refinement: sources that can also deliver uniform draws
 *
 * Needed by moment-matching schemes (QE's exponential branch):
 * `fillUniform(step, pathBegin, nPaths, streamBegin, uniformStreams, out)`
 * writes a (uniformStreams x nPaths) block of U(0,1) draws from disjoint
 * keyed streams (independent of the Gaussian factors).
 */
template <typename S>
concept UniformRandomSource =
    requires(const S& s, std::size_t step, std::size_t pathBegin, std::size_t nPaths,
             std::size_t streamBegin, std::size_t uniformStreams, Eigen::MatrixXd& out) {
        s.fillUniform(step, pathBegin, nPaths, streamBegin, uniformStreams, out);
    };

/**
 * @brief Noise source: standardized factor-space draws, one step at a time
 *
 * Contract for `fill(step, pathBegin, nPaths, out)` writing a
 * (q x nPaths) block Z with
 *
 *     Z(j, p) = z_j^{(p)}(t_step),   E[z] = 0,  Var[z] = 1
 *
 * standardized in time: NO dt scaling (the scheme applies sqrt(dt_k) from
 * the grid); the source carries the dependence structure it was built for
 * (iid, correlated, factor-projected). The draws are a pure function of
 * (source state, step, path), so a block fill is bitwise identical to the
 * corresponding per-path fills — see RandomSource.h for the keying
 * construction that guarantees it.
 *
 * v1 sources are const and thread-safe; future stateful sources (Hawkes)
 * relax this with an explicit reset().
 */
template <typename S>
concept RandomSource = requires(const S& s, std::size_t step, std::size_t pathBegin,
                                std::size_t nPaths, Eigen::MatrixXd& out) {
    { s.factorCount() } -> std::convertible_to<std::size_t>;
    s.fill(step, pathBegin, nPaths, out);
};

// ── Process-bundle adapters ────────────────────────────────────────────
//
// A process bundle exposes named members (`drift`, `diffusion`, moments); the engine takes separate
// f/g functors, so these adapters perform the trivial lift
//
//     driftOf(m)(x, t, theta, out)            := m.drift(x, t, theta, out)
//     diffusionOf(m)(x, t, theta, j, out)     := m.diffusion(x, t, theta, j, out)
//
// captured by value: models are cheap, stateless descriptions.

template <typename M>
auto driftOf(const M& model) {
    return [model](const auto& x, double t, const auto& theta, auto& out) {
        model.drift(x, t, theta, out);
    };
}

template <typename M>
auto diffusionOf(const M& model) {
    return [model](const auto& x, double t, const auto& theta, std::size_t factor, auto& out) {
        model.diffusion(x, t, theta, factor, out);
    };
}

} // namespace quantape::mc

#endif // QUANTAPE_MC_SDE_PRIMITIVES_H
