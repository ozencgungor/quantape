#ifndef QUANTAPE_MC_SCHEMES_H
#define QUANTAPE_MC_SCHEMES_H

#include "quantape/mc/SdePrimitives.h"

#include <Eigen/Dense>

#include <cmath>
#include <cstddef>

namespace quantape::mc {
/**
 * @file Schemes.h
 * @brief Discretization schemes (stateless policies, Stan-free core)
 *
 * A scheme advances one block by one grid interval, given the current
 * state block X (d x B), the step context (t_k, dt_k from the grid), the
 * standardized noise block Z (q x B from the source) and the model
 * functors; it applies all dt-dependent scaling and writes X_{k+1}.
 *
 * ## Scheme interface
 *
 *   static constexpr std::size_t uniformStreams = 0;
 *   template <typename Scalar> struct Scratch { ... };   // reused buffers
 *   template <typename Scalar, typename DriftF, typename DiffusionF>
 *   void step(const StateMatrix<Scalar>& x, StateMatrix<Scalar>& xNext,
 *             double t, double dt, const Eigen::MatrixXd& z,
 *             const Eigen::MatrixXd& uniforms, const DriftF& drift,
 *             const DiffusionF& diffusion, const std::vector<Scalar>& theta,
 *             Scratch<Scalar>& scratch) const;
 *
 * Schemes are cheap values held by the simulator (moment-matching schemes
 * carry a moments functor). `uniformStreams` tells the engine how many
 * uniform streams to request from the source (0 for diffusion schemes).
 * The simulator constructs `Scratch(nDims, blockPaths)` once per block and
 * reuses it across steps (zero per-step allocation); derivative buffers
 * (Milstein) live in the scheme's Scratch (see SchemesStan.h).
 */

/// Euler-Maruyama (explicit, strong order 1/2 in general)
struct Euler {
    template <typename Scalar>
    struct Scratch {
        StateMatrix<Scalar> driftBuf;
        StateMatrix<Scalar> gBuf;
        Scratch(Eigen::Index nDims, Eigen::Index nPaths)
            : driftBuf(nDims, nPaths), gBuf(nDims, nPaths) {}
    };

    /**
     * X_{k+1}(:,p) = X_k(:,p)
     *              + f(X_k(:,p), t_k, theta) dt
     *              + sum_j g_j(X_k(:,p), t_k, theta) sqrt(dt) Z(j,p)
     */
    static constexpr std::size_t uniformStreams = 0;

    template <typename Scalar, typename DriftF, typename DiffusionF>
        requires Drift<DriftF, Scalar> && Diffusion<DiffusionF, Scalar>
    void step(const StateMatrix<Scalar>& x, StateMatrix<Scalar>& xNext, double t, double dt,
              const Eigen::MatrixXd& z, const Eigen::MatrixXd& /*uniforms*/, const DriftF& drift,
              const DiffusionF& diffusion, const std::vector<Scalar>& theta,
              Scratch<Scalar>& scratch) const {
        const std::size_t q = static_cast<std::size_t>(z.rows());

        drift(x, t, theta, scratch.driftBuf);
        xNext = x + Scalar(dt) * scratch.driftBuf;

        const Scalar sqrtDt(std::sqrt(dt));
        for (std::size_t j = 0; j < q; ++j) {
            diffusion(x, t, theta, j, scratch.gBuf);
            Eigen::Matrix<Scalar, 1, Eigen::Dynamic> zj =
                z.row(static_cast<Eigen::Index>(j)).template cast<Scalar>();
            zj *= sqrtDt;
            // xNext(:,p) += gBuf(:,p) * zj(p)
            xNext.array() += scratch.gBuf.array().rowwise() * zj.array();
        }
    }
};

/// Predictor-corrector (Heun/trapezoidal drift; diffusion at the left point)
struct PredictorCorrector {
    template <typename Scalar>
    struct Scratch {
        StateMatrix<Scalar> driftLeft;
        StateMatrix<Scalar> driftRight;
        StateMatrix<Scalar> gBuf;
        StateMatrix<Scalar> xPredict;
        Scratch(Eigen::Index nDims, Eigen::Index nPaths)
            : driftLeft(nDims, nPaths), driftRight(nDims, nPaths), gBuf(nDims, nPaths),
              xPredict(nDims, nPaths) {}
    };

    /**
     * Xtilde   = X_k + f(X_k, t_k, theta) dt                       (predictor)
     * X_{k+1}  = X_k + 1/2 (f(X_k, t_k) + f(Xtilde, t_k + dt)) dt  (corrector)
     *                + sum_j g_j(X_k, t_k, theta) sqrt(dt) Z(j,p)
     */
    static constexpr std::size_t uniformStreams = 0;

    template <typename Scalar, typename DriftF, typename DiffusionF>
        requires Drift<DriftF, Scalar> && Diffusion<DiffusionF, Scalar>
    void step(const StateMatrix<Scalar>& x, StateMatrix<Scalar>& xNext, double t, double dt,
              const Eigen::MatrixXd& z, const Eigen::MatrixXd& /*uniforms*/, const DriftF& drift,
              const DiffusionF& diffusion, const std::vector<Scalar>& theta,
              Scratch<Scalar>& scratch) const {
        const std::size_t q = static_cast<std::size_t>(z.rows());

        drift(x, t, theta, scratch.driftLeft);
        scratch.xPredict = x + Scalar(dt) * scratch.driftLeft;
        drift(scratch.xPredict, t + dt, theta, scratch.driftRight);
        xNext = x + Scalar(0.5 * dt) * (scratch.driftLeft + scratch.driftRight);

        const Scalar sqrtDt(std::sqrt(dt));
        for (std::size_t j = 0; j < q; ++j) {
            diffusion(x, t, theta, j, scratch.gBuf);
            Eigen::Matrix<Scalar, 1, Eigen::Dynamic> zj =
                z.row(static_cast<Eigen::Index>(j)).template cast<Scalar>();
            zj *= sqrtDt;
            xNext.array() += scratch.gBuf.array().rowwise() * zj.array();
        }
    }
};

} // namespace quantape::mc

#endif // QUANTAPE_MC_SCHEMES_H
