#ifndef QUANTAPE_MC_SCHEMES_STAN_H
#define QUANTAPE_MC_SCHEMES_STAN_H

#include "quantape/math/StanMath.h"

#include "quantape/mc/SdePrimitives.h"

#include <cmath>
#include <cstddef>
#include <vector>

namespace quantape::mc {
/**
 * @file SchemesStan.h
 * @brief Derivative-based schemes (Stan AD): Milstein via forward-mode fvar
 *
 * ## Milstein (diagonal / commutative-noise form)
 *
 *     X_{k+1}(:,p) = X_k(:,p)
 *                  + f(X_k) dt
 *                  + sum_j g_j(X_k) sqrt(dt) Z(j,p)
 *                  + 1/2 sum_j (Dg_j . g_j)(X_k) ( dt Z(j,p)^2 - dt )
 *
 * where `Dg_j . g_j` is the directional derivative of g_j along g_j,
 * obtained exactly by ONE forward-mode sweep per factor: the state is
 * dualised as
 *
 *     xDual(i,p) = fvar<Scalar>( X_k(i,p), g_j(X_k)(i,p) )
 *
 * and the tangent part of g_j(xDual) is the required derivative — no tape,
 * no step size, one model evaluation per factor.
 *
 * Strong order 1 for *commutative* noise (scalar GBM, diagonal models).
 * For non-commutative multi-factor noise the Levy-area terms are omitted
 * (the scheme then degrades to strong order 1/2 in those directions) —
 * documented restriction, not a bug.
 */
struct Milstein {
    template <typename Scalar>
    struct Scratch {
        using Dual = stan::math::fvar<Scalar>;
        StateMatrix<Scalar> driftBuf;
        StateMatrix<Scalar> gBuf;
        StateMatrix<Dual> xDual;
        StateMatrix<Dual> gDual;
        std::vector<Dual> thetaDual;
        Scratch(Eigen::Index nDims, Eigen::Index nPaths)
            : driftBuf(nDims, nPaths), gBuf(nDims, nPaths), xDual(nDims, nPaths),
              gDual(nDims, nPaths) {}
    };

    static constexpr std::size_t uniformStreams = 0;

    template <typename Scalar, typename DriftF, typename DiffusionF>
        requires Drift<DriftF, Scalar> && Diffusion<DiffusionF, Scalar>
    void step(const StateMatrix<Scalar>& x, StateMatrix<Scalar>& xNext, double t, double dt,
              const Eigen::MatrixXd& z, const Eigen::MatrixXd& /*uniforms*/, const DriftF& drift,
              const DiffusionF& diffusion, const std::vector<Scalar>& theta,
              Scratch<Scalar>& scratch) const {
        using Dual = typename Scratch<Scalar>::Dual;
        const std::size_t q = static_cast<std::size_t>(z.rows());
        const Eigen::Index nDims = x.rows();
        const Eigen::Index nPaths = x.cols();

        drift(x, t, theta, scratch.driftBuf);
        xNext = x + Scalar(dt) * scratch.driftBuf;

        scratch.thetaDual.clear();
        scratch.thetaDual.reserve(theta.size());
        for (const Scalar& th : theta) {
            scratch.thetaDual.emplace_back(th);
        }

        const double sqrtDt = std::sqrt(dt);
        for (std::size_t j = 0; j < q; ++j) {
            diffusion(x, t, theta, j, scratch.gBuf);

            // Dualise the state along g_j(x); tangent part of g_j(xDual)
            // is the directional derivative Dg_j . g_j.
            for (Eigen::Index p = 0; p < nPaths; ++p) {
                for (Eigen::Index i = 0; i < nDims; ++i) {
                    scratch.xDual(i, p) = Dual(x(i, p), scratch.gBuf(i, p));
                }
            }
            diffusion(scratch.xDual, t, scratch.thetaDual, j, scratch.gDual);

            for (Eigen::Index p = 0; p < nPaths; ++p) {
                const double zj = z(static_cast<Eigen::Index>(j), p);
                const Scalar noiseScale(sqrtDt * zj);
                const Scalar milsteinScale(0.5 * (dt * zj * zj - dt));
                for (Eigen::Index i = 0; i < nDims; ++i) {
                    xNext(i, p) +=
                        scratch.gBuf(i, p) * noiseScale + scratch.gDual(i, p).d_ * milsteinScale;
                }
            }
        }
    }
};

} // namespace quantape::mc

#endif // QUANTAPE_MC_SCHEMES_STAN_H
