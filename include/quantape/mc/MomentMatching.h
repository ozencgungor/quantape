#ifndef QUANTAPE_MC_MOMENT_MATCHING_H
#define QUANTAPE_MC_MOMENT_MATCHING_H

#include "quantape/mc/SdePrimitives.h"

#include <Eigen/Dense>

#include <cmath>
#include <cstddef>
#include <utility>
#include <vector>

namespace quantape::mc {
/**
 * @file MomentMatching.h
 * @brief Moment-matching schemes: Andersen's QE sampler and a generic 1-D
 * moment-matching step
 *
 * ## QE sampling rule (Andersen 2008)
 *
 * Given the conditional mean `m = E[X' | X]` and variance `s2 = Var[X' | X]`
 * of the next state, with `psi = s2 / m^2` and a standard normal `z` plus a
 * uniform `u`:
 *
 *  - Quadratic branch (`psi <= psiC`, default `psiC = 1.5`), positivity
 *    by construction:
 *
 *        b2 = 2/psi - 1 + sqrt(2/psi) * sqrt(2/psi - 1)
 *        a  = m / (1 + b2)
 *        X' = a (b + z)^2,   b = sqrt(b2)
 *
 *  - Exponential branch (`psi > psiC`), mass at zero:
 *
 *        p    = (psi - 1)/(psi + 1)
 *        beta = (1 - p)/m
 *        X'   = 0                                   if u <= p
 *               -log((1 - u)/(1 - p)) / beta        otherwise
 *
 * Both branches reproduce the input moments exactly:
 * `E[X'] = m`, `Var[X'] = s2`, and `X' >= 0` whenever `m >= 0`. This is why
 * QE is the standard positivity-preserving scheme for CIR/Heston variance.
 * When `m <= 0` the state is absorbing at zero (returns 0).
 *
 * ## Generic 1-D moment-matching scheme
 *
 * `MomentMatching1D<Moments>` applies the QE rule to a scalar state whose
 * conditional moments are supplied by a `Moments` functor
 * (`m.mean(x, t, dt, theta, out)`, `m.variance(...)`). It consumes
 * `uniformStreams = 1` uniform stream from the source (exponential branch)
 * and the first Gaussian factor for `z`; drift/diffusion are not used for
 * the moment-matched component.
 *
 * Multi-dimensional / coupled moment matching (Heston QE on (S, V)) is a
 * process-specific scheme built from `qeSample` (see processes/).
 */

/// Standard normal CDF (for consumers wanting uniforms from normals)
inline double normalCdf(double x) {
    return 0.5 * std::erfc(-x * M_SQRT1_2);
}

/// Andersen QE draw from conditional moments (m, s2) with z ~ N(0,1),
/// u ~ U(0,1). Exact in the first two moments; non-negative for m >= 0.
template <typename Scalar>
Scalar qeSample(const Scalar& m, const Scalar& s2, const Scalar& z, const Scalar& u,
                double psiC = 1.5) {
    using std::log;
    using std::sqrt;
    if (!(m > Scalar(0.0))) {
        return Scalar(0.0); // absorbing at zero
    }
    const Scalar psi = s2 / (m * m);
    if (psi <= Scalar(psiC)) {
        const Scalar twoOverPsi = Scalar(2.0) / psi;
        const Scalar b2 =
            twoOverPsi - Scalar(1.0) + sqrt(twoOverPsi) * sqrt(twoOverPsi - Scalar(1.0));
        const Scalar a = m / (Scalar(1.0) + b2);
        const Scalar t = sqrt(b2) + z;
        return a * t * t;
    }
    const Scalar p = (psi - Scalar(1.0)) / (psi + Scalar(1.0));
    const Scalar beta = (Scalar(1.0) - p) / m;
    if (u <= p) {
        return Scalar(0.0);
    }
    return -log((Scalar(1.0) - u) / (Scalar(1.0) - p)) / beta;
}

/**
 * @brief Generic 1-D moment-matching scheme (QE rule, positivity preserving)
 *
 * The model provides conditional moments through `Moments`; the scheme
 * draws z from the first Gaussian factor and u from the first uniform
 * stream. Exact conditional first/second moments at every grid step, so
 * the unconditional moment recursions match the model's exactly.
 */
template <typename Moments>
struct MomentMatching1D {
    static constexpr std::size_t uniformStreams = 1;

    Moments moments;

    explicit MomentMatching1D(Moments m = Moments{}) : moments(std::move(m)) {}

    template <typename Scalar>
    struct Scratch {
        StateMatrix<Scalar> meanBuf;
        StateMatrix<Scalar> varianceBuf;
        Scratch(Eigen::Index nDims, Eigen::Index nPaths)
            : meanBuf(nDims, nPaths), varianceBuf(nDims, nPaths) {}
    };

    template <typename Scalar, typename DriftF, typename DiffusionF>
        requires Drift<DriftF, Scalar> && Diffusion<DiffusionF, Scalar>
    void step(const StateMatrix<Scalar>& x, StateMatrix<Scalar>& xNext, double t, double dt,
              const Eigen::MatrixXd& z, const Eigen::MatrixXd& uniforms, const DriftF& /*drift*/,
              const DiffusionF& /*diffusion*/, const std::vector<Scalar>& theta,
              Scratch<Scalar>& scratch) const {
        if (x.rows() != 1) {
            throw std::invalid_argument("MomentMatching1D: the state must be one-dimensional");
        }
        moments.mean(x, t, dt, theta, scratch.meanBuf);
        moments.variance(x, t, dt, theta, scratch.varianceBuf);
        for (Eigen::Index p = 0; p < x.cols(); ++p) {
            xNext(0, p) = qeSample(scratch.meanBuf(0, p), scratch.varianceBuf(0, p),
                                   Scalar(z(0, p)), Scalar(uniforms(0, p)));
        }
    }
};

} // namespace quantape::mc

#endif // QUANTAPE_MC_MOMENT_MATCHING_H
