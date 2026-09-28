#ifndef QUANTAPE_MODELS_HESTON_QE_PROCESS_H
#define QUANTAPE_MODELS_HESTON_QE_PROCESS_H

#include "quantape/mc/MomentMatching.h"
#include "quantape/mc/SdePrimitives.h"
#include "quantape/mc/processes/SdeProcesses.h"

#include <Eigen/Dense>

#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <utility>
#include <vector>

namespace quantape::processes {
/**
 * @file HestonQeProcess.h
 * @brief Andersen QE scheme for Heston process, state X = (lnS, V)
 *
 * Process-specific coupled scheme built from `mc::qeSample`; it is the
 * variance-optimal positivity-preserving integrator for Heston.
 *
 * ## Specification
 *
 * 1. Variance (exact conditional-moment matching): with the CIR moments
 *    `m = E[V'|V]`, `s2 = Var[V'|V]` (supplied by the bundle's
 *    `varianceMoments`),
 *
 *        V' = qeSample(m, s2, z_0, u_0)
 *
 *    exact in the first two conditional moments, `V' >= 0` by construction.
 *
 * 2. Log-price (exact Brownian decomposition with trapezoidal integrated
 *    variance `Vbar = (V + V')/2`): using
 *    `sqrt(V) dW_V = (dV - kappa(level - V) dt)/eta`,
 *
 *        lnS' = lnS + mu dt
 *             + (rho/eta)(V' - V - kappa level dt)
 *             + (kappa rho/eta - 1/2) Vbar dt
 *             + sqrt((1 - rho^2) Vbar dt) z_perp
 *
 *    The correlation enters through the realized variance increment
 *    exactly; `z_perp` is an independent standard normal (Gaussian factor
 *    1), while the variance sampling consumes Gaussian factor 0 and
 *    uniform stream 0.
 *
 * Positivity/Feller: works for any parameters, including the
 * Feller-violating regime (2 kappa level < eta^2) where Euler on the
 * variance needs truncation.
 */
template <typename Process = HestonProcess>
struct HestonQeProcess {
    static constexpr std::size_t uniformStreams = 1;

    Process process;

    explicit HestonQeProcess(Process p = Process{}) : process(std::move(p)) {}

    template <typename Scalar>
    struct Scratch {
        mc::StateMatrix<Scalar> vMean;
        mc::StateMatrix<Scalar> vVar;
        Scratch(Eigen::Index nDims, Eigen::Index nPaths)
            : vMean(nDims, nPaths), vVar(nDims, nPaths) {}
    };

    template <typename Scalar, typename DriftF, typename DiffusionF>
        requires mc::Drift<DriftF, Scalar> && mc::Diffusion<DiffusionF, Scalar>
    void step(const mc::StateMatrix<Scalar>& x, mc::StateMatrix<Scalar>& xNext, double t, double dt,
              const Eigen::MatrixXd& z, const Eigen::MatrixXd& uniforms, const DriftF& /*drift*/,
              const DiffusionF& /*diffusion*/, const std::vector<Scalar>& theta,
              Scratch<Scalar>& scratch) const {
        using std::sqrt;
        if (x.rows() != 2) {
            throw std::invalid_argument("HestonQeProcess: state must be (lnS, V)");
        }
        process.varianceMoments(x, t, dt, theta, scratch.vMean, scratch.vVar);

        const double mu = process.mu;
        const double rho = process.rho;
        const double eta = process.eta;
        const double kappa = process.kappa;
        const double level = process.level;
        const double rhoOverEta = rho / eta;
        const double kappaRhoOverEta = kappa * rhoOverEta;
        const double oneMinusRho2 = 1.0 - rho * rho;

        for (Eigen::Index p = 0; p < x.cols(); ++p) {
            const Scalar v = x(1, p);
            const Scalar vNext = mc::qeSample(scratch.vMean(0, p), scratch.vVar(0, p),
                                              Scalar(z(0, p)), Scalar(uniforms(0, p)));
            const Scalar vBar = Scalar(0.5) * (v + vNext);
            const Scalar vBarPos = vBar > Scalar(0.0) ? vBar : Scalar(0.0);
            const Scalar diffusionScale = sqrt(Scalar(oneMinusRho2 * dt) * vBarPos);
            xNext(0, p) = x(0, p) + Scalar(mu * dt) +
                          Scalar(rhoOverEta) * (vNext - v - Scalar(kappa * level * dt)) +
                          Scalar(kappaRhoOverEta - 0.5) * vBar * Scalar(dt) +
                          diffusionScale * Scalar(z(1, p));
            xNext(1, p) = vNext;
        }
    }
};

} // namespace quantape::processes

#endif // QUANTAPE_MODELS_HESTON_QE_PROCESS_H
