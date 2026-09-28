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
    bool martingaleCorrection = true; ///< Andersen log-price drift correction

    explicit HestonQeProcess(Process p = Process{}, bool martingaleCorrectionIn = true)
        : process(std::move(p)), martingaleCorrection(martingaleCorrectionIn) {}

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

        Scalar mu, kappa, level, eta, rho;
        process.coefficients(theta, mu, kappa, level, eta, rho);
        const Scalar rhoOverEta = rho / eta;
        const Scalar kappaRhoOverEta = kappa * rhoOverEta;
        const Scalar oneMinusRho2 = Scalar(1.0) - rho * rho;
        const Scalar dtScalar(dt);

        for (Eigen::Index p = 0; p < x.cols(); ++p) {
            const Scalar v = x(1, p);
            const Scalar vNext = mc::qeSample(scratch.vMean(0, p), scratch.vVar(0, p),
                                              Scalar(z(0, p)), Scalar(uniforms(0, p)));
            const Scalar vBar = Scalar(0.5) * (v + vNext);
            const Scalar vBarPos = vBar > Scalar(0.0) ? vBar : Scalar(0.0);
            const Scalar diffusionScale = sqrt(oneMinusRho2 * dtScalar * vBarPos);
            Scalar correction = Scalar(0.0);
            if (martingaleCorrection) {
                // Choose the drift so that E[S'|S,V] = e^{mu dt} S exactly
                // under the QE sampler: subtract the deterministic part of
                // E[exp(lnS'-lnS)|V] and the sampler's MGF at the exponent
                // beta = rho/eta + (a + (1-rho^2)/2) dt/2,
                // a = kappa rho/eta - 1/2.
                const Scalar aCoef = kappaRhoOverEta - Scalar(0.5);
                const Scalar bCoef = aCoef + Scalar(0.5) * oneMinusRho2;
                const Scalar beta = rhoOverEta + Scalar(0.5) * dtScalar * bCoef;
                const Scalar mgf = mc::qeVarianceMGF(beta, scratch.vMean(0, p), scratch.vVar(0, p));
                using std::log;
                correction = rhoOverEta * (v + kappa * level * dtScalar) -
                             Scalar(0.5) * dtScalar * bCoef * v - log(mgf);
            }
            xNext(0, p) = x(0, p) + mu * dtScalar +
                          rhoOverEta * (vNext - v - kappa * level * dtScalar) +
                          (kappaRhoOverEta - Scalar(0.5)) * vBar * dtScalar +
                          diffusionScale * Scalar(z(1, p)) + correction;
            xNext(1, p) = vNext;
        }
    }
};

} // namespace quantape::processes

#endif // QUANTAPE_MODELS_HESTON_QE_PROCESS_H
