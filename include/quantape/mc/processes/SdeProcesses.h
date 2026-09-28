#ifndef QUANTAPE_MODELS_SDE_PROCESSES_H
#define QUANTAPE_MODELS_SDE_PROCESSES_H

#include "quantape/mc/SdePrimitives.h"

#include <Eigen/Dense>

#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <vector>

namespace quantape::processes {
/**
 * @file SdeProcesses.h
 * @brief Standard SDE model bundles for the simulation engine
 *
 * A model bundle carries its parameters and exposes the functors the
 * engine and its schemes consume:
 *
 *   drift(x, t, theta, out)              out(:,p) = f(X_k(:,p), t, theta)
 *   diffusion(x, t, theta, j, out)       out(:,p) = g_j(X_k(:,p), t, theta)
 *   mean/variance(x, t, dt, theta, out)  conditional first/second moments
 *                                        (moment-matching schemes)
 *
 * Batch convention: StateMatrix<Scalar> is (nDims x nPaths), one path per
 * column. theta (piecewise-constant per grid interval) is accepted for
 * time-dependent extensions; the standard bundles below use their member
 * parameters and ignore theta.
 *
 * Models and states:
 *
 *   GbmProcess    dS   = mu S dt + sigma S dW                  X = (S)
 *   OuProcess     dX   = kappa (level - X) dt + sigma dW       X = (X)
 *   CirProcess    dV   = kappa (level - V) dt + sigma sqrt(V) dW  X = (V)
 *   HestonProcess dlnS = (mu - V/2) dt + sqrt(V) dW_S,
 *               dV   = kappa (level - V) dt + eta sqrt(V) dW_V,
 *               dW_S = rho dW_V + sqrt(1-rho^2) dW_perp      X = (lnS, V)
 *
 * The Heston factor fields are g_1 = (rho sqrt(V), eta sqrt(V)) for the
 * V-Brownian and g_2 = (sqrt(1-rho^2) sqrt(V), 0) for the orthogonal one,
 * so the engine's X += sum_j g_j dW_j reproduces the correlation exactly.
 * (The legacy `GBMProcess.h`/`HestonProcess.h` classes use a different,
 * update-based API and are not wrapped here; they retire once parity is
 * established.)
 */

// ── GBM: dS = mu S dt + sigma S dW ──

struct GbmProcess {
    double mu = 0.0;
    double sigma = 0.2;

    template <typename Scalar>
    void drift(const mc::StateMatrix<Scalar>& x, double, const std::vector<Scalar>&,
               mc::StateMatrix<Scalar>& out) const {
        out = (Scalar(mu) * x.array()).matrix();
    }

    template <typename Scalar>
    void diffusion(const mc::StateMatrix<Scalar>& x, double, const std::vector<Scalar>&,
                   std::size_t factor, mc::StateMatrix<Scalar>& out) const {
        if (factor != 0) {
            throw std::invalid_argument("GbmProcess: one factor only");
        }
        out = (Scalar(sigma) * x.array()).matrix();
    }
};

// ── OU: dX = kappa (level - X) dt + sigma dW ──

struct OuProcess {
    double kappa = 1.0;
    double level = 0.0;
    double sigma = 0.2;

    template <typename Scalar>
    void drift(const mc::StateMatrix<Scalar>& x, double, const std::vector<Scalar>&,
               mc::StateMatrix<Scalar>& out) const {
        out = (Scalar(kappa) * (Scalar(level) - x.array())).matrix();
    }

    template <typename Scalar>
    void diffusion(const mc::StateMatrix<Scalar>& x, double, const std::vector<Scalar>&,
                   std::size_t factor, mc::StateMatrix<Scalar>& out) const {
        if (factor != 0) {
            throw std::invalid_argument("OuProcess: one factor only");
        }
        out.resize(x.rows(), x.cols());
        out.setConstant(Scalar(sigma));
    }

    /// Exact OU conditional moments (usable with MomentMatching1D)
    template <typename Scalar>
    void mean(const mc::StateMatrix<Scalar>& x, double, double dt, const std::vector<Scalar>&,
              mc::StateMatrix<Scalar>& out) const {
        using std::exp;
        const Scalar decay = exp(Scalar(-kappa * dt));
        out = x * decay;
        out.array() += (Scalar(level) * (Scalar(1.0) - decay));
    }

    template <typename Scalar>
    void variance(const mc::StateMatrix<Scalar>& x, double, double dt, const std::vector<Scalar>&,
                  mc::StateMatrix<Scalar>& out) const {
        using std::exp;
        const Scalar decay = exp(Scalar(-2.0 * kappa * dt));
        out.resize(x.rows(), x.cols());
        out.setConstant(Scalar(sigma * sigma * (1.0 - decay) / (2.0 * kappa)));
    }
};

// ── CIR: dV = kappa (level - V) dt + sigma sqrt(V) dW ──

struct CirProcess {
    double kappa = 2.0;
    double level = 0.04;
    double sigma = 0.2;

    template <typename Scalar>
    void drift(const mc::StateMatrix<Scalar>& x, double, const std::vector<Scalar>&,
               mc::StateMatrix<Scalar>& out) const {
        out = (Scalar(kappa) * (Scalar(level) - x.array())).matrix();
    }

    template <typename Scalar>
    void diffusion(const mc::StateMatrix<Scalar>& x, double, const std::vector<Scalar>&,
                   std::size_t factor, mc::StateMatrix<Scalar>& out) const {
        if (factor != 0) {
            throw std::invalid_argument("CirProcess: one factor only");
        }
        out = (Scalar(sigma) * x.array().cwiseMax(Scalar(0.0)).sqrt()).matrix();
    }

    /// Exact CIR conditional mean and variance (usable with MomentMatching1D)
    template <typename Scalar>
    void mean(const mc::StateMatrix<Scalar>& x, double, double dt, const std::vector<Scalar>&,
              mc::StateMatrix<Scalar>& out) const {
        using std::exp;
        const Scalar decay = exp(Scalar(-kappa * dt));
        out = x * decay;
        out.array() += (Scalar(level) * (Scalar(1.0) - decay));
    }

    template <typename Scalar>
    void variance(const mc::StateMatrix<Scalar>& x, double, double dt, const std::vector<Scalar>&,
                  mc::StateMatrix<Scalar>& out) const {
        using std::exp;
        const Scalar decay = exp(Scalar(-kappa * dt));
        const Scalar oneMinus = Scalar(1.0) - decay;
        out = x * (Scalar(sigma * sigma) * decay * oneMinus / Scalar(kappa));
        out.array() += Scalar(level * sigma * sigma * oneMinus * oneMinus / (2.0 * kappa));
    }
};

// ── Heston: X = (lnS, V) with two correlated factors ──

struct HestonProcess {
    double mu = 0.0;     ///< drift of lnS (r - q in the pricing measure)
    double kappa = 2.0;  ///< variance mean reversion
    double level = 0.04; ///< long-run variance
    double eta = 0.3;    ///< vol-of-vol
    double rho = -0.7;   ///< correlation of the two Brownians

    /// theta layout for the engine's per-interval parameter vector:
    /// `{mu, kappa, level, eta, rho}`. Empty theta falls back to the members
    /// (backward compatible); the initial variance v0 is carried by the state
    /// so gradients w.r.t. v0 flow through the initial condition.
    template <typename Scalar>
    void coefficients(const std::vector<Scalar>& theta, Scalar& muOut, Scalar& kappaOut,
                      Scalar& levelOut, Scalar& etaOut, Scalar& rhoOut) const {
        if (theta.empty()) {
            muOut = Scalar(mu);
            kappaOut = Scalar(kappa);
            levelOut = Scalar(level);
            etaOut = Scalar(eta);
            rhoOut = Scalar(rho);
            return;
        }
        if (theta.size() < 5) {
            throw std::invalid_argument(
                "HestonProcess: theta must be {mu, kappa, level, eta, rho}");
        }
        muOut = theta[0];
        kappaOut = theta[1];
        levelOut = theta[2];
        etaOut = theta[3];
        rhoOut = theta[4];
    }

    template <typename Scalar>
    void drift(const mc::StateMatrix<Scalar>& x, double, const std::vector<Scalar>& theta,
               mc::StateMatrix<Scalar>& out) const {
        if (x.rows() != 2) {
            throw std::invalid_argument("HestonProcess: state must be (lnS, V)");
        }
        Scalar mu, kappa, level, eta, rho;
        coefficients(theta, mu, kappa, level, eta, rho);
        out.resize(2, x.cols());
        out.row(0).array() = mu - Scalar(0.5) * x.row(1).array();
        out.row(1).array() = kappa * (level - x.row(1).array());
    }

    /// Factor fields: j = 0 -> V-Brownian (rho sqrt(V), eta sqrt(V));
    /// j = 1 -> orthogonal (sqrt(1-rho^2) sqrt(V), 0).
    template <typename Scalar>
    void diffusion(const mc::StateMatrix<Scalar>& x, double, const std::vector<Scalar>& theta,
                   std::size_t factor, mc::StateMatrix<Scalar>& out) const {
        if (x.rows() != 2) {
            throw std::invalid_argument("HestonProcess: state must be (lnS, V)");
        }
        using std::sqrt;
        Scalar mu, kappa, level, eta, rho;
        coefficients(theta, mu, kappa, level, eta, rho);
        const auto sqrtV = x.row(1).array().cwiseMax(Scalar(0.0)).sqrt();
        out.resize(2, x.cols());
        if (factor == 0) {
            out.row(0).array() = rho * sqrtV;
            out.row(1).array() = eta * sqrtV;
        } else if (factor == 1) {
            out.row(0).array() = sqrt(Scalar(1.0) - rho * rho) * sqrtV;
            out.row(1).setZero();
        } else {
            throw std::invalid_argument("HestonProcess: two factors only");
        }
    }

    /// Exact CIR conditional moments of the variance (row 1), for QE.
    ///
    /// Scalar loop with per-step coefficients hoisted: Eigen expression
    /// templates on AD scalars build a temporary per element, which the
    /// pathwise-gradient path measured as a hot spot.
    template <typename Scalar>
    void varianceMoments(const mc::StateMatrix<Scalar>& x, double, double dt,
                         const std::vector<Scalar>& theta, mc::StateMatrix<Scalar>& meanOut,
                         mc::StateMatrix<Scalar>& varOut) const {
        using std::exp;
        Scalar mu, kappa, level, eta, rho;
        coefficients(theta, mu, kappa, level, eta, rho);
        const Scalar decay = exp(Scalar(-1.0) * kappa * Scalar(dt));
        const Scalar meanAdd = level * (Scalar(1.0) - decay);
        const Scalar varSlope = eta * eta * decay * (Scalar(1.0) - decay) / kappa;
        const Scalar varAdd = level * (eta * eta * (Scalar(1.0) - decay) * (Scalar(1.0) - decay) /
                                       (Scalar(2.0) * kappa));
        meanOut.resize(1, x.cols());
        varOut.resize(1, x.cols());
        for (Eigen::Index p = 0; p < x.cols(); ++p) {
            const Scalar v = x(1, p);
            meanOut(0, p) = v * decay + meanAdd;
            varOut(0, p) = v * varSlope + varAdd;
        }
    }
};

} // namespace quantape::processes

#endif // QUANTAPE_MODELS_SDE_PROCESSES_H
