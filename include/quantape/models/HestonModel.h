#ifndef QUANTAPE_MODELS_HESTON_MODEL_H
#define QUANTAPE_MODELS_HESTON_MODEL_H

#include "quantape/math/Integrals/ExponentialFittingLaguerre.h"
#include "quantape/math/Integrals/GaussLegendre.h"
#include "quantape/math/SpecialFunctions/TrigIntegrals.h"
#include "quantape/pricing/BlackScholes.h"
#include "quantape/util/Constants.h"

#include <Eigen/Dense>

#include <array>
#include <cmath>
#include <complex>
#include <cstddef>
#include <optional>
#include <stdexcept>
#include <vector>

namespace quantape::models {
using ::quantape::util::kPi;

/**
 * @file HestonModel.h
 * @brief Heston stochastic-volatility model (semi-analytic European pricer, H0)
 *
 * Model layer (`quantape/models/`): the *process* (drift/diffusion/moments
 * and the QE scheme) lives in `quantape/mc/processes/`; this header owns
 * the model-level mathematics — the characteristic function, the analytic
 * pricing formula, and instrument sensitivities — used for calibration and
 * IFT/KKT market-risk propagation.
 *
 * ## Formulation (H0)
 *
 * Gatheral's normalized characteristic function in the branch-cut-safe
 * ("little Heston trap") form:
 *
 *     phi(z) = exp{ (v0/s^2) (1-e^{-DT})/(1-G e^{-DT}) (k - i r s z - D)
 *                 + (k th/s^2) ((k - i r s z - D)T - 2 ln((1-G e^{-DT})/(1-G))) }
 *     D = sqrt((k - i r s z)^2 + (z^2 + i z) s^2),   G = (k - i r s z - D)/(k - i r s z + D)
 *
 * with the Lewis formula and a Black–Scholes control variate:
 *
 *     C = BS(S, K, T, sigma_BS)
 *       + sqrt(S e^{(r-q)T} K) e^{-rT}/pi
 *         * integral_0^{uMax} Re[ (phi_BS(u-i/2) - phi(u-i/2))/(u^2+1/4)
 *                                e^{i u mu} ] du,   mu = ln(S/K) + (r-q)T
 *
 * Pricing the Lewis integral by truncation at `uMax` with an arbitrary-order
 * Gauss–Legendre rule (`math/Integrals/GaussLegendre.h`) or the 64-node
 * exponentially fitted Gauss–Laguerre rule (default). The control-variate
 * volatility is the CF-matched
 * `sigma_BS^2 = -(8/T) ln Re phi(-i/2)` (fallback: variance-matched
 * `(1-e^{-kT})(v0-th)/(kT) + th`), and `sigma_BS` is **held fixed** during
 * differentiation so the complex-step gradient needs no complex normal CDF
 * (the identity holds for any control volatility).
 *
 * ## Complex-step Jacobians
 *
 * `callGradient`/`callHessian` cover the model parameters; `fullGradient`
 * and `fullHessian` cover the full 9-parameter vector
 * `[v0, kappa, theta, sigma, rho, S, K, r, q]` (market sensitivities moved
 * to the front of the calibration chain exactly as for `GBS`). All of them
 * perturb parameters into the complex plane and read the imaginary part of
 * the complex-valued pricer core, so instrument Jacobians need no tape and
 * no finite differences. Branches (control-variate choice, quadrature
 * rule/scale) are pinned at the primal point; the market leg is
 * differentiated through the complex continuation of the Black–Scholes
 * base (`PhiComplex`) and of the asymptotic reference. The Hessian uses
 * complex-step directional second derivatives,
 * `delta' H delta = 2 (f(x) - Re f(x + i delta))`, i.e. `n(n+1)/2 + 1`
 * evaluations instead of the 4n^2 of central differences.
 *
 * ## Scope
 *
 * Analytic European calls with value/gradient/Hessian, the precomputed
 * quote surrogates used by calibration/IFT (no tape through the
 * quadrature), and the `make_callback_var` instantiation
 * (`models/HestonStanPrimitives.h`) that puts the whole price+gradient
 * (or price+Hessian) on a 1-2 node tape for Stan `var`/`fvar<var>`.
 */

struct HestonParams {
    double v0 = 0.04;    ///< initial variance
    double kappa = 2.0;  ///< variance mean-reversion speed
    double theta = 0.04; ///< long-run variance
    double sigma = 0.3;  ///< vol-of-vol
    double rho = -0.7;   ///< spot/variance correlation
};

struct HestonMarket {
    double spot = 1.0;
    double strike = 1.0;
    double rate = 0.0;     ///< continuously compounded discount rate r
    double dividend = 0.0; ///< continuous dividend yield q
    double tMax = 1.0;
};

enum class HestonQuadrature {
    GaussLegendre,      ///< fixed truncated Gauss–Legendre (reference)
    ExponentiallyFitted ///< 64-node exponentially fitted Gauss–Laguerre (default)
};

enum class HestonControlVariate {
    Auto,            ///< rule-based choice (QuantLib optimalControlVariate logic)
    CfMatched,       ///< sigma_BS^2 = -(8/T) ln Re phi(-i/2) (Andersen-Piterbarg opt)
    VarianceMatched, ///< sigma_BS^2 = (1-e^{-kT})(v0-th)/(kT) + th (Andersen-Piterbarg)
    Asymptotic,      ///< asymptotic expansion of the characteristic function
    None,            ///< plain Lewis, no control variate
};

/// Full parameter layout of the calibration-grade Jacobians:
/// model parameters first, then the market leg (`tMax` stays a quote-level
/// double, as for `GBS`).
enum HestonParamIndex : int {
    HESTON_V0 = 0,
    HESTON_KAPPA = 1,
    HESTON_THETA = 2,
    HESTON_SIGMA = 3,
    HESTON_RHO = 4,
    HESTON_SPOT = 5,
    HESTON_STRIKE = 6,
    HESTON_RATE = 7,
    HESTON_DIVIDEND = 8,
    HESTON_PARAM_COUNT = 9
};

using HestonFullPoint = std::array<double, HESTON_PARAM_COUNT>;

inline HestonFullPoint toFullPoint(const HestonParams& p, const HestonMarket& m) {
    return {p.v0, p.kappa, p.theta, p.sigma, p.rho, m.spot, m.strike, m.rate, m.dividend};
}

inline HestonParams paramsFromFull(const HestonFullPoint& x) {
    return {x[HESTON_V0], x[HESTON_KAPPA], x[HESTON_THETA], x[HESTON_SIGMA], x[HESTON_RHO]};
}

inline HestonMarket marketFromFull(const HestonFullPoint& x, double tMax) {
    return {x[HESTON_SPOT], x[HESTON_STRIKE], x[HESTON_RATE], x[HESTON_DIVIDEND], tMax};
}

struct HestonConfig {
    std::size_t quadratureOrder = 512;
    double uMax = 200.0; ///< truncation of the Lewis integral (Gauss–Legendre only)
    HestonControlVariate controlVariate = HestonControlVariate::Auto;
    HestonQuadrature quadrature = HestonQuadrature::ExponentiallyFitted;

    void validate() const {
        if (quadratureOrder < 8) {
            throw std::invalid_argument("HestonModel: quadratureOrder must be >= 8");
        }
        if (!(uMax > 0.0)) {
            throw std::invalid_argument("HestonModel: uMax must be positive");
        }
    }
};

class HestonModel {
public:
    explicit HestonModel(HestonConfig config = {}) : m_config(config) { m_config.validate(); }

    const HestonConfig& config() const { return m_config; }

    /// Gatheral trap-free normalized characteristic function.
    static std::complex<double> characteristic(const std::complex<double>& z,
                                               const std::array<std::complex<double>, 5>& p,
                                               double tMax) {
        const std::complex<double> i(0.0, 1.0);
        const std::complex<double> a = p[1] - i * p[4] * p[3] * z;
        const std::complex<double> d = std::sqrt(a * a + (z * z + i * z) * p[3] * p[3]);
        const std::complex<double> g = (a - d) / (a + d);
        const std::complex<double> e = std::exp(-d * tMax);
        const std::complex<double> oneMinusGe = 1.0 - g * e;
        const std::complex<double> s2 = p[3] * p[3];
        const std::complex<double> term1 = (p[0] / s2) * (1.0 - e) / oneMinusGe * (a - d);
        const std::complex<double> term2 =
            (p[1] * p[2] / s2) * ((a - d) * tMax - 2.0 * std::log(oneMinusGe / (1.0 - g)));
        return std::exp(term1 + term2);
    }

    /// Loop-invariant contour constants (one assembly per quadrature pass).
    struct ContourConstants {
        std::complex<double> aConst;  ///< p1 - p4 p3 / 2
        std::complex<double> arho;    ///< -p4 p3  (the `i u` coefficient)
        std::complex<double> var;     ///< p3^2
        std::complex<double> t1Coeff; ///< p0 / p3^2
        std::complex<double> t2Coeff; ///< p1 p2 / p3^2
        double tMax = 1.0;
    };

    static ContourConstants contourConstants(const std::complex<double>* p, double tMax) {
        const std::complex<double> s2 = p[3] * p[3];
        ContourConstants c;
        c.aConst = p[1] - 0.5 * p[4] * p[3];
        c.arho = -(p[4] * p[3]);
        c.var = s2;
        c.t1Coeff = p[0] / s2;
        c.t2Coeff = p[1] * p[2] / s2;
        c.tMax = tMax;
        return c;
    }

    /// Characteristic function on the Lewis contour `z = u - i/2` for real
    /// `u`, using `z^2 + i z = u^2 + 1/4` to stay in real/simple arithmetic.
    /// This is the hot path; it equals `characteristic(u - i/2, p, tMax)`
    /// (gated in `test_heston_analytic`).
    static std::complex<double> characteristicOnContour(double u, const ContourConstants& c) {
        const std::complex<double> i(0.0, 1.0);
        const std::complex<double> a = c.aConst + i * (c.arho * u);
        const std::complex<double> d = std::sqrt(a * a + (u * u + 0.25) * c.var);
        const std::complex<double> g = (a - d) / (a + d);
        const std::complex<double> e = std::exp(-d * c.tMax);
        const std::complex<double> oneMinusGe = 1.0 - g * e;
        const std::complex<double> term1 = c.t1Coeff * (1.0 - e) / oneMinusGe * (a - d);
        const std::complex<double> term2 =
            c.t2Coeff * ((a - d) * c.tMax - 2.0 * std::log(oneMinusGe / (1.0 - g)));
        return std::exp(term1 + term2);
    }

    static std::complex<double> characteristicOnContour(double u, const std::complex<double>* p,
                                                        double tMax) {
        return characteristicOnContour(u, contourConstants(p, tMax));
    }

    /// Discounted European call price.
    double call(const HestonParams& params, const HestonMarket& market) const {
        return callFull(toFullPoint(params, market), market.tMax);
    }

    /// Discounted European call price on the full 9-parameter point.
    double callFull(const HestonFullPoint& x, double tMax) const {
        const auto xc = toComplexFull(x);
        return priceComplex(xc, tMax, cvState(xc, tMax)).real();
    }

    /// Undiscounted call (`e^{rT} * call`), matching the simulation engine's
    /// convention (engines return undiscounted expectations).
    double undiscountedCall(const HestonParams& params, const HestonMarket& market) const {
        return std::exp(market.rate * market.tMax) * call(params, market);
    }

    /// Complex-step parameter gradient, layout `[dv0, dkappa, dtheta, dsigma, drho]`.
    ///
    /// Symmetric complex step, `h = 1e-8`: the pricer's complex value has a
    /// nonzero imaginary part at the base point (the Lewis integral is
    /// complex), so the one-sided form would divide that offset by h.
    /// Differencing the +ih and -ih evaluations cancels both the offset and
    /// the node-level imaginary roundoff, leaving `O(h^2)` truncation. The
    /// control variate is held fixed (the CV identity holds for any fixed
    /// reference).
    Eigen::Matrix<double, 1, 5> callGradient(const HestonParams& params,
                                             const HestonMarket& market) const {
        const auto xc = toComplexFull(toFullPoint(params, market));
        const CvState cv = cvState(xc, market.tMax);
        const double h = 1e-8;
        Eigen::Matrix<double, 1, 5> out;
        for (int j = 0; j < 5; ++j) {
            auto up = xc;
            auto down = xc;
            up[static_cast<std::size_t>(j)] += std::complex<double>(0.0, h);
            down[static_cast<std::size_t>(j)] -= std::complex<double>(0.0, h);
            out(0, j) = (priceComplex(up, market.tMax, cv).imag() -
                         priceComplex(down, market.tMax, cv).imag()) /
                        (2.0 * h);
        }
        return out;
    }

    /// Complex-step gradient on the full 9-parameter point.
    Eigen::Matrix<double, 1, 9> fullGradient(const HestonFullPoint& point, double tMax) const {
        const auto xc = toComplexFull(point);
        const CvState cv = cvState(xc, tMax);
        const double h = 1e-8;
        Eigen::Matrix<double, 1, 9> out;
        for (int j = 0; j < 9; ++j) {
            auto up = xc;
            auto down = xc;
            up[static_cast<std::size_t>(j)] += std::complex<double>(0.0, h);
            down[static_cast<std::size_t>(j)] -= std::complex<double>(0.0, h);
            out(0, j) = (priceComplex(up, tMax, cv).imag() - priceComplex(down, tMax, cv).imag()) /
                        (2.0 * h);
        }
        return out;
    }

    /// Model-parameter Hessian (complex-step directional second derivatives,
    /// `n(n+1)/2 + 1` evaluations). Building block for the precomputed quote
    /// surrogates used by calibration/IFT (no tape through the quadrature).
    Eigen::Matrix<double, 5, 5> callHessian(const HestonParams& params,
                                            const HestonMarket& market) const {
        const auto xc = toComplexFull(toFullPoint(params, market));
        const CvState cv = cvState(xc, market.tMax);
        const std::array<double, 5> values = {params.v0, params.kappa, params.theta, params.sigma,
                                              params.rho};
        std::array<double, 5> delta{};
        for (std::size_t i = 0; i < 5; ++i) {
            delta[i] = 1e-4 * std::max(1.0, std::fabs(values[i]));
        }
        return directionHessian<5>(delta, [&](const std::array<double, 5>& d) {
            auto xp = xc;
            for (std::size_t i = 0; i < 5; ++i) {
                xp[i] += std::complex<double>(0.0, d[i]);
            }
            return priceComplex(xp, market.tMax, cv).real();
        });
    }

    /// Hessian on the full 9-parameter point (same directional complex-step
    /// scheme; symmetric by construction).
    Eigen::Matrix<double, 9, 9> fullHessian(const HestonFullPoint& point, double tMax) const {
        const auto xc = toComplexFull(point);
        const CvState cv = cvState(xc, tMax);
        std::array<double, 9> delta{};
        for (std::size_t i = 0; i < 9; ++i) {
            delta[i] = 1e-4 * std::max(1.0, std::fabs(point[i]));
        }
        return directionHessian<9>(delta, [&](const std::array<double, 9>& d) {
            auto xp = xc;
            for (std::size_t i = 0; i < 9; ++i) {
                xp[i] += std::complex<double>(0.0, d[i]);
            }
            return priceComplex(xp, tMax, cv).real();
        });
    }

    static std::array<std::complex<double>, 5> toComplex(const HestonParams& p) {
        return {std::complex<double>(p.v0), std::complex<double>(p.kappa),
                std::complex<double>(p.theta), std::complex<double>(p.sigma),
                std::complex<double>(p.rho)};
    }

    static std::array<std::complex<double>, 9> toComplexFull(const HestonFullPoint& x) {
        std::array<std::complex<double>, 9> out{};
        for (std::size_t i = 0; i < 9; ++i) {
            out[i] = std::complex<double>(x[i], 0.0);
        }
        return out;
    }

    /// Standard normal CDF on the complex plane: Hermite-Taylor expansion
    /// around the real part (exact to machine precision for the tiny
    /// imaginary steps of the Jacobians, where `erfc` is unavailable).
    static std::complex<double> PhiComplex(const std::complex<double>& z) {
        const double x = z.real();
        const double e = z.imag();
        const double nd = std::exp(-0.5 * x * x) / std::sqrt(2.0 * kPi);
        const double d1 = nd;                 // N'
        const double d2 = -x * nd;            // N''
        const double d3 = (x * x - 1.0) * nd; // N'''
        const double d4 = (3.0 * x - x * x * x) * nd;
        const std::complex<double> ie(0.0, e);
        const std::complex<double> ie2 = ie * ie;
        return 0.5 * std::erfc(-x * M_SQRT1_2) + ie * d1 + ie2 * (0.5 * d2) +
               ie2 * ie * (d3 / 6.0) + ie2 * ie2 * (d4 / 24.0);
    }

private:
    /// Fixed control-variate state: one reference CF per run, held fixed
    /// under complex-step differentiation (the Lewis identity holds for any
    /// fixed reference, so derivatives only flow through phi_H).
    struct CvState {
        HestonControlVariate kind = HestonControlVariate::None;
        double sigmaBs = 0.0;               ///< for Black-Scholes references
        std::complex<double> phi{0.0, 0.0}; ///< asymptotic reference: e^{u phi + psi}
        std::complex<double> psi{0.0, 0.0};
        std::complex<double> closed{0.0, 0.0}; ///< int_0^inf e^{phi u+psi} e^{i u mu}/(u^2+1/4) du
        double base = 0.0; ///< exact Lewis constant minus the reference integral
    };

    /// Effective control variate: `Auto` follows QuantLib's
    /// optimalControlVariate rule (asymptotic when the CF decays slowly and
    /// the first-order phase is small; Black-Scholes CF-matched otherwise).
    HestonControlVariate effectiveCv(const std::array<std::complex<double>, 5>& p,
                                     double tMax) const {
        if (m_config.controlVariate != HestonControlVariate::Auto) {
            return m_config.controlVariate;
        }
        if (tMax > 0.15) {
            const double v0 = p[0].real(), kappa = p[1].real(), theta = p[2].real();
            const double sigma = p[3].real(), rho = p[4].real();
            const double cInf = std::sqrt(1.0 - rho * rho) * (v0 + tMax * kappa * theta) / sigma;
            const double psiReal = ((kappa - 0.5 * rho * sigma) * (v0 + tMax * kappa * theta) +
                                    kappa * theta * std::log(4.0 * (1.0 - rho * rho))) /
                                   (sigma * sigma);
            if (cInf < 0.15 && psiReal < 0.1) {
                return HestonControlVariate::Asymptotic;
            }
        }
        return HestonControlVariate::CfMatched;
    }

    /// Assemble the fixed CV state (real parameters, one quadrature pass for
    /// the degenerate reference sum).
    CvState cvState(const std::array<std::complex<double>, 9>& x, double tMax) const {
        std::array<std::complex<double>, 5> p = {x[0], x[1], x[2], x[3], x[4]};
        CvState cv;
        cv.kind = effectiveCv(p, tMax);
        if (cv.kind == HestonControlVariate::None) {
            return cv;
        }
        if (cv.kind == HestonControlVariate::Asymptotic) {
            const double v0 = x[0].real(), kappa = x[1].real(), theta = x[2].real();
            const double sigma = x[3].real(), rho = x[4].real();
            const double t = tMax;
            const double r1 = std::sqrt(1.0 - rho * rho);
            cv.phi = std::complex<double>(-(v0 + t * kappa * theta) / sigma * r1,
                                          -(v0 + t * kappa * theta) / sigma * rho);
            cv.psi = std::complex<double>(
                         (kappa - 0.5 * rho * sigma) * (v0 + t * kappa * theta) +
                             kappa * theta * std::log(4.0 * (1.0 - rho * rho)),
                         -((0.5 * rho * rho * sigma - kappa * rho) / r1 * (v0 + kappa * theta * t) -
                           2.0 * kappa * theta * std::atan(rho / r1))) /
                     (sigma * sigma);
            const double mu = std::log(x[5].real() / x[6].real()) + (x[7].real() - x[8].real()) * t;
            const std::complex<double> pf = cv.phi + std::complex<double>(0.0, mu);
            using quantape::math::cosIntegral;
            using quantape::math::sinIntegral;
            cv.closed =
                std::exp(cv.psi) * (-2.0 * cosIntegral(-0.5 * pf) * std::sin(0.5 * pf) +
                                    std::cos(0.5 * pf) * (kPi.hi + 2.0 * sinIntegral(0.5 * pf)));
            // Reference price from the affine Lewis identity
            //   C(phi) = S e^{-qT} - P Re[ integral e^{i u mu} phi(u - i/2) ... ]
            // (verified numerically for two Black-Scholes references:
            // bs(sigma) + P L[sigma] = S e^{-qT} exactly). The asymptotic
            // reference is defined in z-space, so the contour factor
            // e^{-i phi/2} is explicit.
            const double spot = x[5].real(), strike = x[6].real();
            const double rate = x[7].real(), dividend = x[8].real();
            const double prefactor = std::sqrt(spot * std::exp((rate - dividend) * t) * strike) *
                                     std::exp(-rate * t) / kPi;
            const std::complex<double> contour = std::exp(std::complex<double>(0.0, -0.5) * cv.phi);
            cv.base = spot * std::exp(-dividend * t) - prefactor * (contour * cv.closed).real();
            return cv;
        }
        cv.sigmaBs = controlVolatility(p, tMax);
        return cv;
    }

    /// CF-matched or variance-matched control volatility from the
    /// *real* parts of the parameters (held fixed under differentiation).
    double controlVolatility(const std::array<std::complex<double>, 5>& p, double tMax) const {
        if (m_config.controlVariate == HestonControlVariate::None) {
            return 0.0;
        }
        if (m_config.controlVariate == HestonControlVariate::CfMatched && tMax > 0.0) {
            const auto phi = characteristic(std::complex<double>(0.0, -0.5), p, tMax);
            const double re = phi.real();
            if (re > 0.0 && re < 1.0) {
                return std::sqrt(-8.0 / tMax * std::log(re));
            }
        }
        const double kappa = p[1].real();
        const double v0 = p[0].real();
        const double theta = p[2].real();
        const double variance =
            (kappa * tMax > 0.0) ? (1.0 - std::exp(-kappa * tMax)) / (kappa * tMax) * (v0 - theta)
                                 : (v0 - theta);
        return std::sqrt(std::max(0.0, variance + theta));
    }

    /// Black-Scholes call continued to the complex plane: the control-variate
    /// base must move with complex market inputs. The normal CDF is the
    /// Hermite-Taylor `PhiComplex` (the imaginary steps are tiny).
    static std::complex<double> gbsComplex(const std::complex<double>& S,
                                           const std::complex<double>& K,
                                           const std::complex<double>& rDisc,
                                           const std::complex<double>& b, double vol, double T) {
        const std::complex<double> DF = std::exp(-rDisc * T);
        const std::complex<double> F = S * std::exp(b * T);
        if (vol <= 0.0) {
            // intrinsic value; the branch is pinned at the primal point
            return (F.real() > K.real()) ? DF * (F - K) : std::complex<double>(0.0, 0.0);
        }
        const double s = vol * std::sqrt(T);
        const std::complex<double> dp = (std::log(F / K) + 0.5 * s * s) / s;
        const std::complex<double> dm = dp - s;
        return DF * (F * PhiComplex(dp) - K * PhiComplex(dm));
    }

    /// Complex continuation of the asymptotic reference price
    /// `S e^{-qT} - P Re[e^{-i phi/2} closed(mu)]` with complex market leg.
    static std::complex<double> asymptoticBaseComplex(const std::array<std::complex<double>, 9>& x,
                                                      double tMax, const CvState& cv) {
        const std::complex<double> i(0.0, 1.0);
        const std::complex<double> mu = std::log(x[5] / x[6]) + (x[7] - x[8]) * tMax;
        const std::complex<double> pf = cv.phi + i * mu;
        using quantape::math::cosIntegral;
        using quantape::math::sinIntegral;
        const std::complex<double> closed =
            std::exp(cv.psi) * (-2.0 * cosIntegral(-0.5 * pf) * std::sin(0.5 * pf) +
                                std::cos(0.5 * pf) * (kPi.hi + 2.0 * sinIntegral(0.5 * pf)));
        const std::complex<double> prefactor =
            std::sqrt(x[5] * std::exp((x[7] - x[8]) * tMax) * x[6]) * std::exp(-x[7] * tMax) /
            kPi.hi;
        const std::complex<double> contour = std::exp(std::complex<double>(0.0, -0.5) * cv.phi);
        return x[5] * std::exp(-x[8] * tMax) - prefactor * (contour * closed);
    }

    /// Complex-valued pricer core: all nine parameters may carry an imaginary
    /// complex-step perturbation; the control-variate state is fixed.
    std::complex<double> priceComplex(const std::array<std::complex<double>, 9>& x, double tMax,
                                      const CvState& cv) const {
        const std::complex<double> i(0.0, 1.0);
        const std::complex<double> mu = std::log(x[5] / x[6]) + (x[7] - x[8]) * tMax;
        const bool asymptotic = (cv.kind == HestonControlVariate::Asymptotic);
        const double halfVarT = 0.5 * cv.sigmaBs * cv.sigmaBs * tMax;
        // e^{phi z + psi} on the contour: e^{-i phi/2} e^{phi u}.
        const std::complex<double> cvContour = std::exp(cv.psi - i * (0.5 * cv.phi));

        const ContourConstants cf = contourConstants(x.data(), tMax);
        const double muRe = mu.real(), muIm = mu.imag();
        const auto integrand = [&](double u) -> std::complex<double> {
            const double w = u * u + 0.25; // z^2 + i z for z = u - i/2
            const std::complex<double> phiH = characteristicOnContour(u, cf);
            const std::complex<double> phiCv =
                asymptotic ? cvContour * std::exp(cv.phi * u)
                           : std::complex<double>(std::exp(-halfVarT * w), 0.0);
            // e^{i u mu} = (cos, sin)(u muRe) * e^{-u muIm}
            const double phase = u * muRe;
            std::complex<double> osc(std::cos(phase), std::sin(phase));
            const double damp = -u * muIm;
            if (damp != 0.0) {
                osc *= std::exp(damp);
            }
            return (phiCv - phiH) / w * osc;
        };

        std::complex<double> integral(0.0, 0.0);
        if (m_config.quadrature == HestonQuadrature::ExponentiallyFitted && !asymptotic) {
            // EFGL rule on the (real) Lewis integrand, used with the
            // Black-Scholes control-variate family: the residual decays
            // fast, so the 64-node rule is exact to machine precision.
            // (The asymptotic reference has a near-zero oscillation
            // frequency in its qualifying regime, which degenerates the
            // fitted rule to the omega = 0 row with ~1e-7 accuracy; that
            // branch keeps the Gauss-Legendre rule below.)
            // Scale selection per the reference: Andersen-Piterbarg
            // variance scaling for the BS CVs.
            const double v0 = x[0].real(), kappa = x[1].real(), theta = x[2].real();
            const double vAvg =
                (kappa * tMax > 0.0)
                    ? (1.0 - std::exp(-kappa * tMax)) / (kappa * tMax) * (v0 - theta) + theta
                    : theta;
            const double scaling =
                std::max(0.25, std::min(1000.0, 0.25 / std::sqrt(0.5 * vAvg * tMax)));
            // Complex-valued integrand: the rule's row/scale are selected
            // from real (primal) inputs only, so the whole path is
            // holomorphic in the parameters and complex-step gradients see
            // the integral's response (a real-only lambda would zero them).
            integral = m_efgl.template integrate<std::complex<double>>(
                mu.real(), scaling,
                [&](const std::complex<double>& u) { return integrand(u.real()); });
        } else {
            // integral_0^{uMax} A(u) du = (uMax/2) sum_i w_i A(uMax (t_i + 1)/2)
            //
            // Adaptive truncation for the asymptotic reference: its integrand
            // decays like e^{-|Re phi| u}, so a fixed uMax under-integrates
            // slowly-decaying corners (|Re phi| ~ 1e-2 needs uMax ~ 1e4).
            double uMax = m_config.uMax;
            if (asymptotic) {
                const double decay = std::fabs(cv.phi.real());
                // Gauss-Legendre's first node sits at ~1.45 uMax/n^2; the
                // extension is only applied when that node still resolves
                // the integrand's structure near the origin.
                const std::size_t order = legendreRule().nodes().size();
                if (decay > 0.0 && 40.0 / decay <= 0.173 * static_cast<double>(order * order)) {
                    uMax = std::min(1.0e6, 40.0 / decay);
                }
            }
            const double scale = 0.5 * uMax;
            const auto& t = legendreRule().nodes();
            const auto& w = legendreRule().weights();
            for (std::size_t k = 0; k < t.size(); ++k) {
                integral +=
                    std::complex<double>(w[k] * scale, 0.0) * integrand(scale * (t[k] + 1.0));
            }
        }

        std::complex<double> base;
        if (asymptotic) {
            const bool marketReal = x[5].imag() == 0.0 && x[6].imag() == 0.0 &&
                                    x[7].imag() == 0.0 && x[8].imag() == 0.0;
            base = marketReal ? std::complex<double>(cv.base, 0.0)
                              : asymptoticBaseComplex(x, tMax, cv);
        } else {
            base = gbsComplex(x[5], x[6], x[7], x[7] - x[8], cv.sigmaBs, tMax);
        }
        const std::complex<double> prefactor =
            std::sqrt(x[5] * std::exp((x[7] - x[8]) * tMax) * x[6]) * std::exp(-x[7] * tMax) /
            kPi.hi;
        return base + prefactor * integral;
    }

    /// Hessian from symmetric complex-step directional second derivatives.
    ///
    /// For holomorphic `F`, `Re F(x + i delta) = F - delta' Im F' - 1/2
    /// delta' H delta + O(|delta|^3)`: the one-sided form would divide the
    /// first-order term by delta. Summing the +i delta and -i delta
    /// evaluations cancels it exactly, leaving
    /// `delta' H delta = 2 F(x) - Re F(x + i delta) - Re F(x - i delta)`.
    /// n diagonal plus n(n-1)/2 pair directions, n(n+1) + 1 evaluations.
    /// `eval` maps an n-vector of imaginary perturbations to Re F.
    template <std::size_t N, typename EvalT>
    static Eigen::Matrix<double, N, N> directionHessian(const std::array<double, N>& delta,
                                                        const EvalT& eval) {
        const std::array<double, N> zero{};
        const double f0 = eval(zero);
        Eigen::Matrix<double, N, N> hessian;
        std::array<double, N> q{};
        for (std::size_t i = 0; i < N; ++i) {
            std::array<double, N> d = zero;
            d[i] = delta[i];
            std::array<double, N> neg = zero;
            neg[i] = -delta[i];
            q[i] = 2.0 * f0 - eval(d) - eval(neg);
            hessian(i, i) = q[i] / (delta[i] * delta[i]);
        }
        for (std::size_t i = 0; i < N; ++i) {
            for (std::size_t j = i + 1; j < N; ++j) {
                std::array<double, N> d = zero;
                d[i] = delta[i];
                d[j] = delta[j];
                std::array<double, N> neg = zero;
                neg[i] = -delta[i];
                neg[j] = -delta[j];
                const double qij = 2.0 * f0 - eval(d) - eval(neg);
                hessian(i, j) = (qij - q[i] - q[j]) / (2.0 * delta[i] * delta[j]);
                hessian(j, i) = hessian(i, j);
            }
        }
        return hessian;
    }

    /// Gauss-Legendre rule, built on first use: the EFGL default path never
    /// touches it (the previous eager 512-node eigendecomposition dominated
    /// every model construction).
    const quantape::math::GaussLegendre& legendreRule() const {
        if (!m_rule) {
            m_rule.emplace(m_config.quadratureOrder);
        }
        return *m_rule;
    }

    HestonConfig m_config;
    mutable std::optional<quantape::math::GaussLegendre> m_rule;
    quantape::math::ExponentialFittingLaguerre m_efgl;
};

/// Precomputed per-quote surrogate: exact value, gradient and Hessian at a
/// parameter point. Calibration objectives then evaluate a 5-dimensional
/// quadratic per quote (nanoseconds) instead of re-running the quadrature
/// under AD — no tape is built through the integration. Validated by
/// `test_heston_analytic` (Taylor consistency) and by the H6 chain gates
/// (bump-recalibrate agreement).
struct HestonQuoteSurrogate {
    double value = 0.0;
    Eigen::Matrix<double, 1, 5> gradient;
    Eigen::Matrix<double, 5, 5> hessian;

    /// value + g . dtheta + 1/2 dtheta' H dtheta
    double eval(const Eigen::Matrix<double, 5, 1>& dtheta) const {
        return value + gradient.dot(dtheta) + 0.5 * dtheta.dot(hessian * dtheta);
    }
};

/// Build a quote surrogate (one value, one complex-step gradient, one
/// complex-step-directional Hessian of the analytic pricer).
inline HestonQuoteSurrogate makeQuoteSurrogate(const HestonModel& model, const HestonParams& params,
                                               const HestonMarket& market) {
    HestonQuoteSurrogate surrogate;
    surrogate.value = model.call(params, market);
    surrogate.gradient = model.callGradient(params, market);
    surrogate.hessian = model.callHessian(params, market);
    return surrogate;
}

/// Full 9-parameter surrogate `[v0, kappa, theta, sigma, rho, S, K, r, q]`:
/// one build per quote, then the calibration objective evaluates a quadratic
/// in nanoseconds. This is the tape-free counterpart of the
/// `make_callback_var` instantiation in `models/HestonStanPrimitives.h`.
struct HestonFullSurrogate {
    double value = 0.0;
    Eigen::Matrix<double, 1, 9> gradient;
    Eigen::Matrix<double, 9, 9> hessian;

    /// value + g . d x + 1/2 dx' H dx
    double eval(const Eigen::Matrix<double, 9, 1>& dx) const {
        return value + gradient.dot(dx) + 0.5 * dx.dot(hessian * dx);
    }
};

inline HestonFullSurrogate makeFullSurrogate(const HestonModel& model, const HestonFullPoint& point,
                                             double tMax) {
    HestonFullSurrogate surrogate;
    surrogate.value = model.callFull(point, tMax);
    surrogate.gradient = model.fullGradient(point, tMax);
    surrogate.hessian = model.fullHessian(point, tMax);
    return surrogate;
}

} // namespace quantape::models

#endif // QUANTAPE_MODELS_HESTON_MODEL_H
