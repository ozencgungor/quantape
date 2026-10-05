#ifndef QUANTAPE_MATH_LEVENBERG_MARQUARDT_H
#define QUANTAPE_MATH_LEVENBERG_MARQUARDT_H

#include "quantape/math/LinearAlgebra/DenseSolve.h"
#include "quantape/math/Optimization/OptimizerPrimitives.h"

#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <utility>
#include <vector>

namespace quantape::math {
/**
 * @file LevenbergMarquardt.h
 * @brief Damped least-squares solver
 *
 * Minimizes `0.5 * ||r(x)||^2` for a residual callable
 * `void r(const std::vector<double>& x, std::vector<double>& out)`. The
 * damped normal equations `(J^T J + lambda diag(J^T J)) dx = -J^T r` are
 * solved with `solveDense` and lambda adapts on accepted/rejected steps.
 *
 * Two Jacobian routes: central differences (default overload) and an exact
 * user-supplied Jacobian overload; the Stan-backed exact-AD route lives in
 * `LevenbergMarquardtStanPrimitives.h`. For risk, do **not** differentiate
 * through the solver: the optimum is mapped to quotes by the IFT layer
 * (`calibrationIft`/`kktCalibrationJacobian`), exactly like the other
 * calibrations in this library.
 */

/// Iteration options. Tolerances are non-negative and are disabled at 0.
struct LevenbergMarquardtOptions {
    double lambda0 = 1e-3;      ///< initial damping (must be > 0)
    double nu = 10.0;           ///< multiplicative lambda update (must be > 0)
    double gradientTol = 1e-10; ///< absolute tolerance on `||J^T r||`
    double xtol = 1e-12;        ///< relative step tolerance: `||dx|| <= xtol*(1+||x||)`
    double xtolAbs = 0.0;       ///< absolute step tolerance (`||dx|| <= xtolAbs`, 0 = off)
    double ftol = 1e-12;        ///< relative cost tolerance: `df <= ftol*(1+cost)`
    double ftolAbs = 0.0;       ///< absolute cost tolerance (`df <= ftolAbs`, 0 = off)
    int maxIterations = 200;    ///< iteration budget (>= 0)
    double stepEpsilon = 1e-7;  ///< relative FD step (used by the FD overload)
};

struct LevenbergMarquardtResult {
    OptimizeResult status = OptimizeResult::Failure;
    double cost = 0.0;
    /// `J^T r` recomputed at the RETURNED `x` (never a pre-step snapshot).
    std::vector<double> gradient;
    /// `||J^T r||` at the returned `x`.
    double gradientNorm = 0.0;
    /// True when the returned `x` is stationary: `||J^T r|| <= gradientTol`,
    /// or the undamped Gauss-Newton step `(J^T J)^{-1} J^T r` is small in the
    /// x-scale `<= kStationarityStepTol * (1 + ||x||)` (a rank-deficient
    /// `J^T J` leaves this false). The IFT layer refuses to build derivatives
    /// at a non-stationary point.
    bool stationary = false;
    int iterations = 0; ///< 1-based iteration count of the returned run
};

namespace detail {

/// x-scaled tolerance for accepting the returned point as stationary when
/// `||J^T r||` exceeds the caller's absolute `gradientTol` (e.g. a
/// RoundoffLimited exit at a genuinely converged point).
constexpr double kStationarityStepTol = 1e-8;

inline void validateLevenbergMarquardtOptions(const LevenbergMarquardtOptions& options) {
    if (!(options.lambda0 > 0.0)) {
        throw std::invalid_argument("levenbergMarquardt: lambda0 must be > 0");
    }
    if (!(options.nu > 0.0)) {
        throw std::invalid_argument("levenbergMarquardt: nu must be > 0");
    }
    if (options.gradientTol < 0.0 || options.xtol < 0.0 || options.xtolAbs < 0.0 ||
        options.ftol < 0.0 || options.ftolAbs < 0.0) {
        throw std::invalid_argument("levenbergMarquardt: tolerances must be >= 0");
    }
    if (options.maxIterations < 0) {
        throw std::invalid_argument("levenbergMarquardt: maxIterations must be >= 0");
    }
    if (!(options.stepEpsilon > 0.0)) {
        throw std::invalid_argument("levenbergMarquardt: stepEpsilon must be > 0");
    }
}

template <typename Residual, typename Jacobian>
LevenbergMarquardtResult levenbergMarquardtCore(const Residual& residuals, const Jacobian& jacobian,
                                                std::vector<double>& x,
                                                const LevenbergMarquardtOptions& options) {
    validateLevenbergMarquardtOptions(options);
    const std::size_t n = x.size();
    if (n == 0) {
        throw std::invalid_argument("levenbergMarquardt: empty parameter vector");
    }
    std::vector<double> residual;
    const auto evaluate = [&](const std::vector<double>& point, std::vector<double>& out) -> double {
        residuals(point, out);
        double cost = 0.0;
        for (const double value : out) {
            cost += 0.5 * value * value;
        }
        return cost;
    };
    double cost = evaluate(x, residual);
    const std::size_t m = residual.size();
    if (m == 0) {
        throw std::invalid_argument("levenbergMarquardt: empty residual vector");
    }
    std::vector<double> jacobianFlat(m * n, 0.0);
    std::vector<double> candidateResidual;
    double lambda = options.lambda0;
    LevenbergMarquardtResult result;
    bool finished = false;

    for (int iteration = 0; iteration < options.maxIterations && !finished; ++iteration) {
        result.iterations = iteration + 1;
        jacobian(x, jacobianFlat);
        std::vector<double> jtj(n * n, 0.0);
        std::vector<double> jtr(n, 0.0);
        for (std::size_t i = 0; i < m; ++i) {
            for (std::size_t a = 0; a < n; ++a) {
                const double jia = jacobianFlat[i * n + a];
                jtr[a] += jia * residual[i];
                for (std::size_t b = 0; b < n; ++b) {
                    jtj[a * n + b] += jia * jacobianFlat[i * n + b];
                }
            }
        }
        double gradientNorm = 0.0;
        for (const double value : jtr) {
            gradientNorm += value * value;
        }
        gradientNorm = std::sqrt(gradientNorm);
        if (gradientNorm <= options.gradientTol) {
            result.status = OptimizeResult::GradientTolReached;
            finished = true;
            break;
        }
        bool stepAccepted = false;
        for (int attempt = 0; attempt < 20; ++attempt) {
            std::vector<double> damped = jtj;
            for (std::size_t j = 0; j < n; ++j) {
                const double diagonal = jtj[j * n + j];
                damped[j * n + j] += lambda * (diagonal > 0.0 ? diagonal : 1.0);
            }
            std::vector<double> rhs = jtr;
            for (double& value : rhs) {
                value = -value;
            }
            const std::vector<double> delta = solveDense(std::move(damped), n, rhs);
            std::vector<double> candidate = x;
            for (std::size_t j = 0; j < n; ++j) {
                candidate[j] += delta[j];
            }
            const double candidateCost = evaluate(candidate, candidateResidual);
            if (candidateCost < cost) {
                double stepNorm = 0.0;
                for (const double value : delta) {
                    stepNorm += value * value;
                }
                stepNorm = std::sqrt(stepNorm);
                x = candidate;
                residual = candidateResidual;
                const double improvement = cost - candidateCost;
                cost = candidateCost;
                lambda = std::max(lambda / options.nu, 1e-14);
                stepAccepted = true;
                double xNorm = 0.0;
                for (const double value : x) {
                    xNorm += value * value;
                }
                xNorm = std::sqrt(xNorm);
                if ((options.xtol > 0.0 && stepNorm <= options.xtol * (1.0 + xNorm)) ||
                    (options.xtolAbs > 0.0 && stepNorm <= options.xtolAbs)) {
                    result.status = OptimizeResult::XtolReached;
                    finished = true;
                } else if ((options.ftol > 0.0 && improvement <= options.ftol * (1.0 + cost)) ||
                           (options.ftolAbs > 0.0 && improvement <= options.ftolAbs)) {
                    result.status = OptimizeResult::FtolReached;
                    finished = true;
                }
                break;
            }
            lambda *= options.nu;
            if (lambda > 1e14) {
                break;
            }
        }
        if (!stepAccepted) {
            result.status = OptimizeResult::RoundoffLimited;
            finished = true;
        }
    }
    if (!finished) {
        result.status = OptimizeResult::MaxEvalReached;
    }
    result.cost = cost;

    // Recompute J^T r at the returned x (an accepted step would otherwise
    // leave `gradientNorm` as a stale pre-step snapshot).
    jacobian(x, jacobianFlat);
    std::vector<double> jtr(n, 0.0);
    for (std::size_t i = 0; i < m; ++i) {
        for (std::size_t a = 0; a < n; ++a) {
            jtr[a] += jacobianFlat[i * n + a] * residual[i];
        }
    }
    double gradientNorm = 0.0;
    for (const double value : jtr) {
        gradientNorm += value * value;
    }
    result.gradientNorm = std::sqrt(gradientNorm);
    result.gradient = std::move(jtr);
    result.stationary = result.gradientNorm <= options.gradientTol;
    if (!result.stationary && result.gradientNorm > 0.0) {
        std::vector<double> jtj(n * n, 0.0);
        for (std::size_t i = 0; i < m; ++i) {
            for (std::size_t a = 0; a < n; ++a) {
                for (std::size_t b = 0; b < n; ++b) {
                    jtj[a * n + b] += jacobianFlat[i * n + a] * jacobianFlat[i * n + b];
                }
            }
        }
        try {
            const std::vector<double> newtonStep =
                solveDense(std::move(jtj), n, result.gradient);
            double stepNorm = 0.0;
            for (const double value : newtonStep) {
                stepNorm += value * value;
            }
            stepNorm = std::sqrt(stepNorm);
            double xNorm = 0.0;
            for (const double value : x) {
                xNorm += value * value;
            }
            xNorm = std::sqrt(xNorm);
            result.stationary = stepNorm <= kStationarityStepTol * (1.0 + xNorm);
        } catch (const std::invalid_argument&) {
            result.stationary = false;
        }
    }
    return result;
}

} // namespace detail

/**
 * @brief Optimizer-style wrapper: `StopCriteria` + `OptimizerState`, same call
 * shape as `LBFGS`/`SLSQP`, plus the LM-specific options.
 *
 * `StopCriteria` mapping: `ftol_rel`/`ftol_abs` -> `ftol`/`ftolAbs`,
 * `xtol_rel`/`xtol_abs` -> `xtol`/`xtolAbs`, `grad_tol` -> `gradientTol`,
 * `maxeval` -> `maxIterations`. `OptimizerState::grad` receives the true
 * `J^T r` vector at the returned point (never a scalar norm).
 */
class LevenbergMarquardt {
public:
    explicit LevenbergMarquardt(LevenbergMarquardtOptions options = {})
        : m_options(std::move(options)) {}

    LevenbergMarquardt(StopCriteria criteria, LevenbergMarquardtOptions options = {})
        : m_options(std::move(options)) {
        if (criteria.maxeval > 0) {
            m_options.maxIterations = criteria.maxeval;
        }
        if (criteria.ftol_rel > 0.0) {
            m_options.ftol = criteria.ftol_rel;
        }
        if (criteria.ftol_abs > 0.0) {
            m_options.ftolAbs = criteria.ftol_abs;
        }
        if (criteria.xtol_rel > 0.0) {
            m_options.xtol = criteria.xtol_rel;
        }
        if (criteria.xtol_abs > 0.0) {
            m_options.xtolAbs = criteria.xtol_abs;
        }
        if (criteria.grad_tol > 0.0) {
            m_options.gradientTol = criteria.grad_tol;
        }
    }

    const LevenbergMarquardtOptions& options() const { return m_options; }

    /// Central-difference Jacobian; writes the optimized point into `state`.
    template <typename Residual>
    LevenbergMarquardtResult minimize(const Residual& residuals, std::vector<double>& x,
                                      OptimizerState& state) const {
        const std::size_t n = x.size();
        std::vector<double> residual;
        std::vector<double> perturbed;
        std::vector<double> shifted;
        const auto jacobian = [&](const std::vector<double>& point, std::vector<double>& flat) {
            residuals(point, residual);
            const std::size_t m = residual.size();
            if (n == 0 || m != flat.size() / n) {
                throw std::invalid_argument(
                    "levenbergMarquardt: residual size changed between evaluations");
            }
            for (std::size_t j = 0; j < n; ++j) {
                const double step = m_options.stepEpsilon * (std::abs(point[j]) + 1.0);
                perturbed = point;
                perturbed[j] += step;
                residuals(perturbed, shifted);
                for (std::size_t i = 0; i < m; ++i) {
                    flat[i * n + j] = shifted[i];
                }
                perturbed[j] = point[j] - step;
                residuals(perturbed, shifted);
                for (std::size_t i = 0; i < m; ++i) {
                    flat[i * n + j] = (flat[i * n + j] - shifted[i]) / (2.0 * step);
                }
            }
        };
        LevenbergMarquardtResult result =
            detail::levenbergMarquardtCore(residuals, jacobian, x, m_options);
        fillState(result, x, state);
        return result;
    }

    template <typename Residual>
    LevenbergMarquardtResult minimize(const Residual& residuals, std::vector<double>& x) const {
        OptimizerState state;
        return minimize(residuals, x, state);
    }

    /// Exact user-supplied Jacobian (`void jacobian(point, flatRowMajor)`).
    template <typename Residual, typename Jacobian>
    LevenbergMarquardtResult minimize(const Residual& residuals, const Jacobian& jacobian,
                                      std::vector<double>& x, OptimizerState& state) const {
        LevenbergMarquardtResult result =
            detail::levenbergMarquardtCore(residuals, jacobian, x, m_options);
        fillState(result, x, state);
        return result;
    }

    /// Write the final run state shared by the double and the Var/Ift paths.
    static void fillState(const LevenbergMarquardtResult& result, const std::vector<double>& x,
                          OptimizerState& state) {
        state.x = x;
        state.f = result.cost;
        state.iterations = static_cast<std::size_t>(result.iterations);
        state.message = to_string(result.status);
        state.grad = result.gradient;
    }

private:
    LevenbergMarquardtOptions m_options;
};

/// LM with a central-difference Jacobian.
template <typename Residual>
LevenbergMarquardtResult levenbergMarquardt(const Residual& residuals, std::vector<double>& x,
                                            const LevenbergMarquardtOptions& options = {}) {
    OptimizerState state;
    return LevenbergMarquardt(options).minimize(residuals, x, state);
}

/// LM with an exact user-supplied Jacobian
/// (`void jacobian(point, flatRowMajorJacobian)`).
template <typename Residual, typename Jacobian>
LevenbergMarquardtResult levenbergMarquardt(const Residual& residuals, const Jacobian& jacobian,
                                            std::vector<double>& x,
                                            const LevenbergMarquardtOptions& options = {}) {
    return detail::levenbergMarquardtCore(residuals, jacobian, x, options);
}

} // namespace quantape::math

#endif // QUANTAPE_MATH_LEVENBERG_MARQUARDT_H
