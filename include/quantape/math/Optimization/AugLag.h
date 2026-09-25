#ifndef AUGLAG_H
#define AUGLAG_H

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <utility>
#include <vector>

#include "Constraint.h"
#include "LBFGS.h"
#include "OptimizerPrimitives.h"
#include "SLSQP.h"

namespace quantape::math {
/**
 * @file AugLag.h
 * @brief Augmented Lagrangian optimizer (AUGLAG semantics) with exact AD
 *
 * Outer loop (Rockafellar-style augmented Lagrangian, as in NLopt's AUGLAG):
 *   L_A(x; lambda, mu, rho) = f(x)
 *       + sum_k [ lambda_k h_k + (rho/2) h_k^2 ]
 *       + sum_i [ (max(0, mu_i + rho g_i)^2 - mu_i^2) / (2 rho) ]
 *       + (rho/2) * box-bound violations (v1: penalty only, no bound duals)
 *
 * Each outer iteration solves the smooth unconstrained problem with the
 * L-BFGS inner solver (exact AD gradients), then updates
 *   lambda <- lambda + rho h,   mu <- max(0, mu + rho g),
 * and increases rho when the feasibility improvement stalls. Convergence is
 * declared on primal feasibility plus KKT stationarity with the current
 * multipliers.
 *
 * DoubleT: AD scalar (scalar-generic objective/constraints, reverse pass per
 * evaluation) or double (NLopt-style `f(x, grad)` / `con(x, c, J)` callbacks,
 * analytic or caller-side Stan).
 */
template <typename DoubleT>
    requires OptimizationScalar<DoubleT>
class AugLag : public Optimizer<DoubleT, AugLag<DoubleT>> {
public:
    using Base = Optimizer<DoubleT, AugLag<DoubleT>>;

    /**
     * @param criteria Stop criteria for the outer loop
     * @param memory L-BFGS memory of the inner solver (0 = heuristic)
     * @param initial_penalty starting rho
     * @param penalty_growth multiplicative rho escalation
     */
    explicit AugLag(StopCriteria criteria = {}, int memory = 0, double initial_penalty = 0.0,
                    double penalty_growth = 10.0)
        : Base(std::move(criteria)), m_memory(memory), m_rho0(initial_penalty),
          m_growth(penalty_growth) {}

    // ── Constrained entry points ──

    template <typename F, typename GI, typename GE = NoConstraint>
        requires ObjectiveEvaluator<F, DoubleT> && ConstraintEvaluator<GI, DoubleT> &&
                 ConstraintEvaluator<GE, DoubleT>
    OptimizeResult minimize(const F& f, const GI& inequality, const GE& equality,
                            const Bounds& bounds, std::vector<double>& x,
                            OptimizerState& state) const {
        if (x.empty()) {
            throw std::invalid_argument("AugLag: empty parameter vector");
        }
        state = OptimizerState{};
        state.x = x;
        state.start_time = nowSeconds();
        const OptimizeResult result = run(f, inequality, equality, bounds, state);
        x = state.x;
        return result;
    }

    template <typename F, typename GI, typename GE = NoConstraint>
        requires ObjectiveEvaluator<F, DoubleT> && ConstraintEvaluator<GI, DoubleT> &&
                 ConstraintEvaluator<GE, DoubleT>
    OptimizeResult minimize(const F& f, const GI& inequality, const GE& equality,
                            const Bounds& bounds, std::vector<double>& x) const {
        OptimizerState state;
        return minimize(f, inequality, equality, bounds, x, state);
    }

    template <typename F, typename GI>
        requires ObjectiveEvaluator<F, DoubleT> && ConstraintEvaluator<GI, DoubleT>
    OptimizeResult minimize(const F& f, const GI& inequality, const Bounds& bounds,
                            std::vector<double>& x, OptimizerState& state) const {
        return minimize(f, inequality, NoConstraint{}, bounds, x, state);
    }

    template <typename F, typename GI>
        requires ObjectiveEvaluator<F, DoubleT> && ConstraintEvaluator<GI, DoubleT>
    OptimizeResult minimize(const F& f, const GI& inequality, const Bounds& bounds,
                            std::vector<double>& x) const {
        OptimizerState state;
        return minimize(f, inequality, NoConstraint{}, bounds, x, state);
    }

    /// Unconstrained entry point (inherited base dispatch)
    template <typename F>
        requires ObjectiveEvaluator<F, DoubleT>
    OptimizeResult minimizeImpl(const F& f, OptimizerState& state) const {
        return run(f, NoConstraint{}, NoConstraint{}, Bounds::unbounded(state.x.size()), state);
    }

private:
    template <typename F, typename GI, typename GE>
    OptimizeResult run(const F& f, const GI& inequality, const GE& equality, const Bounds& bounds,
                       OptimizerState& state) const {
        static_assert(!std::is_same_v<DoubleT, double> || ValueGradObjective<F>,
                      "AugLag<double> requires a value+gradient objective: "
                      "double f(const std::vector<double>& x, std::vector<double>& grad)");
        static_assert(!std::is_same_v<DoubleT, double> || std::is_same_v<GI, NoConstraint> ||
                          ValueJacobianConstraint<GI>,
                      "AugLag<double> requires inequality constraints as void con(x, c, J)");
        static_assert(!std::is_same_v<DoubleT, double> || std::is_same_v<GE, NoConstraint> ||
                          ValueJacobianConstraint<GE>,
                      "AugLag<double> requires equality constraints as void con(x, c, J)");

        const std::size_t n = state.x.size();
        const StopCriteria& criteria = this->criteria();
        const Bounds bnd = bounds.empty() ? Bounds::unbounded(n) : bounds;
        // Practical tolerances: primal feasibility is tight; the projected
        // KKT residual is limited by the linear multiplier convergence at
        // fixed rho (increasing rho further destroys the inner conditioning).
        constexpr double kFeasTol = 1e-9;
        constexpr double kKktTol = 1e-6;
        constexpr int kMaxOuter = 60;

        std::vector<double> g, Jg, h, Jh;
        std::vector<DoubleT> theta_scratch(n);

        const auto max_abs = [](const std::vector<double>& v) {
            double m = 0.0;
            for (double x : v) {
                m = std::max(m, std::fabs(x));
            }
            return m;
        };

        // Evaluate f, grad, constraint values and Jacobians at z
        const auto evaluate_point = [&](const std::vector<double>& z, double& f_out,
                                        std::vector<double>& grad_out, std::vector<double>& g_out,
                                        std::vector<double>& jg_out, std::vector<double>& h_out,
                                        std::vector<double>& jh_out) {
            f_out = this->evalValueGradInto(f, z, theta_scratch, grad_out);
            ++state.evals;
            ++state.grad_evals;
            if constexpr (std::is_same_v<GI, NoConstraint>) {
                g_out.clear();
                jg_out.clear();
            } else {
                this->evalConstraint(inequality, z, g_out, jg_out);
                ++state.evals;
                ++state.grad_evals;
            }
            if constexpr (std::is_same_v<GE, NoConstraint>) {
                h_out.clear();
                jh_out.clear();
            } else {
                this->evalConstraint(equality, z, h_out, jh_out);
                ++state.evals;
                ++state.grad_evals;
            }
        };

        evaluate_point(state.x, state.f, state.grad, g, Jg, h, Jh);

        std::vector<double> lambda(h.size(), 0.0);
        std::vector<double> mu(g.size(), 0.0);
        std::vector<double> mu_lo(n, 0.0), mu_up(n, 0.0);

        // Birgin-Martinez starting rho: max(1e-6, min(10, 2|f|/con2))
        double rho = m_rho0;
        if (rho <= 0.0) {
            double con2 = 0.0;
            for (double gi : g) {
                if (gi > 0.0) {
                    con2 += gi * gi;
                }
            }
            for (double hi : h) {
                con2 += hi * hi;
            }
            rho = (con2 > 0.0) ? std::max(1e-6, std::min(10.0, 2.0 * std::fabs(state.f) / con2))
                               : 10.0;
        }
        double prev_icm = std::numeric_limits<double>::max();
        double kkt_norm = std::numeric_limits<double>::max();
        int repeated_failures = 0;

        // SLSQP feasibility polish: the augmented Lagrangian converges linearly
        // near the boundary; SLSQP (same objective/constraints) tightens the
        // point to exact feasibility quadratically. Only for constrained runs.
        const auto polish = [&]() -> bool {
            if constexpr (std::is_same_v<GI, NoConstraint> && std::is_same_v<GE, NoConstraint>) {
                return false;
            }
            SLSQP<DoubleT> polisher(criteria, kFeasTol);
            std::vector<double> x_polish = state.x;
            OptimizerState polish_state;
            const OptimizeResult pr =
                polisher.minimize(f, inequality, equality, bnd, x_polish, polish_state);
            if (pr != OptimizeResult::Success && pr != OptimizeResult::GradientTolReached) {
                return false;
            }
            state.x = x_polish;
            evaluate_point(state.x, state.f, state.grad, g, Jg, h, Jh);
            // Adopt the polisher's (exact) multipliers for the IFT layer
            state.ineq_multipliers = polish_state.ineq_multipliers;
            state.eq_multipliers = polish_state.eq_multipliers;
            state.message = "KKT conditions satisfied (SLSQP polish)";
            return true;
        };

        for (int outer = 0; outer < kMaxOuter; ++outer) {
            if (this->stopByEvalOrTime(state)) {
                state.message = "evaluation/time budget exhausted";
                return stopTime(criteria, state.start_time) ? OptimizeResult::MaxTimeReached
                                                            : OptimizeResult::MaxEvalReached;
            }
            if (state.f <= criteria.stopval) {
                state.message = "stop value reached";
                return OptimizeResult::StopvalReached;
            }

            // ── Inner solve of the augmented objective
            // (tight gradient tolerance: accurate inner solves give accurate
            // multiplier updates and keep rho from escalating needlessly)
            StopCriteria inner_criteria;
            inner_criteria.grad_tol = 1e-10;
            inner_criteria.maxeval = 2000;
            std::vector<double> x_inner = state.x;

            if constexpr (!std::is_same_v<DoubleT, double>) {
                const auto augmented = [&, lambda, mu, mu_lo, mu_up, rho](const auto& theta) {
                    using S = typename std::decay_t<decltype(theta)>::value_type;
                    ++state.evals;
                    ++state.grad_evals;
                    S value = f(theta);
                    if constexpr (!std::is_same_v<GI, NoConstraint>) {
                        std::vector<S> gv;
                        inequality(theta, gv);
                        for (std::size_t i = 0; i < gv.size(); ++i) {
                            const S t = S(mu[i]) + S(rho) * gv[i];
                            // Smoothed max(0, t): keeps the augmented objective
                            // C^1 so the inner Wolfe search is well-posed; the
                            // bias is O(delta) and the dual updates stay exact.
                            const double delta = 1e-12 * (1.0 + std::fabs(mu[i]));
                            using std::sqrt;
                            const S pos = S(0.5) * (t + sqrt(t * t + S(delta * delta)));
                            value += (pos * pos - S(mu[i] * mu[i])) / S(2.0 * rho);
                        }
                    }
                    if constexpr (!std::is_same_v<GE, NoConstraint>) {
                        std::vector<S> hv;
                        equality(theta, hv);
                        for (std::size_t k = 0; k < hv.size(); ++k) {
                            value += S(lambda[k]) * hv[k] + S(0.5 * rho) * hv[k] * hv[k];
                        }
                    }
                    if (!bnd.empty()) {
                        for (std::size_t j = 0; j < theta.size(); ++j) {
                            using std::sqrt;
                            if (bnd.hasLower(j)) {
                                const S gv = S(bnd.lower[j]) - theta[j];
                                const S t = S(mu_lo[j]) + S(rho) * gv;
                                const double delta = 1e-12 * (1.0 + std::fabs(mu_lo[j]));
                                const S pos = S(0.5) * (t + sqrt(t * t + S(delta * delta)));
                                value += (pos * pos - S(mu_lo[j] * mu_lo[j])) / S(2.0 * rho);
                            }
                            if (bnd.hasUpper(j)) {
                                const S gv = theta[j] - S(bnd.upper[j]);
                                const S t = S(mu_up[j]) + S(rho) * gv;
                                const double delta = 1e-12 * (1.0 + std::fabs(mu_up[j]));
                                const S pos = S(0.5) * (t + sqrt(t * t + S(delta * delta)));
                                value += (pos * pos - S(mu_up[j] * mu_up[j])) / S(2.0 * rho);
                            }
                        }
                    }
                    return value;
                };
                LBFGS<DoubleT> inner(inner_criteria, m_memory);
                const auto inner_result = inner.minimize(augmented, x_inner);
                if (inner_result == OptimizeResult::RoundoffLimited ||
                    inner_result == OptimizeResult::Failure) {
                    ++repeated_failures;
                    if (repeated_failures >= 10) {
                        if (polish()) {
                            return OptimizeResult::Success;
                        }
                        state.message = "inner solver failed repeatedly";
                        return OptimizeResult::RoundoffLimited;
                    }
                } else {
                    repeated_failures = 0;
                }
            } else {
                const auto augmented = [&, lambda, mu, mu_lo, mu_up,
                                        rho](const std::vector<double>& x,
                                             std::vector<double>& grad_out) {
                    ++state.evals;
                    ++state.grad_evals;
                    std::vector<double> grad;
                    double value = f(x, grad);
                    if constexpr (!std::is_same_v<GI, NoConstraint>) {
                        std::vector<double> c, J;
                        inequality(x, c, J);
                        const std::size_t m = c.size();
                        for (std::size_t i = 0; i < m; ++i) {
                            const double t = mu[i] + rho * c[i];
                            const double delta = 1e-12 * (1.0 + std::fabs(mu[i]));
                            const double root = std::sqrt(t * t + delta * delta);
                            const double pos = 0.5 * (t + root);
                            const double dpos = 0.5 * (1.0 + t / root);
                            value += (pos * pos - mu[i] * mu[i]) / (2.0 * rho);
                            for (std::size_t j = 0; j < x.size(); ++j) {
                                grad[j] += pos * dpos * J[i * x.size() + j];
                            }
                        }
                    }
                    if constexpr (!std::is_same_v<GE, NoConstraint>) {
                        std::vector<double> c, J;
                        equality(x, c, J);
                        const std::size_t m = c.size();
                        for (std::size_t k = 0; k < m; ++k) {
                            value += lambda[k] * c[k] + 0.5 * rho * c[k] * c[k];
                            for (std::size_t j = 0; j < x.size(); ++j) {
                                grad[j] += (lambda[k] + rho * c[k]) * J[k * x.size() + j];
                            }
                        }
                    }
                    for (std::size_t j = 0; j < x.size(); ++j) {
                        const auto smooth_term = [&](double t, double mu_side, double sign) {
                            const double delta = 1e-12 * (1.0 + std::fabs(mu_side));
                            const double root = std::sqrt(t * t + delta * delta);
                            const double pos = 0.5 * (t + root);
                            const double dpos = 0.5 * (1.0 + t / root);
                            value += (pos * pos - mu_side * mu_side) / (2.0 * rho);
                            grad[j] += sign * pos * dpos;
                        };
                        if (bnd.hasLower(j)) {
                            smooth_term(mu_lo[j] + rho * (bnd.lower[j] - x[j]), mu_lo[j], -1.0);
                        }
                        if (bnd.hasUpper(j)) {
                            smooth_term(mu_up[j] + rho * (x[j] - bnd.upper[j]), mu_up[j], +1.0);
                        }
                    }
                    grad_out = std::move(grad);
                    return value;
                };
                LBFGS<double> inner(inner_criteria, m_memory);
                const auto inner_result = inner.minimize(augmented, x_inner);
                if (inner_result == OptimizeResult::RoundoffLimited ||
                    inner_result == OptimizeResult::Failure) {
                    ++repeated_failures;
                    if (repeated_failures >= 10) {
                        if (polish()) {
                            return OptimizeResult::Success;
                        }
                        state.message = "inner solver failed repeatedly";
                        return OptimizeResult::RoundoffLimited;
                    }
                } else {
                    repeated_failures = 0;
                }
            }
            state.x = x_inner;

            // ── Refresh values/Jacobians and update multipliers
            evaluate_point(state.x, state.f, state.grad, g, Jg, h, Jh);
            for (std::size_t k = 0; k < lambda.size(); ++k) {
                lambda[k] += rho * h[k];
            }
            for (std::size_t i = 0; i < mu.size(); ++i) {
                mu[i] = std::max(0.0, mu[i] + rho * g[i]);
            }
            for (std::size_t j = 0; j < n; ++j) {
                if (bnd.hasLower(j)) {
                    mu_lo[j] = std::max(0.0, mu_lo[j] + rho * (bnd.lower[j] - state.x[j]));
                }
                if (bnd.hasUpper(j)) {
                    mu_up[j] = std::max(0.0, mu_up[j] + rho * (state.x[j] - bnd.upper[j]));
                }
            }

            // ── Primal feasibility and KKT stationarity with current duals
            const double inequality_violation = maxViolation(g);
            const double equality_violation = max_abs(h);
            const double bound_violation = bnd.violation(state.x);
            const double violation =
                std::max({inequality_violation, equality_violation, bound_violation});
            std::vector<double> stationarity = state.grad;
            for (std::size_t i = 0; i < g.size(); ++i) {
                if (mu[i] > 0.0) {
                    for (std::size_t j = 0; j < n; ++j) {
                        stationarity[j] += mu[i] * Jg[i * n + j];
                    }
                }
            }
            for (std::size_t k = 0; k < h.size(); ++k) {
                for (std::size_t j = 0; j < n; ++j) {
                    stationarity[j] += lambda[k] * Jh[k * n + j];
                }
            }
            // Project-free KKT: bound duals contribute like any inequality
            for (std::size_t j = 0; j < n; ++j) {
                if (bnd.hasLower(j) && mu_lo[j] > 0.0) {
                    stationarity[j] -= mu_lo[j];
                }
                if (bnd.hasUpper(j) && mu_up[j] > 0.0) {
                    stationarity[j] += mu_up[j];
                }
            }
            kkt_norm = max_abs(stationarity);

            if (violation <= kFeasTol && kkt_norm <= kKktTol) {
                state.message = "KKT conditions satisfied";
                state.ineq_multipliers = mu;
                state.eq_multipliers = lambda;
                return OptimizeResult::Success;
            }
            // NLopt ICM: complementarity-aware infeasibility
            //   max over inequalities of |max(g_j, -mu_j/rho)|, |h_k| for equalities.
            double icm = 0.0;
            for (std::size_t i = 0; i < g.size(); ++i) {
                icm = std::max(icm, std::fabs(std::max(g[i], -mu[i] / rho)));
            }
            for (std::size_t k = 0; k < h.size(); ++k) {
                icm = std::max(icm, std::fabs(h[k]));
            }
            // Escalate rho while the ICM stalls (NLopt: ICM > 0.5 * prev_ICM);
            // keep it moderate so the unconstrained inner solver stays healthy.
            if (icm > 0.5 * prev_icm && rho < 1e6) {
                rho *= m_growth;
            }
            prev_icm = icm;
        }

        if (polish()) {
            return OptimizeResult::Success;
        }
        state.message = "outer iteration limit reached";
        return OptimizeResult::Failure;
    }

    int m_memory;
    double m_rho0;
    double m_growth;
};

} // namespace quantape::math

#endif // AUGLAG_H
