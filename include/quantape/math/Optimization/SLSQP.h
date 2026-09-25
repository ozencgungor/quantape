#ifndef SLSQP_H
#define SLSQP_H

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <utility>
#include <vector>

#include "Constraint.h"
#include "OptimizerPrimitives.h"
#include "QpSolver.h"

namespace quantape::math {
/**
 * @file SLSQP.h
 * @brief Sequential quadratic programming with exact AD gradients/Jacobians
 *
 * Port of NLopt's LD_SLSQP semantics (Kraft's SQP):
 *   - damped BFGS approximation of the Lagrangian Hessian (identity start);
 *   - each iteration solves the QP subproblem
 *       min 0.5 d' B d + grad(f)' d
 *       s.t. A d + b <= 0 (linearized inequalities and box bounds)
 *            A d + b  = 0 (linearized equalities)
 *     with the dense active-set solver in QpSolver.h;
 *   - L1 exact-penalty merit line search with multiplier-driven penalty;
 *   - convergence when the QP step vanishes at a feasible point (KKT).
 *
 * Gradients and constraint Jacobians are exact: AD backends use the
 * scalar-generic shape (one reverse pass per evaluation, plus one per
 * constraint row); the double backend uses NLopt-style `f(x, grad)` and
 * `con(x, c, J)` callbacks, which may be analytic or use Stan on the
 * caller's side.
 *
 * @tparam DoubleT Derivative backend (var/fvar<...>, or double with
 *                 value+gradient callbacks)
 */
template <typename DoubleT>
    requires OptimizationScalar<DoubleT>
class SLSQP : public Optimizer<DoubleT, SLSQP<DoubleT>> {
public:
    using Base = Optimizer<DoubleT, SLSQP<DoubleT>>;

    explicit SLSQP(StopCriteria criteria = {}, double feasibility_tol = 1e-10)
        : Base(std::move(criteria)), m_feasibility_tol(feasibility_tol) {}

    double feasibilityTolerance() const { return m_feasibility_tol; }
    void setFeasibilityTolerance(double tol) { m_feasibility_tol = tol; }

    // ── Constrained entry points ──

    template <typename F, typename GI, typename GE = NoConstraint>
        requires ObjectiveEvaluator<F, DoubleT> && ConstraintEvaluator<GI, DoubleT> &&
                 ConstraintEvaluator<GE, DoubleT>
    OptimizeResult minimize(const F& f, const GI& inequality, const GE& equality,
                            const Bounds& bounds, std::vector<double>& x,
                            OptimizerState& state) const {
        if (x.empty()) {
            throw std::invalid_argument("SLSQP: empty parameter vector");
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

    /// Inequalities only
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

    // ── Unconstrained entry point (inherited base dispatch) ──

    // Internal, public for CRTP access
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
                      "SLSQP<double> requires a value+gradient objective: "
                      "double f(const std::vector<double>& x, std::vector<double>& grad)");
        static_assert(!std::is_same_v<DoubleT, double> || std::is_same_v<GI, NoConstraint> ||
                          ValueJacobianConstraint<GI>,
                      "SLSQP<double> requires inequality constraints as "
                      "void con(x, c, J) with row-major J");
        static_assert(!std::is_same_v<DoubleT, double> || std::is_same_v<GE, NoConstraint> ||
                          ValueJacobianConstraint<GE>,
                      "SLSQP<double> requires equality constraints as "
                      "void con(x, c, J) with row-major J");

        const std::size_t n = state.x.size();
        const StopCriteria& criteria = this->criteria();
        const Bounds bnd = bounds.empty() ? Bounds::unbounded(n) : bounds;
        const double step_tol = 1e-10;

        // NLopt contract: the start point must satisfy the box bounds
        if (!bnd.feasible(state.x)) {
            throw std::invalid_argument("SLSQP: initial point violates box bounds");
        }
        std::vector<DoubleT> theta_scratch(n);
        std::vector<double> fg_scratch(n);

        // Current constraint data
        std::vector<double> g_ineq, J_ineq, h_eq, J_eq;

        const auto eval_constraints = [&](const std::vector<double>& z, std::vector<double>& gi,
                                          std::vector<double>& ji, std::vector<double>& hh,
                                          std::vector<double>& jh) {
            if constexpr (std::is_same_v<GI, NoConstraint>) {
                gi.clear();
                ji.clear();
            } else {
                this->evalConstraint(inequality, z, gi, ji);
                ++state.evals;
                ++state.grad_evals;
            }
            if constexpr (std::is_same_v<GE, NoConstraint>) {
                hh.clear();
                jh.clear();
            } else {
                this->evalConstraint(equality, z, hh, jh);
                ++state.evals;
                ++state.grad_evals;
            }
        };

        const auto eval_value_grad_into = [&](const std::vector<double>& z,
                                              std::vector<double>& grad_out) {
            ++state.evals;
            ++state.grad_evals;
            return this->evalValueGradInto(f, z, theta_scratch, grad_out);
        };

        const auto max_abs = [](const std::vector<double>& v) {
            double m = 0.0;
            for (double x : v) {
                m = std::max(m, std::fabs(x));
            }
            return m;
        };

        eval_constraints(state.x, g_ineq, J_ineq, h_eq, J_eq);
        state.f = eval_value_grad_into(state.x, state.grad);

        const std::size_t mi = g_ineq.size();
        const std::size_t me = h_eq.size();

        const auto violation = [&]() {
            double v = std::max(maxViolation(g_ineq), max_abs(h_eq));
            v = std::max(v, bnd.violation(state.x));
            return v;
        };
        // L1 exact-penalty weights per constraint row (NLopt's update:
        // mu_j <- max(|lambda_j|, (mu_j + |lambda_j|)/2), over ALL rows)
        std::vector<double> mu_rows;
        const auto merit = [&](double f_value, const std::vector<double>& gi,
                               const std::vector<double>& hh, const std::vector<double>& x) {
            double m = f_value;
            std::size_t r = 0;
            for (std::size_t i = 0; i < gi.size(); ++i) {
                const double v = std::max(0.0, gi[i]);
                if (v > 0.0) {
                    m += mu_rows[r] * v;
                }
                ++r;
            }
            for (std::size_t k = 0; k < hh.size(); ++k) {
                m += mu_rows[r] * std::fabs(hh[k]);
                ++r;
            }
            for (std::size_t j = 0; j < n; ++j) {
                if (bnd.hasLower(j)) {
                    const double v = std::max(0.0, bnd.lower[j] - x[j]);
                    if (v > 0.0) {
                        m += mu_rows[r] * v;
                    }
                    ++r;
                }
                if (bnd.hasUpper(j)) {
                    const double v = std::max(0.0, x[j] - bnd.upper[j]);
                    if (v > 0.0) {
                        m += mu_rows[r] * v;
                    }
                    ++r;
                }
            }
            return m;
        };

        // Damped-BFGS Hessian (identity start, row-major)
        std::vector<double> B(n * n, 0.0);
        for (std::size_t j = 0; j < n; ++j) {
            B[j * n + j] = 1.0;
        }

        std::vector<double> x_try(n), g_try(mi), J_try(mi * n), h_try(me), H_try(me * n);
        std::vector<double> s(n), y(n), Bs(n), x_prev(n), grad_prev(n);
        std::size_t box_rows = 0;
        for (std::size_t j = 0; j < n; ++j) {
            box_rows += (bnd.hasLower(j) ? 1u : 0u) + (bnd.hasUpper(j) ? 1u : 0u);
        }
        QpProblem qp;
        int ntesf = 0;
        int ntesx = 0;

        for (int iteration = 0; iteration < 1000; ++iteration) {
            if (this->stopByEvalOrTime(state)) {
                state.message = "evaluation/time budget exhausted";
                return stopTime(criteria, state.start_time) ? OptimizeResult::MaxTimeReached
                                                            : OptimizeResult::MaxEvalReached;
            }
            if (state.f <= criteria.stopval) {
                state.message = "stop value reached";
                return OptimizeResult::StopvalReached;
            }

            // ── QP subproblem over linearized constraints and bounds.
            //    Buffers are reused across iterations: A is one flat m x n
            //    block, rows are overwritten in place (no per-iteration
            //    allocations; profiling showed this path at 20-30ms for
            //    n = 100 all-active bounds).
            const std::size_t m_total = mi + me + box_rows;
            qp.A.assign(m_total * n, 0.0);
            qp.b.resize(m_total);
            qp.equality.resize(m_total);
            qp.g = state.grad;
            qp.B = B;

            std::size_t r = 0;
            for (std::size_t i = 0; i < mi; ++i, ++r) {
                std::copy(J_ineq.begin() + static_cast<std::ptrdiff_t>(i * n),
                          J_ineq.begin() + static_cast<std::ptrdiff_t>((i + 1) * n),
                          qp.A.begin() + static_cast<std::ptrdiff_t>(r * n));
                qp.b[r] = g_ineq[i];
                qp.equality[r] = 0;
            }
            for (std::size_t k = 0; k < me; ++k, ++r) {
                std::copy(J_eq.begin() + static_cast<std::ptrdiff_t>(k * n),
                          J_eq.begin() + static_cast<std::ptrdiff_t>((k + 1) * n),
                          qp.A.begin() + static_cast<std::ptrdiff_t>(r * n));
                qp.b[r] = h_eq[k];
                qp.equality[r] = 1;
            }
            for (std::size_t j = 0; j < n; ++j) {
                if (bnd.hasLower(j)) {
                    // d_j >= lb - x  <=>  -d_j + (lb - x) <= 0
                    qp.A[r * n + j] = -1.0;
                    qp.b[r] = bnd.lower[j] - state.x[j];
                    qp.equality[r] = 0;
                    ++r;
                }
                if (bnd.hasUpper(j)) {
                    // d_j <= ub - x  <=>  d_j + (x - ub) <= 0
                    qp.A[r * n + j] = 1.0;
                    qp.b[r] = state.x[j] - bnd.upper[j];
                    qp.equality[r] = 0;
                    ++r;
                }
            }

            const QpResult qp_result = solveActiveSetQp(qp);
            if (!qp_result.success) {
                state.message = "QP subproblem failed";
                return violation() > m_feasibility_tol ? OptimizeResult::Infeasible
                                                       : OptimizeResult::Failure;
            }
            const std::vector<double>& d = qp_result.d;
            const double d_norm = max_abs(d);

            if (d_norm <= step_tol) {
                if (violation() <= m_feasibility_tol) {
                    state.message = "KKT conditions satisfied";
                    // Export the final multipliers for the IFT layer
                    // (ImplicitFunction.h): QP rows are [ineq][eq][bounds].
                    state.ineq_multipliers.assign(qp_result.lambda.begin(),
                                                  qp_result.lambda.begin() +
                                                      static_cast<std::ptrdiff_t>(mi));
                    state.eq_multipliers.assign(
                        qp_result.lambda.begin() + static_cast<std::ptrdiff_t>(mi),
                        qp_result.lambda.begin() + static_cast<std::ptrdiff_t>(mi + me));
                    return OptimizeResult::Success;
                }
                state.message = "no feasible descent direction found";
                return OptimizeResult::Infeasible;
            }

            // ── NLopt mu update: mu_j <- max(|lam_j|, (mu_j + |lam_j|)/2)
            if (mu_rows.size() != qp.rows()) {
                mu_rows.assign(qp.rows(), 0.0);
            }
            for (std::size_t i = 0; i < qp.rows(); ++i) {
                const double abs_lam = std::fabs(qp_result.lambda[i]);
                mu_rows[i] = std::max(abs_lam, 0.5 * (mu_rows[i] + abs_lam));
            }

            // ── L1-merit backtracking line search
            const double merit_0 = merit(state.f, g_ineq, h_eq, state.x);
            const double f_prev = state.f;
            x_prev = state.x;
            grad_prev = state.grad;
            double alpha = 1.0;
            bool accepted = false;
            for (int ls = 0; ls < 30; ++ls) {
                for (std::size_t j = 0; j < n; ++j) {
                    x_try[j] = state.x[j] + alpha * d[j];
                }
                const double f_try = eval_value_grad_into(x_try, fg_scratch);
                eval_constraints(x_try, g_try, J_try, h_try, H_try);
                const double merit_try = merit(f_try, g_try, h_try, x_try);
                if (merit_try < merit_0 - 1e-12 * (1.0 + std::fabs(merit_0))) {
                    accepted = true;
                    // move the new data into place
                    state.f = f_try;
                    state.grad = std::move(fg_scratch);
                    break;
                }
                alpha *= 0.5;
            }
            if (!accepted) {
                state.message = "merit line search failed";
                return OptimizeResult::RoundoffLimited;
            }

            // ── Damped BFGS update of the Lagrangian Hessian
            for (std::size_t j = 0; j < n; ++j) {
                s[j] = x_try[j] - state.x[j];
            }
            // y = gradL(new) - gradL(old) with the same multipliers
            for (std::size_t j = 0; j < n; ++j) {
                y[j] = state.grad[j] - grad_prev[j];
            }
            for (std::size_t i = 0; i < mi; ++i) {
                const double lam = qp_result.lambda[i];
                if (lam == 0.0) {
                    continue;
                }
                for (std::size_t j = 0; j < n; ++j) {
                    y[j] += lam * (J_try[i * n + j] - J_ineq[i * n + j]);
                }
            }
            for (std::size_t k = 0; k < me; ++k) {
                const double lam = qp_result.lambda[mi + k];
                if (lam == 0.0) {
                    continue;
                }
                for (std::size_t j = 0; j < n; ++j) {
                    y[j] += lam * (H_try[k * n + j] - J_eq[k * n + j]);
                }
            }

            // Bs = B s
            for (std::size_t r = 0; r < n; ++r) {
                double sum = 0.0;
                for (std::size_t c = 0; c < n; ++c) {
                    sum += B[r * n + c] * s[c];
                }
                Bs[r] = sum;
            }
            double sBs = 0.0;
            double sy = 0.0;
            for (std::size_t j = 0; j < n; ++j) {
                sBs += s[j] * Bs[j];
                sy += s[j] * y[j];
            }
            if (sBs > 1e-14) {
                if (sy < 0.2 * sBs) {
                    const double theta = 0.8 * sBs / (sBs - sy);
                    for (std::size_t j = 0; j < n; ++j) {
                        y[j] = theta * y[j] + (1.0 - theta) * Bs[j];
                    }
                    sy = 0.0;
                    for (std::size_t j = 0; j < n; ++j) {
                        sy += s[j] * y[j];
                    }
                }
                if (sy > 1e-14) {
                    for (std::size_t r = 0; r < n; ++r) {
                        for (std::size_t c = 0; c < n; ++c) {
                            B[r * n + c] += y[r] * y[c] / sy - Bs[r] * Bs[c] / sBs;
                        }
                    }
                }
            }

            // ── Commit the step
            state.x = x_try;
            g_ineq = g_try;
            J_ineq = J_try;
            h_eq = h_try;
            J_eq = H_try;
            ++state.iterations;

            // ── Consecutive-pass ftol / xtol tests
            if (criteria.ftol_rel > 0.0 || criteria.ftol_abs > 0.0) {
                ntesf = stopFtol(criteria, state.f, f_prev) ? ntesf + 1 : 0;
                if (ntesf >= std::max(1, criteria.mtesf)) {
                    state.message = "function tolerance";
                    return OptimizeResult::FtolReached;
                }
            }
            if (criteria.xtol_rel > 0.0 || criteria.xtol_abs > 0.0) {
                ntesx = stopX(criteria, state.x, x_prev) ? ntesx + 1 : 0;
                if (ntesx >= std::max(1, criteria.mtesx)) {
                    state.message = "parameter tolerance";
                    return OptimizeResult::XtolReached;
                }
            }
        }

        state.message = "iteration limit reached";
        return OptimizeResult::Failure;
    }

    double m_feasibility_tol;
};

} // namespace quantape::math

#endif // SLSQP_H
