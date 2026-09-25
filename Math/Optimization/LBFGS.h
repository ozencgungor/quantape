#ifndef LBFGS_H
#define LBFGS_H

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <utility>
#include <vector>

#include "LineSearch.h"
#include "OptimizerPrimitives.h"

namespace Math {
/**
 * @file LBFGS.h
 * @brief Limited-memory BFGS (PLIS semantics) with exact AD gradients
 *
 * Algorithm (NLopt's LD_LBFGS, Luksan PLIS, translated to the standard
 * two-loop formulation):
 *   - direction by the two-loop recursion over the last `count` curvature
 *     pairs with the scaling gamma = s.y / y.y (NLopt's b/a);
 *   - restart to steepest descent when the memory is empty, the latest
 *     curvature fails (s.y <= 0), or the uniform-descent test
 *     g.d <= -told ||g|| ||d|| fails;
 *   - strong-Wolfe line search (LineSearch.h), constants matching PS1L01;
 *   - stop criteria and two-consecutive-pass ftol/xtol tests from
 *     OptimizerPrimitives.h.
 *
 * DoubleT selects the gradient backend:
 *   - `var` / `fvar<...>`: scalar-generic objective, one reverse pass per
 *     evaluation (OptimizerStanPrimitives.h);
 *   - `double`: AD-free. The objective must supply its gradient in the same
 *     call (ValueGradObjective, NLopt's `objgrad`); that callback may be
 *     analytic or use Stan internally on the caller's side.
 * Iteration state is plain double, and all work buffers are local to
 * minimizeImpl (state-free, re-entrant).
 */
template <typename DoubleT>
    requires OptimizationScalar<DoubleT>
class LBFGS : public Optimizer<DoubleT, LBFGS<DoubleT>> {
public:
    using Base = Optimizer<DoubleT, LBFGS<DoubleT>>;

    /**
     * @param criteria Stop criteria
     * @param memory L-BFGS curvature pairs to keep; 0 = heuristic
     *               (max(10, n), capped by maxeval)
     */
    explicit LBFGS(StopCriteria criteria = {}, int memory = 0)
        : Base(std::move(criteria)), m_memory(memory) {}

    // Inspectors / modifiers
    int memory() const { return m_memory; }
    void setMemory(int memory) { m_memory = memory; }

    const LineSearchOptions& lineSearchOptions() const { return m_options; }
    void setLineSearchOptions(const LineSearchOptions& options) { m_options = options; }

    // Internal, public for CRTP access (like Solver1D's solveImpl)
    template <typename F>
        requires ObjectiveEvaluator<F, DoubleT>
    OptimizeResult minimizeImpl(const F& f, OptimizerState& state) const {
        // AD backends need the scalar-generic objective; the double backend
        // needs a value+gradient callback (a value-only double objective has
        // no gradient to iterate with).
        static_assert(!std::is_same_v<DoubleT, double> || ValueGradObjective<F>,
                      "LBFGS<double> requires a value+gradient objective: "
                      "double f(const std::vector<double>& x, std::vector<double>& grad)");

        const std::size_t n = state.x.size();
        const std::size_t m = resolveMemory(n);
        const StopCriteria& criteria = this->criteria();
        const int mtesf = std::max(1, criteria.mtesf);
        const int mtesx = std::max(1, criteria.mtesx);

        // Work buffers, allocated once per minimize() call
        std::vector<double> x_prev(n), g_prev(n), direction(n);
        std::vector<double> s_hist(m * n, 0.0), y_hist(m * n, 0.0);
        std::vector<double> rho(m, 0.0), alpha_hist(m, 0.0);
        std::vector<DoubleT> theta_scratch(n); // AD parameter scratch (unused for double)
        std::size_t count = 0;                 // stored curvature pairs
        std::size_t next = 0;                  // ring slot for the next pair

        // Evaluation closure: value + gradient written into the caller's
        // buffer (no per-evaluation allocation beyond the tape nodes)
        const auto evaluate_into = [&](const std::vector<double>& z,
                                       std::vector<double>& grad_out) {
            ++state.evals;
            ++state.grad_evals;
            return this->evalValueGradInto(f, z, theta_scratch, grad_out);
        };

        // Initial evaluation
        state.f = evaluate_into(state.x, state.grad);

        int ntesf = 0;
        int ntesx = 0;
        int failures = 0;

        const auto dot_pair = [&](const std::vector<double>& vec, const std::vector<double>& hist,
                                  std::size_t p) {
            double sum = 0.0;
            for (std::size_t j = 0; j < n; ++j) {
                sum += vec[j] * hist[p * n + j];
            }
            return sum;
        };

        while (true) {
            // ── Budget / gradient / stop-value tests
            if (this->stopByGradient(state.grad)) {
                state.message = "gradient tolerance";
                return OptimizeResult::GradientTolReached;
            }
            if (state.f <= criteria.stopval) {
                state.message = "stop value reached";
                return OptimizeResult::StopvalReached;
            }
            if (this->stopByEvalOrTime(state)) {
                state.message = "evaluation/time budget exhausted";
                return stopTime(criteria, state.start_time) ? OptimizeResult::MaxTimeReached
                                                            : OptimizeResult::MaxEvalReached;
            }

            // ── L-BFGS direction: two-loop recursion (q, then r, then d = -r)
            direction = state.grad;
            if (count > 0) {
                for (std::size_t i = 0; i < count; ++i) {
                    const std::size_t p = (next + m - 1 - i) % m; // i = 0 is newest
                    const double a = rho[p] * dot_pair(direction, s_hist, p);
                    alpha_hist[i] = a;
                    for (std::size_t j = 0; j < n; ++j) {
                        direction[j] -= a * y_hist[p * n + j];
                    }
                }

                // Scaling gamma = s.y / y.y of the newest pair (NLopt's b/a)
                const std::size_t p0 = (next + m - 1) % m;
                double sy = 0.0;
                double yy = 0.0;
                for (std::size_t j = 0; j < n; ++j) {
                    sy += s_hist[p0 * n + j] * y_hist[p0 * n + j];
                    yy += y_hist[p0 * n + j] * y_hist[p0 * n + j];
                }
                const double gamma = sy / yy; // sy > 0 by construction, yy > 0
                for (std::size_t j = 0; j < n; ++j) {
                    direction[j] *= gamma;
                }

                for (std::size_t i = count; i-- > 0;) {
                    const std::size_t p = (next + m - 1 - i) % m;
                    const double beta = rho[p] * dot_pair(direction, y_hist, p);
                    const double a = alpha_hist[i];
                    for (std::size_t j = 0; j < n; ++j) {
                        direction[j] += (a - beta) * s_hist[p * n + j];
                    }
                }
            }
            for (std::size_t j = 0; j < n; ++j) {
                direction[j] = -direction[j];
            }

            // Normalize the steepest-descent direction when the memory is
            // empty: an unscaled -g explodes on badly conditioned problems
            // (gradient ~ 1e16 makes the line search fail outright).
            if (count == 0) {
                double g_max = 1.0;
                for (std::size_t j = 0; j < n; ++j) {
                    g_max = std::max(g_max, std::fabs(state.grad[j]));
                }
                for (std::size_t j = 0; j < n; ++j) {
                    direction[j] /= g_max;
                }
            }

            // ── Uniform descent test (PLIS told = 1e-4): fall back to
            //    steepest descent and reset the curvature memory
            const double gd = detail::dotProduct(state.grad, direction);
            const double norm_g = detail::euclideanNorm(state.grad);
            const double norm_d = detail::euclideanNorm(direction);
            if (!(norm_d > 0.0) || !(gd < -1e-4 * norm_g * norm_d)) {
                for (std::size_t j = 0; j < n; ++j) {
                    direction[j] = -state.grad[j];
                }
                count = 0;
                next = 0;
            }

            // ── Line search along direction
            x_prev = state.x;
            g_prev = state.grad;
            const double f_prev = state.f;

            LineSearchResult ls =
                wolfeLineSearch([&](const std::vector<double>& z,
                                    std::vector<double>& g) { return evaluate_into(z, g); },
                                state.x, direction, f_prev, g_prev, m_options);

            if (ls.success) {
                failures = 0;
            } else if (ls.alpha > 0.0 && ls.f < f_prev) {
                // Decreasing point but no Wolfe certificate: accept, reset the
                // memory, and give up after repeated failures.
                ++failures;
                count = 0;
                next = 0;
                if (failures >= 3) {
                    state.message = "line search failed to satisfy Wolfe conditions";
                    return OptimizeResult::RoundoffLimited;
                }
            } else {
                state.message = "line search made no progress";
                return OptimizeResult::RoundoffLimited;
            }

            // ── Accept the step
            for (std::size_t j = 0; j < n; ++j) {
                state.x[j] = x_prev[j] + ls.alpha * direction[j];
            }
            state.f = ls.f;
            if (ls.grad.size() == n) {
                state.grad = std::move(ls.grad);
            } else {
                state.f = evaluate_into(state.x, state.grad);
            }
            ++state.iterations;

            // ── Curvature pair (s, y); drop the memory on b <= 0
            double b = 0.0;
            for (std::size_t j = 0; j < n; ++j) {
                b += (state.x[j] - x_prev[j]) * (state.grad[j] - g_prev[j]);
            }
            if (b > 0.0 && std::isfinite(b)) {
                const std::size_t p = next;
                for (std::size_t j = 0; j < n; ++j) {
                    s_hist[p * n + j] = state.x[j] - x_prev[j];
                    y_hist[p * n + j] = state.grad[j] - g_prev[j];
                }
                rho[p] = 1.0 / b;
                next = (next + 1) % m;
                count = std::min(count + 1, m);
            } else {
                count = 0;
                next = 0;
            }

            // ── Consecutive-pass ftol / xtol tests
            if (criteria.ftol_rel > 0.0 || criteria.ftol_abs > 0.0) {
                ntesf = stopFtol(criteria, state.f, f_prev) ? ntesf + 1 : 0;
                if (ntesf >= mtesf) {
                    state.message = "function tolerance";
                    return OptimizeResult::FtolReached;
                }
            }
            if (criteria.xtol_rel > 0.0 || criteria.xtol_abs > 0.0) {
                ntesx = stopX(criteria, state.x, x_prev) ? ntesx + 1 : 0;
                if (ntesx >= mtesx) {
                    state.message = "parameter tolerance";
                    return OptimizeResult::XtolReached;
                }
            }
        }
    }

private:
    /// Curvature pairs to keep; 0 = heuristic: max(10, n), capped by maxeval
    std::size_t resolveMemory(std::size_t n) const {
        int memory = m_memory;
        if (memory <= 0) {
            memory = static_cast<int>(std::max<std::size_t>(10, n));
        }
        if (this->criteria().maxeval > 0) {
            memory = std::min(memory, std::max(1, this->criteria().maxeval));
        }
        return static_cast<std::size_t>(std::max(memory, 1));
    }

    int m_memory;
    LineSearchOptions m_options;
};

} // namespace Math

#endif // LBFGS_H
