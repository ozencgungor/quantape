#ifndef TNEWTON_H
#define TNEWTON_H

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
 * @file TNewton.h
 * @brief Truncated Newton with EXACT Hessian-vector products (PNET semantics)
 *
 * Mirrors NLopt's PNET (luksan/pnet.c, algorithm NLOPT_LD_TNEWTON*), with the
 * finite-difference Hv replaced by exact forward-over-reverse HVPs (one fvar
 * tape per inner step, no extra gradient noise floor).
 *
 * Inner CG solves H d = -g inexactly with the PNET forcing term
 *   eta = min(0.8, sqrt(||g||_2), 1/nit)^2,   eta >= 1e-14
 * and stops when ||r||^2 <= eta * ||r0||^2 after at most n + 3 inner steps.
 *
 * PNET semantics preserved:
 *  - breakdown test: PNET rejects curvature alf <= 1/eta9 = 1e-120 (safe only
 *    because its FD Hv has a noise floor). With EXACT products the curvature
 *    can legitimately be ~1e-17 on tiny residuals, so the rejection is judged
 *    RELATIVE to the vector scales: pAp > 1e-14 * ||p|| * ||Ap||. Breakdown on
 *    the FIRST inner step falls back to the steepest-descent direction s = -g
 *    (snorm = gnorm), exactly as PNET does.
 *  - descent test: PNET's uniform descent criterion
 *        g . s + told * ||g||_2 * ||s||_2 <= 0,   told = 1e-4
 *    (an absolute test like gd < -1e-10 rejects the legitimate ~1e-16
 *    Newton directional derivative near the minimum and forces linear
 *    convergence).
 *  - restart semantics (mos1): CG breakdown on the FIRST inner step always
 *    falls back to s = -g in both modes (PNET iterd = 0). Only a failed
 *    descent test distinguishes: restart == true (NLOPT_LD_TNEWTON_RESTART)
 *    uses s = -g; restart == false (plain NLOPT_LD_TNEWTON) recomputes the
 *    CG direction and fails with OptimizeResult::Failure (PNET iterm = -10)
 *    if it still is not a descent direction.
 *  - line search: PS1L01-style scaled step bounds
 *        rmin = 1e-10 * ||g||_2 / ||s||_2
 *        rmax = min(1e10 * ||g||_2 / ||s||_2, 1e16 / ||s||_2)
 *    which also replace any direction normalization (a huge direction is
 *    capped by rmax, not rescaled).
 *  - forcing-term norm: PNET uses the 2-norm ||g||_2 (termination stays on
 *    the max partial derivative, as in PNET's pyfut1).
 *
 * v1 uses an identity preconditioner (mos2 = 1); the L-BFGS preconditioner
 * (mos2 = 2, NLOPT_LD_TNEWTON_PRECOND*) is a follow-up.
 *
 * Requires an AD backend: the objective must accept std::vector<fvar<var>>
 * in addition to DoubleT. Iteration state is double; all work is local to
 * minimizeImpl (re-entrant).
 */
template <typename DoubleT>
    requires HvpBackend<DoubleT>
class TNewton : public Optimizer<DoubleT, TNewton<DoubleT>> {
public:
    using Base = Optimizer<DoubleT, TNewton<DoubleT>>;

    explicit TNewton(StopCriteria criteria = {}, bool restart = true)
        : Base(std::move(criteria)), m_restart(restart) {}

    const LineSearchOptions& lineSearchOptions() const { return m_options; }
    void setLineSearchOptions(const LineSearchOptions& options) { m_options = options; }

    // Internal, public for CRTP access (like Solver1D's solveImpl)
    template <typename F>
        requires ObjectiveEvaluator<F, DoubleT>
    OptimizeResult minimizeImpl(const F& f, OptimizerState& state) const {
        const std::size_t n = state.x.size();
        const StopCriteria& criteria = this->criteria();
        const int mtesf = std::max(1, criteria.mtesf);
        const int mtesx = std::max(1, criteria.mtesx);

        std::vector<DoubleT> theta_scratch(n);
        std::vector<double> x_prev(n), g_prev(n), direction(n);
        std::vector<double> r(n), p(n), Ap(n);

        const auto evaluate_into = [&](const std::vector<double>& z,
                                       std::vector<double>& grad_out) {
            ++state.evals;
            ++state.grad_evals;
            return this->evalValueGradInto(f, z, theta_scratch, grad_out);
        };

        state.f = evaluate_into(state.x, state.grad);

        int ntesf = 0;
        int ntesx = 0;
        int failures = 0;

        // PNET truncated-CG direction computation; returns false when the CG
        // failed on its first step (direction left as s = -g, PNET iterd = 0)
        const auto cg_direction = [&]() {
            // PNET forcing term: eta = min(0.8, sqrt(||g||_2), 1/nit)^2
            const double gnorm = std::sqrt(dot(state.grad, state.grad));
            double eta = std::min(0.8, std::sqrt(gnorm));
            if (state.iterations > 0) {
                eta = std::min(eta, 1.0 / static_cast<double>(state.iterations));
            }
            eta = std::max(eta * eta, 1e-14);

            bool first_step_breakdown = false;
            for (std::size_t j = 0; j < n; ++j) {
                r[j] = -state.grad[j];
                p[j] = r[j];
                direction[j] = 0.0;
            }
            double rs = dot(r, r);
            const double rs0 = rs;
            const std::size_t max_inner = n + 3;

            for (std::size_t inner = 0; inner < max_inner; ++inner) {
                if (rs <= 1e-30) {
                    break;
                }
                Ap = this->evalHvp(f, state.x, p);
                ++state.evals; // one objective evaluation per HVP (fvar pass)
                ++state.grad_evals;
                const double pAp = dot(p, Ap);
                // PNET rejects alf <= 1/eta9 (1e-120, absolute); with exact
                // products the curvature may be legitimately ~1e-17 on tiny
                // residuals, so judge relative to the vector scales.
                const double p_norm = std::sqrt(dot(p, p));
                const double ap_norm = std::sqrt(dot(Ap, Ap));
                if (!(pAp > 1e-14 * p_norm * ap_norm) || !std::isfinite(pAp)) {
                    if (inner == 0) {
                        first_step_breakdown = true;
                    }
                    break;
                }
                const double alpha = rs / pAp;
                for (std::size_t j = 0; j < n; ++j) {
                    direction[j] += alpha * p[j];
                    r[j] -= alpha * Ap[j];
                }
                const double rs_new = dot(r, r);
                if (rs_new <= eta * rs0) {
                    rs = rs_new;
                    break;
                }
                const double beta = rs_new / rs;
                for (std::size_t j = 0; j < n; ++j) {
                    p[j] = r[j] + beta * p[j];
                }
                rs = rs_new;
            }
            return first_step_breakdown;
        };

        // PNET uniform descent criterion (told = 1e-4)
        const auto is_descent = [&]() {
            const double gnorm = std::sqrt(dot(state.grad, state.grad));
            const double snorm = std::sqrt(dot(direction, direction));
            if (snorm <= 0.0) {
                return false;
            }
            return dot(state.grad, direction) + 1e-4 * gnorm * snorm <= 0.0;
        };

        // PNET: failed descent test / first-step breakdown -> s = -g
        const auto steepest_descent_fallback = [&]() {
            for (std::size_t j = 0; j < n; ++j) {
                direction[j] = -state.grad[j];
            }
        };

        while (true) {
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

            // ── Truncated CG for H d = -g (PNET forcing term, 2-norm)
            const bool first_step_breakdown = cg_direction();

            // ── Descent test (PNET's told = 1e-4 uniform criterion)
            if (first_step_breakdown) {
                // PNET: CG breakdown on the FIRST inner step -> steepest
                // descent in BOTH mos1 modes (iterd = 0, s = -g, snorm = gnorm);
                // this is not the irest/descent-test restart path.
                steepest_descent_fallback();
            } else if (!is_descent()) {
                if (m_restart) {
                    // NLOPT_LD_TNEWTON_RESTART (mos1 = 2): steepest descent
                    steepest_descent_fallback();
                } else {
                    // Plain NLOPT_LD_TNEWTON (mos1 = 1): recompute the CG once
                    // (PNET re-enters direction determination with irest); the
                    // exact HVP makes the second CG identical, so a second
                    // failure terminates with PNET's iterm = -10.
                    cg_direction();
                    if (!is_descent()) {
                        state.message = "restart failed (no descent direction)";
                        return OptimizeResult::Failure;
                    }
                }
            }

            // ── Line search with PNET-scaled step bounds
            const double gnorm = std::sqrt(dot(state.grad, state.grad));
            const double snorm = std::sqrt(dot(direction, direction));
            LineSearchOptions ls_options = m_options;
            if (snorm > 0.0) {
                ls_options.alpha_min = std::max(ls_options.alpha_min, 1e-10 * gnorm / snorm);
                ls_options.alpha_max =
                    std::min(ls_options.alpha_max, std::min(1e10 * gnorm / snorm, 1e16 / snorm));
            }

            x_prev = state.x;
            g_prev = state.grad;
            const double f_prev = state.f;

            LineSearchResult ls =
                wolfeLineSearch([&](const std::vector<double>& z,
                                    std::vector<double>& g) { return evaluate_into(z, g); },
                                state.x, direction, f_prev, g_prev, ls_options);

            if (ls.success) {
                failures = 0;
            } else if (ls.alpha > 0.0 && ls.f < f_prev) {
                ++failures;
                if (failures >= 3) {
                    state.message = "line search failed to satisfy Wolfe conditions";
                    return OptimizeResult::RoundoffLimited;
                }
            } else {
                state.message = "line search made no progress";
                return OptimizeResult::RoundoffLimited;
            }

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
    static double dot(const std::vector<double>& a, const std::vector<double>& b) {
        double s = 0.0;
        for (std::size_t i = 0; i < a.size(); ++i) {
            s += a[i] * b[i];
        }
        return s;
    }

    bool m_restart;
    LineSearchOptions m_options;
};

} // namespace Math

#endif // TNEWTON_H
