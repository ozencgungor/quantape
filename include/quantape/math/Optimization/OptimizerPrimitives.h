#ifndef OPTIMIZER_PRIMITIVES_H
#define OPTIMIZER_PRIMITIVES_H

#include "quantape/math/Autodiff/PrimalExtraction.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <concepts>
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

namespace quantape::math {
/**
 * @file OptimizerPrimitives.h
 * @brief Shared building blocks for the optimizer classes (Stan-free)
 *
 * Mirrors the design of Math/Solvers and Math/Interpolations:
 * - an `Optimizer<DoubleT, Impl>` base holding configuration only, with
 *   all working state local to a minimize() call (thread-safe, re-entrant);
 * - `DoubleT` selects the derivative backend: `double` = value-only,
 *   `var` = exact gradients, `fvar<...>` = gradients/HVPs; the AD entry
 *   points are declared here and defined in OptimizerStanPrimitives.h;
 * - algorithms derive as `class LBFGS<DoubleT> : public Optimizer<DoubleT,
 *   LBFGS<DoubleT>>` and implement `minimizeImpl(objective, state)`.
 *
 * Stop-criteria semantics follow NLopt (see docs/ad_optimizers.md §2.2):
 * relative/absolute f and x tolerances, consecutive-pass counters, weighted
 * x-norms, gradient tolerance, stop value, evaluation and time budgets.
 */

// ============================================================================
// Result codes
// ============================================================================

/// Outcome of an optimization run (nlopt_result semantics)
enum class OptimizeResult {
    Success,
    StopvalReached,
    FtolReached,
    XtolReached,
    GradientTolReached,
    MaxEvalReached,
    MaxTimeReached,
    RoundoffLimited,
    ForcedStop,
    Infeasible,
    Failure
};

inline const char* to_string(OptimizeResult result) {
    switch (result) {
        case OptimizeResult::Success:
            return "Success";
        case OptimizeResult::StopvalReached:
            return "StopvalReached";
        case OptimizeResult::FtolReached:
            return "FtolReached";
        case OptimizeResult::XtolReached:
            return "XtolReached";
        case OptimizeResult::GradientTolReached:
            return "GradientTolReached";
        case OptimizeResult::MaxEvalReached:
            return "MaxEvalReached";
        case OptimizeResult::MaxTimeReached:
            return "MaxTimeReached";
        case OptimizeResult::RoundoffLimited:
            return "RoundoffLimited";
        case OptimizeResult::ForcedStop:
            return "ForcedStop";
        case OptimizeResult::Infeasible:
            return "Infeasible";
        case OptimizeResult::Failure:
            return "Failure";
    }
    return "Unknown";
}

// ============================================================================
// Configuration and state
// ============================================================================

/**
 * @brief Stop criteria (NLopt semantics)
 *
 * `mtesx`/`mtesf` require that many consecutive passes before stopping, which
 * guards against premature exits on flat or noisy stretches. `x_weights`
 * (empty = all ones) scales the relative x-norm coordinate-wise.
 */
struct StopCriteria {
    double ftol_rel = 0.0;
    double ftol_abs = 0.0;
    double xtol_rel = 0.0;
    double xtol_abs = 0.0;
    double grad_tol = 1e-10;
    /// Stop when f <= stopval; -max is the "never" sentinel (finite to stay
    /// well-defined under the project's release -ffast-math flags).
    double stopval = -std::numeric_limits<double>::max();
    int maxeval = 0;      ///< 0 = unlimited
    double maxtime = 0.0; ///< seconds, 0 = unlimited
    int mtesx = 2;
    int mtesf = 2;
    std::vector<double> x_weights; ///< optional per-coordinate weights
};

/// Running state of one minimize() call
struct OptimizerState {
    std::vector<double> x;                         ///< current iterate (returned to the caller)
    std::vector<double> grad;                      ///< current gradient (empty for double-only)
    double f = std::numeric_limits<double>::max(); ///< current objective value
    std::size_t evals = 0;                         ///< objective evaluations
    std::size_t grad_evals = 0;                    ///< gradient evaluations
    std::size_t iterations = 0;
    double start_time = 0.0; ///< seconds, see quantape::math::nowSeconds()
    std::string message;     ///< human-readable outcome detail

    /// Final inequality multipliers (lambda >= 0, Lagrangian f + sum lambda_i g_i)
    /// — exported by SLSQP/AUGLAG at the KKT-satisfied exit for the IFT layer
    /// (ImplicitFunction.h); empty for unconstrained runs.
    std::vector<double> ineq_multipliers;
    /// Final equality multipliers (nu, Lagrangian f + sum nu_k h_k)
    std::vector<double> eq_multipliers;
};

/// Monotonic clock in seconds (used for maxtime)
inline double nowSeconds() {
    using clock = std::chrono::steady_clock;
    return std::chrono::duration<double>(clock::now().time_since_epoch()).count();
}

// ============================================================================
// Concepts
// ============================================================================

namespace detail {
/// Recursive detection of a val() chain ending in an arithmetic type
/// (double <- var <- `fvar<var>` <- ...). A trait rather than a concept because
/// C++20 forbids self-referential concept definitions.
template <typename T>
struct is_optimization_scalar : std::is_arithmetic<T> {};

template <typename T>
    requires requires(const T& x) { x.val(); }
struct is_optimization_scalar<T>
    : is_optimization_scalar<std::remove_cvref_t<decltype(std::declval<const T&>().val())>> {};
} // namespace detail

/// Scalar accepted by the optimizers: double or an AD scalar exposing val()
template <typename T>
concept OptimizationScalar = quantape::math::detail::is_optimization_scalar<T>::value;

/// Backend that can produce exact gradients (anything but double)
template <typename DoubleT>
concept GradientBackend = OptimizationScalar<DoubleT> && !std::is_same_v<DoubleT, double>;

/// Backend that can produce exact HVPs (var/fvar; hvp runs `fvar<var>` internally)
template <typename DoubleT>
concept HvpBackend = GradientBackend<DoubleT>;

/**
 * @brief Scalar-generic objective over a parameter vector
 *
 * The same call site is used with double, var and fvar<...>; the exact return
 * type is required so a double-returning lambda is rejected by the var
 * instantiation instead of silently slicing the AD graph.
 */
template <typename F, typename S>
concept VectorObjective = requires(const F& f, const std::vector<S>& x) {
    { f(x) } -> std::same_as<S>;
};

/**
 * @brief Scalar-generic constraint writer: fills @p out with g(x), g <= 0
 *        (resizing as needed).
 */
template <typename F, typename S>
concept VectorConstraint =
    requires(const F& con, const std::vector<S>& x, std::vector<S>& out) { con(x, out); };

/**
 * @brief NLopt-style objective for the double backend: returns f(x) and fills
 *        the gradient in the same call.
 *
 *   double f(const std::vector<double>& x, std::vector<double>& grad);
 *
 * The callback may be fully analytic or use AD internally (owned by the
 * caller's translation unit, e.g. via quantape::math::detail::valueGrad or hand-rolled
 * Stan code). This header and the optimizer stay Stan-free either way.
 */
template <typename F>
concept ValueGradObjective =
    requires(const F& f, const std::vector<double>& x, std::vector<double>& grad) {
        { f(x, grad) } -> std::same_as<double>;
    };

/**
 * @brief NLopt-style constraint for the double backend: values + row-major
 *        Jacobian in one call.
 *
 *   void con(const std::vector<double>& x, std::vector<double>& c,
 *            std::vector<double>& J);   // J is m rows x n cols, row-major
 */
template <typename F>
concept ValueJacobianConstraint =
    requires(const F& con, const std::vector<double>& x, std::vector<double>& c,
             std::vector<double>& J) { con(x, c, J); };

/**
 * @brief Entry-point objective constraint for the optimizers
 *
 * AD backends use the scalar-generic shape (VectorObjective); the double
 * backend additionally accepts a value + gradient callback
 * (ValueGradObjective, NLopt's `objgrad`).
 */
template <typename F, typename S>
concept ObjectiveEvaluator =
    VectorObjective<F, S> || (std::is_same_v<S, double> && ValueGradObjective<F>);

// ============================================================================
// Stop tests (NLopt semantics)
// ============================================================================

namespace detail {
/// |new - old| < abs  ||  |new - old| < rel * 0.5 * (|new| + |old|)
/// || (rel > 0 && new == old)  — the last clause catches 0 == 0.
inline bool relStop(double old_value, double new_value, double rel, double abs) {
    if (!std::isfinite(old_value)) {
        return false;
    }
    const double diff = std::fabs(new_value - old_value);
    return diff < abs || diff < rel * 0.5 * (std::fabs(new_value) + std::fabs(old_value)) ||
           (rel > 0.0 && new_value == old_value);
}

/// Weighted L1 norm of v, or of v - base when base is non-empty
inline double weightedNorm(const std::vector<double>& v, const std::vector<double>& base,
                           const std::vector<double>& weights) {
    double sum = 0.0;
    for (std::size_t i = 0; i < v.size(); ++i) {
        const double w = weights.size() == v.size() ? weights[i] : 1.0;
        sum += w * std::fabs(base.empty() ? v[i] : v[i] - base[i]);
    }
    return sum;
}
} // namespace detail

/// Function-value test: relative or absolute change below tolerance
inline bool stopFtol(const StopCriteria& criteria, double f, double old_f) {
    return quantape::math::detail::relStop(old_f, f, criteria.ftol_rel, criteria.ftol_abs);
}

/// Iterate test: weighted |dx| below xtol_rel * weighted |x| (or all |dx| < xtol_abs)
inline bool stopX(const StopCriteria& criteria, const std::vector<double>& x,
                  const std::vector<double>& old_x) {
    if (criteria.xtol_rel > 0.0) {
        const double diff = quantape::math::detail::weightedNorm(x, old_x, criteria.x_weights);
        const double norm = quantape::math::detail::weightedNorm(x, {}, criteria.x_weights);
        if (diff < criteria.xtol_rel * norm) {
            return true;
        }
    }
    if (criteria.xtol_abs > 0.0) {
        for (std::size_t i = 0; i < x.size(); ++i) {
            if (std::fabs(x[i] - old_x[i]) >= criteria.xtol_abs) {
                return false;
            }
        }
        return true;
    }
    return false;
}

/// Step test: same norm comparison on the step vector
inline bool stopDx(const StopCriteria& criteria, const std::vector<double>& x,
                   const std::vector<double>& dx) {
    if (criteria.xtol_rel > 0.0) {
        const double step = quantape::math::detail::weightedNorm(dx, {}, criteria.x_weights);
        const double norm = quantape::math::detail::weightedNorm(x, {}, criteria.x_weights);
        if (step < criteria.xtol_rel * norm) {
            return true;
        }
    }
    if (criteria.xtol_abs > 0.0) {
        for (double d : dx) {
            if (std::fabs(d) >= criteria.xtol_abs) {
                return false;
            }
        }
        return true;
    }
    return false;
}

/// Gradient test: max-abs component below grad_tol
inline bool stopGrad(const StopCriteria& criteria, const std::vector<double>& grad) {
    if (criteria.grad_tol <= 0.0) {
        return false;
    }
    double max_abs = 0.0;
    for (double g : grad) {
        max_abs = std::max(max_abs, std::fabs(g));
    }
    return max_abs <= criteria.grad_tol;
}

/// Evaluation budget
inline bool stopEvals(const StopCriteria& criteria, std::size_t evals) {
    return criteria.maxeval > 0 && evals >= static_cast<std::size_t>(criteria.maxeval);
}

/// Time budget
inline bool stopTime(const StopCriteria& criteria, double start_time) {
    return criteria.maxtime > 0.0 && nowSeconds() - start_time >= criteria.maxtime;
}

// ============================================================================
// AD entry points (declared Stan-free; defined in OptimizerStanPrimitives.h)
// ============================================================================

namespace detail {
/**
 * Value + exact gradient of a scalar AD objective at a double point.
 * DoubleT = var: one reverse pass; DoubleT = fvar<...>: reverse over the
 * value part. Never instantiated for double.
 */
template <typename DoubleT, typename F>
std::pair<double, std::vector<double>> valueGrad(const F& f, const std::vector<double>& x);

/**
 * Buffer-reusing variant: writes into caller-provided `theta` (AD scratch)
 * and `grad`, returning the primal value. Avoids per-evaluation vector
 * allocation, which profiling showed dominates the optimizer hot path.
 */
template <typename DoubleT, typename F>
double valueGrad(const F& f, const std::vector<double>& x, std::vector<DoubleT>& theta,
                 std::vector<double>& grad);

/**
 * Exact Hessian-vector product H(x) v by forward-over-reverse (`fvar<var>`
 * tangent seeding, one pass). No finite differences, no step size. The
 * objective must be callable with `std::vector<fvar<var>>`.
 */
template <typename F>
std::vector<double> hvp(const F& f, const std::vector<double>& x, const std::vector<double>& v);

/**
 * Constraint values and row-major Jacobian at a double point. One evaluation,
 * one reverse pass per constraint row.
 */
template <typename DoubleT, typename G>
void constraintValueJacobian(const G& g, const std::vector<double>& x, std::vector<double>& c,
                             std::vector<double>& jacobian);
} // namespace detail

// ============================================================================
// Optimizer base
// ============================================================================

/**
 * @brief Base class for minimizers
 *
 * Concrete optimizers are declared as:
 *   class LBFGS : public `Optimizer<var, LBFGS<var>>` { ... }
 *
 * and must provide, publicly (the base calls it on the static type):
 *   template `<typename F>` requires VectorObjective<F, DoubleT>
 *   OptimizeResult minimizeImpl(const F& f, OptimizerState& state) const;
 *
 * The base holds configuration (StopCriteria) only; minimize() copies the
 * caller's start point into a local OptimizerState, dispatches to
 * Impl::minimizeImpl, and writes the final iterate back out. The same object
 * is therefore safe to share across threads and to call recursively.
 *
 * @tparam DoubleT Derivative backend: double (value-only), var (exact
 *                 gradients), fvar<...> (gradients and HVPs)
 * @tparam Impl Derived optimizer implementation
 */
template <typename DoubleT, typename Impl>
    requires OptimizationScalar<DoubleT>
class Optimizer {
public:
    using Scalar = DoubleT;

    explicit Optimizer(StopCriteria criteria = {}) : m_criteria(std::move(criteria)) {}

    /**
     * @brief Minimize @p f starting from @p x (in place)
     * @param f Objective, callable as DoubleT(const std::vector<DoubleT>&)
     * @param x In: start point; out: best point found
     * @param state Out: full run state (counters, message, final x)
     */
    template <typename F>
        requires ObjectiveEvaluator<F, DoubleT>
    OptimizeResult minimize(const F& f, std::vector<double>& x, OptimizerState& state) const {
        if (x.empty()) {
            throw std::invalid_argument("Optimizer: empty parameter vector");
        }
        state = OptimizerState{};
        state.x = x;
        state.start_time = nowSeconds();
        const OptimizeResult result = static_cast<const Impl*>(this)->minimizeImpl(f, state);
        x = state.x;
        return result;
    }

    /// Convenience overload: the run state is discarded
    template <typename F>
        requires ObjectiveEvaluator<F, DoubleT>
    OptimizeResult minimize(const F& f, std::vector<double>& x) const {
        OptimizerState state;
        return minimize(f, x, state);
    }

    // Modifiers / inspectors
    const StopCriteria& criteria() const { return m_criteria; }
    void setCriteria(const StopCriteria& criteria) { m_criteria = criteria; }

protected:
    // ── Stop helpers (bound to the stored criteria) ──

    bool stopByGradient(const std::vector<double>& grad) const {
        return stopGrad(m_criteria, grad);
    }

    bool stopByEvalOrTime(const OptimizerState& state) const {
        return stopEvals(m_criteria, state.evals) || stopTime(m_criteria, state.start_time);
    }

    /// Stop value or function tolerance (use the free functions for a
    /// specific result code)
    bool stopByValue(double f, double old_f) const {
        return f <= m_criteria.stopval || stopFtol(m_criteria, f, old_f);
    }

    // ── Objective evaluation (AD dispatch by DoubleT) ──

    /**
     * @brief Value + gradient at x
     *
     * double + ValueGradObjective: one callback call fills both.
     * double + scalar objective: value only, gradient empty.
     * AD scalars: exact gradient from one reverse pass
     * (OptimizerStanPrimitives.h).
     */
    template <typename F>
    std::pair<double, std::vector<double>> evalValueGrad(const F& f,
                                                         const std::vector<double>& x) const {
        if constexpr (std::is_same_v<DoubleT, double> && ValueGradObjective<F>) {
            std::vector<double> grad(x.size(), 0.0);
            const double value = f(x, grad);
            return {value, std::move(grad)};
        } else if constexpr (std::is_same_v<DoubleT, double>) {
            return {static_cast<double>(f(x)), std::vector<double>()};
        } else {
            return quantape::math::detail::valueGrad<DoubleT>(f, x);
        }
    }

    /// Evaluate and store value + gradient into the state, bumping counters
    template <typename F>
    void updateValueGrad(const F& f, OptimizerState& state) const {
        auto value_and_grad = evalValueGrad(f, state.x);
        state.f = value_and_grad.first;
        state.grad = std::move(value_and_grad.second);
        ++state.evals;
        ++state.grad_evals;
    }

    /**
     * @brief Value + gradient written into caller-provided buffers
     *
     * The allocation-free variant used by the algorithm loops: `theta_scratch`
     * caches the AD parameter vector across evaluations and `grad_out` is
     * reused. double + paired objective: the callback writes `grad_out`
     * directly; AD backends: one reverse pass (OptimizerStanPrimitives.h).
     */
    template <typename F>
    double evalValueGradInto(const F& f, const std::vector<double>& x,
                             std::vector<DoubleT>& theta_scratch,
                             std::vector<double>& grad_out) const {
        if constexpr (std::is_same_v<DoubleT, double> && ValueGradObjective<F>) {
            (void)theta_scratch;
            return f(x, grad_out);
        } else if constexpr (std::is_same_v<DoubleT, double>) {
            (void)theta_scratch;
            grad_out.clear();
            return static_cast<double>(f(x));
        } else {
            return quantape::math::detail::valueGrad<DoubleT>(f, x, theta_scratch, grad_out);
        }
    }

    /// Exact Hessian-vector product (AD scalars only)
    template <typename F>
    std::vector<double> evalHvp(const F& f, const std::vector<double>& x,
                                const std::vector<double>& v) const {
        static_assert(!std::is_same_v<DoubleT, double>,
                      "Optimizer: exact HVP requires an AD scalar (var / fvar<...>)");
        return quantape::math::detail::hvp(f, x, v);
    }

    /// Constraint values + Jacobian (AD scalars: reverse passes; double:
    /// value+Jacobian callback)
    template <typename G>
    void evalConstraint(const G& g, const std::vector<double>& x, std::vector<double>& c,
                        std::vector<double>& jacobian) const {
        if constexpr (std::is_same_v<DoubleT, double>) {
            static_assert(ValueJacobianConstraint<G>,
                          "double-backend constraints require void con(x, c, J)");
            g(x, c, jacobian);
        } else {
            quantape::math::detail::constraintValueJacobian<DoubleT>(g, x, c, jacobian);
        }
    }

private:
    StopCriteria m_criteria;
};

} // namespace quantape::math

#endif // OPTIMIZER_PRIMITIVES_H
