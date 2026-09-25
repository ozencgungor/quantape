#ifndef LINE_SEARCH_H
#define LINE_SEARCH_H

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <utility>
#include <vector>

#include "OptimizerPrimitives.h"

namespace quantape::math {
/**
 * @file LineSearch.h
 * @brief Strong-Wolfe line search for the optimizers (Stan-free)
 *
 * Nocedal & Wright, Algorithms 3.5 (bracket) and 3.6 (zoom), with a robust
 * bisection zoom. The constants default to NLopt's PS1L01 values
 * (`c1 = 1e-4` ~ `told`, `c2 = 0.9` ~ `tolp`) so the behaviour stays
 * comparable. All work is local to the call; no allocation beyond the
 * accepted point's gradient.
 */

/// Line-search parameters (defaults match NLopt's PS1L01 where comparable)
struct LineSearchOptions {
    double c1 = 1e-4; ///< Armijo sufficient-decrease constant
    double c2 = 0.9;  ///< Strong-Wolfe curvature constant
    double alpha_min = 1e-10;
    double alpha_max = 1e10;
    double alpha0 = 1.0; ///< first trial step
    int max_iterations = 20;
};

struct LineSearchResult {
    double alpha = 0.0;                            ///< accepted step
    double f = std::numeric_limits<double>::max(); ///< f(x + alpha d) at the accepted point
    std::vector<double> grad; ///< gradient at the accepted point (empty if none)
    bool success = false;     ///< strong-Wolfe conditions satisfied
};

namespace detail {
inline double dotProduct(const std::vector<double>& a, const std::vector<double>& b) {
    double sum = 0.0;
    for (std::size_t i = 0; i < a.size(); ++i) {
        sum += a[i] * b[i];
    }
    return sum;
}

inline double euclideanNorm(const std::vector<double>& v) {
    return std::sqrt(dotProduct(v, v));
}
} // namespace detail

/**
 * @brief Minimize phi(alpha) = f(x + alpha d) by a strong-Wolfe search
 *
 * @param eval Callable `double(const std::vector<double>& z,
 *             std::vector<double>& grad)` returning f(z) and writing its
 *             gradient into the provided buffer (the search reuses one
 *             scratch buffer, so evaluations allocate nothing)
 * @param x Current point
 * @param d Search direction (must be a descent direction, g0 . d < 0)
 * @param f0 Objective at x
 * @param g0 Gradient at x
 * @param opts Line-search parameters
 * @return The accepted step and its objective/gradient; `success == false`
 *         means the Wolfe conditions were not met, but `alpha`/`f` still
 *         describe the best decreasing point found (or alpha 0 for a
 *         non-descent direction; `grad` may be empty in that case).
 */
template <typename Eval>
LineSearchResult
wolfeLineSearch(const Eval& eval, const std::vector<double>& x, const std::vector<double>& d,
                double f0, const std::vector<double>& g0, const LineSearchOptions& opts = {}) {
    const std::size_t n = x.size();
    std::vector<double> x_trial(n);
    std::vector<double> g_trial(n);

    LineSearchResult best; // best decreasing point seen (fallback)

    const auto evaluate = [&](double alpha, double& dphi_out) {
        for (std::size_t i = 0; i < n; ++i) {
            x_trial[i] = x[i] + alpha * d[i];
        }
        const double f = eval(x_trial, g_trial);
        dphi_out = quantape::math::detail::dotProduct(g_trial, d);
        return f;
    };

    const double dphi0 = quantape::math::detail::dotProduct(g0, d);
    if (!(dphi0 < 0.0)) {
        // Not a descent direction: do not move.
        best.alpha = 0.0;
        best.f = f0;
        return best;
    }

    // Zoom on [lo, hi] with phi(lo) known to satisfy Armijo and phi(hi) known
    // to violate it (or fail curvature).
    const auto zoom = [&](double lo, double hi, double f_lo) {
        LineSearchResult local_best;
        if (f_lo < best.f) {
            best.f = f_lo;
            best.alpha = lo;
        }
        for (int i = 0; i < opts.max_iterations; ++i) {
            const double alpha = 0.5 * (lo + hi);
            if (alpha < opts.alpha_min) {
                break;
            }
            double dphi_j = 0.0;
            const double f_j = evaluate(alpha, dphi_j);
            if (f_j < best.f) {
                best.alpha = alpha;
                best.f = f_j;
            }
            if (f_j > f0 + opts.c1 * alpha * dphi0 || f_j >= f_lo) {
                hi = alpha;
            } else {
                if (std::fabs(dphi_j) <= -opts.c2 * dphi0) {
                    local_best.alpha = alpha;
                    local_best.f = f_j;
                    local_best.grad = std::move(g_trial);
                    local_best.success = true;
                    return local_best;
                }
                if (dphi_j * (hi - lo) >= 0.0) {
                    hi = lo;
                }
                lo = alpha;
                f_lo = f_j;
            }
            if (std::fabs(hi - lo) <= 1e-16 * std::max(1.0, std::fabs(lo))) {
                break;
            }
        }
        return local_best; // success == false: caller decides
    };

    // Bracket phase
    double alpha_prev = 0.0;
    double f_prev = f0;
    double alpha = std::clamp(opts.alpha0, opts.alpha_min, opts.alpha_max);

    for (int i = 1; i <= opts.max_iterations; ++i) {
        double dphi_i = 0.0;
        const double f_i = evaluate(alpha, dphi_i);
        if (f_i < best.f) {
            best.alpha = alpha;
            best.f = f_i;
        }

        if (f_i > f0 + opts.c1 * alpha * dphi0 || (i > 1 && f_i >= f_prev)) {
            auto zoomed = zoom(alpha_prev, alpha, f_prev);
            if (zoomed.success) {
                return zoomed;
            }
            return best;
        }
        if (std::fabs(dphi_i) <= -opts.c2 * dphi0) {
            LineSearchResult result;
            result.alpha = alpha;
            result.f = f_i;
            result.grad = std::move(g_trial);
            result.success = true;
            return result;
        }
        if (dphi_i >= 0.0) {
            auto zoomed = zoom(alpha, alpha_prev, f_i);
            if (zoomed.success) {
                return zoomed;
            }
            return best;
        }

        alpha_prev = alpha;
        f_prev = f_i;
        if (alpha >= opts.alpha_max) {
            break;
        }
        alpha = std::min(2.0 * alpha, opts.alpha_max);
    }

    return best; // no Wolfe point within the budget
}

} // namespace quantape::math

#endif // LINE_SEARCH_H
