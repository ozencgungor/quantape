#ifndef CONSTRAINT_H
#define CONSTRAINT_H

#include <algorithm>
#include <cmath>
#include <concepts>
#include <cstddef>
#include <limits>
#include <type_traits>
#include <vector>

#include "OptimizerPrimitives.h"

namespace Math {
/**
 * @file Constraint.h
 * @brief Constraints and bounds for the optimizers (Stan-free)
 *
 * Convention: inequality constraints are g(x) <= 0, equalities h(x) = 0.
 * For the double backend a callback supplies values and the row-major
 * Jacobian in one call; for AD backends the scalar-generic writer shape is
 * used and the Jacobian is produced by OptimizerStanPrimitives.h.
 */

/// Parameter box bounds; kNoLower/kNoUpper mark unbounded sides.
struct Bounds {
    static constexpr double kNoLower = -std::numeric_limits<double>::max();
    static constexpr double kNoUpper = std::numeric_limits<double>::max();

    std::vector<double> lower;
    std::vector<double> upper;

    static Bounds unbounded(std::size_t n) {
        return Bounds{std::vector<double>(n, kNoLower), std::vector<double>(n, kNoUpper)};
    }

    /// Build from user vectors; ±infinity maps to the unbounded sentinels
    static Bounds fromVectors(const std::vector<double>& lo, const std::vector<double>& hi) {
        Bounds bounds;
        bounds.lower.resize(lo.size());
        bounds.upper.resize(hi.size());
        for (std::size_t i = 0; i < lo.size(); ++i) {
            bounds.lower[i] = std::isfinite(lo[i]) ? lo[i] : kNoLower;
        }
        for (std::size_t i = 0; i < hi.size(); ++i) {
            bounds.upper[i] = std::isfinite(hi[i]) ? hi[i] : kNoUpper;
        }
        return bounds;
    }

    bool empty() const { return upper.empty(); }
    std::size_t size() const { return upper.size(); }
    bool hasLower(std::size_t i) const { return lower[i] > kNoLower; }
    bool hasUpper(std::size_t i) const { return upper[i] < kNoUpper; }

    bool feasible(const std::vector<double>& x) const {
        for (std::size_t i = 0; i < x.size(); ++i) {
            if (hasLower(i) && x[i] < lower[i])
                return false;
            if (hasUpper(i) && x[i] > upper[i])
                return false;
        }
        return true;
    }

    /// Maximum box violation (0 if feasible)
    double violation(const std::vector<double>& x) const {
        double worst = 0.0;
        for (std::size_t i = 0; i < x.size(); ++i) {
            if (hasLower(i))
                worst = std::max(worst, lower[i] - x[i]);
            if (hasUpper(i))
                worst = std::max(worst, x[i] - upper[i]);
        }
        return worst;
    }

    /// Clip x into the box (no-op for free coordinates)
    void project(std::vector<double>& x) const {
        for (std::size_t i = 0; i < x.size(); ++i) {
            if (hasLower(i))
                x[i] = std::max(x[i], lower[i]);
            if (hasUpper(i))
                x[i] = std::min(x[i], upper[i]);
        }
    }
};

/// No-op constraint used by the unconstrained entry points
struct NoConstraint {
    template <typename S>
    void operator()(const std::vector<S>&, std::vector<S>& out) const {
        out.clear();
    }
};

/// Entry-point constraint constraint for the optimizers
template <typename F, typename S>
concept ConstraintEvaluator =
    VectorConstraint<F, S> || (std::is_same_v<S, double> && ValueJacobianConstraint<F>);

/// Maximum inequality violation max_i max(0, g_i)
inline double maxViolation(const std::vector<double>& g) {
    double worst = 0.0;
    for (double gi : g) {
        worst = std::max(worst, gi);
    }
    return worst;
}

/// Sum of equality residuals sum_k |h_k|
inline double equalityResidual(const std::vector<double>& h) {
    double sum = 0.0;
    for (double hi : h) {
        sum += std::fabs(hi);
    }
    return sum;
}

/// L1 exact-penalty value: sum max(0, g_i) + sum |h_k|
inline double l1Penalty(const std::vector<double>& g, const std::vector<double>& h) {
    double penalty = 0.0;
    for (double gi : g) {
        penalty += std::max(0.0, gi);
    }
    for (double hi : h) {
        penalty += std::fabs(hi);
    }
    return penalty;
}

} // namespace Math

#endif // CONSTRAINT_H
