#ifndef SOLVER1D_BASE_H
#define SOLVER1D_BASE_H

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <type_traits>

#include "SolverPrimitives.h"

namespace quantape::math {
/**
 * @brief Base class for 1-D root finders using CRTP
 *
 * Concrete solvers are declared as:
 *   class BrentSolver : public Solver1D<double, BrentSolver> { ... }
 *
 * Design based on QuantLib's Solver1D, with two deliberate differences:
 * - the objective is a template parameter instead of std::function, so the
 *   evaluation loop inlines and no type-erased call / heap allocation is paid;
 * - solve() keeps all working state in a SolverState local to the call rather
 *   than in mutable members, so a solver is safe to share across threads and
 *   to call recursively from inside another objective.
 *
 * AD dispatch is automatic by DoubleT, like the Integrals classes: include
 * Solvers/SolverStanPrimitives.h (plus a Stan Math header) and
 * Solver<stan::math::var>::solve returns exact implicit-function-theorem
 * sensitivities, Solver<fvar<...>> keeps the pathwise route. See that header.
 *
 * @tparam DoubleT Numeric type (double or an AD scalar exposing val())
 * @tparam Impl Derived solver implementation (CRTP)
 */
template <typename DoubleT, typename Impl>
    requires SolverScalar<DoubleT>
class Solver1D {
public:
    Solver1D() = default;

    /**
     * @brief Solve for a root inside the bracket [xMin, xMax]
     * @param f Objective, callable as DoubleT(DoubleT)
     * @param accuracy Target accuracy (bracket width / step size)
     * @param guess Initial guess, strictly inside the bracket
     * @param xMin Lower bracket, f(xMin) and f(xMax) must straddle zero
     * @param xMax Upper bracket
     * @return Root value
     */
    template <typename F>
        requires SolverFunction<F, DoubleT>
    DoubleT solve(const F& f, double accuracy, DoubleT guess, DoubleT xMin, DoubleT xMax) const {
        if constexpr (std::is_same_v<DoubleT, double>) {
            return solveGeneric(f, accuracy, guess, xMin, xMax);
        } else {
            return quantape::math::detail::solveWithSensitivity(static_cast<const Impl&>(*this), f,
                                                                accuracy, guess, xMin, xMax);
        }
    }

    /**
     * @brief Solve for a root with automatic bracketing
     * @param f Objective, callable as DoubleT(DoubleT)
     * @param accuracy Target accuracy
     * @param guess Initial guess
     * @param step Initial step size for bracketing
     * @return Root value
     */
    template <typename F>
        requires SolverFunction<F, DoubleT>
    DoubleT solve(const F& f, double accuracy, DoubleT guess, DoubleT step) const {
        if constexpr (std::is_same_v<DoubleT, double>) {
            return solveGeneric(f, accuracy, guess, step);
        } else {
            return quantape::math::detail::solveWithAutoBracketSensitivity(
                static_cast<const Impl&>(*this), f, accuracy, guess, step);
        }
    }

    /**
     * @brief Value path of solve() (internal, public so the AD primitives in
     *        SolverStanPrimitives.h can reuse it for the pathwise route).
     *
     * Solves for a root inside the bracket [xMin, xMax].
     */
    template <typename F>
        requires SolverFunction<F, DoubleT>
    DoubleT solveGeneric(const F& f, double accuracy, DoubleT guess, DoubleT xMin,
                         DoubleT xMax) const {
        if (accuracy <= 0.0) {
            throw std::invalid_argument("accuracy must be positive");
        }
        // Use at least machine epsilon
        accuracy = std::max(accuracy, std::numeric_limits<double>::epsilon());

        if (quantape::math::detail::primalValue(xMin) >=
            quantape::math::detail::primalValue(xMax)) {
            throw std::invalid_argument("invalid range: xMin >= xMax");
        }
        if (m_lowerBoundEnforced && quantape::math::detail::primalValue(xMin) < m_lowerBound) {
            throw std::invalid_argument("xMin < enforced lower bound");
        }
        if (m_upperBoundEnforced && quantape::math::detail::primalValue(xMax) > m_upperBound) {
            throw std::invalid_argument("xMax > enforced upper bound");
        }

        SolverState<DoubleT> s;
        s.xMin = xMin;
        s.xMax = xMax;
        s.maxEvaluations = m_maxEvaluations;

        s.fxMin = f(s.xMin);
        if (isZero(s.fxMin, s.fScale)) {
            return s.xMin;
        }
        s.fxMax = f(s.xMax);
        if (isZero(s.fxMax, s.fScale)) {
            return s.xMax;
        }
        s.evaluations = 2;
        s.fScale = std::max(1.0, std::max(std::fabs(quantape::math::detail::primalValue(s.fxMin)),
                                          std::fabs(quantape::math::detail::primalValue(s.fxMax))));

        if (!oppositeSigns(s.fxMin, s.fxMax)) {
            throw std::runtime_error("root not bracketed");
        }
        if (quantape::math::detail::primalValue(guess) <=
                quantape::math::detail::primalValue(s.xMin) ||
            quantape::math::detail::primalValue(guess) >=
                quantape::math::detail::primalValue(s.xMax)) {
            throw std::invalid_argument("guess must be strictly between xMin and xMax");
        }
        s.root = guess;

        // Call derived class implementation
        return static_cast<const Impl*>(this)->solveImpl(f, accuracy, s);
    }

    /**
     * @brief Value path of solve() with automatic bracketing (internal).
     */
    template <typename F>
        requires SolverFunction<F, DoubleT>
    DoubleT solveGeneric(const F& f, double accuracy, DoubleT guess, DoubleT step) const {
        if (accuracy <= 0.0) {
            throw std::invalid_argument("accuracy must be positive");
        }
        accuracy = std::max(accuracy, std::numeric_limits<double>::epsilon());

        SolverState<DoubleT> s;
        s.maxEvaluations = m_maxEvaluations;

        const double growthFactor = 1.6;
        int flipflop = -1;

        s.root = guess;
        s.fxMax = f(s.root);
        if (isZero(s.fxMax, s.fScale)) {
            return s.root;
        } else if (quantape::math::detail::primalValue(s.fxMax) > 0.0) {
            s.xMin = enforceBounds(s.root - step);
            s.fxMin = f(s.xMin);
            s.xMax = s.root;
        } else {
            s.xMin = s.root;
            s.fxMin = s.fxMax;
            s.xMax = enforceBounds(s.root + step);
            s.fxMax = f(s.xMax);
        }
        s.evaluations = 2;

        while (s.evaluations <= s.maxEvaluations) {
            s.fScale =
                std::max(1.0, std::max(std::fabs(quantape::math::detail::primalValue(s.fxMin)),
                                       std::fabs(quantape::math::detail::primalValue(s.fxMax))));
            if (isZero(s.fxMin, s.fScale)) {
                return s.xMin;
            }
            if (isZero(s.fxMax, s.fScale)) {
                return s.xMax;
            }
            if (oppositeSigns(s.fxMin, s.fxMax)) {
                s.root = (s.xMax + s.xMin) / DoubleT(2.0);
                return static_cast<const Impl*>(this)->solveImpl(f, accuracy, s);
            }

            if (std::fabs(quantape::math::detail::primalValue(s.fxMin)) <
                std::fabs(quantape::math::detail::primalValue(s.fxMax))) {
                s.xMin = enforceBounds(s.xMin + DoubleT(growthFactor) * (s.xMin - s.xMax));
                s.fxMin = f(s.xMin);
            } else if (std::fabs(quantape::math::detail::primalValue(s.fxMin)) >
                       std::fabs(quantape::math::detail::primalValue(s.fxMax))) {
                s.xMax = enforceBounds(s.xMax + DoubleT(growthFactor) * (s.xMax - s.xMin));
                s.fxMax = f(s.xMax);
            } else if (flipflop == -1) {
                s.xMin = enforceBounds(s.xMin + DoubleT(growthFactor) * (s.xMin - s.xMax));
                s.fxMin = f(s.xMin);
                flipflop = 1;
            } else {
                s.xMax = enforceBounds(s.xMax + DoubleT(growthFactor) * (s.xMax - s.xMin));
                s.fxMax = f(s.xMax);
                flipflop = -1;
            }
            ++s.evaluations;
        }

        throw std::runtime_error("unable to bracket root in max function evaluations");
    }

    // Modifiers
    void setMaxEvaluations(std::size_t evaluations) { m_maxEvaluations = evaluations; }

    /// Mirror a configuration onto this solver (used by the AD path to copy
    /// the caller's settings onto the double-precision twin).
    void setConfig(const SolverConfig& config) {
        m_maxEvaluations = config.maxEvaluations;
        m_lowerBound = config.lowerBound;
        m_lowerBoundEnforced = config.lowerBoundEnforced;
        m_upperBound = config.upperBound;
        m_upperBoundEnforced = config.upperBoundEnforced;
    }

    void setLowerBound(double lowerBound) {
        m_lowerBound = lowerBound;
        m_lowerBoundEnforced = true;
    }

    void setUpperBound(double upperBound) {
        m_upperBound = upperBound;
        m_upperBoundEnforced = true;
    }

    // Inspectors
    std::size_t maxEvaluations() const { return m_maxEvaluations; }

    SolverConfig config() const {
        return SolverConfig{m_maxEvaluations, m_lowerBoundEnforced, m_lowerBound,
                            m_upperBoundEnforced, m_upperBound};
    }

protected:
    /// Primal value of a scalar (recursive for fvar<var>)
    static double value(const DoubleT& x) { return quantape::math::detail::primalValue(x); }

    static bool isZero(const DoubleT& x, double scale) {
        return quantape::math::detail::isZero(value(x), scale);
    }

    static bool close(const DoubleT& x, const DoubleT& y) {
        return quantape::math::detail::close(value(x), value(y));
    }

    static bool oppositeSigns(const DoubleT& a, const DoubleT& b) {
        return quantape::math::detail::oppositeSigns(value(a), value(b));
    }

private:
    std::size_t m_maxEvaluations = 100;
    double m_lowerBound = 0.0;
    double m_upperBound = 0.0;
    bool m_lowerBoundEnforced = false;
    bool m_upperBoundEnforced = false;

    /// Clamp an auto-bracket endpoint to the enforced bounds.
    ///
    /// Note for AD scalars: a clamped endpoint is a constant node, so if the
    /// root lands exactly on an enforced bound its derivative w.r.t. the
    /// caller's inputs is lost. Set bounds beyond the reachable root when the
    /// derivative matters.
    DoubleT enforceBounds(DoubleT x) const {
        if (m_lowerBoundEnforced && quantape::math::detail::primalValue(x) < m_lowerBound) {
            return DoubleT(m_lowerBound);
        }
        if (m_upperBoundEnforced && quantape::math::detail::primalValue(x) > m_upperBound) {
            return DoubleT(m_upperBound);
        }
        return x;
    }
};
} // namespace quantape::math

#endif // SOLVER1D_BASE_H
