#ifndef NEWTON_SOLVER_H
#define NEWTON_SOLVER_H

#include <algorithm>
#include <cmath>
#include <concepts>
#include <stdexcept>
#include <utility>

#include "Solver1DBase.h"

namespace quantape::math {
namespace detail {
/**
 * @brief Whether an objective provides its own derivative
 *
 * A functor with a derivative(x) member is used directly; everything else
 * falls back to central finite differences (see newtonDerivative).
 */
template <typename F, typename T>
concept HasDerivative = requires(const F& f, const T& x) {
    { f.derivative(x) } -> std::convertible_to<T>;
};

/**
 * @brief Derivative of the objective at x
 *
 * Honest fallback semantics: without a derivative() member this is central
 * finite differences, second-order in value but with the usual round-off
 * floor, and it costs two extra function evaluations. It is well-defined for
 * AD scalars too, but for stan::math::var prefer NewtonSolverWithDerivative
 * (or give the objective a derivative() method) to keep the AD graph small
 * and the derivative exact.
 */
template <typename DoubleT, typename F>
DoubleT newtonDerivative(const F& f, const DoubleT& x) {
    if constexpr (HasDerivative<F, DoubleT>) {
        return f.derivative(x);
    } else {
        const double h = 1e-8 * std::max(1.0, std::fabs(primalValue(x)));
        return (f(x + DoubleT(h)) - f(x - DoubleT(h))) / DoubleT(2.0 * h);
    }
}

/**
 * @brief Shared Newton iteration for the solvers below
 *
 * Falls back to a bisection step whenever the Newton step leaves the bracket.
 */
template <typename DoubleT, typename F, typename DerivativeFn>
DoubleT newtonIterate(const F& f, const DerivativeFn& df, double accuracy,
                      SolverState<DoubleT>& s) {
    DoubleT froot = f(s.root);
    DoubleT dfroot = df(s.root);
    ++s.evaluations;

    if (isZero(primalValue(dfroot), 1.0)) {
        throw std::runtime_error("Newton solver: derivative is zero");
    }

    while (s.evaluations <= s.maxEvaluations) {
        const DoubleT dx = froot / dfroot;
        s.root = s.root - dx;

        // Check if jumped out of brackets
        if (primalValue(s.root) < primalValue(s.xMin) ||
            primalValue(s.root) > primalValue(s.xMax)) {
            // Outside brackets - fall back to bisection for this step
            s.root = (s.xMin + s.xMax) / DoubleT(2.0);
        }

        if (std::fabs(primalValue(dx)) < accuracy) {
            return s.root;
        }

        froot = f(s.root);
        dfroot = df(s.root);
        ++s.evaluations;

        if (isZero(primalValue(dfroot), 1.0)) {
            throw std::runtime_error("Newton solver: derivative became zero");
        }
    }

    throw std::runtime_error("Newton solver: maximum number of evaluations exceeded");
}
} // namespace detail

/**
 * @brief Newton-Raphson method for 1D root finding
 *
 * Classic Newton's method using the update formula:
 *   x_{n+1} = x_n - f(x_n) / f'(x_n)
 *
 * Quadratic convergence when near the root. The derivative comes from the
 * objective's derivative(x) method when present, otherwise central finite
 * differences. For AD objectives, NewtonSolverWithDerivative avoids the
 * finite-difference cost and round-off.
 *
 * Falls back to bisection if a Newton step would jump outside the brackets.
 *
 * @tparam DoubleT Numeric type (double or stan::math::var)
 */
template <typename DoubleT>
class NewtonSolver : public Solver1D<DoubleT, NewtonSolver<DoubleT>> {
public:
    using Base = Solver1D<DoubleT, NewtonSolver<DoubleT>>;

    NewtonSolver() = default;

    template <typename F>
    DoubleT solveImpl(const F& f, double accuracy, SolverState<DoubleT>& s) const {
        const auto derivativeFn = [&f](const DoubleT& x) {
            return quantape::math::detail::newtonDerivative(f, x);
        };
        return quantape::math::detail::newtonIterate(f, derivativeFn, accuracy, s);
    }
};

/**
 * @brief Newton solver with a user-supplied derivative
 *
 * No type erasure: the derivative functor is a template parameter and is
 * stored by value, so the update loop inlines both calls. More efficient
 * than finite differences and exact for AD objectives.
 *
 * @tparam DoubleT Numeric type (double or stan::math::var)
 * @tparam Derivative Callable DoubleT -> DoubleT
 */
template <typename DoubleT, typename Derivative>
    requires std::invocable<const Derivative&, const DoubleT&> &&
             std::convertible_to<std::invoke_result_t<const Derivative&, const DoubleT&>, DoubleT>
class NewtonSolverWithDerivative
    : public Solver1D<DoubleT, NewtonSolverWithDerivative<DoubleT, Derivative>> {
public:
    using Base = Solver1D<DoubleT, NewtonSolverWithDerivative<DoubleT, Derivative>>;

    explicit NewtonSolverWithDerivative(Derivative derivative)
        : m_derivative(std::move(derivative)) {}

    template <typename F>
    DoubleT solveImpl(const F& f, double accuracy, SolverState<DoubleT>& s) const {
        return quantape::math::detail::newtonIterate(f, m_derivative, accuracy, s);
    }

private:
    Derivative m_derivative;
};
} // namespace quantape::math

#endif // NEWTON_SOLVER_H
