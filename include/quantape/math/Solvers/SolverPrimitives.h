#ifndef SOLVER_PRIMITIVES_H
#define SOLVER_PRIMITIVES_H

#include "quantape/math/Autodiff/PrimalExtraction.h"

#include <algorithm>
#include <cmath>
#include <concepts>
#include <cstddef>
#include <limits>
#include <type_traits>
#include <utility>

namespace quantape::math {
/**
 * @file SolverPrimitives.h
 * @brief Shared building blocks for the 1-D solvers
 *
 * Kept free of Stan Math includes (same convention as NumericalMethods.h):
 * AD scalars are recognised structurally through val(), so the solvers can be
 * instantiated with double, stan::math::var or stan::math::fvar<stan::math::var>
 * without pulling in the Stan / TBB include paths until they are actually used.
 */

namespace detail {
/// Recursive detection of a val() chain ending in an arithmetic type
/// (double <- var <- `fvar<var>` <- ...). A trait is used instead of a
/// self-referential concept, which C++20 forbids.
template <typename T>
struct is_solver_scalar : std::is_arithmetic<T> {};

template <typename T>
    requires requires(const T& x) { x.val(); }
struct is_solver_scalar<T>
    : is_solver_scalar<std::remove_cvref_t<decltype(std::declval<const T&>().val())>> {};
} // namespace detail

/**
 * @brief Scalar accepted by the solvers: an arithmetic type, or an AD scalar
 *        whose val() chain terminates in an arithmetic type
 *        (stan::math::var, stan::math::fvar<stan::math::var>, ...).
 */
template <typename T>
concept SolverScalar = quantape::math::detail::is_solver_scalar<T>::value;

/**
 * @brief Unary callable DoubleT -> DoubleT accepted as a solver objective.
 *
 * The exact return type is required so that, e.g., a double-returning lambda
 * is rejected by the var instantiation instead of silently slicing the AD
 * graph.
 */
template <typename F, typename T>
concept SolverFunction =
    std::invocable<const F&, T> && std::same_as<std::invoke_result_t<const F&, T>, T>;

/**
 * @brief Working state of one solve() call
 *
 * All of it is local to solve(), so a solver object holds configuration only:
 * it can be shared between threads, and a solve can be nested inside another
 * solve's objective.
 *
 * @tparam DoubleT Numeric type (double or an AD scalar)
 */
template <typename DoubleT>
struct SolverState {
    DoubleT root{};              ///< Current iterate
    DoubleT xMin{};              ///< Lower bracket
    DoubleT xMax{};              ///< Upper bracket
    DoubleT fxMin{};             ///< Objective at xMin
    DoubleT fxMax{};             ///< Objective at xMax
    std::size_t evaluations = 0; ///< Function evaluations so far
    std::size_t maxEvaluations = 100;
    double fScale = 1.0; ///< Typical |f| over the bracket, for |f| ~ 0 tests
};

/**
 * @brief Configuration snapshot of a solver
 *
 * Used by the AD path to mirror the caller's settings (iteration budget and
 * enforced brackets) onto the double-precision twin of the solver.
 */
struct SolverConfig {
    std::size_t maxEvaluations = 100;
    bool lowerBoundEnforced = false;
    double lowerBound = 0.0;
    bool upperBoundEnforced = false;
    double upperBound = 0.0;
};

/**
 * @brief Rebinds a solver to a different scalar type
 *
 * All 1-D solvers take the scalar as their single template parameter, so
 *   SolverRebindT<double, BrentSolver<var>> == BrentSolver<double>.
 * Solvers with extra template parameters (NewtonSolverWithDerivative) have no
 * mapping and stay on the pathwise AD route.
 */
template <typename NewDoubleT, typename SolverT>
struct SolverRebind;

template <typename NewDoubleT, template <typename> class Solver, typename OldDoubleT>
struct SolverRebind<NewDoubleT, Solver<OldDoubleT>> {
    using type = Solver<NewDoubleT>;
};

template <typename NewDoubleT, typename SolverT>
using SolverRebindT = typename SolverRebind<NewDoubleT, SolverT>::type;

template <typename SolverT>
concept RebindableToDouble = requires { typename SolverRebind<double, SolverT>::type; };

namespace detail {
/// Whether a and b lie on strictly opposite sides of zero (overflow-free;
/// -0.0 counts as non-negative so a root exactly at a bound stays bracketed).
inline bool oppositeSigns(double a, double b) {
    return (a < 0.0) != (b < 0.0);
}

/// Relative proximity with an absolute floor: degrades to the classic 42*eps
/// test near zero and stays meaningful for roots / function values far from 1.
inline bool close(double x, double y) {
    const double scale = std::max({1.0, std::fabs(x), std::fabs(y)});
    return std::fabs(x - y) <= 42.0 * std::numeric_limits<double>::epsilon() * scale;
}

/// |x| ~ 0 against a reference magnitude (typically SolverState::fScale).
inline bool isZero(double x, double scale) {
    return std::fabs(x) <= 42.0 * std::numeric_limits<double>::epsilon() * std::max(1.0, scale);
}

/**
 * @brief AD entry points of the 1-D solvers (declared Stan-free)
 *
 * solve() routes here whenever DoubleT is not double. The definitions live in
 * SolverStanPrimitives.h, which must be included alongside the solvers for AD
 * scalars (the integrators follow the same split in
 * Integrals/IntegratorStanPrimitives.h). Never instantiated for double, so the
 * umbrella stays Stan-free.
 */
template <typename DoubleT, typename SolverT, typename F>
DoubleT solveWithSensitivity(const SolverT& solver, const F& f, double accuracy, DoubleT guess,
                             DoubleT xMin, DoubleT xMax);

template <typename DoubleT, typename SolverT, typename F>
DoubleT solveWithAutoBracketSensitivity(const SolverT& solver, const F& f, double accuracy,
                                        DoubleT guess, DoubleT step);
} // namespace detail
} // namespace quantape::math

#endif // SOLVER_PRIMITIVES_H
