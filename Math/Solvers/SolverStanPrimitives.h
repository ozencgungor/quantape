#ifndef SOLVER_STAN_PRIMITIVES_H
#define SOLVER_STAN_PRIMITIVES_H

//
// SolverStanPrimitives.h -- AD dispatch for the 1-D solvers
//
// Include this header together with a Stan Math header and use the solvers as
// usual: the DoubleT template parameter selects the AD path automatically,
// exactly like Integrals/IntegratorStanPrimitives.h does for the integrators.
//
//   Math::BrentSolver<stan::math::var>  solver;   // exact IFT gradient
//   solver.solve(f, 1e-12, guess, xMin, xMax);    // caller sees no difference
//
// DoubleT = stan::math::var
// -------------------------
// The iteration is NOT differentiated. Instead:
//   1. the same solver (same algorithm, same bracketing, same iteration budget
//      and enforced bounds) solves the value problem at double precision, with
//      the objective evaluated at DoubleT nodes whose primal is extracted;
//   2. one isolated reverse pass at the double root gives f_x = df/dx;
//   3. the result is the single Newton polish
//          result = x - f(x) / f_x        (x an independent var at the root)
//      whose sensitivity is the implicit-function-theorem value
//          dx0/dtheta = - f_theta / f_x
//      exactly, independent of solver, tolerance and iteration count. This
//      includes bisection, whose pathwise derivative is identically zero.
// The internal reverse pass runs in a nested scope and its raw f_theta
// adjoints are cleared before returning, so the caller's adjoints and tape are
// untouched (set_zero_all_adjoints() during forward construction).
//
// DoubleT = stan::math::fvar<...> (and higher fvar nesting)
// --------------------------------------------------------
// The generic pathwise route is kept: the iterates carry the AD graph, so
// derivatives of every order converge with the iteration for smooth solvers
// (Newton, secant, false position, Ridder, Brent). NewtonSolverWithDerivative
// has no double-precision twin to rebind (extra template parameter) and also
// stays pathwise. Bisection's pathwise derivatives are zero at every order:
// use one of the smooth solvers when differentiating through a root, or use
// var (which never differentiates the iteration).
//
// Notes
// -----
// - The objective only needs to be callable as DoubleT -> DoubleT (also when
//   the arguments are AD scalars); a generic lambda or a var-typed functor both
//   work. The double value pass is synthesised, so no second overload of f is
//   required. Scalar-generic objectives (callable with double) are evaluated
//   directly in the value pass — no tape per evaluation; AD-only objectives
//   fall back to DoubleT nodes in a recovered nested scope.
// - The IFT derivative is with respect to parameters entering f only.
//   AD-dependent bracket endpoints / guesses are used by value.
//
// This header is Stan-only: NumericalMethods.h and Solvers/SolverPrimitives.h
// stay Stan-free, and nothing here is instantiated for double.
//

#include "Math/StanMath.h"

#include <cmath>
#include <stdexcept>

#include "SolverPrimitives.h"

namespace Math {
namespace detail {

/**
 * @brief Build the IFT-polished root at a double-precision value
 *
 * x is an independent var at the root; the returned expression has value
 * x - f(x)/f_x and sensitivity -f_theta/f_x.
 *
 * f is evaluated ONCE: the f_x pass re-uses the same tape as y (no fresh
 * evaluation, no nested tape), then the raw adjoints the pass pushed into
 * the caller's parameters (and f's internal nodes) are cleared so the
 * caller's own reverse pass starts clean.
 */
template <typename F>
stan::math::var implicitRoot(const F& f, double x0) {
    stan::math::var x(x0);
    stan::math::var y = f(x);

    // f_x from the same tape (isolated pass, no fresh evaluation)
    double fPrime = 0.0;
    y.grad();
    fPrime = x.adj();

    // The pass above also pushed the raw f_theta into the outer parameters
    // (reverse-mode chain rules write operand adjoints directly). Clear them
    // so the caller's own reverse pass starts clean; the forward graph of y
    // is untouched.
    stan::math::set_zero_all_adjoints();

    if (!std::isfinite(fPrime) || fPrime == 0.0) {
        throw std::runtime_error("SolverStanPrimitives: f_x = 0 at the root");
    }

    // One Newton polish: value ~ x0, derivative exactly -f_theta / f_x
    return x - y / fPrime;
}

/**
 * @brief Double-precision value solve for the var path
 *
 * Objectives the double twin accepts DIRECTLY (scalar-generic lambdas whose
 * result is double-convertible) are evaluated with zero tape cost. Anything
 * else (AD-only functors, or lambdas that return an AD type even for double
 * arguments) falls back to DoubleT nodes in a nested scope, which is
 * recovered at scope exit — the same contract as before.
 */
template <typename SolverT, typename F>
double solveValue(const SolverT& solver, const F& f, double accuracy, double guess, double xMin,
                  double xMax) {
    if constexpr (requires { solver.solve(f, accuracy, guess, xMin, xMax); }) {
        return solver.solve(f, accuracy, guess, xMin, xMax);
    } else {
        stan::math::nested_rev_autodiff nested;
        const auto fValue = [&](double x) { return primalValue(f(stan::math::var(x))); };
        return solver.solve(fValue, accuracy, guess, xMin, xMax);
    }
}

/**
 * @brief Double-precision value solve for the var path (auto-bracketing)
 */
template <typename SolverT, typename F>
double solveValue(const SolverT& solver, const F& f, double accuracy, double guess, double step) {
    if constexpr (requires { solver.solve(f, accuracy, guess, step); }) {
        return solver.solve(f, accuracy, guess, step);
    } else {
        stan::math::nested_rev_autodiff nested;
        const auto fValue = [&](double x) { return primalValue(f(stan::math::var(x))); };
        return solver.solve(fValue, accuracy, guess, step);
    }
}

/**
 * @brief var path: double-precision solve + implicit-function-theorem polish
 */
template <typename DoubleT, typename SolverT, typename F>
DoubleT solveWithSensitivity(const SolverT& solver, const F& f, double accuracy, DoubleT guess,
                             DoubleT xMin, DoubleT xMax) {
    if constexpr (stan::is_var<DoubleT>::value && RebindableToDouble<SolverT>) {
        SolverRebindT<double, SolverT> doubleSolver;
        doubleSolver.setConfig(solver.config());

        const double x0 = solveValue(doubleSolver, f, accuracy, primalValue(guess),
                                     primalValue(xMin), primalValue(xMax));

        return implicitRoot(f, x0);
    } else {
        // fvar<...> (and solvers without a double twin): pathwise
        return solver.solveGeneric(f, accuracy, guess, xMin, xMax);
    }
}

/**
 * @brief var path for automatic bracketing: value solve + IFT polish
 */
template <typename DoubleT, typename SolverT, typename F>
DoubleT solveWithAutoBracketSensitivity(const SolverT& solver, const F& f, double accuracy,
                                        DoubleT guess, DoubleT step) {
    if constexpr (stan::is_var<DoubleT>::value && RebindableToDouble<SolverT>) {
        SolverRebindT<double, SolverT> doubleSolver;
        doubleSolver.setConfig(solver.config());

        const double x0 =
            solveValue(doubleSolver, f, accuracy, primalValue(guess), primalValue(step));

        return implicitRoot(f, x0);
    } else {
        return solver.solveGeneric(f, accuracy, guess, step);
    }
}

} // namespace detail
} // namespace Math

#endif // SOLVER_STAN_PRIMITIVES_H
