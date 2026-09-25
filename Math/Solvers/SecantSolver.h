#ifndef SECANT_SOLVER_H
#define SECANT_SOLVER_H

#include <cmath>
#include <stdexcept>

#include "Solver1DBase.h"

namespace Math {
/**
 * @brief Secant method for 1D root finding
 *
 * The secant method approximates the derivative using finite differences:
 *   x_{n+1} = x_n - f(x_n) * (x_n - x_{n-1}) / (f(x_n) - f(x_{n-1}))
 *
 * Superlinear convergence (~1.618) without requiring derivative computation.
 * Faster than bisection but not as robust (may fail to converge).
 *
 * @tparam DoubleT Numeric type (double or stan::math::var)
 */
template <typename DoubleT>
class SecantSolver : public Solver1D<DoubleT, SecantSolver<DoubleT>> {
public:
    using Base = Solver1D<DoubleT, SecantSolver<DoubleT>>;

    SecantSolver() = default;

    template <typename F>
    DoubleT solveImpl(const F& f, double accuracy, SolverState<DoubleT>& s) const {
        DoubleT fl, froot, dx, xl;

        // Pick the bound with smaller function value as most recent guess
        if (std::fabs(Base::value(s.fxMin)) < std::fabs(Base::value(s.fxMax))) {
            s.root = s.xMin;
            froot = s.fxMin;
            xl = s.xMax;
            fl = s.fxMax;
        } else {
            s.root = s.xMax;
            froot = s.fxMax;
            xl = s.xMin;
            fl = s.fxMin;
        }

        while (s.evaluations <= s.maxEvaluations) {
            // Secant update formula
            const DoubleT denominator = froot - fl;
            if (Base::value(denominator) == 0.0) {
                throw std::runtime_error("SecantSolver: zero denominator");
            }

            dx = (xl - s.root) * froot / denominator;
            xl = s.root;
            fl = froot;
            s.root = s.root + dx;
            froot = f(s.root);
            ++s.evaluations;

            if (std::fabs(Base::value(dx)) < accuracy || Base::isZero(froot, s.fScale)) {
                return s.root;
            }
        }

        throw std::runtime_error("SecantSolver: maximum number of evaluations exceeded");
    }
};
} // namespace Math

#endif // SECANT_SOLVER_H
