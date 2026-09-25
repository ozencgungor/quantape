#ifndef FALSE_POSITION_SOLVER_H
#define FALSE_POSITION_SOLVER_H

#include <cmath>
#include <stdexcept>

#include "Solver1DBase.h"

namespace Math {
/**
 * @brief False Position (Regula Falsi) method for 1D root finding
 *
 * The false position method is a bracketing method similar to bisection,
 * but uses linear interpolation instead of bisecting the interval:
 *
 *   x_new = x_max - f(x_max) * (x_max - x_min) / (f(x_max) - f(x_min))
 *
 * More efficient than bisection but can be slow if one endpoint becomes "stuck".
 * Superlinear convergence in practice.
 *
 * @tparam DoubleT Numeric type (double or stan::math::var)
 */
template <typename DoubleT>
class FalsePositionSolver : public Solver1D<DoubleT, FalsePositionSolver<DoubleT>> {
public:
    using Base = Solver1D<DoubleT, FalsePositionSolver<DoubleT>>;

    FalsePositionSolver() = default;

    template <typename F>
    DoubleT solveImpl(const F& f, double accuracy, SolverState<DoubleT>& s) const {
        DoubleT froot;

        // Main iteration loop
        while (s.evaluations <= s.maxEvaluations) {
            // False position formula (linear interpolation)
            const DoubleT denominator = s.fxMax - s.fxMin;
            if (Base::value(denominator) == 0.0) {
                throw std::runtime_error("FalsePositionSolver: zero denominator");
            }

            const DoubleT dx = s.fxMax * (s.xMax - s.xMin) / denominator;
            s.root = s.xMax - dx;

            froot = f(s.root);
            ++s.evaluations;

            // Check for convergence
            if (Base::isZero(froot, s.fScale)) {
                return s.root;
            }

            // Update brackets
            if (Base::oppositeSigns(froot, s.fxMax)) {
                // Root is between root and xMax
                s.xMin = s.xMax;
                s.fxMin = s.fxMax;
                s.xMax = s.root;
                s.fxMax = froot;
            } else {
                // Root is between xMin and root
                s.xMax = s.root;
                s.fxMax = froot;
            }

            // Check if interval is small enough
            if (std::fabs(Base::value(s.xMax) - Base::value(s.xMin)) <= accuracy) {
                return s.root;
            }
        }

        throw std::runtime_error("FalsePositionSolver: maximum number of evaluations exceeded");
    }
};
} // namespace Math

#endif // FALSE_POSITION_SOLVER_H
