#ifndef FALSE_POSITION_SOLVER_H
#define FALSE_POSITION_SOLVER_H

#include <cmath>
#include <stdexcept>

#include "Solver1DBase.h"

namespace quantape::math {
/**
 * @brief False Position (Regula Falsi) method for 1D root finding
 *
 * The false position method is a bracketing method similar to bisection,
 * but uses linear interpolation instead of bisecting the interval:
 *
 *   x_new = x_max - f(x_max) * (x_max - x_min) / (f(x_max) - f(x_min))
 *
 * Plain regula falsi can stall with one endpoint stuck forever (for convex or
 * concave objectives), so this implementation uses the Illinois modification:
 * when the same bracket end is retained on two consecutive steps, its function
 * value is halved. That pulls the next interpolant towards the interior and
 * restores superlinear convergence without giving up the bracket. Convergence
 * is judged on the bracket width (the documented accuracy = bracket width),
 * with an exact zero as the only value-based exit.
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
        // Which bracket end the last step retained: 0 = none yet, -1 = xMin,
        // +1 = xMax. The second consecutive retention halves that end's value
        // (Illinois), which cures one-sided stagnation.
        int retained = 0;

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

            // Exact zeros are roots; a merely small |f| is not (it would
            // override the bracket-width accuracy contract).
            if (Base::value(froot) == 0.0) {
                return s.root;
            }

            // Update brackets, keeping xMin < xMax and the root bracketed.
            if (Base::oppositeSigns(froot, s.fxMin)) {
                // Root is between xMin and root: replace xMax, xMin retained.
                s.xMax = s.root;
                s.fxMax = froot;
                if (retained == -1) {
                    s.fxMin = s.fxMin / DoubleT(2.0);
                }
                retained = -1;
            } else {
                // Root is between root and xMax: replace xMin, xMax retained.
                s.xMin = s.root;
                s.fxMin = froot;
                if (retained == 1) {
                    s.fxMax = s.fxMax / DoubleT(2.0);
                }
                retained = 1;
            }

            // Check if interval is small enough
            if (std::fabs(Base::value(s.xMax) - Base::value(s.xMin)) <= accuracy) {
                return s.root;
            }
        }

        throw std::runtime_error("FalsePositionSolver: maximum number of evaluations exceeded");
    }
};
} // namespace quantape::math

#endif // FALSE_POSITION_SOLVER_H
