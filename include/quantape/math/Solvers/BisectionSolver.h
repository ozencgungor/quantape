#ifndef BISECTION_SOLVER_H
#define BISECTION_SOLVER_H

#include <cmath>
#include <stdexcept>

#include "Solver1DBase.h"

namespace quantape::math {
/**
 * @brief Bisection method for 1D root finding
 *
 * The simplest and most robust bracketing method. Guaranteed to converge
 * for continuous functions, but convergence is only linear (slow).
 *
 * Algorithm repeatedly halves the interval, keeping the half that
 * brackets the root.
 *
 * @tparam DoubleT Numeric type (double or stan::math::var)
 */
template <typename DoubleT>
class BisectionSolver : public Solver1D<DoubleT, BisectionSolver<DoubleT>> {
public:
    using Base = Solver1D<DoubleT, BisectionSolver<DoubleT>>;

    BisectionSolver() = default;

    /// Objective is a template parameter (no std::function): the loop inlines
    /// and works for any callable DoubleT -> DoubleT, including AD lambdas.
    template <typename F>
    DoubleT solveImpl(const F& f, double accuracy, SolverState<DoubleT>& s) const {
        DoubleT dx, xMid, fMid;

        // Orient the search so that f>0 lies at root + dx
        if (Base::value(s.fxMin) < 0.0) {
            dx = s.xMax - s.xMin;
            s.root = s.xMin;
        } else {
            dx = s.xMin - s.xMax;
            s.root = s.xMax;
        }

        while (s.evaluations <= s.maxEvaluations) {
            dx = dx / DoubleT(2.0);
            xMid = s.root + dx;
            fMid = f(xMid);
            ++s.evaluations;

            if (Base::value(fMid) <= 0.0) {
                s.root = xMid;
            }

            if (std::fabs(Base::value(dx)) < accuracy || Base::isZero(fMid, s.fScale)) {
                return s.root;
            }
        }

        throw std::runtime_error("BisectionSolver: maximum number of evaluations exceeded");
    }
};
} // namespace quantape::math

#endif // BISECTION_SOLVER_H
