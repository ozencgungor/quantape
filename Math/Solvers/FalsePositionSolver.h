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
    using FunctionType = typename Base::FunctionType;

    FalsePositionSolver() = default;

    DoubleT solveImpl(const FunctionType& f, double accuracy) const {
        DoubleT froot;

        // Main iteration loop
        while (this->m_evaluationNumber <= this->maxEvaluations()) {
            // False position formula (linear interpolation)
            DoubleT dx =
                this->m_fxMax * (this->m_xMax - this->m_xMin) / (this->m_fxMax - this->m_fxMin);
            this->m_root = this->m_xMax - dx;

            froot = f(this->m_root);
            ++this->m_evaluationNumber;

            // Check for convergence
            if (isClose(froot, DoubleT(0.0))) {
                return this->m_root;
            }

            // Update brackets
            if (value(froot) * value(this->m_fxMax) < 0.0) {
                // Root is between root_ and xMax
                this->m_xMin = this->m_xMax;
                this->m_fxMin = this->m_fxMax;
                this->m_xMax = this->m_root;
                this->m_fxMax = froot;
            } else {
                // Root is between xMin and root_
                this->m_xMax = this->m_root;
                this->m_fxMax = froot;
            }

            // Check if interval is small enough
            if (std::fabs(value(this->m_xMax) - value(this->m_xMin)) <= accuracy) {
                return this->m_root;
            }
        }

        throw std::runtime_error("FalsePositionSolver: maximum number of evaluations exceeded");
    }

private:
    static double value(const DoubleT& x) {
        if constexpr (std::is_same_v<DoubleT, double>) {
            return x;
        } else {
            return x.val();
        }
    }

    static bool isClose(const DoubleT& x, const DoubleT& y) {
        constexpr double EPSILON = std::numeric_limits<double>::epsilon();
        return std::fabs(value(x) - value(y)) < 42.0 * EPSILON;
    }
};
} // namespace Math

#endif // FALSE_POSITION_SOLVER_H
