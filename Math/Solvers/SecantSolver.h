#ifndef SECANT_SOLVER_H
#define SECANT_SOLVER_H

#include "Solver1DBase.h"
#include <cmath>
#include <stdexcept>

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
template<typename DoubleT>
class SecantSolver : public Solver1D<DoubleT, SecantSolver<DoubleT>> {
public:
    using Base = Solver1D<DoubleT, SecantSolver<DoubleT>>;
    using FunctionType = typename Base::FunctionType;

    SecantSolver() = default;

    DoubleT solveImpl(const FunctionType& f, double accuracy) const {
        DoubleT fl, froot, dx, xl;

        // Pick the bound with smaller function value as most recent guess
        if (std::fabs(value(this->m_fxMin)) < std::fabs(value(this->m_fxMax))) {
            this->m_root = this->m_xMin;
            froot = this->m_fxMin;
            xl = this->m_xMax;
            fl = this->m_fxMax;
        } else {
            this->m_root = this->m_xMax;
            froot = this->m_fxMax;
            xl = this->m_xMin;
            fl = this->m_fxMin;
        }

        while (this->m_evaluationNumber <= this->maxEvaluations()) {
            // Secant update formula
            dx = (xl - this->m_root) * froot / (froot - fl);
            xl = this->m_root;
            fl = froot;
            this->m_root = this->m_root + dx;
            froot = f(this->m_root);
            ++this->m_evaluationNumber;

            if (std::fabs(value(dx)) < accuracy || isClose(froot, DoubleT(0.0))) {
                return this->m_root;
            }
        }

        throw std::runtime_error("SecantSolver: maximum number of evaluations exceeded");
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

#endif // SECANT_SOLVER_H
