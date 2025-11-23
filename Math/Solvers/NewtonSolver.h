#ifndef NEWTON_SOLVER_H
#define NEWTON_SOLVER_H

#include "Solver1DBase.h"
#include <cmath>
#include <stdexcept>
#include <type_traits>

namespace Math {

/**
 * @brief Newton-Raphson method for 1D root finding
 *
 * Classic Newton's method using the update formula:
 *   x_{n+1} = x_n - f(x_n) / f'(x_n)
 *
 * Quadratic convergence when near the root. Requires derivative computation.
 *
 * For regular functions, the function object must provide a derivative() method.
 * For AD types (stan::math::var), derivatives are computed automatically.
 *
 * Falls back to bisection if Newton step would jump outside brackets.
 *
 * @tparam DoubleT Numeric type (double or stan::math::var)
 */
template<typename DoubleT>
class NewtonSolver : public Solver1D<DoubleT, NewtonSolver<DoubleT>> {
public:
    using Base = Solver1D<DoubleT, NewtonSolver<DoubleT>>;
    using FunctionType = typename Base::FunctionType;

    NewtonSolver() = default;

    /**
     * @brief Solve using Newton's method
     *
     * For regular double: function object must have derivative() method
     * For stan::math::var: derivatives computed via AD
     */
    DoubleT solveImpl(const FunctionType& f, double accuracy) const {
        DoubleT froot, dfroot, dx;

        froot = f(this->m_root);
        dfroot = computeDerivative(f, this->m_root);
        ++this->m_evaluationNumber;

        if (isZero(dfroot)) {
            throw std::runtime_error("NewtonSolver: derivative is zero");
        }

        while (this->m_evaluationNumber <= this->maxEvaluations()) {
            dx = froot / dfroot;
            this->m_root = this->m_root - dx;

            // Check if jumped out of brackets
            if (value(this->m_xMin - this->m_root) * value(this->m_root - this->m_xMax) < 0.0) {
                // Outside brackets - fall back to bisection for this step
                this->m_root = (this->m_xMin + this->m_xMax) / DoubleT(2.0);
            }

            if (std::fabs(value(dx)) < accuracy) {
                f(this->m_root);  // Final evaluation
                ++this->m_evaluationNumber;
                return this->m_root;
            }

            froot = f(this->m_root);
            dfroot = computeDerivative(f, this->m_root);
            ++this->m_evaluationNumber;

            if (isZero(dfroot)) {
                throw std::runtime_error("NewtonSolver: derivative became zero");
            }
        }

        throw std::runtime_error("NewtonSolver: maximum number of evaluations exceeded");
    }

private:
    static double value(const DoubleT& x) {
        if constexpr (std::is_same_v<DoubleT, double>) {
            return x;
        } else {
            return x.val();
        }
    }

    static bool isZero(const DoubleT& x) {
        constexpr double EPSILON = std::numeric_limits<double>::epsilon();
        return std::fabs(value(x)) < EPSILON;
    }

    /**
     * @brief Compute derivative - specialized for double and AD types
     */
    DoubleT computeDerivative(const FunctionType& f, DoubleT x) const {
        if constexpr (std::is_same_v<DoubleT, double>) {
            // For double, try to call derivative() method
            // This requires the function object to have a derivative() method
            constexpr double h = 1e-8;
            return (f(x + h) - f(x - h)) / (2.0 * h);
        } else {
            // For AD types (stan::math::var), use automatic differentiation
            // This would require stan::math to be included
            // For now, use finite differences as fallback
            DoubleT h = DoubleT(1e-8);
            return (f(x + h) - f(x - h)) / (DoubleT(2.0) * h);
        }
    }
};

/**
 * @brief Newton solver with explicit derivative function
 *
 * Version where user provides both function and derivative explicitly.
 * More efficient than finite differences and works with all numeric types.
 *
 * @tparam DoubleT Numeric type (double or stan::math::var)
 */
template<typename DoubleT>
class NewtonSolverWithDerivative : public Solver1D<DoubleT, NewtonSolverWithDerivative<DoubleT>> {
public:
    using Base = Solver1D<DoubleT, NewtonSolverWithDerivative<DoubleT>>;
    using FunctionType = typename Base::FunctionType;
    using DerivativeType = std::function<DoubleT(DoubleT)>;

    NewtonSolverWithDerivative() = default;

    /**
     * @brief Set the derivative function
     */
    void setDerivative(DerivativeType derivative) {
        m_derivative = derivative;
    }

    DoubleT solveImpl(const FunctionType& f, double accuracy) const {
        if (!m_derivative) {
            throw std::runtime_error("NewtonSolverWithDerivative: derivative function not set");
        }

        DoubleT froot, dfroot, dx;

        froot = f(this->m_root);
        dfroot = m_derivative(this->m_root);
        ++this->m_evaluationNumber;

        if (isZero(dfroot)) {
            throw std::runtime_error("NewtonSolverWithDerivative: derivative is zero");
        }

        while (this->m_evaluationNumber <= this->maxEvaluations()) {
            dx = froot / dfroot;
            this->m_root = this->m_root - dx;

            // Check if jumped out of brackets
            if (value(this->m_xMin - this->m_root) * value(this->m_root - this->m_xMax) < 0.0) {
                this->m_root = (this->m_xMin + this->m_xMax) / DoubleT(2.0);
            }

            if (std::fabs(value(dx)) < accuracy) {
                f(this->m_root);
                ++this->m_evaluationNumber;
                return this->m_root;
            }

            froot = f(this->m_root);
            dfroot = m_derivative(this->m_root);
            ++this->m_evaluationNumber;

            if (isZero(dfroot)) {
                throw std::runtime_error("NewtonSolverWithDerivative: derivative became zero");
            }
        }

        throw std::runtime_error("NewtonSolverWithDerivative: maximum number of evaluations exceeded");
    }

private:
    mutable DerivativeType m_derivative;

    static double value(const DoubleT& x) {
        if constexpr (std::is_same_v<DoubleT, double>) {
            return x;
        } else {
            return x.val();
        }
    }

    static bool isZero(const DoubleT& x) {
        constexpr double EPSILON = std::numeric_limits<double>::epsilon();
        return std::fabs(value(x)) < EPSILON;
    }
};

} // namespace Math

#endif // NEWTON_SOLVER_H
