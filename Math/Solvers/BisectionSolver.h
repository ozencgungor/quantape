#ifndef BISECTION_SOLVER_H
#define BISECTION_SOLVER_H

#include "Solver1DBase.h"
#include <cmath>
#include <stdexcept>

namespace Math {
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
    template<typename DoubleT>
    class BisectionSolver : public Solver1D<DoubleT, BisectionSolver<DoubleT> > {
    public:
        using Base = Solver1D<DoubleT, BisectionSolver<DoubleT> >;
        using FunctionType = typename Base::FunctionType;

        BisectionSolver() = default;

        DoubleT solveImpl(const FunctionType &f, double accuracy) const {
            DoubleT dx, xMid, fMid;

            // Orient the search so that f>0 lies at root + dx
            if (value(this->m_fxMin) < 0.0) {
                dx = this->m_xMax - this->m_xMin;
                this->m_root = this->m_xMin;
            } else {
                dx = this->m_xMin - this->m_xMax;
                this->m_root = this->m_xMax;
            }

            while (this->m_evaluationNumber <= this->maxEvaluations()) {
                dx = dx / DoubleT(2.0);
                xMid = this->m_root + dx;
                fMid = f(xMid);
                ++this->m_evaluationNumber;

                if (value(fMid) <= 0.0) {
                    this->m_root = xMid;
                }

                if (std::fabs(value(dx)) < accuracy || isClose(fMid, DoubleT(0.0))) {
                    f(this->m_root); // Final evaluation
                    ++this->m_evaluationNumber;
                    return this->m_root;
                }
            }

            throw std::runtime_error("BisectionSolver: maximum number of evaluations exceeded");
        }

    private:
        static double value(const DoubleT &x) {
            if constexpr (std::is_same_v<DoubleT, double>) {
                return x;
            } else {
                return x.val();
            }
        }

        static bool isClose(const DoubleT &x, const DoubleT &y) {
            constexpr double EPSILON = std::numeric_limits<double>::epsilon();
            return std::fabs(value(x) - value(y)) < 42.0 * EPSILON;
        }
    };
} // namespace Math

#endif // BISECTION_SOLVER_H
