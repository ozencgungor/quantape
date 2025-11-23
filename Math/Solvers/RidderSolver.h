#ifndef RIDDER_SOLVER_H
#define RIDDER_SOLVER_H

#include "Solver1DBase.h"
#include <cmath>
#include <stdexcept>
#include <limits>

namespace Math {
    /**
     * @brief Ridder's method for 1D root finding
     *
     * Ridder's method uses an exponential formula to achieve superlinear
     * convergence without requiring derivatives. It's more robust than
     * secant method and faster than bisection.
     *
     * The algorithm uses the formula:
     *   x_new = x_mid + (x_mid - x_min) * sign(f_min - f_max) * f_mid / sqrt(f_mid² - f_min*f_max)
     *
     * Convergence order: ~1.839
     *
     * @tparam DoubleT Numeric type (double or stan::math::var)
     */
    template<typename DoubleT>
    class RidderSolver : public Solver1D<DoubleT, RidderSolver<DoubleT> > {
    public:
        using Base = Solver1D<DoubleT, RidderSolver<DoubleT> >;
        using FunctionType = typename Base::FunctionType;

        RidderSolver() = default;

        DoubleT solveImpl(const FunctionType &f, double accuracy) const {
            DoubleT fxMid, froot, s, xMid, nextRoot;

            // Ridder algorithm provides accuracy 100x below requested in practice
            double xAccuracy = accuracy / 100.0;

            // Initialize with unlikely value
            this->m_root = DoubleT(std::numeric_limits<double>::lowest());

            while (this->m_evaluationNumber <= this->maxEvaluations()) {
                xMid = (this->m_xMin + this->m_xMax) / DoubleT(2.0);

                // First of two function evaluations per iteration
                fxMid = f(xMid);
                ++this->m_evaluationNumber;

                // Compute discriminant
                s = sqrt(fxMid * fxMid - this->m_fxMin * this->m_fxMax);

                if (isClose(s, DoubleT(0.0))) {
                    f(this->m_root);
                    ++this->m_evaluationNumber;
                    return this->m_root;
                }

                // Ridder's update formula
                DoubleT sign = (value(this->m_fxMin) >= value(this->m_fxMax)) ? DoubleT(1.0) : DoubleT(-1.0);
                nextRoot = xMid + (xMid - this->m_xMin) * sign * fxMid / s;

                if (std::fabs(value(nextRoot) - value(this->m_root)) <= xAccuracy) {
                    f(this->m_root);
                    ++this->m_evaluationNumber;
                    return this->m_root;
                }

                this->m_root = nextRoot;

                // Second of two function evaluations per iteration
                froot = f(this->m_root);
                ++this->m_evaluationNumber;

                if (isClose(froot, DoubleT(0.0))) {
                    return this->m_root;
                }

                // Update brackets to keep root bracketed
                if (signCopy(fxMid, froot) != value(fxMid)) {
                    this->m_xMin = xMid;
                    this->m_fxMin = fxMid;
                    this->m_xMax = this->m_root;
                    this->m_fxMax = froot;
                } else if (signCopy(this->m_fxMin, froot) != value(this->m_fxMin)) {
                    this->m_xMax = this->m_root;
                    this->m_fxMax = froot;
                } else if (signCopy(this->m_fxMax, froot) != value(this->m_fxMax)) {
                    this->m_xMin = this->m_root;
                    this->m_fxMin = froot;
                } else {
                    throw std::runtime_error("RidderSolver: internal error in bracketing logic");
                }

                if (std::fabs(value(this->m_xMax) - value(this->m_xMin)) <= xAccuracy) {
                    f(this->m_root);
                    ++this->m_evaluationNumber;
                    return this->m_root;
                }
            }

            throw std::runtime_error("RidderSolver: maximum number of evaluations exceeded");
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

        static DoubleT sqrt(const DoubleT &x) {
            if constexpr (std::is_same_v<DoubleT, double>) {
                return std::sqrt(x);
            } else {
                // For AD types, use their sqrt implementation
                using std::sqrt;
                return sqrt(x);
            }
        }

        /**
         * @brief Returns |a| if b >= 0, else -|a|
         */
        static double signCopy(const DoubleT &a, const DoubleT &b) {
            double abs_a = std::fabs(value(a));
            return value(b) >= 0.0 ? abs_a : -abs_a;
        }
    };
} // namespace Math

#endif // RIDDER_SOLVER_H
