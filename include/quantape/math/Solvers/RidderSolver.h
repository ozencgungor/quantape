#ifndef RIDDER_SOLVER_H
#define RIDDER_SOLVER_H

#include <cmath>
#include <limits>
#include <stdexcept>
#include <type_traits>

#include "Solver1DBase.h"

namespace quantape::math {
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
template <typename DoubleT>
class RidderSolver : public Solver1D<DoubleT, RidderSolver<DoubleT>> {
public:
    using Base = Solver1D<DoubleT, RidderSolver<DoubleT>>;

    RidderSolver() = default;

    template <typename F>
    DoubleT solveImpl(const F& f, double accuracy, SolverState<DoubleT>& s) const {
        DoubleT fxMid, froot, discriminant, xMid, nextRoot;

        // Ridder algorithm provides accuracy 100x below requested in practice
        const double xAccuracy = accuracy / 100.0;

        // Initialize with unlikely value
        s.root = DoubleT(std::numeric_limits<double>::lowest());

        while (s.evaluations <= s.maxEvaluations) {
            xMid = (s.xMin + s.xMax) / DoubleT(2.0);

            // First of two function evaluations per iteration
            fxMid = f(xMid);
            ++s.evaluations;

            // Compute discriminant
            discriminant = sqrtValue(fxMid * fxMid - s.fxMin * s.fxMax);

            if (Base::value(discriminant) == 0.0) {
                return xMid;
            }

            // Ridder's update formula
            const DoubleT sign =
                (Base::value(s.fxMin) >= Base::value(s.fxMax)) ? DoubleT(1.0) : DoubleT(-1.0);
            nextRoot = xMid + (xMid - s.xMin) * sign * fxMid / discriminant;

            if (std::fabs(Base::value(nextRoot) - Base::value(s.root)) <= xAccuracy) {
                return nextRoot;
            }

            s.root = nextRoot;

            // Second of two function evaluations per iteration
            froot = f(s.root);
            ++s.evaluations;

            if (Base::isZero(froot, s.fScale)) {
                return s.root;
            }

            // Update brackets to keep root bracketed
            if (Base::oppositeSigns(fxMid, froot)) {
                s.xMin = xMid;
                s.fxMin = fxMid;
                s.xMax = s.root;
                s.fxMax = froot;
            } else if (Base::oppositeSigns(s.fxMin, froot)) {
                s.xMax = s.root;
                s.fxMax = froot;
            } else if (Base::oppositeSigns(s.fxMax, froot)) {
                s.xMin = s.root;
                s.fxMin = froot;
            } else {
                throw std::runtime_error("RidderSolver: internal error in bracketing logic");
            }

            if (std::fabs(Base::value(s.xMax) - Base::value(s.xMin)) <= xAccuracy) {
                return s.root;
            }
        }

        throw std::runtime_error("RidderSolver: maximum number of evaluations exceeded");
    }

private:
    /// sqrt for double (std::) and AD scalars (ADL finds the AD overload)
    static DoubleT sqrtValue(const DoubleT& x) {
        if constexpr (std::is_arithmetic_v<DoubleT>) {
            return std::sqrt(x);
        } else {
            using std::sqrt;
            return sqrt(x);
        }
    }
};
} // namespace quantape::math

#endif // RIDDER_SOLVER_H
