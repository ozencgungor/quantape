#ifndef SIMPSON_INTEGRATOR_H
#define SIMPSON_INTEGRATOR_H

#include <cmath>
#include <stdexcept>
#include <type_traits>

#include "TrapezoidIntegrator.h"

namespace Math {
/**
 * @brief Simpson's rule integrator with adaptive refinement
 *
 * Uses Richardson extrapolation on the trapezoid rule to achieve
 * fourth-order accuracy. The formula is:
 *   Simpson = (4 * Trapezoid_fine - Trapezoid_coarse) / 3
 *
 * This is equivalent to Simpson's 1/3 rule with adaptive refinement.
 *
 * @tparam DoubleT Numeric type (double or stan::math::var)
 */
template <typename DoubleT>
class SimpsonIntegrator : public Integrator<DoubleT> {
public:
    using FunctionType = typename Integrator<DoubleT>::FunctionType;

    SimpsonIntegrator(double accuracy, size_t maxIterations)
        : Integrator<DoubleT>(accuracy, maxIterations) {}

    /**
     * @brief Integrate f from a to b.
     *
     * double: the adaptive Richardson extrapolation implemented below. var
     * and fvar<var>: the converged quadrature rule is extracted in one
     * double-valued pass and the AD integrand samples are combined with it
     * through a single callback var per output (IntegratorStanPrimitives.h).
     */
    DoubleT operator()(const FunctionType& f, DoubleT a, DoubleT b) const {
        if constexpr (std::is_same_v<DoubleT, double>) {
            return Integrator<DoubleT>::operator()(f, a, b);
        } else {
            return integratePrimitives(f, a, b);
        }
    }

protected:
    DoubleT integrate(const FunctionType& f, DoubleT a, DoubleT b) const override {
        // Start from the coarsest trapezoid
        size_t N = 1;
        DoubleT I = (f(a) + f(b)) * (b - a) / DoubleT(2.0);
        this->increaseNumberOfEvaluations(2);

        DoubleT adjI = I; // Simpson-adjusted integral
        size_t i = 1;
        DoubleT newI, newAdjI;

        do {
            // Refine using default trapezoid policy
            newI = DefaultPolicy<DoubleT>::integrate(f, a, b, I, N);
            this->increaseNumberOfEvaluations(N);
            N *= 2;

            // Richardson extrapolation: Simpson = (4*newI - I)/3
            newAdjI = (DoubleT(4.0) * newI - I) / DoubleT(3.0);

            // Check convergence (also don't exit too early)
            if (i > 5) {
                double error = std::fabs(value(adjI) - value(newAdjI));
                this->setAbsoluteError(error);
                if (error <= this->absoluteAccuracy()) {
                    return newAdjI;
                }
            }

            I = newI;
            adjI = newAdjI;
            ++i;
        } while (i < this->maxEvaluations());

        throw std::runtime_error("SimpsonIntegrator: max number of iterations reached");
    }

private:
    // Recursive primal extraction: double -> itself, var -> .val(),
    // fvar<var> -> .val().val()
    static double value(double x) { return x; }
    template <typename T>
    static double value(const T& x) {
        return value(x.val());
    }

    // Defined for stan::math::var / stan::math::fvar<var> in
    // IntegratorStanPrimitives.h. Never ODR-used for double (nor for the
    // RuleScalar probe, which enters through the base operator()).
    DoubleT integratePrimitives(const FunctionType& f, DoubleT a, DoubleT b) const;
};
} // namespace Math

#endif // SIMPSON_INTEGRATOR_H
