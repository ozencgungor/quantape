#ifndef TRAPEZOID_INTEGRATOR_H
#define TRAPEZOID_INTEGRATOR_H

#include <cmath>
#include <stdexcept>
#include <type_traits>

#include "Integrator.h"

namespace Math {
/**
 * @brief Integration policies for trapezoid integration
 *
 * Policies define different refinement strategies for adaptive integration.
 * Each policy must provide:
 * - integrate() method to refine the integral estimate
 * - nbEvaluations() to report how many new evaluations are added per refinement
 */

/**
 * @brief Default policy: doubles the number of intervals at each refinement
 */
template <typename DoubleT>
struct DefaultPolicy {
    using FunctionType = std::function<DoubleT(DoubleT)>;

    static DoubleT integrate(const FunctionType& f, DoubleT a, DoubleT b, DoubleT I, size_t N) {
        DoubleT sum = DoubleT(0.0);
        DoubleT dx = (b - a) / DoubleT(N);
        DoubleT x = a + dx / DoubleT(2.0);

        for (size_t i = 0; i < N; ++i) {
            sum = sum + f(x);
            x = x + dx;
        }

        return (I + dx * sum) / DoubleT(2.0);
    }

    static size_t nbEvaluations() { return 2; }
};

/**
 * @brief MidPoint policy: triples the number of intervals at each refinement
 */
template <typename DoubleT>
struct MidPointPolicy {
    using FunctionType = std::function<DoubleT(DoubleT)>;

    static DoubleT integrate(const FunctionType& f, DoubleT a, DoubleT b, DoubleT I, size_t N) {
        DoubleT sum = DoubleT(0.0);
        DoubleT dx = (b - a) / DoubleT(N);
        DoubleT x = a + dx / DoubleT(6.0);
        DoubleT D = DoubleT(2.0) * dx / DoubleT(3.0);

        for (size_t i = 0; i < N; ++i) {
            sum = sum + f(x) + f(x + D);
            x = x + dx;
        }

        return (I + dx * sum) / DoubleT(3.0);
    }

    static size_t nbEvaluations() { return 3; }
};

/**
 * @brief Trapezoid integrator with adaptive refinement
 *
 * Integrates a function using the trapezoid rule with adaptive refinement.
 * The number of intervals is repeatedly increased until the target accuracy
 * is reached or max evaluations is exceeded.
 *
 * Design based on QuantLib's TrapezoidIntegral but templated for AD support.
 *
 * @tparam DoubleT Numeric type (double or stan::math::var)
 * @tparam IntegrationPolicy Policy for refinement strategy
 */
template <typename DoubleT, template <typename> class IntegrationPolicy = DefaultPolicy>
class TrapezoidIntegrator : public Integrator<DoubleT> {
public:
    using FunctionType = typename Integrator<DoubleT>::FunctionType;

    TrapezoidIntegrator(double accuracy, size_t maxIterations)
        : Integrator<DoubleT>(accuracy, maxIterations) {}

    /**
     * @brief Integrate f from a to b.
     *
     * double: the adaptive refinement implemented below. var and fvar<var>:
     * the converged quadrature rule is extracted in one double-valued pass
     * and the AD integrand samples are combined with it through a single
     * callback var per output (IntegratorStanPrimitives.h).
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

        // Refine until convergence or max iterations
        size_t i = 1;
        DoubleT newI;

        do {
            newI = IntegrationPolicy<DoubleT>::integrate(f, a, b, I, N);
            this->increaseNumberOfEvaluations(N *
                                              (IntegrationPolicy<DoubleT>::nbEvaluations() - 1));
            N *= IntegrationPolicy<DoubleT>::nbEvaluations();

            // Check convergence (also don't exit too early)
            if (i > 5) {
                // For AD types, we need to extract the value for comparison
                double error = std::fabs(value(I) - value(newI));
                this->setAbsoluteError(error);
                if (error <= this->absoluteAccuracy()) {
                    return newI;
                }
            }

            I = newI;
            ++i;
        } while (i < this->maxEvaluations());

        throw std::runtime_error("TrapezoidIntegrator: max number of iterations reached");
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

// Convenient typedefs
template <typename DoubleT>
using TrapezoidIntegratorDefault = TrapezoidIntegrator<DoubleT, DefaultPolicy>;

template <typename DoubleT>
using TrapezoidIntegratorMidPoint = TrapezoidIntegrator<DoubleT, MidPointPolicy>;
} // namespace Math

#endif // TRAPEZOID_INTEGRATOR_H
