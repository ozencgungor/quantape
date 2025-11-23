#ifndef INTEGRATOR_H
#define INTEGRATOR_H

#include <functional>
#include <cstddef>
#include <cmath>

namespace Math {
    /**
     * @brief Base integrator class template for AD-compatible numerical integration
     *
     * This class is templated on the numeric type DoubleT, which can be:
     * - double (for regular computation)
     * - stan::math::var (for automatic differentiation)
     *
     * Design based on QuantLib's Integrator class but adapted for AD support.
     *
     * @tparam DoubleT Numeric type (double or stan::math::var)
     */
    template<typename DoubleT>
    class Integrator {
    public:
        using FunctionType = std::function<DoubleT(DoubleT)>;

        /**
         * @brief Construct integrator with accuracy and max evaluations
         * @param absoluteAccuracy Target absolute accuracy for convergence
         * @param maxEvaluations Maximum number of function evaluations
         */
        Integrator(double absoluteAccuracy, size_t maxEvaluations)
            : m_absoluteAccuracy(absoluteAccuracy)
              , m_maxEvaluations(maxEvaluations)
              , m_absoluteError(0.0)
              , m_evaluations(0) {
        }

        virtual ~Integrator() = default;

        /**
         * @brief Integrate function f from a to b
         * @param f Function to integrate
         * @param a Lower bound
         * @param b Upper bound
         * @return Integral value
         */
        DoubleT operator()(const FunctionType &f, DoubleT a, DoubleT b) const {
            m_evaluations = 0;
            m_absoluteError = 0.0;
            return integrate(f, a, b);
        }

        // Modifiers
        void setAbsoluteAccuracy(double accuracy) { m_absoluteAccuracy = accuracy; }
        void setMaxEvaluations(size_t evaluations) { m_maxEvaluations = evaluations; }

        // Inspectors
        double absoluteAccuracy() const { return m_absoluteAccuracy; }
        size_t maxEvaluations() const { return m_maxEvaluations; }
        double absoluteError() const { return m_absoluteError; }
        size_t numberOfEvaluations() const { return m_evaluations; }

        virtual bool integrationSuccess() const {
            return m_absoluteError <= m_absoluteAccuracy;
        }

    protected:
        /**
         * @brief Pure virtual integration method to be implemented by derived classes
         * @param f Function to integrate
         * @param a Lower bound
         * @param b Upper bound
         * @return Integral value
         */
        virtual DoubleT integrate(const FunctionType &f, DoubleT a, DoubleT b) const = 0;

        void setAbsoluteError(double error) const { m_absoluteError = error; }
        void setNumberOfEvaluations(size_t evaluations) const { m_evaluations = evaluations; }
        void increaseNumberOfEvaluations(size_t increase) const { m_evaluations += increase; }

    private:
        double m_absoluteAccuracy;
        size_t m_maxEvaluations;
        mutable double m_absoluteError;
        mutable size_t m_evaluations;
    };
} // namespace Math

#endif // INTEGRATOR_H
