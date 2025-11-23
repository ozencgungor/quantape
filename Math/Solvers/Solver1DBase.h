#ifndef SOLVER1D_BASE_H
#define SOLVER1D_BASE_H

#include <functional>
#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <limits>

namespace Math {
    /**
     * @brief Base class for 1-D solvers using CRTP
     *
     * Uses the "Curiously Recurring Template Pattern" (CRTP) for static polymorphism.
     * Concrete solvers are declared as:
     *   class BrentSolver : public Solver1D<double, BrentSolver> { ... }
     *
     * Design based on QuantLib's Solver1D but templated for AD support.
     *
     * @tparam DoubleT Numeric type (double or stan::math::var)
     * @tparam Impl Derived solver implementation (CRTP)
     */
    template<typename DoubleT, typename Impl>
    class Solver1D {
    public:
        using FunctionType = std::function<DoubleT(DoubleT)>;

        Solver1D()
            : m_evaluationNumber(0)
              , m_maxEvaluations(100)
              , m_lowerBound(0.0)
              , m_upperBound(0.0)
              , m_lowerBoundEnforced(false)
              , m_upperBoundEnforced(false) {
        }

        /**
         * @brief Solve for root with bracketing
         * @param f Function to find root of
         * @param accuracy Target accuracy
         * @param guess Initial guess
         * @param xMin Lower bracket
         * @param xMax Upper bracket
         * @return Root value
         */
        DoubleT solve(const FunctionType &f,
                      double accuracy,
                      DoubleT guess,
                      DoubleT xMin,
                      DoubleT xMax) const {
            if (accuracy <= 0.0) {
                throw std::invalid_argument("accuracy must be positive");
            }

            // Use at least machine epsilon
            constexpr double EPSILON = std::numeric_limits<double>::epsilon();
            accuracy = std::max(accuracy, EPSILON);

            m_xMin = xMin;
            m_xMax = xMax;

            if (value(m_xMin) >= value(m_xMax)) {
                throw std::invalid_argument("invalid range: xMin >= xMax");
            }

            if (m_lowerBoundEnforced && value(m_xMin) < m_lowerBound) {
                throw std::invalid_argument("xMin < enforced lower bound");
            }

            if (m_upperBoundEnforced && value(m_xMax) > m_upperBound) {
                throw std::invalid_argument("xMax > enforced upper bound");
            }

            m_fxMin = f(m_xMin);
            if (isClose(m_fxMin, DoubleT(0.0))) {
                return m_xMin;
            }

            m_fxMax = f(m_xMax);
            if (isClose(m_fxMax, DoubleT(0.0))) {
                return m_xMax;
            }

            m_evaluationNumber = 2;

            // Check bracketing
            if (value(m_fxMin) * value(m_fxMax) >= 0.0) {
                throw std::runtime_error("root not bracketed");
            }

            if (value(guess) <= value(m_xMin) || value(guess) >= value(m_xMax)) {
                throw std::invalid_argument("guess must be strictly between xMin and xMax");
            }

            m_root = guess;

            // Call derived class implementation
            return static_cast<const Impl *>(this)->solveImpl(f, accuracy);
        }

        /**
         * @brief Solve for root with automatic bracketing
         * @param f Function to find root of
         * @param accuracy Target accuracy
         * @param guess Initial guess
         * @param step Initial step size for bracketing
         * @return Root value
         */
        DoubleT solve(const FunctionType &f,
                      double accuracy,
                      DoubleT guess,
                      DoubleT step) const {
            if (accuracy <= 0.0) {
                throw std::invalid_argument("accuracy must be positive");
            }

            constexpr double EPSILON = std::numeric_limits<double>::epsilon();
            accuracy = std::max(accuracy, EPSILON);

            const double growthFactor = 1.6;
            int flipflop = -1;

            m_root = guess;
            m_fxMax = f(m_root);

            if (isClose(m_fxMax, DoubleT(0.0))) {
                return m_root;
            } else if (value(m_fxMax) > 0.0) {
                m_xMin = enforceBounds(m_root - step);
                m_fxMin = f(m_xMin);
                m_xMax = m_root;
            } else {
                m_xMin = m_root;
                m_fxMin = m_fxMax;
                m_xMax = enforceBounds(m_root + step);
                m_fxMax = f(m_xMax);
            }

            m_evaluationNumber = 2;

            while (m_evaluationNumber <= m_maxEvaluations) {
                if (value(m_fxMin) * value(m_fxMax) <= 0.0) {
                    if (isClose(m_fxMin, DoubleT(0.0))) return m_xMin;
                    if (isClose(m_fxMax, DoubleT(0.0))) return m_xMax;
                    m_root = (m_xMax + m_xMin) / DoubleT(2.0);
                    return static_cast<const Impl *>(this)->solveImpl(f, accuracy);
                }

                if (std::fabs(value(m_fxMin)) < std::fabs(value(m_fxMax))) {
                    m_xMin = enforceBounds(m_xMin + DoubleT(growthFactor) * (m_xMin - m_xMax));
                    m_fxMin = f(m_xMin);
                } else if (std::fabs(value(m_fxMin)) > std::fabs(value(m_fxMax))) {
                    m_xMax = enforceBounds(m_xMax + DoubleT(growthFactor) * (m_xMax - m_xMin));
                    m_fxMax = f(m_xMax);
                } else if (flipflop == -1) {
                    m_xMin = enforceBounds(m_xMin + DoubleT(growthFactor) * (m_xMin - m_xMax));
                    m_fxMin = f(m_xMin);
                    m_evaluationNumber++;
                    flipflop = 1;
                } else if (flipflop == 1) {
                    m_xMax = enforceBounds(m_xMax + DoubleT(growthFactor) * (m_xMax - m_xMin));
                    m_fxMax = f(m_xMax);
                    flipflop = -1;
                }
                m_evaluationNumber++;
            }

            throw std::runtime_error("unable to bracket root in max function evaluations");
        }

        // Modifiers
        void setMaxEvaluations(size_t evaluations) { m_maxEvaluations = evaluations; }

        void setLowerBound(double lowerBound) {
            m_lowerBound = lowerBound;
            m_lowerBoundEnforced = true;
        }

        void setUpperBound(double upperBound) {
            m_upperBound = upperBound;
            m_upperBoundEnforced = true;
        }

        // Inspectors
        size_t maxEvaluations() const { return m_maxEvaluations; }

    protected:
        mutable DoubleT m_root, m_xMin, m_xMax, m_fxMin, m_fxMax;
        mutable size_t m_evaluationNumber;

    private:
        size_t m_maxEvaluations;
        double m_lowerBound, m_upperBound;
        bool m_lowerBoundEnforced, m_upperBoundEnforced;

        DoubleT enforceBounds(DoubleT x) const {
            if (m_lowerBoundEnforced && value(x) < m_lowerBound) {
                return DoubleT(m_lowerBound);
            }
            if (m_upperBoundEnforced && value(x) > m_upperBound) {
                return DoubleT(m_upperBound);
            }
            return x;
        }

        static bool isClose(const DoubleT &x, const DoubleT &y) {
            constexpr double EPSILON = std::numeric_limits<double>::epsilon();
            return std::fabs(value(x) - value(y)) < 42.0 * EPSILON;
        }

        static double value(const DoubleT &x) {
            if constexpr (std::is_same_v<DoubleT, double>) {
                return x;
            } else {
                return x.val(); // For stan::math::var
            }
        }
    };
} // namespace Math

#endif // SOLVER1D_BASE_H
