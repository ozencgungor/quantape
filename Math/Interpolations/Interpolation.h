#ifndef INTERPOLATION_H
#define INTERPOLATION_H

#include <vector>
#include <algorithm>
#include <stdexcept>
#include <cmath>

namespace Math {
    /**
     * @brief Base class for 1D interpolations
     *
     * Provides common functionality for all 1D interpolation methods:
     * - Binary search for x location
     * - Range checking
     * - Value extraction for AD types
     *
     * Derived classes must implement:
     * - valueImpl(x): Interpolated value at x
     * - derivativeImpl(x): Derivative at x
     *
     * @tparam DoubleT Numeric type (double or stan::math::var)
     */
    template<typename DoubleT>
    class Interpolation {
    public:
        /**
         * @brief Construct empty interpolation
         */
        Interpolation() = default;

        virtual ~Interpolation() = default;

        /**
         * @brief Get interpolated value at x
         * @param x Point at which to interpolate
         * @param allowExtrapolation Allow extrapolation outside data range
         * @return Interpolated value
         */
        DoubleT operator()(DoubleT x, bool allowExtrapolation = false) const {
            if (!allowExtrapolation && !isInRange(x)) {
                throw std::runtime_error("Interpolation: x is out of range");
            }
            return valueImpl(x);
        }

        /**
         * @brief Get derivative at x
         * @param x Point at which to compute derivative
         * @param allowExtrapolation Allow extrapolation outside data range
         * @return Derivative value
         */
        DoubleT derivative(DoubleT x, bool allowExtrapolation = false) const {
            if (!allowExtrapolation && !isInRange(x)) {
                throw std::runtime_error("Interpolation: x is out of range for derivative");
            }
            return derivativeImpl(x);
        }

        /**
         * @brief Get minimum x value
         */
        double xMin() const {
            if (m_x.empty()) {
                throw std::runtime_error("Interpolation: no data");
            }
            return value(m_x.front());
        }

        /**
         * @brief Get maximum x value
         */
        double xMax() const {
            if (m_x.empty()) {
                throw std::runtime_error("Interpolation: no data");
            }
            return value(m_x.back());
        }

        /**
         * @brief Get number of data points
         */
        size_t size() const {
            return m_x.size();
        }

        /**
         * @brief Check if x is in interpolation range
         */
        bool isInRange(DoubleT x) const {
            if (m_x.empty()) return false;
            double xVal = value(x);
            return xVal >= value(m_x.front()) && xVal <= value(m_x.back());
        }

    protected:
        std::vector<DoubleT> m_x; ///< X coordinates
        std::vector<DoubleT> m_y; ///< Y coordinates

        /**
         * @brief Locate interval containing x using binary search
         * @param x Point to locate
         * @return Index i such that x[i] <= x < x[i+1]
         */
        size_t locate(DoubleT x) const {
            if (m_x.size() < 2) {
                throw std::runtime_error("Interpolation: need at least 2 points");
            }

            double xVal = value(x);

            // Handle boundary cases
            if (xVal <= value(m_x.front())) return 0;
            if (xVal >= value(m_x.back())) return m_x.size() - 2;

            // Binary search for the interval
            size_t left = 0;
            size_t right = m_x.size() - 1;

            while (right - left > 1) {
                size_t mid = left + (right - left) / 2;
                if (value(m_x[mid]) <= xVal) {
                    left = mid;
                } else {
                    right = mid;
                }
            }

            return left;
        }

        /**
         * @brief Extract double value from DoubleT
         */
        static double value(const DoubleT &x) {
            if constexpr (std::is_same_v<DoubleT, double>) {
                return x;
            } else {
                return x.val(); // For stan::math::var
            }
        }

        /**
         * @brief Validate data for interpolation
         */
        void validate() const {
            if (m_x.size() != m_y.size()) {
                throw std::runtime_error("Interpolation: x and y must have same size");
            }
            if (m_x.size() < 2) {
                throw std::runtime_error("Interpolation: need at least 2 points");
            }

            // Check that x values are sorted
            for (size_t i = 1; i < m_x.size(); ++i) {
                if (value(m_x[i]) <= value(m_x[i - 1])) {
                    throw std::runtime_error("Interpolation: x values must be strictly increasing");
                }
            }
        }

        /**
         * @brief Compute interpolated value (to be implemented by derived classes)
         */
        virtual DoubleT valueImpl(DoubleT x) const = 0;

        /**
         * @brief Compute derivative (to be implemented by derived classes)
         */
        virtual DoubleT derivativeImpl(DoubleT x) const = 0;
    };
} // namespace Math

#endif // INTERPOLATION_H
