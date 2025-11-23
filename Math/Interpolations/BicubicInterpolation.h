#ifndef BICUBIC_INTERPOLATION_H
#define BICUBIC_INTERPOLATION_H

#include <vector>

#include "CubicInterpolation.h"
#include "Interpolation2D.h"

namespace Math {

/**
 * @brief Bicubic interpolation using cubic interpolation along each axis
 *
 * Performs 2D interpolation by:
 * 1. Interpolating along x-direction for each y row
 * 2. Interpolating the results along y-direction
 *
 * Supports all derivative approximation methods from CubicInterpolation:
 * - Spline (DEFAULT): Natural cubic spline (C^2 continuous)
 * - Parabolic: Local parabolic approximation
 * - Akima: Akima's shape-preserving method
 * - Kruger: Harmonic mean (monotonicity-preserving)
 * - Harmonic: Weighted harmonic mean
 *
 * @tparam DoubleT Numeric type (double or stan::math::var)
 */
template <typename DoubleT>
class BicubicInterpolation : public Interpolation2D<DoubleT> {
public:
    using DerivativeApprox = typename CubicInterpolation<DoubleT>::DerivativeApprox;

    /**
     * @brief Construct bicubic interpolation
     * @param x X coordinates - accepts std::vector, Eigen::Vector, etc.
     * @param y Y coordinates - accepts std::vector, Eigen::Vector, etc.
     * @param z Z values (2D grid) - accepts std::vector<std::vector<>>, Eigen::Matrix, etc.
     * @param method Derivative approximation method (default: Spline)
     *
     * Note: z is organized as z[row][col] where row corresponds to y and col to x
     */
    template <typename ContainerX, typename ContainerY, typename Container2D>
    BicubicInterpolation(const ContainerX& x, const ContainerY& y, const Container2D& z,
                         DerivativeApprox method = DerivativeApprox::Spline)
        : m_method(method) {
        m_x = this->toVector(x);
        m_y = this->toVector(y);
        m_z = this->toVector2D(z);

        // Validate dimensions
        if (m_z.size() != m_y.size()) {
            throw std::runtime_error("BicubicInterpolation: z rows must match y size");
        }
        for (const auto& row : m_z) {
            if (row.size() != m_x.size()) {
                throw std::runtime_error("BicubicInterpolation: z columns must match x size");
            }
        }
    }

protected:
    DoubleT valueImpl(DoubleT x, DoubleT y) const override {
        // Step 1: Interpolate along x-direction for each y row
        std::vector<DoubleT> y_values(m_y.size());
        for (size_t i = 0; i < m_y.size(); ++i) {
            CubicInterpolation<DoubleT> x_interp(m_x, m_z[i], m_method);
            y_values[i] = x_interp(x, true); // Allow extrapolation
        }

        // Step 2: Interpolate along y-direction
        CubicInterpolation<DoubleT> y_interp(m_y, y_values, m_method);
        return y_interp(y, true); // Allow extrapolation
    }

    bool isInRange(DoubleT x, DoubleT y) const override {
        if (m_x.empty() || m_y.empty())
            return false;

        double x_val = value(x);
        double y_val = value(y);

        bool x_in_range = (x_val >= value(m_x.front()) && x_val <= value(m_x.back()));
        bool y_in_range = (y_val >= value(m_y.front()) && y_val <= value(m_y.back()));

        return x_in_range && y_in_range;
    }

private:
    DerivativeApprox m_method;
    std::vector<DoubleT> m_x, m_y;
    std::vector<std::vector<DoubleT>> m_z;

    static double value(const DoubleT& x) {
        if constexpr (std::is_same_v<DoubleT, double>) {
            return x;
        } else {
            return x.val();
        }
    }
};

} // namespace Math

#endif // BICUBIC_INTERPOLATION_H
