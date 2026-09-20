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
 * 1. Interpolating along x-direction for each y row (precomputed at construction)
 * 2. Interpolating the results along y-direction (per query)
 *
 * The x-row CubicInterpolation objects are precomputed at construction time.
 * This means:
 * - For double: coefficient computation happens once, not per-query
 * - For var: O(ny*nx) coefficient tape nodes are created once at construction,
 *   not repeated per-query. Each query adds only ny+1 evaluation tape nodes
 *   (via the CubicInterpolation<var> specialization) plus O(ny) for the
 *   y-column coefficient construction.
 *
 * @tparam DoubleT Numeric type (double, stan::math::var, stan::math::fvar<var>)
 */
template <typename DoubleT>
class BicubicInterpolation : public Interpolation2D<DoubleT, BicubicInterpolation<DoubleT>> {
    using Base = Interpolation2D<DoubleT, BicubicInterpolation<DoubleT>>;
    friend Base;

public:
    using DerivativeApprox = typename CubicInterpolation<DoubleT>::DerivativeApprox;

    template <typename ContainerX, typename ContainerY, typename Container2D>
    BicubicInterpolation(const ContainerX& x, const ContainerY& y, const Container2D& z,
                         DerivativeApprox method = DerivativeApprox::Spline)
        : m_method(method) {
        m_x = this->toDoubleVector(x);
        m_y = this->toDoubleVector(y);
        m_z = this->toVector2D(z);

        if (m_z.size() != m_y.size()) {
            throw std::runtime_error("BicubicInterpolation: z rows must match y size");
        }
        for (const auto& row : m_z) {
            if (row.size() != m_x.size()) {
                throw std::runtime_error("BicubicInterpolation: z columns must match x size");
            }
        }

        // Precompute x-row cubic interpolations.
        // Coefficient computation happens here (once), not per-query.
        // For var: this puts ny*O(nx) tape nodes on the tape at construction.
        m_x_interps.reserve(m_y.size());
        for (size_t j = 0; j < m_y.size(); ++j) {
            m_x_interps.emplace_back(m_x, m_z[j], m_method);
        }
    }

    DoubleT valueImpl(DoubleT x, DoubleT y) const {
        // Step 1: Evaluate precomputed x-row cubics
        // For var: ny * 1 tape node (via CubicInterpolation<var> specialization)
        std::vector<DoubleT> y_values(m_y.size());
        for (size_t j = 0; j < m_y.size(); ++j) {
            y_values[j] = m_x_interps[j](x, true);
        }

        // Step 2: Build y-cubic from intermediate results and evaluate
        // For var: O(ny) tape nodes for coefficient construction + 1 for evaluation
        CubicInterpolation<DoubleT> y_interp(m_y, y_values, m_method);
        return y_interp(y, true);
    }

    bool isInRange(DoubleT x, DoubleT y) const {
        if (m_x.empty() || m_y.empty())
            return false;
        double xv = this->extractDouble(x);
        double yv = this->extractDouble(y);
        return xv >= m_x.front() && xv <= m_x.back() && yv >= m_y.front() && yv <= m_y.back();
    }

private:
    DerivativeApprox m_method;
    std::vector<double> m_x, m_y;                         ///< Grid coordinates (double)
    std::vector<std::vector<DoubleT>> m_z;                ///< Node values (DoubleT, AD-active)
    std::vector<CubicInterpolation<DoubleT>> m_x_interps; ///< Precomputed x-row cubics
};

} // namespace Math

#endif // BICUBIC_INTERPOLATION_H
