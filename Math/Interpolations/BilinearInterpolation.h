#ifndef BILINEAR_INTERPOLATION_H
#define BILINEAR_INTERPOLATION_H

#include <vector>

#include "Interpolation2D.h"

namespace Math {

/**
 * @brief Bilinear interpolation on a 2D grid
 *
 * f(x,y) = w00*z[j][i] + w10*z[j][i+1] + w01*z[j+1][i] + w11*z[j+1][i+1]
 * where weights are double (depend only on grid + query point).
 *
 * Linear in z values — Hessian w.r.t. z is zero.
 * var and fvar<var> specializations in InterpolationStanPrimitives.h.
 */
template <typename DoubleT>
class BilinearInterpolation : public Interpolation2D<DoubleT, BilinearInterpolation<DoubleT>> {
    using Base = Interpolation2D<DoubleT, BilinearInterpolation<DoubleT>>;
    friend Base;

public:
    template <typename ContainerX, typename ContainerY, typename Container2D>
    BilinearInterpolation(const ContainerX& x, const ContainerY& y, const Container2D& z) {
        m_x = this->toDoubleVector(x);
        m_y = this->toDoubleVector(y);
        m_z = this->toVector2D(z);

        if (m_z.size() != m_y.size()) {
            throw std::runtime_error("BilinearInterpolation: z rows must match y size");
        }
        for (const auto& row : m_z) {
            if (row.size() != m_x.size()) {
                throw std::runtime_error("BilinearInterpolation: z columns must match x size");
            }
        }
    }

    DoubleT valueImpl(DoubleT x, DoubleT y) const {
        double xv = this->extractDouble(x);
        double yv = this->extractDouble(y);

        size_t i = locateX(xv);
        size_t j = locateY(yv);

        // Grid coordinates are double — off tape
        double x1 = m_x[i], x2 = m_x[i + 1];
        double y1 = m_y[j], y2 = m_y[j + 1];

        // Weights are double — off tape
        double inv_dx = 1.0 / (x2 - x1);
        double inv_dy = 1.0 / (y2 - y1);
        double wx0 = (x2 - xv) * inv_dx;
        double wx1 = (xv - x1) * inv_dx;
        double wy0 = (y2 - yv) * inv_dy;
        double wy1 = (yv - y1) * inv_dy;

        // z values are DoubleT — on tape
        return wy0 * (wx0 * m_z[j][i] + wx1 * m_z[j][i + 1]) +
               wy1 * (wx0 * m_z[j + 1][i] + wx1 * m_z[j + 1][i + 1]);
    }

    bool isInRange(DoubleT x, DoubleT y) const {
        if (m_x.empty() || m_y.empty())
            return false;
        double xv = this->extractDouble(x);
        double yv = this->extractDouble(y);
        return xv >= m_x.front() && xv <= m_x.back() && yv >= m_y.front() && yv <= m_y.back();
    }

    const std::vector<double>& xGrid() const { return m_x; }
    const std::vector<double>& yGrid() const { return m_y; }

private:
    std::vector<double> m_x, m_y;                ///< Grid coordinates (double)
    std::vector<std::vector<DoubleT>> m_z;        ///< Node values (DoubleT, AD-active)

    size_t locateX(double xv) const {
        if (xv <= m_x.front())
            return 0;
        if (xv >= m_x.back())
            return m_x.size() - 2;
        for (size_t i = 0; i < m_x.size() - 1; ++i) {
            if (m_x[i + 1] >= xv)
                return i;
        }
        return m_x.size() - 2;
    }

    size_t locateY(double yv) const {
        if (yv <= m_y.front())
            return 0;
        if (yv >= m_y.back())
            return m_y.size() - 2;
        for (size_t i = 0; i < m_y.size() - 1; ++i) {
            if (m_y[i + 1] >= yv)
                return i;
        }
        return m_y.size() - 2;
    }
};

} // namespace Math

#endif // BILINEAR_INTERPOLATION_H
