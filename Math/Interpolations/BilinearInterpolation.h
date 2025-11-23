#ifndef BILINEAR_INTERPOLATION_H
#define BILINEAR_INTERPOLATION_H

#include <vector>

#include "Interpolation2D.h"

namespace Math {

template <typename DoubleT>
class BilinearInterpolation : public Interpolation2D<DoubleT> {
public:
    /**
     * @brief Construct bilinear interpolation
     * @param x X coordinates - accepts std::vector, Eigen::Vector, etc.
     * @param y Y coordinates - accepts std::vector, Eigen::Vector, etc.
     * @param z Z values (2D grid) - accepts std::vector<std::vector<>>, Eigen::Matrix, etc.
     */
    template <typename ContainerX, typename ContainerY, typename Container2D>
    BilinearInterpolation(const ContainerX& x, const ContainerY& y, const Container2D& z) {
        m_x = this->toVector(x);
        m_y = this->toVector(y);
        m_z = this->toVector2D(z);

        // Validate dimensions
        if (m_z.size() != m_y.size()) {
            throw std::runtime_error("BilinearInterpolation: z rows must match y size");
        }
        for (const auto& row : m_z) {
            if (row.size() != m_x.size()) {
                throw std::runtime_error("BilinearInterpolation: z columns must match x size");
            }
        }
    }

protected:
    DoubleT valueImpl(DoubleT x, DoubleT y) const override {
        size_t i = locateX(x);
        size_t j = locateY(y);

        DoubleT x1 = m_x[i];
        DoubleT x2 = m_x[i + 1];
        DoubleT y1 = m_y[j];
        DoubleT y2 = m_y[j + 1];

        DoubleT q11 = m_z[j][i];
        DoubleT q12 = m_z[j + 1][i];
        DoubleT q21 = m_z[j][i + 1];
        DoubleT q22 = m_z[j + 1][i + 1];

        DoubleT r1 = ((x2 - x) / (x2 - x1)) * q11 + ((x - x1) / (x2 - x1)) * q21;
        DoubleT r2 = ((x2 - x) / (x2 - x1)) * q12 + ((x - x1) / (x2 - x1)) * q22;

        return ((y2 - y) / (y2 - y1)) * r1 + ((y - y1) / (y2 - y1)) * r2;
    }

    bool isInRange(DoubleT x, DoubleT y) const override {
        // implementation
        return true;
    }

private:
    std::vector<DoubleT> m_x, m_y;
    std::vector<std::vector<DoubleT>> m_z;

    size_t locateX(DoubleT x) const {
        // simplified search
        for (size_t i = 0; i < m_x.size() - 1; ++i) {
            if (x >= m_x[i] && x <= m_x[i + 1]) {
                return i;
            }
        }
        return m_x.size() - 2;
    }

    size_t locateY(DoubleT y) const {
        // simplified search
        for (size_t i = 0; i < m_y.size() - 1; ++i) {
            if (y >= m_y[i] && y <= m_y[i + 1]) {
                return i;
            }
        }
        return m_y.size() - 2;
    }
};

} // namespace Math

#endif // BILINEAR_INTERPOLATION_H
