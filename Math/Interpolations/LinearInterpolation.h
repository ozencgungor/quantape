#ifndef LINEAR_INTERPOLATION_H
#define LINEAR_INTERPOLATION_H

#include "Interpolation.h"

namespace Math {

    template<typename DoubleT>
    class LinearInterpolation : public Interpolation<DoubleT> {
    public:
        /**
         * @brief Construct linear interpolation
         * @param x X coordinates (accepts std::vector, Eigen::Vector, etc.)
         * @param y Y coordinates (accepts std::vector, Eigen::Vector, etc.)
         */
        template<typename ContainerX, typename ContainerY>
        LinearInterpolation(const ContainerX& x, const ContainerY& y) {
            this->m_x = this->toVector(x);
            this->m_y = this->toVector(y);
            this->validate();
        }

    private:
        DoubleT valueImpl(DoubleT x) const override {
            size_t i = this->locate(x);
            DoubleT x1 = this->m_x[i];
            DoubleT y1 = this->m_y[i];
            DoubleT x2 = this->m_x[i+1];
            DoubleT y2 = this->m_y[i+1];

            return y1 + (y2 - y1) * (x - x1) / (x2 - x1);
        }

        DoubleT derivativeImpl(DoubleT x) const override {
            size_t i = this->locate(x);
            DoubleT x1 = this->m_x[i];
            DoubleT y1 = this->m_y[i];
            DoubleT x2 = this->m_x[i+1];
            DoubleT y2 = this->m_y[i+1];
            
            return (y2 - y1) / (x2 - x1);
        }
    };

} // namespace Math

#endif // LINEAR_INTERPOLATION_H
