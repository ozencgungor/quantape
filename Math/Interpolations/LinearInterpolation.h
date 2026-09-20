#ifndef LINEAR_INTERPOLATION_H
#define LINEAR_INTERPOLATION_H

#include "Interpolation.h"

namespace Math {

/**
 * @brief Linear interpolation with AD-optimized evaluation
 *
 * For a query point x in segment [x_i, x_{i+1}]:
 *   f(x) = w0 * y[i] + w1 * y[i+1]
 * where w0, w1 are double weights (depend only on grid + query point).
 *
 * AD properties:
 *   - df/dy[i] = w0, df/dy[i+1] = w1 (double constants)
 *   - d2f/dy[j]dy[k] = 0 (linear in node values, Hessian is zero)
 *   - Generic path: 3 tape nodes (2 multiplies + 1 add)
 *   - var specialization: 1 tape node via make_callback_var
 *   - fvar<var> specialization: 1 callback var, zero Hessian
 *
 * var and fvar<var> specializations are in InterpolationStanPrimitives.h.
 */
template <typename DoubleT>
class LinearInterpolation : public Interpolation<DoubleT, LinearInterpolation<DoubleT>> {
    using Base = Interpolation<DoubleT, LinearInterpolation<DoubleT>>;
    friend Base;

public:
    template <typename ContainerX, typename ContainerY>
    LinearInterpolation(const ContainerX& x, const ContainerY& y) {
        this->m_x = this->toDoubleVector(x);
        this->m_y = this->toVector(y);
        this->validate();
    }

    // valueImpl and derivativeImpl: primary template works for all DoubleT.
    // Explicit specializations for var and fvar<var> in InterpolationStanPrimitives.h
    // reduce tape nodes from 3 to 1.

    DoubleT valueImpl(DoubleT x) const {
        size_t i = this->locate(x);

        double x1 = this->m_x[i];
        double x2 = this->m_x[i + 1];
        double xv = this->extractDouble(x);

        double inv_dx = 1.0 / (x2 - x1);
        double w0 = (x2 - xv) * inv_dx;
        double w1 = (xv - x1) * inv_dx;

        return w0 * this->m_y[i] + w1 * this->m_y[i + 1];
    }

    DoubleT derivativeImpl(DoubleT x) const {
        size_t i = this->locate(x);
        double inv_dx = 1.0 / (this->m_x[i + 1] - this->m_x[i]);

        return (this->m_y[i + 1] - this->m_y[i]) * inv_dx;
    }
};

} // namespace Math

#endif // LINEAR_INTERPOLATION_H
