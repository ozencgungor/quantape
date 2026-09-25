#ifndef LINEAR_INTERPOLATION_H
#define LINEAR_INTERPOLATION_H

#include "Interpolation.h"

namespace quantape::math {

/**
 * @brief Linear interpolation with AD-optimized evaluation
 *
 * For a query point x in segment [x_i, x_{i+1}]:
 *   f(x) = w0 * y[i] + w1 * y[i+1]
 * with w0 = (x_{i+1} - x)/(x_{i+1} - x_i), w1 = 1 - w0.
 *
 * AD properties (default evaluation):
 *   - weights are DoubleT: df/dx = (y[i+1] - y[i])/(x_{i+1} - x_i) (segment
 *     slope), df/dy[i] = w0, df/dy[i+1] = w1
 *   - mixed d2f/dx dy[i] = -inv_dx, d2f/dx dy[i+1] = +inv_dx
 *   - d2f/dy_j dy_k = 0 (linear in node values)
 *
 * evaluateFixed()/derivativeFixed() use the node-minimal callback paths in
 * InterpolationStanPrimitives.h (1 tape node, x treated as passive).
 *
 * Segment selection stays primal-pinned (subgradient convention at knots).
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

    /// Default (AD-aware) evaluation: the query coordinate is on the tape.
    DoubleT valueImpl(DoubleT x) const {
        size_t i = this->locate(x);

        double x1 = this->m_x[i];
        double x2 = this->m_x[i + 1];
        double inv_dx = 1.0 / (x2 - x1);

        // Weights are DoubleT — x contributes df/dx = slope, and the mixed
        // d2f/dxdy block (the grid positions stay double).
        DoubleT w0 = (DoubleT(x2) - x) * inv_dx;
        DoubleT w1 = (x - DoubleT(x1)) * inv_dx;

        return w0 * this->m_y[i] + w1 * this->m_y[i + 1];
    }

    DoubleT derivativeImpl(DoubleT x) const {
        size_t i = this->locate(x);
        double inv_dx = 1.0 / (this->m_x[i + 1] - this->m_x[i]);

        return (this->m_y[i + 1] - this->m_y[i]) * inv_dx;
    }

    /// Passive-abscissa policy (fast path): x adjoint intentionally pushed
    /// only by the var/fvar<var> specializations in InterpolationStanPrimitives.h.
    DoubleT valueFixedImpl(DoubleT x) const { return valueImpl(x); }

    DoubleT derivativeFixedImpl(DoubleT x) const { return derivativeImpl(x); }
};

} // namespace quantape::math

#endif // LINEAR_INTERPOLATION_H
