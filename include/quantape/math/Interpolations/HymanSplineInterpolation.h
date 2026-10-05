#ifndef QUANTAPE_MATH_HYMAN_SPLINE_INTERPOLATION_H
#define QUANTAPE_MATH_HYMAN_SPLINE_INTERPOLATION_H

#include "quantape/math/Autodiff/PrimalExtraction.h"
#include "quantape/math/Interpolations/Interpolation.h"
#include "quantape/math/Solvers/TridiagonalSolver.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <type_traits>
#include <vector>

namespace quantape::math {
/**
 * @file HymanSplineInterpolation.h
 * @brief Natural cubic spline with Hyman's monotonicity filter
 *
 * Node second derivatives solve the natural cubic spline tridiagonal system
 * (`M_0 = M_{n-1} = 0`,
 * `h_{i-1} M_{i-1} + 2(h_{i-1} + h_i) M_i + h_i M_{i+1} = 6(s_i - s_{i-1})`).
 * The resulting node derivatives are then filtered with Hyman's rule: a slope
 * is pinned to zero at a sign change of the adjacent secants, otherwise it is
 * the smallest of `|d_i|`, `3|s_{i-1}|` and `3|s_i|` with the secant's sign.
 * Evaluation uses the cubic Hermite basis, so the interpolant is C1 and
 * monotone in the data range wherever the data is monotone.
 *
 * The node slopes are computed from the primal values and stored as constants;
 * the interpolant is therefore linear in the node values along the pinned
 * branch (AD flows to the nodes only). The curve marks this scheme as
 * pricing-only: exact branch-pinned node weights for the risk chain are
 * unavailable.
 *
 * @tparam DoubleT Numeric type (`double` or an AD scalar).
 */
template <typename DoubleT>
class HymanSplineInterpolation
    : public Interpolation<DoubleT, HymanSplineInterpolation<DoubleT>> {
    using Base = Interpolation<DoubleT, HymanSplineInterpolation<DoubleT>>;
    friend Base;

public:
    template <typename ContainerX, typename ContainerY>
    HymanSplineInterpolation(const ContainerX& x, const ContainerY& y) {
        this->m_x = this->toDoubleVector(x);
        this->m_y = this->toVector(y);
        this->validate();
        buildSlopes();
    }

    DoubleT valueImpl(DoubleT x) const { return evaluate(toDouble(x)); }

    DoubleT derivativeImpl(DoubleT x) const { return derivativeAt(toDouble(x)); }

    /// Node slopes after the Hyman filter (double constants along the pinned
    /// branch).
    const std::vector<double>& slopes() const { return m_slopes; }

    /// Node second derivatives of the unfiltered natural spline.
    const std::vector<double>& secondDerivatives() const { return m_secondDerivatives; }

private:
    static double toDouble(const DoubleT& x) {
        if constexpr (std::is_same_v<DoubleT, double>) {
            return x;
        } else {
            // Recursive primal extraction: handles var and `fvar<var>`.
            return quantape::math::detail::primalValue(x);
        }
    }

    double primalAt(std::size_t i) const { return toDouble(this->m_y[i]); }

    void buildSlopes() {
        const std::size_t n = this->m_x.size();
        if (n < 2) {
            throw std::invalid_argument("HymanSplineInterpolation: need at least two points");
        }
        std::vector<double> secant(n - 1);
        for (std::size_t i = 0; i + 1 < n; ++i) {
            secant[i] = (primalAt(i + 1) - primalAt(i)) / (this->m_x[i + 1] - this->m_x[i]);
        }
        m_slopes.assign(n, 0.0);
        m_secondDerivatives.assign(n, 0.0);
        if (n == 2) {
            m_slopes[0] = secant[0];
            m_slopes[1] = secant[0];
            return;
        }
        // Natural spline: solve for the interior second derivatives.
        const std::size_t interior = n - 2;
        std::vector<double> sub(interior, 0.0);
        std::vector<double> diag(interior, 0.0);
        std::vector<double> super(interior, 0.0);
        std::vector<double> rhs(interior, 0.0);
        for (std::size_t k = 0; k < interior; ++k) {
            const double hLeft = this->m_x[k + 1] - this->m_x[k];
            const double hRight = this->m_x[k + 2] - this->m_x[k + 1];
            sub[k] = k > 0 ? hLeft : 0.0;
            diag[k] = 2.0 * (hLeft + hRight);
            super[k] = k + 1 < interior ? hRight : 0.0;
            rhs[k] = 6.0 * (secant[k + 1] - secant[k]);
        }
        const std::vector<double> M = TridiagonalSolver<double>::solve(sub, diag, super, rhs);
        for (std::size_t k = 0; k < interior; ++k) {
            m_secondDerivatives[k + 1] = M[k];
        }
        // Natural-spline node derivatives.
        std::vector<double> derivative(n, 0.0);
        derivative[0] =
            secant[0] - (this->m_x[1] - this->m_x[0]) * m_secondDerivatives[1] / 6.0;
        for (std::size_t i = 1; i + 1 < n; ++i) {
            const double hLeft = this->m_x[i] - this->m_x[i - 1];
            derivative[i] =
                secant[i - 1] +
                hLeft * (m_secondDerivatives[i - 1] + 2.0 * m_secondDerivatives[i]) / 6.0;
        }
        derivative[n - 1] =
            secant[n - 2] +
            (this->m_x[n - 1] - this->m_x[n - 2]) * m_secondDerivatives[n - 2] / 6.0;
        // Hyman filter: zero at a sign change, otherwise cap at three times
        // each adjacent secant.
        for (std::size_t i = 0; i < n; ++i) {
            const double before = i == 0 ? secant[0] : secant[i - 1];
            const double after = i == n - 1 ? secant[n - 2] : secant[i];
            if (before * after <= 0.0) {
                m_slopes[i] = 0.0;
            } else {
                const double magnitude = std::min(
                    std::abs(derivative[i]),
                    std::min(3.0 * std::abs(before), 3.0 * std::abs(after)));
                m_slopes[i] = std::copysign(magnitude, after);
            }
        }
    }

    std::size_t segment(double x) const {
        const std::size_t n = this->m_x.size();
        if (!std::isfinite(x)) {
            throw std::invalid_argument(
                "HymanSplineInterpolation::segment: non-finite query time");
        }
        if (x <= this->m_x.front()) {
            return 0;
        }
        if (x >= this->m_x.back()) {
            return n - 2;
        }
        const auto it = std::upper_bound(this->m_x.begin(), this->m_x.end(), x);
        return static_cast<std::size_t>(it - this->m_x.begin()) - 1;
    }

    DoubleT evaluate(double x) const {
        const std::size_t i = segment(x);
        const double h = this->m_x[i + 1] - this->m_x[i];
        const double u = (x - this->m_x[i]) / h;
        const double u2 = u * u;
        const double u3 = u2 * u;
        const double h00 = 2.0 * u3 - 3.0 * u2 + 1.0;
        const double h10 = u3 - 2.0 * u2 + u;
        const double h01 = -2.0 * u3 + 3.0 * u2;
        const double h11 = u3 - u2;
        return h00 * this->m_y[i] + h10 * h * m_slopes[i] + h01 * this->m_y[i + 1] +
               h11 * h * m_slopes[i + 1];
    }

    DoubleT derivativeAt(double x) const {
        const std::size_t i = segment(x);
        const double h = this->m_x[i + 1] - this->m_x[i];
        const double u = (x - this->m_x[i]) / h;
        const double u2 = u * u;
        const double d00 = 6.0 * u2 - 6.0 * u;
        const double d10 = 3.0 * u2 - 4.0 * u + 1.0;
        const double d01 = -6.0 * u2 + 6.0 * u;
        const double d11 = 3.0 * u2 - 2.0 * u;
        return (d00 * this->m_y[i] + d01 * this->m_y[i + 1]) / h + d10 * m_slopes[i] +
               d11 * m_slopes[i + 1];
    }

    std::vector<double> m_slopes;
    std::vector<double> m_secondDerivatives;
};

} // namespace quantape::math

#endif // QUANTAPE_MATH_HYMAN_SPLINE_INTERPOLATION_H
