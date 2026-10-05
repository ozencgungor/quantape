#ifndef QUANTAPE_MATH_MONOTONE_CUBIC_INTERPOLATION_H
#define QUANTAPE_MATH_MONOTONE_CUBIC_INTERPOLATION_H

#include "quantape/math/Autodiff/PrimalExtraction.h"
#include "quantape/math/Interpolations/Interpolation.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <type_traits>
#include <vector>

namespace quantape::math {
/**
 * @file MonotoneCubicInterpolation.h
 * @brief Shape-preserving cubic Hermite interpolation (Steffen, 1990)
 *
 * Slopes follow Steffen's local monotonicity limiter (a refined
 * Fritsch-Carlson rule), so the interpolant never overshoots the data in any
 * interval where the data is monotone. It is a local cubic Hermite scheme
 * (C1, stencil up to 4 nodes) widely used for forward-curve construction.
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
class MonotoneCubicInterpolation
    : public Interpolation<DoubleT, MonotoneCubicInterpolation<DoubleT>> {
    using Base = Interpolation<DoubleT, MonotoneCubicInterpolation<DoubleT>>;
    friend Base;

public:
    template <typename ContainerX, typename ContainerY>
    MonotoneCubicInterpolation(const ContainerX& x, const ContainerY& y) {
        this->m_x = this->toDoubleVector(x);
        this->m_y = this->toVector(y);
        this->validate();
        buildSlopes();
    }

    DoubleT valueImpl(DoubleT x) const { return evaluate(toDouble(x)); }

    DoubleT derivativeImpl(DoubleT x) const { return derivativeAt(toDouble(x)); }

    /// Node slopes (double constants along the pinned branch).
    const std::vector<double>& slopes() const { return m_slopes; }

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
            throw std::invalid_argument("MonotoneCubicInterpolation: need at least two points");
        }
        std::vector<double> secant(n - 1);
        for (std::size_t i = 0; i + 1 < n; ++i) {
            secant[i] = (primalAt(i + 1) - primalAt(i)) / (this->m_x[i + 1] - this->m_x[i]);
        }
        m_slopes.assign(n, 0.0);
        if (n == 2) {
            m_slopes[0] = secant[0];
            m_slopes[1] = secant[0];
            return;
        }
        for (std::size_t i = 1; i + 1 < n; ++i) {
            const double hLeft = this->m_x[i] - this->m_x[i - 1];
            const double hRight = this->m_x[i + 1] - this->m_x[i];
            const double p = (secant[i - 1] * hRight + secant[i] * hLeft) / (hLeft + hRight);
            const double sMin = std::min(std::abs(secant[i - 1]), std::abs(secant[i]));
            const double sign = (secant[i - 1] > 0.0 && secant[i] > 0.0)
                                    ? 1.0
                                    : ((secant[i - 1] < 0.0 && secant[i] < 0.0) ? -1.0 : 0.0);
            m_slopes[i] = sign * std::min(sMin, 0.5 * std::abs(p));
        }
        // Fritsch-Carlson one-sided endpoint rule.
        const auto clampEndpoint = [](double d, double s) {
            if (d * s <= 0.0) {
                return 0.0;
            }
            if (std::abs(d) > 3.0 * std::abs(s)) {
                return 3.0 * s;
            }
            return d;
        };
        {
            const double h0 = this->m_x[1] - this->m_x[0];
            const double h1 = this->m_x[2] - this->m_x[1];
            const double d = ((2.0 * h0 + h1) * secant[0] - h0 * secant[1]) / (h0 + h1);
            m_slopes[0] = clampEndpoint(d, secant[0]);
        }
        {
            const std::size_t last = n - 1;
            const double hm = this->m_x[last] - this->m_x[last - 1];
            const double hm1 = this->m_x[last - 1] - this->m_x[last - 2];
            const double d =
                ((2.0 * hm + hm1) * secant[last - 1] - hm * secant[last - 2]) / (hm + hm1);
            m_slopes[last] = clampEndpoint(d, secant[last - 1]);
        }
    }

    std::size_t segment(double x) const {
        const std::size_t n = this->m_x.size();
        if (!std::isfinite(x)) {
            throw std::invalid_argument(
                "MonotoneCubicInterpolation::segment: non-finite query time");
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
};

} // namespace quantape::math

#endif // QUANTAPE_MATH_MONOTONE_CUBIC_INTERPOLATION_H
