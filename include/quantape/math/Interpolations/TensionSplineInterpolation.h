#ifndef QUANTAPE_MATH_TENSION_SPLINE_INTERPOLATION_H
#define QUANTAPE_MATH_TENSION_SPLINE_INTERPOLATION_H

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
 * @file TensionSplineInterpolation.h
 * @brief Exponential tension spline (Andersen): C2 with controllable locality
 *
 * On each segment the interpolant satisfies `S'''' = sigma^2 S''` and is
 * written in terms of the node second derivatives `M_i`:
 *
 *   S(t) = M_i  sinh(sigma (x_{i+1}-t)) / (sigma^2 sinh(p_i))
 *        + M_{i+1} sinh(sigma (t-x_i))    / (sigma^2 sinh(p_i))
 *        + (y_i - M_i/sigma^2)     (x_{i+1}-t)/h_i
 *        + (y_{i+1} - M_{i+1}/sigma^2) (t-x_i)/h_i
 *
 * with `p_i = sigma h_i`. Natural boundary conditions (`M_0 = M_{n-1} = 0`);
 * C1 continuity gives a tridiagonal system solved with `TridiagonalSolver`.
 * As `sigma -> 0` the scheme tends to the natural cubic spline; growing sigma
 * localizes the influence of nodes (bandwidth measured in the tests).
 *
 * The abscissa is passive: `valueImpl`/`derivativeImpl` convert `x` to double
 * and never push an adjoint through it (curve evaluation treats time as data).
 * The interpolant is linear in the node values, so exact node weights can be
 * assembled by the risk chain when needed.
 *
 * @tparam DoubleT Numeric type (`double` or an AD scalar).
 */
template <typename DoubleT>
class TensionSplineInterpolation
    : public Interpolation<DoubleT, TensionSplineInterpolation<DoubleT>> {
    using Base = Interpolation<DoubleT, TensionSplineInterpolation<DoubleT>>;
    friend Base;

public:
    template <typename ContainerX, typename ContainerY>
    TensionSplineInterpolation(const ContainerX& x, const ContainerY& y, double tension) {
        if (!(tension > 0.0)) {
            throw std::invalid_argument("TensionSplineInterpolation: tension must be positive");
        }
        this->m_x = this->toDoubleVector(x);
        this->m_y = this->toVector(y);
        this->validate();
        m_sigma = tension;
        build();
    }

    /// Interpolated value (abscissa passive).
    DoubleT valueImpl(DoubleT x) const { return valueAt(toDouble(x)); }

    /// First derivative (abscissa passive). Flat beyond the last node.
    DoubleT derivativeImpl(DoubleT x) const { return derivativeAt(toDouble(x)); }

    /// Second derivative (the tension-spline `M(t)`); for diagnostics.
    DoubleT secondDerivativeImpl(DoubleT x) const { return secondDerivativeAt(toDouble(x)); }

    /// Tension parameter.
    double tension() const { return m_sigma; }

    /// Node second derivatives `M_i` (linear in the node values).
    const std::vector<DoubleT>& secondDerivatives() const { return m_m; }

    /// Reduced tridiagonal system for the `M` values (size `n - 2`); exposed
    /// so curve layers can assemble exact node weights for the risk chain.
    const std::vector<double>& systemSubDiagonal() const { return m_systemSub; }
    const std::vector<double>& systemDiagonal() const { return m_systemDiag; }
    const std::vector<double>& systemSuperDiagonal() const { return m_systemSuper; }

private:
    static double toDouble(const DoubleT& x) {
        if constexpr (std::is_same_v<DoubleT, double>) {
            return x;
        } else {
            return quantape::math::detail::primalValue(x);
        }
    }

    static double safeRatio(double numerator, double denominator) {
        return denominator == 0.0 ? 0.0 : numerator / denominator;
    }

    void build() {
        const std::size_t n = this->m_x.size();
        m_h.resize(n - 1);
        m_alpha.resize(n - 1);
        m_beta.resize(n - 1);
        m_a.resize(n - 1);
        m_c.resize(n - 1);
        for (std::size_t j = 0; j + 1 < n; ++j) {
            const double h = this->m_x[j + 1] - this->m_x[j];
            m_h[j] = h;
            const double p = m_sigma * h;
            const double sinhP = std::sinh(p);
            if (p < 1e-6) {
                // Series expansions keep the small-tension limit stable.
                m_alpha[j] = -(1.0 / 6.0 - 7.0 * p * p / 360.0);
                m_beta[j] = 1.0 / 3.0 - p * p / 45.0;
                m_a[j] = h * (1.0 / 6.0 - 7.0 * p * p / 360.0);
            } else {
                m_alpha[j] = (1.0 / (p * p)) * (p / sinhP - 1.0);
                m_beta[j] = (1.0 / (p * p)) * (p / std::tanh(p) - 1.0);
                m_a[j] = h * (1.0 / (p * p)) * (1.0 - p / sinhP);
            }
            m_c[j] = h * m_beta[j];
        }

        m_m.assign(n, DoubleT(0));
        if (n <= 2) {
            m_firstSlope = (this->m_y[1] - this->m_y[0]) / m_h[0];
            m_lastSlope = m_firstSlope;
            return;
        }
        const std::size_t interior = n - 2;
        m_systemSub.assign(interior, 0.0);
        m_systemDiag.assign(interior, 0.0);
        m_systemSuper.assign(interior, 0.0);
        std::vector<DoubleT> rhs(interior);
        for (std::size_t i = 0; i < interior; ++i) {
            const std::size_t k = i + 1; // node index
            m_systemSub[i] = interior > 1 ? m_a[k - 1] : 0.0;
            m_systemDiag[i] = m_c[k - 1] + m_c[k];
            m_systemSuper[i] = interior > 1 ? m_a[k] : 0.0;
            rhs[i] = (this->m_y[k + 1] - this->m_y[k]) / m_h[k] -
                     (this->m_y[k] - this->m_y[k - 1]) / m_h[k - 1];
        }
        const std::vector<DoubleT> solution =
            TridiagonalSolver<DoubleT>::solve(m_systemSub, m_systemDiag, m_systemSuper, rhs);
        for (std::size_t i = 0; i < interior; ++i) {
            m_m[i + 1] = solution[i];
        }
        m_firstSlope = (this->m_y[1] - this->m_y[0]) / m_h[0] + (m_a[0] / m_h[0]) * m_m[1];
        const std::size_t last = n - 2;
        m_lastSlope = (this->m_y[last + 1] - this->m_y[last]) / m_h[last] +
                      (m_a[last] / m_h[last]) * m_m[last];
    }

    std::size_t segment(double x) const {
        const std::size_t n = this->m_x.size();
        if (!std::isfinite(x)) {
            throw std::invalid_argument(
                "TensionSplineInterpolation::segment: non-finite query time");
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

    DoubleT valueAt(double x) const {
        const std::size_t j = segment(x);
        const double h = m_h[j];
        const double p = m_sigma * h;
        const double sinhP = std::sinh(p);
        const double xLeft = this->m_x[j];
        const double xRight = this->m_x[j + 1];
        const double right = xRight - x;
        const double left = x - xLeft;
        const double invLambda = safeRatio(1.0, m_sigma * m_sigma * sinhP);
        return m_m[j] * (invLambda * std::sinh(m_sigma * right)) +
               m_m[j + 1] * (invLambda * std::sinh(m_sigma * left)) +
               (this->m_y[j] - m_m[j] / (m_sigma * m_sigma)) * (right / h) +
               (this->m_y[j + 1] - m_m[j + 1] / (m_sigma * m_sigma)) * (left / h);
    }

    DoubleT derivativeAt(double x) const {
        const std::size_t j = segment(x);
        const double h = m_h[j];
        const double p = m_sigma * h;
        const double sinhP = std::sinh(p);
        const double xLeft = this->m_x[j];
        const double xRight = this->m_x[j + 1];
        const double invLambda = safeRatio(1.0, m_sigma * m_sigma * sinhP);
        const double right = xRight - x;
        const double left = x - xLeft;
        return m_m[j] * (-invLambda * m_sigma * std::cosh(m_sigma * right)) +
               m_m[j + 1] * (invLambda * m_sigma * std::cosh(m_sigma * left)) - this->m_y[j] / h +
               m_m[j] / (m_sigma * m_sigma * h) + this->m_y[j + 1] / h -
               m_m[j + 1] / (m_sigma * m_sigma * h);
    }

    DoubleT secondDerivativeAt(double x) const {
        const std::size_t j = segment(x);
        const double h = m_h[j];
        const double p = m_sigma * h;
        const double sinhP = std::sinh(p);
        const double right = this->m_x[j + 1] - x;
        const double left = x - this->m_x[j];
        const double invSinh = safeRatio(1.0, sinhP);
        return m_m[j] * (invSinh * std::sinh(m_sigma * right)) +
               m_m[j + 1] * (invSinh * std::sinh(m_sigma * left));
    }

    double m_sigma = 0.0;
    DoubleT m_firstSlope{};
    DoubleT m_lastSlope{};
    std::vector<double> m_h;
    std::vector<double> m_alpha;
    std::vector<double> m_beta;
    std::vector<double> m_a;
    std::vector<double> m_c;
    std::vector<double> m_systemSub;
    std::vector<double> m_systemDiag;
    std::vector<double> m_systemSuper;
    std::vector<DoubleT> m_m;
};

} // namespace quantape::math

#endif // QUANTAPE_MATH_TENSION_SPLINE_INTERPOLATION_H
