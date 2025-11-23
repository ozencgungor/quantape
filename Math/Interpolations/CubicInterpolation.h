#ifndef CUBIC_INTERPOLATION_H
#define CUBIC_INTERPOLATION_H

#include "Interpolation.h"
#include <vector>
#include <stdexcept>
#include <cmath>
#include <algorithm>

namespace Math {

/**
 * @brief Cubic interpolation with various derivative approximation schemes
 *
 * Implements cubic interpolation following QuantLib's approach with multiple
 * derivative approximation methods. The polynomial form for each segment i is:
 *   P[i](x) = y[i] + a[i]*(x-x[i]) + b[i]*(x-x[i])^2 + c[i]*(x-x[i])^3
 *
 * Where a[i] is the first derivative at x[i], computed using one of several schemes:
 * - Spline: C^2 continuous natural cubic spline (DEFAULT, global method)
 * - Parabolic: Local parabolic approximation
 * - Akima: Akima's method (weights based on adjacent slopes)
 * - Kruger: Harmonic mean of adjacent slopes
 * - Harmonic: Weighted harmonic mean
 *
 * @tparam DoubleT Numeric type (double or stan::math::var)
 */
template<typename DoubleT>
class CubicInterpolation : public Interpolation<DoubleT> {
public:
    enum DerivativeApprox {
        Spline,         // Natural cubic spline (C^2 continuous, DEFAULT)
        Parabolic,      // Local parabolic approximation
        Akima,          // Akima's method
        Kruger,         // Kruger's harmonic mean method
        Harmonic        // Weighted harmonic mean
    };

    /**
     * @brief Construct cubic interpolation with specified derivative scheme
     * @param x X coordinates (must be sorted) - accepts std::vector, Eigen::Vector, etc.
     * @param y Y coordinates - accepts std::vector, Eigen::Vector, etc.
     * @param da Derivative approximation method (default: Spline)
     */
    template<typename ContainerX, typename ContainerY>
    CubicInterpolation(const ContainerX& x,
                      const ContainerY& y,
                      DerivativeApprox da = Spline)
        : m_da(da)
    {
        this->m_x = this->toVector(x);
        this->m_y = this->toVector(y);
        this->validate();
        calculateCoefficients();
    }

protected:
    DoubleT valueImpl(DoubleT x) const override {
        size_t i = this->locate(x);
        if (i >= m_a.size()) i = m_a.size() - 1;

        DoubleT dx = x - this->m_x[i];
        // P[i](x) = y[i] + a[i]*dx + b[i]*dx^2 + c[i]*dx^3
        return this->m_y[i] + dx * (m_a[i] + dx * (m_b[i] + dx * m_c[i]));
    }

    DoubleT derivativeImpl(DoubleT x) const override {
        size_t i = this->locate(x);
        if (i >= m_a.size()) i = m_a.size() - 1;

        DoubleT dx = x - this->m_x[i];
        // P'[i](x) = a[i] + 2*b[i]*dx + 3*c[i]*dx^2
        return m_a[i] + dx * (DoubleT(2.0) * m_b[i] + DoubleT(3.0) * m_c[i] * dx);
    }

private:
    DerivativeApprox m_da;
    std::vector<DoubleT> m_a;  ///< First derivative at x[i]
    std::vector<DoubleT> m_b;  ///< Coefficient for (x-x[i])^2
    std::vector<DoubleT> m_c;  ///< Coefficient for (x-x[i])^3

    /**
     * @brief Calculate cubic coefficients using local derivative schemes
     */
    void calculateCoefficients() {
        size_t n = this->m_x.size();

        if (n < 2) {
            throw std::runtime_error("CubicInterpolation: need at least 2 points");
        }

        // Special case: only 2 points, use linear interpolation
        if (n == 2) {
            m_a.resize(1);
            m_b.resize(1);
            m_c.resize(1);
            m_a[0] = (this->m_y[1] - this->m_y[0]) / (this->m_x[1] - this->m_x[0]);
            m_b[0] = DoubleT(0.0);
            m_c[0] = DoubleT(0.0);
            return;
        }

        // Compute segment lengths and slopes
        std::vector<DoubleT> dx(n - 1);
        std::vector<DoubleT> S(n - 1);  // Slopes of secant lines
        for (size_t i = 0; i < n - 1; ++i) {
            dx[i] = this->m_x[i+1] - this->m_x[i];
            S[i] = (this->m_y[i+1] - this->m_y[i]) / dx[i];
        }

        // Compute first derivatives at each knot
        std::vector<DoubleT> derivatives(n);

        switch (m_da) {
            case Spline:
                computeSplineDerivatives(derivatives, dx, S);
                break;
            case Parabolic:
                computeParabolicDerivatives(derivatives, dx, S);
                break;
            case Akima:
                computeAkimaDerivatives(derivatives, dx, S);
                break;
            case Kruger:
                computeKrugerDerivatives(derivatives, dx, S);
                break;
            case Harmonic:
                computeHarmonicDerivatives(derivatives, dx, S);
                break;
            default:
                throw std::runtime_error("CubicInterpolation: unknown derivative approximation");
        }

        // Compute cubic coefficients from derivatives
        m_a.resize(n - 1);
        m_b.resize(n - 1);
        m_c.resize(n - 1);

        for (size_t i = 0; i < n - 1; ++i) {
            m_a[i] = derivatives[i];
            m_b[i] = (DoubleT(3.0) * S[i] - derivatives[i+1] - DoubleT(2.0) * derivatives[i]) / dx[i];
            m_c[i] = (derivatives[i+1] + derivatives[i] - DoubleT(2.0) * S[i]) / (dx[i] * dx[i]);
        }
    }

    /**
     * @brief Solve tridiagonal system using Thomas algorithm
     */
    std::vector<DoubleT> solveTridiagonal(
        const std::vector<DoubleT>& lower,  // n-1 elements (subdiagonal)
        const std::vector<DoubleT>& diag,   // n elements (main diagonal)
        const std::vector<DoubleT>& upper,  // n-1 elements (superdiagonal)
        const std::vector<DoubleT>& rhs     // n elements (right-hand side)
    ) const {
        size_t n = rhs.size();
        if (n == 0) return {};
        if (n == 1) return {rhs[0] / diag[0]};

        std::vector<DoubleT> c_prime(n - 1);
        std::vector<DoubleT> d_prime(n);
        std::vector<DoubleT> x(n);

        // Forward elimination
        c_prime[0] = upper[0] / diag[0];
        d_prime[0] = rhs[0] / diag[0];

        for (size_t i = 1; i < n - 1; ++i) {
            DoubleT denom = diag[i] - lower[i-1] * c_prime[i-1];
            if (this->value(denom) == 0.0) {
                throw std::runtime_error("CubicInterpolation: singular matrix in tridiagonal solve");
            }
            c_prime[i] = upper[i] / denom;
            d_prime[i] = (rhs[i] - lower[i-1] * d_prime[i-1]) / denom;
        }

        // Last row
        DoubleT denom = diag[n-1] - lower[n-2] * c_prime[n-2];
        if (this->value(denom) == 0.0) {
            throw std::runtime_error("CubicInterpolation: singular matrix in tridiagonal solve");
        }
        d_prime[n-1] = (rhs[n-1] - lower[n-2] * d_prime[n-2]) / denom;

        // Back substitution
        x[n-1] = d_prime[n-1];
        for (int i = n - 2; i >= 0; --i) {
            x[i] = d_prime[i] - c_prime[i] * x[i+1];
        }

        return x;
    }

    /**
     * @brief Natural cubic spline derivative computation (QuantLib lines 401-472)
     *
     * Solves tridiagonal system for first derivatives with natural boundary
     * conditions (second derivative = 0 at endpoints).
     */
    void computeSplineDerivatives(std::vector<DoubleT>& deriv,
                                  const std::vector<DoubleT>& dx,
                                  const std::vector<DoubleT>& S) const {
        size_t n = this->m_x.size();

        // Set up tridiagonal system to solve for first derivatives
        std::vector<DoubleT> lower(n - 1);
        std::vector<DoubleT> diag(n);
        std::vector<DoubleT> upper(n - 1);
        std::vector<DoubleT> rhs(n);

        // Interior equations: continuity of second derivative
        for (size_t i = 1; i < n - 1; ++i) {
            lower[i-1] = dx[i-1];
            diag[i] = DoubleT(2.0) * (dx[i-1] + dx[i]);
            upper[i] = dx[i];
            rhs[i] = DoubleT(3.0) * (dx[i] * S[i-1] + dx[i-1] * S[i]);
        }

        // Natural boundary conditions: second derivative = 0 at boundaries
        // At x[0]: 2*b[0] = 0 => 2*derivative[0] + derivative[1] = 3*S[0]
        diag[0] = DoubleT(2.0);
        upper[0] = DoubleT(1.0);
        rhs[0] = DoubleT(3.0) * S[0];

        // At x[n-1]: 2*b[n-2] + 6*c[n-2]*dx[n-2] = 0
        // => derivative[n-2] + 2*derivative[n-1] = 3*S[n-2]
        lower[n-2] = DoubleT(1.0);
        diag[n-1] = DoubleT(2.0);
        rhs[n-1] = DoubleT(3.0) * S[n-2];

        // Solve for first derivatives at all knots
        deriv = solveTridiagonal(lower, diag, upper, rhs);
    }

    /**
     * @brief Parabolic derivative approximation (QuantLib lines 568-575)
     */
    void computeParabolicDerivatives(std::vector<DoubleT>& deriv,
                                    const std::vector<DoubleT>& dx,
                                    const std::vector<DoubleT>& S) const {
        size_t n = this->m_x.size();

        // Intermediate points: weighted average of adjacent slopes
        for (size_t i = 1; i < n - 1; ++i) {
            deriv[i] = (dx[i-1] * S[i] + dx[i] * S[i-1]) / (dx[i] + dx[i-1]);
        }

        // End points: extrapolation
        deriv[0] = ((DoubleT(2.0) * dx[0] + dx[1]) * S[0] - dx[0] * S[1]) / (dx[0] + dx[1]);
        deriv[n-1] = ((DoubleT(2.0) * dx[n-2] + dx[n-3]) * S[n-2] - dx[n-2] * S[n-3]) / (dx[n-2] + dx[n-3]);
    }

    /**
     * @brief Akima derivative approximation (QuantLib lines 596-613)
     */
    void computeAkimaDerivatives(std::vector<DoubleT>& deriv,
                                const std::vector<DoubleT>& dx,
                                const std::vector<DoubleT>& S) const {
        size_t n = this->m_x.size();

        // First point (special formula)
        DoubleT w1 = abs_impl(S[1] - S[0]);
        DoubleT w2 = abs_impl(DoubleT(2.0) * S[0] * S[1] - DoubleT(4.0) * S[0] * S[0] * S[1]);
        if (this->value(w1 + w2) == 0.0) {
            deriv[0] = S[0];
        } else {
            deriv[0] = (w1 * DoubleT(2.0) * S[0] * S[1] + w2 * S[0]) / (w1 + w2);
        }

        // Second point
        w1 = abs_impl(S[2] - S[1]);
        w2 = abs_impl(S[0] - DoubleT(2.0) * S[0] * S[1]);
        if (this->value(w1 + w2) == 0.0) {
            deriv[1] = S[1];
        } else {
            deriv[1] = (w1 * S[0] + w2 * S[1]) / (w1 + w2);
        }

        // Interior points
        for (size_t i = 2; i < n - 2; ++i) {
            // Handle special cases where slopes are equal
            if ((this->value(S[i-2]) == this->value(S[i-1])) && (this->value(S[i]) != this->value(S[i+1]))) {
                deriv[i] = S[i-1];
            } else if ((this->value(S[i-2]) != this->value(S[i-1])) && (this->value(S[i]) == this->value(S[i+1]))) {
                deriv[i] = S[i];
            } else if (this->value(S[i]) == this->value(S[i-1])) {
                deriv[i] = S[i];
            } else if ((this->value(S[i-2]) == this->value(S[i-1])) &&
                      (this->value(S[i-1]) != this->value(S[i])) &&
                      (this->value(S[i]) == this->value(S[i+1]))) {
                deriv[i] = (S[i-1] + S[i]) / DoubleT(2.0);
            } else {
                w1 = abs_impl(S[i+1] - S[i]);
                w2 = abs_impl(S[i-1] - S[i-2]);
                if (this->value(w1 + w2) == 0.0) {
                    deriv[i] = (S[i-1] + S[i]) / DoubleT(2.0);
                } else {
                    deriv[i] = (w1 * S[i-1] + w2 * S[i]) / (w1 + w2);
                }
            }
        }

        // Second-to-last point
        w1 = abs_impl(DoubleT(2.0) * S[n-2] * S[n-3] - S[n-2]);
        w2 = abs_impl(S[n-3] - S[n-4]);
        if (this->value(w1 + w2) == 0.0) {
            deriv[n-2] = S[n-2];
        } else {
            deriv[n-2] = (w1 * S[n-3] + w2 * S[n-2]) / (w1 + w2);
        }

        // Last point
        w1 = abs_impl(DoubleT(4.0) * S[n-2] * S[n-2] * S[n-3] - DoubleT(2.0) * S[n-2] * S[n-3]);
        w2 = abs_impl(S[n-2] - S[n-3]);
        if (this->value(w1 + w2) == 0.0) {
            deriv[n-1] = S[n-2];
        } else {
            deriv[n-1] = (w1 * S[n-2] + w2 * DoubleT(2.0) * S[n-2] * S[n-3]) / (w1 + w2);
        }
    }

    /**
     * @brief Kruger derivative approximation (QuantLib lines 614-629)
     */
    void computeKrugerDerivatives(std::vector<DoubleT>& deriv,
                                 const std::vector<DoubleT>& dx,
                                 const std::vector<DoubleT>& S) const {
        size_t n = this->m_x.size();

        // Intermediate points: harmonic mean if same sign, 0 if opposite sign
        for (size_t i = 1; i < n - 1; ++i) {
            if (this->value(S[i-1] * S[i]) < 0.0) {
                // Slope changes sign at point
                deriv[i] = DoubleT(0.0);
            } else {
                // Harmonic mean: 2/(1/a + 1/b)
                deriv[i] = DoubleT(2.0) / (DoubleT(1.0) / S[i-1] + DoubleT(1.0) / S[i]);
            }
        }

        // End points: extrapolation
        deriv[0] = (DoubleT(3.0) * S[0] - deriv[1]) / DoubleT(2.0);
        deriv[n-1] = (DoubleT(3.0) * S[n-2] - deriv[n-2]) / DoubleT(2.0);
    }

    /**
     * @brief Harmonic derivative approximation (QuantLib lines 630-663)
     */
    void computeHarmonicDerivatives(std::vector<DoubleT>& deriv,
                                   const std::vector<DoubleT>& dx,
                                   const std::vector<DoubleT>& S) const {
        size_t n = this->m_x.size();

        // Intermediate points: weighted harmonic mean
        for (size_t i = 1; i < n - 1; ++i) {
            DoubleT w1 = DoubleT(2.0) * dx[i] + dx[i-1];
            DoubleT w2 = dx[i] + DoubleT(2.0) * dx[i-1];

            if (this->value(S[i-1] * S[i]) <= 0.0) {
                // Slope changes sign at point
                deriv[i] = DoubleT(0.0);
            } else {
                // Weighted harmonic mean
                deriv[i] = (w1 + w2) / (w1 / S[i-1] + w2 / S[i]);
            }
        }

        // Left endpoint
        deriv[0] = ((DoubleT(2.0) * dx[0] + dx[1]) * S[0] - dx[0] * S[1]) / (dx[1] + dx[0]);
        if (this->value(deriv[0] * S[0]) < 0.0) {
            deriv[0] = DoubleT(0.0);
        } else if (this->value(S[0] * S[1]) < 0.0) {
            if (abs_impl(deriv[0]) > abs_impl(DoubleT(3.0) * S[0])) {
                deriv[0] = DoubleT(3.0) * S[0];
            }
        }

        // Right endpoint
        deriv[n-1] = ((DoubleT(2.0) * dx[n-2] + dx[n-3]) * S[n-2] - dx[n-2] * S[n-3]) / (dx[n-3] + dx[n-2]);
        if (this->value(deriv[n-1] * S[n-2]) < 0.0) {
            deriv[n-1] = DoubleT(0.0);
        } else if (this->value(S[n-2] * S[n-3]) < 0.0) {
            if (abs_impl(deriv[n-1]) > abs_impl(DoubleT(3.0) * S[n-2])) {
                deriv[n-1] = DoubleT(3.0) * S[n-2];
            }
        }
    }

    /**
     * @brief AD-compatible absolute value
     */
    static DoubleT abs_impl(const DoubleT& x) {
        if constexpr (std::is_same_v<DoubleT, double>) {
            return std::abs(x);
        } else {
            using std::abs;
            return abs(x);  // ADL will find stan::math::abs
        }
    }
};

} // namespace Math

#endif // CUBIC_INTERPOLATION_H
