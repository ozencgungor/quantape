#ifndef CUBIC_INTERPOLATION_H
#define CUBIC_INTERPOLATION_H

#include <cmath>
#include <stdexcept>
#include <vector>

#include "Interpolation.h"

namespace Math {

/**
 * @brief Cubic interpolation with various derivative approximation schemes
 *
 * Implements cubic interpolation following QuantLib's approach with multiple
 * derivative approximation methods. The polynomial form for each segment i is:
 *   P[i](x) = y[i] + a[i]*(x-x[i]) + b[i]*(x-x[i])^2 + c[i]*(x-x[i])^3
 *
 * Grid coordinates (m_x) are double. Coefficients m_a, m_b, m_c are DoubleT
 * since they depend on node values m_y through the derivative computation.
 *
 * For the Spline scheme, the tridiagonal matrix is pure double (depends only
 * on grid spacing), while the RHS is DoubleT. This keeps matrix operations
 * off the AD tape — only the RHS assembly and back-substitution touch m_y.
 *
 * var and fvar<var> specializations are in InterpolationStanPrimitives.h.
 *
 * @tparam DoubleT Numeric type (double, stan::math::var, stan::math::fvar<var>)
 */
template <typename DoubleT>
class CubicInterpolation : public Interpolation<DoubleT, CubicInterpolation<DoubleT>> {
    using Base = Interpolation<DoubleT, CubicInterpolation<DoubleT>>;
    friend Base;

public:
    enum DerivativeApprox {
        Spline,    // Natural cubic spline (C^2 continuous, DEFAULT)
        Parabolic, // Local parabolic approximation
        Akima,     // Akima's method
        Kruger,    // Kruger's harmonic mean method
        Harmonic   // Weighted harmonic mean
    };

    template <typename ContainerX, typename ContainerY>
    CubicInterpolation(const ContainerX& x, const ContainerY& y, DerivativeApprox da = Spline)
        : m_da(da) {
        this->m_x = this->toDoubleVector(x);
        this->m_y = this->toVector(y);
        this->validate();
        calculateCoefficients();
    }

    DoubleT valueImpl(DoubleT x) const {
        size_t i = this->locate(x);
        if (i >= m_a.size())
            i = m_a.size() - 1;

        double dx = this->extractDouble(x) - this->m_x[i];
        // P[i](x) = y[i] + a[i]*dx + b[i]*dx^2 + c[i]*dx^3
        // dx is double — only y[i], a[i], b[i], c[i] are on tape
        return this->m_y[i] + dx * (m_a[i] + dx * (m_b[i] + dx * m_c[i]));
    }

    DoubleT derivativeImpl(DoubleT x) const {
        size_t i = this->locate(x);
        if (i >= m_a.size())
            i = m_a.size() - 1;

        double dx = this->extractDouble(x) - this->m_x[i];
        // P'[i](x) = a[i] + 2*b[i]*dx + 3*c[i]*dx^2
        return m_a[i] + dx * (2.0 * m_b[i] + 3.0 * m_c[i] * dx);
    }

    /// Access coefficients (DoubleT, AD-active)
    const std::vector<DoubleT>& aCoeffs() const { return m_a; }
    const std::vector<DoubleT>& bCoeffs() const { return m_b; }
    const std::vector<DoubleT>& cCoeffs() const { return m_c; }

private:
    DerivativeApprox m_da;
    std::vector<DoubleT> m_a; ///< First derivative at x[i]
    std::vector<DoubleT> m_b; ///< Coefficient for (x-x[i])^2
    std::vector<DoubleT> m_c; ///< Coefficient for (x-x[i])^3

    void calculateCoefficients() {
        size_t n = this->m_x.size();

        if (n < 2) {
            throw std::runtime_error("CubicInterpolation: need at least 2 points");
        }

        if (n == 2) {
            m_a.resize(1);
            m_b.resize(1);
            m_c.resize(1);
            double inv_dx = 1.0 / (this->m_x[1] - this->m_x[0]);
            m_a[0] = (this->m_y[1] - this->m_y[0]) * inv_dx;
            m_b[0] = DoubleT(0.0);
            m_c[0] = DoubleT(0.0);
            return;
        }

        // Grid spacing is pure double — off tape
        std::vector<double> dx(n - 1);
        for (size_t i = 0; i < n - 1; ++i) {
            dx[i] = this->m_x[i + 1] - this->m_x[i];
        }

        // Slopes are DoubleT (depend on m_y)
        std::vector<DoubleT> S(n - 1);
        for (size_t i = 0; i < n - 1; ++i) {
            S[i] = (this->m_y[i + 1] - this->m_y[i]) / dx[i];
        }

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
            m_b[i] = (3.0 * S[i] - derivatives[i + 1] - 2.0 * derivatives[i]) / dx[i];
            m_c[i] = (derivatives[i + 1] + derivatives[i] - 2.0 * S[i]) / (dx[i] * dx[i]);
        }
    }

    /**
     * @brief Solve tridiagonal system using Thomas algorithm
     *
     * Matrix (lower, diag, upper) is double — pure grid operations, off tape.
     * RHS is DoubleT — depends on node values m_y through slopes S.
     * This separation keeps the forward elimination of the matrix off tape;
     * only the RHS processing and back-substitution produce tape nodes.
     */
    std::vector<DoubleT> solveTridiagonal(const std::vector<double>& lower,
                                          const std::vector<double>& diag,
                                          const std::vector<double>& upper,
                                          const std::vector<DoubleT>& rhs) const {
        size_t n = rhs.size();
        if (n == 0)
            return {};
        if (n == 1)
            return {rhs[0] / diag[0]};

        // c_prime is pure double (matrix-only)
        std::vector<double> c_prime(n - 1);
        // d_prime is DoubleT (touches RHS)
        std::vector<DoubleT> d_prime(n);
        std::vector<DoubleT> x(n);

        // Forward elimination
        c_prime[0] = upper[0] / diag[0];
        d_prime[0] = rhs[0] / diag[0];

        for (size_t i = 1; i < n - 1; ++i) {
            double denom = diag[i] - lower[i - 1] * c_prime[i - 1];
            if (denom == 0.0) {
                throw std::runtime_error(
                    "CubicInterpolation: singular matrix in tridiagonal solve");
            }
            c_prime[i] = upper[i] / denom;
            d_prime[i] = (rhs[i] - lower[i - 1] * d_prime[i - 1]) / denom;
        }

        // Last row
        double denom = diag[n - 1] - lower[n - 2] * c_prime[n - 2];
        if (denom == 0.0) {
            throw std::runtime_error("CubicInterpolation: singular matrix in tridiagonal solve");
        }
        d_prime[n - 1] = (rhs[n - 1] - lower[n - 2] * d_prime[n - 2]) / denom;

        // Back substitution
        x[n - 1] = d_prime[n - 1];
        for (int i = n - 2; i >= 0; --i) {
            x[i] = d_prime[i] - c_prime[i] * x[i + 1];
        }

        return x;
    }

    /**
     * @brief Natural cubic spline derivative computation
     *
     * Tridiagonal matrix is pure double (grid spacing only).
     * RHS is DoubleT (depends on slopes S which depend on m_y).
     */
    void computeSplineDerivatives(std::vector<DoubleT>& deriv, const std::vector<double>& dx,
                                  const std::vector<DoubleT>& S) const {
        size_t n = this->m_x.size();

        std::vector<double> lower(n - 1);
        std::vector<double> diag(n);
        std::vector<double> upper(n - 1);
        std::vector<DoubleT> rhs(n);

        for (size_t i = 1; i < n - 1; ++i) {
            lower[i - 1] = dx[i - 1];
            diag[i] = 2.0 * (dx[i - 1] + dx[i]);
            upper[i] = dx[i];
            rhs[i] = 3.0 * (dx[i] * S[i - 1] + dx[i - 1] * S[i]);
        }

        // Natural boundary conditions
        diag[0] = 2.0;
        upper[0] = 1.0;
        rhs[0] = 3.0 * S[0];

        lower[n - 2] = 1.0;
        diag[n - 1] = 2.0;
        rhs[n - 1] = 3.0 * S[n - 2];

        deriv = solveTridiagonal(lower, diag, upper, rhs);
    }

    void computeParabolicDerivatives(std::vector<DoubleT>& deriv, const std::vector<double>& dx,
                                     const std::vector<DoubleT>& S) const {
        size_t n = this->m_x.size();

        for (size_t i = 1; i < n - 1; ++i) {
            deriv[i] = (dx[i - 1] * S[i] + dx[i] * S[i - 1]) / (dx[i] + dx[i - 1]);
        }

        deriv[0] = ((2.0 * dx[0] + dx[1]) * S[0] - dx[0] * S[1]) / (dx[0] + dx[1]);
        deriv[n - 1] = ((2.0 * dx[n - 2] + dx[n - 3]) * S[n - 2] - dx[n - 2] * S[n - 3]) /
                       (dx[n - 2] + dx[n - 3]);
    }

    void computeAkimaDerivatives(std::vector<DoubleT>& deriv, const std::vector<double>& dx,
                                 const std::vector<DoubleT>& S) const {
        size_t n = this->m_x.size();

        DoubleT w1 = abs_impl(S[1] - S[0]);
        DoubleT w2 = abs_impl(2.0 * S[0] * S[1] - 4.0 * S[0] * S[0] * S[1]);
        if (this->extractDouble(w1 + w2) == 0.0) {
            deriv[0] = S[0];
        } else {
            deriv[0] = (w1 * 2.0 * S[0] * S[1] + w2 * S[0]) / (w1 + w2);
        }

        w1 = abs_impl(S[2] - S[1]);
        w2 = abs_impl(S[0] - 2.0 * S[0] * S[1]);
        if (this->extractDouble(w1 + w2) == 0.0) {
            deriv[1] = S[1];
        } else {
            deriv[1] = (w1 * S[0] + w2 * S[1]) / (w1 + w2);
        }

        for (size_t i = 2; i < n - 2; ++i) {
            double si_m2 = this->extractDouble(S[i - 2]);
            double si_m1 = this->extractDouble(S[i - 1]);
            double si = this->extractDouble(S[i]);
            double si_p1 = this->extractDouble(S[i + 1]);

            if ((si_m2 == si_m1) && (si != si_p1)) {
                deriv[i] = S[i - 1];
            } else if ((si_m2 != si_m1) && (si == si_p1)) {
                deriv[i] = S[i];
            } else if (si == si_m1) {
                deriv[i] = S[i];
            } else if ((si_m2 == si_m1) && (si_m1 != si) && (si == si_p1)) {
                deriv[i] = (S[i - 1] + S[i]) / 2.0;
            } else {
                w1 = abs_impl(S[i + 1] - S[i]);
                w2 = abs_impl(S[i - 1] - S[i - 2]);
                if (this->extractDouble(w1 + w2) == 0.0) {
                    deriv[i] = (S[i - 1] + S[i]) / 2.0;
                } else {
                    deriv[i] = (w1 * S[i - 1] + w2 * S[i]) / (w1 + w2);
                }
            }
        }

        w1 = abs_impl(2.0 * S[n - 2] * S[n - 3] - S[n - 2]);
        w2 = abs_impl(S[n - 3] - S[n - 4]);
        if (this->extractDouble(w1 + w2) == 0.0) {
            deriv[n - 2] = S[n - 2];
        } else {
            deriv[n - 2] = (w1 * S[n - 3] + w2 * S[n - 2]) / (w1 + w2);
        }

        w1 = abs_impl(4.0 * S[n - 2] * S[n - 2] * S[n - 3] - 2.0 * S[n - 2] * S[n - 3]);
        w2 = abs_impl(S[n - 2] - S[n - 3]);
        if (this->extractDouble(w1 + w2) == 0.0) {
            deriv[n - 1] = S[n - 2];
        } else {
            deriv[n - 1] = (w1 * S[n - 2] + w2 * 2.0 * S[n - 2] * S[n - 3]) / (w1 + w2);
        }
    }

    void computeKrugerDerivatives(std::vector<DoubleT>& deriv, const std::vector<double>& dx,
                                  const std::vector<DoubleT>& S) const {
        size_t n = this->m_x.size();

        for (size_t i = 1; i < n - 1; ++i) {
            if (this->extractDouble(S[i - 1] * S[i]) < 0.0) {
                deriv[i] = DoubleT(0.0);
            } else {
                deriv[i] = 2.0 / (1.0 / S[i - 1] + 1.0 / S[i]);
            }
        }

        deriv[0] = (3.0 * S[0] - deriv[1]) / 2.0;
        deriv[n - 1] = (3.0 * S[n - 2] - deriv[n - 2]) / 2.0;
    }

    void computeHarmonicDerivatives(std::vector<DoubleT>& deriv, const std::vector<double>& dx,
                                    const std::vector<DoubleT>& S) const {
        size_t n = this->m_x.size();

        for (size_t i = 1; i < n - 1; ++i) {
            double w1 = 2.0 * dx[i] + dx[i - 1];
            double w2 = dx[i] + 2.0 * dx[i - 1];

            if (this->extractDouble(S[i - 1] * S[i]) <= 0.0) {
                deriv[i] = DoubleT(0.0);
            } else {
                deriv[i] = (w1 + w2) / (w1 / S[i - 1] + w2 / S[i]);
            }
        }

        deriv[0] = ((2.0 * dx[0] + dx[1]) * S[0] - dx[0] * S[1]) / (dx[1] + dx[0]);
        if (this->extractDouble(deriv[0] * S[0]) < 0.0) {
            deriv[0] = DoubleT(0.0);
        } else if (this->extractDouble(S[0] * S[1]) < 0.0) {
            if (abs_impl(deriv[0]) > abs_impl(3.0 * S[0])) {
                deriv[0] = 3.0 * S[0];
            }
        }

        deriv[n - 1] = ((2.0 * dx[n - 2] + dx[n - 3]) * S[n - 2] - dx[n - 2] * S[n - 3]) /
                       (dx[n - 3] + dx[n - 2]);
        if (this->extractDouble(deriv[n - 1] * S[n - 2]) < 0.0) {
            deriv[n - 1] = DoubleT(0.0);
        } else if (this->extractDouble(S[n - 2] * S[n - 3]) < 0.0) {
            if (abs_impl(deriv[n - 1]) > abs_impl(3.0 * S[n - 2])) {
                deriv[n - 1] = 3.0 * S[n - 2];
            }
        }
    }

    static DoubleT abs_impl(const DoubleT& x) {
        if constexpr (std::is_same_v<DoubleT, double>) {
            return std::abs(x);
        } else {
            using std::abs;
            return abs(x);
        }
    }
};

} // namespace Math

#endif // CUBIC_INTERPOLATION_H
