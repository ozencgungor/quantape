#ifndef CUBIC_INTERPOLATION_H
#define CUBIC_INTERPOLATION_H

#include <cmath>
#include <stdexcept>
#include <vector>

#include "Interpolation.h"
#include "ProbeDual.h"

namespace quantape::math {

/// Derivative approximation schemes for cubic interpolation. Namespace-scope
/// (not nested in the class template): a nested enum of one instantiation is
/// a DISTINCT type from the same-named nested enum of another instantiation,
/// which breaks passing the method across DoubleT instantiations.
enum class CubicDerivativeApprox {
    Spline,    // Natural cubic spline (C^2 continuous, DEFAULT)
    Parabolic, // Local parabolic approximation
    Akima,     // Akima's method
    Kruger,    // Kruger's harmonic mean method
    Harmonic   // Weighted harmonic mean
};

/// Precomputed probe weights for methods EXACTLY linear in the node values
/// (Spline/Parabolic):
///   P(x) in segment i = y_i + Σ_j (Wa[i,j] dx + Wb[i,j] dx² + Wc[i,j] dx³) y_j
/// The weights depend only on the GRID, not on the node values — so one
/// instance is reusable across many interpolators/evaluations on the same
/// grid. That is what makes the 2D (tensor-product) case cheap: the
/// y-direction weights are built once and shared by every per-query
/// y-interpolation.
struct CubicWeightMatrix {
    size_t n = 0;                   ///< grid size
    size_t seg = 0;                 ///< number of segments
    std::vector<double> Wa, Wb, Wc; ///< flat (seg * n) rows

    bool empty() const { return seg == 0; }
};

/**
 * @brief Cubic interpolation with various derivative approximation schemes
 *
 * Implements cubic interpolation following QuantLib's approach with multiple
 * derivative approximation methods. The polynomial form for each segment i is:
 *   P[i](x) = y[i] + a[i]*(x-x[i]) + b[i]*(x-x[i])^2 + c[i]*(x-x[i])^3
 *
 * Grid coordinates (m_x) are double; coefficients m_a, m_b, m_c are DoubleT
 * since they depend on node values m_y. The default evaluation builds
 * dx = x - x[i] as DoubleT, so the query coordinate is differentiated and
 * mixed d2P/dxdy blocks are exact for the active (primal-pinned) branch.
 *
 * AD dispatch:
 *   - default (operator()/derivative()): coefficient path with DoubleT dx —
 *     x is on the tape. Construction builds the coefficient vectors once
 *     (O(n) tape nodes), every evaluation is then a short expression.
 *   - passive-abscissa fast path (evaluateFixed/derivativeFixed):
 *     Spline/Parabolic use the precomputed global weight matrix (one tape
 *     node per evaluation, x adjoint not pushed); Akima/Kruger/Harmonic use
 *     the branch-pinned ProbeDual linearization.
 *
 * @tparam DoubleT Numeric type (double, stan::math::var, stan::math::fvar<var>)
 * @tparam Smooth  Compile-time default for the smoothing parameter: when true,
 *                 sign/abs branch conditions in the adaptive methods are
 *                 replaced by C^1 approximations (sigmoid blends, smooth abs).
 *                 Makes the interpolant differentiable everywhere at the cost
 *                 of slightly changing its values.
 */
template <typename DoubleT, bool Smooth = false>
class CubicInterpolation : public Interpolation<DoubleT, CubicInterpolation<DoubleT, Smooth>> {
    using Base = Interpolation<DoubleT, CubicInterpolation<DoubleT, Smooth>>;
    friend Base;

public:
    using DerivativeApprox = CubicDerivativeApprox; // back-compat alias

    template <typename ContainerX, typename ContainerY>
    CubicInterpolation(const ContainerX& x, const ContainerY& y,
                       DerivativeApprox da = DerivativeApprox::Spline, bool smooth = Smooth)
        : m_da(da), m_smooth(smooth) {
        this->m_x = this->toDoubleVector(x);
        this->m_y = this->toVector(y);
        this->validate();
        this->cacheSharedValues(); // one snapshot for the Fixed-path callbacks
        calculateCoefficients();
        if constexpr (!std::is_same_v<DoubleT, double>) {
            buildWeightMatrix(); // passive-abscissa fast path (linear methods)
        }
    }

    /// Constructor with a precomputed grid weight matrix (shared across
    /// interpolators on the same grid — the 2D per-query y-interpolation
    /// uses this to avoid re-probing on every evaluation).
    template <typename ContainerX, typename ContainerY>
    CubicInterpolation(const ContainerX& x, const ContainerY& y, DerivativeApprox da, bool smooth,
                       const CubicWeightMatrix& precomputed)
        : m_da(da), m_smooth(smooth) {
        this->m_x = this->toDoubleVector(x);
        this->m_y = this->toVector(y);
        this->validate();
        this->cacheSharedValues(); // one snapshot for the Fixed-path callbacks
        calculateCoefficients();
        if constexpr (!std::is_same_v<DoubleT, double>) {
            if (!precomputed.empty())
                applyWeights(precomputed);
            else
                buildWeightMatrix(); // fallback (e.g., adaptive methods)
        }
    }

    /// Default (AD-aware) value: dx is DoubleT, so x is on the tape.
    DoubleT valueImpl(DoubleT x) const { return coefficientValue(x); }

    DoubleT derivativeImpl(DoubleT x) const { return coefficientDerivative(x); }

    /// Passive-abscissa policy (fast path): x adjoint not pushed.
    DoubleT valueFixedImpl(DoubleT x) const {
        if constexpr (std::is_same_v<DoubleT, double>) {
            return coefficientValue(x);
        } else {
            if (m_useWeights)
                return weightMatrixValue(x); // Spline/Parabolic: global weights
            return localWeightsValue(x);     // adaptive methods: branch-pinned probe
        }
    }

    DoubleT derivativeFixedImpl(DoubleT x) const {
        if constexpr (std::is_same_v<DoubleT, double>) {
            return coefficientDerivative(x);
        } else {
            if (m_useWeights)
                return weightMatrixDerivative(x);
            return localWeightsDerivative(x);
        }
    }

    /// Access coefficients (DoubleT, AD-active)
    const std::vector<DoubleT>& aCoeffs() const { return m_a; }
    const std::vector<DoubleT>& bCoeffs() const { return m_b; }
    const std::vector<DoubleT>& cCoeffs() const { return m_c; }

    /// Whether this instance uses the weight-matrix representation
    /// (Spline/Parabolic with an AD DoubleT).
    bool usesWeightMatrix() const { return m_useWeights; }

private:
    DerivativeApprox m_da;
    bool m_smooth = false;
    std::vector<DoubleT> m_a; ///< First derivative at x[i]
    std::vector<DoubleT> m_b; ///< Coefficient for (x-x[i])^2
    std::vector<DoubleT> m_c; ///< Coefficient for (x-x[i])^3

    // ── Weight-matrix representation (AD path, Spline/Parabolic only) ──
    //
    // For methods that are EXACTLY linear in the node values:
    //   P(x) in segment i = y_i + Σ_j (W_a[i,j] dx + W_b[i,j] dx^2
    //                                       + W_c[i,j] dx^3) * y_j
    // with all weights pure double. Built once by probing the double
    // solver with unit vectors. Akima/Kruger/Harmonic are excluded: their
    // branch selection depends on y, so precomputed weights would go
    // stale when a branch flips.
    bool m_useWeights = false;
    std::vector<double> m_Wa, m_Wb, m_Wc; ///< flat (segment * n) rows

    // Defined for stan::math::var / stan::math::fvar<var> in
    // InterpolationStanPrimitives.h (passive-abscissa fast path). Never
    // ODR-used for double.
    DoubleT weightMatrixValue(DoubleT x) const;
    DoubleT weightMatrixDerivative(DoubleT x) const;
    DoubleT localWeightsValue(DoubleT x) const;
    DoubleT localWeightsDerivative(DoubleT x) const;

    void buildWeightMatrix() { applyWeights(probeWeights(this->m_x, m_da)); }

    void applyWeights(const CubicWeightMatrix& W) {
        if (W.empty())
            return; // adaptive methods: not linear in y, no global weights
        m_useWeights = true;
        m_Wa = W.Wa;
        m_Wb = W.Wb;
        m_Wc = W.Wc;
    }

    /// Probe the interpolation operator with unit vectors on a double
    /// instance of the same method, reading off each response's
    /// coefficients. O(n) solves, O(n^2) total — pure double, never on any
    /// tape. Public/static so 2D interpolators can cache one per grid.
public:
    static CubicWeightMatrix probeWeights(const std::vector<double>& x, DerivativeApprox da) {
        CubicWeightMatrix W;
        const bool linearInY =
            (da == DerivativeApprox::Spline || da == DerivativeApprox::Parabolic);
        if (!linearInY)
            return W; // adaptive methods: data-dependent branch selection

        const size_t n = x.size();
        const size_t seg = (n == 2) ? 1 : n - 1;
        W.n = n;
        W.seg = seg;
        W.Wa.assign(seg * n, 0.0);
        W.Wb.assign(seg * n, 0.0);
        W.Wc.assign(seg * n, 0.0);

        for (size_t j = 0; j < n; ++j) {
            std::vector<double> e(n, 0.0);
            e[j] = 1.0;
            const CubicInterpolation<double> probe(x, e, da, false);
            const auto& pa = probe.aCoeffs();
            const auto& pb = probe.bCoeffs();
            const auto& pc = probe.cCoeffs();
            for (size_t i = 0; i < seg; ++i) {
                W.Wa[i * n + j] = pa[i];
                W.Wb[i * n + j] = pb[i];
                W.Wc[i * n + j] = pc[i];
            }
        }
        return W;
    }

private:
    // ── Coefficient path (double fast path) ──

    DoubleT coefficientValue(DoubleT x) const {
        size_t i = this->locate(x);
        if (i >= m_a.size())
            i = m_a.size() - 1;

        // dx is DoubleT: x is on the tape together with y[i], a[i], b[i], c[i]
        DoubleT dx = x - DoubleT(this->m_x[i]);
        // P[i](x) = y[i] + a[i]*dx + b[i]*dx^2 + c[i]*dx^3
        return this->m_y[i] + dx * (m_a[i] + dx * (m_b[i] + dx * m_c[i]));
    }

    DoubleT coefficientDerivative(DoubleT x) const {
        size_t i = this->locate(x);
        if (i >= m_a.size())
            i = m_a.size() - 1;

        DoubleT dx = x - DoubleT(this->m_x[i]);
        // P'[i](x) = a[i] + 2*b[i]*dx + 3*c[i]*dx^2
        return m_a[i] + dx * (DoubleT(2.0) * m_b[i] + DoubleT(3.0) * m_c[i] * dx);
    }

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
            case DerivativeApprox::Spline:
                derivatives = computeSplineDerivatives(n, dx, S, m_smooth);
                break;
            case DerivativeApprox::Parabolic:
                derivatives = computeParabolicDerivatives(n, dx, S, m_smooth);
                break;
            case DerivativeApprox::Akima:
                derivatives = computeAkimaDerivatives(n, dx, S, m_smooth);
                break;
            case DerivativeApprox::Kruger:
                derivatives = computeKrugerDerivatives(n, dx, S, m_smooth);
                break;
            case DerivativeApprox::Harmonic:
                derivatives = computeHarmonicDerivatives(n, dx, S, m_smooth);
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

    // ── Shared type plumbing: primal extraction and abs ──

    static double primalOf(double x) { return x; }
    static double primalOf(const quantape::math::detail::ProbeDual& x) {
        return quantape::math::detail::primal(x);
    }
    template <typename T>
    static double primalOf(const T& x) {
        return primalOf(x.val());
    }

    template <typename T>
    static T absImpl(const T& x) {
        if constexpr (std::is_same_v<T, double>) {
            return std::abs(x);
        } else {
            using std::abs;
            return abs(x); // ADL: stan::math::abs, or quantape::math::detail::abs(ProbeDual)
        }
    }

    template <typename T>
    static T smoothAbs(const T& x) {
        if constexpr (std::is_same_v<T, double>) {
            return quantape::math::detail::smoothAbs(x);
        } else if constexpr (std::is_same_v<T, quantape::math::detail::ProbeDual>) {
            return quantape::math::detail::smoothAbs(x);
        } else {
            const double eps = 1e-8;
            return sqrt(x * x + eps * eps); // stan types: generic C^1 abs
        }
    }

    template <typename T>
    static T weightAbs(const T& x, bool smooth) {
        return smooth ? smoothAbs(x) : absImpl(x);
    }

    // ── Derivative approximation schemes (static, scalar-generic) ──
    // All five run for double, DoubleT (coefficient path), and ProbeDual
    // (branch-pinned linearization). Branch conditions always use primalOf.

    template <typename T>
    static std::vector<T>
    solveTridiagonalT(const std::vector<double>& lower, const std::vector<double>& diag,
                      const std::vector<double>& upper, const std::vector<T>& rhs) {
        const size_t n = rhs.size();
        if (n == 0)
            return {};
        if (n == 1)
            return {rhs[0] / diag[0]};

        std::vector<double> c_prime(n - 1);
        std::vector<T> d_prime(n);
        std::vector<T> x(n);

        c_prime[0] = upper[0] / diag[0];
        d_prime[0] = rhs[0] / diag[0];

        for (size_t i = 1; i < n - 1; ++i) {
            double denom = diag[i] - lower[i - 1] * c_prime[i - 1];
            if (denom == 0.0)
                throw std::runtime_error("CubicInterpolation: singular tridiagonal system");
            c_prime[i] = upper[i] / denom;
            d_prime[i] = (rhs[i] - lower[i - 1] * d_prime[i - 1]) / denom;
        }

        double denom = diag[n - 1] - lower[n - 2] * c_prime[n - 2];
        if (denom == 0.0)
            throw std::runtime_error("CubicInterpolation: singular tridiagonal system");
        d_prime[n - 1] = (rhs[n - 1] - lower[n - 2] * d_prime[n - 2]) / denom;

        x[n - 1] = d_prime[n - 1];
        for (int i = static_cast<int>(n) - 2; i >= 0; --i) {
            x[i] = d_prime[i] - c_prime[i] * x[i + 1];
        }
        return x;
    }

    template <typename T>
    static std::vector<T> computeSplineDerivatives(size_t n, const std::vector<double>& dx,
                                                   const std::vector<T>& S, bool /*smooth*/) {
        std::vector<double> lower(n - 1);
        std::vector<double> diag(n);
        std::vector<double> upper(n - 1);
        std::vector<T> rhs(n);

        for (size_t i = 1; i < n - 1; ++i) {
            lower[i - 1] = dx[i - 1];
            diag[i] = 2.0 * (dx[i - 1] + dx[i]);
            upper[i] = dx[i];
            rhs[i] = 3.0 * (dx[i] * S[i - 1] + dx[i - 1] * S[i]);
        }

        diag[0] = 2.0;
        upper[0] = 1.0;
        rhs[0] = 3.0 * S[0];

        lower[n - 2] = 1.0;
        diag[n - 1] = 2.0;
        rhs[n - 1] = 3.0 * S[n - 2];

        return solveTridiagonalT(lower, diag, upper, rhs);
    }

    template <typename T>
    static std::vector<T> computeParabolicDerivatives(size_t n, const std::vector<double>& dx,
                                                      const std::vector<T>& S, bool /*smooth*/) {
        std::vector<T> deriv(n);
        for (size_t i = 1; i < n - 1; ++i) {
            deriv[i] = (dx[i - 1] * S[i] + dx[i] * S[i - 1]) / (dx[i] + dx[i - 1]);
        }
        deriv[0] = ((2.0 * dx[0] + dx[1]) * S[0] - dx[0] * S[1]) / (dx[0] + dx[1]);
        deriv[n - 1] = ((2.0 * dx[n - 2] + dx[n - 3]) * S[n - 2] - dx[n - 2] * S[n - 3]) /
                       (dx[n - 2] + dx[n - 3]);
        return deriv;
    }

    template <typename T>
    static std::vector<T> computeAkimaDerivatives(size_t n, const std::vector<double>& /*dx*/,
                                                  const std::vector<T>& S, bool smooth) {
        std::vector<T> deriv(n);

        T w1 = weightAbs(S[1] - S[0], smooth);
        T w2 = weightAbs(2.0 * S[0] * S[1] - 4.0 * S[0] * S[0] * S[1], smooth);
        if (primalOf(w1 + w2) == 0.0) {
            deriv[0] = S[0];
        } else {
            deriv[0] = (w1 * 2.0 * S[0] * S[1] + w2 * S[0]) / (w1 + w2);
        }

        w1 = weightAbs(S[2] - S[1], smooth);
        w2 = weightAbs(S[0] - 2.0 * S[0] * S[1], smooth);
        if (primalOf(w1 + w2) == 0.0) {
            deriv[1] = S[1];
        } else {
            deriv[1] = (w1 * S[0] + w2 * S[1]) / (w1 + w2);
        }

        for (size_t i = 2; i < n - 2; ++i) {
            double si_m2 = primalOf(S[i - 2]);
            double si_m1 = primalOf(S[i - 1]);
            double si = primalOf(S[i]);
            double si_p1 = primalOf(S[i + 1]);

            if ((si_m2 == si_m1) && (si != si_p1)) {
                deriv[i] = S[i - 1];
            } else if ((si_m2 != si_m1) && (si == si_p1)) {
                deriv[i] = S[i];
            } else if (si == si_m1) {
                deriv[i] = S[i];
            } else if ((si_m2 == si_m1) && (si_m1 != si) && (si == si_p1)) {
                deriv[i] = (S[i - 1] + S[i]) / 2.0;
            } else {
                w1 = weightAbs(S[i + 1] - S[i], smooth);
                w2 = weightAbs(S[i - 1] - S[i - 2], smooth);
                if (primalOf(w1 + w2) == 0.0) {
                    deriv[i] = (S[i - 1] + S[i]) / 2.0;
                } else {
                    deriv[i] = (w1 * S[i - 1] + w2 * S[i]) / (w1 + w2);
                }
            }
        }

        w1 = weightAbs(2.0 * S[n - 2] * S[n - 3] - S[n - 2], smooth);
        w2 = weightAbs(S[n - 3] - S[n - 4], smooth);
        if (primalOf(w1 + w2) == 0.0) {
            deriv[n - 2] = S[n - 2];
        } else {
            deriv[n - 2] = (w1 * S[n - 3] + w2 * S[n - 2]) / (w1 + w2);
        }

        w1 = weightAbs(4.0 * S[n - 2] * S[n - 2] * S[n - 3] - 2.0 * S[n - 2] * S[n - 3], smooth);
        w2 = weightAbs(S[n - 2] - S[n - 3], smooth);
        if (primalOf(w1 + w2) == 0.0) {
            deriv[n - 1] = S[n - 2];
        } else {
            deriv[n - 1] = (w1 * S[n - 2] + w2 * 2.0 * S[n - 2] * S[n - 3]) / (w1 + w2);
        }

        return deriv;
    }

    template <typename T>
    static std::vector<T> computeKrugerDerivatives(size_t n, const std::vector<double>& /*dx*/,
                                                   const std::vector<T>& S, bool smooth) {
        std::vector<T> deriv(n);

        // sigmoid gain for the branch blend; scale-aware via slope magnitude
        double s_scale = 0.0;
        if (smooth) {
            for (size_t i = 0; i < n - 1; ++i)
                s_scale = std::max(s_scale, std::abs(primalOf(S[i])));
            s_scale = std::max(s_scale, 1e-12);
        }
        const double kappa = 1e4 / (s_scale * s_scale);

        for (size_t i = 1; i < n - 1; ++i) {
            const double pa = primalOf(S[i - 1]);
            const double pb = primalOf(S[i]);
            if (smooth) {
                const double blend = quantape::math::detail::sigmoid(pa * pb, kappa);
                if (std::abs(pa + pb) < 1e-12) {
                    deriv[i] = T(0.0);
                } else {
                    deriv[i] = T(blend) * (T(2.0) / (T(1.0) / S[i - 1] + T(1.0) / S[i]));
                }
            } else {
                if (pa * pb < 0.0) {
                    deriv[i] = T(0.0);
                } else {
                    deriv[i] = T(2.0) / (T(1.0) / S[i - 1] + T(1.0) / S[i]);
                }
            }
        }

        deriv[0] = (3.0 * S[0] - deriv[1]) / 2.0;
        deriv[n - 1] = (3.0 * S[n - 2] - deriv[n - 2]) / 2.0;
        return deriv;
    }

    template <typename T>
    static std::vector<T> computeHarmonicDerivatives(size_t n, const std::vector<double>& dx,
                                                     const std::vector<T>& S, bool smooth) {
        std::vector<T> deriv(n);

        double s_scale = 0.0;
        if (smooth) {
            for (size_t i = 0; i < n - 1; ++i)
                s_scale = std::max(s_scale, std::abs(primalOf(S[i])));
            s_scale = std::max(s_scale, 1e-12);
        }
        const double kappa = 1e4 / (s_scale * s_scale);

        for (size_t i = 1; i < n - 1; ++i) {
            double w1 = 2.0 * dx[i] + dx[i - 1];
            double w2 = dx[i] + 2.0 * dx[i - 1];

            const double pa = primalOf(S[i - 1]);
            const double pb = primalOf(S[i]);
            if (smooth) {
                const double blend = quantape::math::detail::sigmoid(pa * pb, kappa);
                const double denom = primalOf(T(w1) / S[i - 1] + T(w2) / S[i]);
                if (std::abs(denom) < 1e-12) {
                    deriv[i] = T(0.0);
                } else {
                    deriv[i] = T(blend * (w1 + w2)) / (T(w1) / S[i - 1] + T(w2) / S[i]);
                }
            } else {
                if (pa * pb <= 0.0) {
                    deriv[i] = T(0.0);
                } else {
                    deriv[i] = T(w1 + w2) / (T(w1) / S[i - 1] + T(w2) / S[i]);
                }
            }
        }

        deriv[0] = ((2.0 * dx[0] + dx[1]) * S[0] - dx[0] * S[1]) / (dx[1] + dx[0]);
        if (primalOf(deriv[0] * S[0]) < 0.0) {
            deriv[0] = T(0.0);
        } else if (primalOf(S[0] * S[1]) < 0.0) {
            if (primalOf(absImpl(deriv[0])) > primalOf(absImpl(3.0 * S[0]))) {
                deriv[0] = 3.0 * S[0];
            }
        }

        deriv[n - 1] = ((2.0 * dx[n - 2] + dx[n - 3]) * S[n - 2] - dx[n - 2] * S[n - 3]) /
                       (dx[n - 3] + dx[n - 2]);
        if (primalOf(deriv[n - 1] * S[n - 2]) < 0.0) {
            deriv[n - 1] = T(0.0);
        } else if (primalOf(S[n - 2] * S[n - 3]) < 0.0) {
            if (primalOf(absImpl(deriv[n - 1])) > primalOf(absImpl(3.0 * S[n - 2]))) {
                deriv[n - 1] = 3.0 * S[n - 2];
            }
        }

        return deriv;
    }

    // ── Dual coefficient computation for the adaptive-method AD path ──
    // Public for the Stan specializations (member functions access private
    // anyway); static, reusable, and never instantiated for double.

public:
    static void computeCoefficientsDual(const std::vector<double>& x,
                                        const std::vector<quantape::math::detail::ProbeDual>& y,
                                        DerivativeApprox da, bool smooth,
                                        std::vector<quantape::math::detail::ProbeDual>& a,
                                        std::vector<quantape::math::detail::ProbeDual>& b,
                                        std::vector<quantape::math::detail::ProbeDual>& c) {
        const size_t n = x.size();
        if (n < 2)
            throw std::runtime_error("CubicInterpolation: need at least 2 points");

        const size_t seg = (n == 2) ? 1 : n - 1;
        a.resize(seg);
        b.resize(seg);
        c.resize(seg);

        if (n == 2) {
            const double inv_dx = 1.0 / (x[1] - x[0]);
            a[0] = (y[1] - y[0]) * inv_dx;
            b[0] = quantape::math::detail::ProbeDual(0.0, n);
            c[0] = quantape::math::detail::ProbeDual(0.0, n);
            return;
        }

        std::vector<double> dx(n - 1);
        for (size_t i = 0; i < n - 1; ++i)
            dx[i] = x[i + 1] - x[i];

        std::vector<quantape::math::detail::ProbeDual> S(n - 1);
        for (size_t i = 0; i < n - 1; ++i)
            S[i] = (y[i + 1] - y[i]) / dx[i];

        std::vector<quantape::math::detail::ProbeDual> derivatives(n);
        switch (da) {
            case DerivativeApprox::Spline:
                derivatives = computeSplineDerivatives(n, dx, S, smooth);
                break;
            case DerivativeApprox::Parabolic:
                derivatives = computeParabolicDerivatives(n, dx, S, smooth);
                break;
            case DerivativeApprox::Akima:
                derivatives = computeAkimaDerivatives(n, dx, S, smooth);
                break;
            case DerivativeApprox::Kruger:
                derivatives = computeKrugerDerivatives(n, dx, S, smooth);
                break;
            case DerivativeApprox::Harmonic:
                derivatives = computeHarmonicDerivatives(n, dx, S, smooth);
                break;
            default:
                throw std::runtime_error("CubicInterpolation: unknown derivative approximation");
        }

        for (size_t i = 0; i < n - 1; ++i) {
            a[i] = derivatives[i];
            b[i] = (quantape::math::detail::ProbeDual(3.0, n) * S[i] - derivatives[i + 1] -
                    quantape::math::detail::ProbeDual(2.0, n) * derivatives[i]) /
                   dx[i];
            c[i] = (derivatives[i + 1] + derivatives[i] -
                    quantape::math::detail::ProbeDual(2.0, n) * S[i]) /
                   (dx[i] * dx[i]);
        }
    }
};

} // namespace quantape::math

#endif // CUBIC_INTERPOLATION_H
