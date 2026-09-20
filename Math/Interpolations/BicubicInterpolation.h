#ifndef BICUBIC_INTERPOLATION_H
#define BICUBIC_INTERPOLATION_H

#include <vector>

#include "CubicInterpolation.h"
#include "Interpolation2D.h"

namespace Math {

/**
 * @brief Bicubic interpolation using cubic interpolation along each axis
 *
 * Tensor product: interpolate along x for each y row, then interpolate the
 * results along y. The x-row CubicInterpolation objects are precomputed at
 * construction.
 *
 * AD behaviour (after the weight-matrix / branch-pinned-probe rework):
 *   - construction puts NOTHING on the tape for any method (rows skip the
 *     coefficient path for AD types; linear methods build their weight
 *     matrices with pure-double probes);
 *   - a query costs O(ny) tape nodes (one per x-row evaluation), a
 *     tape-free y-interpolation (its grid weight matrix is cached here —
 *     the weights depend only on the grid, so they are shared across all
 *     queries), and one y-evaluation tape node with O(ny) adjoint pushes;
 *   - for exactly-linear methods (Spline/Parabolic) the full bicubic is
 *     linear in z, so the z-Hessian is identically zero;
 *   - adaptive methods (Akima/Kruger/Harmonic) evaluate through the
 *     branch-pinned probe in each direction, giving the active-branch
 *     subgradient per evaluation.
 *
 * @tparam DoubleT Numeric type (double, stan::math::var, stan::math::fvar<var>)
 * @tparam Smooth  Compile-time default for the runtime smoothing flag. The
 *                 AD specializations exist for the default Smooth=false
 *                 instantiation — pass smooth=true to the constructor at
 *                 runtime rather than instantiating Smooth=true.
 */
template <typename DoubleT, bool Smooth = false>
class BicubicInterpolation
    : public Interpolation2D<DoubleT, BicubicInterpolation<DoubleT, Smooth>> {
    static_assert(!Smooth || std::is_same_v<DoubleT, double>,
                  "AD specializations exist for Smooth=false; pass smooth=true at runtime");
    using Base = Interpolation2D<DoubleT, BicubicInterpolation<DoubleT, Smooth>>;
    friend Base;

public:
    using DerivativeApprox = CubicDerivativeApprox;

    template <typename ContainerX, typename ContainerY, typename Container2D>
    BicubicInterpolation(const ContainerX& x, const ContainerY& y, const Container2D& z,
                         DerivativeApprox method = DerivativeApprox::Spline,
                         bool smooth = Smooth)
        : m_method(method), m_smooth(smooth) {
        m_x = this->toDoubleVector(x);
        m_y = this->toDoubleVector(y);
        m_z = this->toVector2D(z);

        if (m_z.size() != m_y.size()) {
            throw std::runtime_error("BicubicInterpolation: z rows must match y size");
        }
        for (const auto& row : m_z) {
            if (row.size() != m_x.size()) {
                throw std::runtime_error("BicubicInterpolation: z columns must match x size");
            }
        }

        // Precompute x-row cubic interpolations (tape-free for AD types).
        m_x_interps.reserve(m_y.size());
        for (size_t j = 0; j < m_y.size(); ++j) {
            m_x_interps.emplace_back(m_x, m_z[j], m_method, m_smooth);
        }

        // Cache the y-direction grid weights ONCE (linear methods only).
        // The weights depend only on m_y, so every per-query
        // y-interpolation reuses them instead of re-probing.
        if constexpr (!std::is_same_v<DoubleT, double>) {
            m_y_weights = CubicInterpolation<DoubleT>::probeWeights(m_y, m_method);
        }
    }

    DoubleT valueImpl(DoubleT x, DoubleT y) const {
        // Step 1: evaluate the precomputed x-row cubics
        std::vector<DoubleT> y_values(m_y.size());
        for (size_t j = 0; j < m_y.size(); ++j) {
            y_values[j] = m_x_interps[j](x, true);
        }

        // Step 2: y-direction cubic on the intermediate results
        if constexpr (!std::is_same_v<DoubleT, double>) {
            if (!m_y_weights.empty()) {
                const CubicInterpolation<DoubleT> y_interp(
                    m_y, y_values, m_method, m_smooth, m_y_weights);
                return y_interp(y, true);
            }
        }
        const CubicInterpolation<DoubleT> y_interp(m_y, y_values, m_method, m_smooth);
        return y_interp(y, true);
    }

    bool isInRange(DoubleT x, DoubleT y) const {
        if (m_x.empty() || m_y.empty())
            return false;
        double xv = this->extractDouble(x);
        double yv = this->extractDouble(y);
        return xv >= m_x.front() && xv <= m_x.back() && yv >= m_y.front() && yv <= m_y.back();
    }

    /// Whether the cached y-direction weight matrix is in use
    /// (Spline/Parabolic with an AD DoubleT).
    bool usesWeightMatrix() const { return !m_y_weights.empty(); }

private:
    DerivativeApprox m_method;
    bool m_smooth = false;
    std::vector<double> m_x, m_y;                         ///< Grid coordinates (double)
    std::vector<std::vector<DoubleT>> m_z;                ///< Node values (DoubleT, AD-active)
    std::vector<CubicInterpolation<DoubleT>> m_x_interps; ///< Precomputed x-row cubics
    CubicWeightMatrix m_y_weights;                        ///< Cached y-grid weights (AD, linear)
};

} // namespace Math

#endif // BICUBIC_INTERPOLATION_H
