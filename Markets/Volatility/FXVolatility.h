#ifndef FXVOLATILITY_H
#define FXVOLATILITY_H

#include "Math/Interpolations/BicubicInterpolation.h"

#include <memory>
#include <stdexcept>
#include <vector>

#include "Markets/Descriptors/FXDescriptor.h"

namespace Markets {

/**
 * @brief FX volatility surface
 *
 * Template class for FX volatility surfaces compatible with AD.
 * Stores implied volatilities as a 2D surface (expiry x delta or strike).
 * FX vols are typically quoted in delta terms (25D PUT, ATM, 25D CALL, etc.)
 *
 * @tparam DoubleT Numeric type (double or stan::math::var for AD)
 * @tparam ContainerT Container type for vol surface (default: vector<vector<DoubleT>>)
 */
template <typename DoubleT, typename ContainerT = std::vector<std::vector<DoubleT>>>
class FXVolatility {
public:
    /**
     * @brief Constructor with vol surface
     * @param expiries Option expiries (in years)
     * @param deltas Delta levels (e.g., 0.25, 0.50, 0.75 for 25D PUT, ATM, 25D CALL)
     * @param volSurface Implied volatilities (expiries x deltas)
     * @param refSpot Reference FX spot rate
     * @param descriptor Vol surface metadata
     * @param deltaType Type: "DELTA" or "STRIKE"
     */
    template <typename VectorT, typename SurfaceT>
    FXVolatility(const VectorT& expiries, const VectorT& deltas, const SurfaceT& volSurface,
                 DoubleT refSpot, const FXDescriptor& descriptor = FXDescriptor(),
                 const std::string& deltaType = "DELTA")
        : m_descriptor(descriptor), m_refSpot(refSpot), m_deltaType(deltaType) {
        // Convert to vectors
        m_expiries = toVector(expiries);
        m_deltas = toVector(deltas);
        m_volSurface = toSurface(volSurface);

        // Validate dimensions
        if (m_volSurface.size() != m_expiries.size()) {
            throw std::runtime_error("FXVolatility: volSurface rows must match expiries size");
        }
        for (const auto& row : m_volSurface) {
            if (row.size() != m_deltas.size()) {
                throw std::runtime_error("FXVolatility: volSurface columns must match deltas size");
            }
        }

        // Create interpolator for vol surface (use Bicubic with Spline)
        m_interpolator = std::make_unique<Math::BicubicInterpolation<DoubleT>>(
            m_expiries, m_deltas, m_volSurface, Math::CubicInterpolation<DoubleT>::Spline);
    }

    /**
     * @brief Copy constructor
     */
    FXVolatility(const FXVolatility& other)
        : m_descriptor(other.m_descriptor), m_refSpot(other.m_refSpot),
          m_deltaType(other.m_deltaType), m_expiries(other.m_expiries), m_deltas(other.m_deltas),
          m_volSurface(other.m_volSurface) {
        m_interpolator = std::make_unique<Math::BicubicInterpolation<DoubleT>>(
            m_expiries, m_deltas, m_volSurface, Math::CubicInterpolation<DoubleT>::Spline);
    }

    /**
     * @brief Copy assignment
     */
    FXVolatility& operator=(const FXVolatility& other) {
        if (this != &other) {
            m_descriptor = other.m_descriptor;
            m_refSpot = other.m_refSpot;
            m_deltaType = other.m_deltaType;
            m_expiries = other.m_expiries;
            m_deltas = other.m_deltas;
            m_volSurface = other.m_volSurface;
            m_interpolator = std::make_unique<Math::BicubicInterpolation<DoubleT>>(
                m_expiries, m_deltas, m_volSurface, Math::CubicInterpolation<DoubleT>::Spline);
        }
        return *this;
    }

    /**
     * @brief Get volatility at given expiry and delta
     * @param expiry Option expiry in years
     * @param delta Option delta (e.g., 0.25 for 25 delta put)
     * @param allowExtrapolation Allow extrapolation beyond surface range
     * @return Implied volatility
     */
    DoubleT vol(DoubleT expiry, DoubleT delta, bool allowExtrapolation = true) const {
        return (*m_interpolator)(expiry, delta, allowExtrapolation);
    }

    /**
     * @brief Get volatility at given expiry and strike
     * @param expiry Option expiry in years
     * @param strike Strike level
     * @param allowExtrapolation Allow extrapolation beyond surface range
     * @return Implied volatility
     *
     * Note: If surface is in delta terms, this will use delta values directly.
     * For strike-based lookups, use strikeType="STRIKE" in constructor.
     */
    DoubleT volByStrike(DoubleT expiry, DoubleT strike, bool allowExtrapolation = true) const {
        if (m_deltaType == "STRIKE") {
            // Deltas are actually strikes
            return vol(expiry, strike, allowExtrapolation);
        } else {
            // Would need to convert strike to delta (requires forward, rd, rf, vol)
            // For now, throw error - this is non-trivial
            throw std::runtime_error("FXVolatility::volByStrike: strike-to-delta conversion not "
                                     "implemented. Use vol(expiry, delta) directly.");
        }
    }

    /**
     * @brief Get ATM volatility at given expiry
     * @param expiry Option expiry in years
     * @param allowExtrapolation Allow extrapolation beyond surface range
     * @return ATM implied volatility
     *
     * Note: Assumes ATM delta is 0.50 (50 delta)
     */
    DoubleT atmVol(DoubleT expiry, bool allowExtrapolation = true) const {
        return vol(expiry, DoubleT(0.50), allowExtrapolation);
    }

    /**
     * @brief Get descriptor
     */
    const FXDescriptor& descriptor() const { return m_descriptor; }

    /**
     * @brief Get reference spot
     */
    DoubleT referenceSpot() const { return m_refSpot; }

    /**
     * @brief Get delta type
     */
    const std::string& deltaType() const { return m_deltaType; }

    /**
     * @brief Get expiries
     */
    const std::vector<DoubleT>& expiries() const { return m_expiries; }

    /**
     * @brief Get deltas
     */
    const std::vector<DoubleT>& deltas() const { return m_deltas; }

    /**
     * @brief Get vol surface
     */
    const std::vector<std::vector<DoubleT>>& surface() const { return m_volSurface; }

    /**
     * @brief Scale entire surface by a scalar (multiply each element)
     * @param scalar Scaling factor
     */
    void scale(DoubleT scalar) {
        for (auto& row : m_volSurface) {
            for (auto& vol : row) {
                vol = vol * scalar;
            }
        }
        // Recreate interpolator with scaled surface
        m_interpolator = std::make_unique<Math::BicubicInterpolation<DoubleT>>(
            m_expiries, m_deltas, m_volSurface, Math::CubicInterpolation<DoubleT>::Spline);
    }

    /**
     * @brief Shift entire surface by a constant (add to each element)
     * @param shift Shift amount
     */
    void shift(DoubleT shift) {
        for (auto& row : m_volSurface) {
            for (auto& vol : row) {
                vol = vol + shift;
            }
        }
        // Recreate interpolator with shifted surface
        m_interpolator = std::make_unique<Math::BicubicInterpolation<DoubleT>>(
            m_expiries, m_deltas, m_volSurface, Math::CubicInterpolation<DoubleT>::Spline);
    }

    /**
     * @brief Apply a function to each element of the surface
     * @param func Function to apply: DoubleT f(DoubleT)
     */
    template <typename Func>
    void applyFunction(Func func) {
        for (auto& row : m_volSurface) {
            for (auto& vol : row) {
                vol = func(vol);
            }
        }
        // Recreate interpolator with transformed surface
        m_interpolator = std::make_unique<Math::BicubicInterpolation<DoubleT>>(
            m_expiries, m_deltas, m_volSurface, Math::CubicInterpolation<DoubleT>::Spline);
    }

    /**
     * @brief Bump a specific point in the surface
     * @param expiryIndex Index of expiry to bump
     * @param deltaIndex Index of delta to bump
     * @param bumpSize Size of bump
     */
    void bump(size_t expiryIndex, size_t deltaIndex, DoubleT bumpSize) {
        if (expiryIndex >= m_volSurface.size()) {
            throw std::runtime_error("FXVolatility::bump: invalid expiry index");
        }
        if (deltaIndex >= m_volSurface[expiryIndex].size()) {
            throw std::runtime_error("FXVolatility::bump: invalid delta index");
        }
        m_volSurface[expiryIndex][deltaIndex] = m_volSurface[expiryIndex][deltaIndex] + bumpSize;
        // Recreate interpolator
        m_interpolator = std::make_unique<Math::BicubicInterpolation<DoubleT>>(
            m_expiries, m_deltas, m_volSurface, Math::CubicInterpolation<DoubleT>::Spline);
    }

    /**
     * @brief Apply element-wise operation with coordinates
     * @param func Function to apply: DoubleT f(DoubleT vol, DoubleT expiry, DoubleT delta)
     *
     * Useful for operations that depend on position in the surface
     */
    template <typename Func>
    void applyFunctionWithCoords(Func func) {
        for (size_t i = 0; i < m_volSurface.size(); ++i) {
            for (size_t j = 0; j < m_volSurface[i].size(); ++j) {
                m_volSurface[i][j] = func(m_volSurface[i][j], m_expiries[i], m_deltas[j]);
            }
        }
        // Recreate interpolator
        m_interpolator = std::make_unique<Math::BicubicInterpolation<DoubleT>>(
            m_expiries, m_deltas, m_volSurface, Math::CubicInterpolation<DoubleT>::Spline);
    }

    /**
     * @brief Multiply surface by scalar (operator overload)
     */
    FXVolatility& operator*=(DoubleT scalar) {
        scale(scalar);
        return *this;
    }

    /**
     * @brief Add scalar to surface (operator overload)
     */
    FXVolatility& operator+=(DoubleT scalar) {
        shift(scalar);
        return *this;
    }

private:
    FXDescriptor m_descriptor;
    DoubleT m_refSpot;
    std::string m_deltaType;
    std::vector<DoubleT> m_expiries;
    std::vector<DoubleT> m_deltas;
    std::vector<std::vector<DoubleT>> m_volSurface;
    std::unique_ptr<Math::BicubicInterpolation<DoubleT>> m_interpolator;

    /**
     * @brief Convert container to vector
     */
    template <typename VectorT>
    std::vector<DoubleT> toVector(const VectorT& container) const {
        std::vector<DoubleT> result;
        result.reserve(container.size());
        for (const auto& val : container) {
            result.push_back(static_cast<DoubleT>(val));
        }
        return result;
    }

    /**
     * @brief Convert 2D container to vector<vector<DoubleT>>
     */
    template <typename SurfaceT>
    std::vector<std::vector<DoubleT>> toSurface(const SurfaceT& surface) const {
        std::vector<std::vector<DoubleT>> result;
        result.reserve(surface.size());
        for (const auto& row : surface) {
            std::vector<DoubleT> rowVec;
            rowVec.reserve(row.size());
            for (const auto& val : row) {
                rowVec.push_back(static_cast<DoubleT>(val));
            }
            result.push_back(rowVec);
        }
        return result;
    }
};

} // namespace Markets

#endif // FXVOLATILITY_H
