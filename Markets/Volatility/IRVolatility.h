#ifndef IRVOLATILITY_H
#define IRVOLATILITY_H

#include "Math/Interpolations/BicubicInterpolation.h"

#include <memory>
#include <stdexcept>
#include <vector>

#include "Markets/Descriptors/IRVolDescriptor.h"

namespace Markets {

/**
 * @brief IR volatility types
 */
enum class IRVolType {
    Swaption, ///< Swaption volatility (expiry x tenor = time to expiry x swap tenor)
    Cap       ///< Cap/Floor volatility (expiry x tenor = time to caplet x forward rate tenor)
};

/**
 * @brief Interest rate volatility surface
 *
 * Template class for IR volatility surfaces compatible with AD.
 * Stores ATM volatilities as a 2D surface (expiry x tenor) with optional smile adjustments.
 *
 * For Swaption vols:
 *   - Expiry axis: time to option expiry (e.g., 1y, 2y, 5y)
 *   - Tenor axis: underlying swap tenor (e.g., 1y, 5y, 10y)
 *   - Surface represents ATM swaption volatilities
 *
 * For Cap vols:
 *   - Expiry axis: time to caplet (e.g., 1y, 2y, 5y)
 *   - Tenor axis: forward rate tenor (typically fixed, e.g., 3M)
 *   - Surface represents ATM cap/floor volatilities
 *
 * @tparam DoubleT Numeric type (double or stan::math::var for AD)
 * @tparam VolType Type of IR volatility (Swaption or Cap)
 * @tparam ContainerT Container type for vol surface (default: vector<vector<DoubleT>>)
 */
template <typename DoubleT, IRVolType VolType = IRVolType::Swaption,
          typename ContainerT = std::vector<std::vector<DoubleT>>>
class IRVolatility {
public:
    /**
     * @brief Constructor with ATM vol surface
     * @param expiries Option expiries (in years)
     * @param tenors Underlying tenors (in years)
     * @param atmVolSurface ATM volatilities (expiries x tenors)
     * @param descriptor Vol surface metadata
     */
    template <typename VectorT, typename SurfaceT>
    IRVolatility(const VectorT& expiries, const VectorT& tenors, const SurfaceT& atmVolSurface,
                 const IRVolDescriptor& descriptor = IRVolDescriptor())
        : m_descriptor(descriptor), m_hasSmile(false) {
        // Convert to vectors
        m_expiries = toVector(expiries);
        m_tenors = toVector(tenors);
        m_atmVolSurface = toSurface(atmVolSurface);

        // Validate dimensions
        if (m_atmVolSurface.size() != m_expiries.size()) {
            throw std::runtime_error("IRVolatility: atmVolSurface rows must match expiries size");
        }
        for (const auto& row : m_atmVolSurface) {
            if (row.size() != m_tenors.size()) {
                throw std::runtime_error(
                    "IRVolatility: atmVolSurface columns must match tenors size");
            }
        }

        // Create interpolator for ATM surface (use Bicubic with Spline)
        // Note: BicubicInterpolation expects (x, y, z) where z[y_index][x_index]
        // So we pass (tenors, expiries, z) since our z is organized as z[expiry][tenor]
        m_atmInterpolator = std::make_unique<Math::BicubicInterpolation<DoubleT>>(
            m_tenors, m_expiries, m_atmVolSurface, Math::CubicInterpolation<DoubleT>::Spline);
    }

    /**
     * @brief Copy constructor
     */
    IRVolatility(const IRVolatility& other)
        : m_descriptor(other.m_descriptor), m_expiries(other.m_expiries), m_tenors(other.m_tenors),
          m_atmVolSurface(other.m_atmVolSurface), m_hasSmile(other.m_hasSmile),
          m_strikes(other.m_strikes) {
        m_atmInterpolator = std::make_unique<Math::BicubicInterpolation<DoubleT>>(
            m_tenors, m_expiries, m_atmVolSurface, Math::CubicInterpolation<DoubleT>::Spline);
    }

    /**
     * @brief Copy assignment
     */
    IRVolatility& operator=(const IRVolatility& other) {
        if (this != &other) {
            m_descriptor = other.m_descriptor;
            m_expiries = other.m_expiries;
            m_tenors = other.m_tenors;
            m_atmVolSurface = other.m_atmVolSurface;
            m_hasSmile = other.m_hasSmile;
            m_strikes = other.m_strikes;
            m_atmInterpolator = std::make_unique<Math::BicubicInterpolation<DoubleT>>(
                m_tenors, m_expiries, m_atmVolSurface, Math::CubicInterpolation<DoubleT>::Spline);
        }
        return *this;
    }

    /**
     * @brief Get ATM volatility at given expiry and tenor
     * @param expiry Option expiry in years
     * @param tenor Underlying tenor in years
     * @param allowExtrapolation Allow extrapolation beyond surface range
     * @return ATM volatility
     */
    DoubleT atmVol(DoubleT expiry, DoubleT tenor, bool allowExtrapolation = true) const {
        // Note: interpolator is (x=tenors, y=expiries), so we call it with (tenor, expiry)
        return (*m_atmInterpolator)(tenor, expiry, allowExtrapolation);
    }

    /**
     * @brief Get volatility at given expiry, tenor, and strike
     * @param expiry Option expiry in years
     * @param tenor Underlying tenor in years
     * @param strike Strike rate (absolute, e.g., 0.05 for 5%)
     * @param allowExtrapolation Allow extrapolation beyond surface range
     * @return Volatility with smile adjustment
     *
     * Note: If no smile is set, returns ATM vol. Otherwise applies smile adjustment.
     */
    DoubleT vol(DoubleT expiry, DoubleT tenor, DoubleT strike,
                bool allowExtrapolation = true) const {
        DoubleT atmVol = this->atmVol(expiry, tenor, allowExtrapolation);

        if (!m_hasSmile) {
            // No smile adjustment, return ATM
            return atmVol;
        }

        // Apply smile adjustment (to be implemented when smile surface is added)
        // For now, just return ATM vol
        // TODO: Implement SABR or strike-based smile adjustment
        return atmVol;
    }

    /**
     * @brief Set smile surface (for future extension)
     * @param strikes Strike levels
     * @param smileAdjustments Smile adjustments (expiry x tenor x strike)
     *
     * Note: This is a placeholder for future smile functionality
     */
    template <typename VectorT, typename SurfaceT>
    void setSmileSurface(const VectorT& strikes, const SurfaceT& smileAdjustments) {
        m_strikes = toVector(strikes);
        // TODO: Store smile adjustments and create interpolator
        m_hasSmile = true;
    }

    /**
     * @brief Get volatility type
     */
    static constexpr IRVolType volType() { return VolType; }

    /**
     * @brief Get descriptor
     */
    const IRVolDescriptor& descriptor() const { return m_descriptor; }

    /**
     * @brief Get expiries
     */
    const std::vector<DoubleT>& expiries() const { return m_expiries; }

    /**
     * @brief Get tenors
     */
    const std::vector<DoubleT>& tenors() const { return m_tenors; }

    /**
     * @brief Get ATM vol surface
     */
    const std::vector<std::vector<DoubleT>>& atmSurface() const { return m_atmVolSurface; }

    /**
     * @brief Scale entire surface by a scalar (multiply each element)
     * @param scalar Scaling factor
     */
    void scale(DoubleT scalar) {
        for (auto& row : m_atmVolSurface) {
            for (auto& vol : row) {
                vol = vol * scalar;
            }
        }
        // Recreate interpolator with scaled surface
        m_atmInterpolator = std::make_unique<Math::BicubicInterpolation<DoubleT>>(
            m_tenors, m_expiries, m_atmVolSurface, Math::CubicInterpolation<DoubleT>::Spline);
    }

    /**
     * @brief Shift entire surface by a constant (add to each element)
     * @param shift Shift amount
     */
    void shift(DoubleT shift) {
        for (auto& row : m_atmVolSurface) {
            for (auto& vol : row) {
                vol = vol + shift;
            }
        }
        // Recreate interpolator with shifted surface
        m_atmInterpolator = std::make_unique<Math::BicubicInterpolation<DoubleT>>(
            m_tenors, m_expiries, m_atmVolSurface, Math::CubicInterpolation<DoubleT>::Spline);
    }

    /**
     * @brief Apply a function to each element of the surface
     * @param func Function to apply: DoubleT f(DoubleT)
     */
    template <typename Func>
    void applyFunction(Func func) {
        for (auto& row : m_atmVolSurface) {
            for (auto& vol : row) {
                vol = func(vol);
            }
        }
        // Recreate interpolator with transformed surface
        m_atmInterpolator = std::make_unique<Math::BicubicInterpolation<DoubleT>>(
            m_tenors, m_expiries, m_atmVolSurface, Math::CubicInterpolation<DoubleT>::Spline);
    }

    /**
     * @brief Bump a specific point in the surface
     * @param expiryIndex Index of expiry to bump
     * @param tenorIndex Index of tenor to bump
     * @param bumpSize Size of bump
     */
    void bump(size_t expiryIndex, size_t tenorIndex, DoubleT bumpSize) {
        if (expiryIndex >= m_atmVolSurface.size()) {
            throw std::runtime_error("IRVolatility::bump: invalid expiry index");
        }
        if (tenorIndex >= m_atmVolSurface[expiryIndex].size()) {
            throw std::runtime_error("IRVolatility::bump: invalid tenor index");
        }
        m_atmVolSurface[expiryIndex][tenorIndex] =
            m_atmVolSurface[expiryIndex][tenorIndex] + bumpSize;
        // Recreate interpolator
        m_atmInterpolator = std::make_unique<Math::BicubicInterpolation<DoubleT>>(
            m_tenors, m_expiries, m_atmVolSurface, Math::CubicInterpolation<DoubleT>::Spline);
    }

    /**
     * @brief Multiply surface by scalar (operator overload)
     */
    IRVolatility& operator*=(DoubleT scalar) {
        scale(scalar);
        return *this;
    }

    /**
     * @brief Add scalar to surface (operator overload)
     */
    IRVolatility& operator+=(DoubleT scalar) {
        shift(scalar);
        return *this;
    }

private:
    IRVolDescriptor m_descriptor;
    std::vector<DoubleT> m_expiries;
    std::vector<DoubleT> m_tenors;
    std::vector<std::vector<DoubleT>> m_atmVolSurface;
    std::unique_ptr<Math::BicubicInterpolation<DoubleT>> m_atmInterpolator;

    // Smile data (for future extension)
    bool m_hasSmile;
    std::vector<DoubleT> m_strikes;

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

#endif // IRVOLATILITY_H
