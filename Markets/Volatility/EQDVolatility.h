#ifndef EQDVOLATILITY_H
#define EQDVOLATILITY_H

#include "Math/Interpolations/BicubicInterpolation.h"

#include <memory>
#include <stdexcept>
#include <vector>

#include "Markets/Descriptors/EQDDescriptor.h"

namespace Markets {

/**
 * @brief Equity volatility surface
 *
 * Template class for equity volatility surfaces compatible with AD.
 * Stores implied volatilities as a 2D surface (expiry x strike).
 *
 * @tparam DoubleT Numeric type (double or stan::math::var for AD)
 * @tparam ContainerT Container type for vol surface (default: vector<vector<DoubleT>>)
 */
template <typename DoubleT, typename ContainerT = std::vector<std::vector<DoubleT>>>
class EQDVolatility {
public:
    /**
     * @brief Constructor with vol surface
     * @param expiries Option expiries (in years)
     * @param strikes Strike levels (can be absolute or moneyness)
     * @param volSurface Implied volatilities (expiries x strikes)
     * @param refSpot Reference spot price
     * @param descriptor Vol surface metadata
     * @param strikeType Type of strikes: "ABSOLUTE" or "MONEYNESS"
     */
    template <typename VectorT, typename SurfaceT>
    EQDVolatility(const VectorT& expiries, const VectorT& strikes, const SurfaceT& volSurface,
                  DoubleT refSpot, const EQDDescriptor& descriptor = EQDDescriptor(),
                  const std::string& strikeType = "ABSOLUTE")
        : m_descriptor(descriptor), m_refSpot(refSpot), m_strikeType(strikeType) {
        // Convert to vectors
        m_expiries = toVector(expiries);
        m_strikes = toVector(strikes);
        m_volSurface = toSurface(volSurface);

        // Validate dimensions
        if (m_volSurface.size() != m_expiries.size()) {
            throw std::runtime_error(
                "EQDVolatility: volSurface rows must match expiries size. Got " +
                std::to_string(m_volSurface.size()) + " rows, expected " +
                std::to_string(m_expiries.size()));
        }
        for (const auto& row : m_volSurface) {
            if (row.size() != m_strikes.size()) {
                throw std::runtime_error(
                    "EQDVolatility: volSurface columns must match strikes size. Got " +
                    std::to_string(row.size()) + " columns, expected " +
                    std::to_string(m_strikes.size()));
            }
        }

        // Create interpolator for vol surface (use Bicubic with Spline)
        // Note: BicubicInterpolation expects (x, y, z) where z[y_index][x_index]
        // So we pass (strikes, expiries, z) since our z is organized as z[expiry][strike]
        m_interpolator = std::make_unique<Math::BicubicInterpolation<DoubleT>>(
            m_strikes, m_expiries, m_volSurface, Math::CubicInterpolation<DoubleT>::Spline);
    }

    /**
     * @brief Copy constructor
     */
    EQDVolatility(const EQDVolatility& other)
        : m_descriptor(other.m_descriptor), m_refSpot(other.m_refSpot),
          m_strikeType(other.m_strikeType), m_expiries(other.m_expiries),
          m_strikes(other.m_strikes), m_volSurface(other.m_volSurface) {
        m_interpolator = std::make_unique<Math::BicubicInterpolation<DoubleT>>(
            m_strikes, m_expiries, m_volSurface, Math::CubicInterpolation<DoubleT>::Spline);
    }

    /**
     * @brief Copy assignment
     */
    EQDVolatility& operator=(const EQDVolatility& other) {
        if (this != &other) {
            m_descriptor = other.m_descriptor;
            m_refSpot = other.m_refSpot;
            m_strikeType = other.m_strikeType;
            m_expiries = other.m_expiries;
            m_strikes = other.m_strikes;
            m_volSurface = other.m_volSurface;
            m_interpolator = std::make_unique<Math::BicubicInterpolation<DoubleT>>(
                m_strikes, m_expiries, m_volSurface, Math::CubicInterpolation<DoubleT>::Spline);
        }
        return *this;
    }

    /**
     * @brief Get volatility at given expiry and strike
     * @param expiry Option expiry in years
     * @param strike Strike level
     * @param allowExtrapolation Allow extrapolation beyond surface range
     * @return Implied volatility
     */
    DoubleT vol(DoubleT expiry, DoubleT strike, bool allowExtrapolation = true) const {
        // Note: interpolator is (x=strikes, y=expiries), so we call it with (strike, expiry)
        return (*m_interpolator)(strike, expiry, allowExtrapolation);
    }

    /**
     * @brief Get volatility at given expiry and moneyness
     * @param expiry Option expiry in years
     * @param moneyness Strike/Spot ratio
     * @param spot Current spot price
     * @param allowExtrapolation Allow extrapolation beyond surface range
     * @return Implied volatility
     */
    DoubleT volByMoneyness(DoubleT expiry, DoubleT moneyness, DoubleT spot,
                           bool allowExtrapolation = true) const {
        if (m_strikeType == "MONEYNESS") {
            // Strikes are already in moneyness terms
            return vol(expiry, moneyness, allowExtrapolation);
        } else {
            // Convert moneyness to absolute strike
            DoubleT strike = moneyness * spot;
            return vol(expiry, strike, allowExtrapolation);
        }
    }

    /**
     * @brief Get volatility at given expiry and log-moneyness
     * @param expiry Option expiry in years
     * @param logMoneyness ln(Strike/Spot)
     * @param spot Current spot price
     * @param allowExtrapolation Allow extrapolation beyond surface range
     * @return Implied volatility
     */
    DoubleT volByLogMoneyness(DoubleT expiry, DoubleT logMoneyness, DoubleT spot,
                              bool allowExtrapolation = true) const {
        DoubleT moneyness = exp_impl(logMoneyness);
        return volByMoneyness(expiry, moneyness, spot, allowExtrapolation);
    }

    /**
     * @brief Get descriptor
     */
    const EQDDescriptor& descriptor() const { return m_descriptor; }

    /**
     * @brief Get reference spot
     */
    DoubleT referenceSpot() const { return m_refSpot; }

    /**
     * @brief Get strike type
     */
    const std::string& strikeType() const { return m_strikeType; }

    /**
     * @brief Get expiries
     */
    const std::vector<DoubleT>& expiries() const { return m_expiries; }

    /**
     * @brief Get strikes
     */
    const std::vector<DoubleT>& strikes() const { return m_strikes; }

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
            m_strikes, m_expiries, m_volSurface, Math::CubicInterpolation<DoubleT>::Spline);
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
            m_strikes, m_expiries, m_volSurface, Math::CubicInterpolation<DoubleT>::Spline);
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
            m_strikes, m_expiries, m_volSurface, Math::CubicInterpolation<DoubleT>::Spline);
    }

    /**
     * @brief Bump a specific point in the surface
     * @param expiryIndex Index of expiry to bump
     * @param strikeIndex Index of strike to bump
     * @param bumpSize Size of bump
     */
    void bump(size_t expiryIndex, size_t strikeIndex, DoubleT bumpSize) {
        if (expiryIndex >= m_volSurface.size()) {
            throw std::runtime_error("EQDVolatility::bump: invalid expiry index");
        }
        if (strikeIndex >= m_volSurface[expiryIndex].size()) {
            throw std::runtime_error("EQDVolatility::bump: invalid strike index");
        }
        m_volSurface[expiryIndex][strikeIndex] = m_volSurface[expiryIndex][strikeIndex] + bumpSize;
        // Recreate interpolator
        m_interpolator = std::make_unique<Math::BicubicInterpolation<DoubleT>>(
            m_strikes, m_expiries, m_volSurface, Math::CubicInterpolation<DoubleT>::Spline);
    }

    /**
     * @brief Apply element-wise operation with coordinates
     * @param func Function to apply: DoubleT f(DoubleT vol, DoubleT expiry, DoubleT strike)
     *
     * Useful for operations that depend on position in the surface
     */
    template <typename Func>
    void applyFunctionWithCoords(Func func) {
        for (size_t i = 0; i < m_volSurface.size(); ++i) {
            for (size_t j = 0; j < m_volSurface[i].size(); ++j) {
                m_volSurface[i][j] = func(m_volSurface[i][j], m_expiries[i], m_strikes[j]);
            }
        }
        // Recreate interpolator
        m_interpolator = std::make_unique<Math::BicubicInterpolation<DoubleT>>(
            m_strikes, m_expiries, m_volSurface, Math::CubicInterpolation<DoubleT>::Spline);
    }

    /**
     * @brief Multiply surface by scalar (operator overload)
     */
    EQDVolatility& operator*=(DoubleT scalar) {
        scale(scalar);
        return *this;
    }

    /**
     * @brief Add scalar to surface (operator overload)
     */
    EQDVolatility& operator+=(DoubleT scalar) {
        shift(scalar);
        return *this;
    }

private:
    EQDDescriptor m_descriptor;
    DoubleT m_refSpot;
    std::string m_strikeType;
    std::vector<DoubleT> m_expiries;
    std::vector<DoubleT> m_strikes;
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

    /**
     * @brief AD-compatible exponential
     */
    static DoubleT exp_impl(const DoubleT& x) {
        if constexpr (std::is_same_v<DoubleT, double>) {
            return std::exp(x);
        } else {
            using std::exp;
            return exp(x); // ADL for stan::math::exp
        }
    }
};

} // namespace Markets

#endif // EQDVOLATILITY_H
