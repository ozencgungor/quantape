#ifndef FXRATE_H
#define FXRATE_H

#include "Markets/Curves/IRCurve.h"
#include "Markets/Descriptors/FXDescriptor.h"
#include <memory>
#include <stdexcept>
#include <optional>

namespace Markets {

/**
 * @brief FX rate data container
 *
 * Template class for FX market data compatible with AD.
 * Stores spot FX rate and optional interest rate curves for both currencies.
 * Convention: FX rate is quoted as Domestic/Foreign (e.g., USDEUR = USD per 1 EUR)
 *
 * @tparam DoubleT Numeric type (double or stan::math::var for AD)
 */
template <typename DoubleT>
class FXRate {
public:
    /**
     * @brief Constructor with spot rate only
     * @param spot FX spot rate (Domestic/Foreign)
     * @param descriptor FX metadata
     */
    FXRate(DoubleT spot, const FXDescriptor& descriptor = FXDescriptor())
        : m_spot(spot), m_descriptor(descriptor), m_hasCurves(false) {
        // Validate spot is positive
        if (value_impl(spot) <= 0.0) {
            throw std::runtime_error("FXRate: spot rate must be positive");
        }
    }

    /**
     * @brief Constructor with spot rate and curves
     * @param spot FX spot rate (Domestic/Foreign)
     * @param domesticCurve Domestic currency interest rate curve
     * @param foreignCurve Foreign currency interest rate curve
     * @param descriptor FX metadata
     */
    FXRate(DoubleT spot, const IRCurve<DoubleT>& domesticCurve,
           const IRCurve<DoubleT>& foreignCurve,
           const FXDescriptor& descriptor = FXDescriptor())
        : m_spot(spot),
          m_descriptor(descriptor),
          m_hasCurves(true),
          m_domesticCurve(std::make_unique<IRCurve<DoubleT>>(domesticCurve)),
          m_foreignCurve(std::make_unique<IRCurve<DoubleT>>(foreignCurve)) {
        // Validate spot is positive
        if (value_impl(spot) <= 0.0) {
            throw std::runtime_error("FXRate: spot rate must be positive");
        }
    }

    /**
     * @brief Get spot FX rate
     */
    DoubleT spot() const { return m_spot; }

    /**
     * @brief Set spot FX rate
     */
    void setSpot(DoubleT newSpot) {
        if (value_impl(newSpot) <= 0.0) {
            throw std::runtime_error("FXRate: spot rate must be positive");
        }
        m_spot = newSpot;
    }

    /**
     * @brief Get forward FX rate at time t
     * @param t Time in years
     * @return Forward FX rate F(t) = S * exp((rd - rf) * t)
     *
     * Uses covered interest rate parity:
     * F(t) = S * DF_foreign(t) / DF_domestic(t)
     *
     * @throws std::runtime_error if curves are not set
     */
    DoubleT forward(DoubleT t, bool allowExtrapolation = true) const {
        if (!m_hasCurves) {
            throw std::runtime_error(
                "FXRate::forward: curves not set. Use constructor with curves "
                "or call setCurves().");
        }

        DoubleT dfDom = m_domesticCurve->discountFactor(t, allowExtrapolation);
        DoubleT dfFor = m_foreignCurve->discountFactor(t, allowExtrapolation);

        // F = S * DF_for / DF_dom = S * exp((rd - rf) * t)
        return m_spot * dfFor / dfDom;
    }

    /**
     * @brief Get domestic discount factor at time t
     * @param t Time in years
     * @return Domestic discount factor
     *
     * @throws std::runtime_error if curves are not set
     */
    DoubleT domesticDiscountFactor(DoubleT t,
                                   bool allowExtrapolation = true) const {
        if (!m_hasCurves) {
            throw std::runtime_error(
                "FXRate::domesticDiscountFactor: curves not set");
        }
        return m_domesticCurve->discountFactor(t, allowExtrapolation);
    }

    /**
     * @brief Get foreign discount factor at time t
     * @param t Time in years
     * @return Foreign discount factor
     *
     * @throws std::runtime_error if curves are not set
     */
    DoubleT foreignDiscountFactor(DoubleT t,
                                  bool allowExtrapolation = true) const {
        if (!m_hasCurves) {
            throw std::runtime_error(
                "FXRate::foreignDiscountFactor: curves not set");
        }
        return m_foreignCurve->discountFactor(t, allowExtrapolation);
    }

    /**
     * @brief Get domestic zero rate at time t
     * @param t Time in years
     * @return Domestic zero rate
     */
    DoubleT domesticRate(DoubleT t, bool allowExtrapolation = true) const {
        if (!m_hasCurves) {
            throw std::runtime_error("FXRate::domesticRate: curves not set");
        }
        return m_domesticCurve->zeroRate(t, allowExtrapolation);
    }

    /**
     * @brief Get foreign zero rate at time t
     * @param t Time in years
     * @return Foreign zero rate
     */
    DoubleT foreignRate(DoubleT t, bool allowExtrapolation = true) const {
        if (!m_hasCurves) {
            throw std::runtime_error("FXRate::foreignRate: curves not set");
        }
        return m_foreignCurve->zeroRate(t, allowExtrapolation);
    }

    /**
     * @brief Set interest rate curves
     */
    void setCurves(const IRCurve<DoubleT>& domesticCurve,
                   const IRCurve<DoubleT>& foreignCurve) {
        m_domesticCurve = std::make_unique<IRCurve<DoubleT>>(domesticCurve);
        m_foreignCurve = std::make_unique<IRCurve<DoubleT>>(foreignCurve);
        m_hasCurves = true;
    }

    /**
     * @brief Check if curves are set
     */
    bool hasCurves() const { return m_hasCurves; }

    /**
     * @brief Get descriptor
     */
    const FXDescriptor& descriptor() const { return m_descriptor; }

    /**
     * @brief Get domestic curve (if available)
     */
    const IRCurve<DoubleT>* domesticCurve() const {
        return m_hasCurves ? m_domesticCurve.get() : nullptr;
    }

    /**
     * @brief Get foreign curve (if available)
     */
    const IRCurve<DoubleT>* foreignCurve() const {
        return m_hasCurves ? m_foreignCurve.get() : nullptr;
    }

private:
    DoubleT m_spot;
    FXDescriptor m_descriptor;
    bool m_hasCurves;
    std::unique_ptr<IRCurve<DoubleT>> m_domesticCurve;
    std::unique_ptr<IRCurve<DoubleT>> m_foreignCurve;

    /**
     * @brief Extract value for validation (handles both double and AD types)
     */
    static double value_impl(const DoubleT& x) {
        if constexpr (std::is_same_v<DoubleT, double>) {
            return x;
        } else {
            return x.val(); // For stan::math::var
        }
    }
};

} // namespace Markets

#endif // FXRATE_H
