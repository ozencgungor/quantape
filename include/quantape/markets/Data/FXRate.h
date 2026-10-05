#ifndef FXRATE_H
#define FXRATE_H

#include "quantape/markets/Curves/DiscountCurve.h"
#include "quantape/markets/Descriptors/FXDescriptor.h"

#include <memory>
#include <optional>
#include <stdexcept>
#include <type_traits>

namespace quantape::markets {

/**
 * @brief FX rate data container
 *
 * Template class for FX market data compatible with AD.
 * Stores spot FX rate and optional interest rate curves for both currencies.
 * Convention: the rate is quoted as quote-currency units per one unit of the
 * base currency (e.g., EURUSD = USD per 1 EUR).
 *
 * @tparam DoubleT Numeric type (double or stan::math::var for AD)
 */
template <typename DoubleT>
class FXRate {
public:
    /**
     * @brief Constructor with spot rate only
     * @param spot FX spot rate (quote units per 1 base)
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
     * @param spot FX spot rate (quote units per 1 base)
     * @param baseCurve Base-currency interest rate curve
     * @param quoteCurve Quote-currency interest rate curve
     * @param descriptor FX metadata
     */
    FXRate(DoubleT spot, const DiscountCurve<DoubleT>& baseCurve,
           const DiscountCurve<DoubleT>& quoteCurve,
           const FXDescriptor& descriptor = FXDescriptor())
        : m_spot(spot), m_descriptor(descriptor), m_hasCurves(true),
          m_baseCurve(std::make_unique<DiscountCurve<DoubleT>>(baseCurve)),
          m_quoteCurve(std::make_unique<DiscountCurve<DoubleT>>(quoteCurve)) {
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
     * @param allowExtrapolation If true, extrapolate beyond the curve range
     *
     * @return Forward FX rate F(t)
     *
     * Uses covered interest rate parity:
     * F(t) = S * DF_base(t) / DF_quote(t)
     *
     * @throws std::runtime_error if curves are not set
     */
    DoubleT forward(DoubleT t, bool allowExtrapolation = true) const {
        if (!m_hasCurves) {
            throw std::runtime_error("FXRate::forward: curves not set. Use constructor with curves "
                                     "or call setCurves().");
        }

        DoubleT dfBase = m_baseCurve->discountFactor(t, allowExtrapolation);
        DoubleT dfQuote = m_quoteCurve->discountFactor(t, allowExtrapolation);

        // F = S * DF_base / DF_quote
        return m_spot * dfBase / dfQuote;
    }

    /**
     * @brief Get base-currency discount factor at time t
     * @param t Time in years
     * @param allowExtrapolation If true, extrapolate beyond the curve range
     *
     * @return Base-currency discount factor
     *
     * @throws std::runtime_error if curves are not set
     */
    DoubleT baseDiscountFactor(DoubleT t, bool allowExtrapolation = true) const {
        if (!m_hasCurves) {
            throw std::runtime_error("FXRate::baseDiscountFactor: curves not set");
        }
        return m_baseCurve->discountFactor(t, allowExtrapolation);
    }

    /**
     * @brief Get quote-currency discount factor at time t
     * @param t Time in years
     * @param allowExtrapolation If true, extrapolate beyond the curve range
     *
     * @return Quote-currency discount factor
     *
     * @throws std::runtime_error if curves are not set
     */
    DoubleT quoteDiscountFactor(DoubleT t, bool allowExtrapolation = true) const {
        if (!m_hasCurves) {
            throw std::runtime_error("FXRate::quoteDiscountFactor: curves not set");
        }
        return m_quoteCurve->discountFactor(t, allowExtrapolation);
    }

    /**
     * @brief Get base-currency zero rate at time t
     * @param t Time in years
     * @param allowExtrapolation If true, extrapolate beyond the curve range
     *
     * @return Base-currency zero rate
     */
    DoubleT baseRate(DoubleT t, bool allowExtrapolation = true) const {
        if (!m_hasCurves) {
            throw std::runtime_error("FXRate::baseRate: curves not set");
        }
        return m_baseCurve->zeroRate(t, allowExtrapolation);
    }

    /**
     * @brief Get quote-currency zero rate at time t
     * @param t Time in years
     * @param allowExtrapolation If true, extrapolate beyond the curve range
     *
     * @return Quote-currency zero rate
     */
    DoubleT quoteRate(DoubleT t, bool allowExtrapolation = true) const {
        if (!m_hasCurves) {
            throw std::runtime_error("FXRate::quoteRate: curves not set");
        }
        return m_quoteCurve->zeroRate(t, allowExtrapolation);
    }

    /**
     * @brief Set interest rate curves
     * @param baseCurve Base-currency interest rate curve
     * @param quoteCurve Quote-currency interest rate curve
     */
    void setCurves(const DiscountCurve<DoubleT>& baseCurve,
                   const DiscountCurve<DoubleT>& quoteCurve) {
        m_baseCurve = std::make_unique<DiscountCurve<DoubleT>>(baseCurve);
        m_quoteCurve = std::make_unique<DiscountCurve<DoubleT>>(quoteCurve);
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
     * @brief Get base curve (if available)
     */
    const DiscountCurve<DoubleT>* baseCurve() const {
        return m_hasCurves ? m_baseCurve.get() : nullptr;
    }

    /**
     * @brief Get quote curve (if available)
     */
    const DiscountCurve<DoubleT>* quoteCurve() const {
        return m_hasCurves ? m_quoteCurve.get() : nullptr;
    }

    /**
     * @deprecated The domestic currency is the quote currency after the
     * base/quote correction. Use quoteDiscountFactor().
     */
    [[deprecated("use quoteDiscountFactor; the domestic currency is the quote currency")]]
    DoubleT domesticDiscountFactor(DoubleT t, bool allowExtrapolation = true) const {
        return quoteDiscountFactor(t, allowExtrapolation);
    }

    /**
     * @deprecated The foreign currency is the base currency after the
     * base/quote correction. Use baseDiscountFactor().
     */
    [[deprecated("use baseDiscountFactor; the foreign currency is the base currency")]]
    DoubleT foreignDiscountFactor(DoubleT t, bool allowExtrapolation = true) const {
        return baseDiscountFactor(t, allowExtrapolation);
    }

    /**
     * @deprecated The domestic currency is the quote currency after the
     * base/quote correction. Use quoteRate().
     */
    [[deprecated("use quoteRate; the domestic currency is the quote currency")]]
    DoubleT domesticRate(DoubleT t, bool allowExtrapolation = true) const {
        return quoteRate(t, allowExtrapolation);
    }

    /**
     * @deprecated The foreign currency is the base currency after the
     * base/quote correction. Use baseRate().
     */
    [[deprecated("use baseRate; the foreign currency is the base currency")]]
    DoubleT foreignRate(DoubleT t, bool allowExtrapolation = true) const {
        return baseRate(t, allowExtrapolation);
    }

    /**
     * @deprecated The domestic currency is the quote currency after the
     * base/quote correction. Use quoteCurve().
     */
    [[deprecated("use quoteCurve; the domestic currency is the quote currency")]]
    const DiscountCurve<DoubleT>* domesticCurve() const {
        return quoteCurve();
    }

    /**
     * @deprecated The foreign currency is the base currency after the
     * base/quote correction. Use baseCurve().
     */
    [[deprecated("use baseCurve; the foreign currency is the base currency")]]
    const DiscountCurve<DoubleT>* foreignCurve() const {
        return baseCurve();
    }

private:
    DoubleT m_spot;
    FXDescriptor m_descriptor;
    bool m_hasCurves;
    std::unique_ptr<DiscountCurve<DoubleT>> m_baseCurve;
    std::unique_ptr<DiscountCurve<DoubleT>> m_quoteCurve;

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

} // namespace quantape::markets

#endif // FXRATE_H
