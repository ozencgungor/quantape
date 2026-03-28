#ifndef EQDDATA_H
#define EQDDATA_H

#include <memory>
#include <stdexcept>

#include "Markets/Curves/IRCurve.h"
#include "Markets/Curves/YieldCurve.h"
#include "Markets/Descriptors/EQDDescriptor.h"

namespace Markets {

/**
 * @brief Equity data container
 *
 * Template class for equity market data compatible with AD.
 * Stores spot price, dividend yield curve, and discount curve.
 *
 * @tparam DoubleT Numeric type (double or stan::math::var for AD)
 */
template <typename DoubleT>
class EQDData {
public:
    /**
     * @brief Constructor with spot and curves
     * @param spot Current spot price
     * @param dividendCurve Dividend yield curve
     * @param discountCurve Discount curve for present value calculations
     * @param descriptor Equity metadata
     */
    EQDData(DoubleT spot, const YieldCurve<DoubleT>& dividendCurve,
            const IRCurve<DoubleT>& discountCurve,
            const EQDDescriptor& descriptor = EQDDescriptor())
        : m_spot(spot), m_descriptor(descriptor),
          m_dividendCurve(std::make_unique<YieldCurve<DoubleT>>(dividendCurve)),
          m_discountCurve(std::make_unique<IRCurve<DoubleT>>(discountCurve)) {
        // Validate spot is positive
        if (value_impl(spot) <= 0.0) {
            throw std::runtime_error("EQDData: spot price must be positive");
        }
    }

    /**
     * @brief Get current spot price
     */
    DoubleT spot() const { return m_spot; }

    /**
     * @brief Set spot price
     */
    void setSpot(DoubleT newSpot) {
        if (value_impl(newSpot) <= 0.0) {
            throw std::runtime_error("EQDData: spot price must be positive");
        }
        m_spot = newSpot;
    }

    /**
     * @brief Get dividend yield at time t
     * @param t Time in years
     * @return Continuously compounded dividend yield
     */
    DoubleT dividendYield(DoubleT t, bool allowExtrapolation = true) const {
        return m_dividendCurve->yield(t, allowExtrapolation);
    }

    /**
     * @brief Get forward price at time t
     * @param t Time in years
     * @return Forward price F(t) = S * exp((r - q) * t)
     *
     * Uses cost-of-carry model: F = S * DF_div(t) / DF_disc(t)
     */
    DoubleT forward(DoubleT t, bool allowExtrapolation = true) const {
        DoubleT dfDiv = m_dividendCurve->discountFactor(t, allowExtrapolation);
        DoubleT dfDisc = m_discountCurve->discountFactor(t, allowExtrapolation);
        // F = S / DF_div * DF_disc = S * exp(q*t) * exp(-r*t)
        return m_spot * dfDisc / dfDiv;
    }

    /**
     * @brief Get discount factor at time t (from discount curve)
     * @param t Time in years
     * @return Discount factor
     */
    DoubleT discountFactor(DoubleT t, bool allowExtrapolation = true) const {
        return m_discountCurve->discountFactor(t, allowExtrapolation);
    }

    /**
     * @brief Get risk-free rate at time t (from discount curve)
     * @param t Time in years
     * @return Zero rate
     */
    DoubleT riskFreeRate(DoubleT t, bool allowExtrapolation = true) const {
        return m_discountCurve->zeroRate(t, allowExtrapolation);
    }

    /**
     * @brief Calculate present value of a future cash flow
     * @param cashflow Future cash flow amount
     * @param t Time of cash flow in years
     * @return Present value = cashflow * DF(t)
     */
    DoubleT presentValue(DoubleT cashflow, DoubleT t, bool allowExtrapolation = true) const {
        DoubleT df = discountFactor(t, allowExtrapolation);
        return cashflow * df;
    }

    /**
     * @brief Get descriptor
     */
    const EQDDescriptor& descriptor() const { return m_descriptor; }

    /**
     * @brief Get dividend curve
     */
    const YieldCurve<DoubleT>& dividendCurve() const { return *m_dividendCurve; }

    /**
     * @brief Get discount curve
     */
    const IRCurve<DoubleT>& discountCurve() const { return *m_discountCurve; }

private:
    DoubleT m_spot;
    EQDDescriptor m_descriptor;
    std::unique_ptr<YieldCurve<DoubleT>> m_dividendCurve;
    std::unique_ptr<IRCurve<DoubleT>> m_discountCurve;

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

#endif // EQDDATA_H
