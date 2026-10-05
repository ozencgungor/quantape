#ifndef FX_DESCRIPTOR_H
#define FX_DESCRIPTOR_H

#include <string>
#include <unordered_set>

namespace quantape::markets {

/**
 * @brief Check whether a currency code is supported
 *
 * Accepts the ISO 4217 codes in the supported FX currency set.
 */
inline bool isKnownCurrency(const std::string& code) {
    static const std::unordered_set<std::string> known = {
        "AED", "ARS", "AUD", "BGN", "BHD", "BRL", "CAD", "CHF", "CLP", "CNY", "COP", "CZK", "DKK",
        "EGP", "EUR", "GBP", "GHS", "HKD", "HRK", "HUF", "IDR", "ILS", "INR", "ISK", "JOD", "JPY",
        "KES", "KRW", "KWD", "MAD", "MXN", "MYR", "NGN", "NOK", "NZD", "OMR", "PEN", "PHP", "PLN",
        "QAR", "RON", "RUB", "SAR", "SEK", "SGD", "THB", "TND", "TRY", "TWD", "USD", "VND", "ZAR"};
    return known.find(code) != known.end();
}

/**
 * @brief Descriptor for FX rates and volatility
 *
 * Pair semantics are base/quote: a spot rate is expressed in quote-currency
 * units per one unit of the base currency. `pair()` returns base + quote, so
 * an EUR/USD rate quoted at 1.10 reads "EURUSD" and means 1.10 USD per EUR.
 */
struct FXDescriptor {
    std::string baseCcy;       ///< Base currency, the currency being priced (e.g., "EUR")
    std::string quoteCcy;      ///< Quote currency, the price currency (e.g., "USD")
    std::string referenceDate; ///< Reference date in YYYY-MM-DD format

    /**
     * @brief Default constructor
     */
    FXDescriptor() : baseCcy("EUR"), quoteCcy("USD"), referenceDate("") {}

    /**
     * @brief Constructor with all fields
     *
     * @param base Base currency code
     * @param quote Quote currency code
     * @param refDate Reference date in YYYY-MM-DD format
     */
    FXDescriptor(const std::string& base, const std::string& quote, const std::string& refDate = "")
        : baseCcy(base), quoteCcy(quote), referenceDate(refDate) {}

    /**
     * @brief Get currency pair identifier in base/quote order (e.g., "EURUSD")
     */
    std::string pair() const { return baseCcy + quoteCcy; }

    /**
     * @brief Get full identifier (e.g., "EURUSD.2026-09-29")
     */
    std::string identifier() const { return baseCcy + quoteCcy + "." + referenceDate; }

    /**
     * @brief Check both currency codes against the supported set
     */
    bool hasKnownCurrencies() const {
        return isKnownCurrency(baseCcy) && isKnownCurrency(quoteCcy);
    }

    /**
     * @deprecated The old quote-then-base ordering made the "domestic"
     * currency the quote currency. Use `quoteCcy` directly.
     */
    [[deprecated("domesticCcy was the quote currency; use quoteCcy")]]
    const std::string& domesticCcy() const {
        return quoteCcy;
    }

    /**
     * @deprecated The old quote-then-base ordering made the "foreign"
     * currency the base currency. Use `baseCcy` directly.
     */
    [[deprecated("foreignCcy was the base currency; use baseCcy")]]
    const std::string& foreignCcy() const {
        return baseCcy;
    }
};

} // namespace quantape::markets

#endif // FX_DESCRIPTOR_H
