#ifndef EQD_DESCRIPTOR_H
#define EQD_DESCRIPTOR_H

#include <string>

namespace Markets {

/**
 * @brief Descriptor for equity data and volatility
 *
 * Contains metadata for equity instruments
 */
struct EQDDescriptor {
    std::string ticker;        ///< Ticker symbol (e.g., "AAPL", "SPX")
    std::string exchange;      ///< Exchange code (e.g., "NASDAQ", "NYSE")
    std::string currency;      ///< Currency code (e.g., "USD", "EUR")
    std::string referenceDate; ///< Reference date in YYYY-MM-DD format

    /**
     * @brief Default constructor
     */
    EQDDescriptor() : ticker(""), exchange(""), currency("USD"), referenceDate("") {}

    /**
     * @brief Constructor with all fields
     */
    EQDDescriptor(const std::string& tick, const std::string& exch, const std::string& ccy,
                  const std::string& refDate = "")
        : ticker(tick), exchange(exch), currency(ccy), referenceDate(refDate) {}

    /**
     * @brief Get full identifier
     */
    std::string identifier() const { return ticker + "." + exchange + "." + currency; }
};

} // namespace Markets

#endif // EQD_DESCRIPTOR_H
