#ifndef IRVOL_DESCRIPTOR_H
#define IRVOL_DESCRIPTOR_H

#include <string>

namespace Markets {

/**
 * @brief Descriptor for interest rate volatility surfaces
 *
 * Contains metadata for IR vol surfaces
 */
struct IRVolDescriptor {
    std::string currency;      ///< Currency code (e.g., "USD", "EUR")
    std::string volType;       ///< Volatility type (e.g., "SWAPTION", "CAPFLOOR")
    std::string referenceDate; ///< Reference date in YYYY-MM-DD format
    std::string index;         ///< Underlying index (e.g., "LIBOR3M", "SOFR")

    /**
     * @brief Default constructor
     */
    IRVolDescriptor() : currency("USD"), volType("SWAPTION"), referenceDate(""), index("") {}

    /**
     * @brief Constructor with all fields
     */
    IRVolDescriptor(const std::string& ccy, const std::string& type, const std::string& refDate,
                    const std::string& idx = "")
        : currency(ccy), volType(type), referenceDate(refDate), index(idx) {}

    /**
     * @brief Get full vol identifier
     */
    std::string identifier() const { return currency + "." + volType + "." + referenceDate; }
};

} // namespace Markets

#endif // IRVOL_DESCRIPTOR_H
