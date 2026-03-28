#ifndef IRCURVE_DESCRIPTOR_H
#define IRCURVE_DESCRIPTOR_H

#include <string>

namespace Markets {

/**
 * @brief Descriptor for interest rate curves
 *
 * Contains metadata and identification information for IR curves
 */
struct IRCurveDescriptor {
    std::string currency;       ///< Currency code (e.g., "USD", "EUR")
    std::string curveName;      ///< Curve identifier (e.g., "OIS", "LIBOR3M")
    std::string referenceDate;  ///< Reference date in YYYY-MM-DD format

    /**
     * @brief Default constructor
     */
    IRCurveDescriptor() : currency("USD"), curveName(""), referenceDate("") {}

    /**
     * @brief Constructor with all fields
     */
    IRCurveDescriptor(const std::string& ccy, const std::string& name,
                      const std::string& refDate = "")
        : currency(ccy), curveName(name), referenceDate(refDate) {}

    /**
     * @brief Get full curve identifier
     */
    std::string identifier() const {
        return currency + "." + curveName + "." + referenceDate;
    }
};

} // namespace Markets

#endif // IRCURVE_DESCRIPTOR_H
