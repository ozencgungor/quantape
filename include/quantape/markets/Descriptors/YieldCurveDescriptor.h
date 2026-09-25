#ifndef YIELDCURVE_DESCRIPTOR_H
#define YIELDCURVE_DESCRIPTOR_H

#include <string>

namespace quantape::markets {

/**
 * @brief Descriptor for yield curves
 *
 * Contains metadata for yield curves (dividends, repo, etc.)
 */
struct YieldCurveDescriptor {
    std::string currency;      ///< Currency code (e.g., "USD", "EUR")
    std::string curveName;     ///< Curve identifier (e.g., "DIV", "REPO")
    std::string referenceDate; ///< Reference date in YYYY-MM-DD format
    std::string curveType;     ///< Type: "DIVIDEND", "REPO", etc.

    /**
     * @brief Default constructor
     */
    YieldCurveDescriptor() : currency("USD"), curveName(""), referenceDate(""), curveType("") {}

    /**
     * @brief Constructor with all fields
     */
    YieldCurveDescriptor(const std::string& ccy, const std::string& name,
                         const std::string& refDate = "", const std::string& type = "")
        : currency(ccy), curveName(name), referenceDate(refDate), curveType(type) {}

    /**
     * @brief Get full curve identifier
     */
    std::string identifier() const { return currency + "." + curveName + "." + referenceDate; }
};

} // namespace quantape::markets

#endif // YIELDCURVE_DESCRIPTOR_H
