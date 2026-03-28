#ifndef FX_DESCRIPTOR_H
#define FX_DESCRIPTOR_H

#include <string>

namespace Markets {

/**
 * @brief Descriptor for FX rates and volatility
 *
 * Contains metadata for FX instruments
 */
struct FXDescriptor {
    std::string domesticCcy;   ///< Domestic currency (e.g., "USD")
    std::string foreignCcy;    ///< Foreign currency (e.g., "EUR")
    std::string referenceDate; ///< Reference date in YYYY-MM-DD format

    /**
     * @brief Default constructor
     */
    FXDescriptor() : domesticCcy("USD"), foreignCcy("EUR"), referenceDate("") {}

    /**
     * @brief Constructor with all fields
     */
    FXDescriptor(const std::string& domCcy, const std::string& forCcy,
                 const std::string& refDate = "")
        : domesticCcy(domCcy), foreignCcy(forCcy), referenceDate(refDate) {}

    /**
     * @brief Get currency pair identifier
     */
    std::string pair() const { return domesticCcy + foreignCcy; }

    /**
     * @brief Get full identifier
     */
    std::string identifier() const { return domesticCcy + foreignCcy + "." + referenceDate; }
};

} // namespace Markets

#endif // FX_DESCRIPTOR_H
