#ifndef CREDIT_DESCRIPTOR_H
#define CREDIT_DESCRIPTOR_H

#include <string>

namespace Markets {

/**
 * @brief Descriptor for credit curves
 *
 * Contains metadata for survival probability curves
 */
struct CreditDescriptor {
    std::string issuerName;    ///< Name of the issuer
    std::string currency;      ///< Currency code (e.g., "USD", "EUR")
    std::string seniority;     ///< Debt seniority (e.g., "SENIOR", "SUBORDINATED")
    std::string referenceDate; ///< Reference date in YYYY-MM-DD format

    /**
     * @brief Default constructor
     */
    CreditDescriptor() : issuerName(""), currency("USD"), seniority("SENIOR"), referenceDate("") {}

    /**
     * @brief Constructor with all fields
     */
    CreditDescriptor(const std::string& issuer, const std::string& ccy, const std::string& sen,
                     const std::string& refDate = "")
        : issuerName(issuer), currency(ccy), seniority(sen), referenceDate(refDate) {}

    /**
     * @brief Get full curve identifier
     */
    std::string identifier() const {
        return issuerName + "." + currency + "." + seniority + "." + referenceDate;
    }
};

} // namespace Markets

#endif // CREDIT_DESCRIPTOR_H
