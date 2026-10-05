#pragma once

#include "quantape/datetime/Period.h"

#include <string>
#include <string_view>

namespace quantape::markets {
/**
 * @file Curve.h
 * @brief Curve identity types for the multi-currency stack
 *
 * Roles and keys are data: a curve is described by currency, role,
 * index tenor and collateral currency. No curve behaviour lives here.
 */

/// Role of a curve inside the stack.
enum class CurveRole {
    Discount,     ///< Collateral discount curve (root of a currency tree)
    Forecast,     ///< Forwarding curve for an index (IBOR / term RFR)
    TenorBasis,   ///< Tenor basis curve over a parent
    IborOisBasis, ///< IBOR-OIS basis curve over a parent
    XccyBasis,    ///< Cross-currency basis curve over a parent
    TurnOverlay,  ///< Turn-of-year risk factor (overlay amplitude or turn quote)
};

constexpr std::string_view curveRoleName(CurveRole role) noexcept {
    switch (role) {
        case CurveRole::Discount:
            return "Discount";
        case CurveRole::Forecast:
            return "Forecast";
        case CurveRole::TenorBasis:
            return "TenorBasis";
        case CurveRole::IborOisBasis:
            return "IborOisBasis";
        case CurveRole::XccyBasis:
            return "XccyBasis";
        case CurveRole::TurnOverlay:
            return "TurnOverlay";
    }
    return "Unknown";
}

/// Stable identity of a curve in a `MultiCurveSet`.
struct CurveKey {
    std::string currency;
    CurveRole role = CurveRole::Discount;
    datetime::Period indexTenor{1, datetime::TimeUnit::Days};
    std::string collateral;

    friend bool operator==(const CurveKey&, const CurveKey&) = default;
};

} // namespace quantape::markets
