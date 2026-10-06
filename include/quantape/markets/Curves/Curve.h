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
    FxSpot,       ///< FX spot factor (one solved value per pair)
    FxVol,        ///< FX volatility factor (delta-space quote per pair)
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
        case CurveRole::FxSpot:
            return "FxSpot";
        case CurveRole::FxVol:
            return "FxVol";
    }
    return "Unknown";
}

/// True for the FX risk roles that report as side factors rather than as
/// curves of the interest-rate tree.
constexpr bool isFxRole(CurveRole role) noexcept {
    return role == CurveRole::FxSpot || role == CurveRole::FxVol;
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
