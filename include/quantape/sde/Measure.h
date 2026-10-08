#pragma once

#include "quantape/markets/Curves/Curve.h"

#include <cstdint>
#include <string>
#include <vector>

namespace quantape::sde {
/**
 * @file Measure.h
 * @brief Measure and collateral specifications for pricing requests
 *
 * Plain data types describing the pricing measure and the collateral curve
 * used for discounting. No behaviour lives here.
 */

/// Discount curve referenced by full key, with an optional cross-currency
/// pair when collateral is posted in a different currency.
struct CollateralSpec {
    markets::CurveKey key;
    bool hasFxPair = false;
    std::string fxPair;
};

/// Pricing measure under which cashflows are valued.
enum class MeasureKind : std::uint8_t {
    RiskNeutral,        ///< Risk-neutral measure
    TForward,           ///< Forward measure to a single payment time
    Annuity,            ///< Annuity (swap) measure
    CollateralCurrency, ///< Measure induced by the collateral currency
};

/// Measure selection with the data each measure kind needs.
struct MeasureSpec {
    MeasureKind kind = MeasureKind::RiskNeutral;
    double forwardTime = 0.0;
    std::vector<double> annuityWeights;
};

} // namespace quantape::sde
