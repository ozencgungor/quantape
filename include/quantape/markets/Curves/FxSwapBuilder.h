#pragma once

#include "quantape/datetime/Date.h"
#include "quantape/datetime/DayCounter.h"
#include "quantape/datetime/TimeConversion.h"
#include "quantape/markets/Curves/BootstrapInstrument.h"
#include "quantape/markets/Curves/DiscountCurve.h"
#include "quantape/markets/Data/FxQuote.h"

#include <cstddef>
#include <stdexcept>
#include <vector>

namespace quantape::markets {
/**
 * @file FxSwapBuilder.h
 * @brief FX forward-points pillars and the collateral discount bootstrap
 *
 * An FX swap pillar carries the near (spot) settlement date, the far maturity,
 * the spot rate and the quoted forward points or far outright. The far
 * outright follows covered interest parity with the near leg pinned to spot,
 *
 *   `F(T) = S D_base(T) / D_quote(T)`,
 *
 * where the correct base/quote assignment depends on which leg is collateral:
 *
 *   - base collateral:  `F(T) = S D_domestic(T) / D_foreign(T)`;
 *   - quote collateral: `F(T) = S D_foreign(T) / D_domestic(T)`.
 *
 * `foreignDiscount` is always the non-collateral (foreign) curve, mirroring
 * `XccyPillar`; `isFxBaseCollateral` selects the branch. Quoted forward points
 * are the model outright minus spot. `bootstrapFxDiscountCurve` solves the
 * non-collateral discount nodes one pillar at a time with `math::BrentSolver`,
 * then re-solves in triangular (Gauss-Seidel) order if the first pass leaves a
 * residual, mirroring `bootstrapXccyDiscountCurve`.
 */

/// One FX swap or forward pillar: near and far settlement dates, the spot rate
/// and the quoted forward points or far outright. `convention` selects which
/// quote field is read: `Points` reads `points`, `Outright` reads `outright`.
struct FxSwapPillar {
    datetime::Date start;    ///< Near (spot) settlement date
    datetime::Date maturity; ///< Far settlement date
    double spot = 0.0;       ///< Spot rate (quote units per base unit)
    double points = 0.0;     ///< Forward points quoted from spot (price difference)
    double outright = 0.0;   ///< Far outright, used with `QuoteConvention::Outright`
    QuoteConvention convention = QuoteConvention::Points;
    /// True when the base currency is the collateral currency, so the pair's
    /// non-collateral (foreign) curve is the quote currency.
    bool isFxBaseCollateral = false;
    datetime::DayCounter dayCounter{datetime::DayCount::Actual360}; ///< Quote day count
};

/// Model far outright of an FX swap pillar under covered interest parity. The
/// near leg is the spot rate, so only the far maturity enters the CIP ratio.
inline double impliedFxOutright(const DiscountCurve<double>& foreignDiscount,
                                const DiscountCurve<double>& domesticDiscount, double spot,
                                const FxSwapPillar& pillar, const datetime::Date& referenceDate,
                                const datetime::DayCounter& zeroDayCounter) {
    const double t = datetime::yearFraction(referenceDate, pillar.maturity, zeroDayCounter);
    if (pillar.isFxBaseCollateral) {
        return spot * domesticDiscount.discount(t) / foreignDiscount.discount(t);
    }
    return spot * foreignDiscount.discount(t) / domesticDiscount.discount(t);
}

/// Model forward points of an FX swap pillar: the CIP outright minus spot.
inline double impliedFxForwardPoints(const DiscountCurve<double>& foreignDiscount,
                                     const DiscountCurve<double>& domesticDiscount, double spot,
                                     const FxSwapPillar& pillar,
                                     const datetime::Date& referenceDate,
                                     const datetime::DayCounter& zeroDayCounter) {
    return impliedFxOutright(foreignDiscount, domesticDiscount, spot, pillar, referenceDate,
                             zeroDayCounter) -
           spot;
}

/// Quoted value of an FX swap pillar in its own convention.
inline double fxPillarTargetQuote(const FxSwapPillar& pillar) {
    return pillar.convention == QuoteConvention::Points ? pillar.points : pillar.outright;
}

/// Model value of an FX swap pillar in its own convention.
inline double fxPillarImpliedQuote(const DiscountCurve<double>& foreignDiscount,
                                   const DiscountCurve<double>& domesticDiscount,
                                   const FxSwapPillar& pillar, const datetime::Date& referenceDate,
                                   const datetime::DayCounter& zeroDayCounter) {
    return pillar.convention == QuoteConvention::Points
               ? impliedFxForwardPoints(foreignDiscount, domesticDiscount, pillar.spot, pillar,
                                        referenceDate, zeroDayCounter)
               : impliedFxOutright(foreignDiscount, domesticDiscount, pillar.spot, pillar,
                                   referenceDate, zeroDayCounter);
}

// The generic node solver (`detail::bootstrapNodesByBrent`) lives in
// `BootstrapInstrument.h`, shared with the cross-currency bootstraps.

/// Exact-fit bootstrap of the non-collateral (foreign) discount curve from FX
/// swap points or outrights; the collateral (domestic) curve stays frozen.
/// Pillar maturities must be strictly increasing and are used directly as the
/// solved node dates.
inline DiscountCurve<double>
bootstrapFxDiscountCurve(const DiscountCurve<double>& domesticDiscount,
                         const datetime::Date& referenceDate,
                         const datetime::DayCounter& zeroDayCounter, InterpolationSpace space,
                         InterpolationScheme scheme, const std::vector<FxSwapPillar>& pillars,
                         double accuracy = 1e-14, double tension = 0.0, int switchIndex = 1) {
    if (pillars.empty()) {
        throw std::invalid_argument("bootstrapFxDiscountCurve: no pillars");
    }
    const std::size_t count = pillars.size();
    std::vector<double> nodeTimes(count);
    std::vector<datetime::Date> maturityDates(count);
    for (std::size_t i = 0; i < count; ++i) {
        nodeTimes[i] = datetime::yearFraction(referenceDate, pillars[i].maturity, zeroDayCounter);
        maturityDates[i] = pillars[i].maturity;
        if (!(nodeTimes[i] > (i == 0 ? 0.0 : nodeTimes[i - 1]))) {
            throw std::invalid_argument(
                "bootstrapFxDiscountCurve: maturities must be strictly increasing");
        }
        if (!(pillars[i].spot > 0.0)) {
            throw std::invalid_argument("bootstrapFxDiscountCurve: spot must be positive");
        }
    }
    const auto residual = [&](std::size_t i, const DiscountCurve<double>& foreignDiscount) {
        const FxSwapPillar& pillar = pillars[i];
        const double t = nodeTimes[i];
        const double model =
            pillar.isFxBaseCollateral
                ? pillar.spot * domesticDiscount.discount(t) / foreignDiscount.discount(t)
                : pillar.spot * foreignDiscount.discount(t) / domesticDiscount.discount(t);
        const double implied =
            pillar.convention == QuoteConvention::Points ? model - pillar.spot : model;
        return implied - fxPillarTargetQuote(pillar);
    };
    const std::vector<double> zeros =
        detail::bootstrapNodesByBrent(space, scheme, tension, switchIndex, accuracy, nodeTimes,
                                      residual, "bootstrapFxDiscountCurve");
    return DiscountCurve<double>(referenceDate, maturityDates, zeroDayCounter, zeros, space, scheme,
                                 tension, switchIndex);
}

} // namespace quantape::markets
