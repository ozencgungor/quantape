#pragma once

#include "quantape/datetime/Date.h"
#include "quantape/datetime/DayCounter.h"
#include "quantape/instruments/FxInstruments.h"
#include "quantape/markets/Curves/BootstrapInstrument.h"
#include "quantape/markets/Curves/DiscountCurve.h"
#include "quantape/markets/Data/FxQuote.h"
#include "quantape/pricing/Fx.h"

#include <cstddef>
#include <stdexcept>
#include <vector>

namespace quantape::markets {
/**
 * @file FxSwapBuilder.h
 * @brief FX forward-points pillars and the collateral discount bootstrap
 *
 * An FX swap pillar is the plain-data `FxSwap` instrument: the near (spot)
 * settlement date, the far maturity, the spot rate and the quoted forward
 * points or far outright. The far outright follows covered interest parity
 * with the near leg pinned to spot,
 *
 *   `F(T) = S D_base(T) / D_quote(T)`,
 *
 * where the correct base/quote assignment depends on which leg is collateral:
 *
 *   - base collateral:  `F(T) = S D_domestic(T) / D_foreign(T)`;
 *   - quote collateral: `F(T) = S D_foreign(T) / D_domestic(T)`.
 *
 * `foreignDiscount` is always the non-collateral (foreign) curve, mirroring
 * `XccyPillar`; `isFxBaseCollateral` selects the branch. The CIP arithmetic
 * lives once in `quantape/pricing/Fx.h`; this header keeps the pillar spelling
 * and the bootstrap driver. `bootstrapFxDiscountCurve` solves the non-collateral
 * discount nodes one pillar at a time through the shared
 * `detail::bootstrapNodesByBrent`, then re-solves in triangular (Gauss-Seidel)
 * order if the first pass leaves a residual.
 */

/// One FX swap or forward pillar: the dated `FxSwap` instrument, kept under its
/// original name for the curve-side consumers.
using FxSwapPillar = instruments::FxSwap;

/// Quoted value of an FX swap pillar in its own convention.
inline double fxPillarTargetQuote(const FxSwapPillar& pillar) {
    return pillar.target();
}

/// CIP far outright of an FX swap pillar from an explicit spot: delegates to
/// the instrument helper with the base/quote curves assigned by collateral
/// direction.
inline double impliedFxOutright(const DiscountCurve<double>& foreignDiscount,
                                const DiscountCurve<double>& domesticDiscount, double spot,
                                const FxSwapPillar& pillar, const datetime::Date& referenceDate,
                                const datetime::DayCounter& zeroDayCounter) {
    if (pillar.isFxBaseCollateral) {
        const FxSet<DiscountCurve<double>, DiscountCurve<double>> curves{domesticDiscount,
                                                                         foreignDiscount};
        return impliedFxOutright(pillar, curves, spot, referenceDate, zeroDayCounter);
    }
    const FxSet<DiscountCurve<double>, DiscountCurve<double>> curves{foreignDiscount,
                                                                     domesticDiscount};
    return impliedFxOutright(pillar, curves, spot, referenceDate, zeroDayCounter);
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

/// Exact-fit bootstrap of the non-collateral (foreign) discount curve from FX
/// swap points or outrights; the collateral (domestic) curve stays frozen.
/// Pillar maturities must be strictly increasing and are used directly as the
/// solved node dates. The residual evaluates the plain-data instrument through
/// its `impliedQuote` concept member.
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
    std::vector<FxSwapPillar> instruments(count);
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
        instruments[i] = pillars[i];
        instruments[i].tradeDate = referenceDate;
        instruments[i].zeroDayCounter = zeroDayCounter;
        instruments[i].pointsScale = 1.0;
    }
    const auto residual = [&](std::size_t i, const DiscountCurve<double>& foreignDiscount) {
        const FxSwapPillar& instrument = instruments[i];
        if (instrument.isFxBaseCollateral) {
            const FxSet<DiscountCurve<double>, DiscountCurve<double>> curves{domesticDiscount,
                                                                             foreignDiscount};
            return instrument.template impliedQuote<double>(curves) - instrument.target();
        }
        const FxSet<DiscountCurve<double>, DiscountCurve<double>> curves{foreignDiscount,
                                                                         domesticDiscount};
        return instrument.template impliedQuote<double>(curves) - instrument.target();
    };
    const std::vector<double> zeros =
        detail::bootstrapNodesByBrent(space, scheme, tension, switchIndex, accuracy, nodeTimes,
                                      residual, "bootstrapFxDiscountCurve");
    return DiscountCurve<double>(referenceDate, maturityDates, zeroDayCounter, zeros, space, scheme,
                                 tension, switchIndex);
}

} // namespace quantape::markets
