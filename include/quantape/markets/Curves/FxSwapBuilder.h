#pragma once

#include "quantape/datetime/Date.h"
#include "quantape/datetime/DayCounter.h"
#include "quantape/datetime/TimeConversion.h"
#include "quantape/markets/Curves/DiscountCurve.h"
#include "quantape/markets/Data/FxQuote.h"
#include "quantape/math/Solvers/BrentSolver.h"

#include <cmath>
#include <cstddef>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
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

namespace detail {

/// Sequential exact-fit node solver shared by the FX-swap and cross-currency
/// bootstraps: one Brent solve per node against a trial curve moved in place
/// from the current nodes, then a triangular Gauss-Seidel re-pass when the
/// first pass leaves residual error. `residual(i, curve)` returns the model
/// quote minus its target. Node 0 (t = 0) is pinned to zero. One trial curve
/// is built per node solve, so the Brent steps only move the solved node
/// through `CurveTrialUpdater` instead of rebuilding the grid each time.
template <typename Residual>
std::vector<double>
bootstrapNodesByBrent(InterpolationSpace space, InterpolationScheme scheme, double tension,
                      int switchIndex, double accuracy, const std::vector<double>& nodeTimes,
                      const Residual& residual, std::string_view context = "bootstrap") {
    const std::size_t count = nodeTimes.size();
    const quantape::math::BrentSolver<double> solver;
    std::vector<double> zeros(count, 0.0);
    std::vector<double> trialTimes;
    std::vector<double> trialZeros;
    const auto worstResidual = [&]() {
        trialTimes.assign(1, 0.0);
        trialZeros.assign(1, 0.0);
        if (trialTimes.capacity() < count + 1) {
            trialTimes.reserve(count + 1);
            trialZeros.reserve(count + 1);
        }
        for (std::size_t j = 0; j < count; ++j) {
            trialTimes.push_back(nodeTimes[j]);
            trialZeros.push_back(zeros[j]);
        }
        const DiscountCurve<double> curve(trialTimes, trialZeros, space, scheme, tension,
                                          switchIndex);
        double worst = 0.0;
        for (std::size_t i = 0; i < count; ++i) {
            const double check = residual(i, curve);
            if (!std::isfinite(check)) {
                worst = 1e300;
            } else if (std::abs(check) > worst) {
                worst = std::abs(check);
            }
        }
        return worst;
    };
    const auto solveNodes = [&](bool multiPass) {
        for (int pass = 0; pass < (multiPass ? 50 : 1); ++pass) {
            const std::vector<double> previous = zeros;
            double lastMove = 0.0;
            for (std::size_t i = 0; i < count; ++i) {
                const std::size_t lastNode = multiPass && pass == 0 ? i : count - 1;
                // The trial curve is rebuilt once per node solve and then only
                // the solved node moves; the scheme state stays valid because
                // the node grid is fixed for the whole solve.
                std::optional<DiscountCurve<double>> trialCurve;
                bool cacheValid = false;
                double cachedZero = 0.0;
                double cachedResidual = 0.0;
                const auto objective = [&](double trialZero) {
                    if (cacheValid && trialZero == cachedZero) {
                        return cachedResidual;
                    }
                    if (!trialCurve.has_value()) {
                        trialTimes.assign(1, 0.0);
                        trialZeros.assign(1, 0.0);
                        if (trialTimes.capacity() < lastNode + 2) {
                            trialTimes.reserve(lastNode + 2);
                            trialZeros.reserve(lastNode + 2);
                        }
                        for (std::size_t j = 0; j <= lastNode; ++j) {
                            trialTimes.push_back(nodeTimes[j]);
                            trialZeros.push_back(zeros[j]);
                        }
                        trialCurve.emplace(trialTimes, trialZeros, space, scheme, tension,
                                           switchIndex);
                    }
                    CurveTrialUpdater::setNode(*trialCurve, i + 1, trialZero);
                    cachedResidual = residual(i, *trialCurve);
                    cachedZero = trialZero;
                    cacheValid = true;
                    return cachedResidual;
                };
                const double guess = i == 0 ? 0.0 : zeros[i - 1];
                double lower = guess - 0.5;
                double upper = guess + 0.5;
                double fLower = objective(lower);
                double fUpper = objective(upper);
                int widen = 0;
                while (fLower * fUpper > 0.0 && widen < 12) {
                    lower -= 0.5;
                    upper += 0.5;
                    fLower = objective(lower);
                    fUpper = objective(upper);
                    ++widen;
                }
                if (!(fLower * fUpper <= 0.0) || !std::isfinite(fLower) || !std::isfinite(fUpper)) {
                    throw std::runtime_error(std::string(context) + ": failed to bracket pillar " +
                                             std::to_string(i));
                }
                const double root = solver.solve(objective, accuracy, guess, lower, upper);
                const double move = std::abs(root - previous[i]);
                if (move > lastMove) {
                    lastMove = move;
                }
                zeros[i] = root;
            }
            if (!multiPass || lastMove < 1e-15) {
                break;
            }
        }
    };
    solveNodes(false);
    if (!(worstResidual() < 1e-9)) {
        solveNodes(true);
    }
    if (!(worstResidual() < 1e-9)) {
        throw std::runtime_error(std::string(context) + ": fixed point did not converge");
    }
    return zeros;
}

} // namespace detail

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
