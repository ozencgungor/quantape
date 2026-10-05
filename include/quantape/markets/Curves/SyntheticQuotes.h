#pragma once

// Synthetic market quotes from the overnight discount curve plus the
// integrated overnight-versus-index basis ("synthetic deposits" / FRAs), for
// pinning the short end of an index curve where no direct quote exists.

#include "quantape/datetime/Date.h"
#include "quantape/datetime/DayCounter.h"
#include "quantape/datetime/TimeConversion.h"
#include "quantape/markets/Curves/DiscountCurve.h"
#include "quantape/markets/Curves/SpreadCurve.h"

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <vector>

namespace quantape::markets {
/**
 * @file SyntheticQuotes.h
 * @brief Synthetic index quotes from the OIS curve and the basis
 *
 * The integrated basis between an index curve and its overnight parent over
 * `[T1, T2]` is the log-growth difference
 *
 *   `Delta = ln( (P_x(T1)/P_x(T2)) / (P_on(T1)/P_on(T2)) )`
 *
 * so the index simple rate follows exactly from the overnight rate and the
 * integrated basis,
 *
 *   `1 + R_x tau_x = (1 + R_on tau_on) * exp(Delta)`.
 *
 * This is the exact form of the standard market convention
 * `R_x tau_x = R_on tau_on + Delta` (its first-order expansion); the overnight
 * rate comes from the discount curve and the basis from a spread curve, so a
 * curve with missing short-end quotations can be pinned with
 * market-consistent synthetic deposits.
 */
inline double integratedBasis(const SpreadCurve<double>& basisCurve,
                              const DiscountCurve<double>& overnightCurve, double t1, double t2) {
    if (!(t2 > t1)) {
        throw std::invalid_argument("integratedBasis: t2 must be > t1");
    }
    const double indexGrowth = basisCurve.discount(t1) / basisCurve.discount(t2);
    const double overnightGrowth = overnightCurve.discount(t1) / overnightCurve.discount(t2);
    if (!(indexGrowth > 0.0) || !(overnightGrowth > 0.0)) {
        throw std::invalid_argument("integratedBasis: non-positive growth factors");
    }
    return std::log(indexGrowth) - std::log(overnightGrowth);
}

/// Synthetic index FRA quote over `[start, maturity]`. `overnightDayCounter`
/// is validation-only (it checks the overnight accrual is positive); the
/// returned rate depends on the discount ratios, in the exact form
/// `1 + R_x tau_x = (1 + R_on tau_on) exp(Delta)`.
inline double syntheticFraQuote(const SpreadCurve<double>& basisCurve,
                                const DiscountCurve<double>& overnightCurve,
                                const datetime::Date& referenceDate, const datetime::Date& start,
                                const datetime::Date& maturity,
                                const datetime::DayCounter& overnightDayCounter,
                                const datetime::DayCounter& indexDayCounter,
                                const datetime::DayCounter& zeroDayCounter) {
    const double t1 = datetime::yearFraction(referenceDate, start, zeroDayCounter);
    const double t2 = datetime::yearFraction(referenceDate, maturity, zeroDayCounter);
    const double tauOvernight = datetime::yearFraction(start, maturity, overnightDayCounter);
    const double tauIndex = datetime::yearFraction(start, maturity, indexDayCounter);
    if (!(tauOvernight > 0.0) || !(tauIndex > 0.0)) {
        throw std::invalid_argument("syntheticFraQuote: non-positive accrual");
    }
    const double overnightGrowth = overnightCurve.discount(t1) / overnightCurve.discount(t2);
    const double delta = integratedBasis(basisCurve, overnightCurve, t1, t2);
    return (overnightGrowth * std::exp(delta) - 1.0) / tauIndex;
}

/// How a quote strip is extended beyond its last observation.
enum class QuoteExtrapolation : std::uint8_t { Flat, Linear };

/// Extrapolated quote at a time beyond the last observation: `Flat` keeps the
/// last quote, `Linear` extends the slope of the last segment.
inline double extrapolatedQuote(const std::vector<double>& times, const std::vector<double>& quotes,
                                double targetTime,
                                QuoteExtrapolation mode = QuoteExtrapolation::Flat) {
    if (times.size() != quotes.size() || times.empty()) {
        throw std::invalid_argument("extrapolatedQuote: malformed quote strip");
    }
    for (std::size_t i = 1; i < times.size(); ++i) {
        if (!(times[i] > times[i - 1])) {
            throw std::invalid_argument("extrapolatedQuote: times must be strictly increasing");
        }
    }
    if (!(targetTime > times.back())) {
        throw std::invalid_argument("extrapolatedQuote: target time is inside the quoted range");
    }
    if (mode == QuoteExtrapolation::Flat || times.size() < 2) {
        return quotes.back();
    }
    const std::size_t last = times.size() - 1;
    const double slope = (quotes[last] - quotes[last - 1]) / (times[last] - times[last - 1]);
    return quotes[last] + slope * (targetTime - times[last]);
}

/// Synthetic index deposit quote (accrual starts at the reference date).
inline double syntheticDepositQuote(const SpreadCurve<double>& basisCurve,
                                    const DiscountCurve<double>& overnightCurve,
                                    const datetime::Date& referenceDate,
                                    const datetime::Date& maturity,
                                    const datetime::DayCounter& overnightDayCounter,
                                    const datetime::DayCounter& indexDayCounter,
                                    const datetime::DayCounter& zeroDayCounter) {
    return syntheticFraQuote(basisCurve, overnightCurve, referenceDate, referenceDate, maturity,
                             overnightDayCounter, indexDayCounter, zeroDayCounter);
}

} // namespace quantape::markets
