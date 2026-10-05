#ifndef FX_QUOTE_H
#define FX_QUOTE_H

// Double-based FX quote records: spot quotes and forward/swap quotes sharing a
// bid/ask envelope, plus forward-point and settlement helpers. All rates are
// quoted in quote-currency units per one unit of the base currency (EURUSD ~
// 1.10 USD per EUR). Values are plain doubles today; an optional later AD
// template can wrap these records and hold them as constants while carrying
// the differentiated value alongside.

#include "quantape/datetime/BusinessDayConvention.h"
#include "quantape/datetime/Calendar.h"
#include "quantape/datetime/Date.h"
#include "quantape/datetime/DayCounter.h"
#include "quantape/datetime/TimeConversion.h"
#include "quantape/markets/Descriptors/FXDescriptor.h"

#include <cmath>
#include <cstdint>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>

namespace quantape::markets {

/// How a forward quote is expressed: `Points` is the price difference from
/// spot, `Outright` is the forward price itself.
enum class QuoteConvention : std::uint8_t { Points, Outright };

/// Settlement helper surfaced from datetime: `spotDate(trade, jointCalendar,
/// spotLag, convention)` for a single joint calendar and `spotDate(trade,
/// baseCalendar, quoteCalendar, baseLag, quoteLag, convention)` when the two
/// currency calendars are passed separately. The joint overloads default to
/// ModifiedFollowing.
using datetime::spotDate;

/// Forward points to outright: F = S + points * pointsScale. `pointsScale` is
/// the price value of one point (1e-4 for most pairs, 1e-2 for JPY pairs).
inline double pointsToOutright(double spot, double points, double pointsScale = 1.0) {
    return spot + points * pointsScale;
}

/// Inverse of pointsToOutright: points = (F - S) / pointsScale.
inline double outrightToPoints(double spot, double outright, double pointsScale = 1.0) {
    return (outright - spot) / pointsScale;
}

/// Convert a quote value between the two representations. `from` identifies
/// the input convention, so the result is in the other one.
inline double convertQuote(double value, QuoteConvention from, double spot,
                           double pointsScale = 1.0) {
    return from == QuoteConvention::Points ? pointsToOutright(spot, value, pointsScale)
                                           : outrightToPoints(spot, value, pointsScale);
}

/// Annualized forward premium or discount of an outright against spot:
/// (F / S - 1) / tau with tau = yearFraction(start, maturity, dayCounter).
inline double annualizedForwardPremium(double spot, double outright, const datetime::Date& start,
                                       const datetime::Date& maturity,
                                       const datetime::DayCounter& dayCounter) {
    const double tau = dayCounter.yearFraction(start, maturity);
    if (tau <= 0.0) {
        throw std::invalid_argument("annualizedForwardPremium: maturity must be after start");
    }
    if (spot <= 0.0) {
        throw std::invalid_argument("annualizedForwardPremium: spot must be positive");
    }
    return (outright / spot - 1.0) / tau;
}

/// Inverse of annualizedForwardPremium: F = S * (1 + premium * tau).
inline double outrightFromPremium(double spot, double premium, const datetime::Date& start,
                                  const datetime::Date& maturity,
                                  const datetime::DayCounter& dayCounter) {
    return spot * (1.0 + premium * dayCounter.yearFraction(start, maturity));
}

/**
 * @brief Shared bid/ask envelope for FX quotes
 *
 * Carries identification and the quoted spread. Bid/ask are expressed in the
 * quote's own convention, so forward-point quotes may be negative; `ask` must
 * not be below `bid` and an explicitly supplied mid must lie inside the
 * spread. When no mid is supplied, `mid()` returns the arithmetic average.
 */
class FxQuote {
public:
    FxQuote() = default;

    FxQuote(std::string id, std::string source, double bid, double ask,
            std::optional<double> mid = std::nullopt)
        : m_id(std::move(id)), m_source(std::move(source)), m_bid(bid), m_ask(ask), m_mid(mid) {
        validate();
    }

    const std::string& id() const { return m_id; }
    const std::string& source() const { return m_source; }
    double bid() const { return m_bid; }
    double ask() const { return m_ask; }
    bool hasMid() const { return m_mid.has_value(); }

    /// Supplied mid when present, otherwise the bid/ask average.
    double mid() const { return m_mid.has_value() ? *m_mid : 0.5 * (m_bid + m_ask); }

    double spread() const { return m_ask - m_bid; }

    /// Throws std::invalid_argument when the spread is inverted, a value is not
    /// finite, or the optional mid is outside [bid, ask].
    void validate() const {
        if (!std::isfinite(m_bid) || !std::isfinite(m_ask)) {
            throw std::invalid_argument("FxQuote: bid/ask must be finite");
        }
        if (m_ask < m_bid) {
            throw std::invalid_argument("FxQuote: ask must not be below bid");
        }
        if (m_mid.has_value() && (*m_mid < m_bid || *m_mid > m_ask)) {
            throw std::invalid_argument("FxQuote: mid must lie within [bid, ask]");
        }
    }

private:
    std::string m_id;
    std::string m_source;
    double m_bid = 0.0;
    double m_ask = 0.0;
    std::optional<double> m_mid;
};

/**
 * @brief Spot FX quote
 *
 * A spot rate for one pair, quoted as quote-currency units per one unit of the
 * base currency (e.g., EURUSD ~ 1.10 USD per EUR). The spot must be positive
 * and both currencies must be known.
 */
class FxSpotQuote : public FxQuote {
public:
    FxSpotQuote() = default;

    FxSpotQuote(FxQuote quote, FXDescriptor descriptor, double spot)
        : FxQuote(std::move(quote)), m_descriptor(std::move(descriptor)), m_spot(spot) {
        validate();
    }

    const FXDescriptor& descriptor() const { return m_descriptor; }
    const std::string& baseCcy() const { return m_descriptor.baseCcy; }
    const std::string& quoteCcy() const { return m_descriptor.quoteCcy; }
    double spot() const { return m_spot; }

    /// Throws std::invalid_argument on an invalid envelope, a non-positive
    /// spot, or unknown currencies.
    void validate() const {
        FxQuote::validate();
        if (!std::isfinite(m_spot) || !(m_spot > 0.0)) {
            throw std::invalid_argument("FxSpotQuote: spot must be positive");
        }
        if (!m_descriptor.hasKnownCurrencies()) {
            throw std::invalid_argument("FxSpotQuote: unknown base or quote currency");
        }
    }

private:
    FXDescriptor m_descriptor;
    double m_spot = 0.0;
};

/**
 * @brief FX swap or forward quote
 *
 * Quoted for the period from the near leg (`startDate`) to the far leg
 * (`maturityDate`). The envelope holds the quote in `convention()` units:
 * forward points when `QuoteConvention::Points`, outright otherwise. Points
 * are converted with `spot()` and `pointsScale()` (1e-4 for most pairs, 1e-2
 * for JPY pairs), so the same record serves both quoting styles.
 */
class FxSwapQuote : public FxQuote {
public:
    FxSwapQuote() = default;

    FxSwapQuote(FxQuote quote, FXDescriptor descriptor, double spot, datetime::Date spotDate,
                datetime::Date startDate, datetime::Date maturityDate, double pointsScale = 1.0,
                QuoteConvention convention = QuoteConvention::Points)
        : FxQuote(std::move(quote)), m_descriptor(std::move(descriptor)), m_spot(spot),
          m_spotDate(spotDate), m_startDate(startDate), m_maturityDate(maturityDate),
          m_pointsScale(pointsScale), m_convention(convention) {
        validate();
    }

    const FXDescriptor& descriptor() const { return m_descriptor; }
    const std::string& baseCcy() const { return m_descriptor.baseCcy; }
    const std::string& quoteCcy() const { return m_descriptor.quoteCcy; }
    double spot() const { return m_spot; }
    const datetime::Date& spotDate() const { return m_spotDate; }
    const datetime::Date& startDate() const { return m_startDate; }
    const datetime::Date& maturityDate() const { return m_maturityDate; }
    const datetime::Date& nearDate() const { return m_startDate; }
    const datetime::Date& farDate() const { return m_maturityDate; }
    double pointsScale() const { return m_pointsScale; }
    QuoteConvention convention() const { return m_convention; }

    /// Mid quote as forward points, converted when stored as an outright.
    double points() const {
        return m_convention == QuoteConvention::Points
                   ? mid()
                   : outrightToPoints(m_spot, mid(), m_pointsScale);
    }

    /// Mid quote as an outright forward, converted when stored as points.
    double outright() const {
        return m_convention == QuoteConvention::Points
                   ? pointsToOutright(m_spot, mid(), m_pointsScale)
                   : mid();
    }

    double bidPoints() const {
        return m_convention == QuoteConvention::Points
                   ? bid()
                   : outrightToPoints(m_spot, bid(), m_pointsScale);
    }

    double askPoints() const {
        return m_convention == QuoteConvention::Points
                   ? ask()
                   : outrightToPoints(m_spot, ask(), m_pointsScale);
    }

    double bidOutright() const {
        return m_convention == QuoteConvention::Points
                   ? pointsToOutright(m_spot, bid(), m_pointsScale)
                   : bid();
    }

    double askOutright() const {
        return m_convention == QuoteConvention::Points
                   ? pointsToOutright(m_spot, ask(), m_pointsScale)
                   : ask();
    }

    /// Annualized premium of the mid outright over near-to-far.
    double annualizedPremium(const datetime::DayCounter& dayCounter) const {
        return annualizedForwardPremium(m_spot, outright(), m_startDate, m_maturityDate,
                                        dayCounter);
    }

    /// Throws std::invalid_argument on an invalid envelope, a non-positive
    /// spot or scale, unknown currencies, a maturity at or before the start,
    /// or a non-positive outright quote.
    void validate() const {
        FxQuote::validate();
        if (!std::isfinite(m_spot) || !(m_spot > 0.0)) {
            throw std::invalid_argument("FxSwapQuote: spot must be positive");
        }
        if (!std::isfinite(m_pointsScale) || !(m_pointsScale > 0.0)) {
            throw std::invalid_argument("FxSwapQuote: points scale must be positive");
        }
        if (!m_descriptor.hasKnownCurrencies()) {
            throw std::invalid_argument("FxSwapQuote: unknown base or quote currency");
        }
        if (m_maturityDate <= m_startDate) {
            throw std::invalid_argument("FxSwapQuote: maturity must be after start");
        }
        if (m_convention == QuoteConvention::Outright && bid() <= 0.0) {
            throw std::invalid_argument("FxSwapQuote: outright quote must be positive");
        }
    }

private:
    FXDescriptor m_descriptor;
    double m_spot = 0.0;
    datetime::Date m_spotDate;
    datetime::Date m_startDate;
    datetime::Date m_maturityDate;
    double m_pointsScale = 1.0;
    QuoteConvention m_convention = QuoteConvention::Points;
};

} // namespace quantape::markets

#endif // FX_QUOTE_H
