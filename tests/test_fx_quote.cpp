// test_fx_quote.cpp — FX quote envelopes, forward-point conversion,
// annualization and joint-calendar spot settlement.
#include "quantape/datetime/BusinessDayConvention.h"
#include "quantape/datetime/Calendar.h"
#include "quantape/datetime/Date.h"
#include "quantape/datetime/DayCounter.h"
#include "quantape/log/Log.h"
#include "quantape/markets/Data/FxQuote.h"
#include "quantape/util/Check.h"

#include <cmath>
#include <optional>
#include <stdexcept>

using namespace quantape::markets;
using quantape::datetime::BusinessDayConvention;
using quantape::datetime::Calendar;
using quantape::datetime::Date;
using quantape::datetime::DayCount;
using quantape::datetime::DayCounter;

namespace {

bool close(double a, double b, double tol = 1e-12) {
    return std::abs(a - b) <= tol * (1.0 + std::abs(b));
}

template <typename Callable>
bool throwsInvalidArgument(Callable&& callable) {
    try {
        callable();
    } catch (const std::invalid_argument&) {
        return true;
    } catch (...) {
        return false;
    }
    return false;
}

void testEnvelope() {
    QTA_LOG_INFO("test", "=== FX Quote Envelope Test ===");

    const FxQuote averaged("eurusd.spot", "vendorA", 1.0998, 1.1002);
    CHECK(averaged.id() == "eurusd.spot");
    CHECK(averaged.source() == "vendorA");
    CHECK(!averaged.hasMid());
    CHECK(close(averaged.mid(), 1.1000));
    CHECK(close(averaged.spread(), 0.0004));

    const FxQuote observed("eurusd.spot", "vendorB", 1.0999, 1.1001, 1.1000);
    CHECK(observed.hasMid());
    CHECK(close(observed.mid(), 1.1000));

    // Forward points may be negative; only the spread ordering is enforced.
    const FxQuote negativePoints("eurusd.3m", "vendorA", -130.0, -120.0);
    CHECK(close(negativePoints.mid(), -125.0));

    CHECK(throwsInvalidArgument([] {
        const FxQuote inverted("id", "source", 1.2, 1.1);
        (void)inverted;
    }));
    CHECK(throwsInvalidArgument([] {
        const FxQuote outside("id", "source", 1.0, 1.2, 1.3);
        (void)outside;
    }));
}

void testPairSemantics() {
    QTA_LOG_INFO("test", "=== Pair Semantics Test ===");

    const FXDescriptor eurusd("EUR", "USD", "2026-09-29");
    CHECK(eurusd.baseCcy == "EUR");
    CHECK(eurusd.quoteCcy == "USD");
    CHECK(eurusd.pair() == "EURUSD");
    CHECK(eurusd.identifier() == "EURUSD.2026-09-29");
    CHECK(eurusd.hasKnownCurrencies());

    const FXDescriptor usdjpy("USD", "JPY");
    CHECK(usdjpy.pair() == "USDJPY");
    CHECK(usdjpy.identifier() == "USDJPY.");

    const FXDescriptor unknown("XXX", "USD");
    CHECK(!unknown.hasKnownCurrencies());

    // Deprecated aliases keep the old domestic/foreign meaning: the domestic
    // currency is the quote and the foreign currency is the base.
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
    CHECK(eurusd.domesticCcy() == "USD");
    CHECK(eurusd.foreignCcy() == "EUR");
#pragma GCC diagnostic pop
}

void testSpotQuote() {
    QTA_LOG_INFO("test", "=== FX Spot Quote Test ===");

    const FXDescriptor eurusd("EUR", "USD", "2026-09-29");
    const FxSpotQuote quote(FxQuote("eurusd.spot", "vendorA", 1.0998, 1.1002), eurusd, 1.1000);
    CHECK(quote.baseCcy() == "EUR");
    CHECK(quote.quoteCcy() == "USD");
    CHECK(quote.descriptor().pair() == "EURUSD");
    CHECK(close(quote.spot(), 1.1000));
    CHECK(close(quote.mid(), 1.1000));
    CHECK(close(quote.spread(), 0.0004));

    CHECK(throwsInvalidArgument([] {
        const FxSpotQuote bad(FxQuote("id", "source", 1.0, 1.1), FXDescriptor("EUR", "USD"), -1.0);
        (void)bad;
    }));
    CHECK(throwsInvalidArgument([] {
        const FxSpotQuote bad(FxQuote("id", "source", 1.0, 1.1), FXDescriptor("XXX", "USD"), 1.1);
        (void)bad;
    }));
    CHECK(throwsInvalidArgument([] {
        const FxSpotQuote bad(FxQuote("id", "source", 1.2, 1.1), FXDescriptor("EUR", "USD"), 1.1);
        (void)bad;
    }));
}

void testPointsOutrightRoundTrip() {
    QTA_LOG_INFO("test", "=== Points/Outright Conversion Test ===");

    const double spot = 1.1000;
    const double points = 125.0;
    const double scale = 1e-4;
    const double outright = pointsToOutright(spot, points, scale);
    CHECK(close(outright, 1.1125));
    CHECK(close(outrightToPoints(spot, outright, scale), points));
    CHECK(close(convertQuote(points, QuoteConvention::Points, spot, scale), outright));
    CHECK(close(convertQuote(outright, QuoteConvention::Outright, spot, scale), points));

    // JPY-style scaling: one point is 1e-2.
    CHECK(close(pointsToOutright(157.25, 35.0, 1e-2), 157.60));
    CHECK(close(outrightToPoints(157.25, 157.60, 1e-2), 35.0));

    // Discount points round-trip as well.
    CHECK(close(pointsToOutright(spot, -125.0, scale), 1.0875));
    CHECK(close(outrightToPoints(spot, 1.0875, scale), -125.0));
}

void testAnnualization() {
    QTA_LOG_INFO("test", "=== Annualization Test ===");

    const DayCounter act360(DayCount::Actual360);
    const Date start(2026, 9, 29);
    const Date maturity(2027, 3, 29);
    const double tau = act360.yearFraction(start, maturity);
    CHECK(close(tau, 181.0 / 360.0));

    const double premium = annualizedForwardPremium(1.1, 1.1125, start, maturity, act360);
    CHECK(close(premium, (1.1125 / 1.1 - 1.0) / tau));
    CHECK(close(outrightFromPremium(1.1, premium, start, maturity, act360), 1.1125));

    CHECK(throwsInvalidArgument([] {
        const DayCounter act360(DayCount::Actual360);
        (void)annualizedForwardPremium(1.1, 1.2, Date(2026, 3, 29), Date(2026, 3, 29), act360);
    }));
}

void testSwapQuote() {
    QTA_LOG_INFO("test", "=== FX Swap Quote Test ===");

    const FXDescriptor eurusd("EUR", "USD", "2026-09-29");
    const Date spotDate(2026, 10, 1);
    const Date start(2026, 10, 1);
    const Date maturity(2027, 1, 4);

    const FxSwapQuote pointsQuote(FxQuote("eurusd.3m", "vendorA", 120.0, 130.0), eurusd, 1.1000,
                                  spotDate, start, maturity, 1e-4, QuoteConvention::Points);
    CHECK(pointsQuote.convention() == QuoteConvention::Points);
    CHECK(pointsQuote.spotDate() == spotDate);
    CHECK(pointsQuote.nearDate() == start);
    CHECK(pointsQuote.farDate() == maturity);
    CHECK(close(pointsQuote.points(), 125.0));
    CHECK(close(pointsQuote.bidPoints(), 120.0));
    CHECK(close(pointsQuote.askPoints(), 130.0));
    CHECK(close(pointsQuote.outright(), 1.1125));
    CHECK(close(pointsQuote.bidOutright(), 1.1000 + 120.0 * 1e-4));
    CHECK(close(pointsQuote.askOutright(), 1.1000 + 130.0 * 1e-4));

    const FxSwapQuote outrightQuote(FxQuote("eurusd.3m.o", "vendorB", 1.1120, 1.1130), eurusd,
                                    1.1000, spotDate, start, maturity, 1e-4,
                                    QuoteConvention::Outright);
    CHECK(close(outrightQuote.outright(), 1.1125));
    CHECK(close(outrightQuote.points(), 125.0));
    CHECK(close(outrightQuote.points(), outrightToPoints(1.1000, outrightQuote.outright(), 1e-4)));

    const DayCounter act360(DayCount::Actual360);
    const double tau = act360.yearFraction(start, maturity);
    CHECK(close(pointsQuote.annualizedPremium(act360),
                (pointsQuote.outright() / pointsQuote.spot() - 1.0) / tau));

    // Negative points are valid while the outright stays positive.
    const FxSwapQuote discount(FxQuote("eurusd.3m", "vendorC", -130.0, -120.0), eurusd, 1.1000,
                               spotDate, start, maturity, 1e-4, QuoteConvention::Points);
    CHECK(close(discount.outright(), 1.0875));

    // Maturity at or before the start is rejected.
    CHECK(throwsInvalidArgument([&] {
        const FxSwapQuote bad(FxQuote("id", "source", 120.0, 130.0), eurusd, 1.1, spotDate,
                              maturity, start);
        (void)bad;
    }));
    CHECK(throwsInvalidArgument([&] {
        const FxSwapQuote bad(FxQuote("id", "source", 120.0, 130.0), eurusd, -1.1, spotDate, start,
                              maturity);
        (void)bad;
    }));
    CHECK(throwsInvalidArgument([&] {
        const FxSwapQuote bad(FxQuote("id", "source", 120.0, 130.0), FXDescriptor("XXX", "USD"),
                              1.1, spotDate, start, maturity);
        (void)bad;
    }));
    CHECK(throwsInvalidArgument([&] {
        const FxSwapQuote bad(FxQuote("id", "source", 120.0, 130.0), eurusd, 1.1, spotDate, start,
                              maturity, 0.0);
        (void)bad;
    }));
    CHECK(throwsInvalidArgument([&] {
        const FxSwapQuote bad(FxQuote("id", "source", -1.0, -0.5), eurusd, 1.1, spotDate, start,
                              maturity, 1e-4, QuoteConvention::Outright);
        (void)bad;
    }));
}

void testSpotDates() {
    QTA_LOG_INFO("test", "=== Spot Date Test ===");

    const Calendar target = Calendar::target();
    const Calendar sifma = Calendar::sifma();
    const Calendar uk = Calendar::unitedKingdom();

    // EUR/USD T+2: Good Friday and Easter Monday push settlement to Tuesday.
    CHECK(spotDate(Date(2026, 4, 2), target, sifma, 2, 2) == Date(2026, 4, 7));
    // Weekend roll.
    CHECK(spotDate(Date(2026, 9, 25), target, sifma, 2, 2) == Date(2026, 9, 28));
    // Month-end weekend (May 31 -> June 1): ModifiedFollowing rolls back to
    // Friday May 29 while Following crosses into June.
    CHECK(spotDate(Date(2026, 5, 29), target, sifma, 2, 2) == Date(2026, 5, 29));
    CHECK(spotDate(Date(2026, 5, 29), target, sifma, 2, 2, BusinessDayConvention::Following) ==
          Date(2026, 6, 1));

    // Joint-calendar overload: the later of the two lags governs.
    const Calendar joint = Calendar::joint(target, sifma);
    CHECK(spotDate(Date(2026, 9, 25), joint, 0, 2) == Date(2026, 9, 28));

    // GBP T+0 settles on the trade date when it is a business day.
    CHECK(spotDate(Date(2026, 9, 28), uk, 0) == Date(2026, 9, 28));
    // GBP T+0 over Good Friday and Easter Monday.
    CHECK(spotDate(Date(2026, 4, 3), uk, 0) == Date(2026, 4, 7));
    // GBP T+0 ModifiedFollowing at the August bank holiday rolls back.
    CHECK(spotDate(Date(2026, 8, 31), uk, 0, BusinessDayConvention::ModifiedFollowing) ==
          Date(2026, 8, 28));
    CHECK(spotDate(Date(2026, 8, 31), uk, 0, BusinessDayConvention::Following) == Date(2026, 9, 1));

    // The legacy datetime signature keeps its Following default.
    CHECK(quantape::datetime::spotDate(Date(2026, 4, 1), target, 2) == Date(2026, 4, 7));
}

} // namespace

int main() {
    try {
        testEnvelope();
        testPairSemantics();
        testSpotQuote();
        testPointsOutrightRoundTrip();
        testAnnualization();
        testSwapQuote();
        testSpotDates();

        QTA_LOG_INFO("test", "test_fx_quote: ok");
        return 0;
    } catch (const std::exception& e) {
        QTA_LOG_ERROR("test", "Error: {}", e.what());
        return 1;
    }
}
