// test_fx_quote.cpp — FX quote envelopes, forward-point conversion,
// annualization and joint-calendar spot settlement.
#include "quantape/math/StanMath.h"

#include "quantape/datetime/BusinessDayConvention.h"
#include "quantape/datetime/Calendar.h"
#include "quantape/datetime/Date.h"
#include "quantape/datetime/DayCounter.h"
#include "quantape/markets/Data/FxQuote.h"

#include <cmath>
#include <stdexcept>

#include "support/GtestSupport.h"

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

// Shared joint calendar for the EUR/USD spot gates, kept process-lifetime for
// reuse across the cases that need it.
const Calendar& targetSifmaJoint() {
    static const Calendar joint = Calendar::joint(Calendar::target(), Calendar::sifma());
    return joint;
}

} // namespace

TEST(FxQuoteEnvelope, midSpreadAndOrdering) {
    const FxQuote averaged("eurusd.spot", "vendorA", 1.0998, 1.1002);
    EXPECT_EQ(averaged.id(), "eurusd.spot");
    EXPECT_EQ(averaged.source(), "vendorA");
    EXPECT_FALSE(averaged.hasMid());
    EXPECT_TRUE(close(averaged.mid(), 1.1000));
    EXPECT_TRUE(close(averaged.spread(), 0.0004));

    const FxQuote observed("eurusd.spot", "vendorB", 1.0999, 1.1001, 1.1000);
    EXPECT_TRUE(observed.hasMid());
    EXPECT_TRUE(close(observed.mid(), 1.1000));

    // Forward points may be negative; only the spread ordering is enforced.
    const FxQuote negativePoints("eurusd.3m", "vendorA", -130.0, -120.0);
    EXPECT_TRUE(close(negativePoints.mid(), -125.0));

    EXPECT_TRUE(throwsInvalidArgument([] {
        const FxQuote inverted("id", "source", 1.2, 1.1);
        (void)inverted;
    }));
    EXPECT_TRUE(throwsInvalidArgument([] {
        const FxQuote outside("id", "source", 1.0, 1.2, 1.3);
        (void)outside;
    }));
}

TEST(FxDescriptor, pairIdentifierAndDomesticForeign) {
    const FXDescriptor eurusd("EUR", "USD", "2026-09-29");
    EXPECT_EQ(eurusd.baseCcy, "EUR");
    EXPECT_EQ(eurusd.quoteCcy, "USD");
    EXPECT_EQ(eurusd.pair(), "EURUSD");
    EXPECT_EQ(eurusd.identifier(), "EURUSD.2026-09-29");
    EXPECT_TRUE(eurusd.hasKnownCurrencies());

    const FXDescriptor usdjpy("USD", "JPY");
    EXPECT_EQ(usdjpy.pair(), "USDJPY");
    EXPECT_EQ(usdjpy.identifier(), "USDJPY.");

    const FXDescriptor unknown("XXX", "USD");
    EXPECT_FALSE(unknown.hasKnownCurrencies());

    // Deprecated aliases keep the old domestic/foreign meaning: the domestic
    // currency is the quote and the foreign currency is the base.
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
    EXPECT_EQ(eurusd.domesticCcy(), "USD");
    EXPECT_EQ(eurusd.foreignCcy(), "EUR");
#pragma GCC diagnostic pop
}

TEST(FxSpotQuote, spotMidSpreadAndValidation) {
    const FXDescriptor eurusd("EUR", "USD", "2026-09-29");
    const FxSpotQuote quote(FxQuote("eurusd.spot", "vendorA", 1.0998, 1.1002), eurusd, 1.1000);
    EXPECT_EQ(quote.baseCcy(), "EUR");
    EXPECT_EQ(quote.quoteCcy(), "USD");
    EXPECT_EQ(quote.descriptor().pair(), "EURUSD");
    EXPECT_TRUE(close(quote.spot(), 1.1000));
    EXPECT_TRUE(close(quote.mid(), 1.1000));
    EXPECT_TRUE(close(quote.spread(), 0.0004));

    EXPECT_TRUE(throwsInvalidArgument([] {
        const FxSpotQuote bad(FxQuote("id", "source", 1.0, 1.1), FXDescriptor("EUR", "USD"), -1.0);
        (void)bad;
    }));
    EXPECT_TRUE(throwsInvalidArgument([] {
        const FxSpotQuote bad(FxQuote("id", "source", 1.0, 1.1), FXDescriptor("XXX", "USD"), 1.1);
        (void)bad;
    }));
    EXPECT_TRUE(throwsInvalidArgument([] {
        const FxSpotQuote bad(FxQuote("id", "source", 1.2, 1.1), FXDescriptor("EUR", "USD"), 1.1);
        (void)bad;
    }));
}

TEST(FxQuoteConversion, pointsOutrightRoundTrip) {
    const double spot = 1.1000;
    const double points = 125.0;
    const double scale = 1e-4;
    const double outright = pointsToOutright(spot, points, scale);
    EXPECT_TRUE(close(outright, 1.1125));
    EXPECT_TRUE(close(outrightToPoints(spot, outright, scale), points));
    EXPECT_TRUE(close(convertQuote(points, QuoteConvention::Points, spot, scale), outright));
    EXPECT_TRUE(close(convertQuote(outright, QuoteConvention::Outright, spot, scale), points));

    // JPY-style scaling: one point is 1e-2.
    EXPECT_TRUE(close(pointsToOutright(157.25, 35.0, 1e-2), 157.60));
    EXPECT_TRUE(close(outrightToPoints(157.25, 157.60, 1e-2), 35.0));

    // Discount points round-trip as well.
    EXPECT_TRUE(close(pointsToOutright(spot, -125.0, scale), 1.0875));
    EXPECT_TRUE(close(outrightToPoints(spot, 1.0875, scale), -125.0));
}

TEST(FxAnnualization, forwardPremiumRoundTrip) {
    const DayCounter act360(DayCount::Actual360);
    const Date start(2026, 9, 29);
    const Date maturity(2027, 3, 29);
    const double tau = act360.yearFraction(start, maturity);
    EXPECT_TRUE(close(tau, 181.0 / 360.0));

    const double premium = annualizedForwardPremium(1.1, 1.1125, start, maturity, act360);
    EXPECT_TRUE(close(premium, (1.1125 / 1.1 - 1.0) / tau));
    EXPECT_TRUE(close(outrightFromPremium(1.1, premium, start, maturity, act360), 1.1125));

    EXPECT_TRUE(throwsInvalidArgument([] {
        const DayCounter act360(DayCount::Actual360);
        (void)annualizedForwardPremium(1.1, 1.2, Date(2026, 3, 29), Date(2026, 3, 29), act360);
    }));
}

TEST(FxSwapQuote, pointsOutrightAndValidation) {
    const FXDescriptor eurusd("EUR", "USD", "2026-09-29");
    const Date spotDate(2026, 10, 1);
    const Date start(2026, 10, 1);
    const Date maturity(2027, 1, 4);

    const FxSwapQuote pointsQuote(FxQuote("eurusd.3m", "vendorA", 120.0, 130.0), eurusd, 1.1000,
                                  spotDate, start, maturity, 1e-4, QuoteConvention::Points);
    EXPECT_TRUE(pointsQuote.convention() == QuoteConvention::Points);
    EXPECT_TRUE(pointsQuote.spotDate() == spotDate);
    EXPECT_TRUE(pointsQuote.nearDate() == start);
    EXPECT_TRUE(pointsQuote.farDate() == maturity);
    EXPECT_TRUE(close(pointsQuote.points(), 125.0));
    EXPECT_TRUE(close(pointsQuote.bidPoints(), 120.0));
    EXPECT_TRUE(close(pointsQuote.askPoints(), 130.0));
    EXPECT_TRUE(close(pointsQuote.outright(), 1.1125));
    EXPECT_TRUE(close(pointsQuote.bidOutright(), 1.1000 + 120.0 * 1e-4));
    EXPECT_TRUE(close(pointsQuote.askOutright(), 1.1000 + 130.0 * 1e-4));

    const FxSwapQuote outrightQuote(FxQuote("eurusd.3m.o", "vendorB", 1.1120, 1.1130), eurusd,
                                    1.1000, spotDate, start, maturity, 1e-4,
                                    QuoteConvention::Outright);
    EXPECT_TRUE(close(outrightQuote.outright(), 1.1125));
    EXPECT_TRUE(close(outrightQuote.points(), 125.0));
    EXPECT_TRUE(
        close(outrightQuote.points(), outrightToPoints(1.1000, outrightQuote.outright(), 1e-4)));

    const DayCounter act360(DayCount::Actual360);
    const double tau = act360.yearFraction(start, maturity);
    EXPECT_TRUE(close(pointsQuote.annualizedPremium(act360),
                      (pointsQuote.outright() / pointsQuote.spot() - 1.0) / tau));

    // Negative points are valid while the outright stays positive.
    const FxSwapQuote discount(FxQuote("eurusd.3m", "vendorC", -130.0, -120.0), eurusd, 1.1000,
                               spotDate, start, maturity, 1e-4, QuoteConvention::Points);
    EXPECT_TRUE(close(discount.outright(), 1.0875));

    // Maturity at or before the start is rejected.
    EXPECT_TRUE(throwsInvalidArgument([&] {
        const FxSwapQuote bad(FxQuote("id", "source", 120.0, 130.0), eurusd, 1.1, spotDate,
                              maturity, start);
        (void)bad;
    }));
    EXPECT_TRUE(throwsInvalidArgument([&] {
        const FxSwapQuote bad(FxQuote("id", "source", 120.0, 130.0), eurusd, -1.1, spotDate, start,
                              maturity);
        (void)bad;
    }));
    EXPECT_TRUE(throwsInvalidArgument([&] {
        const FxSwapQuote bad(FxQuote("id", "source", 120.0, 130.0), FXDescriptor("XXX", "USD"),
                              1.1, spotDate, start, maturity);
        (void)bad;
    }));
    EXPECT_TRUE(throwsInvalidArgument([&] {
        const FxSwapQuote bad(FxQuote("id", "source", 120.0, 130.0), eurusd, 1.1, spotDate, start,
                              maturity, 0.0);
        (void)bad;
    }));
    EXPECT_TRUE(throwsInvalidArgument([&] {
        const FxSwapQuote bad(FxQuote("id", "source", -1.0, -0.5), eurusd, 1.1, spotDate, start,
                              maturity, 1e-4, QuoteConvention::Outright);
        (void)bad;
    }));
}

TEST(FxSpotDate, holidayRollsAndJointCalendar) {
    const Calendar target = Calendar::target();
    const Calendar uk = Calendar::unitedKingdom();

    // EUR/USD T+2: Good Friday and Easter Monday push settlement to Tuesday.
    EXPECT_TRUE(spotDate(Date(2026, 4, 2), targetSifmaJoint(), 2, 2) == Date(2026, 4, 7));
    // Weekend roll.
    EXPECT_TRUE(spotDate(Date(2026, 9, 25), targetSifmaJoint(), 2, 2) == Date(2026, 9, 28));
    // Month-end weekend (May 31 -> June 1): ModifiedFollowing rolls back to
    // Friday May 29 while Following crosses into June.
    EXPECT_TRUE(spotDate(Date(2026, 5, 29), targetSifmaJoint(), 2, 2) == Date(2026, 5, 29));
    EXPECT_TRUE(spotDate(Date(2026, 5, 29), targetSifmaJoint(), 2, 2,
                         BusinessDayConvention::Following) == Date(2026, 6, 1));

    // Joint-calendar overload: the later of the two lags governs.
    EXPECT_TRUE(spotDate(Date(2026, 9, 25), targetSifmaJoint(), 0, 2) == Date(2026, 9, 28));

    // GBP T+0 settles on the trade date when it is a business day.
    EXPECT_TRUE(spotDate(Date(2026, 9, 28), uk, 0) == Date(2026, 9, 28));
    // GBP T+0 over Good Friday and Easter Monday.
    EXPECT_TRUE(spotDate(Date(2026, 4, 3), uk, 0) == Date(2026, 4, 7));
    // GBP T+0 ModifiedFollowing at the August bank holiday rolls back.
    EXPECT_TRUE(spotDate(Date(2026, 8, 31), uk, 0, BusinessDayConvention::ModifiedFollowing) ==
                Date(2026, 8, 28));
    EXPECT_TRUE(spotDate(Date(2026, 8, 31), uk, 0, BusinessDayConvention::Following) ==
                Date(2026, 9, 1));

    // The legacy datetime signature keeps its Following default.
    EXPECT_TRUE(quantape::datetime::spotDate(Date(2026, 4, 1), target, 2) == Date(2026, 4, 7));
}
