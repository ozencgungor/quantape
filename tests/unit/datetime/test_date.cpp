// test_date.cpp — Date conversions, arithmetic, parsing; naive reference sweep
#include "quantape/datetime/Date.h"
#include "quantape/datetime/Period.h"

#include <cstdint>
#include <random>
#include <stdexcept>
#include <string>

#include "support/GtestSupport.h"

using namespace quantape::datetime;

namespace {

// Independent reference implementation (Howard Hinnant's civil algorithms).
int refDaysFromCivil(int y, unsigned m, unsigned d) {
    y -= m <= 2;
    const int era = (y >= 0 ? y : y - 399) / 400;
    const unsigned yoe = static_cast<unsigned>(y - era * 400);
    const unsigned doy = (153 * (m + (m > 2 ? -3 : 9)) + 2) / 5 + d - 1;
    const unsigned doe = yoe * 365 + yoe / 4 - yoe / 100 + doy;
    return era * 146097 + static_cast<int>(doe) - 719468;
}

void refCivilFromDays(std::int64_t z, int& y, unsigned& m, unsigned& d) {
    z += 719468;
    const std::int64_t era = (z >= 0 ? z : z - 146096) / 146097;
    const unsigned doe = static_cast<unsigned>(z - era * 146097);
    const unsigned yoe = (doe - doe / 1460 + doe / 36524 - doe / 146096) / 365;
    y = static_cast<int>(yoe) + static_cast<int>(era * 400);
    const unsigned doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    const unsigned mp = (5 * doy + 2) / 153;
    d = doy - (153 * mp + 2) / 5 + 1;
    m = mp + (mp < 10 ? 3 : -9);
    y += (m <= 2);
}

} // namespace

TEST(DateArithmetic, knownSerialsWeekdaysAndMonthArithmetic) {
    // Known serials and weekdays.
    EXPECT_EQ(Date(1970, 1, 1).serial(), 0);
    EXPECT_EQ(Date(2000, 1, 1).serial(), 10957);
    EXPECT_EQ(Date(2026, 9, 28).weekday(), Weekday::Monday);
    EXPECT_EQ(Date(2000, 1, 1).weekday(), Weekday::Saturday);
    EXPECT_TRUE(Date::isLeapYear(2000) && !Date::isLeapYear(1900) && Date::isLeapYear(2024));

    // Month arithmetic: clipping and end-of-month preservation.
    EXPECT_EQ(Date(2021, 1, 31).plusMonths(1, false), Date(2021, 2, 28));
    EXPECT_EQ(Date(2021, 1, 31).plusMonths(1, true), Date(2021, 2, 28));
    EXPECT_EQ(Date(2020, 2, 29).plusMonths(1, true), Date(2020, 3, 31));
    EXPECT_EQ(Date(2020, 2, 29).plusYears(1, true), Date(2021, 2, 28));
    EXPECT_EQ(Date(2021, 4, 30).plusMonths(1, true), Date(2021, 5, 31));
    EXPECT_EQ(Date(2021, 1, 15).plusMonths(13), Date(2022, 2, 15));
    EXPECT_EQ(Date(2021, 1, 15).plusMonths(-13), Date(2019, 12, 15));
    EXPECT_EQ(Date(2020, 1, 31).endOfMonth(), Date(2020, 1, 31));
    EXPECT_EQ(Date(2020, 2, 10).endOfMonth(), Date(2020, 2, 29));
    EXPECT_EQ(Date(2020, 2, 10).dayOfYear(), 41);
    EXPECT_EQ(Date(2020, 12, 31).dayOfYear(), 366);
}

TEST(DateArithmetic, civilSweepMatchesReference) {
    // Full sweep 1600-2400 against the naive reference.
    for (int y = 1600; y <= 2400; ++y) {
        for (unsigned m = 1; m <= 12; ++m) {
            const unsigned last = Date::daysInMonth(y, m);
            for (unsigned d = 1; d <= last; ++d) {
                SCOPED_TRACE(::testing::Message() << y << '-' << m << '-' << d);
                const Date date(y, m, d);
                EXPECT_EQ(date.serial(), refDaysFromCivil(y, m, d));
                int ry = 0;
                unsigned rm = 0, rd = 0;
                refCivilFromDays(date.serial(), ry, rm, rd);
                EXPECT_TRUE(ry == y && rm == m && rd == d);
                EXPECT_TRUE(date.year() == y && date.month() == m && date.dayOfMonth() == d);
            }
        }
    }
}

TEST(DateArithmetic, randomInt32SerialRoundTrip) {
    // Full int32 serial sweep in random samples (overflow safety).
    std::mt19937_64 rng(20260928);
    for (int i = 0; i < 20000; ++i) {
        const std::int32_t serial = static_cast<std::int32_t>(rng());
        SCOPED_TRACE(::testing::Message() << "serial=" << serial);
        const Date date = Date::fromSerial(serial);
        int ry = 0;
        unsigned rm = 0, rd = 0;
        refCivilFromDays(serial, ry, rm, rd);
        EXPECT_TRUE(date.year() == ry && date.month() == rm && date.dayOfMonth() == rd);
        EXPECT_EQ(Date(ry, rm, rd).serial(), serial);
    }
}

TEST(DateParsing, isoRoundTripAndRejectsInvalidInput) {
    EXPECT_EQ(Date::parse("2026-09-28"), Date(2026, 9, 28));
    EXPECT_EQ(Date(2026, 9, 28).toIso(), "2026-09-28");
    EXPECT_THROW((void)Date::parse("2026-13-01"), std::invalid_argument);
}

TEST(PeriodParsing, parseAdvanceAndOrdering) {
    EXPECT_EQ(Period::parse("3M").advance(Date(2021, 1, 31)), Date(2021, 4, 30));
    EXPECT_EQ(Period::parse("1Y").advance(Date(2020, 2, 29)), Date(2021, 2, 28));
    EXPECT_EQ(Period::parse("2W").advance(Date(2021, 1, 1)), Date(2021, 1, 15));
    EXPECT_EQ(Period::parse("10D").toString(), "10D");
    EXPECT_EQ(Period::parse("1Y"), Period(1, TimeUnit::Years));
    EXPECT_TRUE(Period(1, TimeUnit::Years) > Period(11, TimeUnit::Months));
}
