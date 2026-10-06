// test_calendar.cpp — holiday rules, adjustments, advance, joint calendars
#include "quantape/datetime/Calendar.h"

#include <string>

#include "support/GtestSupport.h"

using namespace quantape::datetime;

TEST(CalendarRules, weekendMaskAndNoHolidays) {
    const Calendar weekends = Calendar::weekendsOnly();

    EXPECT_FALSE(weekends.isBusinessDay(Date(2026, 9, 26))); // Saturday
    EXPECT_FALSE(weekends.isBusinessDay(Date(2026, 9, 27))); // Sunday
    EXPECT_TRUE(weekends.isBusinessDay(Date(2026, 9, 28)));  // Monday
    EXPECT_TRUE(Calendar::noHolidays().isBusinessDay(Date(2026, 9, 26)));
}

TEST(CalendarRules, unitedStatesHolidaysAndAdHocClosures) {
    const Calendar sifma = Calendar::sifma();
    const Calendar fed = Calendar::federalReserve();

    // US: Independence Day 2026 is a Saturday, observed Friday July 3.
    EXPECT_FALSE(sifma.isBusinessDay(Date(2026, 7, 3)));
    EXPECT_FALSE(fed.isBusinessDay(Date(2026, 7, 3)));
    EXPECT_TRUE(sifma.isBusinessDay(Date(2026, 7, 6)));
    EXPECT_FALSE(sifma.isBusinessDay(Date(2026, 1, 1)));   // New Year
    EXPECT_FALSE(sifma.isBusinessDay(Date(2026, 1, 19)));  // MLK, 3rd Monday
    EXPECT_FALSE(sifma.isBusinessDay(Date(2026, 2, 16)));  // Presidents
    EXPECT_FALSE(sifma.isBusinessDay(Date(2026, 11, 26))); // Thanksgiving
    EXPECT_FALSE(sifma.isBusinessDay(Date(2026, 12, 25)));
    // Good Friday: SIFMA closed, Federal Reserve open.
    EXPECT_FALSE(sifma.isBusinessDay(Date(2026, 4, 3)));
    EXPECT_TRUE(fed.isBusinessDay(Date(2026, 4, 3)));
    // Ad-hoc closures.
    EXPECT_FALSE(sifma.isBusinessDay(Date(2001, 9, 12)));
    EXPECT_FALSE(sifma.isBusinessDay(Date(2012, 10, 30)));
    EXPECT_FALSE(sifma.isBusinessDay(Date(2018, 12, 5)));
}

TEST(CalendarRules, internationalHolidayClosures) {
    const Calendar target = Calendar::target();
    const Calendar uk = Calendar::unitedKingdom();
    const Calendar jp = Calendar::japan();

    {
        // TARGET: Easter holidays, May 1, Christmas/Boxing, no observed shifts.
        SCOPED_TRACE("TARGET");
        EXPECT_FALSE(target.isBusinessDay(Date(2026, 4, 3))); // Good Friday
        EXPECT_FALSE(target.isBusinessDay(Date(2026, 4, 6))); // Easter Monday
        EXPECT_FALSE(target.isBusinessDay(Date(2026, 5, 1)));
        EXPECT_FALSE(target.isBusinessDay(Date(2026, 12, 25)));
        EXPECT_FALSE(target.isBusinessDay(Date(2026, 12, 26)));
        EXPECT_TRUE(target.isBusinessDay(Date(2026, 12, 28)));
        EXPECT_TRUE(target.isBusinessDay(Date::parse("2028-01-03"))); // Jan 1 2028 Saturday
    }
    {
        // UK: bank holidays and one-offs.
        SCOPED_TRACE("UK");
        EXPECT_FALSE(uk.isBusinessDay(Date(2026, 5, 4)));  // Early May
        EXPECT_FALSE(uk.isBusinessDay(Date(2026, 5, 25))); // Spring
        EXPECT_FALSE(uk.isBusinessDay(Date(2026, 8, 31))); // Summer
        EXPECT_FALSE(uk.isBusinessDay(Date(2026, 12, 25)));
        EXPECT_FALSE(uk.isBusinessDay(Date(2026, 12, 26))); // Boxing (Saturday)
        EXPECT_TRUE(uk.isBusinessDay(Date(2026, 12, 29)));
        EXPECT_FALSE(uk.isBusinessDay(Date(2022, 9, 19))); // State funeral
        EXPECT_FALSE(uk.isBusinessDay(Date(2023, 5, 8)));  // Coronation
    }
    {
        // Japan: fixed, Happy Monday and equinox approximations.
        SCOPED_TRACE("Japan");
        EXPECT_FALSE(jp.isBusinessDay(Date(2026, 1, 1)));
        EXPECT_FALSE(jp.isBusinessDay(Date(2026, 1, 12))); // Coming of Age, 2nd Monday
        EXPECT_FALSE(jp.isBusinessDay(Date(2026, 2, 11)));
        EXPECT_FALSE(jp.isBusinessDay(Date(2026, 2, 23)));
        EXPECT_FALSE(jp.isBusinessDay(Date(2026, 3, 20))); // Vernal equinox
        EXPECT_FALSE(jp.isBusinessDay(Date(2026, 5, 4)));
        EXPECT_FALSE(jp.isBusinessDay(Date(2026, 7, 20))); // Marine, 3rd Monday
        EXPECT_FALSE(jp.isBusinessDay(Date(2026, 8, 11)));
        EXPECT_FALSE(jp.isBusinessDay(Date(2026, 9, 23)));  // Autumn equinox
        EXPECT_FALSE(jp.isBusinessDay(Date(2026, 10, 12))); // Sports, 2nd Monday
        EXPECT_FALSE(jp.isBusinessDay(Date(2026, 11, 23)));
        EXPECT_TRUE(jp.isBusinessDay(Date(2026, 3, 19)));
    }
}

TEST(CalendarConventions, adjustmentAdvanceAndBusinessDayCounting) {
    const Calendar sifma = Calendar::sifma();
    const Calendar target = Calendar::target();
    const Calendar weekends = Calendar::weekendsOnly();

    // Business-day conventions.
    const Date saturday = Date(2026, 10, 31);
    EXPECT_EQ(saturday.weekday(), Weekday::Saturday);
    EXPECT_EQ(sifma.adjust(saturday, BusinessDayConvention::Following), Date(2026, 11, 2));
    EXPECT_EQ(sifma.adjust(saturday, BusinessDayConvention::ModifiedFollowing), Date(2026, 10, 30));
    EXPECT_EQ(sifma.adjust(saturday, BusinessDayConvention::Preceding), Date(2026, 10, 30));
    EXPECT_EQ(sifma.adjust(Date(2026, 11, 1), BusinessDayConvention::Nearest), Date(2026, 11, 2));
    EXPECT_EQ(sifma.adjust(Date(2026, 11, 1), BusinessDayConvention::ModifiedPreceding),
              Date(2026, 11, 2));
    // Half-month: Saturday 15th follows into the second half -> previous day.
    const Date mid = Date(2026, 8, 15);
    EXPECT_EQ(mid.weekday(), Weekday::Saturday);
    EXPECT_EQ(sifma.adjust(mid, BusinessDayConvention::HalfMonthModifiedFollowing),
              Date(2026, 8, 14));
    EXPECT_EQ(sifma.adjust(Date(2026, 8, 8), BusinessDayConvention::HalfMonthModifiedFollowing),
              Date(2026, 8, 10)); // stays in the first half

    // Advance with end-of-month and adjustment.
    EXPECT_EQ(sifma.advance(Date(2026, 1, 31), 1, TimeUnit::Months,
                            BusinessDayConvention::ModifiedFollowing, true),
              Date(2026, 2, 27));
    // Feb 15 2026 is a Sunday; Feb 16 is Presidents' Day for SIFMA.
    EXPECT_EQ(sifma.advance(Date(2026, 1, 15), 1, TimeUnit::Months,
                            BusinessDayConvention::ModifiedFollowing, true),
              Date(2026, 2, 17));
    EXPECT_EQ(target.advance(Date(2026, 4, 2), 1, TimeUnit::Days), Date(2026, 4, 7));

    // Business-day counting: (from, to], weekends and holidays excluded.
    EXPECT_EQ(weekends.businessDaysBetween(Date(2020, 1, 6), Date(2020, 1, 13)), 5);
    EXPECT_EQ(weekends.businessDaysBetween(Date(2020, 1, 13), Date(2020, 1, 6)), -5);
    EXPECT_EQ(weekends.businessDaysBetween(Date(2020, 1, 6), Date(2020, 1, 6)), 0);
    EXPECT_EQ(sifma.businessDaysBetween(Date(2026, 7, 2), Date(2026, 7, 7)), 2); // Jul3 closed
}

TEST(CalendarComposition, jointExtraHolidaysAndReopenings) {
    const Calendar sifma = Calendar::sifma();
    const Calendar target = Calendar::target();

    // Joint calendar: union of holidays, union of weekend masks.
    const Calendar joint = Calendar::joint(sifma, target);
    EXPECT_FALSE(joint.isBusinessDay(Date(2026, 4, 3)));   // both closed
    EXPECT_FALSE(joint.isBusinessDay(Date(2026, 4, 6)));   // TARGET Easter Monday
    EXPECT_FALSE(joint.isBusinessDay(Date(2026, 11, 26))); // SIFMA Thanksgiving
    EXPECT_TRUE(joint.isBusinessDay(Date(2026, 4, 7)));
    EXPECT_EQ(joint.name(), "SIFMA+TARGET");

    // Per-instance extra holidays and reopenings.
    const Date event(2026, 9, 28);
    const Calendar custom = sifma.withExtraHolidays({event});
    EXPECT_FALSE(custom.isBusinessDay(event));
    EXPECT_TRUE(sifma.isBusinessDay(event));
    EXPECT_FALSE(sifma.withExtraHolidays({event}).withoutHoliday(event).isHoliday(event));
}

TEST(CalendarComposition, recycledRuleSetsKeepTheirOwnHolidayCache) {
    // Regression: the year cache was process-wide and keyed by the raw rule-set
    // address, so a freed temporary joint calendar's holidays leaked into the
    // next rule set that reused its heap block. Thanksgiving 2024 is a SIFMA
    // holiday but a TARGET business day.
    const Date thanksgiving(2024, 11, 28);
    for (int i = 0; i < 64; ++i) {
        SCOPED_TRACE(i);
        {
            const Calendar closed = Calendar::joint(Calendar::sifma(), Calendar::japan());
            ASSERT_FALSE(closed.isBusinessDay(thanksgiving));
        }
        const Calendar targetOnly = Calendar::joint(Calendar::target(), Calendar::weekendsOnly());
        EXPECT_TRUE(targetOnly.isBusinessDay(thanksgiving));
        EXPECT_TRUE(Calendar::target().isBusinessDay(thanksgiving));
        EXPECT_FALSE(Calendar::sifma().isBusinessDay(thanksgiving));
    }
}
