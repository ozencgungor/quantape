// test_calendar.cpp — holiday rules, adjustments, advance, joint calendars
#include "quantape/datetime/Calendar.h"
#include "quantape/log/Log.h"
#include "quantape/util/Check.h"

using namespace quantape::datetime;

int main() {
    const Calendar sifma = Calendar::sifma();
    const Calendar fed = Calendar::federalReserve();
    const Calendar target = Calendar::target();
    const Calendar uk = Calendar::unitedKingdom();
    const Calendar jp = Calendar::japan();
    const Calendar weekends = Calendar::weekendsOnly();

    // Weekend masking.
    CHECK(!weekends.isBusinessDay(Date(2026, 9, 26))); // Saturday
    CHECK(!weekends.isBusinessDay(Date(2026, 9, 27))); // Sunday
    CHECK(weekends.isBusinessDay(Date(2026, 9, 28)));  // Monday
    CHECK(Calendar::noHolidays().isBusinessDay(Date(2026, 9, 26)));

    // US: Independence Day 2026 is a Saturday, observed Friday July 3.
    CHECK(!sifma.isBusinessDay(Date(2026, 7, 3)));
    CHECK(!fed.isBusinessDay(Date(2026, 7, 3)));
    CHECK(sifma.isBusinessDay(Date(2026, 7, 6)));
    CHECK(!sifma.isBusinessDay(Date(2026, 1, 1)));   // New Year
    CHECK(!sifma.isBusinessDay(Date(2026, 1, 19)));  // MLK, 3rd Monday
    CHECK(!sifma.isBusinessDay(Date(2026, 2, 16)));  // Presidents
    CHECK(!sifma.isBusinessDay(Date(2026, 11, 26))); // Thanksgiving
    CHECK(!sifma.isBusinessDay(Date(2026, 12, 25)));
    // Good Friday: SIFMA closed, Federal Reserve open.
    CHECK(!sifma.isBusinessDay(Date(2026, 4, 3)));
    CHECK(fed.isBusinessDay(Date(2026, 4, 3)));
    // Ad-hoc closures.
    CHECK(!sifma.isBusinessDay(Date(2001, 9, 12)));
    CHECK(!sifma.isBusinessDay(Date(2012, 10, 30)));
    CHECK(!sifma.isBusinessDay(Date(2018, 12, 5)));

    // TARGET: Easter holidays, May 1, Christmas/Boxing, no observed shifts.
    CHECK(!target.isBusinessDay(Date(2026, 4, 3))); // Good Friday
    CHECK(!target.isBusinessDay(Date(2026, 4, 6))); // Easter Monday
    CHECK(!target.isBusinessDay(Date(2026, 5, 1)));
    CHECK(!target.isBusinessDay(Date(2026, 12, 25)));
    CHECK(!target.isBusinessDay(Date(2026, 12, 26)));
    CHECK(target.isBusinessDay(Date(2026, 12, 28)));
    CHECK(target.isBusinessDay(Date::parse("2028-01-03"))); // Jan 1 2028 Saturday, no shift

    // UK: bank holidays and one-offs.
    CHECK(!uk.isBusinessDay(Date(2026, 5, 4)));  // Early May
    CHECK(!uk.isBusinessDay(Date(2026, 5, 25))); // Spring
    CHECK(!uk.isBusinessDay(Date(2026, 8, 31))); // Summer
    CHECK(!uk.isBusinessDay(Date(2026, 12, 25)));
    CHECK(!uk.isBusinessDay(Date(2026, 12, 26))); // Boxing (Saturday)
    CHECK(uk.isBusinessDay(Date(2026, 12, 29)));
    CHECK(!uk.isBusinessDay(Date(2022, 9, 19))); // State funeral
    CHECK(!uk.isBusinessDay(Date(2023, 5, 8)));  // Coronation

    // Japan: fixed, Happy Monday and equinox approximations.
    CHECK(!jp.isBusinessDay(Date(2026, 1, 1)));
    CHECK(!jp.isBusinessDay(Date(2026, 1, 12))); // Coming of Age, 2nd Monday
    CHECK(!jp.isBusinessDay(Date(2026, 2, 11)));
    CHECK(!jp.isBusinessDay(Date(2026, 2, 23)));
    CHECK(!jp.isBusinessDay(Date(2026, 3, 20))); // Vernal equinox
    CHECK(!jp.isBusinessDay(Date(2026, 5, 4)));
    CHECK(!jp.isBusinessDay(Date(2026, 7, 20))); // Marine, 3rd Monday
    CHECK(!jp.isBusinessDay(Date(2026, 8, 11)));
    CHECK(!jp.isBusinessDay(Date(2026, 9, 23)));  // Autumn equinox
    CHECK(!jp.isBusinessDay(Date(2026, 10, 12))); // Sports, 2nd Monday
    CHECK(!jp.isBusinessDay(Date(2026, 11, 23)));
    CHECK(jp.isBusinessDay(Date(2026, 3, 19)));

    // Business-day conventions.
    const Date saturday = Date(2026, 10, 31);
    CHECK(saturday.weekday() == Weekday::Saturday);
    CHECK(sifma.adjust(saturday, BusinessDayConvention::Following) == Date(2026, 11, 2));
    CHECK(sifma.adjust(saturday, BusinessDayConvention::ModifiedFollowing) == Date(2026, 10, 30));
    CHECK(sifma.adjust(saturday, BusinessDayConvention::Preceding) == Date(2026, 10, 30));
    CHECK(sifma.adjust(Date(2026, 11, 1), BusinessDayConvention::Nearest) == Date(2026, 11, 2));
    CHECK(sifma.adjust(Date(2026, 11, 1), BusinessDayConvention::ModifiedPreceding) ==
          Date(2026, 11, 2));
    // Half-month: Saturday 15th follows into the second half -> previous day.
    const Date mid = Date(2026, 8, 15);
    CHECK(mid.weekday() == Weekday::Saturday);
    CHECK(sifma.adjust(mid, BusinessDayConvention::HalfMonthModifiedFollowing) ==
          Date(2026, 8, 14));
    CHECK(sifma.adjust(Date(2026, 8, 8), BusinessDayConvention::HalfMonthModifiedFollowing) ==
          Date(2026, 8, 10)); // stays in the first half

    // Advance with end-of-month and adjustment.
    CHECK(sifma.advance(Date(2026, 1, 31), 1, TimeUnit::Months,
                        BusinessDayConvention::ModifiedFollowing, true) == Date(2026, 2, 27));
    // Feb 15 2026 is a Sunday; Feb 16 is Presidents' Day for SIFMA.
    CHECK(sifma.advance(Date(2026, 1, 15), 1, TimeUnit::Months,
                        BusinessDayConvention::ModifiedFollowing, true) == Date(2026, 2, 17));
    CHECK(target.advance(Date(2026, 4, 2), 1, TimeUnit::Days) == Date(2026, 4, 7));

    // Business-day counting: (from, to], weekends and holidays excluded.
    CHECK(weekends.businessDaysBetween(Date(2020, 1, 6), Date(2020, 1, 13)) == 5);
    CHECK(weekends.businessDaysBetween(Date(2020, 1, 13), Date(2020, 1, 6)) == -5);
    CHECK(weekends.businessDaysBetween(Date(2020, 1, 6), Date(2020, 1, 6)) == 0);
    CHECK(sifma.businessDaysBetween(Date(2026, 7, 2), Date(2026, 7, 7)) == 2); // Jul3 closed

    // Joint calendar: union of holidays, union of weekend masks.
    const Calendar joint = Calendar::joint(sifma, target);
    CHECK(!joint.isBusinessDay(Date(2026, 4, 3)));   // both closed
    CHECK(!joint.isBusinessDay(Date(2026, 4, 6)));   // TARGET Easter Monday
    CHECK(!joint.isBusinessDay(Date(2026, 11, 26))); // SIFMA Thanksgiving
    CHECK(joint.isBusinessDay(Date(2026, 4, 7)));
    CHECK(joint.name() == "SIFMA+TARGET");

    // Per-instance extra holidays and reopenings.
    const Date event(2026, 9, 28);
    const Calendar custom = sifma.withExtraHolidays({event});
    CHECK(!custom.isBusinessDay(event));
    CHECK(sifma.isBusinessDay(event));
    CHECK(!sifma.withExtraHolidays({event}).withoutHoliday(event).isHoliday(event));

    QTA_LOG_INFO("test", "test_calendar: ok");
    return 0;
}
