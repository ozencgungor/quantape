// test_schedule.cpp — schedule generation, ICMA/BUS252, time conversion
#include "quantape/datetime/Imm.h"
#include "quantape/datetime/Schedule.h"
#include "quantape/datetime/TimeConversion.h"

#include <cmath>
#include <cstddef>
#include <vector>

#include "support/GtestSupport.h"

using namespace quantape::datetime;

namespace {

bool close(double a, double b, double tol = 1e-14) {
    return std::abs(a - b) <= tol * (1.0 + std::abs(b));
}

} // namespace

TEST(ScheduleGeneration, regularSemiannualTimeConversionAndIcma) {
    const Calendar target = Calendar::target();

    // Regular semiannual swap: 5 years, 11 dates.
    const Schedule semi(Date::parse("2026-01-15"), Date::parse("2031-01-15"),
                        Period(6, TimeUnit::Months), target,
                        BusinessDayConvention::ModifiedFollowing, DateGeneration::Forward, false);
    EXPECT_EQ(semi.size(), 11);
    EXPECT_EQ(semi.startDate(), Date(2026, 1, 15));
    EXPECT_EQ(semi.endDate(), Date(2031, 1, 15));
    EXPECT_EQ(semi.dates()[1], Date(2026, 7, 15));
    for (std::size_t i = 0; i + 1 < semi.size(); ++i) {
        EXPECT_TRUE(semi.isRegular(i));
    }
    EXPECT_EQ(semi.frequency(), Frequency::Semiannual);
    EXPECT_EQ(semi.tenorMonths(), 6);

    // Time conversion: first date maps to zero, monotone, grid steps match.
    const DayCounter act365(DayCount::Actual365Fixed);
    const std::vector<double> times = scheduleTimes(semi.dates(), semi.startDate(), act365);
    EXPECT_EQ(times.size(), semi.size());
    EXPECT_EQ(times.front(), 0.0);
    for (std::size_t i = 1; i < times.size(); ++i) {
        EXPECT_TRUE(times[i] > times[i - 1]);
    }
    EXPECT_TRUE(
        close(times.back(), static_cast<double>(semi.endDate() - semi.startDate()) / 365.0, 1e-12));
    EXPECT_EQ(timeGridFromDates(semi.dates(), semi.startDate(), act365).nSteps(), semi.size() - 1);

    // ACT/ACT ICMA on the regular schedule: each semi period contributes 0.5.
    DayCountContext context;
    context.schedule = &semi;
    DayCounter icma(DayCount::ActualActualICMA, context);
    EXPECT_TRUE(close(icma.yearFractionUncached(semi.dates()[0], semi.dates()[2]), 1.0));
    EXPECT_TRUE(close(icma.yearFractionUncached(semi.dates()[0], semi.dates()[1]), 0.5));
    EXPECT_TRUE(close(icma.yearFractionUncached(semi.dates()[0], semi.endDate()), 5.0));
}

TEST(ScheduleGeneration, forwardAndBackwardStubGeneration) {
    const Calendar target = Calendar::target();

    // Forward yearly with a short back stub.
    const Schedule stub(Date::parse("2026-02-15"), Date::parse("2030-08-15"),
                        Period(1, TimeUnit::Years), target,
                        BusinessDayConvention::ModifiedFollowing, DateGeneration::Forward, false);
    EXPECT_EQ(stub.size(), 6);
    EXPECT_EQ(stub.startDate(), Date(2026, 2, 16)); // Feb 15 2026 is a Sunday
    EXPECT_EQ(stub.endDate(), Date(2030, 8, 15));
    EXPECT_TRUE(stub.isRegular(0));
    EXPECT_TRUE(stub.isRegular(3));
    EXPECT_FALSE(stub.isRegular(4)); // 2030-02-15 -> 2030-08-15 stub
    EXPECT_FALSE(stub.isRegular(5));

    // Backward yearly with a front stub.
    const Schedule backStub(
        Date::parse("2026-02-15"), Date::parse("2030-08-15"), Period(1, TimeUnit::Years), target,
        BusinessDayConvention::ModifiedFollowing, DateGeneration::Backward, false);
    EXPECT_EQ(backStub.size(), 6);
    EXPECT_EQ(backStub.startDate(), Date(2026, 2, 16));
    EXPECT_FALSE(backStub.isRegular(0)); // front stub
    EXPECT_TRUE(backStub.isRegular(1));
    EXPECT_EQ(backStub.endDate(), Date(2030, 8, 15));
}

TEST(ScheduleGeneration, endOfMonthZeroRulesAndSpotDate) {
    const Calendar target = Calendar::target();

    // End-of-month monthly schedule.
    const Schedule monthly(Date::parse("2026-01-31"), Date::parse("2027-01-31"),
                           Period(1, TimeUnit::Months), target,
                           BusinessDayConvention::ModifiedFollowing, DateGeneration::Forward, true);
    EXPECT_EQ(monthly.size(), 13);
    EXPECT_EQ(monthly.unadjustedDates().back(), Date(2027, 1, 31));
    EXPECT_EQ(monthly.endDate(), Date(2027, 1, 29)); // Jan 31 is a Sunday, MF rolls back
    for (std::size_t i = 0; i + 1 < monthly.size(); ++i) {
        EXPECT_TRUE(monthly.isRegular(i));
    }

    // Zero rule: two dates.
    const Schedule zero(Date::parse("2026-01-15"), Date::parse("2027-01-15"),
                        Period(1, TimeUnit::Years), target, BusinessDayConvention::Following,
                        DateGeneration::Zero, false);
    EXPECT_EQ(zero.size(), 2);

    // Zero rule ignores the tenor, so a zero tenor is accepted.
    const Schedule zeroTenor(Date::parse("2026-01-15"), Date::parse("2027-01-15"),
                             Period(0, TimeUnit::Months), target, BusinessDayConvention::Following,
                             DateGeneration::Zero, false);
    EXPECT_EQ(zeroTenor.size(), 2);

    // Spot date: T+2 with adjustment.
    EXPECT_EQ(spotDate(Date(2026, 4, 1), target, 2), Date(2026, 4, 7)); // Easter 2026
}

TEST(ScheduleConventions, icmaStubAndBus252Contexts) {
    const Calendar target = Calendar::target();

    // ACT/ACT ICMA with a short front stub on a real coupon schedule.
    {
        const Schedule stubSchedule(
            Date::parse("2026-02-15"), Date::parse("2030-08-15"), Period(1, TimeUnit::Years),
            target, BusinessDayConvention::ModifiedFollowing, DateGeneration::Forward, false);
        DayCountContext context;
        context.schedule = &stubSchedule;
        DayCounter icma(DayCount::ActualActualICMA, context);
        const double full =
            icma.yearFractionUncached(stubSchedule.dates()[0], stubSchedule.endDate());
        EXPECT_TRUE(full > 4.4 && full < 4.6); // ~4.5y of coupon time
    }

    // BUS/252 with a calendar in the context.
    {
        const Calendar weekends = Calendar::weekendsOnly();
        DayCountContext context;
        context.calendar = &weekends;
        DayCounter bus252(DayCount::Bus252, context);
        EXPECT_TRUE(
            close(bus252.yearFractionUncached(Date(2020, 1, 6), Date(2020, 1, 13)), 5.0 / 252.0));
        EXPECT_EQ(bus252.yearFractionUncached(Date(2020, 1, 6), Date(2020, 1, 6)), 0.0);
    }
}

TEST(ScheduleConventions, immDatesGridAndTerminationConventions) {
    // IMM dates and IMM schedule grid.
    EXPECT_EQ(quantape::datetime::immDate(2027, 3), quantape::datetime::Date::parse("2027-03-17"));
    EXPECT_EQ(quantape::datetime::immDate(2027, 6), quantape::datetime::Date::parse("2027-06-16"));
    EXPECT_EQ(quantape::datetime::immDate(2027, 9), quantape::datetime::Date::parse("2027-09-15"));
    EXPECT_EQ(quantape::datetime::immDate(2027, 12), quantape::datetime::Date::parse("2027-12-15"));
    EXPECT_TRUE(quantape::datetime::isIMMDate(quantape::datetime::Date::parse("2027-03-17")));
    EXPECT_FALSE(quantape::datetime::isIMMDate(quantape::datetime::Date::parse("2027-03-10")));
    EXPECT_EQ(quantape::datetime::nextIMMDate(quantape::datetime::Date::parse("2027-03-17")),
              quantape::datetime::Date::parse("2027-06-16"));
    EXPECT_EQ(quantape::datetime::nextIMMDate(quantape::datetime::Date::parse("2027-12-16")),
              quantape::datetime::Date::parse("2028-03-15"));
    const std::vector<quantape::datetime::Date> immGrid = quantape::datetime::immSchedule(
        quantape::datetime::Date::parse("2026-09-29"),
        quantape::datetime::Date::parse("2027-12-15"), quantape::datetime::Calendar::noHolidays());
    EXPECT_EQ(immGrid.size(), 6);
    EXPECT_EQ(immGrid.front(), quantape::datetime::Date::parse("2026-09-29"));
    EXPECT_EQ(immGrid[1], quantape::datetime::Date::parse("2026-12-16"));
    EXPECT_EQ(immGrid.back(), quantape::datetime::Date::parse("2027-12-15"));

    // Separate effective/termination conventions.
    const quantape::datetime::Schedule anchored(
        quantape::datetime::Date::parse("2027-07-31"),
        quantape::datetime::Date::parse("2028-04-30"),
        quantape::datetime::Period(3, quantape::datetime::TimeUnit::Months),
        quantape::datetime::Calendar::weekendsOnly(),
        quantape::datetime::BusinessDayConvention::ModifiedFollowing,
        quantape::datetime::DateGeneration::Forward, false,
        quantape::datetime::BusinessDayConvention::Unadjusted,
        quantape::datetime::BusinessDayConvention::Following);
    EXPECT_EQ(anchored.startDate(), quantape::datetime::Date::parse("2027-07-31"));
    EXPECT_EQ(anchored.dates()[1], quantape::datetime::Date::parse("2027-10-29"));
    EXPECT_EQ(anchored.endDate(), quantape::datetime::Date::parse("2028-05-01"));
}
