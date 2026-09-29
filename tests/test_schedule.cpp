// test_schedule.cpp — schedule generation, ICMA/BUS252, time conversion
#include "quantape/log/Log.h"
#include "quantape/util/Check.h"
#include "quantape/datetime/Schedule.h"
#include "quantape/datetime/TimeConversion.h"

#include <cmath>

using namespace quantape::datetime;

namespace {

bool close(double a, double b, double tol = 1e-14) {
    return std::abs(a - b) <= tol * (1.0 + std::abs(b));
}

}  // namespace

int main() {
    const Calendar target = Calendar::target();
    const Calendar weekends = Calendar::weekendsOnly();

    // Regular semiannual swap: 5 years, 11 dates.
    const Schedule semi(Date::parse("2026-01-15"), Date::parse("2031-01-15"),
                        Period(6, TimeUnit::Months), target,
                        BusinessDayConvention::ModifiedFollowing, DateGeneration::Forward, false);
    CHECK(semi.size() == 11);
    CHECK(semi.startDate() == Date(2026, 1, 15));
    CHECK(semi.endDate() == Date(2031, 1, 15));
    CHECK(semi.dates()[1] == Date(2026, 7, 15));
    for (std::size_t i = 0; i + 1 < semi.size(); ++i) {
        CHECK(semi.isRegular(i));
    }
    CHECK(semi.frequency() == Frequency::Semiannual);
    CHECK(semi.tenorMonths() == 6);

    // Forward yearly with a short back stub.
    const Schedule stub(Date::parse("2026-02-15"), Date::parse("2030-08-15"),
                        Period(1, TimeUnit::Years), target,
                        BusinessDayConvention::ModifiedFollowing, DateGeneration::Forward, false);
    CHECK(stub.size() == 6);
    CHECK(stub.startDate() == Date(2026, 2, 16));  // Feb 15 2026 is a Sunday
    CHECK(stub.endDate() == Date(2030, 8, 15));
    CHECK(stub.isRegular(0));
    CHECK(stub.isRegular(3));
    CHECK(!stub.isRegular(4));  // 2030-02-15 -> 2030-08-15 stub
    CHECK(!stub.isRegular(5));

    // Backward yearly with a front stub.
    const Schedule backStub(Date::parse("2026-02-15"), Date::parse("2030-08-15"),
                            Period(1, TimeUnit::Years), target,
                            BusinessDayConvention::ModifiedFollowing,
                            DateGeneration::Backward, false);
    CHECK(backStub.size() == 6);
    CHECK(backStub.startDate() == Date(2026, 2, 16));
    CHECK(!backStub.isRegular(0));  // front stub
    CHECK(backStub.isRegular(1));
    CHECK(backStub.endDate() == Date(2030, 8, 15));

    // End-of-month monthly schedule.
    const Schedule monthly(Date::parse("2026-01-31"), Date::parse("2027-01-31"),
                           Period(1, TimeUnit::Months), target,
                           BusinessDayConvention::ModifiedFollowing,
                           DateGeneration::Forward, true);
    CHECK(monthly.size() == 13);
    CHECK(monthly.unadjustedDates().back() == Date(2027, 1, 31));
    CHECK(monthly.endDate() == Date(2027, 1, 29));  // Jan 31 is a Sunday, MF rolls back
    for (std::size_t i = 0; i + 1 < monthly.size(); ++i) {
        CHECK(monthly.isRegular(i));
    }

    // Zero rule: two dates.
    const Schedule zero(Date::parse("2026-01-15"), Date::parse("2027-01-15"),
                        Period(1, TimeUnit::Years), target,
                        BusinessDayConvention::Following, DateGeneration::Zero, false);
    CHECK(zero.size() == 2);

    // Time conversion: first date maps to zero, monotone, grid steps match.
    const DayCounter act365(DayCount::Actual365Fixed);
    const std::vector<double> times = scheduleTimes(semi.dates(), semi.startDate(), act365);
    CHECK(times.size() == semi.size());
    CHECK(times.front() == 0.0);
    for (std::size_t i = 1; i < times.size(); ++i) {
        CHECK(times[i] > times[i - 1]);
    }
    CHECK(close(times.back(),
                static_cast<double>(semi.endDate() - semi.startDate()) / 365.0, 1e-12));
    CHECK(timeGridFromDates(semi.dates(), semi.startDate(), act365).nSteps() == semi.size() - 1);

    // Spot date: T+2 with adjustment.
    CHECK(spotDate(Date(2026, 4, 1), target, 2) == Date(2026, 4, 7));  // Easter 2026

    // ACT/ACT ICMA on the regular schedule: each semi period contributes 0.5.
    {
        DayCountContext context;
        context.schedule = &semi;
        DayCounter icma(DayCount::ActualActualICMA, context);
        CHECK(close(icma.yearFractionUncached(semi.dates()[0], semi.dates()[2]), 1.0));
        CHECK(close(icma.yearFractionUncached(semi.dates()[0], semi.dates()[1]), 0.5));
        CHECK(close(icma.yearFractionUncached(semi.dates()[0], semi.endDate()), 5.0));
    }

    // ACT/ACT ICMA with a short front stub on a real coupon schedule.
    {
        const Schedule stubSchedule(Date::parse("2026-02-15"), Date::parse("2030-08-15"),
                                    Period(1, TimeUnit::Years), target,
                                    BusinessDayConvention::ModifiedFollowing,
                                    DateGeneration::Forward, false);
        DayCountContext context;
        context.schedule = &stubSchedule;
        DayCounter icma(DayCount::ActualActualICMA, context);
        const double full = icma.yearFractionUncached(stubSchedule.dates()[0],
                                                      stubSchedule.endDate());
        CHECK(full > 4.4 && full < 4.6);  // ~4.5y of coupon time
    }

    // BUS/252 with a calendar in the context.
    {
        DayCountContext context;
        context.calendar = &weekends;
        DayCounter bus252(DayCount::Bus252, context);
        CHECK(close(bus252.yearFractionUncached(Date(2020, 1, 6), Date(2020, 1, 13)),
                    5.0 / 252.0));
        CHECK(bus252.yearFractionUncached(Date(2020, 1, 6), Date(2020, 1, 6)) == 0.0);
    }

    QTA_LOG_INFO("test", "test_schedule: ok");
    return 0;
}
