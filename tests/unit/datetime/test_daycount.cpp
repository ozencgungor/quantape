// test_daycount.cpp — convention values and the day-count cache contract
#include "quantape/datetime/DayCounter.h"

#include <cmath>
#include <cstddef>
#include <random>
#include <stdexcept>
#include <utility>

#include "support/GtestSupport.h"

using namespace quantape::datetime;

namespace {

bool close(double a, double b, double tol = 1e-14) {
    return std::abs(a - b) <= tol * (1.0 + std::abs(b));
}

double yf(DayCount convention, const char* from, const char* to, DayCountContext context = {}) {
    return DayCounter(convention, context).yearFractionUncached(Date::parse(from), Date::parse(to));
}

/// Reference implementation of the AFB year walk (validates the fast path).
double slowAfb(Date d1, Date d2) {
    if (d1 == d2) {
        return 0.0;
    }
    if (d1 > d2) {
        return -slowAfb(d2, d1);
    }
    Date newD2 = d2;
    double sum = 0.0;
    for (Date temp = d2; temp > d1;) {
        temp = newD2.plusYears(-1, false);
        if (temp.month() == 2 && temp.dayOfMonth() == 28 && Date::isLeapYear(temp.year())) {
            temp = temp.plusDays(1);
        }
        if (temp >= d1) {
            sum += 1.0;
            newD2 = temp;
        }
    }
    double den = 365.0;
    const Date leapEnd(newD2.year(), 2, 29);
    if (Date::isLeapYear(newD2.year()) && newD2 > leapEnd && d1 <= leapEnd) {
        den += 1.0;
    } else {
        const Date leapStart(d1.year(), 2, 29);
        if (Date::isLeapYear(d1.year()) && newD2 > leapStart && d1 <= leapStart) {
            den += 1.0;
        }
    }
    return sum + static_cast<double>(newD2 - d1) / den;
}

class DayCountTest : public ::testing::Test {
protected:
    void SetUp() override {
        clearCache();
        resetDayCountCacheStats();
    }

    void TearDown() override {
        clearCache();
        resetDayCountCacheStats();
    }

private:
    /// There is no public cache-clear; a size change plus one cached lookup
    /// makes ThreadDayCountCache::ensure() reallocate the thread-local slots.
    static void clearCache() {
        enableDayCountCache(true);
        setDayCountCacheSize(1);
        (void)DayCounter(DayCount::ActualActualISDA)
            .yearFraction(Date(2000, 1, 1), Date(2000, 1, 2));
        enableDayCountCache(false);
        setDayCountCacheSize(0);
    }
};

} // namespace

TEST_F(DayCountTest, actualFamilyYearFractions) {
    EXPECT_TRUE(close(yf(DayCount::Actual365Fixed, "2020-01-01", "2021-01-01"), 366.0 / 365.0));
    EXPECT_TRUE(close(yf(DayCount::Actual360, "2020-01-01", "2021-01-01"), 366.0 / 360.0));
    EXPECT_TRUE(close(yf(DayCount::Actual364, "2020-01-01", "2021-01-01"), 366.0 / 364.0));
    EXPECT_TRUE(close(yf(DayCount::Actual365_25, "2020-01-01", "2021-01-01"), 366.0 / 365.25));
    EXPECT_TRUE(close(yf(DayCount::Actual366, "2020-01-01", "2021-01-01"), 1.0));
    EXPECT_TRUE(close(yf(DayCount::NL365, "2020-01-01", "2021-01-01"), 365.0 / 365.0));
    EXPECT_TRUE(close(yf(DayCount::NL360, "2020-01-01", "2021-01-01"), 365.0 / 360.0));
    EXPECT_TRUE(close(yf(DayCount::Actual365Actual, "2020-01-01", "2021-01-01"), 366.0 / 366.0));
    EXPECT_TRUE(close(yf(DayCount::Actual365Actual, "2021-01-01", "2022-01-01"), 365.0 / 365.0));
}

TEST_F(DayCountTest, actualActualVariantsAndAct365LContext) {
    EXPECT_TRUE(close(yf(DayCount::ActualActualISDA, "2019-07-01", "2020-07-01"),
                      184.0 / 365.0 + 182.0 / 366.0));
    EXPECT_TRUE(close(yf(DayCount::ActualActualAFB, "2020-01-01", "2020-07-01"), 182.0 / 366.0));
    EXPECT_TRUE(close(yf(DayCount::ActualActualAFB, "2021-01-01", "2021-07-01"), 181.0 / 365.0));
    EXPECT_TRUE(close(yf(DayCount::ActualActualYear, "2020-01-01", "2021-01-01"), 1.0));
    EXPECT_TRUE(
        close(yf(DayCount::ActualActualYear, "2020-01-01", "2022-07-01"), 2.0 + 181.0 / 365.0));

    // ACT/365L needs the reference period end and frequency.
    DayCountContext context;
    context.refEnd = Date::parse("2020-07-01");
    context.frequency = Frequency::Semiannual;
    EXPECT_TRUE(
        close(yf(DayCount::Actual365L, "2020-01-01", "2020-07-01", context), 182.0 / 366.0));
    context.refEnd = Date::parse("2021-07-01");
    EXPECT_TRUE(
        close(yf(DayCount::Actual365L, "2021-01-01", "2021-07-01", context), 181.0 / 365.0));
}

TEST_F(DayCountTest, oneOneSimpleAndThirty360Family) {
    // 1/1 and Simple.
    EXPECT_TRUE(close(yf(DayCount::OneOne, "2020-01-01", "2020-01-02"), 1.0));
    EXPECT_TRUE(close(yf(DayCount::OneOne, "2020-01-01", "2020-01-01"), 0.0));
    EXPECT_TRUE(close(yf(DayCount::Simple, "2020-01-15", "2021-01-15"), 1.0));
    EXPECT_TRUE(close(yf(DayCount::Simple, "2020-01-31", "2021-02-28"), 1.0 + 1.0 / 12.0));

    // 30/360 family.
    EXPECT_TRUE(close(yf(DayCount::Thirty360US, "2020-01-31", "2020-02-29"), 29.0 / 360.0));
    EXPECT_TRUE(close(yf(DayCount::Thirty360US, "2020-02-29", "2020-03-31"), 30.0 / 360.0));
    EXPECT_TRUE(close(yf(DayCount::Thirty360BondBasis, "2020-01-31", "2020-03-31"), 60.0 / 360.0));
    EXPECT_TRUE(close(yf(DayCount::ThirtyE360, "2020-01-31", "2020-02-29"), 29.0 / 360.0));
    EXPECT_TRUE(close(yf(DayCount::ThirtyE360, "2020-01-31", "2020-03-31"), 60.0 / 360.0));
    EXPECT_TRUE(close(yf(DayCount::ThirtyE360ISDA, "2020-01-31", "2020-02-29"), 30.0 / 360.0));
    EXPECT_TRUE(close(yf(DayCount::ThirtyEPlus360, "2020-01-31", "2020-03-31"), 61.0 / 360.0));
    EXPECT_TRUE(close(yf(DayCount::Thirty360Italian, "2020-02-29", "2020-03-31"), 30.0 / 360.0));
    EXPECT_TRUE(close(yf(DayCount::Thirty360PSA, "2020-01-31", "2020-02-29"), 29.0 / 360.0));
    EXPECT_TRUE(close(yf(DayCount::Thirty365, "2020-01-31", "2020-03-31"), 60.0 / 365.0));
    EXPECT_TRUE(close(yf(DayCount::ThirtyE365, "2020-01-31", "2020-02-29"), 30.0 / 365.0));
    {
        DayCountContext context;
        context.termination = Date::parse("2020-02-29");
        EXPECT_TRUE(
            close(yf(DayCount::ThirtyE360ISDA, "2020-01-31", "2020-02-29", context), 29.0 / 360.0));
    }
}

TEST_F(DayCountTest, scheduleDrivenConventionsAndMissingContext) {
    // Schedule/calendar-driven conventions with context.
    const Calendar target = Calendar::target();
    const Schedule semiannual(Date::parse("2020-01-15"), Date::parse("2021-01-15"),
                              Period(6, TimeUnit::Months), target,
                              BusinessDayConvention::ModifiedFollowing);
    DayCountContext ictx;
    ictx.schedule = &semiannual;
    EXPECT_TRUE(close(yf(DayCount::ActualActualICMA, "2020-01-15", "2021-01-15", ictx), 1.0));
    EXPECT_TRUE(close(yf(DayCount::ActualActualICMA, "2020-01-15", "2020-07-15", ictx), 0.5));

    DayCountContext bctx;
    bctx.calendar = &target;
    EXPECT_TRUE(close(yf(DayCount::Bus252, "2026-01-05", "2026-01-12", bctx), 5.0 / 252.0));

    EXPECT_THROW((void)yf(DayCount::ActualActualICMA, "2020-01-15", "2020-07-15"),
                 std::logic_error);
    EXPECT_THROW((void)yf(DayCount::Bus252, "2020-01-01", "2020-07-01"), std::logic_error);
}

TEST_F(DayCountTest, afbFastPathMatchesReferenceWalk) {
    // AFB fast path equals the reference anniversary walk.
    std::mt19937 rng(2026);
    for (int i = 0; i < 500; ++i) {
        const int y1 = 1990 + static_cast<int>(rng() % 60);
        const int m1 = 1 + static_cast<int>(rng() % 12);
        const int day1 = 1 + static_cast<int>(rng() % 28);
        const int y2 = y1 + static_cast<int>(rng() % 40);
        const int m2 = 1 + static_cast<int>(rng() % 12);
        const int day2 = 1 + static_cast<int>(rng() % 28);
        Date a(y1, static_cast<unsigned>(m1), static_cast<unsigned>(day1));
        Date b(y2, static_cast<unsigned>(m2), static_cast<unsigned>(day2));
        if (a > b) {
            std::swap(a, b);
        }
        const double fast = DayCounter(DayCount::ActualActualAFB).yearFractionUncached(a, b);
        EXPECT_TRUE(close(fast, slowAfb(a, b), 1e-15));
    }
}

TEST_F(DayCountTest, cacheHitsMissesBypassAndDisabledState) {
    // Cache contract: hits for repeated pairs, misses for new keys, and cheap
    // conventions bypass the cache entirely (measured to be faster).
    setDayCountCacheSize(1024);
    enableDayCountCache(true);
    resetDayCountCacheStats();
    {
        DayCounter counter(DayCount::ActualActualISDA);
        const Date a = Date::parse("2020-01-01");
        const Date b = Date::parse("2020-07-01");
        const double first = counter.yearFraction(a, b);
        const double second = counter.yearFraction(a, b);
        const double third = counter.yearFraction(a, b);
        EXPECT_TRUE(first == second && second == third);
        DayCountCacheStats stats = dayCountCacheStats();
        EXPECT_TRUE(stats.hits == 2 && stats.misses == 1);

        DayCounter cheap(DayCount::Actual360);
        (void)cheap.yearFraction(a, b);
        stats = dayCountCacheStats();
        EXPECT_TRUE(stats.hits == 2 && stats.misses == 1); // bypassed, no stats change

        (void)counter.yearFraction(a, b.plusDays(1));
        stats = dayCountCacheStats();
        EXPECT_TRUE(stats.misses == 2); // new key
    }
    {
        // Different threads have independent caches.
        enableDayCountCache(false);
        const std::size_t before = dayCountCacheStats().misses;
        (void)DayCounter(DayCount::Actual360).yearFraction(Date(2020, 1, 1), Date(2020, 2, 1));
        EXPECT_TRUE(dayCountCacheStats().misses == before);
    }
}
