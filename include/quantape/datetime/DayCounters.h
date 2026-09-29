#pragma once

// Day-count convention implementations. Included by DayCounter.h at the end
// (the enum/context/cache declarations are already visible).
//
// Each date is decoded once per call (Date::decompose): earlier versions used
// separate year()/month()/dayOfMonth() accessors, each re-running the civil
// conversion.

#include "quantape/datetime/Calendar.h"
#include "quantape/datetime/Schedule.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace quantape::datetime {

inline void enableDayCountCache(bool enabled) noexcept {
    detail::cacheEnabledFlag().store(enabled, std::memory_order_relaxed);
}
inline bool dayCountCacheEnabled() noexcept {
    return detail::cacheEnabledFlag().load(std::memory_order_relaxed);
}
inline void setDayCountCacheSize(std::size_t slots) {
    std::uint32_t bits = 0;
    if (slots > 0) {
        bits = 1;
        while ((std::size_t{1} << bits) < slots) {
            ++bits;
        }
    }
    detail::cacheSlotBits().store(bits, std::memory_order_relaxed);
}
inline std::size_t dayCountCacheSize() noexcept {
    return std::size_t{1} << detail::cacheSlotBits().load(std::memory_order_relaxed);
}
inline std::size_t dayCountCacheBytes() noexcept {
    return dayCountCacheSize() * sizeof(detail::DayCountCacheSlot);
}
inline DayCountCacheStats dayCountCacheStats() noexcept {
    DayCountCacheStats stats;
    stats.hits = detail::cacheHits().load(std::memory_order_relaxed);
    stats.misses = detail::cacheMisses().load(std::memory_order_relaxed);
    stats.slots = dayCountCacheSize();
    stats.bytes = dayCountCacheBytes();
    return stats;
}
inline void resetDayCountCacheStats() noexcept {
    detail::cacheHits().store(0, std::memory_order_relaxed);
    detail::cacheMisses().store(0, std::memory_order_relaxed);
}

namespace detail {

struct Ymd {
    int year;
    unsigned month;
    unsigned day;
};

inline Ymd ymd(const Date& date) {
    Ymd out{0, 0, 0};
    date.decompose(out.year, out.month, out.day);
    return out;
}

inline bool isLastOfFebruary(const Ymd& date) {
    return date.month == 2 && date.day == 28 + (Date::isLeapYear(date.year) ? 1 : 0);
}

inline Date ensureLeapDay(int year) {
    while (!Date::isLeapYear(year)) {
        year += 4;
    }
    return Date(year, 2, 29);
}

/// Strata semantics: the next leap day strictly after `date`.
inline Date nextLeapDay(const Date& date) {
    const Ymd value = ymd(date);
    if (value.month == 2 && value.day == 29) {
        return ensureLeapDay(value.year + 4);
    }
    if (Date::isLeapYear(value.year) && value.month <= 2) {
        return Date(value.year, 2, 29);
    }
    return ensureLeapDay((value.year / 4) * 4 + 4);
}

/// Number of leap years in [1, year] (floor-safe for negative years).
inline std::int64_t leapYearsUpTo(int year) {
    return floorDiv(year, 4) - floorDiv(year, 100) + floorDiv(year, 400);
}

/// Number of leap days <= date (arithmetic, no year walk).
inline int leapDaysUpTo(const Date& date) {
    const Ymd value = ymd(date);
    std::int64_t count = leapYearsUpTo(value.year - 1);
    if (Date::isLeapYear(value.year) &&
        (value.month > 2 || (value.month == 2 && value.day == 29))) {
        ++count;
    }
    return static_cast<int>(count);
}

/// Number of leap days in (d1, d2] (Strata semantics).
inline int numberOfLeapDays(const Date& d1, const Date& d2) {
    return leapDaysUpTo(d2) - leapDaysUpTo(d1);
}

inline double thirty360Days(const Ymd& a, int day1, const Ymd& b, int day2) {
    return static_cast<double>(360 * (b.year - a.year) + 30 * (static_cast<int>(b.month) -
                                                               static_cast<int>(a.month)) +
                               (day2 - day1));
}

inline double actualActualISDA(const Date& d1, const Date& d2) {
    if (d1 == d2) {
        return 0.0;
    }
    if (d1 > d2) {
        return -actualActualISDA(d2, d1);
    }
    const Ymd a = ymd(d1);
    const Ymd b = ymd(d2);
    const bool leap1 = Date::isLeapYear(a.year);
    const double dib1 = leap1 ? 366.0 : 365.0;
    const double dib2 = Date::isLeapYear(b.year) ? 366.0 : 365.0;
    const double daysToYearEnd =
        dib1 - static_cast<double>(dayOfYearImpl(a.year, a.month, a.day)) + 1.0;
    const double daysFromYearStart =
        static_cast<double>(dayOfYearImpl(b.year, b.month, b.day)) - 1.0;
    return static_cast<double>(b.year - a.year - 1) + daysToYearEnd / dib1 +
           daysFromYearStart / dib2;
}

/// AFB anniversary rule: subtract years, bump a clipped Feb 28 in a leap year
/// back to Feb 29.
inline Date afbAnniversary(const Date& date, int yearsBack) {
    Date temp = date.plusYears(-yearsBack, false);
    const Ymd value = ymd(temp);
    if (value.month == 2 && value.day == 28 && Date::isLeapYear(value.year)) {
        temp = temp.plusDays(1);
    }
    return temp;
}

inline double actualActualAFB(const Date& d1, const Date& d2) {
    if (d1 == d2) {
        return 0.0;
    }
    if (d1 > d2) {
        return -actualActualAFB(d2, d1);
    }
    // Anniversary arithmetic: k whole-year steps back from d2 while the
    // anniversary is still after d1. days/365 is an upper bound, so the
    // correction loop runs at most a couple of times.
    const std::int64_t totalDays = d2 - d1;
    int k = static_cast<int>(totalDays / 365);
    while (k > 0 && afbAnniversary(d2, k) < d1) {
        --k;
    }
    const Date newD2 = k > 0 ? afbAnniversary(d2, k) : d2;
    double den = 365.0;
    const Ymd end = ymd(newD2);
    const Ymd start = ymd(d1);
    if (Date::isLeapYear(end.year)) {
        const Date leap(end.year, 2, 29);
        if (newD2 > leap && d1 <= leap) {
            den += 1.0;
        }
    } else if (Date::isLeapYear(start.year)) {
        const Date leap(start.year, 2, 29);
        if (newD2 > leap && d1 <= leap) {
            den += 1.0;
        }
    }
    return static_cast<double>(k) + static_cast<double>(newD2 - d1) / den;
}

/// Coupon dates with the irregular first/last periods replaced by their
/// notional (quasi) coupon dates, per the ICMA reference-period definition.
inline std::vector<Date> quasiCouponDates(const Schedule& schedule) {
    const Calendar& calendar = schedule.calendar();
    const BusinessDayConvention convention = schedule.businessDayConvention();
    const bool endOfMonth = schedule.endOfMonth();
    const Period& tenor = schedule.tenor();
    const std::vector<Date>& original = schedule.dates();
    const std::size_t n = original.size();
    std::vector<Date> dates = original;
    if (n < 2) {
        return dates;
    }
    if (!schedule.isRegular(0)) {
        const Date issue = dates[0];
        const Date notionalFirst = calendar.advance(original[1], -tenor.length(), tenor.unit(),
                                                    convention, endOfMonth);
        dates[0] = notionalFirst;
        if (notionalFirst > issue) {
            dates.insert(dates.begin(),
                         calendar.advance(notionalFirst, -tenor.length(), tenor.unit(),
                                          convention, endOfMonth));
        }
    }
    if (!schedule.isRegular(n - 2)) {
        const Date notionalLast = calendar.advance(original[n - 2], tenor.length(), tenor.unit(),
                                                   convention, endOfMonth);
        dates[n - 1] = notionalLast;
        if (notionalLast < schedule.endDate()) {
            dates.push_back(calendar.advance(notionalLast, tenor.length(), tenor.unit(),
                                             convention, endOfMonth));
        }
    }
    return dates;
}

inline double actualActualICMA(const Date& d1, const Date& d2, const DayCountContext& context) {
    if (d1 == d2) {
        return 0.0;
    }
    if (d1 > d2) {
        return -actualActualICMA(d2, d1, context);
    }
    if (context.schedule == nullptr) {
        throw std::logic_error("ACT/ACT ICMA requires a schedule in the context");
    }
    const std::vector<Date> dates = quasiCouponDates(*context.schedule);
    if (dates.size() < 2) {
        throw std::logic_error("ACT/ACT ICMA: schedule has fewer than two dates");
    }
    const Date first = *std::min_element(dates.begin(), dates.end());
    const Date last = *std::max_element(dates.begin(), dates.end());
    if (d1 < first || d2 > last) {
        throw std::invalid_argument("ACT/ACT ICMA: dates outside the schedule");
    }
    double sum = 0.0;
    for (std::size_t i = 0; i + 1 < dates.size(); ++i) {
        const Date refStart = dates[i];
        const Date refEnd = dates[i + 1];
        if (d1 < refEnd && d2 > refStart) {
            const Date a = std::max(d1, refStart);
            const Date b = std::min(d2, refEnd);
            double refDays = static_cast<double>(refEnd - refStart);
            int couponsPerYear = 1;
            if (refDays < 16.0) {
                refDays = static_cast<double>(a.plusYears(1, false) - a);
            } else {
                const long months = std::lround(12.0 * refDays / 365.0);
                if (months == 0) {
                    throw std::logic_error("ACT/ACT ICMA: reference period too short");
                }
                couponsPerYear = static_cast<int>(std::lround(12.0 / static_cast<double>(months)));
            }
            sum += static_cast<double>(b - a) / (refDays * couponsPerYear);
        }
    }
    return sum;
}

/// Strata "Act/Act Year": whole 1-year blocks from d1 plus the remainder over
/// the length of that block's year.
inline double actualActualYear(const Date& d1, const Date& d2) {
    if (d1 == d2) {
        return 0.0;
    }
    if (d1 > d2) {
        return -actualActualYear(d2, d1);
    }
    Date start = d1;
    int yearsAdded = 0;
    while (d2 > start.plusYears(1, false)) {
        start = d1.plusYears(++yearsAdded, false);
    }
    if (start == d2) {
        return static_cast<double>(yearsAdded + 1);
    }
    const double actualDays = static_cast<double>(d2 - start);
    const double actualDaysInYear = static_cast<double>(start.plusYears(1, false) - start);
    return yearsAdded + actualDays / actualDaysInYear;
}

inline double yearFractionFor(DayCount convention, const Date& d1, const Date& d2,
                              const DayCountContext& context) {
    const double days = static_cast<double>(d2 - d1);
    switch (convention) {
        case DayCount::Actual360:
            return days / 360.0;
        case DayCount::Actual364:
            return days / 364.0;
        case DayCount::Actual365Fixed:
            return days / 365.0;
        case DayCount::Actual365_25:
            return days / 365.25;
        case DayCount::Actual366:
            return days / 366.0;
        case DayCount::NL365:
            return (days - numberOfLeapDays(d1, d2)) / 365.0;
        case DayCount::NL360:
            return (days - numberOfLeapDays(d1, d2)) / 360.0;
        case DayCount::ActualActualISDA:
            return actualActualISDA(d1, d2);
        case DayCount::ActualActualAFB:
            return actualActualAFB(d1, d2);
        case DayCount::ActualActualYear:
            return actualActualYear(d1, d2);
        case DayCount::Actual365Actual: {
            const Date leap = nextLeapDay(d1);
            return days / (leap > d2 ? 365.0 : 366.0);
        }
        case DayCount::Actual365L: {
            if (d1 == d2) {
                return 0.0;
            }
            if (!context.refEnd.has_value()) {
                throw std::logic_error("ACT/365L requires the reference period end in the context");
            }
            const Date nextCoupon = *context.refEnd;
            if (context.frequency == Frequency::Annual) {
                const Date leap = nextLeapDay(d1);
                return days / (leap > nextCoupon ? 365.0 : 366.0);
            }
            return days / (Date::isLeapYear(nextCoupon.year()) ? 366.0 : 365.0);
        }
        case DayCount::OneOne:
            return d1 == d2 ? 0.0 : 1.0;
        case DayCount::Simple: {
            const Ymd a = ymd(d1);
            const Ymd b = ymd(d2);
            const int dm1 = static_cast<int>(a.day);
            const int dm2 = static_cast<int>(b.day);
            if (dm1 == dm2 || (dm1 > dm2 && d2.isEndOfMonth()) ||
                (dm1 < dm2 && d1.isEndOfMonth())) {
                return static_cast<double>(b.year - a.year) +
                       (static_cast<double>(b.month) - static_cast<double>(a.month)) / 12.0;
            }
            int day1 = dm1;
            int day2 = dm2;
            if (day1 == 31) {
                day1 = 30;
            }
            if (day2 == 31 && day1 == 30) {
                day2 = 30;
            }
            return thirty360Days(a, day1, b, day2) / 360.0;
        }
        case DayCount::Thirty360US: {
            const Ymd a = ymd(d1);
            const Ymd b = ymd(d2);
            int day1 = static_cast<int>(a.day);
            int day2 = static_cast<int>(b.day);
            if (isLastOfFebruary(a)) {
                if (isLastOfFebruary(b)) {
                    day2 = 30;
                }
                day1 = 30;
            }
            if (day2 == 31 && day1 >= 30) {
                day2 = 30;
            }
            if (day1 == 31) {
                day1 = 30;
            }
            return thirty360Days(a, day1, b, day2) / 360.0;
        }
        case DayCount::ThirtyU360EOM: {
            const Ymd a = ymd(d1);
            const Ymd b = ymd(d2);
            int day1 = static_cast<int>(a.day);
            int day2 = static_cast<int>(b.day);
            if (isLastOfFebruary(a)) {
                if (isLastOfFebruary(b)) {
                    day2 = 30;
                }
                day1 = 30;
            }
            if (day2 == 31 && day1 >= 30) {
                day2 = 30;
            }
            if (day1 == 31) {
                day1 = 30;
            }
            return thirty360Days(a, day1, b, day2) / 360.0;
        }
        case DayCount::Thirty360BondBasis: {
            const Ymd a = ymd(d1);
            const Ymd b = ymd(d2);
            int day1 = static_cast<int>(a.day);
            int day2 = static_cast<int>(b.day);
            if (day1 == 31) {
                day1 = 30;
            }
            if (day2 == 31 && day1 == 30) {
                day2 = 30;
            }
            return thirty360Days(a, day1, b, day2) / 360.0;
        }
        case DayCount::ThirtyE360: {
            const Ymd a = ymd(d1);
            const Ymd b = ymd(d2);
            int day1 = static_cast<int>(a.day);
            int day2 = static_cast<int>(b.day);
            if (day1 == 31) {
                day1 = 30;
            }
            if (day2 == 31) {
                day2 = 30;
            }
            return thirty360Days(a, day1, b, day2) / 360.0;
        }
        case DayCount::ThirtyE360ISDA: {
            const Ymd a = ymd(d1);
            const Ymd b = ymd(d2);
            int day1 = static_cast<int>(a.day);
            int day2 = static_cast<int>(b.day);
            if (day1 == 31) {
                day1 = 30;
            }
            if (day2 == 31) {
                day2 = 30;
            }
            if (isLastOfFebruary(a)) {
                day1 = 30;
            }
            const bool d2IsTermination =
                context.termination.has_value() && *context.termination == d2;
            if (!d2IsTermination && isLastOfFebruary(b)) {
                day2 = 30;
            }
            return thirty360Days(a, day1, b, day2) / 360.0;
        }
        case DayCount::ThirtyEPlus360: {
            const Ymd a = ymd(d1);
            const Ymd b = ymd(d2);
            int day1 = static_cast<int>(a.day);
            int day2 = static_cast<int>(b.day);
            int month2 = static_cast<int>(b.month);
            if (day1 == 31) {
                day1 = 30;
            }
            if (day2 == 31) {
                day2 = 1;
                month2 += 1;  // no need to carry December into January (Strata)
            }
            return static_cast<double>(360 * (b.year - a.year) +
                                       30 * (month2 - static_cast<int>(a.month)) +
                                       (day2 - day1)) /
                   360.0;
        }
        case DayCount::Thirty360Italian: {
            const Ymd a = ymd(d1);
            const Ymd b = ymd(d2);
            int day1 = static_cast<int>(a.day);
            int day2 = static_cast<int>(b.day);
            if (day1 == 31) {
                day1 = 30;
            }
            if (day2 == 31) {
                day2 = 30;
            }
            if (a.month == 2 && day1 > 27) {
                day1 = 30;
            }
            if (b.month == 2 && day2 > 27) {
                day2 = 30;
            }
            return thirty360Days(a, day1, b, day2) / 360.0;
        }
        case DayCount::Thirty360PSA: {
            const Ymd a = ymd(d1);
            const Ymd b = ymd(d2);
            int day1 = static_cast<int>(a.day);
            int day2 = static_cast<int>(b.day);
            if (day1 == static_cast<int>(Date::daysInMonth(a.year, a.month))) {
                day1 = 30;
            }
            if (day2 == 31 && day1 == 30) {
                day2 = 30;
            }
            return thirty360Days(a, day1, b, day2) / 360.0;
        }
        case DayCount::Thirty365: {
            const Ymd a = ymd(d1);
            const Ymd b = ymd(d2);
            int day1 = static_cast<int>(a.day);
            int day2 = static_cast<int>(b.day);
            if (day1 == 31) {
                day1 = 30;
            }
            if (day2 == 31) {
                day2 = 30;
            }
            return thirty360Days(a, day1, b, day2) / 365.0;
        }
        case DayCount::ThirtyE365: {
            const Ymd a = ymd(d1);
            const Ymd b = ymd(d2);
            int day1 = static_cast<int>(a.day);
            int day2 = static_cast<int>(b.day);
            if (day1 == static_cast<int>(Date::daysInMonth(a.year, a.month))) {
                day1 = 30;
            }
            if (day2 == static_cast<int>(Date::daysInMonth(b.year, b.month))) {
                day2 = 30;
            }
            return thirty360Days(a, day1, b, day2) / 365.0;
        }
        case DayCount::ActualActualICMA:
            return actualActualICMA(d1, d2, context);
        case DayCount::Bus252: {
            if (context.calendar == nullptr) {
                throw std::logic_error("BUS/252 requires a calendar in the context");
            }
            return static_cast<double>(context.calendar->businessDaysBetween(d1, d2)) / 252.0;
        }
    }
    return days / 365.0;
}

}  // namespace detail

}  // namespace quantape::datetime
