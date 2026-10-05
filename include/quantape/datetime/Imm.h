#pragma once

// IMM dates: third Wednesday of March, June, September and December, the
// settlement cycle for exchange-traded futures and IMM swaps.

#include "quantape/datetime/Calendar.h"
#include "quantape/datetime/Date.h"

#include <stdexcept>
#include <vector>

namespace quantape::datetime {

/// True when `month` is a quarterly IMM cycle month (Mar/Jun/Sep/Dec).
inline bool isIMMCycleMonth(unsigned month) {
    return month == 3 || month == 6 || month == 9 || month == 12;
}

/// IMM date of the given quarterly cycle month: the third Wednesday.
inline Date immDate(int year, unsigned cycleMonth) {
    if (!isIMMCycleMonth(cycleMonth)) {
        throw std::invalid_argument("immDate: not an IMM cycle month");
    }
    return detail::nthWeekdayOfMonth(year, cycleMonth, Weekday::Wednesday, 3);
}

inline bool isIMMDate(const Date& date) {
    return isIMMCycleMonth(date.month()) && date == immDate(date.year(), date.month());
}

/// First IMM date strictly after `date`.
inline Date nextIMMDate(const Date& date) {
    for (const unsigned month : {3u, 6u, 9u, 12u}) {
        if (const Date candidate = immDate(date.year(), month); candidate > date) {
            return candidate;
        }
    }
    return immDate(date.year() + 1, 3);
}

/// Coupon dates anchored on the IMM grid between effective and termination;
/// non-IMM effective/termination dates produce front/back stubs.
inline std::vector<Date> immSchedule(
    const Date& effective, const Date& termination, const Calendar& calendar,
    BusinessDayConvention convention = BusinessDayConvention::ModifiedFollowing) {
    if (termination < effective) {
        throw std::invalid_argument("immSchedule: termination before effective");
    }
    std::vector<Date> dates{calendar.adjust(effective, convention)};
    Date next = nextIMMDate(effective);
    while (next < termination) {
        const Date adjusted = calendar.adjust(next, convention);
        if (adjusted > dates.back()) {
            dates.push_back(adjusted);
        }
        next = nextIMMDate(next);
    }
    const Date end = calendar.adjust(termination, convention);
    if (end > dates.back()) {
        dates.push_back(end);
    }
    return dates;
}

}  // namespace quantape::datetime
