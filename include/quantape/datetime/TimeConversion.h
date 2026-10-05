#pragma once

// Conversions between dates, schedules and engine time. Time grids are pure
// doubles (non-differentiable); conversion happens once at schedule
// construction.

#include "quantape/datetime/Calendar.h"
#include "quantape/datetime/Date.h"
#include "quantape/datetime/DayCounter.h"
#include "quantape/mc/TimeGrid.h"

#include <algorithm>
#include <vector>

namespace quantape::datetime {

inline double yearFraction(const Date& from, const Date& to, const DayCounter& dayCounter) {
    return dayCounter.yearFraction(from, to);
}

/// Times of each date relative to `reference` (reference maps to 0).
inline std::vector<double> scheduleTimes(const std::vector<Date>& dates, const Date& reference,
                                         const DayCounter& dayCounter) {
    std::vector<double> times;
    times.reserve(dates.size());
    for (const Date& date : dates) {
        times.push_back(dayCounter.yearFraction(reference, date));
    }
    return times;
}

inline mc::TimeGrid timeGridFromDates(const std::vector<Date>& dates, const Date& reference,
                                      const DayCounter& dayCounter) {
    return mc::TimeGrid(scheduleTimes(dates, reference, dayCounter));
}

/// Spot date: trade date advanced by a settlement lag on the given calendar.
inline Date spotDate(const Date& trade, const Calendar& calendar, int lag,
                     BusinessDayConvention convention = BusinessDayConvention::Following) {
    return calendar.advance(trade, lag, TimeUnit::Days, convention, false);
}

/// Spot date on an already joined calendar: the trade date is advanced by
/// `max(baseLag, quoteLag)` days and adjusted on `jointCalendar`. Defaults to
/// ModifiedFollowing, the standard FX spot convention.
inline Date spotDate(const Date& trade, const Calendar& jointCalendar, int baseLag, int quoteLag,
                     BusinessDayConvention convention = BusinessDayConvention::ModifiedFollowing) {
    return jointCalendar.advance(trade, std::max(baseLag, quoteLag), TimeUnit::Days, convention,
                                 false);
}

/// Spot date from two settlement calendars and lags: joins the calendars and
/// advances by `max(baseLag, quoteLag)` days. Defaults to ModifiedFollowing.
inline Date spotDate(const Date& trade, const Calendar& baseCalendar, const Calendar& quoteCalendar,
                     int baseLag, int quoteLag,
                     BusinessDayConvention convention = BusinessDayConvention::ModifiedFollowing) {
    return spotDate(trade, Calendar::joint(baseCalendar, quoteCalendar), baseLag, quoteLag,
                    convention);
}

} // namespace quantape::datetime
