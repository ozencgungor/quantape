#pragma once

// Calendar: value type holding an immutable rule set (weekend mask + holiday
// rules + ad-hoc closures) and per-instance extra holidays. Holidays are
// generated per year and cached in the rule set itself, so schedule generation
// hits a small sorted vector and cache entries die with the rule set.
//
// Calendars: SIFMA, Federal Reserve, TARGET, United Kingdom, Japan, plus
// weekendsOnly()/noHolidays() and joint(). Japan uses the standard equinox
// approximations and substitute-holiday rule; extreme one-offs are listed.

#include "quantape/datetime/BusinessDayConvention.h"
#include "quantape/datetime/Date.h"
#include "quantape/datetime/Period.h"

#include <algorithm>
#include <cstdint>
#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

namespace quantape::datetime {

namespace detail {

struct CalendarRuleSet {
    std::string name;
    std::uint8_t weekendMask = 0; // bit i set => Weekday(i) is a weekend
    std::vector<Date> closures;   // sorted ad-hoc closures
    std::vector<std::shared_ptr<const CalendarRuleSet>> parents; // joint calendars
    std::function<std::vector<Date>(int)> rule;                  // named calendars

    CalendarRuleSet() = default;
    // A copy is a new rule set and therefore starts with an empty cache; the
    // cache is deliberately not copied (it is keyed to this instance).
    CalendarRuleSet(const CalendarRuleSet& other)
        : name(other.name), weekendMask(other.weekendMask), closures(other.closures),
          parents(other.parents), rule(other.rule) {}

    // Lazy per-year holiday cache owned by the rule set itself: entries live
    // exactly as long as any Calendar referencing this rule set, so a freed
    // rule set's heap address can never alias another rule set's holidays.
    mutable std::mutex cacheMutex;
    mutable std::unordered_map<int, std::shared_ptr<const std::vector<Date>>> yearCache;
};

inline Date nthWeekdayOfMonth(int year, unsigned month, Weekday weekday, int n) {
    const Date first(year, month, 1);
    const int offset = (static_cast<int>(weekday) - static_cast<int>(first.weekday()) + 7) % 7;
    return first.plusDays(offset + 7 * (n - 1));
}

inline Date lastWeekdayOfMonth(int year, unsigned month, Weekday weekday) {
    const Date last = Date(year, month, Date::daysInMonth(year, month));
    const int offset = (static_cast<int>(last.weekday()) - static_cast<int>(weekday) + 7) % 7;
    return last.plusDays(-offset);
}

/// Anonymous Gregorian computus.
inline Date easterSunday(int year) {
    const int a = year % 19;
    const int b = year / 100;
    const int c = year % 100;
    const int d = b / 4;
    const int e = b % 4;
    const int f = (b + 8) / 25;
    const int g = (b - f + 1) / 3;
    const int h = (19 * a + b - d - g + 15) % 30;
    const int i = c / 4;
    const int k = c % 4;
    const int l = (32 + 2 * e + 2 * i - h - k) % 7;
    const int m = (a + 11 * h + 22 * l) / 451;
    const int month = (h + l - 7 * m + 114) / 31;
    const int day = ((h + l - 7 * m + 114) % 31) + 1;
    return Date(year, static_cast<unsigned>(month), static_cast<unsigned>(day));
}

/// Saturday -> previous Friday, Sunday -> next Monday (US observed rule).
inline Date observedSatSun(Date date) {
    if (date.weekday() == Weekday::Saturday) {
        return date.plusDays(-1);
    }
    if (date.weekday() == Weekday::Sunday) {
        return date.plusDays(1);
    }
    return date;
}

inline void addFixedObserved(std::vector<Date>& dates, int year, unsigned month, unsigned day) {
    dates.push_back(observedSatSun(Date(year, month, day)));
}

inline std::vector<Date> usCommonHolidays(int year, bool goodFriday) {
    std::vector<Date> dates;
    addFixedObserved(dates, year, 1, 1);                             // New Year
    dates.push_back(nthWeekdayOfMonth(year, 1, Weekday::Monday, 3)); // MLK
    dates.push_back(nthWeekdayOfMonth(year, 2, Weekday::Monday, 3)); // Presidents
    if (goodFriday) {
        dates.push_back(easterSunday(year).plusDays(-2));
    }
    dates.push_back(lastWeekdayOfMonth(year, 5, Weekday::Monday)); // Memorial
    if (year >= 2022) {
        addFixedObserved(dates, year, 6, 19); // Juneteenth
    }
    addFixedObserved(dates, year, 7, 4);                                // Independence
    dates.push_back(nthWeekdayOfMonth(year, 9, Weekday::Monday, 1));    // Labor
    dates.push_back(nthWeekdayOfMonth(year, 10, Weekday::Monday, 2));   // Columbus
    addFixedObserved(dates, year, 11, 11);                              // Veterans
    dates.push_back(nthWeekdayOfMonth(year, 11, Weekday::Thursday, 4)); // Thanksgiving
    addFixedObserved(dates, year, 12, 25);                              // Christmas
    return dates;
}

inline std::vector<Date> targetHolidays(int year) {
    // No weekend substitution: TARGET treats weekend days as non-business anyway.
    const Date easter = easterSunday(year);
    return {Date(year, 1, 1), easter.plusDays(-2), easter.plusDays(1),
            Date(year, 5, 1), Date(year, 12, 25),  Date(year, 12, 26)};
}

inline std::vector<Date> ukHolidays(int year) {
    std::vector<Date> dates;
    // New Year with substitution (Saturday -> Monday +2, Sunday -> Monday +1).
    const Date newYear(year, 1, 1);
    dates.push_back(newYear.weekday() == Weekday::Saturday
                        ? newYear.plusDays(2)
                        : (newYear.weekday() == Weekday::Sunday ? newYear.plusDays(1) : newYear));
    const Date easter = easterSunday(year);
    dates.push_back(easter.plusDays(-2));                            // Good Friday
    dates.push_back(easter.plusDays(1));                             // Easter Monday
    dates.push_back(nthWeekdayOfMonth(year, 5, Weekday::Monday, 1)); // Early May
    dates.push_back(lastWeekdayOfMonth(year, 5, Weekday::Monday));   // Spring
    dates.push_back(lastWeekdayOfMonth(year, 8, Weekday::Monday));   // Summer
    const Date christmas(year, 12, 25);
    const Date boxing(year, 12, 26);
    dates.push_back(christmas);
    dates.push_back(boxing);
    if (christmas.weekday() == Weekday::Saturday) {
        dates.push_back(christmas.plusDays(2)); // Mon 27
    } else if (christmas.weekday() == Weekday::Sunday) {
        dates.push_back(christmas.plusDays(2)); // Tue 27 (Boxing Mon 26 already added)
    }
    if (boxing.weekday() == Weekday::Saturday) {
        dates.push_back(boxing.plusDays(2)); // Mon 28
    } else if (boxing.weekday() == Weekday::Sunday) {
        dates.push_back(boxing.plusDays(2)); // Tue 28
    }
    return dates;
}

inline Date equinox(int year, bool vernal) {
    const double base = vernal ? 20.8431 : 23.2488;
    const int day =
        static_cast<int>(base + 0.242194 * (year - 1980) - static_cast<int>((year - 1980) / 4));
    return Date(year, vernal ? 3 : 9, static_cast<unsigned>(day));
}

inline std::vector<Date> japanHolidays(int year) {
    std::vector<Date> dates;
    dates.push_back(Date(year, 1, 1));
    dates.push_back(nthWeekdayOfMonth(year, 1, Weekday::Monday, 2)); // Coming of Age
    dates.push_back(Date(year, 2, 11));                              // Foundation
    if (year >= 2020) {
        dates.push_back(Date(year, 2, 23)); // Emperor's Birthday
    }
    dates.push_back(equinox(year, true));                             // Vernal equinox
    dates.push_back(Date(year, 4, 29));                               // Showa
    dates.push_back(Date(year, 5, 3));                                // Constitution
    dates.push_back(Date(year, 5, 4));                                // Greenery
    dates.push_back(Date(year, 5, 5));                                // Children's Day
    dates.push_back(nthWeekdayOfMonth(year, 7, Weekday::Monday, 3));  // Marine
    dates.push_back(Date(year, 8, 11));                               // Mountain
    dates.push_back(nthWeekdayOfMonth(year, 9, Weekday::Monday, 3));  // Respect for Aged
    dates.push_back(equinox(year, false));                            // Autumn equinox
    dates.push_back(nthWeekdayOfMonth(year, 10, Weekday::Monday, 2)); // Sports
    dates.push_back(Date(year, 11, 3));                               // Culture
    dates.push_back(Date(year, 11, 23));                              // Labor Thanksgiving

    // Substitute holidays: a Sunday holiday moves to the next non-holiday day.
    for (std::size_t i = 0; i < dates.size(); ++i) {
        if (dates[i].weekday() != Weekday::Sunday) {
            continue;
        }
        Date candidate = dates[i].plusDays(1);
        while (std::find(dates.begin(), dates.end(), candidate) != dates.end()) {
            candidate = candidate.plusDays(1);
        }
        dates.push_back(candidate);
    }

    if (year == 2019) {
        dates.insert(dates.end(),
                     {Date(2019, 4, 30), Date(2019, 5, 1), Date(2019, 5, 2), Date(2019, 10, 22)});
    } else if (year == 2020) {
        dates.insert(dates.end(), {Date(2020, 7, 23), Date(2020, 7, 24), Date(2020, 8, 10)});
    } else if (year == 2021) {
        dates.insert(dates.end(),
                     {Date(2021, 7, 22), Date(2021, 7, 23), Date(2021, 8, 8), Date(2021, 8, 9)});
    }
    return dates;
}

inline std::shared_ptr<const CalendarRuleSet>
makeRuleSet(std::string name, std::uint8_t weekendMask, std::function<std::vector<Date>(int)> rule,
            std::vector<Date> closures = {}) {
    auto set = std::make_shared<CalendarRuleSet>();
    set->name = std::move(name);
    set->weekendMask = weekendMask;
    std::sort(closures.begin(), closures.end());
    closures.erase(std::unique(closures.begin(), closures.end()), closures.end());
    set->closures = std::move(closures);
    set->rule = std::move(rule);
    return set;
}

inline std::vector<Date> computeYearHolidays(const CalendarRuleSet& rules, int year) {
    std::vector<Date> dates;
    if (rules.rule) {
        dates = rules.rule(year);
    } else {
        for (const auto& parent : rules.parents) {
            const std::vector<Date> parentDates = computeYearHolidays(*parent, year);
            dates.insert(dates.end(), parentDates.begin(), parentDates.end());
        }
    }
    for (const Date& closure : rules.closures) {
        if (closure.year() == year) {
            dates.push_back(closure);
        }
    }
    std::sort(dates.begin(), dates.end());
    dates.erase(std::unique(dates.begin(), dates.end()), dates.end());
    return dates;
}

/// Lazy per-year holidays for a rule set, cached in the rule set's own map.
inline std::shared_ptr<const std::vector<Date>>
yearHolidaysFor(const std::shared_ptr<const CalendarRuleSet>& rules, int year) {
    std::lock_guard lock(rules->cacheMutex);
    const auto it = rules->yearCache.find(year);
    if (it != rules->yearCache.end()) {
        return it->second;
    }
    auto holidays = std::make_shared<const std::vector<Date>>(computeYearHolidays(*rules, year));
    rules->yearCache.emplace(year, holidays);
    return holidays;
}

inline bool sortedContains(const std::vector<Date>& dates, const Date& date) {
    return std::binary_search(dates.begin(), dates.end(), date);
}

} // namespace detail

class Calendar {
public:
    /// Default: weekends-only (no holiday rules).
    Calendar() : rules_(defaultRules()) {}

    static Calendar noHolidays() {
        static const auto rules = detail::makeRuleSet("NoHolidays", 0, nullptr);
        return Calendar(rules);
    }
    static Calendar weekendsOnly() {
        static const auto rules = detail::makeRuleSet("WeekendsOnly", kSatSun, nullptr);
        return Calendar(rules);
    }
    static Calendar sifma() {
        static const auto rules = detail::makeRuleSet(
            "SIFMA", kSatSun, [](int year) { return detail::usCommonHolidays(year, true); },
            usClosures());
        return Calendar(rules);
    }
    static Calendar federalReserve() {
        static const auto rules = detail::makeRuleSet(
            "FederalReserve", kSatSun,
            [](int year) { return detail::usCommonHolidays(year, false); }, usClosures());
        return Calendar(rules);
    }
    static Calendar target() {
        static const auto rules = detail::makeRuleSet("TARGET", kSatSun, detail::targetHolidays);
        return Calendar(rules);
    }
    static Calendar unitedKingdom() {
        static const auto rules =
            detail::makeRuleSet("UnitedKingdom", kSatSun, detail::ukHolidays, ukClosures());
        return Calendar(rules);
    }
    static Calendar japan() {
        static const auto rules = detail::makeRuleSet("Japan", kSatSun, detail::japanHolidays);
        return Calendar(rules);
    }
    static Calendar joint(const Calendar& a, const Calendar& b) {
        auto rules = std::make_shared<detail::CalendarRuleSet>();
        rules->name = a.rules_->name + "+" + b.rules_->name;
        rules->weekendMask = a.rules_->weekendMask | b.rules_->weekendMask;
        rules->parents = {a.rules_, b.rules_};
        Calendar out(rules);
        out.extraHolidays_ = mergeHolidays(a.extraHolidays_, b.extraHolidays_);
        return out;
    }

    bool isWeekend(const Date& date) const {
        return (rules_->weekendMask & (1u << static_cast<unsigned>(date.weekday()))) != 0;
    }
    bool isHoliday(const Date& date) const {
        if (detail::sortedContains(extraHolidays_, date) ||
            detail::sortedContains(rules_->closures, date)) {
            return true;
        }
        const auto holidays = detail::yearHolidaysFor(rules_, date.year());
        return detail::sortedContains(*holidays, date);
    }
    bool isBusinessDay(const Date& date) const { return !isWeekend(date) && !isHoliday(date); }

    Date adjust(const Date& date,
                BusinessDayConvention convention = BusinessDayConvention::Following) const {
        switch (convention) {
            case BusinessDayConvention::Unadjusted:
                return date;
            case BusinessDayConvention::Following:
                return nextBusinessDay(date);
            case BusinessDayConvention::ModifiedFollowing: {
                const Date next = nextBusinessDay(date);
                return next.month() == date.month() ? next : previousBusinessDay(date);
            }
            case BusinessDayConvention::HalfMonthModifiedFollowing: {
                const Date next = nextBusinessDay(date);
                const bool sameHalf = next.month() == date.month() &&
                                      (next.dayOfMonth() <= 15) == (date.dayOfMonth() <= 15);
                return sameHalf ? next : previousBusinessDay(date);
            }
            case BusinessDayConvention::Preceding:
                return previousBusinessDay(date);
            case BusinessDayConvention::ModifiedPreceding: {
                const Date previous = previousBusinessDay(date);
                return previous.month() == date.month() ? previous : nextBusinessDay(date);
            }
            case BusinessDayConvention::Nearest: {
                if (isBusinessDay(date)) {
                    return date;
                }
                const Date next = nextBusinessDay(date);
                const Date previous = previousBusinessDay(date);
                return (next - date) <= (date - previous) ? next : previous;
            }
        }
        return date;
    }

    Date advance(const Date& date, int n, TimeUnit unit,
                 BusinessDayConvention convention = BusinessDayConvention::Following,
                 bool endOfMonth = false) const {
        Date unadjusted = date;
        switch (unit) {
            case TimeUnit::Days:
                unadjusted = date.plusDays(n);
                break;
            case TimeUnit::Weeks:
                unadjusted = date.plusWeeks(n);
                break;
            case TimeUnit::Months:
                unadjusted = date.plusMonths(n, endOfMonth);
                break;
            case TimeUnit::Years:
                unadjusted = date.plusYears(n, endOfMonth);
                break;
        }
        return adjust(unadjusted, convention);
    }
    Date advance(const Date& date, const Period& period,
                 BusinessDayConvention convention = BusinessDayConvention::Following,
                 bool endOfMonth = false) const {
        return advance(date, period.length(), period.unit(), convention, endOfMonth);
    }

    /// Business days in (from, to] (negative when to < from).
    int businessDaysBetween(const Date& from, const Date& to) const {
        if (from == to) {
            return 0;
        }
        if (to < from) {
            return -businessDaysBetween(to, from);
        }
        int count = 0;
        for (Date date = from.plusDays(1); date <= to; date = date.plusDays(1)) {
            if (isBusinessDay(date)) {
                ++count;
            }
        }
        return count;
    }

    const std::string& name() const { return rules_->name; }

    Calendar withExtraHolidays(std::vector<Date> extra) const {
        Calendar out = *this;
        out.extraHolidays_ = mergeHolidays(extraHolidays_, std::move(extra));
        return out;
    }
    Calendar withoutHoliday(const Date& date) const {
        Calendar out = *this;
        out.extraHolidays_.erase(
            std::remove(out.extraHolidays_.begin(), out.extraHolidays_.end(), date),
            out.extraHolidays_.end());
        if (detail::sortedContains(out.rules_->closures, date)) {
            auto rules = std::make_shared<detail::CalendarRuleSet>(*out.rules_);
            rules->closures.erase(std::remove(rules->closures.begin(), rules->closures.end(), date),
                                  rules->closures.end());
            out.rules_ = std::move(rules);
        }
        return out;
    }

private:
    static constexpr std::uint8_t kSatSun = 0x60; // Saturday | Sunday bits

    static std::vector<Date> usClosures() {
        return {Date(2001, 9, 11), Date(2001, 9, 12), Date(2001, 9, 13),  Date(2001, 9, 14),
                Date(2004, 6, 11), Date(2007, 1, 2),  Date(2012, 10, 29), Date(2012, 10, 30),
                Date(2018, 12, 5), Date(2025, 1, 9)};
    }
    static std::vector<Date> ukClosures() {
        return {Date(2011, 4, 29), Date(2012, 6, 4),  Date(2012, 6, 5), Date(2020, 5, 8),
                Date(2022, 6, 3),  Date(2022, 9, 19), Date(2023, 5, 8)};
    }

    static std::vector<Date> mergeHolidays(std::vector<Date> a, std::vector<Date> b) {
        a.insert(a.end(), b.begin(), b.end());
        std::sort(a.begin(), a.end());
        a.erase(std::unique(a.begin(), a.end()), a.end());
        return a;
    }

    explicit Calendar(std::shared_ptr<const detail::CalendarRuleSet> rules)
        : rules_(std::move(rules)) {}

    static std::shared_ptr<const detail::CalendarRuleSet> defaultRules() {
        static const auto rules = detail::makeRuleSet("WeekendsOnly", kSatSun, nullptr);
        return rules;
    }

    /// First business day on or after `date`.
    Date nextBusinessDay(Date date) const {
        while (!isBusinessDay(date)) {
            date = date.plusDays(1);
        }
        return date;
    }
    /// Last business day on or before `date`.
    Date previousBusinessDay(Date date) const {
        while (!isBusinessDay(date)) {
            date = date.plusDays(-1);
        }
        return date;
    }

    std::shared_ptr<const detail::CalendarRuleSet> rules_;
    std::vector<Date> extraHolidays_;
};

} // namespace quantape::datetime
