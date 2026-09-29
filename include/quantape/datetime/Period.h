#pragma once

// Period / Tenor: an integer length in days, weeks, months or years.
// Comparison uses an approximate month scale for ordering only; exact
// arithmetic is always through Date (calendar-date) or a Schedule.

#include "quantape/datetime/Date.h"

#include <compare>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <string_view>

namespace quantape::datetime {

enum class TimeUnit : std::uint8_t { Days, Weeks, Months, Years };

class Period {
public:
    constexpr Period() = default;
    constexpr Period(int length, TimeUnit unit) : length_(length), unit_(unit) {}

    /// Parses "10D", "2W", "3M", "1Y"; throws std::invalid_argument.
    static Period parse(std::string_view text) {
        if (text.size() < 2) {
            throw std::invalid_argument("Period::parse: expected <n><D|W|M|Y>");
        }
        const char suffix = text.back();
        TimeUnit unit;
        switch (suffix) {
            case 'D': unit = TimeUnit::Days; break;
            case 'W': unit = TimeUnit::Weeks; break;
            case 'M': unit = TimeUnit::Months; break;
            case 'Y': unit = TimeUnit::Years; break;
            default: throw std::invalid_argument("Period::parse: unknown unit");
        }
        int value = 0;
        for (std::size_t i = 0; i + 1 < text.size(); ++i) {
            const char c = text[i];
            if (c < '0' || c > '9') {
                throw std::invalid_argument("Period::parse: non-digit length");
            }
            value = value * 10 + (c - '0');
        }
        return Period(value, unit);
    }

    constexpr int length() const { return length_; }
    constexpr TimeUnit unit() const { return unit_; }
    constexpr bool isNegative() const { return length_ < 0; }

    /// Approximate month scale, ordering only (1M = 4W = 30D).
    constexpr std::int32_t approxMonthsTimes4() const {
        switch (unit_) {
            case TimeUnit::Days: return length_ * 4 / 30;
            case TimeUnit::Weeks: return length_;
            case TimeUnit::Months: return length_ * 4;
            case TimeUnit::Years: return length_ * 48;
        }
        return 0;
    }

    constexpr auto operator<=>(const Period& other) const {
        return approxMonthsTimes4() <=> other.approxMonthsTimes4();
    }
    constexpr bool operator==(const Period& other) const {
        return length_ == other.length_ && unit_ == other.unit_;
    }

    /// Calendar-date advance (no business-day adjustment).
    constexpr Date advance(const Date& date, bool endOfMonth = false) const {
        switch (unit_) {
            case TimeUnit::Days: return date.plusDays(length_);
            case TimeUnit::Weeks: return date.plusWeeks(length_);
            case TimeUnit::Months: return date.plusMonths(length_, endOfMonth);
            case TimeUnit::Years: return date.plusYears(length_, endOfMonth);
        }
        return date;
    }

    std::string toString() const {
        constexpr char suffixes[] = {'D', 'W', 'M', 'Y'};
        return std::to_string(length_) + suffixes[static_cast<int>(unit_)];
    }

private:
    int length_ = 0;
    TimeUnit unit_ = TimeUnit::Days;
};

using Tenor = Period;

}  // namespace quantape::datetime
