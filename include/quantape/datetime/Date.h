#pragma once

// Date: day-count serial (days since 1970-01-01) with fast, overflow-safe
// Gregorian conversion (Ben Joffe's Julian-Map/bucket algorithms; see
// internal_docs/datetime_design.md section 2.4). Dates are non-differentiable
// integer values; all arithmetic is exact and constexpr.

#include <compare>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <stdexcept>
#include <string>
#include <string_view>

namespace quantape::datetime {

enum class Weekday : std::uint8_t {
    Monday,
    Tuesday,
    Wednesday,
    Thursday,
    Friday,
    Saturday,
    Sunday
};

inline constexpr const char* weekdayName(Weekday day) {
    constexpr const char* names[] = {"Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"};
    return names[static_cast<unsigned>(day)];
}

namespace detail {

/// floor division / modulo for signed values.
constexpr std::int64_t floorDiv(std::int64_t a, std::int64_t b) {
    const std::int64_t q = a / b;
    return (a % b != 0 && ((a < 0) != (b < 0))) ? q - 1 : q;
}

/// Civil (y, m, d) -> serial days since 1970-01-01, full int32 range safe.
/// Inverse of civilFromSerial; see the design doc for the constants.
constexpr std::int32_t serialFromCivil(int year, unsigned month, unsigned day) {
    const std::int64_t y = static_cast<std::int64_t>(year);
    const std::int64_t m = static_cast<std::int64_t>(month);
    const std::int64_t d = static_cast<std::int64_t>(day);
    const bool bump = m <= 2;
    const std::int64_t yy = y + 5'880'000 - (bump ? 1 : 0);
    const std::int64_t cent = floorDiv(yy, 100);
    const std::int64_t y_days = yy * 365 + floorDiv(yy, 4) - cent + floorDiv(cent, 4);
    const std::int64_t m_days = (979 * m + (bump ? 8829 : -2919)) / 32;
    const std::int64_t rata = y_days + m_days + d - 2'148'345'369;
    return static_cast<std::int32_t>(rata);
}

/// Serial -> civil date, full int32 input range safe (bucket technique).
constexpr void civilFromSerial(std::int32_t serial, int& year, unsigned& month, unsigned& day) {
    const std::int64_t d0 = static_cast<std::int64_t>(serial) + 2'147'483'648ll;
    const std::int64_t bucket = d0 >> 20;
    const std::int64_t qday = (d0 - bucket * 1'022'679) * 4 + 524'943;
    const std::int64_t cent = qday / 146'097;
    const std::int64_t qjul = qday - (cent & ~std::int64_t{3}) + cent * 4;
    const std::int64_t y = qjul / 1461;
    const std::int64_t yday = (qjul % 1461) / 4;
    const std::int64_t n = yday * 2141 + 197'913;
    const std::int64_t m = n / 65'536;
    const std::int64_t dd = (n % 65'536) / 2141;
    const bool bump = yday >= 306;
    year = static_cast<int>(y + bucket * 2800 - 5'878'000 + (bump ? 1 : 0));
    month = static_cast<unsigned>(bump ? m - 12 : m);
    day = static_cast<unsigned>(dd + 1);
}

/// Drepper-Neri-Schneider: the %25 test has false positives that the bitmask
/// cancels, and compiles to a multiply instead of two divisions.
constexpr bool isLeapYearImpl(int year) {
    const bool quarter_century = (year % 25 == 0);
    return (year & (quarter_century ? 15 : 3)) == 0;
}

constexpr unsigned daysInMonthImpl(int year, unsigned month) {
    constexpr unsigned lengths[] = {31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31};
    if (month < 1 || month > 12) {
        return 0;
    }
    if (month == 2 && isLeapYearImpl(year)) {
        return 29;
    }
    return lengths[month - 1];
}

constexpr unsigned dayOfYearImpl(int year, unsigned month, unsigned day) {
    constexpr unsigned cumulative[] = {0, 31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334};
    return cumulative[month - 1] + day + ((month > 2 && isLeapYearImpl(year)) ? 1u : 0u);
}

} // namespace detail

class Date {
public:
    constexpr Date() = default;
    /// Precondition: 1 <= month <= 12 and 1 <= day <= daysInMonth(year, month).
    /// Use parse()/checked() for validated construction.
    constexpr Date(int year, unsigned month, unsigned day)
        : serial_(detail::serialFromCivil(year, month, day)) {}

    static constexpr Date fromSerial(std::int32_t serial) {
        Date date;
        date.serial_ = serial;
        return date;
    }

    /// Validated construction; throws std::invalid_argument.
    static Date checked(int year, unsigned month, unsigned day) {
        if (month < 1 || month > 12 || day < 1 || day > detail::daysInMonthImpl(year, month)) {
            throw std::invalid_argument("Date: invalid year/month/day");
        }
        return Date(year, month, day);
    }

    /// Strict ISO "YYYY-MM-DD"; throws std::invalid_argument.
    static Date parse(std::string_view text);

    constexpr std::int32_t serial() const { return serial_; }
    constexpr int year() const {
        int y = 0;
        unsigned m = 0, d = 0;
        detail::civilFromSerial(serial_, y, m, d);
        return y;
    }
    constexpr unsigned month() const {
        int y = 0;
        unsigned m = 0, d = 0;
        detail::civilFromSerial(serial_, y, m, d);
        return m;
    }
    constexpr unsigned dayOfMonth() const {
        int y = 0;
        unsigned m = 0, d = 0;
        detail::civilFromSerial(serial_, y, m, d);
        return d;
    }
    /// Single decode of all three civil fields (cheaper than year()+month()+dayOfMonth()).
    constexpr void decompose(int& year_out, unsigned& month_out, unsigned& day_out) const {
        detail::civilFromSerial(serial_, year_out, month_out, day_out);
    }
    constexpr Weekday weekday() const {
        // Neri's unsigned-cast trick: full-range mod 7 without sign handling.
        // Sunday = 0 on the epoch; 2^32 % 7 == 4 already biases negative days.
        const std::uint32_t u = static_cast<std::uint32_t>(serial_);
        const std::uint32_t sunday0 = (u + (serial_ >= 0 ? 4u : 0u)) % 7u;
        return static_cast<Weekday>((sunday0 + 6u) % 7u); // Monday = 0
    }
    static constexpr bool isLeapYear(int year) { return detail::isLeapYearImpl(year); }
    static constexpr unsigned daysInMonth(int year, unsigned month) {
        return detail::daysInMonthImpl(year, month);
    }
    constexpr bool isEndOfMonth() const { return dayOfMonth() == daysInMonth(year(), month()); }
    constexpr Date endOfMonth() const {
        return Date(year(), month(), daysInMonth(year(), month()));
    }
    constexpr Date firstOfMonth() const { return Date(year(), month(), 1); }
    constexpr unsigned dayOfYear() const {
        int y = 0;
        unsigned m = 0, d = 0;
        detail::civilFromSerial(serial_, y, m, d);
        return detail::dayOfYearImpl(y, m, d);
    }

    constexpr Date plusDays(std::int32_t days) const { return fromSerial(serial_ + days); }
    constexpr Date plusWeeks(std::int32_t weeks) const { return plusDays(weeks * 7); }
    /// Month arithmetic with clipping; `preserveEndOfMonth` maps month-end to
    /// month-end (used by schedule generation).
    constexpr Date plusMonths(int months, bool preserveEndOfMonth = false) const {
        const bool eom = preserveEndOfMonth && isEndOfMonth();
        const std::int64_t total = static_cast<std::int64_t>(year()) * 12 + (month() - 1) + months;
        const int y = static_cast<int>(detail::floorDiv(total, 12));
        const unsigned m = static_cast<unsigned>(total - detail::floorDiv(total, 12) * 12) + 1;
        const unsigned last = daysInMonth(y, m);
        const unsigned d = eom ? last : (dayOfMonth() > last ? last : dayOfMonth());
        return Date(y, m, d);
    }
    constexpr Date plusYears(int years, bool preserveEndOfMonth = false) const {
        return plusMonths(years * 12, preserveEndOfMonth);
    }

    constexpr auto operator<=>(const Date&) const = default;
    constexpr std::int32_t operator-(const Date& other) const { return serial_ - other.serial_; }
    constexpr Date operator+(std::int32_t days) const { return plusDays(days); }
    constexpr Date operator-(std::int32_t days) const { return plusDays(-days); }
    constexpr Date& operator++() { return *this = plusDays(1); }
    constexpr Date operator++(int) {
        Date copy = *this;
        ++(*this);
        return copy;
    }
    constexpr Date& operator--() { return *this = plusDays(-1); }
    constexpr Date operator--(int) {
        Date copy = *this;
        --(*this);
        return copy;
    }

    std::string toIso() const;

private:
    std::int32_t serial_ = 0;
};

inline std::string Date::toIso() const {
    int y = 0;
    unsigned m = 0, d = 0;
    detail::civilFromSerial(serial_, y, m, d);
    char buffer[11];
    auto writeDigits = [&buffer](int pos, unsigned value, int digits) {
        for (int i = digits - 1; i >= 0; --i) {
            buffer[pos + i] = static_cast<char>('0' + value % 10);
            value /= 10;
        }
    };
    writeDigits(0, y < 0 ? 0u : static_cast<unsigned>(y), 4);
    buffer[4] = '-';
    writeDigits(5, m, 2);
    buffer[7] = '-';
    writeDigits(8, d, 2);
    return std::string(buffer, 10);
}

inline Date Date::parse(std::string_view text) {
    if (text.size() != 10 || text[4] != '-' || text[7] != '-') {
        throw std::invalid_argument("Date::parse: expected YYYY-MM-DD");
    }
    auto parseDigits = [&text](std::size_t pos, std::size_t count) {
        unsigned value = 0;
        for (std::size_t i = 0; i < count; ++i) {
            const char c = text[pos + i];
            if (c < '0' || c > '9') {
                throw std::invalid_argument("Date::parse: non-digit");
            }
            value = value * 10 + static_cast<unsigned>(c - '0');
        }
        return value;
    };
    const int year = static_cast<int>(parseDigits(0, 4));
    const unsigned month = parseDigits(5, 2);
    const unsigned day = parseDigits(8, 2);
    return checked(year, month, day);
}

} // namespace quantape::datetime

namespace std {
template <>
struct hash<quantape::datetime::Date> {
    std::size_t operator()(const quantape::datetime::Date& date) const noexcept {
        return std::hash<std::int32_t>{}(date.serial());
    }
};
} // namespace std
