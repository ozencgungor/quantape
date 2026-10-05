#pragma once

#include <cstdint>

namespace quantape::datetime {

enum class BusinessDayConvention : std::uint8_t {
    Unadjusted,
    Following,
    ModifiedFollowing,
    HalfMonthModifiedFollowing, // "bi-monthly": halves are 1st-15th and 16th+
    Preceding,
    ModifiedPreceding,
    Nearest, // ties go forward
};

} // namespace quantape::datetime
