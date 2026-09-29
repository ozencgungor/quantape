#pragma once

#include <cstdint>

namespace quantape::datetime {

enum class Frequency : std::uint8_t {
    Annual,
    Semiannual,
    Quarterly,
    Bimonthly,
    Monthly,
    Weekly,
    Daily,
    Other,
};

constexpr int periodsPerYear(Frequency frequency) {
    switch (frequency) {
        case Frequency::Annual: return 1;
        case Frequency::Semiannual: return 2;
        case Frequency::Quarterly: return 4;
        case Frequency::Bimonthly: return 6;
        case Frequency::Monthly: return 12;
        case Frequency::Weekly: return 52;
        case Frequency::Daily: return 365;
        case Frequency::Other: return 0;
    }
    return 0;
}

}  // namespace quantape::datetime
