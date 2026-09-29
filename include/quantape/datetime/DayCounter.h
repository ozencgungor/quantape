#pragma once

// DayCounter: runtime day-count convention value with the full catalogue.
//
// A DayCounter is a small value type (convention + optional context). The hot
// path yearFraction(d1, d2) can be served from a per-thread direct-mapped
// cache keyed by the date pair and convention; context-dependent conventions
// (schedule/calendar/termination) bypass the cache unless their context is
// empty. Cache sizes are configurable; memory is slots * 24 bytes per thread.

#include "quantape/datetime/Date.h"
#include "quantape/datetime/Frequency.h"

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <stdexcept>
#include <string_view>
#include <vector>

namespace quantape::datetime {

class Calendar;
class Schedule;

enum class DayCount : std::uint8_t {
    Actual360,
    Actual364,
    Actual365Fixed,
    Actual365_25,
    Actual366,
    NL365,
    NL360,
    ActualActualISDA,
    ActualActualAFB,
    ActualActualYear,
    ActualActualICMA,
    Actual365Actual,
    Actual365L,
    OneOne,
    Simple,
    Thirty360US,
    ThirtyU360EOM,
    Thirty360BondBasis,
    ThirtyE360,
    ThirtyE360ISDA,
    ThirtyEPlus360,
    Thirty360Italian,
    Thirty360PSA,
    Thirty365,
    ThirtyE365,
    Bus252,
};

constexpr std::string_view dayCountName(DayCount convention) {
    switch (convention) {
        case DayCount::Actual360: return "ACT/360";
        case DayCount::Actual364: return "ACT/364";
        case DayCount::Actual365Fixed: return "ACT/365F";
        case DayCount::Actual365_25: return "ACT/365.25";
        case DayCount::Actual366: return "ACT/366";
        case DayCount::NL365: return "NL/365";
        case DayCount::NL360: return "NL/360";
        case DayCount::ActualActualISDA: return "ACT/ACT ISDA";
        case DayCount::ActualActualAFB: return "ACT/ACT AFB";
        case DayCount::ActualActualYear: return "ACT/ACT Year";
        case DayCount::ActualActualICMA: return "ACT/ACT ICMA";
        case DayCount::Actual365Actual: return "ACT/365 Actual";
        case DayCount::Actual365L: return "ACT/365L";
        case DayCount::OneOne: return "1/1";
        case DayCount::Simple: return "SIMPLE";
        case DayCount::Thirty360US: return "30/360 US";
        case DayCount::ThirtyU360EOM: return "30U/360 EOM";
        case DayCount::Thirty360BondBasis: return "30/360 Bond Basis";
        case DayCount::ThirtyE360: return "30E/360";
        case DayCount::ThirtyE360ISDA: return "30E/360 ISDA";
        case DayCount::ThirtyEPlus360: return "30E+/360";
        case DayCount::Thirty360Italian: return "30/360 Italian";
        case DayCount::Thirty360PSA: return "30/360 PSA";
        case DayCount::Thirty365: return "30/365";
        case DayCount::ThirtyE365: return "30E/365";
        case DayCount::Bus252: return "BUS/252";
    }
    return "?";
}

/// Extra inputs for conventions that are not pure functions of (d1, d2):
/// ICMA needs the coupon schedule, ACT/365L and AFB-style rules need the
/// reference period, BUS/252 needs a calendar, 30E/360 ISDA needs the
/// termination date. Unused fields are ignored.
struct DayCountContext {
    std::optional<Date> refStart;
    std::optional<Date> refEnd;
    std::optional<Date> termination;
    Frequency frequency = Frequency::Annual;
    const Calendar* calendar = nullptr;  // BUS/252
    const Schedule* schedule = nullptr;  // ACT/ACT ICMA
};

struct DayCountCacheStats {
    std::uint64_t hits = 0;
    std::uint64_t misses = 0;
    std::size_t slots = 0;      // configured slots in the calling thread
    std::size_t bytes = 0;      // slots * slot size
};

void enableDayCountCache(bool enabled) noexcept;
bool dayCountCacheEnabled() noexcept;
/// Rounds up to a power of two; 0 disables. Thread-local storage, so threads
/// that already logged a year fraction resize lazily on their next call.
void setDayCountCacheSize(std::size_t slots);
std::size_t dayCountCacheSize() noexcept;
std::size_t dayCountCacheBytes() noexcept;
DayCountCacheStats dayCountCacheStats() noexcept;
void resetDayCountCacheStats() noexcept;

namespace detail {

/// Convention implementation (defined in DayCounters.h).
double yearFractionFor(DayCount convention, const Date& d1, const Date& d2,
                       const DayCountContext& context);

struct DayCountCacheSlot {
    std::uint64_t key = 0;
    double value = 0.0;
    DayCount convention = static_cast<DayCount>(0xFF);
};

constexpr std::uint64_t splitmix64(std::uint64_t x) {
    x += 0x9E3779B97F4A7C15ULL;
    x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ULL;
    x = (x ^ (x >> 27)) * 0x94D049BB133111EBULL;
    return x ^ (x >> 31);
}

inline std::atomic<bool>& cacheEnabledFlag() noexcept {
    static std::atomic<bool> flag{true};
    return flag;
}
inline std::atomic<std::uint32_t>& cacheSlotBits() noexcept {
    static std::atomic<std::uint32_t> bits{16};  // 65,536 slots = 1.5 MB
    return bits;
}
inline std::atomic<std::uint64_t>& cacheHits() noexcept {
    static std::atomic<std::uint64_t> value{0};
    return value;
}
inline std::atomic<std::uint64_t>& cacheMisses() noexcept {
    static std::atomic<std::uint64_t> value{0};
    return value;
}

struct ThreadDayCountCache {
    std::vector<DayCountCacheSlot> slots;
    std::uint32_t bits = 0;

    void ensure(std::uint32_t requestedBits) {
        if (bits != requestedBits) {
            bits = requestedBits;
            slots.assign(std::size_t{1} << requestedBits, DayCountCacheSlot{});
        }
    }
};

inline ThreadDayCountCache& threadDayCountCache() {
    thread_local ThreadDayCountCache cache;
    return cache;
}

/// Caching only pays for the multi-branch/year-walking conventions; the cheap
/// day/denominator formulas are faster than a cache lookup (measured).
inline bool cacheWorthy(DayCount convention) {
    switch (convention) {
        case DayCount::ActualActualISDA:
        case DayCount::ActualActualAFB:
        case DayCount::ActualActualYear:
        case DayCount::Actual365Actual:
        case DayCount::NL365:
        case DayCount::NL360:
        case DayCount::Simple:
        case DayCount::Thirty360US:
        case DayCount::ThirtyU360EOM:
        case DayCount::Thirty360BondBasis:
        case DayCount::ThirtyE360:
        case DayCount::ThirtyE360ISDA:
        case DayCount::ThirtyEPlus360:
        case DayCount::Thirty360Italian:
        case DayCount::Thirty360PSA:
        case DayCount::Thirty365:
        case DayCount::ThirtyE365:
            return true;
        default:
            return false;
    }
}

/// True when the result depends only on (d1, d2, convention) and the
/// convention is expensive enough to be worth a lookup: schedule- and
/// calendar-driven conventions, ACT/365L (reference period) and 30E/360 ISDA
/// with an explicit termination date bypass the cache.
inline bool cacheUsable(DayCount convention, const DayCountContext& context) {
    if (!cacheWorthy(convention)) {
        return false;
    }
    if (convention == DayCount::ActualActualICMA || convention == DayCount::Actual365L ||
        convention == DayCount::Bus252) {
        return false;
    }
    if (convention == DayCount::ThirtyE360ISDA && context.termination.has_value()) {
        return false;
    }
    return true;
}

}  // namespace detail

class DayCounter {
public:
    DayCounter() = default;
    explicit DayCounter(DayCount convention, DayCountContext context = {})
        : convention_(convention), context_(context) {}

    /// Cached when possible; falls back to yearFractionUncached.
    double yearFraction(const Date& d1, const Date& d2) const {
        if (!detail::cacheEnabledFlag().load(std::memory_order_relaxed) ||
            !detail::cacheUsable(convention_, context_)) {
            return detail::yearFractionFor(convention_, d1, d2, context_);
        }
        auto& cache = detail::threadDayCountCache();
        const std::uint32_t bits = detail::cacheSlotBits().load(std::memory_order_relaxed);
        if (bits == 0) {
            return detail::yearFractionFor(convention_, d1, d2, context_);
        }
        cache.ensure(bits);

        const std::uint64_t key =
            (static_cast<std::uint64_t>(static_cast<std::uint32_t>(d1.serial())) << 32) |
            static_cast<std::uint32_t>(d2.serial());
        const std::uint64_t index =
            detail::splitmix64(key ^ (static_cast<std::uint64_t>(convention_) + 1) *
                                           0x9E3779B97F4A7C15ULL) &
            (cache.slots.size() - 1);
        detail::DayCountCacheSlot& slot = cache.slots[static_cast<std::size_t>(index)];
        if (slot.key == key && slot.convention == convention_) {
            detail::cacheHits().fetch_add(1, std::memory_order_relaxed);
            return slot.value;
        }
        detail::cacheMisses().fetch_add(1, std::memory_order_relaxed);
        const double value = detail::yearFractionFor(convention_, d1, d2, context_);
        slot.key = key;
        slot.value = value;
        slot.convention = convention_;
        return value;
    }

    double yearFractionUncached(const Date& d1, const Date& d2) const {
        return detail::yearFractionFor(convention_, d1, d2, context_);
    }

    DayCount convention() const { return convention_; }
    const DayCountContext& context() const { return context_; }
    std::string_view name() const { return dayCountName(convention_); }

private:
    DayCount convention_ = DayCount::Actual365Fixed;
    DayCountContext context_{};
};

}  // namespace quantape::datetime

#include "quantape/datetime/DayCounters.h"
