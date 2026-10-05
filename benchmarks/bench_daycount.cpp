// bench_daycount.cpp — cached vs uncached day-count fractions and cache sizes
//
// Workloads:
//   schedule  repeated coupon-period pairs (what annuity/bootstrap code asks)
//   random    wide random date pairs (cache-hostile working set)
//   mixed     90% hot pairs + 10% random
//
// Reports ns/call per convention and cache size, plus cache bytes and process
// peak RSS so the memory pressure is visible next to the timings.

#include "quantape/datetime/DayCounter.h"
#include "quantape/format/Number.h"
#include "quantape/log/Log.h"

#include <chrono>
#include <cstdint>
#include <random>
#include <vector>

#if defined(__APPLE__)
#include <sys/resource.h>
#elif defined(__linux__)
#include <sys/resource.h>
#endif

using namespace quantape::datetime;

namespace {

volatile double g_sink = 0.0;

struct Pair {
    Date begin;
    Date end;
};

std::vector<Pair> schedulePairs() {
    std::vector<Pair> pairs;
    const Date origin(2015, 1, 1);
    for (int schedule = 0; schedule < 40; ++schedule) {
        const Date start = origin.plusMonths(schedule * 7, false);
        for (int coupon = 1; coupon <= 20; ++coupon) {
            pairs.push_back({start, start.plusMonths(6 * coupon, false)});
        }
    }
    return pairs;
}

std::vector<Pair> randomPairs(std::size_t count) {
    std::mt19937_64 rng(42);
    std::vector<Pair> pairs(count);
    for (Pair& pair : pairs) {
        const auto begin = static_cast<std::int32_t>(-30'000 + static_cast<int>(rng() % 60'000));
        const auto end = static_cast<std::int32_t>(begin + 30 + static_cast<int>(rng() % 3'600));
        pair = {Date::fromSerial(begin), Date::fromSerial(end)};
    }
    return pairs;
}

double peakRssBytes() {
#if defined(RUSAGE_SELF)
    struct rusage usage{};
    getrusage(RUSAGE_SELF, &usage);
#if defined(__APPLE__)
    return static_cast<double>(usage.ru_maxrss);
#else
    return static_cast<double>(usage.ru_maxrss) * 1024.0;
#endif
#else
    return 0.0;
#endif
}

template <typename Fn>
double nsPerCall(std::size_t iterations, const Fn& fn) {
    const auto start = std::chrono::steady_clock::now();
    double sum = 0.0;
    for (std::size_t i = 0; i < iterations; ++i) {
        sum += fn(i);
    }
    const auto stop = std::chrono::steady_clock::now();
    g_sink = sum;
    return std::chrono::duration<double, std::nano>(stop - start).count() /
           static_cast<double>(iterations);
}

struct Workload {
    const char* name;
    const std::vector<Pair>& pairs;
    std::size_t iterations;
};

} // namespace

int main() {
    const std::vector<Pair> schedule = schedulePairs();
    const std::vector<Pair> random = randomPairs(100'000);
    std::vector<Pair> mixed;
    mixed.reserve(100'000);
    {
        std::mt19937_64 rng(7);
        for (std::size_t i = 0; i < 100'000; ++i) {
            if (i % 10 == 0) {
                mixed.push_back(random[i]);
            } else {
                mixed.push_back(schedule[rng() % 64]);
            }
        }
    }
    const Workload workloads[] = {
        {"schedule", schedule, 2'000'000},
        {"random", random, 2'000'000},
        {"mixed", mixed, 2'000'000},
    };

    const DayCount conventions[] = {
        DayCount::Actual365Fixed,  DayCount::Actual360,  DayCount::ActualActualISDA,
        DayCount::ActualActualAFB, DayCount::ThirtyE360, DayCount::Actual365Actual,
    };
    const std::size_t cacheSizes[] = {1'024, 65'536, 1'048'576};

    QTA_LOG_INFO("bench", "day-count cache benchmark (2M calls per cell)");

    for (DayCount convention : conventions) {
        DayCounter counter(convention);
        const std::string convName(dayCountName(convention));

        enableDayCountCache(false);
        for (const Workload& workload : workloads) {
            double sum = 0.0;
            const double ns = nsPerCall(workload.iterations, [&](std::size_t i) {
                const Pair& pair = workload.pairs[i % workload.pairs.size()];
                sum += counter.yearFractionUncached(pair.begin, pair.end);
                return 0.0;
            });
            g_sink = sum;
            QTA_LOG_INFO("bench", "{} {:<9} uncached  {} ns/call", convName, workload.name,
                         quantape::format::num(ns, 4));
        }

        for (std::size_t slots : cacheSizes) {
            setDayCountCacheSize(slots);
            enableDayCountCache(true);
            // Warm the thread-local allocation up so timing excludes it.
            (void)counter.yearFraction(schedule[0].begin, schedule[0].end);
            resetDayCountCacheStats();
            for (const Workload& workload : workloads) {
                const double ns = nsPerCall(workload.iterations, [&](std::size_t i) {
                    const Pair& pair = workload.pairs[i % workload.pairs.size()];
                    return counter.yearFraction(pair.begin, pair.end);
                });
                const DayCountCacheStats stats = dayCountCacheStats();
                if (stats.hits + stats.misses == 0) {
                    QTA_LOG_INFO("bench", "{} {:<9} cache bypassed (cheap convention)", convName,
                                 workload.name);
                    continue;
                }
                const double hitRate = static_cast<double>(stats.hits) /
                                       static_cast<double>(stats.hits + stats.misses);
                QTA_LOG_INFO("bench", "{} {:<9} {:>8} slots  {} ns/call  hit={}", convName,
                             workload.name, slots, quantape::format::num(ns, 4),
                             quantape::format::num(hitRate, 3));
            }
        }
        enableDayCountCache(false);
    }

    setDayCountCacheSize(1'048'576);
    enableDayCountCache(true);
    const double rssBefore = peakRssBytes();
    for (const Pair& pair : schedule) {
        (void)DayCounter(DayCount::Actual365Fixed).yearFraction(pair.begin, pair.end);
    }
    const double rssAfter = peakRssBytes();
    const std::size_t cacheBytes = dayCountCacheBytes();
    enableDayCountCache(false);
    setDayCountCacheSize(65'536);

    QTA_LOG_INFO("bench", "cache memory: 1Mi slots = {} bytes; peak RSS {} -> {} MB",
                 quantape::format::num(static_cast<double>(cacheBytes), 4),
                 quantape::format::num(rssBefore / (1024.0 * 1024.0), 2),
                 quantape::format::num(rssAfter / (1024.0 * 1024.0), 2));
    QTA_LOG_INFO("bench", "done");
    return 0;
}
