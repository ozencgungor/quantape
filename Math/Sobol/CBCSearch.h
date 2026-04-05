#pragma once
/**
 * @file CBCSearch.h
 * @brief Component-by-component search for optimal Sobol initial direction numbers.
 *
 * Three quality levels:
 *   Level 0 (RANDOM):  Random valid m_i. Seconds for 100K dims.
 *   Level 1 (PROP_A):  CBC optimizing single-dimension t-value (Property A).
 *   Level 2 (PROJ_2D): CBC optimizing 2D projection quality against existing dims.
 *
 * Parallelization:
 *   Each polynomial is independent → embarrassingly parallel.
 *   process_batch() uses a thread pool internally.
 *   For container dispatch: serialize WorkItem to JSON, deserialize SearchResult.
 */

#include <atomic>
#include <cstdint>
#include <functional>
#include <limits>
#include <mutex>
#include <thread>
#include <vector>

#include "DirectionNumbers.h"
#include "GF2.h"

namespace mc {
namespace sobol {

enum class SearchLevel { RANDOM = 0, PROP_A = 1, PROJ_2D = 2 };

struct SearchResult {
    uint32_t dim;
    uint32_t degree;
    uint64_t polynomial;
    uint64_t a_encoded;
    std::vector<uint64_t> m;
    int t_val;
};

// ═══════════════════════════════════════════════════════════════════════════
// Level 0: Random valid direction numbers (instant)
// ═══════════════════════════════════════════════════════════════════════════

inline SearchResult search_random(uint64_t poly, int s, uint64_t seed) {
    // Simple splitmix64 for reproducible per-polynomial randomness
    auto rng = [&seed]() -> uint64_t {
        seed += 0x9e3779b97f4a7c15ULL;
        uint64_t z = seed;
        z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
        z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
        return z ^ (z >> 31);
    };

    std::vector<uint64_t> m(s);
    m[0] = 1;
    for (int k = 1; k < s; ++k) {
        uint64_t range = 1ULL << k; // number of valid odd values in [1, 2^{k+1})
        uint64_t mk = (rng() % range) * 2 + 1;
        if (mk >= (1ULL << (k + 1)))
            mk -= 2; // clamp
        m[k] = mk;
    }

    auto v = compute_v(poly, s, m);
    int t = t_value(v);

    return {0, (uint32_t)s, poly, gf2::encode_a(poly, s), std::move(m), t};
}

// ═══════════════════════════════════════════════════════════════════════════
// Level 1: CBC minimizing t-value (Property A search)
// ═══════════════════════════════════════════════════════════════════════════

inline SearchResult search_property_a(uint64_t poly, int s) {
    std::vector<uint64_t> best_m;
    best_m.reserve(s);
    best_m.push_back(1); // m_1 = 1 always

    for (int k = 1; k < s; ++k) {
        uint64_t best_mk = 1;
        int best_t = std::numeric_limits<int>::max();

        uint64_t max_mk = (1ULL << (k + 1)) - 1;
        for (uint64_t mk = 1; mk <= max_mk; mk += 2) {
            auto trial = best_m;
            trial.push_back(mk);

            auto v = compute_v(poly, s, trial);
            int t = t_value(v);

            if (t < best_t) {
                best_t = t;
                best_mk = mk;
                if (t == 0)
                    break; // can't improve on Property A
            }
        }
        best_m.push_back(best_mk);
    }

    auto v = compute_v(poly, s, best_m);
    int t = t_value(v);

    return {0, (uint32_t)s, poly, gf2::encode_a(poly, s), std::move(best_m), t};
}

// ═══════════════════════════════════════════════════════════════════════════
// Level 2: CBC with 2D projection quality against existing dimensions
// ═══════════════════════════════════════════════════════════════════════════

inline SearchResult search_proj_2d(uint64_t poly, int s,
                                   const std::vector<std::vector<uint64_t>>& existing_v,
                                   int n_check = 100) {
    int n_exist = std::min(n_check, (int)existing_v.size());

    std::vector<uint64_t> best_m;
    best_m.reserve(s);
    best_m.push_back(1);

    for (int k = 1; k < s; ++k) {
        uint64_t best_mk = 1;
        int64_t best_score = std::numeric_limits<int64_t>::max();

        uint64_t max_mk = (1ULL << (k + 1)) - 1;
        for (uint64_t mk = 1; mk <= max_mk; mk += 2) {
            auto trial = best_m;
            trial.push_back(mk);

            auto v = compute_v(poly, s, trial);
            int t = t_value(v);

            // Projection deficit sum (lower is better)
            int64_t proj_deficit = 0;
            for (int j = 0; j < n_exist; ++j) {
                int res = resolution_2d(existing_v[j], v);
                proj_deficit += (W - res);
            }

            // Combined score: t-value dominates, projection breaks ties
            int64_t score = (int64_t)t * 1000000LL + proj_deficit;

            if (score < best_score) {
                best_score = score;
                best_mk = mk;
            }
        }
        best_m.push_back(best_mk);
    }

    auto v = compute_v(poly, s, best_m);
    int t = t_value(v);

    return {0, (uint32_t)s, poly, gf2::encode_a(poly, s), std::move(best_m), t};
}

// ═══════════════════════════════════════════════════════════════════════════
// Batch processing with thread pool
// ═══════════════════════════════════════════════════════════════════════════

struct WorkItem {
    uint32_t dim;
    uint64_t polynomial;
    int degree;
    SearchLevel level;
    uint64_t seed; // for RANDOM mode
};

using ProgressCallback = std::function<void(uint32_t completed, uint32_t total)>;

inline std::vector<SearchResult>
process_batch(const std::vector<WorkItem>& items,
              const std::vector<std::vector<uint64_t>>& existing_v = {}, int n_proj_check = 100,
              int num_threads = 0, ProgressCallback on_progress = nullptr) {
    if (num_threads <= 0)
        num_threads = std::max(1u, std::thread::hardware_concurrency());

    std::vector<SearchResult> results(items.size());
    std::atomic<size_t> next{0};
    std::atomic<uint32_t> done{0};
    uint32_t total = (uint32_t)items.size();

    auto worker = [&]() {
        for (;;) {
            size_t idx = next.fetch_add(1, std::memory_order_relaxed);
            if (idx >= items.size())
                break;

            const auto& wi = items[idx];
            SearchResult sr;

            switch (wi.level) {
                case SearchLevel::RANDOM:
                    sr = search_random(wi.polynomial, wi.degree, wi.seed);
                    break;
                case SearchLevel::PROP_A:
                    sr = search_property_a(wi.polynomial, wi.degree);
                    break;
                case SearchLevel::PROJ_2D:
                    sr = search_proj_2d(wi.polynomial, wi.degree, existing_v, n_proj_check);
                    break;
            }

            sr.dim = wi.dim;
            results[idx] = std::move(sr);

            uint32_t d = done.fetch_add(1, std::memory_order_relaxed) + 1;
            if (on_progress && (d % 100 == 0 || d == total))
                on_progress(d, total);
        }
    };

    std::vector<std::thread> threads;
    threads.reserve(num_threads);
    for (int i = 0; i < num_threads; ++i)
        threads.emplace_back(worker);
    for (auto& t : threads)
        t.join();

    return results;
}

// ═══════════════════════════════════════════════════════════════════════════
// Work item serialization (for container dispatch)
// Simple text format: one work item per line
//   dim degree polynomial_hex level seed
// ═══════════════════════════════════════════════════════════════════════════

inline std::string serialize_work(const std::vector<WorkItem>& items) {
    std::string out;
    out.reserve(items.size() * 40);
    for (auto& wi : items) {
        char buf[128];
        snprintf(buf, sizeof(buf), "%u %d 0x%llx %d %llu\n", wi.dim, wi.degree,
                 (unsigned long long)wi.polynomial, (int)wi.level, (unsigned long long)wi.seed);
        out += buf;
    }
    return out;
}

inline std::vector<WorkItem> deserialize_work(const std::string& data) {
    std::vector<WorkItem> items;
    std::istringstream iss(data);
    std::string line;
    while (std::getline(iss, line)) {
        if (line.empty())
            continue;
        WorkItem wi{};
        int level;
        unsigned long long poly, seed;
        if (sscanf(line.c_str(), "%u %d 0x%llx %d %llu", &wi.dim, &wi.degree, &poly, &level,
                   &seed) == 5) {
            wi.polynomial = poly;
            wi.level = (SearchLevel)level;
            wi.seed = seed;
            items.push_back(wi);
        }
    }
    return items;
}

inline std::string serialize_results(const std::vector<SearchResult>& results) {
    std::string out;
    for (auto& sr : results) {
        char buf[64];
        snprintf(buf, sizeof(buf), "%u\t%u\t%llu", sr.dim, sr.degree,
                 (unsigned long long)sr.a_encoded);
        out += buf;
        for (auto mi : sr.m) {
            snprintf(buf, sizeof(buf), "\t%llu", (unsigned long long)mi);
            out += buf;
        }
        out += "\n";
    }
    return out;
}

} // namespace sobol
} // namespace mc
