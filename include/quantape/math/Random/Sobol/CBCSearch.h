#pragma once
/**
 * @file CBCSearch.h
 * @brief Component-by-component search for good Sobol initial direction numbers.
 *
 * Implements the search criterion of
 *   S. Joe, F. Y. Kuo, "Constructing Sobol sequences with better two-dimensional
 *   projections", SIAM J. Sci. Comput. 30 (2008) 2635-2654,
 * i.e. minimise D(q)(d; mmin, mmax) over candidate direction-number sets of the
 * new dimension, with weights 0.9999^{j-1} that favour projections onto earlier
 * dimensions. Unlike the previous implementation, candidates are always fully
 * specified before evaluation (no uninitialised prefix reads).
 *
 * Search levels:
 *   RANDOM    random valid m_i (valid direction numbers; no criterion).
 *   WINDOWED  criterion against the last `criterion.window` dimensions.
 *   FULL      criterion against all previous dimensions (Joe-Kuo style).
 *
 * Property A (det(V_d) == 1 mod 2) can be enforced incrementally for dimensions
 * d <= 1111, matching Joe-Kuo; it is irrelevant for later extension dimensions.
 *
 * Candidate evaluation is parallel over a thread pool; dimensions are processed
 * sequentially (CBC needs the already-chosen prefix). Results are deterministic
 * for a fixed seed: ties are broken by candidate index.
 */
#include <algorithm>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <functional>
#include <limits>
#include <mutex>
#include <thread>
#include <utility>
#include <vector>

#include "SobolQuality.h"

namespace quantape::math::mc {
namespace sobol {

enum class SearchLevel { RANDOM = 0, WINDOWED = 1, FULL = 2 };

struct SearchResult {
    std::uint32_t dim = 0;
    std::uint32_t degree = 0;
    std::uint64_t polynomial = 0;
    std::uint64_t a_encoded = 0;
    std::vector<std::uint64_t> m;
    double score = -1.0; ///< D(q) criterion value (lower is better); -1 for RANDOM
};

struct SearchContext {
    /// Index i holds dimension i+1 (previous[0] is the identity of dimension 1).
    std::vector<DirectionMatrix> previous;
    /// weights[i] = criterion.weightBase^i (kept in step with `previous`).
    std::vector<double> weights;
    CriterionOptions criterion;
    /// Candidates per dimension; <= 0 selects max(64, 2000000 / dim) as in the paper.
    int candidates = 0;
    /// Enforce Property A for dimensions <= 1111 (sequential state below).
    bool enforcePropertyA = false;
    PropertyAChecker propertyA{0};
};

namespace detail {

inline std::uint64_t splitmix64(std::uint64_t& state) {
    state += 0x9E3779B97F4A7C15ULL;
    std::uint64_t z = state;
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    return z ^ (z >> 31);
}

inline std::uint64_t splitmix64At(std::uint64_t seed) {
    return splitmix64(seed);
}

/// Random valid initial direction numbers: m_1 = 1, m_k odd and < 2^k.
inline std::vector<std::uint64_t> randomDirectionSet(std::uint64_t& state, int s) {
    std::vector<std::uint64_t> m(static_cast<std::size_t>(s), 1);
    for (int k = 2; k <= s; ++k) {
        const std::uint64_t span = 1ULL << (k - 1);
        m[static_cast<std::size_t>(k - 1)] = ((splitmix64(state) % span) << 1) | 1;
    }
    return m;
}

/// All valid initial direction sets for small degrees (2^{s(s-1)/2} sets).
inline std::vector<std::vector<std::uint64_t>> exhaustiveDirectionSets(int s) {
    std::vector<std::vector<std::uint64_t>> sets;
    std::vector<std::uint64_t> m(static_cast<std::size_t>(s), 1);
    const auto recurse = [&](auto&& self, int k) -> void {
        if (k > s) {
            sets.push_back(m);
            return;
        }
        const std::uint64_t top = 1ULL << k;
        for (std::uint64_t v = 1; v < top; v += 2) {
            m[static_cast<std::size_t>(k - 1)] = v;
            self(self, k + 1);
        }
    };
    recurse(recurse, 2);
    return sets;
}

template <typename F>
void parallelFor(std::size_t count, int numThreads, const F& fn) {
    if (numThreads <= 1 || count <= 1) {
        for (std::size_t i = 0; i < count; ++i) {
            fn(i);
        }
        return;
    }
    const std::size_t nThreads = std::min<std::size_t>(static_cast<std::size_t>(numThreads), count);
    std::atomic<std::size_t> next{0};
    std::exception_ptr failure;
    std::mutex failureMutex;
    const auto worker = [&]() {
        try {
            for (;;) {
                const std::size_t i = next.fetch_add(1, std::memory_order_relaxed);
                if (i >= count) {
                    return;
                }
                fn(i);
            }
        } catch (...) {
            const std::lock_guard<std::mutex> lock(failureMutex);
            if (!failure) {
                failure = std::current_exception();
            }
        }
    };
    std::vector<std::thread> threads;
    threads.reserve(nThreads - 1);
    for (std::size_t t = 0; t + 1 < nThreads; ++t) {
        threads.emplace_back(worker);
    }
    worker();
    for (std::thread& t : threads) {
        t.join();
    }
    if (failure) {
        std::rethrow_exception(failure);
    }
}

} // namespace detail

// ═══════════════════════════════════════════════════════════════════════════
// Workers
// ═══════════════════════════════════════════════════════════════════════════

/// Level 0: random valid direction numbers.
inline SearchResult searchRandom(std::uint64_t polynomial, int s, std::uint64_t seed) {
    std::uint64_t state = seed;
    SearchResult result;
    result.degree = static_cast<std::uint32_t>(s);
    result.polynomial = polynomial;
    result.a_encoded = gf2::encode_a(polynomial, s);
    result.m = detail::randomDirectionSet(state, s);
    result.score = -1.0;
    return result;
}

/// Levels 1/2: minimise D(q) over candidate sets of the new dimension.
inline SearchResult searchCbc(std::uint64_t polynomial, int s, std::uint64_t seed,
                              std::uint32_t dim, SearchContext& context, int numThreads,
                              SearchLevel level) {
    SearchResult result;
    result.degree = static_cast<std::uint32_t>(s);
    result.polynomial = polynomial;
    result.a_encoded = gf2::encode_a(polynomial, s);

    const CriterionOptions options =
        (level == SearchLevel::FULL)
            ? CriterionOptions{0, context.criterion.mMin, context.criterion.mMax,
                               context.criterion.weightBase, context.criterion.exponent}
            : context.criterion;

    std::vector<std::vector<std::uint64_t>> candidates;
    if (s <= 6) {
        candidates = detail::exhaustiveDirectionSets(s);
    } else {
        int count = context.candidates;
        if (count <= 0) {
            count = std::max(64, static_cast<int>(2000000 / std::max<std::uint32_t>(dim, 1)));
        }
        candidates.reserve(static_cast<std::size_t>(count));
        std::uint64_t state = seed;
        for (int i = 0; i < count; ++i) {
            candidates.push_back(detail::randomDirectionSet(state, s));
        }
    }

    if (context.weights.size() != context.previous.size()) {
        context.weights.resize(context.previous.size());
        for (std::size_t i = 0; i < context.weights.size(); ++i) {
            context.weights[i] = std::pow(context.criterion.weightBase, static_cast<double>(i));
        }
    }

    std::vector<double> scores(candidates.size(), kNoScore);
    const auto evaluate = [&](std::size_t i, double abortAbove) {
        const auto& m = candidates[i];
        if (context.enforcePropertyA) {
            Entry candidateEntry{};
            candidateEntry.dim = dim;
            candidateEntry.s = static_cast<std::uint32_t>(s);
            candidateEntry.a = gf2::encode_a(polynomial, s);
            candidateEntry.m = m;
            if (!context.propertyA.accepts(candidateEntry)) {
                return;
            }
        }
        const DirectionMatrix dm = directionMatrixFor(polynomial, s, m);
        scores[i] = criterionD(dm, context.previous, context.weights, options, abortAbove);
    };

    // Seed pass establishes a deterministic pruning threshold for the parallel
    // pass (candidates pruned against it keep an infinite score; ties between
    // the remaining candidates are broken by candidate index).
    double pruneThreshold = kNoScore;
    const std::size_t seedCount = std::min<std::size_t>(16, candidates.size());
    for (std::size_t i = 0; i < seedCount; ++i) {
        evaluate(i, kNoScore);
        pruneThreshold = std::min(pruneThreshold, scores[i]);
    }
    detail::parallelFor(candidates.size() - seedCount, numThreads,
                        [&](std::size_t k) { evaluate(k + seedCount, pruneThreshold); });

    std::size_t best = candidates.size();
    for (std::size_t i = 0; i < candidates.size(); ++i) {
        if (scores[i] < kNoScore && (best == candidates.size() || scores[i] < scores[best])) {
            best = i;
        }
    }
    if (best == candidates.size()) {
        // No candidate satisfied Property A; fall back to a random set and flag it.
        std::uint64_t state = seed;
        result.m = detail::randomDirectionSet(state, s);
        result.score = kNoScore;
        return result;
    }
    result.m = candidates[best];
    result.score = scores[best];
    return result;
}

// ═══════════════════════════════════════════════════════════════════════════
// Batch processing
// ═══════════════════════════════════════════════════════════════════════════

struct WorkItem {
    std::uint32_t dim = 0;
    std::uint64_t polynomial = 0;
    int degree = 0;
    SearchLevel level = SearchLevel::RANDOM;
    std::uint64_t seed = 0;
};

using ProgressCallback = std::function<void(std::uint32_t completed, std::uint32_t total)>;
/// Called (CBC mode, sequentially in dimension order) once a result is final.
using ResultCallback = std::function<void(std::size_t index, const SearchResult& result)>;

inline std::vector<SearchResult> process_batch(const std::vector<WorkItem>& items,
                                               SearchContext& context, int num_threads = 0,
                                               ProgressCallback on_progress = nullptr,
                                               ResultCallback on_result = nullptr) {
    if (num_threads <= 0) {
        num_threads = static_cast<int>(std::max(1u, std::thread::hardware_concurrency()));
    }
    std::vector<SearchResult> results(items.size());
    const auto total = static_cast<std::uint32_t>(items.size());

    const bool anyCbc = std::any_of(items.begin(), items.end(), [](const WorkItem& w) {
        return w.level != SearchLevel::RANDOM;
    });

    if (!anyCbc) {
        // Random mode: items are independent.
        std::atomic<std::size_t> next{0};
        std::atomic<std::uint32_t> done{0};
        detail::parallelFor(items.size(), num_threads, [&](std::size_t idx) {
            const auto& wi = items[idx];
            SearchResult sr = searchRandom(wi.polynomial, wi.degree, wi.seed);
            sr.dim = wi.dim;
            results[idx] = std::move(sr);
            if (on_progress) {
                const std::uint32_t d = done.fetch_add(1, std::memory_order_relaxed) + 1;
                if (d % 100 == 0 || d == total) {
                    on_progress(d, total);
                }
            }
        });
        return results;
    }

    // CBC: dimensions are sequential; candidates of each dimension run in parallel.
    for (std::size_t idx = 0; idx < items.size(); ++idx) {
        const auto& wi = items[idx];
        SearchResult sr;
        if (wi.level == SearchLevel::RANDOM) {
            sr = searchRandom(wi.polynomial, wi.degree, wi.seed);
        } else {
            sr = searchCbc(wi.polynomial, wi.degree, wi.seed, wi.dim, context, num_threads,
                           wi.level);
            // Extend the prefix for the next dimension.
            Entry chosen{};
            chosen.dim = wi.dim;
            chosen.s = static_cast<std::uint32_t>(wi.degree);
            chosen.a = sr.a_encoded;
            chosen.m = sr.m;
            context.previous.push_back(directionMatrix(chosen));
            context.weights.push_back(std::pow(context.criterion.weightBase,
                                               static_cast<double>(context.weights.size())));
            if (context.enforcePropertyA && wi.dim <= 1111) {
                context.propertyA.add(chosen);
            }
        }
        sr.dim = wi.dim;
        results[idx] = std::move(sr);
        if (on_result) {
            on_result(idx, results[idx]);
        }
        if (on_progress) {
            on_progress(static_cast<std::uint32_t>(idx + 1), total);
        }
    }
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
    for (const auto& wi : items) {
        char buf[128];
        std::snprintf(buf, sizeof(buf), "%u %d 0x%llx %d %llu\n", wi.dim, wi.degree,
                      (unsigned long long)wi.polynomial, (int)wi.level,
                      (unsigned long long)wi.seed);
        out += buf;
    }
    return out;
}

inline std::vector<WorkItem> deserialize_work(const std::string& data) {
    std::vector<WorkItem> items;
    std::istringstream iss(data);
    std::string line;
    while (std::getline(iss, line)) {
        if (line.empty()) {
            continue;
        }
        WorkItem wi{};
        int level = 0;
        unsigned long long poly = 0;
        unsigned long long seed = 0;
        if (std::sscanf(line.c_str(), "%u %d 0x%llx %d %llu", &wi.dim, &wi.degree, &poly, &level,
                        &seed) == 5) {
            wi.polynomial = poly;
            wi.level = static_cast<SearchLevel>(level);
            wi.seed = seed;
            items.push_back(wi);
        }
    }
    return items;
}

inline std::string serialize_results(const std::vector<SearchResult>& results) {
    std::string out;
    for (const auto& sr : results) {
        char buf[64];
        std::snprintf(buf, sizeof(buf), "%u\t%u\t%llu", sr.dim, sr.degree,
                      (unsigned long long)sr.a_encoded);
        out += buf;
        for (auto mi : sr.m) {
            std::snprintf(buf, sizeof(buf), "\t%llu", (unsigned long long)mi);
            out += buf;
        }
        out += "\n";
    }
    return out;
}

} // namespace sobol
} // namespace quantape::math::mc
