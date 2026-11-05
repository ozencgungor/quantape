#ifndef LIGHTAPE_OPS_DERIVATIVE_CACHE_H
#define LIGHTAPE_OPS_DERIVATIVE_CACHE_H

#include <cstddef>
#include <cstdint>

namespace lightape {
/**
 * @file DerivativeCache.h
 * @brief Fixed-capacity per-thread ring cache for precomputed block derivatives
 *
 * Heston-style callback use: a block gradient/Hessian is expensive, so compute
 * it once per primal point, `insert(point, payload)`, then have the External
 * callbacks `find(point)` and read the cached numbers (values only — never
 * tape slots: a cached node index aliases a different scalar on the next
 * replay). When the primal parameters move (new outer iteration, new
 * calibration state), call `bumpEpoch()`: every entry recorded in the previous
 * epoch is invalidated.
 *
 * The intended instance is `threadLocal()`, whose `static thread_local`
 * backing store gives each thread its own cache (no contention in parallel
 * path loops). `find` is a linear scan — `Capacity` is small (default 32), so
 * the constant beats a hash map and keeps the object allocation-free. Entries
 * fill the ring in insertion order; once full, `insert` overwrites slots
 * round-robin. Re-inserting a live key refreshes its value in place.
 *
 * `Key` needs `operator==`; `Key` and `Value` are stored by value and must be
 * default-constructible and copy-assignable.
 */
template <typename Key, typename Value, std::size_t Capacity = 32>
class DerivativeCache {
public:
    static_assert(Capacity > 0, "DerivativeCache: Capacity must be at least 1");

    DerivativeCache() = default;

    /// Drop every stored entry (epoch unchanged).
    void clear() {
        m_size = 0;
        m_next = 0;
    }

    /// Advance the epoch and invalidate every entry stored so far.
    void bumpEpoch() {
        ++m_epoch;
        m_size = 0;
        m_next = 0;
    }

    /// Current epoch; entries inserted now stay valid until the next bump.
    std::uint64_t epoch() const noexcept { return m_epoch; }

    /// Pointer to the stored value for `key`, or nullptr on a miss. The
    /// pointer stays valid until the entry is evicted, cleared or bumped.
    const Value* find(const Key& key) const {
        for (std::size_t i = 0; i < m_size; ++i) {
            if (m_entries[i].key == key) {
                return &m_entries[i].value;
            }
        }
        return nullptr;
    }

    /// Insert or refresh `key`; when full, evict the round-robin victim.
    void insert(const Key& key, const Value& value) {
        for (std::size_t i = 0; i < m_size; ++i) {
            if (m_entries[i].key == key) {
                m_entries[i].value = value;
                return;
            }
        }
        std::size_t slot = 0;
        if (m_size < Capacity) {
            slot = m_size++;
        } else {
            slot = evictNext();
        }
        m_entries[slot].key = key;
        m_entries[slot].value = value;
    }

    std::size_t size() const noexcept { return m_size; }

    /// Per-thread cache instance backing the callback pattern above.
    static DerivativeCache& threadLocal() {
        static thread_local DerivativeCache cache;
        return cache;
    }

private:
    std::size_t evictNext() {
        const std::size_t slot = m_next;
        m_next = (m_next + 1) % Capacity;
        return slot;
    }

    struct Entry {
        Key key{};
        Value value{};
    };

    Entry m_entries[Capacity];
    std::size_t m_size = 0;
    std::size_t m_next = 0;
    std::uint64_t m_epoch = 0;
};

} // namespace lightape

#endif // LIGHTAPE_OPS_DERIVATIVE_CACHE_H
