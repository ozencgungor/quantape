#ifndef QUANTAPE_MATH_RANDOM_SOBOL_SOBOLGENERATOR_H
#define QUANTAPE_MATH_RANDOM_SOBOL_SOBOLGENERATOR_H

#include "quantape/math/Random/InverseNormal.h"
#include "quantape/math/Random/Sobol/DirectionNumbers.h"
#include "quantape/math/Random/Sobol/GF2.h"

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#if defined(__unix__) || defined(__APPLE__)
#define QUANTAPE_SOBOL_HAS_MMAP 1
#include <fcntl.h>
#include <sys/mman.h>
#include <unistd.h>
#else
#define QUANTAPE_SOBOL_HAS_MMAP 0
#endif

#ifndef QUANTAPE_SOBOL_DEFAULT_TABLE_PATH
#define QUANTAPE_SOBOL_DEFAULT_TABLE_PATH ""
#endif

namespace quantape::math::mc {
namespace sobol {
/**
 * @file SobolGenerator.h
 * @brief Sobol sequence generator over a Joe-Kuo direction-number table.
 *
 * ## RNG requirements
 *
 * - **Seed reproducibility**: every draw is a pure function of
 *   (point, dimension, shiftSeed); separate generator instances with the same
 *   configuration produce bitwise-identical values. No hidden state.
 * - **Jump-ahead**: any point index is addressable directly in O(popcount)
 *   word XORs (`uniformBits(point, dim)`); skipping to point p costs the same
 *   as reading it. Blocked/parallel evaluation is therefore bitwise
 *   identical to sequential evaluation (jump == index, no cursor to advance).
 * - **Stream splitting**: `replica(seed)` (engine source) or the explicit
 *   `shiftSeed` overloads give independent, structure-preserving streams
 *   without copying the table; `shiftFor(dim, seed)` derives the per-dimension
 *   shift for arbitrary seeds.
 * - **Thread safety**: instances are immutable and const; the tables are
 *   shared read-only (one heap copy per process via `sharedFromFile`, or a
 *   zero-copy read-only `mmap` for the compiled-in binary asset), so N engine
 *   threads touch the same physical pages.
 *
 * ## Storage modes
 *
 * 1. Text (`fromFile`): parses a Joe-Kuo table, builds `maxBits` direction
 *    words per dimension. `maxBits <= 32` stores 32-bit words (~8 MiB for
 *    65,536 dims), `maxBits = 64` full precision (~32 MiB). `maxDimension`
 *    restricts the build to the dimensions a layout actually uses (~0.1 MiB
 *    for a 252-step x 4-factor layout).
 * 2. Binary asset (`fromBinary`, `fromDefaultTable`): a `.qsb` file produced
 *    by `tools/sobol_to_binary.cpp` at build time and selected at compile time
 *    via `QUANTAPE_SOBOL_TABLE_PATH` (CMake `-DQUANTAPE_SOBOL_TABLE=...`).
 *    The file is memory-mapped read-only: zero heap, zero parse, pages shared
 *    across threads and processes.
 */
struct SobolOptions {
    std::uint64_t shiftSeed = 0; ///< 0 = unshifted (use with pointOffset > 0)
    std::uint64_t pointOffset = 1;
    bool validate = true;
    std::uint32_t maxBits = 32;     ///< direction words per dimension (1..64)
    std::uint32_t maxDimension = 0; ///< 0 = all table dimensions
    bool keepEntries = false;       ///< retain the parsed table (text mode)
};

namespace detail {

/// Read-only file mapping (mmap when available, heap copy otherwise).
class MappedFile {
public:
    static std::shared_ptr<const MappedFile> open(const std::string& path) {
        std::FILE* file = std::fopen(path.c_str(), "rb");
        if (file == nullptr) {
            throw std::runtime_error("SobolGenerator: cannot open " + path);
        }
        std::fseek(file, 0, SEEK_END);
        const long size = std::ftell(file);
        std::fseek(file, 0, SEEK_SET);
        if (size <= 0) {
            std::fclose(file);
            throw std::runtime_error("SobolGenerator: empty file " + path);
        }
#if QUANTAPE_SOBOL_HAS_MMAP
        const int fd = fileno(file);
        void* base = mmap(nullptr, static_cast<std::size_t>(size), PROT_READ, MAP_PRIVATE, fd, 0);
        std::fclose(file);
        if (base == MAP_FAILED) {
            throw std::runtime_error("SobolGenerator: mmap failed for " + path);
        }
        return std::shared_ptr<const MappedFile>(
            new MappedFile(base, static_cast<std::size_t>(size)));
#else
        std::vector<std::uint8_t> buffer(static_cast<std::size_t>(size));
        const std::size_t got = std::fread(buffer.data(), 1, buffer.size(), file);
        std::fclose(file);
        buffer.resize(got);
        MappedFile* raw = new MappedFile(nullptr, buffer.size());
        raw->m_heap = std::move(buffer);
        return std::shared_ptr<const MappedFile>(raw);
#endif
    }

    ~MappedFile() {
#if QUANTAPE_SOBOL_HAS_MMAP
        if (m_base != nullptr) {
            munmap(m_base, m_size);
        }
#endif
    }

    MappedFile(const MappedFile&) = delete;
    MappedFile& operator=(const MappedFile&) = delete;

    const std::uint8_t* data() const {
#if QUANTAPE_SOBOL_HAS_MMAP
        return static_cast<const std::uint8_t*>(m_base);
#else
        return m_heap.data();
#endif
    }
    std::size_t size() const { return m_size; }

private:
    MappedFile(void* base, std::size_t size) : m_base(base), m_size(size) {}
    void* m_base = nullptr;
    std::size_t m_size = 0;
#if !QUANTAPE_SOBOL_HAS_MMAP
    std::vector<std::uint8_t> m_heap;
#endif
};

/// Binary asset layout: "QSB1" | u32 dims | u32 maxBits | u32 wordBytes | u32 pad
/// then (dims-1) x maxBits words, row-major by dimension.
inline constexpr std::size_t kBinaryHeader = 20;
inline constexpr char kBinaryMagic[5] = "QSB1";

} // namespace detail

class SobolGenerator {
public:
    explicit SobolGenerator(std::vector<Entry> entries, SobolOptions options = {})
        : m_options(options), m_entries(std::move(entries)) {
        if (m_entries.empty()) {
            throw std::invalid_argument("SobolGenerator: empty direction-number table");
        }
        if (m_options.maxBits < 1 || m_options.maxBits > 64) {
            throw std::invalid_argument("SobolGenerator: maxBits must be in 1..64");
        }
        m_dimensionCount = m_entries.back().dim;
        if (m_options.validate) {
            validate();
        }
        m_preparedDimension = m_dimensionCount;
        if (m_options.maxDimension != 0) {
            m_preparedDimension = std::min(m_dimensionCount, m_options.maxDimension);
        }
        if (m_preparedDimension < 1) {
            throw std::invalid_argument("SobolGenerator: nothing prepared");
        }
        m_use32 = (m_options.maxBits <= 32);
        build();
        if (!m_options.keepEntries) {
            m_entries.clear();
            m_entries.shrink_to_fit();
        }
    }

    /// Text table (parses and builds words; private heap).
    static SobolGenerator fromFile(const std::string& path, SobolOptions options = {}) {
        return SobolGenerator(load_joe_kuo(path), options);
    }

    /// Binary asset (zero-copy read-only mapping; no parsing).
    static SobolGenerator fromBinary(const std::string& path, SobolOptions options = {}) {
        return SobolGenerator(detail::MappedFile::open(path), options);
    }

    /// Compile-time selected table (CMake QUANTAPE_SOBOL_TABLE); binary if the
    /// configured file ends in .qsb, text otherwise.
    static SobolGenerator fromDefaultTable(SobolOptions options = {}) {
        const std::string path = defaultTablePath();
        if (path.empty()) {
            throw std::runtime_error("SobolGenerator: no compile-time table configured "
                                     "(set -DQUANTAPE_SOBOL_TABLE=<file> when configuring CMake)");
        }
        if (path.size() >= 4 && path.compare(path.size() - 4, 4, ".qsb") == 0) {
            return fromBinary(path, options);
        }
        return fromFile(path, options);
    }

    /// One instance per (path, options) per process: all threads/engine
    /// instances share the same immutable tables.
    static std::shared_ptr<const SobolGenerator> sharedFromFile(const std::string& path,
                                                                SobolOptions options = {}) {
        return sharedLookup("text:" + path, options, [&] { return fromFile(path, options); });
    }

    static std::shared_ptr<const SobolGenerator> sharedFromBinary(const std::string& path,
                                                                  SobolOptions options = {}) {
        return sharedLookup("bin:" + path, options, [&] { return fromBinary(path, options); });
    }

    static std::shared_ptr<const SobolGenerator> sharedFromDefaultTable(SobolOptions options = {}) {
        const std::string path = defaultTablePath();
        if (path.empty()) {
            throw std::runtime_error("SobolGenerator: no compile-time table configured");
        }
        if (path.size() >= 4 && path.compare(path.size() - 4, 4, ".qsb") == 0) {
            return sharedFromBinary(path, options);
        }
        return sharedFromFile(path, options);
    }

    static std::string defaultTablePath() { return QUANTAPE_SOBOL_DEFAULT_TABLE_PATH; }

    std::uint32_t dimensionCount() const { return m_dimensionCount; }
    std::uint32_t preparedDimension() const { return m_preparedDimension; }
    std::uint32_t maxBits() const { return m_options.maxBits; }
    std::uint64_t shiftSeed() const { return m_options.shiftSeed; }
    std::uint64_t pointOffset() const { return m_options.pointOffset; }
    std::uint32_t firstDimension() const { return 2; }
    bool mapped() const { return m_mapping != nullptr; }
    const std::vector<Entry>& entries() const { return m_entries; }

    /// 64-bit uniform bits with the configured shift.
    std::uint64_t uniformBits(std::uint64_t point, std::uint32_t dim) const {
        return uniformBits(point, dim, m_options.shiftSeed);
    }

    /// 64-bit uniform bits with an explicit (replica) shift seed.
    std::uint64_t uniformBits(std::uint64_t point, std::uint32_t dim,
                              std::uint64_t shiftSeed) const {
        const std::uint64_t absolute = point + m_options.pointOffset;
        if (dim < 1 || dim > m_preparedDimension) {
            throw std::out_of_range("SobolGenerator: dimension out of range");
        }
        if (m_options.maxBits < 64 && absolute >= (1ULL << m_options.maxBits)) {
            throw std::out_of_range("SobolGenerator: point exceeds maxBits");
        }
        return bitsImpl(absolute, dim, shiftSeed);
    }

    double uniform(std::uint64_t point, std::uint32_t dim) const {
        return uniform(point, dim, m_options.shiftSeed);
    }

    double uniform(std::uint64_t point, std::uint32_t dim, std::uint64_t shiftSeed) const {
        return static_cast<double>(uniformBits(point, dim, shiftSeed) >> 11) * 0x1.0p-53;
    }

    double normal(std::uint64_t point, std::uint32_t dim) const {
        return normal(point, dim, m_options.shiftSeed);
    }

    double normal(std::uint64_t point, std::uint32_t dim, std::uint64_t shiftSeed) const {
        return normalFromBits(uniformBits(point, dim, shiftSeed));
    }

    /// Inverse-normal of a raw 64-bit Sobol word (midpoint mapped).
    static double normalFromBits(std::uint64_t bits) {
        return inverseNormal((static_cast<double>(bits) + 0.5) * 0x1.0p-64);
    }

    /// Per-dimension digital shift derived from a seed (pure function).
    static std::uint64_t shiftFor(std::uint32_t dim, std::uint64_t shiftSeed) {
        constexpr std::uint64_t kSalt = 0x9E3779B97F4A7C15ULL;
        return mix64(shiftSeed ^ (static_cast<std::uint64_t>(dim) * kSalt));
    }

    /// Shift used by the configured seed (precomputed; explicit seeds hash).
    std::uint64_t configuredShift(std::uint32_t dim) const {
        return m_options.shiftSeed ? m_shifts[dim] : 0;
    }

private:
    SobolGenerator(std::shared_ptr<const detail::MappedFile> mapping, SobolOptions options)
        : m_options(options), m_mapping(std::move(mapping)) {
        const std::uint8_t* data = m_mapping->data();
        if (m_mapping->size() < detail::kBinaryHeader ||
            std::memcmp(data, detail::kBinaryMagic, 4) != 0) {
            throw std::invalid_argument("SobolGenerator: not a QSB1 binary asset");
        }
        std::uint32_t dims = 0, wordBytes = 0;
        std::memcpy(&dims, data + 4, 4);
        std::memcpy(&m_assetBits, data + 8, 4);
        std::memcpy(&wordBytes, data + 12, 4);
        m_dimensionCount = dims;
        m_preparedDimension = dims;
        if (m_options.maxDimension != 0) {
            m_preparedDimension = std::min(dims, m_options.maxDimension);
        }
        if (m_options.maxBits == 0) {
            m_options.maxBits = m_assetBits;
        }
        if (m_assetBits < 1 || m_assetBits > 64 || m_options.maxBits > m_assetBits ||
            (wordBytes != 4 && wordBytes != 8)) {
            throw std::invalid_argument("SobolGenerator: bad binary asset header");
        }
        const std::size_t needed =
            detail::kBinaryHeader + static_cast<std::size_t>(dims - 1) * m_assetBits * wordBytes;
        if (m_mapping->size() < needed) {
            throw std::invalid_argument("SobolGenerator: truncated binary asset");
        }
        m_use32 = (wordBytes == 4);
        const std::uint8_t* words = data + detail::kBinaryHeader;
        m_ext32 = m_use32 ? reinterpret_cast<const std::uint32_t*>(words) : nullptr;
        m_ext64 = m_use32 ? nullptr : reinterpret_cast<const std::uint64_t*>(words);
        buildFirstWords();
        buildShifts();
    }

    static std::shared_ptr<const SobolGenerator>
    sharedLookup(const std::string& key, const SobolOptions& options,
                 const std::function<SobolGenerator()>& create) {
        static std::mutex mutex;
        static std::unordered_map<std::string, std::weak_ptr<const SobolGenerator>> cache;
        std::string full =
            key + "|" + std::to_string(options.shiftSeed) + "|" +
            std::to_string(options.pointOffset) + "|" + std::to_string(options.maxBits) + "|" +
            std::to_string(options.maxDimension) + "|" + std::to_string(options.keepEntries);
        const std::lock_guard<std::mutex> lock(mutex);
        auto it = cache.find(full);
        if (it != cache.end()) [[likely]] {
            if (auto existing = it->second.lock()) {
                return existing;
            }
        }
        auto created = std::make_shared<const SobolGenerator>(create());
        cache[full] = created;
        return created;
    }

    void validate() const {
        for (std::size_t i = 0; i < m_entries.size(); ++i) {
            const Entry& e = m_entries[i];
            if (e.dim != i + 2) {
                throw std::invalid_argument("SobolGenerator: dimensions must be contiguous from 2");
            }
            if (e.m.size() != e.s || e.s == 0 || e.s > 64) {
                throw std::invalid_argument("SobolGenerator: invalid degree / direction numbers");
            }
            for (std::uint32_t k = 1; k <= e.s; ++k) {
                const std::uint64_t mk = e.m[k - 1];
                if ((mk & 1u) != 0 && mk < (1ULL << k)) {
                    continue;
                }
                throw std::invalid_argument("SobolGenerator: direction number out of range");
            }
        }
    }

    void buildFirstWords() {
        const std::size_t bits = m_options.maxBits;
        if (m_use32) {
            m_first32.assign(bits, 0);
            for (std::size_t k = 1; k <= bits; ++k) {
                m_first32[k - 1] = 1u << (32 - k);
            }
        } else {
            m_first64.assign(bits, 0);
            for (std::size_t k = 1; k <= bits; ++k) {
                m_first64[k - 1] = 1ULL << (64 - k);
            }
        }
    }

    void buildShifts() {
        if (m_options.shiftSeed == 0) {
            return;
        }
        m_shifts.assign(static_cast<std::size_t>(m_preparedDimension) + 1, 0);
        for (std::uint32_t d = 1; d <= m_preparedDimension; ++d) {
            m_shifts[d] = shiftFor(d, m_options.shiftSeed);
        }
    }

    void build() {
        buildFirstWords();
        const std::size_t dims = m_preparedDimension;
        const std::size_t bits = m_options.maxBits;
        if (m_use32) {
            m_words32.assign((dims - 1) * bits, 0);
        } else {
            m_words64.assign((dims - 1) * bits, 0);
        }
        for (std::size_t i = 0; i + 2 <= dims; ++i) {
            const Entry& e = m_entries[i];
            const std::uint64_t poly = gf2::decode_poly(static_cast<int>(e.s), e.a);
            std::uint64_t m[65] = {0};
            for (std::uint32_t k = 1; k <= e.s && k <= bits; ++k) {
                m[k] = e.m[k - 1];
            }
            for (std::size_t k = e.s + 1; k <= bits; ++k) {
                std::uint64_t value = (m[k - e.s] << e.s) ^ m[k - e.s];
                for (std::uint32_t j = 1; j < e.s; ++j) {
                    if ((poly >> (e.s - j)) & 1) {
                        value ^= (m[k - j] << j);
                    }
                }
                m[k] = value;
            }
            if (m_use32) {
                std::uint32_t* w = m_words32.data() + i * bits;
                for (std::size_t k = 1; k <= bits; ++k) {
                    w[k - 1] = static_cast<std::uint32_t>(m[k]) << (32 - static_cast<int>(k));
                }
            } else {
                std::uint64_t* w = m_words64.data() + i * bits;
                for (std::size_t k = 1; k <= bits; ++k) {
                    w[k - 1] = m[k] << (64 - static_cast<int>(k));
                }
            }
        }
        buildShifts();
    }

    /// Hot path: XOR the direction words selected by the bits of `p`.
    /// `p` is the absolute point index (pointOffset already applied).
    std::uint64_t bitsImpl(std::uint64_t p, std::uint32_t dim, std::uint64_t shiftSeed) const {
        std::uint64_t x = 0;
        const std::uint32_t bits = m_options.maxBits;
        if (dim == 1) {
            if (m_use32) {
                const std::uint32_t* w = m_first32.data();
                while (p != 0) {
                    const int bit = __builtin_ctzll(p);
                    x ^= static_cast<std::uint64_t>(w[bit]) << 32;
                    p &= p - 1;
                }
            } else {
                const std::uint64_t* w = m_first64.data();
                while (p != 0) {
                    const int bit = __builtin_ctzll(p);
                    x ^= w[bit];
                    p &= p - 1;
                }
            }
        } else if (m_use32) {
            const std::uint32_t* w =
                m_ext32 ? m_ext32 + static_cast<std::size_t>(dim - 2) * bits
                        : m_words32.data() + static_cast<std::size_t>(dim - 2) * bits;
            while (p != 0) {
                const int bit = __builtin_ctzll(p);
                x ^= static_cast<std::uint64_t>(w[bit]) << 32;
                p &= p - 1;
            }
        } else {
            const std::uint64_t* w =
                m_ext64 ? m_ext64 + static_cast<std::size_t>(dim - 2) * bits
                        : m_words64.data() + static_cast<std::size_t>(dim - 2) * bits;
            while (p != 0) {
                const int bit = __builtin_ctzll(p);
                x ^= w[bit];
                p &= p - 1;
            }
        }
        if (shiftSeed != 0) {
            x ^= (shiftSeed == m_options.shiftSeed) ? m_shifts[dim] : shiftFor(dim, shiftSeed);
        }
        return x;
    }

    static std::uint64_t mix64(std::uint64_t x) {
        x += 0x9E3779B97F4A7C15ULL;
        x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ULL;
        x = (x ^ (x >> 27)) * 0x94D049BB133111EBULL;
        return x ^ (x >> 31);
    }

    SobolOptions m_options;
    std::vector<Entry> m_entries;
    std::shared_ptr<const detail::MappedFile> m_mapping;
    std::uint32_t m_dimensionCount = 1;
    std::uint32_t m_preparedDimension = 1;
    std::uint32_t m_assetBits = 0;
    bool m_use32 = true;
    const std::uint32_t* m_ext32 = nullptr;
    const std::uint64_t* m_ext64 = nullptr;
    std::vector<std::uint32_t> m_words32;
    std::vector<std::uint64_t> m_words64;
    std::vector<std::uint32_t> m_first32;
    std::vector<std::uint64_t> m_first64;
    std::vector<std::uint64_t> m_shifts;
};

} // namespace sobol
} // namespace quantape::math::mc

#endif // QUANTAPE_MATH_RANDOM_SOBOL_SOBOLGENERATOR_H
