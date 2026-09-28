#ifndef QUANTAPE_MATH_RANDOM_SOBOL_SOBOLGENERATOR_H
#define QUANTAPE_MATH_RANDOM_SOBOL_SOBOLGENERATOR_H

#include "quantape/math/Random/InverseNormal.h"
#include "quantape/math/Random/Sobol/DirectionNumbers.h"
#include "quantape/math/Random/Sobol/GF2.h"

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <map>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
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
    std::uint64_t shiftSeed = 0;    ///< 0 = unshifted (use with pointOffset > 0)
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
        raw->heap_ = std::move(buffer);
        return std::shared_ptr<const MappedFile>(raw);
#endif
    }

    ~MappedFile() {
#if QUANTAPE_SOBOL_HAS_MMAP
        if (base_ != nullptr) {
            munmap(base_, size_);
        }
#endif
    }

    MappedFile(const MappedFile&) = delete;
    MappedFile& operator=(const MappedFile&) = delete;

    const std::uint8_t* data() const {
#if QUANTAPE_SOBOL_HAS_MMAP
        return static_cast<const std::uint8_t*>(base_);
#else
        return heap_.data();
#endif
    }
    std::size_t size() const { return size_; }

private:
    MappedFile(void* base, std::size_t size) : base_(base), size_(size) {}
    void* base_ = nullptr;
    std::size_t size_ = 0;
#if !QUANTAPE_SOBOL_HAS_MMAP
    std::vector<std::uint8_t> heap_;
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
        : options_(options), entries_(std::move(entries)) {
        if (entries_.empty()) {
            throw std::invalid_argument("SobolGenerator: empty direction-number table");
        }
        if (options_.maxBits < 1 || options_.maxBits > 64) {
            throw std::invalid_argument("SobolGenerator: maxBits must be in 1..64");
        }
        dimensionCount_ = entries_.back().dim;
        if (options_.validate) {
            validate();
        }
        preparedDimension_ = dimensionCount_;
        if (options_.maxDimension != 0) {
            preparedDimension_ = std::min(dimensionCount_, options_.maxDimension);
        }
        if (preparedDimension_ < 1) {
            throw std::invalid_argument("SobolGenerator: nothing prepared");
        }
        use32_ = (options_.maxBits <= 32);
        build();
        if (!options_.keepEntries) {
            entries_.clear();
            entries_.shrink_to_fit();
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
            throw std::runtime_error(
                "SobolGenerator: no compile-time table configured "
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

    std::uint32_t dimensionCount() const { return dimensionCount_; }
    std::uint32_t preparedDimension() const { return preparedDimension_; }
    std::uint32_t maxBits() const { return options_.maxBits; }
    std::uint64_t shiftSeed() const { return options_.shiftSeed; }
    std::uint64_t pointOffset() const { return options_.pointOffset; }
    std::uint32_t firstDimension() const { return 2; }
    bool mapped() const { return mapping_ != nullptr; }
    const std::vector<Entry>& entries() const { return entries_; }

    /// 64-bit uniform bits with the configured shift.
    std::uint64_t uniformBits(std::uint64_t point, std::uint32_t dim) const {
        return uniformBits(point, dim, options_.shiftSeed);
    }

    /// 64-bit uniform bits with an explicit (replica) shift seed.
    std::uint64_t uniformBits(std::uint64_t point, std::uint32_t dim,
                              std::uint64_t shiftSeed) const {
        const std::uint64_t absolute = point + options_.pointOffset;
        if (dim < 1 || dim > preparedDimension_) {
            throw std::out_of_range("SobolGenerator: dimension out of range");
        }
        if (options_.maxBits < 64 && absolute >= (1ULL << options_.maxBits)) {
            throw std::out_of_range("SobolGenerator: point exceeds maxBits");
        }
        return bitsImpl(absolute, dim, shiftSeed);
    }

    double uniform(std::uint64_t point, std::uint32_t dim) const {
        return uniform(point, dim, options_.shiftSeed);
    }

    double uniform(std::uint64_t point, std::uint32_t dim, std::uint64_t shiftSeed) const {
        return static_cast<double>(uniformBits(point, dim, shiftSeed) >> 11) * 0x1.0p-53;
    }

    double normal(std::uint64_t point, std::uint32_t dim) const {
        return normal(point, dim, options_.shiftSeed);
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
        return options_.shiftSeed ? shifts_[dim] : 0;
    }

private:
    SobolGenerator(std::shared_ptr<const detail::MappedFile> mapping, SobolOptions options)
        : options_(options), mapping_(std::move(mapping)) {
        const std::uint8_t* data = mapping_->data();
        if (mapping_->size() < detail::kBinaryHeader ||
            std::memcmp(data, detail::kBinaryMagic, 4) != 0) {
            throw std::invalid_argument("SobolGenerator: not a QSB1 binary asset");
        }
        std::uint32_t dims = 0, wordBytes = 0;
        std::memcpy(&dims, data + 4, 4);
        std::memcpy(&assetBits_, data + 8, 4);
        std::memcpy(&wordBytes, data + 12, 4);
        dimensionCount_ = dims;
        preparedDimension_ = dims;
        if (options_.maxDimension != 0) {
            preparedDimension_ = std::min(dims, options_.maxDimension);
        }
        if (options_.maxBits == 0) {
            options_.maxBits = assetBits_;
        }
        if (assetBits_ < 1 || assetBits_ > 64 || options_.maxBits > assetBits_ ||
            (wordBytes != 4 && wordBytes != 8)) {
            throw std::invalid_argument("SobolGenerator: bad binary asset header");
        }
        const std::size_t needed =
            detail::kBinaryHeader + static_cast<std::size_t>(dims - 1) * assetBits_ * wordBytes;
        if (mapping_->size() < needed) {
            throw std::invalid_argument("SobolGenerator: truncated binary asset");
        }
        use32_ = (wordBytes == 4);
        const std::uint8_t* words = data + detail::kBinaryHeader;
        ext32_ = use32_ ? reinterpret_cast<const std::uint32_t*>(words) : nullptr;
        ext64_ = use32_ ? nullptr : reinterpret_cast<const std::uint64_t*>(words);
        buildFirstWords();
        buildShifts();
    }

    static std::shared_ptr<const SobolGenerator>
    sharedLookup(const std::string& key, const SobolOptions& options,
                 const std::function<SobolGenerator()>& create) {
        static std::mutex mutex;
        static std::map<std::string, std::weak_ptr<const SobolGenerator>> cache;
        std::string full = key + "|" + std::to_string(options.shiftSeed) + "|" +
                           std::to_string(options.pointOffset) + "|" +
                           std::to_string(options.maxBits) + "|" +
                           std::to_string(options.maxDimension) + "|" +
                           std::to_string(options.keepEntries);
        const std::lock_guard<std::mutex> lock(mutex);
        auto it = cache.find(full);
        if (it != cache.end()) {
            if (auto existing = it->second.lock()) {
                return existing;
            }
        }
        auto created = std::make_shared<const SobolGenerator>(create());
        cache[full] = created;
        return created;
    }

    void validate() const {
        for (std::size_t i = 0; i < entries_.size(); ++i) {
            const Entry& e = entries_[i];
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
        const std::size_t bits = options_.maxBits;
        if (use32_) {
            first32_.assign(bits, 0);
            for (std::size_t k = 1; k <= bits; ++k) {
                first32_[k - 1] = 1u << (32 - k);
            }
        } else {
            first64_.assign(bits, 0);
            for (std::size_t k = 1; k <= bits; ++k) {
                first64_[k - 1] = 1ULL << (64 - k);
            }
        }
    }

    void buildShifts() {
        if (options_.shiftSeed == 0) {
            return;
        }
        shifts_.assign(static_cast<std::size_t>(preparedDimension_) + 1, 0);
        for (std::uint32_t d = 1; d <= preparedDimension_; ++d) {
            shifts_[d] = shiftFor(d, options_.shiftSeed);
        }
    }

    void build() {
        buildFirstWords();
        const std::size_t dims = preparedDimension_;
        const std::size_t bits = options_.maxBits;
        if (use32_) {
            words32_.assign((dims - 1) * bits, 0);
        } else {
            words64_.assign((dims - 1) * bits, 0);
        }
        for (std::size_t i = 0; i + 2 <= dims; ++i) {
            const Entry& e = entries_[i];
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
            if (use32_) {
                std::uint32_t* w = words32_.data() + i * bits;
                for (std::size_t k = 1; k <= bits; ++k) {
                    w[k - 1] = static_cast<std::uint32_t>(m[k]) << (32 - static_cast<int>(k));
                }
            } else {
                std::uint64_t* w = words64_.data() + i * bits;
                for (std::size_t k = 1; k <= bits; ++k) {
                    w[k - 1] = m[k] << (64 - static_cast<int>(k));
                }
            }
        }
        buildShifts();
    }

    /// Hot path: XOR the direction words selected by the bits of `p`.
    /// `p` is the absolute point index (pointOffset already applied).
    std::uint64_t bitsImpl(std::uint64_t p, std::uint32_t dim,
                           std::uint64_t shiftSeed) const {
        std::uint64_t x = 0;
        const std::uint32_t bits = options_.maxBits;
        if (dim == 1) {
            if (use32_) {
                const std::uint32_t* w = first32_.data();
                while (p != 0) {
                    const int bit = __builtin_ctzll(p);
                    x ^= static_cast<std::uint64_t>(w[bit]) << 32;
                    p &= p - 1;
                }
            } else {
                const std::uint64_t* w = first64_.data();
                while (p != 0) {
                    const int bit = __builtin_ctzll(p);
                    x ^= w[bit];
                    p &= p - 1;
                }
            }
        } else if (use32_) {
            const std::uint32_t* w =
                ext32_ ? ext32_ + static_cast<std::size_t>(dim - 2) * bits
                       : words32_.data() + static_cast<std::size_t>(dim - 2) * bits;
            while (p != 0) {
                const int bit = __builtin_ctzll(p);
                x ^= static_cast<std::uint64_t>(w[bit]) << 32;
                p &= p - 1;
            }
        } else {
            const std::uint64_t* w =
                ext64_ ? ext64_ + static_cast<std::size_t>(dim - 2) * bits
                       : words64_.data() + static_cast<std::size_t>(dim - 2) * bits;
            while (p != 0) {
                const int bit = __builtin_ctzll(p);
                x ^= w[bit];
                p &= p - 1;
            }
        }
        if (shiftSeed != 0) {
            x ^= (shiftSeed == options_.shiftSeed) ? shifts_[dim] : shiftFor(dim, shiftSeed);
        }
        return x;
    }

    static std::uint64_t mix64(std::uint64_t x) {
        x += 0x9E3779B97F4A7C15ULL;
        x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ULL;
        x = (x ^ (x >> 27)) * 0x94D049BB133111EBULL;
        return x ^ (x >> 31);
    }

    SobolOptions options_;
    std::vector<Entry> entries_;
    std::shared_ptr<const detail::MappedFile> mapping_;
    std::uint32_t dimensionCount_ = 1;
    std::uint32_t preparedDimension_ = 1;
    std::uint32_t assetBits_ = 0;
    bool use32_ = true;
    const std::uint32_t* ext32_ = nullptr;
    const std::uint64_t* ext64_ = nullptr;
    std::vector<std::uint32_t> words32_;
    std::vector<std::uint64_t> words64_;
    std::vector<std::uint32_t> first32_;
    std::vector<std::uint64_t> first64_;
    std::vector<std::uint64_t> shifts_;
};

} // namespace sobol
} // namespace quantape::math::mc

#endif // QUANTAPE_MATH_RANDOM_SOBOL_SOBOLGENERATOR_H
