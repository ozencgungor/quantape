#ifndef QUANTAPE_MC_RANDOM_SOURCE_H
#define QUANTAPE_MC_RANDOM_SOURCE_H

#include "quantape/math/Random/McFarlandNormal.h"
#include "quantape/math/Random/ZigguratNormal.h"
#include "quantape/mc/SdePrimitives.h"

#include <Eigen/Dense>

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <stdexcept>

namespace quantape::mc {
/**
 * @file RandomSource.h
 * @brief Noise sources for the SDE simulation engine
 *
 * ## Keyed counter-based streams (reproducibility specification)
 *
 * Every logical draw is addressed by (seed, path p, step k, index i) and turned
 * into a 64-bit key by the single mixing function `keyForIndex`:
 *
 *     key(seed, p, i) = splitmix64( seed XOR (p * kPathSalt) XOR (i * kIndexSalt) )
 *     u_m             = splitmix64( key + m * kCounterSalt ),  m = 1, 2, ...   (retries)
 *
 * Gaussian draws and uniform streams use **disjoint index spaces**:
 *
 *     normal  (k, j) -> i = k * q + j                              (j < q)
 *     uniform (k, s) -> i = kUniformIndexTag + k * Stride + s      (tag = 2^63)
 *
 * The uniform indices occupy the upper half of the 64-bit index space while
 * Gaussian indices are `k*q + j`, so the two sets cannot intersect for any
 * factor count, step count or number of uniform streams. This is load-bearing:
 * `qeSample` reads its uniform only on the QE exponential branch, so an aliased
 * uniform would silently make the uniform of step k the *normal* of step k+1 —
 * which removes the independence the scheme's martingale correction assumes.
 * The uniform layout is reproducible too: a fixed stride (not the call's stream
 * count) makes `(step, stream) -> index` injective for every call pattern.
 *
 * The draw itself is `z = S(key, u_1, ...)` where S is a rejection sampler
 * (Ziggurat/McFarland). The loop consumes a *variable* number of words, but all
 * come from that one key, so the draw is a pure function of (seed, p, k, i):
 *
 *     draw(seed, p, k, i)  ==  the same value in any block run  (bitwise)
 *
 * independently of block size, path order and thread count. This is the
 * "keyed stream" idea: independence across draws is by *key*, not by position
 * in a shared sequence — required because rejection sampling rules out O(1)
 * skip-ahead on a shared sequential stream (PCG advance() cannot know the
 * variable consumption). Because a key is a bijection of the index, disjoint
 * index sets give independent draws.
 *
 * Mixer quality is load-bearing: SplitMix64 is full-width and well mixed,
 * and the samplers use the full 64-bit word (including the sign bit) to
 * form the sample, so a truncated/weak mixer would bias the tails.
 *
 * Two sampler policies are provided: McFarland (consumes the KeyedStream
 * directly; the existing generator is templated on the RNG) and Ziggurat
 * (takes a keyed seed, owns its internal xoshiro stream and reads one shared
 * process-wide ziggurat table).
 *
 * ## Extension protocol
 *
 * Derive from `RandomSourceBase<Derived>` and implement
 *
 *     std::size_t factorCount() const;
 *     void fillPath(std::size_t pathIndex, std::size_t step,
 *                   Eigen::MatrixXd& out, std::size_t col) const;
 *
 * writing `factorCount()` standardized draws into column `col`; the base
 * provides the block `fill(...)` the engine calls.
 */

/// SplitMix64 finalizer + counter step (full-width 64-bit mixer)
inline std::uint64_t splitmix64(std::uint64_t x) {
    x += 0x9E3779B97F4A7C15ULL;
    x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ULL;
    x = (x ^ (x >> 27)) * 0x94D049BB133111EBULL;
    return x ^ (x >> 31);
}

/// Counter-based bit stream for one logical normal; takes a pre-mixed key
/// and exposes the PCG-style 64-bit uniform interface the samplers consume:
/// the m-th uniform is `splitmix64(key + m * PSI)`.
class KeyedStream {
public:
    explicit KeyedStream(std::uint64_t key) : m_key(key) {}

    /// First draw is the (already fully mixed) key; rejection retries use
    /// splitmix64(key + m * PSI) — one mix per normal on the common path.
    std::uint64_t operator()() {
        if (m_counter++ == 0) {
            return m_key;
        }
        return splitmix64(m_key + m_counter * kCounterSalt);
    }

private:
    static constexpr std::uint64_t kCounterSalt = 0xBF58476D1CE4E5B9ULL;
    std::uint64_t m_key;
    std::uint64_t m_counter = 0;
};

/// Key derivation from a raw draw index. The single definition of the
/// index -> key map; `keyFor` (Gaussian) and `uniformKey` (uniform streams)
/// are the two documented ways to form the index.
inline std::uint64_t keyForIndex(std::uint64_t seed, std::size_t pathIndex, std::uint64_t index) {
    constexpr std::uint64_t kPathSalt = 0x9E3779B97F4A7C15ULL;
    constexpr std::uint64_t kIndexSalt = 0xBF58476D1CE4E5B9ULL;
    return splitmix64(seed ^ (static_cast<std::uint64_t>(pathIndex) * kPathSalt) ^
                      (index * kIndexSalt));
}

/// Key for gaussian factor `factor` at `step` of path `pathIndex`:
/// index `step * factorCount + factor`.
inline std::uint64_t keyFor(std::uint64_t seed, std::size_t pathIndex, std::size_t step,
                            std::size_t factor, std::size_t factorCount) {
    return keyForIndex(seed, pathIndex, static_cast<std::uint64_t>(step) * factorCount + factor);
}

/// Uniform indices are tagged into the upper half of the 64-bit index space.
/// A gaussian index is `step * factorCount + factor`, far below this tag for
/// any representable grid, so the two index sets are disjoint by construction
/// and a uniform stream can never be generated from a normal's key.
inline constexpr std::uint64_t kUniformIndexTag = 1ULL << 63;

/// Fixed per-step stride for uniform streams (maximum streams per step). Fixed
/// rather than taken from the call so that `(step, stream) -> index` is
/// injective for every call pattern, including calls that request fewer streams.
inline constexpr std::uint64_t kUniformStreamStride = 64;

/// Key for uniform stream `stream` at `step` of path `pathIndex`.
inline std::uint64_t uniformKey(std::uint64_t seed, std::size_t pathIndex, std::size_t step,
                                std::size_t stream) {
    return keyForIndex(seed, pathIndex,
                       kUniformIndexTag + static_cast<std::uint64_t>(step) * kUniformStreamStride +
                           stream);
}

/// Map a 64-bit key to a double uniform in [0, 1) (53-bit construction)
inline double toUniform(std::uint64_t key) {
    return static_cast<double>(key >> 11) * 0x1.0p-53;
}

/// Sampler policies (swappable): each takes the mixed key and returns one
/// standard normal E[z] = 0, Var[z] = 1 from an independent keyed stream.
struct McFarlandSampler {
    static double sample(std::uint64_t key) {
        KeyedStream stream(key);
        return quantape::math::mc::McFarlandNormal<KeyedStream>(stream)();
    }
};

/// The process-wide ziggurat table (256 layers + fast-path tables), built once
/// on first use. A function-local `static const` gives thread-safe one-time
/// initialization (C++11 magic statics); `generateZigguratTables()` costs
/// ~0.4 ms (64 bisections x 254 recurrence steps with erfc/log/sqrt) and must
/// never run per draw.
inline const quantape::math::mc::ZigguratTables& zigguratTables() {
    static const quantape::math::mc::ZigguratTables tables =
        quantape::math::mc::generateZigguratTables();
    return tables;
}

namespace detail {
/// Exponential-tail sampler, mirroring `ZigguratNormal::sampleTail()` exactly
/// (same draw order and arithmetic) so the keyed stream stays bitwise stable.
inline double zigguratTail(const quantape::math::mc::ZigguratTables& tab,
                           quantape::math::mc::Xoshiro256ss& rng) {
    const double r = tab.r;
    while (true) {
        const double u1 = rng.uniform01();
        const double u2 = rng.uniform01();
        const double tailX = -std::log(u1) / r + r;
        if (-2.0 * std::log(u2) >= (tailX - r) * (tailX - r)) {
            return tailX;
        }
    }
}

/// One standard normal from `key`, mirroring `ZigguratNormal::operator()`
/// bit-for-bit but reading the shared table instead of regenerating one per
/// draw. Any edit here must keep that identity, which
/// `MomentMatching.zigguratDrawsMatchPerDrawTablePath` pins.
inline double zigguratDraw(const quantape::math::mc::ZigguratTables& tab, std::uint64_t key) {
    quantape::math::mc::Xoshiro256ss rng(key);
    while (true) {
        const std::uint64_t u = rng();
        const int i = static_cast<int>(u & 0xFF);
        const auto s = static_cast<std::int64_t>(u);
        const double x = static_cast<double>(s) * tab.wtab[i];

        // Fast path: |s| < ktab[i]
        const auto absS = static_cast<std::uint64_t>(s < 0 ? -s : s);
        if (absS < tab.ktab[i]) {
            return x;
        }

        // Slow path: base-layer tail, then interior/top rejection.
        if (i == 0) {
            const double absX = std::abs(x);
            if (absX < tab.r) {
                return x; // under the curve in the base rectangle
            }
            const double tailX = zigguratTail(tab, rng);
            return (s < 0) ? -tailX : tailX;
        }
        const double absX = std::abs(x);
        const double yLo = tab.y[i - 1];
        const double yHi = (i < quantape::math::mc::ZigguratTables::N - 1) ? tab.y[i] : 1.0;
        const double yTest = yLo + rng.uniform01() * (yHi - yLo);
        if (yTest < std::exp(-0.5 * absX * absX)) {
            return x;
        }
    }
}
} // namespace detail

struct ZigguratSampler {
    // The 0.4 ms ziggurat table is shared (built once); only the per-key
    // xoshiro stream is constructed per draw, so this is a usable sampler
    // instead of rebuilding the whole table for every normal.
    static double sample(std::uint64_t key) { return detail::zigguratDraw(zigguratTables(), key); }
};

/**
 * @brief implement per-path `fillPath`, get block `fill` free
 *
 * Derived must provide:
 *   std::size_t factorCount() const;
 *   void fillPath(std::size_t pathIndex, std::size_t step,
 *                 Eigen::MatrixXd& out, std::size_t col) const;
 * writing factorCount() standardized draws into column `col`.
 */
template <typename Derived>
class RandomSourceBase {
public:
    void fill(std::size_t step, std::size_t pathBegin, std::size_t nPaths,
              Eigen::MatrixXd& out) const {
        const Derived& self = *static_cast<const Derived*>(this);
        out.resize(static_cast<Eigen::Index>(self.factorCount()),
                   static_cast<Eigen::Index>(nPaths));
        for (std::size_t j = 0; j < nPaths; ++j) {
            self.fillPath(pathBegin + j, step, out, j);
        }
    }
};

/**
 * @brief Independent standard normals: Z(j, p) = z_j^{(p)}(t_step) ~ N(0, 1)
 *
 * q independent factors (E[z_j z_k] = delta_jk, independent across paths
 * and steps), constructed by keyed streams:
 *
 *     Z(j, p) = Sampler::sample(keyFor(seed, pathBegin + p, step, j, q))
 *
 * Stateless, const, thread-safe; block fills are exactly reproducible and
 * independent of block size, order and thread count. This is the source
 * for every model in SDE factor form: the factor projection (Cholesky
 * L, PCA loadings S) normally lives in the model's diffusion,
 * g_j = L.col(j) or S.col(j).
 */
template <typename Sampler = McFarlandSampler>
class IidGaussianSource : public RandomSourceBase<IidGaussianSource<Sampler>> {
public:
    IidGaussianSource(std::size_t factorCount, std::uint64_t seed)
        : m_factorCount(factorCount), m_seed(seed) {}

    std::size_t factorCount() const { return m_factorCount; }

    /// Block fill with the step-dependent key salt hoisted out of the path
    /// loop (the path salt advances by addition); the per-draw key is
    /// identical to `keyFor`, so fills stay bitwise equal to per-path fills.
    void fill(std::size_t step, std::size_t pathBegin, std::size_t nPaths,
              Eigen::MatrixXd& out) const {
        constexpr std::uint64_t kPathSalt = 0x9E3779B97F4A7C15ULL;
        constexpr std::uint64_t kIndexSalt = 0xBF58476D1CE4E5B9ULL;
        out.resize(static_cast<Eigen::Index>(m_factorCount), static_cast<Eigen::Index>(nPaths));
        const std::uint64_t indexBase =
            static_cast<std::uint64_t>(step) * static_cast<std::uint64_t>(m_factorCount);
        for (std::size_t j = 0; j < m_factorCount; ++j) {
            const std::uint64_t indexSalt = (indexBase + j) * kIndexSalt;
            std::uint64_t pathSalt = static_cast<std::uint64_t>(pathBegin) * kPathSalt;
            for (std::size_t p = 0; p < nPaths; ++p) {
                const std::uint64_t key = splitmix64(m_seed ^ pathSalt ^ indexSalt);
                out(static_cast<Eigen::Index>(j), static_cast<Eigen::Index>(p)) =
                    Sampler::sample(key);
                pathSalt += kPathSalt;
            }
        }
    }

    void fillPath(std::size_t pathIndex, std::size_t step, Eigen::MatrixXd& out,
                  std::size_t col) const {
        for (std::size_t i = 0; i < m_factorCount; ++i) {
            const std::uint64_t key = keyFor(m_seed, pathIndex, step, i, m_factorCount);
            out(static_cast<Eigen::Index>(i), static_cast<Eigen::Index>(col)) =
                Sampler::sample(key);
        }
    }

    /// Uniform streams for moment-matching schemes: disjoint key space (see the
    /// file header), one draw per (path, step, stream), `U(0,1)`. The index is
    /// `kUniformIndexTag + step*kUniformStreamStride + stream`, so it can never
    /// coincide with a gaussian index `step*factorCount + factor`.
    void fillUniform(std::size_t step, std::size_t pathBegin, std::size_t nPaths,
                     std::size_t streamBegin, std::size_t uniformStreams,
                     Eigen::MatrixXd& out) const {
        if (streamBegin + uniformStreams > kUniformStreamStride) [[unlikely]] {
            throw std::out_of_range("IidGaussianSource: at most 64 uniform streams per step");
        }
        out.resize(static_cast<Eigen::Index>(uniformStreams), static_cast<Eigen::Index>(nPaths));
        for (std::size_t s = 0; s < uniformStreams; ++s) {
            for (std::size_t j = 0; j < nPaths; ++j) {
                out(static_cast<Eigen::Index>(s), static_cast<Eigen::Index>(j)) =
                    toUniform(uniformKey(m_seed, pathBegin + j, step, streamBegin + s));
            }
        }
    }

private:
    std::size_t m_factorCount;
    std::uint64_t m_seed;
};

} // namespace quantape::mc

#endif // QUANTAPE_MC_RANDOM_SOURCE_H
