#ifndef QUANTAPE_MC_RANDOM_SOURCE_H
#define QUANTAPE_MC_RANDOM_SOURCE_H

#include "quantape/math/Random/McFarlandNormal.h"
#include "quantape/math/Random/ZigguratNormal.h"
#include "quantape/mc/SdePrimitives.h"

#include <Eigen/Dense>

#include <cstddef>
#include <cstdint>

namespace quantape::mc {
/**
 * @file RandomSource.h
 * @brief Noise sources for the SDE simulation engine
 *
 * ## Keyed counter-based streams (reproducibility specification)
 *
 * Every logical normal draw is addressed by four indices
 * (seed, path p, step k, factor j). Writing
 *
 *     normalIndex(k, j) = k * q + j
 *     key(seed, p, k, j) = splitmix64( seed XOR
 *                                      splitmix64(p * PHI + normalIndex(k, j)) )
 *     u_m = splitmix64(key + m * PSI),   m = 0, 1, 2, ...        (PSI = counter salt)
 *
 * the draw is `z_j^{(p)}(t_k) = S(u_0, u_1, ...)` where S is a rejection
 * sampler (Ziggurat/McFarland). The rejection loop consumes a *variable*
 * number m of uniforms, but all of them come from that one normal's own
 * counter stream, so the draw is a pure function of (seed, p, k, j):
 *
 *     z(seed, p, k, j)  ==  path p of any block run  (bitwise)
 *
 * independently of block size, path order and thread count. This is the
 * "keyed stream" idea: independence across draws is by *key*, not by
 * position in a shared sequence — required because rejection sampling
 * rules out O(1) skip-ahead on a shared sequential stream (PCG advance()
 * cannot know the variable consumption).
 *
 * Mixer quality is load-bearing: SplitMix64 is full-width and well mixed,
 * and the samplers use the full 64-bit word (including the sign bit) to
 * form the sample, so a truncated/weak mixer would bias the tails.
 *
 * Two sampler policies are provided: McFarland (consumes the KeyedStream
 * directly; the existing generator is templated on the RNG) and Ziggurat
 * (takes a keyed seed and owns its internal PCG stream).
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

/// Key derivation for one logical normal: (seed, path, step, factor) ->
/// `splitmix64(seed XOR path * PHI XOR normalIndex * PSI)` (one mix; both
/// operands are odd-constant-spread before the XOR, then fully mixed).
inline std::uint64_t keyFor(std::uint64_t seed, std::size_t pathIndex, std::size_t step,
                            std::size_t factor, std::size_t factorCount) {
    constexpr std::uint64_t kPathSalt = 0x9E3779B97F4A7C15ULL;
    constexpr std::uint64_t kIndexSalt = 0xBF58476D1CE4E5B9ULL;
    const std::uint64_t normalIndex =
        static_cast<std::uint64_t>(step) * static_cast<std::uint64_t>(factorCount) +
        static_cast<std::uint64_t>(factor);
    return splitmix64(seed ^ (static_cast<std::uint64_t>(pathIndex) * kPathSalt) ^
                      (normalIndex * kIndexSalt));
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

struct ZigguratSampler {
    // ZigguratNormal takes a seed and owns a PCG stream internally: the
    // keyed seed makes each normal's stream independent and reproducible.
    static double sample(std::uint64_t key) { return quantape::math::mc::ZigguratNormal(key)(); }
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

    void fillPath(std::size_t pathIndex, std::size_t step, Eigen::MatrixXd& out,
                  std::size_t col) const {
        for (std::size_t i = 0; i < m_factorCount; ++i) {
            const std::uint64_t key = keyFor(m_seed, pathIndex, step, i, m_factorCount);
            out(static_cast<Eigen::Index>(i), static_cast<Eigen::Index>(col)) =
                Sampler::sample(key);
        }
    }

    /// Uniform streams for moment-matching schemes: disjoint key space
    /// (factor indices beyond the Gaussian factors), one draw per
    /// (path, step, stream).
    void fillUniform(std::size_t step, std::size_t pathBegin, std::size_t nPaths,
                     std::size_t streamBegin, std::size_t uniformStreams,
                     Eigen::MatrixXd& out) const {
        out.resize(static_cast<Eigen::Index>(uniformStreams), static_cast<Eigen::Index>(nPaths));
        for (std::size_t s = 0; s < uniformStreams; ++s) {
            for (std::size_t j = 0; j < nPaths; ++j) {
                const std::uint64_t key =
                    keyFor(m_seed, pathBegin + j, step, m_factorCount + 1 + streamBegin + s,
                           m_factorCount);
                out(static_cast<Eigen::Index>(s), static_cast<Eigen::Index>(j)) = toUniform(key);
            }
        }
    }

private:
    std::size_t m_factorCount;
    std::uint64_t m_seed;
};

} // namespace quantape::mc

#endif // QUANTAPE_MC_RANDOM_SOURCE_H
