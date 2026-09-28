#ifndef QUANTAPE_MC_SOBOL_SOURCE_H
#define QUANTAPE_MC_SOBOL_SOURCE_H

#include "quantape/math/Random/Sobol/SobolGenerator.h"
#include "quantape/mc/SdePrimitives.h"

#include <Eigen/Dense>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <utility>

namespace quantape::mc {
/**
 * @file SobolSource.h
 * @brief QMC random source: Sobol points for the SDE engine
 *
 * Adapts `quantape::math::mc::sobol::SobolGenerator` to the engine's
 * `RandomSource` / `UniformRandomSource` contracts. The key property that
 * makes this a drop-in source: Sobol points are **pure functions of
 * (point index, dimension, shift seed)** with jump-ahead — no cursor — so
 *
 *     path p reads Sobol point (pointOffset + p),
 *     factor j of step k reads dimension k*q + j + 1,
 *     uniform stream s of step k reads dimension nSteps*q + k*S + s + 1,
 *
 * and every existing engine guarantee carries over unchanged: blocked
 * evaluation == per-path evaluation bitwise, `simulatePath(i)` bitwise,
 * block-size invariance, deterministic parallel schedules (TBB and
 * multiprocessing shards), and common random numbers across theta (the
 * draws never involve the parameters, so all pathwise AD modes work with
 * QMC unchanged).
 *
 * Replications for QMC error estimation come from the generator's
 * structure-preserving **digital shift**: build one generator per replica
 * with a different `shiftSeed` (tables are shared/read-only — use the
 * `sharedFromFile` / `sharedFromBinary` / `sharedFromDefaultTable`
 * factories so replicas do not copy the direction words).
 *
 * Path construction is *step-major*: dimension `k*q + j` (increments in
 * time order). This keeps the source stateless and shard-addressable; a
 * Brownian-bridge reordering (lower effective dimension, better QMC
 * constants) needs a path-scoped simulator pass and is future work
 * (`sde_simulator_design.md` S7).
 *
 * Cost: one Sobol point costs O(popcount(point)) word XORs plus the
 * inverse-normal map, versus a few ns for the hashed i.i.d. source — the
 * trade is accepted for the (log N)^s / N QMC error rate.
 */

class SobolSource {
public:
    /// `generator` must have at least `nSteps*(factorCount + uniformStreams)`
    /// prepared dimensions. Its configured `shiftSeed` selects the replica.
    SobolSource(std::shared_ptr<const math::mc::sobol::SobolGenerator> generator,
                std::size_t factorCount, std::size_t nSteps, std::size_t uniformStreams = 0)
        : generator_(std::move(generator)), factorCount_(factorCount), nSteps_(nSteps),
          uniformStreams_(uniformStreams) {
        if (!generator_) {
            throw std::invalid_argument("SobolSource: generator must not be null");
        }
        if (factorCount_ == 0) {
            throw std::invalid_argument("SobolSource: factorCount must be positive");
        }
        const std::size_t required =
            nSteps_ * factorCount_ + nSteps_ * uniformStreams_;
        if (required > generator_->preparedDimension()) {
            throw std::invalid_argument(
                "SobolSource: table too small for layout (need " + std::to_string(required) +
                " dimensions, have " + std::to_string(generator_->preparedDimension()) + ")");
        }
    }

    /// Convenience: shared generator from a text Joe-Kuo table.
    static SobolSource fromFile(const std::string& path, std::size_t factorCount,
                                std::size_t nSteps, std::size_t uniformStreams = 0,
                                std::uint64_t shiftSeed = 0, std::uint32_t maxBits = 32) {
        math::mc::sobol::SobolOptions options;
        options.shiftSeed = shiftSeed;
        options.maxBits = maxBits;
        options.maxDimension = static_cast<std::uint32_t>(nSteps * (factorCount + uniformStreams));
        return SobolSource(math::mc::sobol::SobolGenerator::sharedFromFile(path, options),
                           factorCount, nSteps, uniformStreams);
    }

    /// Convenience: shared generator from the compile-time table (if configured).
    static SobolSource fromDefaultTable(std::size_t factorCount, std::size_t nSteps,
                                        std::size_t uniformStreams = 0,
                                        std::uint64_t shiftSeed = 0, std::uint32_t maxBits = 32) {
        math::mc::sobol::SobolOptions options;
        options.shiftSeed = shiftSeed;
        options.maxBits = maxBits;
        options.maxDimension = static_cast<std::uint32_t>(nSteps * (factorCount + uniformStreams));
        return SobolSource(math::mc::sobol::SobolGenerator::sharedFromDefaultTable(options),
                           factorCount, nSteps, uniformStreams);
    }

    std::size_t factorCount() const { return factorCount_; }
    std::size_t nSteps() const { return nSteps_; }
    std::size_t uniformStreams() const { return uniformStreams_; }
    std::uint64_t shiftSeed() const { return generator_->shiftSeed(); }

    /// Standardized Gaussian draws: `out(j, p)` = Phi^{-1}(Sobol value) at
    /// dimension `k*q + j + 1` of point `pointOffset + pathBegin + p`.
    void fill(std::size_t step, std::size_t pathBegin, std::size_t nPaths,
              Eigen::MatrixXd& out) const {
        out.resize(static_cast<Eigen::Index>(factorCount_), static_cast<Eigen::Index>(nPaths));
        for (std::size_t p = 0; p < nPaths; ++p) {
            const std::uint64_t point = pathBegin + p;
            for (std::size_t j = 0; j < factorCount_; ++j) {
                out(static_cast<Eigen::Index>(j), static_cast<Eigen::Index>(p)) =
                    generator_->normal(point, normalDim(step, j));
            }
        }
    }

    /// Uniform draws on disjoint dimensions (QE schemes and other
    /// uniform-consuming schemes).
    void fillUniform(std::size_t step, std::size_t pathBegin, std::size_t nPaths,
                     std::size_t streamBegin, std::size_t uniformStreamCount,
                     Eigen::MatrixXd& out) const {
        if (streamBegin + uniformStreamCount > uniformStreams_) {
            throw std::out_of_range("SobolSource: uniform stream out of range");
        }
        out.resize(static_cast<Eigen::Index>(uniformStreamCount),
                   static_cast<Eigen::Index>(nPaths));
        for (std::size_t p = 0; p < nPaths; ++p) {
            const std::uint64_t point = pathBegin + p;
            for (std::size_t s = 0; s < uniformStreamCount; ++s) {
                out(static_cast<Eigen::Index>(s), static_cast<Eigen::Index>(p)) =
                    generator_->uniform(point, uniformDim(step, streamBegin + s));
            }
        }
    }

public:
    /// First usable dimension is 1 (the generator's `firstDimension()`).
    std::uint32_t normalDim(std::size_t step, std::size_t factor) const {
        return static_cast<std::uint32_t>(step * factorCount_ + factor + 1);
    }
    std::uint32_t uniformDim(std::size_t step, std::size_t stream) const {
        return static_cast<std::uint32_t>(nSteps_ * factorCount_ + step * uniformStreams_ +
                                          stream + 1);
    }

private:
    std::shared_ptr<const math::mc::sobol::SobolGenerator> generator_;
    std::size_t factorCount_;
    std::size_t nSteps_;
    std::size_t uniformStreams_;
};

} // namespace quantape::mc

#endif // QUANTAPE_MC_SOBOL_SOURCE_H
