// bench_sobol.cpp — Sobol generator throughput, memory, and engine QMC costs
//
//   ./build/bench_sobol [reps] [table]
//
// Sections:
//   1. table load + prepare (full 65,536 dims vs 1,008-dim layout slice)
//   2. random-access uniformBits / normal throughput (ns per draw)
//   3. batch fillPath cost for a 252-step x 4-factor layout
//   4. engine path cost: GBM (1 factor) and 4-factor geometric model,
//      Sobol vs iid source, sequential vs parallel schedule
#include "quantape/log/Log.h"
#include "quantape/math/Random/Sobol/SobolGenerator.h"
#include "quantape/mc/RandomSource.h"
#include "quantape/mc/SdeSimulator.h"
#include "quantape/mc/SobolSource.h"
#include "quantape/mc/TimeGrid.h"
#include "quantape/mc/processes/SdeProcesses.h"

#include <Eigen/Dense>

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <functional>
#include <memory>
#include <random>
#include <vector>

using namespace quantape::math::mc;
using namespace quantape::math::mc::sobol;
using namespace quantape::mc;
using namespace quantape::processes;

namespace {

double bench_ns(const char* name, double per_unit, int reps, const std::function<double()>& fn) {
    fn();
    const auto t0 = std::chrono::steady_clock::now();
    double sink = 0.0;
    for (int r = 0; r < reps; ++r) {
        sink += fn();
    }
    const auto t1 = std::chrono::steady_clock::now();
    const double ns = std::chrono::duration<double, std::nano>(t1 - t0).count() / reps;
    std::printf("%-46s %10.1f us/run   %7.2f ns/unit   (sink %.3e)\n", name, ns / 1000.0,
                per_unit > 0 ? ns / per_unit : 0.0, sink);
    std::fflush(stdout);
    return ns;
}

void report_memory(const char* label, std::uint32_t prepared, std::uint32_t maxBits) {
    const double bytes_per_word = (maxBits <= 32) ? 4.0 : 8.0;
    const double mib =
        (static_cast<double>(prepared - 1) * maxBits * bytes_per_word) / (1024 * 1024);
    std::printf("  %-34s prepared=%6u dims  maxBits=%2u  words ~ %.1f MiB\n", label, prepared,
                maxBits, mib);
}

/// Four independent geometric factors (exercises q = 4 per step).
struct FourFactorGbm {
    template <typename S>
    void drift(const StateMatrix<S>& x, double, const std::vector<S>&, StateMatrix<S>& out) const {
        out = (x.array() * S(0.03)).matrix();
    }
    template <typename S>
    void diffusion(const StateMatrix<S>& x, double, const std::vector<S>&, std::size_t factor,
                   StateMatrix<S>& out) const {
        out = (x.array() * S(0.10 + 0.05 * static_cast<double>(factor))).matrix();
    }
};

} // namespace

int main(int argc, char** argv) {
    const int reps = argc > 1 ? std::atoi(argv[1]) : 3;
    std::string table = argc > 2 ? argv[2] : "joe-kuo-65536-refined-w256.txt";
    const auto exists = [](const std::string& path) {
        return !path.empty() && std::filesystem::exists(path);
    };
    // Resolve the text table: explicit path, then parent directories (IDE /
    // CMake run configurations often start in the build directory), then the
    // compile-time binary asset. Missing everything is an error.
    if (!exists(table)) {
        for (const char* prefix : {"../", "../../", "../../../"}) {
            const std::string candidate = std::string(prefix) + table;
            if (exists(candidate)) {
                table = candidate;
                break;
            }
        }
    }
    const std::string defaultPath = SobolGenerator::defaultTablePath();
    const bool haveText = exists(table);
    const bool haveAsset = exists(defaultPath);
    if (!haveText && !haveAsset) {
        QTA_LOG_ERROR("bench",
                      "no Sobol table found (looked for '{}' and the compile-time asset '{}'); "
                      "pass a table path or run from the repo root",
                      table, defaultPath);
        return 1;
    }
    std::printf("Sobol generator benchmark, reps=%d, table=%s%s\n", reps,
                haveText ? table.c_str() : defaultPath.c_str(),
                haveText ? "" : " (compile-time binary asset)");

    // ── 0. Compile-time table (QSB1 binary asset, zero-copy mmap) ──
    {
        const std::string def = SobolGenerator::defaultTablePath();
        if (!def.empty() && std::filesystem::exists(def)) {
            const auto t0 = std::chrono::steady_clock::now();
            auto first = SobolGenerator::sharedFromDefaultTable(SobolOptions{0, 1, false});
            const auto t1 = std::chrono::steady_clock::now();
            auto second = SobolGenerator::sharedFromDefaultTable(SobolOptions{0, 1, false});
            const auto t2 = std::chrono::steady_clock::now();
            std::printf("compile-time table (mmap): cold %.3f ms, cached %.3f us, mapped=%s, "
                        "shared=%s, heap words = 0 MiB\n",
                        std::chrono::duration<double, std::milli>(t1 - t0).count(),
                        std::chrono::duration<double, std::micro>(t2 - t1).count(),
                        first->mapped() ? "yes" : "no", first.get() == second.get() ? "yes" : "NO");
        } else {
            std::printf("compile-time table: not configured\n");
        }
    }

    // ── 1. Load / prepare (text table; skipped when only the binary asset is
    // available) ──
    if (haveText) {
        const auto t0 = std::chrono::steady_clock::now();
        auto full = SobolGenerator::fromFile(table, SobolOptions{0, 1, false, 32, 0, false});
        const auto t1 = std::chrono::steady_clock::now();
        const double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        std::printf("load+prepare full table: %.1f ms\n", ms);
        report_memory("full table (32-bit words)", full.preparedDimension(), full.maxBits());

        const auto t2 = std::chrono::steady_clock::now();
        auto slice = SobolGenerator::fromFile(table, SobolOptions{0, 1, false, 32, 1008, false});
        const auto t3 = std::chrono::steady_clock::now();
        std::printf("load+prepare 1,008-dim slice: %.1f ms\n",
                    std::chrono::duration<double, std::milli>(t3 - t2).count());
        report_memory("layout slice (252 steps x 4)", slice.preparedDimension(), slice.maxBits());
        auto wide = SobolGenerator::fromFile(table, SobolOptions{0, 1, false, 64, 0, false});
        report_memory("full table (64-bit words)", wide.preparedDimension(), wide.maxBits());
    } else {
        std::printf("text table not found: skipping load/prepare timings\n");
    }

    std::shared_ptr<const SobolGenerator> shared;
    if (haveAsset) {
        shared =
            SobolGenerator::sharedFromDefaultTable(SobolOptions{12345, 1, false, 32, 0, false});
    } else {
        shared = std::make_shared<const SobolGenerator>(
            SobolGenerator::fromFile(table, SobolOptions{12345, 1, false, 32, 0, false}));
    }

    // ── 2. Random-access throughput ──
    {
        std::mt19937_64 rng(7);
        const std::size_t draws = 2000000;
        std::vector<std::uint64_t> points(draws);
        std::vector<std::uint32_t> dims(draws);
        for (std::size_t i = 0; i < draws; ++i) {
            points[i] = rng() & ((1ULL << 20) - 1);
            dims[i] = 1 + static_cast<std::uint32_t>(rng() % shared->preparedDimension());
        }
        bench_ns("uniformBits random access (no shift)", static_cast<double>(draws), reps, [&] {
            double sink = 0.0;
            for (std::size_t i = 0; i < draws; ++i) {
                sink += static_cast<double>(shared->uniformBits(points[i], dims[i], 0) & 0xFF);
            }
            return sink;
        });
        bench_ns("uniformBits random access (shifted)", static_cast<double>(draws), reps, [&] {
            double sink = 0.0;
            for (std::size_t i = 0; i < draws; ++i) {
                sink += static_cast<double>(shared->uniformBits(points[i], dims[i], 999) & 0xFF);
            }
            return sink;
        });
        bench_ns("normal random access (shifted)", static_cast<double>(draws), reps, [&] {
            double sink = 0.0;
            for (std::size_t i = 0; i < draws; ++i) {
                sink += shared->normal(points[i], dims[i], 999);
            }
            return sink;
        });
    }

    // ── 3. Batch path fill (252 steps x 4 factors) ──
    {
        const std::size_t steps = 252, paths = 4096;
        SobolGaussianSource source(shared, 4, steps, 1, 0, 777);
        Eigen::MatrixXd out;
        bench_ns("source fill 252 steps x 4096 paths", static_cast<double>(steps * paths), reps,
                 [&] {
                     double sink = 0.0;
                     for (std::size_t k = 0; k < steps; ++k) {
                         source.fill(k, 0, paths, out);
                         sink += out(0, 0);
                     }
                     return sink;
                 });
    }

    // ── 4. Engine path costs ──
    {
        const std::size_t steps = 252;
        const TimeGrid grid(1.0, steps);
        std::vector<std::vector<double>> theta(steps);
        const auto x0 = Eigen::VectorXd::Constant(1, 100.0);

        const std::size_t nPaths = 65536, blockSize = 4096;
        const GbmProcess model{0.03, 0.2};
        SdeSimulator<double> simulator(grid, theta);

        const SobolGaussianSource qmc(shared, 1, steps, 1, 0, 4242);
        bench_ns(
            "engine GBM seq (sobol, 65k x 252)", static_cast<double>(nPaths * steps), reps, [&] {
                const auto blocks = simulator.simulate(x0, driftOf(model), diffusionOf(model), qmc,
                                                       nPaths, blockSize, Schedule::Sequential);
                return blocks.back().states.back()(0, 0);
            });
        bench_ns(
            "engine GBM par (sobol, 65k x 252)", static_cast<double>(nPaths * steps), reps, [&] {
                const auto blocks = simulator.simulate(x0, driftOf(model), diffusionOf(model), qmc,
                                                       nPaths, blockSize, Schedule::Parallel);
                return blocks.back().states.back()(0, 0);
            });
        const IidGaussianSource<> iid(1, 4242);
        bench_ns("engine GBM par (iid, 65k x 252)", static_cast<double>(nPaths * steps), reps, [&] {
            const auto blocks = simulator.simulate(x0, driftOf(model), diffusionOf(model), iid,
                                                   nPaths, blockSize, Schedule::Parallel);
            return blocks.back().states.back()(0, 0);
        });

        // 4 factors x 252 steps (1,008 dimensions)
        const std::size_t nPaths4 = 16384;
        const auto x04 = Eigen::VectorXd::Constant(4, 100.0);
        const FourFactorGbm model4;
        const SobolGaussianSource qmc4(shared, 4, steps, 1, 0, 4242);
        bench_ns("engine 4-factor seq (sobol, 16k x 252)", static_cast<double>(nPaths4 * steps),
                 reps, [&] {
                     const auto blocks =
                         simulator.simulate(x04, driftOf(model4), diffusionOf(model4), qmc4,
                                            nPaths4, 2048, Schedule::Sequential);
                     return blocks.back().states.back()(0, 0);
                 });
        bench_ns("engine 4-factor par (sobol, 16k x 252)", static_cast<double>(nPaths4 * steps),
                 reps, [&] {
                     const auto blocks =
                         simulator.simulate(x04, driftOf(model4), diffusionOf(model4), qmc4,
                                            nPaths4, 2048, Schedule::Parallel);
                     return blocks.back().states.back()(0, 0);
                 });
    }

    // ── 5. Cheap replicas (no table rebuild) ──
    {
        const std::size_t steps = 252;
        SobolGaussianSource base(shared, 4, steps, 1, 0, 1);
        bench_ns("replica() construction", 1.0, 2000, [&] {
            double sink = 0.0;
            for (int r = 0; r < 100; ++r) {
                const auto rep = base.replica(static_cast<std::uint64_t>(r) + 2);
                sink += static_cast<double>(rep.shiftSeed() & 0xFF);
            }
            return sink;
        });
    }

    return 0;
}
