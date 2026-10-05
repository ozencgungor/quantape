// bench_sde.cpp — timing harness for the SDE simulation engine
//
// Measures per-draw / per-path-step costs (streamed, so large path tensors
// are not allocated), parallel-schedule scaling, and pathwise-gradient
// costs (per-path var tapes + tape size per path). Each case is run `reps`
// times after a warmup; the report shows p50/p99 wall time, throughput in
// millions of units/s and allocations made during the timed region:
//   ./build/bench_sde [reps]
//   /usr/bin/sample bench_sde 5 -file /tmp/prof.txt
#include "quantape/math/StanMath.h"

#include "quantape/format/Number.h"
#include "quantape/log/Log.h"
#include "quantape/mc/Estimator.h"
#include "quantape/mc/ForwardStan.h"
#include "quantape/mc/Gradients.h"
#include "quantape/mc/MomentMatching.h"
#include "quantape/mc/Parallel.h"
#include "quantape/mc/RandomSource.h"
#include "quantape/mc/Schemes.h"
#include "quantape/mc/SchemesStan.h"
#include "quantape/mc/SdePrimitives.h"
#include "quantape/mc/SdeSimulator.h"
#include "quantape/mc/SobolSource.h"
#include "quantape/mc/StateDerivatives.h"
#include "quantape/mc/TimeGrid.h"
#include "quantape/mc/mcfwdrev/ForwardGradients.h"
#include "quantape/mc/mcfwdrev/LeanGradients.h"
#include "quantape/mc/processes/HestonQeProcess.h"
#include "quantape/mc/processes/SdeProcesses.h"

#include <Eigen/Dense>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <memory>
#include <new>
#include <random>
#include <string>
#include <type_traits>
#include <vector>

using quantape::mc::Euler;
using quantape::mc::IidGaussianSource;
using quantape::mc::Milstein;
using quantape::mc::Schedule;
using quantape::mc::SdeSimulator;
using quantape::mc::TimeGrid;
using quantape::processes::GbmProcess;
using quantape::processes::HestonProcess;
using quantape::processes::HestonQeProcess;

namespace {

// ── allocation counters (compile-time opt-in, see bench_sde_alloc) ──
//
// Eigen and the Stan arena allocate through malloc, not operator new, so
// counting every allocation needs the malloc/free symbols replaced; that
// replaces the system allocator fast path and distorts timings badly (the
// per-step buffers make millions of tiny allocations). The counters and
// overrides therefore live behind QUANTAPE_SDE_ALLOC_COUNTS and are built
// into the separate bench_sde_alloc target, whose allocation column is the
// metric and whose timing column is not comparable to bench_sde.
#if defined(QUANTAPE_SDE_ALLOC_COUNTS)
std::atomic<std::uint64_t> g_allocations{0};
std::atomic<std::uint64_t> g_allocatedBytes{0};

void countAllocation(std::size_t bytes) {
    g_allocations.fetch_add(1, std::memory_order_relaxed);
    g_allocatedBytes.fetch_add(bytes, std::memory_order_relaxed);
}
#endif

volatile double g_sink = 0.0;

} // namespace

#if defined(QUANTAPE_SDE_ALLOC_COUNTS) && defined(__APPLE__)
#include <malloc/malloc.h>

// Eigen and the Stan arena call malloc/free from header code instantiated in
// this executable; defining the symbols here counts those allocations too.
extern "C" void* malloc(std::size_t size) {
    countAllocation(size);
    return malloc_zone_malloc(malloc_default_zone(), size);
}
extern "C" void free(void* pointer) {
    if (pointer != nullptr) {
        malloc_zone_free(malloc_default_zone(), pointer);
    }
}
#endif

#if defined(QUANTAPE_SDE_ALLOC_COUNTS)
void* operator new(std::size_t size) {
    countAllocation(size);
#if defined(__APPLE__)
    void* p = malloc_zone_malloc(malloc_default_zone(), size);
#else
    void* p = std::malloc(size);
#endif
    if (p != nullptr) {
        return p;
    }
    throw std::bad_alloc();
}
void* operator new[](std::size_t size) {
    countAllocation(size);
#if defined(__APPLE__)
    void* p = malloc_zone_malloc(malloc_default_zone(), size);
#else
    void* p = std::malloc(size);
#endif
    if (p != nullptr) {
        return p;
    }
    throw std::bad_alloc();
}
void* operator new(std::size_t size, std::align_val_t alignment) {
    countAllocation(size);
    void* p = nullptr;
    if (posix_memalign(&p, static_cast<std::size_t>(alignment), size) == 0) {
        return p;
    }
    throw std::bad_alloc();
}
void* operator new[](std::size_t size, std::align_val_t alignment) {
    return ::operator new(size, alignment);
}
void* operator new(std::size_t size, const std::nothrow_t&) noexcept {
    countAllocation(size);
#if defined(__APPLE__)
    return malloc_zone_malloc(malloc_default_zone(), size);
#else
    return std::malloc(size);
#endif
}
void* operator new[](std::size_t size, const std::nothrow_t&) noexcept {
    return ::operator new(size, std::nothrow);
}
void operator delete(void* p) noexcept {
#if defined(__APPLE__)
    if (p != nullptr) {
        malloc_zone_free(malloc_default_zone(), p);
    }
#else
    std::free(p);
#endif
}
void operator delete[](void* p) noexcept {
    ::operator delete(p);
}
void operator delete(void* p, std::size_t) noexcept {
    ::operator delete(p);
}
void operator delete[](void* p, std::size_t) noexcept {
    ::operator delete(p);
}
void operator delete(void* p, std::align_val_t) noexcept {
#if defined(__APPLE__)
    if (p != nullptr) {
        malloc_zone_free(malloc_default_zone(), p);
    }
#else
    std::free(p);
#endif
}
void operator delete[](void* p, std::align_val_t) noexcept {
    ::operator delete(p);
}
void operator delete(void* p, std::size_t, std::align_val_t) noexcept {
    ::operator delete(p);
}
void operator delete[](void* p, std::size_t, std::align_val_t) noexcept {
    ::operator delete(p);
}
void operator delete(void* p, const std::nothrow_t&) noexcept {
    ::operator delete(p);
}
void operator delete[](void* p, const std::nothrow_t&) noexcept {
    ::operator delete(p);
}
#endif // QUANTAPE_SDE_ALLOC_COUNTS

namespace {

struct Timing {
    double p50Us = 0.0;
    double p99Us = 0.0;
    double meanUs = 0.0;
    std::uint64_t allocations = 0;
    std::uint64_t allocatedBytes = 0;
};

/// `per_unit` is paths or path-steps, used for ns/unit and Munit/s.
void bench_us(const std::string& name, double per_unit, int reps,
              const std::function<double()>& fn) {
    fn(); // warmup (also pre-touches allocations that repeat per run)
    std::vector<double> samples;
    samples.reserve(static_cast<std::size_t>(reps));
    Timing timing;
#if defined(QUANTAPE_SDE_ALLOC_COUNTS)
    timing.allocations = UINT64_MAX;
#endif
    for (int i = 0; i < reps; ++i) {
#if defined(QUANTAPE_SDE_ALLOC_COUNTS)
        g_allocations.store(0, std::memory_order_relaxed);
        g_allocatedBytes.store(0, std::memory_order_relaxed);
#endif
        const auto t0 = std::chrono::steady_clock::now();
        g_sink = fn();
        const auto t1 = std::chrono::steady_clock::now();
        samples.push_back(std::chrono::duration<double, std::micro>(t1 - t0).count());
#if defined(QUANTAPE_SDE_ALLOC_COUNTS)
        const std::uint64_t allocations = g_allocations.load(std::memory_order_relaxed);
        if (allocations < timing.allocations) {
            timing.allocations = allocations;
            timing.allocatedBytes = g_allocatedBytes.load(std::memory_order_relaxed);
        }
#endif
    }
    std::sort(samples.begin(), samples.end());
    const auto quantile = [&](double q) {
        const std::size_t index =
            static_cast<std::size_t>(std::ceil(q * static_cast<double>(samples.size()))) - 1;
        return samples[std::min(index, samples.size() - 1)];
    };
    timing.p50Us = quantile(0.5);
    timing.p99Us = quantile(0.99);
    double sum = 0.0;
    for (double sample : samples) {
        sum += sample;
    }
    timing.meanUs = sum / static_cast<double>(samples.size());
    const double nsPerUnit = per_unit > 0.0 ? timing.meanUs * 1000.0 / per_unit : 0.0;
    const double mUnitsPerSecond = per_unit > 0.0 ? per_unit / timing.meanUs : 0.0;
    QTA_LOG_INFO("bench",
                 "{}  p50 {} us  p99 {} us  {} ns/unit  {} Munit/s  allocs {}  {} KB  sink {}",
                 name, quantape::format::num(timing.p50Us, 6),
                 quantape::format::num(timing.p99Us, 6), quantape::format::num(nsPerUnit, 3),
                 quantape::format::num(mUnitsPerSecond, 3), timing.allocations,
                 quantape::format::num(static_cast<double>(timing.allocatedBytes) / 1024.0, 3),
                 quantape::format::num(static_cast<double>(g_sink), 6));
}

struct LinearModel {
    Eigen::MatrixXd driftMatrix;                    // out = driftMatrix * x
    std::vector<Eigen::MatrixXd> diffusionMatrices; // q matrices
    template <typename Scalar>
    void drift(const quantape::mc::StateMatrix<Scalar>& x, double, const std::vector<Scalar>&,
               quantape::mc::StateMatrix<Scalar>& out) const {
        if constexpr (std::is_same_v<Scalar, double>) {
            out.noalias() = driftMatrix * x;
        } else {
            out.noalias() = driftMatrix.template cast<Scalar>() * x;
        }
    }
    template <typename Scalar>
    void diffusion(const quantape::mc::StateMatrix<Scalar>& x, double, const std::vector<Scalar>&,
                   std::size_t factor, quantape::mc::StateMatrix<Scalar>& out) const {
        if constexpr (std::is_same_v<Scalar, double>) {
            out.noalias() = diffusionMatrices[factor] * x;
        } else {
            out.noalias() = diffusionMatrices[factor].template cast<Scalar>() * x;
        }
    }
};

// ── Sobol fixture fallback for machines without a configured table ──

std::vector<quantape::math::mc::sobol::Entry> sobolFixture(int nDims, std::uint64_t seed) {
    std::mt19937_64 rng(seed);
    std::vector<quantape::math::mc::sobol::Entry> out;
    std::uint32_t dim = 2;
    for (int degree = 1; degree <= 10 && static_cast<int>(out.size()) < nDims; ++degree) {
        for (const std::uint64_t poly : quantape::math::mc::gf2::enumerate_primitive(degree)) {
            if (static_cast<int>(out.size()) >= nDims) {
                break;
            }
            quantape::math::mc::sobol::Entry e;
            e.dim = dim++;
            e.s = static_cast<std::uint32_t>(degree);
            e.a = quantape::math::mc::gf2::encode_a(poly, degree);
            e.m.resize(static_cast<std::size_t>(degree));
            for (int k = 1; k <= degree; ++k) {
                e.m[static_cast<std::size_t>(k - 1)] = ((rng() % (1ULL << (k - 1))) << 1) | 1;
            }
            out.push_back(std::move(e));
        }
    }
    return out;
}

// ── Theta-driven GBM (θ = {mu, sigma}) and terminal-call payoff for the
//    pathwise-gradient benches (bundles carry their params in doubles).

struct GbmThetaModel {
    template <typename Scalar>
    void drift(const quantape::mc::StateMatrix<Scalar>& x, double, const std::vector<Scalar>& th,
               quantape::mc::StateMatrix<Scalar>& out) const {
        out = (th[0] * x.array()).matrix();
    }
    template <typename Scalar>
    void diffusion(const quantape::mc::StateMatrix<Scalar>& x, double,
                   const std::vector<Scalar>& th, std::size_t,
                   quantape::mc::StateMatrix<Scalar>& out) const {
        out = (th[1] * x.array()).matrix();
    }
};

struct TerminalCallPayoff {
    double strike = 100.0;
    template <typename Scalar>
    Scalar operator()(const quantape::mc::PathBlock<Scalar>& path) const {
        const Scalar s = path.states.back()(0, 0);
        return s > Scalar(strike) ? s - Scalar(strike) : Scalar(0.0);
    }
};

struct TerminalValuePayoff {
    template <typename Scalar>
    Scalar operator()(const quantape::mc::PathBlock<Scalar>& path) const {
        return path.states.back()(0, 0);
    }
};

// Theta-driven CIR (θ = {kappa, level, sigma}) for QE gradient benches
struct CirThetaModel {
    template <typename Scalar>
    void drift(const quantape::mc::StateMatrix<Scalar>& x, double, const std::vector<Scalar>& th,
               quantape::mc::StateMatrix<Scalar>& out) const {
        out = (th[0] * (th[1] - x.array())).matrix();
    }
    template <typename Scalar>
    void diffusion(const quantape::mc::StateMatrix<Scalar>& x, double,
                   const std::vector<Scalar>& th, std::size_t,
                   quantape::mc::StateMatrix<Scalar>& out) const {
        out = (th[2] * x.array().cwiseMax(Scalar(0.0)).sqrt()).matrix();
    }
};

struct TerminalCallBatchPayoff {
    double strike = 0.0;
    template <typename Scalar>
    void operator()(const quantape::mc::PathBlock<Scalar>& path,
                    Eigen::Matrix<Scalar, Eigen::Dynamic, 1>& out) const {
        const auto& x = path.states.back();
        out.resize(x.cols());
        for (Eigen::Index p = 0; p < x.cols(); ++p) {
            const Scalar s = x(0, p);
            out(p) = s > Scalar(strike) ? s - Scalar(strike) : Scalar(0.0);
        }
    }
};

// Multi-factor diagonal model (HJM-like proxy): d tenors, q factors,
// theta = {kappa, sig_0..sig_{q-1}}, factor loadings sig_j * 0.9^tenor.
struct MultiFactorModel {
    std::size_t dims = 8;
    std::size_t factors = 3;

    template <typename Scalar>
    void drift(const quantape::mc::StateMatrix<Scalar>& x, double, const std::vector<Scalar>& th,
               quantape::mc::StateMatrix<Scalar>& out) const {
        out = (-th[0] * x.array()).matrix();
    }
    template <typename Scalar>
    void diffusion(const quantape::mc::StateMatrix<Scalar>& x, double,
                   const std::vector<Scalar>& th, std::size_t j,
                   quantape::mc::StateMatrix<Scalar>& out) const {
        out.resize(x.rows(), x.cols());
        for (Eigen::Index i = 0; i < x.rows(); ++i) {
            // Constant loadings as plain doubles: constructing AD leaves for
            // constants inflates the tape (model-side style matters).
            const double load = std::pow(0.9, static_cast<double>(i));
            out.row(i) = (th[1 + j] * (x.row(i).array() * load)).matrix();
        }
    }
};

struct CirThetaMoments {
    template <typename Scalar>
    void mean(const quantape::mc::StateMatrix<Scalar>& x, double, double dt,
              const std::vector<Scalar>& th, quantape::mc::StateMatrix<Scalar>& out) const {
        using std::exp;
        const Scalar decay = exp(-th[0] * Scalar(dt));
        out = x * decay;
        out.array() += (th[1] * (Scalar(1.0) - decay));
    }
    template <typename Scalar>
    void variance(const quantape::mc::StateMatrix<Scalar>& x, double, double dt,
                  const std::vector<Scalar>& th, quantape::mc::StateMatrix<Scalar>& out) const {
        using std::exp;
        const Scalar decay = exp(-th[0] * Scalar(dt));
        const Scalar oneMinus = Scalar(1.0) - decay;
        out = x * (th[2] * th[2] * decay * oneMinus / th[0]);
        out.array() += (th[1] * th[2] * th[2] * oneMinus * oneMinus / (Scalar(2.0) * th[0]));
    }
};

} // namespace

int main(int argc, char** argv) {
    const int reps = argc > 1 ? std::atoi(argv[1]) : 20;
    const bool onlyGrad = argc > 2 && std::strcmp(argv[2], "grad") == 0;

    // Lean reverse-tape only (profiling target).
    if (argc > 2 && std::strcmp(argv[2], "lean") == 0) {
        const std::size_t nPaths = 20000;
        const std::size_t nSteps = 252;
        const TimeGrid grid(1.0, nSteps);
        const std::vector<double> theta = {0.05, 0.2};
        const std::vector<std::vector<double>> thetaSteps(nSteps, theta);
        const auto x0 = Eigen::VectorXd::Constant(1, 100.0);
        const GbmThetaModel model;
        {
            const SdeSimulator<double, Euler> simulator(grid, thetaSteps);
            const IidGaussianSource<> source(1, 7);
            bench_us("lean euler gbm 20kx252", double(nPaths), reps, [&] {
                const auto g = quantape::mc::simulateGradientLean(
                    simulator, x0, theta, quantape::mc::driftOf(model),
                    quantape::mc::diffusionOf(model), source, TerminalCallPayoff{100.0}, nPaths,
                    Schedule::Parallel);
                return g.value + g.gradient(0);
            });
        }
        {
            const SdeSimulator<double, Milstein> simulator(grid, thetaSteps);
            const IidGaussianSource<> source(1, 7);
            bench_us("lean milstein gbm 20kx252", double(nPaths), reps, [&] {
                const auto g = quantape::mc::simulateGradientLean(
                    simulator, x0, theta, quantape::mc::driftOf(model),
                    quantape::mc::diffusionOf(model), source, TerminalCallPayoff{100.0}, nPaths,
                    Schedule::Parallel);
                return g.value + g.gradient(0);
            });
        }
        return 0;
    }

    // Multiprocessing shard mode: one process per core, single-threaded.
    if (argc > 4 && std::strcmp(argv[2], "shard") == 0) {
        const std::size_t shardIndex = static_cast<std::size_t>(std::atoi(argv[3]));
        const std::size_t shardCount = static_cast<std::size_t>(std::atoi(argv[4]));
        const std::size_t totalPaths = 20000;
        const std::size_t nSteps = 252;
        const std::size_t chunk = (totalPaths + shardCount - 1) / shardCount;
        const std::size_t begin = std::min(shardIndex * chunk, totalPaths);
        const std::size_t count = std::min(chunk, totalPaths - begin);
        const TimeGrid grid(1.0, nSteps);
        const std::vector<double> theta = {0.05, 0.2};
        const std::vector<std::vector<double>> thetaSteps(nSteps, theta);
        const SdeSimulator<double, Euler> simulator(grid, thetaSteps);
        const IidGaussianSource<> source(1, 7);
        const auto x0 = Eigen::VectorXd::Constant(1, 100.0);
        QTA_LOG_INFO("bench", "shard {}/{}: paths [{}, {})", shardIndex, shardCount, begin,
                     begin + count);
        bench_us("gradient euler gbm shard (sequential)", double(count), reps, [&] {
            const auto s = quantape::mc::simulateGradientSamples(
                simulator, x0, theta, quantape::mc::driftOf(GbmThetaModel{}),
                quantape::mc::diffusionOf(GbmThetaModel{}), source, TerminalCallPayoff{100.0},
                begin, count, Schedule::Sequential);
            return s.values[0] + s.gradients(0, 0);
        });
        return 0;
    }

    QTA_LOG_INFO("bench", "SDE engine benchmark, reps={}{}", reps,
                 onlyGrad ? " (gradients only)" : "");

    if (!onlyGrad) {
        // ── 1. Source: keyed draws ──
        {
            const std::size_t nPaths = 65536;
            IidGaussianSource<> source(2, 1234);
            Eigen::MatrixXd z;
            bench_us("source fill q=2, 65536 paths", static_cast<double>(nPaths * 2), reps, [&] {
                source.fill(0, 0, nPaths, z);
                return z(0, 0) + z(1, static_cast<Eigen::Index>(nPaths - 1));
            });
            bench_us("source fillUniform 65536 paths", static_cast<double>(nPaths), reps, [&] {
                source.fillUniform(0, 0, nPaths, 0, 1, z);
                return z(0, 0) + z(0, static_cast<Eigen::Index>(nPaths - 1));
            });
        }

        // ── 2. Euler GBM streamed (true engine cost) + block sizes ──
        {
            const GbmProcess model{0.05, 0.2};
            const std::size_t nPaths = 100000;
            const std::size_t nSteps = 252;
            const TimeGrid grid(1.0, nSteps);
            std::vector<std::vector<double>> theta(nSteps);
            const auto x0 = Eigen::VectorXd::Constant(1, 100.0);
            for (std::size_t blockSize :
                 {std::size_t(1024), std::size_t(4096), std::size_t(16384)}) {
                const SdeSimulator<double> simulator(grid, theta);
                const IidGaussianSource<> source(1, 7);
                const std::string name =
                    "euler gbm stream 100k x 252 (block " + std::to_string(blockSize) + ")";
                bench_us(name, static_cast<double>(nPaths * nSteps), reps, [&] {
                    double checksum = 0.0;
                    simulator.simulateBlocks(
                        x0, quantape::mc::driftOf(model), quantape::mc::diffusionOf(model), source,
                        nPaths, blockSize,
                        [&checksum](const quantape::mc::PathBlock<double>& block, std::size_t) {
                            checksum += block.states.back()(0, 0);
                        });
                    return checksum;
                });
            }
            {
                const SdeSimulator<double> simulator(grid, theta);
                const IidGaussianSource<> source(1, 7);
                const std::size_t blockSize = 1024;
                const std::size_t nBlocks = (nPaths + blockSize - 1) / blockSize;
                std::vector<double> partials(nBlocks, 0.0);
                bench_us("euler gbm stream 100k x 252 (parallel)",
                         static_cast<double>(nPaths * nSteps), reps, [&] {
                             simulator.simulateBlocks(
                                 x0, quantape::mc::driftOf(model), quantape::mc::diffusionOf(model),
                                 source, nPaths, blockSize,
                                 [&partials](const quantape::mc::PathBlock<double>& block,
                                             std::size_t blockIndex) {
                                     partials[blockIndex] = block.states.back()(0, 0);
                                 },
                                 Schedule::Parallel);
                             double checksum = 0.0;
                             for (double v : partials) {
                                 checksum += v;
                             }
                             return checksum;
                         });
            }
        }

        // ── 3. Parallel schedule scaling (collecting, small enough to fit) ──
        {
            const GbmProcess model{0.05, 0.2};
            const std::size_t nPaths = 20000;
            const std::size_t nSteps = 252;
            const TimeGrid grid(1.0, nSteps);
            std::vector<std::vector<double>> theta(nSteps);
            const auto x0 = Eigen::VectorXd::Constant(1, 100.0);
            const SdeSimulator<double> simulator(grid, theta);
            for (Schedule schedule : {Schedule::Sequential, Schedule::Parallel}) {
                const IidGaussianSource<> source(1, 7);
                bench_us(schedule == Schedule::Parallel ? "euler gbm 20k x 252 (parallel)"
                                                        : "euler gbm 20k x 252 (sequential)",
                         static_cast<double>(nPaths * nSteps), reps, [&] {
                             const auto blocks = simulator.simulate(
                                 x0, quantape::mc::driftOf(model), quantape::mc::diffusionOf(model),
                                 source, nPaths, 4096, schedule);
                             return blocks.front().states.back()(0, 0);
                         });
            }
        }

        // ── 4. Heston QE (2 factors + uniform), streamed + parallel ──
        {
            const HestonProcess model{0.02, 2.0, 0.04, 0.3, -0.7};
            const auto x0 = (Eigen::Vector2d() << std::log(100.0), 0.04).finished();
            const std::size_t nSteps = 252;
            const TimeGrid grid(1.0, nSteps);
            std::vector<std::vector<double>> theta(nSteps);
            {
                const std::size_t nPaths = 50000;
                const SdeSimulator<double, HestonQeProcess<HestonProcess>> simulator(
                    grid, theta, HestonQeProcess<HestonProcess>{model});
                const IidGaussianSource<> source(2, 9);
                bench_us(
                    "heston qe stream 50k x 252", static_cast<double>(nPaths * nSteps), reps, [&] {
                        double checksum = 0.0;
                        simulator.simulateBlocks(
                            x0, quantape::mc::driftOf(model), quantape::mc::diffusionOf(model),
                            source, nPaths, 4096,
                            [&checksum](const quantape::mc::PathBlock<double>& block, std::size_t) {
                                checksum += block.states.back()(1, 0);
                            });
                        return checksum;
                    });
                bench_us(
                    "heston qe stream 50k x 252 (parallel)", static_cast<double>(nPaths * nSteps),
                    reps, [&] {
                        double checksum = 0.0;
                        simulator.simulateBlocks(
                            x0, quantape::mc::driftOf(model), quantape::mc::diffusionOf(model),
                            source, nPaths, 4096,
                            [&checksum](const quantape::mc::PathBlock<double>& block, std::size_t) {
                                checksum += block.states.back()(1, 0);
                            },
                            Schedule::Parallel);
                        return checksum;
                    });
            }
            {
                const std::size_t nPaths = 10000;
                const SdeSimulator<double, HestonQeProcess<HestonProcess>> simulator(
                    grid, theta, HestonQeProcess<HestonProcess>{model});
                const IidGaussianSource<> source(2, 9);
                bench_us("heston qe 10k x 252 (parallel)", static_cast<double>(nPaths * nSteps),
                         reps, [&] {
                             const auto blocks = simulator.simulate(
                                 x0, quantape::mc::driftOf(model), quantape::mc::diffusionOf(model),
                                 source, nPaths, 2048, Schedule::Parallel);
                             return blocks.front().states.back()(1, 0);
                         });
            }
        }

        // ── 5. Estimator (sequential vs parallel over blocks) ──
        {
            const GbmProcess model{0.0, 0.2};
            const std::size_t nPaths = 200000;
            const std::size_t nSteps = 64;
            const TimeGrid grid(1.0, nSteps);
            std::vector<std::vector<double>> theta(nSteps);
            const SdeSimulator<double> simulator(grid, theta);
            const IidGaussianSource<> source(1, 11);
            const auto blocks = simulator.simulate(
                Eigen::VectorXd::Constant(1, 100.0), quantape::mc::driftOf(model),
                quantape::mc::diffusionOf(model), source, nPaths, 8192);
            const auto payoff = [](const quantape::mc::PathBlock<double>& b, Eigen::VectorXd& out) {
                const auto& x = b.states.back();
                out = (x.row(0).array() - 100.0).max(0.0).matrix().transpose();
            };
            bench_us("estimator 200k (sequential)", static_cast<double>(nPaths), reps, [&] {
                const auto e = quantape::mc::estimate(blocks, payoff, Schedule::Sequential);
                return e.mean + e.stdError;
            });
            bench_us("estimator 200k (parallel)", static_cast<double>(nPaths), reps, [&] {
                const auto e = quantape::mc::estimate(blocks, payoff, Schedule::Parallel);
                return e.mean + e.stdError;
            });
        }

        // ── 6. Multi-dimensional linear model ──
        {
            const std::size_t dims = 64;
            const std::size_t factors = 8;
            const std::size_t nPaths = 2048;
            const std::size_t nSteps = 32;
            LinearModel model;
            model.driftMatrix = Eigen::MatrixXd::Random(static_cast<Eigen::Index>(dims),
                                                        static_cast<Eigen::Index>(dims)) *
                                0.1;
            model.diffusionMatrices.resize(factors);
            for (auto& m : model.diffusionMatrices) {
                m = Eigen::MatrixXd::Random(static_cast<Eigen::Index>(dims),
                                            static_cast<Eigen::Index>(dims)) *
                    0.1;
            }
            const TimeGrid grid(1.0, nSteps);
            std::vector<std::vector<double>> theta(nSteps);
            const SdeSimulator<double> simulator(grid, theta);
            const IidGaussianSource<> source(factors, 13);
            const auto x0 = Eigen::VectorXd::Constant(static_cast<Eigen::Index>(dims), 1.0);
            bench_us("linear model d=64 q=8 stream 2048 x 32", static_cast<double>(nPaths * nSteps),
                     reps, [&] {
                         double checksum = 0.0;
                         simulator.simulateBlocks(
                             x0, quantape::mc::driftOf(model), quantape::mc::diffusionOf(model),
                             source, nPaths, 512,
                             [&checksum](const quantape::mc::PathBlock<double>& block,
                                         std::size_t) { checksum += block.states.back()(0, 0); });
                         return checksum;
                     });
        }

    } // !onlyGrad

    // ── 7. Pathwise gradients (per-path var tapes, S5a/S5b) ──
    {
        const char* envPaths = std::getenv("BENCH_GRAD_PATHS");
        const std::size_t nPaths =
            envPaths != nullptr ? static_cast<std::size_t>(std::strtoul(envPaths, nullptr, 10))
                                : 20000;
        const std::size_t nSteps = 252;
        const std::size_t blockSize = 1024;
        const TimeGrid grid(1.0, nSteps);
        const std::vector<double> theta = {0.05, 0.2};
        const std::vector<std::vector<double>> thetaSteps(nSteps, theta);
        const auto x0 = Eigen::VectorXd::Constant(1, 100.0);
        const TerminalCallPayoff payoff{100.0};
        const GbmThetaModel model;
        const std::size_t nBlocks = (nPaths + blockSize - 1) / blockSize;

        // double-engine reference: streamed parallel, same config
        {
            const SdeSimulator<double, Euler> simulator(grid, thetaSteps);
            const IidGaussianSource<> source(1, 7);
            const std::string name =
                "double euler stream " + std::to_string(nPaths) + "x252 (parallel)";
            bench_us(name, double(nPaths * nSteps), reps, [&] {
                std::vector<double> partials(nBlocks, 0.0);
                simulator.simulateBlocks(
                    x0, quantape::mc::driftOf(model), quantape::mc::diffusionOf(model), source,
                    nPaths, blockSize,
                    [&partials](const quantape::mc::PathBlock<double>& block, std::size_t index) {
                        partials[index] = block.states.back()(0, 0);
                    },
                    Schedule::Parallel);
                double checksum = 0.0;
                for (double v : partials) {
                    checksum += v;
                }
                return checksum;
            });
        }
        {
            const SdeSimulator<double, Euler> simulator(grid, thetaSteps);
            const IidGaussianSource<> source(1, 7);
            const std::string name =
                "gradient euler gbm " + std::to_string(nPaths) + "x252 (parallel)";
            bench_us(name, double(nPaths), reps, [&] {
                const auto g = quantape::mc::simulateGradient(
                    simulator, x0, theta, quantape::mc::driftOf(model),
                    quantape::mc::diffusionOf(model), source, payoff, nPaths, Schedule::Parallel);
                return g.value + g.gradient(0) + g.gradient(2);
            });
        }
        {
            const SdeSimulator<double, Milstein> simulator(grid, thetaSteps);
            const IidGaussianSource<> source(1, 7);
            const std::string name =
                "gradient milstein gbm " + std::to_string(nPaths) + "x252 (parallel)";
            bench_us(name, double(nPaths), reps, [&] {
                const auto g = quantape::mc::simulateGradient(
                    simulator, x0, theta, quantape::mc::driftOf(model),
                    quantape::mc::diffusionOf(model), source, payoff, nPaths, Schedule::Parallel);
                return g.value + g.gradient(0) + g.gradient(2);
            });
        }
        {
            const SdeSimulator<double, Euler> simulator(grid, thetaSteps);
            const IidGaussianSource<> source(1, 7);
            const std::string name =
                "gradient euler gbm " + std::to_string(nPaths) + "x252 (forward, N=3)";
            bench_us(name, double(nPaths), reps, [&] {
                const auto g = quantape::mc::simulateGradientForward<3>(
                    simulator, x0, theta, quantape::mc::driftOf(model),
                    quantape::mc::diffusionOf(model), source, payoff, nPaths, Schedule::Parallel);
                return g.value + g.gradient(0) + g.gradient(2);
            });
        }
        {
            const SdeSimulator<double, Euler> simulator(grid, thetaSteps);
            const IidGaussianSource<> source(1, 7);
            const std::string name =
                "gradient euler gbm " + std::to_string(nPaths) + "x252 (stan fvar fwd)";
            bench_us(name, double(nPaths), reps, [&] {
                const auto g = quantape::mc::simulateGradientStanForward(
                    simulator, x0, theta, quantape::mc::driftOf(model),
                    quantape::mc::diffusionOf(model), source, payoff, nPaths, Schedule::Parallel);
                return g.value + g.gradient(0) + g.gradient(2);
            });
        }
        {
            const SdeSimulator<double, Milstein> simulator(grid, thetaSteps);
            const IidGaussianSource<> source(1, 7);
            const std::string name =
                "gradient milstein gbm " + std::to_string(nPaths) + "x252 (forward, N=3)";
            bench_us(name, double(nPaths), reps, [&] {
                const auto g = quantape::mc::simulateGradientForward<3>(
                    simulator, x0, theta, quantape::mc::driftOf(model),
                    quantape::mc::diffusionOf(model), source, payoff, nPaths, Schedule::Parallel);
                return g.value + g.gradient(0) + g.gradient(2);
            });
        }
        {
            const SdeSimulator<double, Euler> simulator(grid, thetaSteps);
            const IidGaussianSource<> source(1, 7);
            const std::string name =
                "gradient euler gbm " + std::to_string(nPaths) + "x252 (forward block64)";
            bench_us(name, double(nPaths), reps, [&] {
                const auto g = quantape::mc::simulateGradientForwardBlock<3>(
                    simulator, x0, theta, quantape::mc::driftOf(model),
                    quantape::mc::diffusionOf(model), source, TerminalCallBatchPayoff{100.0},
                    nPaths, 64, Schedule::Parallel);
                return g.value + g.gradient(0) + g.gradient(2);
            });
        }
        {
            const std::size_t cirPaths = 10000;
            const std::size_t cirSteps = 252;
            const TimeGrid cirGrid(1.0, cirSteps);
            const std::vector<std::vector<double>> cirThetaSteps(cirSteps);
            const SdeSimulator<double, quantape::mc::MomentMatching1D<CirThetaMoments>> simulator(
                cirGrid, cirThetaSteps, quantape::mc::MomentMatching1D<CirThetaMoments>{});
            const IidGaussianSource<> source(1, 9);
            const std::vector<double> cirTheta = {2.0, 0.04, 0.2};
            Eigen::VectorXd v0(1);
            v0(0) = 0.04;
            bench_us("gradient cir qe 10kx252 (forward, N=4)", double(cirPaths), reps, [&] {
                const auto g = quantape::mc::simulateGradientForward<4>(
                    simulator, v0, cirTheta, quantape::mc::driftOf(CirThetaModel{}),
                    quantape::mc::diffusionOf(CirThetaModel{}), source, TerminalValuePayoff{},
                    cirPaths, Schedule::Parallel);
                return g.value + g.gradient(1);
            });
        }
        {
            const SdeSimulator<double, Euler> simulator(grid, thetaSteps);
            const IidGaussianSource<> source(1, 7);
            const std::string name =
                "gradient euler gbm " + std::to_string(nPaths) + "x252 (lean reverse)";
            bench_us(name, double(nPaths), reps, [&] {
                const auto g = quantape::mc::simulateGradientLean(
                    simulator, x0, theta, quantape::mc::driftOf(model),
                    quantape::mc::diffusionOf(model), source, payoff, nPaths, Schedule::Parallel);
                return g.value + g.gradient(0) + g.gradient(2);
            });
        }
        {
            const SdeSimulator<double, Milstein> simulator(grid, thetaSteps);
            const IidGaussianSource<> source(1, 7);
            const std::string name =
                "gradient milstein gbm " + std::to_string(nPaths) + "x252 (lean reverse)";
            bench_us(name, double(nPaths), reps, [&] {
                const auto g = quantape::mc::simulateGradientLean(
                    simulator, x0, theta, quantape::mc::driftOf(model),
                    quantape::mc::diffusionOf(model), source, payoff, nPaths, Schedule::Parallel);
                return g.value + g.gradient(0) + g.gradient(2);
            });
        }
        {
            const SdeSimulator<double, Euler> simulator(grid, thetaSteps);
            const IidGaussianSource<> source(1, 7);
            const std::string name =
                "gradient euler gbm " + std::to_string(nPaths) + "x252 (checkpointed)";
            bench_us(name, double(nPaths), reps, [&] {
                const auto g = quantape::mc::simulateGradientCheckpointed(
                    simulator, x0, theta, quantape::mc::driftOf(model),
                    quantape::mc::diffusionOf(model), source, payoff, nPaths, Schedule::Parallel);
                return g.value + g.gradient(0) + g.gradient(2);
            });
        }
        {
            const std::size_t longPaths = 2000;
            const std::size_t longSteps = 4096;
            const TimeGrid longGrid(1.0, longSteps);
            const std::vector<std::vector<double>> longThetaSteps(longSteps, theta);
            const SdeSimulator<double, Euler> simulator(longGrid, longThetaSteps);
            const IidGaussianSource<> source(1, 7);
            const auto longX0 = Eigen::VectorXd::Constant(1, 100.0);
            bench_us("gradient euler gbm 2k x 4096 (checkpointed)", double(longPaths), reps, [&] {
                const auto g = quantape::mc::simulateGradientCheckpointed(
                    simulator, longX0, theta, quantape::mc::driftOf(model),
                    quantape::mc::diffusionOf(model), source, payoff, longPaths,
                    Schedule::Parallel);
                return g.value + g.gradient(0) + g.gradient(2);
            });
        }

        // Tape scope per path: nodes on the nested stack after one path
        {
            const SdeSimulator<double, Euler> simulator(grid, thetaSteps);
            const IidGaussianSource<> source(1, 7);
            stan::math::nested_rev_autodiff scope;
            std::vector<stan::math::var> thetaVars = {theta[0], theta[1]};
            Eigen::Matrix<stan::math::var, Eigen::Dynamic, 1> x0Vars(1);
            x0Vars(0) = x0(0);
            const auto path = simulator.template simulatePathSharedTheta<stan::math::var>(
                x0Vars, quantape::mc::driftOf(model), quantape::mc::diffusionOf(model), source, 0,
                thetaVars);
            stan::math::var y = payoff.template operator()<stan::math::var>(path);
            const std::size_t nodes = stan::math::nested_size();
            QTA_LOG_INFO("bench", "tape nodes/path (euler gbm 252 steps): {} (~{} KB at 64 B/node)",
                         nodes,
                         quantape::format::num(static_cast<double>(nodes) * 64.0 / 1024.0, 4));
            y.grad();
        }
        {
            const SdeSimulator<double, Milstein> simulator(grid, thetaSteps);
            const IidGaussianSource<> source(1, 7);
            stan::math::nested_rev_autodiff scope;
            std::vector<stan::math::var> thetaVars = {theta[0], theta[1]};
            Eigen::Matrix<stan::math::var, Eigen::Dynamic, 1> x0Vars(1);
            x0Vars(0) = x0(0);
            const auto path = simulator.template simulatePathSharedTheta<stan::math::var>(
                x0Vars, quantape::mc::driftOf(model), quantape::mc::diffusionOf(model), source, 0,
                thetaVars);
            stan::math::var y = payoff.template operator()<stan::math::var>(path);
            const std::size_t nodes = stan::math::nested_size();
            QTA_LOG_INFO(
                "bench", "tape nodes/path (milstein gbm 252 steps): {} (~{} KB at 64 B/node)",
                nodes, quantape::format::num(static_cast<double>(nodes) * 64.0 / 1024.0, 4));
            y.grad();
        }
        {
            {
                // forward-mode state derivatives (S5c optimized): one dual pass
                const std::size_t derivPathsFwd = 5000;
                const SdeSimulator<double, Euler> simulator(grid, thetaSteps);
                const IidGaussianSource<> source(1, 7);
                bench_us("state derivatives euler gbm 5kx252 (forward)", double(derivPathsFwd),
                         reps, [&] {
                             double checksum = 0.0;
                             const auto sink =
                                 [&checksum](std::size_t,
                                             const quantape::mc::StateMatrix<double>& dX) {
                                     checksum += dX(0, 0);
                                 };
                             for (std::size_t i = 0; i < derivPathsFwd; ++i) {
                                 quantape::mc::simulatePathDerivativesForward<3>(
                                     simulator, x0, theta, quantape::mc::driftOf(model),
                                     quantape::mc::diffusionOf(model), source, i, sink);
                             }
                             return checksum;
                         });
            }
            // state-derivative pass (S5c): (d + p) forward sweeps per path,
            // streamed per step; sequential here (callers parallelize paths)
            const std::size_t derivPaths = 5000;
            const SdeSimulator<double, Euler> simulator(grid, thetaSteps);
            const IidGaussianSource<> source(1, 7);
            bench_us("state derivatives euler gbm 5k x 252", double(derivPaths), reps, [&] {
                double checksum = 0.0;
                const auto sink = [&checksum](std::size_t,
                                              const quantape::mc::StateMatrix<double>& dX) {
                    checksum += dX(0, 0);
                };
                for (std::size_t i = 0; i < derivPaths; ++i) {
                    quantape::mc::simulatePathDerivatives(
                        simulator, x0, theta, quantape::mc::driftOf(model),
                        quantape::mc::diffusionOf(model), source, i, sink);
                }
                return checksum;
            });
        }
        {
            // one-step tape of the checkpointed gluing (cache-resident)
            const SdeSimulator<double, Euler> simulator(grid, thetaSteps);
            const IidGaussianSource<> source(1, 7);
            stan::math::nested_rev_autodiff scope;
            std::vector<stan::math::var> thetaVars = {theta[0], theta[1]};
            quantape::mc::StateMatrix<stan::math::var> x(1, 1);
            quantape::mc::StateMatrix<stan::math::var> xNext(1, 1);
            x(0, 0) = x0(0);
            Eigen::MatrixXd z;
            Eigen::MatrixXd uniforms;
            Euler::Scratch<stan::math::var> scratch(1, 1);
            simulator.template stepPath<stan::math::var>(
                x, xNext, 0, 0, source, quantape::mc::driftOf(model),
                quantape::mc::diffusionOf(model), thetaVars, z, uniforms, scratch);
            QTA_LOG_INFO("bench", "checkpointed step tape nodes: {} (~{} KB at 64 B/node)",
                         stan::math::nested_size(),
                         quantape::format::num(
                             static_cast<double>(stan::math::nested_size()) * 64.0 / 1024.0, 4));
        }
    }

    // ── 8. Many-factor reverse-vs-forward benchmark (HJM-like proxy) ──
    if (!onlyGrad) {
        const std::size_t dims = 8;
        const std::size_t factors = 3;
        const std::size_t nPaths = 2000;
        const std::size_t nSteps = 252;
        const TimeGrid grid(1.0, nSteps);
        const std::vector<double> theta = {0.5, 0.02, 0.01, 0.005};
        const std::vector<std::vector<double>> thetaSteps(nSteps, theta);
        Eigen::VectorXd x0 = Eigen::VectorXd::Constant(static_cast<Eigen::Index>(dims), 0.03);
        const MultiFactorModel model;

        {
            const SdeSimulator<double, Euler> simulator(grid, thetaSteps);
            const IidGaussianSource<> source(factors, 7);
            const std::string name = "mf d=" + std::to_string(dims) +
                                     " q=" + std::to_string(factors) + " double stream 2kx252";
            bench_us(name, double(nPaths * nSteps), reps, [&] {
                double checksum = 0.0;
                simulator.simulateBlocks(
                    x0, quantape::mc::driftOf(model), quantape::mc::diffusionOf(model), source,
                    nPaths, 256,
                    [&checksum](const quantape::mc::PathBlock<double>& block, std::size_t) {
                        checksum += block.states.back()(0, 0);
                    },
                    Schedule::Parallel);
                return checksum;
            });
        }
        {
            const SdeSimulator<double, Euler> simulator(grid, thetaSteps);
            const IidGaussianSource<> source(factors, 7);
            bench_us("mf d=8 q=3 gradient lean reverse 2kx252", double(nPaths), reps, [&] {
                const auto g = quantape::mc::simulateGradientLean(
                    simulator, x0, theta, quantape::mc::driftOf(model),
                    quantape::mc::diffusionOf(model), source, TerminalCallPayoff{0.018}, nPaths,
                    Schedule::Parallel);
                return g.value + g.gradient(0);
            });
            const std::size_t leanNodes = quantape::mc::leanPathTapeNodes(
                simulator, x0, theta, quantape::mc::driftOf(model),
                quantape::mc::diffusionOf(model), source, TerminalCallPayoff{0.018}, 0);
            QTA_LOG_INFO("bench", "mf lean tape nodes/path (252 steps): {} (~{} KB at 48 B/node)",
                         leanNodes,
                         quantape::format::num(static_cast<double>(leanNodes) * 48.0 / 1024.0, 4));
        }
        {
            const SdeSimulator<double, Euler> simulator(grid, thetaSteps);
            const IidGaussianSource<> source(factors, 7);
            bench_us("mf d=8 q=3 gradient reverse 2kx252", double(nPaths), reps, [&] {
                const auto g = quantape::mc::simulateGradient(
                    simulator, x0, theta, quantape::mc::driftOf(model),
                    quantape::mc::diffusionOf(model), source, TerminalCallPayoff{0.018}, nPaths,
                    Schedule::Parallel);
                return g.value + g.gradient(0);
            });
        }
        {
            const SdeSimulator<double, Euler> simulator(grid, thetaSteps);
            const IidGaussianSource<> source(factors, 7);
            bench_us("mf d=8 q=3 gradient checkpointed 2kx252", double(nPaths), reps, [&] {
                const auto g = quantape::mc::simulateGradientCheckpointed(
                    simulator, x0, theta, quantape::mc::driftOf(model),
                    quantape::mc::diffusionOf(model), source, TerminalCallPayoff{0.018}, nPaths,
                    Schedule::Parallel);
                return g.value + g.gradient(0);
            });
        }
        {
            const SdeSimulator<double, Euler> simulator(grid, thetaSteps);
            const IidGaussianSource<> source(factors, 7);
            bench_us("mf d=8 q=3 gradient forward N=12 per-path 2kx252", double(nPaths), reps, [&] {
                const auto g = quantape::mc::simulateGradientForward<12>(
                    simulator, x0, theta, quantape::mc::driftOf(model),
                    quantape::mc::diffusionOf(model), source, TerminalCallPayoff{0.018}, nPaths,
                    Schedule::Parallel);
                return g.value + g.gradient(0);
            });
        }
        {
            const SdeSimulator<double, Euler> simulator(grid, thetaSteps);
            const IidGaussianSource<> source(factors, 7);
            bench_us("mf d=8 q=3 gradient forward N=12 block64 2kx252", double(nPaths), reps, [&] {
                const auto g = quantape::mc::simulateGradientForwardBlock<12>(
                    simulator, x0, theta, quantape::mc::driftOf(model),
                    quantape::mc::diffusionOf(model), source, TerminalCallBatchPayoff{0.018},
                    nPaths, 64, Schedule::Parallel);
                return g.value + g.gradient(0);
            });
        }
    }

    // ── 9. Sobol/QMC source: generation cost + engine cost ──
    if (!onlyGrad) {
        const std::size_t nPaths = 20000;
        const std::size_t nSteps = 252;
        const std::vector<double> theta = {0.05, 0.2};
        const std::vector<std::vector<double>> thetaSteps(nSteps, theta);
        const TimeGrid grid(1.0, nSteps);
        const auto x0 = Eigen::VectorXd::Constant(1, 100.0);
        const GbmThetaModel model;
        const TerminalCallPayoff payoff{100.0};

        std::shared_ptr<const quantape::math::mc::sobol::SobolGenerator> generator;
        const std::string tablePath = quantape::math::mc::sobol::SobolGenerator::defaultTablePath();
        if (!tablePath.empty()) {
            generator = quantape::math::mc::sobol::SobolGenerator::sharedFromDefaultTable();
        } else {
            generator = std::make_shared<const quantape::math::mc::sobol::SobolGenerator>(
                sobolFixture(1024, 20240927));
        }
        const quantape::mc::SobolSource sobolSource(generator, 1, nSteps, 0);

        QTA_LOG_INFO("bench", "sobol source: {}, {} dims",
                     tablePath.empty() ? "fixture" : "configured table",
                     generator->preparedDimension());
        {
            Eigen::MatrixXd z;
            bench_us("sobol source fill q=1 20k paths", double(nPaths), reps, [&] {
                sobolSource.fill(0, 0, nPaths, z);
                return z(0, static_cast<Eigen::Index>(nPaths - 1));
            });
        }
        {
            const SdeSimulator<double, Euler> simulator(grid, thetaSteps);
            bench_us("euler gbm 20k x 252 sobol stream (seq)", double(nPaths * nSteps), reps, [&] {
                double checksum = 0.0;
                simulator.simulateBlocks(
                    x0, quantape::mc::driftOf(model), quantape::mc::diffusionOf(model), sobolSource,
                    nPaths, 1024,
                    [&checksum](const quantape::mc::PathBlock<double>& block, std::size_t) {
                        checksum += block.states.back()(0, 0);
                    },
                    Schedule::Sequential);
                return checksum;
            });
        }
        {
            const SdeSimulator<double, Euler> simulator(grid, thetaSteps);
            bench_us("euler gbm 20k x 252 sobol stream (parallel)", double(nPaths * nSteps), reps,
                     [&] {
                         double checksum = 0.0;
                         simulator.simulateBlocks(
                             x0, quantape::mc::driftOf(model), quantape::mc::diffusionOf(model),
                             sobolSource, nPaths, 1024,
                             [&checksum](const quantape::mc::PathBlock<double>& block,
                                         std::size_t) { checksum += block.states.back()(0, 0); },
                             Schedule::Parallel);
                         return checksum;
                     });
        }
        {
            const SdeSimulator<double, Euler> simulator(grid, thetaSteps);
            bench_us("gradient euler gbm 20k x 252 (sobol)", double(nPaths), reps, [&] {
                const auto g = quantape::mc::simulateGradient(
                    simulator, x0, theta, quantape::mc::driftOf(model),
                    quantape::mc::diffusionOf(model), sobolSource, payoff, nPaths,
                    Schedule::Parallel);
                return g.value + g.gradient(0);
            });
        }
    }

    return 0;
}
