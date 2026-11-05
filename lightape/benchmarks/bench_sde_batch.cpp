// bench_sde_batch.cpp — lightape re-implementation of the quantape SDE
// pathwise-gradient benchmark (`benchmarks/bench_sde.cpp`).
//
// Same workload as the quantape "gradient euler gbm 20kx252" case: Euler-
// Maruyama GBM dX = r X dt + sigma X dW, X0 = 100, r = 0.05, sigma = 0.2,
// T = 1 on the quantape TimeGrid (per-step dt_k = t_{k+1} - t_k), payoff
// max(X_T - 100, 0), 20 000 paths, one factor, seed 7 (quantape's keyed
// IidGaussianSource, so the modes here and the quantape baseline see exactly
// the same draws).
//
// Three recording strategies:
//   scalar — one lightape::RevScalar tape per path (252 Euler steps),
//            reversed once; the per-path adjoints are accumulated.
//   batch  — lightape::batch::BatchTape<W> over W paths as lanes: one tape
//            records W paths with a shared (broadcast) parameter Input;
//            one reverse sweep with a per-lane seed of 1 yields the
//            per-lane theta/x0 adjoints, summed over lanes per batch.
// Modes are timed single-threaded and on a plain std::thread pool sized to
// std::thread::hardware_concurrency() (main thread participates), no TBB.
//
// Build (quantape Release bench flags; -march=native is a one-off probe
// allowance, the committed lightape benchmarks stay portable):
//   clang++ -O3 -DNDEBUG -std=gnu++20 -march=native -ffp-contract=off \
//     -I lightape/include -I include -isystem third-party/eigen -pthread \
//     lightape/benchmarks/bench_sde_batch.cpp -o bench_sde_batch
//
// The quantape headers are used header-only (IidGaussianSource + McFarland
// sampler); no quantape library is linked.
#include "quantape/mc/RandomSource.h"

#include <Eigen/Dense>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <functional>
#include <mutex>
#include <thread>
#include <vector>

#include "lightape/batch/BatchTape.h"

namespace {

using lightape::BatchScalar;
using lightape::BatchTape;
using lightape::RevScalar;
using lightape::RevTape;
using quantape::mc::IidGaussianSource;

// ── quantape bench_sde.cpp configuration (Euler GBM pathwise gradient) ──
constexpr std::size_t kPaths = 20000;
constexpr std::size_t kSteps = 252;
constexpr double kT = 1.0;
constexpr double kR = 0.05;
constexpr double kSigma = 0.2;
constexpr double kX0 = 100.0;
constexpr double kStrike = 100.0;
constexpr std::uint64_t kSeed = 7;
constexpr std::size_t kFactors = 1;
constexpr std::size_t kSensitivities = 3; // [x0, r, sigma], quantape layout

/// quantape `TimeGrid(tMax, nSteps)` per-step scaling: t_k = tMax*k/N and
/// dt_k = t_{k+1} - t_k (not the rounded tMax/N), so paths reproduce the
/// quantape simulator bitwise; sqrt(dt_k) is the scheme's noise scale.
struct StepScaling {
    std::array<double, kSteps> dt{};
    std::array<double, kSteps> sqrtDt{};

    StepScaling() {
        double prev = 0.0;
        for (std::size_t k = 0; k < kSteps; ++k) {
            const double next = kT * static_cast<double>(k + 1) / static_cast<double>(kSteps);
            dt[k] = next - prev;
            sqrtDt[k] = std::sqrt(dt[k]);
            prev = next;
        }
    }
};

const StepScaling kStepScaling;

using Source = IidGaussianSource<>;

/// Raw (unnormalized) per-mode sums; divide by kPaths for the MC estimate.
struct GradSums {
    double value = 0.0;
    double dx0 = 0.0;
    double dr = 0.0;
    double dsigma = 0.0;
};

GradSums& operator+=(GradSums& a, const GradSums& b) {
    a.value += b.value;
    a.dx0 += b.dx0;
    a.dr += b.dr;
    a.dsigma += b.dsigma;
    return a;
}

// ── Euler-Maruyama GBM step ──
//
// Mirrors quantape's `Euler::step` for dX = r X dt + sigma X dW:
//   drift = r * x; next = x + dt * drift; g = sigma * x; next += g * zScaled
// with the same statement order, so the recorded tape structure and the
// floating-point operation order match the quantape lean/Stan paths.

RevScalar eulerStep(const RevScalar& x, const RevScalar& r, const RevScalar& sigma, double dt,
                    double zScaled) {
    return x + RevScalar(dt) * (r * x) + (sigma * x) * zScaled;
}

template <std::size_t W>
BatchScalar<W> eulerStep(const BatchScalar<W>& x, const BatchScalar<W>& r,
                         const BatchScalar<W>& sigma, double dt,
                         const std::array<double, W>& zScaled) {
    return x + BatchScalar<W>(dt) * (r * x) + (sigma * x) * BatchScalar<W>(zScaled);
}

// ── scalar RevScalar mode ──

struct ScalarPath {
    RevScalar y;
    RevScalar x0;
    RevScalar r;
    RevScalar sigma;
};

ScalarPath recordScalarPath(const Source& source, std::size_t pathIndex, Eigen::MatrixXd& z) {
    RevTape& tape = RevTape::active();
    tape.clear();
    const RevScalar r(kR, tape.input(kR));
    const RevScalar sigma(kSigma, tape.input(kSigma));
    const RevScalar x0(kX0, tape.input(kX0));
    RevScalar x = x0;
    for (std::size_t k = 0; k < kSteps; ++k) {
        source.fill(k, pathIndex, 1, z);
        x = eulerStep(x, r, sigma, kStepScaling.dt[k], z(0, 0) * kStepScaling.sqrtDt[k]);
    }
    return {fmax(x - kStrike, 0.0), x0, r, sigma};
}

GradSums scalarRange(const Source& source, std::size_t pathBegin, std::size_t nPaths) {
    GradSums sums;
    Eigen::MatrixXd z(kFactors, 1);
    for (std::size_t p = 0; p < nPaths; ++p) {
        const ScalarPath path = recordScalarPath(source, pathBegin + p, z);
        sums.value += path.y.value;
        if (path.y.node == RevTape::kNone) {
            continue; // out-of-the-money: constant payoff, zero sensitivities
        }
        RevTape::active().reverse(path.y.node);
        sums.dx0 += RevTape::active().adjoint(path.x0.node);
        sums.dr += RevTape::active().adjoint(path.r.node);
        sums.dsigma += RevTape::active().adjoint(path.sigma.node);
    }
    return sums;
}

std::array<double, kSensitivities> scalarPathAdjoints(const Source& source, std::size_t pathIndex) {
    std::array<double, kSensitivities> out{};
    Eigen::MatrixXd z(kFactors, 1);
    const ScalarPath path = recordScalarPath(source, pathIndex, z);
    if (path.y.node == RevTape::kNone) {
        return out;
    }
    RevTape::active().reverse(path.y.node);
    out[0] = RevTape::active().adjoint(path.x0.node);
    out[1] = RevTape::active().adjoint(path.r.node);
    out[2] = RevTape::active().adjoint(path.sigma.node);
    return out;
}

// ── BatchTape<W> mode: W paths per tape, one lane per path ──

template <std::size_t W>
struct BatchPath {
    BatchScalar<W> y;
    BatchScalar<W> x0;
    BatchScalar<W> r;
    BatchScalar<W> sigma;
};

template <std::size_t W>
BatchPath<W> recordBatchPath(const Source& source, std::size_t pathBegin, Eigen::MatrixXd& z) {
    BatchTape<W>& tape = BatchTape<W>::active();
    tape.clear();
    std::array<double, W> rLanes;
    std::array<double, W> sigmaLanes;
    std::array<double, W> x0Lanes;
    rLanes.fill(kR);
    sigmaLanes.fill(kSigma);
    x0Lanes.fill(kX0);
    const BatchScalar<W> r(tape.input(rLanes));
    const BatchScalar<W> sigma(tape.input(sigmaLanes));
    const BatchScalar<W> x0(tape.input(x0Lanes));
    BatchScalar<W> x = x0;
    for (std::size_t k = 0; k < kSteps; ++k) {
        source.fill(k, pathBegin, W, z);
        std::array<double, W> zScaled{};
        for (std::size_t l = 0; l < W; ++l) {
            zScaled[l] = z(0, static_cast<Eigen::Index>(l)) * kStepScaling.sqrtDt[k];
        }
        x = eulerStep(x, r, sigma, kStepScaling.dt[k], zScaled);
    }
    return {fmax(x - kStrike, BatchScalar<W>(0.0)), x0, r, sigma};
}

template <std::size_t W>
GradSums batchRange(const Source& source, std::size_t pathBegin, std::size_t nPaths) {
    GradSums sums;
    std::array<double, W> seed;
    seed.fill(1.0);
    Eigen::MatrixXd z(kFactors, W);
    const std::size_t nBatch = nPaths / W;
    for (std::size_t b = 0; b < nBatch; ++b) {
        const BatchPath<W> path = recordBatchPath<W>(source, pathBegin + b * W, z);
        for (std::size_t l = 0; l < W; ++l) {
            sums.value += path.y.value[l];
        }
        if (path.y.node == BatchTape<W>::kNone) {
            continue; // every lane out of the money
        }
        BatchTape<W>::active().reverse(path.y.node, seed);
        const std::array<double, W>& ax = BatchTape<W>::active().adjoint(path.x0.node);
        const std::array<double, W>& ar = BatchTape<W>::active().adjoint(path.r.node);
        const std::array<double, W>& as = BatchTape<W>::active().adjoint(path.sigma.node);
        for (std::size_t l = 0; l < W; ++l) {
            sums.dx0 += ax[l];
            sums.dr += ar[l];
            sums.dsigma += as[l];
        }
    }
    const std::size_t tail = nPaths - nBatch * W;
    if (tail != 0) { // odd path count: scalar tail on the (separate) RevTape
        sums += scalarRange(source, pathBegin + nBatch * W, tail);
    }
    return sums;
}

template <std::size_t W>
std::array<std::array<double, kSensitivities>, W> batchLaneAdjoints(const Source& source,
                                                                    std::size_t pathBegin) {
    std::array<std::array<double, kSensitivities>, W> out{};
    Eigen::MatrixXd z(kFactors, W);
    const BatchPath<W> path = recordBatchPath<W>(source, pathBegin, z);
    if (path.y.node == BatchTape<W>::kNone) {
        return out;
    }
    std::array<double, W> seed;
    seed.fill(1.0);
    BatchTape<W>::active().reverse(path.y.node, seed);
    const std::array<double, W>& ax = BatchTape<W>::active().adjoint(path.x0.node);
    const std::array<double, W>& ar = BatchTape<W>::active().adjoint(path.r.node);
    const std::array<double, W>& as = BatchTape<W>::active().adjoint(path.sigma.node);
    for (std::size_t l = 0; l < W; ++l) {
        out[l] = {ax[l], ar[l], as[l]};
    }
    return out;
}

// ── minimal persistent std::thread pool (main thread is worker 0) ──

class ThreadPool {
public:
    explicit ThreadPool(unsigned workers) {
        m_threads.reserve(workers);
        for (unsigned t = 0; t < workers; ++t) {
            m_threads.emplace_back([this, t] { workerLoop(t + 1); });
        }
    }

    ~ThreadPool() {
        {
            std::lock_guard<std::mutex> lock(m_mutex);
            m_stop = true;
        }
        m_work.notify_all();
        for (std::thread& thread : m_threads) {
            thread.join();
        }
    }

    ThreadPool(const ThreadPool&) = delete;
    ThreadPool& operator=(const ThreadPool&) = delete;

    /// Total workers, main thread included.
    std::size_t size() const { return m_threads.size() + 1; }

    /// Run `job(workerIndex)` on every worker; returns when all are done.
    /// `job` must not throw. Main thread runs index 0.
    void run(const std::function<void(unsigned)>& job) {
        {
            std::lock_guard<std::mutex> lock(m_mutex);
            m_job = &job;
            m_remaining = m_threads.size() + 1;
            ++m_generation;
        }
        m_work.notify_all();
        job(0);
        finishOne();
        waitDone();
    }

private:
    void workerLoop(unsigned me) {
        unsigned seen = 0;
        for (;;) {
            const std::function<void(unsigned)>* job = nullptr;
            {
                std::unique_lock<std::mutex> lock(m_mutex);
                m_work.wait(lock, [&] { return m_stop || m_generation != seen; });
                if (m_stop) {
                    return;
                }
                seen = m_generation;
                job = m_job;
            }
            (*job)(me);
            finishOne();
        }
    }

    void finishOne() {
        std::lock_guard<std::mutex> lock(m_mutex);
        if (--m_remaining == 0) {
            m_done.notify_one();
        }
    }

    void waitDone() {
        std::unique_lock<std::mutex> lock(m_mutex);
        m_done.wait(lock, [&] { return m_remaining == 0; });
    }

    std::vector<std::thread> m_threads;
    std::mutex m_mutex;
    std::condition_variable m_work;
    std::condition_variable m_done;
    const std::function<void(unsigned)>* m_job = nullptr;
    std::size_t m_remaining = 0;
    unsigned m_generation = 0;
    bool m_stop = false;
};

/// Contiguous static partition of `nTasks` over `pool.size()` workers.
template <typename F>
void parallelPartition(ThreadPool& pool, std::size_t nTasks, const F& fn) {
    const std::size_t workers = pool.size();
    const std::size_t chunk = (nTasks + workers - 1) / workers;
    pool.run([&](unsigned w) {
        const std::size_t begin =
            std::min<std::size_t>(static_cast<std::size_t>(w) * chunk, nTasks);
        const std::size_t end = std::min<std::size_t>(begin + chunk, nTasks);
        if (begin < end) {
            fn(begin, end, w);
        }
    });
}

GradSums scalarParallel(ThreadPool& pool, const Source& source) {
    std::vector<GradSums> partials(pool.size());
    parallelPartition(pool, kPaths, [&](std::size_t begin, std::size_t end, unsigned w) {
        partials[w] = scalarRange(source, begin, end - begin);
    });
    GradSums total;
    for (const GradSums& partial : partials) {
        total += partial;
    }
    return total;
}

template <std::size_t W>
GradSums batchParallel(ThreadPool& pool, const Source& source) {
    constexpr std::size_t kBatches = kPaths / W;
    std::vector<GradSums> partials(pool.size());
    parallelPartition(pool, kBatches, [&](std::size_t begin, std::size_t end, unsigned w) {
        partials[w] = batchRange<W>(source, begin * W, (end - begin) * W);
    });
    GradSums total;
    for (const GradSums& partial : partials) {
        total += partial;
    }
    return total;
}

// ── timing / reporting ──

struct Mode {
    const char* name;
    std::function<GradSums()> run;
};

double timedMs(const std::function<GradSums()>& run) {
    const auto t0 = std::chrono::steady_clock::now();
    const GradSums sums = run();
    const auto t1 = std::chrono::steady_clock::now();
    static volatile double sink = 0.0;
    sink = sums.value + sums.dx0 + sums.dr + sums.dsigma;
    (void)sink;
    return std::chrono::duration<double, std::milli>(t1 - t0).count();
}

double median(std::vector<double> samples) {
    std::sort(samples.begin(), samples.end());
    const std::size_t n = samples.size();
    return n % 2 == 1 ? samples[n / 2] : 0.5 * (samples[n / 2 - 1] + samples[n / 2]);
}

double relDiff(double a, double b) {
    const double scale = std::max({1.0, std::abs(a), std::abs(b)});
    return std::abs(a - b) / scale;
}

} // namespace

int main(int argc, char** argv) {
    const int rounds = argc > 1 ? std::atoi(argv[1]) : 5;
    const unsigned hw = std::max(1u, std::thread::hardware_concurrency());

    Source source(kFactors, kSeed);
    ThreadPool pool(hw > 0 ? hw - 1 : 0);

    std::printf("== lightape SDE batch bench: Euler GBM, %zu paths x %zu steps ==\n", kPaths,
                kSteps);
    std::printf("   r=%.2f sigma=%.2f x0=%.1f strike=%.1f T=%.1f dt=%.17g seed=%llu factors=%zu\n",
                kR, kSigma, kX0, kStrike, kT, kStepScaling.dt[0],
                static_cast<unsigned long long>(kSeed), kFactors);
    std::printf("   quantape config: bench_sde \"gradient euler gbm 20kx252 (lean reverse)\", "
                "Schedule::Parallel, IidGaussianSource(1, 7)\n");
    std::printf("   threads: hardware_concurrency=%u (pool: %zu incl. main), rounds=%d\n\n", hw,
                pool.size(), rounds);

    std::vector<Mode> modes = {
        {"scalar RevScalar (seq)", [&] { return scalarRange(source, 0, kPaths); }},
        {"scalar RevScalar (par)", [&] { return scalarParallel(pool, source); }},
        {"BatchTape W=2 (seq)", [&] { return batchRange<2>(source, 0, kPaths); }},
        {"BatchTape W=2 (par)", [&] { return batchParallel<2>(pool, source); }},
        {"BatchTape W=4 (seq)", [&] { return batchRange<4>(source, 0, kPaths); }},
        {"BatchTape W=4 (par)", [&] { return batchParallel<4>(pool, source); }},
        {"BatchTape W=8 (seq)", [&] { return batchRange<8>(source, 0, kPaths); }},
        {"BatchTape W=8 (par)", [&] { return batchParallel<8>(pool, source); }},
    };
    const std::size_t nModes = modes.size();

    // Warmup once per mode; the deterministic result is the canonical gradient.
    std::vector<GradSums> canonical(nModes);
    for (std::size_t i = 0; i < nModes; ++i) {
        canonical[i] = modes[i].run();
    }

    // Interleaved rounds: rotate the mode order each round.
    std::vector<std::vector<double>> samples(nModes);
    for (int r = 0; r < rounds; ++r) {
        for (std::size_t j = 0; j < nModes; ++j) {
            const std::size_t i = (j + static_cast<std::size_t>(r)) % nModes;
            samples[i].push_back(timedMs(modes[i].run));
        }
    }

    const double n = static_cast<double>(kPaths);
    std::printf("%-24s %10s %12s %12s %14s %14s %14s %14s\n", "mode", "median ms", "ns/path",
                "Mpath/s", "value", "dV/dx0", "dV/dr", "dV/dsigma");
    for (std::size_t i = 0; i < nModes; ++i) {
        const double ms = median(samples[i]);
        const GradSums& g = canonical[i];
        std::printf("%-24s %10.3f %12.1f %12.4f %14.10f %14.10f %14.10f %14.10f\n", modes[i].name,
                    ms, ms * 1e6 / n, n / (ms * 1e3), g.value / n, g.dx0 / n, g.dr / n,
                    g.dsigma / n);
    }

    // Raw round samples for transparency.
    std::printf("\nraw round samples (ms, mode order as above):\n");
    for (std::size_t i = 0; i < nModes; ++i) {
        std::printf("  %-24s", modes[i].name);
        for (double ms : samples[i]) {
            std::printf(" %10.3f", ms);
        }
        std::printf("\n");
    }

    // Full precision, for exact comparison against the quantape reference
    // (same draws; quantape commit 3387c0d, a probe of simulateGradientLean /
    // simulateGradient gives value 10.971547897285625, dV/dx0
    // 0.67311547897285795, dV/dr 67.267032711161647, dV/dsigma
    // 39.272376338302998; the scalar tape here reproduces all but dV/dr
    // bitwise, and dV/dr lands 1 ulp high).
    const auto printFull = [n](const char* label, const GradSums& g) {
        std::printf("  %-24s value %.17g  dV/dx0 %.17g  dV/dr %.17g  dV/dsigma %.17g\n", label,
                    g.value / n, g.dx0 / n, g.dr / n, g.dsigma / n);
    };
    std::printf("\nfull precision (mean over paths):\n");
    printFull(modes[0].name, canonical[0]);
    printFull(modes[6].name, canonical[6]);
    printFull(modes[7].name, canonical[7]);

    // Gradient cross-check. The scalar and batch tapes record the same
    // statements in the same order; only the summation order of the per-path
    // adjoints differs, so the aggregates must agree far below 1e-12 rel.
    const GradSums& reference = canonical[0];
    std::printf("\ngradient cross-check vs scalar RevScalar (seq), rel tol 1e-12:\n");
    bool allPass = true;
    for (std::size_t i = 1; i < nModes; ++i) {
        const GradSums& g = canonical[i];
        const double dmax =
            std::max({relDiff(g.value, reference.value), relDiff(g.dx0, reference.dx0),
                      relDiff(g.dr, reference.dr), relDiff(g.dsigma, reference.dsigma)});
        const bool pass = dmax <= 1e-12;
        allPass = allPass && pass;
        std::printf("  %-24s max rel diff %.3e  %s\n", modes[i].name, dmax, pass ? "PASS" : "FAIL");
    }

    // Lane-level gate: every lane of BatchTape<8> must reproduce the scalar
    // per-path adjoints bitwise (the batch bitwise contract, libm backend).
    std::size_t laneComparisons = 0;
    double laneMaxRel = 0.0;
    double laneMaxAbs = 0.0;
    for (std::size_t b = 0; b < kPaths / 8; ++b) {
        const auto lanes = batchLaneAdjoints<8>(source, b * 8);
        for (std::size_t l = 0; l < 8; ++l) {
            const auto scalar = scalarPathAdjoints(source, b * 8 + l);
            for (std::size_t s = 0; s < kSensitivities; ++s) {
                laneMaxRel = std::max(laneMaxRel, relDiff(lanes[l][s], scalar[s]));
                laneMaxAbs = std::max(laneMaxAbs, std::abs(lanes[l][s] - scalar[s]));
            }
            ++laneComparisons;
        }
    }
    std::printf("\nlane gate BatchTape<8> vs scalar per-path (%zu paths): max rel diff %.3e, "
                "max abs diff %.3e -> %s\n",
                laneComparisons, laneMaxRel, laneMaxAbs,
                laneMaxRel == 0.0 ? "bitwise equal" : "differs");

    // Tape-size diagnostics for one path / one W-path batch.
    {
        Eigen::MatrixXd z(kFactors, 1);
        const ScalarPath path = recordScalarPath(source, 0, z);
        const std::size_t scalarNodes = RevTape::active().size();
        std::printf("tape nodes: scalar/path %zu (payoff node %s)", scalarNodes,
                    path.y.node == RevTape::kNone ? "none" : "live");
        Eigen::MatrixXd zb(kFactors, 8);
        const BatchPath<8> batch = recordBatchPath<8>(source, 0, zb);
        std::printf(", BatchTape<8>/8-paths %zu (payoff node %s)\n", BatchTape<8>::active().size(),
                    batch.y.node == BatchTape<8>::kNone ? "none" : "live");
    }

    std::printf("\noverall: %s\n",
                allPass ? "ALL GRADIENT CROSS-CHECKS PASS" : "CROSS-CHECK FAILED");
    return allPass ? 0 : 1;
}
