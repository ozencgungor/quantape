#include <atomic>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <thread>
#include <utility>
#include <vector>

#include "lightape/lightape.h"
#include <gtest/gtest.h>

// Execution-seam tests: SerialExecutor / ThreadPoolExecutor / collectChunks,
// the WorkerHooks used for per-thread tape lifecycle, and bitwise
// serial-vs-parallel gates over RevTape record/reverse chunks.
//
// Determinism gates compare raw IEEE-754 bit patterns (decision #22):
// `std::bit_cast<std::uint64_t>` keeps +0.0/-0.0 distinct and NaN bitwise.
// Every test that touches the shared hook counters resets them first; gtest
// may run cases in one process.

namespace {

using lightape::RevScalar;
using lightape::RevTape;
using lightape::exec::collectChunks;
using lightape::exec::SerialExecutor;
using lightape::exec::ThreadPoolExecutor;
using lightape::exec::WorkerHooks;
using lightape::exec::WorkerScope;

std::atomic<int> g_startCount{0};
std::atomic<int> g_stopCount{0};
std::atomic<int> g_stopSawTape{0};

void resetHookCounters() {
    g_startCount.store(0);
    g_stopCount.store(0);
    g_stopSawTape.store(0);
}

/// onStart recipe: attach this worker's tape before any chunk runs.
void countingStart() noexcept {
    RevTape::active();
    g_startCount.fetch_add(1, std::memory_order_relaxed);
}

/// onStop recipe: the TLS tape must be alive and releasable on the exiting
/// worker, then the flag records that the release ran on the worker thread.
void countingStop() noexcept {
    (void)RevTape::active();
    RevTape::releaseCurrentThread();
    g_stopSawTape.fetch_add(1, std::memory_order_relaxed);
    g_stopCount.fetch_add(1, std::memory_order_relaxed);
}

/// Per-chunk deterministic composite: 32 refined terms recorded on this
/// thread's `RevTape`, one reverse sweep, value + gradient packed into a
/// double. The per-chunk `clear()` is the intended TLS-reuse pattern: no
/// state leaks between chunks and no tape is shared across threads.
double chunkComposite(std::size_t begin, std::size_t end) {
    RevTape& tape = RevTape::active();
    tape.clear();
    const double seed =
        0.5 + 0.125 * static_cast<double>(begin) + 0.0625 * static_cast<double>(end - begin);
    RevScalar x(seed, tape.input(seed));
    RevScalar acc = x;
    for (std::size_t k = 0; k < 32; ++k) {
        const double c = 0.01 * static_cast<double>(k + 1);
        acc = acc * c + x;
        acc = sqrt(acc * acc + 1.0);
    }
    tape.reverse(acc.node);
    return acc.value + tape.adjoint(x.node);
}

/// One SDE-like path: 32 Euler steps on the thread-local tape, then reverse
/// and return terminal value + sensitivity to the initial state.
double sdePathValue(std::size_t path) {
    RevTape& tape = RevTape::active();
    tape.clear();
    const double s0 = 0.5 + 0.01 * static_cast<double>(path % 17);
    RevScalar s(s0, tape.input(s0));
    const std::uint32_t root = s.node;
    for (std::size_t step = 0; step < 32; ++step) {
        const double drift = 0.02 + 0.0001 * static_cast<double>(step);
        const double vol = 0.3 + 0.01 * static_cast<double>(path % 5);
        const double z = 0.25 * static_cast<double>((path * 7 + step * 13) % 11) - 1.0;
        s = s + (drift * s + vol * z) * (1.0 / 32.0);
    }
    tape.reverse(s.node);
    return s.value + tape.adjoint(root);
}

/// Chunk result: fixed ascending path order inside the chunk, so the ordered
/// merge by collectChunks is the only ordering the result depends on.
double sdeChunk(std::size_t begin, std::size_t end) {
    double acc = 0.0;
    for (std::size_t path = begin; path < end; ++path) {
        acc += sdePathValue(path);
    }
    return acc;
}

std::uint64_t bits(double value) {
    return std::bit_cast<std::uint64_t>(value);
}

// ── WorkerScope ─────────────────────────────────────────────────────────────

TEST(WorkerScopeTest, CallsStartThenStopAndIgnoresEmptyHooks) {
    resetHookCounters();
    {
        const WorkerScope scope(WorkerHooks{&countingStart, &countingStop});
        EXPECT_EQ(g_startCount.load(), 1);
        EXPECT_EQ(g_stopCount.load(), 0);
    }
    EXPECT_EQ(g_stopCount.load(), 1);

    {
        const WorkerScope scope; // empty hooks: no calls, no crash
        EXPECT_EQ(g_startCount.load(), 1);
        EXPECT_EQ(g_stopCount.load(), 1);
    }
    EXPECT_EQ(g_startCount.load(), 1);
    EXPECT_EQ(g_stopCount.load(), 1);
}

// ── SerialExecutor ──────────────────────────────────────────────────────────

TEST(SerialExecutorTest, CoversEveryBatchExactlyOnceInOrder) {
    const SerialExecutor serial;
    std::vector<std::pair<std::size_t, std::size_t>> chunks;
    serial.runBatches(10, 3,
                      [&](std::size_t begin, std::size_t end) { chunks.emplace_back(begin, end); });
    ASSERT_EQ(chunks.size(), 4u);
    EXPECT_EQ(chunks[0], (std::pair<std::size_t, std::size_t>{0, 3}));
    EXPECT_EQ(chunks[1], (std::pair<std::size_t, std::size_t>{3, 6}));
    EXPECT_EQ(chunks[2], (std::pair<std::size_t, std::size_t>{6, 9}));
    EXPECT_EQ(chunks[3], (std::pair<std::size_t, std::size_t>{9, 10}));

    std::vector<int> hits(10, 0);
    serial.runBatches(10, 3, [&](std::size_t begin, std::size_t end) {
        for (std::size_t i = begin; i < end; ++i) {
            ++hits[i];
        }
    });
    for (std::size_t i = 0; i < hits.size(); ++i) {
        EXPECT_EQ(hits[i], 1) << "batch " << i;
    }
}

TEST(SerialExecutorTest, ZeroBatchesMakesNoCalls) {
    const SerialExecutor serial;
    int calls = 0;
    serial.runBatches(0, 4, [&](std::size_t, std::size_t) { ++calls; });
    EXPECT_EQ(calls, 0);
    EXPECT_TRUE(collectChunks(serial, 0, 4, [](std::size_t, std::size_t) { return 0; }).empty());
}

TEST(SerialExecutorTest, ZeroGrainIsTreatedAsOne) {
    const SerialExecutor serial;
    std::vector<std::size_t> begins;
    serial.runBatches(3, 0, [&](std::size_t begin, std::size_t) { begins.push_back(begin); });
    EXPECT_EQ(begins, (std::vector<std::size_t>{0, 1, 2}));
}

TEST(SerialExecutorTest, GrainAtLeastBatchesIsOneCall) {
    const SerialExecutor serial;
    int calls = 0;
    std::size_t seenBegin = 99;
    std::size_t seenEnd = 0;
    serial.runBatches(5, 5, [&](std::size_t begin, std::size_t end) {
        ++calls;
        seenBegin = begin;
        seenEnd = end;
    });
    EXPECT_EQ(calls, 1);
    EXPECT_EQ(seenBegin, 0u);
    EXPECT_EQ(seenEnd, 5u);

    calls = 0;
    serial.runBatches(5, 9, [&](std::size_t, std::size_t) { ++calls; });
    EXPECT_EQ(calls, 1);
}

TEST(SerialExecutorTest, ThreadsIsOne) {
    EXPECT_EQ(SerialExecutor().threads(), 1u);
    EXPECT_EQ(SerialExecutor(WorkerHooks{}).threads(), 1u);
}

TEST(SerialExecutorTest, HooksOpenAndCloseAroundEachRun) {
    resetHookCounters();
    {
        const SerialExecutor serial(WorkerHooks{&countingStart, &countingStop});
        serial.runBatches(4, 2, [](std::size_t, std::size_t) {});
        EXPECT_EQ(g_startCount.load(), 1);
        EXPECT_EQ(g_stopCount.load(), 1); // the scope closes with each run
        serial.runBatches(4, 2, [](std::size_t, std::size_t) {});
        EXPECT_EQ(g_startCount.load(), 2);
        EXPECT_EQ(g_stopCount.load(), 2);
    }
    EXPECT_EQ(g_startCount.load(), 2);
    EXPECT_EQ(g_stopCount.load(), 2);
}

// ── ThreadPoolExecutor ──────────────────────────────────────────────────────

TEST(ThreadPoolExecutorTest, ThreadsReportedAndWorkersStartLazily) {
    resetHookCounters();
    {
        const ThreadPoolExecutor unused(3, WorkerHooks{&countingStart, &countingStop});
        EXPECT_EQ(unused.threads(), 3u);
        EXPECT_EQ(g_startCount.load(), 0);
        EXPECT_EQ(g_stopCount.load(), 0);
    }
    // Never used: no thread was ever spawned, so no hook ran.
    EXPECT_EQ(g_startCount.load(), 0);
    EXPECT_EQ(g_stopCount.load(), 0);

    {
        const ThreadPoolExecutor pool(2, WorkerHooks{&countingStart, &countingStop});
        const std::vector<std::size_t> values = collectChunks(
            pool, 8, 2, [](std::size_t begin, std::size_t end) { return begin + end; });
        ASSERT_EQ(values.size(), 4u);
        EXPECT_EQ(values[0], 2u);
        EXPECT_EQ(values[1], 6u);
        EXPECT_EQ(values[2], 10u);
        EXPECT_EQ(values[3], 14u);
    }
    EXPECT_EQ(g_startCount.load(), 2);
    EXPECT_EQ(g_stopCount.load(), 2);
}

TEST(ThreadPoolExecutorTest, DefaultThreadCountIsEffective) {
    const ThreadPoolExecutor pool;
    EXPECT_GE(pool.threads(), 1u);
    const unsigned int hardware = std::thread::hardware_concurrency();
    if (hardware != 0) {
        EXPECT_EQ(pool.threads(), static_cast<std::size_t>(hardware));
    }
}

TEST(ThreadPoolExecutorTest, ChunksExactlyOnceUnderContention) {
    constexpr std::size_t nBatches = 4096;
    const ThreadPoolExecutor pool(4);
    std::unique_ptr<std::atomic<std::size_t>[]> hits(new std::atomic<std::size_t>[nBatches]);
    for (std::size_t i = 0; i < nBatches; ++i) {
        hits[i].store(0, std::memory_order_relaxed);
    }
    pool.runBatches(nBatches, 1, [&](std::size_t begin, std::size_t end) {
        for (std::size_t i = begin; i < end; ++i) {
            hits[i].fetch_add(1, std::memory_order_relaxed);
        }
    });
    std::size_t bad = 0;
    std::size_t total = 0;
    for (std::size_t i = 0; i < nBatches; ++i) {
        const std::size_t count = hits[i].load(std::memory_order_relaxed);
        bad += (count == 1) ? 0 : 1;
        total += count;
    }
    EXPECT_EQ(bad, 0u);
    EXPECT_EQ(total, nBatches);
}

TEST(ThreadPoolExecutorTest, CollectChunksMatchesSerialBitwise) {
    constexpr std::size_t nBatches = 96;
    constexpr std::size_t grain = 5;
    const SerialExecutor serial;
    const ThreadPoolExecutor pool(4);
    const std::vector<double> reference = collectChunks(serial, nBatches, grain, &chunkComposite);
    ASSERT_EQ(reference.size(), 20u);

    for (int repeat = 0; repeat < 20; ++repeat) {
        const std::vector<double> parallel = collectChunks(pool, nBatches, grain, &chunkComposite);
        ASSERT_EQ(parallel.size(), reference.size());
        for (std::size_t i = 0; i < reference.size(); ++i) {
            EXPECT_EQ(bits(parallel[i]), bits(reference[i]))
                << "repeat " << repeat << " chunk " << i;
        }
    }
}

TEST(ThreadPoolExecutorTest, OneThreadPoolMatchesSerialBitwise) {
    constexpr std::size_t nBatches = 64;
    constexpr std::size_t grain = 3;
    const SerialExecutor serial;
    const ThreadPoolExecutor pool(1);
    const std::vector<double> reference = collectChunks(serial, nBatches, grain, &chunkComposite);
    const std::vector<double> oneThread = collectChunks(pool, nBatches, grain, &chunkComposite);
    ASSERT_EQ(oneThread.size(), reference.size());
    for (std::size_t i = 0; i < reference.size(); ++i) {
        EXPECT_EQ(bits(oneThread[i]), bits(reference[i])) << "chunk " << i;
    }
}

TEST(ThreadPoolExecutorTest, ExceptionRethrownAndOtherChunksFinish) {
    constexpr std::size_t nBatches = 32;
    const ThreadPoolExecutor pool(4);
    std::unique_ptr<std::atomic<std::size_t>[]> hits(new std::atomic<std::size_t>[nBatches]);
    for (std::size_t i = 0; i < nBatches; ++i) {
        hits[i].store(0, std::memory_order_relaxed);
    }
    std::atomic<std::size_t> completed{0};
    const auto body = [&](std::size_t begin, std::size_t end) {
        for (std::size_t i = begin; i < end; ++i) {
            hits[i].fetch_add(1, std::memory_order_relaxed);
        }
        if (begin == 0) {
            // Deterministic failure ordering: let every other chunk finish
            // before throwing, so "all other chunks complete exactly once" is
            // testable even though the seam permits skipping unclaimed chunks.
            while (completed.load(std::memory_order_acquire) != nBatches - 1) {}
            throw std::runtime_error("chunk 0 failed");
        }
        completed.fetch_add(1, std::memory_order_release);
    };
    EXPECT_THROW(pool.runBatches(nBatches, 1, body), std::runtime_error);
    std::size_t bad = 0;
    for (std::size_t i = 0; i < nBatches; ++i) {
        bad += (hits[i].load(std::memory_order_relaxed) == 1) ? 0 : 1;
    }
    EXPECT_EQ(bad, 0u);
    EXPECT_EQ(completed.load(), nBatches - 1);
}

TEST(ThreadPoolExecutorTest, FailedRunNeverRepeatsAChunk) {
    constexpr std::size_t nBatches = 256;
    const ThreadPoolExecutor pool(4);
    std::unique_ptr<std::atomic<std::size_t>[]> hits(new std::atomic<std::size_t>[nBatches]);
    for (std::size_t i = 0; i < nBatches; ++i) {
        hits[i].store(0, std::memory_order_relaxed);
    }
    const auto body = [&](std::size_t begin, std::size_t) {
        hits[begin].fetch_add(1, std::memory_order_relaxed);
        throw std::logic_error("fail fast");
    };
    EXPECT_THROW(pool.runBatches(nBatches, 1, body), std::logic_error);
    std::size_t duplicated = 0;
    std::size_t ran = 0;
    for (std::size_t i = 0; i < nBatches; ++i) {
        const std::size_t count = hits[i].load(std::memory_order_relaxed);
        duplicated += (count > 1) ? 1 : 0;
        ran += count;
    }
    EXPECT_EQ(duplicated, 0u);
    EXPECT_GT(ran, 0u);
    EXPECT_LE(ran, nBatches);
}

TEST(ThreadPoolExecutorTest, ConcurrentCallersSerializeRuns) {
    constexpr std::size_t nCallers = 4;
    constexpr std::size_t nBatches = 32;
    constexpr std::size_t grain = 4;
    const ThreadPoolExecutor pool(4);
    std::vector<std::vector<std::size_t>> results(nCallers);
    std::vector<std::thread> callers;
    callers.reserve(nCallers);
    for (std::size_t caller = 0; caller < nCallers; ++caller) {
        callers.emplace_back([&, caller] {
            results[caller] =
                collectChunks(pool, nBatches, grain, [caller](std::size_t begin, std::size_t end) {
                    return caller * 1000 + begin + end;
                });
        });
    }
    for (std::thread& caller : callers) {
        caller.join();
    }
    for (std::size_t caller = 0; caller < nCallers; ++caller) {
        ASSERT_EQ(results[caller].size(), nBatches / grain);
        for (std::size_t i = 0; i < results[caller].size(); ++i) {
            const std::size_t begin = i * grain;
            const std::size_t end = begin + grain;
            EXPECT_EQ(results[caller][i], caller * 1000 + begin + end)
                << "caller " << caller << " chunk " << i;
        }
    }
}

TEST(ThreadPoolExecutorTest, HooksFireOncePerStartedWorker) {
    resetHookCounters();
    {
        const ThreadPoolExecutor pool(3, WorkerHooks{&countingStart, &countingStop});
        pool.runBatches(32, 4, [](std::size_t, std::size_t) {});
        EXPECT_EQ(g_stopCount.load(), 0); // workers stay alive with the pool
    }
    EXPECT_EQ(g_startCount.load(), 3);
    EXPECT_EQ(g_stopCount.load(), 3);
    EXPECT_EQ(g_stopSawTape.load(), 3);

    // The main thread's tape is untouched and usable after every worker
    // released its own on exit (onStop ran on the worker threads).
    RevTape& tape = RevTape::active();
    tape.clear();
    RevScalar x(2.0, tape.input(2.0));
    RevScalar y = x * x;
    tape.reverse(y.node);
    EXPECT_DOUBLE_EQ(tape.adjoint(x.node), 4.0);
}

// ── integration: SDE-like per-chunk gradient loop ───────────────────────────

TEST(ExecutorIntegration, SdeLikeGradientChunksMergeInOrder) {
    constexpr std::size_t nPaths = 128;
    constexpr std::size_t grain = 7;
    const SerialExecutor serial;
    const ThreadPoolExecutor pool(4);
    const std::vector<double> reference = collectChunks(serial, nPaths, grain, &sdeChunk);
    ASSERT_EQ(reference.size(), (nPaths + grain - 1) / grain);

    for (int repeat = 0; repeat < 10; ++repeat) {
        const std::vector<double> parallel = collectChunks(pool, nPaths, grain, &sdeChunk);
        ASSERT_EQ(parallel.size(), reference.size());
        for (std::size_t i = 0; i < reference.size(); ++i) {
            EXPECT_EQ(bits(parallel[i]), bits(reference[i]))
                << "repeat " << repeat << " chunk " << i;
        }
    }

    // Both executors called RevTape::active() per chunk; the caller's tape is
    // still reusable after the pool workers released theirs.
    RevTape& tape = RevTape::active();
    tape.clear();
    const double seed = 1.25;
    RevScalar x(seed, tape.input(seed));
    RevScalar f = exp(x) + x * x;
    tape.reverse(f.node);
    EXPECT_DOUBLE_EQ(tape.adjoint(x.node), std::exp(seed) + 2.0 * seed);
}

} // namespace
