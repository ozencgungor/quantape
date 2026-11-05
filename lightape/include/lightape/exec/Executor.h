#ifndef LIGHTAPE_EXEC_EXECUTOR_H
#define LIGHTAPE_EXEC_EXECUTOR_H

#include <atomic>
#include <condition_variable>
#include <cstddef>
#include <exception>
#include <memory>
#include <mutex>
#include <thread>
#include <type_traits>
#include <utility>
#include <vector>

namespace lightape::exec {

/**
 * @file Executor.h
 * @brief Pool-agnostic execution seam: serial reference + std::thread pool.
 *
 * The engine is TBB-free and pool-agnostic. Every parallel entry point
 * (pathwise gradients, batched paths, calibration/risk drivers) takes a
 * caller-owned executor by reference and calls
 * `runBatches(nBatches, grain, fn)`; the same call sites run under
 * `SerialExecutor` (the single-threaded bitwise reference),
 * `ThreadPoolExecutor` (the post-Stan std::thread pool) or a quantape-side
 * TBB adapter (while Stan/TBB coexist). There is deliberately **no global or
 * default executor singleton**: the driver owns the pool, its thread budget
 * and its lifetime, so `global_control`/arena limits, isolation and
 * phase-boundary reclamation stay explicit at the call site.
 *
 * ## The seam contract
 *
 * - `runBatches(n, grain, fn)` invokes `fn(begin, end)` exactly once per
 *   consecutive chunk `[i*grain, min((i+1)*grain, n))`, `i` in
 *   `[0, ceil(n/grain))`, with `begin` aligned to `grain`. `n == 0` makes no
 *   calls; `grain == 0` is treated as 1 by both executors (the seam stays
 *   total rather than looping); `grain >= n > 0` is one call `(0, n)`.
 * - Chunk completion order is unspecified and may differ per run; chunk
 *   index membership is fixed, which is what `collectChunks` exploits.
 * - `runBatches` is `const`: executing is a query over the pool. A backend
 *   must be callable through `const Exec&`.
 * - `threads()` returns the effective worker count of the backend
 *   (`SerialExecutor` is always 1).
 * - Per-worker resources attach/release through `WorkerHooks`
 *   (`onStart`/`onStop`, both `noexcept`). The intended recipe for the
 *   thread-local tapes of this engine is
 *   `hooks.onStart = []() noexcept { RevTape::active(); }` and
 *   `hooks.onStop = []() noexcept { RevTape::releaseCurrentThread(); }`;
 *   `BatchTape<W>::active()`/`releaseCurrentThread()` use the identical
 *   recipe per lane width. Hooks must be `noexcept`: scheduler callbacks
 *   (oneTBB observers) must never throw, and a throwing hook is a terminate.
 *
 * ## TBB adapter contract (quantape side; lightape has no TBB include)
 *
 * The adapter lives in exactly one compiled TU on the quantape side and
 * implements the same two members:
 *
 *     class TbbExecutor {
 *     public:
 *         explicit TbbExecutor(WorkerHooks hooks) : m_hooks(hooks) {}
 *
 *         template <class F>
 *         void runBatches(std::size_t nBatches, std::size_t grain, F&& fn) const {
 *             if (nBatches == 0) return;
 *             if (grain == 0) grain = 1;
 *             const std::size_t nChunks = 1 + (nBatches - 1) / grain;
 *             tbb::parallel_for(
 *                 tbb::blocked_range<std::size_t>(0, nChunks),
 *                 [&](const tbb::blocked_range<std::size_t>& r) {
 *                     const WorkerScope scope(m_hooks);  // per-task attach/release
 *                     for (std::size_t i = r.begin(); i != r.end(); ++i) {
 *                         const std::size_t begin = i * grain;
 *                         const std::size_t end =
 *                             (nBatches - begin < grain) ? nBatches : begin + grain;
 *                         fn(begin, end);
 *                     }
 *                 },
 *                 tbb::static_partitioner());
 *         }
 *
 *         std::size_t threads() const noexcept;  // arena max_concurrency
 *     };
 *
 * - The chunk invariants above are the adapter's whole obligation:
 *   `parallel_for` over the chunk *index* keeps `begin` aligned to `grain`
 *   (`collectChunks` places results by `begin / grain`), and
 *   `static_partitioner` pins the split tree so the partition does not move
 *   with the worker count (determinism policy, decision #22).
 * - Hooks: either attach a `tbb::task_scheduler_observer` to an arena the
 *   adapter owns (long-lived workers) or put a `WorkerScope` in the task
 *   body as above (per-task; it structurally avoids the `observer_proxy`
 *   teardown canary documented in `threading_tbb.md` §1.3).
 * - Exceptions: `tbb::parallel_for` captures the body exception, cancels
 *   remaining tasks and rethrows on the calling thread after the algorithm
 *   completes; that matches this seam's "first failure wins, unfinished
 *   chunks may be skipped" policy.
 * - Nested parallelism: the seam has no nesting magic. A driver already
 *   inside a parallel region passes a `SerialExecutor` (or a separately
 *   budgeted pool/arena) for the inner work; re-entering the *same*
 *   executor from inside its own `fn` deadlocks by construction. See
 *   `threading_tbb.md` §3.7 for the outer-loop-owns-the-budget policy.
 */

/// Per-worker lifecycle hooks. `onStart` runs on the worker thread when its
/// `WorkerScope` opens, `onStop` when it closes (worker exit, or the end of a
/// serial run). Both are `noexcept` because backends invoke them from
/// scheduler callbacks that must not throw.
struct WorkerHooks {
    void (*onStart)() noexcept = nullptr;
    void (*onStop)() noexcept = nullptr;
};

/// RAII lifecycle scope: calls `hooks.onStart` on construction and
/// `hooks.onStop` on destruction, each only when set. Non-copyable and
/// non-movable so the start/stop pair can never be duplicated or dropped.
/// Custom backends (including a future TBB adapter) wrap their per-worker
/// section in one of these to attach/release the worker's tape resources.
class WorkerScope {
public:
    explicit WorkerScope(WorkerHooks hooks = {}) noexcept : m_onStop(hooks.onStop) {
        if (hooks.onStart != nullptr) {
            hooks.onStart();
        }
    }

    ~WorkerScope() noexcept {
        if (m_onStop != nullptr) {
            m_onStop();
        }
    }

    WorkerScope(const WorkerScope&) = delete;
    WorkerScope& operator=(const WorkerScope&) = delete;

private:
    void (*m_onStop)() noexcept;
};

/// Single-threaded reference executor. `runBatches` walks the chunks on the
/// calling thread in ascending order and opens one `WorkerScope` per call, so
/// the serial hooks fire on the caller exactly once per `runBatches` (the
/// caller is the worker). This is the bitwise reference every parallel
/// backend is gated against and the escape hatch for nested work.
class SerialExecutor {
public:
    explicit SerialExecutor(WorkerHooks hooks = {}) noexcept : m_hooks(hooks) {}

    template <class F>
    void runBatches(std::size_t nBatches, std::size_t grain, F&& fn) const {
        if (nBatches == 0) {
            return;
        }
        if (grain == 0) {
            grain = 1;
        }
        const WorkerScope scope(m_hooks);
        for (std::size_t begin = 0; begin < nBatches;) {
            const std::size_t end = (nBatches - begin < grain) ? nBatches : begin + grain;
            fn(begin, end);
            begin = end;
        }
    }

    std::size_t threads() const noexcept { return 1; }

private:
    WorkerHooks m_hooks;
};

/// Fixed-size `std::thread` pool. Construction is cheap and spawns nothing:
/// workers start lazily on the first `runBatches` (a constructed-but-unused
/// executor is free), sleep on a condition variable between runs (no busy
/// spin) and are always joined by the destructor, which sets the stop flag
/// first and guarantees `onStop` ran for every started worker.
///
/// `threads == 0` means `std::thread::hardware_concurrency()` (at least 1);
/// `threads()` reports the effective count. `runBatches` blocks the calling
/// thread (the caller is not a worker); calls on one executor are serialized,
/// a second caller waits for the first run to finish. Workers claim `grain`
/// sized chunks from an atomic index, so every chunk runs exactly once and
/// may complete out of order. The first exception thrown by `fn` is captured
/// and rethrown by `runBatches` after all workers have left the run; workers
/// stop claiming new chunks once a failure is recorded, so chunks not yet
/// started may be skipped (chunks already claimed run to completion).
///
/// Never call `runBatches` on the same executor from inside its own `fn`
/// (deadlock); nested work uses a `SerialExecutor` or a bounded pool.
class ThreadPoolExecutor {
public:
    explicit ThreadPoolExecutor(std::size_t threads = 0, WorkerHooks hooks = {})
        : m_threads(threads == 0 ? defaultThreads() : threads), m_hooks(hooks) {}

    ~ThreadPoolExecutor() {
        {
            std::lock_guard<std::mutex> lock(m_mutex);
            m_stop = true;
        }
        m_jobReady.notify_all();
        for (std::thread& worker : m_workers) {
            worker.join();
        }
    }

    ThreadPoolExecutor(const ThreadPoolExecutor&) = delete;
    ThreadPoolExecutor& operator=(const ThreadPoolExecutor&) = delete;

    std::size_t threads() const noexcept { return m_threads; }

    template <class F>
    void runBatches(std::size_t nBatches, std::size_t grain, F&& fn) const {
        if (nBatches == 0) {
            return;
        }
        if (grain == 0) {
            grain = 1;
        }
        ensureStarted();

        Job job;
        job.fn.owner = const_cast<void*>(static_cast<const void*>(std::addressof(fn)));
        job.fn.invoke = &invokeBatch<std::remove_reference_t<F>>;
        job.nBatches = nBatches;
        job.grain = grain;
        job.nChunks = 1 + (nBatches - 1) / grain;

        std::exception_ptr error;
        {
            std::unique_lock<std::mutex> lock(m_mutex);
            // One run at a time per executor: wait until no other run is
            // active, then publish this job (stack-local; workers only reach
            // it through m_job, which is cleared before runBatches returns).
            m_runDone.wait(lock, [this] { return m_job == nullptr; });
            m_job = &job;
            lock.unlock();
            m_jobReady.notify_all();
            lock.lock();
            // The last worker out clears m_job; wait for that before reading
            // the captured error and letting the stack job go out of scope.
            m_runDone.wait(lock, [this, &job] { return m_job != &job; });
            error = job.error;
        }
        if (error != nullptr) {
            std::rethrow_exception(error);
        }
    }

private:
    /// Type-erased `fn(begin, end)`: the workers are not templated, so the
    /// callable is passed as its address plus a per-F trampoline. `owner`
    /// outlives the job because `runBatches` returns only after every worker
    /// has left it.
    struct BatchFn {
        void* owner = nullptr;
        void (*invoke)(void* fn, std::size_t begin, std::size_t end) = nullptr;
    };

    template <class F>
    static void invokeBatch(void* fn, std::size_t begin, std::size_t end) {
        (*static_cast<std::remove_reference_t<F>*>(fn))(begin, end);
    }

    /// One `runBatches` invocation: immutable chunk geometry plus the atomic
    /// claim cursor and failure state. `error` and `participants` are guarded
    /// by the executor mutex; `next`/`failed` are lock-free.
    struct Job {
        BatchFn fn;
        std::size_t nBatches = 0;
        std::size_t grain = 1;
        std::size_t nChunks = 0;
        std::atomic<std::size_t> next{0};
        std::atomic<int> failed{0};
        std::exception_ptr error;     // first failure wins
        std::size_t participants = 0; // workers currently inside the job
    };

    static std::size_t defaultThreads() noexcept {
        const unsigned int hardware = std::thread::hardware_concurrency();
        return hardware == 0 ? 1 : static_cast<std::size_t>(hardware);
    }

    /// Spawn the workers once, guarded by the mutex so concurrent first calls
    /// start exactly one set. Called before the run lock is taken.
    void ensureStarted() const {
        std::lock_guard<std::mutex> lock(m_mutex);
        if (m_started) {
            return;
        }
        m_started = true;
        m_workers.reserve(m_threads);
        for (std::size_t i = 0; i < m_threads; ++i) {
            m_workers.emplace_back([this] { workerMain(); });
        }
    }

    /// Worker lifetime: one `WorkerScope` around everything (onStart once on
    /// entry, onStop at exit), then a wait/claim loop. Chunks are claimed
    /// with `next.fetch_add` so each runs exactly once; after a failure only
    /// already-claimed chunks finish. A worker that runs out of chunks waits
    /// for the job to complete before it is available again, so it can never
    /// re-enter a job whose participants have already been counted (the last
    /// participant out clears `m_job` under the mutex).
    void workerMain() const {
        const WorkerScope scope(m_hooks);
        std::unique_lock<std::mutex> lock(m_mutex);
        for (;;) {
            m_jobReady.wait(lock, [this] { return m_stop || m_job != nullptr; });
            if (m_stop) {
                return;
            }
            Job* job = m_job;
            ++job->participants;
            for (;;) {
                if (job->failed.load(std::memory_order_relaxed) != 0) {
                    break;
                }
                const std::size_t index = job->next.fetch_add(1, std::memory_order_relaxed);
                if (index >= job->nChunks) {
                    break;
                }
                const std::size_t begin = index * job->grain;
                const std::size_t end =
                    (job->nBatches - begin < job->grain) ? job->nBatches : begin + job->grain;
                lock.unlock();
                bool threw = false;
                try {
                    job->fn.invoke(job->fn.owner, begin, end);
                } catch (...) {
                    threw = true;
                    lock.lock();
                    if (!job->error) {
                        job->error = std::current_exception();
                    }
                    job->failed.store(1, std::memory_order_relaxed);
                }
                if (!threw) {
                    lock.lock();
                }
            }
            if (--job->participants == 0) {
                m_job = nullptr;
                m_jobReady.notify_all(); // workers parked on this job may leave
                m_runDone.notify_all();
            } else {
                m_jobReady.wait(lock, [this, job] { return m_stop || m_job != job; });
                if (m_stop) {
                    return;
                }
            }
        }
    }

    const std::size_t m_threads;
    const WorkerHooks m_hooks;
    mutable std::mutex m_mutex;
    mutable std::condition_variable m_jobReady; // workers wait for work
    mutable std::condition_variable m_runDone;  // callers wait for completion
    mutable std::vector<std::thread> m_workers;
    mutable Job* m_job = nullptr;
    mutable bool m_started = false;
    mutable bool m_stop = false;
};

/// Run `fn(begin, end)` over the chunks of `ex` and return the chunk results
/// in chunk order. Each chunk is computed independently and stored at its
/// own index, so the returned vector is identical — bitwise, when `fn` is —
/// regardless of the executor, scheduling, or thread count. `Result` is the
/// decayed return type of `fn` and must be default-constructible; the
/// function may not return `void`.
///
/// The guarantee is per chunk: `fn` must not share state across chunks
/// (a reduction accumulator, a common tape, a scratch buffer, ...) or the
/// result becomes scheduling-dependent. The intended pattern is exactly what
/// `SerialExecutor` and `ThreadPoolExecutor` are gated on: a per-chunk
/// independent `RevTape` recording (`RevTape::active().clear(); ...;
/// reverse()`), values read into the chunk result, then the ordered merge
/// this function performs. `begin` is always a multiple of `grain`, so the
/// chunk index is `begin / grain`.
template <class Exec, class F>
auto collectChunks(const Exec& ex, std::size_t nBatches, std::size_t grain, F&& fn) {
    using Result = std::remove_cvref_t<std::invoke_result_t<F&, std::size_t, std::size_t>>;
    if (grain == 0) {
        grain = 1;
    }
    const std::size_t nChunks = (nBatches == 0) ? 0 : 1 + (nBatches - 1) / grain;
    std::vector<Result> results(nChunks);
    ex.runBatches(nBatches, grain, [&results, &fn, grain](std::size_t begin, std::size_t end) {
        results[begin / grain] = fn(begin, end);
    });
    return results;
}

} // namespace lightape::exec

#endif // LIGHTAPE_EXEC_EXECUTOR_H
