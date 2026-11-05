#ifndef LIGHTAPE_REV_REVOLVE_H
#define LIGHTAPE_REV_REVOLVE_H

#include <algorithm>
#include <cstddef>
#include <limits>
#include <utility>
#include <vector>

#include "lightape/rev/Checkpoint.h"

namespace lightape {
/**
 * @file Revolve.h
 * @brief Offline binomial (Griewank/Revolve) checkpoint scheduler + segment
 *        gluing helper.
 *
 * `Checkpoint.h` provides the *building blocks* (segment recording windows and
 * the callback-based `Segment`), but deliberately no policy: it documents that
 * the "Revolve/binomial scheduler (when to evict and in which order to replay)
 * is intentionally out of scope". This header supplies that policy for the
 * common case where the whole sequence of `steps` segments is known up front
 * and the caller can hold at most `slots` checkpoint states.
 *
 * # Recurrence
 *
 * The schedule is the classical Griewank binomial (Revolve) construction
 * (Griewank & Walther, *Algorithm 799: revolve*, ACM TOMS 26(1), 2000):
 *
 *   - `binomial(n, c)` is the ordinary binomial coefficient C(n, c), computed
 *     directly with exact intermediate divisions and saturation (no table).
 *   - With `c` checkpoints and `r` repetitions the maximum number of steps
 *     that can be traversed is `capacity(c, r) = binomial(c + r, c)`
 *     (the `MAXRGE` function of Algorithm 799). The minimal repetition depth
 *     for `n` steps is therefore `r = min{ r : capacity(c, r) >= n }`, the
 *     `NUMFRW` recurrence; the total number of forward steps is
 *     `r*n - capacity(c, r)*r/(c+1)`.
 *   - Snapshot placement is the forward phase of Algorithm 799's `revolve`
 *     controller ("binomial tree"): starting from `capo = 0`, the next
 *     snapshot state is the binomial offset
 *       `bino1 = C(N-1, ds)`, `bino4 = C(N-2, ds)`, ...
 *     with `N = ds + reps`; the branch among the five offsets is the
 *     divisibility test of the paper (all `size_t` arithmetic is exact:
 *     every offset is a binomial coefficient below the current range and is
 *     therefore representable whenever `steps` is). This yields at most
 *     `slots` snapshot states in the first pass.
 *
 * # Interpretation of `slots`
 *
 * `slots` is the number of checkpoint states the caller can keep *besides* the
 * initial state at step 0 (which every caller has for free). Internally the
 * scheduler uses `snaps = min(slots + 1, steps)` total simultaneous snapshot
 * slots, matching Algorithm 799's `SNAPS` count of stored states. Note that
 * the naive selection `b = max{ c : C(n, c) <= slots }` is degenerate for
 * realistic budgets (e.g. `n = 16, slots = 4` gives `b = 0`, no checkpoint at
 * all); this scheduler uses the capacity recurrence above instead, for which
 * `n = 16, slots = 4` stores four checkpoints at a 1.56x recompute bound.
 *
 * # Semantics
 *
 * A `steps`-segment sequence has states `0 ... steps`; segment `i` advances
 * state `i -> i+1`. A checkpoint at state `c` stores the state *after* segment
 * `c-1`. The interface is:
 *
 *   - `checkpointSteps()`: the states to snapshot (excluding the free state 0),
 *     strictly increasing, at most `slots` of them.
 *   - `restartPoint(step)`: the largest scheduled checkpoint `<= step`, or 0
 *     when there is none (the initial state always restarts the sequence).
 *   - `replayPlan(from, to)`: a contiguous list of half-open segment ranges
 *     `[a, b)` that tiles `[restartPoint(from), to]`; every range starts at a
 *     checkpoint (or 0) and re-recording the ranges in order reaches `to`.
 *   - `shouldKeep(step)` / `advance(step, willRecompute)`: the online budget
 *     driver. A caller recording segment by segment calls `advance(state)` at
 *     each boundary; `shouldKeep(state)` answers whether that recording stays
 *     a checkpoint. `willRecompute` records that the caller will replay the
 *     state later; `release(state)` drops a snapshot early.
 *
 * The scheduler owns integer bookkeeping only: it never touches a tape and
 * carries no per-step user data.
 */

/**
 * Immutable, offline schedule of checkpoint states for `steps` segments with
 * a budget of `slots` simultaneous checkpoints.
 *
 * The schedule is a pure value: constructing it is O(steps + slots) time and
 * memory (two `steps`-sized flag vectors), while all queries are O(log slots).
 */
class BinomialCheckpointSchedule {
public:
    /// Saturation marker for binomial coefficients beyond `size_t` range.
    static constexpr std::size_t kSaturated = std::numeric_limits<std::size_t>::max();

    /// Ordinary binomial coefficient C(n, c), exact below saturation.
    ///
    /// Computed with the symmetric multiplicative recurrence
    /// `C(n, i+1) = C(n, i) * (n-i) / (i+1)` capped at `c = min(c, n-c)`,
    /// so every intermediate is an earlier binomial of the same row and the
    /// division is always exact. Returns `kSaturated` when the coefficient
    /// exceeds the `size_t` range (callers compare against `steps`, so a
    /// saturated value simply reads as "larger than any step count").
    static std::size_t binomial(std::size_t n, std::size_t c) {
        if (c > n) {
            return 0;
        }
        c = std::min(c, n - c);
        std::size_t result = 1;
        for (std::size_t i = 0; i < c; ++i) {
            const std::size_t numerator = n - i;
            if (result > kSaturated / numerator) {
                return kSaturated;
            }
            result = result * numerator / (i + 1);
        }
        return result;
    }

    /// Griewank capacity: maximum steps with `checkpoints` checkpoints and
    /// `repetitions` repetitions, `binomial(checkpoints + repetitions,
    /// checkpoints)` (Algorithm 799 `MAXRGE`), saturating.
    static std::size_t capacity(std::size_t checkpoints, std::size_t repetitions) {
        return binomial(checkpoints + repetitions, checkpoints);
    }

    /// Minimal repetition depth `r` with `capacity(checkpoints, r) >= steps`
    /// (Algorithm 799 `NUMFRW`). Zero for `steps == 0`; for `checkpoints == 0`
    /// no schedule exists and the full length is returned as the depth.
    static std::size_t minimalRepetitions(std::size_t checkpoints, std::size_t steps) {
        if (steps == 0) {
            return 0;
        }
        if (checkpoints == 0) {
            return steps;
        }
        std::size_t repetitions = 0;
        while (capacity(checkpoints, repetitions) < steps) {
            ++repetitions;
        }
        return repetitions;
    }

    /// Build the schedule for `steps` segments and `slots` checkpoint states
    /// (state 0 excluded: the caller always has it).
    explicit BinomialCheckpointSchedule(std::size_t steps, std::size_t slots)
        : m_steps(steps), m_slots(slots) {
        // Algorithm 799 counts the initial state among SNAPS; the caller gets
        // it for free, so the internal budget is slots + 1, capped at steps
        // (there is nothing to store beyond one checkpoint per transition).
        const std::size_t snaps = slots >= steps ? steps : slots + 1;
        if (snaps >= 2 && steps >= 2) {
            m_checkpoints = firstPassSnapshots(steps, snaps);
        }
        m_scheduled.assign(steps + 1, 0);
        for (const std::size_t step : m_checkpoints) {
            m_scheduled[step] = 1;
        }
        m_liveFlags.assign(steps + 1, 0);
        m_repetitions = minimalRepetitions(snaps, steps);
        if (snaps == 0 || steps == 0) {
            m_forwardSteps = steps;
        } else {
            const std::size_t range = capacity(snaps, m_repetitions);
            m_forwardSteps = m_repetitions * steps - range * m_repetitions / (snaps + 1);
        }
    }

    std::size_t steps() const noexcept { return m_steps; }
    std::size_t slots() const noexcept { return m_slots; }

    /// Number of stored checkpoint states (the free initial state 0 excluded).
    std::size_t checkpoints() const noexcept { return m_checkpoints.size(); }

    /// The states to snapshot, strictly increasing and each in `(0, steps)`.
    std::vector<std::size_t> checkpointSteps() const { return m_checkpoints; }

    /// Minimal repetition depth of the underlying Griewank recurrence.
    std::size_t repetitionDepth() const noexcept { return m_repetitions; }

    /// Algorithm 799 `NUMFRW` estimate of total forward steps (initial sweep
    /// plus recomputations) for the dynamic Revolve schedule; the static
    /// interval replay implemented here re-records the segments of each
    /// checkpoint interval at most once, i.e. at most `2 * steps` total.
    std::size_t estimatedForwardSteps() const noexcept { return m_forwardSteps; }

    /// `estimatedForwardSteps() / steps()` (0 when there are no steps).
    double estimatedRecomputeFactor() const noexcept {
        return m_steps == 0 ? 0.0
                            : static_cast<double>(m_forwardSteps) / static_cast<double>(m_steps);
    }

    /// Largest scheduled checkpoint `<= step`; 0 when there is none.
    std::size_t restartPoint(std::size_t step) const {
        if (step > m_steps) {
            step = m_steps;
        }
        const auto it = std::upper_bound(m_checkpoints.begin(), m_checkpoints.end(), step);
        if (it == m_checkpoints.begin()) {
            return 0;
        }
        return *std::prev(it);
    }

    /// Contiguous half-open segment ranges to re-record to reach `to` from the
    /// nearest checkpoint at or before `from`.
    ///
    /// The plan tiles `[restartPoint(from), to]` (`to` clamped to `steps`):
    /// ranges are non-empty, adjacent (`plan[i].second == plan[i+1].first`),
    /// every range starts at a checkpoint (or at 0) and the last range ends at
    /// `to`. An empty plan is returned when `steps == 0` or `from >= to`.
    std::vector<std::pair<std::size_t, std::size_t>> replayPlan(std::size_t from,
                                                                std::size_t to) const {
        std::vector<std::pair<std::size_t, std::size_t>> plan;
        if (m_steps == 0 || from >= to) {
            return plan;
        }
        if (to > m_steps) {
            to = m_steps;
        }
        if (from > to) {
            from = to;
        }
        std::size_t pos = restartPoint(from);
        while (pos < to) {
            const auto it = std::upper_bound(m_checkpoints.begin(), m_checkpoints.end(), pos);
            std::size_t next = to;
            if (it != m_checkpoints.end()) {
                next = std::min(*it, to);
            }
            if (next <= pos) {
                next = to;
            }
            plan.emplace_back(pos, next);
            pos = next;
        }
        return plan;
    }

    /// True when `step` is a scheduled checkpoint snapshot.
    bool shouldKeep(std::size_t step) const noexcept {
        return step <= m_steps && m_scheduled[step] != 0;
    }

    /// Online budget driver: record that the caller reached state `step`.
    ///
    /// When `step` is a scheduled checkpoint and a slot is free it becomes
    /// live (`liveCheckpoints()` grows); `willRecompute` marks that the state
    /// will be replayed later and is accumulated in `recomputeSteps()`. Only
    /// monotone forward movement matters; `step` is clamped to `steps`.
    void advance(std::size_t step, bool willRecompute) {
        if (step > m_steps) {
            step = m_steps;
        }
        if (step > m_current) {
            m_current = step;
        }
        if (willRecompute) {
            ++m_recomputeSteps;
        }
        if (shouldKeep(step) && m_liveFlags[step] == 0 && m_live < m_slots) {
            m_liveFlags[step] = 1;
            ++m_live;
        }
    }

    /// Drop a live checkpoint snapshot (e.g. after its tape was swept).
    void release(std::size_t step) noexcept {
        if (step <= m_steps && m_liveFlags[step] != 0) {
            m_liveFlags[step] = 0;
            --m_live;
        }
    }

    /// Last state reached through `advance`.
    std::size_t currentStep() const noexcept { return m_current; }

    /// Number of checkpoint snapshots currently held (always `<= slots`).
    std::size_t liveCheckpoints() const noexcept { return m_live; }

    /// Number of `advance` calls that declared `willRecompute == true`.
    std::size_t recomputeSteps() const noexcept { return m_recomputeSteps; }

    /// Release every snapshot and reset the driver counters.
    void reset() noexcept {
        std::fill(m_liveFlags.begin(), m_liveFlags.end(), 0);
        m_live = 0;
        m_current = 0;
        m_recomputeSteps = 0;
    }

private:
    static std::size_t saturatingAdd(std::size_t a, std::size_t b) {
        return a > kSaturated - b ? kSaturated : a + b;
    }

    static std::size_t saturatingSub(std::size_t a, std::size_t b) { return a < b ? 0 : a - b; }

    /// Forward phase of Algorithm 799's `revolve` controller: walk the state
    /// forward using the binomial offsets and collect the snapshot states
    /// (state 0 excluded). `snaps` is the total simultaneous budget including
    /// state 0.
    static std::vector<std::size_t> firstPassSnapshots(std::size_t steps, std::size_t snaps) {
        std::vector<std::size_t> stored;
        if (steps <= 1 || snaps < 2) {
            return stored;
        }
        std::size_t capo = 0;  // start of the interval under construction
        std::size_t taken = 0; // snapshots taken, counting the free state 0
        while (steps - capo > 1) {
            ++taken;
            if (taken > 1) {
                // The state reached by the previous advance is snapshotted.
                stored.push_back(capo);
            }
            if (taken > snaps) {
                break; // out of simultaneous slots: reverse phase begins
            }
            // The snapshot just taken is released by the recursion; ds counts
            // the total slots available to the remaining interval.
            const std::size_t ds = snaps - (taken - 1);
            const std::size_t d = steps - capo;
            const std::size_t reps = minimalRepetitions(ds, d);
            const std::size_t n = ds + reps;
            // Algorithm 799's five binomial offsets, written as the binomial
            // coefficients they exactly equal (MAXRGE's multiply/divide
            // recurrence, cross-checked against the original source).
            const std::size_t bino1 = binomial(n - 1, ds);
            const std::size_t bino2 = ds > 1 ? binomial(n - 2, ds - 1) : 1;
            const std::size_t bino3 = ds == 1 ? 0 : (ds > 2 ? binomial(n - 3, ds - 2) : 1);
            const std::size_t bino4 = binomial(n - 2, ds);
            const std::size_t bino5 = ds < 3 ? 0 : (ds > 3 ? binomial(n - 3, ds - 3) : 1);
            const std::size_t previous = capo;
            if (d <= saturatingAdd(bino1, bino3)) {
                capo = saturatingAdd(capo, bino4);
            } else if (d >= saturatingSub(capacity(ds, reps), bino5)) {
                capo = saturatingAdd(capo, bino1);
            } else {
                const std::size_t skip = saturatingAdd(bino2, bino3);
                capo = skip < d ? steps - skip : capo + 1;
            }
            if (capo <= previous) {
                capo = previous + 1;
            }
            if (capo + 1 >= steps) {
                break; // the final transition is the first reverse turn
            }
        }
        return stored;
    }

    std::size_t m_steps = 0;
    std::size_t m_slots = 0;
    std::size_t m_repetitions = 0;
    std::size_t m_forwardSteps = 0;
    std::vector<std::size_t> m_checkpoints;
    std::vector<unsigned char> m_scheduled;
    std::vector<unsigned char> m_liveFlags;
    std::size_t m_current = 0;
    std::size_t m_live = 0;
    std::size_t m_recomputeSteps = 0;
};

/**
 * Callback driver that glues a `BinomialCheckpointSchedule` to a sequence of
 * lightape `Segment`/`SegmentRecorder` callbacks.
 *
 * The caller registers one `CheckpointCallback` per segment (segment `i`
 * advances state `i -> i+1`) with `store(i, cb)`; each callback re-executes
 * the segment on the active tape and writes fresh output handles into its own
 * context. `replayTo(state)` then runs the stored callbacks in order from the
 * current state up to `state`, returning the state reached, so a caller that
 * restored the primal values at `restartPoint(from)` can reconstruct the tape
 * through `to` with `replayTo(to)`.
 *
 * Pure callback-based (function pointers + `void*`, no `std::function`, no
 * allocation beyond the callback table); the schedule is borrowed and must
 * outlive the driver.
 */
class SegmentSchedule {
public:
    explicit SegmentSchedule(const BinomialCheckpointSchedule& schedule) noexcept
        : m_schedule(&schedule) {}

    /// Register the re-execution hook for segment `step`. Re-registering
    /// replaces the previous hook. Holes (unregistered steps) are allowed:
    /// `replayTo` skips their execution but still advances the state.
    void store(std::size_t step, CheckpointCallback callback) {
        if (step >= m_callbacks.size()) {
            m_callbacks.resize(step + 1);
        }
        m_callbacks[step] = callback;
    }

    /// Re-run every registered callback for segments `[currentStep(), step)`
    /// in increasing order and return the state reached (`step`, clamped to
    /// the schedule's `steps()`, or the unchanged current state when it is
    /// already at/past `step`). Every executed callback counts once in
    /// `recordings()`.
    std::size_t replayTo(std::size_t step) {
        if (step > m_schedule->steps()) {
            step = m_schedule->steps();
        }
        while (m_current < step) {
            if (m_current < m_callbacks.size() && m_callbacks[m_current].replay()) {
                ++m_recordings;
            }
            ++m_current;
        }
        return m_current;
    }

    /// Largest scheduled checkpoint at or before `step` (delegates to the
    /// schedule; 0 means the initial state).
    std::size_t restartPoint(std::size_t step) const { return m_schedule->restartPoint(step); }

    /// Move the cursor without replaying, e.g. after restoring the primal
    /// state stored at `step`.
    void resetTo(std::size_t step) noexcept { m_current = std::min(step, m_schedule->steps()); }

    std::size_t currentStep() const noexcept { return m_current; }

    /// Number of callbacks actually re-executed by `replayTo`.
    std::size_t recordings() const noexcept { return m_recordings; }

    void resetRecordings() noexcept { m_recordings = 0; }

    const BinomialCheckpointSchedule& schedule() const noexcept { return *m_schedule; }

private:
    const BinomialCheckpointSchedule* m_schedule;
    std::vector<CheckpointCallback> m_callbacks;
    std::size_t m_current = 0;
    std::size_t m_recordings = 0;
};

} // namespace lightape

#endif // LIGHTAPE_REV_REVOLVE_H
