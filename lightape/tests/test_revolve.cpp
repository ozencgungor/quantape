#include <cstddef>
#include <cstdint>
#include <string>
#include <utility>
#include <vector>

#include "lightape/lightape.h"
#include "lightape/rev/Revolve.h"
#include <gtest/gtest.h>

namespace {

using lightape::BinomialCheckpointSchedule;
using lightape::CheckpointCallback;
using lightape::RevScalar;
using lightape::RevTape;
using lightape::Segment;
using lightape::SegmentRecorder;
using lightape::SegmentSchedule;

// ── scheduler: exact binomial coefficients and the Griewank recurrence ──

TEST(BinomialCheckpointSchedule, BinomialCoefficientsAndCapacity) {
    EXPECT_EQ(BinomialCheckpointSchedule::binomial(0, 0), 1u);
    EXPECT_EQ(BinomialCheckpointSchedule::binomial(5, 2), 10u);
    EXPECT_EQ(BinomialCheckpointSchedule::binomial(10, 5), 252u);
    EXPECT_EQ(BinomialCheckpointSchedule::binomial(52, 5), 2598960u);
    EXPECT_EQ(BinomialCheckpointSchedule::binomial(10, 3),
              BinomialCheckpointSchedule::binomial(10, 7));
    EXPECT_EQ(BinomialCheckpointSchedule::binomial(5, 7), 0u);

    EXPECT_EQ(BinomialCheckpointSchedule::capacity(4, 3), 35u); // C(7,4)
    EXPECT_EQ(BinomialCheckpointSchedule::capacity(5, 2), 21u); // C(7,5)
    EXPECT_EQ(BinomialCheckpointSchedule::capacity(1, 15), 16u);

    // Minimal repetitions from the recurrence: C(6,4) = 15 < 16 <= C(7,4).
    EXPECT_EQ(BinomialCheckpointSchedule::minimalRepetitions(4, 16), 3u);
    EXPECT_EQ(BinomialCheckpointSchedule::minimalRepetitions(5, 16), 2u);
    EXPECT_EQ(BinomialCheckpointSchedule::minimalRepetitions(2, 64), 10u);
    EXPECT_EQ(BinomialCheckpointSchedule::minimalRepetitions(0, 0), 0u);

    // Saturation is a clamp, never wrap-around.
    EXPECT_EQ(BinomialCheckpointSchedule::binomial(200, 100),
              BinomialCheckpointSchedule::kSaturated);
}

TEST(BinomialCheckpointSchedule, CanonicalRevolvePlacement) {
    // First-pass snapshot states of Algorithm 799: with the internal budget
    // snaps = slots + 1 (state 0 free) these are the classic placements.
    const BinomialCheckpointSchedule a(16, 4);
    EXPECT_EQ(a.checkpointSteps(), (std::vector<std::size_t>{6, 9, 11, 13}));
    EXPECT_EQ(a.checkpoints(), 4u);
    EXPECT_EQ(a.estimatedForwardSteps(), 25u); // NUMFRW(16, 5)
    EXPECT_LE(a.estimatedRecomputeFactor(), 3.0);

    const BinomialCheckpointSchedule b(32, 4);
    EXPECT_EQ(b.checkpointSteps(), (std::vector<std::size_t>{7, 16, 23, 28}));

    const BinomialCheckpointSchedule c(64, 8);
    EXPECT_EQ(c.checkpointSteps(), (std::vector<std::size_t>{10, 19, 27, 34, 40, 48, 55, 60}));
    EXPECT_EQ(c.checkpoints(), 8u);

    // More slots than steps: store every state and need no recompute.
    const BinomialCheckpointSchedule dense(6, 8);
    EXPECT_EQ(dense.checkpointSteps(), (std::vector<std::size_t>{1, 2, 3, 4}));
    EXPECT_LE(dense.estimatedForwardSteps(), dense.steps());
}

// ── scheduler: exhaustive structural properties, n <= 64, slots <= 8 ──

TEST(BinomialCheckpointSchedule, MemoryBoundAndRestartPointExhaustive) {
    for (std::size_t n = 0; n <= 64; ++n) {
        for (std::size_t slots = 0; slots <= 8; ++slots) {
            const BinomialCheckpointSchedule schedule(n, slots);
            SCOPED_TRACE("n=" + std::to_string(n) + " slots=" + std::to_string(slots));

            EXPECT_EQ(schedule.steps(), n);
            EXPECT_EQ(schedule.slots(), slots);

            const std::vector<std::size_t> steps = schedule.checkpointSteps();
            EXPECT_EQ(steps.size(), schedule.checkpoints());
            EXPECT_LE(steps.size(), slots); // memory bound
            EXPECT_LE(steps.size(), n > 0 ? n - 1 : 0);
            for (std::size_t i = 0; i < steps.size(); ++i) {
                EXPECT_LT(steps[i], n);
                EXPECT_GT(steps[i], 0u);
                if (i > 0) {
                    EXPECT_LT(steps[i - 1], steps[i]); // strictly increasing
                }
                EXPECT_TRUE(schedule.shouldKeep(steps[i]));
            }

            // restartPoint is the largest checkpoint <= step, 0 by default.
            for (std::size_t step = 0; step <= n; ++step) {
                const std::size_t restart = schedule.restartPoint(step);
                EXPECT_LE(restart, step);
                EXPECT_TRUE(restart == 0 || schedule.shouldKeep(restart));
                for (const std::size_t checkpoint : steps) {
                    if (checkpoint <= step) {
                        EXPECT_LE(checkpoint, restart);
                    }
                }
                if (step > 0) {
                    EXPECT_LE(schedule.restartPoint(step - 1), restart); // monotone
                }
            }
            EXPECT_EQ(schedule.restartPoint(n), steps.empty() ? 0 : steps.back());
        }
    }
}

TEST(BinomialCheckpointSchedule, ReplayPlanCoversIntervalExhaustively) {
    for (std::size_t n = 0; n <= 64; ++n) {
        for (std::size_t slots = 0; slots <= 8; ++slots) {
            const BinomialCheckpointSchedule schedule(n, slots);
            for (std::size_t from = 0; from <= n; ++from) {
                for (std::size_t to = from; to <= n; ++to) {
                    SCOPED_TRACE("n=" + std::to_string(n) + " slots=" + std::to_string(slots) +
                                 " from=" + std::to_string(from) + " to=" + std::to_string(to));
                    const auto plan = schedule.replayPlan(from, to);
                    if (from == to) {
                        EXPECT_TRUE(plan.empty());
                        continue;
                    }
                    ASSERT_FALSE(plan.empty());
                    EXPECT_EQ(plan.front().first, schedule.restartPoint(from));
                    EXPECT_LE(plan.front().first, from);
                    EXPECT_EQ(plan.back().second, to);
                    for (std::size_t i = 0; i < plan.size(); ++i) {
                        EXPECT_LT(plan[i].first, plan[i].second);
                        EXPECT_TRUE(plan[i].first == 0 || schedule.shouldKeep(plan[i].first));
                        if (i > 0) {
                            EXPECT_EQ(plan[i - 1].second, plan[i].first); // contiguous
                        }
                    }
                    // Tiles [restartPoint(from), to] and therefore [from, to].
                    EXPECT_LE(plan.front().first, from);
                    EXPECT_GE(plan.back().second, to);
                }
            }
        }
    }
}

TEST(BinomialCheckpointSchedule, OnlineBudgetDriver) {
    BinomialCheckpointSchedule schedule(16, 4);
    EXPECT_EQ(schedule.currentStep(), 0u);
    EXPECT_EQ(schedule.liveCheckpoints(), 0u);

    for (std::size_t step = 1; step <= 16; ++step) {
        schedule.advance(step, step % 2 == 0);
    }
    EXPECT_EQ(schedule.currentStep(), 16u);
    EXPECT_EQ(schedule.liveCheckpoints(), schedule.checkpoints());
    EXPECT_LE(schedule.liveCheckpoints(), schedule.slots());
    EXPECT_EQ(schedule.recomputeSteps(), 8u);

    schedule.release(6);
    EXPECT_EQ(schedule.liveCheckpoints(), schedule.checkpoints() - 1);
    schedule.reset();
    EXPECT_EQ(schedule.liveCheckpoints(), 0u);
    EXPECT_EQ(schedule.currentStep(), 0u);
    EXPECT_EQ(schedule.recomputeSteps(), 0u);
}

// ── callback driver without a tape ──

struct CountContext {
    int calls = 0;
};

void countCallback(void* ctx) {
    ++static_cast<CountContext*>(ctx)->calls;
}

TEST(SegmentSchedule, ReplaysCallbacksInOrder) {
    const BinomialCheckpointSchedule schedule(8, 2);
    SegmentSchedule driver(schedule);
    std::vector<CountContext> contexts(8);
    for (std::size_t i = 0; i < contexts.size(); ++i) {
        CheckpointCallback callback;
        callback.ctx = &contexts[i];
        callback.fn = &countCallback;
        driver.store(i, callback);
    }

    EXPECT_EQ(driver.restartPoint(8), schedule.restartPoint(8));
    EXPECT_EQ(driver.replayTo(8), 8u);
    EXPECT_EQ(driver.recordings(), 8u);
    for (const CountContext& context : contexts) {
        EXPECT_EQ(context.calls, 1);
    }
    // Already at the requested state: nothing re-runs.
    EXPECT_EQ(driver.replayTo(8), 8u);
    EXPECT_EQ(driver.recordings(), 8u);

    // Restore to the scheduled restart point and replay the tail.
    const std::size_t restart = driver.restartPoint(5);
    driver.resetTo(restart);
    driver.resetRecordings();
    EXPECT_EQ(driver.replayTo(6), 6u);
    EXPECT_EQ(driver.recordings(), 6u - restart);
    for (std::size_t i = 0; i < contexts.size(); ++i) {
        const int expected = (i >= restart && i < 6) ? 2 : 1;
        EXPECT_EQ(contexts[i].calls, expected) << "segment " << i;
    }

    // Unregistered holes advance the cursor without counting a recording.
    SegmentSchedule sparse(schedule);
    CountContext hole;
    CheckpointCallback callback;
    callback.ctx = &hole;
    callback.fn = &countCallback;
    sparse.store(3, callback);
    EXPECT_EQ(sparse.replayTo(5), 5u);
    EXPECT_EQ(sparse.recordings(), 1u);
    EXPECT_EQ(hole.calls, 1);
}

// ── integration: SegmentRecorder segments, checkpoint + replay ──

constexpr std::size_t kSteps = 16;
constexpr double kTolerance = 1e-15;

double scaleOf(std::size_t step) {
    return 1.0 + 0.01 * static_cast<double>(step + 1);
}
double shiftOf(std::size_t step) {
    return 0.05 * static_cast<double>(step) - 0.4;
}

/// Re-execution context of one segment: `output = input * scale + shift`,
/// recorded on `tape` inside a fresh `SegmentRecorder` (so a replay really
/// re-records a tape segment, not just a detached value).
struct StepContext {
    RevTape* tape = nullptr;
    RevScalar* input = nullptr;
    RevScalar output;
    Segment segment;
    double scale = 1.0;
    double shift = 0.0;
};

void replayStep(void* ctx) {
    auto& step = *static_cast<StepContext*>(ctx);
    SegmentRecorder recorder(*step.tape);
    step.output = *step.input * step.scale + step.shift;
    step.segment = recorder.close({step.output});
}

class RevolveIntegrationTest : public ::testing::Test {
protected:
    void SetUp() override { RevTape::active().clear(); }
    void TearDown() override { RevTape::active().clear(); }
};

TEST_F(RevolveIntegrationTest, CheckpointedReplayMatchesFullRecording) {
    RevTape& tape = RevTape::active();

    // Primal states of the 16-segment recurrence and the no-checkpoint
    // reference gradient, recorded once and then discarded.
    std::vector<double> primal(kSteps + 1);
    primal[0] = 0.7;
    for (std::size_t k = 0; k < kSteps; ++k) {
        primal[k + 1] = primal[k] * scaleOf(k) + shiftOf(k);
    }

    tape.clear();
    const std::uint32_t referenceInput = tape.input(primal[0]);
    RevScalar reference(primal[0], referenceInput);
    for (std::size_t k = 0; k < kSteps; ++k) {
        reference = reference * scaleOf(k) + shiftOf(k);
    }
    const double referenceValue = reference.value;
    tape.reverse(reference.node);
    const double referenceGradient = tape.adjoint(referenceInput);
    ASSERT_DOUBLE_EQ(referenceValue, primal[kSteps]);

    // Initial recording: one SegmentRecorder per step. Checkpoint states from
    // the schedule are marked kept; everything else is discarded below.
    const BinomialCheckpointSchedule schedule(kSteps, 4);
    ASSERT_EQ(schedule.checkpointSteps(), (std::vector<std::size_t>{6, 9, 11, 13}));

    tape.clear();
    std::vector<StepContext> steps(kSteps);
    std::vector<std::size_t> kept;
    std::size_t initialRecordings = 0;
    RevScalar state(primal[0], tape.input(primal[0]));
    for (std::size_t k = 0; k < kSteps; ++k) {
        steps[k].tape = &tape;
        steps[k].scale = scaleOf(k);
        steps[k].shift = shiftOf(k);
        SegmentRecorder recorder(tape);
        steps[k].output = state * scaleOf(k) + shiftOf(k);
        steps[k].segment = recorder.close({steps[k].output});
        state = steps[k].output;
        ++initialRecordings;
        if (schedule.shouldKeep(k + 1)) {
            kept.push_back(k + 1);
        }
    }
    EXPECT_EQ(initialRecordings, kSteps);
    EXPECT_EQ(kept, schedule.checkpointSteps());
    EXPECT_LE(kept.size(), schedule.slots());
    EXPECT_DOUBLE_EQ(steps.back().output.value, referenceValue);

    // Discard every recorded segment (checkpoint snapshots are the stored
    // primal states above; the tape is reclaimed wholesale) and replay from
    // the nearest checkpoint at or before state 0: the full chain.
    tape.clear();
    RevScalar entry(primal[0], tape.input(primal[0]));
    for (std::size_t k = 0; k < kSteps; ++k) {
        steps[k].input = (k == 0) ? &entry : &steps[k - 1].output;
    }
    SegmentSchedule driver(schedule);
    for (std::size_t k = 0; k < kSteps; ++k) {
        CheckpointCallback callback;
        callback.ctx = &steps[k];
        callback.fn = &replayStep;
        driver.store(k, callback);
    }
    driver.resetTo(0);
    driver.resetRecordings();
    EXPECT_EQ(driver.replayTo(kSteps), kSteps);
    EXPECT_EQ(driver.recordings(), kSteps);
    for (const StepContext& step : steps) {
        EXPECT_EQ(step.segment.end - step.segment.begin, 2u); // Mul + Add
        ASSERT_EQ(step.segment.outputs.size(), 1u);
    }

    // Recompute budget: initial pass + one replay pass is 2x, inside the
    // decision #20 bound of 1.5-3x.
    const std::size_t totalRecordings = initialRecordings + driver.recordings();
    EXPECT_LE(totalRecordings, 3 * kSteps);

    tape.reverse(steps.back().output.node);
    EXPECT_DOUBLE_EQ(steps.back().output.value, referenceValue);
    EXPECT_NEAR(tape.adjoint(entry.node), referenceGradient, kTolerance);

    // Tail replay from an intermediate checkpoint: rebuild only from
    // restartPoint(10) and check the gradient against a fresh recording that
    // starts at the same primal state.
    const std::size_t from = 10;
    const std::size_t restart = schedule.restartPoint(from);
    ASSERT_LE(restart, from);
    ASSERT_LT(restart, kSteps);

    tape.clear();
    const std::uint32_t tailInput = tape.input(primal[restart]);
    RevScalar tail(primal[restart], tailInput);
    for (std::size_t k = restart; k < kSteps; ++k) {
        tail = tail * scaleOf(k) + shiftOf(k);
    }
    tape.reverse(tail.node);
    const double tailGradient = tape.adjoint(tailInput);

    tape.clear();
    RevScalar tailEntry(primal[restart], tape.input(primal[restart]));
    for (std::size_t k = restart; k < kSteps; ++k) {
        steps[k].input = (k == restart) ? &tailEntry : &steps[k - 1].output;
    }
    driver.resetTo(restart);
    driver.resetRecordings();
    EXPECT_EQ(driver.replayTo(kSteps), kSteps);
    EXPECT_EQ(driver.recordings(), kSteps - restart);
    EXPECT_LE(driver.recordings(), kSteps);
    tape.reverse(steps.back().output.node);
    EXPECT_NEAR(tape.adjoint(tailEntry.node), tailGradient, kTolerance);
}

} // namespace
