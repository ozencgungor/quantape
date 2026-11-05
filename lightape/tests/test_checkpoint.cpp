#include <cstddef>
#include <cstdint>
#include <type_traits>

#include "lightape/lightape.h"
#include "lightape/rev/Checkpoint.h"
#include <gtest/gtest.h>

namespace {

using lightape::ExtArg;
using lightape::RevScalar;
using lightape::RevTape;
using lightape::ScopedNestedRecording;
using lightape::Segment;
using lightape::SegmentRecorder;

static_assert(std::is_same_v<lightape::NestedTape, ScopedNestedRecording>);

class CheckpointTest : public ::testing::Test {
protected:
    void SetUp() override { RevTape::active().clear(); }
    void TearDown() override { RevTape::active().clear(); }
};

void countReplay(void* ctx) {
    ++(*static_cast<int*>(ctx));
}

struct SquareReplay {
    RevScalar in;
    RevScalar out;
    int calls = 0;
};

void replaySquare(void* ctx) {
    auto& state = *static_cast<SquareReplay*>(ctx);
    ++state.calls;
    state.out = state.in * state.in;
}

struct Payload {
    double a = 0.0;
    double b = 0.0;
};

void payloadVjp(const void* payload, double seed, const ExtArg* args, std::uint32_t /*n*/,
                RevTape& tape) noexcept {
    const auto* p = static_cast<const Payload*>(payload);
    tape.addAdjoint(args[0].node, seed * p->a);
    tape.addAdjoint(args[1].node, seed * p->b);
}

TEST_F(CheckpointTest, NestedScopeTruncatesUnlessKept) {
    RevTape& tape = RevTape::active();
    RevScalar x(2.0, tape.input(2.0));
    const std::size_t base = tape.position();

    {
        ScopedNestedRecording scope; // defaults to RevTape::active()
        const RevScalar t = x * x + 1.0;
        EXPECT_DOUBLE_EQ(t.value, 5.0);
        EXPECT_EQ(tape.position(), base + 2);
    }
    EXPECT_EQ(tape.position(), base);

    RevScalar kept(0.0);
    {
        ScopedNestedRecording scope(tape);
        kept = x * x;
        scope.keep();
    }
    EXPECT_EQ(tape.position(), base + 1);
    tape.reverse(kept.node);
    EXPECT_DOUBLE_EQ(tape.adjoint(x.node), 4.0);
}

TEST_F(CheckpointTest, TwoNestedScopesRestoreLifo) {
    RevTape& tape = RevTape::active();
    const RevScalar x(1.0, tape.input(1.0));
    const std::size_t base = tape.position();
    {
        ScopedNestedRecording outer(tape);
        EXPECT_EQ(outer.beginPosition(), base);
        const RevScalar a(2.0, tape.input(2.0));
        const std::size_t afterA = tape.position();
        {
            ScopedNestedRecording inner(tape);
            EXPECT_EQ(inner.beginPosition(), afterA);
            const RevScalar b(3.0, tape.input(3.0));
            EXPECT_DOUBLE_EQ(b.value, 3.0);
            EXPECT_EQ(tape.position(), afterA + 1);
        }
        EXPECT_EQ(tape.position(), afterA);
        const RevScalar c(4.0, tape.input(4.0));
        EXPECT_EQ(c.node, static_cast<std::uint32_t>(afterA));
        EXPECT_DOUBLE_EQ(a.value, 2.0);
    }
    EXPECT_EQ(tape.position(), base);
    EXPECT_DOUBLE_EQ(x.value, 1.0);
}

TEST_F(CheckpointTest, AbortLeavesTapeUntouched) {
    RevTape& tape = RevTape::active();
    RevScalar x(1.5, tape.input(1.5));
    const std::size_t base = tape.position();
    {
        SegmentRecorder rec(tape);
        const RevScalar a = x * x;
        const RevScalar b = a + x;
        EXPECT_DOUBLE_EQ(a.value, 2.25);
        EXPECT_DOUBLE_EQ(b.value, 3.75);
        EXPECT_EQ(tape.position(), base + 2);
        rec.abort();
        EXPECT_EQ(tape.position(), base); // tape as before the recorder
    }
    EXPECT_EQ(tape.position(), base); // destructor is a no-op after abort

    {
        SegmentRecorder rec(tape);
        const RevScalar dropped = x * 2.0;
        EXPECT_DOUBLE_EQ(dropped.value, 3.0);
        EXPECT_EQ(tape.position(), base + 1);
    }
    EXPECT_EQ(tape.position(), base); // unclosed recorder rolls back

    const RevScalar f = x * 5.0;
    tape.reverse(f.node);
    EXPECT_DOUBLE_EQ(tape.adjoint(x.node), 5.0);
}

TEST_F(CheckpointTest, SegmentCloseKeepsRecordingAndReplayCounts) {
    RevTape& tape = RevTape::active();
    RevScalar x(2.0, tape.input(2.0));
    const std::size_t base = tape.position();

    SegmentRecorder rec(tape);
    const RevScalar y = x * x + 1.0;
    const Segment seg = rec.close({y});
    EXPECT_EQ(seg.begin, base);
    EXPECT_EQ(seg.end, base + 2);
    EXPECT_EQ(tape.position(), seg.end); // close keeps the recording

    ASSERT_EQ(seg.outputs.size(), 1u);
    EXPECT_EQ(seg.outputs[0].node, y.node);
    EXPECT_DOUBLE_EQ(seg.outputs[0].value, 5.0);

    int calls = 0;
    EXPECT_FALSE(seg.replay()); // no callback installed
    EXPECT_EQ(calls, 0);
    Segment replayable = seg;
    replayable.callback.ctx = &calls;
    replayable.callback.fn = &countReplay;
    EXPECT_TRUE(replayable.replay());
    EXPECT_TRUE(replayable.replay());
    EXPECT_EQ(calls, 2);

    tape.reverse(y.node);
    EXPECT_DOUBLE_EQ(tape.adjoint(x.node), 4.0);
}

TEST_F(CheckpointTest, SegmentCallbackReplaysAtNewPrimalPoint) {
    RevTape& tape = RevTape::active();
    const std::size_t base = tape.position();

    SquareReplay state;
    SegmentRecorder rec(tape);
    state.in = RevScalar(2.0, tape.input(2.0));
    state.out = state.in * state.in;
    Segment seg = rec.close({state.out});
    EXPECT_EQ(seg.begin, base);
    EXPECT_EQ(seg.end, base + 2);
    ASSERT_EQ(seg.outputs.size(), 1u);
    EXPECT_DOUBLE_EQ(seg.outputs[0].value, 4.0);

    // Evict the window, then replay the sub-computation at a new primal point
    // into the freed slots; the replayed window has the same shape.
    tape.resetTo(seg.begin);
    EXPECT_EQ(tape.position(), base);
    state.in = RevScalar(3.0, tape.input(3.0));
    state.out = RevScalar(0.0);
    seg.callback.ctx = &state;
    seg.callback.fn = &replaySquare;
    EXPECT_TRUE(seg.replay());
    EXPECT_EQ(state.calls, 1);
    EXPECT_DOUBLE_EQ(state.out.value, 9.0);
    EXPECT_EQ(state.out.node, static_cast<std::uint32_t>(base + 1));
    EXPECT_EQ(tape.position(), seg.end);

    tape.reverse(state.out.node);
    EXPECT_DOUBLE_EQ(tape.adjoint(state.in.node), 6.0);
}

TEST_F(CheckpointTest, SurvivingPrefixUsableAfterResetTo) {
    RevTape& tape = RevTape::active();
    RevScalar x(2.0, tape.input(2.0));
    RevScalar y(3.0, tape.input(3.0));
    const std::size_t prefixEnd = tape.position();

    SegmentRecorder rec(tape);
    const RevScalar step = x * y + 1.0;
    const Segment seg = rec.close({step});
    EXPECT_EQ(seg.begin, prefixEnd);
    EXPECT_EQ(seg.end, prefixEnd + 2);

    tape.resetTo(seg.begin);
    EXPECT_EQ(tape.position(), prefixEnd);

    // x and y survive the truncation; fresh work reuses the freed slots.
    const RevScalar f = x - y;
    EXPECT_EQ(f.node, static_cast<std::uint32_t>(prefixEnd));
    tape.reverse(f.node);
    EXPECT_DOUBLE_EQ(tape.adjoint(x.node), 1.0);
    EXPECT_DOUBLE_EQ(tape.adjoint(y.node), -1.0);
}

TEST_F(CheckpointTest, ExternalInsideSegmentTruncatesAndRerecords) {
    RevTape& tape = RevTape::active();
    RevScalar x(1.5, tape.input(1.5));
    RevScalar y(2.5, tape.input(2.5));
    const std::size_t base = tape.position();

    const RevScalar args[2] = {x, y};
    SegmentRecorder rec(tape);
    const Payload first{4.0, -3.0};
    const RevScalar dropped =
        lightape::detail::makeExternal(4.0 * 1.5 - 3.0 * 2.5, args, 2, payloadVjp, first);
    EXPECT_DOUBLE_EQ(dropped.value, -1.5);
    const Segment seg = rec.close({dropped});
    EXPECT_EQ(seg.begin, base);
    EXPECT_EQ(seg.end, base + 1);

    // Truncate the segment: its External record and payload copy must go with
    // it.
    tape.resetTo(seg.begin);
    tape.resetAdjoints();
    EXPECT_EQ(tape.position(), base);

    // Re-record in the same window with a different payload; the sweep must
    // use the new payload, not the truncated one.
    const Payload second{2.0, 5.0};
    const RevScalar again =
        lightape::detail::makeExternal(2.0 * 1.5 + 5.0 * 2.5, args, 2, payloadVjp, second);
    EXPECT_DOUBLE_EQ(again.value, 15.5);
    EXPECT_EQ(tape.position(), base + 1);
    tape.reverse(again.node);
    EXPECT_DOUBLE_EQ(tape.adjoint(x.node), 2.0);
    EXPECT_DOUBLE_EQ(tape.adjoint(y.node), 5.0);
}

} // namespace
