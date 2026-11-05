#ifndef LIGHTAPE_REV_CHECKPOINT_H
#define LIGHTAPE_REV_CHECKPOINT_H

#include <cstddef>
#include <utility>
#include <vector>

#include "lightape/rev/LeanReverse.h"

namespace lightape {
/**
 * @file Checkpoint.h
 * @brief Nested recording scopes and segment bookkeeping over the lean tape.
 *
 * Bookkeeping layer built strictly on `RevTape`'s public lifecycle API
 * (`active()`, `position()`, `resetTo()`), implementing the nesting + segment
 * primitive of decisions #8/#20:
 *
 * - `ScopedNestedRecording` (aliased `NestedTape`) is the *nested recording*
 *   window: it captures the tape position on construction and truncates back
 *   to it on scope exit unless `keep()` is called. `TapeScope` looks similar
 *   but has a different intent: `TapeScope` is a generic truncation window
 *   for scratch nodes, while a nested recording exists to capture one
 *   sub-computation (e.g. one ODE step) that will be replayed later at a
 *   different primal point.
 * - `SegmentRecorder` packages one sub-computation as a `Segment`
 *   (`begin`, `end`, output handles); `close()` keeps the recording,
 *   `abort()` rolls the tape back to `begin`.
 * - `Segment` carries an optional `CheckpointCallback`. `Segment::replay()`
 *   runs it. This is the only replay mechanism: the tape stores primal values
 *   and opcodes, never the source computation, so it cannot re-execute a
 *   segment on its own (there is deliberately no
 *   `replay(const Segment&, const std::vector<RevScalar>&)`); the caller
 *   re-executes through the callback.
 *
 * Exactly like `TapeScope`, these windows require LIFO discipline: no other
 * tape mutation may interleave with an open scope/recorder. Multiple windows
 * may nest, each inner one closed/aborted before its outer one.
 *
 * Checkpointing policy (decision #20): per-path record -> sweep -> clear
 * stays the default for short tapes. For long paths, record segment by segment
 * in forward order, each segment's inputs being the previous segment's
 * outputs, and close each segment with its outputs. When memory must be freed,
 * `resetTo(segment.begin)` once the segment has been swept, then re-record it
 * later with `segment.replay()` before the next sweep. The Revolve/binomial
 * *scheduler* (when to evict and in which order to replay) is intentionally
 * out of scope here: `Segment` is the building block, not the scheduler.
 */

/// Optional, caller-owned re-execution hook stored in a `Segment`.
///
/// `fn` is a plain function pointer (no `std::function`, no virtuals); `ctx`
/// is a caller-owned pointer carrying whatever the re-execution needs (state,
/// inputs, output handles). A default-constructed callback has `fn == nullptr`
/// and `replay()` returns false. Lifetime: `ctx` and everything it references
/// must stay alive for as long as the segment may be replayed; the callback
/// owns nothing.
class CheckpointCallback {
public:
    using ReplayFn = void (*)(void* ctx);

    void* ctx = nullptr;
    ReplayFn fn = nullptr;

    /// Run the hook once; true when a hook was installed.
    bool replay() const {
        if (fn == nullptr) {
            return false;
        }
        fn(ctx);
        return true;
    }
};

/// One recorded tape window `[begin, end)` plus the sub-computation's output
/// handles.
///
/// @warning The output handles (and every node id recorded inside the window)
/// are only valid until the next `resetTo` at or before `begin` (or
/// `clear()`): truncation invalidates all slots at/after the reset position
/// and the arena reuses them for later recordings. The copied primal `value`s
/// stay readable, but the `node` ids become stale; after a truncation the only
/// way to get valid handles back is `replay()`, which re-executes the
/// sub-computation and writes fresh handles through its context.
class Segment {
public:
    std::size_t begin = 0;
    std::size_t end = 0;
    std::vector<RevScalar> outputs;

    /// Caller-owned replay hook; `fn == nullptr` means "recording only".
    CheckpointCallback callback;

    /// Re-run the stored callback; false when the segment has no callback.
    bool replay() const { return callback.replay(); }
};

/// Records one segment on `tape` without truncating on close.
///
/// The constructor captures `tape.position()` as the window begin; `close()`
/// packages `[begin, position())` and the outputs, while `abort()` rolls the
/// tape back to the begin and leaves it exactly as it was before the recorder
/// existed. If neither runs, the destructor aborts. Single-shot: call
/// `close()` or `abort()` at most once. LIFO discipline: no other tape
/// mutation may interleave with an open recorder.
class SegmentRecorder {
public:
    explicit SegmentRecorder(RevTape& tape) noexcept : m_tape(&tape), m_begin(tape.position()) {}
    ~SegmentRecorder() { abort(); }

    SegmentRecorder(const SegmentRecorder&) = delete;
    SegmentRecorder& operator=(const SegmentRecorder&) = delete;

    std::size_t beginPosition() const noexcept { return m_begin; }

    /// Close the segment; the recording stays on the tape.
    Segment close(std::vector<RevScalar> outputs) {
        Segment segment;
        segment.begin = m_begin;
        segment.end = m_tape->position();
        segment.outputs = std::move(outputs);
        m_tape = nullptr;
        return segment;
    }

    /// Discard the partial recording: the tape is left exactly as it was
    /// before the recorder was constructed.
    void abort() noexcept {
        if (m_tape != nullptr) {
            m_tape->resetTo(m_begin);
            m_tape = nullptr;
        }
    }

private:
    RevTape* m_tape;
    std::size_t m_begin;
};

/// Nested recording scope on `RevTape::active()` (default constructor) or an
/// explicit tape.
///
/// The intent-distinct sibling of `TapeScope`: both truncate on scope exit,
/// but a nested recording marks a sub-computation being recorded for later
/// replay, so the caller extracts a `Segment` (or keeps the output handles)
/// before the scope truncates them away, or calls `keep()` to let the
/// recording survive.
class ScopedNestedRecording {
public:
    explicit ScopedNestedRecording(RevTape& tape) noexcept
        : m_tape(&tape), m_begin(tape.position()) {}
    ScopedNestedRecording() : ScopedNestedRecording(RevTape::active()) {}

    ~ScopedNestedRecording() {
        if (m_tape != nullptr) {
            m_tape->resetTo(m_begin);
        }
    }

    ScopedNestedRecording(const ScopedNestedRecording&) = delete;
    ScopedNestedRecording& operator=(const ScopedNestedRecording&) = delete;

    std::size_t beginPosition() const noexcept { return m_begin; }

    /// Keep everything recorded inside this scope (skip the truncation).
    void keep() noexcept { m_tape = nullptr; }

private:
    RevTape* m_tape;
    std::size_t m_begin;
};

/// Name used for the nested recording scope in the checkpointing plan.
using NestedTape = ScopedNestedRecording;

} // namespace lightape

#endif // LIGHTAPE_REV_CHECKPOINT_H
