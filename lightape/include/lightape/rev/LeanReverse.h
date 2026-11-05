#ifndef LIGHTAPE_REV_LEAN_REVERSE_H
#define LIGHTAPE_REV_LEAN_REVERSE_H

#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <memory>
#include <new>
#include <type_traits>
#include <utility>
#include <vector>

#include "lightape/base/Digamma.h"

namespace lightape {
/**
 * @file LeanReverse.h
 * @brief Lean reverse-mode scalar (fixed-layout node arena, opcode sweep)
 *
 * A minimal replacement for general-purpose AD scalars in the pathwise
 * gradient layer. Same plug-in pattern as `Tangent` (the engine is
 * scalar-generic and uses ADL math): each operation appends one
 * fixed-layout 32-byte node to a thread-local arena and returns a
 * `RevScalar` holding (value, node index). The reverse pass walks the
 * arena backwards with a `switch` on the opcode — no virtual dispatch,
 * no per-node allocation, sequential memory.
 *
 * Two structural chages over Stan `var` in this hot path:
 * - **constants are node-less** (`RevScalar(double)`), so model-side
 *   scalings never become constant AD leaves (the many-factor proxy
 *   showed up to ~30% of reverse cost was such leaves);
 * - **no global-stack interaction**: the arena is a plain
 *   `thread_local RevTape` reused across paths (`clear()` keeps
 *   capacity), so there is no `nested_rev_autodiff`/`STAN_THREADS`
 *   machinery per path.
 *
 * Forward ops collapse to nothing when both operands are constants, so
 * constant subexpressions cost no nodes at all.
 *
 * Supported ops (the set the schemes/processes/payoffs use): + - * /,
 * unary -, sqrt, cbrt, exp, exp2, log, log2, log10, fabs, copysign,
 * fmin, fmax, asinh, acosh, atanh, signbit; comparisons on the primal.
 * Branches (`fmin/fmax`, Eigen `cwiseMax`, ternaries) share the selected
 * operand's node, so adjoints flow along the taken branch (a.e. correct).
 *
 * Record path (decisions #19; 2026-10 state-of-the-art study §2.4/§4.1):
 * `pushDirect` stages each node with one inline store at the cursor plus
 * an inline `m_size < m_capacity` check (`[[likely]]`), so steady-state
 * recording crosses no call boundary and only falls back to `ensureCapacity`
 * at a chunk boundary. Constant operands stay node-less: their primal lives
 * in the parent-value slot (`lhsValue`/`rhsValue`) and the parent slot is the
 * dead sink. The optional statement cache (`useCse(true)`, default off)
 * merges immediately-repeated `(op, lhs, rhs)` statements through a 4-entry
 * direct-mapped table; a cached node is returned only when the recomputed
 * primal is bitwise equal, otherwise a fresh node is recorded.
 */

enum class RevOp : std::uint8_t {
    Input,
    Add,
    Sub,
    Mul,
    Div,
    Neg,
    Sqrt,
    Exp,
    Log,
    Expm1,
    Log1p,
    Abs,
    Pow,
    Sin,
    Cos,
    Tan,
    Asin,
    Acos,
    Atan,
    Sinh,
    Cosh,
    Tanh,
    Erf,
    Erfc,
    LogGamma,
    TGamma,
    Hypot,
    Atan2,
    External,
    MinL,
    MinR,
    MaxL,
    MaxR,
    Cbrt,
    Exp2,
    Log2,
    Log10,
    CopySign,
    Digamma,
    Trigamma,
    Tetragamma,
    Asinh,
    Acosh,
    Atanh,
};

/// Tape node: value, adjoint, parent node indices *and parent values*.
/// Caching the parent values inline makes the reverse sweep independent of
/// parent-node reads (constants have no node; their values live here) and
/// keeps the sweep sequential.
struct RevNode {
    double value = 0.0;
    double adjoint = 0.0;
    double lhsValue = 0.0;
    double rhsValue = 0.0;
    std::uint32_t lhs = 0u;
    std::uint32_t rhs = 0u;
    RevOp op = RevOp::Input;
    std::uint32_t ext = 0u;
};

static_assert(sizeof(RevNode) == 48, "RevNode layout gate: 48-byte AoS node");

/// 2/sqrt(pi): erf/erfc partial factor (libm-free constant).
inline constexpr double kTwoOverSqrtPi = 1.12837916709551257390;

/// ln 2 / ln 10 (correctly rounded) for the exp2/log2/log10 partials.
inline constexpr double kLn2 = 0.69314718055994530942;
inline constexpr double kLn10 = 2.30258509299404568402;

class RevTape;

/// One parent of an `External` block: cached value + tape slot (0 = constant).
struct ExtArg {
    double value = 0.0;
    std::uint32_t node = 0u;
};

/// External VJP: given the block's payload and the incoming adjoint seed,
/// scatter `seed * d(out)/d(arg_i)` into the parent slots. Must be `noexcept`
/// and must not touch the tape beyond `addAdjoint`.
using ExtVjpFn = void (*)(const void* payload, double seed, const ExtArg* args, std::uint32_t nArgs,
                          RevTape& tape) noexcept;

/// External directional derivative (JVP) along parent tangents `sArgs`.
using ExtSvpFn = double (*)(const void* payload, const ExtArg* args, const double* sArgs,
                            std::uint32_t nArgs) noexcept;

/// External contracted second derivative used by the Hessian replay: for each
/// parent p, `bArgs[p] += adjoint * sum_q d2F/da_p da_q * sArgs[q]`.
using ExtVhpFn = void (*)(const void* payload, double adjoint, const ExtArg* args,
                          const double* sArgs, std::uint32_t nArgs, double* bArgs) noexcept;

struct RevScalar;

/// Thread-local reverse tape: chunked append-only arena (default 4096-node,
/// 64 B aligned chunks, per-instance size configurable with
/// `configureChunkSize`) with a manual cursor and chunks reused across paths.
/// Chunks are allocated on demand and kept until the thread exits, so a
/// per-path tape that fits in one chunk never allocates after the first use.
/// Arena scheme: configuring one chunk at least as large as the expected tape
/// gives a monolithic contiguous arena (best cache locality, one allocation),
/// while small chunks bound the reallocation cost of growth.
class RevTape {
public:
    /// Node 0 is a permanent dead sink: constant operands point at it, so
    /// the reverse sweep needs no null-parent branches; its adjoint is
    /// ignored.
    static constexpr std::uint32_t kNone = 0u;

    static constexpr std::size_t kChunkShift = 12;
    static constexpr std::size_t kChunkNodes = std::size_t{1} << kChunkShift;
    static constexpr std::size_t kChunkMask = kChunkNodes - 1;

    /// Shared accessor for the thread-local tape pointer. Trivially-
    /// initialized (no dynamic TLS init in the per-operation path); the arena
    /// itself is created once per thread on first `active()`.
    static RevTape*& currentPtr() {
        static thread_local RevTape* tape = nullptr;
        return tape;
    }

    static RevTape& active() {
        RevTape*& tape = currentPtr();
        if (tape == nullptr) [[unlikely]] {
            tape = new RevTape();
        }
        return *tape;
    }

    /// Destroy this thread's tape and forget it (a null tape is a no-op), so
    /// the next `active()` allocates a fresh tape. Must not be called while a
    /// TapeScope, an external payload/callback or any other code still
    /// references the tape; every `RevScalar` node index becomes invalid.
    /// Useful to reclaim the arena between phases.
    static void releaseCurrentThread();

    /// Run `f()` inside a TapeScope on the calling thread's tape; see the
    /// out-of-line definition for the truncation contract.
    template <typename F>
    static decltype(auto) run(F&& f);

    RevTape() {
        m_chunks.push_back(makeChunk(m_chunkNodes));
        m_capacity = m_chunkNodes;
    }

    /// Node count of every chunk (a power of two; defaults to `kChunkNodes`).
    std::size_t chunkSize() const noexcept { return m_chunkNodes; }

    /// Reconfigure the per-chunk node count before the first push. Legal only
    /// while the tape holds nothing but the sink (`m_size == 1`: freshly
    /// constructed or after `clear()`/`resetTo(1)`); otherwise a no-op.
    /// `chunkNodes` is rounded up to the next power of two with a minimum of
    /// 64, and the preallocated chunks are replaced by one fresh chunk of that
    /// size. Configuring a chunk at least as large as the expected tape gives
    /// a single monolithic contiguous arena (cache locality, one allocation);
    /// smaller chunks bound the reallocation cost of growth.
    void configureChunkSize(std::size_t chunkNodes) {
        if (m_size != 1) {
            return;
        }
        std::size_t size = 64;
        std::size_t shift = 6;
        while (size < chunkNodes) {
            size <<= 1;
            ++shift;
            if (size == 0) { // size_t overflow: clamp to the largest power of two
                size = std::size_t{1} << (sizeof(std::size_t) * 8 - 1);
                shift = sizeof(std::size_t) * 8 - 1;
                break;
            }
        }
        m_chunkShift = shift;
        m_chunkNodes = size;
        m_chunkMask = size - 1;
        m_chunks.clear();
        m_chunks.push_back(makeChunk(size));
        m_capacity = size;
        m_singleChunk = true;
        invalidateCse(); // chunks were replaced; cached slots are gone
    }

    /// Reset for a new recording: keep the sink and every allocated chunk;
    /// external records, argument cache and payload copies are released.
    /// The CSE cache is dropped (its node indices die with the recording);
    /// the `useCse` setting itself survives.
    void clear() {
        m_size = 1;
        m_ext.clear();
        m_extArgs.clear();
        invalidateCse();
    }

    /// Absolute node count including the sink (the next free index).
    std::size_t position() const noexcept { return m_size; }
    std::size_t size() const noexcept { return m_size - 1; } // excluding sink

    void reserve(std::size_t n) { ensureCapacity(n + 1); }

    /// Truncate to a previously recorded position (LIFO). Chunks stay
    /// allocated; surviving adjoints are preserved (call resetAdjoints()
    /// before re-sweeping a sub-tape). The CSE cache is dropped, so no
    /// truncated slot can ever be returned.
    void resetTo(std::size_t pos) {
        if (pos >= 1 && pos <= m_size) {
            m_size = pos;
            truncateExternal(pos);
            invalidateCse();
        }
    }

    /// Zero every adjoint (sink included) without touching the recording.
    void resetAdjoints() {
        for (std::size_t i = 0; i < m_size; ++i) {
            node(i).adjoint = 0.0;
        }
    }

    /// Append one op node: the public staging entry point. Node-less
    /// constants (`kNone`) point at the dead sink and cache their primal in
    /// the corresponding `lhsValue`/`rhsValue` slot, so no constant leaf is
    /// ever recorded. All ops except `Input` participate in the optional CSE
    /// cache (see `useCse`).
    std::uint32_t push(RevOp op, std::uint32_t lhs, std::uint32_t rhs, double value,
                       double lhsValue = 0.0, double rhsValue = 0.0) {
        return pushDirect(op, lhs, rhs, value, lhsValue, rhsValue);
    }

    /// Record-path primitive used by `revBinary`/`revUnary`: stage the node
    /// write directly at the cursor with the capacity check inline
    /// (`m_size < m_capacity` is the common case, marked `[[likely]]`), so
    /// the steady-state push calls into no function and stores the node into
    /// the arena. Only a full chunk (`m_size ==
    /// m_capacity`) falls back to `ensureCapacity`. `Input` nodes are never
    /// CSE candidates: two inputs with the same primal are distinct variables.
    std::uint32_t pushDirect(RevOp op, std::uint32_t lhs, std::uint32_t rhs, double value,
                             double lhsValue = 0.0, double rhsValue = 0.0) {
        if (m_useCse && op != RevOp::Input) [[unlikely]] {
            const std::uint32_t hit = cseLookup(op, lhs, rhs, value);
            if (hit != kNone) {
                return hit;
            }
        }
        if (m_size < m_capacity) [[likely]] {
            return stageNode(op, lhs, rhs, value, lhsValue, rhsValue);
        }
        ensureCapacity(m_size + 1);
        return stageNode(op, lhs, rhs, value, lhsValue, rhsValue);
    }

    /// Enable/disable record-time statement CSE (default **off**, so the
    /// default tape is byte-for-byte the pre-CSE behavior). When on,
    /// `push`/`pushDirect` consult a 4-entry direct-mapped cache keyed by
    /// `(op, lhs, rhs)`. A hit returns the existing node only when the
    /// recomputed primal is bitwise equal to the cached node's value;
    /// otherwise a fresh node is recorded (and replaces the entry). Sharing
    /// the node is the correct reverse-mode statement merge: the shared node
    /// accumulates the adjoints of all its uses before its single sweep.
    /// `Input` and `External` records never participate (`External` has its
    /// own `pushExternal` path). Toggling either way drops the cache;
    /// `clear()`/`resetTo()`/`configureChunkSize()` invalidate it so a
    /// truncated or reallocated slot can never be returned.
    void useCse(bool on) noexcept {
        m_useCse = on;
        invalidateCse();
    }
    bool cseEnabled() const noexcept { return m_useCse; }

    /// Independent variable (what gradients are read from).
    std::uint32_t input(double value) { return push(RevOp::Input, kNone, kNone, value); }

    double value(std::uint32_t idx) const noexcept { return node(idx).value; }
    double adjoint(std::uint32_t idx) const noexcept { return node(idx).adjoint; }

    /// Accumulate into a parent slot (called by External VJP callbacks).
    /// Constant slots (`kNone`) are ignored so the sink stays clean.
    void addAdjoint(std::uint32_t parent, double add) noexcept {
        if (parent == kNone) {
            return;
        }
        node(parent).adjoint += add;
    }

    /// Record one `External` block node over `nArgs` cached parent scalars
    /// (`lhs` = arg begin, `rhs` = nArgs, `ext` = side-table index). When
    /// `payload` is non-null its `payloadSize` bytes are copied into the tape
    /// (payloads must be trivially copyable; the callback receives the copy).
    std::uint32_t pushExternal(double value, ExtVjpFn vjp, const RevScalar* args,
                               std::uint32_t nArgs, const void* payload, std::size_t payloadSize,
                               ExtSvpFn svp = nullptr, ExtVhpFn vhp = nullptr);

    /// Reverse sweep seeded at `root`. Consume-on-use: non-leaf adjoints are
    /// zeroed as they are read (so a finished sweep leaves no stale state),
    /// while leaf (`Input`) adjoints are kept for the caller to read out.
    /// A tape that fits its first chunk (the common case) sweeps flat; longer
    /// tapes walk chunk by chunk backwards, where only parent updates index
    /// across chunks (power-of-two chunk size -> shift/mask).
    void reverse(std::uint32_t root, double seed = 1.0) { reverse(&root, &seed, 1); }

    /// Multi-root sweep: seed several outputs (e.g. softmax components) before
    /// one backward pass, so the whole Jacobian-transpose comes from a single
    /// sweep. Seeds replace any existing adjoint at those roots.
    void reverse(const std::uint32_t* roots, const double* seeds, std::size_t count) {
        for (std::size_t r = 0; r < count; ++r) {
            node(roots[r]).adjoint = seeds[r];
        }
        const std::size_t top = m_size - 1;
        if (top == 0) {
            return;
        }
        if (m_chunks.size() == 1) [[likely]] {
            // Fast path: the whole tape lives in chunk 0, so the sweep walks a
            // flat array and parent updates are plain pointer offsets; the
            // chunked geometry is only needed once a tape spans chunks.
            if (m_keepAdjoints) {
                sweep<true, true>(top);
            } else {
                sweep<false, true>(top);
            }
        } else if (m_keepAdjoints) {
            sweep<true, false>(top);
        } else {
            sweep<false, false>(top);
        }
    }

private:
    /// Parent accessor for the sweep: a flat pointer offset in the dense
    /// single-chunk layout, otherwise the hoisted chunk shift/mask lookup.
    template <bool Dense>
    RevNode& parentNode(std::uint32_t p, RevNode* denseNodes, std::size_t shift,
                        std::size_t mask) noexcept {
        if constexpr (Dense) {
            return denseNodes[p];
        } else {
            return m_chunks[p >> shift]->nodes[p & mask];
        }
    }

    /// One backward pass over nodes `1..top` in descending index order.
    /// `Dense` selects the single-chunk layout (every node in `m_chunks[0]`),
    /// where parent indexing is a flat pointer offset; otherwise the chunk
    /// shift/mask geometry is used. `KeepAdjoints` folds consume-on-read away
    /// at compile time, so neither mode pays for the other's zeroing.
    template <bool KeepAdjoints, bool Dense>
    void sweep(std::size_t top) {
        const std::size_t chunkShift = m_chunkShift;
        const std::size_t chunkMask = m_chunkMask;
        const std::size_t chunkNodes = m_chunkNodes;
        RevNode* const denseNodes = m_chunks[0]->nodes.get();
        const std::size_t lastChunk = Dense ? 0 : (top >> chunkShift);
        const std::size_t lastOffset = Dense ? top + 1 : (top & chunkMask) + 1;
        for (std::size_t c = lastChunk + 1; c-- > 0;) {
            RevNode* nodes = Dense ? denseNodes : m_chunks[c]->nodes.get();
            std::size_t j = (c == lastChunk) ? lastOffset : chunkNodes;
            const std::size_t stop = (c == 0) ? 1 : 0;
            while (j > stop) {
                RevNode& n = nodes[j - 1];
                --j;
                if (n.op == RevOp::Input) {
                    continue; // leaves keep their adjoints for readout
                }
                const double a = n.adjoint;
                if constexpr (!KeepAdjoints) {
                    n.adjoint = 0.0;
                }
                if (a == 0.0) {
                    continue;
                }
                switch (n.op) {
                    case RevOp::Input:
                        break;
                    case RevOp::Add:
                        parentNode<Dense>(n.lhs, denseNodes, chunkShift, chunkMask).adjoint += a;
                        parentNode<Dense>(n.rhs, denseNodes, chunkShift, chunkMask).adjoint += a;
                        break;
                    case RevOp::Sub:
                        parentNode<Dense>(n.lhs, denseNodes, chunkShift, chunkMask).adjoint += a;
                        parentNode<Dense>(n.rhs, denseNodes, chunkShift, chunkMask).adjoint -= a;
                        break;
                    case RevOp::Mul:
                        parentNode<Dense>(n.lhs, denseNodes, chunkShift, chunkMask).adjoint +=
                            a * n.rhsValue;
                        parentNode<Dense>(n.rhs, denseNodes, chunkShift, chunkMask).adjoint +=
                            a * n.lhsValue;
                        break;
                    case RevOp::Div: {
                        const double rv = n.rhsValue;
                        parentNode<Dense>(n.lhs, denseNodes, chunkShift, chunkMask).adjoint +=
                            a / rv;
                        parentNode<Dense>(n.rhs, denseNodes, chunkShift, chunkMask).adjoint -=
                            a * n.value / rv;
                        break;
                    }
                    case RevOp::Neg:
                        parentNode<Dense>(n.lhs, denseNodes, chunkShift, chunkMask).adjoint -= a;
                        break;
                    case RevOp::Sqrt:
                        parentNode<Dense>(n.lhs, denseNodes, chunkShift, chunkMask).adjoint +=
                            a * 0.5 / n.value;
                        break;
                    case RevOp::Exp:
                        parentNode<Dense>(n.lhs, denseNodes, chunkShift, chunkMask).adjoint +=
                            a * n.value;
                        break;
                    case RevOp::Log:
                        parentNode<Dense>(n.lhs, denseNodes, chunkShift, chunkMask).adjoint +=
                            a / n.lhsValue;
                        break;
                    case RevOp::Expm1:
                        parentNode<Dense>(n.lhs, denseNodes, chunkShift, chunkMask).adjoint +=
                            a * std::exp(n.lhsValue);
                        break;
                    case RevOp::Log1p:
                        parentNode<Dense>(n.lhs, denseNodes, chunkShift, chunkMask).adjoint +=
                            a / (1.0 + n.lhsValue);
                        break;
                    case RevOp::Abs:
                        parentNode<Dense>(n.lhs, denseNodes, chunkShift, chunkMask).adjoint +=
                            a * (n.lhsValue >= 0.0 ? 1.0 : -1.0);
                        break;
                    case RevOp::Pow:
                        parentNode<Dense>(n.lhs, denseNodes, chunkShift, chunkMask).adjoint +=
                            a * n.rhsValue * std::pow(n.lhsValue, n.rhsValue - 1.0);
                        parentNode<Dense>(n.rhs, denseNodes, chunkShift, chunkMask).adjoint +=
                            a * n.value * std::log(n.lhsValue);
                        break;
                    case RevOp::Sin:
                        parentNode<Dense>(n.lhs, denseNodes, chunkShift, chunkMask).adjoint +=
                            a * std::cos(n.lhsValue);
                        break;
                    case RevOp::Cos:
                        parentNode<Dense>(n.lhs, denseNodes, chunkShift, chunkMask).adjoint -=
                            a * std::sin(n.lhsValue);
                        break;
                    case RevOp::Tan:
                        parentNode<Dense>(n.lhs, denseNodes, chunkShift, chunkMask).adjoint +=
                            a * (1.0 + n.value * n.value);
                        break;
                    case RevOp::Asin:
                        parentNode<Dense>(n.lhs, denseNodes, chunkShift, chunkMask).adjoint +=
                            a / std::sqrt(1.0 - n.lhsValue * n.lhsValue);
                        break;
                    case RevOp::Acos:
                        parentNode<Dense>(n.lhs, denseNodes, chunkShift, chunkMask).adjoint -=
                            a / std::sqrt(1.0 - n.lhsValue * n.lhsValue);
                        break;
                    case RevOp::Atan:
                        parentNode<Dense>(n.lhs, denseNodes, chunkShift, chunkMask).adjoint +=
                            a / (1.0 + n.lhsValue * n.lhsValue);
                        break;
                    case RevOp::Sinh:
                        parentNode<Dense>(n.lhs, denseNodes, chunkShift, chunkMask).adjoint +=
                            a * std::cosh(n.lhsValue);
                        break;
                    case RevOp::Cosh:
                        parentNode<Dense>(n.lhs, denseNodes, chunkShift, chunkMask).adjoint +=
                            a * std::sinh(n.lhsValue);
                        break;
                    case RevOp::Tanh:
                        parentNode<Dense>(n.lhs, denseNodes, chunkShift, chunkMask).adjoint +=
                            a * (1.0 - n.value * n.value);
                        break;
                    case RevOp::Asinh:
                        parentNode<Dense>(n.lhs, denseNodes, chunkShift, chunkMask).adjoint +=
                            a / std::sqrt(1.0 + n.lhsValue * n.lhsValue);
                        break;
                    case RevOp::Acosh:
                        parentNode<Dense>(n.lhs, denseNodes, chunkShift, chunkMask).adjoint +=
                            a / std::sqrt(n.lhsValue * n.lhsValue - 1.0);
                        break;
                    case RevOp::Atanh:
                        parentNode<Dense>(n.lhs, denseNodes, chunkShift, chunkMask).adjoint +=
                            a / (1.0 - n.lhsValue * n.lhsValue);
                        break;
                    case RevOp::Erf:
                        parentNode<Dense>(n.lhs, denseNodes, chunkShift, chunkMask).adjoint +=
                            a * kTwoOverSqrtPi * std::exp(-n.lhsValue * n.lhsValue);
                        break;
                    case RevOp::Erfc:
                        parentNode<Dense>(n.lhs, denseNodes, chunkShift, chunkMask).adjoint -=
                            a * kTwoOverSqrtPi * std::exp(-n.lhsValue * n.lhsValue);
                        break;
                    case RevOp::LogGamma:
                    case RevOp::TGamma:
                    case RevOp::Digamma:
                    case RevOp::Trigamma:
                    case RevOp::Tetragamma:
                        // Partial precomputed at record time in rhsValue.
                        parentNode<Dense>(n.lhs, denseNodes, chunkShift, chunkMask).adjoint +=
                            a * n.rhsValue;
                        break;
                    case RevOp::Hypot: {
                        const double h = n.value;
                        if (h != 0.0) {
                            parentNode<Dense>(n.lhs, denseNodes, chunkShift, chunkMask).adjoint +=
                                a * n.lhsValue / h;
                            parentNode<Dense>(n.rhs, denseNodes, chunkShift, chunkMask).adjoint +=
                                a * n.rhsValue / h;
                        }
                        break;
                    }
                    case RevOp::Atan2: {
                        // atan2(y, x): lhs = y, rhs = x.
                        const double d = n.lhsValue * n.lhsValue + n.rhsValue * n.rhsValue;
                        if (d > 0.0) {
                            parentNode<Dense>(n.lhs, denseNodes, chunkShift, chunkMask).adjoint +=
                                a * n.rhsValue / d;
                            parentNode<Dense>(n.rhs, denseNodes, chunkShift, chunkMask).adjoint -=
                                a * n.lhsValue / d;
                        }
                        break;
                    }
                    case RevOp::External: {
                        const ExtRec& rec = m_ext[n.ext];
                        if (rec.vjp != nullptr) {
                            rec.vjp(rec.payload, a, m_extArgs.data() + n.lhs, n.rhs, *this);
                        }
                        break;
                    }
                    case RevOp::MinL:
                    case RevOp::MaxL:
                        parentNode<Dense>(n.lhs, denseNodes, chunkShift, chunkMask).adjoint += a;
                        break;
                    case RevOp::MinR:
                    case RevOp::MaxR:
                        parentNode<Dense>(n.rhs, denseNodes, chunkShift, chunkMask).adjoint += a;
                        break;
                    case RevOp::Cbrt: {
                        const double z = n.value;
                        parentNode<Dense>(n.lhs, denseNodes, chunkShift, chunkMask).adjoint +=
                            a / (3.0 * z * z);
                        break;
                    }
                    case RevOp::Exp2:
                        parentNode<Dense>(n.lhs, denseNodes, chunkShift, chunkMask).adjoint +=
                            a * kLn2 * n.value;
                        break;
                    case RevOp::Log2:
                        parentNode<Dense>(n.lhs, denseNodes, chunkShift, chunkMask).adjoint +=
                            a / (n.lhsValue * kLn2);
                        break;
                    case RevOp::Log10:
                        parentNode<Dense>(n.lhs, denseNodes, chunkShift, chunkMask).adjoint +=
                            a / (n.lhsValue * kLn10);
                        break;
                    case RevOp::CopySign: {
                        // z = copysign(a, b): a.e. dz/da = sign(b) sign(a),
                        // dz/db = 0 (sign(0) = +1, the house convention).
                        const double signB = n.rhsValue >= 0.0 ? 1.0 : -1.0;
                        const double signA = n.lhsValue >= 0.0 ? 1.0 : -1.0;
                        parentNode<Dense>(n.lhs, denseNodes, chunkShift, chunkMask).adjoint +=
                            a * signB * signA;
                        break;
                    }
                }
            }
        }
    }

public:
    /// Reverse sweep that keeps every node's adjoint (replay-based second
    /// order). Clears stale adjoints first and restores consume-on-read after.
    void reverseStore(std::uint32_t root) {
        resetAdjoints();
        m_keepAdjoints = true;
        reverse(root, 1.0);
        m_keepAdjoints = false;
    }

    /// Exact Hessian-vector product H(x) v by forward-over-reverse replay.
    /// `inputs[k]` are the tape slots of x, `direction[k]` the components of
    /// v; out[k] = (H v)_k. Returns false (no output written) when an external
    /// block lacks svp/vhp: exactness policy, never approximate.
    bool hessianVector(std::uint32_t root, const std::uint32_t* inputs, std::size_t nInputs,
                       const double* direction, double* out) {
        reverseStore(root);
        const std::size_t total = m_size;
        std::vector<double> s(total, 0.0);
        for (std::size_t k = 0; k < nInputs; ++k) {
            s[inputs[k]] = direction[k];
        }
        std::vector<double> buf;
        for (std::size_t i = 1; i < total; ++i) {
            const RevNode& n = node(i);
            if (n.op == RevOp::Input) {
                continue;
            }
            if (n.op == RevOp::External) {
                const ExtRec& rec = m_ext[n.ext];
                if (rec.svp == nullptr) {
                    return false;
                }
                const ExtArg* args = m_extArgs.data() + n.lhs;
                buf.resize(n.rhs);
                for (std::uint32_t j = 0; j < n.rhs; ++j) {
                    buf[j] = s[args[j].node];
                }
                s[i] = rec.svp(rec.payload, args, buf.data(), n.rhs);
                continue;
            }
            double fa = 0.0;
            double fb = 0.0;
            double dfa = 0.0;
            double dfb = 0.0;
            localDerivs(n, s.data(), fa, fb, dfa, dfb);
            s[i] = fa * s[n.lhs] + fb * s[n.rhs];
        }
        std::vector<double> b(total, 0.0);
        std::vector<double> before;
        std::vector<double> bArgs;
        for (std::size_t i = total; i-- > 1;) {
            const RevNode& n = node(i);
            if (n.op == RevOp::Input) {
                continue;
            }
            const double ai = n.adjoint;
            const double bi = b[i];
            if (n.op == RevOp::External) {
                const ExtRec& rec = m_ext[n.ext];
                if (rec.vjp == nullptr || rec.vhp == nullptr) {
                    return false;
                }
                const ExtArg* args = m_extArgs.data() + n.lhs;
                if (bi != 0.0) {
                    before.resize(n.rhs);
                    for (std::uint32_t j = 0; j < n.rhs; ++j) {
                        const std::uint32_t p = args[j].node;
                        before[j] = (p == kNone) ? 0.0 : node(p).adjoint;
                    }
                    rec.vjp(rec.payload, bi, args, n.rhs, *this);
                    for (std::uint32_t j = 0; j < n.rhs; ++j) {
                        const std::uint32_t p = args[j].node;
                        if (p != kNone) {
                            b[p] += node(p).adjoint - before[j];
                            node(p).adjoint = before[j];
                        }
                    }
                }
                if (ai != 0.0) {
                    bArgs.assign(n.rhs, 0.0);
                    buf.resize(n.rhs);
                    for (std::uint32_t j = 0; j < n.rhs; ++j) {
                        buf[j] = s[args[j].node];
                    }
                    rec.vhp(rec.payload, ai, args, buf.data(), n.rhs, bArgs.data());
                    for (std::uint32_t j = 0; j < n.rhs; ++j) {
                        if (args[j].node != kNone) {
                            b[args[j].node] += bArgs[j];
                        }
                    }
                }
                continue;
            }
            double fa = 0.0;
            double fb = 0.0;
            double dfa = 0.0;
            double dfb = 0.0;
            localDerivs(n, s.data(), fa, fb, dfa, dfb);
            b[n.lhs] += bi * fa + ai * dfa;
            b[n.rhs] += bi * fb + ai * dfb;
        }
        for (std::size_t k = 0; k < nInputs; ++k) {
            out[k] = b[inputs[k]];
        }
        return true;
    }

private:
    struct ExtRec {
        ExtVjpFn vjp = nullptr;
        ExtSvpFn svp = nullptr;
        ExtVhpFn vhp = nullptr;
        const void* payload = nullptr;
        std::unique_ptr<std::byte[]> payloadCopy;
        std::uint32_t owner = 0;
        std::uint32_t argBegin = 0;
        std::uint32_t nArgs = 0;
    };

    void truncateExternal(std::size_t pos) {
        while (!m_ext.empty() && m_ext.back().owner >= pos) {
            m_ext.pop_back();
        }
        if (m_ext.empty()) {
            m_extArgs.clear();
        } else {
            const ExtRec& last = m_ext.back();
            m_extArgs.resize(last.argBegin + last.nArgs);
        }
    }

    /// First partials of one node's local map plus their directional
    /// derivatives along the forward tangents `s` (for the second pass).
    void localDerivs(const RevNode& n, const double* s, double& fa, double& fb, double& dfa,
                     double& dfb) const {
        fa = 0.0;
        fb = 0.0;
        dfa = 0.0;
        dfb = 0.0;
        const double x = n.lhsValue;
        const double y = n.rhsValue;
        const double z = n.value;
        const double sl = s[n.lhs];
        const double sr = s[n.rhs];
        switch (n.op) {
            case RevOp::Input:
            case RevOp::External:
                break;
            case RevOp::Add:
                fa = 1.0;
                fb = 1.0;
                break;
            case RevOp::Sub:
                fa = 1.0;
                fb = -1.0;
                break;
            case RevOp::Mul:
                fa = y;
                fb = x;
                dfa = sr;
                dfb = sl;
                break;
            case RevOp::Div: {
                const double inv = 1.0 / y;
                fa = inv;
                fb = -x * inv * inv;
                dfa = -inv * inv * sr;
                dfb = -inv * inv * sl + 2.0 * x * inv * inv * inv * sr;
                break;
            }
            case RevOp::Neg:
                fa = -1.0;
                break;
            case RevOp::Sqrt: {
                const double inv = 1.0 / z;
                fa = 0.5 * inv;
                dfa = -0.25 * inv * inv * inv * sl;
                break;
            }
            case RevOp::Exp:
                fa = z;
                dfa = z * sl;
                break;
            case RevOp::Log:
                fa = 1.0 / x;
                dfa = -sl / (x * x);
                break;
            case RevOp::Expm1: {
                const double e = std::exp(x);
                fa = e;
                dfa = e * sl;
                break;
            }
            case RevOp::Log1p: {
                const double w = 1.0 + x;
                fa = 1.0 / w;
                dfa = -sl / (w * w);
                break;
            }
            case RevOp::Abs:
                fa = x >= 0.0 ? 1.0 : -1.0;
                break;
            case RevOp::Pow: {
                const double lx = std::log(x);
                const double bma = std::pow(x, y - 1.0);
                const double fab = bma * (1.0 + y * lx);
                fa = y * bma;
                fb = z * lx;
                dfa = y * (y - 1.0) * std::pow(x, y - 2.0) * sl + fab * sr;
                dfb = fab * sl + z * lx * lx * sr;
                break;
            }
            case RevOp::Sin:
                fa = std::cos(x);
                dfa = -std::sin(x) * sl;
                break;
            case RevOp::Cos:
                fa = -std::sin(x);
                dfa = -std::cos(x) * sl;
                break;
            case RevOp::Tan: {
                const double sec2 = 1.0 + z * z;
                fa = sec2;
                dfa = 2.0 * z * sec2 * sl;
                break;
            }
            case RevOp::Asin: {
                const double w = 1.0 - x * x;
                const double sw = std::sqrt(w);
                fa = 1.0 / sw;
                dfa = x * sl / (w * sw);
                break;
            }
            case RevOp::Acos: {
                const double w = 1.0 - x * x;
                const double sw = std::sqrt(w);
                fa = -1.0 / sw;
                dfa = -x * sl / (w * sw);
                break;
            }
            case RevOp::Atan: {
                const double w = 1.0 + x * x;
                fa = 1.0 / w;
                dfa = -2.0 * x * sl / (w * w);
                break;
            }
            case RevOp::Sinh:
                fa = std::cosh(x);
                dfa = std::sinh(x) * sl;
                break;
            case RevOp::Cosh:
                fa = std::sinh(x);
                dfa = std::cosh(x) * sl;
                break;
            case RevOp::Tanh: {
                const double w = 1.0 - z * z;
                fa = w;
                dfa = -2.0 * z * w * sl;
                break;
            }
            case RevOp::Asinh: {
                // z = asinh x: f' = 1/sqrt(1+x^2), f'' = -x/(1+x^2)^{3/2}.
                const double w = 1.0 + x * x;
                const double sw = std::sqrt(w);
                fa = 1.0 / sw;
                dfa = -x * sl / (w * sw);
                break;
            }
            case RevOp::Acosh: {
                // z = acosh x (x >= 1): f' = 1/sqrt(x^2-1),
                // f'' = -x/(x^2-1)^{3/2}.
                const double w = x * x - 1.0;
                const double sw = std::sqrt(w);
                fa = 1.0 / sw;
                dfa = -x * sl / (w * sw);
                break;
            }
            case RevOp::Atanh: {
                // z = atanh x (|x| < 1): f' = 1/(1-x^2),
                // f'' = 2x/(1-x^2)^2.
                const double w = 1.0 - x * x;
                fa = 1.0 / w;
                dfa = 2.0 * x * sl / (w * w);
                break;
            }
            case RevOp::Erf: {
                const double e = kTwoOverSqrtPi * std::exp(-x * x);
                fa = e;
                dfa = -2.0 * x * e * sl;
                break;
            }
            case RevOp::Erfc: {
                const double e = kTwoOverSqrtPi * std::exp(-x * x);
                fa = -e;
                dfa = 2.0 * x * e * sl;
                break;
            }
            case RevOp::LogGamma:
                fa = digamma(x);
                dfa = trigamma(x) * sl;
                break;
            case RevOp::TGamma: {
                const double psi = digamma(x);
                fa = z * psi;
                dfa = z * (psi * psi + trigamma(x)) * sl;
                break;
            }
            case RevOp::Digamma:
                // f = psi(x): f' = trigamma(x), f'' = tetragamma(x).
                fa = detail::trigammaDouble(x);
                dfa = detail::tetragammaDouble(x) * sl;
                break;
            case RevOp::Trigamma:
                // f = psi'(x): f' = psi''(x), f'' = psi'''(x).
                fa = detail::tetragammaDouble(x);
                dfa = detail::pentagammaDouble(x) * sl;
                break;
            case RevOp::Tetragamma:
                // f = psi''(x): f' = psi'''(x), f'' = psi''''(x).
                fa = detail::pentagammaDouble(x);
                dfa = detail::psi4Double(x) * sl;
                break;
            case RevOp::Hypot: {
                if (z == 0.0) {
                    break;
                }
                const double invh = 1.0 / z;
                const double invh3 = invh * invh * invh;
                fa = x * invh;
                fb = y * invh;
                dfa = y * y * invh3 * sl - x * y * invh3 * sr;
                dfb = -x * y * invh3 * sl + x * x * invh3 * sr;
                break;
            }
            case RevOp::Atan2: {
                const double d = x * x + y * y;
                if (d == 0.0) {
                    break;
                }
                const double inv = 1.0 / d;
                const double inv2 = inv * inv;
                fa = y * inv;
                fb = -x * inv;
                dfa = -2.0 * x * y * inv2 * sl + (x * x - y * y) * inv2 * sr;
                dfb = (x * x - y * y) * inv2 * sl + 2.0 * x * y * inv2 * sr;
                break;
            }
            case RevOp::MinL:
            case RevOp::MaxL:
                fa = 1.0;
                break;
            case RevOp::MinR:
            case RevOp::MaxR:
                fb = 1.0;
                break;
            case RevOp::Cbrt: {
                // z = x^(1/3): f' = 1/(3 z^2), f'' = -2/(9 z^5).
                const double inv = 1.0 / z;
                const double inv2 = inv * inv;
                fa = inv2 / 3.0;
                dfa = -2.0 * inv2 * inv2 * inv * sl / 9.0;
                break;
            }
            case RevOp::Exp2:
                // z = 2^x: f' = ln2 z, f'' = (ln2)^2 z.
                fa = kLn2 * z;
                dfa = kLn2 * kLn2 * z * sl;
                break;
            case RevOp::Log2:
                // z = log2 x: f' = 1/(x ln2), f'' = -1/(x^2 ln2).
                fa = 1.0 / (x * kLn2);
                dfa = -sl / (x * x * kLn2);
                break;
            case RevOp::Log10:
                // z = log10 x: f' = 1/(x ln10), f'' = -1/(x^2 ln10).
                fa = 1.0 / (x * kLn10);
                dfa = -sl / (x * x * kLn10);
                break;
            case RevOp::CopySign:
                // Locally linear in a, constant in b: all second derivatives 0.
                fa = (y >= 0.0 ? 1.0 : -1.0) * (x >= 0.0 ? 1.0 : -1.0);
                break;
        }
    }

    /// Matching aligned-array delete for the 64-byte aligned node buffer.
    struct NodeDeleter {
        void operator()(RevNode* p) const noexcept { ::operator delete[](p, std::align_val_t{64}); }
    };

    /// One runtime-sized arena block; the node buffer is 64-byte aligned so
    /// each chunk starts on a cache line.
    struct Chunk {
        std::unique_ptr<RevNode[], NodeDeleter> nodes;
    };

    static std::unique_ptr<Chunk> makeChunk(std::size_t n) {
        auto chunk = std::make_unique<Chunk>();
        chunk->nodes.reset(new (std::align_val_t{64}) RevNode[n]);
        return chunk;
    }

    RevNode& node(std::size_t i) noexcept {
        return m_chunks[i >> m_chunkShift]->nodes[i & m_chunkMask];
    }
    const RevNode& node(std::size_t i) const noexcept {
        return m_chunks[i >> m_chunkShift]->nodes[i & m_chunkMask];
    }

    /// Stage one node at the cursor and bump it. The caller guarantees
    /// `m_size < m_capacity` (the capacity check lives in `pushDirect`), so
    /// this is one aggregate store into an already-allocated chunk plus the
    /// cache insert when CSE is on. A single-chunk arena (the default until
    /// the tape outgrows its first chunk) indexes flat, so the hot push keeps
    /// the chunk table out of the address computation.
    std::uint32_t stageNode(RevOp op, std::uint32_t lhs, std::uint32_t rhs, double value,
                            double lhsValue, double rhsValue) {
        RevNode& cell = m_singleChunk ? m_chunks[0]->nodes[m_size] : node(m_size);
        cell.value = value;
        cell.adjoint = 0.0;
        cell.lhsValue = lhsValue;
        cell.rhsValue = rhsValue;
        cell.lhs = lhs;
        cell.rhs = rhs;
        cell.op = op; // `ext` is External-only and stays untouched otherwise
        const std::uint32_t idx = static_cast<std::uint32_t>(m_size++);
        if (m_useCse && op != RevOp::Input) [[unlikely]] {
            cseInsert(op, lhs, rhs, value, idx);
        }
        return idx;
    }

    /// One 4-entry direct-mapped statement-cache slot: `(op, lhs, rhs)` key,
    /// the node that recorded it and that node's primal bit pattern (the
    /// reuse guard). `node == kNone` marks the slot empty.
    struct CseEntry {
        RevOp op = RevOp::Input;
        std::uint32_t lhs = kNone;
        std::uint32_t rhs = kNone;
        std::uint32_t node = kNone;
        std::uint64_t valueBits = 0u;
    };

    static constexpr std::size_t kCseWays = 4;

    static std::size_t cseIndex(RevOp op, std::uint32_t lhs, std::uint32_t rhs) noexcept {
        std::uint64_t h = 1469598103934665603ull; // FNV-1a over the key fields
        h = (h ^ static_cast<std::uint64_t>(op)) * 1099511628211ull;
        h = (h ^ static_cast<std::uint64_t>(lhs)) * 1099511628211ull;
        h = (h ^ static_cast<std::uint64_t>(rhs)) * 1099511628211ull;
        return static_cast<std::size_t>(h) & (kCseWays - 1);
    }

    std::uint32_t cseLookup(RevOp op, std::uint32_t lhs, std::uint32_t rhs,
                            double value) const noexcept {
        const CseEntry& e = m_cse[cseIndex(op, lhs, rhs)];
        if (e.node == kNone || e.op != op || e.lhs != lhs || e.rhs != rhs ||
            e.valueBits != std::bit_cast<std::uint64_t>(value)) {
            return kNone;
        }
        return e.node;
    }

    void cseInsert(RevOp op, std::uint32_t lhs, std::uint32_t rhs, double value,
                   std::uint32_t node) noexcept {
        CseEntry& e = m_cse[cseIndex(op, lhs, rhs)];
        e.op = op;
        e.lhs = lhs;
        e.rhs = rhs;
        e.node = node;
        e.valueBits = std::bit_cast<std::uint64_t>(value);
    }

    void invalidateCse() noexcept {
        for (CseEntry& e : m_cse) {
            e.node = kNone;
        }
    }

    void ensureCapacity(std::size_t total) {
        if (total <= m_capacity) [[likely]] {
            return;
        }
        const std::size_t need = (total + m_chunkNodes - 1) >> m_chunkShift;
        m_chunks.reserve(need);
        while (m_chunks.size() < need) {
            m_chunks.push_back(makeChunk(m_chunkNodes));
        }
        m_capacity = m_chunks.size() << m_chunkShift;
        m_singleChunk = (m_chunks.size() == 1);
    }

    std::vector<std::unique_ptr<Chunk>> m_chunks;
    std::vector<ExtRec> m_ext;
    std::vector<ExtArg> m_extArgs;
    std::size_t m_size = 1; // sink lives at index 0
    std::size_t m_capacity = 0;
    std::size_t m_chunkShift = kChunkShift;
    std::size_t m_chunkNodes = kChunkNodes;
    std::size_t m_chunkMask = kChunkMask;
    bool m_singleChunk = true; // m_chunks.size() == 1: flat first-chunk indexing
    bool m_keepAdjoints = false;
    bool m_useCse = false;
    CseEntry m_cse[kCseWays]{}; // all slots empty (`node == kNone`)
};

/// RAII recording window over a `RevTape`: remembers the tape position at
/// construction and truncates back to it in the destructor, so every node
/// appended inside the scope is discarded (LIFO). `dismiss()` keeps the
/// recording. The tape must outlive the scope, and no other tape mutation may
/// interleave with the window.
class TapeScope {
public:
    explicit TapeScope(RevTape& tape) : m_tape(&tape), m_position(tape.position()) {}
    ~TapeScope() {
        if (m_tape != nullptr) {
            m_tape->resetTo(m_position);
        }
    }
    TapeScope(const TapeScope&) = delete;
    TapeScope& operator=(const TapeScope&) = delete;

    /// Keep everything recorded inside this scope (skip the truncation).
    void dismiss() noexcept { m_tape = nullptr; }

private:
    RevTape* m_tape;
    std::size_t m_position;
};

/// Run `f()` inside a TapeScope on the calling thread's tape: nodes recorded
/// by `f` are truncated when `run` returns (whether or not `f` reverses) and
/// the result is passed through; use `TapeScope::dismiss` directly when a
/// recording must survive.
template <typename F>
decltype(auto) RevTape::run(F&& f) {
    TapeScope scope(active());
    return std::forward<F>(f)();
}

inline void RevTape::releaseCurrentThread() {
    RevTape*& tape = currentPtr();
    delete tape;
    tape = nullptr;
}

/// Value + tape node index; `kNone` = plain constant (no node).
struct RevScalar {
    double value = 0.0;
    std::uint32_t node = RevTape::kNone;

    RevScalar() = default;
    RevScalar(double v) : value(v) {} // implicit: constants never hit the tape
    RevScalar(double v, std::uint32_t n) : value(v), node(n) {}
};

inline std::uint32_t RevTape::pushExternal(double value, ExtVjpFn vjp, const RevScalar* args,
                                           std::uint32_t nArgs, const void* payload,
                                           std::size_t payloadSize, ExtSvpFn svp, ExtVhpFn vhp) {
    ensureCapacity(m_size + 1);
    const std::uint32_t argBegin = static_cast<std::uint32_t>(m_extArgs.size());
    for (std::uint32_t i = 0; i < nArgs; ++i) {
        m_extArgs.push_back(ExtArg{args[i].value, args[i].node});
    }
    ExtRec rec;
    rec.vjp = vjp;
    rec.svp = svp;
    rec.vhp = vhp;
    rec.owner = static_cast<std::uint32_t>(m_size);
    rec.argBegin = argBegin;
    rec.nArgs = nArgs;
    if (payload != nullptr && payloadSize > 0) {
        rec.payloadCopy = std::make_unique<std::byte[]>(payloadSize);
        std::memcpy(rec.payloadCopy.get(), payload, payloadSize);
        rec.payload = rec.payloadCopy.get();
    }
    m_ext.push_back(std::move(rec));
    RevNode& cell = node(m_size);
    cell = RevNode{value, 0.0, 0.0, 0.0, argBegin, nArgs, RevOp::External};
    cell.ext = static_cast<std::uint32_t>(m_ext.size() - 1);
    return static_cast<std::uint32_t>(m_size++);
}

namespace detail {

/// Binary record. A constant operand (`kNone`) contributes no parent node:
/// the node stores the sink in that parent slot and the constant's primal in
/// the matching value slot. This is why `x * 2.0` is one node with
/// `rhs == kNone` / `rhsValue == 2.0`, and stays that way through
/// `pushDirect` — there is no extra constant-leaf node or extra store.
inline RevScalar revBinary(RevOp op, const RevScalar& a, const RevScalar& b, double value) {
    if (a.node == RevTape::kNone && b.node == RevTape::kNone) {
        return RevScalar(value);
    }
    return RevScalar(value,
                     RevTape::active().pushDirect(op, a.node, b.node, value, a.value, b.value));
}

inline RevScalar revUnary(RevOp op, const RevScalar& a, double value) {
    if (a.node == RevTape::kNone) {
        return RevScalar(value);
    }
    return RevScalar(value,
                     RevTape::active().pushDirect(op, a.node, RevTape::kNone, value, a.value));
}

/// Unary op whose VJP factor is precomputed at record time and stashed in the
/// otherwise-free `rhsValue` slot (digamma/gamma-based rules, reused partials),
/// so the sweep is a single multiply.
inline RevScalar revUnaryPrecomputed(RevOp op, const RevScalar& a, double value, double partial) {
    if (a.node == RevTape::kNone) {
        return RevScalar(value);
    }
    return RevScalar(
        value, RevTape::active().pushDirect(op, a.node, RevTape::kNone, value, a.value, partial));
}

/// Record an `External` block with `nArgs` parent scalars and an optional
/// trivially-copyable payload (copied into the tape). Collapses to a constant
/// when every parent is a plain double.
inline RevScalar makeExternal(double value, const RevScalar* args, std::uint32_t nArgs,
                              ExtVjpFn vjp, const void* payload = nullptr,
                              std::size_t payloadSize = 0, ExtSvpFn svp = nullptr,
                              ExtVhpFn vhp = nullptr) {
    bool active = false;
    for (std::uint32_t i = 0; i < nArgs; ++i) {
        if (args[i].node != RevTape::kNone) {
            active = true;
            break;
        }
    }
    if (!active) {
        return RevScalar(value);
    }
    return RevScalar(value, RevTape::active().pushExternal(value, vjp, args, nArgs, payload,
                                                           payloadSize, svp, vhp));
}

template <typename P>
RevScalar makeExternal(double value, const RevScalar* args, std::uint32_t nArgs, ExtVjpFn vjp,
                       const P& payload, ExtSvpFn svp = nullptr, ExtVhpFn vhp = nullptr) {
    static_assert(std::is_trivially_copyable_v<P>, "External payloads must be trivially copyable");
    return makeExternal(value, args, nArgs, vjp, &payload, sizeof(P), svp, vhp);
}

inline void fmaVjp(const void* /*payload*/, double seed, const ExtArg* a, std::uint32_t /*n*/,
                   RevTape& tape) noexcept {
    tape.addAdjoint(a[0].node, seed * a[1].value);
    tape.addAdjoint(a[1].node, seed * a[0].value);
    tape.addAdjoint(a[2].node, seed);
}

inline void logSumExpVjp(const void* /*payload*/, double seed, const ExtArg* a, std::uint32_t n,
                         RevTape& tape) noexcept {
    double m = a[0].value;
    for (std::uint32_t i = 1; i < n; ++i) {
        if (a[i].value > m) {
            m = a[i].value;
        }
    }
    double s = 0.0;
    for (std::uint32_t i = 0; i < n; ++i) {
        s += std::exp(a[i].value - m);
    }
    const double inv = seed / s;
    for (std::uint32_t i = 0; i < n; ++i) {
        tape.addAdjoint(a[i].node, inv * std::exp(a[i].value - m));
    }
}

inline double fmaSvp(const void* /*payload*/, const ExtArg* a, const double* s,
                     std::uint32_t /*n*/) noexcept {
    return a[1].value * s[0] + a[0].value * s[1] + s[2];
}

inline void fmaVhp(const void* /*payload*/, double adjoint, const ExtArg* /*a*/, const double* s,
                   std::uint32_t /*n*/, double* b) noexcept {
    b[0] += adjoint * s[1];
    b[1] += adjoint * s[0];
}

inline double logSumExpSvp(const void* /*payload*/, const ExtArg* a, const double* s,
                           std::uint32_t n) noexcept {
    double m = a[0].value;
    for (std::uint32_t i = 1; i < n; ++i) {
        if (a[i].value > m) {
            m = a[i].value;
        }
    }
    double z = 0.0;
    double sum = 0.0;
    for (std::uint32_t i = 0; i < n; ++i) {
        const double e = std::exp(a[i].value - m);
        z += e;
        sum += e * s[i];
    }
    return sum / z;
}

inline void logSumExpVhp(const void* /*payload*/, double adjoint, const ExtArg* a, const double* s,
                         std::uint32_t n, double* b) noexcept {
    double m = a[0].value;
    for (std::uint32_t i = 1; i < n; ++i) {
        if (a[i].value > m) {
            m = a[i].value;
        }
    }
    double z = 0.0;
    for (std::uint32_t i = 0; i < n; ++i) {
        z += std::exp(a[i].value - m);
    }
    double dot = 0.0;
    for (std::uint32_t i = 0; i < n; ++i) {
        dot += std::exp(a[i].value - m) * s[i];
    }
    dot /= z;
    for (std::uint32_t i = 0; i < n; ++i) {
        const double o = std::exp(a[i].value - m) / z;
        b[i] += adjoint * o * (s[i] - dot);
    }
}

} // namespace detail

/// Exact fused multiply-add: one External node over three parents (the node
/// has two parent slots, so the third lives in the argument cache).
inline RevScalar fma(const RevScalar& a, const RevScalar& b, const RevScalar& c) {
    const RevScalar args[3] = {a, b, c};
    return detail::makeExternal(std::fma(a.value, b.value, c.value), args, 3, detail::fmaVjp,
                                nullptr, 0, detail::fmaSvp, detail::fmaVhp);
}

/// Stable log-sum-exp over `n` scalars: one External node; the VJP scatters
/// the softmax weights. Every-constant arguments collapse to a constant.
inline RevScalar logSumExp(const RevScalar* xs, std::size_t n) {
    if (n == 0) {
        return RevScalar(0.0);
    }
    double m = xs[0].value;
    for (std::size_t i = 1; i < n; ++i) {
        if (xs[i].value > m) {
            m = xs[i].value;
        }
    }
    double s = 0.0;
    for (std::size_t i = 0; i < n; ++i) {
        s += std::exp(xs[i].value - m);
    }
    const double value = m + std::log(s);
    return detail::makeExternal(value, xs, static_cast<std::uint32_t>(n), detail::logSumExpVjp,
                                nullptr, 0, detail::logSumExpSvp, detail::logSumExpVhp);
}

// ── compound assignment (rebinds to a fresh node: sharing-safe) ──

inline RevScalar& operator+=(RevScalar& a, const RevScalar& b) {
    a = detail::revBinary(RevOp::Add, a, b, a.value + b.value);
    return a;
}
inline RevScalar& operator-=(RevScalar& a, const RevScalar& b) {
    a = detail::revBinary(RevOp::Sub, a, b, a.value - b.value);
    return a;
}
inline RevScalar& operator*=(RevScalar& a, const RevScalar& b) {
    a = detail::revBinary(RevOp::Mul, a, b, a.value * b.value);
    return a;
}
inline RevScalar& operator/=(RevScalar& a, const RevScalar& b) {
    a = detail::revBinary(RevOp::Div, a, b, a.value / b.value);
    return a;
}

// ── binary arithmetic ──

inline RevScalar operator+(const RevScalar& a, const RevScalar& b) {
    return detail::revBinary(RevOp::Add, a, b, a.value + b.value);
}
inline RevScalar operator-(const RevScalar& a, const RevScalar& b) {
    return detail::revBinary(RevOp::Sub, a, b, a.value - b.value);
}
inline RevScalar operator*(const RevScalar& a, const RevScalar& b) {
    return detail::revBinary(RevOp::Mul, a, b, a.value * b.value);
}
inline RevScalar operator/(const RevScalar& a, const RevScalar& b) {
    return detail::revBinary(RevOp::Div, a, b, a.value / b.value);
}
inline RevScalar operator-(const RevScalar& a) {
    return detail::revUnary(RevOp::Neg, a, -a.value);
}

inline RevScalar operator+(const RevScalar& a, double b) {
    return a + RevScalar(b);
}
inline RevScalar operator+(double a, const RevScalar& b) {
    return RevScalar(a) + b;
}
inline RevScalar operator-(const RevScalar& a, double b) {
    return a - RevScalar(b);
}
inline RevScalar operator-(double a, const RevScalar& b) {
    return RevScalar(a) - b;
}
inline RevScalar operator*(const RevScalar& a, double b) {
    return a * RevScalar(b);
}
inline RevScalar operator*(double a, const RevScalar& b) {
    return RevScalar(a) * b;
}
inline RevScalar operator/(const RevScalar& a, double b) {
    return a / RevScalar(b);
}
inline RevScalar operator/(double a, const RevScalar& b) {
    return RevScalar(a) / b;
}

// ── comparisons on the primal value ──

inline bool operator<(const RevScalar& a, const RevScalar& b) {
    return a.value < b.value;
}
inline bool operator>(const RevScalar& a, const RevScalar& b) {
    return a.value > b.value;
}
inline bool operator<=(const RevScalar& a, const RevScalar& b) {
    return a.value <= b.value;
}
inline bool operator>=(const RevScalar& a, const RevScalar& b) {
    return a.value >= b.value;
}
inline bool operator==(const RevScalar& a, const RevScalar& b) {
    return a.value == b.value;
}
inline bool operator!=(const RevScalar& a, const RevScalar& b) {
    return a.value != b.value;
}

// ── elementary functions ──

inline RevScalar sqrt(const RevScalar& x) {
    using std::sqrt;
    return detail::revUnary(RevOp::Sqrt, x, sqrt(x.value));
}
inline RevScalar exp(const RevScalar& x) {
    using std::exp;
    return detail::revUnary(RevOp::Exp, x, exp(x.value));
}
inline RevScalar log(const RevScalar& x) {
    using std::log;
    return detail::revUnary(RevOp::Log, x, log(x.value));
}
inline RevScalar expm1(const RevScalar& x) {
    using std::expm1;
    return detail::revUnary(RevOp::Expm1, x, expm1(x.value));
}
inline RevScalar log1p(const RevScalar& x) {
    using std::log1p;
    return detail::revUnary(RevOp::Log1p, x, log1p(x.value));
}
inline RevScalar cbrt(const RevScalar& x) {
    using std::cbrt;
    return detail::revUnary(RevOp::Cbrt, x, cbrt(x.value));
}
inline RevScalar exp2(const RevScalar& x) {
    using std::exp2;
    return detail::revUnary(RevOp::Exp2, x, exp2(x.value));
}
inline RevScalar log2(const RevScalar& x) {
    using std::log2;
    return detail::revUnary(RevOp::Log2, x, log2(x.value));
}
inline RevScalar log10(const RevScalar& x) {
    using std::log10;
    return detail::revUnary(RevOp::Log10, x, log10(x.value));
}
inline RevScalar pow(const RevScalar& a, const RevScalar& b) {
    return detail::revBinary(RevOp::Pow, a, b, std::pow(a.value, b.value));
}
inline RevScalar pow(const RevScalar& a, double b) {
    return pow(a, RevScalar(b));
}
inline RevScalar pow(double a, const RevScalar& b) {
    return pow(RevScalar(a), b);
}
inline RevScalar sin(const RevScalar& x) {
    return detail::revUnary(RevOp::Sin, x, std::sin(x.value));
}
inline RevScalar cos(const RevScalar& x) {
    return detail::revUnary(RevOp::Cos, x, std::cos(x.value));
}
/// sin and cos of `x` in one call. Two nodes by design (one Sin + one Cos
/// sharing the parent); there is no dual-output node yet, so the forward pass
/// evaluates both libm functions. A fused primal would need POSIX `__sincos`
/// (not `std`-only / portable), and a true dual-output node would need a
/// node-model change (two results per node), so `sincos` stays a two-node
/// composite here. Constant arguments stay node-less.
inline void sincos(const RevScalar& x, RevScalar& s, RevScalar& c) {
    s = sin(x);
    c = cos(x);
}
/// Plain-double `sincos` for API symmetry with the AD overloads.
inline void sincos(double x, double& s, double& c) {
    s = std::sin(x);
    c = std::cos(x);
}
inline RevScalar tan(const RevScalar& x) {
    return detail::revUnary(RevOp::Tan, x, std::tan(x.value));
}
inline RevScalar asin(const RevScalar& x) {
    return detail::revUnary(RevOp::Asin, x, std::asin(x.value));
}
inline RevScalar acos(const RevScalar& x) {
    return detail::revUnary(RevOp::Acos, x, std::acos(x.value));
}
inline RevScalar atan(const RevScalar& x) {
    return detail::revUnary(RevOp::Atan, x, std::atan(x.value));
}
inline RevScalar sinh(const RevScalar& x) {
    return detail::revUnary(RevOp::Sinh, x, std::sinh(x.value));
}
inline RevScalar cosh(const RevScalar& x) {
    return detail::revUnary(RevOp::Cosh, x, std::cosh(x.value));
}
inline RevScalar tanh(const RevScalar& x) {
    return detail::revUnary(RevOp::Tanh, x, std::tanh(x.value));
}
/// Inverse hyperbolic functions (std primals): `asinh` all reals, `acosh`
/// x >= 1, `atanh` |x| < 1 — outside the domain the primal is NaN and the
/// partials follow libm's `sqrt`/division rules.
inline RevScalar asinh(const RevScalar& x) {
    using std::asinh;
    return detail::revUnary(RevOp::Asinh, x, asinh(x.value));
}
inline RevScalar acosh(const RevScalar& x) {
    using std::acosh;
    return detail::revUnary(RevOp::Acosh, x, acosh(x.value));
}
inline RevScalar atanh(const RevScalar& x) {
    using std::atanh;
    return detail::revUnary(RevOp::Atanh, x, atanh(x.value));
}
inline RevScalar erf(const RevScalar& x) {
    return detail::revUnary(RevOp::Erf, x, std::erf(x.value));
}
inline RevScalar erfc(const RevScalar& x) {
    return detail::revUnary(RevOp::Erfc, x, std::erfc(x.value));
}
/// Digamma psi(x) for x > 0 (NaN otherwise); the VJP is
/// `a * trigamma(x)` (partial precomputed at record time).
inline RevScalar digamma(const RevScalar& x) {
    return detail::revUnaryPrecomputed(RevOp::Digamma, x, detail::digammaDouble(x.value),
                                       detail::trigammaDouble(x.value));
}
/// Trigamma psi'(x) for x > 0 (NaN otherwise); the VJP is
/// `a * tetragamma(x)`, so the reverse scalar is exact to second order.
inline RevScalar trigamma(const RevScalar& x) {
    return detail::revUnaryPrecomputed(RevOp::Trigamma, x, detail::trigammaDouble(x.value),
                                       detail::tetragammaDouble(x.value));
}
/// Tetragamma psi''(x) for x > 0 (NaN otherwise); the VJP is
/// `a * pentagamma(x)`. Needed by `trigamma(Tangent)` so fwd-over-rev
/// gamma chains carry the third derivative exactly.
inline RevScalar tetragamma(const RevScalar& x) {
    return detail::revUnaryPrecomputed(RevOp::Tetragamma, x, detail::tetragammaDouble(x.value),
                                       detail::pentagammaDouble(x.value));
}
inline RevScalar lgamma(const RevScalar& x) {
    return detail::revUnaryPrecomputed(RevOp::LogGamma, x, std::lgamma(x.value),
                                       detail::digammaDouble(x.value));
}
inline RevScalar tgamma(const RevScalar& x) {
    const double value = std::tgamma(x.value);
    return detail::revUnaryPrecomputed(RevOp::TGamma, x, value,
                                       value * detail::digammaDouble(x.value));
}
inline RevScalar hypot(const RevScalar& a, const RevScalar& b) {
    return detail::revBinary(RevOp::Hypot, a, b, std::hypot(a.value, b.value));
}
inline RevScalar atan2(const RevScalar& y, const RevScalar& x) {
    return detail::revBinary(RevOp::Atan2, y, x, std::atan2(y.value, x.value));
}
/// `copysign(a, b)`: magnitude of `a` with the sign of `b`. A.e. the `a`
/// adjoint is scaled by `sign(b) sign(a)` and `b` receives nothing; sign(0)
/// is +1 (the same convention as `Tangent::copysign`).
inline RevScalar copysign(const RevScalar& a, const RevScalar& b) {
    using std::copysign;
    return detail::revBinary(RevOp::CopySign, a, b, copysign(a.value, b.value));
}
inline RevScalar copysign(const RevScalar& a, double b) {
    return copysign(a, RevScalar(b));
}
inline RevScalar copysign(double a, const RevScalar& b) {
    return copysign(RevScalar(a), b);
}
inline RevScalar fabs(const RevScalar& x) {
    using std::fabs;
    return detail::revUnary(RevOp::Abs, x, fabs(x.value));
}
inline RevScalar abs(const RevScalar& x) {
    return fabs(x);
}
inline bool isnan(const RevScalar& x) {
    using std::isnan;
    return isnan(x.value);
}
inline bool isinf(const RevScalar& x) {
    using std::isinf;
    return isinf(x.value);
}
inline bool isfinite(const RevScalar& x) {
    using std::isfinite;
    return isfinite(x.value);
}
/// Predicate only (no gradient): the sign bit of the primal, so `-0.0` is
/// negative and a NaN payload's sign is observed.
inline bool signbit(const RevScalar& x) {
    using std::signbit;
    return signbit(x.value);
}

/// Branch selection shares the winner's node (derivative along the branch).
inline RevScalar fmin(const RevScalar& a, const RevScalar& b) {
    return a.value <= b.value ? a : b;
}
inline RevScalar fmax(const RevScalar& a, const RevScalar& b) {
    return a.value >= b.value ? a : b;
}
inline RevScalar fmin(const RevScalar& a, double b) {
    return fmin(a, RevScalar(b));
}
inline RevScalar fmax(const RevScalar& a, double b) {
    return fmax(a, RevScalar(b));
}

/// Stable softmax over `n` scalars: one shared `logSumExp` node plus one
/// exp/sub chain per component. All outputs are ordinary tape nodes, so a
/// downstream scalar objective reverses in one sweep; the full Jacobian
/// transpose is available via the multi-root `RevTape::reverse` overload.
inline void softmax(const RevScalar* xs, std::size_t n, RevScalar* out) {
    if (n == 0) {
        return;
    }
    const RevScalar lse = logSumExp(xs, n);
    for (std::size_t i = 0; i < n; ++i) {
        out[i] = exp(xs[i] - lse);
    }
}

} // namespace lightape

#endif // LIGHTAPE_REV_LEAN_REVERSE_H
