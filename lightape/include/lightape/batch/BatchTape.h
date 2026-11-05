#ifndef LIGHTAPE_BATCH_BATCHTAPE_H
#define LIGHTAPE_BATCH_BATCHTAPE_H

#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <new>
#include <vector>

#include "lightape/batch/VectorMath.h"
#include "lightape/rev/LeanReverse.h"

namespace lightape {
/**
 * @file BatchTape.h
 * @brief SoA batched reverse tape: W scalar programs through one shared op
 * structure (decisions #9/#17; codegen_simd.md §3(e)).
 *
 * `BatchTape<W>` records one straight-line structure of `RevOp` statements
 * for W independent evaluations ("lanes") and stores every node's lanes
 * contiguously (value/adjoint/parent-value planes of `std::array<double,W>`),
 * so the sweep's inner width-W loops autovectorize (NEON 2, AVX2 4, AVX-512 8
 * fp64 lanes) with no intrinsics and no vector-math dependency. Lane `l`
 * executes exactly the scalar `RevScalar` arithmetic for lane `l`'s inputs in
 * the scalar tape's node order; sibling lanes only share the instruction
 * stream, never a reduction, so nothing is ever reassociated.
 *
 * Supported opcodes, with the same `RevOp` values and the same local rules as
 * `RevTape`: Input, Add, Sub, Mul, Div, Neg, Sqrt, Exp, Log, Expm1, Log1p,
 * Abs, Pow, Sin, Cos, Tan, Asin, Acos, Atan, Sinh, Cosh, Tanh, Asinh, Acosh,
 * Atanh, Erf, Erfc, Cbrt, Exp2, Log2, Log10, Hypot, Atan2. Every
 * elementary-function lane value
 * is computed by the `VectorMath<W>` kernels (the math-backend seam,
 * `batch/VectorMath.h`) instead of calling libm directly here, and the sweep's
 * per-lane partials that need an elementary function (`exp`, `sin`, `cos`,
 * `sqrt`, hyperbolic and `pow`) route through the same kernels, so swapping
 * the backend changes every lane evaluation in one place. The `LibmScalar`
 * backend is the bitwise reference; see `VectorMath.h`.
 * `fmin`/`fmax` keep `RevTape`'s branch-sharing: when every lane picks one
 * operand, that operand is returned untouched and no node is recorded. A
 * mixed decision records one `MinL`/`MaxL` node whose `lhsValue` plane stores
 * the per-lane winner mask (`1.0` = lhs) and whose sweep routes each lane's
 * adjoint to its winner. This mask-carrying use of `MinL`/`MaxL` is the only
 * opcode meaning that differs from the scalar tape.
 *
 * Bitwise contract (gated by `tests/test_batch_tape.cpp`): under one build
 * with the same flags, each lane's primal and each input's adjoint are
 * bitwise equal to a scalar `RevScalar` tape running lane `l`'s values
 * through the same program. With lane-uniform branches this is structural
 * (the batch records the same statements and sweeps them in the same order).
 * A mixed `fmin`/`fmax` adds one select node: per-lane values stay bitwise
 * exact and adjoints stay bitwise exact when the selected operand has a
 * single consumer, but with several consumers the selected parent's adjoint
 * additions are grouped through the select node and may differ in the last
 * ulp from the branch-sharing scalar tape.
 *
 * One deliberate sweep simplification: the scalar tape skips a node whose own
 * adjoint is exactly zero, while the batch skips a node only when *all* lanes
 * are zero (per-lane skipping would break the vector lanes; decision #9 keeps
 * the kernel branch-free). A zero lane inside an otherwise live node still
 * evaluates its `0 * v` / `0 / v` terms; for finite operands those are true
 * zeros (the tested regime), but a zero lane carrying an inf/NaN operand
 * would pick up a NaN where the scalar sweep skipped the node outright.
 */

/// One lane-valued scalar: W independent values plus the slot of the shared
/// statement that produced them (`kNone` = node-less constant, the `RevScalar`
/// rule, so model-side W-lane constants never become tape leaves). Recording
/// operators always target the calling thread's `BatchTape<W>::active()`.
template <std::size_t W>
struct BatchScalar {
    static_assert(W >= 1 && W <= 16, "BatchScalar lane width must be in [1, 16]");

    static constexpr std::uint32_t kNone = 0u;

    std::uint32_t node = kNone;
    std::array<double, W> value{};

    BatchScalar() = default;
    /// Implicit broadcast constant (one double for all W lanes).
    BatchScalar(double v) { value.fill(v); }
    explicit BatchScalar(const std::array<double, W>& v) : value(v) {}
    BatchScalar(const std::array<double, W>& v, std::uint32_t n) : node(n), value(v) {}
};

namespace detail {

/// 64-byte-aligned array allocation with the matching aligned delete, so
/// every lane plane starts on a cache line (mirrors `RevTape`'s chunks).
template <typename T>
struct BatchAlignedDelete {
    void operator()(T* p) const noexcept { ::operator delete[](p, std::align_val_t{64}); }
};

template <typename T>
using BatchAlignedArray = std::unique_ptr<T[], BatchAlignedDelete<T>>;

template <typename T>
BatchAlignedArray<T> makeBatchAlignedArray(std::size_t n) {
    return BatchAlignedArray<T>(new (std::align_val_t{64}) T[n]);
}

} // namespace detail

/// Chunked SoA reverse tape for W lanes; see the file comment for the
/// structure, the supported opcodes and the bitwise contract. The recording
/// target is a thread-local `active()` tape per W, exactly like `RevTape`.
template <std::size_t W>
class BatchTape {
    static_assert(W >= 1 && W <= 16, "BatchTape lane width must be in [1, 16]");

public:
    /// Node 0 is a permanent dead sink: node-less operands point at it (their
    /// primal lives in the parent-value plane), so the sweep needs no
    /// null-parent branches and its adjoint is ignored.
    static constexpr std::uint32_t kNone = BatchScalar<W>::kNone;

    static constexpr std::size_t kChunkShift = 12;
    static constexpr std::size_t kChunkNodes = std::size_t{1} << kChunkShift;
    static constexpr std::size_t kChunkMask = kChunkNodes - 1;
    static_assert(kChunkNodes % W == 0, "chunk size must be a multiple of the lane width");

    /// Shared accessor for the thread-local tape pointer (no dynamic TLS init
    /// in the per-operation path; created once per thread on first `active()`).
    static BatchTape*& currentPtr() {
        static thread_local BatchTape* tape = nullptr;
        return tape;
    }

    static BatchTape& active() {
        BatchTape*& tape = currentPtr();
        if (tape == nullptr) [[unlikely]] {
            tape = new BatchTape();
        }
        return *tape;
    }

    /// Destroy this thread's tape and forget it (a null tape is a no-op).
    /// Invalidates every `BatchScalar` node index handed out before the call.
    static void releaseCurrentThread() {
        BatchTape*& tape = currentPtr();
        delete tape;
        tape = nullptr;
    }

    BatchTape() {
        m_chunks.push_back(makeChunk(kChunkNodes));
        m_capacity = kChunkNodes;
    }

    BatchTape(const BatchTape&) = delete;
    BatchTape& operator=(const BatchTape&) = delete;

    /// Independent variable: one `Input` node carrying the W lane values.
    BatchScalar<W> input(const std::array<double, W>& v) {
        return BatchScalar<W>(v, push(RevOp::Input, kNone, kNone, v));
    }

    /// Low-level append: `op` over parent slots `lhs`/`rhs` (`kNone` =
    /// node-less constant) with per-lane `value`; `lhsValue`/`rhsValue` cache
    /// the operand primals (constant operands live only here, exactly as in
    /// `RevTape`). The capacity check is inline; a full chunk falls back to
    /// `ensureCapacity`.
    std::uint32_t push(RevOp op, std::uint32_t lhs, std::uint32_t rhs,
                       const std::array<double, W>& value,
                       const std::array<double, W>& lhsValue = {},
                       const std::array<double, W>& rhsValue = {}) {
        if (m_size < m_capacity) [[likely]] {
            return stageNode(op, lhs, rhs, value, lhsValue, rhsValue);
        }
        ensureCapacity(m_size + 1);
        return stageNode(op, lhs, rhs, value, lhsValue, rhsValue);
    }

    /// Reverse sweep seeded at `root` with one seed per lane. Consume-on-use
    /// like `RevTape`: non-leaf adjoints are zeroed as they are read (a
    /// finished sweep leaves no stale state), Input adjoints stay for the
    /// caller to read out. A node is swept only when at least one lane's
    /// adjoint is nonzero (whole-vector skip; see the file comment).
    void reverse(std::uint32_t root, const std::array<double, W>& seed) {
        adjointRef(root) = seed;
        const std::size_t top = m_size - 1;
        if (top == 0) {
            return;
        }
        const std::size_t lastChunk = top >> kChunkShift;
        const std::size_t lastOffset = (top & kChunkMask) + 1;
        for (std::size_t c = lastChunk + 1; c-- > 0;) {
            Chunk& chunk = *m_chunks[c];
            std::size_t j = (c == lastChunk) ? lastOffset : kChunkNodes;
            const std::size_t stop = (c == 0) ? 1 : 0;
            while (j > stop) {
                sweepNode(chunk, --j);
            }
        }
    }

    /// Same sweep with one seed broadcast to every lane.
    void reverse(std::uint32_t root, double seed = 1.0) {
        std::array<double, W> s{};
        s.fill(seed);
        reverse(root, s);
    }

    const std::array<double, W>& adjoint(std::uint32_t idx) const noexcept {
        return m_chunks[idx >> kChunkShift]->adjoints[idx & kChunkMask];
    }
    const std::array<double, W>& value(std::uint32_t idx) const noexcept {
        return m_chunks[idx >> kChunkShift]->values[idx & kChunkMask];
    }

    /// Reset for a new recording: keep the sink and every allocated chunk
    /// (capacity is reused across paths), drop nothing else.
    void clear() noexcept { m_size = 1; }

    /// Absolute node count including the sink (the next free index).
    std::size_t position() const noexcept { return m_size; }
    std::size_t size() const noexcept { return m_size - 1; } // excluding sink

    void reserve(std::size_t n) { ensureCapacity(n + 1); }

private:
    /// One chunk of SoA planes: structure planes (`op`, `lhs`, `rhs`) plus the
    /// three lane planes and the adjoint plane, each 64-byte aligned.
    struct Chunk {
        detail::BatchAlignedArray<RevOp> op;
        detail::BatchAlignedArray<std::uint32_t> lhs;
        detail::BatchAlignedArray<std::uint32_t> rhs;
        detail::BatchAlignedArray<std::array<double, W>> values;
        detail::BatchAlignedArray<std::array<double, W>> adjoints;
        detail::BatchAlignedArray<std::array<double, W>> lhsValues;
        detail::BatchAlignedArray<std::array<double, W>> rhsValues;
    };

    static std::unique_ptr<Chunk> makeChunk(std::size_t n) {
        auto chunk = std::make_unique<Chunk>();
        chunk->op = detail::makeBatchAlignedArray<RevOp>(n);
        chunk->lhs = detail::makeBatchAlignedArray<std::uint32_t>(n);
        chunk->rhs = detail::makeBatchAlignedArray<std::uint32_t>(n);
        chunk->values = detail::makeBatchAlignedArray<std::array<double, W>>(n);
        chunk->adjoints = detail::makeBatchAlignedArray<std::array<double, W>>(n);
        chunk->lhsValues = detail::makeBatchAlignedArray<std::array<double, W>>(n);
        chunk->rhsValues = detail::makeBatchAlignedArray<std::array<double, W>>(n);
        return chunk;
    }

    /// Sweep one node: consume its adjoints and scatter the per-lane local
    /// derivative into the parents. Every lane loop mirrors one scalar
    /// `RevTape::reverse` rule statement-for-statement (with no cross-lane
    /// reduction), which is where the per-lane bitwise equality comes from.
    void sweepNode(Chunk& chunk, std::size_t off) {
        const RevOp op = chunk.op[off];
        if (op == RevOp::Input) {
            return; // leaves keep their adjoints for readout
        }
        std::array<double, W> a = chunk.adjoints[off];
        bool live = false;
        for (std::size_t l = 0; l < W; ++l) {
            live = live || (a[l] != 0.0);
        }
        if (!live) {
            return;
        }
        chunk.adjoints[off] = {};
        const std::uint32_t lhs = chunk.lhs[off];
        const std::uint32_t rhs = chunk.rhs[off];
        const std::array<double, W>& x = chunk.values[off];
        const std::array<double, W>& xv = chunk.lhsValues[off];
        const std::array<double, W>& yv = chunk.rhsValues[off];
        std::array<double, W>& adjl = adjointRef(lhs);
        std::array<double, W>& adjr = adjointRef(rhs);
        switch (op) {
            case RevOp::Input:
                break;
            case RevOp::Add:
                for (std::size_t l = 0; l < W; ++l) {
                    adjl[l] += a[l];
                }
                for (std::size_t l = 0; l < W; ++l) {
                    adjr[l] += a[l];
                }
                break;
            case RevOp::Sub:
                for (std::size_t l = 0; l < W; ++l) {
                    adjl[l] += a[l];
                }
                for (std::size_t l = 0; l < W; ++l) {
                    adjr[l] -= a[l];
                }
                break;
            case RevOp::Mul:
                for (std::size_t l = 0; l < W; ++l) {
                    adjl[l] += a[l] * yv[l];
                }
                for (std::size_t l = 0; l < W; ++l) {
                    adjr[l] += a[l] * xv[l];
                }
                break;
            case RevOp::Div:
                for (std::size_t l = 0; l < W; ++l) {
                    adjl[l] += a[l] / yv[l];
                }
                for (std::size_t l = 0; l < W; ++l) {
                    adjr[l] -= a[l] * x[l] / yv[l];
                }
                break;
            case RevOp::Neg:
                for (std::size_t l = 0; l < W; ++l) {
                    adjl[l] -= a[l];
                }
                break;
            case RevOp::Sqrt:
                for (std::size_t l = 0; l < W; ++l) {
                    adjl[l] += a[l] * 0.5 / x[l];
                }
                break;
            case RevOp::Exp:
                for (std::size_t l = 0; l < W; ++l) {
                    adjl[l] += a[l] * x[l];
                }
                break;
            case RevOp::Log:
                for (std::size_t l = 0; l < W; ++l) {
                    adjl[l] += a[l] / xv[l];
                }
                break;
            case RevOp::Abs:
                for (std::size_t l = 0; l < W; ++l) {
                    adjl[l] += a[l] * (xv[l] >= 0.0 ? 1.0 : -1.0);
                }
                break;
            case RevOp::Pow: {
                std::array<double, W> ym1{};
                std::array<double, W> p{};
                std::array<double, W> lx{};
                for (std::size_t l = 0; l < W; ++l) {
                    ym1[l] = yv[l] - 1.0;
                }
                batch::VectorMath<W>::vPow(xv.data(), ym1.data(), p.data());
                batch::VectorMath<W>::vLog(xv.data(), lx.data());
                for (std::size_t l = 0; l < W; ++l) {
                    adjl[l] += a[l] * yv[l] * p[l];
                }
                for (std::size_t l = 0; l < W; ++l) {
                    adjr[l] += a[l] * x[l] * lx[l];
                }
                break;
            }
            case RevOp::Expm1: {
                std::array<double, W> e{};
                batch::VectorMath<W>::vExp(xv.data(), e.data());
                for (std::size_t l = 0; l < W; ++l) {
                    adjl[l] += a[l] * e[l];
                }
                break;
            }
            case RevOp::Log1p:
                for (std::size_t l = 0; l < W; ++l) {
                    adjl[l] += a[l] / (1.0 + xv[l]);
                }
                break;
            case RevOp::Sin: {
                std::array<double, W> c{};
                batch::VectorMath<W>::vCos(xv.data(), c.data());
                for (std::size_t l = 0; l < W; ++l) {
                    adjl[l] += a[l] * c[l];
                }
                break;
            }
            case RevOp::Cos: {
                std::array<double, W> s{};
                batch::VectorMath<W>::vSin(xv.data(), s.data());
                for (std::size_t l = 0; l < W; ++l) {
                    adjl[l] -= a[l] * s[l];
                }
                break;
            }
            case RevOp::Tan:
                for (std::size_t l = 0; l < W; ++l) {
                    adjl[l] += a[l] * (1.0 + x[l] * x[l]);
                }
                break;
            case RevOp::Asin: {
                std::array<double, W> w{};
                std::array<double, W> r{};
                for (std::size_t l = 0; l < W; ++l) {
                    w[l] = 1.0 - xv[l] * xv[l];
                }
                batch::VectorMath<W>::vSqrt(w.data(), r.data());
                for (std::size_t l = 0; l < W; ++l) {
                    adjl[l] += a[l] / r[l];
                }
                break;
            }
            case RevOp::Acos: {
                std::array<double, W> w{};
                std::array<double, W> r{};
                for (std::size_t l = 0; l < W; ++l) {
                    w[l] = 1.0 - xv[l] * xv[l];
                }
                batch::VectorMath<W>::vSqrt(w.data(), r.data());
                for (std::size_t l = 0; l < W; ++l) {
                    adjl[l] -= a[l] / r[l];
                }
                break;
            }
            case RevOp::Atan:
                for (std::size_t l = 0; l < W; ++l) {
                    adjl[l] += a[l] / (1.0 + xv[l] * xv[l]);
                }
                break;
            case RevOp::Sinh: {
                std::array<double, W> c{};
                batch::VectorMath<W>::vCosh(xv.data(), c.data());
                for (std::size_t l = 0; l < W; ++l) {
                    adjl[l] += a[l] * c[l];
                }
                break;
            }
            case RevOp::Cosh: {
                std::array<double, W> s{};
                batch::VectorMath<W>::vSinh(xv.data(), s.data());
                for (std::size_t l = 0; l < W; ++l) {
                    adjl[l] += a[l] * s[l];
                }
                break;
            }
            case RevOp::Tanh:
                for (std::size_t l = 0; l < W; ++l) {
                    adjl[l] += a[l] * (1.0 - x[l] * x[l]);
                }
                break;
            case RevOp::Asinh: {
                std::array<double, W> w{};
                std::array<double, W> r{};
                for (std::size_t l = 0; l < W; ++l) {
                    w[l] = 1.0 + xv[l] * xv[l];
                }
                batch::VectorMath<W>::vSqrt(w.data(), r.data());
                for (std::size_t l = 0; l < W; ++l) {
                    adjl[l] += a[l] / r[l];
                }
                break;
            }
            case RevOp::Acosh: {
                std::array<double, W> w{};
                std::array<double, W> r{};
                for (std::size_t l = 0; l < W; ++l) {
                    w[l] = xv[l] * xv[l] - 1.0;
                }
                batch::VectorMath<W>::vSqrt(w.data(), r.data());
                for (std::size_t l = 0; l < W; ++l) {
                    adjl[l] += a[l] / r[l];
                }
                break;
            }
            case RevOp::Atanh:
                for (std::size_t l = 0; l < W; ++l) {
                    adjl[l] += a[l] / (1.0 - xv[l] * xv[l]);
                }
                break;
            case RevOp::Erf: {
                std::array<double, W> arg{};
                std::array<double, W> e{};
                for (std::size_t l = 0; l < W; ++l) {
                    arg[l] = -xv[l] * xv[l];
                }
                batch::VectorMath<W>::vExp(arg.data(), e.data());
                for (std::size_t l = 0; l < W; ++l) {
                    adjl[l] += a[l] * kTwoOverSqrtPi * e[l];
                }
                break;
            }
            case RevOp::Erfc: {
                std::array<double, W> arg{};
                std::array<double, W> e{};
                for (std::size_t l = 0; l < W; ++l) {
                    arg[l] = -xv[l] * xv[l];
                }
                batch::VectorMath<W>::vExp(arg.data(), e.data());
                for (std::size_t l = 0; l < W; ++l) {
                    adjl[l] -= a[l] * kTwoOverSqrtPi * e[l];
                }
                break;
            }
            case RevOp::Cbrt:
                for (std::size_t l = 0; l < W; ++l) {
                    adjl[l] += a[l] / (3.0 * x[l] * x[l]);
                }
                break;
            case RevOp::Exp2:
                for (std::size_t l = 0; l < W; ++l) {
                    adjl[l] += a[l] * kLn2 * x[l];
                }
                break;
            case RevOp::Log2:
                for (std::size_t l = 0; l < W; ++l) {
                    adjl[l] += a[l] / (xv[l] * kLn2);
                }
                break;
            case RevOp::Log10:
                for (std::size_t l = 0; l < W; ++l) {
                    adjl[l] += a[l] / (xv[l] * kLn10);
                }
                break;
            case RevOp::Hypot:
                for (std::size_t l = 0; l < W; ++l) {
                    const double h = x[l];
                    if (h != 0.0) {
                        adjl[l] += a[l] * xv[l] / h;
                        adjr[l] += a[l] * yv[l] / h;
                    }
                }
                break;
            case RevOp::Atan2:
                // atan2(y, x): lhs = y, rhs = x, same rule as the scalar sweep.
                for (std::size_t l = 0; l < W; ++l) {
                    const double d = xv[l] * xv[l] + yv[l] * yv[l];
                    if (d > 0.0) {
                        adjl[l] += a[l] * yv[l] / d;
                        adjr[l] -= a[l] * xv[l] / d;
                    }
                }
                break;
            case RevOp::MinL:
            case RevOp::MaxL:
                // Batch `fmin`/`fmax` mixed lanes: `lhsValues` carries the
                // per-lane winner mask (1.0 = lhs), unlike the scalar rule.
                for (std::size_t l = 0; l < W; ++l) {
                    if (xv[l] != 0.0) {
                        adjl[l] += a[l];
                    } else {
                        adjr[l] += a[l];
                    }
                }
                break;
            case RevOp::MinR:
            case RevOp::MaxR:
                // Uniform rhs-winner analogue of the scalar rule.
                for (std::size_t l = 0; l < W; ++l) {
                    adjr[l] += a[l];
                }
                break;
            case RevOp::LogGamma:
            case RevOp::TGamma:
            case RevOp::External:
            case RevOp::CopySign:
            case RevOp::Digamma:
            case RevOp::Trigamma:
            case RevOp::Tetragamma:
                break; // not recorded by the batch scalar API
        }
    }

    /// Stage one node at the cursor (the caller guarantees capacity) and bump
    /// it; one plane-wise store per field, adjoints start at zero.
    std::uint32_t stageNode(RevOp op, std::uint32_t lhs, std::uint32_t rhs,
                            const std::array<double, W>& value,
                            const std::array<double, W>& lhsValue,
                            const std::array<double, W>& rhsValue) {
        Chunk& chunk = *m_chunks[m_size >> kChunkShift];
        const std::size_t off = m_size & kChunkMask;
        chunk.op[off] = op;
        chunk.lhs[off] = lhs;
        chunk.rhs[off] = rhs;
        chunk.values[off] = value;
        chunk.lhsValues[off] = lhsValue;
        chunk.rhsValues[off] = rhsValue;
        chunk.adjoints[off] = {};
        return static_cast<std::uint32_t>(m_size++);
    }

    void ensureCapacity(std::size_t total) {
        if (total <= m_capacity) [[likely]] {
            return;
        }
        const std::size_t need = (total + kChunkNodes - 1) >> kChunkShift;
        m_chunks.reserve(need);
        while (m_chunks.size() < need) {
            m_chunks.push_back(makeChunk(kChunkNodes));
        }
        m_capacity = m_chunks.size() << kChunkShift;
    }

    std::array<double, W>& adjointRef(std::uint32_t idx) noexcept {
        return m_chunks[idx >> kChunkShift]->adjoints[idx & kChunkMask];
    }

    std::vector<std::unique_ptr<Chunk>> m_chunks;
    std::size_t m_size = 1; // sink lives at index 0
    std::size_t m_capacity = 0;
    // Chunk geometry is compile-time (`kChunkShift`/`kChunkNodes`/`kChunkMask`):
    // no per-instance copies exist, so the inlined record path reads constants
    // and never reloads geometry members after a plane store.
};

namespace detail {

/// Binary record: collapses to a node-less constant when both operands are
/// constants (so `x * 2.0` is one node with `rhs == kNone` / `rhsValue == 2.0`,
/// exactly like `RevTape`).
template <std::size_t W>
BatchScalar<W> batchBinary(RevOp op, const BatchScalar<W>& a, const BatchScalar<W>& b,
                           const std::array<double, W>& value) {
    if (a.node == BatchTape<W>::kNone && b.node == BatchTape<W>::kNone) {
        return BatchScalar<W>(value);
    }
    return BatchScalar<W>(value,
                          BatchTape<W>::active().push(op, a.node, b.node, value, a.value, b.value));
}

template <std::size_t W>
BatchScalar<W> batchUnary(RevOp op, const BatchScalar<W>& a, const std::array<double, W>& value) {
    if (a.node == BatchTape<W>::kNone) {
        return BatchScalar<W>(value);
    }
    return BatchScalar<W>(
        value, BatchTape<W>::active().push(op, a.node, BatchTape<W>::kNone, value, a.value));
}

} // namespace detail

// ── binary arithmetic ──

template <std::size_t W>
BatchScalar<W> operator+(const BatchScalar<W>& a, const BatchScalar<W>& b) {
    std::array<double, W> v{};
    for (std::size_t l = 0; l < W; ++l) {
        v[l] = a.value[l] + b.value[l];
    }
    return detail::batchBinary(RevOp::Add, a, b, v);
}

template <std::size_t W>
BatchScalar<W> operator-(const BatchScalar<W>& a, const BatchScalar<W>& b) {
    std::array<double, W> v{};
    for (std::size_t l = 0; l < W; ++l) {
        v[l] = a.value[l] - b.value[l];
    }
    return detail::batchBinary(RevOp::Sub, a, b, v);
}

template <std::size_t W>
BatchScalar<W> operator*(const BatchScalar<W>& a, const BatchScalar<W>& b) {
    std::array<double, W> v{};
    for (std::size_t l = 0; l < W; ++l) {
        v[l] = a.value[l] * b.value[l];
    }
    return detail::batchBinary(RevOp::Mul, a, b, v);
}

template <std::size_t W>
BatchScalar<W> operator/(const BatchScalar<W>& a, const BatchScalar<W>& b) {
    std::array<double, W> v{};
    for (std::size_t l = 0; l < W; ++l) {
        v[l] = a.value[l] / b.value[l];
    }
    return detail::batchBinary(RevOp::Div, a, b, v);
}

template <std::size_t W>
BatchScalar<W> operator-(const BatchScalar<W>& a) {
    std::array<double, W> v{};
    for (std::size_t l = 0; l < W; ++l) {
        v[l] = -a.value[l];
    }
    return detail::batchUnary(RevOp::Neg, a, v);
}

// ── arithmetic with a broadcast double (node-less operand) ──

template <std::size_t W>
BatchScalar<W> operator+(const BatchScalar<W>& a, double b) {
    return a + BatchScalar<W>(b);
}
template <std::size_t W>
BatchScalar<W> operator+(double a, const BatchScalar<W>& b) {
    return BatchScalar<W>(a) + b;
}
template <std::size_t W>
BatchScalar<W> operator-(const BatchScalar<W>& a, double b) {
    return a - BatchScalar<W>(b);
}
template <std::size_t W>
BatchScalar<W> operator-(double a, const BatchScalar<W>& b) {
    return BatchScalar<W>(a) - b;
}
template <std::size_t W>
BatchScalar<W> operator*(const BatchScalar<W>& a, double b) {
    return a * BatchScalar<W>(b);
}
template <std::size_t W>
BatchScalar<W> operator*(double a, const BatchScalar<W>& b) {
    return BatchScalar<W>(a) * b;
}
template <std::size_t W>
BatchScalar<W> operator/(const BatchScalar<W>& a, double b) {
    return a / BatchScalar<W>(b);
}
template <std::size_t W>
BatchScalar<W> operator/(double a, const BatchScalar<W>& b) {
    return BatchScalar<W>(a) / b;
}

// ── comparisons on the primal: one bool per lane ──

template <std::size_t W>
std::array<bool, W> operator<(const BatchScalar<W>& a, const BatchScalar<W>& b) {
    std::array<bool, W> m{};
    for (std::size_t l = 0; l < W; ++l) {
        m[l] = a.value[l] < b.value[l];
    }
    return m;
}
template <std::size_t W>
std::array<bool, W> operator>(const BatchScalar<W>& a, const BatchScalar<W>& b) {
    std::array<bool, W> m{};
    for (std::size_t l = 0; l < W; ++l) {
        m[l] = a.value[l] > b.value[l];
    }
    return m;
}
template <std::size_t W>
std::array<bool, W> operator<=(const BatchScalar<W>& a, const BatchScalar<W>& b) {
    std::array<bool, W> m{};
    for (std::size_t l = 0; l < W; ++l) {
        m[l] = a.value[l] <= b.value[l];
    }
    return m;
}
template <std::size_t W>
std::array<bool, W> operator>=(const BatchScalar<W>& a, const BatchScalar<W>& b) {
    std::array<bool, W> m{};
    for (std::size_t l = 0; l < W; ++l) {
        m[l] = a.value[l] >= b.value[l];
    }
    return m;
}
template <std::size_t W>
std::array<bool, W> operator==(const BatchScalar<W>& a, const BatchScalar<W>& b) {
    std::array<bool, W> m{};
    for (std::size_t l = 0; l < W; ++l) {
        m[l] = a.value[l] == b.value[l];
    }
    return m;
}
template <std::size_t W>
std::array<bool, W> operator!=(const BatchScalar<W>& a, const BatchScalar<W>& b) {
    std::array<bool, W> m{};
    for (std::size_t l = 0; l < W; ++l) {
        m[l] = a.value[l] != b.value[l];
    }
    return m;
}

// ── elementary functions (primal lane values via VectorMath<W>) ──

template <std::size_t W>
BatchScalar<W> sqrt(const BatchScalar<W>& x) {
    std::array<double, W> v{};
    batch::VectorMath<W>::vSqrt(x.value.data(), v.data());
    return detail::batchUnary(RevOp::Sqrt, x, v);
}

template <std::size_t W>
BatchScalar<W> exp(const BatchScalar<W>& x) {
    std::array<double, W> v{};
    batch::VectorMath<W>::vExp(x.value.data(), v.data());
    return detail::batchUnary(RevOp::Exp, x, v);
}

template <std::size_t W>
BatchScalar<W> log(const BatchScalar<W>& x) {
    std::array<double, W> v{};
    batch::VectorMath<W>::vLog(x.value.data(), v.data());
    return detail::batchUnary(RevOp::Log, x, v);
}

template <std::size_t W>
BatchScalar<W> expm1(const BatchScalar<W>& x) {
    std::array<double, W> v{};
    batch::VectorMath<W>::vExpm1(x.value.data(), v.data());
    return detail::batchUnary(RevOp::Expm1, x, v);
}

template <std::size_t W>
BatchScalar<W> log1p(const BatchScalar<W>& x) {
    std::array<double, W> v{};
    batch::VectorMath<W>::vLog1p(x.value.data(), v.data());
    return detail::batchUnary(RevOp::Log1p, x, v);
}

template <std::size_t W>
BatchScalar<W> cbrt(const BatchScalar<W>& x) {
    std::array<double, W> v{};
    batch::VectorMath<W>::vCbrt(x.value.data(), v.data());
    return detail::batchUnary(RevOp::Cbrt, x, v);
}

template <std::size_t W>
BatchScalar<W> exp2(const BatchScalar<W>& x) {
    std::array<double, W> v{};
    batch::VectorMath<W>::vExp2(x.value.data(), v.data());
    return detail::batchUnary(RevOp::Exp2, x, v);
}

template <std::size_t W>
BatchScalar<W> log2(const BatchScalar<W>& x) {
    std::array<double, W> v{};
    batch::VectorMath<W>::vLog2(x.value.data(), v.data());
    return detail::batchUnary(RevOp::Log2, x, v);
}

template <std::size_t W>
BatchScalar<W> log10(const BatchScalar<W>& x) {
    std::array<double, W> v{};
    batch::VectorMath<W>::vLog10(x.value.data(), v.data());
    return detail::batchUnary(RevOp::Log10, x, v);
}

template <std::size_t W>
BatchScalar<W> sin(const BatchScalar<W>& x) {
    std::array<double, W> v{};
    batch::VectorMath<W>::vSin(x.value.data(), v.data());
    return detail::batchUnary(RevOp::Sin, x, v);
}

template <std::size_t W>
BatchScalar<W> cos(const BatchScalar<W>& x) {
    std::array<double, W> v{};
    batch::VectorMath<W>::vCos(x.value.data(), v.data());
    return detail::batchUnary(RevOp::Cos, x, v);
}

/// sin and cos of `x` in one call. Two nodes by design (one Sin + one Cos
/// sharing the parent), exactly like the scalar `sincos`; constant arguments
/// stay node-less.
template <std::size_t W>
void sincos(const BatchScalar<W>& x, BatchScalar<W>& s, BatchScalar<W>& c) {
    std::array<double, W> sv{};
    std::array<double, W> cv{};
    batch::VectorMath<W>::vSinCos(x.value.data(), sv.data(), cv.data());
    s = detail::batchUnary(RevOp::Sin, x, sv);
    c = detail::batchUnary(RevOp::Cos, x, cv);
}

template <std::size_t W>
BatchScalar<W> tan(const BatchScalar<W>& x) {
    std::array<double, W> v{};
    batch::VectorMath<W>::vTan(x.value.data(), v.data());
    return detail::batchUnary(RevOp::Tan, x, v);
}

template <std::size_t W>
BatchScalar<W> asin(const BatchScalar<W>& x) {
    std::array<double, W> v{};
    batch::VectorMath<W>::vAsin(x.value.data(), v.data());
    return detail::batchUnary(RevOp::Asin, x, v);
}

template <std::size_t W>
BatchScalar<W> acos(const BatchScalar<W>& x) {
    std::array<double, W> v{};
    batch::VectorMath<W>::vAcos(x.value.data(), v.data());
    return detail::batchUnary(RevOp::Acos, x, v);
}

template <std::size_t W>
BatchScalar<W> atan(const BatchScalar<W>& x) {
    std::array<double, W> v{};
    batch::VectorMath<W>::vAtan(x.value.data(), v.data());
    return detail::batchUnary(RevOp::Atan, x, v);
}

template <std::size_t W>
BatchScalar<W> sinh(const BatchScalar<W>& x) {
    std::array<double, W> v{};
    batch::VectorMath<W>::vSinh(x.value.data(), v.data());
    return detail::batchUnary(RevOp::Sinh, x, v);
}

template <std::size_t W>
BatchScalar<W> cosh(const BatchScalar<W>& x) {
    std::array<double, W> v{};
    batch::VectorMath<W>::vCosh(x.value.data(), v.data());
    return detail::batchUnary(RevOp::Cosh, x, v);
}

template <std::size_t W>
BatchScalar<W> tanh(const BatchScalar<W>& x) {
    std::array<double, W> v{};
    batch::VectorMath<W>::vTanh(x.value.data(), v.data());
    return detail::batchUnary(RevOp::Tanh, x, v);
}

/// Inverse hyperbolic lane values (`asinh` all reals, `acosh` x >= 1,
/// `atanh` |x| < 1); the lane values come from the libm-only VectorMath
/// entries, the sweep matches the scalar `RevTape` partials lane for lane.
template <std::size_t W>
BatchScalar<W> asinh(const BatchScalar<W>& x) {
    std::array<double, W> v{};
    batch::VectorMath<W>::vAsinh(x.value.data(), v.data());
    return detail::batchUnary(RevOp::Asinh, x, v);
}

template <std::size_t W>
BatchScalar<W> acosh(const BatchScalar<W>& x) {
    std::array<double, W> v{};
    batch::VectorMath<W>::vAcosh(x.value.data(), v.data());
    return detail::batchUnary(RevOp::Acosh, x, v);
}

template <std::size_t W>
BatchScalar<W> atanh(const BatchScalar<W>& x) {
    std::array<double, W> v{};
    batch::VectorMath<W>::vAtanh(x.value.data(), v.data());
    return detail::batchUnary(RevOp::Atanh, x, v);
}

template <std::size_t W>
BatchScalar<W> erf(const BatchScalar<W>& x) {
    std::array<double, W> v{};
    batch::VectorMath<W>::vErf(x.value.data(), v.data());
    return detail::batchUnary(RevOp::Erf, x, v);
}

template <std::size_t W>
BatchScalar<W> erfc(const BatchScalar<W>& x) {
    std::array<double, W> v{};
    batch::VectorMath<W>::vErfc(x.value.data(), v.data());
    return detail::batchUnary(RevOp::Erfc, x, v);
}

template <std::size_t W>
BatchScalar<W> fabs(const BatchScalar<W>& x) {
    using std::fabs;
    std::array<double, W> v{};
    for (std::size_t l = 0; l < W; ++l) {
        v[l] = fabs(x.value[l]);
    }
    return detail::batchUnary(RevOp::Abs, x, v);
}

template <std::size_t W>
BatchScalar<W> abs(const BatchScalar<W>& x) {
    return fabs(x);
}

template <std::size_t W>
BatchScalar<W> pow(const BatchScalar<W>& a, const BatchScalar<W>& b) {
    std::array<double, W> v{};
    batch::VectorMath<W>::vPow(a.value.data(), b.value.data(), v.data());
    return detail::batchBinary(RevOp::Pow, a, b, v);
}

/// `pow(a, b)` with a W-lane constant exponent: the exponent stays node-less
/// (its primals cache in the node's `rhsValue` plane), like `RevTape`'s
/// `pow(RevScalar, double)`.
template <std::size_t W>
BatchScalar<W> pow(const BatchScalar<W>& a, const std::array<double, W>& b) {
    std::array<double, W> v{};
    batch::VectorMath<W>::vPow(a.value.data(), b.data(), v.data());
    if (a.node == BatchTape<W>::kNone) {
        return BatchScalar<W>(v);
    }
    return BatchScalar<W>(
        v, BatchTape<W>::active().push(RevOp::Pow, a.node, BatchTape<W>::kNone, v, a.value, b));
}

template <std::size_t W>
BatchScalar<W> pow(const BatchScalar<W>& a, double b) {
    std::array<double, W> e{};
    e.fill(b);
    return pow(a, e);
}

/// `atan2(y, x)` (lhs = y, rhs = x), same partials as the scalar rule.
template <std::size_t W>
BatchScalar<W> atan2(const BatchScalar<W>& y, const BatchScalar<W>& x) {
    std::array<double, W> v{};
    batch::VectorMath<W>::vAtan2(y.value.data(), x.value.data(), v.data());
    return detail::batchBinary(RevOp::Atan2, y, x, v);
}

template <std::size_t W>
BatchScalar<W> hypot(const BatchScalar<W>& a, const BatchScalar<W>& b) {
    std::array<double, W> v{};
    batch::VectorMath<W>::vHypot(a.value.data(), b.value.data(), v.data());
    return detail::batchBinary(RevOp::Hypot, a, b, v);
}

// ── fused composites (same statement structure as the scalar ops) ──

/// phi(x) = exp(-x^2/2)/sqrt(2*pi), composed exactly like `lightape::normpdf`
/// (`kInvSqrt2Pi * exp(-0.5 * (x * x))`), so the recorded statements and the
/// per-lane bits match the scalar tape's composite.
template <std::size_t W>
BatchScalar<W> normpdf(const BatchScalar<W>& x) {
    return BatchScalar<W>(batch::detail::kInvSqrt2Pi) * exp(BatchScalar<W>(-0.5) * (x * x));
}

/// Phi(x) = erfc(-x/sqrt(2))/2, composed exactly like `lightape::normcdf`
/// (`0.5 * erfc(-kInvSqrt2 * x)`), i.e. through the batched `erfc` op.
template <std::size_t W>
BatchScalar<W> normcdf(const BatchScalar<W>& x) {
    return BatchScalar<W>(0.5) * erfc(BatchScalar<W>(-batch::detail::kInvSqrt2) * x);
}

/// `fmin` on the primal: a lane-uniform decision returns the winner untouched
/// (`RevTape`'s node sharing, so the statement structure matches the scalar
/// tape exactly); a mixed decision records one mask-carrying `MinL` node.
template <std::size_t W>
BatchScalar<W> fmin(const BatchScalar<W>& a, const BatchScalar<W>& b) {
    std::array<bool, W> take{};
    bool allA = true;
    bool allB = true;
    for (std::size_t l = 0; l < W; ++l) {
        take[l] = a.value[l] <= b.value[l];
        allA = allA && take[l];
        allB = allB && !take[l];
    }
    if (allA) {
        return a;
    }
    if (allB) {
        return b;
    }
    std::array<double, W> v{};
    std::array<double, W> mask{};
    for (std::size_t l = 0; l < W; ++l) {
        v[l] = take[l] ? a.value[l] : b.value[l];
        mask[l] = take[l] ? 1.0 : 0.0;
    }
    return BatchScalar<W>(
        v, BatchTape<W>::active().push(RevOp::MinL, a.node, b.node, v, mask, b.value));
}

/// `fmax` counterpart of `fmin` (mask-carrying `MaxL` on mixed lanes).
template <std::size_t W>
BatchScalar<W> fmax(const BatchScalar<W>& a, const BatchScalar<W>& b) {
    std::array<bool, W> take{};
    bool allA = true;
    bool allB = true;
    for (std::size_t l = 0; l < W; ++l) {
        take[l] = a.value[l] >= b.value[l];
        allA = allA && take[l];
        allB = allB && !take[l];
    }
    if (allA) {
        return a;
    }
    if (allB) {
        return b;
    }
    std::array<double, W> v{};
    std::array<double, W> mask{};
    for (std::size_t l = 0; l < W; ++l) {
        v[l] = take[l] ? a.value[l] : b.value[l];
        mask[l] = take[l] ? 1.0 : 0.0;
    }
    return BatchScalar<W>(
        v, BatchTape<W>::active().push(RevOp::MaxL, a.node, b.node, v, mask, b.value));
}

} // namespace lightape

#endif // LIGHTAPE_BATCH_BATCHTAPE_H
