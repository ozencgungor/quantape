#ifndef LIGHTAPE_REV_REVTAPET_H
#define LIGHTAPE_REV_REVTAPET_H

#include <cmath>
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <new>
#include <type_traits>
#include <vector>

#include "lightape/rev/LeanReverse.h"

namespace lightape {
/**
 * @file RevTapeT.h
 * @brief rev-over-rev storage scaffold: the lean reverse tape over a payload
 * type `T` (decision #8; `rev_rev_and_api.md` Part A).
 *
 * `RevTapeT<double>` is the storage/sweep design of `RevTape` expressed over a
 * scalar payload type `T` instead of hard-coded `double` values: the same
 * 4096-node 64-byte-aligned chunks, the same sink node 0, node-less constants
 * (a constant lives in the parent-value slot), the same consume-on-use reverse
 * sweep with the zero-skip, and a thread-local tape per `T` instantiation
 * (`RevTapeT<RevScalar>` and `RevTapeT<double>` can coexist on one thread,
 * which is exactly what a nested rev-over-rev Hessian needs).
 *
 * Scope (deliberately storage-only, decision #8). Op set is the arithmetic
 * core used by every reverse rule: Input/Add/Sub/Mul/Div/Neg/Sqrt/Exp/Log/
 * Abs/Pow/MinL/MinR/MaxL/MaxR. `External` and the special functions
 * (expm1/log1p, the trigonometric/hyperbolic family, erf/erfc, the gamma
 * family, cbrt/exp2/log2/log10, hypot/atan2/copysign) are deliberately
 * absent: a nested scalar's public op layer must be ported to the templates
 * first, and rev-over-rev only needs pure elementary expressions. There is
 * also no CSE cache, no `External` side table, no `configureChunkSize`, no
 * multi-root/multi-seed sweep and no replay helpers; those are enabled
 * together with the first real driver.
 *
 * Enablement path to `T = RevScalar` (rev-over-rev Hessian). All value and
 * adjoint arithmetic below is already spelled in `T`, so the sweep is valid
 * for a nested AD scalar once the public ops layer (`ops/`; `rev/fwd/ops` in
 * `module_layout.md`) is templated over `RevScalarT<T>` instead of the
 * concrete `RevScalar`, and the nested scalar supplies the `isZero()`
 * full-payload test used by the zero-skip (an `operator==` on the primal
 * alone would wrongly skip a node that carries only derivative adjoints).
 * The node layout then grows with `T` exactly as the plan predicts:
 * `RevNodeT<double>` stays 48 B (asserted below), while
 * `RevNodeT<RevScalar>` is 80 B (4 x 16 B payload + 12 B meta + padding, the
 * "80 B ops" of decision #8) and `RevNodeT<Tangent<double,N>>` follows the
 * `32N + 48` B family; larger nodes mean fewer nodes per 4096-node chunk in
 * absolute bytes, which is expected and decision-tracked.
 *
 * Requirements on `T` (documented by the `RevPayloadT` concept below):
 * trivially copyable, default-constructible, constructible from `double`
 * (constant operands), the arithmetic operators `+ - * /` and unary `-`, ADL
 * `sqrt/exp/log/fabs/pow` for the primal evaluations, comparisons
 * (`< <= > >= == !=`) on the primal for branch decisions, and a zero
 * predicate (`isZero()` member, else full-payload `==`).
 */

namespace detail {

// Bring the standard math into `lightape::detail` so the requirement concept
// can spell unqualified calls exactly like the engine does: ordinary lookup
// finds the std overloads for double, ADL adds the payload type's overloads.

using std::exp;
using std::fabs;
using std::log;
using std::pow;
using std::sqrt;

/// Sweep zero-skip predicate. A node is skipped only when *every* component
/// of its adjoint is zero, so a nested AD scalar whose primal adjoint is zero
/// but whose derivative adjoint is live still sweeps. A payload type can
/// supply a componentwise `isZero()`; otherwise full-object equality `== T{}`
/// is used (the `double` case, where the two coincide). The nested
/// `RevScalar` must get `isZero()` before `RevTapeT<RevScalar>` is enabled:
/// its `operator==` compares the primal only.
template <typename T>
constexpr bool revTapeZero(const T& t) {
    if constexpr (requires {
                      { t.isZero() } -> std::convertible_to<bool>;
                  }) {
        return static_cast<bool>(t.isZero());
    } else {
        return t == T{};
    }
}

/// Unqualified primal-evaluation helpers for the sweep: `using std::...` plus
/// ADL, exactly like the free operators, so a payload type's own `pow`/`log`
/// overloads are found (a qualified `detail::pow` would only see the std
/// declarations brought in above).
template <typename T>
constexpr T primalPow(const T& a, const T& b) {
    using std::pow;
    return pow(a, b);
}
template <typename T>
constexpr T primalLog(const T& a) {
    using std::log;
    return log(a);
}

/// Documented requirements on the payload scalar `T` of `RevTapeT<T>`
/// (checked by the tape's `static_assert`, so a bad instantiation fails with
/// this name rather than a template avalanche).
template <typename T>
concept RevPayloadT = std::is_trivially_copyable_v<T> && std::is_default_constructible_v<T> &&
                      std::is_constructible_v<T, double> && requires(const T& a, const T& b) {
                          { a + b } -> std::convertible_to<T>;
                          { a - b } -> std::convertible_to<T>;
                          { a * b } -> std::convertible_to<T>;
                          { a / b } -> std::convertible_to<T>;
                          { -a } -> std::convertible_to<T>;
                          { sqrt(a) } -> std::convertible_to<T>;
                          { exp(a) } -> std::convertible_to<T>;
                          { log(a) } -> std::convertible_to<T>;
                          { fabs(a) } -> std::convertible_to<T>;
                          { pow(a, b) } -> std::convertible_to<T>;
                          { a < b } -> std::convertible_to<bool>;
                          { a <= b } -> std::convertible_to<bool>;
                          { a > b } -> std::convertible_to<bool>;
                          { a >= b } -> std::convertible_to<bool>;
                          { a == b } -> std::convertible_to<bool>;
                          { a != b } -> std::convertible_to<bool>;
                          { revTapeZero(a) } -> std::convertible_to<bool>;
                      };

} // namespace detail

/// Tape node with a `T` payload: value, adjoint, cached parent values and the
/// parent node indices. Caching the parent values inline keeps the sweep
/// independent of parent-node reads (constants have no node; their values
/// live here) and keeps the sweep sequential, exactly like `RevNode`.
/// `T = double` reproduces the 48-byte `RevNode`; nested payloads grow the
/// node as the plan's `32N + 48` family (decision #8).
template <typename T>
struct RevNodeT {
    T value{};
    T adjoint{};
    T lhsValue{};
    T rhsValue{};
    std::uint32_t lhs = 0u;
    std::uint32_t rhs = 0u;
    RevOp op = RevOp::Input;
};

static_assert(sizeof(RevNodeT<double>) == 48, "RevNodeT<double> layout gate: 48-byte AoS node");

template <typename T>
class RevTapeT;

/// Value + tape node index; `kNone` = plain constant (no node). The nested
/// scalar analogue of `RevScalar`: `RevScalarT<double>` is layout-compatible
/// with `{double, uint32}` and `RevScalarT<RevScalar>` is the rev-over-rev
/// outer scalar (24 B; 16 B payload + index + padding).
template <typename T>
struct RevScalarT {
    T value{};
    std::uint32_t node = 0u;

    RevScalarT() = default;
    RevScalarT(T v) : value(v) {} // implicit: constants never hit the tape
    RevScalarT(T v, std::uint32_t n) : value(v), node(n) {}
};

/// Thread-local reverse tape over payload `T`: chunked append-only arena
/// (4096-node, 64-byte-aligned chunks) with a manual cursor and chunks reused
/// across paths. This is the storage/sweep half of the `RevTapeT<T>`
/// generalization (decision #8): `RevTapeT<double>` performs the same
/// arithmetic and the same statement recording as `RevTape`, while every
/// value/adjoint read/write is a `T` expression, so the sweep stays valid
/// when `T` is itself an AD scalar.
template <typename T>
class RevTapeT {
    static_assert(detail::RevPayloadT<T>,
                  "RevTapeT<T>: T must be a payload scalar (see RevPayloadT)");

public:
    /// Node 0 is a permanent dead sink: constant operands point at it, so the
    /// reverse sweep needs no null-parent branches; its adjoint is ignored.
    static constexpr std::uint32_t kNone = 0u;

    static constexpr std::size_t kChunkShift = 12;
    static constexpr std::size_t kChunkNodes = std::size_t{1} << kChunkShift;
    static constexpr std::size_t kChunkMask = kChunkNodes - 1;

    /// Shared accessor for the thread-local tape pointer. The `thread_local`
    /// lives inside this member function, so every `T` instantiation owns its
    /// own pointer: `RevTapeT<double>` and `RevTapeT<RevScalar>` never
    /// collide on one thread. Trivially initialized (no dynamic TLS init in
    /// the per-operation path); the arena itself is created once per thread
    /// on first `active()`.
    static RevTapeT*& currentPtr() {
        static thread_local RevTapeT* tape = nullptr;
        return tape;
    }

    static RevTapeT& active() {
        RevTapeT*& tape = currentPtr();
        if (tape == nullptr) [[unlikely]] {
            tape = new RevTapeT();
        }
        return *tape;
    }

    /// Destroy this thread's tape and forget it (a null tape is a no-op).
    /// Every `RevScalarT<T>` node index becomes invalid.
    static void releaseCurrentThread() {
        RevTapeT*& tape = currentPtr();
        delete tape;
        tape = nullptr;
    }

    RevTapeT() {
        m_chunks.push_back(makeChunk(kChunkNodes));
        m_capacity = kChunkNodes;
    }

    RevTapeT(const RevTapeT&) = delete;
    RevTapeT& operator=(const RevTapeT&) = delete;

    /// Reset for a new recording: keep the sink and every allocated chunk.
    void clear() noexcept { m_size = 1; }

    /// Absolute node count including the sink (the next free index).
    std::size_t position() const noexcept { return m_size; }
    std::size_t size() const noexcept { return m_size - 1; } // excluding sink

    void reserve(std::size_t n) { ensureCapacity(n + 1); }

    /// Truncate to a previously recorded position (LIFO); surviving chunks
    /// stay allocated and surviving adjoints are preserved (call
    /// `resetAdjoints()` before re-sweeping a sub-tape).
    void resetTo(std::size_t pos) {
        if (pos >= 1 && pos <= m_size) {
            m_size = pos;
        }
    }

    /// Zero every adjoint (sink included) without touching the recording.
    void resetAdjoints() {
        for (std::size_t i = 0; i < m_size; ++i) {
            node(i).adjoint = T{};
        }
    }

    /// Independent variable (what gradients are read from).
    std::uint32_t input(const T& value) { return push(RevOp::Input, kNone, kNone, value); }

    /// Append one op node: node-less constants (`kNone`) point at the dead
    /// sink and cache their primal in the corresponding `lhsValue`/`rhsValue`
    /// slot, so no constant leaf is ever recorded. No CSE cache and no
    /// `External`: this scaffold records the arithmetic core verbatim.
    std::uint32_t push(RevOp op, std::uint32_t lhs, std::uint32_t rhs, const T& value,
                       const T& lhsValue = T{}, const T& rhsValue = T{}) {
        if (m_size < m_capacity) [[likely]] {
            return stageNode(op, lhs, rhs, value, lhsValue, rhsValue);
        }
        ensureCapacity(m_size + 1);
        return stageNode(op, lhs, rhs, value, lhsValue, rhsValue);
    }

    const T& value(std::uint32_t idx) const noexcept { return node(idx).value; }
    const T& adjoint(std::uint32_t idx) const noexcept { return node(idx).adjoint; }

    /// Reverse sweep seeded at `root`. Consume-on-use: non-leaf adjoints are
    /// zeroed as they are read (a finished sweep leaves no stale state),
    /// while leaf (`Input`) adjoints are kept for the caller to read out.
    /// Nodes with an all-zero adjoint are skipped (`detail::revTapeZero`).
    /// The sweep walks chunk by chunk backwards; the whole node arithmetic is
    /// expressed in `T`, which is what makes the same code a rev-over-rev
    /// sweep when `T` is an AD scalar.
    void reverse(std::uint32_t root, const T& seed = T(1.0)) {
        node(root).adjoint = seed;
        if (m_size <= 1) {
            return;
        }
        const std::size_t top = m_size - 1;
        const std::size_t lastChunk = top >> kChunkShift;
        const std::size_t lastOffset = (top & kChunkMask) + 1;
        for (std::size_t c = lastChunk + 1; c-- > 0;) {
            RevNodeT<T>* nodes = m_chunks[c]->nodes.get();
            std::size_t j = (c == lastChunk) ? lastOffset : kChunkNodes;
            const std::size_t stop = (c == 0) ? 1 : 0;
            while (j > stop) {
                RevNodeT<T>& n = nodes[--j];
                if (n.op == RevOp::Input) {
                    continue; // leaves keep their adjoints for readout
                }
                T a = n.adjoint;
                n.adjoint = T{};
                if (detail::revTapeZero(a)) {
                    continue;
                }
                switch (n.op) {
                    case RevOp::Add:
                        addTo(n.lhs, a);
                        addTo(n.rhs, a);
                        break;
                    case RevOp::Sub:
                        addTo(n.lhs, a);
                        subFrom(n.rhs, a);
                        break;
                    case RevOp::Mul:
                        addTo(n.lhs, a * n.rhsValue);
                        addTo(n.rhs, a * n.lhsValue);
                        break;
                    case RevOp::Div: {
                        const T rv = n.rhsValue;
                        addTo(n.lhs, a / rv);
                        subFrom(n.rhs, a * n.value / rv);
                        break;
                    }
                    case RevOp::Neg:
                        subFrom(n.lhs, a);
                        break;
                    case RevOp::Sqrt:
                        addTo(n.lhs, a * T(0.5) / n.value);
                        break;
                    case RevOp::Exp:
                        addTo(n.lhs, a * n.value);
                        break;
                    case RevOp::Log:
                        addTo(n.lhs, a / n.lhsValue);
                        break;
                    case RevOp::Abs:
                        addTo(n.lhs, a * (n.lhsValue >= T(0.0) ? T(1.0) : T(-1.0)));
                        break;
                    case RevOp::Pow:
                        addTo(n.lhs,
                              a * n.rhsValue * detail::primalPow(n.lhsValue, n.rhsValue - T(1.0)));
                        addTo(n.rhs, a * n.value * detail::primalLog(n.lhsValue));
                        break;
                    case RevOp::MinL:
                    case RevOp::MaxL:
                        addTo(n.lhs, a);
                        break;
                    case RevOp::MinR:
                    case RevOp::MaxR:
                        addTo(n.rhs, a);
                        break;
                    default:
                        // External and the special functions are deliberately
                        // absent from this storage scaffold (see the file
                        // comment); such an opcode can never be recorded here.
                        break;
                }
            }
        }
    }

private:
    /// Matching aligned-array delete for the 64-byte aligned node buffer.
    struct NodeDeleter {
        void operator()(RevNodeT<T>* p) const noexcept {
            ::operator delete[](p, std::align_val_t{64});
        }
    };

    /// One runtime-sized arena block; the node buffer is 64-byte aligned so
    /// each chunk starts on a cache line.
    struct Chunk {
        std::unique_ptr<RevNodeT<T>[], NodeDeleter> nodes;
    };

    static std::unique_ptr<Chunk> makeChunk(std::size_t n) {
        auto chunk = std::make_unique<Chunk>();
        chunk->nodes.reset(new (std::align_val_t{64}) RevNodeT<T>[n]);
        return chunk;
    }

    RevNodeT<T>& node(std::size_t i) noexcept {
        return m_chunks[i >> kChunkShift]->nodes[i & kChunkMask];
    }
    const RevNodeT<T>& node(std::size_t i) const noexcept {
        return m_chunks[i >> kChunkShift]->nodes[i & kChunkMask];
    }

    /// Parent-slot accumulation spelled with `+`/`-` only, so the payload
    /// protocol does not have to provide compound assignment.
    void addTo(std::uint32_t parent, const T& delta) {
        T& slot = node(parent).adjoint;
        slot = slot + delta;
    }
    void subFrom(std::uint32_t parent, const T& delta) {
        T& slot = node(parent).adjoint;
        slot = slot - delta;
    }

    /// Stage one node at the cursor (the caller guarantees capacity) and bump
    /// it; the adjoint starts zeroed.
    std::uint32_t stageNode(RevOp op, std::uint32_t lhs, std::uint32_t rhs, const T& value,
                            const T& lhsValue, const T& rhsValue) {
        RevNodeT<T>& cell = m_chunks[m_size >> kChunkShift]->nodes[m_size & kChunkMask];
        cell.value = value;
        cell.adjoint = T{};
        cell.lhsValue = lhsValue;
        cell.rhsValue = rhsValue;
        cell.lhs = lhs;
        cell.rhs = rhs;
        cell.op = op;
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

    std::vector<std::unique_ptr<Chunk>> m_chunks;
    std::size_t m_size = 1; // sink lives at index 0
    std::size_t m_capacity = 0;
};

namespace detail {

/// Binary record. A constant operand (`kNone`) contributes no parent node:
/// the node stores the sink in that parent slot and the constant's primal in
/// the matching value slot, exactly like `revBinary` on the concrete tape.
template <typename T>
RevScalarT<T> revBinaryT(RevOp op, const RevScalarT<T>& a, const RevScalarT<T>& b, const T& value) {
    if (a.node == RevTapeT<T>::kNone && b.node == RevTapeT<T>::kNone) {
        return RevScalarT<T>(value);
    }
    return RevScalarT<T>(value,
                         RevTapeT<T>::active().push(op, a.node, b.node, value, a.value, b.value));
}

/// Unary record whose primal is evaluated by the caller (ADL math on `T`).
template <typename T>
RevScalarT<T> revUnaryT(RevOp op, const RevScalarT<T>& a, const T& value) {
    if (a.node == RevTapeT<T>::kNone) {
        return RevScalarT<T>(value);
    }
    return RevScalarT<T>(
        value, RevTapeT<T>::active().push(op, a.node, RevTapeT<T>::kNone, value, a.value));
}

} // namespace detail

// ── compound assignment (rebinds to a fresh node: sharing-safe) ──

template <typename T>
RevScalarT<T>& operator+=(RevScalarT<T>& a, const RevScalarT<T>& b) {
    a = detail::revBinaryT(RevOp::Add, a, b, a.value + b.value);
    return a;
}
template <typename T>
RevScalarT<T>& operator-=(RevScalarT<T>& a, const RevScalarT<T>& b) {
    a = detail::revBinaryT(RevOp::Sub, a, b, a.value - b.value);
    return a;
}
template <typename T>
RevScalarT<T>& operator*=(RevScalarT<T>& a, const RevScalarT<T>& b) {
    a = detail::revBinaryT(RevOp::Mul, a, b, a.value * b.value);
    return a;
}
template <typename T>
RevScalarT<T>& operator/=(RevScalarT<T>& a, const RevScalarT<T>& b) {
    a = detail::revBinaryT(RevOp::Div, a, b, a.value / b.value);
    return a;
}

// ── binary arithmetic ──

template <typename T>
RevScalarT<T> operator+(const RevScalarT<T>& a, const RevScalarT<T>& b) {
    return detail::revBinaryT(RevOp::Add, a, b, a.value + b.value);
}
template <typename T>
RevScalarT<T> operator-(const RevScalarT<T>& a, const RevScalarT<T>& b) {
    return detail::revBinaryT(RevOp::Sub, a, b, a.value - b.value);
}
template <typename T>
RevScalarT<T> operator*(const RevScalarT<T>& a, const RevScalarT<T>& b) {
    return detail::revBinaryT(RevOp::Mul, a, b, a.value * b.value);
}
template <typename T>
RevScalarT<T> operator/(const RevScalarT<T>& a, const RevScalarT<T>& b) {
    return detail::revBinaryT(RevOp::Div, a, b, a.value / b.value);
}
template <typename T>
RevScalarT<T> operator-(const RevScalarT<T>& a) {
    return detail::revUnaryT(RevOp::Neg, a, -a.value);
}

// ── arithmetic with a plain-double constant (node-less operand) ──

template <typename T>
RevScalarT<T> operator+(const RevScalarT<T>& a, double b) {
    return a + RevScalarT<T>(T(b));
}
template <typename T>
RevScalarT<T> operator+(double a, const RevScalarT<T>& b) {
    return RevScalarT<T>(T(a)) + b;
}
template <typename T>
RevScalarT<T> operator-(const RevScalarT<T>& a, double b) {
    return a - RevScalarT<T>(T(b));
}
template <typename T>
RevScalarT<T> operator-(double a, const RevScalarT<T>& b) {
    return RevScalarT<T>(T(a)) - b;
}
template <typename T>
RevScalarT<T> operator*(const RevScalarT<T>& a, double b) {
    return a * RevScalarT<T>(T(b));
}
template <typename T>
RevScalarT<T> operator*(double a, const RevScalarT<T>& b) {
    return RevScalarT<T>(T(a)) * b;
}
template <typename T>
RevScalarT<T> operator/(const RevScalarT<T>& a, double b) {
    return a / RevScalarT<T>(T(b));
}
template <typename T>
RevScalarT<T> operator/(double a, const RevScalarT<T>& b) {
    return RevScalarT<T>(T(a)) / b;
}

// ── comparisons on the primal value ──

template <typename T>
bool operator<(const RevScalarT<T>& a, const RevScalarT<T>& b) {
    return a.value < b.value;
}
template <typename T>
bool operator>(const RevScalarT<T>& a, const RevScalarT<T>& b) {
    return a.value > b.value;
}
template <typename T>
bool operator<=(const RevScalarT<T>& a, const RevScalarT<T>& b) {
    return a.value <= b.value;
}
template <typename T>
bool operator>=(const RevScalarT<T>& a, const RevScalarT<T>& b) {
    return a.value >= b.value;
}
template <typename T>
bool operator==(const RevScalarT<T>& a, const RevScalarT<T>& b) {
    return a.value == b.value;
}
template <typename T>
bool operator!=(const RevScalarT<T>& a, const RevScalarT<T>& b) {
    return a.value != b.value;
}

// ── elementary functions (primal via ADL math on T) ──

template <typename T>
RevScalarT<T> sqrt(const RevScalarT<T>& x) {
    using std::sqrt;
    return detail::revUnaryT(RevOp::Sqrt, x, sqrt(x.value));
}
template <typename T>
RevScalarT<T> exp(const RevScalarT<T>& x) {
    using std::exp;
    return detail::revUnaryT(RevOp::Exp, x, exp(x.value));
}
template <typename T>
RevScalarT<T> log(const RevScalarT<T>& x) {
    using std::log;
    return detail::revUnaryT(RevOp::Log, x, log(x.value));
}
template <typename T>
RevScalarT<T> pow(const RevScalarT<T>& a, const RevScalarT<T>& b) {
    using std::pow;
    return detail::revBinaryT(RevOp::Pow, a, b, pow(a.value, b.value));
}
template <typename T>
RevScalarT<T> pow(const RevScalarT<T>& a, double b) {
    return pow(a, RevScalarT<T>(T(b)));
}
template <typename T>
RevScalarT<T> pow(double a, const RevScalarT<T>& b) {
    return pow(RevScalarT<T>(T(a)), b);
}
template <typename T>
RevScalarT<T> fabs(const RevScalarT<T>& x) {
    using std::fabs;
    return detail::revUnaryT(RevOp::Abs, x, fabs(x.value));
}
template <typename T>
RevScalarT<T> abs(const RevScalarT<T>& x) {
    return fabs(x);
}

/// Branch selection shares the winner's node (derivative along the branch),
/// exactly like the concrete tape's `fmin`/`fmax`.
template <typename T>
RevScalarT<T> fmin(const RevScalarT<T>& a, const RevScalarT<T>& b) {
    return a.value <= b.value ? a : b;
}
template <typename T>
RevScalarT<T> fmax(const RevScalarT<T>& a, const RevScalarT<T>& b) {
    return a.value >= b.value ? a : b;
}
template <typename T>
RevScalarT<T> fmin(const RevScalarT<T>& a, double b) {
    return fmin(a, RevScalarT<T>(T(b)));
}
template <typename T>
RevScalarT<T> fmax(const RevScalarT<T>& a, double b) {
    return fmax(a, RevScalarT<T>(T(b)));
}

} // namespace lightape

#endif // LIGHTAPE_REV_REVTAPET_H
