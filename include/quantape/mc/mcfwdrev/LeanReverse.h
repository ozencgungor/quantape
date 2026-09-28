#ifndef QUANTAPE_MC_LEAN_REVERSE_H
#define QUANTAPE_MC_LEAN_REVERSE_H

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace quantape::mc {
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
 * Two structural wins over Stan `var` in this hot path:
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
 * unary -, sqrt, exp, log, fabs, fmin, fmax; comparisons on the primal.
 * Branches (`fmin/fmax`, Eigen `cwiseMax`, ternaries) share the selected
 * operand's node, so adjoints flow along the taken branch (a.e. correct).
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
    Abs,
    MinL,
    MinR,
    MaxL,
    MaxR,
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
};

/// Thread-local reverse tape: append-only arena with a manual cursor
/// (no per-push bounds/branch beyond one predictable capacity check) and
/// capacity reused across paths.
class RevTape {
public:
    /// Node 0 is a permanent dead sink: constant operands point at it, so
    /// the reverse sweep needs no null-parent branches; its adjoint is
    /// ignored.
    static constexpr std::uint32_t kNone = 0u;

    /// Trivially-initialized TLS *pointer* (no dynamic TLS init in the
    /// per-operation path); the arena itself is created once per thread.
    static RevTape& active() {
        static thread_local RevTape* tape = nullptr;
        if (tape == nullptr) {
            tape = new RevTape();
        }
        return *tape;
    }

    RevTape() { m_nodes.resize(1024); }

    void clear() { m_size = 1; } // keep the sink at index 0
    void reserve(std::size_t n) {
        if (m_nodes.size() < n + 1) {
            m_nodes.resize(n + 1);
        }
    }
    std::size_t size() const { return m_size - 1; } // excluding sink

    std::uint32_t push(RevOp op, std::uint32_t lhs, std::uint32_t rhs, double value,
                       double lhsValue = 0.0, double rhsValue = 0.0) {
        if (m_size == m_nodes.size()) {
            m_nodes.resize(m_nodes.size() * 2 + 256);
        }
        m_nodes[m_size] = RevNode{value, 0.0, lhsValue, rhsValue, lhs, rhs, op};
        return static_cast<std::uint32_t>(m_size++);
    }

    /// Independent variable (what gradients are read from).
    std::uint32_t input(double value) { return push(RevOp::Input, kNone, kNone, value); }

    double value(std::uint32_t node) const { return m_nodes[node].value; }
    double adjoint(std::uint32_t node) const { return m_nodes[node].adjoint; }

    /// Reverse sweep seeded at `root` (adjoints start zero at push); walks
    /// real nodes only (index >= 1), node 0 absorbs constant parents.
    void reverse(std::uint32_t root, double seed = 1.0) {
        m_nodes[root].adjoint = seed;
        for (std::size_t i = m_size; i-- > 1;) {
            RevNode& n = m_nodes[i];
            const double a = n.adjoint;
            if (a == 0.0 || n.op == RevOp::Input) {
                continue;
            }
            switch (n.op) {
                case RevOp::Input:
                    break;
                case RevOp::Add:
                    m_nodes[n.lhs].adjoint += a;
                    m_nodes[n.rhs].adjoint += a;
                    break;
                case RevOp::Sub:
                    m_nodes[n.lhs].adjoint += a;
                    m_nodes[n.rhs].adjoint -= a;
                    break;
                case RevOp::Mul:
                    m_nodes[n.lhs].adjoint += a * n.rhsValue;
                    m_nodes[n.rhs].adjoint += a * n.lhsValue;
                    break;
                case RevOp::Div: {
                    const double rv = n.rhsValue;
                    m_nodes[n.lhs].adjoint += a / rv;
                    m_nodes[n.rhs].adjoint -= a * n.value / rv;
                    break;
                }
                case RevOp::Neg:
                    m_nodes[n.lhs].adjoint -= a;
                    break;
                case RevOp::Sqrt:
                    m_nodes[n.lhs].adjoint += a * 0.5 / n.value;
                    break;
                case RevOp::Exp:
                    m_nodes[n.lhs].adjoint += a * n.value;
                    break;
                case RevOp::Log:
                    m_nodes[n.lhs].adjoint += a / n.lhsValue;
                    break;
                case RevOp::Abs:
                    m_nodes[n.lhs].adjoint += a * (n.lhsValue >= 0.0 ? 1.0 : -1.0);
                    break;
                case RevOp::MinL:
                case RevOp::MaxL:
                    m_nodes[n.lhs].adjoint += a;
                    break;
                case RevOp::MinR:
                case RevOp::MaxR:
                    m_nodes[n.rhs].adjoint += a;
                    break;
            }
        }
    }

private:
    std::vector<RevNode> m_nodes;
    std::size_t m_size = 1; // sink lives at index 0
};

/// Value + tape node index; `kNone` = plain constant (no node).
struct RevScalar {
    double value = 0.0;
    std::uint32_t node = RevTape::kNone;

    RevScalar() = default;
    RevScalar(double v) : value(v) {} // implicit: constants never hit the tape
    RevScalar(double v, std::uint32_t n) : value(v), node(n) {}
};

namespace detail {

inline RevScalar revBinary(RevOp op, const RevScalar& a, const RevScalar& b, double value) {
    if (a.node == RevTape::kNone && b.node == RevTape::kNone) {
        return RevScalar(value);
    }
    return RevScalar(value, RevTape::active().push(op, a.node, b.node, value, a.value, b.value));
}

inline RevScalar revUnary(RevOp op, const RevScalar& a, double value) {
    if (a.node == RevTape::kNone) {
        return RevScalar(value);
    }
    return RevScalar(value, RevTape::active().push(op, a.node, RevTape::kNone, value, a.value));
}

} // namespace detail

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
inline RevScalar fabs(const RevScalar& x) {
    using std::fabs;
    return detail::revUnary(RevOp::Abs, x, fabs(x.value));
}
inline RevScalar abs(const RevScalar& x) {
    return fabs(x);
}

/// Branch selection shares the winner's node (derivative along the branch).
inline RevScalar fmin(const RevScalar& a, const RevScalar& b) {
    return a.value <= b.value ? a : b;
}
inline RevScalar fmax(const RevScalar& a, const RevScalar& b) {
    return a.value >= b.value ? a : b;
}

} // namespace quantape::mc

// Eigen scalar traits (same recipe as Tangent)
namespace Eigen {
template <>
struct NumTraits<quantape::mc::RevScalar> : NumTraits<double> {
    using Real = quantape::mc::RevScalar;
    using NonInteger = quantape::mc::RevScalar;
    using Nested = quantape::mc::RevScalar;
    using Literal = quantape::mc::RevScalar;
    enum {
        IsComplex = 0,
        IsInteger = 0,
        IsSigned = 1,
        RequireInitialization = 1,
        ReadCost = 1,
        AddCost = 4,
        MulCost = 8,
    };
};
} // namespace Eigen

#endif // QUANTAPE_MC_LEAN_REVERSE_H
