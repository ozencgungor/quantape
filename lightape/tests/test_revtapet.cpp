#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include "lightape/rev/RevTapeT.h"
#include <gtest/gtest.h>

namespace {

using lightape::RevScalar;
using lightape::RevScalarT;
using lightape::RevTape;
using lightape::RevTapeT;

/// Raw IEEE-754 bit pattern: the comparison the bitwise contract needs (plain
/// double `==` would hide a signed-zero or NaN-payload difference).
std::uint64_t Bits(double v) {
    return std::bit_cast<std::uint64_t>(v);
}

/// One scalar-generic program over the storage scaffold's op set (arithmetic
/// plus sqrt/exp/log/pow/fabs and the two branch ops). Written with `using
/// std::...` + ADL so the same body instantiates for `RevScalar` (the
/// reference tape) and `RevScalarT<double>`. The branches are value-uniform.
template <typename S>
S Composite(const S& x1, const S& x2, const S& x3, const S& x4) {
    using std::exp;
    using std::fabs;
    using std::fmax;
    using std::fmin;
    using std::log;
    using std::pow;
    using std::sqrt;
    const S one(1.0);
    const S a = exp(x1) - log(x2);
    const S b = sqrt(fabs(a)) + x3 * x4;
    const S c = -b / (x1 + 2.0);
    const S d = pow(b, c);
    const S e = fmin(d, d + one);
    const S f = fmax(e, e - one);
    const S g = pow(fabs(x4), 1.5);
    return f * 0.5 + g * 0.25 + a / (x2 + one);
}

/// Record `Composite` once on `RevTapeT<double>`, reverse it and return the
/// root value, the four input adjoints and the recorded node count.
struct TapeTResult {
    double value = 0.0;
    std::array<double, 4> adjoint{};
    std::size_t nodes = 0;
};

TapeTResult RunCompositeTapeT() {
    const double v[4] = {0.35, 1.20, 0.80, 1.60};
    RevTapeT<double>& tape = RevTapeT<double>::active();
    tape.clear();
    RevScalarT<double> x[4] = {
        RevScalarT<double>(v[0], tape.input(v[0])), RevScalarT<double>(v[1], tape.input(v[1])),
        RevScalarT<double>(v[2], tape.input(v[2])), RevScalarT<double>(v[3], tape.input(v[3]))};
    const RevScalarT<double> out = Composite(x[0], x[1], x[2], x[3]);
    TapeTResult result;
    result.value = out.value;
    result.nodes = tape.size();
    tape.reverse(out.node, 1.0);
    for (std::size_t i = 0; i < 4; ++i) {
        result.adjoint[i] = tape.adjoint(x[i].node);
    }
    return result;
}

/// `RunCompositeTapeT` on the reference tape, in the same statement order.
TapeTResult RunCompositeRevTape() {
    const double v[4] = {0.35, 1.20, 0.80, 1.60};
    RevTape& tape = RevTape::active();
    tape.clear();
    RevScalar x[4] = {RevScalar(v[0], tape.input(v[0])), RevScalar(v[1], tape.input(v[1])),
                      RevScalar(v[2], tape.input(v[2])), RevScalar(v[3], tape.input(v[3]))};
    const RevScalar out = Composite(x[0], x[1], x[2], x[3]);
    TapeTResult result;
    result.value = out.value;
    result.nodes = tape.size();
    tape.reverse(out.node, 1.0);
    for (std::size_t i = 0; i < 4; ++i) {
        result.adjoint[i] = tape.adjoint(x[i].node);
    }
    return result;
}

TEST(RevTapeTTest, DoubleCompositeMatchesRevTapeBitwise) {
    const TapeTResult got = RunCompositeTapeT();
    const TapeTResult want = RunCompositeRevTape();
    EXPECT_EQ(got.nodes, want.nodes) << "recorded statement count differs";
    EXPECT_EQ(Bits(got.value), Bits(want.value)) << "root value";
    for (std::size_t i = 0; i < 4; ++i) {
        EXPECT_EQ(Bits(got.adjoint[i]), Bits(want.adjoint[i])) << "input " << i;
    }
}

TEST(RevTapeTTest, NodeAndScalarLayoutsMatchThePlan) {
    static_assert(sizeof(lightape::RevNodeT<double>) == 48, "48-byte double node");
    static_assert(sizeof(RevScalarT<double>) == 16, "16-byte scalar wrapper");
    SUCCEED();
}

TEST(RevTapeTTest, ChunkCrossingMatchesRevTapeBitwise) {
    constexpr std::size_t n = RevTapeT<double>::kChunkNodes + 17;

    RevTapeT<double>& tape = RevTapeT<double>::active();
    tape.clear();
    std::vector<RevScalarT<double>> xs;
    xs.reserve(n);
    RevScalarT<double> acc(0.0);
    for (std::size_t i = 0; i < n; ++i) {
        const double v = 0.5 + 0.001 * static_cast<double>(i % 1000);
        xs.emplace_back(v, tape.input(v));
        acc = acc + xs[i] * 3.0;
    }
    EXPECT_GT(tape.size(), RevTapeT<double>::kChunkNodes);
    tape.reverse(acc.node, 1.0);

    RevTape& ref = RevTape::active();
    ref.clear();
    std::vector<RevScalar> rxs;
    rxs.reserve(n);
    RevScalar racc(0.0);
    for (std::size_t i = 0; i < n; ++i) {
        const double v = 0.5 + 0.001 * static_cast<double>(i % 1000);
        rxs.emplace_back(v, ref.input(v));
        racc = racc + rxs[i] * 3.0;
    }
    EXPECT_EQ(ref.size(), tape.size());
    ref.reverse(racc.node, 1.0);

    EXPECT_EQ(Bits(acc.value), Bits(racc.value)) << "accumulator value";
    for (std::size_t i = 0; i < n; ++i) {
        EXPECT_EQ(Bits(tape.adjoint(xs[i].node)), Bits(ref.adjoint(rxs[i].node))) << "i=" << i;
    }
}

TEST(RevTapeTTest, ResetToAndClearTruncateAndReuse) {
    RevTapeT<double>& tape = RevTapeT<double>::active();
    tape.clear();
    RevScalarT<double> x(2.0, tape.input(2.0));
    RevScalarT<double> y(3.0, tape.input(3.0));
    const std::size_t pos = tape.position();
    RevScalarT<double> acc = x;
    for (std::size_t i = 0; i < RevTapeT<double>::kChunkNodes + 5; ++i) {
        acc = acc + 1.0;
    }
    EXPECT_GT(tape.size(), RevTapeT<double>::kChunkNodes);
    tape.resetAdjoints();
    tape.resetTo(pos);
    EXPECT_EQ(tape.position(), pos);
    EXPECT_EQ(tape.size(), pos - 1);

    RevScalarT<double> f = x * y;
    tape.reverse(f.node, 1.0);
    EXPECT_EQ(Bits(tape.adjoint(x.node)), Bits(3.0));
    EXPECT_EQ(Bits(tape.adjoint(y.node)), Bits(2.0));

    tape.clear();
    EXPECT_EQ(tape.size(), 0u);
    RevScalarT<double> z(4.0, tape.input(4.0));
    EXPECT_EQ(z.node, 1u);
    RevScalarT<double> g = sqrt(z);
    tape.reverse(g.node, 1.0);
    EXPECT_EQ(Bits(tape.adjoint(z.node)), Bits(0.25));
}

TEST(RevTapeTTest, RepeatedRunsAreBitIdentical) {
    const TapeTResult first = RunCompositeTapeT();
    for (int run = 1; run < 3; ++run) {
        const TapeTResult again = RunCompositeTapeT();
        EXPECT_EQ(again.nodes, first.nodes) << "run " << run;
        EXPECT_EQ(Bits(again.value), Bits(first.value)) << "run " << run;
        for (std::size_t i = 0; i < 4; ++i) {
            EXPECT_EQ(Bits(again.adjoint[i]), Bits(first.adjoint[i]))
                << "run " << run << " input " << i;
        }
    }

    // Re-sweeping the same recording (resetAdjoints + reverse) is also stable.
    RevTapeT<double>& tape = RevTapeT<double>::active();
    RevScalarT<double> x(0.75, tape.input(0.75));
    RevScalarT<double> out = exp(x) * 3.0;
    tape.reverse(out.node, 1.0);
    const std::uint64_t adj = Bits(tape.adjoint(x.node));
    tape.resetAdjoints();
    tape.reverse(out.node, 1.0);
    EXPECT_EQ(Bits(tape.adjoint(x.node)), adj);
}

// ── synthetic non-double payload: proves the storage/sweep is T-generic ──

/// A trivially-copyable first-order dual: value `v` plus derivative `d`.
/// The tape only ever calls the documented payload protocol (arithmetic,
/// ADL math, primal comparisons), so this is a complete stand-in for the
/// eventual nested `RevScalar` payload.
struct Dual2 {
    double v = 0.0;
    double d = 0.0;

    Dual2() = default;
    Dual2(double value) : v(value) {} // implicit constant (zero derivative)
    Dual2(double value, double deriv) : v(value), d(deriv) {}

    /// Full-payload zero test used by the sweep's zero-skip; note this is not
    /// the primal `operator==` below.
    bool isZero() const { return v == 0.0 && d == 0.0; }
};

inline Dual2 operator+(const Dual2& a, const Dual2& b) {
    return Dual2(a.v + b.v, a.d + b.d);
}
inline Dual2 operator-(const Dual2& a, const Dual2& b) {
    return Dual2(a.v - b.v, a.d - b.d);
}
inline Dual2 operator*(const Dual2& a, const Dual2& b) {
    return Dual2(a.v * b.v, a.d * b.v + a.v * b.d);
}
inline Dual2 operator/(const Dual2& a, const Dual2& b) {
    return Dual2(a.v / b.v, (a.d * b.v - a.v * b.d) / (b.v * b.v));
}
[[maybe_unused]] inline Dual2 operator-(const Dual2& a) {
    return Dual2(-a.v, -a.d);
}

inline Dual2 sqrt(const Dual2& a) {
    const double s = std::sqrt(a.v);
    return Dual2(s, a.d * 0.5 / s);
}
inline Dual2 exp(const Dual2& a) {
    const double e = std::exp(a.v);
    return Dual2(e, a.d * e);
}
inline Dual2 log(const Dual2& a) {
    return Dual2(std::log(a.v), a.d / a.v);
}
inline Dual2 fabs(const Dual2& a) {
    return a.v >= 0.0 ? a : Dual2(-a.v, -a.d);
}
inline Dual2 pow(const Dual2& a, const Dual2& b) {
    const double p = std::pow(a.v, b.v);
    const double da = b.v * std::pow(a.v, b.v - 1.0);
    const double db = p * std::log(a.v);
    return Dual2(p, da * a.d + db * b.d);
}
// ADL min/max complete the documented payload surface; the tape's branch ops
// themselves only need the primal comparisons below (`fmin`/`fmax` share the
// winner's node there), so these stay unused inside the test.
[[maybe_unused]] inline Dual2 fmin(const Dual2& a, const Dual2& b) {
    return a.v <= b.v ? a : b;
}
[[maybe_unused]] inline Dual2 fmax(const Dual2& a, const Dual2& b) {
    return a.v >= b.v ? a : b;
}

// Comparisons on the primal component (branch decisions only).
[[maybe_unused]] inline bool operator<(const Dual2& a, const Dual2& b) {
    return a.v < b.v;
}
inline bool operator<=(const Dual2& a, const Dual2& b) {
    return a.v <= b.v;
}
[[maybe_unused]] inline bool operator>(const Dual2& a, const Dual2& b) {
    return a.v > b.v;
}
inline bool operator>=(const Dual2& a, const Dual2& b) {
    return a.v >= b.v;
}
[[maybe_unused]] inline bool operator==(const Dual2& a, const Dual2& b) {
    return a.v == b.v;
}
[[maybe_unused]] inline bool operator!=(const Dual2& a, const Dual2& b) {
    return a.v != b.v;
}

/// `T = RevScalar` (the actual rev-over-rev scalar) is deliberately **not**
/// instantiated here. The storage and sweep below are already generic, but
/// decision #8 parks the rev-rev chain behind the first opaque/large-n
/// driver, and enabling it needs the public ops layer (`ops/`) templated over
/// `RevScalarT<T>` plus a componentwise zero predicate on the nested scalar;
/// `Dual2` exercises exactly those two contracts without pulling the whole
/// ops surface into the scaffold.
template <typename S>
S DualProgram(const S& x0, const S& x1, const S& x2, const S& x3) {
    using std::exp;
    using std::fabs;
    using std::fmax;
    using std::fmin;
    using std::log;
    using std::pow;
    using std::sqrt;
    const S a = exp(x0 * x1);
    const S b = sqrt(fabs(x2)) + x3 * x3;
    const S c = log(x1) + a / b;
    const S d = pow(x3, 1.5);
    const S e = fmin(c, c + 1.0);
    const S f = fmax(e, e - 1.0);
    return f * 0.5 + d + a * c;
}

TEST(RevTapeTTest, Dual2PayloadSweepsValueAndDerivative) {
    constexpr std::size_t n = 4;
    const double xv[n] = {0.7, 1.3, 0.9, 0.6};
    const double dir[n] = {1.0, -0.5, 0.25, 2.0};

    RevTapeT<Dual2>& tape = RevTapeT<Dual2>::active();
    tape.clear();
    RevScalarT<Dual2> x[n];
    for (std::size_t i = 0; i < n; ++i) {
        const Dual2 in(xv[i], dir[i]);
        x[i] = RevScalarT<Dual2>(in, tape.input(in));
    }
    const RevScalarT<Dual2> out = DualProgram(x[0], x[1], x[2], x[3]);
    tape.reverse(out.node, Dual2(1.0)); // seed (value adjoint 1, tangent 0)

    // Reference: the concrete tape gives the gradient exactly and the exact
    // closed-form HVP through its replay engine.
    RevTape& ref = RevTape::active();
    ref.clear();
    RevScalar rx[n];
    std::uint32_t slots[n];
    for (std::size_t i = 0; i < n; ++i) {
        slots[i] = ref.input(xv[i]);
        rx[i] = RevScalar(xv[i], slots[i]);
    }
    const RevScalar rout = DualProgram(rx[0], rx[1], rx[2], rx[3]);
    ref.reverse(rout.node, 1.0);
    double grad[n];
    for (std::size_t i = 0; i < n; ++i) {
        grad[i] = ref.adjoint(slots[i]);
    }
    double hv[n];
    ASSERT_TRUE(ref.hessianVector(rout.node, slots, n, dir, hv));

    // Primal and the value component of every adjoint are the same double
    // arithmetic as the reference tape, so they gate bitwise; the derivative
    // component is the H * direction pair and gates against the exact replay
    // HVP (the dual chain rule and the closed-form local rules may round
    // differently, so this one is tolerance-gated).
    EXPECT_EQ(Bits(out.value.v), Bits(rout.value)) << "root value";
    double dot = 0.0;
    for (std::size_t i = 0; i < n; ++i) {
        dot += grad[i] * dir[i];
    }
    EXPECT_NEAR(out.value.d, dot, 1e-12 * std::max(1.0, std::fabs(dot)));
    for (std::size_t i = 0; i < n; ++i) {
        const Dual2 adj = tape.adjoint(x[i].node);
        EXPECT_EQ(Bits(adj.v), Bits(grad[i])) << "input " << i << " value adjoint";
        EXPECT_NEAR(adj.d, hv[i], 1e-12 * std::max(1.0, std::fabs(hv[i])))
            << "input " << i << " derivative adjoint";
    }
}

TEST(RevTapeTTest, Dual2NodeLayoutIsEightyBytes) {
    // 4 x 16 B payload + 8 B parent indices + opcode/padding: the 80 B
    // rev-over-rev node of decision #8, shown here on the synthetic payload.
    static_assert(sizeof(lightape::RevNodeT<Dual2>) == 80, "80-byte nested node");
    SUCCEED();
}

} // namespace
