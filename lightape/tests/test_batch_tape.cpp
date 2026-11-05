#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include "lightape/batch/BatchTape.h"
#include "lightape/batch/VectorMath.h"
#include "lightape/ops/Probability.h"
#include <gtest/gtest.h>

namespace {

using lightape::BatchScalar;
using lightape::BatchTape;
using lightape::RevScalar;
using lightape::RevTape;

/// Raw IEEE-754 bit pattern: the comparison the bitwise contract needs (plain
/// double `==` would hide a signed-zero or NaN-payload difference).
std::uint64_t Bits(double v) {
    return std::bit_cast<std::uint64_t>(v);
}

/// Value gate of the batch-vs-scalar contract. The default (libm) build is
/// the bitwise reference; the SLEEF build (`LIGHTAPE_VECTOR_MATH_SLEEF`) is
/// differential only, so the same gates run with a documented relative
/// tolerance instead (plan §6.4; test_vector_math.cpp is the tight ULP gate).
void ExpectValue(double got, double want, const std::string& label) {
#ifdef LIGHTAPE_VECTOR_MATH_SLEEF
    EXPECT_NEAR(got, want, 1e-13 * std::max(1.0, std::fabs(want))) << label;
#else
    EXPECT_EQ(Bits(got), Bits(want)) << label;
#endif
}

/// Adjoint counterpart of `ExpectValue`; adjoints chain more partials, so the
/// documented SLEEF tolerance is one decade looser.
void ExpectAdjoint(double got, double want, const std::string& label) {
#ifdef LIGHTAPE_VECTOR_MATH_SLEEF
    EXPECT_NEAR(got, want, 1e-12 * std::max(1.0, std::fabs(want))) << label;
#else
    EXPECT_EQ(Bits(got), Bits(want)) << label;
#endif
}

/// `base + step * lane`, so W>1 lanes carry different values.
template <std::size_t W>
std::array<double, W> Lane(double base, double step) {
    std::array<double, W> v{};
    for (std::size_t l = 0; l < W; ++l) {
        v[l] = base + step * static_cast<double>(l);
    }
    return v;
}

/// One scalar-generic program over every supported opcode. The two branch ops
/// are lane-uniform by construction (`d <= d + 1` always picks `d`,
/// `e >= e - 1` always picks `e`), so the batch and the scalar tape record the
/// same statements.
template <typename S>
S Composite(const S& x1, const S& x2, const S& x3, const S& x4) {
    const S one(1.0);
    const S a = exp(x1) - log(x2);
    const S b = sqrt(abs(a)) + x3 * x4;
    const S c = -b / (x1 + 2.0);
    const S d = pow(b, c);
    const S e = fmin(d, d + one);
    const S f = fmax(e, e - one);
    const S g = pow(abs(x4), 1.5);
    return f * 0.5 + g * 0.25 + a / (x2 + one);
}

template <std::size_t W>
void CheckCompositeAgainstScalar() {
    const std::array<std::array<double, W>, 4> in = {Lane<W>(0.35, 0.05), Lane<W>(1.20, 0.08),
                                                     Lane<W>(0.80, 0.03), Lane<W>(1.60, 0.10)};
    std::array<double, W> seed{};
    for (std::size_t l = 0; l < W; ++l) {
        seed[l] = 0.7 + 0.1 * static_cast<double>(l);
    }

    BatchTape<W>& tape = BatchTape<W>::active();
    tape.clear();
    std::array<BatchScalar<W>, 4> x{};
    for (std::size_t i = 0; i < 4; ++i) {
        x[i] = tape.input(in[i]);
    }
    const BatchScalar<W> out = Composite(x[0], x[1], x[2], x[3]);
    const std::size_t batchNodes = tape.size();
    tape.reverse(out.node, seed);

    RevTape& scalar = RevTape::active();
    std::array<std::array<double, W>, 4> refAdj{};
    std::array<double, W> refValue{};
    for (std::size_t l = 0; l < W; ++l) {
        scalar.clear();
        std::array<RevScalar, 4> sx{};
        for (std::size_t i = 0; i < 4; ++i) {
            sx[i] = RevScalar(in[i][l], scalar.input(in[i][l]));
        }
        const RevScalar sout = Composite(sx[0], sx[1], sx[2], sx[3]);
        if (l == 0) {
            EXPECT_EQ(scalar.size(), batchNodes);
        }
        scalar.reverse(sout.node, seed[l]);
        refValue[l] = sout.value;
        for (std::size_t i = 0; i < 4; ++i) {
            refAdj[i][l] = scalar.adjoint(sx[i].node);
        }
    }

    for (std::size_t l = 0; l < W; ++l) {
        ExpectValue(out.value[l], refValue[l], "lane " + std::to_string(l));
        for (std::size_t i = 0; i < 4; ++i) {
            ExpectAdjoint(tape.adjoint(x[i].node)[l], refAdj[i][l],
                          "lane " + std::to_string(l) + " input " + std::to_string(i));
        }
    }
}

TEST(BatchTapeTest, CompositeMatchesScalarBitwiseForAllWidths) {
    CheckCompositeAgainstScalar<1>();
    CheckCompositeAgainstScalar<2>();
    CheckCompositeAgainstScalar<4>();
    CheckCompositeAgainstScalar<8>();
}

template <std::size_t W>
void CheckConstantCollapse() {
    BatchTape<W>& tape = BatchTape<W>::active();
    tape.clear();
    const std::array<double, W> av = Lane<W>(1.5, 0.75);
    const std::array<double, W> bv = Lane<W>(2.0, -0.25);
    const BatchScalar<W> a(av);
    const BatchScalar<W> b(bv);
    const BatchScalar<W> f = (a * b + a / b - exp(b)) / sqrt(fabs(a));
    EXPECT_EQ(f.node, BatchTape<W>::kNone);
    EXPECT_EQ(tape.size(), 0u);
    for (std::size_t l = 0; l < W; ++l) {
        const double expected =
            (av[l] * bv[l] + av[l] / bv[l] - std::exp(bv[l])) / std::sqrt(std::fabs(av[l]));
        ExpectValue(f.value[l], expected, "lane " + std::to_string(l));
    }

    // A constant operand of an active op stays node-less in its slot.
    const std::array<double, W> xv = Lane<W>(0.5, 0.25);
    const BatchScalar<W> x = tape.input(xv);
    const BatchScalar<W> g = x * 2.0;
    EXPECT_EQ(g.node, 2u);
    EXPECT_EQ(tape.size(), 2u);
    tape.reverse(g.node, 1.0);
    for (std::size_t l = 0; l < W; ++l) {
        ExpectAdjoint(tape.adjoint(x.node)[l], 2.0, "lane " + std::to_string(l));
    }
}

TEST(BatchTapeTest, ConstantsCollapseToNodeLess) {
    CheckConstantCollapse<1>();
    CheckConstantCollapse<2>();
    CheckConstantCollapse<4>();
    CheckConstantCollapse<8>();
}

template <std::size_t W>
void CheckClearReuse() {
    BatchTape<W>& tape = BatchTape<W>::active();
    tape.clear();
    const std::array<double, W> xv = Lane<W>(2.0, 0.5);
    const BatchScalar<W> x = tape.input(xv);
    const BatchScalar<W> f = x * x;
    EXPECT_EQ(tape.position(), 3u);
    EXPECT_EQ(tape.size(), 2u);
    tape.reverse(f.node, 1.0);
    for (std::size_t l = 0; l < W; ++l) {
        ExpectAdjoint(tape.adjoint(x.node)[l], xv[l] + xv[l], "lane " + std::to_string(l));
    }

    tape.clear();
    EXPECT_EQ(tape.size(), 0u);
    EXPECT_EQ(tape.position(), 1u);
    const std::array<double, W> yv = Lane<W>(4.0, 0.5);
    const BatchScalar<W> y = tape.input(yv);
    EXPECT_EQ(y.node, 1u);
    const BatchScalar<W> g = sqrt(y);
    tape.reverse(g.node, 1.0);
    for (std::size_t l = 0; l < W; ++l) {
        ExpectAdjoint(tape.adjoint(y.node)[l], 0.5 / std::sqrt(yv[l]), "lane " + std::to_string(l));
    }
}

TEST(BatchTapeTest, ClearResetsAndReusesSlots) {
    CheckClearReuse<1>();
    CheckClearReuse<2>();
    CheckClearReuse<4>();
    CheckClearReuse<8>();
}

TEST(BatchTapeTest, ChunkBoundarySweep) {
    constexpr std::size_t W = 2;
    BatchTape<W>& tape = BatchTape<W>::active();
    tape.clear();
    constexpr std::size_t n = BatchTape<W>::kChunkNodes + 17;
    std::vector<BatchScalar<W>> xs;
    xs.reserve(n);
    BatchScalar<W> acc(0.0);
    for (std::size_t i = 0; i < n; ++i) {
        const std::array<double, W> v = Lane<W>(0.5 + 0.001 * static_cast<double>(i % 1000), 0.25);
        xs.push_back(tape.input(v));
        acc = acc + xs[i] * 3.0;
    }
    EXPECT_GT(tape.size(), BatchTape<W>::kChunkNodes);
    tape.reverse(acc.node, 1.0);
    for (std::size_t i = 0; i < n; ++i) {
        for (std::size_t l = 0; l < W; ++l) {
            EXPECT_EQ(Bits(tape.adjoint(xs[i].node)[l]), Bits(3.0)) << "i=" << i << " lane " << l;
        }
    }
}

template <std::size_t W>
void CheckPowLaneConstant() {
    const std::array<double, W> bv = Lane<W>(1.25, 0.15);
    const std::array<double, W> ev = Lane<W>(0.5, 0.35);
    BatchTape<W>& tape = BatchTape<W>::active();
    tape.clear();
    const BatchScalar<W> x = tape.input(bv);
    const BatchScalar<W> f = pow(x, ev);
    tape.reverse(f.node, 1.0);

    RevTape& scalar = RevTape::active();
    for (std::size_t l = 0; l < W; ++l) {
        scalar.clear();
        const RevScalar sx(bv[l], scalar.input(bv[l]));
        const RevScalar sf = pow(sx, ev[l]);
        scalar.reverse(sf.node, 1.0);
        ExpectValue(f.value[l], sf.value, "lane " + std::to_string(l));
        ExpectAdjoint(tape.adjoint(x.node)[l], scalar.adjoint(sx.node),
                      "lane " + std::to_string(l));
    }
}

TEST(BatchTapeTest, PowLaneConstantMatchesScalarBitwise) {
    CheckPowLaneConstant<1>();
    CheckPowLaneConstant<2>();
    CheckPowLaneConstant<4>();
    CheckPowLaneConstant<8>();
}

TEST(BatchTapeTest, ComparisonsReturnPerLaneMasks) {
    constexpr std::size_t W = 4;
    const BatchScalar<W> a(Lane<W>(0.0, 1.0));  // 0 1 2 3
    const BatchScalar<W> b(Lane<W>(2.0, -1.0)); // 2 1 0 -1
    const std::array<bool, W> lt = a < b;
    const std::array<bool, W> le = a <= b;
    const std::array<bool, W> gt = a > b;
    const std::array<bool, W> ge = a >= b;
    const std::array<bool, W> eq = a == b;
    const std::array<bool, W> ne = a != b;
    const std::array<bool, W> ltExpected = {true, false, false, false};
    const std::array<bool, W> leExpected = {true, true, false, false};
    const std::array<bool, W> gtExpected = {false, false, true, true};
    const std::array<bool, W> geExpected = {false, true, true, true};
    const std::array<bool, W> eqExpected = {false, true, false, false};
    const std::array<bool, W> neExpected = {true, false, true, true};
    for (std::size_t l = 0; l < W; ++l) {
        EXPECT_EQ(lt[l], ltExpected[l]) << "lane " << l;
        EXPECT_EQ(le[l], leExpected[l]) << "lane " << l;
        EXPECT_EQ(gt[l], gtExpected[l]) << "lane " << l;
        EXPECT_EQ(ge[l], geExpected[l]) << "lane " << l;
        EXPECT_EQ(eq[l], eqExpected[l]) << "lane " << l;
        EXPECT_EQ(ne[l], neExpected[l]) << "lane " << l;
    }
}

/// Mixed fmin/fmax decisions: each selected value has a single consumer, so
/// the mask-carrying select node must still reproduce the scalar tape bitwise.
template <std::size_t W>
void CheckMixedMinMax() {
    const std::array<double, W> av = Lane<W>(0.0, 1.0);
    const std::array<double, W> bv = Lane<W>(static_cast<double>(W) - 1.0, -1.0);
    const std::array<double, W> cv = Lane<W>(1.5, 0.5);
    BatchTape<W>& tape = BatchTape<W>::active();
    tape.clear();
    const BatchScalar<W> a = tape.input(av);
    const BatchScalar<W> b = tape.input(bv);
    const BatchScalar<W> c = tape.input(cv);
    const BatchScalar<W> f = fmin(a, b) * c + fmax(a, b) * c;
    tape.reverse(f.node, 1.0);

    RevTape& scalar = RevTape::active();
    for (std::size_t l = 0; l < W; ++l) {
        scalar.clear();
        const RevScalar sa(av[l], scalar.input(av[l]));
        const RevScalar sb(bv[l], scalar.input(bv[l]));
        const RevScalar sc(cv[l], scalar.input(cv[l]));
        const RevScalar sf = fmin(sa, sb) * sc + fmax(sa, sb) * sc;
        scalar.reverse(sf.node, 1.0);
        EXPECT_EQ(Bits(f.value[l]), Bits(sf.value)) << "lane " << l;
        EXPECT_EQ(Bits(tape.adjoint(a.node)[l]), Bits(scalar.adjoint(sa.node))) << "lane " << l;
        EXPECT_EQ(Bits(tape.adjoint(b.node)[l]), Bits(scalar.adjoint(sb.node))) << "lane " << l;
        EXPECT_EQ(Bits(tape.adjoint(c.node)[l]), Bits(scalar.adjoint(sc.node))) << "lane " << l;
    }
}

TEST(BatchTapeTest, MixedMinMaxSelectRoutesLanes) {
    CheckMixedMinMax<1>();
    CheckMixedMinMax<2>();
    CheckMixedMinMax<4>();
    CheckMixedMinMax<8>();
}

TEST(BatchTapeTest, SoftmaxObjectiveMatchesScalarBitwise) {
    constexpr std::size_t W = 8;
    constexpr std::size_t N = 6;
    std::array<std::array<double, W>, N> v{};
    std::array<std::array<double, W>, N> w{};
    for (std::size_t i = 0; i < N; ++i) {
        v[i] = Lane<W>(0.2 * static_cast<double>(i) - 0.5, 0.03 * static_cast<double>(i + 1));
        w[i] = Lane<W>(1.0 - 0.2 * static_cast<double>(i), 0.05 * static_cast<double>(i));
    }

    BatchTape<W>& tape = BatchTape<W>::active();
    tape.clear();
    std::array<BatchScalar<W>, N> x{};
    for (std::size_t i = 0; i < N; ++i) {
        x[i] = tape.input(v[i]);
    }
    std::array<BatchScalar<W>, N> p{};
    for (std::size_t i = 0; i < N; ++i) {
        p[i] = exp(x[i]);
    }
    BatchScalar<W> denom(0.0);
    for (std::size_t i = 0; i < N; ++i) {
        denom = denom + p[i];
    }
    BatchScalar<W> f(0.0);
    for (std::size_t i = 0; i < N; ++i) {
        f = f + (p[i] / denom) * BatchScalar<W>(w[i]);
    }
    std::array<double, W> seed{};
    seed.fill(1.0);
    tape.reverse(f.node, seed);

    RevTape& scalar = RevTape::active();
    std::array<std::array<double, W>, N> refAdj{};
    std::array<double, W> refValue{};
    for (std::size_t l = 0; l < W; ++l) {
        scalar.clear();
        std::array<RevScalar, N> sx{};
        for (std::size_t i = 0; i < N; ++i) {
            sx[i] = RevScalar(v[i][l], scalar.input(v[i][l]));
        }
        std::array<RevScalar, N> sp{};
        for (std::size_t i = 0; i < N; ++i) {
            sp[i] = exp(sx[i]);
        }
        RevScalar denom2(0.0);
        for (std::size_t i = 0; i < N; ++i) {
            denom2 = denom2 + sp[i];
        }
        RevScalar f2(0.0);
        for (std::size_t i = 0; i < N; ++i) {
            f2 = f2 + (sp[i] / denom2) * RevScalar(w[i][l]);
        }
        scalar.reverse(f2.node, 1.0);
        refValue[l] = f2.value;
        for (std::size_t i = 0; i < N; ++i) {
            refAdj[i][l] = scalar.adjoint(sx[i].node);
        }
    }

    for (std::size_t l = 0; l < W; ++l) {
        ExpectValue(f.value[l], refValue[l], "lane " + std::to_string(l));
        for (std::size_t i = 0; i < N; ++i) {
            ExpectAdjoint(tape.adjoint(x[i].node)[l], refAdj[i][l],
                          "lane " + std::to_string(l) + " component " + std::to_string(i));
        }
    }
}

// ── math-backend seam and full op set ──

/// Record `program` once over W lanes and run it lane by lane on a scalar
/// `RevTape`; the root primal, every input adjoint and the recorded statement
/// count must match bitwise. The scalar program is the reference evaluation
/// order, so a mismatch is either a wrong partial or a structural divergence.
template <std::size_t W, typename Program>
void CheckProgramBits(const char* name, const std::vector<std::array<double, W>>& in,
                      const std::array<double, W>& seed, Program program) {
    SCOPED_TRACE(name);
    using Batch = BatchScalar<W>;
    BatchTape<W>& tape = BatchTape<W>::active();
    tape.clear();
    std::vector<Batch> x;
    x.reserve(in.size());
    for (const std::array<double, W>& v : in) {
        x.push_back(tape.input(v));
    }
    const Batch out = program(x);
    const std::size_t batchNodes = tape.size();
    tape.reverse(out.node, seed);

    RevTape& scalar = RevTape::active();
    std::vector<std::array<double, W>> refAdj(in.size());
    std::array<double, W> refValue{};
    for (std::size_t l = 0; l < W; ++l) {
        scalar.clear();
        std::vector<RevScalar> sx;
        sx.reserve(in.size());
        for (const std::array<double, W>& v : in) {
            sx.emplace_back(v[l], scalar.input(v[l]));
        }
        const RevScalar sout = program(sx);
        if (l == 0) {
            EXPECT_EQ(scalar.size(), batchNodes) << "recorded statement count differs";
        }
        scalar.reverse(sout.node, seed[l]);
        refValue[l] = sout.value;
        for (std::size_t i = 0; i < in.size(); ++i) {
            refAdj[i][l] = scalar.adjoint(sx[i].node);
        }
    }

    for (std::size_t l = 0; l < W; ++l) {
        ExpectValue(out.value[l], refValue[l], "value lane " + std::to_string(l));
        for (std::size_t i = 0; i < in.size(); ++i) {
            ExpectAdjoint(tape.adjoint(x[i].node)[l], refAdj[i][l],
                          "adjoint lane " + std::to_string(l) + " input " + std::to_string(i));
        }
    }
}

template <std::size_t W, typename Program>
void CheckUnaryBits(const char* name, double base, double step, Program program) {
    CheckProgramBits<W>(name, {Lane<W>(base, step)}, Lane<W>(0.7, 0.1), program);
}

template <std::size_t W, typename Program>
void CheckBinaryBits(const char* name, double baseA, double stepA, double baseB, double stepB,
                     Program program) {
    CheckProgramBits<W>(name, {Lane<W>(baseA, stepA), Lane<W>(baseB, stepB)}, Lane<W>(0.7, 0.1),
                        program);
}

template <typename Program>
void CheckUnaryAllWidths(const char* name, double base, double step, Program program) {
    CheckUnaryBits<1>(name, base, step, program);
    CheckUnaryBits<2>(name, base, step, program);
    CheckUnaryBits<4>(name, base, step, program);
    CheckUnaryBits<8>(name, base, step, program);
}

template <typename Program>
void CheckBinaryAllWidths(const char* name, double baseA, double stepA, double baseB, double stepB,
                          Program program) {
    CheckBinaryBits<1>(name, baseA, stepA, baseB, stepB, program);
    CheckBinaryBits<2>(name, baseA, stepA, baseB, stepB, program);
    CheckBinaryBits<4>(name, baseA, stepA, baseB, stepB, program);
    CheckBinaryBits<8>(name, baseA, stepA, baseB, stepB, program);
}

TEST(BatchTapeTest, ExpLogOpsMatchScalarBitwise) {
    CheckUnaryAllWidths("exp", 0.3, 0.04, [](const auto& x) { return lightape::exp(x[0]); });
    CheckUnaryAllWidths("log", 0.5, 0.05, [](const auto& x) { return lightape::log(x[0]); });
    CheckUnaryAllWidths("expm1", 0.3, 0.04, [](const auto& x) { return lightape::expm1(x[0]); });
    CheckUnaryAllWidths("log1p", 0.3, 0.04, [](const auto& x) { return lightape::log1p(x[0]); });
    CheckUnaryAllWidths("exp2", 0.2, 0.03, [](const auto& x) { return lightape::exp2(x[0]); });
    CheckUnaryAllWidths("log2", 0.5, 0.05, [](const auto& x) { return lightape::log2(x[0]); });
    CheckUnaryAllWidths("log10", 0.5, 0.05, [](const auto& x) { return lightape::log10(x[0]); });
}

TEST(BatchTapeTest, TrigOpsMatchScalarBitwise) {
    CheckUnaryAllWidths("sin", 0.2, 0.05, [](const auto& x) { return lightape::sin(x[0]); });
    CheckUnaryAllWidths("cos", 0.2, 0.05, [](const auto& x) { return lightape::cos(x[0]); });
    CheckUnaryAllWidths("tan", 0.1, 0.03, [](const auto& x) { return lightape::tan(x[0]); });
    CheckUnaryAllWidths("asin", -0.45, 0.05, [](const auto& x) { return lightape::asin(x[0]); });
    CheckUnaryAllWidths("acos", -0.45, 0.05, [](const auto& x) { return lightape::acos(x[0]); });
    CheckUnaryAllWidths("atan", 0.2, 0.05, [](const auto& x) { return lightape::atan(x[0]); });
}

TEST(BatchTapeTest, HyperbolicOpsMatchScalarBitwise) {
    CheckUnaryAllWidths("sinh", 0.2, 0.05, [](const auto& x) { return lightape::sinh(x[0]); });
    CheckUnaryAllWidths("cosh", 0.2, 0.05, [](const auto& x) { return lightape::cosh(x[0]); });
    CheckUnaryAllWidths("tanh", 0.2, 0.05, [](const auto& x) { return lightape::tanh(x[0]); });
}

TEST(BatchTapeTest, ErrorAndRootOpsMatchScalarBitwise) {
    CheckUnaryAllWidths("sqrt", 0.6, 0.05, [](const auto& x) { return lightape::sqrt(x[0]); });
    CheckUnaryAllWidths("cbrt", 0.5, 0.05, [](const auto& x) { return lightape::cbrt(x[0]); });
    CheckUnaryAllWidths("erf", 0.2, 0.05, [](const auto& x) { return lightape::erf(x[0]); });
    CheckUnaryAllWidths("erfc", 0.2, 0.05, [](const auto& x) { return lightape::erfc(x[0]); });
}

TEST(BatchTapeTest, BinaryMathOpsMatchScalarBitwise) {
    CheckBinaryAllWidths("pow", 1.1, 0.03, 0.7, 0.04,
                         [](const auto& x) { return lightape::pow(x[0], x[1]); });
    CheckBinaryAllWidths("hypot", 1.1, 0.03, 0.6, 0.05,
                         [](const auto& x) { return lightape::hypot(x[0], x[1]); });
    CheckBinaryAllWidths("atan2", 0.3, 0.04, 1.1, 0.03,
                         [](const auto& x) { return lightape::atan2(x[0], x[1]); });
}

TEST(BatchTapeTest, SinCosCompositeMatchesScalarBitwise) {
    CheckUnaryAllWidths("sincos", 0.2, 0.05, [](const auto& x) {
        auto s = x[0];
        auto c = x[0];
        lightape::sincos(x[0], s, c);
        return s * s + s * c + c;
    });
}

TEST(BatchTapeTest, NormCompositesMatchScalarBitwise) {
    CheckUnaryAllWidths("normpdf", -0.8, 0.2,
                        [](const auto& x) { return lightape::normpdf(x[0]); });
    CheckUnaryAllWidths("normcdf", -0.8, 0.2,
                        [](const auto& x) { return lightape::normcdf(x[0]); });
}

/// Composite from the plan's fused-composite gate: a normcdf/erf/pow mix over
/// exp, checked bitwise against the same program on the scalar `RevScalar` tape.
template <std::size_t W>
void CheckMixedMathComposite() {
    CheckProgramBits<W>("normcdf(exp(x)) + erf(x)/pow(x, 2.5)", {Lane<W>(0.25, 0.04)},
                        Lane<W>(0.7, 0.1), [](const auto& x) {
                            return lightape::normcdf(lightape::exp(x[0])) +
                                   lightape::erf(x[0]) / lightape::pow(x[0], 2.5);
                        });
}

TEST(BatchTapeTest, MixedMathCompositeMatchesScalarBitwise) {
    CheckMixedMathComposite<1>();
    CheckMixedMathComposite<2>();
    CheckMixedMathComposite<4>();
    CheckMixedMathComposite<8>();
}

/// The fused normal kernels must reproduce `lightape::normpdf`/`normcdf` on
/// plain doubles bitwise under the libm build (they are the same per-lane
/// expressions) and within the documented SLEEF tolerance otherwise.
template <std::size_t W>
void CheckVectorMathNormKernels() {
    const std::array<double, W> x = Lane<W>(-2.0, 0.5);
    std::array<double, W> pdf{};
    std::array<double, W> cdf{};
    lightape::batch::VectorMath<W>::vNormPdf(x.data(), pdf.data());
    lightape::batch::VectorMath<W>::vNormCdf(x.data(), cdf.data());
    for (std::size_t l = 0; l < W; ++l) {
        ExpectValue(pdf[l], lightape::normpdf(x[l]), "vNormPdf lane " + std::to_string(l));
        ExpectValue(cdf[l], lightape::normcdf(x[l]), "vNormCdf lane " + std::to_string(l));
    }
}

TEST(BatchTapeTest, VectorMathNormKernelsMatchReference) {
    CheckVectorMathNormKernels<1>();
    CheckVectorMathNormKernels<2>();
    CheckVectorMathNormKernels<4>();
    CheckVectorMathNormKernels<8>();
}

TEST(BatchTapeTest, VectorMathBackendTagMatchesBuild) {
#ifdef LIGHTAPE_VECTOR_MATH_SLEEF
    EXPECT_EQ(lightape::batch::VectorMath<1>::backend, lightape::batch::MathBackend::Sleef);
    EXPECT_EQ(lightape::batch::VectorMath<8>::backend, lightape::batch::MathBackend::Sleef);
#else
    EXPECT_EQ(lightape::batch::VectorMath<1>::backend, lightape::batch::MathBackend::LibmScalar);
    EXPECT_EQ(lightape::batch::VectorMath<8>::backend, lightape::batch::MathBackend::LibmScalar);
#endif
}

} // namespace
