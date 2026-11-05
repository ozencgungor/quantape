#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <string>
#include <type_traits>

#include "lightape/rev/RevScalarN.h"
#include <gtest/gtest.h>

namespace {

using lightape::RevScalar;
using lightape::RevScalarN;
using lightape::RevTape;
using lightape::RevTapeN;

/// The aliases are the SoA batch engine, nothing else (decision #9/#15).
static_assert(std::is_same_v<RevScalarN<8>, lightape::BatchScalar<8>>,
              "RevScalarN<N> must alias BatchScalar<N>");
static_assert(std::is_same_v<RevTapeN<8>, lightape::BatchTape<8>>,
              "RevTapeN<N> must alias BatchTape<N>");

/// Raw IEEE-754 bit pattern: the bitwise contract comparison (plain double
/// `==` would hide a signed-zero or NaN-payload difference).
std::uint64_t Bits(double v) {
    return std::bit_cast<std::uint64_t>(v);
}

/// Value gate mirroring `test_batch_tape.cpp`: bitwise on the libm reference
/// build, the documented relative tolerance under the SLEEF backend.
void ExpectValue(double got, double want, const std::string& label) {
#ifdef LIGHTAPE_VECTOR_MATH_SLEEF
    EXPECT_NEAR(got, want, 1e-13 * std::max(1.0, std::fabs(want))) << label;
#else
    EXPECT_EQ(Bits(got), Bits(want)) << label;
#endif
}

/// Adjoint counterpart of `ExpectValue` (one decade looser under SLEEF).
void ExpectAdjoint(double got, double want, const std::string& label) {
#ifdef LIGHTAPE_VECTOR_MATH_SLEEF
    EXPECT_NEAR(got, want, 1e-12 * std::max(1.0, std::fabs(want))) << label;
#else
    EXPECT_EQ(Bits(got), Bits(want)) << label;
#endif
}

/// `base + step * lane`, so all W lanes carry different values.
template <std::size_t W>
std::array<double, W> Lane(double base, double step) {
    std::array<double, W> v{};
    for (std::size_t l = 0; l < W; ++l) {
        v[l] = base + step * static_cast<double>(l);
    }
    return v;
}

/// One lane-generic program over the supported op set (+ - * / Neg Sqrt Exp
/// Log Abs Pow), shared by the batched and the scalar reference evaluation.
template <typename S>
S ShortProgram(const S& x0, const S& x1, const S& x2, const S& x3, const S& x4, const S& x5,
               const S& x6) {
    using std::exp;
    using std::fabs;
    using std::log;
    using std::pow;
    using std::sqrt;
    return exp(x0 * x1) + sqrt(fabs(x2)) - x3 / x4 + pow(x5, 2.5) + log(x6) + x6 * x6;
}

/// 8-lane gradient through `RevTapeN<8>` (= `BatchTape<8>`), gated per lane
/// against the same program on the scalar `RevTape` (the `BatchTape` bitwise
/// contract; lanes never share a reduction, so each lane reproduces the
/// scalar tape of that lane's values).
TEST(RevScalarNTest, EightLaneGradientMatchesScalarBitwise) {
    constexpr std::size_t W = 8;
    const std::array<std::array<double, W>, 7> in = {
        Lane<W>(0.70, 0.05), Lane<W>(1.30, 0.04), Lane<W>(0.90, 0.03), Lane<W>(0.60, 0.02),
        Lane<W>(1.10, 0.06), Lane<W>(1.40, 0.05), Lane<W>(0.80, 0.07)};

    RevTapeN<W>& tape = RevTapeN<W>::active();
    tape.clear();
    std::array<RevScalarN<W>, 7> x{};
    for (std::size_t i = 0; i < 7; ++i) {
        x[i] = tape.input(in[i]);
    }
    const RevScalarN<W> out = ShortProgram(x[0], x[1], x[2], x[3], x[4], x[5], x[6]);
    const std::size_t batchNodes = tape.size();
    tape.reverse(out.node, 1.0);

    RevTape& scalar = RevTape::active();
    std::array<std::array<double, W>, 7> refAdj{};
    std::array<double, W> refValue{};
    for (std::size_t l = 0; l < W; ++l) {
        scalar.clear();
        std::array<RevScalar, 7> sx{};
        for (std::size_t i = 0; i < 7; ++i) {
            sx[i] = RevScalar(in[i][l], scalar.input(in[i][l]));
        }
        const RevScalar sout = ShortProgram(sx[0], sx[1], sx[2], sx[3], sx[4], sx[5], sx[6]);
        if (l == 0) {
            EXPECT_EQ(scalar.size(), batchNodes) << "recorded statement count differs";
        }
        scalar.reverse(sout.node, 1.0);
        refValue[l] = sout.value;
        for (std::size_t i = 0; i < 7; ++i) {
            refAdj[i][l] = scalar.adjoint(sx[i].node);
        }
    }

    for (std::size_t l = 0; l < W; ++l) {
        ExpectValue(out.value[l], refValue[l], "lane " + std::to_string(l));
        for (std::size_t i = 0; i < 7; ++i) {
            ExpectAdjoint(tape.adjoint(x[i].node)[l], refAdj[i][l],
                          "lane " + std::to_string(l) + " input " + std::to_string(i));
        }
    }
}

/// The per-lane readout is also available straight off the tape: values and
/// adjoints by lane index, and a per-lane seed vector through `reverse`.
TEST(RevScalarNTest, PerLaneReadoutAndSeedVector) {
    constexpr std::size_t W = 8;
    const std::array<double, W> xv = Lane<W>(2.0, 0.25);
    std::array<double, W> seed{};
    for (std::size_t l = 0; l < W; ++l) {
        seed[l] = 0.5 + 0.1 * static_cast<double>(l);
    }

    RevTapeN<W>& tape = RevTapeN<W>::active();
    tape.clear();
    const RevScalarN<W> x = tape.input(xv);
    const RevScalarN<W> f = x * x;
    tape.reverse(f.node, seed);
    for (std::size_t l = 0; l < W; ++l) {
        ExpectValue(tape.value(f.node)[l], xv[l] * xv[l], "lane " + std::to_string(l));
        ExpectAdjoint(tape.adjoint(x.node)[l], seed[l] * xv[l] + seed[l] * xv[l],
                      "lane " + std::to_string(l));
    }
}

} // namespace
