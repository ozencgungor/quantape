#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <type_traits>

#include "lightape/ops/ElementaryExtras.h"
#include <gtest/gtest.h>

namespace {

using lightape::RevScalar;
using lightape::RevTape;
using lightape::Tangent;

using D1 = Tangent<double, 1>;
using D2 = Tangent<double, 2>;
using I2 = Tangent<double, 2>;
using O2 = Tangent<I2, 2>;

// High-precision references (mpmath, 50 dps) at the test points.
constexpr double kCbrtVal = 0.8879040017426008;     // 0.7^(1/3)
constexpr double kCbrtFp = 0.42281142940123845;     // 1/(3 z^2)
constexpr double kCbrtFpp = -0.40267755181070325;   // -2/(9 z^5)
constexpr double kExp2Val = 2.4622888266898326;     // 2^1.3
constexpr double kExp2Fp = 1.7067285579443132;      // ln2 2^x
constexpr double kExp2Fpp = 1.183014087920242;      // (ln2)^2 2^x
constexpr double kLog2Val = 1.584962500721156;      // log2 3
constexpr double kLog2Fp = 0.4808983469629878;      // 1/(x ln2)
constexpr double kLog2Fpp = -0.1602994489876626;    // -1/(x^2 ln2)
constexpr double kLog10Val = 0.8450980400142568;    // log10 7
constexpr double kLog10Fp = 0.06204206884332169;    // 1/(x ln10)
constexpr double kLog10Fpp = -0.008863152691903098; // -1/(x^2 ln10)

constexpr double kLog1mexpVal = -0.6863410028083851; // log(1 - e^-0.7)
constexpr double kLog1mexpFp = -0.9864338636344633;
constexpr double kLog1mexpFpp = -1.9594856309592783;
constexpr double kLogDiffVal = 1.3193184318233655; // log(e^1.5 - e^-0.3)
constexpr double kLogDiffFa = 1.198033626515006;
constexpr double kLogDiffFb = -0.19803362651500592;
constexpr double kLogDiffFaa = -0.23725094374569078;
constexpr double kLogDiffFab = 0.23725094374569078;
constexpr double kLogDiffFbb = -0.23725094374569078;
constexpr double kMultLogVal = 2.0794415416798357; // 3 ln 2
constexpr double kLn2 = 0.69314718055994530942;

// invErfc references at y in {1e-12, 0.025, 0.5, 0.9, 0.999}; fp = f'(y),
// fpp = 2 x fp^2.
constexpr double kInvErfcY[5] = {1e-12, 0.025, 0.5, 0.9, 0.999};
constexpr double kInvErfcX[5] = {5.042029745639059, 1.584911068059481, 0.4769362762044699,
                                 0.08885599049425769, 0.0008862271574665521};
constexpr double kInvErfcFp[5] = {-97321105893.54887, -10.926170294494911, -1.1125848189719498,
                                  -0.8932517253051876, -0.8862276214943954};
constexpr double kInvErfcFpp[5] = {9.551013739178229e22, 378.41716185156497, 1.1807463499934,
                                   0.14179614878847469, 0.0013920845503350683};

class ElementaryExtrasTest : public ::testing::Test {
protected:
    void SetUp() override { RevTape::active().clear(); }
    void TearDown() override { RevTape::active().clear(); }
};

/// Rev-mode value + gradient of a unary function at `xv`.
template <typename F>
void CheckRevValueGrad(const char* label, double xv, double expectedValue, double expectedGrad,
                       const F& fn, double gradTol = 1e-12) {
    SCOPED_TRACE(::testing::Message() << label << " at x=" << xv);
    RevTape& tape = RevTape::active();
    RevScalar x(xv, tape.input(xv));
    const RevScalar f = fn(x);
    EXPECT_NEAR(f.value, expectedValue, 1e-13);
    tape.reverse(f.node);
    EXPECT_NEAR(tape.adjoint(x.node), expectedGrad, gradTol);
}

/// Hessian-vector product of a unary function along v = 1 vs the analytic f''.
template <typename F>
void CheckRevHvp(const char* label, double xv, double expectedFpp, const F& fn,
                 double tol = 1e-12) {
    SCOPED_TRACE(::testing::Message() << label << " at x=" << xv);
    RevTape& tape = RevTape::active();
    RevScalar x(xv, tape.input(xv));
    const RevScalar f = fn(x);
    const std::uint32_t inputs[1] = {x.node};
    const double v[1] = {1.0};
    double out[1] = {0.0};
    ASSERT_TRUE(tape.hessianVector(f.node, inputs, 1, v, out));
    EXPECT_NEAR(out[0], expectedFpp, tol);
}

/// Forward value + first derivative of a unary function at `xv`.
template <typename F>
void CheckTangentValueGrad(const char* label, double xv, double expectedValue, double expectedGrad,
                           const F& fn) {
    SCOPED_TRACE(::testing::Message() << label << " at x=" << xv);
    const D1 x(xv, {1.0});
    const D1 f = fn(x);
    EXPECT_NEAR(f.value, expectedValue, 1e-14);
    EXPECT_NEAR(f.d[0], expectedGrad, 1e-14);
}

/// Nested-Tangent second derivative of a unary function at `xv`.
template <typename F>
void CheckTangentSecond(const char* label, double xv, double expectedFpp, const F& fn,
                        double tol = 1e-11) {
    SCOPED_TRACE(::testing::Message() << label << " at x=" << xv);
    using Inner = Tangent<double, 1>;
    using Outer = Tangent<Inner, 1>;
    Outer t;
    t.value.value = xv;
    t.value.d[0] = 1.0;
    t.d[0].value = 1.0;
    t.d[0].d[0] = 0.0;
    EXPECT_NEAR(fn(t).d[0].d[0], expectedFpp, tol);
}

/// Two-variable nested-Tangent point: `f.d[i].d[j]` is d2f/dxi dxj.
O2 MakeVar2(double v, std::size_t j) {
    O2 t;
    t.value.value = v;
    t.value.d[j] = 1.0;
    t.d[j].value = 1.0;
    return t;
}

TEST_F(ElementaryExtrasTest, RevCbrtExp2Log2Log10) {
    CheckRevValueGrad("cbrt", 0.7, kCbrtVal, kCbrtFp,
                      [](const RevScalar& x) { return lightape::cbrt(x); });
    CheckRevValueGrad("cbrt", -8.0, -2.0, 1.0 / 12.0,
                      [](const RevScalar& x) { return lightape::cbrt(x); });
    CheckRevValueGrad("exp2", 1.3, kExp2Val, kExp2Fp,
                      [](const RevScalar& x) { return lightape::exp2(x); });
    CheckRevValueGrad("log2", 3.0, kLog2Val, kLog2Fp,
                      [](const RevScalar& x) { return lightape::log2(x); });
    CheckRevValueGrad("log10", 7.0, kLog10Val, kLog10Fp,
                      [](const RevScalar& x) { return lightape::log10(x); });
}

TEST_F(ElementaryExtrasTest, RevCbrtExp2Log2Log10HessianVector) {
    CheckRevHvp("cbrt", 0.7, kCbrtFpp, [](const RevScalar& x) { return lightape::cbrt(x); });
    CheckRevHvp("exp2", 1.3, kExp2Fpp, [](const RevScalar& x) { return lightape::exp2(x); });
    CheckRevHvp("log2", 3.0, kLog2Fpp, [](const RevScalar& x) { return lightape::log2(x); });
    CheckRevHvp("log10", 7.0, kLog10Fpp, [](const RevScalar& x) { return lightape::log10(x); });
}

TEST_F(ElementaryExtrasTest, TangentCbrtExp2Log2Log10) {
    CheckTangentValueGrad("cbrt", 0.7, kCbrtVal, kCbrtFp,
                          [](const auto& x) { return lightape::cbrt(x); });
    CheckTangentValueGrad("cbrt", -8.0, -2.0, 1.0 / 12.0,
                          [](const auto& x) { return lightape::cbrt(x); });
    CheckTangentValueGrad("exp2", 1.3, kExp2Val, kExp2Fp,
                          [](const auto& x) { return lightape::exp2(x); });
    CheckTangentValueGrad("log2", 3.0, kLog2Val, kLog2Fp,
                          [](const auto& x) { return lightape::log2(x); });
    CheckTangentValueGrad("log10", 7.0, kLog10Val, kLog10Fp,
                          [](const auto& x) { return lightape::log10(x); });
}

TEST_F(ElementaryExtrasTest, RevCopysign) {
    struct Case {
        double a;
        double b;
        double value;
        double da;
    };
    const Case cases[] = {
        {2.5, -3.0, -2.5, -1.0}, {-2.5, -3.0, -2.5, 1.0}, {2.5, 3.0, 2.5, 1.0},
        {-2.5, 3.0, 2.5, -1.0},  {2.5, -0.0, -2.5, 1.0}, // a.e. sign(0) = +1, so the derivative is
                                                         // +1
    };
    for (const Case& c : cases) {
        SCOPED_TRACE(::testing::Message() << "a=" << c.a << " b=" << c.b);
        RevTape& tape = RevTape::active();
        RevScalar a(c.a, tape.input(c.a));
        RevScalar b(c.b, tape.input(c.b));
        const RevScalar f = lightape::copysign(a, b);
        EXPECT_DOUBLE_EQ(f.value, c.value);
        tape.reverse(f.node);
        EXPECT_DOUBLE_EQ(tape.adjoint(a.node), c.da);
        EXPECT_DOUBLE_EQ(tape.adjoint(b.node), 0.0);
    }
}

TEST_F(ElementaryExtrasTest, RevCopysignMixedOverloads) {
    RevTape& tape = RevTape::active();
    RevScalar a(2.5, tape.input(2.5));
    const RevScalar fb = lightape::copysign(a, -3.0);
    EXPECT_DOUBLE_EQ(fb.value, -2.5);
    tape.reverse(fb.node);
    EXPECT_DOUBLE_EQ(tape.adjoint(a.node), -1.0);

    tape.clear();
    RevScalar b(-3.0, tape.input(-3.0));
    const RevScalar fa = lightape::copysign(2.5, b);
    EXPECT_DOUBLE_EQ(fa.value, -2.5);
    tape.reverse(fa.node);
    EXPECT_DOUBLE_EQ(tape.adjoint(b.node), 0.0);
}

TEST_F(ElementaryExtrasTest, TangentCopysign) {
    const D1 a(2.5, {1.0});
    const D1 bNeg(-3.0, {0.0});
    const D1 bPos(3.0, {0.0});
    const D1 negated = lightape::copysign(a, bNeg);
    EXPECT_DOUBLE_EQ(negated.value, -2.5);
    EXPECT_DOUBLE_EQ(negated.d[0], -1.0);

    const D1 kept = lightape::copysign(a, bPos);
    EXPECT_DOUBLE_EQ(kept.value, 2.5);
    EXPECT_DOUBLE_EQ(kept.d[0], 1.0);

    // Negative magnitude with positive sign: derivative sign(a) sign(b) = -1.
    const D1 mirrored = lightape::copysign(D1(-2.5, {1.0}), bPos);
    EXPECT_DOUBLE_EQ(mirrored.value, 2.5);
    EXPECT_DOUBLE_EQ(mirrored.d[0], -1.0);

    // The sign source carries no gradient.
    const D1 directed = lightape::copysign(D1(2.5, {0.0}), D1(-3.0, {1.0}));
    EXPECT_DOUBLE_EQ(directed.value, -2.5);
    EXPECT_DOUBLE_EQ(directed.d[0], 0.0);

    // Signed zero: primal keeps the sign bit, the a.e. derivative stays +1.
    const D1 negZero = lightape::copysign(a, D1(-0.0, {0.0}));
    EXPECT_DOUBLE_EQ(negZero.value, -2.5);
    EXPECT_TRUE(std::signbit(negZero.value));
    EXPECT_DOUBLE_EQ(negZero.d[0], 1.0);
}

TEST_F(ElementaryExtrasTest, Log1mexpValuesAndGradients) {
    CheckRevValueGrad("log1mexp", -0.7, kLog1mexpVal, kLog1mexpFp,
                      [](const RevScalar& x) { return lightape::log1mexp(x); });
    CheckTangentValueGrad("log1mexp", -0.7, kLog1mexpVal, kLog1mexpFp,
                          [](const auto& x) { return lightape::log1mexp(x); });
    CheckTangentSecond("log1mexp''", -0.7, kLog1mexpFpp,
                       [](const auto& x) { return lightape::log1mexp(x); });
}

TEST_F(ElementaryExtrasTest, LogDiffExpValuesAndGradients) {
    RevTape& tape = RevTape::active();
    RevScalar a(1.5, tape.input(1.5));
    RevScalar b(-0.3, tape.input(-0.3));
    const RevScalar f = lightape::logDiffExp(a, b);
    EXPECT_NEAR(f.value, kLogDiffVal, 1e-14);
    tape.reverse(f.node);
    EXPECT_NEAR(tape.adjoint(a.node), kLogDiffFa, 1e-13);
    EXPECT_NEAR(tape.adjoint(b.node), kLogDiffFb, 1e-13);

    // First derivatives in forward mode.
    const D2 va(1.5, {1.0, 0.0});
    const D2 vb(-0.3, {0.0, 1.0});
    const D2 fwd = lightape::logDiffExp(va, vb);
    EXPECT_NEAR(fwd.value, kLogDiffVal, 1e-14);
    EXPECT_NEAR(fwd.d[0], kLogDiffFa, 1e-13);
    EXPECT_NEAR(fwd.d[1], kLogDiffFb, 1e-13);

    // Nested Tangent: the full 2x2 second-derivative matrix.
    const O2 oa = MakeVar2(1.5, 0);
    const O2 ob = MakeVar2(-0.3, 1);
    const O2 nest = lightape::logDiffExp(oa, ob);
    EXPECT_NEAR(nest.value.value, kLogDiffVal, 1e-14);
    EXPECT_NEAR(nest.value.d[0], kLogDiffFa, 1e-13);
    EXPECT_NEAR(nest.value.d[1], kLogDiffFb, 1e-13);
    EXPECT_NEAR(nest.d[0].d[0], kLogDiffFaa, 1e-12);
    EXPECT_NEAR(nest.d[0].d[1], kLogDiffFab, 1e-12);
    EXPECT_NEAR(nest.d[1].d[0], kLogDiffFab, 1e-12);
    EXPECT_NEAR(nest.d[1].d[1], kLogDiffFbb, 1e-12);
}

TEST_F(ElementaryExtrasTest, MultiplyLogValuesAndGradients) {
    RevTape& tape = RevTape::active();
    RevScalar a(2.0, tape.input(2.0));
    RevScalar b(3.0, tape.input(3.0));
    const RevScalar f = lightape::multiplyLog(a, b);
    EXPECT_NEAR(f.value, kMultLogVal, 1e-15);
    tape.reverse(f.node);
    EXPECT_NEAR(tape.adjoint(a.node), 1.5, 1e-14);
    EXPECT_NEAR(tape.adjoint(b.node), kLn2, 1e-15);

    // Nested Tangent: f_aa = -b/a^2, f_ab = 1/a, f_bb = 0.
    const O2 oa = MakeVar2(2.0, 0);
    const O2 ob = MakeVar2(3.0, 1);
    const O2 nest = lightape::multiplyLog(oa, ob);
    EXPECT_NEAR(nest.value.value, kMultLogVal, 1e-15);
    EXPECT_NEAR(nest.value.d[0], 1.5, 1e-15);
    EXPECT_NEAR(nest.value.d[1], kLn2, 1e-15);
    EXPECT_NEAR(nest.d[0].d[0], -0.75, 1e-15);
    EXPECT_NEAR(nest.d[0].d[1], 0.5, 1e-15);
    EXPECT_NEAR(nest.d[1].d[0], 0.5, 1e-15);
    EXPECT_NEAR(nest.d[1].d[1], 0.0, 1e-15);
}

TEST_F(ElementaryExtrasTest, MultiplyLogZeroBranchIsPrimalPinned) {
    RevTape& tape = RevTape::active();
    RevScalar a(3.0, tape.input(3.0));
    RevScalar b(0.0, tape.input(0.0));
    const RevScalar f = lightape::multiplyLog(a, b);
    EXPECT_DOUBLE_EQ(f.value, 0.0);
    EXPECT_EQ(f.node, RevTape::kNone); // constant node: zero gradient everywhere

    const D1 fd = lightape::multiplyLog(D1(3.0, {1.0}), D1(0.0, {1.0}));
    EXPECT_DOUBLE_EQ(fd.value, 0.0);
    EXPECT_DOUBLE_EQ(fd.d[0], 0.0);
}

TEST_F(ElementaryExtrasTest, InvErfcRoundTrip) {
    for (int i = 0; i < 5; ++i) {
        const double y = kInvErfcY[i];
        const double x = lightape::invErfc(y);
        SCOPED_TRACE(::testing::Message() << "y=" << y);
        EXPECT_NEAR(x, kInvErfcX[i], 1e-14);
        EXPECT_NEAR(std::erfc(x), y, 1e-14 * y);
    }
    // Mirror branch y > 1: erfc is decreasing, so x is negative.
    const double xLarge = lightape::invErfc(1.999);
    EXPECT_LT(xLarge, 0.0);
    EXPECT_NEAR(std::erfc(xLarge), 1.999, 1e-14);
}

TEST_F(ElementaryExtrasTest, InvErfcBoundaries) {
    const double posInf = lightape::invErfc(0.0);
    EXPECT_TRUE(std::isinf(posInf));
    EXPECT_GT(posInf, 0.0);
    const double negInf = lightape::invErfc(2.0);
    EXPECT_TRUE(std::isinf(negInf));
    EXPECT_LT(negInf, 0.0);
    EXPECT_TRUE(std::isnan(lightape::invErfc(-0.1)));
    EXPECT_TRUE(std::isnan(lightape::invErfc(2.1)));
    EXPECT_TRUE(std::isnan(lightape::invErfc(std::numeric_limits<double>::quiet_NaN())));
}

TEST_F(ElementaryExtrasTest, InvErfcGradientAndHessianVector) {
    for (int i = 0; i < 5; ++i) {
        SCOPED_TRACE(::testing::Message() << "y=" << kInvErfcY[i]);
        RevTape& tape = RevTape::active();
        RevScalar y(kInvErfcY[i], tape.input(kInvErfcY[i]));
        const RevScalar x = lightape::invErfc(y);
        EXPECT_NEAR(x.value, kInvErfcX[i], 1e-14);

        tape.reverse(x.node);
        EXPECT_NEAR(tape.adjoint(y.node), kInvErfcFp[i], 1e-12 * std::fabs(kInvErfcFp[i]));

        const std::uint32_t inputs[1] = {y.node};
        const double v[1] = {1.0};
        double out[1] = {0.0};
        ASSERT_TRUE(tape.hessianVector(x.node, inputs, 1, v, out));
        EXPECT_NEAR(out[0], kInvErfcFpp[i], 1e-12 * std::fabs(kInvErfcFpp[i]));
    }
    // Constant argument collapses to a node-less constant.
    EXPECT_EQ(lightape::invErfc(RevScalar(0.5)).node, RevTape::kNone);
}

} // namespace
