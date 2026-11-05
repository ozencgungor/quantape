#include <cmath>
#include <cstddef>
#include <cstdint>

#include "lightape/ops/P1Functions.h"
#include <gtest/gtest.h>

namespace {

using lightape::ExtArg;
using lightape::RevScalar;
using lightape::RevTape;
using lightape::Tangent;

using D1 = Tangent<double, 1>;

// High-precision references (mpmath, 60 dps) at the test points below.
constexpr double kInvPhi975 = 1.959963984540054235524594;
constexpr double kInvPhi25 = -0.674489750196081743202227;
constexpr double kInvPhi9 = 1.281551565544600466965103;
constexpr double kInvPhi999 = 3.0902323061678135415404;
constexpr double kInvPhiFp975 = 17.11008308033271533483244; // 1/phi(y)
constexpr double kInvPhiFpp9 = 41.60927029199777352248676;  // y/phi(y)^2, p = 0.9

constexpr double kGammaP21 = 0.2642411176571153568089525; // 1 - 2/e
constexpr double kGammaP05Half = 0.6826894921370858971704651;
constexpr double kGammaQ21 = 0.7357588823428846431910475;  // 2/e
constexpr double kGammaP2513 = 0.238634732154986083338386; // (a,z) = (2.5,1.3)
constexpr double kGammaQ2513 = 0.761365267845013916661614;
constexpr double kGammaGradA = -0.2296268983355593883975944;
constexpr double kGammaGradZ = 0.3038757209668302820822963; // z^{a-1} e^{-z}/Gamma(a)
constexpr double kGammaHaa = 0.1582639263086444380799648;
constexpr double kGammaHaz = -0.1339461011076966954621194;
constexpr double kGammaHzz = 0.04675011091797388955112251;

constexpr double kGammaPF42 = 0.8644747766224704756800241; // (a,z) = (2.5,4.2), CF branch
constexpr double kGammaF42GradA = -0.1330101453776599874434909;
constexpr double kGammaF42GradZ = 0.09709590451964102566168165;

constexpr double kIncBeta2340 = 0.5248; // exact rational
constexpr double kIncBeta5270 = 0.420175;
constexpr double kIncBeta5270GradA = -0.09944489456945687730961797; // symmetry branch
constexpr double kIncBeta5270GradB = 0.2524157812682858369224838;
constexpr double kIncBeta5270GradX = 2.1609;
constexpr double kIncBetaHalfQuarter = 0.3333333333333333333333333; // 1/3
// (a,b,x) = (2,3,0.4)
constexpr double kBetaGradA = -0.2408693760875565782083151;
constexpr double kBetaGradB = 0.1563443364135987726592603;
constexpr double kBetaGradX = 1.728;
constexpr double kBetaHaa = 0.05839660125154110656084272;
constexpr double kBetaHab = 0.008923407670034274097305519;
constexpr double kBetaHax = 0.288649615321460047362865;
constexpr double kBetaHbb = -0.03572935927737488875151085;
constexpr double kBetaHbx = 0.1252933221323680994208716;
constexpr double kBetaHxx = -1.44;

class P1FunctionsTest : public ::testing::Test {
protected:
    void SetUp() override { RevTape::active().clear(); }
    void TearDown() override { RevTape::active().clear(); }
};

TEST_F(P1FunctionsTest, DoubleReferenceValues) {
    EXPECT_NEAR(lightape::invPhi(0.975), kInvPhi975, 1e-13);
    EXPECT_DOUBLE_EQ(lightape::invPhi(0.5), 0.0);
    EXPECT_NEAR(lightape::invPhi(0.25), kInvPhi25, 1e-13);
    EXPECT_NEAR(lightape::invPhi(0.9), kInvPhi9, 1e-13);
    EXPECT_NEAR(lightape::invPhi(0.999), kInvPhi999, 1e-13);

    EXPECT_NEAR(lightape::gammaP(2.0, 1.0), kGammaP21, 1e-13);
    EXPECT_NEAR(lightape::gammaP(0.5, 0.5), kGammaP05Half, 1e-13);
    EXPECT_NEAR(lightape::gammaQ(2.0, 1.0), kGammaQ21, 1e-13);
    EXPECT_NEAR(lightape::gammaP(2.5, 1.3), kGammaP2513, 1e-13);
    EXPECT_NEAR(lightape::gammaQ(2.5, 1.3), kGammaQ2513, 1e-13);

    EXPECT_NEAR(lightape::beta(2.0, 3.0), 1.0 / 12.0, 1e-15);
    EXPECT_NEAR(lightape::incBeta(2.0, 3.0, 0.4), kIncBeta2340, 1e-13);
    EXPECT_NEAR(lightape::incBeta(5.0, 2.0, 0.7), kIncBeta5270, 1e-13);
    EXPECT_NEAR(lightape::incBeta(0.5, 0.5, 0.25), kIncBetaHalfQuarter, 1e-13);
}

TEST_F(P1FunctionsTest, InvPhiGradient) {
    RevTape& tape = RevTape::active();
    RevScalar p(0.975, tape.input(0.975));
    const RevScalar y = lightape::invPhi(p);
    EXPECT_NEAR(y.value, kInvPhi975, 1e-13);
    tape.reverse(y.node);
    EXPECT_NEAR(tape.adjoint(p.node), kInvPhiFp975, 1e-12);
}

TEST_F(P1FunctionsTest, InvPhiHessianVector) {
    RevTape& tape = RevTape::active();
    RevScalar p(0.9, tape.input(0.9));
    const RevScalar y = lightape::invPhi(p);
    EXPECT_NEAR(y.value, kInvPhi9, 1e-13);

    const std::uint32_t inputs[1] = {p.node};
    const double v[1] = {1.0};
    double out[1] = {0.0};
    ASSERT_TRUE(tape.hessianVector(y.node, inputs, 1, v, out));
    EXPECT_NEAR(out[0], kInvPhiFpp9 * v[0], 1e-12);
}

TEST_F(P1FunctionsTest, GammaGradients) {
    RevTape& tape = RevTape::active();
    RevScalar a(2.5, tape.input(2.5));
    RevScalar z(1.3, tape.input(1.3));

    const RevScalar p = lightape::gammaP(a, z);
    EXPECT_NEAR(p.value, kGammaP2513, 1e-13);
    tape.reverse(p.node);
    EXPECT_NEAR(tape.adjoint(a.node), kGammaGradA, 1e-12);
    EXPECT_NEAR(tape.adjoint(z.node), kGammaGradZ, 1e-12);

    tape.clear();
    RevScalar a2(2.5, tape.input(2.5));
    RevScalar z2(1.3, tape.input(1.3));
    const RevScalar q = lightape::gammaQ(a2, z2);
    EXPECT_NEAR(q.value, kGammaQ2513, 1e-13);
    tape.reverse(q.node);
    EXPECT_NEAR(tape.adjoint(a2.node), -kGammaGradA, 1e-12);
    EXPECT_NEAR(tape.adjoint(z2.node), -kGammaGradZ, 1e-12);
}

TEST_F(P1FunctionsTest, GammaPHessianVector) {
    RevTape& tape = RevTape::active();
    RevScalar a(2.5, tape.input(2.5));
    RevScalar z(1.3, tape.input(1.3));
    const RevScalar f = lightape::gammaP(a, z);

    const std::uint32_t inputs[2] = {a.node, z.node};
    const double v[2] = {0.7, -0.4};
    double out[2] = {0.0, 0.0};
    ASSERT_TRUE(tape.hessianVector(f.node, inputs, 2, v, out));
    EXPECT_NEAR(out[0], kGammaHaa * v[0] + kGammaHaz * v[1], 1e-12);
    EXPECT_NEAR(out[1], kGammaHaz * v[0] + kGammaHzz * v[1], 1e-12);
}

TEST_F(P1FunctionsTest, IncBetaGradients) {
    RevTape& tape = RevTape::active();
    RevScalar a(2.0, tape.input(2.0));
    RevScalar b(3.0, tape.input(3.0));
    RevScalar x(0.4, tape.input(0.4));
    const RevScalar f = lightape::incBeta(a, b, x);
    EXPECT_NEAR(f.value, kIncBeta2340, 1e-13);
    tape.reverse(f.node);
    EXPECT_NEAR(tape.adjoint(a.node), kBetaGradA, 1e-12);
    EXPECT_NEAR(tape.adjoint(b.node), kBetaGradB, 1e-12);
    EXPECT_NEAR(tape.adjoint(x.node), kBetaGradX, 1e-12);
}

TEST_F(P1FunctionsTest, IncBetaHessianVector) {
    RevTape& tape = RevTape::active();
    RevScalar a(2.0, tape.input(2.0));
    RevScalar b(3.0, tape.input(3.0));
    RevScalar x(0.4, tape.input(0.4));
    const RevScalar f = lightape::incBeta(a, b, x);

    const std::uint32_t inputs[3] = {a.node, b.node, x.node};
    const double v[3] = {0.3, -0.5, 0.8};
    double out[3] = {0.0, 0.0, 0.0};
    ASSERT_TRUE(tape.hessianVector(f.node, inputs, 3, v, out));
    EXPECT_NEAR(out[0], kBetaHaa * v[0] + kBetaHab * v[1] + kBetaHax * v[2], 1e-12);
    EXPECT_NEAR(out[1], kBetaHab * v[0] + kBetaHbb * v[1] + kBetaHbx * v[2], 1e-12);
    EXPECT_NEAR(out[2], kBetaHax * v[0] + kBetaHbx * v[1] + kBetaHxx * v[2], 1e-12);
}

TEST_F(P1FunctionsTest, ContinuedFractionAndSymmetryBranches) {
    RevTape& tape = RevTape::active();
    RevScalar a(2.5, tape.input(2.5));
    RevScalar z(4.2, tape.input(4.2));
    const RevScalar f = lightape::gammaP(a, z); // x >= a + 1: continued fraction
    EXPECT_NEAR(f.value, kGammaPF42, 1e-13);
    tape.reverse(f.node);
    EXPECT_NEAR(tape.adjoint(a.node), kGammaF42GradA, 1e-12);
    EXPECT_NEAR(tape.adjoint(z.node), kGammaF42GradZ, 1e-12);

    tape.clear();
    RevScalar ba(5.0, tape.input(5.0));
    RevScalar bb(2.0, tape.input(2.0));
    RevScalar bx(0.7, tape.input(0.7));
    const RevScalar bi = lightape::incBeta(ba, bb, bx); // 1 - I_{1-x}(b,a): symmetry
    EXPECT_NEAR(bi.value, kIncBeta5270, 1e-13);
    tape.reverse(bi.node);
    EXPECT_NEAR(tape.adjoint(ba.node), kIncBeta5270GradA, 1e-12);
    EXPECT_NEAR(tape.adjoint(bb.node), kIncBeta5270GradB, 1e-12);
    EXPECT_NEAR(tape.adjoint(bx.node), kIncBeta5270GradX, 1e-12);
}

TEST_F(P1FunctionsTest, TangentDirectionalDerivatives) {
    const D1 p1(0.975, {1.0});
    const D1 y1 = lightape::invPhi(p1);
    EXPECT_NEAR(y1.value, kInvPhi975, 1e-13);
    EXPECT_NEAR(y1.d[0], kInvPhiFp975, 1e-12);

    // gammaP: one direction in a, one in z.
    const D1 a1(2.5, {1.0});
    const D1 z1(1.3, {0.0});
    const D1 pa = lightape::gammaP(a1, z1);
    EXPECT_NEAR(pa.value, kGammaP2513, 1e-13);
    EXPECT_NEAR(pa.d[0], kGammaGradA, 1e-12);

    const D1 a2(2.5, {0.0});
    const D1 z2(1.3, {1.0});
    const D1 pz = lightape::gammaP(a2, z2);
    EXPECT_NEAR(pz.d[0], kGammaGradZ, 1e-12);

    const D1 q2 = lightape::gammaQ(a2, z2);
    EXPECT_NEAR(q2.value, kGammaQ2513, 1e-13);
    EXPECT_NEAR(q2.d[0], -kGammaGradZ, 1e-12);

    // incBeta: combined direction (0.3, -0.5, 0.8).
    const D1 ba(2.0, {0.3});
    const D1 bb(3.0, {-0.5});
    const D1 bx(0.4, {0.8});
    const D1 bi = lightape::incBeta(ba, bb, bx);
    EXPECT_NEAR(bi.value, kIncBeta2340, 1e-13);
    EXPECT_NEAR(bi.d[0], kBetaGradA * 0.3 + kBetaGradB * -0.5 + kBetaGradX * 0.8, 1e-12);
}

TEST_F(P1FunctionsTest, FirstOrderOnlyBlockFailsLoudly) {
    RevTape& tape = RevTape::active();
    RevScalar a(2.0, tape.input(2.0));
    RevScalar b(3.0, tape.input(3.0));
    RevScalar x(0.4, tape.input(0.4));
    const RevScalar args[3] = {a, b, x};
    auto vjp = [](const void* /*payload*/, double seed, const ExtArg* arg, std::uint32_t /*n*/,
                  RevTape& t) noexcept {
        t.addAdjoint(arg[0].node, seed * kBetaGradA);
        t.addAdjoint(arg[1].node, seed * kBetaGradB);
        t.addAdjoint(arg[2].node, seed * kBetaGradX);
    };
    const RevScalar f = lightape::makeVjpVar(kIncBeta2340, args, 3, vjp);

    const std::uint32_t inputs[3] = {a.node, b.node, x.node};
    const double v[3] = {1.0, 1.0, 1.0};
    double out[3] = {-99.0, -99.0, -99.0};
    EXPECT_FALSE(tape.hessianVector(f.node, inputs, 3, v, out));
    EXPECT_DOUBLE_EQ(out[0], -99.0);
    EXPECT_DOUBLE_EQ(out[1], -99.0);
    EXPECT_DOUBLE_EQ(out[2], -99.0);
}

} // namespace
