#include <cmath>
#include <cstddef>
#include <cstdint>

#include "lightape/ops/Bessel.h"
#include <gtest/gtest.h>

namespace {

using lightape::RevScalar;
using lightape::RevTape;
using lightape::Tangent;

using D1 = Tangent<double, 1>;

// High-precision references (mpmath, 60 dps) at the test points below.
constexpr double kI0p1 = 1.266065877752008335598245;
constexpr double kI1p2 = 1.590636854637329063382254;
constexpr double kI25p3 = 1.515339446681965137740579;
constexpr double kI17p23 = 1.302163297967226501848037;
constexpr double kI05p05 = 0.5879930867904163254888734;
constexpr double kI5p05 = 0.000008223171313109263961618051;
constexpr double kI25p25 = 5083317345.336952902744636;
constexpr double kI05p20 = 43279746.27242892806939711;

constexpr double kK0p1 = 0.4210244382407083333356274;
constexpr double kK1p2 = 0.1398658818165224272845988;
constexpr double kK25p3 = 0.08406063197411738265285773;
constexpr double kK05p05 = 1.075047603499920238722756;
constexpr double kK5p10 = 0.00005754184998531227927637402;
constexpr double kK17p23 = 0.1331550038778150676570207;
constexpr double kK05p20 = 5.776373974707444652757424e-10;

// (nu, z) = (1.7, 2.3): K Hessian entries (mpmath, 60 dps).
constexpr double kK17hNuNu = 0.09129863938824718058078814;
constexpr double kK17hNuZ = -0.1390532353470713164581826;
constexpr double kK17hZZ = 0.2863299853088954005501883;

constexpr double kLogI0p1 = 0.2359143585071786486894148;
constexpr double kLogI1p2 = 0.4641344735461597442558639;
constexpr double kLogI25p3 = 0.4156394710800817737298289;
constexpr double kLogI17p23 = 0.2640269567876036555339634;
constexpr double kLogI05p05 = -0.5310400883117819780933573;
constexpr double kLogI05p20 = 17.5831953300183317572537;

// (nu, z) = (1.7, 2.3): I and logI gradient/Hessian entries.
constexpr double kI17gNu = -1.056705988316677492807791;
constexpr double kI17gZ = 1.456160291314173764405267;
constexpr double kI17hNuNu = 0.4062157113901042620926097;
constexpr double kI17hNuZ = -0.7653275205123993984349454;
constexpr double kI17hZZ = 1.380439906871325732889382;
constexpr double kLogI17gNu = -0.8115003624862364984564715;
constexpr double kLogI17gZ = 1.118262428059021469997436;
constexpr double kLogI17hNuNu = -0.3465783299796358550664276;
constexpr double kLogI17hNuZ = 0.3197349244730403853750192;
constexpr double kLogI17hZZ = -0.1903981140643654240768152;

// (nu, z) = (0, 1): I Hessian at an integer order (mixed entry must stay
// finite even though besselINuDeriv's domain is nu > -1).
constexpr double kI0p1hNuNu = -1.445214342481081310881768;
constexpr double kI0p1hNuZ = 0.60190723019723457473754;
constexpr double kI0p1hZZ = 0.7009067737595233083905486;

void expectRel(double got, double expected, double rtol, const char* what) {
    SCOPED_TRACE(what);
    if (expected == 0.0) {
        EXPECT_NEAR(got, 0.0, rtol);
        return;
    }
    EXPECT_NEAR(got / expected, 1.0, rtol);
}

class BesselAdTest : public ::testing::Test {
protected:
    void SetUp() override { RevTape::active().clear(); }
    void TearDown() override { RevTape::active().clear(); }
};

TEST_F(BesselAdTest, DoubleReferenceValuesI) {
    expectRel(lightape::besselI(0.0, 1.0), kI0p1, 1e-13, "I_0(1)");
    expectRel(lightape::besselI(1.0, 2.0), kI1p2, 1e-13, "I_1(2)");
    expectRel(lightape::besselI(2.5, 3.0), kI25p3, 1e-13, "I_2.5(3)");
    expectRel(lightape::besselI(1.7, 2.3), kI17p23, 1e-13, "I_1.7(2.3)");
    expectRel(lightape::besselI(0.5, 0.5), kI05p05, 1e-13, "I_0.5(0.5)");
    expectRel(lightape::besselI(5.0, 0.5), kI5p05, 1e-13, "I_5(0.5)");
    // Asymptotic branch (z >= 15 + nu/2 and z >= nu^2/2).
    expectRel(lightape::besselI(2.5, 25.0), kI25p25, 1e-13, "I_2.5(25)");
    expectRel(lightape::besselI(0.5, 20.0), kI05p20, 1e-13, "I_0.5(20)");
}

TEST_F(BesselAdTest, DoubleReferenceValuesK) {
    expectRel(lightape::besselK(0.0, 1.0), kK0p1, 1e-13, "K_0(1)");
    expectRel(lightape::besselK(1.0, 2.0), kK1p2, 1e-13, "K_1(2)");
    expectRel(lightape::besselK(2.5, 3.0), kK25p3, 1e-13, "K_2.5(3)");
    expectRel(lightape::besselK(0.5, 0.5), kK05p05, 1e-13, "K_0.5(0.5)");
    expectRel(lightape::besselK(5.0, 10.0), kK5p10, 1e-13, "K_5(10)");
    expectRel(lightape::besselK(1.7, 2.3), kK17p23, 1e-13, "K_1.7(2.3)");
    // Asymptotic branch (z >= 16).
    expectRel(lightape::besselK(0.5, 20.0), kK05p20, 1e-13, "K_0.5(20)");
}

TEST_F(BesselAdTest, DoubleReferenceValuesLogI) {
    expectRel(lightape::logBesselI(0.0, 1.0), kLogI0p1, 1e-13, "logI_0(1)");
    expectRel(lightape::logBesselI(1.0, 2.0), kLogI1p2, 1e-13, "logI_1(2)");
    expectRel(lightape::logBesselI(2.5, 3.0), kLogI25p3, 1e-13, "logI_2.5(3)");
    expectRel(lightape::logBesselI(1.7, 2.3), kLogI17p23, 1e-13, "logI_1.7(2.3)");
    expectRel(lightape::logBesselI(0.5, 0.5), kLogI05p05, 1e-13, "logI_0.5(0.5)");
    expectRel(lightape::logBesselI(0.5, 20.0), kLogI05p20, 1e-13, "logI_0.5(20)");
}

TEST_F(BesselAdTest, BesselIGradient) {
    RevTape& tape = RevTape::active();
    RevScalar nu(1.7, tape.input(1.7));
    RevScalar z(2.3, tape.input(2.3));
    const RevScalar f = lightape::besselI(nu, z);
    EXPECT_NEAR(f.value, kI17p23, 1e-13);
    tape.reverse(f.node);
    // dI/dnu against the ported order-derivative kernel and mpmath.
    EXPECT_NEAR(tape.adjoint(nu.node), lightape::detail::besselINuDeriv(1.7, 2.3), 1e-11);
    EXPECT_NEAR(tape.adjoint(nu.node), kI17gNu, 1e-11);
    // dI/dz = I_{nu-1} - nu I_nu / z against the ported I kernel and mpmath.
    const double gZ =
        lightape::detail::besselI(0.7, 2.3) - 1.7 * lightape::detail::besselI(1.7, 2.3) / 2.3;
    EXPECT_NEAR(tape.adjoint(z.node), gZ, 1e-11);
    EXPECT_NEAR(tape.adjoint(z.node), kI17gZ, 1e-11);
}

TEST_F(BesselAdTest, BesselKGradient) {
    RevTape& tape = RevTape::active();
    RevScalar nu(1.7, tape.input(1.7));
    RevScalar z(2.3, tape.input(2.3));
    const RevScalar f = lightape::besselK(nu, z);
    EXPECT_NEAR(f.value, kK17p23, 1e-13);
    tape.reverse(f.node);
    // dK/dnu against the ported order-derivative kernel.
    EXPECT_NEAR(tape.adjoint(nu.node), lightape::detail::besselKNuDeriv(1.7, 2.3), 1e-11);
    // dK/dz = -(K_{nu-1}+K_{nu+1})/2 against the ported besselKPrime.
    EXPECT_NEAR(tape.adjoint(z.node), lightape::detail::besselKPrime(1.7, 2.3), 1e-11);
}

TEST_F(BesselAdTest, LogBesselIGradient) {
    RevTape& tape = RevTape::active();
    RevScalar nu(0.5, tape.input(0.5));
    RevScalar z(0.5, tape.input(0.5));
    const RevScalar f = lightape::logBesselI(nu, z);
    EXPECT_NEAR(f.value, kLogI05p05, 1e-13);
    tape.reverse(f.node);
    // dlogI/dnu = I_nu'/I and dlogI/dz = I_z/I from the ported I kernels.
    const double i = lightape::detail::besselI(0.5, 0.5);
    EXPECT_NEAR(tape.adjoint(nu.node), lightape::detail::besselINuDeriv(0.5, 0.5) / i, 1e-11);
    const double iz = lightape::detail::besselI(-0.5, 0.5) - 0.5 * i / 0.5;
    EXPECT_NEAR(tape.adjoint(z.node), iz / i, 1e-11);
}

TEST_F(BesselAdTest, BesselIHessianVector) {
    RevTape& tape = RevTape::active();
    RevScalar nu(1.7, tape.input(1.7));
    RevScalar z(2.3, tape.input(2.3));
    const RevScalar f = lightape::besselI(nu, z);

    const std::uint32_t inputs[2] = {nu.node, z.node};
    const double v[2] = {0.6, 0.5};
    double out[2] = {-99.0, -99.0};
    ASSERT_TRUE(tape.hessianVector(f.node, inputs, 2, v, out));
    EXPECT_NEAR(out[0], kI17hNuNu * v[0] + kI17hNuZ * v[1], 1e-11);
    EXPECT_NEAR(out[1], kI17hNuZ * v[0] + kI17hZZ * v[1], 1e-11);
}

TEST_F(BesselAdTest, BesselIHessianAtIntegerOrder) {
    RevTape& tape = RevTape::active();
    RevScalar nu(0.0, tape.input(0.0));
    RevScalar z(1.0, tape.input(1.0));
    const RevScalar f = lightape::besselI(nu, z);
    EXPECT_NEAR(f.value, kI0p1, 1e-13);

    const std::uint32_t inputs[2] = {nu.node, z.node};
    double out[2] = {0.0, 0.0};
    const double eNu[2] = {1.0, 0.0};
    ASSERT_TRUE(tape.hessianVector(f.node, inputs, 2, eNu, out));
    EXPECT_NEAR(out[0], kI0p1hNuNu, 1e-11);
    EXPECT_NEAR(out[1], kI0p1hNuZ, 1e-11);
    const double eZ[2] = {0.0, 1.0};
    ASSERT_TRUE(tape.hessianVector(f.node, inputs, 2, eZ, out));
    EXPECT_NEAR(out[0], kI0p1hNuZ, 1e-11);
    EXPECT_NEAR(out[1], kI0p1hZZ, 1e-11);
}

TEST_F(BesselAdTest, LogBesselIHessianVector) {
    RevTape& tape = RevTape::active();
    RevScalar nu(1.7, tape.input(1.7));
    RevScalar z(2.3, tape.input(2.3));
    const RevScalar f = lightape::logBesselI(nu, z);

    const std::uint32_t inputs[2] = {nu.node, z.node};
    const double v[2] = {0.6, 0.5};
    double out[2] = {-99.0, -99.0};
    ASSERT_TRUE(tape.hessianVector(f.node, inputs, 2, v, out));
    EXPECT_NEAR(out[0], kLogI17hNuNu * v[0] + kLogI17hNuZ * v[1], 1e-11);
    EXPECT_NEAR(out[1], kLogI17hNuZ * v[0] + kLogI17hZZ * v[1], 1e-11);
}

TEST_F(BesselAdTest, BesselKHessianVector) {
    RevTape& tape = RevTape::active();
    RevScalar nu(1.7, tape.input(1.7));
    RevScalar z(2.3, tape.input(2.3));
    const RevScalar f = lightape::besselK(nu, z);

    const std::uint32_t inputs[2] = {nu.node, z.node};
    const double v[2] = {0.6, 0.5};
    double out[2] = {-99.0, -99.0};
    ASSERT_TRUE(tape.hessianVector(f.node, inputs, 2, v, out));
    EXPECT_NEAR(out[0], kK17hNuNu * v[0] + kK17hNuZ * v[1], 1e-11);
    EXPECT_NEAR(out[1], kK17hNuZ * v[0] + kK17hZZ * v[1], 1e-11);

    // Far outside the quadrature's finite range (K_90(0.01) overflows) the
    // pack is non-finite, so the wrapper must record first order only and
    // hessianVector must fail loudly without writing outputs.
    RevScalar nuBig(90.0, tape.input(90.0));
    RevScalar zSmall(0.01, tape.input(0.01));
    const RevScalar fBig = lightape::besselK(nuBig, zSmall);
    const std::uint32_t bigInputs[2] = {nuBig.node, zSmall.node};
    double bigOut[2] = {-99.0, -99.0};
    EXPECT_FALSE(tape.hessianVector(fBig.node, bigInputs, 2, v, bigOut));
    EXPECT_DOUBLE_EQ(bigOut[0], -99.0);
    EXPECT_DOUBLE_EQ(bigOut[1], -99.0);
}

TEST_F(BesselAdTest, TangentDirectionalDerivatives) {
    const D1 nu(1.7, {0.6});
    const D1 z(2.3, {0.5});

    const D1 fi = lightape::besselI(nu, z);
    EXPECT_NEAR(fi.value, kI17p23, 1e-13);
    EXPECT_NEAR(fi.d[0], kI17gNu * 0.6 + kI17gZ * 0.5, 1e-11);

    const D1 fk = lightape::besselK(nu, z);
    EXPECT_NEAR(fk.value, kK17p23, 1e-13);
    EXPECT_NEAR(fk.d[0],
                lightape::detail::besselKNuDeriv(1.7, 2.3) * 0.6 +
                    lightape::detail::besselKPrime(1.7, 2.3) * 0.5,
                1e-11);

    const D1 fl = lightape::logBesselI(nu, z);
    EXPECT_NEAR(fl.value, kLogI17p23, 1e-13);
    EXPECT_NEAR(fl.d[0], kLogI17gNu * 0.6 + kLogI17gZ * 0.5, 1e-11);
}

} // namespace
