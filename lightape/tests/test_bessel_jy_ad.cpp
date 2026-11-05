#include <cmath>
#include <cstdint>
#include <string>

#include "lightape/ops/P2BesselJY.h"
#include <gtest/gtest.h>

namespace {

using lightape::RevScalar;
using lightape::RevTape;
using lightape::Tangent;

using D1 = Tangent<double, 1>;

// Hardcoded references (mpmath, 60 dps) at the test points below. The value
// quoted for J_2.5(3) in the task text (0.2714653900417834) was wrong; the
// mpmath value verified here is used instead.
constexpr double kJ0p1 = 0.76519768655796655145;
constexpr double kJ1p1 = 0.44005058574493351596;
constexpr double kJ0p2 = 0.223890779141235668052;
constexpr double kJ1p2 = 0.576724807756873387202;
constexpr double kJ0p10 = -0.245935764451348335198;
constexpr double kJ0p25 = 0.0962667832759581161735; // asymptotic
constexpr double kJ05p20 = 0.162880763855029870911; // asymptotic, half-integer
constexpr double kJ25p3 = 0.412710032209715993437;
constexpr double kJ17p23 = 0.484369094801492494731;
constexpr double kJm15p2 = -0.395623281358703517079; // negative order

constexpr double kY0p1 = 0.0882569642156769579829;
constexpr double kY1p1 = -0.781212821300288716547;
constexpr double kY0p2 = 0.510375672649745119597;
constexpr double kY0p10 = 0.0556711672835993914245;
constexpr double kY2p3 = -0.160400393484923729676;  // integer series, n = 2
constexpr double kY0p25 = -0.127249432268006137834; // asymptotic
constexpr double kY05p20 = -0.0728069047850618485501;
constexpr double kY25p3 = -0.369040730073797897345;
constexpr double kY17p23 = -0.340416331964779091182;

// (nu, z) = (2.5, 3): gradient/Hessian entries.
constexpr double kJ25gNu = -0.192038042727917664498;
constexpr double kJ25gZ = 0.133793188245661777020;
constexpr double kJ25hNuNu = -0.114775497560272613918;
constexpr double kJ25hNuZ = 0.157247638755839994379;
constexpr double kJ25hZZ = -0.170703572590411590335;
constexpr double kY25gNu = -0.371661914037953584259;
constexpr double kY25gZ = 0.394542032449000195956;
constexpr double kY25hNuNu = 0.167445169418791623652;
constexpr double kY25hNuZ = -0.077510924512256731009;
constexpr double kY25hZZ = -0.0187515655160062633521;

// (nu, z) = (1.7, 2.3): gradient/Hessian entries.
constexpr double kJ17gNu = -0.20997393112508831319;
constexpr double kJ17gZ = 0.117024184676278181743;
constexpr double kJ17hNuNu = -0.221644951122060337013;
constexpr double kJ17hNuZ = 0.276855693614494686175;
constexpr double kJ17hZZ = -0.270631654495089188159;
constexpr double kY17gNu = -0.476295271300551616408;
constexpr double kY17gZ = 0.489201859122849265028;
constexpr double kY17hNuNu = 0.256805852276404188931;
constexpr double kY17hNuZ = -0.0975797765102432396159;
constexpr double kY17hZZ = -0.0582542682924543460732;

// (nu, z) = (0, 1): J order derivatives are finite at nu = 0 (the series
// kernels' digamma weights are regular there).
constexpr double kJ0gNu = 0.138633715204053999681;
constexpr double kJ0gZ = -0.44005058574493351596;
constexpr double kJ0hNuNu = -1.74845155792253084406;
constexpr double kJ0hNuZ = 1.22712623014357148924;
constexpr double kJ0hZZ = -0.32514710081303303549;
constexpr double kY0gZ = 0.781212821300288716547; // = -Y_1(1)

// (nu, z) = (0, 1): Y order derivatives now come from the differentiated
// integral bridge (the reflection form is 0/0 at integer order).
constexpr double kY0gNu = -1.201969715317206499136662;
constexpr double kY0hNuNu = 0.4171829860159614332033243;
constexpr double kY0hNuZ = 0.6912298436920842628831417;
constexpr double kY0hZZ = -0.8694697855159656745300768;

// Integral-bridge band: the old transition-gap cells. mpmath, 60 dps.
constexpr double kJ10p20 = 0.1864825580239450832141083;
constexpr double kY10p20 = -0.04389465351565839489936544;
constexpr double kJ10p25 = -0.07517984394852328384132298;
constexpr double kY10p25 = -0.1487183904998064975723111;
constexpr double kJ10p30 = -0.1298768939985887681859474;
constexpr double kY10p30 = 0.07505670212239711328867641;
constexpr double kJ20p50 = -0.1167043527595797373414582;
constexpr double kY20p50 = 0.01644263394811577765001393;

// Bridge-band z-derivatives (mpmath): the differentiated integrals are
// checked against both mpmath and the recurrence
// `J'_nu = J_{nu-1} - (nu/z) J_nu`.
struct BridgeZDerivRef {
    double nu;
    double z;
    double jz;
    double yz;
};
constexpr BridgeZDerivRef kBridgeZDerivs[] = {
    {10.0, 25.0, 0.1381640524528476960223562, -0.0654065655197664086225987},
    {10.0, 30.0, -0.06835110313735413306350469, -0.1238900176591579383547424},
    {20.0, 50.0, -0.0136832939801300339173348, -0.107171718599032422071629},
    {2.7, 17.3, 0.01133779406776809176340189, 0.1904462382696564226780157},
    {0.5, 20.0, 0.06873488568868610177737285, 0.1647009364746564171245164},
};

// Integer-order Y order derivatives from the differentiated integral bridge
// (mpmath `diff` in order, 50 dps).
struct YIntegerOrderRef {
    double nu;
    double z;
    double gNu;
    double gNuNu;
};
constexpr YIntegerOrderRef kYIntegerOrderRefs[] = {
    {0.0, 0.5, -1.47414492602178358424176, 2.001060612840167786782362},
    {0.0, 3.0, 0.4084886555357891538926154, -0.9943265382981388944062333},
    {0.0, 20.0, -0.2623617292303385731557405, -0.1461414955240923736441709},
    {1.0, 0.5, -1.269591870427369895494198, -2.103775596983052868503575},
    {1.0, 3.0, -0.4069758966151807245863046, -0.41998244529123102334972},
    {1.0, 20.0, -0.1018491961231832624243786, 0.3857021907178476377929325},
    {5.0, 0.5, -22952.66268619931558890685, -68072.63545425014539962829},
    {5.0, 3.0, -1.709898849816946134358652, -2.161163060828893449894073},
    {5.0, 20.0, -0.1999824870009825586330981, 0.1789029501466907324666169},
};

// (nu, z) = (0, 25): integer order in the asymptotic region, where the
// order derivative IS available (finite P/Q polynomials).
constexpr double kJ0_25gNu = -0.199882940793320032165;
constexpr double kJ0_25gZ = 0.125350249580289904652;
constexpr double kJ0_25hNuNu = -0.232366377470464171769;
constexpr double kJ0_25hNuZ = 0.155241745658778314982;
constexpr double kJ0_25hZZ = -0.101280793259169712360;
constexpr double kY0_25gNu = -0.151215509562235393586;
constexpr double kY0_25gZ = 0.0988299647832374100533;
constexpr double kY0_25hNuNu = 0.317721224333117248288;
constexpr double kY0_25hNuZ = -0.196899711603542913853;
constexpr double kY0_25hZZ = 0.123296233676676641432;

// (nu, z) = (0.5, 20): asymptotic, non-integer.
constexpr double kJ05gNu = -0.112445481065510625281;
constexpr double kJ05gZ = 0.0687348856886861017774;
constexpr double kJ05hNuNu = -0.385517403838976630259;
constexpr double kJ05hNuZ = 0.254541235204869718209;
constexpr double kJ05hZZ = -0.166215707662054782330;
constexpr double kY05gNu = -0.251830874767072326624;
constexpr double kY05gZ = 0.164700936474656417125;
constexpr double kY05hNuNu = 0.181709300492703492616;
constexpr double kY05hNuZ = -0.106348039029421341837;
constexpr double kY05hZZ = 0.0645263536458383640386;

void expectRel(double got, double expected, double rtol, const char* what) {
    SCOPED_TRACE(what);
    if (expected == 0.0) {
        EXPECT_NEAR(got, 0.0, rtol);
        return;
    }
    EXPECT_NEAR(got / expected, 1.0, rtol);
}

class BesselJYAdTest : public ::testing::Test {
protected:
    void SetUp() override { RevTape::active().clear(); }
    void TearDown() override { RevTape::active().clear(); }
};

TEST_F(BesselJYAdTest, DoubleReferenceValuesJ) {
    expectRel(lightape::besselJ(0.0, 1.0), kJ0p1, 1e-13, "J_0(1)");
    expectRel(lightape::besselJ(1.0, 1.0), kJ1p1, 1e-13, "J_1(1)");
    expectRel(lightape::besselJ(0.0, 2.0), kJ0p2, 1e-13, "J_0(2)");
    expectRel(lightape::besselJ(1.0, 2.0), kJ1p2, 1e-13, "J_1(2)");
    expectRel(lightape::besselJ(0.0, 10.0), kJ0p10, 1e-13, "J_0(10)");
    expectRel(lightape::besselJ(2.5, 3.0), kJ25p3, 1e-13, "J_2.5(3)");
    expectRel(lightape::besselJ(1.7, 2.3), kJ17p23, 1e-13, "J_1.7(2.3)");
    expectRel(lightape::besselJ(-1.5, 2.0), kJm15p2, 1e-13, "J_-1.5(2)");
    // Asymptotic branch (z >= 15 + |nu| and z >= nu^2/2).
    expectRel(lightape::besselJ(0.0, 25.0), kJ0p25, 1e-13, "J_0(25)");
    expectRel(lightape::besselJ(0.5, 20.0), kJ05p20, 1e-13, "J_0.5(20)");
    // Integral-bridge band: the former transition gap was ~1e-7 at (10, 25),
    // ~2.6e-5 at (10, 30) and O(1) at (20, 50).
    expectRel(lightape::besselJ(10.0, 20.0), kJ10p20, 1e-12, "J_10(20)");
    expectRel(lightape::besselJ(10.0, 25.0), kJ10p25, 1e-12, "J_10(25)");
    expectRel(lightape::besselJ(10.0, 30.0), kJ10p30, 1e-12, "J_10(30)");
    expectRel(lightape::besselJ(20.0, 50.0), kJ20p50, 1e-12, "J_20(50)");
}

TEST_F(BesselJYAdTest, DoubleReferenceValuesY) {
    expectRel(lightape::besselY(0.0, 1.0), kY0p1, 1e-13, "Y_0(1)");
    expectRel(lightape::besselY(1.0, 1.0), kY1p1, 1e-13, "Y_1(1)");
    expectRel(lightape::besselY(0.0, 2.0), kY0p2, 1e-13, "Y_0(2)");
    // The integer-order DLMF 10.8.1 sums cancel here (largest term ~678 vs a
    // sum of ~-0.5), so Y_0(10) carries ~5e-14 absolute (~8e-13 relative)
    // double-precision cancellation error.
    expectRel(lightape::besselY(0.0, 10.0), kY0p10, 1e-12, "Y_0(10)");
    expectRel(lightape::besselY(2.0, 3.0), kY2p3, 1e-13, "Y_2(3)");
    expectRel(lightape::besselY(2.5, 3.0), kY25p3, 1e-13, "Y_2.5(3)");
    expectRel(lightape::besselY(1.7, 2.3), kY17p23, 1e-12, "Y_1.7(2.3)");
    // Asymptotic branch.
    expectRel(lightape::besselY(0.0, 25.0), kY0p25, 1e-13, "Y_0(25)");
    expectRel(lightape::besselY(0.5, 20.0), kY05p20, 1e-13, "Y_0.5(20)");
    // Integral-bridge band (the old gap, and the integer-order series
    // cancellation at the larger in-band z).
    expectRel(lightape::besselY(10.0, 20.0), kY10p20, 1e-12, "Y_10(20)");
    expectRel(lightape::besselY(10.0, 25.0), kY10p25, 1e-12, "Y_10(25)");
    expectRel(lightape::besselY(10.0, 30.0), kY10p30, 1e-12, "Y_10(30)");
    expectRel(lightape::besselY(20.0, 50.0), kY20p50, 1e-12, "Y_20(50)");
}

TEST_F(BesselJYAdTest, ZDerivativeRecurrence) {
    expectRel(lightape::besselJPrime(2.5, 3.0), kJ25gZ, 1e-13, "J'_2.5(3)");
    expectRel(lightape::besselYPrime(2.5, 3.0), kY25gZ, 1e-13, "Y'_2.5(3)");
    expectRel(lightape::besselJPrime(0.0, 1.0), kJ0gZ, 1e-13, "J'_0(1)");
    expectRel(lightape::besselYPrime(0.0, 1.0), kY0gZ, 1e-13, "Y'_0(1)");
    // Bridge band: the differentiated integral kernels must match mpmath and
    // the recurrence J'_nu = J_{nu-1} - (nu/z) J_nu (same for Y).
    for (const BridgeZDerivRef& r : kBridgeZDerivs) {
        SCOPED_TRACE("bridge z-derivative at nu = " + std::to_string(r.nu) +
                     ", z = " + std::to_string(r.z));
        const double jz = lightape::besselJPrime(r.nu, r.z);
        const double jrec =
            lightape::besselJ(r.nu - 1.0, r.z) - r.nu * lightape::besselJ(r.nu, r.z) / r.z;
        expectRel(jz, r.jz, 1e-12, "J' vs mpmath");
        expectRel(jz, jrec, 1e-12, "J' vs recurrence");
        const double yz = lightape::besselYPrime(r.nu, r.z);
        const double yrec =
            lightape::besselY(r.nu - 1.0, r.z) - r.nu * lightape::besselY(r.nu, r.z) / r.z;
        expectRel(yz, r.yz, 1e-12, "Y' vs mpmath");
        expectRel(yz, yrec, 1e-12, "Y' vs recurrence");
    }
}

TEST_F(BesselJYAdTest, BesselJGradientAndHessian) {
    RevTape& tape = RevTape::active();
    RevScalar nu(2.5, tape.input(2.5));
    RevScalar z(3.0, tape.input(3.0));
    const RevScalar f = lightape::besselJ(nu, z);
    EXPECT_NEAR(f.value, kJ25p3, 1e-13);
    tape.reverse(f.node);
    EXPECT_NEAR(tape.adjoint(nu.node), kJ25gNu, 1e-11);
    EXPECT_NEAR(tape.adjoint(z.node), kJ25gZ, 1e-11);

    const std::uint32_t inputs[2] = {nu.node, z.node};
    const double v[2] = {0.6, 0.5};
    double out[2] = {-99.0, -99.0};
    ASSERT_TRUE(tape.hessianVector(f.node, inputs, 2, v, out));
    EXPECT_NEAR(out[0], kJ25hNuNu * v[0] + kJ25hNuZ * v[1], 1e-11);
    EXPECT_NEAR(out[1], kJ25hNuZ * v[0] + kJ25hZZ * v[1], 1e-11);
}

TEST_F(BesselJYAdTest, BesselYGradientAndHessian) {
    RevTape& tape = RevTape::active();
    RevScalar nu(2.5, tape.input(2.5));
    RevScalar z(3.0, tape.input(3.0));
    const RevScalar f = lightape::besselY(nu, z);
    EXPECT_NEAR(f.value, kY25p3, 1e-13);
    tape.reverse(f.node);
    EXPECT_NEAR(tape.adjoint(nu.node), kY25gNu, 1e-11);
    EXPECT_NEAR(tape.adjoint(z.node), kY25gZ, 1e-11);

    const std::uint32_t inputs[2] = {nu.node, z.node};
    const double v[2] = {0.6, 0.5};
    double out[2] = {-99.0, -99.0};
    ASSERT_TRUE(tape.hessianVector(f.node, inputs, 2, v, out));
    EXPECT_NEAR(out[0], kY25hNuNu * v[0] + kY25hNuZ * v[1], 1e-11);
    EXPECT_NEAR(out[1], kY25hNuZ * v[0] + kY25hZZ * v[1], 1e-11);
}

TEST_F(BesselJYAdTest, GradientAndHessianAtOneSeven) {
    RevTape& tape = RevTape::active();
    RevScalar nu(1.7, tape.input(1.7));
    RevScalar z(2.3, tape.input(2.3));
    const RevScalar fj = lightape::besselJ(nu, z);
    EXPECT_NEAR(fj.value, kJ17p23, 1e-13);
    tape.reverse(fj.node);
    EXPECT_NEAR(tape.adjoint(nu.node), kJ17gNu, 1e-11);
    EXPECT_NEAR(tape.adjoint(z.node), kJ17gZ, 1e-11);

    const std::uint32_t inputs[2] = {nu.node, z.node};
    const double v[2] = {0.6, 0.5};
    double out[2] = {-99.0, -99.0};
    ASSERT_TRUE(tape.hessianVector(fj.node, inputs, 2, v, out));
    EXPECT_NEAR(out[0], kJ17hNuNu * v[0] + kJ17hNuZ * v[1], 1e-11);
    EXPECT_NEAR(out[1], kJ17hNuZ * v[0] + kJ17hZZ * v[1], 1e-11);
}

TEST_F(BesselJYAdTest, GradientAtOneSevenY) {
    RevTape& tape = RevTape::active();
    RevScalar nu(1.7, tape.input(1.7));
    RevScalar z(2.3, tape.input(2.3));
    const RevScalar fy = lightape::besselY(nu, z);
    EXPECT_NEAR(fy.value, kY17p23, 1e-12);
    tape.reverse(fy.node);
    EXPECT_NEAR(tape.adjoint(nu.node), kY17gNu, 1e-11);
    EXPECT_NEAR(tape.adjoint(z.node), kY17gZ, 1e-11);

    const std::uint32_t inputs[2] = {nu.node, z.node};
    const double v[2] = {0.6, 0.5};
    double out[2] = {-99.0, -99.0};
    ASSERT_TRUE(tape.hessianVector(fy.node, inputs, 2, v, out));
    EXPECT_NEAR(out[0], kY17hNuNu * v[0] + kY17hNuZ * v[1], 1e-11);
    EXPECT_NEAR(out[1], kY17hNuZ * v[0] + kY17hZZ * v[1], 1e-11);
}

TEST_F(BesselJYAdTest, BesselJHessianAtIntegerOrderZero) {
    RevTape& tape = RevTape::active();
    RevScalar nu(0.0, tape.input(0.0));
    RevScalar z(1.0, tape.input(1.0));
    const RevScalar f = lightape::besselJ(nu, z);
    EXPECT_NEAR(f.value, kJ0p1, 1e-13);
    tape.reverse(f.node);
    EXPECT_NEAR(tape.adjoint(nu.node), kJ0gNu, 1e-11);
    EXPECT_NEAR(tape.adjoint(z.node), kJ0gZ, 1e-11);

    const std::uint32_t inputs[2] = {nu.node, z.node};
    const double v[2] = {1.0, 0.5};
    double out[2] = {-99.0, -99.0};
    ASSERT_TRUE(tape.hessianVector(f.node, inputs, 2, v, out));
    EXPECT_NEAR(out[0], kJ0hNuNu * v[0] + kJ0hNuZ * v[1], 1e-11);
    EXPECT_NEAR(out[1], kJ0hNuZ * v[0] + kJ0hZZ * v[1], 1e-11);
}

TEST_F(BesselJYAdTest, BesselYIntegerOrderHasFiniteOrderDerivative) {
    RevTape& tape = RevTape::active();
    RevScalar nu(0.0, tape.input(0.0));
    RevScalar z(1.0, tape.input(1.0));
    const RevScalar f = lightape::besselY(nu, z);
    EXPECT_NEAR(f.value, kY0p1, 1e-13);
    tape.reverse(f.node);
    // dY/dnu at integer order now comes from the differentiated integral
    // bridge (the reflection form is 0/0 there); full second order is
    // recorded and replayed.
    EXPECT_NEAR(tape.adjoint(nu.node), kY0gNu, 1e-11);
    EXPECT_NEAR(tape.adjoint(z.node), kY0gZ, 1e-11);
    EXPECT_TRUE(std::isfinite(tape.adjoint(nu.node)));

    const std::uint32_t inputs[2] = {nu.node, z.node};
    const double v[2] = {1.0, 0.5};
    double out[2] = {-99.0, -99.0};
    ASSERT_TRUE(tape.hessianVector(f.node, inputs, 2, v, out));
    EXPECT_NEAR(out[0], kY0hNuNu * v[0] + kY0hNuZ * v[1], 1e-11);
    EXPECT_NEAR(out[1], kY0hNuZ * v[0] + kY0hZZ * v[1], 1e-11);

    // Order-derivative kernels at integer orders n in {0, 1, 5} and
    // z in {0.5, 3, 20}: finite and <= 1e-11 relative against mpmath.
    for (std::size_t i = 0; i < sizeof(kYIntegerOrderRefs) / sizeof(kYIntegerOrderRefs[0]); ++i) {
        const YIntegerOrderRef& r = kYIntegerOrderRefs[i];
        SCOPED_TRACE("integer-order Y derivative, point " + std::to_string(i));
        const double gNu = lightape::besselYNuDeriv(r.nu, r.z);
        const double gNuNu = lightape::besselYNuDeriv2(r.nu, r.z);
        EXPECT_TRUE(std::isfinite(gNu));
        EXPECT_TRUE(std::isfinite(gNuNu));
        expectRel(gNu, r.gNu, 1e-11, "dY/dnu");
        expectRel(gNuNu, r.gNuNu, 1e-11, "d2Y/dnu2");
    }
}

TEST_F(BesselJYAdTest, AsymptoticGradientAndHessianJ) {
    RevTape& tape = RevTape::active();
    RevScalar nu(0.5, tape.input(0.5));
    RevScalar z(20.0, tape.input(20.0));
    const RevScalar fj = lightape::besselJ(nu, z);
    EXPECT_NEAR(fj.value, kJ05p20, 1e-13);
    tape.reverse(fj.node);
    EXPECT_NEAR(tape.adjoint(nu.node), kJ05gNu, 1e-11);
    EXPECT_NEAR(tape.adjoint(z.node), kJ05gZ, 1e-11);

    const std::uint32_t inputs[2] = {nu.node, z.node};
    const double vj[2] = {0.6, 0.5};
    double out[2] = {-99.0, -99.0};
    ASSERT_TRUE(tape.hessianVector(fj.node, inputs, 2, vj, out));
    EXPECT_NEAR(out[0], kJ05hNuNu * vj[0] + kJ05hNuZ * vj[1], 1e-11);
    EXPECT_NEAR(out[1], kJ05hNuZ * vj[0] + kJ05hZZ * vj[1], 1e-11);
}

TEST_F(BesselJYAdTest, AsymptoticGradientAndHessianY) {
    RevTape& tape = RevTape::active();
    RevScalar nu(0.5, tape.input(0.5));
    RevScalar z(20.0, tape.input(20.0));
    const RevScalar fy = lightape::besselY(nu, z);
    EXPECT_NEAR(fy.value, kY05p20, 1e-13);
    tape.reverse(fy.node);
    EXPECT_NEAR(tape.adjoint(nu.node), kY05gNu, 1e-11);
    EXPECT_NEAR(tape.adjoint(z.node), kY05gZ, 1e-11);

    const std::uint32_t inputs[2] = {nu.node, z.node};
    const double vy[2] = {0.6, 0.5};
    double out[2] = {-99.0, -99.0};
    ASSERT_TRUE(tape.hessianVector(fy.node, inputs, 2, vy, out));
    EXPECT_NEAR(out[0], kY05hNuNu * vy[0] + kY05hNuZ * vy[1], 1e-11);
    EXPECT_NEAR(out[1], kY05hNuZ * vy[0] + kY05hZZ * vy[1], 1e-11);
}

TEST_F(BesselJYAdTest, AsymptoticIntegerOrderY) {
    RevTape& tape = RevTape::active();
    RevScalar nu(0.0, tape.input(0.0));
    RevScalar z(25.0, tape.input(25.0));
    const RevScalar fy = lightape::besselY(nu, z);
    EXPECT_NEAR(fy.value, kY0p25, 1e-13);
    tape.reverse(fy.node);
    EXPECT_NEAR(tape.adjoint(nu.node), kY0_25gNu, 1e-11);
    EXPECT_NEAR(tape.adjoint(z.node), kY0_25gZ, 1e-11);

    const std::uint32_t inputs[2] = {nu.node, z.node};
    const double v[2] = {0.6, 0.5};
    double out[2] = {-99.0, -99.0};
    ASSERT_TRUE(tape.hessianVector(fy.node, inputs, 2, v, out));
    EXPECT_NEAR(out[0], kY0_25hNuNu * v[0] + kY0_25hNuZ * v[1], 1e-11);
    EXPECT_NEAR(out[1], kY0_25hNuZ * v[0] + kY0_25hZZ * v[1], 1e-11);
}

TEST_F(BesselJYAdTest, AsymptoticIntegerOrderJ) {
    RevTape& tape = RevTape::active();
    RevScalar nu(0.0, tape.input(0.0));
    RevScalar z(25.0, tape.input(25.0));
    const RevScalar fj = lightape::besselJ(nu, z);
    EXPECT_NEAR(fj.value, kJ0p25, 1e-13);
    tape.reverse(fj.node);
    EXPECT_NEAR(tape.adjoint(nu.node), kJ0_25gNu, 1e-11);
    EXPECT_NEAR(tape.adjoint(z.node), kJ0_25gZ, 1e-11);

    const std::uint32_t inputs[2] = {nu.node, z.node};
    const double v[2] = {0.6, 0.5};
    double out[2] = {-99.0, -99.0};
    ASSERT_TRUE(tape.hessianVector(fj.node, inputs, 2, v, out));
    EXPECT_NEAR(out[0], kJ0_25hNuNu * v[0] + kJ0_25hNuZ * v[1], 1e-11);
    EXPECT_NEAR(out[1], kJ0_25hNuZ * v[0] + kJ0_25hZZ * v[1], 1e-11);
}

TEST_F(BesselJYAdTest, TangentDirectionalDerivatives) {
    const D1 nu(1.7, {0.6});
    const D1 z(2.3, {0.5});

    const D1 fj = lightape::besselJ(nu, z);
    EXPECT_NEAR(fj.value, kJ17p23, 1e-13);
    EXPECT_NEAR(fj.d[0], kJ17gNu * 0.6 + kJ17gZ * 0.5, 1e-11);

    const D1 fy = lightape::besselY(nu, z);
    EXPECT_NEAR(fy.value, kY17p23, 1e-12);
    EXPECT_NEAR(fy.d[0], kY17gNu * 0.6 + kY17gZ * 0.5, 1e-11);
}

TEST_F(BesselJYAdTest, ConstantNuSlot) {
    RevTape& tape = RevTape::active();
    const RevScalar nu(2.5); // node-less constant
    RevScalar z(3.0, tape.input(3.0));
    const RevScalar f = lightape::besselJ(nu, z);
    EXPECT_NEAR(f.value, kJ25p3, 1e-13);
    tape.reverse(f.node);
    EXPECT_NEAR(tape.adjoint(z.node), kJ25gZ, 1e-11);

    const std::uint32_t inputs[1] = {z.node};
    const double v[1] = {1.0};
    double out[1] = {-99.0};
    ASSERT_TRUE(tape.hessianVector(f.node, inputs, 1, v, out));
    EXPECT_NEAR(out[0], kJ25hZZ, 1e-11);
}

} // namespace
