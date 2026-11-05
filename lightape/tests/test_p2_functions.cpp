#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>

#include "lightape/ops/P2Functions.h"
#include <gtest/gtest.h>

namespace {

using lightape::RevScalar;
using lightape::RevTape;
using lightape::Tangent;

using D1 = Tangent<double, 1>;
using D2 = Tangent<double, 2>;

class P2FunctionsTest : public ::testing::Test {
protected:
    void SetUp() override { RevTape::active().clear(); }
    void TearDown() override { RevTape::active().clear(); }
};

// ── fdim ──

TEST_F(P2FunctionsTest, FdimDouble) {
    EXPECT_DOUBLE_EQ(lightape::fdim(3.0, 1.0), 2.0);
    EXPECT_DOUBLE_EQ(lightape::fdim(1.0, 3.0), 0.0);
    EXPECT_DOUBLE_EQ(lightape::fdim(2.0, 2.0), 0.0);
    EXPECT_DOUBLE_EQ(lightape::fdim(-1.0, -3.0), 2.0);
}

TEST_F(P2FunctionsTest, FdimRevGradient) {
    RevTape& tape = RevTape::active();
    RevScalar a(3.0, tape.input(3.0));
    RevScalar b(1.0, tape.input(1.0));
    const std::size_t before = tape.size();
    const RevScalar f = lightape::fdim(a, b);
    EXPECT_DOUBLE_EQ(f.value, 2.0);
    EXPECT_EQ(tape.size(), before + 1u); // one shared Sub node, no extra nodes
    tape.reverse(f.node);
    EXPECT_DOUBLE_EQ(tape.adjoint(a.node), 1.0);
    EXPECT_DOUBLE_EQ(tape.adjoint(b.node), -1.0);

    // Flat branch: node-less zero, zero gradient on both arguments.
    tape.clear();
    RevScalar a2(1.0, tape.input(1.0));
    RevScalar b2(3.0, tape.input(3.0));
    const RevScalar f2 = lightape::fdim(a2, b2);
    EXPECT_DOUBLE_EQ(f2.value, 0.0);
    EXPECT_EQ(f2.node, RevTape::kNone);
    EXPECT_DOUBLE_EQ(tape.adjoint(a2.node), 0.0);
    EXPECT_DOUBLE_EQ(tape.adjoint(b2.node), 0.0);
}

TEST_F(P2FunctionsTest, FdimTangent) {
    const D2 a(3.0, {1.0, 0.0});
    const D2 b(1.0, {0.0, 1.0});
    const D2 f = lightape::fdim(a, b);
    EXPECT_DOUBLE_EQ(f.value, 2.0);
    EXPECT_DOUBLE_EQ(f.d[0], 1.0);
    EXPECT_DOUBLE_EQ(f.d[1], -1.0);

    const D2 flatA(1.0, {1.0, 0.0});
    const D2 flatB(3.0, {0.0, 1.0});
    const D2 flat = lightape::fdim(flatA, flatB);
    EXPECT_DOUBLE_EQ(flat.value, 0.0);
    EXPECT_DOUBLE_EQ(flat.d[0], 0.0);
    EXPECT_DOUBLE_EQ(flat.d[1], 0.0);
}

// ── fmod ──

TEST_F(P2FunctionsTest, FmodDouble) {
    EXPECT_DOUBLE_EQ(lightape::fmod(5.5, 2.0), 1.5);
    EXPECT_DOUBLE_EQ(lightape::fmod(-5.5, 2.0), -1.5);
    EXPECT_DOUBLE_EQ(lightape::fmod(5.5, -2.0), 1.5);
    EXPECT_DOUBLE_EQ(lightape::fmod(1.0, 0.3), std::fmod(1.0, 0.3));
}

TEST_F(P2FunctionsTest, FmodRevGradientAndZeroHvp) {
    RevTape& tape = RevTape::active();
    RevScalar a(5.5, tape.input(5.5));
    RevScalar b(2.0, tape.input(2.0));
    const RevScalar f = lightape::fmod(a, b);
    EXPECT_DOUBLE_EQ(f.value, 1.5);
    tape.reverse(f.node);
    EXPECT_DOUBLE_EQ(tape.adjoint(a.node), 1.0);  // d/da = 1
    EXPECT_DOUBLE_EQ(tape.adjoint(b.node), -2.0); // d/db = -trunc(2.75)

    const std::uint32_t inputs[2] = {a.node, b.node};
    const double v[2] = {0.3, -0.7};
    double out[2] = {1.0, 1.0};
    ASSERT_TRUE(tape.hessianVector(f.node, inputs, 2, v, out));
    EXPECT_DOUBLE_EQ(out[0], 0.0);
    EXPECT_DOUBLE_EQ(out[1], 0.0);

    // Negative quotient: trunc(-2.75) = -2, so d/db = +2 and the value is -1.5.
    tape.clear();
    RevScalar a2(-5.5, tape.input(-5.5));
    RevScalar b2(2.0, tape.input(2.0));
    const RevScalar f2 = lightape::fmod(a2, b2);
    EXPECT_DOUBLE_EQ(f2.value, -1.5);
    tape.reverse(f2.node);
    EXPECT_DOUBLE_EQ(tape.adjoint(a2.node), 1.0);
    EXPECT_DOUBLE_EQ(tape.adjoint(b2.node), 2.0);
}

TEST_F(P2FunctionsTest, FmodTangent) {
    const D2 a(5.5, {1.0, 0.0});
    const D2 b(2.0, {0.0, 1.0});
    const D2 f = lightape::fmod(a, b);
    EXPECT_DOUBLE_EQ(f.value, 1.5);
    EXPECT_DOUBLE_EQ(f.d[0], 1.0);
    EXPECT_DOUBLE_EQ(f.d[1], -2.0);

    const D2 negA(-5.5, {1.0, 0.0});
    const D2 negF = lightape::fmod(negA, b);
    EXPECT_DOUBLE_EQ(negF.value, -1.5);
    EXPECT_DOUBLE_EQ(negF.d[0], 1.0);
    EXPECT_DOUBLE_EQ(negF.d[1], 2.0);
}

// ── remainder ──

TEST_F(P2FunctionsTest, RemainderDouble) {
    EXPECT_DOUBLE_EQ(lightape::remainder(5.5, 2.0), -0.5); // n = 3
    EXPECT_DOUBLE_EQ(lightape::remainder(5.4, 2.0), -0.6); // n = 3
    EXPECT_DOUBLE_EQ(lightape::remainder(7.5, 2.0), -0.5); // n = 4 (half to even)
    EXPECT_DOUBLE_EQ(lightape::remainder(0.5, 1.0), 0.5);  // n = 0 (half to even)
    EXPECT_DOUBLE_EQ(lightape::remainder(-5.5, 2.0), 0.5); // n = -3
}

TEST_F(P2FunctionsTest, RemainderRevGradientAndZeroHvp) {
    RevTape& tape = RevTape::active();
    RevScalar a(5.5, tape.input(5.5));
    RevScalar b(2.0, tape.input(2.0));
    const RevScalar f = lightape::remainder(a, b);
    EXPECT_DOUBLE_EQ(f.value, -0.5);
    tape.reverse(f.node);
    EXPECT_DOUBLE_EQ(tape.adjoint(a.node), 1.0);  // d/da = 1
    EXPECT_DOUBLE_EQ(tape.adjoint(b.node), -3.0); // d/db = -nearbyint(2.75)

    const std::uint32_t inputs[2] = {a.node, b.node};
    const double v[2] = {0.3, -0.7};
    double out[2] = {1.0, 1.0};
    ASSERT_TRUE(tape.hessianVector(f.node, inputs, 2, v, out));
    EXPECT_DOUBLE_EQ(out[0], 0.0);
    EXPECT_DOUBLE_EQ(out[1], 0.0);

    // Half-to-even at 0.5/1: n = 0, so the b-gradient vanishes.
    tape.clear();
    RevScalar a2(0.5, tape.input(0.5));
    RevScalar b2(1.0, tape.input(1.0));
    const RevScalar f2 = lightape::remainder(a2, b2);
    EXPECT_DOUBLE_EQ(f2.value, 0.5);
    tape.reverse(f2.node);
    EXPECT_DOUBLE_EQ(tape.adjoint(a2.node), 1.0);
    EXPECT_DOUBLE_EQ(tape.adjoint(b2.node), 0.0);
}

TEST_F(P2FunctionsTest, RemainderTangent) {
    const D2 a(5.5, {1.0, 0.0});
    const D2 b(2.0, {0.0, 1.0});
    const D2 f = lightape::remainder(a, b);
    EXPECT_DOUBLE_EQ(f.value, -0.5);
    EXPECT_DOUBLE_EQ(f.d[0], 1.0);
    EXPECT_DOUBLE_EQ(f.d[1], -3.0);
}

// ── lambertW0 ──

// mpmath (60 dps) references at the points below.
constexpr double kLambertX[9] = {-0.36, -0.2, -0.05, 0.0, 1e-6, 0.5, 1.0, 2.0, 10.0};
constexpr double kLambertW[9] = {
    -0.8060843159708176, -0.25917110181907377, -0.05270598355154635, 0.0,
    9.999990000015e-07,  0.35173371124919584,  0.5671432904097838,   0.8526055020137255,
    1.7455280027406994};
constexpr double kLambertFp[9] = {
    11.54689014005681, 1.749196760921836,  1.1127692698651033, 1.0,
    0.9999980000045,   0.5204186421068739, 0.3618962566348892, 0.23010983749291725,
    0.0635771334693451};
constexpr double kLambertFpp[9] = {
    -820.9010073028068,   -7.189778342381777,  -2.545405526448661,  -2.0,
    -1.9999910000319998,  -0.4711971881257444, -0.2145406462821437, -0.08153219555043212,
    -0.005514283068880891};

TEST_F(P2FunctionsTest, LambertW0ValuesAndRoundTrip) {
    for (int i = 0; i < 9; ++i) {
        const double x = kLambertX[i];
        SCOPED_TRACE(::testing::Message() << "x=" << x);
        const double w = lightape::lambertW0(x);
        EXPECT_NEAR(w, kLambertW[i], 1e-15) << "value";
        // Round trip w e^w = x, relative away from x = 0.
        EXPECT_NEAR(w * std::exp(w), x, 1e-14 * std::fmax(1.0, std::fabs(x))) << "round trip";
    }
}

TEST_F(P2FunctionsTest, LambertW0Boundaries) {
    EXPECT_TRUE(std::isnan(lightape::lambertW0(-0.5)));
    EXPECT_TRUE(std::isnan(lightape::lambertW0(std::numeric_limits<double>::quiet_NaN())));
    EXPECT_DOUBLE_EQ(lightape::lambertW0(0.0), 0.0);
    EXPECT_DOUBLE_EQ(lightape::lambertW0(-lightape::kP2InvE), -1.0);
    const double inf = lightape::lambertW0(std::numeric_limits<double>::infinity());
    EXPECT_TRUE(std::isinf(inf));
    EXPECT_GT(inf, 0.0);
}

TEST_F(P2FunctionsTest, LambertW0RevGradientAndHvp) {
    for (int i = 0; i < 9; ++i) {
        SCOPED_TRACE(::testing::Message() << "x=" << kLambertX[i]);
        RevTape& tape = RevTape::active();
        RevScalar x(kLambertX[i], tape.input(kLambertX[i]));
        const RevScalar f = lightape::lambertW0(x);
        EXPECT_NEAR(f.value, kLambertW[i], 1e-15);
        tape.reverse(f.node);
        EXPECT_NEAR(tape.adjoint(x.node), kLambertFp[i],
                    1e-12 * std::fmax(1.0, std::fabs(kLambertFp[i])));

        const std::uint32_t inputs[1] = {x.node};
        const double v[1] = {1.0};
        double out[1] = {0.0};
        ASSERT_TRUE(tape.hessianVector(f.node, inputs, 1, v, out));
        EXPECT_NEAR(out[0], kLambertFpp[i], 1e-12 * std::fmax(1.0, std::fabs(kLambertFpp[i])));
    }
}

TEST_F(P2FunctionsTest, LambertW0Tangent) {
    for (int i = 0; i < 9; ++i) {
        SCOPED_TRACE(::testing::Message() << "x=" << kLambertX[i]);
        const D1 x(kLambertX[i], {1.0});
        const D1 f = lightape::lambertW0(x);
        EXPECT_NEAR(f.value, kLambertW[i], 1e-15);
        EXPECT_NEAR(f.d[0], kLambertFp[i], 1e-14 * std::fmax(1.0, std::fabs(kLambertFp[i])));
    }
}

// ── Owen's T ──

// mpmath (60 dps) references: value, dT/dh, dT/da, d2T/dh2, d2T/dhda, d2T/da2.
struct OwensCase {
    double h;
    double a;
    double value;
    double gh;
    double ga;
    double hhh;
    double hha;
    double haa;
};

const OwensCase kOwensCases[] = {
    {1.0, 1.0, 0.0667418821657009666, -0.0825954355170083458, 0.0292749157621595803,
     0.0240456039926891851, -0.0585498315243191607, -0.0585498315243191607},
    {0.0, 0.5, 0.0737918088252166371, 0.0, 0.127323954473516269, -0.0795774715459476679, 0.0,
     -0.101859163578813015},
    {2.0, 3.0, 0.0113750659715450406, -0.0269954832033272055, 3.28042787482862596e-11,
     0.0539909654225260485, -6.56085574965725193e-10, -4.13333912228406871e-10},
    {0.5, 0.5, 0.064488602847503757, -0.0347510748052619068, 0.108905949513225139,
     -0.0506906810431347584, -0.0680662184457657118, -0.100738003299733253},
    {-1.3, 0.7, 0.0371528818647279012, 0.0545961046914704756, 0.0303274151120053343,
     0.0393434421370900604, 0.0587442030719543321, -0.0643728899008630268},
    {1.5, -2.5, -0.0334026482389603216, 0.0647473462387513503, 6.29894414102151467e-6,
     -0.0970068509955710104, -0.000068501017533608972, 0.0000397756602008470646},
    {3.7, 5.2, 0.0000538998667386941307, -0.000212390135275375704, 2.50179598740036982e-86,
     0.000785843500518890142, -2.59556330100813597e-84, -1.79025765670778086e-84},
    {0.0, 1.0, 0.125, 0.0, 0.0795774715459476679, -0.159154943091895336, 0.0,
     -0.0795774715459476679},
};

TEST_F(P2FunctionsTest, OwensTDoubleReferences) {
    for (const OwensCase& c : kOwensCases) {
        SCOPED_TRACE(::testing::Message() << "h=" << c.h << " a=" << c.a);
        EXPECT_NEAR(lightape::owensT(c.h, c.a), c.value,
                    1e-13 * std::fmax(1.0, std::fabs(c.value)));
    }
    // Exact boundary values: T(0,a) = atan(a)/(2 pi), T(0,1) = 1/8.
    EXPECT_DOUBLE_EQ(lightape::owensT(0.0, 1.0), 0.125);
    EXPECT_DOUBLE_EQ(lightape::owensT(1.0, 0.0), 0.0);
    EXPECT_DOUBLE_EQ(lightape::owensT(-1.0, 0.0), 0.0);
    EXPECT_TRUE(std::isnan(lightape::owensT(0.5, std::numeric_limits<double>::quiet_NaN())));
    EXPECT_TRUE(std::isnan(lightape::owensT(std::numeric_limits<double>::quiet_NaN(), 0.5)));
}

TEST_F(P2FunctionsTest, OwensTSymmetries) {
    for (const OwensCase& c : kOwensCases) {
        SCOPED_TRACE(::testing::Message() << "h=" << c.h << " a=" << c.a);
        EXPECT_DOUBLE_EQ(lightape::owensT(-c.h, c.a), lightape::owensT(c.h, c.a));
        EXPECT_DOUBLE_EQ(lightape::owensT(c.h, -c.a), -lightape::owensT(c.h, c.a));
    }
}

TEST_F(P2FunctionsTest, OwensTRevGradientAndHvp) {
    for (const OwensCase& c : kOwensCases) {
        SCOPED_TRACE(::testing::Message() << "h=" << c.h << " a=" << c.a);
        RevTape& tape = RevTape::active();
        RevScalar h(c.h, tape.input(c.h));
        RevScalar a(c.a, tape.input(c.a));
        const RevScalar f = lightape::owensT(h, a);
        EXPECT_NEAR(f.value, c.value, 1e-13 * std::fmax(1.0, std::fabs(c.value)));
        tape.reverse(f.node);
        EXPECT_NEAR(tape.adjoint(h.node), c.gh, 1e-13 * std::fmax(1.0, std::fabs(c.gh)));
        EXPECT_NEAR(tape.adjoint(a.node), c.ga, 1e-13 * std::fmax(1.0, std::fabs(c.ga)));

        const std::uint32_t inputs[2] = {h.node, a.node};
        const double v[2] = {0.7, -0.4};
        double out[2] = {0.0, 0.0};
        ASSERT_TRUE(tape.hessianVector(f.node, inputs, 2, v, out));
        EXPECT_NEAR(out[0], c.hhh * v[0] + c.hha * v[1],
                    1e-12 * std::fmax(1.0, std::fabs(c.hhh)) +
                        1e-12 * std::fmax(1.0, std::fabs(c.hha)));
        EXPECT_NEAR(out[1], c.hha * v[0] + c.haa * v[1],
                    1e-12 * std::fmax(1.0, std::fabs(c.haa)) +
                        1e-12 * std::fmax(1.0, std::fabs(c.hha)));
    }
}

TEST_F(P2FunctionsTest, OwensTTangent) {
    // mpmath reference at (h, a) = (1.5, 0.8).
    const double value = 0.02885096725414943268;
    const double gh = -0.049855250811763152317;
    const double ga = 0.015335677410530725577;
    const D2 h(1.5, {1.0, 0.0});
    const D2 a(0.8, {0.0, 1.0});
    const D2 f = lightape::owensT(h, a);
    EXPECT_NEAR(f.value, value, 1e-14);
    EXPECT_NEAR(f.d[0], gh, 1e-14);
    EXPECT_NEAR(f.d[1], ga, 1e-14);
}

} // namespace
