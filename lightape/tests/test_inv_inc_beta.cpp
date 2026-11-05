#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>

#include "lightape/ops/P2Quantiles.h"
#include <gtest/gtest.h>

namespace {

using lightape::RevScalar;
using lightape::RevTape;
using lightape::Tangent;

using D1 = Tangent<double, 1>;
using D2 = Tangent<double, 2>;
using D3 = Tangent<double, 3>;

class P2QuantilesTest : public ::testing::Test {
protected:
    void SetUp() override { RevTape::active().clear(); }
    void TearDown() override { RevTape::active().clear(); }
};

// ── round trip and symmetry ──

struct RoundTripCase {
    double a;
    double b;
    double p;
};

// Includes p near 0 and p near 1 and the a,b < 1 cases.
const RoundTripCase kRoundTrip[] = {
    {2.0, 3.0, 0.4},  {5.0, 2.0, 0.7},  {0.5, 0.5, 0.25}, {0.5, 0.5, 1e-8},
    {0.5, 0.5, 0.99}, {0.9, 0.4, 0.99}, {1.5, 1.5, 0.5},  {2.0, 3.0, 0.999},
};

TEST_F(P2QuantilesTest, RoundTrip) {
    for (const RoundTripCase& c : kRoundTrip) {
        SCOPED_TRACE(::testing::Message() << "a=" << c.a << " b=" << c.b << " p=" << c.p);
        const double x = lightape::invIncBeta(c.a, c.b, c.p);
        EXPECT_GT(x, 0.0);
        EXPECT_LT(x, 1.0);
        EXPECT_NEAR(lightape::incBeta(c.a, c.b, x), c.p, 1e-13);
    }
}

TEST_F(P2QuantilesTest, Symmetry) {
    for (const RoundTripCase& c : kRoundTrip) {
        SCOPED_TRACE(::testing::Message() << "a=" << c.a << " b=" << c.b << " p=" << c.p);
        const double x = lightape::invIncBeta(c.a, c.b, c.p);
        const double mirrored = 1.0 - lightape::invIncBeta(c.b, c.a, 1.0 - c.p);
        EXPECT_NEAR(x, mirrored, 1e-14);
    }
}

TEST_F(P2QuantilesTest, EndpointsAndInvalid) {
    EXPECT_DOUBLE_EQ(lightape::invIncBeta(2.0, 3.0, 0.0), 0.0);
    EXPECT_DOUBLE_EQ(lightape::invIncBeta(2.0, 3.0, 1.0), 1.0);
    EXPECT_DOUBLE_EQ(lightape::invIncBeta(0.5, 0.5, 0.0), 0.0);
    EXPECT_DOUBLE_EQ(lightape::invIncBeta(0.5, 0.5, 1.0), 1.0);
    EXPECT_TRUE(std::isnan(lightape::invIncBeta(0.0, 3.0, 0.5)));
    EXPECT_TRUE(std::isnan(lightape::invIncBeta(2.0, -1.0, 0.5)));
    EXPECT_TRUE(std::isnan(lightape::invIncBeta(2.0, 3.0, -0.1)));
    EXPECT_TRUE(std::isnan(lightape::invIncBeta(2.0, 3.0, 1.1)));
    EXPECT_TRUE(
        std::isnan(lightape::invIncBeta(2.0, 3.0, std::numeric_limits<double>::quiet_NaN())));
}

// ── mpmath 50 dps references for four triples ──
// x = I^{-1}(p), ga = dx/da, gb = dx/db, gp = dx/dp = 1/pdf,
// haa..hpp = the (a,b,p) Hessian of x. Generated with mpmath 50 dps:
// x by bisection on `betainc(a,b,0,t,regularized=True) - p` (residual < 1e-40),
// ga/gb from the integral representation of I_a/I_b (quad, 50 dps) divided
// through by the pdf, gp = 1/pdf, and the second derivatives by central
// differences of those exact gradients (h = 1e-15 vs 1e-12 agree to ~1e-24
// relative or better except T4.haa at ~2e-12, still far below the test tol).

struct QuantCase {
    double a;
    double b;
    double p;
    double x;
    double ga;
    double gb;
    double gp;
    double haa;
    double hab;
    double hap;
    double hbb;
    double hbp;
    double hpp;
};

const QuantCase kQuant[] = {
    {2.0, 3.0, 0.4, 0.3291665033784078698985863, 0.1405809356268578103513672,
     -0.07885464884186805293555508, 0.5625662056172987289528769, -0.0552906555480505326964608,
     -0.01640017134035892042343594, 0.01119500147197011521110463, 0.0378452890130640026468092,
     -0.1010565919950889455940971, -0.01791611896901217369677814},
    {0.5, 0.5, 1e-8, 2.467401100272339451773016e-16, 1.803764050232258370074242e-14,
     -6.84108846385711501366232e-16, 4.93480220054467849767482e-8, 1.246119127627359189357898e-12,
     -4.945448336506701446935789e-14, 3.508832056453622579498854e-6, 3.520237322840197007347837e-15,
     -1.368217692771422696568834e-7, 4.93480220054467687418997},
    {0.9, 0.4, 0.99, 0.999988527766802107168453, 1.679914393773025931986368e-5,
     -3.293499555874015832841595e-4, 2.868055949241914893718535e-3, -4.720770626316992398650988e-5,
     4.728071813738308026602953e-4, -4.199755599077540325468074e-3, -7.805244166890068333761404e-3,
     0.07516721575909446984820436, -0.4302072172679524969095805},
    {0.3, 4.0, 0.6, 0.03784421933710745587616495, 0.2531536169029426234861871,
     -0.01010695169592476937453175, 0.2299896925739641433001455, 0.2344789516758189757421881,
     -0.06490621041018580207844934, 0.8982902288274894798391836, 5.381340283641398423373999e-3,
     -0.06016915606317711741444528, 1.143324597056783249954496},
};

TEST_F(P2QuantilesTest, DoubleValuesAgainstMpmath) {
    for (const QuantCase& c : kQuant) {
        SCOPED_TRACE(::testing::Message() << "a=" << c.a << " b=" << c.b << " p=" << c.p);
        const double x = lightape::invIncBeta(c.a, c.b, c.p);
        EXPECT_NEAR(x, c.x, 1e-14 * std::fabs(c.x));
    }
}

// ── reverse mode: gradient and exact Hessian ──

TEST_F(P2QuantilesTest, GradientWrtP) {
    for (const QuantCase& c : kQuant) {
        SCOPED_TRACE(::testing::Message() << "a=" << c.a << " b=" << c.b << " p=" << c.p);
        RevTape& tape = RevTape::active();
        RevScalar a(c.a, tape.input(c.a));
        RevScalar b(c.b, tape.input(c.b));
        RevScalar p(c.p, tape.input(c.p));
        const RevScalar f = lightape::invIncBeta(a, b, p);
        EXPECT_NEAR(f.value, c.x, 1e-14 * std::fabs(c.x));
        tape.reverse(f.node);
        EXPECT_NEAR(tape.adjoint(p.node), c.gp, 1e-11 * std::fabs(c.gp));
    }
}

TEST_F(P2QuantilesTest, GradientWrtShapes) {
    for (const QuantCase& c : kQuant) {
        SCOPED_TRACE(::testing::Message() << "a=" << c.a << " b=" << c.b << " p=" << c.p);
        RevTape& tape = RevTape::active();
        RevScalar a(c.a, tape.input(c.a));
        RevScalar b(c.b, tape.input(c.b));
        RevScalar p(c.p, tape.input(c.p));
        const RevScalar f = lightape::invIncBeta(a, b, p);
        tape.reverse(f.node);
        EXPECT_NEAR(tape.adjoint(a.node), c.ga, 1e-10 * std::fabs(c.ga));
        EXPECT_NEAR(tape.adjoint(b.node), c.gb, 1e-10 * std::fabs(c.gb));
    }
}

TEST_F(P2QuantilesTest, HessianVectorAgainstMpmath) {
    const double v[3] = {0.3, -0.5, 0.8};
    for (const QuantCase& c : kQuant) {
        SCOPED_TRACE(::testing::Message() << "a=" << c.a << " b=" << c.b << " p=" << c.p);
        RevTape& tape = RevTape::active();
        RevScalar a(c.a, tape.input(c.a));
        RevScalar b(c.b, tape.input(c.b));
        RevScalar p(c.p, tape.input(c.p));
        const RevScalar f = lightape::invIncBeta(a, b, p);

        const std::uint32_t inputs[3] = {a.node, b.node, p.node};
        double out[3] = {0.0, 0.0, 0.0};
        ASSERT_TRUE(tape.hessianVector(f.node, inputs, 3, v, out));
        const double expected[3] = {
            c.haa * v[0] + c.hab * v[1] + c.hap * v[2],
            c.hab * v[0] + c.hbb * v[1] + c.hbp * v[2],
            c.hap * v[0] + c.hbp * v[1] + c.hpp * v[2],
        };
        for (int k = 0; k < 3; ++k) {
            EXPECT_NEAR(out[k], expected[k], 1e-9 * std::fmax(1.0, std::fabs(expected[k])));
        }
    }
}

TEST_F(P2QuantilesTest, HessianVectorFailsLoudlyWithoutVhp) {
    RevTape& tape = RevTape::active();
    RevScalar a(2.0, tape.input(2.0));
    RevScalar b(3.0, tape.input(3.0));
    RevScalar p(0.4, tape.input(0.4));
    const RevScalar args[3] = {a, b, p};
    auto vjp = [](const void* /*payload*/, double seed, const lightape::ExtArg* arg,
                  std::uint32_t /*n*/, RevTape& t) noexcept {
        t.addAdjoint(arg[0].node, seed * 0.1405809356268578103513672);
        t.addAdjoint(arg[1].node, seed * -0.07885464884186805293555508);
        t.addAdjoint(arg[2].node, seed * 0.5625662056172987289528769);
    };
    const RevScalar f = lightape::makeVjpVar(0.3291665033784078698985863, args, 3, vjp);

    const std::uint32_t inputs[3] = {a.node, b.node, p.node};
    const double v[3] = {1.0, 1.0, 1.0};
    double out[3] = {-99.0, -99.0, -99.0};
    EXPECT_FALSE(tape.hessianVector(f.node, inputs, 3, v, out));
    EXPECT_DOUBLE_EQ(out[0], -99.0);
}

// ── forward mode ──

TEST_F(P2QuantilesTest, TangentDirectionalDerivatives) {
    for (const QuantCase& c : kQuant) {
        SCOPED_TRACE(::testing::Message() << "a=" << c.a << " b=" << c.b << " p=" << c.p);
        const D3 a(c.a, {1.0, 0.0, 0.0});
        const D3 b(c.b, {0.0, 1.0, 0.0});
        const D3 p(c.p, {0.0, 0.0, 1.0});
        const D3 f = lightape::invIncBeta(a, b, p);
        EXPECT_NEAR(f.value, c.x, 1e-14 * std::fabs(c.x));
        EXPECT_NEAR(f.d[0], c.ga, 1e-10 * std::fabs(c.ga));
        EXPECT_NEAR(f.d[1], c.gb, 1e-10 * std::fabs(c.gb));
        EXPECT_NEAR(f.d[2], c.gp, 1e-11 * std::fabs(c.gp));

        const D1 aa(c.a, {0.3});
        const D1 bb(c.b, {-0.5});
        const D1 pp(c.p, {0.8});
        const D1 g = lightape::invIncBeta(aa, bb, pp);
        EXPECT_NEAR(g.value, c.x, 1e-14 * std::fabs(c.x));
        EXPECT_NEAR(g.d[0], c.ga * 0.3 + c.gb * -0.5 + c.gp * 0.8,
                    1e-10 * std::fmax(1.0, std::fabs(g.d[0])));
    }
}

TEST_F(P2QuantilesTest, TangentNestedLevels) {
    // The first-derivative factors are locally constant on nested levels, so
    // the inner tangent of Tangent<Tangent<double,2>,1> is not differentiated
    // (the RevScalar block carries the true Hessian).
    const D2 a(2.0, {1.0, 0.0});
    const D2 b(3.0, {0.0, 1.0});
    const D2 p(0.4, {0.0, 0.0});
    const D2 f = lightape::invIncBeta(a, b, p);
    EXPECT_NEAR(f.value, kQuant[0].x, 1e-15);
    EXPECT_NEAR(f.d[0], kQuant[0].ga, 1e-12 * std::fabs(kQuant[0].ga));
    EXPECT_NEAR(f.d[1], kQuant[0].gb, 1e-12 * std::fabs(kQuant[0].gb));
}

// ── guess handling ──

TEST_F(P2QuantilesTest, GuessIsHonored) {
    // Probe: the seed used by the public API is the built-in one.
    const auto bySeed = lightape::detail::invIncBetaSolve(2.0, 3.0, 0.4, -1.0);
    const auto byExactGuess = lightape::detail::invIncBetaSolve(2.0, 3.0, 0.4, kQuant[0].x);
    EXPECT_GT(bySeed.iterations, 0);
    EXPECT_LT(byExactGuess.iterations, bySeed.iterations);
    EXPECT_NEAR(byExactGuess.iterations, 1, 0);
    EXPECT_NEAR(byExactGuess.x, kQuant[0].x, 1e-15 * std::fabs(kQuant[0].x));

    // The public API returns the same root with or without a good guess, also
    // through the symmetry path (p > 1/2).
    EXPECT_NEAR(lightape::invIncBeta(2.0, 3.0, 0.4, kQuant[0].x),
                lightape::invIncBeta(2.0, 3.0, 0.4), 1e-14 * std::fabs(kQuant[0].x));
    EXPECT_NEAR(lightape::invIncBeta(0.9, 0.4, 0.99, kQuant[2].x),
                lightape::invIncBeta(0.9, 0.4, 0.99), 1e-13 * std::fabs(kQuant[2].x));

    // A tiny-tail case where the seed needs many steps: the exact root guess
    // still lands in one evaluation.
    const auto hardSeed = lightape::detail::invIncBetaSolve(0.5, 0.5, 1e-8, -1.0);
    const auto hardGuess = lightape::detail::invIncBetaSolve(0.5, 0.5, 1e-8, kQuant[1].x);
    EXPECT_LT(hardGuess.iterations, hardSeed.iterations);
    EXPECT_NEAR(hardGuess.x, kQuant[1].x, 1e-13 * std::fabs(kQuant[1].x));
}

} // namespace
