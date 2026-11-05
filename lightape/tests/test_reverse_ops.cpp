#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>

#include "lightape/lightape.h"
#include <gtest/gtest.h>

namespace {

using lightape::RevScalar;
using lightape::RevTape;
using lightape::Tangent;

/// std libm tolerance: 1e-13 relative, 1e-15 absolute near zero.
double LibmTol(double expected) {
    return 1e-15 + 1e-13 * std::abs(expected);
}

class ReverseOpsTest : public ::testing::Test {
protected:
    void SetUp() override { RevTape::active().clear(); }
    void TearDown() override { RevTape::active().clear(); }
};

/// Reverse gradient of a unary op against its analytic derivative.
template <typename F, typename DF>
void CheckUnaryGradient(const char* label, double x, const F& f, const DF& analytic) {
    SCOPED_TRACE(::testing::Message() << label << " at x=" << x);
    RevTape& tape = RevTape::active();
    RevScalar xv(x, tape.input(x));
    RevScalar y = f(xv);
    ASSERT_NE(y.node, RevTape::kNone);
    tape.reverse(y.node);
    const double expected = analytic(x);
    EXPECT_NEAR(tape.adjoint(xv.node), expected, LibmTol(expected));
}

/// Reverse gradients of a binary op against both analytic partials.
template <typename F, typename DFA, typename DFB>
void CheckBinaryGradient(const char* label, double a, double b, const F& f, const DFA& dfa,
                         const DFB& dfb) {
    SCOPED_TRACE(::testing::Message() << label << " at (" << a << ", " << b << ")");
    RevTape& tape = RevTape::active();
    RevScalar av(a, tape.input(a));
    RevScalar bv(b, tape.input(b));
    RevScalar out = f(av, bv);
    ASSERT_NE(out.node, RevTape::kNone);
    tape.reverse(out.node);
    const double ea = dfa(a, b);
    const double eb = dfb(a, b);
    EXPECT_NEAR(tape.adjoint(av.node), ea, LibmTol(ea));
    EXPECT_NEAR(tape.adjoint(bv.node), eb, LibmTol(eb));
}

TEST_F(ReverseOpsTest, Expm1Log1pGradients) {
    CheckUnaryGradient(
        "expm1", 0.37, [](const RevScalar& x) { return expm1(x); },
        [](double x) { return std::exp(x); });
    CheckUnaryGradient(
        "expm1", -0.8, [](const RevScalar& x) { return expm1(x); },
        [](double x) { return std::exp(x); });
    CheckUnaryGradient(
        "log1p", 0.42, [](const RevScalar& x) { return log1p(x); },
        [](double x) { return 1.0 / (1.0 + x); });
    CheckUnaryGradient(
        "log1p", -0.4, [](const RevScalar& x) { return log1p(x); },
        [](double x) { return 1.0 / (1.0 + x); });
}

TEST_F(ReverseOpsTest, PowBothPartials) {
    const auto dfx = [](double x, double y) { return y * std::pow(x, y - 1.0); };
    const auto dfy = [](double x, double y) { return std::pow(x, y) * std::log(x); };
    CheckBinaryGradient(
        "pow(x, y)", 1.3, 2.4, [](const RevScalar& x, const RevScalar& y) { return pow(x, y); },
        dfx, dfy);
    CheckBinaryGradient(
        "pow(x, y)", 2.5, -0.7, [](const RevScalar& x, const RevScalar& y) { return pow(x, y); },
        dfx, dfy);
}

TEST_F(ReverseOpsTest, SinCosTanGradients) {
    CheckUnaryGradient(
        "sin", 0.3, [](const RevScalar& x) { return sin(x); },
        [](double x) { return std::cos(x); });
    CheckUnaryGradient(
        "sin", -1.1, [](const RevScalar& x) { return sin(x); },
        [](double x) { return std::cos(x); });
    CheckUnaryGradient(
        "cos", 0.7, [](const RevScalar& x) { return cos(x); },
        [](double x) { return -std::sin(x); });
    CheckUnaryGradient(
        "cos", 2.5, [](const RevScalar& x) { return cos(x); },
        [](double x) { return -std::sin(x); });
    CheckUnaryGradient(
        "tan", 0.4, [](const RevScalar& x) { return tan(x); },
        [](double x) { return 1.0 + std::tan(x) * std::tan(x); });
    CheckUnaryGradient(
        "tan", -0.9, [](const RevScalar& x) { return tan(x); },
        [](double x) { return 1.0 + std::tan(x) * std::tan(x); });
}

TEST_F(ReverseOpsTest, InverseTrigGradients) {
    CheckUnaryGradient(
        "asin", 0.3, [](const RevScalar& x) { return asin(x); },
        [](double x) { return 1.0 / std::sqrt(1.0 - x * x); });
    CheckUnaryGradient(
        "asin", -0.6, [](const RevScalar& x) { return asin(x); },
        [](double x) { return 1.0 / std::sqrt(1.0 - x * x); });
    CheckUnaryGradient(
        "acos", 0.3, [](const RevScalar& x) { return acos(x); },
        [](double x) { return -1.0 / std::sqrt(1.0 - x * x); });
    CheckUnaryGradient(
        "acos", -0.5, [](const RevScalar& x) { return acos(x); },
        [](double x) { return -1.0 / std::sqrt(1.0 - x * x); });
    CheckUnaryGradient(
        "atan", 0.7, [](const RevScalar& x) { return atan(x); },
        [](double x) { return 1.0 / (1.0 + x * x); });
    CheckUnaryGradient(
        "atan", -2.0, [](const RevScalar& x) { return atan(x); },
        [](double x) { return 1.0 / (1.0 + x * x); });
}

TEST_F(ReverseOpsTest, HyperbolicGradients) {
    CheckUnaryGradient(
        "sinh", 0.6, [](const RevScalar& x) { return sinh(x); },
        [](double x) { return std::cosh(x); });
    CheckUnaryGradient(
        "sinh", -1.3, [](const RevScalar& x) { return sinh(x); },
        [](double x) { return std::cosh(x); });
    CheckUnaryGradient(
        "cosh", 0.6, [](const RevScalar& x) { return cosh(x); },
        [](double x) { return std::sinh(x); });
    CheckUnaryGradient(
        "cosh", -1.3, [](const RevScalar& x) { return cosh(x); },
        [](double x) { return std::sinh(x); });
    CheckUnaryGradient(
        "tanh", 0.4, [](const RevScalar& x) { return tanh(x); },
        [](double x) { return 1.0 - std::tanh(x) * std::tanh(x); });
    CheckUnaryGradient(
        "tanh", -1.5, [](const RevScalar& x) { return tanh(x); },
        [](double x) { return 1.0 - std::tanh(x) * std::tanh(x); });
}

TEST_F(ReverseOpsTest, ErfErfcGradients) {
    const auto d_erf = [](double x) { return lightape::kTwoOverSqrtPi * std::exp(-x * x); };
    const auto d_erfc = [](double x) { return -lightape::kTwoOverSqrtPi * std::exp(-x * x); };
    CheckUnaryGradient("erf", 0.0, [](const RevScalar& x) { return lightape::erf(x); }, d_erf);
    CheckUnaryGradient("erf", 0.5, [](const RevScalar& x) { return lightape::erf(x); }, d_erf);
    CheckUnaryGradient("erf", -1.2, [](const RevScalar& x) { return lightape::erf(x); }, d_erf);
    CheckUnaryGradient("erfc", -1.0, [](const RevScalar& x) { return lightape::erfc(x); }, d_erfc);
    CheckUnaryGradient("erfc", 0.3, [](const RevScalar& x) { return lightape::erfc(x); }, d_erfc);
    CheckUnaryGradient("erfc", 2.0, [](const RevScalar& x) { return lightape::erfc(x); }, d_erfc);
}

TEST_F(ReverseOpsTest, LogGammaTGammaGradients) {
    const double x0 = 3.7;
    RevTape& tape = RevTape::active();

    RevScalar x(x0, tape.input(x0));
    RevScalar lg = lightape::lgamma(x);
    ASSERT_NE(lg.node, RevTape::kNone);
    EXPECT_DOUBLE_EQ(lg.value, std::lgamma(x0));
    tape.reverse(lg.node);
    EXPECT_DOUBLE_EQ(tape.adjoint(x.node), lightape::digamma(x0));

    RevScalar y(x0, tape.input(x0));
    RevScalar tg = lightape::tgamma(y);
    ASSERT_NE(tg.node, RevTape::kNone);
    EXPECT_DOUBLE_EQ(tg.value, std::tgamma(x0));
    tape.reverse(tg.node);
    EXPECT_DOUBLE_EQ(tape.adjoint(y.node), std::tgamma(x0) * lightape::digamma(x0));
}

TEST_F(ReverseOpsTest, HypotAtan2Gradients) {
    CheckBinaryGradient(
        "hypot", 3.0, -4.0, [](const RevScalar& a, const RevScalar& b) { return hypot(a, b); },
        [](double a, double b) { return a / std::hypot(a, b); },
        [](double a, double b) { return b / std::hypot(a, b); });
    CheckBinaryGradient(
        "hypot", 0.7, 1.9, [](const RevScalar& a, const RevScalar& b) { return hypot(a, b); },
        [](double a, double b) { return a / std::hypot(a, b); },
        [](double a, double b) { return b / std::hypot(a, b); });
    // atan2(y, x): the first argument is the numerator.
    CheckBinaryGradient(
        "atan2(y, x)", 1.2, -0.7,
        [](const RevScalar& y, const RevScalar& x) { return atan2(y, x); },
        [](double y, double x) { return x / (x * x + y * y); },
        [](double y, double x) { return -y / (x * x + y * y); });
    CheckBinaryGradient(
        "atan2(y, x)", -0.8, 1.1,
        [](const RevScalar& y, const RevScalar& x) { return atan2(y, x); },
        [](double y, double x) { return x / (x * x + y * y); },
        [](double y, double x) { return -y / (x * x + y * y); });
}

TEST_F(ReverseOpsTest, AbsSignGradients) {
    CheckUnaryGradient(
        "abs", 2.5, [](const RevScalar& x) { return lightape::abs(x); },
        [](double) { return 1.0; });
    CheckUnaryGradient(
        "abs", -2.5, [](const RevScalar& x) { return lightape::abs(x); },
        [](double) { return -1.0; });
    CheckUnaryGradient(
        "fabs", 0.25, [](const RevScalar& x) { return lightape::fabs(x); },
        [](double) { return 1.0; });
    CheckUnaryGradient(
        "fabs", -0.25, [](const RevScalar& x) { return lightape::fabs(x); },
        [](double) { return -1.0; });
}

TEST_F(ReverseOpsTest, FminFmaxShareWinnerNode) {
    RevTape& tape = RevTape::active();
    RevScalar a(1.0, tape.input(1.0));
    RevScalar b(2.0, tape.input(2.0));
    RevScalar lo = lightape::fmin(a, b);
    RevScalar hi = lightape::fmax(a, b);
    EXPECT_EQ(lo.node, a.node);
    EXPECT_EQ(hi.node, b.node);
    tape.reverse((lo + hi).node);
    EXPECT_DOUBLE_EQ(tape.adjoint(a.node), 1.0);
    EXPECT_DOUBLE_EQ(tape.adjoint(b.node), 1.0);

    // Flip the winner: adjoints follow the taken branch.
    RevScalar p(3.0, tape.input(3.0));
    RevScalar q(2.0, tape.input(2.0));
    RevScalar g = lightape::fmin(p, q) * lightape::fmax(p, q);
    tape.reverse(g.node);
    EXPECT_DOUBLE_EQ(tape.adjoint(p.node), 2.0);
    EXPECT_DOUBLE_EQ(tape.adjoint(q.node), 3.0);
}

TEST_F(ReverseOpsTest, MinMaxOpcodeSweeps) {
    RevTape& tape = RevTape::active();
    RevScalar a(1.0, tape.input(1.0));
    RevScalar b(2.0, tape.input(2.0));
    RevScalar c(3.0, tape.input(3.0));
    RevScalar d(4.0, tape.input(4.0));

    const std::uint32_t minl =
        tape.push(lightape::RevOp::MinL, a.node, b.node, 1.0, a.value, b.value);
    tape.reverse(minl);
    EXPECT_DOUBLE_EQ(tape.adjoint(a.node), 1.0);
    EXPECT_DOUBLE_EQ(tape.adjoint(b.node), 0.0);

    tape.resetAdjoints();
    const std::uint32_t minr =
        tape.push(lightape::RevOp::MinR, c.node, d.node, 3.0, c.value, d.value);
    tape.reverse(minr);
    EXPECT_DOUBLE_EQ(tape.adjoint(c.node), 0.0);
    EXPECT_DOUBLE_EQ(tape.adjoint(d.node), 1.0);

    tape.resetAdjoints();
    const std::uint32_t maxl =
        tape.push(lightape::RevOp::MaxL, a.node, b.node, 2.0, a.value, b.value);
    tape.reverse(maxl);
    EXPECT_DOUBLE_EQ(tape.adjoint(a.node), 1.0);
    EXPECT_DOUBLE_EQ(tape.adjoint(b.node), 0.0);

    tape.resetAdjoints();
    const std::uint32_t maxr =
        tape.push(lightape::RevOp::MaxR, c.node, d.node, 4.0, c.value, d.value);
    tape.reverse(maxr);
    EXPECT_DOUBLE_EQ(tape.adjoint(c.node), 0.0);
    EXPECT_DOUBLE_EQ(tape.adjoint(d.node), 1.0);
}

TEST_F(ReverseOpsTest, NewOpsOnConstantsCreateNoNodes) {
    RevTape& tape = RevTape::active();
    RevScalar a(0.5);
    RevScalar b(1.5);

    EXPECT_EQ(sqrt(a).node, RevTape::kNone);
    EXPECT_EQ(exp(a).node, RevTape::kNone);
    EXPECT_EQ(log(a).node, RevTape::kNone);
    EXPECT_EQ(expm1(a).node, RevTape::kNone);
    EXPECT_EQ(log1p(a).node, RevTape::kNone);
    EXPECT_EQ(pow(a, b).node, RevTape::kNone);
    EXPECT_EQ(sin(a).node, RevTape::kNone);
    EXPECT_EQ(cos(a).node, RevTape::kNone);
    EXPECT_EQ(tan(a).node, RevTape::kNone);
    EXPECT_EQ(asin(a).node, RevTape::kNone);
    EXPECT_EQ(acos(a).node, RevTape::kNone);
    EXPECT_EQ(atan(a).node, RevTape::kNone);
    EXPECT_EQ(sinh(a).node, RevTape::kNone);
    EXPECT_EQ(cosh(a).node, RevTape::kNone);
    EXPECT_EQ(tanh(a).node, RevTape::kNone);
    EXPECT_EQ(lightape::erf(a).node, RevTape::kNone);
    EXPECT_EQ(lightape::erfc(a).node, RevTape::kNone);
    EXPECT_EQ(lightape::lgamma(b).node, RevTape::kNone);
    EXPECT_EQ(lightape::tgamma(b).node, RevTape::kNone);
    EXPECT_EQ(lightape::hypot(a, b).node, RevTape::kNone);
    EXPECT_EQ(lightape::atan2(a, b).node, RevTape::kNone);
    EXPECT_EQ(lightape::fabs(a).node, RevTape::kNone);
    EXPECT_EQ(lightape::abs(a).node, RevTape::kNone);
    EXPECT_EQ(lightape::fmin(a, b).node, RevTape::kNone);
    EXPECT_EQ(lightape::fmax(a, b).node, RevTape::kNone);
    EXPECT_EQ(tape.size(), 0u);
}

TEST_F(ReverseOpsTest, PrecomputedPartialsCollapseWhenConstant) {
    RevTape& tape = RevTape::active();
    const double x0 = 3.7;

    RevScalar cg = lightape::lgamma(RevScalar(x0));
    EXPECT_EQ(cg.node, RevTape::kNone);
    EXPECT_DOUBLE_EQ(cg.value, std::lgamma(x0));
    RevScalar ct = lightape::tgamma(RevScalar(x0));
    EXPECT_EQ(ct.node, RevTape::kNone);
    EXPECT_DOUBLE_EQ(ct.value, std::tgamma(x0));
    EXPECT_EQ(tape.position(), 1u);
    EXPECT_EQ(tape.size(), 0u);

    RevScalar x(x0, tape.input(x0));
    const std::size_t before = tape.position();
    RevScalar lg = lightape::lgamma(x);
    EXPECT_NE(lg.node, RevTape::kNone);
    EXPECT_EQ(tape.position(), before + 1u);
    tape.reverse(lg.node);
    EXPECT_DOUBLE_EQ(tape.adjoint(x.node), lightape::digamma(x0));

    tape.resetAdjoints();
    tape.resetTo(before);
    RevScalar tg = lightape::tgamma(x);
    EXPECT_NE(tg.node, RevTape::kNone);
    EXPECT_EQ(tape.position(), before + 1u);
    tape.reverse(tg.node);
    EXPECT_DOUBLE_EQ(tape.adjoint(x.node), std::tgamma(x0) * lightape::digamma(x0));
}

TEST_F(ReverseOpsTest, LifecycleResetAdjointsAndResetTo) {
    RevTape& tape = RevTape::active();
    const std::size_t beforeInputs = tape.position();
    EXPECT_EQ(beforeInputs, 1u);
    EXPECT_EQ(tape.size(), 0u);

    RevScalar x(1.3, tape.input(1.3));
    RevScalar y(0.7, tape.input(0.7));
    EXPECT_EQ(x.node, beforeInputs);
    EXPECT_EQ(y.node, beforeInputs + 1u);
    const std::size_t afterInputs = tape.position();
    EXPECT_EQ(afterInputs, beforeInputs + 2u);
    EXPECT_EQ(tape.size(), afterInputs - 1u);

    // Output A: full composite, one sweep, readout.
    RevScalar sx = sin(x);
    RevScalar ey = exp(y);
    RevScalar pw = pow(x, y);
    RevScalar ax = sx * ey;
    RevScalar a = ax + pw;
    tape.reverse(a.node);

    const double gax = std::cos(1.3) * std::exp(0.7) + 0.7 * std::pow(1.3, -0.3);
    const double gay = std::sin(1.3) * std::exp(0.7) + std::pow(1.3, 0.7) * std::log(1.3);
    EXPECT_NEAR(tape.adjoint(x.node), gax, LibmTol(gax));
    EXPECT_NEAR(tape.adjoint(y.node), gay, LibmTol(gay));

    // Consume-on-use: non-leaf adjoints are zeroed by the sweep...
    EXPECT_DOUBLE_EQ(tape.adjoint(a.node), 0.0);
    EXPECT_DOUBLE_EQ(tape.adjoint(ax.node), 0.0);
    EXPECT_DOUBLE_EQ(tape.adjoint(sx.node), 0.0);
    EXPECT_DOUBLE_EQ(tape.adjoint(ey.node), 0.0);
    EXPECT_DOUBLE_EQ(tape.adjoint(pw.node), 0.0);
    // ...while leaf adjoints stay readable.
    EXPECT_NE(tape.adjoint(x.node), 0.0);
    EXPECT_NE(tape.adjoint(y.node), 0.0);

    // Drop A's nodes but keep the input leaves, and clear stale adjoints.
    tape.resetAdjoints();
    tape.resetTo(afterInputs);
    EXPECT_EQ(tape.position(), afterInputs);
    EXPECT_EQ(tape.size(), afterInputs - 1u);
    EXPECT_DOUBLE_EQ(tape.value(x.node), 1.3);
    EXPECT_DOUBLE_EQ(tape.adjoint(x.node), 0.0);
    EXPECT_DOUBLE_EQ(tape.adjoint(y.node), 0.0);

    // Output B reuses the same input nodes.
    RevScalar b = x * y - x / y;
    tape.reverse(b.node);
    EXPECT_NEAR(tape.adjoint(x.node), 0.7 - 1.0 / 0.7, 1e-14);
    EXPECT_NEAR(tape.adjoint(y.node), 1.3 + 1.3 / (0.7 * 0.7), 1e-14);

    // Truncate below the inputs: recording restarts at beforeInputs and reuses
    // the freed indices (the sink at node 0 cannot be truncated away).
    tape.resetAdjoints();
    tape.resetTo(beforeInputs);
    EXPECT_EQ(tape.position(), beforeInputs);
    EXPECT_EQ(tape.size(), 0u);
    tape.resetTo(0);
    EXPECT_EQ(tape.position(), beforeInputs);
    EXPECT_DOUBLE_EQ(tape.value(0), 0.0);
    EXPECT_DOUBLE_EQ(tape.adjoint(0), 0.0);

    RevScalar x2(1.3, tape.input(1.3));
    RevScalar y2(0.7, tape.input(0.7));
    EXPECT_EQ(x2.node, x.node);
    EXPECT_EQ(y2.node, y.node);
    RevScalar c = x2 * y2 - x2 / y2;
    tape.reverse(c.node);
    EXPECT_NEAR(tape.adjoint(x2.node), 0.7 - 1.0 / 0.7, 1e-14);
    EXPECT_NEAR(tape.adjoint(y2.node), 1.3 + 1.3 / (0.7 * 0.7), 1e-14);
}

TEST_F(ReverseOpsTest, DifferentialAgainstForwardTangent) {
    constexpr double x0 = 1.3;
    constexpr double y0 = 0.7;
    constexpr double vx = 0.6;
    constexpr double vy = -0.4;

    RevTape& tape = RevTape::active();
    RevScalar x(x0, tape.input(x0));
    RevScalar y(y0, tape.input(y0));
    RevScalar f = sin(x) * exp(y) + pow(x, y);
    tape.reverse(f.node);
    const double gx = tape.adjoint(x.node);
    const double gy = tape.adjoint(y.node);

    Tangent<double, 2> tx(x0, {1.0, 0.0});
    Tangent<double, 2> ty(y0, {0.0, 1.0});
    Tangent<double, 2> ft = sin(tx) * exp(ty) + pow(tx, ty);

    const double directional = ft.d[0] * vx + ft.d[1] * vy;
    const double dot = gx * vx + gy * vy;
    EXPECT_NEAR(gx, ft.d[0], LibmTol(ft.d[0]));
    EXPECT_NEAR(gy, ft.d[1], LibmTol(ft.d[1]));
    EXPECT_NEAR(dot, directional, LibmTol(directional));
}

TEST_F(ReverseOpsTest, HypotAtOriginNoGradient) {
    RevTape& tape = RevTape::active();
    RevScalar x(0.0, tape.input(0.0));
    RevScalar y(0.0, tape.input(0.0));
    RevScalar h = lightape::hypot(x, y);
    ASSERT_NE(h.node, RevTape::kNone);
    EXPECT_EQ(tape.size(), 3u);
    EXPECT_DOUBLE_EQ(h.value, 0.0);
    tape.reverse(h.node);
    EXPECT_DOUBLE_EQ(tape.adjoint(x.node), 0.0);
    EXPECT_DOUBLE_EQ(tape.adjoint(y.node), 0.0);
}

TEST_F(ReverseOpsTest, Atan2AtOriginNoGradient) {
    RevTape& tape = RevTape::active();
    RevScalar y(0.0, tape.input(0.0));
    RevScalar x(0.0, tape.input(0.0));
    RevScalar a = lightape::atan2(y, x);
    ASSERT_NE(a.node, RevTape::kNone);
    EXPECT_EQ(tape.size(), 3u);
    EXPECT_DOUBLE_EQ(a.value, 0.0);
    tape.reverse(a.node);
    EXPECT_DOUBLE_EQ(tape.adjoint(y.node), 0.0);
    EXPECT_DOUBLE_EQ(tape.adjoint(x.node), 0.0);
}

TEST_F(ReverseOpsTest, AbsAtZeroUsesPositiveSign) {
    RevTape& tape = RevTape::active();
    RevScalar pz(0.0, tape.input(0.0));
    RevScalar nz(-0.0, tape.input(-0.0));
    RevScalar f = lightape::fabs(pz) + lightape::fabs(nz);
    tape.reverse(f.node);
    EXPECT_DOUBLE_EQ(tape.adjoint(pz.node), 1.0);
    EXPECT_DOUBLE_EQ(tape.adjoint(nz.node), 1.0);
}

TEST_F(ReverseOpsTest, PredicatesOnRevScalar) {
    RevTape& tape = RevTape::active();
    const double qnan = std::numeric_limits<double>::quiet_NaN();
    const double inf = std::numeric_limits<double>::infinity();
    RevScalar cnan(qnan);
    RevScalar cinf(inf);
    RevScalar cfin(-2.0);
    RevScalar fin(1.0, tape.input(1.0));

    EXPECT_TRUE(lightape::isnan(cnan));
    EXPECT_FALSE(lightape::isnan(cfin));
    EXPECT_FALSE(lightape::isnan(fin));
    EXPECT_TRUE(lightape::isinf(cinf));
    EXPECT_FALSE(lightape::isinf(cfin));
    EXPECT_FALSE(lightape::isinf(cnan));
    EXPECT_TRUE(lightape::isfinite(cfin));
    EXPECT_TRUE(lightape::isfinite(fin));
    EXPECT_FALSE(lightape::isfinite(cnan));
    EXPECT_FALSE(lightape::isfinite(cinf));
    EXPECT_EQ(tape.size(), 1u);
}

} // namespace
