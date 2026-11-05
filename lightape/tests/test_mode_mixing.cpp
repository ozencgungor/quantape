#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>

#include "lightape/lightape.h"
#include "lightape/mix/FwdRev.h"
#include <gtest/gtest.h>

namespace {

using lightape::FwdRevScalar;
using lightape::RevScalar;
using lightape::RevTape;
using lightape::Tangent;

using Plain2 = Tangent<double, 2>;
using Mixed2 = Tangent<RevScalar, 2>;

class ModeMixingTest : public ::testing::Test {
protected:
    void SetUp() override { RevTape::active().clear(); }
    void TearDown() override { RevTape::active().clear(); }
};

/// Compare every (value, tangent) component of a mixed operation with the
/// same expression evaluated on plain Tangents. The formulas are the same
/// doubles, but the RevScalar path keeps the expression tree inside tape
/// operators, so FMA contraction and the `pow` formulation can differ in the
/// last ULP: compare to a tight tolerance instead of bitwise.
void ExpectSame(const char* label, const Plain2& plain, const Mixed2& mixed) {
    SCOPED_TRACE(label);
    const auto tol = [](double v) { return 1e-14 * (1.0 + std::fabs(v)); };
    EXPECT_NEAR(mixed.value.value, plain.value, tol(plain.value));
    EXPECT_NEAR(mixed.d[0].value, plain.d[0], tol(plain.d[0]));
    EXPECT_NEAR(mixed.d[1].value, plain.d[1], tol(plain.d[1]));
}

/// Stable log-sum-exp over `FwdRevScalar<N>` values, generic over the scalar
/// interface (ADL math, primal-pinned `fmax`).
template <std::size_t N>
FwdRevScalar<N> LogSumExpMixed(const FwdRevScalar<N>* xs, std::size_t n) {
    using T = FwdRevScalar<N>;
    T m = xs[0];
    for (std::size_t i = 1; i < n; ++i) {
        m = fmax(m, xs[i]);
    }
    T z = exp(xs[0] - m);
    for (std::size_t i = 1; i < n; ++i) {
        z = z + exp(xs[i] - m);
    }
    return m + log(z);
}

TEST_F(ModeMixingTest, AdlResolvesToRevScalarOverloads) {
    RevTape& tape = RevTape::active();
    const RevScalar x(2.0, tape.input(2.0));
    using std::exp; // a std overload is visible; ADL must still pick lightape
    const std::size_t before = tape.size();
    const RevScalar y = exp(x);
    EXPECT_EQ(tape.size(), before + 1); // one Exp node, not a constant fold
    tape.resetAdjoints();
    tape.reverse(y.node);
    EXPECT_DOUBLE_EQ(tape.adjoint(x.node), std::exp(2.0));
}

/// `Tangent<RevScalar, 2>` scalar surface against `Tangent<double, 2>`:
/// arithmetic, mixed double/int literals, comparisons, fmin/fmax,
/// sqrt/exp/log/pow, trig, hyperbolic, erf/erfc, the elementary extras and
/// the primal predicates.
TEST_F(ModeMixingTest, ScalarOpsMatchPlainTangent) {
    RevTape& tape = RevTape::active();
    const double xv = 0.7;
    const double yv = 1.3;
    const Plain2 xd(xv, {0.8, -0.6});
    const Plain2 yd(yv, {0.5, 0.25});
    const Mixed2 x(RevScalar(xv, tape.input(xv)), {0.8, -0.6});
    const Mixed2 y(RevScalar(yv, tape.input(yv)), {0.5, 0.25});

    // arithmetic + mixed literals
    ExpectSame("add", xd + yd, x + y);
    ExpectSame("sub", xd - yd, x - y);
    ExpectSame("mul", xd * yd, x * y);
    ExpectSame("div", xd / yd, x / y);
    ExpectSame("neg", -xd, -x);
    ExpectSame("add double rhs", xd + 0.5, x + 0.5);
    ExpectSame("add double lhs", 0.5 + xd, 0.5 + x);
    ExpectSame("sub double rhs", xd - 0.5, x - 0.5);
    ExpectSame("sub double lhs", 0.5 - xd, 0.5 - x);
    ExpectSame("mul double rhs", xd * 0.5, x * 0.5);
    ExpectSame("mul double lhs", 0.5 * xd, 0.5 * x);
    ExpectSame("div double rhs", xd / 0.5, x / 0.5);
    ExpectSame("div double lhs", 0.5 / xd, 0.5 / x);
    ExpectSame("mul int lhs", 3.0 * sin(xd), 3 * sin(x));

    Plain2 pd = xd;
    pd += yd;
    pd -= yd;
    pd *= yd;
    pd /= yd;
    pd += 1.0;
    pd *= 0.25;
    Mixed2 pm = x;
    pm += y;
    pm -= y;
    pm *= y;
    pm /= y;
    pm += 1.0;
    pm *= 0.25;
    ExpectSame("compound", pd, pm);

    // comparisons (homogeneous + mixed)
    EXPECT_EQ(x < y, xd < yd);
    EXPECT_EQ(x > y, xd > yd);
    EXPECT_EQ(x <= y, xd <= yd);
    EXPECT_EQ(x >= y, xd >= yd);
    EXPECT_EQ(x == x, xd == xd);
    EXPECT_EQ(x != y, xd != yd);
    EXPECT_EQ(x < 0.5, xd < 0.5);
    EXPECT_EQ(x > 0.5, xd > 0.5);
    EXPECT_EQ(x <= 0.7, xd <= 0.7);
    EXPECT_EQ(x >= 0.7, xd >= 0.7);
    EXPECT_EQ(x == 0.7, xd == 0.7);
    EXPECT_EQ(x != 0.7, xd != 0.7);
    EXPECT_EQ(0.5 < x, 0.5 < xd);
    EXPECT_EQ(0.5 > x, 0.5 > xd);
    EXPECT_EQ(0.7 <= x, 0.7 <= xd);
    EXPECT_EQ(0.7 >= x, 0.7 >= xd);
    EXPECT_EQ(0.7 == x, 0.7 == xd);
    EXPECT_EQ(0.7 != x, 0.7 != xd);

    // fmin/fmax, homogeneous and mixed
    ExpectSame("fmin", fmin(xd, yd), fmin(x, y));
    ExpectSame("fmax", fmax(xd, yd), fmax(x, y));
    ExpectSame("fmin rhs", fmin(xd, 1.0), fmin(x, 1.0));
    ExpectSame("fmax rhs", fmax(xd, 1.0), fmax(x, 1.0));
    ExpectSame("fmin lhs", fmin(1.0, xd), fmin(1.0, x));
    ExpectSame("fmax lhs", fmax(1.0, xd), fmax(1.0, x));

    // elementary functions
    ExpectSame("sqrt", sqrt(xd), sqrt(x));
    ExpectSame("exp", exp(xd), exp(x));
    ExpectSame("log", log(xd), log(x));
    ExpectSame("expm1", expm1(xd), expm1(x));
    ExpectSame("log1p", log1p(xd), log1p(x));
    ExpectSame("cbrt", cbrt(xd), cbrt(x));
    ExpectSame("exp2", exp2(xd), exp2(x));
    ExpectSame("log2", log2(xd), log2(x));
    ExpectSame("log10", log10(xd), log10(x));
    ExpectSame("pow both", pow(xd, yd), pow(x, y));
    ExpectSame("pow rhs double", pow(xd, 1.7), pow(x, 1.7));
    ExpectSame("pow lhs double", pow(1.7, xd), pow(1.7, x));
    ExpectSame("fabs", fabs(xd), fabs(x));
    ExpectSame("abs", abs(xd), abs(x));
    ExpectSame("copysign", copysign(xd, yd), copysign(x, y));
    ExpectSame("copysign rhs", copysign(xd, -1.0), copysign(x, -1.0));
    ExpectSame("copysign lhs", copysign(-1.0, xd), copysign(-1.0, x));
    ExpectSame("hypot", hypot(xd, yd), hypot(x, y));
    ExpectSame("hypot rhs", hypot(xd, 2.0), hypot(x, 2.0));
    ExpectSame("hypot lhs", hypot(2.0, xd), hypot(2.0, x));
    ExpectSame("atan2", atan2(xd, yd), atan2(x, y));
    ExpectSame("atan2 rhs", atan2(xd, 2.0), atan2(x, 2.0));
    ExpectSame("atan2 lhs", atan2(2.0, xd), atan2(2.0, x));
    ExpectSame("fma", fma(xd, yd, xd), fma(x, y, x));

    // trig / hyperbolic / error
    ExpectSame("sin", sin(xd), sin(x));
    ExpectSame("cos", cos(xd), cos(x));
    ExpectSame("tan", tan(xd), tan(x));
    ExpectSame("asin", asin(xd), asin(x));
    ExpectSame("acos", acos(xd), acos(x));
    ExpectSame("atan", atan(xd), atan(x));
    ExpectSame("sinh", sinh(xd), sinh(x));
    ExpectSame("cosh", cosh(xd), cosh(x));
    ExpectSame("tanh", tanh(xd), tanh(x));
    ExpectSame("erf", erf(xd), erf(x));
    ExpectSame("erfc", erfc(xd), erfc(x));

    Plain2 ps;
    Plain2 pc;
    sincos(xd, ps, pc);
    Mixed2 ms;
    Mixed2 mc;
    sincos(x, ms, mc);
    ExpectSame("sincos sin", ps, ms);
    ExpectSame("sincos cos", pc, mc);

    // primal predicates
    EXPECT_EQ(isnan(x), isnan(xd));
    EXPECT_EQ(isinf(x), isinf(xd));
    EXPECT_EQ(isfinite(x), isfinite(xd));

    // the mixed recording is live on the tape (ADL wired to RevScalar)
    EXPECT_NE(fabs(x).value.node, RevTape::kNone);
    EXPECT_NE(x.value.node, RevTape::kNone);
}

/// f = exp(x y) + log(x) - 3 sin(y): value, gradient and the 2-direction HVP
/// matrix (out[k] = H v_k) against the analytic Hessian and against the
/// tape's replay-based `hessianVector` segment by segment.
TEST_F(ModeMixingTest, ExpLogSinHvpMatchesAnalyticAndReplay) {
    constexpr std::size_t kDirs = 2;
    RevTape& tape = RevTape::active();
    const double xv = 0.7;
    const double yv = 1.3;
    const double dirs[kDirs][2] = {{0.8, -0.6}, {0.3, 0.9}};

    const FwdRevScalar<kDirs> x(RevScalar(xv, tape.input(xv)), {dirs[0][0], dirs[1][0]});
    const FwdRevScalar<kDirs> y(RevScalar(yv, tape.input(yv)), {dirs[0][1], dirs[1][1]});
    const FwdRevScalar<kDirs> f = exp(x * y) + log(x) - 3 * sin(y);

    const double exy = std::exp(xv * yv);
    EXPECT_DOUBLE_EQ(f.value.value, exy + std::log(xv) - 3.0 * std::sin(yv));

    const std::uint32_t inputs[2] = {x.value.node, y.value.node};
    const std::size_t recorded = tape.size();
    double mixed[2 * kDirs] = {};
    lightape::reverseHvps(f, 2, inputs, mixed);
    EXPECT_EQ(tape.size(), recorded); // sweeps record nothing

    const double h[2][2] = {{yv * yv * exy - 1.0 / (xv * xv), exy * (1.0 + xv * yv)},
                            {exy * (1.0 + xv * yv), xv * xv * exy + 3.0 * std::sin(yv)}};
    for (std::size_t k = 0; k < kDirs; ++k) {
        for (std::size_t i = 0; i < 2; ++i) {
            const double expected = h[i][0] * dirs[k][0] + h[i][1] * dirs[k][1];
            EXPECT_NEAR(mixed[k * 2 + i], expected, 1e-12) << "k=" << k << " i=" << i;
        }
        const double dir[2] = {dirs[k][0], dirs[k][1]};
        double replay[2] = {};
        ASSERT_TRUE(lightape::hessianVector(tape, f.value.node, inputs, 2, dir, replay));
        for (std::size_t i = 0; i < 2; ++i) {
            EXPECT_NEAR(mixed[k * 2 + i], replay[i], 1e-12) << "k=" << k << " i=" << i;
        }
    }

    // gradient of the mixed value recording
    tape.resetAdjoints();
    tape.reverse(f.value.node);
    EXPECT_NEAR(tape.adjoint(inputs[0]), yv * exy + 1.0 / xv, 1e-14);
    EXPECT_NEAR(tape.adjoint(inputs[1]), xv * exy - 3.0 * std::cos(yv), 1e-14);
}

/// Softmax objective f = sum_i w_i softmax(x)_i: value, gradient and HVPs
/// (n = 4, two directions) against the analytic Hessian and replay.
TEST_F(ModeMixingTest, SoftmaxObjectiveHvpMatchesAnalyticAndReplay) {
    constexpr std::size_t n = 4;
    constexpr std::size_t kDirs = 2;
    RevTape& tape = RevTape::active();
    const double values[n] = {-0.5, 0.25, 1.5, 0.75};
    const double w[n] = {2.0, -1.0, 0.5, 3.0};
    const double dirs[kDirs][n] = {{0.4, -0.3, 0.9, 0.1}, {0.2, 0.7, -0.5, 0.3}};

    FwdRevScalar<kDirs> xs[n];
    std::uint32_t inputs[n];
    for (std::size_t i = 0; i < n; ++i) {
        std::array<RevScalar, kDirs> di;
        for (std::size_t k = 0; k < kDirs; ++k) {
            di[k] = dirs[k][i];
        }
        xs[i] = FwdRevScalar<kDirs>(RevScalar(values[i], tape.input(values[i])), di);
        inputs[i] = xs[i].value.node;
    }
    const FwdRevScalar<kDirs> lse = LogSumExpMixed(xs, n);
    FwdRevScalar<kDirs> f(RevScalar(0.0));
    for (std::size_t i = 0; i < n; ++i) {
        f = f + exp(xs[i] - lse) * w[i];
    }

    double m = values[0];
    for (std::size_t i = 1; i < n; ++i) {
        m = std::max(m, values[i]);
    }
    double z = 0.0;
    for (std::size_t i = 0; i < n; ++i) {
        z += std::exp(values[i] - m);
    }
    double o[n];
    for (std::size_t i = 0; i < n; ++i) {
        o[i] = std::exp(values[i] - m) / z;
    }
    double s = 0.0;
    for (std::size_t i = 0; i < n; ++i) {
        s += w[i] * o[i];
    }
    EXPECT_NEAR(f.value.value, s, 1e-14);

    tape.resetAdjoints();
    tape.reverse(f.value.node);
    for (std::size_t i = 0; i < n; ++i) {
        EXPECT_NEAR(tape.adjoint(inputs[i]), o[i] * (w[i] - s), 1e-13) << "i=" << i;
    }

    double mixed[n * kDirs] = {};
    lightape::reverseHvps(f, n, inputs, mixed);
    for (std::size_t k = 0; k < kDirs; ++k) {
        for (std::size_t i = 0; i < n; ++i) {
            double expected = 0.0;
            for (std::size_t j = 0; j < n; ++j) {
                const double delta = (i == j) ? 1.0 : 0.0;
                const double hij = (w[i] - s) * o[i] * (delta - o[j]) - o[i] * o[j] * (w[j] - s);
                expected += hij * dirs[k][j];
            }
            EXPECT_NEAR(mixed[k * n + i], expected, 1e-12) << "k=" << k << " i=" << i;
        }
        const double* dir = dirs[k];
        double replay[n] = {};
        ASSERT_TRUE(lightape::hessianVector(tape, f.value.node, inputs, n, dir, replay));
        for (std::size_t i = 0; i < n; ++i) {
            EXPECT_NEAR(mixed[k * n + i], replay[i], 1e-12) << "k=" << k << " i=" << i;
        }
    }
}

/// A function linear along the seeded directions has zero curvature: its HVP
/// rows are exactly zero and the value/gradient readout still works. A
/// constant function exercises the node-less tangent path of `reverseHvps`.
TEST_F(ModeMixingTest, LinearFunctionAndZeroDirectionsGiveZeroRows) {
    constexpr std::size_t kDirs = 2;
    RevTape& tape = RevTape::active();
    const FwdRevScalar<kDirs> x(RevScalar(1.5, tape.input(1.5)), {1.0, 0.0});
    const FwdRevScalar<kDirs> y(RevScalar(-0.5, tape.input(-0.5)), {0.0, 0.0});
    const FwdRevScalar<kDirs> f = 2.0 * x + 3.0 * y;

    EXPECT_DOUBLE_EQ(f.value.value, 2.0 * 1.5 + 3.0 * -0.5);

    const std::uint32_t inputs[2] = {x.value.node, y.value.node};
    double out[2 * kDirs] = {-1.0, -1.0, -1.0, -1.0};
    lightape::reverseHvps(f, 2, inputs, out);
    for (std::size_t k = 0; k < kDirs; ++k) {
        for (std::size_t i = 0; i < 2; ++i) {
            EXPECT_DOUBLE_EQ(out[k * 2 + i], 0.0);
        }
    }

    tape.resetAdjoints();
    tape.reverse(f.value.node);
    EXPECT_DOUBLE_EQ(tape.adjoint(inputs[0]), 2.0);
    EXPECT_DOUBLE_EQ(tape.adjoint(inputs[1]), 3.0);

    // Node-less tangents (a constant value) write zero rows directly.
    const FwdRevScalar<kDirs> c(RevScalar(3.0));
    EXPECT_EQ(c.d[0].node, RevTape::kNone);
    EXPECT_EQ(c.d[1].node, RevTape::kNone);
    double zeros[2 * kDirs] = {-1.0, -1.0, -1.0, -1.0};
    lightape::reverseHvps(c, 2, inputs, zeros);
    for (std::size_t k = 0; k < kDirs; ++k) {
        for (std::size_t i = 0; i < 2; ++i) {
            EXPECT_DOUBLE_EQ(zeros[k * 2 + i], 0.0);
        }
    }
}

} // namespace
