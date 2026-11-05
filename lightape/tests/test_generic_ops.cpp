#include <cmath>
#include <utility>

#include "lightape/lightape.h"
#include "lightape/ops/Generic.h"
#include <gtest/gtest.h>

namespace {

using lightape::clamp;
using lightape::clamp01;
using lightape::cube;
using lightape::lerp;
using lightape::logit;
using lightape::logSumExp2;
using lightape::RevScalar;
using lightape::RevTape;
using lightape::safeDiv;
using lightape::sigmoid;
using lightape::signum;
using lightape::smoothAbs;
using lightape::smoothMax;
using lightape::softplus;
using lightape::sqr;
using lightape::Tangent;
using lightape::unitSign;

using D1 = Tangent<double, 1>;
using D2 = Tangent<D1, 1>;
using D3 = Tangent<D2, 1>;

double Sigmoid(double x) {
    return 1.0 / (1.0 + std::exp(-x));
}

// Third-order input jets: the value carries the inner unit tangent; the outer
// tangents carry unit values with zero inner tangents. Then f.d[0].d[0].d[0]
// is f''' and f.d[0].d[0].value is f''.
D2 Seed2(double v) {
    D2 x(D1(v, {1.0}));
    x.d[0] = D1(1.0);
    return x;
}
D2 Const2(double v) {
    D2 x{D1(v)};
    x.d[0] = D1(0.0);
    return x;
}
D3 Seed3(double v) {
    D3 x(Seed2(v));
    x.d[0] = Const2(1.0);
    return x;
}
D3 Const3(double v) {
    D3 x(Const2(v));
    x.d[0] = Const2(0.0);
    return x;
}

class GenericOpsRevTest : public ::testing::Test {
protected:
    void SetUp() override { RevTape::active().clear(); }
    void TearDown() override { RevTape::active().clear(); }

    /// d f / dx at x0 on a fresh tape (0 for a constant result).
    template <typename F>
    double grad(double x0, const F& f) {
        RevTape& tape = RevTape::active();
        tape.clear();
        RevScalar x(x0, tape.input(x0));
        const RevScalar y = f(x);
        if (y.node == RevTape::kNone) {
            return 0.0;
        }
        tape.reverse(y.node);
        return tape.adjoint(x.node);
    }

    /// (d f / dx, d f / dy) at (x0, y0) on a fresh tape.
    template <typename F>
    std::pair<double, double> grad2(double x0, double y0, const F& f) {
        RevTape& tape = RevTape::active();
        tape.clear();
        RevScalar x(x0, tape.input(x0));
        RevScalar y(y0, tape.input(y0));
        const RevScalar z = f(x, y);
        if (z.node == RevTape::kNone) {
            return {0.0, 0.0};
        }
        tape.reverse(z.node);
        return {tape.adjoint(x.node), tape.adjoint(y.node)};
    }
};

TEST(GenericOpsDoubleTest, ValuesMatchStdFormulas) {
    EXPECT_DOUBLE_EQ(clamp(0.25, 0.3, 0.9), 0.3);
    EXPECT_DOUBLE_EQ(clamp(0.5, 0.3, 0.9), 0.5);
    EXPECT_DOUBLE_EQ(clamp(0.95, 0.3, 0.9), 0.9);
    EXPECT_DOUBLE_EQ(clamp01(-0.5), 0.0);
    EXPECT_DOUBLE_EQ(clamp01(0.25), 0.25);
    EXPECT_DOUBLE_EQ(clamp01(1.5), 1.0);

    EXPECT_DOUBLE_EQ(signum(-2.0), -1.0);
    EXPECT_DOUBLE_EQ(signum(0.0), 0.0);
    EXPECT_DOUBLE_EQ(signum(3.0), 1.0);
    EXPECT_DOUBLE_EQ(unitSign(-1e-300), -1.0);
    EXPECT_DOUBLE_EQ(unitSign(0.0), 1.0);
    EXPECT_DOUBLE_EQ(unitSign(2.0), 1.0);

    EXPECT_DOUBLE_EQ(sqr(-1.5), 2.25);
    EXPECT_DOUBLE_EQ(cube(0.5), 0.125);

    EXPECT_DOUBLE_EQ(lerp(0.25, 0.75, 0.5), 0.5);
    EXPECT_DOUBLE_EQ(lerp(-1.0, 2.0, 0.25), -0.25);

    EXPECT_DOUBLE_EQ(safeDiv(3.0, 2.0, 1e-12), 1.5);
    EXPECT_DOUBLE_EQ(safeDiv(3.0, 0.0, 1e-12), 0.0);
    EXPECT_DOUBLE_EQ(safeDiv(3.0, 1e-13, 1e-12), 0.0);

    EXPECT_NEAR(softplus(0.7), std::log1p(std::exp(0.7)), 1e-15);
    EXPECT_DOUBLE_EQ(softplus(50.0), 50.0); // stable branch: no exp overflow
    EXPECT_NEAR(softplus(-50.0), std::exp(-50.0), 1e-30);

    EXPECT_NEAR(sigmoid(0.8), 1.0 / (1.0 + std::exp(-0.8)), 1e-15);
    EXPECT_NEAR(logit(0.3), std::log(0.3 / 0.7), 1e-15);
    EXPECT_NEAR(sigmoid(logit(0.42)), 0.42, 1e-14);

    EXPECT_NEAR(smoothMax(0.9, -0.2, 2.0),
                std::fmax(0.9, -0.2) + std::log1p(std::exp(-std::fabs(1.1) * 2.0)) / 2.0, 1e-15);
    EXPECT_NEAR(logSumExp2(0.9, -0.2), std::log(std::exp(0.9) + std::exp(-0.2)), 1e-15);
    EXPECT_DOUBLE_EQ(smoothAbs(0.7, 1e-3), std::sqrt(0.7 * 0.7 + 1e-6));
}

TEST_F(GenericOpsRevTest, ClampGradients) {
    const auto f = [](const RevScalar& x) { return clamp(x, RevScalar(0.3), RevScalar(0.9)); };
    EXPECT_DOUBLE_EQ(grad(0.5, f), 1.0);
    EXPECT_DOUBLE_EQ(grad(0.1, f), 0.0);
    EXPECT_DOUBLE_EQ(grad(0.95, f), 0.0);

    // The saturated branch returns the bound itself, so its adjoint is live.
    RevTape& tape = RevTape::active();
    tape.clear();
    RevScalar x(-1.0, tape.input(-1.0));
    RevScalar lo(0.3, tape.input(0.3));
    RevScalar hi(0.9, tape.input(0.9));
    const RevScalar y = clamp(x, lo, hi);
    ASSERT_EQ(y.node, lo.node);
    tape.reverse(y.node);
    EXPECT_DOUBLE_EQ(tape.adjoint(lo.node), 1.0);
    EXPECT_DOUBLE_EQ(tape.adjoint(x.node), 0.0);
    EXPECT_DOUBLE_EQ(tape.adjoint(hi.node), 0.0);
}

TEST_F(GenericOpsRevTest, Clamp01Gradients) {
    EXPECT_DOUBLE_EQ(grad(0.4, [](const RevScalar& x) { return clamp01(x); }), 1.0);
    EXPECT_DOUBLE_EQ(grad(-0.4, [](const RevScalar& x) { return clamp01(x); }), 0.0);
    EXPECT_DOUBLE_EQ(grad(1.4, [](const RevScalar& x) { return clamp01(x); }), 0.0);
}

TEST_F(GenericOpsRevTest, SignumAndUnitSignAreAeConstant) {
    // d/dx [x * sign(x)] = sign(x): the sign itself carries no derivative.
    EXPECT_DOUBLE_EQ(grad(2.0, [](const RevScalar& x) { return x * signum(x); }), 1.0);
    EXPECT_DOUBLE_EQ(grad(-3.0, [](const RevScalar& x) { return x * signum(x); }), -1.0);
    EXPECT_DOUBLE_EQ(grad(2.0, [](const RevScalar& x) { return x * unitSign(x); }), 1.0);
    EXPECT_DOUBLE_EQ(grad(-3.0, [](const RevScalar& x) { return x * unitSign(x); }), -1.0);

    RevTape& tape = RevTape::active();
    tape.clear();
    RevScalar x(2.0, tape.input(2.0));
    EXPECT_EQ(signum(x).node, RevTape::kNone);
    EXPECT_EQ(unitSign(x).node, RevTape::kNone);
    EXPECT_EQ(tape.size(), 1u); // only the input node
}

TEST_F(GenericOpsRevTest, SqrCubeLerpGradients) {
    EXPECT_DOUBLE_EQ(grad(1.3, [](const RevScalar& x) { return sqr(x); }), 2.6);
    EXPECT_NEAR(grad(-0.7, [](const RevScalar& x) { return cube(x); }), 3.0 * 0.49, 1e-15);
    EXPECT_NEAR(grad(0.9, [](const RevScalar& x) { return lerp(RevScalar(0.25), x, 0.7); }), 0.7,
                1e-15);
    EXPECT_NEAR(grad(0.9, [](const RevScalar& x) { return lerp(x, RevScalar(0.8), 0.5); }), 0.5,
                1e-15);
}

TEST_F(GenericOpsRevTest, SafeDivGradients) {
    EXPECT_NEAR(grad(1.4, [](const RevScalar& x) { return safeDiv(x, RevScalar(0.7), 1e-9); }),
                1.0 / 0.7, 1e-15);
    EXPECT_NEAR(grad(1.25, [](const RevScalar& x) { return safeDiv(RevScalar(1.0), x, 1e-9); }),
                -1.0 / 1.5625, 1e-15);
    EXPECT_DOUBLE_EQ(
        grad(2.0, [](const RevScalar& x) { return safeDiv(x, RevScalar(1e-15), 1e-12); }), 0.0);
}

TEST_F(GenericOpsRevTest, SoftplusSigmoidLogitGradients) {
    const double x0 = 0.7;
    const double s = Sigmoid(x0);
    EXPECT_NEAR(grad(x0, [](const RevScalar& x) { return softplus(x); }), s, 1e-15);
    EXPECT_NEAR(grad(x0, [](const RevScalar& x) { return sigmoid(x); }), s * (1.0 - s), 1e-15);
    EXPECT_NEAR(grad(0.3, [](const RevScalar& x) { return logit(x); }), 1.0 / 0.21, 1e-15);
}

TEST_F(GenericOpsRevTest, SmoothMaxAndLogSumExp2Gradients) {
    const double a = 0.6;
    const double b = -0.4;
    const double k = 2.5;
    const double w = Sigmoid(k * (a - b));
    const auto [da, db] =
        grad2(a, b, [k](const RevScalar& x, const RevScalar& y) { return smoothMax(x, y, k); });
    EXPECT_NEAR(da, w, 1e-15);
    EXPECT_NEAR(db, 1.0 - w, 1e-15);

    const double w1 = Sigmoid(a - b);
    const auto [ea, eb] =
        grad2(a, b, [](const RevScalar& x, const RevScalar& y) { return logSumExp2(x, y); });
    EXPECT_NEAR(ea, w1, 1e-15);
    EXPECT_NEAR(eb, 1.0 - w1, 1e-15);
}

TEST_F(GenericOpsRevTest, SmoothAbsGradient) {
    const double x0 = 0.7;
    const double eps = 1e-3;
    EXPECT_NEAR(grad(x0, [](const RevScalar& x) { return smoothAbs(x, 1e-3); }),
                x0 / std::sqrt(x0 * x0 + eps * eps), 1e-15);
}

TEST_F(GenericOpsRevTest, SoftplusThirdOrderNestedTangent) {
    const double x0 = 0.7;
    const D3 f = softplus(Seed3(x0));
    const double s = Sigmoid(x0);
    EXPECT_NEAR(f.value.value.value, std::log1p(std::exp(x0)), 1e-15);
    EXPECT_NEAR(f.value.value.d[0], s, 1e-14);
    EXPECT_NEAR(f.d[0].value.d[0], s * (1.0 - s), 1e-14);
    EXPECT_NEAR(f.d[0].d[0].d[0], s * (1.0 - s) * (1.0 - 2.0 * s), 1e-13);
}

TEST_F(GenericOpsRevTest, SmoothMaxThirdOrderNestedTangent) {
    const double a = 0.9;
    const double b = 0.3;
    const double k = 2.0;
    const D3 f = smoothMax(Seed3(a), Const3(b), k);

    // w, and its derivatives, computed from the primal (no finite differences).
    const double d = a - b;
    const double w = Sigmoid(k * d);
    const double value = std::fmax(a, b) + std::log1p(std::exp(-std::fabs(d) * k)) / k;
    const double f1 = w;
    const double f2 = k * w * (1.0 - w);
    const double f3 = k * k * w * (1.0 - w) * (1.0 - 2.0 * w);

    EXPECT_NEAR(f.value.value.value, value, 1e-14);
    EXPECT_NEAR(f.d[0].value.value, f1, 1e-13);
    EXPECT_NEAR(f.d[0].value.d[0], f2, 1e-13);
    EXPECT_NEAR(f.d[0].d[0].value, f2, 1e-13);
    EXPECT_NEAR(f.d[0].d[0].d[0], f3, 1e-13);

    // Symmetric check through the second argument: f_b = 1 - w.
    const D3 g = smoothMax(Const3(a), Seed3(b), k);
    EXPECT_NEAR(g.d[0].value.value, 1.0 - w, 1e-13);
    EXPECT_NEAR(g.d[0].d[0].value, f2, 1e-13);
    EXPECT_NEAR(g.d[0].d[0].d[0], -f3, 1e-13);
}

TEST(GenericOpsMixedTest, DoubleRevAndTangentCalls) {
    // double family: literals only.
    EXPECT_NEAR(smoothMax(0.3, -0.2, 2.0), 0.3 + std::log1p(std::exp(-1.0)) / 2.0, 1e-15);
    EXPECT_NEAR(smoothAbs(0.7, 1e-3), std::sqrt(0.49 + 1e-6), 1e-15);

    // reverse family: AD values with plain-double hyperparameters.
    {
        RevTape& tape = RevTape::active();
        tape.clear();
        RevScalar x(0.5, tape.input(0.5));
        const RevScalar f = smoothMax(x, RevScalar(0.25), 2.0) + lerp(x, RevScalar(1.5), 0.5) +
                            smoothAbs(x, 1e-6) + sigmoid(2.0 * x + 1.0);
        tape.reverse(f.node);
        const double w = Sigmoid(2.0 * (0.5 - 0.25));
        const double expected =
            w + 0.5 + 0.5 / std::sqrt(0.25 + 1e-12) + 2.0 * Sigmoid(2.0) * (1.0 - Sigmoid(2.0));
        EXPECT_NEAR(tape.adjoint(x.node), expected, 1e-12);
    }

    // forward family: AD value with plain-double literals through operators.
    {
        Tangent<double, 1> x(0.5, {1.0});
        const Tangent<double, 1> f =
            smoothAbs(2.0 * x + 1.0, 1e-6) + safeDiv(x, Tangent<double, 1>(0.5), 1e-12);
        const double m = std::sqrt(4.0 + 1e-12);
        EXPECT_NEAR(f.value, m + 1.0, 1e-15);
        EXPECT_NEAR(f.d[0], 4.0 / m + 2.0, 1e-14);
    }

    // nested forward family: literal injection inside the op at second order.
    {
        const D2 x = Seed2(0.5);
        const D2 f = softplus(x);
        const double s = Sigmoid(0.5);
        EXPECT_NEAR(f.value.value, std::log1p(std::exp(0.5)), 1e-15);
        EXPECT_NEAR(f.d[0].d[0], s * (1.0 - s), 1e-14);
    }
}

} // namespace
