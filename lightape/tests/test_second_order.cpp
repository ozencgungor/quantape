#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>

#include "lightape/lightape.h"
#include "lightape/rev/SecondOrder.h"
#include <gtest/gtest.h>

namespace {

using lightape::ExtArg;
using lightape::RevScalar;
using lightape::RevTape;

class SecondOrderTest : public ::testing::Test {
protected:
    void SetUp() override { RevTape::active().clear(); }
    void TearDown() override { RevTape::active().clear(); }
};

void OnlyVjp(const void* payload, double seed, const ExtArg* args, std::uint32_t n,
             RevTape& tape) noexcept {
    (void)payload;
    for (std::uint32_t i = 0; i < n; ++i) {
        tape.addAdjoint(args[i].node, seed);
    }
}

TEST_F(SecondOrderTest, GradientMatchesReverseReadout) {
    RevTape& tape = RevTape::active();
    const double xv = 0.7;
    const double yv = 1.3;
    RevScalar x(xv, tape.input(xv));
    RevScalar y(yv, tape.input(yv));
    const RevScalar f = exp(x * y) + log(x) - 3.0 * sin(y);

    const double exy = std::exp(xv * yv);
    const double expected[2] = {yv * exy + 1.0 / xv, xv * exy - 3.0 * std::cos(yv)};
    const std::uint32_t inputs[2] = {x.node, y.node};
    double out[2] = {0.0, 0.0};
    lightape::gradient(tape, f.node, inputs, 2, out);
    EXPECT_NEAR(out[0], expected[0], 1e-14);
    EXPECT_NEAR(out[1], expected[1], 1e-14);

    tape.resetAdjoints();
    tape.reverse(f.node);
    EXPECT_DOUBLE_EQ(out[0], tape.adjoint(x.node));
    EXPECT_DOUBLE_EQ(out[1], tape.adjoint(y.node));
}

TEST_F(SecondOrderTest, DenseHessianExpLogSin) {
    RevTape& tape = RevTape::active();
    const double xv = 0.7;
    const double yv = 1.3;
    RevScalar x(xv, tape.input(xv));
    RevScalar y(yv, tape.input(yv));
    const RevScalar f = exp(x * y) + log(x) - 3.0 * sin(y);

    const double exy = std::exp(xv * yv);
    const double hxx = yv * yv * exy - 1.0 / (xv * xv);
    const double hxy = exy * (1.0 + xv * yv);
    const double hyy = xv * xv * exy + 3.0 * std::sin(yv);
    const std::uint32_t inputs[2] = {x.node, y.node};
    double h[4] = {0.0, 0.0, 0.0, 0.0};
    ASSERT_TRUE(lightape::hessian(tape, f.node, inputs, 2, h));
    EXPECT_NEAR(h[0], hxx, 1e-12);
    EXPECT_NEAR(h[1], hxy, 1e-12);
    EXPECT_NEAR(h[2], hxy, 1e-12);
    EXPECT_NEAR(h[3], hyy, 1e-12);

    const double v[2] = {0.8, -0.6};
    double hv[2] = {0.0, 0.0};
    ASSERT_TRUE(lightape::hessianVector(tape, f.node, inputs, 2, v, hv));
    EXPECT_NEAR(hv[0], h[0] * v[0] + h[1] * v[1], 1e-12);
    EXPECT_NEAR(hv[1], h[2] * v[0] + h[3] * v[1], 1e-12);

    double h2[4] = {0.0, 0.0, 0.0, 0.0};
    EXPECT_TRUE(lightape::hessianSymmetric(tape, f.node, inputs, 2, h2));
    EXPECT_NEAR(h2[1], h2[2], 1e-12);
}

TEST_F(SecondOrderTest, DenseHessianLogSumExp) {
    RevTape& tape = RevTape::active();
    constexpr std::size_t n = 5;
    const double values[n] = {-0.5, 0.25, 1.5, 0.75, -1.0};
    RevScalar xs[n];
    std::uint32_t inputs[n];
    for (std::size_t i = 0; i < n; ++i) {
        xs[i] = RevScalar(values[i], tape.input(values[i]));
        inputs[i] = xs[i].node;
    }
    const RevScalar f = lightape::logSumExp(xs, n);

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

    double h[n * n] = {};
    ASSERT_TRUE(lightape::hessian(tape, f.node, inputs, n, h));
    for (std::size_t i = 0; i < n; ++i) {
        for (std::size_t j = 0; j < n; ++j) {
            const double expected = (i == j ? o[i] : 0.0) - o[i] * o[j];
            EXPECT_NEAR(h[i * n + j], expected, 1e-13) << "i=" << i << " j=" << j;
        }
    }
}

TEST_F(SecondOrderTest, DenseHessianSoftmaxObjective) {
    RevTape& tape = RevTape::active();
    constexpr std::size_t n = 4;
    const double values[n] = {-0.5, 0.25, 1.5, 0.75};
    const double w[n] = {2.0, -1.0, 0.5, 3.0};
    RevScalar xs[n];
    RevScalar outv[n];
    std::uint32_t inputs[n];
    for (std::size_t i = 0; i < n; ++i) {
        xs[i] = RevScalar(values[i], tape.input(values[i]));
        inputs[i] = xs[i].node;
    }
    lightape::softmax(xs, n, outv);
    RevScalar f(0.0);
    for (std::size_t i = 0; i < n; ++i) {
        f = f + outv[i] * w[i];
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

    double h[n * n] = {};
    ASSERT_TRUE(lightape::hessian(tape, f.node, inputs, n, h));
    for (std::size_t i = 0; i < n; ++i) {
        for (std::size_t j = 0; j < n; ++j) {
            const double delta = (i == j) ? 1.0 : 0.0;
            const double expected = (w[i] - s) * o[i] * (delta - o[j]) - o[i] * o[j] * (w[j] - s);
            EXPECT_NEAR(h[i * n + j], expected, 1e-12) << "i=" << i << " j=" << j;
        }
    }
    EXPECT_TRUE(lightape::hessianSymmetric(tape, f.node, inputs, n, h));
}

TEST_F(SecondOrderTest, VjpOnlyExternalFails) {
    RevTape& tape = RevTape::active();
    RevScalar x(1.0, tape.input(1.0));
    RevScalar y(2.0, tape.input(2.0));
    const RevScalar args[2] = {x, y};
    const RevScalar f = lightape::detail::makeExternal(3.0, args, 2, OnlyVjp);
    const std::uint32_t inputs[2] = {x.node, y.node};
    double out[4] = {0.0, 0.0, 0.0, 0.0};
    EXPECT_FALSE(lightape::hessian(tape, f.node, inputs, 2, out));
}

TEST_F(SecondOrderTest, InputsNotFirstSlots) {
    RevTape& tape = RevTape::active();
    const double xv = 0.7;
    const double yv = 1.3;
    const double zv = -0.4;
    RevScalar dummy(0.3, tape.input(0.3));
    RevScalar x(xv, tape.input(xv));
    RevScalar y(yv, tape.input(yv));
    RevScalar z(zv, tape.input(zv));
    const RevScalar f = exp(x * y) + log(x) - 3.0 * sin(y) + z * y;
    EXPECT_NE(dummy.node, x.node);

    const double exy = std::exp(xv * yv);
    const std::uint32_t inputs[3] = {x.node, y.node, z.node};
    double h[9] = {};
    ASSERT_TRUE(lightape::hessian(tape, f.node, inputs, 3, h));
    const double hxx = yv * yv * exy - 1.0 / (xv * xv);
    const double hxy = exy * (1.0 + xv * yv);
    const double hyy = xv * xv * exy + 3.0 * std::sin(yv);
    EXPECT_NEAR(h[0], hxx, 1e-12);
    EXPECT_NEAR(h[1], hxy, 1e-12);
    EXPECT_NEAR(h[2], 0.0, 1e-12);
    EXPECT_NEAR(h[3], hxy, 1e-12);
    EXPECT_NEAR(h[4], hyy, 1e-12);
    EXPECT_NEAR(h[5], 1.0, 1e-12);
    EXPECT_NEAR(h[6], 0.0, 1e-12);
    EXPECT_NEAR(h[7], 1.0, 1e-12);
    EXPECT_NEAR(h[8], 0.0, 1e-12);
}

} // namespace
