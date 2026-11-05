#include <cmath>
#include <cstddef>
#include <cstdint>

#include "lightape/lightape.h"
#include <gtest/gtest.h>

namespace {

using lightape::ExtArg;
using lightape::RevScalar;
using lightape::RevTape;

class HessianTest : public ::testing::Test {
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

TEST_F(HessianTest, ExpLogSinAnalytic) {
    RevTape& tape = RevTape::active();
    const double xv = 0.7;
    const double yv = 1.3;
    RevScalar x(xv, tape.input(xv));
    RevScalar y(yv, tape.input(yv));
    RevScalar f = exp(x * y) + log(x) - 3.0 * sin(y);

    const double exy = std::exp(xv * yv);
    const double hxx = yv * yv * exy - 1.0 / (xv * xv);
    const double hxy = exy * (1.0 + xv * yv);
    const double hyy = xv * xv * exy + 3.0 * std::sin(yv);
    const double v[2] = {0.8, -0.6};
    const double expected[2] = {hxx * v[0] + hxy * v[1], hxy * v[0] + hyy * v[1]};

    const std::uint32_t inputs[2] = {x.node, y.node};
    double out[2] = {0.0, 0.0};
    ASSERT_TRUE(tape.hessianVector(f.node, inputs, 2, v, out));
    EXPECT_NEAR(out[0], expected[0], 1e-12);
    EXPECT_NEAR(out[1], expected[1], 1e-12);
}

TEST_F(HessianTest, LogSumExpAnalytic) {
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
    double v[n] = {0.3, -0.7, 1.1, 0.2, -0.4};
    double out[n] = {0.0, 0.0, 0.0, 0.0, 0.0};
    ASSERT_TRUE(tape.hessianVector(f.node, inputs, n, v, out));
    double dot = 0.0;
    for (std::size_t i = 0; i < n; ++i) {
        dot += o[i] * v[i];
    }
    for (std::size_t i = 0; i < n; ++i) {
        EXPECT_NEAR(out[i], o[i] * v[i] - o[i] * dot, 1e-13) << "i=" << i;
    }
}

TEST_F(HessianTest, SoftmaxObjectiveAnalytic) {
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
    const double v[n] = {0.4, -0.3, 0.9, 0.1};
    double out[n] = {0.0, 0.0, 0.0, 0.0};
    ASSERT_TRUE(tape.hessianVector(f.node, inputs, n, v, out));
    for (std::size_t i = 0; i < n; ++i) {
        double expected = 0.0;
        for (std::size_t j = 0; j < n; ++j) {
            const double delta = (i == j) ? 1.0 : 0.0;
            const double hij = (w[i] - s) * o[i] * (delta - o[j]) - o[i] * o[j] * (w[j] - s);
            expected += hij * v[j];
        }
        EXPECT_NEAR(out[i], expected, 1e-13) << "i=" << i;
    }
}

TEST_F(HessianTest, FmaThroughChain) {
    RevTape& tape = RevTape::active();
    const double xv = 0.4;
    const double yv = 1.7;
    RevScalar x(xv, tape.input(xv));
    RevScalar y(yv, tape.input(yv));
    RevScalar f = fma(x, y, exp(x)); // xy + e^x

    const double ex = std::exp(xv);
    const double v[2] = {0.5, 0.25};
    const double expected[2] = {ex * v[0] + v[1], v[0]};
    const std::uint32_t inputs[2] = {x.node, y.node};
    double out[2] = {0.0, 0.0};
    ASSERT_TRUE(tape.hessianVector(f.node, inputs, 2, v, out));
    EXPECT_NEAR(out[0], expected[0], 1e-13);
    EXPECT_NEAR(out[1], expected[1], 1e-13);
}

TEST_F(HessianTest, MissingSecondOrderFailsLoudly) {
    RevTape& tape = RevTape::active();
    RevScalar x(1.0, tape.input(1.0));
    RevScalar y(2.0, tape.input(2.0));
    const RevScalar args[2] = {x, y};
    const RevScalar f = lightape::detail::makeExternal(3.0, args, 2, OnlyVjp);
    const std::uint32_t inputs[2] = {x.node, y.node};
    const double v[2] = {1.0, 1.0};
    double out[2] = {-99.0, -99.0};
    EXPECT_FALSE(tape.hessianVector(f.node, inputs, 2, v, out));
    EXPECT_DOUBLE_EQ(out[0], -99.0);
    EXPECT_DOUBLE_EQ(out[1], -99.0);
}

TEST_F(HessianTest, ReverseStoreKeepsIntermediates) {
    RevTape& tape = RevTape::active();
    RevScalar x(0.6, tape.input(0.6));
    RevScalar y(1.1, tape.input(1.1));
    RevScalar u = x * y;
    RevScalar f = exp(u);
    tape.reverseStore(f.node);
    EXPECT_NEAR(tape.adjoint(u.node), std::exp(0.66), 1e-15);
    EXPECT_NEAR(tape.adjoint(x.node), std::exp(0.66) * 1.1, 1e-15);
    EXPECT_NEAR(tape.adjoint(y.node), std::exp(0.66) * 0.6, 1e-15);
}

} // namespace
