#include <cmath>
#include <cstddef>
#include <cstdint>

#include "lightape/lightape.h"
#include <gtest/gtest.h>

namespace {

using lightape::RevScalar;
using lightape::RevTape;

class SoftmaxTest : public ::testing::Test {
protected:
    void SetUp() override { RevTape::active().clear(); }
    void TearDown() override { RevTape::active().clear(); }
};

constexpr std::size_t kN = 4;

void ReferenceSoftmax(const double* v, std::size_t n, double* out) {
    double m = v[0];
    for (std::size_t i = 1; i < n; ++i) {
        if (v[i] > m) {
            m = v[i];
        }
    }
    double s = 0.0;
    for (std::size_t i = 0; i < n; ++i) {
        out[i] = std::exp(v[i] - m);
        s += out[i];
    }
    for (std::size_t i = 0; i < n; ++i) {
        out[i] /= s;
    }
}

TEST_F(SoftmaxTest, ValuesMatchReferenceAndSumToOne) {
    RevTape& tape = RevTape::active();
    const double v[kN] = {1000.0, 1001.0, 999.5, 1000.25};
    RevScalar xs[kN];
    RevScalar out[kN];
    for (std::size_t i = 0; i < kN; ++i) {
        xs[i] = RevScalar(v[i], tape.input(v[i]));
    }
    lightape::softmax(xs, kN, out);
    double ref[kN];
    ReferenceSoftmax(v, kN, ref);
    double sum = 0.0;
    for (std::size_t i = 0; i < kN; ++i) {
        EXPECT_NEAR(out[i].value, ref[i], 1e-13) << "i=" << i;
        sum += out[i].value;
    }
    EXPECT_NEAR(sum, 1.0, 1e-13);
}

TEST_F(SoftmaxTest, SingleRootObjectiveGradient) {
    RevTape& tape = RevTape::active();
    const double v[kN] = {-0.5, 0.25, 1.5, 0.75};
    const double w[kN] = {2.0, -1.0, 0.5, 3.0};
    RevScalar xs[kN];
    RevScalar out[kN];
    for (std::size_t i = 0; i < kN; ++i) {
        xs[i] = RevScalar(v[i], tape.input(v[i]));
    }
    lightape::softmax(xs, kN, out);
    RevScalar f(0.0);
    for (std::size_t i = 0; i < kN; ++i) {
        f = f + out[i] * w[i];
    }
    tape.reverse(f.node);
    double o[kN];
    ReferenceSoftmax(v, kN, o);
    double s = 0.0;
    for (std::size_t i = 0; i < kN; ++i) {
        s += w[i] * o[i];
    }
    for (std::size_t i = 0; i < kN; ++i) {
        EXPECT_NEAR(tape.adjoint(xs[i].node), o[i] * (w[i] - s), 1e-15) << "i=" << i;
    }
}

TEST_F(SoftmaxTest, MultiRootJacobianTranspose) {
    RevTape& tape = RevTape::active();
    const double v[kN] = {-0.5, 0.25, 1.5, 0.75};
    const double seed[kN] = {0.3, -0.7, 1.1, 0.2};
    RevScalar xs[kN];
    RevScalar out[kN];
    for (std::size_t i = 0; i < kN; ++i) {
        xs[i] = RevScalar(v[i], tape.input(v[i]));
    }
    lightape::softmax(xs, kN, out);
    const std::uint32_t roots[kN] = {out[0].node, out[1].node, out[2].node, out[3].node};
    tape.reverse(roots, seed, kN);
    double o[kN];
    ReferenceSoftmax(v, kN, o);
    double weighted = 0.0;
    for (std::size_t i = 0; i < kN; ++i) {
        weighted += seed[i] * o[i];
    }
    for (std::size_t i = 0; i < kN; ++i) {
        EXPECT_NEAR(tape.adjoint(xs[i].node), o[i] * (seed[i] - weighted), 1e-15) << "i=" << i;
    }
}

TEST_F(SoftmaxTest, NodeCountAndConstantCollapse) {
    RevTape& tape = RevTape::active();
    const double v[3] = {0.5, -0.25, 1.0};
    RevScalar xs[3] = {RevScalar(v[0], tape.input(v[0])), RevScalar(v[1], tape.input(v[1])),
                       RevScalar(v[2], tape.input(v[2]))};
    const std::size_t before = tape.size();
    RevScalar out[3];
    lightape::softmax(xs, 3, out);
    EXPECT_EQ(tape.size(), before + 1u + 2u * 3u);

    tape.clear();
    const RevScalar cs[3] = {RevScalar(v[0]), RevScalar(v[1]), RevScalar(v[2])};
    RevScalar cout[3];
    lightape::softmax(cs, 3, cout);
    EXPECT_EQ(tape.size(), 0u);
    for (std::size_t i = 0; i < 3; ++i) {
        EXPECT_EQ(cout[i].node, RevTape::kNone);
    }
    double ref[3];
    ReferenceSoftmax(v, 3, ref);
    EXPECT_NEAR(cout[0].value + cout[1].value + cout[2].value, 1.0, 1e-15);
    EXPECT_NEAR(cout[2].value, ref[2], 1e-13);
}

TEST_F(SoftmaxTest, SingleComponentHasZeroGradient) {
    RevTape& tape = RevTape::active();
    RevScalar x(3.0, tape.input(3.0));
    RevScalar out[1];
    lightape::softmax(&x, 1, out);
    EXPECT_DOUBLE_EQ(out[0].value, 1.0);
    tape.reverse(out[0].node);
    EXPECT_DOUBLE_EQ(tape.adjoint(x.node), 0.0);
    lightape::softmax(nullptr, 0, nullptr);
}

} // namespace
