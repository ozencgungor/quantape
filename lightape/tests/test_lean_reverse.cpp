#include <cmath>
#include <cstdint>
#include <vector>

#include "lightape/lightape.h"
#include <gtest/gtest.h>

namespace {

using lightape::RevScalar;
using lightape::RevTape;

class LeanReverseTest : public ::testing::Test {
protected:
    void SetUp() override { RevTape::active().clear(); }
    void TearDown() override { RevTape::active().clear(); }
};

TEST_F(LeanReverseTest, ConstantsAreNodeLess) {
    RevScalar a(2.0);
    RevScalar b(3.0);
    RevScalar c = a * b + a / b - a;
    EXPECT_EQ(c.node, RevTape::kNone);
    EXPECT_DOUBLE_EQ(c.value, 2.0 * 3.0 + 2.0 / 3.0 - 2.0);
    EXPECT_EQ(RevTape::active().size(), 0u);
}

TEST_F(LeanReverseTest, BinaryChainGradients) {
    RevTape& tape = RevTape::active();
    RevScalar x(2.0, tape.input(2.0));
    RevScalar y(3.0, tape.input(3.0));
    RevScalar f = x * y + x;
    tape.reverse(f.node);
    EXPECT_DOUBLE_EQ(tape.adjoint(x.node), 4.0);
    EXPECT_DOUBLE_EQ(tape.adjoint(y.node), 2.0);
}

TEST_F(LeanReverseTest, UnaryChainGradients) {
    RevTape& tape = RevTape::active();
    const double a = 0.25;
    RevScalar x(a, tape.input(a));
    RevScalar g = exp(sqrt(x));
    tape.reverse(g.node);
    EXPECT_DOUBLE_EQ(g.value, std::exp(std::sqrt(a)));
    EXPECT_NEAR(tape.adjoint(x.node), std::exp(std::sqrt(a)) / (2.0 * std::sqrt(a)), 1e-14);
}

TEST_F(LeanReverseTest, DivisionAndLogGradients) {
    RevTape& tape = RevTape::active();
    RevScalar p(0.7, tape.input(0.7));
    RevScalar q(1.3, tape.input(1.3));
    RevScalar h = log(p / q) - p * p;
    tape.reverse(h.node);
    EXPECT_NEAR(tape.adjoint(p.node), 1.0 / 0.7 - 2.0 * 0.7, 1e-13);
    EXPECT_NEAR(tape.adjoint(q.node), -1.0 / 1.3, 1e-13);
}

TEST_F(LeanReverseTest, BranchesShareWinnerNode) {
    RevTape& tape = RevTape::active();
    RevScalar a(1.0, tape.input(1.0));
    RevScalar b(2.0, tape.input(2.0));
    RevScalar lo = fmin(a, b);
    RevScalar hi = fmax(a, b);
    EXPECT_EQ(lo.node, a.node);
    EXPECT_EQ(hi.node, b.node);
    RevScalar f = lo + hi;
    tape.reverse(f.node);
    EXPECT_DOUBLE_EQ(tape.adjoint(a.node), 1.0);
    EXPECT_DOUBLE_EQ(tape.adjoint(b.node), 1.0);
}

TEST_F(LeanReverseTest, AbsNegAndConstantOperand) {
    RevTape& tape = RevTape::active();
    RevScalar x(-3.0, tape.input(-3.0));
    RevScalar f = -fabs(x) * 2.0;
    tape.reverse(f.node);
    EXPECT_DOUBLE_EQ(f.value, -6.0);
    EXPECT_DOUBLE_EQ(tape.adjoint(x.node), 2.0);
}

TEST_F(LeanReverseTest, ClearResetsSinkAndReusesNodes) {
    RevTape& tape = RevTape::active();
    RevScalar x(1.0, tape.input(1.0));
    RevScalar f = x + 1.0;
    EXPECT_DOUBLE_EQ(f.value, 2.0);
    EXPECT_GT(tape.size(), 0u);
    tape.clear();
    EXPECT_EQ(tape.size(), 0u);
    RevScalar y(4.0, tape.input(4.0));
    EXPECT_EQ(y.node, 1u);
    RevScalar g = sqrt(y);
    tape.reverse(g.node);
    EXPECT_DOUBLE_EQ(tape.adjoint(y.node), 0.25);
}

TEST_F(LeanReverseTest, ZeroAdjointSweepSkipsDeadBranches) {
    RevTape& tape = RevTape::active();
    RevScalar x(2.0, tape.input(2.0));
    RevScalar dead = x * x;
    EXPECT_DOUBLE_EQ(dead.value, 4.0);
    RevScalar live = x + 1.0;
    RevScalar f = live * 3.0;
    tape.reverse(f.node);
    EXPECT_DOUBLE_EQ(tape.adjoint(x.node), 3.0);
}

TEST_F(LeanReverseTest, ChunkBoundarySweep) {
    RevTape& tape = RevTape::active();
    constexpr std::size_t n = RevTape::kChunkNodes + 257;
    std::vector<RevScalar> xs;
    xs.reserve(n);
    RevScalar acc(0.0);
    for (std::size_t i = 0; i < n; ++i) {
        const double v = 0.5 + 0.001 * static_cast<double>(i % 1000);
        xs.emplace_back(v, tape.input(v));
        acc = acc + xs[i] * 3.0;
    }
    EXPECT_GT(tape.size(), RevTape::kChunkNodes);
    tape.reverse(acc.node);
    for (std::size_t i = 0; i < n; ++i) {
        EXPECT_DOUBLE_EQ(tape.adjoint(xs[i].node), 3.0) << "i=" << i;
    }
    tape.resetAdjoints();
    for (std::size_t i = 1; i <= tape.size(); i += 97) {
        EXPECT_DOUBLE_EQ(tape.adjoint(static_cast<std::uint32_t>(i)), 0.0);
    }
}

TEST_F(LeanReverseTest, ChunkReuseAfterClear) {
    RevTape& tape = RevTape::active();
    RevScalar acc(0.0);
    for (std::size_t i = 0; i < RevTape::kChunkNodes + 10; ++i) {
        RevScalar x(1.0, tape.input(1.0));
        acc = acc + x;
    }
    EXPECT_GT(tape.size(), RevTape::kChunkNodes);
    tape.clear();
    EXPECT_EQ(tape.size(), 0u);
    RevScalar y(4.0, tape.input(4.0));
    EXPECT_EQ(y.node, 1u);
    RevScalar g = sqrt(y);
    tape.reverse(g.node);
    EXPECT_DOUBLE_EQ(tape.adjoint(y.node), 0.25);
}

TEST_F(LeanReverseTest, ResetToTruncatesAcrossChunks) {
    RevTape& tape = RevTape::active();
    RevScalar x(2.0, tape.input(2.0));
    RevScalar y(3.0, tape.input(3.0));
    const std::size_t pos = tape.position();
    RevScalar acc = x;
    for (std::size_t i = 0; i < RevTape::kChunkNodes + 5; ++i) {
        acc = acc + 1.0;
    }
    EXPECT_GT(tape.size(), RevTape::kChunkNodes);
    tape.resetAdjoints();
    tape.resetTo(pos);
    EXPECT_EQ(tape.position(), pos);
    EXPECT_EQ(tape.size(), pos - 1);
    RevScalar f = x * y;
    tape.reverse(f.node);
    EXPECT_DOUBLE_EQ(tape.adjoint(x.node), 3.0);
    EXPECT_DOUBLE_EQ(tape.adjoint(y.node), 2.0);
}

} // namespace
