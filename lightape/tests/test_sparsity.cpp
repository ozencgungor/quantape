#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <utility>
#include <vector>

#include "lightape/lightape.h"
#include "lightape/rev/SecondOrder.h"
#include "lightape/rev/Sparsity.h"
#include <gtest/gtest.h>

namespace {

using lightape::ExtArg;
using lightape::RevScalar;
using lightape::RevTape;
using lightape::SparsityPattern;

int g_hvpCalls = 0;

class SparsityTest : public ::testing::Test {
protected:
    void SetUp() override {
        RevTape::active().clear();
        g_hvpCalls = 0;
    }
    void TearDown() override {
        RevTape::active().clear();
        g_hvpCalls = 0;
    }
};

void OnlyVjp(const void* payload, double seed, const ExtArg* args, std::uint32_t n,
             RevTape& tape) noexcept {
    (void)payload;
    for (std::uint32_t i = 0; i < n; ++i) {
        tape.addAdjoint(args[i].node, seed);
    }
}

// f = sum exp(x_i) + sum_{d<=width, i+d<n} x_i x_{i+d}: Hessian e^{x_i} on the
// diagonal, 1 at offsets 1..width, zero elsewhere.
RevScalar buildBanded(RevTape& tape, std::size_t n, std::size_t width,
                      const std::vector<double>& xv, std::vector<std::uint32_t>& inputs) {
    std::vector<RevScalar> x(n);
    for (std::size_t i = 0; i < n; ++i) {
        x[i] = RevScalar(xv[i], tape.input(xv[i]));
        inputs[i] = x[i].node;
    }
    RevScalar f(0.0);
    for (std::size_t i = 0; i < n; ++i) {
        f = f + exp(x[i]);
        for (std::size_t d = 1; d <= width && i + d < n; ++d) {
            f = f + x[i] * x[i + d];
        }
    }
    return f;
}

// Same function packed into a single External block, so the svp callback runs
// exactly once per hessianVector call and counts real HVP invocations.
void bandedVjp(const void* /*payload*/, double seed, const ExtArg* args, std::uint32_t n,
               RevTape& tape) noexcept {
    for (std::uint32_t i = 0; i < n; ++i) {
        double g = std::exp(args[i].value);
        if (i > 0) {
            g += args[i - 1].value;
        }
        if (i + 1 < n) {
            g += args[i + 1].value;
        }
        tape.addAdjoint(args[i].node, seed * g);
    }
}

double bandedSvp(const void* /*payload*/, const ExtArg* args, const double* s,
                 std::uint32_t n) noexcept {
    ++g_hvpCalls;
    double out = 0.0;
    for (std::uint32_t i = 0; i < n; ++i) {
        double g = std::exp(args[i].value);
        if (i > 0) {
            g += args[i - 1].value;
        }
        if (i + 1 < n) {
            g += args[i + 1].value;
        }
        out += g * s[i];
    }
    return out;
}

void bandedVhp(const void* /*payload*/, double adjoint, const ExtArg* args, const double* s,
               std::uint32_t n, double* b) noexcept {
    for (std::uint32_t i = 0; i < n; ++i) {
        b[i] += adjoint * std::exp(args[i].value) * s[i];
        if (i > 0) {
            b[i] += adjoint * s[i - 1];
        }
        if (i + 1 < n) {
            b[i] += adjoint * s[i + 1];
        }
    }
}

RevScalar buildExternalBanded(RevTape& tape, std::size_t n, const std::vector<double>& xv,
                              std::vector<std::uint32_t>& inputs) {
    std::vector<RevScalar> x(n);
    for (std::size_t i = 0; i < n; ++i) {
        x[i] = RevScalar(xv[i], tape.input(xv[i]));
        inputs[i] = x[i].node;
    }
    double value = 0.0;
    for (std::size_t i = 0; i < n; ++i) {
        value += std::exp(xv[i]);
        if (i + 1 < n) {
            value += xv[i] * xv[i + 1];
        }
    }
    return lightape::detail::makeExternal(value, x.data(), static_cast<std::uint32_t>(n), bandedVjp,
                                          nullptr, 0, bandedSvp, bandedVhp);
}

bool isPatternEdge(const SparsityPattern& pattern, std::size_t i, std::size_t j) {
    for (const std::pair<std::size_t, std::size_t>& e : pattern.entries) {
        if ((e.first == i && e.second == j) || (e.first == j && e.second == i)) {
            return true;
        }
    }
    return false;
}

// Independent O(n^3) recheck: every pair at distance 1 or 2 must differ.
void expectValidDistance2Coloring(const SparsityPattern& pattern,
                                  const std::vector<std::size_t>& color) {
    const std::size_t n = pattern.n;
    for (std::size_t i = 0; i < n; ++i) {
        for (std::size_t j = i + 1; j < n; ++j) {
            if (isPatternEdge(pattern, i, j)) {
                EXPECT_NE(color[i], color[j]) << "adjacent " << i << "," << j;
                continue;
            }
            bool shareNeighbor = false;
            for (std::size_t k = 0; k < n && !shareNeighbor; ++k) {
                if (k == i || k == j) {
                    continue;
                }
                shareNeighbor = isPatternEdge(pattern, i, k) && isPatternEdge(pattern, j, k);
            }
            if (shareNeighbor) {
                EXPECT_NE(color[i], color[j]) << "distance-2 " << i << "," << j;
            }
        }
    }
}

TEST_F(SparsityTest, GreedyColoringTridiagonalPathHasThreeColors) {
    const std::size_t n = 24;
    const SparsityPattern pattern = SparsityPattern::banded(n, 1);
    std::vector<std::size_t> color;
    EXPECT_EQ(lightape::greedyColoring(pattern, color), 3u);
    ASSERT_EQ(color.size(), n);
    for (std::size_t i = 0; i < n; ++i) {
        EXPECT_EQ(color[i], i % 3) << "i=" << i;
    }
    expectValidDistance2Coloring(pattern, color);
}

TEST_F(SparsityTest, GreedyColoringPentadiagonalHasFiveColors) {
    const std::size_t n = 24;
    const SparsityPattern pattern = SparsityPattern::banded(n, 2);
    std::vector<std::size_t> color;
    EXPECT_EQ(lightape::greedyColoring(pattern, color), 5u);
    for (std::size_t i = 0; i < n; ++i) {
        EXPECT_EQ(color[i], i % 5) << "i=" << i;
    }
    expectValidDistance2Coloring(pattern, color);
}

TEST_F(SparsityTest, GreedyColoringHeptadiagonalHasSevenColors) {
    // Standard bandwidth: |i - j| <= 3; distance-2 needs 2*3 + 1 colors.
    const SparsityPattern pattern = SparsityPattern::banded(24, 3);
    std::vector<std::size_t> color;
    EXPECT_EQ(lightape::greedyColoring(pattern, color), 7u);
    expectValidDistance2Coloring(pattern, color);
}

TEST_F(SparsityTest, GreedyColoringDiagonalIsOneColor) {
    SparsityPattern pattern;
    pattern.n = 10;
    for (std::size_t i = 0; i < pattern.n; ++i) {
        pattern.entries.emplace_back(i, i);
    }
    std::vector<std::size_t> color;
    EXPECT_EQ(lightape::greedyColoring(pattern, color), 1u);
    ASSERT_EQ(color.size(), pattern.n);
    for (std::size_t i = 0; i < pattern.n; ++i) {
        EXPECT_EQ(color[i], 0u);
    }
}

TEST_F(SparsityTest, GreedyColoringEmptyPattern) {
    SparsityPattern empty;
    std::vector<std::size_t> color;
    EXPECT_EQ(lightape::greedyColoring(empty, color), 0u);
    EXPECT_TRUE(color.empty());

    empty.n = 6; // vertices but no edges
    EXPECT_EQ(lightape::greedyColoring(empty, color), 1u);
    ASSERT_EQ(color.size(), 6u);
    for (const std::size_t c : color) {
        EXPECT_EQ(c, 0u);
    }
}

TEST_F(SparsityTest, GreedyColoringIsDeterministic) {
    const SparsityPattern pattern = SparsityPattern::banded(24, 2);
    std::vector<std::size_t> first;
    std::vector<std::size_t> second;
    EXPECT_EQ(lightape::greedyColoring(pattern, first), lightape::greedyColoring(pattern, second));
    EXPECT_EQ(first, second);
}

TEST_F(SparsityTest, SymmetricHelperMaterializesTransposes) {
    SparsityPattern pattern;
    pattern.n = 4;
    pattern.entries = {{0, 1}, {2, 2}, {1, 3}};
    pattern.symmetric();
    EXPECT_EQ(pattern.entries.size(), 5u);
    EXPECT_TRUE(isPatternEdge(pattern, 0, 1));
    EXPECT_TRUE(isPatternEdge(pattern, 1, 0));
    EXPECT_TRUE(isPatternEdge(pattern, 1, 3));
    EXPECT_TRUE(isPatternEdge(pattern, 3, 1));
    EXPECT_TRUE(isPatternEdge(pattern, 2, 2));
}

TEST_F(SparsityTest, ColoredHessianMatchesDenseTridiagonal) {
    RevTape& tape = RevTape::active();
    const std::size_t n = 24;
    std::vector<double> xv(n);
    for (std::size_t i = 0; i < n; ++i) {
        xv[i] = 0.2 + 0.05 * static_cast<double>(i % 7);
    }
    std::vector<std::uint32_t> inputs(n);
    const RevScalar f = buildBanded(tape, n, 1, xv, inputs);
    const SparsityPattern pattern = SparsityPattern::banded(n, 1);

    std::vector<double> dense(n * n, 0.0);
    std::vector<double> sparse(n * n, -1.0);
    ASSERT_TRUE(lightape::hessian(tape, f.node, inputs.data(), n, dense.data()));
    ASSERT_TRUE(lightape::coloredHessian(tape, f.node, inputs.data(), n, pattern, sparse.data()));

    for (std::size_t i = 0; i < n; ++i) {
        for (std::size_t j = 0; j < n; ++j) {
            EXPECT_NEAR(sparse[i * n + j], dense[i * n + j], 1e-12) << "i=" << i << " j=" << j;
            const double expected =
                (i == j) ? std::exp(xv[i]) : ((i + 1 == j || j + 1 == i) ? 1.0 : 0.0);
            EXPECT_NEAR(sparse[i * n + j], expected, 1e-12) << "i=" << i << " j=" << j;
        }
    }
}

TEST_F(SparsityTest, ColoredHessianMatchesDensePentadiagonal) {
    RevTape& tape = RevTape::active();
    const std::size_t n = 24;
    std::vector<double> xv(n);
    for (std::size_t i = 0; i < n; ++i) {
        xv[i] = -0.3 + 0.04 * static_cast<double>(i % 5);
    }
    std::vector<std::uint32_t> inputs(n);
    const RevScalar f = buildBanded(tape, n, 2, xv, inputs);
    const SparsityPattern pattern = SparsityPattern::banded(n, 2);

    std::vector<std::size_t> color;
    ASSERT_EQ(lightape::greedyColoring(pattern, color), 5u);

    std::vector<double> dense(n * n, 0.0);
    std::vector<double> sparse(n * n, -1.0);
    ASSERT_TRUE(lightape::hessian(tape, f.node, inputs.data(), n, dense.data()));
    ASSERT_TRUE(lightape::coloredHessian(tape, f.node, inputs.data(), n, pattern, sparse.data()));

    for (std::size_t i = 0; i < n; ++i) {
        for (std::size_t j = 0; j < n; ++j) {
            EXPECT_NEAR(sparse[i * n + j], dense[i * n + j], 1e-12) << "i=" << i << " j=" << j;
        }
    }
}

TEST_F(SparsityTest, ColoredHessianDiagonalOnlyFunction) {
    RevTape& tape = RevTape::active();
    const std::size_t n = 12;
    std::vector<double> xv(n);
    for (std::size_t i = 0; i < n; ++i) {
        xv[i] = 0.1 + 0.1 * static_cast<double>(i);
    }
    std::vector<std::uint32_t> inputs(n);
    const RevScalar f = buildBanded(tape, n, 0, xv, inputs);

    SparsityPattern diagonal;
    diagonal.n = n;
    for (std::size_t i = 0; i < n; ++i) {
        diagonal.entries.emplace_back(i, i);
    }
    std::vector<std::size_t> color;
    ASSERT_EQ(lightape::greedyColoring(diagonal, color), 1u);

    std::vector<double> dense(n * n, 0.0);
    std::vector<double> sparse(n * n, -1.0);
    ASSERT_TRUE(lightape::hessian(tape, f.node, inputs.data(), n, dense.data()));
    ASSERT_TRUE(lightape::coloredHessian(tape, f.node, inputs.data(), n, diagonal, sparse.data()));

    for (std::size_t i = 0; i < n; ++i) {
        for (std::size_t j = 0; j < n; ++j) {
            const double expected = (i == j) ? std::exp(xv[i]) : 0.0;
            EXPECT_NEAR(sparse[i * n + j], expected, 1e-12) << "i=" << i << " j=" << j;
            EXPECT_NEAR(sparse[i * n + j], dense[i * n + j], 1e-12);
        }
    }
}

TEST_F(SparsityTest, ColoredHessianUpperTrianglePattern) {
    RevTape& tape = RevTape::active();
    const std::size_t n = 24;
    std::vector<double> xv(n);
    for (std::size_t i = 0; i < n; ++i) {
        xv[i] = 0.05 * static_cast<double>(i + 1);
    }
    std::vector<std::uint32_t> inputs(n);
    const RevScalar f = buildBanded(tape, n, 1, xv, inputs);

    SparsityPattern upper;
    upper.n = n;
    for (std::size_t i = 0; i < n; ++i) {
        upper.entries.emplace_back(i, i);
        if (i + 1 < n) {
            upper.entries.emplace_back(i, i + 1);
        }
    }
    std::vector<double> dense(n * n, 0.0);
    std::vector<double> sparse(n * n, -1.0);
    ASSERT_TRUE(lightape::hessian(tape, f.node, inputs.data(), n, dense.data()));
    ASSERT_TRUE(lightape::coloredHessian(tape, f.node, inputs.data(), n, upper, sparse.data()));
    for (std::size_t i = 0; i < n; ++i) {
        for (std::size_t j = 0; j < n; ++j) {
            EXPECT_NEAR(sparse[i * n + j], dense[i * n + j], 1e-12) << "i=" << i << " j=" << j;
        }
    }
}

TEST_F(SparsityTest, ColoredHessianScratchBufferPath) {
    RevTape& tape = RevTape::active();
    const std::size_t n = 24;
    std::vector<double> xv(n);
    for (std::size_t i = 0; i < n; ++i) {
        xv[i] = 0.15 + 0.03 * static_cast<double>(i);
    }
    std::vector<std::uint32_t> inputs(n);
    const RevScalar f = buildBanded(tape, n, 1, xv, inputs);
    const SparsityPattern pattern = SparsityPattern::banded(n, 1);

    std::vector<double> reference(n * n, 0.0);
    std::vector<double> scratchResult(n * n, -1.0);
    std::vector<double> scratch(2 * n, 0.0);
    ASSERT_TRUE(
        lightape::coloredHessian(tape, f.node, inputs.data(), n, pattern, reference.data()));
    ASSERT_TRUE(lightape::coloredHessian(tape, f.node, inputs.data(), n, pattern,
                                         scratchResult.data(), scratch.data()));
    for (std::size_t t = 0; t < n * n; ++t) {
        EXPECT_DOUBLE_EQ(scratchResult[t], reference[t]) << "t=" << t;
    }
}

TEST_F(SparsityTest, ColoredHessianHvpCallCountVsDense) {
    RevTape& tape = RevTape::active();
    const std::size_t n = 24;
    std::vector<double> xv(n);
    for (std::size_t i = 0; i < n; ++i) {
        xv[i] = 0.1 + 0.02 * static_cast<double>(i);
    }
    std::vector<std::uint32_t> inputs(n);
    const RevScalar f = buildExternalBanded(tape, n, xv, inputs);
    const SparsityPattern pattern = SparsityPattern::banded(n, 1);

    std::vector<double> dense(n * n, 0.0);
    std::vector<double> sparse(n * n, -1.0);
    g_hvpCalls = 0;
    ASSERT_TRUE(lightape::coloredHessian(tape, f.node, inputs.data(), n, pattern, sparse.data()));
    const int coloredCalls = g_hvpCalls;
    g_hvpCalls = 0;
    ASSERT_TRUE(lightape::hessian(tape, f.node, inputs.data(), n, dense.data()));
    const int denseCalls = g_hvpCalls;

    EXPECT_EQ(coloredCalls, 3);                 // one exact HVP per color
    EXPECT_EQ(denseCalls, static_cast<int>(n)); // one exact HVP per column
    for (std::size_t i = 0; i < n; ++i) {
        for (std::size_t j = 0; j < n; ++j) {
            EXPECT_NEAR(sparse[i * n + j], dense[i * n + j], 1e-12) << "i=" << i << " j=" << j;
        }
    }
}

TEST_F(SparsityTest, ColoredHessianEmptyPatternReturnsZeroMatrix) {
    RevTape& tape = RevTape::active();
    const std::size_t n = 6;
    std::vector<double> xv(n, 0.5);
    std::vector<std::uint32_t> inputs(n);
    const RevScalar f = buildExternalBanded(tape, n, xv, inputs);

    SparsityPattern empty;
    empty.n = n;
    std::vector<double> out(n * n, 7.0);
    g_hvpCalls = 0;
    EXPECT_TRUE(lightape::coloredHessian(tape, f.node, inputs.data(), n, empty, out.data()));
    EXPECT_EQ(g_hvpCalls, 0);
    for (const double v : out) {
        EXPECT_DOUBLE_EQ(v, 0.0);
    }
}

TEST_F(SparsityTest, ColoredHessianVjpOnlyExternalFails) {
    RevTape& tape = RevTape::active();
    RevScalar x(1.0, tape.input(1.0));
    RevScalar y(2.0, tape.input(2.0));
    const RevScalar args[2] = {x, y};
    const RevScalar f = lightape::detail::makeExternal(3.0, args, 2, OnlyVjp);
    const std::uint32_t inputs[2] = {x.node, y.node};

    SparsityPattern pattern;
    pattern.n = 2;
    pattern.entries = {{0, 1}};
    double out[4] = {-7.0, -7.0, -7.0, -7.0};
    EXPECT_FALSE(lightape::coloredHessian(tape, f.node, inputs, 2, pattern, out));
}

TEST_F(SparsityTest, ColoredHessianRejectsBadPatterns) {
    RevTape& tape = RevTape::active();
    RevScalar x(1.0, tape.input(1.0));
    RevScalar y(2.0, tape.input(2.0));
    const RevScalar f = x * y;
    const std::uint32_t inputs[2] = {x.node, y.node};
    double out[4] = {0.0, 0.0, 0.0, 0.0};

    SparsityPattern wrongSize;
    wrongSize.n = 3;
    wrongSize.entries = {{0, 1}};
    EXPECT_FALSE(lightape::coloredHessian(tape, f.node, inputs, 2, wrongSize, out));

    SparsityPattern outOfRange;
    outOfRange.n = 2;
    outOfRange.entries = {{0, 2}};
    EXPECT_FALSE(lightape::coloredHessian(tape, f.node, inputs, 2, outOfRange, out));
}

TEST_F(SparsityTest, PolarizationIdentityMatchesDenseCrossPair) {
    RevTape& tape = RevTape::active();
    const std::size_t n = 6;
    std::vector<double> xv(n);
    for (std::size_t i = 0; i < n; ++i) {
        xv[i] = 0.2 + 0.1 * static_cast<double>(i);
    }
    std::vector<std::uint32_t> inputs(n);
    const RevScalar f = buildBanded(tape, n, 2, xv, inputs);

    std::vector<double> dense(n * n, 0.0);
    ASSERT_TRUE(lightape::hessian(tape, f.node, inputs.data(), n, dense.data()));

    const std::pair<std::size_t, std::size_t> pairs[3] = {{0, 2}, {0, 3}, {1, 4}};
    for (const std::pair<std::size_t, std::size_t>& pair : pairs) {
        const std::size_t i = pair.first;
        const std::size_t j = pair.second;
        std::vector<double> dir(n, 0.0);
        std::vector<double> hvp(n, 0.0);
        std::vector<double> unitI(n, 0.0);
        std::vector<double> hvpI(n, 0.0);
        std::vector<double> unitJ(n, 0.0);
        std::vector<double> hvpJ(n, 0.0);
        dir[i] = 1.0;
        dir[j] = 1.0;
        unitI[i] = 1.0;
        unitJ[j] = 1.0;
        ASSERT_TRUE(tape.hessianVector(f.node, inputs.data(), n, dir.data(), hvp.data()));
        ASSERT_TRUE(tape.hessianVector(f.node, inputs.data(), n, unitI.data(), hvpI.data()));
        ASSERT_TRUE(tape.hessianVector(f.node, inputs.data(), n, unitJ.data(), hvpJ.data()));
        const double hij = 0.5 * ((hvp[i] - hvpI[i]) + (hvp[j] - hvpJ[j]));
        EXPECT_NEAR(hij, dense[i * n + j], 1e-12) << "i=" << i << " j=" << j;
        EXPECT_NEAR(hij, dense[j * n + i], 1e-12) << "i=" << i << " j=" << j;
        // The suggested wrong form cancels to zero: (H e_j)_i = H_ji = H_ij.
        const double wrong = hvp[i] - hvpI[i] - hvpJ[i];
        EXPECT_NEAR(wrong, 0.0, 1e-12) << "i=" << i << " j=" << j;
    }
}

} // namespace
