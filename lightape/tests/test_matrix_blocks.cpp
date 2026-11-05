#include <Eigen/Core>

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>

#include "lightape/eigen/MatrixBlocks.h"
#include "lightape/lightape.h"
#include <gtest/gtest.h>

namespace {

using lightape::RevScalar;
using lightape::RevTape;

using RevMatrix = Eigen::Matrix<RevScalar, Eigen::Dynamic, Eigen::Dynamic>;
using RevVector = Eigen::Matrix<RevScalar, Eigen::Dynamic, 1>;
using DoubleMatrix = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>;

class MatrixBlocksTest : public ::testing::Test {
protected:
    void SetUp() override { RevTape::active().clear(); }
    void TearDown() override { RevTape::active().clear(); }
};

void fillVector(RevTape& tape, RevVector& x, const double* values, int n) {
    ASSERT_EQ(x.size(), static_cast<Eigen::Index>(n));
    for (int i = 0; i < n; ++i) {
        x(i) = RevScalar(values[i], tape.input(values[i]));
    }
}

/// Column-major fill of a `rows x cols` matrix of independent inputs.
void fillMatrix(RevTape& tape, RevMatrix& A, const double* values, int rows, int cols) {
    ASSERT_EQ(A.rows(), static_cast<Eigen::Index>(rows));
    ASSERT_EQ(A.cols(), static_cast<Eigen::Index>(cols));
    for (int j = 0; j < cols; ++j) {
        for (int i = 0; i < rows; ++i) {
            const double v = values[j * rows + i];
            A(i, j) = RevScalar(v, tape.input(v));
        }
    }
}

TEST_F(MatrixBlocksTest, DotGradientMatchesScalarPath) {
    constexpr int n = 4;
    const double xv[n] = {0.5, -1.25, 2.0, 0.75};
    const double yv[n] = {1.5, 0.25, -0.75, 2.5};
    RevTape& tape = RevTape::active();

    RevVector x(n), y(n);
    fillVector(tape, x, xv, n);
    fillVector(tape, y, yv, n);
    const RevScalar block = lightape::dot(x, y);
    ASSERT_NE(block.node, RevTape::kNone);
    tape.reverse(block.node);
    double blockAdjX[n];
    double blockAdjY[n];
    for (int i = 0; i < n; ++i) {
        blockAdjX[i] = tape.adjoint(x(i).node);
        blockAdjY[i] = tape.adjoint(y(i).node);
    }

    tape.clear();
    fillVector(tape, x, xv, n);
    fillVector(tape, y, yv, n);
    const RevScalar scalar = x.dot(y);
    tape.reverse(scalar.node);

    EXPECT_NEAR(block.value, scalar.value, 1e-15);
    for (int i = 0; i < n; ++i) {
        EXPECT_NEAR(blockAdjX[i], tape.adjoint(x(i).node), 1e-13) << "x " << i;
        EXPECT_NEAR(blockAdjY[i], tape.adjoint(y(i).node), 1e-13) << "y " << i;
        EXPECT_NEAR(blockAdjX[i], yv[i], 1e-15) << "x analytic " << i;
        EXPECT_NEAR(blockAdjY[i], xv[i], 1e-15) << "y analytic " << i;
    }
}

TEST_F(MatrixBlocksTest, DotNodeCountOnePerBlockAndConstantsCollapse) {
    constexpr int n = 3;
    const double xv[n] = {1.0, 2.0, 3.0};
    const double yv[n] = {-1.0, 0.5, 2.0};
    RevTape& tape = RevTape::active();

    RevVector x(n), y(n);
    fillVector(tape, x, xv, n);
    fillVector(tape, y, yv, n);
    const std::size_t before = tape.size();
    const RevScalar f = lightape::dot(x, y);
    EXPECT_EQ(tape.size(), before + 1u);
    EXPECT_NE(f.node, RevTape::kNone);

    tape.clear();
    RevVector cx(n), cy(n);
    for (int i = 0; i < n; ++i) {
        cx(i) = RevScalar(xv[i]);
        cy(i) = RevScalar(yv[i]);
    }
    const RevScalar g = lightape::dot(cx, cy);
    EXPECT_EQ(g.node, RevTape::kNone);
    EXPECT_EQ(tape.size(), 0u);
    EXPECT_DOUBLE_EQ(g.value, 6.0);
}

TEST_F(MatrixBlocksTest, SumGradientMatchesScalarPath) {
    constexpr int n = 5;
    const double xv[n] = {0.25, -1.5, 3.0, 0.5, -0.75};
    RevTape& tape = RevTape::active();

    RevVector x(n);
    fillVector(tape, x, xv, n);
    const RevScalar block = lightape::sum(x);
    ASSERT_NE(block.node, RevTape::kNone);
    tape.reverse(block.node);
    double blockAdj[n];
    for (int i = 0; i < n; ++i) {
        blockAdj[i] = tape.adjoint(x(i).node);
    }

    tape.clear();
    fillVector(tape, x, xv, n);
    const RevScalar scalar = x.sum();
    tape.reverse(scalar.node);

    EXPECT_NEAR(block.value, scalar.value, 1e-15);
    for (int i = 0; i < n; ++i) {
        EXPECT_NEAR(blockAdj[i], tape.adjoint(x(i).node), 1e-13) << "x " << i;
        EXPECT_DOUBLE_EQ(blockAdj[i], 1.0) << "x analytic " << i;
    }
}

TEST_F(MatrixBlocksTest, SumOverMatrixRecordsOneNode) {
    constexpr int rows = 2;
    constexpr int cols = 3;
    const double values[rows * cols] = {1.0, -2.0, 0.5, 4.0, 0.25, -1.5};
    RevTape& tape = RevTape::active();
    RevMatrix M(rows, cols);
    fillMatrix(tape, M, values, rows, cols);

    const std::size_t before = tape.size();
    const RevScalar f = lightape::sum(M);
    EXPECT_EQ(tape.size(), before + 1u);
    double expected = 0.0;
    for (double v : values) {
        expected += v;
    }
    EXPECT_NEAR(f.value, expected, 1e-15);

    tape.reverse(f.node);
    for (int j = 0; j < cols; ++j) {
        for (int i = 0; i < rows; ++i) {
            EXPECT_DOUBLE_EQ(tape.adjoint(M(i, j).node), 1.0);
        }
    }
}

TEST_F(MatrixBlocksTest, MatvecGradientMatchesScalarPath) {
    constexpr int m = 3;
    constexpr int k = 4;
    const double av[m * k] = {2.0, -1.0, 0.5, 1.5, 0.25, -2.0, 0.75, 1.25, 3.0, -0.5, 2.5, 0.125};
    const double xv[k] = {0.5, -1.5, 2.0, 0.75};
    RevTape& tape = RevTape::active();

    RevMatrix A(m, k);
    RevVector x(k);
    fillMatrix(tape, A, av, m, k);
    fillVector(tape, x, xv, k);
    const RevScalar block = lightape::matrixVectorProduct(A, x);
    ASSERT_NE(block.node, RevTape::kNone);
    tape.reverse(block.node);
    double blockAdjA[m * k];
    double blockAdjX[k];
    for (int j = 0; j < k; ++j) {
        blockAdjX[j] = tape.adjoint(x(j).node);
        for (int i = 0; i < m; ++i) {
            blockAdjA[j * m + i] = tape.adjoint(A(i, j).node);
        }
    }

    tape.clear();
    fillMatrix(tape, A, av, m, k);
    fillVector(tape, x, xv, k);
    const RevScalar scalar = (A * x).sum();
    tape.reverse(scalar.node);

    EXPECT_NEAR(block.value, scalar.value, 1e-15);
    for (int j = 0; j < k; ++j) {
        EXPECT_NEAR(blockAdjX[j], tape.adjoint(x(j).node), 1e-13) << "x " << j;
        double colSum = 0.0;
        for (int i = 0; i < m; ++i) {
            colSum += av[j * m + i];
            EXPECT_NEAR(blockAdjA[j * m + i], tape.adjoint(A(i, j).node), 1e-13)
                << "A " << i << "," << j;
            EXPECT_NEAR(blockAdjA[j * m + i], xv[j], 1e-15) << "A analytic " << i << "," << j;
        }
        EXPECT_NEAR(blockAdjX[j], colSum, 1e-15) << "x analytic " << j;
    }
}

TEST_F(MatrixBlocksTest, MatvecNodeCountOnePerBlockWithPassiveMatrix) {
    constexpr int m = 2;
    constexpr int k = 3;
    DoubleMatrix A(m, k);
    A << 1.0, -0.5, 2.0, 0.25, 3.0, -1.0;
    const double xv[k] = {0.5, 1.5, -0.25};
    RevTape& tape = RevTape::active();
    RevVector x(k);
    fillVector(tape, x, xv, k);

    const std::size_t before = tape.size();
    const RevScalar f = lightape::matrixVectorProduct(A, x);
    ASSERT_NE(f.node, RevTape::kNone);
    EXPECT_EQ(tape.size(), before + 1u);

    tape.reverse(f.node);
    for (int j = 0; j < k; ++j) {
        double colSum = 0.0;
        for (int i = 0; i < m; ++i) {
            colSum += A(i, j);
        }
        EXPECT_NEAR(tape.adjoint(x(j).node), colSum, 1e-13) << "x " << j;
    }
}

TEST_F(MatrixBlocksTest, DotHvpMatchesScalarPath) {
    constexpr int n = 4;
    const double xv[n] = {0.5, -1.25, 2.0, 0.75};
    const double yv[n] = {1.5, 0.25, -0.75, 2.5};
    const double direction[2 * n] = {0.1, -0.2, 0.3, 0.4, -0.5, 0.6, -0.7, 0.8};
    RevTape& tape = RevTape::active();

    RevVector x(n), y(n);
    fillVector(tape, x, xv, n);
    fillVector(tape, y, yv, n);
    const RevScalar block = lightape::dot(x, y);
    std::uint32_t inputs[2 * n];
    for (int i = 0; i < n; ++i) {
        inputs[i] = x(i).node;
        inputs[n + i] = y(i).node;
    }
    double outBlock[2 * n] = {};
    ASSERT_TRUE(tape.hessianVector(block.node, inputs, 2 * n, direction, outBlock));

    tape.clear();
    fillVector(tape, x, xv, n);
    fillVector(tape, y, yv, n);
    const RevScalar scalar = x.dot(y);
    for (int i = 0; i < n; ++i) {
        inputs[i] = x(i).node;
        inputs[n + i] = y(i).node;
    }
    double outScalar[2 * n] = {};
    ASSERT_TRUE(tape.hessianVector(scalar.node, inputs, 2 * n, direction, outScalar));

    for (int i = 0; i < n; ++i) {
        EXPECT_NEAR(outBlock[i], outScalar[i], 1e-12) << "x " << i;
        EXPECT_NEAR(outBlock[n + i], outScalar[n + i], 1e-12) << "y " << i;
        EXPECT_NEAR(outBlock[i], direction[n + i], 1e-12) << "x analytic " << i;
        EXPECT_NEAR(outBlock[n + i], direction[i], 1e-12) << "y analytic " << i;
    }
}

TEST_F(MatrixBlocksTest, DotSelfVectorSharesSlots) {
    constexpr int n = 3;
    const double xv[n] = {0.5, -1.5, 2.0};
    const double direction[n] = {0.25, -0.75, 1.25};
    RevTape& tape = RevTape::active();

    RevVector x(n);
    fillVector(tape, x, xv, n);
    const RevScalar f = lightape::dot(x, x);
    tape.reverse(f.node);
    for (int i = 0; i < n; ++i) {
        EXPECT_NEAR(tape.adjoint(x(i).node), 2.0 * xv[i], 1e-13) << "x " << i;
    }

    tape.clear();
    fillVector(tape, x, xv, n);
    const RevScalar g = lightape::dot(x, x);
    std::uint32_t inputs[n];
    for (int i = 0; i < n; ++i) {
        inputs[i] = x(i).node;
    }
    double out[n] = {};
    ASSERT_TRUE(tape.hessianVector(g.node, inputs, n, direction, out));
    for (int i = 0; i < n; ++i) {
        EXPECT_NEAR(out[i], 2.0 * direction[i], 1e-12) << "x " << i;
    }
}

TEST_F(MatrixBlocksTest, MatvecHvpMatchesScalarPath) {
    constexpr int m = 2;
    constexpr int k = 3;
    const double av[m * k] = {2.0, -1.0, 0.5, 1.5, 0.25, -2.0};
    const double xv[k] = {0.5, -1.5, 2.0};
    const double direction[m * k + k] = {0.1, -0.2, 0.3, 0.4, -0.5, 0.6, 0.7, -0.8, 0.9};
    RevTape& tape = RevTape::active();

    RevMatrix A(m, k);
    RevVector x(k);
    fillMatrix(tape, A, av, m, k);
    fillVector(tape, x, xv, k);
    const RevScalar block = lightape::matrixVectorProduct(A, x);
    std::uint32_t inputs[m * k + k];
    for (int j = 0; j < k; ++j) {
        inputs[k * m + j] = x(j).node;
        for (int i = 0; i < m; ++i) {
            inputs[j * m + i] = A(i, j).node;
        }
    }
    double outBlock[m * k + k] = {};
    ASSERT_TRUE(tape.hessianVector(block.node, inputs, m * k + k, direction, outBlock));

    tape.clear();
    fillMatrix(tape, A, av, m, k);
    fillVector(tape, x, xv, k);
    const RevScalar scalar = (A * x).sum();
    for (int j = 0; j < k; ++j) {
        inputs[k * m + j] = x(j).node;
        for (int i = 0; i < m; ++i) {
            inputs[j * m + i] = A(i, j).node;
        }
    }
    double outScalar[m * k + k] = {};
    ASSERT_TRUE(tape.hessianVector(scalar.node, inputs, m * k + k, direction, outScalar));

    for (int i = 0; i < m * k; ++i) {
        EXPECT_NEAR(outBlock[i], outScalar[i], 1e-12) << "A " << i;
    }
    for (int j = 0; j < k; ++j) {
        EXPECT_NEAR(outBlock[m * k + j], outScalar[m * k + j], 1e-12) << "x " << j;
        double rowSum = 0.0;
        for (int i = 0; i < m; ++i) {
            rowSum += direction[j * m + i];
        }
        EXPECT_NEAR(outBlock[m * k + j], rowSum, 1e-12) << "x analytic " << j;
        for (int i = 0; i < m; ++i) {
            EXPECT_NEAR(outBlock[j * m + i], direction[m * k + j], 1e-12)
                << "A analytic " << i << "," << j;
        }
    }
}

TEST_F(MatrixBlocksTest, SumHvpIsExactlyZero) {
    constexpr int n = 4;
    const double xv[n] = {0.5, -1.5, 2.0, 0.25};
    const double direction[n] = {0.3, 0.4, -0.5, 0.6};
    RevTape& tape = RevTape::active();
    RevVector x(n);
    fillVector(tape, x, xv, n);
    const RevScalar f = lightape::sum(x);
    std::uint32_t inputs[n];
    for (int i = 0; i < n; ++i) {
        inputs[i] = x(i).node;
    }
    double out[n];
    for (int i = 0; i < n; ++i) {
        out[i] = 42.0;
    }
    ASSERT_TRUE(tape.hessianVector(f.node, inputs, n, direction, out));
    for (int i = 0; i < n; ++i) {
        EXPECT_DOUBLE_EQ(out[i], 0.0) << "x " << i;
    }
}

TEST_F(MatrixBlocksTest, MixedDoubleMatrixRevVectorExpression) {
    constexpr int m = 2;
    constexpr int k = 3;
    DoubleMatrix A(m, k);
    A << 1.0, -0.5, 2.0, 0.25, 3.0, -1.0;
    const double xv[k] = {0.5, 1.5, -0.25};
    RevTape& tape = RevTape::active();

    RevVector x(k);
    fillVector(tape, x, xv, k);
    const RevScalar expression = (A * x).sum();
    ASSERT_NE(expression.node, RevTape::kNone);
    tape.reverse(expression.node);
    double expressionAdj[k];
    for (int j = 0; j < k; ++j) {
        expressionAdj[j] = tape.adjoint(x(j).node);
    }

    tape.clear();
    fillVector(tape, x, xv, k);
    const std::size_t before = tape.size();
    const RevScalar block = lightape::matrixVectorProduct(A, x);
    ASSERT_NE(block.node, RevTape::kNone);
    EXPECT_EQ(tape.size(), before + 1u);
    EXPECT_NEAR(block.value, expression.value, 1e-15);
    tape.reverse(block.node);
    for (int j = 0; j < k; ++j) {
        EXPECT_NEAR(tape.adjoint(x(j).node), expressionAdj[j], 1e-13) << "x " << j;
        double colSum = 0.0;
        for (int i = 0; i < m; ++i) {
            colSum += A(i, j);
        }
        EXPECT_NEAR(tape.adjoint(x(j).node), colSum, 1e-13) << "x analytic " << j;
    }

    tape.clear();
    fillVector(tape, x, xv, k);
    const RevScalar hvpRoot = lightape::matrixVectorProduct(A, x);
    std::uint32_t inputs[k];
    for (int j = 0; j < k; ++j) {
        inputs[j] = x(j).node;
    }
    const double direction[k] = {0.3, -0.4, 0.5};
    double out[k] = {};
    ASSERT_TRUE(tape.hessianVector(hvpRoot.node, inputs, k, direction, out));
    for (int j = 0; j < k; ++j) {
        EXPECT_DOUBLE_EQ(out[j], 0.0) << "x " << j;
    }
}

TEST(MatrixBlocksSmokeTest, AllFiniteAndHasNaN) {
    RevMatrix m(2, 2);
    m(0, 0) = RevScalar(1.0);
    m(0, 1) = RevScalar(2.0);
    m(1, 0) = RevScalar(3.0);
    m(1, 1) = RevScalar(4.0);
    EXPECT_TRUE(m.allFinite());
    EXPECT_FALSE(m.hasNaN());

    m(1, 1) = RevScalar(std::numeric_limits<double>::quiet_NaN());
    EXPECT_TRUE(m.hasNaN());
    EXPECT_FALSE(m.allFinite());
}

} // namespace
