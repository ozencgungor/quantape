#include <Eigen/Core>

#include <cmath>
#include <cstdint>
#include <type_traits>
#include <utility>

#include "lightape/eigen/ForwardScalarEigen.h"
#include "lightape/eigen/LeanReverseEigen.h"
#include "lightape/lightape.h"
#include <gtest/gtest.h>

namespace {

using lightape::RevScalar;
using lightape::RevTape;
using lightape::Tangent;

using RevMatrix = Eigen::Matrix<RevScalar, Eigen::Dynamic, Eigen::Dynamic>;
using RevVector = Eigen::Matrix<RevScalar, Eigen::Dynamic, 1>;
using DoubleVector = Eigen::Matrix<double, Eigen::Dynamic, 1>;
using Tan2 = Tangent<double, 2>;
using TanVector = Eigen::Matrix<Tan2, Eigen::Dynamic, 1>;

class EigenLeanReverseTest : public ::testing::Test {
protected:
    void SetUp() override { RevTape::active().clear(); }
    void TearDown() override { RevTape::active().clear(); }
};

TEST(EigenScalarsTest, ScalarBinaryOpTraits) {
    static_assert(
        std::is_same_v<Eigen::ScalarBinaryOpTraits<RevScalar, RevScalar>::ReturnType, RevScalar>);
    static_assert(
        std::is_same_v<Eigen::ScalarBinaryOpTraits<RevScalar, double>::ReturnType, RevScalar>);
    static_assert(
        std::is_same_v<Eigen::ScalarBinaryOpTraits<double, RevScalar>::ReturnType, RevScalar>);
    static_assert(
        std::is_same_v<
            Eigen::ScalarBinaryOpTraits<
                RevScalar, double, Eigen::internal::scalar_sum_op<RevScalar, double>>::ReturnType,
            RevScalar>);
    static_assert(std::is_same_v<Eigen::ScalarBinaryOpTraits<Tan2, Tan2>::ReturnType, Tan2>);
    static_assert(std::is_same_v<Eigen::ScalarBinaryOpTraits<Tan2, double>::ReturnType, Tan2>);
    static_assert(std::is_same_v<Eigen::ScalarBinaryOpTraits<double, Tan2>::ReturnType, Tan2>);
    static_assert(std::is_same_v<
                  Eigen::ScalarBinaryOpTraits<
                      double, Tan2, Eigen::internal::scalar_quotient_op<double, Tan2>>::ReturnType,
                  Tan2>);

    using MixedSum =
        decltype(std::declval<const RevVector&>() + std::declval<const DoubleVector&>());
    static_assert(std::is_same_v<MixedSum::Scalar, RevScalar>);
    using MixedProduct = decltype(std::declval<const DoubleVector&>().cwiseProduct(
        std::declval<const RevVector&>()));
    static_assert(std::is_same_v<MixedProduct::Scalar, RevScalar>);

    SUCCEED();
}

TEST_F(EigenLeanReverseTest, MatrixVectorAdjointsMatchAnalyticJacobian) {
    constexpr int n = 3;
    const double a[n][n] = {{2.0, -1.0, 0.5}, {1.5, 0.25, -2.0}, {0.75, 1.25, 3.0}};
    const double xValue[n] = {0.5, -1.5, 2.0};

    RevTape& tape = RevTape::active();
    RevMatrix A(n, n);
    RevVector x(n);
    for (int i = 0; i < n; ++i) {
        x(i) = RevScalar(xValue[i], tape.input(xValue[i]));
        for (int j = 0; j < n; ++j) {
            A(i, j) = RevScalar(a[i][j]);
        }
    }

    RevScalar f = (A * x).sum();
    double expected = 0.0;
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            expected += a[i][j] * xValue[j];
        }
    }
    EXPECT_NEAR(f.value, expected, 1e-15);
    EXPECT_NE(f.node, RevTape::kNone);

    tape.reverse(f.node);
    for (int j = 0; j < n; ++j) {
        double jacobian = 0.0; // d f / d x_j = column sum of A
        for (int i = 0; i < n; ++i) {
            jacobian += a[i][j];
        }
        EXPECT_NEAR(tape.adjoint(x(j).node), jacobian, 1e-14) << "column " << j;
    }
}

TEST(EigenScalarsTest, TangentDotMatchesAnalyticTangents) {
    constexpr int n = 3;
    TanVector x(n);
    TanVector y(n);
    double xSum = 0.0;
    double ySum = 0.0;
    for (int i = 0; i < n; ++i) {
        x(i) = Tan2(0.5 + i, {1.0, 0.0});
        y(i) = Tan2(1.5 - 0.25 * i, {0.0, 1.0});
        xSum += 0.5 + i;
        ySum += 1.5 - 0.25 * i;
    }

    Tan2 p = x.dot(y);
    double expected = 0.0;
    for (int i = 0; i < n; ++i) {
        expected += (0.5 + i) * (1.5 - 0.25 * i);
    }
    EXPECT_NEAR(p.value, expected, 1e-15);
    EXPECT_DOUBLE_EQ(p.d[0], ySum); // d/ds x(s) . y = sum y_i
    EXPECT_DOUBLE_EQ(p.d[1], xSum); // d/dt x . y(t) = sum x_i
}

TEST_F(EigenLeanReverseTest, MixedDoubleRevScalarCwiseExpression) {
    constexpr int n = 3;
    const double d[n] = {1.5, -2.0, 0.25};
    const double xValue[n] = {-0.5, 3.0, 1.25};

    RevTape& tape = RevTape::active();
    DoubleVector dVec(n);
    RevVector x(n);
    for (int i = 0; i < n; ++i) {
        dVec(i) = d[i];
        x(i) = RevScalar(xValue[i], tape.input(xValue[i]));
    }

    RevScalar f = (dVec.cwiseProduct(x)).sum();
    double expected = 0.0;
    for (int i = 0; i < n; ++i) {
        expected += d[i] * xValue[i];
    }
    EXPECT_NEAR(f.value, expected, 1e-15);
    tape.reverse(f.node);
    for (int i = 0; i < n; ++i) {
        EXPECT_DOUBLE_EQ(tape.adjoint(x(i).node), d[i]);
    }

    // Reversed operand order: RevScalar + double, all adjoints one.
    tape.clear();
    for (int i = 0; i < n; ++i) {
        x(i) = RevScalar(xValue[i], tape.input(xValue[i]));
    }
    RevScalar g = (x + dVec).sum();
    double xSum = 0.0;
    for (int i = 0; i < n; ++i) {
        xSum += xValue[i];
    }
    EXPECT_NEAR(g.value, xSum + d[0] + d[1] + d[2], 1e-15);
    tape.reverse(g.node);
    for (int i = 0; i < n; ++i) {
        EXPECT_DOUBLE_EQ(tape.adjoint(x(i).node), 1.0);
    }
}

} // namespace
