#include <cmath>

#include "lightape/lightape.h"
#include <gtest/gtest.h>

namespace {

using lightape::Tangent;

TEST(ForwardScalarTest, ArithmeticOnTwoDirections) {
    using T = Tangent<double, 2>;
    T x(1.5, {1.0, 0.0});
    T y(2.5, {0.0, 1.0});
    T f = x * y + x - y / x;
    EXPECT_NEAR(f.value, 1.5 * 2.5 + 1.5 - 2.5 / 1.5, 1e-15);
    EXPECT_NEAR(f.d[0], 2.5 + 1.0 + 2.5 / (1.5 * 1.5), 1e-15);
    EXPECT_NEAR(f.d[1], 1.5 - 1.0 / 1.5, 1e-15);
}

TEST(ForwardScalarTest, ElementaryFunctions) {
    using T = Tangent<double, 1>;
    T x(0.25, {1.0});
    T g = exp(sqrt(x));
    EXPECT_DOUBLE_EQ(g.value, std::exp(0.5));
    EXPECT_NEAR(g.d[0], std::exp(0.5) / (2.0 * 0.5), 1e-15);
    T l = log(x);
    EXPECT_DOUBLE_EQ(l.value, std::log(0.25));
    EXPECT_DOUBLE_EQ(l.d[0], 4.0);
}

TEST(ForwardScalarTest, BranchPinnedByPrimal) {
    using T = Tangent<double, 1>;
    T a(1.0, {1.0});
    T b(2.0, {0.0});
    T lo = fmin(a, b);
    EXPECT_DOUBLE_EQ(lo.value, 1.0);
    EXPECT_DOUBLE_EQ(lo.d[0], 1.0);
    T hi = fmax(a, b);
    EXPECT_DOUBLE_EQ(hi.value, 2.0);
    EXPECT_DOUBLE_EQ(hi.d[0], 0.0);
}

TEST(ForwardScalarTest, AbsGradientFollowsSign) {
    using T = Tangent<double, 1>;
    T x(-3.0, {1.0});
    T a = fabs(x);
    EXPECT_DOUBLE_EQ(a.value, 3.0);
    EXPECT_DOUBLE_EQ(a.d[0], -1.0);
}

TEST(ForwardScalarTest, CompoundAssignmentMatchesFreshExpression) {
    using T = Tangent<double, 2>;
    T x(0.4, {1.0, 0.0});
    T y(1.2, {0.0, 1.0});
    T acc(0.0);
    acc += x * y;
    acc -= x;
    acc /= y;
    T fresh = (x * y - x) / y;
    EXPECT_DOUBLE_EQ(acc.value, fresh.value);
    EXPECT_DOUBLE_EQ(acc.d[0], fresh.d[0]);
    EXPECT_DOUBLE_EQ(acc.d[1], fresh.d[1]);
}

} // namespace
