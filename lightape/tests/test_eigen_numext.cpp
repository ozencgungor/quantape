#include <Eigen/Core>

#include <cmath>
#include <limits>

#include "lightape/eigen/ForwardScalarEigen.h"
#include "lightape/eigen/LeanReverseEigen.h"
#include "lightape/lightape.h"
#include <gtest/gtest.h>

namespace {

using lightape::RevScalar;
using lightape::RevTape;
using lightape::Tangent;

using RevVector = Eigen::Matrix<RevScalar, Eigen::Dynamic, 1>;
using Tan2 = Tangent<double, 2>;
using TanVector = Eigen::Matrix<Tan2, Eigen::Dynamic, 1>;

constexpr double kNaN = std::numeric_limits<double>::quiet_NaN();
constexpr double kInf = std::numeric_limits<double>::infinity();

class EigenNumextRevTest : public ::testing::Test {
protected:
    void SetUp() override { RevTape::active().clear(); }
    void TearDown() override { RevTape::active().clear(); }
};

// ── predicates ──

TEST_F(EigenNumextRevTest, PredicatesRevScalar) {
    RevVector x(3);
    x(0) = RevScalar(0.5);
    x(1) = RevScalar(-2.0);
    x(2) = RevScalar(3.25);

    EXPECT_TRUE(Eigen::numext::isfinite(x(0)));
    EXPECT_FALSE(Eigen::numext::isnan(x(0)));
    EXPECT_FALSE(Eigen::numext::isinf(x(0)));
    for (int i = 0; i < x.size(); ++i) {
        EXPECT_TRUE(Eigen::numext::isfinite(x(i)));
    }

    EXPECT_TRUE(x.allFinite());
    EXPECT_FALSE(x.hasNaN());
    EXPECT_TRUE(x.array().isFinite().all());
    EXPECT_FALSE(x.array().isNaN().any());
    EXPECT_FALSE(x.array().isInf().any());

    RevVector withNaN = x;
    withNaN(1) = RevScalar(kNaN);
    EXPECT_TRUE(Eigen::numext::isnan(withNaN(1)));
    EXPECT_FALSE(Eigen::numext::isfinite(withNaN(1)));
    EXPECT_FALSE(withNaN.allFinite());
    EXPECT_TRUE(withNaN.hasNaN());
    EXPECT_TRUE(withNaN.array().isNaN().any());

    RevVector withInf = x;
    withInf(0) = RevScalar(kInf);
    EXPECT_TRUE(Eigen::numext::isinf(withInf(0)));
    EXPECT_FALSE(Eigen::numext::isfinite(withInf(0)));
    EXPECT_FALSE(Eigen::numext::isnan(withInf(0)));
    EXPECT_FALSE(withInf.allFinite());
    EXPECT_FALSE(withInf.hasNaN()); // +inf is not NaN
}

TEST(EigenNumextTest, PredicatesTangent) {
    TanVector x(3);
    x(0) = Tan2(0.5, {1.0, 0.0});
    x(1) = Tan2(-2.0, {0.0, 1.0});
    x(2) = Tan2(3.25, {0.0, 0.0});

    EXPECT_TRUE(Eigen::numext::isfinite(x(0)));
    EXPECT_FALSE(Eigen::numext::isnan(x(0)));
    EXPECT_FALSE(Eigen::numext::isinf(x(0)));

    EXPECT_TRUE(x.allFinite());
    EXPECT_FALSE(x.hasNaN());
    EXPECT_TRUE(x.array().isFinite().all());
    EXPECT_FALSE(x.array().isNaN().any());
    EXPECT_FALSE(x.array().isInf().any());

    TanVector withNaN = x;
    withNaN(1) = Tan2(kNaN, {0.0, 0.0});
    EXPECT_TRUE(Eigen::numext::isnan(withNaN(1)));
    EXPECT_FALSE(withNaN.allFinite());
    EXPECT_TRUE(withNaN.hasNaN());

    TanVector withInf = x;
    withInf(0) = Tan2(kInf, {0.0, 0.0});
    EXPECT_TRUE(Eigen::numext::isinf(withInf(0)));
    EXPECT_FALSE(withInf.allFinite());
    EXPECT_FALSE(withInf.hasNaN());
}

// ── array().tanh() / array().pow(2.0): value and gradient ──

TEST_F(EigenNumextRevTest, ArrayTanhPowRevScalar) {
    const double v[3] = {-0.75, 0.25, 1.5};
    RevVector x(3);
    RevTape& tape = RevTape::active();
    for (int i = 0; i < 3; ++i) {
        x(i) = RevScalar(v[i], tape.input(v[i]));
    }

    RevScalar f = x.array().tanh().sum() + x.array().pow(2.0).sum();
    double expected = 0.0;
    for (int i = 0; i < 3; ++i) {
        expected += std::tanh(v[i]) + v[i] * v[i];
    }
    EXPECT_NEAR(f.value, expected, 1e-14);

    tape.reverse(f.node);
    for (int i = 0; i < 3; ++i) {
        const double t = std::tanh(v[i]);
        EXPECT_NEAR(tape.adjoint(x(i).node), (1.0 - t * t) + 2.0 * v[i], 1e-13)
            << "coefficient " << i;
    }
}

TEST(EigenNumextTest, ArrayTanhPowTangent) {
    const double v[2] = {-0.5, 0.75};
    TanVector x(2);
    x(0) = Tan2(v[0], {1.0, 0.0});
    x(1) = Tan2(v[1], {0.0, 1.0});

    Tan2 f = x.array().tanh().sum() + x.array().pow(2.0).sum();
    double expected = 0.0;
    double d0 = 0.0;
    double d1 = 0.0;
    for (int i = 0; i < 2; ++i) {
        const double t = std::tanh(v[i]);
        expected += t + v[i] * v[i];
        const double partial = (1.0 - t * t) + 2.0 * v[i];
        if (i == 0) {
            d0 += partial;
        } else {
            d1 += partial;
        }
    }
    EXPECT_NEAR(f.value, expected, 1e-14);
    EXPECT_NEAR(f.d[0], d0, 1e-13);
    EXPECT_NEAR(f.d[1], d1, 1e-13);
}

// `array().pow(<int>)` regressed to the generic functor once the integer
// traits exist; this guards the ADL bridge for integer exponents.
TEST_F(EigenNumextRevTest, ArrayPowIntegerExponentRevScalar) {
    const double v[3] = {0.5, 1.25, -2.0};
    RevVector x(3);
    RevTape& tape = RevTape::active();
    for (int i = 0; i < 3; ++i) {
        x(i) = RevScalar(v[i], tape.input(v[i]));
    }

    RevScalar f = x.array().pow(2).sum();
    double expected = 0.0;
    for (int i = 0; i < 3; ++i) {
        expected += v[i] * v[i];
    }
    EXPECT_NEAR(f.value, expected, 1e-14);
    tape.reverse(f.node);
    for (int i = 0; i < 3; ++i) {
        EXPECT_NEAR(tape.adjoint(x(i).node), 2.0 * v[i], 1e-13);
    }
}

TEST(EigenNumextTest, ArrayPowIntegerExponentTangent) {
    const double v[2] = {0.5, 1.25};
    TanVector x(2);
    x(0) = Tan2(v[0], {1.0, 0.0});
    x(1) = Tan2(v[1], {0.0, 1.0});

    Tan2 f = x.array().pow(2).sum();
    EXPECT_NEAR(f.value, v[0] * v[0] + v[1] * v[1], 1e-14);
    EXPECT_NEAR(f.d[0], 2.0 * v[0], 1e-13);
    EXPECT_NEAR(f.d[1], 2.0 * v[1], 1e-13);
}

// ── numext::abs2 / conj / real / imag ──

TEST_F(EigenNumextRevTest, Abs2ConjRealImagRevScalar) {
    const double v = -1.25;
    RevTape& tape = RevTape::active();
    RevScalar x(v, tape.input(v));

    RevScalar a2 = Eigen::numext::abs2(x);
    RevScalar cj = Eigen::numext::conj(x);
    RevScalar re = Eigen::numext::real(x);
    RevScalar im = Eigen::numext::imag(x);
    EXPECT_DOUBLE_EQ(a2.value, v * v);
    EXPECT_DOUBLE_EQ(cj.value, v);
    EXPECT_DOUBLE_EQ(re.value, v);
    EXPECT_DOUBLE_EQ(im.value, 0.0);

    // imag is a plain constant (no node): d/dx (x^2 + x + x) = 2v + 2.
    RevScalar f = a2 + cj + re;
    EXPECT_NEAR(f.value, v * v + 2.0 * v, 1e-15);
    tape.reverse(f.node);
    EXPECT_NEAR(tape.adjoint(x.node), 2.0 * v + 2.0, 1e-14);
}

TEST_F(EigenNumextRevTest, MatrixSquaredNormRevScalar) {
    const double v[3] = {0.5, -1.5, 2.25};
    RevVector x(3);
    RevTape& tape = RevTape::active();
    for (int i = 0; i < 3; ++i) {
        x(i) = RevScalar(v[i], tape.input(v[i]));
    }

    RevScalar s = x.squaredNorm();
    double expected = 0.0;
    for (int i = 0; i < 3; ++i) {
        expected += v[i] * v[i];
    }
    EXPECT_NEAR(s.value, expected, 1e-14);
    tape.reverse(s.node);
    for (int i = 0; i < 3; ++i) {
        EXPECT_NEAR(tape.adjoint(x(i).node), 2.0 * v[i], 1e-13);
    }
}

TEST(EigenNumextTest, Abs2ConjRealImagTangent) {
    const double v = -1.25;
    const double r0 = 1.0;
    const double r1 = 0.5;
    Tan2 x(v, {r0, r1});

    Tan2 a2 = Eigen::numext::abs2(x);
    Tan2 cj = Eigen::numext::conj(x);
    Tan2 re = Eigen::numext::real(x);
    Tan2 im = Eigen::numext::imag(x);
    EXPECT_DOUBLE_EQ(a2.value, v * v);
    EXPECT_DOUBLE_EQ(a2.d[0], 2.0 * v * r0);
    EXPECT_DOUBLE_EQ(a2.d[1], 2.0 * v * r1);
    EXPECT_DOUBLE_EQ(cj.value, v);
    EXPECT_DOUBLE_EQ(cj.d[0], r0);
    EXPECT_DOUBLE_EQ(cj.d[1], r1);
    EXPECT_DOUBLE_EQ(re.value, v);
    EXPECT_DOUBLE_EQ(re.d[0], r0);
    EXPECT_DOUBLE_EQ(re.d[1], r1);
    EXPECT_DOUBLE_EQ(im.value, 0.0);
    EXPECT_DOUBLE_EQ(im.d[0], 0.0);
    EXPECT_DOUBLE_EQ(im.d[1], 0.0);
}

// ── numext::pow with integer and scalar exponents ──

TEST_F(EigenNumextRevTest, PowIntegerAndScalarExponentsRevScalar) {
    const double v = 2.5;
    RevTape& tape = RevTape::active();
    RevScalar x(v, tape.input(v));

    RevScalar pInt = Eigen::numext::pow(x, 2);
    RevScalar pLong = Eigen::numext::pow(x, 3L);
    RevScalar pDouble = Eigen::numext::pow(x, 2.5);
    RevScalar pBase = Eigen::numext::pow(2, x);

    const double expected = v * v + v * v * v + std::pow(v, 2.5) + std::pow(2.0, v);
    RevScalar f = pInt + pLong + pDouble + pBase;
    EXPECT_NEAR(f.value, expected, 1e-13);

    tape.reverse(f.node);
    const double expectedD =
        2.0 * v + 3.0 * v * v + 2.5 * std::pow(v, 1.5) + std::pow(2.0, v) * std::log(2.0);
    EXPECT_NEAR(tape.adjoint(x.node), expectedD, 1e-12);
}

TEST(EigenNumextTest, PowIntegerAndScalarExponentsTangent) {
    const double v = 2.5;
    const double r0 = 1.0;
    const double r1 = -0.25;
    Tan2 x(v, {r0, r1});

    Tan2 pInt = Eigen::numext::pow(x, 2);
    Tan2 pLong = Eigen::numext::pow(x, 3L);
    Tan2 pDouble = Eigen::numext::pow(x, 2.5);
    Tan2 pBase = Eigen::numext::pow(2, x);

    const double partial =
        2.0 * v + 3.0 * v * v + 2.5 * std::pow(v, 1.5) + std::pow(2.0, v) * std::log(2.0);
    Tan2 f = pInt + pLong + pDouble + pBase;
    EXPECT_NEAR(f.value, v * v + v * v * v + std::pow(v, 2.5) + std::pow(2.0, v), 1e-13);
    EXPECT_NEAR(f.d[0], partial * r0, 1e-12);
    EXPECT_NEAR(f.d[1], partial * r1, 1e-12);
}

} // namespace
