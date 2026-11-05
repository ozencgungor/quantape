#include <cmath>

#include "lightape/lightape.h"
#include <gtest/gtest.h>

namespace {

using lightape::Tangent;

constexpr double kTwoOverSqrtPi = 1.12837916709551257390;
constexpr double kPi = 3.14159265358979323846;

/// Forward tangent of a single-direction input against the analytic derivative.
template <typename F, typename DF>
void CheckDerivative(const char* label, double x, const F& f, const DF& analytic,
                     double tol = 1e-12) {
    SCOPED_TRACE(::testing::Message() << label << " at x=" << x);
    Tangent<double, 1> t(x, {1.0});
    EXPECT_NEAR(f(t).d[0], analytic(x), tol);
}

TEST(ForwardOpsTest, PowTangentTangent) {
    using T = Tangent<double, 2>;
    T a(1.3, {1.0, 0.0});
    T b(2.4, {0.0, 1.0});
    T f = pow(a, b);
    const double value = std::pow(1.3, 2.4);
    EXPECT_NEAR(f.value, value, 1e-15);
    EXPECT_NEAR(f.d[0], 2.4 * std::pow(1.3, 1.4), 1e-13);
    EXPECT_NEAR(f.d[1], value * std::log(1.3), 1e-14);
}

TEST(ForwardOpsTest, PowTangentScalar) {
    CheckDerivative(
        "pow(x, 1.3)", 1.7, [](const auto& x) { return pow(x, 1.3); },
        [](double x) { return 1.3 * std::pow(x, 0.3); });
    CheckDerivative(
        "pow(x, -0.4)", 0.8, [](const auto& x) { return pow(x, -0.4); },
        [](double x) { return -0.4 * std::pow(x, -1.4); });
    // Integer exponent on a negative base: the real branch exists.
    CheckDerivative(
        "pow(x, 3)", -2.0, [](const auto& x) { return pow(x, 3.0); },
        [](double x) { return 3.0 * std::pow(x, 2.0); });
}

TEST(ForwardOpsTest, PowScalarTangent) {
    CheckDerivative(
        "pow(2.5, x)", 1.4, [](const auto& x) { return pow(2.5, x); },
        [](double x) { return std::pow(2.5, x) * std::log(2.5); });
    CheckDerivative(
        "pow(0.3, x)", -0.7, [](const auto& x) { return pow(0.3, x); },
        [](double x) { return std::pow(0.3, x) * std::log(0.3); });
}

TEST(ForwardOpsTest, SinCosTan) {
    CheckDerivative(
        "sin", 0.3, [](const auto& x) { return sin(x); }, [](double x) { return std::cos(x); });
    CheckDerivative(
        "sin", -1.1, [](const auto& x) { return sin(x); }, [](double x) { return std::cos(x); });
    CheckDerivative(
        "cos", 0.7, [](const auto& x) { return cos(x); }, [](double x) { return -std::sin(x); });
    CheckDerivative(
        "cos", 2.5, [](const auto& x) { return cos(x); }, [](double x) { return -std::sin(x); });
    CheckDerivative(
        "tan", 0.4, [](const auto& x) { return tan(x); },
        [](double x) { return 1.0 + std::tan(x) * std::tan(x); });
    CheckDerivative(
        "tan", -0.9, [](const auto& x) { return tan(x); },
        [](double x) { return 1.0 + std::tan(x) * std::tan(x); });
}

TEST(ForwardOpsTest, InverseTrig) {
    CheckDerivative(
        "asin", 1e-8, [](const auto& x) { return asin(x); },
        [](double x) { return 1.0 / std::sqrt(1.0 - x * x); });
    CheckDerivative(
        "asin", 0.3, [](const auto& x) { return asin(x); },
        [](double x) { return 1.0 / std::sqrt(1.0 - x * x); });
    CheckDerivative(
        "asin", -0.6, [](const auto& x) { return asin(x); },
        [](double x) { return 1.0 / std::sqrt(1.0 - x * x); });
    CheckDerivative(
        "acos", 0.3, [](const auto& x) { return acos(x); },
        [](double x) { return -1.0 / std::sqrt(1.0 - x * x); });
    CheckDerivative(
        "acos", -0.5, [](const auto& x) { return acos(x); },
        [](double x) { return -1.0 / std::sqrt(1.0 - x * x); });
    CheckDerivative(
        "atan", 0.7, [](const auto& x) { return atan(x); },
        [](double x) { return 1.0 / (1.0 + x * x); });
    CheckDerivative(
        "atan", -2.0, [](const auto& x) { return atan(x); },
        [](double x) { return 1.0 / (1.0 + x * x); });
}

TEST(ForwardOpsTest, Expm1Log1p) {
    // Near zero the derivative is exp(x) ~ 1; expm1's primal avoids cancellation.
    CheckDerivative(
        "expm1", 1e-9, [](const auto& x) { return expm1(x); },
        [](double x) { return std::exp(x); });
    CheckDerivative(
        "expm1", 0.5, [](const auto& x) { return expm1(x); }, [](double x) { return std::exp(x); });
    CheckDerivative(
        "expm1", -0.7, [](const auto& x) { return expm1(x); },
        [](double x) { return std::exp(x); });
    CheckDerivative(
        "log1p", 1e-9, [](const auto& x) { return log1p(x); },
        [](double x) { return 1.0 / (1.0 + x); });
    CheckDerivative(
        "log1p", 0.5, [](const auto& x) { return log1p(x); },
        [](double x) { return 1.0 / (1.0 + x); });
    CheckDerivative(
        "log1p", -0.4, [](const auto& x) { return log1p(x); },
        [](double x) { return 1.0 / (1.0 + x); });
}

TEST(ForwardOpsTest, ErfErfc) {
    CheckDerivative(
        "erf", 0.0, [](const auto& x) { return erf(x); },
        [](double x) { return kTwoOverSqrtPi * std::exp(-x * x); });
    CheckDerivative(
        "erf", 0.5, [](const auto& x) { return erf(x); },
        [](double x) { return kTwoOverSqrtPi * std::exp(-x * x); });
    CheckDerivative(
        "erf", -1.2, [](const auto& x) { return erf(x); },
        [](double x) { return kTwoOverSqrtPi * std::exp(-x * x); });
    // Negative argument keeps erfc ~ 2, so the derivative is well conditioned.
    CheckDerivative(
        "erfc", -1.0, [](const auto& x) { return erfc(x); },
        [](double x) { return -kTwoOverSqrtPi * std::exp(-x * x); });
    CheckDerivative(
        "erfc", 0.3, [](const auto& x) { return erfc(x); },
        [](double x) { return -kTwoOverSqrtPi * std::exp(-x * x); });
    CheckDerivative(
        "erfc", 2.0, [](const auto& x) { return erfc(x); },
        [](double x) { return -kTwoOverSqrtPi * std::exp(-x * x); });
}

TEST(ForwardOpsTest, NormCdf) {
    const auto derivative = [](double x) { return std::exp(-0.5 * x * x) / std::sqrt(2.0 * kPi); };
    CheckDerivative("normcdf", 0.0, [](const auto& x) { return normcdf(x); }, derivative);
    CheckDerivative("normcdf", 1.5, [](const auto& x) { return normcdf(x); }, derivative);
    CheckDerivative("normcdf", -2.0, [](const auto& x) { return normcdf(x); }, derivative);

    Tangent<double, 1> zero(0.0, {1.0});
    EXPECT_DOUBLE_EQ(normcdf(zero).value, 0.5);
}

TEST(ForwardOpsTest, FloatInstantiation) {
    using TF = Tangent<float, 1>;
    TF x(0.5f, {1.0f});

    TF s = sin(x);
    EXPECT_FLOAT_EQ(s.value, std::sin(0.5f));
    EXPECT_FLOAT_EQ(s.d[0], std::cos(0.5f));

    TF p = pow(x, 3.0f);
    EXPECT_FLOAT_EQ(p.value, 0.125f);
    EXPECT_FLOAT_EQ(p.d[0], 0.75f);

    TF l = log1p(x);
    EXPECT_FLOAT_EQ(l.value, std::log1p(0.5f));

    TF g = normcdf(x);
    EXPECT_NEAR(g.value, 0.5f * std::erfc(-0.5f / std::sqrt(2.0f)), 1e-6f);
    EXPECT_NEAR(g.d[0], std::exp(-0.125f) / std::sqrt(2.0f * static_cast<float>(kPi)), 1e-6f);
}

} // namespace
