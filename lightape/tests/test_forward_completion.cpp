#include <cmath>
#include <limits>

#include "lightape/lightape.h"
#include <gtest/gtest.h>

namespace {

using lightape::digamma;
using lightape::Tangent;
using lightape::trigamma;

constexpr double kZeta3 = 1.2020569031595942854;

/// Forward tangent of a single-direction input against the analytic derivative.
template <typename F, typename DF>
void CheckDerivative(const char* label, double x, const F& f, const DF& analytic,
                     double tol = 1e-12) {
    SCOPED_TRACE(::testing::Message() << label << " at x=" << x);
    Tangent<double, 1> t(x, {1.0});
    EXPECT_NEAR(f(t).d[0], analytic(x), tol);
}

/// Second derivative from a two-level nested Tangent.
template <typename F, typename D2F>
void CheckSecondDerivative(const char* label, double x, const F& f, const D2F& analytic,
                           double tol = 1e-10) {
    SCOPED_TRACE(::testing::Message() << label << " at x=" << x);
    using Inner = Tangent<double, 1>;
    using Outer = Tangent<Inner, 1>;
    Outer t;
    t.value.value = x;
    t.value.d[0] = 1.0;
    t.d[0].value = 1.0;
    t.d[0].d[0] = 0.0;
    EXPECT_NEAR(f(t).d[0].d[0], analytic(x), tol);
}

/// Third derivative from a three-level nested Tangent.
template <typename F, typename D3F>
void CheckThirdDerivative(const char* label, double x, const F& f, const D3F& analytic,
                          double tol = 1e-10) {
    SCOPED_TRACE(::testing::Message() << label << " at x=" << x);
    using L1 = Tangent<double, 1>;
    using L2 = Tangent<L1, 1>;
    using L3 = Tangent<L2, 1>;
    L3 t;
    t.value.value.value = x;
    t.value.value.d[0] = 1.0;
    t.value.d[0].value = 1.0;
    t.d[0].value.value = 1.0;
    EXPECT_NEAR(f(t).d[0].d[0].d[0], analytic(x), tol);
}

TEST(ForwardCompletionTest, Hyperbolic) {
    CheckDerivative(
        "sinh", 0.7, [](const auto& x) { return sinh(x); }, [](double x) { return std::cosh(x); });
    CheckDerivative(
        "sinh", -1.3, [](const auto& x) { return sinh(x); }, [](double x) { return std::cosh(x); });
    CheckDerivative(
        "cosh", 0.7, [](const auto& x) { return cosh(x); }, [](double x) { return std::sinh(x); });
    CheckDerivative(
        "cosh", -1.3, [](const auto& x) { return cosh(x); }, [](double x) { return std::sinh(x); });
    CheckDerivative(
        "tanh", 0.4, [](const auto& x) { return tanh(x); },
        [](double x) { return 1.0 - std::tanh(x) * std::tanh(x); });
    CheckDerivative(
        "tanh", -2.1, [](const auto& x) { return tanh(x); },
        [](double x) { return 1.0 - std::tanh(x) * std::tanh(x); });

    // Values too, at a point where std::sinh/cosh are exact to the last bit.
    Tangent<double, 1> t(0.0, {1.0});
    EXPECT_DOUBLE_EQ(sinh(t).value, 0.0);
    EXPECT_DOUBLE_EQ(sinh(t).d[0], 1.0);
    EXPECT_DOUBLE_EQ(cosh(t).value, 1.0);
    EXPECT_DOUBLE_EQ(cosh(t).d[0], 0.0);
    EXPECT_DOUBLE_EQ(tanh(t).value, 0.0);
    EXPECT_DOUBLE_EQ(tanh(t).d[0], 1.0);
}

TEST(ForwardCompletionTest, HypotBinaryAndMixed) {
    using T = Tangent<double, 2>;
    T a(3.0, {1.0, 0.0});
    T b(-4.0, {0.0, 1.0});
    const T r = hypot(a, b);
    EXPECT_DOUBLE_EQ(r.value, 5.0);
    EXPECT_DOUBLE_EQ(r.d[0], 3.0 / 5.0);
    EXPECT_DOUBLE_EQ(r.d[1], -4.0 / 5.0);

    // Degenerate input: finite primal, derivative pinned to zero.
    const T z = hypot(T(0.0, {1.0, 0.0}), T(0.0, {0.0, 1.0}));
    EXPECT_DOUBLE_EQ(z.value, 0.0);
    EXPECT_DOUBLE_EQ(z.d[0], 0.0);
    EXPECT_DOUBLE_EQ(z.d[1], 0.0);

    // Mixed overloads: scalar is node-less.
    Tangent<double, 1> x(3.0, {1.0});
    const auto ra = hypot(x, -4.0);
    EXPECT_DOUBLE_EQ(ra.value, 5.0);
    EXPECT_DOUBLE_EQ(ra.d[0], 3.0 / 5.0);
    const auto rb = hypot(3.0, Tangent<double, 1>(-4.0, {1.0}));
    EXPECT_DOUBLE_EQ(rb.value, 5.0);
    EXPECT_DOUBLE_EQ(rb.d[0], -4.0 / 5.0);
}

TEST(ForwardCompletionTest, Atan2BinaryAndMixed) {
    auto check = [](double y, double x) {
        using T = Tangent<double, 2>;
        const T r = atan2(T(y, {1.0, 0.0}), T(x, {0.0, 1.0}));
        const double d = y * y + x * x;
        EXPECT_NEAR(r.value, std::atan2(y, x), 1e-15);
        EXPECT_NEAR(r.d[0], x / d, 1e-15);
        EXPECT_NEAR(r.d[1], -y / d, 1e-15);
    };
    check(1.0, 2.0);
    check(-3.0, 4.0);
    check(-0.5, -2.5);

    Tangent<double, 1> y(1.0, {1.0});
    const auto r1 = atan2(y, 2.0);
    EXPECT_NEAR(r1.value, std::atan2(1.0, 2.0), 1e-15);
    EXPECT_DOUBLE_EQ(r1.d[0], 2.0 / 5.0);
    const auto r2 = atan2(1.0, Tangent<double, 1>(2.0, {1.0}));
    EXPECT_NEAR(r2.value, std::atan2(1.0, 2.0), 1e-15);
    EXPECT_DOUBLE_EQ(r2.d[0], -1.0 / 5.0);
}

TEST(ForwardCompletionTest, FmaIsComposite) {
    using T = Tangent<double, 3>;
    const T a(1.5, {1.0, 0.0, 0.0});
    const T b(-2.0, {0.0, 1.0, 0.0});
    const T c(0.75, {0.0, 0.0, 1.0});
    const T got = fma(a, b, c);
    const T want = a * b + c;
    EXPECT_DOUBLE_EQ(got.value, want.value);
    EXPECT_DOUBLE_EQ(got.d[0], want.d[0]);
    EXPECT_DOUBLE_EQ(got.d[1], want.d[1]);
    EXPECT_DOUBLE_EQ(got.d[2], want.d[2]);
    EXPECT_DOUBLE_EQ(got.value, 1.5 * -2.0 + 0.75);
    EXPECT_DOUBLE_EQ(got.d[0], -2.0);
    EXPECT_DOUBLE_EQ(got.d[1], 1.5);
    EXPECT_DOUBLE_EQ(got.d[2], 1.0);
}

TEST(ForwardCompletionTest, CopysignSignedZero) {
    const Tangent<double, 1> mag(3.0, {1.0});
    const Tangent<double, 1> neg(-2.0, {0.0});
    const Tangent<double, 1> pos(2.0, {0.0});

    const auto rn = copysign(mag, neg);
    EXPECT_DOUBLE_EQ(rn.value, -3.0);
    EXPECT_DOUBLE_EQ(rn.d[0], -1.0);
    const auto rp = copysign(mag, pos);
    EXPECT_DOUBLE_EQ(rp.value, 3.0);
    EXPECT_DOUBLE_EQ(rp.d[0], 1.0);

    // Magnitude of `a` wins: a negative `a` comes out positive for b > 0.
    EXPECT_DOUBLE_EQ(copysign(Tangent<double, 1>(-3.0, {1.0}), pos).value, 3.0);

    // Signed zeros: the primal keeps the sign bit of b (std::copysign), so
    // copysign(3, -0.0) = -3 even though -0.0 >= 0.0.
    const auto negZero = copysign(mag, Tangent<double, 1>(-0.0, {0.0}));
    EXPECT_DOUBLE_EQ(negZero.value, -3.0);
    EXPECT_TRUE(std::signbit(negZero.value));
    const auto posZero = copysign(mag, Tangent<double, 1>(0.0, {0.0}));
    EXPECT_DOUBLE_EQ(posZero.value, 3.0);
    EXPECT_FALSE(std::signbit(posZero.value));
}

TEST(ForwardCompletionTest, LogGammaTGammaFirstDerivative) {
    const double xs[] = {0.5, 1.5, 3.7};
    for (const double x : xs) {
        CheckDerivative(
            "lgamma", x, [](const auto& t) { return lgamma(t); },
            [](double v) { return lightape::digamma(v); }, 1e-13);
        CheckDerivative(
            "tgamma", x, [](const auto& t) { return tgamma(t); },
            [](double v) { return std::tgamma(v) * lightape::digamma(v); }, 1e-12);
    }

    // The Tangent digamma/trigamma overloads themselves.
    Tangent<double, 1> t(2.0, {1.0});
    EXPECT_NEAR(digamma(t).value, digamma(2.0), 1e-15);
    EXPECT_NEAR(digamma(t).d[0], trigamma(2.0), 1e-14);
    EXPECT_NEAR(trigamma(t).value, trigamma(2.0), 1e-15);
}

TEST(ForwardCompletionTest, LogGammaTGammaSecondDerivative) {
    const double xs[] = {0.5, 1.5, 3.7};
    for (const double x : xs) {
        CheckSecondDerivative(
            "lgamma''", x, [](const auto& t) { return lgamma(t); },
            [](double v) { return lightape::trigamma(v); });
        CheckSecondDerivative(
            "tgamma''", x, [](const auto& t) { return tgamma(t); },
            [](double v) {
                const double d = lightape::digamma(v);
                return std::tgamma(v) * (d * d + lightape::trigamma(v));
            });
    }

    Tangent<double, 1> t(2.0, {1.0});
    EXPECT_NEAR(digamma(t).d[0], trigamma(2.0), 1e-14);
}

TEST(ForwardCompletionTest, LogGammaThirdDerivative) {
    // lgamma''' = tetragamma, checked at arguments with exact references:
    // psi''(1) = -2 zeta(3), psi''(2) = psi''(1) + 2, psi''(1/2) = -14 zeta(3).
    CheckThirdDerivative(
        "lgamma'''", 1.0, [](const auto& t) { return lgamma(t); },
        [](double) { return -2.0 * kZeta3; }, 1e-13);
    CheckThirdDerivative(
        "lgamma'''", 2.0, [](const auto& t) { return lgamma(t); },
        [](double) { return -2.0 * kZeta3 + 2.0; }, 1e-13);
    CheckThirdDerivative(
        "lgamma'''", 0.5, [](const auto& t) { return lgamma(t); },
        [](double) { return -14.0 * kZeta3; }, 1e-12);
}

TEST(ForwardCompletionTest, MixedComparisons) {
    const Tangent<double, 1> x(2.5, {1.0});
    EXPECT_TRUE(x < 3.0);
    EXPECT_TRUE(2.0 < x);
    EXPECT_TRUE(x > 2.0);
    EXPECT_TRUE(3.0 > x);
    EXPECT_TRUE(x <= 2.5);
    EXPECT_TRUE(2.5 <= x);
    EXPECT_TRUE(x >= 2.5);
    EXPECT_TRUE(2.5 >= x);
    EXPECT_TRUE(x == 2.5);
    EXPECT_TRUE(2.5 == x);
    EXPECT_TRUE(x != 2.0);
    EXPECT_TRUE(2.0 != x);
    EXPECT_FALSE(x < 2.5);
    EXPECT_FALSE(x == 3.0);
    EXPECT_FALSE(x > 2.5);

    // Deduction from a generic body with an S literal.
    const auto belowHalf = [](const auto& v) { return v < 0.5; };
    EXPECT_TRUE(belowHalf(Tangent<double, 1>(0.25, {1.0})));
    EXPECT_FALSE(belowHalf(Tangent<double, 1>(0.75, {1.0})));
}

TEST(ForwardCompletionTest, Predicates) {
    const double nan = std::numeric_limits<double>::quiet_NaN();
    const double inf = std::numeric_limits<double>::infinity();
    const Tangent<double, 1> finite(1.0, {1.0});
    EXPECT_FALSE(isnan(finite));
    EXPECT_FALSE(isinf(finite));
    EXPECT_TRUE(isfinite(finite));
    EXPECT_TRUE(isnan(Tangent<double, 1>(nan, {0.0})));
    EXPECT_FALSE(isfinite(Tangent<double, 1>(nan, {0.0})));
    EXPECT_TRUE(isinf(Tangent<double, 1>(inf, {0.0})));
    EXPECT_FALSE(isfinite(Tangent<double, 1>(inf, {0.0})));
}

TEST(ForwardCompletionTest, FloatInstantiation) {
    using F = Tangent<float, 1>;
    const F x(0.5f, {1.0f});

    const F s = sinh(x);
    EXPECT_FLOAT_EQ(s.value, std::sinh(0.5f));
    EXPECT_FLOAT_EQ(s.d[0], std::cosh(0.5f));

    const F h = hypot(x, F(0.3f, {0.0f}));
    EXPECT_FLOAT_EQ(h.value, std::hypot(0.5f, 0.3f));
    EXPECT_FLOAT_EQ(h.d[0], 0.5f / std::hypot(0.5f, 0.3f));

    const F a = atan2(F(1.0f, {1.0f}), x);
    EXPECT_FLOAT_EQ(a.value, std::atan2(1.0f, 0.5f));
    EXPECT_FLOAT_EQ(a.d[0], -0.5f / (1.0f + 0.25f));

    const F fm = fma(x, F(2.0f, {1.0f}), F(1.0f, {1.0f}));
    EXPECT_FLOAT_EQ(fm.value, 2.0f);
    EXPECT_FLOAT_EQ(fm.d[0], 3.5f);

    const F cs = copysign(F(3.0f, {1.0f}), F(-1.0f, {0.0f}));
    EXPECT_FLOAT_EQ(cs.value, -3.0f);
    EXPECT_FLOAT_EQ(cs.d[0], -1.0f);

    const F lg = lgamma(F(2.0f, {1.0f}));
    EXPECT_FLOAT_EQ(lg.value, std::lgamma(2.0f));
    EXPECT_NEAR(lg.d[0], static_cast<float>(lightape::digamma(2.0)), 1e-6f);

    const F tg = tgamma(F(2.0f, {1.0f}));
    EXPECT_FLOAT_EQ(tg.value, std::tgamma(2.0f));
    EXPECT_NEAR(tg.d[0], static_cast<float>(std::tgamma(2.0) * lightape::digamma(2.0)), 1e-5f);

    EXPECT_TRUE(x < 1.0f);
    EXPECT_TRUE(0.25f < x);
    EXPECT_TRUE(x == 0.5f);
    EXPECT_TRUE(isfinite(x));
}

} // namespace
