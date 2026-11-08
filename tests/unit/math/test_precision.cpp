// test_precision.cpp — in-house high-precision (double-double) gates
//
// 1. arithmetic: error-free transformations, cancellation, products,
//    division, sqrt, exp/log round-trips, sin/cos identities
// 2. special functions: Si/Ci anchors against their defining series
// 3. an oscillatory integral whose exact value is a complex double-double
//    (k! / (1 - i w)^(k+1)) computed with our own quadrature-free formula

#include "quantape/math/Integrals/DoubleExponentialIntegrator.h"
#include "quantape/math/Precision/DoubleDouble.h"

#include <algorithm>
#include <cmath>
#include <initializer_list>
#include <limits>
#include <ostream>
#include <string>

#include "support/GtestSupport.h"

using quantape::math::DoubleDouble;

namespace {

// Relative gate with the legacy semantics: the error is the double part of the
// double-double difference, and the denominator floor is 1e-300.
void checkRel(const std::string& label, const DoubleDouble& got, const DoubleDouble& expected,
              double tol) {
    const double denom = std::max(1e-300, std::fabs(expected.value()));
    const double err = std::fabs((got - expected).value()) / denom;
    CHECK_CLOSE(label, err, 0.0, tol);
}

// Si(x) = sum (-1)^k x^(2k+1)/((2k+1)! (2k+1)) in double-double.
DoubleDouble siSeries(const DoubleDouble& x) {
    DoubleDouble term = x;
    DoubleDouble sum = x;
    const DoubleDouble x2 = x * x;
    for (int k = 1; k < 60; ++k) {
        const double d = static_cast<double>(2 * k) * static_cast<double>(2 * k + 1);
        term = -term * x2 / DoubleDouble(d);
        const DoubleDouble add = term / DoubleDouble(static_cast<double>(2 * k + 1));
        sum = sum + add;
        if (add.hi == 0.0) {
            break;
        }
    }
    return sum;
}

struct SiAnchor {
    const char* id;
    const char* label;
    double x;
    double expected;
    double tol;
};

void PrintTo(const SiAnchor& anchor, std::ostream* os) {
    *os << anchor.id;
}

class PrecisionSiTest : public ::testing::TestWithParam<SiAnchor> {};

} // namespace

TEST(Precision, errorFreeArithmetic) {
    // Exact cancellation: (1 + 1e-30) - 1 = 1e-30 in double-double.
    const DoubleDouble big(1.0);
    const DoubleDouble tiny(1e-30);
    const DoubleDouble sum = big + tiny;
    checkRel("small addition", sum - big, tiny, 1e-20);

    // (a*b)/b round trip for awkward values
    double worst = 0.0;
    for (double a : {1.0 / 3.0, 1e-8, 1e8, 0.7, 12345.6789}) {
        for (double b : {7.0 / 11.0, 1e-3, 1e3, 1.9, 98765.4321}) {
            const DoubleDouble q = (DoubleDouble(a) * DoubleDouble(b)) / DoubleDouble(b);
            worst = std::max(worst, std::fabs((q - DoubleDouble(a)).value()) /
                                        std::max(1e-300, std::fabs(a)));
        }
    }
    SCOPED_TRACE("product round trip");
    EXPECT_LT(worst, 1e-30);

    // sqrt: (sqrt(x))^2 == x
    worst = 0.0;
    for (double x : {2.0, 1e-6, 1e6, 0.123456789012345, 42.0, 1e30}) {
        const DoubleDouble r = quantape::math::sqrt(DoubleDouble(x));
        worst = std::max(worst, std::fabs((r * r - DoubleDouble(x)).value()) / x);
    }
    SCOPED_TRACE("sqrt squared");
    EXPECT_LT(worst, 1e-30);
}

TEST(Precision, expLogRoundTrip) {
    // Absolute error at the double-double floor (near zero the logarithm's
    // absolute precision is the meaningful measure).
    double worst = 0.0;
    for (double x : {0.1, 1.0, 2.5, 17.0, 1e-4, 300.0}) {
        const DoubleDouble l = quantape::math::log(quantape::math::exp(DoubleDouble(x)));
        worst = std::max(worst, std::fabs((l - DoubleDouble(x)).value()));
    }
    SCOPED_TRACE("exp/log round trip");
    EXPECT_LT(worst, 1e-29);
}

TEST(Precision, trigonometricIdentities) {
    // sin^2 + cos^2 == 1
    double worst = 0.0;
    for (double x : {0.0, 0.3, 1.0, 3.0, 12.0, 1234.5678}) {
        const DoubleDouble s = quantape::math::sin(DoubleDouble(x));
        const DoubleDouble c = quantape::math::cos(DoubleDouble(x));
        worst = std::max(worst, std::fabs((s * s + c * c - DoubleDouble(1.0)).value()));
    }
    SCOPED_TRACE("sin^2 + cos^2");
    EXPECT_LT(worst, 1e-31);

    // sin(pi/2) == 1 (exact pi/2 as a double-double)
    const DoubleDouble sPi2 =
        quantape::math::sin(DoubleDouble(1.5707963267948966, 6.123233995736766e-17));
    checkRel("sin(pi/2)", sPi2, DoubleDouble(1.0), 1e-31);
}

TEST_P(PrecisionSiTest, seriesAnchors) {
    // Anchors from the high-precision literature.
    SCOPED_TRACE(GetParam().label);
    const DoubleDouble value = siSeries(DoubleDouble(GetParam().x));
    checkRel(GetParam().label, value, DoubleDouble(GetParam().expected), GetParam().tol);
}

INSTANTIATE_TEST_SUITE_P(
    Table, PrecisionSiTest,
    ::testing::Values(SiAnchor{"Si1", "Si(1)", 1.0, 0.94608307036718301494, 1e-15},
                      SiAnchor{"Si10", "Si(10)", 10.0, 1.6583475942188740493, 1e-14}),
    [](const ::testing::TestParamInfo<SiAnchor>& info) { return info.param.id; });

TEST(Precision, oscillatoryMomentsMatchClosedForm) {
    // int_0^inf e^{-x} x^k cos(w x) dx = Re k! / (1 - i w)^{k+1}, computed as
    // a complex double-double division chain, against a double-double
    // tanh-sinh quadrature of the same integrand (independent reference).
    auto moment = [](int k, double w) {
        DoubleDouble re(1.0);
        DoubleDouble im(0.0);
        const DoubleDouble denRe(1.0);
        const DoubleDouble denIm(-w);
        for (int j = 0; j <= k; ++j) {
            const DoubleDouble d = denRe * denRe + denIm * denIm;
            const DoubleDouble nre = re * denRe + im * denIm;
            const DoubleDouble nim = im * denRe - re * denIm;
            re = nre / d;
            im = nim / d;
        }
        for (int j = 2; j <= k; ++j) { // k! (exact in double for k <= 8)
            re = re * DoubleDouble(static_cast<double>(j));
            im = im * DoubleDouble(static_cast<double>(j));
        }
        return re;
    };

    const double w = 1.3;
    double worst = 0.0;
    for (int k = 0; k <= 8; ++k) {
        const DoubleDouble quadrature =
            quantape::math::integrateDoubleExponential<DoubleDouble>([&](const DoubleDouble& x) {
                DoubleDouble xk(1.0);
                for (int m = 0; m < k; ++m) {
                    xk = xk * x;
                }
                return quantape::math::exp(-x) * xk * quantape::math::cos(DoubleDouble(w) * x);
            });
        const DoubleDouble reference = moment(k, w);
        worst = std::max(worst, std::fabs((quadrature - reference).value()) /
                                    std::max(1e-300, std::fabs(reference.value())));
    }
    SCOPED_TRACE("oscillatory moments k=0..8");
    EXPECT_LT(worst, 1e-10);
}

namespace {

// Independent reference for log(1+d): the direct Maclaurin series
// d - d^2/2 + d^3/3 - ..., not the 2 atanh(d/(2+d)) form used by the library.
DoubleDouble log1pSeries(double d) {
    const DoubleDouble dd(d);
    const DoubleDouble negD(-d);
    DoubleDouble term = dd;
    DoubleDouble sum = dd;
    for (int k = 2; k < 200; ++k) {
        term = term * negD;
        const DoubleDouble add = term / DoubleDouble(static_cast<double>(k));
        sum = sum + add;
        if (add.hi == 0.0 && add.lo == 0.0) {
            break;
        }
    }
    return sum;
}

// Round-trip helper: scale by a power of two before squaring, so the
// double-double product's error term stays normal (the unscaled (sqrt(a))^2
// round trip cannot resolve below ~1e-24 for a ~ 1e-300).
DoubleDouble scalePow2(const DoubleDouble& a, int exponent) {
    return DoubleDouble(std::ldexp(a.hi, exponent), std::ldexp(a.lo, exponent));
}

} // namespace

// IEEE propagation for inf/nan/zero in sqrt, log, exp and division: the
// compensated corrections are skipped where they would form inf-inf/0*inf,
// and the IEEE value of the leading component is returned instead.
TEST(Precision, nonFiniteSemantics) {
    const double inf = std::numeric_limits<double>::infinity();
    const double nan = std::numeric_limits<double>::quiet_NaN();
    using quantape::math::sqrt;

    const DoubleDouble sqrtInf = sqrt(DoubleDouble(inf));
    EXPECT_TRUE(std::isinf(sqrtInf.value()));
    EXPECT_GT(sqrtInf.hi, 0.0);
    EXPECT_TRUE(std::isnan(sqrt(DoubleDouble(-inf)).value()));
    EXPECT_TRUE(std::isnan(sqrt(DoubleDouble(nan)).value()));
    EXPECT_TRUE(std::isnan(sqrt(DoubleDouble(-1.0)).value()));

    const DoubleDouble logInf = quantape::math::log(DoubleDouble(inf));
    EXPECT_TRUE(std::isinf(logInf.value()));
    EXPECT_GT(logInf.hi, 0.0);
    const DoubleDouble logZero = quantape::math::log(DoubleDouble(0.0));
    EXPECT_TRUE(std::isinf(logZero.value()));
    EXPECT_LT(logZero.hi, 0.0);
    const DoubleDouble logNegZero = quantape::math::log(DoubleDouble(-0.0));
    EXPECT_TRUE(std::isinf(logNegZero.value()));
    EXPECT_LT(logNegZero.hi, 0.0);
    EXPECT_TRUE(std::isnan(quantape::math::log(DoubleDouble(-1.0)).value()));
    EXPECT_TRUE(std::isnan(quantape::math::log(DoubleDouble(nan)).value()));

    const DoubleDouble expInf = quantape::math::exp(DoubleDouble(inf));
    EXPECT_TRUE(std::isinf(expInf.value()));
    EXPECT_GT(expInf.hi, 0.0);
    const DoubleDouble expNegInf = quantape::math::exp(DoubleDouble(-inf));
    EXPECT_EQ(expNegInf.value(), 0.0);
    EXPECT_FALSE(std::signbit(expNegInf.value()));
    EXPECT_TRUE(std::isnan(quantape::math::exp(DoubleDouble(nan)).value()));

    const DoubleDouble one(1.0);
    const DoubleDouble zero(0.0);
    const DoubleDouble divZero = one / zero;
    EXPECT_TRUE(std::isinf(divZero.value()));
    EXPECT_GT(divZero.hi, 0.0);
    const DoubleDouble divNegZero = one / DoubleDouble(-0.0);
    EXPECT_TRUE(std::isinf(divNegZero.value()));
    EXPECT_LT(divNegZero.hi, 0.0);
    EXPECT_TRUE(std::isnan((zero / zero).value()));
    EXPECT_TRUE(std::isnan((DoubleDouble(inf) / DoubleDouble(inf)).value()));
    const DoubleDouble divByInf = DoubleDouble(2.0) / DoubleDouble(inf);
    EXPECT_EQ(divByInf.value(), 0.0);
    const DoubleDouble infOverTwo = DoubleDouble(inf) / DoubleDouble(2.0);
    EXPECT_TRUE(std::isinf(infOverTwo.value()));
    EXPECT_GT(infOverTwo.hi, 0.0);
    EXPECT_TRUE(std::isnan((DoubleDouble(nan) / one).value()));
    EXPECT_TRUE(std::isnan((one / DoubleDouble(nan)).value()));
}

// log near 1: the direct 2 atanh(d/(2+d)) form on d = x-1 in double-double
// keeps the relative accuracy of the tiny result; the frexp+ln2 reduction is
// used only away from 1, where its two terms are well separated.
TEST(Precision, logNearOne) {
    for (const double d : {1e-15, 1e-20, 1e-30}) {
        SCOPED_TRACE(::testing::Message() << "d=" << d);
        const DoubleDouble a = DoubleDouble(1.0) + DoubleDouble(d);
        const DoubleDouble got = quantape::math::log(a);

        // Relative error against the direct-series reference at the
        // double-double floor.
        checkRel("log(1+d) vs direct series", got, log1pSeries(d), 1e-31);

        // Round trip: exp(log(1+d)) - 1 reproduces the increment d.
        checkRel("exp(log(1+d)) increment", quantape::math::exp(got) - DoubleDouble(1.0),
                 DoubleDouble(d), 1e-25);

        // log(exp(d)) = d up to the intrinsic d/2 of the second-order term.
        checkRel("log(exp(d))", quantape::math::log(quantape::math::exp(DoubleDouble(d))),
                 DoubleDouble(d), 1e-12);
    }
}

// sqrt at extreme exponents: the Newton refinement's compensated residual is
// ~1e-32 of the numerator, so the inputs are scaled by 2^(2m) to keep it
// normal; the scaled squared round-trip then measures the full precision.
TEST(Precision, sqrtScaling) {
    using quantape::math::sqrt;
    for (const double x : {1e-300, 1e-308, 1e-200, 1e-100, 1e-10, 1e10, 1e100, 1e200, 1e300}) {
        SCOPED_TRACE(::testing::Message() << "x=" << x);
        const DoubleDouble a(x);
        const DoubleDouble r = sqrt(a);
        int exponent = 0;
        std::frexp(a.hi, &exponent);
        const int scale = -2 * (exponent / 2);
        const DoubleDouble rScaled = scalePow2(r, scale / 2);
        const DoubleDouble aScaled = scalePow2(a, scale);
        const double rel =
            std::fabs((rScaled * rScaled - aScaled).value()) / std::fabs(aScaled.value());
        EXPECT_LT(rel, 1e-30);
    }
}
