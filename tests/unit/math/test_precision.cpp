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
