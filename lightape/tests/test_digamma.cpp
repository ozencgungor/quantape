#include <cmath>
#include <limits>

#include "lightape/base/Digamma.h"
#include <gtest/gtest.h>

namespace {

using lightape::digamma;
using lightape::trigamma;

constexpr double kEulerGamma = 0.57721566490153286061;
constexpr double kLn2 = 0.69314718055994530942;
constexpr double kPiSq = 9.86960440108935861883;

TEST(DigammaTest, DigammaReferenceValues) {
    EXPECT_NEAR(digamma(1.0), -kEulerGamma, 1e-13);
    EXPECT_NEAR(digamma(2.0), 1.0 - kEulerGamma, 1e-13);
    EXPECT_NEAR(digamma(0.5), -kEulerGamma - 2.0 * kLn2, 1e-13);
    EXPECT_NEAR(digamma(10.0), 2.2517525890667211, 1e-13);
}

TEST(DigammaTest, TrigammaReferenceValues) {
    EXPECT_NEAR(trigamma(1.0), kPiSq / 6.0, 1e-13);
    EXPECT_NEAR(trigamma(0.5), kPiSq / 2.0, 1e-13);
    EXPECT_NEAR(trigamma(10.0), 0.10516633568168574612, 1e-13);
}

TEST(DigammaTest, RecurrenceBelowAsymptoticRegime) {
    const double xs[] = {0.25, 1.0, 3.7, 11.5};
    for (const double x : xs) {
        const double expected = digamma(x) + 1.0 / x;
        EXPECT_NEAR(digamma(x + 1.0), expected, 1e-13 * std::fabs(expected));
        const double expected2 = trigamma(x) - 1.0 / (x * x);
        EXPECT_NEAR(trigamma(x + 1.0), expected2, 1e-13 * std::fabs(expected2));
    }
}

TEST(DigammaTest, LargeArgumentAsymptoticContinuity) {
    EXPECT_NEAR(digamma(51.0) - digamma(50.0), 1.0 / 50.0, 1e-15);
    EXPECT_NEAR(trigamma(51.0) - trigamma(50.0), -1.0 / (50.0 * 50.0), 1e-15);
}

TEST(DigammaTest, PoleBehaviorMatchesReference) {
    EXPECT_TRUE(std::isnan(digamma(0.0)));
    EXPECT_TRUE(std::isnan(digamma(-0.0)));
    EXPECT_TRUE(std::isnan(digamma(-1.0)));
    EXPECT_TRUE(std::isnan(digamma(-2.0)));
    EXPECT_TRUE(std::isnan(digamma(-0.5)));
    EXPECT_TRUE(std::isnan(trigamma(0.0)));
    EXPECT_TRUE(std::isnan(trigamma(-0.0)));
    EXPECT_TRUE(std::isnan(trigamma(-1.0)));
    EXPECT_TRUE(std::isnan(trigamma(-2.0)));
    EXPECT_TRUE(std::isnan(trigamma(-0.5)));
    const double nan = std::numeric_limits<double>::quiet_NaN();
    EXPECT_TRUE(std::isnan(digamma(nan)));
    EXPECT_TRUE(std::isnan(trigamma(nan)));
}

TEST(DigammaTest, DigammaAsymptoticSlope) {
    const double x = 1.0e8;
    EXPECT_NEAR(digamma(x), std::log(x) - 0.5 / x, 1e-13);
}

} // namespace
