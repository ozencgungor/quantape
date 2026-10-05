/**
 * @file test_hull_white_convexity.cpp
 * @brief Independent validation of the Hull-White futures convexity adjustment
 *
 * A flat-curve reference is built from the affine Hull-White bond algebra:
 * with constant volatility sigma, mean reversion a, flat short rate r0 and
 * accrual tau over [T1, T1 + tau],
 *
 *   b        = (1 - e^{-a tau}) / a,
 *   variance = sigma^2 (1 - e^{-2 a T1}) / (2 a),
 *   alpha    = r0 + sigma^2 (1 - e^{-a T1})^2 / (2 a^2),
 *   A        = e^{-r0 tau} exp(b f0 - sigma^2 (1 - e^{-2 a T1}) b^2 / (4 a)),  f0 = r0,
 *   expected = exp(b alpha + b^2 variance / 2) / A,
 *   adjustment = (expected - 1) / tau - (ratio - 1) / tau,   ratio = e^{r0 tau}.
 *
 * `expected` is the model expectation of the accrual compound factor and
 * `ratio` the bond ratio over the same period, so the reference is independent
 * of the production exponent algebra. The a -> 0 series branch and reverse-mode
 * derivatives are gated as well.
 */

#include "quantape/math/StanMath.h"

#include "quantape/log/Log.h"
#include "quantape/markets/Curves/HullWhiteConvexity.h"
#include "quantape/util/Check.h"

#include <algorithm>
#include <cmath>
#include <cstddef>

using namespace quantape;

namespace {

using stan::math::var;

struct FlatReference {
    long double expected = 1.0L;
    long double ratio = 1.0L;
};

/// Flat-curve textbook reference: bonds are affine in the state, the exponent
/// is Gaussian, and its moment generating function gives `expected`.
FlatReference flatCurveReference(long double sigma, long double a, long double T1, long double tau,
                                 long double r0, long double ratio) {
    const long double b = (1.0L - std::exp(-a * tau)) / a;
    const long double variance = sigma * sigma * (1.0L - std::exp(-2.0L * a * T1)) / (2.0L * a);
    const long double oneMinusExpiry = 1.0L - std::exp(-a * T1);
    const long double alpha = r0 + sigma * sigma / (2.0L * a * a) * oneMinusExpiry * oneMinusExpiry;
    const long double f0 = r0;
    const long double A =
        std::exp(-r0 * tau) *
        std::exp(b * f0 - sigma * sigma / (4.0L * a) * (1.0L - std::exp(-2.0L * a * T1)) * b * b);
    FlatReference reference;
    reference.expected = std::exp(b * alpha + b * b * variance / 2.0L) / A;
    reference.ratio = ratio;
    return reference;
}

bool withinRelative(long double got, long double expected, long double relativeTolerance,
                    long double absoluteFloor = 0.0L) {
    const long double scale = std::max(std::abs(got), std::abs(expected));
    return std::abs(got - expected) <= relativeTolerance * scale + absoluteFloor;
}

void checkReferenceCell(double sigma, double a, double T1, double tau, double r0) {
    const double ratio = std::exp(r0 * tau);
    const FlatReference reference = flatCurveReference(sigma, a, T1, tau, r0, ratio);
    const long double referenceAdjustment =
        (reference.expected - 1.0L) / tau - (reference.ratio - 1.0L) / tau;
    const double adjustment = markets::hullWhiteFuturesAdjustment(sigma, a, T1, tau, ratio);
    // The adjustment is a difference of two O(1) growth factors and the
    // helper receives the rounded exp(r0 tau), so the relative gate carries a
    // small absolute floor for that cancellation.
    CHECK(
        withinRelative(static_cast<long double>(adjustment), referenceAdjustment, 1e-12L, 5e-15L));
    // The accrual growth factor ratio + tau * C is cancellation-free and must
    // match the model expectation to machine precision.
    const long double growth =
        static_cast<long double>(reference.ratio) + tau * static_cast<long double>(adjustment);
    CHECK(withinRelative(growth, reference.expected, 1e-12L));
    QTA_LOG_INFO("test", "hw reference sigma={} a={} T1={} tau={} r0={} adjust={} expected={}",
                 sigma, a, T1, tau, r0, adjustment, static_cast<double>(reference.expected));
}

void testFlatCurveReference() {
    const double sigma[3] = {0.012, 0.008, 0.03};
    const double meanReversion[3] = {0.045, 0.02, 0.4};
    const double expiry[3] = {2.0, 10.0, 1.0};
    const double accrual[3] = {0.25, 0.25, 0.5};
    const double r0[3] = {0.0, 0.02, 0.05};
    for (std::size_t c = 0; c < 3; ++c) {
        for (std::size_t r = 0; r < 3; ++r) {
            checkReferenceCell(sigma[c], meanReversion[c], expiry[c], accrual[c], r0[r]);
        }
    }
}

void testZeroVolatility() {
    const double adjustment = markets::hullWhiteFuturesAdjustment(0.0, 0.045, 2.0, 0.25, 1.005);
    CHECK(adjustment == 0.0);
    const double other = markets::hullWhiteFuturesAdjustment(0.0, 0.0, 10.0, 0.5, 1.25);
    CHECK(other == 0.0);
    QTA_LOG_INFO("test", "hw zero volatility adjustments {}, {}", adjustment, other);
}

void testSeriesContinuity() {
    const double series = markets::hullWhiteFuturesAdjustment(0.01, 0.0, 1.0, 0.25, 1.0);
    const double threshold = markets::hullWhiteFuturesAdjustment(0.01, 1e-8, 1.0, 0.25, 1.0);
    const double pastThreshold =
        markets::hullWhiteFuturesAdjustment(0.01, 1.000001e-8, 1.0, 0.25, 1.0);
    CHECK(series > 0.0);
    CHECK(withinRelative(threshold, series, 1e-4L));
    CHECK(withinRelative(pastThreshold, threshold, 1e-4L));
    // The exact branch converges to the series from below as a -> 0.
    CHECK(threshold < series);
    QTA_LOG_INFO("test", "hw series={} threshold={} past={}", series, threshold, pastThreshold);
}

void testGradients() {
    stan::math::recover_memory();
    var sigma = 0.012;
    var meanReversion = 0.045;
    const double expiry = 2.0;
    const double accrual = 0.25;
    const double ratio = std::exp(0.02 * accrual);
    var adjustment =
        markets::hullWhiteFuturesAdjustment(sigma, meanReversion, expiry, accrual, ratio);
    stan::math::grad(adjustment.vi_);

    const auto primal = [&](double sigmaValue, double meanReversionValue) {
        return markets::hullWhiteFuturesAdjustment(sigmaValue, meanReversionValue, expiry, accrual,
                                                   ratio);
    };
    const double epsilon = 1e-6;
    const double sigmaAdjoint = sigma.adj();
    const double meanReversionAdjoint = meanReversion.adj();
    util::checkClose("hw convexity d/d sigma", sigmaAdjoint,
                     (primal(sigma.val() + epsilon, meanReversion.val()) -
                      primal(sigma.val() - epsilon, meanReversion.val())) /
                         (2.0 * epsilon),
                     1e-6);
    util::checkClose("hw convexity d/d mean reversion", meanReversionAdjoint,
                     (primal(sigma.val(), meanReversion.val() + epsilon) -
                      primal(sigma.val(), meanReversion.val() - epsilon)) /
                         (2.0 * epsilon),
                     1e-6);
    QTA_LOG_INFO("test", "hw gradients sigma={} meanReversion={}", sigmaAdjoint,
                 meanReversionAdjoint);
}

} // namespace

int main() {
    testFlatCurveReference();
    testZeroVolatility();
    testSeriesContinuity();
    testGradients();
    QTA_LOG_INFO("test", "test_hull_white_convexity: ok");
    return 0;
}
