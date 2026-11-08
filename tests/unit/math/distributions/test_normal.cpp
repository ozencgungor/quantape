// test_normal.cpp — normal distribution values, identities, tails, quantile,
// and the analytic derivative packs vs finite-difference oracles.
#include "quantape/math/Distributions/Normal.h"

#include <cmath>
#include <limits>

#include "support/GtestSupport.h"

namespace {

constexpr double kTol = 1e-14;

quantape::math::Normal<double> standard() {
    return quantape::math::Normal<double>{0.0, 1.0};
}

} // namespace

TEST(NormalDistribution, knownValues) {
    const auto n = standard();
    CHECK_CLOSE("phi(0)", n.pdf(0.0), 0.3989422804014327, 1e-15);
    CHECK_CLOSE("phi(1)", n.pdf(1.0), 0.24197072451914337, 1e-15);
    CHECK_CLOSE("Phi(0)", n.cdf(0.0), 0.5, 1e-16);
    CHECK_CLOSE("Phi(1)", n.cdf(1.0), 0.8413447460685429, 1e-15);
    CHECK_CLOSE("Phi(2)", n.cdf(2.0), 0.9772498680518208, 1e-15);
    CHECK_CLOSE("Phi(-1)", n.cdf(-1.0), 0.15865525393145707, 1e-15);
    CHECK_CLOSE("1 - Phi(2)", n.ccdf(2.0), 0.02275013194817921, 1e-15);
}

TEST(NormalDistribution, symmetryAndComplement) {
    const auto n = standard();
    for (double x : {-4.5, -2.0, -0.3, 0.0, 0.7, 1.5, 3.9, 7.0}) {
        CHECK_CLOSE("cdf + ccdf", n.cdf(x) + n.ccdf(x), 1.0, 1e-15);
        CHECK_CLOSE("cdf symmetry", n.cdf(-x), n.ccdf(x), 1e-15);
        CHECK_CLOSE("pdf symmetry", n.pdf(-x), n.pdf(x), 1e-15);
    }
}

TEST(NormalDistribution, generalParameters) {
    const quantape::math::Normal<double> n{1.25, 0.4};
    // Phi((x - mu)/sigma) against the standard kernel.
    for (double x : {-1.0, 0.0, 1.25, 2.0, 3.5}) {
        const double z = (x - 1.25) / 0.4;
        CHECK_CLOSE("general cdf", n.cdf(x), standard().cdf(z), 1e-15);
        CHECK_CLOSE("general pdf", n.pdf(x), standard().pdf(z) / 0.4, 1e-15);
    }
}

TEST(NormalDistribution, logFormsSurviveDeepTails) {
    const auto n = standard();
    const double lcdf = n.logCdf(-40.0);
    const double lccdf = n.logCcdf(40.0);
    EXPECT_TRUE(std::isfinite(lcdf));
    EXPECT_TRUE(std::isfinite(lccdf));
    CHECK_CLOSE("deep logcdf equals log(ccdf)", lcdf, n.logCcdf(40.0), 1e-12);
    // Mills-ratio consistency: Phi(-40)/phi(-40) * 40 -> 1.
    const double lphi = n.logPdf(-40.0);
    CHECK_CLOSE("Mills ratio", std::exp(lcdf - lphi) * 40.0, 1.0, 2e-3);
}

TEST(NormalDistribution, quantile) {
    const auto n = standard();
    CHECK_CLOSE("inv(0.5)", n.inv(0.5), 0.0, 1e-16);
    CHECK_CLOSE("inv(0.9)", n.inv(0.9), 1.2815515655446004, 1e-14);
    CHECK_CLOSE("inv(0.95)", n.inv(0.95), 1.6448536269514722, 1e-14);
    CHECK_CLOSE("inv(0.975)", n.inv(0.975), 1.9599639845400545, 1e-14);
    // Deep-tail value against the reference (the symmetry identity is
    // condition-limited at p = 1e-10: dq[q] ~ 1e-6 in the upper-tail branch).
    CHECK_CLOSE_REL("inv(1e-10)", n.inv(1e-10), -6.361340902404056, 1e-9);
    CHECK_CLOSE_REL("inv(1 - 1e-10)", n.inv(1.0 - 1e-10), 6.361340902404056, 1e-6);
    // Round-trip tolerance scales with 1/phi(x): near |x|=5 the cdf is ~1-3e-7
    // and double precision bounds the achievable x-resolution to ~1e-10.
    for (double x : {-5.0, -2.5, -0.4, 0.0, 0.3, 1.7, 4.2, 5.0}) {
        CHECK_CLOSE_REL("inv(cdf) round trip", n.inv(n.cdf(x)), x, 1e-10, 1e-12);
    }
    EXPECT_TRUE(std::isinf(n.inv(0.0)) && n.inv(0.0) < 0.0);
    EXPECT_TRUE(std::isinf(n.inv(1.0)) && n.inv(1.0) > 0.0);
    EXPECT_TRUE(std::isnan(n.inv(-0.5)));
    EXPECT_TRUE(std::isnan(n.inv(1.5)));
}

TEST(NormalDistribution, edgeValues) {
    const auto n = standard();
    EXPECT_TRUE(std::isnan(n.cdf(std::numeric_limits<double>::quiet_NaN())));
    CHECK_CLOSE("cdf(-inf)", n.cdf(-std::numeric_limits<double>::infinity()), 0.0, 0.0);
    CHECK_CLOSE("cdf(+inf)", n.cdf(std::numeric_limits<double>::infinity()), 1.0, 0.0);
    CHECK_CLOSE("pdf(-inf)", n.pdf(-std::numeric_limits<double>::infinity()), 0.0, 0.0);
    // Signed zero: pdf is even and cdf(0) is exactly 0.5 both ways.
    CHECK_CLOSE("pdf(-0)", n.pdf(-0.0), n.pdf(0.0), 0.0);
}

TEST(NormalDistribution, derivativePacksMatchFiniteDifferences) {
    const quantape::math::Normal<double> n{0.3, 1.4};
    for (double x : {-2.2, -0.5, 0.0, 0.8, 2.9, 5.5}) {
        const double h = 1e-5 * std::max(1.0, std::fabs(x));
        const auto pdfPack = n.pdfD1D2(x);
        const auto cdfPack = n.cdfD1D2(x);
        CHECK_CLOSE("pdf pack value", pdfPack.value, n.pdf(x), 0.0);
        CHECK_CLOSE("cdf pack d1", cdfPack.d1, n.pdf(x), 0.0);
        CHECK_CLOSE("pdf d1 vs FD", pdfPack.d1, (n.pdf(x + h) - n.pdf(x - h)) / (2 * h), 1e-8);
        // Second differences need a larger step to beat cancellation (eps/h^2).
        const double h2 = 1e-3 * std::max(1.0, std::fabs(x));
        CHECK_CLOSE("pdf d2 vs FD", pdfPack.d2,
                    (n.pdf(x + h2) - 2.0 * n.pdf(x) + n.pdf(x - h2)) / (h2 * h2), 1e-6);
        CHECK_CLOSE("cdf d2 vs FD", cdfPack.d2,
                    (n.cdf(x + h2) - 2.0 * n.cdf(x) + n.cdf(x - h2)) / (h2 * h2), 1e-6);

        const auto gh = n.cdfGradHess(x);
        CHECK_CLOSE("gradHess value", gh.value, n.cdf(x), 0.0);
        CHECK_CLOSE("gradHess gx", gh.gx, n.pdf(x), 0.0);
        const double hm = 1e-5 * std::max(1.0, std::fabs(0.3));
        const double hs = 1e-5 * 1.4;
        CHECK_CLOSE("gmu vs FD", gh.gmu,
                    (quantape::math::Normal<double>{0.3 + hm, 1.4}.cdf(x) -
                     quantape::math::Normal<double>{0.3 - hm, 1.4}.cdf(x)) /
                        (2.0 * hm),
                    1e-9);
        CHECK_CLOSE("gsig vs FD", gh.gsig,
                    (quantape::math::Normal<double>{0.3, 1.4 + hs}.cdf(x) -
                     quantape::math::Normal<double>{0.3, 1.4 - hs}.cdf(x)) /
                        (2.0 * hs),
                    1e-9);
        CHECK_CLOSE("hxx vs FD", gh.hxx,
                    (n.cdf(x + h2) - 2.0 * n.cdf(x) + n.cdf(x - h2)) / (h2 * h2), 1e-6);
    }
}

// ---------------------------------------------------------------------------
// The AS241 lower-tail rational is fitted in r = sqrt(-log p) up to ~27.5, so
// the quantile is finite for every representable 0 < p < 1, including the
// subnormal range (the smallest subnormal has r = 27.28). References: Boost
// `quantile(normal_distribution<cpp_bin_float_100>{0,1}, p)`.
// ---------------------------------------------------------------------------

TEST(NormalDistribution, quantileResolvesSubnormalLowerTail) {
    using quantape::math::detail::normalInvDouble;
    CHECK_CLOSE_REL("inv(5e-324)", normalInvDouble(5e-324), -38.4674056171443462508, 1e-14);
    CHECK_CLOSE_REL("inv(7e-311)", normalInvDouble(7e-311), -37.6725226359137007645, 1e-14);
    CHECK_CLOSE_REL("inv(1e-310)", normalInvDouble(1e-310), -37.6630603319495237319, 1e-14);
    CHECK_CLOSE_REL("inv(1e-300)", normalInvDouble(1e-300), -37.0470962993611992, 1e-14);
    // Only the degenerate endpoints are infinite; NaN and out-of-range inputs
    // propagate NaN.
    EXPECT_TRUE(std::isinf(normalInvDouble(0.0)) && normalInvDouble(0.0) < 0.0);
    EXPECT_TRUE(std::isinf(normalInvDouble(1.0)) && normalInvDouble(1.0) > 0.0);
    EXPECT_TRUE(std::isnan(normalInvDouble(-1e-300)));
    EXPECT_TRUE(std::isnan(normalInvDouble(std::nextafter(1.0, 2.0))));
}
