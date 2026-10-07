// test_beta.cpp — Beta pdf/cdf/log tails, incomplete-beta kernel and quantile gates.

#include "quantape/math/Distributions/Beta.h"
#include "quantape/math/Distributions/detail/IncompleteBeta.h"

#include <array>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <numbers>
#include <sstream>
#include <string>
#include <utility>

#include "support/GtestSupport.h"

using quantape::math::Beta;
using quantape::math::detail::betaCdfD1D2;
using quantape::math::detail::betaCdfGradHess;
using quantape::math::detail::BetaCdfGradHess;
using quantape::math::detail::betaPdfD1D2;
using quantape::math::detail::betaPdfGradHess;
using quantape::math::detail::BetaPdfGradHess;
using quantape::math::detail::BetaValueD1D2;
using quantape::math::detail::incBeta;
using quantape::math::detail::incBetaLog;

namespace {

constexpr double kTol = 1e-12;

/// Composite Simpson rule with Kahan compensation (n even).
template <class F>
double simpsonIntegrate(const F& f, double a, double b, int n) {
    const double h = (b - a) / n;
    double sum = f(a) + f(b);
    double compensation = 0.0;
    for (int i = 1; i < n; ++i) {
        const double term = (i % 2 == 1 ? 4.0 : 2.0) * f(a + i * h);
        const double y = term - compensation;
        const double t = sum + y;
        compensation = (t - sum) - y;
        sum = t;
    }
    return sum * h / 3.0;
}

/// P(Bin(n,x) >= k) summed directly from the k-th pmf (no 1 - small).
double binomialTail(int n, int k, double x) {
    if (k <= 0) {
        return 1.0;
    }
    const double logPk = std::lgamma(n + 1.0) - std::lgamma(k + 1.0) - std::lgamma(n - k + 1.0) +
                         k * std::log(x) + (n - k) * std::log1p(-x);
    double term = std::exp(logPk);
    double tail = term;
    for (int j = k; j < n; ++j) {
        term *= double(n - j) / double(j + 1) * x / (1.0 - x);
        tail += term;
    }
    return tail;
}

/// Five-point central first derivative, O(h^4).
template <class F>
double centralD1(const F& f, double x, double h) {
    return (-f(x + 2 * h) + 8 * f(x + h) - 8 * f(x - h) + f(x - 2 * h)) / (12 * h);
}

} // namespace

TEST(BetaDistribution, uniformSpecialCase) {
    const Beta<double> uniform{1.0, 1.0};
    for (double x : {0.0, 0.05, 0.25, 0.5, 0.75, 0.95, 1.0}) {
        CHECK_CLOSE("Beta(1,1) pdf = 1", uniform.pdf(x), 1.0, kTol);
        CHECK_CLOSE("Beta(1,1) cdf = x", uniform.cdf(x), x, kTol);
        CHECK_CLOSE("Beta(1,1) ccdf = 1-x", uniform.ccdf(x), 1.0 - x, kTol);
        CHECK_CLOSE("Beta(1,1) logPdf = 0", uniform.logPdf(x), 0.0, kTol);
        CHECK_CLOSE("Beta(1,1) logCdf = log x", uniform.logCdf(x), std::log(x), kTol);
        CHECK_CLOSE("Beta(1,1) logCcdf = log1p(-x)", uniform.logCcdf(x), std::log1p(-x), kTol);
        CHECK_CLOSE("Beta(1,1) inv = p", uniform.inv(x), x, kTol);
    }
}

TEST(BetaDistribution, integerParametersMatchBinomialTail) {
    // I_x(a,b) = P(Bin(a+b-1, x) >= a) for positive integers a, b.
    for (int a : {1, 2, 3, 5}) {
        for (int b : {1, 2, 4, 7}) {
            const Beta<double> dist{static_cast<double>(a), static_cast<double>(b)};
            for (double x : {0.05, 0.2, 0.5, 0.8, 0.95}) {
                CHECK_CLOSE("binomial tail identity", dist.cdf(x), binomialTail(a + b - 1, a, x),
                            1e-12);
            }
        }
    }
}

TEST(BetaDistribution, halfIntegerClosedForm) {
    // I_x(1/2,1/2) = (2/pi) asin(sqrt(x)); ccdf(x) = (2/pi) asin(sqrt(1-x)).
    const Beta<double> half{0.5, 0.5};
    for (double x : {1e-12, 1e-6, 0.01, 0.2, 0.5, 0.8, 0.99}) {
        const double expected = (2.0 / std::numbers::pi) * std::asin(std::sqrt(x));
        CHECK_CLOSE_REL("asin closed form", half.cdf(x), expected, 1e-12, 1e-15);
    }
    for (double x : {0.01, 0.2, 0.5, 0.8, 0.99}) {
        // asin(sqrt(1-x)) is ill-conditioned only as x -> 0, where ccdf -> 1.
        const double expected = (2.0 / std::numbers::pi) * std::asin(std::sqrt(1.0 - x));
        CHECK_CLOSE_REL("asin mirrored ccdf", half.ccdf(x), expected, 1e-12, 1e-15);
    }
}

TEST(BetaDistribution, cdfMatchesSimpsonOfPdf) {
    const std::array<std::pair<double, double>, 4> shapes = {
        {{2.0, 3.0}, {3.5, 7.0}, {1.0, 4.0}, {8.0, 3.5}}};
    for (const auto& [a, b] : shapes) {
        const Beta<double> dist{a, b};
        for (double x : {0.15, 0.4, 0.65, 0.9}) {
            const double integral =
                simpsonIntegrate([&](double t) { return dist.pdf(t); }, 0.0, x, 20000);
            CHECK_CLOSE_REL("cdf vs Simpson(0,x) of pdf", dist.cdf(x), integral, 1e-10, 1e-14);
        }
    }
}

TEST(BetaDistribution, normalizationAndSymmetry) {
    for (double a : {0.3, 2.0, 40.0}) {
        for (double b : {0.3, 3.0, 40.0}) {
            for (double x : {0.01, 0.2, 0.5, 0.8, 0.99}) {
                CHECK_CLOSE("symmetry I_x(a,b)+I_(1-x)(b,a) = 1",
                            incBeta(a, b, x) + incBeta(b, a, 1.0 - x), 1.0, 1e-13);
                const Beta<double> dist{a, b};
                CHECK_CLOSE("ccdf = 1 - cdf", dist.ccdf(x), 1.0 - dist.cdf(x), 1e-13);
            }
        }
    }
    for (const auto& [a, b] : {std::pair<double, double>{2.0, 3.0}, {3.5, 7.0}, {40.0, 40.0}}) {
        const Beta<double> dist{a, b};
        const double integral =
            simpsonIntegrate([&](double t) { return dist.pdf(t); }, 0.0, 1.0, 100000);
        CHECK_CLOSE_REL("normalization integral = 1", integral, 1.0, 1e-11);
        CHECK_CLOSE_REL("incBeta(a,b,1) = 1", incBeta(a, b, 1.0), 1.0, 1e-15);
    }
}

TEST(BetaDistribution, cdfIsMonotone) {
    const Beta<double> dist{2.5, 6.0};
    double previous = 0.0;
    for (int i = 0; i <= 200; ++i) {
        const double x = double(i) / 200.0;
        const double value = dist.cdf(x);
        SCOPED_TRACE(x);
        EXPECT_LE(previous, value);
        previous = value;
    }
    CHECK_CLOSE("cdf reaches 1", dist.cdf(1.0), 1.0, 0.0);
}

TEST(BetaDistribution, logTailsAreStable) {
    for (double a : {0.5, 2.0, 30.0}) {
        for (double b : {0.5, 3.0, 30.0}) {
            const Beta<double> dist{a, b};
            for (double x : {1e-30, 1e-8, 0.01, 0.3, 0.5, 0.7, 0.99, 1.0 - 1e-8, 1.0 - 1e-30}) {
                if (dist.cdf(x) > 0.0) {
                    CHECK_CLOSE_REL("exp(logCdf) = cdf", std::exp(dist.logCdf(x)), dist.cdf(x),
                                    1e-12, 1e-300);
                }
                if (dist.ccdf(x) > 0.0) {
                    CHECK_CLOSE_REL("exp(logCcdf) = ccdf", std::exp(dist.logCcdf(x)), dist.ccdf(x),
                                    1e-12, 1e-300);
                }
                CHECK_CLOSE("log1p-style identity: logCdf+logCcdf mutual",
                            std::exp(dist.logCdf(x)) + std::exp(dist.logCcdf(x)), 1.0, 1e-12);
            }
        }
    }
    // Deep tail: I_x(2,3) = 6x^2 - 8x^3 + 3x^4 ~ 6x^2 for x -> 0.
    const Beta<double> dist{2.0, 3.0};
    const double x = 1e-150;
    const double closed = 6.0 * x * x;
    CHECK_CLOSE_REL("deep-tail logCdf ~ log(6x^2)", dist.logCdf(x), std::log(closed), 1e-12);
    CHECK_CLOSE_REL("deep-tail exp(logCdf) ~ 6x^2", std::exp(dist.logCdf(x)), closed, 1e-12);
    CHECK_CLOSE("deep-tail logCcdf ~ -6x^2", dist.logCcdf(x), -closed, 1e-310);
}

TEST(BetaDistribution, quantileRoundTrip) {
    const std::array<std::pair<double, double>, 6> shapes = {
        {{2.0, 3.0}, {0.5, 0.5}, {10.0, 1.0}, {1.0, 10.0}, {1000.0, 1000.0}, {0.5, 2.0}}};
    for (const auto& [a, b] : shapes) {
        const Beta<double> dist{a, b};
        for (double p : {1e-12, 1e-8, 1e-4, 0.01, 0.3, 0.5, 0.7, 0.99, 1.0 - 1e-6}) {
            const double quantile = dist.inv(p);
            CHECK_CLOSE_REL("cdf(inv(p)) = p", dist.cdf(quantile), p, 1e-9, 1e-15);
            SCOPED_TRACE(p);
            EXPECT_GE(quantile, 0.0);
            EXPECT_LE(quantile, 1.0);
        }
    }
    const Beta<double> dist{2.0, 3.0};
    CHECK_CLOSE("inv(0) = 0", dist.inv(0.0), 0.0, 0.0);
    CHECK_CLOSE("inv(1) = 1", dist.inv(1.0), 1.0, 0.0);
    CHECK_CLOSE("inv(1/2) monotone centre", dist.cdf(dist.inv(0.5)), 0.5, 1e-12);
}

TEST(BetaDistribution, edgeAndInvalidInputs) {
    const Beta<double> dist{2.0, 3.0};
    CHECK_CLOSE("cdf(0) = 0", dist.cdf(0.0), 0.0, 0.0);
    CHECK_CLOSE("cdf(1) = 1", dist.cdf(1.0), 1.0, 0.0);
    CHECK_CLOSE("ccdf(0) = 1", dist.ccdf(0.0), 1.0, 0.0);
    CHECK_CLOSE("ccdf(1) = 0", dist.ccdf(1.0), 0.0, 0.0);
    CHECK_CLOSE("pdf outside support = 0", dist.pdf(-0.5), 0.0, 0.0);
    CHECK_CLOSE("pdf above support = 0", dist.pdf(1.5), 0.0, 0.0);
    CHECK_CLOSE("cdf below support = 0", dist.cdf(-0.5), 0.0, 0.0);
    CHECK_CLOSE("cdf above support = 1", dist.cdf(1.5), 1.0, 0.0);
    CHECK_CLOSE("logPdf outside = -inf", dist.logPdf(-0.5),
                -std::numeric_limits<double>::infinity(), 0.0);

    const Beta<double> singularLow{0.5, 2.0};
    const Beta<double> singularHigh{2.0, 0.5};
    EXPECT_TRUE(std::isinf(singularLow.pdf(0.0)));
    EXPECT_TRUE(std::isinf(singularHigh.pdf(1.0)));
    CHECK_CLOSE("a>1 pdf(0) = 0", dist.pdf(0.0), 0.0, 0.0);
    const Beta<double> noSingularHigh{3.0, 2.0};
    CHECK_CLOSE("b>1 pdf(1) = 0", noSingularHigh.pdf(1.0), 0.0, 0.0);

    const Beta<double> invalid{0.0, 2.0};
    EXPECT_TRUE(std::isnan(invalid.pdf(0.5)));
    EXPECT_TRUE(std::isnan(invalid.cdf(0.5)));
    EXPECT_TRUE(std::isnan(invalid.logCdf(0.5)));
    EXPECT_TRUE(std::isnan(invalid.ccdf(0.5)));
    EXPECT_TRUE(std::isnan(invalid.logCcdf(0.5)));
    EXPECT_TRUE(std::isnan(invalid.inv(0.5)));
    EXPECT_TRUE(std::isnan(incBeta(0.0, 2.0, 0.5)));
    EXPECT_TRUE(std::isnan(dist.cdf(std::numeric_limits<double>::quiet_NaN())));
    EXPECT_TRUE(std::isnan(dist.inv(1.5)));
    EXPECT_TRUE(std::isnan(dist.inv(-0.1)));
}

TEST(BetaDistribution, largeParameters) {
    const Beta<double> symmetric{1000.0, 1000.0};
    CHECK_CLOSE_REL("symmetric median cdf(1/2) = 1/2", symmetric.cdf(0.5), 0.5, 1e-9, 0.0);
    for (double p : {0.01, 0.1, 0.5, 0.9, 0.99}) {
        CHECK_CLOSE_REL("large-shape inv round trip", symmetric.cdf(symmetric.inv(p)), p, 1e-9,
                        0.0);
    }
    // Closed form for b = 1: I_x(a,1) = x^a.
    const Beta<double> power{1000.0, 1.0};
    for (double x : {0.9, 0.99, 0.999}) {
        CHECK_CLOSE_REL("I_x(1000,1) = x^1000", power.cdf(x), std::exp(1000.0 * std::log(x)), 1e-11,
                        0.0);
    }
}

TEST(BetaDistribution, derivativePacksMatchCentralDifferences) {
    for (double a : {0.7, 2.0, 11.0}) {
        for (double b : {1.3, 3.0, 7.0}) {
            const Beta<double> dist{a, b};
            const auto pdf = [&](double t) { return dist.pdf(t); };
            for (double x : {0.13, 0.37, 0.5, 0.72, 0.91}) {
                const BetaValueD1D2 cdfPack = betaCdfD1D2(a, b, x);
                CHECK_CLOSE_REL("cdf pack value", cdfPack.value, dist.cdf(x), 1e-14);
                CHECK_CLOSE_REL("cdf pack d1 = pdf", cdfPack.d1, dist.pdf(x), 1e-12);
                CHECK_CLOSE_REL("cdf pack d2 vs FD", cdfPack.d2, centralD1(pdf, x, 1e-4), 1e-7,
                                1e-12);
                const BetaValueD1D2 pdfPack = betaPdfD1D2(a, b, x);
                CHECK_CLOSE_REL("pdf pack value", pdfPack.value, dist.pdf(x), 1e-14);
                CHECK_CLOSE_REL("pdf pack d1 = pdf", pdfPack.d1, pdfPack.value, 1e-15);
                CHECK_CLOSE_REL("pdf pack d2 vs FD", pdfPack.d2, centralD1(pdf, x, 1e-4), 1e-7,
                                1e-12);
            }
        }
    }
}

TEST(BetaDistribution, logKernelMatchesValueKernel) {
    for (double a : {0.5, 2.0, 25.0}) {
        for (double b : {0.5, 4.0, 25.0}) {
            for (double x : {1e-20, 1e-4, 0.1, 0.5, 0.9, 1.0 - 1e-4, 1.0 - 1e-20}) {
                CHECK_CLOSE_REL("exp(incBetaLog) = incBeta", std::exp(incBetaLog(a, b, x)),
                                incBeta(a, b, x), 1e-12, 1e-300);
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Beta parameter-derivative packs.
//
// Reference values: tests/data/distributions/beta_t_paramderiv_reference.csv
// (the `beta` section; test_student_t.cpp reads the `student-t` section),
// generated with .venv/bin/python (mpmath 1.4.1, mp.dps = 60) from
//     F = mp.betainc(a, b, 0, x, regularized=True)
//     f = mp.exp((a-1)*mp.log(x) + (b-1)*mp.log1p(-x) - mp.log(mp.beta(a, b)))
// with mp.diff for every column and 20 significant digits in the file. The
// grid covers alpha, beta in {0.5, 1, 2.7, 10} with x in (0,1) (both
// continued-fraction branches); value/first-order entries are gated at 1e-10
// relative, second-order at 1e-9.
// ---------------------------------------------------------------------------

TEST(BetaParameterDerivatives, referenceCsvMatchesMpmath) {
    const std::filesystem::path relative = "distributions/beta_t_paramderiv_reference.csv";
    SKIP_UNLESS_ASSET(std::filesystem::is_regular_file(testDataDir() / relative),
                      "beta parameter-derivative reference csv absent");
    std::ifstream in(RequireDataFile(relative));
    SKIP_UNLESS_ASSET(in.good(), "beta parameter-derivative reference csv unreadable");

    std::string line;
    int rows = 0;
    while (std::getline(in, line)) {
        if (line.empty()) {
            continue;
        }
        if (line[0] == '#') {
            if (line.find("student-t") != std::string::npos) {
                break;
            }
            continue;
        }
        std::stringstream fields(line);
        double f[23] = {};
        int n = 0;
        std::string token;
        while (n < 23 && std::getline(fields, token, ',')) {
            f[n++] = std::stod(token);
        }
        ASSERT_EQ(n, 23) << "malformed reference row: " << line;
        const double a = f[0];
        const double b = f[1];
        const double x = f[2];
        const BetaCdfGradHess c = betaCdfGradHess(a, b, x);
        const BetaPdfGradHess p = betaPdfGradHess(a, b, x);
        const std::string label =
            "a=" + std::to_string(a) + " b=" + std::to_string(b) + " x=" + std::to_string(x);
        CHECK_CLOSE_REL(label + " cdf value", c.value, f[3], 1e-10);
        CHECK_CLOSE_REL(label + " cdf gAlpha", c.gAlpha, f[4], 1e-10);
        CHECK_CLOSE_REL(label + " cdf gBeta", c.gBeta, f[5], 1e-10);
        CHECK_CLOSE_REL(label + " cdf gx", c.gx, f[6], 1e-10);
        CHECK_CLOSE_REL(label + " cdf hAA", c.hAA, f[7], 1e-9);
        CHECK_CLOSE_REL(label + " cdf hAB", c.hAB, f[8], 1e-9);
        CHECK_CLOSE_REL(label + " cdf hAx", c.hAx, f[9], 1e-9);
        CHECK_CLOSE_REL(label + " cdf hBB", c.hBB, f[10], 1e-9);
        CHECK_CLOSE_REL(label + " cdf hBx", c.hBx, f[11], 1e-9);
        CHECK_CLOSE_REL(label + " cdf hxx", c.hxx, f[12], 1e-9);
        CHECK_CLOSE_REL(label + " pdf value", p.value, f[13], 1e-10);
        CHECK_CLOSE_REL(label + " pdf gAlpha", p.gAlpha, f[14], 1e-10);
        CHECK_CLOSE_REL(label + " pdf gBeta", p.gBeta, f[15], 1e-10);
        CHECK_CLOSE_REL(label + " pdf gx", p.gx, f[16], 1e-10);
        CHECK_CLOSE_REL(label + " pdf hAA", p.hAA, f[17], 1e-9);
        CHECK_CLOSE_REL(label + " pdf hAB", p.hAB, f[18], 1e-9);
        CHECK_CLOSE_REL(label + " pdf hAx", p.hAx, f[19], 1e-9);
        CHECK_CLOSE_REL(label + " pdf hBB", p.hBB, f[20], 1e-9);
        CHECK_CLOSE_REL(label + " pdf hBx", p.hBx, f[21], 1e-9);
        CHECK_CLOSE_REL(label + " pdf hxx", p.hxx, f[22], 1e-9);
        ++rows;
    }
    EXPECT_EQ(rows, 40);
}

TEST(BetaParameterDerivatives, xEntriesReproduceExistingPacksBitwise) {
    for (double a : {0.5, 1.0, 2.7, 10.0}) {
        for (double b : {0.5, 1.0, 2.7, 10.0}) {
            for (double x : {1e-6, 0.01, 0.3, 0.5, 0.7, 0.99, 1.0 - 1e-6}) {
                const std::string label = "a=" + std::to_string(a) + " b=" + std::to_string(b) +
                                          " x=" + std::to_string(x);
                const BetaValueD1D2 oldCdf = betaCdfD1D2(a, b, x);
                const BetaCdfGradHess cdf = betaCdfGradHess(a, b, x);
                CHECK_CLOSE(label + " cdf value", cdf.value, oldCdf.value, 0.0);
                CHECK_CLOSE(label + " cdf gx", cdf.gx, oldCdf.d1, 0.0);
                CHECK_CLOSE(label + " cdf hxx", cdf.hxx, oldCdf.d2, 0.0);
                const BetaValueD1D2 oldPdf = betaPdfD1D2(a, b, x);
                const BetaPdfGradHess pdf = betaPdfGradHess(a, b, x);
                CHECK_CLOSE(label + " pdf value", pdf.value, oldPdf.value, 0.0);
                CHECK_CLOSE(label + " pdf gx", pdf.gx, oldPdf.d2, 0.0);
            }
        }
    }
}

TEST(BetaParameterDerivatives, parameterEntriesMatchFiniteDifferences) {
    for (double a : {0.7, 2.7}) {
        for (double b : {1.3, 6.0}) {
            for (double x : {0.2, 0.55, 0.85}) {
                const double ha = 1e-5 * a;
                const double hb = 1e-5 * b;
                const double hx = 1e-5 * x;
                const std::string label = "a=" + std::to_string(a) + " b=" + std::to_string(b) +
                                          " x=" + std::to_string(x);
                const BetaCdfGradHess c = betaCdfGradHess(a, b, x);
                const BetaPdfGradHess p = betaPdfGradHess(a, b, x);

                const auto cdfValue = [](double aa, double bb, double xx) {
                    return betaCdfGradHess(aa, bb, xx).value;
                };
                const auto cdfGA = [](double aa, double bb, double xx) {
                    return betaCdfGradHess(aa, bb, xx).gAlpha;
                };
                const auto cdfGB = [](double aa, double bb, double xx) {
                    return betaCdfGradHess(aa, bb, xx).gBeta;
                };
                const auto cdfGx = [](double aa, double bb, double xx) {
                    return betaCdfGradHess(aa, bb, xx).gx;
                };
                const auto pdfValue = [](double aa, double bb, double xx) {
                    return betaPdfGradHess(aa, bb, xx).value;
                };
                const auto pdfGA = [](double aa, double bb, double xx) {
                    return betaPdfGradHess(aa, bb, xx).gAlpha;
                };
                const auto pdfGB = [](double aa, double bb, double xx) {
                    return betaPdfGradHess(aa, bb, xx).gBeta;
                };
                const auto pdfGx = [](double aa, double bb, double xx) {
                    return betaPdfGradHess(aa, bb, xx).gx;
                };

                CHECK_CLOSE_REL(label + " cdf gAlpha vs FD", c.gAlpha,
                                (cdfValue(a + ha, b, x) - cdfValue(a - ha, b, x)) / (2.0 * ha),
                                1e-5, 1e-7);
                CHECK_CLOSE_REL(label + " cdf gBeta vs FD", c.gBeta,
                                (cdfValue(a, b + hb, x) - cdfValue(a, b - hb, x)) / (2.0 * hb),
                                1e-5, 1e-7);
                CHECK_CLOSE_REL(label + " cdf hAA vs FD", c.hAA,
                                (cdfGA(a + ha, b, x) - cdfGA(a - ha, b, x)) / (2.0 * ha), 1e-5,
                                1e-7);
                CHECK_CLOSE_REL(label + " cdf hAB vs FD", c.hAB,
                                (cdfGA(a, b + hb, x) - cdfGA(a, b - hb, x)) / (2.0 * hb), 1e-5,
                                1e-7);
                CHECK_CLOSE_REL(label + " cdf hAx vs FD", c.hAx,
                                (cdfGA(a, b, x + hx) - cdfGA(a, b, x - hx)) / (2.0 * hx), 1e-5,
                                1e-7);
                CHECK_CLOSE_REL(label + " cdf hBB vs FD", c.hBB,
                                (cdfGB(a, b + hb, x) - cdfGB(a, b - hb, x)) / (2.0 * hb), 1e-5,
                                1e-7);
                CHECK_CLOSE_REL(label + " cdf hBx vs FD", c.hBx,
                                (cdfGB(a, b, x + hx) - cdfGB(a, b, x - hx)) / (2.0 * hx), 1e-5,
                                1e-7);
                CHECK_CLOSE_REL(label + " cdf hxx vs FD", c.hxx,
                                (cdfGx(a, b, x + hx) - cdfGx(a, b, x - hx)) / (2.0 * hx), 1e-5,
                                1e-7);

                CHECK_CLOSE_REL(label + " pdf gAlpha vs FD", p.gAlpha,
                                (pdfValue(a + ha, b, x) - pdfValue(a - ha, b, x)) / (2.0 * ha),
                                1e-5, 1e-7);
                CHECK_CLOSE_REL(label + " pdf gBeta vs FD", p.gBeta,
                                (pdfValue(a, b + hb, x) - pdfValue(a, b - hb, x)) / (2.0 * hb),
                                1e-5, 1e-7);
                CHECK_CLOSE_REL(label + " pdf hAA vs FD", p.hAA,
                                (pdfGA(a + ha, b, x) - pdfGA(a - ha, b, x)) / (2.0 * ha), 1e-5,
                                1e-7);
                CHECK_CLOSE_REL(label + " pdf hAB vs FD", p.hAB,
                                (pdfGA(a, b + hb, x) - pdfGA(a, b - hb, x)) / (2.0 * hb), 1e-5,
                                1e-7);
                CHECK_CLOSE_REL(label + " pdf hAx vs FD", p.hAx,
                                (pdfGA(a, b, x + hx) - pdfGA(a, b, x - hx)) / (2.0 * hx), 1e-5,
                                1e-7);
                CHECK_CLOSE_REL(label + " pdf hBB vs FD", p.hBB,
                                (pdfGB(a, b + hb, x) - pdfGB(a, b - hb, x)) / (2.0 * hb), 1e-5,
                                1e-7);
                CHECK_CLOSE_REL(label + " pdf hBx vs FD", p.hBx,
                                (pdfGB(a, b, x + hx) - pdfGB(a, b, x - hx)) / (2.0 * hx), 1e-5,
                                1e-7);
                CHECK_CLOSE_REL(label + " pdf hxx vs FD", p.hxx,
                                (pdfGx(a, b, x + hx) - pdfGx(a, b, x - hx)) / (2.0 * hx), 1e-5,
                                1e-7);
            }
        }
    }
}

TEST(BetaParameterDerivatives, symmetryIdentitiesBothBranches) {
    // I_x(a,b) + I_{1-x}(b,a) = 1 differentiated on both arguments: the pair
    // below always evaluates one side in each continued-fraction branch.
    for (const auto& [a, b] : {std::pair<double, double>{0.5, 0.5},
                               {2.7, 1.0},
                               {0.5, 10.0},
                               {10.0, 0.5},
                               {10.0, 10.0}}) {
        for (double x : {0.05, 0.25, 0.5, 0.75, 0.95}) {
            const std::string label =
                "a=" + std::to_string(a) + " b=" + std::to_string(b) + " x=" + std::to_string(x);
            const BetaCdfGradHess j = betaCdfGradHess(a, b, x);
            const BetaCdfGradHess k = betaCdfGradHess(b, a, 1.0 - x);
            CHECK_CLOSE(label + " value sum", j.value + k.value, 1.0, 1e-12);
            CHECK_CLOSE(label + " gAlpha + gBeta'", j.gAlpha + k.gBeta, 0.0, 1e-12);
            CHECK_CLOSE(label + " gBeta + gAlpha'", j.gBeta + k.gAlpha, 0.0, 1e-12);
            CHECK_CLOSE(label + " gx - gx'", j.gx - k.gx, 0.0, 1e-12);
            CHECK_CLOSE(label + " hAA + hBB'", j.hAA + k.hBB, 0.0, 1e-12);
            CHECK_CLOSE(label + " hAB + hAB'", j.hAB + k.hAB, 0.0, 1e-12);
            CHECK_CLOSE(label + " hBB + hAA'", j.hBB + k.hAA, 0.0, 1e-12);
            CHECK_CLOSE(label + " hAx - hBx'", j.hAx - k.hBx, 0.0, 1e-12);
            CHECK_CLOSE(label + " hBx - hAx'", j.hBx - k.hAx, 0.0, 1e-12);
            CHECK_CLOSE(label + " hxx + hxx'", j.hxx + k.hxx, 0.0, 1e-12);
        }
    }
}

TEST(BetaParameterDerivatives, largeShapesAndTails) {
    // Symmetric shapes around the exact median x = 1/2.
    const BetaCdfGradHess mid = betaCdfGradHess(1000.0, 1000.0, 0.5);
    CHECK_CLOSE_REL("large symmetric median", mid.value, 0.5, 1e-9, 0.0);
    CHECK_CLOSE("large symmetric gAlpha = -gBeta", mid.gAlpha + mid.gBeta, 0.0, 1e-12);
    CHECK_CLOSE("large symmetric hAA = -hBB", mid.hAA + mid.hBB, 0.0, 1e-12);
    CHECK_CLOSE("large symmetric hAB = 0", mid.hAB, 0.0, 1e-12);

    // Closed form I_x(a,1) = x^a and its a-derivatives, a = 1000.
    for (double x : {0.9, 0.99, 0.999}) {
        const double base = std::exp(1000.0 * std::log(x));
        const BetaCdfGradHess p = betaCdfGradHess(1000.0, 1.0, x);
        CHECK_CLOSE_REL("I_x(1000,1) = x^1000", p.value, base, 1e-10, 0.0);
        CHECK_CLOSE_REL("d/da x^a", p.gAlpha, base * std::log(x), 1e-10, 0.0);
        CHECK_CLOSE_REL("d2/da2 x^a", p.hAA, base * std::log(x) * std::log(x), 1e-10, 0.0);
    }

    // Deep tail: I_x(2,3) = 6x^2 - 8x^3 + 3x^4 exactly for x -> 0.
    const double small = 1e-8;
    const BetaCdfGradHess tail = betaCdfGradHess(2.0, 3.0, small);
    const double tailExact =
        6.0 * small * small - 8.0 * small * small * small + 3.0 * small * small * small * small;
    CHECK_CLOSE_REL("deep-tail value = 6x^2 - 8x^3 + 3x^4", tail.value, tailExact, 1e-10, 0.0);
    const double ha = 1e-5;
    CHECK_CLOSE_REL("deep-tail gAlpha vs FD", tail.gAlpha,
                    (betaCdfGradHess(2.0 + ha, 3.0, small).value -
                     betaCdfGradHess(2.0 - ha, 3.0, small).value) /
                        (2.0 * ha),
                    1e-5, 1e-12);
    // Upper tail: 1 - I_x(2,3) = I_{1-x}(3,2) ~ 4 eps^3; the complement loses
    // ~eps/4e-9 relative digits to the `1 - value` subtraction.
    const double eps = 1e-3;
    const BetaCdfGradHess upper = betaCdfGradHess(2.0, 3.0, 1.0 - eps);
    const BetaCdfGradHess mirror = betaCdfGradHess(3.0, 2.0, eps);
    CHECK_CLOSE_REL("direct mirror ~ 4 eps^3 - 3 eps^4", mirror.value,
                    4.0 * eps * eps * eps - 3.0 * eps * eps * eps * eps, 1e-10, 0.0);
    CHECK_CLOSE_REL("upper-tail complement = mirror value", 1.0 - upper.value, mirror.value, 1e-5,
                    0.0);
}
