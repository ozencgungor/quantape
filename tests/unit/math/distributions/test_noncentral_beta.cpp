// test_noncentral_beta.cpp — noncentral beta CDF/ccdf/density, the noncentral
// F transform, independent Poisson-mixture/Simpson oracles, full
// (alpha, beta, ncp, x) analytic derivative packs, and mpmath CSV gates.
#include "quantape/math/Distributions/Beta.h"
#include "quantape/math/Distributions/NoncentralBeta.h"

#include <cmath>
#include <filesystem>
#include <fstream>
#include <limits>
#include <sstream>
#include <string>

#include "support/GtestSupport.h"

namespace {

using quantape::math::Beta;
using quantape::math::NoncentralBeta;
using quantape::math::NoncentralF;
using quantape::math::detail::betaCcdfDouble;
using quantape::math::detail::betaCdfDouble;
using quantape::math::detail::betaPdfDouble;
using quantape::math::detail::NoncentralBetaCdfGradHess;
using quantape::math::detail::NoncentralBetaPdfGradHess;
using quantape::math::detail::NoncentralFCdfGradHess;
using quantape::math::detail::NoncentralFPdfGradHess;

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

/// Five-point central first derivative, O(h^4).
template <class F>
double centralD1(const F& f, double x, double h) {
    return (-f(x + 2 * h) + 8 * f(x + h) - 8 * f(x - h) + f(x - 2 * h)) / (12 * h);
}

constexpr double kInf = std::numeric_limits<double>::infinity();
constexpr double kNaN = std::numeric_limits<double>::quiet_NaN();

} // namespace

// ---------------------------------------------------------------------------
// Value/reduction tests.
// ---------------------------------------------------------------------------

TEST(NoncentralBetaDistribution, reducesToCentralAtZeroNcp) {
    for (double a : {0.5, 1.0, 2.5, 7.0}) {
        for (double b : {0.5, 3.0, 9.0}) {
            const Beta<double> central{a, b};
            const NoncentralBeta<double> nc{a, b, 0.0};
            for (double x : {0.02, 0.2, 0.5, 0.8, 0.98}) {
                const std::string label = "a=" + std::to_string(a) + " b=" + std::to_string(b) +
                                          " x=" + std::to_string(x);
                CHECK_CLOSE_REL(label + " cdf", nc.cdf(x), central.cdf(x), 1e-14);
                CHECK_CLOSE_REL(label + " ccdf", nc.ccdf(x), central.ccdf(x), 1e-14);
                CHECK_CLOSE_REL(label + " pdf", nc.pdf(x), central.pdf(x), 1e-14);
                CHECK_CLOSE_REL(label + " logPdf", nc.logPdf(x), central.logPdf(x), 1e-13);
                const NoncentralBetaCdfGradHess c = nc.cdfGradHess(x);
                const NoncentralBetaPdfGradHess p = nc.pdfGradHess(x);
                CHECK_CLOSE_REL(label + " pack cdf", c.value, central.cdf(x), 1e-14);
                CHECK_CLOSE_REL(label + " pack gx", c.gx, central.pdf(x), 1e-13);
                CHECK_CLOSE_REL(label + " pack pdf", p.value, central.pdf(x), 1e-14);
                CHECK_CLOSE_REL(label + " pack pgx", p.gx,
                                quantape::math::detail::betaPdfGradHess(a, b, x).gx, 1e-13);
            }
        }
    }
}

TEST(NoncentralBetaDistribution, independentPoissonMixture) {
    for (double ncp : {0.5, 2.0, 10.0}) {
        const NoncentralBeta<double> nc{2.5, 3.7, ncp};
        for (double x : {0.1, 0.4, 0.85}) {
            // Direct factorial weights and central beta CDFs (independent of
            // the implementation's mode-centred window).
            double sum = 0.0;
            double sumQ = 0.0;
            double w = std::exp(-0.5 * ncp);
            for (int j = 0; j <= 200; ++j) {
                sum += w * betaCdfDouble(2.5 + j, 3.7, x);
                sumQ += w * betaCcdfDouble(2.5 + j, 3.7, x);
                w *= (0.5 * ncp) / (j + 1.0);
            }
            const std::string label = "ncp=" + std::to_string(ncp) + " x=" + std::to_string(x);
            CHECK_CLOSE_REL(label + " cdf mixture", nc.cdf(x), sum, 1e-12);
            CHECK_CLOSE_REL(label + " ccdf mixture", nc.ccdf(x), sumQ, 1e-12);
            CHECK_CLOSE(label + " cdf+ccdf", nc.cdf(x) + nc.ccdf(x), 1.0, 1e-13);
        }
    }
}

TEST(NoncentralBetaDistribution, cdfMatchesSimpsonOfPdf) {
    // The density has integrable endpoint singularities for alpha, beta < 1,
    // so the integral check runs on intervals with a positive lower bound
    // (interval differences of the CDF).
    const double cases[4][3] = {{0.5, 2.0, 1.0}, {2.0, 0.5, 3.0}, {3.5, 7.0, 5.0}, {1.0, 1.0, 2.0}};
    for (const auto& c : cases) {
        const NoncentralBeta<double> nc{c[0], c[1], c[2]};
        for (const double lo : {0.05, 0.25}) {
            for (const double hi : {0.5, 0.75, 0.95}) {
                if (hi <= lo) {
                    continue;
                }
                const double integrated =
                    simpsonIntegrate([&](double t) { return nc.pdf(t); }, lo, hi, 40000);
                CHECK_CLOSE_REL("cdf difference vs Simpson(pdf)", nc.cdf(hi) - nc.cdf(lo),
                                integrated, 5e-9);
            }
        }
    }
}

TEST(NoncentralBetaDistribution, logPdfConsistency) {
    for (double ncp : {0.0, 1.0, 10.0}) {
        for (double a : {0.5, 2.5}) {
            for (double b : {0.5, 4.0}) {
                const NoncentralBeta<double> nc{a, b, ncp};
                for (double x : {1e-4, 0.2, 0.5, 0.8, 1.0 - 1e-4}) {
                    const double f = nc.pdf(x);
                    if (f > 0.0) {
                        CHECK_CLOSE("logPdf vs log(pdf)", nc.logPdf(x), std::log(f), 1e-12);
                    }
                    CHECK_CLOSE_REL("exp(logPdf) vs pdf", std::exp(nc.logPdf(x)), f, 1e-12, 1e-300);
                }
            }
        }
    }
    // The log-space mixture keeps deep-tail densities finite where the direct
    // mixture underflows (ncp = 80, x far in the upper tail).
    const NoncentralBeta<double> deep{2.0, 3.0, 80.0};
    const double f = deep.pdf(0.999);
    if (f > 0.0) {
        CHECK_CLOSE("deep logPdf", deep.logPdf(0.999), std::log(f), 1e-10);
    }
    EXPECT_TRUE(std::isfinite(deep.logPdf(0.999)));
}

// ---------------------------------------------------------------------------
// Full parameter-derivative packs.
// ---------------------------------------------------------------------------

// mpmath reference gates (tests/data/distributions/ncbeta_reference.csv,
// mpmath 1.4.1 dps=30, formal Poisson mixture differentiated with mp.diff and
// 20 significant digits). First-order entries gated at 1e-9 relative,
// second-order at 1e-8 (atol floors 1e-15/1e-14 cover the near-zero entries).
// Measured max relative errors over the committed 40-row grid (alpha, beta in
// {0.5, ..., 100}, ncp <= 30, x in (0,1)): CDF pack 3.7e-10 (gNcp at a
// near-zero value), all other entries <= 3.0e-12; PDF pack <= 5.5e-12.
TEST(NoncentralBetaParameterDerivatives, referenceCsvMatchesMpmath) {
    const std::filesystem::path relative = "distributions/ncbeta_reference.csv";
    SKIP_UNLESS_ASSET(std::filesystem::is_regular_file(testDataDir() / relative),
                      "ncbeta reference csv absent");
    std::ifstream in(RequireDataFile(relative));
    SKIP_UNLESS_ASSET(in.good(), "ncbeta reference csv unreadable");

    std::string line;
    std::getline(in, line); // header row
    int rows = 0;
    while (std::getline(in, line)) {
        if (line.empty() || line[0] == '#') {
            continue;
        }
        std::stringstream ss(line);
        std::string token;
        double f[34] = {};
        int n = 0;
        while (n < 34 && std::getline(ss, token, ',')) {
            f[n++] = std::stod(token);
        }
        ASSERT_EQ(n, 34) << "malformed reference row: " << line;
        const double a = f[0];
        const double b = f[1];
        const double ncp = f[2];
        const double x = f[3];
        const std::string label = "a=" + std::to_string(a) + " b=" + std::to_string(b) +
                                  " ncp=" + std::to_string(ncp) + " x=" + std::to_string(x);
        const NoncentralBetaCdfGradHess c =
            quantape::math::detail::noncentralBetaCdfGradHess(a, b, ncp, x);
        CHECK_CLOSE_REL(label + " cdf", c.value, f[4], 1e-10);
        CHECK_CLOSE_REL(label + " gAlpha", c.gAlpha, f[5], 1e-9, 1e-15);
        CHECK_CLOSE_REL(label + " gBeta", c.gBeta, f[6], 1e-9, 1e-15);
        CHECK_CLOSE_REL(label + " gNcp", c.gNcp, f[7], 1e-9, 1e-15);
        CHECK_CLOSE_REL(label + " gx", c.gx, f[8], 1e-9, 1e-15);
        CHECK_CLOSE_REL(label + " hAA", c.hAA, f[9], 1e-8, 1e-14);
        CHECK_CLOSE_REL(label + " hAB", c.hAB, f[10], 1e-8, 1e-14);
        CHECK_CLOSE_REL(label + " hANcp", c.hANcp, f[11], 1e-8, 1e-14);
        CHECK_CLOSE_REL(label + " hAx", c.hAx, f[12], 1e-8, 1e-14);
        CHECK_CLOSE_REL(label + " hBB", c.hBB, f[13], 1e-8, 1e-14);
        CHECK_CLOSE_REL(label + " hBNcp", c.hBNcp, f[14], 1e-8, 1e-14);
        CHECK_CLOSE_REL(label + " hBx", c.hBx, f[15], 1e-8, 1e-14);
        CHECK_CLOSE_REL(label + " hNcpNcp", c.hNcpNcp, f[16], 1e-8, 1e-15);
        CHECK_CLOSE_REL(label + " hNcpX", c.hNcpX, f[17], 1e-8, 1e-14);
        CHECK_CLOSE_REL(label + " hxx", c.hxx, f[18], 1e-8, 1e-14);
        const NoncentralBetaPdfGradHess p =
            quantape::math::detail::noncentralBetaPdfGradHess(a, b, ncp, x);
        CHECK_CLOSE_REL(label + " pdf", p.value, f[19], 1e-10);
        CHECK_CLOSE_REL(label + " pgAlpha", p.gAlpha, f[20], 1e-9, 1e-15);
        CHECK_CLOSE_REL(label + " pgBeta", p.gBeta, f[21], 1e-9, 1e-15);
        CHECK_CLOSE_REL(label + " pgNcp", p.gNcp, f[22], 1e-9, 1e-15);
        CHECK_CLOSE_REL(label + " pgx", p.gx, f[23], 1e-9, 1e-15);
        CHECK_CLOSE_REL(label + " phAA", p.hAA, f[24], 1e-8, 1e-14);
        CHECK_CLOSE_REL(label + " phAB", p.hAB, f[25], 1e-8, 1e-14);
        CHECK_CLOSE_REL(label + " phANcp", p.hANcp, f[26], 1e-8, 1e-14);
        CHECK_CLOSE_REL(label + " phAx", p.hAx, f[27], 1e-8, 1e-14);
        CHECK_CLOSE_REL(label + " phBB", p.hBB, f[28], 1e-8, 1e-14);
        CHECK_CLOSE_REL(label + " phBNcp", p.hBNcp, f[29], 1e-8, 1e-14);
        CHECK_CLOSE_REL(label + " phBx", p.hBx, f[30], 1e-8, 1e-14);
        CHECK_CLOSE_REL(label + " phNcpNcp", p.hNcpNcp, f[31], 1e-8, 1e-15);
        CHECK_CLOSE_REL(label + " phNcpX", p.hNcpX, f[32], 1e-8, 1e-14);
        CHECK_CLOSE_REL(label + " phxx", p.hxx, f[33], 1e-8, 1e-14);
        ++rows;
    }
    EXPECT_EQ(rows, 40);
}

TEST(NoncentralBetaParameterDerivatives, xEntriesReproduceCentralPacksAtZeroNcp) {
    for (double a : {0.5, 1.0, 2.5, 10.0}) {
        for (double b : {0.5, 1.0, 3.0, 10.0}) {
            for (double x : {1e-6, 0.1, 0.5, 0.9, 1.0 - 1e-6}) {
                const std::string label = "a=" + std::to_string(a) + " b=" + std::to_string(b) +
                                          " x=" + std::to_string(x);
                const auto c = quantape::math::detail::noncentralBetaCdfGradHess(a, b, 0.0, x);
                const auto bc = quantape::math::detail::betaCdfGradHess(a, b, x);
                CHECK_CLOSE(label + " cdf value", c.value, bc.value, 0.0);
                CHECK_CLOSE(label + " cdf gAlpha", c.gAlpha, bc.gAlpha, 0.0);
                CHECK_CLOSE(label + " cdf gBeta", c.gBeta, bc.gBeta, 0.0);
                CHECK_CLOSE(label + " cdf gx", c.gx, bc.gx, 0.0);
                CHECK_CLOSE(label + " cdf hxx", c.hxx, bc.hxx, 0.0);
                const auto p = quantape::math::detail::noncentralBetaPdfGradHess(a, b, 0.0, x);
                const auto bp = quantape::math::detail::betaPdfGradHess(a, b, x);
                CHECK_CLOSE(label + " pdf value", p.value, bp.value, 0.0);
                CHECK_CLOSE(label + " pdf gAlpha", p.gAlpha, bp.gAlpha, 0.0);
                CHECK_CLOSE(label + " pdf gx", p.gx, bp.gx, 0.0);
                CHECK_CLOSE(label + " pdf hxx", p.hxx, bp.hxx, 0.0);
            }
        }
    }
}

TEST(NoncentralBetaParameterDerivatives, packsMatchFiniteDifferences) {
    for (double a : {0.7, 2.5}) {
        for (double b : {1.3, 3.0}) {
            for (double ncp : {1.0}) {
                for (double x : {0.25, 0.6}) {
                    const std::string label = "a=" + std::to_string(a) + " b=" + std::to_string(b) +
                                              " ncp=" + std::to_string(ncp) +
                                              " x=" + std::to_string(x);
                    const auto c = quantape::math::detail::noncentralBetaCdfGradHess(a, b, ncp, x);
                    const auto p = quantape::math::detail::noncentralBetaPdfGradHess(a, b, ncp, x);
                    const auto cdf = [](double aa, double bb, double nn, double xx) {
                        return quantape::math::detail::noncentralBetaCdfDouble(aa, bb, nn, xx);
                    };
                    const auto pdf = [](double aa, double bb, double nn, double xx) {
                        return quantape::math::detail::noncentralBetaPdfDouble(aa, bb, nn, xx);
                    };
                    const auto cgA = [](double aa, double bb, double nn, double xx) {
                        return quantape::math::detail::noncentralBetaCdfGradHess(aa, bb, nn, xx)
                            .gAlpha;
                    };
                    const auto cgB = [](double aa, double bb, double nn, double xx) {
                        return quantape::math::detail::noncentralBetaCdfGradHess(aa, bb, nn, xx)
                            .gBeta;
                    };
                    const auto cgN = [](double aa, double bb, double nn, double xx) {
                        return quantape::math::detail::noncentralBetaCdfGradHess(aa, bb, nn, xx)
                            .gNcp;
                    };
                    const auto pgA = [](double aa, double bb, double nn, double xx) {
                        return quantape::math::detail::noncentralBetaPdfGradHess(aa, bb, nn, xx)
                            .gAlpha;
                    };
                    const auto pgB = [](double aa, double bb, double nn, double xx) {
                        return quantape::math::detail::noncentralBetaPdfGradHess(aa, bb, nn, xx)
                            .gBeta;
                    };
                    const auto pgN = [](double aa, double bb, double nn, double xx) {
                        return quantape::math::detail::noncentralBetaPdfGradHess(aa, bb, nn, xx)
                            .gNcp;
                    };
                    const double h = 1e-5;
                    CHECK_CLOSE(label + " cdf gAlpha vs FD", c.gAlpha,
                                centralD1([&](double t) { return cdf(t, b, ncp, x); }, a, h), 1e-6);
                    CHECK_CLOSE(label + " cdf gBeta vs FD", c.gBeta,
                                centralD1([&](double t) { return cdf(a, t, ncp, x); }, b, h), 1e-6);
                    CHECK_CLOSE(label + " cdf gNcp vs FD", c.gNcp,
                                centralD1([&](double t) { return cdf(a, b, t, x); }, ncp, h), 1e-6);
                    CHECK_CLOSE(label + " cdf gx vs FD", c.gx,
                                centralD1([&](double t) { return cdf(a, b, ncp, t); }, x, h), 1e-6);
                    CHECK_CLOSE(label + " cdf hAA vs FD", c.hAA,
                                centralD1([&](double t) { return cgA(t, b, ncp, x); }, a, h), 1e-5);
                    CHECK_CLOSE(label + " cdf hAB vs FD", c.hAB,
                                centralD1([&](double t) { return cgA(a, t, ncp, x); }, b, h), 1e-5);
                    CHECK_CLOSE(label + " cdf hANcp vs FD", c.hANcp,
                                centralD1([&](double t) { return cgA(a, b, t, x); }, ncp, h), 1e-5);
                    CHECK_CLOSE(label + " cdf hAx vs FD", c.hAx,
                                centralD1([&](double t) { return cgA(a, b, ncp, t); }, x, h), 1e-5);
                    CHECK_CLOSE(label + " cdf hBB vs FD", c.hBB,
                                centralD1([&](double t) { return cgB(a, t, ncp, x); }, b, h), 1e-5);
                    CHECK_CLOSE(label + " cdf hBNcp vs FD", c.hBNcp,
                                centralD1([&](double t) { return cgB(a, b, t, x); }, ncp, h), 1e-5);
                    CHECK_CLOSE(label + " cdf hBx vs FD", c.hBx,
                                centralD1([&](double t) { return cgB(a, b, ncp, t); }, x, h), 1e-5);
                    CHECK_CLOSE(label + " cdf hNcpNcp vs FD", c.hNcpNcp,
                                centralD1([&](double t) { return cgN(a, b, t, x); }, ncp, h), 1e-5);
                    CHECK_CLOSE(label + " cdf hNcpX vs FD", c.hNcpX,
                                centralD1([&](double t) { return cgN(a, b, ncp, t); }, x, h), 1e-5);
                    CHECK_CLOSE(label + " cdf hxx vs FD", c.hxx,
                                centralD1(
                                    [&](double t) {
                                        return quantape::math::detail::noncentralBetaCdfGradHess(
                                                   a, b, ncp, t)
                                            .gx;
                                    },
                                    x, h),
                                1e-5);
                    CHECK_CLOSE(label + " pdf value vs mix", p.value, pdf(a, b, ncp, x), 1e-14);
                    CHECK_CLOSE(label + " pdf gAlpha vs FD", p.gAlpha,
                                centralD1([&](double t) { return pdf(t, b, ncp, x); }, a, h), 1e-6);
                    CHECK_CLOSE(label + " pdf gBeta vs FD", p.gBeta,
                                centralD1([&](double t) { return pdf(a, t, ncp, x); }, b, h), 1e-6);
                    CHECK_CLOSE(label + " pdf gNcp vs FD", p.gNcp,
                                centralD1([&](double t) { return pdf(a, b, t, x); }, ncp, h), 1e-6);
                    CHECK_CLOSE(label + " pdf gx vs FD", p.gx,
                                centralD1([&](double t) { return pdf(a, b, ncp, t); }, x, h), 1e-6);
                    CHECK_CLOSE(label + " pdf hAA vs FD", p.hAA,
                                centralD1([&](double t) { return pgA(t, b, ncp, x); }, a, h), 1e-5);
                    CHECK_CLOSE(label + " pdf hAB vs FD", p.hAB,
                                centralD1([&](double t) { return pgA(a, t, ncp, x); }, b, h), 1e-5);
                    CHECK_CLOSE(label + " pdf hANcp vs FD", p.hANcp,
                                centralD1([&](double t) { return pgA(a, b, t, x); }, ncp, h), 1e-5);
                    CHECK_CLOSE(label + " pdf hAx vs FD", p.hAx,
                                centralD1([&](double t) { return pgA(a, b, ncp, t); }, x, h), 1e-5);
                    CHECK_CLOSE(label + " pdf hBB vs FD", p.hBB,
                                centralD1([&](double t) { return pgB(a, t, ncp, x); }, b, h), 1e-5);
                    CHECK_CLOSE(label + " pdf hBNcp vs FD", p.hBNcp,
                                centralD1([&](double t) { return pgB(a, b, t, x); }, ncp, h), 1e-5);
                    CHECK_CLOSE(label + " pdf hBx vs FD", p.hBx,
                                centralD1([&](double t) { return pgB(a, b, ncp, t); }, x, h), 1e-5);
                    CHECK_CLOSE(label + " pdf hNcpNcp vs FD", p.hNcpNcp,
                                centralD1([&](double t) { return pgN(a, b, t, x); }, ncp, h), 1e-5);
                    CHECK_CLOSE(label + " pdf hNcpX vs FD", p.hNcpX,
                                centralD1([&](double t) { return pgN(a, b, ncp, t); }, x, h), 1e-5);
                    CHECK_CLOSE(label + " pdf hxx vs FD", p.hxx,
                                centralD1(
                                    [&](double t) {
                                        return quantape::math::detail::noncentralBetaPdfGradHess(
                                                   a, b, ncp, t)
                                            .gx;
                                    },
                                    x, h),
                                1e-5);
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Edges: x = 0/1, large ncp, small shapes, invalid inputs.
// ---------------------------------------------------------------------------

TEST(NoncentralBetaDistribution, edgesAndLargeNcp) {
    const NoncentralBeta<double> dist{2.0, 3.0, 2.0};
    CHECK_CLOSE("cdf(0) = 0", dist.cdf(0.0), 0.0, 0.0);
    CHECK_CLOSE("cdf(1) = 1", dist.cdf(1.0), 1.0, 0.0);
    CHECK_CLOSE("ccdf(0) = 1", dist.ccdf(0.0), 1.0, 0.0);
    CHECK_CLOSE("ccdf(1) = 0", dist.ccdf(1.0), 0.0, 0.0);
    CHECK_CLOSE("pdf outside = 0", dist.pdf(-0.1), 0.0, 0.0);
    CHECK_CLOSE("pdf outside = 0", dist.pdf(1.1), 0.0, 0.0);
    CHECK_CLOSE("cdf below = 0", dist.cdf(-0.1), 0.0, 0.0);
    CHECK_CLOSE("cdf above = 1", dist.cdf(1.1), 1.0, 0.0);
    CHECK_CLOSE("logPdf outside = -inf", dist.logPdf(-0.1), -kInf, 0.0);

    // Singular endpoint for alpha < 1 (the j = 0 term carries the +inf).
    const NoncentralBeta<double> singular{0.5, 2.0, 1.0};
    EXPECT_TRUE(std::isinf(singular.pdf(0.0)));
    EXPECT_TRUE(singular.pdf(0.0) > 0.0);
    const auto singularPack = singular.cdfGradHess(0.0);
    CHECK_CLOSE("singular pack value", singularPack.value, 0.0, 0.0);
    CHECK_CLOSE("singular pack gAlpha", singularPack.gAlpha, 0.0, 0.0);
    EXPECT_TRUE(std::isinf(singularPack.gx));

    // x = 1 for beta < 1.
    const NoncentralBeta<double> singularHigh{2.0, 0.5, 1.0};
    EXPECT_TRUE(std::isinf(singularHigh.pdf(1.0)));

    // alpha = 1: the endpoint density is finite but the one-sided curvature is
    // not represented (NaN gx, exactly like the central pack).
    const NoncentralBeta<double> unitShape{1.0, 2.0, 1.0};
    EXPECT_TRUE(std::isnan(unitShape.pdfGradHess(0.0).gx));

    // Large ncp: the mode-centred window still resolves the bulk; the tail
    // rule holds and the density value is positive.
    const NoncentralBeta<double> large{2.0, 3.0, 200.0};
    for (double x : {0.05, 0.3, 0.7, 0.99}) {
        const double c = large.cdf(x);
        const double q = large.ccdf(x);
        EXPECT_TRUE(c >= 0.0 && c <= 1.0) << "x=" << x;
        CHECK_CLOSE("large ncp cdf+ccdf", c + q, 1.0, 1e-12);
    }
    EXPECT_TRUE(large.pdf(0.5) > 0.0);

    // Deep lower tail stays representable in log space.
    const NoncentralBeta<double> deep{1.5, 2.5, 3.0};
    EXPECT_TRUE(std::isfinite(deep.logPdf(1e-12)));

    // Invalid inputs propagate NaN.
    EXPECT_TRUE(std::isnan((NoncentralBeta<double>{0.0, 2.0, 1.0}).cdf(0.5)));
    EXPECT_TRUE(std::isnan((NoncentralBeta<double>{2.0, 0.0, 1.0}).pdf(0.5)));
    EXPECT_TRUE(std::isnan((NoncentralBeta<double>{2.0, 3.0, -1.0}).cdf(0.5)));
    EXPECT_TRUE(std::isnan(dist.cdf(kNaN)));
    const auto nanPack = quantape::math::detail::noncentralBetaCdfGradHess(0.0, 2.0, 1.0, 0.5);
    EXPECT_TRUE(std::isnan(nanPack.value));
    const auto zeroPack = quantape::math::detail::noncentralBetaPdfGradHess(2.0, 3.0, 1.0, -0.5);
    CHECK_CLOSE("pdf pack below support", zeroPack.value, 0.0, 0.0);
    CHECK_CLOSE("pdf pack below support gx", zeroPack.gx, 0.0, 0.0);
}

// ---------------------------------------------------------------------------
// Noncentral F: transform, independent mixture oracle, packs, edges.
// ---------------------------------------------------------------------------

TEST(NoncentralFDistribution, transformMatchesNoncentralBeta) {
    for (double n1 : {3.0, 10.0}) {
        for (double n2 : {5.0, 20.0}) {
            for (double ncp : {0.0, 2.0}) {
                const NoncentralF<double> nf{n1, n2, ncp};
                const NoncentralBeta<double> nb{0.5 * n1, 0.5 * n2, ncp};
                for (double y : {0.2, 1.0, 4.0}) {
                    const double d = n2 + n1 * y;
                    const double x = n1 * y / d;
                    const double jac = (n1 * n2) / (d * d);
                    const std::string label =
                        "n1=" + std::to_string(n1) + " n2=" + std::to_string(n2) +
                        " ncp=" + std::to_string(ncp) + " y=" + std::to_string(y);
                    CHECK_CLOSE_REL(label + " cdf", nf.cdf(y), nb.cdf(x), 1e-14);
                    CHECK_CLOSE_REL(label + " ccdf", nf.ccdf(y), nb.ccdf(x), 1e-14);
                    CHECK_CLOSE_REL(label + " pdf", nf.pdf(y), nb.pdf(x) * jac, 1e-13);
                    CHECK_CLOSE(label + " logPdf", nf.logPdf(y), nb.logPdf(x) + std::log(jac),
                                1e-12);
                }
            }
        }
    }
}

TEST(NoncentralFDistribution, matchesIndependentPoissonMixture) {
    for (double n1 : {3.0, 10.0}) {
        for (double n2 : {5.0, 20.0}) {
            for (double ncp : {0.0, 2.0, 12.0}) {
                const NoncentralF<double> nf{n1, n2, ncp};
                for (double y : {0.3, 1.5, 5.0}) {
                    const double d = n2 + n1 * y;
                    const double x = n1 * y / d;
                    double sum = 0.0;
                    double sumPdf = 0.0;
                    double w = std::exp(-0.5 * ncp);
                    for (int j = 0; j <= 200; ++j) {
                        sum += w * betaCdfDouble(0.5 * n1 + j, 0.5 * n2, x);
                        sumPdf += w * betaPdfDouble(0.5 * n1 + j, 0.5 * n2, x);
                        w *= (0.5 * ncp) / (j + 1.0);
                    }
                    const std::string label =
                        "n1=" + std::to_string(n1) + " n2=" + std::to_string(n2) +
                        " ncp=" + std::to_string(ncp) + " y=" + std::to_string(y);
                    CHECK_CLOSE_REL(label + " cdf mixture", nf.cdf(y), sum, 1e-12);
                    CHECK_CLOSE_REL(label + " pdf mixture", nf.pdf(y), sumPdf * n1 * n2 / (d * d),
                                    1e-12);
                }
            }
        }
    }
}

TEST(NoncentralFParameterDerivatives, packsMatchFiniteDifferences) {
    for (double n1 : {3.0, 11.0}) {
        for (double n2 : {5.0, 17.0}) {
            for (double ncp : {1.0}) {
                for (double y : {0.4, 2.5}) {
                    const std::string label =
                        "n1=" + std::to_string(n1) + " n2=" + std::to_string(n2) +
                        " ncp=" + std::to_string(ncp) + " y=" + std::to_string(y);
                    const NoncentralFCdfGradHess c =
                        quantape::math::detail::noncentralFCdfGradHess(n1, n2, ncp, y);
                    const NoncentralFPdfGradHess p =
                        quantape::math::detail::noncentralFPdfGradHess(n1, n2, ncp, y);
                    const auto cdf = [](double a, double b, double n, double t) {
                        return quantape::math::detail::noncentralFCdfDouble(a, b, n, t);
                    };
                    const auto pdf = [](double a, double b, double n, double t) {
                        return quantape::math::detail::noncentralFPdfDouble(a, b, n, t);
                    };
                    const auto cg1 = [](double a, double b, double n, double t) {
                        return quantape::math::detail::noncentralFCdfGradHess(a, b, n, t).gNu1;
                    };
                    const auto cg2 = [](double a, double b, double n, double t) {
                        return quantape::math::detail::noncentralFCdfGradHess(a, b, n, t).gNu2;
                    };
                    const auto cgN = [](double a, double b, double n, double t) {
                        return quantape::math::detail::noncentralFCdfGradHess(a, b, n, t).gNcp;
                    };
                    const auto cgy = [](double a, double b, double n, double t) {
                        return quantape::math::detail::noncentralFCdfGradHess(a, b, n, t).gy;
                    };
                    const auto pg1 = [](double a, double b, double n, double t) {
                        return quantape::math::detail::noncentralFPdfGradHess(a, b, n, t).gNu1;
                    };
                    const auto pg2 = [](double a, double b, double n, double t) {
                        return quantape::math::detail::noncentralFPdfGradHess(a, b, n, t).gNu2;
                    };
                    const auto pgN = [](double a, double b, double n, double t) {
                        return quantape::math::detail::noncentralFPdfGradHess(a, b, n, t).gNcp;
                    };
                    const auto pgy = [](double a, double b, double n, double t) {
                        return quantape::math::detail::noncentralFPdfGradHess(a, b, n, t).gy;
                    };
                    const double h = 1e-5;
                    CHECK_CLOSE_REL(label + " cdf value vs FD", c.value, cdf(n1, n2, ncp, y),
                                    1e-14);
                    CHECK_CLOSE(label + " cdf gNu1 vs FD", c.gNu1,
                                centralD1([&](double t) { return cdf(t, n2, ncp, y); }, n1, h),
                                1e-6);
                    CHECK_CLOSE(label + " cdf gNu2 vs FD", c.gNu2,
                                centralD1([&](double t) { return cdf(n1, t, ncp, y); }, n2, h),
                                1e-6);
                    CHECK_CLOSE(label + " cdf gNcp vs FD", c.gNcp,
                                centralD1([&](double t) { return cdf(n1, n2, t, y); }, ncp, h),
                                1e-6);
                    CHECK_CLOSE(label + " cdf gy vs FD", c.gy,
                                centralD1([&](double t) { return cdf(n1, n2, ncp, t); }, y, h),
                                1e-6);
                    CHECK_CLOSE(label + " cdf hNu1Nu1 vs FD", c.hNu1Nu1,
                                centralD1([&](double t) { return cg1(t, n2, ncp, y); }, n1, h),
                                1e-5);
                    CHECK_CLOSE(label + " cdf hNu1Nu2 vs FD", c.hNu1Nu2,
                                centralD1([&](double t) { return cg1(n1, t, ncp, y); }, n2, h),
                                1e-5);
                    CHECK_CLOSE(label + " cdf hNu1Ncp vs FD", c.hNu1Ncp,
                                centralD1([&](double t) { return cg1(n1, n2, t, y); }, ncp, h),
                                1e-5);
                    CHECK_CLOSE(label + " cdf hNu1Y vs FD", c.hNu1Y,
                                centralD1([&](double t) { return cg1(n1, n2, ncp, t); }, y, h),
                                1e-5);
                    CHECK_CLOSE(label + " cdf hNu2Nu2 vs FD", c.hNu2Nu2,
                                centralD1([&](double t) { return cg2(n1, t, ncp, y); }, n2, h),
                                1e-5);
                    CHECK_CLOSE(label + " cdf hNu2Ncp vs FD", c.hNu2Ncp,
                                centralD1([&](double t) { return cg2(n1, n2, t, y); }, ncp, h),
                                1e-5);
                    CHECK_CLOSE(label + " cdf hNu2Y vs FD", c.hNu2Y,
                                centralD1([&](double t) { return cg2(n1, n2, ncp, t); }, y, h),
                                1e-5);
                    CHECK_CLOSE(label + " cdf hNcpNcp vs FD", c.hNcpNcp,
                                centralD1([&](double t) { return cgN(n1, n2, t, y); }, ncp, h),
                                1e-5);
                    CHECK_CLOSE(label + " cdf hNcpY vs FD", c.hNcpY,
                                centralD1([&](double t) { return cgN(n1, n2, ncp, t); }, y, h),
                                1e-5);
                    CHECK_CLOSE(label + " cdf hyy vs FD", c.hyy,
                                centralD1([&](double t) { return cgy(n1, n2, ncp, t); }, y, h),
                                1e-5);
                    CHECK_CLOSE_REL(label + " pdf value vs FD", p.value, pdf(n1, n2, ncp, y),
                                    1e-14);
                    CHECK_CLOSE(label + " pdf gNu1 vs FD", p.gNu1,
                                centralD1([&](double t) { return pdf(t, n2, ncp, y); }, n1, h),
                                1e-6);
                    CHECK_CLOSE(label + " pdf gNu2 vs FD", p.gNu2,
                                centralD1([&](double t) { return pdf(n1, t, ncp, y); }, n2, h),
                                1e-6);
                    CHECK_CLOSE(label + " pdf gNcp vs FD", p.gNcp,
                                centralD1([&](double t) { return pdf(n1, n2, t, y); }, ncp, h),
                                1e-6);
                    CHECK_CLOSE(label + " pdf gy vs FD", p.gy,
                                centralD1([&](double t) { return pdf(n1, n2, ncp, t); }, y, h),
                                1e-6);
                    CHECK_CLOSE(label + " pdf hNu1Nu1 vs FD", p.hNu1Nu1,
                                centralD1([&](double t) { return pg1(t, n2, ncp, y); }, n1, h),
                                1e-5);
                    CHECK_CLOSE(label + " pdf hNu1Nu2 vs FD", p.hNu1Nu2,
                                centralD1([&](double t) { return pg1(n1, t, ncp, y); }, n2, h),
                                1e-5);
                    CHECK_CLOSE(label + " pdf hNu1Ncp vs FD", p.hNu1Ncp,
                                centralD1([&](double t) { return pg1(n1, n2, t, y); }, ncp, h),
                                1e-5);
                    CHECK_CLOSE(label + " pdf hNu1Y vs FD", p.hNu1Y,
                                centralD1([&](double t) { return pg1(n1, n2, ncp, t); }, y, h),
                                1e-5);
                    CHECK_CLOSE(label + " pdf hNu2Nu2 vs FD", p.hNu2Nu2,
                                centralD1([&](double t) { return pg2(n1, t, ncp, y); }, n2, h),
                                1e-5);
                    CHECK_CLOSE(label + " pdf hNu2Ncp vs FD", p.hNu2Ncp,
                                centralD1([&](double t) { return pg2(n1, n2, t, y); }, ncp, h),
                                1e-5);
                    CHECK_CLOSE(label + " pdf hNu2Y vs FD", p.hNu2Y,
                                centralD1([&](double t) { return pg2(n1, n2, ncp, t); }, y, h),
                                1e-5);
                    CHECK_CLOSE(label + " pdf hNcpNcp vs FD", p.hNcpNcp,
                                centralD1([&](double t) { return pgN(n1, n2, t, y); }, ncp, h),
                                1e-5);
                    CHECK_CLOSE(label + " pdf hNcpY vs FD", p.hNcpY,
                                centralD1([&](double t) { return pgN(n1, n2, ncp, t); }, y, h),
                                1e-5);
                    CHECK_CLOSE(label + " pdf hyy vs FD", p.hyy,
                                centralD1([&](double t) { return pgy(n1, n2, ncp, t); }, y, h),
                                1e-4);
                }
            }
        }
    }
}

TEST(NoncentralFDistribution, edgesAndInvalidInputs) {
    const NoncentralF<double> dist{3.0, 7.0, 2.0};
    CHECK_CLOSE("cdf(0) = 0", dist.cdf(0.0), 0.0, 0.0);
    CHECK_CLOSE("ccdf(0) = 1", dist.ccdf(0.0), 1.0, 0.0);
    CHECK_CLOSE("cdf(-1) = 0", dist.cdf(-1.0), 0.0, 0.0);
    CHECK_CLOSE("pdf(-1) = 0", dist.pdf(-1.0), 0.0, 0.0);
    CHECK_CLOSE("cdf(inf) = 1", dist.cdf(kInf), 1.0, 0.0);
    CHECK_CLOSE("ccdf(inf) = 0", dist.ccdf(kInf), 0.0, 0.0);
    CHECK_CLOSE("pdf(inf) = 0", dist.pdf(kInf), 0.0, 0.0);
    CHECK_CLOSE("logPdf(inf) = -inf", dist.logPdf(kInf), -kInf, 0.0);
    const NoncentralFCdfGradHess infPack = dist.cdfGradHess(kInf);
    CHECK_CLOSE("cdf pack(inf)", infPack.value, 1.0, 0.0);
    CHECK_CLOSE("cdf pack(inf) gF", infPack.gy, 0.0, 0.0);
    EXPECT_TRUE(std::isnan((NoncentralF<double>{0.0, 7.0, 2.0}).cdf(1.0)));
    EXPECT_TRUE(std::isnan((NoncentralF<double>{3.0, -1.0, 2.0}).pdf(1.0)));
    EXPECT_TRUE(std::isnan((NoncentralF<double>{3.0, 7.0, -2.0}).cdf(1.0)));
    EXPECT_TRUE(std::isnan(dist.cdf(kNaN)));
    const NoncentralFPdfGradHess zeroPack =
        quantape::math::detail::noncentralFPdfGradHess(3.0, 7.0, 2.0, -1.0);
    CHECK_CLOSE("pdf pack below support", zeroPack.value, 0.0, 0.0);
}
