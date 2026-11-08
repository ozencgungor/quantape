// test_gamma.cpp — Gamma / InverseGamma laws: closed forms, quadrature, quantiles,
// parameter-derivative packs (mpmath references, FD cross-checks, both regimes).

#include "quantape/math/Distributions/Gamma.h"
#include "quantape/math/Integrals/GaussLegendre.h"
#include "quantape/math/Integrals/GaussianQuadrature.h"

#include <cmath>
#include <filesystem>
#include <fstream>
#include <limits>
#include <sstream>
#include <string>

#include "support/GtestSupport.h"

using quantape::math::Gamma;
using quantape::math::GaussLegendre;
using quantape::math::GaussLegendreIntegrator;
using quantape::math::InverseGamma;
using quantape::math::detail::gammaCdfD1D2;
using quantape::math::detail::gammaCdfGradHess;
using quantape::math::detail::gammaPdfD1D2;
using quantape::math::detail::gammaPdfGradHess;
using quantape::math::detail::inverseGammaCdfGradHess;
using quantape::math::detail::inverseGammaPdfGradHess;
using quantape::math::detail::ValueD1D2;

namespace {

constexpr double kTol = 1e-12;
constexpr double kInf = std::numeric_limits<double>::infinity();

template <class F>
double centralD1(const F& f, double x, double h) {
    return (f(x + h) - f(x - h)) / (2.0 * h);
}

template <class F>
double centralD2(const F& f, double x, double h) {
    return (f(x + h) - 2.0 * f(x) + f(x - h)) / (h * h);
}

template <class Dist>
void checkMonotoneCdf(const Dist& dist, double lo, double hi, int steps) {
    double prev = dist.cdf(lo);
    for (int i = 1; i <= steps; ++i) {
        const double x = lo + (hi - lo) * static_cast<double>(i) / static_cast<double>(steps);
        SCOPED_TRACE(x);
        const double cur = dist.cdf(x);
        EXPECT_LE(prev, cur);
        prev = cur;
    }
}

} // namespace

TEST(GammaDistribution, cdfMatchesExponentialClosedForm) {
    const Gamma<double> g{1.0, 1.5};
    for (double x : {0.0, 0.05, 0.5, 2.0, 10.0, 80.0}) {
        CHECK_CLOSE("cdf vs 1-exp(-1.5x)", g.cdf(x), -std::expm1(-1.5 * x), kTol);
        CHECK_CLOSE("ccdf vs exp(-1.5x)", g.ccdf(x), std::exp(-1.5 * x), kTol);
    }
    CHECK_CLOSE("pdf(0) = rate", g.pdf(0.0), 1.5, kTol);
    CHECK_CLOSE("logPdf(0) = log(rate)", g.logPdf(0.0), std::log(1.5), kTol);
}

TEST(GammaDistribution, cdfMatchesQuadratureOfPdf) {
    const GaussLegendre rule(160);
    const Gamma<double> g{2.5, 1.3};
    for (double x : {0.8, 2.0, 5.0, 12.0}) {
        const auto density = [&g](double t) { return g.pdf(t); };
        CHECK_CLOSE("GL160 integral of pdf", rule.integrate(density, 0.0, x), g.cdf(x), 1e-9);
    }
    // Integer shape has no x^(shape-1) endpoint singularity, so a tabulated
    // order-20 rule already reproduces the CDF to machine precision.
    const GaussLegendreIntegrator<double> gauss20(20);
    const Gamma<double> smooth{3.0, 1.0};
    for (double x : {2.0, 4.0, 8.0}) {
        const auto density = [&smooth](double t) { return smooth.pdf(t); };
        CHECK_CLOSE("GL20 integral of pdf", gauss20(density, 0.0, x), smooth.cdf(x), kTol);
    }
}

TEST(GammaDistribution, cdfDerivativeMatchesPdf) {
    const Gamma<double> g{2.5, 1.3};
    for (double x : {0.3, 1.0, 2.5, 6.0}) {
        const double h = 1e-5 * (x > 1.0 ? x : 1.0);
        CHECK_CLOSE("central difference of cdf vs pdf",
                    centralD1([&g](double t) { return g.cdf(t); }, x, h), g.pdf(x), 1e-8);
    }
}

TEST(GammaDistribution, derivativePacksMatchFiniteDifferences) {
    const auto pdfPack = [](double x) { return gammaPdfD1D2(2.5, 1.3, x); };
    const auto cdfPack = [](double x) { return gammaCdfD1D2(2.5, 1.3, x); };
    const auto pdfValue = [&pdfPack](double x) { return pdfPack(x).value; };
    const auto cdfValue = [&cdfPack](double x) { return cdfPack(x).value; };
    const Gamma<double> g{2.5, 1.3};

    const double xp = 10.0;
    const double hp = 1e-5 * xp;
    const ValueD1D2 packP = pdfPack(xp);
    CHECK_CLOSE("pdf pack value", packP.value, g.pdf(xp), kTol);
    CHECK_CLOSE("pdf pack d1 vs FD", packP.d1, centralD1(pdfValue, xp, hp), 1e-7);
    CHECK_CLOSE("pdf pack d2 vs FD", packP.d2, centralD2(pdfValue, xp, hp), 1e-7);

    const double xc = 0.1;
    const double hc = 1e-5;
    const ValueD1D2 packC = cdfPack(xc);
    CHECK_CLOSE("cdf pack value", packC.value, g.cdf(xc), kTol);
    CHECK_CLOSE("cdf pack d1 vs FD", packC.d1, centralD1(cdfValue, xc, hc), 1e-7);
    CHECK_CLOSE("cdf pack d2 vs FD", packC.d2, centralD2(cdfValue, xc, hc), 1e-7);
}

TEST(GammaDistribution, normalizationMonotoneAndEdges) {
    const Gamma<double> g{2.5, 1.3};
    CHECK_CLOSE("cdf(+inf) = 1", g.cdf(kInf), 1.0, kTol);
    CHECK_CLOSE("ccdf(+inf) = 0", g.ccdf(kInf), 0.0, kTol);
    CHECK_CLOSE("logCdf(+inf) = 0", g.logCdf(kInf), 0.0, kTol);
    CHECK_CLOSE("logCcdf(+inf) = -inf", g.logCcdf(kInf), -kInf, 0.0);
    CHECK_CLOSE("cdf(-1) = 0", g.cdf(-1.0), 0.0, kTol);
    CHECK_CLOSE("pdf(-1) = 0", g.pdf(-1.0), 0.0, kTol);
    CHECK_CLOSE("logCdf(-1) = -inf", g.logCdf(-1.0), -kInf, 0.0);
    CHECK_CLOSE("ccdf(-1) = 1", g.ccdf(-1.0), 1.0, kTol);
    CHECK_CLOSE("pdf(+inf) = 0", g.pdf(kInf), 0.0, kTol);
    checkMonotoneCdf(g, 0.0, 15.0, 30);
    for (double x : {0.05, 1.0, 4.0, 20.0}) {
        CHECK_CLOSE("cdf + ccdf = 1", g.cdf(x) + g.ccdf(x), 1.0, kTol);
    }
    EXPECT_TRUE(std::isnan(g.cdf(std::numeric_limits<double>::quiet_NaN())));
    EXPECT_TRUE(std::isnan(Gamma<double>{-1.0, 1.0}.pdf(1.0)));
}

TEST(GammaDistribution, halfShapeEdgesAndLogTails) {
    const Gamma<double> g{0.5, 2.0};
    CHECK_CLOSE("cdf(0)", g.cdf(0.0), 0.0, kTol);
    CHECK_CLOSE("ccdf(0)", g.ccdf(0.0), 1.0, kTol);
    CHECK_CLOSE("logCdf(0)", g.logCdf(0.0), -kInf, 0.0);
    CHECK_CLOSE("logCcdf(0)", g.logCcdf(0.0), 0.0, kTol);
    EXPECT_TRUE(std::isinf(g.pdf(0.0)) && g.pdf(0.0) > 0.0);
    EXPECT_TRUE(std::isinf(g.logPdf(0.0)) && g.logPdf(0.0) > 0.0);
    CHECK_CLOSE("P(1/2, y) = erf(sqrt(y))", g.cdf(0.7), std::erf(std::sqrt(1.4)), kTol);

    const Gamma<double> deep{2.0, 1.0};
    const double x = 1e-20;
    CHECK_CLOSE("logCdf deep tail = log(cdf)", deep.logCdf(x), std::log(deep.cdf(x)), kTol);
    CHECK_CLOSE("logCcdf far tail = log(ccdf)", deep.logCcdf(50.0), std::log(deep.ccdf(50.0)),
                kTol);
}

TEST(GammaDistribution, quantileRoundTrips) {
    const Gamma<double> g{2.5, 1.3};
    for (double p : {1e-10, 1e-6, 0.01, 0.25, 0.5, 0.9, 1.0 - 1e-6}) {
        CHECK_CLOSE_REL("cdf(inv(p)) = p", g.cdf(g.inv(p)), p, 1e-10);
    }
    for (double x0 : {0.05, 0.4, 1.0, 3.0, 8.0, 12.0}) {
        CHECK_CLOSE_REL("inv(cdf(x)) = x", g.inv(g.cdf(x0)), x0, 1e-10);
    }
    const Gamma<double> half{0.5, 1.0};
    for (double p : {1e-8, 1e-3, 0.3, 0.8, 0.999}) {
        CHECK_CLOSE_REL("half-shape cdf(inv(p)) = p", half.cdf(half.inv(p)), p, 1e-9);
    }
    CHECK_CLOSE("inv(0) = 0", g.inv(0.0), 0.0, kTol);
    CHECK_CLOSE("inv(-0.5) = 0", g.inv(-0.5), 0.0, kTol);
    CHECK_CLOSE("inv(1) = +inf", g.inv(1.0), kInf, 0.0);
}

TEST(GammaDistribution, quantileUltraTailsAreFinite) {
    const Gamma<double> g{2.5, 1.3};
    const double x = g.inv(1e-100);
    EXPECT_TRUE(std::isfinite(x) && x > 0.0);
    CHECK_CLOSE_REL("cdf(inv(1e-100)) = 1e-100", g.cdf(x), 1e-100, 1e-10);
    const double tiny = g.inv(1e-300);
    EXPECT_TRUE(tiny > 0.0 && std::isfinite(tiny));
}

TEST(InverseGamma, closedFormPdfCdfAndEdges) {
    const InverseGamma<double> g{1.0, 1.0};
    for (double x : {0.2, 0.5, 1.0, 2.0, 10.0}) {
        CHECK_CLOSE("cdf vs exp(-scale/x)", g.cdf(x), std::exp(-1.0 / x), kTol);
        CHECK_CLOSE("pdf vs exp(-scale/x)/x^2", g.pdf(x), std::exp(-1.0 / x) / (x * x), kTol);
        CHECK_CLOSE("logCdf vs log(cdf)", g.logCdf(x), std::log(g.cdf(x)), kTol);
        CHECK_CLOSE("logCcdf vs log(ccdf)", g.logCcdf(x), std::log(g.ccdf(x)), kTol);
    }
    const InverseGamma<double> g23{2.0, 3.0};
    CHECK_CLOSE("cdf(1) = Q(2,3) = 4e^-3", g23.cdf(1.0), 4.0 * std::exp(-3.0), kTol);
    CHECK_CLOSE("cdf(0)", g23.cdf(0.0), 0.0, kTol);
    CHECK_CLOSE("logCdf(0)", g23.logCdf(0.0), -kInf, 0.0);
    CHECK_CLOSE("ccdf(0)", g23.ccdf(0.0), 1.0, kTol);
    CHECK_CLOSE("cdf(+inf)", g23.cdf(kInf), 1.0, kTol);
    CHECK_CLOSE("ccdf(+inf)", g23.ccdf(kInf), 0.0, kTol);
    CHECK_CLOSE("logCcdf(+inf)", g23.logCcdf(kInf), -kInf, 0.0);
    CHECK_CLOSE("pdf(0)", g23.pdf(0.0), 0.0, kTol);
    CHECK_CLOSE("pdf(+inf)", g23.pdf(kInf), 0.0, kTol);
    CHECK_CLOSE("logPdf(0)", g23.logPdf(0.0), -kInf, 0.0);
    checkMonotoneCdf(g23, 0.0, 30.0, 30);
}

TEST(InverseGamma, cdfMatchesQuadratureAndPdfDerivative) {
    const GaussLegendre rule(160);
    const InverseGamma<double> g{3.0, 2.0};
    for (double x : {0.5, 1.0, 2.0, 5.0, 20.0}) {
        const auto density = [&g](double t) { return g.pdf(t); };
        CHECK_CLOSE("GL160 integral of pdf", rule.integrate(density, 1e-9, x), g.cdf(x), 1e-9);
    }
    for (double x : {0.5, 1.0, 3.0, 8.0}) {
        const double h = 1e-5 * (x > 1.0 ? x : 1.0);
        CHECK_CLOSE("central difference of cdf vs pdf",
                    centralD1([&g](double t) { return g.cdf(t); }, x, h), g.pdf(x), 1e-8);
    }
}

TEST(InverseGamma, reciprocalGammaIdentity) {
    const Gamma<double> gam{2.5, 1.3};
    const InverseGamma<double> inv{2.5, 1.3};
    for (double x : {0.1, 0.5, 1.7, 6.0}) {
        CHECK_CLOSE("cdf = gamma ccdf(1/x)", inv.cdf(x), gam.ccdf(1.0 / x), kTol);
        CHECK_CLOSE("ccdf = gamma cdf(1/x)", inv.ccdf(x), gam.cdf(1.0 / x), kTol);
        CHECK_CLOSE("logCdf = gamma logCcdf(1/x)", inv.logCdf(x), gam.logCcdf(1.0 / x), kTol);
        CHECK_CLOSE("logCcdf = gamma logCdf(1/x)", inv.logCcdf(x), gam.logCdf(1.0 / x), kTol);
    }
}

// ---------------------------------------------------------------------------
// Gamma / InverseGamma parameter-derivative packs.
//
// Reference values: tests/data/distributions/gamma_paramderiv_reference.csv.
// Generated with .venv/bin/python (mpmath 1.4.1, mp.dps = 60) from
//   F(a, r, x) = mpmath.gammainc(a, 0, r*x, regularized=True)
// with `mpmath.diff` for the shape entries and the closed-form
// first/second y-derivatives for the rate/x entries; 20 significant digits
// in the file. The grid covers both kernel regimes (y = rate*x below and
// above a + 1) for shape in {0.5, 1, 2, 7.3} and rate in {0.2, 1, 3}.
// ---------------------------------------------------------------------------

TEST(GammaParameterDerivatives, referenceCsvMatchesMpmath) {
    const std::filesystem::path relative = "distributions/gamma_paramderiv_reference.csv";
    SKIP_UNLESS_ASSET(std::filesystem::is_regular_file(testDataDir() / relative),
                      "gamma parameter-derivative reference csv absent");
    std::ifstream in(RequireDataFile(relative));
    SKIP_UNLESS_ASSET(in.good(), "gamma parameter-derivative reference csv unreadable");

    std::string line;
    std::getline(in, line); // header
    int rows = 0;
    while (std::getline(in, line)) {
        if (line.empty()) {
            continue;
        }
        std::stringstream fields(line);
        double f[13] = {};
        int n = 0;
        std::string token;
        while (n < 13 && std::getline(fields, token, ',')) {
            f[n++] = std::stod(token);
        }
        ASSERT_EQ(n, 13) << "malformed reference row: " << line;
        const double shape = f[0];
        const double rate = f[1];
        const double x = f[2];
        const auto pack = gammaCdfGradHess(shape, rate, x);
        const std::string label = "shape=" + std::to_string(shape) +
                                  " rate=" + std::to_string(rate) + " x=" + std::to_string(x);
        CHECK_CLOSE_REL(label + " value", pack.value, f[3], 1e-10);
        CHECK_CLOSE_REL(label + " gShape", pack.gShape, f[4], 1e-10);
        CHECK_CLOSE_REL(label + " gRate", pack.gRate, f[5], 1e-10);
        CHECK_CLOSE_REL(label + " gx", pack.gx, f[6], 1e-10);
        CHECK_CLOSE_REL(label + " hShapeShape", pack.hShapeShape, f[7], 1e-9);
        CHECK_CLOSE_REL(label + " hShapeRate", pack.hShapeRate, f[8], 1e-9);
        CHECK_CLOSE_REL(label + " hShapeX", pack.hShapeX, f[9], 1e-9);
        CHECK_CLOSE_REL(label + " hRateRate", pack.hRateRate, f[10], 1e-9);
        CHECK_CLOSE_REL(label + " hRateX", pack.hRateX, f[11], 1e-9);
        CHECK_CLOSE_REL(label + " hxx", pack.hxx, f[12], 1e-9);
        ++rows;
    }
    EXPECT_GT(rows, 0);
}

TEST(GammaParameterDerivatives, xEntriesReproduceExistingPacksBitwise) {
    for (double shape : {0.5, 1.0, 2.0, 7.3}) {
        for (double rate : {0.2, 1.0, 3.0}) {
            for (double x : {1e-6, 0.02, 0.5, 1.0, 2.0, 9.0, 100.0}) {
                const std::string label = "shape=" + std::to_string(shape) +
                                          " rate=" + std::to_string(rate) +
                                          " x=" + std::to_string(x);
                const auto cdf = gammaCdfGradHess(shape, rate, x);
                const auto cdfOld = gammaCdfD1D2(shape, rate, x);
                CHECK_CLOSE(label + " cdf value", cdf.value, cdfOld.value, 0.0);
                CHECK_CLOSE(label + " cdf gx", cdf.gx, cdfOld.d1, 0.0);
                CHECK_CLOSE(label + " cdf hxx", cdf.hxx, cdfOld.d2, 0.0);
                const auto pdf = gammaPdfGradHess(shape, rate, x);
                const auto pdfOld = gammaPdfD1D2(shape, rate, x);
                CHECK_CLOSE(label + " pdf value", pdf.value, pdfOld.value, 0.0);
                CHECK_CLOSE(label + " pdf gx", pdf.gx, pdfOld.d1, 0.0);
                CHECK_CLOSE(label + " pdf hxx", pdf.hxx, pdfOld.d2, 0.0);
            }
        }
    }
}

TEST(GammaParameterDerivatives, secondPartialsMatchFiniteDifferences) {
    for (double shape : {0.7, 2.5}) {
        for (double rate : {0.4, 1.7}) {
            for (double factor : {0.4, 2.0}) {
                const double x = factor * (shape + 1.0) / rate;
                const auto p = gammaCdfGradHess(shape, rate, x);
                const auto d = gammaPdfGradHess(shape, rate, x);
                const double ha = 1e-5 * shape;
                const double hr = 1e-5 * rate;
                const double hx = 1e-5 * x;
                const std::string label = "shape=" + std::to_string(shape) +
                                          " rate=" + std::to_string(rate) +
                                          " x=" + std::to_string(x);
                CHECK_CLOSE_REL(label + " hShapeShape vs FD", p.hShapeShape,
                                (gammaCdfGradHess(shape + ha, rate, x).gShape -
                                 gammaCdfGradHess(shape - ha, rate, x).gShape) /
                                    (2.0 * ha),
                                1e-5, 1e-6);
                CHECK_CLOSE_REL(label + " hShapeRate vs FD", p.hShapeRate,
                                (gammaCdfGradHess(shape, rate + hr, x).gShape -
                                 gammaCdfGradHess(shape, rate - hr, x).gShape) /
                                    (2.0 * hr),
                                1e-5, 1e-6);
                CHECK_CLOSE_REL(label + " hShapeX vs FD", p.hShapeX,
                                (gammaCdfGradHess(shape, rate, x + hx).gShape -
                                 gammaCdfGradHess(shape, rate, x - hx).gShape) /
                                    (2.0 * hx),
                                1e-5, 1e-6);
                CHECK_CLOSE_REL(label + " hRateRate vs FD", p.hRateRate,
                                (gammaCdfGradHess(shape, rate + hr, x).gRate -
                                 gammaCdfGradHess(shape, rate - hr, x).gRate) /
                                    (2.0 * hr),
                                1e-5, 1e-6);
                CHECK_CLOSE_REL(label + " hRateX vs FD", p.hRateX,
                                (gammaCdfGradHess(shape, rate, x + hx).gRate -
                                 gammaCdfGradHess(shape, rate, x - hx).gRate) /
                                    (2.0 * hx),
                                1e-5, 1e-6);
                CHECK_CLOSE_REL(label + " hxx vs FD", p.hxx,
                                (gammaCdfGradHess(shape, rate, x + hx).gx -
                                 gammaCdfGradHess(shape, rate, x - hx).gx) /
                                    (2.0 * hx),
                                1e-5, 1e-6);
                CHECK_CLOSE_REL(label + " pdf gShape vs FD", d.gShape,
                                (gammaPdfGradHess(shape + ha, rate, x).value -
                                 gammaPdfGradHess(shape - ha, rate, x).value) /
                                    (2.0 * ha),
                                1e-5, 1e-6);
                CHECK_CLOSE_REL(label + " pdf hxx vs FD", d.hxx,
                                (gammaPdfGradHess(shape, rate, x + hx).gx -
                                 gammaPdfGradHess(shape, rate, x - hx).gx) /
                                    (2.0 * hx),
                                1e-5, 1e-6);
            }
        }
    }
}

TEST(GammaParameterDerivatives, edgeGridCoversSeriesAndContinuedFraction) {
    for (double shape : {0.5, 1.0, 2.0, 7.3}) {
        for (double rate : {0.2, 1.0, 3.0}) {
            const struct {
                double x;
                bool series;
                const char* name;
            } cases[] = {
                {1e-8 / rate, true, "near-zero"},
                {0.5 * (shape + 1.0) / rate, true, "series"},
                {1.5 * (shape + 1.0) / rate, false, "cf"},
                {(shape + 60.0) / rate, false, "large-x"},
            };
            for (const auto& c : cases) {
                const std::string label = std::string(c.name) + " shape=" + std::to_string(shape) +
                                          " rate=" + std::to_string(rate);
                if (c.series) {
                    EXPECT_LT(rate * c.x, shape + 1.0);
                } else {
                    EXPECT_GE(rate * c.x, shape + 1.0);
                }
                const auto p = gammaCdfGradHess(shape, rate, c.x);
                const auto d = gammaPdfGradHess(shape, rate, c.x);
                const Gamma<double> g{shape, rate};
                EXPECT_GE(p.value, 0.0);
                EXPECT_LE(p.value, 1.0);
                for (double e :
                     {p.gShape, p.gRate, p.gx, p.hShapeShape, p.hShapeRate, p.hShapeX, p.hRateRate,
                      p.hRateX, p.hxx, d.value, d.gShape, d.gRate, d.gx, d.hShapeShape,
                      d.hShapeRate, d.hShapeX, d.hRateRate, d.hRateX, d.hxx}) {
                    EXPECT_TRUE(std::isfinite(e)) << label;
                }
                CHECK_CLOSE_REL(label + " gx = density", p.gx, g.pdf(c.x), 1e-13);
                CHECK_CLOSE_REL(label + " gRate = (x/rate) gx", p.gRate, (c.x / rate) * p.gx,
                                1e-12);
                CHECK_CLOSE_REL(label + " hRateRate = (x/rate)^2 hxx", p.hRateRate,
                                (c.x / rate) * (c.x / rate) * p.hxx, 1e-12);
                CHECK_CLOSE_REL(label + " cdf gx = pdf value", p.gx, d.value, 1e-12);
                CHECK_CLOSE_REL(label + " cdf hxx = pdf gx", p.hxx, d.gx, 1e-12);
                CHECK_CLOSE_REL(label + " cdf hShapeX = pdf gShape", p.hShapeX, d.gShape, 1e-11);
                CHECK_CLOSE_REL(label + " cdf hRateX = pdf gRate", p.hRateX, d.gRate, 1e-11);
            }
        }
    }
}

TEST(GammaParameterDerivatives, edgesAndInvalidInputs) {
    const auto cdfZero = gammaCdfGradHess(2.0, 1.0, 0.0);
    const auto pdfZero = gammaPdfGradHess(2.0, 1.0, 0.0);
    EXPECT_TRUE(std::isnan(cdfZero.value));
    EXPECT_TRUE(std::isnan(pdfZero.value));
    EXPECT_TRUE(std::isnan(gammaCdfGradHess(-1.0, 1.0, 1.0).value));
    EXPECT_TRUE(std::isnan(gammaPdfGradHess(1.0, -1.0, 1.0).value));
    EXPECT_TRUE(
        std::isnan(gammaCdfGradHess(1.0, 1.0, std::numeric_limits<double>::quiet_NaN()).value));

    const auto cdfNeg = gammaCdfGradHess(2.5, 1.3, -1.0);
    const auto pdfNeg = gammaPdfGradHess(2.5, 1.3, -1.0);
    EXPECT_EQ(cdfNeg.value, 0.0);
    EXPECT_EQ(cdfNeg.gShape, 0.0);
    EXPECT_EQ(cdfNeg.gRate, 0.0);
    EXPECT_EQ(cdfNeg.gx, 0.0);
    EXPECT_EQ(cdfNeg.hShapeShape, 0.0);
    EXPECT_EQ(cdfNeg.hShapeRate, 0.0);
    EXPECT_EQ(cdfNeg.hShapeX, 0.0);
    EXPECT_EQ(cdfNeg.hRateRate, 0.0);
    EXPECT_EQ(cdfNeg.hRateX, 0.0);
    EXPECT_EQ(cdfNeg.hxx, 0.0);
    EXPECT_EQ(pdfNeg.value, 0.0);
    EXPECT_EQ(pdfNeg.gShape, 0.0);
    EXPECT_EQ(pdfNeg.gx, 0.0);
    EXPECT_EQ(pdfNeg.hxx, 0.0);

    const auto cdfInf = gammaCdfGradHess(2.5, 1.3, kInf);
    CHECK_CLOSE("cdf(+inf) value", cdfInf.value, 1.0, 0.0);
    CHECK_CLOSE("cdf(+inf) gShape", cdfInf.gShape, 0.0, 0.0);
    CHECK_CLOSE("cdf(+inf) gx", cdfInf.gx, 0.0, 0.0);
    CHECK_CLOSE("cdf(+inf) hxx", cdfInf.hxx, 0.0, 0.0);
    const auto pdfInf = gammaPdfGradHess(2.5, 1.3, kInf);
    CHECK_CLOSE("pdf(+inf) value", pdfInf.value, 0.0, 0.0);
    CHECK_CLOSE("pdf(+inf) gShape", pdfInf.gShape, 0.0, 0.0);
}

TEST(InverseGammaParameterDerivatives, chainRuleFromGammaPacks) {
    for (double shape : {0.7, 2.5}) {
        for (double scale : {0.4, 3.0}) {
            const double xSmall = 1e-6 * scale;
            const double xSeries = scale / (0.5 * (shape + 1.0));
            const double xCf = scale / (1.5 * (shape + 1.0));
            const double xLarge = 50.0 * scale / (shape + 1.0);
            for (double x : {xSmall, xSeries, xCf, xLarge}) {
                const auto c = inverseGammaCdfGradHess(shape, scale, x);
                const auto p = inverseGammaPdfGradHess(shape, scale, x);
                const auto g = gammaCdfGradHess(shape, scale, 1.0 / x);
                const auto gp = gammaPdfGradHess(shape, scale, 1.0 / x);
                const InverseGamma<double> ig{shape, scale};
                const double x2 = x * x;
                const double x3 = x2 * x;
                const double x4 = x2 * x2;
                const double x5 = x4 * x;
                const double x6 = x3 * x3;
                const std::string label = "shape=" + std::to_string(shape) +
                                          " scale=" + std::to_string(scale) +
                                          " x=" + std::to_string(x);
                CHECK_CLOSE(label + " cdf value vs direct", c.value, ig.cdf(x), 1e-13);
                CHECK_CLOSE_REL(label + " pdf value vs direct", p.value, ig.pdf(x), 1e-13);
                CHECK_CLOSE(label + " cdf value = 1 - G(1/x)", c.value, 1.0 - g.value, 1e-15);
                CHECK_CLOSE(label + " gShape", c.gShape, -g.gShape, 0.0);
                CHECK_CLOSE(label + " gScale", c.gScale, -g.gRate, 0.0);
                CHECK_CLOSE(label + " gx", c.gx, g.gx / x2, 0.0);
                CHECK_CLOSE(label + " hShapeShape", c.hShapeShape, -g.hShapeShape, 0.0);
                CHECK_CLOSE(label + " hShapeScale", c.hShapeScale, -g.hShapeRate, 0.0);
                CHECK_CLOSE(label + " hShapeX", c.hShapeX, g.hShapeX / x2, 0.0);
                CHECK_CLOSE(label + " hScaleScale", c.hScaleScale, -g.hRateRate, 0.0);
                CHECK_CLOSE(label + " hScaleX", c.hScaleX, g.hRateX / x2, 0.0);
                CHECK_CLOSE(label + " hxx", c.hxx, -g.hxx / x4 - 2.0 * g.gx / x3, 1e-15);
                CHECK_CLOSE(label + " pdf value = g(1/x)/x^2", p.value, gp.value / x2, 1e-14);
                CHECK_CLOSE(label + " pdf gShape", p.gShape, gp.gShape / x2, 1e-14);
                CHECK_CLOSE(label + " pdf gScale", p.gScale, gp.gRate / x2, 1e-14);
                CHECK_CLOSE(label + " pdf gx", p.gx, -gp.gx / x4 - 2.0 * gp.value / x3, 1e-14);
                CHECK_CLOSE_REL(label + " pdf hxx", p.hxx,
                                gp.hxx / x6 + 6.0 * gp.gx / x5 + 6.0 * gp.value / x4, 1e-13);
            }
        }
    }
}

TEST(InverseGammaParameterDerivatives, secondPartialsMatchFiniteDifferences) {
    for (double shape : {0.7, 2.5}) {
        for (double scale : {0.4, 3.0}) {
            for (double factor : {0.5, 2.5}) {
                const double x = scale / (factor * (shape + 1.0));
                const auto c = inverseGammaCdfGradHess(shape, scale, x);
                const double ha = 1e-5 * shape;
                const double hs = 1e-5 * scale;
                const double hx = 1e-5 * x;
                const std::string label = "shape=" + std::to_string(shape) +
                                          " scale=" + std::to_string(scale) +
                                          " x=" + std::to_string(x);
                CHECK_CLOSE_REL(label + " hShapeShape vs FD", c.hShapeShape,
                                (inverseGammaCdfGradHess(shape + ha, scale, x).gShape -
                                 inverseGammaCdfGradHess(shape - ha, scale, x).gShape) /
                                    (2.0 * ha),
                                1e-5, 1e-6);
                CHECK_CLOSE_REL(label + " hScaleScale vs FD", c.hScaleScale,
                                (inverseGammaCdfGradHess(shape, scale + hs, x).gScale -
                                 inverseGammaCdfGradHess(shape, scale - hs, x).gScale) /
                                    (2.0 * hs),
                                1e-5, 1e-6);
                CHECK_CLOSE_REL(label + " hShapeX vs FD", c.hShapeX,
                                (inverseGammaCdfGradHess(shape, scale, x + hx).gShape -
                                 inverseGammaCdfGradHess(shape, scale, x - hx).gShape) /
                                    (2.0 * hx),
                                1e-5, 1e-6);
                CHECK_CLOSE_REL(label + " hxx vs FD", c.hxx,
                                (inverseGammaCdfGradHess(shape, scale, x + hx).gx -
                                 inverseGammaCdfGradHess(shape, scale, x - hx).gx) /
                                    (2.0 * hx),
                                1e-5, 1e-6);
            }
        }
    }
}

TEST(InverseGammaParameterDerivatives, edgesAndInvalidInputs) {
    const auto cdfNeg = inverseGammaCdfGradHess(2.0, 1.0, -3.0);
    const auto pdfNeg = inverseGammaPdfGradHess(2.0, 1.0, -3.0);
    EXPECT_EQ(cdfNeg.value, 0.0);
    EXPECT_EQ(cdfNeg.gx, 0.0);
    EXPECT_EQ(cdfNeg.hxx, 0.0);
    EXPECT_EQ(pdfNeg.value, 0.0);
    EXPECT_EQ(pdfNeg.gx, 0.0);
    EXPECT_EQ(pdfNeg.hxx, 0.0);
    EXPECT_TRUE(std::isnan(inverseGammaCdfGradHess(2.0, 1.0, 0.0).value));
    EXPECT_TRUE(std::isnan(inverseGammaPdfGradHess(2.0, 1.0, 0.0).value));
    EXPECT_TRUE(std::isnan(inverseGammaCdfGradHess(-1.0, 1.0, 1.0).value));
    const auto cdfInf = inverseGammaCdfGradHess(2.0, 1.0, kInf);
    CHECK_CLOSE("cdf(+inf) value", cdfInf.value, 1.0, 0.0);
    CHECK_CLOSE("cdf(+inf) gShape", cdfInf.gShape, 0.0, 0.0);
    const auto pdfInf = inverseGammaPdfGradHess(2.0, 1.0, kInf);
    CHECK_CLOSE("pdf(+inf) value", pdfInf.value, 0.0, 0.0);
    CHECK_CLOSE("pdf(+inf) gx", pdfInf.gx, 0.0, 0.0);
}

// ---------------------------------------------------------------------------
// Incomplete gamma at large shape. The series/CF budget scales with `sqrt(a)`
// (`9 sqrt(a) + 100` above a = 1e6) and the log prefactor uses the Stirling
// form, which keeps `P(a,a)` at ~1e-13 relative accuracy up to the documented
// cap a <= 1e8. References: 80-digit Decimal power series with the Stirling
// lgamma series, cross-checked against Boost `cpp_bin_float_100` `gamma_p`
// (identical to 20 digits at a >= 1e6).
// ---------------------------------------------------------------------------

TEST(IncompleteGamma, largeShapeMatchesHighPrecision) {
    using quantape::math::detail::gammaLogP;
    using quantape::math::detail::gammaLogQ;
    using quantape::math::detail::gammaP;
    using quantape::math::detail::gammaQ;

    CHECK_CLOSE_REL("P(1e6,1e6)", gammaP(1e6, 1e6), 0.500132980760872591244, 1e-12);
    CHECK_CLOSE_REL("P(1e7,1e7)", gammaP(1e7, 1e7), 0.500042052208723698334, 1e-12);
    CHECK_CLOSE_REL("P(1e8,1e8)", gammaP(1e8, 1e8), 0.500013298076014119871, 1e-11);

    // Continued-fraction branch (x >= a + 1) and its complement.
    CHECK_CLOSE_REL("Q(1e6,1e6+10)", gammaQ(1e6, 1e6 + 10.0), 0.495877683203090427170, 1e-12);
    CHECK_CLOSE_REL("Q(1e7,1e7+100)", gammaQ(1e7, 1e7 + 100.0), 0.487344450632985278707, 1e-12);
    CHECK_CLOSE_REL("P(1e8,1e8+1000)", gammaP(1e8, 1e8 + 1000.0), 0.539840936679227943608, 1e-11);

    // log forms keep the relative accuracy of the value (references are the
    // logarithms of the high-precision probabilities).
    CHECK_CLOSE_REL("logP(1e6,1e6)", gammaLogP(1e6, 1e6), -0.692881254399695925555, 1e-12);
    CHECK_CLOSE_REL("logQ(1e6,1e6)", gammaLogQ(1e6, 1e6), -0.693413177455728244051, 1e-12);
    CHECK_CLOSE_REL("logP(1e8,1e8)", gammaLogP(1e8, 1e8), -0.693120584761588450178, 1e-11);
    CHECK_CLOSE_REL("logQ(1e8,1e8)", gammaLogQ(1e8, 1e8), -0.693173777065657471616, 1e-11);
    CHECK_CLOSE_REL("logQ(1e8,1e8+1000)", gammaLogQ(1e8, 1e8 + 1000.0), -0.776183059442754994481,
                    1e-11);

    // Deep lower tail: P(1e7, 0.9e7) = 4.75557e-23284, finite only in log
    // space. The Stirling prefactor keeps the exponent's relative accuracy.
    CHECK_CLOSE_REL("logP(1e7,0.9e7)", gammaLogP(1e7, 0.9e7), -53611.8319885368239050, 1e-10);
    CHECK_CLOSE("logQ(1e7,0.9e7)", gammaLogQ(1e7, 0.9e7), 0.0, 1e-12);
}
