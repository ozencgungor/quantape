// test_chi_square_poisson.cpp — ChiSquare / Poisson laws: closed forms, quadrature,
// derivatives, parameter-derivative packs.

#include "quantape/math/Distributions/ChiSquare.h"
#include "quantape/math/Distributions/Poisson.h"
#include "quantape/math/Integrals/GaussLegendre.h"
#include "quantape/math/Integrals/GaussianQuadrature.h"

#include <cmath>
#include <limits>
#include <string>

#include "support/GtestSupport.h"

using quantape::math::ChiSquare;
using quantape::math::GaussLegendre;
using quantape::math::GaussLegendreIntegrator;
using quantape::math::Poisson;
using quantape::math::detail::chiSquareCdfD1D2;
using quantape::math::detail::chiSquareCdfGradHess;
using quantape::math::detail::chiSquarePdfD1D2;
using quantape::math::detail::chiSquarePdfGradHess;
using quantape::math::detail::gammaCdfGradHess;
using quantape::math::detail::gammaPdfGradHess;
using quantape::math::detail::poissonCdfD1D2;
using quantape::math::detail::poissonPmfD1D2;
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

template <class P>
void checkMonotonePmf(const P& dist, int lo, int hi) {
    double prev = dist.pmf(lo);
    for (int k = lo + 1; k <= hi; ++k) {
        SCOPED_TRACE(k);
        const double cur = dist.pmf(k);
        EXPECT_LE(prev, cur);
        prev = cur;
    }
}

} // namespace

TEST(ChiSquareDistribution, cdfMatchesClosedForms) {
    const ChiSquare<double> c2{2.0};
    for (double x : {0.0, 0.5, 2.0, 8.0, 40.0}) {
        CHECK_CLOSE("df=2 cdf vs 1-exp(-x/2)", c2.cdf(x), -std::expm1(-x / 2.0), kTol);
        CHECK_CLOSE("df=2 ccdf vs exp(-x/2)", c2.ccdf(x), std::exp(-x / 2.0), kTol);
    }
    CHECK_CLOSE("df=2 pdf(0) = 1/2", c2.pdf(0.0), 0.5, kTol);
    const ChiSquare<double> c1{1.0};
    for (double x : {0.01, 0.5, 2.0, 9.0}) {
        CHECK_CLOSE("df=1 cdf vs erf", c1.cdf(x), std::erf(std::sqrt(x / 2.0)), kTol);
        CHECK_CLOSE("df=1 ccdf vs erfc", c1.ccdf(x), std::erfc(std::sqrt(x / 2.0)), kTol);
        CHECK_CLOSE("df=1 logCdf = log(cdf)", c1.logCdf(x), std::log(c1.cdf(x)), kTol);
    }
    EXPECT_TRUE(std::isinf(c1.pdf(0.0)) && c1.pdf(0.0) > 0.0);
    CHECK_CLOSE("df=4 pdf(0) = 0", ChiSquare<double>{4.0}.pdf(0.0), 0.0, kTol);
}

TEST(ChiSquareDistribution, cdfMatchesQuadratureOfPdf) {
    const ChiSquare<double> c4{4.0};
    const GaussLegendreIntegrator<double> gauss20(20);
    const GaussLegendre rule(160);
    for (double x : {0.5, 2.0, 8.0, 25.0}) {
        const auto density = [&c4](double t) { return c4.pdf(t); };
        CHECK_CLOSE("GL20 integral of pdf", gauss20(density, 0.0, x), c4.cdf(x), kTol);
        CHECK_CLOSE("GL160 integral of pdf", rule.integrate(density, 0.0, x), c4.cdf(x), kTol);
    }
}

TEST(ChiSquareDistribution, derivativePacksMatchFiniteDifferences) {
    const auto pdfPack = [](double x) { return chiSquarePdfD1D2(4.0, x); };
    const auto cdfPack = [](double x) { return chiSquareCdfD1D2(4.0, x); };
    const auto pdfValue = [&pdfPack](double x) { return pdfPack(x).value; };
    const auto cdfValue = [&cdfPack](double x) { return cdfPack(x).value; };
    const ChiSquare<double> c4{4.0};

    const double xp = 12.0;
    const double hp = 1e-5 * xp;
    const ValueD1D2 packP = pdfPack(xp);
    CHECK_CLOSE("pdf pack value", packP.value, c4.pdf(xp), kTol);
    CHECK_CLOSE("pdf pack d1 vs FD", packP.d1, centralD1(pdfValue, xp, hp), 1e-7);
    CHECK_CLOSE("pdf pack d2 vs FD", packP.d2, centralD2(pdfValue, xp, hp), 1e-7);

    const double xc = 0.2;
    const double hc = 1e-5;
    const ValueD1D2 packC = cdfPack(xc);
    CHECK_CLOSE("cdf pack value", packC.value, c4.cdf(xc), kTol);
    CHECK_CLOSE("cdf pack d1 vs FD", packC.d1, centralD1(cdfValue, xc, hc), 1e-7);
    CHECK_CLOSE("cdf pack d2 vs FD", packC.d2, centralD2(cdfValue, xc, hc), 1e-7);
}

TEST(ChiSquareDistribution, normalizationMonotoneAndEdges) {
    const ChiSquare<double> c4{4.0};
    CHECK_CLOSE("cdf(+inf)", c4.cdf(kInf), 1.0, kTol);
    CHECK_CLOSE("ccdf(+inf)", c4.ccdf(kInf), 0.0, kTol);
    CHECK_CLOSE("logCdf(+inf)", c4.logCdf(kInf), 0.0, kTol);
    CHECK_CLOSE("logCcdf(+inf)", c4.logCcdf(kInf), -kInf, 0.0);
    CHECK_CLOSE("cdf(-1)", c4.cdf(-1.0), 0.0, kTol);
    CHECK_CLOSE("pdf(-1)", c4.pdf(-1.0), 0.0, kTol);
    CHECK_CLOSE("ccdf(-1)", c4.ccdf(-1.0), 1.0, kTol);
    checkMonotoneCdf(c4, 0.0, 30.0, 30);
    for (double x : {0.5, 2.0, 8.0, 25.0}) {
        CHECK_CLOSE("cdf + ccdf = 1", c4.cdf(x) + c4.ccdf(x), 1.0, kTol);
    }
    EXPECT_TRUE(std::isnan(ChiSquare<double>{0.0}.cdf(1.0)));
}

TEST(ChiSquareDistribution, quantileRoundTrips) {
    const ChiSquare<double> c1{1.0};
    for (double p : {1e-8, 1e-3, 0.25, 0.5, 0.9, 0.999}) {
        CHECK_CLOSE_REL("df=1 cdf(inv(p)) = p", c1.cdf(c1.inv(p)), p, 1e-10);
    }
    const ChiSquare<double> c5{5.0};
    for (double p : {1e-6, 0.01, 0.5, 0.99}) {
        CHECK_CLOSE_REL("df=5 cdf(inv(p)) = p", c5.cdf(c5.inv(p)), p, 1e-10);
    }
    for (double x0 : {0.2, 1.0, 3.0, 8.0}) {
        CHECK_CLOSE_REL("df=5 inv(cdf(x)) = x", c5.inv(c5.cdf(x0)), x0, 1e-10);
    }
    CHECK_CLOSE("inv(0) = 0", c5.inv(0.0), 0.0, kTol);
    CHECK_CLOSE("inv(1) = +inf", c5.inv(1.0), kInf, 0.0);
}

TEST(PoissonDistribution, pmfAndCdfMatchPartialSums) {
    const Poisson<double> p{3.0};
    double ref = 0.0;
    double term = std::exp(-3.0);
    for (int k = 0; k <= 12; ++k) {
        if (k > 0) {
            term *= 3.0 / static_cast<double>(k);
        }
        ref += term;
        const double direct =
            std::pow(3.0, k) * std::exp(-3.0) / std::tgamma(static_cast<double>(k) + 1.0);
        CHECK_CLOSE("pmf vs direct term", p.pmf(k), direct, kTol);
        CHECK_CLOSE("cdf vs partial sum", p.cdf(k), ref, kTol);
        CHECK_CLOSE("ccdf = 1 - cdf", p.ccdf(k), 1.0 - p.cdf(k), kTol);
        CHECK_CLOSE("logPmf round trip", p.pmf(k), std::exp(p.logPmf(k)), kTol);
    }
    CHECK_CLOSE("cdf(-1) = 0", p.cdf(-1), 0.0, kTol);
    CHECK_CLOSE("ccdf(-1) = 1", p.ccdf(-1), 1.0, kTol);
    CHECK_CLOSE("pmf(-1) = 0", p.pmf(-1), 0.0, kTol);
    CHECK_CLOSE("logPmf(-1) = -inf", p.logPmf(-1), -kInf, 0.0);
}

TEST(PoissonDistribution, pmfSumsToOneAndRecurrence) {
    const Poisson<double> p{4.0};
    double total = 0.0;
    for (int k = 0; k <= 100; ++k) {
        total += p.pmf(k);
    }
    CHECK_CLOSE("total mass", total, 1.0, 1e-13);
    for (int k = 0; k < 10; ++k) {
        CHECK_CLOSE("recurrence pmf(k+1) = pmf(k) lambda/(k+1)", p.pmf(k + 1),
                    p.pmf(k) * 4.0 / static_cast<double>(k + 1), 1e-14);
    }
    CHECK_CLOSE("logPmf far tail matches lgamma form", p.logPmf(400),
                400.0 * std::log(4.0) - 4.0 - std::lgamma(401.0), 1e-10);
    CHECK_CLOSE("pmf underflows to 0 far in tail", p.pmf(400), 0.0, 0.0);
    checkMonotonePmf(p, 0, 4);
}

TEST(PoissonDistribution, edgeCases) {
    const Poisson<double> zero{0.0};
    CHECK_CLOSE("lambda=0 pmf(0)", zero.pmf(0), 1.0, kTol);
    CHECK_CLOSE("lambda=0 pmf(3)", zero.pmf(3), 0.0, kTol);
    CHECK_CLOSE("lambda=0 cdf(0)", zero.cdf(0), 1.0, kTol);
    CHECK_CLOSE("lambda=0 ccdf(0)", zero.ccdf(0), 0.0, kTol);
    CHECK_CLOSE("lambda=0 logPmf(0)", zero.logPmf(0), 0.0, kTol);
    const Poisson<double> neg{-1.0};
    EXPECT_TRUE(std::isnan(neg.pmf(1)));
    EXPECT_TRUE(std::isnan(neg.cdf(1)));
    const Poisson<double> p{2.0};
    CHECK_CLOSE("cdf(0) = e^-lambda", p.cdf(0), std::exp(-2.0), kTol);
    CHECK_CLOSE("cdf(1) = e^-lambda (1+lambda)", p.cdf(1), std::exp(-2.0) * 3.0, kTol);
}

TEST(PoissonDistribution, derivativePacksMatchFiniteDifferences) {
    const auto pmfPack = [](double lambda) { return poissonPmfD1D2(lambda, 8); };
    const auto cdfPack = [](double lambda) { return poissonCdfD1D2(lambda, 0); };
    const auto pmfValue = [&pmfPack](double lambda) { return pmfPack(lambda).value; };
    const auto cdfValue = [&cdfPack](double lambda) { return cdfPack(lambda).value; };

    const double lambdaP = 2.0;
    const double hP = 1e-5 * lambdaP;
    const ValueD1D2 packP = pmfPack(lambdaP);
    const Poisson<double> p{2.0};
    CHECK_CLOSE("pmf pack value", packP.value, p.pmf(8), kTol);
    CHECK_CLOSE("pmf pack d1 vs FD", packP.d1, centralD1(pmfValue, lambdaP, hP), 1e-7);
    CHECK_CLOSE("pmf pack d2 vs FD", packP.d2, centralD2(pmfValue, lambdaP, hP), 1e-7);

    const double lambdaC = 5.0;
    const double hC = 1e-5 * lambdaC;
    const ValueD1D2 packC = cdfPack(lambdaC);
    const Poisson<double> q{5.0};
    CHECK_CLOSE("cdf pack value", packC.value, q.cdf(0), kTol);
    CHECK_CLOSE("cdf pack d1 vs FD", packC.d1, centralD1(cdfValue, lambdaC, hC), 1e-7);
    CHECK_CLOSE("cdf pack d2 vs FD", packC.d2, centralD2(cdfValue, lambdaC, hC), 1e-7);
}

// ---------------------------------------------------------------------------
// ChiSquare parameter-derivative packs. The pack is the Gamma pack at
// (shape = df/2, rate = 1/2) with d/ddf = (1/2) d/dshape, so the tests check
// the scaling identities against `gamma*GradHess`, bitwise agreement of the
// x entries with the existing `chiSquare*D1D2` packs, FD cross-checks and a
// handful of hardcoded mpmath values (mp.dps = 60, generator named below).
// ---------------------------------------------------------------------------

TEST(ChiSquareParameterDerivatives, matchesGammaPacksAndExistingPacks) {
    for (double df : {0.5, 1.0, 2.0, 4.0, 7.3}) {
        for (double x : {0.05, 0.3, 1.0, 2.5, 7.0, 20.0, 60.0}) {
            const std::string label = "df=" + std::to_string(df) + " x=" + std::to_string(x);
            const auto c = chiSquareCdfGradHess(df, x);
            const auto g = gammaCdfGradHess(df / 2.0, 0.5, x);
            CHECK_CLOSE(label + " cdf value", c.value, g.value, 0.0);
            CHECK_CLOSE(label + " gDf", c.gDf, 0.5 * g.gShape, 0.0);
            CHECK_CLOSE(label + " gx", c.gx, g.gx, 0.0);
            CHECK_CLOSE(label + " hDfDf", c.hDfDf, 0.25 * g.hShapeShape, 0.0);
            CHECK_CLOSE(label + " hDfX", c.hDfX, 0.5 * g.hShapeX, 0.0);
            CHECK_CLOSE(label + " hxx", c.hxx, g.hxx, 0.0);
            const auto p = chiSquarePdfGradHess(df, x);
            const auto gp = gammaPdfGradHess(df / 2.0, 0.5, x);
            CHECK_CLOSE(label + " pdf value", p.value, gp.value, 0.0);
            CHECK_CLOSE(label + " pdf gDf", p.gDf, 0.5 * gp.gShape, 0.0);
            CHECK_CLOSE(label + " pdf gx", p.gx, gp.gx, 0.0);
            CHECK_CLOSE(label + " pdf hDfDf", p.hDfDf, 0.25 * gp.hShapeShape, 0.0);
            CHECK_CLOSE(label + " pdf hDfX", p.hDfX, 0.5 * gp.hShapeX, 0.0);
            CHECK_CLOSE(label + " pdf hxx", p.hxx, gp.hxx, 0.0);

            const auto cOld = chiSquareCdfD1D2(df, x);
            const auto pOld = chiSquarePdfD1D2(df, x);
            CHECK_CLOSE(label + " cdf value vs D1D2", c.value, cOld.value, 0.0);
            CHECK_CLOSE(label + " cdf gx vs D1D2", c.gx, cOld.d1, 0.0);
            CHECK_CLOSE(label + " cdf hxx vs D1D2", c.hxx, cOld.d2, 0.0);
            CHECK_CLOSE(label + " pdf value vs D1D2", p.value, pOld.value, 0.0);
            CHECK_CLOSE(label + " pdf gx vs D1D2", p.gx, pOld.d1, 0.0);
            CHECK_CLOSE(label + " pdf hxx vs D1D2", p.hxx, pOld.d2, 0.0);
        }
    }
}

TEST(ChiSquareParameterDerivatives, referenceValuesFromMpmath) {
    // .venv/bin/python, mpmath 1.4.1, mp.dps = 60:
    //   F(df, x) = mpmath.gammainc(df/2, 0, x/2, regularized=True),
    //   value/gDf/gx/hDfDf/hDfX/hxx from mpmath.diff and closed forms.
    const struct {
        double df;
        double x;
        double value;
        double gDf;
        double gx;
        double hDfDf;
        double hDfX;
        double hxx;
    } rows[] = {
        {3.5, 2.2, 0.37918793294589289144, -0.16759517875015874043, 0.19451161029219108328,
         0.043034689087861719035, -0.014798664445690952901, -0.030945028910121314057},
        {0.5, 0.3, 0.66682819768464335922, -0.5881744892821712124, 0.49246606879378133801,
         0.34341211360498141268, 0.57380510080166103021, -1.4773982063813440596},
        {6.0, 7.5, 0.72293155663389269359, -0.097519454134109893913, 0.082679575275032021143,
         -0.010728319143585610856, 0.016493397285318636268, -0.0192919008975074716},
        {1.0, 3.0, 0.91673548333644959815, -0.11509717100868038585, 0.05139344326792309227,
         -0.058400298823340706736, 0.060874894579504735909, -0.03426229551194872818},
    };
    for (const auto& row : rows) {
        const std::string label = "df=" + std::to_string(row.df) + " x=" + std::to_string(row.x);
        const auto c = chiSquareCdfGradHess(row.df, row.x);
        CHECK_CLOSE_REL(label + " value", c.value, row.value, 1e-10);
        CHECK_CLOSE_REL(label + " gDf", c.gDf, row.gDf, 1e-10);
        CHECK_CLOSE_REL(label + " gx", c.gx, row.gx, 1e-10);
        CHECK_CLOSE_REL(label + " hDfDf", c.hDfDf, row.hDfDf, 1e-9);
        CHECK_CLOSE_REL(label + " hDfX", c.hDfX, row.hDfX, 1e-9);
        CHECK_CLOSE_REL(label + " hxx", c.hxx, row.hxx, 1e-9);
    }
}

TEST(ChiSquareParameterDerivatives, finiteDifferenceCrossCheck) {
    for (double df : {1.0, 3.5}) {
        for (double x : {0.4, 2.0, 9.0}) {
            const auto c = chiSquareCdfGradHess(df, x);
            const auto p = chiSquarePdfGradHess(df, x);
            const double hd = 1e-5 * df;
            const double hx = 1e-5 * x;
            const std::string label = "df=" + std::to_string(df) + " x=" + std::to_string(x);
            CHECK_CLOSE_REL(
                label + " hDfDf vs FD", c.hDfDf,
                (chiSquareCdfGradHess(df + hd, x).gDf - chiSquareCdfGradHess(df - hd, x).gDf) /
                    (2.0 * hd),
                1e-5, 1e-6);
            CHECK_CLOSE_REL(
                label + " hDfX vs FD", c.hDfX,
                (chiSquareCdfGradHess(df, x + hx).gDf - chiSquareCdfGradHess(df, x - hx).gDf) /
                    (2.0 * hx),
                1e-5, 1e-6);
            CHECK_CLOSE_REL(
                label + " hxx vs FD", c.hxx,
                (chiSquareCdfGradHess(df, x + hx).gx - chiSquareCdfGradHess(df, x - hx).gx) /
                    (2.0 * hx),
                1e-5, 1e-6);
            CHECK_CLOSE_REL(
                label + " pdf gDf vs FD", p.gDf,
                (chiSquarePdfGradHess(df + hd, x).value - chiSquarePdfGradHess(df - hd, x).value) /
                    (2.0 * hd),
                1e-5, 1e-6);
        }
    }
}

TEST(ChiSquareParameterDerivatives, edgesAndInvalidInputs) {
    EXPECT_TRUE(std::isnan(chiSquareCdfGradHess(0.0, 1.0).value));
    EXPECT_TRUE(std::isnan(chiSquareCdfGradHess(-2.0, 1.0).value));
    EXPECT_TRUE(std::isnan(chiSquarePdfGradHess(4.0, 0.0).value));
    const auto cdfNeg = chiSquareCdfGradHess(4.0, -1.0);
    EXPECT_EQ(cdfNeg.value, 0.0);
    EXPECT_EQ(cdfNeg.gx, 0.0);
    EXPECT_EQ(cdfNeg.hxx, 0.0);
    const auto cdfInf = chiSquareCdfGradHess(4.0, kInf);
    CHECK_CLOSE("cdf(+inf) value", cdfInf.value, 1.0, 0.0);
    CHECK_CLOSE("cdf(+inf) gDf", cdfInf.gDf, 0.0, 0.0);
    const auto pdfInf = chiSquarePdfGradHess(4.0, kInf);
    CHECK_CLOSE("pdf(+inf) value", pdfInf.value, 0.0, 0.0);
}
