// test_noncentral_chi_square.cpp — Bessel I, noncentral chi-square CDF/PDF,
// log-space companions, analytic derivative packs, lambda recurrences, and
// independent mixture/integration/mpmath-reference oracles.
#include "quantape/math/Distributions/ChiSquare.h"
#include "quantape/math/Distributions/NoncentralChiSquare.h"

#include <cmath>
#include <filesystem>
#include <fstream>
#include <limits>
#include <sstream>
#include <string>

#include "support/GtestSupport.h"

namespace {

using quantape::math::detail::besselI;

// Independent composite Simpson integration of f on [a, b].
template <class F>
double simpson(const F& f, double a, double b, int panels) {
    if (panels % 2 == 1) {
        ++panels;
    }
    const double h = (b - a) / panels;
    double sum = f(a) + f(b);
    for (int i = 1; i < panels; ++i) {
        sum += (i % 2 == 1 ? 4.0 : 2.0) * f(a + i * h);
    }
    return sum * h / 3.0;
}

double centralCdf(double df, double x) {
    return quantape::math::ChiSquare<double>{df}.cdf(x);
}

} // namespace

TEST(BesselI, knownValues) {
    using quantape::math::detail::besselI;
    CHECK_CLOSE_REL("I0(1)", besselI(0.0, 1.0), 1.2660658777520084, 1e-13);
    CHECK_CLOSE_REL("I1(1)", besselI(1.0, 1.0), 0.565159103992485, 1e-13);
    CHECK_CLOSE_REL("I0(2)", besselI(0.0, 2.0), 2.2795853023360673, 1e-13);
    CHECK_CLOSE_REL("I0(20) asymptotic", besselI(0.0, 20.0), 4.355828255955353e7, 1e-10);
    // I_{1/2}(z) = sqrt(2/(pi z)) sinh z; and the three-term recurrence
    // I_{nu-1}(z) - I_{nu+1}(z) = (2 nu / z) I_nu(z) at nu=1, z=3.
    CHECK_CLOSE_REL("I(0.5,3) closed form", besselI(0.5, 3.0),
                    std::sqrt(2.0 / (M_PI * 3.0)) * std::sinh(3.0), 1e-11);
    CHECK_CLOSE_REL("I recurrence", besselI(2.0, 3.0),
                    besselI(0.0, 3.0) - (2.0 / 3.0) * besselI(1.0, 3.0), 1e-11);
}

TEST(NoncentralChiSquare, reducesToCentralAtZeroNcp) {
    for (double nu : {0.5, 1.0, 2.0, 3.0, 5.0}) {
        const quantape::math::NoncentralChiSquare<double> nc{nu, 0.0};
        for (double x : {0.25, 1.0, 3.0, 8.0}) {
            CHECK_CLOSE_REL("ncp=0 cdf", nc.cdf(x), centralCdf(nu, x), 1e-13);
            CHECK_CLOSE_REL("ncp=0 pdf", nc.pdf(x), quantape::math::ChiSquare<double>{nu}.pdf(x),
                            1e-12);
        }
    }
}

TEST(NoncentralChiSquare, cdfMatchesPdfIntegral) {
    // The density has an integrable x^(nu/2-1) singularity at 0 for nu<2, so
    // the integral check runs on intervals with a positive lower bound.
    const double cases[4][2] = {{1.0, 1.0}, {3.0, 2.0}, {0.5, 5.0}, {5.0, 20.0}};
    for (const auto& c : cases) {
        const quantape::math::NoncentralChiSquare<double> nc{c[0], c[1]};
        for (const double lo : {0.25, 1.0}) {
            for (const double hi : {1.5, 3.0, 8.0}) {
                if (hi <= lo) {
                    continue;
                }
                const double integrated =
                    simpson([&](double t) { return nc.pdf(t); }, lo, hi, 20000);
                CHECK_CLOSE_REL("cdf difference vs Simpson(pdf)", nc.cdf(hi) - nc.cdf(lo),
                                integrated, 5e-9);
            }
        }
    }
}

TEST(NoncentralChiSquare, independentPoissonMixture) {
    for (double lam : {1.0, 5.0, 20.0}) {
        const quantape::math::NoncentralChiSquare<double> nc{3.0, lam};
        for (double x : {0.5, 2.0, 6.0}) {
            // Direct factorial weights and central CDFs (independent of the header).
            double sum = 0.0;
            double w = std::exp(-0.5 * lam);
            for (int j = 0; j <= 200; ++j) {
                sum += w * centralCdf(3.0 + 2.0 * j, x);
                w *= (0.5 * lam) / (j + 1.0);
            }
            CHECK_CLOSE_REL("mixture check", nc.cdf(x), sum, 1e-12);
        }
    }
}

TEST(NoncentralChiSquare, lambdaRecurrencesMatchFiniteDifferences) {
    const double nu = 3.0;
    const double lam = 4.0;
    const double x = 2.5;
    const double h = 1e-5;
    const double fd1 = (quantape::math::NoncentralChiSquare<double>{nu, lam + h}.cdf(x) -
                        quantape::math::NoncentralChiSquare<double>{nu, lam - h}.cdf(x)) /
                       (2.0 * h);
    CHECK_CLOSE_REL("dF/dlambda", quantape::math::detail::nccsCdfLambdaD1(nu, lam, x), fd1, 1e-7);
    // d2F/dxdlambda = (f(nu+2) - f(nu))/2; verify against the x-derivative of
    // dF/dlambda by central differences.
    const double fdX = (quantape::math::detail::nccsCdfLambdaD1(nu, lam, x + h) -
                        quantape::math::detail::nccsCdfLambdaD1(nu, lam, x - h)) /
                       (2.0 * h);
    CHECK_CLOSE_REL("d2F/dxdlambda", quantape::math::detail::nccsCdfXLambdaD2(nu, lam, x), fdX,
                    1e-6);
}

TEST(NoncentralChiSquare, pdfIntegratesToOneAndEdges) {
    const quantape::math::NoncentralChiSquare<double> nc{3.0, 2.0};
    // x = t^2 substitution smooths the x^(nu/2-1) cusp at 0 so Simpson
    // converges at high order.
    const double mass = simpson(
        [&](double t) {
            const double x = t * t;
            return 2.0 * t * nc.pdf(x);
        },
        0.0, std::sqrt(200.0), 40000);
    CHECK_CLOSE_REL("pdf integrates to 1", mass, 1.0, 1e-10);
    CHECK_CLOSE("cdf(0)", nc.cdf(0.0), 0.0, 0.0);
    CHECK_CLOSE("cdf(inf)", nc.cdf(std::numeric_limits<double>::infinity()), 1.0, 0.0);

    EXPECT_TRUE(std::isinf(quantape::math::NoncentralChiSquare<double>{1.5, 2.0}.pdf(0.0)));
    CHECK_CLOSE("pdf(0), nu=2", (quantape::math::NoncentralChiSquare<double>{2.0, 2.0}).pdf(0.0),
                std::exp(-1.0), 1e-15);
    CHECK_CLOSE("pdf(0), nu=4", (quantape::math::NoncentralChiSquare<double>{4.0, 2.0}).pdf(0.0),
                0.0, 0.0);

    // x = +inf limits are 0 for the density and all derivative entries.
    const auto atInf = (quantape::math::NoncentralChiSquare<double>{3.0, 2.0})
                           .pdfD1D2(std::numeric_limits<double>::infinity());
    CHECK_CLOSE("pdf(inf)", atInf.value, 0.0, 0.0);
    CHECK_CLOSE("pdf'(inf)", atInf.d1, 0.0, 0.0);
    CHECK_CLOSE("pdf''(inf)", atInf.d2, 0.0, 0.0);

    // lambda*x underflow falls back to the central limit (the noncentral
    // correction is O(1e-300)); this used to hit logBesselI(0, 0) = NaN.
    const quantape::math::NoncentralChiSquare<double> tinyNcp{3.0, 1e-300};
    const quantape::math::ChiSquare<double> central3{3.0};
    const auto tinyPack = tinyNcp.pdfD1D2(1e-30);
    const auto centralPack = quantape::math::detail::chiSquarePdfD1D2(3.0, 1e-30);
    CHECK_CLOSE_REL("underflow pdf", tinyNcp.pdf(1e-30), central3.pdf(1e-30), 1e-12);
    CHECK_CLOSE_REL("underflow logPdf", tinyNcp.logPdf(1e-30), central3.logPdf(1e-30), 1e-12);
    CHECK_CLOSE_REL("underflow pack value", tinyPack.value, centralPack.value, 1e-12);
    CHECK_CLOSE_REL("underflow pack d1", tinyPack.d1, centralPack.d1, 1e-12);
    CHECK_CLOSE_REL("underflow pack d2", tinyPack.d2, centralPack.d2, 1e-12);
}

TEST(BesselI, negativeOrdersAndAsymptoticStability) {
    using quantape::math::detail::besselI;
    using quantape::math::detail::logBesselI;
    // Negative integer orders are reflected (I_{-k} = I_k); the series used to
    // divide by Gamma(0) for nu = -1.
    for (double z : {0.05, 0.5, 3.0, 20.0, 60.0}) {
        CHECK_CLOSE_REL("I_-1 = I_1", besselI(-1.0, z), besselI(1.0, z), 1e-13);
        CHECK_CLOSE_REL("I_-2 = I_2", besselI(-2.0, z), besselI(2.0, z), 1e-13);
    }
    // Negative non-integer orders: the sign comes from Gamma(nu+1) and must
    // survive; mpmath 1.4.1 reference values (dps=30).
    CHECK_CLOSE_REL("I(-1.75, sqrt(0.1))", besselI(-1.75, std::sqrt(0.1)), -5.0350638199450551569,
                    1e-11);
    CHECK_CLOSE_REL("I(-2.75, sqrt(0.1))", besselI(-2.75, std::sqrt(0.1)), 56.939044982237554862,
                    1e-11);
    CHECK_CLOSE_REL("I(-1.75, 5)", besselI(-1.75, 5.0), 19.377920827237081554, 1e-11);
    CHECK_CLOSE("log I(-1.75, 13)", logBesselI(-1.75, 13.0), std::log(43742.797371289587079), 1e-9);
    // The asymptotic correction must truncate at its smallest term: the old
    // 1e-17-only loop diverged for z ~ 15..20 (values of order 1e15 for
    // orders ~ -2).
    CHECK_CLOSE_REL("I(-2.75, 14)", besselI(-2.75, 14.0), 97861.662639927024827, 1e-10);
    CHECK_CLOSE_REL("I(0, 16)", besselI(0.0, 16.0), 893446.22792010501707, 1e-10);
    CHECK_CLOSE_REL("I(4, 20)", besselI(4.0, 20.0), 28935060.318764870575, 1e-10);
}

// The analytic x-derivative pack, verified against central differences of the
// density (h = 1e-5 for d1) and of the analytic d1 (h = 1e-4 for d2). The
// high-precision mpmath probe over a wider grid reaches 2.3e-12 (d1) and
// 7.4e-11 (d2); the FD gates below are deliberately looser.
TEST(NoncentralChiSquare, pdfD1D2MatchesFiniteDifferences) {
    for (double nu : {0.5, 1.0, 2.0, 3.0, 5.0, 10.0}) {
        for (double lambda : {0.0, 1.0, 5.0, 20.0, 100.0}) {
            const quantape::math::NoncentralChiSquare<double> nc{nu, lambda};
            for (double x : {0.25, 1.0, 2.5, 10.0}) {
                const auto pack = nc.pdfD1D2(x);
                const double h1 = 1e-5;
                const double fd1 = (nc.pdf(x + h1) - nc.pdf(x - h1)) / (2.0 * h1);
                CHECK_CLOSE("pdf d1 vs FD", pack.d1, fd1,
                            1e-6 * (1.0 + std::fabs(pack.value) + std::fabs(fd1)));

                const double h2 = 1e-4;
                const double fd2 =
                    (quantape::math::detail::noncentralChiSquarePdfPrime(nu, lambda, x + h2) -
                     quantape::math::detail::noncentralChiSquarePdfPrime(nu, lambda, x - h2)) /
                    (2.0 * h2);
                CHECK_CLOSE("pdf d2 vs FD of d1", pack.d2, fd2, 1e-3 * (1.0 + std::fabs(fd2)));

                const double h3 = 1e-3;
                const double fd2b = (nc.pdf(x + h3) - 2.0 * nc.pdf(x) + nc.pdf(x - h3)) / (h3 * h3);
                CHECK_CLOSE("pdf d2 vs 2nd FD", pack.d2, fd2b, 1e-2 * (1.0 + std::fabs(fd2b)));
            }
        }
    }
}

// Analytic lambda / mixed derivatives against central differences at h = 1e-5.
// dF/dlambda and d2F/dxdlambda also chain to the independent mixture CDF.
TEST(NoncentralChiSquare, lambdaAndCrossDerivativesMatchFiniteDifferences) {
    for (double nu : {0.5, 1.0, 2.0, 3.0, 5.0, 10.0}) {
        for (double lambda : {1.0, 5.0, 20.0, 100.0}) {
            for (double x : {0.25, 1.0, 2.5, 10.0}) {
                const double h = 1e-5;
                const double fd1 =
                    (quantape::math::NoncentralChiSquare<double>{nu, lambda + h}.cdf(x) -
                     quantape::math::NoncentralChiSquare<double>{nu, lambda - h}.cdf(x)) /
                    (2.0 * h);
                CHECK_CLOSE("dF/dlambda", quantape::math::detail::nccsCdfLambdaD1(nu, lambda, x),
                            fd1, 1e-6 * (1.0 + std::fabs(fd1)));

                const double fd2 = (quantape::math::detail::nccsCdfLambdaD1(nu, lambda + h, x) -
                                    quantape::math::detail::nccsCdfLambdaD1(nu, lambda - h, x)) /
                                   (2.0 * h);
                CHECK_CLOSE("d2F/dlambda2", quantape::math::detail::nccsCdfLambdaD2(nu, lambda, x),
                            fd2, 1e-5 * (1.0 + std::fabs(fd2)));

                const double fdX = (quantape::math::detail::nccsCdfLambdaD1(nu, lambda, x + h) -
                                    quantape::math::detail::nccsCdfLambdaD1(nu, lambda, x - h)) /
                                   (2.0 * h);
                CHECK_CLOSE("d2F/dxdlambda",
                            quantape::math::detail::nccsCdfXLambdaD2(nu, lambda, x), fdX,
                            1e-5 * (1.0 + std::fabs(fdX)));
            }
        }
    }
}

TEST(NoncentralChiSquare, logPdfAndLogCcdf) {
    // logPdf tracks log(pdf) on the double-representable grid (log-space
    // assembly loses only the ~eps*max(x, lambda) cancellation term).
    for (double nu : {0.5, 2.0, 5.0, 10.0}) {
        for (double lambda : {0.0, 1.0, 20.0, 100.0, 500.0}) {
            const quantape::math::NoncentralChiSquare<double> nc{nu, lambda};
            for (double x : {0.25, 1.0, 5.0, 25.0, 100.0}) {
                const double f = nc.pdf(x);
                if (f > 0.0) {
                    CHECK_CLOSE("logPdf vs log(pdf)", nc.logPdf(x), std::log(f), 1e-12);
                }
                // The double ccdf keeps relative accuracy only until its
                // centred-weight cutoff (deep tails drift by ~1e-8 relative);
                // logCcdf is deep-tail exact (see the mpmath checks below), so
                // the consistency check is limited to the well-resolved bulk.
                const double q = nc.ccdf(x);
                if (q > 1e-6 && q < 1.0) {
                    CHECK_CLOSE("logCcdf vs log(ccdf)", nc.logCcdf(x), std::log(q), 1e-11);
                    CHECK_CLOSE_REL("exp(logCcdf) vs ccdf", std::exp(nc.logCcdf(x)), q, 1e-11);
                }
            }
        }
    }
    // Deep tails: pdf / ccdf underflow but the log-space variants stay finite
    // (mpmath dps=80 references; agreement 1.1e-13 / 1.4e-12).
    const quantape::math::NoncentralChiSquare<double> deepPdf{5.0, 2000.0};
    CHECK_CLOSE("deep pdf underflow", deepPdf.pdf(1.0), 0.0, 0.0);
    CHECK_CLOSE("deep logPdf", deepPdf.logPdf(1.0), -965.01424309350514147, 1e-9);
    const quantape::math::NoncentralChiSquare<double> deepTail{3.0, 500.0};
    CHECK_CLOSE("deep ccdf underflow", deepTail.ccdf(5000.0), 0.0, 0.0);
    CHECK_CLOSE("deep logCcdf", deepTail.logCcdf(5000.0), -1172.5074171978013815, 1e-9);
    const quantape::math::NoncentralChiSquare<double> deepTail2{10.0, 500.0};
    CHECK_CLOSE("deep logCcdf 2", deepTail2.logCcdf(2000.0), -250.91245370116891605, 1e-9);
}

TEST(NoncentralChiSquare, referenceData) {
    const std::filesystem::path relative = "distributions/nccs_reference.csv";
    SKIP_UNLESS_ASSET(std::filesystem::is_regular_file(testDataDir() / relative),
                      "nccs reference csv absent");
    const std::filesystem::path path = RequireDataFile(relative);
    std::ifstream in(path);
    SKIP_UNLESS_ASSET(in.good(), "nccs reference csv unreadable");

    std::string line;
    std::getline(in, line); // header nu,lambda,x,cdf,pdf
    int rows = 0;
    int pdfRows = 0;
    while (std::getline(in, line)) {
        if (line.empty()) {
            continue;
        }
        std::stringstream ss(line);
        std::string nuS;
        std::string lamS;
        std::string xS;
        std::string cdfS;
        std::string pdfS;
        std::getline(ss, nuS, ',');
        std::getline(ss, lamS, ',');
        std::getline(ss, xS, ',');
        std::getline(ss, cdfS, ',');
        std::getline(ss, pdfS, ',');
        const double nu = std::stod(nuS);
        const double lambda = std::stod(lamS);
        const double x = std::stod(xS);
        const double expectedCdf = std::stod(cdfS);
        const std::string label = "nu=" + nuS + " lambda=" + lamS + " x=" + xS;
        const quantape::math::NoncentralChiSquare<double> nc{nu, lambda};
        CHECK_CLOSE_REL(label + " cdf", nc.cdf(x), expectedCdf, 1e-10, 1e-12);
        if (!pdfS.empty()) {
            const double expectedPdf = std::stod(pdfS);
            CHECK_CLOSE_REL(label + " pdf", nc.pdf(x), expectedPdf, 1e-9);
            ++pdfRows;
        }
        ++rows;
    }
    EXPECT_GT(rows, 0);
    EXPECT_GT(pdfRows, 0);
}

TEST(NoncentralChiSquare, monotoneCdf) {
    const quantape::math::NoncentralChiSquare<double> nc{2.5, 3.0};
    double prev = 0.0;
    for (double x = 0.0; x <= 20.0; x += 0.5) {
        const double v = nc.cdf(x);
        EXPECT_TRUE(v >= prev) << "x=" << x;
        prev = v;
    }
}

// ---------------------------------------------------------------------------
// Full (nu, lambda, x) CDF pack (workstream C): the nu partials differentiate
// the Poisson mixture term-by-term through `gammaCdfGradHess`; the lambda/x
// entries reuse the established recurrences and packs.
// ---------------------------------------------------------------------------

// The established entries are reproduced exactly; the nu entries reduce to the
// central gamma pack at lambda = 0.
TEST(NoncentralChiSquare, cdfGradHessMatchesExistingEntries) {
    for (double nu : {0.5, 1.0, 3.5, 10.0}) {
        for (double lambda : {0.0, 1.0, 20.0}) {
            for (double x : {0.25, 1.0, 5.0}) {
                const std::string label = "nu=" + std::to_string(nu) +
                                          " lambda=" + std::to_string(lambda) +
                                          " x=" + std::to_string(x);
                const auto pack =
                    quantape::math::detail::noncentralChiSquareCdfGradHess(nu, lambda, x);
                const auto xd = quantape::math::detail::noncentralChiSquareCdfD1D2(nu, lambda, x);
                CHECK_CLOSE_REL(label + " value", pack.value, xd.value, 1e-15);
                CHECK_CLOSE_REL(label + " gx", pack.gx, xd.d1, 1e-15);
                CHECK_CLOSE_REL(label + " hxx", pack.hxx, xd.d2, 1e-15);
                CHECK_CLOSE_REL(label + " gLambda", pack.gLambda,
                                quantape::math::detail::nccsCdfLambdaD1(nu, lambda, x), 1e-15);
                CHECK_CLOSE_REL(label + " hLambdaLambda", pack.hLambdaLambda,
                                quantape::math::detail::nccsCdfLambdaD2(nu, lambda, x), 1e-15);
                CHECK_CLOSE_REL(label + " hLambdaX", pack.hLambdaX,
                                quantape::math::detail::nccsCdfXLambdaD2(nu, lambda, x), 1e-15);
            }
        }
    }
    // lambda = 0: the nu entries are the central chi-square gamma pack.
    for (double nu : {0.5, 1.0, 3.5, 10.0}) {
        for (double x : {0.25, 1.0, 5.0}) {
            const auto nc = quantape::math::detail::noncentralChiSquareCdfGradHess(nu, 0.0, x);
            const auto g = quantape::math::detail::gammaCdfGradHess(0.5 * nu, 1.0, 0.5 * x);
            CHECK_CLOSE_REL("ncp=0 gNu", nc.gNu, 0.5 * g.gShape, 1e-14);
            CHECK_CLOSE_REL("ncp=0 hNuNu", nc.hNuNu, 0.25 * g.hShapeShape, 1e-13);
            CHECK_CLOSE_REL("ncp=0 hNuX", nc.hNuX, 0.25 * g.hShapeX, 1e-13);
        }
    }
}

// Analytic nu / mixed entries against central differences (h = 1e-5). The
// oracle is deliberately loose; the mpmath CSV below is the reference gate.
TEST(NoncentralChiSquare, cdfGradHessNuMatchesFiniteDifferences) {
    const auto cdf = [](double nu, double lambda, double x) {
        return quantape::math::NoncentralChiSquare<double>{nu, lambda}.cdf(x);
    };
    const auto gNu = [](double nu, double lambda, double x) {
        return quantape::math::detail::noncentralChiSquareCdfGradHess(nu, lambda, x).gNu;
    };
    for (double nu : {0.5, 1.0, 2.0, 3.5, 7.0, 20.0}) {
        for (double lambda : {1.0, 5.0, 20.0}) {
            for (double x : {0.25, 1.0, 2.5, 10.0}) {
                const auto pack =
                    quantape::math::detail::noncentralChiSquareCdfGradHess(nu, lambda, x);
                const std::string label = "nu=" + std::to_string(nu) +
                                          " lambda=" + std::to_string(lambda) +
                                          " x=" + std::to_string(x);
                const double h = 1e-5;
                const double fdNu = (cdf(nu + h, lambda, x) - cdf(nu - h, lambda, x)) / (2.0 * h);
                CHECK_CLOSE(label + " gNu vs FD", pack.gNu, fdNu, 1e-6 * (1.0 + std::fabs(fdNu)));
                const double fdNuNu = (gNu(nu + h, lambda, x) - gNu(nu - h, lambda, x)) / (2.0 * h);
                CHECK_CLOSE(label + " hNuNu vs FD", pack.hNuNu, fdNuNu,
                            1e-5 * (1.0 + std::fabs(fdNuNu)));
                const double fdNuX = (gNu(nu, lambda, x + h) - gNu(nu, lambda, x - h)) / (2.0 * h);
                CHECK_CLOSE(label + " hNuX vs FD", pack.hNuX, fdNuX,
                            1e-5 * (1.0 + std::fabs(fdNuX)));
                const double fdNuLambda =
                    (gNu(nu, lambda + h, x) - gNu(nu, lambda - h, x)) / (2.0 * h);
                CHECK_CLOSE(label + " hNuLambda vs FD", pack.hNuLambda, fdNuLambda,
                            1e-5 * (1.0 + std::fabs(fdNuLambda)));
            }
        }
    }
}

// mpmath reference gates (tests/data/distributions/nccs_nu_reference.csv,
// mpmath 1.4.1 dps=40, formal Poisson mixture differentiated with mp.diff,
// 20 significant digits). Gate 1e-8 relative; measured max errors on the
// committed 30-row grid (nu <= 1000, lambda <= 200): cdf 1.1e-13, gNu 2.7e-13,
// hNuNu 5.1e-13, hNuLambda 8.8e-12, hNuX 2.3e-13. The accuracy ceiling at
// large nu follows the incomplete-gamma shape-jet (workstream A): ~1e-15
// relative at moderate shape, degrading toward ~1e-11 around shape ~1e4, with
// the second-order nu entries additionally subject to signed mixture
// cancellation; the 1e-8 gate is the documented safe envelope.
TEST(NoncentralChiSquare, cdfGradHessNuReferenceCsv) {
    const std::filesystem::path relative = "distributions/nccs_nu_reference.csv";
    SKIP_UNLESS_ASSET(std::filesystem::is_regular_file(testDataDir() / relative),
                      "nccs nu reference csv absent");
    const std::filesystem::path path = RequireDataFile(relative);
    std::ifstream in(path);
    SKIP_UNLESS_ASSET(in.good(), "nccs nu reference csv unreadable");

    std::string line;
    std::getline(in, line); // header nu,lambda,x,cdf,gNu,hNuNu,hNuLambda,hNuX
    int rows = 0;
    while (std::getline(in, line)) {
        if (line.empty() || line[0] == '#') {
            continue;
        }
        std::stringstream ss(line);
        std::string token;
        double f[8] = {};
        int n = 0;
        while (n < 8 && std::getline(ss, token, ',')) {
            f[n++] = std::stod(token);
        }
        ASSERT_EQ(n, 8) << "malformed reference row: " << line;
        const double nu = f[0];
        const double lambda = f[1];
        const double x = f[2];
        const std::string label = "nu=" + std::to_string(nu) + " lambda=" + std::to_string(lambda) +
                                  " x=" + std::to_string(x);
        const auto pack = quantape::math::detail::noncentralChiSquareCdfGradHess(nu, lambda, x);
        CHECK_CLOSE_REL(label + " cdf", pack.value, f[3], 1e-9, 1e-15);
        CHECK_CLOSE_REL(label + " gNu", pack.gNu, f[4], 1e-8, 1e-15);
        CHECK_CLOSE_REL(label + " hNuNu", pack.hNuNu, f[5], 1e-8, 1e-14);
        CHECK_CLOSE_REL(label + " hNuLambda", pack.hNuLambda, f[6], 1e-8, 1e-14);
        CHECK_CLOSE_REL(label + " hNuX", pack.hNuX, f[7], 1e-8, 1e-14);
        ++rows;
    }
    EXPECT_EQ(rows, 30);
}

// ---------------------------------------------------------------------------
// Density pack: value, (nu, lambda, x) gradient and Hessian vs mpmath
// (dps = 40, generator recipe in the workstream probe). The pack uses the
// full-range Poisson weights, so the low-j terms that dominate small x are
// included.
// ---------------------------------------------------------------------------

TEST(NoncentralChiSquare, pdfGradHessMpmathGates) {
    struct Point {
        double nu, lam, x, value, gNu, gLambda, gx, hNuNu, hNuLambda, hLambdaLambda;
    };
    const Point pts[] = {
        {1.0, 0.5, 0.1, 0.95805357023037372, -0.51795693995680921, -0.43190683375484577,
         -5.0336948794653498, -0.85549055729983045, 0.18740906322453684, 0.19317616834842994},
        {2.5, 3.0, 1.0, 0.1072261946163866, -0.044334386698954519, -0.036360143335945037,
         0.024952313262648139, -0.0010389699068358411, 0.010122439739059225, 0.011300380078110494},
        {5.0, 20.0, 4.0, 0.00083426368352553003, -0.0004096448803236336, -0.00026967264092436946,
         0.00063301304375128651, 0.00017680457373557335, 0.00012393856879728422,
         8.4387504114464013e-5},
        {10.0, 100.0, 60.0, 0.00058071159784988302, -8.9245744625219086e-5, -7.8214299779821817e-5,
         0.00010192747284025018, 1.1831675747180932e-5, 1.0644315255248331e-5,
         9.5360650375775306e-6},
        {1.0, 0.5, 0.05, 1.3721459001227669, -1.200716029927276, -0.6520523362867927,
         -14.067325813543144, -0.60834226090068762, 0.53694118899646033, 0.30929889498457942},
    };
    for (const auto& q : pts) {
        const auto p = quantape::math::NoncentralChiSquare<double>{q.nu, q.lam}.pdfGradHess(q.x);
        CHECK_CLOSE_REL("pdf pack value", p.value, q.value, 1e-10);
        CHECK_CLOSE_REL("pdf pack gNu", p.gNu, q.gNu, 1e-10);
        CHECK_CLOSE_REL("pdf pack gLambda", p.gLambda, q.gLambda, 1e-10);
        CHECK_CLOSE_REL("pdf pack gx", p.gx, q.gx, 1e-10);
        CHECK_CLOSE_REL("pdf pack hNuNu", p.hNuNu, q.hNuNu, 1e-10);
        CHECK_CLOSE_REL("pdf pack hNuLambda", p.hNuLambda, q.hNuLambda, 1e-10);
        CHECK_CLOSE_REL("pdf pack hLambdaLambda", p.hLambdaLambda, q.hLambdaLambda, 1e-10);
        // x entries agree with the Bessel-based accessors.
        const auto xPack = quantape::math::NoncentralChiSquare<double>{q.nu, q.lam}.pdfD1D2(q.x);
        CHECK_CLOSE("x pack value agrees", p.value, xPack.value, 0.0);
        CHECK_CLOSE_REL("x pack d1 agrees", p.gx, xPack.d1, 1e-12);
    }
}

TEST(NoncentralChiSquare, pdfGradHessFiniteDifferenceOracle) {
    const double nu = 2.5;
    const double lam = 3.0;
    const double x = 1.0;
    const double h = 1e-5;
    const auto pack = quantape::math::NoncentralChiSquare<double>{nu, lam}.pdfGradHess(x);
    const auto pdfAt = [](double n, double l, double xx) {
        return quantape::math::NoncentralChiSquare<double>{n, l}.pdf(xx);
    };
    CHECK_CLOSE_REL("gNu vs FD", pack.gNu,
                    (pdfAt(nu + h, lam, x) - pdfAt(nu - h, lam, x)) / (2.0 * h), 1e-6, 1e-7);
    CHECK_CLOSE_REL("gLambda vs FD", pack.gLambda,
                    (pdfAt(nu, lam + h, x) - pdfAt(nu, lam - h, x)) / (2.0 * h), 1e-6, 1e-7);
    CHECK_CLOSE_REL("gx vs FD", pack.gx,
                    (pdfAt(nu, lam, x + h) - pdfAt(nu, lam, x - h)) / (2.0 * h), 1e-6, 1e-7);
}
