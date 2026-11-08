// test_student_t.cpp — Student's t pdf/cdf/log tails, closed forms and quantile gates.

#include "quantape/math/Distributions/StudentT.h"

#include <cmath>
#include <filesystem>
#include <fstream>
#include <limits>
#include <numbers>
#include <sstream>
#include <string>

#include "support/GtestSupport.h"

using quantape::math::StudentT;
using quantape::math::detail::BetaValueD1D2;
using quantape::math::detail::studentTCdfD1D2;
using quantape::math::detail::studentTCdfGradHess;
using quantape::math::detail::StudentTCdfGradHess;
using quantape::math::detail::studentTPdfD1D2;
using quantape::math::detail::studentTPdfGradHess;
using quantape::math::detail::StudentTPdfGradHess;

namespace {

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

double normalCdf(double x) {
    return 0.5 * std::erfc(-x / std::numbers::sqrt2);
}

} // namespace

TEST(StudentTDistribution, cauchySpecialCase) {
    const StudentT<double> cauchy{1.0};
    for (double x : {-20.0, -3.0, -0.5, 0.0, 0.5, 3.0, 20.0}) {
        CHECK_CLOSE("Cauchy cdf = 1/2 + atan(x)/pi", cauchy.cdf(x),
                    0.5 + std::atan(x) / std::numbers::pi, 1e-14);
        CHECK_CLOSE_REL("Cauchy pdf = 1/(pi(1+x^2))", cauchy.pdf(x),
                        1.0 / (std::numbers::pi * (1.0 + x * x)), 1e-13, 1e-16);
        CHECK_CLOSE("Cauchy ccdf = 1 - cdf", cauchy.ccdf(x), 0.5 - std::atan(x) / std::numbers::pi,
                    1e-14);
    }
    CHECK_CLOSE("Cauchy inv(1/2) = 0", cauchy.inv(0.5), 0.0, 0.0);
    for (double p : {0.05, 0.1, 0.25, 0.4, 0.6, 0.75, 0.9, 0.95}) {
        CHECK_CLOSE_REL("Cauchy inv = tan(pi(p-1/2))", cauchy.inv(p),
                        std::tan(std::numbers::pi * (p - 0.5)), 1e-12, 1e-15);
    }
}

TEST(StudentTDistribution, twoDegreesClosedForm) {
    const StudentT<double> t2{2.0};
    for (double x : {-8.0, -2.0, -0.5, 0.0, 0.5, 2.0, 8.0}) {
        CHECK_CLOSE_REL("t(2) cdf = 1/2 + x/(2 sqrt(2+x^2))", t2.cdf(x),
                        0.5 + x / (2.0 * std::sqrt(2.0 + x * x)), 1e-13, 1e-15);
        CHECK_CLOSE_REL("t(2) pdf = (2+x^2)^(-3/2)", t2.pdf(x), std::pow(2.0 + x * x, -1.5), 1e-13,
                        1e-16);
        // Rationalized ccdf: 1/(sqrt(2+x^2) (sqrt(2+x^2) + x)).
        CHECK_CLOSE_REL("t(2) ccdf rationalized", t2.ccdf(x),
                        1.0 / (std::sqrt(2.0 + x * x) * (std::sqrt(2.0 + x * x) + x)), 1e-12,
                        1e-15);
    }
    for (double p : {0.01, 0.1, 0.3, 0.5, 0.7, 0.9, 0.99}) {
        const double expected = (2.0 * p - 1.0) * std::sqrt(2.0 / (4.0 * p * (1.0 - p)));
        CHECK_CLOSE_REL("t(2) inv closed form", t2.inv(p), expected, 1e-12, 1e-15);
    }
}

TEST(StudentTDistribution, largeNuApproachesNormal) {
    const StudentT<double> dist{1e5};
    for (double x : {-4.0, -2.0, -0.5, 0.0, 0.5, 2.0, 4.0}) {
        CHECK_CLOSE_REL("t(1e5) cdf ~ normal", dist.cdf(x), normalCdf(x), 1e-3, 1e-3);
        CHECK_CLOSE_REL("t(1e5) pdf ~ normal", dist.pdf(x),
                        std::exp(-0.5 * x * x) / std::sqrt(2.0 * std::numbers::pi), 1e-3, 1e-3);
    }
}

TEST(StudentTDistribution, symmetryAndMonotonicity) {
    for (double nu : {0.7, 2.5, 7.0, 100.0}) {
        const StudentT<double> dist{nu};
        double previous = 0.0;
        for (int i = -200; i <= 200; ++i) {
            const double x = double(i) / 20.0;
            const double value = dist.cdf(x);
            SCOPED_TRACE(x);
            EXPECT_LE(previous, value);
            previous = value;
        }
        for (double x : {-7.0, -1.0, -0.2, 0.2, 1.0, 7.0}) {
            CHECK_CLOSE("cdf(-x) = 1 - cdf(x)", dist.cdf(-x) + dist.cdf(x), 1.0, 1e-13);
            CHECK_CLOSE("ccdf(x) = cdf(-x)", dist.ccdf(x), dist.cdf(-x), 1e-15);
            CHECK_CLOSE_REL("pdf even", dist.pdf(-x), dist.pdf(x), 1e-15, 0.0);
        }
    }
}

TEST(StudentTDistribution, cdfMatchesSimpsonOfPdf) {
    for (double nu : {2.5, 5.0, 30.0}) {
        const StudentT<double> dist{nu};
        for (double y : {0.5, 1.5, 4.0}) {
            const double integral =
                simpsonIntegrate([&](double t) { return dist.pdf(t); }, 0.0, y, 20000);
            CHECK_CLOSE_REL("integral_0^y pdf = 1/2 - ccdf(y)", integral, 0.5 - dist.ccdf(y), 1e-10,
                            1e-14);
        }
    }
}

TEST(StudentTDistribution, quantileRoundTrip) {
    for (double nu : {1.0, 2.0, 3.0, 10.0, 100.0}) {
        const StudentT<double> dist{nu};
        for (double p : {1e-10, 1e-6, 1e-3, 0.01, 0.3, 0.5, 0.7, 0.99, 1.0 - 1e-6, 1.0 - 1e-10}) {
            CHECK_CLOSE_REL("cdf(inv(p)) = p", dist.cdf(dist.inv(p)), p, 1e-9, 1e-300);
        }
    }
    const StudentT<double> dist{3.0};
    CHECK_CLOSE("inv(0.5) = 0", dist.inv(0.5), 0.0, 0.0);
    CHECK_CLOSE("inv(0) = -inf", dist.inv(0.0), -std::numeric_limits<double>::infinity(), 0.0);
    CHECK_CLOSE("inv(1) = +inf", dist.inv(1.0), std::numeric_limits<double>::infinity(), 0.0);
}

TEST(StudentTDistribution, tailClosedFormAndLogAccuracy) {
    // t(3) closed-form tail for x >= 0.
    const StudentT<double> t3{3.0};
    const double s3 = std::sqrt(3.0);
    for (double x : {0.5, 1.0, 3.0}) {
        const double exact =
            0.5 + (1.0 / std::numbers::pi) * ((x / s3) / (1.0 + x * x / 3.0) + std::atan(x / s3));
        CHECK_CLOSE_REL("t(3) exact cdf", t3.cdf(x), exact, 1e-12, 1e-15);
    }
    // Deep tail: ccdf(x) -> 2 sqrt(3)/(pi x^3), checked in log space.
    const double logAsymConst = std::log(2.0 * s3 / std::numbers::pi);
    for (double x : {1e6, 1e10, 1e100}) {
        const double asym = std::exp(logAsymConst - 3.0 * std::log(x));
        CHECK_CLOSE_REL("t(3) deep ccdf ~ 2 sqrt(3)/(pi x^3)", t3.ccdf(x), asym, 1e-10, 0.0);
        CHECK_CLOSE_REL("t(3) deep logCcdf ~ log asymptotic", t3.logCcdf(x),
                        logAsymConst - 3.0 * std::log(x), 1e-10, 1e-14);
    }
    // exp(log) consistency across the whole range, including far tails.
    for (double nu : {0.5, 3.0, 25.0}) {
        const StudentT<double> dist{nu};
        for (double x : {-1e100, -40.0, -1.0, 0.0, 1.0, 40.0, 1e100}) {
            CHECK_CLOSE_REL("exp(logCdf) = cdf", std::exp(dist.logCdf(x)), dist.cdf(x), 1e-12,
                            1e-300);
            CHECK_CLOSE_REL("exp(logCcdf) = ccdf", std::exp(dist.logCcdf(x)), dist.ccdf(x), 1e-12,
                            1e-300);
        }
    }
    CHECK_CLOSE("t cdf(0) = 1/2", t3.cdf(0.0), 0.5, 0.0);
    CHECK_CLOSE("t logCdf(0) = -log 2", t3.logCdf(0.0), -std::numbers::ln2, 1e-15);
}

TEST(StudentTDistribution, derivativePacksMatchCentralDifferences) {
    for (double nu : {0.5, 1.0, 2.5, 8.0, 40.0}) {
        const StudentT<double> dist{nu};
        const auto pdf = [&](double t) { return dist.pdf(t); };
        for (double x : {-4.0, -1.3, -0.4, 0.2, 1.1, 3.5}) {
            const BetaValueD1D2 cdfPack = studentTCdfD1D2(nu, x);
            CHECK_CLOSE_REL("t cdf pack value", cdfPack.value, dist.cdf(x), 1e-14, 1e-16);
            CHECK_CLOSE_REL("t cdf pack d1 = pdf", cdfPack.d1, dist.pdf(x), 1e-12, 1e-16);
            CHECK_CLOSE_REL("t cdf pack d2 vs FD", cdfPack.d2, centralD1(pdf, x, 1e-4), 1e-7,
                            1e-14);
            const BetaValueD1D2 pdfPack = studentTPdfD1D2(nu, x);
            CHECK_CLOSE_REL("t pdf pack value", pdfPack.value, dist.pdf(x), 1e-14, 1e-16);
            CHECK_CLOSE_REL("t pdf pack d1 = pdf", pdfPack.d1, pdfPack.value, 1e-15, 0.0);
            CHECK_CLOSE_REL("t pdf pack d2 vs FD", pdfPack.d2, centralD1(pdf, x, 1e-4), 1e-7,
                            1e-14);
        }
    }
}

TEST(StudentTDistribution, invalidParameters) {
    const StudentT<double> invalid{0.0};
    EXPECT_TRUE(std::isnan(invalid.pdf(0.5)));
    EXPECT_TRUE(std::isnan(invalid.cdf(0.5)));
    EXPECT_TRUE(std::isnan(invalid.logPdf(0.5)));
    EXPECT_TRUE(std::isnan(invalid.logCdf(0.5)));
    EXPECT_TRUE(std::isnan(invalid.ccdf(0.5)));
    EXPECT_TRUE(std::isnan(invalid.logCcdf(0.5)));
    EXPECT_TRUE(std::isnan(invalid.inv(0.5)));
    const StudentT<double> negative{-2.0};
    EXPECT_TRUE(std::isnan(negative.cdf(0.0)));
    const StudentT<double> valid{3.0};
    EXPECT_TRUE(std::isnan(valid.cdf(std::numeric_limits<double>::quiet_NaN())));
    EXPECT_TRUE(std::isnan(valid.inv(1.5)));
    EXPECT_TRUE(std::isnan(valid.inv(-0.1)));
    CHECK_CLOSE("cdf(+inf) = 1", valid.cdf(std::numeric_limits<double>::infinity()), 1.0, 0.0);
    CHECK_CLOSE("cdf(-inf) = 0", valid.cdf(-std::numeric_limits<double>::infinity()), 0.0, 0.0);
}

// ---------------------------------------------------------------------------
// Student-t parameter-derivative packs.
//
// Reference values: the `student-t` section of
// tests/data/distributions/beta_t_paramderiv_reference.csv, generated with
// .venv/bin/python (mpmath 1.4.1, mp.dps = 60) from
//     F = 1 - (1/2) mp.betainc(nu/2, 1/2, 0, nu/(nu+x^2), regularized=True)
//     f = gamma((nu+1)/2) / (sqrt(nu pi) gamma(nu/2)) (1 + x^2/nu)^(-(nu+1)/2)
// with mp.diff for every column and 20 significant digits in the file. The
// grid covers nu in {0.5, 1, 2.7, 10, 50} with x in (-5,5); value/first-order
// entries are gated at 1e-10 relative, second-order at 1e-9.
// ---------------------------------------------------------------------------

TEST(StudentTParameterDerivatives, referenceCsvMatchesMpmath) {
    const std::filesystem::path relative = "distributions/beta_t_paramderiv_reference.csv";
    SKIP_UNLESS_ASSET(std::filesystem::is_regular_file(testDataDir() / relative),
                      "student-t parameter-derivative reference csv absent");
    std::ifstream in(RequireDataFile(relative));
    SKIP_UNLESS_ASSET(in.good(), "student-t parameter-derivative reference csv unreadable");

    std::string line;
    bool inStudentT = false;
    int rows = 0;
    while (std::getline(in, line)) {
        if (line.empty()) {
            continue;
        }
        if (line[0] == '#') {
            inStudentT = line.find("student-t") != std::string::npos;
            continue;
        }
        if (!inStudentT) {
            continue;
        }
        std::stringstream fields(line);
        double f[14] = {};
        int n = 0;
        std::string token;
        while (n < 14 && std::getline(fields, token, ',')) {
            f[n++] = std::stod(token);
        }
        ASSERT_EQ(n, 14) << "malformed reference row: " << line;
        const double nu = f[0];
        const double x = f[1];
        const StudentTCdfGradHess c = studentTCdfGradHess(nu, x);
        const StudentTPdfGradHess p = studentTPdfGradHess(nu, x);
        const std::string label = "nu=" + std::to_string(nu) + " x=" + std::to_string(x);
        CHECK_CLOSE_REL(label + " cdf value", c.value, f[2], 1e-10);
        CHECK_CLOSE_REL(label + " cdf gNu", c.gNu, f[3], 1e-10);
        CHECK_CLOSE_REL(label + " cdf gx", c.gx, f[4], 1e-10);
        CHECK_CLOSE_REL(label + " cdf hNuNu", c.hNuNu, f[5], 1e-9);
        CHECK_CLOSE_REL(label + " cdf hNux", c.hNux, f[6], 1e-9);
        CHECK_CLOSE_REL(label + " cdf hxx", c.hxx, f[7], 1e-9);
        CHECK_CLOSE_REL(label + " pdf value", p.value, f[8], 1e-10);
        CHECK_CLOSE_REL(label + " pdf gNu", p.gNu, f[9], 1e-10);
        CHECK_CLOSE_REL(label + " pdf gx", p.gx, f[10], 1e-10);
        CHECK_CLOSE_REL(label + " pdf hNuNu", p.hNuNu, f[11], 1e-9);
        CHECK_CLOSE_REL(label + " pdf hNux", p.hNux, f[12], 1e-9);
        CHECK_CLOSE_REL(label + " pdf hxx", p.hxx, f[13], 1e-9);
        ++rows;
    }
    EXPECT_EQ(rows, 25);
}

TEST(StudentTParameterDerivatives, xEntriesReproduceExistingPacksBitwise) {
    for (double nu : {0.5, 2.7, 10.0, 50.0}) {
        for (double x : {-4.5, -1.3, -0.4, 0.0, 0.2, 1.1, 3.5}) {
            const std::string label = "nu=" + std::to_string(nu) + " x=" + std::to_string(x);
            const BetaValueD1D2 oldCdf = studentTCdfD1D2(nu, x);
            const StudentTCdfGradHess cdf = studentTCdfGradHess(nu, x);
            CHECK_CLOSE(label + " cdf value", cdf.value, oldCdf.value, 0.0);
            CHECK_CLOSE(label + " cdf gx", cdf.gx, oldCdf.d1, 0.0);
            CHECK_CLOSE(label + " cdf hxx", cdf.hxx, oldCdf.d2, 0.0);
            const BetaValueD1D2 oldPdf = studentTPdfD1D2(nu, x);
            const StudentTPdfGradHess pdf = studentTPdfGradHess(nu, x);
            CHECK_CLOSE(label + " pdf value", pdf.value, oldPdf.value, 0.0);
            CHECK_CLOSE(label + " pdf gx", pdf.gx, oldPdf.d2, 0.0);

            // Cross-pack consistency: F_nux = f_nu and F_xx = f_x.
            CHECK_CLOSE_REL(label + " cdf hNux = pdf gNu", cdf.hNux, pdf.gNu, 1e-9, 1e-12);
            CHECK_CLOSE_REL(label + " cdf hxx = pdf gx", cdf.hxx, pdf.gx, 1e-12, 0.0);
        }
    }
}

TEST(StudentTParameterDerivatives, parameterEntriesMatchFiniteDifferences) {
    for (double nu : {0.7, 2.5, 8.0}) {
        for (double x : {-3.0, -0.7, 0.4, 1.3, 3.1}) {
            const double hnu = 1e-5 * nu;
            const double hx = 1e-5 * std::fmax(1.0, std::fabs(x));
            const std::string label = "nu=" + std::to_string(nu) + " x=" + std::to_string(x);
            const StudentTCdfGradHess c = studentTCdfGradHess(nu, x);
            const StudentTPdfGradHess p = studentTPdfGradHess(nu, x);

            const auto cdfValue = [](double n, double xx) {
                return studentTCdfGradHess(n, xx).value;
            };
            const auto cdfGNu = [](double n, double xx) { return studentTCdfGradHess(n, xx).gNu; };
            const auto cdfGx = [](double n, double xx) { return studentTCdfGradHess(n, xx).gx; };
            const auto pdfValue = [](double n, double xx) {
                return studentTPdfGradHess(n, xx).value;
            };
            const auto pdfGNu = [](double n, double xx) { return studentTPdfGradHess(n, xx).gNu; };
            const auto pdfGx = [](double n, double xx) { return studentTPdfGradHess(n, xx).gx; };

            CHECK_CLOSE_REL(label + " cdf gNu vs FD", c.gNu,
                            (cdfValue(nu + hnu, x) - cdfValue(nu - hnu, x)) / (2.0 * hnu), 1e-5,
                            1e-9);
            CHECK_CLOSE_REL(label + " cdf gx vs FD", c.gx,
                            (cdfValue(nu, x + hx) - cdfValue(nu, x - hx)) / (2.0 * hx), 1e-5, 1e-9);
            CHECK_CLOSE_REL(label + " cdf hNuNu vs FD", c.hNuNu,
                            (cdfGNu(nu + hnu, x) - cdfGNu(nu - hnu, x)) / (2.0 * hnu), 1e-5, 1e-9);
            CHECK_CLOSE_REL(label + " cdf hNux vs FD(nu,x)", c.hNux,
                            (cdfGNu(nu, x + hx) - cdfGNu(nu, x - hx)) / (2.0 * hx), 1e-5, 1e-9);
            CHECK_CLOSE_REL(label + " cdf hNux vs FD(x,nu)", c.hNux,
                            (cdfGx(nu + hnu, x) - cdfGx(nu - hnu, x)) / (2.0 * hnu), 1e-5, 1e-9);
            CHECK_CLOSE_REL(label + " cdf hxx vs FD", c.hxx,
                            (cdfGx(nu, x + hx) - cdfGx(nu, x - hx)) / (2.0 * hx), 1e-5, 1e-9);

            CHECK_CLOSE_REL(label + " pdf gNu vs FD", p.gNu,
                            (pdfValue(nu + hnu, x) - pdfValue(nu - hnu, x)) / (2.0 * hnu), 1e-5,
                            1e-9);
            CHECK_CLOSE_REL(label + " pdf gx vs FD", p.gx,
                            (pdfValue(nu, x + hx) - pdfValue(nu, x - hx)) / (2.0 * hx), 1e-5, 1e-9);
            CHECK_CLOSE_REL(label + " pdf hNuNu vs FD", p.hNuNu,
                            (pdfGNu(nu + hnu, x) - pdfGNu(nu - hnu, x)) / (2.0 * hnu), 1e-5, 1e-9);
            CHECK_CLOSE_REL(label + " pdf hNux vs FD(nu,x)", p.hNux,
                            (pdfGNu(nu, x + hx) - pdfGNu(nu, x - hx)) / (2.0 * hx), 1e-5, 1e-9);
            CHECK_CLOSE_REL(label + " pdf hNux vs FD(x,nu)", p.hNux,
                            (pdfGx(nu + hnu, x) - pdfGx(nu - hnu, x)) / (2.0 * hnu), 1e-5, 1e-9);
            CHECK_CLOSE_REL(label + " pdf hxx vs FD", p.hxx,
                            (pdfGx(nu, x + hx) - pdfGx(nu, x - hx)) / (2.0 * hx), 1e-5, 1e-9);
        }
    }
}

TEST(StudentTParameterDerivatives, symmetryTailsAndLargeNu) {
    // F(-x) = 1 - F(x) and the even/odd structure of every entry.
    for (double nu : {0.5, 2.7, 10.0}) {
        for (double x : {0.3, 1.3, 3.5}) {
            const std::string label = "nu=" + std::to_string(nu) + " x=" + std::to_string(x);
            const StudentTCdfGradHess j = studentTCdfGradHess(nu, x);
            const StudentTCdfGradHess k = studentTCdfGradHess(nu, -x);
            CHECK_CLOSE(label + " cdf value mirror", j.value + k.value, 1.0, 1e-12);
            CHECK_CLOSE(label + " cdf gNu odd", j.gNu + k.gNu, 0.0, 1e-12);
            CHECK_CLOSE(label + " cdf gx even", j.gx - k.gx, 0.0, 1e-12);
            CHECK_CLOSE(label + " cdf hNuNu odd", j.hNuNu + k.hNuNu, 0.0, 1e-12);
            CHECK_CLOSE(label + " cdf hNux even", j.hNux - k.hNux, 0.0, 1e-12);
            CHECK_CLOSE(label + " cdf hxx odd", j.hxx + k.hxx, 0.0, 1e-12);

            const StudentTPdfGradHess p = studentTPdfGradHess(nu, x);
            const StudentTPdfGradHess q = studentTPdfGradHess(nu, -x);
            CHECK_CLOSE(label + " pdf value even", p.value - q.value, 0.0, 1e-15);
            CHECK_CLOSE(label + " pdf gNu even", p.gNu - q.gNu, 0.0, 1e-12);
            CHECK_CLOSE(label + " pdf gx odd", p.gx + q.gx, 0.0, 1e-12);
            CHECK_CLOSE(label + " pdf hNuNu even", p.hNuNu - q.hNuNu, 0.0, 1e-12);
            CHECK_CLOSE(label + " pdf hNux odd", p.hNux + q.hNux, 0.0, 1e-12);
            CHECK_CLOSE(label + " pdf hxx even", p.hxx - q.hxx, 0.0, 1e-12);
        }
    }

    // Deep tails: the packs stay finite and keep the old x entries bit-for-bit.
    for (double x : {1e4, 1e6}) {
        const StudentTCdfGradHess c = studentTCdfGradHess(3.0, x);
        const StudentTPdfGradHess p = studentTPdfGradHess(3.0, x);
        EXPECT_TRUE(std::isfinite(c.gNu));
        EXPECT_TRUE(std::isfinite(c.hNuNu));
        EXPECT_TRUE(std::isfinite(c.hNux));
        EXPECT_TRUE(std::isfinite(p.gNu));
        EXPECT_TRUE(std::isfinite(p.hNuNu));
        EXPECT_TRUE(std::isfinite(p.hNux));
        CHECK_CLOSE("tail cdf value", c.value, studentTCdfD1D2(3.0, x).value, 0.0);
        CHECK_CLOSE("tail cdf hxx", c.hxx, studentTCdfD1D2(3.0, x).d2, 0.0);
        CHECK_CLOSE("tail pdf value", p.value, studentTPdfD1D2(3.0, x).value, 0.0);
    }

    // Large nu: 4th-order (Richardson) FD cross-check. The value-level
    // `lgamma` cancellation documented in the header limits both the analytic
    // and the FD estimate near nu ~ 1e3 to ~1e-6 relative.
    for (double x : {0.5, 1.0}) {
        const double nu = 1000.0;
        const StudentTCdfGradHess c = studentTCdfGradHess(nu, x);
        const StudentTPdfGradHess p = studentTPdfGradHess(nu, x);
        const auto richards = [](const auto& f, double h) {
            return (4.0 * f(h) - f(2.0 * h)) / 3.0;
        };
        const double gNuFd = richards(
            [&](double h) {
                return (studentTCdfGradHess(nu + h, x).value -
                        studentTCdfGradHess(nu - h, x).value) /
                       (2.0 * h);
            },
            0.25);
        const double hNuNuFd = richards(
            [&](double h) {
                return (studentTCdfGradHess(nu + h, x).gNu - studentTCdfGradHess(nu - h, x).gNu) /
                       (2.0 * h);
            },
            0.25);
        const double pdfGNuFd = richards(
            [&](double h) {
                return (studentTPdfGradHess(nu + h, x).value -
                        studentTPdfGradHess(nu - h, x).value) /
                       (2.0 * h);
            },
            0.25);
        const std::string label = "nu=1e3 x=" + std::to_string(x);
        CHECK_CLOSE_REL(label + " cdf gNu vs Richardson FD", c.gNu, gNuFd, 1e-4, 1e-10);
        CHECK_CLOSE_REL(label + " cdf hNuNu vs Richardson FD", c.hNuNu, hNuNuFd, 1e-4, 1e-10);
        CHECK_CLOSE_REL(label + " pdf gNu vs Richardson FD", p.gNu, pdfGNuFd, 1e-4, 1e-10);
    }
}

TEST(StudentTParameterDerivatives, edgeBehavior) {
    for (const auto& [nu, x] : {std::pair<double, double>{0.0, 0.5},
                                {-2.0, 0.5},
                                {3.0, std::numeric_limits<double>::quiet_NaN()},
                                {std::numeric_limits<double>::quiet_NaN(), 0.5}}) {
        const StudentTCdfGradHess c = studentTCdfGradHess(nu, x);
        const StudentTPdfGradHess p = studentTPdfGradHess(nu, x);
        for (double v : {c.value, c.gNu, c.gx, c.hNuNu, c.hNux, c.hxx, p.value, p.gNu, p.gx,
                         p.hNuNu, p.hNux, p.hxx}) {
            EXPECT_TRUE(std::isnan(v));
        }
    }

    // x = 0: one-sided limits; mixed entry equals the density nu-derivative.
    const StudentTCdfGradHess atZero = studentTCdfGradHess(2.7, 0.0);
    const StudentTPdfGradHess density = studentTPdfGradHess(2.7, 0.0);
    CHECK_CLOSE("cdf(0) = 1/2", atZero.value, 0.5, 0.0);
    CHECK_CLOSE("cdf gNu(0) = 0", atZero.gNu, 0.0, 0.0);
    CHECK_CLOSE("cdf hNuNu(0) = 0", atZero.hNuNu, 0.0, 0.0);
    CHECK_CLOSE("cdf hxx(0) = 0", atZero.hxx, 0.0, 0.0);
    CHECK_CLOSE("cdf gx(0) = pdf(0)", atZero.gx, density.value, 0.0);
    CHECK_CLOSE("cdf hNux(0) = pdf_nu(0)", atZero.hNux, density.gNu, 0.0);
    CHECK_CLOSE("pdf gx(0) = 0", density.gx, 0.0, 0.0);
    CHECK_CLOSE_REL("pdf hxx(0) = -(nu+1) pdf/nu", density.hxx, -(2.7 + 1.0) * density.value / 2.7,
                    1e-15);

    const double inf = std::numeric_limits<double>::infinity();
    const StudentTCdfGradHess up = studentTCdfGradHess(3.0, inf);
    const StudentTCdfGradHess down = studentTCdfGradHess(3.0, -inf);
    CHECK_CLOSE("cdf(+inf) = 1", up.value, 1.0, 0.0);
    CHECK_CLOSE("cdf(-inf) = 0", down.value, 0.0, 0.0);
    for (double v : {up.gNu, up.gx, up.hNuNu, up.hNux, up.hxx, down.gNu, down.gx, down.hNuNu,
                     down.hNux, down.hxx}) {
        CHECK_CLOSE("flat +-inf pack", v, 0.0, 0.0);
    }
    const StudentTPdfGradHess pdfUp = studentTPdfGradHess(3.0, inf);
    for (double v : {pdfUp.value, pdfUp.gNu, pdfUp.gx, pdfUp.hNuNu, pdfUp.hNux, pdfUp.hxx}) {
        CHECK_CLOSE("flat pdf +inf pack", v, 0.0, 0.0);
    }
}

// The t quantile is a power law, so it stays finite down to the smallest
// subnormal p (the normal-quantile seed resolves that range). References:
// Boost `quantile(students_t_distribution<cpp_bin_float_100>{5}, p)`.
TEST(StudentTDistribution, invResolvesSubnormalTail) {
    const StudentT<double> dist{5.0};
    CHECK_CLOSE_REL("t5 inv(5e-324)", dist.inv(5e-324), -7.18948599151996756084e64, 1e-12);
    CHECK_CLOSE_REL("t5 inv(7e-311)", dist.inv(7e-311), -1.68436095377787928605e62, 1e-12);
    // The quantile reproduces the smallest positive double exactly.
    const double pmin = std::numeric_limits<double>::denorm_min();
    CHECK_CLOSE("t5 cdf(inv(denorm_min))", dist.cdf(dist.inv(pmin)), pmin, 0.0);
}
