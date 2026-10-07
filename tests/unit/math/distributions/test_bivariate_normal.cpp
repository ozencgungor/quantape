// Tests for the standard bivariate normal distribution (CDF/PDF and the
// analytic gradient/Hessian pack). The CDF is gated three ways: exact limits,
// independent identities, and high-precision mpmath reference values in
// tests/data/distributions/bvn_reference.csv (generated with two independent
// high-precision quadratures that agreed to 1e-51; see the CSV header).
//
// Finite-difference gates use Richardson-extrapolated central differences
// (base step h = 1e-4): plain central differences at the curvature scales of
// the Hessian near |rho| = 0.99 carry O(h^2) truncation and O(eps/h^2) roundoff
// near 1e-6, so the extrapolated stencil is what makes the 1e-6 absolute gate
// meaningful. The analytic pack itself was verified against mpmath
// derivatives at 50 digits (max scaled error ~4e-15).

#include "quantape/math/Distributions/BivariateNormal.h"

#include <cmath>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#include "support/GtestSupport.h"

namespace {

constexpr double kTol = 1e-12;
constexpr double kPi = 3.14159265358979323846;

using quantape::math::BivariateNormal;
using quantape::math::detail::BvnGradHess;

double phi(double x) {
    return 0.5 * std::erfc(-x / std::sqrt(2.0));
}
double phiPdf(double x) {
    return std::exp(-0.5 * x * x) / std::sqrt(2.0 * kPi);
}

struct RefRow {
    double a;
    double b;
    double rho;
    double cdf;
};

std::vector<RefRow> loadReference() {
    std::vector<RefRow> rows;
    std::ifstream in(RequireDataFile("distributions/bvn_reference.csv"));
    std::string line;
    std::getline(in, line);
    while (std::getline(in, line)) {
        if (line.empty()) {
            continue;
        }
        std::stringstream ss(line);
        std::string field;
        double values[4] = {0.0, 0.0, 0.0, 0.0};
        int count = 0;
        while (std::getline(ss, field, ',') && count < 4) {
            values[count++] = std::stod(field);
        }
        if (count == 4) {
            rows.push_back({values[0], values[1], values[2], values[3]});
        }
    }
    return rows;
}

/// Richardson-extrapolated central first derivative (error O(h^4)).
template <class F>
double richFirst(F f, double x, double h) {
    const double d1 = (f(x + h) - f(x - h)) / (2.0 * h);
    const double d2 = (f(x + 2.0 * h) - f(x - 2.0 * h)) / (4.0 * h);
    return (4.0 * d1 - d2) / 3.0;
}

/// Richardson-extrapolated central second derivative (error O(h^4)).
template <class F>
double richSecond(F f, double x, double h) {
    const double d1 = (f(x + h) - 2.0 * f(x) + f(x - h)) / (h * h);
    const double d2 = (f(x + 2.0 * h) - 2.0 * f(x) + f(x - 2.0 * h)) / (4.0 * h * h);
    return (4.0 * d1 - d2) / 3.0;
}

/// Richardson-extrapolated central mixed derivative for f(x,y).
template <class F>
double richMixed(F f, double x, double y, double h) {
    const double d1 =
        (f(x + h, y + h) - f(x + h, y - h) - f(x - h, y + h) + f(x - h, y - h)) / (4.0 * h * h);
    const double d2 = (f(x + 2.0 * h, y + 2.0 * h) - f(x + 2.0 * h, y - 2.0 * h) -
                       f(x - 2.0 * h, y + 2.0 * h) + f(x - 2.0 * h, y - 2.0 * h)) /
                      (16.0 * h * h);
    return (4.0 * d1 - d2) / 3.0;
}

std::string pointLabel(const char* what, double a, double b, double rho) {
    std::ostringstream out;
    out.precision(6);
    out << what << " a=" << a << " b=" << b << " rho=" << rho;
    return out.str();
}

} // namespace

TEST(BivariateNormalDistribution, limitsAndSymmetries) {
    const double grid[] = {-3.0, -1.0, -0.2, 0.0, 0.5, 2.0, 4.0};

    for (double a : grid) {
        for (double b : grid) {
            const double phia = phi(a);
            const double phib = phi(b);
            CHECK_CLOSE(pointLabel("rho=0", a, b, 0.0), BivariateNormal<double>{0.0}.cdf(a, b),
                        phia * phib, kTol);
            CHECK_CLOSE(pointLabel("rho=1", a, b, 1.0), BivariateNormal<double>{1.0}.cdf(a, b),
                        phi(std::min(a, b)), kTol);
            const double limit = std::max(0.0, phia + phib - 1.0);
            CHECK_CLOSE(pointLabel("rho=-1", a, b, -1.0), BivariateNormal<double>{-1.0}.cdf(a, b),
                        limit, kTol);
        }
    }

    const double rhos[] = {-0.99, -0.9, -0.25, 0.0, 0.25, 0.9, 0.99};
    for (double a : grid) {
        for (double b : grid) {
            for (double rho : rhos) {
                const BivariateNormal<double> d{rho};
                const double got = d.cdf(a, b);
                CHECK_CLOSE(pointLabel("symmetry", a, b, rho), got, d.cdf(b, a), kTol);

                // P(X<=a,Y<=b) + P(X<=a,Y>b) = Phi(a); the second term is
                // Phi2(a,-b;-rho) by the reflection of Y.
                const BivariateNormal<double> reflected{-rho};
                CHECK_CLOSE(pointLabel("reflection", a, b, rho), got + reflected.cdf(a, -b), phi(a),
                            kTol);

                // Phi2(a,b;rho) - Phi2(-a,-b;rho) = Phi(a) + Phi(b) - 1.
                CHECK_CLOSE(pointLabel("complement", a, b, rho), got - d.cdf(-a, -b),
                            phi(a) + phi(b) - 1.0, kTol);
            }
        }
    }
}

TEST(BivariateNormalDistribution, pdfMatchesDensity) {
    const double points[][2] = {{0.0, 0.0}, {1.0, -1.0}, {-2.0, 0.5}, {0.3, 2.0}};
    const double rhos[] = {0.0, -0.5, 0.9, -0.9};
    for (const auto& p : points) {
        for (double rho : rhos) {
            const BivariateNormal<double> d{rho};
            const double s2 = (1.0 - rho) * (1.0 + rho);
            const double q = p[0] * p[0] - 2.0 * rho * p[0] * p[1] + p[1] * p[1];
            const double expected = std::exp(-q / (2.0 * s2)) / (2.0 * kPi * std::sqrt(s2));
            CHECK_CLOSE(pointLabel("pdf", p[0], p[1], rho), d.pdf(p[0], p[1]), expected, 1e-15);
            CHECK_CLOSE(pointLabel("pdf symmetry", p[0], p[1], rho), d.pdf(p[0], p[1]),
                        d.pdf(p[1], p[0]), 1e-15);
        }
        CHECK_CLOSE("pdf independent", BivariateNormal<double>{0.0}.pdf(p[0], p[1]),
                    phiPdf(p[0]) * phiPdf(p[1]), 1e-15);
    }
}

TEST(BivariateNormalDistribution, referenceValues) {
    const std::vector<RefRow> rows = loadReference();
    EXPECT_FALSE(rows.empty());
    double worst = 0.0;
    for (const RefRow& row : rows) {
        const BivariateNormal<double> d{row.rho};
        const double got = d.cdf(row.a, row.b);
        worst = std::max(worst, std::abs(got - row.cdf));
        CHECK_CLOSE(pointLabel("reference", row.a, row.b, row.rho), got, row.cdf, 1e-10);
    }
    EXPECT_LT(worst, 1e-10);
}

TEST(BivariateNormalDistribution, derivativePackAgainstCentralDifferences) {
    struct Point {
        double a;
        double b;
        double rho;
    };
    const Point points[] = {
        {0.0, 0.0, 0.0},   {1.0, 1.0, 0.0},    {-1.0, 0.5, 0.0},   {0.3, -0.7, 0.0},
        {0.0, 0.0, 0.5},   {1.0, -1.0, 0.5},   {-0.5, 2.0, 0.5},   {2.0, 0.5, -0.5},
        {1.0, 0.5, 0.9},   {-1.0, -1.0, 0.9},  {0.25, 1.5, 0.9},   {2.0, -2.0, -0.9},
        {0.5, 0.5, 0.99},  {-0.5, 0.25, 0.99}, {1.5, -1.5, -0.99}, {-2.0, 2.0, -0.99},
        {0.0, 0.5, -0.99}, {0.5, 0.0, 0.99},   {3.0, 1.0, 0.75},   {-3.0, -1.0, -0.75},
    };
    const double h = 1e-4;
    const double tol = 1e-6;

    for (const Point& p : points) {
        const BvnGradHess pack = quantape::math::detail::bvnCdfGradHess(p.a, p.b, p.rho);
        const BivariateNormal<double> d{p.rho};
        const auto cdfA = [&](double a) { return d.cdf(a, p.b); };
        const auto cdfB = [&](double b) { return d.cdf(p.a, b); };
        const auto cdfR = [&](double rho) { return BivariateNormal<double>{rho}.cdf(p.a, p.b); };
        const auto cdfAB = [&](double a, double b) { return d.cdf(a, b); };
        const auto cdfAR = [&](double a, double rho) {
            return BivariateNormal<double>{rho}.cdf(a, p.b);
        };
        const auto cdfBR = [&](double b, double rho) {
            return BivariateNormal<double>{rho}.cdf(p.a, b);
        };

        CHECK_CLOSE(pointLabel("value", p.a, p.b, p.rho), pack.value, d.cdf(p.a, p.b), kTol);
        CHECK_CLOSE(pointLabel("ga", p.a, p.b, p.rho), pack.ga, richFirst(cdfA, p.a, h), tol);
        CHECK_CLOSE(pointLabel("gb", p.a, p.b, p.rho), pack.gb, richFirst(cdfB, p.b, h), tol);
        CHECK_CLOSE(pointLabel("gr", p.a, p.b, p.rho), pack.gr, richFirst(cdfR, p.rho, h), tol);
        CHECK_CLOSE(pointLabel("haa", p.a, p.b, p.rho), pack.haa, richSecond(cdfA, p.a, h), tol);
        CHECK_CLOSE(pointLabel("hbb", p.a, p.b, p.rho), pack.hbb, richSecond(cdfB, p.b, h), tol);
        CHECK_CLOSE(pointLabel("hrr", p.a, p.b, p.rho), pack.hrr, richSecond(cdfR, p.rho, h), tol);
        CHECK_CLOSE(pointLabel("hab", p.a, p.b, p.rho), pack.hab, richMixed(cdfAB, p.a, p.b, h),
                    tol);
        CHECK_CLOSE(pointLabel("har", p.a, p.b, p.rho), pack.har, richMixed(cdfAR, p.a, p.rho, h),
                    tol);
        CHECK_CLOSE(pointLabel("hbr", p.a, p.b, p.rho), pack.hbr, richMixed(cdfBR, p.b, p.rho, h),
                    tol);
        CHECK_CLOSE(pointLabel("gr=pdf", p.a, p.b, p.rho), pack.gr, d.pdf(p.a, p.b), 1e-15);

        // (a <-> b) symmetry of the pack.
        const BvnGradHess swapped = quantape::math::detail::bvnCdfGradHess(p.b, p.a, p.rho);
        CHECK_CLOSE(pointLabel("value sw", p.a, p.b, p.rho), pack.value, swapped.value, kTol);
        CHECK_CLOSE(pointLabel("ga/gb sw", p.a, p.b, p.rho), pack.ga, swapped.gb, kTol);
        CHECK_CLOSE(pointLabel("gb/ga sw", p.a, p.b, p.rho), pack.gb, swapped.ga, kTol);
        CHECK_CLOSE(pointLabel("haa/hbb sw", p.a, p.b, p.rho), pack.haa, swapped.hbb, kTol);
        CHECK_CLOSE(pointLabel("hab sw", p.a, p.b, p.rho), pack.hab, swapped.hab, kTol);
        CHECK_CLOSE(pointLabel("har/hbr sw", p.a, p.b, p.rho), pack.har, swapped.hbr, kTol);
        CHECK_CLOSE(pointLabel("hbr/har sw", p.a, p.b, p.rho), pack.hbr, swapped.har, kTol);
    }
}

TEST(BivariateNormalDistribution, edgeCases) {
    const double inf = std::numeric_limits<double>::infinity();
    const double nan = std::numeric_limits<double>::quiet_NaN();

    EXPECT_TRUE(std::isnan(BivariateNormal<double>{0.3}.cdf(nan, 0.0)));
    EXPECT_TRUE(std::isnan(BivariateNormal<double>{0.3}.cdf(0.0, nan)));
    EXPECT_TRUE(std::isnan(BivariateNormal<double>{nan}.cdf(0.0, 0.0)));
    EXPECT_TRUE(std::isnan(BivariateNormal<double>{2.0}.cdf(0.0, 0.0)));
    EXPECT_TRUE(std::isnan(BivariateNormal<double>{-2.0}.cdf(0.0, 0.0)));
    EXPECT_TRUE(std::isnan(BivariateNormal<double>{inf}.cdf(0.0, 0.0)));

    CHECK_CLOSE("inf,inf", BivariateNormal<double>{0.3}.cdf(inf, inf), 1.0, kTol);
    CHECK_CLOSE("inf,inf rho=1", BivariateNormal<double>{1.0}.cdf(inf, inf), 1.0, kTol);
    CHECK_CLOSE("inf,0", BivariateNormal<double>{-0.7}.cdf(inf, 0.0), 0.5, kTol);
    CHECK_CLOSE("inf,-inf", BivariateNormal<double>{0.3}.cdf(inf, -inf), 0.0, kTol);
    CHECK_CLOSE("-inf,0.5", BivariateNormal<double>{0.3}.cdf(-inf, 0.5), 0.0, kTol);
    CHECK_CLOSE("0,inf", BivariateNormal<double>{-0.7}.cdf(0.0, inf), 0.5, kTol);
    CHECK_CLOSE("inf,2 rho=1", BivariateNormal<double>{1.0}.cdf(inf, 2.0), phi(2.0), kTol);
    CHECK_CLOSE("inf,2 rho=-1", BivariateNormal<double>{-1.0}.cdf(inf, 2.0), phi(2.0), kTol);

    // Exact a=b=0 formula Phi2(0,0;rho) = 1/4 + asin(rho)/(2 pi) covers the
    // extreme representable correlations.
    const double rhos[] = {-0.9999999999999999, -0.999999, -0.5, 0.0, 0.5, 0.999999,
                           0.9999999999999999};
    for (double rho : rhos) {
        const double expected = 0.25 + std::asin(rho) / (2.0 * kPi);
        CHECK_CLOSE(pointLabel("asin formula", 0.0, 0.0, rho),
                    BivariateNormal<double>{rho}.cdf(0.0, 0.0), expected, 1e-12);
    }

    // rho = -1 with a + b < 0 must clamp at zero rather than go negative.
    CHECK_CLOSE("rho=-1 clamp", BivariateNormal<double>{-1.0}.cdf(-3.0, -3.0), 0.0, kTol);
    CHECK_CLOSE("rho=-1 sum", BivariateNormal<double>{-1.0}.cdf(-3.0, 4.0),
                phi(-3.0) + phi(4.0) - 1.0, kTol);
    CHECK_CLOSE("rho=-1 positive", BivariateNormal<double>{-1.0}.cdf(3.0, 3.0),
                phi(3.0) + phi(3.0) - 1.0, kTol);
    CHECK_CLOSE("rho=1 min", BivariateNormal<double>{1.0}.cdf(1.5, -0.5), phi(-0.5), kTol);
}
