// test_lognormal_exponential.cpp — Lognormal and Exponential distribution gates.
//
// cdf references come from a local composite-Simpson integration of the pdf
// (plain TU: no Eigen/GaussLegendre dependency). The derivative packs are
// checked with central differences at h = 1e-5 max(1, |x|): d1 against the
// value and d2 against the analytic d1. A direct second difference at this h is
// cancellation-limited to ~eps |f| / h^2 ~ 1e-5 and could not meet the 1e-7
// tolerance, while differencing the (independently FD-verified) d1 does.
//
// The parameter gradient/Hessian packs (`lognormalPdfGradHess`,
// `lognormalCdfGradHess`, `exponentialPdfGradHess`, `exponentialCdfGradHess`)
// are gated three ways: (1) mpmath high-precision references in
// tests/data/distributions/lnexp_ig_paramderiv_reference.csv, generated with
// `.venv/bin/python` + mpmath at dps=50 by numerically differentiating the
// value functions (`mp.diff` on tuples of arguments) and cross-checked against
// the analytic closed forms to < 1e-30; (2) central differences of the
// analytic pack values/gradients at the same h; (3) the exact
// Normal-in-log-space identity for the Lognormal plus explicit closed-form
// anchors at x = 0 and x = +inf.

#include "quantape/math/Distributions/Exponential.h"
#include "quantape/math/Distributions/Lognormal.h"
#include "quantape/math/Distributions/Normal.h"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <limits>
#include <sstream>
#include <string>
#include <vector>

#include "support/GtestSupport.h"

namespace {

constexpr double kTol = 1e-12;

/// Composite Simpson rule with n (even) intervals on [a, b].
template <class F>
double compositeSimpson(const F& f, double a, double b, long n) {
    if (n % 2 != 0) {
        ++n;
    }
    const double h = (b - a) / static_cast<double>(n);
    double sum = f(a) + f(b);
    for (long i = 1; i < n; ++i) {
        sum += ((i & 1) ? 4.0 : 2.0) * f(a + static_cast<double>(i) * h);
    }
    return sum * h / 3.0;
}

template <class F>
double centralD1(const F& f, double x, double h) {
    return (f(x + h) - f(x - h)) / (2.0 * h);
}

} // namespace

TEST(LognormalDistribution, knownValues) {
    const quantape::math::Lognormal<double> d{0.0, 1.0};
    CHECK_CLOSE("cdf(1) = 0.5", d.cdf(1.0), 0.5, kTol);
    CHECK_CLOSE("pdf(1) = 1 / sqrt(2 pi)", d.pdf(1.0), 1.0 / std::sqrt(2.0 * quantape::util::kPi),
                kTol);
    CHECK_CLOSE("logPdf(1) = ln pdf(1)", d.logPdf(1.0), std::log(d.pdf(1.0)), kTol);
    CHECK_CLOSE("cdf(e^1.96) = Phi(1.96)", d.cdf(std::exp(1.96)), 0.97500210485177952, kTol);
    CHECK_CLOSE("ccdf(1) = 0.5", d.ccdf(1.0), 0.5, kTol);

    const quantape::math::Lognormal<double> s{0.7, 0.35};
    CHECK_CLOSE("median inv(0.5) = e^mu", s.inv(0.5), std::exp(0.7), kTol);
    CHECK_CLOSE("logCcdf = ln ccdf", s.logCcdf(0.4), std::log(s.ccdf(0.4)), kTol);
}

TEST(LognormalDistribution, cdfMatchesPdfIntegral) {
    const quantape::math::Lognormal<double> d{0.3, 0.7};
    for (double x : {0.05, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0}) {
        const double got = d.cdf(x);
        const double expected =
            compositeSimpson([&](double t) { return d.pdf(t); }, 0.0, x, 200000);
        CHECK_CLOSE("cdf(x) = int_0^x pdf", got, expected, 1e-11);
    }
}

TEST(LognormalDistribution, invRoundTrip) {
    const quantape::math::Lognormal<double> d{0.3, 0.7};
    for (double x : {0.05, 0.2, 0.7, 1.0, 2.5, 10.0}) {
        CHECK_CLOSE_REL("x = inv(cdf(x))", d.inv(d.cdf(x)), x, 1e-8);
    }
    for (double p : {1e-6, 0.01, 0.25, 0.5, 0.9, 0.999, 0.9999}) {
        CHECK_CLOSE("p = cdf(inv(p))", d.cdf(d.inv(p)), p, 1e-9);
    }
    CHECK_CLOSE("inv(0) = 0", d.inv(0.0), 0.0, 0.0);
    EXPECT_TRUE(std::isinf(d.inv(1.0)));
}

TEST(LognormalDistribution, reciprocalSymmetryIdentity) {
    // X ~ LN(mu, sigma^2) => 1/X ~ LN(-mu, sigma^2).
    const quantape::math::Lognormal<double> d{0.4, 0.9};
    const quantape::math::Lognormal<double> flipped{-0.4, 0.9};
    for (double x : {0.1, 0.5, 1.0, 3.0, 20.0}) {
        CHECK_CLOSE("cdf_{mu}(x) + cdf_{-mu}(1/x) = 1", d.cdf(x) + flipped.cdf(1.0 / x), 1.0,
                    1e-14);
        CHECK_CLOSE("pdf_{mu}(x) = pdf_{-mu}(1/x) / x^2", d.pdf(x), flipped.pdf(1.0 / x) / (x * x),
                    1e-14);
    }
}

TEST(LognormalDistribution, tailsAndEdges) {
    const quantape::math::Lognormal<double> d{0.2, 0.8};
    const double inf = std::numeric_limits<double>::infinity();

    CHECK_CLOSE("pdf(0) = 0", d.pdf(0.0), 0.0, 0.0);
    CHECK_CLOSE("cdf(0) = 0", d.cdf(0.0), 0.0, 0.0);
    CHECK_CLOSE("ccdf(0) = 1", d.ccdf(0.0), 1.0, 0.0);
    CHECK_CLOSE("logCcdf(0) = 0", d.logCcdf(0.0), 0.0, 0.0);
    CHECK_CLOSE("pdf(-1) = 0", d.pdf(-1.0), 0.0, 0.0);
    CHECK_CLOSE("cdf(-1) = 0", d.cdf(-1.0), 0.0, 0.0);
    EXPECT_TRUE(std::isinf(d.logPdf(0.0)) && d.logPdf(0.0) < 0.0);
    EXPECT_TRUE(std::isinf(d.logCdf(0.0)) && d.logCdf(0.0) < 0.0);

    CHECK_CLOSE("pdf(1e-300) = 0", d.pdf(1e-300), 0.0, 0.0);
    CHECK_CLOSE("cdf(1e-300) = 0", d.cdf(1e-300), 0.0, 0.0);
    CHECK_CLOSE("cdf(+inf) = 1", d.cdf(inf), 1.0, 1e-15);
    CHECK_CLOSE("ccdf(+inf) = 0", d.ccdf(inf), 0.0, 1e-15);
    CHECK_CLOSE("pdf(+inf) = 0", d.pdf(inf), 0.0, 0.0);

    const double nan = std::numeric_limits<double>::quiet_NaN();
    EXPECT_TRUE(std::isnan(d.pdf(nan)));
    EXPECT_TRUE(std::isnan(d.cdf(nan)));
    EXPECT_TRUE(std::isnan(d.logPdf(nan)));
    EXPECT_TRUE(std::isnan(d.logCdf(nan)));
    EXPECT_TRUE(std::isnan(d.ccdf(nan)));
    EXPECT_TRUE(std::isnan(d.inv(nan)));
}

TEST(LognormalDistribution, smallSigmaIsSharp) {
    const quantape::math::Lognormal<double> d{0.0, 1e-4};
    CHECK_CLOSE("cdf(exp(-1e-3)) ~ 0", d.cdf(std::exp(-1e-3)), 0.0, 1e-15);
    CHECK_CLOSE("cdf(exp(+1e-3)) ~ 1", d.cdf(std::exp(1e-3)), 1.0, 1e-15);
    CHECK_CLOSE_REL("peak pdf = 1 / (sigma sqrt(2 pi))", d.pdf(1.0),
                    1.0 / (1e-4 * std::sqrt(2.0 * quantape::util::kPi)), 1e-12);
}

TEST(LognormalDistribution, derivativePacksMatchCentralDifferences) {
    const double mu = 0.3;
    const double sigma = 0.7;
    for (double x : {0.2, 0.5, 1.0, 2.0, 4.0}) {
        const double h = 1e-5 * std::max(1.0, std::fabs(x));
        const auto pdfPack = [&](double t) {
            return quantape::math::detail::lognormalPdfD1D2(t, mu, sigma);
        };
        const auto cdfPack = [&](double t) {
            return quantape::math::detail::lognormalCdfD1D2(t, mu, sigma);
        };
        const auto pdf = [&](double t) { return pdfPack(t).value; };
        const auto cdf = [&](double t) { return cdfPack(t).value; };
        const auto pdfD1 = [&](double t) { return pdfPack(t).d1; };
        const auto cdfD1 = [&](double t) { return cdfPack(t).d1; };
        const auto p = pdfPack(x);
        const auto c = cdfPack(x);

        CHECK_CLOSE("pdf pack d1 = d(pdf)/dx", p.d1, centralD1(pdf, x, h), 1e-7);
        CHECK_CLOSE("pdf pack d2 = d(pdf')/dx", p.d2, centralD1(pdfD1, x, h), 1e-7);
        CHECK_CLOSE("cdf pack d1 = d(cdf)/dx", c.d1, centralD1(cdf, x, h), 1e-7);
        CHECK_CLOSE("cdf pack d2 = d(cdf')/dx", c.d2, centralD1(cdfD1, x, h), 1e-7);
        CHECK_CLOSE("cdf pack d1 = pdf", c.d1, p.value, 1e-15);
    }
}

TEST(ExponentialDistribution, closedForms) {
    const quantape::math::Exponential<double> d{2.0};
    for (double x : {0.0, 0.1, 1.0, 5.0}) {
        CHECK_CLOSE("pdf = lambda e^{-lambda x}", d.pdf(x), 2.0 * std::exp(-2.0 * x), kTol);
        CHECK_CLOSE("cdf = 1 - e^{-lambda x}", d.cdf(x), 1.0 - std::exp(-2.0 * x), kTol);
        CHECK_CLOSE("ccdf = e^{-lambda x}", d.ccdf(x), std::exp(-2.0 * x), kTol);
        CHECK_CLOSE("logPdf = ln lambda - lambda x", d.logPdf(x), std::log(2.0) - 2.0 * x, kTol);
        CHECK_CLOSE("logCcdf = -lambda x", d.logCcdf(x), -2.0 * x, kTol);
    }
    CHECK_CLOSE("pdf(0) = lambda", d.pdf(0.0), 2.0, kTol);
    CHECK_CLOSE("logCdf(1) = ln(1 - e^-2)", d.logCdf(1.0), std::log(1.0 - std::exp(-2.0)), kTol);
    CHECK_CLOSE("inv(0.5) = ln 2 / lambda", d.inv(0.5), std::log(2.0) / 2.0, kTol);
}

TEST(ExponentialDistribution, cdfMatchesPdfIntegral) {
    const quantape::math::Exponential<double> d{1.7};
    for (double x : {0.05, 0.3, 1.0, 3.0, 8.0}) {
        const double expected = compositeSimpson([&](double t) { return d.pdf(t); }, 0.0, x, 40000);
        CHECK_CLOSE("cdf(x) = int_0^x pdf", d.cdf(x), expected, 1e-13);
    }
}

TEST(ExponentialDistribution, invRoundTripAndEdges) {
    // x <= 8 keeps lambda x <= 6.4, where cdf(x) is still resolved well below
    // 1 and the double round trip meets 1e-12; for lambda x ~ 16 the p -> 1
    // quantization alone limits the recoverable x to ~2e-10.
    const quantape::math::Exponential<double> d{0.8};
    for (double x : {0.001, 0.1, 1.0, 5.0, 8.0}) {
        const double p = d.cdf(x);
        CHECK_CLOSE("x = inv(cdf(x))", d.inv(p), x, 1e-12);
        CHECK_CLOSE("p = cdf(inv(p))", d.cdf(d.inv(p)), p, 1e-15);
    }
    CHECK_CLOSE("inv(0) = 0", d.inv(0.0), 0.0, 0.0);
    CHECK_CLOSE("inv(-0.5) = 0", d.inv(-0.5), 0.0, 0.0);
    EXPECT_TRUE(std::isinf(d.inv(1.0)));

    CHECK_CLOSE("pdf(-1) = 0", d.pdf(-1.0), 0.0, 0.0);
    CHECK_CLOSE("cdf(-1) = 0", d.cdf(-1.0), 0.0, 0.0);
    CHECK_CLOSE("ccdf(-1) = 1", d.ccdf(-1.0), 1.0, 0.0);
    CHECK_CLOSE("logCcdf(-1) = 0", d.logCcdf(-1.0), 0.0, 0.0);
    EXPECT_TRUE(std::isinf(d.logPdf(-1.0)) && d.logPdf(-1.0) < 0.0);
    EXPECT_TRUE(std::isinf(d.logCdf(-1.0)) && d.logCdf(-1.0) < 0.0);
    EXPECT_TRUE(std::isinf(d.logCdf(0.0)) && d.logCdf(0.0) < 0.0);
    CHECK_CLOSE("ccdf(+inf) = 0", d.ccdf(std::numeric_limits<double>::infinity()), 0.0, 1e-300);

    const double nan = std::numeric_limits<double>::quiet_NaN();
    EXPECT_TRUE(std::isnan(d.pdf(nan)));
    EXPECT_TRUE(std::isnan(d.cdf(nan)));
    EXPECT_TRUE(std::isnan(d.logPdf(nan)));
    EXPECT_TRUE(std::isnan(d.inv(nan)));
}

TEST(ExponentialDistribution, derivativePacksMatchCentralDifferences) {
    const double lambda = 1.7;
    for (double x : {0.1, 0.5, 1.0, 3.0, 6.0}) {
        const double h = 1e-5 * std::max(1.0, std::fabs(x));
        const auto pdfPack = [&](double t) {
            return quantape::math::detail::exponentialPdfD1D2(t, lambda);
        };
        const auto cdfPack = [&](double t) {
            return quantape::math::detail::exponentialCdfD1D2(t, lambda);
        };
        const auto pdf = [&](double t) { return pdfPack(t).value; };
        const auto cdf = [&](double t) { return cdfPack(t).value; };
        const auto pdfD1 = [&](double t) { return pdfPack(t).d1; };
        const auto cdfD1 = [&](double t) { return cdfPack(t).d1; };
        const auto p = pdfPack(x);
        const auto c = cdfPack(x);

        CHECK_CLOSE("pdf pack d1 = d(pdf)/dx", p.d1, centralD1(pdf, x, h), 1e-7);
        CHECK_CLOSE("pdf pack d2 = d(pdf')/dx", p.d2, centralD1(pdfD1, x, h), 1e-7);
        CHECK_CLOSE("cdf pack d1 = d(cdf)/dx", c.d1, centralD1(cdf, x, h), 1e-7);
        CHECK_CLOSE("cdf pack d2 = d(cdf')/dx", c.d2, centralD1(cdfD1, x, h), 1e-7);
        CHECK_CLOSE("cdf pack d1 = pdf", c.d1, p.value, 1e-15);
    }
}

namespace {

/// One row of the mpmath parameter-derivative reference CSV. Columns:
/// dist,mu,sigma,lambda,x,pack,value,g1,g2,g3,h11,h12,h13,h22,h23,h33; the
/// empty optional columns are skipped via `present`. The generic `g1..g3` /
/// `h11..h33` names map to the natural pack fields per family in the tests.
/// Generated by `.venv/bin/python` + mpmath (dps=50, mp.diff with argument
/// tuples) from the value functions and cross-checked against the analytic
/// closed forms to < 1e-30 before writing.
struct ParamDerivRefRow {
    std::string dist;
    std::string pack;
    double mu = 0.0;
    double sigma = 0.0;
    double lambda = 0.0;
    double x = 0.0;
    double entry[10] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
    bool present[10] = {false, false, false, false, false, false, false, false, false, false};
};

std::vector<ParamDerivRefRow> loadParamDerivReference() {
    std::vector<ParamDerivRefRow> rows;
    std::ifstream in(RequireDataFile("distributions/lnexp_ig_paramderiv_reference.csv"));
    std::string line;
    std::getline(in, line); // header
    while (std::getline(in, line)) {
        if (line.empty()) {
            continue;
        }
        std::stringstream ss(line);
        std::vector<std::string> field;
        std::string token;
        while (std::getline(ss, token, ',')) {
            field.push_back(token);
        }
        // std::getline drops trailing empty fields; the optional columns are
        // exactly the empty ones, so pad back to the full width.
        while (field.size() < 16) {
            field.emplace_back();
        }
        if (field.size() != 16) {
            continue;
        }
        ParamDerivRefRow row;
        row.dist = field[0];
        row.mu = field[1].empty() ? 0.0 : std::stod(field[1]);
        row.sigma = field[2].empty() ? 0.0 : std::stod(field[2]);
        row.lambda = field[3].empty() ? 0.0 : std::stod(field[3]);
        row.x = std::stod(field[4]);
        row.pack = field[5];
        for (int i = 0; i < 10; ++i) {
            if (!field[6 + i].empty()) {
                row.entry[i] = std::stod(field[6 + i]);
                row.present[i] = true;
            }
        }
        rows.push_back(row);
    }
    return rows;
}

const char* kPackEntryNames[10] = {"value", "g1",  "g2",  "g3",  "h11",
                                   "h12",   "h13", "h22", "h23", "h33"};

/// Reference gate: first-order entries at 1e-10 rel, second-order at 1e-9,
/// with a small absolute floor (the tables contain a few exact zeros).
void checkReferenceEntry(const std::string& label, double got, double expected, int index) {
    if (index <= 3) {
        CHECK_CLOSE_REL(label + " " + kPackEntryNames[index], got, expected, 1e-10, 1e-13);
    } else {
        CHECK_CLOSE_REL(label + " " + kPackEntryNames[index], got, expected, 1e-9, 1e-12);
    }
}

/// FD gate: gradients at 1e-5 rel / 1e-7 abs of the value's central difference,
/// Hessian entries at 1e-5 rel / 1e-6 abs of the analytic gradient's central
/// difference (the probe worst case across the grids is ~8e-9 scaled).
void checkFdEntry(const std::string& label, double got, double fd, bool secondOrder) {
    if (secondOrder) {
        CHECK_CLOSE_REL(label, got, fd, 1e-5, 1e-6);
    } else {
        CHECK_CLOSE_REL(label, got, fd, 1e-5, 1e-7);
    }
}

std::string rowLabel(const ParamDerivRefRow& row) {
    return row.dist + " " + row.pack + " mu=" + std::to_string(row.mu) +
           " sigma=" + std::to_string(row.sigma) + " lambda=" + std::to_string(row.lambda) +
           " x=" + std::to_string(row.x);
}

} // namespace

TEST(LognormalDistribution, gradHessMatchesMpmathReference) {
    int rows = 0;
    for (const ParamDerivRefRow& row : loadParamDerivReference()) {
        if (row.dist != "ln") {
            continue;
        }
        ++rows;
        const std::string label = rowLabel(row);
        double got[10];
        if (row.pack == "pdf") {
            const auto p = quantape::math::detail::lognormalPdfGradHess(row.mu, row.sigma, row.x);
            got[0] = p.value;
            got[1] = p.gmu;
            got[2] = p.gsig;
            got[3] = p.gx;
            got[4] = p.hmumu;
            got[5] = p.hmusig;
            got[6] = p.hmux;
            got[7] = p.hsigsig;
            got[8] = p.hsigx;
            got[9] = p.hxx;
        } else {
            const auto c = quantape::math::detail::lognormalCdfGradHess(row.mu, row.sigma, row.x);
            got[0] = c.value;
            got[1] = c.gmu;
            got[2] = c.gsig;
            got[3] = c.gx;
            got[4] = c.hmumu;
            got[5] = c.hmusig;
            got[6] = c.hmux;
            got[7] = c.hsigsig;
            got[8] = c.hsigx;
            got[9] = c.hxx;
        }
        for (int i = 0; i < 10; ++i) {
            if (row.present[i]) {
                checkReferenceEntry(label, got[i], row.entry[i], i);
            }
        }
    }
    EXPECT_GT(rows, 0);
}

TEST(ExponentialDistribution, gradHessMatchesMpmathReference) {
    int rows = 0;
    for (const ParamDerivRefRow& row : loadParamDerivReference()) {
        if (row.dist != "ex") {
            continue;
        }
        ++rows;
        const std::string label = rowLabel(row);
        // g1 = gLambda, g2 = gx; h11 = hLL, h12 = hLx, h22 = hxx.
        double got[10] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
        if (row.pack == "pdf") {
            const auto p = quantape::math::detail::exponentialPdfGradHess(row.lambda, row.x);
            got[0] = p.value;
            got[1] = p.gLambda;
            got[2] = p.gx;
            got[4] = p.hLL;
            got[5] = p.hLx;
            got[7] = p.hxx;
        } else {
            const auto c = quantape::math::detail::exponentialCdfGradHess(row.lambda, row.x);
            got[0] = c.value;
            got[1] = c.gLambda;
            got[2] = c.gx;
            got[4] = c.hLL;
            got[5] = c.hLx;
            got[7] = c.hxx;
        }
        for (int i = 0; i < 10; ++i) {
            if (row.present[i]) {
                checkReferenceEntry(label, got[i], row.entry[i], i);
            }
        }
    }
    EXPECT_GT(rows, 0);
}

TEST(LognormalDistribution, gradHessMatchesCentralDifferences) {
    const double mu = 0.3;
    const double sigma = 0.7;
    const double hmu = 1e-5 * std::max(1.0, mu);
    const double hs = 1e-5 * std::max(1.0, sigma);
    const auto P = [](double m, double s, double t) {
        return quantape::math::detail::lognormalPdfGradHess(m, s, t);
    };
    const auto C = [](double m, double s, double t) {
        return quantape::math::detail::lognormalCdfGradHess(m, s, t);
    };
    for (double x : {0.2, 0.5, 1.0, 2.0, 4.0, 10.0}) {
        const double hx = 1e-5 * std::max(1.0, x);
        const auto p = P(mu, sigma, x);
        const auto c = C(mu, sigma, x);
        checkFdEntry("ln pdf gmu vs FD", p.gmu,
                     centralD1([&](double m) { return P(m, sigma, x).value; }, mu, hmu), false);
        checkFdEntry("ln pdf gsig vs FD", p.gsig,
                     centralD1([&](double s) { return P(mu, s, x).value; }, sigma, hs), false);
        checkFdEntry("ln pdf gx vs FD", p.gx,
                     centralD1([&](double t) { return P(mu, sigma, t).value; }, x, hx), false);
        checkFdEntry("ln cdf gmu vs FD", c.gmu,
                     centralD1([&](double m) { return C(m, sigma, x).value; }, mu, hmu), false);
        checkFdEntry("ln cdf gsig vs FD", c.gsig,
                     centralD1([&](double s) { return C(mu, s, x).value; }, sigma, hs), false);
        checkFdEntry("ln cdf gx vs FD", c.gx,
                     centralD1([&](double t) { return C(mu, sigma, t).value; }, x, hx), false);
        checkFdEntry("ln pdf hmumu vs FD", p.hmumu,
                     centralD1([&](double m) { return P(m, sigma, x).gmu; }, mu, hmu), true);
        checkFdEntry("ln pdf hmusig vs FD", p.hmusig,
                     centralD1([&](double s) { return P(mu, s, x).gmu; }, sigma, hs), true);
        checkFdEntry("ln pdf hmusig symmetry", p.hmusig,
                     centralD1([&](double m) { return P(m, sigma, x).gsig; }, mu, hmu), true);
        checkFdEntry("ln pdf hmux vs FD", p.hmux,
                     centralD1([&](double t) { return P(mu, sigma, t).gmu; }, x, hx), true);
        checkFdEntry("ln pdf hmux symmetry", p.hmux,
                     centralD1([&](double m) { return P(m, sigma, x).gx; }, mu, hmu), true);
        checkFdEntry("ln pdf hsigsig vs FD", p.hsigsig,
                     centralD1([&](double s) { return P(mu, s, x).gsig; }, sigma, hs), true);
        checkFdEntry("ln pdf hsigx vs FD", p.hsigx,
                     centralD1([&](double t) { return P(mu, sigma, t).gsig; }, x, hx), true);
        checkFdEntry("ln pdf hsigx symmetry", p.hsigx,
                     centralD1([&](double s) { return P(mu, s, x).gx; }, sigma, hs), true);
        checkFdEntry("ln pdf hxx vs FD", p.hxx,
                     centralD1([&](double t) { return P(mu, sigma, t).gx; }, x, hx), true);
        checkFdEntry("ln cdf hmumu vs FD", c.hmumu,
                     centralD1([&](double m) { return C(m, sigma, x).gmu; }, mu, hmu), true);
        checkFdEntry("ln cdf hmusig vs FD", c.hmusig,
                     centralD1([&](double s) { return C(mu, s, x).gmu; }, sigma, hs), true);
        checkFdEntry("ln cdf hmusig symmetry", c.hmusig,
                     centralD1([&](double m) { return C(m, sigma, x).gsig; }, mu, hmu), true);
        checkFdEntry("ln cdf hmux vs FD", c.hmux,
                     centralD1([&](double t) { return C(mu, sigma, t).gmu; }, x, hx), true);
        checkFdEntry("ln cdf hmux symmetry", c.hmux,
                     centralD1([&](double m) { return C(m, sigma, x).gx; }, mu, hmu), true);
        checkFdEntry("ln cdf hsigsig vs FD", c.hsigsig,
                     centralD1([&](double s) { return C(mu, s, x).gsig; }, sigma, hs), true);
        checkFdEntry("ln cdf hsigx vs FD", c.hsigx,
                     centralD1([&](double t) { return C(mu, sigma, t).gsig; }, x, hx), true);
        checkFdEntry("ln cdf hsigx symmetry", c.hsigx,
                     centralD1([&](double s) { return C(mu, s, x).gx; }, sigma, hs), true);
        checkFdEntry("ln cdf hxx vs FD", c.hxx,
                     centralD1([&](double t) { return C(mu, sigma, t).gx; }, x, hx), true);
    }
}

TEST(ExponentialDistribution, gradHessMatchesCentralDifferences) {
    const double lambda = 1.7;
    const double hl = 1e-5 * std::max(1.0, lambda);
    const auto P = [](double l, double t) {
        return quantape::math::detail::exponentialPdfGradHess(l, t);
    };
    const auto C = [](double l, double t) {
        return quantape::math::detail::exponentialCdfGradHess(l, t);
    };
    for (double x : {0.1, 0.5, 1.0, 3.0, 6.0}) {
        const double hx = 1e-5 * std::max(1.0, x);
        const auto p = P(lambda, x);
        const auto c = C(lambda, x);
        checkFdEntry("ex pdf gLambda vs FD", p.gLambda,
                     centralD1([&](double l) { return P(l, x).value; }, lambda, hl), false);
        checkFdEntry("ex pdf gx vs FD", p.gx,
                     centralD1([&](double t) { return P(lambda, t).value; }, x, hx), false);
        checkFdEntry("ex cdf gLambda vs FD", c.gLambda,
                     centralD1([&](double l) { return C(l, x).value; }, lambda, hl), false);
        checkFdEntry("ex cdf gx vs FD", c.gx,
                     centralD1([&](double t) { return C(lambda, t).value; }, x, hx), false);
        checkFdEntry("ex pdf hLL vs FD", p.hLL,
                     centralD1([&](double l) { return P(l, x).gLambda; }, lambda, hl), true);
        checkFdEntry("ex pdf hLx vs FD", p.hLx,
                     centralD1([&](double t) { return P(lambda, t).gLambda; }, x, hx), true);
        checkFdEntry("ex pdf hLx symmetry", p.hLx,
                     centralD1([&](double l) { return P(l, x).gx; }, lambda, hl), true);
        checkFdEntry("ex pdf hxx vs FD", p.hxx,
                     centralD1([&](double t) { return P(lambda, t).gx; }, x, hx), true);
        checkFdEntry("ex cdf hLL vs FD", c.hLL,
                     centralD1([&](double l) { return C(l, x).gLambda; }, lambda, hl), true);
        checkFdEntry("ex cdf hLx vs FD", c.hLx,
                     centralD1([&](double t) { return C(lambda, t).gLambda; }, x, hx), true);
        checkFdEntry("ex cdf hLx symmetry", c.hLx,
                     centralD1([&](double l) { return C(l, x).gx; }, lambda, hl), true);
        checkFdEntry("ex cdf hxx vs FD", c.hxx,
                     centralD1([&](double t) { return C(lambda, t).gx; }, x, hx), true);
    }
}

TEST(LognormalDistribution, gradHessMatchesNormalInLogSpace) {
    // Y = mu + sigma Z ~ N(mu, sigma^2) => X = exp(Y): cdf_LN(mu,sigma,x) =
    // cdf_N(ln x; mu, sigma) and pdf_LN = pdf_N(ln x; mu, sigma) / x, so the
    // Lognormal (mu, sigma) pack block must equal the Normal pack at y = ln x
    // (divided by x for the density). Checked for small, moderate and large
    // sigma at standardized points x = exp(mu + z sigma).
    for (double sigma : {1e-6, 1e-3, 0.7, 2.5}) {
        const double mu = 0.3;
        for (double z : {-1.5, 0.0, 1.5}) {
            const double x = std::exp(mu + z * sigma);
            const double y = std::log(x);
            const auto lp = quantape::math::detail::lognormalPdfGradHess(mu, sigma, x);
            const auto np = quantape::math::detail::normalPdfGradHess(y, mu, sigma);
            const auto lc = quantape::math::detail::lognormalCdfGradHess(mu, sigma, x);
            const auto nc = quantape::math::detail::normalCdfGradHess(y, mu, sigma);
            const std::string label = "sigma=" + std::to_string(sigma) + " z=" + std::to_string(z);

            CHECK_CLOSE_REL(label + " pdf value", lp.value, np.value / x, 1e-13);
            CHECK_CLOSE_REL(label + " pdf gmu", lp.gmu, np.gmu / x, 1e-13);
            CHECK_CLOSE_REL(label + " pdf gsig", lp.gsig, np.gsig / x, 1e-13);
            CHECK_CLOSE_REL(label + " pdf hmumu", lp.hmumu, np.hmumu / x, 1e-13);
            CHECK_CLOSE_REL(label + " pdf hmusig", lp.hmusig, np.hmusig / x, 1e-13);
            CHECK_CLOSE_REL(label + " pdf hsigsig", lp.hsigsig, np.hsigsig / x, 1e-13);
            CHECK_CLOSE_REL(label + " pdf gx chain", lp.gx * x * x, np.gx - np.value, 1e-12);

            CHECK_CLOSE_REL(label + " cdf value", lc.value, nc.value, 1e-14);
            CHECK_CLOSE_REL(label + " cdf gmu", lc.gmu, nc.gmu, 1e-13);
            CHECK_CLOSE_REL(label + " cdf gsig", lc.gsig, nc.gsig, 1e-13);
            CHECK_CLOSE_REL(label + " cdf hmumu", lc.hmumu, nc.hmumu, 1e-13);
            CHECK_CLOSE_REL(label + " cdf hmusig", lc.hmusig, nc.hmusig, 1e-13);
            CHECK_CLOSE_REL(label + " cdf hsigsig", lc.hsigsig, nc.hsigsig, 1e-13);
            CHECK_CLOSE_REL(label + " cdf gx chain", lc.gx * x, nc.gx, 1e-13);
        }
    }
}

TEST(LognormalDistribution, gradHessValueAnchorsAndEdges) {
    const double mu = 0.2;
    const double sigma = 0.8;
    const double inf = std::numeric_limits<double>::infinity();

    // Value anchors: bit-identical to the existing double kernels.
    for (double x : {0.05, 0.5, 2.0, 7.0}) {
        const auto p = quantape::math::detail::lognormalPdfGradHess(mu, sigma, x);
        const auto c = quantape::math::detail::lognormalCdfGradHess(mu, sigma, x);
        EXPECT_EQ(p.value, quantape::math::detail::lognormalPdfDouble(x, mu, sigma));
        EXPECT_EQ(c.value, quantape::math::detail::lognormalCdfDouble(x, mu, sigma));
        EXPECT_EQ(p.gx, quantape::math::detail::lognormalPdfD1D2<double>(x, mu, sigma).d1);
        EXPECT_EQ(c.hxx, quantape::math::detail::lognormalCdfD1D2<double>(x, mu, sigma).d2);
        // cdf x-derivatives are the pdf values.
        EXPECT_EQ(c.gx, p.value);
        EXPECT_EQ(c.hxx, p.gx);
    }

    // x <= 0: zero pack, cdf value 0.
    for (double x : {-1.0, 0.0}) {
        const auto p = quantape::math::detail::lognormalPdfGradHess(mu, sigma, x);
        const auto c = quantape::math::detail::lognormalCdfGradHess(mu, sigma, x);
        for (double v :
             {p.value, p.gmu, p.gsig, p.gx, p.hmumu, p.hmusig, p.hmux, p.hsigsig, p.hsigx, p.hxx,
              c.value, c.gmu, c.gsig, c.gx, c.hmumu, c.hmusig, c.hmux, c.hsigsig, c.hsigx, c.hxx}) {
            EXPECT_EQ(v, 0.0);
        }
    }

    // x = +inf: pdf pack vanishes, cdf saturates with zero partials.
    const auto pInf = quantape::math::detail::lognormalPdfGradHess(mu, sigma, inf);
    for (double v : {pInf.value, pInf.gmu, pInf.gsig, pInf.gx, pInf.hmumu, pInf.hmusig, pInf.hmux,
                     pInf.hsigsig, pInf.hsigx, pInf.hxx}) {
        EXPECT_EQ(v, 0.0);
    }
    const auto cInf = quantape::math::detail::lognormalCdfGradHess(mu, sigma, inf);
    EXPECT_EQ(cInf.value, 1.0);
    for (double v : {cInf.gmu, cInf.gsig, cInf.gx, cInf.hmumu, cInf.hmusig, cInf.hmux, cInf.hsigsig,
                     cInf.hsigx, cInf.hxx}) {
        EXPECT_EQ(v, 0.0);
    }

    // Tiny/huge finite x stay finite (density underflows, partials with it).
    for (double x : {1e-300, 1e-12, 1e12, 1e300}) {
        const auto p = quantape::math::detail::lognormalPdfGradHess(mu, sigma, x);
        const auto c = quantape::math::detail::lognormalCdfGradHess(mu, sigma, x);
        for (double v :
             {p.value, p.gmu, p.gsig, p.gx, p.hmumu, p.hmusig, p.hmux, p.hsigsig, p.hsigx, p.hxx,
              c.value, c.gmu, c.gsig, c.gx, c.hmumu, c.hmusig, c.hmux, c.hsigsig, c.hsigx, c.hxx}) {
            EXPECT_TRUE(std::isfinite(v));
        }
    }

    // NaN propagates through every entry, including the boundary branches.
    const double nan = std::numeric_limits<double>::quiet_NaN();
    const auto pNan = quantape::math::detail::lognormalPdfGradHess(mu, sigma, nan);
    const auto pNanMu = quantape::math::detail::lognormalPdfGradHess(nan, sigma, 1.0);
    const auto cNan = quantape::math::detail::lognormalCdfGradHess(mu, sigma, nan);
    for (double v : {pNan.value, pNan.gmu,     pNan.gsig,  pNan.gx,  pNan.hmumu,   pNan.hmusig,
                     pNan.hmux,  pNan.hsigsig, pNan.hsigx, pNan.hxx, pNanMu.value, pNanMu.gx,
                     cNan.value, cNan.gmu,     cNan.gsig,  cNan.gx,  cNan.hmumu,   cNan.hmusig,
                     cNan.hmux,  cNan.hsigsig, cNan.hsigx, cNan.hxx}) {
        EXPECT_TRUE(std::isnan(v));
    }
}

TEST(ExponentialDistribution, gradHessClosedFormsAndEdges) {
    const double lambda = 2.0;
    const double inf = std::numeric_limits<double>::infinity();

    // x = 0 right-hand branch, all closed forms as documented.
    const auto p0 = quantape::math::detail::exponentialPdfGradHess(lambda, 0.0);
    CHECK_CLOSE("pdf(0)", p0.value, lambda, 0.0);
    CHECK_CLOSE("pdf gLambda(0)", p0.gLambda, 1.0, 0.0);
    CHECK_CLOSE("pdf gx(0)", p0.gx, -lambda * lambda, 0.0);
    CHECK_CLOSE("pdf hLL(0)", p0.hLL, 0.0, 0.0);
    CHECK_CLOSE("pdf hLx(0)", p0.hLx, -2.0 * lambda, 0.0);
    CHECK_CLOSE("pdf hxx(0)", p0.hxx, lambda * lambda * lambda, 0.0);
    const auto c0 = quantape::math::detail::exponentialCdfGradHess(lambda, 0.0);
    CHECK_CLOSE("cdf(0)", c0.value, 0.0, 0.0);
    CHECK_CLOSE("cdf gLambda(0)", c0.gLambda, 0.0, 0.0);
    CHECK_CLOSE("cdf gx(0)", c0.gx, lambda, 0.0);
    CHECK_CLOSE("cdf hLL(0)", c0.hLL, 0.0, 0.0);
    CHECK_CLOSE("cdf hLx(0)", c0.hLx, 1.0, 0.0);
    CHECK_CLOSE("cdf hxx(0)", c0.hxx, -lambda * lambda, 0.0);

    // Value anchors against the existing kernels.
    for (double x : {0.2, 1.0, 4.0}) {
        const auto p = quantape::math::detail::exponentialPdfGradHess(lambda, x);
        const auto c = quantape::math::detail::exponentialCdfGradHess(lambda, x);
        EXPECT_EQ(p.value, quantape::math::detail::exponentialPdfDouble(x, lambda));
        EXPECT_EQ(c.value, quantape::math::detail::exponentialCdfDouble(x, lambda));
        EXPECT_EQ(c.gx, p.value);
        EXPECT_EQ(c.hxx, p.gx);
        // pdf = lambda * ccdf and its lambda derivative identity.
        CHECK_CLOSE("pdf = lambda ccdf", p.value, lambda * std::exp(-lambda * x), 1e-15);
        CHECK_CLOSE("cdf gLambda = x ccdf", c.gLambda, x * std::exp(-lambda * x), 1e-15);
    }

    // x < 0: zero pack; x = +inf: pdf zero, cdf saturates.
    for (double x : {-1.0, -1e-3}) {
        const auto p = quantape::math::detail::exponentialPdfGradHess(lambda, x);
        const auto c = quantape::math::detail::exponentialCdfGradHess(lambda, x);
        for (double v : {p.value, p.gLambda, p.gx, p.hLL, p.hLx, p.hxx, c.value, c.gLambda, c.gx,
                         c.hLL, c.hLx, c.hxx}) {
            EXPECT_EQ(v, 0.0);
        }
    }
    const auto pInf = quantape::math::detail::exponentialPdfGradHess(lambda, inf);
    for (double v : {pInf.value, pInf.gLambda, pInf.gx, pInf.hLL, pInf.hLx, pInf.hxx}) {
        EXPECT_EQ(v, 0.0);
    }
    const auto cInf = quantape::math::detail::exponentialCdfGradHess(lambda, inf);
    EXPECT_EQ(cInf.value, 1.0);
    for (double v : {cInf.gLambda, cInf.gx, cInf.hLL, cInf.hLx, cInf.hxx}) {
        EXPECT_EQ(v, 0.0);
    }

    // NaN propagates.
    const double nan = std::numeric_limits<double>::quiet_NaN();
    const auto pNan = quantape::math::detail::exponentialPdfGradHess(lambda, nan);
    const auto pNanL = quantape::math::detail::exponentialPdfGradHess(nan, 1.0);
    const auto cNan = quantape::math::detail::exponentialCdfGradHess(lambda, nan);
    for (double v : {pNan.value, pNan.gLambda, pNan.gx, pNan.hLL, pNan.hLx, pNan.hxx, pNanL.value,
                     pNanL.gLambda, pNanL.gx, pNanL.hLL, pNanL.hLx, pNanL.hxx, cNan.value,
                     cNan.gLambda, cNan.gx, cNan.hLL, cNan.hLx, cNan.hxx}) {
        EXPECT_TRUE(std::isnan(v));
    }
}

// The lognormal quantile is exp(mu + sigma * normalInv(p)); it is finite over
// the whole subnormal range of p. Reference: Boost
// `quantile(lognormal_distribution<cpp_bin_float_100>{0,1}, p)`.
TEST(LognormalDistribution, invResolvesSubnormalTail) {
    const quantape::math::Lognormal<double> d{0.0, 1.0};
    CHECK_CLOSE_REL("inv(5e-324)", d.inv(5e-324), 1.96706181266275515367e-17, 1e-12);
    CHECK_CLOSE_REL("inv(7e-311)", d.inv(7e-311), 4.35543262402098417728e-17, 1e-12);
}
