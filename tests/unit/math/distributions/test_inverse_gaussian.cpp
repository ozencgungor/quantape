// test_inverse_gaussian.cpp — Inverse Gaussian (Wald) distribution gates.
//
// The cdf reference is a local composite-Simpson integration of the pdf over
// (0, x]; the pdf has an integrable-type rise near the origin (exponent
// -lambda / (2x) kills it super-exponentially), so a plain [0, x] grid is
// accurate without a transformed variable. Tolerance is 1e-10 in mid-range;
// the far-right tail is checked through cdf -> 1 and a segmented integral over
// (0, 100] rather than a per-point cdf comparison. The derivative packs are
// checked with central differences at h = 1e-5 max(1, |x|): d1 against the
// value and d2 against the analytic d1 (a direct second difference at this h
// would be cancellation-limited to ~eps |f| / h^2 ~ 1e-5).
//
// The parameter gradient/Hessian packs (`inverseGaussianPdfGradHess`,
// `inverseGaussianCdfGradHess`) are gated four ways: (1) mpmath high-precision
// references in tests/data/distributions/lnexp_ig_paramderiv_reference.csv,
// generated with `.venv/bin/python` + mpmath at dps=50 by numerically
// differentiating the value functions (`mp.diff` on argument tuples) and
// cross-checked against the analytic closed forms to < 1e-30; (2) central
// differences of the analytic pack values/gradients (both directions of each
// cross entry, so symmetry is exercised); (3) the exact scaling identity
// X ~ IG(mu, lambda) => c X ~ IG(c mu, c lambda); (4) value anchors against the
// existing kernels plus the documented overflow regime (2 lambda/mu >~ 709,
// where the cdf's exp(2 lambda/mu) factor overflows) and the lambda -> infinity
// normal limit.

#include "quantape/math/Distributions/InverseGaussian.h"
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

/// Composite Simpson over the consecutive segments edges[0..count-1]; the
/// segment split keeps the grid fine where the pdf rises (beats one uniform
/// grid over [0, 100] by ~9 orders of magnitude at the same evaluation count).
template <class F>
double compositeSimpsonSegments(const F& f, const double* edges, int count, long n) {
    double total = 0.0;
    for (int k = 0; k + 1 < count; ++k) {
        total += compositeSimpson(f, edges[k], edges[k + 1], n);
    }
    return total;
}

template <class F>
double centralD1(const F& f, double x, double h) {
    return (f(x + h) - f(x - h)) / (2.0 * h);
}

} // namespace

TEST(InverseGaussianDistribution, cdfMatchesPdfIntegral) {
    const quantape::math::InverseGaussian<double> d{1.0, 1.0};
    for (double x : {0.5, 1.0, 2.0, 4.0}) {
        const double expected =
            compositeSimpson([&](double t) { return d.pdf(t); }, 0.0, x, 200000);
        CHECK_CLOSE("IG(1,1) cdf(x) = int_0^x pdf", d.cdf(x), expected, 1e-10);
    }

    const quantape::math::InverseGaussian<double> d2{2.0, 4.0};
    for (double x : {0.5, 1.0, 2.0, 4.0, 8.0}) {
        const double expected =
            compositeSimpson([&](double t) { return d2.pdf(t); }, 0.0, x, 200000);
        CHECK_CLOSE("IG(2,4) cdf(x) = int_0^x pdf", d2.cdf(x), expected, 1e-10);
    }
}

TEST(InverseGaussianDistribution, pdfIntegratesToOne) {
    const quantape::math::InverseGaussian<double> d{1.0, 1.0};
    const double edges[] = {0.0, 0.25, 1.0, 3.0, 10.0, 100.0};
    const double total =
        compositeSimpsonSegments([&](double t) { return d.pdf(t); }, edges, 6, 200000);
    CHECK_CLOSE("int_0^100 pdf = 1", total, 1.0, 1e-10);
}

TEST(InverseGaussianDistribution, cdfLimitsAndMonotonicity) {
    const quantape::math::InverseGaussian<double> d{1.0, 1.0};
    CHECK_CLOSE("cdf(0) = 0", d.cdf(0.0), 0.0, 0.0);
    CHECK_CLOSE("cdf(-1) = 0", d.cdf(-1.0), 0.0, 0.0);
    CHECK_CLOSE("cdf(+inf) = 1", d.cdf(std::numeric_limits<double>::infinity()), 1.0, 1e-15);
    CHECK_CLOSE("cdf(1e4) = 1", d.cdf(1e4), 1.0, 1e-15);
    CHECK_CLOSE("cdf(1) regression value", d.cdf(1.0), 0.6681020012231705, 1e-12);

    double previous = 0.0;
    for (double x = 0.05; x <= 8.0; x += 0.05) {
        const double current = d.cdf(x);
        EXPECT_GE(current + 1e-14, previous) << "cdf decreased at x = " << x;
        previous = current;
    }
}

TEST(InverseGaussianDistribution, scalingIdentity) {
    // X ~ IG(mu, lambda) => c X ~ IG(c mu, c lambda).
    const quantape::math::InverseGaussian<double> base{1.0, 2.0};
    const quantape::math::InverseGaussian<double> scaled{3.0, 6.0};
    for (double x : {0.5, 1.0, 3.0, 9.0}) {
        CHECK_CLOSE("cdf_{c mu, c lambda}(x) = cdf_{mu, lambda}(x / c)", scaled.cdf(x),
                    base.cdf(x / 3.0), 1e-13);
    }
}

TEST(InverseGaussianDistribution, derivativePacksMatchCentralDifferences) {
    const double mu = 1.0;
    const double lambda = 2.0;
    for (double x : {0.5, 0.7, 1.0, 1.5, 3.0, 6.0}) {
        const double h = 1e-5 * std::max(1.0, std::fabs(x));
        const auto pdfPack = [&](double t) {
            return quantape::math::detail::inverseGaussianPdfD1D2(t, mu, lambda);
        };
        const auto cdfPack = [&](double t) {
            return quantape::math::detail::inverseGaussianCdfD1D2(t, mu, lambda);
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

TEST(InverseGaussianDistribution, edgesAndTails) {
    const quantape::math::InverseGaussian<double> d{1.0, 1.0};

    CHECK_CLOSE("pdf(0) = 0", d.pdf(0.0), 0.0, 0.0);
    CHECK_CLOSE("pdf(-1) = 0", d.pdf(-1.0), 0.0, 0.0);
    CHECK_CLOSE("ccdf(0) = 1", d.ccdf(0.0), 1.0, 0.0);
    EXPECT_TRUE(std::isinf(d.logPdf(0.0)) && d.logPdf(0.0) < 0.0);
    CHECK_CLOSE("pdf(+inf) = 0", d.pdf(std::numeric_limits<double>::infinity()), 0.0, 0.0);
    CHECK_CLOSE("ccdf(1) = 1 - cdf(1)", d.ccdf(1.0), 1.0 - d.cdf(1.0), kTol);
    CHECK_CLOSE("logPdf = ln pdf", d.logPdf(1.5), std::log(d.pdf(1.5)), kTol);

    const double nan = std::numeric_limits<double>::quiet_NaN();
    EXPECT_TRUE(std::isnan(d.pdf(nan)));
    EXPECT_TRUE(std::isnan(d.cdf(nan)));
    EXPECT_TRUE(std::isnan(d.logPdf(nan)));
    EXPECT_TRUE(std::isnan(d.ccdf(nan)));
}

namespace {

/// One row of the shared mpmath parameter-derivative reference CSV; the IG
/// rows map g1 = gmu, g2 = gLambda, g3 = gx and h11 = hmumu,
/// h12 = hmuLambda, h13 = hmux, h22 = hLL, h23 = hLx, h33 = hxx.
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

} // namespace

TEST(InverseGaussianDistribution, gradHessMatchesMpmathReference) {
    int rows = 0;
    for (const ParamDerivRefRow& row : loadParamDerivReference()) {
        if (row.dist != "ig") {
            continue;
        }
        ++rows;
        const std::string label = "ig " + row.pack + " mu=" + std::to_string(row.mu) +
                                  " lambda=" + std::to_string(row.lambda) +
                                  " x=" + std::to_string(row.x);
        double got[10];
        if (row.pack == "pdf") {
            const auto p =
                quantape::math::detail::inverseGaussianPdfGradHess(row.mu, row.lambda, row.x);
            got[0] = p.value;
            got[1] = p.gmu;
            got[2] = p.gLambda;
            got[3] = p.gx;
            got[4] = p.hmumu;
            got[5] = p.hmuLambda;
            got[6] = p.hmux;
            got[7] = p.hLL;
            got[8] = p.hLx;
            got[9] = p.hxx;
        } else {
            const auto c =
                quantape::math::detail::inverseGaussianCdfGradHess(row.mu, row.lambda, row.x);
            got[0] = c.value;
            got[1] = c.gmu;
            got[2] = c.gLambda;
            got[3] = c.gx;
            got[4] = c.hmumu;
            got[5] = c.hmuLambda;
            got[6] = c.hmux;
            got[7] = c.hLL;
            got[8] = c.hLx;
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

TEST(InverseGaussianDistribution, gradHessMatchesCentralDifferences) {
    struct Point {
        double mu;
        double lambda;
        double x;
    };
    const auto P = [](double m, double l, double t) {
        return quantape::math::detail::inverseGaussianPdfGradHess(m, l, t);
    };
    const auto C = [](double m, double l, double t) {
        return quantape::math::detail::inverseGaussianCdfGradHess(m, l, t);
    };
    for (const Point pt :
         {Point{1.0, 1.0, 0.5}, Point{1.0, 1.0, 2.0}, Point{2.0, 4.0, 1.0}, Point{2.0, 4.0, 5.0},
          Point{0.7, 0.3, 1.4}, Point{1.5, 2.5, 4.0}, Point{0.5, 5.0, 0.6}}) {
        const double hm = 1e-5 * std::max(1.0, pt.mu);
        const double hl = 1e-5 * std::max(1.0, pt.lambda);
        const double hx = 1e-5 * std::max(1.0, pt.x);
        const double mu = pt.mu;
        const double lambda = pt.lambda;
        const double x = pt.x;
        const auto p = P(mu, lambda, x);
        const auto c = C(mu, lambda, x);
        checkFdEntry("ig pdf gmu vs FD", p.gmu,
                     centralD1([&](double m) { return P(m, lambda, x).value; }, mu, hm), false);
        checkFdEntry("ig pdf gLambda vs FD", p.gLambda,
                     centralD1([&](double l) { return P(mu, l, x).value; }, lambda, hl), false);
        checkFdEntry("ig pdf gx vs FD", p.gx,
                     centralD1([&](double t) { return P(mu, lambda, t).value; }, x, hx), false);
        checkFdEntry("ig cdf gmu vs FD", c.gmu,
                     centralD1([&](double m) { return C(m, lambda, x).value; }, mu, hm), false);
        checkFdEntry("ig cdf gLambda vs FD", c.gLambda,
                     centralD1([&](double l) { return C(mu, l, x).value; }, lambda, hl), false);
        checkFdEntry("ig cdf gx vs FD", c.gx,
                     centralD1([&](double t) { return C(mu, lambda, t).value; }, x, hx), false);
        checkFdEntry("ig pdf hmumu vs FD", p.hmumu,
                     centralD1([&](double m) { return P(m, lambda, x).gmu; }, mu, hm), true);
        checkFdEntry("ig pdf hmuLambda vs FD", p.hmuLambda,
                     centralD1([&](double l) { return P(mu, l, x).gmu; }, lambda, hl), true);
        checkFdEntry("ig pdf hmuLambda symmetry", p.hmuLambda,
                     centralD1([&](double m) { return P(m, lambda, x).gLambda; }, mu, hm), true);
        checkFdEntry("ig pdf hmux vs FD", p.hmux,
                     centralD1([&](double t) { return P(mu, lambda, t).gmu; }, x, hx), true);
        checkFdEntry("ig pdf hmux symmetry", p.hmux,
                     centralD1([&](double m) { return P(m, lambda, x).gx; }, mu, hm), true);
        checkFdEntry("ig pdf hLL vs FD", p.hLL,
                     centralD1([&](double l) { return P(mu, l, x).gLambda; }, lambda, hl), true);
        checkFdEntry("ig pdf hLx vs FD", p.hLx,
                     centralD1([&](double t) { return P(mu, lambda, t).gLambda; }, x, hx), true);
        checkFdEntry("ig pdf hLx symmetry", p.hLx,
                     centralD1([&](double l) { return P(mu, l, x).gx; }, lambda, hl), true);
        checkFdEntry("ig pdf hxx vs FD", p.hxx,
                     centralD1([&](double t) { return P(mu, lambda, t).gx; }, x, hx), true);
        checkFdEntry("ig cdf hmumu vs FD", c.hmumu,
                     centralD1([&](double m) { return C(m, lambda, x).gmu; }, mu, hm), true);
        checkFdEntry("ig cdf hmuLambda vs FD", c.hmuLambda,
                     centralD1([&](double l) { return C(mu, l, x).gmu; }, lambda, hl), true);
        checkFdEntry("ig cdf hmuLambda symmetry", c.hmuLambda,
                     centralD1([&](double m) { return C(m, lambda, x).gLambda; }, mu, hm), true);
        checkFdEntry("ig cdf hmux vs FD", c.hmux,
                     centralD1([&](double t) { return C(mu, lambda, t).gmu; }, x, hx), true);
        checkFdEntry("ig cdf hmux symmetry", c.hmux,
                     centralD1([&](double m) { return C(m, lambda, x).gx; }, mu, hm), true);
        checkFdEntry("ig cdf hLL vs FD", c.hLL,
                     centralD1([&](double l) { return C(mu, l, x).gLambda; }, lambda, hl), true);
        checkFdEntry("ig cdf hLx vs FD", c.hLx,
                     centralD1([&](double t) { return C(mu, lambda, t).gLambda; }, x, hx), true);
        checkFdEntry("ig cdf hLx symmetry", c.hLx,
                     centralD1([&](double l) { return C(mu, l, x).gx; }, lambda, hl), true);
        checkFdEntry("ig cdf hxx vs FD", c.hxx,
                     centralD1([&](double t) { return C(mu, lambda, t).gx; }, x, hx), true);
    }
}

TEST(InverseGaussianDistribution, gradHessScalingIdentity) {
    // X ~ IG(mu, lambda) => c X ~ IG(c mu, c lambda): the cdf at c x is the cdf
    // at x (value and x rows equal, parameter rows divided by c, second rows by
    // c^2); the density at c x is the density at x divided by c (parameter rows
    // c^2, second rows c^3, x rows c^2).
    const double c = 3.0;
    const double mu = 1.0;
    const double lambda = 2.0;
    for (double x : {0.5, 1.0, 3.0, 9.0}) {
        const auto b = quantape::math::detail::inverseGaussianPdfGradHess(mu, lambda, x);
        const auto s =
            quantape::math::detail::inverseGaussianPdfGradHess(c * mu, c * lambda, c * x);
        const auto bc = quantape::math::detail::inverseGaussianCdfGradHess(mu, lambda, x);
        const auto sc =
            quantape::math::detail::inverseGaussianCdfGradHess(c * mu, c * lambda, c * x);
        const std::string label = "x=" + std::to_string(x);

        CHECK_CLOSE_REL(label + " pdf value", s.value, b.value / c, 1e-12, 1e-14);
        CHECK_CLOSE_REL(label + " pdf gmu", s.gmu, b.gmu / (c * c), 1e-12, 1e-14);
        CHECK_CLOSE_REL(label + " pdf gLambda", s.gLambda, b.gLambda / (c * c), 1e-12, 1e-14);
        CHECK_CLOSE_REL(label + " pdf gx", s.gx, b.gx / (c * c), 1e-12, 1e-14);
        CHECK_CLOSE_REL(label + " pdf hmumu", s.hmumu, b.hmumu / (c * c * c), 1e-12, 1e-14);
        CHECK_CLOSE_REL(label + " pdf hmuLambda", s.hmuLambda, b.hmuLambda / (c * c * c), 1e-12,
                        1e-14);
        CHECK_CLOSE_REL(label + " pdf hmux", s.hmux, b.hmux / (c * c * c), 1e-12, 1e-14);
        CHECK_CLOSE_REL(label + " pdf hLL", s.hLL, b.hLL / (c * c * c), 1e-12, 1e-14);
        CHECK_CLOSE_REL(label + " pdf hLx", s.hLx, b.hLx / (c * c * c), 1e-12, 1e-14);
        CHECK_CLOSE_REL(label + " pdf hxx", s.hxx, b.hxx / (c * c * c), 1e-12, 1e-14);

        CHECK_CLOSE_REL(label + " cdf value", sc.value, bc.value, 1e-12, 1e-14);
        CHECK_CLOSE_REL(label + " cdf gmu", sc.gmu, bc.gmu / c, 1e-12, 1e-14);
        CHECK_CLOSE_REL(label + " cdf gLambda", sc.gLambda, bc.gLambda / c, 1e-12, 1e-14);
        CHECK_CLOSE_REL(label + " cdf gx", sc.gx, bc.gx / c, 1e-12, 1e-14);
        CHECK_CLOSE_REL(label + " cdf hmumu", sc.hmumu, bc.hmumu / (c * c), 1e-12, 1e-14);
        CHECK_CLOSE_REL(label + " cdf hmuLambda", sc.hmuLambda, bc.hmuLambda / (c * c), 1e-12,
                        1e-14);
        CHECK_CLOSE_REL(label + " cdf hmux", sc.hmux, bc.hmux / (c * c), 1e-12, 1e-14);
        CHECK_CLOSE_REL(label + " cdf hLL", sc.hLL, bc.hLL / (c * c), 1e-12, 1e-14);
        CHECK_CLOSE_REL(label + " cdf hLx", sc.hLx, bc.hLx / (c * c), 1e-12, 1e-14);
        CHECK_CLOSE_REL(label + " cdf hxx", sc.hxx, bc.hxx / (c * c), 1e-12, 1e-14);
    }
}

TEST(InverseGaussianDistribution, normalLimitAtLargeLambda) {
    // IG(mu, lambda) -> N(mu, mu^3 / lambda) as lambda grows; the convergence
    // is Edgeworth-slow (O(1/sqrt(lambda)), e.g. the skewness 3/sqrt(lambda)),
    // so the gates are percent-level and only cover the value and the density
    // (parameter partials of the limit distribution have an extra sigma(mu,
    // lambda) chain contribution and do not converge to the plain Normal
    // partials). lambda stays below the exp(2 lambda/mu) overflow (~709 here).
    const double mu = 1.0;
    for (double lambda : {100.0, 300.0}) {
        const double sigma = std::sqrt(mu * mu * mu / lambda);
        const bool tighter = lambda > 200.0;
        const double valueTol = tighter ? 4e-2 : 6e-2;
        const double densityTol = tighter ? 9e-2 : 1.5e-1;
        for (double z : {-1.0, 0.0, 1.0}) {
            const double x = mu + z * sigma;
            const auto igc = quantape::math::detail::inverseGaussianCdfGradHess(mu, lambda, x);
            const auto igp = quantape::math::detail::inverseGaussianPdfGradHess(mu, lambda, x);
            const auto nc = quantape::math::detail::normalCdfGradHess(x, mu, sigma);
            const auto np = quantape::math::detail::normalPdfGradHess(x, mu, sigma);
            const std::string label =
                "lambda=" + std::to_string(lambda) + " z=" + std::to_string(z);
            CHECK_CLOSE_REL(label + " cdf -> normal cdf", igc.value, nc.value, valueTol, 1e-3);
            CHECK_CLOSE_REL(label + " gx -> normal pdf", igp.value, np.value, densityTol, 1e-6);
        }
    }
}

TEST(InverseGaussianDistribution, gradHessValueAnchorsAndOverflow) {
    const double inf = std::numeric_limits<double>::infinity();

    // Value anchors: bit-identical to the existing double kernels, and the cdf
    // x rows coincide with the pdf pack value/gx.
    for (const double x : {0.5, 1.0, 2.0, 5.0}) {
        const double mu = 1.0;
        const double lambda = 2.0;
        const auto p = quantape::math::detail::inverseGaussianPdfGradHess(mu, lambda, x);
        const auto c = quantape::math::detail::inverseGaussianCdfGradHess(mu, lambda, x);
        EXPECT_EQ(p.value, quantape::math::detail::inverseGaussianPdfDouble(x, mu, lambda));
        EXPECT_EQ(c.value, quantape::math::detail::inverseGaussianCdfDouble(x, mu, lambda));
        EXPECT_EQ(c.gx, p.value);
        EXPECT_EQ(c.hxx, p.gx);
    }

    // Closed forms at x = mu: L_mu = 0, so gmu = 0; L_lambda = 1/(2 lambda),
    // L_x = -3/(2 mu).
    {
        const double mu = 2.0;
        const double lambda = 5.0;
        const auto p = quantape::math::detail::inverseGaussianPdfGradHess(mu, lambda, mu);
        CHECK_CLOSE("IG pdf at mu gmu", p.gmu, 0.0, 0.0);
        CHECK_CLOSE_REL("IG pdf at mu gLambda", p.gLambda, p.value / (2.0 * lambda), 1e-15);
        CHECK_CLOSE_REL("IG pdf at mu gx", p.gx, -1.5 * p.value / mu, 1e-15);
        CHECK_CLOSE_REL("IG pdf at mu hmux", p.hmux, p.value * lambda / (mu * mu * mu), 1e-15);
        CHECK_CLOSE_REL("IG pdf at mu hmuLambda", p.hmuLambda, 0.0, 1e-15);
        CHECK_CLOSE_REL("IG pdf at mu hLL", p.hLL, -p.value / (4.0 * lambda * lambda), 1e-15);
    }

    // x <= 0: zero pack; x = +inf: pdf zero pack, cdf saturates.
    for (double x : {-1.0, 0.0}) {
        const auto p = quantape::math::detail::inverseGaussianPdfGradHess(1.0, 1.0, x);
        const auto c = quantape::math::detail::inverseGaussianCdfGradHess(1.0, 1.0, x);
        for (double v :
             {p.value, p.gmu, p.gLambda, p.gx, p.hmumu, p.hmuLambda, p.hmux, p.hLL, p.hLx, p.hxx,
              c.value, c.gmu, c.gLambda, c.gx, c.hmumu, c.hmuLambda, c.hmux, c.hLL, c.hLx, c.hxx}) {
            EXPECT_EQ(v, 0.0);
        }
    }
    const auto pInf = quantape::math::detail::inverseGaussianPdfGradHess(1.0, 1.0, inf);
    for (double v : {pInf.value, pInf.gmu, pInf.gLambda, pInf.gx, pInf.hmumu, pInf.hmuLambda,
                     pInf.hmux, pInf.hLL, pInf.hLx, pInf.hxx}) {
        EXPECT_EQ(v, 0.0);
    }
    const auto cInf = quantape::math::detail::inverseGaussianCdfGradHess(1.0, 1.0, inf);
    EXPECT_EQ(cInf.value, 1.0);
    for (double v : {cInf.gmu, cInf.gLambda, cInf.gx, cInf.hmumu, cInf.hmuLambda, cInf.hmux,
                     cInf.hLL, cInf.hLx, cInf.hxx}) {
        EXPECT_EQ(v, 0.0);
    }

    // Tiny/huge finite x: the deep-underflow guards return finite zeros.
    for (double x : {1e-300, 1e-12, 1e12, 1e300}) {
        const auto p = quantape::math::detail::inverseGaussianPdfGradHess(1.0, 1.0, x);
        const auto c = quantape::math::detail::inverseGaussianCdfGradHess(1.0, 1.0, x);
        for (double v :
             {p.value, p.gmu, p.gLambda, p.gx, p.hmumu, p.hmuLambda, p.hmux, p.hLL, p.hLx, p.hxx,
              c.value, c.gmu, c.gLambda, c.gx, c.hmumu, c.hmuLambda, c.hmux, c.hLL, c.hLx, c.hxx}) {
            EXPECT_TRUE(std::isfinite(v));
        }
    }

    // exp(2 lambda/mu) overflow regime (2 lambda/mu = 800 here > ~709): the
    // cdf value is the anchored kernel value (inf * 0 -> NaN in this corner,
    // inf where the second Phi term is still subnormal), the x rows stay
    // finite because they are anchored on the pdf, and the mixed partials
    // inherit the overflow. The pdf is elementary and stays finite.
    {
        const double mu = 1.0;
        const double lambda = 400.0;
        const auto c = quantape::math::detail::inverseGaussianCdfGradHess(mu, lambda, mu);
        const quantape::math::InverseGaussian<double> d{mu, lambda};
        const double anchor = d.cdf(mu);
        EXPECT_TRUE((std::isnan(c.value) && std::isnan(anchor)) || c.value == anchor);
        EXPECT_TRUE(std::isnan(c.value)); // documented corner: inf * 0
        EXPECT_TRUE(std::isfinite(c.gx));
        EXPECT_TRUE(std::isfinite(c.hxx));
        EXPECT_TRUE(std::isnan(c.hmumu) || std::isinf(c.hmumu));
        EXPECT_TRUE(std::isnan(c.hLL) || std::isinf(c.hLL));

        const auto p = quantape::math::detail::inverseGaussianPdfGradHess(mu, lambda, mu);
        for (double v :
             {p.value, p.gmu, p.gLambda, p.gx, p.hmumu, p.hmuLambda, p.hmux, p.hLL, p.hLx, p.hxx}) {
            EXPECT_TRUE(std::isfinite(v));
        }
        EXPECT_EQ(p.value, d.pdf(mu));
    }

    // NaN propagates through every entry.
    const double nan = std::numeric_limits<double>::quiet_NaN();
    const auto pNan = quantape::math::detail::inverseGaussianPdfGradHess(1.0, 1.0, nan);
    const auto cNan = quantape::math::detail::inverseGaussianCdfGradHess(1.0, 1.0, nan);
    const auto pNanMu = quantape::math::detail::inverseGaussianPdfGradHess(nan, 1.0, 1.0);
    const auto pNanLambda = quantape::math::detail::inverseGaussianPdfGradHess(1.0, nan, 1.0);
    for (double v :
         {pNan.value,   pNan.gmu, pNan.gLambda, pNan.gx,        pNan.hmumu,       pNan.hmuLambda,
          pNan.hmux,    pNan.hLL, pNan.hLx,     pNan.hxx,       cNan.value,       cNan.gmu,
          cNan.gLambda, cNan.gx,  cNan.hmumu,   cNan.hmuLambda, cNan.hmux,        cNan.hLL,
          cNan.hLx,     cNan.hxx, pNanMu.value, pNanMu.gx,      pNanLambda.value, pNanLambda.gmu}) {
        EXPECT_TRUE(std::isnan(v));
    }
}
