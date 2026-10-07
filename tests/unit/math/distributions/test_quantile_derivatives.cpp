// test_quantile_derivatives.cpp — Workstream F: quantile derivative packs
// (`invGradHess`) for every distribution with an `inv`, plus the newly added
// InverseGaussian and NoncentralChiSquare quantiles.
//
// Reference values: tests/data/distributions/quantile_reference.csv, generated
// with .venv/bin/python + mpmath 1.4.1 at mp.dps = 50 (dps used only for the
// solve; values printed to 25 significant digits). Generator method: for each
// family/parameter/p row the quantile Q is the deterministic high-precision
// bisection of the CDF (upper bracket doubled until F(hi) >= p, lowered until
// F(lo) <= p, then ~500 halvings), and the derivatives come from the exact
// implicit-differentiation identities
//
//   F(Q(p, theta), theta) = p,  f = F_x,  f' = F_xx,
//   Q_p = 1/f,  Q_theta = -F_theta/f,  Q_pp = -f'/f^3,
//   Q_p,theta = -(F_x,theta + f' Q_theta)/f^2,
//   Q_theta,theta = -(F_theta,theta + 2 F_theta,x Q_theta
//                      + f' Q_theta^2)/f,
//
// with every CDF partial evaluated by `mpmath.diff` (nested lambdas for the
// mixed entries) at the high-precision Q. The CDFs are the mpmath primitives
// (`gammainc`, `betainc`, `erfc`) plus the explicit Poisson mixture for the
// noncentral chi-square and the two-normal form for the inverse Gaussian.
//
// Gate: value and first-order entries 1e-10 relative, second-order entries
// 1e-8 relative, with a small absolute floor for the exact zeros/symmetry
// rows. Central-difference oracles below are secondary checks at 1e-6.

#include "quantape/math/Distributions/Beta.h"
#include "quantape/math/Distributions/ChiSquare.h"
#include "quantape/math/Distributions/Exponential.h"
#include "quantape/math/Distributions/Gamma.h"
#include "quantape/math/Distributions/InverseGaussian.h"
#include "quantape/math/Distributions/Lognormal.h"
#include "quantape/math/Distributions/NoncentralChiSquare.h"
#include "quantape/math/Distributions/Normal.h"
#include "quantape/math/Distributions/StudentT.h"

#include <cmath>
#include <filesystem>
#include <fstream>
#include <limits>
#include <sstream>
#include <string>
#include <vector>

#include "support/GtestSupport.h"

namespace {

using quantape::math::Beta;
using quantape::math::ChiSquare;
using quantape::math::Exponential;
using quantape::math::Gamma;
using quantape::math::InverseGaussian;
using quantape::math::Lognormal;
using quantape::math::NoncentralChiSquare;
using quantape::math::Normal;
using quantape::math::StudentT;

constexpr double kInf = std::numeric_limits<double>::infinity();

/// One parsed row of the shared quantile reference CSV. `present[i]` marks the
/// optional entries (params 2, g2, hP2, h12, h22) that one-parameter families
/// leave empty.
struct RefRow {
    std::string family;
    double p1 = 0.0;
    double p2 = 0.0;
    double p = 0.0;
    double q = 0.0;
    double gP = 0.0;
    double g1 = 0.0;
    double g2 = 0.0;
    double hPP = 0.0;
    double hP1 = 0.0;
    double hP2 = 0.0;
    double h11 = 0.0;
    double h12 = 0.0;
    double h22 = 0.0;
    bool hasP2 = false;
    bool hasG2 = false;
    bool hasHP2 = false;
    bool hasH12 = false;
    bool hasH22 = false;
};

constexpr const char* kQuantileReference = "distributions/quantile_reference.csv";

bool quantileReferencePresent() {
    return std::filesystem::is_regular_file(testDataDir() / kQuantileReference);
}

std::vector<RefRow> loadQuantileReference() {
    std::vector<RefRow> rows;
    std::ifstream in(RequireDataFile(kQuantileReference));
    std::string line;
    std::getline(in, line); // header
    while (std::getline(in, line)) {
        if (line.empty()) {
            continue;
        }
        std::stringstream ss(line);
        std::vector<std::string> f;
        std::string token;
        while (std::getline(ss, token, ',')) {
            f.push_back(token);
        }
        while (f.size() < 14) {
            f.emplace_back();
        }
        RefRow row;
        row.family = f[0];
        row.p1 = std::stod(f[1]);
        if (!f[2].empty()) {
            row.p2 = std::stod(f[2]);
            row.hasP2 = true;
        }
        row.p = std::stod(f[3]);
        row.q = std::stod(f[4]);
        row.gP = std::stod(f[5]);
        row.g1 = std::stod(f[6]);
        if (!f[7].empty()) {
            row.g2 = std::stod(f[7]);
            row.hasG2 = true;
        }
        row.hPP = std::stod(f[8]);
        row.hP1 = std::stod(f[9]);
        if (!f[10].empty()) {
            row.hP2 = std::stod(f[10]);
            row.hasHP2 = true;
        }
        row.h11 = std::stod(f[11]);
        if (!f[12].empty()) {
            row.h12 = std::stod(f[12]);
            row.hasH12 = true;
        }
        if (!f[13].empty()) {
            row.h22 = std::stod(f[13]);
            row.hasH22 = true;
        }
        rows.push_back(row);
    }
    return rows;
}

std::string labelOf(const RefRow& row) {
    return row.family + " p1=" + std::to_string(row.p1) +
           (row.hasP2 ? " p2=" + std::to_string(row.p2) : "") + " p=" + std::to_string(row.p);
}

void checkValue(const RefRow& row, double got) {
    CHECK_CLOSE_REL(labelOf(row) + " q", got, row.q, 1e-10, 1e-12);
}

void checkFirst(const RefRow& row, const std::string& name, double got, double expected) {
    CHECK_CLOSE_REL(labelOf(row) + " " + name, got, expected, 1e-10, 1e-12);
}

void checkSecond(const RefRow& row, const std::string& name, double got, double expected) {
    CHECK_CLOSE_REL(labelOf(row) + " " + name, got, expected, 1e-8, 1e-10);
}

/// Central difference of a scalar function.
template <class F>
double centralD1(const F& f, double x, double h) {
    return (f(x + h) - f(x - h)) / (2.0 * h);
}

} // namespace

// ---------------------------------------------------------------------------
// mpmath reference gates (one test per family; rows are filtered from the
// shared CSV).
// ---------------------------------------------------------------------------

TEST(QuantileDerivatives, normalReferenceCsv) {
    SKIP_UNLESS_ASSET(quantileReferencePresent(), "quantile reference csv absent");
    int rows = 0;
    for (const RefRow& r : loadQuantileReference()) {
        if (r.family != "normal") {
            continue;
        }
        ++rows;
        const Normal<double> d{r.p1, r.p2};
        const auto pack = d.invGradHess(r.p);
        checkValue(r, pack.value);
        checkFirst(r, "gP", pack.gP, r.gP);
        checkFirst(r, "gmu", pack.gmu, r.g1);
        checkFirst(r, "gsig", pack.gsig, r.g2);
        checkSecond(r, "hPP", pack.hPP, r.hPP);
        checkSecond(r, "hPmu", pack.hPmu, r.hP1);
        checkSecond(r, "hPsig", pack.hPsig, r.hP2);
        checkSecond(r, "hmumu", pack.hmumu, r.h11);
        checkSecond(r, "hmusig", pack.hmusig, r.h12);
        checkSecond(r, "hsigsig", pack.hsigsig, r.h22);
    }
    EXPECT_GT(rows, 0);
}

TEST(QuantileDerivatives, lognormalReferenceCsv) {
    SKIP_UNLESS_ASSET(quantileReferencePresent(), "quantile reference csv absent");
    int rows = 0;
    for (const RefRow& r : loadQuantileReference()) {
        if (r.family != "lognormal") {
            continue;
        }
        ++rows;
        const Lognormal<double> d{r.p1, r.p2};
        const auto pack = d.invGradHess(r.p);
        checkValue(r, pack.value);
        checkFirst(r, "gP", pack.gP, r.gP);
        checkFirst(r, "gmu", pack.gmu, r.g1);
        checkFirst(r, "gsig", pack.gsig, r.g2);
        checkSecond(r, "hPP", pack.hPP, r.hPP);
        checkSecond(r, "hPmu", pack.hPmu, r.hP1);
        checkSecond(r, "hPsig", pack.hPsig, r.hP2);
        checkSecond(r, "hmumu", pack.hmumu, r.h11);
        checkSecond(r, "hmusig", pack.hmusig, r.h12);
        checkSecond(r, "hsigsig", pack.hsigsig, r.h22);
    }
    EXPECT_GT(rows, 0);
}

TEST(QuantileDerivatives, exponentialReferenceCsv) {
    SKIP_UNLESS_ASSET(quantileReferencePresent(), "quantile reference csv absent");
    int rows = 0;
    for (const RefRow& r : loadQuantileReference()) {
        if (r.family != "exponential") {
            continue;
        }
        ++rows;
        const Exponential<double> d{r.p1};
        const auto pack = d.invGradHess(r.p);
        checkValue(r, pack.value);
        checkFirst(r, "gP", pack.gP, r.gP);
        checkFirst(r, "gLambda", pack.gLambda, r.g1);
        checkSecond(r, "hPP", pack.hPP, r.hPP);
        checkSecond(r, "hPL", pack.hPL, r.hP1);
        checkSecond(r, "hLL", pack.hLL, r.h11);
    }
    EXPECT_GT(rows, 0);
}

TEST(QuantileDerivatives, gammaReferenceCsv) {
    SKIP_UNLESS_ASSET(quantileReferencePresent(), "quantile reference csv absent");
    int rows = 0;
    for (const RefRow& r : loadQuantileReference()) {
        if (r.family != "gamma") {
            continue;
        }
        ++rows;
        const Gamma<double> d{r.p1, r.p2};
        const auto pack = d.invGradHess(r.p);
        checkValue(r, pack.value);
        checkFirst(r, "gP", pack.gP, r.gP);
        checkFirst(r, "gShape", pack.gShape, r.g1);
        checkFirst(r, "gRate", pack.gRate, r.g2);
        checkSecond(r, "hPP", pack.hPP, r.hPP);
        checkSecond(r, "hPShape", pack.hPShape, r.hP1);
        checkSecond(r, "hPRate", pack.hPRate, r.hP2);
        checkSecond(r, "hShapeShape", pack.hShapeShape, r.h11);
        checkSecond(r, "hShapeRate", pack.hShapeRate, r.h12);
        checkSecond(r, "hRateRate", pack.hRateRate, r.h22);
    }
    EXPECT_GT(rows, 0);
}

TEST(QuantileDerivatives, chiSquareReferenceCsv) {
    SKIP_UNLESS_ASSET(quantileReferencePresent(), "quantile reference csv absent");
    int rows = 0;
    for (const RefRow& r : loadQuantileReference()) {
        if (r.family != "chisquare") {
            continue;
        }
        ++rows;
        const ChiSquare<double> d{r.p1};
        const auto pack = d.invGradHess(r.p);
        checkValue(r, pack.value);
        checkFirst(r, "gP", pack.gP, r.gP);
        checkFirst(r, "gDf", pack.gDf, r.g1);
        checkSecond(r, "hPP", pack.hPP, r.hPP);
        checkSecond(r, "hPDf", pack.hPDf, r.hP1);
        checkSecond(r, "hDfDf", pack.hDfDf, r.h11);
    }
    EXPECT_GT(rows, 0);
}

TEST(QuantileDerivatives, betaReferenceCsv) {
    SKIP_UNLESS_ASSET(quantileReferencePresent(), "quantile reference csv absent");
    int rows = 0;
    for (const RefRow& r : loadQuantileReference()) {
        if (r.family != "beta") {
            continue;
        }
        ++rows;
        const Beta<double> d{r.p1, r.p2};
        const auto pack = d.invGradHess(r.p);
        checkValue(r, pack.value);
        checkFirst(r, "gP", pack.gP, r.gP);
        checkFirst(r, "gAlpha", pack.gAlpha, r.g1);
        checkFirst(r, "gBeta", pack.gBeta, r.g2);
        checkSecond(r, "hPP", pack.hPP, r.hPP);
        checkSecond(r, "hPAlpha", pack.hPAlpha, r.hP1);
        checkSecond(r, "hPBeta", pack.hPBeta, r.hP2);
        checkSecond(r, "hAA", pack.hAA, r.h11);
        checkSecond(r, "hAB", pack.hAB, r.h12);
        checkSecond(r, "hBB", pack.hBB, r.h22);
    }
    EXPECT_GT(rows, 0);
}

TEST(QuantileDerivatives, studentTReferenceCsv) {
    SKIP_UNLESS_ASSET(quantileReferencePresent(), "quantile reference csv absent");
    int rows = 0;
    for (const RefRow& r : loadQuantileReference()) {
        if (r.family != "studentt") {
            continue;
        }
        ++rows;
        const StudentT<double> d{r.p1};
        const auto pack = d.invGradHess(r.p);
        checkValue(r, pack.value);
        checkFirst(r, "gP", pack.gP, r.gP);
        checkFirst(r, "gNu", pack.gNu, r.g1);
        checkSecond(r, "hPP", pack.hPP, r.hPP);
        checkSecond(r, "hPNu", pack.hPNu, r.hP1);
        checkSecond(r, "hNuNu", pack.hNuNu, r.h11);
    }
    EXPECT_GT(rows, 0);
}

TEST(QuantileDerivatives, inverseGaussianReferenceCsv) {
    SKIP_UNLESS_ASSET(quantileReferencePresent(), "quantile reference csv absent");
    int rows = 0;
    for (const RefRow& r : loadQuantileReference()) {
        if (r.family != "ig") {
            continue;
        }
        ++rows;
        const InverseGaussian<double> d{r.p1, r.p2};
        const auto pack = d.invGradHess(r.p);
        checkValue(r, pack.value);
        checkFirst(r, "gP", pack.gP, r.gP);
        checkFirst(r, "gmu", pack.gmu, r.g1);
        checkFirst(r, "gLambda", pack.gLambda, r.g2);
        checkSecond(r, "hPP", pack.hPP, r.hPP);
        checkSecond(r, "hPmu", pack.hPmu, r.hP1);
        checkSecond(r, "hPLambda", pack.hPLambda, r.hP2);
        checkSecond(r, "hmumu", pack.hmumu, r.h11);
        checkSecond(r, "hmuLambda", pack.hmuLambda, r.h12);
        checkSecond(r, "hLL", pack.hLL, r.h22);
    }
    EXPECT_GT(rows, 0);
}

TEST(QuantileDerivatives, noncentralChiSquareReferenceCsv) {
    SKIP_UNLESS_ASSET(quantileReferencePresent(), "quantile reference csv absent");
    int rows = 0;
    for (const RefRow& r : loadQuantileReference()) {
        if (r.family != "nccs") {
            continue;
        }
        ++rows;
        const NoncentralChiSquare<double> d{r.p1, r.p2};
        const auto pack = d.invGradHess(r.p);
        checkValue(r, pack.value);
        checkFirst(r, "gP", pack.gP, r.gP);
        checkFirst(r, "gNu", pack.gNu, r.g1);
        checkFirst(r, "gLambda", pack.gLambda, r.g2);
        checkSecond(r, "hPP", pack.hPP, r.hPP);
        checkSecond(r, "hPNu", pack.hPNu, r.hP1);
        checkSecond(r, "hPLambda", pack.hPLambda, r.hP2);
        checkSecond(r, "hNuNu", pack.hNuNu, r.h11);
        checkSecond(r, "hNuLambda", pack.hNuLambda, r.h12);
        checkSecond(r, "hLambdaLambda", pack.hLambdaLambda, r.h22);
    }
    EXPECT_GT(rows, 0);
}

// ---------------------------------------------------------------------------
// Value anchors: the pack value is bitwise `inv(p)`.
// ---------------------------------------------------------------------------

TEST(QuantileDerivatives, packValueIsBitwiseInv) {
    for (double p : {1e-5, 0.01, 0.25, 0.5, 0.755, 0.99, 0.9999}) {
        const std::string tag = "p=" + std::to_string(p);
        const Normal<double> n{0.3, 1.4};
        const Lognormal<double> ln{-0.2, 0.8};
        const Exponential<double> e{1.7};
        const Gamma<double> g{2.5, 1.3};
        const ChiSquare<double> c{2.5};
        const Beta<double> b{2.0, 3.0};
        const StudentT<double> t{3.0};
        const InverseGaussian<double> ig{1.0, 2.0};
        const NoncentralChiSquare<double> nc{2.0, 5.0};
        EXPECT_EQ(n.invGradHess(p).value, n.inv(p)) << tag;
        EXPECT_EQ(ln.invGradHess(p).value, ln.inv(p)) << tag;
        EXPECT_EQ(e.invGradHess(p).value, e.inv(p)) << tag;
        EXPECT_EQ(g.invGradHess(p).value, g.inv(p)) << tag;
        EXPECT_EQ(c.invGradHess(p).value, c.inv(p)) << tag;
        EXPECT_EQ(b.invGradHess(p).value, b.inv(p)) << tag;
        EXPECT_EQ(t.invGradHess(p).value, t.inv(p)) << tag;
        EXPECT_EQ(ig.invGradHess(p).value, ig.inv(p)) << tag;
        EXPECT_EQ(nc.invGradHess(p).value, nc.inv(p)) << tag;
    }
}

// ---------------------------------------------------------------------------
// Round trips, monotonicity and edges.
// ---------------------------------------------------------------------------

template <class Dist>
void checkRoundTripAndMonotone(const Dist& d, const std::vector<double>& ps, double rtol,
                               double atol) {
    double prev = -kInf;
    for (double p : ps) {
        const double q = d.inv(p);
        SCOPED_TRACE(std::string("p=") + std::to_string(p));
        EXPECT_GT(q, prev);
        prev = q;
        CHECK_CLOSE_REL("cdf(inv(p)) = p", d.cdf(q), p, rtol, atol);
    }
}

TEST(QuantileDerivatives, roundTripsAndMonotonicity) {
    const std::vector<double> ps = {1e-4, 0.001, 0.01, 0.1, 0.5, 0.9, 0.99, 0.999, 0.9999};
    checkRoundTripAndMonotone(Normal<double>{0.3, 1.4}, ps, 1e-13, 1e-15);
    checkRoundTripAndMonotone(Lognormal<double>{-0.3, 0.7}, ps, 1e-13, 1e-15);
    checkRoundTripAndMonotone(Exponential<double>{1.7}, ps, 1e-13, 1e-15);
    checkRoundTripAndMonotone(Gamma<double>{2.5, 1.3}, ps, 1e-12, 1e-14);
    checkRoundTripAndMonotone(ChiSquare<double>{2.5}, ps, 1e-12, 1e-14);
    checkRoundTripAndMonotone(Beta<double>{2.0, 3.0}, ps, 1e-12, 1e-14);
    checkRoundTripAndMonotone(StudentT<double>{3.0}, ps, 1e-11, 1e-14);
    checkRoundTripAndMonotone(InverseGaussian<double>{1.0, 1.0}, ps, 1e-13, 1e-15);
    checkRoundTripAndMonotone(InverseGaussian<double>{2.0, 4.0}, ps, 1e-13, 1e-15);
    checkRoundTripAndMonotone(NoncentralChiSquare<double>{2.0, 5.0}, ps, 1e-12, 1e-14);
    checkRoundTripAndMonotone(NoncentralChiSquare<double>{4.0, 20.0}, ps, 1e-12, 1e-14);
}

TEST(QuantileDerivatives, deepLowerTails) {
    // Regression for the relative-step Newton termination: with nu < 2 the
    // density diverges at 0, so an absolute step floor stopped the solve
    // while `cdf(Q) - p` was still ~2% of p (noncentral df = 1, ncp = 0.5).
    const NoncentralChiSquare<double> nc{1.0, 0.5};
    for (double p : {1e-10, 1e-8, 1e-6}) {
        CHECK_CLOSE_REL("nccs deep lower tail", nc.cdf(nc.inv(p)), p, 1e-12);
    }
    const InverseGaussian<double> ig{1.0, 1.0};
    for (double p : {1e-10, 1e-8, 1e-6}) {
        CHECK_CLOSE_REL("ig deep lower tail", ig.cdf(ig.inv(p)), p, 1e-12);
    }
}

TEST(QuantileDerivatives, inverseGaussianEdgesAndKnownValues) {
    const InverseGaussian<double> d{1.0, 1.0};
    CHECK_CLOSE("inv(0) = 0", d.inv(0.0), 0.0, 0.0);
    CHECK_CLOSE("inv(-0.5) = 0", d.inv(-0.5), 0.0, 0.0);
    CHECK_CLOSE("inv(1) = +inf", d.inv(1.0), kInf, 0.0);
    // Median of IG(1,1): F^{-1}(1/2) = 0.6758413056952391... (mpmath).
    CHECK_CLOSE_REL("IG(1,1) median", d.inv(0.5), 0.6758413056952391, 1e-12);
    // Scaling identity X ~ IG(mu, lambda) => c X ~ IG(c mu, c lambda):
    // IG(2,4) = 2 IG(1,2), so the quantiles scale by 2.
    const InverseGaussian<double> ig12{1.0, 2.0};
    const InverseGaussian<double> ig24{2.0, 4.0};
    CHECK_CLOSE_REL("IG(2,4) median = 2 * IG(1,2) median", ig24.inv(0.5), 2.0 * ig12.inv(0.5),
                    1e-12);
    const InverseGaussian<double> scaled{3.0, 3.0};
    CHECK_CLOSE_REL("IG scaling", scaled.inv(0.9), 3.0 * d.inv(0.9), 1e-11);
    // Limits: cdf(inv(p)) with the CDF's own saturation; NaN and invalid
    // parameters propagate.
    const double nan = std::numeric_limits<double>::quiet_NaN();
    EXPECT_TRUE(std::isnan(d.inv(nan)));
    EXPECT_TRUE(std::isnan(InverseGaussian<double>{-1.0, 1.0}.inv(0.5)));
    EXPECT_TRUE(std::isnan(InverseGaussian<double>{1.0, -1.0}.inv(0.5)));
    const auto pack = d.invGradHess(0.5);
    EXPECT_TRUE(std::isnan(d.invGradHess(0.0).gP));
    EXPECT_TRUE(std::isnan(d.invGradHess(1.0).gP));
    EXPECT_EQ(d.invGradHess(0.0).value, 0.0);
    EXPECT_EQ(d.invGradHess(1.0).value, kInf);
    // Pack equations: gP = 1/f, gtheta = -F_theta/f at the quantile.
    const auto p0 = d.invGradHess(0.5);
    const auto c = d.cdfGradHess(pack.value);
    CHECK_CLOSE_REL("gP = 1/pdf(Q)", p0.gP, 1.0 / c.gx, 1e-12);
    CHECK_CLOSE_REL("gmu = -F_mu/pdf(Q)", p0.gmu, -c.gmu / c.gx, 1e-12);
    CHECK_CLOSE_REL("gLambda = -F_lambda/pdf(Q)", p0.gLambda, -c.gLambda / c.gx, 1e-12);
}

TEST(QuantileDerivatives, noncentralChiSquareEdgesAndReduction) {
    const NoncentralChiSquare<double> d{2.0, 5.0};
    CHECK_CLOSE("inv(0) = 0", d.inv(0.0), 0.0, 0.0);
    CHECK_CLOSE("inv(-1) = 0", d.inv(-1.0), 0.0, 0.0);
    CHECK_CLOSE("inv(1) = +inf", d.inv(1.0), kInf, 0.0);
    const double nan = std::numeric_limits<double>::quiet_NaN();
    EXPECT_TRUE(std::isnan(d.inv(nan)));
    EXPECT_TRUE(std::isnan(NoncentralChiSquare<double>{-1.0, 5.0}.inv(0.5)));
    EXPECT_TRUE(std::isnan(NoncentralChiSquare<double>{2.0, -1.0}.inv(0.5)));
    EXPECT_TRUE(std::isnan(d.invGradHess(0.0).gP));
    EXPECT_TRUE(std::isnan(d.invGradHess(1.0).gP));
    EXPECT_EQ(d.invGradHess(1.0).value, kInf);

    // ncp = 0 reduces bitwise to the central chi-square quantile. The packs
    // are not bitwise equal there (the noncentral nu/lambda partials use the
    // rate-1 mixture rather than the central rate-1/2 path, and dQ/dlambda at
    // ncp = 0 is genuinely non-zero), so only the quantile reduction is exact.
    for (double nu : {0.5, 1.0, 3.0, 11.5}) {
        const NoncentralChiSquare<double> central_reduction{nu, 0.0};
        const ChiSquare<double> chi{nu};
        for (double p : {1e-5, 0.01, 0.3, 0.9, 0.999}) {
            EXPECT_EQ(central_reduction.inv(p), chi.inv(p));
        }
        const auto a = central_reduction.invGradHess(0.7);
        const auto b = chi.invGradHess(0.7);
        CHECK_CLOSE_REL("ncp=0 gP matches central", a.gP, b.gP, 1e-12, 1e-14);
        CHECK_CLOSE_REL("ncp=0 gNu matches central gDf", a.gNu, b.gDf, 1e-12, 1e-14);
        CHECK_CLOSE_REL("ncp=0 hNuNu matches central hDfDf", a.hNuNu, b.hDfDf, 1e-12, 1e-14);
    }

    // Moderate large-ncp regime stays consistent (mixture window).
    const NoncentralChiSquare<double> big{4.0, 100.0};
    for (double p : {0.01, 0.5, 0.99}) {
        CHECK_CLOSE_REL("ncp=100 round trip", big.cdf(big.inv(p)), p, 1e-12, 1e-14);
    }
}

// ---------------------------------------------------------------------------
// Secondary central-difference oracles (1e-6): first-order entries from the
// raw inverse and second-order entries from the analytic first-order fields.
// The absolute floor absorbs differences of quantities that are exactly zero
// (symmetry rows) where the FD estimate itself is cancellation noise.
// ---------------------------------------------------------------------------

#define CHECK_FD(label, got, fd) CHECK_CLOSE_REL(label, got, fd, 1e-5, 1e-8)

TEST(QuantileDerivatives, finiteDifferenceOracles) {
    const double hp = 1e-5;
    const double hh = 1e-5 * 0.7;

    { // Normal(mu, sigma)
        const Normal<double> d{0.3, 1.4};
        const double p = 0.91;
        const auto pk = d.invGradHess(p);
        CHECK_FD("normal gP FD", pk.gP, centralD1([&](double t) { return d.inv(t); }, p, hp));
        CHECK_FD("normal gmu FD", pk.gmu,
                 centralD1([&](double m) { return Normal<double>{m, 1.4}.inv(p); }, 0.3, hh));
        CHECK_FD("normal gsig FD", pk.gsig,
                 centralD1([&](double s) { return Normal<double>{0.3, s}.inv(p); }, 1.4, hh));
        CHECK_FD("normal hPP FD", pk.hPP,
                 centralD1([&](double t) { return d.invGradHess(t).gP; }, p, hp));
        CHECK_FD("normal hPmu FD", pk.hPmu,
                 centralD1([&](double t) { return d.invGradHess(t).gmu; }, p, hp));
        CHECK_FD("normal hmumu FD", pk.hmumu,
                 centralD1([&](double m) { return Normal<double>{m, 1.4}.invGradHess(p).gmu; }, 0.3,
                           hh));
        CHECK_FD("normal hmusig FD", pk.hmusig,
                 centralD1([&](double s) { return Normal<double>{0.3, s}.invGradHess(p).gmu; }, 1.4,
                           hh));
        CHECK_FD("normal hsigsig FD", pk.hsigsig,
                 centralD1([&](double s) { return Normal<double>{0.3, s}.invGradHess(p).gsig; },
                           1.4, hh));
    }
    { // Gamma(shape, rate)
        const Gamma<double> d{2.5, 1.3};
        const double p = 0.85;
        const auto pk = d.invGradHess(p);
        CHECK_FD("gamma gP FD", pk.gP, centralD1([&](double t) { return d.inv(t); }, p, hp));
        CHECK_FD("gamma gShape FD", pk.gShape,
                 centralD1([&](double a) { return Gamma<double>{a, 1.3}.inv(p); }, 2.5, hh));
        CHECK_FD("gamma gRate FD", pk.gRate,
                 centralD1([&](double r) { return Gamma<double>{2.5, r}.inv(p); }, 1.3, hh));
        CHECK_FD("gamma hPP FD", pk.hPP,
                 centralD1([&](double t) { return d.invGradHess(t).gP; }, p, hp));
        CHECK_FD("gamma hShapeShape FD", pk.hShapeShape,
                 centralD1([&](double a) { return Gamma<double>{a, 1.3}.invGradHess(p).gShape; },
                           2.5, hh));
        CHECK_FD("gamma hShapeRate FD", pk.hShapeRate,
                 centralD1([&](double r) { return Gamma<double>{2.5, r}.invGradHess(p).gShape; },
                           1.3, hh));
        CHECK_FD("gamma hRateRate FD", pk.hRateRate,
                 centralD1([&](double r) { return Gamma<double>{2.5, r}.invGradHess(p).gRate; },
                           1.3, hh));
    }
    { // Beta(alpha, beta), symmetric-ish
        const Beta<double> d{2.0, 3.0};
        const double p = 0.77;
        const auto pk = d.invGradHess(p);
        CHECK_FD("beta gP FD", pk.gP, centralD1([&](double t) { return d.inv(t); }, p, hp));
        CHECK_FD("beta gAlpha FD", pk.gAlpha,
                 centralD1([&](double a) { return Beta<double>{a, 3.0}.inv(p); }, 2.0, hh));
        CHECK_FD("beta gBeta FD", pk.gBeta,
                 centralD1([&](double b) { return Beta<double>{2.0, b}.inv(p); }, 3.0, hh));
        CHECK_FD("beta hPP FD", pk.hPP,
                 centralD1([&](double t) { return d.invGradHess(t).gP; }, p, hp));
        CHECK_FD("beta hAA FD", pk.hAA,
                 centralD1([&](double a) { return Beta<double>{a, 3.0}.invGradHess(p).gAlpha; },
                           2.0, hh));
        CHECK_FD("beta hAB FD", pk.hAB,
                 centralD1([&](double b) { return Beta<double>{2.0, b}.invGradHess(p).gAlpha; },
                           3.0, hh));
        CHECK_FD("beta hBB FD", pk.hBB,
                 centralD1([&](double b) { return Beta<double>{2.0, b}.invGradHess(p).gBeta; }, 3.0,
                           hh));
    }
    { // StudentT(nu)
        const StudentT<double> d{3.0};
        const double p = 0.92;
        const auto pk = d.invGradHess(p);
        CHECK_FD("t gP FD", pk.gP, centralD1([&](double t) { return d.inv(t); }, p, hp));
        CHECK_FD("t gNu FD", pk.gNu,
                 centralD1([&](double nu) { return StudentT<double>{nu}.inv(p); }, 3.0, hh));
        CHECK_FD("t hPP FD", pk.hPP,
                 centralD1([&](double t) { return d.invGradHess(t).gP; }, p, hp));
        CHECK_FD("t hPNu FD", pk.hPNu,
                 centralD1([&](double t) { return d.invGradHess(t).gNu; }, p, hp));
        CHECK_FD(
            "t hNuNu FD", pk.hNuNu,
            centralD1([&](double nu) { return StudentT<double>{nu}.invGradHess(p).gNu; }, 3.0, hh));
    }
    { // Exponential(lambda)
        const Exponential<double> d{0.8};
        const double p = 0.62;
        const auto pk = d.invGradHess(p);
        CHECK_FD("exp gP FD", pk.gP, centralD1([&](double t) { return d.inv(t); }, p, hp));
        CHECK_FD("exp gLambda FD", pk.gLambda,
                 centralD1([&](double l) { return Exponential<double>{l}.inv(p); }, 0.8, hh));
        CHECK_FD("exp hPP FD", pk.hPP,
                 centralD1([&](double t) { return d.invGradHess(t).gP; }, p, hp));
        CHECK_FD("exp hPL FD", pk.hPL,
                 centralD1([&](double t) { return d.invGradHess(t).gLambda; }, p, hp));
        CHECK_FD("exp hLL FD", pk.hLL,
                 centralD1([&](double l) { return Exponential<double>{l}.invGradHess(p).gLambda; },
                           0.8, hh));
    }
    { // ChiSquare(df)
        const ChiSquare<double> d{4.5};
        const double p = 0.8;
        const auto pk = d.invGradHess(p);
        CHECK_FD("chisq gP FD", pk.gP, centralD1([&](double t) { return d.inv(t); }, p, hp));
        CHECK_FD("chisq gDf FD", pk.gDf,
                 centralD1([&](double df) { return ChiSquare<double>{df}.inv(p); }, 4.5, hh));
        CHECK_FD("chisq hPP FD", pk.hPP,
                 centralD1([&](double t) { return d.invGradHess(t).gP; }, p, hp));
        CHECK_FD("chisq hDfDf FD", pk.hDfDf,
                 centralD1([&](double df) { return ChiSquare<double>{df}.invGradHess(p).gDf; }, 4.5,
                           hh));
    }
    { // InverseGaussian(mu, lambda)
        const InverseGaussian<double> d{1.0, 2.0};
        const double p = 0.85;
        const auto pk = d.invGradHess(p);
        CHECK_FD("ig gP FD", pk.gP, centralD1([&](double t) { return d.inv(t); }, p, hp));
        CHECK_FD(
            "ig gmu FD", pk.gmu,
            centralD1([&](double m) { return InverseGaussian<double>{m, 2.0}.inv(p); }, 1.0, hh));
        CHECK_FD(
            "ig gLambda FD", pk.gLambda,
            centralD1([&](double l) { return InverseGaussian<double>{1.0, l}.inv(p); }, 2.0, hh));
        CHECK_FD("ig hPP FD", pk.hPP,
                 centralD1([&](double t) { return d.invGradHess(t).gP; }, p, hp));
        CHECK_FD(
            "ig hmumu FD", pk.hmumu,
            centralD1([&](double m) { return InverseGaussian<double>{m, 2.0}.invGradHess(p).gmu; },
                      1.0, hh));
        CHECK_FD(
            "ig hLL FD", pk.hLL,
            centralD1(
                [&](double l) { return InverseGaussian<double>{1.0, l}.invGradHess(p).gLambda; },
                2.0, hh));
    }
    { // NoncentralChiSquare(df, ncp)
        const NoncentralChiSquare<double> d{2.0, 5.0};
        const double p = 0.8;
        const auto pk = d.invGradHess(p);
        CHECK_FD("nccs gP FD", pk.gP, centralD1([&](double t) { return d.inv(t); }, p, hp));
        CHECK_FD("nccs gNu FD", pk.gNu,
                 centralD1([&](double nu) { return NoncentralChiSquare<double>{nu, 5.0}.inv(p); },
                           2.0, hh));
        CHECK_FD("nccs gLambda FD", pk.gLambda,
                 centralD1([&](double l) { return NoncentralChiSquare<double>{2.0, l}.inv(p); },
                           5.0, hh));
        CHECK_FD("nccs hPP FD", pk.hPP,
                 centralD1([&](double t) { return d.invGradHess(t).gP; }, p, hp));
        CHECK_FD(
            "nccs hNuNu FD", pk.hNuNu,
            centralD1(
                [&](double nu) { return NoncentralChiSquare<double>{nu, 5.0}.invGradHess(p).gNu; },
                2.0, hh));
        CHECK_FD("nccs hLambdaLambda FD", pk.hLambdaLambda,
                 centralD1(
                     [&](double l) {
                         return NoncentralChiSquare<double>{2.0, l}.invGradHess(p).gLambda;
                     },
                     5.0, hh));
    }
}

#undef CHECK_FD

// ---------------------------------------------------------------------------
// Cross-family consistency: the pack derivatives are the exact implicit
// derivatives of the CDF, checked by the identity
//   f Q_theta + F_theta = 0  and  f Q_p = 1
// at the quantile (all quantities analytic, no finite differences).
// ---------------------------------------------------------------------------

TEST(QuantileDerivatives, implicitIdentityResiduals) {
    const auto check = [](const std::string& label, double f, double qp, double qtheta,
                          double ftheta) {
        CHECK_CLOSE_REL(label + " f Q_p = 1", f * qp, 1.0, 1e-12);
        CHECK_CLOSE_REL(label + " f Q_theta + F_theta = 0", f * qtheta + ftheta, 0.0, 1e-11, 1e-12);
    };
    {
        const Normal<double> d{0.3, 1.4};
        const auto pk = d.invGradHess(0.93);
        const auto c = d.cdfGradHess(pk.value);
        check("normal mu", c.gx, pk.gP, pk.gmu, c.gmu);
        check("normal sig", c.gx, pk.gP, pk.gsig, c.gsig);
    }
    {
        const Gamma<double> d{2.5, 1.3};
        const auto pk = d.invGradHess(0.93);
        const auto c = quantape::math::detail::gammaCdfGradHess(2.5, 1.3, pk.value);
        check("gamma shape", c.gx, pk.gP, pk.gShape, c.gShape);
        check("gamma rate", c.gx, pk.gP, pk.gRate, c.gRate);
    }
    {
        const Beta<double> d{2.0, 3.0};
        const auto pk = d.invGradHess(0.93);
        const auto c = quantape::math::detail::betaCdfGradHess(2.0, 3.0, pk.value);
        check("beta alpha", c.gx, pk.gP, pk.gAlpha, c.gAlpha);
        check("beta beta", c.gx, pk.gP, pk.gBeta, c.gBeta);
    }
    {
        const StudentT<double> d{3.0};
        const auto pk = d.invGradHess(0.93);
        const auto c = quantape::math::detail::studentTCdfGradHess(3.0, pk.value);
        check("t nu", c.gx, pk.gP, pk.gNu, c.gNu);
    }
    {
        const InverseGaussian<double> d{1.0, 2.0};
        const auto pk = d.invGradHess(0.93);
        const auto c = d.cdfGradHess(pk.value);
        check("ig mu", c.gx, pk.gP, pk.gmu, c.gmu);
        check("ig lambda", c.gx, pk.gP, pk.gLambda, c.gLambda);
    }
    {
        const NoncentralChiSquare<double> d{2.0, 5.0};
        const auto pk = d.invGradHess(0.93);
        const auto c = d.cdfGradHess(pk.value);
        check("nccs nu", c.gx, pk.gP, pk.gNu, c.gNu);
        check("nccs lambda", c.gx, pk.gP, pk.gLambda, c.gLambda);
    }
}
