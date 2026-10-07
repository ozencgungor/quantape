// test_trivariate_normal.cpp — trivariate normal CDF value/gradient/Hessian
// gates.  The CDF is gated three ways: exact limits and identities, the
// analytic packs against Richardson-extrapolated finite differences, and
// high-precision mpmath references in tests/data/distributions/tvn_reference.csv
// (2-D conditional quadrature at 40 digits for the values; the analytic
// derivative pack validated against mpmath differentiation).  Documented
// reference gates: value 1e-10, first derivatives 1e-9, Hessian 1e-8.

#include "quantape/math/Distributions/BivariateNormal.h"
#include "quantape/math/Distributions/TrivariateNormal.h"

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
constexpr double kPi = 3.14159265358979323846;

using quantape::math::BivariateNormal;
using quantape::math::TrivariateNormal;
using quantape::math::detail::TvnGradHess;
using quantape::math::detail::tvnHessIndex;

double phi(double x) {
    return 0.5 * std::erfc(-x / std::sqrt(2.0));
}

struct RefRow {
    double p[6];
    double cdf;
    double g[6];
    double h[21];
};

std::vector<RefRow> loadReference() {
    std::vector<RefRow> rows;
    std::ifstream in(RequireDataFile("distributions/tvn_reference.csv"));
    std::string line;
    while (std::getline(in, line)) {
        if (line.empty() || line[0] == '#' || line.rfind("h1,", 0) == 0) {
            continue;
        }
        std::stringstream ss(line);
        std::string field;
        std::vector<double> values;
        while (std::getline(ss, field, ',')) {
            values.push_back(std::stod(field));
        }
        if (values.size() != 34) {
            continue;
        }
        RefRow row{};
        for (int i = 0; i < 6; ++i) {
            row.p[i] = values[static_cast<std::size_t>(i)];
        }
        row.cdf = values[6];
        for (int i = 0; i < 6; ++i) {
            row.g[i] = values[7 + static_cast<std::size_t>(i)];
        }
        for (int i = 0; i < 21; ++i) {
            row.h[i] = values[13 + static_cast<std::size_t>(i)];
        }
        rows.push_back(row);
    }
    return rows;
}

std::string pointLabel(const char* what, const double* p) {
    std::ostringstream out;
    out.precision(6);
    out << what << " h=(" << p[0] << "," << p[1] << "," << p[2] << ") r=(" << p[3] << "," << p[4]
        << "," << p[5] << ")";
    return out.str();
}

double cdfOf(const double* p) {
    return TrivariateNormal{p[3], p[4], p[5]}.cdf(p[0], p[1], p[2]);
}

/// Richardson-extrapolated central first derivative (error O(h^4)).
template <class F>
double richFirst(F f, const double* p, int i, double h) {
    double plus[6];
    double minus[6];
    double plus2[6];
    double minus2[6];
    std::copy(p, p + 6, plus);
    std::copy(p, p + 6, minus);
    std::copy(p, p + 6, plus2);
    std::copy(p, p + 6, minus2);
    plus[i] = p[i] + h;
    minus[i] = p[i] - h;
    plus2[i] = p[i] + 2.0 * h;
    minus2[i] = p[i] - 2.0 * h;
    const double d1 = (f(plus) - f(minus)) / (2.0 * h);
    const double d2 = (f(plus2) - f(minus2)) / (4.0 * h);
    return (4.0 * d1 - d2) / 3.0;
}

/// Richardson-extrapolated central second derivative (error O(h^4)).
template <class F>
double richSecond(F f, const double* p, int i, double h) {
    double plus[6];
    double minus[6];
    double plus2[6];
    double minus2[6];
    std::copy(p, p + 6, plus);
    std::copy(p, p + 6, minus);
    std::copy(p, p + 6, plus2);
    std::copy(p, p + 6, minus2);
    plus[i] = p[i] + h;
    minus[i] = p[i] - h;
    plus2[i] = p[i] + 2.0 * h;
    minus2[i] = p[i] - 2.0 * h;
    const double f0 = f(p);
    const double d1 = (f(plus) - 2.0 * f0 + f(minus)) / (h * h);
    const double d2 = (f(plus2) - 2.0 * f0 + f(minus2)) / (4.0 * h * h);
    return (4.0 * d1 - d2) / 3.0;
}

/// Richardson-extrapolated central mixed derivative (error O(h^4)).
template <class F>
double richMixed(F f, const double* p, int i, int j, double h) {
    double pp[6];
    double pm[6];
    double mp[6];
    double mm[6];
    double pp2[6];
    double pm2[6];
    double mp2[6];
    double mm2[6];
    for (int k = 0; k < 6; ++k) {
        pp[k] = pm[k] = mp[k] = mm[k] = p[k];
        pp2[k] = pm2[k] = mp2[k] = mm2[k] = p[k];
    }
    pp[i] += h;
    pm[i] += h;
    mp[i] -= h;
    mm[i] -= h;
    pp[j] += h;
    mp[j] += h;
    pm[j] -= h;
    mm[j] -= h;
    pp2[i] += 2.0 * h;
    pm2[i] += 2.0 * h;
    mp2[i] -= 2.0 * h;
    mm2[i] -= 2.0 * h;
    pp2[j] += 2.0 * h;
    mp2[j] += 2.0 * h;
    pm2[j] -= 2.0 * h;
    mm2[j] -= 2.0 * h;
    const double d1 = (f(pp) - f(pm) - f(mp) + f(mm)) / (4.0 * h * h);
    const double d2 = (f(pp2) - f(pm2) - f(mp2) + f(mm2)) / (16.0 * h * h);
    return (4.0 * d1 - d2) / 3.0;
}

/// Correlation matrix entry for a triple ordered (r12,r13,r23).
double corrEntry(int i, int j, const double* r) {
    if (i == j) {
        return 1.0;
    }
    const int lo = std::min(i, j);
    const int hi = std::max(i, j);
    if (lo == 0 && hi == 1) {
        return r[0];
    }
    if (lo == 0 && hi == 2) {
        return r[1];
    }
    return r[2];
}

} // namespace

TEST(TrivariateNormalDistribution, referenceValues) {
    const std::vector<RefRow> rows = loadReference();
    EXPECT_GT(rows.size(), 30u);
    double worstValue = 0.0;
    double worstGrad = 0.0;
    double worstHess = 0.0;
    for (const RefRow& row : rows) {
        const TrivariateNormal d{row.p[3], row.p[4], row.p[5]};
        const TvnGradHess pack = d.cdfGradHess(row.p[0], row.p[1], row.p[2]);
        CHECK_CLOSE(pointLabel("reference cdf", row.p), pack.value, row.cdf, 1e-10);
        worstValue = std::max(worstValue, std::abs(pack.value - row.cdf));
        for (int i = 0; i < 6; ++i) {
            CHECK_CLOSE(pointLabel("reference gradient", row.p), pack.g[i], row.g[i], 1e-9);
            worstGrad = std::max(worstGrad, std::abs(pack.g[i] - row.g[i]));
        }
        for (int i = 0; i < 21; ++i) {
            CHECK_CLOSE(pointLabel("reference Hessian", row.p), pack.h[i], row.h[i], 1e-8);
            worstHess = std::max(worstHess, std::abs(pack.h[i] - row.h[i]));
        }
    }
    EXPECT_LT(worstValue, 1e-10);
    EXPECT_LT(worstGrad, 1e-9);
    EXPECT_LT(worstHess, 1e-8);
}

TEST(TrivariateNormalDistribution, exactLimitsAndDegenerateCorrelations) {
    const double inf = std::numeric_limits<double>::infinity();
    const double nan = std::numeric_limits<double>::quiet_NaN();
    const double grid[] = {-2.0, -0.5, 0.0, 1.0, 2.5};

    // Independent margins: r = 0 factors into univariate CDFs.
    const TrivariateNormal independent{0.0, 0.0, 0.0};
    for (double a : grid) {
        for (double b : grid) {
            for (double c : grid) {
                CHECK_CLOSE("r=0 product", independent.cdf(a, b, c), phi(a) * phi(b) * phi(c),
                            kTol);
            }
        }
    }

    // Infinite arguments: -inf kills the probability, +inf marginalizes.
    const TrivariateNormal d{0.4, -0.3, 0.5};
    CHECK_CLOSE("all +inf", d.cdf(inf, inf, inf), 1.0, kTol);
    CHECK_CLOSE("-inf", d.cdf(inf, -inf, inf), 0.0, kTol);
    CHECK_CLOSE("h1 +inf", d.cdf(inf, 1.0, -0.5), (BivariateNormal<double>{0.5}.cdf(1.0, -0.5)),
                kTol);
    CHECK_CLOSE("h2 +inf", d.cdf(1.0, inf, -0.5), (BivariateNormal<double>{-0.3}.cdf(1.0, -0.5)),
                kTol);
    CHECK_CLOSE("h3 +inf", d.cdf(1.0, -0.5, inf), (BivariateNormal<double>{0.4}.cdf(1.0, -0.5)),
                kTol);
    CHECK_CLOSE("two +inf", d.cdf(inf, inf, 0.7), phi(0.7), kTol);

    // Singular correlations with consistent companions.
    const TrivariateNormal r12One{1.0, 0.3, 0.3};
    const TrivariateNormal r13One{0.3, 1.0, 0.3};
    const TrivariateNormal r23One{0.3, 0.3, 1.0};
    const TrivariateNormal r23Minus{0.3, -0.3, -1.0};
    const TrivariateNormal r12Minus{-1.0, -0.3, 0.3};
    const BivariateNormal<double> bvnHalf{0.3};
    const BivariateNormal<double> bvnNegHalf{-0.3};
    CHECK_CLOSE("r12=1", r12One.cdf(1.0, -0.5, 0.2), bvnHalf.cdf(-0.5, 0.2), kTol);
    CHECK_CLOSE("r13=1", r13One.cdf(1.0, -0.5, 0.2), bvnHalf.cdf(-0.5, 0.2), kTol);
    CHECK_CLOSE("r23=1", r23One.cdf(1.0, -0.5, 0.2), bvnHalf.cdf(1.0, -0.5), kTol);
    CHECK_CLOSE("r23=-1 positive", r23Minus.cdf(1.0, 0.5, -0.2),
                bvnHalf.cdf(1.0, 0.5) - bvnHalf.cdf(1.0, 0.2), kTol);
    CHECK_CLOSE("r23=-1 empty", r23Minus.cdf(-1.0, -2.0, -2.0), 0.0, kTol);
    CHECK_CLOSE("r12=-1", r12Minus.cdf(1.0, -0.5, 0.2),
                bvnNegHalf.cdf(1.0, 0.2) - bvnNegHalf.cdf(0.5, 0.2), kTol);

    // NaN and invalid inputs.
    EXPECT_TRUE(std::isnan(TrivariateNormal{0.3, 0.2, 0.1}.cdf(nan, 0.0, 0.0)));
    EXPECT_TRUE(std::isnan(TrivariateNormal{0.3, 0.2, 0.1}.cdf(0.0, nan, 0.0)));
    EXPECT_TRUE(std::isnan(TrivariateNormal{0.3, 0.2, 0.1}.cdf(0.0, 0.0, nan)));
    EXPECT_TRUE(std::isnan(TrivariateNormal{nan, 0.2, 0.1}.cdf(0.0, 0.0, 0.0)));
    EXPECT_TRUE(std::isnan(TrivariateNormal{2.0, 0.2, 0.1}.cdf(0.0, 0.0, 0.0)));
    EXPECT_TRUE(std::isnan(TrivariateNormal{-1.5, 0.2, 0.1}.cdf(0.0, 0.0, 0.0)));
    EXPECT_TRUE(std::isnan(TrivariateNormal{1.0, 0.5, -0.5}.cdf(0.0, 0.0, 0.0)));
}

TEST(TrivariateNormalDistribution, orthantClosedForm) {
    const double triples[][3] = {
        {0.0, 0.0, 0.0},   {0.5, 0.3, 0.2},     {-0.5, 0.3, -0.2},  {0.8, -0.1, 0.3},
        {0.9, 0.88, 0.87}, {-0.9, 0.88, -0.87}, {0.99, 0.98, 0.97},
    };
    for (const auto& t : triples) {
        const double expected =
            0.125 + (std::asin(t[0]) + std::asin(t[1]) + std::asin(t[2])) / (4.0 * kPi);
        const double got = TrivariateNormal{t[0], t[1], t[2]}.cdf(0.0, 0.0, 0.0);
        CHECK_CLOSE("orthant", got, expected, kTol);
    }
}

TEST(TrivariateNormalDistribution, permutationSymmetry) {
    const double points[][3] = {
        {0.5, -0.3, 0.9}, {-1.2, 1.7, 0.4}, {2.0, -2.0, 1.0}, {0.0, 0.0, 0.0}, {-0.7, 0.2, 1.4}};
    const double triples[][3] = {{0.4, -0.5, 0.35}, {0.8, 0.3, -0.2}, {-0.2, 0.7, 0.5}};
    const int permutations[6][3] = {{0, 1, 2}, {0, 2, 1}, {1, 0, 2},
                                    {1, 2, 0}, {2, 0, 1}, {2, 1, 0}};
    for (const auto& p : points) {
        for (const auto& r : triples) {
            const double base = TrivariateNormal{r[0], r[1], r[2]}.cdf(p[0], p[1], p[2]);
            for (const auto& perm : permutations) {
                const double hp[3] = {p[perm[0]], p[perm[1]], p[perm[2]]};
                const double rp[3] = {corrEntry(perm[0], perm[1], r),
                                      corrEntry(perm[0], perm[2], r),
                                      corrEntry(perm[1], perm[2], r)};
                const double got = TrivariateNormal{rp[0], rp[1], rp[2]}.cdf(hp[0], hp[1], hp[2]);
                CHECK_CLOSE("permutation", got, base, 1e-13);
            }
        }
    }
}

TEST(TrivariateNormalDistribution, monotoneAndBounded) {
    const double triples[][3] = {{0.0, 0.0, 0.0}, {0.6, -0.4, 0.3}, {-0.8, 0.5, -0.3}};
    const double hs[] = {-3.0, -1.5, -0.5, 0.0, 0.25, 1.0, 2.0, 3.5};
    for (const auto& r : triples) {
        const TrivariateNormal d{r[0], r[1], r[2]};
        for (double a : hs) {
            for (double b : hs) {
                for (double c : hs) {
                    const double value = d.cdf(a, b, c);
                    EXPECT_GE(value, 0.0);
                    EXPECT_LE(value, 1.0);
                    EXPECT_GE(d.cdf(a + 1e-3, b, c), value - 1e-12);
                    EXPECT_GE(d.cdf(a, b + 1e-3, c), value - 1e-12);
                    EXPECT_GE(d.cdf(a, b, c + 1e-3), value - 1e-12);
                }
            }
        }
    }
}

TEST(TrivariateNormalDistribution, derivativePackAgainstCentralDifferences) {
    const double points[][6] = {
        {0.5, -0.3, 0.9, 0.4, -0.5, 0.5}, {-1.2, 1.7, 0.4, -0.7, 0.25, -0.5},
        {0.3, 0.2, -0.4, 0.6, 0.35, 0.5}, {-2.0, 0.5, 1.5, -0.55, 0.7, -0.6},
        {2.0, -2.0, 1.0, 0.9, 0.85, 0.8},
    };
    const double h = 1e-4;
    for (const auto& p : points) {
        const TvnGradHess pack =
            quantape::math::detail::tvnCdfGradHess(p[0], p[1], p[2], p[3], p[4], p[5]);
        for (int i = 0; i < 6; ++i) {
            CHECK_CLOSE(pointLabel("gradient fd", p), pack.g[i], richFirst(cdfOf, p, i, h), 1e-6);
        }
        for (int i = 0; i < 6; ++i) {
            for (int j = i; j < 6; ++j) {
                const double expected =
                    i == j ? richSecond(cdfOf, p, i, h) : richMixed(cdfOf, p, i, j, h);
                CHECK_CLOSE(pointLabel("Hessian fd", p), pack.h[tvnHessIndex(i, j)], expected,
                            1e-5);
            }
        }
    }
}

TEST(TrivariateNormalDistribution, gradientIdentitiesAndFirstOrderPack) {
    const double points[][6] = {
        {0.5, -0.3, 0.9, 0.4, -0.5, 0.5},
        {-1.2, 1.7, 0.4, -0.7, 0.25, -0.5},
        {2.0, -2.0, 1.0, 0.9, 0.85, 0.8},
    };
    for (const auto& p : points) {
        const TrivariateNormal d{p[3], p[4], p[5]};
        const TvnGradHess pack = d.cdfGradHess(p[0], p[1], p[2]);
        const quantape::math::detail::TvnCdfGrad first = d.cdfGrad(p[0], p[1], p[2]);
        CHECK_CLOSE("first-order value", first.value, pack.value, kTol);
        for (int i = 0; i < 6; ++i) {
            CHECK_CLOSE("first-order gradient", first.g[i], pack.g[i], kTol);
        }

        // d/dh1 = phi(h1) Phi2(conditional); d/dr12 = phi2(h1,h2;r12)
        // Phi(conditional), both independent closed forms.
        const double s12 = std::sqrt(1.0 - p[3] * p[3]);
        const double s13 = std::sqrt(1.0 - p[4] * p[4]);
        const double rho1 = (p[5] - p[3] * p[4]) / (s12 * s13);
        const double phi2_1 = BivariateNormal<double>{rho1}.cdf((p[1] - p[3] * p[0]) / s12,
                                                                (p[2] - p[4] * p[0]) / s13);
        CHECK_CLOSE("g1 identity", pack.g[0],
                    quantape::math::detail::normalPdfDouble(p[0]) * phi2_1, kTol);
        const double det = 1.0 - p[3] * p[3] - p[4] * p[4] - p[5] * p[5] + 2.0 * p[3] * p[4] * p[5];
        const double d12 = 1.0 - p[3] * p[3];
        const double alpha = (p[4] - p[3] * p[5]) / d12;
        const double beta = (p[5] - p[3] * p[4]) / d12;
        const double s = std::sqrt(det / d12);
        const double z = (p[2] - alpha * p[0] - beta * p[1]) / s;
        const double phi2_pdf = quantape::math::detail::bvnPdf(p[0], p[1], p[3]);
        CHECK_CLOSE("g4 identity", pack.g[3], phi2_pdf * phi(z), kTol);
    }

    const double nan = std::numeric_limits<double>::quiet_NaN();
    const TvnGradHess invalid =
        quantape::math::detail::tvnCdfGradHess(0.0, 0.0, 0.0, 1.0, 0.0, 0.0);
    EXPECT_TRUE(std::isnan(invalid.value));
    EXPECT_TRUE(std::isnan(invalid.g[0]));
    EXPECT_TRUE(std::isnan(invalid.h[0]));
    const TvnGradHess nonFinite =
        quantape::math::detail::tvnCdfGradHess(nan, 0.0, 0.0, 0.0, 0.0, 0.0);
    EXPECT_TRUE(std::isnan(nonFinite.g[5]));
}
