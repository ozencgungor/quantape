#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>

#include "lightape/base/BivariateNormal.h"
#include "lightape/ops/BivariateNormal.h"
#include <gtest/gtest.h>

namespace {

using lightape::RevScalar;
using lightape::RevTape;
using lightape::Tangent;

using D1 = Tangent<double, 1>;
using D2 = Tangent<D1, 1>;

// High-precision (mpmath, 60 dps) reference values; the tracked CSV
// tests/data/distributions/bvn_reference.csv agrees with the generator to
// < 5e-21, and the two rows below are taken verbatim from it.
struct BvnRef {
    double a;
    double b;
    double rho;
    double cdf;
};

constexpr BvnRef kBvnRefs[] = {
    {0.0, 0.0, 0.0, 0.25},
    {0.0, 0.0, -0.5, 0.1666666666666666666666667},
    {0.0, 0.0, 0.5, 0.3333333333333333333333333},
    {0.0, 0.0, 0.1, 0.2659421402146299628671463},
    {0.5, -1.25, 0.3, 0.09035075536163286962336887},
    {-0.75, 1.5, -0.8, 0.1674741005716130031725631},
    {2.0, -1.0, 0.75, 0.1586546363138815336731849},
    {1.25, 2.0, 0.95, 0.8942521644044758904603106},
    {-2.0, 0.5, -0.99, 3.318783820923575604372953e-29},
    {3.0, -1.0, 0.6, 0.1586550837268695453565966},
    {-1.5, -0.5, 0.95, 0.066790910076327419418},
    {-2.0, 2.0, 0.99, 0.022750131948179207200},
};

// Closed-form partials of the true Phi2 (same formulas as the External pack).
double normPdfStd(double x) {
    return 0.39894228040143267794 * std::exp(-0.5 * x * x);
}

double normCdfStd(double x) {
    return 0.5 * std::erfc(-0.70710678118654752440 * x);
}

double bvnPdfStd(double a, double b, double rho) {
    const double s2 = (1.0 - rho) * (1.0 + rho);
    return std::exp(-(a * a - 2.0 * rho * a * b + b * b) / (2.0 * s2)) /
           (6.28318530717958647693 * std::sqrt(s2));
}

double partialA(double a, double b, double rho) {
    const double s = std::sqrt((1.0 - rho) * (1.0 + rho));
    return normPdfStd(a) * normCdfStd((b - rho * a) / s);
}

double partialB(double a, double b, double rho) {
    const double s = std::sqrt((1.0 - rho) * (1.0 + rho));
    return normPdfStd(b) * normCdfStd((a - rho * b) / s);
}

double partialRho(double a, double b, double rho) {
    return bvnPdfStd(a, b, rho);
}

// mpmath (60 dps, mp.diff on the exact Plackett integral) gradient/Hessian at
// (0.4, -0.8, 0.35) and (-1.1, 0.6, -0.6). Order: ga, gb, gr then
// haa, hab, har, hbb, hbr, hrr.
struct BvnGradHessRef {
    double a;
    double b;
    double rho;
    double ga;
    double gb;
    double gr;
    double haa;
    double hab;
    double har;
    double hbb;
    double hbr;
    double hrr;
};

constexpr BvnGradHessRef kGradHessRefs[] = {
    {0.4, -0.8, 0.35, 0.0581193363243045177817539, 0.2219194725321626278143066,
     0.09479736285777276597027768, -0.05642681152994227438788296, 0.09479736285777276597027768,
     -0.07346120426585240069789653, 0.1443565010255096461219767, 0.1015493117792665555991681,
     -0.04088256981160204355177386},
    {-1.1, 0.6, -0.6, 0.1024139109320930010678678, 0.05914168677448771945220785,
     0.108332940108462445042571, 0.1776550660903797748909053, 0.108332940108462445042571,
     0.1252599620004097164099595, 0.02951475200038483426195267, 0.01015621313516836244457767,
     -0.08981900991414511382055103},
};

D2 Seed2(double v) {
    D2 x(D1(v, {1.0}));
    x.d[0] = D1(1.0);
    return x;
}

class BvnAdTest : public ::testing::Test {
protected:
    void SetUp() override { RevTape::active().clear(); }
    void TearDown() override { RevTape::active().clear(); }
};

TEST_F(BvnAdTest, DoubleKernelMatchesHighPrecisionReferences) {
    for (const BvnRef& r : kBvnRefs) {
        SCOPED_TRACE(r.a);
        EXPECT_NEAR(lightape::bvnCdf(r.a, r.b, r.rho), r.cdf, 1e-14);
    }
}

TEST_F(BvnAdTest, DoubleKernelEdgeCases) {
    const double inf = std::numeric_limits<double>::infinity();
    const double nan = std::numeric_limits<double>::quiet_NaN();
    EXPECT_DOUBLE_EQ(lightape::bvnCdf(0.0, 0.0, 0.0), 0.25);
    EXPECT_DOUBLE_EQ(lightape::bvnCdf(0.3, 0.7, 1.0), normCdfStd(0.3));
    EXPECT_DOUBLE_EQ(lightape::bvnCdf(0.3, 0.7, -1.0),
                     std::fmax(normCdfStd(0.3) + normCdfStd(0.7) - 1.0, 0.0));
    EXPECT_DOUBLE_EQ(lightape::bvnCdf(inf, 0.4, 0.5), normCdfStd(0.4));
    EXPECT_DOUBLE_EQ(lightape::bvnCdf(-inf, 0.4, 0.5), 0.0);
    EXPECT_DOUBLE_EQ(lightape::bvnCdf(0.4, inf, 0.5), normCdfStd(0.4));
    EXPECT_DOUBLE_EQ(lightape::bvnCdf(0.4, -inf, 0.5), 0.0);
    EXPECT_TRUE(std::isnan(lightape::bvnCdf(nan, 0.0, 0.5)));
    EXPECT_TRUE(std::isnan(lightape::bvnCdf(0.0, nan, 0.5)));
    EXPECT_TRUE(std::isnan(lightape::bvnCdf(0.0, 0.0, nan)));
    EXPECT_TRUE(std::isnan(lightape::bvnCdf(0.0, 0.0, 1.5)));
    EXPECT_TRUE(std::isnan(lightape::bvnCdf(0.0, 0.0, -1.5)));
}

TEST_F(BvnAdTest, ReverseAdjointsMatchClosedForms) {
    RevTape& tape = RevTape::active();
    for (const BvnGradHessRef& r : kGradHessRefs) {
        SCOPED_TRACE(r.a);
        tape.clear();
        RevScalar a(r.a, tape.input(r.a));
        RevScalar b(r.b, tape.input(r.b));
        RevScalar rho(r.rho, tape.input(r.rho));
        const RevScalar f = lightape::bvnCdf(a, b, rho);
        EXPECT_NEAR(f.value, lightape::bvnCdf(r.a, r.b, r.rho), 0.0);
        EXPECT_EQ(tape.size(), 4u); // three inputs + one External block
        tape.reverse(f.node);
        EXPECT_NEAR(tape.adjoint(a.node), partialA(r.a, r.b, r.rho), 1e-12);
        EXPECT_NEAR(tape.adjoint(b.node), partialB(r.a, r.b, r.rho), 1e-12);
        EXPECT_NEAR(tape.adjoint(rho.node), partialRho(r.a, r.b, r.rho), 1e-12);
        EXPECT_NEAR(tape.adjoint(a.node), r.ga, 1e-12);
        EXPECT_NEAR(tape.adjoint(b.node), r.gb, 1e-12);
        EXPECT_NEAR(tape.adjoint(rho.node), r.gr, 1e-12);
    }
}

TEST_F(BvnAdTest, HessianVectorMatchesHighPrecisionHessian) {
    RevTape& tape = RevTape::active();
    for (const BvnGradHessRef& r : kGradHessRefs) {
        SCOPED_TRACE(r.a);
        tape.clear();
        RevScalar a(r.a, tape.input(r.a));
        RevScalar b(r.b, tape.input(r.b));
        RevScalar rho(r.rho, tape.input(r.rho));
        const RevScalar f = lightape::bvnCdf(a, b, rho);

        const std::uint32_t inputs[3] = {a.node, b.node, rho.node};
        const double h[3][3] = {
            {r.haa, r.hab, r.har}, {r.hab, r.hbb, r.hbr}, {r.har, r.hbr, r.hrr}};
        for (int col = 0; col < 3; ++col) {
            double v[3] = {0.0, 0.0, 0.0};
            v[col] = 1.0;
            double out[3] = {0.0, 0.0, 0.0};
            ASSERT_TRUE(tape.hessianVector(f.node, inputs, 3, v, out));
            for (int row = 0; row < 3; ++row) {
                EXPECT_NEAR(out[row], h[row][col], 1e-12) << "row=" << row << " col=" << col;
            }
        }
        tape.resetAdjoints();
    }
}

TEST_F(BvnAdTest, TangentDirectionalDerivative) {
    const double a = 0.4;
    const double b = -0.8;
    const double rho = 0.35;
    const double ta = 0.7;
    const double tb = -1.3;
    const double tr = 0.4;

    const D1 da(a, {ta});
    const D1 db(b, {tb});
    const D1 dr(rho, {tr});
    const D1 f = lightape::bvnCdf(da, db, dr);

    EXPECT_DOUBLE_EQ(f.value, lightape::bvnCdf(a, b, rho));
    const double expected =
        partialA(a, b, rho) * ta + partialB(a, b, rho) * tb + partialRho(a, b, rho) * tr;
    EXPECT_NEAR(f.d[0], expected, 1e-12);
    EXPECT_NEAR(f.d[0], -0.20989283372168913, 1e-12);
}

TEST_F(BvnAdTest, NestedTangentSecondOrder) {
    const double a = 0.4;
    const double b = -0.8;
    const double rho = 0.35;

    const D2 f = lightape::bvnCdf(Seed2(a), Seed2(b), Seed2(rho));

    // Directional first derivative along (1,1,1): analytic in both slots, and
    // the value path (generic kernel) agrees to quadrature accuracy.
    const double gradSum = 0.37483617171423991156633818;
    const double direct = lightape::bvnCdf(a, b, rho);
    // The nested-Tangent generic kernel and the direct double kernel evaluate
    // the same quadrature in different (but equivalent) order, so allow a few
    // ulps; glibc/FMA contraction across compilers puts them 1 ulp apart.
    EXPECT_NEAR(f.value.value, direct,
                4.0 * std::numeric_limits<double>::epsilon() * std::fabs(direct));
    EXPECT_NEAR(f.d[0].value, gradSum, 1e-12);
    EXPECT_NEAR(f.value.d[0], gradSum, 1e-10);

    // Second directional derivative: v^T H v with v = (1,1,1).
    const double vtHv = 0.29281806042633916992541838;
    EXPECT_NEAR(f.d[0].d[0], vtHv, 1e-12);
}

TEST_F(BvnAdTest, ReverseConstantsCollapse) {
    const RevScalar f = lightape::bvnCdf(RevScalar(0.4), RevScalar(-0.8), RevScalar(0.35));
    EXPECT_EQ(f.node, RevTape::kNone);
    EXPECT_DOUBLE_EQ(f.value, lightape::bvnCdf(0.4, -0.8, 0.35));
    EXPECT_EQ(RevTape::active().size(), 0u);
}

} // namespace
