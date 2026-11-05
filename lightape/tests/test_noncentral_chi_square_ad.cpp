#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>

#include "lightape/ops/NoncentralChiSquare.h"
#include "lightape/ops/P1Functions.h"
#include <gtest/gtest.h>

namespace {

using lightape::RevScalar;
using lightape::RevTape;
using lightape::Tangent;

using D1 = Tangent<double, 1>;

/// Mixed absolute/relative tolerance: `|got - expected| <= atol + rtol*|expected|`.
void expectMixed(const char* label, double got, double expected, double rtol, double atol) {
    EXPECT_NEAR(got, expected, atol + rtol * std::fabs(expected))
        << label << " got=" << got << " expected=" << expected;
}

// ---------------------------------------------------------------------------
// Tracked reference data (hardcoded from
// quantape/tests/data/distributions/nccs_reference.csv and
// nccs_nu_reference.csv, same owner; 17-20 significant digits).
// ---------------------------------------------------------------------------

struct ValueRow {
    double nu, lambda, x, cdf, pdf;
};

constexpr ValueRow kValueRows[] = {
    {0.5, 0, 1.3499395786223368e-24, 9.99999999999998289e-7, 185193473810978865.0},
    {0.5, 20, 0.5, 0.000138390062214630114, 0.00035178019929772703},
    {1, 20, 0.56751933800029453, 0.0000999999999999999787, 0.000263254999338386587},
    {2, 20, 15.352257685855623, 0.249999999999999671, 0.0411756248154762389},
    {3, 100, 103, 0.519725933228331202, 0.0197272341092720394},
    {10, 100, 161.91717019824191, 0.990000000000005624, 0.00106438836449409799},
};

struct NuRow {
    double nu, lambda, x, cdf, gNu, hNuNu, hNuLambda, hNuX;
};

constexpr NuRow kNuRows[] = {
    {0.5, 0, 0.1, 0.51655532083046562739, -0.7232044811356783278, 0.86439261926471621959,
     0.34344446323647100295, 0.76406295507195233934},
    {1, 1, 1.0, 0.4772498680518207928, -0.24354850544877071347, 0.052467513800687564122,
     0.071748095617013101549, 0.061993854989678811356},
    {2, 5, 0.7, 0.03574740623502126656, -0.03076179994734895394, 0.022577823083055495716,
     0.012629226740010052145, -0.031587616264042633218},
    {7, 1, 7.0, 0.47298971874031617393, -0.10056492480179157512, 0.0052997597703561340247,
     0.0098836138820258865488, 0.0012831014264978480005},
};

// mpmath 1.4.1 (dps = 45) references for the AD checks; the generator was
// cross-validated against the committed CSVs to ~1e-17 relative.
struct DerivRow {
    double nu, lambda, x, value;
    double gNu, gLambda, gx;
    double hNuNu, hNuLambda, hNuX, hLambdaLambda, hLambdaX, hxx;
};

constexpr DerivRow kCdfDerivs[] = {
    {2.5, 3, 1.0, 0.08528475941635898774763196, -0.06298551227446301662467906,
     -0.0345059079444965260555288, 0.1072261946163866005363114, 0.03809316320114866148846517,
     0.0240895072208360696529927, -0.04433438669895451901015789, 0.01375938317972404879484949,
     -0.03636014333594503724039132, 0.02495231326264813894921534},
    {1, 20, 2.0, 0.001114384469944167103653179, -0.0007197344117057192805225932,
     -0.0004157611003685135291201333, 0.001314760481486324058190165, 0.0004203303218071387084128161,
     0.0002573481680695062476272796, -0.0006998630312092492906493737,
     0.0001525365536191533998785617, -0.0004494996905589052645350159,
     0.001092735140727824601958043},
    {7, 1, 7.0, 0.4729897187403161739331967, -0.1005649248017915751192673,
     -0.09198996580003171308883469, 0.1015646495683806219022264, 0.005299759770356134024691786,
     0.009883613882025886548794964, 0.001283101426497848000515501, 0.01248358971079467569754627,
     -0.004787341884174454406695867, -0.007938523809766395051115583},
    {0.5, 5, 0.3, 0.07162598037408681635210105, -0.0696795920989619073118765,
     -0.03216098433310179662466648, 0.1106757459367984634047601, 0.05318104905082781736272518,
     0.03015544944830552221332707, -0.01551175811667150809874897, 0.01436826900510203436429852,
     -0.0392573808018483333900468, -0.06401903503454708532999376},
};

constexpr DerivRow kPdfDerivs[] = {
    {1, 0.5, 0.1, 0.9580535702303737238268948, -0.5179569399568092067565272,
     -0.4319068337548457723356443, -5.033694879465349780324036, -0.8554905572998304497135235,
     0.1874090632245368381047972, 7.281183987144980040117329, 0.193176168348429940651595,
     2.73280085661009777632984, 74.3117698649481634213084},
    {2.5, 3, 1.0, 0.1072261946163866005363114, -0.04433438669895451901015789,
     -0.03636014333594503724039132, 0.02495231326264813894921534, -0.001038969906835841142509631,
     0.0101224397390592246785826, 0.02856243315167782554120615, 0.01130038007811049422277091,
     0.00570391503664844914558799, -0.03026327388258591809408792},
    {5, 20, 4.0, 0.0008342636835255300309661344, -0.0004096448803236336024442423,
     -0.0002696726409243694631399594, 0.0006330130437512865078447724,
     0.0001768045737355733533495282, 0.0001239385687972842178463547, -0.000248930786341517462702535,
     0.00008438750411446401290384401, -0.0001816702014134585223524065,
     0.000332518750463499963537342},
};

/// Direct Poisson-mixture oracle: `sum_j w_j gammaP(nu/2+j, x/2)`.
double poissonMixtureCdf(double nu, double lambda, double x) {
    const double half = 0.5 * lambda;
    double w = std::exp(-half);
    double sum = 0.0;
    for (int j = 0; j <= 2000; ++j) {
        sum += w * lightape::gammaP(0.5 * nu + j, 0.5 * x);
        w *= half / (j + 1.0);
        if (!(w > 1e-18)) {
            break;
        }
    }
    return sum;
}

class NoncentralChiSquareOpsTest : public ::testing::Test {
protected:
    void SetUp() override { RevTape::active().clear(); }
    void TearDown() override { RevTape::active().clear(); }
};

// ---------------------------------------------------------------------------
// Double value kernels against the tracked references.
// ---------------------------------------------------------------------------

TEST_F(NoncentralChiSquareOpsTest, DoubleKernelsMatchReferenceCsv) {
    for (const ValueRow& r : kValueRows) {
        const std::string label = "nu=" + std::to_string(r.nu) +
                                  " lambda=" + std::to_string(r.lambda) +
                                  " x=" + std::to_string(r.x);
        expectMixed((label + " cdf").c_str(), lightape::noncentralChiSquareCdf(r.nu, r.lambda, r.x),
                    r.cdf, 1e-12, 1e-15);
        expectMixed((label + " pdf").c_str(), lightape::noncentralChiSquarePdf(r.nu, r.lambda, r.x),
                    r.pdf, 1e-12, 0.0);
        // Ccdf = 1 - Cdf on the same input.
        const double ccdf = lightape::noncentralChiSquareCcdf(r.nu, r.lambda, r.x);
        expectMixed((label + " ccdf complement").c_str(), ccdf, 1.0 - r.cdf, 1e-12, 1e-15);
    }
}

TEST_F(NoncentralChiSquareOpsTest, DoubleKernelsMatchNuReferenceCsv) {
    for (const NuRow& r : kNuRows) {
        const std::string label = "nu=" + std::to_string(r.nu) +
                                  " lambda=" + std::to_string(r.lambda) +
                                  " x=" + std::to_string(r.x);
        // CDF + nu partials through the `External` factors: gNu is the vjp on
        // the nu parent, hNuNu/hNuLambda/hNuX are rows of the Hessian payload.
        RevTape& tape = RevTape::active();
        RevScalar nu(r.nu, tape.input(r.nu));
        RevScalar lambda(r.lambda, tape.input(r.lambda));
        RevScalar x(r.x, tape.input(r.x));
        const RevScalar f = lightape::noncentralChiSquareCdf(nu, lambda, x);
        expectMixed((label + " cdf").c_str(), f.value, r.cdf, 1e-12, 1e-15);
        tape.reverse(f.node);
        expectMixed((label + " gNu").c_str(), tape.adjoint(nu.node), r.gNu, 1e-12, 1e-15);

        const std::uint32_t inputs[3] = {nu.node, lambda.node, x.node};
        const double e0[3] = {1.0, 0.0, 0.0};
        double out[3] = {0.0, 0.0, 0.0};
        ASSERT_TRUE(tape.hessianVector(f.node, inputs, 3, e0, out));
        expectMixed((label + " hNuNu").c_str(), out[0], r.hNuNu, 1e-12, 1e-14);
        expectMixed((label + " hNuLambda").c_str(), out[1], r.hNuLambda, 1e-12, 1e-14);
        expectMixed((label + " hNuX").c_str(), out[2], r.hNuX, 1e-12, 1e-14);
        tape.clear();
        for (double v : out) {
            EXPECT_TRUE(std::isfinite(v) || std::isnan(r.cdf));
        }
    }
}

// ---------------------------------------------------------------------------
// Known closed forms.
// ---------------------------------------------------------------------------

TEST_F(NoncentralChiSquareOpsTest, KnownClosedForms) {
    // nu = 2, lambda = 0: F(x) = 1 - exp(-x/2), f(x) = exp(-x/2)/2.
    EXPECT_DOUBLE_EQ(lightape::noncentralChiSquareCdf(2.0, 0.0, 2.0), 1.0 - std::exp(-1.0));
    EXPECT_DOUBLE_EQ(lightape::noncentralChiSquareCcdf(2.0, 0.0, 2.0), std::exp(-1.0));
    EXPECT_DOUBLE_EQ(lightape::noncentralChiSquarePdf(2.0, 0.0, 2.0), 0.5 * std::exp(-1.0));

    // nu = 4, lambda = 0: P(2, 1) = 1 - 2/e.
    EXPECT_DOUBLE_EQ(lightape::noncentralChiSquareCdf(4.0, 0.0, 2.0), 1.0 - 2.0 * std::exp(-1.0));

    // nu = 1, lambda = 0: F(x) = erf(sqrt(x/2)), f(x) = x^{-1/2} e^{-x/2}/sqrt(2 pi).
    EXPECT_DOUBLE_EQ(lightape::noncentralChiSquareCdf(1.0, 0.0, 2.0), std::erf(1.0));
    constexpr double kPi = 3.14159265358979323846;
    expectMixed("chi2_1 pdf", lightape::noncentralChiSquarePdf(1.0, 0.0, 2.0),
                std::exp(-1.0) / (2.0 * std::sqrt(kPi)), 1e-14, 0.0);

    // x = 0 limits: f(0) = exp(-lambda/2) at nu = 2, 0 for nu > 2, Inf for nu < 2.
    EXPECT_DOUBLE_EQ(lightape::noncentralChiSquarePdf(2.0, 3.0, 0.0), std::exp(-1.5));
    EXPECT_DOUBLE_EQ(lightape::noncentralChiSquarePdf(4.0, 3.0, 0.0), 0.0);
    EXPECT_TRUE(std::isinf(lightape::noncentralChiSquarePdf(1.0, 3.0, 0.0)));

    // Poisson mixture cross-check (direct sum of the same kernel).
    for (double nu : {0.5, 1.0, 3.5, 10.0}) {
        for (double lambda : {0.25, 1.0, 5.0}) {
            const double x = 2.0;
            expectMixed("mixture cdf vs direct sum",
                        lightape::noncentralChiSquareCdf(nu, lambda, x),
                        poissonMixtureCdf(nu, lambda, x), 1e-13, 1e-15);
        }
    }

    // lambda = 0 reduces to the central incomplete gamma.
    for (double nu : {0.5, 1.0, 5.0}) {
        for (double x : {0.25, 2.0, 11.0}) {
            EXPECT_DOUBLE_EQ(lightape::noncentralChiSquareCdf(nu, 0.0, x),
                             lightape::gammaP(0.5 * nu, 0.5 * x));
            EXPECT_DOUBLE_EQ(lightape::noncentralChiSquareCcdf(nu, 0.0, x),
                             lightape::gammaQ(0.5 * nu, 0.5 * x));
        }
    }
}

TEST_F(NoncentralChiSquareOpsTest, EdgesAndInvalidInputs) {
    const double nan = std::numeric_limits<double>::quiet_NaN();
    EXPECT_TRUE(std::isnan(lightape::noncentralChiSquareCdf(0.0, 1.0, 1.0)));
    EXPECT_TRUE(std::isnan(lightape::noncentralChiSquareCdf(1.0, -1.0, 1.0)));
    EXPECT_TRUE(std::isnan(lightape::noncentralChiSquareCdf(1.0, 1.0, nan)));
    EXPECT_TRUE(std::isnan(lightape::noncentralChiSquarePdf(1.0, 1.0, -0.5)));

    EXPECT_DOUBLE_EQ(lightape::noncentralChiSquareCdf(1.0, 1.0, -1.0), 0.0);
    EXPECT_DOUBLE_EQ(lightape::noncentralChiSquareCcdf(1.0, 1.0, -1.0), 1.0);
    EXPECT_DOUBLE_EQ(lightape::noncentralChiSquareCdf(1.0, 1.0, 0.0), 0.0);
    EXPECT_DOUBLE_EQ(lightape::noncentralChiSquareCcdf(1.0, 1.0, 0.0), 1.0);
    EXPECT_DOUBLE_EQ(
        lightape::noncentralChiSquareCdf(1.0, 1.0, std::numeric_limits<double>::infinity()), 1.0);
    EXPECT_DOUBLE_EQ(
        lightape::noncentralChiSquareCcdf(1.0, 1.0, std::numeric_limits<double>::infinity()), 0.0);
    EXPECT_DOUBLE_EQ(
        lightape::noncentralChiSquarePdf(1.0, 1.0, std::numeric_limits<double>::infinity()), 0.0);
}

// ---------------------------------------------------------------------------
// RevScalar adjoints vs mpmath references.
// ---------------------------------------------------------------------------

TEST_F(NoncentralChiSquareOpsTest, CdfRevScalarGradientsMatchMpmath) {
    for (const DerivRow& r : kCdfDerivs) {
        RevTape& tape = RevTape::active();
        RevScalar nu(r.nu, tape.input(r.nu));
        RevScalar lambda(r.lambda, tape.input(r.lambda));
        RevScalar x(r.x, tape.input(r.x));
        const RevScalar f = lightape::noncentralChiSquareCdf(nu, lambda, x);
        expectMixed("cdf value", f.value, r.value, 1e-13, 1e-15);
        tape.reverse(f.node);
        expectMixed("dF/dnu", tape.adjoint(nu.node), r.gNu, 1e-12, 1e-14);
        expectMixed("dF/dlambda", tape.adjoint(lambda.node), r.gLambda, 1e-12, 1e-14);
        expectMixed("dF/dx", tape.adjoint(x.node), r.gx, 1e-12, 1e-14);
        tape.clear();
    }
}

TEST_F(NoncentralChiSquareOpsTest, CcdfRevScalarGradientsMatchMpmath) {
    for (const DerivRow& r : kCdfDerivs) {
        RevTape& tape = RevTape::active();
        RevScalar nu(r.nu, tape.input(r.nu));
        RevScalar lambda(r.lambda, tape.input(r.lambda));
        RevScalar x(r.x, tape.input(r.x));
        const RevScalar q = lightape::noncentralChiSquareCcdf(nu, lambda, x);
        expectMixed("ccdf value", q.value, 1.0 - r.value, 1e-12, 1e-15);
        tape.reverse(q.node);
        expectMixed("dQ/dnu", tape.adjoint(nu.node), -r.gNu, 1e-12, 1e-14);
        expectMixed("dQ/dlambda", tape.adjoint(lambda.node), -r.gLambda, 1e-12, 1e-14);
        expectMixed("dQ/dx", tape.adjoint(x.node), -r.gx, 1e-12, 1e-14);
        tape.clear();
    }
}

TEST_F(NoncentralChiSquareOpsTest, PdfRevScalarGradientsMatchMpmath) {
    for (const DerivRow& r : kPdfDerivs) {
        RevTape& tape = RevTape::active();
        RevScalar nu(r.nu, tape.input(r.nu));
        RevScalar lambda(r.lambda, tape.input(r.lambda));
        RevScalar x(r.x, tape.input(r.x));
        const RevScalar f = lightape::noncentralChiSquarePdf(nu, lambda, x);
        expectMixed("pdf value", f.value, r.value, 1e-13, 1e-15);
        tape.reverse(f.node);
        expectMixed("df/dnu", tape.adjoint(nu.node), r.gNu, 1e-12, 1e-14);
        expectMixed("df/dlambda", tape.adjoint(lambda.node), r.gLambda, 1e-12, 1e-14);
        expectMixed("df/dx", tape.adjoint(x.node), r.gx, 1e-12, 1e-14);
        tape.clear();
    }
}

// ---------------------------------------------------------------------------
// hessianVector: every wrapper records an exact vhp, so the replay succeeds.
// ---------------------------------------------------------------------------

TEST_F(NoncentralChiSquareOpsTest, CdfHessianVectorMatchesMpmath) {
    const double v[3] = {0.7, -0.4, 0.9};
    for (const DerivRow& r : kCdfDerivs) {
        RevTape& tape = RevTape::active();
        RevScalar nu(r.nu, tape.input(r.nu));
        RevScalar lambda(r.lambda, tape.input(r.lambda));
        RevScalar x(r.x, tape.input(r.x));
        const RevScalar f = lightape::noncentralChiSquareCdf(nu, lambda, x);

        const std::uint32_t inputs[3] = {nu.node, lambda.node, x.node};
        double out[3] = {0.0, 0.0, 0.0};
        ASSERT_TRUE(tape.hessianVector(f.node, inputs, 3, v, out));

        const double rowNu[3] = {r.hNuNu, r.hNuLambda, r.hNuX};
        const double rowLambda[3] = {r.hNuLambda, r.hLambdaLambda, r.hLambdaX};
        const double rowX[3] = {r.hNuX, r.hLambdaX, r.hxx};
        expectMixed("H nu row", out[0], rowNu[0] * v[0] + rowNu[1] * v[1] + rowNu[2] * v[2], 1e-12,
                    1e-13);
        expectMixed("H lambda row", out[1],
                    rowLambda[0] * v[0] + rowLambda[1] * v[1] + rowLambda[2] * v[2], 1e-12, 1e-13);
        expectMixed("H x row", out[2], rowX[0] * v[0] + rowX[1] * v[1] + rowX[2] * v[2], 1e-12,
                    1e-13);
        tape.clear();
    }
}

TEST_F(NoncentralChiSquareOpsTest, CcdfHessianVectorMatchesMpmath) {
    const double v[3] = {-0.3, 0.8, 0.5};
    for (const DerivRow& r : kCdfDerivs) {
        RevTape& tape = RevTape::active();
        RevScalar nu(r.nu, tape.input(r.nu));
        RevScalar lambda(r.lambda, tape.input(r.lambda));
        RevScalar x(r.x, tape.input(r.x));
        const RevScalar q = lightape::noncentralChiSquareCcdf(nu, lambda, x);

        const std::uint32_t inputs[3] = {nu.node, lambda.node, x.node};
        double out[3] = {0.0, 0.0, 0.0};
        ASSERT_TRUE(tape.hessianVector(q.node, inputs, 3, v, out));

        const double rowNu[3] = {r.hNuNu, r.hNuLambda, r.hNuX};
        const double rowLambda[3] = {r.hNuLambda, r.hLambdaLambda, r.hLambdaX};
        const double rowX[3] = {r.hNuX, r.hLambdaX, r.hxx};
        expectMixed("H nu row", out[0], -(rowNu[0] * v[0] + rowNu[1] * v[1] + rowNu[2] * v[2]),
                    1e-12, 1e-13);
        expectMixed("H lambda row", out[1],
                    -(rowLambda[0] * v[0] + rowLambda[1] * v[1] + rowLambda[2] * v[2]), 1e-12,
                    1e-13);
        expectMixed("H x row", out[2], -(rowX[0] * v[0] + rowX[1] * v[1] + rowX[2] * v[2]), 1e-12,
                    1e-13);
        tape.clear();
    }
}

TEST_F(NoncentralChiSquareOpsTest, PdfHessianVectorMatchesMpmath) {
    const double v[3] = {-0.3, 0.8, 0.5};
    for (const DerivRow& r : kPdfDerivs) {
        RevTape& tape = RevTape::active();
        RevScalar nu(r.nu, tape.input(r.nu));
        RevScalar lambda(r.lambda, tape.input(r.lambda));
        RevScalar x(r.x, tape.input(r.x));
        const RevScalar f = lightape::noncentralChiSquarePdf(nu, lambda, x);

        const std::uint32_t inputs[3] = {nu.node, lambda.node, x.node};
        double out[3] = {0.0, 0.0, 0.0};
        ASSERT_TRUE(tape.hessianVector(f.node, inputs, 3, v, out));

        const double rowNu[3] = {r.hNuNu, r.hNuLambda, r.hNuX};
        const double rowLambda[3] = {r.hNuLambda, r.hLambdaLambda, r.hLambdaX};
        const double rowX[3] = {r.hNuX, r.hLambdaX, r.hxx};
        expectMixed("H nu row", out[0], rowNu[0] * v[0] + rowNu[1] * v[1] + rowNu[2] * v[2], 1e-12,
                    1e-13);
        expectMixed("H lambda row", out[1],
                    rowLambda[0] * v[0] + rowLambda[1] * v[1] + rowLambda[2] * v[2], 1e-12, 1e-13);
        expectMixed("H x row", out[2], rowX[0] * v[0] + rowX[1] * v[1] + rowX[2] * v[2], 1e-12,
                    1e-13);
        tape.clear();
    }
}

TEST_F(NoncentralChiSquareOpsTest, FirstOrderAtBoundariesStaysHonest) {
    RevTape& tape = RevTape::active();
    RevScalar nu(2.0, tape.input(2.0));
    RevScalar lambda(1.0, tape.input(1.0));
    RevScalar x(0.0, tape.input(0.0));
    const RevScalar q = lightape::noncentralChiSquareCcdf(nu, lambda, x);
    EXPECT_DOUBLE_EQ(q.value, 1.0);
    tape.reverse(q.node);
    // x = 0: the one-sided parameter/mixed limits are not represented, so the
    // precomputed factors are NaN (fail loud, never a fabricated finite value).
    EXPECT_TRUE(std::isnan(tape.adjoint(nu.node)));
    EXPECT_TRUE(std::isnan(tape.adjoint(x.node)));
    tape.clear();
}

// ---------------------------------------------------------------------------
// Tangent directional derivatives: same factors at the primal.
// ---------------------------------------------------------------------------

TEST_F(NoncentralChiSquareOpsTest, TangentDirectionalDerivatives) {
    for (const DerivRow& r : kCdfDerivs) {
        const D1 nu(r.nu, {0.3});
        const D1 lambda(r.lambda, {-0.5});
        const D1 x(r.x, {0.8});
        const D1 f = lightape::noncentralChiSquareCdf(nu, lambda, x);
        expectMixed("cdf tangent value", f.value, r.value, 1e-13, 1e-15);
        expectMixed("cdf tangent d", f.d[0], r.gNu * 0.3 + r.gLambda * -0.5 + r.gx * 0.8, 1e-12,
                    1e-14);

        const D1 q = lightape::noncentralChiSquareCcdf(nu, lambda, x);
        expectMixed("ccdf tangent d", q.d[0], -(r.gNu * 0.3 + r.gLambda * -0.5 + r.gx * 0.8), 1e-12,
                    1e-14);
    }
    for (const DerivRow& r : kPdfDerivs) {
        const D1 nu(r.nu, {-0.4});
        const D1 lambda(r.lambda, {0.6});
        const D1 x(r.x, {0.2});
        const D1 f = lightape::noncentralChiSquarePdf(nu, lambda, x);
        expectMixed("pdf tangent value", f.value, r.value, 1e-13, 1e-15);
        expectMixed("pdf tangent d", f.d[0], r.gNu * -0.4 + r.gLambda * 0.6 + r.gx * 0.2, 1e-12,
                    1e-14);
    }
}

} // namespace
