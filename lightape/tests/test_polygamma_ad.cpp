#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>

#include "lightape/lightape.h"
#include "lightape/mix/FwdRev.h"
#include <gtest/gtest.h>

namespace {

using lightape::digamma;
using lightape::FwdRevScalar;
using lightape::RevScalar;
using lightape::RevTape;
using lightape::Tangent;
using lightape::tetragamma;
using lightape::trigamma;

/// mpmath (50 dps) references: `psi^(n)(x)` for n = 0..4 (digamma, trigamma,
/// tetragamma, pentagamma, psi4). Points cover the recurrence regime
/// (x < 12) and the asymptotic regime (x >= 12).
struct PolygammaRef {
    double x;
    double psi[5];
};

constexpr PolygammaRef kRefs[] = {
    {0.25,
     {-4.22745353337626541, 17.1973291545071107, -129.32773993753692, 1538.7821440091884,
      -24584.3753886379337}},
    {0.5,
     {-1.96351002602142348, 4.93480220054467931, -16.82879664423432, 97.4090910340024372,
      -771.474249826667225}},
    {1.0,
     {-0.577215664901532861, 1.64493406684822644, -2.40411380631918857, 6.49393940226682915,
      -24.8862661234408782}},
    {2.5,
     {0.703156640645243187, 0.490357756100234865, -0.236204051641727403, 0.223905848817252051,
      -0.313755999506731363}},
    {7.3,
     {1.9178203356379861, 0.146795768131427098, -0.0215108144416202509, 0.00629315871319848922,
      -0.00275689561937467106}},
    {11.9,
     {2.43393353688253745, 0.0876632011900041863, -0.00767993915932830737, 0.00134478392369591216,
      -0.000352991398445411578}},
    {12.0,
     {2.44266167997581202, 0.0869018728717683908, -0.00754720536899891194, 0.00131009323107082597,
      -0.000340910050701089393}},
    {30.0,
     {3.38443813268552488, 0.0338950603577399442, -0.00114876520372859984, 0.0000778600366473612906,
      -7.91494134018102404e-6}},
};

/// Absolute 1e-13 for small magnitudes, relative 1e-13 above 1.
inline double tol(double ref) {
    return 1e-13 * std::max(1.0, std::fabs(ref));
}

class PolygammaAdTest : public ::testing::Test {
protected:
    void SetUp() override { RevTape::active().clear(); }
    void TearDown() override { RevTape::active().clear(); }
};

// ── double kernels vs mpmath ──

TEST_F(PolygammaAdTest, ValuesMatchMpmath) {
    for (const PolygammaRef& r : kRefs) {
        SCOPED_TRACE(::testing::Message() << "x=" << r.x);
        for (int n = 0; n <= 4; ++n) {
            const double got = lightape::polygamma(n, r.x);
            EXPECT_NEAR(got, r.psi[n], tol(r.psi[n])) << "n=" << n;
        }
        EXPECT_NEAR(digamma(r.x), r.psi[0], tol(r.psi[0]));
        EXPECT_NEAR(trigamma(r.x), r.psi[1], tol(r.psi[1]));
        EXPECT_NEAR(tetragamma(r.x), r.psi[2], tol(r.psi[2]));
        EXPECT_NEAR(lightape::pentagamma(r.x), r.psi[3], tol(r.psi[3]));
    }
}

TEST_F(PolygammaAdTest, RecurrenceAndPoles) {
    for (int n = 0; n <= 4; ++n) {
        for (const double x : {0.3, 0.9, 1.25, 2.7, 6.4, 11.999, 12.5}) {
            SCOPED_TRACE(::testing::Message() << "n=" << n << " x=" << x);
            const double step = lightape::polygamma(n, x + 1.0) - lightape::polygamma(n, x);
            const double nFact = (n == 0)   ? 1.0
                                 : (n == 1) ? 1.0
                                 : (n == 2) ? 2.0
                                 : (n == 3) ? 6.0
                                            : 24.0;
            const double expected = ((n % 2 == 0) ? 1.0 : -1.0) * nFact / std::pow(x, n + 1);
            EXPECT_NEAR(step, expected, 1e-12 * std::max(1.0, std::fabs(expected)));
        }
    }
    const double nan = std::numeric_limits<double>::quiet_NaN();
    for (const double x : {0.0, -0.0, -1.0, -2.0, -0.5, nan}) {
        EXPECT_TRUE(std::isnan(lightape::tetragamma(x)));
        EXPECT_TRUE(std::isnan(lightape::pentagamma(x)));
        EXPECT_TRUE(std::isnan(lightape::polygamma(3, x)));
        EXPECT_TRUE(std::isnan(lightape::polygamma(4, x)));
    }
    EXPECT_TRUE(std::isnan(lightape::polygamma(-1, 1.0)));
}

// ── RevScalar adjoints ──

TEST_F(PolygammaAdTest, RevScalarVjpMatchesMpmath) {
    const double xv = 0.7;
    const double value[] = {-1.22002355369793461, 2.83404915669461063, -6.43499287419092255};
    const double vjp[] = {2.83404915669461063, -6.43499287419092255, 25.8791496784277316};
    using Kernel = RevScalar (*)(const RevScalar&);
    const Kernel kernels[] = {&lightape::digamma, &lightape::trigamma, &lightape::tetragamma};
    const char* labels[] = {"digamma", "trigamma", "tetragamma"};

    for (int i = 0; i < 3; ++i) {
        SCOPED_TRACE(labels[i]);
        RevTape& tape = RevTape::active();
        tape.clear();
        const RevScalar x(xv, tape.input(xv));
        const RevScalar y = kernels[i](x);
        EXPECT_NEAR(y.value, value[i], tol(value[i]));
        tape.reverse(y.node);
        EXPECT_NEAR(tape.adjoint(x.node), vjp[i], tol(vjp[i]));
    }
}

// ── replay Hessian-vector products vs mpmath second derivatives ──

TEST_F(PolygammaAdTest, ReplayHvpMatchesMpmathSecondDerivative) {
    const double xv = 0.7;
    const double v = 0.6;
    const double second[] = {-6.43499287419092255, 25.8791496784277316, -144.70825364302461};
    using Kernel = RevScalar (*)(const RevScalar&);
    const Kernel kernels[] = {&lightape::digamma, &lightape::trigamma, &lightape::tetragamma};
    const char* labels[] = {"digamma", "trigamma", "tetragamma"};

    for (int i = 0; i < 3; ++i) {
        SCOPED_TRACE(labels[i]);
        RevTape& tape = RevTape::active();
        tape.clear();
        const RevScalar x(xv, tape.input(xv));
        const RevScalar y = kernels[i](x);
        const std::uint32_t inputs[1] = {x.node};
        double out[1] = {0.0};
        ASSERT_TRUE(tape.hessianVector(y.node, inputs, 1, &v, out));
        EXPECT_NEAR(out[0], second[i] * v, tol(second[i] * v));
    }
}

// ── plain and nested Tangents ──

TEST_F(PolygammaAdTest, PlainTangentFirstDerivatives) {
    const double xv = 0.7;
    const Tangent<double, 1> t(xv, {1.0});
    EXPECT_NEAR(digamma(t).d[0], 2.83404915669461063, 1e-13);
    EXPECT_NEAR(trigamma(t).d[0], -6.43499287419092255, 1e-13);
    EXPECT_NEAR(tetragamma(t).d[0], 25.8791496784277316, 1e-13);
}

TEST_F(PolygammaAdTest, NestedTangentSecondDerivatives) {
    using Inner = Tangent<double, 1>;
    using Outer = Tangent<Inner, 1>;
    Outer t;
    t.value.value = 0.7;
    t.value.d[0] = 1.0;
    t.d[0].value = 1.0;
    t.d[0].d[0] = 0.0;

    const Outer d = digamma(t);
    EXPECT_NEAR(d.value.value, -1.22002355369793461, 1e-13);
    EXPECT_NEAR(d.d[0].value, 2.83404915669461063, 1e-13);
    EXPECT_NEAR(d.d[0].d[0], -6.43499287419092255, 1e-13);

    const Outer tr = trigamma(t);
    EXPECT_NEAR(tr.value.value, 2.83404915669461063, 1e-13);
    EXPECT_NEAR(tr.d[0].value, -6.43499287419092255, 1e-13);
    EXPECT_NEAR(tr.d[0].d[0], 25.8791496784277316, 1e-13);
}

// ── end to end: fwd-over-rev gamma chain ──

/// f(x) = w0 lgamma + w1 tgamma + w2 digamma + w3 trigamma at x = 2.3.
/// `FwdRevScalar<2>` mode mixing (`reverseHvps`) must match the replay
/// `hessianVector` and the mpmath-derived f'' = w0 psi' + w1 Gamma(psi^2+psi')
/// + w2 psi'' + w3 psi'''.
TEST_F(PolygammaAdTest, FwdRevGammaChainHvpMatchesReplayAndMpmath) {
    constexpr std::size_t kDirs = 2;
    RevTape& tape = RevTape::active();
    const double xv = 2.3;
    const double dirs[kDirs] = {0.7, -0.45};
    const double w[4] = {1.1, -0.6, 0.9, 1.7};

    const FwdRevScalar<kDirs> x(RevScalar(xv, tape.input(xv)), {dirs[0], dirs[1]});
    const FwdRevScalar<kDirs> f =
        w[0] * lgamma(x) + w[1] * tgamma(x) + w[2] * digamma(x) + w[3] * trigamma(x);

    // Values (mpmath).
    const double lg = 0.154189454959630581;
    const double gam = 1.16671190519816035;
    const double psi = 0.600039880363969575;
    const double tri = 0.542537458665258408;
    const double expectedValue = w[0] * lg + w[1] * gam + w[2] * psi + w[3] * tri;
    EXPECT_NEAR(f.value.value, expectedValue, tol(expectedValue));

    const std::uint32_t inputs[1] = {x.value.node};
    const std::size_t recorded = tape.size();
    double mixed[kDirs] = {};
    lightape::reverseHvps(f, 1, inputs, mixed);
    EXPECT_EQ(tape.size(), recorded); // sweeps record nothing

    const double tetra = -0.288130243372939752;
    const double penta = 0.30025713443427237;
    const double gampp = 1.05305703444215114;
    const double fpp = w[0] * tri + w[1] * gampp + w[2] * tetra + w[3] * penta;

    for (std::size_t k = 0; k < kDirs; ++k) {
        SCOPED_TRACE(::testing::Message() << "direction " << k);
        const double expected = fpp * dirs[k];
        EXPECT_NEAR(mixed[k], expected, tol(expected));
        double replay[1] = {0.0};
        ASSERT_TRUE(tape.hessianVector(f.value.node, inputs, 1, &dirs[k], replay));
        EXPECT_NEAR(mixed[k], replay[0], tol(expected));
        EXPECT_NEAR(replay[0], expected, tol(expected));
    }
}

} // namespace
