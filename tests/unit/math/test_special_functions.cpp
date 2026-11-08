// test_special_functions.cpp — complex Si/Ci over the whole |z| range
//
// The complex Si/Ci are valid for every |z|: the Heston asymptotic control
// variate evaluates them at z = phi/2 + i mu/2 with
// mu = ln(S/K) + (r-q)T, which grows without bound as the strike K -> 0
// while the CV parameter phi stays small (K = 1e-8 already gives mu ~ 18.4,
// |z| ~ 9.2; K = 1e-300 gives |z| ~ 350).
//
// Reference values are the defining power series
//   Si(z) = sum (-1)^k z^(2k+1) / ((2k+1)! (2k+1)),
//   Ci(z) = gamma + log z + sum (-1)^k z^(2k) / ((2k)! (2k))
// evaluated at 260 decimal digits with Python's stdlib `decimal` (no
// mpmath/scipy needed), then rounded to double. The defining series converges
// everywhere; the |z| <= 200 grid points need hundreds of terms and ~90
// digits of cancellation, all covered by that precision.
//
// The grid exercises the plain-double series (|z| <= 8), the compensated
// series (8 < |z| <= 40), the asymptotic expansion (|z| > 40), the real
// axis, moderate imaginary parts, the near-imaginary axis the pricer uses,
// the negative-real reflection (Im z = 0 is the upper side of the Ci cut),
// and the exact Heston CV argument for K = 1e-8.

#include "quantape/math/SpecialFunctions/TrigIntegrals.h"
#include "quantape/util/Constants.h"

#include <algorithm>
#include <cmath>
#include <complex>
#include <initializer_list>
#include <ostream>

#include "support/GtestSupport.h"

using quantape::math::cosIntegral;
using quantape::math::sinIntegral;

namespace {

struct SiCiCase {
    const char* id;
    double re;
    double im;
    double siRe;
    double siIm;
    double ciRe;
    double ciIm;
};

void PrintTo(const SiCiCase& value, std::ostream* os) {
    *os << value.id;
}

class SpecialFunctionsSiCiTest : public ::testing::TestWithParam<SiCiCase> {};

/// |got - (re + i im)| / |re + i im|; the references are all nonzero.
double relativeDistance(const std::complex<double>& got, double re, double im) {
    return std::hypot(got.real() - re, got.imag() - im) / std::hypot(re, im);
}

} // namespace

TEST_P(SpecialFunctionsSiCiTest, matchesHighPrecisionReference) {
    const SiCiCase& c = GetParam();
    SCOPED_TRACE(c.id);
    const std::complex<double> z(c.re, c.im);
    const std::complex<double> si = sinIntegral(z);
    const std::complex<double> ci = cosIntegral(z);
    EXPECT_LT(relativeDistance(si, c.siRe, c.siIm), 1e-12) << "Si(" << z << ")";
    EXPECT_LT(relativeDistance(ci, c.ciRe, c.ciIm), 1e-12) << "Ci(" << z << ")";
}

INSTANTIATE_TEST_SUITE_P(
    All, SpecialFunctionsSiCiTest,
    ::testing::Values(
        // Pipe boundary of the plain-double series and the compensated series.
        SiCiCase{"z8p000_0", 8.0, 0.0, 1.5741868217069421, 0.0, 0.12243388253200956, 0.0},
        SiCiCase{"z8p000_1", 8.0, 1.0, 1.5936765298529996, 0.14366751029759436, 0.18763816913493608,
                 -0.02688357178476987},
        SiCiCase{"z8p050_0", 8.05, 0.0, 1.5803258679842447, 0.0, 0.12137380035053173, 0.0},
        // Compensated-series region (8 < |z| <= 40).
        SiCiCase{"z9p000_3", 9.0, 3.0, 2.6262869999173075, 0.22127925370217663, 0.22500823411237567,
                 -1.052116688319898},
        SiCiCase{"z10p000_10", 10.0, 10.0, 218.45961732963931, -788.0752136719208,
                 -788.0752129049309, -216.8888180446986},
        SiCiCase{"z15p000_5", 15.0, 5.0, 5.860662378226692, 2.0664564842877455, 2.0668245716796303,
                 -4.289672047437387},
        SiCiCase{"z20p000_20", 20.0, 20.0, 2915852.7356184153, 8294658.699394486, 8294658.699394486,
                 -2915851.1648220886},
        SiCiCase{"z30p000_9", 30.0, 9.0, -50.99377250492894, -119.30196238196322,
                 -119.30196622784143, 52.56456947706812},
        SiCiCase{"z39p900_0", 39.9, 0.0, 1.5850396761272794, 0.0, 0.02059344105145697, 0.0},
        // Boundary of the compensated series and the asymptotic expansion.
        SiCiCase{"z40p000_0", 40.0, 0.0, 1.5869851193547846, 0.0, 0.019020007896208765, 0.0},
        SiCiCase{"z40p000_5", 40.0, 5.0, 2.931784930116251, 1.2454112646939235, 1.2455503991191652,
                 -1.3608971668924423},
        // Asymptotic region (|z| > 40), real axis / moderate / 45 degrees.
        SiCiCase{"z30p000_30", 30.0, 30.0, -102089678917.13615, -77298242136.84662,
                 -77298242136.84662, 102089678918.70695},
        SiCiCase{"z45p000_10", 45.0, 10.0, -82.00246754467504, 224.84905589350916,
                 224.84905658120758, 83.5732631737829},
        SiCiCase{"z50p000_5", 50.0, 5.0, 0.11733832015573818, -0.2730162983774192,
                 -0.2730665046271741, 1.4533340387556257},
        SiCiCase{"z70p000_21", 70.0, 21.0, -3590189.7266025124, 8314139.19743701, 8314139.19743701,
                 3590191.2973988396},
        SiCiCase{"z100p000_10", 100.0, 10.0, -97.60357926409506, -46.83756444785367,
                 -46.837564717509956, 99.1743752290921},
        SiCiCase{"z100p000_30", 100.0, 30.0, -49734475675.25443, -12633392981.315174,
                 -12633392981.315174, 49734475676.825226},
        // Near-imaginary axis: where the Heston CV arguments live for tiny K.
        SiCiCase{"z0p000_9", 0.0, 9.0, 0.0, 518.9391515822219, 518.9391391348677,
                 1.5707963267948966},
        SiCiCase{"z0p000_20", 0.0, 20.0, 0.0, 12807826.332028294, 12807826.332028294,
                 1.5707963267948966},
        SiCiCase{"z0p100_30", 0.1, 30.0, 17783032374.03699, 183626422282.88995, 183626422282.88995,
                 -17783032372.466194},
        SiCiCase{"z0p000_100", 0.0, 100.0, 0.0, 1.35777637242694e+41, 1.35777637242694e+41,
                 1.5707963267948966},
        SiCiCase{"z0p000_m40", 0.0, -40.0, 0.0, -3019859131805621.0, 3019859131805621.0,
                 -1.5707963267948966},
        // Negative-real reflection (Re z < 0 is continued from Re z >= 0).
        SiCiCase{"zm20p000_5", -20.0, 5.0, -0.7655824265820264, 3.539353616384862,
                 3.5395978130077275, 2.3365888366065843},
        SiCiCase{"zm50p000_3", -50.0, 3.0, -1.3747945845286775, -0.04469220429770776,
                 -0.04502797706404388, 2.9465245542813263},
        SiCiCase{"zm100p000_m30", -100.0, -30.0, 49734475675.25443, 12633392981.315174,
                 -12633392981.315174, 49734475673.68363},
        SiCiCase{"zm200p000_0", -200.0, 0.0, -1.5683823393394698, 0.0, -0.004378446093027825,
                 3.141592653589793}),
    [](const ::testing::TestParamInfo<SiCiCase>& info) { return info.param.id; });

TEST(SpecialFunctionsSiCi, hestonAsymptoticCvArgument) {
    // The exact argument the default Heston CV (Asymptotic branch) builds at
    // v0 = theta = 0.04, kappa = 2.5, sigma = 4, rho = -0.5, S = 1, K = 1e-8,
    // r = q = 0, T = 1; Si/Ci must be finite and match the reference there.
    const double v0 = 0.04;
    const double kappa = 2.5;
    const double theta = 0.04;
    const double sigma = 4.0;
    const double rho = -0.5;
    const double t = 1.0;
    const double a = v0 + t * kappa * theta;
    const double r1 = std::sqrt(1.0 - rho * rho);
    const std::complex<double> phi(-a / sigma * r1, -a / sigma * rho);
    const double mu = std::log(1.0 / 1e-8) + (0.0 - 0.0) * t;
    const std::complex<double> z = 0.5 * (phi + std::complex<double>(0.0, mu));
    ASSERT_TRUE(std::abs(z) > 8.0);
    const std::complex<double> si = sinIntegral(z);
    const std::complex<double> ci = cosIntegral(z);
    EXPECT_TRUE(std::isfinite(si.real()) && std::isfinite(si.imag()));
    EXPECT_TRUE(std::isfinite(ci.real()) && std::isfinite(ci.imag()));
    // 260-digit reference (see the file comment).
    EXPECT_LT(relativeDistance(si, -8.291579612058937, 627.7812662566137), 1e-12);
    EXPECT_LT(relativeDistance(ci, 627.781256477371, 9.862376101805914), 1e-12);
}

TEST(SpecialFunctionsSiCi, asymptoticMatchesRealBranch) {
    // Independent cross-check of the asymptotic expansion: on the positive
    // real axis the complex entry point must reproduce the rational
    // (GALSIM) real branch, which uses a completely different algorithm.
    for (const double x : {8.5, 9.0, 20.0, 30.0, 39.5, 40.5, 41.0, 50.0, 100.0}) {
        SCOPED_TRACE(::testing::Message() << "x=" << x);
        const std::complex<double> si = sinIntegral(std::complex<double>(x, 0.0));
        const std::complex<double> ci = cosIntegral(std::complex<double>(x, 0.0));
        CHECK_CLOSE_REL("complex Si == real Si", si.real(), quantape::math::sinIntegral(x), 1e-13);
        CHECK_CLOSE("complex Si imaginary is zero", si.imag(), 0.0, 1e-300);
        CHECK_CLOSE_REL("complex Ci == real Ci", ci.real(), quantape::math::cosIntegral(x), 1e-13);
        CHECK_CLOSE("complex Ci imaginary is zero", ci.imag(), 0.0, 1e-300);
    }
}

TEST(SpecialFunctionsSiCi, reflectionAndBranchCut) {
    // Si is odd and entire; Ci(-z) = Ci(z) - i pi for Im z > 0 and
    // Ci(-z) = Ci(z) + i pi for Im z < 0 (the z^2 power series is entire,
    // only log z changes leaf).
    for (const std::complex<double> z :
         {std::complex<double>(9.0, 3.0), std::complex<double>(30.0, 10.0),
          std::complex<double>(30.0, -10.0), std::complex<double>(-50.0, 10.0),
          std::complex<double>(0.0, 100.0), std::complex<double>(0.0, -100.0)}) {
        SCOPED_TRACE(::testing::Message() << "z=" << z);
        const std::complex<double> siMinus = sinIntegral(-z);
        const std::complex<double> si = sinIntegral(z);
        const std::complex<double> ciMinus = cosIntegral(-z);
        const std::complex<double> ci = cosIntegral(z);
        const double scale = std::max(1.0, std::abs(z));
        CHECK_CLOSE("Si odd", (siMinus + si).real(), 0.0, 1e-12 * scale);
        CHECK_CLOSE("Si odd (imag)", (siMinus + si).imag(), 0.0, 1e-12 * scale);
        const std::complex<double> shift = ciMinus - ci;
        CHECK_CLOSE("Ci leaf real", shift.real(), 0.0, 1e-12 * scale);
        const double expectedLeaf =
            std::signbit(z.imag()) ? quantape::util::kPi.hi : -quantape::util::kPi.hi;
        CHECK_CLOSE("Ci leaf imaginary", shift.imag(), expectedLeaf, 1e-12 * scale);
    }
}

TEST(SpecialFunctionsSiCi, continuousAcrossBranchSwitches) {
    // The three complex branches must meet: sample just below/above |z| = 8
    // and |z| = 40 along two directions.
    for (const double angle : {0.0, 0.7853981633974483}) {
        const std::complex<double> dir(std::cos(angle), std::sin(angle));
        for (const double radius : {8.0, 40.0}) {
            SCOPED_TRACE(::testing::Message() << "radius=" << radius << " angle=" << angle);
            const std::complex<double> lo = dir * (radius * (1.0 - 1e-13));
            const std::complex<double> hi = dir * (radius * (1.0 + 1e-13));
            const std::complex<double> dsi = sinIntegral(hi) - sinIntegral(lo);
            const std::complex<double> dci = cosIntegral(hi) - cosIntegral(lo);
            const double siScale = std::max(1.0, std::abs(sinIntegral(hi)));
            const double ciScale = std::max(1.0, std::abs(cosIntegral(hi)));
            EXPECT_LT(std::abs(dsi) / siScale, 1e-10) << "Si";
            EXPECT_LT(std::abs(dci) / ciScale, 1e-10) << "Ci";
        }
    }
}

TEST(SpecialFunctionsSiCi, smallArgumentsMatchSeries) {
    // The |z| <= 8 plain-double path matches the defining series (it serves
    // the Heston CV for moderate strikes).
    CHECK_CLOSE("Si(1)", sinIntegral(std::complex<double>(1.0, 0.0)).real(), 0.94608307036718301494,
                1e-13);
    CHECK_CLOSE("Ci(1)", cosIntegral(std::complex<double>(1.0, 0.0)).real(), 0.33740392290096813466,
                1e-13);
    CHECK_CLOSE("Si(0)", std::abs(sinIntegral(std::complex<double>(0.0, 0.0))), 0.0, 1e-300);
    const std::complex<double> tiny = sinIntegral(std::complex<double>(1e-4, 0.0));
    CHECK_CLOSE_REL("Si(z) ~ z for tiny z", tiny.real(), 1e-4, 1e-8);
}

TEST(SpecialFunctionsSiCi, imaginaryAxisUsesHyperbolicSeries) {
    // On the Stokes line z = i b the asymptotic expansion's remainder is
    // O(1) absolute (e^{-|b|} times |cos i b| ~ e^{|b|}), which limits its
    // component-wise accuracy. Both pure-imaginary branches use the
    // non-alternating hyperbolic series instead:
    //   Si(i b) = i Shi(b),  Ci(i b) = Chi(|b|) + i (pi/2) sign(b),
    // which keep both components to double precision.
    struct ImaginaryCase {
        double b;
        double shi;
    };
    for (const ImaginaryCase c :
         {ImaginaryCase{41.0, 8003324571622521.0}, ImaginaryCase{60.0, 9.680911069646383e23},
          ImaginaryCase{100.0, 1.35777637242694e41}}) {
        SCOPED_TRACE(::testing::Message() << "b=" << c.b);
        const std::complex<double> si = sinIntegral(std::complex<double>(0.0, c.b));
        const std::complex<double> ci = cosIntegral(std::complex<double>(0.0, c.b));
        CHECK_CLOSE("Si(i b) real part is zero", si.real(), 0.0, 1e-300);
        CHECK_CLOSE_REL("Si(i b) = i Shi(b)", si.imag(), c.shi, 1e-14);
        CHECK_CLOSE_REL("Ci(i b) real = Chi(b)", ci.real(), c.shi, 1e-14);
        CHECK_CLOSE("Ci(i b) imaginary = pi/2", ci.imag(), quantape::util::kHalfPi.hi, 1e-15);

        const std::complex<double> siNeg = sinIntegral(std::complex<double>(0.0, -c.b));
        const std::complex<double> ciNeg = cosIntegral(std::complex<double>(0.0, -c.b));
        CHECK_CLOSE_REL("Si(-i b) = -Si(i b)", siNeg.imag(), -c.shi, 1e-14);
        CHECK_CLOSE("Si(-i b) real", siNeg.real(), 0.0, 1e-300);
        CHECK_CLOSE_REL("Ci(-i b) real = Chi(b)", ciNeg.real(), c.shi, 1e-14);
        CHECK_CLOSE("Ci(-i b) imaginary = -pi/2", ciNeg.imag(), -quantape::util::kHalfPi.hi, 1e-15);
    }
}
