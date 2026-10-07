// test_bessel.cpp — modified Bessel K_nu: value/log/derivative gates, known
// mpmath values, Wronskian and recurrence identities, finite-difference
// derivative oracle, regime boundaries, under/overflow and edge behavior.
#include "quantape/math/Distributions/detail/BesselI.h"
#include "quantape/math/Distributions/detail/BesselK.h"

#include <cmath>
#include <limits>

#include "support/GtestSupport.h"

namespace {

using quantape::math::detail::besselI;
using quantape::math::detail::besselINuDeriv;
using quantape::math::detail::besselINuDeriv2;
using quantape::math::detail::besselK;
using quantape::math::detail::besselKNuDeriv;
using quantape::math::detail::besselKNuLogDeriv;
using quantape::math::detail::besselKPrime;
using quantape::math::detail::logBesselK;
using quantape::util::kPi;

constexpr double kInf = std::numeric_limits<double>::infinity();
constexpr double kNaN = std::numeric_limits<double>::quiet_NaN();

} // namespace

// mpmath (dps = 30) references for K_0, K_1, K_2, K_{1/2}, K_5 and the
// half-integer closed forms K_{1/2}(z) = sqrt(pi/2z) e^{-z},
// K_{3/2}(z) = K_{1/2}(z) (1 + 1/z), K_{5/2}(z) = K_{1/2}(z)(1 + 3/z + 3/z^2).
TEST(BesselK, knownValues) {
    CHECK_CLOSE_REL("K_0(1)", besselK(0.0, 1.0), 0.42102443824070834, 1e-13);
    CHECK_CLOSE_REL("K_1(1)", besselK(1.0, 1.0), 0.6019072301972346, 1e-13);
    CHECK_CLOSE_REL("K_0(2)", besselK(0.0, 2.0), 0.11389387274953344, 1e-13);
    CHECK_CLOSE_REL("K_0(0.5)", besselK(0.0, 0.5), 0.9244190712276659, 1e-13);
    CHECK_CLOSE_REL("K_1(0.5)", besselK(1.0, 0.5), 1.6564411200033007, 1e-13);
    CHECK_CLOSE_REL("K_2(1)", besselK(2.0, 1.0), 1.6248388986351774, 1e-13);
    CHECK_CLOSE_REL("K_{1/2}(2)", besselK(0.5, 2.0), 0.11993777196806145, 1e-13);
    CHECK_CLOSE_REL("K_5(3)", besselK(5.0, 3.0), 0.9377736023868081, 1e-13);
    CHECK_CLOSE_REL("K_0(0.25)", besselK(0.0, 0.25), 1.5415067512483027, 1e-13);
    CHECK_CLOSE_REL("K_2.5(0.75)", besselK(2.5, 0.75), 7.063970623611755, 1e-13);
    CHECK_CLOSE_REL("K_7.25(1.5)", besselK(7.25, 1.5), 4253.442579936807, 1e-13);
    // The high-order small-z corner exercises Temme-seed + long recurrence.
    CHECK_CLOSE_REL("K_20(0.4)", besselK(20.0, 0.4), 5.78829239785711e30, 1e-12);

    for (double z : {0.25, 1.0, 5.0, 20.0}) {
        const double kHalf = std::sqrt(kPi / (2.0 * z)) * std::exp(-z);
        CHECK_CLOSE_REL("K_{1/2} closed form", besselK(0.5, z), kHalf, 1e-14);
        CHECK_CLOSE_REL("K_{3/2} closed form", besselK(1.5, z), kHalf * (1.0 + 1.0 / z), 1e-14);
        CHECK_CLOSE_REL("K_{5/2} closed form", besselK(2.5, z),
                        kHalf * (1.0 + 3.0 / z + 3.0 / (z * z)), 1e-14);
    }
}

// Regime-boundary values (Temme z <= 1, continued fraction 1 < z < 16,
// asymptotic z >= 16) all come from the same mpmath reference table.
TEST(BesselK, regimeBoundaryValues) {
    CHECK_CLOSE_REL("K_3.25(0.9999999)", besselK(3.25, 0.9999999), 10.894861303731016, 1e-12);
    CHECK_CLOSE_REL("K_3.25(1.0000001)", besselK(3.25, 1.0000001), 10.894853770353068, 1e-12);
    CHECK_CLOSE_REL("K_0(15.999999)", besselK(0.0, 15.999999), 3.4994152710954786e-8, 1e-12);
    CHECK_CLOSE_REL("K_0(16.000001)", besselK(0.0, 16.000001), 3.4994080567812435e-8, 1e-12);
    CHECK_CLOSE_REL("K_10(16)", besselK(10.0, 16.0), 6.719822688821985e-7, 1e-12);
}

// I_nu K_{nu+1} + I_{nu+1} K_nu = 1/z on a grid; besselI is the independent
// implementation from BesselI.h (max observed relative error 1.8e-14).
TEST(BesselK, wronskianIdentity) {
    for (double nu : {0.0, 0.25, 1.0, 2.5, 7.0, 12.3, 20.0}) {
        for (double z : {0.01, 0.1, 0.5, 1.0, 2.5, 5.0, 10.0, 18.0, 40.0}) {
            const double wronskian =
                besselI(nu, z) * besselK(nu + 1.0, z) + besselI(nu + 1.0, z) * besselK(nu, z);
            CHECK_CLOSE_REL("I_nu K_{nu+1} + I_{nu+1} K_nu = 1/z", wronskian * z, 1.0, 1e-12);
        }
    }
}

// K_{nu+1} = K_{nu-1} + (2 nu / z) K_nu. Orders below 1 reflect through the
// even-order symmetry (`besselK` enforces nu >= 0, so the test does too).
TEST(BesselK, orderRecurrence) {
    for (double nu : {0.0, 0.3, 1.0, 2.5, 9.0, 19.5}) {
        for (double z : {0.01, 0.1, 0.75, 1.0, 3.0, 12.0, 30.0}) {
            const double lhs = besselK(nu + 1.0, z);
            const double rhs = besselK(std::fabs(nu - 1.0), z) + (2.0 * nu / z) * besselK(nu, z);
            CHECK_CLOSE_REL("K recurrence", lhs, rhs, 1e-12);
        }
    }
}

// K'_nu = -(K_{nu-1} + K_{nu+1})/2 against a central difference of besselK
// (truncation-limited, so gated at the FD oracle tolerance 1e-6).
TEST(BesselK, derivativeMatchesFiniteDifferences) {
    for (double nu : {0.0, 0.5, 1.0, 2.5, 5.0, 10.0}) {
        for (double z : {0.25, 0.75, 1.0, 2.0, 5.0, 15.0}) {
            const double h = 1e-6 * (1.0 + z);
            const double fd = (besselK(nu, z + h) - besselK(nu, z - h)) / (2.0 * h);
            CHECK_CLOSE_REL("K' vs central FD", besselKPrime(nu, z), fd, 1e-6);
        }
    }
    // Exact limits: K'_0 = -K_1 and K'_{1/2} = -(K_{1/2} + K_{3/2})/2.
    CHECK_CLOSE_REL("K'_0(3) = -K_1(3)", besselKPrime(0.0, 3.0), -besselK(1.0, 3.0), 1e-14);
    const double kHalf = std::sqrt(kPi / 6.0) * std::exp(-3.0);
    CHECK_CLOSE_REL("K'_{1/2}(3)", besselKPrime(0.5, 3.0),
                    -0.5 * (kHalf + kHalf * (1.0 + 1.0 / 3.0)), 1e-14);
}

// logK is assembled in log space; where K is representable it must track
// log(K), and mpmath references pin the deep-tail magnitudes.
TEST(BesselK, logKMatchesLogAndReferences) {
    for (double nu : {0.0, 1.0, 2.5, 7.25, 20.0}) {
        for (double z : {0.5, 1.0, 2.0, 5.0, 16.0, 50.0, 100.0, 200.0, 700.0}) {
            const double k = besselK(nu, z);
            if (k > 0.0 && std::isfinite(k)) {
                // Loose enough for the deep-tail points, where |log K| ~ 700
                // and one double ulp of the log value is ~1.1e-13.
                CHECK_CLOSE("logK vs log(K)", logBesselK(nu, z), std::log(k), 1e-12);
            }
        }
    }
    CHECK_CLOSE("logK_0(100)", logBesselK(0.0, 100.0), -102.07803755445829, 1e-12);
    CHECK_CLOSE("logK_0(700)", logBesselK(0.0, 700.0), -703.0499272589439, 1e-12);
    CHECK_CLOSE("logK_20(0.01)", logBesselK(20.0, 0.01), 144.61308302181087, 1e-12);
    CHECK_CLOSE_REL("exp(logK_20(0.01))", std::exp(logBesselK(20.0, 0.01)), besselK(20.0, 0.01),
                    1e-13);
}

// Underflow/overflow and the log-space fallback: K underflows to 0 beyond
// z ~ 745 and overflows for tiny z at large order, while logK stays finite.
TEST(BesselK, underflowOverflowAndLogSpace) {
    CHECK_CLOSE("K_0(745) underflow", besselK(0.0, 745.0), 0.0, 0.0);
    CHECK_CLOSE("K_0(800) underflow", besselK(0.0, 800.0), 0.0, 0.0);
    CHECK_CLOSE("logK_0(800)", logBesselK(0.0, 800.0), -803.1166706636599, 1e-11);
    CHECK_CLOSE("logK_20(1e-16)", logBesselK(20.0, 1e-16), 789.336910375933, 1e-11);
    CHECK_CLOSE_REL("K_0(700) deep-tail value", besselK(0.0, 700.0), 4.669776431685377e-306, 1e-13);
}

// Domain gates: nu < 0, non-finite nu, z <= 0 and NaN inputs are NaN;
// z = +inf is the exponential limit 0 (log -inf).
TEST(BesselK, domainEdges) {
    EXPECT_TRUE(std::isnan(besselK(0.0, -1.0)));
    EXPECT_TRUE(std::isnan(besselK(0.0, 0.0)));
    EXPECT_TRUE(std::isnan(besselK(-1.0, 1.0)));
    EXPECT_TRUE(std::isnan(besselK(-0.25, 1.0)));
    EXPECT_TRUE(std::isnan(besselK(kNaN, 1.0)));
    EXPECT_TRUE(std::isnan(besselK(0.0, kNaN)));
    EXPECT_TRUE(std::isnan(besselK(kInf, 1.0)));
    EXPECT_TRUE(std::isnan(logBesselK(0.0, -1.0)));
    EXPECT_TRUE(std::isnan(logBesselK(-1.0, 1.0)));
    EXPECT_TRUE(std::isnan(logBesselK(kNaN, 1.0)));
    EXPECT_TRUE(std::isnan(besselKPrime(0.0, -1.0)));
    EXPECT_TRUE(std::isnan(besselKPrime(-1.0, 1.0)));
    CHECK_CLOSE("K(0, +inf)", besselK(0.0, kInf), 0.0, 0.0);
    CHECK_CLOSE("K(nu, +inf)", besselK(5.0, kInf), 0.0, 0.0);
    CHECK_CLOSE("KPrime(0, +inf)", besselKPrime(0.0, kInf), 0.0, 0.0);
    EXPECT_TRUE(std::isinf(logBesselK(0.0, kInf)) && logBesselK(0.0, kInf) < 0.0);
}

// Sanity: K_nu(z) decreases in z for fixed order (positive argument) and
// increases in order for fixed z.
TEST(BesselK, monotonicitySanity) {
    for (double nu : {0.0, 0.5, 3.3}) {
        double previous = besselK(nu, 0.1);
        for (double z = 0.3; z <= 20.0; z += 0.3) {
            const double current = besselK(nu, z);
            EXPECT_TRUE(current < previous) << "nu=" << nu << " z=" << z;
            previous = current;
        }
    }
    for (double z : {0.5, 5.0, 18.0}) {
        double previous = besselK(0.0, z);
        for (double nu = 0.5; nu <= 20.0; nu += 0.5) {
            const double current = besselK(nu, z);
            EXPECT_TRUE(current > previous) << "nu=" << nu << " z=" << z;
            previous = current;
        }
    }
}

// ---------------------------------------------------------------------------
// Order derivatives dI/dnu, d2I/dnu2, dK/dnu and the K log-derivative.
// References: mpmath 1.4.1, dps = 40, `mp.diff` (comment lists the points).
// ---------------------------------------------------------------------------

TEST(BesselOrderDerivatives, IOrderDerivativeMpmathGates) {
    CHECK_CLOSE_REL("dI(0.5,0.5)", besselINuDeriv(0.5, 0.5), -0.85257509960176786, 1e-11);
    CHECK_CLOSE_REL("d2I(0.5,0.5)", besselINuDeriv2(0.5, 0.5), 0.70753265903757494, 1e-11);
    CHECK_CLOSE_REL("dI(1.7,2.3)", besselINuDeriv(1.7, 2.3), -1.0567059883166774, 1e-11);
    CHECK_CLOSE_REL("d2I(1.7,2.3)", besselINuDeriv2(1.7, 2.3), 0.40621571139010426, 1e-11);
    CHECK_CLOSE_REL("dI(5.0,0.5)", besselINuDeriv(5.0, 0.5), -2.5443679964152405e-5, 1e-11);
    CHECK_CLOSE_REL("d2I(5.0,0.5)", besselINuDeriv2(5.0, 0.5), 7.7240111245766605e-5, 1e-11);
    CHECK_CLOSE_REL("dI(2.5,25.0)", besselINuDeriv(2.5, 25.0), -518012862.09790704, 1e-11);
    CHECK_CLOSE_REL("d2I(2.5,25.0)", besselINuDeriv2(2.5, 25.0), -153655538.07002567, 1e-11);
    CHECK_CLOSE_REL("dI(0.25,0.1)", besselINuDeriv(0.25, 0.1), -1.4479384233460546, 1e-11);
    CHECK_CLOSE_REL("d2I(0.25,0.1)", besselINuDeriv2(0.25, 0.1), 3.386049386006769, 1e-11);
}

TEST(BesselOrderDerivatives, KOrderDerivativeMpmathGates) {
    // z = 0.5 sits on the Gauss-Hermite threshold; ~1e-9 observed.
    CHECK_CLOSE_REL("dK(0.5,0.5)", besselKNuDeriv(0.5, 0.5), 0.64110180271904842, 5e-9);
    CHECK_CLOSE_REL("ldK(0.5,0.5)", besselKNuLogDeriv(0.5, 0.5), 0.59634736232319407, 5e-9);
    CHECK_CLOSE_REL("dK(1.0,0.5)", besselKNuDeriv(1.0, 0.5), 1.8488381424553317, 1e-9);
    CHECK_CLOSE_REL("ldK(1.0,0.5)", besselKNuLogDeriv(1.0, 0.5), 1.1161508369531707, 1e-9);
    CHECK_CLOSE_REL("dK(2.5,5.0)", besselKNuDeriv(2.5, 5.0), 0.0028989772717665015, 1e-9);
    CHECK_CLOSE_REL("ldK(2.5,5.0)", besselKNuLogDeriv(2.5, 5.0), 0.44628658933063361, 1e-9);
    CHECK_CLOSE_REL("dK(5.0,10.0)", besselKNuDeriv(5.0, 10.0), 2.661788999450625e-5, 1e-9);
    CHECK_CLOSE_REL("ldK(5.0,10.0)", besselKNuLogDeriv(5.0, 10.0), 0.46258314602850867, 1e-9);
    CHECK_CLOSE_REL("dK(0.5,1.0)", besselKNuDeriv(0.5, 1.0), 0.16659724500287904, 1e-9);
    CHECK_CLOSE_REL("ldK(0.5,1.0)", besselKNuLogDeriv(0.5, 1.0), 0.36132861688822258, 1e-9);
    CHECK_CLOSE_REL("dK(10.0,5.0)", besselKNuDeriv(10.0, 5.0), 13.701150911996571, 1e-9);
    CHECK_CLOSE_REL("ldK(10.0,5.0)", besselKNuLogDeriv(10.0, 5.0), 1.4040131884000932, 1e-9);
    // Small-z reflection path (order away from an integer).
    CHECK_CLOSE_REL("dK(0.5,0.2)", besselKNuDeriv(0.5, 0.2), 2.4042301953445728, 1e-9);
    CHECK_CLOSE_REL("dK(1.5,0.1)", besselKNuDeriv(1.5, 0.1), 119.92191657101574, 1e-9);
    CHECK_CLOSE_REL("dK(2.5,0.3)", besselKNuDeriv(2.5, 0.3), 196.13341573475271, 1e-9);
}

TEST(BesselOrderDerivatives, finiteDifferenceOracle) {
    for (double nu : {0.75, 2.0, 4.5}) {
        for (double z : {0.5, 2.0, 8.0}) {
            const double h = 1e-5;
            const double fdI = (besselI(nu + h, z) - besselI(nu - h, z)) / (2.0 * h);
            CHECK_CLOSE_REL("dI vs FD", besselINuDeriv(nu, z), fdI, 1e-6, 1e-6);
            const double fdK = (besselK(nu + h, z) - besselK(nu - h, z)) / (2.0 * h);
            CHECK_CLOSE_REL("dK vs FD", besselKNuDeriv(nu, z), fdK, 1e-6, 1e-6);
        }
    }
}

TEST(BesselOrderDerivatives, edges) {
    EXPECT_TRUE(std::isnan(besselINuDeriv(-1.5, 1.0)));
    EXPECT_TRUE(std::isnan(besselINuDeriv(1.0, -1.0)));
    EXPECT_TRUE(std::isnan(besselINuDeriv2(1.0, kNaN)));
    EXPECT_TRUE(std::isnan(besselKNuDeriv(-0.5, 1.0)));
    EXPECT_TRUE(std::isnan(besselKNuDeriv(1.0, 0.0)));
    CHECK_CLOSE("dK at z=inf", besselKNuDeriv(2.0, kInf), 0.0, 0.0);
    CHECK_CLOSE("ldK at z=inf", besselKNuLogDeriv(2.0, kInf), 0.0, 0.0);
}
