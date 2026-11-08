// test_integrators.cpp — quadrature/integrator gates for the Integrals headers
//
//   * GaussLaguerre (GaussLaguerre.h): the n-point rule integrates
//     x^k e^{-x} exactly for k <= 2n-1, which requires every weight —
//     including the extreme nodes (w ~ 1e-101 at n=64) — to stay relatively
//     accurate, and every node to be a root polished to double-double
//     accuracy.
//   * GaussLobattoIntegrator (GaussLobattoIntegrator.h): absolute-only mode
//     (relAccuracy == 0) handles a zero integral with the absolute tolerance,
//     relative accuracy with a zero scale is rejected, and reversed bounds
//     return the negated integral, consistent with Trapezoid/Simpson.
//   * TanhSinhIntegrator (TanhSinhIntegrator.h): absoluteError() and
//     integrationSuccess() form an honest bound — the conservative level
//     difference plus the unresolved endpoint sliver — so a run that cannot
//     meet the tolerance fails the gate instead of certifying a wrong value,
//     and an exhausted evaluation budget yields a failure status rather than
//     an exception. Smoothly decaying integrands keep converging in a
//     handful of levels.
#include "quantape/math/Integrals/GaussLaguerre.h"
#include "quantape/math/Integrals/GaussLobattoIntegrator.h"
#include "quantape/math/Integrals/TanhSinhIntegrator.h"
#include "quantape/util/Constants.h"

#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <string>
#include <vector>

#include "support/GtestSupport.h"

using namespace quantape::math;
using quantape::util::kPi;

namespace {

constexpr double kGlMomentTol = 1e-12;
constexpr std::size_t kMaxEvals = 1'000'000;

/// Worst |sum_i w_i x_i^k / k! - 1| over the 2n exactness exponents.
///
/// The division by k! is folded into a per-node recursion
/// t_i <- t_i * x_i / (k+1) so no term overflows even at n=128, k=255
/// (255! is far beyond the double range); every term stays O(w_i) bounded.
struct MomentCheck {
    double worst = 0.0;
    std::size_t worstK = 0;
};

MomentCheck normalizedMomentError(const GaussLaguerre& rule) {
    const std::vector<double>& x = rule.nodes();
    const std::vector<double>& w = rule.weights();
    std::vector<double> term = w; // t_i^(0) = w_i
    MomentCheck check;
    for (std::size_t k = 0; k < 2 * rule.order(); ++k) {
        double sum = 0.0;
        for (std::size_t i = 0; i < term.size(); ++i) {
            sum += term[i];
        }
        const double err = std::fabs(sum - 1.0);
        if (err > check.worst) {
            check.worst = err;
            check.worstK = k;
        }
        for (std::size_t i = 0; i < term.size(); ++i) {
            term[i] *= x[i] / static_cast<double>(k + 1);
        }
    }
    return check;
}

// Localized interior peak used by the tanh-sinh under-resolution gates.
double interiorPeak(double x) {
    const double d = x - 0.2;
    return 1.0 / (1.0 + 1e10 * d * d);
}
const double kPeakIntegral = 1e-5 * kPi; // atan(1e5*0.8) - atan(-1e5*0.2)

double endpointSingular(double x) {
    return std::pow(x, -0.75);
}
constexpr double kEndpointSingularIntegral = 4.0;

} // namespace

// ── Gauss-Laguerre: high-order weights and moment exactness ─────────────────

TEST(MathIntegrators, gaussLaguerreHighOrderMomentsStayExact) {
    // An n-point Gauss-Laguerre rule integrates x^k e^{-x} exactly for
    // k <= 2n-1, so the normalized moment sum_i w_i x_i^k / k! is 1 to
    // double precision for every order and every such exponent.
    for (std::size_t n : {8, 16, 20, 32, 64, 128}) {
        const GaussLaguerre rule(n);
        const MomentCheck check = normalizedMomentError(rule);
        const std::string label = "n=" + std::to_string(n) + " worst normalized moment error (k=" +
                                  std::to_string(check.worstK) + ")";
        CHECK_CLOSE(label, check.worst, 0.0, kGlMomentTol);
    }
}

TEST(MathIntegrators, gaussLaguerreExtremeWeightMatchesReference) {
    // Reference (100-digit Decimal evaluation of w = 1/(x [L'_n(x)]^2)):
    // largest node and its weight for n = 64. Both must match the reference
    // and every weight must be positive and sum to 1.
    constexpr double kRefNode = 234.80957917132616;
    constexpr double kRefWeight = 2.0890635084369527e-101;
    const GaussLaguerre rule(64);
    CHECK_CLOSE("n=64 largest node", rule.nodes().back(), kRefNode, 1e-12);
    CHECK_CLOSE_REL("n=64 weight at the largest node", rule.weights().back(), kRefWeight, 1e-9);
    // The weights must all be positive and the rule must reproduce
    // int_0^inf e^{-x} dx = 1.
    double sum = 0.0;
    for (double w : rule.weights()) {
        EXPECT_GT(w, 0.0) << "Gauss-Laguerre weight must be positive";
        sum += w;
    }
    CHECK_CLOSE("n=64 weight sum", sum, 1.0, 1e-13);
}

// ── Gauss-Lobatto: zero integrals, zero-width intervals, reversed bounds ────

TEST(MathIntegrators, gaussLobattoZeroIntegralAbsoluteOnly) {
    // Absolute-only mode (relAccuracy == 0): the 13-point estimate is exactly
    // zero here, and the absolute tolerance remains well defined, so the
    // integral is 0 and the zero-width interval is 0.
    GaussLobattoIntegrator<double> integ(1e-9, 1000);
    double oddCubic = 1.0;
    double oddLinear = 1.0;
    double zero = 1.0;
    ASSERT_NO_THROW(oddCubic = integ([](double x) { return x * x * x; }, -1.0, 1.0));
    ASSERT_NO_THROW(oddLinear = integ([](double x) { return x; }, -1.0, 1.0));
    ASSERT_NO_THROW(zero = integ([](double) { return 0.0; }, -1.0, 1.0));
    CHECK_CLOSE("int_-1^1 x^3", oddCubic, 0.0, 1e-14);
    CHECK_CLOSE("int_-1^1 x", oddLinear, 0.0, 1e-14);
    CHECK_CLOSE("int_-1^1 0", zero, 0.0, 1e-14);

    // Zero-width interval: the integral is zero, not a machine-number error.
    double empty = 1.0;
    ASSERT_NO_THROW(empty = integ([](double x) { return x * x; }, 3.0, 3.0));
    CHECK_CLOSE("zero-width interval", empty, 0.0, 1e-14);

    // Relative accuracy with a zero scale has no meaning and is rejected.
    GaussLobattoIntegrator<double> relative(1e-9, 1000, 1e-9);
    EXPECT_THROW(relative([](double x) { return x * x * x; }, -1.0, 1.0), std::runtime_error);
}

TEST(MathIntegrators, gaussLobattoReversedBoundsNegate) {
    // Reversed bounds return the negated integral, consistent with
    // Trapezoid/Simpson (and with tanh-sinh).
    GaussLobattoIntegrator<double> integ(1e-9, 1000);
    double reversedSquare = 0.0;
    double reversedLinear = 0.0;
    double reversedCubic = 0.0;
    ASSERT_NO_THROW(reversedSquare = integ([](double x) { return x * x; }, 1.0, 0.0));
    ASSERT_NO_THROW(reversedLinear = integ([](double x) { return x; }, 1.0, 0.0));
    ASSERT_NO_THROW(reversedCubic = integ([](double x) { return x * x * x; }, 1.0, -1.0));
    CHECK_CLOSE("int_1^0 x^2", reversedSquare, -1.0 / 3.0, 1e-13);
    CHECK_CLOSE("int_1^0 x", reversedLinear, -0.5, 1e-13);
    CHECK_CLOSE("int_1^-1 x^3", reversedCubic, 0.0, 1e-13);
}

// ── Tanh-sinh: honest error bound and failure status ───────────────────────

TEST(MathIntegrators, tanhSinhEndpointSingularityReportsFailure) {
    // x^-3/4 is integrable, but the plain path's COMP_MIN cut leaves an
    // unresolved endpoint mass (~6e-4 here). Refinement cannot remove it, so
    // absoluteError() covers that mass and integrationSuccess() fails the
    // 1e-6 gate.
    TanhSinhIntegrator<double> integ(1e-6, kMaxEvals);
    const double value = integ(endpointSingular, 0.0, 1.0);
    const double actualError = std::fabs(value - kEndpointSingularIntegral);
    CHECK_CLOSE("int_0^1 x^-3/4", value, kEndpointSingularIntegral, 1e-3);
    EXPECT_FALSE(integ.integrationSuccess());
    EXPECT_GE(integ.absoluteError(), actualError);
}

TEST(MathIntegrators, tanhSinhUnresolvedInteriorPeakReportsFailure) {
    // The peak is narrower than the node spacing; the level differences jump
    // and are not a convergent sequence, so they are not an error bound.
    // absoluteError() must cover the actual error and integrationSuccess()
    // must fail.
    TanhSinhIntegrator<double> integ(1e-6, kMaxEvals);
    const double value = integ(interiorPeak, 0.0, 1.0);
    const double actualError = std::fabs(value - kPeakIntegral);
    CHECK_CLOSE("int_0^1 peak", value, kPeakIntegral, 1e-5);
    EXPECT_FALSE(integ.integrationSuccess());
    EXPECT_GE(integ.absoluteError(), actualError);
}

TEST(MathIntegrators, tanhSinhHardIntegralFailsWithoutThrowing) {
    // A starved evaluation budget on a legitimate integrand yields a failure
    // status without raising.
    TanhSinhIntegrator<double> integ(1e-12, 64);
    double value = -1.0;
    ASSERT_NO_THROW(value = integ(interiorPeak, 0.0, 1.0));
    (void)value;
    EXPECT_FALSE(integ.integrationSuccess());
}

TEST(MathIntegrators, tanhSinhSmoothDecayStaysFast) {
    // The exponentially decaying use case converges in a handful of levels
    // (~100 evaluations): the conservative estimator does not add refinement.
    TanhSinhIntegrator<double> integ(1e-10, kMaxEvals);
    const double value = integ([](double x) { return std::exp(-x); }, 0.0, 1.0);
    CHECK_CLOSE("int_0^1 exp(-x)", value, 1.0 - std::exp(-1.0), 1e-13);
    EXPECT_TRUE(integ.integrationSuccess());
    EXPECT_LT(integ.numberOfEvaluations(), 1000U);
}
