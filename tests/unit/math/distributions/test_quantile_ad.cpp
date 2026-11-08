// test_quantile_ad.cpp — scalar-generic quantiles on the Stan tape: the generic
// (non-double) tier must instantiate for `stan::math::var` and expose
// `dQ/dp = 1/pdf(Q)`.
#include "quantape/math/StanMath.h" // must be first (Eigen plugin ordering)

#include "quantape/math/Distributions/Beta.h"
#include "quantape/math/Distributions/ChiSquare.h"
#include "quantape/math/Distributions/Gamma.h"
#include "quantape/math/Distributions/Lognormal.h"
#include "quantape/math/Distributions/StudentT.h"
#include "quantape/math/Distributions/detail/IncompleteGamma.h"

#include <cmath>

#include "support/GtestSupport.h"
#include "support/StanTapeFixture.h"

using stan::math::var;

namespace {

// The generic tier refines the double seed with Newton steps, so the two values
// agree to the double solver's converged tolerance.
constexpr double kValueTol = 1e-11;
constexpr double kAdjointTol = 1e-11;

} // namespace

class QuantileAdTest : public StanTapeTest {};

TEST_F(QuantileAdTest, gammaInvCarriesInverseDensityAdjoint) {
    const quantape::math::Gamma<double> d{2.5, 1.3};
    for (double pv : {1e-6, 1e-3, 0.1, 0.3, 0.5, 0.9, 0.999}) {
        SCOPED_TRACE(::testing::Message() << "p=" << pv);
        var p = pv;
        var q = quantape::math::Gamma<var>{2.5, 1.3}.inv(p);
        q.grad();
        const double expected = d.inv(pv);
        CHECK_CLOSE_REL("value matches the double tier", q.val(), expected, kValueTol);
        CHECK_CLOSE_REL("dQ/dp = 1/pdf(Q)", p.adj(), 1.0 / d.pdf(expected), kAdjointTol);
    }
}

// Parameter adjoints ride the same Newton step: at the solution
// `dQ/dtheta = -F_theta / pdf(Q)`.
TEST_F(QuantileAdTest, gammaInvCarriesParameterAdjoints) {
    var shape = 2.5;
    var rate = 1.3;
    var p = 0.3;
    var q = quantape::math::Gamma<var>{shape, rate}.inv(p);
    q.grad();
    const quantape::math::Gamma<double> d{2.5, 1.3};
    const double x = d.inv(0.3);
    const auto pack = quantape::math::detail::gammaCdfGradHess(2.5, 1.3, x);
    CHECK_CLOSE_REL("value", q.val(), x, kValueTol);
    CHECK_CLOSE_REL("dQ/dshape", shape.adj(), -pack.gShape / pack.gx, kAdjointTol);
    CHECK_CLOSE_REL("dQ/drate", rate.adj(), -pack.gRate / pack.gx, kAdjointTol);
}

TEST_F(QuantileAdTest, chiSquareInvCarriesInverseDensityAdjoint) {
    const quantape::math::ChiSquare<double> d{5.0};
    for (double pv : {1e-6, 0.05, 0.3, 0.5, 0.8, 0.999}) {
        SCOPED_TRACE(::testing::Message() << "p=" << pv);
        var p = pv;
        var q = quantape::math::ChiSquare<var>{5.0}.inv(p);
        q.grad();
        const double expected = d.inv(pv);
        CHECK_CLOSE_REL("value matches the double tier", q.val(), expected, kValueTol);
        CHECK_CLOSE_REL("dQ/dp = 1/pdf(Q)", p.adj(), 1.0 / d.pdf(expected), kAdjointTol);
    }
}

TEST_F(QuantileAdTest, betaInvCarriesInverseDensityAdjoint) {
    const quantape::math::Beta<double> d{2.0, 3.5};
    // p > 1/2 exercises the mirrored small-tail solve.
    for (double pv : {1e-6, 0.05, 0.3, 0.5, 0.8, 0.98, 0.999}) {
        SCOPED_TRACE(::testing::Message() << "p=" << pv);
        var p = pv;
        var q = quantape::math::Beta<var>{2.0, 3.5}.inv(p);
        q.grad();
        const double expected = d.inv(pv);
        CHECK_CLOSE_REL("value matches the double tier", q.val(), expected, kValueTol);
        CHECK_CLOSE_REL("dQ/dp = 1/pdf(Q)", p.adj(), 1.0 / d.pdf(expected), kAdjointTol);
    }
}

TEST_F(QuantileAdTest, studentTInvCarriesInverseDensityAdjoint) {
    const quantape::math::StudentT<double> d{5.0};
    // p = 1/2 is the x = 0 point (the value is 0, the adjoint is 1/pdf(0)).
    for (double pv : {1e-6, 1e-3, 0.3, 0.5, 0.7, 0.9, 1.0 - 1e-6}) {
        SCOPED_TRACE(::testing::Message() << "p=" << pv);
        var p = pv;
        var q = quantape::math::StudentT<var>{5.0}.inv(p);
        q.grad();
        const double expected = d.inv(pv);
        CHECK_CLOSE_REL("value matches the double tier", q.val(), expected, kValueTol, 1e-15);
        CHECK_CLOSE_REL("dQ/dp = 1/pdf(Q)", p.adj(), 1.0 / d.pdf(expected), kAdjointTol);
    }
}

TEST_F(QuantileAdTest, lognormalInvCarriesInverseDensityAdjoint) {
    const quantape::math::Lognormal<double> d{0.2, 0.7};
    for (double pv : {1e-6, 0.1, 0.3, 0.5, 0.9, 1.0 - 1e-6}) {
        SCOPED_TRACE(::testing::Message() << "p=" << pv);
        var p = pv;
        var q = quantape::math::Lognormal<var>{0.2, 0.7}.inv(p);
        q.grad();
        const double expected = d.inv(pv);
        CHECK_CLOSE_REL("value matches the double tier", q.val(), expected, kValueTol);
        CHECK_CLOSE_REL("dQ/dp = 1/pdf(Q)", p.adj(), 1.0 / d.pdf(expected), kAdjointTol);
    }
}
