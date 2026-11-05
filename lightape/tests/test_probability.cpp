#include <cmath>

#include "lightape/lightape.h"
#include <gtest/gtest.h>

namespace {

using lightape::RevScalar;
using lightape::RevTape;
using lightape::Tangent;

using D1 = Tangent<double, 1>;
using D2 = Tangent<D1, 1>;
using D3 = Tangent<D2, 1>;

D1 Seed1(double v) {
    return D1(v, {1.0});
}

D2 Seed2(double v) {
    D2 x(D1(v, {1.0}));
    x.d[0] = D1(1.0);
    return x;
}

D3 Seed3(double v) {
    D2 inner(D1(v, {1.0}));
    inner.d[0] = D1(1.0);
    D3 x(inner);
    x.d[0] = D2(D1(1.0));
    return x;
}

class ProbabilityTest : public ::testing::Test {
protected:
    void SetUp() override { RevTape::active().clear(); }
    void TearDown() override { RevTape::active().clear(); }
};

TEST_F(ProbabilityTest, DoubleReferenceValues) {
    EXPECT_DOUBLE_EQ(lightape::normpdf(0.0), 0.39894228040143267794);
    EXPECT_DOUBLE_EQ(lightape::normpdf(1.0), 0.24197072451914334980);
    EXPECT_DOUBLE_EQ(lightape::normpdf(-1.0), 0.24197072451914334980);
    EXPECT_NEAR(lightape::normcdf(0.0), 0.5, 1e-16);
    EXPECT_NEAR(lightape::normcdf(1.0), 0.84134474606854294858, 1e-15);
    EXPECT_NEAR(lightape::normcdf(-1.0), 0.15865525393145705142, 1e-15);
    EXPECT_NEAR(lightape::normcdf(-8.0), 6.2209605742717858e-16, 1e-29);
}

TEST_F(ProbabilityTest, ReverseGradients) {
    RevTape& tape = RevTape::active();
    RevScalar x(1.3, tape.input(1.3));
    RevScalar phi = lightape::normpdf(x);
    RevScalar Phi = lightape::normcdf(x);
    tape.reverse(Phi.node);
    EXPECT_NEAR(tape.adjoint(x.node), lightape::normpdf(1.3), 1e-15);
    tape.resetAdjoints();
    tape.reverse(phi.node);
    EXPECT_NEAR(tape.adjoint(x.node), -1.3 * lightape::normpdf(1.3), 1e-15);
}

TEST_F(ProbabilityTest, ForwardDerivativesThroughThirdOrder) {
    const double x = 0.4;
    const double phi = lightape::normpdf(x);
    const double phi1 = -x * phi;
    const double phi2 = (x * x - 1.0) * phi;
    const double phi3 = (3.0 * x - x * x * x) * phi;

    const D1 p1 = lightape::normpdf(Seed1(x));
    EXPECT_NEAR(p1.value, phi, 1e-15);
    EXPECT_NEAR(p1.d[0], phi1, 1e-15);

    const D2 p2 = lightape::normpdf(Seed2(x));
    EXPECT_NEAR(p2.value.value, phi, 1e-15);
    EXPECT_NEAR(p2.d[0].value, phi1, 1e-15);
    EXPECT_NEAR(p2.d[0].d[0], phi2, 1e-14);

    const D3 p3 = lightape::normpdf(Seed3(x));
    EXPECT_NEAR(p3.value.value.value, phi, 1e-15);
    EXPECT_NEAR(p3.d[0].value.value, phi1, 1e-15);
    EXPECT_NEAR(p3.d[0].d[0].value, phi2, 1e-14);
    EXPECT_NEAR(p3.d[0].d[0].d[0], phi3, 1e-13);

    const double cdf1 = phi;
    const double cdf2 = phi1;
    const D2 c2 = lightape::normcdf(Seed2(x));
    EXPECT_NEAR(c2.value.value, lightape::normcdf(x), 1e-15);
    EXPECT_NEAR(c2.d[0].value, cdf1, 1e-15);
    EXPECT_NEAR(c2.d[0].d[0], cdf2, 1e-14);
}

TEST_F(ProbabilityTest, MixedScalarCallsResolve) {
    EXPECT_DOUBLE_EQ(lightape::normpdf(RevScalar(0.0)).value, lightape::normpdf(0.0));
    EXPECT_DOUBLE_EQ(lightape::normcdf(RevScalar(0.0)).value, 0.5);
}

} // namespace
