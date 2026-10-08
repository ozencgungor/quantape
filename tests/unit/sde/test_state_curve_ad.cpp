// test_state_curve_ad.cpp — AD value/gradient gates for the loading evaluation
#include "quantape/math/StanMath.h" // must be first (Eigen plugin ordering)

#include "quantape/markets/Curves/DiscountCurve.h"
#include "quantape/sde/StateCurveMap.h"

#include <array>
#include <cmath>
#include <cstddef>
#include <span>
#include <vector>

#include "support/GtestSupport.h"
#include "support/StanTapeFixture.h"

using namespace quantape;
using stan::math::fvar;
using stan::math::var;

namespace {

using markets::DiscountCurve;

constexpr std::size_t kNObs = 4;
constexpr std::size_t kNPay = 8;

DiscountCurve<double> makeOisCurve() {
    return DiscountCurve<double>({0.0, 1.0, 2.0, 3.0, 5.0}, {0.0, 0.02, 0.025, 0.03, 0.035},
                                 markets::InterpolationSpace::LogDiscount,
                                 markets::InterpolationScheme::Linear);
}

std::vector<double> obsTimes() {
    return {0.0, 0.5, 1.0, 2.0};
}
std::vector<double> payTimes() {
    return {0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0};
}

sde::GaussianStateCurve::Loadings makeLoadings() {
    sde::GaussianStateCurve::Loadings loadings;
    loadings.factor.assign(2, std::vector<double>(kNObs * kNPay, 0.0));
    loadings.correlation = {1.0, 0.35, 0.35, 1.0};
    const std::vector<double> obs = obsTimes();
    const std::vector<double> pay = payTimes();
    for (std::size_t k = 0; k < kNObs; ++k) {
        for (std::size_t j = 0; j < kNPay; ++j) {
            const double tau = pay[j] - obs[k];
            loadings.factor[0][k * kNPay + j] = 0.04 * tau;
            loadings.factor[1][k * kNPay + j] = 0.015 * tau * tau;
        }
    }
    return loadings;
}

sde::GaussianStateCurve makeMap() {
    return sde::GaussianStateCurve::fromCurve(makeOisCurve(), obsTimes(), payTimes(),
                                              makeLoadings());
}

constexpr double kStep = 1e-6;

} // namespace

class StateCurveAdTest : public StanTapeTest {};

TEST_F(StateCurveAdTest, varGradientMatchesCentralDifference) {
    const sde::GaussianStateCurve map = makeMap();
    const double x0 = 0.12;
    const double x1 = -0.07;
    std::array<var, 2> state{var(x0), var(x1)};

    var y = map.bond(0, 5, std::span<const var>(state));
    y.grad();
    const double ad0 = state[0].adj();
    const double ad1 = state[1].adj();

    const auto eval = [&](double a, double b) {
        const std::array<double, 2> s{a, b};
        return map.bond(0, 5, std::span<const double>(s));
    };
    const double fd0 = (eval(x0 + kStep, x1) - eval(x0 - kStep, x1)) / (2.0 * kStep);
    const double fd1 = (eval(x0, x1 + kStep) - eval(x0, x1 - kStep)) / (2.0 * kStep);
    CHECK_CLOSE("state adjoint x0", ad0, fd0, 1e-8);
    CHECK_CLOSE("state adjoint x1", ad1, fd1, 1e-8);
}

TEST_F(StateCurveAdTest, varGradientOfLoadingEvaluationMatchesCentralDifference) {
    const double baseRatio = 0.97;
    const double variance = 0.013;
    const std::array<double, 2> state{0.55, 0.32};
    std::array<var, 2> loading{var(0.21), var(-0.08)};

    var y = sde::GaussianStateCurve::analyticBond(baseRatio, std::span<const var>(loading),
                                                  std::span<const double>(state), variance);
    y.grad();
    const double ad0 = loading[0].adj();
    const double ad1 = loading[1].adj();

    const auto eval = [&](double l0, double l1) {
        const std::array<double, 2> l{l0, l1};
        return sde::GaussianStateCurve::analyticBond(baseRatio, std::span<const double>(l),
                                                     std::span<const double>(state), variance);
    };
    const double fd0 = (eval(0.21 + kStep, -0.08) - eval(0.21 - kStep, -0.08)) / (2.0 * kStep);
    const double fd1 = (eval(0.21, -0.08 + kStep) - eval(0.21, -0.08 - kStep)) / (2.0 * kStep);
    CHECK_CLOSE("loading adjoint 0", ad0, fd0, 1e-8);
    CHECK_CLOSE("loading adjoint 1", ad1, fd1, 1e-8);
    CHECK_CLOSE_REL("loading value", y.val(), eval(0.21, -0.08), 1e-15);
}

TEST_F(StateCurveAdTest, fvarVarDirectionalDerivativeMatchesCentralDifference) {
    const sde::GaussianStateCurve map = makeMap();
    const double x0 = 0.12;
    const double x1 = -0.07;
    const double v0 = 0.7;
    const double v1 = -0.4;
    std::array<fvar<var>, 2> state{fvar<var>(var(x0), v0), fvar<var>(var(x1), v1)};

    const fvar<var> y = map.bond(0, 5, std::span<const fvar<var>>(state));
    const double value = y.val_.val();
    const double directional = y.d_.val();

    const auto eval = [&](double a, double b) {
        const std::array<double, 2> s{a, b};
        return map.bond(0, 5, std::span<const double>(s));
    };
    CHECK_CLOSE("fvar value", value, eval(x0, x1), 1e-15);
    const double fd =
        (eval(x0 + kStep * v0, x1 + kStep * v1) - eval(x0 - kStep * v0, x1 - kStep * v1)) /
        (2.0 * kStep);
    CHECK_CLOSE("fvar directional tangent", directional, fd, 1e-8);
}

TEST_F(StateCurveAdTest, varAnnuityGradientMatchesCentralDifference) {
    const sde::GaussianStateCurve map = makeMap();
    const std::vector<double> accruals(map.payCount(), 0.25);
    const double x0 = 0.09;
    const double x1 = -0.04;
    std::array<var, 2> state{var(x0), var(x1)};

    var y = map.annuity(1, 0, std::span<const var>(state), std::span<const double>(accruals));
    y.grad();
    const double ad0 = state[0].adj();
    const double ad1 = state[1].adj();

    const auto eval = [&](double a, double b) {
        const std::array<double, 2> s{a, b};
        return map.annuity(1, 0, std::span<const double>(s), std::span<const double>(accruals));
    };
    const double fd0 = (eval(x0 + kStep, x1) - eval(x0 - kStep, x1)) / (2.0 * kStep);
    const double fd1 = (eval(x0, x1 + kStep) - eval(x0, x1 - kStep)) / (2.0 * kStep);
    CHECK_CLOSE("annuity adjoint x0", ad0, fd0, 1e-8);
    CHECK_CLOSE("annuity adjoint x1", ad1, fd1, 1e-8);
}
