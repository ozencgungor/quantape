/**
 * @file test_levenberg_marquardt_ad.cpp
 * @brief Levenberg-Marquardt AD fitter and IFT sensitivity gates
 */

#include "quantape/math/StanMath.h"

#include "quantape/calibration/LevenbergMarquardtIft.h"
#include "quantape/math/NumericalMethods.h"
#include "quantape/math/Solvers/SolverStanPrimitives.h"

#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <vector>

#include "support/GtestSupport.h"
#include "support/StanTapeFixture.h"

using namespace quantape;

namespace {

/// Fit statuses that mean "stopped at a usable optimum" (the IFT layer
/// additionally requires `LevenbergMarquardtResult::stationary`).
bool usable(const math::LevenbergMarquardtResult& fit) {
    return fit.status == math::OptimizeResult::GradientTolReached ||
           fit.status == math::OptimizeResult::FtolReached ||
           fit.status == math::OptimizeResult::XtolReached ||
           fit.status == math::OptimizeResult::RoundoffLimited;
}

} // namespace

class LevenbergMarquardtAdTest : public StanTapeTest {};

TEST_F(LevenbergMarquardtAdTest, exactAdJacobian) {
    // Scalar-generic quadratic residual (double and var).
    const auto quadraticResidual = [](const auto& p, auto& out) {
        out.clear();
        for (int i = 0; i < 21; ++i) {
            const double t = 0.5 * i;
            out.push_back(p[0] + p[1] * t + p[2] * t * t - (1.0 - 0.5 * t + 0.25 * t * t));
        }
    };
    std::vector<double> params{0.0, 0.0, 0.0};
    const math::LevenbergMarquardtResult fit =
        math::levenbergMarquardtAd(quadraticResidual, params);
    EXPECT_TRUE(usable(fit));
    EXPECT_TRUE(fit.stationary);
    EXPECT_TRUE(fit.gradient.size() == 3);
    EXPECT_TRUE(fit.iterations >= 1);
    CHECK_CLOSE("LM-AD quadratic a", params[0], 1.0, 1e-8);
    CHECK_CLOSE("LM-AD quadratic b", params[1], -0.5, 1e-8);
    CHECK_CLOSE("LM-AD quadratic c", params[2], 0.25, 1e-8);
}

TEST_F(LevenbergMarquardtAdTest, differentialAndTape) {
    const double trueA = 2.0;
    const double trueB = -0.3;
    std::vector<double> times;
    std::vector<double> quotes;
    for (int i = 1; i <= 10; ++i) {
        const double t = 0.5 * i;
        times.push_back(t);
        quotes.push_back(trueA * std::exp(trueB * t) * (1.0 + 0.01 * i));
    }
    const auto residual = [&](const auto& x, const auto& m, auto& out) {
        using std::exp;
        out.clear();
        for (std::size_t i = 0; i < times.size(); ++i) {
            out.push_back(x[0] * exp(x[1] * times[i]) - m[i]);
        }
    };

    const std::vector<double> x0{1.0, -0.1};
    math::LevenbergMarquardtOptions lmOptions;
    lmOptions.xtol = 0.0;
    lmOptions.ftol = 0.0;
    lmOptions.gradientTol = 1e-16;
    lmOptions.maxIterations = 2000;
    lmOptions.lambda0 = 1e-10;

    std::vector<double> pHat;
    math::IftResult ift;
    std::vector<double> dpDm;
    math::LevenbergMarquardtResult lm;
    const math::OptimizeResult status = math::levenbergMarquardtDifferential(
        residual, x0, quotes, pHat, ift, &dpDm, lmOptions, {}, &lm);
    EXPECT_TRUE(usable(lm));
    EXPECT_TRUE(lm.stationary);
    EXPECT_TRUE(status == lm.status);

    // FD of the refit map validates the IFT Jacobian.
    const double epsilon = 1e-5;
    const auto refit = [&](const std::vector<double>& data) {
        std::vector<double> point = x0;
        math::IftResult scratch;
        std::vector<double> unused;
        (void)math::levenbergMarquardtDifferential(residual, x0, data, point, scratch, &unused,
                                                   lmOptions);
        return point;
    };
    for (std::size_t j = 0; j < quotes.size(); ++j) {
        std::vector<double> up = quotes;
        std::vector<double> down = quotes;
        up[j] += epsilon;
        down[j] -= epsilon;
        const std::vector<double> upFit = refit(up);
        const std::vector<double> downFit = refit(down);
        for (std::size_t k = 0; k < 2; ++k) {
            const double fd = (upFit[k] - downFit[k]) / (2.0 * epsilon);
            CHECK_CLOSE("LM differential vs FD", dpDm[k * quotes.size() + j], fd, 1e-4);
        }
    }

    // Var variant: the optimum is attached to the caller's tape.
    std::vector<stan::math::var> market(quotes.size());
    for (std::size_t j = 0; j < quotes.size(); ++j) {
        market[j] = quotes[j];
    }
    std::vector<stan::math::var> pHatVar;
    math::OptimizerState state;
    const math::OptimizeResult varStatus = math::levenbergMarquardtDifferentialVar(
        residual, market, x0, pHatVar, nullptr, &state, lmOptions);
    EXPECT_TRUE(varStatus == lm.status);
    EXPECT_TRUE(pHatVar.size() == 2);
    CHECK_CLOSE("LM var optimum a", pHatVar[0].val(), pHat[0], 1e-12);
    CHECK_CLOSE("LM var optimum b", pHatVar[1].val(), pHat[1], 1e-12);

    // The Var path fills OptimizerState exactly like the double path: the
    // gradient is the true J^T r vector (not a scalar norm).
    EXPECT_TRUE(state.x.size() == 2);
    CHECK_CLOSE_SEQ("LM var state x", state.x, pHat, 1e-12);
    EXPECT_TRUE(state.grad.size() == lm.gradient.size());
    CHECK_CLOSE_SEQ("LM var state grad = J^T r", state.grad, lm.gradient, 0.0);
    CHECK_CLOSE("LM var state f", state.f, lm.cost, 0.0);
    EXPECT_TRUE(state.iterations == static_cast<std::size_t>(lm.iterations));

    // grad() on an optimum var seeds its adjoint to one and pushes
    // dp/dm into the market leaves through the callback.
    stan::math::set_zero_all_adjoints();
    stan::math::grad(pHatVar[0].vi_);
    for (std::size_t j = 0; j < quotes.size(); ++j) {
        CHECK_CLOSE("LM var adjoint row 0", market[j].adj(), dpDm[0 * quotes.size() + j], 1e-12);
    }
    stan::math::set_zero_all_adjoints();
    stan::math::grad(pHatVar[1].vi_);
    for (std::size_t j = 0; j < quotes.size(); ++j) {
        CHECK_CLOSE("LM var adjoint row 1", market[j].adj(), dpDm[1 * quotes.size() + j], 1e-12);
    }
}

/// Curve-bootstrap-shaped refit map: a single implied discount factor fit to
/// a multi-term quote. `BrentSolver<var>` never differentiates the iteration;
/// the returned root is the exact implicit-function-theorem polish, so its
/// adjoints must equal `-R_m / R_x` analytically. The scale parameter
/// multiplies several residual terms.
TEST_F(LevenbergMarquardtAdTest, brentImplicitRootVectorQuotes) {
    const double t0 = 1.0;
    const double t1 = 2.0;
    const double root = 0.9;
    // m = [scale, price, c0, c1]; R(x, m) = m0*(m2*x^t0 + m3*x^t1) - m1
    std::vector<stan::math::var> m(4);
    m[0] = 1.5;
    m[2] = 0.4;
    m[3] = 0.6;
    m[1] = m[0].val() * (m[2].val() * std::pow(root, t0) + m[3].val() * std::pow(root, t1));

    const auto residual = [&](const auto& x) {
        using std::pow;
        return m[0] * (m[2] * pow(x, t0) + m[3] * pow(x, t1)) - m[1];
    };

    math::BrentSolver<stan::math::var> solver;
    solver.setMaxEvaluations(300);
    stan::math::var xHat = solver.solve(residual, 1e-12, stan::math::var(0.8), stan::math::var(0.5),
                                        stan::math::var(1.0));
    xHat.grad();

    CHECK_CLOSE("brent-quote root", xHat.val(), root, 1e-9);
    // R_x, then dx/dm_j = -R_m_j / R_x.
    const double rx = m[0].val() * (m[2].val() * t0 * std::pow(root, t0 - 1.0) +
                                    m[3].val() * t1 * std::pow(root, t1 - 1.0));
    CHECK_CLOSE("brent-quote d x/d scale", m[0].adj(),
                -(m[2].val() * std::pow(root, t0) + m[3].val() * std::pow(root, t1)) / rx, 1e-9);
    CHECK_CLOSE("brent-quote d x/d price", m[1].adj(), 1.0 / rx, 1e-9);
    CHECK_CLOSE("brent-quote d x/d c0", m[2].adj(), -m[0].val() * std::pow(root, t0) / rx, 1e-9);
    CHECK_CLOSE("brent-quote d x/d c1", m[3].adj(), -m[0].val() * std::pow(root, t1) / rx, 1e-9);

    // Direct implicitRoot idiom gate at the analytic root.
    stan::math::set_zero_all_adjoints();
    const stan::math::var direct = math::detail::implicitRoot(residual, root);
    CHECK_CLOSE("implicitRoot value", direct.val(), root, 1e-12);
    stan::math::grad(direct.vi_);
    CHECK_CLOSE("implicitRoot d x/d price", m[1].adj(), 1.0 / rx, 1e-9);
}

/// Direct unit gate on the non-stationarity refusal used by the IFT layer.
TEST_F(LevenbergMarquardtAdTest, stationarityGate) {
    math::LevenbergMarquardtResult stale;
    stale.gradientNorm = 1.0;
    stale.stationary = false;
    EXPECT_THROW(math::detail::requireStationary(stale), std::runtime_error);

    math::LevenbergMarquardtResult ok;
    ok.gradientNorm = 0.0;
    ok.stationary = true;
    EXPECT_NO_THROW(math::detail::requireStationary(ok)); // must not throw
}

/// Small map (2 parameters, 3 residuals): the Var callback adjoints must match
/// central finite differences of the double refit map.
TEST_F(LevenbergMarquardtAdTest, smallLeastSquaresVarGate) {
    const double trueA = 1.8;
    const double trueB = -0.4;
    const std::vector<double> times{0.5, 1.0, 1.5};
    std::vector<double> quotes(times.size());
    for (std::size_t i = 0; i < times.size(); ++i) {
        quotes[i] = trueA * std::exp(trueB * times[i]);
    }
    const auto residual = [&](const auto& x, const auto& m, auto& out) {
        using std::exp;
        out.clear();
        for (std::size_t i = 0; i < times.size(); ++i) {
            out.push_back(x[0] * exp(x[1] * times[i]) - m[i]);
        }
    };
    const std::vector<double> x0{1.0, -0.1};
    math::LevenbergMarquardtOptions lmOptions;
    lmOptions.xtol = 0.0;
    lmOptions.ftol = 0.0;
    lmOptions.gradientTol = 1e-16;
    lmOptions.maxIterations = 2000;
    lmOptions.lambda0 = 1e-10;

    std::vector<double> pHat;
    math::IftResult ift;
    std::vector<double> dpDm;
    math::LevenbergMarquardtResult lm;
    const math::OptimizeResult status = math::levenbergMarquardtDifferential(
        residual, x0, quotes, pHat, ift, &dpDm, lmOptions, {}, &lm);
    EXPECT_TRUE(usable(lm));
    EXPECT_TRUE(status == lm.status);
    EXPECT_TRUE(lm.stationary);

    // Central FD of the double refit map.
    const double epsilon = 1e-6;
    for (std::size_t j = 0; j < quotes.size(); ++j) {
        const auto refit = [&](const std::vector<double>& data) {
            std::vector<double> point = x0;
            math::IftResult scratch;
            std::vector<double> unused;
            (void)math::levenbergMarquardtDifferential(residual, x0, data, point, scratch, &unused,
                                                       lmOptions);
            return point;
        };
        std::vector<double> up = quotes;
        std::vector<double> down = quotes;
        up[j] += epsilon;
        down[j] -= epsilon;
        const std::vector<double> upFit = refit(up);
        const std::vector<double> downFit = refit(down);
        for (std::size_t k = 0; k < 2; ++k) {
            const double fd = (upFit[k] - downFit[k]) / (2.0 * epsilon);
            CHECK_CLOSE("small map dp/dm vs FD", dpDm[k * quotes.size() + j], fd, 1e-5);
        }
    }

    // Var path: same optimum, same state reporting, and a downstream scalar's
    // adjoint equals the chain-rule combination of dp/dm.
    std::vector<stan::math::var> market(quotes.size());
    for (std::size_t j = 0; j < quotes.size(); ++j) {
        market[j] = quotes[j];
    }
    std::vector<stan::math::var> pHatVar;
    math::OptimizerState state;
    const math::OptimizeResult varStatus = math::levenbergMarquardtDifferentialVar(
        residual, market, x0, pHatVar, &ift, &state, lmOptions);
    EXPECT_TRUE(varStatus == lm.status);
    CHECK_CLOSE("small map var x0", pHatVar[0].val(), pHat[0], 1e-12);
    CHECK_CLOSE("small map var x1", pHatVar[1].val(), pHat[1], 1e-12);
    EXPECT_TRUE(state.grad.size() == 2);

    stan::math::var phi = 0.5 * pHatVar[0] - 0.25 * pHatVar[1];
    stan::math::set_zero_all_adjoints();
    phi.grad();
    for (std::size_t j = 0; j < quotes.size(); ++j) {
        CHECK_CLOSE("small map var adjoint", market[j].adj(),
                    0.5 * dpDm[0 * quotes.size() + j] - 0.25 * dpDm[1 * quotes.size() + j], 1e-12);
    }
}

/// A parameter pinned by BOTH bounds must add a single KKT row: the previous
/// code pushed the same index twice, making the system rank deficient and
/// silently switching to the pseudo-inverse path.
TEST_F(LevenbergMarquardtAdTest, pinnedBoundDedup) {
    const auto f2 = [](const auto& x, const auto& m) {
        using Sx = typename std::decay_t<decltype(x)>::value_type;
        const Sx d = x[0] - Sx(m[0]);
        return Sx(0.5) * d * d;
    };
    math::Bounds bounds;
    bounds.lower = {1.0};
    bounds.upper = {1.0};
    math::IftResult ift;
    std::vector<double> dpDm;
    std::vector<double> dlambdaDm;
    std::vector<double> dnuDm;
    math::iftKkt(f2, math::NoConstraint{}, math::NoConstraint{}, bounds, {1.0}, {2.0}, {}, {}, dpDm,
                 dlambdaDm, dnuDm, ift);
    EXPECT_TRUE(ift.activeBounds.size() == 1);
    EXPECT_TRUE(ift.activeBounds[0] == 0);
    EXPECT_TRUE(!ift.pseudo_inverse);
    CHECK_CLOSE("pinned bound dp/dm", dpDm[0], 0.0, 1e-12);
}

/// Empty inputs and invalid options are rejected with std::invalid_argument
/// instead of indexing inside the residual.
TEST_F(LevenbergMarquardtAdTest, guards) {
    const auto residual = [](const auto& x, auto& out) {
        out.clear();
        out.push_back(x[0] - 1.0);
    };
    std::vector<double> empty;
    EXPECT_THROW((void)math::levenbergMarquardtAd(residual, empty), std::invalid_argument);

    const auto quadratic = [](const std::vector<double>& p, std::vector<double>& out) {
        out.clear();
        for (int i = 0; i < 5; ++i) {
            out.push_back(p[0] - 1.0);
        }
    };
    std::vector<double> params{0.0};
    math::LevenbergMarquardtOptions bad;
    bad.nu = 0.0;
    EXPECT_THROW((void)math::levenbergMarquardt(quadratic, params, bad), std::invalid_argument);

    bad = {};
    bad.lambda0 = -1.0;
    EXPECT_THROW((void)math::levenbergMarquardt(quadratic, params, bad), std::invalid_argument);
}

/// OptimizerState contract: `grad` is the true J^T r vector, `iterations` is
/// 1-based, and the double/Var paths report identically.
TEST_F(LevenbergMarquardtAdTest, stateContract) {
    const auto quadratic = [](const std::vector<double>& p, std::vector<double>& out) {
        out.clear();
        for (int i = 0; i < 21; ++i) {
            const double t = 0.5 * i;
            out.push_back(p[0] + p[1] * t + p[2] * t * t - (1.0 - 0.5 * t + 0.25 * t * t));
        }
    };
    std::vector<double> params{0.0, 0.0, 0.0};
    math::OptimizerState state;
    const math::LevenbergMarquardtResult fit =
        math::LevenbergMarquardt().minimize(quadratic, params, state);
    EXPECT_TRUE(usable(fit));
    EXPECT_TRUE(state.grad.size() == 3);
    CHECK_CLOSE_SEQ("state grad = J^T r", state.grad, fit.gradient, 0.0);
    EXPECT_TRUE(state.iterations == static_cast<std::size_t>(fit.iterations));
    EXPECT_TRUE(fit.iterations >= 1);
}
