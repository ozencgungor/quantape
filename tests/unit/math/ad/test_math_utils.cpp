// test_math_utils.cpp — interpolation/solver primitives, including their
// Stan AD dispatch (var and fvar<var>).
#include "quantape/math/StanMath.h"

#include "quantape/math/Interpolations/BilinearInterpolation.h"
#include "quantape/math/Interpolations/CubicInterpolation.h"
#include "quantape/math/Interpolations/HymanSplineInterpolation.h"
#include "quantape/math/Interpolations/InterpolationStanPrimitives.h"
#include "quantape/math/Interpolations/LogLinearInterpolation.h"
#include "quantape/math/Interpolations/MonotoneCubicInterpolation.h"
#include "quantape/math/Interpolations/TensionSplineInterpolation.h"
#include "quantape/math/LinearAlgebra/TriangularSolve.h"
#include "quantape/math/Optimization/LevenbergMarquardt.h"
#include "quantape/math/Solvers/FixedPointIterator.h"
#include "quantape/math/Solvers/TridiagonalSolver.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <vector>

#include "support/GtestSupport.h"
#include "support/StanTapeFixture.h"

using namespace quantape;

class MathUtilsTest : public StanTapeTest {};

namespace {

void testTriangularSolve() {
    // L = [[2,0,0],[1,3,0],[4,1,5]] row-major.
    const std::vector<double> l{2.0, 0.0, 0.0, 1.0, 3.0, 0.0, 4.0, 1.0, 5.0};
    const std::vector<double> xTrue{1.0, 2.0, 3.0};
    // b = L xTrue = [2, 7, 21].
    const std::vector<double> b{2.0, 7.0, 21.0};
    const std::vector<double> x = math::solveLowerTriangular<double>(l, 3, b);
    CHECK_CLOSE_SEQ("lower solve", x, xTrue, 1e-14);

    // L^T x = bT with bT = L^T xTrue = [16, 9, 15].
    const std::vector<double> bT{16.0, 9.0, 15.0};
    const std::vector<double> xT = math::solveLowerTranspose<double>(l, 3, bT);
    CHECK_CLOSE_SEQ("lower-transpose solve", xT, xTrue, 1e-14);

    const std::vector<double> singular{0.0, 0.0, 0.0, 1.0, 3.0, 0.0, 4.0, 1.0, 5.0};
    EXPECT_THROW((void)math::solveLowerTriangular<double>(singular, 3, b), std::invalid_argument);

    EXPECT_THROW((void)math::solveLowerTriangular<double>(l, 2, b), std::invalid_argument);
}

void testLevenbergMarquardt() {
    // Exact quadratic fit through dense samples.
    const auto quadratic = [](const std::vector<double>& p, std::vector<double>& out) {
        out.clear();
        for (int i = 0; i < 21; ++i) {
            const double t = 0.5 * i;
            out.push_back(p[0] + p[1] * t + p[2] * t * t - (1.0 - 0.5 * t + 0.25 * t * t));
        }
    };
    std::vector<double> params{0.0, 0.0, 0.0};
    const math::LevenbergMarquardtResult fit = math::levenbergMarquardt(quadratic, params);
    EXPECT_TRUE(fit.status == math::OptimizeResult::XtolReached ||
                fit.status == math::OptimizeResult::FtolReached ||
                fit.status == math::OptimizeResult::GradientTolReached);
    CHECK_CLOSE("LM quadratic a", params[0], 1.0, 1e-8);
    CHECK_CLOSE("LM quadratic b", params[1], -0.5, 1e-8);
    CHECK_CLOSE("LM quadratic c", params[2], 0.25, 1e-8);

    // Exact-Jacobian overload agrees with the finite-difference route.
    const auto quadraticJacobian = [](const std::vector<double>&, std::vector<double>& flat) {
        // r_i = p0 + p1 t + p2 t^2 - y_i with dr/dp = [1, t, t^2].
        for (int i = 0; i < 21; ++i) {
            const double t = 0.5 * i;
            flat[static_cast<std::size_t>(3 * i + 0)] = 1.0;
            flat[static_cast<std::size_t>(3 * i + 1)] = t;
            flat[static_cast<std::size_t>(3 * i + 2)] = t * t;
        }
    };
    std::vector<double> exactParams{0.2, 0.1, 0.05};
    const math::LevenbergMarquardtResult exactFit =
        math::levenbergMarquardt(quadratic, quadraticJacobian, exactParams);
    EXPECT_TRUE(exactFit.status == math::OptimizeResult::XtolReached ||
                exactFit.status == math::OptimizeResult::FtolReached ||
                exactFit.status == math::OptimizeResult::GradientTolReached);
    CHECK_CLOSE("LM exact-J a", exactParams[0], 1.0, 1e-8);
    CHECK_CLOSE("LM exact-J b", exactParams[1], -0.5, 1e-8);
    CHECK_CLOSE("LM exact-J c", exactParams[2], 0.25, 1e-8);

    // Rosenbrock valley.
    const auto rosenbrock = [](const std::vector<double>& p, std::vector<double>& out) {
        out = {1.0 - p[0], 10.0 * (p[1] - p[0] * p[0])};
    };
    std::vector<double> rosen{-1.2, 1.0};
    math::LevenbergMarquardtOptions options;
    options.maxIterations = 400;
    const math::LevenbergMarquardtResult valley =
        math::levenbergMarquardt(rosenbrock, rosen, options);
    EXPECT_TRUE(valley.status != math::OptimizeResult::Failure);
    CHECK_CLOSE("LM rosenbrock x", rosen[0], 1.0, 1e-5);
    CHECK_CLOSE("LM rosenbrock y", rosen[1], 1.0, 1e-5);
}

void testTensionSplineMath() {
    const std::vector<double> x{0.0, 1.0, 2.0, 3.0, 4.0};
    const std::vector<double> y{0.0, 0.03, 0.035, 0.038, 0.04};
    const math::TensionSplineInterpolation<double> interp(x, y, 8.0);
    for (std::size_t j = 0; j < x.size(); ++j) {
        CHECK_CLOSE("tension node value", interp(x[j]), y[j], 1e-13);
    }
    // C1 continuity at the interior knots (finite differences at 1e-7).
    const double eps = 1e-7;
    for (std::size_t j = 1; j + 1 < x.size(); ++j) {
        const double left = (interp(x[j]) - interp(x[j] - eps)) / eps;
        const double right = (interp(x[j] + eps) - interp(x[j])) / eps;
        CHECK_CLOSE("tension derivative continuity", left, right, 1e-6);
    }
    // sigma -> 0 approaches the natural cubic spline.
    const math::TensionSplineInterpolation<double> small(x, y, 1e-3);
    const math::CubicInterpolation<double> cubic(x, y, math::CubicDerivativeApprox::Spline);
    for (const double t : {0.5, 1.5, 2.5, 3.5}) {
        CHECK_CLOSE("tension sigma->0 cubic limit", small(t), cubic(t), 1e-5);
    }
}

void testFixedPointIterator() {
    std::vector<double> state{0.0, 0.0};
    const auto pass = [](std::vector<double>& x) {
        x[0] = 0.5 * x[0] + 1.0;
        x[1] = -0.25 * x[1] + 0.5;
    };
    const auto norm = [](const std::vector<double>& a, const std::vector<double>& b) {
        double m = 0.0;
        for (std::size_t i = 0; i < a.size(); ++i) {
            m = std::max(m, std::abs(b[i] - a[i]));
        }
        return m;
    };
    math::FixedPointOptions options;
    options.maxPasses = 60;
    const math::FixedPointResult result = math::fixedPointIterate(pass, state, norm, options);
    EXPECT_TRUE(result.converged);
    CHECK_CLOSE("fixed point x0", state[0], 2.0, 1e-10);
    CHECK_CLOSE("fixed point x1", state[1], 0.4, 1e-10);

    math::FixedPointOptions limited;
    limited.maxPasses = 3;
    std::vector<double> divergent{0.0};
    const auto drift = [](std::vector<double>& x) { x[0] += 1.0; };
    const auto scalarNorm = [](const std::vector<double>& a, const std::vector<double>& b) {
        return std::abs(b[0] - a[0]);
    };
    const math::FixedPointResult failed =
        math::fixedPointIterate(drift, divergent, scalarNorm, limited);
    EXPECT_FALSE(failed.converged);
    EXPECT_EQ(failed.passes, 3);
}

void testCubicAkimaSmallGrids() {
    // n == 3 (smallest Akima stencil): parabolic fallback. For y = x^2 on
    // {0, 1, 2} the fallback derivatives are exact at every node.
    {
        const std::vector<double> x{0.0, 1.0, 2.0};
        const std::vector<double> y{0.0, 1.0, 4.0};
        const math::CubicInterpolation<double> interp(x, y, math::CubicDerivativeApprox::Akima);
        EXPECT_EQ(interp.aCoeffs().size(), 2U);
        CHECK_CLOSE_SEQ("akima n=3 node slopes", interp.aCoeffs(), (std::vector<double>{0.0, 2.0}),
                        1e-14);
        CHECK_CLOSE("akima n=3 value at 0.5", interp(0.5), 0.25, 1e-13);
        CHECK_CLOSE("akima n=3 value at 1.5", interp(1.5), 2.25, 1e-13);
        CHECK_CLOSE("akima n=3 value at 2.0", interp(2.0), 4.0, 1e-13);
    }
    // n == 2: single linear slope, evaluated at both nodes and the midpoint.
    {
        const std::vector<double> x{0.0, 2.0};
        const std::vector<double> y{1.0, 5.0};
        const math::CubicInterpolation<double> interp(x, y, math::CubicDerivativeApprox::Akima);
        EXPECT_EQ(interp.aCoeffs().size(), 1U);
        CHECK_CLOSE("akima n=2 slope", interp.aCoeffs()[0], 2.0, 1e-14);
        CHECK_CLOSE("akima n=2 value at 0", interp(0.0), 1.0, 1e-13);
        CHECK_CLOSE("akima n=2 value at 1", interp(1.0), 3.0, 1e-13);
        CHECK_CLOSE("akima n=2 value at 2", interp(2.0), 5.0, 1e-13);
    }
}

void testMonotoneCubicFvarVar() {
    using FvarVar = stan::math::fvar<stan::math::var>;
    const std::vector<double> x{0.0, 1.0, 2.0, 3.0};
    const std::vector<double> y{0.0, 1.0, 4.0, 9.0};
    std::vector<FvarVar> yAd;
    yAd.reserve(y.size());
    for (const double value : y) {
        yAd.emplace_back(value);
    }
    const math::MonotoneCubicInterpolation<double> reference(x, y);
    const math::MonotoneCubicInterpolation<FvarVar> interp(x, yAd);
    const FvarVar atPoint(1.5);
    CHECK_CLOSE("monotone fvar<var> value", interp(atPoint).val_.val(), reference(1.5), 1e-13);
    CHECK_CLOSE("monotone fvar<var> derivative", interp.derivative(atPoint).val_.val(),
                reference.derivative(1.5), 1e-13);
}

void testTridiagonalMixedScalars() {
    using stan::math::var;
    // [[4, -1, 0], [-1, 4, -1], [0, -1, 4]] x = [1, 2, 3]; the first
    // sub-diagonal and last super-diagonal entries are unused.
    const std::vector<double> sub{0.0, -1.0, -1.0};
    const std::vector<double> diag{4.0, 4.0, 4.0};
    const std::vector<double> super{-1.0, -1.0, 0.0};
    const std::vector<var> rhs{1.0, 2.0, 3.0};
    const std::vector<var> x = math::TridiagonalSolver<var>::solve(sub, diag, super, rhs);
    const std::vector<double> expected{13.0 / 28.0, 6.0 / 7.0, 27.0 / 28.0};
    ASSERT_EQ(x.size(), expected.size());
    for (std::size_t i = 0; i < expected.size(); ++i) {
        CHECK_CLOSE("tridiagonal mixed rhs", x[i].val(), expected[i], 1e-13);
    }

    // Real caller: the tension spline stores passive double diagonals and
    // passes a vector<DoubleT> right-hand side.
    const std::vector<double> tx{0.0, 1.0, 2.0, 3.0};
    const std::vector<double> ty{0.0, 0.03, 0.035, 0.038};
    const std::vector<var> tyAd{0.0, 0.03, 0.035, 0.038};
    const math::TensionSplineInterpolation<double> tensionRef(tx, ty, 8.0);
    const math::TensionSplineInterpolation<var> tensionAd(tx, tyAd, 8.0);
    CHECK_CLOSE("tension var value", tensionAd(1.5).val(), tensionRef(1.5), 1e-12);
}

void testLogLinearDerivativeFixedHessian() {
    // Finite-difference reference for f'(x) with f = exp((1-t) log y0 + t log y1)
    // at t = 0.5, inv_dx = 1.3, y0 = 1.7, y1 = 2.4.
    const std::vector<double> x{0.0, 1.0 / 1.3};
    const double query = 0.5 / 1.3;
    const std::vector<double> y{1.7, 2.4};

    Eigen::VectorXd flat(2);
    flat << y[0], y[1];
    const auto runner = [&](const auto& v) {
        using S = typename std::decay_t<decltype(v)>::Scalar;
        const std::vector<S> nodes{v(0), v(1)};
        const math::LogLinearInterpolation<S> interp(x, nodes);
        return interp.derivativeFixed(S(query));
    };
    double value = 0.0;
    Eigen::VectorXd gradient;
    Eigen::Matrix<double, -1, -1> hessian;
    stan::math::hessian(runner, flat, value, gradient, hessian);
    CHECK_CLOSE("loglin derivative fixed h00", hessian(0, 0), -0.078327, 1e-5);
    CHECK_CLOSE("loglin derivative fixed h11", hessian(1, 1), -0.039295, 1e-5);
    CHECK_CLOSE("loglin derivative fixed h01", hessian(0, 1), 0.055485, 1e-5);
    CHECK_CLOSE("loglin derivative fixed h10", hessian(1, 0), 0.055485, 1e-5);
}

double hymanPinnedValue(const std::vector<double>& x, const std::vector<double>& y,
                        const std::vector<double>& slopes, double t) {
    std::size_t i = 0;
    while (i + 2 < x.size() && t >= x[i + 1]) {
        ++i;
    }
    const double h = x[i + 1] - x[i];
    const double u = (t - x[i]) / h;
    const double u2 = u * u;
    const double u3 = u2 * u;
    const double h00 = 2.0 * u3 - 3.0 * u2 + 1.0;
    const double h10 = u3 - 2.0 * u2 + u;
    const double h01 = -2.0 * u3 + 3.0 * u2;
    const double h11 = u3 - u2;
    return h00 * y[i] + h10 * h * slopes[i] + h01 * y[i + 1] + h11 * h * slopes[i + 1];
}

void testHymanSplineMath() {
    // Strictly increasing data on irregular nodes: y = 3x + sin(x).
    const std::vector<double> x{0.0, 0.4, 1.1, 1.9, 3.2, 4.7};
    std::vector<double> y(x.size());
    for (std::size_t i = 0; i < x.size(); ++i) {
        y[i] = 3.0 * x[i] + std::sin(x[i]);
    }
    const math::HymanSplineInterpolation<double> interp(x, y);
    EXPECT_EQ(interp.slopes().size(), x.size());
    for (std::size_t i = 0; i < x.size(); ++i) {
        CHECK_CLOSE("hyman node value", interp(x[i]), y[i], 1e-13);
    }
    // C1 continuity at the interior knots.
    const double eps = 1e-10;
    for (std::size_t i = 1; i + 1 < x.size(); ++i) {
        CHECK_CLOSE("hyman C1 continuity", interp.derivative(x[i] - eps),
                    interp.derivative(x[i] + eps), 1e-9);
    }
    // Monotone values inside every local data range, non-negative derivative.
    for (std::size_t i = 0; i + 1 < x.size(); ++i) {
        const double low = std::min(y[i], y[i + 1]);
        const double high = std::max(y[i], y[i + 1]);
        for (int k = 0; k <= 20; ++k) {
            const double t = x[i] + (x[i + 1] - x[i]) * (static_cast<double>(k) / 20.0);
            const double value = interp(t);
            EXPECT_GE(value, low - 1e-12);
            EXPECT_LE(value, high + 1e-12);
            EXPECT_GE(interp.derivative(t), -1e-12);
        }
    }

    // Hump data with turning points: no overshoot on any monotone segment.
    const std::vector<double> humpX{0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0};
    const std::vector<double> humpY{0.0, 1.1, 1.4, 1.05, 0.6, 0.75, 1.3};
    const math::HymanSplineInterpolation<double> hump(humpX, humpY);
    for (std::size_t i = 0; i + 1 < humpX.size(); ++i) {
        const double low = std::min(humpY[i], humpY[i + 1]);
        const double high = std::max(humpY[i], humpY[i + 1]);
        for (int k = 0; k <= 40; ++k) {
            const double t = humpX[i] + (humpX[i + 1] - humpX[i]) * (static_cast<double>(k) / 40.0);
            EXPECT_GE(hump(t), low - 1e-12);
            EXPECT_LE(hump(t), high + 1e-12);
        }
    }
}

void testHymanSplineAd() {
    using stan::math::var;
    using FvarVar = stan::math::fvar<stan::math::var>;
    const std::vector<double> x{0.0, 0.4, 1.1, 1.9, 3.2, 4.7};
    std::vector<double> y(x.size());
    for (std::size_t i = 0; i < x.size(); ++i) {
        y[i] = 3.0 * x[i] + std::sin(x[i]);
    }
    const math::HymanSplineInterpolation<double> reference(x, y);
    const std::vector<double> queries{0.2, 0.75, 1.5, 2.4, 3.9, 4.6};
    const auto pinnedObjective = [&](const std::vector<double>& nodes) {
        double total = 0.0;
        for (const double query : queries) {
            total += hymanPinnedValue(x, nodes, reference.slopes(), query);
        }
        return total;
    };

    {
        std::vector<var> yAd(y.begin(), y.end());
        const math::HymanSplineInterpolation<var> interp(x, yAd);
        var objective = 0.0;
        for (const double query : queries) {
            const var value = interp(var(query));
            CHECK_CLOSE("hyman var value", value.val(), reference(query), 1e-12);
            objective += value;
        }
        objective.grad();
        const double epsilon = 1e-6;
        for (std::size_t j = 0; j < y.size(); ++j) {
            std::vector<double> plus = y;
            std::vector<double> minus = y;
            plus[j] += epsilon;
            minus[j] -= epsilon;
            const double fd = (pinnedObjective(plus) - pinnedObjective(minus)) / (2.0 * epsilon);
            CHECK_CLOSE("hyman var gradient", yAd[j].adj(), fd, 1e-6);
        }
    }
    {
        std::vector<FvarVar> yAd;
        yAd.reserve(y.size());
        for (const double value : y) {
            yAd.emplace_back(value);
        }
        const math::HymanSplineInterpolation<FvarVar> interp(x, yAd);
        const FvarVar atPoint(1.5);
        CHECK_CLOSE("hyman fvar<var> value", interp(atPoint).val_.val(), reference(1.5), 1e-13);
        const double epsilon = 1e-6;
        for (std::size_t j = 0; j < y.size(); ++j) {
            std::vector<FvarVar> directional = yAd;
            directional[j].d_ = 1.0;
            const math::HymanSplineInterpolation<FvarVar> interpolant(x, directional);
            std::vector<double> plus = y;
            std::vector<double> minus = y;
            plus[j] += epsilon;
            minus[j] -= epsilon;
            const double fd = (hymanPinnedValue(x, plus, reference.slopes(), 1.5) -
                               hymanPinnedValue(x, minus, reference.slopes(), 1.5)) /
                              (2.0 * epsilon);
            CHECK_CLOSE("hyman fvar<var> gradient", interpolant(atPoint).d_.val(), fd, 1e-6);
        }
    }
}

void testBilinearInvalidGrid() {
    // x grid with a single point: locateX would compute size() - 2.
    const std::vector<double> x1{0.0};
    const std::vector<double> y1{0.0, 1.0};
    const std::vector<std::vector<double>> z1{{1.0}, {2.0}};
    EXPECT_THROW((void)math::BilinearInterpolation<double>(x1, y1, z1), std::invalid_argument);

    const std::vector<double> x2{0.0, 1.0};
    const std::vector<double> y2{0.0};
    const std::vector<std::vector<double>> z2{{1.0, 2.0}};
    EXPECT_THROW((void)math::BilinearInterpolation<double>(x2, y2, z2), std::invalid_argument);

    // Non-increasing x grid breaks locateX's contract.
    const std::vector<double> x3{1.0, 1.0};
    const std::vector<double> y3{0.0, 1.0};
    const std::vector<std::vector<double>> z3{{1.0, 2.0}, {3.0, 4.0}};
    EXPECT_THROW((void)math::BilinearInterpolation<double>(x3, y3, z3), std::invalid_argument);
}

} // namespace

TEST_F(MathUtilsTest, triangularSolveAndRejectsBadInput) {
    testTriangularSolve();
}

TEST_F(MathUtilsTest, levenbergMarquardtFitsQuadraticAndRosenbrock) {
    testLevenbergMarquardt();
}

TEST_F(MathUtilsTest, tensionSplineNodesContinuityAndCubicLimit) {
    testTensionSplineMath();
}

TEST_F(MathUtilsTest, fixedPointConvergesAndStopsAtMaxPasses) {
    testFixedPointIterator();
}

TEST_F(MathUtilsTest, cubicAkimaSmallGrids) {
    testCubicAkimaSmallGrids();
}

TEST_F(MathUtilsTest, monotoneCubicFvarVar) {
    testMonotoneCubicFvarVar();
}

TEST_F(MathUtilsTest, tridiagonalMixedScalars) {
    testTridiagonalMixedScalars();
}

TEST_F(MathUtilsTest, logLinearDerivativeFixedHessian) {
    testLogLinearDerivativeFixedHessian();
}

TEST_F(MathUtilsTest, hymanSplineMonotoneValues) {
    testHymanSplineMath();
}

TEST_F(MathUtilsTest, hymanSplineAdGradients) {
    testHymanSplineAd();
}

TEST_F(MathUtilsTest, bilinearRejectsInvalidGrids) {
    testBilinearInvalidGrid();
}
