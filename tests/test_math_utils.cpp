#include "quantape/math/StanMath.h"

#include "quantape/log/Log.h"
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
#include "quantape/util/Check.h"

#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <vector>

using namespace quantape;

namespace {

void testTriangularSolve() {
    // L = [[2,0,0],[1,3,0],[4,1,5]] row-major.
    const std::vector<double> l{2.0, 0.0, 0.0, 1.0, 3.0, 0.0, 4.0, 1.0, 5.0};
    const std::vector<double> xTrue{1.0, 2.0, 3.0};
    // b = L xTrue = [2, 7, 21].
    const std::vector<double> b{2.0, 7.0, 21.0};
    const std::vector<double> x = math::solveLowerTriangular<double>(l, 3, b);
    util::checkClose("lower solve", x, xTrue, 1e-14);

    // L^T x = bT with bT = L^T xTrue = [16, 9, 15].
    const std::vector<double> bT{16.0, 9.0, 15.0};
    const std::vector<double> xT = math::solveLowerTranspose<double>(l, 3, bT);
    util::checkClose("lower-transpose solve", xT, xTrue, 1e-14);

    bool threw = false;
    try {
        const std::vector<double> singular{0.0, 0.0, 0.0, 1.0, 3.0, 0.0, 4.0, 1.0, 5.0};
        (void)math::solveLowerTriangular<double>(singular, 3, b);
    } catch (const std::invalid_argument&) {
        threw = true;
    }
    CHECK(threw);

    threw = false;
    try {
        (void)math::solveLowerTriangular<double>(l, 2, b);
    } catch (const std::invalid_argument&) {
        threw = true;
    }
    CHECK(threw);
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
    CHECK(fit.status == math::OptimizeResult::XtolReached ||
          fit.status == math::OptimizeResult::FtolReached ||
          fit.status == math::OptimizeResult::GradientTolReached);
    util::checkClose("LM quadratic a", params[0], 1.0, 1e-8);
    util::checkClose("LM quadratic b", params[1], -0.5, 1e-8);
    util::checkClose("LM quadratic c", params[2], 0.25, 1e-8);

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
    CHECK(exactFit.status == math::OptimizeResult::XtolReached ||
          exactFit.status == math::OptimizeResult::FtolReached ||
          exactFit.status == math::OptimizeResult::GradientTolReached);
    util::checkClose("LM exact-J a", exactParams[0], 1.0, 1e-8);
    util::checkClose("LM exact-J b", exactParams[1], -0.5, 1e-8);
    util::checkClose("LM exact-J c", exactParams[2], 0.25, 1e-8);

    // Rosenbrock valley.
    const auto rosenbrock = [](const std::vector<double>& p, std::vector<double>& out) {
        out = {1.0 - p[0], 10.0 * (p[1] - p[0] * p[0])};
    };
    std::vector<double> rosen{-1.2, 1.0};
    math::LevenbergMarquardtOptions options;
    options.maxIterations = 400;
    const math::LevenbergMarquardtResult valley =
        math::levenbergMarquardt(rosenbrock, rosen, options);
    CHECK(valley.status != math::OptimizeResult::Failure);
    util::checkClose("LM rosenbrock x", rosen[0], 1.0, 1e-5);
    util::checkClose("LM rosenbrock y", rosen[1], 1.0, 1e-5);
}

void testTensionSplineMath() {
    const std::vector<double> x{0.0, 1.0, 2.0, 3.0, 4.0};
    const std::vector<double> y{0.0, 0.03, 0.035, 0.038, 0.04};
    const math::TensionSplineInterpolation<double> interp(x, y, 8.0);
    for (std::size_t j = 0; j < x.size(); ++j) {
        util::checkClose("tension node value", interp(x[j]), y[j], 1e-13);
    }
    // C1 continuity at the interior knots (finite differences at 1e-7).
    const double eps = 1e-7;
    for (std::size_t j = 1; j + 1 < x.size(); ++j) {
        const double left = (interp(x[j]) - interp(x[j] - eps)) / eps;
        const double right = (interp(x[j] + eps) - interp(x[j])) / eps;
        util::checkClose("tension derivative continuity", left, right, 1e-6);
    }
    // sigma -> 0 approaches the natural cubic spline.
    const math::TensionSplineInterpolation<double> small(x, y, 1e-3);
    const math::CubicInterpolation<double> cubic(x, y, math::CubicDerivativeApprox::Spline);
    for (const double t : {0.5, 1.5, 2.5, 3.5}) {
        util::checkClose("tension sigma->0 cubic limit", small(t), cubic(t), 1e-5);
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
    CHECK(result.converged);
    util::checkClose("fixed point x0", state[0], 2.0, 1e-10);
    util::checkClose("fixed point x1", state[1], 0.4, 1e-10);

    math::FixedPointOptions limited;
    limited.maxPasses = 3;
    std::vector<double> divergent{0.0};
    const auto drift = [](std::vector<double>& x) { x[0] += 1.0; };
    const auto scalarNorm = [](const std::vector<double>& a, const std::vector<double>& b) {
        return std::abs(b[0] - a[0]);
    };
    const math::FixedPointResult failed =
        math::fixedPointIterate(drift, divergent, scalarNorm, limited);
    CHECK(!failed.converged);
    CHECK(failed.passes == 3);
}

void testCubicAkimaSmallGrids() {
    // n == 3 (smallest Akima stencil): parabolic fallback. For y = x^2 on
    // {0, 1, 2} the fallback derivatives are exact at every node.
    {
        const std::vector<double> x{0.0, 1.0, 2.0};
        const std::vector<double> y{0.0, 1.0, 4.0};
        const math::CubicInterpolation<double> interp(x, y, math::CubicDerivativeApprox::Akima);
        CHECK(interp.aCoeffs().size() == 2);
        util::checkClose("akima n=3 node slopes", interp.aCoeffs(), std::vector<double>{0.0, 2.0},
                         1e-14);
        util::checkClose("akima n=3 value at 0.5", interp(0.5), 0.25, 1e-13);
        util::checkClose("akima n=3 value at 1.5", interp(1.5), 2.25, 1e-13);
        util::checkClose("akima n=3 value at 2.0", interp(2.0), 4.0, 1e-13);
    }
    // n == 2: single linear slope, evaluated at both nodes and the midpoint.
    {
        const std::vector<double> x{0.0, 2.0};
        const std::vector<double> y{1.0, 5.0};
        const math::CubicInterpolation<double> interp(x, y, math::CubicDerivativeApprox::Akima);
        CHECK(interp.aCoeffs().size() == 1);
        util::checkClose("akima n=2 slope", interp.aCoeffs()[0], 2.0, 1e-14);
        util::checkClose("akima n=2 value at 0", interp(0.0), 1.0, 1e-13);
        util::checkClose("akima n=2 value at 1", interp(1.0), 3.0, 1e-13);
        util::checkClose("akima n=2 value at 2", interp(2.0), 5.0, 1e-13);
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
    util::checkClose("monotone fvar<var> value", interp(atPoint).val_.val(), reference(1.5), 1e-13);
    util::checkClose("monotone fvar<var> derivative", interp.derivative(atPoint).val_.val(),
                     reference.derivative(1.5), 1e-13);
    stan::math::recover_memory();
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
    CHECK(x.size() == expected.size());
    for (std::size_t i = 0; i < expected.size(); ++i) {
        util::checkClose("tridiagonal mixed rhs", x[i].val(), expected[i], 1e-13);
    }

    // Real caller: the tension spline stores passive double diagonals and
    // passes a vector<DoubleT> right-hand side.
    const std::vector<double> tx{0.0, 1.0, 2.0, 3.0};
    const std::vector<double> ty{0.0, 0.03, 0.035, 0.038};
    const std::vector<var> tyAd{0.0, 0.03, 0.035, 0.038};
    const math::TensionSplineInterpolation<double> tensionRef(tx, ty, 8.0);
    const math::TensionSplineInterpolation<var> tensionAd(tx, tyAd, 8.0);
    util::checkClose("tension var value", tensionAd(1.5).val(), tensionRef(1.5), 1e-12);
    stan::math::recover_memory();
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
    util::checkClose("loglin derivative fixed h00", hessian(0, 0), -0.078327, 1e-5);
    util::checkClose("loglin derivative fixed h11", hessian(1, 1), -0.039295, 1e-5);
    util::checkClose("loglin derivative fixed h01", hessian(0, 1), 0.055485, 1e-5);
    util::checkClose("loglin derivative fixed h10", hessian(1, 0), 0.055485, 1e-5);
    stan::math::recover_memory();
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
    CHECK(interp.slopes().size() == x.size());
    for (std::size_t i = 0; i < x.size(); ++i) {
        util::checkClose("hyman node value", interp(x[i]), y[i], 1e-13);
    }
    // C1 continuity at the interior knots.
    const double eps = 1e-10;
    for (std::size_t i = 1; i + 1 < x.size(); ++i) {
        util::checkClose("hyman C1 continuity", interp.derivative(x[i] - eps),
                         interp.derivative(x[i] + eps), 1e-9);
    }
    // Monotone values inside every local data range, non-negative derivative.
    for (std::size_t i = 0; i + 1 < x.size(); ++i) {
        const double low = std::min(y[i], y[i + 1]);
        const double high = std::max(y[i], y[i + 1]);
        for (int k = 0; k <= 20; ++k) {
            const double t = x[i] + (x[i + 1] - x[i]) * (static_cast<double>(k) / 20.0);
            const double value = interp(t);
            CHECK(value >= low - 1e-12);
            CHECK(value <= high + 1e-12);
            CHECK(interp.derivative(t) >= -1e-12);
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
            CHECK(hump(t) >= low - 1e-12);
            CHECK(hump(t) <= high + 1e-12);
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

    stan::math::recover_memory();
    {
        std::vector<var> yAd(y.begin(), y.end());
        const math::HymanSplineInterpolation<var> interp(x, yAd);
        var objective = 0.0;
        for (const double query : queries) {
            const var value = interp(var(query));
            util::checkClose("hyman var value", value.val(), reference(query), 1e-12);
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
            util::checkClose("hyman var gradient", yAd[j].adj(), fd, 1e-6);
        }
    }
    stan::math::recover_memory();
    {
        std::vector<FvarVar> yAd;
        yAd.reserve(y.size());
        for (const double value : y) {
            yAd.emplace_back(value);
        }
        const math::HymanSplineInterpolation<FvarVar> interp(x, yAd);
        const FvarVar atPoint(1.5);
        util::checkClose("hyman fvar<var> value", interp(atPoint).val_.val(), reference(1.5),
                         1e-13);
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
            util::checkClose("hyman fvar<var> gradient", interpolant(atPoint).d_.val(), fd, 1e-6);
        }
    }
    stan::math::recover_memory();
}

void testBilinearInvalidGrid() {
    bool threw = false;
    try {
        // x grid with a single point: locateX would compute size() - 2.
        const std::vector<double> x{0.0};
        const std::vector<double> y{0.0, 1.0};
        const std::vector<std::vector<double>> z{{1.0}, {2.0}};
        (void)math::BilinearInterpolation<double>(x, y, z);
    } catch (const std::invalid_argument&) {
        threw = true;
    }
    CHECK(threw);

    threw = false;
    try {
        const std::vector<double> x{0.0, 1.0};
        const std::vector<double> y{0.0};
        const std::vector<std::vector<double>> z{{1.0, 2.0}};
        (void)math::BilinearInterpolation<double>(x, y, z);
    } catch (const std::invalid_argument&) {
        threw = true;
    }
    CHECK(threw);

    threw = false;
    try {
        // Non-increasing x grid breaks locateX's contract.
        const std::vector<double> x{1.0, 1.0};
        const std::vector<double> y{0.0, 1.0};
        const std::vector<std::vector<double>> z{{1.0, 2.0}, {3.0, 4.0}};
        (void)math::BilinearInterpolation<double>(x, y, z);
    } catch (const std::invalid_argument&) {
        threw = true;
    }
    CHECK(threw);
}

} // namespace

int main() {
    testTriangularSolve();
    testLevenbergMarquardt();
    testTensionSplineMath();
    testFixedPointIterator();
    testCubicAkimaSmallGrids();
    testMonotoneCubicFvarVar();
    testHymanSplineMath();
    testHymanSplineAd();
    testTridiagonalMixedScalars();
    testLogLinearDerivativeFixedHessian();
    testBilinearInvalidGrid();
    QTA_LOG_INFO("test", "test_math_utils: ok");
    return 0;
}
