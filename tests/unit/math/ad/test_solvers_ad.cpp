// test_solvers_ad.cpp — validates the automatic AD dispatch of the 1-D solvers
// (Math/Solvers/SolverStanPrimitives.h)
//
// For f(x; theta) = x^2 - theta, root x0 = sqrt(theta) at theta = 4:
//   value      x0        = 2
//   gradient   dx0/dtheta = 1/(2 sqrt(theta)) = 0.25
//   Hessian    d2x0/dtheta2 = -1/(4 theta^(3/2)) = -0.03125
//
//   var:       every solver (including bisection) must return the exact
//              implicit-function-theorem gradient via the double-precision
//              value solve + Newton polish
//   fvar<...>: pathwise route, Hessian via stan::math::hessian for the smooth
//              solvers (bisection's pathwise derivative is identically zero
//              and is documented as unsupported)
//   auto-bracketing: var gradient through the guess+step overload
#include "quantape/math/StanMath.h"

#include "quantape/math/Interpolations.h"
#include "quantape/math/Interpolations/InterpolationStanPrimitives.h"
#include "quantape/math/NumericalMethods.h"
#include "quantape/math/Solvers/SolverStanPrimitives.h"

#include <cmath>
#include <vector>

#include "support/GtestSupport.h"
#include "support/StanTapeFixture.h"

using stan::math::var;

class SolversAdTest : public StanTapeTest {};

namespace {

template <typename SolverT>
void checkVarGradient(const char* name, double tol) {
    SCOPED_TRACE(name);

    var theta = 4.0;
    SolverT solver;
    solver.setMaxEvaluations(300);
    auto f = [&theta](const auto& x) { return x * x - theta; };

    var root = solver.solve(f, 1e-12, var(1.5), var(0.0), var(3.0));
    root.grad();

    CHECK_CLOSE("var value", root.val(), 2.0, 1e-9);
    CHECK_CLOSE("var gradient", theta.adj(), 0.25, tol);
}

template <template <typename> class SolverTemplate>
void checkHessian(const char* name, double tol) {
    SCOPED_TRACE(name);

    Eigen::VectorXd x0(1);
    x0 << 4.0;

    const auto runner = [&](const auto& theta_eig) {
        using S = typename std::decay_t<decltype(theta_eig)>::Scalar;
        std::vector<S> theta(theta_eig.data(), theta_eig.data() + theta_eig.size());
        SolverTemplate<S> solver;
        solver.setMaxEvaluations(300);
        auto f = [&](const S& x) { return x * x - theta[0]; };
        return solver.solve(f, 1e-12, S(1.5), S(0.0), S(3.0));
    };

    double fx = 0.0;
    Eigen::VectorXd grad;
    Eigen::Matrix<double, -1, -1> hess;
    stan::math::hessian(runner, x0, fx, grad, hess);

    CHECK_CLOSE("fvar value", fx, 2.0, 1e-9);
    CHECK_CLOSE("fvar gradient", grad(0), 0.25, tol);
    CHECK_CLOSE("fvar hessian", hess(0, 0), -0.03125, tol);
}

} // namespace

TEST_F(SolversAdTest, varImplicitGradients) {
    checkVarGradient<quantape::math::BisectionSolver<var>>("bisection", 1e-12);
    checkVarGradient<quantape::math::BrentSolver<var>>("brent", 1e-12);
    checkVarGradient<quantape::math::SecantSolver<var>>("secant", 1e-12);
    checkVarGradient<quantape::math::FalsePositionSolver<var>>("falsepos", 1e-12);
    checkVarGradient<quantape::math::RidderSolver<var>>("ridder", 1e-12);
    checkVarGradient<quantape::math::NewtonSolver<var>>("newton-fd", 1e-12);
}

TEST_F(SolversAdTest, varOnlyObjectiveGradient) {
    var theta = 4.0;
    quantape::math::BrentSolver<var> solver;
    solver.setMaxEvaluations(300);
    // Non-generic objective: only callable with var (no double overload)
    auto f = [&theta](const var& x) { return x * x - theta; };

    var root = solver.solve(f, 1e-12, var(1.5), var(0.0), var(3.0));
    root.grad();
    CHECK_CLOSE("brent / var-only objective gradient", theta.adj(), 0.25, 1e-9);
}

TEST_F(SolversAdTest, autoBracketingGradient) {
    var theta = 1.0;
    quantape::math::BrentSolver<var> solver;
    solver.setMaxEvaluations(300);
    auto f = [&theta](const auto& x) { return x * x * x - x - 2.0 * theta; };

    var root = solver.solve(f, 1e-12, var(1.5), var(0.1)); // guess + step overload
    root.grad();

    const double r = root.val();
    const double expected = 2.0 / (3.0 * r * r - 1.0); // implicit derivative
    CHECK_CLOSE("brent-auto / var gradient", theta.adj(), expected, 1e-9);
}

TEST_F(SolversAdTest, newtonWithDerivativeGradient) {
    var theta = 4.0;
    auto f = [&theta](const var& x) { return x * x - theta; };
    auto df = [](const var& x) { return var(2.0) * x; };
    quantape::math::NewtonSolverWithDerivative<var, decltype(df)> solver(df);
    solver.setMaxEvaluations(300);

    var root = solver.solve(f, 1e-12, var(1.5), var(0.0), var(3.0));
    root.grad();
    CHECK_CLOSE("newton-with-derivative / var gradient", theta.adj(), 0.25, 1e-9);
}

TEST_F(SolversAdTest, compositeExpAndRootSquare) {
    // Solver output is a plain tape variable: it composes with any AD op.
    {
        var theta = 4.0;
        quantape::math::BrentSolver<var> solver;
        solver.setMaxEvaluations(300);
        auto f = [&theta](const auto& x) { return x * x - theta; };
        var root = solver.solve(f, 1e-12, var(1.5), var(0.0), var(3.0));

        var z = stan::math::exp(root);
        z.grad();
        CHECK_CLOSE("composite / exp(solve) value", z.val(), std::exp(2.0), 1e-9);
        CHECK_CLOSE("composite / d exp(root)/dtheta", theta.adj(), std::exp(2.0) * 0.25, 1e-9);
    }
    {
        var theta = 4.0;
        quantape::math::BrentSolver<var> solver;
        solver.setMaxEvaluations(300);
        auto f = [&theta](const auto& x) { return x * x - theta; };
        var root = solver.solve(f, 1e-12, var(1.5), var(0.0), var(3.0));

        var z = root * root; // d(z)/dtheta = 2 * root * 0.25 = 1
        z.grad();
        CHECK_CLOSE("composite / d root^2/dtheta", theta.adj(), 1.0, 1e-9);
    }
}

TEST_F(SolversAdTest, compositeInterpolationAndFixedAbscissa) {
    // interpolate(solve): the interpolation weights are DoubleT by
    // default, so the query coordinate carries the solve's graph.
    // theta = 2.25 -> root = 1.5, interp(1.5) = 2.5 on segment [1,2];
    // dz/dtheta = (dz/dx) * (dx/dtheta) = 3 * 1/(2*1.5) = 1.0
    var theta = 2.25;
    std::vector<var> xs{0.0, 1.0, 2.0, 3.0};
    std::vector<var> ys{0.0, 1.0, 4.0, 9.0};
    quantape::math::LinearInterpolation<var> interp(xs, ys);

    quantape::math::BrentSolver<var> solver;
    solver.setMaxEvaluations(300);
    auto f = [&theta](const auto& x) { return x * x - theta; };

    var root = solver.solve(f, 1e-12, var(1.5), var(0.0), var(3.0));
    var z = interp(root);
    z.grad();

    CHECK_CLOSE("composite / interp(solve) value", z.val(), 2.5, 1e-9);
    CHECK_CLOSE("composite / knot y[1] adjoint", ys[1].adj(), 0.5, 1e-12);
    CHECK_CLOSE("composite / knot y[2] adjoint", ys[2].adj(), 0.5, 1e-12);
    CHECK_CLOSE("composite / dz/dtheta", theta.adj(), 1.0, 1e-9);

    // Passive-abscissa policy drops the x-path (dz/dtheta = 0)
    var theta_fixed = 2.25;
    std::vector<var> xs_fixed{0.0, 1.0, 2.0, 3.0};
    std::vector<var> ys_fixed{0.0, 1.0, 4.0, 9.0};
    quantape::math::LinearInterpolation<var> interp_fixed(xs_fixed, ys_fixed);
    quantape::math::BrentSolver<var> solver_fixed;
    solver_fixed.setMaxEvaluations(300);
    auto f_fixed = [&theta_fixed](const auto& x) { return x * x - theta_fixed; };
    var root_fixed = solver_fixed.solve(f_fixed, 1e-12, var(1.5), var(0.0), var(3.0));
    var z_fixed = interp_fixed.evaluateFixed(root_fixed);
    z_fixed.grad();
    CHECK_CLOSE("composite / evaluateFixed dz/dtheta = 0", theta_fixed.adj(), 0.0, 1e-12);
}

TEST_F(SolversAdTest, fvarHessianMatchesAnalytic) {
    // bisection deliberately excluded: its pathwise derivative is zero.
    checkHessian<quantape::math::BrentSolver>("brent", 1e-8);
    checkHessian<quantape::math::SecantSolver>("secant", 1e-8);
    checkHessian<quantape::math::FalsePositionSolver>("falsepos", 1e-8);
    checkHessian<quantape::math::RidderSolver>("ridder", 1e-8);
    checkHessian<quantape::math::NewtonSolver>("newton-fd", 1e-8);
}
