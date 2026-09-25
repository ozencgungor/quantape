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
//
// Run: ./test_solvers_ad
#include "quantape/math/Interpolations.h"
#include "quantape/math/Interpolations/InterpolationStanPrimitives.h"
#include "quantape/math/NumericalMethods.h"
#include "quantape/math/Solvers/SolverStanPrimitives.h"
#include "quantape/math/StanMath.h"

#include <cmath>
#include <cstdio>
#include <vector>

using stan::math::var;

namespace {

int failures = 0;

void check(const char* label, double got, double expected, double tol) {
    const bool ok = std::fabs(got - expected) <= tol;
    std::printf("  %-42s %s  got=%.12f expected=%.12f err=%.3g\n", label, ok ? "ok" : "FAIL", got,
                expected, std::fabs(got - expected));
    if (!ok) {
        ++failures;
    }
}

template <typename SolverT>
void checkVarGradient(const char* name, double tol) {
    stan::math::recover_memory();

    var theta = 4.0;
    SolverT solver;
    solver.setMaxEvaluations(300);
    auto f = [&theta](const auto& x) { return x * x - theta; };

    var root = solver.solve(f, 1e-12, var(1.5), var(0.0), var(3.0));
    root.grad();

    char label[128];
    std::snprintf(label, sizeof(label), "%s / var value", name);
    check(label, root.val(), 2.0, 1e-9);
    std::snprintf(label, sizeof(label), "%s / var gradient", name);
    check(label, theta.adj(), 0.25, tol);
}

template <template <typename> class SolverTemplate>
void checkHessian(const char* name, double tol) {
    stan::math::recover_memory();

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

    char label[128];
    std::snprintf(label, sizeof(label), "%s / fvar value", name);
    check(label, fx, 2.0, 1e-9);
    std::snprintf(label, sizeof(label), "%s / fvar gradient", name);
    check(label, grad(0), 0.25, tol);
    std::snprintf(label, sizeof(label), "%s / fvar hessian", name);
    check(label, hess(0, 0), -0.03125, tol);
}

void checkAutoBracket() {
    stan::math::recover_memory();

    var theta = 1.0;
    quantape::math::BrentSolver<var> solver;
    solver.setMaxEvaluations(300);
    auto f = [&theta](const auto& x) { return x * x * x - x - 2.0 * theta; };

    var root = solver.solve(f, 1e-12, var(1.5), var(0.1)); // guess + step overload
    root.grad();

    const double r = root.val();
    const double expected = 2.0 / (3.0 * r * r - 1.0); // implicit derivative
    check("brent-auto / var gradient", theta.adj(), expected, 1e-9);
}

void checkVarOnlyObjective() {
    stan::math::recover_memory();

    var theta = 4.0;
    quantape::math::BrentSolver<var> solver;
    solver.setMaxEvaluations(300);
    // Non-generic objective: only callable with var (no double overload)
    auto f = [&theta](const var& x) { return x * x - theta; };

    var root = solver.solve(f, 1e-12, var(1.5), var(0.0), var(3.0));
    root.grad();
    check("brent / var-only objective gradient", theta.adj(), 0.25, 1e-9);
}

void checkNewtonWithDerivative() {
    stan::math::recover_memory();

    var theta = 4.0;
    auto f = [&theta](const var& x) { return x * x - theta; };
    auto df = [](const var& x) { return var(2.0) * x; };
    quantape::math::NewtonSolverWithDerivative<var, decltype(df)> solver(df);
    solver.setMaxEvaluations(300);

    var root = solver.solve(f, 1e-12, var(1.5), var(0.0), var(3.0));
    root.grad();
    check("newton-with-derivative / var gradient", theta.adj(), 0.25, 1e-9);
}

void checkComposite() {
    // Solver output is a plain tape variable: it composes with any AD op.
    {
        stan::math::recover_memory();
        var theta = 4.0;
        quantape::math::BrentSolver<var> solver;
        solver.setMaxEvaluations(300);
        auto f = [&theta](const auto& x) { return x * x - theta; };
        var root = solver.solve(f, 1e-12, var(1.5), var(0.0), var(3.0));

        var z = stan::math::exp(root);
        z.grad();
        check("composite / exp(solve) value", z.val(), std::exp(2.0), 1e-9);
        check("composite / d exp(root)/dtheta", theta.adj(), std::exp(2.0) * 0.25, 1e-9);
    }
    {
        stan::math::recover_memory();
        var theta = 4.0;
        quantape::math::BrentSolver<var> solver;
        solver.setMaxEvaluations(300);
        auto f = [&theta](const auto& x) { return x * x - theta; };
        var root = solver.solve(f, 1e-12, var(1.5), var(0.0), var(3.0));

        var z = root * root; // d(z)/dtheta = 2 * root * 0.25 = 1
        z.grad();
        check("composite / d root^2/dtheta", theta.adj(), 1.0, 1e-9);
    }
    {
        // interpolate(solve): the interpolation weights are DoubleT by
        // default, so the query coordinate carries the solve's graph.
        // theta = 2.25 -> root = 1.5, interp(1.5) = 2.5 on segment [1,2];
        // dz/dtheta = (dz/dx) * (dx/dtheta) = 3 * 1/(2*1.5) = 1.0
        stan::math::recover_memory();
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

        check("composite / interp(solve) value", z.val(), 2.5, 1e-9);
        check("composite / knot y[1] adjoint", ys[1].adj(), 0.5, 1e-12);
        check("composite / knot y[2] adjoint", ys[2].adj(), 0.5, 1e-12);
        check("composite / dz/dtheta", theta.adj(), 1.0, 1e-9);

        // Passive-abscissa policy drops the x-path (dz/dtheta = 0)
        stan::math::recover_memory();
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
        check("composite / evaluateFixed dz/dtheta = 0", theta_fixed.adj(), 0.0, 1e-12);
    }
}

} // namespace

int main() {
    std::printf("=== var: implicit-function-theorem gradients ===\n");
    checkVarGradient<quantape::math::BisectionSolver<var>>("bisection", 1e-12);
    checkVarGradient<quantape::math::BrentSolver<var>>("brent", 1e-12);
    checkVarGradient<quantape::math::SecantSolver<var>>("secant", 1e-12);
    checkVarGradient<quantape::math::FalsePositionSolver<var>>("falsepos", 1e-12);
    checkVarGradient<quantape::math::RidderSolver<var>>("ridder", 1e-12);
    checkVarGradient<quantape::math::NewtonSolver<var>>("newton-fd", 1e-12);
    checkVarOnlyObjective();
    checkAutoBracket();
    checkNewtonWithDerivative();
    checkComposite();

    std::printf("=== fvar<...>: pathwise value/gradient/Hessian ===\n");
    checkHessian<quantape::math::BrentSolver>("brent", 1e-8);
    checkHessian<quantape::math::SecantSolver>("secant", 1e-8);
    checkHessian<quantape::math::FalsePositionSolver>("falsepos", 1e-8);
    checkHessian<quantape::math::RidderSolver>("ridder", 1e-8);
    checkHessian<quantape::math::NewtonSolver>("newton-fd", 1e-8);

    stan::math::recover_memory();

    if (failures == 0) {
        std::printf("\nAll solver AD tests passed.\n");
        return 0;
    }
    std::printf("\n%d solver AD test(s) FAILED.\n", failures);
    return 1;
}
