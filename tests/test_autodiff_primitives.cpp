// test_autodiff_primitives.cpp — Phase 3: Hessian-vector products and
// cross-primitive AD composition.
//
//   1. hvp: forward-over-reverse Hessian-vector product vs analytic f''(x)v
//   2. solve(interp): a solve whose objective evaluates an interpolant built
//      from solved knots — gradients chain through both primitives, the
//      interpolation weights AND the evaluation point
//   3. integrate(interp): d/dtheta of an integral over an interpolated curve
//      equals the quadrature/trapezoid weights on the knots
//
// Run: ./test_autodiff_primitives
#include "quantape/math/Autodiff/Hvp.h"
#include "quantape/math/Integrals/IntegratorStanPrimitives.h"
#include "quantape/math/Interpolations/InterpolationStanPrimitives.h"
#include "quantape/math/NumericalMethods.h"
#include "quantape/math/Solvers/SolverStanPrimitives.h"
#include "quantape/math/StanMath.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

#define CHECK(cond)                                                                                \
    do {                                                                                           \
        if (!(cond)) {                                                                             \
            std::fprintf(stderr, "FAIL: %s (line %d)\n", #cond, __LINE__);                         \
            std::exit(1);                                                                          \
        }                                                                                          \
    } while (0)

using stan::math::var;

namespace {

void checkClose(const char* label, double got, double expected, double tol) {
    if (std::fabs(got - expected) > tol) {
        std::fprintf(stderr, "FAIL: %s got=%.15g expected=%.15g err=%.3g\n", label, got, expected,
                     std::fabs(got - expected));
        std::exit(1);
    }
    std::printf("  %-46s ok  got=%.12f expected=%.12f\n", label, got, expected);
}

void testHvp() {
    std::printf("=== Hessian-vector product ===\n");
    stan::math::recover_memory();

    auto f = [](const auto& x) { return x * x * x * x - 3.0 * x * x + 2.0 * x; };
    const double x = 0.7, v = 2.3;
    const double analytic = (12.0 * x * x - 6.0) * v;
    checkClose("hvp f''(x)*v", quantape::math::hvp(f, x, v), analytic, 1e-10);

    // matches the Hessian-vector product from a dense Hessian
    double fx = 0.0;
    Eigen::VectorXd grad;
    Eigen::Matrix<double, -1, -1> H;
    Eigen::VectorXd xv(1);
    xv << x;
    stan::math::hessian(
        [&](const auto& xx) {
            using S = typename std::decay_t<decltype(xx)>::Scalar;
            return f(xx(0));
        },
        xv, fx, grad, H);
    checkClose("hvp vs dense Hessian", quantape::math::hvp(f, x, v), H(0, 0) * v, 1e-9);
}

/// Double-precision twin of the composite pipeline for FD cross-checks.
const std::vector<double> kGrid{0.0, 1.0, 2.0, 3.0};

double fdSolveInterpPipeline(const std::vector<double>& theta, double h, int perturb) {
    std::vector<double> knots(theta.size());
    for (size_t i = 0; i < theta.size(); ++i) {
        knots[i] = std::sqrt(theta[i] + (static_cast<int>(i) == perturb ? h : 0.0));
    }
    const quantape::math::LinearInterpolation<double> interp(kGrid, knots);
    quantape::math::BisectionSolver<double> solver;
    auto f = [&](double x) { return interp(x) - 2.5; };
    return solver.solve(f, 1e-12, 1.5, 1.0, 2.0);
}

void testSolveInterp() {
    std::printf("=== solve(interp): objective evaluates an interpolant ===\n");
    stan::math::recover_memory();

    // knots k_i = sqrt(theta_i), theta = {1,4,9,16} -> knots {1,2,3,4};
    // interp(1.5) = 2.5 on segment [1,2] with slope 1
    std::vector<var> theta{1.0, 4.0, 9.0, 16.0};
    std::vector<var> knots(4);
    for (size_t i = 0; i < 4; ++i)
        knots[i] = stan::math::sqrt(theta[i]);

    auto f = [&](const var& x) {
        const quantape::math::LinearInterpolation<var> interp(kGrid, knots);
        return interp(x) - 2.5;
    };

    quantape::math::BrentSolver<var> solver;
    solver.setMaxEvaluations(200);
    var root = solver.solve(f, 1e-12, var(1.5), var(1.0), var(2.0));
    root.grad();

    checkClose("solve(interp) value", root.val(), 1.5, 1e-9);
    // x* = 1.5 sits on segment [1,2] (knots k1 = 2 at x=1, k2 = 3 at x=2).
    // dI/dx = 1, w1 = w2 = 0.5, dk_i/dtheta_i = 1/(2 sqrt(theta_i))
    checkClose("d root/d theta0 (knot 0 off segment)", theta[0].adj(), 0.0, 1e-12);
    checkClose("d root/d theta1", theta[1].adj(), -0.5 * 0.25, 1e-9);
    checkClose("d root/d theta2", theta[2].adj(), -0.5 / 6.0, 1e-9);
    checkClose("d root/d theta3 (knot 3 off segment)", theta[3].adj(), 0.0, 1e-12);

    // FD cross-check of the full pipeline
    for (int i = 0; i < 4; ++i) {
        const double h = 1e-6;
        const double xp = fdSolveInterpPipeline({1.0, 4.0, 9.0, 16.0}, h, i);
        const double xm = fdSolveInterpPipeline({1.0, 4.0, 9.0, 16.0}, -h, i);
        char label[64];
        std::snprintf(label, sizeof(label), "solve(interp) FD d root/d theta%d", i);
        checkClose(label, theta[static_cast<size_t>(i)].adj(), (xp - xm) / (2.0 * h), 1e-5);
    }
}

void testIntegrateInterp() {
    std::printf("=== integrate(interp): integral over an interpolated curve ===\n");
    stan::math::recover_memory();

    std::vector<var> knots{1.0, 2.0, 3.0, 4.0};

    // integral of the piecewise-linear interpolant over [0,3] = sum of
    // trapezoids = 0.5 k0 + k1 + k2 + 0.5 k3
    var total = 0.0;
    for (size_t i = 0; i + 1 < kGrid.size(); ++i) {
        quantape::math::TrapezoidIntegratorDefault<var> integ(1e-12, 100);
        total += integ(
            [&](const var& x) {
                const quantape::math::LinearInterpolation<var> interp(kGrid, knots);
                return interp(x);
            },
            var(kGrid[i]), var(kGrid[i + 1]));
    }
    total.grad();

    const double weights[4] = {0.5, 1.0, 1.0, 0.5};
    checkClose("integrate(interp) value", total.val(), 0.5 * 1 + 2 + 3 + 0.5 * 4, 1e-9);
    for (size_t i = 0; i < 4; ++i) {
        char label[64];
        std::snprintf(label, sizeof(label), "d integral/d knot%d", i);
        checkClose(label, knots[i].adj(), weights[i], 1e-9);
    }
}

void testNestedSolve() {
    std::printf("=== nested solve: outer objective calls an inner solve ===\n");
    stan::math::recover_memory();

    // inner: y = sqrt(theta); outer: x solves x^2 = y -> x = theta^{1/4}
    // dx/dtheta = 0.25 * theta^{-3/4}; at theta = 16: x = 2, dx/dtheta = 1/32
    var theta = 16.0;

    auto inner = [&](const var& z) { return z * z - theta; };
    quantape::math::BrentSolver<var> innerSolver;
    innerSolver.setMaxEvaluations(200);
    var y = innerSolver.solve(inner, 1e-12, var(1.0), var(0.0), var(10.0));

    auto outer = [&](const var& x) { return x * x - y; };
    quantape::math::BrentSolver<var> outerSolver;
    outerSolver.setMaxEvaluations(200);
    var x = outerSolver.solve(outer, 1e-12, var(1.0), var(0.0), var(5.0));
    x.grad();

    checkClose("nested solve value", x.val(), 2.0, 1e-9);
    checkClose("nested solve dx/dtheta", theta.adj(), 0.03125, 1e-9);
}

} // namespace

int main() {
    testHvp();
    testSolveInterp();
    testIntegrateInterp();
    testNestedSolve();
    stan::math::recover_memory();
    std::printf("\ntest_autodiff_primitives: all invariants hold\n");
    return 0;
}
