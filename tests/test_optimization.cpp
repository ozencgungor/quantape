// test_optimization.cpp — validates the optimizer building blocks
// (Math/Optimization): stop criteria, the Optimizer<DoubleT, Impl> CRTP base,
// and the AD evaluation helpers valueGrad / hvp / constraintValueJacobian.
//
//   1. stop criteria: relative/absolute f and x tests, dx, gradient,
//      evaluation budget (NLopt semantics)
//   2. valueGrad: exact gradients at var and fvar<var>
//   3. hvp: exact Hessian-vector products (no finite differences)
//   4. constraintValueJacobian: values + row-major Jacobian
//   5. CRTP base plumbing: a throwaway steepest-descent method drives the
//      base's evaluation helpers and stop logic; result codes and counters
//      are checked for each exit path
//
// Run: ./test_optimization
#include "Math/Optimization/AugLag.h"
#include "Math/Optimization/Constraint.h"
#include "Math/Optimization/LBFGS.h"
#include "Math/Optimization/OptimizerStanPrimitives.h"
#include "Math/Optimization/QpSolver.h"
#include "Math/Optimization/SLSQP.h"
#include "Math/Optimization/TNewton.h"
#include "Math/StanMath.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <utility>
#include <vector>

#define CHECK(cond)                                                                                \
    do {                                                                                           \
        if (!(cond)) {                                                                             \
            std::fprintf(stderr, "FAIL: %s (line %d)\n", #cond, __LINE__);                         \
            std::exit(1);                                                                          \
        }                                                                                          \
    } while (0)

using stan::math::fvar;
using stan::math::var;

namespace {

void checkClose(const char* label, double got, double expected, double tol) {
    if (std::fabs(got - expected) > tol) {
        std::fprintf(stderr, "FAIL: %s got=%.15g expected=%.15g err=%.3g\n", label, got, expected,
                     std::fabs(got - expected));
        std::exit(1);
    }
}

/// f(x) = sum_i w_i (x_i - c_i)^2, grad_i = 2 w_i (x_i - c_i), H = diag(2 w_i)
struct Quadratic {
    std::vector<double> w{1.0, 2.0, 0.5};
    std::vector<double> c{2.0, -3.0, 1.0};

    template <typename S>
    S operator()(const std::vector<S>& x) const {
        S sum = S(0.0);
        for (std::size_t i = 0; i < x.size(); ++i) {
            const S d = x[i] - S(c[i]);
            sum += S(w[i]) * d * d;
        }
        return sum;
    }

    double value(const std::vector<double>& x) const {
        double sum = 0.0;
        for (std::size_t i = 0; i < x.size(); ++i) {
            const double d = x[i] - c[i];
            sum += w[i] * d * d;
        }
        return sum;
    }

    std::vector<double> grad(const std::vector<double>& x) const {
        std::vector<double> g(x.size());
        for (std::size_t i = 0; i < x.size(); ++i) {
            g[i] = 2.0 * w[i] * (x[i] - c[i]);
        }
        return g;
    }
};

/// g0 = x0^2 + x1^2 - 1, g1 = x0 - x1
struct TwoConstraints {
    template <typename S>
    void operator()(const std::vector<S>& x, std::vector<S>& out) const {
        out.resize(2);
        out[0] = x[0] * x[0] + x[1] * x[1] - S(1.0);
        out[1] = x[0] - x[1];
    }
};

void testStopCriteria() {
    Math::StopCriteria criteria;

    // function value: relative
    criteria.ftol_rel = 1e-6;
    CHECK(Math::stopFtol(criteria, 1.0 + 1e-9, 1.0));
    CHECK(!Math::stopFtol(criteria, 1.1, 1.0));
    // function value: absolute
    criteria = {};
    criteria.ftol_abs = 1e-8;
    CHECK(Math::stopFtol(criteria, 1.0 + 1e-9, 1.0));
    CHECK(!Math::stopFtol(criteria, 1.0 + 1e-7, 1.0));
    // zero crossing
    criteria = {};
    criteria.ftol_rel = 1e-8;
    CHECK(Math::stopFtol(criteria, 0.0, 0.0));
    // non-finite old value never stops
    criteria = {};
    criteria.ftol_rel = 1e-8;
    // Non-finite old value never stops; pow avoids an infinity literal that
    // would warn under the release -ffast-math flags.
    const double huge = std::pow(10.0, 400.0);
    CHECK(!Math::stopFtol(criteria, 0.0, huge));

    // iterate: relative and absolute
    criteria = {};
    criteria.xtol_rel = 1e-6;
    const std::vector<double> x{1.0, 2.0};
    CHECK(Math::stopX(criteria, x, std::vector<double>{1.0 + 1e-9, 2.0}));
    CHECK(!Math::stopX(criteria, x, std::vector<double>{1.0 + 1e-3, 2.0}));
    criteria = {};
    criteria.xtol_abs = 1e-8;
    CHECK(Math::stopX(criteria, x, std::vector<double>{1.0 + 1e-9, 2.0 - 1e-9}));
    CHECK(!Math::stopX(criteria, x, std::vector<double>{1.0, 2.0 + 1e-7}));

    // step
    criteria = {};
    criteria.xtol_abs = 1e-8;
    CHECK(Math::stopDx(criteria, x, std::vector<double>{1e-9, -1e-9}));
    CHECK(!Math::stopDx(criteria, x, std::vector<double>{1e-9, 1e-7}));

    // gradient
    criteria = {};
    criteria.grad_tol = 1e-8;
    CHECK(Math::stopGrad(criteria, std::vector<double>{1e-9, -1e-9}));
    CHECK(!Math::stopGrad(criteria, std::vector<double>{1e-9, -1e-7}));
    criteria.grad_tol = 0.0;
    CHECK(!Math::stopGrad(criteria, std::vector<double>{0.0, 0.0}));

    // budgets
    criteria = {};
    criteria.maxeval = 5;
    CHECK(Math::stopEvals(criteria, 5));
    CHECK(!Math::stopEvals(criteria, 4));
    criteria = {};
    CHECK(!Math::stopTime(criteria, Math::nowSeconds()));
    criteria.maxtime = 1e-6;
    const double start = Math::nowSeconds() - 1.0;
    CHECK(Math::stopTime(criteria, start));
    CHECK(std::string(Math::to_string(Math::OptimizeResult::GradientTolReached)) ==
          "GradientTolReached");
}

void testValueGrad() {
    const Quadratic q;
    const std::vector<double> x0{0.5, -1.0, 2.0};

    {
        stan::math::recover_memory();
        auto result = Math::detail::valueGrad<var>(q, x0);
        checkClose("valueGrad<var> value", result.first, q.value(x0), 1e-12);
        const auto expected = q.grad(x0);
        for (std::size_t i = 0; i < x0.size(); ++i) {
            checkClose("valueGrad<var> grad", result.second[i], expected[i], 1e-12);
        }
    }
    {
        stan::math::recover_memory();
        auto result = Math::detail::valueGrad<fvar<var>>(q, x0);
        checkClose("valueGrad<fvar<var>> value", result.first, q.value(x0), 1e-12);
        const auto expected = q.grad(x0);
        for (std::size_t i = 0; i < x0.size(); ++i) {
            checkClose("valueGrad<fvar<var>> grad", result.second[i], expected[i], 1e-12);
        }
    }
}

void testHvp() {
    const Quadratic q;
    const std::vector<double> x0{0.5, -1.0, 2.0};
    const std::vector<double> v{1.0, -2.0, 0.5};

    stan::math::recover_memory();
    const auto hv = Math::detail::hvp(q, x0, v);
    for (std::size_t i = 0; i < x0.size(); ++i) {
        const double expected = 2.0 * q.w[i] * v[i]; // H = diag(2 w_i)
        checkClose("hvp", hv[i], expected, 1e-12);
    }

    // linearity: H(a v + b u) = a Hv + b Hu
    const std::vector<double> u{0.0, 1.0, -1.0};
    const auto hu = Math::detail::hvp(q, x0, u);
    for (std::size_t i = 0; i < x0.size(); ++i) {
        checkClose("hvp linearity", hv[i] - 0.5 * hu[i], 2.0 * q.w[i] * (v[i] - 0.5 * u[i]), 1e-12);
    }
}

void testConstraintJacobian() {
    const std::vector<double> x{0.6, 0.8};
    std::vector<double> c;
    std::vector<double> jacobian;

    {
        stan::math::recover_memory();
        Math::detail::constraintValueJacobian<var>(TwoConstraints{}, x, c, jacobian);
        CHECK(c.size() == 2 && jacobian.size() == 4);
        checkClose("constraint g0", c[0], 0.0, 1e-12);
        checkClose("constraint g1", c[1], -0.2, 1e-12);
        checkClose("constraint dg0/dx0", jacobian[0], 1.2, 1e-12);
        checkClose("constraint dg0/dx1", jacobian[1], 1.6, 1e-12);
        checkClose("constraint dg1/dx0", jacobian[2], 1.0, 1e-12);
        checkClose("constraint dg1/dx1", jacobian[3], -1.0, 1e-12);
    }
    {
        stan::math::recover_memory();
        Math::detail::constraintValueJacobian<fvar<var>>(TwoConstraints{}, x, c, jacobian);
        checkClose("constraint<fvar> dg0/dx0", jacobian[0], 1.2, 1e-12);
        checkClose("constraint<fvar> dg0/dx1", jacobian[1], 1.6, 1e-12);
        checkClose("constraint<fvar> dg1/dx1", jacobian[3], -1.0, 1e-12);
    }
}

double dot(const std::vector<double>& a, const std::vector<double>& b) {
    double sum = 0.0;
    for (std::size_t i = 0; i < a.size(); ++i) {
        sum += a[i] * b[i];
    }
    return sum;
}

void testLineSearch() {
    // f(z) = sum_i (z_i - 1)^2, minimum at (1, 1, 1)
    auto eval = [](const std::vector<double>& z, std::vector<double>& grad) {
        grad.resize(z.size());
        double f = 0.0;
        for (std::size_t i = 0; i < z.size(); ++i) {
            const double d = z[i] - 1.0;
            f += d * d;
            grad[i] = 2.0 * d;
        }
        return f;
    };

    const std::vector<double> x{2.0, 2.0, 2.0};
    const std::vector<double> g0{2.0, 2.0, 2.0};
    const double f0 = 3.0;
    const std::vector<double> d{-1.0, -1.0, -1.0}; // exact Newton direction

    const Math::LineSearchOptions options;
    const auto ls = Math::wolfeLineSearch(eval, x, d, f0, g0, options);
    CHECK(ls.success);
    checkClose("line search alpha", ls.alpha, 1.0, 1e-12);
    checkClose("line search f", ls.f, 0.0, 1e-24);
    CHECK(ls.grad.size() == x.size());

    // Strong-Wolfe conditions hold at the accepted point
    const double dphi0 = dot(g0, d);
    CHECK(ls.f <= f0 + options.c1 * ls.alpha * dphi0 + 1e-15);
    CHECK(std::fabs(dot(ls.grad, d)) <= -options.c2 * dphi0 + 1e-15);

    // Tight c2 forces the bracket+zoom path: exact step at alpha = 5
    {
        Math::LineSearchOptions tight = options;
        tight.c2 = 0.1;
        const std::vector<double> d_small{-0.2, -0.2, -0.2};
        const auto ls_zoom = Math::wolfeLineSearch(eval, x, d_small, f0, g0, tight);
        CHECK(ls_zoom.success);
        checkClose("line search zoom alpha", ls_zoom.alpha, 5.0, 1e-4);
        checkClose("line search zoom f", ls_zoom.f, 0.0, 1e-8);
    }
    // Oversized direction: first trial violates Armijo, zoom shrinks to 0.25
    {
        const std::vector<double> d_big{-4.0, -4.0, -4.0}; // exact alpha = 0.25
        const auto ls_shrink = Math::wolfeLineSearch(eval, x, d_big, f0, g0, options);
        CHECK(ls_shrink.success);
        checkClose("line search shrink alpha", ls_shrink.alpha, 0.25, 1e-12);
        checkClose("line search shrink f", ls_shrink.f, 0.0, 1e-24);
    }

    // Non-descent direction: no move, no success
    const std::vector<double> d_up{1.0, 1.0, 1.0};
    const auto bad = Math::wolfeLineSearch(eval, x, d_up, f0, g0, options);
    CHECK(!bad.success);
    CHECK(bad.alpha == 0.0);
}

struct Rosenbrock {
    template <typename S>
    S operator()(const std::vector<S>& x) const {
        const S t1 = x[1] - x[0] * x[0];
        const S t2 = S(1.0) - x[0];
        return S(100.0) * t1 * t1 + t2 * t2;
    }
};

struct Quadratic1D {
    template <typename S>
    S operator()(const std::vector<S>& x) const {
        const S d = x[0] - S(2.0);
        return d * d;
    }
};

struct WeightedQuadratic {
    std::vector<double> w{1.0, 1e2, 1e4};
    std::vector<double> c{1.0, -2.0, 0.5};

    template <typename S>
    S operator()(const std::vector<S>& x) const {
        S sum = S(0.0);
        for (std::size_t i = 0; i < x.size(); ++i) {
            const S d = x[i] - S(c[i]);
            sum += S(w[i]) * d * d;
        }
        return sum;
    }
};

void testLbfgs() {
    const Quadratic q;
    const std::vector<double> start{0.0, 0.0, 0.0};

    // 1-D quadratic: the first Wolfe step lands exactly on the minimum
    {
        Math::StopCriteria criteria;
        criteria.grad_tol = 1e-14;
        criteria.maxeval = 100;
        Math::LBFGS<var> optimizer(criteria, 1);
        std::vector<double> x{0.0};
        CHECK(optimizer.minimize(Quadratic1D{}, x) == Math::OptimizeResult::GradientTolReached);
        checkClose("lbfgs 1d x", x[0], 2.0, 1e-12);
    }
    // 3-D quadratic: fast convergence to the analytic minimizer
    {
        Math::StopCriteria criteria;
        criteria.grad_tol = 1e-12;
        criteria.maxeval = 1000;
        Math::LBFGS<var> optimizer(criteria);
        std::vector<double> x = start;
        Math::OptimizerState state;
        CHECK(optimizer.minimize(q, x, state) == Math::OptimizeResult::GradientTolReached);
        for (std::size_t i = 0; i < x.size(); ++i) {
            checkClose("lbfgs 3d x", x[i], q.c[i], 1e-8);
        }
        CHECK(state.iterations < 20);
        CHECK(state.evals > 0 && state.grad_evals > 0);
    }
    // Ill-conditioned weighted quadratic: memory earns its keep
    {
        Math::StopCriteria criteria;
        criteria.grad_tol = 1e-10;
        criteria.maxeval = 5000;
        Math::LBFGS<var> optimizer(criteria, 10);
        std::vector<double> x = start;
        CHECK(optimizer.minimize(WeightedQuadratic{}, x) ==
              Math::OptimizeResult::GradientTolReached);
        checkClose("lbfgs illcond x0", x[0], 1.0, 1e-6);
        checkClose("lbfgs illcond x1", x[1], -2.0, 1e-6);
        checkClose("lbfgs illcond x2", x[2], 0.5, 1e-6);
    }
    // Rosenbrock: classic curved-valley test at var
    {
        Math::StopCriteria criteria;
        criteria.grad_tol = 1e-8;
        criteria.maxeval = 2000;
        Math::LBFGS<var> optimizer(criteria, 10);
        std::vector<double> x{-1.2, 1.0};
        CHECK(optimizer.minimize(Rosenbrock{}, x) == Math::OptimizeResult::GradientTolReached);
        checkClose("rosenbrock x", x[0], 1.0, 1e-5);
        checkClose("rosenbrock y", x[1], 1.0, 1e-5);
    }
    // fvar<var> backend drives the same loop through the value reverse path
    {
        Math::StopCriteria criteria;
        criteria.grad_tol = 1e-10;
        criteria.maxeval = 1000;
        Math::LBFGS<fvar<var>> optimizer(criteria);
        std::vector<double> x = start;
        CHECK(optimizer.minimize(q, x) == Math::OptimizeResult::GradientTolReached);
        checkClose("lbfgs fvar x", x[0], q.c[0], 1e-6);
    }
    // maxeval exit
    {
        Math::StopCriteria criteria;
        criteria.grad_tol = 0.0;
        criteria.maxeval = 5;
        Math::LBFGS<var> optimizer(criteria);
        std::vector<double> x = start;
        CHECK(optimizer.minimize(q, x) == Math::OptimizeResult::MaxEvalReached);
    }
    // ftol exit (gradient stop disabled, one pass suffices)
    {
        Math::StopCriteria criteria;
        criteria.grad_tol = 0.0;
        criteria.ftol_abs = 1e-8;
        criteria.mtesf = 1;
        criteria.maxeval = 500;
        Math::LBFGS<var> optimizer(criteria);
        std::vector<double> x = start;
        CHECK(optimizer.minimize(q, x) == Math::OptimizeResult::FtolReached);
    }
    // xtol exit (gradient stop disabled, one pass suffices)
    {
        Math::StopCriteria criteria;
        criteria.grad_tol = 0.0;
        criteria.xtol_abs = 1e-4;
        criteria.mtesx = 1;
        criteria.maxeval = 500;
        Math::LBFGS<var> optimizer(criteria);
        std::vector<double> x = start;
        CHECK(optimizer.minimize(q, x) == Math::OptimizeResult::XtolReached);
    }
    // Small memory: ring buffer wraps (m = 2 < n = 3) and still converges
    {
        Math::StopCriteria criteria;
        criteria.grad_tol = 1e-10;
        criteria.maxeval = 1000;
        Math::LBFGS<var> optimizer(criteria, 2);
        WeightedQuadratic ill;
        std::vector<double> x = start;
        CHECK(optimizer.minimize(ill, x) == Math::OptimizeResult::GradientTolReached);
        checkClose("lbfgs ring x0", x[0], 1.0, 1e-6);
        checkClose("lbfgs ring x1", x[1], -2.0, 1e-6);
        checkClose("lbfgs ring x2", x[2], 0.5, 1e-6);
    }
    // memory inspector
    {
        Math::LBFGS<var> optimizer;
        CHECK(optimizer.memory() == 0);
        optimizer.setMemory(7);
        CHECK(optimizer.memory() == 7);
    }
}

/// Rosenbrock with analytic gradient in one call (NLopt-style objgrad)
struct RosenbrockValueGrad {
    double operator()(const std::vector<double>& x, std::vector<double>& grad) const {
        const double t1 = x[1] - x[0] * x[0];
        const double t2 = 1.0 - x[0];
        grad.resize(2);
        grad[0] = -400.0 * x[0] * t1 - 2.0 * t2;
        grad[1] = 200.0 * t1;
        return 100.0 * t1 * t1 + t2 * t2;
    }
};

/// Same gradient obtained through Stan inside the callback (caller-side AD)
struct RosenbrockValueGradAD {
    double operator()(const std::vector<double>& x, std::vector<double>& grad) const {
        stan::math::nested_rev_autodiff nested;
        std::vector<var> theta{var(x[0]), var(x[1])};
        const auto f = [](const std::vector<var>& z) {
            const var t1 = z[1] - z[0] * z[0];
            const var t2 = var(1.0) - z[0];
            return var(100.0) * t1 * t1 + t2 * t2;
        };
        var y = f(theta);
        y.grad();
        grad.resize(2);
        grad[0] = theta[0].adj();
        grad[1] = theta[1].adj();
        return y.val();
    }
};

void testLbfgsDoubleMode() {
    // Concept surface: both paired callbacks qualify; the scalar-generic
    // shape remains valid for the base entry point as well.
    static_assert(Math::ValueGradObjective<RosenbrockValueGrad>);
    static_assert(Math::ValueGradObjective<RosenbrockValueGradAD>);
    static_assert(Math::ObjectiveEvaluator<RosenbrockValueGrad, double>);
    static_assert(Math::ObjectiveEvaluator<RosenbrockValueGradAD, double>);
    static_assert(Math::ObjectiveEvaluator<Rosenbrock, double>);

    // Pure double with analytic gradient: no AD primitives involved
    {
        stan::math::recover_memory();
        Math::StopCriteria criteria;
        criteria.grad_tol = 1e-8;
        criteria.maxeval = 2000;
        Math::LBFGS<double> optimizer(criteria, 10);
        std::vector<double> x{-1.2, 1.0};
        Math::OptimizerState state;
        CHECK(optimizer.minimize(RosenbrockValueGrad{}, x, state) ==
              Math::OptimizeResult::GradientTolReached);
        checkClose("lbfgs<double> x", x[0], 1.0, 1e-5);
        checkClose("lbfgs<double> y", x[1], 1.0, 1e-5);
        CHECK(state.grad.size() == 2);
        CHECK(state.grad_evals > 0);
    }
    // Gradient supplied by caller-side Stan inside the callback
    {
        stan::math::recover_memory();
        Math::StopCriteria criteria;
        criteria.grad_tol = 1e-8;
        criteria.maxeval = 2000;
        Math::LBFGS<double> optimizer(criteria, 10);
        std::vector<double> x{-1.2, 1.0};
        CHECK(optimizer.minimize(RosenbrockValueGradAD{}, x) ==
              Math::OptimizeResult::GradientTolReached);
        checkClose("lbfgs<double+ad> x", x[0], 1.0, 1e-5);
        checkClose("lbfgs<double+ad> y", x[1], 1.0, 1e-5);
    }
}

/// Rosenbrock value only (for finite-difference gradient references)
double rosenbrockValue(const std::vector<double>& x) {
    const double t1 = x[1] - x[0] * x[0];
    const double t2 = 1.0 - x[0];
    return 100.0 * t1 * t1 + t2 * t2;
}

void testGradientSanity() {
    // The same point through three gradient sources must agree with central
    // finite differences of the double objective.
    const std::vector<double> x0{-1.2, 1.0};
    const double h = 1e-6;
    const auto fd_gradient = [&](std::size_t i) {
        auto xp = x0;
        auto xm = x0;
        xp[i] += h;
        xm[i] -= h;
        return (rosenbrockValue(xp) - rosenbrockValue(xm)) / (2.0 * h);
    };

    // 1) paired analytic callback (double mode)
    {
        std::vector<double> g;
        const double f = RosenbrockValueGrad{}(x0, g);
        checkClose("grad sanity value (analytic)", f, rosenbrockValue(x0), 1e-12);
        for (std::size_t i = 0; i < x0.size(); ++i) {
            const double fd = fd_gradient(i);
            checkClose("grad sanity double analytic", g[i], fd, 1e-5 * (1.0 + std::fabs(fd)));
        }
    }
    // 2) caller-side Stan callback (double mode)
    {
        std::vector<double> g;
        const double f = RosenbrockValueGradAD{}(x0, g);
        checkClose("grad sanity value (caller AD)", f, rosenbrockValue(x0), 1e-12);
        for (std::size_t i = 0; i < x0.size(); ++i) {
            const double fd = fd_gradient(i);
            checkClose("grad sanity double caller-AD", g[i], fd, 1e-5 * (1.0 + std::fabs(fd)));
        }
    }
    // 3) scalar-generic AD path (var backend) used by LBFGS<var>
    {
        stan::math::recover_memory();
        const auto f = [](const std::vector<var>& x) {
            const var t1 = x[1] - x[0] * x[0];
            const var t2 = var(1.0) - x[0];
            return var(100.0) * t1 * t1 + t2 * t2;
        };
        const auto result = Math::detail::valueGrad<var>(f, x0);
        checkClose("grad sanity value (var)", result.first, rosenbrockValue(x0), 1e-12);
        for (std::size_t i = 0; i < x0.size(); ++i) {
            const double fd = fd_gradient(i);
            checkClose("grad sanity var AD", result.second[i], fd, 1e-5 * (1.0 + std::fabs(fd)));
        }
    }
}

// ── Constrained optimization: QP solver, bounds, SLSQP ───────────────────

void testQpSolver() {
    // Unconstrained: min 0.5 d'Bd + g'd -> B d = -g
    {
        Math::QpProblem qp;
        qp.g = {-2.0, -4.0};
        qp.B = {2.0, 0.0, 0.0, 4.0};
        const auto res = Math::solveActiveSetQp(qp);
        CHECK(res.success);
        checkClose("qp unconstrained d0", res.d[0], 1.0, 1e-12);
        checkClose("qp unconstrained d1", res.d[1], 1.0, 1e-12);
    }
    // Equality: min 0.5||d||^2 + g'd s.t. d0 + d1 = 1
    {
        Math::QpProblem qp;
        qp.g = {-1.0, -1.0};
        qp.B = {1.0, 0.0, 0.0, 1.0};
        qp.A = {{1.0, 1.0}};
        qp.b = {-1.0};
        qp.equality = {1};
        const auto res = Math::solveActiveSetQp(qp);
        CHECK(res.success);
        checkClose("qp equality d0", res.d[0], 0.5, 1e-12);
        checkClose("qp equality d1", res.d[1], 0.5, 1e-12);
        checkClose("qp equality lambda", res.lambda[0], 0.5, 1e-12);
    }
    // Active inequality: min 0.5||d||^2 - 2 d0 s.t. d0 <= 1
    {
        Math::QpProblem qp;
        qp.g = {-2.0, 0.0};
        qp.B = {1.0, 0.0, 0.0, 1.0};
        qp.A = {{1.0, 0.0}};
        qp.b = {-1.0};
        qp.equality = {0};
        const auto res = Math::solveActiveSetQp(qp);
        CHECK(res.success);
        checkClose("qp active d0", res.d[0], 1.0, 1e-10);
        checkClose("qp active d1", res.d[1], 0.0, 1e-10);
        checkClose("qp active lambda", res.lambda[0], 1.0, 1e-10);
    }
    // Inactive inequality
    {
        Math::QpProblem qp;
        qp.g = {-0.5, 0.0};
        qp.B = {1.0, 0.0, 0.0, 1.0};
        qp.A = {{1.0, 0.0}};
        qp.b = {-1.0};
        qp.equality = {0};
        const auto res = Math::solveActiveSetQp(qp);
        CHECK(res.success);
        checkClose("qp inactive d0", res.d[0], 0.5, 1e-10);
        checkClose("qp inactive lambda", res.lambda[0], 0.0, 1e-10);
    }
}

void testConstraintHelpers() {
    const double inf = std::numeric_limits<double>::infinity();
    const Math::Bounds bounds = Math::Bounds::fromVectors({0.0, -inf}, {1.0, 5.0});
    CHECK(bounds.hasLower(0) && bounds.hasUpper(0));
    CHECK(!bounds.hasLower(1) && bounds.hasUpper(1));

    std::vector<double> x{2.0, 10.0};
    CHECK(!bounds.feasible(x));
    checkClose("bounds violation", bounds.violation(x), 5.0, 1e-15);
    bounds.project(x);
    checkClose("bounds project x0", x[0], 1.0, 1e-15);
    checkClose("bounds project x1", x[1], 5.0, 1e-15);
    CHECK(bounds.feasible(x));

    checkClose("max violation", Math::maxViolation({-1.0, 0.25, 0.0}), 0.25, 1e-15);
    checkClose("l1 penalty", Math::l1Penalty({0.5, -1.0, 0.0}, {2.0, -0.5}), 3.0, 1e-15);
}

// Test objectives / constraints for SLSQP
struct ShiftTarget {
    std::vector<double> target;
    template <typename S>
    S operator()(const std::vector<S>& x) const {
        S sum = S(0.0);
        for (std::size_t i = 0; i < x.size(); ++i) {
            const S d = x[i] - S(target[i]);
            sum += d * d;
        }
        return sum;
    }
};

struct ShiftTargetValueGrad {
    std::vector<double> target;
    double operator()(const std::vector<double>& x, std::vector<double>& grad) const {
        grad.resize(x.size());
        double f = 0.0;
        for (std::size_t i = 0; i < x.size(); ++i) {
            const double d = x[i] - target[i];
            f += d * d;
            grad[i] = 2.0 * d;
        }
        return f;
    }
};

struct SumEqual {
    template <typename S>
    void operator()(const std::vector<S>& x, std::vector<S>& h) const {
        h.resize(1);
        h[0] = x[0] + x[1] - S(2.0);
    }
};

struct SumLeq {
    template <typename S>
    void operator()(const std::vector<S>& x, std::vector<S>& g) const {
        g.resize(1);
        g[0] = x[0] + x[1] - S(2.0);
    }
};

struct SumLeqValueJac {
    void operator()(const std::vector<double>& x, std::vector<double>& c,
                    std::vector<double>& J) const {
        c.assign(1, x[0] + x[1] - 2.0);
        J.assign(2, 1.0); // 1 x 2 row-major
    }
};

struct TwoInfeasible {
    template <typename S>
    void operator()(const std::vector<S>& x, std::vector<S>& g) const {
        g.resize(2);
        g[0] = x[0];
        g[1] = S(1.0) - x[0];
    }
};

// Arbitrage-freeness: prices decreasing in strike and butterfly nonnegative
struct ArbConstraints {
    template <typename S>
    void operator()(const std::vector<S>& p, std::vector<S>& g) const {
        g.resize(3);
        g[0] = p[1] - p[0];
        g[1] = p[2] - p[1];
        g[2] = -(p[0] - S(2.0) * p[1] + p[2]);
    }
};

void testSlSqp() {
    Math::StopCriteria criteria;
    criteria.maxeval = 5000;

    // Equality-constrained quadratic: min ||x-(2,2)||^2 s.t. x0+x1=2 -> (1,1)
    {
        stan::math::recover_memory();
        Math::SLSQP<var> solver(criteria);
        std::vector<double> x{0.0, 0.0};
        const auto none = Math::Bounds::unbounded(x.size());
        Math::OptimizerState state;
        const auto result = solver.minimize(ShiftTarget{{2.0, 2.0}}, Math::NoConstraint{},
                                            SumEqual{}, none, x, state);
        CHECK(result == Math::OptimizeResult::Success);
        checkClose("slsqp eq x0", x[0], 1.0, 1e-7);
        checkClose("slsqp eq x1", x[1], 1.0, 1e-7);
        checkClose("slsqp eq f", state.f, 2.0, 1e-10);
    }
    // Active inequality: same objective s.t. x0+x1 <= 2 -> (1,1)
    {
        stan::math::recover_memory();
        Math::SLSQP<var> solver(criteria);
        std::vector<double> x{0.0, 0.0};
        const auto none = Math::Bounds::unbounded(x.size());
        const auto result = solver.minimize(ShiftTarget{{2.0, 2.0}}, SumLeq{}, none, x);
        CHECK(result == Math::OptimizeResult::Success);
        checkClose("slsqp active x0", x[0], 1.0, 1e-6);
        checkClose("slsqp active x1", x[1], 1.0, 1e-6);
    }
    // Inactive inequality: target already strictly feasible -> unconstrained min
    {
        stan::math::recover_memory();
        Math::SLSQP<var> solver(criteria);
        std::vector<double> x{-1.0, -1.0};
        const auto none = Math::Bounds::unbounded(x.size());
        const auto result = solver.minimize(ShiftTarget{{0.5, 0.5}}, SumLeq{}, none, x);
        CHECK(result == Math::OptimizeResult::Success);
        checkClose("slsqp inactive x0", x[0], 0.5, 1e-7);
        checkClose("slsqp inactive x1", x[1], 0.5, 1e-7);
    }
    // Bounds: min (x0-3)^2 s.t. x0 <= 1 -> x0 = 1
    {
        stan::math::recover_memory();
        Math::SLSQP<var> solver(criteria);
        std::vector<double> x{0.0};
        const auto bounds = Math::Bounds::fromVectors({Math::Bounds::kNoLower}, {1.0});
        const auto result = solver.minimize(ShiftTarget{{3.0}}, Math::NoConstraint{}, bounds, x);
        CHECK(result == Math::OptimizeResult::Success);
        checkClose("slsqp bound x0", x[0], 1.0, 1e-7);
    }
    // Arbitrage-style: violated market quotes -> feasible monotone butterfly-free fit
    {
        stan::math::recover_memory();
        const std::vector<double> market{1.0, 1.5, 1.0}; // violates monotonicity and butterfly
        Math::SLSQP<var> solver(criteria);
        std::vector<double> p = market;
        const auto none = Math::Bounds::unbounded(p.size());
        Math::OptimizerState state;
        const auto result = solver.minimize(ShiftTarget{market}, ArbConstraints{}, none, p, state);
        CHECK(result == Math::OptimizeResult::Success);
        CHECK(p[0] >= p[1] - 1e-7);
        CHECK(p[1] >= p[2] - 1e-7);
        CHECK(p[0] - 2.0 * p[1] + p[2] >= -1e-7);
        double fit = 0.0;
        for (std::size_t i = 0; i < p.size(); ++i) {
            const double d = p[i] - market[i];
            fit += d * d;
        }
        CHECK(fit <= 0.25 + 1e-9); // better than the equal-price feasible point (1,1,1)
    }
    // Infeasible system: x0 <= 0 and x0 >= 1
    {
        stan::math::recover_memory();
        Math::SLSQP<var> solver(criteria);
        std::vector<double> x{0.5};
        const auto none = Math::Bounds::unbounded(x.size());
        const auto result = solver.minimize(ShiftTarget{{0.0}}, TwoInfeasible{}, none, x);
        CHECK(result == Math::OptimizeResult::Infeasible);
    }
    // Double mode with paired objective/constraint callbacks (no AD)
    {
        Math::SLSQP<double> solver(criteria);
        std::vector<double> x{0.0, 0.0};
        const auto none = Math::Bounds::unbounded(x.size());
        const auto result =
            solver.minimize(ShiftTargetValueGrad{{2.0, 2.0}}, SumLeqValueJac{}, none, x);
        CHECK(result == Math::OptimizeResult::Success);
        checkClose("slsqp<double> x0", x[0], 1.0, 1e-6);
        checkClose("slsqp<double> x1", x[1], 1.0, 1e-6);
    }
}

void testAugLag() {
    Math::StopCriteria criteria;
    criteria.maxeval = 100000;

    // Equality-constrained quadratic: min ||x-(2,2)||^2 s.t. x0+x1=2 -> (1,1)
    {
        stan::math::recover_memory();
        Math::AugLag<var> solver(criteria);
        std::vector<double> x{0.0, 0.0};
        const auto none = Math::Bounds::unbounded(x.size());
        const auto result =
            solver.minimize(ShiftTarget{{2.0, 2.0}}, Math::NoConstraint{}, SumEqual{}, none, x);
        CHECK(result == Math::OptimizeResult::Success);
        checkClose("auglag eq x0", x[0], 1.0, 1e-5);
        checkClose("auglag eq x1", x[1], 1.0, 1e-5);
    }
    // Active inequality -> (1,1)
    {
        stan::math::recover_memory();
        Math::AugLag<var> solver(criteria);
        std::vector<double> x{0.0, 0.0};
        const auto none = Math::Bounds::unbounded(x.size());
        const auto result = solver.minimize(ShiftTarget{{2.0, 2.0}}, SumLeq{}, none, x);
        CHECK(result == Math::OptimizeResult::Success);
        checkClose("auglag active x0", x[0], 1.0, 1e-5);
        checkClose("auglag active x1", x[1], 1.0, 1e-5);
    }
    // Inactive inequality -> unconstrained target
    {
        stan::math::recover_memory();
        Math::AugLag<var> solver(criteria);
        std::vector<double> x{-1.0, -1.0};
        const auto none = Math::Bounds::unbounded(x.size());
        const auto result = solver.minimize(ShiftTarget{{0.5, 0.5}}, SumLeq{}, none, x);
        CHECK(result == Math::OptimizeResult::Success);
        checkClose("auglag inactive x0", x[0], 0.5, 1e-5);
        checkClose("auglag inactive x1", x[1], 0.5, 1e-5);
    }
    // Bounds (penalized, projected-gradient KKT): min (x0-3)^2 s.t. x0 <= 1
    {
        stan::math::recover_memory();
        Math::AugLag<var> solver(criteria);
        std::vector<double> x{0.0};
        const auto bounds = Math::Bounds::fromVectors({Math::Bounds::kNoLower}, {1.0});
        const auto result = solver.minimize(ShiftTarget{{3.0}}, Math::NoConstraint{}, bounds, x);
        CHECK(result == Math::OptimizeResult::Success);
        checkClose("auglag bound x0", x[0], 1.0, 1e-5);
    }
    // Double mode with paired callbacks
    {
        Math::AugLag<double> solver(criteria);
        std::vector<double> x{0.0, 0.0};
        const auto none = Math::Bounds::unbounded(x.size());
        const auto result =
            solver.minimize(ShiftTargetValueGrad{{2.0, 2.0}}, SumLeqValueJac{}, none, x);
        CHECK(result == Math::OptimizeResult::Success);
        checkClose("auglag<double> x0", x[0], 1.0, 1e-5);
        checkClose("auglag<double> x1", x[1], 1.0, 1e-5);
    }
    // Agreement with SLSQP on the same constrained problem
    {
        stan::math::recover_memory();
        Math::StopCriteria sc;
        sc.maxeval = 100000;
        Math::SLSQP<var> slsqp(sc);
        Math::AugLag<var> auglag(sc);
        std::vector<double> x_slsqp{0.0, 0.0};
        std::vector<double> x_auglag{0.0, 0.0};
        const auto none = Math::Bounds::unbounded(2);
        CHECK(slsqp.minimize(ShiftTarget{{2.0, 2.0}}, SumLeq{}, none, x_slsqp) ==
              Math::OptimizeResult::Success);
        CHECK(auglag.minimize(ShiftTarget{{2.0, 2.0}}, SumLeq{}, none, x_auglag) ==
              Math::OptimizeResult::Success);
        for (std::size_t i = 0; i < x_slsqp.size(); ++i) {
            checkClose("slsqp vs auglag", x_auglag[i], x_slsqp[i], 1e-4);
        }
    }
}

struct Himmelblau {
    template <typename S>
    S operator()(const std::vector<S>& x) const {
        const S a = x[0] * x[0] + x[1] - S(11.0);
        const S b = x[0] + x[1] * x[1] - S(7.0);
        return a * a + b * b;
    }
};

// f = x0^4 - x0^2 + x1^2: at (0.2, 0) the Hessian is [[-1.52, 0], [0, 2]]
// (negative curvature along e0) while the gradient (-0.368, 0) is nonzero.
struct QuarticNonconvex {
    template <typename S>
    S operator()(const std::vector<S>& x) const {
        const S x0 = x[0];
        return x0 * x0 * x0 * x0 - x0 * x0 + x[1] * x[1];
    }
};

void testTNewton() {
    Math::StopCriteria criteria;
    criteria.maxeval = 100000;

    // 2-D Rosenbrock at var
    {
        stan::math::recover_memory();
        Math::TNewton<var> solver(criteria);
        std::vector<double> x{-1.2, 1.0};
        Math::OptimizerState state;
        CHECK(solver.minimize(Rosenbrock{}, x, state) == Math::OptimizeResult::GradientTolReached);
        checkClose("tnewton rosenbrock x0", x[0], 1.0, 1e-8);
        checkClose("tnewton rosenbrock x1", x[1], 1.0, 1e-8);
        std::printf("  tnewton rosenbrock2: iters=%zu evals=%zu\n", state.iterations, state.evals);
    }
    // 3-D quadratic: exact Newton directions, fast convergence
    {
        stan::math::recover_memory();
        const Quadratic q;
        Math::TNewton<var> solver(criteria);
        std::vector<double> x{0.0, 0.0, 0.0};
        Math::OptimizerState state;
        CHECK(solver.minimize(q, x, state) == Math::OptimizeResult::GradientTolReached);
        for (std::size_t i = 0; i < x.size(); ++i) {
            checkClose("tnewton quadratic x", x[i], q.c[i], 1e-9);
        }
        std::printf("  tnewton quadratic3: iters=%zu evals=%zu\n", state.iterations, state.evals);
    }
    // fvar<var> backend drives the same path
    {
        stan::math::recover_memory();
        Math::TNewton<fvar<var>> solver(criteria);
        std::vector<double> x{-1.2, 1.0};
        CHECK(solver.minimize(Rosenbrock{}, x) == Math::OptimizeResult::GradientTolReached);
        checkClose("tnewton fvar x0", x[0], 1.0, 1e-8);
        checkClose("tnewton fvar x1", x[1], 1.0, 1e-8);
    }
    // Exact Newton converges quadratically on a well-scaled quadratic and
    // on Himmelblau (Hessian 2I at the start)
    {
        stan::math::recover_memory();
        Math::TNewton<var> solver(criteria);
        std::vector<double> x{0.0, 0.0};
        const auto r = solver.minimize(Himmelblau{}, x);
        CHECK(r == Math::OptimizeResult::GradientTolReached);
        const double f = Himmelblau{}(x);
        CHECK(f < 1e-12);
        std::printf("  tnewton himmelblau(0,0): x=(%.4f, %.4f) f=%.2e\n", x[0], x[1], f);
    }
    // maxeval exit
    {
        stan::math::recover_memory();
        Math::StopCriteria tight;
        tight.grad_tol = 0.0;
        tight.maxeval = 5;
        Math::TNewton<var> solver(tight);
        std::vector<double> x{-1.2, 1.0};
        CHECK(solver.minimize(Rosenbrock{}, x) == Math::OptimizeResult::MaxEvalReached);
    }
    // Negative curvature on the first CG step: BOTH modes fall back to
    // steepest descent (PNET iterd = 0) and converge to a quartic minimum
    // (NLopt parity: plain TNEWTON and TNEWTON_RESTART both succeed here)
    {
        stan::math::recover_memory();
        Math::TNewton<var> plain(criteria, /*restart=*/false);
        std::vector<double> x0{0.2, 0.0};
        CHECK(plain.minimize(QuarticNonconvex{}, x0) == Math::OptimizeResult::GradientTolReached);
        checkClose("tnewton quartic x0", std::fabs(x0[0]), 0.70710678, 1e-6);
        checkClose("tnewton quartic x1", x0[1], 0.0, 1e-6);

        stan::math::recover_memory();
        Math::TNewton<var> restarting(criteria, /*restart=*/true);
        std::vector<double> x1{0.2, 0.0};
        CHECK(restarting.minimize(QuarticNonconvex{}, x1) ==
              Math::OptimizeResult::GradientTolReached);
        checkClose("tnewton quartic restart x0", std::fabs(x1[0]), 0.70710678, 1e-6);
        std::printf("  tnewton quartic(0.2,0): plain and restart both converge\n");
    }
}

// ── CRTP smoke test: a throwaway steepest-descent method ──────────────────
template <typename DoubleT>
class SteepestDescent : public Math::Optimizer<DoubleT, SteepestDescent<DoubleT>> {
public:
    using Base = Math::Optimizer<DoubleT, SteepestDescent<DoubleT>>;

    explicit SteepestDescent(Math::StopCriteria criteria = {}, double step = 0.1)
        : Base(std::move(criteria)), m_step(step) {}

    // internal, public for CRTP access (like Solver1D's solveImpl)
    template <typename F>
        requires Math::VectorObjective<F, DoubleT>
    Math::OptimizeResult minimizeImpl(const F& f, Math::OptimizerState& state) const {
        this->updateValueGrad(f, state);
        while (true) {
            if (this->stopByGradient(state.grad)) {
                state.message = "gradient tolerance";
                return Math::OptimizeResult::GradientTolReached;
            }
            if (this->stopByEvalOrTime(state)) {
                return Math::stopTime(this->criteria(), state.start_time)
                           ? Math::OptimizeResult::MaxTimeReached
                           : Math::OptimizeResult::MaxEvalReached;
            }
            const double old_f = state.f;
            const std::vector<double> old_x = state.x;
            for (std::size_t i = 0; i < state.x.size(); ++i) {
                state.x[i] -= m_step * state.grad[i];
            }
            this->updateValueGrad(f, state);
            ++state.iterations;

            if (state.f <= this->criteria().stopval) {
                return Math::OptimizeResult::StopvalReached;
            }
            if (Math::stopFtol(this->criteria(), state.f, old_f)) {
                return Math::OptimizeResult::FtolReached;
            }
            if (Math::stopX(this->criteria(), state.x, old_x)) {
                return Math::OptimizeResult::XtolReached;
            }
        }
    }

private:
    double m_step;
};

// Minimal value-only method: exercises the double backend of the base
template <typename DoubleT>
class ValueProbe : public Math::Optimizer<DoubleT, ValueProbe<DoubleT>> {
public:
    using Base = Math::Optimizer<DoubleT, ValueProbe<DoubleT>>;
    using Base::Base;

    template <typename F>
        requires Math::VectorObjective<F, DoubleT>
    Math::OptimizeResult minimizeImpl(const F& f, Math::OptimizerState& state) const {
        this->updateValueGrad(f, state);
        return Math::OptimizeResult::Success;
    }
};

void testBasePlumbing() {
    const Quadratic q;
    const std::vector<double> start{0.0, 0.0, 0.0};

    // double backend: value only, no gradient
    {
        ValueProbe<double> probe;
        std::vector<double> x = start;
        Math::OptimizerState state;
        CHECK(probe.minimize(q, x, state) == Math::OptimizeResult::Success);
        checkClose("double backend value", state.f, q.value(start), 1e-14);
        CHECK(state.grad.empty());
        CHECK(state.evals == 1);
        CHECK(x == start);
    }
    // var backend: full convergence to the analytic minimizer
    {
        Math::StopCriteria criteria;
        criteria.grad_tol = 1e-12;
        criteria.maxeval = 100000;
        SteepestDescent<var> optimizer(criteria, 0.1);
        std::vector<double> x = start;
        Math::OptimizerState state;
        CHECK(optimizer.minimize(q, x, state) == Math::OptimizeResult::GradientTolReached);
        CHECK(state.evals > 0 && state.grad_evals > 0 && state.iterations > 0);
        for (std::size_t i = 0; i < x.size(); ++i) {
            checkClose("var backend x", x[i], q.c[i], 1e-6);
        }
    }
    // fvar backend drives the same path through val_ reverse
    {
        Math::StopCriteria criteria;
        criteria.grad_tol = 1e-10;
        criteria.maxeval = 100000;
        SteepestDescent<fvar<var>> optimizer(criteria, 0.1);
        std::vector<double> x = start;
        CHECK(optimizer.minimize(q, x) == Math::OptimizeResult::GradientTolReached);
        for (std::size_t i = 0; i < x.size(); ++i) {
            checkClose("fvar backend x", x[i], q.c[i], 1e-5);
        }
    }
    // maxeval exit
    {
        Math::StopCriteria criteria;
        criteria.grad_tol = 0.0;
        criteria.maxeval = 5;
        SteepestDescent<var> optimizer(criteria, 0.1);
        std::vector<double> x = start;
        CHECK(optimizer.minimize(q, x) == Math::OptimizeResult::MaxEvalReached);
    }
    // stopval exit
    {
        Math::StopCriteria criteria;
        criteria.grad_tol = 0.0;
        criteria.maxeval = 100000;
        criteria.stopval = 1e-6;
        SteepestDescent<var> optimizer(criteria, 0.1);
        std::vector<double> x = start;
        CHECK(optimizer.minimize(q, x) == Math::OptimizeResult::StopvalReached);
    }
    // xtol exit
    {
        Math::StopCriteria criteria;
        criteria.grad_tol = 0.0;
        criteria.xtol_abs = 1e-3;
        criteria.maxeval = 100000;
        SteepestDescent<var> optimizer(criteria, 0.1);
        std::vector<double> x = start;
        CHECK(optimizer.minimize(q, x) == Math::OptimizeResult::XtolReached);
    }
    // empty parameter vector rejected
    {
        SteepestDescent<var> optimizer;
        std::vector<double> empty;
        bool threw = false;
        try {
            optimizer.minimize(q, empty);
        } catch (const std::invalid_argument&) {
            threw = true;
        }
        CHECK(threw);
    }
}

} // namespace

int main() {
    testStopCriteria();
    testValueGrad();
    testHvp();
    testConstraintJacobian();
    testLineSearch();
    testLbfgs();
    testLbfgsDoubleMode();
    testGradientSanity();
    testQpSolver();
    testConstraintHelpers();
    testSlSqp();
    testAugLag();
    testTNewton();
    testBasePlumbing();
    stan::math::recover_memory();
    std::printf("test_optimization: all invariants hold\n");
    return 0;
}
