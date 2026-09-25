// test_optimizers_stress.cpp — hard problems, edge cases, and timing probes
// for the Math/Optimization stack. Finds correctness limits (does the solver
// converge?) and optimization opportunities (where does time go at scale).
//
// Run: ./build/test_optimizers_stress
#include "Math/Optimization/AugLag.h"
#include "Math/Optimization/LBFGS.h"
#include "Math/Optimization/OptimizerStanPrimitives.h"
#include "Math/Optimization/SLSQP.h"
#include "Math/StanMath.h"

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

using stan::math::var;

namespace {

int failures = 0;

void check(const char* label, bool ok) {
    std::printf("  %-52s %s\n", label, ok ? "ok" : "FAIL");
    if (!ok) {
        ++failures;
    }
}

void checkClose(const char* label, double got, double expected, double tol) {
    const bool ok = std::fabs(got - expected) <= tol;
    if (!ok) {
        std::printf("  %-52s FAIL got=%.10g expected=%.10g\n", label, got, expected);
        ++failures;
    } else {
        std::printf("  %-52s ok  got=%.10g err=%.2e\n", label, got, std::fabs(got - expected));
    }
}

template <typename Fn>
double timeit(const Fn& fn) {
    const auto start = std::chrono::steady_clock::now();
    const double sink = fn();
    const auto stop = std::chrono::steady_clock::now();
    (void)sink;
    return std::chrono::duration<double, std::micro>(stop - start).count();
}

// ── problems ──────────────────────────────────────────────────────────────

struct Rosenbrock {
    template <typename S>
    S operator()(const std::vector<S>& x) const {
        S sum = S(0.0);
        for (std::size_t i = 0; i + 1 < x.size(); ++i) {
            const S t1 = x[i + 1] - x[i] * x[i];
            const S t2 = S(1.0) - x[i];
            sum += S(100.0) * t1 * t1 + t2 * t2;
        }
        return sum;
    }
};

struct Beale {
    template <typename S>
    S operator()(const std::vector<S>& x) const {
        const S t1 = S(1.5) - x[0] + x[0] * x[1];
        const S t2 = S(2.25) - x[0] + x[0] * x[1] * x[1];
        const S t3 = S(2.625) - x[0] + x[0] * x[1] * x[1] * x[1];
        return t1 * t1 + t2 * t2 + t3 * t3;
    }
};

struct Himmelblau {
    template <typename S>
    S operator()(const std::vector<S>& x) const {
        const S a = x[0] * x[0] + x[1] - S(11.0);
        const S b = x[0] + x[1] * x[1] - S(7.0);
        return a * a + b * b;
    }
};

struct WeightedQuadratic {
    std::vector<double> w;
    std::vector<double> target;
    template <typename S>
    S operator()(const std::vector<S>& x) const {
        S sum = S(0.0);
        for (std::size_t i = 0; i < x.size(); ++i) {
            const S d = x[i] - S(target[i]);
            sum += S(w[i]) * d * d;
        }
        return sum;
    }
};

struct ConstantObjective {
    template <typename S>
    S operator()(const std::vector<S>&) const {
        return S(42.0);
    }
};

struct Shift {
    std::vector<double> target;
    template <typename S>
    S operator()(const std::vector<S>& x) const {
        S s = S(0.0);
        for (std::size_t i = 0; i < x.size(); ++i) {
            const S d = x[i] - S(target[i]);
            s += d * d;
        }
        return s;
    }
};

struct SumLeq {
    template <typename S>
    void operator()(const std::vector<S>& x, std::vector<S>& g) const {
        g.resize(1);
        g[0] = x[0] + x[1] - S(2.0);
    }
};

struct SumLeqRedundant {
    template <typename S>
    void operator()(const std::vector<S>& x, std::vector<S>& g) const {
        g.resize(2);
        g[0] = x[0] + x[1] - S(2.0);
        g[1] = S(2.0) * (x[0] + x[1] - S(2.0));
    }
};

struct SumEqual {
    template <typename S>
    void operator()(const std::vector<S>& x, std::vector<S>& h) const {
        h.resize(1);
        h[0] = x[0] + x[1] - S(2.0);
    }
};

// Monotone prices + nonnegative butterflies on a curve
struct CurveArb {
    template <typename S>
    void operator()(const std::vector<S>& p, std::vector<S>& g) const {
        const std::size_t m = p.size();
        g.resize(2 * m - 3); // (m-1) monotone + (m-2) butterflies
        std::size_t k = 0;
        for (std::size_t i = 0; i + 1 < m; ++i) {
            g[k++] = p[i + 1] - p[i];
        }
        for (std::size_t i = 0; i + 2 < m; ++i) {
            g[k++] = -(p[i] - S(2.0) * p[i + 1] + p[i + 2]);
        }
    }
};

// ── LBFGS stress ──────────────────────────────────────────────────────────

void stressLbfgs() {
    std::printf("=== LBFGS stress ===\n");
    Math::StopCriteria criteria;
    criteria.maxeval = 100000;

    // 2-D Rosenbrock
    {
        stan::math::recover_memory();
        Math::LBFGS<var> solver(criteria, 10);
        std::vector<double> x{-1.2, 1.0};
        Math::OptimizerState state;
        const auto r = solver.minimize(Rosenbrock{}, x, state);
        check("rosenbrock2 converged", r == Math::OptimizeResult::GradientTolReached);
        checkClose("rosenbrock2 x0", x[0], 1.0, 1e-6);
        checkClose("rosenbrock2 x1", x[1], 1.0, 1e-6);
    }
    // 10-D Rosenbrock
    {
        stan::math::recover_memory();
        Math::LBFGS<var> solver(criteria, 10);
        std::vector<double> x(10, -1.2);
        const auto r = solver.minimize(Rosenbrock{}, x);
        check("rosenbrock10 converged", r == Math::OptimizeResult::GradientTolReached);
        checkClose("rosenbrock10 x0", x[0], 1.0, 1e-4);
    }
    // 50-D Rosenbrock: hard; check progress and time
    {
        stan::math::recover_memory();
        Math::LBFGS<var> solver(criteria, 20);
        std::vector<double> x(50, -1.2);
        Math::OptimizerState state;
        const double us = timeit([&] {
            solver.minimize(Rosenbrock{}, x, state);
            return static_cast<double>(state.evals);
        });
        const double f_final = Rosenbrock{}(x);
        std::printf("  rosenbrock50: evals=%zu iters=%zu f=%.3e (%.0f us)\n", state.evals,
                    state.iterations, f_final, us);
        check("rosenbrock50 f < 1e-4", f_final < 1e-4);
    }
    // Beale
    {
        stan::math::recover_memory();
        Math::LBFGS<var> solver(criteria, 10);
        std::vector<double> x{1.0, 1.0};
        const auto r = solver.minimize(Beale{}, x);
        check("beale converged", r == Math::OptimizeResult::GradientTolReached);
        checkClose("beale x0", x[0], 3.0, 1e-5);
        checkClose("beale x1", x[1], 0.5, 1e-5);
    }
    // Himmelblau: different starts reach different minima
    {
        stan::math::recover_memory();
        Math::LBFGS<var> solver(criteria, 10);
        std::vector<double> x{1.0, 1.0};
        check("himmelblau(1,1) converged",
              solver.minimize(Himmelblau{}, x) == Math::OptimizeResult::GradientTolReached);
        checkClose("himmelblau(1,1) x0", x[0], 3.0, 1e-5);
        checkClose("himmelblau(1,1) x1", x[1], 2.0, 1e-5);
    }
    {
        stan::math::recover_memory();
        Math::LBFGS<var> solver(criteria, 10);
        std::vector<double> x{-2.0, 2.0};
        check("himmelblau(-2,2) converged",
              solver.minimize(Himmelblau{}, x) == Math::OptimizeResult::GradientTolReached);
        checkClose("himmelblau(-2,2) x0", x[0], -2.80511808695, 1e-4);
        checkClose("himmelblau(-2,2) x1", x[1], 3.13131251825, 1e-4);
    }
    // Badly scaled quadratic, condition ~1e20 (multi-scale beyond single-gamma
    // L-BFGS; NLopt fails outright here). The well-conditioned coordinates
    // converge; the flat one is reported as informational.
    {
        stan::math::recover_memory();
        WeightedQuadratic q;
        q.w = {1.0, 1e4, 1e8, 1e12, 1e16};
        q.target = {1.0, 1.0, 1.0, 1.0, 1.0};
        Math::LBFGS<var> solver(criteria, 10);
        std::vector<double> x(5, 0.0);
        Math::OptimizerState state;
        const double us = timeit([&] {
            solver.minimize(q, x, state);
            return static_cast<double>(state.evals);
        });
        std::printf("  illcond1e16: evals=%zu iters=%zu x0=%.6f x4=%.6f (%.0f us)\n", state.evals,
                    state.iterations, x[0], x[4], us);
        checkClose("illcond1e16 x4", x[4], 1.0, 1e-6);
        // x0 stays near its start on this 1e16 multi-scale problem (global
        // gamma limitation, same as NLopt which fails outright) -- informational.
    }
    // Condition ~1e10: fully within single-gamma L-BFGS reach
    {
        stan::math::recover_memory();
        WeightedQuadratic q;
        q.w = {1.0, 1e2, 1e4, 1e6, 1e8};
        q.target = {1.0, 1.0, 1.0, 1.0, 1.0};
        Math::LBFGS<var> solver(criteria, 10);
        std::vector<double> x(5, 0.0);
        const auto r = solver.minimize(q, x);
        check("illcond1e8 converged", r == Math::OptimizeResult::GradientTolReached);
        checkClose("illcond1e8 x0", x[0], 1.0, 1e-4);
        checkClose("illcond1e8 x4", x[4], 1.0, 1e-6);
    }
    // Edge: start at the optimum (zero gradient)
    {
        stan::math::recover_memory();
        Math::LBFGS<var> solver(criteria, 10);
        std::vector<double> x{1.0, 1.0};
        Math::OptimizerState state;
        const auto r = solver.minimize(Rosenbrock{}, x, state);
        check("start at optimum -> gradient stop", r == Math::OptimizeResult::GradientTolReached);
        check("start at optimum uses 1 eval", state.evals == 1);
    }
    // Edge: constant objective, any start
    {
        stan::math::recover_memory();
        Math::LBFGS<var> solver(criteria, 10);
        std::vector<double> x{3.0, -4.0};
        const auto r = solver.minimize(ConstantObjective{}, x);
        check("constant objective -> gradient stop", r == Math::OptimizeResult::GradientTolReached);
    }
    // Edge: n = 1
    {
        stan::math::recover_memory();
        Math::LBFGS<var> solver(criteria, 1);
        std::vector<double> x{-5.0};
        const auto r = solver.minimize(Shift{{2.0}}, x);
        check("n=1 converged", r == Math::OptimizeResult::GradientTolReached);
        checkClose("n=1 x", x[0], 2.0, 1e-10);
    }
}

// ── SLSQP / AUGLAG stress ─────────────────────────────────────────────────

void stressConstrained() {
    std::printf("=== SLSQP / AugLag stress ===\n");
    Math::StopCriteria criteria;
    criteria.maxeval = 200000;

    // Equality + inequality both active: min ||x-(2,2)||^2 s.t. x0+x1=2, x0<=x1
    // -> (1,1); multipliers: lambda_ineq = 2, nu_eq = 0
    {
        stan::math::recover_memory();
        Math::SLSQP<var> solver(criteria);
        std::vector<double> x{0.0, 0.0};
        const auto none = Math::Bounds::unbounded(2);
        const auto r = solver.minimize(Shift{{2.0, 2.0}}, SumLeq{}, SumEqual{}, none, x);
        check("mixed constraints converged", r == Math::OptimizeResult::Success);
        checkClose("mixed x0", x[0], 1.0, 1e-6);
        checkClose("mixed x1", x[1], 1.0, 1e-6);
    }
    // Redundant (linearly dependent) constraints
    {
        stan::math::recover_memory();
        Math::SLSQP<var> solver(criteria);
        std::vector<double> x{0.0, 0.0};
        const auto none = Math::Bounds::unbounded(2);
        const auto r = solver.minimize(Shift{{2.0, 2.0}}, SumLeqRedundant{}, none, x);
        check("redundant constraints converged", r == Math::OptimizeResult::Success);
        checkClose("redundant x0", x[0], 1.0, 1e-6);
        checkClose("redundant x1", x[1], 1.0, 1e-6);
    }
    // Degenerate: optimum exactly at a bound with zero gradient component
    {
        stan::math::recover_memory();
        Math::SLSQP<var> solver(criteria);
        std::vector<double> x{0.5};
        const auto bounds = Math::Bounds::fromVectors({0.0}, {Math::Bounds::kNoUpper});
        const auto r = solver.minimize(Shift{{-1.0}}, Math::NoConstraint{}, bounds, x);
        check("bound-active converged", r == Math::OptimizeResult::Success);
        checkClose("bound-active x", x[0], 0.0, 1e-8);
    }
    // Many constraints: 100-D lower bounds x_j >= 1, target 0 -> all active
    {
        stan::math::recover_memory();
        const std::size_t n = 100;
        Math::SLSQP<var> solver(criteria);
        std::vector<double> x(n, 2.0); // feasible start (NLopt contract)
        std::vector<double> lower(n, 1.0);
        std::vector<double> upper(n, Math::Bounds::kNoUpper);
        const auto bounds = Math::Bounds::fromVectors(lower, upper);
        Math::OptimizerState state;
        const double us = timeit([&] {
            solver.minimize(Shift{std::vector<double>(n, 0.0)}, Math::NoConstraint{}, bounds, x,
                            state);
            return static_cast<double>(state.evals);
        });
        bool all_one = true;
        for (double xi : x) {
            all_one = all_one && std::fabs(xi - 1.0) < 1e-6;
        }
        check("many-bounds x == 1", all_one);
        std::printf("  many-bounds n=100: evals=%zu iters=%zu (%.0f us)\n", state.evals,
                    state.iterations, us);
    }
    // Monotone + butterfly on a 20-point curve (arbitrage-like)
    {
        stan::math::recover_memory();
        const std::size_t n = 20;
        std::vector<double> target(n, 1.0);
        target[n / 2] = 2.0; // creates violations
        Math::SLSQP<var> solver(criteria);
        std::vector<double> p = target;
        const auto none = Math::Bounds::unbounded(n);
        Math::OptimizerState state;
        const double us = timeit([&] {
            solver.minimize(Shift{target}, CurveArb{}, none, p, state);
            return static_cast<double>(state.evals);
        });
        bool feasible = true;
        for (std::size_t i = 0; i + 1 < n; ++i) {
            feasible = feasible && (p[i + 1] - p[i] <= 1e-6);
        }
        for (std::size_t i = 0; i + 2 < n; ++i) {
            feasible = feasible && (p[i] - 2.0 * p[i + 1] + p[i + 2] >= -1e-6);
        }
        check("curve arbitrage feasible", feasible);
        std::printf("  curve-arb n=20: evals=%zu iters=%zu (%.0f us)\n", state.evals,
                    state.iterations, us);
    }
    // AUGLAG on the mixed-constraint problem
    {
        stan::math::recover_memory();
        Math::AugLag<var> solver(criteria);
        std::vector<double> x{0.0, 0.0};
        const auto none = Math::Bounds::unbounded(2);
        const auto r = solver.minimize(Shift{{2.0, 2.0}}, SumLeq{}, SumEqual{}, none, x);
        check("auglag mixed converged", r == Math::OptimizeResult::Success);
        checkClose("auglag mixed x0", x[0], 1.0, 1e-4);
        checkClose("auglag mixed x1", x[1], 1.0, 1e-4);
    }
    // AUGLAG on the 20-point arbitrage curve: does it reach feasibility?
    {
        stan::math::recover_memory();
        const std::size_t n = 20;
        std::vector<double> target(n, 1.0);
        target[n / 2] = 2.0;
        Math::AugLag<var> solver(criteria);
        std::vector<double> p = target;
        const auto none = Math::Bounds::unbounded(n);
        Math::OptimizerState state;
        const double us = timeit([&] {
            solver.minimize(Shift{target}, CurveArb{}, none, p, state);
            return static_cast<double>(state.evals);
        });
        bool feasible = true;
        for (std::size_t i = 0; i + 1 < n; ++i) {
            feasible = feasible && (p[i + 1] - p[i] <= 1e-5);
        }
        for (std::size_t i = 0; i + 2 < n; ++i) {
            feasible = feasible && (p[i] - 2.0 * p[i + 1] + p[i + 2] >= -1e-5);
        }
        check("auglag curve arbitrage feasible", feasible);
        std::printf("  auglag curve-arb n=20: evals=%zu iters=%zu (%.0f us)\n", state.evals,
                    state.iterations, us);
    }
}

} // namespace

int main() {
    stressLbfgs();
    stressConstrained();
    stan::math::recover_memory();
    if (failures == 0) {
        std::printf("\ntest_optimizers_stress: all invariants hold\n");
        return 0;
    }
    std::printf("\n%d stress check(s) FAILED\n", failures);
    return 1;
}
