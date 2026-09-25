// test_optimizers_bench.cpp — timing harness for the Math/Optimization stack.
//
// Measures wall time, iterations and evaluations per method on fixed
// problems; used with the sampling profiler to locate bottlenecks:
//
//   ./build/test_optimizers_bench [reps]
//   /usr/bin/sample test_optimizers_bench 5 -file /tmp/prof.txt
//
// Run: ./test_optimizers_bench 200
#include "quantape/math/Optimization/AugLag.h"
#include "quantape/math/Optimization/ImplicitFunction.h"
#include "quantape/math/Optimization/LBFGS.h"
#include "quantape/math/Optimization/OptimizerStanPrimitives.h"
#include "quantape/math/Optimization/SLSQP.h"
#include "quantape/math/Optimization/TNewton.h"
#include "quantape/math/StanMath.h"

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <vector>

using stan::math::var;

namespace {

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

struct RosenbrockValueGrad {
    double operator()(const std::vector<double>& x, std::vector<double>& grad) const {
        grad.assign(x.size(), 0.0);
        double sum = 0.0;
        for (std::size_t i = 0; i + 1 < x.size(); ++i) {
            const double t1 = x[i + 1] - x[i] * x[i];
            const double t2 = 1.0 - x[i];
            sum += 100.0 * t1 * t1 + t2 * t2;
            grad[i] += -400.0 * x[i] * t1 - 2.0 * t2;
            grad[i + 1] += 200.0 * t1;
        }
        return sum;
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

struct Arb {
    template <typename S>
    void operator()(const std::vector<S>& p, std::vector<S>& g) const {
        g.resize(3);
        g[0] = p[1] - p[0];
        g[1] = p[2] - p[1];
        g[2] = -(p[0] - S(2.0) * p[1] + p[2]);
    }
};

// ── IFT fixtures ──

struct Lsq2Arg {
    std::vector<std::vector<double>> a;
    template <typename Sx, typename Sm>
    auto operator()(const std::vector<Sx>& x, const std::vector<Sm>& m) const {
        using S = decltype(x[0] * Sx(0.0) + m[0] * Sm(0.0));
        S sum = S(0.0);
        for (std::size_t k = 0; k < m.size(); ++k) {
            S r = S(0.0);
            for (std::size_t i = 0; i < x.size(); ++i) {
                r += x[i] * S(a[k][i]);
            }
            r -= S(m[k]);
            sum += S(0.5) * r * r;
        }
        return sum;
    }
};

struct SepQuad2Arg {
    template <typename Sx, typename Sm>
    auto operator()(const std::vector<Sx>& x, const std::vector<Sm>& m) const {
        const Sx d0 = x[0] - Sx(m[0]);
        const Sx d1 = x[1] - Sx(m[1]);
        return Sx(0.5) * (d0 * d0 + d1 * d1);
    }
};

struct X0Cap2Arg {
    template <typename Sx, typename Sm>
    void operator()(const std::vector<Sx>& x, const std::vector<Sm>&, std::vector<Sx>& out) const {
        out.clear();
        out.push_back(x[0] - Sx(0.8));
    }
};

struct SumOne2Arg {
    template <typename Sx, typename Sm>
    void operator()(const std::vector<Sx>& x, const std::vector<Sm>&, std::vector<Sx>& out) const {
        out.clear();
        out.push_back(x[0] + x[1] - Sx(1.0));
    }
};

template <typename Fn>
void bench(const char* name, int reps, const Fn& fn) {
    // warmup
    double sink = fn();
    const auto start = std::chrono::steady_clock::now();
    for (int i = 0; i < reps; ++i) {
        sink += fn();
    }
    const auto stop = std::chrono::steady_clock::now();
    const double us = std::chrono::duration<double, std::micro>(stop - start).count() / reps;
    std::printf("%-34s %10.1f us/run   (sink %.3e)\n", name, us, sink);
    std::fflush(stdout);
}

// Each run returns (evals << 8) + iterations so counts stay visible per rep
template <typename Solver, typename F, typename... Rest>
double run_optimizer(Solver& solver, const F& f, std::vector<double> x0, Rest&&... rest) {
    quantape::math::OptimizerState state;
    const auto result = solver.minimize(f, std::forward<Rest>(rest)..., x0, state);
    if (result != quantape::math::OptimizeResult::Success &&
        result != quantape::math::OptimizeResult::GradientTolReached) {
        std::printf("  [warn] result=%s\n", quantape::math::to_string(result));
    }
    return static_cast<double>(state.evals * 256 + state.iterations);
}

} // namespace

int main(int argc, char** argv) {
    const int reps = argc > 1 ? std::atoi(argv[1]) : 200;
    std::printf("optimizer benchmark, reps=%d\n", reps);

    quantape::math::StopCriteria criteria;
    criteria.maxeval = 100000;
    criteria.grad_tol = 1e-8;
    const auto none2 = quantape::math::Bounds::unbounded(2);
    const auto none3 = quantape::math::Bounds::unbounded(3);

    bench("LBFGS<var> rosenbrock n=2", reps, [&] {
        quantape::math::LBFGS<var> solver(criteria, 10);
        return run_optimizer(solver, Rosenbrock{}, std::vector<double>{-1.2, 1.0});
    });
    bench("LBFGS<double> rosenbrock n=2", reps, [&] {
        quantape::math::LBFGS<double> solver(criteria, 10);
        return run_optimizer(solver, RosenbrockValueGrad{}, std::vector<double>{-1.2, 1.0});
    });
    bench("LBFGS<var> rosenbrock n=10", reps, [&] {
        quantape::math::LBFGS<var> solver(criteria, 10);
        return run_optimizer(solver, Rosenbrock{}, std::vector<double>(10, -1.2));
    });
    bench("SLSQP<var> ineq-quadratic n=2", reps, [&] {
        quantape::math::SLSQP<var> solver(criteria);
        return run_optimizer(solver, Shift{{2.0, 2.0}}, std::vector<double>{0.0, 0.0}, SumLeq{},
                             none2);
    });
    bench("SLSQP<var> arbitrage n=3", reps, [&] {
        quantape::math::SLSQP<var> solver(criteria);
        return run_optimizer(solver, Shift{{1.0, 1.5, 1.0}}, std::vector<double>{1.0, 1.5, 1.0},
                             Arb{}, none3);
    });
    bench("AugLag<var> ineq-quadratic n=2", reps, [&] {
        quantape::math::AugLag<var> solver(criteria);
        return run_optimizer(solver, Shift{{2.0, 2.0}}, std::vector<double>{0.0, 0.0}, SumLeq{},
                             none2);
    });
    bench("TNewton<var> rosenbrock n=2", reps, [&] {
        quantape::math::TNewton<var> solver(criteria);
        return run_optimizer(solver, Rosenbrock{}, std::vector<double>{-1.2, 1.0});
    });
    bench("TNewton<var> rosenbrock n=10", reps, [&] {
        quantape::math::TNewton<var> solver(criteria);
        return run_optimizer(solver, Rosenbrock{}, std::vector<double>(10, -1.2));
    });
    bench("IFT unconstrained LSQ n=3/M=5", reps, [&] {
        stan::math::recover_memory();
        Lsq2Arg obj{{{1.0, 0.2, -0.1},
                     {0.5, 1.0, 0.3},
                     {0.1, 0.4, 1.0},
                     {0.8, -0.3, 0.6},
                     {-0.2, 0.7, 0.9}}};
        const std::vector<double> m{0.8, 1.2, -0.5, 0.3, 1.7};
        const std::vector<double> x_hat{0.427726234046, 1.199304415223, -0.025418852119};
        quantape::math::IftResult ift;
        std::vector<double> dp;
        quantape::math::iftUnconstrained(obj, x_hat, m, dp, ift);
        return ift.condition_number;
    });
    bench("IFT KKT combined n=2", reps, [&] {
        stan::math::recover_memory();
        const std::vector<double> m{2.0, -1.0};
        const std::vector<double> x_hat{0.8, 0.2};
        quantape::math::IftResult ift;
        std::vector<double> dp, dl, dn;
        quantape::math::iftKkt(SepQuad2Arg{}, X0Cap2Arg{}, SumOne2Arg{}, none2, x_hat, m, {2.4},
                               {-1.2}, dp, dl, dn, ift);
        return ift.condition_number + dp[0] + dl[0] + dn[1];
    });
    bench("minimizeDifferential KKT n=2", reps, [&] {
        stan::math::recover_memory();
        const std::vector<double> m{2.0, -1.0};
        std::vector<double> x{0.0, 0.0};
        quantape::math::OptimizerState state;
        quantape::math::IftResult ift;
        const auto r = quantape::math::minimizeDifferential(SepQuad2Arg{}, X0Cap2Arg{},
                                                            SumOne2Arg{}, none2, m, x, state, ift,
                                                            nullptr, nullptr, nullptr, criteria);
        return static_cast<double>(r) + x[0] + x[1];
    });
    return 0;
}
