/**
 * @file test_optimizers_stress.cpp
 * @brief Hard problems, edge cases, and timing probes for the optimization stack
 *
 * Finds correctness limits (does the solver converge?) and optimization
 * opportunities (where does time go at scale). Timing probes are recorded as
 * test properties, never asserted.
 */

#include "quantape/math/StanMath.h"

#include "quantape/math/Optimization/AugLag.h"
#include "quantape/math/Optimization/LBFGS.h"
#include "quantape/math/Optimization/OptimizerStanPrimitives.h"
#include "quantape/math/Optimization/SLSQP.h"

#include <chrono>
#include <cmath>
#include <string>
#include <vector>

#include "support/GtestSupport.h"
#include "support/StanTapeFixture.h"

using stan::math::var;

namespace {

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

} // namespace

class OptimizersStressTest : public StanTapeTest {};

TEST_F(OptimizersStressTest, rosenbrockDimensions) {
    quantape::math::StopCriteria criteria;
    criteria.maxeval = 100000;

    // 2-D Rosenbrock
    {
        SCOPED_TRACE("rosenbrock2");
        quantape::math::LBFGS<var> solver(criteria, 10);
        std::vector<double> x{-1.2, 1.0};
        quantape::math::OptimizerState state;
        const auto r = solver.minimize(Rosenbrock{}, x, state);
        EXPECT_TRUE(r == quantape::math::OptimizeResult::GradientTolReached);
        CHECK_CLOSE("rosenbrock2 x0", x[0], 1.0, 1e-6);
        CHECK_CLOSE("rosenbrock2 x1", x[1], 1.0, 1e-6);
    }
    // 10-D Rosenbrock
    {
        SCOPED_TRACE("rosenbrock10");
        quantape::math::LBFGS<var> solver(criteria, 10);
        std::vector<double> x(10, -1.2);
        const auto r = solver.minimize(Rosenbrock{}, x);
        EXPECT_TRUE(r == quantape::math::OptimizeResult::GradientTolReached);
        CHECK_CLOSE("rosenbrock10 x0", x[0], 1.0, 1e-4);
    }
    // 50-D Rosenbrock: hard; check progress and time
    {
        SCOPED_TRACE("rosenbrock50");
        quantape::math::LBFGS<var> solver(criteria, 20);
        std::vector<double> x(50, -1.2);
        quantape::math::OptimizerState state;
        const double us = timeit([&] {
            solver.minimize(Rosenbrock{}, x, state);
            return static_cast<double>(state.evals);
        });
        const double f_final = Rosenbrock{}(x);
        ::testing::Test::RecordProperty("rosenbrock50_us", quantape::util::num(us, 6));
        ::testing::Test::RecordProperty("rosenbrock50_evals", std::to_string(state.evals));
        EXPECT_TRUE(f_final < 1e-4);
    }
}

TEST_F(OptimizersStressTest, bealeAndHimmelblauBasins) {
    quantape::math::StopCriteria criteria;
    criteria.maxeval = 100000;

    // Beale
    {
        SCOPED_TRACE("beale");
        quantape::math::LBFGS<var> solver(criteria, 10);
        std::vector<double> x{1.0, 1.0};
        const auto r = solver.minimize(Beale{}, x);
        EXPECT_TRUE(r == quantape::math::OptimizeResult::GradientTolReached);
        CHECK_CLOSE("beale x0", x[0], 3.0, 1e-5);
        CHECK_CLOSE("beale x1", x[1], 0.5, 1e-5);
    }
    // Himmelblau: different starts reach different minima
    {
        SCOPED_TRACE("himmelblau(1,1)");
        quantape::math::LBFGS<var> solver(criteria, 10);
        std::vector<double> x{1.0, 1.0};
        EXPECT_TRUE(solver.minimize(Himmelblau{}, x) ==
                    quantape::math::OptimizeResult::GradientTolReached);
        CHECK_CLOSE("himmelblau(1,1) x0", x[0], 3.0, 1e-5);
        CHECK_CLOSE("himmelblau(1,1) x1", x[1], 2.0, 1e-5);
    }
    {
        SCOPED_TRACE("himmelblau(-2,2)");
        quantape::math::LBFGS<var> solver(criteria, 10);
        std::vector<double> x{-2.0, 2.0};
        EXPECT_TRUE(solver.minimize(Himmelblau{}, x) ==
                    quantape::math::OptimizeResult::GradientTolReached);
        CHECK_CLOSE("himmelblau(-2,2) x0", x[0], -2.80511808695, 1e-4);
        CHECK_CLOSE("himmelblau(-2,2) x1", x[1], 3.13131251825, 1e-4);
    }
}

TEST_F(OptimizersStressTest, illConditionedQuadratic) {
    quantape::math::StopCriteria criteria;
    criteria.maxeval = 100000;

    // Badly scaled quadratic, condition ~1e20 (multi-scale beyond single-gamma
    // L-BFGS; NLopt fails outright here). The well-conditioned coordinates
    // converge; the flat one is reported as informational.
    {
        SCOPED_TRACE("illcond1e16");
        WeightedQuadratic q;
        q.w = {1.0, 1e4, 1e8, 1e12, 1e16};
        q.target = {1.0, 1.0, 1.0, 1.0, 1.0};
        quantape::math::LBFGS<var> solver(criteria, 10);
        std::vector<double> x(5, 0.0);
        quantape::math::OptimizerState state;
        const double us = timeit([&] {
            solver.minimize(q, x, state);
            return static_cast<double>(state.evals);
        });
        ::testing::Test::RecordProperty("illcond1e16_us", quantape::util::num(us, 6));
        ::testing::Test::RecordProperty("illcond1e16_evals", std::to_string(state.evals));
        CHECK_CLOSE("illcond1e16 x4", x[4], 1.0, 1e-6);
        // x0 stays near its start on this 1e16 multi-scale problem (global
        // gamma limitation, same as NLopt which fails outright) -- informational.
    }
    // Condition ~1e10: fully within single-gamma L-BFGS reach
    {
        SCOPED_TRACE("illcond1e8");
        WeightedQuadratic q;
        q.w = {1.0, 1e2, 1e4, 1e6, 1e8};
        q.target = {1.0, 1.0, 1.0, 1.0, 1.0};
        quantape::math::LBFGS<var> solver(criteria, 10);
        std::vector<double> x(5, 0.0);
        const auto r = solver.minimize(q, x);
        EXPECT_TRUE(r == quantape::math::OptimizeResult::GradientTolReached);
        CHECK_CLOSE("illcond1e8 x0", x[0], 1.0, 1e-4);
        CHECK_CLOSE("illcond1e8 x4", x[4], 1.0, 1e-6);
    }
}

TEST_F(OptimizersStressTest, optimumAndConstantObjectiveEdges) {
    quantape::math::StopCriteria criteria;
    criteria.maxeval = 100000;

    // Edge: start at the optimum (zero gradient)
    {
        SCOPED_TRACE("start at optimum");
        quantape::math::LBFGS<var> solver(criteria, 10);
        std::vector<double> x{1.0, 1.0};
        quantape::math::OptimizerState state;
        const auto r = solver.minimize(Rosenbrock{}, x, state);
        EXPECT_TRUE(r == quantape::math::OptimizeResult::GradientTolReached);
        EXPECT_TRUE(state.evals == 1);
    }
    // Edge: constant objective, any start
    {
        SCOPED_TRACE("constant objective");
        quantape::math::LBFGS<var> solver(criteria, 10);
        std::vector<double> x{3.0, -4.0};
        const auto r = solver.minimize(ConstantObjective{}, x);
        EXPECT_TRUE(r == quantape::math::OptimizeResult::GradientTolReached);
    }
    // Edge: n = 1
    {
        SCOPED_TRACE("n=1");
        quantape::math::LBFGS<var> solver(criteria, 1);
        std::vector<double> x{-5.0};
        const auto r = solver.minimize(Shift{{2.0}}, x);
        EXPECT_TRUE(r == quantape::math::OptimizeResult::GradientTolReached);
        CHECK_CLOSE("n=1 x", x[0], 2.0, 1e-10);
    }
}

TEST_F(OptimizersStressTest, mixedRedundantAndBoundActive) {
    quantape::math::StopCriteria criteria;
    criteria.maxeval = 200000;

    // Equality + inequality both active: min ||x-(2,2)||^2 s.t. x0+x1=2, x0<=x1
    // -> (1,1); multipliers: lambda_ineq = 2, nu_eq = 0
    {
        SCOPED_TRACE("mixed constraints");
        quantape::math::SLSQP<var> solver(criteria);
        std::vector<double> x{0.0, 0.0};
        const auto none = quantape::math::Bounds::unbounded(2);
        const auto r = solver.minimize(Shift{{2.0, 2.0}}, SumLeq{}, SumEqual{}, none, x);
        EXPECT_TRUE(r == quantape::math::OptimizeResult::Success);
        CHECK_CLOSE("mixed x0", x[0], 1.0, 1e-6);
        CHECK_CLOSE("mixed x1", x[1], 1.0, 1e-6);
    }
    // Redundant (linearly dependent) constraints
    {
        SCOPED_TRACE("redundant constraints");
        quantape::math::SLSQP<var> solver(criteria);
        std::vector<double> x{0.0, 0.0};
        const auto none = quantape::math::Bounds::unbounded(2);
        const auto r = solver.minimize(Shift{{2.0, 2.0}}, SumLeqRedundant{}, none, x);
        EXPECT_TRUE(r == quantape::math::OptimizeResult::Success);
        CHECK_CLOSE("redundant x0", x[0], 1.0, 1e-6);
        CHECK_CLOSE("redundant x1", x[1], 1.0, 1e-6);
    }
    // Degenerate: optimum exactly at a bound with zero gradient component
    {
        SCOPED_TRACE("bound-active");
        quantape::math::SLSQP<var> solver(criteria);
        std::vector<double> x{0.5};
        const auto bounds =
            quantape::math::Bounds::fromVectors({0.0}, {quantape::math::Bounds::kNoUpper});
        const auto r = solver.minimize(Shift{{-1.0}}, quantape::math::NoConstraint{}, bounds, x);
        EXPECT_TRUE(r == quantape::math::OptimizeResult::Success);
        CHECK_CLOSE("bound-active x", x[0], 0.0, 1e-8);
    }
    // AUGLAG on the mixed-constraint problem
    {
        SCOPED_TRACE("auglag mixed");
        quantape::math::AugLag<var> solver(criteria);
        std::vector<double> x{0.0, 0.0};
        const auto none = quantape::math::Bounds::unbounded(2);
        const auto r = solver.minimize(Shift{{2.0, 2.0}}, SumLeq{}, SumEqual{}, none, x);
        EXPECT_TRUE(r == quantape::math::OptimizeResult::Success);
        CHECK_CLOSE("auglag mixed x0", x[0], 1.0, 1e-4);
        CHECK_CLOSE("auglag mixed x1", x[1], 1.0, 1e-4);
    }
}

TEST_F(OptimizersStressTest, hundredDimensionalBounds) {
    quantape::math::StopCriteria criteria;
    criteria.maxeval = 200000;
    // Many constraints: 100-D lower bounds x_j >= 1, target 0 -> all active
    const std::size_t n = 100;
    quantape::math::SLSQP<var> solver(criteria);
    std::vector<double> x(n, 2.0); // feasible start (NLopt contract)
    std::vector<double> lower(n, 1.0);
    std::vector<double> upper(n, quantape::math::Bounds::kNoUpper);
    const auto bounds = quantape::math::Bounds::fromVectors(lower, upper);
    quantape::math::OptimizerState state;
    const double us = timeit([&] {
        solver.minimize(Shift{std::vector<double>(n, 0.0)}, quantape::math::NoConstraint{}, bounds,
                        x, state);
        return static_cast<double>(state.evals);
    });
    bool all_one = true;
    for (double xi : x) {
        all_one = all_one && std::fabs(xi - 1.0) < 1e-6;
    }
    ::testing::Test::RecordProperty("many_bounds_us", quantape::util::num(us, 6));
    ::testing::Test::RecordProperty("many_bounds_evals", std::to_string(state.evals));
    EXPECT_TRUE(all_one);
}

TEST_F(OptimizersStressTest, curveArbitrageSlSqpAndAugLag) {
    quantape::math::StopCriteria criteria;
    criteria.maxeval = 200000;
    const std::size_t n = 20;
    std::vector<double> target(n, 1.0);
    target[n / 2] = 2.0; // creates violations
    const auto none = quantape::math::Bounds::unbounded(n);

    // Monotone + butterfly on a 20-point curve (arbitrage-like)
    {
        SCOPED_TRACE("curve arbitrage (SLSQP)");
        quantape::math::SLSQP<var> solver(criteria);
        std::vector<double> p = target;
        quantape::math::OptimizerState state;
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
        ::testing::Test::RecordProperty("curve_arb_slsqp_us", quantape::util::num(us, 6));
        ::testing::Test::RecordProperty("curve_arb_slsqp_evals", std::to_string(state.evals));
        EXPECT_TRUE(feasible);
    }
    // AUGLAG on the 20-point arbitrage curve: does it reach feasibility?
    {
        SCOPED_TRACE("curve arbitrage (AugLag)");
        quantape::math::AugLag<var> solver(criteria);
        std::vector<double> p = target;
        quantape::math::OptimizerState state;
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
        ::testing::Test::RecordProperty("curve_arb_auglag_us", quantape::util::num(us, 6));
        ::testing::Test::RecordProperty("curve_arb_auglag_evals", std::to_string(state.evals));
        EXPECT_TRUE(feasible);
    }
}
