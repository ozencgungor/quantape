/**
 * @file test_optimization.cpp
 * @brief Validates the optimizer building blocks (Math/Optimization):
 * stop criteria, the Optimizer<DoubleT, Impl> CRTP base, and the AD
 * evaluation helpers valueGrad / hvp / constraintValueJacobian.
 *
 *   1. stop criteria: relative/absolute f and x tests, dx, gradient,
 *      evaluation budget (NLopt semantics)
 *   2. valueGrad: exact gradients at var and `fvar<var>`
 *   3. hvp: exact Hessian-vector products (no finite differences)
 *   4. constraintValueJacobian: values + row-major Jacobian
 *   5. CRTP base plumbing: a throwaway steepest-descent method drives the
 *      base's evaluation helpers and stop logic; result codes and counters
 *      are checked for each exit path
 */

#include "quantape/math/StanMath.h"

#include "quantape/math/Optimization/AugLag.h"
#include "quantape/math/Optimization/Constraint.h"
#include "quantape/math/Optimization/LBFGS.h"
#include "quantape/math/Optimization/OptimizerStanPrimitives.h"
#include "quantape/math/Optimization/QpSolver.h"
#include "quantape/math/Optimization/SLSQP.h"
#include "quantape/math/Optimization/TNewton.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <random>
#include <stdexcept>
#include <utility>
#include <vector>

#include "support/GtestSupport.h"
#include "support/StanTapeFixture.h"

using stan::math::fvar;
using stan::math::var;

namespace {

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

/// Rosenbrock value only (for finite-difference gradient references)
double rosenbrockValue(const std::vector<double>& x) {
    const double t1 = x[1] - x[0] * x[0];
    const double t2 = 1.0 - x[0];
    return 100.0 * t1 * t1 + t2 * t2;
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

// ── CRTP smoke test: a throwaway steepest-descent method ──────────────────
template <typename DoubleT>
class SteepestDescent : public quantape::math::Optimizer<DoubleT, SteepestDescent<DoubleT>> {
public:
    using Base = quantape::math::Optimizer<DoubleT, SteepestDescent<DoubleT>>;

    explicit SteepestDescent(quantape::math::StopCriteria criteria = {}, double step = 0.1)
        : Base(std::move(criteria)), m_step(step) {}

    // internal, public for CRTP access (like Solver1D's solveImpl)
    template <typename F>
        requires quantape::math::VectorObjective<F, DoubleT>
    quantape::math::OptimizeResult minimizeImpl(const F& f,
                                                quantape::math::OptimizerState& state) const {
        this->updateValueGrad(f, state);
        while (true) {
            if (this->stopByGradient(state.grad)) {
                state.message = "gradient tolerance";
                return quantape::math::OptimizeResult::GradientTolReached;
            }
            if (this->stopByEvalOrTime(state)) {
                return quantape::math::stopTime(this->criteria(), state.start_time)
                           ? quantape::math::OptimizeResult::MaxTimeReached
                           : quantape::math::OptimizeResult::MaxEvalReached;
            }
            const double old_f = state.f;
            const std::vector<double> old_x = state.x;
            for (std::size_t i = 0; i < state.x.size(); ++i) {
                state.x[i] -= m_step * state.grad[i];
            }
            this->updateValueGrad(f, state);
            ++state.iterations;

            if (state.f <= this->criteria().stopval) {
                return quantape::math::OptimizeResult::StopvalReached;
            }
            if (quantape::math::stopFtol(this->criteria(), state.f, old_f)) {
                return quantape::math::OptimizeResult::FtolReached;
            }
            if (quantape::math::stopX(this->criteria(), state.x, old_x)) {
                return quantape::math::OptimizeResult::XtolReached;
            }
        }
    }

private:
    double m_step;
};

// Minimal value-only method: exercises the double backend of the base
template <typename DoubleT>
class ValueProbe : public quantape::math::Optimizer<DoubleT, ValueProbe<DoubleT>> {
public:
    using Base = quantape::math::Optimizer<DoubleT, ValueProbe<DoubleT>>;
    using Base::Base;

    template <typename F>
        requires quantape::math::VectorObjective<F, DoubleT>
    quantape::math::OptimizeResult minimizeImpl(const F& f,
                                                quantape::math::OptimizerState& state) const {
        this->updateValueGrad(f, state);
        return quantape::math::OptimizeResult::Success;
    }
};

double dot(const std::vector<double>& a, const std::vector<double>& b) {
    double sum = 0.0;
    for (std::size_t i = 0; i < a.size(); ++i) {
        sum += a[i] * b[i];
    }
    return sum;
}

// ── QP solver reference helpers (dense, double-only) ──────────────────────

double qpObjectiveValue(const quantape::math::QpProblem& qp, const std::vector<double>& d) {
    const std::size_t n = qp.dimension();
    double f = 0.0;
    for (std::size_t i = 0; i < n; ++i) {
        f += qp.g[i] * d[i];
        for (std::size_t j = 0; j < n; ++j) {
            f += 0.5 * qp.B[i * n + j] * d[i] * d[j];
        }
    }
    return f;
}

double qpMaxAbsValue(const std::vector<double>& v) {
    double m = 0.0;
    for (double x : v) {
        m = std::max(m, std::fabs(x));
    }
    return m;
}

/// Primal feasibility with the solver's per-row relative scale:
///   r_i <= tol (1 + |b_i| + sum_j |A_ij| |d_j|)
bool qpFeasibleAt(const quantape::math::QpProblem& qp, const std::vector<double>& d, double tol) {
    const std::size_t n = qp.dimension();
    const std::size_t m = qp.rows();
    for (std::size_t i = 0; i < m; ++i) {
        double r = qp.b[i];
        double scale = 1.0 + std::fabs(qp.b[i]);
        for (std::size_t j = 0; j < n; ++j) {
            r += qp.A[i * n + j] * d[j];
            scale += std::fabs(qp.A[i * n + j]) * std::fabs(d[j]);
        }
        if (qp.equality[i] ? std::fabs(r) > tol * scale : r > tol * scale) {
            return false;
        }
    }
    return true;
}

/// Reference optimum by enumerating every subset of inequality rows.  Rows
/// linearly dependent on the selected set are dropped before the KKT solve
/// (the feasibility gate still checks every original row, so an inconsistent
/// dependency rejects the candidate).  Returns false when no KKT point exists,
/// i.e. when the QP is infeasible.
bool qpBruteForce(const quantape::math::QpProblem& qp, std::vector<double>& best_d) {
    const std::size_t n = qp.dimension();
    const std::size_t m = qp.rows();
    std::vector<std::size_t> inequalities;
    for (std::size_t i = 0; i < m; ++i) {
        if (!qp.equality[i]) {
            inequalities.push_back(i);
        }
    }
    best_d.clear();
    double best_f = 0.0;
    const std::size_t combinations = std::size_t{1} << inequalities.size();
    for (std::size_t mask = 0; mask < combinations; ++mask) {
        std::vector<std::size_t> active;
        for (std::size_t i = 0; i < m; ++i) {
            if (qp.equality[i]) {
                active.push_back(i);
            }
        }
        for (std::size_t b = 0; b < inequalities.size(); ++b) {
            if ((mask & (std::size_t{1} << b)) != 0) {
                active.push_back(inequalities[b]);
            }
        }
        // Greedy independent reduction (modified Gram-Schmidt, one re-pass).
        std::vector<std::vector<double>> orthonormal;
        std::vector<std::size_t> kept;
        for (std::size_t row_index : active) {
            std::vector<double> v(qp.A.begin() + static_cast<std::ptrdiff_t>(row_index * n),
                                  qp.A.begin() + static_cast<std::ptrdiff_t>((row_index + 1) * n));
            double norm2 = 0.0;
            for (double x : v) {
                norm2 += x * x;
            }
            if (!(norm2 > 0.0)) {
                continue;
            }
            for (int pass = 0; pass < 2; ++pass) {
                for (const auto& q : orthonormal) {
                    double coefficient = 0.0;
                    for (std::size_t j = 0; j < n; ++j) {
                        coefficient += q[j] * v[j];
                    }
                    for (std::size_t j = 0; j < n; ++j) {
                        v[j] -= coefficient * q[j];
                    }
                }
            }
            double remaining = 0.0;
            for (double x : v) {
                remaining += x * x;
            }
            if (remaining <= 1e-20 * norm2) {
                continue;
            }
            const double norm = std::sqrt(remaining);
            for (double& x : v) {
                x /= norm;
            }
            orthonormal.push_back(std::move(v));
            kept.push_back(row_index);
        }

        const std::size_t k = kept.size();
        const std::size_t dim = n + k;
        std::vector<double> matrix(dim * dim, 0.0);
        std::vector<double> rhs(dim, 0.0);
        for (std::size_t i = 0; i < n; ++i) {
            for (std::size_t j = 0; j < n; ++j) {
                matrix[i * dim + j] = qp.B[i * n + j];
            }
            rhs[i] = -qp.g[i];
        }
        for (std::size_t w = 0; w < k; ++w) {
            const std::size_t row_index = kept[w];
            for (std::size_t j = 0; j < n; ++j) {
                const double a = qp.A[row_index * n + j];
                matrix[(n + w) * dim + j] = a;
                matrix[j * dim + (n + w)] = a;
            }
            rhs[n + w] = -qp.b[row_index];
        }
        if (!quantape::math::detail::solveDenseInPlace(matrix, rhs, dim)) {
            continue;
        }
        const std::vector<double> d(rhs.begin(), rhs.begin() + static_cast<std::ptrdiff_t>(n));
        const std::vector<double> lambda(rhs.begin() + static_cast<std::ptrdiff_t>(n), rhs.end());
        if (!qpFeasibleAt(qp, d, 1e-8)) {
            continue;
        }
        bool dual_feasible = true;
        for (std::size_t w = 0; w < k; ++w) {
            if (!qp.equality[kept[w]] && lambda[w] < -1e-7 * (1.0 + qpMaxAbsValue(lambda))) {
                dual_feasible = false;
            }
        }
        if (!dual_feasible) {
            continue;
        }
        std::vector<double> stationarity = qp.g;
        double max_bd = 0.0;
        double max_atl = 0.0;
        for (std::size_t j = 0; j < n; ++j) {
            for (std::size_t i = 0; i < n; ++i) {
                const double term = qp.B[j * n + i] * d[i];
                stationarity[j] += term;
                max_bd = std::max(max_bd, std::fabs(term));
            }
        }
        for (std::size_t w = 0; w < k; ++w) {
            if (lambda[w] == 0.0) {
                continue;
            }
            for (std::size_t j = 0; j < n; ++j) {
                const double term = lambda[w] * qp.A[kept[w] * n + j];
                stationarity[j] += term;
                max_atl = std::max(max_atl, std::fabs(term));
            }
        }
        const double scale = 1.0 + qpMaxAbsValue(qp.g) + max_bd + max_atl;
        if (qpMaxAbsValue(stationarity) > 1e-7 * scale) {
            continue;
        }
        const double f = qpObjectiveValue(qp, d);
        if (best_d.empty() || f < best_f) {
            best_f = f;
            best_d = d;
        }
    }
    return !best_d.empty();
}

/// Deterministic pseudo-random stream for the differential QP tests.
struct QpRandom {
    std::mt19937_64 gen;
    explicit QpRandom(std::uint64_t seed) : gen(seed) {}
    double uniform(double lo, double hi) {
        return std::uniform_real_distribution<double>(lo, hi)(gen);
    }
    int integer(int lo, int hi) { return std::uniform_int_distribution<int>(lo, hi)(gen); }
};

/// Well-scaled random QP: B = M M' + diag (SPD), ~1/4 equality rows.
quantape::math::QpProblem makeRandomQp(QpRandom& rng, int n, int m) {
    quantape::math::QpProblem qp;
    qp.g.resize(static_cast<std::size_t>(n));
    qp.B.assign(static_cast<std::size_t>(n * n), 0.0);
    qp.A.assign(static_cast<std::size_t>(m * n), 0.0);
    qp.b.resize(static_cast<std::size_t>(m));
    qp.equality.assign(static_cast<std::size_t>(m), 0);
    std::vector<double> factor(static_cast<std::size_t>(n * n));
    for (double& x : factor) {
        x = rng.uniform(-1.0, 1.0);
    }
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            double sum = 0.0;
            for (int k = 0; k < n; ++k) {
                sum += factor[static_cast<std::size_t>(i * n + k)] *
                       factor[static_cast<std::size_t>(j * n + k)];
            }
            qp.B[static_cast<std::size_t>(i * n + j)] = sum;
        }
    }
    for (int i = 0; i < n; ++i) {
        qp.B[static_cast<std::size_t>(i * n + i)] += rng.uniform(0.05, 1.0);
        qp.g[static_cast<std::size_t>(i)] = rng.uniform(-3.0, 3.0);
    }
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < n; ++j) {
            qp.A[static_cast<std::size_t>(i * n + j)] = rng.uniform(-2.0, 2.0);
        }
        qp.b[static_cast<std::size_t>(i)] = rng.uniform(-2.0, 2.0);
        qp.equality[static_cast<std::size_t>(i)] = rng.integer(0, 3) == 0 ? 1 : 0;
    }
    return qp;
}

/// Three lower bounds on x0, all violated at x0 = 0; the binding one is 2.0358.
struct ThreeLowerBoundsValueJac {
    void operator()(const std::vector<double>& x, std::vector<double>& c,
                    std::vector<double>& J) const {
        c = {2.0358 - x[0], 2.0358 - 1.7 * x[0], 2.0358 - 1.3 * x[0]};
        J = {-1.0, -1.7, -1.3};
    }
};

} // namespace

class StopCriteriaTest : public StanTapeTest {};
class ValueGradTest : public StanTapeTest {};
class OptimizationAdTest : public StanTapeTest {};
class ConstraintJacobianTest : public StanTapeTest {};
class LineSearchTest : public StanTapeTest {};
class LbfgsAdTest : public StanTapeTest {};
class LbfgsDoubleTest : public StanTapeTest {};
class GradientSanityTest : public StanTapeTest {};
class ConstraintsTest : public StanTapeTest {};
class SlSqpAdTest : public StanTapeTest {};
class TNewtonAdTest : public StanTapeTest {};
class OptimizerBaseTest : public StanTapeTest {};

TEST_F(StopCriteriaTest, relativeAbsoluteAndBudgetSemantics) {
    quantape::math::StopCriteria criteria;

    // function value: relative
    criteria.ftol_rel = 1e-6;
    EXPECT_TRUE(quantape::math::stopFtol(criteria, 1.0 + 1e-9, 1.0));
    EXPECT_TRUE(!quantape::math::stopFtol(criteria, 1.1, 1.0));
    // function value: absolute
    criteria = {};
    criteria.ftol_abs = 1e-8;
    EXPECT_TRUE(quantape::math::stopFtol(criteria, 1.0 + 1e-9, 1.0));
    EXPECT_TRUE(!quantape::math::stopFtol(criteria, 1.0 + 1e-7, 1.0));
    // zero crossing
    criteria = {};
    criteria.ftol_rel = 1e-8;
    EXPECT_TRUE(quantape::math::stopFtol(criteria, 0.0, 0.0));
    // non-finite old value never stops
    criteria = {};
    criteria.ftol_rel = 1e-8;
    // Non-finite old value never stops; pow avoids an infinity literal that
    // would warn under the release -ffast-math flags.
    const double huge = std::pow(10.0, 400.0);
    EXPECT_TRUE(!quantape::math::stopFtol(criteria, 0.0, huge));

    // iterate: relative and absolute
    criteria = {};
    criteria.xtol_rel = 1e-6;
    const std::vector<double> x{1.0, 2.0};
    EXPECT_TRUE(quantape::math::stopX(criteria, x, std::vector<double>{1.0 + 1e-9, 2.0}));
    EXPECT_TRUE(!quantape::math::stopX(criteria, x, std::vector<double>{1.0 + 1e-3, 2.0}));
    criteria = {};
    criteria.xtol_abs = 1e-8;
    EXPECT_TRUE(quantape::math::stopX(criteria, x, std::vector<double>{1.0 + 1e-9, 2.0 - 1e-9}));
    EXPECT_TRUE(!quantape::math::stopX(criteria, x, std::vector<double>{1.0, 2.0 + 1e-7}));

    // step
    criteria = {};
    criteria.xtol_abs = 1e-8;
    EXPECT_TRUE(quantape::math::stopDx(criteria, x, std::vector<double>{1e-9, -1e-9}));
    EXPECT_TRUE(!quantape::math::stopDx(criteria, x, std::vector<double>{1e-9, 1e-7}));

    // gradient
    criteria = {};
    criteria.grad_tol = 1e-8;
    EXPECT_TRUE(quantape::math::stopGrad(criteria, std::vector<double>{1e-9, -1e-9}));
    EXPECT_TRUE(!quantape::math::stopGrad(criteria, std::vector<double>{1e-9, -1e-7}));
    criteria.grad_tol = 0.0;
    EXPECT_TRUE(!quantape::math::stopGrad(criteria, std::vector<double>{0.0, 0.0}));

    // budgets
    criteria = {};
    criteria.maxeval = 5;
    EXPECT_TRUE(quantape::math::stopEvals(criteria, 5));
    EXPECT_TRUE(!quantape::math::stopEvals(criteria, 4));
    criteria = {};
    EXPECT_TRUE(!quantape::math::stopTime(criteria, quantape::math::nowSeconds()));
    criteria.maxtime = 1e-6;
    const double start = quantape::math::nowSeconds() - 1.0;
    EXPECT_TRUE(quantape::math::stopTime(criteria, start));
    EXPECT_TRUE(std::string(quantape::math::to_string(
                    quantape::math::OptimizeResult::GradientTolReached)) == "GradientTolReached");
}

TEST_F(ValueGradTest, matchesAnalyticForVarAndFvar) {
    const Quadratic q;
    const std::vector<double> x0{0.5, -1.0, 2.0};

    {
        auto result = quantape::math::detail::valueGrad<var>(q, x0);
        CHECK_CLOSE("valueGrad<var> value", result.first, q.value(x0), 1e-12);
        const auto expected = q.grad(x0);
        for (std::size_t i = 0; i < x0.size(); ++i) {
            CHECK_CLOSE("valueGrad<var> grad", result.second[i], expected[i], 1e-12);
        }
    }
    {
        auto result = quantape::math::detail::valueGrad<fvar<var>>(q, x0);
        CHECK_CLOSE("valueGrad<fvar<var>> value", result.first, q.value(x0), 1e-12);
        const auto expected = q.grad(x0);
        for (std::size_t i = 0; i < x0.size(); ++i) {
            CHECK_CLOSE("valueGrad<fvar<var>> grad", result.second[i], expected[i], 1e-12);
        }
    }
}

TEST_F(OptimizationAdTest, hvpExactAndLinear) {
    const Quadratic q;
    const std::vector<double> x0{0.5, -1.0, 2.0};
    const std::vector<double> v{1.0, -2.0, 0.5};

    const auto hv = quantape::math::detail::hvp(q, x0, v);
    for (std::size_t i = 0; i < x0.size(); ++i) {
        const double expected = 2.0 * q.w[i] * v[i]; // H = diag(2 w_i)
        CHECK_CLOSE("hvp", hv[i], expected, 1e-12);
    }

    // linearity: H(a v + b u) = a Hv + b Hu
    const std::vector<double> u{0.0, 1.0, -1.0};
    const auto hu = quantape::math::detail::hvp(q, x0, u);
    for (std::size_t i = 0; i < x0.size(); ++i) {
        CHECK_CLOSE("hvp linearity", hv[i] - 0.5 * hu[i], 2.0 * q.w[i] * (v[i] - 0.5 * u[i]),
                    1e-12);
    }
}

TEST_F(ConstraintJacobianTest, valuesAndRowMajorJacobian) {
    const std::vector<double> x{0.6, 0.8};
    std::vector<double> c;
    std::vector<double> jacobian;

    {
        quantape::math::detail::constraintValueJacobian<var>(TwoConstraints{}, x, c, jacobian);
        EXPECT_TRUE(c.size() == 2 && jacobian.size() == 4);
        CHECK_CLOSE("constraint g0", c[0], 0.0, 1e-12);
        CHECK_CLOSE("constraint g1", c[1], -0.2, 1e-12);
        CHECK_CLOSE("constraint dg0/dx0", jacobian[0], 1.2, 1e-12);
        CHECK_CLOSE("constraint dg0/dx1", jacobian[1], 1.6, 1e-12);
        CHECK_CLOSE("constraint dg1/dx0", jacobian[2], 1.0, 1e-12);
        CHECK_CLOSE("constraint dg1/dx1", jacobian[3], -1.0, 1e-12);
    }
    {
        quantape::math::detail::constraintValueJacobian<fvar<var>>(TwoConstraints{}, x, c,
                                                                   jacobian);
        CHECK_CLOSE("constraint<fvar> dg0/dx0", jacobian[0], 1.2, 1e-12);
        CHECK_CLOSE("constraint<fvar> dg0/dx1", jacobian[1], 1.6, 1e-12);
        CHECK_CLOSE("constraint<fvar> dg1/dx1", jacobian[3], -1.0, 1e-12);
    }
}

TEST_F(LineSearchTest, wolfeZoomAndNonDescentPaths) {
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

    const quantape::math::LineSearchOptions options;
    const auto ls = quantape::math::wolfeLineSearch(eval, x, d, f0, g0, options);
    EXPECT_TRUE(ls.success);
    CHECK_CLOSE("line search alpha", ls.alpha, 1.0, 1e-12);
    CHECK_CLOSE("line search f", ls.f, 0.0, 1e-24);
    EXPECT_TRUE(ls.grad.size() == x.size());

    // Strong-Wolfe conditions hold at the accepted point
    const double dphi0 = dot(g0, d);
    EXPECT_TRUE(ls.f <= f0 + options.c1 * ls.alpha * dphi0 + 1e-15);
    EXPECT_TRUE(std::fabs(dot(ls.grad, d)) <= -options.c2 * dphi0 + 1e-15);

    // Tight c2 forces the bracket+zoom path: exact step at alpha = 5
    {
        quantape::math::LineSearchOptions tight = options;
        tight.c2 = 0.1;
        const std::vector<double> d_small{-0.2, -0.2, -0.2};
        const auto ls_zoom = quantape::math::wolfeLineSearch(eval, x, d_small, f0, g0, tight);
        EXPECT_TRUE(ls_zoom.success);
        CHECK_CLOSE("line search zoom alpha", ls_zoom.alpha, 5.0, 1e-4);
        CHECK_CLOSE("line search zoom f", ls_zoom.f, 0.0, 1e-8);
    }
    // Oversized direction: first trial violates Armijo, zoom shrinks to 0.25
    {
        const std::vector<double> d_big{-4.0, -4.0, -4.0}; // exact alpha = 0.25
        const auto ls_shrink = quantape::math::wolfeLineSearch(eval, x, d_big, f0, g0, options);
        EXPECT_TRUE(ls_shrink.success);
        CHECK_CLOSE("line search shrink alpha", ls_shrink.alpha, 0.25, 1e-12);
        CHECK_CLOSE("line search shrink f", ls_shrink.f, 0.0, 1e-24);
    }

    // Non-descent direction: no move, no success
    const std::vector<double> d_up{1.0, 1.0, 1.0};
    const auto bad = quantape::math::wolfeLineSearch(eval, x, d_up, f0, g0, options);
    EXPECT_TRUE(!bad.success);
    EXPECT_TRUE(bad.alpha == 0.0);
}

TEST_F(LbfgsAdTest, knownProblemsExitCodesAndSmallMemory) {
    const Quadratic q;
    const std::vector<double> start{0.0, 0.0, 0.0};

    // 1-D quadratic: the first Wolfe step lands exactly on the minimum
    {
        quantape::math::StopCriteria criteria;
        criteria.grad_tol = 1e-14;
        criteria.maxeval = 100;
        quantape::math::LBFGS<var> optimizer(criteria, 1);
        std::vector<double> x{0.0};
        EXPECT_TRUE(optimizer.minimize(Quadratic1D{}, x) ==
                    quantape::math::OptimizeResult::GradientTolReached);
        CHECK_CLOSE("lbfgs 1d x", x[0], 2.0, 1e-12);
    }
    // 3-D quadratic: fast convergence to the analytic minimizer
    {
        quantape::math::StopCriteria criteria;
        criteria.grad_tol = 1e-12;
        criteria.maxeval = 1000;
        quantape::math::LBFGS<var> optimizer(criteria);
        std::vector<double> x = start;
        quantape::math::OptimizerState state;
        EXPECT_TRUE(optimizer.minimize(q, x, state) ==
                    quantape::math::OptimizeResult::GradientTolReached);
        for (std::size_t i = 0; i < x.size(); ++i) {
            CHECK_CLOSE("lbfgs 3d x", x[i], q.c[i], 1e-8);
        }
        EXPECT_TRUE(state.iterations < 20);
        EXPECT_TRUE(state.evals > 0 && state.grad_evals > 0);
    }
    // Ill-conditioned weighted quadratic: memory earns its keep
    {
        quantape::math::StopCriteria criteria;
        criteria.grad_tol = 1e-10;
        criteria.maxeval = 5000;
        quantape::math::LBFGS<var> optimizer(criteria, 10);
        std::vector<double> x = start;
        EXPECT_TRUE(optimizer.minimize(WeightedQuadratic{}, x) ==
                    quantape::math::OptimizeResult::GradientTolReached);
        CHECK_CLOSE("lbfgs illcond x0", x[0], 1.0, 1e-6);
        CHECK_CLOSE("lbfgs illcond x1", x[1], -2.0, 1e-6);
        CHECK_CLOSE("lbfgs illcond x2", x[2], 0.5, 1e-6);
    }
    // Rosenbrock: classic curved-valley test at var
    {
        quantape::math::StopCriteria criteria;
        criteria.grad_tol = 1e-8;
        criteria.maxeval = 2000;
        quantape::math::LBFGS<var> optimizer(criteria, 10);
        std::vector<double> x{-1.2, 1.0};
        EXPECT_TRUE(optimizer.minimize(Rosenbrock{}, x) ==
                    quantape::math::OptimizeResult::GradientTolReached);
        CHECK_CLOSE("rosenbrock x", x[0], 1.0, 1e-5);
        CHECK_CLOSE("rosenbrock y", x[1], 1.0, 1e-5);
    }
    // `fvar<var>` backend drives the same loop through the value reverse path
    {
        quantape::math::StopCriteria criteria;
        criteria.grad_tol = 1e-10;
        criteria.maxeval = 1000;
        quantape::math::LBFGS<fvar<var>> optimizer(criteria);
        std::vector<double> x = start;
        EXPECT_TRUE(optimizer.minimize(q, x) == quantape::math::OptimizeResult::GradientTolReached);
        CHECK_CLOSE("lbfgs fvar x", x[0], q.c[0], 1e-6);
    }
    // maxeval exit
    {
        quantape::math::StopCriteria criteria;
        criteria.grad_tol = 0.0;
        criteria.maxeval = 5;
        quantape::math::LBFGS<var> optimizer(criteria);
        std::vector<double> x = start;
        EXPECT_TRUE(optimizer.minimize(q, x) == quantape::math::OptimizeResult::MaxEvalReached);
    }
    // ftol exit (gradient stop disabled, one pass suffices)
    {
        quantape::math::StopCriteria criteria;
        criteria.grad_tol = 0.0;
        criteria.ftol_abs = 1e-8;
        criteria.mtesf = 1;
        criteria.maxeval = 500;
        quantape::math::LBFGS<var> optimizer(criteria);
        std::vector<double> x = start;
        EXPECT_TRUE(optimizer.minimize(q, x) == quantape::math::OptimizeResult::FtolReached);
    }
    // xtol exit (gradient stop disabled, one pass suffices)
    {
        quantape::math::StopCriteria criteria;
        criteria.grad_tol = 0.0;
        criteria.xtol_abs = 1e-4;
        criteria.mtesx = 1;
        criteria.maxeval = 500;
        quantape::math::LBFGS<var> optimizer(criteria);
        std::vector<double> x = start;
        EXPECT_TRUE(optimizer.minimize(q, x) == quantape::math::OptimizeResult::XtolReached);
    }
    // Small memory: ring buffer wraps (m = 2 < n = 3) and still converges
    {
        quantape::math::StopCriteria criteria;
        criteria.grad_tol = 1e-10;
        criteria.maxeval = 1000;
        quantape::math::LBFGS<var> optimizer(criteria, 2);
        WeightedQuadratic ill;
        std::vector<double> x = start;
        EXPECT_TRUE(optimizer.minimize(ill, x) ==
                    quantape::math::OptimizeResult::GradientTolReached);
        CHECK_CLOSE("lbfgs ring x0", x[0], 1.0, 1e-6);
        CHECK_CLOSE("lbfgs ring x1", x[1], -2.0, 1e-6);
        CHECK_CLOSE("lbfgs ring x2", x[2], 0.5, 1e-6);
    }
    // memory inspector
    {
        quantape::math::LBFGS<var> optimizer;
        EXPECT_TRUE(optimizer.memory() == 0);
        optimizer.setMemory(7);
        EXPECT_TRUE(optimizer.memory() == 7);
    }
}

TEST_F(LbfgsDoubleTest, pairedCallbacksAndCallerSideStan) {
    // Concept surface: both paired callbacks qualify; the scalar-generic
    // shape remains valid for the base entry point as well.
    static_assert(quantape::math::ValueGradObjective<RosenbrockValueGrad>);
    static_assert(quantape::math::ValueGradObjective<RosenbrockValueGradAD>);
    static_assert(quantape::math::ObjectiveEvaluator<RosenbrockValueGrad, double>);
    static_assert(quantape::math::ObjectiveEvaluator<RosenbrockValueGradAD, double>);
    static_assert(quantape::math::ObjectiveEvaluator<Rosenbrock, double>);

    // Pure double with analytic gradient: no AD primitives involved
    {
        quantape::math::StopCriteria criteria;
        criteria.grad_tol = 1e-8;
        criteria.maxeval = 2000;
        quantape::math::LBFGS<double> optimizer(criteria, 10);
        std::vector<double> x{-1.2, 1.0};
        quantape::math::OptimizerState state;
        EXPECT_TRUE(optimizer.minimize(RosenbrockValueGrad{}, x, state) ==
                    quantape::math::OptimizeResult::GradientTolReached);
        CHECK_CLOSE("lbfgs<double> x", x[0], 1.0, 1e-5);
        CHECK_CLOSE("lbfgs<double> y", x[1], 1.0, 1e-5);
        EXPECT_TRUE(state.grad.size() == 2);
        EXPECT_TRUE(state.grad_evals > 0);
    }
    // Gradient supplied by caller-side Stan inside the callback
    {
        quantape::math::StopCriteria criteria;
        criteria.grad_tol = 1e-8;
        criteria.maxeval = 2000;
        quantape::math::LBFGS<double> optimizer(criteria, 10);
        std::vector<double> x{-1.2, 1.0};
        EXPECT_TRUE(optimizer.minimize(RosenbrockValueGradAD{}, x) ==
                    quantape::math::OptimizeResult::GradientTolReached);
        CHECK_CLOSE("lbfgs<double+ad> x", x[0], 1.0, 1e-5);
        CHECK_CLOSE("lbfgs<double+ad> y", x[1], 1.0, 1e-5);
    }
}

TEST_F(GradientSanityTest, matchesCentralDifferences) {
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
        CHECK_CLOSE("grad sanity value (analytic)", f, rosenbrockValue(x0), 1e-12);
        for (std::size_t i = 0; i < x0.size(); ++i) {
            const double fd = fd_gradient(i);
            CHECK_CLOSE("grad sanity double analytic", g[i], fd, 1e-5 * (1.0 + std::fabs(fd)));
        }
    }
    // 2) caller-side Stan callback (double mode)
    {
        std::vector<double> g;
        const double f = RosenbrockValueGradAD{}(x0, g);
        CHECK_CLOSE("grad sanity value (caller AD)", f, rosenbrockValue(x0), 1e-12);
        for (std::size_t i = 0; i < x0.size(); ++i) {
            const double fd = fd_gradient(i);
            CHECK_CLOSE("grad sanity double caller-AD", g[i], fd, 1e-5 * (1.0 + std::fabs(fd)));
        }
    }
    // 3) scalar-generic AD path (var backend) used by LBFGS<var>
    {
        const auto f = [](const std::vector<var>& x) {
            const var t1 = x[1] - x[0] * x[0];
            const var t2 = var(1.0) - x[0];
            return var(100.0) * t1 * t1 + t2 * t2;
        };
        const auto result = quantape::math::detail::valueGrad<var>(f, x0);
        CHECK_CLOSE("grad sanity value (var)", result.first, rosenbrockValue(x0), 1e-12);
        for (std::size_t i = 0; i < x0.size(); ++i) {
            const double fd = fd_gradient(i);
            CHECK_CLOSE("grad sanity var AD", result.second[i], fd, 1e-5 * (1.0 + std::fabs(fd)));
        }
    }
}

TEST_F(ConstraintsTest, qpBoundsAndPenaltyHelpers) {
    // Unconstrained: min 0.5 d'Bd + g'd -> B d = -g
    {
        quantape::math::QpProblem qp;
        qp.g = {-2.0, -4.0};
        qp.B = {2.0, 0.0, 0.0, 4.0};
        const auto res = quantape::math::solveActiveSetQp(qp);
        EXPECT_TRUE(res.success);
        CHECK_CLOSE("qp unconstrained d0", res.d[0], 1.0, 1e-12);
        CHECK_CLOSE("qp unconstrained d1", res.d[1], 1.0, 1e-12);
    }
    // Equality: min 0.5||d||^2 + g'd s.t. d0 + d1 = 1
    {
        quantape::math::QpProblem qp;
        qp.g = {-1.0, -1.0};
        qp.B = {1.0, 0.0, 0.0, 1.0};
        qp.A = {{1.0, 1.0}};
        qp.b = {-1.0};
        qp.equality = {1};
        const auto res = quantape::math::solveActiveSetQp(qp);
        EXPECT_TRUE(res.success);
        CHECK_CLOSE("qp equality d0", res.d[0], 0.5, 1e-12);
        CHECK_CLOSE("qp equality d1", res.d[1], 0.5, 1e-12);
        CHECK_CLOSE("qp equality lambda", res.lambda[0], 0.5, 1e-12);
    }
    // Active inequality: min 0.5||d||^2 - 2 d0 s.t. d0 <= 1
    {
        quantape::math::QpProblem qp;
        qp.g = {-2.0, 0.0};
        qp.B = {1.0, 0.0, 0.0, 1.0};
        qp.A = {{1.0, 0.0}};
        qp.b = {-1.0};
        qp.equality = {0};
        const auto res = quantape::math::solveActiveSetQp(qp);
        EXPECT_TRUE(res.success);
        CHECK_CLOSE("qp active d0", res.d[0], 1.0, 1e-10);
        CHECK_CLOSE("qp active d1", res.d[1], 0.0, 1e-10);
        CHECK_CLOSE("qp active lambda", res.lambda[0], 1.0, 1e-10);
    }
    // Inactive inequality
    {
        quantape::math::QpProblem qp;
        qp.g = {-0.5, 0.0};
        qp.B = {1.0, 0.0, 0.0, 1.0};
        qp.A = {{1.0, 0.0}};
        qp.b = {-1.0};
        qp.equality = {0};
        const auto res = quantape::math::solveActiveSetQp(qp);
        EXPECT_TRUE(res.success);
        CHECK_CLOSE("qp inactive d0", res.d[0], 0.5, 1e-10);
        CHECK_CLOSE("qp inactive lambda", res.lambda[0], 0.0, 1e-10);
    }

    const double inf = std::numeric_limits<double>::infinity();
    const quantape::math::Bounds bounds =
        quantape::math::Bounds::fromVectors({0.0, -inf}, {1.0, 5.0});
    EXPECT_TRUE(bounds.hasLower(0) && bounds.hasUpper(0));
    EXPECT_TRUE(!bounds.hasLower(1) && bounds.hasUpper(1));

    std::vector<double> x{2.0, 10.0};
    EXPECT_TRUE(!bounds.feasible(x));
    CHECK_CLOSE("bounds violation", bounds.violation(x), 5.0, 1e-15);
    bounds.project(x);
    CHECK_CLOSE("bounds project x0", x[0], 1.0, 1e-15);
    CHECK_CLOSE("bounds project x1", x[1], 5.0, 1e-15);
    EXPECT_TRUE(bounds.feasible(x));

    CHECK_CLOSE("max violation", quantape::math::maxViolation({-1.0, 0.25, 0.0}), 0.25, 1e-15);
    CHECK_CLOSE("l1 penalty", quantape::math::l1Penalty({0.5, -1.0, 0.0}, {2.0, -0.5}), 3.0, 1e-15);
}

TEST_F(SlSqpAdTest, constrainedSolversAndAugLagAgreement) {
    quantape::math::StopCriteria criteria;
    criteria.maxeval = 5000;

    // Equality-constrained quadratic: min ||x-(2,2)||^2 s.t. x0+x1=2 -> (1,1)
    {
        quantape::math::SLSQP<var> solver(criteria);
        std::vector<double> x{0.0, 0.0};
        const auto none = quantape::math::Bounds::unbounded(x.size());
        quantape::math::OptimizerState state;
        const auto result = solver.minimize(ShiftTarget{{2.0, 2.0}}, quantape::math::NoConstraint{},
                                            SumEqual{}, none, x, state);
        EXPECT_TRUE(result == quantape::math::OptimizeResult::Success);
        CHECK_CLOSE("slsqp eq x0", x[0], 1.0, 1e-7);
        CHECK_CLOSE("slsqp eq x1", x[1], 1.0, 1e-7);
        CHECK_CLOSE("slsqp eq f", state.f, 2.0, 1e-10);
    }
    // Active inequality: same objective s.t. x0+x1 <= 2 -> (1,1)
    {
        quantape::math::SLSQP<var> solver(criteria);
        std::vector<double> x{0.0, 0.0};
        const auto none = quantape::math::Bounds::unbounded(x.size());
        const auto result = solver.minimize(ShiftTarget{{2.0, 2.0}}, SumLeq{}, none, x);
        EXPECT_TRUE(result == quantape::math::OptimizeResult::Success);
        CHECK_CLOSE("slsqp active x0", x[0], 1.0, 1e-6);
        CHECK_CLOSE("slsqp active x1", x[1], 1.0, 1e-6);
    }
    // Inactive inequality: target already strictly feasible -> unconstrained min
    {
        quantape::math::SLSQP<var> solver(criteria);
        std::vector<double> x{-1.0, -1.0};
        const auto none = quantape::math::Bounds::unbounded(x.size());
        const auto result = solver.minimize(ShiftTarget{{0.5, 0.5}}, SumLeq{}, none, x);
        EXPECT_TRUE(result == quantape::math::OptimizeResult::Success);
        CHECK_CLOSE("slsqp inactive x0", x[0], 0.5, 1e-7);
        CHECK_CLOSE("slsqp inactive x1", x[1], 0.5, 1e-7);
    }
    // Bounds: min (x0-3)^2 s.t. x0 <= 1 -> x0 = 1
    {
        quantape::math::SLSQP<var> solver(criteria);
        std::vector<double> x{0.0};
        const auto bounds =
            quantape::math::Bounds::fromVectors({quantape::math::Bounds::kNoLower}, {1.0});
        const auto result =
            solver.minimize(ShiftTarget{{3.0}}, quantape::math::NoConstraint{}, bounds, x);
        EXPECT_TRUE(result == quantape::math::OptimizeResult::Success);
        CHECK_CLOSE("slsqp bound x0", x[0], 1.0, 1e-7);
    }
    // Arbitrage-style: violated market quotes -> feasible monotone butterfly-free fit
    {
        const std::vector<double> market{1.0, 1.5, 1.0}; // violates monotonicity and butterfly
        quantape::math::SLSQP<var> solver(criteria);
        std::vector<double> p = market;
        const auto none = quantape::math::Bounds::unbounded(p.size());
        quantape::math::OptimizerState state;
        const auto result = solver.minimize(ShiftTarget{market}, ArbConstraints{}, none, p, state);
        EXPECT_TRUE(result == quantape::math::OptimizeResult::Success);
        EXPECT_TRUE(p[0] >= p[1] - 1e-7);
        EXPECT_TRUE(p[1] >= p[2] - 1e-7);
        EXPECT_TRUE(p[0] - 2.0 * p[1] + p[2] >= -1e-7);
        double fit = 0.0;
        for (std::size_t i = 0; i < p.size(); ++i) {
            const double d = p[i] - market[i];
            fit += d * d;
        }
        EXPECT_TRUE(fit <= 0.25 + 1e-9); // better than the equal-price feasible point (1,1,1)
    }
    // Infeasible system: x0 <= 0 and x0 >= 1
    {
        quantape::math::SLSQP<var> solver(criteria);
        std::vector<double> x{0.5};
        const auto none = quantape::math::Bounds::unbounded(x.size());
        const auto result = solver.minimize(ShiftTarget{{0.0}}, TwoInfeasible{}, none, x);
        EXPECT_TRUE(result == quantape::math::OptimizeResult::Infeasible);
    }
    // Double mode with paired objective/constraint callbacks (no AD)
    {
        quantape::math::SLSQP<double> solver(criteria);
        std::vector<double> x{0.0, 0.0};
        const auto none = quantape::math::Bounds::unbounded(x.size());
        const auto result =
            solver.minimize(ShiftTargetValueGrad{{2.0, 2.0}}, SumLeqValueJac{}, none, x);
        EXPECT_TRUE(result == quantape::math::OptimizeResult::Success);
        CHECK_CLOSE("slsqp<double> x0", x[0], 1.0, 1e-6);
        CHECK_CLOSE("slsqp<double> x1", x[1], 1.0, 1e-6);
    }

    // ── AugLag on the same constraint families ──
    quantape::math::StopCriteria augCriteria;
    augCriteria.maxeval = 100000;

    // Equality-constrained quadratic: min ||x-(2,2)||^2 s.t. x0+x1=2 -> (1,1)
    {
        quantape::math::AugLag<var> solver(augCriteria);
        std::vector<double> x{0.0, 0.0};
        const auto none = quantape::math::Bounds::unbounded(x.size());
        const auto result = solver.minimize(ShiftTarget{{2.0, 2.0}}, quantape::math::NoConstraint{},
                                            SumEqual{}, none, x);
        EXPECT_TRUE(result == quantape::math::OptimizeResult::Success);
        CHECK_CLOSE("auglag eq x0", x[0], 1.0, 1e-5);
        CHECK_CLOSE("auglag eq x1", x[1], 1.0, 1e-5);
    }
    // Active inequality -> (1,1)
    {
        quantape::math::AugLag<var> solver(augCriteria);
        std::vector<double> x{0.0, 0.0};
        const auto none = quantape::math::Bounds::unbounded(x.size());
        const auto result = solver.minimize(ShiftTarget{{2.0, 2.0}}, SumLeq{}, none, x);
        EXPECT_TRUE(result == quantape::math::OptimizeResult::Success);
        CHECK_CLOSE("auglag active x0", x[0], 1.0, 1e-5);
        CHECK_CLOSE("auglag active x1", x[1], 1.0, 1e-5);
    }
    // Inactive inequality -> unconstrained target
    {
        quantape::math::AugLag<var> solver(augCriteria);
        std::vector<double> x{-1.0, -1.0};
        const auto none = quantape::math::Bounds::unbounded(x.size());
        const auto result = solver.minimize(ShiftTarget{{0.5, 0.5}}, SumLeq{}, none, x);
        EXPECT_TRUE(result == quantape::math::OptimizeResult::Success);
        CHECK_CLOSE("auglag inactive x0", x[0], 0.5, 1e-5);
        CHECK_CLOSE("auglag inactive x1", x[1], 0.5, 1e-5);
    }
    // Bounds (penalized, projected-gradient KKT): min (x0-3)^2 s.t. x0 <= 1
    {
        quantape::math::AugLag<var> solver(augCriteria);
        std::vector<double> x{0.0};
        const auto bounds =
            quantape::math::Bounds::fromVectors({quantape::math::Bounds::kNoLower}, {1.0});
        const auto result =
            solver.minimize(ShiftTarget{{3.0}}, quantape::math::NoConstraint{}, bounds, x);
        EXPECT_TRUE(result == quantape::math::OptimizeResult::Success);
        CHECK_CLOSE("auglag bound x0", x[0], 1.0, 1e-5);
    }
    // Double mode with paired callbacks
    {
        quantape::math::AugLag<double> solver(augCriteria);
        std::vector<double> x{0.0, 0.0};
        const auto none = quantape::math::Bounds::unbounded(x.size());
        const auto result =
            solver.minimize(ShiftTargetValueGrad{{2.0, 2.0}}, SumLeqValueJac{}, none, x);
        EXPECT_TRUE(result == quantape::math::OptimizeResult::Success);
        CHECK_CLOSE("auglag<double> x0", x[0], 1.0, 1e-5);
        CHECK_CLOSE("auglag<double> x1", x[1], 1.0, 1e-5);
    }
    // Agreement with SLSQP on the same constrained problem
    {
        quantape::math::StopCriteria sc;
        sc.maxeval = 100000;
        quantape::math::SLSQP<var> slsqp(sc);
        quantape::math::AugLag<var> auglag(sc);
        std::vector<double> x_slsqp{0.0, 0.0};
        std::vector<double> x_auglag{0.0, 0.0};
        const auto none = quantape::math::Bounds::unbounded(2);
        EXPECT_TRUE(slsqp.minimize(ShiftTarget{{2.0, 2.0}}, SumLeq{}, none, x_slsqp) ==
                    quantape::math::OptimizeResult::Success);
        EXPECT_TRUE(auglag.minimize(ShiftTarget{{2.0, 2.0}}, SumLeq{}, none, x_auglag) ==
                    quantape::math::OptimizeResult::Success);
        for (std::size_t i = 0; i < x_slsqp.size(); ++i) {
            CHECK_CLOSE("slsqp vs auglag", x_auglag[i], x_slsqp[i], 1e-4);
        }
    }
}

TEST_F(TNewtonAdTest, rosenbrockQuadraticFvarHimmelblauMaxEvalAndNegativeCurvature) {
    quantape::math::StopCriteria criteria;
    criteria.maxeval = 100000;

    // 2-D Rosenbrock at var
    {
        quantape::math::TNewton<var> solver(criteria);
        std::vector<double> x{-1.2, 1.0};
        quantape::math::OptimizerState state;
        EXPECT_TRUE(solver.minimize(Rosenbrock{}, x, state) ==
                    quantape::math::OptimizeResult::GradientTolReached);
        CHECK_CLOSE("tnewton rosenbrock x0", x[0], 1.0, 1e-8);
        CHECK_CLOSE("tnewton rosenbrock x1", x[1], 1.0, 1e-8);
    }
    // 3-D quadratic: exact Newton directions, fast convergence
    {
        const Quadratic q;
        quantape::math::TNewton<var> solver(criteria);
        std::vector<double> x{0.0, 0.0, 0.0};
        quantape::math::OptimizerState state;
        EXPECT_TRUE(solver.minimize(q, x, state) ==
                    quantape::math::OptimizeResult::GradientTolReached);
        for (std::size_t i = 0; i < x.size(); ++i) {
            CHECK_CLOSE("tnewton quadratic x", x[i], q.c[i], 1e-9);
        }
    }
    // `fvar<var>` backend drives the same path
    {
        quantape::math::TNewton<fvar<var>> solver(criteria);
        std::vector<double> x{-1.2, 1.0};
        EXPECT_TRUE(solver.minimize(Rosenbrock{}, x) ==
                    quantape::math::OptimizeResult::GradientTolReached);
        CHECK_CLOSE("tnewton fvar x0", x[0], 1.0, 1e-8);
        CHECK_CLOSE("tnewton fvar x1", x[1], 1.0, 1e-8);
    }
    // Exact Newton converges quadratically on a well-scaled quadratic and
    // on Himmelblau (Hessian 2I at the start)
    {
        quantape::math::TNewton<var> solver(criteria);
        std::vector<double> x{0.0, 0.0};
        const auto r = solver.minimize(Himmelblau{}, x);
        EXPECT_TRUE(r == quantape::math::OptimizeResult::GradientTolReached);
        const double f = Himmelblau{}(x);
        EXPECT_TRUE(f < 1e-12);
    }
    // maxeval exit
    {
        quantape::math::StopCriteria tight;
        tight.grad_tol = 0.0;
        tight.maxeval = 5;
        quantape::math::TNewton<var> solver(tight);
        std::vector<double> x{-1.2, 1.0};
        EXPECT_TRUE(solver.minimize(Rosenbrock{}, x) ==
                    quantape::math::OptimizeResult::MaxEvalReached);
    }
    // Negative curvature on the first CG step: BOTH modes fall back to
    // steepest descent (PNET iterd = 0) and converge to a quartic minimum
    // (NLopt parity: plain TNEWTON and TNEWTON_RESTART both succeed here)
    {
        // Flat quartic minimum: on x86 the line search can hit roundoff before
        // the 1e-10 gradient tolerance is distinguishable from zero there, so
        // both successful stop codes are accepted; the solution assertions
        // below are the real gate (NLopt parity: both modes succeed).
        const auto convergedQuartic = [](quantape::math::OptimizeResult r) {
            return r == quantape::math::OptimizeResult::GradientTolReached ||
                   r == quantape::math::OptimizeResult::RoundoffLimited;
        };

        quantape::math::TNewton<var> plain(criteria, /*restart=*/false);
        std::vector<double> x0{0.2, 0.0};
        EXPECT_TRUE(convergedQuartic(plain.minimize(QuarticNonconvex{}, x0)));
        CHECK_CLOSE("tnewton quartic x0", std::fabs(x0[0]), 0.70710678, 1e-6);
        CHECK_CLOSE("tnewton quartic x1", x0[1], 0.0, 1e-6);

        quantape::math::TNewton<var> restarting(criteria, /*restart=*/true);
        std::vector<double> x1{0.2, 0.0};
        EXPECT_TRUE(convergedQuartic(restarting.minimize(QuarticNonconvex{}, x1)));
        CHECK_CLOSE("tnewton quartic restart x0", std::fabs(x1[0]), 0.70710678, 1e-6);
    }
}

TEST_F(OptimizerBaseTest, scalarBackendsExitPathsAndEmptyVectorRejected) {
    const Quadratic q;
    const std::vector<double> start{0.0, 0.0, 0.0};

    // double backend: value only, no gradient
    {
        ValueProbe<double> probe;
        std::vector<double> x = start;
        quantape::math::OptimizerState state;
        EXPECT_TRUE(probe.minimize(q, x, state) == quantape::math::OptimizeResult::Success);
        CHECK_CLOSE("double backend value", state.f, q.value(start), 1e-14);
        EXPECT_TRUE(state.grad.empty());
        EXPECT_TRUE(state.evals == 1);
        EXPECT_TRUE(x == start);
    }
    // var backend: full convergence to the analytic minimizer
    {
        quantape::math::StopCriteria criteria;
        criteria.grad_tol = 1e-12;
        criteria.maxeval = 100000;
        SteepestDescent<var> optimizer(criteria, 0.1);
        std::vector<double> x = start;
        quantape::math::OptimizerState state;
        EXPECT_TRUE(optimizer.minimize(q, x, state) ==
                    quantape::math::OptimizeResult::GradientTolReached);
        EXPECT_TRUE(state.evals > 0 && state.grad_evals > 0 && state.iterations > 0);
        for (std::size_t i = 0; i < x.size(); ++i) {
            CHECK_CLOSE("var backend x", x[i], q.c[i], 1e-6);
        }
    }
    // fvar backend drives the same path through val_ reverse
    {
        quantape::math::StopCriteria criteria;
        criteria.grad_tol = 1e-10;
        criteria.maxeval = 100000;
        SteepestDescent<fvar<var>> optimizer(criteria, 0.1);
        std::vector<double> x = start;
        EXPECT_TRUE(optimizer.minimize(q, x) == quantape::math::OptimizeResult::GradientTolReached);
        for (std::size_t i = 0; i < x.size(); ++i) {
            CHECK_CLOSE("fvar backend x", x[i], q.c[i], 1e-5);
        }
    }
    // maxeval exit
    {
        quantape::math::StopCriteria criteria;
        criteria.grad_tol = 0.0;
        criteria.maxeval = 5;
        SteepestDescent<var> optimizer(criteria, 0.1);
        std::vector<double> x = start;
        EXPECT_TRUE(optimizer.minimize(q, x) == quantape::math::OptimizeResult::MaxEvalReached);
    }
    // stopval exit
    {
        quantape::math::StopCriteria criteria;
        criteria.grad_tol = 0.0;
        criteria.maxeval = 100000;
        criteria.stopval = 1e-6;
        SteepestDescent<var> optimizer(criteria, 0.1);
        std::vector<double> x = start;
        EXPECT_TRUE(optimizer.minimize(q, x) == quantape::math::OptimizeResult::StopvalReached);
    }
    // xtol exit
    {
        quantape::math::StopCriteria criteria;
        criteria.grad_tol = 0.0;
        criteria.xtol_abs = 1e-3;
        criteria.maxeval = 100000;
        SteepestDescent<var> optimizer(criteria, 0.1);
        std::vector<double> x = start;
        EXPECT_TRUE(optimizer.minimize(q, x) == quantape::math::OptimizeResult::XtolReached);
    }
    // empty parameter vector rejected
    {
        SteepestDescent<var> optimizer;
        std::vector<double> empty;
        EXPECT_THROW(optimizer.minimize(q, empty), std::invalid_argument);
    }
}

// ── QP solver regressions: infeasible starts, phase-I and KKT gates ───────
//
// The previous active-set implementation seeded its working set with every
// constraint violated at d = 0.  That made the initial KKT system singular
// whenever more than n constraints were violated, and the "dependent row"
// recovery could drop a violated row for good, so the solver could return
// success with a primal-infeasible point.  These cases pin the two-phase
// redesign (LP phase I + feasible-start active set + explicit KKT gate).

TEST(QpSolverTest, oneDimensionalViolatedStartReturnsFeasibleOptimum) {
    quantape::math::QpProblem qp;
    qp.g = {0.5935};
    qp.B = {0.5888};
    qp.A = {0.0794, 0.4366};
    qp.b = {0.8543, 0.9473};
    qp.equality = {0, 0};

    const auto result = quantape::math::solveActiveSetQp(qp);
    ASSERT_TRUE(result.success);

    // True optimum: row 0 binds, d = -b0 / A0 (the old solver returned
    // d = -2.1697 with row-0 residual +0.682).
    const double expected_d = -0.8543 / 0.0794;
    CHECK_CLOSE_REL("1-D violated-start primal", result.d[0], expected_d, 1e-8);
    const double expected_lambda0 = -(0.5888 * expected_d + 0.5935) / 0.0794;
    CHECK_CLOSE_REL("1-D violated-start multiplier", result.lambda[0], expected_lambda0, 1e-7);
    CHECK_CLOSE("1-D violated-start inactive multiplier", result.lambda[1], 0.0, 1e-14);
    EXPECT_TRUE(qpFeasibleAt(qp, result.d, 1e-9));
    CHECK_CLOSE("1-D violated-start stationarity",
                result.d[0] * qp.B[0] + qp.g[0] + result.lambda[0] * qp.A[0], 0.0, 1e-10);
}

TEST(QpSolverTest, conflictingConstraintsReportFailure) {
    // d <= 0 and -d + 1 <= 0 (i.e. d >= 1): no primal-feasible point.
    quantape::math::QpProblem qp;
    qp.g = {0.0};
    qp.B = {1.0};
    qp.A = {1.0, -1.0};
    qp.b = {0.0, 1.0};
    qp.equality = {0, 0};
    const auto result = quantape::math::solveActiveSetQp(qp);
    EXPECT_TRUE(!result.success);
}

TEST(QpSolverTest, redundantRowsAndEqualitiesStayRegular) {
    // x + y = 1 enforced twice (consistent dependent equalities) plus the
    // implied inequality x + y <= 2.  The KKT systems are singular unless the
    // working set keeps a maximal independent subset.
    quantape::math::QpProblem qp;
    qp.g = {-2.0, -2.0};
    qp.B = {1.0, 0.0, 0.0, 1.0};
    qp.A = {1.0, 1.0, 2.0, 2.0, 1.0, 1.0};
    qp.b = {-1.0, -2.0, -2.0};
    qp.equality = {1, 1, 0};
    const auto result = quantape::math::solveActiveSetQp(qp);
    ASSERT_TRUE(result.success);
    CHECK_CLOSE_REL("redundant equality d0", result.d[0], 0.5, 1e-9);
    CHECK_CLOSE_REL("redundant equality d1", result.d[1], 0.5, 1e-9);
}

TEST(QpSolverTest, randomizedDifferentialAgainstActiveSetEnumeration) {
    QpRandom rng(0x5eed1234u);
    int feasible_cases = 0;
    int infeasible_cases = 0;
    constexpr int kCases = 3000;
    for (int c = 0; c < kCases; ++c) {
        const int n = rng.integer(1, 3);
        const int m = rng.integer(0, 6);
        const quantape::math::QpProblem qp = makeRandomQp(rng, n, m);
        const auto result = quantape::math::solveActiveSetQp(qp);
        std::vector<double> reference;
        const bool reference_feasible = qpBruteForce(qp, reference);
        SCOPED_TRACE(::testing::Message()
                     << "random QP case " << c << " (n=" << n << ", m=" << m << ")");
        if (!reference_feasible) {
            ++infeasible_cases;
            EXPECT_TRUE(!result.success);
            continue;
        }
        ++feasible_cases;
        if (!result.success) {
            EXPECT_TRUE(result.success);
            continue;
        }
        EXPECT_TRUE(qpFeasibleAt(qp, result.d, 1e-7));
        for (std::size_t j = 0; j < result.d.size(); ++j) {
            CHECK_CLOSE_REL("random QP primal component", result.d[j], reference[j], 1e-6, 1e-9);
        }
        CHECK_CLOSE_REL("random QP objective", qpObjectiveValue(qp, result.d),
                        qpObjectiveValue(qp, reference), 1e-7, 1e-9);
    }
    // Both branches must be exercised materially, otherwise the test proves
    // little about the phase-I infeasibility detection.
    EXPECT_GT(feasible_cases, 500);
    EXPECT_GT(infeasible_cases, 500);
}

TEST(QpSolverTest, rowOrderPermutationInvariance) {
    QpRandom rng(0xabcdef01u);
    int solved_cases = 0;
    constexpr int kCases = 1000;
    for (int c = 0; c < kCases; ++c) {
        const int n = rng.integer(1, 3);
        const int m = rng.integer(1, 6);
        const quantape::math::QpProblem qp = makeRandomQp(rng, n, m);
        std::vector<std::size_t> order(static_cast<std::size_t>(m));
        for (std::size_t i = 0; i < order.size(); ++i) {
            order[i] = i;
        }
        std::shuffle(order.begin(), order.end(), rng.gen);
        quantape::math::QpProblem permuted;
        permuted.g = qp.g;
        permuted.B = qp.B;
        permuted.A.assign(qp.A.size(), 0.0);
        permuted.b.resize(static_cast<std::size_t>(m));
        permuted.equality.assign(static_cast<std::size_t>(m), 0);
        for (int i = 0; i < m; ++i) {
            const std::size_t source = order[static_cast<std::size_t>(i)];
            for (int j = 0; j < n; ++j) {
                permuted.A[static_cast<std::size_t>(i * n + j)] = qp.A[static_cast<std::size_t>(
                    source * static_cast<std::size_t>(n) + static_cast<std::size_t>(j))];
            }
            permuted.b[static_cast<std::size_t>(i)] = qp.b[source];
            permuted.equality[static_cast<std::size_t>(i)] = qp.equality[source];
        }

        const auto base = quantape::math::solveActiveSetQp(qp);
        const auto shuffled = quantape::math::solveActiveSetQp(permuted);
        SCOPED_TRACE(::testing::Message()
                     << "permuted QP case " << c << " (n=" << n << ", m=" << m << ")");
        ASSERT_EQ(base.success, shuffled.success);
        if (!base.success) {
            continue;
        }
        ++solved_cases;
        for (std::size_t j = 0; j < base.d.size(); ++j) {
            CHECK_CLOSE_REL("row-order permutation primal", shuffled.d[j], base.d[j], 1e-6, 1e-9);
        }
    }
    EXPECT_GT(solved_cases, 400);
}

TEST(SlSqpQpRegressionTest, threeViolatedConstraintsAtStartBecomeFeasible) {
    // n = 1 with three constraints all violated at x0 = 0, feasible set
    // x0 >= 2.0358.  The old QP recovery dropped at most one of the three
    // dependent violated rows and the subproblem was declared infeasible
    // (OptimizeResult::Infeasible, 0 iterations).
    quantape::math::StopCriteria criteria;
    criteria.maxeval = 5000;
    quantape::math::SLSQP<double> solver(criteria);
    std::vector<double> x{0.0};
    quantape::math::OptimizerState state;
    const auto result = solver.minimize(ShiftTargetValueGrad{{0.0}}, ThreeLowerBoundsValueJac{},
                                        quantape::math::Bounds::unbounded(1), x, state);
    EXPECT_TRUE(result == quantape::math::OptimizeResult::Success);
    CHECK_CLOSE_REL("SLSQP three violated constraints x0", x[0], 2.0358, 1e-7);
    EXPECT_TRUE(state.iterations > 0);
    EXPECT_TRUE(state.ineq_multipliers.size() == 3);
    for (double mu : state.ineq_multipliers) {
        EXPECT_TRUE(mu >= -1e-8);
    }
}
