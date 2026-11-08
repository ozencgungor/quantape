// test_solvers.cpp — all six 1-D root finders and the tridiagonal solver.

#include "quantape/math/NumericalMethods.h"

#include <cmath>
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#include "support/GtestSupport.h"

using namespace quantape::math;

namespace {

struct Case {
    const char* name;
    double (*f)(double);
    double lo, hi, guess, root;
};

template <typename SolverT>
void runSolver(const char* solverName, const Case& c, double tol = 1e-9) {
    SolverT solver;
    solver.setMaxEvaluations(200);
    double got = 0.0;
    try {
        got = solver.solve(c.f, 1e-12, c.guess, c.lo, c.hi);
    } catch (const std::exception& e) {
        ADD_FAILURE() << solverName << " / " << c.name << " threw: " << e.what();
        return;
    }
    CHECK_CLOSE(std::string(solverName) + " / " + c.name, got, c.root, tol);
}

/// Same as runSolver but leaves the solver's default evaluation budget (100)
/// in place, which is the budget the library's call sites use.
template <typename SolverT>
void runSolverWithDefaultBudget(const char* solverName, const Case& c, double tol = 1e-9) {
    SolverT solver;
    double got = 0.0;
    try {
        got = solver.solve(c.f, 1e-12, c.guess, c.lo, c.hi);
    } catch (const std::exception& e) {
        ADD_FAILURE() << solverName << " / " << c.name << " threw: " << e.what();
        return;
    }
    CHECK_CLOSE(std::string(solverName) + " / " + c.name, got, c.root, tol);
}

struct CountingFn {
    double (*f)(double);
    std::size_t* count;
    double operator()(double x) const {
        ++*count;
        return f(x);
    }
};

/// Root-finding probe with an explicit budget/accuracy; reports a throw as a
/// failure instead of letting it escape the test body.
template <typename SolverT>
void expectRoot(const char* label, double (*f)(double), double guess, double lo, double hi,
                double expected, double tol, std::size_t maxEvaluations = 100,
                double accuracy = 1e-12) {
    SolverT solver;
    solver.setMaxEvaluations(maxEvaluations);
    try {
        const double got = solver.solve(f, accuracy, guess, lo, hi);
        CHECK_CLOSE(label, got, expected, tol);
    } catch (const std::exception& e) {
        ADD_FAILURE() << label << " threw: " << e.what();
    }
}

/// f(x) = 1e6 (x - r)^3 + 1e-3 (x - r) with r = 0.5123: near the root the
/// residual is tiny over a wide neighbourhood, so an |f| ~ 0 test that is not
/// scaled by the local slope stops far short of `accuracy`.
constexpr double kCubicShiftRoot = 0.5123;
double cubicShift(double x) {
    const double d = x - kCubicShiftRoot;
    return 1e6 * d * d * d + 1e-3 * d;
}

/// f(x) = 1e-15 (x - 0.5): every |f| on [0, 1] is below the old absolute
/// ~9.3e-15 floor, yet the root is 0.5 away from xMin.
double tinyScaleLinear(double x) {
    return 1e-15 * (x - 0.5);
}

/// f(x) = exp(x - 0.7) - 1 - 0.3 (x - 0.7), root at x = 0.7: convex on the
/// left, so plain regula falsi keeps the right end stuck and exhausts a
/// 100-evaluation budget.
double falsePositionStagnation(double x) {
    return std::exp(x - 0.7) - 1.0 - 0.3 * (x - 0.7);
}

/// The shared golden table (every solver must reproduce these roots).
const Case* goldenCases(std::size_t& count) {
    static const Case cases[] = {
        {"x^2-2", [](double x) { return x * x - 2.0; }, 0.0, 3.0, 1.5, 1.4142135623730951},
        {"cos(x)-x", [](double x) { return std::cos(x) - x; }, 0.0, 1.0, 0.5, 0.7390851332151607},
        {"e^x-3x", [](double x) { return std::exp(x) - 3.0 * x; }, 0.1, 0.9, 0.3,
         0.6190612867359451},
        {"x^3-2x-5", [](double x) { return std::pow(x, 3) - 2.0 * x - 5.0; }, 2.0, 3.0, 2.5,
         2.0945514815423265},
        {"log(x)-1", [](double x) { return std::log(x) - 1.0; }, 0.5, 3.0, 2.0, 2.7182818284590452},
    };
    count = sizeof(cases) / sizeof(cases[0]);
    return cases;
}

} // namespace

TEST(Solvers, goldenRootsMatchAnalytic) {
    std::size_t count = 0;
    const Case* cases = goldenCases(count);

    for (std::size_t i = 0; i < count; ++i) {
        const Case& c = cases[i];
        runSolver<BisectionSolver<double>>("bisection", c);
        runSolver<BrentSolver<double>>("brent", c);
        runSolver<SecantSolver<double>>("secant", c);
        runSolver<FalsePositionSolver<double>>("falsepos", c);
        runSolver<RidderSolver<double>>("ridder", c);
        runSolver<NewtonSolver<double>>("newton-fd", c);
    }
}

TEST(Solvers, goldenRootsFitDefaultBudget) {
    // The library's call sites (curve bootstraps) use the solver defaults:
    // 100 evaluations and accuracy 1e-12. Every solver must still find these
    // roots without raising the budget.
    std::size_t count = 0;
    const Case* cases = goldenCases(count);

    for (std::size_t i = 0; i < count; ++i) {
        const Case& c = cases[i];
        runSolverWithDefaultBudget<BisectionSolver<double>>("bisection", c);
        runSolverWithDefaultBudget<BrentSolver<double>>("brent", c);
        runSolverWithDefaultBudget<SecantSolver<double>>("secant", c);
        runSolverWithDefaultBudget<FalsePositionSolver<double>>("falsepos", c);
        runSolverWithDefaultBudget<RidderSolver<double>>("ridder", c);
        runSolverWithDefaultBudget<NewtonSolver<double>>("newton-fd", c);
    }
}

TEST(Solvers, brentInterpolatesInsteadOfBisecting) {
    // Regression for the zeroin-formula bug: with a broken secant/IQI branch
    // Brent silently degraded to bisection and used the same eval count.
    auto f = [](double x) { return std::cos(x) - x; };

    std::size_t bisectionEvals = 0;
    std::size_t brentEvals = 0;

    {
        BisectionSolver<double> solver;
        solver.setMaxEvaluations(200);
        const CountingFn counted{f, &bisectionEvals};
        solver.solve(counted, 1e-12, 0.5, 0.0, 1.0);
    }
    {
        BrentSolver<double> solver;
        solver.setMaxEvaluations(200);
        const CountingFn counted{f, &brentEvals};
        solver.solve(counted, 1e-12, 0.5, 0.0, 1.0);
    }

    EXPECT_LE(brentEvals * 2, bisectionEvals) << "brent must use at most half the bisection evals";
    EXPECT_LE(brentEvals, 20);
}

TEST(Solvers, newtonExplicitAndMemberDerivative) {
    auto f = [](double x) { return x * x - 2.0; };
    auto df = [](double x) { return 2.0 * x; };
    const double expected = std::sqrt(2.0);

    NewtonSolverWithDerivative<double, decltype(df)> solver(df);
    solver.setMaxEvaluations(100);
    const double got = solver.solve(f, 1e-12, 1.5, 1.0, 2.0);
    CHECK_CLOSE("newton-explicit / x^2-2", got, expected, 1e-12);

    // An objective carrying its own derivative() must be used automatically
    struct WithMember {
        double operator()(double x) const { return x * x - 2.0; }
        double derivative(double x) const { return 2.0 * x; }
    };
    NewtonSolver<double> memberSolver;
    memberSolver.setMaxEvaluations(100);
    const double gotMember = memberSolver.solve(WithMember{}, 1e-12, 1.5, 1.0, 2.0);
    CHECK_CLOSE("newton-member / x^2-2", gotMember, expected, 1e-12);
}

TEST(Solvers, autoBracketingFromGuess) {
    auto f = [](double x) { return x * x * x - x - 2.0; };
    BrentSolver<double> solver;
    solver.setMaxEvaluations(200);
    const double got = solver.solve(f, 1e-12, 1.5, 0.1);
    CHECK_CLOSE("brent / x^3-x-2 from guess+step", got, 1.5213797068045676, 1e-9);
}

TEST(Solvers, tridiagonalKnownSolution) {
    const std::vector<double> a{0.0, 1.0, 1.0};
    const std::vector<double> b{2.0, 2.0, 2.0};
    const std::vector<double> c{1.0, 1.0, 0.0};
    const std::vector<double> d{1.0, 2.0, 3.0};
    const std::vector<double> expected{0.5, 0.0, 1.5};

    const std::vector<double> x = TridiagonalSolver<double>::solve(a, b, c, d);
    for (std::size_t i = 0; i < expected.size(); ++i) {
        CHECK_CLOSE("thomas / x" + std::to_string(i), x[i], expected[i], 1e-14);
    }
}

TEST(Solvers, rejectsInvalidInputs) {
    auto flat = [](double x) { return x * x + 1.0; };
    auto linear = [](double x) { return x - 0.5; };

    SCOPED_TRACE("accuracy <= 0");
    EXPECT_THROW(BrentSolver<double>().solve(linear, 0.0, 0.25, 0.0, 1.0), std::exception);

    SCOPED_TRACE("not bracketed");
    EXPECT_THROW(BrentSolver<double>().solve(flat, 1e-10, 0.5, 0.0, 1.0), std::exception);

    SCOPED_TRACE("guess outside bracket");
    EXPECT_THROW(BrentSolver<double>().solve(linear, 1e-10, 2.0, 0.0, 1.0), std::exception);

    SCOPED_TRACE("xMin >= xMax");
    EXPECT_THROW(BrentSolver<double>().solve(linear, 1e-10, 0.5, 1.0, 0.0), std::exception);

    SCOPED_TRACE("tridiagonal size mismatch");
    EXPECT_THROW(
        TridiagonalSolver<double>::solve({0.0, 1.0}, {2.0, 2.0, 2.0}, {1.0, 0.0}, {1.0, 2.0}),
        std::exception);

    SCOPED_TRACE("tridiagonal zero pivot");
    EXPECT_THROW(TridiagonalSolver<double>::solve({0.0, 1.0}, {1.0, 1.0}, {1.0, 0.0}, {1.0, 2.0}),
                 std::exception);
}

TEST(Solvers, accuracyContractBeatsTinyResidual) {
    // Regression for the |f| ~ 0 early exit. f(x) = 1e6 (x-r)^3 + 1e-3 (x-r)
    // has a residual below the old fScale-derived threshold while still being
    // ~5e-7 (Brent) / ~1.4e-6 (bisection) away from the root. The accuracy
    // contract (bracket width / step size, here 1e-12) must win.
    SCOPED_TRACE("cubic-shift");
    expectRoot<BrentSolver<double>>("brent / tiny residual", cubicShift, 0.5, 0.0, 1.0,
                                    kCubicShiftRoot, 1e-10);
    expectRoot<BisectionSolver<double>>("bisection / tiny residual", cubicShift, 0.5, 0.0, 1.0,
                                        kCubicShiftRoot, 1e-10);
    expectRoot<SecantSolver<double>>("secant / tiny residual", cubicShift, 0.5, 0.0, 1.0,
                                     kCubicShiftRoot, 1e-10);
    expectRoot<FalsePositionSolver<double>>("falsepos / tiny residual", cubicShift, 0.5, 0.0, 1.0,
                                            kCubicShiftRoot, 1e-10);
    expectRoot<RidderSolver<double>>("ridder / tiny residual", cubicShift, 0.5, 0.0, 1.0,
                                     kCubicShiftRoot, 1e-10);
}

TEST(Solvers, belowUnitResidualIsNotARoot) {
    // Regression for fScale still being 1.0 at the initial zero checks: any
    // |f| <= 42*eps (~9.3e-15) was accepted as an absolute zero. Here the whole
    // bracket has |f| <= 5e-16, so every solver returned xMin = 0 (error 0.5).
    SCOPED_TRACE("tiny-scale linear");
    expectRoot<BrentSolver<double>>("brent / tiny |f|", tinyScaleLinear, 0.5, 0.0, 1.0, 0.5, 1e-10);
    expectRoot<BisectionSolver<double>>("bisection / tiny |f|", tinyScaleLinear, 0.5, 0.0, 1.0, 0.5,
                                        1e-10);
    expectRoot<SecantSolver<double>>("secant / tiny |f|", tinyScaleLinear, 0.5, 0.0, 1.0, 0.5,
                                     1e-10);
    expectRoot<FalsePositionSolver<double>>("falsepos / tiny |f|", tinyScaleLinear, 0.5, 0.0, 1.0,
                                            0.5, 1e-10);
    expectRoot<RidderSolver<double>>("ridder / tiny |f|", tinyScaleLinear, 0.5, 0.0, 1.0, 0.5,
                                     1e-10);
}

TEST(Solvers, autoBracketTinyResidualIsNotARoot) {
    // The auto-bracketing entry point had the same absolute |f| floor at the
    // initial guess (fScale was still 1.0), so it returned the guess 0 with
    // error 0.5 for an objective whose values are all tiny.
    BrentSolver<double> solver;
    const double got = solver.solve(tinyScaleLinear, 1e-12, 0.0, 0.1);
    CHECK_CLOSE("brent auto / tiny |f|", got, 0.5, 1e-9);
}

TEST(Solvers, falsePositionDefaultBudgetFindsBenignRoot) {
    // Plain regula falsi kept the right end stuck on this convex function and
    // exhausted the DEFAULT 100-evaluation budget (bisection needs 44). The
    // Illinois modification must converge well inside the default budget.
    FalsePositionSolver<double> solver;
    std::size_t evaluations = 0;
    const CountingFn counted{falsePositionStagnation, &evaluations};

    double got = std::numeric_limits<double>::quiet_NaN();
    try {
        got = solver.solve(counted, 1e-12, 1.5, 0.0, 3.0);
    } catch (const std::exception& e) {
        ADD_FAILURE() << "falsepos / default budget threw: " << e.what();
    }

    CHECK_CLOSE("falsepos / exp stagnation root", got, 0.7, 1e-9);
    EXPECT_LT(evaluations, solver.maxEvaluations())
        << "Illinois must converge before the default 100-evaluation budget";
    EXPECT_EQ(solver.maxEvaluations(), 100u) << "test must exercise the default budget";
}

TEST(Solvers, tridiagonalRejectsZeroFirstPivot) {
    // [[0, 1], [1, 0]] x = [1, 2] is well posed (x = [2, 1]) but has b[0] = 0:
    // the unpivoted Thomas elimination divides by it unchecked. Before the fix
    // this returned [nan, nan] without throwing.
    std::vector<double> x;
    try {
        x = TridiagonalSolver<double>::solve({0.0, 1.0}, {0.0, 1.0}, {1.0, 0.0}, {1.0, 2.0});
    } catch (const std::exception& e) {
        EXPECT_NE(std::string(e.what()).find("pivot"), std::string::npos)
            << "unclear exception message: " << e.what();
        return;
    }
    ADD_FAILURE() << "b[0] = 0 did not throw; got x = [" << x[0] << ", " << x[1] << "]";
}

TEST(Solvers, tridiagonalRejectsNearZeroPivots) {
    // b[0] = 1e-300 with a unit super-diagonal: the exact solution is ~[1, 1]
    // but the unpivoted division produced x[0] ~ 1.5e284 without throwing.
    SCOPED_TRACE("near-zero b[0]");
    const double tiny = 1e-300;
    EXPECT_THROW(
        TridiagonalSolver<double>::solve({0.0, 1.0}, {tiny, 1.0}, {1.0, 0.0}, {tiny + 1.0, 2.0}),
        std::exception);

    // Interior pivot = 1 - 1*1 = 2.2e-16 (round-off-level cancellation in a
    // numerically singular 2x2); the old `denom == 0.0` test cannot see it.
    SCOPED_TRACE("cancelling interior pivot");
    const double eps = std::numeric_limits<double>::epsilon();
    EXPECT_THROW(
        TridiagonalSolver<double>::solve({0.0, 1.0}, {1.0, 1.0 + eps}, {1.0, 0.0}, {1.0, 2.0}),
        std::exception);
}

TEST(Solvers, tridiagonalRejectsNonFinitePivot) {
    // b[0] = NaN used to pass the `denom == 0.0` test (NaN != 0) and poison
    // the whole solution silently.
    const double nan = std::numeric_limits<double>::quiet_NaN();
    EXPECT_THROW(TridiagonalSolver<double>::solve({0.0, 1.0}, {nan, 1.0}, {1.0, 0.0}, {1.0, 2.0}),
                 std::exception);
}

TEST(Solvers, tridiagonalSolvesWellPosedNonDominantSystem) {
    // The pivot guard is relative: a non-diagonally-dominant but well-
    // conditioned system [[1, 10], [10, 1]] must still solve (this one did
    // before the fix as well; the guard must not turn it into a false error).
    const std::vector<double> x =
        TridiagonalSolver<double>::solve({0.0, 10.0}, {1.0, 1.0}, {10.0, 0.0}, {11.0, 11.0});
    CHECK_CLOSE("thomas / non-dominant x0", x[0], 1.0, 1e-14);
    CHECK_CLOSE("thomas / non-dominant x1", x[1], 1.0, 1e-14);
}
