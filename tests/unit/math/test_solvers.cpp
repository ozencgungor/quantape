// test_solvers.cpp — all six 1-D root finders and the tridiagonal solver.

#include "quantape/math/NumericalMethods.h"

#include <cmath>
#include <cstddef>
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

struct CountingFn {
    double (*f)(double);
    std::size_t* count;
    double operator()(double x) const {
        ++*count;
        return f(x);
    }
};

} // namespace

TEST(Solvers, goldenRootsMatchAnalytic) {
    const Case cases[] = {
        {"x^2-2", [](double x) { return x * x - 2.0; }, 0.0, 3.0, 1.5, 1.4142135623730951},
        {"cos(x)-x", [](double x) { return std::cos(x) - x; }, 0.0, 1.0, 0.5, 0.7390851332151607},
        {"e^x-3x", [](double x) { return std::exp(x) - 3.0 * x; }, 0.1, 0.9, 0.3,
         0.6190612867359451},
        {"x^3-2x-5", [](double x) { return std::pow(x, 3) - 2.0 * x - 5.0; }, 2.0, 3.0, 2.5,
         2.0945514815423265},
        {"log(x)-1", [](double x) { return std::log(x) - 1.0; }, 0.5, 3.0, 2.0, 2.7182818284590452},
    };

    for (const Case& c : cases) {
        runSolver<BisectionSolver<double>>("bisection", c);
        runSolver<BrentSolver<double>>("brent", c);
        runSolver<SecantSolver<double>>("secant", c);
        runSolver<FalsePositionSolver<double>>("falsepos", c);
        runSolver<RidderSolver<double>>("ridder", c);
        runSolver<NewtonSolver<double>>("newton-fd", c);
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
