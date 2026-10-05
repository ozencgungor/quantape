#include "quantape/log/Log.h"
#include "quantape/math/NumericalMethods.h"
#include "quantape/util/Check.h"

#include <cmath>
#include <cstddef>
#include <cstdio>
#include <stdexcept>

using namespace quantape::math;

namespace {

int failures = 0;

void report(const char* label, bool ok, double got, double expected, double err) {
    if (ok) {
        QTA_LOG_INFO("test", "  {} ok  got={} expected={} err={}", label,
                     quantape::util::num(got, 15), quantape::util::num(expected, 15),
                     quantape::util::num(err, 3));
    } else {
        QTA_LOG_ERROR("test", "  {} FAIL  got={} expected={} err={}", label,
                      quantape::util::num(got, 15), quantape::util::num(expected, 15),
                      quantape::util::num(err, 3));
    }
    if (!ok) {
        ++failures;
    }
}

struct Case {
    const char* name;
    double (*f)(double);
    double lo, hi, guess, root;
};

template <typename SolverT>
void runSolver(const char* solverName, const Case& c, double tol = 1e-9) {
    SolverT solver;
    solver.setMaxEvaluations(200);
    try {
        const double got = solver.solve(c.f, 1e-12, c.guess, c.lo, c.hi);
        char label[128];
        std::snprintf(label, sizeof(label), "%s / %s", solverName, c.name);
        const double err = std::fabs(got - c.root);
        report(label, err <= tol, got, c.root, err);
    } catch (const std::exception& e) {
        QTA_LOG_ERROR("test", "  {} FAIL  threw: {}", solverName, e.what());
        ++failures;
    }
}

struct CountingFn {
    double (*f)(double);
    std::size_t* count;
    double operator()(double x) const {
        ++*count;
        return f(x);
    }
};

void testGoldenRoots() {
    QTA_LOG_INFO("test", "=== Golden roots (accuracy 1e-12) ===");

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

void testBrentActuallyInterpolates() {
    QTA_LOG_INFO("test", "=== Brent interpolation regression ===");

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

    QTA_LOG_INFO("test", "  cos(x)-x evaluation counts bisection={} brent={}", bisectionEvals,
                 brentEvals);
    const bool ok = brentEvals * 2 <= bisectionEvals && brentEvals <= 20;
    if (ok) {
        QTA_LOG_INFO("test", "  interpolation in use ok (brent must use < half the evals)");
    } else {
        QTA_LOG_ERROR("test", "  interpolation in use FAIL (brent must use < half the evals)");
        ++failures;
    }
}

void testNewtonWithDerivative() {
    QTA_LOG_INFO("test", "=== Newton with explicit derivative ===");

    auto f = [](double x) { return x * x - 2.0; };
    auto df = [](double x) { return 2.0 * x; };

    NewtonSolverWithDerivative<double, decltype(df)> solver(df);
    solver.setMaxEvaluations(100);
    const double got = solver.solve(f, 1e-12, 1.5, 1.0, 2.0);
    const double expected = std::sqrt(2.0);
    report("newton-explicit / x^2-2", std::fabs(got - expected) <= 1e-12, got, expected,
           std::fabs(got - expected));

    // An objective carrying its own derivative() must be used automatically
    struct WithMember {
        double operator()(double x) const { return x * x - 2.0; }
        double derivative(double x) const { return 2.0 * x; }
    };
    NewtonSolver<double> memberSolver;
    memberSolver.setMaxEvaluations(100);
    const double gotMember = memberSolver.solve(WithMember{}, 1e-12, 1.5, 1.0, 2.0);
    report("newton-member / x^2-2", std::fabs(gotMember - expected) <= 1e-12, gotMember, expected,
           std::fabs(gotMember - expected));
}

void testAutoBracketing() {
    QTA_LOG_INFO("test", "=== Auto-bracketing ===");

    auto f = [](double x) { return x * x * x - x - 2.0; };
    BrentSolver<double> solver;
    solver.setMaxEvaluations(200);
    const double got = solver.solve(f, 1e-12, 1.5, 0.1);
    const double expected = 1.5213797068045676;
    report("brent / x^3-x-2 from guess+step", std::fabs(got - expected) <= 1e-9, got, expected,
           std::fabs(got - expected));
}

void testTridiagonal() {
    QTA_LOG_INFO("test", "=== Tridiagonal solver ===");

    const std::vector<double> a{0.0, 1.0, 1.0};
    const std::vector<double> b{2.0, 2.0, 2.0};
    const std::vector<double> c{1.0, 1.0, 0.0};
    const std::vector<double> d{1.0, 2.0, 3.0};
    const std::vector<double> expected{0.5, 0.0, 1.5};

    const std::vector<double> x = TridiagonalSolver<double>::solve(a, b, c, d);
    for (std::size_t i = 0; i < expected.size(); ++i) {
        char label[64];
        std::snprintf(label, sizeof(label), "thomas / x%zu", i);
        report(label, std::fabs(x[i] - expected[i]) <= 1e-14, x[i], expected[i],
               std::fabs(x[i] - expected[i]));
    }
}

template <typename Fn>
void expectThrow(const char* label, Fn&& fn) {
    try {
        fn();
        QTA_LOG_ERROR("test", "  {} FAIL  no exception", label);
        ++failures;
    } catch (const std::exception&) {
        QTA_LOG_INFO("test", "  {} ok    threw as expected", label);
    }
}

void testErrorHandling() {
    QTA_LOG_INFO("test", "=== Error handling ===");

    auto flat = [](double x) { return x * x + 1.0; };
    auto linear = [](double x) { return x - 0.5; };

    expectThrow("accuracy <= 0", [&] {
        BrentSolver<double> s;
        s.solve(linear, 0.0, 0.25, 0.0, 1.0);
    });
    expectThrow("not bracketed", [&] {
        BrentSolver<double> s;
        s.solve(flat, 1e-10, 0.5, 0.0, 1.0);
    });
    expectThrow("guess outside bracket", [&] {
        BrentSolver<double> s;
        s.solve(linear, 1e-10, 2.0, 0.0, 1.0);
    });
    expectThrow("xMin >= xMax", [&] {
        BrentSolver<double> s;
        s.solve(linear, 1e-10, 0.5, 1.0, 0.0);
    });
    expectThrow("tridiagonal size mismatch", [&] {
        const std::vector<double> a{0.0, 1.0};
        const std::vector<double> b{2.0, 2.0, 2.0};
        const std::vector<double> c{1.0, 0.0};
        const std::vector<double> d{1.0, 2.0};
        TridiagonalSolver<double>::solve(a, b, c, d);
    });
    expectThrow("tridiagonal zero pivot", [&] {
        const std::vector<double> a{0.0, 1.0};
        const std::vector<double> b{1.0, 1.0};
        const std::vector<double> c{1.0, 0.0};
        const std::vector<double> d{1.0, 2.0};
        TridiagonalSolver<double>::solve(a, b, c, d);
    });
}

} // namespace

int main() {
    testGoldenRoots();
    testBrentActuallyInterpolates();
    testNewtonWithDerivative();
    testAutoBracketing();
    testTridiagonal();
    testErrorHandling();

    if (failures == 0) {
        QTA_LOG_INFO("test", "All solver tests passed.");
        return 0;
    }
    QTA_LOG_ERROR("test", "{} solver test(s) FAILED.", failures);
    return 1;
}
