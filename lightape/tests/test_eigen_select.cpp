#include <Eigen/Core>

#include <cmath>
#include <cstddef>
#include <type_traits>

#include "lightape/eigen/ForwardScalarEigen.h"
#include "lightape/eigen/LeanReverseEigen.h"
#include "lightape/eigen/Select.h"
#include "lightape/lightape.h"
#include <gtest/gtest.h>

namespace {

using lightape::RevScalar;
using lightape::RevTape;
using lightape::Tangent;

using RevArray = Eigen::Array<RevScalar, Eigen::Dynamic, 1>;
using DoubleArray = Eigen::Array<double, Eigen::Dynamic, 1>;
using DoubleMatrix = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>;
using BoolArray = Eigen::Array<bool, Eigen::Dynamic, 1>;
using Tan2 = Tangent<double, 2>;
using TanArray = Eigen::Array<Tan2, Eigen::Dynamic, 1>;

class EigenSelectTest : public ::testing::Test {
protected:
    void SetUp() override { RevTape::active().clear(); }
    void TearDown() override { RevTape::active().clear(); }
};

// ── values: bool / int / double conditions against Eigen's native select ──

TEST_F(EigenSelectTest, DoubleArraysMatchNativeSelect) {
    DoubleArray a(4);
    DoubleArray b(4);
    a << 1.5, -2.0, 3.25, 0.0;
    b << -1.0, 2.5, 4.0, -3.5;

    BoolArray c(4);
    c << true, false, true, false;
    const auto got = lightape::select(c, a, b);
    const auto native = (c.select(a, b)).eval();
    for (int i = 0; i < 4; ++i) {
        EXPECT_DOUBLE_EQ(got(i), native(i)) << "coefficient " << i;
    }

    Eigen::Array<int, 4, 1> ci;
    ci << 0, 1, -2, 0;
    const auto gotInt = lightape::select(ci, a, b);
    for (int i = 0; i < 4; ++i) {
        EXPECT_DOUBLE_EQ(gotInt(i), ci(i) != 0 ? a(i) : b(i)) << "coefficient " << i;
    }

    DoubleArray cd(4);
    cd << 0.0, 0.25, -1.0, 0.0;
    const auto gotDouble = lightape::select(cd, a, b);
    for (int i = 0; i < 4; ++i) {
        EXPECT_DOUBLE_EQ(gotDouble(i), cd(i) != 0.0 ? a(i) : b(i)) << "coefficient " << i;
    }
}

TEST_F(EigenSelectTest, TwoDimensionalAndMatrixOverloads) {
    DoubleMatrix m(2, 3);
    DoubleMatrix n(2, 3);
    m << 1.0, 2.0, 3.0, 4.0, 5.0, 6.0;
    n << -1.0, -2.0, -3.0, -4.0, -5.0, -6.0;
    Eigen::Matrix<bool, 2, 3> c;
    c << true, false, true, false, true, false;

    const auto arrayResult = lightape::select(c.array(), m.array(), n.array());
    static_assert(std::is_same_v<std::decay_t<decltype(arrayResult)>, Eigen::Array<double, 2, 3>>);
    for (int r = 0; r < 2; ++r) {
        for (int col = 0; col < 3; ++col) {
            EXPECT_DOUBLE_EQ(arrayResult(r, col), c(r, col) ? m(r, col) : n(r, col));
        }
    }

    const auto matrixResult = lightape::select(c, m, n);
    static_assert(
        std::is_same_v<std::decay_t<decltype(matrixResult)>, Eigen::Matrix<double, 2, 3>>);
    EXPECT_TRUE(matrixResult.isApprox((c.select(m, n)).eval()));
}

// ── RevScalar: laziness, node count, gradients ──

TEST_F(EigenSelectTest, RevScalarSelectIsLazyAndGradientsAreExact) {
    constexpr int n = 5;
    const double xv[n] = {1.5, -0.5, 2.0, -1.0, 0.25};
    const double yv[n] = {0.5, 1.5, 2.5, 3.5, 4.5};

    RevTape& tape = RevTape::active();
    RevArray x(n);
    RevArray y(n);
    for (int i = 0; i < n; ++i) {
        x(i) = RevScalar(xv[i], tape.input(xv[i]));
        y(i) = RevScalar(yv[i], tape.input(yv[i]));
    }
    BoolArray c(n);
    c << true, false, true, false, true;

    const auto thenExpr = x.array().exp();
    const auto elseExpr = y.array().log();
    const std::size_t before = tape.size();
    const RevArray out = lightape::select(c, thenExpr, elseExpr);
    const std::size_t recorded = tape.size() - before;

    // One node per coefficient: only the winning branch was recorded.
    EXPECT_EQ(recorded, static_cast<std::size_t>(n));
    for (int i = 0; i < n; ++i) {
        EXPECT_DOUBLE_EQ(out(i).value, c(i) ? std::exp(xv[i]) : std::log(yv[i]))
            << "coefficient " << i;
        EXPECT_NE(out(i).node, RevTape::kNone) << "coefficient " << i;
    }

    const RevScalar f = out.sum();
    tape.reverse(f.node);
    for (int i = 0; i < n; ++i) {
        EXPECT_DOUBLE_EQ(tape.adjoint(x(i).node), c(i) ? std::exp(xv[i]) : 0.0) << "x " << i;
        EXPECT_DOUBLE_EQ(tape.adjoint(y(i).node), c(i) ? 0.0 : 1.0 / yv[i]) << "y " << i;
    }
}

// Eigen 5 spells `select` as a `CwiseTernaryOp` over (then, else, cond) whose
// ternary evaluator computes all three coefficients before applying the
// functor: both AD branches are recorded (orphan nodes, decision #16). This
// functor matches Eigen 5's `scalar_boolean_select_op` argument order.
struct EagerSelectOp {
    RevScalar operator()(const RevScalar& thenValue, const RevScalar& elseValue,
                         bool condition) const {
        return condition ? thenValue : elseValue;
    }
};

/// Builds exactly the expression shape Eigen 5's `.select()` returns and
/// evaluates it. On Eigen 3.4 the dedicated lazy `Select` would hide the
/// hazard, so the demonstration uses this CwiseTernaryOp (the same mechanism
/// Stan main overrides with `lazy_select_evaluator`).
template <typename ThenExpr, typename ElseExpr, typename CondExpr>
RevArray eigen5StyleEagerSelect(const ThenExpr& thenExpr, const ElseExpr& elseExpr,
                                const CondExpr& cond) {
    const Eigen::CwiseTernaryOp<EagerSelectOp, ThenExpr, ElseExpr, CondExpr> expr(thenExpr,
                                                                                  elseExpr, cond);
    return expr.eval();
}

TEST_F(EigenSelectTest, NativeStyleSelectRecordsBothBranches) {
    constexpr int n = 4;
    const double xv[n] = {0.25, -1.0, 2.0, -0.5};
    const double yv[n] = {1.5, 2.5, 0.5, 3.0};
    BoolArray c(n);
    c << true, false, true, false;

    RevTape& tape = RevTape::active();
    RevArray x(n);
    RevArray y(n);
    for (int i = 0; i < n; ++i) {
        x(i) = RevScalar(xv[i], tape.input(xv[i]));
        y(i) = RevScalar(yv[i], tape.input(yv[i]));
    }

    std::size_t before = tape.size();
    const RevArray safe = lightape::select(c, x.array().exp(), y.array().log());
    const std::size_t lazyNodes = tape.size() - before;
    EXPECT_EQ(lazyNodes, static_cast<std::size_t>(n));
    for (int i = 0; i < n; ++i) {
        EXPECT_DOUBLE_EQ(safe(i).value, c(i) ? std::exp(xv[i]) : std::log(yv[i]));
    }

    // Same inputs recorded again through the eager select path.
    tape.clear();
    for (int i = 0; i < n; ++i) {
        x(i) = RevScalar(xv[i], tape.input(xv[i]));
        y(i) = RevScalar(yv[i], tape.input(yv[i]));
    }
    const auto thenExpr = x.array().exp();
    const auto elseExpr = y.array().log();

    before = tape.size();
#if EIGEN_VERSION_AT_LEAST(5, 0, 0)
    // Genuine Eigen 5 native select: the exact hazard decision #16 warns about.
    const RevArray eager = c.select(thenExpr, elseExpr);
#else
    const RevArray eager = eigen5StyleEagerSelect(thenExpr, elseExpr, c);
#endif
    const std::size_t eagerNodes = tape.size() - before;
    EXPECT_EQ(eagerNodes, static_cast<std::size_t>(2 * n));
    EXPECT_GT(eagerNodes, lazyNodes);
    for (int i = 0; i < n; ++i) {
        EXPECT_DOUBLE_EQ(eager(i).value, c(i) ? std::exp(xv[i]) : std::log(yv[i]));
    }
}

// ── Tangent: directional derivatives on the winning branch only ──

TEST_F(EigenSelectTest, TangentDirectionalDerivatives) {
    constexpr int n = 4;
    const double xv[n] = {0.5, -1.0, 2.0, 0.25};
    const double yv[n] = {1.0, 2.0, 0.5, 4.0};
    BoolArray c(n);
    c << true, true, false, false;

    TanArray x(n);
    TanArray y(n);
    for (int i = 0; i < n; ++i) {
        x(i) = Tan2(xv[i], {1.0, 0.0});
        y(i) = Tan2(yv[i], {0.0, 1.0});
    }

    const TanArray out = lightape::select(c, x.array().exp(), y.array().log());
    for (int i = 0; i < n; ++i) {
        const double thenValue = std::exp(xv[i]);
        const double elseValue = std::log(yv[i]);
        EXPECT_DOUBLE_EQ(out(i).value, c(i) ? thenValue : elseValue) << "coefficient " << i;
        EXPECT_DOUBLE_EQ(out(i).d[0], c(i) ? thenValue : 0.0) << "coefficient " << i;
        EXPECT_DOUBLE_EQ(out(i).d[1], c(i) ? 0.0 : 1.0 / yv[i]) << "coefficient " << i;
    }
}

// ── mixed double/AD branches and AD conditions ──

TEST_F(EigenSelectTest, MixedDoubleAndRevScalarBranches) {
    constexpr int n = 4;
    const double xv[n] = {1.0, -2.0, 0.5, 3.0};
    const double dv[n] = {10.0, 20.0, 30.0, 40.0};
    BoolArray c(n);
    c << true, false, true, false;

    RevTape& tape = RevTape::active();
    RevArray x(n);
    for (int i = 0; i < n; ++i) {
        x(i) = RevScalar(xv[i], tape.input(xv[i]));
    }
    DoubleArray d(n);
    for (int i = 0; i < n; ++i) {
        d(i) = dv[i];
    }

    // AD then-branch, plain-double else-branch: common scalar is RevScalar.
    const std::size_t before = tape.size();
    const auto out = lightape::select(c, x.array() * 2.0, d);
    static_assert(std::is_same_v<std::decay_t<decltype(out)>::Scalar, RevScalar>);
    EXPECT_EQ(tape.size() - before, 2u); // one multiply per selected coefficient
    for (int i = 0; i < n; ++i) {
        EXPECT_DOUBLE_EQ(out(i).value, c(i) ? 2.0 * xv[i] : dv[i]) << "coefficient " << i;
        if (!c(i)) {
            EXPECT_EQ(out(i).node, RevTape::kNone); // plain constant: no node
        }
    }

    const RevScalar f = out.sum();
    tape.reverse(f.node);
    for (int i = 0; i < n; ++i) {
        EXPECT_DOUBLE_EQ(tape.adjoint(x(i).node), c(i) ? 2.0 : 0.0) << "coefficient " << i;
    }

    // Reversed mixing: plain-double then-branch, AD else-branch.
    tape.clear();
    for (int i = 0; i < n; ++i) {
        x(i) = RevScalar(xv[i], tape.input(xv[i]));
    }
    const std::size_t before2 = tape.size();
    const auto out2 = lightape::select(c, d, x.array() + 1.0);
    static_assert(std::is_same_v<std::decay_t<decltype(out2)>::Scalar, RevScalar>);
    EXPECT_EQ(tape.size() - before2, 2u); // one add per non-selected coefficient
    const RevScalar g = out2.sum();
    tape.reverse(g.node);
    for (int i = 0; i < n; ++i) {
        EXPECT_DOUBLE_EQ(out2(i).value, c(i) ? dv[i] : xv[i] + 1.0) << "coefficient " << i;
        EXPECT_DOUBLE_EQ(tape.adjoint(x(i).node), c(i) ? 0.0 : 1.0) << "coefficient " << i;
    }
}

TEST_F(EigenSelectTest, AdConditionIsReadFromPrimalOnly) {
    constexpr int n = 4;
    const double cv[n] = {1.0, -2.0, 0.0, 0.25}; // nonzero -> then
    const double xv[n] = {1.0, 2.0, 3.0, 4.0};
    const double yv[n] = {-1.0, -2.0, -3.0, -4.0};

    RevTape& tape = RevTape::active();
    RevArray cond(n);
    RevArray x(n);
    RevArray y(n);
    for (int i = 0; i < n; ++i) {
        cond(i) = RevScalar(cv[i], tape.input(cv[i]));
        x(i) = RevScalar(xv[i], tape.input(xv[i]));
        y(i) = RevScalar(yv[i], tape.input(yv[i]));
    }

    const auto out = lightape::select(cond, x.array() + 1.0, y.array() - 1.0);
    for (int i = 0; i < n; ++i) {
        EXPECT_DOUBLE_EQ(out(i).value, cv[i] != 0.0 ? xv[i] + 1.0 : yv[i] - 1.0)
            << "coefficient " << i;
    }

    const RevScalar f = out.sum();
    tape.reverse(f.node);
    for (int i = 0; i < n; ++i) {
        EXPECT_DOUBLE_EQ(tape.adjoint(cond(i).node), 0.0) << "condition " << i;
        EXPECT_DOUBLE_EQ(tape.adjoint(x(i).node), cv[i] != 0.0 ? 1.0 : 0.0) << "x " << i;
        EXPECT_DOUBLE_EQ(tape.adjoint(y(i).node), cv[i] != 0.0 ? 0.0 : 1.0) << "y " << i;
    }

    // A Tangent condition follows the same primal-value rule.
    TanArray tcond(n);
    for (int i = 0; i < n; ++i) {
        tcond(i) = Tan2(cv[i], {1.0, 0.0});
    }
    DoubleArray a(n);
    DoubleArray b(n);
    a << 1.0, 2.0, 3.0, 4.0;
    b << -1.0, -2.0, -3.0, -4.0;
    const auto tout = lightape::select(tcond, a, b);
    for (int i = 0; i < n; ++i) {
        EXPECT_DOUBLE_EQ(tout(i), cv[i] != 0.0 ? a(i) : b(i)) << "coefficient " << i;
    }
}

} // namespace
