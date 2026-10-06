/**
 * @file test_ift.cpp
 * @brief Implicit-function-theorem sensitivity layer
 *
 * First-order sensitivities of optima w.r.t. market data (P4.8):
 *   - unconstrained:  dp/dm = -H^{-1} G       (exact AD Hessians)
 *   - constrained:    KKT system of §6.2      (multipliers from SLSQP/AUGLAG)
 *
 * Every fixture is verified against analytic references and/or
 * bump-and-recalibrate finite differences (the gates from
 * docs/ad_optimizers.md §8.3: IFT vs FD <= 1e-6), plus multiplier
 * sensitivities vs one-sided re-optimization.
 */

#include "quantape/math/StanMath.h"

#include "quantape/calibration/ImplicitFunction.h"
#include "quantape/math/Optimization/AugLag.h"
#include "quantape/math/Optimization/LBFGS.h"
#include "quantape/math/Optimization/OptimizerStanPrimitives.h"
#include "quantape/math/Optimization/SLSQP.h"
#include "quantape/math/Optimization/TNewton.h"

#include <Eigen/Dense>

#include <cmath>
#include <vector>

#include "support/GtestSupport.h"
#include "support/StanTapeFixture.h"

using stan::math::var;

namespace {

bool converged(quantape::math::OptimizeResult r) {
    return r == quantape::math::OptimizeResult::Success ||
           r == quantape::math::OptimizeResult::GradientTolReached ||
           r == quantape::math::OptimizeResult::FtolReached ||
           r == quantape::math::OptimizeResult::XtolReached;
}

// ============================================================================
// Fixture 1: linear least squares (unconstrained)
//   f2(x, m) = 1/2 * sum_k (x . a_k - m_k)^2
//   p(m) = (A^T A)^{-1} A^T m,   dp/dm = (A^T A)^{-1} A^T  (analytic)
// ============================================================================
struct LsqObjective {
    std::vector<std::vector<double>> a; // M basis vectors, each of length n

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

// ============================================================================
// Fixture 2: nonlinear shifted Rosenbrock (unconstrained)
//   f2 = 100 (x1 - x0^2)^2 + (x0 - m)^2   =>  p(m) = (m, m^2), dp/dm = (1, 2m)
// ============================================================================
struct ShiftedRosenbrock {
    template <typename Sx, typename Sm>
    auto operator()(const std::vector<Sx>& x, const std::vector<Sm>& m) const {
        const Sx t1 = x[1] - x[0] * x[0];
        const Sx t2 = x[0] - Sx(m[0]);
        return Sx(100.0) * t1 * t1 + t2 * t2;
    }
};

// ============================================================================
// Fixture 3: active bound
//   f2 = 1/2 (x - m)^2, g2: x >= 0  (i.e. -x <= 0), m = -0.5
//   p = 0, lambda = -m, dp/dm = 0, dlambda/dm = -1
// ============================================================================
struct BoundQuadratic {
    template <typename Sx, typename Sm>
    auto operator()(const std::vector<Sx>& x, const std::vector<Sm>& m) const {
        const Sx d = x[0] - Sx(m[0]);
        return Sx(0.5) * d * d;
    }
};
struct NonNegative {
    template <typename Sx, typename Sm>
    void operator()(const std::vector<Sx>& x, const std::vector<Sm>& /*m*/,
                    std::vector<Sx>& out) const {
        out.clear();
        out.push_back(-x[0]); // -x <= 0  <=>  x >= 0
    }
};

// ============================================================================
// Fixture 4: inactive inequality -> unconstrained fallback
//   same quadratic, g: x <= 10, m = 2:  p = 2, dp/dm = 1, lambda = 0
// ============================================================================
struct UpperTen {
    template <typename Sx, typename Sm>
    void operator()(const std::vector<Sx>& x, const std::vector<Sm>& /*m*/,
                    std::vector<Sx>& out) const {
        out.clear();
        out.push_back(x[0] - Sx(10.0)); // x - 10 <= 0
    }
};

// ============================================================================
// Fixture 5: equality projection
//   f2 = 1/2 ||x - m||^2, h: x0 + x1 = 0
//   p = ((m0-m1)/2, (m1-m0)/2); dp/dm = [[1/2,-1/2],[-1/2,1/2]]; nu = (m0+m1)/2
// ============================================================================
struct TwoQuad {
    template <typename Sx, typename Sm>
    auto operator()(const std::vector<Sx>& x, const std::vector<Sm>& m) const {
        const Sx d0 = x[0] - Sx(m[0]);
        const Sx d1 = x[1] - Sx(m[1]);
        return Sx(0.5) * (d0 * d0 + d1 * d1);
    }
};
struct SumZero {
    template <typename Sx, typename Sm>
    void operator()(const std::vector<Sx>& x, const std::vector<Sm>& /*m*/,
                    std::vector<Sx>& out) const {
        out.clear();
        out.push_back(x[0] + x[1]);
    }
};

// ============================================================================
// Fixture 6: combined active inequality + equality
//   f2 = 1/2 (x0-m0)^2 + 1/2 (x1-m1)^2, h: x0 + x1 = 1, g: x0 <= 0.8
//   m = (2, -1):  p = (0.8, 0.2), lambda = 2.4, nu = -1.2
//   dp/dm = 0;  dlambda/dm = (1, -1);  dnu/dm = (0, 1)
// ============================================================================
struct SepQuad {
    template <typename Sx, typename Sm>
    auto operator()(const std::vector<Sx>& x, const std::vector<Sm>& m) const {
        const Sx d0 = x[0] - Sx(m[0]);
        const Sx d1 = x[1] - Sx(m[1]);
        return Sx(0.5) * (d0 * d0 + d1 * d1);
    }
};
struct SumOne {
    template <typename Sx, typename Sm>
    void operator()(const std::vector<Sx>& x, const std::vector<Sm>& /*m*/,
                    std::vector<Sx>& out) const {
        out.clear();
        out.push_back(x[0] + x[1] - Sx(1.0));
    }
};
struct X0Cap {
    template <typename Sx, typename Sm>
    void operator()(const std::vector<Sx>& x, const std::vector<Sm>& /*m*/,
                    std::vector<Sx>& out) const {
        out.clear();
        out.push_back(x[0] - Sx(0.8)); // x0 <= 0.8
    }
};

// ============================================================================
// Fixture 7: degenerate (linearly dependent active inequalities)
//   g1 = x - 1 <= 0 and g2 = 2x - 2 <= 0 both active at x = 1, m = 2
//   -> singular KKT system -> pseudo-inverse path; dp/dm = 0 still correct
// ============================================================================
struct DoubleIneq {
    template <typename Sx, typename Sm>
    void operator()(const std::vector<Sx>& x, const std::vector<Sm>& /*m*/,
                    std::vector<Sx>& out) const {
        out.clear();
        out.push_back(x[0] - Sx(1.0));
        out.push_back(Sx(2.0) * x[0] - Sx(2.0));
    }
};

// ============================================================================
// Fixture 12: NONLINEAR active inequality (exercises lambda*H_g in H_L)
//   f2 = 1/2 ((x0-m0)^2 + (x1-m1)^2), g: x0^2 + x1^2 - 1 <= 0 (unit disk)
//   m = (1.5, 0):  p = (1, 0), lambda = 0.25
//   H_L = 1.5 I, J = (2, 0)
//   dp/dm = [[0, 0], [0, 2/3]],  dlambda/dm = [0.5, 0]
// ============================================================================
struct UnitDisk {
    template <typename Sx, typename Sm>
    void operator()(const std::vector<Sx>& x, const std::vector<Sm>& /*m*/,
                    std::vector<Sx>& out) const {
        out.clear();
        out.push_back(x[0] * x[0] + x[1] * x[1] - Sx(1.0));
    }
};

} // namespace

class IftUnconstrainedTest : public StanTapeTest {};
class IftKktTest : public StanTapeTest {};
class IftCompositionTest : public StanTapeTest {};

TEST_F(IftUnconstrainedTest, lsqAnalyticAndBumpRecalibrate) {
    // n = 3 parameters, M = 5 quotes
    const std::vector<std::vector<double>> a{
        {1.0, 0.2, -0.1}, {0.5, 1.0, 0.3}, {0.1, 0.4, 1.0}, {0.8, -0.3, 0.6}, {-0.2, 0.7, 0.9}};
    const std::vector<double> m{0.8, 1.2, -0.5, 0.3, 1.7};
    const std::size_t n = 3, M = 5;

    // Analytic reference: A^T A and A^T
    Eigen::MatrixXd A(M, n);
    for (std::size_t k = 0; k < M; ++k)
        for (std::size_t i = 0; i < n; ++i)
            A(static_cast<Eigen::Index>(k), static_cast<Eigen::Index>(i)) = a[k][i];
    Eigen::MatrixXd AtA = A.transpose() * A;
    Eigen::MatrixXd AtA_inv_At = AtA.ldlt().solve(A.transpose());
    Eigen::VectorXd mv(M);
    for (std::size_t k = 0; k < M; ++k)
        mv(static_cast<Eigen::Index>(k)) = m[k];
    Eigen::VectorXd p_ref = AtA.ldlt().solve(A.transpose() * mv);

    std::vector<double> x = {0.0, 0.0, 0.0};

    // IFT at the exact optimum
    LsqObjective obj{a};
    std::vector<double> p_hat(3);
    for (std::size_t i = 0; i < n; ++i)
        p_hat[i] = p_ref(static_cast<Eigen::Index>(i));
    quantape::math::IftResult ift;
    std::vector<double> dp_dm;
    quantape::math::iftUnconstrained(obj, p_hat, m, dp_dm, ift);
    EXPECT_TRUE(!ift.regularized);
    EXPECT_TRUE(dp_dm.size() == n * M);
    for (std::size_t i = 0; i < n; ++i)
        for (std::size_t j = 0; j < M; ++j)
            CHECK_CLOSE("lsq dp/dm vs analytic", dp_dm[i * M + j],
                        AtA_inv_At(static_cast<Eigen::Index>(i), static_cast<Eigen::Index>(j)),
                        1e-10);
    // Condition number should match lambda_max/lambda_min of A^T A
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es(AtA);
    CHECK_CLOSE("lsq cond", ift.conditionNumber,
                es.eigenvalues().maxCoeff() / es.eigenvalues().minCoeff(), 1e-9);

    // Bump-and-recalibrate central FD cross-check (grad_tol tight for clean FD)
    quantape::math::StopCriteria criteria;
    criteria.grad_tol = 1e-12;
    criteria.maxeval = 100000;
    quantape::math::TNewton<var> solver(criteria);
    const double h = 1e-4;
    for (std::size_t j = 0; j < M; ++j) {
        std::vector<double> m_plus = m, m_minus = m;
        m_plus[j] += h;
        m_minus[j] -= h;
        const auto fx = [&](const auto& theta) { return obj(theta, m_plus); };
        std::vector<double> xp = x;
        EXPECT_TRUE(converged(solver.minimize(fx, xp)));
        const auto fm = [&](const auto& theta) { return obj(theta, m_minus); };
        std::vector<double> xm = x;
        EXPECT_TRUE(converged(solver.minimize(fm, xm)));
        for (std::size_t i = 0; i < n; ++i) {
            const double fd = (xp[i] - xm[i]) / (2.0 * h);
            CHECK_CLOSE("lsq dp/dm vs FD", dp_dm[i * M + j], fd, 1e-6);
        }
    }
}

TEST_F(IftUnconstrainedTest, shiftedRosenbrockAndHessianSymmetry) {
    const std::vector<double> m{1.0};
    const std::vector<double> x_hat{1.0, 1.0};

    // Hessian symmetry + exactness vs stan::math::hessian
    quantape::math::IftResult ift;
    std::vector<double> dp_dm;
    quantape::math::iftUnconstrained(ShiftedRosenbrock{}, x_hat, m, dp_dm, ift);
    CHECK_CLOSE("rosen-shift dp0/dm", dp_dm[0], 1.0, 1e-9);
    CHECK_CLOSE("rosen-shift dp1/dm", dp_dm[1], 2.0, 1e-9);
    EXPECT_TRUE(!ift.regularized);

    // Symmetry: dense Hessian H_01 == H_10 (via detail helper)
    std::vector<stan::math::fvar<var>> theta;
    std::vector<double> e, col, H;
    const auto fx = [&](const auto& th) { return ShiftedRosenbrock{}(th, m); };
    quantape::math::detail::denseHessian(fx, x_hat, H, theta, e, col);
    CHECK_CLOSE("rosen-shift H symmetry", H[1], H[2], 1e-10);

    // FD bump-and-recalibrate
    quantape::math::StopCriteria criteria;
    criteria.grad_tol = 1e-12;
    criteria.maxeval = 100000;
    quantape::math::TNewton<var> solver(criteria);
    const double h = 1e-4;
    std::vector<double> mp{1.0 + h}, mm{1.0 - h};
    const auto fp = [&](const auto& th) { return ShiftedRosenbrock{}(th, mp); };
    std::vector<double> xp{-1.2, 1.0};
    EXPECT_TRUE(converged(solver.minimize(fp, xp)));
    const auto fm = [&](const auto& th) { return ShiftedRosenbrock{}(th, mm); };
    std::vector<double> xm{-1.2, 1.0};
    EXPECT_TRUE(converged(solver.minimize(fm, xm)));
    CHECK_CLOSE("rosen-shift dp0/dm FD", dp_dm[0], (xp[0] - xm[0]) / (2.0 * h), 1e-6);
    CHECK_CLOSE("rosen-shift dp1/dm FD", dp_dm[1], (xp[1] - xm[1]) / (2.0 * h), 1e-6);
}

TEST_F(IftKktTest, activeBoundAndOneSidedFiniteDifference) {
    const std::vector<double> m{-0.5};
    const std::vector<double> x_hat{0.0};
    const std::vector<double> lambda{0.5};

    quantape::math::IftResult ift;
    std::vector<double> dp_dm, dlam_dm, dnu_dm;
    quantape::math::iftKkt(BoundQuadratic{}, NonNegative{}, quantape::math::NoConstraint{},
                           quantape::math::Bounds{}, x_hat, m, lambda, {}, dp_dm, dlam_dm, dnu_dm,
                           ift);
    EXPECT_TRUE(ift.activeInequalities.size() == 1);
    CHECK_CLOSE("bound-active dp/dm", dp_dm[0], 0.0, 1e-12);
    CHECK_CLOSE("bound-active dlam/dm", dlam_dm[0], -1.0, 1e-12);

    // One-sided FD (moves deeper into the active side): p stays 0
    {
        quantape::math::StopCriteria criteria;
        criteria.grad_tol = 1e-12;
        criteria.maxeval = 100000;
        quantape::math::SLSQP<var> solver(criteria);
        const auto fx = [&](const auto& th) { return BoundQuadratic{}(th, m); };
        const auto gx = [&](const auto& th, auto& out) { NonNegative{}(th, m, out); };
        std::vector<double> xp = x_hat;
        EXPECT_TRUE(solver.minimize(fx, gx, quantape::math::Bounds{}, xp) ==
                    quantape::math::OptimizeResult::Success);
        EXPECT_TRUE(std::fabs(xp[0]) < 1e-10);
        CHECK_CLOSE("bound-active one-sided FD", 0.0, (xp[0] - x_hat[0]) / (-1e-4), 1e-10);
    }
}

TEST_F(IftKktTest, inactiveInequalityFallback) {
    const std::vector<double> m{2.0};
    const std::vector<double> x_hat{2.0};
    quantape::math::IftResult ift;
    std::vector<double> dp_dm, dlam_dm, dnu_dm;
    quantape::math::iftKkt(BoundQuadratic{}, UpperTen{}, quantape::math::NoConstraint{},
                           quantape::math::Bounds{}, x_hat, m, {0.0}, {}, dp_dm, dlam_dm, dnu_dm,
                           ift);
    EXPECT_TRUE(ift.activeInequalities.empty());
    CHECK_CLOSE("inactive-ineq dp/dm", dp_dm[0], 1.0, 1e-12);
    EXPECT_TRUE(dlam_dm.empty());
}

TEST_F(IftKktTest, equalityProjectionAndMultiplierFiniteDifference) {
    const std::vector<double> m{1.0, 3.0};
    const std::vector<double> x_hat{-1.0, 1.0};
    const std::vector<double> nu{2.0}; // nu = (m0+m1)/2

    quantape::math::IftResult ift;
    std::vector<double> dp_dm, dlam_dm, dnu_dm;
    quantape::math::iftKkt(TwoQuad{}, quantape::math::NoConstraint{}, SumZero{},
                           quantape::math::Bounds{}, x_hat, m, {}, nu, dp_dm, dlam_dm, dnu_dm, ift);
    CHECK_CLOSE("eq dp0/dm0", dp_dm[0], 0.5, 1e-12);
    CHECK_CLOSE("eq dp0/dm1", dp_dm[1], -0.5, 1e-12);
    CHECK_CLOSE("eq dp1/dm0", dp_dm[2], -0.5, 1e-12);
    CHECK_CLOSE("eq dp1/dm1", dp_dm[3], 0.5, 1e-12);
    CHECK_CLOSE("eq dnu/dm0", dnu_dm[0], 0.5, 1e-12);
    CHECK_CLOSE("eq dnu/dm1", dnu_dm[1], 0.5, 1e-12);

    // FD of nu via re-optimization (SLSQP exports multipliers)
    quantape::math::StopCriteria criteria;
    criteria.grad_tol = 1e-12;
    criteria.maxeval = 100000;
    quantape::math::SLSQP<var> solver(criteria);
    const double h = 1e-4;
    for (std::size_t j = 0; j < 2; ++j) {
        std::vector<double> mj = m;
        mj[j] += h;
        const auto fx = [&](const auto& th) { return TwoQuad{}(th, mj); };
        const auto hx = [&](const auto& th, auto& out) { SumZero{}(th, mj, out); };
        std::vector<double> xp = x_hat;
        quantape::math::OptimizerState state;
        EXPECT_TRUE(solver.minimize(fx, quantape::math::NoConstraint{}, hx,
                                    quantape::math::Bounds{}, xp,
                                    state) == quantape::math::OptimizeResult::Success);
        CHECK_CLOSE("eq dnu/dm FD", dnu_dm[j], (state.eq_multipliers[0] - nu[0]) / h, 1e-6);
        for (std::size_t i = 0; i < 2; ++i)
            CHECK_CLOSE("eq dp/dm FD", dp_dm[i * 2 + j], (xp[i] - x_hat[i]) / h, 1e-6);
    }
}

TEST_F(IftKktTest, combinedActiveAndAugLagMultiplierExport) {
    const std::vector<double> m{2.0, -1.0};
    const std::vector<double> x_hat{0.8, 0.2};
    const std::vector<double> lambda{2.4};
    const std::vector<double> nu{-1.2};

    quantape::math::IftResult ift;
    std::vector<double> dp_dm, dlam_dm, dnu_dm;
    quantape::math::iftKkt(SepQuad{}, X0Cap{}, SumOne{}, quantape::math::Bounds{}, x_hat, m, lambda,
                           nu, dp_dm, dlam_dm, dnu_dm, ift);
    EXPECT_TRUE(ift.activeInequalities.size() == 1);
    for (std::size_t i = 0; i < 4; ++i)
        CHECK_CLOSE("combined dp/dm", dp_dm[i], 0.0, 1e-12);
    CHECK_CLOSE("combined dlam/dm0", dlam_dm[0], 1.0, 1e-12);
    CHECK_CLOSE("combined dlam/dm1", dlam_dm[1], -1.0, 1e-12);
    CHECK_CLOSE("combined dnu/dm0", dnu_dm[0], 0.0, 1e-12);
    CHECK_CLOSE("combined dnu/dm1", dnu_dm[1], 1.0, 1e-12);

    // Full pipeline: minimizeDifferential (SLSQP) — multiplier export path
    quantape::math::StopCriteria criteria;
    criteria.grad_tol = 1e-12;
    criteria.maxeval = 100000;
    std::vector<double> x{0.0, 0.0};
    quantape::math::OptimizerState state;
    quantape::math::IftResult ift2;
    std::vector<double> dp2, dl2, dn2;
    const quantape::math::OptimizeResult r =
        quantape::math::minimizeDifferential(SepQuad{}, X0Cap{}, SumOne{}, quantape::math::Bounds{},
                                             m, x, state, ift2, &dp2, &dl2, &dn2, criteria);
    EXPECT_TRUE(r == quantape::math::OptimizeResult::Success);
    EXPECT_TRUE(state.ineq_multipliers.size() == 1);
    EXPECT_TRUE(state.eq_multipliers.size() == 1);
    CHECK_CLOSE("pipeline lambda", state.ineq_multipliers[0], 2.4, 1e-8);
    CHECK_CLOSE("pipeline nu", state.eq_multipliers[0], -1.2, 1e-8);
    for (std::size_t i = 0; i < 4; ++i)
        CHECK_CLOSE("pipeline dp/dm", dp2[i], 0.0, 1e-8);
    CHECK_CLOSE("pipeline dlam/dm0", dl2[0], 1.0, 1e-8);
    CHECK_CLOSE("pipeline dnu/dm1", dn2[1], 1.0, 1e-8);

    // One-sided FD of the multipliers near the active set (m0 - eps keeps
    // the inequality active; m1 + eps as well)
    quantape::math::SLSQP<var> solver(criteria);
    for (std::size_t j = 0; j < 2; ++j) {
        std::vector<double> mj = m;
        mj[j] += 1e-4;
        const auto fx = [&](const auto& th) { return SepQuad{}(th, mj); };
        const auto gx = [&](const auto& th, auto& out) { X0Cap{}(th, mj, out); };
        const auto hx = [&](const auto& th, auto& out) { SumOne{}(th, mj, out); };
        std::vector<double> xp = x_hat;
        quantape::math::OptimizerState st;
        EXPECT_TRUE(solver.minimize(fx, gx, hx, quantape::math::Bounds{}, xp, st) ==
                    quantape::math::OptimizeResult::Success);
        CHECK_CLOSE("combined dlam FD", dlam_dm[j], (st.ineq_multipliers[0] - lambda[0]) / 1e-4,
                    1e-6);
        CHECK_CLOSE("combined dnu FD", dnu_dm[j], (st.eq_multipliers[0] - nu[0]) / 1e-4, 1e-6);
    }

    // ========================================================================
    // Fixture 11: AUGLAG multiplier export feeds the same IFT
    // ========================================================================
    {
        const std::vector<double> m_aug{2.0, -1.0};
        quantape::math::StopCriteria augCriteria;
        augCriteria.grad_tol = 1e-12;
        augCriteria.maxeval = 100000;

        quantape::math::AugLag<var> augSolver(augCriteria);
        const auto fx = [&](const auto& th) { return SepQuad{}(th, m_aug); };
        const auto gx = [&](const auto& th, auto& out) { X0Cap{}(th, m_aug, out); };
        const auto hx = [&](const auto& th, auto& out) { SumOne{}(th, m_aug, out); };
        std::vector<double> x_aug{0.0, 0.0};
        quantape::math::OptimizerState augState;
        const quantape::math::OptimizeResult r_aug =
            augSolver.minimize(fx, gx, hx, quantape::math::Bounds{}, x_aug, augState);
        EXPECT_TRUE(r_aug == quantape::math::OptimizeResult::Success);
        EXPECT_TRUE(augState.ineq_multipliers.size() == 1);
        CHECK_CLOSE("auglag x0", x_aug[0], 0.8, 1e-7);
        CHECK_CLOSE("auglag x1", x_aug[1], 0.2, 1e-7);

        quantape::math::IftResult augIft;
        std::vector<double> augDp, augDlam, augDnu;
        quantape::math::iftKkt(SepQuad{}, X0Cap{}, SumOne{}, quantape::math::Bounds{}, x_aug, m_aug,
                               augState.ineq_multipliers, augState.eq_multipliers, augDp, augDlam,
                               augDnu, augIft);
        for (std::size_t i = 0; i < 4; ++i)
            CHECK_CLOSE("auglag dp/dm", augDp[i], 0.0, 1e-6);
        CHECK_CLOSE("auglag dlam/dm0", augDlam[0], 1.0, 1e-6);
        CHECK_CLOSE("auglag dnu/dm1", augDnu[1], 1.0, 1e-6);
    }
}

TEST_F(IftKktTest, degeneratePseudoInverseFallback) {
    const std::vector<double> m{2.0};
    const std::vector<double> x_hat{1.0};
    // KKT at x = 1: grad f = x - m = -1; stationarity needs lambda1 + 2 lambda2 = 1.
    // BOTH rows strictly active (0.5, 0.25): J_A = [1; 2] is rank 1 -> the
    // complementarity rows are linearly dependent and the system is singular.
    const std::vector<double> lambda{0.5, 0.25};

    quantape::math::IftResult ift;
    std::vector<double> dp_dm, dlam_dm, dnu_dm;
    quantape::math::iftKkt(BoundQuadratic{}, DoubleIneq{}, quantape::math::NoConstraint{},
                           quantape::math::Bounds{}, x_hat, m, lambda, {}, dp_dm, dlam_dm, dnu_dm,
                           ift);
    EXPECT_TRUE(ift.activeInequalities.size() == 2);
    EXPECT_TRUE(ift.pseudo_inverse);
    EXPECT_TRUE(ift.rank < 3);
    CHECK_CLOSE("degenerate dp/dm", dp_dm[0], 0.0, 1e-8);
}

TEST_F(IftKktTest, boundsOnlyActiveSet) {
    const std::vector<double> m{-0.5, 3.0};
    const std::vector<double> x_hat{0.0, 3.0};
    quantape::math::Bounds b = quantape::math::Bounds::fromVectors({0.0, -1e30}, {1e30, 1e30});

    quantape::math::IftResult ift;
    std::vector<double> dp_dm, dlam_dm, dnu_dm;
    quantape::math::iftKkt(TwoQuad{}, quantape::math::NoConstraint{},
                           quantape::math::NoConstraint{}, b, x_hat, m, {}, {}, dp_dm, dlam_dm,
                           dnu_dm, ift);
    EXPECT_TRUE(ift.activeBounds.size() == 1);
    EXPECT_TRUE(ift.activeBounds[0] == 0);
    CHECK_CLOSE("bounds dp0/dm0", dp_dm[0], 0.0, 1e-12);
    CHECK_CLOSE("bounds dp0/dm1", dp_dm[1], 0.0, 1e-12);
    CHECK_CLOSE("bounds dp1/dm0", dp_dm[2], 0.0, 1e-12);
    CHECK_CLOSE("bounds dp1/dm1", dp_dm[3], 1.0, 1e-12);
}

TEST_F(IftUnconstrainedTest, illConditionedConditionNumber) {
    // f2 = 1/2 (x0 - m0)^2 + 1/2 * eps * (x1 - m1)^2 with eps = 1e-8
    const double eps = 1e-8;
    const auto f2 = [eps](const auto& x, const auto& m) {
        using Sx = typename std::decay_t<decltype(x)>::value_type;
        const Sx d0 = x[0] - Sx(m[0]);
        const Sx d1 = x[1] - Sx(m[1]);
        return Sx(0.5) * (d0 * d0 + Sx(eps) * d1 * d1);
    };
    const std::vector<double> m{1.0, 2.0};
    const std::vector<double> x_hat{1.0, 2.0};
    quantape::math::IftResult ift;
    std::vector<double> dp_dm;
    quantape::math::iftUnconstrained(f2, x_hat, m, dp_dm, ift);
    CHECK_CLOSE("cond number", ift.conditionNumber, 1.0 / eps, 1e-6);
    EXPECT_TRUE(!ift.regularized);
    CHECK_CLOSE("ill-cond dp/dm diagonal", dp_dm[0], 1.0, 1e-12);
}

TEST_F(IftCompositionTest, unconstrainedAndConstrainedVarComposition) {
    // ========================================================================
    // Fixture 10: var composition on the caller's tape
    //   downstream phi = w . p_hat; grad_m phi must equal w^T dp/dm exactly
    // ========================================================================
    {
        const std::vector<std::vector<double>> a{{1.0, 0.2}, {0.5, 1.0}, {0.1, 0.4}};
        const std::vector<double> m{0.8, 1.2, -0.5};
        LsqObjective obj{a};

        std::vector<var> m_var(3);
        for (std::size_t j = 0; j < 3; ++j)
            m_var[j] = m[j];
        std::vector<var> p_hat;
        const quantape::math::OptimizeResult r = quantape::math::minimizeDifferentialVar(
            obj, quantape::math::NoConstraint{}, quantape::math::NoConstraint{},
            quantape::math::Bounds{}, m_var, {0.0, 0.0}, p_hat, nullptr, nullptr,
            quantape::math::StopCriteria{});
        EXPECT_TRUE(converged(r));
        EXPECT_TRUE(p_hat.size() == 2);

        const std::vector<double> w{0.3, -0.7};
        var phi = w[0] * p_hat[0] + w[1] * p_hat[1];
        phi.grad();

        // reference: analytic dp/dm for this LSQ
        Eigen::MatrixXd A(3, 2);
        for (std::size_t k = 0; k < 3; ++k)
            for (std::size_t i = 0; i < 2; ++i)
                A(static_cast<Eigen::Index>(k), static_cast<Eigen::Index>(i)) = a[k][i];
        Eigen::MatrixXd d = (A.transpose() * A).ldlt().solve(A.transpose());
        for (std::size_t j = 0; j < 3; ++j) {
            double ref = 0.0;
            for (std::size_t i = 0; i < 2; ++i)
                ref += w[i] * d(static_cast<Eigen::Index>(i), static_cast<Eigen::Index>(j));
            CHECK_CLOSE("var composition grad_m", m_var[j].adj(), ref, 1e-10);
        }
    }

    // ========================================================================
    // Fixture 14: constrained var composition (SLSQP export wiring inside
    // minimizeDifferentialVar): phi = p0 - p1 with both constraints active ->
    // dp/dm = 0 -> grad_m phi = 0 exactly
    // ========================================================================
    {
        const std::vector<double> m{2.0, -1.0};
        std::vector<var> m_var(2);
        m_var[0] = m[0];
        m_var[1] = m[1];
        std::vector<var> p_hat;
        quantape::math::IftResult ift;
        quantape::math::OptimizerState state;
        const quantape::math::OptimizeResult r = quantape::math::minimizeDifferentialVar(
            SepQuad{}, X0Cap{}, SumOne{}, quantape::math::Bounds{}, m_var, {0.0, 0.0}, p_hat, &ift,
            &state, quantape::math::StopCriteria{});
        EXPECT_TRUE(r == quantape::math::OptimizeResult::Success);
        CHECK_CLOSE("constrained var p0", p_hat[0].val(), 0.8, 1e-7);
        CHECK_CLOSE("constrained var p1", p_hat[1].val(), 0.2, 1e-7);
        EXPECT_TRUE(ift.activeInequalities.size() == 1);

        var phi = p_hat[0] - p_hat[1];
        phi.grad();
        CHECK_CLOSE("constrained var grad_m0", m_var[0].adj(), 0.0, 1e-8);
        CHECK_CLOSE("constrained var grad_m1", m_var[1].adj(), 0.0, 1e-8);
    }
}

TEST_F(IftKktTest, nonlinearUnitDiskConstraint) {
    const std::vector<double> m{1.5, 0.0};
    const std::vector<double> x_hat{1.0, 0.0};
    const std::vector<double> lambda{0.25};

    quantape::math::IftResult ift;
    std::vector<double> dp_dm, dlam_dm, dnu_dm;
    quantape::math::iftKkt(TwoQuad{}, UnitDisk{}, quantape::math::NoConstraint{},
                           quantape::math::Bounds{}, x_hat, m, lambda, {}, dp_dm, dlam_dm, dnu_dm,
                           ift);
    EXPECT_TRUE(ift.activeInequalities.size() == 1);
    CHECK_CLOSE("disk dp0/dm0", dp_dm[0], 0.0, 1e-12);
    CHECK_CLOSE("disk dp0/dm1", dp_dm[1], 0.0, 1e-12);
    CHECK_CLOSE("disk dp1/dm0", dp_dm[2], 0.0, 1e-12);
    CHECK_CLOSE("disk dp1/dm1", dp_dm[3], 2.0 / 3.0, 1e-12);
    CHECK_CLOSE("disk dlam/dm0", dlam_dm[0], 0.5, 1e-12);
    CHECK_CLOSE("disk dlam/dm1", dlam_dm[1], 0.0, 1e-12);

    // One-sided FD of p and lambda
    quantape::math::StopCriteria criteria;
    criteria.grad_tol = 1e-12;
    criteria.maxeval = 100000;
    quantape::math::SLSQP<var> solver(criteria);
    const double h = 1e-4;
    {
        std::vector<double> mj{1.5, h};
        const auto fx = [&](const auto& th) { return TwoQuad{}(th, mj); };
        const auto gx = [&](const auto& th, auto& out) { UnitDisk{}(th, mj, out); };
        std::vector<double> xp = x_hat;
        quantape::math::OptimizerState st;
        // SLSQP nails the point (x1 -> 2/3*h) but reports RoundoffLimited at
        // the curved boundary (merit search); the primal FD is still exact
        const auto rj = solver.minimize(fx, gx, quantape::math::Bounds{}, xp, st);
        EXPECT_TRUE(converged(rj) || rj == quantape::math::OptimizeResult::RoundoffLimited);
        CHECK_CLOSE("disk dp1/dm1 FD", dp_dm[3], (xp[1] - x_hat[1]) / h, 1e-6);
    }
    {
        std::vector<double> mj{1.5 + h, 0.0};
        const auto fx = [&](const auto& th) { return TwoQuad{}(th, mj); };
        const auto gx = [&](const auto& th, auto& out) { UnitDisk{}(th, mj, out); };
        std::vector<double> xp = x_hat;
        quantape::math::OptimizerState st;
        const auto rj = solver.minimize(fx, gx, quantape::math::Bounds{}, xp, st);
        EXPECT_TRUE(converged(rj) || rj == quantape::math::OptimizeResult::RoundoffLimited);
        CHECK_CLOSE("disk dp0/dm0 FD", dp_dm[0], (xp[0] - x_hat[0]) / h, 1e-6);
    }
}

TEST_F(IftUnconstrainedTest, ridgeEscalationAndFlatPositiveDefinite) {
    // Fixture 13: ridge escalation (indefinite H)
    //   f2 = -1/2 (x0 - m0)^2: H = -1 < 0 -> automatic ridge. At a genuine
    //   maximum the regularized value is a documented distortion; assert the
    //   flags and finiteness, plus the near-flat-PD boundary case below.
    const auto f2 = [](const auto& x, const auto& m) {
        using Sx = typename std::decay_t<decltype(x)>::value_type;
        const Sx d = x[0] - Sx(m[0]);
        return Sx(-0.5) * d * d;
    };
    quantape::math::IftResult ift;
    std::vector<double> dp_dm;
    quantape::math::iftUnconstrained(f2, {1.0}, {1.0}, dp_dm, ift);
    EXPECT_TRUE(ift.regularized);
    EXPECT_TRUE(ift.ridgeUsed > 0.0);
    EXPECT_TRUE(std::isfinite(dp_dm[0]));

    // Near-flat but positive definite: no ridge needed, exact answer
    const auto flat = [](const auto& x, const auto& m) {
        using Sx = typename std::decay_t<decltype(x)>::value_type;
        const Sx d0 = x[0] - Sx(m[0]);
        const Sx d1 = x[1] - Sx(m[1]);
        return Sx(0.5) * (d0 * d0 + Sx(1e-16) * d1 * d1);
    };
    quantape::math::IftResult ift2;
    std::vector<double> dp2;
    quantape::math::iftUnconstrained(flat, {1.0, 2.0}, {1.0, 2.0}, dp2, ift2);
    EXPECT_TRUE(!ift2.regularized);
    CHECK_CLOSE("flat-PD dp0/dm0", dp2[0], 1.0, 1e-12);
    CHECK_CLOSE("flat-PD dp1/dm1", dp2[3], 1.0, 1e-12);
}
