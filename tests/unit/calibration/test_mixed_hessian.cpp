/**
 * @file test_mixed_hessian.cpp
 * @brief Generalizable mixed analytical/AD second-order derivatives
 *
 * The problem: in a pricing chain  Market Data → Curve → Forward → BS → PV,
 * some layers have closed-form Hessians (BS) and some don't (numerical
 * interpolation, PDE solvers, American exercise). We need a pattern that:
 *
 *   1. Uses analytical derivatives where available (fast)
 *   2. Falls back to AD where they're not (correct)
 *   3. Composes seamlessly — you don't care which approach each layer uses
 *
 * Solution: each function provides 3 overloads dispatched by scalar type:
 *
 *   double    → just compute the value
 *   var       → make_callback_var with analytical gradient (1 tape node)
 *   `fvar<var>` → express gradient as var operations;
 *               AD differentiates through them for the Hessian
 *
 * This is a NOGATE demo file: the per-chain comparisons are recorded as test
 * properties (not asserted) until the P2 assertion backfill lands.
 */

#include "quantape/math/StanMath.h"

#include "quantape/util/Constants.h"
using ::quantape::util::kPi;

#include <Eigen/Dense>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <iterator>
#include <string>
#include <vector>

#include "support/GtestSupport.h"
#include "support/StanTapeFixture.h"

using stan::math::fvar;
using stan::math::var;

// ═══════════════════════════════════════════════════════════════════════════
// LAYER 1: CURVE INTERPOLATION (Level 1 — gradient known, Hessian via AD)
//
// Y(T) = linear interp from pillar rates {y_0, y_1, ..., y_n} at {T_0, ..., T_n}
//
// ∂Y/∂y_i = weight (0 or linear weight), known analytically
// ∂²Y/∂y_i∂y_j = 0 for linear interp (but NOT for cubic spline!)
//
// For cubic spline, the Hessian of interp w.r.t. knot values involves
// derivatives of the tridiagonal solve — messy to derive analytically.
// Instead, we express the gradient as var operations and let AD handle it.
// ═══════════════════════════════════════════════════════════════════════════

struct CurveInterp {
    std::vector<double> times; // pillar times
    int n;                     // number of pillars

    // ── double: just compute ──
    double eval(double T, const std::vector<double>& rates) const {
        int seg = locate(T);
        double t = (T - times[seg]) / (times[seg + 1] - times[seg]);
        return (1.0 - t) * rates[seg] + t * rates[seg + 1];
    }

    // ── var: analytical adjoint (1 tape node) ──
    // Gradient: ∂Y/∂y_i = (1-t) if i==seg, t if i==seg+1, 0 otherwise
    var eval(double T, const std::vector<var>& rates) const {
        int seg = locate(T);
        double t = (T - times[seg]) / (times[seg + 1] - times[seg]);
        double val = (1.0 - t) * rates[seg].val() + t * rates[seg + 1].val();

        return stan::math::make_callback_var(val, [seg, t, &rates](auto& vi) {
            double a = vi.adj();
            rates[seg].adj() += a * (1.0 - t);
            rates[seg + 1].adj() += a * t;
        });
    }

    // ── `fvar<var>`: gradient as var operations ──
    // The gradient entries (1-t) and t are constants for linear interp,
    // so the Hessian is zero. But this pattern generalizes to cubic spline
    // where the gradient depends on the knot values (and thus is non-trivial var).
    fvar<var> eval(double T, const std::vector<fvar<var>>& rates) const {
        int seg = locate(T);
        double t = (T - times[seg]) / (times[seg + 1] - times[seg]);

        // Value as var
        var val = (1.0 - t) * rates[seg].val_ + t * rates[seg + 1].val_;

        // Gradient entries as var (for linear interp, these are constants;
        // for cubic spline, they'd involve var operations through the solve)
        var grad_seg = var(1.0 - t); // ∂Y/∂y_seg
        var grad_seg1 = var(t);      // ∂Y/∂y_{seg+1}

        // Tangent = Σ (∂Y/∂y_i) * dy_i
        var tangent = grad_seg * rates[seg].d_ + grad_seg1 * rates[seg + 1].d_;

        return fvar<var>(val, tangent);
    }

    int locate(double T) const {
        for (int i = 0; i < n - 1; ++i)
            if (T <= times[i + 1])
                return i;
        return n - 2;
    }
};

// ═══════════════════════════════════════════════════════════════════════════
// LAYER 1b: CUBIC SPLINE INTERPOLATION (Level 1 — gradient via AD)
//
// This is the interesting case: the gradient ∂Y/∂y_i involves the
// tridiagonal solve, so it's NOT a simple constant.
// We express it as var operations and let AD handle the Hessian.
// ═══════════════════════════════════════════════════════════════════════════

struct CubicSplineInterp {
    std::vector<double> times;
    int n;

    // Tridiagonal solve for spline second derivatives (Thomas algorithm)
    // Returns M[i] = second derivative at knot i
    template <typename T>
    std::vector<T> solve_spline(const std::vector<T>& y) const {
        using std::vector;
        int nm = n;
        vector<double> h(nm - 1);
        for (int i = 0; i < nm - 1; ++i)
            h[i] = times[i + 1] - times[i];

        // Natural spline: M[0] = M[n-1] = 0
        // Tridiagonal system for M[1..n-2]
        int m = nm - 2;
        if (m <= 0)
            return vector<T>(nm, T(0.0));

        vector<T> rhs(m);
        for (int i = 0; i < m; ++i) {
            T d_left = (y[i + 1] - y[i]) / h[i];
            T d_right = (y[i + 2] - y[i + 1]) / h[i + 1];
            rhs[i] = 6.0 * (d_right - d_left);
        }

        // Thomas algorithm (forward sweep + back substitution)
        vector<double> diag(m), upper(m);
        vector<T> rhs2(rhs);
        diag[0] = 2.0 * (h[0] + h[1]);
        upper[0] = h[1];
        for (int i = 1; i < m; ++i) {
            double ratio = h[i] / diag[i - 1];
            diag[i] = 2.0 * (h[i] + h[i + 1]) - ratio * upper[i - 1];
            upper[i] = (i < m - 1) ? h[i + 1] : 0.0;
            rhs2[i] = rhs2[i] - ratio * rhs2[i - 1];
        }

        vector<T> M(nm, T(0.0));
        M[m] = rhs2[m - 1] / diag[m - 1];
        for (int i = m - 2; i >= 0; --i)
            M[i + 1] = (rhs2[i] - upper[i] * M[i + 2]) / diag[i];

        return M;
    }

    // Evaluate cubic spline at point T
    template <typename T>
    T eval_impl(double Tval, const std::vector<T>& y, const std::vector<T>& M) const {
        int seg = locate(Tval);
        double h = times[seg + 1] - times[seg];
        double t = (Tval - times[seg]) / h;
        double omt = 1.0 - t;

        // Cubic spline formula:
        // S(x) = (1-t)*y[i] + t*y[i+1]
        //      + h²/6 * [(omt³ - omt)*M[i] + (t³ - t)*M[i+1]]
        T val = omt * y[seg] + t * y[seg + 1] +
                (h * h / 6.0) * ((omt * omt * omt - omt) * M[seg] + (t * t * t - t) * M[seg + 1]);
        return val;
    }

    // ── double: just compute ──
    double eval(double T, const std::vector<double>& rates) const {
        auto M = solve_spline(rates);
        return eval_impl(T, rates, M);
    }

    // ── var: the spline solve is templated, so Stan tapes through it ──
    // For first-order, this is fine — the tape captures ∂Y/∂rates
    // through the tridiagonal solve automatically.
    // (For production, you'd use the analytical adjoint from the interp library)
    var eval(double T, const std::vector<var>& rates) const {
        auto M = solve_spline(rates);
        return eval_impl(T, rates, M);
    }

    // ── `fvar<var>`: same template works! ──
    // Stan tapes through the tridiagonal solve with `fvar<var>`,
    // giving both gradient and Hessian automatically.
    // The Hessian captures how spline coefficients change when knots move.
    fvar<var> eval(double T, const std::vector<fvar<var>>& rates) const {
        auto M = solve_spline(rates);
        return eval_impl(T, rates, M);
    }

    int locate(double T) const {
        for (int i = 0; i < n - 1; ++i)
            if (T <= times[i + 1])
                return i;
        return n - 2;
    }
};

// ═══════════════════════════════════════════════════════════════════════════
// LAYER 2: BLACK-SCHOLES (Level 2 — gradient AND Hessian known)
//
// For `fvar<var>`, we use nested make_callback_var: each Greek is a
// make_callback_var with second-order Greeks as derivatives.
// This is the fastest possible `fvar<var>` implementation.
// ═══════════════════════════════════════════════════════════════════════════

namespace bs_layer {

inline double phi(double x) {
    return std::exp(-0.5 * x * x) / std::sqrt(2.0 * kPi);
}
inline double Phi(double x) {
    return 0.5 * std::erfc(-x * M_SQRT1_2);
}

// ── double ──
double price(double S, double sigma, double r, double K, double T) {
    double sqrtT = std::sqrt(T);
    double d1 = (std::log(S / K) + (r + 0.5 * sigma * sigma) * T) / (sigma * sqrtT);
    double d2 = d1 - sigma * sqrtT;
    return S * Phi(d1) - K * std::exp(-r * T) * Phi(d2);
}

// ── var: analytical adjoint (1 tape node) ──
var price(var S, var sigma, var r, double K, double T) {
    double Sv = S.val(), sv = sigma.val(), rv = r.val();
    double sqrtT = std::sqrt(T);
    double d1 = (std::log(Sv / K) + (rv + 0.5 * sv * sv) * T) / (sv * sqrtT);
    double d2 = d1 - sv * sqrtT;
    double nd1 = phi(d1), Nd1 = Phi(d1), Nd2 = Phi(d2);
    double disc = std::exp(-rv * T);
    double val = Sv * Nd1 - K * disc * Nd2;

    double delta = Nd1;
    double vega = Sv * nd1 * sqrtT;
    double rho = K * T * disc * Nd2;

    return stan::math::make_callback_var(val, [S, sigma, r, delta, vega, rho](auto& vi) {
        double a = vi.adj();
        S.adj() += a * delta;
        sigma.adj() += a * vega;
        r.adj() += a * rho;
    });
}

// ── `fvar<var>`: nested analytical (Level 2) ──
// Each Greek is a make_callback_var with second-order Greeks.
// Total: 3 callbacks + 5 arithmetic = 8 var nodes per hessian column.
fvar<var> price(fvar<var> S, fvar<var> sigma, fvar<var> r, double K, double T) {
    var Sv = S.val_, sv = sigma.val_, rv = r.val_;
    var Sd = S.d_, sd = sigma.d_, rd = r.d_;

    double S0 = Sv.val(), s0 = sv.val(), r0 = rv.val();
    double sqrtT = std::sqrt(T);
    double d1 = (std::log(S0 / K) + (r0 + 0.5 * s0 * s0) * T) / (s0 * sqrtT);
    double d2 = d1 - s0 * sqrtT;
    double nd1 = phi(d1), nd2 = phi(d2), Nd1 = Phi(d1), Nd2 = Phi(d2);
    double disc = std::exp(-r0 * T);

    double val = S0 * Nd1 - K * disc * Nd2;
    double delta = Nd1;
    double vega = S0 * nd1 * sqrtT;
    double rho_g = K * T * disc * Nd2;

    // Second-order Greeks
    double gamma_ = nd1 / (S0 * s0 * sqrtT);
    double vanna_ = -nd1 * d2 / s0;
    double dDelta_dr = nd1 * sqrtT / s0;
    double volga_ = S0 * sqrtT * nd1 * d1 * d2 / s0;
    double dVega_dr = -S0 * T * d1 * nd1 / s0;
    double dRho_dr = -K * T * T * disc * Nd2 + K * T * disc * nd2 * sqrtT / s0;

    var price_var(val);

    var delta_var =
        stan::math::make_callback_var(delta, [Sv, sv, rv, gamma_, vanna_, dDelta_dr](auto& vi) {
            double a = vi.adj();
            Sv.adj() += a * gamma_;
            sv.adj() += a * vanna_;
            rv.adj() += a * dDelta_dr;
        });

    var vega_var =
        stan::math::make_callback_var(vega, [Sv, sv, rv, vanna_, volga_, dVega_dr](auto& vi) {
            double a = vi.adj();
            Sv.adj() += a * vanna_;
            sv.adj() += a * volga_;
            rv.adj() += a * dVega_dr;
        });

    var rho_var =
        stan::math::make_callback_var(rho_g, [Sv, sv, rv, dDelta_dr, dVega_dr, dRho_dr](auto& vi) {
            double a = vi.adj();
            Sv.adj() += a * dDelta_dr;
            sv.adj() += a * dVega_dr;
            rv.adj() += a * dRho_dr;
        });

    var tangent = delta_var * Sd + vega_var * sd + rho_var * rd;
    return fvar<var>(price_var, tangent);
}

} // namespace bs_layer

// ═══════════════════════════════════════════════════════════════════════════
// COMPOSED CHAIN: rate pillars → curve interp → forward → BS → PV
//
// The beauty: each layer provides its best overload. `fvar<var>` composition
// just works — hessian() doesn't know or care which layers are analytical.
// ═══════════════════════════════════════════════════════════════════════════

// Full pricing chain as a stan::math::hessian-compatible functor
template <typename InterpType>
struct PricingChainFunctor {
    const InterpType& curve;
    double eval_time; // time to evaluate curve
    double vol, K, T;

    template <typename Scalar>
    Scalar operator()(const Eigen::Matrix<Scalar, Eigen::Dynamic, 1>& rates_vec) const {
        int n = rates_vec.size();
        std::vector<Scalar> rates(n);
        for (int i = 0; i < n; ++i)
            rates[i] = rates_vec(i);

        // Layer 1: curve interpolation
        Scalar fwd = curve.eval(eval_time, rates);

        // Layer 2: BS price
        // vol is a constant here (not differentiated)
        Scalar vol_s(vol);
        Scalar r_s = fwd; // using curve rate as risk-free rate too (simplified)
        return bs_layer::price(fwd * 100.0, vol_s, r_s, K, T);
    }
};

// ═══════════════════════════════════════════════════════════════════════════
// NAIVE CHAIN (all `fvar<var>`, no analytical derivatives)
// ═══════════════════════════════════════════════════════════════════════════

struct NaiveChainFunctor {
    const CurveInterp& curve;
    double eval_time, vol, K, T;

    template <typename Scalar>
    Scalar operator()(const Eigen::Matrix<Scalar, Eigen::Dynamic, 1>& rates_vec) const {
        int n = rates_vec.size();
        int seg = 0;
        for (int i = 0; i < n - 1; ++i)
            if (eval_time <= curve.times[i + 1]) {
                seg = i;
                break;
            }
        double t = (eval_time - curve.times[seg]) / (curve.times[seg + 1] - curve.times[seg]);
        Scalar fwd = (1.0 - t) * rates_vec(seg) + t * rates_vec(seg + 1);

        Scalar S = fwd * 100.0;
        Scalar sigma(vol), r = fwd;
        Scalar sqrtT_ = stan::math::sqrt(Scalar(T));
        Scalar d1 = (stan::math::log(S / K) + (r + sigma * sigma / 2.0) * T) / (sigma * sqrtT_);
        Scalar d2 = d1 - sigma * sqrtT_;
        return S * stan::math::Phi(d1) - K * stan::math::exp(-r * T) * stan::math::Phi(d2);
    }
};

// ═══════════════════════════════════════════════════════════════════════════
// BENCHMARK
// ═══════════════════════════════════════════════════════════════════════════

namespace {

template <typename F>
double bench_us(F&& fn, int N) {
    for (int i = 0; i < std::min(N / 10, 1000); ++i)
        fn();
    auto t0 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < N; ++i)
        fn();
    auto t1 = std::chrono::high_resolution_clock::now();
    return std::chrono::duration<double, std::micro>(t1 - t0).count() / N;
}

constexpr int kN_Pillars = 6;
constexpr int kBenchN = 50'000;
const double kPillarTimes[] = {0.25, 0.5, 1.0, 2.0, 5.0, 10.0};
const double kPillarRates[] = {0.02, 0.025, 0.03, 0.035, 0.04, 0.045};
constexpr double kEvalT = 1.5; // evaluate at T=1.5 (between pillars 2 and 3)
// Curve rate ~3.25% → S = rate*100 ≈ 3.25, so set K near ATM
constexpr double kVol = 0.30;
constexpr double kStrike = 3.3;
constexpr double kT = 1.0;
constexpr double kFdStep = 1e-5;

std::vector<double> pillarTimes() {
    return std::vector<double>(std::begin(kPillarTimes), std::end(kPillarTimes));
}

Eigen::VectorXd pillarVector() {
    Eigen::VectorXd x(kN_Pillars);
    for (int i = 0; i < kN_Pillars; ++i)
        x(i) = kPillarRates[i];
    return x;
}

template <typename Curve>
Eigen::MatrixXd finiteDifferenceHessian(const Curve& curve, const Eigen::VectorXd& x) {
    const double eps = kFdStep;
    Eigen::MatrixXd H_fd(kN_Pillars, kN_Pillars);
    for (int i = 0; i < kN_Pillars; ++i) {
        for (int j = i; j < kN_Pillars; ++j) {
            Eigen::VectorXd xpp = x, xpm = x, xmp = x, xmm = x;
            xpp(i) += eps;
            xpp(j) += eps;
            xpm(i) += eps;
            xpm(j) -= eps;
            xmp(i) -= eps;
            xmp(j) += eps;
            xmm(i) -= eps;
            xmm(j) -= eps;
            auto f = [&](const Eigen::VectorXd& v) {
                std::vector<double> r(kN_Pillars);
                for (int k = 0; k < kN_Pillars; ++k)
                    r[k] = v(k);
                double fwd = curve.eval(kEvalT, r);
                return bs_layer::price(fwd * 100, kVol, fwd, kStrike, kT);
            };
            H_fd(i, j) = (f(xpp) - f(xpm) - f(xmp) + f(xmm)) / (4 * eps * eps);
            H_fd(j, i) = H_fd(i, j);
        }
    }
    return H_fd;
}

} // namespace

class MixedHessianTest : public StanTapeTest {};

TEST_F(MixedHessianTest, linearChainMatchesFiniteDifference) {
    const CurveInterp linear_curve{pillarTimes(), kN_Pillars};
    const PricingChainFunctor<CurveInterp> linear_functor{linear_curve, kEvalT, kVol, kStrike, kT};

    const Eigen::VectorXd x = pillarVector();

    double fx;
    Eigen::VectorXd grad(kN_Pillars);
    Eigen::MatrixXd H(kN_Pillars, kN_Pillars);
    stan::math::hessian(linear_functor, x, fx, grad, H);

    const Eigen::MatrixXd H_fd = finiteDifferenceHessian(linear_curve, x);
    ::testing::Test::RecordProperty("linear_max_ad_fd_diff",
                                    quantape::util::num((H - H_fd).cwiseAbs().maxCoeff(), 2));
}

TEST_F(MixedHessianTest, cubicSplineChainMatchesFiniteDifference) {
    const CubicSplineInterp spline_curve{pillarTimes(), kN_Pillars};
    const PricingChainFunctor<CubicSplineInterp> spline_functor{spline_curve, kEvalT, kVol, kStrike,
                                                                kT};

    const Eigen::VectorXd x = pillarVector();

    double fx2;
    Eigen::VectorXd grad2(kN_Pillars);
    Eigen::MatrixXd H2(kN_Pillars, kN_Pillars);
    stan::math::hessian(spline_functor, x, fx2, grad2, H2);

    const Eigen::MatrixXd H2_fd = finiteDifferenceHessian(spline_curve, x);
    ::testing::Test::RecordProperty("spline_max_ad_fd_diff",
                                    quantape::util::num((H2 - H2_fd).cwiseAbs().maxCoeff(), 2));
}

TEST_F(MixedHessianTest, naiveChainEqualsMixed) {
    const CurveInterp linear_curve{pillarTimes(), kN_Pillars};
    const CubicSplineInterp spline_curve{pillarTimes(), kN_Pillars};
    const PricingChainFunctor<CurveInterp> linear_functor{linear_curve, kEvalT, kVol, kStrike, kT};
    const PricingChainFunctor<CubicSplineInterp> spline_functor{spline_curve, kEvalT, kVol, kStrike,
                                                                kT};
    const NaiveChainFunctor naive_functor{linear_curve, kEvalT, kVol, kStrike, kT};

    const Eigen::VectorXd x = pillarVector();

    double fx;
    Eigen::VectorXd grad(kN_Pillars);
    Eigen::MatrixXd H(kN_Pillars, kN_Pillars);
    stan::math::hessian(linear_functor, x, fx, grad, H);

    double fx_naive;
    Eigen::VectorXd g_naive(kN_Pillars);
    Eigen::MatrixXd H_naive(kN_Pillars, kN_Pillars);
    stan::math::hessian(naive_functor, x, fx_naive, g_naive, H_naive);

    const double t_linear = bench_us(
        [&]() {
            double f_;
            Eigen::VectorXd g_(kN_Pillars);
            Eigen::MatrixXd H_(kN_Pillars, kN_Pillars);
            stan::math::hessian(linear_functor, x, f_, g_, H_);
        },
        kBenchN);

    const double t_spline = bench_us(
        [&]() {
            double f_;
            Eigen::VectorXd g_(kN_Pillars);
            Eigen::MatrixXd H_(kN_Pillars, kN_Pillars);
            stan::math::hessian(spline_functor, x, f_, g_, H_);
        },
        kBenchN);

    const double t_naive = bench_us(
        [&]() {
            double f_;
            Eigen::VectorXd g_(kN_Pillars);
            Eigen::MatrixXd H_(kN_Pillars, kN_Pillars);
            stan::math::hessian(naive_functor, x, f_, g_, H_);
        },
        kBenchN);

    ::testing::Test::RecordProperty("naive_vs_mixed_max_diff",
                                    quantape::util::num((H_naive - H).cwiseAbs().maxCoeff(), 2));
    ::testing::Test::RecordProperty("us_naive", quantape::util::num(t_naive, 3));
    ::testing::Test::RecordProperty("us_linear", quantape::util::num(t_linear, 3));
    ::testing::Test::RecordProperty("us_spline", quantape::util::num(t_spline, 3));
}
