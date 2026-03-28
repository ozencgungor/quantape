/**
 * @file test_bs_hessian_bench.cpp
 * @brief Black-Scholes Hessian: 4 approaches to second-order Greeks
 *
 * In practice, PV = BS(Fwd(r_i, ...), vol(T,K), ...) and we need:
 *   d²PV/dr_i² = Gamma_BS · (dFwd/dr_i)² + Delta_BS · d²Fwd/dr_i²
 *
 * The Fwd's second derivative comes from interpolation (curve stripping).
 * So we need efficient second-order derivatives of BS AND the ability to
 * chain them through the curve's Hessian.
 *
 * This benchmark compares 4 approaches to computing the 3×3 Hessian
 * of BS w.r.t. (S, σ, r), with K and T fixed.
 *
 * Approaches:
 *   1. Naive:             stan::math::hessian on fully templated BS
 *   2. Reverse-on-Greeks: reverse-mode on each analytical Greek expression
 *   3. Nested analytical: fvar<var> functor with make_callback_var Greeks
 *   4. Full analytical:   all second-order Greeks in double, zero AD
 */

#include <stan/math.hpp>
#include <stan/math/mix.hpp>
#include <Eigen/Dense>

#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>

using stan::math::var;
using stan::math::fvar;

constexpr double K_FIX = 100.0;
constexpr double T_FIX = 1.0;

// ═══════════════════════════════════════════════════════════════════════════
// HELPERS
// ═══════════════════════════════════════════════════════════════════════════

namespace bs {

inline double phi(double x) {
    return std::exp(-0.5 * x * x) / std::sqrt(2.0 * M_PI);
}
inline double Phi(double x) {
    return 0.5 * std::erfc(-x * M_SQRT1_2);
}

struct D1D2 { double d1, d2; };

inline D1D2 d1d2(double S, double sigma, double r) {
    double sqrtT = std::sqrt(T_FIX);
    double d1 = (std::log(S / K_FIX) + (r + 0.5 * sigma * sigma) * T_FIX)
                / (sigma * sqrtT);
    return {d1, d1 - sigma * sqrtT};
}

inline double price(double S, double sigma, double r) {
    auto [d1, d2] = d1d2(S, sigma, r);
    return S * Phi(d1) - K_FIX * std::exp(-r * T_FIX) * Phi(d2);
}

// First-order Greeks
struct G1 { double delta, vega, rho; };

inline G1 greeks1(double S, double sigma, double r) {
    auto [d1, d2] = d1d2(S, sigma, r);
    double nd1 = phi(d1), sqrtT = std::sqrt(T_FIX), disc = std::exp(-r * T_FIX);
    return {
        Phi(d1),
        S * nd1 * sqrtT,
        K_FIX * T_FIX * disc * Phi(d2)
    };
}

// Second-order Greeks (6 unique entries of the 3×3 Hessian)
struct G2 {
    double gamma;       // ∂²C/∂S²
    double vanna;       // ∂²C/∂S∂σ
    double dDelta_dr;   // ∂²C/∂S∂r
    double volga;       // ∂²C/∂σ²
    double dVega_dr;    // ∂²C/∂σ∂r
    double dRho_dr;     // ∂²C/∂r²
};

inline G2 greeks2(double S, double sigma, double r) {
    auto [d1, d2] = d1d2(S, sigma, r);
    double nd1 = phi(d1), nd2 = phi(d2);
    double sqrtT = std::sqrt(T_FIX), disc = std::exp(-r * T_FIX);
    double T = T_FIX;

    return {
        nd1 / (S * sigma * sqrtT),                             // Gamma
        -nd1 * d2 / sigma,                                     // Vanna
        nd1 * sqrtT / sigma,                                   // ∂Delta/∂r
        S * sqrtT * nd1 * d1 * d2 / sigma,                     // Volga
        -S * T * d1 * nd1 / sigma,                              // ∂Vega/∂r
        -K_FIX*T*T*disc*Phi(d2) + K_FIX*T*disc*nd2*sqrtT/sigma // ∂Rho/∂r
    };
}

} // namespace bs

// ═══════════════════════════════════════════════════════════════════════════
// APPROACH 1: NAIVE — stan::math::hessian on fully templated BS
//
// Every operation (log, exp, Phi, erfc internals) goes through fvar<var>.
// Stan tapes ~30+ var nodes per Hessian column × 3 columns = ~90 reverse ops.
// ═══════════════════════════════════════════════════════════════════════════

struct BSNaiveFunctor {
    template <typename T>
    T operator()(const Eigen::Matrix<T, Eigen::Dynamic, 1>& theta) const {
        T S = theta(0), sigma = theta(1), r = theta(2);
        T sqrtT = stan::math::sqrt(T(T_FIX));
        T sigSqrtT = sigma * sqrtT;
        T d1 = (stan::math::log(S / K_FIX)
                + (r + sigma * sigma / 2.0) * T_FIX) / sigSqrtT;
        T d2 = d1 - sigSqrtT;
        return S * stan::math::Phi(d1)
               - K_FIX * stan::math::exp(-r * T_FIX) * stan::math::Phi(d2);
    }
};

// ═══════════════════════════════════════════════════════════════════════════
// APPROACH 2: REVERSE-ON-GREEKS
//
// Idea: the Greeks have simpler expressions than the price itself.
//   Delta = Φ(d₁)         — just d₁ and one Φ call
//   Vega  = S·φ(d₁)·√T   — d₁, one exp, two products
//   Rho   = K·T·e^{-rT}·Φ(d₂) — d₁, d₂, one Φ, one exp
//
// We create a var tape for each Greek and call grad() to get its
// partial derivatives = one Hessian row. Cost: 3 reverse passes,
// each ~10-15 var nodes (versus ~30+ for the full BS through fvar<var>).
//
// No fvar<var> needed. Pure reverse-mode.
// ═══════════════════════════════════════════════════════════════════════════

void hessian_reverse_on_greeks(double S0, double s0, double r0,
                                double& fx, Eigen::Vector3d& g,
                                Eigen::Matrix3d& H) {
    fx = bs::price(S0, s0, r0);
    auto g1 = bs::greeks1(S0, s0, r0);
    g << g1.delta, g1.vega, g1.rho;

    double sqrtT = std::sqrt(T_FIX);

    // Row 0: ∂Delta/∂(S,σ,r) — Gamma, Vanna, ∂Delta/∂r
    {
        var S(S0), sigma(s0), r(r0);
        var d1 = (log(S / K_FIX) + (r + sigma*sigma/2.0)*T_FIX) / (sigma*sqrtT);
        var delta = stan::math::Phi(d1);
        stan::math::grad(delta.vi_);
        H(0,0) = S.adj(); H(0,1) = sigma.adj(); H(0,2) = r.adj();
        stan::math::recover_memory();
    }

    // Row 1: ∂Vega/∂(S,σ,r) — Vanna, Volga, ∂Vega/∂r
    {
        var S(S0), sigma(s0), r(r0);
        var d1 = (log(S / K_FIX) + (r + sigma*sigma/2.0)*T_FIX) / (sigma*sqrtT);
        var phi_d1 = exp(-d1*d1/2.0) / std::sqrt(2.0*M_PI);
        var vega = S * phi_d1 * sqrtT;
        stan::math::grad(vega.vi_);
        H(1,0) = S.adj(); H(1,1) = sigma.adj(); H(1,2) = r.adj();
        stan::math::recover_memory();
    }

    // Row 2: ∂Rho/∂(S,σ,r)
    {
        var S(S0), sigma(s0), r(r0);
        var d1 = (log(S / K_FIX) + (r + sigma*sigma/2.0)*T_FIX) / (sigma*sqrtT);
        var d2 = d1 - sigma * sqrtT;
        var rho_g = K_FIX * T_FIX * exp(-r*T_FIX) * stan::math::Phi(d2);
        stan::math::grad(rho_g.vi_);
        H(2,0) = S.adj(); H(2,1) = sigma.adj(); H(2,2) = r.adj();
        stan::math::recover_memory();
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// APPROACH 3: NESTED ANALYTICAL ADJOINT
//
// The ultimate tape-minimal approach for stan::math::hessian:
//
// When called with fvar<var>, we construct:
//   value   = var(price)              — just a plain var, 0 tape ops for value
//   tangent = δ_var·Sd + ν_var·sd + ρ_var·rd     — 5 arithmetic var nodes
//
// where each Greek_var = make_callback_var(Greek_double, {2nd-order Greeks})
// is a SINGLE var node with a callback that encodes the analytical Hessian row.
//
// Total: 3 callbacks + 5 arithmetic = 8 var nodes per Hessian column.
// Versus ~30+ for approach 1.
//
// When hessian() calls grad(tangent) for column j:
//   adj(tangent) = 1
//   → adj(Greek_j_var) = 1 (only the j-th Greek gets adjoint)
//   → callback fires, pushing 2nd-order Greeks into S.adj(), σ.adj(), r.adj()
//   → hessian reads those adjoints as H[j][:]
// ═══════════════════════════════════════════════════════════════════════════

struct BSNestedFunctor {
    template <typename T>
    T operator()(const Eigen::Matrix<T, Eigen::Dynamic, 1>& theta) const {
        if constexpr (std::is_same_v<T, fvar<var>>) {
            return compute_nested(theta);
        } else {
            // Fallback for other types (var, double)
            T S = theta(0), sigma = theta(1), r = theta(2);
            T sqrtT = stan::math::sqrt(T(T_FIX));
            T d1 = (stan::math::log(S / K_FIX)
                    + (r + sigma*sigma/2.0)*T_FIX) / (sigma*sqrtT);
            T d2 = d1 - sigma*sqrtT;
            return S*stan::math::Phi(d1) - K_FIX*stan::math::exp(-r*T_FIX)*stan::math::Phi(d2);
        }
    }

    fvar<var> compute_nested(
        const Eigen::Matrix<fvar<var>, Eigen::Dynamic, 1>& theta) const {

        // Value-level (var) and tangent-level (var) of each fvar<var> input
        var Sv = theta(0).val_, sv = theta(1).val_, rv = theta(2).val_;
        var Sd = theta(0).d_,   sd = theta(1).d_,   rd = theta(2).d_;

        // Everything in double — no tape cost
        double S0 = Sv.val(), s0 = sv.val(), r0 = rv.val();
        double val = bs::price(S0, s0, r0);
        auto g1 = bs::greeks1(S0, s0, r0);
        auto g2 = bs::greeks2(S0, s0, r0);

        // Price: just a plain var (its callback is never invoked by hessian)
        var price_var(val);

        // Each Greek is a make_callback_var whose callback encodes the
        // corresponding Hessian row (i.e., second-order Greeks).
        var delta_var = stan::math::make_callback_var(
            g1.delta,
            [Sv, sv, rv,
             gamma=g2.gamma, vanna=g2.vanna, dDdr=g2.dDelta_dr](auto& vi) {
                double a = vi.adj();
                Sv.adj() += a * gamma;
                sv.adj() += a * vanna;
                rv.adj() += a * dDdr;
            });

        var vega_var = stan::math::make_callback_var(
            g1.vega,
            [Sv, sv, rv,
             vanna=g2.vanna, volga=g2.volga, dVdr=g2.dVega_dr](auto& vi) {
                double a = vi.adj();
                Sv.adj() += a * vanna;  // ∂Vega/∂S = Vanna (symmetry)
                sv.adj() += a * volga;
                rv.adj() += a * dVdr;
            });

        var rho_var = stan::math::make_callback_var(
            g1.rho,
            [Sv, sv, rv,
             dRdS=g2.dDelta_dr, dRds=g2.dVega_dr, dRdr=g2.dRho_dr](auto& vi) {
                // ∂Rho/∂S = ∂Delta/∂r, ∂Rho/∂σ = ∂Vega/∂r (Schwarz symmetry)
                double a = vi.adj();
                Sv.adj() += a * dRdS;
                sv.adj() += a * dRds;
                rv.adj() += a * dRdr;
            });

        // Tangent = ∇C · tangent_direction = Σ Greek_i · d_i
        var tangent = delta_var * Sd + vega_var * sd + rho_var * rd;

        return fvar<var>(price_var, tangent);
    }
};

// ═══════════════════════════════════════════════════════════════════════════
// APPROACH 4: FULL ANALYTICAL — pure double, zero AD
//
// All 6 unique Hessian entries are known closed-form expressions.
// Cost: one pass through d₁, d₂, φ(d₁), Φ(d₂) plus arithmetic.
// ═══════════════════════════════════════════════════════════════════════════

void hessian_full_analytical(double S0, double s0, double r0,
                              double& fx, Eigen::Vector3d& g,
                              Eigen::Matrix3d& H) {
    fx = bs::price(S0, s0, r0);
    auto g1 = bs::greeks1(S0, s0, r0);
    auto g2 = bs::greeks2(S0, s0, r0);

    g << g1.delta, g1.vega, g1.rho;

    H << g2.gamma,     g2.vanna,    g2.dDelta_dr,
         g2.vanna,     g2.volga,    g2.dVega_dr,
         g2.dDelta_dr, g2.dVega_dr, g2.dRho_dr;
}

// ═══════════════════════════════════════════════════════════════════════════
// IR GAMMA FUNCTOR (for practical chain-rule benchmark)
//
// Simulates rate → Fwd(rate) → BS(Fwd, vol, ...)
// where Fwd(r) = Fwd0 + dFwd_dr·(r-r0) + ½·d²Fwd_dr²·(r-r0)²
// (In practice, Fwd comes from curve stripping / interpolation)
// ═══════════════════════════════════════════════════════════════════════════

struct IRGammaFunctor {
    double Fwd0, s0, r0, dFdr, d2Fdr2;
    template <typename T>
    T operator()(const Eigen::Matrix<T, Eigen::Dynamic, 1>& theta) const {
        T rate = theta(0);
        T dr = rate - r0;
        T fwd = Fwd0 + dFdr * dr + 0.5 * d2Fdr2 * dr * dr;
        T sqrtT = stan::math::sqrt(T(T_FIX));
        T d1 = (stan::math::log(fwd / K_FIX)
                + (rate + T(s0)*T(s0)/2.0)*T_FIX) / (T(s0)*sqrtT);
        T d2 = d1 - T(s0)*sqrtT;
        return fwd * stan::math::Phi(d1)
               - K_FIX * stan::math::exp(-rate*T_FIX) * stan::math::Phi(d2);
    }
};

// ═══════════════════════════════════════════════════════════════════════════
// BENCHMARK HARNESS
// ═══════════════════════════════════════════════════════════════════════════

template <typename F>
double bench_us(F&& fn, int N) {
    for (int i = 0; i < std::min(N / 10, 1000); ++i) fn();
    auto t0 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < N; ++i) fn();
    auto t1 = std::chrono::high_resolution_clock::now();
    return std::chrono::duration<double, std::micro>(t1 - t0).count() / N;
}

int main() {
    std::cout << std::fixed;
    std::cout << "╔═══════════════════════════════════════════════════════════════╗\n";
    std::cout << "║  Black-Scholes Hessian: 4 approaches to second-order Greeks  ║\n";
    std::cout << "╚═══════════════════════════════════════════════════════════════╝\n\n";

    double S0 = 105.0, s0 = 0.20, r0 = 0.05;

    // ── 1. Correctness: compare all 4 Hessians ──
    std::cout << "── Correctness (S=" << S0 << ", σ=" << s0
              << ", r=" << r0 << ", K=" << K_FIX << ", T=" << T_FIX << ") ──\n\n";

    Eigen::VectorXd x(3);
    x << S0, s0, r0;

    // Approach 1: naive hessian
    double fx1; Eigen::VectorXd g1(3); Eigen::MatrixXd H1(3,3);
    stan::math::hessian(BSNaiveFunctor{}, x, fx1, g1, H1);

    // Approach 2: reverse-on-greeks
    double fx2; Eigen::Vector3d g2; Eigen::Matrix3d H2;
    hessian_reverse_on_greeks(S0, s0, r0, fx2, g2, H2);

    // Approach 3: nested analytical
    double fx3; Eigen::VectorXd g3(3); Eigen::MatrixXd H3(3,3);
    stan::math::hessian(BSNestedFunctor{}, x, fx3, g3, H3);

    // Approach 4: full analytical
    double fx4; Eigen::Vector3d g4; Eigen::Matrix3d H4;
    hessian_full_analytical(S0, s0, r0, fx4, g4, H4);

    const char* labels[3] = {"S", "σ", "r"};

    auto print_hessian = [&](const char* name, const Eigen::MatrixXd& H) {
        std::cout << "  " << name << ":\n";
        for (int i = 0; i < 3; ++i) {
            std::cout << "    [";
            for (int j = 0; j < 3; ++j)
                std::cout << std::setw(14) << std::setprecision(8) << H(i,j);
            std::cout << " ]  " << labels[i] << "\n";
        }
        std::cout << "\n";
    };

    print_hessian("Naive (fvar<var>)", H1);
    print_hessian("Reverse-on-Greeks", Eigen::MatrixXd(H2));
    print_hessian("Nested analytical", H3);
    print_hessian("Full analytical",   Eigen::MatrixXd(H4));

    // Max absolute differences vs full analytical
    std::cout << "  Max |H - H_analytical|:\n";
    std::cout << "    Naive:             " << std::scientific << std::setprecision(2)
              << (H1 - Eigen::MatrixXd(H4)).cwiseAbs().maxCoeff() << "\n";
    std::cout << "    Reverse-on-Greeks: "
              << (Eigen::MatrixXd(H2) - Eigen::MatrixXd(H4)).cwiseAbs().maxCoeff() << "\n";
    std::cout << "    Nested analytical: "
              << (H3 - Eigen::MatrixXd(H4)).cwiseAbs().maxCoeff() << "\n";

    // Also verify gradient
    std::cout << std::fixed << "\n  Gradient: [Delta, Vega, Rho]\n";
    std::cout << "    Naive:      [" << std::setprecision(8)
              << g1(0) << ", " << g1(1) << ", " << g1(2) << "]\n";
    std::cout << "    Analytical: [" << g4(0) << ", " << g4(1) << ", " << g4(2) << "]\n";

    // Verify against finite differences
    std::cout << "\n── Finite difference verification ──\n\n";
    double eps = 1e-5;
    Eigen::Matrix3d H_fd;
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            Eigen::Vector3d xpp = x, xpm = x, xmp = x, xmm = x;
            xpp(i) += eps; xpp(j) += eps;
            xpm(i) += eps; xpm(j) -= eps;
            xmp(i) -= eps; xmp(j) += eps;
            xmm(i) -= eps; xmm(j) -= eps;
            auto f = [](const Eigen::Vector3d& v) {
                return bs::price(v(0), v(1), v(2));
            };
            H_fd(i,j) = (f(xpp) - f(xpm) - f(xmp) + f(xmm)) / (4.0 * eps * eps);
        }
    }
    print_hessian("Finite differences", Eigen::MatrixXd(H_fd));
    std::cout << "  Max |H_analytical - H_fd|: " << std::scientific << std::setprecision(2)
              << (Eigen::MatrixXd(H4) - Eigen::MatrixXd(H_fd)).cwiseAbs().maxCoeff() << "\n";

    // ── 2. Named second-order Greeks ──
    std::cout << std::fixed << "\n── Named second-order Greeks ──\n\n";
    auto g2s = bs::greeks2(S0, s0, r0);
    std::cout << "  Gamma (∂²C/∂S²)  = " << std::setprecision(8) << g2s.gamma << "\n";
    std::cout << "  Vanna (∂²C/∂S∂σ) = " << g2s.vanna << "\n";
    std::cout << "  Charm (∂²C/∂S∂r) = " << g2s.dDelta_dr << "\n";
    std::cout << "  Volga (∂²C/∂σ²)  = " << g2s.volga << "\n";
    std::cout << "  Veta  (∂²C/∂σ∂r) = " << g2s.dVega_dr << "\n";
    std::cout << "  ∂²C/∂r²          = " << g2s.dRho_dr << "\n";

    // ── 3. Timing benchmark ──
    constexpr int N = 100'000;
    std::cout << "\n── Performance (" << N << " calls) ──\n\n";

    auto t_naive = bench_us([&]() {
        double fx_; Eigen::VectorXd g_(3); Eigen::MatrixXd H_(3,3);
        stan::math::hessian(BSNaiveFunctor{}, x, fx_, g_, H_);
    }, N);

    auto t_rog = bench_us([&]() {
        double fx_; Eigen::Vector3d g_; Eigen::Matrix3d H_;
        hessian_reverse_on_greeks(S0, s0, r0, fx_, g_, H_);
    }, N);

    auto t_nested = bench_us([&]() {
        double fx_; Eigen::VectorXd g_(3); Eigen::MatrixXd H_(3,3);
        stan::math::hessian(BSNestedFunctor{}, x, fx_, g_, H_);
    }, N);

    auto t_anal = bench_us([&]() {
        volatile double S_ = S0;  // prevent constant folding
        double fx_; Eigen::Vector3d g_; Eigen::Matrix3d H_;
        hessian_full_analytical(S_, s0, r0, fx_, g_, H_);
        volatile double sink = H_(0,0) + H_(1,1) + H_(2,2);
        (void)sink;
    }, N);

    std::cout << "  Approach                       us/call   vs Naive\n";
    std::cout << "  " << std::string(55, '-') << "\n";

    auto row = [&](const char* name, double t) {
        std::cout << "  " << std::setw(30) << std::left << name << std::right
                  << std::setw(10) << std::setprecision(3) << t
                  << std::setw(10) << std::setprecision(2) << t_naive / t << "x\n";
    };

    row("1. Naive (fvar<var>)",    t_naive);
    row("2. Reverse-on-Greeks",    t_rog);
    row("3. Nested analytical",    t_nested);
    row("4. Full analytical (dbl)", t_anal);

    std::cout << "\n  Full analytical is " << std::setprecision(0)
              << t_anal * 1000.0 << " ns/call — pure arithmetic, no AD overhead.\n";

    // ── 4. Practical scenario: IR Gamma through the chain rule ──
    std::cout << "\n── Practical: IR Gamma via chain rule ──\n\n";
    std::cout << "  PV = BS(Fwd(r_i), vol, ...)\n";
    std::cout << "  d²PV/dr_i² = Gamma · (dFwd/dr_i)² + Delta · d²Fwd/dr_i²\n\n";

    // Simulate: Fwd depends on discount factors from curve interpolation
    double Fwd = 105.0, dFwd_dr = 95.0, d2Fwd_dr2 = -90.0;  // typical sensitivities
    auto g1_vals = bs::greeks1(Fwd, s0, r0);
    auto g2_vals = bs::greeks2(Fwd, s0, r0);

    double ir_gamma = g2_vals.gamma * dFwd_dr * dFwd_dr
                    + g1_vals.delta * d2Fwd_dr2;

    std::cout << "  Fwd = " << Fwd << ", dFwd/dr = " << dFwd_dr
              << ", d²Fwd/dr² = " << d2Fwd_dr2 << "\n";
    std::cout << "  Delta   = " << std::setprecision(6) << g1_vals.delta << "\n";
    std::cout << "  Gamma   = " << g2_vals.gamma << "\n";
    std::cout << "  IR Gamma = Gamma·(dFwd/dr)² + Delta·d²Fwd/dr²\n";
    std::cout << "           = " << std::setprecision(4) << g2_vals.gamma << " × " << dFwd_dr*dFwd_dr
              << " + " << g1_vals.delta << " × " << d2Fwd_dr2 << "\n";
    std::cout << "           = " << std::setprecision(6) << ir_gamma << "\n";

    // Time the chain-rule approach vs letting AD do it
    auto t_chain = bench_us([&]() {
        volatile double S_ = Fwd;  // prevent constant folding
        auto g1_ = bs::greeks1(S_, s0, r0);
        auto g2_ = bs::greeks2(S_, s0, r0);
        volatile double ir_g = g2_.gamma * dFwd_dr * dFwd_dr + g1_.delta * d2Fwd_dr2;
        (void)ir_g;
    }, N);

    // AD approach: create var for rate, let Fwd depend on it, compute BS, take second deriv
    IRGammaFunctor ir_functor{Fwd, s0, r0, dFwd_dr, d2Fwd_dr2};
    Eigen::VectorXd xr(1);
    xr(0) = r0;

    auto t_ad_chain = bench_us([&]() {
        double fx_; Eigen::VectorXd g_(1); Eigen::MatrixXd H_(1,1);
        stan::math::hessian(ir_functor, xr, fx_, g_, H_);
        volatile double ir_g = H_(0,0);
        (void)ir_g;
    }, N);

    std::cout << "\n  Chain rule (analytical):  " << std::setprecision(1) << t_chain * 1000.0 << " ns\n";
    std::cout << "  Full AD (fvar<var>):      " << std::setprecision(3) << t_ad_chain << " us ("
              << std::setprecision(0) << t_ad_chain * 1000.0 << " ns)\n";
    std::cout << "  Speedup:                  " << std::setprecision(1) << t_ad_chain / t_chain << "x\n";

    std::cout << "\n═══════════════════════════════════════════════════════════════\n";
    return 0;
}
