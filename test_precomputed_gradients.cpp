/**
 * @file test_precomputed_gradients.cpp
 * @brief Mixed analytical/AD Hessian using precomputed_gradients (Stan Math 2.x+)
 *        and custom vari subclass (any Stan Math version)
 *
 * Two approaches compared:
 *   A) precomputed_gradients — available since Stan Math 2.x
 *   B) Custom vari subclass   — works on ANY Stan Math version
 *
 * Both achieve the same thing as make_callback_var (Stan 4.0+):
 * create a var node with known partial derivatives.
 */

#include <stan/math/fwd/core.hpp>
#include <stan/math/mix/functor/hessian.hpp>
#include <stan/math/prim/fun/Eigen.hpp>
#include <stan/math/rev/core.hpp>
#include <stan/math/rev/fun/Phi.hpp>
#include <stan/math/rev/fun/exp.hpp>
#include <stan/math/rev/fun/log.hpp>

#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <vector>

using stan::math::fvar;
using stan::math::var;

// ═══════════════════════════════════════════════════════════════════════════
// BS helpers (plain double)
// ═══════════════════════════════════════════════════════════════════════════

namespace bs {

inline double phi(double x) {
    return std::exp(-0.5 * x * x) / std::sqrt(2.0 * M_PI);
}
inline double Phi(double x) {
    return 0.5 * std::erfc(-x * M_SQRT1_2);
}

struct Greeks {
    double val, delta, vega, rho;
    double gamma, vanna, volga;
    double dDelta_dr, dVega_dr, dRho_dr;
};

Greeks compute(double S, double sigma, double r, double K, double T) {
    double sqrtT = std::sqrt(T);
    double d1 = (std::log(S / K) + (r + 0.5 * sigma * sigma) * T) / (sigma * sqrtT);
    double d2 = d1 - sigma * sqrtT;
    double nd1 = phi(d1), Nd1 = Phi(d1), Nd2 = Phi(d2);
    double disc = std::exp(-r * T);

    Greeks g{};
    g.val = S * Nd1 - K * disc * Nd2;
    g.delta = Nd1;
    g.vega = S * nd1 * sqrtT;
    g.rho = K * T * disc * Nd2;

    g.gamma = nd1 / (S * sigma * sqrtT);
    g.vanna = -nd1 * d2 / sigma;
    g.volga = S * sqrtT * nd1 * d1 * d2 / sigma;
    g.dDelta_dr = nd1 * sqrtT / sigma;
    g.dVega_dr = -S * T * d1 * nd1 / sigma;
    g.dRho_dr = -K * T * T * disc * Nd2 + K * T * disc * phi(d2) * sqrtT / sigma;
    return g;
}

} // namespace bs

// ═══════════════════════════════════════════════════════════════════════════
// APPROACH A: precomputed_gradients (Stan Math 2.x+)
//
// precomputed_gradients(value, vector<var> operands, vector<double> partials)
// Creates a var whose adjoint propagation is:
//   operand[i].adj() += result.adj() * partials[i]
// ═══════════════════════════════════════════════════════════════════════════

namespace approach_a {

// ── double overload: just compute ──
double price(double S, double sigma, double r, double K, double T) {
    return bs::compute(S, sigma, r, K, T).val;
}

// ── var overload: 1st order analytical ──
var price(var S, var sigma, var r, double K, double T) {
    auto g = bs::compute(S.val(), sigma.val(), r.val(), K, T);

    std::vector<var> operands = {S, sigma, r};
    std::vector<double> partials = {g.delta, g.vega, g.rho};
    return stan::math::precomputed_gradients(g.val, operands, partials);
}

// ── fvar<var> overload: Greeks as var for Hessian ──
// Each Greek is itself a precomputed_gradients node with 2nd-order partials.
// When hessian() calls grad() on the tangent, reverse-mode flows through
// these Greek-var nodes, yielding the Hessian.
fvar<var> price(fvar<var> S, fvar<var> sigma, fvar<var> r, double K, double T) {
    var Sv = S.val_, sv = sigma.val_, rv = r.val_;
    auto g = bs::compute(Sv.val(), sv.val(), rv.val(), K, T);

    var price_var(g.val);

    std::vector<var> ops = {Sv, sv, rv};

    // Delta as a var node: ∂delta/∂S = gamma, ∂delta/∂σ = vanna, ∂delta/∂r = dDelta_dr
    var delta_var = stan::math::precomputed_gradients(
        g.delta, ops, std::vector<double>{g.gamma, g.vanna, g.dDelta_dr});

    // Vega as a var node
    var vega_var = stan::math::precomputed_gradients(
        g.vega, ops, std::vector<double>{g.vanna, g.volga, g.dVega_dr});

    // Rho as a var node (symmetric cross-Greeks)
    var rho_var = stan::math::precomputed_gradients(
        g.rho, ops, std::vector<double>{g.dDelta_dr, g.dVega_dr, g.dRho_dr});

    // Tangent = Σ Greek_i * d_i
    var tangent = delta_var * S.d_ + vega_var * sigma.d_ + rho_var * r.d_;
    return fvar<var>(price_var, tangent);
}

} // namespace approach_a

// ═══════════════════════════════════════════════════════════════════════════
// APPROACH B: Custom vari subclass (works on ANY Stan Math version)
//
// This is what precomputed_gradients does internally — you just
// write the chain() method yourself.
//
// Stan Math 4.x: inherit from vari_value<double>
// Stan Math 2.x/3.x: inherit from vari
// ═══════════════════════════════════════════════════════════════════════════

namespace approach_b {

// In Stan Math 4.x, vari is vari_value<double>
// In Stan Math 2.x, vari is just vari
// Either way, stan::math::vari works as the base class name via typedef.

struct bs_vari : public stan::math::vari_value<double> {
    stan::math::vari_value<double>* S_vi_;
    stan::math::vari_value<double>* sigma_vi_;
    stan::math::vari_value<double>* r_vi_;
    double delta_, vega_, rho_;

    bs_vari(double val, var S, var sigma, var r, double delta, double vega, double rho)
        : vari_value<double>(val), S_vi_(S.vi_), sigma_vi_(sigma.vi_), r_vi_(r.vi_), delta_(delta),
          vega_(vega), rho_(rho) {}

    void chain() override {
        S_vi_->adj_ += adj_ * delta_;
        sigma_vi_->adj_ += adj_ * vega_;
        r_vi_->adj_ += adj_ * rho_;
    }
};

struct greek_vari : public stan::math::vari_value<double> {
    stan::math::vari_value<double>* S_vi_;
    stan::math::vari_value<double>* sigma_vi_;
    stan::math::vari_value<double>* r_vi_;
    double dS_, dsigma_, dr_;

    greek_vari(double val, var S, var sigma, var r, double dS, double dsigma, double dr)
        : vari_value<double>(val), S_vi_(S.vi_), sigma_vi_(sigma.vi_), r_vi_(r.vi_), dS_(dS),
          dsigma_(dsigma), dr_(dr) {}

    void chain() override {
        S_vi_->adj_ += adj_ * dS_;
        sigma_vi_->adj_ += adj_ * dsigma_;
        r_vi_->adj_ += adj_ * dr_;
    }
};

// ── double overload ──
double price(double S, double sigma, double r, double K, double T) {
    return bs::compute(S, sigma, r, K, T).val;
}

// ── var overload ──
var price(var S, var sigma, var r, double K, double T) {
    auto g = bs::compute(S.val(), sigma.val(), r.val(), K, T);
    return var(new bs_vari(g.val, S, sigma, r, g.delta, g.vega, g.rho));
}

// ── fvar<var> overload ──
fvar<var> price(fvar<var> S, fvar<var> sigma, fvar<var> r, double K, double T) {
    var Sv = S.val_, sv = sigma.val_, rv = r.val_;
    auto g = bs::compute(Sv.val(), sv.val(), rv.val(), K, T);

    var price_var(g.val);

    var delta_var(new greek_vari(g.delta, Sv, sv, rv, g.gamma, g.vanna, g.dDelta_dr));
    var vega_var(new greek_vari(g.vega, Sv, sv, rv, g.vanna, g.volga, g.dVega_dr));
    var rho_var(new greek_vari(g.rho, Sv, sv, rv, g.dDelta_dr, g.dVega_dr, g.dRho_dr));

    var tangent = delta_var * S.d_ + vega_var * sigma.d_ + rho_var * r.d_;
    return fvar<var>(price_var, tangent);
}

} // namespace approach_b

// ═══════════════════════════════════════════════════════════════════════════
// APPROACH C: Analytical 1st-order Greeks as var, AD for 2nd-order
//
// Write delta, vega, rho as var expressions using Stan-overloaded math.
// Call grad() on each Greek to get one row of the Hessian.
// No fvar<var>, no hessian(), no hand-coded 2nd-order Greeks.
// ═══════════════════════════════════════════════════════════════════════════

namespace approach_c {

void compute(double S0, double sig0, double r0, double K, double T, double& price,
             Eigen::VectorXd& grad, Eigen::MatrixXd& H) {
    double sqrtT = std::sqrt(T);

    // Price (plain double)
    auto g = bs::compute(S0, sig0, r0, K, T);
    price = g.val;

    // Row 0: differentiate delta = Phi(d1) w.r.t. (S, sigma, r)
    {
        stan::math::nested_rev_autodiff nested;
        var S(S0), sigma(sig0), r(r0);
        var d1 = (log(S / K) + (r + 0.5 * sigma * sigma) * T) / (sigma * sqrtT);
        var delta = stan::math::Phi(d1);
        grad(0) = delta.val();
        stan::math::grad(delta.vi_);
        H(0, 0) = S.adj();     // gamma
        H(0, 1) = sigma.adj(); // vanna
        H(0, 2) = r.adj();     // dDelta/dr
    }

    // Row 1: differentiate vega = S * phi(d1) * sqrt(T)
    {
        stan::math::nested_rev_autodiff nested;
        var S(S0), sigma(sig0), r(r0);
        var d1 = (log(S / K) + (r + 0.5 * sigma * sigma) * T) / (sigma * sqrtT);
        var nd1 = exp(-0.5 * d1 * d1) / std::sqrt(2.0 * M_PI);
        var vega = S * nd1 * sqrtT;
        grad(1) = vega.val();
        stan::math::grad(vega.vi_);
        H(1, 0) = S.adj();     // vanna (= H(0,1) by symmetry)
        H(1, 1) = sigma.adj(); // volga
        H(1, 2) = r.adj();     // dVega/dr
    }

    // Row 2: differentiate rho = K * T * exp(-r*T) * Phi(d2)
    {
        stan::math::nested_rev_autodiff nested;
        var S(S0), sigma(sig0), r(r0);
        var d1 = (log(S / K) + (r + 0.5 * sigma * sigma) * T) / (sigma * sqrtT);
        var d2 = d1 - sigma * sqrtT;
        var Nd2 = stan::math::Phi(d2);
        var disc = exp(-r * T);
        var rho = K * T * disc * Nd2;
        grad(2) = rho.val();
        stan::math::grad(rho.vi_);
        H(2, 0) = S.adj();     // dDelta/dr (= H(0,2))
        H(2, 1) = sigma.adj(); // dVega/dr  (= H(1,2))
        H(2, 2) = r.adj();     // dRho/dr
    }
}

} // namespace approach_c

// ═══════════════════════════════════════════════════════════════════════════
// TEST HARNESS
// ═══════════════════════════════════════════════════════════════════════════

// Functors for stan::math::hessian (must be callable with Eigen vector)
struct FunctorA {
    double K, T;
    template <typename Vec>
    auto operator()(const Vec& v) const {
        return approach_a::price(v(0), v(1), v(2), K, T);
    }
};

struct FunctorB {
    double K, T;
    template <typename Vec>
    auto operator()(const Vec& v) const {
        return approach_b::price(v(0), v(1), v(2), K, T);
    }
};

void print_result(const char* label, double fx, const Eigen::VectorXd& grad,
                  const Eigen::MatrixXd& H) {
    std::cout << "  " << label << ":\n";
    std::cout << "    PV    = " << std::setprecision(6) << fx << "\n";
    std::cout << "    Grad  = [" << std::setprecision(4) << grad(0) << ", " << grad(1) << ", "
              << grad(2) << "]\n";
    std::cout << "    Hessian:\n";
    for (int i = 0; i < 3; ++i) {
        std::cout << "      [";
        for (int j = 0; j < 3; ++j)
            std::cout << std::setw(12) << std::setprecision(4) << H(i, j);
        std::cout << " ]\n";
    }
    std::cout << "\n";
}

template <typename F>
double bench_us(F&& fn, int N) {
    for (int i = 0; i < 100; ++i)
        fn();
    auto t0 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < N; ++i)
        fn();
    auto t1 = std::chrono::high_resolution_clock::now();
    return std::chrono::duration<double, std::micro>(t1 - t0).count() / N;
}

int main() {
    std::cout << std::fixed;
    std::cout << "\n═══════════════════════════════════════════════════════════\n";
    std::cout << " precomputed_gradients vs custom vari vs naive AD\n";
    std::cout << "═══════════════════════════════════════════════════════════\n\n";

    Eigen::VectorXd x(3);
    x << 100.0, 0.25, 0.05; // S, sigma, r
    double K = 100.0, T = 1.0;

    double fx;
    Eigen::VectorXd grad(3);
    Eigen::MatrixXd H(3, 3);

    // A) precomputed_gradients
    FunctorA fa{K, T};
    stan::math::hessian(fa, x, fx, grad, H);
    print_result("A) precomputed_gradients", fx, grad, H);

    // B) Custom vari
    FunctorB fb{K, T};
    stan::math::hessian(fb, x, fx, grad, H);
    print_result("B) Custom vari", fx, grad, H);

    // Verify A and B give identical results
    double fx_b;
    Eigen::VectorXd grad_b(3);
    Eigen::MatrixXd H_b(3, 3);
    stan::math::hessian(fb, x, fx_b, grad_b, H_b);

    std::cout << "  Max |H_A - H_B|: " << std::scientific << std::setprecision(2)
              << (H - H_b).cwiseAbs().maxCoeff() << std::fixed << "\n\n";

    // C) Analytical Greeks + AD for 2nd order
    double fx_c;
    Eigen::VectorXd grad_c(3);
    Eigen::MatrixXd H_c(3, 3);
    approach_c::compute(x(0), x(1), x(2), K, T, fx_c, grad_c, H_c);
    print_result("C) Analytical 1st + AD 2nd", fx_c, grad_c, H_c);

    // Compare C vs A
    stan::math::hessian(fa, x, fx, grad, H);
    std::cout << "  Max |H_A - H_C|: " << std::scientific << std::setprecision(2)
              << (H - H_c).cwiseAbs().maxCoeff() << std::fixed << "\n\n";

    // Benchmark
    std::cout << "── Performance (50,000 Hessian evals) ──\n\n";
    constexpr int N = 50'000;

    double t_a = bench_us(
        [&]() {
            double f_;
            Eigen::VectorXd g_(3);
            Eigen::MatrixXd H_(3, 3);
            stan::math::hessian(fa, x, f_, g_, H_);
        },
        N);
    double t_b = bench_us(
        [&]() {
            double f_;
            Eigen::VectorXd g_(3);
            Eigen::MatrixXd H_(3, 3);
            stan::math::hessian(fb, x, f_, g_, H_);
        },
        N);

    auto row = [&](const char* name, double t) {
        std::cout << "  " << std::setw(30) << std::left << name << std::right
                  << std::setprecision(3) << std::setw(8) << t << " us\n";
    };
    double t_c = bench_us(
        [&]() {
            double f_;
            Eigen::VectorXd g_(3);
            Eigen::MatrixXd H_(3, 3);
            approach_c::compute(x(0), x(1), x(2), K, T, f_, g_, H_);
        },
        N);

    row("A) precomputed_gradients", t_a);
    row("B) Custom vari", t_b);
    row("C) Analytical 1st + AD 2nd", t_c);

    std::cout << "\n";
    return 0;
}
