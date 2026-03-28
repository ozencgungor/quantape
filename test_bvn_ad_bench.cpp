/**
 * @file test_bvn_ad_bench.cpp
 * @brief Bivariate normal CDF: Genz algorithm + analytical adjoint
 *
 * Φ₂(x, y, ρ) has no closed form — computed via Gauss-Legendre quadrature.
 * Naive AD tapes every quadrature node. But the partial derivatives ARE known:
 *
 *   ∂Φ₂/∂x = φ(x) · Φ((y - ρx) / √(1-ρ²))
 *   ∂Φ₂/∂y = φ(y) · Φ((x - ρy) / √(1-ρ²))
 *   ∂Φ₂/∂ρ = φ₂(x, y, ρ) = bivariate normal PDF
 *
 * This is a textbook case for analytical adjoints.
 */

#include <stan/math.hpp>

#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <random>
#include <vector>

using stan::math::var;

// ═══════════════════════════════════════════════════════════════════════════
// GENZ (2004) BIVARIATE NORMAL CDF — double implementation
//
// Based on Alan Genz's TVPACK algorithm.
// Uses Gauss-Legendre quadrature with 20 points for |ρ| < 0.925,
// and a series expansion approach for |ρ| ≥ 0.925.
// ═══════════════════════════════════════════════════════════════════════════

namespace bvn_detail {

inline double phi(double x) {
    return std::exp(-0.5 * x * x) / std::sqrt(2.0 * M_PI);
}

inline double Phi(double x) {
    return 0.5 * std::erfc(-x * M_SQRT1_2);
}

// Gauss-Legendre 20-point quadrature on [-1, 1]
// Weights and abscissae from standard tables
constexpr int GL_N = 20;
constexpr double GL_x[GL_N] = {
    -0.9931285991850949, -0.9639719272779138, -0.9122344282513259,
    -0.8391169718222188, -0.7463319064601508, -0.6360536807265150,
    -0.5108670019508271, -0.3737060887154196, -0.2277858511416451,
    -0.0765265211334973,
     0.0765265211334973,  0.2277858511416451,  0.3737060887154196,
     0.5108670019508271,  0.6360536807265150,  0.7463319064601508,
     0.8391169718222188,  0.9122344282513259,  0.9639719272779138,
     0.9931285991850949
};
constexpr double GL_w[GL_N] = {
     0.0176140071391521,  0.0406014298003869,  0.0626720483341091,
     0.0832767415767048,  0.1019301198172404,  0.1181945319615184,
     0.1316886384491766,  0.1420961093183821,  0.1491729864726037,
     0.1527533871307258,
     0.1527533871307258,  0.1491729864726037,  0.1420961093183821,
     0.1316886384491766,  0.1181945319615184,  0.1019301198172404,
     0.0832767415767048,  0.0626720483341091,  0.0406014298003869,
     0.0176140071391521
};

// Core Genz algorithm for P(X > dh, Y > dk | corr = r)
inline double bvnu(double dh, double dk, double r) {
    if (std::abs(r) < 1e-15) {
        return Phi(-dh) * Phi(-dk);
    }

    double tp = 2.0 * M_PI;

    if (std::abs(r) < 0.925) {
        // ── Low/moderate correlation: direct GL quadrature ──
        // ∫₀^{asr} f(sin θ) dθ = (asr/2) ∫₋₁¹ f(sin(asr(1+t)/2)) dt
        // Our 20-point GL rule covers [-1,1] fully, so one pass suffices.
        double hs = (dh * dh + dk * dk) / 2.0;
        double asr = std::asin(r);
        double sum = 0.0;

        for (int i = 0; i < GL_N; ++i) {
            double sn = std::sin(asr * (1.0 + GL_x[i]) / 2.0);
            sum += GL_w[i] * std::exp((sn * dh * dk - hs) / (1.0 - sn * sn));
        }
        return Phi(-dh) * Phi(-dk) + sum * asr / (2.0 * tp);
    }

    // ── High correlation: decomposition approach ──
    double h, k, hk;
    if (r < 0) {
        dk = -dk;
        hk = -1.0;
    } else {
        hk = 1.0;
    }
    h = dh; k = dk;

    double as_val = (1.0 - r) * (1.0 + r);
    double a = std::sqrt(as_val);
    double bs = (h - k) * (h - k);
    double c = (4.0 - hk * h * k) / 8.0;
    double d = (12.0 - hk * h * k) / 16.0;
    double asr = -(bs / as_val + hk * h * k) / 2.0;

    double bvn_val;
    if (asr > -100.0) {
        bvn_val = a * std::exp(asr)
                  * (1.0 - c * (bs - as_val) * (1.0 - d * bs / 5.0) / 3.0
                     + c * d * as_val * as_val / 5.0);
    } else {
        bvn_val = 0.0;
    }

    if (-hk * h * k < 100.0) {
        double b = std::sqrt(bs);
        bvn_val -= std::exp(-hk * h * k / 2.0) * std::sqrt(tp) * Phi(-b / a) * b
                   * (1.0 - c * bs * (1.0 - d * bs / 5.0) / 3.0);
    }

    a = a / 2.0;
    for (int i = 0; i < GL_N; ++i) {
        for (int j = -1; j <= 1; j += 2) {
            double xs = (a * (j * GL_x[i] + 1.0));
            xs = xs * xs;
            double rs = std::sqrt(1.0 - xs);
            asr = -(bs / xs + hk * h * k) / 2.0;
            if (asr > -100.0) {
                bvn_val += a * GL_w[i] * std::exp(asr)
                           * (std::exp(-hk * h * k * (1.0 - rs) / (2.0 * (1.0 + rs))) / rs
                              - (1.0 + c * xs * (1.0 + d * xs)));
            }
        }
    }

    bvn_val = -bvn_val / tp;

    if (r > 0) {
        bvn_val += Phi(-std::max(h, k));
    } else {
        bvn_val = -bvn_val;
        if (k > h) {
            bvn_val += Phi(k) - Phi(h);
        }
    }

    return std::max(0.0, std::min(1.0, bvn_val));
}

} // namespace bvn_detail

// ═══════════════════════════════════════════════════════════════════════════
// PUBLIC INTERFACE
// ═══════════════════════════════════════════════════════════════════════════

// Φ₂(x, y, ρ) = P(X ≤ x, Y ≤ y) with corr(X,Y) = ρ
inline double bivariateNormalCdf(double x, double y, double rho) {
    return bvn_detail::bvnu(-x, -y, rho);
}

// Bivariate normal PDF
inline double bivariateNormalPdf(double x, double y, double rho) {
    double onemrho2 = 1.0 - rho * rho;
    double z = (x * x - 2.0 * rho * x * y + y * y) / onemrho2;
    return std::exp(-0.5 * z) / (2.0 * M_PI * std::sqrt(onemrho2));
}

// ═══════════════════════════════════════════════════════════════════════════
// NAIVE TEMPLATED VERSION (for Stan AD)
//
// This re-implements the Genz algorithm with DoubleT so Stan can tape it.
// Every sin, exp, sqrt in the quadrature goes on the tape.
// ═══════════════════════════════════════════════════════════════════════════

template <typename DoubleT>
DoubleT bivariateNormalCdfNaive(DoubleT x, DoubleT y, DoubleT rho) {
    using std::asin;
    using std::exp;
    using std::sin;
    using std::sqrt;
    using stan::math::Phi;

    DoubleT dh = -x, dk = -y, r = rho;

    double r_val;
    if constexpr (std::is_same_v<DoubleT, double>)
        r_val = r;
    else
        r_val = r.val();

    // Only shortcut for double — for var, we need the tape to capture ∂/∂ρ
    if constexpr (std::is_same_v<DoubleT, double>) {
        if (std::abs(r_val) < 1e-15) {
            return Phi(x) * Phi(y);
        }
    }

    DoubleT tp = 2.0 * M_PI;

    if (std::abs(r_val) < 0.925) {
        // ── Low/moderate correlation: single-pass GL quadrature ──
        DoubleT hs = (dh * dh + dk * dk) / 2.0;
        DoubleT asr = asin(r);
        DoubleT sum = DoubleT(0.0);

        for (int i = 0; i < bvn_detail::GL_N; ++i) {
            DoubleT sn = sin(asr * (1.0 + bvn_detail::GL_x[i]) / 2.0);
            sum += bvn_detail::GL_w[i] * exp((sn * dh * dk - hs) / (DoubleT(1.0) - sn * sn));
        }
        return Phi(x) * Phi(y) + sum * asr / (DoubleT(2.0) * tp);
    }

    // ── High correlation: decomposition approach (mirrors bvnu exactly) ──
    DoubleT hk_val;
    if (r_val < 0) {
        dk = -dk;
        hk_val = DoubleT(-1.0);
    } else {
        hk_val = DoubleT(1.0);
    }
    DoubleT h = dh, k = dk;

    DoubleT as_val = (DoubleT(1.0) - r) * (DoubleT(1.0) + r);
    DoubleT a = sqrt(as_val);
    DoubleT bs = (h - k) * (h - k);
    DoubleT c = (DoubleT(4.0) - hk_val * h * k) / 8.0;
    DoubleT d = (DoubleT(12.0) - hk_val * h * k) / 16.0;
    DoubleT asr = -(bs / as_val + hk_val * h * k) / 2.0;

    double asr_val;
    if constexpr (std::is_same_v<DoubleT, double>)
        asr_val = asr;
    else
        asr_val = asr.val();

    DoubleT bvn_val = DoubleT(0.0);
    if (asr_val > -100.0) {
        bvn_val = a * exp(asr)
                  * (DoubleT(1.0) - c * (bs - as_val) * (DoubleT(1.0) - d * bs / 5.0) / 3.0
                     + c * d * as_val * as_val / 5.0);
    }

    double hk_hk_val;
    if constexpr (std::is_same_v<DoubleT, double>)
        hk_hk_val = -hk_val * h * k;
    else
        hk_hk_val = (-hk_val * h * k).val();

    if (hk_hk_val < 100.0) {
        DoubleT b = sqrt(bs);
        bvn_val = bvn_val - exp(-hk_val * h * k / 2.0) * sqrt(tp) * Phi(-b / a) * b
                   * (DoubleT(1.0) - c * bs * (DoubleT(1.0) - d * bs / 5.0) / 3.0);
    }

    a = a / 2.0;
    for (int i = 0; i < bvn_detail::GL_N; ++i) {
        for (int j = -1; j <= 1; j += 2) {
            DoubleT xs = a * (DoubleT(j) * bvn_detail::GL_x[i] + 1.0);
            xs = xs * xs;
            DoubleT rs = sqrt(DoubleT(1.0) - xs);
            DoubleT asr_inner = -(bs / xs + hk_val * h * k) / 2.0;

            double asr_inner_val;
            if constexpr (std::is_same_v<DoubleT, double>)
                asr_inner_val = asr_inner;
            else
                asr_inner_val = asr_inner.val();

            if (asr_inner_val > -100.0) {
                bvn_val = bvn_val + a * bvn_detail::GL_w[i] * exp(asr_inner)
                           * (exp(-hk_val * h * k * (DoubleT(1.0) - rs) / (DoubleT(2.0) * (DoubleT(1.0) + rs))) / rs
                              - (DoubleT(1.0) + c * xs * (DoubleT(1.0) + d * xs)));
            }
        }
    }

    bvn_val = -bvn_val / tp;

    if (r_val > 0) {
        // Phi(-max(h,k)) — need to branch on double values
        double h_val, k_val;
        if constexpr (std::is_same_v<DoubleT, double>) {
            h_val = h; k_val = k;
        } else {
            h_val = h.val(); k_val = k.val();
        }
        if (h_val >= k_val)
            bvn_val = bvn_val + Phi(-h);
        else
            bvn_val = bvn_val + Phi(-k);
    } else {
        bvn_val = -bvn_val;
        double k_val_cmp, h_val_cmp;
        if constexpr (std::is_same_v<DoubleT, double>) {
            k_val_cmp = k; h_val_cmp = h;
        } else {
            k_val_cmp = k.val(); h_val_cmp = h.val();
        }
        if (k_val_cmp > h_val_cmp) {
            bvn_val = bvn_val + Phi(k) - Phi(h);
        }
    }

    // Clamp — use value-based check to avoid taping the clamp when not needed
    double bvn_check;
    if constexpr (std::is_same_v<DoubleT, double>)
        bvn_check = bvn_val;
    else
        bvn_check = bvn_val.val();

    if (bvn_check < 0.0) return DoubleT(0.0);
    if (bvn_check > 1.0) return DoubleT(1.0);
    return bvn_val;
}

// ═══════════════════════════════════════════════════════════════════════════
// ANALYTICAL ADJOINT VERSION
//
// Forward:  compute Φ₂(x, y, ρ) in double via Genz
// Adjoint:  use the known closed-form partial derivatives
//
//   ∂Φ₂/∂x = φ(x) · Φ((y - ρx) / √(1-ρ²))
//   ∂Φ₂/∂y = φ(y) · Φ((x - ρy) / √(1-ρ²))
//   ∂Φ₂/∂ρ = φ₂(x, y, ρ)   [bivariate normal PDF]
//
// Total: 1 tape node. The derivatives cost ~3 exp + 2 erfc calls.
// ═══════════════════════════════════════════════════════════════════════════

inline var bivariateNormalCdfAnalytical(const var& x_v, const var& y_v,
                                        const var& rho_v) {
    double x   = x_v.val();
    double y   = y_v.val();
    double rho = rho_v.val();

    // Forward: full Genz algorithm in double
    double val = bivariateNormalCdf(x, y, rho);

    // Analytical partial derivatives
    double onemrho2 = 1.0 - rho * rho;
    double sqrt_onemrho2 = std::sqrt(onemrho2);

    // ∂Φ₂/∂x = φ(x) · Φ((y - ρx) / √(1-ρ²))
    double dval_dx = bvn_detail::phi(x)
                   * bvn_detail::Phi((y - rho * x) / sqrt_onemrho2);

    // ∂Φ₂/∂y = φ(y) · Φ((x - ρy) / √(1-ρ²))
    double dval_dy = bvn_detail::phi(y)
                   * bvn_detail::Phi((x - rho * y) / sqrt_onemrho2);

    // ∂Φ₂/∂ρ = φ₂(x, y, ρ) = bivariate normal PDF
    double dval_drho = bivariateNormalPdf(x, y, rho);

    return stan::math::make_callback_var(
        val,
        [x_v, y_v, rho_v, dval_dx, dval_dy, dval_drho](auto& vi) {
            double adj = vi.adj();
            x_v.adj()   += adj * dval_dx;
            y_v.adj()   += adj * dval_dy;
            rho_v.adj() += adj * dval_drho;
        });
}

// ═══════════════════════════════════════════════════════════════════════════
// BENCHMARK
// ═══════════════════════════════════════════════════════════════════════════

struct BenchResult { double per_call_us; };

template <typename F>
BenchResult bench(F&& fn, int N) {
    for (int i = 0; i < std::min(N / 10, 1000); ++i) fn();
    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < N; ++i) fn();
    auto end = std::chrono::high_resolution_clock::now();
    double us = std::chrono::duration<double, std::micro>(end - start).count();
    return {us / N};
}

inline std::size_t countVariNodes(auto&& fn) {
    stan::math::recover_memory();
    auto* stack = stan::math::ChainableStack::instance_;
    std::size_t before = stack->var_stack_.size();
    fn();
    std::size_t after = stack->var_stack_.size();
    stan::math::recover_memory();
    return after - before;
}

inline std::size_t measureArenaBytes(auto&& fn, int N) {
    stan::math::recover_memory();
    auto* stack = stan::math::ChainableStack::instance_;
    std::size_t before = stack->memalloc_.bytes_allocated();
    for (int i = 0; i < N; ++i) fn();
    std::size_t after = stack->memalloc_.bytes_allocated();
    stan::math::recover_memory();
    return after - before;
}

int main() {
    std::cout << std::fixed;

    std::cout << "╔════════════════════════════════════════════════════════════════╗\n";
    std::cout << "║  Bivariate Normal CDF: Naive Autodiff vs Analytical Adjoint   ║\n";
    std::cout << "╚════════════════════════════════════════════════════════════════╝\n\n";

    // ── 1. Correctness against known values ──
    std::cout << "── Correctness: Φ₂ values and derivatives ──\n\n";

    struct TestCase { double x, y, rho; const char* label; };
    std::vector<TestCase> cases = {
        { 0.0,  0.0,  0.0,  "independent"     },
        { 0.0,  0.0,  0.5,  "moderate ρ"      },
        { 1.0, -1.0,  0.3,  "asymmetric"      },
        {-0.5,  0.5,  0.7,  "high ρ"          },
        { 1.5,  1.5,  0.9,  "very high ρ"     },
        { 0.0,  0.0, -0.5,  "negative ρ"      },
        {-2.0, -2.0,  0.5,  "deep tail"       },
        { 2.0,  2.0,  0.8,  "upper tail"      },
    };

    std::cout << "  " << std::setw(16) << "Case"
              << std::setw(12) << "Φ₂ value"
              << std::setw(12) << "∂/∂x"
              << std::setw(12) << "∂/∂y"
              << std::setw(12) << "∂/∂ρ" << "\n";
    std::cout << "  " << std::string(64, '-') << "\n";

    // Also verify naive vs analytical
    double max_val_diff = 0, max_dx_diff = 0, max_dy_diff = 0, max_drho_diff = 0;

    for (const auto& tc : cases) {
        // Naive
        var xn(tc.x), yn(tc.y), rn(tc.rho);
        var res_n = bivariateNormalCdfNaive<var>(xn, yn, rn);
        stan::math::grad(res_n.vi_);
        double val_n = res_n.val(), dx_n = xn.adj(), dy_n = yn.adj(), drho_n = rn.adj();
        stan::math::recover_memory();

        // Analytical
        var xa(tc.x), ya(tc.y), ra(tc.rho);
        var res_a = bivariateNormalCdfAnalytical(xa, ya, ra);
        stan::math::grad(res_a.vi_);
        double val_a = res_a.val(), dx_a = xa.adj(), dy_a = ya.adj(), drho_a = ra.adj();
        stan::math::recover_memory();

        max_val_diff  = std::max(max_val_diff,  std::abs(val_n - val_a));
        max_dx_diff   = std::max(max_dx_diff,   std::abs(dx_n - dx_a));
        max_dy_diff   = std::max(max_dy_diff,   std::abs(dy_n - dy_a));
        max_drho_diff = std::max(max_drho_diff,  std::abs(drho_n - drho_a));

        std::cout << "  " << std::setw(16) << tc.label
                  << std::setw(12) << std::setprecision(8) << val_a
                  << std::setw(12) << std::setprecision(8) << dx_a
                  << std::setw(12) << std::setprecision(8) << dy_a
                  << std::setw(12) << std::setprecision(8) << drho_a << "\n";
    }

    std::cout << "\n  Max |naive - analytical| differences:\n";
    std::cout << "    Value: " << std::scientific << std::setprecision(2) << max_val_diff << "\n";
    std::cout << "    ∂/∂x:  " << max_dx_diff << "\n";
    std::cout << "    ∂/∂y:  " << max_dy_diff << "\n";
    std::cout << "    ∂/∂ρ:  " << max_drho_diff << "\n";

    // ── 2. Derivative verification via finite differences ──
    std::cout << std::fixed << "\n── Finite difference verification (x=0.5, y=-0.3, ρ=0.4) ──\n\n";
    {
        double x0 = 0.5, y0 = -0.3, rho0 = 0.4;
        double eps = 1e-7;

        double f0 = bivariateNormalCdf(x0, y0, rho0);
        double fd_dx   = (bivariateNormalCdf(x0+eps, y0, rho0) - f0) / eps;
        double fd_dy   = (bivariateNormalCdf(x0, y0+eps, rho0) - f0) / eps;
        double fd_drho = (bivariateNormalCdf(x0, y0, rho0+eps) - f0) / eps;

        // Analytical
        var xa(x0), ya(y0), ra(rho0);
        var res = bivariateNormalCdfAnalytical(xa, ya, ra);
        stan::math::grad(res.vi_);

        std::cout << "  " << std::setw(10) << "" << std::setw(16) << "Fin. Diff."
                  << std::setw(16) << "Analytical" << std::setw(14) << "Abs Diff" << "\n";
        std::cout << "  " << std::string(56, '-') << "\n";

        auto fdrow = [](const char* name, double fd, double an) {
            std::cout << "  " << std::setw(10) << name
                      << std::setw(16) << std::setprecision(10) << fd
                      << std::setw(16) << an
                      << std::setw(14) << std::scientific << std::setprecision(2)
                      << std::abs(fd - an) << std::fixed << "\n";
        };
        fdrow("∂/∂x", fd_dx, xa.adj());
        fdrow("∂/∂y", fd_dy, ya.adj());
        fdrow("∂/∂ρ", fd_drho, ra.adj());
        stan::math::recover_memory();
    }

    // ── 3. Memory measurement ──
    std::cout << "\n── Memory: tape nodes and arena usage ──\n\n";

    double x_test = 0.5, y_test = -0.3, rho_test = 0.4;

    auto naive_no_recover = [&]() {
        var x(x_test), y(y_test), rho(rho_test);
        var res = bivariateNormalCdfNaive<var>(x, y, rho);
        (void)res;
    };
    auto analytical_no_recover = [&]() {
        var x(x_test), y(y_test), rho(rho_test);
        var res = bivariateNormalCdfAnalytical(x, y, rho);
        (void)res;
    };

    auto nodes_naive = countVariNodes(naive_no_recover);
    auto nodes_anal  = countVariNodes(analytical_no_recover);

    constexpr int MEM_BATCH = 500;
    auto bytes_naive = measureArenaBytes(naive_no_recover, MEM_BATCH);
    auto bytes_anal  = measureArenaBytes(analytical_no_recover, MEM_BATCH);

    std::cout << "  " << std::setw(28) << "" << std::setw(12) << "Naive" << std::setw(12) << "Analytical" << std::setw(8) << "Ratio" << "\n";
    std::cout << "  " << std::string(60, '-') << "\n";
    std::cout << "  " << std::setw(28) << "Tape (vari) nodes/call"
              << std::setw(12) << nodes_naive
              << std::setw(12) << nodes_anal
              << std::setw(8) << std::setprecision(0) << (double)nodes_naive / std::max(nodes_anal, (std::size_t)1) << "x\n";
    std::cout << "  " << std::setw(28) << ("Arena bytes/" + std::to_string(MEM_BATCH) + " calls")
              << std::setw(12) << bytes_naive
              << std::setw(12) << bytes_anal
              << std::setw(8) << std::setprecision(1) << (double)bytes_naive / std::max(bytes_anal, (std::size_t)1) << "x\n";

    // ── 4. Timing benchmark ──
    constexpr int N = 100'000;

    std::cout << "\n── Performance (" << N << " calls) ──\n\n";

    // Double baseline
    auto t_dbl = bench([&]() {
        volatile double r = bivariateNormalCdf(x_test, y_test, rho_test);
        (void)r;
    }, N);

    // Naive AD
    auto t_naive = bench([&]() {
        var x(x_test), y(y_test), rho(rho_test);
        var res = bivariateNormalCdfNaive<var>(x, y, rho);
        stan::math::grad(res.vi_);
        stan::math::recover_memory();
    }, N);

    // Analytical AD
    auto t_anal = bench([&]() {
        var x(x_test), y(y_test), rho(rho_test);
        var res = bivariateNormalCdfAnalytical(x, y, rho);
        stan::math::grad(res.vi_);
        stan::math::recover_memory();
    }, N);

    std::cout << "  double (no AD):    " << std::setprecision(3) << t_dbl.per_call_us << " us/call  (baseline)\n";
    std::cout << "  Naive autodiff:    " << std::setprecision(3) << t_naive.per_call_us << " us/call  ("
              << std::setprecision(1) << t_naive.per_call_us / t_dbl.per_call_us << "x vs double)\n";
    std::cout << "  Analytical adj:    " << std::setprecision(3) << t_anal.per_call_us << " us/call  ("
              << std::setprecision(1) << t_anal.per_call_us / t_dbl.per_call_us << "x vs double)\n";
    std::cout << "  Analytical/Naive:  " << std::setprecision(2) << t_naive.per_call_us / t_anal.per_call_us << "x speedup\n";

    // ── 5. Sweep over correlation range ──
    std::cout << "\n── Timing across ρ range ──\n\n";
    std::cout << "  " << std::setw(8) << "ρ"
              << std::setw(12) << "double us"
              << std::setw(12) << "Naive us"
              << std::setw(12) << "Anal. us"
              << std::setw(10) << "Naive/dbl"
              << std::setw(10) << "Anal/dbl"
              << std::setw(10) << "Speedup" << "\n";
    std::cout << "  " << std::string(74, '-') << "\n";

    for (double rho : {-0.9, -0.5, 0.0, 0.3, 0.5, 0.7, 0.9, 0.95}) {
        auto td = bench([&]() {
            volatile double r = bivariateNormalCdf(0.5, -0.3, rho);
            (void)r;
        }, N / 2);

        auto tn = bench([&]() {
            var x(0.5), y(-0.3), r(rho);
            var res = bivariateNormalCdfNaive<var>(x, y, r);
            stan::math::grad(res.vi_);
            stan::math::recover_memory();
        }, N / 2);

        auto ta = bench([&]() {
            var x(0.5), y(-0.3), r(rho);
            var res = bivariateNormalCdfAnalytical(x, y, r);
            stan::math::grad(res.vi_);
            stan::math::recover_memory();
        }, N / 2);

        std::cout << "  " << std::setw(8) << std::setprecision(2) << rho
                  << std::setw(12) << std::setprecision(3) << td.per_call_us
                  << std::setw(12) << tn.per_call_us
                  << std::setw(12) << ta.per_call_us
                  << std::setw(10) << std::setprecision(1) << tn.per_call_us / td.per_call_us
                  << std::setw(10) << ta.per_call_us / td.per_call_us
                  << std::setw(10) << std::setprecision(2) << tn.per_call_us / ta.per_call_us
                  << "\n";
    }

    // ── 6. Application: Gaussian copula CDO pricing ──
    std::cout << "\n── Application: Gaussian copula (100 names, sweep over ρ) ──\n";
    std::cout << "  (Each name requires Φ₂ for joint default probability)\n\n";

    constexpr int N_NAMES = 100;
    constexpr int COPULA_ITERS = 10'000;

    std::mt19937 rng(42);
    std::uniform_real_distribution<> barrier_dist(-2.0, 1.0);
    std::vector<double> barriers(N_NAMES);
    for (auto& b : barriers) b = barrier_dist(rng);

    double rho_copula = 0.3;

    auto copula_double = [&]() {
        double sum = 0;
        for (int i = 0; i < N_NAMES; ++i)
            for (int j = i + 1; j < std::min(i + 5, N_NAMES); ++j)  // 4 nearest neighbors
                sum += bivariateNormalCdf(barriers[i], barriers[j], rho_copula);
        return sum;
    };

    auto copula_naive = [&]() {
        var rho(rho_copula);
        var sum = 0.0;
        for (int i = 0; i < N_NAMES; ++i)
            for (int j = i + 1; j < std::min(i + 5, N_NAMES); ++j) {
                var bi(barriers[i]), bj(barriers[j]);
                sum += bivariateNormalCdfNaive<var>(bi, bj, rho);
            }
        stan::math::grad(sum.vi_);
        stan::math::recover_memory();
    };

    auto copula_analytical = [&]() {
        var rho(rho_copula);
        var sum = 0.0;
        for (int i = 0; i < N_NAMES; ++i)
            for (int j = i + 1; j < std::min(i + 5, N_NAMES); ++j) {
                var bi(barriers[i]), bj(barriers[j]);
                sum += bivariateNormalCdfAnalytical(bi, bj, rho);
            }
        stan::math::grad(sum.vi_);
        stan::math::recover_memory();
    };

    auto ct_dbl  = bench(copula_double, COPULA_ITERS);
    auto ct_naive = bench(copula_naive, COPULA_ITERS);
    auto ct_anal  = bench(copula_analytical, COPULA_ITERS);

    std::cout << "  double (no AD):    " << std::setprecision(1) << ct_dbl.per_call_us << " us  (baseline)\n";
    std::cout << "  Naive autodiff:    " << ct_naive.per_call_us << " us  ("
              << std::setprecision(1) << ct_naive.per_call_us / ct_dbl.per_call_us << "x vs double)\n";
    std::cout << "  Analytical adj:    " << ct_anal.per_call_us << " us  ("
              << std::setprecision(1) << ct_anal.per_call_us / ct_dbl.per_call_us << "x vs double)\n";
    std::cout << "  Analytical/Naive:  " << std::setprecision(2) << ct_naive.per_call_us / ct_anal.per_call_us << "x speedup\n";

    std::cout << "\n════════════════════════════════════════════════════════════════\n";
    return 0;
}
