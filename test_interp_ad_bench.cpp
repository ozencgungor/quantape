/**
 * @file test_interp_ad_bench.cpp
 * @brief Benchmark: naive templated interpolation AD vs analytical adjoints
 *
 * For each interpolation type we compare:
 *   Naive:      existing Math:: classes instantiated with stan::math::var
 *   Analytical: hand-coded make_callback_var with precomputed ∂output/∂y_data
 *
 * Interpolation types tested:
 *   1. Linear 1D           (2 active weights)
 *   2. Bilinear 2D         (4 active weights)
 *   3. Cubic Spline 1D     (adjoint tridiagonal solve — all y-values contribute)
 */

#include "Math/Interpolations.h"

#include <stan/math.hpp>

#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <random>
#include <vector>

using stan::math::var;
using namespace Math;

// ═══════════════════════════════════════════════════════════════════════════
// HELPERS
// ═══════════════════════════════════════════════════════════════════════════

// Extract double from var or double
inline double dval(const var& v) {
    return v.val();
}
inline double dval(double v) {
    return v;
}

// Binary search matching the library's locate()
inline std::size_t locate(const std::vector<double>& x, double xVal) {
    if (xVal <= x.front())
        return 0;
    if (xVal >= x.back())
        return x.size() - 2;
    std::size_t left = 0, right = x.size() - 1;
    while (right - left > 1) {
        std::size_t mid = left + (right - left) / 2;
        if (x[mid] <= xVal)
            left = mid;
        else
            right = mid;
    }
    return left;
}

// ═══════════════════════════════════════════════════════════════════════════
// 1. LINEAR 1D — ANALYTICAL ADJOINT
//
//    val = (1-t)*y[i] + t*y[i+1],   t = (x - x[i]) / (x[i+1] - x[i])
//    ∂val/∂y[i]   = 1-t
//    ∂val/∂y[i+1] = t
//    ∂val/∂x      = (y[i+1] - y[i]) / (x[i+1] - x[i])
// ═══════════════════════════════════════════════════════════════════════════

inline var linearInterpAnalytical(const var& x_v, const std::vector<double>& x_grid,
                                  const std::vector<var>& y_data) {
    double x = x_v.val();
    std::size_t i = locate(x_grid, x);

    double x1 = x_grid[i], x2 = x_grid[i + 1];
    double dx = x2 - x1;
    double t = (x - x1) / dx;

    double y1 = y_data[i].val(), y2 = y_data[i + 1].val();
    double val = (1.0 - t) * y1 + t * y2;

    double w0 = 1.0 - t;
    double w1 = t;
    double dval_dx = (y2 - y1) / dx;

    return stan::math::make_callback_var(val, [x_v, &y_data, i, w0, w1, dval_dx](auto& vi) {
        double adj = vi.adj();
        y_data[i].adj() += adj * w0;
        y_data[i + 1].adj() += adj * w1;
        x_v.adj() += adj * dval_dx;
    });
}

// ═══════════════════════════════════════════════════════════════════════════
// 2. BILINEAR 2D — ANALYTICAL ADJOINT
//
//    val = (1-u)(1-v)*z00 + u(1-v)*z10 + (1-u)v*z01 + uv*z11
//    where u = (x-x[i])/(x[i+1]-x[i]),  v = (y-y[j])/(y[j+1]-y[j])
// ═══════════════════════════════════════════════════════════════════════════

inline std::size_t locateLinear(const std::vector<double>& grid, double val) {
    for (std::size_t i = 0; i < grid.size() - 1; ++i) {
        if (val >= grid[i] && val <= grid[i + 1])
            return i;
    }
    return grid.size() - 2;
}

inline var bilinearInterpAnalytical(const var& x_v, const var& y_v,
                                    const std::vector<double>& x_grid,
                                    const std::vector<double>& y_grid,
                                    const std::vector<std::vector<var>>& z_data) {
    double x = x_v.val(), y = y_v.val();
    std::size_t i = locateLinear(x_grid, x);
    std::size_t j = locateLinear(y_grid, y);

    double x1 = x_grid[i], x2 = x_grid[i + 1];
    double y1 = y_grid[j], y2 = y_grid[j + 1];
    double dx = x2 - x1, dy = y2 - y1;
    double u = (x - x1) / dx, v = (y - y1) / dy;

    // z_data is z[row=y][col=x], matching the library convention
    double z00 = z_data[j][i].val();
    double z10 = z_data[j][i + 1].val();
    double z01 = z_data[j + 1][i].val();
    double z11 = z_data[j + 1][i + 1].val();

    double w00 = (1.0 - u) * (1.0 - v);
    double w10 = u * (1.0 - v);
    double w01 = (1.0 - u) * v;
    double w11 = u * v;

    double val = w00 * z00 + w10 * z10 + w01 * z01 + w11 * z11;

    double dval_dx = ((1.0 - v) * (z10 - z00) + v * (z11 - z01)) / dx;
    double dval_dy = ((1.0 - u) * (z01 - z00) + u * (z11 - z10)) / dy;

    return stan::math::make_callback_var(
        val, [x_v, y_v, &z_data, i, j, w00, w10, w01, w11, dval_dx, dval_dy](auto& vi) {
            double adj = vi.adj();
            z_data[j][i].adj() += adj * w00;
            z_data[j][i + 1].adj() += adj * w10;
            z_data[j + 1][i].adj() += adj * w01;
            z_data[j + 1][i + 1].adj() += adj * w11;
            x_v.adj() += adj * dval_dx;
            y_v.adj() += adj * dval_dy;
        });
}

// ═══════════════════════════════════════════════════════════════════════════
// 3. CUBIC SPLINE 1D — ANALYTICAL ADJOINT
//
//    The output depends on ALL y-values through the tridiagonal solve.
//    Strategy:
//      Forward:  compute coefficients using double arithmetic
//      Adjoint:  propagate through polynomial eval → coefficient computation
//                → tridiagonal solve (adjoint = transposed solve)
// ═══════════════════════════════════════════════════════════════════════════

inline var cubicSplineInterpAnalytical(const var& x_v, const std::vector<double>& x_grid,
                                       const std::vector<var>& y_data) {
    std::size_t n = x_grid.size();
    double x = x_v.val();

    // ── Forward pass: all in double ──

    // Extract y values
    std::vector<double> y(n);
    for (std::size_t k = 0; k < n; ++k)
        y[k] = y_data[k].val();

    // Segment lengths and slopes
    std::vector<double> dx(n - 1), S(n - 1);
    for (std::size_t k = 0; k < n - 1; ++k) {
        dx[k] = x_grid[k + 1] - x_grid[k];
        S[k] = (y[k + 1] - y[k]) / dx[k];
    }

    // Build tridiagonal system for natural cubic spline
    // A · deriv = rhs,  where rhs is linear in S (thus linear in y)
    std::vector<double> lower(n - 1), diag(n), upper(n - 1), rhs(n);

    for (std::size_t k = 1; k < n - 1; ++k) {
        lower[k - 1] = dx[k - 1];
        diag[k] = 2.0 * (dx[k - 1] + dx[k]);
        upper[k] = dx[k];
        rhs[k] = 3.0 * (dx[k] * S[k - 1] + dx[k - 1] * S[k]);
    }
    diag[0] = 2.0;
    upper[0] = 1.0;
    rhs[0] = 3.0 * S[0];
    lower[n - 2] = 1.0;
    diag[n - 1] = 2.0;
    rhs[n - 1] = 3.0 * S[n - 2];

    // Thomas algorithm forward sweep
    std::vector<double> c_prime(n - 1), d_prime(n);
    c_prime[0] = upper[0] / diag[0];
    d_prime[0] = rhs[0] / diag[0];
    for (std::size_t k = 1; k < n - 1; ++k) {
        double denom = diag[k] - lower[k - 1] * c_prime[k - 1];
        c_prime[k] = upper[k] / denom;
        d_prime[k] = (rhs[k] - lower[k - 1] * d_prime[k - 1]) / denom;
    }
    {
        double denom = diag[n - 1] - lower[n - 2] * c_prime[n - 2];
        d_prime[n - 1] = (rhs[n - 1] - lower[n - 2] * d_prime[n - 2]) / denom;
    }

    // Back substitution → derivatives at knots
    std::vector<double> deriv(n);
    deriv[n - 1] = d_prime[n - 1];
    for (int k = (int)n - 2; k >= 0; --k) {
        deriv[k] = d_prime[k] - c_prime[k] * deriv[k + 1];
    }

    // Cubic coefficients
    std::vector<double> a_coeff(n - 1), b_coeff(n - 1), c_coeff(n - 1);
    for (std::size_t k = 0; k < n - 1; ++k) {
        a_coeff[k] = deriv[k];
        b_coeff[k] = (3.0 * S[k] - deriv[k + 1] - 2.0 * deriv[k]) / dx[k];
        c_coeff[k] = (deriv[k + 1] + deriv[k] - 2.0 * S[k]) / (dx[k] * dx[k]);
    }

    // Evaluate
    std::size_t seg = locate(x_grid, x);
    if (seg >= n - 1)
        seg = n - 2;
    double t = x - x_grid[seg];
    double val = y[seg] + t * (a_coeff[seg] + t * (b_coeff[seg] + t * c_coeff[seg]));

    // ── Adjoint pass: compute ∂val/∂y[k] for all k ──
    // We propagate backwards through:
    //   (1) polynomial evaluation → ∂val/∂{y[seg], deriv[seg], deriv[seg+1], S[seg]}
    //   (2) coefficient computation → ∂val/∂deriv[k] (accumulated)
    //   (3) tridiagonal solve → ∂val/∂rhs[k]
    //   (4) rhs construction → ∂val/∂S[k] → ∂val/∂y[k]

    // (1) Partials of val w.r.t. polynomial ingredients
    //   val = y[seg] + a*t + b*t² + c*t³
    //   a = deriv[seg]
    //   b = (3S - deriv[seg+1] - 2*deriv[seg]) / dx[seg]
    //   c = (deriv[seg+1] + deriv[seg] - 2S) / dx[seg]²
    double dval_dy_seg = 1.0; // direct y[seg] term
    double dval_da = t;
    double dval_db = t * t;
    double dval_dc = t * t * t;

    double da_dderiv_seg = 1.0;
    double db_dderiv_seg = -2.0 / dx[seg];
    double db_dderiv_seg1 = -1.0 / dx[seg];
    double db_dS_seg = 3.0 / dx[seg];
    double dc_dderiv_seg = 1.0 / (dx[seg] * dx[seg]);
    double dc_dderiv_seg1 = 1.0 / (dx[seg] * dx[seg]);
    double dc_dS_seg = -2.0 / (dx[seg] * dx[seg]);

    // ∂val/∂deriv[seg], ∂val/∂deriv[seg+1], ∂val/∂S[seg]
    double dval_dderiv_seg =
        dval_da * da_dderiv_seg + dval_db * db_dderiv_seg + dval_dc * dc_dderiv_seg;
    double dval_dderiv_seg1 = dval_db * db_dderiv_seg1 + dval_dc * dc_dderiv_seg1;
    double dval_dS_seg = dval_db * db_dS_seg + dval_dc * dc_dS_seg;

    // (2) Adjoint of tridiagonal solve: A·deriv = rhs
    //     If ∂val/∂deriv = g, then ∂val/∂rhs = A⁻ᵀ g
    //     i.e., solve Aᵀ λ = g  →  ∂val/∂rhs = λ
    std::vector<double> g(n, 0.0);
    g[seg] = dval_dderiv_seg;
    g[seg + 1] = dval_dderiv_seg1;

    // Solve Aᵀ λ = g using Thomas on the transposed system
    // Aᵀ has: lower → upper of A, upper → lower of A, diag same
    // So Aᵀ: sub = upper, diag = diag, super = lower
    std::vector<double> t_lower(n - 1), t_upper(n - 1);
    for (std::size_t k = 0; k < n - 1; ++k) {
        t_lower[k] = upper[k]; // Aᵀ sub-diagonal = A's super-diagonal
        t_upper[k] = lower[k]; // Aᵀ super-diagonal = A's sub-diagonal
    }

    // Thomas on transposed system
    std::vector<double> tc_prime(n - 1), td_prime(n);
    tc_prime[0] = t_upper[0] / diag[0];
    td_prime[0] = g[0] / diag[0];
    for (std::size_t k = 1; k < n - 1; ++k) {
        double denom = diag[k] - t_lower[k - 1] * tc_prime[k - 1];
        tc_prime[k] = t_upper[k] / denom;
        td_prime[k] = (g[k] - t_lower[k - 1] * td_prime[k - 1]) / denom;
    }
    {
        double denom = diag[n - 1] - t_lower[n - 2] * tc_prime[n - 2];
        td_prime[n - 1] = (g[n - 1] - t_lower[n - 2] * td_prime[n - 2]) / denom;
    }

    std::vector<double> lambda(n);
    lambda[n - 1] = td_prime[n - 1];
    for (int k = (int)n - 2; k >= 0; --k) {
        lambda[k] = td_prime[k] - tc_prime[k] * lambda[k + 1];
    }
    // Now lambda[k] = ∂val/∂rhs[k]

    // (3) rhs → S → y  chain
    //   rhs[0]   = 3 S[0]                                    → depends on y[0], y[1]
    //   rhs[k]   = 3(dx[k]*S[k-1] + dx[k-1]*S[k])           → depends on y[k-1], y[k], y[k+1]
    //   rhs[n-1] = 3 S[n-2]                                  → depends on y[n-2], y[n-1]
    //
    //   S[k] = (y[k+1] - y[k]) / dx[k]
    //   ∂S[k]/∂y[k]   = -1/dx[k]
    //   ∂S[k]/∂y[k+1] = +1/dx[k]

    // Accumulate ∂val/∂S[k] from both the direct polynomial term and the rhs terms
    std::vector<double> dval_dS(n - 1, 0.0);

    // Direct polynomial contribution (only segment seg)
    dval_dS[seg] += dval_dS_seg;

    // Contribution via rhs: ∂val/∂rhs[k] · ∂rhs[k]/∂S[j]
    // rhs[0] = 3*S[0] → ∂rhs[0]/∂S[0] = 3
    dval_dS[0] += lambda[0] * 3.0;

    // Interior: rhs[k] = 3*(dx[k]*S[k-1] + dx[k-1]*S[k])
    for (std::size_t k = 1; k < n - 1; ++k) {
        dval_dS[k - 1] += lambda[k] * 3.0 * dx[k];
        dval_dS[k] += lambda[k] * 3.0 * dx[k - 1];
    }

    // rhs[n-1] = 3*S[n-2] → ∂rhs[n-1]/∂S[n-2] = 3
    dval_dS[n - 2] += lambda[n - 1] * 3.0;

    // Finally: S[k] → y[k], y[k+1]
    std::vector<double> dval_dy(n, 0.0);
    dval_dy[seg] += dval_dy_seg; // direct y[seg] in polynomial

    for (std::size_t k = 0; k < n - 1; ++k) {
        dval_dy[k] += dval_dS[k] * (-1.0 / dx[k]);
        dval_dy[k + 1] += dval_dS[k] * (1.0 / dx[k]);
    }

    // ∂val/∂x
    double dval_dx = a_coeff[seg] + t * (2.0 * b_coeff[seg] + 3.0 * c_coeff[seg] * t);

    return stan::math::make_callback_var(val, [x_v, &y_data, dval_dy, dval_dx, n](auto& vi) {
        double adj = vi.adj();
        for (std::size_t k = 0; k < n; ++k) {
            y_data[k].adj() += adj * dval_dy[k];
        }
        x_v.adj() += adj * dval_dx;
    });
}

// ═══════════════════════════════════════════════════════════════════════════
// MEMORY MEASUREMENT
// ═══════════════════════════════════════════════════════════════════════════

// countVariNodes: fn must NOT call recover_memory
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
    for (int i = 0; i < N; ++i)
        fn();
    std::size_t after = stack->memalloc_.bytes_allocated();
    stan::math::recover_memory();
    return after - before;
}

// ═══════════════════════════════════════════════════════════════════════════
// BENCHMARK FRAMEWORK
// ═══════════════════════════════════════════════════════════════════════════

struct BenchResult {
    double total_us;
    double per_call_us;
};

template <typename F>
BenchResult benchmark(F&& fn, int N) {
    // warm-up
    for (int i = 0; i < std::min(N / 10, 1000); ++i)
        fn();

    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < N; ++i)
        fn();
    auto end = std::chrono::high_resolution_clock::now();

    double us = std::chrono::duration<double, std::micro>(end - start).count();
    return {us, us / N};
}

void printSection(const char* title) {
    std::cout << "\n" << std::string(65, '=') << "\n";
    std::cout << "  " << title << "\n";
    std::cout << std::string(65, '=') << "\n\n";
}

void printMemory(const char* label, std::size_t nodes_naive, std::size_t nodes_analytical,
                 std::size_t bytes_naive, std::size_t bytes_analytical, int batch) {
    std::cout << "  Memory (" << label << "):\n";
    std::cout << "    " << std::setw(26) << "" << std::setw(12) << "Naive" << std::setw(12)
              << "Analytical" << std::setw(8) << "Ratio" << "\n";
    std::cout << "    " << std::string(58, '-') << "\n";
    std::cout << "    " << std::setw(26) << "Tape nodes/call" << std::setw(12) << nodes_naive
              << std::setw(12) << nodes_analytical << std::setw(8) << std::setprecision(1)
              << (double)nodes_naive / std::max(nodes_analytical, (std::size_t)1) << "x\n";
    std::cout << "    " << std::setw(26) << ("Arena bytes/" + std::to_string(batch) + " calls")
              << std::setw(12) << bytes_naive << std::setw(12) << bytes_analytical << std::setw(8)
              << std::setprecision(1)
              << (double)bytes_naive / std::max(bytes_analytical, (std::size_t)1) << "x\n\n";
}

void printTiming(BenchResult dbl, BenchResult naive, BenchResult analytical) {
    std::cout << "  Timing:\n";
    std::cout << "    double (no AD):  " << std::setprecision(3) << dbl.per_call_us
              << " us/call  (baseline)\n";
    std::cout << "    Naive autodiff:  " << std::setprecision(3) << naive.per_call_us
              << " us/call  (" << std::setprecision(1) << naive.per_call_us / dbl.per_call_us
              << "x vs double)\n";
    std::cout << "    Analytical adj:  " << std::setprecision(3) << analytical.per_call_us
              << " us/call  (" << std::setprecision(1) << analytical.per_call_us / dbl.per_call_us
              << "x vs double)\n";
    std::cout << "    Analytical/Naive speedup: " << std::setprecision(2)
              << naive.per_call_us / analytical.per_call_us << "x\n\n";
}

// ═══════════════════════════════════════════════════════════════════════════
// 1. LINEAR INTERPOLATION BENCHMARK
// ═══════════════════════════════════════════════════════════════════════════

void benchLinear() {
    printSection("1D Linear Interpolation (20 knots)");

    constexpr int NKNOTS = 20;
    constexpr int N = 100'000;
    constexpr int MEM_BATCH = 1000;

    // Build grid: sin(x) on [0, 2pi]
    std::vector<double> x_grid(NKNOTS);
    std::vector<double> y_vals(NKNOTS);
    for (int i = 0; i < NKNOTS; ++i) {
        x_grid[i] = i * 2.0 * M_PI / (NKNOTS - 1);
        y_vals[i] = std::sin(x_grid[i]);
    }

    double x_eval = 2.5; // evaluation point

    // ── Naive: library with var ──
    auto naive_once = [&]() {
        std::vector<var> y_var(NKNOTS);
        for (int i = 0; i < NKNOTS; ++i)
            y_var[i] = y_vals[i];
        var x_v = x_eval;
        LinearInterpolation<var> interp(std::vector<var>(x_grid.begin(), x_grid.end()), y_var);
        var result = interp(x_v);
        stan::math::grad(result.vi_);
        stan::math::recover_memory();
    };

    // ── Analytical adjoint ──
    auto analytical_once = [&]() {
        std::vector<var> y_var(NKNOTS);
        for (int i = 0; i < NKNOTS; ++i)
            y_var[i] = y_vals[i];
        var x_v = x_eval;
        var result = linearInterpAnalytical(x_v, x_grid, y_var);
        stan::math::grad(result.vi_);
        stan::math::recover_memory();
    };

    // ── Correctness ──
    {
        std::vector<var> y_naive(NKNOTS), y_anal(NKNOTS);
        for (int i = 0; i < NKNOTS; ++i) {
            y_naive[i] = y_vals[i];
            y_anal[i] = y_vals[i];
        }

        var x_n = x_eval;
        LinearInterpolation<var> interp(std::vector<var>(x_grid.begin(), x_grid.end()), y_naive);
        var res_n = interp(x_n);
        stan::math::grad(res_n.vi_);

        std::vector<double> adj_naive(NKNOTS);
        for (int i = 0; i < NKNOTS; ++i)
            adj_naive[i] = y_naive[i].adj();
        double dx_naive = x_n.adj();
        double val_naive = res_n.val();
        stan::math::recover_memory();

        var x_a = x_eval;
        for (int i = 0; i < NKNOTS; ++i)
            y_anal[i] = y_vals[i];
        var res_a = linearInterpAnalytical(x_a, x_grid, y_anal);
        stan::math::grad(res_a.vi_);

        std::cout << "  Correctness:\n";
        std::cout << "    Value:  naive=" << std::setprecision(10) << val_naive
                  << "  analytical=" << res_a.val()
                  << "  diff=" << std::abs(val_naive - res_a.val()) << "\n";
        std::cout << "    dx/dx:  naive=" << dx_naive << "  analytical=" << x_a.adj()
                  << "  diff=" << std::abs(dx_naive - x_a.adj()) << "\n";

        double max_adj_diff = 0;
        for (int i = 0; i < NKNOTS; ++i)
            max_adj_diff = std::max(max_adj_diff, std::abs(adj_naive[i] - y_anal[i].adj()));
        std::cout << "    Max |dy_adj diff|: " << max_adj_diff << "\n\n";
        stan::math::recover_memory();
    }

    // ── Memory ──
    auto naive_no_recover = [&]() {
        std::vector<var> y_var(NKNOTS);
        for (int i = 0; i < NKNOTS; ++i)
            y_var[i] = y_vals[i];
        var x_v = x_eval;
        LinearInterpolation<var> interp(std::vector<var>(x_grid.begin(), x_grid.end()), y_var);
        var result = interp(x_v);
        (void)result;
    };
    auto analytical_no_recover = [&]() {
        std::vector<var> y_var(NKNOTS);
        for (int i = 0; i < NKNOTS; ++i)
            y_var[i] = y_vals[i];
        var x_v = x_eval;
        var result = linearInterpAnalytical(x_v, x_grid, y_var);
        (void)result;
    };
    auto nodes_naive = countVariNodes(naive_no_recover);
    auto nodes_anal = countVariNodes(analytical_no_recover);
    auto bytes_naive = measureArenaBytes(naive_no_recover, MEM_BATCH);
    auto bytes_anal = measureArenaBytes(analytical_no_recover, MEM_BATCH);

    printMemory("linear 1D", nodes_naive, nodes_anal, bytes_naive, bytes_anal, MEM_BATCH);

    // ── Timing ──
    auto double_once = [&]() {
        LinearInterpolation<double> interp(x_grid, y_vals);
        volatile double result = interp(x_eval);
        (void)result;
    };
    auto t_dbl = benchmark(double_once, N);
    auto t_naive = benchmark(naive_once, N);
    auto t_anal = benchmark(analytical_once, N);
    printTiming(t_dbl, t_naive, t_anal);
}

// ═══════════════════════════════════════════════════════════════════════════
// 2. BILINEAR INTERPOLATION BENCHMARK
// ═══════════════════════════════════════════════════════════════════════════

void benchBilinear() {
    printSection("2D Bilinear Interpolation (10x10 grid)");

    constexpr int NX = 10, NY = 10;
    constexpr int N = 100'000;
    constexpr int MEM_BATCH = 1000;

    // Grid: f(x,y) = sin(x)*cos(y) on [0,pi]x[0,pi]
    std::vector<double> x_grid(NX), y_grid(NY);
    std::vector<std::vector<double>> z_vals(NY, std::vector<double>(NX));
    for (int i = 0; i < NX; ++i)
        x_grid[i] = i * M_PI / (NX - 1);
    for (int j = 0; j < NY; ++j)
        y_grid[j] = j * M_PI / (NY - 1);
    for (int j = 0; j < NY; ++j)
        for (int i = 0; i < NX; ++i)
            z_vals[j][i] = std::sin(x_grid[i]) * std::cos(y_grid[j]);

    double x_eval = 1.2, y_eval = 0.8;

    // Helper to create z var grid
    auto make_z_var = [&]() {
        std::vector<std::vector<var>> z(NY, std::vector<var>(NX));
        for (int j = 0; j < NY; ++j)
            for (int i = 0; i < NX; ++i)
                z[j][i] = z_vals[j][i];
        return z;
    };

    // ── Naive ──
    auto naive_once = [&]() {
        auto z_var = make_z_var();
        var x_v = x_eval, y_v = y_eval;
        BilinearInterpolation<var> interp(std::vector<var>(x_grid.begin(), x_grid.end()),
                                          std::vector<var>(y_grid.begin(), y_grid.end()), z_var);
        var result = interp(x_v, y_v);
        stan::math::grad(result.vi_);
        stan::math::recover_memory();
    };

    // ── Analytical ──
    auto analytical_once = [&]() {
        auto z_var = make_z_var();
        var x_v = x_eval, y_v = y_eval;
        var result = bilinearInterpAnalytical(x_v, y_v, x_grid, y_grid, z_var);
        stan::math::grad(result.vi_);
        stan::math::recover_memory();
    };

    // ── Correctness ──
    {
        auto z_n = make_z_var();
        var x_n = x_eval, y_n = y_eval;
        BilinearInterpolation<var> interp(std::vector<var>(x_grid.begin(), x_grid.end()),
                                          std::vector<var>(y_grid.begin(), y_grid.end()), z_n);
        var res_n = interp(x_n, y_n);
        stan::math::grad(res_n.vi_);

        double val_naive = res_n.val();
        std::vector<std::vector<double>> adj_naive(NY, std::vector<double>(NX));
        for (int j = 0; j < NY; ++j)
            for (int i = 0; i < NX; ++i)
                adj_naive[j][i] = z_n[j][i].adj();
        stan::math::recover_memory();

        auto z_a = make_z_var();
        var x_a = x_eval, y_a = y_eval;
        var res_a = bilinearInterpAnalytical(x_a, y_a, x_grid, y_grid, z_a);
        stan::math::grad(res_a.vi_);

        double max_diff = 0;
        for (int j = 0; j < NY; ++j)
            for (int i = 0; i < NX; ++i)
                max_diff = std::max(max_diff, std::abs(adj_naive[j][i] - z_a[j][i].adj()));

        std::cout << "  Correctness:\n";
        std::cout << "    Value diff: " << std::setprecision(10)
                  << std::abs(val_naive - res_a.val()) << "\n";
        std::cout << "    Max |dz_adj diff|: " << max_diff << "\n\n";
        stan::math::recover_memory();
    }

    // ── Memory ──
    auto naive_no_recover = [&]() {
        auto z_var = make_z_var();
        var x_v = x_eval, y_v = y_eval;
        BilinearInterpolation<var> interp(std::vector<var>(x_grid.begin(), x_grid.end()),
                                          std::vector<var>(y_grid.begin(), y_grid.end()), z_var);
        var result = interp(x_v, y_v);
        (void)result;
    };
    auto analytical_no_recover = [&]() {
        auto z_var = make_z_var();
        var x_v = x_eval, y_v = y_eval;
        var result = bilinearInterpAnalytical(x_v, y_v, x_grid, y_grid, z_var);
        (void)result;
    };
    auto nodes_naive = countVariNodes(naive_no_recover);
    auto nodes_anal = countVariNodes(analytical_no_recover);
    auto bytes_naive = measureArenaBytes(naive_no_recover, MEM_BATCH);
    auto bytes_anal = measureArenaBytes(analytical_no_recover, MEM_BATCH);

    printMemory("bilinear 2D", nodes_naive, nodes_anal, bytes_naive, bytes_anal, MEM_BATCH);

    auto double_once = [&]() {
        BilinearInterpolation<double> interp(x_grid, y_grid, z_vals);
        volatile double result = interp(x_eval, y_eval);
        (void)result;
    };
    auto t_dbl = benchmark(double_once, N);
    auto t_naive = benchmark(naive_once, N);
    auto t_anal = benchmark(analytical_once, N);
    printTiming(t_dbl, t_naive, t_anal);
}

// ═══════════════════════════════════════════════════════════════════════════
// 3. CUBIC SPLINE INTERPOLATION BENCHMARK
// ═══════════════════════════════════════════════════════════════════════════

void benchCubicSpline() {
    printSection("1D Cubic Spline Interpolation (20 knots)");

    constexpr int NKNOTS = 20;
    constexpr int N = 50'000;
    constexpr int MEM_BATCH = 500;

    // Grid: sin(x) on [0, 2pi]
    std::vector<double> x_grid(NKNOTS);
    std::vector<double> y_vals(NKNOTS);
    for (int i = 0; i < NKNOTS; ++i) {
        x_grid[i] = i * 2.0 * M_PI / (NKNOTS - 1);
        y_vals[i] = std::sin(x_grid[i]);
    }

    double x_eval = 2.5;

    auto naive_once = [&]() {
        std::vector<var> y_var(NKNOTS);
        for (int i = 0; i < NKNOTS; ++i)
            y_var[i] = y_vals[i];
        var x_v = x_eval;
        CubicInterpolation<var> interp(std::vector<var>(x_grid.begin(), x_grid.end()), y_var,
                                       CubicInterpolation<var>::Spline);
        var result = interp(x_v);
        stan::math::grad(result.vi_);
        stan::math::recover_memory();
    };

    auto analytical_once = [&]() {
        std::vector<var> y_var(NKNOTS);
        for (int i = 0; i < NKNOTS; ++i)
            y_var[i] = y_vals[i];
        var x_v = x_eval;
        var result = cubicSplineInterpAnalytical(x_v, x_grid, y_var);
        stan::math::grad(result.vi_);
        stan::math::recover_memory();
    };

    // ── Correctness: verify all n sensitivities match ──
    {
        std::vector<var> y_naive(NKNOTS), y_anal(NKNOTS);
        for (int i = 0; i < NKNOTS; ++i) {
            y_naive[i] = y_vals[i];
            y_anal[i] = y_vals[i];
        }

        var x_n = x_eval;
        CubicInterpolation<var> interp(std::vector<var>(x_grid.begin(), x_grid.end()), y_naive,
                                       CubicInterpolation<var>::Spline);
        var res_n = interp(x_n);
        stan::math::grad(res_n.vi_);

        double val_naive = res_n.val();
        double dx_naive = x_n.adj();
        std::vector<double> adj_naive(NKNOTS);
        for (int i = 0; i < NKNOTS; ++i)
            adj_naive[i] = y_naive[i].adj();
        stan::math::recover_memory();

        var x_a = x_eval;
        for (int i = 0; i < NKNOTS; ++i)
            y_anal[i] = y_vals[i];
        var res_a = cubicSplineInterpAnalytical(x_a, x_grid, y_anal);
        stan::math::grad(res_a.vi_);

        std::cout << "  Correctness:\n";
        std::cout << "    Value:  naive=" << std::setprecision(10) << val_naive
                  << "  analytical=" << res_a.val()
                  << "  diff=" << std::abs(val_naive - res_a.val()) << "\n";
        std::cout << "    dval/dx:  naive=" << dx_naive << "  analytical=" << x_a.adj()
                  << "  diff=" << std::abs(dx_naive - x_a.adj()) << "\n";

        double max_diff = 0;
        int worst_idx = 0;
        std::cout << "\n    Sensitivities dy[k] (first 5 and worst):\n";
        std::cout << "      " << std::setw(6) << "k" << std::setw(16) << "Naive" << std::setw(16)
                  << "Analytical" << std::setw(14) << "Abs Diff" << "\n";
        for (int i = 0; i < NKNOTS; ++i) {
            double d = std::abs(adj_naive[i] - y_anal[i].adj());
            if (d > max_diff) {
                max_diff = d;
                worst_idx = i;
            }
            if (i < 5) {
                std::cout << "      " << std::setw(6) << i << std::setw(16) << adj_naive[i]
                          << std::setw(16) << y_anal[i].adj() << std::setw(14) << d << "\n";
            }
        }
        std::cout << "      " << std::setw(6) << worst_idx << std::setw(16) << adj_naive[worst_idx]
                  << std::setw(16) << y_anal[worst_idx].adj() << std::setw(14) << max_diff
                  << " (worst)\n";
        std::cout << "\n    Max |dy_adj diff| across all " << NKNOTS << " knots: " << max_diff
                  << "\n\n";
        stan::math::recover_memory();
    }

    // ── Memory ──
    auto naive_no_recover = [&]() {
        std::vector<var> y_var(NKNOTS);
        for (int i = 0; i < NKNOTS; ++i)
            y_var[i] = y_vals[i];
        var x_v = x_eval;
        CubicInterpolation<var> interp(std::vector<var>(x_grid.begin(), x_grid.end()), y_var,
                                       CubicInterpolation<var>::Spline);
        var result = interp(x_v);
        (void)result;
    };
    auto analytical_no_recover = [&]() {
        std::vector<var> y_var(NKNOTS);
        for (int i = 0; i < NKNOTS; ++i)
            y_var[i] = y_vals[i];
        var x_v = x_eval;
        var result = cubicSplineInterpAnalytical(x_v, x_grid, y_var);
        (void)result;
    };
    auto nodes_naive = countVariNodes(naive_no_recover);
    auto nodes_anal = countVariNodes(analytical_no_recover);
    auto bytes_naive = measureArenaBytes(naive_no_recover, MEM_BATCH);
    auto bytes_anal = measureArenaBytes(analytical_no_recover, MEM_BATCH);

    printMemory("cubic spline", nodes_naive, nodes_anal, bytes_naive, bytes_anal, MEM_BATCH);

    auto double_once = [&]() {
        CubicInterpolation<double> interp(x_grid, y_vals, CubicInterpolation<double>::Spline);
        volatile double result = interp(x_eval);
        (void)result;
    };
    auto t_dbl = benchmark(double_once, N);
    auto t_naive = benchmark(naive_once, N);
    auto t_anal = benchmark(analytical_once, N);
    printTiming(t_dbl, t_naive, t_anal);
}

// ═══════════════════════════════════════════════════════════════════════════
// MAIN
// ═══════════════════════════════════════════════════════════════════════════

int main() {
    std::cout << std::fixed;

    std::cout << "╔═════════════════════════════════════════════════════════════╗\n";
    std::cout << "║  Interpolation AD Benchmark: Naive Autodiff vs Analytical  ║\n";
    std::cout << "╚═════════════════════════════════════════════════════════════╝\n";

    try {
        benchLinear();
        benchBilinear();
        benchCubicSpline();

        std::cout << "\n" << std::string(65, '=') << "\n";
        std::cout << "  All benchmarks complete.\n";
        std::cout << std::string(65, '=') << "\n";
    } catch (const std::exception& e) {
        std::cerr << "ERROR: " << e.what() << "\n";
        return 1;
    }

    return 0;
}
