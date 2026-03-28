/**
 * @file test_fast_exp_log.cpp
 * @brief Benchmark: fast bit-trick exp/log vs std::exp/log, with AD
 *
 * Exploits:
 *   - IEEE 754 bit layout for fast approximate exp and log
 *   - d/dx exp(x) = exp(x)  → adjoint is free (just reuse the value)
 *   - d/dx log(x) = 1/x     → adjoint is one division you already have
 *
 * Accuracy tiers:
 *   1. Schraudolph:  raw bit trick (~2-3% relative error)
 *   2. Refined:      bit trick + cubic correction (~0.01% error)
 *   3. std::exp/log: full precision baseline
 */

#include <stan/math.hpp>

#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <random>
#include <vector>

using stan::math::var;

// ═══════════════════════════════════════════════════════════════════════════
// FAST EXP IMPLEMENTATIONS
// ═══════════════════════════════════════════════════════════════════════════

// ── Schraudolph's method (1999): ~2-3% relative error ──
// exp(x) ≈ reinterpret(a * x + b) where a,b chosen to map x → IEEE 754 bits
inline double fast_exp_schraudolph(double x) {
    // Constants: a = 2^52 / ln(2), b = 2^52 * (1023 - correction)
    // correction ≈ 0.04367744890362246 (minimizes max relative error)
    constexpr double a = 6497320848556798.0;   // 2^52 / ln(2)
    constexpr double b = 4606853616395542.0;   // 2^52 * (1023 - 0.04367...)
    int64_t i = static_cast<int64_t>(a * x + b);
    double result;
    std::memcpy(&result, &i, sizeof(result));
    return result;
}

// ── Range-reduced with polynomial: ~1e-6 relative error ──
// Split x = n*ln(2) + r, compute 2^n * P(r) with minimax polynomial
inline double fast_exp_refined(double x) {
    // Range reduction: x = n * ln(2) + r,  |r| <= ln(2)/2
    constexpr double LOG2E = 1.4426950408889634;   // 1/ln(2)
    constexpr double LN2_HI = 0.6931471805599453;  // ln(2) high bits
    constexpr double LN2_LO = 2.3190468138462996e-17; // ln(2) low bits

    double n = std::round(x * LOG2E);
    double r = x - n * LN2_HI - n * LN2_LO;

    // Minimax polynomial for exp(r)-1 on [-ln2/2, ln2/2]
    // P(r) = 1 + r + r²/2 + r³/6 + r⁴/24 + r⁵/120
    double r2 = r * r;
    double p = 1.0 + r * (1.0 + r * (0.5 + r * (0.16666666666666666
               + r * (0.041666666666666664 + r * 0.008333333333333333))));

    // Reconstruct: exp(x) = 2^n * P(r)
    // Multiply by 2^n via bit manipulation on the exponent
    int64_t bits;
    std::memcpy(&bits, &p, sizeof(bits));
    bits += static_cast<int64_t>(n) << 52;
    double result;
    std::memcpy(&result, &bits, sizeof(result));
    return result;
}

// ═══════════════════════════════════════════════════════════════════════════
// FAST LOG IMPLEMENTATIONS
// ═══════════════════════════════════════════════════════════════════════════

// ── Bit-trick log: ~2-3% relative error ──
inline double fast_log_schraudolph(double x) {
    int64_t bits;
    std::memcpy(&bits, &x, sizeof(bits));
    // Inverse of the exp trick: log(x) ≈ (bits - b) / a
    constexpr double a = 6497320848556798.0;
    constexpr double b = 4606853616395542.0;
    return (static_cast<double>(bits) - b) / a;
}

// ── Range-reduced log: ~1e-7 relative error ──
inline double fast_log_refined(double x) {
    // Extract exponent and mantissa via bit manipulation
    int64_t bits;
    std::memcpy(&bits, &x, sizeof(bits));

    int exponent = static_cast<int>((bits >> 52) & 0x7FF) - 1023;

    // Set exponent to 0 → mantissa in [1, 2)
    int64_t mantissa_bits = (bits & 0x000FFFFFFFFFFFFFLL) | 0x3FF0000000000000LL;
    double m;
    std::memcpy(&m, &mantissa_bits, sizeof(m));

    // log(x) = exponent * ln(2) + log(m),  m ∈ [1, 2)
    // Remap: let f = m - 1, f ∈ [0, 1)
    // log(1+f) ≈ f - f²/2 + f³/3 - f⁴/4 + f⁵/5 - f⁶/6  (Taylor, or better: minimax)
    // Better: use Padé or shift to [sqrt(2)/2, sqrt(2)] for faster convergence

    // Shift range to [sqrt(2)/2, sqrt(2)] for better polynomial convergence
    if (m > 1.4142135623730951) {  // sqrt(2)
        m *= 0.5;
        exponent += 1;
    }

    double f = m - 1.0;
    double f2 = f * f;

    // Minimax polynomial for log(1+f), f ∈ [-0.2929, 0.4142]
    // Horner form: f * (1 - f/2 + f²/3 - f³/4 + f⁴/5 - f⁵/6 + f⁶/7)
    double log_m = f * (1.0 + f * (-0.5 + f * (0.33333333333333333
                   + f * (-0.25 + f * (0.2 + f * (-0.16666666666666666
                   + f * 0.14285714285714285))))));

    return exponent * 0.6931471805599453 + log_m;
}

// ═══════════════════════════════════════════════════════════════════════════
// STAN AD WRAPPERS — analytical adjoints using make_callback_var
// ═══════════════════════════════════════════════════════════════════════════

// ── exp wrappers: d/dx exp(x) = exp(x) — adjoint IS the value ──

inline var std_exp_analytical(const var& x) {
    double val = std::exp(x.val());
    return stan::math::make_callback_var(val, [x, val](auto& vi) {
        x.adj() += vi.adj() * val;
    });
}

inline var fast_exp_schraudolph_var(const var& x) {
    double val = fast_exp_schraudolph(x.val());
    return stan::math::make_callback_var(val, [x, val](auto& vi) {
        x.adj() += vi.adj() * val;
    });
}

inline var fast_exp_refined_var(const var& x) {
    double val = fast_exp_refined(x.val());
    return stan::math::make_callback_var(val, [x, val](auto& vi) {
        x.adj() += vi.adj() * val;
    });
}

// ── log wrappers: d/dx log(x) = 1/x ──

inline var std_log_analytical(const var& x) {
    double xv = x.val();
    double val = std::log(xv);
    return stan::math::make_callback_var(val, [x, xv](auto& vi) {
        x.adj() += vi.adj() / xv;
    });
}

inline var fast_log_schraudolph_var(const var& x) {
    double xv = x.val();
    double val = fast_log_schraudolph(xv);
    return stan::math::make_callback_var(val, [x, xv](auto& vi) {
        x.adj() += vi.adj() / xv;
    });
}

inline var fast_log_refined_var(const var& x) {
    double xv = x.val();
    double val = fast_log_refined(xv);
    return stan::math::make_callback_var(val, [x, xv](auto& vi) {
        x.adj() += vi.adj() / xv;
    });
}

// ═══════════════════════════════════════════════════════════════════════════
// ACCURACY MEASUREMENT
// ═══════════════════════════════════════════════════════════════════════════

struct AccuracyStats {
    double max_rel_error;
    double avg_rel_error;
    double max_abs_error;
};

template <typename F>
AccuracyStats measureAccuracy(F&& approx, auto&& exact, const std::vector<double>& inputs) {
    AccuracyStats stats{0, 0, 0};
    int count = 0;
    for (double x : inputs) {
        double ref = exact(x);
        double val = approx(x);
        if (std::abs(ref) > 1e-300) {
            double rel = std::abs((val - ref) / ref);
            stats.max_rel_error = std::max(stats.max_rel_error, rel);
            stats.avg_rel_error += rel;
            stats.max_abs_error = std::max(stats.max_abs_error, std::abs(val - ref));
            ++count;
        }
    }
    if (count > 0) stats.avg_rel_error /= count;
    return stats;
}

// ═══════════════════════════════════════════════════════════════════════════
// BENCHMARK HARNESS
// ═══════════════════════════════════════════════════════════════════════════

struct BenchResult {
    double per_call_ns;
};

template <typename F>
BenchResult benchNs(F&& fn, const std::vector<double>& inputs, int iters) {
    // warm-up
    for (int i = 0; i < std::min(iters / 10, 1000); ++i)
        for (double x : inputs) fn(x);

    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < iters; ++i)
        for (double x : inputs) fn(x);
    auto end = std::chrono::high_resolution_clock::now();

    double ns = std::chrono::duration<double, std::nano>(end - start).count();
    return {ns / (iters * inputs.size())};
}

// For AD functions: create var, call, grad, recover
template <typename F>
BenchResult benchNsAD(F&& fn, const std::vector<double>& inputs, int iters) {
    for (int i = 0; i < std::min(iters / 10, 1000); ++i) {
        for (double x : inputs) {
            var v(x);
            var r = fn(v);
            stan::math::grad(r.vi_);
            stan::math::recover_memory();
        }
    }

    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < iters; ++i) {
        for (double x : inputs) {
            var v(x);
            var r = fn(v);
            stan::math::grad(r.vi_);
            stan::math::recover_memory();
        }
    }
    auto end = std::chrono::high_resolution_clock::now();

    double ns = std::chrono::duration<double, std::nano>(end - start).count();
    return {ns / (iters * inputs.size())};
}

// AD tape measurement
template <typename F>
std::size_t countNodes(F&& fn) {
    stan::math::recover_memory();
    auto* stack = stan::math::ChainableStack::instance_;
    std::size_t before = stack->var_stack_.size();
    var v(1.5);
    var r = fn(v);
    (void)r;
    std::size_t after = stack->var_stack_.size();
    stan::math::recover_memory();
    return after - before;
}

void printRow(const char* name, BenchResult dbl, BenchResult ad, AccuracyStats acc, std::size_t nodes) {
    std::cout << "  " << std::setw(26) << name
              << std::setw(10) << std::setprecision(1) << dbl.per_call_ns
              << std::setw(10) << std::setprecision(1) << ad.per_call_ns
              << std::setw(10) << std::setprecision(1) << ad.per_call_ns / dbl.per_call_ns
              << std::setw(14) << std::setprecision(2) << std::scientific << acc.max_rel_error
              << std::setw(8) << std::fixed << nodes
              << "\n";
}

// ═══════════════════════════════════════════════════════════════════════════
// MAIN
// ═══════════════════════════════════════════════════════════════════════════

int main() {
    std::cout << std::fixed;

    // Generate test inputs
    std::mt19937 rng(42);
    constexpr int NPTS = 1000;
    constexpr int ITERS = 5000;

    // exp inputs: x ∈ [-5, 5]  (covers most practical range)
    std::vector<double> exp_inputs(NPTS);
    std::uniform_real_distribution<> exp_dist(-5.0, 5.0);
    for (auto& x : exp_inputs) x = exp_dist(rng);

    // log inputs: x ∈ [0.01, 100]
    std::vector<double> log_inputs(NPTS);
    std::uniform_real_distribution<> log_dist(0.01, 100.0);
    for (auto& x : log_inputs) x = log_dist(rng);

    // ═══════════════════════════════════════════════════════════════
    // EXP BENCHMARK
    // ═══════════════════════════════════════════════════════════════
    std::cout << "╔═══════════════════════════════════════════════════════════════════════════╗\n";
    std::cout << "║  Fast exp/log Benchmark: Bit-tricks + Analytical AD                      ║\n";
    std::cout << "╚═══════════════════════════════════════════════════════════════════════════╝\n\n";

    std::cout << "── exp(x), x ∈ [-5, 5], " << NPTS << " points x " << ITERS << " iters ──\n\n";

    std::cout << "  " << std::setw(26) << "Method"
              << std::setw(10) << "dbl ns"
              << std::setw(10) << "AD ns"
              << std::setw(10) << "AD/dbl"
              << std::setw(14) << "Max Rel Err"
              << std::setw(8) << "Nodes" << "\n";
    std::cout << "  " << std::string(78, '-') << "\n";

    auto std_exp_fn = [](double x) -> double { return std::exp(x); };

    // Accuracy
    auto acc_std = measureAccuracy(std_exp_fn, std_exp_fn, exp_inputs);
    auto acc_schr = measureAccuracy(fast_exp_schraudolph, std_exp_fn, exp_inputs);
    auto acc_ref = measureAccuracy(fast_exp_refined, std_exp_fn, exp_inputs);

    // Double timing
    auto dbl_std  = benchNs([](double x) { volatile double r = std::exp(x); (void)r; }, exp_inputs, ITERS);
    auto dbl_schr = benchNs([](double x) { volatile double r = fast_exp_schraudolph(x); (void)r; }, exp_inputs, ITERS);
    auto dbl_ref  = benchNs([](double x) { volatile double r = fast_exp_refined(x); (void)r; }, exp_inputs, ITERS);

    // AD timing — stan::math::exp (naive, goes through tape)
    auto ad_stan = benchNsAD([](const var& x) { return stan::math::exp(x); }, exp_inputs, ITERS);
    auto nodes_stan = countNodes([](const var& x) { return stan::math::exp(x); });

    // AD timing — std::exp with analytical adjoint
    auto ad_std_anal = benchNsAD(std_exp_analytical, exp_inputs, ITERS);
    auto nodes_std_anal = countNodes(std_exp_analytical);

    // AD timing — Schraudolph with analytical adjoint
    auto ad_schr = benchNsAD(fast_exp_schraudolph_var, exp_inputs, ITERS);
    auto nodes_schr = countNodes(fast_exp_schraudolph_var);

    // AD timing — Refined with analytical adjoint
    auto ad_refn = benchNsAD(fast_exp_refined_var, exp_inputs, ITERS);
    auto nodes_refn = countNodes(fast_exp_refined_var);

    printRow("std::exp (double ref)", dbl_std, ad_stan, acc_std, nodes_stan);
    printRow("std::exp + anal. adj", dbl_std, ad_std_anal, acc_std, nodes_std_anal);
    printRow("Schraudolph + anal. adj", dbl_schr, ad_schr, acc_schr, nodes_schr);
    printRow("Refined + anal. adj", dbl_ref, ad_refn, acc_ref, nodes_refn);

    std::cout << "\n  Key: dbl ns = double-only time, AD ns = var + grad time, Nodes = tape nodes\n";

    // ═══════════════════════════════════════════════════════════════
    // LOG BENCHMARK
    // ═══════════════════════════════════════════════════════════════
    std::cout << "\n── log(x), x ∈ [0.01, 100], " << NPTS << " points x " << ITERS << " iters ──\n\n";

    std::cout << "  " << std::setw(26) << "Method"
              << std::setw(10) << "dbl ns"
              << std::setw(10) << "AD ns"
              << std::setw(10) << "AD/dbl"
              << std::setw(14) << "Max Rel Err"
              << std::setw(8) << "Nodes" << "\n";
    std::cout << "  " << std::string(78, '-') << "\n";

    auto std_log_fn = [](double x) -> double { return std::log(x); };

    auto lacc_std = measureAccuracy(std_log_fn, std_log_fn, log_inputs);
    auto lacc_schr = measureAccuracy(fast_log_schraudolph, std_log_fn, log_inputs);
    auto lacc_ref = measureAccuracy(fast_log_refined, std_log_fn, log_inputs);

    auto ldbl_std  = benchNs([](double x) { volatile double r = std::log(x); (void)r; }, log_inputs, ITERS);
    auto ldbl_schr = benchNs([](double x) { volatile double r = fast_log_schraudolph(x); (void)r; }, log_inputs, ITERS);
    auto ldbl_ref  = benchNs([](double x) { volatile double r = fast_log_refined(x); (void)r; }, log_inputs, ITERS);

    auto lad_stan = benchNsAD([](const var& x) { return stan::math::log(x); }, log_inputs, ITERS);
    auto lnodes_stan = countNodes([](const var& x) { return stan::math::log(x); });

    auto lad_std_anal = benchNsAD(std_log_analytical, log_inputs, ITERS);
    auto lnodes_std_anal = countNodes(std_log_analytical);

    auto lad_schr = benchNsAD(fast_log_schraudolph_var, log_inputs, ITERS);
    auto lnodes_schr = countNodes(fast_log_schraudolph_var);

    auto lad_refn = benchNsAD(fast_log_refined_var, log_inputs, ITERS);
    auto lnodes_refn = countNodes(fast_log_refined_var);

    printRow("std::log (Stan naive)", ldbl_std, lad_stan, lacc_std, lnodes_stan);
    printRow("std::log + anal. adj", ldbl_std, lad_std_anal, lacc_std, lnodes_std_anal);
    printRow("Schraudolph + anal. adj", ldbl_schr, lad_schr, lacc_schr, lnodes_schr);
    printRow("Refined + anal. adj", ldbl_ref, lad_refn, lacc_ref, lnodes_refn);

    // ═══════════════════════════════════════════════════════════════
    // COMPOUND BENCHMARK: exp(a*x + b) — common in finance (discounting)
    // ═══════════════════════════════════════════════════════════════
    std::cout << "\n── Compound: exp(a*x + b) with a=-0.05, b=0.01, x ∈ [0, 30] ──\n";
    std::cout << "  (Models discount factor computation: exp(-r*T))\n\n";

    std::vector<double> compound_inputs(NPTS);
    std::uniform_real_distribution<> compound_dist(0.0, 30.0);
    for (auto& x : compound_inputs) x = compound_dist(rng);

    double a_param = -0.05, b_param = 0.01;

    // Double baseline
    auto cdbl = benchNs([&](double x) {
        volatile double r = std::exp(a_param * x + b_param);
        (void)r;
    }, compound_inputs, ITERS);

    // Stan naive: all operations on tape
    auto cad_naive = benchNsAD([&](const var& x) {
        return stan::math::exp(a_param * x + b_param);
    }, compound_inputs, ITERS);
    auto cnodes_naive = countNodes([&](const var& x) {
        return stan::math::exp(a_param * x + b_param);
    });

    // Analytical: compute value in double, push adjoint manually
    // d/dx exp(a*x + b) = a * exp(a*x + b)
    auto cad_anal = benchNsAD([&](const var& x) -> var {
        double xv = x.val();
        double val = std::exp(a_param * xv + b_param);
        double dval_dx = a_param * val;
        return stan::math::make_callback_var(val, [x, dval_dx](auto& vi) {
            x.adj() += vi.adj() * dval_dx;
        });
    }, compound_inputs, ITERS);
    auto cnodes_anal = countNodes([&](const var& x) -> var {
        double xv = x.val();
        double val = std::exp(a_param * xv + b_param);
        double dval_dx = a_param * val;
        return stan::math::make_callback_var(val, [x, dval_dx](auto& vi) {
            x.adj() += vi.adj() * dval_dx;
        });
    });

    // Fast exp + analytical
    auto cad_fast = benchNsAD([&](const var& x) -> var {
        double xv = x.val();
        double val = fast_exp_refined(a_param * xv + b_param);
        double dval_dx = a_param * val;
        return stan::math::make_callback_var(val, [x, dval_dx](auto& vi) {
            x.adj() += vi.adj() * dval_dx;
        });
    }, compound_inputs, ITERS);
    auto cnodes_fast = countNodes([&](const var& x) -> var {
        double xv = x.val();
        double val = fast_exp_refined(a_param * xv + b_param);
        double dval_dx = a_param * val;
        return stan::math::make_callback_var(val, [x, dval_dx](auto& vi) {
            x.adj() += vi.adj() * dval_dx;
        });
    });

    std::cout << "  " << std::setw(26) << "Method"
              << std::setw(10) << "dbl ns"
              << std::setw(10) << "AD ns"
              << std::setw(10) << "AD/dbl"
              << std::setw(8) << "Nodes" << "\n";
    std::cout << "  " << std::string(64, '-') << "\n";

    auto crow = [](const char* name, BenchResult dbl, BenchResult ad, std::size_t nodes) {
        std::cout << "  " << std::setw(26) << name
                  << std::setw(10) << std::setprecision(1) << dbl.per_call_ns
                  << std::setw(10) << std::setprecision(1) << ad.per_call_ns
                  << std::setw(10) << std::setprecision(1) << ad.per_call_ns / dbl.per_call_ns
                  << std::setw(8) << nodes << "\n";
    };

    crow("Stan naive (a*x+b+exp)", cdbl, cad_naive, cnodes_naive);
    crow("std::exp + anal. adj", cdbl, cad_anal, cnodes_anal);
    crow("fast_exp + anal. adj", cdbl, cad_fast, cnodes_fast);

    std::cout << "\n  Stan naive creates " << cnodes_naive << " tape nodes for exp(a*x+b)\n";
    std::cout << "  Analytical adjoint: " << cnodes_anal << " tape node, derivative = a*exp(a*x+b)\n";

    // ═══════════════════════════════════════════════════════════════
    // BLACK-SCHOLES WITH FAST PRIMITIVES
    // ═══════════════════════════════════════════════════════════════
    std::cout << "\n── Black-Scholes call price using fast exp+log primitives ──\n\n";

    // Full BS with fast exp and log + analytical adjoint for the whole thing
    // vs. BS built from fast_exp_var and fast_log_var as building blocks
    auto bs_double = [](double S, double K, double sigma, double r, double T) {
        double sqrtT = std::sqrt(T);
        double d1 = (std::log(S/K) + (r + 0.5*sigma*sigma)*T) / (sigma*sqrtT);
        double d2 = d1 - sigma*sqrtT;
        double Nd1 = 0.5 * std::erfc(-d1 * M_SQRT1_2);
        double Nd2 = 0.5 * std::erfc(-d2 * M_SQRT1_2);
        return S * Nd1 - K * std::exp(-r*T) * Nd2;
    };

    auto bs_fast_double = [](double S, double K, double sigma, double r, double T) {
        double sqrtT = std::sqrt(T);
        double d1 = (fast_log_refined(S/K) + (r + 0.5*sigma*sigma)*T) / (sigma*sqrtT);
        double d2 = d1 - sigma*sqrtT;
        double Nd1 = 0.5 * std::erfc(-d1 * M_SQRT1_2);
        double Nd2 = 0.5 * std::erfc(-d2 * M_SQRT1_2);
        return S * Nd1 - K * fast_exp_refined(-r*T) * Nd2;
    };

    // Check accuracy
    double ref_price = bs_double(100, 105, 0.2, 0.05, 1.0);
    double fast_price = bs_fast_double(100, 105, 0.2, 0.05, 1.0);
    std::cout << "  BS price (std):  " << std::setprecision(10) << ref_price << "\n";
    std::cout << "  BS price (fast): " << fast_price << "\n";
    std::cout << "  Rel error:       " << std::scientific << std::setprecision(2)
              << std::abs(fast_price - ref_price) / ref_price << "\n\n";

    constexpr int BS_N = 100'000;

    // Time double versions
    auto bs_std_time = benchNs([&](double x) {
        volatile double p = bs_double(100+x*0.01, 105, 0.2, 0.05, 1.0);
        (void)p;
    }, exp_inputs, BS_N / NPTS);

    auto bs_fast_time = benchNs([&](double x) {
        volatile double p = bs_fast_double(100+x*0.01, 105, 0.2, 0.05, 1.0);
        (void)p;
    }, exp_inputs, BS_N / NPTS);

    std::cout << std::fixed;
    std::cout << "  BS double (std::exp/log): " << std::setprecision(1) << bs_std_time.per_call_ns << " ns\n";
    std::cout << "  BS double (fast exp/log): " << bs_fast_time.per_call_ns << " ns\n";
    std::cout << "  Speedup: " << std::setprecision(2) << bs_std_time.per_call_ns / bs_fast_time.per_call_ns << "x\n";

    std::cout << "\n═══════════════════════════════════════════════════════════════════════════\n";
    return 0;
}
