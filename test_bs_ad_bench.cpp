/**
 * @file test_bs_ad_bench.cpp
 * @brief Benchmark: naive templated Black-Scholes AD vs analytical adjoint
 *
 * Compares:
 *   1. Naive:      template<DoubleT> blackScholes using Stan primitives (log, exp, Phi)
 *                  — Stan builds the full expression graph automatically
 *   2. Analytical: hand-coded Greeks via make_callback_var (1 tape node)
 *
 * Measures: correctness, wall-clock time, and AD arena memory consumption.
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
// 1. NAIVE TEMPLATED BLACK-SCHOLES  (works for double and var)
//    Stan auto-builds the AD tape from every primitive operation.
// ═══════════════════════════════════════════════════════════════════════════

template <typename DoubleT>
DoubleT blackScholesNaive(DoubleT S, DoubleT K, DoubleT sigma, DoubleT r, DoubleT T) {
    using stan::math::Phi; // standard normal CDF, AD-aware
    using std::exp;
    using std::log;
    using std::sqrt;

    DoubleT sqrtT = sqrt(T);
    DoubleT d1 = (log(S / K) + (r + 0.5 * sigma * sigma) * T) / (sigma * sqrtT);
    DoubleT d2 = d1 - sigma * sqrtT;

    return S * Phi(d1) - K * exp(-r * T) * Phi(d2);
}

// ═══════════════════════════════════════════════════════════════════════════
// 2. ANALYTICAL ADJOINT BLACK-SCHOLES  (hand-coded Greeks, 1 tape node)
// ═══════════════════════════════════════════════════════════════════════════

namespace detail {

inline double phi_pdf(double x) {
    return std::exp(-0.5 * x * x) / std::sqrt(2.0 * M_PI);
}

inline double Phi_cdf(double x) {
    return 0.5 * std::erfc(-x * M_SQRT1_2);
}

} // namespace detail

inline var blackScholesAnalytical(const var& S_v, const var& K_v, const var& sigma_v,
                                  const var& r_v, const var& T_v) {
    const double S = S_v.val();
    const double K = K_v.val();
    const double sigma = sigma_v.val();
    const double r = r_v.val();
    const double T = T_v.val();

    const double sqrtT = std::sqrt(T);
    const double d1 = (std::log(S / K) + (r + 0.5 * sigma * sigma) * T) / (sigma * sqrtT);
    const double d2 = d1 - sigma * sqrtT;

    const double Nd1 = detail::Phi_cdf(d1);
    const double Nd2 = detail::Phi_cdf(d2);
    const double nd1 = detail::phi_pdf(d1);
    const double disc = std::exp(-r * T);

    const double price = S * Nd1 - K * disc * Nd2;

    // Analytical Greeks
    const double dC_dS = Nd1;                          // Delta
    const double dC_dK = -disc * Nd2;                  // Strike sens
    const double dC_dsigma = S * nd1 * sqrtT;          // Vega
    const double dC_dr = K * T * disc * Nd2;           // Rho
    const double dC_dT = 0.5 * S * nd1 * sigma / sqrtT // Theta (∂C/∂T)
                         + r * K * disc * Nd2;

    return stan::math::make_callback_var(
        price, [S_v, K_v, sigma_v, r_v, T_v, dC_dS, dC_dK, dC_dsigma, dC_dr, dC_dT](auto& vi) {
            const double adj = vi.adj();
            S_v.adj() += adj * dC_dS;
            K_v.adj() += adj * dC_dK;
            sigma_v.adj() += adj * dC_dsigma;
            r_v.adj() += adj * dC_dr;
            T_v.adj() += adj * dC_dT;
        });
}

// ═══════════════════════════════════════════════════════════════════════════
// DISPATCH: template that routes var → analytical, double → naive
// ═══════════════════════════════════════════════════════════════════════════

template <typename DoubleT>
DoubleT blackScholes(DoubleT S, DoubleT K, DoubleT sigma, DoubleT r, DoubleT T) {
    if constexpr (std::is_same_v<DoubleT, var>) {
        return blackScholesAnalytical(S, K, sigma, r, T);
    } else {
        return blackScholesNaive(S, K, sigma, r, T);
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// MEMORY MEASUREMENT
// ═══════════════════════════════════════════════════════════════════════════

// Measure bytes used on Stan's AD arena for N pricings WITHOUT recovering
// This gives a realistic picture of memory under portfolio-level usage.
template <typename PricingFunc>
std::size_t measureArenaBytesAccumulated(PricingFunc&& pricer, int N) {
    stan::math::recover_memory();
    auto* stack = stan::math::ChainableStack::instance_;
    std::size_t before = stack->memalloc_.bytes_allocated();

    for (int i = 0; i < N; ++i) {
        var S(100.0 + i * 0.001), K(105.0), sigma(0.20), r(0.05), T(1.0);
        var price = pricer(S, K, sigma, r, T);
        // Don't recover — accumulate on the tape like a real portfolio sweep
        (void)price;
    }

    std::size_t after = stack->memalloc_.bytes_allocated();
    stan::math::recover_memory();
    return after - before;
}

// Count vari nodes on the AD stack for a single pricing call
template <typename PricingFunc>
std::size_t countVariNodes(PricingFunc&& pricer) {
    stan::math::recover_memory();

    auto* stack = stan::math::ChainableStack::instance_;
    std::size_t nodes_before = stack->var_stack_.size();

    var S(100.0), K(105.0), sigma(0.20), r(0.05), T(1.0);
    var price = pricer(S, K, sigma, r, T);

    std::size_t nodes_after = stack->var_stack_.size();

    stan::math::grad(price.vi_);
    stan::math::recover_memory();

    return nodes_after - nodes_before;
}

// ═══════════════════════════════════════════════════════════════════════════
// BENCHMARK HARNESS
// ═══════════════════════════════════════════════════════════════════════════

struct Greeks {
    double price, delta, strike_sens, vega, rho, theta;
};

template <typename PricingFunc>
Greeks computeGreeks(PricingFunc&& pricer) {
    var S(100.0), K(105.0), sigma(0.20), r(0.05), T(1.0);
    var price = pricer(S, K, sigma, r, T);
    stan::math::grad(price.vi_);

    Greeks g;
    g.price = price.val();
    g.delta = S.adj();
    g.strike_sens = K.adj();
    g.vega = sigma.adj();
    g.rho = r.adj();
    g.theta = T.adj();

    stan::math::recover_memory();
    return g;
}

template <typename PricingFunc>
double benchmarkMicroseconds(PricingFunc&& pricer, int N) {
    // Warm-up
    for (int i = 0; i < 100; ++i) {
        var S(100.0), K(105.0), sigma(0.20), r(0.05), T(1.0);
        var price = pricer(S, K, sigma, r, T);
        stan::math::grad(price.vi_);
        stan::math::recover_memory();
    }

    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < N; ++i) {
        var S(100.0), K(105.0), sigma(0.20), r(0.05), T(1.0);
        var price = pricer(S, K, sigma, r, T);
        stan::math::grad(price.vi_);
        stan::math::recover_memory();
    }
    auto end = std::chrono::high_resolution_clock::now();

    return std::chrono::duration<double, std::micro>(end - start).count();
}

int main() {
    std::cout << std::setprecision(10) << std::fixed;

    constexpr int N = 100'000;

    auto naive_pricer = [](var S, var K, var sigma, var r, var T) {
        return blackScholesNaive<var>(S, K, sigma, r, T);
    };
    auto analytical_pricer = [](var S, var K, var sigma, var r, var T) {
        return blackScholesAnalytical(S, K, sigma, r, T);
    };

    // ── 1. Verify correctness ──
    std::cout << "═══════════════════════════════════════════════════════════\n";
    std::cout << "  Black-Scholes AD Benchmark: Naive vs Analytical\n";
    std::cout << "═══════════════════════════════════════════════════════════\n\n";

    Greeks gNaive = computeGreeks(naive_pricer);
    Greeks gAnalytical = computeGreeks(analytical_pricer);

    std::cout << "── Correctness (S=100, K=105, σ=0.20, r=0.05, T=1.0) ──\n\n";
    std::cout << std::setw(18) << "" << std::setw(18) << "Naive (autodiff)" << std::setw(18)
              << "Analytical" << std::setw(18) << "Abs Diff" << "\n";
    std::cout << std::string(72, '-') << "\n";

    auto row = [](const char* name, double naive, double analytical) {
        std::cout << std::setw(18) << name << std::setw(18) << naive << std::setw(18) << analytical
                  << std::setw(18) << std::abs(naive - analytical) << "\n";
    };

    row("Price", gNaive.price, gAnalytical.price);
    row("Delta", gNaive.delta, gAnalytical.delta);
    row("Strike sens", gNaive.strike_sens, gAnalytical.strike_sens);
    row("Vega", gNaive.vega, gAnalytical.vega);
    row("Rho", gNaive.rho, gAnalytical.rho);
    row("Theta", gNaive.theta, gAnalytical.theta);

    // ── 2. Memory measurement ──
    std::cout << "\n── Memory: AD arena + tape usage ──\n\n";

    std::size_t nodes_naive = countVariNodes(naive_pricer);
    std::size_t nodes_analytical = countVariNodes(analytical_pricer);

    constexpr int MEM_BATCH = 1000;
    std::size_t bytes_naive_batch = measureArenaBytesAccumulated(naive_pricer, MEM_BATCH);
    std::size_t bytes_analytical_batch = measureArenaBytesAccumulated(analytical_pricer, MEM_BATCH);

    double bytes_per_call_naive = (double)bytes_naive_batch / MEM_BATCH;
    double bytes_per_call_analytical = (double)bytes_analytical_batch / MEM_BATCH;

    std::cout << std::setprecision(10);
    std::cout << "  " << std::setw(28) << "" << std::setw(14) << "Naive" << std::setw(14)
              << "Analytical" << std::setw(10) << "Ratio" << "\n";
    std::cout << "  " << std::string(66, '-') << "\n";
    std::cout << "  " << std::setw(28) << "Tape (vari) nodes/call" << std::setw(14) << nodes_naive
              << std::setw(14) << nodes_analytical << std::setw(10) << std::setprecision(1)
              << (double)nodes_naive / nodes_analytical << "x\n";
    std::cout << "  " << std::setw(28) << std::setprecision(0) << "Arena bytes/call"
              << std::setw(14) << bytes_per_call_naive << std::setw(14) << bytes_per_call_analytical
              << std::setw(10) << std::setprecision(1)
              << bytes_per_call_naive / bytes_per_call_analytical << "x\n";
    std::cout << "  " << std::setw(28) << std::setprecision(0)
              << ("Arena for " + std::to_string(MEM_BATCH) + " calls") << std::setw(14)
              << bytes_naive_batch << std::setw(14) << bytes_analytical_batch << std::setw(10)
              << std::setprecision(1) << (double)bytes_naive_batch / bytes_analytical_batch
              << "x\n";

    constexpr int PORTFOLIO_SIZE = 10'000;
    std::cout << "\n  Projected for " << PORTFOLIO_SIZE
              << "-option portfolio (no recover between options):\n";
    std::cout << "    Naive:      " << std::setprecision(2)
              << bytes_per_call_naive * PORTFOLIO_SIZE / (1024.0 * 1024.0) << " MB\n";
    std::cout << "    Analytical: " << std::setprecision(2)
              << bytes_per_call_analytical * PORTFOLIO_SIZE / (1024.0 * 1024.0) << " MB\n";

    // ── 3. Performance benchmark (including double baseline) ──
    std::cout << std::setprecision(10);
    std::cout << "\n── Performance (" << N << " pricing + gradient calls) ──\n\n";

    // Double baseline: just price, no AD, no gradient
    auto double_bench_start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < N; ++i) {
        volatile double price = blackScholesNaive<double>(100.0, 105.0, 0.20, 0.05, 1.0);
        (void)price;
    }
    auto double_bench_end = std::chrono::high_resolution_clock::now();
    double us_double =
        std::chrono::duration<double, std::micro>(double_bench_end - double_bench_start).count();
    double avg_double = us_double / N;

    double us_naive = benchmarkMicroseconds(naive_pricer, N);
    double us_analytical = benchmarkMicroseconds(analytical_pricer, N);

    double avg_naive = us_naive / N;
    double avg_analytical = us_analytical / N;

    std::cout << "  double (no AD, price only):\n";
    std::cout << "    Per call: " << std::setprecision(3) << avg_double << " us  (baseline)\n\n";

    std::cout << "  Naive autodiff (price + gradient):\n";
    std::cout << "    Per call: " << std::setprecision(3) << avg_naive << " us  ("
              << std::setprecision(1) << avg_naive / avg_double << "x vs double)\n\n";

    std::cout << "  Analytical adjoint (price + gradient):\n";
    std::cout << "    Per call: " << std::setprecision(3) << avg_analytical << " us  ("
              << std::setprecision(1) << avg_analytical / avg_double << "x vs double)\n\n";

    std::cout << "  Analytical/Naive speedup: " << std::setprecision(2)
              << avg_naive / avg_analytical << "x\n\n";

    // ── 4. Portfolio-scale benchmark ──
    constexpr int PORTFOLIO_ITERS = 100;

    std::cout << "── Portfolio (" << PORTFOLIO_SIZE << " options x " << PORTFOLIO_ITERS
              << " iterations) ──\n\n";

    std::vector<double> spots(PORTFOLIO_SIZE), strikes(PORTFOLIO_SIZE), vols(PORTFOLIO_SIZE),
        rates(PORTFOLIO_SIZE), expiries(PORTFOLIO_SIZE);

    std::mt19937 rng(42);
    std::uniform_real_distribution<> spot_dist(80, 120);
    std::uniform_real_distribution<> strike_dist(70, 130);
    std::uniform_real_distribution<> vol_dist(0.10, 0.50);
    std::uniform_real_distribution<> rate_dist(0.01, 0.10);
    std::uniform_real_distribution<> expiry_dist(0.25, 5.0);

    for (int i = 0; i < PORTFOLIO_SIZE; ++i) {
        spots[i] = spot_dist(rng);
        strikes[i] = strike_dist(rng);
        vols[i] = vol_dist(rng);
        rates[i] = rate_dist(rng);
        expiries[i] = expiry_dist(rng);
    }

    // Double baseline: price-only, no AD
    auto port_double_start = std::chrono::high_resolution_clock::now();
    for (int iter = 0; iter < PORTFOLIO_ITERS; ++iter) {
        for (int i = 0; i < PORTFOLIO_SIZE; ++i) {
            volatile double price =
                blackScholesNaive<double>(spots[i], strikes[i], vols[i], rates[i], expiries[i]);
            (void)price;
        }
    }
    auto port_double_end = std::chrono::high_resolution_clock::now();
    double ms_double_port =
        std::chrono::duration<double, std::milli>(port_double_end - port_double_start).count();

    auto portfolio_bench = [&](auto pricer, const char* label) {
        auto start = std::chrono::high_resolution_clock::now();
        for (int iter = 0; iter < PORTFOLIO_ITERS; ++iter) {
            for (int i = 0; i < PORTFOLIO_SIZE; ++i) {
                var S(spots[i]), K(strikes[i]), sigma(vols[i]), r(rates[i]), T(expiries[i]);
                var price = pricer(S, K, sigma, r, T);
                stan::math::grad(price.vi_);
                stan::math::recover_memory();
            }
        }
        auto end = std::chrono::high_resolution_clock::now();
        double ms = std::chrono::duration<double, std::milli>(end - start).count();
        return ms;
    };

    double ms_naive_port = portfolio_bench(naive_pricer, "Naive (autodiff)");
    double ms_analytical_port = portfolio_bench(analytical_pricer, "Analytical (Greeks)");

    std::cout << "  " << std::setw(22) << "double (no AD)" << ": " << std::setprecision(1)
              << ms_double_port << " ms  (" << std::setprecision(3)
              << ms_double_port / PORTFOLIO_ITERS << " ms per sweep)  (baseline)\n";
    std::cout << "  " << std::setw(22) << "Naive (autodiff)" << ": " << std::setprecision(1)
              << ms_naive_port << " ms  (" << std::setprecision(3)
              << ms_naive_port / PORTFOLIO_ITERS << " ms per sweep)  " << std::setprecision(1)
              << ms_naive_port / ms_double_port << "x vs double\n";
    std::cout << "  " << std::setw(22) << "Analytical (Greeks)" << ": " << std::setprecision(1)
              << ms_analytical_port << " ms  (" << std::setprecision(3)
              << ms_analytical_port / PORTFOLIO_ITERS << " ms per sweep)  " << std::setprecision(1)
              << ms_analytical_port / ms_double_port << "x vs double\n";

    std::cout << "\n  Analytical/Naive speedup: " << std::setprecision(2)
              << ms_naive_port / ms_analytical_port << "x\n";

    std::cout << "\n═══════════════════════════════════════════════════════════\n";
    return 0;
}
