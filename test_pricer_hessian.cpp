/**
 * @file test_pricer_hessian.cpp
 * @brief Production architecture for mixed 1st/2nd order AD through a pricer hierarchy
 *
 * Architecture:
 *
 *   PricerBase
 *     └── price<DoubleT>(MarketEnv<DoubleT>&) → DoubleT
 *     └── loadMarketData<DoubleT>(MarketEnv<DoubleT>&, const RawMarketData&)
 *
 *   TradePricer
 *     └── computeGreeks()    — 1st order: price<var>(), grad(), collect adjoints
 *     └── computeHessian()   — 2nd order: loop over columns with fvar<var>
 *
 * Each pricer's price<DoubleT> has 3 tiers via if constexpr:
 *   DoubleT == double    → plain evaluation
 *   DoubleT == var       → make_callback_var with analytical gradient (if available)
 *   DoubleT == fvar<var> → nested analytical (if available), else falls through
 *
 * Pricers that DON'T override for fvar<var> still work — the templated
 * code compiles with fvar<var> and Stan tapes everything (Level 0).
 * Pricers that DO override get the speed benefit.
 *
 * The TradePricer doesn't know or care which level each pricer uses.
 */

#include <stan/math.hpp>
#include <stan/math/mix.hpp>
#include <Eigen/Dense>

#include <chrono>
#include <cmath>
#include <functional>
#include <iomanip>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

using stan::math::var;
using stan::math::fvar;

// ═══════════════════════════════════════════════════════════════════════════
// MARKET ENVIRONMENT — holds market data with the active scalar type
//
// In production this would reference yield curves, vol surfaces, etc.
// Here we simplify to a flat vector of "market data points".
// ═══════════════════════════════════════════════════════════════════════════

template <typename T>
struct MarketEnv {
    std::vector<T> data;          // all market data as a flat vector
    int n_rates = 0;              // first n_rates entries are rates
    int n_vols = 0;               // next n_vols entries are vols

    T rate(int i) const { return data[i]; }
    T vol(int i) const { return data[n_rates + i]; }
};

// ═══════════════════════════════════════════════════════════════════════════
// PRICER BASE — CRTP for compile-time polymorphism
//
// We use CRTP because price() must be templated on DoubleT.
// Runtime polymorphism (virtual) is achieved via type erasure below.
// ═══════════════════════════════════════════════════════════════════════════

template <typename Derived>
struct PricerCRTP {
    template <typename T>
    T price(const MarketEnv<T>& env) const {
        return static_cast<const Derived*>(this)->priceImpl(env);
    }

    std::string name() const {
        return static_cast<const Derived*>(this)->nameImpl();
    }
};

// ═══════════════════════════════════════════════════════════════════════════
// TYPE ERASURE — wraps any CRTP pricer into a common interface
//
// This allows TradePricer to hold heterogeneous pricers in a vector,
// while still dispatching to templated price<T>() at compile time.
//
// The key trick: we store std::function<T(MarketEnv<T>&)> for each T
// we might need (double, var, fvar<var>).
// ═══════════════════════════════════════════════════════════════════════════

struct PricerHandle {
    std::function<double(const MarketEnv<double>&)>       price_double;
    std::function<var(const MarketEnv<var>&)>              price_var;
    std::function<fvar<var>(const MarketEnv<fvar<var>>&)>  price_fvar_var;
    std::string name;

    template <typename Derived>
    static PricerHandle create(const Derived& pricer) {
        PricerHandle h;
        h.name = pricer.name();
        // Capture the pricer and bind each scalar type
        h.price_double = [p = pricer](const MarketEnv<double>& e) {
            return p.template price<double>(e);
        };
        h.price_var = [p = pricer](const MarketEnv<var>& e) {
            return p.template price<var>(e);
        };
        h.price_fvar_var = [p = pricer](const MarketEnv<fvar<var>>& e) {
            return p.template price<fvar<var>>(e);
        };
        return h;
    }

    template <typename T>
    T price(const MarketEnv<T>& env) const {
        if constexpr (std::is_same_v<T, double>)
            return price_double(env);
        else if constexpr (std::is_same_v<T, var>)
            return price_var(env);
        else
            return price_fvar_var(env);
    }
};

// ═══════════════════════════════════════════════════════════════════════════
// PRICER 1: VANILLA EUROPEAN (Level 2 — full analytical 1st + 2nd order)
//
// This pricer has closed-form Greeks and second-order Greeks.
// For var:       make_callback_var with Delta/Vega/Rho
// For fvar<var>: nested make_callback_var with Gamma/Vanna/Volga
// ═══════════════════════════════════════════════════════════════════════════

struct VanillaEuropeanPricer : PricerCRTP<VanillaEuropeanPricer> {
    double K = 100.0, T = 1.0;

    std::string nameImpl() const { return "VanillaEuropean (Level 2)"; }

    template <typename DoubleT>
    DoubleT priceImpl(const MarketEnv<DoubleT>& env) const {
        DoubleT S = env.rate(0) * 100.0;  // simplified: rate → spot
        DoubleT sigma = env.vol(0);
        DoubleT r = env.rate(1);

        if constexpr (std::is_same_v<DoubleT, var>) {
            return priceAnalytical(S, sigma, r);
        } else if constexpr (std::is_same_v<DoubleT, fvar<var>>) {
            return priceFvarVar(S, sigma, r);
        } else {
            return priceNaive(S, sigma, r);
        }
    }

private:
    // ── Helpers ──
    static double phi(double x) { return std::exp(-0.5*x*x) / std::sqrt(2*M_PI); }
    static double Phi(double x) { return 0.5 * std::erfc(-x * M_SQRT1_2); }

    struct D1D2 { double d1, d2, nd1, nd2, Nd1, Nd2, disc, sqrtT; };
    D1D2 compute(double S, double sigma, double r) const {
        double sqrtT = std::sqrt(T);
        double d1 = (std::log(S/K) + (r + 0.5*sigma*sigma)*T) / (sigma*sqrtT);
        double d2 = d1 - sigma*sqrtT;
        return {d1, d2, phi(d1), phi(d2), Phi(d1), Phi(d2),
                std::exp(-r*T), sqrtT};
    }

    // ── double ──
    double priceNaive(double S, double sigma, double r) const {
        auto c = compute(S, sigma, r);
        return S * c.Nd1 - K * c.disc * c.Nd2;
    }

    // ── var: Level 2 first-order (1 tape node) ──
    var priceAnalytical(var S, var sigma, var r) const {
        double Sv = S.val(), sv = sigma.val(), rv = r.val();
        auto c = compute(Sv, sv, rv);
        double val = Sv * c.Nd1 - K * c.disc * c.Nd2;

        double delta = c.Nd1;
        double vega  = Sv * c.nd1 * c.sqrtT;
        double rho   = K * T * c.disc * c.Nd2;

        return stan::math::make_callback_var(val,
            [S, sigma, r, delta, vega, rho](auto& vi) {
                double a = vi.adj();
                S.adj()     += a * delta;
                sigma.adj() += a * vega;
                r.adj()     += a * rho;
            });
    }

    // ── fvar<var>: Level 2 second-order (nested make_callback_var) ──
    fvar<var> priceFvarVar(fvar<var> S, fvar<var> sigma, fvar<var> r) const {
        var Sv = S.val_, sv = sigma.val_, rv = r.val_;
        var Sd = S.d_,   sd = sigma.d_,   rd = r.d_;

        double S0 = Sv.val(), s0 = sv.val(), r0 = rv.val();
        auto c = compute(S0, s0, r0);
        double val = S0 * c.Nd1 - K * c.disc * c.Nd2;

        // 1st order Greeks
        double delta = c.Nd1;
        double vega  = S0 * c.nd1 * c.sqrtT;
        double rho_g = K * T * c.disc * c.Nd2;

        // 2nd order Greeks
        double gamma_   = c.nd1 / (S0 * s0 * c.sqrtT);
        double vanna_   = -c.nd1 * c.d2 / s0;
        double dDdr     = c.nd1 * c.sqrtT / s0;
        double volga_   = S0 * c.sqrtT * c.nd1 * c.d1 * c.d2 / s0;
        double dVdr     = -S0 * T * c.d1 * c.nd1 / s0;
        double dRdr     = -K*T*T*c.disc*c.Nd2 + K*T*c.disc*c.nd2*c.sqrtT/s0;

        var price_var(val);

        var delta_var = stan::math::make_callback_var(delta,
            [Sv, sv, rv, gamma_, vanna_, dDdr](auto& vi) {
                double a = vi.adj();
                Sv.adj() += a * gamma_; sv.adj() += a * vanna_; rv.adj() += a * dDdr;
            });
        var vega_var = stan::math::make_callback_var(vega,
            [Sv, sv, rv, vanna_, volga_, dVdr](auto& vi) {
                double a = vi.adj();
                Sv.adj() += a * vanna_; sv.adj() += a * volga_; rv.adj() += a * dVdr;
            });
        var rho_var = stan::math::make_callback_var(rho_g,
            [Sv, sv, rv, dDdr, dVdr, dRdr](auto& vi) {
                double a = vi.adj();
                Sv.adj() += a * dDdr; sv.adj() += a * dVdr; rv.adj() += a * dRdr;
            });

        var tangent = delta_var * Sd + vega_var * sd + rho_var * rd;
        return fvar<var>(price_var, tangent);
    }
};

// ═══════════════════════════════════════════════════════════════════════════
// PRICER 2: EXOTIC (Level 0 — no analytical Greeks, AD tapes everything)
//
// Simulates a pricer that uses numerical methods internally (PDE, MC, etc.)
// The same templated code works for double, var, AND fvar<var>.
// No special overloads needed — it "just works" at the cost of more tape.
// ═══════════════════════════════════════════════════════════════════════════

struct ExoticPricer : PricerCRTP<ExoticPricer> {
    double K = 100.0, T = 1.0;

    std::string nameImpl() const { return "Exotic (Level 0 — naive)"; }

    template <typename DoubleT>
    DoubleT priceImpl(const MarketEnv<DoubleT>& env) const {
        // Same BS formula but NO analytical shortcuts — everything taped
        DoubleT S = env.rate(0) * 100.0;
        DoubleT sigma = env.vol(0);
        DoubleT r = env.rate(1);

        using std::log; using std::exp; using std::sqrt;
        DoubleT sqrtT = sqrt(DoubleT(T));
        DoubleT d1 = (log(S / K) + (r + sigma*sigma/2.0)*T) / (sigma*sqrtT);
        DoubleT d2 = d1 - sigma*sqrtT;
        return S * stan::math::Phi(d1) - K * exp(-r*T) * stan::math::Phi(d2);
    }
};

// ═══════════════════════════════════════════════════════════════════════════
// PRICER 3: DIGITAL (Level 1 — gradient as var, Hessian via AD)
//
// Has a known analytical gradient but expressing the Hessian
// analytically is tedious. Instead, the gradient is expressed as
// var operations in the fvar<var> overload, and AD handles the rest.
// ═══════════════════════════════════════════════════════════════════════════

struct DigitalPricer : PricerCRTP<DigitalPricer> {
    double K = 100.0, T = 1.0;

    std::string nameImpl() const { return "Digital (Level 1 — grad as var)"; }

    template <typename DoubleT>
    DoubleT priceImpl(const MarketEnv<DoubleT>& env) const {
        DoubleT S = env.rate(0) * 100.0;
        DoubleT sigma = env.vol(0);
        DoubleT r = env.rate(1);

        if constexpr (std::is_same_v<DoubleT, var>) {
            return priceAnalytical(S, sigma, r);
        } else if constexpr (std::is_same_v<DoubleT, fvar<var>>) {
            return priceLevel1(S, sigma, r);
        } else {
            return priceNaive(S, sigma, r);
        }
    }

private:
    static double phi(double x) { return std::exp(-0.5*x*x) / std::sqrt(2*M_PI); }
    static double Phi(double x) { return 0.5 * std::erfc(-x * M_SQRT1_2); }

    // Digital call: PV = e^{-rT} · Φ(d₂)
    double priceNaive(double S, double sigma, double r) const {
        double sqrtT = std::sqrt(T);
        double d2 = (std::log(S/K) + (r - 0.5*sigma*sigma)*T) / (sigma*sqrtT);
        return std::exp(-r*T) * Phi(d2);
    }

    // ── var: analytical gradient (1 tape node) ──
    var priceAnalytical(var S, var sigma, var r) const {
        double Sv = S.val(), sv = sigma.val(), rv = r.val();
        double sqrtT = std::sqrt(T);
        double d2 = (std::log(Sv/K) + (rv - 0.5*sv*sv)*T) / (sv*sqrtT);
        double disc = std::exp(-rv*T);
        double val = disc * Phi(d2);

        double nd2 = phi(d2);
        double dd2_dS = 1.0 / (Sv * sv * sqrtT);
        double dd2_dsigma = -(std::log(Sv/K) + (rv + 0.5*sv*sv)*T) / (sv*sv*sqrtT);
        double dd2_dr = sqrtT / sv;

        double dPV_dS = disc * nd2 * dd2_dS;
        double dPV_dsigma = disc * nd2 * dd2_dsigma;
        double dPV_dr = -T * val + disc * nd2 * dd2_dr;

        return stan::math::make_callback_var(val,
            [S, sigma, r, dPV_dS, dPV_dsigma, dPV_dr](auto& vi) {
                double a = vi.adj();
                S.adj()     += a * dPV_dS;
                sigma.adj() += a * dPV_dsigma;
                r.adj()     += a * dPV_dr;
            });
    }

    // ── fvar<var>: Level 1 — gradient as var operations ──
    // We express the gradient entries as var, so AD can differentiate
    // through them for the Hessian. No need to derive Hessian by hand.
    fvar<var> priceLevel1(fvar<var> S, fvar<var> sigma, fvar<var> r) const {
        var Sv = S.val_, sv = sigma.val_, rv = r.val_;
        var Sd = S.d_,   sd = sigma.d_,   rd = r.d_;

        // Value in double
        double S0 = Sv.val(), s0 = sv.val(), r0 = rv.val();
        double sqrtT = std::sqrt(T);
        double d2_val = (std::log(S0/K) + (r0 - 0.5*s0*s0)*T) / (s0*sqrtT);
        double val = std::exp(-r0*T) * Phi(d2_val);
        var price_var(val);

        // Gradient entries as var (AD will differentiate through these)
        // d₂ as var — depends on S, σ, r
        var d2_var = (log(Sv / K) + (rv - sv*sv/2.0)*T) / (sv*sqrtT);
        var disc_var = exp(-rv * T);
        var nd2_var = exp(-d2_var * d2_var / 2.0) / std::sqrt(2*M_PI);
        var Nd2_var = stan::math::Phi(d2_var);

        var dd2_dS = 1.0 / (Sv * sv * sqrtT);
        var dd2_dsigma = -(log(Sv/K) + (rv + sv*sv/2.0)*T) / (sv*sv*sqrtT);
        var dd2_dr = var(sqrtT) / sv;

        var grad_S = disc_var * nd2_var * dd2_dS;
        var grad_sigma = disc_var * nd2_var * dd2_dsigma;
        var grad_r = -T * disc_var * Nd2_var + disc_var * nd2_var * dd2_dr;

        // Tangent = J · d (AD can differentiate through grad_* for Hessian)
        var tangent = grad_S * Sd + grad_sigma * sd + grad_r * rd;

        return fvar<var>(price_var, tangent);
    }
};

// ═══════════════════════════════════════════════════════════════════════════
// TRADE PRICER — orchestrates 1st and 2nd order Greeks
//
// This is the main interface. It doesn't know which analytical level
// each pricer uses. It just calls price<T>() and collects adjoints.
// ═══════════════════════════════════════════════════════════════════════════

class TradePricer {
public:
    // ── First-order Greeks (existing pipeline) ──
    struct FirstOrderResult {
        double pv;
        Eigen::VectorXd greeks;  // ∂PV/∂θ_i
    };

    FirstOrderResult computeGreeks(const PricerHandle& pricer,
                                    const std::vector<double>& market_data,
                                    int n_rates, int n_vols) {
        int n = market_data.size();
        MarketEnv<var> env;
        env.n_rates = n_rates;
        env.n_vols = n_vols;
        env.data.resize(n);
        for (int i = 0; i < n; ++i)
            env.data[i] = var(market_data[i]);

        var pv = pricer.price(env);
        stan::math::grad(pv.vi_);

        Eigen::VectorXd greeks(n);
        for (int i = 0; i < n; ++i)
            greeks(i) = env.data[i].adj();

        stan::math::recover_memory();
        return {pv.val(), greeks};
    }

    // ── Second-order Greeks (Hessian) ──
    //
    // Uses fvar<var> column-by-column (same as stan::math::hessian internally).
    // For each column j:
    //   1. Create fvar<var> market data with tangent direction e_j
    //   2. Price with fvar<var>
    //   3. result.d_ = ∂PV/∂θ_j as a var on the tape
    //   4. grad(result.d_) → ∂/∂θ_i(∂PV/∂θ_j) = H[i][j]
    //
    // The pricer's fvar<var> overload determines speed:
    //   Level 0: full tape → slow but correct
    //   Level 1: gradient as var → moderate tape
    //   Level 2: nested analytical → minimal tape
    //
    struct SecondOrderResult {
        double pv;
        Eigen::VectorXd greeks;
        Eigen::MatrixXd hessian;
    };

    SecondOrderResult computeHessian(const PricerHandle& pricer,
                                      const std::vector<double>& market_data,
                                      int n_rates, int n_vols) {
        int n = market_data.size();
        Eigen::VectorXd greeks(n);
        Eigen::MatrixXd H(n, n);
        double pv = 0;

        for (int j = 0; j < n; ++j) {
            // Nested scope: each column gets its own tape
            stan::math::nested_rev_autodiff nested;

            MarketEnv<fvar<var>> env;
            env.n_rates = n_rates;
            env.n_vols = n_vols;
            env.data.resize(n);
            for (int i = 0; i < n; ++i)
                env.data[i] = fvar<var>(var(market_data[i]), i == j ? 1.0 : 0.0);

            fvar<var> result = pricer.price(env);

            if (j == 0)
                pv = result.val_.val();
            greeks(j) = result.d_.val();

            // Reverse pass on the tangent → Hessian column j
            stan::math::grad(result.d_.vi_);
            for (int i = 0; i < n; ++i)
                H(i, j) = env.data[i].val_.adj();
        }

        return {pv, greeks, H};
    }

    // ── Second-order with sparsity ──
    //
    // In practice, most trades depend on only a few market data points.
    // Only compute Hessian columns for active variables.
    //
    SecondOrderResult computeHessianSparse(
        const PricerHandle& pricer,
        const std::vector<double>& market_data,
        int n_rates, int n_vols,
        const std::vector<int>& active_indices)  // which θ_j to differentiate
    {
        int n = market_data.size();
        int n_active = active_indices.size();
        Eigen::VectorXd greeks(n);
        greeks.setZero();
        Eigen::MatrixXd H = Eigen::MatrixXd::Zero(n, n);
        double pv = 0;

        for (int jj = 0; jj < n_active; ++jj) {
            int j = active_indices[jj];
            stan::math::nested_rev_autodiff nested;

            MarketEnv<fvar<var>> env;
            env.n_rates = n_rates;
            env.n_vols = n_vols;
            env.data.resize(n);
            for (int i = 0; i < n; ++i)
                env.data[i] = fvar<var>(var(market_data[i]), i == j ? 1.0 : 0.0);

            fvar<var> result = pricer.price(env);

            if (jj == 0)
                pv = result.val_.val();
            greeks(j) = result.d_.val();

            stan::math::grad(result.d_.vi_);
            for (int i = 0; i < n; ++i)
                H(i, j) = env.data[i].val_.adj();
        }

        // Symmetrize
        for (int jj = 0; jj < n_active; ++jj) {
            int j = active_indices[jj];
            for (int i = 0; i < n; ++i)
                H(j, i) = H(i, j);
        }

        return {pv, greeks, H};
    }
};

// ═══════════════════════════════════════════════════════════════════════════
// BENCHMARK
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
    std::cout << "║  Pricer Hierarchy: Mixed 1st/2nd Order AD                    ║\n";
    std::cout << "╚═══════════════════════════════════════════════════════════════╝\n\n";

    // Market data: 2 rates + 1 vol = 3 parameters
    std::vector<double> market_data = {1.05, 0.05, 0.20};  // S_proxy, r, vol
    int n_rates = 2, n_vols = 1;

    // Create pricers at different analytical levels
    VanillaEuropeanPricer vanilla;
    ExoticPricer exotic;
    DigitalPricer digital;

    auto h_vanilla = PricerHandle::create(vanilla);
    auto h_exotic  = PricerHandle::create(exotic);
    auto h_digital = PricerHandle::create(digital);

    TradePricer tp;

    std::vector<PricerHandle*> pricers = {&h_vanilla, &h_exotic, &h_digital};

    for (auto* pricer : pricers) {
        std::cout << "── " << pricer->name << " ──\n\n";

        // First order
        auto fo = tp.computeGreeks(*pricer, market_data, n_rates, n_vols);
        std::cout << "  PV = " << std::setprecision(6) << fo.pv << "\n";
        std::cout << "  Greeks: [";
        for (int i = 0; i < fo.greeks.size(); ++i)
            std::cout << (i ? ", " : "") << std::setprecision(6) << fo.greeks(i);
        std::cout << "]\n";

        // Second order
        auto so = tp.computeHessian(*pricer, market_data, n_rates, n_vols);
        std::cout << "  Hessian:\n";
        for (int i = 0; i < so.hessian.rows(); ++i) {
            std::cout << "    [";
            for (int j = 0; j < so.hessian.cols(); ++j)
                std::cout << std::setw(12) << std::setprecision(4) << so.hessian(i,j);
            std::cout << " ]\n";
        }

        // Verify gradient consistency
        double max_grad_diff = (fo.greeks - so.greeks).cwiseAbs().maxCoeff();
        std::cout << "  Max |gradient_1st - gradient_2nd|: "
                  << std::scientific << std::setprecision(2) << max_grad_diff
                  << std::fixed << "\n";

        // Verify Hessian symmetry
        double max_sym = (so.hessian - so.hessian.transpose()).cwiseAbs().maxCoeff();
        std::cout << "  Hessian symmetry check: " << std::scientific << max_sym
                  << std::fixed << "\n\n";
    }

    // ── Timing comparison ──
    std::cout << "── Performance ──\n\n";
    constexpr int N = 50'000;

    std::cout << "  " << std::setw(35) << std::left << "Pricer" << std::right
              << std::setw(12) << "1st order"
              << std::setw(12) << "2nd order"
              << std::setw(10) << "ratio\n";
    std::cout << "  " << std::string(69, '-') << "\n";

    for (auto* pricer : pricers) {
        auto t1 = bench_us([&]() {
            tp.computeGreeks(*pricer, market_data, n_rates, n_vols);
        }, N);

        auto t2 = bench_us([&]() {
            tp.computeHessian(*pricer, market_data, n_rates, n_vols);
        }, N);

        std::cout << "  " << std::setw(35) << std::left << pricer->name << std::right
                  << std::setw(10) << std::setprecision(3) << t1 << " us"
                  << std::setw(10) << std::setprecision(3) << t2 << " us"
                  << std::setw(8) << std::setprecision(1) << t2 / t1 << "x\n";
    }

    // ── Sparse Hessian demo ──
    std::cout << "\n── Sparse Hessian (only rate columns) ──\n\n";
    std::vector<int> rate_indices = {0, 1};  // only differentiate w.r.t. rates
    auto sparse = tp.computeHessianSparse(h_vanilla, market_data, n_rates, n_vols, rate_indices);
    std::cout << "  Hessian (only rate columns computed):\n";
    for (int i = 0; i < sparse.hessian.rows(); ++i) {
        std::cout << "    [";
        for (int j = 0; j < sparse.hessian.cols(); ++j)
            std::cout << std::setw(12) << std::setprecision(4) << sparse.hessian(i,j);
        std::cout << " ]\n";
    }
    std::cout << "  (Saves " << (1 - (double)rate_indices.size() / market_data.size()) * 100
              << "% of Hessian columns)\n";

    // ── Architecture diagram ──
    std::cout << "\n── Architecture ──\n\n";
    std::cout << "  TradePricer.computeHessian():\n";
    std::cout << "    for j = 0..n-1:                      ← one column per market data point\n";
    std::cout << "      nested_rev_autodiff scope\n";
    std::cout << "      md[i] = fvar<var>(var(θ_i), i==j)  ← tangent direction e_j\n";
    std::cout << "      fvar<var> pv = pricer.price(md)     ← pricer uses best overload\n";
    std::cout << "      grad(pv.d_)                         ← reverse on tangent\n";
    std::cout << "      H[:,j] = md[i].val_.adj()           ← read Hessian column\n\n";
    std::cout << "  Pricer overloads (each independent, composable):\n";
    std::cout << "    Level 0: template just works with fvar<var>     (no code change)\n";
    std::cout << "    Level 1: gradient as var → AD handles Hessian   (moderate speed)\n";
    std::cout << "    Level 2: nested make_callback_var               (maximum speed)\n";

    std::cout << "\n═══════════════════════════════════════════════════════════════\n";
    return 0;
}
