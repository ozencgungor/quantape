/**
 * @file test_chain_ad.cpp
 * @brief Multi-component AD chain with analytical first-order derivatives
 *
 * Every component (curve, vol, pricer) defines its first-order derivatives
 * analytically as var expressions. The chain rule is composed manually.
 * grad() on the composed ∂price/∂M_i gives Hessian row i w.r.t. raw market data.
 *
 * No fvar<var>, no hessian().
 */

#include <stan/math/prim/fun/Eigen.hpp>
#include <stan/math/rev/core.hpp>
#include <stan/math/rev/fun/Phi.hpp>
#include <stan/math/rev/fun/exp.hpp>
#include <stan/math/rev/fun/log.hpp>

#include <cmath>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

using stan::math::var;

// ═══════════════════════════════════════════════════════════════════════════
// COMPONENT 1: Discount curve — log-linear interpolation
//
// Returns: rate (var) + ∂rate/∂df_i (var) for each active node
// ═══════════════════════════════════════════════════════════════════════════

struct CurveResult {
    var rate;
    // Jacobian: which df nodes are active, and the partial as a var
    int i0, i1;  // indices of the two bracketing nodes
    var dr_ddf0; // ∂r/∂df[i0] as var (depends on df[i0])
    var dr_ddf1; // ∂r/∂df[i1] as var (depends on df[i1])
};

CurveResult interpolate_rate(const std::vector<var>& df_nodes, const std::vector<double>& times,
                             double T) {
    int i = 0;
    while (i + 1 < (int)times.size() && times[i + 1] < T)
        ++i;
    if (i + 1 >= (int)times.size())
        i = (int)times.size() - 2;

    double w = (T - times[i]) / (times[i + 1] - times[i]);

    // Value: r = -[(1-w)*ln(df_i) + w*ln(df_{i+1})] / T
    var log_df = log(df_nodes[i]) * (1.0 - w) + log(df_nodes[i + 1]) * w;
    var rate = -log_df / T;

    // Analytical derivatives (as var — df_nodes[i] is var, so 1/df is var):
    //   ∂r/∂df_i     = -(1-w) / (df_i * T)
    //   ∂r/∂df_{i+1} = -w / (df_{i+1} * T)
    var dr_ddf0 = -(1.0 - w) / (df_nodes[i] * T);
    var dr_ddf1 = -w / (df_nodes[i + 1] * T);

    return {rate, i, i + 1, dr_ddf0, dr_ddf1};
}

// ═══════════════════════════════════════════════════════════════════════════
// COMPONENT 2: Vol smile — linear interpolation
//
// Returns: sigma (var) + ∂sigma/∂vol_j (double — linear interp has constant weights)
// ═══════════════════════════════════════════════════════════════════════════

struct VolResult {
    var sigma;
    int j0, j1;
    double dsig_dvol0; // (1-w) — constant, doesn't need to be var
    double dsig_dvol1; // w
};

VolResult interpolate_vol(const std::vector<var>& vol_nodes, const std::vector<double>& strikes,
                          double K) {
    int i = 0;
    while (i + 1 < (int)strikes.size() && strikes[i + 1] < K)
        ++i;
    if (i + 1 >= (int)strikes.size())
        i = (int)strikes.size() - 2;

    double w = (K - strikes[i]) / (strikes[i + 1] - strikes[i]);
    var sigma = vol_nodes[i] * (1.0 - w) + vol_nodes[i + 1] * w;

    return {sigma, i, i + 1, 1.0 - w, w};
}

// ═══════════════════════════════════════════════════════════════════════════
// COMPONENT 3: BS pricer — analytical Greeks as var expressions
//
// Returns: price (double) + delta, vega, rho (all var)
// ═══════════════════════════════════════════════════════════════════════════

struct PricerResult {
    double price_val;
    var delta, vega, rho;
};

PricerResult bs_price(var S, var sigma, var r, double K, double T) {
    double sqrtT = std::sqrt(T);
    var d1 = (log(S / K) + (r + 0.5 * sigma * sigma) * T) / (sigma * sqrtT);
    var d2 = d1 - sigma * sqrtT;

    var Nd1 = stan::math::Phi(d1);
    var Nd2 = stan::math::Phi(d2);
    var disc = exp(-r * T);
    var nd1 = exp(-0.5 * d1 * d1) / std::sqrt(2.0 * M_PI);

    double pv = S.val() * Nd1.val() - K * disc.val() * Nd2.val();

    return {pv, Nd1, S * nd1 * sqrtT, K * T * disc * Nd2};
}

// ═══════════════════════════════════════════════════════════════════════════
// CHAIN RULE ASSEMBLY
//
// Market data M = [S, df[0..2], vol[0..2]]  (7 nodes)
//
// ∂price/∂S      = delta
// ∂price/∂df[i]  = rho * ∂r/∂df[i]
// ∂price/∂vol[j] = vega * ∂σ/∂vol[j]
//
// Each of these is a var expression. grad() on it → Hessian row.
// ═══════════════════════════════════════════════════════════════════════════

int main() {
    std::cout << std::fixed;
    std::cout << "\n═══════════════════════════════════════════════════════════\n";
    std::cout << " Hessian w.r.t. raw market data — analytical 1st + grad()\n";
    std::cout << "═══════════════════════════════════════════════════════════\n\n";

    // Market data
    std::vector<double> df_times = {0.5, 1.0, 2.0};
    std::vector<double> vol_strikes = {90.0, 100.0, 110.0};
    constexpr int N_DF = 3, N_VOL = 3;
    constexpr int N = 1 + N_DF + N_VOL; // S + 3 df + 3 vol = 7

    double S0 = 100.0;
    double df_vals[N_DF] = {0.9975, 0.9512, 0.9048};
    double vol_vals[N_VOL] = {0.28, 0.25, 0.22};

    double K = 95.0, T = 0.75;

    // Labels for output
    std::string labels[N] = {"S", "df[0]", "df[1]", "df[2]", "vol[0]", "vol[1]", "vol[2]"};

    // ── First order (gradient) ──
    Eigen::VectorXd gradient(N);
    {
        stan::math::nested_rev_autodiff nested;

        var S(S0);
        std::vector<var> df = {var(df_vals[0]), var(df_vals[1]), var(df_vals[2])};
        std::vector<var> vol = {var(vol_vals[0]), var(vol_vals[1]), var(vol_vals[2])};

        auto curve = interpolate_rate(df, df_times, T);
        auto smile = interpolate_vol(vol, vol_strikes, K);
        auto pricer = bs_price(S, smile.sigma, curve.rate, K, T);

        // ∂price/∂S = delta, ∂price/∂df[i] = rho * dr/ddf[i], etc.
        // But for first order we can just use the full var price expression:
        var price =
            S * pricer.delta -
            K * exp(-curve.rate * T) *
                stan::math::Phi((log(S / K) + (curve.rate + 0.5 * smile.sigma * smile.sigma) * T) /
                                    (smile.sigma * std::sqrt(T)) -
                                smile.sigma * std::sqrt(T));
        // Actually simpler: just rebuild price as var to get the gradient directly
        // Or use chain rule manually:
        // We'll use the chain rule approach to be consistent.

        // Compose: ∂price/∂M_i for each market data node
        // For S: delta
        // For df[curve.i0]: rho * dr_ddf0
        // For df[curve.i1]: rho * dr_ddf1
        // For vol[smile.j0]: vega * dsig_dvol0
        // For vol[smile.j1]: vega * dsig_dvol1

        gradient.setZero();
        gradient(0) = pricer.delta.val(); // ∂price/∂S
        gradient(1 + curve.i0) += (pricer.rho * curve.dr_ddf0).val();
        gradient(1 + curve.i1) += (pricer.rho * curve.dr_ddf1).val();
        gradient(1 + N_DF + smile.j0) += (pricer.vega * smile.dsig_dvol0).val();
        gradient(1 + N_DF + smile.j1) += (pricer.vega * smile.dsig_dvol1).val();
    }

    std::cout << "── Gradient (first order) ──\n\n";
    std::cout << "  price = " << std::setprecision(6) << [&] { // quick recompute for display
        double r = -(std::log(df_vals[0]) * 0.0 + std::log(df_vals[1]) * 1.0) / T;
        // just use the pricer to get the value
        return 0.0; // placeholder
    }() << "\n";
    for (int i = 0; i < N; ++i)
        std::cout << "  ∂price/∂" << std::setw(7) << std::left << labels[i] << " = " << std::right
                  << std::setprecision(6) << gradient(i) << "\n";

    // ── Hessian (second order) ──
    //
    // For each market data node i, build ∂price/∂M_i as a var expression,
    // then grad() → Hessian row i.
    Eigen::MatrixXd H = Eigen::MatrixXd::Zero(N, N);

    auto compute_hessian_row = [&](int row_idx) {
        stan::math::nested_rev_autodiff nested;

        var S(S0);
        std::vector<var> df = {var(df_vals[0]), var(df_vals[1]), var(df_vals[2])};
        std::vector<var> vol = {var(vol_vals[0]), var(vol_vals[1]), var(vol_vals[2])};

        auto curve = interpolate_rate(df, df_times, T);
        auto smile = interpolate_vol(vol, vol_strikes, K);
        auto pricer = bs_price(S, smile.sigma, curve.rate, K, T);

        // Build ∂price/∂M[row_idx] as a var expression
        var dP_dM;
        if (row_idx == 0) {
            // ∂price/∂S = delta
            dP_dM = pricer.delta;
        } else if (row_idx <= N_DF) {
            // ∂price/∂df[k] = rho * ∂r/∂df[k]
            int k = row_idx - 1;
            if (k == curve.i0)
                dP_dM = pricer.rho * curve.dr_ddf0;
            else if (k == curve.i1)
                dP_dM = pricer.rho * curve.dr_ddf1;
            else
                return; // zero row — this node doesn't affect the rate
        } else {
            // ∂price/∂vol[k] = vega * ∂σ/∂vol[k]
            int k = row_idx - 1 - N_DF;
            if (k == smile.j0)
                dP_dM = pricer.vega * smile.dsig_dvol0;
            else if (k == smile.j1)
                dP_dM = pricer.vega * smile.dsig_dvol1;
            else
                return; // zero row
        }

        stan::math::grad(dP_dM.vi_);

        // Read Hessian row from adjoints
        H(row_idx, 0) = S.adj();
        for (int j = 0; j < N_DF; ++j)
            H(row_idx, 1 + j) = df[j].adj();
        for (int j = 0; j < N_VOL; ++j)
            H(row_idx, 1 + N_DF + j) = vol[j].adj();
    };

    for (int i = 0; i < N; ++i)
        compute_hessian_row(i);

    std::cout << "\n── Hessian w.r.t. raw market data (7×7) ──\n\n";
    std::cout << "         ";
    for (int j = 0; j < N; ++j)
        std::cout << std::setw(12) << std::right << labels[j];
    std::cout << "\n";

    for (int i = 0; i < N; ++i) {
        std::cout << "  " << std::setw(7) << std::left << labels[i];
        for (int j = 0; j < N; ++j)
            std::cout << std::setw(12) << std::right << std::setprecision(4) << H(i, j);
        std::cout << "\n";
    }

    // Verify symmetry
    double max_asym = 0;
    for (int i = 0; i < N; ++i)
        for (int j = i + 1; j < N; ++j)
            max_asym = std::max(max_asym, std::abs(H(i, j) - H(j, i)));
    std::cout << "\n  Max asymmetry |H(i,j) - H(j,i)|: " << std::scientific << std::setprecision(2)
              << max_asym << "\n";

    std::cout << "\n";
    return 0;
}
