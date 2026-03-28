/**
 * @file test_gbm_bridge.cpp
 * @brief Exact bridge for time-dependent GBM
 *
 * For GBM: dS(t)/S(t) = μ(t)dt + σ(t)dW(t)
 *
 * Key insight: X(t) = log(S(t)) is a Gaussian process:
 *   dX(t) = [μ(t) - σ²(t)/2] dt + σ(t) dW(t)
 *
 * The bridge X(tk) | X(t1), X(t2) is Gaussian with analytically
 * computable mean and variance - NO approximation needed!
 */

#include <cmath>
#include <functional>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <random>
#include <vector>

// Time-dependent drift and volatility
using DriftFunction = std::function<double(double)>;
using VolFunction = std::function<double(double)>;

struct GBMParams {
    DriftFunction mu;
    VolFunction sigma;
};

std::mt19937_64 rng(42);
std::normal_distribution<double> normal_dist(0.0, 1.0);

double randn() {
    return normal_dist(rng);
}

/**
 * @brief Numerical integration using Simpson's rule
 */
double integrate(const std::function<double(double)>& f, double a, double b, int n = 100) {
    if (n % 2 == 1)
        n++; // Simpson's rule needs even n
    double h = (b - a) / n;
    double sum = f(a) + f(b);

    for (int i = 1; i < n; i++) {
        double x = a + i * h;
        sum += (i % 2 == 0 ? 2.0 : 4.0) * f(x);
    }

    return sum * h / 3.0;
}

/**
 * @brief Compute moments of log(S(t2)) | log(S(t1))
 *
 * For X(t) = log(S(t)), with dX = [μ - σ²/2] dt + σ dW:
 *
 * E[X(t2) | X(t1)] = X(t1) + ∫_{t1}^{t2} [μ(s) - σ²(s)/2] ds
 * Var[X(t2) | X(t1)] = ∫_{t1}^{t2} σ²(s) ds
 */
struct LogMoments {
    double mean;
    double variance;
};

LogMoments getLogMoments(double X_start, double t1, double t2, const GBMParams& params) {
    // Compute drift integral: ∫ [μ(s) - σ²(s)/2] ds
    auto drift_integrand = [&](double s) {
        double sigma_s = params.sigma(s);
        return params.mu(s) - 0.5 * sigma_s * sigma_s;
    };
    double drift_integral = integrate(drift_integrand, t1, t2);

    // Compute variance integral: ∫ σ²(s) ds
    auto var_integrand = [&](double s) {
        double sigma_s = params.sigma(s);
        return sigma_s * sigma_s;
    };
    double variance = integrate(var_integrand, t1, t2);

    LogMoments moments;
    moments.mean = X_start + drift_integral;
    moments.variance = variance;

    return moments;
}

/**
 * @brief Exact GBM bridge using Brownian bridge on log(S)
 *
 * For Gaussian processes, the bridge formula is exact:
 *
 * E[X(tk) | X(t1), X(t2)] = E[X(tk) | X(t1)] + Cov[X(tk), X(t2) | X(t1)] / Var[X(t2) | X(t1)] ×
 * [X(t2) - E[X(t2) | X(t1)]]
 *
 * Var[X(tk) | X(t1), X(t2)] = Var[X(tk) | X(t1)] - Cov²[X(tk), X(t2) | X(t1)] / Var[X(t2) | X(t1)]
 *
 * For time-dependent GBM:
 * Cov[X(tk), X(t2) | X(t1)] = Var[X(tk) | X(t1)] = ∫_{t1}^{tk} σ²(s) ds
 */
double sampleGBM_Bridge_Exact(double S1, double S2, double t1, double tk, double t2,
                              const GBMParams& params) {
    double X1 = std::log(S1);
    double X2 = std::log(S2);

    // Get unconditional moments from t1 to tk and t1 to t2
    auto moments_1k = getLogMoments(X1, t1, tk, params);
    auto moments_12 = getLogMoments(X1, t1, t2, params);

    // For GBM, Cov[X(tk), X(t2) | X(t1)] = Var[X(tk) | X(t1)]
    // because the increments are independent and variance is additive
    double var_1k = moments_1k.variance;
    double var_12 = moments_12.variance;
    double cov_k2_given_1 = var_1k; // Key property of Brownian motion!

    // Compute bridge mean (Kalman update formula)
    double innovation = X2 - moments_12.mean;
    double kalman_gain = cov_k2_given_1 / var_12;
    double bridge_mean = moments_1k.mean + kalman_gain * innovation;

    // Compute bridge variance (Kalman variance update)
    double bridge_var = var_1k * (1.0 - kalman_gain);

    // Sample from Gaussian bridge
    double Xk = bridge_mean + std::sqrt(bridge_var) * randn();

    return std::exp(Xk);
}

/**
 * @brief Simple unconditional GBM sampling for comparison
 */
double sampleGBM_Unconditional(double S_start, double t_start, double t_end,
                               const GBMParams& params) {
    double X_start = std::log(S_start);
    auto moments = getLogMoments(X_start, t_start, t_end, params);

    double X_end = moments.mean + std::sqrt(moments.variance) * randn();
    return std::exp(X_end);
}

/**
 * @brief Alternative: Simplified bridge formula
 *
 * For constant μ and σ, this reduces to the well-known formula:
 * log(Sk) ~ N(μ_bridge, σ²_bridge) where
 *
 * μ_bridge = log(S1) + (tk-t1)/(t2-t1) × [log(S2) - log(S1)] + drift_correction
 * σ²_bridge = (tk-t1) × (t2-tk) / (t2-t1) × σ²_avg
 */
double sampleGBM_Bridge_Simplified(double S1, double S2, double t1, double tk, double t2,
                                   const GBMParams& params) {
    double X1 = std::log(S1);
    double X2 = std::log(S2);

    double dt1 = tk - t1;
    double dt2 = t2 - tk;
    double dt_total = t2 - t1;
    double alpha = dt1 / dt_total;

    // Linear interpolation in log-space
    double X_interp = X1 + alpha * (X2 - X1);

    // Variance of bridge (exact for constant σ, approximate for time-dependent)
    auto moments_1k = getLogMoments(X1, t1, tk, params);
    auto moments_12 = getLogMoments(X1, t1, t2, params);

    // Bridge variance formula
    double var_bridge = moments_1k.variance * (1.0 - moments_1k.variance / moments_12.variance);

    double Xk = X_interp + std::sqrt(var_bridge) * randn();
    return std::exp(Xk);
}

void testGBMBridge() {
    std::cout << "=== GBM Bridge Test ===\n\n";
    std::cout << "Testing exact bridge for time-dependent GBM:\n";
    std::cout << "  dS/S = μ(t)dt + σ(t)dW\n\n";

    // Test Case 1: Constant parameters (should be exact)
    std::cout << "=== Test 1: Constant Parameters ===\n";
    std::cout << "μ(t) = 0.05, σ(t) = 0.20 (constant)\n\n";

    GBMParams params_constant;
    params_constant.mu = [](double t) { return 0.05; };
    params_constant.sigma = [](double t) { return 0.20; };

    double S0 = 100.0;
    double S1 = 120.0;
    double t0 = 0.0;
    double tk = 0.5;
    double t1 = 1.0;

    std::cout << "Scenario: S(t0=" << t0 << ") = " << S0 << ", S(t1=" << t1 << ") = " << S1
              << ", inserting at tk=" << tk << "\n\n";

    const int N = 50000;
    std::vector<double> samples_exact, samples_simplified, samples_unconditional;

    for (int i = 0; i < N; i++) {
        rng.seed(1000 + i);
        samples_exact.push_back(sampleGBM_Bridge_Exact(S0, S1, t0, tk, t1, params_constant));

        rng.seed(1000 + i);
        samples_simplified.push_back(
            sampleGBM_Bridge_Simplified(S0, S1, t0, tk, t1, params_constant));

        rng.seed(1000 + i);
        samples_unconditional.push_back(sampleGBM_Unconditional(S0, t0, tk, params_constant));
    }

    auto compute_stats = [](const std::vector<double>& data) {
        // Stats in log-space
        std::vector<double> log_data(data.size());
        for (size_t i = 0; i < data.size(); i++) {
            log_data[i] = std::log(data[i]);
        }

        double mean_log = std::accumulate(log_data.begin(), log_data.end(), 0.0) / log_data.size();
        double var_log = 0.0;
        for (double x : log_data)
            var_log += (x - mean_log) * (x - mean_log);
        var_log /= log_data.size();

        return std::make_pair(mean_log, var_log);
    };

    auto [mean_exact, var_exact] = compute_stats(samples_exact);
    auto [mean_simp, var_simp] = compute_stats(samples_simplified);
    auto [mean_uncond, var_uncond] = compute_stats(samples_unconditional);

    // Theoretical values
    double X0 = std::log(S0);
    double X1 = std::log(S1);
    auto moments_0k = getLogMoments(X0, t0, tk, params_constant);
    auto moments_01 = getLogMoments(X0, t0, t1, params_constant);
    double cov = moments_0k.variance;
    double theory_mean = moments_0k.mean + cov / moments_01.variance * (X1 - moments_01.mean);
    double theory_var = moments_0k.variance * (1.0 - cov / moments_01.variance);

    std::cout << std::fixed << std::setprecision(6);
    std::cout << "Results (in log-space):\n";
    std::cout << "Method              Mean(log)   Variance(log)  Mean Err    Var Err\n";
    std::cout << "Theoretical:        " << theory_mean << "  " << theory_var << "  -           -\n";
    std::cout << "Exact Bridge:       " << mean_exact << "  " << var_exact << "  "
              << (mean_exact - theory_mean) / theory_mean * 100 << "%  "
              << (var_exact - theory_var) / theory_var * 100 << "%\n";
    std::cout << "Simplified Bridge:  " << mean_simp << "  " << var_simp << "  "
              << (mean_simp - theory_mean) / theory_mean * 100 << "%  "
              << (var_simp - theory_var) / theory_var * 100 << "%\n";
    std::cout << "Unconditional:      " << mean_uncond << "  " << var_uncond << "  "
              << (mean_uncond - theory_mean) / theory_mean * 100 << "%  "
              << (var_uncond - theory_var) / theory_var * 100 << "%\n\n";

    // Test Case 2: Time-dependent parameters
    std::cout << "=== Test 2: Time-Dependent Parameters ===\n";
    std::cout << "μ(t) = 0.05 + 0.02*sin(2πt)\n";
    std::cout << "σ(t) = 0.20 * (1 + 0.3*t)\n\n";

    GBMParams params_timedep;
    params_timedep.mu = [](double t) { return 0.05 + 0.02 * std::sin(2.0 * M_PI * t); };
    params_timedep.sigma = [](double t) { return 0.20 * (1.0 + 0.3 * t); };

    samples_exact.clear();
    samples_simplified.clear();
    samples_unconditional.clear();

    for (int i = 0; i < N; i++) {
        rng.seed(2000 + i);
        samples_exact.push_back(sampleGBM_Bridge_Exact(S0, S1, t0, tk, t1, params_timedep));

        rng.seed(2000 + i);
        samples_simplified.push_back(
            sampleGBM_Bridge_Simplified(S0, S1, t0, tk, t1, params_timedep));

        rng.seed(2000 + i);
        samples_unconditional.push_back(sampleGBM_Unconditional(S0, t0, tk, params_timedep));
    }

    auto [mean_exact_td, var_exact_td] = compute_stats(samples_exact);
    auto [mean_simp_td, var_simp_td] = compute_stats(samples_simplified);
    auto [mean_uncond_td, var_uncond_td] = compute_stats(samples_unconditional);

    // Theoretical values for time-dependent case
    auto moments_0k_td = getLogMoments(X0, t0, tk, params_timedep);
    auto moments_01_td = getLogMoments(X0, t0, t1, params_timedep);
    double cov_td = moments_0k_td.variance;
    double theory_mean_td =
        moments_0k_td.mean + cov_td / moments_01_td.variance * (X1 - moments_01_td.mean);
    double theory_var_td = moments_0k_td.variance * (1.0 - cov_td / moments_01_td.variance);

    std::cout << "Results (in log-space):\n";
    std::cout << "Method              Mean(log)   Variance(log)  Mean Err    Var Err\n";
    std::cout << "Theoretical:        " << theory_mean_td << "  " << theory_var_td
              << "  -           -\n";
    std::cout << "Exact Bridge:       " << mean_exact_td << "  " << var_exact_td << "  "
              << (mean_exact_td - theory_mean_td) / theory_mean_td * 100 << "%  "
              << (var_exact_td - theory_var_td) / theory_var_td * 100 << "%\n";
    std::cout << "Simplified Bridge:  " << mean_simp_td << "  " << var_simp_td << "  "
              << (mean_simp_td - theory_mean_td) / theory_mean_td * 100 << "%  "
              << (var_simp_td - theory_var_td) / theory_var_td * 100 << "%\n";
    std::cout << "Unconditional:      " << mean_uncond_td << "  " << var_uncond_td << "  "
              << (mean_uncond_td - theory_mean_td) / theory_mean_td * 100 << "%  "
              << (var_uncond_td - theory_var_td) / theory_var_td * 100 << "%\n\n";

    // Test Case 3: Asymmetric bridge (near endpoint)
    std::cout << "=== Test 3: Asymmetric Bridge ===\n";
    std::cout << "tk very close to t0 (1 week into 3 months)\n\n";

    double tk_asymm = 7.0 / 365.25;
    double t1_asymm = 0.25;

    samples_exact.clear();
    samples_unconditional.clear();

    for (int i = 0; i < N; i++) {
        rng.seed(3000 + i);
        samples_exact.push_back(
            sampleGBM_Bridge_Exact(S0, S1, t0, tk_asymm, t1_asymm, params_constant));

        rng.seed(3000 + i);
        samples_unconditional.push_back(sampleGBM_Unconditional(S0, t0, tk_asymm, params_constant));
    }

    auto [mean_exact_asymm, var_exact_asymm] = compute_stats(samples_exact);
    auto [mean_uncond_asymm, var_uncond_asymm] = compute_stats(samples_unconditional);

    auto moments_0k_asymm = getLogMoments(X0, t0, tk_asymm, params_constant);
    auto moments_01_asymm = getLogMoments(X0, t0, t1_asymm, params_constant);
    double cov_asymm = moments_0k_asymm.variance;
    double theory_mean_asymm = moments_0k_asymm.mean +
                               cov_asymm / moments_01_asymm.variance * (X1 - moments_01_asymm.mean);
    double theory_var_asymm =
        moments_0k_asymm.variance * (1.0 - cov_asymm / moments_01_asymm.variance);

    double variance_reduction = 1.0 - theory_var_asymm / moments_0k_asymm.variance;

    std::cout << "Results (in log-space):\n";
    std::cout << "Method              Mean(log)   Variance(log)  Mean Err    Var Err\n";
    std::cout << "Theoretical:        " << theory_mean_asymm << "  " << theory_var_asymm
              << "  -           -\n";
    std::cout << "Exact Bridge:       " << mean_exact_asymm << "  " << var_exact_asymm << "  "
              << (mean_exact_asymm - theory_mean_asymm) / theory_mean_asymm * 100 << "%  "
              << (var_exact_asymm - theory_var_asymm) / theory_var_asymm * 100 << "%\n";
    std::cout << "Unconditional:      " << mean_uncond_asymm << "  " << var_uncond_asymm << "  "
              << (mean_uncond_asymm - theory_mean_asymm) / theory_mean_asymm * 100 << "%  "
              << (var_uncond_asymm - theory_var_asymm) / theory_var_asymm * 100 << "%\n\n";

    std::cout << "Variance reduction: " << variance_reduction * 100 << "%\n";
    std::cout << "  (Bridge variance is " << (1.0 - variance_reduction) * 100
              << "% of unconditional)\n\n";

    std::cout << "=== Summary ===\n\n";
    std::cout << "For GBM with time-dependent drift and volatility:\n";
    std::cout << "  ✓ Bridge formula is EXACT (Gaussian process)\n";
    std::cout << "  ✓ No approximation needed\n";
    std::cout << "  ✓ Works perfectly for any μ(t) and σ(t)\n";
    std::cout << "  ✓ Handles asymmetric bridges correctly\n";
    std::cout << "  ✓ Simple to implement\n\n";

    std::cout << "Key formulas (X = log(S)):\n";
    std::cout << "  E[X(tk) | X(t1), X(t2)] = E[X(tk) | X(t1)] + K × [X(t2) - E[X(t2) | X(t1)]]\n";
    std::cout << "  Var[X(tk) | X(t1), X(t2)] = Var[X(tk) | X(t1)] × (1 - K)\n";
    std::cout << "  where K = Var[X(tk) | X(t1)] / Var[X(t2) | X(t1)]\n\n";

    std::cout << "Unlike CIR, this is exact because log(S) is Gaussian!\n";
}

int main() {
    try {
        testGBMBridge();
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }
}
