/**
 * @file test_improved_bridges.cpp
 * @brief Test improved approximate CIR bridge methods
 *
 * Compares various approximate bridge methods against the exact rejection sampling.
 */

#include <iostream>
#include <iomanip>
#include <vector>
#include <random>
#include <cmath>
#include <algorithm>
#include <numeric>

struct CIRParams {
    double kappa;
    double theta;
    double sigma;
    CIRParams(double k, double t, double s) : kappa(k), theta(t), sigma(s) {}
};

std::mt19937_64 rng(42);
std::normal_distribution<double> normal_dist(0.0, 1.0);
std::uniform_real_distribution<double> uniform_dist(0.0, 1.0);

double randn() { return normal_dist(rng); }
double uniform_random() { return uniform_dist(rng); }

// Andersen QE scheme
double sampleCIR_Andersen(double r_t, double dt, const CIRParams& params) {
    double c = params.sigma * params.sigma / (4.0 * params.kappa);
    double exp_kt = std::exp(-params.kappa * dt);
    double m = params.theta + (r_t - params.theta) * exp_kt;
    double s2 = r_t * c * exp_kt * (1.0 - exp_kt) +
                params.theta * c * (1.0 - exp_kt) * (1.0 - exp_kt);
    double psi = s2 / (m * m);

    if (psi <= 1.5) {
        double b2 = 2.0 / psi - 1.0 + std::sqrt(2.0 / psi) * std::sqrt(2.0 / psi - 1.0);
        double a = m / (1.0 + b2);
        double b = std::sqrt(b2);
        double Z = randn();
        return a * (b + Z) * (b + Z);
    } else {
        double p = (psi - 1.0) / (psi + 1.0);
        double beta = (1.0 - p) / m;
        double U = uniform_random();
        return (U <= p) ? 0.0 : std::log((1.0 - p) / (1.0 - U)) / beta;
    }
}

double evaluateDensity_Andersen(double r_start, double r_end, double dt, const CIRParams& params) {
    double exp_kt = std::exp(-params.kappa * dt);
    double c = params.sigma * params.sigma / (4.0 * params.kappa);
    double m = params.theta + (r_start - params.theta) * exp_kt;
    double s2 = r_start * c * exp_kt * (1.0 - exp_kt) +
                params.theta * c * (1.0 - exp_kt) * (1.0 - exp_kt);
    double psi = s2 / (m * m);

    if (psi <= 1.5) {
        double b2 = 2.0 / psi - 1.0 + std::sqrt(2.0 / psi) * std::sqrt(2.0 / psi - 1.0);
        double a = m / (1.0 + b2);
        if (r_end <= 0.0) return 0.0;
        double sqrt_r = std::sqrt(r_end);
        double sqrt_a = std::sqrt(a);
        double b = std::sqrt(b2);
        double z = sqrt_r / sqrt_a - b;
        double jacobian = 1.0 / (2.0 * sqrt_a * sqrt_r);
        return std::exp(-0.5 * z * z) / std::sqrt(2.0 * M_PI) * jacobian;
    } else {
        double p = (psi - 1.0) / (psi + 1.0);
        double beta = (1.0 - p) / m;
        return (r_end == 0.0) ? p : (1.0 - p) * beta * std::exp(-beta * r_end);
    }
}

// Exact bridge via rejection sampling
double sampleCIRBridge_Exact(double r1, double r2, double t1, double tk, double t2,
                             const CIRParams& params) {
    double dt1 = tk - t1;
    double dt2 = t2 - tk;

    double m_uncond = params.theta + (r1 - params.theta) * std::exp(-params.kappa * dt1);
    double M = evaluateDensity_Andersen(m_uncond, r2, dt2, params) * 1.5;

    for (int attempt = 0; attempt < 100000; attempt++) {
        double y = sampleCIR_Andersen(r1, dt1, params);
        double accept_prob = evaluateDensity_Andersen(y, r2, dt2, params) / M;
        if (uniform_random() < accept_prob) return y;
    }
    return sampleCIR_Andersen(r1, dt1, params);
}

/**
 * @brief Method 1: Original modified bridge (drift correction)
 */
double bridge_Method1_SimpleDrift(double r1, double r2, double t1, double tk, double t2,
                                  const CIRParams& params) {
    double dt1 = tk - t1;
    double dt_total = t2 - t1;
    double alpha = dt1 / dt_total;

    CIRParams modified = params;
    double exp_total = std::exp(-params.kappa * dt_total);
    double expected_r2 = params.theta + (r1 - params.theta) * exp_total;
    double correction = r2 - expected_r2;

    modified.theta = params.theta + (1.0 - alpha) * correction;
    return sampleCIR_Andersen(r1, dt1, modified);
}

/**
 * @brief Method 2: Match conditional mean exactly
 *
 * Compute E[r(tk) | r(t1), r(t2)] and adjust theta to match it
 */
double bridge_Method2_MatchMean(double r1, double r2, double t1, double tk, double t2,
                                const CIRParams& params) {
    double dt1 = tk - t1;
    double dt2 = t2 - tk;
    double dt_total = t2 - t1;

    double exp_k1 = std::exp(-params.kappa * dt1);
    double exp_k2 = std::exp(-params.kappa * dt2);
    double exp_total = std::exp(-params.kappa * dt_total);

    // Theoretical conditional mean: E[r(tk) | r(t1), r(t2)]
    // Using linear interpolation weighted by the conditional expectations
    double w1 = (1.0 - exp_k2) / (1.0 - exp_total);
    double w2 = (exp_k1 - exp_total) / (1.0 - exp_total);

    double target_mean = params.theta * (1.0 - w1 - w2) + w1 * r1 + w2 * r2;

    // Adjust theta so unconditional mean matches target
    double uncond_mean = params.theta + (r1 - params.theta) * exp_k1;

    CIRParams modified = params;
    modified.theta = params.theta + (target_mean - uncond_mean);

    return sampleCIR_Andersen(r1, dt1, modified);
}

/**
 * @brief Method 3: Metropolis-Hastings refinement
 *
 * Start with simple bridge, then do MH steps to correct distribution
 */
double bridge_Method3_MHRefinement(double r1, double r2, double t1, double tk, double t2,
                                   const CIRParams& params, int n_steps = 5) {
    double dt1 = tk - t1;
    double dt2 = t2 - tk;

    // Start with simple bridge
    double y = bridge_Method1_SimpleDrift(r1, r2, t1, tk, t2, params);

    // Target density (unnormalized)
    auto log_target = [&](double x) {
        if (x <= 0) return -std::numeric_limits<double>::infinity();
        return std::log(evaluateDensity_Andersen(r1, x, dt1, params)) +
               std::log(evaluateDensity_Andersen(x, r2, dt2, params));
    };

    double log_target_current = log_target(y);

    // Proposal std based on conditional variance
    double c = params.sigma * params.sigma / (4.0 * params.kappa);
    double exp_k1 = std::exp(-params.kappa * dt1);
    double var_uncond = r1 * c * exp_k1 * (1.0 - exp_k1) +
                        params.theta * c * (1.0 - exp_k1) * (1.0 - exp_k1);
    double proposal_std = 0.3 * std::sqrt(var_uncond);

    // MH steps
    for (int i = 0; i < n_steps; i++) {
        double y_proposed = y + proposal_std * randn();
        if (y_proposed <= 0) continue;

        double log_target_proposed = log_target(y_proposed);
        double log_accept = log_target_proposed - log_target_current;

        if (std::log(uniform_random()) < log_accept) {
            y = y_proposed;
            log_target_current = log_target_proposed;
        }
    }

    return y;
}

/**
 * @brief Method 4: Variance correction
 *
 * Adjust both theta and sigma to match mean and variance
 */
double bridge_Method4_VarianceCorrection(double r1, double r2, double t1, double tk, double t2,
                                         const CIRParams& params) {
    double dt1 = tk - t1;
    double dt2 = t2 - tk;
    double dt_total = t2 - t1;

    double exp_k1 = std::exp(-params.kappa * dt1);
    double exp_k2 = std::exp(-params.kappa * dt2);
    double exp_total = std::exp(-params.kappa * dt_total);

    // Target mean (same as Method 2)
    double w1 = (1.0 - exp_k2) / (1.0 - exp_total);
    double w2 = (exp_k1 - exp_total) / (1.0 - exp_total);
    double target_mean = params.theta * (1.0 - w1 - w2) + w1 * r1 + w2 * r2;

    // Approximate target variance (heuristic)
    // Bridge should have lower variance than unconditional
    double c = params.sigma * params.sigma / (4.0 * params.kappa);
    double var_uncond = r1 * c * exp_k1 * (1.0 - exp_k1) +
                        params.theta * c * (1.0 - exp_k1) * (1.0 - exp_k1);

    // Bridge variance is typically 50-70% of unconditional
    double reduction_factor = 0.6;
    double target_var = var_uncond * reduction_factor;

    // Adjust sigma to match target variance while keeping mean correct
    CIRParams modified = params;

    // First adjust theta for mean
    double uncond_mean = params.theta + (r1 - params.theta) * exp_k1;
    modified.theta = params.theta + (target_mean - uncond_mean);

    // Then adjust sigma for variance (approximately)
    double sigma_scale = std::sqrt(reduction_factor);
    modified.sigma = params.sigma * sigma_scale;

    return sampleCIR_Andersen(r1, dt1, modified);
}

/**
 * @brief Method 5: Two-step bridge
 *
 * Use Brownian bridge on transformed space
 */
double bridge_Method5_TransformedBridge(double r1, double r2, double t1, double tk, double t2,
                                        const CIRParams& params) {
    double dt1 = tk - t1;
    double dt_total = t2 - t1;
    double alpha = dt1 / dt_total;

    // Transform to sqrt(r) space (Lamperti transform)
    double sqrt_r1 = std::sqrt(r1);
    double sqrt_r2 = std::sqrt(r2);

    // CIR in sqrt space: d(sqrt(r)) has approximately constant diffusion
    // Use linear interpolation in sqrt space with added noise

    // Deterministic bridge in sqrt space
    double sqrt_bridge_det = sqrt_r1 + alpha * (sqrt_r2 - sqrt_r1);

    // Add noise scaled by bridge variance
    double c = params.sigma * params.sigma / (4.0 * params.kappa);
    double exp_k1 = std::exp(-params.kappa * dt1);
    double var_uncond = r1 * c * exp_k1 * (1.0 - exp_k1) +
                        params.theta * c * (1.0 - exp_k1) * (1.0 - exp_k1);

    // Variance in sqrt space (approximate)
    double var_sqrt = var_uncond / (4.0 * r1);
    double bridge_var = var_sqrt * alpha * (1.0 - alpha);

    double sqrt_bridge = sqrt_bridge_det + std::sqrt(bridge_var) * randn();

    return std::max(0.0, sqrt_bridge * sqrt_bridge);
}

void testImprovedBridges() {
    std::cout << "=== Testing Improved Approximate Bridge Methods ===\n\n";

    CIRParams params(0.5, 0.04, 0.1);

    double r1 = 0.03;
    double r2 = 0.05;
    double t1 = 0.0;
    double tk = 0.5;
    double t2 = 1.0;

    std::cout << "Scenario: r1=" << r1 << ", r2=" << r2
              << ", tk=" << tk << " (midpoint)\n\n";

    const int N_SAMPLES = 100000;
    std::cout << "Generating " << N_SAMPLES << " samples per method...\n\n";

    std::vector<double> samples_exact;
    std::vector<double> samples_method1;
    std::vector<double> samples_method2;
    std::vector<double> samples_method3;
    std::vector<double> samples_method4;
    std::vector<double> samples_method5;

    samples_exact.reserve(N_SAMPLES);
    samples_method1.reserve(N_SAMPLES);
    samples_method2.reserve(N_SAMPLES);
    samples_method3.reserve(N_SAMPLES);
    samples_method4.reserve(N_SAMPLES);
    samples_method5.reserve(N_SAMPLES);

    for (int i = 0; i < N_SAMPLES; i++) {
        rng.seed(1000 + i);
        samples_exact.push_back(sampleCIRBridge_Exact(r1, r2, t1, tk, t2, params));

        rng.seed(1000 + i);
        samples_method1.push_back(bridge_Method1_SimpleDrift(r1, r2, t1, tk, t2, params));

        rng.seed(1000 + i);
        samples_method2.push_back(bridge_Method2_MatchMean(r1, r2, t1, tk, t2, params));

        rng.seed(1000 + i);
        samples_method3.push_back(bridge_Method3_MHRefinement(r1, r2, t1, tk, t2, params, 3));

        rng.seed(1000 + i);
        samples_method4.push_back(bridge_Method4_VarianceCorrection(r1, r2, t1, tk, t2, params));

        rng.seed(1000 + i);
        samples_method5.push_back(bridge_Method5_TransformedBridge(r1, r2, t1, tk, t2, params));
    }

    // Compute moments
    auto compute_moments = [](const std::vector<double>& data) {
        double mean = std::accumulate(data.begin(), data.end(), 0.0) / data.size();
        double variance = 0.0;
        double skewness = 0.0;
        for (double x : data) {
            double diff = x - mean;
            variance += diff * diff;
            skewness += diff * diff * diff;
        }
        variance /= data.size();
        skewness /= data.size();
        skewness /= std::pow(variance, 1.5);
        return std::make_tuple(mean, variance, std::sqrt(variance), skewness);
    };

    auto [mean_exact, var_exact, std_exact, skew_exact] = compute_moments(samples_exact);
    auto [mean_m1, var_m1, std_m1, skew_m1] = compute_moments(samples_method1);
    auto [mean_m2, var_m2, std_m2, skew_m2] = compute_moments(samples_method2);
    auto [mean_m3, var_m3, std_m3, skew_m3] = compute_moments(samples_method3);
    auto [mean_m4, var_m4, std_m4, skew_m4] = compute_moments(samples_method4);
    auto [mean_m5, var_m5, std_m5, skew_m5] = compute_moments(samples_method5);

    std::cout << std::fixed << std::setprecision(6);
    std::cout << "=== Moment Comparison ===\n\n";
    std::cout << "Method                           Mean        Variance    Std Dev     Skewness\n";
    std::cout << "Exact (Rejection):               " << mean_exact << "   " << var_exact << "   " << std_exact << "   " << skew_exact << "\n";
    std::cout << "Method 1 (Simple Drift):         " << mean_m1 << "   " << var_m1 << "   " << std_m1 << "   " << skew_m1 << "\n";
    std::cout << "Method 2 (Match Mean):           " << mean_m2 << "   " << var_m2 << "   " << std_m2 << "   " << skew_m2 << "\n";
    std::cout << "Method 3 (MH Refinement):        " << mean_m3 << "   " << var_m3 << "   " << std_m3 << "   " << skew_m3 << "\n";
    std::cout << "Method 4 (Variance Correction):  " << mean_m4 << "   " << var_m4 << "   " << std_m4 << "   " << skew_m4 << "\n";
    std::cout << "Method 5 (Transformed Bridge):   " << mean_m5 << "   " << var_m5 << "   " << std_m5 << "   " << skew_m5 << "\n\n";

    std::cout << "=== Errors vs Exact (%) ===\n\n";
    std::cout << "Method                           Mean Err    Var Err     Skew Err\n";

    auto print_errors = [&](const char* name, double mean, double var, double skew) {
        std::cout << std::setw(32) << std::left << name
                  << std::setw(12) << std::right << (mean - mean_exact) / mean_exact * 100
                  << std::setw(12) << (var - var_exact) / var_exact * 100
                  << std::setw(12) << (skew - skew_exact) / skew_exact * 100 << "\n";
    };

    print_errors("Method 1 (Simple Drift):", mean_m1, var_m1, skew_m1);
    print_errors("Method 2 (Match Mean):", mean_m2, var_m2, skew_m2);
    print_errors("Method 3 (MH Refinement):", mean_m3, var_m3, skew_m3);
    print_errors("Method 4 (Variance Correction):", mean_m4, var_m4, skew_m4);
    print_errors("Method 5 (Transformed Bridge):", mean_m5, var_m5, skew_m5);

    std::cout << "\n=== Summary ===\n\n";
    std::cout << "Method 1: Original simple drift correction\n";
    std::cout << "Method 2: Match theoretical conditional mean\n";
    std::cout << "Method 3: MH refinement (3 steps) from Method 1\n";
    std::cout << "Method 4: Adjust both mean and variance\n";
    std::cout << "Method 5: Brownian bridge in sqrt(r) space\n\n";
}

int main() {
    try {
        testImprovedBridges();
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }
}
