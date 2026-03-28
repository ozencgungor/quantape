/**
 * @file test_nonlinear_corrections.cpp
 * @brief Test corrections for CIR bridge when r1 is far from theta
 *
 * The problem: When r1 << theta, linear approximations fail badly.
 * We test several correction strategies:
 *   1. Lamperti transform (work in Y = 2√r/σ space)
 *   2. Log-space approximation
 *   3. Mean reversion correction (adjust for nonlinear pull to theta)
 *   4. Hybrid approach (switch based on distance from theta)
 *   5. Effective parameter adjustment
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
};

std::mt19937_64 rng(42);
std::normal_distribution<double> normal_dist(0.0, 1.0);
std::uniform_real_distribution<double> uniform_dist(0.0, 1.0);

double randn() { return normal_dist(rng); }
double uniform_random() { return uniform_dist(rng); }

// --- Andersen QE scheme ---

double sampleCIR_Andersen(double r_t, double dt, const CIRParams& params) {
    if (params.kappa <= 1e-8) {
        return r_t + params.sigma * std::sqrt(std::max(0.0, r_t)) * randn() * std::sqrt(dt);
    }
    double c = params.sigma * params.sigma / (4.0 * params.kappa);
    double exp_kt = std::exp(-params.kappa * dt);
    double m = params.theta + (r_t - params.theta) * exp_kt;
    double s2 = r_t * c * exp_kt * (1.0 - exp_kt) +
                params.theta * c * (1.0 - exp_kt) * (1.0 - exp_kt);
    double psi = (s2 > 1e-12) ? s2 / (m * m) : 0.0;

    if (psi <= 1.5) {
        double b2 = (psi > 1e-12) ? 2.0 / psi - 1.0 + std::sqrt(2.0 / psi * (2.0/psi - 1.0)) : 1.0;
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
    if (dt < 1e-10) return 0.0;
    if (params.kappa <= 1e-8) return 0.0;
    double exp_kt = std::exp(-params.kappa * dt);
    double c = params.sigma * params.sigma / (4.0 * params.kappa);
    double m = params.theta + (r_start - params.theta) * exp_kt;
    double s2 = r_start * c * exp_kt * (1.0 - exp_kt) +
                params.theta * c * (1.0 - exp_kt) * (1.0 - exp_kt);
    double psi = (s2 > 1e-12) ? s2 / (m * m) : 0.0;

    if (psi <= 1.5) {
        double b2 = (psi > 1e-12) ? 2.0 / psi - 1.0 + std::sqrt(2.0 / psi * (2.0/psi - 1.0)) : 1.0;
        double a = m / (1.0 + b2);
        if (r_end <= 1e-12) return 0.0;
        double sqrt_r = std::sqrt(r_end);
        double sqrt_a = std::sqrt(a);
        double b = std::sqrt(b2);
        double z = sqrt_r / sqrt_a - b;
        double jacobian = 1.0 / (2.0 * sqrt_a * sqrt_r);
        return std::exp(-0.5 * z * z) / std::sqrt(2.0 * M_PI) * jacobian;
    } else {
        double p = (psi - 1.0) / (psi + 1.0);
        double beta = (1.0 - p) / m;
        return (r_end < 1e-12) ? p : (1.0 - p) * beta * std::exp(-beta * r_end);
    }
}

double sampleCIRBridge_Exact(double r1, double r2, double t1, double tk, double t2, const CIRParams& params) {
    double dt1 = tk - t1;
    double dt2 = t2 - tk;
    if (dt1 < 1e-8) return r1;
    if (dt2 < 1e-8) return r2;

    double m_uncond = params.theta + (r1 - params.theta) * std::exp(-params.kappa * dt1);
    double M = evaluateDensity_Andersen(m_uncond, r2, dt2, params) * 1.5;
    if (M <= 1e-12) return m_uncond;

    for (int attempt = 0; attempt < 100000; attempt++) {
        double y = sampleCIR_Andersen(r1, dt1, params);
        double accept_prob = evaluateDensity_Andersen(y, r2, dt2, params) / M;
        if (uniform_random() < accept_prob) return y;
    }
    return sampleCIR_Andersen(r1, dt1, params);
}

double get_cir_variance(double x0, double dt, const CIRParams& params) {
    const double& k = params.kappa;
    if (k <= 1e-8) return 0.0;
    const double& th = params.theta;
    const double& s = params.sigma;
    const double s2 = s * s;
    const double e_kt = std::exp(-k * dt);

    return (x0 * s2 / k) * (e_kt - e_kt * e_kt) + (th * s2 / (2.0 * k)) * std::pow(1.0 - e_kt, 2);
}

// ============================================================================
// Method 1: Lamperti Transform (work in Y = 2√r/σ space)
// ============================================================================
/**
 * Lamperti transform: Y = 2√r/σ
 * Then: dY = (κθ/σ - κσ/4 - κY/2)dt + dW
 *
 * This has CONSTANT diffusion = 1, so variance is linear in time!
 * Easier to approximate.
 */
double bridge_Lamperti(double r1, double r2, double t1, double tk, double t2, const CIRParams& params) {
    double dt1 = tk - t1;
    double dt2 = t2 - tk;
    if (dt1 < 1e-8) return r1;
    if (dt2 < 1e-8) return r2;

    // Transform to Y space
    double Y1 = 2.0 * std::sqrt(std::max(0.0, r1)) / params.sigma;
    double Y2 = 2.0 * std::sqrt(std::max(0.0, r2)) / params.sigma;

    // In Y space: dY = (a - b*Y)dt + dW where a = κθ/σ, b = κ/2
    double a = params.kappa * params.theta / params.sigma;
    double b = params.kappa / 2.0;

    // OU bridge in Y space
    double exp_b1 = std::exp(-b * dt1);
    double exp_b2 = std::exp(-b * dt2);
    double exp_total = std::exp(-b * (dt1 + dt2));

    // Mean
    double E_Y1k = Y1 * exp_b1 + (a / b) * (1.0 - exp_b1);
    double E_Y12 = Y1 * exp_total + (a / b) * (1.0 - exp_total);

    // Variance (constant diffusion = 1)
    double var_1k = (1.0 / (2.0 * b)) * (1.0 - std::exp(-2.0 * b * dt1));
    double var_12 = (1.0 / (2.0 * b)) * (1.0 - std::exp(-2.0 * b * (dt1 + dt2)));

    double cov_k2 = var_1k * exp_b2;
    double K = (var_12 > 1e-12) ? cov_k2 / var_12 : 0.0;

    // Bridge moments in Y space
    double Y_bridge_mean = E_Y1k + K * (Y2 - E_Y12);
    double Y_bridge_var = var_1k * (1.0 - K);

    // Sample in Y space
    double Y_k = Y_bridge_mean + std::sqrt(std::max(0.0, Y_bridge_var)) * randn();

    // Transform back: r = (σY/2)²
    double r_k = (params.sigma * Y_k / 2.0) * (params.sigma * Y_k / 2.0);

    return std::max(0.0, r_k);
}

// ============================================================================
// Method 2: Mean Reversion Correction
// ============================================================================
/**
 * Idea: When r1 << theta, the mean reversion pulls strongly toward theta.
 * The effective mean should weight theta more heavily.
 *
 * Correction: Adjust the bridge mean based on "distance from equilibrium"
 */
double bridge_MeanReversionCorrection(double r1, double r2, double t1, double tk, double t2, const CIRParams& params) {
    double dt1 = tk - t1;
    double dt2 = t2 - tk;
    if (dt1 < 1e-8) return r1;
    if (dt2 < 1e-8) return r2;

    const double& k = params.kappa;
    const double& th = params.theta;

    double exp_k_dt1 = std::exp(-k * dt1);
    double exp_k_dt2 = std::exp(-k * dt2);
    double exp_total = std::exp(-k * (dt1 + dt2));

    // Standard Kalman bridge mean
    double E_1k = r1 * exp_k_dt1 + th * (1.0 - exp_k_dt1);
    double E_12 = r1 * exp_total + th * (1.0 - exp_total);
    double var_1k = get_cir_variance(r1, dt1, params);
    double var_12 = get_cir_variance(r1, dt1 + dt2, params);
    double cov_k2 = var_1k * exp_k_dt2;

    double mean_x_bridge = E_1k;
    if (var_12 > 1e-12) {
        mean_x_bridge += (r2 - E_12) * cov_k2 / var_12;
    }

    // CORRECTION: When r1 far from theta, add pull toward theta
    double distance_from_eq = std::abs(r1 - th) / std::max(th, 1e-8);

    // If r1 << theta, we should pull the mean MORE toward theta
    // Correction factor: increases with distance from equilibrium
    double correction_factor = std::tanh(distance_from_eq);  // 0 when at equilibrium, →1 when far

    // Adjust mean toward theta
    double corrected_mean = mean_x_bridge + correction_factor * 0.3 * (th - mean_x_bridge);

    // Variance correction
    double var_k2_approx = get_cir_variance(corrected_mean, dt2, params);
    double var_x_bridge = (var_1k > 1e-12 && var_k2_approx > 1e-12)
        ? (var_1k * var_k2_approx) / (var_1k + var_k2_approx)
        : 0.0;

    CIRParams modified_params = params;
    if (std::abs(1.0 - exp_k_dt1) > 1e-8) {
        modified_params.theta = (corrected_mean - r1 * exp_k_dt1) / (1.0 - exp_k_dt1);
    }

    double e_2kdt1 = exp_k_dt1 * exp_k_dt1;
    double var_factor = (k > 1e-8) ? ((r1 / k) * (exp_k_dt1 - e_2kdt1) +
                                      (modified_params.theta / (2.0 * k)) * std::pow(1.0 - exp_k_dt1, 2)) : 0.0;

    if (var_factor > 1e-12) {
        double s2_target = var_x_bridge / var_factor;
        modified_params.sigma = std::sqrt(std::max(0.0, s2_target));
    }

    return sampleCIR_Andersen(r1, dt1, modified_params);
}

// ============================================================================
// Method 3: Effective Kappa Adjustment
// ============================================================================
/**
 * Idea: When r1 << theta, the effective mean reversion speed is stronger.
 * Adjust kappa to reflect the nonlinear pull.
 *
 * For CIR: When r is small, mean reversion dominates (theta term is large relative to r)
 */
double bridge_EffectiveKappa(double r1, double r2, double t1, double tk, double t2, const CIRParams& params) {
    double dt1 = tk - t1;
    double dt2 = t2 - tk;
    if (dt1 < 1e-8) return r1;
    if (dt2 < 1e-8) return r2;

    const double& k = params.kappa;
    const double& th = params.theta;

    // Effective kappa increases when far from equilibrium
    // This captures the stronger pull toward theta
    double distance_ratio = (th > 1e-8) ? r1 / th : 1.0;

    // If r1 << theta (distance_ratio << 1), increase kappa
    // If r1 >> theta (distance_ratio >> 1), decrease kappa (less pull down)
    double kappa_multiplier = 1.0;
    if (distance_ratio < 0.5) {
        // r1 much less than theta - increase mean reversion
        kappa_multiplier = 1.0 + 0.5 * (1.0 - 2.0 * distance_ratio);
    } else if (distance_ratio > 2.0) {
        // r1 much greater than theta - decrease mean reversion
        kappa_multiplier = 1.0 / (1.0 + 0.3 * (distance_ratio - 2.0));
    }

    double k_eff = k * kappa_multiplier;

    // Use effective kappa for bridge computation
    double exp_k_dt1 = std::exp(-k_eff * dt1);
    double exp_k_dt2 = std::exp(-k_eff * dt2);
    double exp_total = std::exp(-k_eff * (dt1 + dt2));

    double E_1k = r1 * exp_k_dt1 + th * (1.0 - exp_k_dt1);
    double E_12 = r1 * exp_total + th * (1.0 - exp_total);

    // Use ORIGINAL kappa for variance
    double var_1k = get_cir_variance(r1, dt1, params);
    double var_12 = get_cir_variance(r1, dt1 + dt2, params);
    double cov_k2 = var_1k * std::exp(-k * dt2);  // Original kappa

    double mean_x_bridge = E_1k;
    if (var_12 > 1e-12) {
        mean_x_bridge += (r2 - E_12) * cov_k2 / var_12;
    }

    double var_k2_approx = get_cir_variance(mean_x_bridge, dt2, params);
    double var_x_bridge = (var_1k > 1e-12 && var_k2_approx > 1e-12)
        ? (var_1k * var_k2_approx) / (var_1k + var_k2_approx)
        : 0.0;

    CIRParams modified_params = params;
    if (std::abs(1.0 - exp_k_dt1) > 1e-8) {
        modified_params.theta = (mean_x_bridge - r1 * exp_k_dt1) / (1.0 - exp_k_dt1);
    }

    double e_2kdt1 = exp_k_dt1 * exp_k_dt1;
    double var_factor = (k > 1e-8) ? ((r1 / k) * (exp_k_dt1 - e_2kdt1) +
                                      (modified_params.theta / (2.0 * k)) * std::pow(1.0 - exp_k_dt1, 2)) : 0.0;

    if (var_factor > 1e-12) {
        double s2_target = var_x_bridge / var_factor;
        modified_params.sigma = std::sqrt(std::max(0.0, s2_target));
    }

    return sampleCIR_Andersen(r1, dt1, modified_params);
}

// ============================================================================
// Method 4: Two-Step Bridge (split interval)
// ============================================================================
/**
 * Idea: When r1 << theta, first simulate to a closer point, then bridge.
 * This reduces the distance from equilibrium.
 */
double bridge_TwoStep(double r1, double r2, double t1, double tk, double t2, const CIRParams& params) {
    double dt1 = tk - t1;
    double dt2 = t2 - tk;
    if (dt1 < 1e-8) return r1;
    if (dt2 < 1e-8) return r2;

    // If r1 is far from theta, use two-step approach
    double distance_ratio = (params.theta > 1e-8) ? std::abs(r1 - params.theta) / params.theta : 0.0;

    if (distance_ratio > 0.5) {
        // Split the interval [t1, tk] at tmid
        double tmid = t1 + dt1 / 2.0;

        // First: bridge from (r1, t1) to (r2, t2) at tmid
        double r_mid = sampleCIRBridge_Exact(r1, r2, t1, tmid, t2, params);

        // Second: bridge from (r_mid, tmid) to (r2, t2) at tk
        return sampleCIRBridge_Exact(r_mid, r2, tmid, tk, t2, params);
    } else {
        // Normal bridge
        return sampleCIRBridge_Exact(r1, r2, t1, tk, t2, params);
    }
}

// ============================================================================
// Baseline: v2_corrected from previous test
// ============================================================================
double bridge_v2_corrected(double r1, double r2, double t1, double tk, double t2, const CIRParams& params) {
    double dt1 = tk - t1;
    double dt2 = t2 - tk;
    if (dt1 < 1e-8) return r1;
    if (dt2 < 1e-8) return r2;

    const double& k = params.kappa;
    const double& th = params.theta;

    double exp_k_dt1 = std::exp(-k * dt1);
    double exp_k_dt2 = std::exp(-k * dt2);
    double exp_total = std::exp(-k * (dt1 + dt2));

    double E_1k = r1 * exp_k_dt1 + th * (1.0 - exp_k_dt1);
    double E_12 = r1 * exp_total + th * (1.0 - exp_total);
    double var_1k = get_cir_variance(r1, dt1, params);
    double var_12 = get_cir_variance(r1, dt1 + dt2, params);
    double cov_k2 = var_1k * exp_k_dt2;

    double mean_x_bridge = E_1k;
    if (var_12 > 1e-12) {
        mean_x_bridge += (r2 - E_12) * cov_k2 / var_12;
    }

    double var_k2_approx = get_cir_variance(mean_x_bridge, dt2, params);
    double var_x_bridge = (var_1k > 1e-12 && var_k2_approx > 1e-12)
        ? (var_1k * var_k2_approx) / (var_1k + var_k2_approx)
        : 0.0;

    CIRParams modified_params = params;
    if (std::abs(1.0 - exp_k_dt1) > 1e-8) {
        modified_params.theta = (mean_x_bridge - r1 * exp_k_dt1) / (1.0 - exp_k_dt1);
    }

    double e_2kdt1 = exp_k_dt1 * exp_k_dt1;
    double var_factor = (k > 1e-8) ? ((r1 / k) * (exp_k_dt1 - e_2kdt1) +
                                      (modified_params.theta / (2.0 * k)) * std::pow(1.0 - exp_k_dt1, 2)) : 0.0;

    if (var_factor > 1e-12) {
        double s2_target = var_x_bridge / var_factor;
        modified_params.sigma = std::sqrt(std::max(0.0, s2_target));
    }

    return sampleCIR_Andersen(r1, dt1, modified_params);
}

// ============================================================================
// Test function
// ============================================================================

void testNonlinearCorrections() {
    std::cout << "=== Testing Nonlinear Corrections for Low r1 ===\n\n";

    CIRParams params{0.1, 0.08, 0.2};  // Standard params

    // CRITICAL TEST CASE: r1 << theta
    double r1 = 0.01;  // Much less than theta = 0.08
    double r2 = 0.05;
    double t1 = 0.0;
    double tk = 0.5;
    double t2 = 1.0;

    std::cout << "Parameters:\n";
    std::cout << "  κ = " << params.kappa << ", θ = " << params.theta << ", σ = " << params.sigma << "\n";
    std::cout << "  r1 = " << r1 << " (ratio r1/θ = " << r1/params.theta << ")\n";
    std::cout << "  r2 = " << r2 << "\n";
    std::cout << "  Times: t1=" << t1 << ", tk=" << tk << ", t2=" << t2 << "\n\n";

    const int N_SAMPLES = 20000;

    // Exact samples
    std::cout << "Generating exact samples...\n";
    std::vector<double> samples_exact;
    samples_exact.reserve(N_SAMPLES);
    for (int i = 0; i < N_SAMPLES; i++) {
        rng.seed(1000 + i);
        samples_exact.push_back(sampleCIRBridge_Exact(r1, r2, t1, tk, t2, params));
    }

    auto compute_stats = [](const std::vector<double>& data) {
        double mean = std::accumulate(data.begin(), data.end(), 0.0) / data.size();
        double variance = 0.0;
        for (double x : data) variance += (x - mean) * (x - mean);
        variance /= data.size();
        return std::make_pair(mean, variance);
    };

    auto [mean_exact, var_exact] = compute_stats(samples_exact);

    std::cout << "Exact bridge statistics:\n";
    std::cout << "  Mean: " << mean_exact << "\n";
    std::cout << "  Variance: " << var_exact << "\n\n";

    // Test methods
    struct Method {
        std::string name;
        std::function<double(double, double, double, double, double, const CIRParams&)> func;
    };

    std::vector<Method> methods = {
        {"Baseline (v2_corrected)", bridge_v2_corrected},
        {"Lamperti Transform", bridge_Lamperti},
        {"Mean Reversion Correction", bridge_MeanReversionCorrection},
        {"Effective Kappa", bridge_EffectiveKappa},
        {"Two-Step Bridge", bridge_TwoStep}
    };

    std::cout << std::fixed << std::setprecision(4);
    std::cout << "Method                         Mean        Variance    Mean Err    Var Err\n";
    std::cout << "-----------------------------------------------------------------------------\n";

    for (const auto& method : methods) {
        std::vector<double> samples;
        samples.reserve(N_SAMPLES);

        for (int i = 0; i < N_SAMPLES; i++) {
            rng.seed(1000 + i);
            samples.push_back(method.func(r1, r2, t1, tk, t2, params));
        }

        auto [mean_approx, var_approx] = compute_stats(samples);

        double mean_err = (mean_exact > 1e-9) ? (mean_approx - mean_exact) / mean_exact * 100 : 0.0;
        double var_err = (var_exact > 1e-12) ? (var_approx - var_exact) / var_exact * 100 : 0.0;

        std::cout << std::setw(30) << std::left << method.name
                  << std::setw(12) << std::right << mean_approx
                  << std::setw(12) << var_approx
                  << std::setw(12) << mean_err << "%"
                  << std::setw(11) << var_err << "%\n";
    }

    std::cout << "\n=== Summary ===\n\n";
    std::cout << "Challenge: When r1 << θ, linearization fails.\n";
    std::cout << "Baseline (v2_corrected) had ~46% mean error in this regime.\n\n";
    std::cout << "Tested corrections:\n";
    std::cout << "  1. Lamperti Transform: Work in Y=2√r/σ space (constant diffusion)\n";
    std::cout << "  2. Mean Reversion: Adjust mean toward θ when far from equilibrium\n";
    std::cout << "  3. Effective Kappa: Increase κ when r1 << θ (stronger pull)\n";
    std::cout << "  4. Two-Step Bridge: Split interval to reduce distance from equilibrium\n\n";
}

int main() {
    try {
        testNonlinearCorrections();
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }
}
