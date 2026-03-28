/**
 * @file test_theoretical_bridge_variance.cpp
 * @brief Compute theoretical bridge variance and derive optimal correction
 *
 * Instead of using ad-hoc variance reduction factors, compute the actual
 * theoretical variance of p(r(tk) | r(t1), r(t2)) and derive the correction.
 */

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <random>
#include <vector>

struct CIRParams {
    double kappa;
    double theta;
    double sigma;
    CIRParams(double k, double t, double s) : kappa(k), theta(t), sigma(s) {}
};

std::mt19937_64 rng(42);
std::normal_distribution<double> normal_dist(0.0, 1.0);
std::uniform_real_distribution<double> uniform_dist(0.0, 1.0);

double randn() {
    return normal_dist(rng);
}
double uniform_random() {
    return uniform_dist(rng);
}

double sampleCIR_Andersen(double r_t, double dt, const CIRParams& params) {
    double c = params.sigma * params.sigma / (4.0 * params.kappa);
    double exp_kt = std::exp(-params.kappa * dt);
    double m = params.theta + (r_t - params.theta) * exp_kt;
    double s2 =
        r_t * c * exp_kt * (1.0 - exp_kt) + params.theta * c * (1.0 - exp_kt) * (1.0 - exp_kt);
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
    double s2 =
        r_start * c * exp_kt * (1.0 - exp_kt) + params.theta * c * (1.0 - exp_kt) * (1.0 - exp_kt);
    double psi = s2 / (m * m);

    if (psi <= 1.5) {
        double b2 = 2.0 / psi - 1.0 + std::sqrt(2.0 / psi) * std::sqrt(2.0 / psi - 1.0);
        double a = m / (1.0 + b2);
        if (r_end <= 0.0)
            return 0.0;
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

double evaluateBridgeDensity(double r1, double y, double r2, double t1, double tk, double t2,
                             const CIRParams& params) {
    double dt1 = tk - t1;
    double dt2 = t2 - tk;
    return evaluateDensity_Andersen(r1, y, dt1, params) *
           evaluateDensity_Andersen(y, r2, dt2, params);
}

/**
 * @brief Compute theoretical moments of bridge distribution via numerical integration
 */
struct BridgeMoments {
    double mean;
    double variance;
    double second_moment;
};

BridgeMoments computeTheoreticalBridgeMoments(double r1, double r2, double t1, double tk, double t2,
                                              const CIRParams& params) {
    double r_max = 0.3;
    int n_points = 10000;
    double dr = r_max / n_points;

    // First compute normalization constant Z
    double Z = 0.0;
    for (int i = 0; i < n_points; i++) {
        double r = (i + 0.5) * dr;
        Z += evaluateBridgeDensity(r1, r, r2, t1, tk, t2, params) * dr;
    }

    // Compute first moment: E[r]
    double first_moment = 0.0;
    for (int i = 0; i < n_points; i++) {
        double r = (i + 0.5) * dr;
        double p = evaluateBridgeDensity(r1, r, r2, t1, tk, t2, params) / Z;
        first_moment += r * p * dr;
    }

    // Compute second moment: E[r^2]
    double second_moment = 0.0;
    for (int i = 0; i < n_points; i++) {
        double r = (i + 0.5) * dr;
        double p = evaluateBridgeDensity(r1, r, r2, t1, tk, t2, params) / Z;
        second_moment += r * r * p * dr;
    }

    BridgeMoments moments;
    moments.mean = first_moment;
    moments.second_moment = second_moment;
    moments.variance = second_moment - first_moment * first_moment;

    return moments;
}

/**
 * @brief Compute unconditional moments for comparison
 */
struct UnconditionalMoments {
    double mean;
    double variance;
};

UnconditionalMoments computeUnconditionalMoments(double r_t, double dt, const CIRParams& params) {
    double exp_kt = std::exp(-params.kappa * dt);
    double c = params.sigma * params.sigma / (4.0 * params.kappa);

    UnconditionalMoments moments;
    moments.mean = params.theta + (r_t - params.theta) * exp_kt;
    moments.variance =
        r_t * c * exp_kt * (1.0 - exp_kt) + params.theta * c * (1.0 - exp_kt) * (1.0 - exp_kt);

    return moments;
}

/**
 * @brief Optimal bridge with theoretically derived variance correction
 */
double bridge_TheoreticalVariance(double r1, double r2, double t1, double tk, double t2,
                                  const CIRParams& params) {
    double dt1 = tk - t1;

    // Compute theoretical bridge moments
    BridgeMoments bridge_moments = computeTheoreticalBridgeMoments(r1, r2, t1, tk, t2, params);

    // Compute unconditional moments
    UnconditionalMoments uncond_moments = computeUnconditionalMoments(r1, dt1, params);

    // Adjust parameters to match theoretical moments
    CIRParams modified = params;

    // 1. Adjust theta to match mean
    modified.theta = params.theta + (bridge_moments.mean - uncond_moments.mean);

    // 2. Adjust sigma to match variance
    // Variance scales as sigma^2, so:
    // var_target = var_uncond * (sigma_new / sigma_old)^2
    // sigma_new = sigma_old * sqrt(var_target / var_uncond)
    double variance_ratio = bridge_moments.variance / uncond_moments.variance;
    modified.sigma = params.sigma * std::sqrt(variance_ratio);

    return sampleCIR_Andersen(r1, dt1, modified);
}

/**
 * @brief Exact bridge via rejection
 */
double sampleCIRBridge_Exact(double r1, double r2, double t1, double tk, double t2,
                             const CIRParams& params) {
    double dt1 = tk - t1;
    double dt2 = t2 - tk;
    double m_uncond = params.theta + (r1 - params.theta) * std::exp(-params.kappa * dt1);
    double M = evaluateDensity_Andersen(m_uncond, r2, dt2, params) * 1.5;

    for (int attempt = 0; attempt < 100000; attempt++) {
        double y = sampleCIR_Andersen(r1, dt1, params);
        double accept_prob = evaluateDensity_Andersen(y, r2, dt2, params) / M;
        if (uniform_random() < accept_prob)
            return y;
    }
    return sampleCIR_Andersen(r1, dt1, params);
}

/**
 * @brief Ad-hoc variance correction (for comparison)
 */
double bridge_AdHocVariance(double r1, double r2, double t1, double tk, double t2,
                            const CIRParams& params, double adhoc_factor) {
    double dt1 = tk - t1;
    double dt_total = t2 - t1;
    double exp_k1 = std::exp(-params.kappa * dt1);
    double exp_k2 = std::exp(-params.kappa * (t2 - tk));
    double exp_total = std::exp(-params.kappa * dt_total);

    double w1 = (1.0 - exp_k2) / (1.0 - exp_total);
    double w2 = (exp_k1 - exp_total) / (1.0 - exp_total);
    double target_mean = params.theta * (1.0 - w1 - w2) + w1 * r1 + w2 * r2;

    CIRParams modified = params;
    double uncond_mean = params.theta + (r1 - params.theta) * exp_k1;
    modified.theta = params.theta + (target_mean - uncond_mean);
    modified.sigma = params.sigma * std::sqrt(adhoc_factor);

    return sampleCIR_Andersen(r1, dt1, modified);
}

void testTheoreticalVariance() {
    std::cout << "=== Theoretical Bridge Variance Analysis ===\n\n";

    CIRParams params(0.5, 0.04, 0.1);

    double r1 = 0.03;
    double r2 = 0.05;
    double t1 = 0.0;
    double tk = 0.5;
    double t2 = 1.0;

    std::cout << "Scenario: r1=" << r1 << ", r2=" << r2 << ", tk=" << tk << "\n\n";

    // Compute theoretical moments
    std::cout << "Computing theoretical moments (this may take a moment)...\n";
    BridgeMoments bridge_theory = computeTheoreticalBridgeMoments(r1, r2, t1, tk, t2, params);
    UnconditionalMoments uncond_theory = computeUnconditionalMoments(r1, tk - t1, params);

    std::cout << std::fixed << std::setprecision(6);

    std::cout << "\n=== Theoretical Analysis ===\n\n";
    std::cout << "Unconditional (no endpoint constraint):\n";
    std::cout << "  Mean:     " << uncond_theory.mean << "\n";
    std::cout << "  Variance: " << uncond_theory.variance << "\n\n";

    std::cout << "Bridge (constrained by endpoint r2=" << r2 << "):\n";
    std::cout << "  Mean:     " << bridge_theory.mean << "\n";
    std::cout << "  Variance: " << bridge_theory.variance << "\n\n";

    double variance_ratio = bridge_theory.variance / uncond_theory.variance;
    std::cout << "Variance Reduction:\n";
    std::cout << "  Bridge variance = " << variance_ratio * 100 << "% of unconditional\n";
    std::cout << "  Reduction factor: " << variance_ratio << "\n\n";

    std::cout << "Mean Shift:\n";
    std::cout << "  Bridge mean - Unconditional mean = "
              << (bridge_theory.mean - uncond_theory.mean) << "\n";
    std::cout << "  Relative shift: "
              << (bridge_theory.mean - uncond_theory.mean) / uncond_theory.mean * 100 << "%\n\n";

    // Now test different methods
    const int N_SAMPLES = 50000;
    std::cout << "Testing methods with " << N_SAMPLES << " samples...\n\n";

    std::vector<double> samples_exact;
    std::vector<double> samples_theoretical;
    std::vector<double> samples_adhoc_06;
    std::vector<double> samples_adhoc_05;
    std::vector<double> samples_adhoc_07;

    samples_exact.reserve(N_SAMPLES);
    samples_theoretical.reserve(N_SAMPLES);
    samples_adhoc_06.reserve(N_SAMPLES);
    samples_adhoc_05.reserve(N_SAMPLES);
    samples_adhoc_07.reserve(N_SAMPLES);

    for (int i = 0; i < N_SAMPLES; i++) {
        rng.seed(1000 + i);
        samples_exact.push_back(sampleCIRBridge_Exact(r1, r2, t1, tk, t2, params));

        rng.seed(1000 + i);
        samples_theoretical.push_back(bridge_TheoreticalVariance(r1, r2, t1, tk, t2, params));

        rng.seed(1000 + i);
        samples_adhoc_06.push_back(bridge_AdHocVariance(r1, r2, t1, tk, t2, params, 0.6));

        rng.seed(1000 + i);
        samples_adhoc_05.push_back(bridge_AdHocVariance(r1, r2, t1, tk, t2, params, 0.5));

        rng.seed(1000 + i);
        samples_adhoc_07.push_back(bridge_AdHocVariance(r1, r2, t1, tk, t2, params, 0.7));
    }

    auto compute_moments = [](const std::vector<double>& data) {
        double mean = std::accumulate(data.begin(), data.end(), 0.0) / data.size();
        double variance = 0.0;
        for (double x : data)
            variance += (x - mean) * (x - mean);
        variance /= data.size();
        return std::make_pair(mean, variance);
    };

    auto [mean_exact, var_exact] = compute_moments(samples_exact);
    auto [mean_theoretical, var_theoretical] = compute_moments(samples_theoretical);
    auto [mean_adhoc_06, var_adhoc_06] = compute_moments(samples_adhoc_06);
    auto [mean_adhoc_05, var_adhoc_05] = compute_moments(samples_adhoc_05);
    auto [mean_adhoc_07, var_adhoc_07] = compute_moments(samples_adhoc_07);

    std::cout << "=== Empirical Results ===\n\n";
    std::cout << "Method                          Mean        Variance    Mean Err    Var Err\n";
    std::cout << "Exact (Rejection):              " << mean_exact << "   " << var_exact << "   "
              << 0.0 << "%     " << 0.0 << "%\n";
    std::cout << "Theoretical Variance:           " << mean_theoretical << "   " << var_theoretical
              << "   " << (mean_theoretical - mean_exact) / mean_exact * 100 << "%     "
              << (var_theoretical - var_exact) / var_exact * 100 << "%\n";
    std::cout << "Ad-hoc (factor=0.6):            " << mean_adhoc_06 << "   " << var_adhoc_06
              << "   " << (mean_adhoc_06 - mean_exact) / mean_exact * 100 << "%     "
              << (var_adhoc_06 - var_exact) / var_exact * 100 << "%\n";
    std::cout << "Ad-hoc (factor=0.5):            " << mean_adhoc_05 << "   " << var_adhoc_05
              << "   " << (mean_adhoc_05 - mean_exact) / mean_exact * 100 << "%     "
              << (var_adhoc_05 - var_exact) / var_exact * 100 << "%\n";
    std::cout << "Ad-hoc (factor=0.7):            " << mean_adhoc_07 << "   " << var_adhoc_07
              << "   " << (mean_adhoc_07 - mean_exact) / mean_exact * 100 << "%     "
              << (var_adhoc_07 - var_exact) / var_exact * 100 << "%\n\n";

    std::cout << "=== Comparison to Theory ===\n\n";
    std::cout << "Theoretical bridge moments (numerical integration):\n";
    std::cout << "  Mean:     " << bridge_theory.mean << "\n";
    std::cout << "  Variance: " << bridge_theory.variance << "\n\n";

    std::cout << "Exact rejection sampling (empirical):\n";
    std::cout << "  Mean:     " << mean_exact << "\n";
    std::cout << "  Variance: " << var_exact << "\n";
    std::cout << "  Mean error:     "
              << (mean_exact - bridge_theory.mean) / bridge_theory.mean * 100 << "%\n";
    std::cout << "  Variance error: "
              << (var_exact - bridge_theory.variance) / bridge_theory.variance * 100 << "%\n\n";

    std::cout << "Theoretical variance correction (empirical):\n";
    std::cout << "  Mean:     " << mean_theoretical << "\n";
    std::cout << "  Variance: " << var_theoretical << "\n";
    std::cout << "  Mean error:     "
              << (mean_theoretical - bridge_theory.mean) / bridge_theory.mean * 100 << "%\n";
    std::cout << "  Variance error: "
              << (var_theoretical - bridge_theory.variance) / bridge_theory.variance * 100
              << "%\n\n";

    std::cout << "=== Conclusion ===\n\n";
    std::cout << "The theoretically derived variance ratio is " << variance_ratio << "\n";
    std::cout << "This should be used instead of ad-hoc factors like 0.6\n\n";

    std::cout << "Theoretical method vs Exact:\n";
    std::cout << "  Variance error: " << std::abs((var_theoretical - var_exact) / var_exact * 100)
              << "%\n";
    std::cout << "Ad-hoc factor=0.6 vs Exact:\n";
    std::cout << "  Variance error: " << std::abs((var_adhoc_06 - var_exact) / var_exact * 100)
              << "%\n\n";

    if (std::abs(variance_ratio - 0.6) < 0.05) {
        std::cout << "Note: For this scenario, the theoretical ratio ≈ 0.6, so ad-hoc\n";
        std::cout << "worked by luck. But it will fail for other scenarios!\n";
    }
}

int main() {
    try {
        testTheoreticalVariance();
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }
}
