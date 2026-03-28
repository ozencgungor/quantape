/**
 * @file test_asymmetric_bridge.cpp
 * @brief Test bridge behavior with closely spaced points
 *
 * What happens when we insert tk very close to t0 or t1?
 * E.g., t0=0, t1=0.25, tk=7/365.25 (1 week into 3 months)
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
    if (dt < 1e-10)
        return 0.0; // Avoid numerical issues

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

double sampleCIRBridge_Exact(double r1, double r2, double t1, double tk, double t2,
                             const CIRParams& params) {
    double dt1 = tk - t1;
    double dt2 = t2 - tk;

    if (dt1 < 1e-8) {
        // tk extremely close to t1, just return r1
        return r1;
    }
    if (dt2 < 1e-8) {
        // tk extremely close to t2, just return r2
        return r2;
    }

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

double bridge_AdaptiveVariance(double r1, double r2, double t1, double tk, double t2,
                               const CIRParams& params) {
    double dt1 = tk - t1;
    double dt2 = t2 - tk;
    double dt_total = t2 - t1;

    if (dt1 < 1e-8)
        return r1;
    if (dt2 < 1e-8)
        return r2;

    double alpha = dt1 / dt_total;

    double exp_k1 = std::exp(-params.kappa * dt1);
    double exp_k2 = std::exp(-params.kappa * dt2);
    double exp_total = std::exp(-params.kappa * dt_total);

    // Match conditional mean
    double w1 = (1.0 - exp_k2) / (1.0 - exp_total);
    double w2 = (exp_k1 - exp_total) / (1.0 - exp_total);
    double target_mean = params.theta * (1.0 - w1 - w2) + w1 * r1 + w2 * r2;

    CIRParams modified = params;
    double uncond_mean = params.theta + (r1 - params.theta) * exp_k1;
    modified.theta = params.theta + (target_mean - uncond_mean);

    // Adaptive variance reduction based on position in interval
    // When tk is close to t1 (alpha→0), little constraint from r2
    // When tk is close to t2 (alpha→1), little constraint from r1
    // Maximum constraint at middle (alpha=0.5)

    // Variance reduction is maximum at midpoint, minimum at endpoints
    double asymmetry_factor = 4.0 * alpha * (1.0 - alpha); // Quadratic, peaks at 0.5
    double base_reduction = 0.6;                           // Maximum reduction at midpoint
    double endpoint_reduction = 0.95;                      // Almost no reduction at endpoints

    double variance_factor =
        endpoint_reduction + (base_reduction - endpoint_reduction) * asymmetry_factor;

    modified.sigma = params.sigma * std::sqrt(variance_factor);

    return sampleCIR_Andersen(r1, dt1, modified);
}

double bridge_SimpleVariance(double r1, double r2, double t1, double tk, double t2,
                             const CIRParams& params, double fixed_factor = 0.6) {
    double dt1 = tk - t1;
    double dt_total = t2 - t1;

    if (dt1 < 1e-8)
        return r1;
    if ((t2 - tk) < 1e-8)
        return r2;

    double exp_k1 = std::exp(-params.kappa * dt1);
    double exp_k2 = std::exp(-params.kappa * (t2 - tk));
    double exp_total = std::exp(-params.kappa * dt_total);

    double w1 = (1.0 - exp_k2) / (1.0 - exp_total);
    double w2 = (exp_k1 - exp_total) / (1.0 - exp_total);
    double target_mean = params.theta * (1.0 - w1 - w2) + w1 * r1 + w2 * r2;

    CIRParams modified = params;
    double uncond_mean = params.theta + (r1 - params.theta) * exp_k1;
    modified.theta = params.theta + (target_mean - uncond_mean);
    modified.sigma = params.sigma * std::sqrt(fixed_factor);

    return sampleCIR_Andersen(r1, dt1, modified);
}

void testAsymmetricBridge() {
    std::cout << "=== Asymmetric Bridge Test ===\n\n";

    CIRParams params(0.5, 0.04, 0.1);

    // Test scenario: insert very close to left endpoint
    double r0 = 0.03;
    double r1 = 0.05;
    double t0 = 0.0;
    double t1 = 0.25; // 3 months

    std::vector<double> insertion_times = {
        7.0 / 365.25,  // 1 week (very close to t0)
        14.0 / 365.25, // 2 weeks
        30.0 / 365.25, // 1 month
        60.0 / 365.25, // 2 months
        0.125,         // Middle point
        0.20,          // Close to t1
        0.24           // Very close to t1
    };

    const int N_SAMPLES = 50000;

    std::cout << "Scenario: r(t0=" << t0 << ") = " << r0 << ", r(t1=" << t1 << ") = " << r1
              << "\n\n";

    std::cout << std::fixed << std::setprecision(6);

    for (double tk : insertion_times) {
        double alpha = tk / t1; // Position in interval
        double dt1 = tk - t0;
        double dt2 = t1 - tk;

        std::cout << "=========================================\n";
        std::cout << "Inserting at tk = " << tk << " years";
        if (tk < 0.1) {
            std::cout << " (" << tk * 365.25 << " days)";
        }
        std::cout << "\n";
        std::cout << "Position: " << alpha * 100 << "% through interval\n";
        std::cout << "dt1 = " << dt1 << ", dt2 = " << dt2 << "\n";
        std::cout << "Asymmetry ratio: dt1/dt2 = " << dt1 / dt2 << "\n\n";

        // Generate samples
        std::vector<double> samples_exact;
        std::vector<double> samples_simple;
        std::vector<double> samples_adaptive;
        std::vector<double> samples_unconditional;

        samples_exact.reserve(N_SAMPLES);
        samples_simple.reserve(N_SAMPLES);
        samples_adaptive.reserve(N_SAMPLES);
        samples_unconditional.reserve(N_SAMPLES);

        for (int i = 0; i < N_SAMPLES; i++) {
            rng.seed(1000 + i);
            samples_exact.push_back(sampleCIRBridge_Exact(r0, r1, t0, tk, t1, params));

            rng.seed(1000 + i);
            samples_simple.push_back(bridge_SimpleVariance(r0, r1, t0, tk, t1, params, 0.6));

            rng.seed(1000 + i);
            samples_adaptive.push_back(bridge_AdaptiveVariance(r0, r1, t0, tk, t1, params));

            rng.seed(1000 + i);
            samples_unconditional.push_back(sampleCIR_Andersen(r0, dt1, params));
        }

        auto compute_stats = [](const std::vector<double>& data) {
            double mean = std::accumulate(data.begin(), data.end(), 0.0) / data.size();
            double variance = 0.0;
            for (double x : data)
                variance += (x - mean) * (x - mean);
            variance /= data.size();
            return std::make_pair(mean, variance);
        };

        auto [mean_exact, var_exact] = compute_stats(samples_exact);
        auto [mean_simple, var_simple] = compute_stats(samples_simple);
        auto [mean_adaptive, var_adaptive] = compute_stats(samples_adaptive);
        auto [mean_uncond, var_uncond] = compute_stats(samples_unconditional);

        std::cout << "Results:\n";
        std::cout << "Method                    Mean        Variance    Mean Err    Var Err\n";
        std::cout << "Exact (Rejection):        " << std::setw(10) << mean_exact << "  "
                  << std::setw(10) << var_exact << "  " << std::setw(10) << 0.0 << "%  "
                  << std::setw(10) << 0.0 << "%\n";
        std::cout << "Simple (factor=0.6):      " << std::setw(10) << mean_simple << "  "
                  << std::setw(10) << var_simple << "  " << std::setw(10)
                  << (mean_simple - mean_exact) / mean_exact * 100 << "%  " << std::setw(10)
                  << (var_simple - var_exact) / var_exact * 100 << "%\n";
        std::cout << "Adaptive variance:        " << std::setw(10) << mean_adaptive << "  "
                  << std::setw(10) << var_adaptive << "  " << std::setw(10)
                  << (mean_adaptive - mean_exact) / mean_exact * 100 << "%  " << std::setw(10)
                  << (var_adaptive - var_exact) / var_exact * 100 << "%\n";
        std::cout << "Unconditional:            " << std::setw(10) << mean_uncond << "  "
                  << std::setw(10) << var_uncond << "  " << std::setw(10)
                  << (mean_uncond - mean_exact) / mean_exact * 100 << "%  " << std::setw(10)
                  << (var_uncond - var_exact) / var_exact * 100 << "%\n";

        // Compute actual variance reduction
        double variance_reduction_exact = 1.0 - (var_exact / var_uncond);
        std::cout << "\nVariance reduction (bridge vs unconditional):\n";
        std::cout << "  Exact bridge:   " << variance_reduction_exact * 100 << "%\n";
        std::cout << "  Theory: stronger constraint at midpoint, weaker near endpoints\n\n";
    }

    std::cout << "=========================================\n";
    std::cout << "\n=== Summary ===\n\n";
    std::cout << "Key observations:\n";
    std::cout << "1. Near endpoints: Bridge constraint is weak\n";
    std::cout << "   - Variance reduction minimal\n";
    std::cout << "   - Fixed factor 0.6 over-constrains\n";
    std::cout << "   - Adaptive method should perform better\n\n";
    std::cout << "2. At midpoint: Bridge constraint is strongest\n";
    std::cout << "   - Maximum variance reduction\n";
    std::cout << "   - Fixed factor 0.6 works well\n\n";
    std::cout << "3. Asymmetry matters!\n";
    std::cout << "   - Variance reduction depends on position α = tk/t_total\n";
    std::cout << "   - Optimal factor: f(α) = endpoint + (midpoint - endpoint)·4α(1-α)\n\n";
}

int main() {
    try {
        testAsymmetricBridge();
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }
}
