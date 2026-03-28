/**
 * @file test_cir_bridge_distribution.cpp
 * @brief Test statistical properties of CIR bridge
 *
 * Tests whether the bridge samples from the correct conditional distribution
 * p(r(tk) | r(t1), r(t2)) by comparing empirical samples against theoretical density.
 */

#include <iostream>
#include <iomanip>
#include <vector>
#include <random>
#include <cmath>
#include <algorithm>
#include <numeric>
#include <map>

// CIR parameters structure
struct CIRParams {
    double kappa;
    double theta;
    double sigma;

    CIRParams(double k, double t, double s)
        : kappa(k), theta(t), sigma(s) {}
};

// Global RNG
std::mt19937_64 rng(42);
std::normal_distribution<double> normal_dist(0.0, 1.0);
std::uniform_real_distribution<double> uniform_dist(0.0, 1.0);

double randn() { return normal_dist(rng); }
double uniform_random() { return uniform_dist(rng); }

/**
 * @brief Andersen's QE scheme for CIR transition
 */
double sampleCIR_Andersen(double r_t, double dt, const CIRParams& params) {
    double kappa = params.kappa;
    double theta = params.theta;
    double sigma = params.sigma;

    double c = sigma * sigma / (4.0 * kappa);
    double exp_kt = std::exp(-kappa * dt);

    double m = theta + (r_t - theta) * exp_kt;
    double s2 = r_t * c * exp_kt * (1.0 - exp_kt) +
                theta * c * (1.0 - exp_kt) * (1.0 - exp_kt);

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
        if (U <= p) {
            return 0.0;
        } else {
            return std::log((1.0 - p) / (1.0 - U)) / beta;
        }
    }
}

/**
 * @brief Evaluate CIR transition density using Andersen approximation
 */
double evaluateDensity_Andersen(double r_start, double r_end,
                                double dt, const CIRParams& params) {
    double kappa = params.kappa;
    double theta = params.theta;
    double sigma = params.sigma;

    double exp_kt = std::exp(-kappa * dt);
    double c = sigma * sigma / (4.0 * kappa);

    double m = theta + (r_start - theta) * exp_kt;
    double s2 = r_start * c * exp_kt * (1.0 - exp_kt) +
                theta * c * (1.0 - exp_kt) * (1.0 - exp_kt);
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

        double gaussian_pdf = std::exp(-0.5 * z * z) / std::sqrt(2.0 * M_PI);

        return gaussian_pdf * jacobian;
    } else {
        double p = (psi - 1.0) / (psi + 1.0);
        double beta = (1.0 - p) / m;

        if (r_end == 0.0) {
            return p;
        }
        return (1.0 - p) * beta * std::exp(-beta * r_end);
    }
}

/**
 * @brief Evaluate bridge density: p(r(tk) | r(t1), r(t2))
 *
 * Uses product formula: p(y | r1, r2) ∝ p(y | r1) × p(r2 | y)
 */
double evaluateBridgeDensity(double r1, double y, double r2,
                             double t1, double tk, double t2,
                             const CIRParams& params) {
    double dt1 = tk - t1;
    double dt2 = t2 - tk;

    double p_forward = evaluateDensity_Andersen(r1, y, dt1, params);
    double p_backward = evaluateDensity_Andersen(y, r2, dt2, params);

    return p_forward * p_backward;
}

/**
 * @brief Sample CIR bridge using rejection sampling
 */
double sampleCIRBridge_AndersenRejection(double r1, double r2,
                                         double t1, double tk, double t2,
                                         const CIRParams& params) {
    double dt1 = tk - t1;
    double dt2 = t2 - tk;

    // Find approximate maximum
    double m_uncond = params.theta + (r1 - params.theta) * std::exp(-params.kappa * dt1);
    double m_bridge = 0.5 * (r1 + r2);

    double M = std::max(
        evaluateBridgeDensity(r1, m_uncond, r2, t1, tk, t2, params),
        evaluateBridgeDensity(r1, m_bridge, r2, t1, tk, t2, params)
    );
    M *= 1.5;

    const int MAX_ATTEMPTS = 100000;
    for (int attempt = 0; attempt < MAX_ATTEMPTS; attempt++) {
        double y_proposed = sampleCIR_Andersen(r1, dt1, params);
        double forward_density = evaluateDensity_Andersen(y_proposed, r2, dt2, params);
        double acceptance_prob = forward_density / M;

        if (uniform_random() < acceptance_prob) {
            return y_proposed;
        }
    }

    return sampleCIR_Andersen(r1, dt1, params);
}

/**
 * @brief Modified bridge with drift correction
 */
double sampleCIRBridge_AndersenModified(double r1, double r2,
                                        double t1, double tk, double t2,
                                        const CIRParams& params) {
    double dt1 = tk - t1;
    double dt_total = t2 - t1;
    double alpha = dt1 / dt_total;

    CIRParams modified_params = params;
    double exp_k_dt_total = std::exp(-params.kappa * dt_total);
    double expected_r2_uncond = params.theta + (r1 - params.theta) * exp_k_dt_total;
    double drift_correction = r2 - expected_r2_uncond;

    modified_params.theta = params.theta + (1.0 - alpha) * drift_correction;

    return sampleCIR_Andersen(r1, dt1, modified_params);
}

/**
 * @brief Compute numerical normalization constant for bridge density
 */
double computeNormalizationConstant(double r1, double r2,
                                    double t1, double tk, double t2,
                                    const CIRParams& params,
                                    double r_max = 1.0, int n_points = 1000) {
    double dr = r_max / n_points;
    double integral = 0.0;

    for (int i = 0; i < n_points; i++) {
        double r = (i + 0.5) * dr;
        double density = evaluateBridgeDensity(r1, r, r2, t1, tk, t2, params);
        integral += density * dr;
    }

    return integral;
}

/**
 * @brief Kolmogorov-Smirnov test statistic
 */
double computeKSStatistic(const std::vector<double>& samples,
                          const std::vector<double>& cdf_values) {
    double max_diff = 0.0;
    for (size_t i = 0; i < samples.size(); i++) {
        double empirical_cdf = (i + 1.0) / samples.size();
        double theoretical_cdf = cdf_values[i];
        double diff = std::abs(empirical_cdf - theoretical_cdf);
        max_diff = std::max(max_diff, diff);
    }
    return max_diff;
}

/**
 * @brief Test if bridge samples from correct distribution
 */
void testBridgeDistribution() {
    std::cout << "=== CIR Bridge Distribution Test ===\n\n";

    CIRParams params(0.5, 0.04, 0.1);

    std::cout << "CIR Parameters:\n";
    std::cout << "  kappa: " << params.kappa << "\n";
    std::cout << "  theta: " << params.theta << "\n";
    std::cout << "  sigma: " << params.sigma << "\n\n";

    // Test scenario
    double r1 = 0.03;
    double r2 = 0.05;
    double t1 = 0.0;
    double tk = 0.5;
    double t2 = 1.0;

    std::cout << "Bridge Test Scenario:\n";
    std::cout << "  r(t1=" << t1 << ") = " << r1 << "\n";
    std::cout << "  r(t2=" << t2 << ") = " << r2 << "\n";
    std::cout << "  Inserting at tk=" << tk << "\n\n";

    // Sample many bridge values
    const int N_SAMPLES = 100000;
    std::cout << "Generating " << N_SAMPLES << " bridge samples...\n\n";

    std::vector<double> samples_rejection;
    std::vector<double> samples_modified;
    std::vector<double> samples_unconditional;

    samples_rejection.reserve(N_SAMPLES);
    samples_modified.reserve(N_SAMPLES);
    samples_unconditional.reserve(N_SAMPLES);

    for (int i = 0; i < N_SAMPLES; i++) {
        samples_rejection.push_back(
            sampleCIRBridge_AndersenRejection(r1, r2, t1, tk, t2, params));
        samples_modified.push_back(
            sampleCIRBridge_AndersenModified(r1, r2, t1, tk, t2, params));
        samples_unconditional.push_back(
            sampleCIR_Andersen(r1, tk - t1, params));
    }

    // Compute empirical moments
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

    auto [mean_rej, var_rej, std_rej, skew_rej] = compute_moments(samples_rejection);
    auto [mean_mod, var_mod, std_mod, skew_mod] = compute_moments(samples_modified);
    auto [mean_unc, var_unc, std_unc, skew_unc] = compute_moments(samples_unconditional);

    // Compute theoretical moments by numerical integration
    std::cout << "Computing theoretical moments via numerical integration...\n\n";

    double r_max = 0.3;  // Integration range
    int n_points = 5000;
    double dr = r_max / n_points;

    double Z = computeNormalizationConstant(r1, r2, t1, tk, t2, params, r_max, n_points);

    double theoretical_mean = 0.0;
    double theoretical_variance = 0.0;
    double theoretical_skewness = 0.0;

    for (int i = 0; i < n_points; i++) {
        double r = (i + 0.5) * dr;
        double density = evaluateBridgeDensity(r1, r, r2, t1, tk, t2, params) / Z;
        theoretical_mean += r * density * dr;
    }

    for (int i = 0; i < n_points; i++) {
        double r = (i + 0.5) * dr;
        double density = evaluateBridgeDensity(r1, r, r2, t1, tk, t2, params) / Z;
        double diff = r - theoretical_mean;
        theoretical_variance += diff * diff * density * dr;
        theoretical_skewness += diff * diff * diff * density * dr;
    }

    theoretical_skewness /= std::pow(theoretical_variance, 1.5);

    // Print moment comparison
    std::cout << std::fixed << std::setprecision(6);

    std::cout << "=== Moment Comparison ===\n\n";

    std::cout << "                      Mean        Variance    Std Dev     Skewness\n";
    std::cout << "Theoretical:          " << theoretical_mean << "   "
              << theoretical_variance << "   " << std::sqrt(theoretical_variance) << "   "
              << theoretical_skewness << "\n";
    std::cout << "Rejection Bridge:     " << mean_rej << "   "
              << var_rej << "   " << std_rej << "   " << skew_rej << "\n";
    std::cout << "Modified Bridge:      " << mean_mod << "   "
              << var_mod << "   " << std_mod << "   " << skew_mod << "\n";
    std::cout << "Unconditional:        " << mean_unc << "   "
              << var_unc << "   " << std_unc << "   " << skew_unc << "\n\n";

    std::cout << "=== Moment Errors (vs Theoretical) ===\n\n";
    std::cout << "Rejection Bridge:\n";
    std::cout << "  Mean error:     " << (mean_rej - theoretical_mean) / theoretical_mean * 100 << "%\n";
    std::cout << "  Variance error: " << (var_rej - theoretical_variance) / theoretical_variance * 100 << "%\n";
    std::cout << "  Skewness error: " << (skew_rej - theoretical_skewness) / theoretical_skewness * 100 << "%\n\n";

    std::cout << "Modified Bridge:\n";
    std::cout << "  Mean error:     " << (mean_mod - theoretical_mean) / theoretical_mean * 100 << "%\n";
    std::cout << "  Variance error: " << (var_mod - theoretical_variance) / theoretical_variance * 100 << "%\n";
    std::cout << "  Skewness error: " << (skew_mod - theoretical_skewness) / theoretical_skewness * 100 << "%\n\n";

    std::cout << "Unconditional (WRONG - should NOT match):\n";
    std::cout << "  Mean error:     " << (mean_unc - theoretical_mean) / theoretical_mean * 100 << "%\n";
    std::cout << "  Variance error: " << (var_unc - theoretical_variance) / theoretical_variance * 100 << "%\n";
    std::cout << "  Skewness error: " << (skew_unc - theoretical_skewness) / theoretical_skewness * 100 << "%\n\n";

    // Histogram comparison
    std::cout << "=== Histogram Comparison ===\n\n";

    const int N_BINS = 20;
    std::vector<int> hist_rejection(N_BINS, 0);
    std::vector<int> hist_modified(N_BINS, 0);
    std::vector<int> hist_unconditional(N_BINS, 0);
    std::vector<double> hist_theoretical(N_BINS, 0.0);

    double hist_min = 0.0;
    double hist_max = 0.15;
    double bin_width = (hist_max - hist_min) / N_BINS;

    // Fill empirical histograms
    for (double x : samples_rejection) {
        int bin = std::min(N_BINS - 1, std::max(0, (int)((x - hist_min) / bin_width)));
        hist_rejection[bin]++;
    }
    for (double x : samples_modified) {
        int bin = std::min(N_BINS - 1, std::max(0, (int)((x - hist_min) / bin_width)));
        hist_modified[bin]++;
    }
    for (double x : samples_unconditional) {
        int bin = std::min(N_BINS - 1, std::max(0, (int)((x - hist_min) / bin_width)));
        hist_unconditional[bin]++;
    }

    // Fill theoretical histogram
    for (int i = 0; i < N_BINS; i++) {
        double bin_center = hist_min + (i + 0.5) * bin_width;
        hist_theoretical[i] = evaluateBridgeDensity(r1, bin_center, r2, t1, tk, t2, params) / Z;
    }

    std::cout << "Bin Range         Rejection    Modified     Uncond       Theoretical\n";
    for (int i = 0; i < N_BINS; i++) {
        double bin_start = hist_min + i * bin_width;
        double bin_end = hist_min + (i + 1) * bin_width;

        double empirical_rej = (double)hist_rejection[i] / N_SAMPLES / bin_width;
        double empirical_mod = (double)hist_modified[i] / N_SAMPLES / bin_width;
        double empirical_unc = (double)hist_unconditional[i] / N_SAMPLES / bin_width;

        std::cout << "[" << std::setw(6) << bin_start << ", "
                  << std::setw(6) << bin_end << "]:  "
                  << std::setw(10) << empirical_rej << "  "
                  << std::setw(10) << empirical_mod << "  "
                  << std::setw(10) << empirical_unc << "  "
                  << std::setw(10) << hist_theoretical[i] << "\n";
    }

    // Chi-squared goodness of fit test
    std::cout << "\n=== Chi-Squared Goodness of Fit ===\n\n";

    auto compute_chi_squared = [&](const std::vector<int>& hist) {
        double chi2 = 0.0;
        for (int i = 0; i < N_BINS; i++) {
            double observed = hist[i];
            double expected = hist_theoretical[i] * bin_width * N_SAMPLES;
            if (expected > 5.0) {  // Only use bins with sufficient expected count
                chi2 += (observed - expected) * (observed - expected) / expected;
            }
        }
        return chi2;
    };

    double chi2_rejection = compute_chi_squared(hist_rejection);
    double chi2_modified = compute_chi_squared(hist_modified);
    double chi2_unconditional = compute_chi_squared(hist_unconditional);

    std::cout << "Chi-squared statistic (lower is better):\n";
    std::cout << "  Rejection Bridge:  " << chi2_rejection << "\n";
    std::cout << "  Modified Bridge:   " << chi2_modified << "\n";
    std::cout << "  Unconditional:     " << chi2_unconditional << "\n\n";

    std::cout << "Degrees of freedom: ~" << (N_BINS - 1) << "\n";
    std::cout << "Critical value (95% confidence): ~" << 30.14 << " (for df=19)\n\n";

    if (chi2_rejection < 30.14) {
        std::cout << "✓ Rejection bridge passes chi-squared test!\n";
    } else {
        std::cout << "✗ Rejection bridge fails chi-squared test\n";
    }

    if (chi2_modified < 30.14) {
        std::cout << "✓ Modified bridge passes chi-squared test!\n";
    } else {
        std::cout << "✗ Modified bridge fails chi-squared test\n";
    }
}

int main() {
    try {
        testBridgeDistribution();

        std::cout << "\n========================================\n";
        std::cout << "Distribution test completed!\n";
        std::cout << "========================================\n";

        return 0;
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }
}
