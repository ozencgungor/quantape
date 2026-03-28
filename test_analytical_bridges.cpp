/**
 * @file test_analytical_bridges.cpp
 * @brief Comprehensively tests and compares various CIR bridge methods across a
 *        wide range of scenarios and parameters.
 */

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <map>
#include <numeric>
#include <random>
#include <string>
#include <vector>

// --- Core CIR Simulation and Bridge Components ---

struct CIRParams {
    double kappa;
    double theta;
    double sigma;
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
    if (params.kappa <= 1e-8) {
        return r_t + params.sigma * std::sqrt(r_t) * randn() * std::sqrt(dt);
    }
    double c = params.sigma * params.sigma / (4.0 * params.kappa);
    double exp_kt = std::exp(-params.kappa * dt);
    double m = params.theta + (r_t - params.theta) * exp_kt;
    double s2 =
        r_t * c * exp_kt * (1.0 - exp_kt) + params.theta * c * (1.0 - exp_kt) * (1.0 - exp_kt);
    double psi = (s2 > 1e-12) ? s2 / (m * m) : 0.0;

    if (psi <= 1.5) {
        double b2 =
            (psi > 1e-12) ? 2.0 / psi - 1.0 + std::sqrt(2.0 / psi * (2.0 / psi - 1.0)) : 1.0;
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
        return 0.0;
    if (params.kappa <= 1e-8)
        return 0.0;
    double exp_kt = std::exp(-params.kappa * dt);
    double c = params.sigma * params.sigma / (4.0 * params.kappa);
    double m = params.theta + (r_start - params.theta) * exp_kt;
    double s2 =
        r_start * c * exp_kt * (1.0 - exp_kt) + params.theta * c * (1.0 - exp_kt) * (1.0 - exp_kt);
    double psi = (s2 > 1e-12) ? s2 / (m * m) : 0.0;

    if (psi <= 1.5) {
        double b2 =
            (psi > 1e-12) ? 2.0 / psi - 1.0 + std::sqrt(2.0 / psi * (2.0 / psi - 1.0)) : 1.0;
        double a = m / (1.0 + b2);
        if (r_end <= 1e-12)
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
        return (r_end < 1e-12) ? p : (1.0 - p) * beta * std::exp(-beta * r_end);
    }
}

double sampleCIRBridge_Exact(double r1, double r2, double t1, double tk, double t2,
                             const CIRParams& params) {
    double dt1 = tk - t1;
    double dt2 = t2 - tk;
    if (dt1 < 1e-8)
        return r1;
    if (dt2 < 1e-8)
        return r2;

    double m_uncond = params.theta + (r1 - params.theta) * std::exp(-params.kappa * dt1);
    double M = evaluateDensity_Andersen(m_uncond, r2, dt2, params) * 1.5;
    if (M <= 1e-12)
        return m_uncond;

    for (int attempt = 0; attempt < 100000; attempt++) {
        double y = sampleCIR_Andersen(r1, dt1, params);
        double accept_prob = evaluateDensity_Andersen(y, r2, dt2, params) / M;
        if (uniform_random() < accept_prob)
            return y;
    }
    return sampleCIR_Andersen(r1, dt1, params);
}

// --- Helper for Variance ---

double get_cir_variance(double x0, double dt, const CIRParams& params) {
    const double& k = params.kappa;
    if (k <= 1e-8)
        return 0.0;
    const double& th = params.theta;
    const double& s = params.sigma;
    const double s2 = s * s;
    const double e_kt = std::exp(-k * dt);

    return (x0 * s2 / k) * (e_kt - e_kt * e_kt) + (th * s2 / (2.0 * k)) * std::pow(1.0 - e_kt, 2);
}

// --- X-Space Moment-Matching Bridge Implementations ---

// "v1": The "wrong-but-good" version that does not use updated theta for variance calc
double bridge_x_mm_v1_buggy(double r1, double r2, double t1, double tk, double t2,
                            const CIRParams& params) {
    double dt1 = tk - t1;
    if (dt1 < 1e-8)
        return r1;
    double dt2 = t2 - tk;
    if (dt2 < 1e-8)
        return r2;

    const double& k = params.kappa;
    const double& th = params.theta;

    double exp_k_dt1 = std::exp(-k * dt1);
    double exp_k_dt2 = std::exp(-k * dt2);

    double E_1k = r1 * exp_k_dt1 + th * (1.0 - exp_k_dt1);
    double E_12 = r1 * std::exp(-k * (dt1 + dt2)) + th * (1.0 - std::exp(-k * (dt1 + dt2)));
    double var_1k = get_cir_variance(r1, dt1, params);
    double var_12 = get_cir_variance(r1, dt1 + dt2, params);
    double cov_k2_1 = var_1k * exp_k_dt2;
    double mean_x_bridge = E_1k;
    if (var_12 > 1e-12) {
        mean_x_bridge += (r2 - E_12) * cov_k2_1 / var_12;
    }

    double var_k2_approx = get_cir_variance(mean_x_bridge, dt2, params);
    double var_x_bridge = (var_1k > 1e-12 && var_k2_approx > 1e-12)
                              ? (var_1k * var_k2_approx) / (var_1k + var_k2_approx)
                              : 0.0;

    CIRParams modified_params = params;
    if (std::abs(1.0 - exp_k_dt1) > 1e-8) {
        modified_params.theta = (mean_x_bridge - r1 * exp_k_dt1) / (1.0 - exp_k_dt1);
    }

    if (var_1k > 1e-12) {
        double var_ratio = var_x_bridge / var_1k;
        modified_params.sigma = params.sigma * std::sqrt(std::max(0.0, var_ratio));
    }

    return sampleCIR_Andersen(r1, dt1, modified_params);
}

// "v2": The theoretically corrected version
double bridge_x_mm_v2_corrected(double r1, double r2, double t1, double tk, double t2,
                                const CIRParams& params, bool use_tv_correction) {
    double dt1 = tk - t1;
    if (dt1 < 1e-8)
        return r1;
    double dt2 = t2 - tk;
    if (dt2 < 1e-8)
        return r2;

    const double& k = params.kappa;
    const double& s = params.sigma;
    const double& th = params.theta;

    double exp_k_dt1 = std::exp(-k * dt1);
    double exp_k_dt2 = std::exp(-k * dt2);

    double E_1k = r1 * exp_k_dt1 + th * (1.0 - exp_k_dt1);
    double E_12 = r1 * std::exp(-k * (dt1 + dt2)) + th * (1.0 - std::exp(-k * (dt1 + dt2)));
    double var_1k = get_cir_variance(r1, dt1, params);
    double var_12 = get_cir_variance(r1, dt1 + dt2, params);
    double cov_k2_1 = var_1k * exp_k_dt2;
    double mean_x_bridge = E_1k;
    if (var_12 > 1e-12) {
        mean_x_bridge += (r2 - E_12) * cov_k2_1 / var_12;
    }

    double var_k2_approx = get_cir_variance(mean_x_bridge, dt2, params);
    double var_x_bridge_ou_approx = (var_1k > 1e-12 && var_k2_approx > 1e-12)
                                        ? (var_1k * var_k2_approx) / (var_1k + var_k2_approx)
                                        : 0.0;

    double var_x_bridge = var_x_bridge_ou_approx;
    if (use_tv_correction) {
        double M_b = mean_x_bridge;
        if (M_b > 1e-8 && s > 1e-8 && (var_1k + var_k2_approx) > 1e-12) {
            double T_V =
                (var_1k * var_1k * var_k2_approx + var_1k * var_k2_approx * var_k2_approx) /
                std::pow(var_1k + var_k2_approx, 2);
            double b_prime_sq = (s * s) / (4.0 * M_b);
            double delta_V = -0.5 * b_prime_sq * T_V;
            var_x_bridge += delta_V;
        }
    }

    CIRParams modified_params = params;
    if (std::abs(1.0 - exp_k_dt1) > 1e-8) {
        modified_params.theta = (mean_x_bridge - r1 * exp_k_dt1) / (1.0 - exp_k_dt1);
    }

    double e_2kdt1 = exp_k_dt1 * exp_k_dt1;
    double var_factor = (k > 1e-8)
                            ? ((r1 / k) * (exp_k_dt1 - e_2kdt1) +
                               (modified_params.theta / (2.0 * k)) * std::pow(1.0 - exp_k_dt1, 2))
                            : 0.0;

    if (var_factor > 1e-12) {
        double s2_target = var_x_bridge / var_factor;
        modified_params.sigma = std::sqrt(std::max(0.0, s2_target));
    }

    return sampleCIR_Andersen(r1, dt1, modified_params);
}

// --- Comprehensive Performance Test ---

struct TestCase {
    std::string name;
    CIRParams params;
    double r1, r2, t1, tk, t2;
};

struct MethodResult {
    std::string method_name;
    double mean_err;
    double var_err;
    double skew_err;
};

std::vector<MethodResult> run_full_scenario_test(const TestCase& test_case) {
    std::cout << "\n============================================================" << std::endl;
    std::cout << "--- Testing Scenario: " << test_case.name << " ---" << std::endl;
    std::cout << "  Params: k=" << test_case.params.kappa << ", th=" << test_case.params.theta
              << ", s=" << test_case.params.sigma << std::endl;
    std::cout << "  Values: r1=" << test_case.r1 << ", r2=" << test_case.r2 << std::endl;
    std::cout << "  Time:   t1=" << test_case.t1 << ", tk=" << test_case.tk
              << ", t2=" << test_case.t2 << std::endl;

    const int N_SAMPLES = 20000;

    std::vector<std::string> method_names = {"v1_buggy", "v2_corrected", "v2_Tv"};
    std::vector<MethodResult> results;

    std::vector<double> samples_exact;
    samples_exact.reserve(N_SAMPLES);
    for (int i = 0; i < N_SAMPLES; ++i) {
        rng.seed(1000 + i);
        samples_exact.push_back(sampleCIRBridge_Exact(test_case.r1, test_case.r2, test_case.t1,
                                                      test_case.tk, test_case.t2,
                                                      test_case.params));
    }

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
        double std_dev = std::sqrt(variance);
        if (std_dev > 1e-9) {
            skewness /= (data.size() * std_dev * std_dev * std_dev);
        } else {
            skewness = 0.0;
        }
        return std::make_tuple(mean, variance, skewness);
    };

    auto [mean_exact, var_exact, skew_exact] = compute_moments(samples_exact);

    for (const auto& method_name : method_names) {
        std::vector<double> samples_approx;
        samples_approx.reserve(N_SAMPLES);
        for (int i = 0; i < N_SAMPLES; ++i) {
            rng.seed(1000 + i);
            if (method_name == "v1_buggy") {
                samples_approx.push_back(bridge_x_mm_v1_buggy(test_case.r1, test_case.r2,
                                                              test_case.t1, test_case.tk,
                                                              test_case.t2, test_case.params));
            } else if (method_name == "v2_corrected") {
                samples_approx.push_back(
                    bridge_x_mm_v2_corrected(test_case.r1, test_case.r2, test_case.t1, test_case.tk,
                                             test_case.t2, test_case.params, false));
            } else if (method_name == "v2_Tv") {
                samples_approx.push_back(
                    bridge_x_mm_v2_corrected(test_case.r1, test_case.r2, test_case.t1, test_case.tk,
                                             test_case.t2, test_case.params, true));
            }
        }
        auto [mean_approx, var_approx, skew_approx] = compute_moments(samples_approx);

        MethodResult result;
        result.method_name = method_name;
        result.mean_err = (mean_exact > 1e-9) ? (mean_approx - mean_exact) / mean_exact * 100 : 0.0;
        result.var_err = (var_exact > 1e-12) ? (var_approx - var_exact) / var_exact * 100 : 0.0;
        result.skew_err =
            (std::abs(skew_exact) > 1e-9) ? (skew_approx - skew_exact) / skew_exact * 100 : 0.0;
        results.push_back(result);
    }

    std::cout << "\n  --- Errors vs Exact (%)\\n";
    std::cout << "  Method                    Mean Err    Var Err     Skew Err\n";
    std::cout << "  ------------------------------------------------------------\n";
    auto print_errors = [&](const MethodResult& res) {
        std::cout << "  " << std::setw(25) << std::left << res.method_name << std::setw(12)
                  << std::right << res.mean_err << std::setw(12) << std::right << res.var_err
                  << std::setw(12) << std::right << res.skew_err << "\n";
    };
    for (const auto& res : results) {
        print_errors(res);
    }

    return results;
}

int main() {
    try {
        std::vector<TestCase> test_cases;
        std::vector<double> kappa_grid = {0.01, 0.05, 0.1};
        std::vector<double> sigma_grid = {0.1, 0.2, 0.4};
        std::vector<double> theta_grid = {0.02, 0.1, 0.5};
        std::vector<double> r1_grid = {0.01, 0.1, 0.4};

        // Create time variations
        struct TimeVariation {
            std::string name;
            double tk;
            double t2;
        };
        std::vector<TimeVariation> time_variations = {{" (Midpoint)", 0.5, 1.0},
                                                      {" (Short Tenor)", 7.0 / 365.0, 14.0 / 365.0},
                                                      {" (Long Tenor)", 2.5, 5.0},
                                                      {" (Asymm Near Start)", 0.1, 1.0},
                                                      {" (Asymm Near End)", 0.9, 1.0}};

        // Generate test cases from grids
        int case_count = 0;
        for (double k : kappa_grid) {
            for (double s : sigma_grid) {
                for (double th : theta_grid) {
                    for (double r1 : r1_grid) {
                        CIRParams params = {k, th, s};
                        // For each param set, create all time variations
                        for (const auto& time_var : time_variations) {
                            // Simple r2 logic for stability
                            double r2 = r1 * 0.9 + th * 0.1;
                            test_cases.push_back(
                                {"Grid Case #" + std::to_string(++case_count) + time_var.name,
                                 params, r1, r2, 0.0, time_var.tk, time_var.t2});
                        }
                    }
                }
            }
        }

        std::cout << "Generated " << test_cases.size() << " unique test scenarios." << std::endl;

        std::map<std::string, std::vector<double>> aggregate_mean_errors, aggregate_var_errors,
            aggregate_skew_errors;

        for (const auto& test : test_cases) {
            auto results = run_full_scenario_test(test);
            for (const auto& res : results) {
                aggregate_mean_errors[res.method_name].push_back(std::abs(res.mean_err));
                aggregate_var_errors[res.method_name].push_back(std::abs(res.var_err));
                aggregate_skew_errors[res.method_name].push_back(std::abs(res.skew_err));
            }
        }

        std::cout << "\n\n========================================================" << std::endl;
        std::cout << "           AGGREGATE ERROR ACROSS ALL " << test_cases.size() << " SCENARIOS"
                  << std::endl;
        std::cout << "========================================================" << std::endl;
        std::cout
            << "Method                    Avg Abs Mean Err  Avg Abs Var Err   Avg Abs Skew Err\n";
        std::cout
            << "--------------------------------------------------------------------------------\n";

        std::vector<std::string> method_names = {"v1_buggy", "v2_corrected", "v2_Tv"};
        for (const auto& method : method_names) {
            double avg_mean_err = std::accumulate(aggregate_mean_errors[method].begin(),
                                                  aggregate_mean_errors[method].end(), 0.0) /
                                  aggregate_mean_errors[method].size();
            double avg_var_err = std::accumulate(aggregate_var_errors[method].begin(),
                                                 aggregate_var_errors[method].end(), 0.0) /
                                 aggregate_var_errors[method].size();
            double avg_skew_err = std::accumulate(aggregate_skew_errors[method].begin(),
                                                  aggregate_skew_errors[method].end(), 0.0) /
                                  aggregate_skew_errors[method].size();

            std::cout << std::setw(25) << std::left << method << std::setw(18) << std::right
                      << avg_mean_err << " %" << std::setw(18) << std::right << avg_var_err << " %"
                      << std::setw(18) << std::right << avg_skew_err << " %" << "\n";
        }

        std::cout << "\n\n";

        return 0;
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }
}