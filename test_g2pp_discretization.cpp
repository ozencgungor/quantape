/**
 * @file test_g2pp_discretization.cpp
 * @brief Compare different discretization schemes for G2++ model
 *
 * G2++ Model:
 *   dx(t) = -a*x(t)dt + σ*dW₁(t)
 *   dy(t) = -b*y(t)dt + η*dW₂(t)
 *   Corr(dW₁, dW₂) = ρ
 *
 * Discretization methods tested:
 *   1. Euler-Maruyama (standard first-order)
 *   2. Exponential Euler (exact drift, approximate diffusion)
 *   3. Exact Simulation (analytical conditional distribution)
 *   4. Milstein (second-order correction)
 *   5. Predictor-Corrector
 */

#include <iostream>
#include <iomanip>
#include <vector>
#include <random>
#include <cmath>
#include <array>
#include <chrono>
#include <Eigen/Dense>

using Eigen::Vector2d;
using Eigen::Matrix2d;

struct G2PPParams {
    double a;      // Mean reversion speed for x
    double b;      // Mean reversion speed for y
    double sigma;  // Volatility of x
    double eta;    // Volatility of y
    double rho;    // Correlation between W₁ and W₂

    G2PPParams(double a_, double b_, double sig_, double eta_, double rho_)
        : a(a_), b(b_), sigma(sig_), eta(eta_), rho(rho_) {}
};

std::mt19937_64 rng(42);
std::normal_distribution<double> normal_dist(0.0, 1.0);

double randn() { return normal_dist(rng); }

// Sample correlated normals
std::pair<double, double> sampleCorrelatedNormals(double rho) {
    double Z1 = randn();
    double Z2 = randn();
    double W1 = Z1;
    double W2 = rho * Z1 + std::sqrt(1.0 - rho * rho) * Z2;
    return {W1, W2};
}

// ============================================================================
// Method 1: Euler-Maruyama (standard)
// ============================================================================
/**
 * Simplest discretization: linear approximation of drift and diffusion
 *
 * x_{n+1} = x_n - a*x_n*Δt + σ*√Δt*Z₁
 * y_{n+1} = y_n - b*y_n*Δt + η*√Δt*Z₂
 *
 * Pros: Simple, intuitive
 * Cons: First-order accuracy O(√Δt), can have stability issues for large Δt
 */
Vector2d step_EulerMaruyama(const Vector2d& state, double dt, const G2PPParams& params) {
    auto [dW1, dW2] = sampleCorrelatedNormals(params.rho);

    Vector2d new_state;
    new_state[0] = state[0] - params.a * state[0] * dt + params.sigma * std::sqrt(dt) * dW1;
    new_state[1] = state[1] - params.b * state[1] * dt + params.eta * std::sqrt(dt) * dW2;

    return new_state;
}

// ============================================================================
// Method 2: Exponential Euler (exact drift integration)
// ============================================================================
/**
 * Integrates drift exactly, approximates diffusion
 *
 * For dx = -a*x*dt + σ*dW:
 *   x_{n+1} = x_n*exp(-a*Δt) + σ*∫₀^Δt exp(-a*(Δt-s)) dW(s)
 *            ≈ x_n*exp(-a*Δt) + σ*√[(1-exp(-2a*Δt))/(2a)] * Z
 *
 * This is the EXACT transition for OU process!
 *
 * Pros: Exact for OU, unconditionally stable, excellent for stiff systems
 * Cons: Slightly more complex than Euler
 */
Vector2d step_ExponentialEuler(const Vector2d& state, double dt, const G2PPParams& params) {
    auto [dW1, dW2] = sampleCorrelatedNormals(params.rho);

    // Exact integration for OU process
    double exp_a = std::exp(-params.a * dt);
    double exp_b = std::exp(-params.b * dt);

    double std_x = params.sigma * std::sqrt((1.0 - std::exp(-2.0 * params.a * dt)) / (2.0 * params.a));
    double std_y = params.eta * std::sqrt((1.0 - std::exp(-2.0 * params.b * dt)) / (2.0 * params.b));

    Vector2d new_state;
    new_state[0] = state[0] * exp_a + std_x * dW1;
    new_state[1] = state[1] * exp_b + std_y * dW2;

    return new_state;
}

// ============================================================================
// Method 3: Exact Simulation (analytical conditional distribution)
// ============================================================================
/**
 * Same as Exponential Euler for OU processes!
 *
 * Uses analytical formulas:
 *   E[x(t+Δt) | x(t)] = x(t)*exp(-a*Δt)
 *   Var[x(t+Δt) | x(t)] = σ²/(2a) * (1 - exp(-2a*Δt))
 *
 * Pros: Exact, no discretization error
 * Cons: Only available for special processes (OU, geometric Brownian, etc.)
 */
Vector2d step_Exact(const Vector2d& state, double dt, const G2PPParams& params) {
    // For OU process, this is identical to Exponential Euler
    return step_ExponentialEuler(state, dt, params);
}

// ============================================================================
// Method 4: Milstein (second-order correction)
// ============================================================================
/**
 * Adds second-order correction term involving (dW)²
 *
 * General form: x_{n+1} = x_n + a*Δt + b*dW + 0.5*b*b'*(dW² - Δt)
 *
 * For OU process dx = -a*x*dt + σ*dW:
 *   Diffusion coefficient b(x) = σ (constant!)
 *   Derivative b'(x) = 0
 *   → Milstein correction vanishes, reduces to Euler!
 *
 * Pros: Second-order for general SDEs
 * Cons: No advantage for OU (constant diffusion coefficient)
 */
Vector2d step_Milstein(const Vector2d& state, double dt, const G2PPParams& params) {
    auto [dW1, dW2] = sampleCorrelatedNormals(params.rho);
    double sqrt_dt = std::sqrt(dt);

    // For OU with constant σ, Milstein = Euler (b' = 0)
    Vector2d new_state;
    new_state[0] = state[0] - params.a * state[0] * dt + params.sigma * sqrt_dt * dW1;
    new_state[1] = state[1] - params.b * state[1] * dt + params.eta * sqrt_dt * dW2;

    return new_state;
}

// ============================================================================
// Method 5: Predictor-Corrector (Heun's method)
// ============================================================================
/**
 * Two-stage method: predict with Euler, then correct using average drift
 *
 * Predictor:  x̃ = x_n + drift(x_n)*Δt + σ*√Δt*Z
 * Corrector:  x_{n+1} = x_n + 0.5*[drift(x_n) + drift(x̃)]*Δt + σ*√Δt*Z
 *
 * Pros: Better stability than Euler, second-order for deterministic ODEs
 * Cons: Two drift evaluations, still O(√Δt) for SDEs (not O(Δt))
 */
Vector2d step_PredictorCorrector(const Vector2d& state, double dt, const G2PPParams& params) {
    auto [dW1, dW2] = sampleCorrelatedNormals(params.rho);
    double sqrt_dt = std::sqrt(dt);

    // Predictor step (Euler)
    Vector2d predicted;
    predicted[0] = state[0] - params.a * state[0] * dt + params.sigma * sqrt_dt * dW1;
    predicted[1] = state[1] - params.b * state[1] * dt + params.eta * sqrt_dt * dW2;

    // Corrector step (use average drift)
    Vector2d new_state;
    double drift_x = -params.a * (state[0] + predicted[0]) * 0.5;
    double drift_y = -params.b * (state[1] + predicted[1]) * 0.5;

    new_state[0] = state[0] + drift_x * dt + params.sigma * sqrt_dt * dW1;
    new_state[1] = state[1] + drift_y * dt + params.eta * sqrt_dt * dW2;

    return new_state;
}

// ============================================================================
// Method 6: Implicit Euler (unconditionally stable)
// ============================================================================
/**
 * Backward/implicit discretization for improved stability
 *
 * x_{n+1} = x_n - a*x_{n+1}*Δt + σ*√Δt*Z
 *
 * Solving for x_{n+1}:
 *   x_{n+1} = (x_n + σ*√Δt*Z) / (1 + a*Δt)
 *
 * Pros: Unconditionally stable (great for stiff systems)
 * Cons: Requires solving implicit equation (easy for linear OU though)
 */
Vector2d step_ImplicitEuler(const Vector2d& state, double dt, const G2PPParams& params) {
    auto [dW1, dW2] = sampleCorrelatedNormals(params.rho);
    double sqrt_dt = std::sqrt(dt);

    // For linear OU, can solve exactly
    Vector2d new_state;
    new_state[0] = (state[0] + params.sigma * sqrt_dt * dW1) / (1.0 + params.a * dt);
    new_state[1] = (state[1] + params.eta * sqrt_dt * dW2) / (1.0 + params.b * dt);

    return new_state;
}

// ============================================================================
// Helper functions
// ============================================================================

struct Statistics {
    double mean_x;
    double mean_y;
    double var_x;
    double var_y;
    double cov_xy;
};

Statistics computeStatistics(const std::vector<Vector2d>& paths) {
    int n = paths.size();

    Vector2d mean = Vector2d::Zero();
    for (const auto& p : paths) {
        mean += p;
    }
    mean /= n;

    Matrix2d cov = Matrix2d::Zero();
    for (const auto& p : paths) {
        Vector2d diff = p - mean;
        cov += diff * diff.transpose();
    }
    cov /= n;

    Statistics s;
    s.mean_x = mean[0];
    s.mean_y = mean[1];
    s.var_x = cov(0, 0);
    s.var_y = cov(1, 1);
    s.cov_xy = cov(0, 1);

    return s;
}

// Theoretical moments for OU process
struct TheoreticalMoments {
    double mean_x;
    double mean_y;
    double var_x;
    double var_y;
    double cov_xy;
};

TheoreticalMoments getTheoreticalMoments(const Vector2d& x0, double T, const G2PPParams& params) {
    TheoreticalMoments m;

    double exp_a = std::exp(-params.a * T);
    double exp_b = std::exp(-params.b * T);

    // Mean: x(T) = x(0)*exp(-a*T)
    m.mean_x = x0[0] * exp_a;
    m.mean_y = x0[1] * exp_b;

    // Variance: σ²/(2a) * (1 - exp(-2aT))
    m.var_x = (params.sigma * params.sigma) / (2.0 * params.a) * (1.0 - std::exp(-2.0 * params.a * T));
    m.var_y = (params.eta * params.eta) / (2.0 * params.b) * (1.0 - std::exp(-2.0 * params.b * T));

    // Covariance: ρ*σ*η/(a+b) * (1 - exp(-(a+b)*T))
    m.cov_xy = params.rho * params.sigma * params.eta / (params.a + params.b) *
               (1.0 - std::exp(-(params.a + params.b) * T));

    return m;
}

void compareDiscretizationSchemes() {
    std::cout << "=== G2++ Discretization Schemes Comparison ===\n\n";

    G2PPParams params(0.1, 0.3, 0.01, 0.015, 0.6);

    std::cout << "Model Parameters:\n";
    std::cout << "  a = " << params.a << ", b = " << params.b << "\n";
    std::cout << "  σ = " << params.sigma << ", η = " << params.eta << "\n";
    std::cout << "  ρ = " << params.rho << "\n\n";

    Vector2d x0(0.02, -0.01);
    double T = 1.0;

    std::vector<int> n_steps_vec = {10, 50, 100, 500};
    const int N_PATHS = 10000;

    std::cout << "Initial state: x(0) = " << x0[0] << ", y(0) = " << x0[1] << "\n";
    std::cout << "Terminal time: T = " << T << "\n";
    std::cout << "Number of paths: " << N_PATHS << "\n\n";

    // Theoretical moments
    auto theory = getTheoreticalMoments(x0, T, params);
    std::cout << "Theoretical Terminal Statistics:\n";
    std::cout << std::fixed << std::setprecision(6);
    std::cout << "  E[x(T)] = " << theory.mean_x << "\n";
    std::cout << "  E[y(T)] = " << theory.mean_y << "\n";
    std::cout << "  Var[x(T)] = " << theory.var_x << "\n";
    std::cout << "  Var[y(T)] = " << theory.var_y << "\n";
    std::cout << "  Cov[x(T), y(T)] = " << theory.cov_xy << "\n\n";

    for (int n_steps : n_steps_vec) {
        double dt = T / n_steps;

        std::cout << "========================================\n";
        std::cout << "Time steps: " << n_steps << " (Δt = " << dt << ")\n";
        std::cout << "========================================\n\n";

        // Test each method
        std::vector<std::pair<std::string, std::function<Vector2d(const Vector2d&, double, const G2PPParams&)>>> methods = {
            {"Euler-Maruyama", step_EulerMaruyama},
            {"Exponential Euler", step_ExponentialEuler},
            {"Exact Simulation", step_Exact},
            {"Milstein", step_Milstein},
            {"Predictor-Corrector", step_PredictorCorrector},
            {"Implicit Euler", step_ImplicitEuler}
        };

        for (const auto& [name, method] : methods) {
            std::vector<Vector2d> terminal_states;
            terminal_states.reserve(N_PATHS);

            auto start = std::chrono::high_resolution_clock::now();

            for (int path = 0; path < N_PATHS; path++) {
                rng.seed(1000 + path);

                Vector2d state = x0;
                for (int step = 0; step < n_steps; step++) {
                    state = method(state, dt, params);
                }

                terminal_states.push_back(state);
            }

            auto end = std::chrono::high_resolution_clock::now();
            double elapsed = std::chrono::duration<double, std::milli>(end - start).count();

            auto stats = computeStatistics(terminal_states);

            std::cout << name << ":\n";
            std::cout << "  E[x(T)]:       " << std::setw(10) << stats.mean_x
                      << " (error: " << std::setw(8) << (stats.mean_x - theory.mean_x) / theory.mean_x * 100 << "%)\n";
            std::cout << "  E[y(T)]:       " << std::setw(10) << stats.mean_y
                      << " (error: " << std::setw(8) << (stats.mean_y - theory.mean_y) / theory.mean_y * 100 << "%)\n";
            std::cout << "  Var[x(T)]:     " << std::setw(10) << stats.var_x
                      << " (error: " << std::setw(8) << (stats.var_x - theory.var_x) / theory.var_x * 100 << "%)\n";
            std::cout << "  Var[y(T)]:     " << std::setw(10) << stats.var_y
                      << " (error: " << std::setw(8) << (stats.var_y - theory.var_y) / theory.var_y * 100 << "%)\n";
            std::cout << "  Cov[x,y(T)]:   " << std::setw(10) << stats.cov_xy
                      << " (error: " << std::setw(8) << (stats.cov_xy - theory.cov_xy) / theory.cov_xy * 100 << "%)\n";
            std::cout << "  Time:          " << std::setw(10) << elapsed << " ms\n\n";
        }
    }

    std::cout << "========================================\n";
    std::cout << "=== Summary ===\n";
    std::cout << "========================================\n\n";

    std::cout << "Method Characteristics:\n\n";

    std::cout << "1. Euler-Maruyama:\n";
    std::cout << "   • Simplest method\n";
    std::cout << "   • O(√Δt) weak convergence\n";
    std::cout << "   • Requires small Δt for accuracy\n";
    std::cout << "   • Good for: Quick prototyping\n\n";

    std::cout << "2. Exponential Euler:\n";
    std::cout << "   • EXACT for OU processes\n";
    std::cout << "   • No discretization error\n";
    std::cout << "   • Unconditionally stable\n";
    std::cout << "   • Good for: Production G2++ simulation\n";
    std::cout << "   • ⭐ RECOMMENDED for G2++\n\n";

    std::cout << "3. Exact Simulation:\n";
    std::cout << "   • Same as Exponential Euler for OU\n";
    std::cout << "   • Uses analytical transition density\n";
    std::cout << "   • Good for: G2++, Vasicek, Gaussian models\n\n";

    std::cout << "4. Milstein:\n";
    std::cout << "   • Second-order for general SDEs\n";
    std::cout << "   • No advantage for OU (constant diffusion)\n";
    std::cout << "   • Good for: SDEs with state-dependent volatility (e.g., CIR, Heston)\n\n";

    std::cout << "5. Predictor-Corrector:\n";
    std::cout << "   • Better stability than Euler\n";
    std::cout << "   • 2x cost per step (two drift evaluations)\n";
    std::cout << "   • Good for: General SDEs, when stability is important\n\n";

    std::cout << "6. Implicit Euler:\n";
    std::cout << "   • Unconditionally stable\n";
    std::cout << "   • Good for stiff systems\n";
    std::cout << "   • Requires solving implicit equation (easy for linear OU)\n";
    std::cout << "   • Good for: Stiff SDEs, large time steps\n\n";

    std::cout << "========================================\n";
    std::cout << "For G2++: Use Exponential Euler (exact!)\n";
    std::cout << "For path refinement: Use exact bridge\n";
    std::cout << "========================================\n";
}

int main() {
    try {
        compareDiscretizationSchemes();
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }
}
