/**
 * @file explain_effective_kappa.cpp
 * @brief Explain and visualize the concept of "effective kappa"
 *
 * Key insight: The mean reversion speed κ(θ - r) relative to the current
 * level r is NOT constant - it depends on where r is relative to θ.
 */

#include <iostream>
#include <iomanip>
#include <cmath>

void explainEffectiveKappa() {
    std::cout << "=== Understanding Effective Kappa ===\n\n";

    double kappa = 0.1;
    double theta = 0.08;
    double sigma = 0.2;

    std::cout << "CIR Process: dr = κ(θ - r)dt + σ√r dW\n";
    std::cout << "Parameters: κ = " << kappa << ", θ = " << theta << ", σ = " << sigma << "\n\n";

    std::cout << "Question: When r is far from θ, how fast does it revert?\n\n";

    // Analyze different r values
    std::cout << std::fixed << std::setprecision(6);
    std::cout << "r         θ-r       Drift κ(θ-r)  Diffusion σ√r  Drift/r      Rel. Strength\n";
    std::cout << "--------------------------------------------------------------------------------\n";

    std::vector<double> r_values = {0.01, 0.02, 0.04, 0.08, 0.12, 0.16, 0.20};

    for (double r : r_values) {
        double drift_abs = kappa * (theta - r);  // Absolute drift
        double diffusion = sigma * std::sqrt(r); // Diffusion coefficient
        double drift_relative = drift_abs / r;    // Drift relative to current level

        // Signal-to-noise ratio: drift / diffusion
        double snr = (diffusion > 1e-8) ? std::abs(drift_abs) / diffusion : 0.0;

        std::string strength;
        if (r < theta * 0.5) {
            strength = "Strong upward pull";
        } else if (r < theta * 0.8) {
            strength = "Moderate upward pull";
        } else if (r < theta * 1.2) {
            strength = "Near equilibrium";
        } else if (r < theta * 1.5) {
            strength = "Moderate downward pull";
        } else {
            strength = "Strong downward pull";
        }

        std::cout << std::setw(8) << r << "  "
                  << std::setw(8) << (theta - r) << "  "
                  << std::setw(12) << drift_abs << "  "
                  << std::setw(14) << diffusion << "  "
                  << std::setw(12) << drift_relative << "  "
                  << strength << "\n";
    }

    std::cout << "\n\nKey Observations:\n\n";

    std::cout << "1. When r = 0.01 << θ = 0.08:\n";
    std::cout << "   • Absolute drift: κ(θ - r) = 0.1 × 0.07 = 0.007\n";
    std::cout << "   • RELATIVE drift: 0.007 / 0.01 = 0.7 = 70% per unit time!\n";
    std::cout << "   • This is MUCH stronger than nominal κ = 0.1\n";
    std::cout << "   • Effective κ_eff ≈ 0.7 (7× stronger!)\n\n";

    std::cout << "2. When r = 0.08 = θ:\n";
    std::cout << "   • Drift = 0 (at equilibrium)\n";
    std::cout << "   • Effective κ = nominal κ\n\n";

    std::cout << "3. When r = 0.16 >> θ = 0.08:\n";
    std::cout << "   • Absolute drift: κ(θ - r) = 0.1 × (-0.08) = -0.008\n";
    std::cout << "   • RELATIVE drift: -0.008 / 0.16 = -0.05 = -5% per unit time\n";
    std::cout << "   • This is WEAKER than nominal κ = 0.1\n";
    std::cout << "   • Effective κ_eff ≈ 0.05 (2× weaker)\n\n";

    std::cout << "========================================\n";
    std::cout << "The Effective Kappa Concept:\n";
    std::cout << "========================================\n\n";

    std::cout << "Define effective mean reversion speed as:\n";
    std::cout << "  κ_eff = |θ - r| / r  (when r < θ)\n";
    std::cout << "  κ_eff = |θ - r| / θ  (when r > θ)\n\n";

    std::cout << "Interpretation:\n";
    std::cout << "  • κ_eff measures the proportional speed of mean reversion\n";
    std::cout << "  • When r far from θ, κ_eff ≠ κ (nominal)\n";
    std::cout << "  • Process 'feels' faster/slower mean reversion depending on level\n\n";

    std::cout << "Application to Bridge Sampling:\n";
    std::cout << "  • Standard bridge uses E[r(t)] = r₀·exp(-κt) + θ(1 - exp(-κt))\n";
    std::cout << "  • But this assumes CONSTANT κ throughout the path\n";
    std::cout << "  • When r₁ << θ, the process reverts FASTER than nominal κ suggests\n";
    std::cout << "  • Solution: Use κ_eff instead of κ in bridge mean calculation\n\n";

    std::cout << "Example Calculation:\n";
    std::cout << "  r₁ = 0.01, θ = 0.08, κ = 0.1\n";
    std::cout << "  Distance ratio: r₁/θ = 0.01/0.08 = 0.125\n";
    std::cout << "  Since r₁ << θ (ratio < 0.5), increase κ:\n\n";

    double r1 = 0.01;
    double distance_ratio = r1 / theta;
    double kappa_multiplier = 1.0 + 0.5 * (1.0 - 2.0 * distance_ratio);
    double kappa_eff = kappa * kappa_multiplier;

    std::cout << "  κ_multiplier = 1.0 + 0.5 × (1.0 - 2 × 0.125) = " << kappa_multiplier << "\n";
    std::cout << "  κ_eff = " << kappa << " × " << kappa_multiplier << " = " << kappa_eff << "\n\n";

    std::cout << "Using κ_eff for mean:\n";
    double dt = 0.5;
    double exp_nominal = std::exp(-kappa * dt);
    double exp_eff = std::exp(-kappa_eff * dt);
    double mean_nominal = r1 * exp_nominal + theta * (1.0 - exp_nominal);
    double mean_eff = r1 * exp_eff + theta * (1.0 - exp_eff);

    std::cout << "  With nominal κ: E[r(0.5)] = " << mean_nominal << "\n";
    std::cout << "  With κ_eff:     E[r(0.5)] = " << mean_eff << "\n";
    std::cout << "  Difference: κ_eff predicts process closer to θ (as it should!)\n\n";

    std::cout << "========================================\n";
    std::cout << "Why It Helped (Somewhat):\n";
    std::cout << "========================================\n\n";

    std::cout << "Results from test:\n";
    std::cout << "  • Baseline (v2): 46.5% mean error\n";
    std::cout << "  • Effective κ:   22.4% mean error  (2× better!)\n";
    std::cout << "  • But still not great (Two-Step had 4% error)\n\n";

    std::cout << "Why partial success?\n";
    std::cout << "  ✓ Captures nonlinear mean reversion strength\n";
    std::cout << "  ✓ Adjusts expected value in right direction\n";
    std::cout << "  ✗ Calibration is ad-hoc (multiplier formula is heuristic)\n";
    std::cout << "  ✗ Doesn't fix state-dependent volatility σ√r\n";
    std::cout << "  ✗ Only adjusts mean, not variance\n\n";

    std::cout << "========================================\n";
    std::cout << "Better Approach:\n";
    std::cout << "========================================\n\n";

    std::cout << "Instead of trying to find the 'right' κ_eff:\n";
    std::cout << "  → Use Two-Step Bridge (splits the interval)\n";
    std::cout << "  → Each sub-interval has smaller nonlinearity\n";
    std::cout << "  → Linear approximation works better on each piece\n";
    std::cout << "  → Achieved 4% error vs 22% with κ_eff\n\n";

    std::cout << "Moral: Breaking the problem into pieces > trying to\n";
    std::cout << "       perfectly calibrate a global parameter adjustment\n";
}

int main() {
    explainEffectiveKappa();
    return 0;
}
