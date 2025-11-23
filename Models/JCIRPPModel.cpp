#include "JCIRPPModel.h"
#include <cmath>
#include <algorithm>
#include <stdexcept>
#include <random>

// ============================================================================
// JCIRPPParams Implementation
// ============================================================================

JCIRPPParams::JCIRPPParams(double x0, double kappa, double theta, double sigma,
                           double jumpIntensity, double jumpMean,
                           const std::vector<double> &shifts)
    : x0(x0), kappa(kappa), theta(theta), sigma(sigma),
      shifts(shifts), jumpIntensity(jumpIntensity), jumpMean(jumpMean) {
    if (x0 < 0.0) throw std::invalid_argument("Initial x value must be non-negative");
    if (kappa <= 0.0) throw std::invalid_argument("Mean reversion speed must be positive");
    if (theta < 0.0) throw std::invalid_argument("Long-term mean must be non-negative");
    if (sigma < 0.0) throw std::invalid_argument("Volatility must be non-negative");
    if (jumpIntensity < 0.0) throw std::invalid_argument("Jump intensity must be non-negative");
    if (jumpMean <= 0.0) throw std::invalid_argument("Jump mean must be positive");
}

// ============================================================================
// JCIRPPState Implementation
// ============================================================================

JCIRPPState::JCIRPPState(double intensity, double cumulativeIntensity, int totalJumps)
    : intensity(intensity), cumulativeIntensity(cumulativeIntensity), totalJumps(totalJumps) {
}

// ============================================================================
// JCIRPPModel Implementation
// ============================================================================

JCIRPPModel::JCIRPPModel(const JCIRPPParams &params)
    : m_params(params) {
}

void JCIRPPModel::update(JCIRPPState &current,
                         const JCIRPPState &previous,
                         size_t stepIndex,
                         double dt,
                         const std::vector<double> &dW,
                         unsigned int seed) const {
    if (dW.empty()) {
        throw std::invalid_argument("JCIR++ model requires at least 1 Brownian motion");
    }

    // Initial state
    if (stepIndex == 0) {
        // Get shift at time 0
        double shift = m_params.shifts.empty() ? 0.0 : m_params.shifts[0];
        current.intensity = m_params.x0 + shift;
        current.cumulativeIntensity = 0.0;
        current.totalJumps = 0;
        return;
    }

    // Get previous and current shift values (piecewise constant)
    double prevShift = (stepIndex - 1 < m_params.shifts.size()) ? m_params.shifts[stepIndex - 1] : 0.0;
    double currShift = (stepIndex < m_params.shifts.size()) ? m_params.shifts[stepIndex] : 0.0;

    // Extract x(t) from previous intensity: x = λ - φ
    double x_prev = previous.intensity - prevShift;

    // ========================================================================
    // CIR diffusion component: dx = κ[θ - x]dt + σ√x dW
    // ========================================================================
    double x = std::max(x_prev, 0.0);
    double dx_diffusion = m_params.kappa * (m_params.theta - x) * dt
                          + m_params.sigma * std::sqrt(x) * dW[0];

    // ========================================================================
    // Jump component: J·dN
    // ========================================================================
    // Generate Poisson jumps
    std::mt19937 gen(seed + stepIndex * 1000); // Seed based on step for reproducibility

    // Number of jumps in time interval dt follows Poisson(ν * dt)
    std::poisson_distribution<int> poissonDist(m_params.jumpIntensity * dt);
    int numJumps = poissonDist(gen);

    // Generate jump sizes (exponential distribution with mean jumpMean)
    std::exponential_distribution<double> expDist(1.0 / m_params.jumpMean);
    double totalJumpSize = 0.0;
    for (int i = 0; i < numJumps; ++i) {
        totalJumpSize += expDist(gen);
    }

    // ========================================================================
    // Combine all components
    // ========================================================================
    double x_curr = std::max(x + dx_diffusion + totalJumpSize, 0.0);

    // Add current shift to get intensity: λ(t) = x(t) + φ(t)
    current.intensity = x_curr + currShift;

    // Update cumulative intensity using trapezoidal rule
    // Note: This is approximate since jumps create discontinuities
    // For more accuracy, could use left-continuous or right-continuous integration
    double intensityIncrement = 0.5 * (previous.intensity + current.intensity) * dt;
    current.cumulativeIntensity = previous.cumulativeIntensity + intensityIncrement;

    // Update jump counter
    current.totalJumps = previous.totalJumps + numJumps;
}

// ============================================================================
// Credit Risk Helper Functions
// ============================================================================

namespace JCIRPP {
    // ============================================================================
    // Credit Risk Helper Functions
    // ============================================================================

    double calculateSurvivalProbability(
        const std::map<int, JCIRPPState> &path,
        int fromDay,
        int toDay) {
        auto fromIt = path.find(fromDay);
        auto toIt = path.find(toDay);

        if (fromIt == path.end() || toIt == path.end()) {
            throw std::out_of_range("Day not found in path");
        }

        if (fromDay > toDay) {
            throw std::invalid_argument("fromDay must be <= toDay");
        }

        // Survival probability: SP(t,T) = exp(-∫[t,T] λ(s) ds)
        double integralDifference = toIt->second.cumulativeIntensity - fromIt->second.cumulativeIntensity;
        return std::exp(-integralDifference);
    }

    double calculateDefaultProbability(
        const std::map<int, JCIRPPState> &path,
        int fromDay,
        int toDay) {
        return 1.0 - calculateSurvivalProbability(path, fromDay, toDay);
    }

    std::map<int, double> calculateSurvivalCurve(
        const std::map<int, JCIRPPState> &path,
        int fromDay,
        const std::vector<int> &tenorDays) {
        std::map<int, double> survivalCurve;

        for (int tenor: tenorDays) {
            int toDay = fromDay + tenor;
            if (path.find(toDay) != path.end()) {
                survivalCurve[tenor] = calculateSurvivalProbability(path, fromDay, toDay);
            }
        }

        return survivalCurve;
    }

    double calculateForwardSurvivalProbability(
        const std::map<int, JCIRPPState> &path,
        int observationDay,
        int fromDay,
        int toDay) {
        if (observationDay > fromDay || fromDay > toDay) {
            throw std::invalid_argument("Days must satisfy: observationDay <= fromDay <= toDay");
        }

        return calculateSurvivalProbability(path, fromDay, toDay);
    }

    double getHazardRate(
        const std::map<int, JCIRPPState> &path,
        int day) {
        auto it = path.find(day);
        if (it == path.end()) {
            throw std::out_of_range("Day not found in path");
        }

        return it->second.intensity;
    }

    double getAverageHazardRate(
        const std::map<int, JCIRPPState> &path,
        int fromDay,
        int toDay) {
        auto fromIt = path.find(fromDay);
        auto toIt = path.find(toDay);

        if (fromIt == path.end() || toIt == path.end()) {
            throw std::out_of_range("Day not found in path");
        }

        if (fromDay >= toDay) {
            throw std::invalid_argument("fromDay must be < toDay");
        }

        // Average hazard rate = ∫[t,T] λ(s) ds / (T - t)
        double integralDifference = toIt->second.cumulativeIntensity - fromIt->second.cumulativeIntensity;

        // Convert days to years for proper annualization
        const double DAYS_IN_YEAR = 365.25;
        double timeInYears = (toDay - fromDay) / DAYS_IN_YEAR;

        return integralDifference / timeInYears;
    }

    int getTotalJumps(
        const std::map<int, JCIRPPState> &path,
        int day) {
        auto it = path.find(day);
        if (it == path.end()) {
            throw std::out_of_range("Day not found in path");
        }

        return it->second.totalJumps;
    }

    int getJumpsInPeriod(
        const std::map<int, JCIRPPState> &path,
        int fromDay,
        int toDay) {
        auto fromIt = path.find(fromDay);
        auto toIt = path.find(toDay);

        if (fromIt == path.end() || toIt == path.end()) {
            throw std::out_of_range("Day not found in path");
        }

        if (fromDay > toDay) {
            throw std::invalid_argument("fromDay must be <= toDay");
        }

        return toIt->second.totalJumps - fromIt->second.totalJumps;
    }
} // namespace JCIRPP
