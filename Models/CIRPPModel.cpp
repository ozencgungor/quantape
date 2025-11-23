#include "CIRPPModel.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>

// ============================================================================
// CIRPPParams Implementation
// ============================================================================

CIRPPParams::CIRPPParams(double x0, double kappa, double theta, double sigma,
                         const std::vector<double>& shifts)
    : x0(x0), kappa(kappa), theta(theta), sigma(sigma), shifts(shifts) {
    if (x0 < 0.0)
        throw std::invalid_argument("Initial x value must be non-negative");
    if (kappa <= 0.0)
        throw std::invalid_argument("Mean reversion speed must be positive");
    if (theta < 0.0)
        throw std::invalid_argument("Long-term mean must be non-negative");
    if (sigma < 0.0)
        throw std::invalid_argument("Volatility must be non-negative");
}

// ============================================================================
// CIRPPState Implementation
// ============================================================================

CIRPPState::CIRPPState(double intensity, double cumulativeIntensity)
    : intensity(intensity), cumulativeIntensity(cumulativeIntensity) {}

// ============================================================================
// CIRPPModel Implementation
// ============================================================================

CIRPPModel::CIRPPModel(const CIRPPParams& params) : m_params(params) {}

void CIRPPModel::update(CIRPPState& current, const CIRPPState& previous, size_t stepIndex,
                        double dt, const std::vector<double>& dW) const {
    if (dW.empty()) {
        throw std::invalid_argument("CIR++ model requires at least 1 Brownian motion");
    }

    // Initial state
    if (stepIndex == 0) {
        // Get shift at time 0
        double shift = m_params.shifts.empty() ? 0.0 : m_params.shifts[0];
        current.intensity = m_params.x0 + shift;
        current.cumulativeIntensity = 0.0;
        return;
    }

    // Get previous and current shift values (piecewise constant)
    double prevShift =
        (stepIndex - 1 < m_params.shifts.size()) ? m_params.shifts[stepIndex - 1] : 0.0;
    double currShift = (stepIndex < m_params.shifts.size()) ? m_params.shifts[stepIndex] : 0.0;

    // Extract x(t) from previous intensity: x = λ - φ
    double x_prev = previous.intensity - prevShift;

    // CIR dynamics for x(t): dx = κ[θ - x]dt + σ√x dW
    // Truncate to ensure non-negativity
    double x = std::max(x_prev, 0.0);
    double dx = m_params.kappa * (m_params.theta - x) * dt + m_params.sigma * std::sqrt(x) * dW[0];
    double x_curr = std::max(x + dx, 0.0);

    // Add current shift to get intensity: λ(t) = x(t) + φ(t)
    current.intensity = x_curr + currShift;

    // Update cumulative intensity using trapezoidal rule
    // ∫[t_{i-1}, t_i] λ(s) ds ≈ (λ_{i-1} + λ_i) * dt / 2
    double intensityIncrement = 0.5 * (previous.intensity + current.intensity) * dt;
    current.cumulativeIntensity = previous.cumulativeIntensity + intensityIncrement;
}

// ============================================================================
// Credit Risk Helper Functions
// ============================================================================

namespace CIRPP {
// ============================================================================
// Credit Risk Helper Functions
// ============================================================================

double calculateSurvivalProbability(const std::map<int, CIRPPState>& path, int fromDay, int toDay) {
    auto fromIt = path.find(fromDay);
    auto toIt = path.find(toDay);

    if (fromIt == path.end() || toIt == path.end()) {
        throw std::out_of_range("Day not found in path");
    }

    if (fromDay > toDay) {
        throw std::invalid_argument("fromDay must be <= toDay");
    }

    // Survival probability: SP(t,T) = exp(-∫[t,T] λ(s) ds)
    // = exp(-(CumulativeIntensity[T] - CumulativeIntensity[t]))
    double integralDifference =
        toIt->second.cumulativeIntensity - fromIt->second.cumulativeIntensity;
    return std::exp(-integralDifference);
}

double calculateDefaultProbability(const std::map<int, CIRPPState>& path, int fromDay, int toDay) {
    return 1.0 - calculateSurvivalProbability(path, fromDay, toDay);
}

std::map<int, double> calculateSurvivalCurve(const std::map<int, CIRPPState>& path, int fromDay,
                                             const std::vector<int>& tenorDays) {
    std::map<int, double> survivalCurve;

    for (int tenor : tenorDays) {
        int toDay = fromDay + tenor;
        if (path.find(toDay) != path.end()) {
            survivalCurve[tenor] = calculateSurvivalProbability(path, fromDay, toDay);
        }
    }

    return survivalCurve;
}

double calculateForwardSurvivalProbability(const std::map<int, CIRPPState>& path,
                                           int observationDay, int fromDay, int toDay) {
    // Forward survival probability given information at observationDay
    // P(τ > T2 | τ > T1, F_{T0}) = P(τ > T2 | F_{T0}) / P(τ > T1 | F_{T0})
    // Since we're on a path (no conditioning needed for simulation):
    // = exp(-∫[T1,T2] λ(s) ds)

    if (observationDay > fromDay || fromDay > toDay) {
        throw std::invalid_argument("Days must satisfy: observationDay <= fromDay <= toDay");
    }

    return calculateSurvivalProbability(path, fromDay, toDay);
}

double getHazardRate(const std::map<int, CIRPPState>& path, int day) {
    auto it = path.find(day);
    if (it == path.end()) {
        throw std::out_of_range("Day not found in path");
    }

    return it->second.intensity;
}

double getAverageHazardRate(const std::map<int, CIRPPState>& path, int fromDay, int toDay) {
    auto fromIt = path.find(fromDay);
    auto toIt = path.find(toDay);

    if (fromIt == path.end() || toIt == path.end()) {
        throw std::out_of_range("Day not found in path");
    }

    if (fromDay >= toDay) {
        throw std::invalid_argument("fromDay must be < toDay");
    }

    // Average hazard rate = ∫[t,T] λ(s) ds / (T - t)
    double integralDifference =
        toIt->second.cumulativeIntensity - fromIt->second.cumulativeIntensity;

    // Convert days to years for proper annualization
    const double DAYS_IN_YEAR = 365.25;
    double timeInYears = (toDay - fromDay) / DAYS_IN_YEAR;

    return integralDifference / timeInYears;
}
} // namespace CIRPP
