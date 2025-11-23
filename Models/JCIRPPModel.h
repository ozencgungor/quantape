#ifndef JCIRPP_MODEL_H
#define JCIRPP_MODEL_H

#include <vector>
#include <map>

// JCIR++ (Jump CIR Plus Plus) model parameters (pure data)
// Models intensity: λ(t) = x(t) + φ(t)
// where x(t) follows: dx = κ[θ - x]dt + σ√x dW + J·dN
// - CIR diffusion component
// - Deterministic shift φ(t) for calibration
// - Jump component: Poisson process N(t) with jump sizes J
struct JCIRPPParams {
    double x0; // Initial value of x(t)
    double kappa; // Mean reversion speed
    double theta; // Long-term mean of x(t)
    double sigma; // Volatility
    std::vector<double> shifts; // φ(t) at each schedule point (piecewise constant)
    double jumpIntensity; // Poisson jump intensity ν (jumps per year)
    double jumpMean; // Mean jump size (exponential distribution)

    JCIRPPParams(double x0, double kappa, double theta, double sigma,
                 double jumpIntensity, double jumpMean,
                 const std::vector<double> &shifts = {});
};

// JCIR++ model state (single time point)
struct JCIRPPState {
    double intensity; // λ(t) = x(t) + φ(t)
    double cumulativeIntensity; // ∫[0,t] λ(s) ds (for survival probability calculation)
    int totalJumps; // Cumulative number of jumps up to time t (diagnostic)

    JCIRPPState(double intensity = 0.0, double cumulativeIntensity = 0.0, int totalJumps = 0);
};

// JCIR++ model class
class JCIRPPModel {
public:
    // Constructor takes params struct
    explicit JCIRPPModel(const JCIRPPParams &params);

    // Update function for JCIR++ model
    // Note: Requires a seed for Poisson/exponential random generation
    void update(JCIRPPState &current,
                const JCIRPPState &previous,
                size_t stepIndex,
                double dt,
                const std::vector<double> &dW,
                unsigned int seed) const;

    // Access to parameters
    const JCIRPPParams &getParams() const { return m_params; }

private:
    JCIRPPParams m_params;
};

// Namespace for JCIR++ helper functions
namespace JCIRPP {
    // ========================================================================
    // Credit Risk Helper Functions
    // ========================================================================
    // (Same as CIR++ but adapted for JCIRPPState)

    // Calculate survival probability from day t to day T
    // SP(t,T) = P(τ > T | τ > t) = exp(-∫[t,T] λ(s) ds)
    double calculateSurvivalProbability(
        const std::map<int, JCIRPPState> &path,
        int fromDay,
        int toDay);

    // Calculate default probability from day t to day T
    // DP(t,T) = 1 - SP(t,T)
    double calculateDefaultProbability(
        const std::map<int, JCIRPPState> &path,
        int fromDay,
        int toDay);

    // Calculate term structure of survival probabilities from a given day
    std::map<int, double> calculateSurvivalCurve(
        const std::map<int, JCIRPPState> &path,
        int fromDay,
        const std::vector<int> &tenorDays);

    // Calculate forward survival probability
    double calculateForwardSurvivalProbability(
        const std::map<int, JCIRPPState> &path,
        int observationDay,
        int fromDay,
        int toDay);

    // Calculate instantaneous hazard rate at a given day
    double getHazardRate(
        const std::map<int, JCIRPPState> &path,
        int day);

    // Calculate average hazard rate over a period
    double getAverageHazardRate(
        const std::map<int, JCIRPPState> &path,
        int fromDay,
        int toDay);

    // Get total number of jumps up to a given day
    int getTotalJumps(
        const std::map<int, JCIRPPState> &path,
        int day);

    // Get number of jumps in a period
    int getJumpsInPeriod(
        const std::map<int, JCIRPPState> &path,
        int fromDay,
        int toDay);
}

#endif // JCIRPP_MODEL_H
