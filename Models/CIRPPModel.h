#ifndef CIRPP_MODEL_H
#define CIRPP_MODEL_H

#include <map>
#include <vector>

// CIR++ (Cox-Ingersoll-Ross Plus Plus) model parameters (pure data)
// Models intensity: λ(t) = x(t) + φ(t)
// where x(t) follows CIR: dx = κ[θ - x]dt + σ√x dW
// and φ(t) is a deterministic shift function for calibration
struct CIRPPParams {
    double x0;                  // Initial value of x(t)
    double kappa;               // Mean reversion speed
    double theta;               // Long-term mean of x(t)
    double sigma;               // Volatility
    std::vector<double> shifts; // φ(t) at each schedule point (piecewise constant)

    CIRPPParams(double x0, double kappa, double theta, double sigma,
                const std::vector<double>& shifts = {});
};

// CIR++ model state (single time point)
// Stores both intensity and cumulative intensity for survival probability
struct CIRPPState {
    double intensity;           // λ(t) = x(t) + φ(t)
    double cumulativeIntensity; // ∫[0,t] λ(s) ds (for survival probability calculation)

    CIRPPState(double intensity = 0.0, double cumulativeIntensity = 0.0);
};

// CIR++ model class
class CIRPPModel {
public:
    // Constructor takes params struct
    explicit CIRPPModel(const CIRPPParams& params);

    // Update function for CIR++ model
    void update(CIRPPState& current, const CIRPPState& previous, size_t stepIndex, double dt,
                const std::vector<double>& dW) const;

    // Access to parameters
    const CIRPPParams& getParams() const { return m_params; }

private:
    CIRPPParams m_params;
};

// Namespace for CIR++ helper functions
namespace CIRPP {
// ========================================================================
// Credit Risk Helper Functions
// ========================================================================

// Calculate survival probability from day t to day T
// SP(t,T) = P(τ > T | τ > t) = exp(-∫[t,T] λ(s) ds)
double calculateSurvivalProbability(const std::map<int, CIRPPState>& path, int fromDay, int toDay);

// Calculate default probability from day t to day T
// DP(t,T) = 1 - SP(t,T)
double calculateDefaultProbability(const std::map<int, CIRPPState>& path, int fromDay, int toDay);

// Calculate term structure of survival probabilities from a given day
// Returns map: tenor (in days from fromDay) -> survival probability
std::map<int, double> calculateSurvivalCurve(const std::map<int, CIRPPState>& path, int fromDay,
                                             const std::vector<int>& tenorDays);

// Calculate forward survival probability: P(τ > T2 | τ > T1, FiltrationT0)
// This is the probability of surviving from T1 to T2, given information at T0
double calculateForwardSurvivalProbability(const std::map<int, CIRPPState>& path,
                                           int observationDay, int fromDay, int toDay);

// Calculate instantaneous hazard rate at a given day
double getHazardRate(const std::map<int, CIRPPState>& path, int day);

// Calculate average hazard rate over a period
double getAverageHazardRate(const std::map<int, CIRPPState>& path, int fromDay, int toDay);
} // namespace CIRPP

#endif // CIRPP_MODEL_H
