#ifndef GBM_MODEL_H
#define GBM_MODEL_H

#include <cmath>
#include <type_traits>
#include <vector>

// Geometric Brownian Motion (GBM) model - templated on MarketObject type
// MarketObject can be a scalar (double), vector, matrix, or custom class
// All elements evolve with the SAME Brownian motion

template <typename MarketObject>
struct GBMParams {
    MarketObject initialValue; // Can be any type
    double drift;              // mu (scalar)
    double volatility;         // sigma (scalar)

    GBMParams(const MarketObject& initialValue, double drift, double volatility)
        : initialValue(initialValue), drift(drift), volatility(volatility) {}
};

template <typename MarketObject>
struct GBMState {
    MarketObject value;

    GBMState() = default;

    GBMState(const MarketObject& value) : value(value) {}
};

// Namespace for GBM model functions
namespace GBM {
// Generic update function for any MarketObject type
// Evolution: S_t = S_0 * exp((mu - sigma^2/2)*t + sigma*W_t)
// Discrete form: S_new = S_old * exp((mu - sigma^2/2)*dt + sigma*dW)
template <typename MarketObject>
inline void updateGBM(GBMState<MarketObject>& current, const GBMState<MarketObject>& previous,
                      size_t stepIndex, double dt, const std::vector<double>& dW,
                      const GBMParams<MarketObject>& params) {
    if (dW.empty()) {
        throw std::invalid_argument("GBM model requires at least 1 Brownian motion");
    }

    // Initial state
    if (stepIndex == 0) {
        current.value = params.initialValue;
        return;
    }

    // GBM exact solution between time steps:
    // S_new = S_old * exp((mu - sigma^2/2)*dt + sigma*dW)
    // All elements use the SAME Brownian motion dW[0]
    double drift_correction = (params.drift - 0.5 * params.volatility * params.volatility) * dt;
    double diffusion = params.volatility * dW[0];
    double factor = std::exp(drift_correction + diffusion);
    current.value = previous.value * factor;
}
} // namespace GBM

#endif // GBM_MODEL_H
