#include "CIRModel.h"
#include <cmath>
#include <algorithm>
#include <stdexcept>

// ============================================================================
// CIRParams Implementation
// ============================================================================

CIRParams::CIRParams(double r0, double kappa, double theta, double sigma)
    : r0(r0), kappa(kappa), theta(theta), sigma(sigma) {
    if (r0 < 0.0) throw std::invalid_argument("Initial value must be non-negative");
    if (kappa <= 0.0) throw std::invalid_argument("Mean reversion speed must be positive");
    if (theta < 0.0) throw std::invalid_argument("Long-term mean must be non-negative");
    if (sigma < 0.0) throw std::invalid_argument("Volatility must be non-negative");
}

// ============================================================================
// CIRState Implementation
// ============================================================================

CIRState::CIRState(double value) : value(value) {
}

// ============================================================================
// CIRModel Implementation
// ============================================================================

CIRModel::CIRModel(const CIRParams &params)
    : m_params(params) {
}

void CIRModel::update(CIRState &current,
                      const CIRState &previous,
                      size_t stepIndex,
                      double dt,
                      const std::vector<double> &dW) const {
    if (dW.empty()) {
        throw std::invalid_argument("CIR model requires at least 1 Brownian motion");
    }

    // Initial state
    if (stepIndex == 0) {
        current.value = m_params.r0;
        return;
    }

    // CIR dynamics: dr = kappa*(theta - r)*dt + sigma*sqrt(r)*dW
    // Truncate to ensure non-negativity
    double r = std::max(previous.value, 0.0);
    double dr = m_params.kappa * (m_params.theta - r) * dt
                + m_params.sigma * std::sqrt(r) * dW[0];
    current.value = std::max(r + dr, 0.0);
}
