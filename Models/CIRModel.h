#ifndef CIR_MODEL_H
#define CIR_MODEL_H

#include <vector>

// CIR (Cox-Ingersoll-Ross) model parameters (pure data)
struct CIRParams {
    double r0; // Initial value
    double kappa; // Mean reversion speed
    double theta; // Long-term mean
    double sigma; // Volatility

    CIRParams(double r0, double kappa, double theta, double sigma);
};

// CIR model state (single time point)
struct CIRState {
    double value;

    CIRState(double value = 0.0);
};

// CIR model class
class CIRModel {
public:
    // Constructor takes params struct
    explicit CIRModel(const CIRParams &params);

    // Update function for CIR model
    void update(CIRState &current,
                const CIRState &previous,
                size_t stepIndex,
                double dt,
                const std::vector<double> &dW) const;

    // Access to parameters
    const CIRParams &getParams() const { return m_params; }

private:
    CIRParams m_params;
};

#endif // CIR_MODEL_H
