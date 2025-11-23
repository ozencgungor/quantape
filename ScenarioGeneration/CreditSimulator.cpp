#include "CreditSimulator.h"

#include <stdexcept>

#include "MonteCarloSimulator.h"

CreditSimulator::CreditSimulator(const std::vector<int>& scheduleDays, unsigned int seed)
    : ModelSimulator(scheduleDays, seed, true) {}

CreditSimulator::CreditSimulator(const std::vector<int>& scheduleDays)
    : ModelSimulator(scheduleDays, 0, false) {}

void CreditSimulator::addCredit(const std::string& name, const CIRParams& params) {
    if (m_creditModels.find(name) != m_creditModels.end()) {
        throw std::invalid_argument("Credit '" + name + "' already exists");
    }
    m_creditModels.insert({name, CIRModel(params)});
}

void CreditSimulator::simulateCredit(const std::string& name) {
    // Check if already simulated
    if (m_simulatedCredits.find(name) != m_simulatedCredits.end()) {
        return; // Already simulated
    }

    // Check if credit exists
    auto it = m_creditModels.find(name);
    if (it == m_creditModels.end()) {
        throw std::invalid_argument("Credit '" + name + "' not found");
    }

    const CIRModel& model = it->second;
    size_t creditIndex = std::distance(m_creditModels.begin(), it);

    // Generate single independent Brownian motion for this credit (CIR is 1-factor)
    unsigned int seed = m_useSeed ? m_seed + creditIndex * 1000 : std::random_device{}();
    std::vector<BrownianMotion> brownianMotions =
        generateIndependentBrownianMotions(m_scheduleDays, 1, seed);

    // Create Monte Carlo simulator
    MonteCarloSimulator mcSim(m_scheduleDays, brownianMotions);

    // Storage for this credit's path
    std::map<int, CIRState>& path = m_creditPaths[name];
    std::vector<CIRState> states(m_scheduleDays.size());

    // Define update callback
    auto updateStep = [&](size_t i, double dt, const std::vector<double>& dW) {
        CIRState previous = (i == 0) ? CIRState() : states[i - 1];
        model.update(states[i], previous, i, dt, dW);
        path[m_scheduleDays[i]] = states[i];
    };

    // Run simulation
    mcSim.simulate(updateStep);

    // Mark as simulated
    m_simulatedCredits.insert(name);
}

void CreditSimulator::generateBrownianMotions() {
    // Not needed for lazy evaluation, but kept for interface compatibility
    // Individual credits generate their own Brownian motions on-demand
}

void CreditSimulator::simulate() {
    // Simulate all credits
    for (const auto& entry : m_creditModels) {
        simulateCredit(entry.first);
    }
}

void CreditSimulator::regenerate(unsigned int newSeed) {
    m_seed = newSeed;
    m_simulatedCredits.clear();
    m_creditPaths.clear();
    simulate();
}

const std::map<int, CIRState>& CreditSimulator::getCreditPath(const std::string& name) const {
    auto it = m_creditPaths.find(name);
    if (it == m_creditPaths.end()) {
        throw std::invalid_argument("Credit '" + name + "' has not been simulated");
    }
    return it->second;
}

std::vector<std::string> CreditSimulator::getCreditNames() const {
    std::vector<std::string> names;
    for (const auto& entry : m_creditModels) {
        names.push_back(entry.first);
    }
    return names;
}

bool CreditSimulator::isSimulated(const std::string& name) const {
    return m_simulatedCredits.find(name) != m_simulatedCredits.end();
}
