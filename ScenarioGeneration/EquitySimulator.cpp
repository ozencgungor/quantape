#include "EquitySimulator.h"

#include <stdexcept>

#include "MonteCarloSimulator.h"

EquitySimulator::EquitySimulator(const std::vector<int>& scheduleDays, unsigned int seed)
    : ModelSimulator(scheduleDays, seed, true) {}

EquitySimulator::EquitySimulator(const std::vector<int>& scheduleDays)
    : ModelSimulator(scheduleDays, 0, false) {}

void EquitySimulator::addEquity(const std::string& name, const HestonParams& params) {
    if (m_equityModels.find(name) != m_equityModels.end()) {
        throw std::invalid_argument("Equity '" + name + "' already exists");
    }
    m_equityModels.insert({name, HestonModel(params)});
}

void EquitySimulator::simulateEquity(const std::string& name) {
    // Check if already simulated
    if (m_simulatedEquities.find(name) != m_simulatedEquities.end()) {
        return; // Already simulated
    }

    // Check if equity exists
    auto it = m_equityModels.find(name);
    if (it == m_equityModels.end()) {
        throw std::invalid_argument("Equity '" + name + "' not found");
    }

    const HestonModel& model = it->second;
    size_t equityIndex = std::distance(m_equityModels.begin(), it);

    // Create correlation matrix for Heston (spot and variance)
    std::vector<std::vector<double>> corrMatrix = {{1.0, model.getParams().rho},
                                                   {model.getParams().rho, 1.0}};

    // Generate correlated Brownian motions for this equity
    unsigned int seed = m_useSeed ? m_seed + equityIndex * 1000 : std::random_device{}();
    std::vector<BrownianMotion> correlatedBMs =
        generateCorrelatedBrownianMotions(m_scheduleDays, 2, corrMatrix, seed);

    // Create Monte Carlo simulator with pre-correlated BMs
    MonteCarloSimulator mcSim(m_scheduleDays, correlatedBMs);

    // Storage for this equity's path
    std::map<int, HestonState>& path = m_equityPaths[name];
    std::vector<HestonState> states(m_scheduleDays.size());

    // Define update callback
    auto updateStep = [&](size_t i, double dt, const std::vector<double>& dW) {
        HestonState previous = (i == 0) ? HestonState() : states[i - 1];
        model.update(states[i], previous, i, dt, dW);
        path[m_scheduleDays[i]] = states[i];
    };

    // Run simulation
    mcSim.simulate(updateStep);

    // Mark as simulated
    m_simulatedEquities.insert(name);
}

void EquitySimulator::generateBrownianMotions() {
    // Not needed for lazy evaluation, but kept for interface compatibility
    // Individual equities generate their own Brownian motions on-demand
}

void EquitySimulator::simulate() {
    // Simulate all equities
    for (const auto& entry : m_equityModels) {
        simulateEquity(entry.first);
    }
}

void EquitySimulator::regenerate(unsigned int newSeed) {
    m_seed = newSeed;
    m_simulatedEquities.clear();
    m_equityPaths.clear();
    simulate();
}

const std::map<int, HestonState>& EquitySimulator::getEquityPath(const std::string& name) const {
    auto it = m_equityPaths.find(name);
    if (it == m_equityPaths.end()) {
        throw std::invalid_argument("Equity '" + name + "' has not been simulated");
    }
    return it->second;
}

std::vector<std::string> EquitySimulator::getEquityNames() const {
    std::vector<std::string> names;
    for (const auto& entry : m_equityModels) {
        names.push_back(entry.first);
    }
    return names;
}

bool EquitySimulator::isSimulated(const std::string& name) const {
    return m_simulatedEquities.find(name) != m_simulatedEquities.end();
}
