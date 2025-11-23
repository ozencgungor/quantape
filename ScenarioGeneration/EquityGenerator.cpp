#include "EquityGenerator.h"
#include <stdexcept>

EquityGenerator::EquityGenerator(EquitySimulator &simulator)
    : ModelGenerator(simulator.getScheduleDays()),
      m_simulator(&simulator) {
}

void EquityGenerator::ensureSimulated(const std::string &name) {
    if (!m_simulator->isSimulated(name)) {
        m_simulator->simulateEquity(name);
    }
}

double EquityGenerator::getSpot(const std::string &name, int day) {
    ensureSimulated(name);
    const auto &path = m_simulator->getEquityPath(name);
    auto it = path.find(day);
    if (it == path.end()) {
        throw std::out_of_range("Day not found in equity path");
    }
    return it->second.spot;
}

double EquityGenerator::getVariance(const std::string &name, int day) {
    ensureSimulated(name);
    const auto &path = m_simulator->getEquityPath(name);
    auto it = path.find(day);
    if (it == path.end()) {
        throw std::out_of_range("Day not found in equity path");
    }
    return it->second.variance;
}

HestonState EquityGenerator::getState(const std::string &name, int day) {
    ensureSimulated(name);
    const auto &path = m_simulator->getEquityPath(name);
    auto it = path.find(day);
    if (it == path.end()) {
        throw std::out_of_range("Day not found in equity path");
    }
    return it->second;
}

const std::map<int, HestonState> &EquityGenerator::getPath(const std::string &name) {
    ensureSimulated(name);
    return m_simulator->getEquityPath(name);
}

std::vector<std::string> EquityGenerator::getEquityNames() const {
    return m_simulator->getEquityNames();
}
