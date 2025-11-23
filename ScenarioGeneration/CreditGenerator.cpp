#include "CreditGenerator.h"
#include <stdexcept>

CreditGenerator::CreditGenerator(CreditSimulator &simulator)
    : ModelGenerator(simulator.getScheduleDays()),
      m_simulator(&simulator) {
}

void CreditGenerator::ensureSimulated(const std::string &name) {
    if (!m_simulator->isSimulated(name)) {
        m_simulator->simulateCredit(name);
    }
}

double CreditGenerator::getValue(const std::string &name, int day) {
    ensureSimulated(name);
    const auto &path = m_simulator->getCreditPath(name);
    auto it = path.find(day);
    if (it == path.end()) {
        throw std::out_of_range("Day not found in credit path");
    }
    return it->second.value;
}

CIRState CreditGenerator::getState(const std::string &name, int day) {
    ensureSimulated(name);
    const auto &path = m_simulator->getCreditPath(name);
    auto it = path.find(day);
    if (it == path.end()) {
        throw std::out_of_range("Day not found in credit path");
    }
    return it->second;
}

const std::map<int, CIRState> &CreditGenerator::getPath(const std::string &name) {
    ensureSimulated(name);
    return m_simulator->getCreditPath(name);
}

std::vector<std::string> CreditGenerator::getCreditNames() const {
    return m_simulator->getCreditNames();
}
