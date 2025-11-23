#include "MonteCarloSimulator.h"

#include <stdexcept>

const double DAYS_IN_YEAR = 365.25;

MonteCarloSimulator::MonteCarloSimulator(const std::vector<int>& scheduleDays,
                                         const std::vector<BrownianMotion>& brownianMotions)
    : m_scheduleDays(scheduleDays), m_brownianMotions(&brownianMotions) {
    convertDaysToYears();
}

void MonteCarloSimulator::convertDaysToYears() {
    m_scheduleYears.resize(m_scheduleDays.size());
    for (size_t i = 0; i < m_scheduleDays.size(); ++i) {
        m_scheduleYears[i] = m_scheduleDays[i] / DAYS_IN_YEAR;
    }
}

void MonteCarloSimulator::simulate(
    std::function<void(size_t, double, const std::vector<double>&)> updateStep) {
    if (m_brownianMotions->empty()) {
        throw std::runtime_error("No Brownian motions provided");
    }

    // Time stepping
    for (size_t i = 0; i < m_scheduleDays.size(); ++i) {
        double dt = (i == 0) ? 0.0 : (m_scheduleYears[i] - m_scheduleYears[i - 1]);
        int currentDay = m_scheduleDays[i];
        int previousDay = (i == 0) ? 0 : m_scheduleDays[i - 1];

        // Extract Brownian increments from pre-correlated Brownian motions
        std::vector<double> dW(m_brownianMotions->size());
        for (size_t j = 0; j < m_brownianMotions->size(); ++j) {
            if (i == 0) {
                dW[j] = 0.0;
            } else {
                dW[j] = (*m_brownianMotions)[j].getIncrement(previousDay, currentDay);
            }
        }

        // Call user-provided update function
        updateStep(i, dt, dW);
    }
}

const std::vector<int>& MonteCarloSimulator::getScheduleDays() const {
    return m_scheduleDays;
}

const std::vector<double>& MonteCarloSimulator::getScheduleYears() const {
    return m_scheduleYears;
}
