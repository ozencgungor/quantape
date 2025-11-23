#ifndef MONTE_CARLO_SIMULATOR_H
#define MONTE_CARLO_SIMULATOR_H

#include <functional>
#include <vector>

#include "BrownianMotion.h"

class MonteCarloSimulator {
public:
    // Constructor takes schedule and pre-generated Brownian motions
    MonteCarloSimulator(const std::vector<int>& scheduleDays,
                        const std::vector<BrownianMotion>& brownianMotions);

    // Simulate with a generic update function
    // updateStep signature: void(size_t stepIndex, double dt, const std::vector<double>& dW)
    void simulate(std::function<void(size_t, double, const std::vector<double>&)> updateStep);

    const std::vector<int>& getScheduleDays() const;

    const std::vector<double>& getScheduleYears() const;

private:
    std::vector<int> m_scheduleDays;
    std::vector<double> m_scheduleYears;
    const std::vector<BrownianMotion>* m_brownianMotions;

    void convertDaysToYears();
};

#endif // MONTE_CARLO_SIMULATOR_H
