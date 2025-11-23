#ifndef MODEL_SIMULATOR_H
#define MODEL_SIMULATOR_H

#include "BrownianMotion.h"
#include <vector>

// Abstract base class for all model simulators
class ModelSimulator {
public:
    ModelSimulator(const std::vector<int> &scheduleDays, unsigned int seed, bool useSeed = true);

    virtual ~ModelSimulator() = default;

    // Abstract interface
    virtual void generateBrownianMotions() = 0;

    virtual void simulate() = 0;

    virtual void regenerate(unsigned int newSeed) = 0;

    const std::vector<int> &getScheduleDays() const;

protected:
    std::vector<int> m_scheduleDays;
    unsigned int m_seed;
    bool m_useSeed;
    std::vector<BrownianMotion> m_brownianMotions;
};

#endif // MODEL_SIMULATOR_H
