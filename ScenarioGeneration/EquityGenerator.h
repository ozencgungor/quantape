#ifndef EQUITY_GENERATOR_H
#define EQUITY_GENERATOR_H

#include "ModelGenerator.h"
#include "EquitySimulator.h"
#include "HestonModel.h"
#include <map>
#include <string>

class EquityGenerator : public ModelGenerator {
public:
    explicit EquityGenerator(EquitySimulator &simulator);

    // Query methods (trigger lazy simulation if needed)
    double getSpot(const std::string &name, int day);

    double getVariance(const std::string &name, int day);

    HestonState getState(const std::string &name, int day);

    const std::map<int, HestonState> &getPath(const std::string &name);

    // Utility
    std::vector<std::string> getEquityNames() const;

private:
    EquitySimulator *m_simulator;

    void ensureSimulated(const std::string &name);
};

#endif // EQUITY_GENERATOR_H
