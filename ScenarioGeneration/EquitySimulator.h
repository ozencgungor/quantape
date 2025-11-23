#ifndef EQUITY_SIMULATOR_H
#define EQUITY_SIMULATOR_H

#include <map>
#include <set>
#include <string>

#include "HestonModel.h"
#include "ModelSimulator.h"

class EquitySimulator : public ModelSimulator {
public:
    EquitySimulator(const std::vector<int>& scheduleDays, unsigned int seed);

    EquitySimulator(const std::vector<int>& scheduleDays); // Unseeded

    // Add equity with Heston parameters
    void addEquity(const std::string& name, const HestonParams& params);

    // Lazy evaluation: simulate individual equity (called on-demand)
    void simulateEquity(const std::string& name);

    // Implement base class interface
    void generateBrownianMotions() override;

    void simulate() override; // Simulates all equities
    void regenerate(unsigned int newSeed) override;

    // Accessors
    const std::map<int, HestonState>& getEquityPath(const std::string& name) const;

    std::vector<std::string> getEquityNames() const;

    bool isSimulated(const std::string& name) const;

private:
    std::map<std::string, HestonModel> m_equityModels;
    std::map<std::string, std::map<int, HestonState>> m_equityPaths;
    std::set<std::string> m_simulatedEquities;
};

#endif // EQUITY_SIMULATOR_H
