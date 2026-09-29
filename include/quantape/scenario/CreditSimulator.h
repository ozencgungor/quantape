#ifndef CREDIT_SIMULATOR_H
#define CREDIT_SIMULATOR_H

#include "quantape/mc/processes/CIRProcess.h"
#include "quantape/scenario/ModelSimulator.h"

#include <map>
#include <string>
#include <unordered_map>
#include <unordered_set>

class CreditSimulator : public ModelSimulator {
public:
    CreditSimulator(const std::vector<int>& scheduleDays, unsigned int seed);

    CreditSimulator(const std::vector<int>& scheduleDays); // Unseeded

    // Add credit with CIR parameters
    void addCredit(const std::string& name, const CIRParams& params);

    // Lazy evaluation: simulate individual credit (called on-demand)
    void simulateCredit(const std::string& name);

    // Implement base class interface
    void generateBrownianMotions() override;

    void simulate() override; // Simulates all credits
    void regenerate(unsigned int newSeed) override;

    // Accessors
    const std::map<int, CIRState>& getCreditPath(const std::string& name) const;

    std::vector<std::string> getCreditNames() const;

    bool isSimulated(const std::string& name) const;

private:
    std::map<std::string, CIRProcess> m_creditModels;
    std::unordered_map<std::string, std::map<int, CIRState>> m_creditPaths;
    std::unordered_set<std::string> m_simulatedCredits;
};

#endif // CREDIT_SIMULATOR_H
