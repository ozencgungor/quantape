#ifndef CREDIT_GENERATOR_H
#define CREDIT_GENERATOR_H

#include <map>
#include <string>

#include "CIRModel.h"
#include "CreditSimulator.h"
#include "ModelGenerator.h"

class CreditGenerator : public ModelGenerator {
public:
    explicit CreditGenerator(CreditSimulator& simulator);

    // Query methods (trigger lazy simulation if needed)
    double getValue(const std::string& name, int day);

    CIRState getState(const std::string& name, int day);

    const std::map<int, CIRState>& getPath(const std::string& name);

    // Utility
    std::vector<std::string> getCreditNames() const;

private:
    CreditSimulator* m_simulator;

    void ensureSimulated(const std::string& name);
};

#endif // CREDIT_GENERATOR_H
