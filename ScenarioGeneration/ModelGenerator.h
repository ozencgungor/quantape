#ifndef MODEL_GENERATOR_H
#define MODEL_GENERATOR_H

#include <vector>

// Abstract base class for all model generators
class ModelGenerator {
public:
    ModelGenerator(const std::vector<int> &scheduleDays);

    virtual ~ModelGenerator() = default;

    const std::vector<int> &getScheduleDays() const;

protected:
    std::vector<int> m_scheduleDays;
};

#endif // MODEL_GENERATOR_H
