#include "ModelGenerator.h"

ModelGenerator::ModelGenerator(const std::vector<int> &scheduleDays)
    : m_scheduleDays(scheduleDays) {
}

const std::vector<int> &ModelGenerator::getScheduleDays() const {
    return m_scheduleDays;
}
