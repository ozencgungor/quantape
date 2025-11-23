#include "ModelSimulator.h"

ModelSimulator::ModelSimulator(const std::vector<int>& scheduleDays, unsigned int seed,
                               bool useSeed)
    : m_scheduleDays(scheduleDays), m_seed(seed), m_useSeed(useSeed) {}

const std::vector<int>& ModelSimulator::getScheduleDays() const {
    return m_scheduleDays;
}
