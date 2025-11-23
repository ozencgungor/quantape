#include "BrownianMotion.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>

const double DAYS_IN_YEAR = 365.25;

// ============================================================================
// BrownianMotion Implementation
// ============================================================================

BrownianMotion::BrownianMotion(const std::vector<int>& scheduleDays, unsigned int seed)
    : m_scheduleDays(scheduleDays) {
    validateSchedule();
    convertDaysToYears();
    generatePath(seed);
}

BrownianMotion::BrownianMotion(const std::vector<int>& scheduleDays)
    : m_scheduleDays(scheduleDays) {
    validateSchedule();
    convertDaysToYears();
    generatePathUnseeded();
}

void BrownianMotion::validateSchedule() const {
    if (m_scheduleDays.empty()) {
        throw std::invalid_argument("Schedule cannot be empty");
    }
    if (m_scheduleDays[0] != 0) {
        throw std::invalid_argument("Schedule must start at day 0");
    }
    for (size_t i = 1; i < m_scheduleDays.size(); ++i) {
        if (m_scheduleDays[i] <= m_scheduleDays[i - 1]) {
            throw std::invalid_argument("Schedule days must be strictly increasing");
        }
        if (m_scheduleDays[i] < 0) {
            throw std::invalid_argument("Schedule days must be non-negative");
        }
    }
}

void BrownianMotion::convertDaysToYears() {
    m_scheduleYears.resize(m_scheduleDays.size());
    for (size_t i = 0; i < m_scheduleDays.size(); ++i) {
        m_scheduleYears[i] = m_scheduleDays[i] / DAYS_IN_YEAR;
    }
}

void BrownianMotion::generatePath(unsigned int seed) {
    std::mt19937 gen(seed);
    std::normal_distribution<> dist(0.0, 1.0);

    m_path[m_scheduleDays[0]] = 0.0; // Brownian motion starts at 0

    for (size_t i = 1; i < m_scheduleDays.size(); ++i) {
        double dt = m_scheduleYears[i] - m_scheduleYears[i - 1];
        double dW = dist(gen) * std::sqrt(dt);
        m_path[m_scheduleDays[i]] = m_path[m_scheduleDays[i - 1]] + dW;
    }
}

void BrownianMotion::generatePathUnseeded() {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<> dist(0.0, 1.0);

    m_path[m_scheduleDays[0]] = 0.0;

    for (size_t i = 1; i < m_scheduleDays.size(); ++i) {
        double dt = m_scheduleYears[i] - m_scheduleYears[i - 1];
        double dW = dist(gen) * std::sqrt(dt);
        m_path[m_scheduleDays[i]] = m_path[m_scheduleDays[i - 1]] + dW;
    }
}

double BrownianMotion::getValue(int day) const {
    auto it = m_path.find(day);
    if (it == m_path.end()) {
        throw std::out_of_range("Day not found in Brownian motion path");
    }
    return it->second;
}

double BrownianMotion::getIncrement(int day1, int day2) const {
    return getValue(day2) - getValue(day1);
}

const std::map<int, double>& BrownianMotion::getPath() const {
    return m_path;
}

const std::vector<int>& BrownianMotion::getScheduleDays() const {
    return m_scheduleDays;
}

// ============================================================================
// Correlation Helper Functions
// ============================================================================

// Cholesky decomposition
static std::vector<std::vector<double>>
choleskyDecomposition(const std::vector<std::vector<double>>& correlationMatrix) {
    size_t n = correlationMatrix.size();
    std::vector<std::vector<double>> L(n, std::vector<double>(n, 0.0));

    for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j <= i; ++j) {
            double sum = 0.0;
            for (size_t k = 0; k < j; ++k) {
                sum += L[i][k] * L[j][k];
            }

            if (i == j) {
                double val = correlationMatrix[i][i] - sum;
                if (val <= 0.0) {
                    throw std::runtime_error("Correlation matrix is not positive definite");
                }
                L[i][j] = std::sqrt(val);
            } else {
                L[i][j] = (correlationMatrix[i][j] - sum) / L[j][j];
            }
        }
    }

    return L;
}

std::vector<BrownianMotion>
generateCorrelatedBrownianMotions(const std::vector<int>& scheduleDays, size_t numMotions,
                                  const std::vector<std::vector<double>>& correlationMatrix,
                                  unsigned int seed) {
    // Validate correlation matrix
    if (correlationMatrix.size() != numMotions) {
        throw std::invalid_argument("Correlation matrix size must match number of motions");
    }
    for (const auto& row : correlationMatrix) {
        if (row.size() != numMotions) {
            throw std::invalid_argument("Correlation matrix must be square");
        }
    }

    // Compute Cholesky decomposition
    auto L = choleskyDecomposition(correlationMatrix);

    // Generate independent Brownian motions
    std::vector<BrownianMotion> independentBMs;
    for (size_t i = 0; i < numMotions; ++i) {
        independentBMs.emplace_back(scheduleDays, seed + i);
    }

    // Apply correlation via Cholesky matrix
    std::vector<BrownianMotion> correlatedBMs;
    for (size_t i = 0; i < numMotions; ++i) {
        BrownianMotion correlatedBM(scheduleDays, seed + numMotions + i);

        // For each time point, compute correlated value
        for (int day : scheduleDays) {
            double correlatedValue = 0.0;
            for (size_t j = 0; j <= i; ++j) {
                correlatedValue += L[i][j] * independentBMs[j].getValue(day);
            }
            correlatedBM.m_path[day] = correlatedValue;
        }

        correlatedBMs.push_back(correlatedBM);
    }

    return correlatedBMs;
}

std::vector<BrownianMotion> generateIndependentBrownianMotions(const std::vector<int>& scheduleDays,
                                                               size_t numMotions,
                                                               unsigned int seed) {
    std::vector<BrownianMotion> brownianMotions;
    for (size_t i = 0; i < numMotions; ++i) {
        brownianMotions.emplace_back(scheduleDays, seed + i);
    }
    return brownianMotions;
}
