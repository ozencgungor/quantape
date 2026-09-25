#ifndef BROWNIAN_MOTION_H
#define BROWNIAN_MOTION_H

#include <map>
#include <random>
#include <vector>

class BrownianMotion {
public:
    // Constructor with schedule in days
    BrownianMotion(const std::vector<int>& scheduleDays, unsigned int seed);

    BrownianMotion(const std::vector<int>& scheduleDays); // Unseeded (uses random_device)

    // Access methods
    double getValue(int day) const;

    double getIncrement(int day1, int day2) const;

    const std::map<int, double>& getPath() const;

    const std::vector<int>& getScheduleDays() const;

    // Friend functions for correlation
    friend std::vector<BrownianMotion>
    generateCorrelatedBrownianMotions(const std::vector<int>& scheduleDays, size_t numMotions,
                                      const std::vector<std::vector<double>>& correlationMatrix,
                                      unsigned int seed);

private:
    std::vector<int> m_scheduleDays;
    std::vector<double> m_scheduleYears;
    std::map<int, double> m_path; // day -> Brownian motion value

    void validateSchedule() const;

    void convertDaysToYears();

    void generatePath(unsigned int seed);

    void generatePathUnseeded();
};

// Helper functions for correlation
std::vector<BrownianMotion>
generateCorrelatedBrownianMotions(const std::vector<int>& scheduleDays, size_t numMotions,
                                  const std::vector<std::vector<double>>& correlationMatrix,
                                  unsigned int seed);

std::vector<BrownianMotion> generateIndependentBrownianMotions(const std::vector<int>& scheduleDays,
                                                               size_t numMotions,
                                                               unsigned int seed);

#endif // BROWNIAN_MOTION_H
