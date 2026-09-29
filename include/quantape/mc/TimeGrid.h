#ifndef QUANTAPE_MC_TIMEGRID_H
#define QUANTAPE_MC_TIMEGRID_H

#include <cstddef>
#include <stdexcept>
#include <utility>
#include <vector>

namespace quantape::mc {
/**
 * @file TimeGrid.h
 * @brief Simulation time grid — the single source of time scaling
 *
 * ## Specification
 *
 * A grid is the date set
 *
 *     t_0 < t_1 < ... < t_N,   dt_k = t_{k+1} - t_k  (k = 0..N-1)
 *
 * constructed either explicitly from `times = {t_0, ..., t_N}` (strictly
 * increasing; irregular grids supported) or uniformly from `{tMax, nSteps}`
 * with `N = nSteps` and `t_k = k * tMax / N`.
 *
 * Every dt-dependent quantity in the simulator comes from here: drift
 * multiplications (dt_k), noise scaling (sqrt(dt_k)), and scheme-specific
 * dt scalings such as exact-transition variances. Random sources never
 * scale (they deliver standardized, unit-time draws).
 *
 * CamelCase naming for the whole mc/ API.
 */
class TimeGrid {
public:
    explicit TimeGrid(std::vector<double> times) : m_times(std::move(times)) {
        if (m_times.size() < 2) {
            throw std::invalid_argument("TimeGrid: need at least two dates");
        }
        for (std::size_t k = 1; k < m_times.size(); ++k) {
            if (!(m_times[k] > m_times[k - 1])) {
                throw std::invalid_argument("TimeGrid: times must be strictly increasing");
            }
        }
    }

    TimeGrid(double tMax, std::size_t nSteps) {
        if (!(tMax > 0.0) || nSteps < 1) {
            throw std::invalid_argument("TimeGrid: need tMax > 0 and nSteps >= 1");
        }
        m_times.resize(nSteps + 1);
        for (std::size_t k = 0; k <= nSteps; ++k) {
            m_times[k] = tMax * static_cast<double>(k) / static_cast<double>(nSteps);
        }
    }

    std::size_t nSteps() const noexcept { return m_times.size() - 1; }
    const std::vector<double>& times() const noexcept { return m_times; }
    double time(std::size_t k) const noexcept { return m_times[k]; }
    double dt(std::size_t k) const noexcept { return m_times[k + 1] - m_times[k]; }
    double tMax() const noexcept { return m_times.back(); }

private:
    std::vector<double> m_times;
};

} // namespace quantape::mc

#endif // QUANTAPE_MC_TIMEGRID_H
