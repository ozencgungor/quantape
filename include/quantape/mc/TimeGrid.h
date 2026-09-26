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
    explicit TimeGrid(std::vector<double> times) : times_(std::move(times)) {
        if (times_.size() < 2) {
            throw std::invalid_argument("TimeGrid: need at least two dates");
        }
        for (std::size_t k = 1; k < times_.size(); ++k) {
            if (!(times_[k] > times_[k - 1])) {
                throw std::invalid_argument("TimeGrid: times must be strictly increasing");
            }
        }
    }

    TimeGrid(double tMax, std::size_t nSteps) {
        if (!(tMax > 0.0) || nSteps < 1) {
            throw std::invalid_argument("TimeGrid: need tMax > 0 and nSteps >= 1");
        }
        times_.resize(nSteps + 1);
        for (std::size_t k = 0; k <= nSteps; ++k) {
            times_[k] = tMax * static_cast<double>(k) / static_cast<double>(nSteps);
        }
    }

    std::size_t nSteps() const { return times_.size() - 1; }
    const std::vector<double>& times() const { return times_; }
    double time(std::size_t k) const { return times_[k]; }
    double dt(std::size_t k) const { return times_[k + 1] - times_[k]; }
    double tMax() const { return times_.back(); }

private:
    std::vector<double> times_;
};

} // namespace quantape::mc

#endif // QUANTAPE_MC_TIMEGRID_H
