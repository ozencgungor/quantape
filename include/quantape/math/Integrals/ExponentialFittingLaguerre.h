#ifndef QUANTAPE_MATH_INTEGRALS_EXPONENTIAL_FITTING_LAGUERRE_H
#define QUANTAPE_MATH_INTEGRALS_EXPONENTIAL_FITTING_LAGUERRE_H

#include "quantape/math/Integrals/ExponentialFittingLaguerreTable.h"

#include <cmath>
#include <cstddef>
#include <stdexcept>

namespace quantape::math {
/**
 * @file ExponentialFittingLaguerre.h
 * @brief Exponentially fitted Gauss–Laguerre rule (Conte et al. 2013)
 *
 * Integrates `int_0^inf A(v) dv` where `A` contains an oscillatory factor
 * `e^{i v mu}` and decays slowly (the Heston Lewis integrand at deep
 * wings / high vol-of-vol). The rule is exact for the modified moments of
 * `e^{-x}(f1 sin(w x) + f2 cos(w x))` with the table frequency `w`, so a
 * 64-node rule replaces hundreds of Gauss–Legendre nodes.
 *
 * Runtime selection follows the reference implementation (QuantLib
 * PR #812, `ExponentialFittingHestonEngine`): with `f = frequency` and a
 * caller scaling `s` (itself parameter dependent):
 *
 *   |f| < 0.1 :  row 0, scale = s
 *   otherwise :  row = nearest table omega to |s f|, scale = |omega / f|
 *
 * and the integral is approximated by
 *
 *   int_0^inf A(v) dv  ~=  scale * sum_i rawW_i * A(scale * x_i)
 *
 * with `rawW_i` the table's weightless weights (x_i descending). The rule
 * is a fixed weighted sum; `f` is scalar-generic (double/var/complex
 * integrands) and rows/scales are chosen from real inputs only, so the
 * selection is stable under AD (primal-pinned control flow).
 */
class ExponentialFittingLaguerre {
public:
    std::size_t order() const { return (detail::efglColumnCount() - 1) / 2; }

    template <typename Scalar, typename F>
    Scalar integrate(double frequency, double scaling, const F& f) const {
        const std::size_t n = order();
        const std::size_t columns = detail::efglColumnCount();
        const double* table = detail::efglTable();

        std::size_t row = 0;
        double scale = scaling;
        if (std::fabs(frequency) >= 0.1) {
            const double lookup = std::fabs(scaling * frequency);
            row = nearestRow(lookup);
            const double omega = table[row * columns];
            scale = std::fabs(omega / frequency);
        }
        const double* nodes = table + row * columns + 1;
        const double* weights = nodes + n;

        Scalar sum = Scalar(0.0);
        for (std::size_t i = 0; i < n; ++i) {
            sum += Scalar(weights[i]) * f(Scalar(scale * nodes[i]));
        }
        return Scalar(scale) * sum;
    }

private:
    /// Table omega column is ascending; pick the nearest entry (ties round down).
    std::size_t nearestRow(double lookup) const {
        const std::size_t rows = detail::efglRowCount();
        const std::size_t columns = detail::efglColumnCount();
        const double* table = detail::efglTable();
        std::size_t lo = 0;
        std::size_t hi = rows; // first index with omega >= lookup
        while (lo < hi) {
            const std::size_t mid = lo + (hi - lo) / 2;
            if (table[mid * columns] < lookup) {
                lo = mid + 1;
            } else {
                hi = mid;
            }
        }
        if (lo == 0) {
            return 0;
        }
        if (lo >= rows) {
            return rows - 1;
        }
        const double upper = std::fabs(lookup - table[lo * columns]);
        const double lower = std::fabs(lookup - table[(lo - 1) * columns]);
        return (upper > lower) ? lo - 1 : lo;
    }
};

} // namespace quantape::math

#endif // QUANTAPE_MATH_INTEGRALS_EXPONENTIAL_FITTING_LAGUERRE_H
