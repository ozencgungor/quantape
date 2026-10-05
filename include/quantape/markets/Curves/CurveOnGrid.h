#pragma once

#include "quantape/markets/Curves/DiscountCurve.h"

#include <cstddef>
#include <utility>
#include <vector>

namespace quantape::markets {
/**
 * @file CurveOnGrid.h
 * @brief Frozen curve evaluation on a fixed time grid
 *
 * The SDE hot path never interpolates: `materialize` evaluates a curve once
 * per pricing run on the simulation grid and stores flat arrays that are then
 * accessed by index. Costs O(nSteps) per curve and is state-independent.
 *
 * @tparam DoubleT Numeric type (`double` for pricing kernels, an AD scalar to
 *                 keep the materialized values on a tape).
 */

template <typename DoubleT>
struct CurveOnGrid {
    std::vector<double> times;     ///< Grid times (years, increasing)
    std::vector<DoubleT> discount; ///< D(t_k)
    std::vector<DoubleT> zero;     ///< Continuously compounded z(t_k)
    std::vector<DoubleT> forward;  ///< Continuously compounded f_k on [t_k, t_{k+1})

    std::size_t size() const { return times.size(); }
    std::size_t nSteps() const { return forward.size(); }

    /// Discount factor at grid index k.
    const DoubleT& discountAt(std::size_t k) const { return discount[k]; }

    /// Forward rate over interval k (between grid points k and k+1).
    const DoubleT& forwardAt(std::size_t k) const { return forward[k]; }
};

/// Materialize `curve` on `times` (strictly increasing, non-negative).
/// One interpolation pass plus one exponential per point.
template <typename DoubleT>
CurveOnGrid<DoubleT> materialize(const DiscountCurve<DoubleT>& curve,
                                 const std::vector<double>& times) {
    CurveOnGrid<DoubleT> out;
    out.times = times;
    const std::size_t n = times.size();
    if (n == 0) {
        return out;
    }
    out.discount.resize(n);
    out.zero.resize(n);
    if (n > 1) {
        out.forward.resize(n - 1);
    }
    std::vector<DoubleT> values;
    curve.spaceValuesInto(times, values);
    const bool zeroSpace = curve.space() == InterpolationSpace::Zero;
    for (std::size_t k = 0; k < n; ++k) {
        const double t = times[k];
        out.discount[k] = curve.discountFromSpace(values[k], t);
        out.zero[k] = zeroSpace ? values[k] : (t > 0.0 ? values[k] / t : curve.zero(0.0));
    }
    for (std::size_t k = 0; k + 1 < n; ++k) {
        const double dt = times[k + 1] - times[k];
        const DoubleT x0 = zeroSpace ? values[k] * times[k] : values[k];
        const DoubleT x1 = zeroSpace ? values[k + 1] * times[k + 1] : values[k + 1];
        out.forward[k] = (x1 - x0) / dt;
    }
    return out;
}

} // namespace quantape::markets
