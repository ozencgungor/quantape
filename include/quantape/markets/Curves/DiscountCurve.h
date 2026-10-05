#pragma once

#include "quantape/math/Autodiff/PrimalExtraction.h"
#include "quantape/datetime/Date.h"
#include "quantape/datetime/DayCounter.h"
#include "quantape/datetime/TimeConversion.h"
#include "quantape/markets/Curves/Curve.h"
#include "quantape/math/Interpolations/CubicInterpolation.h"
#include "quantape/math/Interpolations/HymanSplineInterpolation.h"
#include "quantape/math/Interpolations/MonotoneCubicInterpolation.h"
#include "quantape/math/Interpolations/ProbeDual.h"
#include "quantape/math/Interpolations/TensionSplineInterpolation.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>
#include <vector>

namespace quantape::markets {
/**
 * @file DiscountCurve.h
 * @brief POD discount curve on a zero-rate node grid
 *
 * State is continuously compounded zero rates on an ACT/365F zero clock
 * The curve stores flat buffers:
 * knot times, node zeros, space values and (for Akima) pre-computed segment
 * coefficients. Evaluation is allocation-free and branch-light.
 *
 * Interpolation space x scheme:
 *   - `Zero`         + `Linear` : piecewise-linear zero rates;
 *   - `LogDiscount`  + `Linear` : linear in log discount (piecewise-constant
 *                                 instantaneous forward) — the default;
 *   - `Zero`         + `Akima`  : local C1 Akima cubic on zero rates
 *                                 (coefficients from `math::CubicInterpolation`).
 *
 * A node at t = 0 with value 0 is always present (D(0) = 1). Extrapolation
 * beyond the last knot continues the last segment (flat instantaneous forward
 * in `LogDiscount`).
 *
 * @tparam DoubleT Numeric type (`double` or `stan::math::var`).
 */
enum class InterpolationSpace : std::uint8_t {
    Zero,        ///< Zero rates
    LogDiscount, ///< Log discount factors (y = z t)
};

enum class InterpolationScheme : std::uint8_t {
    Linear,        ///< Linear in the chosen space
    Akima,         ///< Akima cubic (local C1)
    TensionSpline, ///< Exponential tension spline (C2, sigma-controlled locality)
    HymanSpline,   ///< Natural cubic spline with Hyman's monotone node-slope
                   ///< filter (C1, monotone; in-range risk weights are
                   ///< unavailable for this scheme)
    MonotoneCubic, ///< Steffen shape-preserving cubic (C1, monotone; in-range
                   ///< risk weights are unavailable for this scheme)
    MixedLinearCubic, ///< Linear up to `switchIndex` segments, Akima beyond
};

constexpr std::string_view interpolationSpaceName(InterpolationSpace space) noexcept {
    switch (space) {
        case InterpolationSpace::Zero: return "Zero";
        case InterpolationSpace::LogDiscount: return "LogDiscount";
    }
    return "Unknown";
}

constexpr std::string_view interpolationSchemeName(InterpolationScheme scheme) noexcept {
    switch (scheme) {
        case InterpolationScheme::Linear: return "Linear";
        case InterpolationScheme::Akima: return "Akima";
        case InterpolationScheme::TensionSpline: return "TensionSpline";
        case InterpolationScheme::HymanSpline: return "HymanSpline";
        case InterpolationScheme::MonotoneCubic: return "MonotoneCubic";
        case InterpolationScheme::MixedLinearCubic: return "MixedLinearCubic";
    }
    return "Unknown";
}

template <typename DoubleT>
class DiscountCurve {
public:
    /// Times constructor (analysis / benchmarks; no calendar dates involved).
    DiscountCurve(std::vector<double> times, std::vector<DoubleT> zeros,
                  InterpolationSpace space = InterpolationSpace::LogDiscount,
                  InterpolationScheme scheme = InterpolationScheme::Linear,
                  double tension = 0.0, int switchIndex = 1)
        : m_tension(tension), m_switchIndex(switchIndex), m_space(space), m_scheme(scheme) {
        m_times = std::move(times);
        m_zeros = std::move(zeros);
        initialize();
    }

    /// Date constructor: knot times are year fractions on `zeroDayCounter`.
    DiscountCurve(const datetime::Date& referenceDate,
                  const std::vector<datetime::Date>& pillarDates,
                  const datetime::DayCounter& zeroDayCounter, std::vector<DoubleT> zeros,
                  InterpolationSpace space = InterpolationSpace::LogDiscount,
                  InterpolationScheme scheme = InterpolationScheme::Linear,
                  double tension = 0.0, int switchIndex = 1)
        : m_referenceDate(referenceDate), m_zeroDayCounter(zeroDayCounter), m_tension(tension),
          m_switchIndex(switchIndex), m_space(space), m_scheme(scheme) {
        if (pillarDates.size() != zeros.size()) {
            throw std::invalid_argument("DiscountCurve: pillarDates/zeros size mismatch");
        }
        m_times.reserve(pillarDates.size());
        for (const datetime::Date& date : pillarDates) {
            m_times.push_back(datetime::yearFraction(referenceDate, date, zeroDayCounter));
        }
        m_zeros = std::move(zeros);
        initialize();
    }

    /// Continuously compounded zero rate at `t`.
    DoubleT zero(double t) const {
        if (m_space == InterpolationSpace::Zero) {
            return spaceValue(t);
        }
        if (t <= 0.0) {
            return firstSlope();
        }
        return spaceValue(t) / t;
    }

    /// Discount factor from a pre-computed space value (batch materialization).
    DoubleT discountFromSpace(const DoubleT& spaceValue, double t) const {
        if (t <= 0.0) {
            return DoubleT(1);
        }
        if (m_space == InterpolationSpace::Zero) {
            return expImpl(-spaceValue * t);
        }
        return expImpl(-spaceValue);
    }

    /// Discount factor at `t` (1 for t <= 0).
    DoubleT discount(double t) const {
        if (t <= 0.0) {
            return DoubleT(1);
        }
        if (m_space == InterpolationSpace::LogDiscount) {
            return expImpl(-spaceValue(t));
        }
        return expImpl(-spaceValue(t) * t);
    }

    /// Compound factor `D(t1) / D(t2)` (t1 <= t2).
    DoubleT compoundingFactor(double t1, double t2) const {
        if (t1 > t2) {
            throw std::invalid_argument("DiscountCurve::compoundingFactor: t1 must be <= t2");
        }
        return discount(t1) / discount(t2);
    }

    /// Compounded overnight rate over [t1, t2] (telescoped product of daily
    /// compoundings), given the period's year fraction. Valid whenever the
    /// instrument satisfies the index's telescoping condition.
    DoubleT compoundedRate(double t1, double t2, double yearFraction) const {
        if (!(yearFraction > 0.0)) {
            throw std::invalid_argument("DiscountCurve::compoundedRate: non-positive year fraction");
        }
        return (compoundingFactor(t1, t2) - 1.0) / yearFraction;
    }

    /// Arithmetic average of simple overnight forwards over consecutive fixing
    /// boundaries `fixingTimes` (size >= 2, strictly increasing).
    DoubleT averagedRate(const std::vector<double>& fixingTimes) const {
        if (fixingTimes.size() < 2) {
            throw std::invalid_argument("DiscountCurve::averagedRate: need two or more times");
        }
        DoubleT sum = 0;
        std::size_t count = 0;
        for (std::size_t k = 1; k < fixingTimes.size(); ++k) {
            const double tau = fixingTimes[k] - fixingTimes[k - 1];
            if (!(tau > 0.0)) {
                throw std::invalid_argument(
                    "DiscountCurve::averagedRate: times must be strictly increasing");
            }
            const DoubleT simple = (discount(fixingTimes[k - 1]) / discount(fixingTimes[k]) - 1.0) /
                                   tau;
            sum += simple;
            ++count;
        }
        return sum / static_cast<double>(count);
    }

    /// Continuously compounded forward rate over (t1, t2].
    DoubleT forward(double t1, double t2) const {
        if (!(t2 > t1)) {
            throw std::invalid_argument("DiscountCurve::forward: t2 must be > t1");
        }
        const DoubleT x1 =
            m_space == InterpolationSpace::Zero ? valueAt(t1) * t1 : spaceValue(t1);
        const DoubleT x2 =
            m_space == InterpolationSpace::Zero ? valueAt(t2) * t2 : spaceValue(t2);
        return (x2 - x1) / (t2 - t1);
    }

    /// Interpolation-only weights: `spaceValue(t) = sum_i w_i * m_values[i]`
    /// for `t` inside the node range (no extrapolation). Local schemes give
    /// few nonzeros; branch-adaptive schemes use the primal-pinned probe.
    void spaceValueWeights(double t, std::vector<double>& weights) const {
        const std::size_t n = m_times.size();
        weights.assign(n, 0.0);
        if (!(t >= m_times.front() && t <= m_times.back())) {
            throw std::invalid_argument("DiscountCurve::spaceValueWeights: t outside node range");
        }
        const std::size_t i = segment(t);
        const double h = m_times[i + 1] - m_times[i];
        const double u = (t - m_times[i]) / h;
        if (m_scheme == InterpolationScheme::Linear ||
            (m_scheme == InterpolationScheme::MixedLinearCubic &&
             i < static_cast<std::size_t>(m_switchIndex))) {
            weights[i] = 1.0 - u;
            weights[i + 1] = u;
            return;
        }
        if (m_scheme == InterpolationScheme::MonotoneCubic) {
            throw std::invalid_argument(
                "DiscountCurve: MonotoneCubic risk weights are unavailable");
        }
        if (m_scheme == InterpolationScheme::HymanSpline) {
            throw std::invalid_argument(
                "DiscountCurve: HymanSpline risk weights are unavailable");
        }
        if (m_scheme == InterpolationScheme::Akima ||
            (m_scheme == InterpolationScheme::MixedLinearCubic &&
             i >= static_cast<std::size_t>(m_switchIndex))) {
            std::vector<quantape::math::detail::ProbeDual> y(n);
            for (std::size_t j = 0; j < n; ++j) {
                y[j] = quantape::math::detail::ProbeDual(primalValue(m_values[j]), n);
                y[j].d[j] = 1.0;
            }
            std::vector<quantape::math::detail::ProbeDual> a;
            std::vector<quantape::math::detail::ProbeDual> b;
            std::vector<quantape::math::detail::ProbeDual> c;
            quantape::math::CubicInterpolation<double>::computeCoefficientsDual(
                m_times, y, quantape::math::CubicDerivativeApprox::Akima, false, a, b, c);
            const double dx = t - m_times[i];
            const quantape::math::detail::ProbeDual value =
                y[i] + a[i] * dx + b[i] * (dx * dx) + c[i] * (dx * dx * dx);
            for (std::size_t j = 0; j < n && j < value.d.size(); ++j) {
                weights[j] = value.d[j];
            }
            return;
        }
        // TensionSpline: value is linear in the node values; use the adjoint of
        // the tridiagonal system for the second-derivative state.
        const double sigma = m_tension;
        const double right = m_times[i + 1] - t;
        const double left = t - m_times[i];
        const double invLambda = m_tensionInvLambda[i];
        weights[i] += right / h;
        weights[i + 1] += left / h;
        if (n <= 2) {
            return;
        }
        std::vector<double> gM(n - 2, 0.0);
        if (i >= 1) {
            gM[i - 1] += invLambda * std::sinh(sigma * right) - (right / h) / (sigma * sigma);
        }
        if (i + 1 <= n - 2) {
            gM[i] += invLambda * std::sinh(sigma * left) - (left / h) / (sigma * sigma);
        }
        const std::vector<double> z = quantape::math::TridiagonalSolver<double>::solve(
            m_tensionSub, m_tensionDiag, m_tensionSuper, gM);
        for (std::size_t r = 1; r + 1 < n; ++r) {
            const double hLeft = m_times[r] - m_times[r - 1];
            const double hRight = m_times[r + 1] - m_times[r];
            weights[r - 1] += z[r - 1] / hLeft;
            weights[r] -= z[r - 1] * (1.0 / hRight + 1.0 / hLeft);
            weights[r + 1] += z[r - 1] / hRight;
        }
    }

    /// `d z(t) / d z_i` for solved node `i` (`t > 0`; beyond the last node the
    /// scheme's extrapolated weights are used). Node 0 (t = 0) is fixed and its
    /// weight is left at zero.
    void zeroNodeWeights(double t, std::vector<double>& weights) const {
        if (!(t >= 0.0)) {
            throw std::invalid_argument(
                "DiscountCurve::zeroNodeWeights: t must be non-negative");
        }
        if (t == 0.0) {
            // D(0) = 1 and the t = 0 node is fixed: no node carries risk.
            weights.assign(m_times.size(), 0.0);
            return;
        }
        if (t > m_times.back()) {
            extrapolatedZeroWeights(t, weights);
            return;
        }
        spaceValueWeights(t, weights);
        if (m_space == InterpolationSpace::LogDiscount) {
            const double invT = 1.0 / t;
            for (std::size_t i = 0; i < weights.size(); ++i) {
                weights[i] *= m_times[i] * invT;
            }
        }
        weights.front() = 0.0;
    }

    /// Space values at `times` (strictly increasing) with one monotone pass;
    /// the batch entry point behind `CurveOnGrid` materialization.
    std::vector<DoubleT> spaceValues(const std::vector<double>& times) const {
        std::vector<DoubleT> out;
        out.reserve(times.size());
        const std::size_t lastSegment = m_times.size() - 2;
        std::size_t seg = 0;
        for (const double t : times) {
            while (seg < lastSegment && t >= m_times[seg + 1]) {
                ++seg;
            }
            if (t <= m_times.front() && m_scheme != InterpolationScheme::Linear) {
                out.push_back(m_values.front() + (t - m_times.front()) * m_firstSlope);
            } else if (t >= m_times.back() && m_scheme != InterpolationScheme::Linear) {
                out.push_back(m_values.back() + (t - m_times.back()) * m_lastSlope);
            } else {
                out.push_back(valueOnSegment(seg, t));
            }
        }
        return out;
    }

    // Compatibility aliases: the migrated dividend/FX consumers used the
    // placeholder curve API. Extrapolation is always flat-forward here, so the
    // `allowExtrapolation` flag is accepted and ignored.
    DoubleT zeroRate(double t, bool = true) const { return zero(t); }
    DoubleT yield(double t, bool = true) const { return zero(t); }
    DoubleT discountFactor(double t, bool = true) const { return discount(t); }
    DoubleT forwardDiscountFactor(double t1, double t2, bool = true) const {
        return compoundingFactor(t1, t2);
    }
    DoubleT forwardRate(double t1, double t2, bool = true) const { return forward(t1, t2); }
    const std::vector<double>& tenors() const { return m_times; }
    const std::vector<DoubleT>& rates() const { return m_zeros; }

    const std::vector<double>& times() const { return m_times; }
    const std::vector<DoubleT>& zeros() const { return m_zeros; }
    InterpolationSpace space() const { return m_space; }
    InterpolationScheme scheme() const { return m_scheme; }
    double tension() const { return m_tension; }
    int switchIndex() const { return m_switchIndex; }
    const datetime::Date& referenceDate() const { return m_referenceDate; }
    const datetime::DayCounter& zeroDayCounter() const { return m_zeroDayCounter; }
    std::size_t size() const { return m_times.size(); }

private:
    /// Weights beyond the last node: central differences of the scheme's own
    /// extrapolated zero curve on a rebuilt passive curve, so the risk
    /// transform matches `discount()`/`zero()` extrapolation exactly. Only
    /// defined for positive query times (the pricing path treats t <= 0 as
    /// D = 1, which carries no node risk).
    void extrapolatedZeroWeights(double t, std::vector<double>& weights) const {
        if (!(t > 0.0)) {
            throw std::invalid_argument(
                "DiscountCurve::extrapolatedZeroWeights: t must be positive");
        }
        const std::size_t n = m_times.size();
        weights.assign(n, 0.0);
        std::vector<double> zeros(n, 0.0);
        for (std::size_t i = 1; i < n; ++i) {
            const double value = primalValue(m_values[i]);
            zeros[i] = m_space == InterpolationSpace::Zero ? value : value / m_times[i];
        }
        const auto rebuilt = [&](const std::vector<double>& nodeZeros) {
            return DiscountCurve<double>(m_times, nodeZeros, m_space, m_scheme, m_tension,
                                         m_switchIndex);
        };
        for (std::size_t i = 1; i < n; ++i) {
            const double step = 1e-6 * (1.0 + std::abs(zeros[i]));
            std::vector<double> plus = zeros;
            std::vector<double> minus = zeros;
            plus[i] += step;
            minus[i] -= step;
            weights[i] = (rebuilt(plus).zero(t) - rebuilt(minus).zero(t)) / (2.0 * step);
        }
    }

    void initialize() {
        if (m_times.size() != m_zeros.size()) {
            throw std::invalid_argument("DiscountCurve: times/zeros size mismatch");
        }
        if (m_times.empty()) {
            throw std::invalid_argument("DiscountCurve: empty node grid");
        }
        if (m_times.front() < 0.0) {
            throw std::invalid_argument("DiscountCurve: negative node time");
        }
        if (m_times.front() > 0.0) {
            m_times.insert(m_times.begin(), 0.0);
            m_zeros.insert(m_zeros.begin(), DoubleT(0));
        } else {
            m_zeros.front() = DoubleT(0);
        }
        if (m_times.size() < 2) {
            throw std::invalid_argument("DiscountCurve: need at least two nodes (t=0 and a pillar)");
        }
        for (std::size_t i = 1; i < m_times.size(); ++i) {
            if (!(m_times[i] > m_times[i - 1])) {
                throw std::invalid_argument("DiscountCurve: node times must be strictly increasing");
            }
        }
        // Akima and the other local cubics act on the chosen space values, so
        // they are available in both Zero and LogDiscount space; the shape
        // guarantees (monotonicity, positivity) are properties of the space.

        if (m_scheme == InterpolationScheme::TensionSpline && !(m_tension > 0.0)) {
            throw std::invalid_argument("DiscountCurve: TensionSpline needs a positive tension");
        }
        m_values.resize(m_zeros.size());
        for (std::size_t i = 0; i < m_zeros.size(); ++i) {
            m_values[i] = m_space == InterpolationSpace::Zero ? m_zeros[i]
                                                              : m_zeros[i] * m_times[i];
        }
        if (m_scheme == InterpolationScheme::MixedLinearCubic) {
            // Short trial curves used inside bootstrap may not have enough
            // segments for the configured switch: degenerate to linear.
            const int maxIndex = static_cast<int>(m_times.size()) - 1;
            m_switchIndex = std::max(1, std::min(m_switchIndex, maxIndex));
        }
        if (m_scheme == InterpolationScheme::Akima ||
            m_scheme == InterpolationScheme::MixedLinearCubic) {
            const quantape::math::CubicInterpolation<DoubleT> interp(
                m_times, m_values, quantape::math::CubicDerivativeApprox::Akima);
            m_a = interp.aCoeffs();
            m_b = interp.bCoeffs();
            m_c = interp.cCoeffs();
            // End slopes follow the scheme actually used on the edge segment:
            // a linear segment under MixedLinearCubic extrapolates linearly,
            // otherwise the stored coefficients give the edge slope under the
            // curve's own convention P(dx) = y + a dx + b dx^2 + c dx^3 (the
            // interpolator's generic derivative(back) resolves the boundary
            // differently).
            const std::size_t last = m_times.size() - 2;
            const double h = m_times[last + 1] - m_times[last];
            const bool mixed = m_scheme == InterpolationScheme::MixedLinearCubic;
            m_firstSlope =
                mixed && 0 < m_switchIndex
                    ? (m_values[1] - m_values[0]) / (m_times[1] - m_times[0])
                    : m_a.front();
            m_lastSlope =
                mixed && last < static_cast<std::size_t>(m_switchIndex)
                    ? (m_values[last + 1] - m_values[last]) / h
                    : m_a[last] + h * (2.0 * m_b[last] + h * 3.0 * m_c[last]);
        }
        if (m_scheme == InterpolationScheme::MonotoneCubic) {
            const quantape::math::MonotoneCubicInterpolation<DoubleT> interp(m_times, m_values);
            m_monotoneSlopes = interp.slopes();
            m_firstSlope = m_monotoneSlopes.front();
            m_lastSlope = m_monotoneSlopes.back();
        }
        if (m_scheme == InterpolationScheme::HymanSpline) {
            const quantape::math::HymanSplineInterpolation<DoubleT> interp(m_times, m_values);
            m_hymanSlopes = interp.slopes();
            m_firstSlope = m_hymanSlopes.front();
            m_lastSlope = m_hymanSlopes.back();
        }
        if (m_scheme == InterpolationScheme::TensionSpline) {
            const quantape::math::TensionSplineInterpolation<DoubleT> interp(m_times, m_values,
                                                                            m_tension);
            m_tensionM = interp.secondDerivatives();
            const std::size_t segments = m_times.size() - 1;
            m_tensionInvLambda.resize(segments);
            for (std::size_t j = 0; j < segments; ++j) {
                const double p = m_tension * (m_times[j + 1] - m_times[j]);
                m_tensionInvLambda[j] = 1.0 / (m_tension * m_tension * std::sinh(p));
            }
            m_tensionSub = interp.systemSubDiagonal();
            m_tensionDiag = interp.systemDiagonal();
            m_tensionSuper = interp.systemSuperDiagonal();
            // Exact end slopes from the tension formula (the interpolator's
            // generic boundary derivative resolves the right end differently).
            const std::size_t last = m_times.size() - 2;
            const double sigma = m_tension;
            const double sigmaSquared = sigma * sigma;
            const double hLast = m_times[last + 1] - m_times[last];
            const double invLast = m_tensionInvLambda[last];
            m_lastSlope = -m_tensionM[last] * invLast * sigma +
                          m_tensionM[last + 1] * invLast * sigma * std::cosh(sigma * hLast) +
                          (m_values[last + 1] - m_values[last] +
                           (m_tensionM[last] - m_tensionM[last + 1]) / sigmaSquared) /
                              hLast;
            const double hFirst = m_times[1] - m_times[0];
            const double invFirst = m_tensionInvLambda[0];
            m_firstSlope =
                -m_tensionM[0] * invFirst * sigma * std::cosh(sigma * hFirst) +
                m_tensionM[1] * invFirst * sigma +
                (m_values[1] - m_values[0] +
                 (m_tensionM[0] - m_tensionM[1]) / sigmaSquared) /
                    hFirst;
        }
    }

    /// Segment index in [0, size - 2].
    std::size_t segment(double t) const {
        if (!std::isfinite(t)) {
            throw std::invalid_argument("DiscountCurve::segment: non-finite query time");
        }
        if (t <= m_times.front()) {
            return 0;
        }
        if (t >= m_times.back()) {
            return m_times.size() - 2;
        }
        const auto it = std::upper_bound(m_times.begin(), m_times.end(), t);
        return static_cast<std::size_t>(it - m_times.begin()) - 1;
    }

    /// Space value on segment `i` at time `t`.
    DoubleT valueOnSegment(std::size_t i, double t) const {
        const double dx = t - m_times[i];
        if (m_scheme == InterpolationScheme::MixedLinearCubic &&
            i < static_cast<std::size_t>(m_switchIndex)) {
            return m_values[i] + dx * (m_values[i + 1] - m_values[i]) / (m_times[i + 1] - m_times[i]);
        }
        if (m_scheme == InterpolationScheme::Akima ||
            m_scheme == InterpolationScheme::MixedLinearCubic) {
            return m_values[i] + dx * (m_a[i] + dx * (m_b[i] + dx * m_c[i]));
        }
        if (m_scheme == InterpolationScheme::MonotoneCubic) {
            const double h = m_times[i + 1] - m_times[i];
            const double u = (t - m_times[i]) / h;
            const double u2 = u * u;
            const double u3 = u2 * u;
            const double h00 = 2.0 * u3 - 3.0 * u2 + 1.0;
            const double h10 = u3 - 2.0 * u2 + u;
            const double h01 = -2.0 * u3 + 3.0 * u2;
            const double h11 = u3 - u2;
            return h00 * m_values[i] + h10 * h * m_monotoneSlopes[i] + h01 * m_values[i + 1] +
                   h11 * h * m_monotoneSlopes[i + 1];
        }
        if (m_scheme == InterpolationScheme::HymanSpline) {
            const double h = m_times[i + 1] - m_times[i];
            const double u = (t - m_times[i]) / h;
            const double u2 = u * u;
            const double u3 = u2 * u;
            const double h00 = 2.0 * u3 - 3.0 * u2 + 1.0;
            const double h10 = u3 - 2.0 * u2 + u;
            const double h01 = -2.0 * u3 + 3.0 * u2;
            const double h11 = u3 - u2;
            return h00 * m_values[i] + h10 * h * m_hymanSlopes[i] + h01 * m_values[i + 1] +
                   h11 * h * m_hymanSlopes[i + 1];
        }
        if (m_scheme == InterpolationScheme::TensionSpline) {
            const std::size_t j = i;
            const double h = m_times[j + 1] - m_times[j];
            const double sigma = m_tension;
            const double invLambda = m_tensionInvLambda[j];
            return m_tensionM[j] * (invLambda * std::sinh(sigma * (m_times[j + 1] - t))) +
                   m_tensionM[j + 1] * (invLambda * std::sinh(sigma * (t - m_times[j]))) +
                   (m_values[j] - m_tensionM[j] / (sigma * sigma)) *
                       ((m_times[j + 1] - t) / h) +
                   (m_values[j + 1] - m_tensionM[j + 1] / (sigma * sigma)) *
                       ((t - m_times[j]) / h);
        }
        return m_values[i] + dx * (m_values[i + 1] - m_values[i]) / (m_times[i + 1] - m_times[i]);
    }

    /// Space value (zero rate or log discount) with scheme-consistent
    /// extrapolation.
    DoubleT valueAt(double t) const {
        if (t <= m_times.front() && m_scheme != InterpolationScheme::Linear) {
            return m_values.front() + (t - m_times.front()) * m_firstSlope;
        }
        if (t >= m_times.back() && m_scheme != InterpolationScheme::Linear) {
            return m_values.back() + (t - m_times.back()) * m_lastSlope;
        }
        return valueOnSegment(segment(t), t);
    }

    DoubleT spaceValue(double t) const { return valueAt(t); }

    /// Initial slope of the space value (limit of z = y/t as t -> 0).
    DoubleT firstSlope() const {
        if (m_scheme == InterpolationScheme::Akima) {
            return m_a.front();
        }
        if (m_scheme == InterpolationScheme::TensionSpline) {
            return m_firstSlope;
        }
        if (m_scheme == InterpolationScheme::MonotoneCubic) {
            return m_monotoneSlopes.front();
        }
        if (m_scheme == InterpolationScheme::HymanSpline) {
            return m_hymanSlopes.front();
        }
        return (m_values[1] - m_values[0]) / (m_times[1] - m_times[0]);
    }

    static double primalValue(const DoubleT& x) {
        return quantape::math::detail::primalValue(x);
    }

    static DoubleT expImpl(const DoubleT& x) {
        if constexpr (std::is_same_v<DoubleT, double>) {
            return std::exp(x);
        } else {
            using std::exp;
            return exp(x); // ADL for AD scalars
        }
    }

    datetime::Date m_referenceDate{};
    datetime::DayCounter m_zeroDayCounter{datetime::DayCount::Actual365Fixed};
    std::vector<double> m_times;
    std::vector<DoubleT> m_zeros;
    std::vector<DoubleT> m_values;
    std::vector<DoubleT> m_a;
    std::vector<DoubleT> m_b;
    std::vector<DoubleT> m_c;
    std::vector<DoubleT> m_tensionM;
    std::vector<double> m_tensionInvLambda;
    std::vector<double> m_tensionSub;
    std::vector<double> m_tensionDiag;
    std::vector<double> m_tensionSuper;
    std::vector<double> m_monotoneSlopes;
    std::vector<double> m_hymanSlopes;
    DoubleT m_firstSlope{};
    DoubleT m_lastSlope{};
    double m_tension = 0.0;
    int m_switchIndex = 1;
    InterpolationSpace m_space = InterpolationSpace::LogDiscount;
    InterpolationScheme m_scheme = InterpolationScheme::Linear;
};

} // namespace quantape::markets
