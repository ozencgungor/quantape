#pragma once

#include "quantape/datetime/Date.h"
#include "quantape/datetime/DayCounter.h"
#include "quantape/datetime/TimeConversion.h"
#include "quantape/markets/Curves/Curve.h"
#include "quantape/markets/Curves/CurveSchemeState.h"

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
 * The curve stores the knot times, node zeros, space values and a
 * `detail::CurveSchemeState` holding only the active scheme's coefficients
 * (cubic a/b/c, tension second derivatives / tridiagonal system, or pinned
 * Hermite slopes). Evaluation is allocation-free and branch-light; scheme
 * dispatch is one `std::visit` per call through the internal variant.
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
    Linear,           ///< Linear in the chosen space
    Akima,            ///< Akima cubic (local C1)
    TensionSpline,    ///< Exponential tension spline (C2, sigma-controlled locality)
    HymanSpline,      ///< Natural cubic spline with Hyman's monotone node-slope
                      ///< filter (C1, monotone; in-range risk weights are
                      ///< unavailable for this scheme)
    MonotoneCubic,    ///< Steffen shape-preserving cubic (C1, monotone; in-range
                      ///< risk weights are unavailable for this scheme)
    MixedLinearCubic, ///< Linear up to `switchIndex` segments, Akima beyond
};

constexpr std::string_view interpolationSpaceName(InterpolationSpace space) noexcept {
    switch (space) {
        case InterpolationSpace::Zero:
            return "Zero";
        case InterpolationSpace::LogDiscount:
            return "LogDiscount";
    }
    return "Unknown";
}

constexpr std::string_view interpolationSchemeName(InterpolationScheme scheme) noexcept {
    switch (scheme) {
        case InterpolationScheme::Linear:
            return "Linear";
        case InterpolationScheme::Akima:
            return "Akima";
        case InterpolationScheme::TensionSpline:
            return "TensionSpline";
        case InterpolationScheme::HymanSpline:
            return "HymanSpline";
        case InterpolationScheme::MonotoneCubic:
            return "MonotoneCubic";
        case InterpolationScheme::MixedLinearCubic:
            return "MixedLinearCubic";
    }
    return "Unknown";
}

namespace detail {
struct CurveTrialUpdater;
}

template <typename DoubleT>
class DiscountCurve {
    friend struct detail::CurveTrialUpdater;

public:
    /// Times constructor (analysis / benchmarks; no calendar dates involved).
    DiscountCurve(std::vector<double> times, std::vector<DoubleT> zeros,
                  InterpolationSpace space = InterpolationSpace::LogDiscount,
                  InterpolationScheme scheme = InterpolationScheme::Linear, double tension = 0.0,
                  int switchIndex = 1)
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
                  InterpolationScheme scheme = InterpolationScheme::Linear, double tension = 0.0,
                  int switchIndex = 1)
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
            throw std::invalid_argument(
                "DiscountCurve::compoundedRate: non-positive year fraction");
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
            const DoubleT simple =
                (discount(fixingTimes[k - 1]) / discount(fixingTimes[k]) - 1.0) / tau;
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
        const DoubleT x1 = m_space == InterpolationSpace::Zero ? valueAt(t1) * t1 : spaceValue(t1);
        const DoubleT x2 = m_space == InterpolationSpace::Zero ? valueAt(t2) * t2 : spaceValue(t2);
        return (x2 - x1) / (t2 - t1);
    }

    /// Interpolation-only weights: `spaceValue(t) = sum_i w_i * m_values[i]`
    /// for `t` inside the node range (no extrapolation). Local schemes give
    /// few nonzeros; the cubic schemes use the coefficient Jacobian
    /// precomputed at initialization.
    void spaceValueWeights(double t, std::vector<double>& weights) const {
        m_state.spaceValueWeights(m_times, t, weights);
    }

    /// `d z(t) / d z_i` for solved node `i` (`t > 0`; beyond the last node the
    /// scheme's extrapolated weights are used). Node 0 (t = 0) is fixed and its
    /// weight is left at zero.
    void zeroNodeWeights(double t, std::vector<double>& weights) const {
        if (!(t >= 0.0)) {
            throw std::invalid_argument("DiscountCurve::zeroNodeWeights: t must be non-negative");
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
        m_state.gridValues(m_times, m_values, times, out);
        return out;
    }

    /// Space values at `times` written into a caller-owned buffer (resized to
    /// `times.size()`); allocation-free when `out` already has capacity.
    void spaceValuesInto(const std::vector<double>& times, std::vector<DoubleT>& out) const {
        m_state.gridValues(m_times, m_values, times, out);
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
    /// Weights beyond the last node: the scheme extends its terminal
    /// derivative, so the space weight is `delta_{i, n-1}` plus the
    /// precomputed terminal-slope gradient times the extension length. Only
    /// defined for positive query times (the pricing path treats t <= 0 as
    /// D = 1, which carries no node risk).
    void extrapolatedZeroWeights(double t, std::vector<double>& weights) const {
        if (!(t > 0.0)) {
            throw std::invalid_argument(
                "DiscountCurve::extrapolatedZeroWeights: t must be positive");
        }
        const std::size_t n = m_times.size();
        weights.assign(n, 0.0);
        const std::vector<double>& gradient = m_state.lastSlopeGradient();
        const double extension = t - m_times.back();
        const double invT = 1.0 / t;
        for (std::size_t i = 1; i < n; ++i) {
            double spaceWeight = (i + 1 == n ? 1.0 : 0.0) + extension * gradient[i];
            if (m_space == InterpolationSpace::LogDiscount) {
                spaceWeight *= m_times[i] * invT;
            }
            weights[i] = spaceWeight;
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
            throw std::invalid_argument(
                "DiscountCurve: need at least two nodes (t=0 and a pillar)");
        }
        for (std::size_t i = 1; i < m_times.size(); ++i) {
            if (!(m_times[i] > m_times[i - 1])) {
                throw std::invalid_argument(
                    "DiscountCurve: node times must be strictly increasing");
            }
        }
        // Akima and the other local cubics act on the chosen space values, so
        // they are available in both Zero and LogDiscount space; the shape
        // guarantees (monotonicity, positivity) are properties of the space.

        if (m_scheme == InterpolationScheme::TensionSpline && !(m_tension > 0.0)) {
            throw std::invalid_argument("DiscountCurve: TensionSpline needs a positive tension");
        }
        if (m_scheme == InterpolationScheme::MixedLinearCubic) {
            // Short trial curves used inside bootstrap may not have enough
            // segments for the configured switch: degenerate to linear.
            const int maxIndex = static_cast<int>(m_times.size()) - 1;
            m_switchIndex = std::max(1, std::min(m_switchIndex, maxIndex));
        }
        rebuildValuesAndState();
    }

    /// Recompute the space values and refresh the scheme state from the
    /// current node values (the node grid is unchanged).
    void rebuildValuesAndState() {
        m_values.resize(m_zeros.size());
        for (std::size_t i = 0; i < m_zeros.size(); ++i) {
            m_values[i] =
                m_space == InterpolationSpace::Zero ? m_zeros[i] : m_zeros[i] * m_times[i];
        }
        if (m_scheme == InterpolationScheme::Linear) {
            m_state.updateLinear(m_times, m_values);
            return;
        }
        using SchemeState = detail::CurveSchemeState<DoubleT>;
        switch (m_scheme) {
            case InterpolationScheme::Linear:
                break;
            case InterpolationScheme::Akima:
                m_state = SchemeState::makeAkima(m_times, m_values);
                break;
            case InterpolationScheme::TensionSpline:
                m_state = SchemeState::makeTension(m_times, m_values, m_tension);
                break;
            case InterpolationScheme::HymanSpline:
                m_state = SchemeState::makeHymanSpline(m_times, m_values);
                break;
            case InterpolationScheme::MonotoneCubic:
                m_state = SchemeState::makeMonotoneCubic(m_times, m_values);
                break;
            case InterpolationScheme::MixedLinearCubic:
                m_state = SchemeState::makeMixedLinearCubic(
                    m_times, m_values, static_cast<std::size_t>(m_switchIndex));
                break;
        }
    }

    /// Trial-curve update used by the bootstrap: replace one node value and
    /// refresh the state, reusing the existing buffers.
    void setNodeValue(std::size_t index, double value) {
        m_zeros[index] = DoubleT(value);
        m_values[index] =
            m_space == InterpolationSpace::Zero ? DoubleT(value) : DoubleT(value) * m_times[index];
        rebuildValuesAndState();
    }

    /// Space value (zero rate or log discount) with scheme-consistent
    /// extrapolation.
    DoubleT valueAt(double t) const { return m_state.valueAt(m_times, m_values, t); }

    DoubleT spaceValue(double t) const { return valueAt(t); }

    /// Initial slope of the space value (limit of z = y/t as t -> 0).
    DoubleT firstSlope() const { return m_state.firstSlope(); }

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
    detail::CurveSchemeState<DoubleT> m_state;
    double m_tension = 0.0;
    int m_switchIndex = 1;
    InterpolationSpace m_space = InterpolationSpace::LogDiscount;
    InterpolationScheme m_scheme = InterpolationScheme::Linear;
};

namespace detail {
/// Bootstrap-only access to the in-place trial-curve node update.
struct CurveTrialUpdater {
    template <typename DoubleT>
    static void setNode(DiscountCurve<DoubleT>& curve, std::size_t index, double value) {
        curve.setNodeValue(index, value);
    }
};
} // namespace detail

} // namespace quantape::markets
