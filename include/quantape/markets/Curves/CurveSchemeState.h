#pragma once

#include "quantape/math/Autodiff/PrimalExtraction.h"
#include "quantape/math/Interpolations/CubicInterpolation.h"
#include "quantape/math/Interpolations/HymanSplineInterpolation.h"
#include "quantape/math/Interpolations/MonotoneCubicInterpolation.h"
#include "quantape/math/Interpolations/ProbeDual.h"
#include "quantape/math/Interpolations/TensionSplineInterpolation.h"
#include "quantape/math/Solvers/TridiagonalSolver.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <variant>
#include <vector>

namespace quantape::markets::detail {
/**
 * @file CurveSchemeState.h
 * @brief Per-scheme interpolation state for `DiscountCurve`
 *
 * The curve keeps only the active scheme's coefficients: Akima/Mixed store the
 * cubic a/b/c vectors plus the primal-pinned coefficient Jacobian used by the
 * risk weights, Tension stores the node second derivatives and the reduced
 * tridiagonal system, and the Hermite schemes (MonotoneCubic / HymanSpline)
 * store their pinned node slopes. Evaluation and weight assembly dispatch once
 * through the `std::variant`, so the curve class itself carries no scheme
 * branch chains.
 *
 * Scheme coefficients follow the curve's AD contract: Akima and tension
 * coefficients are built from the `DoubleT` node values, while the Steffen and
 * Hyman slopes (and every weight artifact) are pinned from primal values.
 *
 * @tparam DoubleT Numeric type (`double` or an AD scalar).
 */

/// First/last scheme slopes with the terminal-slope gradient used by the
/// extrapolated risk weights (`d lastSlope / d y_i`).
template <typename DoubleT>
struct SchemeSlopes {
    DoubleT firstSlope{};
    DoubleT lastSlope{};
    std::vector<double> lastSlopeGradient;
};

/// Cubic coefficients plus the primal coefficient Jacobian (flat segment x
/// node): coefficient a_i depends on the node values as
/// `a_i = sum_j weightA[i, j] y_j`, and likewise for b and c.
template <typename DoubleT>
struct CubicCoefficients {
    std::vector<DoubleT> a;
    std::vector<DoubleT> b;
    std::vector<DoubleT> c;
    std::vector<double> weightA;
    std::vector<double> weightB;
    std::vector<double> weightC;
};

template <typename DoubleT>
struct LinearSchemeState : SchemeSlopes<DoubleT> {};

template <typename DoubleT>
struct AkimaSchemeState : SchemeSlopes<DoubleT> {
    CubicCoefficients<DoubleT> cubic;
};

template <typename DoubleT>
struct MixedLinearCubicSchemeState : SchemeSlopes<DoubleT> {
    CubicCoefficients<DoubleT> cubic;
    std::size_t switchIndex = 1;
};

template <typename DoubleT>
struct TensionSchemeState : SchemeSlopes<DoubleT> {
    std::vector<DoubleT> m;
    std::vector<double> invLambda;
    std::vector<double> sub;
    std::vector<double> diag;
    std::vector<double> super;
    double tension = 0.0;
};

template <typename DoubleT>
struct MonotoneCubicSchemeState : SchemeSlopes<DoubleT> {
    std::vector<double> slopes;
};

template <typename DoubleT>
struct HymanSplineSchemeState : SchemeSlopes<DoubleT> {
    std::vector<double> slopes;
};

/// Segment index in [0, size - 2], with the standalone curve's range policy.
inline std::size_t locateSegment(const std::vector<double>& times, double t) {
    if (!std::isfinite(t)) {
        throw std::invalid_argument("DiscountCurve::segment: non-finite query time");
    }
    if (t <= times.front()) {
        return 0;
    }
    if (t >= times.back()) {
        return times.size() - 2;
    }
    const auto it = std::upper_bound(times.begin(), times.end(), t);
    return static_cast<std::size_t>(it - times.begin()) - 1;
}

template <typename DoubleT>
DoubleT cubicValueOnSegment(const CubicCoefficients<DoubleT>& cubic,
                            const std::vector<double>& times, const std::vector<DoubleT>& values,
                            std::size_t i, double t) {
    const double dx = t - times[i];
    return values[i] + dx * (cubic.a[i] + dx * (cubic.b[i] + dx * cubic.c[i]));
}

template <typename DoubleT>
DoubleT hermiteValueOnSegment(const std::vector<double>& slopes, const std::vector<double>& times,
                              const std::vector<DoubleT>& values, std::size_t i, double t) {
    const double h = times[i + 1] - times[i];
    const double u = (t - times[i]) / h;
    const double u2 = u * u;
    const double u3 = u2 * u;
    const double h00 = 2.0 * u3 - 3.0 * u2 + 1.0;
    const double h10 = u3 - 2.0 * u2 + u;
    const double h01 = -2.0 * u3 + 3.0 * u2;
    const double h11 = u3 - u2;
    return h00 * values[i] + h10 * h * slopes[i] + h01 * values[i + 1] + h11 * h * slopes[i + 1];
}

template <typename DoubleT>
DoubleT tensionValueOnSegment(const TensionSchemeState<DoubleT>& state,
                              const std::vector<double>& times, const std::vector<DoubleT>& values,
                              std::size_t j, double t) {
    const double h = times[j + 1] - times[j];
    const double sigma = state.tension;
    const double invLambda = state.invLambda[j];
    return state.m[j] * (invLambda * std::sinh(sigma * (times[j + 1] - t))) +
           state.m[j + 1] * (invLambda * std::sinh(sigma * (t - times[j]))) +
           (values[j] - state.m[j] / (sigma * sigma)) * ((times[j + 1] - t) / h) +
           (values[j + 1] - state.m[j + 1] / (sigma * sigma)) * ((t - times[j]) / h);
}

template <typename StateT, typename DoubleT>
DoubleT schemeValueOnSegment(const StateT& state, const std::vector<double>& times,
                             const std::vector<DoubleT>& values, std::size_t i, double t) {
    using State = std::decay_t<StateT>;
    if constexpr (std::is_same_v<State, LinearSchemeState<DoubleT>>) {
        const double dx = t - times[i];
        return values[i] + dx * (values[i + 1] - values[i]) / (times[i + 1] - times[i]);
    } else if constexpr (std::is_same_v<State, AkimaSchemeState<DoubleT>>) {
        return cubicValueOnSegment(state.cubic, times, values, i, t);
    } else if constexpr (std::is_same_v<State, MixedLinearCubicSchemeState<DoubleT>>) {
        if (i < state.switchIndex) {
            const double dx = t - times[i];
            return values[i] + dx * (values[i + 1] - values[i]) / (times[i + 1] - times[i]);
        }
        return cubicValueOnSegment(state.cubic, times, values, i, t);
    } else if constexpr (std::is_same_v<State, TensionSchemeState<DoubleT>>) {
        return tensionValueOnSegment(state, times, values, i, t);
    } else {
        return hermiteValueOnSegment(state.slopes, times, values, i, t);
    }
}

/// Linear in-range space weights.
inline void linearSpaceWeights(const std::vector<double>& times, std::size_t i, double t,
                               std::vector<double>& weights) {
    const double h = times[i + 1] - times[i];
    const double u = (t - times[i]) / h;
    weights[i] = 1.0 - u;
    weights[i + 1] = u;
}

/// Cubic (Akima / Mixed cubic tail) in-range space weights from the
/// precomputed coefficient Jacobian: one O(n) pass, no per-query probe.
template <typename DoubleT>
void cubicSpaceWeights(const CubicCoefficients<DoubleT>& cubic, const std::vector<double>& times,
                       std::size_t i, double t, std::vector<double>& weights) {
    const std::size_t n = times.size();
    const double dx = t - times[i];
    const std::size_t base = i * n;
    for (std::size_t j = 0; j < n; ++j) {
        weights[j] = (j == i ? 1.0 : 0.0) +
                     dx * (cubic.weightA[base + j] +
                           dx * (cubic.weightB[base + j] + dx * cubic.weightC[base + j]));
    }
}

/// Tension in-range space weights via the adjoint of the reduced tridiagonal
/// system for the second-derivative state.
template <typename DoubleT>
void tensionSpaceWeights(const TensionSchemeState<DoubleT>& state, const std::vector<double>& times,
                         std::size_t i, double t, std::vector<double>& weights) {
    const std::size_t n = times.size();
    const double h = times[i + 1] - times[i];
    const double sigma = state.tension;
    const double right = times[i + 1] - t;
    const double left = t - times[i];
    const double invLambda = state.invLambda[i];
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
    const std::vector<double> z =
        math::TridiagonalSolver<double>::solve(state.sub, state.diag, state.super, gM);
    for (std::size_t r = 1; r + 1 < n; ++r) {
        const double hLeft = times[r] - times[r - 1];
        const double hRight = times[r + 1] - times[r];
        weights[r - 1] += z[r - 1] / hLeft;
        weights[r] -= z[r - 1] * (1.0 / hRight + 1.0 / hLeft);
        weights[r + 1] += z[r - 1] / hRight;
    }
}

inline std::vector<double> primalValues(const std::vector<double>& values) {
    return values;
}

template <typename DoubleT>
std::vector<double> primalValues(const std::vector<DoubleT>& values) {
    std::vector<double> out(values.size());
    for (std::size_t i = 0; i < values.size(); ++i) {
        out[i] = math::detail::primalValue(values[i]);
    }
    return out;
}

/// Build the primal coefficient Jacobian used by `cubicSpaceWeights`.
template <typename DoubleT>
void buildCubicWeightJacobian(const std::vector<double>& times, const std::vector<DoubleT>& values,
                              CubicCoefficients<DoubleT>& cubic) {
    const std::size_t n = times.size();
    const std::size_t segments = n - 1;
    std::vector<math::detail::ProbeDual> y(n);
    for (std::size_t j = 0; j < n; ++j) {
        y[j] = math::detail::ProbeDual(math::detail::primalValue(values[j]), n);
        y[j].d[j] = 1.0;
    }
    std::vector<math::detail::ProbeDual> a;
    std::vector<math::detail::ProbeDual> b;
    std::vector<math::detail::ProbeDual> c;
    math::CubicInterpolation<double>::computeCoefficientsDual(
        times, y, math::CubicDerivativeApprox::Akima, false, a, b, c);
    cubic.weightA.assign(segments * n, 0.0);
    cubic.weightB.assign(segments * n, 0.0);
    cubic.weightC.assign(segments * n, 0.0);
    for (std::size_t i = 0; i < segments; ++i) {
        for (std::size_t j = 0; j < n; ++j) {
            cubic.weightA[i * n + j] = a[i].d[j];
            cubic.weightB[i * n + j] = b[i].d[j];
            cubic.weightC[i * n + j] = c[i].d[j];
        }
    }
}

/// Terminal tension-spline slope (same expression the curve's extrapolation
/// uses).
template <typename ScalarT>
ScalarT tensionTerminalSlope(double tension, double h, double invLambda,
                             const std::vector<ScalarT>& values,
                             const std::vector<ScalarT>& secondDerivatives, std::size_t last) {
    const double sigma = tension;
    const double sigmaSquared = sigma * sigma;
    return -secondDerivatives[last] * invLambda * sigma +
           secondDerivatives[last + 1] * invLambda * sigma * std::cosh(sigma * h) +
           (values[last + 1] - values[last] +
            (secondDerivatives[last] - secondDerivatives[last + 1]) / sigmaSquared) /
               h;
}

/// Terminal slope of the Steffen monotone limiter at the last node; the
/// formula is local to the last three nodes, so its gradient is too.
inline std::vector<double> monotoneTerminalSlopeGradient(const std::vector<double>& times,
                                                         const std::vector<double>& values) {
    const std::size_t n = times.size();
    std::vector<double> gradient(n, 0.0);
    const double hm = times[n - 1] - times[n - 2];
    const double sm = (values[n - 1] - values[n - 2]) / hm;
    if (n == 2) {
        gradient[n - 2] = -1.0 / hm;
        gradient[n - 1] = 1.0 / hm;
        return gradient;
    }
    const double hm1 = times[n - 2] - times[n - 3];
    const double sm1 = (values[n - 2] - values[n - 3]) / hm1;
    const double hSum = hm + hm1;
    const double d = ((2.0 * hm + hm1) * sm - hm * sm1) / hSum;
    if (d * sm <= 0.0) {
        return gradient;
    }
    if (std::abs(d) > 3.0 * std::abs(sm)) {
        gradient[n - 2] = -3.0 / hm;
        gradient[n - 1] = 3.0 / hm;
        return gradient;
    }
    const double dSm = (2.0 * hm + hm1) / hSum;
    const double dSm1 = -hm / hSum;
    gradient[n - 3] = dSm1 * (-1.0 / hm1);
    gradient[n - 2] = dSm * (-1.0 / hm) + dSm1 * (1.0 / hm1);
    gradient[n - 1] = dSm * (1.0 / hm);
    return gradient;
}

/// Terminal-slope gradient for the Hyman natural spline: one transposed
/// tridiagonal solve for the last interior second derivative, then the
/// Hyman filter branch derivative.
inline std::vector<double>
hymanTerminalSlopeGradient(const std::vector<double>& times, const std::vector<double>& values,
                           const std::vector<double>& secondDerivatives) {
    const std::size_t n = times.size();
    std::vector<double> gradient(n, 0.0);
    const double hLast = times[n - 1] - times[n - 2];
    const double sm = (values[n - 1] - values[n - 2]) / hLast;
    if (n == 2) {
        gradient[n - 2] = -1.0 / hLast;
        gradient[n - 1] = 1.0 / hLast;
        return gradient;
    }
    if (sm == 0.0) {
        return gradient;
    }
    const double d = sm + hLast * secondDerivatives[n - 2] / 6.0;
    if (std::abs(d) > 3.0 * std::abs(sm)) {
        gradient[n - 2] = -3.0 / hLast;
        gradient[n - 1] = 3.0 / hLast;
        return gradient;
    }
    if (d == 0.0) {
        return gradient; // V-shaped kink of the filtered magnitude
    }
    const double filterSign = sm * d > 0.0 ? 1.0 : -1.0;
    const std::size_t interior = n - 2;
    std::vector<double> sub(interior, 0.0);
    std::vector<double> diag(interior, 0.0);
    std::vector<double> super(interior, 0.0);
    std::vector<double> unit(interior, 0.0);
    for (std::size_t k = 0; k < interior; ++k) {
        const double hLeft = times[k + 1] - times[k];
        const double hRight = times[k + 2] - times[k + 1];
        sub[k] = k > 0 ? hLeft : 0.0;
        diag[k] = 2.0 * (hLeft + hRight);
        super[k] = k + 1 < interior ? hRight : 0.0;
    }
    unit[interior - 1] = 1.0;
    const std::vector<double> q = math::TridiagonalSolver<double>::solve(sub, diag, super, unit);
    for (std::size_t j = 1; j < n; ++j) {
        double gM = 0.0;
        if (j >= 2) {
            const double hPrev = times[j] - times[j - 1];
            gM += 6.0 * q[j - 2] / hPrev;
        }
        if (j <= interior) {
            const double hPrev = times[j] - times[j - 1];
            const double hNext = times[j + 1] - times[j];
            gM -= 6.0 * q[j - 1] * (1.0 / hPrev + 1.0 / hNext);
            if (j < interior) {
                gM += 6.0 * q[j] / hNext;
            }
        }
        gradient[j] = filterSign * (hLast / 6.0) * gM;
    }
    gradient[n - 2] -= filterSign / hLast;
    gradient[n - 1] += filterSign / hLast;
    return gradient;
}

/// Terminal-slope gradient for the tension spline: same adjoint construction
/// as Hyman, with the tension end-slope formula weights.
inline std::vector<double>
tensionTerminalSlopeGradient(const std::vector<double>& times, double tension,
                             const std::vector<double>& invLambda, const std::vector<double>& sub,
                             const std::vector<double>& diag, const std::vector<double>& super) {
    const std::size_t n = times.size();
    std::vector<double> gradient(n, 0.0);
    const std::size_t last = n - 2;
    const double hLast = times[last + 1] - times[last];
    const double sigma = tension;
    if (n == 2) {
        gradient[last] = -1.0 / hLast;
        gradient[last + 1] = 1.0 / hLast;
        return gradient;
    }
    const std::size_t interior = n - 2;
    std::vector<double> unit(interior, 0.0);
    unit[interior - 1] = 1.0;
    const std::vector<double> q = math::TridiagonalSolver<double>::solve(sub, diag, super, unit);
    const double mCoefficient = -invLambda[last] * sigma + 1.0 / (sigma * sigma * hLast);
    for (std::size_t j = 1; j < n; ++j) {
        double gM = 0.0;
        if (j >= 2) {
            const double hPrev = times[j] - times[j - 1];
            gM += q[j - 2] / hPrev;
        }
        if (j <= interior) {
            const double hPrev = times[j] - times[j - 1];
            const double hNext = times[j + 1] - times[j];
            gM -= q[j - 1] * (1.0 / hPrev + 1.0 / hNext);
            if (j < interior) {
                gM += q[j] / hNext;
            }
        }
        gradient[j] = mCoefficient * gM;
    }
    gradient[last] -= 1.0 / hLast;
    gradient[last + 1] += 1.0 / hLast;
    return gradient;
}

/**
 * @brief Variant-backed interpolation state for one curve scheme
 *
 * Constructed once by `DiscountCurve::initialize` through the scheme switch;
 * every later evaluation visits the active alternative exactly once.
 */
template <typename DoubleT>
class CurveSchemeState {
public:
    using Storage =
        std::variant<LinearSchemeState<DoubleT>, AkimaSchemeState<DoubleT>,
                     TensionSchemeState<DoubleT>, MonotoneCubicSchemeState<DoubleT>,
                     HymanSplineSchemeState<DoubleT>, MixedLinearCubicSchemeState<DoubleT>>;

    CurveSchemeState() = default;
    explicit CurveSchemeState(Storage storage) : m_storage(std::move(storage)) {}

    /// Space value at `t` with the scheme's own extrapolation policy (linear
    /// extension beyond either end for every scheme except `Linear`, which
    /// continues its edge segments).
    DoubleT valueAt(const std::vector<double>& times, const std::vector<DoubleT>& values,
                    double t) const {
        return std::visit(
            [&](const auto& state) -> DoubleT {
                using State = std::decay_t<decltype(state)>;
                if constexpr (std::is_same_v<State, LinearSchemeState<DoubleT>>) {
                    return schemeValueOnSegment(state, times, values, locateSegment(times, t), t);
                } else {
                    if (t <= times.front()) {
                        return values.front() + (t - times.front()) * state.firstSlope;
                    }
                    if (t >= times.back()) {
                        return values.back() + (t - times.back()) * state.lastSlope;
                    }
                    return schemeValueOnSegment(state, times, values, locateSegment(times, t), t);
                }
            },
            m_storage);
    }

    /// Space values at a sorted grid: one `std::visit` for the whole batch and
    /// one monotone segment pass, so no per-point variant dispatch.
    void gridValues(const std::vector<double>& times, const std::vector<DoubleT>& values,
                    const std::vector<double>& grid, std::vector<DoubleT>& out) const {
        out.resize(grid.size());
        if (grid.empty()) {
            return;
        }
        const std::size_t lastSegment = times.size() - 2;
        std::visit(
            [&](const auto& state) {
                using State = std::decay_t<decltype(state)>;
                std::size_t segment = 0;
                if constexpr (std::is_same_v<State, LinearSchemeState<DoubleT>>) {
                    for (std::size_t k = 0; k < grid.size(); ++k) {
                        const double t = grid[k];
                        while (segment < lastSegment && t >= times[segment + 1]) {
                            ++segment;
                        }
                        out[k] = schemeValueOnSegment(state, times, values, segment, t);
                    }
                } else {
                    for (std::size_t k = 0; k < grid.size(); ++k) {
                        const double t = grid[k];
                        if (t <= times.front()) {
                            out[k] = values.front() + (t - times.front()) * state.firstSlope;
                            continue;
                        }
                        if (t >= times.back()) {
                            out[k] = values.back() + (t - times.back()) * state.lastSlope;
                            continue;
                        }
                        while (segment < lastSegment && t >= times[segment + 1]) {
                            ++segment;
                        }
                        out[k] = schemeValueOnSegment(state, times, values, segment, t);
                    }
                }
            },
            m_storage);
    }

    /// Space value with a caller-supplied segment (batch materialization).
    DoubleT valueOnGrid(const std::vector<double>& times, const std::vector<DoubleT>& values,
                        std::size_t segment, double t) const {
        return std::visit(
            [&](const auto& state) -> DoubleT {
                using State = std::decay_t<decltype(state)>;
                if constexpr (!std::is_same_v<State, LinearSchemeState<DoubleT>>) {
                    if (t <= times.front()) {
                        return values.front() + (t - times.front()) * state.firstSlope;
                    }
                    if (t >= times.back()) {
                        return values.back() + (t - times.back()) * state.lastSlope;
                    }
                }
                return schemeValueOnSegment(state, times, values, segment, t);
            },
            m_storage);
    }

    DoubleT firstSlope() const {
        return std::visit([](const auto& state) { return state.firstSlope; }, m_storage);
    }

    DoubleT lastSlope() const {
        return std::visit([](const auto& state) { return state.lastSlope; }, m_storage);
    }

    const std::vector<double>& lastSlopeGradient() const {
        return std::visit(
            [](const auto& state) -> const std::vector<double>& { return state.lastSlopeGradient; },
            m_storage);
    }

    /// In-range interpolation weights (`spaceValue(t) = sum_i w_i values[i]`).
    /// MonotoneCubic and HymanSpline have no exact risk weights and throw.
    void spaceValueWeights(const std::vector<double>& times, double t,
                           std::vector<double>& weights) const {
        weights.assign(times.size(), 0.0);
        if (!(t >= times.front() && t <= times.back())) {
            throw std::invalid_argument("DiscountCurve::spaceValueWeights: t outside node range");
        }
        const std::size_t i = locateSegment(times, t);
        std::visit(
            [&](const auto& state) {
                using State = std::decay_t<decltype(state)>;
                if constexpr (std::is_same_v<State, LinearSchemeState<DoubleT>>) {
                    linearSpaceWeights(times, i, t, weights);
                } else if constexpr (std::is_same_v<State, AkimaSchemeState<DoubleT>>) {
                    cubicSpaceWeights(state.cubic, times, i, t, weights);
                } else if constexpr (std::is_same_v<State, MixedLinearCubicSchemeState<DoubleT>>) {
                    if (i < state.switchIndex) {
                        linearSpaceWeights(times, i, t, weights);
                    } else {
                        cubicSpaceWeights(state.cubic, times, i, t, weights);
                    }
                } else if constexpr (std::is_same_v<State, TensionSchemeState<DoubleT>>) {
                    tensionSpaceWeights(state, times, i, t, weights);
                } else if constexpr (std::is_same_v<State, MonotoneCubicSchemeState<DoubleT>>) {
                    throw std::invalid_argument(
                        "DiscountCurve: MonotoneCubic risk weights are unavailable");
                } else {
                    throw std::invalid_argument(
                        "DiscountCurve: HymanSpline risk weights are unavailable");
                }
            },
            m_storage);
    }

    static CurveSchemeState makeLinear(const std::vector<double>& times,
                                       const std::vector<DoubleT>& values) {
        const std::size_t n = times.size();
        LinearSchemeState<DoubleT> state;
        const double hFirst = times[1] - times[0];
        const double hLast = times[n - 1] - times[n - 2];
        state.firstSlope = (values[1] - values[0]) / hFirst;
        state.lastSlope = (values[n - 1] - values[n - 2]) / hLast;
        state.lastSlopeGradient.assign(n, 0.0);
        state.lastSlopeGradient[n - 2] = -1.0 / hLast;
        state.lastSlopeGradient[n - 1] = 1.0 / hLast;
        return CurveSchemeState(Storage(std::move(state)));
    }

    /// Refresh the linear state in place after node values changed without
    /// changing the node grid (trial-curve updates inside the bootstrap).
    void updateLinear(const std::vector<double>& times, const std::vector<DoubleT>& values) {
        LinearSchemeState<DoubleT>* state = std::get_if<LinearSchemeState<DoubleT>>(&m_storage);
        if (state == nullptr) {
            m_storage = Storage(LinearSchemeState<DoubleT>{});
            state = std::get_if<LinearSchemeState<DoubleT>>(&m_storage);
        }
        const std::size_t n = times.size();
        const double hFirst = times[1] - times[0];
        const double hLast = times[n - 1] - times[n - 2];
        state->firstSlope = (values[1] - values[0]) / hFirst;
        state->lastSlope = (values[n - 1] - values[n - 2]) / hLast;
        state->lastSlopeGradient.assign(n, 0.0);
        state->lastSlopeGradient[n - 2] = -1.0 / hLast;
        state->lastSlopeGradient[n - 1] = 1.0 / hLast;
    }

    static CurveSchemeState makeAkima(const std::vector<double>& times,
                                      const std::vector<DoubleT>& values) {
        const std::size_t n = times.size();
        const std::size_t last = n - 2;
        const math::CubicInterpolation<DoubleT> interp(times, values,
                                                       math::CubicDerivativeApprox::Akima);
        AkimaSchemeState<DoubleT> state;
        state.cubic.a = interp.aCoeffs();
        state.cubic.b = interp.bCoeffs();
        state.cubic.c = interp.cCoeffs();
        buildCubicWeightJacobian(times, values, state.cubic);
        state.firstSlope = state.cubic.a.front();
        const double h = times[last + 1] - times[last];
        state.lastSlope =
            state.cubic.a[last] + h * (2.0 * state.cubic.b[last] + h * 3.0 * state.cubic.c[last]);
        state.lastSlopeGradient.assign(n, 0.0);
        for (std::size_t j = 0; j < n; ++j) {
            state.lastSlopeGradient[j] = state.cubic.weightA[last * n + j] +
                                         h * (2.0 * state.cubic.weightB[last * n + j] +
                                              h * 3.0 * state.cubic.weightC[last * n + j]);
        }
        return CurveSchemeState(Storage(std::move(state)));
    }

    static CurveSchemeState makeMixedLinearCubic(const std::vector<double>& times,
                                                 const std::vector<DoubleT>& values,
                                                 std::size_t switchIndex) {
        const std::size_t n = times.size();
        const std::size_t last = n - 2;
        const math::CubicInterpolation<DoubleT> interp(times, values,
                                                       math::CubicDerivativeApprox::Akima);
        MixedLinearCubicSchemeState<DoubleT> state;
        state.switchIndex = switchIndex;
        state.cubic.a = interp.aCoeffs();
        state.cubic.b = interp.bCoeffs();
        state.cubic.c = interp.cCoeffs();
        buildCubicWeightJacobian(times, values, state.cubic);
        const double hLast = times[last + 1] - times[last];
        state.firstSlope = 0 < state.switchIndex ? (values[1] - values[0]) / (times[1] - times[0])
                                                 : state.cubic.a.front();
        state.lastSlope = last < state.switchIndex
                              ? (values[last + 1] - values[last]) / hLast
                              : state.cubic.a[last] + hLast * (2.0 * state.cubic.b[last] +
                                                               hLast * 3.0 * state.cubic.c[last]);
        state.lastSlopeGradient.assign(n, 0.0);
        if (last < state.switchIndex) {
            state.lastSlopeGradient[last] = -1.0 / hLast;
            state.lastSlopeGradient[last + 1] = 1.0 / hLast;
        } else {
            for (std::size_t j = 0; j < n; ++j) {
                state.lastSlopeGradient[j] =
                    state.cubic.weightA[last * n + j] +
                    hLast * (2.0 * state.cubic.weightB[last * n + j] +
                             hLast * 3.0 * state.cubic.weightC[last * n + j]);
            }
        }
        return CurveSchemeState(Storage(std::move(state)));
    }

    static CurveSchemeState makeTension(const std::vector<double>& times,
                                        const std::vector<DoubleT>& values, double tension) {
        const std::size_t n = times.size();
        const std::size_t last = n - 2;
        const math::TensionSplineInterpolation<DoubleT> interp(times, values, tension);
        TensionSchemeState<DoubleT> state;
        state.tension = tension;
        state.m = interp.secondDerivatives();
        const std::size_t segments = n - 1;
        state.invLambda.resize(segments);
        for (std::size_t j = 0; j < segments; ++j) {
            const double p = tension * (times[j + 1] - times[j]);
            state.invLambda[j] = 1.0 / (tension * tension * std::sinh(p));
        }
        state.sub = interp.systemSubDiagonal();
        state.diag = interp.systemDiagonal();
        state.super = interp.systemSuperDiagonal();
        const double sigma = tension;
        const double sigmaSquared = sigma * sigma;
        const double hLast = times[last + 1] - times[last];
        state.lastSlope =
            tensionTerminalSlope(tension, hLast, state.invLambda[last], values, state.m, last);
        const double hFirst = times[1] - times[0];
        const double invFirst = state.invLambda[0];
        state.firstSlope =
            -state.m[0] * invFirst * sigma * std::cosh(sigma * hFirst) +
            state.m[1] * invFirst * sigma +
            (values[1] - values[0] + (state.m[0] - state.m[1]) / sigmaSquared) / hFirst;
        state.lastSlopeGradient = tensionTerminalSlopeGradient(times, tension, state.invLambda,
                                                               state.sub, state.diag, state.super);
        return CurveSchemeState(Storage(std::move(state)));
    }

    static CurveSchemeState makeMonotoneCubic(const std::vector<double>& times,
                                              const std::vector<DoubleT>& values) {
        const math::MonotoneCubicInterpolation<DoubleT> interp(times, values);
        MonotoneCubicSchemeState<DoubleT> state;
        state.slopes = interp.slopes();
        state.firstSlope = state.slopes.front();
        state.lastSlope = state.slopes.back();
        state.lastSlopeGradient = monotoneTerminalSlopeGradient(times, primalValues(values));
        return CurveSchemeState(Storage(std::move(state)));
    }

    static CurveSchemeState makeHymanSpline(const std::vector<double>& times,
                                            const std::vector<DoubleT>& values) {
        const math::HymanSplineInterpolation<DoubleT> interp(times, values);
        HymanSplineSchemeState<DoubleT> state;
        state.slopes = interp.slopes();
        state.firstSlope = state.slopes.front();
        state.lastSlope = state.slopes.back();
        state.lastSlopeGradient =
            hymanTerminalSlopeGradient(times, primalValues(values), interp.secondDerivatives());
        return CurveSchemeState(Storage(std::move(state)));
    }

private:
    Storage m_storage;
};

} // namespace quantape::markets::detail
