#pragma once

#include <algorithm>
#include <cmath>
#include <concepts>
#include <cstddef>
#include <span>
#include <stdexcept>

namespace quantape::sde {
/**
 * @file Numeraire.h
 * @brief Collateral-currency numeraire (CSA account) view over a discount curve
 *
 * The numeraire is the collateral account: the inverse of the
 * collateral-currency zero bond,
 *
 *     N(t) = 1 / D_collateral(0, t),      N(0) = 1,
 *
 * i.e. the growth factor of posted cash collateral. The discount factor is one
 * over the numeraire, `D(t) = 1/N(t)` (`algorithmic_diff_for_callables.md`,
 * "the discount factor is one over the numeraire"). Deflated values are
 *
 *     Z(t) = V(t) / N(t) = V(t) * D_collateral(0, t),
 *
 * which is the deflated martingale of `sde_integration_plan.md` P3
 * (`E[Z_k] = Z_0`, `Z = V/N`, by the CSA numeraire). For a self-financing
 * value process this gives
 *
 *     E[Z(T) | F_t] = Z(t),      Z(0) = PV,
 *
 * because N(0) = 1, and a deterministic unit cashflow at T deflates at T to
 * `Z(T) = D(0, T)`, the risk-neutral PV. Do **not** use the decaying discount
 * factor as the numeraire: with `N = D` the deflated units are inverted, the
 * zero-vol limit is not a martingale (`E[Z_T] = 1/D(0,T) != Z_0`) and the PV
 * is off by `1/D(0,T)^2`.
 *
 * Evaluation is scalar-generic: `double`, `var` and `fvar<var>` flow through
 * the curve's own arithmetic (ADL), with no primal extraction. A state-to-curve
 * map is evaluated through its observation/payment `bond` API, a deterministic
 * provider through its `discount`/`zero` API.
 */
template <typename CurveT>
class Numeraire {
public:
    /// Unset view: every evaluation throws `std::invalid_argument`.
    Numeraire() = default;

    /// View of `*curve`; throws `std::invalid_argument` when `curve` is null.
    explicit Numeraire(const CurveT* curve) : m_curve(curve) {
        if (m_curve == nullptr) {
            throw std::invalid_argument("Numeraire: null curve view");
        }
    }

    /// View of `curve` (the caller keeps the curve alive).
    explicit Numeraire(const CurveT& curve) noexcept : m_curve(&curve) {}

    /// Collateral-account numeraire `N(t) = 1 / D_collateral(0, t)`, with
    /// N(0) = 1. `state` is the factor state consumed by a state-to-curve map
    /// and is ignored by a deterministic provider. A state-to-curve map must
    /// carry an observation at t = 0, and `t` must be on its payment grid.
    template <typename Scalar>
    Scalar numeraireValue(double t, std::span<const Scalar> state) const {
        const CurveT& curve = requireCurve();
        if (t <= 0.0) {
            return Scalar(1.0);
        }
        return Scalar(1.0) / discountAt(curve, t, state);
    }

    /// `value` expressed in numeraire units: `V(t) / N(t) = V(t) * D(0, t)`.
    /// For a self-financing value this is a martingale whose expectation is the
    /// PV; at t = 0 it is the value itself (N(0) = 1).
    template <typename Scalar>
    Scalar deflated(Scalar value, double t, std::span<const Scalar> state) const {
        const CurveT& curve = requireCurve();
        if (t <= 0.0) {
            return value;
        }
        return value * discountAt(curve, t, state);
    }

    /// One-step numeraire ratio `N(t1) / N(t2) = D(t2) / D(t1)`: the forward
    /// discount factor that rolls a raw value from t2 back to t1 (the rolling
    /// ratio `N(t_k)/N(t_{k+1})` of the LSMC regression target).
    template <typename Scalar>
    Scalar rollRatio(double t1, double t2, std::span<const Scalar> state) const {
        const CurveT& curve = requireCurve();
        return discountAt(curve, t2, state) / discountAt(curve, t1, state);
    }

private:
    /// Collateral-currency discount factor `D(0, t)` from the curve view; the
    /// reciprocal of `numeraireValue`. Kept as the primitive so that
    /// `deflated` is one multiplication (no reciprocal round trip) and a unit
    /// cashflow prices to `D(0, T)` exactly.
    template <typename Scalar>
    Scalar discountAt(const CurveT& curve, double t, std::span<const Scalar> state) const {
        if (t <= 0.0) {
            return Scalar(1.0);
        }
        if constexpr (requires {
                          curve.bond(std::size_t{}, std::size_t{}, state);
                          curve.obsTimes();
                          curve.payTimes();
                      }) {
            const std::span<const double> obs = curve.obsTimes();
            const std::span<const double> pay = curve.payTimes();
            const std::size_t k = indexAt(obs, 0.0, "Numeraire: curve has no observation at t = 0");
            const std::size_t j = indexAt(pay, t, "Numeraire: t is not on the payment grid");
            return curve.bond(k, j, state);
        } else if constexpr (requires {
                                 { curve.discount(t) } -> std::same_as<Scalar>;
                             }) {
            return curve.discount(t);
        } else if constexpr (requires { curve.zero(t); }) {
            const Scalar zero = Scalar(curve.zero(t));
            using std::exp;
            return exp(-zero * t);
        } else {
            return Scalar(curve.discount(t));
        }
    }

    const CurveT& requireCurve() const {
        if (m_curve == nullptr) {
            throw std::invalid_argument("Numeraire: curve view is null or unset");
        }
        return *m_curve;
    }

    static std::size_t indexAt(std::span<const double> times, double t, const char* message) {
        // Schedules are monotonically increasing, so an exact match on the
        // first node (the t = 0 observation, the first pay date) is the
        // first occurrence by construction; this is the common hot-path hit.
        if (!times.empty() && times.front() == t) {
            return 0;
        }
        if (times.size() <= 16) {
            for (std::size_t i = 0; i < times.size(); ++i) {
                if (times[i] == t) {
                    return i;
                }
            }
            throw std::invalid_argument(message);
        }
        const auto it = std::lower_bound(times.begin(), times.end(), t);
        if (it != times.end() && *it == t) {
            return static_cast<std::size_t>(it - times.begin());
        }
        // Sorted grids hit the branch above; the scan keeps the exact
        // first-occurrence behavior for malformed (unsorted) sequences.
        for (std::size_t i = 0; i < times.size(); ++i) {
            if (times[i] == t) {
                return i;
            }
        }
        throw std::invalid_argument(message);
    }

    const CurveT* m_curve = nullptr;
};

} // namespace quantape::sde
