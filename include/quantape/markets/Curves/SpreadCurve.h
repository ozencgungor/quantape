#pragma once

#include "quantape/markets/Curves/DiscountCurve.h"

#include <concepts>
#include <memory>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <vector>

namespace quantape::markets {
/**
 * @file SpreadCurve.h
 * @brief Additive spread curve over a parent discount curve
 *
 * State is spread nodes on zero rates:
 *   `z_c(t) = z_parent(t) + s(t)`,  `D_c(t) = exp(-z_c(t) t)`.
 * The additive model gives unit spread betas and a triangular risk chain.
 *
 * @tparam DoubleT Numeric type (`double` or an AD scalar).
 * @tparam ParentT Parent curve type (`DiscountCurve` or another `SpreadCurve`).
 */

namespace detail {

/// Scalar gate for the spread parent: the parent's declared zero-rate scalar
/// type must equal the child's. `SpreadCurve` asserts this so a mixed-scalar
/// parent fails at class instantiation with a direct message instead of
/// inside `zero()`.
template <typename DoubleT, typename ParentT>
concept SpreadParentScalar =
    std::same_as<std::decay_t<decltype(std::declval<const ParentT&>().zero(0.0))>, DoubleT>;

} // namespace detail

template <typename DoubleT, typename ParentT = DiscountCurve<DoubleT>>
class SpreadCurve {
    static_assert(detail::SpreadParentScalar<DoubleT, ParentT>,
                  "SpreadCurve: parent zero(0.0) scalar type must equal DoubleT");

public:
    SpreadCurve(std::shared_ptr<const ParentT> parent, std::vector<double> times,
                std::vector<DoubleT> spreads,
                InterpolationScheme scheme = InterpolationScheme::Linear, double tension = 0.0)
        : m_parent(std::move(parent)), m_spread(std::move(times), std::move(spreads),
                                                InterpolationSpace::Zero, scheme, tension) {
        if (!m_parent) {
            throw std::invalid_argument("SpreadCurve: null parent");
        }
    }

    /// Interpolated spread at `t` (continuously compounded zero spread).
    DoubleT spread(double t) const { return m_spread.zero(t); }

    /// Parent plus spread zero rate.
    DoubleT zero(double t) const { return m_parent->zero(t) + m_spread.zero(t); }

    /// Discount factor from the spread-adjusted zero rate.
    DoubleT discount(double t) const {
        if (t <= 0.0) {
            return DoubleT(1);
        }
        return expImpl(-zero(t) * t);
    }

    /// Continuously compounded forward over (t1, t2].
    DoubleT forward(double t1, double t2) const {
        if (!(t2 > t1)) {
            throw std::invalid_argument("SpreadCurve::forward: t2 must be > t1");
        }
        const DoubleT x1 = zero(t1) * t1;
        const DoubleT x2 = zero(t2) * t2;
        return (x2 - x1) / (t2 - t1);
    }

    /// `d z_child(t) / d z_child_i` for the child's solved nodes; with additive
    /// spreads this equals the spread-interpolation weights at `t`.
    void zeroNodeWeights(double t, std::vector<double>& weights) const {
        m_spread.zeroNodeWeights(t, weights);
    }
    const datetime::DayCounter& zeroDayCounter() const { return m_parent->zeroDayCounter(); }
    std::size_t size() const { return m_spread.size(); }

    const ParentT& parent() const { return *m_parent; }
    const std::shared_ptr<const ParentT>& parentPointer() const { return m_parent; }
    const DiscountCurve<DoubleT>& spreadNodes() const { return m_spread; }

private:
    static DoubleT expImpl(const DoubleT& x) {
        if constexpr (std::is_same_v<DoubleT, double>) {
            return std::exp(x);
        } else {
            using std::exp;
            return exp(x);
        }
    }

    std::shared_ptr<const ParentT> m_parent;
    DiscountCurve<DoubleT> m_spread;
};

/// Curves usable by pricing code (deterministic providers): both satisfy it.
/// The check is signature-only: the declared decayed return types must equal
/// `DoubleT` exactly. A class whose declared scalar is `DoubleT` but whose
/// body mixes scalars still satisfies this concept and guards itself instead
/// (see the `SpreadCurve` parent-scalar `static_assert`).
template <typename P, typename DoubleT>
concept CurveProvider = requires(const P& provider, double t1, double t2) {
    requires std::same_as<std::decay_t<decltype(provider.zero(t1))>, DoubleT>;
    requires std::same_as<std::decay_t<decltype(provider.discount(t1))>, DoubleT>;
    requires std::same_as<std::decay_t<decltype(provider.forward(t1, t2))>, DoubleT>;
};

/// A `CurveProvider` that also exposes its solved node grid and interpolation
/// weights, so it can back a spread curve or a chained forecast curve.
template <typename C>
concept CurveNodeProvider =
    CurveProvider<C, double> && requires(const C& curve, double t, std::vector<double>& weights) {
        { curve.size() } -> std::convertible_to<std::size_t>;
        curve.zeroNodeWeights(t, weights);
        { curve.zeroDayCounter() } -> std::same_as<const datetime::DayCounter&>;
    };

} // namespace quantape::markets
