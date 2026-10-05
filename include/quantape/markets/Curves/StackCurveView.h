#pragma once

#include "quantape/markets/Curves/DiscountCurve.h"
#include "quantape/markets/Curves/SpreadCurve.h"
#include "quantape/markets/Curves/TurnOverlay.h"

#include <cstddef>
#include <memory>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <vector>

namespace quantape::markets {
/**
 * @file StackCurveView.h
 * @brief Type-erased curve view and discount-function equality for stack risk
 *
 * The cold path of the stack risk layer is type-erased behind `StackCurveView`:
 * it exposes the node grid, discount factors and zero-node weights of a
 * `DiscountCurve` or any additive `SpreadCurve` chain without templating the
 * engine on the parent type. `make` wraps a caller-owned curve (its identity is
 * the caller's address, so a view made from the same object matches it
 * exactly), while `makeOwned` stores its own copy and uses that copy's address
 * as identity, so a destroyed temporary's address can never be mistaken for
 * another curve. `rebuildWithNode` returns a bumped copy for central
 * differences, and spread views keep a parent view so a bump can re-bind the
 * whole chain.
 *
 * `sameCurveValues` (concrete curves) and `sameCurveView` (views) share one
 * comparison core: same identity, or the same node grid with equal discounts at
 * every node and equal zero-node weights at every segment midpoint. The
 * midpoint weights fingerprint the interpolation space, scheme, tension and
 * switch index, so curves that only share their node values are not
 * interchangeable.
 */

/// Type-erased cold-path view over a double curve node provider.
class StackCurveView {
public:
    using Ptr = std::shared_ptr<const StackCurveView>;
    virtual ~StackCurveView() = default;
    virtual std::size_t size() const = 0;                 // nodes + 1
    virtual const std::vector<double>& times() const = 0; // native node times
    virtual double discount(double t) const = 0;
    virtual void zeroNodeWeights(double t, std::vector<double>& out) const = 0;
    virtual const datetime::DayCounter& zeroDayCounter() const = 0;
    virtual const void* identity() const = 0; // address of the underlying curve
    virtual const StackCurveView* parentView() const { return nullptr; } // spread curves only
    virtual Ptr rebuildWithNode(std::size_t node, double delta,
                                const Ptr& parentOverride) const = 0;

    /// Caller-input view: `identity()` is the caller's curve address, so a
    /// view made from the same object matches it exactly.
    template <CurveNodeProvider C>
    static Ptr make(const C& curve);

    /// Owned view: the adapter keeps its own curve copy and `identity()` is
    /// that copy's address. Rebuilt views use this factory so a destroyed
    /// temporary's address can never be mistaken for another curve.
    template <CurveNodeProvider C>
    static Ptr makeOwned(C&& curve);

private:
    template <typename C>
    class StackCurveViewOf;

    /// Selects the owning `StackCurveViewOf` constructor.
    struct OwnedTag {};
};

template <typename C>
class StackCurveView::StackCurveViewOf final : public StackCurveView {
    static_assert(std::is_same_v<C, void>, "StackCurveViewOf: unsupported curve type");
};

template <>
class StackCurveView::StackCurveViewOf<DiscountCurve<double>> final : public StackCurveView {
public:
    explicit StackCurveViewOf(const DiscountCurve<double>& curve)
        : m_curve(curve), m_identity(&curve) {}

    StackCurveViewOf(OwnedTag, DiscountCurve<double>&& curve)
        : m_curve(std::move(curve)), m_identity(&m_curve) {}

    std::size_t size() const override { return m_curve.size(); }
    const std::vector<double>& times() const override { return m_curve.times(); }
    double discount(double t) const override { return m_curve.discount(t); }
    void zeroNodeWeights(double t, std::vector<double>& out) const override {
        m_curve.zeroNodeWeights(t, out);
    }
    const datetime::DayCounter& zeroDayCounter() const override { return m_curve.zeroDayCounter(); }
    const void* identity() const override { return m_identity; }
    const DiscountCurve<double>& concreteParent() const { return m_curve; }

    Ptr rebuildWithNode(std::size_t node, double delta, const Ptr&) const override {
        std::vector<double> bumped = m_curve.zeros();
        bumped[node] += delta;
        // The node times are inverted to whole-day pillar dates on the curve's
        // own date clock, so the rebuilt curve keeps its reference date and
        // zero day counter and every Jacobian row stays on the same clock. The
        // times constructor is the fallback for grids whose times are not
        // exact whole-day year fractions.
        std::vector<datetime::Date> pillarDates;
        if (detail::recoverPillarDates(m_curve.referenceDate(), m_curve.zeroDayCounter(),
                                       m_curve.times(), pillarDates)) {
            return StackCurveView::makeOwned(DiscountCurve<double>(
                m_curve.referenceDate(), pillarDates, m_curve.zeroDayCounter(), std::move(bumped),
                m_curve.space(), m_curve.scheme(), m_curve.tension(), m_curve.switchIndex()));
        }
        return StackCurveView::makeOwned(
            DiscountCurve<double>(m_curve.times(), std::move(bumped), m_curve.space(),
                                  m_curve.scheme(), m_curve.tension(), m_curve.switchIndex()));
    }

private:
    DiscountCurve<double> m_curve;
    const void* m_identity = nullptr;
};

template <typename ParentT>
class StackCurveView::StackCurveViewOf<SpreadCurve<double, ParentT>> final : public StackCurveView {
public:
    using Curve = SpreadCurve<double, ParentT>;

    explicit StackCurveViewOf(const Curve& curve)
        : m_curve(curve), m_identity(&curve), m_parent(StackCurveView::make(curve.parent())) {}

    StackCurveViewOf(OwnedTag, Curve&& curve)
        : m_curve(std::move(curve)), m_identity(&m_curve),
          m_parent(StackCurveView::make(m_curve.parent())) {}

    std::size_t size() const override { return m_curve.size(); }
    const std::vector<double>& times() const override { return m_curve.spreadNodes().times(); }
    double discount(double t) const override { return m_curve.discount(t); }
    void zeroNodeWeights(double t, std::vector<double>& out) const override {
        m_curve.zeroNodeWeights(t, out);
    }
    const datetime::DayCounter& zeroDayCounter() const override { return m_curve.zeroDayCounter(); }
    const void* identity() const override { return m_identity; }
    const StackCurveView* parentView() const override { return m_parent.get(); }
    const Curve& concreteParent() const { return m_curve; }

    Ptr rebuildWithNode(std::size_t node, double delta, const Ptr& parentOverride) const override {
        const Ptr& parentRef = parentOverride ? parentOverride : m_parent;
        const auto parentAdapter =
            std::dynamic_pointer_cast<const StackCurveViewOf<ParentT>>(parentRef);
        if (parentAdapter == nullptr) {
            throw std::invalid_argument(
                "StackCurveViewOf<SpreadCurve>::rebuildWithNode: parent view type mismatch");
        }
        const DiscountCurve<double>& spreadNodes = m_curve.spreadNodes();
        std::vector<double> spreads = spreadNodes.zeros();
        spreads[node] += delta;
        // Spread nodes have no date constructor: the spread grid is carried by
        // times while the date metadata (reference date and zero clock) lives
        // on the parent copy, which is re-bound through the parent override or
        // the adapter's own parent and therefore keeps its own reconstruction.
        return StackCurveView::makeOwned(
            Curve(std::make_shared<ParentT>(parentAdapter->concreteParent()), spreadNodes.times(),
                  std::move(spreads), spreadNodes.scheme(), spreadNodes.tension()));
    }

private:
    Curve m_curve;
    const void* m_identity = nullptr;
    Ptr m_parent;
};

template <CurveNodeProvider C>
StackCurveView::Ptr StackCurveView::make(const C& curve) {
    return std::make_shared<StackCurveViewOf<std::remove_cvref_t<C>>>(curve);
}

template <CurveNodeProvider C>
StackCurveView::Ptr StackCurveView::makeOwned(C&& curve) {
    using CurveT = std::remove_cvref_t<C>;
    return std::make_shared<StackCurveViewOf<CurveT>>(OwnedTag{}, std::forward<C>(curve));
}

namespace detail {

/// Shared value comparison behind `sameCurveValues` and `sameCurveView`: the
/// same underlying object, or the same node grid with equal discounts at every
/// node and equal zero-node weights at every segment midpoint. The midpoint
/// weights fingerprint the interpolation space, scheme, tension and switch
/// index, so curves that only share their node values are not interchangeable.
template <typename LeftT, typename RightT>
inline bool sameCurveValuesCore(const LeftT& left, const void* leftIdentity, const RightT& right,
                                const void* rightIdentity) {
    if (leftIdentity == rightIdentity) {
        return true;
    }
    if (left.size() != right.size() || left.times() != right.times()) {
        return false;
    }
    const std::vector<double>& times = left.times();
    for (const double t : times) {
        if (left.discount(t) != right.discount(t)) {
            return false;
        }
    }
    std::vector<double> leftWeights;
    std::vector<double> rightWeights;
    for (std::size_t i = 0; i + 1 < times.size(); ++i) {
        const double midpoint = 0.5 * (times[i] + times[i + 1]);
        if (!(midpoint > 0.0)) {
            continue;
        }
        left.zeroNodeWeights(midpoint, leftWeights);
        right.zeroNodeWeights(midpoint, rightWeights);
        if (leftWeights != rightWeights) {
            return false;
        }
    }
    return true;
}

} // namespace detail

/// True when two discount curves define the same discount function: the same
/// object, or the same node grid with equal discounts at every node and equal
/// zero-node weights at every segment midpoint. The midpoint weights
/// fingerprint the interpolation space, scheme, tension and switch index, so
/// curves that only share their node values are not interchangeable.
inline bool sameCurveValues(const DiscountCurve<double>& left, const DiscountCurve<double>& right) {
    return detail::sameCurveValuesCore(left, &left, right, &right);
}

/// Discount-function equality of two views: the same underlying object or the
/// same node grid with equal discount factors at every node and equal zero-node
/// weights at every segment midpoint. The midpoint weights fingerprint the
/// interpolation space, scheme, tension and switch index, so views that only
/// share their node values are not interchangeable.
inline bool sameCurveView(const StackCurveView& left, const StackCurveView& right) {
    return detail::sameCurveValuesCore(left, left.identity(), right, right.identity());
}

} // namespace quantape::markets
