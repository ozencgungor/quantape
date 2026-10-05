#pragma once

#include "quantape/markets/Curves/CurveBuilder.h"
#include "quantape/markets/Curves/CurveRisk.h"
#include "quantape/markets/Curves/CurveRiskReport.h"
#include "quantape/markets/Curves/SpreadCurve.h"
#include "quantape/markets/Curves/TurnOverlay.h"
#include "quantape/markets/Curves/XccyBasisBuilder.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <type_traits>
#include <unordered_map>
#include <utility>
#include <variant>
#include <vector>

namespace quantape::markets {
/**
 * @file StackRisk.h
 * @brief Total quote risk across a curve tree (basis and IRS children)
 *
 * Child instruments reference both curves, so the full instrument Jacobian is
 * block lower-triangular:
 *
 *   `F_full = [[F_p, 0], [C, F_c]]`,  `C = d r_child / d zeta_parent`,
 *
 * and with `L = [[I,0],[P,I]]` (`zeta_child = P zeta_parent + s`) the exact
 * total quote sensitivities are
 *
 *   `x_c        = F_c^{-T} g_s`
 *   `dV/dr_root = F_p^{-T} (g_p - C^T x_c)`
 *   `dV/dr_child = x_c`
 *
 * with `g_p = dV/dz_parent |_s`, `g_s = dV/ds`. No separate `P` propagation
 * term appears in quote space: the cross block `C = d r_c / d z_parent |_s`
 * already carries the parent dependence of the child quotes, and the spread
 * re-solve is encoded in the block inverse of `[[F_p,0],[C,F_c]]`.
 *
 * Children are additive spread curves bootstrapped from basis-swap or par IRS
 * pillars. The general engine supports arbitrary curve trees: each forecast
 * parent, exogenous discount curve and ancestor on their parent chains must
 * match a curve in the stack exactly (grid, discounts and interpolation), so a
 * block is never bound to a curve that only shares its node values.
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

struct StackRiskEntry {
    CurveRole role = CurveRole::Discount; ///< Curve role; turn pillars override per quote
    std::vector<CurveRole> quoteRoles;    ///< Per-quote role (`TurnOverlay` for turn knots)
    std::vector<double> quoteDeltas;
    std::vector<std::string> quoteLabels;  ///< Instrument kind + maturity tag (turn date)
    std::vector<int> quoteYears;           ///< Rounded maturity years (compatibility)
    std::vector<std::string> quoteBuckets; ///< Maturity tag (date / year / month)
};

/// Role buckets of a stack risk table. Sums each quote under its own role, so
/// direct turn knots report under `TurnOverlay` while the rest of their curve
/// keeps the curve role, and the buckets add back up to the full table.
inline std::vector<RiskBucket> stackRoleBuckets(const std::vector<StackRiskEntry>& entries) {
    std::vector<RiskBucket> buckets;
    const auto add = [&](std::string_view label, double delta) {
        for (RiskBucket& bucket : buckets) {
            if (bucket.label == label) {
                bucket.delta += delta;
                return;
            }
        }
        buckets.push_back(RiskBucket{std::string(label), delta});
    };
    for (const StackRiskEntry& entry : entries) {
        if (!entry.quoteRoles.empty() && entry.quoteRoles.size() != entry.quoteDeltas.size()) {
            throw std::invalid_argument("stackRoleBuckets: quote role size mismatch");
        }
        for (std::size_t j = 0; j < entry.quoteDeltas.size(); ++j) {
            const CurveRole role = entry.quoteRoles.empty() ? entry.role : entry.quoteRoles[j];
            add(curveRoleName(role), entry.quoteDeltas[j]);
        }
    }
    return buckets;
}

/// Overlay turn risk merged into a role-bucket list. Overlay turn amplitudes
/// are exogenous factors outside the quote Jacobian, so `stackRoleBuckets`
/// alone understates the `TurnOverlay` role; every `TurnRiskEntry` delta folds
/// into the `TurnOverlay` role bucket (or `role` when given), keeping role
/// sums complete. Labels stay on the original turn entries.
inline std::vector<RiskBucket> addTurnRiskBuckets(std::vector<RiskBucket> buckets,
                                                  const std::vector<TurnRiskEntry>& turns,
                                                  CurveRole role = CurveRole::TurnOverlay) {
    if (turns.empty()) {
        return buckets;
    }
    double total = 0.0;
    for (const TurnRiskEntry& turn : turns) {
        total += turn.delta;
    }
    const std::string label(curveRoleName(role));
    for (RiskBucket& bucket : buckets) {
        if (bucket.label == label) {
            bucket.delta += total;
            return buckets;
        }
    }
    buckets.push_back(RiskBucket{label, total});
    return buckets;
}

/// Role buckets of a stack risk table with exogenous overlay turn risk folded
/// in; equivalent to `addTurnRiskBuckets(stackRoleBuckets(entries), turns)`.
inline std::vector<RiskBucket> addTurnRiskBuckets(const std::vector<StackRiskEntry>& entries,
                                                  const std::vector<TurnRiskEntry>& turns,
                                                  CurveRole role = CurveRole::TurnOverlay) {
    return addTurnRiskBuckets(stackRoleBuckets(entries), turns, role);
}

/// Stack-level quote Hessian in quote space, row-major `dim x dim`.
struct StackQuoteGamma {
    std::size_t dim = 0;
    std::vector<std::string> quoteLabels;  ///< Same order as `stackQuoteRisk` entries
    std::vector<int> quoteYears;           ///< Rounded maturity years (compatibility)
    std::vector<std::string> quoteBuckets; ///< Maturity tag (date / year / month)
    std::vector<CurveRole> roles;          ///< One per quote
    std::vector<double> hessian;           ///< Row-major `dim x dim`

    /// Bounds- and size-checked access to the row-major Hessian.
    double at(std::size_t i, std::size_t j) const {
        if (dim == 0 || hessian.size() != dim * dim || i >= dim || j >= dim) {
            throw std::invalid_argument("StackQuoteGamma::at: index out of bounds");
        }
        return hessian[i * dim + j];
    }
};

struct StackChildInput {
    CurveRole role = CurveRole::Forecast;
    const SpreadCurve<double>* curve = nullptr;
    std::vector<ForecastPillar> pillars; ///< Basis-swap or par IRS instruments
    std::vector<double> dVdSpread;       ///< ∂V/∂s_i over the child nodes (node 0 = 0)
    /// Optional exogenous discount curve for IRS children. It must match the
    /// root grid and values; cross-currency children with their own foreign
    /// discount curves use the cross-currency driver instead.
    const DiscountCurve<double>* discountCurve = nullptr;
};

/// Analytic basis-swap rows: `fRow = ∂b/∂ζ_child` and
/// `cRow = ∂b/∂ζ_parent` for the quoted par basis spread `b`.
inline void basisSwapJacobianRows(const SpreadCurve<double>& child, const BasisPillar& pillar,
                                  const datetime::Date& referenceDate, std::vector<double>& fRow,
                                  std::vector<double>& cRow) {
    const DiscountCurve<double>& parent = child.parent();
    const datetime::Schedule schedule(referenceDate, pillar.maturity, pillar.floatTenor,
                                      pillar.calendar, pillar.businessDayConvention,
                                      datetime::DateGeneration::Forward, false,
                                      datetime::BusinessDayConvention::Unadjusted);
    const std::vector<datetime::Date>& dates = schedule.dates();
    const std::size_t periods = dates.size() - 1;
    const std::size_t nParent = parent.size();
    const std::size_t nChild = child.size();
    std::vector<double> times(periods + 1);
    std::vector<double> taus(periods);
    std::vector<double> parentDf(periods + 1);
    std::vector<double> childDf(periods + 1);
    std::vector<double> parentForward(periods);
    std::vector<double> childForward(periods);
    times[0] = datetime::yearFraction(referenceDate, dates[0], parent.zeroDayCounter());
    parentDf[0] = parent.discount(times[0]);
    childDf[0] = child.discount(times[0]);
    double annuity = 0.0;
    double numerator = 0.0;
    for (std::size_t k = 0; k < periods; ++k) {
        times[k + 1] = datetime::yearFraction(referenceDate, dates[k + 1], parent.zeroDayCounter());
        taus[k] = datetime::yearFraction(dates[k], dates[k + 1], pillar.quoteDayCounter);
        parentDf[k + 1] = parent.discount(times[k + 1]);
        childDf[k + 1] = child.discount(times[k + 1]);
        parentForward[k] = (parentDf[k] / parentDf[k + 1] - 1.0) / taus[k];
        childForward[k] = (childDf[k] / childDf[k + 1] - 1.0) / taus[k];
        annuity += taus[k] * parentDf[k + 1];
        numerator += taus[k] * parentDf[k + 1] * (childForward[k] - parentForward[k]);
    }
    if (!(annuity > 0.0)) {
        throw std::invalid_argument("basisSwapJacobianRows: non-positive annuity");
    }

    fRow.assign(nChild - 1, 0.0);
    cRow.assign(nParent - 1, 0.0);
    const auto weightsAt = [](const auto& curve, double time, std::vector<double>& out) {
        // D(0) = 1 is node-independent; sensitivities at t = 0 are zero.
        if (time <= 0.0) {
            out.assign(curve.size(), 0.0);
            return;
        }
        curve.zeroNodeWeights(time, out);
    };
    std::vector<double> weightsPrevious;
    std::vector<double> weightsCurrent;
    for (std::size_t k = 0; k < periods; ++k) {
        // Own-curve row: only the child forwards vary (parent frozen).
        weightsAt(child, times[k], weightsPrevious);
        weightsAt(child, times[k + 1], weightsCurrent);
        const double childFactor = (1.0 + taus[k] * childForward[k]) / taus[k];
        const double weight = taus[k] * parentDf[k + 1] / annuity;
        for (std::size_t j = 1; j < nChild; ++j) {
            fRow[j - 1] += weight * childFactor *
                           (-times[k] * weightsPrevious[j] + times[k + 1] * weightsCurrent[j]);
        }
        // Cross row: with the spread state fixed, zeta_child(t) = z_parent(t)
        // + s(t) moves with the parent, so BOTH forwards vary:
        //   d(f_c - f_p)/dz_j = (f_c - f_p) * (-t_prev w_prev + t_cur w_cur),
        // the (1 + tau f)/tau factors cancelling in the difference. Parent
        // discounts and the annuity vary as well.
        weightsAt(parent, times[k], weightsPrevious);
        weightsAt(parent, times[k + 1], weightsCurrent);
        const double forwardSpread = childForward[k] - parentForward[k];
        for (std::size_t j = 1; j < nParent; ++j) {
            const double dDiscount = -times[k + 1] * parentDf[k + 1] * weightsCurrent[j];
            const double dForwardSpread =
                forwardSpread * (-times[k] * weightsPrevious[j] + times[k + 1] * weightsCurrent[j]);
            const double dNumerator =
                taus[k] * (dDiscount * forwardSpread + parentDf[k + 1] * dForwardSpread);
            cRow[j - 1] += dNumerator;
        }
    }
    // d b = (dN A - N dA) / A^2, with dA_j = sum_k tau D'_k.
    std::vector<double> dAnnuity(nParent - 1, 0.0);
    for (std::size_t k = 0; k < periods; ++k) {
        weightsAt(parent, times[k + 1], weightsCurrent);
        for (std::size_t j = 1; j < nParent; ++j) {
            dAnnuity[j - 1] += taus[k] * (-times[k + 1] * parentDf[k + 1] * weightsCurrent[j]);
        }
    }
    for (std::size_t j = 0; j < nParent - 1; ++j) {
        cRow[j] = (cRow[j] * annuity - numerator * dAnnuity[j]) / (annuity * annuity);
    }
    if (!pillar.spreadOnParentLeg) {
        for (double& value : fRow) {
            value = -value;
        }
        for (double& value : cRow) {
            value = -value;
        }
    }
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

/// Analytic IRS rows for `R = floatPv / fixedAnnuity` with the floating
/// forwards from the child spread curve over `forecastParent` and every
/// discount factor from `discount`:
/// `fRow = d R / d spread_k` (parent and discount frozen),
/// `parentRow = d R / d z_forecastParent` (spreads and discount frozen) and
/// `discountRow = d R / d z_discount` (spreads and forecast parent frozen).
inline void irsSwapJacobianRows(const SpreadCurve<double>& child,
                                const DiscountCurve<double>& forecastParent,
                                const DiscountCurve<double>& discount, const IrsPillar& pillar,
                                const datetime::Date& referenceDate,
                                const datetime::DayCounter& zeroDayCounter,
                                std::vector<double>& fRow, std::vector<double>& parentRow,
                                std::vector<double>& discountRow) {
    const datetime::Date effective = pillar.start.serial() != 0 ? pillar.start : referenceDate;
    const datetime::Schedule floatSchedule(effective, pillar.maturity, pillar.floatTenor,
                                           pillar.floatCalendar, pillar.businessDayConvention,
                                           datetime::DateGeneration::Forward, false,
                                           datetime::BusinessDayConvention::Unadjusted);
    const datetime::Schedule fixedSchedule(effective, pillar.maturity, pillar.fixedTenor,
                                           pillar.fixedCalendar, pillar.businessDayConvention,
                                           datetime::DateGeneration::Forward, false,
                                           datetime::BusinessDayConvention::Unadjusted);
    const std::vector<datetime::Date>& floatDates = floatSchedule.dates();
    const std::vector<datetime::Date>& fixedDates = fixedSchedule.dates();
    const std::size_t nChild = child.size();
    const std::size_t nParent = forecastParent.size();
    const std::size_t nDiscount = discount.size();
    fRow.assign(nChild - 1, 0.0);
    parentRow.assign(nParent - 1, 0.0);
    discountRow.assign(nDiscount - 1, 0.0);
    const auto weightsAt = [](const auto& curve, double time, std::vector<double>& out) {
        if (time <= 0.0) {
            out.assign(curve.size(), 0.0);
            return;
        }
        curve.zeroNodeWeights(time, out);
    };
    std::vector<double> weights;
    std::vector<double> annuityRow(nDiscount - 1, 0.0);
    double annuity = 0.0;
    for (std::size_t j = 1; j < fixedDates.size(); ++j) {
        const double tau =
            datetime::yearFraction(fixedDates[j - 1], fixedDates[j], pillar.fixedDayCounter);
        const datetime::Date payDate = pillar.fixedCalendar.advance(
            fixedDates[j], datetime::Period(pillar.paymentLag, datetime::TimeUnit::Days),
            pillar.businessDayConvention);
        const double tPay = datetime::yearFraction(referenceDate, payDate, zeroDayCounter);
        const double discountPay = discount.discount(tPay);
        annuity += tau * discountPay;
        weightsAt(discount, tPay, weights);
        for (std::size_t i = 1; i < nDiscount; ++i) {
            annuityRow[i - 1] += -tau * tPay * discountPay * weights[i];
        }
    }
    if (!(annuity > 0.0)) {
        throw std::invalid_argument("irsSwapJacobianRows: non-positive fixed annuity");
    }

    // Coupon discount-factor sensitivities with the forward state frozen.
    std::vector<double> discountNumerator(nDiscount - 1, 0.0);
    std::vector<double> childPrev;
    std::vector<double> childCur;
    std::vector<double> parentPrev;
    std::vector<double> parentCur;
    std::vector<double> payWeights;
    double floatPv = 0.0;
    for (std::size_t k = 1; k < floatDates.size(); ++k) {
        const double tau =
            datetime::yearFraction(floatDates[k - 1], floatDates[k], pillar.floatDayCounter);
        const double tPrev =
            datetime::yearFraction(referenceDate, floatDates[k - 1], zeroDayCounter);
        const double tCur = datetime::yearFraction(referenceDate, floatDates[k], zeroDayCounter);
        const datetime::Date payDate = pillar.floatCalendar.advance(
            floatDates[k], datetime::Period(pillar.paymentLag, datetime::TimeUnit::Days),
            pillar.businessDayConvention);
        const double tPay = datetime::yearFraction(referenceDate, payDate, zeroDayCounter);
        const double discountPay = discount.discount(tPay);
        if (k == 1 && pillar.firstCouponFixed) {
            // A known fixing has no forward dependence; only its payment
            // discount factor moves with the discount curve.
            const double couponPv = tau * discountPay * pillar.firstCouponRate;
            floatPv += couponPv;
            weightsAt(discount, tPay, payWeights);
            for (std::size_t i = 1; i < nDiscount; ++i) {
                discountNumerator[i - 1] += -tPay * couponPv * payWeights[i];
            }
            continue;
        }
        const double childPrevious = child.discount(tPrev);
        const double childCurrent = child.discount(tCur);
        const double forward = (childPrevious / childCurrent - 1.0) / tau;
        floatPv += tau * discountPay * forward;
        const double ratio = childPrevious / childCurrent;
        weightsAt(child, tPrev, childPrev);
        weightsAt(child, tCur, childCur);
        weightsAt(forecastParent, tPrev, parentPrev);
        weightsAt(forecastParent, tCur, parentCur);
        weightsAt(discount, tPay, payWeights);
        // f = (R - 1) / tau with R = D_c(tPrev) / D_c(tCur), so
        // d f = (1 + tau f)/tau * (-tPrev dw + tCur dw) on the curve whose
        // nodes move. A child bump moves only the child curve; a parent bump
        // shifts the child zero curve one-for-one (zeta_child = zeta_parent +
        // s) and leaves the annuity on the frozen discount curve unchanged.
        for (std::size_t i = 1; i < nChild; ++i) {
            fRow[i - 1] +=
                (discountPay * ratio / annuity) * (-tPrev * childPrev[i] + tCur * childCur[i]);
        }
        for (std::size_t i = 1; i < nParent; ++i) {
            parentRow[i - 1] +=
                (discountPay * ratio / annuity) * (-tPrev * parentPrev[i] + tCur * parentCur[i]);
        }
        // Coupon PV is tau * Dp * f with f frozen: the direct discount-factor
        // term keeps tau.
        for (std::size_t i = 1; i < nDiscount; ++i) {
            discountNumerator[i - 1] += -tPay * tau * discountPay * forward * payWeights[i];
        }
    }
    const double denominator = annuity * annuity;
    for (std::size_t i = 1; i < nDiscount; ++i) {
        discountRow[i - 1] =
            (discountNumerator[i - 1] * annuity - floatPv * annuityRow[i - 1]) / denominator;
    }
}

/// Assemble the child own-curve Jacobian `F_c` (row-major `m_c x m_c`) and the
/// cross block `C` (row-major `m_c x m_p`) from basis pillars.
inline void assembleBasisJacobian(const SpreadCurve<double>& child,
                                  const std::vector<BasisPillar>& pillars,
                                  const datetime::Date& referenceDate, std::vector<double>& f,
                                  std::vector<double>& c) {
    if (child.size() != pillars.size() + 1) {
        throw std::invalid_argument(
            "assembleBasisJacobian: child nodes must match the pillar count");
    }
    const std::size_t mChild = pillars.size();
    const std::size_t mParent = child.parent().size() - 1;
    f.assign(mChild * mChild, 0.0);
    c.assign(mChild * mParent, 0.0);
    std::vector<double> fRow;
    std::vector<double> cRow;
    for (std::size_t j = 0; j < mChild; ++j) {
        basisSwapJacobianRows(child, pillars[j], referenceDate, fRow, cRow);
        for (std::size_t i = 0; i < mChild; ++i) {
            f[j * mChild + i] = fRow[i];
        }
        for (std::size_t i = 0; i < mParent; ++i) {
            c[j * mParent + i] = cRow[i];
        }
    }
}

/// Analytic rows of a synthetic money-market forecast pillar
/// `r = (D_f(t1) / D_f(t2) - 1) / tau` (deposit or forward-starting FRA):
/// `ownRow = d r / d spread_k` over the child spread nodes and
/// `parentRow = d r / d z_parent,k` over the forecast parent, whose nodes
/// shift the child zero curve one-for-one. The pillar never references a
/// discount curve, so there is no separate discount sensitivity.
inline void forecastSimpleJacobianRows(const SpreadCurve<double>& child,
                                       const ForecastPillar& pillar,
                                       const datetime::Date& referenceDate,
                                       std::vector<double>& ownRow,
                                       std::vector<double>& parentRow) {
    const DiscountCurve<double>& parent = child.parent();
    const datetime::Date start = pillar.start.serial() != 0 ? pillar.start : referenceDate;
    const datetime::Date maturity =
        pillar.calendar.adjust(pillar.maturity, pillar.businessDayConvention);
    const datetime::DayCounter& zeroDayCounter = child.zeroDayCounter();
    const double t1 = datetime::yearFraction(referenceDate, start, zeroDayCounter);
    const double t2 = datetime::yearFraction(referenceDate, maturity, zeroDayCounter);
    const double tau = datetime::yearFraction(start, maturity, pillar.quoteDayCounter);
    if (!(tau > 0.0)) {
        throw std::invalid_argument("forecastSimpleJacobianRows: non-positive accrual");
    }
    if (!(t1 >= 0.0)) {
        throw std::invalid_argument("forecastSimpleJacobianRows: start before the reference date");
    }
    const double ratio = child.discount(t1) / child.discount(t2);
    ownRow.assign(child.size() - 1, 0.0);
    parentRow.assign(parent.size() - 1, 0.0);
    const auto weightsAt = [](const auto& curve, double time, std::vector<double>& out) {
        // D(0) = 1 is node-independent; sensitivities at t = 0 are zero.
        if (time <= 0.0) {
            out.assign(curve.size(), 0.0);
            return;
        }
        curve.zeroNodeWeights(time, out);
    };
    std::vector<double> previous;
    std::vector<double> current;
    const double factor = ratio / tau;
    weightsAt(child, t1, previous);
    weightsAt(child, t2, current);
    for (std::size_t j = 1; j < child.size(); ++j) {
        ownRow[j - 1] = factor * (-t1 * previous[j] + t2 * current[j]);
    }
    weightsAt(parent, t1, previous);
    weightsAt(parent, t2, current);
    for (std::size_t j = 1; j < parent.size(); ++j) {
        parentRow[j - 1] = factor * (-t1 * previous[j] + t2 * current[j]);
    }
}

/// Core own/parent rows of a forecast-curve rate future, shared by the
/// concrete-curve and view-based entry points: the period forward or the
/// business-day-grid averaged rate on `child` plus the stored convexity
/// adjustment. Convexity is additive and drops out of the row. `ownRow` picks
/// up the child spread-node weights; `parentRow` the `parent` weights, whose
/// nodes shift the child zero curve one-for-one.
template <typename ChildT, typename ParentT>
inline void forecastFutureRowsCore(const ChildT& child, const ParentT& parent,
                                   const ForecastPillar& pillar,
                                   const datetime::Date& referenceDate, std::vector<double>& ownRow,
                                   std::vector<double>& parentRow) {
    const datetime::DayCounter& zeroDayCounter = child.zeroDayCounter();
    ownRow.assign(child.size() - 1, 0.0);
    parentRow.assign(parent.size() - 1, 0.0);
    const auto weightsAt = [](const auto& curve, double time, std::vector<double>& out) {
        // D(0) = 1 is node-independent; sensitivities at t = 0 are zero.
        if (time <= 0.0) {
            out.assign(curve.size(), 0.0);
            return;
        }
        curve.zeroNodeWeights(time, out);
    };
    const double startTime = datetime::yearFraction(referenceDate, pillar.start, zeroDayCounter);
    if (!(startTime >= 0.0)) {
        throw std::invalid_argument(
            "forecastFutureRowsCore: futures fixing before the reference date");
    }
    if (pillar.futureStyle == FutureStyle::Averaged &&
        pillar.averagingStyle == AveragingStyle::Arithmetic) {
        const std::vector<datetime::Date> fixings =
            businessDayFixings(pillar.calendar, pillar.start, pillar.maturity);
        const std::size_t periods = fixings.size() - 1;
        if (periods == 0) {
            throw std::invalid_argument(
                "forecastFutureRowsCore: empty averaged futures reference period");
        }
        std::vector<double> weightsStart;
        std::vector<double> weightsEnd;
        for (std::size_t k = 0; k < periods; ++k) {
            const double tau =
                datetime::yearFraction(fixings[k], fixings[k + 1], pillar.quoteDayCounter);
            if (!(tau > 0.0)) {
                throw std::invalid_argument(
                    "forecastFutureRowsCore: non-positive averaged futures accrual");
            }
            const double t1 = datetime::yearFraction(referenceDate, fixings[k], zeroDayCounter);
            const double t2 = datetime::yearFraction(referenceDate, fixings[k + 1], zeroDayCounter);
            const double d1 = child.discount(t1);
            const double d2 = child.discount(t2);
            const double ratio = d1 / d2;
            const double scale = 1.0 / (tau * d2);
            const auto accumulate = [&](const auto& weightsCurve, std::vector<double>& row) {
                weightsAt(weightsCurve, t1, weightsStart);
                weightsAt(weightsCurve, t2, weightsEnd);
                for (std::size_t j = 1; j < row.size() + 1; ++j) {
                    row[j - 1] +=
                        scale * (-t1 * d1 * weightsStart[j] + ratio * t2 * d2 * weightsEnd[j]);
                }
            };
            accumulate(child, ownRow);
            accumulate(parent, parentRow);
        }
        for (double& value : ownRow) {
            value /= static_cast<double>(periods);
        }
        for (double& value : parentRow) {
            value /= static_cast<double>(periods);
        }
        return;
    }
    // Simple, compounded and compounded-average futures: the daily compounded
    // products telescope to the period forward, so the simple-rate row is
    // exact.
    const double tau =
        datetime::yearFraction(pillar.start, pillar.maturity, pillar.quoteDayCounter);
    if (!(tau > 0.0)) {
        throw std::invalid_argument("forecastFutureRowsCore: non-positive futures accrual");
    }
    const double endTime = datetime::yearFraction(referenceDate, pillar.maturity, zeroDayCounter);
    const double d1 = child.discount(startTime);
    const double d2 = child.discount(endTime);
    std::vector<double> weightsStart;
    std::vector<double> weightsEnd;
    const double factor = d1 / (tau * d2);
    weightsAt(child, startTime, weightsStart);
    weightsAt(child, endTime, weightsEnd);
    for (std::size_t j = 1; j < ownRow.size() + 1; ++j) {
        ownRow[j - 1] = factor * (-startTime * weightsStart[j] + endTime * weightsEnd[j]);
    }
    weightsAt(parent, startTime, weightsStart);
    weightsAt(parent, endTime, weightsEnd);
    for (std::size_t j = 1; j < parentRow.size() + 1; ++j) {
        parentRow[j - 1] = factor * (-startTime * weightsStart[j] + endTime * weightsEnd[j]);
    }
}

/// Analytic rows of a forecast-curve rate future:
/// `ownRow = d r / d s_k` over the child spread nodes and
/// `parentRow = d r / d z_parent,k` over the forecast parent, whose nodes
/// shift the child zero curve one-for-one. The pillar never references a
/// discount curve, so there is no separate discount sensitivity.
inline void forecastFutureJacobianRows(const SpreadCurve<double>& child,
                                       const ForecastPillar& pillar,
                                       const datetime::Date& referenceDate,
                                       std::vector<double>& ownRow,
                                       std::vector<double>& parentRow) {
    forecastFutureRowsCore(child, child.parent(), pillar, referenceDate, ownRow, parentRow);
}

/// Own-curve, forecast-parent and discount rows of one forecast pillar.
/// Basis swaps use the parent row as the cross row and report a zero discount
/// row; IRS pillars report the parent and discount rows separately. Synthetic
/// deposits, FRAs and rate futures move with the child spread nodes and the
/// forecast parent one-for-one and report a zero discount row.
inline void forecastPillarJacobianRows(const SpreadCurve<double>& child,
                                       const ForecastPillar& pillar,
                                       const datetime::Date& referenceDate,
                                       const DiscountCurve<double>* discountCurve,
                                       std::vector<double>& fRow, std::vector<double>& parentRow,
                                       std::vector<double>& discountRow) {
    const DiscountCurve<double>& forecastParent = child.parent();
    const DiscountCurve<double>& discount =
        discountCurve != nullptr ? *discountCurve : forecastParent;
    switch (pillar.kind) {
        case ForecastPillar::Kind::Deposit:
        case ForecastPillar::Kind::Fra:
            forecastSimpleJacobianRows(child, pillar, referenceDate, fRow, parentRow);
            discountRow.assign(discount.size() - 1, 0.0);
            return;
        case ForecastPillar::Kind::Future:
            forecastFutureJacobianRows(child, pillar, referenceDate, fRow, parentRow);
            discountRow.assign(discount.size() - 1, 0.0);
            return;
        case ForecastPillar::Kind::BasisSwap:
            basisSwapJacobianRows(child, pillar.basis, referenceDate, fRow, parentRow);
            discountRow.assign(discount.size() - 1, 0.0);
            return;
        case ForecastPillar::Kind::Irs:
            irsSwapJacobianRows(child, forecastParent, discount, pillar.irs, referenceDate,
                                child.zeroDayCounter(), fRow, parentRow, discountRow);
            return;
    }
    throw std::invalid_argument("forecastPillarJacobianRows: unknown forecast pillar kind");
}

/// Assemble the child own-curve Jacobian `F_c` (row-major `m_c x m_c`) and the
/// cross block `C` (row-major `m_c x m_p`) over the forecast-parent nodes.
/// When an IRS child discounts on its forecast parent (the same object, or a
/// curve with the same grid and values) the cross block is the sum of the
/// parent and discount rows, because the forward and discounting curves are
/// then the same curve.
inline void assembleForecastJacobian(const SpreadCurve<double>& child,
                                     const std::vector<ForecastPillar>& pillars,
                                     const datetime::Date& referenceDate,
                                     const DiscountCurve<double>* discountCurve,
                                     std::vector<double>& f, std::vector<double>& c) {
    if (child.size() != pillars.size() + 1) {
        throw std::invalid_argument(
            "assembleForecastJacobian: child nodes must match the pillar count");
    }
    const std::size_t mChild = pillars.size();
    const DiscountCurve<double>& forecastParent = child.parent();
    const std::size_t mParent = forecastParent.size() - 1;
    const DiscountCurve<double>& discount =
        discountCurve != nullptr ? *discountCurve : forecastParent;
    const bool discountIsParent =
        &discount == &forecastParent || sameCurveValues(discount, forecastParent);
    f.assign(mChild * mChild, 0.0);
    c.assign(mChild * mParent, 0.0);
    std::vector<double> fRow;
    std::vector<double> parentRow;
    std::vector<double> discountRow;
    for (std::size_t j = 0; j < mChild; ++j) {
        forecastPillarJacobianRows(child, pillars[j], referenceDate, discountCurve, fRow, parentRow,
                                   discountRow);
        for (std::size_t i = 0; i < mChild; ++i) {
            f[j * mChild + i] = fRow[i];
        }
        for (std::size_t i = 0; i < mParent; ++i) {
            c[j * mParent + i] = parentRow[i] + (discountIsParent ? discountRow[i] : 0.0);
        }
    }
}

/// Discount-function equality of two views: the same underlying object or the
/// same node grid with equal discount factors at every node and equal zero-node
/// weights at every segment midpoint. The midpoint weights fingerprint the
/// interpolation space, scheme, tension and switch index, so views that only
/// share their node values are not interchangeable.
inline bool sameCurveView(const StackCurveView& left, const StackCurveView& right) {
    return detail::sameCurveValuesCore(left, left.identity(), right, right.identity());
}

/// View-based basis rows for an arbitrary forecast parent and discount curve:
/// `ownRow` over the child spread nodes, `parentRow` over the forecast-parent
/// nodes (both forwards move with the parent, the annuity is frozen), and
/// `discountRow` over the discount nodes (discount factors only). The optional
/// weight sources let the same curve-space partial be expressed on an ancestor
/// curve's node grid, which is how a depth-2 chain's rows reach the root block.
inline void basisSwapJacobianRowsView(const StackCurveView& child, const StackCurveView& parent,
                                      const StackCurveView& discount, const BasisPillar& pillar,
                                      const datetime::Date& referenceDate,
                                      std::vector<double>& ownRow, std::vector<double>& parentRow,
                                      std::vector<double>& discountRow,
                                      const StackCurveView* parentWeights = nullptr,
                                      const StackCurveView* discountWeights = nullptr) {
    const datetime::Schedule schedule(referenceDate, pillar.maturity, pillar.floatTenor,
                                      pillar.calendar, pillar.businessDayConvention,
                                      datetime::DateGeneration::Forward, false,
                                      datetime::BusinessDayConvention::Unadjusted);
    const std::vector<datetime::Date>& dates = schedule.dates();
    const std::size_t periods = dates.size() - 1;
    const std::size_t nChild = child.size();
    const StackCurveView& parentWeightCurve = parentWeights != nullptr ? *parentWeights : parent;
    const StackCurveView& discountWeightCurve =
        discountWeights != nullptr ? *discountWeights : discount;
    const std::size_t nParent = parentWeightCurve.size();
    const std::size_t nDiscount = discountWeightCurve.size();
    const datetime::DayCounter& zeroDayCounter = discount.zeroDayCounter();
    std::vector<double> times(periods + 1);
    std::vector<double> taus(periods);
    std::vector<double> discountDf(periods + 1);
    std::vector<double> childDf(periods + 1);
    std::vector<double> parentDf(periods + 1);
    std::vector<double> childForward(periods);
    std::vector<double> parentForward(periods);
    times[0] = datetime::yearFraction(referenceDate, dates[0], zeroDayCounter);
    discountDf[0] = discount.discount(times[0]);
    childDf[0] = child.discount(times[0]);
    parentDf[0] = parent.discount(times[0]);
    double annuity = 0.0;
    double numerator = 0.0;
    for (std::size_t k = 0; k < periods; ++k) {
        times[k + 1] = datetime::yearFraction(referenceDate, dates[k + 1], zeroDayCounter);
        taus[k] = datetime::yearFraction(dates[k], dates[k + 1], pillar.quoteDayCounter);
        discountDf[k + 1] = discount.discount(times[k + 1]);
        childDf[k + 1] = child.discount(times[k + 1]);
        parentDf[k + 1] = parent.discount(times[k + 1]);
        childForward[k] = (childDf[k] / childDf[k + 1] - 1.0) / taus[k];
        parentForward[k] = (parentDf[k] / parentDf[k + 1] - 1.0) / taus[k];
        annuity += taus[k] * discountDf[k + 1];
        numerator += taus[k] * discountDf[k + 1] * (childForward[k] - parentForward[k]);
    }
    if (!(annuity > 0.0)) {
        throw std::invalid_argument("basisSwapJacobianRowsView: non-positive annuity");
    }
    ownRow.assign(nChild - 1, 0.0);
    parentRow.assign(nParent - 1, 0.0);
    discountRow.assign(nDiscount - 1, 0.0);
    const auto weightsAt = [](const StackCurveView& curve, double time, std::vector<double>& out) {
        // D(0) = 1 is node-independent; sensitivities at t = 0 are zero.
        if (time <= 0.0) {
            out.assign(curve.size(), 0.0);
            return;
        }
        curve.zeroNodeWeights(time, out);
    };
    std::vector<double> weightsPrevious;
    std::vector<double> weightsCurrent;
    for (std::size_t k = 0; k < periods; ++k) {
        weightsAt(child, times[k], weightsPrevious);
        weightsAt(child, times[k + 1], weightsCurrent);
        const double childFactor = (1.0 + taus[k] * childForward[k]) / taus[k];
        const double weight = taus[k] * discountDf[k + 1] / annuity;
        for (std::size_t j = 1; j < nChild; ++j) {
            ownRow[j - 1] += weight * childFactor *
                             (-times[k] * weightsPrevious[j] + times[k + 1] * weightsCurrent[j]);
        }
        weightsAt(parentWeightCurve, times[k], weightsPrevious);
        weightsAt(parentWeightCurve, times[k + 1], weightsCurrent);
        const double forwardSpread = childForward[k] - parentForward[k];
        for (std::size_t j = 1; j < nParent; ++j) {
            parentRow[j - 1] += weight * forwardSpread *
                                (-times[k] * weightsPrevious[j] + times[k + 1] * weightsCurrent[j]);
        }
    }
    std::vector<double> numeratorDiscount(nDiscount - 1, 0.0);
    std::vector<double> annuityDiscount(nDiscount - 1, 0.0);
    for (std::size_t k = 0; k < periods; ++k) {
        weightsAt(discountWeightCurve, times[k + 1], weightsCurrent);
        const double forwardSpread = childForward[k] - parentForward[k];
        for (std::size_t j = 1; j < nDiscount; ++j) {
            const double dDiscount = -times[k + 1] * discountDf[k + 1] * weightsCurrent[j];
            numeratorDiscount[j - 1] += taus[k] * dDiscount * forwardSpread;
            annuityDiscount[j - 1] += taus[k] * dDiscount;
        }
    }
    const double denominator = annuity * annuity;
    for (std::size_t j = 0; j < nDiscount - 1; ++j) {
        discountRow[j] =
            (numeratorDiscount[j] * annuity - numerator * annuityDiscount[j]) / denominator;
    }
    if (!pillar.spreadOnParentLeg) {
        for (double& value : ownRow) {
            value = -value;
        }
        for (double& value : parentRow) {
            value = -value;
        }
        for (double& value : discountRow) {
            value = -value;
        }
    }
}

/// View-based IRS rows for `R = floatPv / fixedAnnuity` with the floating
/// forwards from the child spread curve over `forecastParent` and every
/// discount factor from `discount`. The optional weight sources express the
/// same partials on an ancestor curve's node grid for depth-2 chains.
inline void irsSwapJacobianRowsView(const StackCurveView& child,
                                    const StackCurveView& forecastParent,
                                    const StackCurveView& discount, const IrsPillar& pillar,
                                    const datetime::Date& referenceDate, std::vector<double>& fRow,
                                    std::vector<double>& parentRow,
                                    std::vector<double>& discountRow,
                                    const StackCurveView* parentWeights = nullptr,
                                    const StackCurveView* discountWeights = nullptr) {
    const datetime::Date effective = pillar.start.serial() != 0 ? pillar.start : referenceDate;
    const datetime::Schedule floatSchedule(effective, pillar.maturity, pillar.floatTenor,
                                           pillar.floatCalendar, pillar.businessDayConvention,
                                           datetime::DateGeneration::Forward, false,
                                           datetime::BusinessDayConvention::Unadjusted);
    const datetime::Schedule fixedSchedule(effective, pillar.maturity, pillar.fixedTenor,
                                           pillar.fixedCalendar, pillar.businessDayConvention,
                                           datetime::DateGeneration::Forward, false,
                                           datetime::BusinessDayConvention::Unadjusted);
    const std::vector<datetime::Date>& floatDates = floatSchedule.dates();
    const std::vector<datetime::Date>& fixedDates = fixedSchedule.dates();
    const std::size_t nChild = child.size();
    const StackCurveView& parentWeightCurve =
        parentWeights != nullptr ? *parentWeights : forecastParent;
    const StackCurveView& discountWeightCurve =
        discountWeights != nullptr ? *discountWeights : discount;
    const std::size_t nParent = parentWeightCurve.size();
    const std::size_t nDiscount = discountWeightCurve.size();
    const datetime::DayCounter& zeroDayCounter = discount.zeroDayCounter();
    fRow.assign(nChild - 1, 0.0);
    parentRow.assign(nParent - 1, 0.0);
    discountRow.assign(nDiscount - 1, 0.0);
    const auto weightsAt = [](const StackCurveView& curve, double time, std::vector<double>& out) {
        if (time <= 0.0) {
            out.assign(curve.size(), 0.0);
            return;
        }
        curve.zeroNodeWeights(time, out);
    };
    std::vector<double> weights;
    std::vector<double> annuityRow(nDiscount - 1, 0.0);
    double annuity = 0.0;
    for (std::size_t j = 1; j < fixedDates.size(); ++j) {
        const double tau =
            datetime::yearFraction(fixedDates[j - 1], fixedDates[j], pillar.fixedDayCounter);
        const datetime::Date payDate = pillar.fixedCalendar.advance(
            fixedDates[j], datetime::Period(pillar.paymentLag, datetime::TimeUnit::Days),
            pillar.businessDayConvention);
        const double tPay = datetime::yearFraction(referenceDate, payDate, zeroDayCounter);
        const double discountPay = discount.discount(tPay);
        annuity += tau * discountPay;
        weightsAt(discountWeightCurve, tPay, weights);
        for (std::size_t i = 1; i < nDiscount; ++i) {
            annuityRow[i - 1] += -tau * tPay * discountPay * weights[i];
        }
    }
    if (!(annuity > 0.0)) {
        throw std::invalid_argument("irsSwapJacobianRowsView: non-positive fixed annuity");
    }
    std::vector<double> discountNumerator(nDiscount - 1, 0.0);
    std::vector<double> childPrev;
    std::vector<double> childCur;
    std::vector<double> parentPrev;
    std::vector<double> parentCur;
    std::vector<double> payWeights;
    double floatPv = 0.0;
    for (std::size_t k = 1; k < floatDates.size(); ++k) {
        const double tau =
            datetime::yearFraction(floatDates[k - 1], floatDates[k], pillar.floatDayCounter);
        const double tPrev =
            datetime::yearFraction(referenceDate, floatDates[k - 1], zeroDayCounter);
        const double tCur = datetime::yearFraction(referenceDate, floatDates[k], zeroDayCounter);
        const datetime::Date payDate = pillar.floatCalendar.advance(
            floatDates[k], datetime::Period(pillar.paymentLag, datetime::TimeUnit::Days),
            pillar.businessDayConvention);
        const double tPay = datetime::yearFraction(referenceDate, payDate, zeroDayCounter);
        const double discountPay = discount.discount(tPay);
        if (k == 1 && pillar.firstCouponFixed) {
            const double couponPv = tau * discountPay * pillar.firstCouponRate;
            floatPv += couponPv;
            weightsAt(discountWeightCurve, tPay, payWeights);
            for (std::size_t i = 1; i < nDiscount; ++i) {
                discountNumerator[i - 1] += -tPay * couponPv * payWeights[i];
            }
            continue;
        }
        const double childPrevious = child.discount(tPrev);
        const double childCurrent = child.discount(tCur);
        const double forward = (childPrevious / childCurrent - 1.0) / tau;
        floatPv += tau * discountPay * forward;
        const double ratio = childPrevious / childCurrent;
        weightsAt(child, tPrev, childPrev);
        weightsAt(child, tCur, childCur);
        weightsAt(parentWeightCurve, tPrev, parentPrev);
        weightsAt(parentWeightCurve, tCur, parentCur);
        weightsAt(discountWeightCurve, tPay, payWeights);
        for (std::size_t i = 1; i < nChild; ++i) {
            fRow[i - 1] +=
                (discountPay * ratio / annuity) * (-tPrev * childPrev[i] + tCur * childCur[i]);
        }
        for (std::size_t i = 1; i < nParent; ++i) {
            parentRow[i - 1] +=
                (discountPay * ratio / annuity) * (-tPrev * parentPrev[i] + tCur * parentCur[i]);
        }
        for (std::size_t i = 1; i < nDiscount; ++i) {
            discountNumerator[i - 1] += -tPay * tau * discountPay * forward * payWeights[i];
        }
    }
    const double denominator = annuity * annuity;
    for (std::size_t i = 1; i < nDiscount; ++i) {
        discountRow[i - 1] =
            (discountNumerator[i - 1] * annuity - floatPv * annuityRow[i - 1]) / denominator;
    }
}

/// View-based rows of a synthetic money-market forecast pillar
/// `r = (D_f(t1) / D_f(t2) - 1) / tau`: `ownRow` over the child spread nodes,
/// `parentRow` over the forecast-parent grid (an additive spread curve shifts
/// the child forwards one-for-one with its parent) and a zero discount row.
/// The optional weight sources express the parent partial on an ancestor
/// curve's node grid for depth-2 chains.
inline void
forecastSimpleJacobianRowsView(const StackCurveView& child, const StackCurveView& parent,
                               const StackCurveView& discount, const ForecastPillar& pillar,
                               const datetime::Date& referenceDate, std::vector<double>& ownRow,
                               std::vector<double>& parentRow, std::vector<double>& discountRow,
                               const StackCurveView* parentWeights = nullptr,
                               const StackCurveView* discountWeights = nullptr) {
    const StackCurveView& parentWeightCurve = parentWeights != nullptr ? *parentWeights : parent;
    const StackCurveView& discountWeightCurve =
        discountWeights != nullptr ? *discountWeights : discount;
    const datetime::Date start = pillar.start.serial() != 0 ? pillar.start : referenceDate;
    const datetime::Date maturity =
        pillar.calendar.adjust(pillar.maturity, pillar.businessDayConvention);
    const datetime::DayCounter& zeroDayCounter = child.zeroDayCounter();
    const double t1 = datetime::yearFraction(referenceDate, start, zeroDayCounter);
    const double t2 = datetime::yearFraction(referenceDate, maturity, zeroDayCounter);
    const double tau = datetime::yearFraction(start, maturity, pillar.quoteDayCounter);
    if (!(tau > 0.0)) {
        throw std::invalid_argument("forecastSimpleJacobianRowsView: non-positive accrual");
    }
    if (!(t1 >= 0.0)) {
        throw std::invalid_argument(
            "forecastSimpleJacobianRowsView: start before the reference date");
    }
    const double ratio = child.discount(t1) / child.discount(t2);
    ownRow.assign(child.size() - 1, 0.0);
    parentRow.assign(parentWeightCurve.size() - 1, 0.0);
    discountRow.assign(discountWeightCurve.size() - 1, 0.0);
    const auto weightsAt = [](const StackCurveView& curve, double time, std::vector<double>& out) {
        // D(0) = 1 is node-independent; sensitivities at t = 0 are zero.
        if (time <= 0.0) {
            out.assign(curve.size(), 0.0);
            return;
        }
        curve.zeroNodeWeights(time, out);
    };
    std::vector<double> previous;
    std::vector<double> current;
    const double factor = ratio / tau;
    weightsAt(child, t1, previous);
    weightsAt(child, t2, current);
    for (std::size_t j = 1; j < child.size(); ++j) {
        ownRow[j - 1] = factor * (-t1 * previous[j] + t2 * current[j]);
    }
    weightsAt(parentWeightCurve, t1, previous);
    weightsAt(parentWeightCurve, t2, current);
    for (std::size_t j = 1; j < parentWeightCurve.size(); ++j) {
        parentRow[j - 1] = factor * (-t1 * previous[j] + t2 * current[j]);
    }
}

/// View-based rows of a forecast-curve rate future: `ownRow` over the child
/// spread nodes, `parentRow` over the forecast-parent grid (an additive spread
/// curve shifts the child forwards one-for-one with its parent) and a zero
/// discount row. The optional weight sources express the parent partial on an
/// ancestor curve's node grid for depth-2 chains.
inline void
forecastFutureJacobianRowsView(const StackCurveView& child, const StackCurveView& parent,
                               const StackCurveView& discount, const ForecastPillar& pillar,
                               const datetime::Date& referenceDate, std::vector<double>& ownRow,
                               std::vector<double>& parentRow, std::vector<double>& discountRow,
                               const StackCurveView* parentWeights = nullptr,
                               const StackCurveView* discountWeights = nullptr) {
    const StackCurveView& parentWeightCurve = parentWeights != nullptr ? *parentWeights : parent;
    const StackCurveView& discountWeightCurve =
        discountWeights != nullptr ? *discountWeights : discount;
    forecastFutureRowsCore(child, parentWeightCurve, pillar, referenceDate, ownRow, parentRow);
    discountRow.assign(discountWeightCurve.size() - 1, 0.0);
}

/// One curve of a general stack tree: native quotes, node sensitivities and
/// (optionally) an exogenous discount curve. Inputs are listed in any order;
/// every parent or discount view, and every ancestor in their parent chains,
/// must match a curve in the list.
struct StackCurveInput {
    StackCurveView::Ptr curve;
    CurveRole role = CurveRole::Discount;
    std::vector<CurvePillar> discountPillars; ///< Exactly one of the two is set
    std::vector<ForecastPillar> forecastPillars;
    std::vector<double> dVdNodes; ///< dV/d(zeta_i); size == curve->size(), node 0 unused
    StackCurveView::Ptr discount; ///< Exogenous discounting; null uses the parent view
};

/// Assembled stack quote system over native node coordinates: per-curve block
/// offsets and the row-major instrument Jacobian `F = d r / d zeta`.
struct StackQuoteSystem {
    std::size_t dim = 0;
    std::vector<std::size_t> offsets; ///< First node column of each input curve
    std::vector<double> jacobian;     ///< Row-major dim x dim
};

/// Quote labels, maturity years, maturity-tag buckets and per-quote roles of
/// one stack input in `stackQuoteRisk` entry order. A direct turn knot keeps
/// its bootstrap column but reports under `TurnOverlay` with a `Turn <date>`
/// label.
inline void appendStackQuoteMetadata(const StackCurveInput& input,
                                     const datetime::Date& referenceDate,
                                     std::vector<std::string>& labels, std::vector<int>& years,
                                     std::vector<std::string>& buckets,
                                     std::vector<CurveRole>& roles) {
    if (!input.discountPillars.empty()) {
        for (const CurvePillar& pillar : input.discountPillars) {
            const datetime::Date maturity = pillarRiskMaturity(pillar);
            const double t =
                datetime::yearFraction(referenceDate, maturity, input.curve->zeroDayCounter());
            const std::string tag = riskMaturityTag(maturity, t);
            years.push_back(static_cast<int>(std::lround(t)));
            buckets.push_back(tag);
            labels.push_back(std::string(pillarKindName(pillar.kind)) + " " + tag);
            roles.push_back(input.role);
        }
        return;
    }
    for (const ForecastPillar& pillar : input.forecastPillars) {
        const datetime::Date maturity = forecastPillarRiskMaturity(pillar);
        const double t =
            datetime::yearFraction(referenceDate, maturity, input.curve->zeroDayCounter());
        years.push_back(static_cast<int>(std::lround(t)));
        roles.push_back(pillar.turnPillar ? CurveRole::TurnOverlay : input.role);
        if (pillar.turnPillar) {
            const datetime::Date start = pillar.start.serial() != 0 ? pillar.start : referenceDate;
            const std::string label = "Turn " + start.toIso();
            buckets.push_back(label);
            labels.push_back(label);
        } else {
            const std::string tag = riskMaturityTag(maturity, t);
            buckets.push_back(tag);
            labels.push_back(std::string(forecastPillarKindName(pillar.kind)) + " " + tag);
        }
    }
}

/// Assemble the stack quote Jacobian `F = d r / d zeta` (row-major `dim x dim`)
/// and the per-curve node block offsets. Every instrument row is assembled
/// analytically over the view-native node coordinates.
inline StackQuoteSystem assembleStackQuoteSystem(const std::vector<StackCurveInput>& curves,
                                                 const datetime::Date& referenceDate) {
    if (curves.empty()) {
        throw std::invalid_argument("assembleStackQuoteSystem: no curves");
    }
    StackQuoteSystem system;
    system.offsets.assign(curves.size(), 0);
    for (std::size_t k = 0; k < curves.size(); ++k) {
        const StackCurveInput& input = curves[k];
        if (input.curve == nullptr || input.dVdNodes.size() != input.curve->size()) {
            throw std::invalid_argument("assembleStackQuoteSystem: malformed curve input");
        }
        const bool hasDiscount = !input.discountPillars.empty();
        const bool hasForecast = !input.forecastPillars.empty();
        if (hasDiscount == hasForecast) {
            throw std::invalid_argument(
                "assembleStackQuoteSystem: exactly one pillar set must be non-empty");
        }
        const std::size_t pillars =
            hasDiscount ? input.discountPillars.size() : input.forecastPillars.size();
        if (pillars + 1 != input.curve->size()) {
            throw std::invalid_argument(
                "assembleStackQuoteSystem: pillars must match the curve nodes");
        }
        if (hasForecast && input.curve->parentView() == nullptr) {
            throw std::invalid_argument(
                "assembleStackQuoteSystem: forecast curve without a parent");
        }
        system.offsets[k] = system.dim;
        system.dim += input.curve->size() - 1;
    }
    const std::size_t dim = system.dim;
    // Curve matching walks every stack input and fingerprints interpolation
    // weights, so memoize the result per view identity: each ancestor of each
    // instrument row maps to the same block for the whole assembly.
    std::unordered_map<const void*, std::size_t> matchedBlocks;
    const auto matchCurve = [&](const StackCurveView& view) -> std::size_t {
        const auto cached = matchedBlocks.find(view.identity());
        if (cached != matchedBlocks.end()) {
            return cached->second;
        }
        for (std::size_t m = 0; m < curves.size(); ++m) {
            if (sameCurveView(*curves[m].curve, view)) {
                matchedBlocks.emplace(view.identity(), m);
                return m;
            }
        }
        throw std::invalid_argument("assembleStackQuoteSystem: curve is not part of the stack");
    };
    system.jacobian.assign(dim * dim, 0.0);
    std::vector<double> scratch;
    std::size_t row = 0;
    for (std::size_t k = 0; k < curves.size(); ++k) {
        const StackCurveInput& input = curves[k];
        if (!input.discountPillars.empty()) {
            for (const CurvePillar& pillar : input.discountPillars) {
                if (!pillarJacobianRow(pillar, referenceDate, *input.curve, scratch)) {
                    throw std::invalid_argument("assembleStackQuoteSystem: degenerate pillar");
                }
                for (std::size_t i = 0; i < scratch.size(); ++i) {
                    system.jacobian[row * dim + system.offsets[k] + i] = scratch[i];
                }
                ++row;
            }
            continue;
        }
        const StackCurveView& parent = *input.curve->parentView();
        const StackCurveView& discount = input.discount ? *input.discount : parent;
        const auto assembleRows = [&](const ForecastPillar& pillar,
                                      const StackCurveView* parentWeights,
                                      const StackCurveView* discountWeights,
                                      std::vector<double>& ownRow, std::vector<double>& parentRow,
                                      std::vector<double>& discountRow) {
            switch (pillar.kind) {
                case ForecastPillar::Kind::Irs:
                    irsSwapJacobianRowsView(*input.curve, parent, discount, pillar.irs,
                                            referenceDate, ownRow, parentRow, discountRow,
                                            parentWeights, discountWeights);
                    return;
                case ForecastPillar::Kind::Deposit:
                case ForecastPillar::Kind::Fra:
                    forecastSimpleJacobianRowsView(*input.curve, parent, discount, pillar,
                                                   referenceDate, ownRow, parentRow, discountRow,
                                                   parentWeights, discountWeights);
                    return;
                case ForecastPillar::Kind::Future:
                    forecastFutureJacobianRowsView(*input.curve, parent, discount, pillar,
                                                   referenceDate, ownRow, parentRow, discountRow,
                                                   parentWeights, discountWeights);
                    return;
                case ForecastPillar::Kind::BasisSwap:
                    basisSwapJacobianRowsView(*input.curve, parent, discount, pillar.basis,
                                              referenceDate, ownRow, parentRow, discountRow,
                                              parentWeights, discountWeights);
                    return;
            }
            throw std::invalid_argument("assembleStackQuoteSystem: unknown forecast pillar kind");
        };
        for (const ForecastPillar& pillar : input.forecastPillars) {
            std::vector<double> ownRow;
            std::vector<double> parentRow;
            std::vector<double> discountRow;
            assembleRows(pillar, nullptr, nullptr, ownRow, parentRow, discountRow);
            for (std::size_t i = 0; i < ownRow.size(); ++i) {
                system.jacobian[row * dim + system.offsets[k] + i] = ownRow[i];
            }
            // The native pass already assembled the forecast-parent row on the
            // parent grid and the discount row on the discount grid; the first
            // ancestor of each chain is exactly that view, so reuse those rows
            // instead of assembling them a second time.
            bool reuseParentRow = true;
            // A depth-2 grandchild moves with every curve on its parent chain,
            // so the forecast-parent partial is expressed on each ancestor's
            // node grid as well.
            for (const StackCurveView* ancestor = &parent; ancestor != nullptr;
                 ancestor = ancestor->parentView()) {
                const std::size_t block = matchCurve(*ancestor);
                if (!reuseParentRow) {
                    assembleRows(pillar, ancestor, nullptr, ownRow, parentRow, discountRow);
                }
                for (std::size_t i = 0; i < parentRow.size(); ++i) {
                    system.jacobian[row * dim + system.offsets[block] + i] += parentRow[i];
                }
                reuseParentRow = false;
            }
            // Likewise for the discount curve's own parent chain.
            bool reuseDiscountRow = true;
            for (const StackCurveView* ancestor = &discount; ancestor != nullptr;
                 ancestor = ancestor->parentView()) {
                const std::size_t block = matchCurve(*ancestor);
                if (!reuseDiscountRow) {
                    assembleRows(pillar, nullptr, ancestor, ownRow, parentRow, discountRow);
                }
                for (std::size_t i = 0; i < discountRow.size(); ++i) {
                    system.jacobian[row * dim + system.offsets[block] + i] += discountRow[i];
                }
                reuseDiscountRow = false;
            }
            ++row;
        }
    }
    return system;
}

/// Rebuild every curve view after bumping one native node of the stack. The
/// bumped input receives the node bump; every other input whose parent
/// identity chain reaches a bumped view is rebuilt with the bumped parent, and
/// discount views are re-pointed at the bumped curve they reference.
inline std::vector<StackCurveInput> bumpStackInputs(const std::vector<StackCurveInput>& curves,
                                                    std::size_t bumpedCurve, std::size_t node,
                                                    double delta) {
    std::vector<StackCurveInput> result = curves;
    std::vector<StackCurveView::Ptr> views(curves.size());
    for (std::size_t k = 0; k < curves.size(); ++k) {
        views[k] = curves[k].curve;
    }
    std::unordered_map<const void*, StackCurveView::Ptr> rebuilt;
    views[bumpedCurve] = curves[bumpedCurve].curve->rebuildWithNode(node, delta, nullptr);
    rebuilt[curves[bumpedCurve].curve->identity()] = views[bumpedCurve];
    bool changed = true;
    while (changed) {
        changed = false;
        for (std::size_t k = 0; k < curves.size(); ++k) {
            if (k == bumpedCurve || views[k] != curves[k].curve) {
                continue;
            }
            const StackCurveView* parent = curves[k].curve->parentView();
            if (parent == nullptr) {
                continue;
            }
            const auto found = rebuilt.find(parent->identity());
            if (found != rebuilt.end()) {
                views[k] = curves[k].curve->rebuildWithNode(0, 0.0, found->second);
                rebuilt[curves[k].curve->identity()] = views[k];
                changed = true;
            }
        }
    }
    for (std::size_t k = 0; k < curves.size(); ++k) {
        result[k].curve = views[k];
        if (curves[k].discount == nullptr) {
            continue;
        }
        const auto found = rebuilt.find(curves[k].discount->identity());
        if (found != rebuilt.end()) {
            result[k].discount = found->second;
            continue;
        }
        for (const StackCurveView* ancestor = curves[k].discount->parentView(); ancestor != nullptr;
             ancestor = ancestor->parentView()) {
            const auto ancestorFound = rebuilt.find(ancestor->identity());
            if (ancestorFound != rebuilt.end()) {
                result[k].discount =
                    curves[k].discount->rebuildWithNode(0, 0.0, ancestorFound->second);
                break;
            }
        }
    }
    return result;
}

/// Total quote risk over an arbitrary curve tree. Every instrument row is
/// assembled analytically over the view-native node coordinates, and the full
/// system `F^T x = g` is solved densely, so curve depth and exogenous
/// discounting only change which column block a row term lands in.
inline std::vector<StackRiskEntry> stackQuoteRisk(const std::vector<StackCurveInput>& curves,
                                                  const datetime::Date& referenceDate) {
    const StackQuoteSystem system = assembleStackQuoteSystem(curves, referenceDate);
    const std::size_t dim = system.dim;
    const std::vector<std::size_t>& offset = system.offsets;
    const std::vector<double>& f = system.jacobian;
    std::vector<double> fTranspose(dim * dim);
    for (std::size_t i = 0; i < dim; ++i) {
        for (std::size_t j = 0; j < dim; ++j) {
            fTranspose[j * dim + i] = f[i * dim + j];
        }
    }
    std::vector<double> g(dim);
    for (std::size_t k = 0; k < curves.size(); ++k) {
        for (std::size_t i = 1; i < curves[k].curve->size(); ++i) {
            g[offset[k] + i - 1] = curves[k].dVdNodes[i];
        }
    }
    const std::vector<double> x = quantape::math::solveDense(std::move(fTranspose), dim, g);
    std::vector<StackRiskEntry> result;
    result.reserve(curves.size());
    for (std::size_t k = 0; k < curves.size(); ++k) {
        const StackCurveInput& input = curves[k];
        StackRiskEntry entry;
        entry.role = input.role;
        entry.quoteDeltas.assign(input.curve->size() - 1, 0.0);
        for (std::size_t i = 0; i < entry.quoteDeltas.size(); ++i) {
            entry.quoteDeltas[i] = x[offset[k] + i];
        }
        appendStackQuoteMetadata(input, referenceDate, entry.quoteLabels, entry.quoteYears,
                                 entry.quoteBuckets, entry.quoteRoles);
        result.push_back(std::move(entry));
    }
    return result;
}

/// Depth-1 convenience overload: the root plus additive spread children whose
/// parents must be the root curve. Implemented over the general engine.
inline std::vector<StackRiskEntry> stackQuoteRisk(const DiscountCurve<double>& root,
                                                  const std::vector<CurvePillar>& rootPillars,
                                                  const std::vector<double>& dVdRoot,
                                                  const std::vector<StackChildInput>& children,
                                                  const datetime::Date& referenceDate) {
    const StackCurveView::Ptr rootView = StackCurveView::make(root);
    std::vector<StackCurveInput> inputs;
    inputs.reserve(children.size() + 1);
    StackCurveInput rootInput;
    rootInput.curve = rootView;
    rootInput.role = CurveRole::Discount;
    rootInput.discountPillars = rootPillars;
    rootInput.dVdNodes = dVdRoot;
    inputs.push_back(std::move(rootInput));
    for (const StackChildInput& child : children) {
        if (child.curve == nullptr) {
            throw std::invalid_argument("stackQuoteRisk: malformed child input");
        }
        StackCurveInput input;
        input.curve = StackCurveView::make(*child.curve);
        input.role = child.role;
        input.forecastPillars = child.pillars;
        input.dVdNodes = child.dVdSpread;
        if (child.discountCurve != nullptr) {
            input.discount = sameCurveValues(*child.discountCurve, root)
                                 ? rootView
                                 : StackCurveView::make(*child.discountCurve);
        }
        inputs.push_back(std::move(input));
    }
    return stackQuoteRisk(inputs, referenceDate);
}

/**
 * @brief Exact quote-space stack gamma.
 *
 * `HZeta` is the `dim x dim` row-major portfolio Hessian
 * `d^2 V / dzeta_i dzeta_j` in stack-native node coordinates: per-curve solved
 * nodes (node 0 dropped), curves in input order, the same ordering as the
 * delta columns. With `F = d r / d zeta` assembled exactly as in
 * `stackQuoteRisk`, `J = F^{-1}` the dense inverse, `g` the concatenated node
 * gradient and `x = J^T g` the quote-space delta, the exact transform is
 *
 *   `H_r = J^T HZeta J - sum_r x_r (J^T G_r J)`,
 *
 * where `G_r = dF_r/dzeta` is the `dim x dim` Jacobian of quote row `r` with
 * respect to the native nodes, indexed `[column][bump direction]`. The second
 * term is the bootstrap-curvature correction; the first term alone is the
 * Gauss-Newton approximation. `G` is built by central differences with step
 * `1e-6` on the instrument Jacobian, bumping one native node and rebuilding
 * every curve that depends on it through its parent or discount chain. The
 * result is symmetrized to absorb dense-product round-off.
 */
inline StackQuoteGamma stackQuoteGamma(const std::vector<StackCurveInput>& curves,
                                       const std::vector<double>& HZeta,
                                       const datetime::Date& referenceDate) {
    const StackQuoteSystem system = assembleStackQuoteSystem(curves, referenceDate);
    const std::size_t dim = system.dim;
    if (HZeta.size() != dim * dim) {
        throw std::invalid_argument("stackQuoteGamma: HZeta size mismatch");
    }
    const std::vector<std::size_t>& offset = system.offsets;
    const std::vector<double>& f = system.jacobian;
    // J = F^{-1} by one partial-pivoted LU factorization plus dim
    // back-substitutions, instead of dim independent dense solves.
    const detail::DenseLu factors = detail::factorDenseLu(f, dim);
    std::vector<double> jacobian(dim * dim, 0.0);
    std::vector<double> unit(dim, 0.0);
    for (std::size_t column = 0; column < dim; ++column) {
        std::fill(unit.begin(), unit.end(), 0.0);
        unit[column] = 1.0;
        const std::vector<double> solution = detail::solveDenseLu(factors, unit);
        for (std::size_t k = 0; k < dim; ++k) {
            jacobian[k * dim + column] = solution[k];
        }
    }
    // Node gradient and quote-space delta x = J^T g.
    std::vector<double> g(dim, 0.0);
    for (std::size_t k = 0; k < curves.size(); ++k) {
        for (std::size_t i = 1; i < curves[k].curve->size(); ++i) {
            g[offset[k] + i - 1] = curves[k].dVdNodes[i];
        }
    }
    std::vector<double> x(dim, 0.0);
    for (std::size_t a = 0; a < dim; ++a) {
        double sum = 0.0;
        for (std::size_t i = 0; i < dim; ++i) {
            sum += jacobian[i * dim + a] * g[i];
        }
        x[a] = sum;
    }
    // weighted[c][k] = sum_r x_r dF[r][c] / dzeta_k, accumulated bump by bump.
    const double step = 1e-6;
    std::vector<double> weighted(dim * dim, 0.0);
    for (std::size_t j = 0; j < curves.size(); ++j) {
        for (std::size_t i = 1; i < curves[j].curve->size(); ++i) {
            const std::size_t bump = offset[j] + i - 1;
            const StackQuoteSystem plusSystem =
                assembleStackQuoteSystem(bumpStackInputs(curves, j, i, step), referenceDate);
            const StackQuoteSystem minusSystem =
                assembleStackQuoteSystem(bumpStackInputs(curves, j, i, -step), referenceDate);
            for (std::size_t column = 0; column < dim; ++column) {
                for (std::size_t r = 0; r < dim; ++r) {
                    const double curvature = (plusSystem.jacobian[r * dim + column] -
                                              minusSystem.jacobian[r * dim + column]) /
                                             (2.0 * step);
                    weighted[column * dim + bump] += x[r] * curvature;
                }
            }
        }
    }
    // H_r = J^T (HZeta - M) J.
    std::vector<double> combined = HZeta;
    for (std::size_t k = 0; k < combined.size(); ++k) {
        combined[k] -= weighted[k];
    }
    std::vector<double> tmp(dim * dim, 0.0);
    for (std::size_t i = 0; i < dim; ++i) {
        for (std::size_t b = 0; b < dim; ++b) {
            double sum = 0.0;
            for (std::size_t a = 0; a < dim; ++a) {
                sum += combined[i * dim + a] * jacobian[a * dim + b];
            }
            tmp[i * dim + b] = sum;
        }
    }
    StackQuoteGamma result;
    result.dim = dim;
    result.hessian.assign(dim * dim, 0.0);
    for (std::size_t i = 0; i < dim; ++i) {
        for (std::size_t b = 0; b < dim; ++b) {
            double sum = 0.0;
            for (std::size_t a = 0; a < dim; ++a) {
                sum += jacobian[a * dim + i] * tmp[a * dim + b];
            }
            result.hessian[i * dim + b] = sum;
        }
    }
    // Symmetrize: round-off in the dense products can leave an asymmetric
    // residual even though the exact transform is symmetric.
    for (std::size_t i = 0; i < dim; ++i) {
        for (std::size_t j = i + 1; j < dim; ++j) {
            const double symmetric =
                0.5 * (result.hessian[i * dim + j] + result.hessian[j * dim + i]);
            result.hessian[i * dim + j] = symmetric;
            result.hessian[j * dim + i] = symmetric;
        }
    }
    for (const StackCurveInput& input : curves) {
        appendStackQuoteMetadata(input, referenceDate, result.quoteLabels, result.quoteYears,
                                 result.quoteBuckets, result.roles);
    }
    if (result.hessian.size() != dim * dim || result.quoteLabels.size() != dim ||
        result.quoteYears.size() != dim || result.quoteBuckets.size() != dim ||
        result.roles.size() != dim) {
        throw std::invalid_argument("stackQuoteGamma: metadata size mismatch");
    }
    return result;
}

/// Depth-1 convenience overload: the root plus additive spread children whose
/// parents must be the root curve. Builds the depth-1 input list like the
/// legacy `stackQuoteRisk` and delegates to the general engine.
inline StackQuoteGamma
stackQuoteGamma(const DiscountCurve<double>& root, const std::vector<CurvePillar>& rootPillars,
                const std::vector<double>& dVdRoot, const std::vector<double>& HZeta,
                const datetime::Date& referenceDate, const std::vector<StackChildInput>& children) {
    const StackCurveView::Ptr rootView = StackCurveView::make(root);
    std::vector<StackCurveInput> inputs;
    inputs.reserve(children.size() + 1);
    StackCurveInput rootInput;
    rootInput.curve = rootView;
    rootInput.role = CurveRole::Discount;
    rootInput.discountPillars = rootPillars;
    rootInput.dVdNodes = dVdRoot;
    inputs.push_back(std::move(rootInput));
    for (const StackChildInput& child : children) {
        if (child.curve == nullptr) {
            throw std::invalid_argument("stackQuoteGamma: malformed child input");
        }
        StackCurveInput input;
        input.curve = StackCurveView::make(*child.curve);
        input.role = child.role;
        input.forecastPillars = child.pillars;
        input.dVdNodes = child.dVdSpread;
        if (child.discountCurve != nullptr) {
            input.discount = sameCurveValues(*child.discountCurve, root)
                                 ? rootView
                                 : StackCurveView::make(*child.discountCurve);
        }
        inputs.push_back(std::move(input));
    }
    return stackQuoteGamma(inputs, HZeta, referenceDate);
}

/// Maturity-tag ladder across all curves of the stack (sums quote deltas by
/// maturity tag for every entry; entries without tags fall back to the rounded
/// year).
inline std::vector<RiskBucket> stackYearLadder(const std::vector<StackRiskEntry>& entries) {
    std::vector<RiskBucket> buckets;
    for (const StackRiskEntry& entry : entries) {
        if (entry.quoteYears.size() != entry.quoteDeltas.size()) {
            throw std::invalid_argument("stackYearLadder: quote years and deltas size mismatch");
        }
        if (!entry.quoteBuckets.empty() && entry.quoteBuckets.size() != entry.quoteDeltas.size()) {
            throw std::invalid_argument("stackYearLadder: quote buckets and deltas size mismatch");
        }
        for (std::size_t j = 0; j < entry.quoteDeltas.size(); ++j) {
            const std::string label = entry.quoteBuckets.empty()
                                          ? std::to_string(entry.quoteYears[j]) + "Y"
                                          : entry.quoteBuckets[j];
            bool merged = false;
            for (RiskBucket& bucket : buckets) {
                if (bucket.label == label) {
                    bucket.delta += entry.quoteDeltas[j];
                    merged = true;
                    break;
                }
            }
            if (!merged) {
                buckets.push_back(RiskBucket{label, entry.quoteDeltas[j]});
            }
        }
    }
    return buckets;
}

/// Number of solved nodes of a forecast provider (`SpreadCurve` exposes its
/// spread nodes, `DiscountCurve` its zero nodes).
inline std::size_t forecastNodeCount(const XccyForecastCurve auto& forecast) {
    if constexpr (requires { forecast.spreadNodes(); }) {
        return forecast.spreadNodes().size() - 1;
    } else {
        return forecast.size() - 1;
    }
}

/// Node times (without the fixed node 0) of a forecast provider.
inline std::vector<double> forecastNodeTimes(const XccyForecastCurve auto& forecast) {
    std::vector<double> times;
    if constexpr (requires { forecast.spreadNodes(); }) {
        times.assign(forecast.spreadNodes().times().begin() + 1,
                     forecast.spreadNodes().times().end());
    } else {
        times.assign(forecast.times().begin() + 1, forecast.times().end());
    }
    return times;
}

/// Zero-clock day counter of a forecast provider (spread nodes share the
/// parent's clock).
inline const datetime::DayCounter& forecastNodeDayCounter(const XccyForecastCurve auto& forecast) {
    if constexpr (requires { forecast.zeroDayCounter(); }) {
        return forecast.zeroDayCounter();
    } else {
        return forecast.spreadNodes().zeroDayCounter();
    }
}

/// Calendar date of a native forecast node time. The curves store only times on
/// their zero clock, so the integer day denominator of the common Actual
/// conventions is inverted exactly; for other conventions the round-tripped
/// ACT/365F offset is used, which only affects tag rounding below one year.
inline datetime::Date forecastNodeDate(const datetime::DayCounter& zeroDayCounter,
                                       const datetime::Date& referenceDate, double t) {
    double daysPerYear = 365.0;
    switch (zeroDayCounter.convention()) {
        case datetime::DayCount::Actual360:
            daysPerYear = 360.0;
            break;
        case datetime::DayCount::Actual364:
            daysPerYear = 364.0;
            break;
        case datetime::DayCount::Actual365Fixed:
            daysPerYear = 365.0;
            break;
        case datetime::DayCount::Actual366:
            daysPerYear = 366.0;
            break;
        default:
            break;
    }
    return referenceDate.plusDays(static_cast<std::int32_t>(std::lround(t * daysPerYear)));
}

/// Forecast rebuilt with one solved node bumped (`SpreadCurve`: spread node,
/// `DiscountCurve`: zero node).
inline auto forecastWithNode(const XccyForecastCurve auto& forecast, std::size_t node,
                             double delta) {
    if constexpr (requires { forecast.spreadNodes(); }) {
        const DiscountCurve<double>& nodes = forecast.spreadNodes();
        std::vector<double> spreads = nodes.zeros();
        spreads[node] += delta;
        return SpreadCurve<double>(forecast.parentPointer(), nodes.times(), spreads, nodes.scheme(),
                                   nodes.tension());
    } else {
        std::vector<double> zeros = forecast.zeros();
        zeros[node] += delta;
        return DiscountCurve<double>(forecast.times(), zeros, forecast.space(), forecast.scheme(),
                                     forecast.tension(), forecast.switchIndex());
    }
}

/// Full cross-currency risk rows over every curve the par condition touches:
/// `fRow` foreign discount nodes, `cRow` root nodes (including the domestic
/// forecast parent chain when the forecast is a spread curve over the root),
/// `gRow` foreign forecast nodes and `hRow` domestic forecast nodes, all by
/// central differences on rebuilt curves.
inline void xccySwapJacobianRows(const DiscountCurve<double>& foreignDiscount,
                                 const XccyForecastCurve auto& foreignForecast,
                                 const DiscountCurve<double>& domesticDiscount,
                                 const XccyForecastCurve auto& domesticForecast,
                                 const XccyPillar& pillar, const datetime::Date& referenceDate,
                                 const datetime::DayCounter& zeroDayCounter,
                                 std::vector<double>& fRow, std::vector<double>& cRow,
                                 std::vector<double>& gRow, std::vector<double>& hRow) {
    const double step = 1e-6;
    const auto quoteAt = [&](const auto& foreignForecastRef, const auto& domesticForecastRef,
                             const DiscountCurve<double>& foreignCurve,
                             const DiscountCurve<double>& domesticCurve) {
        return impliedXccyBasisSpread(foreignCurve, foreignForecastRef, domesticCurve,
                                      domesticForecastRef, pillar, referenceDate, zeroDayCounter);
    };
    const auto bumpCurve = [](const DiscountCurve<double>& curve, std::size_t node, double delta) {
        std::vector<double> zeros = curve.zeros();
        zeros[node] += delta;
        return DiscountCurve<double>(curve.times(), zeros, curve.space(), curve.scheme(),
                                     curve.tension(), curve.switchIndex());
    };
    // A spread-curve domestic forecast is parented on the root: root bumps move it too.
    const auto domesticForecastAtRoot = [&](const DiscountCurve<double>& rootCurve) {
        if constexpr (requires {
                          domesticForecast.spreadNodes();
                          domesticForecast.parent();
                      }) {
            const DiscountCurve<double>& nodes = domesticForecast.spreadNodes();
            return SpreadCurve<double>(std::make_shared<DiscountCurve<double>>(rootCurve),
                                       nodes.times(), nodes.zeros(), nodes.scheme(),
                                       nodes.tension());
        } else {
            return domesticForecast;
        }
    };
    // A spread-curve foreign forecast parented on the foreign discount curve
    // moves together with foreign bumps; identity of the parent curve object
    // distinguishes it from an independent base curve on the same grid.
    const auto foreignForecastAtDiscount = [&](const DiscountCurve<double>& foreignCurve) {
        if constexpr (requires {
                          foreignForecast.spreadNodes();
                          foreignForecast.parentPointer();
                      }) {
            const std::shared_ptr<const DiscountCurve<double>>& parent =
                foreignForecast.parentPointer();
            if (parent != nullptr && parent.get() == &foreignDiscount) {
                const DiscountCurve<double>& nodes = foreignForecast.spreadNodes();
                return SpreadCurve<double>(std::make_shared<DiscountCurve<double>>(foreignCurve),
                                           nodes.times(), nodes.zeros(), nodes.scheme(),
                                           nodes.tension());
            }
            return foreignForecast;
        } else {
            return foreignForecast;
        }
    };
    const std::size_t foreignNodes = foreignDiscount.size() - 1;
    fRow.assign(foreignNodes, 0.0);
    for (std::size_t i = 1; i <= foreignNodes; ++i) {
        const DiscountCurve<double> plus = bumpCurve(foreignDiscount, i, step);
        const DiscountCurve<double> minus = bumpCurve(foreignDiscount, i, -step);
        fRow[i - 1] =
            (quoteAt(foreignForecastAtDiscount(plus), domesticForecast, plus, domesticDiscount) -
             quoteAt(foreignForecastAtDiscount(minus), domesticForecast, minus, domesticDiscount)) /
            (2.0 * step);
    }
    const std::size_t domesticNodes = domesticDiscount.size() - 1;
    cRow.assign(domesticNodes, 0.0);
    for (std::size_t i = 1; i <= domesticNodes; ++i) {
        const DiscountCurve<double> plus = bumpCurve(domesticDiscount, i, step);
        const DiscountCurve<double> minus = bumpCurve(domesticDiscount, i, -step);
        cRow[i - 1] =
            (quoteAt(foreignForecast, domesticForecastAtRoot(plus), foreignDiscount, plus) -
             quoteAt(foreignForecast, domesticForecastAtRoot(minus), foreignDiscount, minus)) /
            (2.0 * step);
    }
    const std::size_t foreignForecastNodes = forecastNodeCount(foreignForecast);
    gRow.assign(foreignForecastNodes, 0.0);
    for (std::size_t i = 1; i <= foreignForecastNodes; ++i) {
        gRow[i - 1] = (quoteAt(forecastWithNode(foreignForecast, i, step), domesticForecast,
                               foreignDiscount, domesticDiscount) -
                       quoteAt(forecastWithNode(foreignForecast, i, -step), domesticForecast,
                               foreignDiscount, domesticDiscount)) /
                      (2.0 * step);
    }
    const std::size_t domesticForecastNodes = forecastNodeCount(domesticForecast);
    hRow.assign(domesticForecastNodes, 0.0);
    for (std::size_t i = 1; i <= domesticForecastNodes; ++i) {
        hRow[i - 1] = (quoteAt(foreignForecast, forecastWithNode(domesticForecast, i, step),
                               foreignDiscount, domesticDiscount) -
                       quoteAt(foreignForecast, forecastWithNode(domesticForecast, i, -step),
                               foreignDiscount, domesticDiscount)) /
                      (2.0 * step);
    }
}

/// Assemble F (xccy quote × foreign nodes), C (× root nodes), G (× foreign
/// forecast nodes) and H (× domestic forecast nodes).
inline void assembleXccyJacobianFull(const DiscountCurve<double>& foreignDiscount,
                                     const XccyForecastCurve auto& foreignForecast,
                                     const DiscountCurve<double>& domesticDiscount,
                                     const XccyForecastCurve auto& domesticForecast,
                                     const std::vector<XccyPillar>& pillars,
                                     const datetime::Date& referenceDate, std::vector<double>& f,
                                     std::vector<double>& c, std::vector<double>& g,
                                     std::vector<double>& h) {
    const std::size_t n = pillars.size();
    if (foreignDiscount.size() != n + 1) {
        throw std::invalid_argument(
            "assembleXccyJacobianFull: foreign discount nodes must match the pillar count");
    }
    const std::size_t m = domesticDiscount.size() - 1;
    const std::size_t p = forecastNodeCount(foreignForecast);
    const std::size_t q = forecastNodeCount(domesticForecast);
    f.assign(n * n, 0.0);
    c.assign(n * m, 0.0);
    g.assign(n * p, 0.0);
    h.assign(n * q, 0.0);
    std::vector<double> fRow;
    std::vector<double> cRow;
    std::vector<double> gRow;
    std::vector<double> hRow;
    for (std::size_t j = 0; j < n; ++j) {
        xccySwapJacobianRows(foreignDiscount, foreignForecast, domesticDiscount, domesticForecast,
                             pillars[j], referenceDate, domesticDiscount.zeroDayCounter(), fRow,
                             cRow, gRow, hRow);
        for (std::size_t i = 0; i < n; ++i) {
            f[j * n + i] = fRow[i];
        }
        for (std::size_t i = 0; i < m; ++i) {
            c[j * m + i] = cRow[i];
        }
        for (std::size_t i = 0; i < p; ++i) {
            g[j * p + i] = gRow[i];
        }
        for (std::size_t i = 0; i < q; ++i) {
            h[j * q + i] = hRow[i];
        }
    }
}

/// F/C-only convenience wrapper.
inline void assembleXccyJacobian(const DiscountCurve<double>& foreignDiscount,
                                 const XccyForecastCurve auto& foreignForecast,
                                 const DiscountCurve<double>& domesticDiscount,
                                 const XccyForecastCurve auto& domesticForecast,
                                 const std::vector<XccyPillar>& pillars,
                                 const datetime::Date& referenceDate, std::vector<double>& f,
                                 std::vector<double>& c) {
    std::vector<double> g;
    std::vector<double> h;
    assembleXccyJacobianFull(foreignDiscount, foreignForecast, domesticDiscount, domesticForecast,
                             pillars, referenceDate, f, c, g, h);
}

/// One cross-currency child over the domestic root: `dVdForeignZeros` is the
/// pricing risk over the foreign curve nodes (node 0 dropped).
template <XccyForecastCurve ForeignForecastT, XccyForecastCurve DomesticForecastT>
struct XccyChildInput {
    CurveRole role = CurveRole::XccyBasis; ///< Cross-currency basis child quote role
    /// Roles of the forecast-node factor rows (`XccyFwd`/`XccyDom`): the
    /// foreign and domestic forecast curves a child corrects.
    CurveRole foreignForecastRole = CurveRole::Forecast;
    CurveRole domesticForecastRole = CurveRole::Forecast;
    const DiscountCurve<double>* foreignDiscount = nullptr;
    std::vector<XccyPillar> pillars;
    std::vector<double> dVdForeignZeros;
    std::optional<ForeignForecastT> foreignForecast;
    std::optional<DomesticForecastT> domesticForecast;
    /// Optional portfolio sensitivities over the forecast nodes (node 0 = 0);
    /// the emitted entries add the residual/hedge correction on top.
    std::optional<std::vector<double>> dVdForeignForecast;
    std::optional<std::vector<double>> dVdDomesticForecast;
};

/// Stack risk with any number of (possibly heterogeneous) cross-currency
/// children over the domestic root. Each child resolves `x_c = F^{-T} g` in
/// its own foreign curve; the root right-hand side absorbs `C^T x_c` including
/// the domestic forecast parent chain, and every forecast curve receives its
/// `J^T x_c` correction as a *node-space* entry (labels `XccyFwd`/`XccyDom`),
/// so no forecast-node sensitivity is dropped.
template <typename... ChildTypes>
inline std::vector<StackRiskEntry>
stackQuoteRiskXccy(const DiscountCurve<double>& root, const std::vector<CurvePillar>& rootPillars,
                   const std::vector<double>& dVdRoot,
                   const std::vector<std::variant<ChildTypes...>>& children,
                   const datetime::Date& referenceDate) {
    if (dVdRoot.size() != root.size()) {
        throw std::invalid_argument("stackQuoteRiskXccy: root sensitivity size mismatch");
    }
    if (rootPillars.size() + 1 != root.size()) {
        throw std::invalid_argument("stackQuoteRiskXccy: root pillars must match the root nodes");
    }
    const std::size_t mRoot = rootPillars.size();
    std::vector<double> fRoot;
    std::vector<double> rowScratch;
    assembleQuoteJacobian(root, rootPillars, referenceDate, fRoot, rowScratch);
    std::vector<double> fRootTranspose(mRoot * mRoot);
    for (std::size_t j = 0; j < mRoot; ++j) {
        for (std::size_t i = 0; i < mRoot; ++i) {
            fRootTranspose[i * mRoot + j] = fRoot[j * mRoot + i];
        }
    }
    std::vector<double> rhs(mRoot);
    for (std::size_t j = 0; j < mRoot; ++j) {
        rhs[j] = dVdRoot[j + 1];
    }
    std::vector<StackRiskEntry> childEntries;
    const auto processChild = [&](const auto& child) {
        if (child.foreignDiscount == nullptr || !child.foreignForecast || !child.domesticForecast ||
            child.dVdForeignZeros.size() != child.foreignDiscount->size() ||
            child.pillars.size() + 1 != child.foreignDiscount->size()) {
            throw std::invalid_argument("stackQuoteRiskXccy: malformed child input");
        }
        const std::size_t n = child.pillars.size();
        std::vector<double> f;
        std::vector<double> cross;
        std::vector<double> fwdCross;
        std::vector<double> domCross;
        assembleXccyJacobianFull(*child.foreignDiscount, *child.foreignForecast, root,
                                 *child.domesticForecast, child.pillars, referenceDate, f, cross,
                                 fwdCross, domCross);
        const std::size_t p = forecastNodeCount(*child.foreignForecast);
        const std::size_t q = forecastNodeCount(*child.domesticForecast);
        std::vector<double> fTranspose(n * n);
        for (std::size_t j = 0; j < n; ++j) {
            for (std::size_t i = 0; i < n; ++i) {
                fTranspose[i * n + j] = f[j * n + i];
            }
        }
        std::vector<double> g(n);
        for (std::size_t i = 0; i < n; ++i) {
            g[i] = child.dVdForeignZeros[i + 1];
        }
        const std::vector<double> xChild = quantape::math::solveDense(fTranspose, n, g);
        for (std::size_t i = 0; i < n; ++i) {
            for (std::size_t j = 0; j < mRoot; ++j) {
                rhs[j] -= cross[i * mRoot + j] * xChild[i];
            }
        }
        StackRiskEntry childEntry;
        childEntry.role = child.role;
        childEntry.quoteDeltas = xChild;
        for (const XccyPillar& pillar : child.pillars) {
            const datetime::Date maturity =
                pillar.foreignCalendar.adjust(pillar.maturity, pillar.foreignBusinessDayConvention);
            const double t = datetime::yearFraction(referenceDate, maturity, root.zeroDayCounter());
            const std::string tag = riskMaturityTag(maturity, t);
            childEntry.quoteYears.push_back(static_cast<int>(std::lround(t)));
            childEntry.quoteRoles.push_back(child.role);
            childEntry.quoteBuckets.push_back(tag);
            childEntry.quoteLabels.push_back("Xccy " + tag);
        }
        childEntries.push_back(std::move(childEntry));
        const auto appendFactor = [&](const std::optional<std::vector<double>>& direct,
                                      const std::vector<double>& factorCross, std::size_t nodes,
                                      const auto& forecast, std::string_view tag, CurveRole role) {
            if (nodes == 0) {
                return;
            }
            StackRiskEntry entry;
            entry.role = role;
            entry.quoteDeltas.assign(nodes, 0.0);
            if (direct) {
                if (direct->size() != nodes + 1) {
                    throw std::invalid_argument(
                        "stackQuoteRiskXccy: forecast sensitivity size mismatch");
                }
                for (std::size_t k = 0; k < nodes; ++k) {
                    entry.quoteDeltas[k] = (*direct)[k + 1];
                }
            }
            for (std::size_t i = 0; i < n; ++i) {
                for (std::size_t k = 0; k < nodes; ++k) {
                    entry.quoteDeltas[k] -= factorCross[i * nodes + k] * xChild[i];
                }
            }
            const std::vector<double> times = forecastNodeTimes(forecast);
            const datetime::DayCounter& nodeDayCounter = forecastNodeDayCounter(forecast);
            for (std::size_t k = 0; k < nodes; ++k) {
                const datetime::Date nodeDate =
                    forecastNodeDate(nodeDayCounter, referenceDate, times[k]);
                const std::string bucket = riskMaturityTag(nodeDate, times[k]);
                entry.quoteYears.push_back(static_cast<int>(std::lround(times[k])));
                entry.quoteRoles.push_back(entry.role);
                entry.quoteBuckets.push_back(bucket);
                entry.quoteLabels.push_back(std::string(tag) + " " + bucket);
            }
            childEntries.push_back(std::move(entry));
        };
        appendFactor(child.dVdForeignForecast, fwdCross, p, *child.foreignForecast, "XccyFwd",
                     child.foreignForecastRole);
        appendFactor(child.dVdDomesticForecast, domCross, q, *child.domesticForecast, "XccyDom",
                     child.domesticForecastRole);
    };
    for (const auto& variantChild : children) {
        std::visit(processChild, variantChild);
    }
    std::vector<StackRiskEntry> result;
    StackRiskEntry rootEntry;
    rootEntry.role = CurveRole::Discount;
    rootEntry.quoteDeltas = quantape::math::solveDense(fRootTranspose, mRoot, rhs);
    for (const CurvePillar& pillar : rootPillars) {
        const datetime::Date maturity = pillarRiskMaturity(pillar);
        const double t = datetime::yearFraction(referenceDate, maturity, root.zeroDayCounter());
        const std::string tag = riskMaturityTag(maturity, t);
        rootEntry.quoteYears.push_back(static_cast<int>(std::lround(t)));
        rootEntry.quoteRoles.push_back(rootEntry.role);
        rootEntry.quoteBuckets.push_back(tag);
        rootEntry.quoteLabels.push_back(std::string(pillarKindName(pillar.kind)) + " " + tag);
    }
    result.push_back(std::move(rootEntry));
    for (StackRiskEntry& entry : childEntries) {
        result.push_back(std::move(entry));
    }
    return result;
}

} // namespace quantape::markets
