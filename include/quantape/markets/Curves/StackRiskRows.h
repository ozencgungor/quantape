#pragma once

#include "quantape/markets/Curves/CurveBuilder.h"
#include "quantape/markets/Curves/StackCurveView.h"

#include <cstddef>
#include <stdexcept>
#include <vector>

namespace quantape::markets {
/**
 * @file StackRiskRows.h
 * @brief Analytic instrument Jacobian rows for the stack risk engine
 *
 * One implementation per instrument kind, expressed over `StackCurveView` so
 * the same formulas serve every curve tree depth:
 *
 *  - basis swaps: `basisSwapJacobianRowsView` (own spread, forecast parent and
 *    discount rows);
 *  - par IRS: `irsSwapJacobianRowsCore` / `irsSwapJacobianRowsView` (own,
 *    forecast-parent and discount rows);
 *  - synthetic deposits and FRAs: `forecastSimpleJacobianRowsView`;
 *  - rate futures (simple, compounded, averaged): `forecastFutureRowsCore`
 *    behind `forecastFutureJacobianRowsView`;
 *  - `forecastPillarJacobianRows` dispatches one forecast pillar to its kind;
 *  - discount pillars: `discountPillarJacobianRow` over the view grid;
 *  - `assembleBasisJacobian` / `assembleForecastJacobian` assemble the child
 *    own-curve block and the cross block from those rows.
 *
 * The concrete-curve entry points (`basisSwapJacobianRows`,
 * `irsSwapJacobianRows`, `forecastSimpleJacobianRows`,
 * `forecastFutureJacobianRows`) are thin adapters that wrap the curves in
 * caller views and delegate, so there is no second formula lineage to drift.
 * Discount-curve pillar rows come from `pillarJacobianRow` (CurveRisk.h) and
 * feed the same engine assembly.
 */

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

/// View-based basis rows for an arbitrary forecast parent and discount curve:
/// `ownRow` over the child spread nodes, `parentRow` over the forecast-parent
/// nodes (both forwards move with the parent, the annuity is frozen), and
/// `discountRow` over the discount nodes (discount factors only). The optional
/// weight sources let the same curve-space partial be expressed on an ancestor
/// curve's node grid, which is how a depth-2 chain's rows reach the root block.
void basisSwapJacobianRowsView(const StackCurveView& child, const StackCurveView& parent,
                               const StackCurveView& discount, const BasisPillar& pillar,
                               const datetime::Date& referenceDate, std::vector<double>& ownRow,
                               std::vector<double>& parentRow, std::vector<double>& discountRow,
                               const StackCurveView* parentWeights = nullptr,
                               const StackCurveView* discountWeights = nullptr);

/// Core IRS rows for `R = floatPv / fixedAnnuity` with the floating forwards
/// from the child spread curve over `forecastParent` and every discount factor
/// from `discount`, all year fractions measured on `zeroDayCounter`. The
/// optional weight sources express the same partials on an ancestor curve's
/// node grid for depth-2 chains.
void irsSwapJacobianRowsCore(const StackCurveView& child, const StackCurveView& forecastParent,
                             const StackCurveView& discount, const IrsPillar& pillar,
                             const datetime::Date& referenceDate,
                             const datetime::DayCounter& zeroDayCounter, std::vector<double>& fRow,
                             std::vector<double>& parentRow, std::vector<double>& discountRow,
                             const StackCurveView* parentWeights = nullptr,
                             const StackCurveView* discountWeights = nullptr);

/// View-based IRS rows for `R = floatPv / fixedAnnuity` with the floating
/// forwards from the child spread curve over `forecastParent` and every
/// discount factor from `discount`; year fractions use the discount curve's
/// own zero clock. The optional weight sources express the same partials on an
/// ancestor curve's node grid for depth-2 chains.
void irsSwapJacobianRowsView(const StackCurveView& child, const StackCurveView& forecastParent,
                             const StackCurveView& discount, const IrsPillar& pillar,
                             const datetime::Date& referenceDate, std::vector<double>& fRow,
                             std::vector<double>& parentRow, std::vector<double>& discountRow,
                             const StackCurveView* parentWeights = nullptr,
                             const StackCurveView* discountWeights = nullptr);

/// View-based rows of a synthetic money-market forecast pillar
/// `r = (D_f(t1) / D_f(t2) - 1) / tau`: `ownRow` over the child spread nodes,
/// `parentRow` over the forecast-parent grid (an additive spread curve shifts
/// the child forwards one-for-one with its parent) and a zero discount row.
/// The optional weight sources express the parent partial on an ancestor
/// curve's node grid for depth-2 chains.
void forecastSimpleJacobianRowsView(const StackCurveView& child, const StackCurveView& parent,
                                    const StackCurveView& discount, const ForecastPillar& pillar,
                                    const datetime::Date& referenceDate,
                                    std::vector<double>& ownRow, std::vector<double>& parentRow,
                                    std::vector<double>& discountRow,
                                    const StackCurveView* parentWeights = nullptr,
                                    const StackCurveView* discountWeights = nullptr);

/// View-based rows of a forecast-curve rate future: `ownRow` over the child
/// spread nodes, `parentRow` over the forecast-parent grid (an additive spread
/// curve shifts the child forwards one-for-one with its parent) and a zero
/// discount row. The optional weight sources express the parent partial on an
/// ancestor curve's node grid for depth-2 chains.
void forecastFutureJacobianRowsView(const StackCurveView& child, const StackCurveView& parent,
                                    const StackCurveView& discount, const ForecastPillar& pillar,
                                    const datetime::Date& referenceDate,
                                    std::vector<double>& ownRow, std::vector<double>& parentRow,
                                    std::vector<double>& discountRow,
                                    const StackCurveView* parentWeights = nullptr,
                                    const StackCurveView* discountWeights = nullptr);

/// Concrete-curve basis adapter: wraps the child and its parent in caller
/// views and delegates to `basisSwapJacobianRowsView` with the parent as the
/// discounting curve; the returned cross row is the combined parent plus
/// discount partial.
void basisSwapJacobianRows(const SpreadCurve<double>& child, const BasisPillar& pillar,
                           const datetime::Date& referenceDate, std::vector<double>& fRow,
                           std::vector<double>& cRow);

/// Analytic IRS rows for `R = floatPv / fixedAnnuity` with the floating
/// forwards from the child spread curve over `forecastParent` and every
/// discount factor from `discount`:
/// `fRow = d R / d spread_k` (parent and discount frozen),
/// `parentRow = d R / d z_forecastParent` (spreads and discount frozen) and
/// `discountRow = d R / d z_discount` (spreads and forecast parent frozen).
/// Thin adapter over `irsSwapJacobianRowsCore`; year fractions use the
/// caller-supplied zero clock.
void irsSwapJacobianRows(const SpreadCurve<double>& child,
                         const DiscountCurve<double>& forecastParent,
                         const DiscountCurve<double>& discount, const IrsPillar& pillar,
                         const datetime::Date& referenceDate,
                         const datetime::DayCounter& zeroDayCounter, std::vector<double>& fRow,
                         std::vector<double>& parentRow, std::vector<double>& discountRow);

/// Analytic rows of a synthetic money-market forecast pillar
/// `r = (D_f(t1) / D_f(t2) - 1) / tau` (deposit or forward-starting FRA):
/// `ownRow = d r / d spread_k` over the child spread nodes and
/// `parentRow = d r / d z_parent,k` over the forecast parent, whose nodes
/// shift the child zero curve one-for-one. The pillar never references a
/// discount curve, so there is no separate discount sensitivity. Thin adapter
/// over `forecastSimpleJacobianRowsView`.
void forecastSimpleJacobianRows(const SpreadCurve<double>& child, const ForecastPillar& pillar,
                                const datetime::Date& referenceDate, std::vector<double>& ownRow,
                                std::vector<double>& parentRow);

/// Analytic rows of a forecast-curve rate future:
/// `ownRow = d r / d s_k` over the child spread nodes and
/// `parentRow = d r / d z_parent,k` over the forecast parent, whose nodes
/// shift the child zero curve one-for-one. The pillar never references a
/// discount curve, so there is no separate discount sensitivity. Thin adapter
/// over `forecastFutureJacobianRowsView`.
void forecastFutureJacobianRows(const SpreadCurve<double>& child, const ForecastPillar& pillar,
                                const datetime::Date& referenceDate, std::vector<double>& ownRow,
                                std::vector<double>& parentRow);

/// Own-curve, forecast-parent and discount rows of one forecast pillar.
/// Basis swaps use the parent row as the cross row and report a zero discount
/// row; IRS pillars report the parent and discount rows separately. Synthetic
/// deposits, FRAs and rate futures move with the child spread nodes and the
/// forecast parent one-for-one and report a zero discount row. Thin dispatcher
/// over the view-based rows.
void forecastPillarJacobianRows(const SpreadCurve<double>& child, const ForecastPillar& pillar,
                                const datetime::Date& referenceDate,
                                const DiscountCurve<double>* discountCurve,
                                std::vector<double>& fRow, std::vector<double>& parentRow,
                                std::vector<double>& discountRow);

/// Discount-curve pillar row on a view-native node grid: the single-curve
/// analytic row from `pillarJacobianRow`, applied to the type-erased view so
/// the engine assembles every instrument kind the same way. Returns false when
/// the pillar is numerically degenerate.
bool discountPillarJacobianRow(const CurvePillar& pillar, const datetime::Date& referenceDate,
                               const StackCurveView& curve, std::vector<double>& row);

/// Assemble the child own-curve Jacobian `F_c` (row-major `m_c x m_c`) and the
/// cross block `C` (row-major `m_c x m_p`) from basis pillars.
void assembleBasisJacobian(const SpreadCurve<double>& child,
                           const std::vector<BasisPillar>& pillars,
                           const datetime::Date& referenceDate, std::vector<double>& f,
                           std::vector<double>& c);

/// Assemble the child own-curve Jacobian `F_c` (row-major `m_c x m_c`) and the
/// cross block `C` (row-major `m_c x m_p`) over the forecast-parent nodes.
/// When an IRS child discounts on its forecast parent (the same object, or a
/// curve with the same grid and values) the cross block is the sum of the
/// parent and discount rows, because the forward and discounting curves are
/// then the same curve.
void assembleForecastJacobian(const SpreadCurve<double>& child,
                              const std::vector<ForecastPillar>& pillars,
                              const datetime::Date& referenceDate,
                              const DiscountCurve<double>* discountCurve, std::vector<double>& f,
                              std::vector<double>& c);

} // namespace quantape::markets
