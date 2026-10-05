#pragma once

#include "quantape/markets/Curves/CurveBuilder.h"
#include "quantape/markets/Curves/CurveRisk.h"
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

/// Core IRS rows for `R = floatPv / fixedAnnuity` with the floating forwards
/// from the child spread curve over `forecastParent` and every discount factor
/// from `discount`, all year fractions measured on `zeroDayCounter`. The
/// optional weight sources express the same partials on an ancestor curve's
/// node grid for depth-2 chains.
inline void irsSwapJacobianRowsCore(const StackCurveView& child,
                                    const StackCurveView& forecastParent,
                                    const StackCurveView& discount, const IrsPillar& pillar,
                                    const datetime::Date& referenceDate,
                                    const datetime::DayCounter& zeroDayCounter,
                                    std::vector<double>& fRow, std::vector<double>& parentRow,
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
        throw std::invalid_argument("irsSwapJacobianRows: non-positive fixed annuity");
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

/// View-based IRS rows for `R = floatPv / fixedAnnuity` with the floating
/// forwards from the child spread curve over `forecastParent` and every
/// discount factor from `discount`; year fractions use the discount curve's
/// own zero clock. The optional weight sources express the same partials on an
/// ancestor curve's node grid for depth-2 chains.
inline void irsSwapJacobianRowsView(const StackCurveView& child,
                                    const StackCurveView& forecastParent,
                                    const StackCurveView& discount, const IrsPillar& pillar,
                                    const datetime::Date& referenceDate, std::vector<double>& fRow,
                                    std::vector<double>& parentRow,
                                    std::vector<double>& discountRow,
                                    const StackCurveView* parentWeights = nullptr,
                                    const StackCurveView* discountWeights = nullptr) {
    irsSwapJacobianRowsCore(child, forecastParent, discount, pillar, referenceDate,
                            discount.zeroDayCounter(), fRow, parentRow, discountRow, parentWeights,
                            discountWeights);
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

/// Concrete-curve basis adapter: wraps the child and its parent in caller
/// views and delegates to `basisSwapJacobianRowsView` with the parent as the
/// discounting curve; the returned cross row is the combined parent plus
/// discount partial.
inline void basisSwapJacobianRows(const SpreadCurve<double>& child, const BasisPillar& pillar,
                                  const datetime::Date& referenceDate, std::vector<double>& fRow,
                                  std::vector<double>& cRow) {
    const StackCurveView::Ptr childView = StackCurveView::make(child);
    const StackCurveView::Ptr parentView = StackCurveView::make(child.parent());
    std::vector<double> discountRow;
    basisSwapJacobianRowsView(*childView, *parentView, *parentView, pillar, referenceDate, fRow,
                              cRow, discountRow);
    for (std::size_t i = 0; i < cRow.size(); ++i) {
        cRow[i] += discountRow[i];
    }
}

/// Analytic IRS rows for `R = floatPv / fixedAnnuity` with the floating
/// forwards from the child spread curve over `forecastParent` and every
/// discount factor from `discount`:
/// `fRow = d R / d spread_k` (parent and discount frozen),
/// `parentRow = d R / d z_forecastParent` (spreads and discount frozen) and
/// `discountRow = d R / d z_discount` (spreads and forecast parent frozen).
/// Thin adapter over `irsSwapJacobianRowsCore`; year fractions use the
/// caller-supplied zero clock.
inline void irsSwapJacobianRows(const SpreadCurve<double>& child,
                                const DiscountCurve<double>& forecastParent,
                                const DiscountCurve<double>& discount, const IrsPillar& pillar,
                                const datetime::Date& referenceDate,
                                const datetime::DayCounter& zeroDayCounter,
                                std::vector<double>& fRow, std::vector<double>& parentRow,
                                std::vector<double>& discountRow) {
    const StackCurveView::Ptr childView = StackCurveView::make(child);
    const StackCurveView::Ptr parentView = StackCurveView::make(forecastParent);
    const StackCurveView::Ptr discountView = StackCurveView::make(discount);
    irsSwapJacobianRowsCore(*childView, *parentView, *discountView, pillar, referenceDate,
                            zeroDayCounter, fRow, parentRow, discountRow);
}

/// Analytic rows of a synthetic money-market forecast pillar
/// `r = (D_f(t1) / D_f(t2) - 1) / tau` (deposit or forward-starting FRA):
/// `ownRow = d r / d spread_k` over the child spread nodes and
/// `parentRow = d r / d z_parent,k` over the forecast parent, whose nodes
/// shift the child zero curve one-for-one. The pillar never references a
/// discount curve, so there is no separate discount sensitivity. Thin adapter
/// over `forecastSimpleJacobianRowsView`.
inline void forecastSimpleJacobianRows(const SpreadCurve<double>& child,
                                       const ForecastPillar& pillar,
                                       const datetime::Date& referenceDate,
                                       std::vector<double>& ownRow,
                                       std::vector<double>& parentRow) {
    const StackCurveView::Ptr childView = StackCurveView::make(child);
    const StackCurveView::Ptr parentView = StackCurveView::make(child.parent());
    std::vector<double> discountRow;
    forecastSimpleJacobianRowsView(*childView, *parentView, *parentView, pillar, referenceDate,
                                   ownRow, parentRow, discountRow);
}

/// Analytic rows of a forecast-curve rate future:
/// `ownRow = d r / d s_k` over the child spread nodes and
/// `parentRow = d r / d z_parent,k` over the forecast parent, whose nodes
/// shift the child zero curve one-for-one. The pillar never references a
/// discount curve, so there is no separate discount sensitivity. Thin adapter
/// over `forecastFutureJacobianRowsView`.
inline void forecastFutureJacobianRows(const SpreadCurve<double>& child,
                                       const ForecastPillar& pillar,
                                       const datetime::Date& referenceDate,
                                       std::vector<double>& ownRow,
                                       std::vector<double>& parentRow) {
    const StackCurveView::Ptr childView = StackCurveView::make(child);
    const StackCurveView::Ptr parentView = StackCurveView::make(child.parent());
    std::vector<double> discountRow;
    forecastFutureJacobianRowsView(*childView, *parentView, *parentView, pillar, referenceDate,
                                   ownRow, parentRow, discountRow);
}

/// Own-curve, forecast-parent and discount rows of one forecast pillar.
/// Basis swaps use the parent row as the cross row and report a zero discount
/// row; IRS pillars report the parent and discount rows separately. Synthetic
/// deposits, FRAs and rate futures move with the child spread nodes and the
/// forecast parent one-for-one and report a zero discount row. Thin dispatcher
/// over the view-based rows.
inline void forecastPillarJacobianRows(const SpreadCurve<double>& child,
                                       const ForecastPillar& pillar,
                                       const datetime::Date& referenceDate,
                                       const DiscountCurve<double>* discountCurve,
                                       std::vector<double>& fRow, std::vector<double>& parentRow,
                                       std::vector<double>& discountRow) {
    const DiscountCurve<double>& forecastParent = child.parent();
    const DiscountCurve<double>& discount =
        discountCurve != nullptr ? *discountCurve : forecastParent;
    const StackCurveView::Ptr childView = StackCurveView::make(child);
    const StackCurveView::Ptr parentView = StackCurveView::make(forecastParent);
    const StackCurveView::Ptr discountView = StackCurveView::make(discount);
    switch (pillar.kind) {
        case ForecastPillar::Kind::Deposit:
        case ForecastPillar::Kind::Fra:
            forecastSimpleJacobianRowsView(*childView, *parentView, *discountView, pillar,
                                           referenceDate, fRow, parentRow, discountRow);
            return;
        case ForecastPillar::Kind::Future:
            forecastFutureJacobianRowsView(*childView, *parentView, *discountView, pillar,
                                           referenceDate, fRow, parentRow, discountRow);
            return;
        case ForecastPillar::Kind::BasisSwap:
            basisSwapJacobianRowsView(*childView, *parentView, *parentView, pillar.basis,
                                      referenceDate, fRow, parentRow, discountRow);
            discountRow.assign(discount.size() - 1, 0.0);
            return;
        case ForecastPillar::Kind::Irs:
            irsSwapJacobianRowsCore(*childView, *parentView, *discountView, pillar.irs,
                                    referenceDate, child.zeroDayCounter(), fRow, parentRow,
                                    discountRow);
            return;
    }
    throw std::invalid_argument("forecastPillarJacobianRows: unknown forecast pillar kind");
}

/// Discount-curve pillar row on a view-native node grid: the single-curve
/// analytic row from `pillarJacobianRow`, applied to the type-erased view so
/// the engine assembles every instrument kind the same way. Returns false when
/// the pillar is numerically degenerate.
inline bool discountPillarJacobianRow(const CurvePillar& pillar,
                                      const datetime::Date& referenceDate,
                                      const StackCurveView& curve, std::vector<double>& row) {
    return pillarJacobianRow(pillar, referenceDate, curve, row);
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

} // namespace quantape::markets
