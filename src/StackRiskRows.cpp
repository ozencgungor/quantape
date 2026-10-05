#include "quantape/markets/Curves/StackRiskRows.h"

#include "quantape/markets/Curves/CurveRisk.h"

#include <cstddef>
#include <stdexcept>
#include <vector>

namespace quantape::markets {
namespace {

/// Per-thread scratch for the row builders, so repeated rows reuse capacity
/// instead of re-allocating the same working vectors on every pillar.
struct BasisRowScratch {
    std::vector<double> times;
    std::vector<double> taus;
    std::vector<double> discountDf;
    std::vector<double> childDf;
    std::vector<double> parentDf;
    std::vector<double> childForward;
    std::vector<double> parentForward;
    std::vector<double> weightsPrevious;
    std::vector<double> weightsCurrent;
    std::vector<double> numeratorDiscount;
    std::vector<double> annuityDiscount;
    detail::ScheduleCache schedules;
};

BasisRowScratch& basisRowScratch() {
    static thread_local BasisRowScratch scratch;
    return scratch;
}

struct IrsRowScratch {
    std::vector<double> weights;
    std::vector<double> annuityRow;
    std::vector<double> discountNumerator;
    std::vector<double> childPrev;
    std::vector<double> childCur;
    std::vector<double> parentPrev;
    std::vector<double> parentCur;
    std::vector<double> payWeights;
    detail::ScheduleCache schedules;
};

IrsRowScratch& irsRowScratch() {
    static thread_local IrsRowScratch scratch;
    return scratch;
}

struct SimpleRowScratch {
    std::vector<double> previous;
    std::vector<double> current;
};

SimpleRowScratch& simpleRowScratch() {
    static thread_local SimpleRowScratch scratch;
    return scratch;
}

} // namespace

void basisSwapJacobianRowsView(const StackCurveView& child, const StackCurveView& parent,
                               const StackCurveView& discount, const BasisPillar& pillar,
                               const datetime::Date& referenceDate, std::vector<double>& ownRow,
                               std::vector<double>& parentRow, std::vector<double>& discountRow,
                               const StackCurveView* parentWeights,
                               const StackCurveView* discountWeights) {
    BasisRowScratch& scratch = basisRowScratch();
    const datetime::Schedule& schedule =
        scratch.schedules.get(referenceDate, pillar.maturity, pillar.floatTenor, pillar.calendar,
                              pillar.businessDayConvention);
    const std::vector<datetime::Date>& dates = schedule.dates();
    const std::size_t periods = dates.size() - 1;
    const std::size_t nChild = child.size();
    const StackCurveView& parentWeightCurve = parentWeights != nullptr ? *parentWeights : parent;
    const StackCurveView& discountWeightCurve =
        discountWeights != nullptr ? *discountWeights : discount;
    const std::size_t nParent = parentWeightCurve.size();
    const std::size_t nDiscount = discountWeightCurve.size();
    const datetime::DayCounter& zeroDayCounter = discount.zeroDayCounter();
    std::vector<double>& times = scratch.times;
    std::vector<double>& taus = scratch.taus;
    std::vector<double>& discountDf = scratch.discountDf;
    std::vector<double>& childDf = scratch.childDf;
    std::vector<double>& parentDf = scratch.parentDf;
    std::vector<double>& childForward = scratch.childForward;
    std::vector<double>& parentForward = scratch.parentForward;
    times.resize(periods + 1);
    taus.resize(periods);
    discountDf.resize(periods + 1);
    childDf.resize(periods + 1);
    parentDf.resize(periods + 1);
    childForward.resize(periods);
    parentForward.resize(periods);
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
    std::vector<double>& weightsPrevious = scratch.weightsPrevious;
    std::vector<double>& weightsCurrent = scratch.weightsCurrent;
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
    std::vector<double>& numeratorDiscount = scratch.numeratorDiscount;
    std::vector<double>& annuityDiscount = scratch.annuityDiscount;
    numeratorDiscount.assign(nDiscount - 1, 0.0);
    annuityDiscount.assign(nDiscount - 1, 0.0);
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

void irsSwapJacobianRowsCore(const StackCurveView& child, const StackCurveView& forecastParent,
                             const StackCurveView& discount, const IrsPillar& pillar,
                             const datetime::Date& referenceDate,
                             const datetime::DayCounter& zeroDayCounter, std::vector<double>& fRow,
                             std::vector<double>& parentRow, std::vector<double>& discountRow,
                             const StackCurveView* parentWeights,
                             const StackCurveView* discountWeights) {
    const datetime::Date effective = pillar.start.serial() != 0 ? pillar.start : referenceDate;
    IrsRowScratch& scratch = irsRowScratch();
    const datetime::Schedule& floatSchedule =
        scratch.schedules.get(effective, pillar.maturity, pillar.floatTenor, pillar.floatCalendar,
                              pillar.businessDayConvention);
    const datetime::Schedule& fixedSchedule =
        scratch.schedules.get(effective, pillar.maturity, pillar.fixedTenor, pillar.fixedCalendar,
                              pillar.businessDayConvention);
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
    std::vector<double>& weights = scratch.weights;
    std::vector<double>& annuityRow = scratch.annuityRow;
    annuityRow.assign(nDiscount - 1, 0.0);
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
    std::vector<double>& discountNumerator = scratch.discountNumerator;
    std::vector<double>& childPrev = scratch.childPrev;
    std::vector<double>& childCur = scratch.childCur;
    std::vector<double>& parentPrev = scratch.parentPrev;
    std::vector<double>& parentCur = scratch.parentCur;
    std::vector<double>& payWeights = scratch.payWeights;
    discountNumerator.assign(nDiscount - 1, 0.0);
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

void irsSwapJacobianRowsView(const StackCurveView& child, const StackCurveView& forecastParent,
                             const StackCurveView& discount, const IrsPillar& pillar,
                             const datetime::Date& referenceDate, std::vector<double>& fRow,
                             std::vector<double>& parentRow, std::vector<double>& discountRow,
                             const StackCurveView* parentWeights,
                             const StackCurveView* discountWeights) {
    irsSwapJacobianRowsCore(child, forecastParent, discount, pillar, referenceDate,
                            discount.zeroDayCounter(), fRow, parentRow, discountRow, parentWeights,
                            discountWeights);
}

void forecastSimpleJacobianRowsView(const StackCurveView& child, const StackCurveView& parent,
                                    const StackCurveView& discount, const ForecastPillar& pillar,
                                    const datetime::Date& referenceDate,
                                    std::vector<double>& ownRow, std::vector<double>& parentRow,
                                    std::vector<double>& discountRow,
                                    const StackCurveView* parentWeights,
                                    const StackCurveView* discountWeights) {
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
    SimpleRowScratch& scratch = simpleRowScratch();
    std::vector<double>& previous = scratch.previous;
    std::vector<double>& current = scratch.current;
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

void forecastFutureJacobianRowsView(const StackCurveView& child, const StackCurveView& parent,
                                    const StackCurveView& discount, const ForecastPillar& pillar,
                                    const datetime::Date& referenceDate,
                                    std::vector<double>& ownRow, std::vector<double>& parentRow,
                                    std::vector<double>& discountRow,
                                    const StackCurveView* parentWeights,
                                    const StackCurveView* discountWeights) {
    const StackCurveView& parentWeightCurve = parentWeights != nullptr ? *parentWeights : parent;
    const StackCurveView& discountWeightCurve =
        discountWeights != nullptr ? *discountWeights : discount;
    forecastFutureRowsCore(child, parentWeightCurve, pillar, referenceDate, ownRow, parentRow);
    discountRow.assign(discountWeightCurve.size() - 1, 0.0);
}

void basisSwapJacobianRows(const SpreadCurve<double>& child, const BasisPillar& pillar,
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

void irsSwapJacobianRows(const SpreadCurve<double>& child,
                         const DiscountCurve<double>& forecastParent,
                         const DiscountCurve<double>& discount, const IrsPillar& pillar,
                         const datetime::Date& referenceDate,
                         const datetime::DayCounter& zeroDayCounter, std::vector<double>& fRow,
                         std::vector<double>& parentRow, std::vector<double>& discountRow) {
    const StackCurveView::Ptr childView = StackCurveView::make(child);
    const StackCurveView::Ptr parentView = StackCurveView::make(forecastParent);
    const StackCurveView::Ptr discountView = StackCurveView::make(discount);
    irsSwapJacobianRowsCore(*childView, *parentView, *discountView, pillar, referenceDate,
                            zeroDayCounter, fRow, parentRow, discountRow);
}

void forecastSimpleJacobianRows(const SpreadCurve<double>& child, const ForecastPillar& pillar,
                                const datetime::Date& referenceDate, std::vector<double>& ownRow,
                                std::vector<double>& parentRow) {
    const StackCurveView::Ptr childView = StackCurveView::make(child);
    const StackCurveView::Ptr parentView = StackCurveView::make(child.parent());
    std::vector<double> discountRow;
    forecastSimpleJacobianRowsView(*childView, *parentView, *parentView, pillar, referenceDate,
                                   ownRow, parentRow, discountRow);
}

void forecastFutureJacobianRows(const SpreadCurve<double>& child, const ForecastPillar& pillar,
                                const datetime::Date& referenceDate, std::vector<double>& ownRow,
                                std::vector<double>& parentRow) {
    const StackCurveView::Ptr childView = StackCurveView::make(child);
    const StackCurveView::Ptr parentView = StackCurveView::make(child.parent());
    std::vector<double> discountRow;
    forecastFutureJacobianRowsView(*childView, *parentView, *parentView, pillar, referenceDate,
                                   ownRow, parentRow, discountRow);
}

void forecastPillarJacobianRows(const SpreadCurve<double>& child, const ForecastPillar& pillar,
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

bool discountPillarJacobianRow(const CurvePillar& pillar, const datetime::Date& referenceDate,
                               const StackCurveView& curve, std::vector<double>& row) {
    return pillarJacobianRow(pillar, referenceDate, curve, row);
}

void assembleBasisJacobian(const SpreadCurve<double>& child,
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

void assembleForecastJacobian(const SpreadCurve<double>& child,
                              const std::vector<ForecastPillar>& pillars,
                              const datetime::Date& referenceDate,
                              const DiscountCurve<double>* discountCurve, std::vector<double>& f,
                              std::vector<double>& c) {
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
