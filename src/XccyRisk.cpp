#include "quantape/markets/Curves/XccyRisk.h"

#include "quantape/markets/Curves/CurveRisk.h"
#include "quantape/markets/Curves/XccyBasisBuilder.h"

#include <cstddef>
#include <stdexcept>
#include <utility>
#include <vector>

namespace quantape::markets {

namespace {

/// Per-thread schedule cache: the cross-currency rows rebuild the same two
/// schedules on every call and every finite-difference bump.
detail::ScheduleCache& xccyScheduleCache() {
    static thread_local detail::ScheduleCache cache;
    return cache;
}

} // namespace

double xccyBasisSpreadView(const StackCurveView& foreignDiscount,
                           const StackCurveView& foreignForecast,
                           const StackCurveView& domesticDiscount,
                           const StackCurveView& domesticForecast, const XccyPillar& pillar,
                           const datetime::Date& referenceDate,
                           const datetime::DayCounter& zeroDayCounter) {
    const datetime::Schedule& foreignSchedule =
        xccyScheduleCache().get(referenceDate, pillar.maturity, pillar.foreignTenor,
                                pillar.foreignCalendar, pillar.foreignBusinessDayConvention);
    const datetime::Schedule& domesticSchedule =
        xccyScheduleCache().get(referenceDate, pillar.maturity, pillar.domesticTenor,
                                pillar.domesticCalendar, pillar.domesticBusinessDayConvention);

    const bool foreignResets = pillar.notional == XccyNotionalMode::MtM && pillar.resetForeignLeg;
    const bool domesticResets = pillar.notional == XccyNotionalMode::MtM && !pillar.resetForeignLeg;

    const auto legValue = [&](const datetime::Schedule& schedule,
                              const datetime::DayCounter& accrualDayCounter,
                              const StackCurveView& discount, const StackCurveView& forecast,
                              const StackCurveView& otherDiscount, bool resets, int paymentLag,
                              datetime::BusinessDayConvention businessDayConvention,
                              double& annuity) {
        const std::vector<datetime::Date>& dates = schedule.dates();
        double coupons = 0.0;
        double resetValue = 0.0;
        annuity = 0.0;
        for (std::size_t k = 1; k < dates.size(); ++k) {
            const double tau = datetime::yearFraction(dates[k - 1], dates[k], accrualDayCounter);
            if (!(tau > 0.0)) {
                throw std::invalid_argument("xccyBasisSpreadView: non-positive accrual");
            }
            const double tPrevious =
                datetime::yearFraction(referenceDate, dates[k - 1], zeroDayCounter);
            const double tAccrual = datetime::yearFraction(referenceDate, dates[k], zeroDayCounter);
            const datetime::Date payDate = schedule.calendar().advance(
                dates[k], datetime::Period(paymentLag, datetime::TimeUnit::Days),
                businessDayConvention);
            const double t = datetime::yearFraction(referenceDate, payDate, zeroDayCounter);
            const double df = discount.discount(t);
            const double forward =
                (forecast.discount(tPrevious) / forecast.discount(tAccrual) - 1.0) / tau;
            coupons += tau * df * forward;
            annuity += tau * df;
            if (resets) {
                const double ownStart = discount.discount(tPrevious);
                const double adjustment = otherDiscount.discount(tPrevious) / ownStart;
                resetValue +=
                    adjustment * (discount.discount(tAccrual) * (1.0 + forward * tau) - ownStart);
            }
        }
        if (resets) {
            return resetValue;
        }
        const double tStart = datetime::yearFraction(referenceDate, dates.front(), zeroDayCounter);
        const double tEnd = datetime::yearFraction(referenceDate, dates.back(), zeroDayCounter);
        const double notional = discount.discount(tStart) - discount.discount(tEnd);
        return coupons + notional;
    };

    double foreignAnnuity = 0.0;
    const double foreignValue =
        legValue(foreignSchedule, pillar.foreignDayCounter, foreignDiscount, foreignForecast,
                 domesticDiscount, foreignResets, pillar.foreignPaymentLag,
                 pillar.foreignBusinessDayConvention, foreignAnnuity);
    double domesticAnnuity = 0.0;
    const double domesticValue =
        legValue(domesticSchedule, pillar.domesticDayCounter, domesticDiscount, domesticForecast,
                 foreignDiscount, domesticResets, pillar.domesticPaymentLag,
                 pillar.domesticBusinessDayConvention, domesticAnnuity);
    if (!(foreignAnnuity > 0.0) || !(domesticAnnuity > 0.0)) {
        throw std::invalid_argument("xccyBasisSpreadView: non-positive annuity");
    }
    if (pillar.spreadOnForeignLeg) {
        return (domesticValue - foreignValue) / foreignAnnuity;
    }
    return (foreignValue - domesticValue) / domesticAnnuity;
}

namespace detail {

namespace {

/// Per-thread scratch for the leg row builder, reused across coupons and legs.
/// The previous/current weight buffers let each accrual boundary be evaluated
/// once: the current boundary of one coupon is the previous boundary of the
/// next.
struct LegRowScratch {
    std::vector<double> weights;
    std::vector<double> discountPrevious;
    std::vector<double> discountCurrent;
    std::vector<const StackCurveView*> chainViews;
    std::vector<std::vector<double>> chainPrevious;
    std::vector<std::vector<double>> chainCurrent;
    std::vector<XccyRowBlock*> chainRows;
};

LegRowScratch& legRowScratch() {
    static thread_local LegRowScratch scratch;
    return scratch;
}

} // namespace

void xccyWeightsAt(const StackCurveView& curve, double t, std::vector<double>& out) {
    if (t <= 0.0) {
        out.assign(curve.size(), 0.0);
        return;
    }
    curve.zeroNodeWeights(t, out);
}

XccyRowBlock& xccyRowBlock(std::vector<XccyRowBlock>& blocks, const StackCurveView& curve) {
    for (XccyRowBlock& block : blocks) {
        if (block.curve->identity() == curve.identity()) {
            return block;
        }
    }
    blocks.push_back(XccyRowBlock{&curve, std::vector<double>(curve.size() - 1, 0.0)});
    return blocks.back();
}

void xccyRegisterForecastChain(std::vector<XccyRowBlock>& blocks, const StackCurveView& forecast) {
    xccyRowBlock(blocks, forecast);
    for (const StackCurveView* ancestor = forecast.parentView(); ancestor != nullptr;
         ancestor = ancestor->parentView()) {
        xccyRowBlock(blocks, *ancestor);
    }
}

XccyLegRows xccyLegNodeRows(const datetime::Schedule& schedule,
                            const datetime::DayCounter& accrualDayCounter,
                            const StackCurveView& discount, const StackCurveView& forecast,
                            const StackCurveView* otherDiscount, bool resets, int paymentLag,
                            datetime::BusinessDayConvention businessDayConvention,
                            const datetime::Date& referenceDate,
                            const datetime::DayCounter& zeroDayCounter) {
    if (resets && otherDiscount == nullptr) {
        throw std::invalid_argument("xccyLegNodeRows: resetting leg without its other discount");
    }
    XccyLegRows rows;
    rows.valueRows.reserve(4);
    rows.annuityRows.reserve(1);
    // Register every referenced block before taking references into the
    // vectors, so a later push cannot invalidate an earlier reference.
    xccyRowBlock(rows.valueRows, discount);
    xccyRegisterForecastChain(rows.valueRows, forecast);
    if (resets) {
        xccyRowBlock(rows.valueRows, *otherDiscount);
    }
    xccyRowBlock(rows.annuityRows, discount);
    XccyRowBlock& discountValue = xccyRowBlock(rows.valueRows, discount);
    XccyRowBlock& discountAnnuity = xccyRowBlock(rows.annuityRows, discount);
    XccyRowBlock* otherValue = resets ? &xccyRowBlock(rows.valueRows, *otherDiscount) : nullptr;

    LegRowScratch& scratch = legRowScratch();
    std::vector<double>& weights = scratch.weights;
    std::vector<double>& discountPrevious = scratch.discountPrevious;
    std::vector<double>& discountCurrent = scratch.discountCurrent;
    std::vector<const StackCurveView*>& chainViews = scratch.chainViews;
    std::vector<std::vector<double>>& chainPrevious = scratch.chainPrevious;
    std::vector<std::vector<double>>& chainCurrent = scratch.chainCurrent;
    std::vector<XccyRowBlock*>& chainRows = scratch.chainRows;
    chainViews.clear();
    chainViews.push_back(&forecast);
    for (const StackCurveView* ancestor = forecast.parentView(); ancestor != nullptr;
         ancestor = ancestor->parentView()) {
        chainViews.push_back(ancestor);
    }
    const std::size_t chainLength = chainViews.size();
    chainPrevious.resize(chainLength);
    chainCurrent.resize(chainLength);
    chainRows.resize(chainLength);
    for (std::size_t a = 0; a < chainLength; ++a) {
        chainRows[a] = &xccyRowBlock(rows.valueRows, *chainViews[a]);
    }

    const std::vector<datetime::Date>& dates = schedule.dates();
    for (std::size_t k = 1; k < dates.size(); ++k) {
        const double tau = datetime::yearFraction(dates[k - 1], dates[k], accrualDayCounter);
        if (!(tau > 0.0)) {
            throw std::invalid_argument("xccySwapJacobianRowsView: non-positive accrual");
        }
        const double tPrevious =
            datetime::yearFraction(referenceDate, dates[k - 1], zeroDayCounter);
        const double tAccrual = datetime::yearFraction(referenceDate, dates[k], zeroDayCounter);
        const datetime::Date payDate = schedule.calendar().advance(
            dates[k], datetime::Period(paymentLag, datetime::TimeUnit::Days),
            businessDayConvention);
        const double t = datetime::yearFraction(referenceDate, payDate, zeroDayCounter);
        const double df = discount.discount(t);
        const double ratio = forecast.discount(tPrevious) / forecast.discount(tAccrual);
        const double forward = (ratio - 1.0) / tau;
        rows.annuity += tau * df;
        xccyWeightsAt(discount, t, weights);
        for (std::size_t i = 1; i < discount.size(); ++i) {
            const double dDiscount = -t * df * weights[i];
            discountAnnuity.row[i - 1] += tau * dDiscount;
        }
        double forecastScale = df;
        if (resets) {
            const double ownStart = discount.discount(tPrevious);
            const double ownEnd = discount.discount(tAccrual);
            const double adjustment = otherDiscount->discount(tPrevious) / ownStart;
            forecastScale = adjustment * ownEnd;
            rows.value += adjustment * (ownEnd * (1.0 + forward * tau) - ownStart);
            // Own discount: the reset adjustment's own-curve dependence cancels
            // against the start-notional discount, leaving the accrual end. The
            // accrual boundary weights carry into the next coupon as its start
            // boundary.
            if (k == 1) {
                xccyWeightsAt(discount, tPrevious, discountPrevious);
            }
            xccyWeightsAt(discount, tAccrual, discountCurrent);
            for (std::size_t i = 1; i < discount.size(); ++i) {
                discountValue.row[i - 1] +=
                    adjustment * (1.0 + forward * tau) * ownEnd *
                    (tPrevious * discountPrevious[i] - tAccrual * discountCurrent[i]);
            }
            discountPrevious.swap(discountCurrent);
            // Opposite discount: only the reset adjustment D_other/D_own moves.
            xccyWeightsAt(*otherDiscount, tPrevious, weights);
            for (std::size_t i = 1; i < otherDiscount->size(); ++i) {
                otherValue->row[i - 1] += adjustment * tPrevious * weights[i] *
                                          (ownStart - ownEnd * (1.0 + forward * tau));
            }
        } else {
            rows.value += tau * df * forward;
            for (std::size_t i = 1; i < discount.size(); ++i) {
                discountValue.row[i - 1] += tau * forward * (-t * df * weights[i]);
            }
        }
        // The child forecast's native nodes and every ancestor's nodes shift
        // the child zero curve one-for-one (additive spreads).
        for (std::size_t a = 0; a < chainLength; ++a) {
            const StackCurveView& curve = *chainViews[a];
            if (k == 1) {
                xccyWeightsAt(curve, tPrevious, chainPrevious[a]);
            }
            xccyWeightsAt(curve, tAccrual, chainCurrent[a]);
            XccyRowBlock& row = *chainRows[a];
            for (std::size_t i = 1; i < curve.size(); ++i) {
                row.row[i - 1] +=
                    forecastScale * ratio *
                    (-tPrevious * chainPrevious[a][i] + tAccrual * chainCurrent[a][i]);
            }
            chainPrevious[a].swap(chainCurrent[a]);
        }
    }
    if (resets) {
        return rows;
    }
    const double tStart = datetime::yearFraction(referenceDate, dates.front(), zeroDayCounter);
    const double tEnd = datetime::yearFraction(referenceDate, dates.back(), zeroDayCounter);
    rows.value += discount.discount(tStart) - discount.discount(tEnd);
    xccyWeightsAt(discount, tStart, weights);
    for (std::size_t i = 1; i < discount.size(); ++i) {
        discountValue.row[i - 1] -= tStart * discount.discount(tStart) * weights[i];
    }
    xccyWeightsAt(discount, tEnd, weights);
    for (std::size_t i = 1; i < discount.size(); ++i) {
        discountValue.row[i - 1] += tEnd * discount.discount(tEnd) * weights[i];
    }
    return rows;
}

const std::vector<double>* xccyAnnuityRow(const std::vector<XccyRowBlock>& blocks,
                                          const StackCurveView& curve) {
    for (const XccyRowBlock& block : blocks) {
        if (block.curve->identity() == curve.identity()) {
            return &block.row;
        }
    }
    return nullptr;
}

StackCurveView::Ptr xccyRebuildForBump(const StackCurveView& curve,
                                       const StackCurveView& bumpedBase,
                                       const StackCurveView::Ptr& bumped) {
    if (curve.identity() == bumpedBase.identity()) {
        return bumped;
    }
    std::vector<const StackCurveView*> chain;
    for (const StackCurveView* node = &curve; node != nullptr; node = node->parentView()) {
        chain.push_back(node);
    }
    std::size_t matched = chain.size();
    for (std::size_t i = 1; i < chain.size(); ++i) {
        if (chain[i]->identity() == bumpedBase.identity()) {
            matched = i;
            break;
        }
    }
    if (matched == chain.size()) {
        return nullptr;
    }
    StackCurveView::Ptr rebuilt = bumped;
    for (std::size_t i = matched; i > 0; --i) {
        rebuilt = chain[i - 1]->rebuildWithNode(0, 0.0, rebuilt);
    }
    return rebuilt;
}

void mapXccyRowBlocks(const std::vector<XccyRowBlock>& blocks,
                      const StackCurveView& foreignDiscount, const StackCurveView& domesticDiscount,
                      const StackCurveView& foreignForecast, const StackCurveView& domesticForecast,
                      std::vector<double>& fRow, std::vector<double>& cRow,
                      std::vector<double>& gRow, std::vector<double>& hRow) {
    fRow.assign(foreignDiscount.size() - 1, 0.0);
    cRow.assign(domesticDiscount.size() - 1, 0.0);
    gRow.assign(foreignForecast.size() - 1, 0.0);
    hRow.assign(domesticForecast.size() - 1, 0.0);
    for (const XccyRowBlock& block : blocks) {
        std::vector<double>* target = nullptr;
        if (block.curve->identity() == foreignDiscount.identity()) {
            target = &fRow;
        } else if (block.curve->identity() == domesticDiscount.identity()) {
            target = &cRow;
        } else if (block.curve->identity() == foreignForecast.identity()) {
            target = &gRow;
        } else if (block.curve->identity() == domesticForecast.identity()) {
            target = &hRow;
        } else if (sameCurveView(*block.curve, foreignDiscount)) {
            target = &fRow;
        } else if (sameCurveView(*block.curve, domesticDiscount)) {
            target = &cRow;
        } else if (sameCurveView(*block.curve, foreignForecast)) {
            target = &gRow;
        } else if (sameCurveView(*block.curve, domesticForecast)) {
            target = &hRow;
        }
        if (target == nullptr) {
            continue;
        }
        for (std::size_t i = 0; i < block.row.size(); ++i) {
            (*target)[i] += block.row[i];
        }
    }
}

} // namespace detail

std::vector<XccyRowBlock> xccySwapJacobianRowsView(const StackCurveView& foreignDiscount,
                                                   const StackCurveView& foreignForecast,
                                                   const StackCurveView& domesticDiscount,
                                                   const StackCurveView& domesticForecast,
                                                   const XccyPillar& pillar,
                                                   const datetime::Date& referenceDate,
                                                   const datetime::DayCounter& zeroDayCounter) {
    const datetime::Schedule& foreignSchedule =
        xccyScheduleCache().get(referenceDate, pillar.maturity, pillar.foreignTenor,
                                pillar.foreignCalendar, pillar.foreignBusinessDayConvention);
    const datetime::Schedule& domesticSchedule =
        xccyScheduleCache().get(referenceDate, pillar.maturity, pillar.domesticTenor,
                                pillar.domesticCalendar, pillar.domesticBusinessDayConvention);
    const bool foreignResets = pillar.notional == XccyNotionalMode::MtM && pillar.resetForeignLeg;
    const bool domesticResets = pillar.notional == XccyNotionalMode::MtM && !pillar.resetForeignLeg;
    detail::XccyLegRows foreign = detail::xccyLegNodeRows(
        foreignSchedule, pillar.foreignDayCounter, foreignDiscount, foreignForecast,
        &domesticDiscount, foreignResets, pillar.foreignPaymentLag,
        pillar.foreignBusinessDayConvention, referenceDate, zeroDayCounter);
    detail::XccyLegRows domestic = detail::xccyLegNodeRows(
        domesticSchedule, pillar.domesticDayCounter, domesticDiscount, domesticForecast,
        &foreignDiscount, domesticResets, pillar.domesticPaymentLag,
        pillar.domesticBusinessDayConvention, referenceDate, zeroDayCounter);
    if (!(foreign.annuity > 0.0) || !(domestic.annuity > 0.0)) {
        throw std::invalid_argument("xccySwapJacobianRowsView: non-positive annuity");
    }
    std::vector<XccyRowBlock> result;
    result.reserve(foreign.valueRows.size() + domestic.valueRows.size());
    // A resetting leg references the other leg's discount curve, so the two
    // legs can contribute rows on the same block. Merge by view identity so
    // each referenced block appears once in canonical order.
    const auto appendBlock = [&result](XccyRowBlock&& block) {
        for (XccyRowBlock& existing : result) {
            if (existing.curve->identity() == block.curve->identity()) {
                for (std::size_t i = 0; i < existing.row.size(); ++i) {
                    existing.row[i] += block.row[i];
                }
                return;
            }
        }
        result.push_back(std::move(block));
    };
    const double foreignDifference = domestic.value - foreign.value;
    const double domesticDifference = foreign.value - domestic.value;
    const double denominator = pillar.spreadOnForeignLeg ? foreign.annuity * foreign.annuity
                                                         : domestic.annuity * domestic.annuity;
    // Quote-leg blocks pick up the quotient-rule annuity term; the other leg's
    // rows are divided by the quote annuity only. The leg rows are consumed
    // in place and moved into the result, so each block keeps its own buffer.
    for (XccyRowBlock& block : foreign.valueRows) {
        const std::vector<double>* annuityRow =
            detail::xccyAnnuityRow(foreign.annuityRows, *block.curve);
        std::vector<double>& row = block.row;
        for (std::size_t i = 0; i < row.size(); ++i) {
            const double value = row[i];
            const double annuity = annuityRow != nullptr ? (*annuityRow)[i] : 0.0;
            if (pillar.spreadOnForeignLeg) {
                row[i] = (-value * foreign.annuity - foreignDifference * annuity) / denominator;
            } else {
                row[i] = value * domestic.annuity / denominator;
            }
        }
        appendBlock(std::move(block));
    }
    for (XccyRowBlock& block : domestic.valueRows) {
        const std::vector<double>* annuityRow =
            detail::xccyAnnuityRow(domestic.annuityRows, *block.curve);
        std::vector<double>& row = block.row;
        for (std::size_t i = 0; i < row.size(); ++i) {
            const double value = row[i];
            const double annuity = annuityRow != nullptr ? (*annuityRow)[i] : 0.0;
            if (pillar.spreadOnForeignLeg) {
                row[i] = value * foreign.annuity / denominator;
            } else {
                row[i] = (-value * domestic.annuity - domesticDifference * annuity) / denominator;
            }
        }
        appendBlock(std::move(block));
    }
    return result;
}

std::vector<XccyRowBlock> xccySwapJacobianRowsViewFiniteDifference(
    const StackCurveView& foreignDiscount, const StackCurveView& foreignForecast,
    const StackCurveView& domesticDiscount, const StackCurveView& domesticForecast,
    const XccyPillar& pillar, const datetime::Date& referenceDate,
    const datetime::DayCounter& zeroDayCounter) {
    std::vector<XccyRowBlock> blocks;
    detail::xccyRowBlock(blocks, foreignDiscount);
    detail::xccyRegisterForecastChain(blocks, foreignForecast);
    detail::xccyRowBlock(blocks, domesticDiscount);
    detail::xccyRegisterForecastChain(blocks, domesticForecast);
    const double step = 1e-6;
    for (XccyRowBlock& block : blocks) {
        const StackCurveView& base = *block.curve;
        for (std::size_t node = 1; node < base.size(); ++node) {
            const StackCurveView::Ptr plus = base.rebuildWithNode(node, step, nullptr);
            const StackCurveView::Ptr minus = base.rebuildWithNode(node, -step, nullptr);
            const auto evaluate = [&](const StackCurveView::Ptr& bumped) {
                const StackCurveView::Ptr foreignDiscountView =
                    detail::xccyRebuildForBump(foreignDiscount, base, bumped);
                const StackCurveView::Ptr foreignForecastView =
                    detail::xccyRebuildForBump(foreignForecast, base, bumped);
                const StackCurveView::Ptr domesticDiscountView =
                    detail::xccyRebuildForBump(domesticDiscount, base, bumped);
                const StackCurveView::Ptr domesticForecastView =
                    detail::xccyRebuildForBump(domesticForecast, base, bumped);
                return xccyBasisSpreadView(
                    foreignDiscountView ? *foreignDiscountView : foreignDiscount,
                    foreignForecastView ? *foreignForecastView : foreignForecast,
                    domesticDiscountView ? *domesticDiscountView : domesticDiscount,
                    domesticForecastView ? *domesticForecastView : domesticForecast, pillar,
                    referenceDate, zeroDayCounter);
            };
            block.row[node - 1] = (evaluate(plus) - evaluate(minus)) / (2.0 * step);
        }
    }
    return blocks;
}
} // namespace quantape::markets
