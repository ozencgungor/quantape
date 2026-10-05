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

    const auto legValue = [&](const datetime::Schedule& schedule,
                              const datetime::DayCounter& accrualDayCounter,
                              const StackCurveView& discount, const StackCurveView& forecast,
                              int paymentLag, datetime::BusinessDayConvention businessDayConvention,
                              double& annuity) {
        const std::vector<datetime::Date>& dates = schedule.dates();
        double coupons = 0.0;
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
        }
        const double tStart = datetime::yearFraction(referenceDate, dates.front(), zeroDayCounter);
        const double tEnd = datetime::yearFraction(referenceDate, dates.back(), zeroDayCounter);
        const double notional = discount.discount(tStart) - discount.discount(tEnd);
        return coupons + notional;
    };

    double foreignAnnuity = 0.0;
    const double foreignValue =
        legValue(foreignSchedule, pillar.foreignDayCounter, foreignDiscount, foreignForecast,
                 pillar.foreignPaymentLag, pillar.foreignBusinessDayConvention, foreignAnnuity);
    double domesticAnnuity = 0.0;
    const double domesticValue =
        legValue(domesticSchedule, pillar.domesticDayCounter, domesticDiscount, domesticForecast,
                 pillar.domesticPaymentLag, pillar.domesticBusinessDayConvention, domesticAnnuity);
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
struct LegRowScratch {
    std::vector<double> weights;
    std::vector<double> previous;
    std::vector<double> current;
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
                            int paymentLag, datetime::BusinessDayConvention businessDayConvention,
                            const datetime::Date& referenceDate,
                            const datetime::DayCounter& zeroDayCounter) {
    XccyLegRows rows;
    xccyRowBlock(rows.valueRows, discount);
    xccyRegisterForecastChain(rows.valueRows, forecast);
    xccyRowBlock(rows.annuityRows, discount);
    const std::vector<datetime::Date>& dates = schedule.dates();
    LegRowScratch& scratch = legRowScratch();
    std::vector<double>& weights = scratch.weights;
    std::vector<double>& previous = scratch.previous;
    std::vector<double>& current = scratch.current;
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
        rows.value += tau * df * forward;
        rows.annuity += tau * df;
        xccyWeightsAt(discount, t, weights);
        XccyRowBlock& discountValue = xccyRowBlock(rows.valueRows, discount);
        XccyRowBlock& discountAnnuity = xccyRowBlock(rows.annuityRows, discount);
        for (std::size_t i = 1; i < discount.size(); ++i) {
            const double dDiscount = -t * df * weights[i];
            discountValue.row[i - 1] += tau * forward * dDiscount;
            discountAnnuity.row[i - 1] += tau * dDiscount;
        }
        // The child forecast's native nodes and every ancestor's nodes shift
        // the child zero curve one-for-one (additive spreads).
        xccyWeightsAt(forecast, tPrevious, previous);
        xccyWeightsAt(forecast, tAccrual, current);
        XccyRowBlock& forecastRow = xccyRowBlock(rows.valueRows, forecast);
        for (std::size_t i = 1; i < forecast.size(); ++i) {
            forecastRow.row[i - 1] +=
                df * ratio * (-tPrevious * previous[i] + tAccrual * current[i]);
        }
        for (const StackCurveView* ancestor = forecast.parentView(); ancestor != nullptr;
             ancestor = ancestor->parentView()) {
            xccyWeightsAt(*ancestor, tPrevious, previous);
            xccyWeightsAt(*ancestor, tAccrual, current);
            XccyRowBlock& ancestorRow = xccyRowBlock(rows.valueRows, *ancestor);
            for (std::size_t i = 1; i < ancestor->size(); ++i) {
                ancestorRow.row[i - 1] +=
                    df * ratio * (-tPrevious * previous[i] + tAccrual * current[i]);
            }
        }
    }
    const double tStart = datetime::yearFraction(referenceDate, dates.front(), zeroDayCounter);
    const double tEnd = datetime::yearFraction(referenceDate, dates.back(), zeroDayCounter);
    rows.value += discount.discount(tStart) - discount.discount(tEnd);
    XccyRowBlock& discountValue = xccyRowBlock(rows.valueRows, discount);
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
    detail::XccyLegRows foreign =
        detail::xccyLegNodeRows(foreignSchedule, pillar.foreignDayCounter, foreignDiscount,
                                foreignForecast, pillar.foreignPaymentLag,
                                pillar.foreignBusinessDayConvention, referenceDate, zeroDayCounter);
    detail::XccyLegRows domestic = detail::xccyLegNodeRows(
        domesticSchedule, pillar.domesticDayCounter, domesticDiscount, domesticForecast,
        pillar.domesticPaymentLag, pillar.domesticBusinessDayConvention, referenceDate,
        zeroDayCounter);
    if (!(foreign.annuity > 0.0) || !(domestic.annuity > 0.0)) {
        throw std::invalid_argument("xccySwapJacobianRowsView: non-positive annuity");
    }
    std::vector<XccyRowBlock> result;
    result.reserve(foreign.valueRows.size() + domestic.valueRows.size());
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
        result.push_back(std::move(block));
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
        result.push_back(std::move(block));
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
