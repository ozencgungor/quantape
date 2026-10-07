#pragma once

#include "quantape/datetime/Date.h"
#include "quantape/datetime/DayCounter.h"
#include "quantape/datetime/Period.h"
#include "quantape/datetime/Schedule.h"
#include "quantape/datetime/TimeConversion.h"
#include "quantape/instruments/Cashflow.h"
#include "quantape/instruments/XccyInstruments.h"
#include "quantape/pricing/IrMath.h"
#include "quantape/pricing/XccyMath.h"

#include <cstddef>
#include <stdexcept>
#include <vector>

namespace quantape::pricing {
/**
 * @file Xccy.h
 * @brief Cross-currency instrument overloads and known-flow materialization
 *
 * Composes the plain-data `instruments::XccyBasisSwap` with the prepared-times
 * algebra in `quantape/pricing/XccyMath.h`: the instrument overload forwards to
 * the member `impliedQuote<ScalarT>`, and `xccyCashflows` materializes the
 * quoted spread leg's known interest flows into `CashflowT` records. Bootstrap
 * residuals evaluate the prepared times directly, so Brent never allocates a
 * cashflow vector.
 */

/// Model basis spread of a cross-currency instrument over a cross-currency
/// curve set.
template <typename ScalarT, typename CurveSetT>
ScalarT impliedXccyBasisSpread(const instruments::XccyBasisSwap& pillar, const CurveSetT& curves) {
    return pillar.template impliedQuote<ScalarT>(curves);
}

/// Materializes the quoted spread leg's known interest flows: notional times
/// the quoted spread times each coupon's year fraction, paid on that leg's
/// payment schedule. A positive amount is received by the quoted leg. A
/// resetting notional changes the coupon notional per period, so `MtM` pillars
/// need a forecast curve and are rejected here.
template <typename ScalarT>
std::vector<instruments::CashflowT<ScalarT>>
xccyCashflowsT(const instruments::XccyBasisSwap& swap, const datetime::Date& referenceDate,
               const ScalarT& notional, const instruments::Currency& foreignCurrency,
               const instruments::Currency& domesticCurrency) {
    if (swap.notional == XccyNotionalMode::MtM) {
        throw std::invalid_argument("xccyCashflows: resetting notionals need a forecast curve");
    }
    const bool foreignSpread = swap.spreadOnForeignLeg;
    const datetime::Period& tenor = foreignSpread ? swap.foreignTenor : swap.domesticTenor;
    const datetime::Calendar& calendar =
        foreignSpread ? swap.foreignCalendar : swap.domesticCalendar;
    const datetime::DayCounter& dayCounter =
        foreignSpread ? swap.foreignDayCounter : swap.domesticDayCounter;
    const datetime::BusinessDayConvention businessDayConvention =
        foreignSpread ? swap.foreignBusinessDayConvention : swap.domesticBusinessDayConvention;
    const int paymentLag = foreignSpread ? swap.foreignPaymentLag : swap.domesticPaymentLag;
    const instruments::Currency& currency = foreignSpread ? foreignCurrency : domesticCurrency;
    const datetime::Schedule schedule(referenceDate, swap.maturity, tenor, calendar,
                                      businessDayConvention, datetime::DateGeneration::Forward,
                                      false, datetime::BusinessDayConvention::Unadjusted);
    const std::vector<datetime::Date>& dates = schedule.dates();
    std::vector<instruments::CashflowT<ScalarT>> flows;
    flows.reserve(dates.size() - 1);
    for (std::size_t k = 1; k < dates.size(); ++k) {
        const double tau = datetime::yearFraction(dates[k - 1], dates[k], dayCounter);
        const datetime::Date payDate =
            calendar.advance(dates[k], datetime::Period(paymentLag, datetime::TimeUnit::Days),
                             businessDayConvention);
        flows.push_back(instruments::CashflowT<ScalarT>{payDate, currency,
                                                        notional * ScalarT(swap.spread) * tau});
    }
    return flows;
}

/// Deterministic materialization of a cross-currency spread leg's known flows.
inline std::vector<instruments::Cashflow>
xccyCashflows(const instruments::XccyBasisSwap& swap, const datetime::Date& referenceDate,
              double notional, const instruments::Currency& foreignCurrency,
              const instruments::Currency& domesticCurrency) {
    return xccyCashflowsT<double>(swap, referenceDate, notional, foreignCurrency, domesticCurrency);
}

/// One row of the analytic cross-currency Jacobian, expressed on a curve's
/// native nodes. The curve type is a template parameter, so the header never
/// depends on the type-erased stack view.
template <class ViewT>
struct XccyRowBlockT {
    const ViewT* curve = nullptr; ///< Block the row belongs to
    std::vector<double> row;      ///< d r / d zeta over the block's nodes
};

namespace detail {

/// Per-thread schedule cache of the cross-currency row builders.
inline ScheduleCache& xccyScheduleCache() {
    static thread_local ScheduleCache cache;
    return cache;
}

/// Zero-node weights of `curve` at `t`; the fixed t = 0 node carries no risk.
template <class ViewT>
void xccyWeightsAt(const ViewT& curve, double t, std::vector<double>& out) {
    if (t <= 0.0) {
        out.assign(curve.size(), 0.0);
        return;
    }
    curve.zeroNodeWeights(t, out);
}

/// Existing block of `curve`, or a new zero row in canonical order.
template <class ViewT>
XccyRowBlockT<ViewT>& xccyRowBlock(std::vector<XccyRowBlockT<ViewT>>& blocks, const ViewT& curve) {
    for (XccyRowBlockT<ViewT>& block : blocks) {
        if (block.curve->identity() == curve.identity()) {
            return block;
        }
    }
    blocks.push_back(XccyRowBlockT<ViewT>{&curve, std::vector<double>(curve.size() - 1, 0.0)});
    return blocks.back();
}

/// Register a forecast curve and every ancestor of its parent chain.
template <class ViewT>
void xccyRegisterForecastChain(std::vector<XccyRowBlockT<ViewT>>& blocks, const ViewT& forecast) {
    xccyRowBlock(blocks, forecast);
    for (const ViewT* ancestor = forecast.parentView(); ancestor != nullptr;
         ancestor = ancestor->parentView()) {
        xccyRowBlock(blocks, *ancestor);
    }
}

/// Value, annuity and per-block node rows of one leg.
template <class ViewT>
struct XccyLegRowsT {
    double value = 0.0;
    double annuity = 0.0;
    std::vector<XccyRowBlockT<ViewT>> valueRows;   ///< dV/d zeta per block
    std::vector<XccyRowBlockT<ViewT>> annuityRows; ///< dA/d zeta per block
};

/// Per-thread scratch of the leg row builder, reused across coupons and legs.
template <class ViewT>
struct XccyLegRowScratch {
    std::vector<double> weights;
    std::vector<double> discountPrevious;
    std::vector<double> discountCurrent;
    std::vector<const ViewT*> chainViews;
    std::vector<std::vector<double>> chainPrevious;
    std::vector<std::vector<double>> chainCurrent;
    std::vector<XccyRowBlockT<ViewT>*> chainRows;
};

template <class ViewT>
XccyLegRowScratch<ViewT>& legRowScratch() {
    static thread_local XccyLegRowScratch<ViewT> scratch;
    return scratch;
}

/// Analytic leg rows: each coupon contributes `df * ratio * (-tprev w(tprev) +
/// taccrual w(taccrual))` on the forecast native and ancestor blocks and
/// `tau * forward * dD(tpay)` on the discount block; the notional adds
/// `dD(tStart) - dD(tEnd)`. With `resets` the notional term is replaced by the
/// telescoped reset value and the reset rows are added on the discount,
/// `otherDiscount` and forecast blocks; the annuity rows are unchanged.
template <class ViewT>
XccyLegRowsT<ViewT>
xccyLegNodeRows(const datetime::Schedule& schedule, const datetime::DayCounter& accrualDayCounter,
                const ViewT& discount, const ViewT& forecast, const ViewT* otherDiscount,
                bool resets, int paymentLag, datetime::BusinessDayConvention businessDayConvention,
                const datetime::Date& referenceDate, const datetime::DayCounter& zeroDayCounter) {
    if (resets && otherDiscount == nullptr) {
        throw std::invalid_argument("xccyLegNodeRows: resetting leg without its other discount");
    }
    XccyLegRowsT<ViewT> rows;
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
    XccyRowBlockT<ViewT>& discountValue = xccyRowBlock(rows.valueRows, discount);
    XccyRowBlockT<ViewT>& discountAnnuity = xccyRowBlock(rows.annuityRows, discount);
    XccyRowBlockT<ViewT>* otherValue =
        resets ? &xccyRowBlock(rows.valueRows, *otherDiscount) : nullptr;

    XccyLegRowScratch<ViewT>& scratch = legRowScratch<ViewT>();
    std::vector<double>& weights = scratch.weights;
    std::vector<double>& discountPrevious = scratch.discountPrevious;
    std::vector<double>& discountCurrent = scratch.discountCurrent;
    std::vector<const ViewT*>& chainViews = scratch.chainViews;
    std::vector<std::vector<double>>& chainPrevious = scratch.chainPrevious;
    std::vector<std::vector<double>>& chainCurrent = scratch.chainCurrent;
    std::vector<XccyRowBlockT<ViewT>*>& chainRows = scratch.chainRows;
    chainViews.clear();
    chainViews.push_back(&forecast);
    for (const ViewT* ancestor = forecast.parentView(); ancestor != nullptr;
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
            const ViewT& curve = *chainViews[a];
            if (k == 1) {
                xccyWeightsAt(curve, tPrevious, chainPrevious[a]);
            }
            xccyWeightsAt(curve, tAccrual, chainCurrent[a]);
            XccyRowBlockT<ViewT>& row = *chainRows[a];
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

/// The annuity row of `curve` if the leg references it, else null.
template <class ViewT>
const std::vector<double>* xccyAnnuityRow(const std::vector<XccyRowBlockT<ViewT>>& blocks,
                                          const ViewT& curve) {
    for (const XccyRowBlockT<ViewT>& block : blocks) {
        if (block.curve->identity() == curve.identity()) {
            return &block.row;
        }
    }
    return nullptr;
}

} // namespace detail

/// Model-implied const-notional basis spread evaluated on type-erased views.
/// Mirrors the par condition of the concrete `impliedXccyBasisSpread` for
/// curve trees whose copies are rebuilt per bump.
template <class ViewT>
double xccyBasisSpread(const ViewT& foreignDiscount, const ViewT& foreignForecast,
                       const ViewT& domesticDiscount, const ViewT& domesticForecast,
                       const instruments::XccyBasisSwap& pillar,
                       const datetime::Date& referenceDate,
                       const datetime::DayCounter& zeroDayCounter) {
    const datetime::Schedule& foreignSchedule = detail::xccyScheduleCache().get(
        referenceDate, pillar.maturity, pillar.foreignTenor, pillar.foreignCalendar,
        pillar.foreignBusinessDayConvention);
    const datetime::Schedule& domesticSchedule = detail::xccyScheduleCache().get(
        referenceDate, pillar.maturity, pillar.domesticTenor, pillar.domesticCalendar,
        pillar.domesticBusinessDayConvention);

    const bool foreignResets = pillar.notional == XccyNotionalMode::MtM && pillar.resetForeignLeg;
    const bool domesticResets = pillar.notional == XccyNotionalMode::MtM && !pillar.resetForeignLeg;

    const auto legValue = [&](const datetime::Schedule& schedule,
                              const datetime::DayCounter& accrualDayCounter, const ViewT& discount,
                              const ViewT& forecast, const ViewT& otherDiscount, bool resets,
                              int paymentLag, datetime::BusinessDayConvention businessDayConvention,
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

/// Analytic cross-currency Jacobian rows per referenced block. The blocks are
/// returned in canonical order: the leg discount curve, then the forecast
/// curve's native nodes, then each ancestor of its parent chain, with the
/// foreign leg before the domestic leg and duplicate views merged.
template <class ViewT>
std::vector<XccyRowBlockT<ViewT>> xccySwapJacobianRowsView(
    const ViewT& foreignDiscount, const ViewT& foreignForecast, const ViewT& domesticDiscount,
    const ViewT& domesticForecast, const instruments::XccyBasisSwap& pillar,
    const datetime::Date& referenceDate, const datetime::DayCounter& zeroDayCounter) {
    const datetime::Schedule& foreignSchedule = detail::xccyScheduleCache().get(
        referenceDate, pillar.maturity, pillar.foreignTenor, pillar.foreignCalendar,
        pillar.foreignBusinessDayConvention);
    const datetime::Schedule& domesticSchedule = detail::xccyScheduleCache().get(
        referenceDate, pillar.maturity, pillar.domesticTenor, pillar.domesticCalendar,
        pillar.domesticBusinessDayConvention);
    const bool foreignResets = pillar.notional == XccyNotionalMode::MtM && pillar.resetForeignLeg;
    const bool domesticResets = pillar.notional == XccyNotionalMode::MtM && !pillar.resetForeignLeg;
    detail::XccyLegRowsT<ViewT> foreign = detail::xccyLegNodeRows(
        foreignSchedule, pillar.foreignDayCounter, foreignDiscount, foreignForecast,
        &domesticDiscount, foreignResets, pillar.foreignPaymentLag,
        pillar.foreignBusinessDayConvention, referenceDate, zeroDayCounter);
    detail::XccyLegRowsT<ViewT> domestic = detail::xccyLegNodeRows(
        domesticSchedule, pillar.domesticDayCounter, domesticDiscount, domesticForecast,
        &foreignDiscount, domesticResets, pillar.domesticPaymentLag,
        pillar.domesticBusinessDayConvention, referenceDate, zeroDayCounter);
    if (!(foreign.annuity > 0.0) || !(domestic.annuity > 0.0)) {
        throw std::invalid_argument("xccySwapJacobianRowsView: non-positive annuity");
    }
    std::vector<XccyRowBlockT<ViewT>> result;
    result.reserve(foreign.valueRows.size() + domestic.valueRows.size());
    // A resetting leg references the other leg's discount curve, so the two
    // legs can contribute rows on the same block. Merge by view identity so
    // each referenced block appears once in canonical order.
    const auto appendBlock = [&result](XccyRowBlockT<ViewT>&& block) {
        for (XccyRowBlockT<ViewT>& existing : result) {
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
    for (XccyRowBlockT<ViewT>& block : foreign.valueRows) {
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
    for (XccyRowBlockT<ViewT>& block : domestic.valueRows) {
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

} // namespace quantape::pricing
