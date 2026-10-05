#pragma once

#include "quantape/markets/Curves/StackCurveView.h"
#include "quantape/markets/Curves/XccyBasisBuilder.h"

#include <cstddef>
#include <stdexcept>
#include <vector>

namespace quantape::markets {
/**
 * @file XccyRisk.h
 * @brief Analytic cross-currency risk rows for the stack risk engine
 *
 * The const-notional cross-currency par spread is
 *
 *   `R = (V_quote - V_other) / A_quote`,
 *
 * where each leg contributes `V = sum_k tau_k D(tpay_k) f_k + D(tStart) -
 * D(tEnd)` on its own discount curve and its forecast curve's forwards, and
 * `A = sum_k tau_k D(tpay_k)`. Every partial is analytic: a discount node
 * enters through `dD(t) = -t D(t) w_i(t)`, the notional term through
 * `D(tStart) - D(tEnd)`, and a forecast node through the period ratio's
 * `(-tprev w_i(tprev) + taccrual w_i(taccrual))`. An additive forecast curve
 * shifts its child zero one-for-one with every curve on its parent chain, so
 * the same expression on an ancestor's node weights produces the ancestor
 * block row.
 *
 * Rows are returned per block, keyed by the view whose native nodes they are
 * expressed on, so the engine places each row by view identity. The quote
 * leg's blocks carry the quotient-rule term `dA`; the other leg's rows are
 * divided by the quote annuity only.
 *
 * `xccySwapJacobianRowsViewFiniteDifference` rebuilds each block by a bumped
 * view (following forecast parent chains) and central-differences the value
 * function, so it stays available as an independent reference for the
 * analytic rows and for the gamma curvature.
 */

/// Blocks a cross-currency child row references, by view identity. A reference
/// must be a curve of the same `stackQuoteRisk` input list; the child's own
/// curve is the input the row belongs to.
struct XccyRowInput {
    StackCurveView::Ptr foreignForecast;  ///< Foreign forecasting curve block
    StackCurveView::Ptr domesticForecast; ///< Domestic forecasting curve block
    StackCurveView::Ptr domesticDiscount; ///< Domestic discounting curve block
};

/// One row of the analytic cross-currency Jacobian, expressed on a stack
/// block's native nodes. The engine places `row` into the block matched by
/// `curve`'s view identity.
struct XccyRowBlock {
    const StackCurveView* curve = nullptr; ///< Block the row belongs to
    std::vector<double> row;               ///< d r / d zeta over the block's nodes
};

/// Model-implied const-notional basis spread with the coupons discounted on
/// the type-erased views. Mirrors the par condition of
/// `impliedXccyBasisSpread` for curve trees whose copies are rebuilt per bump.
inline double xccyBasisSpreadView(const StackCurveView& foreignDiscount,
                                  const StackCurveView& foreignForecast,
                                  const StackCurveView& domesticDiscount,
                                  const StackCurveView& domesticForecast, const XccyPillar& pillar,
                                  const datetime::Date& referenceDate,
                                  const datetime::DayCounter& zeroDayCounter) {
    const datetime::Schedule foreignSchedule(
        referenceDate, pillar.maturity, pillar.foreignTenor, pillar.foreignCalendar,
        pillar.foreignBusinessDayConvention, datetime::DateGeneration::Forward, false,
        datetime::BusinessDayConvention::Unadjusted);
    const datetime::Schedule domesticSchedule(
        referenceDate, pillar.maturity, pillar.domesticTenor, pillar.domesticCalendar,
        pillar.domesticBusinessDayConvention, datetime::DateGeneration::Forward, false,
        datetime::BusinessDayConvention::Unadjusted);

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

/// Zero-node weights of `curve` at `t`; the fixed t = 0 node carries no risk.
inline void xccyWeightsAt(const StackCurveView& curve, double t, std::vector<double>& out) {
    if (t <= 0.0) {
        out.assign(curve.size(), 0.0);
        return;
    }
    curve.zeroNodeWeights(t, out);
}

/// Existing block of `curve`, or a new zero row in canonical order.
inline XccyRowBlock& xccyRowBlock(std::vector<XccyRowBlock>& blocks, const StackCurveView& curve) {
    for (XccyRowBlock& block : blocks) {
        if (block.curve->identity() == curve.identity()) {
            return block;
        }
    }
    blocks.push_back(XccyRowBlock{&curve, std::vector<double>(curve.size() - 1, 0.0)});
    return blocks.back();
}

/// Register a forecast curve and every ancestor of its parent chain.
inline void xccyRegisterForecastChain(std::vector<XccyRowBlock>& blocks,
                                      const StackCurveView& forecast) {
    xccyRowBlock(blocks, forecast);
    for (const StackCurveView* ancestor = forecast.parentView(); ancestor != nullptr;
         ancestor = ancestor->parentView()) {
        xccyRowBlock(blocks, *ancestor);
    }
}

/// Value, annuity and per-block node rows of one const-notional leg.
struct XccyLegRows {
    double value = 0.0;
    double annuity = 0.0;
    std::vector<XccyRowBlock> valueRows;   ///< dV/d zeta per block
    std::vector<XccyRowBlock> annuityRows; ///< dA/d zeta per block
};

/// Analytic leg rows: each coupon contributes `df * ratio * (-tprev w(tprev) +
/// taccrual w(taccrual))` on the forecast native and ancestor blocks and
/// `tau * forward * dD(tpay)` on the discount block; the notional adds
/// `dD(tStart) - dD(tEnd)`.
inline XccyLegRows
xccyLegNodeRows(const datetime::Schedule& schedule, const datetime::DayCounter& accrualDayCounter,
                const StackCurveView& discount, const StackCurveView& forecast, int paymentLag,
                datetime::BusinessDayConvention businessDayConvention,
                const datetime::Date& referenceDate, const datetime::DayCounter& zeroDayCounter) {
    XccyLegRows rows;
    xccyRowBlock(rows.valueRows, discount);
    xccyRegisterForecastChain(rows.valueRows, forecast);
    xccyRowBlock(rows.annuityRows, discount);
    const std::vector<datetime::Date>& dates = schedule.dates();
    std::vector<double> weights;
    std::vector<double> previous;
    std::vector<double> current;
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

/// The annuity row of `curve` if the leg references it, else null.
inline const std::vector<double>* xccyAnnuityRow(const std::vector<XccyRowBlock>& blocks,
                                                 const StackCurveView& curve) {
    for (const XccyRowBlock& block : blocks) {
        if (block.curve->identity() == curve.identity()) {
            return &block.row;
        }
    }
    return nullptr;
}

/// A copy of `curve` (or of its parent chain) with `bumpedBase` replaced by
/// `bumped`; null when `bumpedBase` is not `curve` or one of its ancestors.
inline StackCurveView::Ptr xccyRebuildForBump(const StackCurveView& curve,
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

/// Map per-block rows onto the four native blocks of the concrete adapter:
/// foreign discount, domestic discount, foreign forecast and domestic
/// forecast. A parent-chain row lands on a native block with the same view
/// identity, or (for the concrete adapter, where the chain is wrapped
/// locally) on a native block with the same discount function. Blocks of a
/// deeper chain that none of the four curves represents are not expressible
/// in this flat layout and are skipped; the per-block entry point is complete.
inline void
mapXccyRowBlocks(const std::vector<XccyRowBlock>& blocks, const StackCurveView& foreignDiscount,
                 const StackCurveView& domesticDiscount, const StackCurveView& foreignForecast,
                 const StackCurveView& domesticForecast, std::vector<double>& fRow,
                 std::vector<double>& cRow, std::vector<double>& gRow, std::vector<double>& hRow) {
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

/// Analytic cross-currency Jacobian rows per referenced block. The blocks are
/// returned in canonical order: the leg discount curve, then the forecast
/// curve's native nodes, then each ancestor of its parent chain, with the
/// foreign leg before the domestic leg and duplicate views merged.
inline std::vector<XccyRowBlock> xccySwapJacobianRowsView(
    const StackCurveView& foreignDiscount, const StackCurveView& foreignForecast,
    const StackCurveView& domesticDiscount, const StackCurveView& domesticForecast,
    const XccyPillar& pillar, const datetime::Date& referenceDate,
    const datetime::DayCounter& zeroDayCounter) {
    const datetime::Schedule foreignSchedule(
        referenceDate, pillar.maturity, pillar.foreignTenor, pillar.foreignCalendar,
        pillar.foreignBusinessDayConvention, datetime::DateGeneration::Forward, false,
        datetime::BusinessDayConvention::Unadjusted);
    const datetime::Schedule domesticSchedule(
        referenceDate, pillar.maturity, pillar.domesticTenor, pillar.domesticCalendar,
        pillar.domesticBusinessDayConvention, datetime::DateGeneration::Forward, false,
        datetime::BusinessDayConvention::Unadjusted);
    const detail::XccyLegRows foreign =
        detail::xccyLegNodeRows(foreignSchedule, pillar.foreignDayCounter, foreignDiscount,
                                foreignForecast, pillar.foreignPaymentLag,
                                pillar.foreignBusinessDayConvention, referenceDate, zeroDayCounter);
    const detail::XccyLegRows domestic = detail::xccyLegNodeRows(
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
    // rows are divided by the quote annuity only.
    for (const XccyRowBlock& block : foreign.valueRows) {
        const std::vector<double>* annuityRow =
            detail::xccyAnnuityRow(foreign.annuityRows, *block.curve);
        std::vector<double> row(block.row.size(), 0.0);
        for (std::size_t i = 0; i < row.size(); ++i) {
            const double value = block.row[i];
            const double annuity = annuityRow != nullptr ? (*annuityRow)[i] : 0.0;
            if (pillar.spreadOnForeignLeg) {
                row[i] = (-value * foreign.annuity - foreignDifference * annuity) / denominator;
            } else {
                row[i] = value * domestic.annuity / denominator;
            }
        }
        result.push_back(XccyRowBlock{block.curve, std::move(row)});
    }
    for (const XccyRowBlock& block : domestic.valueRows) {
        const std::vector<double>* annuityRow =
            detail::xccyAnnuityRow(domestic.annuityRows, *block.curve);
        std::vector<double> row(block.row.size(), 0.0);
        for (std::size_t i = 0; i < row.size(); ++i) {
            const double value = block.row[i];
            const double annuity = annuityRow != nullptr ? (*annuityRow)[i] : 0.0;
            if (pillar.spreadOnForeignLeg) {
                row[i] = value * foreign.annuity / denominator;
            } else {
                row[i] = (-value * domestic.annuity - domesticDifference * annuity) / denominator;
            }
        }
        result.push_back(XccyRowBlock{block.curve, std::move(row)});
    }
    return result;
}

/// Finite-difference reference for the analytic rows: central-differences the
/// const-notional spread over each block's native nodes, rebuilding every
/// forecast chain whose parent is bumped. Kept for test cross-checks.
inline std::vector<XccyRowBlock> xccySwapJacobianRowsViewFiniteDifference(
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

/// Concrete-curve adapter: wraps every input in a view and maps the analytic
/// per-block rows onto `fRow` (foreign discount), `cRow` (domestic discount),
/// `gRow` (foreign forecast) and `hRow` (domestic forecast), so the native
/// four-vector layout stays available for direct comparisons.
inline void xccySwapJacobianRows(const DiscountCurve<double>& foreignDiscount,
                                 const XccyForecastCurve auto& foreignForecast,
                                 const DiscountCurve<double>& domesticDiscount,
                                 const XccyForecastCurve auto& domesticForecast,
                                 const XccyPillar& pillar, const datetime::Date& referenceDate,
                                 const datetime::DayCounter& zeroDayCounter,
                                 std::vector<double>& fRow, std::vector<double>& cRow,
                                 std::vector<double>& gRow, std::vector<double>& hRow) {
    const StackCurveView::Ptr foreignDiscountView = StackCurveView::make(foreignDiscount);
    const StackCurveView::Ptr foreignForecastView = StackCurveView::make(foreignForecast);
    const StackCurveView::Ptr domesticDiscountView = StackCurveView::make(domesticDiscount);
    const StackCurveView::Ptr domesticForecastView = StackCurveView::make(domesticForecast);
    const std::vector<XccyRowBlock> blocks =
        xccySwapJacobianRowsView(*foreignDiscountView, *foreignForecastView, *domesticDiscountView,
                                 *domesticForecastView, pillar, referenceDate, zeroDayCounter);
    detail::mapXccyRowBlocks(blocks, *foreignDiscountView, *domesticDiscountView,
                             *foreignForecastView, *domesticForecastView, fRow, cRow, gRow, hRow);
}

} // namespace quantape::markets
