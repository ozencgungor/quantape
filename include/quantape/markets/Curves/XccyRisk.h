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
double xccyBasisSpreadView(const StackCurveView& foreignDiscount,
                           const StackCurveView& foreignForecast,
                           const StackCurveView& domesticDiscount,
                           const StackCurveView& domesticForecast, const XccyPillar& pillar,
                           const datetime::Date& referenceDate,
                           const datetime::DayCounter& zeroDayCounter);

namespace detail {

/// Zero-node weights of `curve` at `t`; the fixed t = 0 node carries no risk.
void xccyWeightsAt(const StackCurveView& curve, double t, std::vector<double>& out);

/// Existing block of `curve`, or a new zero row in canonical order.
XccyRowBlock& xccyRowBlock(std::vector<XccyRowBlock>& blocks, const StackCurveView& curve);

/// Register a forecast curve and every ancestor of its parent chain.
void xccyRegisterForecastChain(std::vector<XccyRowBlock>& blocks, const StackCurveView& forecast);

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
XccyLegRows xccyLegNodeRows(const datetime::Schedule& schedule,
                            const datetime::DayCounter& accrualDayCounter,
                            const StackCurveView& discount, const StackCurveView& forecast,
                            int paymentLag, datetime::BusinessDayConvention businessDayConvention,
                            const datetime::Date& referenceDate,
                            const datetime::DayCounter& zeroDayCounter);

/// The annuity row of `curve` if the leg references it, else null.
const std::vector<double>* xccyAnnuityRow(const std::vector<XccyRowBlock>& blocks,
                                          const StackCurveView& curve);

/// A copy of `curve` (or of its parent chain) with `bumpedBase` replaced by
/// `bumped`; null when `bumpedBase` is not `curve` or one of its ancestors.
StackCurveView::Ptr xccyRebuildForBump(const StackCurveView& curve,
                                       const StackCurveView& bumpedBase,
                                       const StackCurveView::Ptr& bumped);

/// Map per-block rows onto the four native blocks of the concrete adapter:
/// foreign discount, domestic discount, foreign forecast and domestic
/// forecast. A parent-chain row lands on a native block with the same view
/// identity, or (for the concrete adapter, where the chain is wrapped
/// locally) on a native block with the same discount function. Blocks of a
/// deeper chain that none of the four curves represents are not expressible
/// in this flat layout and are skipped; the per-block entry point is complete.
void mapXccyRowBlocks(const std::vector<XccyRowBlock>& blocks,
                      const StackCurveView& foreignDiscount, const StackCurveView& domesticDiscount,
                      const StackCurveView& foreignForecast, const StackCurveView& domesticForecast,
                      std::vector<double>& fRow, std::vector<double>& cRow,
                      std::vector<double>& gRow, std::vector<double>& hRow);

} // namespace detail

/// Analytic cross-currency Jacobian rows per referenced block. The blocks are
/// returned in canonical order: the leg discount curve, then the forecast
/// curve's native nodes, then each ancestor of its parent chain, with the
/// foreign leg before the domestic leg and duplicate views merged.
std::vector<XccyRowBlock> xccySwapJacobianRowsView(const StackCurveView& foreignDiscount,
                                                   const StackCurveView& foreignForecast,
                                                   const StackCurveView& domesticDiscount,
                                                   const StackCurveView& domesticForecast,
                                                   const XccyPillar& pillar,
                                                   const datetime::Date& referenceDate,
                                                   const datetime::DayCounter& zeroDayCounter);

/// Finite-difference reference for the analytic rows: central-differences the
/// const-notional spread over each block's native nodes, rebuilding every
/// forecast chain whose parent is bumped. Kept for test cross-checks.
std::vector<XccyRowBlock> xccySwapJacobianRowsViewFiniteDifference(
    const StackCurveView& foreignDiscount, const StackCurveView& foreignForecast,
    const StackCurveView& domesticDiscount, const StackCurveView& domesticForecast,
    const XccyPillar& pillar, const datetime::Date& referenceDate,
    const datetime::DayCounter& zeroDayCounter);

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
