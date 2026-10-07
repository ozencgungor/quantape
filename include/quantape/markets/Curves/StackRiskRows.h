#pragma once

#include "quantape/markets/Curves/CurveBuilder.h"
#include "quantape/markets/Curves/StackCurveView.h"

#include <cstddef>
#include <vector>

namespace quantape::markets {
/**
 * @file StackRiskRows.h
 * @brief Jacobian-row assembly for the stack risk engine
 *
 * The per-instrument row algebra lives in `quantape/pricing/Ir.h` behind
 * `jacobianRow` overloads templated on the type-erased view, so pricing never
 * depends on `StackCurveView`. This header keeps the tagged-pillar dispatcher
 * (`forecastPillarJacobianRows`, a `std::visit` over the concept instrument)
 * and the two block assemblers:
 *
 *  - `assembleBasisJacobian` assembles the child own-curve block and the cross
 *    block from basis pillars;
 *  - `assembleForecastJacobian` assembles the same blocks for deposits, FRAs,
 *    rate futures, par IRS and basis swaps, folding the discount row into the
 *    forecast-parent block when the child discounts on its parent.
 */

/// Own-curve, forecast-parent and discount rows of one forecast pillar.
/// Basis swaps use the parent row as the cross row and report a zero discount
/// row; IRS pillars report the parent and discount rows separately. Synthetic
/// deposits, FRAs and rate futures move with the child spread nodes and the
/// forecast parent one-for-one and report a zero discount row. Dispatches by
/// `std::visit` over the pillar's concept instrument.
void forecastPillarJacobianRows(const SpreadCurve<double>& child, const ForecastPillar& pillar,
                                const datetime::Date& referenceDate,
                                const DiscountCurve<double>* discountCurve,
                                std::vector<double>& fRow, std::vector<double>& parentRow,
                                std::vector<double>& discountRow);

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
