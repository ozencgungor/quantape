#pragma once

#include "quantape/datetime/Calendar.h"
#include "quantape/datetime/Schedule.h"
#include "quantape/markets/Curves/CurveBuilder.h"
#include "quantape/markets/Curves/DiscountCurve.h"
#include "quantape/math/LinearAlgebra/DenseSolve.h"
#include "quantape/pricing/Ir.h"

#include <algorithm>
#include <cmath>
#include <concepts>
#include <cstddef>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <variant>
#include <vector>

namespace quantape::markets {
/**
 * @file CurveRisk.h
 * @brief Quote-space risk for a single bootstrap curve
 *
 * Transforms zero-node sensitivities `∂V/∂z_i` into quote sensitivities
 * `∂V/∂r_j` through the instrument Jacobian `F = ∂r/∂z`:
 *
 *   `∂V/∂r = (F^T)^{-1} ∂V/∂z`,
 *
 * assembled analytically from the instrument definitions and the curve node
 * weights. The system is solved densely because stencil spillover (Akima /
 * tension splines) can put entries just above the diagonal. This is the
 * per-curve partial transform (parents frozen); the full stack derivative
 * adds the tree pass implemented in `StackRisk`.
 *
 * Capability notes: in-range risk weights are analytic for Linear, Akima,
 * TensionSpline and MixedLinearCubic, while MonotoneCubic and HymanSpline
 * throw. Extrapolated weights are analytic for every scheme. Cross-currency
 * rows are finite-difference based.
 */

namespace detail {

/// Partial-pivoted LU factorization of a row-major dense matrix: one O(n^3)
/// elimination reused across right-hand sides, each back-substitution O(n^2).
/// Row swaps touch only the active submatrix, matching the interleaved
/// elimination in `quantape::math::solveDense`; `solveDenseLu` replays the
/// swaps on the right-hand side in the same order.
struct DenseLu {
    std::size_t dim = 0;
    std::vector<double> lu;          ///< Row-major; L below the diagonal, U on/above
    std::vector<std::size_t> pivots; ///< Row swapped into position k at step k
};

/// Factor `matrix` (row-major `dim x dim`) with partial pivoting, consuming
/// the scratch copy. Throws when the matrix is singular.
DenseLu factorDenseLu(std::vector<double> matrix, std::size_t dim);

/// Solve `A x = rhs` from a factorization: pivot swaps, forward substitution
/// against the stored multipliers, then back substitution against U. The
/// arithmetic follows `quantape::math::solveDense` step for step.
std::vector<double> solveDenseLu(const DenseLu& factors, const std::vector<double>& rhs);

/// Solve `A x = rhs` from a factorization into a caller-owned buffer; the
/// buffer is resized to the factorization dimension and reused across solves
/// so a full inverse costs one allocation instead of one per column.
void solveDenseLuInto(const DenseLu& factors, const std::vector<double>& rhs,
                      std::vector<double>& x);

/// Allocation-free mapping of a tagged discount pillar to the concept
/// instrument: the row builders resolve their own schedules from the stored
/// conventions, so no quote times are prepared here.
inline DiscountInstrument riskInstrument(const CurvePillar& pillar) {
    using Instrument = DiscountInstrument::Instrument;
    switch (pillar.kind) {
        case PillarKind::Deposit: {
            instruments::Deposit out;
            out.maturity = pillar.maturity;
            out.start = pillar.start;
            out.calendar = pillar.calendar;
            out.businessDayConvention = pillar.businessDayConvention;
            out.quoteDayCounter = pillar.quoteDayCounter;
            out.quote = pillar.quote;
            return DiscountInstrument{Instrument{std::move(out)}};
        }
        case PillarKind::Repo: {
            instruments::Repo out;
            out.maturity = pillar.maturity;
            out.start = pillar.start;
            out.calendar = pillar.calendar;
            out.businessDayConvention = pillar.businessDayConvention;
            out.quoteDayCounter = pillar.quoteDayCounter;
            out.quote = pillar.quote;
            return DiscountInstrument{Instrument{std::move(out)}};
        }
        case PillarKind::Fra: {
            instruments::Fra out;
            out.start = pillar.start;
            out.maturity = pillar.maturity;
            out.calendar = pillar.calendar;
            out.businessDayConvention = pillar.businessDayConvention;
            out.quoteDayCounter = pillar.quoteDayCounter;
            out.fraConvexityExponent = pillar.fraConvexityExponent;
            out.quote = pillar.quote;
            return DiscountInstrument{Instrument{std::move(out)}};
        }
        case PillarKind::Future: {
            instruments::Future out;
            out.start = pillar.start;
            out.maturity = pillar.maturity;
            out.calendar = pillar.calendar;
            out.quoteDayCounter = pillar.quoteDayCounter;
            out.futureStyle = pillar.futureStyle;
            out.averagingStyle = pillar.averagingStyle;
            out.convexityAdjustment = pillar.convexityAdjustment;
            out.quote = pillar.quote;
            return DiscountInstrument{Instrument{std::move(out)}};
        }
        case PillarKind::OisSwap: {
            detail::OisPillarInstrument out;
            out.maturity = pillar.maturity;
            out.start = pillar.start;
            out.fixedTenor = pillar.fixedTenor;
            out.quoteDayCounter = pillar.quoteDayCounter;
            out.calendar = pillar.calendar;
            out.paymentLag = pillar.paymentLag;
            out.firstCouponFixed = pillar.firstCouponFixed;
            out.firstCouponRate = pillar.firstCouponRate;
            out.businessDayConvention = pillar.businessDayConvention;
            return DiscountInstrument{Instrument{std::move(out)}};
        }
    }
    throw std::invalid_argument("pillarJacobianRow: unknown pillar kind");
}

/// Allocation-free mapping of a tagged forecast pillar to the concept
/// instrument: the row builders resolve their own schedules from the stored
/// conventions, so no quote times are prepared here.
inline ForecastInstrument riskInstrument(const ForecastPillar& pillar) {
    using Instrument = ForecastInstrument::Instrument;
    switch (pillar.kind) {
        case ForecastPillar::Kind::Deposit: {
            instruments::Deposit out;
            out.maturity = pillar.maturity;
            out.start = pillar.start;
            out.calendar = pillar.calendar;
            out.businessDayConvention = pillar.businessDayConvention;
            out.quoteDayCounter = pillar.quoteDayCounter;
            out.quote = pillar.quote;
            return ForecastInstrument{Instrument{std::move(out)}};
        }
        case ForecastPillar::Kind::Fra: {
            instruments::Fra out;
            out.maturity = pillar.maturity;
            out.start = pillar.start;
            out.calendar = pillar.calendar;
            out.businessDayConvention = pillar.businessDayConvention;
            out.quoteDayCounter = pillar.quoteDayCounter;
            out.quote = pillar.quote;
            return ForecastInstrument{Instrument{std::move(out)}};
        }
        case ForecastPillar::Kind::Future: {
            instruments::Future out;
            out.start = pillar.start;
            out.maturity = pillar.maturity;
            out.calendar = pillar.calendar;
            out.quoteDayCounter = pillar.quoteDayCounter;
            out.futureStyle = pillar.futureStyle;
            out.averagingStyle = pillar.averagingStyle;
            out.convexityAdjustment = pillar.convexityAdjustment;
            out.quote = pillar.quote;
            return ForecastInstrument{Instrument{std::move(out)}};
        }
        case ForecastPillar::Kind::Irs:
            return ForecastInstrument{Instrument{pillar.irs}};
        case ForecastPillar::Kind::BasisSwap:
            return ForecastInstrument{Instrument{pillar.basis}};
    }
    throw std::invalid_argument("pillarJacobianRow: unknown forecast pillar kind");
}

} // namespace detail

struct QuoteRisk {
    std::vector<double> quoteDeltas; ///< ∂V/∂r_j per pillar
};

/// Full quote-space Hessian `H_r = J^T H_zeta J` with `J = F^{-1}` under the
/// Gauss-Newton approximation (bootstrap-curvature term omitted). `hessian` is
/// row-major `m x m` over pillars; diagonal entries are the diagonal gamma and
/// off-diagonals the cross gamma needed for IR ladder risk.
struct QuoteGamma {
    std::size_t dim = 0;
    std::vector<double> hessian;
    double at(std::size_t i, std::size_t j) const { return hessian[i * dim + j]; }
};

namespace detail {

/// Quote deltas from an already assembled row-major `F = dr/dz`: forms `F^T`,
/// fills the node-gradient vector and runs the dense solve. Throws with the
/// `transformQuoteRisk` message when the shapes do not match.
QuoteRisk transformQuoteRiskFromJacobian(const std::vector<double>& f, std::size_t m,
                                         const std::vector<double>& dVdZeros);

/// Quote Hessian from an already assembled row-major `F = dr/dz`, consuming the
/// matrix into the LU factorization. Throws with the `transformQuoteGamma`
/// message when the shapes do not match.
QuoteGamma transformQuoteGammaFromJacobian(std::vector<double> f, std::size_t m,
                                           const std::vector<double>& HVdZeros);

} // namespace detail

/// Analytic instrument Jacobian row contributions for one pillar, dispatched
/// by `std::visit` over the pillar's concept instrument. Returns false when
/// the pillar is numerically degenerate (non-positive annuity or discount).
template <typename CurveT>
    requires std::same_as<
        std::remove_cvref_t<decltype(std::declval<const CurveT&>().discount(0.0))>, double>
bool pillarJacobianRow(const CurvePillar& pillar, const datetime::Date& referenceDate,
                       const CurveT& curve, std::vector<double>& row) {
    const DiscountInstrument instrument = detail::riskInstrument(pillar);
    return std::visit(
        [&](const auto& value) -> bool {
            using InstrumentT = std::decay_t<decltype(value)>;
            if constexpr (std::is_same_v<InstrumentT, detail::OisPillarInstrument>) {
                const pricing::OisRowSpec spec{value.start,
                                               value.maturity,
                                               value.fixedTenor,
                                               value.calendar,
                                               value.businessDayConvention,
                                               value.paymentLag,
                                               value.quoteDayCounter,
                                               value.firstCouponFixed,
                                               value.firstCouponRate};
                return pricing::jacobianRow(spec, curve, referenceDate, row);
            } else {
                return pricing::jacobianRow(value, curve, referenceDate, row);
            }
        },
        instrument.instrument());
}

/// Assemble the instrument Jacobian `F = ∂r/∂z` over the solved nodes
/// (row-major `m x m`). Shared by the delta and gamma transforms.
template <typename CurveT>
void assembleQuoteJacobian(const CurveT& curve, const std::vector<CurvePillar>& pillars,
                           const datetime::Date& referenceDate, std::vector<double>& f,
                           std::vector<double>& rowScratch) {
    const std::size_t m = pillars.size();
    if (curve.size() != m + 1) {
        throw std::invalid_argument(
            "assembleQuoteJacobian: curve nodes must match the pillar count");
    }
    f.assign(m * m, 0.0);
    for (std::size_t j = 0; j < m; ++j) {
        if (!pillarJacobianRow(pillars[j], referenceDate, curve, rowScratch)) {
            throw std::invalid_argument("assembleQuoteJacobian: unsupported pillar");
        }
        for (std::size_t i = 0; i < m; ++i) {
            f[j * m + i] = rowScratch[i];
        }
    }
}

namespace detail {

/// Assembled row-major instrument Jacobian plus the zero-node quote deltas.
/// Shared by `transformQuoteRisk` and `curveRiskReport` so a joint delta and
/// gamma pass assembles the instrument rows only once.
template <typename CurveT>
QuoteRisk
transformQuoteRiskAndJacobian(const CurveT& curve, const std::vector<CurvePillar>& pillars,
                              const datetime::Date& referenceDate,
                              const std::vector<double>& dVdZeros, std::vector<double>& f) {
    const std::size_t m = pillars.size();
    if (curve.size() != m + 1 || dVdZeros.size() != curve.size()) {
        throw std::invalid_argument("transformQuoteRisk: size mismatch");
    }
    std::vector<double> row;
    assembleQuoteJacobian(curve, pillars, referenceDate, f, row);
    return transformQuoteRiskFromJacobian(f, m, dVdZeros);
}

} // namespace detail

/// Transform zero-node sensitivities (`dVdZeros`, same size as `curve.zeroNodeWeights`)
/// into quote deltas. Throws when sizes mismatch or a pillar is unsupported.
template <typename CurveT>
QuoteRisk transformQuoteRisk(const CurveT& curve, const std::vector<CurvePillar>& pillars,
                             const datetime::Date& referenceDate,
                             const std::vector<double>& dVdZeros) {
    std::vector<double> f;
    return detail::transformQuoteRiskAndJacobian(curve, pillars, referenceDate, dVdZeros, f);
}

/**
 * @brief Quote-space Hessian from zero-node second-order sensitivities.
 *
 * `HVdZeros` is the `n x n` row-major Hessian `d^2 V / dz_i dz_j` over curve
 * nodes (node 0 fixed; only rows/columns 1..n-1 are used). The transform is
 * `H_r = 0.5 (J^T H_zeta J + (J^T H_zeta J)^T)` with `J = F^{-1}`, so the
 * returned matrix is exactly symmetric.
 *
 * This is the **Gauss-Newton** form: the bootstrap-curvature term
 * `sum_i dV/dz_i * d^2 z_i / dr^2` is omitted. That term is material, not an
 * error bar — on a six-node OIS fixture it changes individual entries by O(1)
 * and flips signs — so use `markets::stackQuoteGamma` for the full transform on
 * a curve stack (it re-assembles the instrument Jacobian at bumped inputs and
 * costs roughly `2 * nNodes` full Jacobian assemblies). The two are not
 * interchangeable, and no error bar is reported for the omission here.
 */
template <typename CurveT>
QuoteGamma transformQuoteGamma(const CurveT& curve, const std::vector<CurvePillar>& pillars,
                               const datetime::Date& referenceDate,
                               const std::vector<double>& HVdZeros) {
    const std::size_t m = pillars.size();
    const std::size_t n = curve.size();
    if (n != m + 1 || HVdZeros.size() != n * n) {
        throw std::invalid_argument("transformQuoteGamma: size mismatch");
    }
    std::vector<double> f;
    std::vector<double> row;
    assembleQuoteJacobian(curve, pillars, referenceDate, f, row);
    return detail::transformQuoteGammaFromJacobian(std::move(f), m, HVdZeros);
}

} // namespace quantape::markets
