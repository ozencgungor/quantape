#pragma once

#include "quantape/datetime/Date.h"
#include "quantape/markets/Curves/Curve.h"
#include "quantape/markets/Curves/CurveBuilder.h"
#include "quantape/markets/Curves/DiscountCurve.h"

#include <cstddef>
#include <string>
#include <utility>
#include <vector>

namespace quantape::markets {
/**
 * @file CurveRiskReport.h
 * @brief v1 risk views over the per-quote deltas
 *
 * Aggregations: per-quote deltas (primary), by instrument kind, by short/long
 * end, by integer maturity year and by curve role, plus the pricing/risk
 * dual-scheme comparison and the hedge solve.
 */

struct RiskBucket {
    std::string label;
    double delta = 0.0;
};

struct CurveRiskReport {
    std::vector<std::string> quoteLabels; ///< Kind + maturity per pillar
    std::vector<double> quoteDeltas;      ///< Primary ∂V/∂r_j
    std::vector<RiskBucket> byKind;
    std::vector<RiskBucket> byEnd;  ///< ShortEnd / LongEnd
    std::vector<RiskBucket> byYear; ///< Maturity tags (date / year / month)
    double horizonYears = 0.0;

    // Second order (filled when node Hessians are supplied): per-quote
    // diagonal gamma, its maturity-tag buckets, and the cross-gamma matrix
    // (row-major, `gammaBucketLabels` x `gammaBucketLabels`).
    std::vector<double> gammaDiagonal;
    std::vector<RiskBucket> gammaDiagByYear;
    std::vector<std::string> gammaBucketLabels;
    std::vector<double> gammaCrossByYear;

    double totalDelta() const {
        double total = 0.0;
        for (const double delta : quoteDeltas) {
            total += delta;
        }
        return total;
    }
};

struct CurveRiskReportOptions {
    double shortEndYears = 2.0;
};

CurveRiskReport curveRiskReport(const DiscountCurve<double>& curve,
                                const std::vector<CurvePillar>& pillars,
                                const datetime::Date& referenceDate,
                                const std::vector<double>& dVdZeros,
                                const CurveRiskReportOptions& options = {},
                                const std::vector<double>* HVdZeros = nullptr);

/// Granular multi-curve view: one bucket per (curve role, instrument kind),
/// e.g. `Discount/OisSwap`, `Forecast/Fra`, `XccyBasis/OisSwap`,
/// `TurnOverlay/...`. This is the granular IR delta breakdown (basis, xccy
/// basis, FRA, par swaps, ...).
std::vector<RiskBucket>
aggregateByRoleAndKind(const std::vector<std::pair<CurveRole, CurveRiskReport>>& reports);

namespace detail {

/// Relative threshold under which a Householder QR diagonal counts as zero.
inline constexpr double kQrRankTolerance = 1e-13;

/// Column-pivoted Householder QR least-squares result: `rank` pivots cleared
/// the threshold, `solution` puts zero weight on the rank-deficient
/// directions, and `scale` is the largest diagonal magnitude seen.
struct QrLeastSquares {
    std::vector<double> solution;
    std::size_t rank = 0;
    double scale = 0.0;
};

/// Least squares `min ||matrix * x - rhs||^2` for a row-major `matrix` with
/// `rows >= cols`. `matrix` and `rhs` are consumed as scratch copies. Columns
/// are pivoted so the largest trailing norm sits on the diagonal: a diagonal
/// below the relative threshold is genuine rank deficiency, and the matching
/// coefficient is left at zero rather than amplified through a near-zero
/// reciprocal.
QrLeastSquares householderQrLeastSquares(std::vector<double> matrix, std::size_t rows,
                                         std::size_t cols, std::vector<double> rhs);

} // namespace detail

/// Least-squares hedge notionals: minimize ||J h + delta||^2 with an optional
/// ridge on the hedge weights. `jacobian` is row-major `m x h` (hedge
/// instrument sensitivities per bucket), `delta` the portfolio bucket deltas.
/// The solve is Householder QR on the augmented system `[J; sqrt(ridge) I]`,
/// which avoids the condition-number squaring of the normal equations.
std::vector<double> solveHedge(const std::vector<double>& jacobian, std::size_t m, std::size_t h,
                               const std::vector<double>& delta, double ridge = 0.0);

/// Dual pricing/risk scheme view: PV uses the pricing curve, the hedge view is
/// transformed through the local risk curve, and the difference is the
/// scheme-basis diagnostic (not a model derivative).
struct DualSchemeRisk {
    std::vector<double> audit; ///< Pricing-scheme deltas (exact view)
    std::vector<double> hedge; ///< Risk-scheme deltas (local hedge view)
    std::vector<double> basis; ///< hedge - audit, per bucket
};

DualSchemeRisk dualSchemeQuoteRisk(const DiscountCurve<double>& pricingCurve,
                                   const DiscountCurve<double>& riskCurve,
                                   const std::vector<CurvePillar>& pillars,
                                   const datetime::Date& referenceDate,
                                   const std::vector<double>& dVdZeros);

/// Role attribution: groups deltas by `CurveRole` of the curves in a stack.
/// The multi-curve stack transform lives in `StackRisk`; this helper sums
/// already-computed quote deltas per role.
std::vector<RiskBucket>
aggregateByRole(const std::vector<std::pair<CurveRole, std::vector<double>>>& quoteDeltasByCurve);

} // namespace quantape::markets
