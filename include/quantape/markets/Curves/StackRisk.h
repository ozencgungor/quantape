#pragma once

#include "quantape/markets/Curves/CurveBuilder.h"
#include "quantape/markets/Curves/CurveRisk.h"
#include "quantape/markets/Curves/CurveRiskReport.h"
#include "quantape/markets/Curves/SpreadCurve.h"
#include "quantape/markets/Curves/StackCurveView.h"
#include "quantape/markets/Curves/StackRiskRows.h"
#include "quantape/markets/Curves/TurnOverlay.h"
#include "quantape/markets/Curves/XccyBasisBuilder.h"
#include "quantape/markets/Curves/XccyRisk.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <type_traits>
#include <unordered_map>
#include <utility>
#include <variant>
#include <vector>

namespace quantape::markets {
/**
 * @file StackRisk.h
 * @brief Total quote risk across a curve tree (basis and IRS children)
 *
 * Child instruments reference both curves, so the full instrument Jacobian is
 * block lower-triangular:
 *
 *   `F_full = [[F_p, 0], [C, F_c]]`,  `C = d r_child / d zeta_parent`,
 *
 * and with `L = [[I,0],[P,I]]` (`zeta_child = P zeta_parent + s`) the exact
 * total quote sensitivities are
 *
 *   `x_c        = F_c^{-T} g_s`
 *   `dV/dr_root = F_p^{-T} (g_p - C^T x_c)`
 *   `dV/dr_child = x_c`
 *
 * with `g_p = dV/dz_parent |_s`, `g_s = dV/ds`. No separate `P` propagation
 * term appears in quote space: the cross block `C = d r_c / d z_parent |_s`
 * already carries the parent dependence of the child quotes, and the spread
 * re-solve is encoded in the block inverse of `[[F_p,0],[C,F_c]]`.
 *
 * Children are additive spread curves bootstrapped from basis-swap or par IRS
 * pillars, or cross-currency basis children whose rows are analytic over each
 * referenced curve's native nodes. The general engine supports arbitrary curve
 * trees: each forecast parent, exogenous discount curve and ancestor on their
 * parent chains must match a curve in the stack exactly (grid, discounts and
 * interpolation), so a block is never bound to a curve that only shares its
 * node values. Cross-currency rows bind their referenced blocks by view
 * identity and share the assembled matrix with every other child.
 */

/// A factor block contributes solved node columns and identity instrument rows.
/// Its output entry is the node-space hedge residual after every quote row, so
/// a forecast curve whose own quotes are not in the stack is reported without
/// dropping its sensitivity. `labelPrefix` heads the emitted point labels.
struct StackFactorInput {
    std::string labelPrefix;
    datetime::DayCounter nodeDayCounter; ///< Clock of the native node grid
};

/// One curve of a general stack tree: native quotes, node sensitivities and
/// (optionally) an exogenous discount curve. Inputs are listed in any order;
/// every parent or discount view, and every ancestor in their parent chains,
/// must match a curve in the list. Exactly one row mode is set per input:
/// discount pillars, forecast pillars, cross-currency pillars, or a factor
/// block.
struct StackCurveInput {
    StackCurveView::Ptr curve;
    CurveRole role = CurveRole::Discount;
    std::vector<CurvePillar> discountPillars;    ///< Discount-curve quote rows
    std::vector<ForecastPillar> forecastPillars; ///< Forecast-curve quote rows
    std::vector<XccyPillar> xccyPillars;         ///< Cross-currency child rows
    std::optional<XccyRowInput> xccy;            ///< References for xccyPillars
    std::optional<StackFactorInput> factor;      ///< Identity-row factor block
    std::vector<double> dVdNodes; ///< dV/d(zeta_i); size == curve->size(), node 0 unused
    StackCurveView::Ptr discount; ///< Exogenous discounting; null uses the parent view
};

/// Assembled stack quote system over native node coordinates: per-curve block
/// offsets and the row-major instrument Jacobian `F = d r / d zeta`.
struct StackQuoteSystem {
    std::size_t dim = 0;
    std::vector<std::size_t> offsets; ///< First node column of each input curve
    std::vector<double> jacobian;     ///< Row-major dim x dim
};

/// One quoted risk row of a stack curve in `stackQuoteRisk` order: the
/// instrument kind plus maturity label, the maturity-tag bucket, the rounded
/// maturity year and the role the quote reports under. `delta` carries the
/// quote-space derivative for a risk entry and the diagonal Hessian element
/// for a gamma entry.
struct QuotePoint {
    std::string label;
    std::string bucket; ///< riskMaturityTag
    int year = 0;       ///< Rounded maturity years
    CurveRole role = CurveRole::Discount;
    double delta = 0.0;
};

/// Throws unless every point carries a label, a maturity-tag bucket and a
/// finite delta; `expected` optionally pins the point count. Consuming a point
/// vector through this gate keeps partial metadata out of every aggregation.
inline void validateQuotePoints(const std::vector<QuotePoint>& points,
                                std::optional<std::size_t> expected = std::nullopt) {
    if (expected.has_value() && points.size() != *expected) {
        throw std::invalid_argument("validateQuotePoints: point count mismatch");
    }
    for (const QuotePoint& point : points) {
        if (point.label.empty()) {
            throw std::invalid_argument("validateQuotePoints: empty quote label");
        }
        if (point.bucket.empty()) {
            throw std::invalid_argument("validateQuotePoints: empty quote bucket");
        }
        if (!std::isfinite(point.delta)) {
            throw std::invalid_argument("validateQuotePoints: non-finite quote delta");
        }
    }
}

/// One curve's quote rows: the curve role plus the per-quote points in
/// `stackQuoteRisk` entry order. A direct turn knot keeps its bootstrap column
/// but reports under `TurnOverlay` with a `Turn <date>` label.
struct StackRiskEntry {
    CurveRole role = CurveRole::Discount; ///< Curve role; turn pillars override per quote
    std::vector<QuotePoint> points;       ///< Quote rows in curve/node order

    void validate() const { validateQuotePoints(points); }
};

/// Role buckets of a stack risk table. Sums each quote under its own role, so
/// direct turn knots report under `TurnOverlay` while the rest of their curve
/// keeps the curve role, and the buckets add back up to the full table.
inline std::vector<RiskBucket> stackRoleBuckets(const std::vector<StackRiskEntry>& entries) {
    std::vector<RiskBucket> buckets;
    const auto add = [&](std::string_view label, double delta) {
        for (RiskBucket& bucket : buckets) {
            if (bucket.label == label) {
                bucket.delta += delta;
                return;
            }
        }
        buckets.push_back(RiskBucket{std::string(label), delta});
    };
    for (const StackRiskEntry& entry : entries) {
        entry.validate();
        for (const QuotePoint& point : entry.points) {
            add(curveRoleName(point.role), point.delta);
        }
    }
    return buckets;
}

/// Overlay turn risk merged into a role-bucket list. Overlay turn amplitudes
/// are exogenous factors outside the quote Jacobian, so `stackRoleBuckets`
/// alone understates the `TurnOverlay` role; every `TurnRiskEntry` delta folds
/// into the `TurnOverlay` role bucket (or `role` when given), keeping role
/// sums complete. Labels stay on the original turn entries.
inline std::vector<RiskBucket> addTurnRiskBuckets(std::vector<RiskBucket> buckets,
                                                  const std::vector<TurnRiskEntry>& turns,
                                                  CurveRole role = CurveRole::TurnOverlay) {
    if (turns.empty()) {
        return buckets;
    }
    double total = 0.0;
    for (const TurnRiskEntry& turn : turns) {
        total += turn.delta;
    }
    const std::string label(curveRoleName(role));
    for (RiskBucket& bucket : buckets) {
        if (bucket.label == label) {
            bucket.delta += total;
            return buckets;
        }
    }
    buckets.push_back(RiskBucket{label, total});
    return buckets;
}

/// Role buckets of a stack risk table with exogenous overlay turn risk folded
/// in; equivalent to `addTurnRiskBuckets(stackRoleBuckets(entries), turns)`.
inline std::vector<RiskBucket> addTurnRiskBuckets(const std::vector<StackRiskEntry>& entries,
                                                  const std::vector<TurnRiskEntry>& turns,
                                                  CurveRole role = CurveRole::TurnOverlay) {
    return addTurnRiskBuckets(stackRoleBuckets(entries), turns, role);
}

/// Stack-level quote Hessian in quote space, row-major `dim x dim`.
struct StackQuoteGamma {
    std::size_t dim = 0;
    std::vector<QuotePoint> points; ///< One per quote, in `stackQuoteRisk` order
    std::vector<double> hessian;    ///< Row-major `dim x dim`

    /// Bounds- and size-checked access to the row-major Hessian.
    double at(std::size_t i, std::size_t j) const {
        if (dim == 0 || hessian.size() != dim * dim || i >= dim || j >= dim) {
            throw std::invalid_argument("StackQuoteGamma::at: index out of bounds");
        }
        return hessian[i * dim + j];
    }

    /// Throws unless the Hessian and the point metadata both match `dim`.
    void validate() const {
        if (hessian.size() != dim * dim) {
            throw std::invalid_argument("StackQuoteGamma::validate: hessian size mismatch");
        }
        validateQuotePoints(points, dim);
    }
};

struct StackChildInput {
    CurveRole role = CurveRole::Forecast;
    const SpreadCurve<double>* curve = nullptr;
    std::vector<ForecastPillar> pillars; ///< Basis-swap or par IRS instruments
    std::vector<double> dVdSpread;       ///< ∂V/∂s_i over the child nodes (node 0 = 0)
    /// Optional exogenous discount curve for IRS children. It must match the
    /// root grid and values; cross-currency children with their own foreign
    /// discount curves use the cross-currency driver instead.
    const DiscountCurve<double>* discountCurve = nullptr;
};

/// Depth-1 stack input list: the root discount curve as the first block and
/// one additive spread child per element, each parented on the root. Shared by
/// the legacy delta and gamma wrappers so both adapt the same way.
inline std::vector<StackCurveInput> makeDepth1Inputs(const DiscountCurve<double>& root,
                                                     const std::vector<CurvePillar>& rootPillars,
                                                     const std::vector<double>& dVdRoot,
                                                     const std::vector<StackChildInput>& children) {
    const StackCurveView::Ptr rootView = StackCurveView::make(root);
    std::vector<StackCurveInput> inputs;
    inputs.reserve(children.size() + 1);
    StackCurveInput rootInput;
    rootInput.curve = rootView;
    rootInput.role = CurveRole::Discount;
    rootInput.discountPillars = rootPillars;
    rootInput.dVdNodes = dVdRoot;
    inputs.push_back(std::move(rootInput));
    for (const StackChildInput& child : children) {
        if (child.curve == nullptr) {
            throw std::invalid_argument("makeDepth1Inputs: malformed child input");
        }
        StackCurveInput input;
        input.curve = StackCurveView::make(*child.curve);
        input.role = child.role;
        input.forecastPillars = child.pillars;
        input.dVdNodes = child.dVdSpread;
        if (child.discountCurve != nullptr) {
            input.discount = sameCurveValues(*child.discountCurve, root)
                                 ? rootView
                                 : StackCurveView::make(*child.discountCurve);
        }
        inputs.push_back(std::move(input));
    }
    return inputs;
}

/// Quote points of one stack input in `stackQuoteRisk` entry order. A direct
/// turn knot keeps its bootstrap column but reports under `TurnOverlay` with a
/// `Turn <date>` label. The caller fills each point's `delta`.
inline void appendStackQuoteMetadata(const StackCurveInput& input,
                                     const datetime::Date& referenceDate,
                                     std::vector<QuotePoint>& points) {
    if (!input.discountPillars.empty()) {
        for (const CurvePillar& pillar : input.discountPillars) {
            const datetime::Date maturity = pillarRiskMaturity(pillar);
            const double t =
                datetime::yearFraction(referenceDate, maturity, input.curve->zeroDayCounter());
            const std::string tag = riskMaturityTag(maturity, t);
            points.push_back(QuotePoint{std::string(pillarKindName(pillar.kind)) + " " + tag, tag,
                                        static_cast<int>(std::lround(t)), input.role, 0.0});
        }
        return;
    }
    for (const ForecastPillar& pillar : input.forecastPillars) {
        const datetime::Date maturity = forecastPillarRiskMaturity(pillar);
        const double t =
            datetime::yearFraction(referenceDate, maturity, input.curve->zeroDayCounter());
        const int year = static_cast<int>(std::lround(t));
        const CurveRole role = pillar.turnPillar ? CurveRole::TurnOverlay : input.role;
        if (pillar.turnPillar) {
            const datetime::Date start = pillar.start.serial() != 0 ? pillar.start : referenceDate;
            const std::string label = "Turn " + start.toIso();
            points.push_back(QuotePoint{label, label, year, role, 0.0});
        } else {
            const std::string tag = riskMaturityTag(maturity, t);
            points.push_back(
                QuotePoint{std::string(forecastPillarKindName(pillar.kind)) + " " + tag, tag, year,
                           role, 0.0});
        }
    }
}

/// Quote points of a cross-currency child input: one `Xccy <tag>` point per
/// pillar, aged on the domestic discount block's zero clock. The caller fills
/// each point's `delta`.
inline void appendXccyQuoteMetadata(const StackCurveInput& input,
                                    const datetime::Date& referenceDate,
                                    std::vector<QuotePoint>& points) {
    const datetime::DayCounter& zeroDayCounter = input.xccy->domesticDiscount->zeroDayCounter();
    for (const XccyPillar& pillar : input.xccyPillars) {
        const datetime::Date maturity =
            pillar.foreignCalendar.adjust(pillar.maturity, pillar.foreignBusinessDayConvention);
        const double t = datetime::yearFraction(referenceDate, maturity, zeroDayCounter);
        const std::string tag = riskMaturityTag(maturity, t);
        points.push_back(
            QuotePoint{"Xccy " + tag, tag, static_cast<int>(std::lround(t)), input.role, 0.0});
    }
}

/// Calendar date of a native forecast node time (definition below).
inline datetime::Date forecastNodeDate(const datetime::DayCounter& zeroDayCounter,
                                       const datetime::Date& referenceDate, double t);

/// Quote points of a factor block: one `<prefix> <tag>` point per solved node,
/// aged on the factor's own node clock. The caller fills each point's `delta`.
inline void appendFactorMetadata(const StackCurveInput& input, const datetime::Date& referenceDate,
                                 std::vector<QuotePoint>& points) {
    const std::vector<double>& times = input.curve->times();
    for (std::size_t i = 1; i < input.curve->size(); ++i) {
        const datetime::Date nodeDate =
            forecastNodeDate(input.factor->nodeDayCounter, referenceDate, times[i]);
        const std::string bucket = riskMaturityTag(nodeDate, times[i]);
        points.push_back(QuotePoint{input.factor->labelPrefix + " " + bucket, bucket,
                                    static_cast<int>(std::lround(times[i])), input.role, 0.0});
    }
}

/// Assemble the stack quote Jacobian `F = d r / d zeta` (row-major `dim x dim`)
/// and the per-curve node block offsets. Every instrument row is assembled
/// analytically over the view-native node coordinates; cross-currency child
/// rows arrive per block and are placed by view identity, and factor blocks
/// contribute identity rows so their solved value is the node-space residual.
inline StackQuoteSystem assembleStackQuoteSystem(const std::vector<StackCurveInput>& curves,
                                                 const datetime::Date& referenceDate) {
    if (curves.empty()) {
        throw std::invalid_argument("assembleStackQuoteSystem: no curves");
    }
    StackQuoteSystem system;
    system.offsets.assign(curves.size(), 0);
    for (std::size_t k = 0; k < curves.size(); ++k) {
        const StackCurveInput& input = curves[k];
        if (input.curve == nullptr || input.dVdNodes.size() != input.curve->size()) {
            throw std::invalid_argument("assembleStackQuoteSystem: malformed curve input");
        }
        const bool hasDiscount = !input.discountPillars.empty();
        const bool hasForecast = !input.forecastPillars.empty();
        const bool hasXccy = input.xccy.has_value();
        const bool hasFactor = input.factor.has_value();
        const int modes =
            (hasDiscount ? 1 : 0) + (hasForecast ? 1 : 0) + (hasXccy ? 1 : 0) + (hasFactor ? 1 : 0);
        if (modes != 1) {
            throw std::invalid_argument(
                "assembleStackQuoteSystem: exactly one pillar set must be non-empty");
        }
        const std::size_t rows = hasDiscount   ? input.discountPillars.size()
                                 : hasForecast ? input.forecastPillars.size()
                                 : hasXccy     ? input.xccyPillars.size()
                                               : input.curve->size() - 1;
        if (hasXccy) {
            if (input.xccyPillars.empty() || input.xccyPillars.size() + 1 != input.curve->size()) {
                throw std::invalid_argument(
                    "assembleStackQuoteSystem: xccy pillars must match the curve nodes");
            }
            if (input.xccy->foreignForecast == nullptr || input.xccy->domesticForecast == nullptr ||
                input.xccy->domesticDiscount == nullptr) {
                throw std::invalid_argument(
                    "assembleStackQuoteSystem: malformed xccy row references");
            }
        } else if (rows + 1 != input.curve->size()) {
            throw std::invalid_argument(
                "assembleStackQuoteSystem: pillars must match the curve nodes");
        }
        if (hasForecast && input.curve->parentView() == nullptr) {
            throw std::invalid_argument(
                "assembleStackQuoteSystem: forecast curve without a parent");
        }
        system.offsets[k] = system.dim;
        system.dim += input.curve->size() - 1;
    }
    const std::size_t dim = system.dim;
    // Curve matching walks every stack input and fingerprints interpolation
    // weights, so memoize the result per view identity: each ancestor of each
    // instrument row maps to the same block for the whole assembly.
    std::unordered_map<const void*, std::size_t> matchedBlocks;
    const auto matchCurve = [&](const StackCurveView& view) -> std::size_t {
        const auto cached = matchedBlocks.find(view.identity());
        if (cached != matchedBlocks.end()) {
            return cached->second;
        }
        for (std::size_t m = 0; m < curves.size(); ++m) {
            if (sameCurveView(*curves[m].curve, view)) {
                matchedBlocks.emplace(view.identity(), m);
                return m;
            }
        }
        throw std::invalid_argument("assembleStackQuoteSystem: curve is not part of the stack");
    };
    // Cross-currency rows bind their referenced blocks by view identity, so a
    // discount block and a forecast block on equal curve values stay distinct,
    // and a forecast curve that happens to be the child's own discount
    // function keeps its own block. A rebuilt ancestor copy (a gamma bump
    // re-binds a forecast parent by value) falls back to discount-function
    // equality, since only the stack can supply the parent's block.
    const auto matchIdentity = [&](const StackCurveView& view) -> std::size_t {
        for (std::size_t m = 0; m < curves.size(); ++m) {
            if (curves[m].curve->identity() == view.identity()) {
                return m;
            }
        }
        for (std::size_t m = 0; m < curves.size(); ++m) {
            if (sameCurveView(*curves[m].curve, view)) {
                return m;
            }
        }
        throw std::invalid_argument(
            "assembleStackQuoteSystem: xccy row block is not part of the stack");
    };
    system.jacobian.assign(dim * dim, 0.0);
    std::vector<double> scratch;
    std::size_t row = 0;
    for (std::size_t k = 0; k < curves.size(); ++k) {
        const StackCurveInput& input = curves[k];
        if (input.factor) {
            for (std::size_t i = 0; i + 1 < input.curve->size(); ++i) {
                system.jacobian[row * dim + system.offsets[k] + i] = 1.0;
                ++row;
            }
            continue;
        }
        if (input.xccy) {
            const XccyRowInput& refs = *input.xccy;
            for (const XccyPillar& pillar : input.xccyPillars) {
                const std::vector<XccyRowBlock> blocks = xccySwapJacobianRowsView(
                    *input.curve, *refs.foreignForecast, *refs.domesticDiscount,
                    *refs.domesticForecast, pillar, referenceDate,
                    refs.domesticDiscount->zeroDayCounter());
                for (const XccyRowBlock& block : blocks) {
                    const std::size_t target = matchIdentity(*block.curve);
                    for (std::size_t i = 0; i < block.row.size(); ++i) {
                        system.jacobian[row * dim + system.offsets[target] + i] += block.row[i];
                    }
                }
                ++row;
            }
            continue;
        }
        if (!input.discountPillars.empty()) {
            for (const CurvePillar& pillar : input.discountPillars) {
                if (!discountPillarJacobianRow(pillar, referenceDate, *input.curve, scratch)) {
                    throw std::invalid_argument("assembleStackQuoteSystem: degenerate pillar");
                }
                for (std::size_t i = 0; i < scratch.size(); ++i) {
                    system.jacobian[row * dim + system.offsets[k] + i] = scratch[i];
                }
                ++row;
            }
            continue;
        }
        const StackCurveView& parent = *input.curve->parentView();
        const StackCurveView& discount = input.discount ? *input.discount : parent;
        const auto assembleRows = [&](const ForecastPillar& pillar,
                                      const StackCurveView* parentWeights,
                                      const StackCurveView* discountWeights,
                                      std::vector<double>& ownRow, std::vector<double>& parentRow,
                                      std::vector<double>& discountRow) {
            switch (pillar.kind) {
                case ForecastPillar::Kind::Irs:
                    irsSwapJacobianRowsView(*input.curve, parent, discount, pillar.irs,
                                            referenceDate, ownRow, parentRow, discountRow,
                                            parentWeights, discountWeights);
                    return;
                case ForecastPillar::Kind::Deposit:
                case ForecastPillar::Kind::Fra:
                    forecastSimpleJacobianRowsView(*input.curve, parent, discount, pillar,
                                                   referenceDate, ownRow, parentRow, discountRow,
                                                   parentWeights, discountWeights);
                    return;
                case ForecastPillar::Kind::Future:
                    forecastFutureJacobianRowsView(*input.curve, parent, discount, pillar,
                                                   referenceDate, ownRow, parentRow, discountRow,
                                                   parentWeights, discountWeights);
                    return;
                case ForecastPillar::Kind::BasisSwap:
                    basisSwapJacobianRowsView(*input.curve, parent, discount, pillar.basis,
                                              referenceDate, ownRow, parentRow, discountRow,
                                              parentWeights, discountWeights);
                    return;
            }
            throw std::invalid_argument("assembleStackQuoteSystem: unknown forecast pillar kind");
        };
        for (const ForecastPillar& pillar : input.forecastPillars) {
            std::vector<double> ownRow;
            std::vector<double> parentRow;
            std::vector<double> discountRow;
            assembleRows(pillar, nullptr, nullptr, ownRow, parentRow, discountRow);
            for (std::size_t i = 0; i < ownRow.size(); ++i) {
                system.jacobian[row * dim + system.offsets[k] + i] = ownRow[i];
            }
            // The native pass already assembled the forecast-parent row on the
            // parent grid and the discount row on the discount grid; the first
            // ancestor of each chain is exactly that view, so reuse those rows
            // instead of assembling them a second time.
            bool reuseParentRow = true;
            // A depth-2 grandchild moves with every curve on its parent chain,
            // so the forecast-parent partial is expressed on each ancestor's
            // node grid as well.
            for (const StackCurveView* ancestor = &parent; ancestor != nullptr;
                 ancestor = ancestor->parentView()) {
                const std::size_t block = matchCurve(*ancestor);
                if (!reuseParentRow) {
                    assembleRows(pillar, ancestor, nullptr, ownRow, parentRow, discountRow);
                }
                for (std::size_t i = 0; i < parentRow.size(); ++i) {
                    system.jacobian[row * dim + system.offsets[block] + i] += parentRow[i];
                }
                reuseParentRow = false;
            }
            // Likewise for the discount curve's own parent chain.
            bool reuseDiscountRow = true;
            for (const StackCurveView* ancestor = &discount; ancestor != nullptr;
                 ancestor = ancestor->parentView()) {
                const std::size_t block = matchCurve(*ancestor);
                if (!reuseDiscountRow) {
                    assembleRows(pillar, nullptr, ancestor, ownRow, parentRow, discountRow);
                }
                for (std::size_t i = 0; i < discountRow.size(); ++i) {
                    system.jacobian[row * dim + system.offsets[block] + i] += discountRow[i];
                }
                reuseDiscountRow = false;
            }
            ++row;
        }
    }
    return system;
}

/// Rebuild every curve view after bumping one native node of the stack. The
/// bumped input receives the node bump; every other input whose parent
/// identity chain reaches a bumped view is rebuilt with the bumped parent,
/// discount views are re-pointed at the bumped curve they reference, and
/// cross-currency block references follow their rebuilt views.
inline std::vector<StackCurveInput> bumpStackInputs(const std::vector<StackCurveInput>& curves,
                                                    std::size_t bumpedCurve, std::size_t node,
                                                    double delta) {
    std::vector<StackCurveInput> result = curves;
    std::vector<StackCurveView::Ptr> views(curves.size());
    for (std::size_t k = 0; k < curves.size(); ++k) {
        views[k] = curves[k].curve;
    }
    std::unordered_map<const void*, StackCurveView::Ptr> rebuilt;
    views[bumpedCurve] = curves[bumpedCurve].curve->rebuildWithNode(node, delta, nullptr);
    rebuilt[curves[bumpedCurve].curve->identity()] = views[bumpedCurve];
    bool changed = true;
    while (changed) {
        changed = false;
        for (std::size_t k = 0; k < curves.size(); ++k) {
            if (k == bumpedCurve || views[k] != curves[k].curve) {
                continue;
            }
            const StackCurveView* parent = curves[k].curve->parentView();
            if (parent == nullptr) {
                continue;
            }
            const auto found = rebuilt.find(parent->identity());
            if (found != rebuilt.end()) {
                views[k] = curves[k].curve->rebuildWithNode(0, 0.0, found->second);
                rebuilt[curves[k].curve->identity()] = views[k];
                changed = true;
            }
        }
    }
    for (std::size_t k = 0; k < curves.size(); ++k) {
        result[k].curve = views[k];
        if (curves[k].discount == nullptr) {
            continue;
        }
        const auto found = rebuilt.find(curves[k].discount->identity());
        if (found != rebuilt.end()) {
            result[k].discount = found->second;
            continue;
        }
        for (const StackCurveView* ancestor = curves[k].discount->parentView(); ancestor != nullptr;
             ancestor = ancestor->parentView()) {
            const auto ancestorFound = rebuilt.find(ancestor->identity());
            if (ancestorFound != rebuilt.end()) {
                result[k].discount =
                    curves[k].discount->rebuildWithNode(0, 0.0, ancestorFound->second);
                break;
            }
        }
    }
    for (StackCurveInput& input : result) {
        if (!input.xccy) {
            continue;
        }
        const auto repoint = [&](StackCurveView::Ptr& reference) {
            const auto found = rebuilt.find(reference->identity());
            if (found != rebuilt.end()) {
                reference = found->second;
            }
        };
        repoint(input.xccy->foreignForecast);
        repoint(input.xccy->domesticForecast);
        repoint(input.xccy->domesticDiscount);
    }
    return result;
}

/// Total quote risk over an arbitrary curve tree. Every instrument row is
/// assembled analytically over the view-native node coordinates, and the full
/// system `F^T x = g` is solved densely, so curve depth and exogenous
/// discounting only change which column block a row term lands in.
inline std::vector<StackRiskEntry> stackQuoteRisk(const std::vector<StackCurveInput>& curves,
                                                  const datetime::Date& referenceDate) {
    const StackQuoteSystem system = assembleStackQuoteSystem(curves, referenceDate);
    const std::size_t dim = system.dim;
    const std::vector<std::size_t>& offset = system.offsets;
    const std::vector<double>& f = system.jacobian;
    std::vector<double> fTranspose(dim * dim);
    for (std::size_t i = 0; i < dim; ++i) {
        for (std::size_t j = 0; j < dim; ++j) {
            fTranspose[j * dim + i] = f[i * dim + j];
        }
    }
    std::vector<double> g(dim);
    for (std::size_t k = 0; k < curves.size(); ++k) {
        for (std::size_t i = 1; i < curves[k].curve->size(); ++i) {
            g[offset[k] + i - 1] = curves[k].dVdNodes[i];
        }
    }
    const std::vector<double> x = quantape::math::solveDense(std::move(fTranspose), dim, g);
    std::vector<StackRiskEntry> result;
    result.reserve(curves.size());
    for (std::size_t k = 0; k < curves.size(); ++k) {
        const StackCurveInput& input = curves[k];
        StackRiskEntry entry;
        entry.role = input.role;
        if (input.factor) {
            appendFactorMetadata(input, referenceDate, entry.points);
        } else if (input.xccy) {
            appendXccyQuoteMetadata(input, referenceDate, entry.points);
        } else {
            appendStackQuoteMetadata(input, referenceDate, entry.points);
        }
        for (std::size_t i = 0; i < entry.points.size(); ++i) {
            entry.points[i].delta = x[offset[k] + i];
        }
        result.push_back(std::move(entry));
    }
    return result;
}

/// Depth-1 convenience overload: the root plus additive spread children whose
/// parents must be the root curve. Implemented over the general engine.
inline std::vector<StackRiskEntry> stackQuoteRisk(const DiscountCurve<double>& root,
                                                  const std::vector<CurvePillar>& rootPillars,
                                                  const std::vector<double>& dVdRoot,
                                                  const std::vector<StackChildInput>& children,
                                                  const datetime::Date& referenceDate) {
    return stackQuoteRisk(makeDepth1Inputs(root, rootPillars, dVdRoot, children), referenceDate);
}

/**
 * @brief Exact quote-space stack gamma.
 *
 * `HZeta` is the `dim x dim` row-major portfolio Hessian
 * `d^2 V / dzeta_i dzeta_j` in stack-native node coordinates: per-curve solved
 * nodes (node 0 dropped), curves in input order, the same ordering as the
 * delta columns. With `F = d r / d zeta` assembled exactly as in
 * `stackQuoteRisk`, `J = F^{-1}` the dense inverse, `g` the concatenated node
 * gradient and `x = J^T g` the quote-space delta, the exact transform is
 *
 *   `H_r = J^T HZeta J - sum_r x_r (J^T G_r J)`,
 *
 * where `G_r = dF_r/dzeta` is the `dim x dim` Jacobian of quote row `r` with
 * respect to the native nodes, indexed `[column][bump direction]`. The second
 * term is the bootstrap-curvature correction; the first term alone is the
 * Gauss-Newton approximation. `G` is built by central differences with step
 * `1e-6` on the instrument Jacobian, bumping one native node and rebuilding
 * every curve that depends on it through its parent or discount chain. Cross-
 * currency rows are analytic, so their curvature is smooth and the same
 * central-difference pass applies. The result is symmetrized to absorb dense-
 * product round-off.
 */
inline StackQuoteGamma stackQuoteGamma(const std::vector<StackCurveInput>& curves,
                                       const std::vector<double>& HZeta,
                                       const datetime::Date& referenceDate) {
    const StackQuoteSystem system = assembleStackQuoteSystem(curves, referenceDate);
    const std::size_t dim = system.dim;
    if (HZeta.size() != dim * dim) {
        throw std::invalid_argument("stackQuoteGamma: HZeta size mismatch");
    }
    const std::vector<std::size_t>& offset = system.offsets;
    const std::vector<double>& f = system.jacobian;
    // J = F^{-1} by one partial-pivoted LU factorization plus dim
    // back-substitutions, instead of dim independent dense solves.
    const detail::DenseLu factors = detail::factorDenseLu(f, dim);
    std::vector<double> jacobian(dim * dim, 0.0);
    std::vector<double> unit(dim, 0.0);
    for (std::size_t column = 0; column < dim; ++column) {
        std::fill(unit.begin(), unit.end(), 0.0);
        unit[column] = 1.0;
        const std::vector<double> solution = detail::solveDenseLu(factors, unit);
        for (std::size_t k = 0; k < dim; ++k) {
            jacobian[k * dim + column] = solution[k];
        }
    }
    // Node gradient and quote-space delta x = J^T g.
    std::vector<double> g(dim, 0.0);
    for (std::size_t k = 0; k < curves.size(); ++k) {
        for (std::size_t i = 1; i < curves[k].curve->size(); ++i) {
            g[offset[k] + i - 1] = curves[k].dVdNodes[i];
        }
    }
    std::vector<double> x(dim, 0.0);
    for (std::size_t a = 0; a < dim; ++a) {
        double sum = 0.0;
        for (std::size_t i = 0; i < dim; ++i) {
            sum += jacobian[i * dim + a] * g[i];
        }
        x[a] = sum;
    }
    // weighted[c][k] = sum_r x_r dF[r][c] / dzeta_k, accumulated bump by bump.
    const double step = 1e-6;
    std::vector<double> weighted(dim * dim, 0.0);
    for (std::size_t j = 0; j < curves.size(); ++j) {
        for (std::size_t i = 1; i < curves[j].curve->size(); ++i) {
            const std::size_t bump = offset[j] + i - 1;
            const StackQuoteSystem plusSystem =
                assembleStackQuoteSystem(bumpStackInputs(curves, j, i, step), referenceDate);
            const StackQuoteSystem minusSystem =
                assembleStackQuoteSystem(bumpStackInputs(curves, j, i, -step), referenceDate);
            for (std::size_t column = 0; column < dim; ++column) {
                for (std::size_t r = 0; r < dim; ++r) {
                    const double curvature = (plusSystem.jacobian[r * dim + column] -
                                              minusSystem.jacobian[r * dim + column]) /
                                             (2.0 * step);
                    weighted[column * dim + bump] += x[r] * curvature;
                }
            }
        }
    }
    // H_r = J^T (HZeta - M) J.
    std::vector<double> combined = HZeta;
    for (std::size_t k = 0; k < combined.size(); ++k) {
        combined[k] -= weighted[k];
    }
    std::vector<double> tmp(dim * dim, 0.0);
    for (std::size_t i = 0; i < dim; ++i) {
        for (std::size_t b = 0; b < dim; ++b) {
            double sum = 0.0;
            for (std::size_t a = 0; a < dim; ++a) {
                sum += combined[i * dim + a] * jacobian[a * dim + b];
            }
            tmp[i * dim + b] = sum;
        }
    }
    StackQuoteGamma result;
    result.dim = dim;
    result.hessian.assign(dim * dim, 0.0);
    for (std::size_t i = 0; i < dim; ++i) {
        for (std::size_t b = 0; b < dim; ++b) {
            double sum = 0.0;
            for (std::size_t a = 0; a < dim; ++a) {
                sum += jacobian[a * dim + i] * tmp[a * dim + b];
            }
            result.hessian[i * dim + b] = sum;
        }
    }
    // Symmetrize: round-off in the dense products can leave an asymmetric
    // residual even though the exact transform is symmetric.
    for (std::size_t i = 0; i < dim; ++i) {
        for (std::size_t j = i + 1; j < dim; ++j) {
            const double symmetric =
                0.5 * (result.hessian[i * dim + j] + result.hessian[j * dim + i]);
            result.hessian[i * dim + j] = symmetric;
            result.hessian[j * dim + i] = symmetric;
        }
    }
    for (const StackCurveInput& input : curves) {
        if (input.factor) {
            appendFactorMetadata(input, referenceDate, result.points);
        } else if (input.xccy) {
            appendXccyQuoteMetadata(input, referenceDate, result.points);
        } else {
            appendStackQuoteMetadata(input, referenceDate, result.points);
        }
    }
    for (std::size_t i = 0; i < dim; ++i) {
        result.points[i].delta = result.hessian[i * dim + i];
    }
    result.validate();
    return result;
}

/// Depth-1 convenience overload: the root plus additive spread children whose
/// parents must be the root curve. Builds the depth-1 input list like the
/// legacy `stackQuoteRisk` and delegates to the general engine.
inline StackQuoteGamma
stackQuoteGamma(const DiscountCurve<double>& root, const std::vector<CurvePillar>& rootPillars,
                const std::vector<double>& dVdRoot, const std::vector<double>& HZeta,
                const datetime::Date& referenceDate, const std::vector<StackChildInput>& children) {
    return stackQuoteGamma(makeDepth1Inputs(root, rootPillars, dVdRoot, children), HZeta,
                           referenceDate);
}

/// Maturity-tag ladder across all curves of the stack (sums quote deltas by
/// maturity tag for every entry).
inline std::vector<RiskBucket> stackYearLadder(const std::vector<StackRiskEntry>& entries) {
    std::vector<RiskBucket> buckets;
    for (const StackRiskEntry& entry : entries) {
        entry.validate();
        for (const QuotePoint& point : entry.points) {
            bool merged = false;
            for (RiskBucket& bucket : buckets) {
                if (bucket.label == point.bucket) {
                    bucket.delta += point.delta;
                    merged = true;
                    break;
                }
            }
            if (!merged) {
                buckets.push_back(RiskBucket{point.bucket, point.delta});
            }
        }
    }
    return buckets;
}

/// Number of solved nodes of a forecast provider (`SpreadCurve` exposes its
/// spread nodes, `DiscountCurve` its zero nodes).
inline std::size_t forecastNodeCount(const XccyForecastCurve auto& forecast) {
    if constexpr (requires { forecast.spreadNodes(); }) {
        return forecast.spreadNodes().size() - 1;
    } else {
        return forecast.size() - 1;
    }
}

/// Zero-clock day counter of a forecast provider (spread nodes share the
/// parent's clock).
inline const datetime::DayCounter& forecastNodeDayCounter(const XccyForecastCurve auto& forecast) {
    if constexpr (requires { forecast.zeroDayCounter(); }) {
        return forecast.zeroDayCounter();
    } else {
        return forecast.spreadNodes().zeroDayCounter();
    }
}

/// Calendar date of a native forecast node time. The curves store only times on
/// their zero clock, so the integer day denominator of the common Actual
/// conventions is inverted exactly; for other conventions the round-tripped
/// ACT/365F offset is used, which only affects tag rounding below one year.
inline datetime::Date forecastNodeDate(const datetime::DayCounter& zeroDayCounter,
                                       const datetime::Date& referenceDate, double t) {
    double daysPerYear = 365.0;
    switch (zeroDayCounter.convention()) {
        case datetime::DayCount::Actual360:
            daysPerYear = 360.0;
            break;
        case datetime::DayCount::Actual364:
            daysPerYear = 364.0;
            break;
        case datetime::DayCount::Actual365Fixed:
            daysPerYear = 365.0;
            break;
        case datetime::DayCount::Actual366:
            daysPerYear = 366.0;
            break;
        default:
            break;
    }
    return referenceDate.plusDays(static_cast<std::int32_t>(std::lround(t * daysPerYear)));
}

/// Assemble F (xccy quote × foreign nodes), C (× root nodes), G (× foreign
/// forecast nodes) and H (× domestic forecast nodes).
inline void assembleXccyJacobianFull(const DiscountCurve<double>& foreignDiscount,
                                     const XccyForecastCurve auto& foreignForecast,
                                     const DiscountCurve<double>& domesticDiscount,
                                     const XccyForecastCurve auto& domesticForecast,
                                     const std::vector<XccyPillar>& pillars,
                                     const datetime::Date& referenceDate, std::vector<double>& f,
                                     std::vector<double>& c, std::vector<double>& g,
                                     std::vector<double>& h) {
    const std::size_t n = pillars.size();
    if (foreignDiscount.size() != n + 1) {
        throw std::invalid_argument(
            "assembleXccyJacobianFull: foreign discount nodes must match the pillar count");
    }
    const std::size_t m = domesticDiscount.size() - 1;
    const std::size_t p = forecastNodeCount(foreignForecast);
    const std::size_t q = forecastNodeCount(domesticForecast);
    f.assign(n * n, 0.0);
    c.assign(n * m, 0.0);
    g.assign(n * p, 0.0);
    h.assign(n * q, 0.0);
    std::vector<double> fRow;
    std::vector<double> cRow;
    std::vector<double> gRow;
    std::vector<double> hRow;
    for (std::size_t j = 0; j < n; ++j) {
        xccySwapJacobianRows(foreignDiscount, foreignForecast, domesticDiscount, domesticForecast,
                             pillars[j], referenceDate, domesticDiscount.zeroDayCounter(), fRow,
                             cRow, gRow, hRow);
        for (std::size_t i = 0; i < n; ++i) {
            f[j * n + i] = fRow[i];
        }
        for (std::size_t i = 0; i < m; ++i) {
            c[j * m + i] = cRow[i];
        }
        for (std::size_t i = 0; i < p; ++i) {
            g[j * p + i] = gRow[i];
        }
        for (std::size_t i = 0; i < q; ++i) {
            h[j * q + i] = hRow[i];
        }
    }
}

/// F/C-only convenience wrapper.
inline void assembleXccyJacobian(const DiscountCurve<double>& foreignDiscount,
                                 const XccyForecastCurve auto& foreignForecast,
                                 const DiscountCurve<double>& domesticDiscount,
                                 const XccyForecastCurve auto& domesticForecast,
                                 const std::vector<XccyPillar>& pillars,
                                 const datetime::Date& referenceDate, std::vector<double>& f,
                                 std::vector<double>& c) {
    std::vector<double> g;
    std::vector<double> h;
    assembleXccyJacobianFull(foreignDiscount, foreignForecast, domesticDiscount, domesticForecast,
                             pillars, referenceDate, f, c, g, h);
}

/// One cross-currency child over the domestic root: `dVdForeignZeros` is the
/// pricing risk over the foreign curve nodes (node 0 dropped). Forecast curves
/// are referenced, so several children can share one forecast object and its
/// factor block.
template <XccyForecastCurve ForeignForecastT, XccyForecastCurve DomesticForecastT>
struct XccyChildInput {
    CurveRole role = CurveRole::XccyBasis; ///< Cross-currency basis child quote role
    /// Roles of the forecast-node factor rows (`XccyFwd`/`XccyDom`): the
    /// foreign and domestic forecast curves a child corrects.
    CurveRole foreignForecastRole = CurveRole::Forecast;
    CurveRole domesticForecastRole = CurveRole::Forecast;
    const DiscountCurve<double>* foreignDiscount = nullptr;
    std::vector<XccyPillar> pillars;
    std::vector<double> dVdForeignZeros;
    const ForeignForecastT* foreignForecast = nullptr;
    const DomesticForecastT* domesticForecast = nullptr;
    /// Optional portfolio sensitivities over the forecast nodes (node 0 = 0);
    /// the emitted entries add the residual/hedge correction on top.
    std::optional<std::vector<double>> dVdForeignForecast;
    std::optional<std::vector<double>> dVdDomesticForecast;
};

/// Stack risk with any number of (possibly heterogeneous) cross-currency
/// children over the domestic root, implemented over the general engine. Each
/// child contributes its `Xccy` quote block bound to the foreign forecast,
/// domestic forecast and root blocks by view identity, and the two forecast
/// curves enter as shared factor blocks. One dense solve covers the root,
/// every child and every forecast factor, so no forecast-node sensitivity is
/// dropped.
///
/// Children that share a forecast object keep the shared factor block but
/// still emit one `XccyFwd`/`XccyDom` entry each, with the node residual
/// `direct_c - G_c^T x_c` of that child; the per-child entries sum to the
/// shared block's solved residual. The returned entries are the root entry,
/// then for each child in order its quote entry followed by its forecast
/// factor entries.
template <typename... ChildTypes>
inline std::vector<StackRiskEntry>
stackQuoteRiskXccy(const DiscountCurve<double>& root, const std::vector<CurvePillar>& rootPillars,
                   const std::vector<double>& dVdRoot,
                   const std::vector<std::variant<ChildTypes...>>& children,
                   const datetime::Date& referenceDate) {
    if (dVdRoot.size() != root.size()) {
        throw std::invalid_argument("stackQuoteRiskXccy: root sensitivity size mismatch");
    }
    if (rootPillars.size() + 1 != root.size()) {
        throw std::invalid_argument("stackQuoteRiskXccy: root pillars must match the root nodes");
    }
    const StackCurveView::Ptr rootView = StackCurveView::make(root);
    std::vector<StackCurveInput> inputs;
    inputs.reserve(1 + 3 * children.size());
    StackCurveInput rootInput;
    rootInput.curve = rootView;
    rootInput.role = CurveRole::Discount;
    rootInput.discountPillars = rootPillars;
    rootInput.dVdNodes = dVdRoot;
    inputs.push_back(std::move(rootInput));
    std::unordered_map<const void*, std::size_t> factorBlocks;
    const auto addFactor = [&](const StackCurveView::Ptr& view, CurveRole role,
                               std::string_view labelPrefix,
                               const datetime::DayCounter& nodeDayCounter,
                               const std::vector<double>* direct) -> std::pair<std::size_t, bool> {
        if (view->size() <= 1) {
            return {0, false};
        }
        const auto found = factorBlocks.find(view->identity());
        if (found != factorBlocks.end()) {
            if (direct != nullptr) {
                std::vector<double>& target = inputs[found->second].dVdNodes;
                if (direct->size() != target.size()) {
                    throw std::invalid_argument(
                        "stackQuoteRiskXccy: forecast sensitivity size mismatch");
                }
                for (std::size_t i = 0; i < target.size(); ++i) {
                    target[i] += (*direct)[i];
                }
            }
            return {found->second, true};
        }
        StackCurveInput input;
        input.curve = view;
        input.role = role;
        input.factor = StackFactorInput{std::string(labelPrefix), nodeDayCounter};
        input.dVdNodes.assign(view->size(), 0.0);
        if (direct != nullptr) {
            if (direct->size() != view->size()) {
                throw std::invalid_argument(
                    "stackQuoteRiskXccy: forecast sensitivity size mismatch");
            }
            input.dVdNodes = *direct;
        }
        inputs.push_back(std::move(input));
        factorBlocks.emplace(view->identity(), inputs.size() - 1);
        return {inputs.size() - 1, true};
    };
    struct XccyChildSplit {
        std::size_t childInput = 0;
        const StackCurveView* foreignDiscount = nullptr;
        const StackCurveView* foreignForecast = nullptr;
        const StackCurveView* domesticForecast = nullptr;
        const StackCurveView* domesticDiscount = nullptr;
        std::size_t foreignFactor = 0;
        std::size_t domesticFactor = 0;
        bool hasForeignFactor = false;
        bool hasDomesticFactor = false;
        CurveRole foreignRole = CurveRole::Forecast;
        CurveRole domesticRole = CurveRole::Forecast;
        std::vector<double> foreignDirect;
        std::vector<double> domesticDirect;
        std::vector<XccyPillar> pillars;
    };
    std::vector<XccyChildSplit> splits;
    splits.reserve(children.size());
    const auto processChild = [&](const auto& child) {
        if (child.foreignDiscount == nullptr || child.foreignForecast == nullptr ||
            child.domesticForecast == nullptr ||
            child.dVdForeignZeros.size() != child.foreignDiscount->size() ||
            child.pillars.size() + 1 != child.foreignDiscount->size()) {
            throw std::invalid_argument("stackQuoteRiskXccy: malformed child input");
        }
        const StackCurveView::Ptr foreignForecastView =
            StackCurveView::make(*child.foreignForecast);
        const StackCurveView::Ptr domesticForecastView =
            StackCurveView::make(*child.domesticForecast);
        XccyChildSplit split;
        split.foreignForecast = foreignForecastView.get();
        split.domesticForecast = domesticForecastView.get();
        split.domesticDiscount = rootView.get();
        split.foreignRole = child.foreignForecastRole;
        split.domesticRole = child.domesticForecastRole;
        split.pillars = child.pillars;
        split.foreignDirect.assign(foreignForecastView->size(), 0.0);
        split.domesticDirect.assign(domesticForecastView->size(), 0.0);
        StackCurveInput childInput;
        childInput.curve = StackCurveView::make(*child.foreignDiscount);
        split.foreignDiscount = childInput.curve.get();
        childInput.role = child.role;
        childInput.xccyPillars = child.pillars;
        childInput.xccy = XccyRowInput{foreignForecastView, domesticForecastView, rootView};
        childInput.dVdNodes = child.dVdForeignZeros;
        inputs.push_back(std::move(childInput));
        split.childInput = inputs.size() - 1;
        const auto foreignFactor =
            addFactor(foreignForecastView, child.foreignForecastRole, "XccyFwd",
                      forecastNodeDayCounter(*child.foreignForecast),
                      child.dVdForeignForecast ? &*child.dVdForeignForecast : nullptr);
        split.foreignFactor = foreignFactor.first;
        split.hasForeignFactor = foreignFactor.second;
        if (child.dVdForeignForecast) {
            if (child.dVdForeignForecast->size() != foreignForecastView->size()) {
                throw std::invalid_argument(
                    "stackQuoteRiskXccy: forecast sensitivity size mismatch");
            }
            split.foreignDirect = *child.dVdForeignForecast;
        }
        const auto domesticFactor =
            addFactor(domesticForecastView, child.domesticForecastRole, "XccyDom",
                      forecastNodeDayCounter(*child.domesticForecast),
                      child.dVdDomesticForecast ? &*child.dVdDomesticForecast : nullptr);
        split.domesticFactor = domesticFactor.first;
        split.hasDomesticFactor = domesticFactor.second;
        if (child.dVdDomesticForecast) {
            if (child.dVdDomesticForecast->size() != domesticForecastView->size()) {
                throw std::invalid_argument(
                    "stackQuoteRiskXccy: forecast sensitivity size mismatch");
            }
            split.domesticDirect = *child.dVdDomesticForecast;
        }
        splits.push_back(std::move(split));
    };
    for (const auto& variantChild : children) {
        std::visit(processChild, variantChild);
    }
    const std::vector<StackRiskEntry> entries = stackQuoteRisk(inputs, referenceDate);
    // Per-child split of a shared forecast factor: the shared solve gives the
    // child quote delta x_c, and the child's own analytic rows over the factor
    // block give `direct_c - G_c^T x_c`. The children's entries add back to the
    // shared block residual.
    const auto splitFactorEntry = [&](const StackRiskEntry& sharedEntry,
                                      const StackRiskEntry& childEntry, const XccyChildSplit& split,
                                      const StackCurveView& forecast,
                                      const std::vector<double>& direct, CurveRole role) {
        StackRiskEntry entry = sharedEntry;
        entry.role = role;
        for (QuotePoint& point : entry.points) {
            point.role = role;
        }
        std::vector<double> gtx(forecast.size(), 0.0);
        for (std::size_t j = 0; j < split.pillars.size(); ++j) {
            const std::vector<XccyRowBlock> blocks = xccySwapJacobianRowsView(
                *split.foreignDiscount, *split.foreignForecast, *split.domesticDiscount,
                *split.domesticForecast, split.pillars[j], referenceDate,
                split.domesticDiscount->zeroDayCounter());
            for (const XccyRowBlock& block : blocks) {
                if (block.curve->identity() != forecast.identity()) {
                    continue;
                }
                for (std::size_t i = 0; i < block.row.size(); ++i) {
                    gtx[i + 1] += block.row[i] * childEntry.points[j].delta;
                }
            }
        }
        for (std::size_t k = 0; k < entry.points.size(); ++k) {
            entry.points[k].delta = direct[k + 1] - gtx[k + 1];
        }
        return entry;
    };
    std::vector<StackRiskEntry> result;
    result.reserve(1 + 3 * splits.size());
    result.push_back(entries[0]);
    for (const XccyChildSplit& split : splits) {
        const StackRiskEntry& childEntry = entries[split.childInput];
        result.push_back(childEntry);
        if (split.hasForeignFactor) {
            result.push_back(splitFactorEntry(entries[split.foreignFactor], childEntry, split,
                                              *split.foreignForecast, split.foreignDirect,
                                              split.foreignRole));
        }
        if (split.hasDomesticFactor) {
            result.push_back(splitFactorEntry(entries[split.domesticFactor], childEntry, split,
                                              *split.domesticForecast, split.domesticDirect,
                                              split.domesticRole));
        }
    }
    return result;
}

} // namespace quantape::markets
