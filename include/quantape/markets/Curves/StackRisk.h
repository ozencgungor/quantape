#pragma once

#include "quantape/datetime/Date.h"
#include "quantape/datetime/DayCounter.h"
#include "quantape/markets/Curves/Curve.h"
#include "quantape/markets/Curves/CurveBuilder.h"
#include "quantape/markets/Curves/CurveRiskReport.h"
#include "quantape/markets/Curves/SpreadCurve.h"
#include "quantape/markets/Curves/StackCurveView.h"
#include "quantape/markets/Curves/StackRiskRows.h"
#include "quantape/markets/Curves/TurnOverlay.h"
#include "quantape/markets/Curves/XccyBasisBuilder.h"
#include "quantape/markets/Curves/XccyRisk.h"

#include <cstddef>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
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
/// `nodeCount` pins the solved nodes explicitly: `0` keeps the curve-native
/// count, while a positive value lets a factor publish fewer values than its
/// anchor view exposes (an FX spot block solves one value on a two-node
/// anchor). When `fixedLabel` is set the factor has no dated node grid and
/// every solved node reports under that label and `fixedBucket`.
struct StackFactorInput {
    std::string labelPrefix;
    datetime::DayCounter nodeDayCounter; ///< Clock of the native node grid
    std::size_t nodeCount = 0;           ///< Solved nodes; 0 = curve->size() - 1
    std::string fixedLabel;              ///< Fixed point label (dated grid absent)
    std::string fixedBucket;             ///< Fixed maturity bucket for the fixed label
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

/// Solved node count of a factor block: the explicit `nodeCount` when set,
/// otherwise the anchor view's native count (its size minus the t = 0 node).
std::size_t stackFactorNodeCount(const StackCurveInput& input);

/// Two-node anchor view for a factor with no natural curve grid (an FX spot
/// value). The identity instrument rows ignore the curve values, so the anchor
/// only supplies block identity and scratch storage.
StackCurveView::Ptr makeFxFactorAnchorView();

/// One-node FX spot factor block: a two-node anchor whose single solved node
/// carries `dVdSpot`, reported as `FxSpot <pair>` under the `FxSpot` bucket and
/// the `CurveRole::FxSpot` role. The node vector keeps the engine layout with
/// node 0 unused, i.e. `dVdNodes = {0, dV/dS}`.
StackCurveInput makeFxSpotFactorInput(StackCurveView::Ptr anchor, double dVdSpot, std::string pair);

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
void validateQuotePoints(const std::vector<QuotePoint>& points,
                         std::optional<std::size_t> expected = std::nullopt);

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
std::vector<RiskBucket> stackRoleBuckets(const std::vector<StackRiskEntry>& entries);

/// Overlay turn risk merged into a role-bucket list. Overlay turn amplitudes
/// are exogenous factors outside the quote Jacobian, so `stackRoleBuckets`
/// alone understates the `TurnOverlay` role; every `TurnRiskEntry` delta folds
/// into the `TurnOverlay` role bucket (or `role` when given), keeping role
/// sums complete. Labels stay on the original turn entries.
std::vector<RiskBucket> addTurnRiskBuckets(std::vector<RiskBucket> buckets,
                                           const std::vector<TurnRiskEntry>& turns,
                                           CurveRole role = CurveRole::TurnOverlay);

/// Role buckets of a stack risk table with exogenous overlay turn risk folded
/// in; equivalent to `addTurnRiskBuckets(stackRoleBuckets(entries), turns)`.
std::vector<RiskBucket> addTurnRiskBuckets(const std::vector<StackRiskEntry>& entries,
                                           const std::vector<TurnRiskEntry>& turns,
                                           CurveRole role = CurveRole::TurnOverlay);

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

/// One FX risk point of the side table: the pair, the reporting role, the
/// quote-space spot delta and the diagonal gamma when a Hessian is supplied.
struct FxRiskPoint {
    std::string pair;
    std::string label;
    CurveRole role = CurveRole::FxSpot;
    double delta = 0.0;
    double gamma = 0.0;
};

/// FX greeks side table over the solved risk entries: every `FxSpot`/`FxVol`
/// point of the stack table becomes one entry keyed by pair (parsed from the
/// fixed `FxSpot <pair>` / `FxVol <pair> ...` label). When `gamma` is supplied
/// its matching quote diagonal fills the second derivative.
std::vector<FxRiskPoint> fxRiskTable(const std::vector<StackRiskEntry>& entries,
                                     const StackQuoteGamma* gamma = nullptr);

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
std::vector<StackCurveInput> makeDepth1Inputs(const DiscountCurve<double>& root,
                                              const std::vector<CurvePillar>& rootPillars,
                                              const std::vector<double>& dVdRoot,
                                              const std::vector<StackChildInput>& children);

/// Quote points of one stack input in `stackQuoteRisk` entry order. A direct
/// turn knot keeps its bootstrap column but reports under `TurnOverlay` with a
/// `Turn <date>` label. The caller fills each point's `delta`.
void appendStackQuoteMetadata(const StackCurveInput& input, const datetime::Date& referenceDate,
                              std::vector<QuotePoint>& points);

/// Quote points of a cross-currency child input: one `Xccy <tag>` point per
/// pillar, aged on the domestic discount block's zero clock. The caller fills
/// each point's `delta`.
void appendXccyQuoteMetadata(const StackCurveInput& input, const datetime::Date& referenceDate,
                             std::vector<QuotePoint>& points);

/// Calendar date of a native forecast node time.
datetime::Date forecastNodeDate(const datetime::DayCounter& zeroDayCounter,
                                const datetime::Date& referenceDate, double t);

/// Quote points of a factor block: one `<prefix> <tag>` point per solved node,
/// aged on the factor's own node clock. A factor with a fixed label reports
/// every solved node under that label and `fixedBucket` instead. The caller
/// fills each point's `delta`.
void appendFactorMetadata(const StackCurveInput& input, const datetime::Date& referenceDate,
                          std::vector<QuotePoint>& points);

/// Assemble the stack quote Jacobian `F = d r / d zeta` (row-major `dim x dim`)
/// and the per-curve node block offsets. Every instrument row is assembled
/// analytically over the view-native node coordinates; cross-currency child
/// rows arrive per block and are placed by view identity, and factor blocks
/// contribute identity rows so their solved value is the node-space residual.
StackQuoteSystem assembleStackQuoteSystem(const std::vector<StackCurveInput>& curves,
                                          const datetime::Date& referenceDate);

/// Rebuild every curve view after bumping one native node of the stack. The
/// bumped input receives the node bump; every other input whose parent
/// identity chain reaches a bumped view is rebuilt with the bumped parent,
/// discount views are re-pointed at the bumped curve they reference, and
/// cross-currency block references follow their rebuilt views.
std::vector<StackCurveInput> bumpStackInputs(const std::vector<StackCurveInput>& curves,
                                             std::size_t bumpedCurve, std::size_t node,
                                             double delta);

/// Total quote risk over an arbitrary curve tree. Every instrument row is
/// assembled analytically over the view-native node coordinates, and the full
/// system `F^T x = g` is solved densely, so curve depth and exogenous
/// discounting only change which column block a row term lands in.
std::vector<StackRiskEntry> stackQuoteRisk(const std::vector<StackCurveInput>& curves,
                                           const datetime::Date& referenceDate);

/// Depth-1 convenience overload: the root plus additive spread children whose
/// parents must be the root curve. Implemented over the general engine.
std::vector<StackRiskEntry> stackQuoteRisk(const DiscountCurve<double>& root,
                                           const std::vector<CurvePillar>& rootPillars,
                                           const std::vector<double>& dVdRoot,
                                           const std::vector<StackChildInput>& children,
                                           const datetime::Date& referenceDate);

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
StackQuoteGamma stackQuoteGamma(const std::vector<StackCurveInput>& curves,
                                const std::vector<double>& HZeta,
                                const datetime::Date& referenceDate);

/// Depth-1 convenience overload: the root plus additive spread children whose
/// parents must be the root curve. Builds the depth-1 input list like the
/// legacy `stackQuoteRisk` and delegates to the general engine.
StackQuoteGamma
stackQuoteGamma(const DiscountCurve<double>& root, const std::vector<CurvePillar>& rootPillars,
                const std::vector<double>& dVdRoot, const std::vector<double>& HZeta,
                const datetime::Date& referenceDate, const std::vector<StackChildInput>& children);

/// Maturity-tag ladder across all curves of the stack (sums quote deltas by
/// maturity tag for every entry).
std::vector<RiskBucket> stackYearLadder(const std::vector<StackRiskEntry>& entries);

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
    // The per-row views are identical for every pillar, so wrap each curve
    // once and reuse the views across the whole block instead of rebuilding
    // four views inside every row.
    const StackCurveView::Ptr foreignDiscountView = StackCurveView::make(foreignDiscount);
    const StackCurveView::Ptr foreignForecastView = StackCurveView::make(foreignForecast);
    const StackCurveView::Ptr domesticDiscountView = StackCurveView::make(domesticDiscount);
    const StackCurveView::Ptr domesticForecastView = StackCurveView::make(domesticForecast);
    std::vector<double> fRow;
    std::vector<double> cRow;
    std::vector<double> gRow;
    std::vector<double> hRow;
    for (std::size_t j = 0; j < n; ++j) {
        const std::vector<XccyRowBlock> blocks = xccySwapJacobianRowsView(
            *foreignDiscountView, *foreignForecastView, *domesticDiscountView,
            *domesticForecastView, pillars[j], referenceDate, domesticDiscount.zeroDayCounter());
        detail::mapXccyRowBlocks(blocks, *foreignDiscountView, *domesticDiscountView,
                                 *foreignForecastView, *domesticForecastView, fRow, cRow, gRow,
                                 hRow);
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
        StackFactorInput factor;
        factor.labelPrefix = std::string(labelPrefix);
        factor.nodeDayCounter = nodeDayCounter;
        input.factor = std::move(factor);
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
