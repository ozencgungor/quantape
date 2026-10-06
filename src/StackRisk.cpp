#include "quantape/markets/Curves/StackRisk.h"

#include "quantape/markets/Curves/CurveRisk.h"
#include "quantape/markets/Curves/StackRiskRows.h"
#include "quantape/math/LinearAlgebra/DenseSolve.h"

#include <algorithm>
#include <cmath>
#include <string>
#include <string_view>
#include <unordered_map>
#include <utility>
#include <vector>

namespace quantape::markets {
std::size_t stackFactorNodeCount(const StackCurveInput& input) {
    if (!input.factor) {
        throw std::invalid_argument("stackFactorNodeCount: input is not a factor block");
    }
    const std::size_t native = input.curve->size() - 1;
    const std::size_t count = input.factor->nodeCount == 0 ? native : input.factor->nodeCount;
    if (count == 0 || count > native) {
        throw std::invalid_argument("stackFactorNodeCount: factor node count out of range");
    }
    return count;
}

StackCurveView::Ptr makeFxFactorAnchorView() {
    return StackCurveView::makeOwned(
        DiscountCurve<double>(std::vector<double>{0.0, 1.0}, std::vector<double>{0.0, 0.0},
                              InterpolationSpace::LogDiscount, InterpolationScheme::Linear));
}

StackCurveInput makeFxSpotFactorInput(StackCurveView::Ptr anchor, double dVdSpot,
                                      std::string pair) {
    if (anchor == nullptr || anchor->size() < 2) {
        throw std::invalid_argument("makeFxSpotFactorInput: anchor must have two nodes");
    }
    StackCurveInput input;
    input.curve = std::move(anchor);
    input.role = CurveRole::FxSpot;
    StackFactorInput factor;
    factor.labelPrefix = "FxSpot";
    factor.nodeDayCounter = input.curve->zeroDayCounter();
    factor.nodeCount = 1;
    factor.fixedLabel = "FxSpot " + std::move(pair);
    factor.fixedBucket = "FxSpot";
    input.factor = std::move(factor);
    input.dVdNodes.assign(input.curve->size(), 0.0);
    input.dVdNodes[1] = dVdSpot;
    return input;
}

void validateQuotePoints(const std::vector<QuotePoint>& points,
                         std::optional<std::size_t> expected) {
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

std::vector<RiskBucket> stackRoleBuckets(const std::vector<StackRiskEntry>& entries) {
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

std::vector<RiskBucket> addTurnRiskBuckets(std::vector<RiskBucket> buckets,
                                           const std::vector<TurnRiskEntry>& turns,
                                           CurveRole role) {
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

std::vector<RiskBucket> addTurnRiskBuckets(const std::vector<StackRiskEntry>& entries,
                                           const std::vector<TurnRiskEntry>& turns,
                                           CurveRole role) {
    return addTurnRiskBuckets(stackRoleBuckets(entries), turns, role);
}

std::vector<StackCurveInput> makeDepth1Inputs(const DiscountCurve<double>& root,
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

void appendStackQuoteMetadata(const StackCurveInput& input, const datetime::Date& referenceDate,
                              std::vector<QuotePoint>& points) {
    if (!input.discountPillars.empty()) {
        points.reserve(points.size() + input.discountPillars.size());
        for (const CurvePillar& pillar : input.discountPillars) {
            const datetime::Date maturity = pillarRiskMaturity(pillar);
            const double t =
                datetime::yearFraction(referenceDate, maturity, input.curve->zeroDayCounter());
            std::string tag = riskMaturityTag(maturity, t);
            const std::string_view kind = pillarKindName(pillar.kind);
            QuotePoint point;
            point.bucket = std::move(tag);
            point.label.reserve(kind.size() + 1 + point.bucket.size());
            point.label.assign(kind);
            point.label.push_back(' ');
            point.label += point.bucket;
            point.year = static_cast<int>(std::lround(t));
            point.role = input.role;
            points.push_back(std::move(point));
        }
        return;
    }
    points.reserve(points.size() + input.forecastPillars.size());
    for (const ForecastPillar& pillar : input.forecastPillars) {
        const datetime::Date maturity = forecastPillarRiskMaturity(pillar);
        const double t =
            datetime::yearFraction(referenceDate, maturity, input.curve->zeroDayCounter());
        const int year = static_cast<int>(std::lround(t));
        const CurveRole role = pillar.turnPillar ? CurveRole::TurnOverlay : input.role;
        QuotePoint point;
        if (pillar.turnPillar) {
            const datetime::Date start = pillar.start.serial() != 0 ? pillar.start : referenceDate;
            point.bucket = "Turn " + start.toIso();
            point.label = point.bucket;
        } else {
            std::string tag = riskMaturityTag(maturity, t);
            const std::string_view kind = forecastPillarKindName(pillar.kind);
            point.bucket = std::move(tag);
            point.label.reserve(kind.size() + 1 + point.bucket.size());
            point.label.assign(kind);
            point.label.push_back(' ');
            point.label += point.bucket;
        }
        point.year = year;
        point.role = role;
        points.push_back(std::move(point));
    }
}

void appendXccyQuoteMetadata(const StackCurveInput& input, const datetime::Date& referenceDate,
                             std::vector<QuotePoint>& points) {
    const datetime::DayCounter& zeroDayCounter = input.xccy->domesticDiscount->zeroDayCounter();
    points.reserve(points.size() + input.xccyPillars.size());
    for (const XccyPillar& pillar : input.xccyPillars) {
        const datetime::Date maturity =
            pillar.foreignCalendar.adjust(pillar.maturity, pillar.foreignBusinessDayConvention);
        const double t = datetime::yearFraction(referenceDate, maturity, zeroDayCounter);
        std::string tag = riskMaturityTag(maturity, t);
        QuotePoint point;
        point.bucket = std::move(tag);
        point.label.reserve(5 + point.bucket.size());
        point.label.assign("Xccy ");
        point.label += point.bucket;
        point.year = static_cast<int>(std::lround(t));
        point.role = input.role;
        points.push_back(std::move(point));
    }
}

void appendFactorMetadata(const StackCurveInput& input, const datetime::Date& referenceDate,
                          std::vector<QuotePoint>& points) {
    const std::size_t nodes = stackFactorNodeCount(input);
    points.reserve(points.size() + nodes);
    if (!input.factor->fixedLabel.empty()) {
        const std::string& bucket = input.factor->fixedBucket.empty() ? input.factor->fixedLabel
                                                                      : input.factor->fixedBucket;
        for (std::size_t i = 0; i < nodes; ++i) {
            QuotePoint point;
            point.bucket = bucket;
            point.label = input.factor->fixedLabel;
            point.year = 0;
            point.role = input.role;
            points.push_back(std::move(point));
        }
        return;
    }
    const std::vector<double>& times = input.curve->times();
    for (std::size_t i = 1; i <= nodes; ++i) {
        const datetime::Date nodeDate =
            forecastNodeDate(input.factor->nodeDayCounter, referenceDate, times[i]);
        std::string bucket = riskMaturityTag(nodeDate, times[i]);
        QuotePoint point;
        point.bucket = std::move(bucket);
        point.label.reserve(input.factor->labelPrefix.size() + 1 + point.bucket.size());
        point.label.assign(input.factor->labelPrefix);
        point.label.push_back(' ');
        point.label += point.bucket;
        point.year = static_cast<int>(std::lround(times[i]));
        point.role = input.role;
        points.push_back(std::move(point));
    }
}

namespace {

/// Assemble `F = d r / d zeta` into `system`. With `changedRows == nullptr`
/// every row is rebuilt from scratch; with a mask only the rows of inputs
/// marked changed are zeroed and rebuilt, while the rest keep the values the
/// caller copied in from the base system. A curve bump can only move the rows
/// of inputs whose own curve, discount view or cross-currency references were
/// rebuilt, so the partial pass is exact and avoids re-evaluating every
/// instrument row for each finite-difference bump.
void assembleStackQuoteSystemInto(const std::vector<StackCurveInput>& curves,
                                  const datetime::Date& referenceDate, StackQuoteSystem& system,
                                  const std::vector<bool>* changedRows = nullptr) {
    if (curves.empty()) {
        throw std::invalid_argument("assembleStackQuoteSystem: no curves");
    }
    static thread_local std::vector<std::size_t> rowCounts;
    rowCounts.resize(curves.size());
    system.dim = 0;
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
        const std::size_t factorNodes = hasFactor ? stackFactorNodeCount(input) : 0;
        const std::size_t rows = hasDiscount   ? input.discountPillars.size()
                                 : hasForecast ? input.forecastPillars.size()
                                 : hasXccy     ? input.xccyPillars.size()
                                               : factorNodes;
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
        } else if (!hasFactor && rows + 1 != input.curve->size()) {
            throw std::invalid_argument(
                "assembleStackQuoteSystem: pillars must match the curve nodes");
        }
        if (hasForecast && input.curve->parentView() == nullptr) {
            throw std::invalid_argument(
                "assembleStackQuoteSystem: forecast curve without a parent");
        }
        rowCounts[k] = rows;
        system.offsets[k] = system.dim;
        system.dim += hasFactor ? factorNodes : input.curve->size() - 1;
    }
    const std::size_t dim = system.dim;
    // Curve matching walks every stack input and fingerprints interpolation
    // weights, so memoize the result per view identity: each ancestor of each
    // instrument row maps to the same block for the whole assembly. The memo
    // and the row scratch are per-thread and persist between assemblies, so
    // repeated finite-difference bumps reuse their capacity.
    static thread_local std::unordered_map<const void*, std::size_t> matchedBlocks;
    matchedBlocks.clear();
    static thread_local std::vector<double> rowScratch;
    static thread_local std::vector<double> ownRowScratch;
    static thread_local std::vector<double> parentRowScratch;
    static thread_local std::vector<double> discountRowScratch;
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
    if (changedRows == nullptr) {
        system.jacobian.assign(dim * dim, 0.0);
    }
    std::vector<double>& scratch = rowScratch;
    std::size_t row = 0;
    for (std::size_t k = 0; k < curves.size(); ++k) {
        const StackCurveInput& input = curves[k];
        if (changedRows != nullptr && !(*changedRows)[k]) {
            row += rowCounts[k];
            continue;
        }
        if (changedRows != nullptr) {
            for (std::size_t i = 0; i < rowCounts[k]; ++i) {
                std::fill_n(system.jacobian.begin() + static_cast<std::ptrdiff_t>((row + i) * dim),
                            static_cast<std::ptrdiff_t>(dim), 0.0);
            }
        }
        if (input.factor) {
            for (std::size_t i = 0; i < rowCounts[k]; ++i) {
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
        std::vector<double>& ownRow = ownRowScratch;
        std::vector<double>& parentRow = parentRowScratch;
        std::vector<double>& discountRow = discountRowScratch;
        for (const ForecastPillar& pillar : input.forecastPillars) {
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
}

/// Rebuild every curve view after bumping one native node into caller-owned
/// scratch: `result` is copy-assigned from `curves` (reusing its nested
/// capacities when it is warm), `views` and `rebuilt` are reused maps/vectors.
void bumpStackInputsInto(const std::vector<StackCurveInput>& curves, std::size_t bumpedCurve,
                         std::size_t node, double delta, std::vector<StackCurveInput>& result,
                         std::vector<StackCurveView::Ptr>& views,
                         std::unordered_map<const void*, StackCurveView::Ptr>& rebuilt) {
    result = curves;
    views.resize(curves.size());
    for (std::size_t k = 0; k < curves.size(); ++k) {
        views[k] = curves[k].curve;
    }
    rebuilt.clear();
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
}

/// Mark the inputs whose rows the bump can have moved: the row of an input
/// reads its own curve, its discount view and (for cross-currency rows) the
/// referenced forecast/discount blocks. Factor rows are identity rows and
/// never move with a curve bump.
void markChangedInputs(const std::vector<StackCurveInput>& base,
                       const std::vector<StackCurveInput>& bumped, std::vector<bool>& changed) {
    for (std::size_t k = 0; k < base.size(); ++k) {
        if (base[k].factor) {
            changed[k] = false;
            continue;
        }
        bool moved = bumped[k].curve != base[k].curve || bumped[k].discount != base[k].discount;
        if (!moved && base[k].xccy && bumped[k].xccy) {
            moved = bumped[k].xccy->foreignForecast != base[k].xccy->foreignForecast ||
                    bumped[k].xccy->domesticForecast != base[k].xccy->domesticForecast ||
                    bumped[k].xccy->domesticDiscount != base[k].xccy->domesticDiscount;
        }
        changed[k] = moved;
    }
}

} // namespace

StackQuoteSystem assembleStackQuoteSystem(const std::vector<StackCurveInput>& curves,
                                          const datetime::Date& referenceDate) {
    StackQuoteSystem system;
    assembleStackQuoteSystemInto(curves, referenceDate, system);
    return system;
}

std::vector<StackCurveInput> bumpStackInputs(const std::vector<StackCurveInput>& curves,
                                             std::size_t bumpedCurve, std::size_t node,
                                             double delta) {
    std::vector<StackCurveInput> result;
    std::vector<StackCurveView::Ptr> views;
    std::unordered_map<const void*, StackCurveView::Ptr> rebuilt;
    bumpStackInputsInto(curves, bumpedCurve, node, delta, result, views, rebuilt);
    return result;
}

std::vector<StackRiskEntry> stackQuoteRisk(const std::vector<StackCurveInput>& curves,
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
        const std::size_t nodes =
            curves[k].factor ? stackFactorNodeCount(curves[k]) : curves[k].curve->size() - 1;
        for (std::size_t i = 1; i <= nodes; ++i) {
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

std::vector<StackRiskEntry> stackQuoteRisk(const DiscountCurve<double>& root,
                                           const std::vector<CurvePillar>& rootPillars,
                                           const std::vector<double>& dVdRoot,
                                           const std::vector<StackChildInput>& children,
                                           const datetime::Date& referenceDate) {
    return stackQuoteRisk(makeDepth1Inputs(root, rootPillars, dVdRoot, children), referenceDate);
}

StackQuoteGamma stackQuoteGamma(const std::vector<StackCurveInput>& curves,
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
    std::vector<double> solution;
    for (std::size_t column = 0; column < dim; ++column) {
        std::fill(unit.begin(), unit.end(), 0.0);
        unit[column] = 1.0;
        detail::solveDenseLuInto(factors, unit, solution);
        for (std::size_t k = 0; k < dim; ++k) {
            jacobian[k * dim + column] = solution[k];
        }
    }
    // Node gradient and quote-space delta x = J^T g.
    std::vector<double> g(dim, 0.0);
    for (std::size_t k = 0; k < curves.size(); ++k) {
        const std::size_t nodes =
            curves[k].factor ? stackFactorNodeCount(curves[k]) : curves[k].curve->size() - 1;
        for (std::size_t i = 1; i <= nodes; ++i) {
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
    // Only the rows of inputs the bump rebuilt are re-assembled: the plus and
    // minus systems start from the base Jacobian and the partial pass zeroes
    // and rebuilds exactly the moved rows, so the curvature of every other row
    // is exactly zero as before.
    const double step = 1e-6;
    std::vector<double> weighted(dim * dim, 0.0);
    StackQuoteSystem plusSystem = system;
    StackQuoteSystem minusSystem = system;
    std::vector<StackCurveInput> bumpedInputs;
    std::vector<StackCurveView::Ptr> bumpedViews;
    std::unordered_map<const void*, StackCurveView::Ptr> rebuiltViews;
    std::vector<bool> changedRows(curves.size(), false);
    for (std::size_t j = 0; j < curves.size(); ++j) {
        const std::size_t nodes =
            curves[j].factor ? stackFactorNodeCount(curves[j]) : curves[j].curve->size() - 1;
        for (std::size_t i = 1; i <= nodes; ++i) {
            const std::size_t bump = offset[j] + i - 1;
            plusSystem.jacobian = system.jacobian;
            bumpStackInputsInto(curves, j, i, step, bumpedInputs, bumpedViews, rebuiltViews);
            markChangedInputs(curves, bumpedInputs, changedRows);
            assembleStackQuoteSystemInto(bumpedInputs, referenceDate, plusSystem, &changedRows);
            minusSystem.jacobian = system.jacobian;
            bumpStackInputsInto(curves, j, i, -step, bumpedInputs, bumpedViews, rebuiltViews);
            markChangedInputs(curves, bumpedInputs, changedRows);
            assembleStackQuoteSystemInto(bumpedInputs, referenceDate, minusSystem, &changedRows);
            // Row-major pass: for each model row accumulate the weighted
            // curvature into every bump column, keeping the plus/minus reads
            // contiguous. The per-column sum still runs over rows in the same
            // order as before, so the result is unchanged.
            for (std::size_t r = 0; r < dim; ++r) {
                const double weight = x[r];
                const double* plusRow = &plusSystem.jacobian[r * dim];
                const double* minusRow = &minusSystem.jacobian[r * dim];
                for (std::size_t column = 0; column < dim; ++column) {
                    const double curvature = (plusRow[column] - minusRow[column]) / (2.0 * step);
                    weighted[column * dim + bump] += weight * curvature;
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

StackQuoteGamma
stackQuoteGamma(const DiscountCurve<double>& root, const std::vector<CurvePillar>& rootPillars,
                const std::vector<double>& dVdRoot, const std::vector<double>& HZeta,
                const datetime::Date& referenceDate, const std::vector<StackChildInput>& children) {
    return stackQuoteGamma(makeDepth1Inputs(root, rootPillars, dVdRoot, children), HZeta,
                           referenceDate);
}

std::vector<RiskBucket> stackYearLadder(const std::vector<StackRiskEntry>& entries) {
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

std::vector<FxRiskPoint> fxRiskTable(const std::vector<StackRiskEntry>& entries,
                                     const StackQuoteGamma* gamma) {
    std::size_t total = 0;
    for (const StackRiskEntry& entry : entries) {
        total += entry.points.size();
    }
    if (gamma != nullptr && gamma->dim != total) {
        throw std::invalid_argument("fxRiskTable: gamma dimension does not match the table");
    }
    std::vector<FxRiskPoint> table;
    std::size_t index = 0;
    for (const StackRiskEntry& entry : entries) {
        for (const QuotePoint& point : entry.points) {
            if (isFxRole(point.role)) {
                FxRiskPoint fx;
                fx.label = point.label;
                fx.role = point.role;
                fx.delta = point.delta;
                fx.gamma = gamma != nullptr ? gamma->at(index, index) : 0.0;
                const std::size_t space = point.label.find(' ');
                const std::size_t pairBegin = space == std::string::npos ? 0 : space + 1;
                const std::size_t pairEnd = point.label.find(' ', pairBegin);
                fx.pair = point.label.substr(pairBegin, pairEnd == std::string::npos
                                                            ? std::string::npos
                                                            : pairEnd - pairBegin);
                table.push_back(std::move(fx));
            }
            ++index;
        }
    }
    return table;
}

datetime::Date forecastNodeDate(const datetime::DayCounter& zeroDayCounter,
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
} // namespace quantape::markets
