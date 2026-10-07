#include "quantape/markets/Curves/XccyRisk.h"

#include "quantape/markets/Curves/CurveRisk.h"
#include "quantape/markets/Curves/XccyBasisBuilder.h"

#include <cstddef>
#include <stdexcept>
#include <utility>
#include <vector>

namespace quantape::markets {
double xccyBasisSpreadView(const StackCurveView& foreignDiscount,
                           const StackCurveView& foreignForecast,
                           const StackCurveView& domesticDiscount,
                           const StackCurveView& domesticForecast, const XccyPillar& pillar,
                           const datetime::Date& referenceDate,
                           const datetime::DayCounter& zeroDayCounter) {
    return pricing::xccyBasisSpread(foreignDiscount, foreignForecast, domesticDiscount,
                                    domesticForecast, pillar, referenceDate, zeroDayCounter);
}

namespace detail {

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
    return pricing::xccySwapJacobianRowsView(foreignDiscount, foreignForecast, domesticDiscount,
                                             domesticForecast, pillar, referenceDate,
                                             zeroDayCounter);
}

std::vector<XccyRowBlock> xccySwapJacobianRowsViewFiniteDifference(
    const StackCurveView& foreignDiscount, const StackCurveView& foreignForecast,
    const StackCurveView& domesticDiscount, const StackCurveView& domesticForecast,
    const XccyPillar& pillar, const datetime::Date& referenceDate,
    const datetime::DayCounter& zeroDayCounter) {
    std::vector<XccyRowBlock> blocks;
    pricing::detail::xccyRowBlock(blocks, foreignDiscount);
    pricing::detail::xccyRegisterForecastChain(blocks, foreignForecast);
    pricing::detail::xccyRowBlock(blocks, domesticDiscount);
    pricing::detail::xccyRegisterForecastChain(blocks, domesticForecast);
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
