#include "quantape/markets/Curves/StackRiskRows.h"

#include "quantape/markets/Curves/CurveRisk.h"
#include "quantape/pricing/Ir.h"

#include <cstddef>
#include <stdexcept>
#include <type_traits>
#include <variant>
#include <vector>

namespace quantape::markets {
namespace {

/// Core of the forecast-pillar dispatcher over already-materialized views, so
/// the assemblers can wrap each curve once and walk every pillar without
/// rebuilding the same three views per row.
void forecastPillarJacobianRowsWithViews(
    const StackCurveView& childView, const StackCurveView& parentView,
    const StackCurveView& discountView, const ForecastPillar& pillar,
    const datetime::Date& referenceDate, const datetime::DayCounter& zeroDayCounter,
    std::vector<double>& fRow, std::vector<double>& parentRow, std::vector<double>& discountRow) {
    const ForecastInstrument instrument = detail::riskInstrument(pillar);
    std::visit(
        [&](const auto& value) {
            using InstrumentT = std::decay_t<decltype(value)>;
            if constexpr (std::is_same_v<InstrumentT, instruments::Irs>) {
                pricing::jacobianRow(value, childView, parentView, discountView, referenceDate,
                                     zeroDayCounter, fRow, parentRow, discountRow);
            } else if constexpr (std::is_same_v<InstrumentT, instruments::BasisSwap>) {
                pricing::jacobianRow(value, childView, parentView, parentView, referenceDate, fRow,
                                     parentRow, discountRow);
                discountRow.assign(discountView.size() - 1, 0.0);
            } else {
                pricing::jacobianRow(value, childView, parentView, discountView, referenceDate,
                                     fRow, parentRow, discountRow);
            }
        },
        instrument.instrument());
}

} // namespace

void forecastPillarJacobianRows(const SpreadCurve<double>& child, const ForecastPillar& pillar,
                                const datetime::Date& referenceDate,
                                const DiscountCurve<double>* discountCurve,
                                std::vector<double>& fRow, std::vector<double>& parentRow,
                                std::vector<double>& discountRow) {
    const DiscountCurve<double>& forecastParent = child.parent();
    const DiscountCurve<double>& discount =
        discountCurve != nullptr ? *discountCurve : forecastParent;
    const StackCurveView::Ptr childView = StackCurveView::make(child);
    const StackCurveView::Ptr parentView = StackCurveView::make(forecastParent);
    const StackCurveView::Ptr discountView = StackCurveView::make(discount);
    forecastPillarJacobianRowsWithViews(*childView, *parentView, *discountView, pillar,
                                        referenceDate, child.zeroDayCounter(), fRow, parentRow,
                                        discountRow);
}

void assembleBasisJacobian(const SpreadCurve<double>& child,
                           const std::vector<BasisPillar>& pillars,
                           const datetime::Date& referenceDate, std::vector<double>& f,
                           std::vector<double>& c) {
    if (child.size() != pillars.size() + 1) {
        throw std::invalid_argument(
            "assembleBasisJacobian: child nodes must match the pillar count");
    }
    const std::size_t mChild = pillars.size();
    const std::size_t mParent = child.parent().size() - 1;
    f.assign(mChild * mChild, 0.0);
    c.assign(mChild * mParent, 0.0);
    const StackCurveView::Ptr childView = StackCurveView::make(child);
    const StackCurveView::Ptr parentView = StackCurveView::make(child.parent());
    std::vector<double> fRow;
    std::vector<double> cRow;
    std::vector<double> discountRow;
    for (std::size_t j = 0; j < mChild; ++j) {
        pricing::jacobianRow(pillars[j], *childView, *parentView, *parentView, referenceDate, fRow,
                             cRow, discountRow);
        for (std::size_t i = 0; i < mParent; ++i) {
            cRow[i] += discountRow[i];
        }
        for (std::size_t i = 0; i < mChild; ++i) {
            f[j * mChild + i] = fRow[i];
        }
        for (std::size_t i = 0; i < mParent; ++i) {
            c[j * mParent + i] = cRow[i];
        }
    }
}

void assembleForecastJacobian(const SpreadCurve<double>& child,
                              const std::vector<ForecastPillar>& pillars,
                              const datetime::Date& referenceDate,
                              const DiscountCurve<double>* discountCurve, std::vector<double>& f,
                              std::vector<double>& c) {
    if (child.size() != pillars.size() + 1) {
        throw std::invalid_argument(
            "assembleForecastJacobian: child nodes must match the pillar count");
    }
    const std::size_t mChild = pillars.size();
    const DiscountCurve<double>& forecastParent = child.parent();
    const std::size_t mParent = forecastParent.size() - 1;
    const DiscountCurve<double>& discount =
        discountCurve != nullptr ? *discountCurve : forecastParent;
    const bool discountIsParent =
        &discount == &forecastParent || sameCurveValues(discount, forecastParent);
    f.assign(mChild * mChild, 0.0);
    c.assign(mChild * mParent, 0.0);
    // Each row reads the same three curve views, so materialize them once for
    // the whole block; the discount view aliases the parent when the caller
    // passes the same object (the default), avoiding a redundant curve copy.
    const StackCurveView::Ptr childView = StackCurveView::make(child);
    const StackCurveView::Ptr parentView = StackCurveView::make(forecastParent);
    const StackCurveView::Ptr discountView =
        &discount == &forecastParent ? parentView : StackCurveView::make(discount);
    std::vector<double> fRow;
    std::vector<double> parentRow;
    std::vector<double> discountRow;
    for (std::size_t j = 0; j < mChild; ++j) {
        forecastPillarJacobianRowsWithViews(*childView, *parentView, *discountView, pillars[j],
                                            referenceDate, child.zeroDayCounter(), fRow, parentRow,
                                            discountRow);
        for (std::size_t i = 0; i < mChild; ++i) {
            f[j * mChild + i] = fRow[i];
        }
        for (std::size_t i = 0; i < mParent; ++i) {
            c[j * mParent + i] = parentRow[i] + (discountIsParent ? discountRow[i] : 0.0);
        }
    }
}
} // namespace quantape::markets
