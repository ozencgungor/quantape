#include "quantape/markets/Curves/StackCurveView.h"

namespace quantape::markets {
bool sameCurveValues(const DiscountCurve<double>& left, const DiscountCurve<double>& right) {
    return detail::sameCurveValuesCore(left, &left, right, &right);
}

bool sameCurveView(const StackCurveView& left, const StackCurveView& right) {
    return detail::sameCurveValuesCore(left, left.identity(), right, right.identity());
}
} // namespace quantape::markets
