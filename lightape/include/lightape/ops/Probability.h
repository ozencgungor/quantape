#ifndef LIGHTAPE_OPS_PROBABILITY_H
#define LIGHTAPE_OPS_PROBABILITY_H

#include <cmath>

#include "lightape/base/ScalarConstant.h"

namespace lightape {

/// 1/sqrt(2*pi) and its inverse as correctly rounded doubles.
inline constexpr double kInvSqrt2Pi = 0.39894228040143267794;
inline constexpr double kInvSqrt2 = 0.70710678118654752440;

/// phi(x) = exp(-x^2/2)/sqrt(2*pi); derivative -x*phi(x), so it composes in
/// forward and reverse mode through exp/mul only (no dedicated opcode).
template <typename T>
T normpdf(const T& x) {
    using std::exp;
    return detail::constantOf<T>(kInvSqrt2Pi) * exp(detail::constantOf<T>(-0.5) * (x * x));
}

/// Phi(x) = erfc(-x/sqrt(2))/2; derivative phi(x). Works for double, RevScalar
/// and (nested) Tangent via ADL: `erfc` resolves to std or the AD overload.
template <typename T>
T normcdf(const T& x) {
    using std::erfc;
    return detail::constantOf<T>(0.5) * erfc(detail::constantOf<T>(-kInvSqrt2) * x);
}

} // namespace lightape

#endif // LIGHTAPE_OPS_PROBABILITY_H
