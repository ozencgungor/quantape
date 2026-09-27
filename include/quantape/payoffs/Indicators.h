#ifndef QUANTAPE_PAYOFFS_INDICATORS_H
#define QUANTAPE_PAYOFFS_INDICATORS_H

#include <cmath>

namespace quantape::payoffs {
/**
 * @file Indicators.h
 * @brief Branchless smoothed indicators for pathwise AD payoffs
 *
 * Pathwise (IPA) differentiation of a discontinuous indicator is zero
 * (`d 1{x>0}/dx = 0` a.e.), which silently produces wrong greeks for
 * digital/barrier payoffs. The remedy used across the AD literature
 * (Savine, *LSM Reloaded* §4.1.2; Antonov et al.) is a smooth binary, and
 * its simplest form is the linear ramp (call spread with width eps):
 *
 *     smoothIndicator(x, eps) = clamp(x/eps + 1/2, 0, 1)
 *
 * which equals Savine's two-term expression
 *
 *     (x/eps + 1/2) 1{x > -eps/2} - (x/eps - 1/2) 1{x > eps/2}
 *
 * and tends to `1{x > 0}` as eps -> 0, with O(eps^2) bias for smooth
 * densities (the ramp is symmetric). Its derivative is
 *
 *     smoothIndicatorDerivative(x, eps) = (1/eps) 1{|x| < eps/2},
 *
 * obtained here as a difference of two ramps so the whole utility is
 * branchless (`fmin`/`fmax`; ADL picks the AD overloads for `var`/`fvar`).
 */

/// Linear-ramp smooth binary: 0 below -eps/2, 1 above +eps/2, slope 1/eps.
template <typename Scalar>
Scalar smoothIndicator(const Scalar& x, double eps) {
    using std::fmax;
    using std::fmin;
    const Scalar t = x / Scalar(eps) + Scalar(0.5);
    return fmin(fmax(t, Scalar(0.0)), Scalar(1.0));
}

/// Derivative of the linear ramp: 1/eps on |x| < eps/2, else 0. A boxcar is
/// intrinsically discontinuous, so this is one comparison returning a
/// constant (compiles to a select; the branch is on the primal value and
/// the result is constant in x, which is the correct a.e. derivative).
template <typename Scalar>
Scalar smoothIndicatorDerivative(const Scalar& x, double eps) {
    using std::fabs;
    const bool inside = fabs(x) < 0.5 * eps;
    return inside ? Scalar(1.0 / eps) : Scalar(0.0);
}

/**
 * @brief Smoothed positive part for payoffs with a kink at zero
 *
 * `smoothPositivePart(x, eps) = x * smoothIndicator(x, eps)` keeps the
 * kink's correct one-sided behavior within |x| < eps/2 while making the
 * function differentiable there. For first-order pathwise greeks the plain
 * `max(x, 0)` is already a.e. differentiable (the kink is measure zero);
 * smoothing matters for second order — provided for the callable layer's
 * gamma work (`callable_ad_design.md` §5).
 */
template <typename Scalar>
Scalar smoothPositivePart(const Scalar& x, double eps) {
    return x * smoothIndicator(x, eps);
}

} // namespace quantape::payoffs

#endif // QUANTAPE_PAYOFFS_INDICATORS_H
