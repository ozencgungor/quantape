#ifndef QUANTAPE_MATH_INTEGRALS_DOUBLE_EXPONENTIAL_INTEGRATOR_H
#define QUANTAPE_MATH_INTEGRALS_DOUBLE_EXPONENTIAL_INTEGRATOR_H

#include <cmath>

namespace quantape::math {
/**
 * @file DoubleExponentialIntegrator.h
 * @brief Ooura-style double-exponential quadrature for `int_0^inf f(x) dx`
 *
 * Independent cross-method validator for the Heston quadratures (EFGL,
 * Gauss–Legendre): the substitution
 *
 *     x(t) = exp(t - exp(-t)),    dx/dt = x (1 + exp(-t))
 *
 * maps the half-line to the whole line with double-exponential endpoint
 * decay and no cancellation near the tail. The symmetric sum converges for
 * oscillatory, exponentially decaying integrands (the Lewis residual class).
 *
 * Scalar-generic: `std::exp` for double, ADL for `DoubleDouble`/custom
 * scalars. Offline use only (thousands of evaluations).
 */
template <typename Scalar, typename F>
Scalar integrateDoubleExponential(const F& f, double step = 0.05, int halfTerms = 1200) {
    using std::exp;
    Scalar sum(0.0);
    for (int j = -halfTerms; j <= halfTerms; ++j) {
        const Scalar t(step * static_cast<double>(j));
        const Scalar emt = exp(-t);
        const Scalar x = exp(t - emt);
        sum = sum + f(x) * x * (Scalar(1.0) + emt);
    }
    return sum * Scalar(step);
}

} // namespace quantape::math

#endif // QUANTAPE_MATH_INTEGRALS_DOUBLE_EXPONENTIAL_INTEGRATOR_H
