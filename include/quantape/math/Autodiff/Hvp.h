#include "quantape/math/StanMath.h"
#ifndef HVP_H
#define HVP_H

namespace quantape::math {

/**
 * @brief Hessian-vector product by forward-over-reverse AD
 *
 * Evaluates the objective at x with an `fvar<var>` whose tangent is v:
 *
 *   y = f(`fvar<var>`(x, v))     =>     y.d_ = f'(x) * v
 *
 * One reverse pass over y.d_ then yields d/dx [f'(x) v] = f''(x) v.
 *
 * The objective must be scalar-generic (callable with `fvar<var>` as well as
 * var/double), the same contract as the integrators and interpolators.
 *
 * Side effects: the internal reverse pass runs during forward construction and
 * clears adjoints afterwards (like the solver IFT path), so call this before
 * the caller's own reverse pass and not inside a chain()/callback.
 *
 * @param f Scalar-generic objective
 * @param x Evaluation point
 * @param v Seed direction
 * @return f''(x) * v
 */
template <typename F>
double hvp(const F& f, double x, double v) {
    using stan::math::fvar;
    using stan::math::var;

    var xv(x);
    var vv(v);
    fvar<var> y = f(fvar<var>(xv, vv));

    stan::math::set_zero_all_adjoints();
    y.d_.grad();
    const double result = xv.adj();
    stan::math::set_zero_all_adjoints();
    return result;
}

} // namespace quantape::math

#endif // HVP_H
