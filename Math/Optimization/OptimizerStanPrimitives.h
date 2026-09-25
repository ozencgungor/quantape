#ifndef OPTIMIZER_STAN_PRIMITIVES_H
#define OPTIMIZER_STAN_PRIMITIVES_H

//
// OptimizerStanPrimitives.h -- AD dispatch for the Math/Optimization classes
//
// Include this header together with a Stan Math header; the
// Optimizer<DoubleT, Impl> base then evaluates gradients and HVPs
// automatically:
//
//   Optimizer<var, LBFGS<var>>::minimize   -> exact gradients (one reverse pass)
//   TNewton<var>::minimize                 -> exact HVP (forward-over-reverse)
//
// Following the pattern of Solvers/SolverStanPrimitives.h and
// Integrals/IntegratorStanPrimitives.h:
//   - OptimizerPrimitives.h stays Stan-free and declares the entry points;
//   - everything here is defined for AD scalars only and is never instantiated
//     for double;
//   - each helper runs in its own nested_rev_autodiff scope, so the tape is
//     recovered immediately and nothing leaks across iterations.
//
// Tape hygiene: the internal reverse passes write operand adjoints directly
// (nested scope or not), so adjoints that must survive are managed explicitly;
// these helpers run during forward construction and must not be called from
// inside a chain()/callback.
//

#include "Math/StanMath.h"

#include <cstddef>
#include <type_traits>
#include <utility>
#include <vector>

#include "OptimizerPrimitives.h"

namespace Math {
namespace detail {

/**
 * @brief Value + exact gradient of a scalar objective at a double point
 *
 * DoubleT = var: one reverse pass over the tape built with var parameters.
 * DoubleT = fvar<...>: reverse over the value part; the tangent part is
 * ignored here (use hvp for second-order information).
 */
template <typename DoubleT, typename F>
std::pair<double, std::vector<double>> valueGrad(const F& f, const std::vector<double>& x) {
    static_assert(!std::is_same_v<DoubleT, double>,
                  "valueGrad requires an AD scalar; use Optimizer::evalValueGrad");

    stan::math::nested_rev_autodiff nested;
    std::vector<DoubleT> theta(x.size());
    for (std::size_t i = 0; i < x.size(); ++i) {
        theta[i] = DoubleT(x[i]);
    }

    DoubleT y = f(theta);

    std::vector<double> grad(x.size());
    if constexpr (stan::is_var<DoubleT>::value) {
        y.grad();
        for (std::size_t i = 0; i < x.size(); ++i) {
            grad[i] = theta[i].adj();
        }
    } else {
        y.val_.grad();
        for (std::size_t i = 0; i < x.size(); ++i) {
            grad[i] = theta[i].val_.adj();
        }
    }
    return {primalValue(y), std::move(grad)};
}

/**
 * @brief Exact Hessian-vector product H(x) v, written into a caller buffer
 *
 * The `theta` fvar<var> scratch vector is reused across calls (the nested
 * scope recovers the tape per call; only the varis are new). This is the
 * allocation-free variant used by dense-Hessian assembly (ImplicitFunction.h).
 */
template <typename F>
inline void hvpInto(const F& f, const std::vector<double>& x, const std::vector<double>& v,
                    std::vector<stan::math::fvar<stan::math::var>>& theta,
                    std::vector<double>& out) {
    using fvar = stan::math::fvar<stan::math::var>;
    using var = stan::math::var;

    stan::math::nested_rev_autodiff nested;
    theta.resize(x.size());
    for (std::size_t i = 0; i < x.size(); ++i) {
        theta[i] = fvar(var(x[i]), var(v[i]));
    }

    fvar y = f(theta);
    y.d_.grad();

    out.resize(x.size());
    for (std::size_t i = 0; i < x.size(); ++i) {
        out[i] = theta[i].val_.adj();
    }
}

/**
 * @brief Exact Hessian-vector product H(x) v
 *
 * Parameters are seeded as fvar<var>(x_i, v_i); y.d_ is the directional
 * derivative grad(f) . v, and one reverse pass over it returns
 * d/dx [grad(f) . v] = H v.
 *
 * One forward-over-reverse pass, machine precision, no step size and no extra
 * gradient evaluation (contrast with NLopt's PNET finite differences). The
 * objective must accept std::vector<fvar<var>>.
 */
template <typename F>
std::vector<double> hvp(const F& f, const std::vector<double>& x, const std::vector<double>& v) {
    std::vector<stan::math::fvar<stan::math::var>> theta;
    std::vector<double> out;
    hvpInto(f, x, v, theta, out);
    return out;
}

/**
 * @brief Value + exact gradient written into caller-provided buffers
 *
 * `theta` caches the AD parameter vector across evaluations (the tape is
 * recovered per call by the nested scope; only the varis are new), `grad` is
 * reused. This is the allocation-free variant used by the optimizer loops.
 */
template <typename DoubleT, typename F>
double valueGrad(const F& f, const std::vector<double>& x, std::vector<DoubleT>& theta,
                 std::vector<double>& grad) {
    static_assert(!std::is_same_v<DoubleT, double>,
                  "valueGrad requires an AD scalar; use Optimizer::evalValueGradInto");

    stan::math::nested_rev_autodiff nested;
    theta.resize(x.size());
    for (std::size_t i = 0; i < x.size(); ++i) {
        theta[i] = DoubleT(x[i]);
    }

    DoubleT y = f(theta);
    grad.resize(x.size());
    if constexpr (stan::is_var<DoubleT>::value) {
        y.grad();
        for (std::size_t i = 0; i < x.size(); ++i) {
            grad[i] = theta[i].adj();
        }
    } else {
        y.val_.grad();
        for (std::size_t i = 0; i < x.size(); ++i) {
            grad[i] = theta[i].val_.adj();
        }
    }
    return primalValue(y);
}

/**
 * @brief Constraint values and row-major Jacobian at a double point
 *
 * One evaluation of g(theta, out); then one isolated reverse pass per
 * constraint row (adjoints are zeroed between rows), filling J with
 * d g_r / d x_i in row-major order.
 */
template <typename DoubleT, typename G>
void constraintValueJacobian(const G& g, const std::vector<double>& x, std::vector<double>& c,
                             std::vector<double>& jacobian) {
    static_assert(!std::is_same_v<DoubleT, double>,
                  "constraintValueJacobian requires an AD scalar");

    stan::math::nested_rev_autodiff nested;

    std::vector<DoubleT> theta(x.size());
    for (std::size_t i = 0; i < x.size(); ++i) {
        theta[i] = DoubleT(x[i]);
    }

    std::vector<DoubleT> out;
    g(theta, out);

    const std::size_t m = out.size();
    const std::size_t n = x.size();
    c.resize(m);
    jacobian.assign(m * n, 0.0);

    for (std::size_t r = 0; r < m; ++r) {
        c[r] = primalValue(out[r]);
    }

    for (std::size_t r = 0; r < m; ++r) {
        // Isolate each row: the previous reverse pass left row adjoints on the
        // parameters (and on any captured data), so clear before the next one.
        stan::math::set_zero_all_adjoints();
        if constexpr (stan::is_var<DoubleT>::value) {
            out[r].grad();
            for (std::size_t i = 0; i < n; ++i) {
                jacobian[r * n + i] = theta[i].adj();
            }
        } else {
            out[r].val_.grad();
            for (std::size_t i = 0; i < n; ++i) {
                jacobian[r * n + i] = theta[i].val_.adj();
            }
        }
    }
}

} // namespace detail
} // namespace Math

#endif // OPTIMIZER_STAN_PRIMITIVES_H
