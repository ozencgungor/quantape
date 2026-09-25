//
// InterpolationStanPrimitives.h -- Stan AD specializations for interpolation
// classes, providing the passive-abscissa fast paths.
//
// The default valueImpl/derivativeImpl are scalar-generic and differentiate
// the query coordinate (weights are DoubleT). The specializations here
// implement the *Fixed* methods (evaluateFixed/derivativeFixed): analytical
// adjoint (var) and nested analytical (`fvar<var>`) callbacks that keep the
// tape-node count minimal and intentionally do NOT push an adjoint into the
// query coordinate. Use them only when x is a constant.
//
// Follows the same pattern as Pricing/StanPrimitives.h for Black76/GBS.
// Include this header when using interpolation with stan::math::var or
// stan::math::`fvar<var>` to get optimized tape usage.
//

#ifndef INTERPOLATION_STAN_PRIMITIVES_H
#define INTERPOLATION_STAN_PRIMITIVES_H

#include "quantape/math/StanMath.h"

#include "BicubicInterpolation.h"
#include "BilinearInterpolation.h"
#include "CubicInterpolation.h"
#include "LinearInterpolation.h"
#include "LogLinearInterpolation.h"

namespace quantape::math {

// ============================================================================
// LinearInterpolation<var>::valueFixedImpl -- 1 tape node (down from 3)
// ============================================================================

template <>
inline stan::math::var
LinearInterpolation<stan::math::var>::valueFixedImpl(stan::math::var x) const {
    using stan::math::make_callback_var;
    using stan::math::var;

    size_t i = this->locate(x);

    double x1 = this->m_x[i];
    double x2 = this->m_x[i + 1];
    double xv = this->extractDouble(x);

    double inv_dx = 1.0 / (x2 - x1);
    double w0 = (x2 - xv) * inv_dx;
    double w1 = (xv - x1) * inv_dx;

    var y0 = this->m_y[i];
    var y1 = this->m_y[i + 1];
    double result = w0 * y0.val() + w1 * y1.val();

    // y0/y1 captured by value: var copies share the varis, and the callback
    // must stay valid even if the interpolation object is destroyed first.
    return make_callback_var(result, [y0, y1, w0, w1](auto& vi) {
        double adj = vi.adj();
        y0.adj() += adj * w0;
        y1.adj() += adj * w1;
    });
}

// ============================================================================
// LinearInterpolation<var>::derivativeFixedImpl -- 1 tape node (down from 2)
// ============================================================================

template <>
inline stan::math::var
LinearInterpolation<stan::math::var>::derivativeFixedImpl(stan::math::var x) const {
    using stan::math::make_callback_var;

    size_t i = this->locate(x);
    double inv_dx = 1.0 / (this->m_x[i + 1] - this->m_x[i]);

    stan::math::var y0 = this->m_y[i];
    stan::math::var y1 = this->m_y[i + 1];
    double result = (y1.val() - y0.val()) * inv_dx;

    return make_callback_var(result, [y0, y1, inv_dx](auto& vi) {
        double adj = vi.adj();
        y0.adj() += adj * (-inv_dx);
        y1.adj() += adj * inv_dx;
    });
}

// ============================================================================
// LinearInterpolation<`fvar<var>`>::valueFixedImpl -- 1 callback var, zero Hessian
//
// Linear in y => d2f/dy_j dy_k = 0 for all j,k.
// Value part: make_callback_var with w0, w1 adjoints.
// Tangent part: w0 * y[i].d_ + w1 * y[i+1].d_ (no Hessian callback vars).
// ============================================================================

template <>
inline stan::math::fvar<stan::math::var>
LinearInterpolation<stan::math::fvar<stan::math::var>>::valueFixedImpl(
    stan::math::fvar<stan::math::var> x) const {
    using stan::math::fvar;
    using stan::math::make_callback_var;
    using stan::math::var;

    size_t i = this->locate(x);

    double x1 = this->m_x[i];
    double x2 = this->m_x[i + 1];
    double xv = this->extractDouble(x);

    double inv_dx = 1.0 / (x2 - x1);
    double w0 = (x2 - xv) * inv_dx;
    double w1 = (xv - x1) * inv_dx;

    auto y0 = this->m_y[i];
    auto y1 = this->m_y[i + 1];

    double y0_val = y0.val_.val();
    double y1_val = y1.val_.val();
    double result = w0 * y0_val + w1 * y1_val;

    // Value part: single callback var pushing w0/w1 to val_ adjoints
    // (y0/y1 captured by value: fvar copies share the varis, and the
    // callback must stay valid if the interpolation object dies first)
    var val = make_callback_var(result, [y0, y1, w0, w1](auto& vi) {
        double adj = vi.adj();
        y0.val_.adj() += adj * w0;
        y1.val_.adj() += adj * w1;
    });

    // Tangent part: d2f/dy_j dy_k = 0, so tangent is just weighted sum
    // of tangent components — no callback vars needed for Hessian
    var tangent = w0 * y0.d_ + w1 * y1.d_;

    return fvar<var>(val, tangent);
}

// ============================================================================
// LinearInterpolation<`fvar<var>`>::derivativeFixedImpl -- 1 callback var, zero Hessian
// ============================================================================

template <>
inline stan::math::fvar<stan::math::var>
LinearInterpolation<stan::math::fvar<stan::math::var>>::derivativeFixedImpl(
    stan::math::fvar<stan::math::var> x) const {
    using stan::math::fvar;
    using stan::math::make_callback_var;
    using stan::math::var;

    size_t i = this->locate(x);
    double inv_dx = 1.0 / (this->m_x[i + 1] - this->m_x[i]);

    auto y0 = this->m_y[i];
    auto y1 = this->m_y[i + 1];

    double y0_val = y0.val_.val();
    double y1_val = y1.val_.val();
    double result = (y1_val - y0_val) * inv_dx;

    var val = make_callback_var(result, [y0, y1, inv_dx](auto& vi) {
        double adj = vi.adj();
        y0.val_.adj() += adj * (-inv_dx);
        y1.val_.adj() += adj * inv_dx;
    });

    var tangent = (-inv_dx) * y0.d_ + inv_dx * y1.d_;

    return fvar<var>(val, tangent);
}

// ============================================================================
// LogLinearInterpolation<var>::valueFixedImpl -- 1 tape node
//
// f(x) = exp( (1-t)*log(y_i) + t*log(y_{i+1}) )
// df/dy_i     = f * (1-t) / y_i
// df/dy_{i+1} = f * t / y_{i+1}
// ============================================================================

template <>
inline stan::math::var
LogLinearInterpolation<stan::math::var>::valueFixedImpl(stan::math::var x) const {
    using stan::math::make_callback_var;

    size_t i = this->locate(x);

    double x1 = this->m_x[i];
    double x2 = this->m_x[i + 1];
    double xv = this->extractDouble(x);
    double t = (xv - x1) / (x2 - x1);

    double y0 = this->m_y[i].val();
    double y1 = this->m_y[i + 1].val();
    double result = std::exp((1.0 - t) * std::log(y0) + t * std::log(y1));

    double dfdyi = result * (1.0 - t) / y0;
    double dfdyi1 = result * t / y1;

    stan::math::var y0v = this->m_y[i];
    stan::math::var y1v = this->m_y[i + 1];
    return make_callback_var(result, [y0v, y1v, dfdyi, dfdyi1](auto& vi) {
        double adj = vi.adj();
        y0v.adj() += adj * dfdyi;
        y1v.adj() += adj * dfdyi1;
    });
}

// ============================================================================
// LogLinearInterpolation<var>::derivativeFixedImpl -- 1 tape node
//
// f'(x) = f(x) * (log(y_{i+1}) - log(y_i)) * inv_dx
// df'/dy_i     = f * inv_dx / y_i * ((1-t)*dL - 1)
// df'/dy_{i+1} = f * inv_dx / y_{i+1} * (t*dL + 1)
// where dL = log(y_{i+1}) - log(y_i)
// ============================================================================

template <>
inline stan::math::var
LogLinearInterpolation<stan::math::var>::derivativeFixedImpl(stan::math::var x) const {
    using stan::math::make_callback_var;

    size_t i = this->locate(x);

    double x1 = this->m_x[i];
    double x2 = this->m_x[i + 1];
    double xv = this->extractDouble(x);
    double inv_dx = 1.0 / (x2 - x1);
    double t = (xv - x1) * inv_dx * (x2 - x1); // == (xv - x1) / (x2 - x1)
    // Simplify: t = (xv - x1) / (x2 - x1)
    t = (xv - x1) * inv_dx * (x2 - x1);
    // Actually: inv_dx = 1/(x2-x1), so t = (xv-x1)/(x2-x1)
    t = (xv - x1) / (x2 - x1);

    double y0 = this->m_y[i].val();
    double y1 = this->m_y[i + 1].val();
    double L0 = std::log(y0), L1 = std::log(y1);
    double dL = L1 - L0;
    double f_val = std::exp((1.0 - t) * L0 + t * L1);
    double result = f_val * dL * inv_dx;

    double dg_dy0 = f_val * inv_dx / y0 * ((1.0 - t) * dL - 1.0);
    double dg_dy1 = f_val * inv_dx / y1 * (t * dL + 1.0);

    stan::math::var y0v = this->m_y[i];
    stan::math::var y1v = this->m_y[i + 1];
    return make_callback_var(result, [y0v, y1v, dg_dy0, dg_dy1](auto& vi) {
        double adj = vi.adj();
        y0v.adj() += adj * dg_dy0;
        y1v.adj() += adj * dg_dy1;
    });
}

// ============================================================================
// LogLinearInterpolation<`fvar<var>`>::valueFixedImpl -- 2 callback vars + tangent
//
// Non-linear in y => Hessian is non-zero.
// d2f/dy_i^2         = -f * t*(1-t) / y_i^2
// d2f/dy_{i+1}^2     = -f * t*(1-t) / y_{i+1}^2
// d2f/(dy_i dy_{i+1}) = f * t*(1-t) / (y_i * y_{i+1})
// ============================================================================

template <>
inline stan::math::fvar<stan::math::var>
LogLinearInterpolation<stan::math::fvar<stan::math::var>>::valueFixedImpl(
    stan::math::fvar<stan::math::var> x) const {
    using stan::math::fvar;
    using stan::math::make_callback_var;
    using stan::math::var;

    size_t i = this->locate(x);

    double x1 = this->m_x[i];
    double x2 = this->m_x[i + 1];
    double xv = this->extractDouble(x);
    double t = (xv - x1) / (x2 - x1);

    auto yi = this->m_y[i];
    auto yi1 = this->m_y[i + 1];

    double y0 = yi.val_.val();
    double y1 = yi1.val_.val();
    double L0 = std::log(y0), L1 = std::log(y1);
    double f_val = std::exp((1.0 - t) * L0 + t * L1);

    // 1st-order gradients
    double g0 = f_val * (1.0 - t) / y0; // df/dy_i
    double g1 = f_val * t / y1;         // df/dy_{i+1}

    // 2nd-order (Hessian elements)
    double h00 = -f_val * t * (1.0 - t) / (y0 * y0); // d2f/dy_i^2
    double h11 = -f_val * t * (1.0 - t) / (y1 * y1); // d2f/dy_{i+1}^2
    double h01 = f_val * t * (1.0 - t) / (y0 * y1);  // d2f/(dy_i dy_{i+1})

    var yiv = yi.val_, yi1v = yi1.val_;

    // Value: single callback var (yi/yi1 captured by value: fvar copies
    // share the varis, and the callback must outlive the interpolator)
    var val = make_callback_var(f_val, [yi, yi1, g0, g1](auto& vi) {
        double adj = vi.adj();
        yi.val_.adj() += adj * g0;
        yi1.val_.adj() += adj * g1;
    });

    // Gradient callback vars encoding Hessian rows
    var dfdyi_var = make_callback_var(g0, [yiv, yi1v, h00, h01](auto& vi) {
        double a = vi.adj();
        yiv.adj() += a * h00;
        yi1v.adj() += a * h01;
    });

    var dfdyi1_var = make_callback_var(g1, [yiv, yi1v, h01, h11](auto& vi) {
        double a = vi.adj();
        yiv.adj() += a * h01;
        yi1v.adj() += a * h11;
    });

    // Tangent: dot product of gradient vars with tangent directions
    var tangent = dfdyi_var * yi.d_ + dfdyi1_var * yi1.d_;

    return fvar<var>(val, tangent);
}

// ============================================================================
// LogLinearInterpolation<`fvar<var>`>::derivativeFixedImpl -- 2 callback vars + tangent
//
// f'(x) = f(x) * dL * inv_dx   where dL = log(y_{i+1}) - log(y_i)
// Gradients and Hessian of f' w.r.t. (y_i, y_{i+1}) computed analytically.
// ============================================================================

template <>
inline stan::math::fvar<stan::math::var>
LogLinearInterpolation<stan::math::fvar<stan::math::var>>::derivativeFixedImpl(
    stan::math::fvar<stan::math::var> x) const {
    using stan::math::fvar;
    using stan::math::make_callback_var;
    using stan::math::var;

    size_t i = this->locate(x);

    double x1 = this->m_x[i];
    double x2 = this->m_x[i + 1];
    double xv = this->extractDouble(x);
    double inv_dx = 1.0 / (x2 - x1);
    double t = (xv - x1) / (x2 - x1);

    auto yi = this->m_y[i];
    auto yi1 = this->m_y[i + 1];

    double y0 = yi.val_.val();
    double y1 = yi1.val_.val();
    double L0 = std::log(y0), L1 = std::log(y1);
    double dL = L1 - L0;
    double f_val = std::exp((1.0 - t) * L0 + t * L1);
    double result = f_val * dL * inv_dx;

    // 1st-order gradients of f'
    double g0 = f_val * inv_dx / y0 * ((1.0 - t) * dL - 1.0);
    double g1 = f_val * inv_dx / y1 * (t * dL + 1.0);

    // 2nd-order (Hessian of f' w.r.t. y_i, y_{i+1})
    // d2f'/dy_i^2 = f*inv_dx/y_i^2 * [((1-t)*dL - 1)*((1-t) - 1) - (-1)]
    //            = f*inv_dx/y_i^2 * [((1-t)*dL - 1)*(-t) + 1]
    double a0 = (1.0 - t) * dL - 1.0;
    double a1 = t * dL + 1.0;
    double h00 = f_val * inv_dx / (y0 * y0) * (a0 * (-t) + 1.0);
    double h11 = f_val * inv_dx / (y1 * y1) * (a1 * (t - 1.0) - 1.0);
    double h01 = f_val * inv_dx / (y0 * y1) * (a0 * t);

    var yiv = yi.val_, yi1v = yi1.val_;

    var val = make_callback_var(result, [yi, yi1, g0, g1](auto& vi) {
        double adj = vi.adj();
        yi.val_.adj() += adj * g0;
        yi1.val_.adj() += adj * g1;
    });

    var dg0_var = make_callback_var(g0, [yiv, yi1v, h00, h01](auto& vi) {
        double a = vi.adj();
        yiv.adj() += a * h00;
        yi1v.adj() += a * h01;
    });

    var dg1_var = make_callback_var(g1, [yiv, yi1v, h01, h11](auto& vi) {
        double a = vi.adj();
        yiv.adj() += a * h01;
        yi1v.adj() += a * h11;
    });

    var tangent = dg0_var * yi.d_ + dg1_var * yi1.d_;

    return fvar<var>(val, tangent);
}

// ============================================================================
// BilinearInterpolation<var>::valueFixedImpl -- 1 tape node (down from 7)
//
// f(x,y) = w00*z00 + w10*z10 + w01*z01 + w11*z11
// where all weights are double. Linear in z => 4 adjoint pushes.
// ============================================================================

template <>
inline stan::math::var
BilinearInterpolation<stan::math::var>::valueFixedImpl(stan::math::var x, stan::math::var y) const {
    using stan::math::make_callback_var;
    using stan::math::var;

    double xv = this->extractDouble(x);
    double yv = this->extractDouble(y);

    size_t i = locateX(xv);
    size_t j = locateY(yv);

    double x1 = m_x[i], x2 = m_x[i + 1];
    double y1 = m_y[j], y2 = m_y[j + 1];

    double inv_dx = 1.0 / (x2 - x1);
    double inv_dy = 1.0 / (y2 - y1);
    double wx0 = (x2 - xv) * inv_dx;
    double wx1 = (xv - x1) * inv_dx;
    double wy0 = (y2 - yv) * inv_dy;
    double wy1 = (yv - y1) * inv_dy;

    double w00 = wy0 * wx0, w10 = wy0 * wx1;
    double w01 = wy1 * wx0, w11 = wy1 * wx1;

    double result = w00 * m_z[j][i].val() + w10 * m_z[j][i + 1].val() + w01 * m_z[j + 1][i].val() +
                    w11 * m_z[j + 1][i + 1].val();

    var z00 = m_z[j][i], z10 = m_z[j][i + 1];
    var z01 = m_z[j + 1][i], z11 = m_z[j + 1][i + 1];

    return make_callback_var(result, [z00, z10, z01, z11, w00, w10, w01, w11](auto& vi) {
        double adj = vi.adj();
        z00.adj() += adj * w00;
        z10.adj() += adj * w10;
        z01.adj() += adj * w01;
        z11.adj() += adj * w11;
    });
}

// ============================================================================
// BilinearInterpolation<`fvar<var>`>::valueFixedImpl -- 1 callback var, zero Hessian
//
// Linear in z => d2f/dz_j dz_k = 0 for all j,k.
// Same pattern as LinearInterpolation<`fvar<var>`>.
// ============================================================================

template <>
inline stan::math::fvar<stan::math::var>
BilinearInterpolation<stan::math::fvar<stan::math::var>>::valueFixedImpl(
    stan::math::fvar<stan::math::var> x, stan::math::fvar<stan::math::var> y) const {
    using stan::math::fvar;
    using stan::math::make_callback_var;
    using stan::math::var;

    double xv = this->extractDouble(x);
    double yv = this->extractDouble(y);

    size_t i = locateX(xv);
    size_t j = locateY(yv);

    double x1 = m_x[i], x2 = m_x[i + 1];
    double y1 = m_y[j], y2 = m_y[j + 1];

    double inv_dx = 1.0 / (x2 - x1);
    double inv_dy = 1.0 / (y2 - y1);
    double wx0 = (x2 - xv) * inv_dx, wx1 = (xv - x1) * inv_dx;
    double wy0 = (y2 - yv) * inv_dy, wy1 = (yv - y1) * inv_dy;

    double w00 = wy0 * wx0, w10 = wy0 * wx1;
    double w01 = wy1 * wx0, w11 = wy1 * wx1;

    auto z00 = m_z[j][i];
    auto z10 = m_z[j][i + 1];
    auto z01 = m_z[j + 1][i];
    auto z11 = m_z[j + 1][i + 1];

    double result =
        w00 * z00.val_.val() + w10 * z10.val_.val() + w01 * z01.val_.val() + w11 * z11.val_.val();

    // Value: single callback var (z values captured by value: fvar copies
    // share the varis, and the callback must outlive the interpolator)
    var val = make_callback_var(result, [z00, z10, z01, z11, w00, w10, w01, w11](auto& vi) {
        double adj = vi.adj();
        z00.val_.adj() += adj * w00;
        z10.val_.adj() += adj * w10;
        z01.val_.adj() += adj * w01;
        z11.val_.adj() += adj * w11;
    });

    // Tangent: Hessian is zero (linear in z), so just weighted tangent sum
    var tangent = w00 * z00.d_ + w10 * z10.d_ + w01 * z01.d_ + w11 * z11.d_;

    return fvar<var>(val, tangent);
}

// ============================================================================
// ============================================================================
// ============================================================================
// CubicInterpolation — weight-matrix specializations (Spline/Parabolic)
//
// P(x) in segment i = y_i + sum_j W_j(dx) * y_j, with W_j(dx) pure double
// (precomputed by probing the double solver). Consequences:
//   - var:        ONE tape node, O(n) adjoint pushes — no construction tape
//   - `fvar<var>`:  the y-Hessian is IDENTICALLY zero (P is linear in y),
//                 so no second-order callback machinery is needed.
//
// SAFETY: the callbacks are SELF-CONTAINED — they capture the node-value
// vector by value (shallow: var_value holds a pointer) and the weights by
// move, never `this`. The interpolator may die before the reverse pass
// runs (stan::math::gradient/hessian destroy the functor's locals before
// grad()), and a `this` capture would dangle. This mirrors the
// Pricing/StanPrimitives.h discipline.
// ============================================================================

template <>
inline stan::math::var
CubicInterpolation<stan::math::var>::weightMatrixValue(stan::math::var x) const {
    using stan::math::make_callback_var;
    using stan::math::var;

    const size_t n = this->m_x.size();
    const size_t seg = m_Wa.size() / n;
    size_t i = this->locate(x);
    if (i >= seg)
        i = seg - 1;

    const double dx = this->extractDouble(x) - this->m_x[i];
    const double dx2 = dx * dx;
    const double dx3 = dx2 * dx;

    double result = this->m_y[i].val(); // the delta(i, j) term
    std::vector<double> w(n, 0.0);
    for (size_t j = 0; j < n; ++j) {
        w[j] = m_Wa[i * n + j] * dx + m_Wb[i * n + j] * dx2 + m_Wc[i * n + j] * dx3;
        result += w[j] * this->m_y[j].val();
    }

    return make_callback_var(result, [y = this->m_y_shared, i, w = std::move(w)](auto& vi) {
        const double adj = vi.adj();
        (*y)[i].adj() += adj; // delta(i, j)
        for (size_t j = 0; j < w.size(); ++j)
            (*y)[j].adj() += adj * w[j];
    });
}

template <>
inline stan::math::var
CubicInterpolation<stan::math::var>::weightMatrixDerivative(stan::math::var x) const {
    using stan::math::make_callback_var;
    using stan::math::var;

    const size_t n = this->m_x.size();
    const size_t seg = m_Wa.size() / n;
    size_t i = this->locate(x);
    if (i >= seg)
        i = seg - 1;

    const double dx = this->extractDouble(x) - this->m_x[i];
    const double dx2 = dx * dx;

    // P'(x) = sum_j (W_a + 2 W_b dx + 3 W_c dx^2) * y_j   (no delta term)
    double result = 0.0;
    std::vector<double> w(n, 0.0);
    for (size_t j = 0; j < n; ++j) {
        w[j] = m_Wa[i * n + j] + 2.0 * m_Wb[i * n + j] * dx + 3.0 * m_Wc[i * n + j] * dx2;
        result += w[j] * this->m_y[j].val();
    }

    return make_callback_var(result, [y = this->m_y_shared, w = std::move(w)](auto& vi) {
        const double adj = vi.adj();
        for (size_t j = 0; j < w.size(); ++j)
            (*y)[j].adj() += adj * w[j];
    });
}

template <>
inline stan::math::fvar<stan::math::var>
CubicInterpolation<stan::math::fvar<stan::math::var>>::weightMatrixValue(
    stan::math::fvar<stan::math::var> x) const {
    using stan::math::fvar;
    using stan::math::make_callback_var;
    using stan::math::var;

    const size_t n = this->m_x.size();
    const size_t seg = m_Wa.size() / n;
    size_t i = this->locate(x);
    if (i >= seg)
        i = seg - 1;

    const double dx = this->extractDouble(x) - this->m_x[i];
    const double dx2 = dx * dx;
    const double dx3 = dx2 * dx;

    double result = this->m_y[i].val_.val();
    std::vector<double> w(n, 0.0);
    for (size_t j = 0; j < n; ++j) {
        w[j] = m_Wa[i * n + j] * dx + m_Wb[i * n + j] * dx2 + m_Wc[i * n + j] * dx3;
        result += w[j] * this->m_y[j].val_.val();
    }

    // Linear in y => y-Hessian is zero: tangent is just the weighted sum
    // of the tangent components — no Hessian callback vars.
    var tangent = this->m_y[i].d_;
    for (size_t j = 0; j < n; ++j)
        tangent += w[j] * this->m_y[j].d_;

    var val = make_callback_var(result, [y = this->m_y_shared, i, w = std::move(w)](auto& vi) {
        const double adj = vi.adj();
        (*y)[i].val_.adj() += adj;
        for (size_t j = 0; j < w.size(); ++j)
            (*y)[j].val_.adj() += adj * w[j];
    });

    return fvar<var>(val, tangent);
}

template <>
inline stan::math::fvar<stan::math::var>
CubicInterpolation<stan::math::fvar<stan::math::var>>::weightMatrixDerivative(
    stan::math::fvar<stan::math::var> x) const {
    using stan::math::fvar;
    using stan::math::make_callback_var;
    using stan::math::var;

    const size_t n = this->m_x.size();
    const size_t seg = m_Wa.size() / n;
    size_t i = this->locate(x);
    if (i >= seg)
        i = seg - 1;

    const double dx = this->extractDouble(x) - this->m_x[i];
    const double dx2 = dx * dx;

    double result = 0.0;
    std::vector<double> w(n, 0.0);
    for (size_t j = 0; j < n; ++j) {
        w[j] = m_Wa[i * n + j] + 2.0 * m_Wb[i * n + j] * dx + 3.0 * m_Wc[i * n + j] * dx2;
        result += w[j] * this->m_y[j].val_.val();
    }

    var tangent = 0.0;
    for (size_t j = 0; j < n; ++j)
        tangent += w[j] * this->m_y[j].d_;

    var val = make_callback_var(result, [y = this->m_y_shared, w = std::move(w)](auto& vi) {
        const double adj = vi.adj();
        for (size_t j = 0; j < w.size(); ++j)
            (*y)[j].val_.adj() += adj * w[j];
    });

    return fvar<var>(val, tangent);
}

// ============================================================================
// CubicInterpolation — branch-pinned local probe (Akima/Kruger/Harmonic)
//
// These methods are piecewise-linear in y with DATA-DEPENDENT branch
// selection, so precomputed weights would go stale. Instead, each
// evaluation runs the coefficient computation once with ProbeDual y-values
// (gradient = identity): every branch decision uses the PRIMAL, so the
// resulting dual coefficients linearize the ACTIVE branch — exactly the
// subgradient the generic tape path would produce, in ONE tape node.
//
// Note: the template parameter Smooth only sets the runtime default; pass
// smooth=true to the constructor — these specializations respect m_smooth.
// Callbacks are self-contained (y captured as the shared m_y_shared
// snapshot — one shared_ptr copy per evaluation instead of a full vector
// copy; weights by move) for the same lifetime reason as the weight-matrix
// specializations above.
// ============================================================================

template <>
inline stan::math::var
CubicInterpolation<stan::math::var>::localWeightsValue(stan::math::var x) const {
    using quantape::math::detail::ProbeDual;
    using stan::math::make_callback_var;
    using stan::math::var;

    const size_t n = this->m_x.size();
    const size_t seg = (n == 2) ? 1 : n - 1;
    size_t i = this->locate(x);
    if (i >= seg)
        i = seg - 1;

    const double dx = this->extractDouble(x) - this->m_x[i];
    const double dx2 = dx * dx;
    const double dx3 = dx2 * dx;

    std::vector<ProbeDual> y(n, ProbeDual(0.0, n));
    for (size_t j = 0; j < n; ++j) {
        y[j].v = this->m_y[j].val();
        y[j].d[j] = 1.0;
    }
    std::vector<ProbeDual> a, b, c;
    CubicInterpolation<var>::computeCoefficientsDual(this->m_x, y, m_da, m_smooth, a, b, c);

    const double result = this->m_y[i].val() + a[i].v * dx + b[i].v * dx2 + c[i].v * dx3;
    std::vector<double> w(n, 0.0);
    w[i] += 1.0; // delta(i, j)
    // Size-safe gradient read: zero entries (or any dual produced from a
    // scalar literal) carry an EMPTY gradient vector; treat missing entries
    // as zero gradients rather than reading out of bounds.
    auto dGet = [](const auto& dual, size_t k) { return k < dual.d.size() ? dual.d[k] : 0.0; };
    for (size_t j = 0; j < n; ++j)
        w[j] += dGet(a[i], j) * dx + dGet(b[i], j) * dx2 + dGet(c[i], j) * dx3;

    return make_callback_var(result, [y = this->m_y_shared, w = std::move(w)](auto& vi) {
        const double adj = vi.adj();
        for (size_t j = 0; j < w.size(); ++j)
            (*y)[j].adj() += adj * w[j];
    });
}

template <>
inline stan::math::var
CubicInterpolation<stan::math::var>::localWeightsDerivative(stan::math::var x) const {
    using quantape::math::detail::ProbeDual;
    using stan::math::make_callback_var;
    using stan::math::var;

    const size_t n = this->m_x.size();
    const size_t seg = (n == 2) ? 1 : n - 1;
    size_t i = this->locate(x);
    if (i >= seg)
        i = seg - 1;

    const double dx = this->extractDouble(x) - this->m_x[i];
    const double dx2 = dx * dx;

    std::vector<ProbeDual> y(n, ProbeDual(0.0, n));
    for (size_t j = 0; j < n; ++j) {
        y[j].v = this->m_y[j].val();
        y[j].d[j] = 1.0;
    }
    std::vector<ProbeDual> a, b, c;
    CubicInterpolation<var>::computeCoefficientsDual(this->m_x, y, m_da, m_smooth, a, b, c);

    const double result = a[i].v + 2.0 * b[i].v * dx + 3.0 * c[i].v * dx2;
    std::vector<double> w(n, 0.0);
    // Size-safe gradient read: zero entries (or any dual produced from a
    // scalar literal) carry an EMPTY gradient vector; treat missing entries
    // as zero gradients rather than reading out of bounds.
    auto dGet = [](const auto& dual, size_t k) { return k < dual.d.size() ? dual.d[k] : 0.0; };
    for (size_t j = 0; j < n; ++j)
        w[j] += dGet(a[i], j) + 2.0 * dGet(b[i], j) * dx + 3.0 * dGet(c[i], j) * dx2;

    return make_callback_var(result, [y = this->m_y_shared, w = std::move(w)](auto& vi) {
        const double adj = vi.adj();
        for (size_t j = 0; j < w.size(); ++j)
            (*y)[j].adj() += adj * w[j];
    });
}

template <>
inline stan::math::fvar<stan::math::var>
CubicInterpolation<stan::math::fvar<stan::math::var>>::localWeightsValue(
    stan::math::fvar<stan::math::var> x) const {
    using quantape::math::detail::ProbeDual;
    using stan::math::fvar;
    using stan::math::make_callback_var;
    using stan::math::var;

    const size_t n = this->m_x.size();
    const size_t seg = (n == 2) ? 1 : n - 1;
    size_t i = this->locate(x);
    if (i >= seg)
        i = seg - 1;

    const double dx = this->extractDouble(x) - this->m_x[i];
    const double dx2 = dx * dx;
    const double dx3 = dx2 * dx;

    std::vector<ProbeDual> y(n, ProbeDual(0.0, n));
    for (size_t j = 0; j < n; ++j) {
        y[j].v = this->m_y[j].val_.val();
        y[j].d[j] = 1.0;
    }
    std::vector<ProbeDual> a, b, c;
    CubicInterpolation<fvar<var>>::computeCoefficientsDual(this->m_x, y, m_da, m_smooth, a, b, c);

    const double result = this->m_y[i].val_.val() + a[i].v * dx + b[i].v * dx2 + c[i].v * dx3;
    std::vector<double> w(n, 0.0);
    w[i] += 1.0;
    // Size-safe gradient read: zero entries (or any dual produced from a
    // scalar literal) carry an EMPTY gradient vector; treat missing entries
    // as zero gradients rather than reading out of bounds.
    auto dGet = [](const auto& dual, size_t k) { return k < dual.d.size() ? dual.d[k] : 0.0; };
    for (size_t j = 0; j < n; ++j)
        w[j] += dGet(a[i], j) * dx + dGet(b[i], j) * dx2 + dGet(c[i], j) * dx3;

    var tangent = this->m_y[i].d_;
    for (size_t j = 0; j < n; ++j)
        tangent += w[j] * this->m_y[j].d_;

    var val = make_callback_var(result, [y = this->m_y_shared, w = std::move(w)](auto& vi) {
        const double adj = vi.adj();
        for (size_t j = 0; j < w.size(); ++j)
            (*y)[j].val_.adj() += adj * w[j];
    });

    return fvar<var>(val, tangent);
}

template <>
inline stan::math::fvar<stan::math::var>
CubicInterpolation<stan::math::fvar<stan::math::var>>::localWeightsDerivative(
    stan::math::fvar<stan::math::var> x) const {
    using quantape::math::detail::ProbeDual;
    using stan::math::fvar;
    using stan::math::make_callback_var;
    using stan::math::var;

    const size_t n = this->m_x.size();
    const size_t seg = (n == 2) ? 1 : n - 1;
    size_t i = this->locate(x);
    if (i >= seg)
        i = seg - 1;

    const double dx = this->extractDouble(x) - this->m_x[i];
    const double dx2 = dx * dx;

    std::vector<ProbeDual> y(n, ProbeDual(0.0, n));
    for (size_t j = 0; j < n; ++j) {
        y[j].v = this->m_y[j].val_.val();
        y[j].d[j] = 1.0;
    }
    std::vector<ProbeDual> a, b, c;
    CubicInterpolation<fvar<var>>::computeCoefficientsDual(this->m_x, y, m_da, m_smooth, a, b, c);

    const double result = a[i].v + 2.0 * b[i].v * dx + 3.0 * c[i].v * dx2;
    std::vector<double> w(n, 0.0);
    // Size-safe gradient read: zero entries (or any dual produced from a
    // scalar literal) carry an EMPTY gradient vector; treat missing entries
    // as zero gradients rather than reading out of bounds.
    auto dGet = [](const auto& dual, size_t k) { return k < dual.d.size() ? dual.d[k] : 0.0; };
    for (size_t j = 0; j < n; ++j)
        w[j] += dGet(a[i], j) + 2.0 * dGet(b[i], j) * dx + 3.0 * dGet(c[i], j) * dx2;

    var tangent = 0.0;
    for (size_t j = 0; j < n; ++j)
        tangent += w[j] * this->m_y[j].d_;

    var val = make_callback_var(result, [y = this->m_y_shared, w = std::move(w)](auto& vi) {
        const double adj = vi.adj();
        for (size_t j = 0; j < w.size(); ++j)
            (*y)[j].val_.adj() += adj * w[j];
    });

    return fvar<var>(val, tangent);
}

} // namespace quantape::math

#endif // INTERPOLATION_STAN_PRIMITIVES_H
