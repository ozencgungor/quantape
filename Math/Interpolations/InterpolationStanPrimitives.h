//
// InterpolationStanPrimitives.h -- Stan AD specializations for interpolation classes
//
// Provides analytical adjoint (var) and nested analytical (fvar<var>)
// specializations for interpolation valueImpl and derivativeImpl methods.
//
// Follows the same pattern as Pricing/StanPrimitives.h for Black76/GBS.
// Include this header when using interpolation with stan::math::var or
// stan::math::fvar<var> to get optimized tape usage.
//

#ifndef INTERPOLATION_STAN_PRIMITIVES_H
#define INTERPOLATION_STAN_PRIMITIVES_H

#include <stan/math.hpp>
#include <stan/math/mix.hpp>

#include "BilinearInterpolation.h"
#include "CubicInterpolation.h"
#include "LinearInterpolation.h"
#include "LogLinearInterpolation.h"

namespace Math {

// ============================================================================
// LinearInterpolation<var>::valueImpl -- 1 tape node (down from 3)
// ============================================================================

template <>
inline stan::math::var
LinearInterpolation<stan::math::var>::valueImpl(stan::math::var x) const {
    using stan::math::make_callback_var;
    using stan::math::var;

    size_t i = this->locate(x);

    double x1 = this->m_x[i];
    double x2 = this->m_x[i + 1];
    double xv = this->extractDouble(x);

    double inv_dx = 1.0 / (x2 - x1);
    double w0 = (x2 - xv) * inv_dx;
    double w1 = (xv - x1) * inv_dx;

    double y0_val = this->m_y[i].val();
    double y1_val = this->m_y[i + 1].val();
    double result = w0 * y0_val + w1 * y1_val;

    return make_callback_var(result, [this, i, w0, w1](auto& vi) {
        double adj = vi.adj();
        this->m_y[i].adj() += adj * w0;
        this->m_y[i + 1].adj() += adj * w1;
    });
}

// ============================================================================
// LinearInterpolation<var>::derivativeImpl -- 1 tape node (down from 2)
// ============================================================================

template <>
inline stan::math::var
LinearInterpolation<stan::math::var>::derivativeImpl(stan::math::var x) const {
    using stan::math::make_callback_var;

    size_t i = this->locate(x);
    double inv_dx = 1.0 / (this->m_x[i + 1] - this->m_x[i]);

    double y0_val = this->m_y[i].val();
    double y1_val = this->m_y[i + 1].val();
    double result = (y1_val - y0_val) * inv_dx;

    return make_callback_var(result, [this, i, inv_dx](auto& vi) {
        double adj = vi.adj();
        this->m_y[i].adj() += adj * (-inv_dx);
        this->m_y[i + 1].adj() += adj * inv_dx;
    });
}

// ============================================================================
// LinearInterpolation<fvar<var>>::valueImpl -- 1 callback var, zero Hessian
//
// Linear in y => d2f/dy_j dy_k = 0 for all j,k.
// Value part: make_callback_var with w0, w1 adjoints.
// Tangent part: w0 * y[i].d_ + w1 * y[i+1].d_ (no Hessian callback vars).
// ============================================================================

template <>
inline stan::math::fvar<stan::math::var>
LinearInterpolation<stan::math::fvar<stan::math::var>>::valueImpl(
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

    const auto& y0 = this->m_y[i];
    const auto& y1 = this->m_y[i + 1];

    double y0_val = y0.val_.val();
    double y1_val = y1.val_.val();
    double result = w0 * y0_val + w1 * y1_val;

    // Value part: single callback var pushing w0/w1 to val_ adjoints
    var val = make_callback_var(result, [&y0, &y1, w0, w1](auto& vi) {
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
// LinearInterpolation<fvar<var>>::derivativeImpl -- 1 callback var, zero Hessian
// ============================================================================

template <>
inline stan::math::fvar<stan::math::var>
LinearInterpolation<stan::math::fvar<stan::math::var>>::derivativeImpl(
    stan::math::fvar<stan::math::var> x) const {
    using stan::math::fvar;
    using stan::math::make_callback_var;
    using stan::math::var;

    size_t i = this->locate(x);
    double inv_dx = 1.0 / (this->m_x[i + 1] - this->m_x[i]);

    const auto& y0 = this->m_y[i];
    const auto& y1 = this->m_y[i + 1];

    double y0_val = y0.val_.val();
    double y1_val = y1.val_.val();
    double result = (y1_val - y0_val) * inv_dx;

    var val = make_callback_var(result, [&y0, &y1, inv_dx](auto& vi) {
        double adj = vi.adj();
        y0.val_.adj() += adj * (-inv_dx);
        y1.val_.adj() += adj * inv_dx;
    });

    var tangent = (-inv_dx) * y0.d_ + inv_dx * y1.d_;

    return fvar<var>(val, tangent);
}

// ============================================================================
// LogLinearInterpolation<var>::valueImpl -- 1 tape node
//
// f(x) = exp( (1-t)*log(y_i) + t*log(y_{i+1}) )
// df/dy_i     = f * (1-t) / y_i
// df/dy_{i+1} = f * t / y_{i+1}
// ============================================================================

template <>
inline stan::math::var
LogLinearInterpolation<stan::math::var>::valueImpl(stan::math::var x) const {
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

    return make_callback_var(result, [this, i, dfdyi, dfdyi1](auto& vi) {
        double adj = vi.adj();
        this->m_y[i].adj() += adj * dfdyi;
        this->m_y[i + 1].adj() += adj * dfdyi1;
    });
}

// ============================================================================
// LogLinearInterpolation<var>::derivativeImpl -- 1 tape node
//
// f'(x) = f(x) * (log(y_{i+1}) - log(y_i)) * inv_dx
// df'/dy_i     = f * inv_dx / y_i * ((1-t)*dL - 1)
// df'/dy_{i+1} = f * inv_dx / y_{i+1} * (t*dL + 1)
// where dL = log(y_{i+1}) - log(y_i)
// ============================================================================

template <>
inline stan::math::var
LogLinearInterpolation<stan::math::var>::derivativeImpl(stan::math::var x) const {
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

    return make_callback_var(result, [this, i, dg_dy0, dg_dy1](auto& vi) {
        double adj = vi.adj();
        this->m_y[i].adj() += adj * dg_dy0;
        this->m_y[i + 1].adj() += adj * dg_dy1;
    });
}

// ============================================================================
// LogLinearInterpolation<fvar<var>>::valueImpl -- 2 callback vars + tangent
//
// Non-linear in y => Hessian is non-zero.
// d2f/dy_i^2         = -f * t*(1-t) / y_i^2
// d2f/dy_{i+1}^2     = -f * t*(1-t) / y_{i+1}^2
// d2f/(dy_i dy_{i+1}) = f * t*(1-t) / (y_i * y_{i+1})
// ============================================================================

template <>
inline stan::math::fvar<stan::math::var>
LogLinearInterpolation<stan::math::fvar<stan::math::var>>::valueImpl(
    stan::math::fvar<stan::math::var> x) const {
    using stan::math::fvar;
    using stan::math::make_callback_var;
    using stan::math::var;

    size_t i = this->locate(x);

    double x1 = this->m_x[i];
    double x2 = this->m_x[i + 1];
    double xv = this->extractDouble(x);
    double t = (xv - x1) / (x2 - x1);

    const auto& yi = this->m_y[i];
    const auto& yi1 = this->m_y[i + 1];

    double y0 = yi.val_.val();
    double y1 = yi1.val_.val();
    double L0 = std::log(y0), L1 = std::log(y1);
    double f_val = std::exp((1.0 - t) * L0 + t * L1);

    // 1st-order gradients
    double g0 = f_val * (1.0 - t) / y0;  // df/dy_i
    double g1 = f_val * t / y1;            // df/dy_{i+1}

    // 2nd-order (Hessian elements)
    double h00 = -f_val * t * (1.0 - t) / (y0 * y0);   // d2f/dy_i^2
    double h11 = -f_val * t * (1.0 - t) / (y1 * y1);   // d2f/dy_{i+1}^2
    double h01 = f_val * t * (1.0 - t) / (y0 * y1);    // d2f/(dy_i dy_{i+1})

    var yiv = yi.val_, yi1v = yi1.val_;

    // Value: single callback var
    var val = make_callback_var(f_val, [&yi, &yi1, g0, g1](auto& vi) {
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
// LogLinearInterpolation<fvar<var>>::derivativeImpl -- 2 callback vars + tangent
//
// f'(x) = f(x) * dL * inv_dx   where dL = log(y_{i+1}) - log(y_i)
// Gradients and Hessian of f' w.r.t. (y_i, y_{i+1}) computed analytically.
// ============================================================================

template <>
inline stan::math::fvar<stan::math::var>
LogLinearInterpolation<stan::math::fvar<stan::math::var>>::derivativeImpl(
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

    const auto& yi = this->m_y[i];
    const auto& yi1 = this->m_y[i + 1];

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

    var val = make_callback_var(result, [&yi, &yi1, g0, g1](auto& vi) {
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
// BilinearInterpolation<var>::valueImpl -- 1 tape node (down from 7)
//
// f(x,y) = w00*z00 + w10*z10 + w01*z01 + w11*z11
// where all weights are double. Linear in z => 4 adjoint pushes.
// ============================================================================

template <>
inline stan::math::var BilinearInterpolation<stan::math::var>::valueImpl(stan::math::var x,
                                                                         stan::math::var y) const {
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

    double result = w00 * m_z[j][i].val() + w10 * m_z[j][i + 1].val() +
                    w01 * m_z[j + 1][i].val() + w11 * m_z[j + 1][i + 1].val();

    return make_callback_var(result, [this, i, j, w00, w10, w01, w11](auto& vi) {
        double adj = vi.adj();
        this->m_z[j][i].adj() += adj * w00;
        this->m_z[j][i + 1].adj() += adj * w10;
        this->m_z[j + 1][i].adj() += adj * w01;
        this->m_z[j + 1][i + 1].adj() += adj * w11;
    });
}

// ============================================================================
// BilinearInterpolation<fvar<var>>::valueImpl -- 1 callback var, zero Hessian
//
// Linear in z => d2f/dz_j dz_k = 0 for all j,k.
// Same pattern as LinearInterpolation<fvar<var>>.
// ============================================================================

template <>
inline stan::math::fvar<stan::math::var> BilinearInterpolation<stan::math::fvar<stan::math::var>>::
    valueImpl(stan::math::fvar<stan::math::var> x,
              stan::math::fvar<stan::math::var> y) const {
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

    const auto& z00 = m_z[j][i];
    const auto& z10 = m_z[j][i + 1];
    const auto& z01 = m_z[j + 1][i];
    const auto& z11 = m_z[j + 1][i + 1];

    double result = w00 * z00.val_.val() + w10 * z10.val_.val() + w01 * z01.val_.val() +
                    w11 * z11.val_.val();

    // Value: single callback var
    var val = make_callback_var(result, [&z00, &z10, &z01, &z11, w00, w10, w01, w11](auto& vi) {
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
// CubicInterpolation<var>::valueImpl -- 1 tape node (down from 6)
//
// P(x) = y[i] + a[i]*dx + b[i]*dx^2 + c[i]*dx^3
// where dx is double, and (y[i], a[i], b[i], c[i]) are var.
// Coefficients a,b,c were computed at construction time and are already on
// the tape (they depend on ALL y values through the spline/derivative solve).
// This specialization eliminates the 6 per-evaluation tape nodes by computing
// the result in double and pushing adjoints analytically to the 4 inputs.
//
// Note: the O(n) tape nodes from coefficient computation at construction
// remain. For O(1) total tape, precompute the full weight matrix W_j(x)
// mapping y values directly to P(x) — see design doc for details.
// ============================================================================

template <>
inline stan::math::var
CubicInterpolation<stan::math::var>::valueImpl(stan::math::var x) const {
    using stan::math::make_callback_var;
    using stan::math::var;

    size_t i = this->locate(x);
    if (i >= m_a.size())
        i = m_a.size() - 1;

    double dx = this->extractDouble(x) - this->m_x[i];
    double dx2 = dx * dx;
    double dx3 = dx2 * dx;

    double yi = this->m_y[i].val();
    double ai = m_a[i].val();
    double bi = m_b[i].val();
    double ci = m_c[i].val();

    double result = yi + dx * (ai + dx * (bi + dx * ci));

    // dP/dy[i] = 1,  dP/da[i] = dx,  dP/db[i] = dx^2,  dP/dc[i] = dx^3
    return make_callback_var(result, [this, i, dx, dx2, dx3](auto& vi) {
        double adj = vi.adj();
        this->m_y[i].adj() += adj;
        this->m_a[i].adj() += adj * dx;
        this->m_b[i].adj() += adj * dx2;
        this->m_c[i].adj() += adj * dx3;
    });
}

// ============================================================================
// CubicInterpolation<var>::derivativeImpl -- 1 tape node (down from 5)
//
// P'(x) = a[i] + 2*b[i]*dx + 3*c[i]*dx^2
// ============================================================================

template <>
inline stan::math::var
CubicInterpolation<stan::math::var>::derivativeImpl(stan::math::var x) const {
    using stan::math::make_callback_var;

    size_t i = this->locate(x);
    if (i >= m_a.size())
        i = m_a.size() - 1;

    double dx = this->extractDouble(x) - this->m_x[i];
    double dx2 = dx * dx;

    double ai = m_a[i].val();
    double bi = m_b[i].val();
    double ci = m_c[i].val();

    double result = ai + dx * (2.0 * bi + 3.0 * ci * dx);

    // dP'/da[i] = 1,  dP'/db[i] = 2*dx,  dP'/dc[i] = 3*dx^2
    return make_callback_var(result, [this, i, dx, dx2](auto& vi) {
        double adj = vi.adj();
        this->m_a[i].adj() += adj;
        this->m_b[i].adj() += adj * 2.0 * dx;
        this->m_c[i].adj() += adj * 3.0 * dx2;
    });
}

// ============================================================================
// CubicInterpolation<fvar<var>>::valueImpl -- 1 callback var, zero Hessian
//
// P(x) is linear in (y[i], a[i], b[i], c[i]):
//   dP/dy[i] = 1,  dP/da[i] = dx,  dP/db[i] = dx^2,  dP/dc[i] = dx^3
// All second derivatives d2P/d(input_j)d(input_k) = 0.
//
// Value: 1 callback var pushing to val_ components.
// Tangent: weighted sum of d_ components (no Hessian callback vars needed).
// Total: 1 callback var + 6 arithmetic nodes for tangent.
// ============================================================================

template <>
inline stan::math::fvar<stan::math::var>
CubicInterpolation<stan::math::fvar<stan::math::var>>::valueImpl(
    stan::math::fvar<stan::math::var> x) const {
    using stan::math::fvar;
    using stan::math::make_callback_var;
    using stan::math::var;

    size_t i = this->locate(x);
    if (i >= m_a.size())
        i = m_a.size() - 1;

    double dx = this->extractDouble(x) - this->m_x[i];
    double dx2 = dx * dx;
    double dx3 = dx2 * dx;

    const auto& yi = this->m_y[i];
    const auto& ai = m_a[i];
    const auto& bi = m_b[i];
    const auto& ci = m_c[i];

    double result = yi.val_.val() + dx * (ai.val_.val() + dx * (bi.val_.val() + dx * ci.val_.val()));

    // Value: 1 callback var
    var val = make_callback_var(result, [&yi, &ai, &bi, &ci, dx, dx2, dx3](auto& vi) {
        double adj = vi.adj();
        yi.val_.adj() += adj;
        ai.val_.adj() += adj * dx;
        bi.val_.adj() += adj * dx2;
        ci.val_.adj() += adj * dx3;
    });

    // Tangent: Hessian is zero (P is linear in its inputs)
    var tangent = yi.d_ + dx * ai.d_ + dx2 * bi.d_ + dx3 * ci.d_;

    return fvar<var>(val, tangent);
}

// ============================================================================
// CubicInterpolation<fvar<var>>::derivativeImpl -- 1 callback var, zero Hessian
//
// P'(x) = a[i] + 2*b[i]*dx + 3*c[i]*dx^2
// Linear in (a[i], b[i], c[i]) => zero Hessian.
// ============================================================================

template <>
inline stan::math::fvar<stan::math::var>
CubicInterpolation<stan::math::fvar<stan::math::var>>::derivativeImpl(
    stan::math::fvar<stan::math::var> x) const {
    using stan::math::fvar;
    using stan::math::make_callback_var;
    using stan::math::var;

    size_t i = this->locate(x);
    if (i >= m_a.size())
        i = m_a.size() - 1;

    double dx = this->extractDouble(x) - this->m_x[i];
    double dx2 = dx * dx;

    const auto& ai = m_a[i];
    const auto& bi = m_b[i];
    const auto& ci = m_c[i];

    double result = ai.val_.val() + dx * (2.0 * bi.val_.val() + 3.0 * ci.val_.val() * dx);

    var val = make_callback_var(result, [&ai, &bi, &ci, dx, dx2](auto& vi) {
        double adj = vi.adj();
        ai.val_.adj() += adj;
        bi.val_.adj() += adj * 2.0 * dx;
        ci.val_.adj() += adj * 3.0 * dx2;
    });

    var tangent = ai.d_ + 2.0 * dx * bi.d_ + 3.0 * dx2 * ci.d_;

    return fvar<var>(val, tangent);
}

} // namespace Math

#endif // INTERPOLATION_STAN_PRIMITIVES_H
