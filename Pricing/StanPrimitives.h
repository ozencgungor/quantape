//
// StanPrimitives.h -- Stan AD specializations for pricing structs
//
// Provides analytical adjoint (var) and nested analytical (fvar<var>)
// specializations for Black76. GBS delegates to Black76, so it gets
// these specializations automatically via the chain rule.
//

#ifndef STANPRIMITIVES_H
#define STANPRIMITIVES_H

#include <stan/math.hpp>

#include "BlackScholes.h"

namespace Pricing {

// ============================================================================
// Black76<var>::price() -- analytical adjoint, 1 tape node
// ============================================================================

template <>
inline stan::math::var Black76<stan::math::var>::price() const {
    using stan::math::make_callback_var;
    using stan::math::var;

    const double D0 = D.val(), F0 = F.val(), K0 = K.val(), sigma0 = sigma.val();
    auto res = black76Analytical(D0, F0, K0, sigma0, T, type);

    return make_callback_var(res.price, [this, g1 = res.g1](auto& vi) {
        const double adj = vi.adj();
        D.adj() += adj * g1.dV_dD;
        F.adj() += adj * g1.dV_dF;
        K.adj() += adj * g1.dV_dK;
        sigma.adj() += adj * g1.dV_dsigma;
    });
}

// ============================================================================
// Black76<fvar<var>>::price() -- nested analytical for stan::math::hessian
//
// Each 1st-order Greek becomes a callback var whose callback encodes
// the corresponding Hessian row. Total: 4 callbacks + 7 arithmetic nodes.
// ============================================================================

template <>
inline stan::math::fvar<stan::math::var> Black76<stan::math::fvar<stan::math::var>>::price() const {
    using stan::math::fvar;
    using stan::math::make_callback_var;
    using stan::math::var;

    var Dv = D.val_, Fv = F.val_, Kv = K.val_, sv = sigma.val_;
    var Dd = D.d_, Fd = F.d_, Kd = K.d_, sd = sigma.d_;

    const double D0 = Dv.val(), F0 = Fv.val(), K0 = Kv.val(), s0 = sv.val();
    auto res = black76Analytical(D0, F0, K0, s0, T, type);
    const auto& g1 = res.g1;
    const auto& g2 = res.g2;

    var price_var(res.price);

    var dV_dD_var = make_callback_var(g1.dV_dD, [Dv, Fv, Kv, sv, d2V_dD_dF = g2.d2V_dD_dF,
                                                 d2V_dD_dK = g2.d2V_dD_dK,
                                                 d2V_dD_ds = g2.d2V_dD_dsigma](auto& vi) {
        double a = vi.adj();
        Fv.adj() += a * d2V_dD_dF;
        Kv.adj() += a * d2V_dD_dK;
        sv.adj() += a * d2V_dD_ds;
    });

    var dV_dF_var = make_callback_var(g1.dV_dF, [Dv, Fv, Kv, sv, d2V_dD_dF = g2.d2V_dD_dF,
                                                 d2V_dF2 = g2.d2V_dF2, d2V_dF_dK = g2.d2V_dF_dK,
                                                 d2V_dF_ds = g2.d2V_dF_dsigma](auto& vi) {
        double a = vi.adj();
        Dv.adj() += a * d2V_dD_dF;
        Fv.adj() += a * d2V_dF2;
        Kv.adj() += a * d2V_dF_dK;
        sv.adj() += a * d2V_dF_ds;
    });

    var dV_dK_var = make_callback_var(g1.dV_dK, [Dv, Fv, Kv, sv, d2V_dD_dK = g2.d2V_dD_dK,
                                                 d2V_dF_dK = g2.d2V_dF_dK, d2V_dK2 = g2.d2V_dK2,
                                                 d2V_dK_ds = g2.d2V_dK_dsigma](auto& vi) {
        double a = vi.adj();
        Dv.adj() += a * d2V_dD_dK;
        Fv.adj() += a * d2V_dF_dK;
        Kv.adj() += a * d2V_dK2;
        sv.adj() += a * d2V_dK_ds;
    });

    var dV_ds_var = make_callback_var(
        g1.dV_dsigma, [Dv, Fv, Kv, sv, d2V_dD_ds = g2.d2V_dD_dsigma, d2V_dF_ds = g2.d2V_dF_dsigma,
                       d2V_dK_ds = g2.d2V_dK_dsigma, d2V_ds2 = g2.d2V_dsigma2](auto& vi) {
            double a = vi.adj();
            Dv.adj() += a * d2V_dD_ds;
            Fv.adj() += a * d2V_dF_ds;
            Kv.adj() += a * d2V_dK_ds;
            sv.adj() += a * d2V_ds2;
        });

    var tangent = dV_dD_var * Dd + dV_dF_var * Fd + dV_dK_var * Kd + dV_ds_var * sd;

    return fvar<var>(price_var, tangent);
}

} // namespace Pricing

#endif // STANPRIMITIVES_H
