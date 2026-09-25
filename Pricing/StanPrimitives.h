//
// StanPrimitives.h -- Stan AD specializations for pricing structs
//
// Provides analytical adjoint (var) and nested analytical (fvar<var>)
// specializations for Black76 and GBS.
//

#ifndef STANPRIMITIVES_H
#define STANPRIMITIVES_H

#include "Math/StanMath.h"

#include "BlackScholes.h"

namespace Pricing {

// ============================================================================
// Black76<var>::price() -- analytical adjoint, 1 tape node
// ============================================================================

template <>
inline stan::math::var Black76<stan::math::var>::price() const {
    using stan::math::make_callback_var;
    using stan::math::var;

    const double D0 = DF.val(), F0 = F.val(), K0 = K.val(), vol0 = vol.val();
    auto res = black76Analytical(D0, F0, K0, vol0, T, type);

    return make_callback_var(res.price, [this, g1 = res.g1](auto& vi) {
        const double adj = vi.adj();
        DF.adj() += adj * g1.dV_dDF;
        F.adj() += adj * g1.dV_dF;
        K.adj() += adj * g1.dV_dK;
        vol.adj() += adj * g1.dV_dvol;
    });
}

// ============================================================================
// Black76<fvar<var>>::price() -- nested analytical for stan::math::hessian
//
// The tangent var = sum_i Greek_i . tangent_i is built as ONE callback var:
// its value is the directional derivative (exact, in double) and its adjoint
// pushes sum_i tangent_i . (Hessian row i) = H . d into the leaves -- exactly
// what hessian()'s column walk needs. 2 callbacks + 0 arithmetic nodes,
// instead of the 4-callback / 7-node fan-out (identical adjoint semantics).
// ============================================================================

template <>
inline stan::math::fvar<stan::math::var> Black76<stan::math::fvar<stan::math::var>>::price() const {
    using stan::math::fvar;
    using stan::math::make_callback_var;
    using stan::math::var;

    var DFv = DF.val_, Fv = F.val_, Kv = K.val_, volv = vol.val_;
    var DFd = DF.d_, Fd = F.d_, Kd = K.d_, vold = vol.d_;

    const double DF0 = DFv.val(), F0 = Fv.val(), K0 = Kv.val(), vol0 = volv.val();
    auto res = black76Analytical(DF0, F0, K0, vol0, T, type);
    const auto& g1 = res.g1;
    const auto& g2 = res.g2;

    var price_var(res.price);

    // Tangent weights (the fvar tangent components, used linearly)
    const double wD = DFd.val(), wF = Fd.val(), wK = Kd.val(), wV = vold.val();
    const double tval = g1.dV_dDF * wD + g1.dV_dF * wF + g1.dV_dK * wK + g1.dV_dvol * wV;

    stan::math::vari* pD = DFv.vi_;
    stan::math::vari* pF = Fv.vi_;
    stan::math::vari* pK = Kv.vi_;
    stan::math::vari* pV = volv.vi_;

    var tangent = make_callback_var(tval, [pD, pF, pK, pV, wD, wF, wK, wV, g2](auto& vi) {
        const double a = vi.adj();
        pD->adj_ += a * (wF * g2.d2V_dDF_dF + wK * g2.d2V_dDF_dK + wV * g2.d2V_dDF_dvol);
        pF->adj_ +=
            a * (wD * g2.d2V_dDF_dF + wF * g2.d2V_dF2 + wK * g2.d2V_dF_dK + wV * g2.d2V_dF_dvol);
        pK->adj_ +=
            a * (wD * g2.d2V_dDF_dK + wF * g2.d2V_dF_dK + wK * g2.d2V_dK2 + wV * g2.d2V_dK_dvol);
        pV->adj_ += a * (wD * g2.d2V_dDF_dvol + wF * g2.d2V_dF_dvol + wK * g2.d2V_dK_dvol +
                         wV * g2.d2V_dvol2);
    });

    return fvar<var>(price_var, tangent);
}

// ============================================================================
// GBS<var>::price() -- analytical adjoint, 1 tape node
// ============================================================================

template <>
inline stan::math::var GBS<stan::math::var>::price() const {
    using stan::math::make_callback_var;
    using stan::math::var;

    const double S0 = S.val(), K0 = K.val(), rDisc0 = rDisc.val(), b0 = b.val(), vol0 = vol.val();

    auto res = gbsAnalytical(S0, K0, rDisc0, b0, vol0, T, type);

    return make_callback_var(res.price, [this, g1 = res.g1](auto& vi) {
        const double adj = vi.adj();
        S.adj() += adj * g1.dV_dS;
        K.adj() += adj * g1.dV_dK;
        rDisc.adj() += adj * g1.dV_drDisc;
        b.adj() += adj * g1.dV_db;
        vol.adj() += adj * g1.dV_dvol;
    });
}

// ============================================================================
// GBS<fvar<var>>::price() -- nested analytical for stan::math::hessian
//
// Single-tangent-callback form (see Black76<fvar<var>> above): the tangent's
// adjoint pushes H . d = sum_i tangent_i . (Hessian row i) into the leaves.
// 2 callbacks + 0 arithmetic nodes instead of the 5-callback / 14-node
// fan-out; identical adjoint semantics.
// ============================================================================

template <>
inline stan::math::fvar<stan::math::var> GBS<stan::math::fvar<stan::math::var>>::price() const {
    using stan::math::fvar;
    using stan::math::make_callback_var;
    using stan::math::var;

    var Sv = S.val_, Kv = K.val_, rDiscv = rDisc.val_, bv = b.val_, volv = vol.val_;
    var Sd = S.d_, Kd = K.d_, rDiscd = rDisc.d_, bd = b.d_, vold = vol.d_;

    const double S0 = Sv.val(), K0 = Kv.val(), rDisc0 = rDiscv.val(), b0 = bv.val(),
                 vol0 = volv.val();

    auto res = gbsAnalytical(S0, K0, rDisc0, b0, vol0, T, type);
    const auto& g1 = res.g1;
    const auto& g2 = res.g2;

    var price_var(res.price);

    const double wS = Sd.val(), wK = Kd.val(), wR = rDiscd.val(), wB = bd.val(), wV = vold.val();
    const double tval =
        g1.dV_dS * wS + g1.dV_dK * wK + g1.dV_drDisc * wR + g1.dV_db * wB + g1.dV_dvol * wV;

    stan::math::vari* pS = Sv.vi_;
    stan::math::vari* pK = Kv.vi_;
    stan::math::vari* pR = rDiscv.vi_;
    stan::math::vari* pB = bv.vi_;
    stan::math::vari* pV = volv.vi_;

    var tangent = make_callback_var(tval, [pS, pK, pR, pB, pV, wS, wK, wR, wB, wV, g2](auto& vi) {
        const double a = vi.adj();
        pS->adj_ += a * (wS * g2.d2V_dS2 + wK * g2.d2V_dS_dK + wR * g2.d2V_dS_drDisc +
                         wB * g2.d2V_dS_db + wV * g2.d2V_dS_dvol);
        pK->adj_ += a * (wS * g2.d2V_dS_dK + wK * g2.d2V_dK2 + wR * g2.d2V_drDisc_dK +
                         wB * g2.d2V_dK_db + wV * g2.d2V_dK_dvol);
        pR->adj_ += a * (wS * g2.d2V_dS_drDisc + wK * g2.d2V_drDisc_dK + wR * g2.d2V_drDisc2 +
                         wB * g2.d2V_drDisc_db + wV * g2.d2V_drDisc_dvol);
        pB->adj_ += a * (wS * g2.d2V_dS_db + wK * g2.d2V_dK_db + wR * g2.d2V_drDisc_db +
                         wB * g2.d2V_db2 + wV * g2.d2V_db_dvol);
        pV->adj_ += a * (wS * g2.d2V_dS_dvol + wK * g2.d2V_dK_dvol + wR * g2.d2V_drDisc_dvol +
                         wB * g2.d2V_db_dvol + wV * g2.d2V_dvol2);
    });

    return fvar<var>(price_var, tangent);
}

} // namespace Pricing

#endif // STANPRIMITIVES_H
