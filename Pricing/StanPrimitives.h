//
// StanPrimitives.h -- Stan AD specializations for pricing structs
//
// Provides analytical adjoint (var) and nested analytical (fvar<var>)
// specializations for Black76 and GBS.
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
// Each 1st-order Greek becomes a callback var whose callback encodes
// the corresponding Hessian row. Total: 4 callbacks + 7 arithmetic nodes.
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

    var dV_dDF_var = make_callback_var(g1.dV_dDF, [DFv, Fv, Kv, volv, d2V_dDF_dF = g2.d2V_dDF_dF,
                                                   d2V_dDF_dK = g2.d2V_dDF_dK,
                                                   d2V_dDF_dvol = g2.d2V_dDF_dvol](auto& vi) {
        double a = vi.adj();
        Fv.adj() += a * d2V_dDF_dF;
        Kv.adj() += a * d2V_dDF_dK;
        volv.adj() += a * d2V_dDF_dvol;
    });

    var dV_dF_var = make_callback_var(g1.dV_dF, [DFv, Fv, Kv, volv, d2V_dDF_dF = g2.d2V_dDF_dF,
                                                 d2V_dF2 = g2.d2V_dF2, d2V_dF_dK = g2.d2V_dF_dK,
                                                 d2V_dF_dvol = g2.d2V_dF_dvol](auto& vi) {
        double a = vi.adj();
        DFv.adj() += a * d2V_dDF_dF;
        Fv.adj() += a * d2V_dF2;
        Kv.adj() += a * d2V_dF_dK;
        volv.adj() += a * d2V_dF_dvol;
    });

    var dV_dK_var = make_callback_var(g1.dV_dK, [DFv, Fv, Kv, volv, d2V_dDF_dK = g2.d2V_dDF_dK,
                                                 d2V_dF_dK = g2.d2V_dF_dK, d2V_dK2 = g2.d2V_dK2,
                                                 d2V_dK_dvol = g2.d2V_dK_dvol](auto& vi) {
        double a = vi.adj();
        DFv.adj() += a * d2V_dDF_dK;
        Fv.adj() += a * d2V_dF_dK;
        Kv.adj() += a * d2V_dK2;
        volv.adj() += a * d2V_dK_dvol;
    });

    var dV_dvol_var =
        make_callback_var(g1.dV_dvol, [DFv, Fv, Kv, volv, d2V_dDF_dvol = g2.d2V_dDF_dvol,
                                       d2V_dF_dvol = g2.d2V_dF_dvol, d2V_dK_dvol = g2.d2V_dK_dvol,
                                       d2V_dvol2 = g2.d2V_dvol2](auto& vi) {
            double a = vi.adj();
            DFv.adj() += a * d2V_dDF_dvol;
            Fv.adj() += a * d2V_dF_dvol;
            Kv.adj() += a * d2V_dK_dvol;
            volv.adj() += a * d2V_dvol2;
        });

    var tangent = dV_dDF_var * DFd + dV_dF_var * Fd + dV_dK_var * Kd + dV_dvol_var * vold;

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
// Each 1st-order Greek becomes a callback var whose callback encodes
// the corresponding Hessian row. Total: 5 callbacks + 14 arithmetic nodes.
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

    // Each callback var encodes one row of the symmetric 5x5 Hessian.
    // Params order: S, K, rDisc, b, vol.

    var dV_dS_var = make_callback_var(
        g1.dV_dS, [Sv, Kv, rDiscv, bv, volv, d2V_dS2 = g2.d2V_dS2, d2V_dS_dK = g2.d2V_dS_dK,
                   d2V_dS_drDisc = g2.d2V_dS_drDisc, d2V_dS_db = g2.d2V_dS_db,
                   d2V_dS_dvol = g2.d2V_dS_dvol](auto& vi) {
            double a = vi.adj();
            Sv.adj() += a * d2V_dS2;
            Kv.adj() += a * d2V_dS_dK;
            rDiscv.adj() += a * d2V_dS_drDisc;
            bv.adj() += a * d2V_dS_db;
            volv.adj() += a * d2V_dS_dvol;
        });

    var dV_dK_var = make_callback_var(
        g1.dV_dK, [Sv, Kv, rDiscv, bv, volv, d2V_dS_dK = g2.d2V_dS_dK, d2V_dK2 = g2.d2V_dK2,
                   d2V_drDisc_dK = g2.d2V_drDisc_dK, d2V_dK_db = g2.d2V_dK_db,
                   d2V_dK_dvol = g2.d2V_dK_dvol](auto& vi) {
            double a = vi.adj();
            Sv.adj() += a * d2V_dS_dK;
            Kv.adj() += a * d2V_dK2;
            rDiscv.adj() += a * d2V_drDisc_dK;
            bv.adj() += a * d2V_dK_db;
            volv.adj() += a * d2V_dK_dvol;
        });

    var dV_drDisc_var = make_callback_var(
        g1.dV_drDisc,
        [Sv, Kv, rDiscv, bv, volv, d2V_dS_drDisc = g2.d2V_dS_drDisc,
         d2V_drDisc_dK = g2.d2V_drDisc_dK, d2V_drDisc2 = g2.d2V_drDisc2,
         d2V_drDisc_db = g2.d2V_drDisc_db, d2V_drDisc_dvol = g2.d2V_drDisc_dvol](auto& vi) {
            double a = vi.adj();
            Sv.adj() += a * d2V_dS_drDisc;
            Kv.adj() += a * d2V_drDisc_dK;
            rDiscv.adj() += a * d2V_drDisc2;
            bv.adj() += a * d2V_drDisc_db;
            volv.adj() += a * d2V_drDisc_dvol;
        });

    var dV_db_var =
        make_callback_var(g1.dV_db, [Sv, Kv, rDiscv, bv, volv, d2V_dS_db = g2.d2V_dS_db,
                                     d2V_dK_db = g2.d2V_dK_db, d2V_drDisc_db = g2.d2V_drDisc_db,
                                     d2V_db2 = g2.d2V_db2, d2V_db_dvol = g2.d2V_db_dvol](auto& vi) {
            double a = vi.adj();
            Sv.adj() += a * d2V_dS_db;
            Kv.adj() += a * d2V_dK_db;
            rDiscv.adj() += a * d2V_drDisc_db;
            bv.adj() += a * d2V_db2;
            volv.adj() += a * d2V_db_dvol;
        });

    var dV_dvol_var = make_callback_var(
        g1.dV_dvol, [Sv, Kv, rDiscv, bv, volv, d2V_dS_dvol = g2.d2V_dS_dvol,
                     d2V_dK_dvol = g2.d2V_dK_dvol, d2V_drDisc_dvol = g2.d2V_drDisc_dvol,
                     d2V_db_dvol = g2.d2V_db_dvol, d2V_dvol2 = g2.d2V_dvol2](auto& vi) {
            double a = vi.adj();
            Sv.adj() += a * d2V_dS_dvol;
            Kv.adj() += a * d2V_dK_dvol;
            rDiscv.adj() += a * d2V_drDisc_dvol;
            bv.adj() += a * d2V_db_dvol;
            volv.adj() += a * d2V_dvol2;
        });

    var tangent = dV_dS_var * Sd + dV_dK_var * Kd + dV_drDisc_var * rDiscd + dV_db_var * bd +
                  dV_dvol_var * vold;

    return fvar<var>(price_var, tangent);
}

} // namespace Pricing

#endif // STANPRIMITIVES_H
