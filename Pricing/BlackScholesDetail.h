//
// BlackScholesDetail.h -- analytical derivatives (pure double) and price implementations
//
// Included at the bottom of BlackScholes.h. Do not include directly.
//

#ifndef BLACKSCHOLESDETAIL_H
#define BLACKSCHOLESDETAIL_H

#include <cmath>

namespace Pricing {

// ============================================================================
// Helpers
// ============================================================================

namespace detail {

inline double phi(double x) {
    return std::exp(-0.5 * x * x) / std::sqrt(2.0 * M_PI);
}

inline double Phi(double x) {
    return 0.5 * std::erfc(-x * M_SQRT1_2);
}

} // namespace detail

// ============================================================================
// Analytical Greeks result types (Black-76 parameterization)
// ============================================================================

struct Black76Greeks1 {
    double price = 0.0;
    double dV_dDF = 0.0;  // discount sensitivity
    double dV_dF = 0.0;   // delta (forward delta)
    double dV_dK = 0.0;   // strike sensitivity
    double dV_dvol = 0.0; // vega
};

struct Black76Greeks2 {
    // d2V/dDF2 = 0 always (price is linear in D)
    double d2V_dDF_dF = 0.0;
    double d2V_dDF_dK = 0.0;
    double d2V_dDF_dvol = 0.0;
    double d2V_dF2 = 0.0; // gamma (forward gamma)
    double d2V_dF_dK = 0.0;
    double d2V_dF_dvol = 0.0; // vanna
    double d2V_dK2 = 0.0;
    double d2V_dK_dvol = 0.0;
    double d2V_dvol2 = 0.0; // volga
};

struct Black76Result {
    double price = 0.0;
    Black76Greeks1 g1;
    Black76Greeks2 g2;
};

struct GBSGreeks1 {
    double price = 0.0;
    double dV_dS = 0.0;     // spot delta
    double dV_drDisc = 0.0; // discount sensitivity
    double dV_dK = 0.0;     // strike sensitivity
    double dV_db = 0.0;     // carry sensitivity
    double dV_dvol = 0.0;   // vega
};

struct GBSGreeks2 {
    double d2V_dS2 = 0.0; // spot gamma
    double d2V_dS_drDisc = 0.0;
    double d2V_dS_dK = 0.0;
    double d2V_dS_db = 0.0;
    double d2V_dS_dvol = 0.0;
    double d2V_drDisc2 = 0.0;
    double d2V_drDisc_dK = 0.0;
    double d2V_drDisc_db = 0.0;
    double d2V_drDisc_dvol = 0.0;
    double d2V_dK2 = 0.0;
    double d2V_dK_db = 0.0;
    double d2V_dK_dvol = 0.0;
    double d2V_db2 = 0.0;
    double d2V_db_dvol = 0.0;
    double d2V_dvol2 = 0.0;
};

struct GBSResult {
    double price = 0.0;
    GBSGreeks1 g1;
    GBSGreeks2 g2;
};

// ============================================================================
// black76Analytical -- full 1st and 2nd order Greeks in double
//
// Black-76: V = DF * (F*N(d+) - K*N(d-))
//   d+ = x/s + s/2,  d- = x/s - s/2,  x = ln(F/K),  s = vol*sqrt(T)
// Key identity: F*n(d+) = K*n(d-)
// All 2nd derivatives are identical for call and put.
// ============================================================================

inline Black76Result black76Analytical(double DF, double F, double K, double vol, double T,
                                       OptionType type) {
    const double sqrtT = std::sqrt(T);
    const double s = vol * sqrtT;
    const double x = std::log(F / K);
    const double half_s = 0.5 * s;
    const double dp = x / s + half_s;
    const double dm = dp - s;

    const double Ndp = detail::Phi(dp);
    const double Ndm = detail::Phi(dm);
    const double ndp = detail::phi(dp);
    const double ndm = detail::phi(dm);

    const double call_undsc = F * Ndp - K * Ndm;
    const double put_undsc = K * (1.0 - Ndm) - F * (1.0 - Ndp);

    const bool is_call = (type == OptionType::Call);
    const double V_undsc = is_call ? call_undsc : put_undsc;
    const double price = DF * V_undsc;

    // 1st order
    const double dVu_dF = is_call ? Ndp : (Ndp - 1.0);
    const double dVu_dK = is_call ? -Ndm : (1.0 - Ndm);
    const double dVu_dvol = F * ndp * sqrtT;

    Black76Greeks1 g1;
    g1.price = price;
    g1.dV_dDF = V_undsc;
    g1.dV_dF = DF * dVu_dF;
    g1.dV_dK = DF * dVu_dK;
    g1.dV_dvol = DF * dVu_dvol;

    // 2nd order
    const double inv_Fs = 1.0 / (F * s);
    const double inv_Ks = 1.0 / (K * s);

    Black76Greeks2 g2;
    g2.d2V_dDF_dF = dVu_dF;
    g2.d2V_dDF_dK = dVu_dK;
    g2.d2V_dDF_dvol = dVu_dvol;
    g2.d2V_dF2 = DF * ndp * inv_Fs;
    g2.d2V_dF_dK = DF * (-ndp * inv_Ks);
    g2.d2V_dF_dvol = DF * (-ndp * dm / vol);
    g2.d2V_dK2 = DF * ndm * inv_Ks;
    g2.d2V_dK_dvol = DF * ndm * dp / vol;
    g2.d2V_dvol2 = DF * F * sqrtT * ndp * dp * dm / vol;

    return {price, g1, g2};
}

// ============================================================================
// Black76<double>::price()
// ============================================================================

template <>
inline double Black76<double>::price() const {
    const double sqrtT = std::sqrt(T);
    const double s = vol * sqrtT;
    const double x = std::log(F / K);
    const double dp = x / s + 0.5 * s;
    const double dm = dp - s;

    const double Ndp = detail::Phi(dp);
    const double Ndm = detail::Phi(dm);

    if (type == OptionType::Call) {
        return DF * (F * Ndp - K * Ndm);
    } else {
        return DF * (K * (1.0 - Ndm) - F * (1.0 - Ndp));
    }
}

// ============================================================================
// gbsAnalytical -- full 1st and 2nd order Greeks in double
//
// GBS delegates to Black-76 via F = S*exp(b*T), DF = exp(-rDisc*T).
// All GBS Greeks are obtained by chain-ruling through the B76 Greeks.
//
// Intermediate derivatives:
//   dF/dS = exp(b*T),  dF/db = T*F,  d2F/db2 = T^2*F,  d2F/dS_db = T*exp(b*T)
//   dDF/drDisc = -T*DF, d2DF/drDisc2 = T^2*DF
//   d2V/dDF2 = 0 (price is linear in DF)
// ============================================================================

inline GBSResult gbsAnalytical(double S, double K, double rDisc, double b, double vol, double T,
                               OptionType type) {
    const double ebT = std::exp(b * T);
    const double F = S * ebT;
    const double DF = std::exp(-rDisc * T);

    // Intermediate derivatives: GBS params -> B76 params
    const double dF_dS = ebT;
    const double dF_db = T * F;
    const double d2F_db2 = T * T * F;
    const double d2F_dS_db = T * ebT;
    // dF/dS^2 = 0 (F is linear in S)

    const double dDF_dr = -T * DF;
    const double d2DF_dr2 = T * T * DF;

    // B76 Greeks
    const auto b76 = black76Analytical(DF, F, K, vol, T, type);
    const auto& bg1 = b76.g1;
    const auto& bg2 = b76.g2;

    // === 1st order ===
    // dV/dS = (dV/dF)(dF/dS)
    // dV/drDisc = (dV/dDF)(dDF/drDisc)
    // dV/db = (dV/dF)(dF/db)
    // dV/dK, dV/dvol pass through directly

    GBSGreeks1 g1;
    g1.price = b76.price;
    g1.dV_dS = bg1.dV_dF * dF_dS;
    g1.dV_dK = bg1.dV_dK;
    g1.dV_dvol = bg1.dV_dvol;
    g1.dV_drDisc = bg1.dV_dDF * dDF_dr;
    g1.dV_db = bg1.dV_dF * dF_db;

    // === 2nd order ===
    // Chain rule: d2V/d(xi)(xj) where xi, xj in {S, rDisc, K, b, vol}
    // mapped through DF(rDisc), F(S,b), K, vol.

    GBSGreeks2 g2;

    // d2V/dS2 = (d2V/dF2)(dF/dS)^2
    // (no d2F/dS2 term since F is linear in S)
    g2.d2V_dS2 = bg2.d2V_dF2 * dF_dS * dF_dS;

    // d2V/dS_drDisc = (d2V/dDF_dF)(dDF/drDisc)(dF/dS)
    g2.d2V_dS_drDisc = bg2.d2V_dDF_dF * dDF_dr * dF_dS;

    // d2V/dS_dK = (d2V/dF_dK)(dF/dS)
    g2.d2V_dS_dK = bg2.d2V_dF_dK * dF_dS;

    // d2V/dS_db = (d2V/dF2)(dF/dS)(dF/db) + (dV/dF)(d2F/dS_db)
    g2.d2V_dS_db = bg2.d2V_dF2 * dF_dS * dF_db + bg1.dV_dF * d2F_dS_db;

    // d2V/dS_dvol = (d2V/dF_dvol)(dF/dS)
    g2.d2V_dS_dvol = bg2.d2V_dF_dvol * dF_dS;

    // d2V/drDisc2 = (dV/dDF)(d2DF/drDisc2)
    // (d2V/dDF2 = 0 since V is linear in DF)
    g2.d2V_drDisc2 = bg1.dV_dDF * d2DF_dr2;

    // d2V/drDisc_dK = (d2V/dDF_dK)(dDF/drDisc)
    g2.d2V_drDisc_dK = bg2.d2V_dDF_dK * dDF_dr;

    // d2V/drDisc_db = (d2V/dDF_dF)(dDF/drDisc)(dF/db)
    g2.d2V_drDisc_db = bg2.d2V_dDF_dF * dDF_dr * dF_db;

    // d2V/drDisc_dvol = (d2V/dDF_dvol)(dDF/drDisc)
    g2.d2V_drDisc_dvol = bg2.d2V_dDF_dvol * dDF_dr;

    // d2V/dK2 = d2V/dK2  (direct from B76)
    g2.d2V_dK2 = bg2.d2V_dK2;

    // d2V/dK_db = (d2V/dF_dK)(dF/db)
    g2.d2V_dK_db = bg2.d2V_dF_dK * dF_db;

    // d2V/dK_dvol = d2V/dK_dvol  (direct from B76)
    g2.d2V_dK_dvol = bg2.d2V_dK_dvol;

    // d2V/db2 = (d2V/dF2)(dF/db)^2 + (dV/dF)(d2F/db2)
    g2.d2V_db2 = bg2.d2V_dF2 * dF_db * dF_db + bg1.dV_dF * d2F_db2;

    // d2V/db_dvol = (d2V/dF_dvol)(dF/db)
    g2.d2V_db_dvol = bg2.d2V_dF_dvol * dF_db;

    // d2V/dvol2 = d2V/dvol2  (direct from B76)
    g2.d2V_dvol2 = bg2.d2V_dvol2;

    return {b76.price, g1, g2};
}

// ============================================================================
// GBS<DoubleT>::price() -- generic, delegates to Black76
//
// F = S * exp(b * T),  D = exp(-r_disc * T)
// b = cost-of-carry (r_fund - q), r_disc = discounting rate (e.g. OIS)
// Stan AD chains through exp/multiply automatically.
// ============================================================================

template <typename DoubleT>
DoubleT GBS<DoubleT>::price() const {
    using std::exp;
    DoubleT F = S * exp(b * T);
    DoubleT DF = exp(-rDisc * T);

    return Black76<DoubleT>{DF, F, K, vol, T, type}.price();
}

} // namespace Pricing

#endif // BLACKSCHOLESDETAIL_H
