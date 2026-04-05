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
    double dV_dD = 0.0;     // discount sensitivity
    double dV_dF = 0.0;     // delta (forward delta)
    double dV_dK = 0.0;     // strike sensitivity
    double dV_dsigma = 0.0; // vega
};

struct Black76Greeks2 {
    // d2V/dD2 = 0 always (price is linear in D)
    double d2V_dD_dF = 0.0;
    double d2V_dD_dK = 0.0;
    double d2V_dD_dsigma = 0.0;
    double d2V_dF2 = 0.0; // gamma (forward gamma)
    double d2V_dF_dK = 0.0;
    double d2V_dF_dsigma = 0.0; // vanna
    double d2V_dK2 = 0.0;
    double d2V_dK_dsigma = 0.0;
    double d2V_dsigma2 = 0.0; // volga
};

struct Black76Result {
    double price = 0.0;
    Black76Greeks1 g1;
    Black76Greeks2 g2;
};

// ============================================================================
// black76Analytical -- full 1st and 2nd order Greeks in double
//
// Black-76: C = D * (F*N(d+) - K*N(d-))
//   d+ = x/s + s/2,  d- = x/s - s/2,  x = ln(F/K),  s = sigma*sqrt(T)
// Key identity: F*n(d+) = K*n(d-)
// All 2nd derivatives are identical for call and put.
// ============================================================================

inline Black76Result black76Analytical(double D, double F, double K, double sigma, double T,
                                       OptionType type) {
    const double sqrtT = std::sqrt(T);
    const double s = sigma * sqrtT;
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
    const double price = D * V_undsc;

    // 1st order
    const double dVu_dF = is_call ? Ndp : (Ndp - 1.0);
    const double dVu_dK = is_call ? -Ndm : (1.0 - Ndm);
    const double dVu_dsigma = F * ndp * sqrtT;

    Black76Greeks1 g1;
    g1.price = price;
    g1.dV_dD = V_undsc;
    g1.dV_dF = D * dVu_dF;
    g1.dV_dK = D * dVu_dK;
    g1.dV_dsigma = D * dVu_dsigma;

    // 2nd order
    const double inv_Fs = 1.0 / (F * s);
    const double inv_Ks = 1.0 / (K * s);

    Black76Greeks2 g2;
    g2.d2V_dD_dF = dVu_dF;
    g2.d2V_dD_dK = dVu_dK;
    g2.d2V_dD_dsigma = dVu_dsigma;
    g2.d2V_dF2 = D * ndp * inv_Fs;
    g2.d2V_dF_dK = D * (-ndp * inv_Ks);
    g2.d2V_dF_dsigma = D * (-ndp * dm / sigma);
    g2.d2V_dK2 = D * ndm * inv_Ks;
    g2.d2V_dK_dsigma = D * ndm * dp / sigma;
    g2.d2V_dsigma2 = D * F * sqrtT * ndp * dp * dm / sigma;

    return {price, g1, g2};
}

// ============================================================================
// Black76<double>::price()
// ============================================================================

template <>
inline double Black76<double>::price() const {
    const double sqrtT = std::sqrt(T);
    const double s = sigma * sqrtT;
    const double x = std::log(F / K);
    const double dp = x / s + 0.5 * s;
    const double dm = dp - s;

    const double Ndp = detail::Phi(dp);
    const double Ndm = detail::Phi(dm);

    if (type == OptionType::Call) {
        return D * (F * Ndp - K * Ndm);
    } else {
        return D * (K * (1.0 - Ndm) - F * (1.0 - Ndp));
    }
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
    DoubleT D = exp(-rDisc * T);
    return Black76<DoubleT>{D, F, K, sigma, T, type}.price();
}

} // namespace Pricing

#endif // BLACKSCHOLESDETAIL_H
