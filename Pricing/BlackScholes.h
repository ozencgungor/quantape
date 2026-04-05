//
// Created by Ozenc Gungor on 5.04.2026.
//

#ifndef BLACKSCHOLES_H
#define BLACKSCHOLES_H

namespace Pricing {

enum class OptionType { Call, Put };

// Black-76: pricing in (D, F, K, sigma, T) parameterization.
// D = discount factor, F = forward, K = strike, sigma = vol, T = time (double).
template <typename DoubleT>
struct Black76 {
    DoubleT D, F, K, sigma;
    double T;
    OptionType type;

    DoubleT price() const;
};

// Generalized Black-Scholes: pricing in (S, K, r_disc, b, sigma, T) parameterization.
// S = spot, K = strike, r_disc = discounting rate (e.g. OIS),
// b = cost-of-carry rate (r_fund - q in the martingale measure),
// sigma = vol, T = time (double).
// Delegates to Black76 with F = S*exp(b*T), D = exp(-r_disc*T).
template <typename DoubleT>
struct GBS {
    DoubleT S, K, rDisc, b, sigma;
    double T;
    OptionType type;

    DoubleT price() const;
};

} // namespace Pricing

#include "BlackScholesDetail.h"

#endif // BLACKSCHOLES_H
