//
// Created by Ozenc Gungor on 5.04.2026.
//

#ifndef BLACKSCHOLES_H
#define BLACKSCHOLES_H

namespace Pricing {

enum class OptionType { Call, Put };

// Black-76: pricing in (DF, F, K, vol, T) parameterization.
// DF = discount factor, F = forward, K = strike, vol = volatility, T = time (double).
template <typename DoubleT>
struct Black76 {
    DoubleT DF, F, K, vol;
    double T;
    OptionType type;

    DoubleT price() const;
};

// Generalized Black-Scholes: pricing in (S, K, rDisc, b, vol, T) parameterization.
// S = spot, K = strike, rDisc = discounting rate (e.g. OIS),
// b = cost-of-carry rate (r_fund - q in the martingale measure),
// vol = volatility, T = time (double).
// Delegates to Black76 with F = S*exp(b*T), DF = exp(-rDisc*T).
template <typename DoubleT>
struct GBS {
    DoubleT S, K, rDisc, b, vol;
    double T;
    OptionType type;

    DoubleT price() const;
};

} // namespace Pricing

#include "BlackScholesDetail.h"

#endif // BLACKSCHOLES_H
