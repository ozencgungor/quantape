#ifndef MARKETDATA_H
#define MARKETDATA_H

/**
 * @file MarketData.h
 * @brief Main header for market data infrastructure
 *
 * This library provides template-based market objects compatible with
 * automatic differentiation (AD) for financial applications.
 *
 * All classes are templated on DoubleT which can be:
 * - double: for regular numerical computations
 * - stan::math::var: for automatic differentiation
 *
 * ## Available Market Objects:
 *
 * ### Curves:
 * - **IRCurve**: Interest rate curve storing zero rates
 * - **YieldCurve**: Yield curve for dividends, repo rates, etc.
 * - **SurvivalProbabilityCurve** (SPCurve): Credit curve for default probabilities
 *
 * ### Volatility Surfaces:
 * - **IRVolatility**: IR vol surface (expiry x tenor) with smile
 * - **EQDVolatility**: Equity vol surface (expiry x strike)
 * - **FXVolatility**: FX vol surface (expiry x delta)
 *
 * ### Market Data:
 * - **EQDData**: Equity spot, dividend curve, discount curve
 * - **FXRate**: FX spot with optional interest rate curves
 *
 * ### Descriptors:
 * - **IRCurveDescriptor**: Metadata for IR curves
 * - **YieldCurveDescriptor**: Metadata for yield curves
 * - **CreditDescriptor**: Metadata for credit curves
 * - **IRVolDescriptor**: Metadata for IR vol surfaces
 * - **EQDDescriptor**: Metadata for equity data
 * - **FXDescriptor**: Metadata for FX data
 *
 * ## Usage Examples:
 *
 * ```cpp
 * // Interest rate curve with zero rates
 * std::vector<double> tenors = {0.5, 1.0, 2.0, 5.0, 10.0};
 * std::vector<double> rates = {0.01, 0.015, 0.02, 0.025, 0.03};
 * Markets::IRCurveDescriptor ircDesc("USD", "OIS", "2024-01-01");
 * Markets::IRCurve<double> oisCurve(tenors, rates, ircDesc);
 *
 * // Get discount factor and forward rate
 * double df = oisCurve.discountFactor(2.5);  // DF at 2.5 years
 * double fwd = oisCurve.forwardRate(1.0, 2.0);  // Forward rate 1y1y
 *
 * // Equity data with spot and curves
 * Markets::EQDDescriptor eqdDesc("SPX", "INDEX", "USD");
 * double spot = 4500.0;
 * Markets::YieldCurve<double> divCurve(tenors, divRates, yieldDesc);
 * Markets::EQDData<double> spxData(spot, divCurve, oisCurve, eqdDesc);
 *
 * // Calculate forward price
 * double forward = spxData.forward(1.0);  // 1-year forward
 *
 * // Volatility surface
 * std::vector<double> expiries = {0.25, 0.5, 1.0, 2.0};
 * std::vector<double> strikes = {4000, 4250, 4500, 4750, 5000};
 * std::vector<std::vector<double>> vols = { ... };  // expiries x strikes
 * Markets::EQDVolatility<double> volSurf(expiries, strikes, vols, spot, eqdDesc);
 *
 * // Get volatility at expiry=1.0, strike=4600
 * double vol = volSurf.vol(1.0, 4600.0);
 *
 * // With automatic differentiation (Stan Math)
 * using ADVariableT = stan::math::var;
 * std::vector<ADVariableT> ad_rates = {0.01, 0.015, 0.02, 0.025, 0.03};
 * Markets::IRCurve<ADVariableT> ad_curve(tenors, ad_rates, ircDesc);
 * ADVariableT ad_df = ad_curve.discountFactor(ADVariableT(2.5));
 *
 * // Compute sensitivity: d(DF)/d(rates)
 * stan::math::grad(ad_df.vi_);
 * // Access gradients via ad_rates[i].adj()
 * ```
 *
 * ## Type Aliases:
 *
 * For convenience:
 * - `SPCurve<DoubleT>` is an alias for `SurvivalProbabilityCurve<DoubleT>`
 * - `SwaptionVolatility<DoubleT>` is an alias for `IRVolatility<DoubleT, IRVolType::Swaption>`
 * - `CapVolatility<DoubleT>` is an alias for `IRVolatility<DoubleT, IRVolType::Cap>`
 */

// Descriptors
#include "Descriptors/CreditDescriptor.h"
#include "Descriptors/EQDDescriptor.h"
#include "Descriptors/FXDescriptor.h"
#include "Descriptors/IRCurveDescriptor.h"
#include "Descriptors/IRVolDescriptor.h"
#include "Descriptors/YieldCurveDescriptor.h"

// Curves
#include "Curves/IRCurve.h"
#include "Curves/SurvivalProbabilityCurve.h"
#include "Curves/YieldCurve.h"

// Volatility Surfaces
#include "Volatility/EQDVolatility.h"
#include "Volatility/FXVolatility.h"
#include "Volatility/IRVolatility.h"

// Market Data
#include "Data/EQDData.h"
#include "Data/FXRate.h"

// Type aliases for convenience
namespace Markets {

/**
 * @brief Alias for SurvivalProbabilityCurve
 */
template <typename DoubleT>
using SPCurve = SurvivalProbabilityCurve<DoubleT>;

/**
 * @brief Alias for Swaption volatility surface
 */
template <typename DoubleT>
using SwaptionVolatility = IRVolatility<DoubleT, IRVolType::Swaption>;

/**
 * @brief Alias for Cap/Floor volatility surface
 */
template <typename DoubleT>
using CapVolatility = IRVolatility<DoubleT, IRVolType::Cap>;

} // namespace Markets

#endif // MARKETDATA_H
