#pragma once

#include "quantape/datetime/Date.h"
#include "quantape/markets/Curves/DiscountCurve.h"
#include "quantape/math/Solvers/BrentSolver.h"

#include <cmath>
#include <concepts>
#include <cstddef>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <variant>
#include <vector>

namespace quantape::markets {
/**
 * @file BootstrapInstrument.h
 * @brief Concept-based bootstrap vocabulary: curve sets, instruments, ladder
 *
 * A bootstrap instrument is plain data with a risk date, a quoted target and a
 * scalar-templated model quote over a lightweight curve-set aggregate. The
 * homogeneous bootstrap paths monomorphize directly over such instruments;
 * heterogeneous ladders close over a `std::variant` through `Ladder` and
 * `std::visit`, so dispatch stays compile-time with no vtable and no heap.
 *
 * The curve sets are reference aggregates rebuilt per Brent trial, so a
 * residual never allocates. The same `impliedQuote<ScalarT>` member template
 * serves the `double` bootstrap, `var` pricing and `fvar<var>` gamma tests.
 */

/// Discount factors read from one curve: `DiscountSet{curve}`.
template <typename DiscountT>
struct DiscountSet {
    const DiscountT& curve;

    /// Discount factor at `t`.
    decltype(auto) discount(double t) const { return curve.discount(t); }
};

/// Forecast plus discounting providers: `ForecastSet{forecast, discount}`.
template <typename ForecastT, typename DiscountT>
struct ForecastSet {
    const ForecastT& forecastCurve;
    const DiscountT& discountCurve;

    /// Forecast discount factor at `t`.
    decltype(auto) forecast(double t) const { return forecastCurve.discount(t); }
    /// Discount factor at `t`.
    decltype(auto) discount(double t) const { return discountCurve.discount(t); }
};

/// Cross-currency provider set: foreign and domestic discount plus forecast
/// curves, in the argument order of the cross-currency leg evaluators.
template <typename ForeignDiscountT, typename ForeignForecastT, typename DomesticDiscountT,
          typename DomesticForecastT>
struct XccySet {
    const ForeignDiscountT& foreignDiscountCurve;
    const ForeignForecastT& foreignForecastCurve;
    const DomesticDiscountT& domesticDiscountCurve;
    const DomesticForecastT& domesticForecastCurve;

    /// Foreign discount factor at `t`.
    decltype(auto) foreignDiscount(double t) const { return foreignDiscountCurve.discount(t); }
    /// Foreign forecast discount factor at `t`.
    decltype(auto) foreignForecast(double t) const { return foreignForecastCurve.discount(t); }
    /// Domestic discount factor at `t`.
    decltype(auto) domesticDiscount(double t) const { return domesticDiscountCurve.discount(t); }
    /// Domestic forecast discount factor at `t`.
    decltype(auto) domesticForecast(double t) const { return domesticForecastCurve.discount(t); }
};

/// Base/quote provider set: `FxSet{base, quote}`.
template <typename BaseT, typename QuoteT>
struct FxSet {
    const BaseT& baseCurve;
    const QuoteT& quoteCurve;

    decltype(auto) base(double t) const { return baseCurve.discount(t); }
    decltype(auto) quote(double t) const { return quoteCurve.discount(t); }
};

/// A curve set that can answer the discount factor at a time in `ScalarT`.
template <typename Set, typename ScalarT>
concept DiscountCurveSet = requires(const Set& set, double t) {
    { set.discount(t) } -> std::convertible_to<ScalarT>;
};

/// Plain-data bootstrap instrument: risk date, quoted target and a model
/// quote over a curve set, all readable on a const instrument.
template <typename Instrument, typename CurveSet, typename ScalarT = double>
concept BootstrapInstrument = requires(const Instrument& instrument, const CurveSet& curves) {
    { instrument.date() } -> std::same_as<datetime::Date>;
    { instrument.target() } -> std::same_as<double>;
    { instrument.template impliedQuote<ScalarT>(curves) } -> std::same_as<ScalarT>;
};

/// Closed-set heterogeneous adaptor: one value at a time out of a fixed
/// instrument set, concept-implemented by `std::visit`.
template <typename... InstrumentTs>
class Ladder {
public:
    /// Closed variant of the ladder's instrument alternatives.
    using Instrument = std::variant<InstrumentTs...>;

    /// Builds a single-value ladder from one instrument alternative.
    explicit Ladder(Instrument instrument) : m_instrument(std::move(instrument)) {}

    /// Stored instrument alternative.
    const Instrument& instrument() const { return m_instrument; }

    /// Risk date of the stored alternative.
    datetime::Date date() const {
        return std::visit([](const auto& instrument) { return instrument.date(); }, m_instrument);
    }

    /// Quoted target of the stored alternative.
    double target() const {
        return std::visit([](const auto& instrument) { return instrument.target(); }, m_instrument);
    }

    /// Model quote of the stored alternative over `curves`.
    template <typename ScalarT, typename CurveSetT>
    ScalarT impliedQuote(const CurveSetT& curves) const {
        return std::visit(
            [&curves](const auto& instrument) -> ScalarT {
                return instrument.template impliedQuote<ScalarT>(curves);
            },
            m_instrument);
    }

private:
    Instrument m_instrument;
};

namespace detail {

/// Sequential exact-fit node solver shared by the FX-swap and cross-currency
/// bootstraps: one Brent solve per node against a trial curve moved in place
/// from the current nodes, then a triangular Gauss-Seidel re-pass when the
/// first pass leaves residual error. `residual(i, curve)` returns the model
/// quote minus its target. Node 0 (t = 0) is pinned to zero. One trial curve
/// is built per node solve, so the Brent steps only move the solved node
/// through `CurveTrialUpdater` instead of rebuilding the grid each time.
template <typename Residual>
std::vector<double>
bootstrapNodesByBrent(InterpolationSpace space, InterpolationScheme scheme, double tension,
                      int switchIndex, double accuracy, const std::vector<double>& nodeTimes,
                      const Residual& residual, std::string_view context = "bootstrap") {
    const std::size_t count = nodeTimes.size();
    const quantape::math::BrentSolver<double> solver;
    std::vector<double> zeros(count, 0.0);
    std::vector<double> trialTimes;
    std::vector<double> trialZeros;
    const auto worstResidual = [&]() {
        trialTimes.assign(1, 0.0);
        trialZeros.assign(1, 0.0);
        if (trialTimes.capacity() < count + 1) {
            trialTimes.reserve(count + 1);
            trialZeros.reserve(count + 1);
        }
        for (std::size_t j = 0; j < count; ++j) {
            trialTimes.push_back(nodeTimes[j]);
            trialZeros.push_back(zeros[j]);
        }
        const DiscountCurve<double> curve(trialTimes, trialZeros, space, scheme, tension,
                                          switchIndex);
        double worst = 0.0;
        for (std::size_t i = 0; i < count; ++i) {
            const double check = residual(i, curve);
            if (!std::isfinite(check)) {
                worst = 1e300;
            } else if (std::abs(check) > worst) {
                worst = std::abs(check);
            }
        }
        return worst;
    };
    const auto solveNodes = [&](bool multiPass) {
        for (int pass = 0; pass < (multiPass ? 50 : 1); ++pass) {
            const std::vector<double> previous = zeros;
            double lastMove = 0.0;
            for (std::size_t i = 0; i < count; ++i) {
                const std::size_t lastNode = multiPass && pass == 0 ? i : count - 1;
                // The trial curve is rebuilt once per node solve and then only
                // the solved node moves; the scheme state stays valid because
                // the node grid is fixed for the whole solve.
                std::optional<DiscountCurve<double>> trialCurve;
                bool cacheValid = false;
                double cachedZero = 0.0;
                double cachedResidual = 0.0;
                const auto objective = [&](double trialZero) {
                    if (cacheValid && trialZero == cachedZero) {
                        return cachedResidual;
                    }
                    if (!trialCurve.has_value()) {
                        trialTimes.assign(1, 0.0);
                        trialZeros.assign(1, 0.0);
                        if (trialTimes.capacity() < lastNode + 2) {
                            trialTimes.reserve(lastNode + 2);
                            trialZeros.reserve(lastNode + 2);
                        }
                        for (std::size_t j = 0; j <= lastNode; ++j) {
                            trialTimes.push_back(nodeTimes[j]);
                            trialZeros.push_back(zeros[j]);
                        }
                        trialCurve.emplace(trialTimes, trialZeros, space, scheme, tension,
                                           switchIndex);
                    }
                    CurveTrialUpdater::setNode(*trialCurve, i + 1, trialZero);
                    cachedResidual = residual(i, *trialCurve);
                    cachedZero = trialZero;
                    cacheValid = true;
                    return cachedResidual;
                };
                const double guess = i == 0 ? 0.0 : zeros[i - 1];
                double lower = guess - 0.5;
                double upper = guess + 0.5;
                double fLower = objective(lower);
                double fUpper = objective(upper);
                int widen = 0;
                while (fLower * fUpper > 0.0 && widen < 12) {
                    lower -= 0.5;
                    upper += 0.5;
                    fLower = objective(lower);
                    fUpper = objective(upper);
                    ++widen;
                }
                if (!(fLower * fUpper <= 0.0) || !std::isfinite(fLower) || !std::isfinite(fUpper)) {
                    throw std::runtime_error(std::string(context) + ": failed to bracket pillar " +
                                             std::to_string(i));
                }
                const double root = solver.solve(objective, accuracy, guess, lower, upper);
                const double move = std::abs(root - previous[i]);
                if (move > lastMove) {
                    lastMove = move;
                }
                zeros[i] = root;
            }
            if (!multiPass || lastMove < 1e-15) {
                break;
            }
        }
    };
    solveNodes(false);
    if (!(worstResidual() < 1e-9)) {
        solveNodes(true);
    }
    if (!(worstResidual() < 1e-9)) {
        throw std::runtime_error(std::string(context) + ": fixed point did not converge");
    }
    return zeros;
}

} // namespace detail

} // namespace quantape::markets
