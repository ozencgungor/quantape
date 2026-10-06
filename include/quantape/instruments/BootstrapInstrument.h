#pragma once

#include "quantape/datetime/Date.h"

#include <concepts>
#include <string_view>
#include <utility>
#include <variant>

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
 *
 * This header is dependency-light on purpose: it pulls only the date library
 * and the standard library, so pricing headers can include it without
 * dragging in a concrete curve type or the root solver.
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

    /// Base discount factor at `t`.
    decltype(auto) base(double t) const { return baseCurve.discount(t); }
    /// Quote discount factor at `t`.
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

} // namespace quantape::markets
