#pragma once

#include "quantape/sde/Measure.h"

#include <concepts>
#include <span>
#include <stdexcept>
#include <string>

namespace quantape::sde {
/**
 * @file MeasureAdjustment.h
 * @brief Drift correction from the canonical SDE to a requested measure
 *
 * A model states the additive drift term that maps its canonical dynamics to
 * the dynamics under the requested pricing measure and currency. Consumers
 * ask `supports` first and gate through `requireMeasureSupport`, so a run
 * never integrates the canonical drift under a measure the model cannot
 * adjust to.
 */

/// Contract: `supports(measure)` says whether the model can express the drift
/// under `measure`; `measureAdjustment(measure, t, state)` returns the additive
/// correction for time `t` and factor `state`, so the dynamic under `measure`
/// is the canonical drift plus this term. Evaluation is scalar-generic:
/// `double`, `var` and `fvar<var>` flow through the caller's scalar type.
template <typename Model, typename Scalar>
concept MeasureAdjustmentModel = requires(const Model& model, const MeasureSpec& measure, double t,
                                          std::span<const Scalar> state) {
    { model.supports(measure) } -> std::same_as<bool>;
    { model.measureAdjustment(measure, t, state) } -> std::same_as<Scalar>;
};

namespace detail {

inline const char* measureKindName(MeasureKind kind) noexcept {
    switch (kind) {
        case MeasureKind::RiskNeutral:
            return "RiskNeutral";
        case MeasureKind::TForward:
            return "TForward";
        case MeasureKind::Annuity:
            return "Annuity";
        case MeasureKind::CollateralCurrency:
            return "CollateralCurrency";
    }
    return "Unknown";
}

} // namespace detail

/// Throws `std::invalid_argument` naming the unsupported measure, so a run
/// fails loudly instead of simulating under the wrong drift.
template <typename Model>
void requireMeasureSupport(const Model& model, const MeasureSpec& measure) {
    if (!model.supports(measure)) {
        throw std::invalid_argument(std::string("requireMeasureSupport: unsupported measure ") +
                                    detail::measureKindName(measure.kind));
    }
}

} // namespace quantape::sde
