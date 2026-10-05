#pragma once

#include "quantape/markets/Curves/TurnOverlay.h"

#include <stan/math/rev/core/nested_rev_autodiff.hpp>

#include <algorithm>
#include <cstddef>
#include <memory>
#include <numeric>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace quantape::markets {
/**
 * @file TurnOverlayRisk.h
 * @brief Reverse-mode risk through a turn-of-year overlay (Overlay mode)
 *
 * This is the Stan-dependent companion of `TurnOverlay.h`: the overlay itself
 * is a plain curve provider, while `turnOverlayRisk` runs on an isolated
 * nested tape. Keeping the split lets double-only translation units include
 * `TurnOverlay.h` without pulling Stan. The scalar-rebind helpers the transform
 * uses live in `TurnOverlay.h` under `detail`.
 */

/// Reverse-mode turn risk through a turn overlay. The overlay amplitudes are
/// exogenous risk factors outside the curve quote Jacobian: `value` is called
/// once on an overlay whose amplitudes are independent AD parameters, and each
/// returned entry carries the corresponding reverse-mode adjoint. The direct
/// alternative marks the turn knots as ordinary curve quotes (`turnPillar`),
/// so the two representations differ in what a "turn delta" is measured
/// against: a rate factor here, a bootstrapped quote there.
///
/// `labels` match `turns` then `bumps` in input order. Returned entries are a
/// single ascending-time merge of both lists: turn points by turn time,
/// funding windows by begin, with turns ahead of bumps at equal times, so
/// labels travel with their shifted amplitudes.
///
/// Tape ownership: the internal graph runs on its own nested
/// `stan::math::nested_rev_autodiff` scope, so its variables are recovered on
/// return and on throw and never accumulate on the caller's tape. `value` must
/// return a reverse-mode scalar exposing `.grad()` (e.g. `stan::math::var`),
/// not `fvar<var>`; capturing outer reverse-mode values in `value` works but
/// leaves their adjoints to the caller's own accounting.
template <typename AdScalar, typename BaseT, typename ValueFn>
std::vector<TurnRiskEntry>
turnOverlayRisk(const BaseT& base, const std::vector<std::pair<double, double>>& turns,
                const std::vector<typename TurnOverlay<double, BaseT>::Bump>& bumps,
                const std::vector<std::string>& labels, ValueFn&& value) {
    using AdBaseT = typename detail::CurveRebind<AdScalar, BaseT>::Type;
    using AdOverlay = TurnOverlay<AdScalar, AdBaseT>;
    const std::size_t turnCount = turns.size();
    if (labels.size() != turnCount + bumps.size()) {
        throw std::invalid_argument("turnOverlayRisk: label count mismatch");
    }
    stan::math::nested_rev_autodiff nested;
    std::vector<AdScalar> amplitudes;
    amplitudes.reserve(labels.size());
    std::vector<typename AdOverlay::Turn> adTurns;
    adTurns.reserve(turnCount);
    for (std::size_t i = 0; i < turnCount; ++i) {
        amplitudes.push_back(AdScalar(turns[i].second));
        adTurns.emplace_back(turns[i].first, amplitudes.back());
    }
    std::vector<typename AdOverlay::Bump> adBumps;
    adBumps.reserve(bumps.size());
    for (std::size_t j = 0; j < bumps.size(); ++j) {
        amplitudes.push_back(AdScalar(bumps[j].amplitude));
        adBumps.push_back(
            typename AdOverlay::Bump{bumps[j].begin, bumps[j].end, amplitudes.back()});
    }
    std::vector<std::size_t> order(labels.size());
    std::iota(order.begin(), order.end(), std::size_t{0});
    const auto amplitudeTime = [&](std::size_t k) {
        return k < turnCount ? turns[k].first : bumps[k - turnCount].begin;
    };
    std::stable_sort(order.begin(), order.end(), [&](std::size_t a, std::size_t b) {
        return amplitudeTime(a) < amplitudeTime(b);
    });
    auto adBase =
        std::make_shared<const AdBaseT>(detail::CurveRebind<AdScalar, BaseT>::convert(base));
    const AdOverlay overlay(std::move(adBase), std::move(adTurns), std::move(adBumps));
    auto objective = value(overlay);
    objective.grad();
    std::vector<TurnRiskEntry> entries;
    entries.reserve(labels.size());
    for (const std::size_t k : order) {
        entries.push_back(TurnRiskEntry{labels[k], amplitudes[k].adj()});
    }
    return entries;
}

} // namespace quantape::markets
