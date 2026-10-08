#pragma once

#include "quantape/datetime/Period.h"
#include "quantape/markets/Curves/Curve.h"
#include "quantape/markets/Curves/CurveConfig.h"
#include "quantape/markets/Curves/CurveOnGrid.h"
#include "quantape/markets/Curves/MultiCurveSet.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace quantape::sde {
/**
 * @file FrozenMarket.h
 * @brief Immutable multi-curve stack snapshot for the SDE layer
 *
 * Holds the built curves of a market stack (OIS base, spread/forecast chains)
 * and the configured FX spot views in stable storage. Access is pointer and
 * index-scan only: a construction-time hash index on the currency groups the
 * entries, so a lookup binary-searches flat 16-byte slots and scans only the
 * (few) curves of that currency: no allocation, no string construction, no
 * curve object required at the call site. Construction is the single
 * allocating step; the snapshot is never mutated afterwards, and the shared
 * curve handles keep the providers alive independently of the source stack.
 */
class FrozenMarket {
public:
    /// One stack curve: identity, role and the type-erased provider.
    struct CurveEntry {
        markets::CurveKey key;
        markets::CurveRole role = markets::CurveRole::Discount;
        std::shared_ptr<const markets::CurveHandle> curve;
    };

    /// FX spot view: quote units per base unit plus the base/quote providers
    /// (the covered-interest-parity forward uses both discount curves).
    struct FxSpotView {
        std::string pair; ///< base+quote, e.g. "EURUSD"
        double spot = 0.0;
        std::shared_ptr<const markets::CurveHandle> base;
        std::shared_ptr<const markets::CurveHandle> quote;

        /// Forward FX rate at `t`: spot * D_base(t) / D_quote(t).
        double forward(double t) const {
            if (!base || !quote) {
                throw std::runtime_error("FrozenMarket::FxSpotView::forward: missing curve");
            }
            return spot * base->discount(t) / quote->discount(t);
        }
    };

    FrozenMarket() = default;

    /// Snapshot of explicit stack entries; throws on a null handle or a
    /// duplicate curve key.
    explicit FrozenMarket(std::vector<CurveEntry> curves, std::vector<FxSpotView> fxSpots = {});

    /// Snapshot of a built set: the shared handles are copied, so the
    /// snapshot outlives `set`. `fxSpots` carries the FX views resolved by
    /// the caller.
    static FrozenMarket fromStack(const markets::MultiCurveSet& set,
                                  std::vector<FxSpotView> fxSpots = {});

    /// Snapshot the output of `markets::buildStack` directly.
    static FrozenMarket fromCurves(std::vector<markets::BuiltCurve> curves,
                                   std::vector<FxSpotView> fxSpots = {});

    std::size_t size() const noexcept { return m_curves.size(); }
    const std::vector<CurveEntry>& curves() const noexcept { return m_curves; }
    const std::vector<FxSpotView>& fxSpots() const noexcept { return m_fxSpots; }

    /// Curve by full key; nullptr when absent. Index scan, no allocation.
    const markets::CurveHandle* curve(const markets::CurveKey& key) const noexcept;

    /// The one discount curve of `currency`; nullptr when absent or ambiguous
    /// (several collateral variants).
    const markets::CurveHandle* discountCurve(std::string_view currency) const noexcept;

    /// Discount curve of `currency` collateralized in `collateral`.
    const markets::CurveHandle* discountCurve(std::string_view currency,
                                              std::string_view collateral) const noexcept;

    /// Forecast curve for `currency` and index tenor; nullptr when absent.
    const markets::CurveHandle* forecastCurve(std::string_view currency,
                                              const datetime::Period& indexTenor) const noexcept;

    /// FX spot view for `pair` ("EURUSD"); nullptr when absent.
    const FxSpotView* fxSpot(std::string_view pair) const noexcept;

    /// Materialize a stack curve on `times` (one interpolation pass); the
    /// frozen arrays are the state-independent base of the evaluation seam.
    /// Throws `std::out_of_range` for an unknown key.
    markets::CurveOnGrid<double> freeze(const markets::CurveKey& key,
                                        const std::vector<double>& times) const;

private:
    /// One flat lookup slot: the currency hash plus the curve entry index.
    /// Slots are sorted by hash, so a lookup binary-searches 16-byte records
    /// and then scans the equal-hash range; the exact currency (and key) is
    /// always compared, so a hash collision can never produce a wrong match.
    struct HashSlot {
        std::uint64_t hash = 0;
        std::uint32_t index = 0;
    };

    /// Slots whose currency hashes to the query's hash; the caller scans the
    /// range and compares the exact key.
    std::pair<const HashSlot*, const HashSlot*>
    currencyRange(std::string_view currency) const noexcept;

    std::vector<CurveEntry> m_curves;
    std::vector<FxSpotView> m_fxSpots;
    std::vector<HashSlot> m_byHash;
};

inline FrozenMarket::FrozenMarket(std::vector<CurveEntry> curves, std::vector<FxSpotView> fxSpots)
    : m_curves(std::move(curves)), m_fxSpots(std::move(fxSpots)) {
    for (std::size_t i = 0; i < m_curves.size(); ++i) {
        if (!m_curves[i].curve) {
            throw std::invalid_argument("FrozenMarket: null curve handle");
        }
        for (std::size_t j = i + 1; j < m_curves.size(); ++j) {
            if (m_curves[i].key == m_curves[j].key) {
                throw std::invalid_argument("FrozenMarket: duplicate curve key");
            }
        }
    }
    m_byHash.reserve(m_curves.size());
    for (std::size_t i = 0; i < m_curves.size(); ++i) {
        m_byHash.push_back(HashSlot{std::hash<std::string_view>{}(m_curves[i].key.currency),
                                    static_cast<std::uint32_t>(i)});
    }
    std::sort(m_byHash.begin(), m_byHash.end(),
              [](const HashSlot& left, const HashSlot& right) { return left.hash < right.hash; });
}

inline std::pair<const FrozenMarket::HashSlot*, const FrozenMarket::HashSlot*>
FrozenMarket::currencyRange(std::string_view currency) const noexcept {
    const std::uint64_t hash = std::hash<std::string_view>{}(currency);
    const auto lower = std::lower_bound(
        m_byHash.begin(), m_byHash.end(), hash,
        [](const HashSlot& slot, std::uint64_t value) { return slot.hash < value; });
    const auto upper = std::upper_bound(
        lower, m_byHash.end(), hash,
        [](std::uint64_t value, const HashSlot& slot) { return value < slot.hash; });
    return {m_byHash.data() + (lower - m_byHash.begin()),
            m_byHash.data() + (upper - m_byHash.begin())};
}

inline FrozenMarket FrozenMarket::fromStack(const markets::MultiCurveSet& set,
                                            std::vector<FxSpotView> fxSpots) {
    std::vector<CurveEntry> entries;
    entries.reserve(set.size());
    for (const markets::BuiltCurve& built : set.curves()) {
        entries.push_back(CurveEntry{built.key, built.role, built.curve});
    }
    return FrozenMarket(std::move(entries), std::move(fxSpots));
}

inline FrozenMarket FrozenMarket::fromCurves(std::vector<markets::BuiltCurve> curves,
                                             std::vector<FxSpotView> fxSpots) {
    std::vector<CurveEntry> entries;
    entries.reserve(curves.size());
    for (markets::BuiltCurve& built : curves) {
        entries.push_back(CurveEntry{std::move(built.key), built.role, std::move(built.curve)});
    }
    return FrozenMarket(std::move(entries), std::move(fxSpots));
}

inline const markets::CurveHandle*
FrozenMarket::curve(const markets::CurveKey& key) const noexcept {
    const auto [first, last] = currencyRange(key.currency);
    for (const HashSlot* it = first; it != last; ++it) {
        const CurveEntry& entry = m_curves[it->index];
        if (entry.key == key) {
            return entry.curve.get();
        }
    }
    return nullptr;
}

inline const markets::CurveHandle*
FrozenMarket::discountCurve(std::string_view currency) const noexcept {
    const auto [first, last] = currencyRange(currency);
    const markets::CurveHandle* found = nullptr;
    for (const HashSlot* it = first; it != last; ++it) {
        const CurveEntry& entry = m_curves[it->index];
        if (entry.key.currency != currency || entry.role != markets::CurveRole::Discount) {
            continue;
        }
        if (found != nullptr) {
            return nullptr; // ambiguous: several collateral variants
        }
        found = entry.curve.get();
    }
    return found;
}

inline const markets::CurveHandle*
FrozenMarket::discountCurve(std::string_view currency, std::string_view collateral) const noexcept {
    const auto [first, last] = currencyRange(currency);
    for (const HashSlot* it = first; it != last; ++it) {
        const CurveEntry& entry = m_curves[it->index];
        if (entry.key.currency == currency && entry.role == markets::CurveRole::Discount &&
            entry.key.collateral == collateral) {
            return entry.curve.get();
        }
    }
    return nullptr;
}

inline const markets::CurveHandle*
FrozenMarket::forecastCurve(std::string_view currency,
                            const datetime::Period& indexTenor) const noexcept {
    const auto [first, last] = currencyRange(currency);
    for (const HashSlot* it = first; it != last; ++it) {
        const CurveEntry& entry = m_curves[it->index];
        if (entry.key.currency == currency && entry.role == markets::CurveRole::Forecast &&
            entry.key.indexTenor == indexTenor) {
            return entry.curve.get();
        }
    }
    return nullptr;
}

inline const FrozenMarket::FxSpotView* FrozenMarket::fxSpot(std::string_view pair) const noexcept {
    for (const FxSpotView& view : m_fxSpots) {
        if (view.pair == pair) {
            return &view;
        }
    }
    return nullptr;
}

inline markets::CurveOnGrid<double> FrozenMarket::freeze(const markets::CurveKey& key,
                                                         const std::vector<double>& times) const {
    const markets::CurveHandle* handle = curve(key);
    if (handle == nullptr) {
        throw std::out_of_range("FrozenMarket::freeze: unknown curve key");
    }
    return markets::materialize(*handle, times);
}

} // namespace quantape::sde
