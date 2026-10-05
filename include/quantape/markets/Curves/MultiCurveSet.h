#pragma once

#include "quantape/markets/Curves/Curve.h"
#include "quantape/markets/Curves/CurveConfig.h"

#include <cstddef>
#include <deque>
#include <memory>
#include <stdexcept>
#include <string_view>
#include <utility>
#include <vector>

namespace quantape::markets {
/**
 * @file MultiCurveSet.h
 * @brief Lookup container for the output of `buildStack`
 *
 * Curves are added once after the stack build, then resolved by setup code
 * into raw references/spans for the hot path. Lookup is a linear scan (a
 * handful of curves); the hot path never looks up a curve by key. Entries live
 * in stable storage, so references returned by `find`, `discountCurve` and
 * `curves` stay valid while further curves are added.
 */
class MultiCurveSet {
public:
    /// Register the output of `buildStack` in one call. The build order is
    /// irrelevant: every lookup is by full `CurveKey`.
    static MultiCurveSet fromBuiltCurves(std::vector<BuiltCurve> curves) {
        MultiCurveSet set;
        for (BuiltCurve& curve : curves) {
            set.add(std::move(curve));
        }
        return set;
    }

    /// Add one built curve; the entry must carry a non-null handle and a key
    /// not already present.
    void add(BuiltCurve curve) {
        if (!curve.curve) {
            throw std::invalid_argument("MultiCurveSet::add: null curve handle");
        }
        ensureUnique(curve.key);
        m_curves.push_back(std::move(curve));
    }

    std::size_t size() const { return m_curves.size(); }
    const std::deque<BuiltCurve>& curves() const { return m_curves; }

    /// Resolve a curve entry by full `CurveKey` (currency, role, index tenor
    /// and collateral). Throws `std::out_of_range` when the key is unknown.
    const BuiltCurve& find(const CurveKey& key) const {
        for (const BuiltCurve& entry : m_curves) {
            if (entry.key == key) {
                return entry;
            }
        }
        throw std::out_of_range("MultiCurveSet::find: unknown curve key");
    }

    /// Root discount curve for a currency when exactly one discount curve
    /// exists for it. Throws `std::out_of_range` if there is none and
    /// `std::invalid_argument` if several differ only by collateral.
    const std::shared_ptr<const CurveHandle>& discountCurve(std::string_view currency) const {
        const std::shared_ptr<const CurveHandle>* found = nullptr;
        for (const BuiltCurve& entry : m_curves) {
            if (entry.role == CurveRole::Discount && entry.key.currency == currency) {
                if (found != nullptr) {
                    throw std::invalid_argument(
                        "MultiCurveSet::discountCurve: multiple discount curves for currency; "
                        "specify the collateral currency");
                }
                found = &entry.curve;
            }
        }
        if (found == nullptr) {
            throw std::out_of_range("MultiCurveSet::discountCurve: unknown currency");
        }
        return *found;
    }

    /// Discount curve for a currency collateralized in `collateral`.
    const std::shared_ptr<const CurveHandle>& discountCurve(std::string_view currency,
                                                            std::string_view collateral) const {
        for (const BuiltCurve& entry : m_curves) {
            if (entry.role == CurveRole::Discount && entry.key.currency == currency &&
                entry.key.collateral == collateral) {
                return entry.curve;
            }
        }
        throw std::out_of_range("MultiCurveSet::discountCurve: unknown currency/collateral");
    }

private:
    void ensureUnique(const CurveKey& key) const {
        for (const BuiltCurve& entry : m_curves) {
            if (entry.key == key) {
                throw std::invalid_argument("MultiCurveSet::add: duplicate curve key");
            }
        }
    }

    std::deque<BuiltCurve> m_curves;
};

} // namespace quantape::markets
