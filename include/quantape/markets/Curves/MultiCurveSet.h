#pragma once

#include "quantape/markets/Curves/Curve.h"
#include "quantape/markets/Curves/DiscountCurve.h"
#include "quantape/markets/Curves/SpreadCurve.h"

#include <deque>
#include <memory>
#include <stdexcept>
#include <string_view>

namespace quantape::markets {
/**
 * @file MultiCurveSet.h
 * @brief Registry of curves keyed by `CurveKey`
 *
 * Cold-path container: curves are added once during the build, then resolved
 * by setup code into raw references/spans for the hot path. Lookup is a linear
 * scan (a handful of curves); the hot path never looks up a curve by key.
 *
 * @tparam DoubleT Numeric type (`double` or an AD scalar).
 */
template <typename DoubleT>
class MultiCurveSet {
public:
    struct CurveEntry {
        CurveKey key;
        std::shared_ptr<const DiscountCurve<DoubleT>> discount;
        std::shared_ptr<const SpreadCurve<DoubleT>> spread;
    };

    void add(const CurveKey& key, std::shared_ptr<const DiscountCurve<DoubleT>> curve) {
        if (!curve) {
            throw std::invalid_argument("MultiCurveSet::add: null curve");
        }
        ensureUnique(key);
        m_curves.push_back(CurveEntry{key, std::move(curve), nullptr});
    }

    void add(const CurveKey& key, std::shared_ptr<const SpreadCurve<DoubleT>> curve) {
        if (!curve) {
            throw std::invalid_argument("MultiCurveSet::add: null curve");
        }
        ensureUnique(key);
        m_curves.push_back(CurveEntry{key, nullptr, std::move(curve)});
    }

    std::size_t size() const { return m_curves.size(); }
    const std::deque<CurveEntry>& curves() const { return m_curves; }

    /// Resolve a curve entry by key (linear scan; cold path).
    ///
    /// Entries live in stable storage, so the returned reference stays valid
    /// when further curves are added.
    const CurveEntry& find(const CurveKey& key) const {
        for (const CurveEntry& entry : m_curves) {
            if (entry.key == key) {
                return entry;
            }
        }
        throw std::out_of_range("MultiCurveSet::find: unknown curve key");
    }

    /// Root discount curve for a currency when exactly one discount curve
    /// exists for it. Throws `std::out_of_range` if there is none and
    /// `std::invalid_argument` if several differ only by collateral.
    const std::shared_ptr<const DiscountCurve<DoubleT>>&
    discountCurve(std::string_view currency) const {
        const std::shared_ptr<const DiscountCurve<DoubleT>>* found = nullptr;
        for (const CurveEntry& entry : m_curves) {
            if (entry.discount && entry.key.currency == currency &&
                entry.key.role == CurveRole::Discount) {
                if (found != nullptr) {
                    throw std::invalid_argument(
                        "MultiCurveSet::discountCurve: multiple discount curves for currency; "
                        "specify the collateral currency");
                }
                found = &entry.discount;
            }
        }
        if (found == nullptr) {
            throw std::out_of_range("MultiCurveSet::discountCurve: unknown currency");
        }
        return *found;
    }

    /// Discount curve for a currency collateralized in `collateral`.
    const std::shared_ptr<const DiscountCurve<DoubleT>>&
    discountCurve(std::string_view currency, std::string_view collateral) const {
        for (const CurveEntry& entry : m_curves) {
            if (entry.discount && entry.key.currency == currency &&
                entry.key.role == CurveRole::Discount && entry.key.collateral == collateral) {
                return entry.discount;
            }
        }
        throw std::out_of_range("MultiCurveSet::discountCurve: unknown currency/collateral");
    }

private:
    void ensureUnique(const CurveKey& key) const {
        for (const CurveEntry& entry : m_curves) {
            if (entry.key == key) {
                throw std::invalid_argument("MultiCurveSet::add: duplicate curve key");
            }
        }
    }

    std::deque<CurveEntry> m_curves;
};

} // namespace quantape::markets
