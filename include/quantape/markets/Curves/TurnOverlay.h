#pragma once

#include "quantape/datetime/Date.h"
#include "quantape/datetime/DayCounter.h"
#include "quantape/markets/Curves/DiscountCurve.h"
#include "quantape/markets/Curves/SpreadCurve.h"

#include <stan/math/rev/core/nested_rev_autodiff.hpp>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <numeric>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

namespace quantape::markets {
/**
 * @file TurnOverlay.h
 * @brief Additive turn-of-year overlay over a yield curve (Overlay mode)
 *
 * A turn is a persistent discount-factor multiplier: for a turn of amplitude
 * `d` at `tTurn`, all zero rates beyond the turn gain `d * tTurn / t`, so the
 * discount factor is multiplied by `exp(-d * tTurn)`. In a sampled FRA curve
 * the overlay shows up as a bump over the accrual periods that span the turn,
 * so the bump width follows the instrument tenor.
 *
 * A `Bump` models a finite elevated-funding window instead: the instantaneous
 * forward is raised by `amplitude` over `[begin, end]`. A sampled FRA rate
 * then gains `amplitude` while its accrual sits inside the window and decays
 * with the overlap on the way out, which reproduces the flat-top year-end
 * spikes seen across several tenors from a single window and amplitude. Both
 * terms are additive on top of the wrapped curve's zero rates.
 *
 * Turn amplitudes are inputs here; they are not calibrated by this overlay.
 * They are `DoubleT` scalars, so a `var`/`fvar<var>` overlay carries the
 * amplitudes as AD parameters and the exogenous turn risk `dV/d(amplitude)`
 * follows from reverse-mode AD instead of bump-and-reprice. The direct-knot
 * alternative tags turn knots on the bootstrap curve instead: there the turn
 * factor is an ordinary curve quote, while an overlay amplitude is an
 * exogenous factor outside the quote Jacobian.
 *
 * The overlay satisfies the `CurveProvider` concept, can wrap any curve
 * provider (`DiscountCurve` or `SpreadCurve`), and is meant to be
 * materialized once per run like any other curve.
 *
 * @tparam DoubleT Numeric type (`double` or an AD scalar).
 * @tparam BaseT Wrapped curve type (defaults to `DiscountCurve<DoubleT>`).
 */
template <typename DoubleT, typename BaseT = DiscountCurve<DoubleT>>
class TurnOverlay {
public:
    /// Flat forward bump over `[begin, end]`: an elevated funding window. The
    /// amplitude is the stored AD parameter.
    struct Bump {
        double begin;
        double end;
        DoubleT amplitude;
    };

    /// One turn point: `(turn time, amplitude)`.
    using Turn = std::pair<double, DoubleT>;

    TurnOverlay(std::shared_ptr<const BaseT> base, std::vector<Turn> turns,
                std::vector<Bump> bumps = {})
        : m_base(std::move(base)), m_bumps(std::move(bumps)) {
        if (!m_base) {
            throw std::invalid_argument("TurnOverlay: null base curve");
        }
        m_turns.reserve(turns.size());
        for (const auto& turn : turns) {
            if (!(turn.first > 0.0)) {
                throw std::invalid_argument("TurnOverlay: turn time must be positive");
            }
            m_turns.push_back(turn);
        }
        std::sort(m_turns.begin(), m_turns.end(),
                  [](const auto& a, const auto& b) { return a.first < b.first; });
        for (const Bump& bump : m_bumps) {
            if (!(bump.begin > 0.0) || !(bump.end > bump.begin)) {
                throw std::invalid_argument("TurnOverlay: bump window must be positive");
            }
        }
        std::sort(m_bumps.begin(), m_bumps.end(),
                  [](const Bump& a, const Bump& b) { return a.begin < b.begin; });
    }

    /// Zero rate with the turn jumps and funding bumps accumulated.
    DoubleT zero(double t) const {
        if (t <= 0.0) {
            return m_base->zero(t);
        }
        DoubleT jump = 0;
        for (const auto& turn : m_turns) {
            if (turn.first <= t) {
                jump += turn.second * (turn.first / t);
            }
        }
        for (const Bump& bump : m_bumps) {
            const double active = activeWidth(bump, t);
            if (active > 0.0) {
                jump += bump.amplitude * (active / t);
            }
        }
        return m_base->zero(t) + jump;
    }

    DoubleT discount(double t) const {
        if (t <= 0.0) {
            return DoubleT(1);
        }
        return expImpl(-zero(t) * t);
    }

    DoubleT forward(double t1, double t2) const {
        if (!(t2 > t1)) {
            throw std::invalid_argument("TurnOverlay::forward: t2 must be > t1");
        }
        const DoubleT x1 = zero(t1) * t1;
        const DoubleT x2 = zero(t2) * t2;
        return (x2 - x1) / (t2 - t1);
    }

    const BaseT& base() const { return *m_base; }
    const std::vector<Turn>& turns() const { return m_turns; }
    const std::vector<Bump>& bumps() const { return m_bumps; }

private:
    /// Length of `[begin, end]` covered by `[0, t]`.
    static double activeWidth(const Bump& bump, double t) {
        return std::max(0.0, std::min(t, bump.end) - std::min(t, bump.begin));
    }

    static DoubleT expImpl(const DoubleT& x) {
        if constexpr (std::is_same_v<DoubleT, double>) {
            return std::exp(x);
        } else {
            using std::exp;
            return exp(x);
        }
    }

    std::shared_ptr<const BaseT> m_base;
    std::vector<Turn> m_turns; ///< (turn time, amplitude)
    std::vector<Bump> m_bumps; ///< flat forward windows
};

namespace detail {

/// Reconstructs, for every knot time, a pillar date whose year fraction on
/// `zeroDayCounter` equals the stored time exactly, all relative to
/// `referenceDate` (`times.front()` is the curve's synthetic zero node). The
/// exact-equality gate makes a false positive impossible; the search assumes
/// the convention's year fraction does not decrease with the date and returns
/// false otherwise (or when a time is not a whole-day year fraction at all),
/// so the caller falls back to the times constructor.
inline bool recoverPillarDates(const datetime::Date& referenceDate,
                               const datetime::DayCounter& zeroDayCounter,
                               const std::vector<double>& times,
                               std::vector<datetime::Date>& pillarDates) {
    pillarDates.clear();
    pillarDates.reserve(times.size());
    const std::int64_t refSerial = referenceDate.serial();
    const std::int64_t maxOffset =
        static_cast<std::int64_t>(std::numeric_limits<std::int32_t>::max()) - refSerial;
    const auto dateAt = [refSerial](std::int64_t offset) {
        return datetime::Date::fromSerial(static_cast<std::int32_t>(refSerial + offset));
    };
    for (const double t : times) {
        if (t == 0.0) {
            pillarDates.push_back(referenceDate);
            continue;
        }
        if (!(t > 0.0)) {
            return false;
        }
        std::int64_t low = 1;
        std::int64_t high = maxOffset;
        if (zeroDayCounter.yearFraction(referenceDate, dateAt(high)) < t) {
            return false;
        }
        while (low < high) {
            const std::int64_t mid = low + (high - low + 1) / 2;
            if (zeroDayCounter.yearFraction(referenceDate, dateAt(mid)) <= t) {
                low = mid;
            } else {
                high = mid - 1;
            }
        }
        if (zeroDayCounter.yearFraction(referenceDate, dateAt(low)) != t) {
            return false;
        }
        pillarDates.push_back(dateAt(low));
    }
    return true;
}

/// Rebuilds a `DiscountCurve` on `AdScalar` node values with the same node
/// grid, interpolation configuration and date metadata. The knot times are
/// inverted to whole-day pillar dates on the curve's own date clock; when that
/// succeeds the date constructor is used and both `referenceDate()` and
/// `zeroDayCounter()` carry over. When inversion fails (knot times that are
/// not exact year fractions on the clock, or a non-monotone convention) the
/// times constructor is used: a times-constructed curve already carries the
/// default metadata, while a date-based curve on such a clock keeps its node
/// grid but loses the date metadata.
template <typename AdScalar>
DiscountCurve<AdScalar> rebindDiscountCurve(const DiscountCurve<double>& curve) {
    std::vector<datetime::Date> pillarDates;
    if (recoverPillarDates(curve.referenceDate(), curve.zeroDayCounter(), curve.times(),
                           pillarDates)) {
        return DiscountCurve<AdScalar>(
            curve.referenceDate(), pillarDates, curve.zeroDayCounter(),
            std::vector<AdScalar>(curve.zeros().begin(), curve.zeros().end()), curve.space(),
            curve.scheme(), curve.tension(), curve.switchIndex());
    }
    return DiscountCurve<AdScalar>(
        curve.times(), std::vector<AdScalar>(curve.zeros().begin(), curve.zeros().end()),
        curve.space(), curve.scheme(), curve.tension(), curve.switchIndex());
}

/// Scalar rebind of a plain double curve provider: same node data, `AdScalar`
/// node values. Used by the turn-overlay risk transform to differentiate
/// amplitudes while every other curve input stays a constant.
template <typename AdScalar, typename CurveT>
struct CurveRebind {
    static_assert(!std::is_same_v<CurveT, CurveT>, "CurveRebind: unsupported curve type");
};

template <typename AdScalar>
struct CurveRebind<AdScalar, DiscountCurve<double>> {
    using Type = DiscountCurve<AdScalar>;

    static Type convert(const DiscountCurve<double>& curve) {
        return rebindDiscountCurve<AdScalar>(curve);
    }
};

/// The parent chain is rebound recursively, so its date metadata survives in
/// full. The internal spread nodes have no date constructor (in either the
/// double or the AD type) and therefore carry the default metadata on both
/// sides; the observable `SpreadCurve::zeroDayCounter()` delegates to the
/// parent and is preserved.
template <typename AdScalar, typename ParentT>
struct CurveRebind<AdScalar, SpreadCurve<double, ParentT>> {
    using AdParent = typename CurveRebind<AdScalar, ParentT>::Type;
    using Type = SpreadCurve<AdScalar, AdParent>;

    static Type convert(const SpreadCurve<double, ParentT>& curve) {
        auto parent = std::make_shared<const AdParent>(
            CurveRebind<AdScalar, ParentT>::convert(curve.parent()));
        const DiscountCurve<double>& nodes = curve.spreadNodes();
        return Type(std::move(parent), nodes.times(),
                    std::vector<AdScalar>(nodes.zeros().begin(), nodes.zeros().end()),
                    nodes.scheme(), nodes.tension());
    }
};

} // namespace detail

/// One exogenous turn risk: the turn amplitude's label and the portfolio
/// sensitivity `dV/d(amplitude)`.
struct TurnRiskEntry {
    std::string label;
    double delta = 0.0;
};

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
