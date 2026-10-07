#pragma once

#include "quantape/datetime/BusinessDayConvention.h"
#include "quantape/datetime/Calendar.h"
#include "quantape/datetime/Date.h"
#include "quantape/datetime/DayCounter.h"
#include "quantape/datetime/Period.h"
#include "quantape/instruments/IrInstruments.h"
#include "quantape/pricing/XccyMath.h"

#include <string_view>

namespace quantape::instruments {
/**
 * @file XccyInstruments.h
 * @brief Plain-data cross-currency basis swap with constant or resetting
 *        notionals
 *
 * The instrument carries both legs' conventions plus the precomputed coupon
 * times (`prepare`) that turn dates into zero times once, so a Brent trial
 * only reads discount factors. The member `impliedQuote<ScalarT>` forwards to
 * the scalar algebra in `pricing/XccyMath.h` over a cross-currency curve set
 * and handles constant-notional and resetting-notional quotes alike.
 *
 * `date()` is the risk maturity adjusted on the foreign leg's calendar,
 * `target()` the quoted basis spread and `kindName()` the short label. The
 * coupled-system pillar pairs the basis swap with its foreign IBOR-vs-OIS
 * basis swap.
 */

/// Cross-currency basis swap: both legs exchange notional at start and
/// maturity, the quoted spread sits on the foreign or domestic leg per
/// `spreadOnForeignLeg`, and the notional is either constant or resets on the
/// selected leg per `resetForeignLeg`.
struct XccyBasisSwap {
    datetime::Date maturity{};
    double spread = 0.0;            ///< Basis spread (decimal)
    bool spreadOnForeignLeg = true; ///< Which leg carries the quoted basis
    /// Notional convention. Analytic stack risk rows cover `Const` pillars
    /// only; resetting pillars need reset-aware rows.
    pricing::XccyNotionalMode notional = pricing::XccyNotionalMode::Const;
    /// `MtM` only: which leg's notional resets (the domestic leg when false).
    /// The reset notional at an accrual start is the opposite leg's
    /// forward-implied level `D_other(tStart)/D_own(tStart)`. Ignored for
    /// `XccyNotionalMode::Const`.
    bool resetForeignLeg = true;
    datetime::Period foreignTenor{3, datetime::TimeUnit::Months};
    datetime::Calendar foreignCalendar{};
    datetime::DayCounter foreignDayCounter{datetime::DayCount::Actual360};
    datetime::Period domesticTenor{3, datetime::TimeUnit::Months};
    datetime::Calendar domesticCalendar{};
    datetime::DayCounter domesticDayCounter{datetime::DayCount::Actual360};
    int foreignPaymentLag = 0; ///< Coupon payment lag in business days
    int domesticPaymentLag = 0;
    datetime::BusinessDayConvention foreignBusinessDayConvention =
        datetime::BusinessDayConvention::ModifiedFollowing;
    datetime::BusinessDayConvention domesticBusinessDayConvention =
        datetime::BusinessDayConvention::ModifiedFollowing;
    pricing::XccyPillarTimes times{};

    /// Risk maturity: the quoted termination rolled on the foreign calendar.
    datetime::Date date() const {
        return foreignCalendar.adjust(maturity, foreignBusinessDayConvention);
    }

    /// Quoted basis spread.
    double target() const { return spread; }

    /// Short kind label.
    static constexpr std::string_view kindName() noexcept { return "Xccy"; }

    /// Precomputes both legs' coupon times with the caller's zero clock.
    XccyBasisSwap& prepare(const datetime::Date& referenceDate,
                           const datetime::DayCounter& zeroDayCounter) {
        times = pricing::makeXccyPillarTimes(*this, referenceDate, zeroDayCounter);
        return *this;
    }

    /// Model basis spread over a cross-currency curve set (`XccySet`): the
    /// foreign and domestic discount members, each with its forecast member.
    template <typename ScalarT, typename CurveSetT>
    ScalarT impliedQuote(const CurveSetT& curves) const {
        return pricing::xccyParSpread<ScalarT>(
            times, spreadOnForeignLeg, curves.foreignDiscountCurve, curves.foreignForecastCurve,
            curves.domesticDiscountCurve, curves.domesticForecastCurve);
    }
};

/// One coupled pillar: the foreign par basis quote (discounted on the xccy
/// curve) and the xccy basis quote (forecast on the foreign spread curve).
struct XccyCoupledPillar {
    XccyBasisSwap xccy;
    BasisSwap basis; ///< Foreign IBOR-vs-OIS par basis, xccy-collateral discounting
};

} // namespace quantape::instruments
