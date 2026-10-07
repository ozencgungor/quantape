#pragma once

#include "quantape/datetime/Date.h"
#include "quantape/datetime/DayCounter.h"
#include "quantape/datetime/Period.h"
#include "quantape/instruments/BootstrapInstrument.h"
#include "quantape/instruments/FxInstruments.h"
#include "quantape/instruments/IrInstruments.h"
#include "quantape/instruments/XccyInstruments.h"
#include "quantape/pricing/IrMath.h"

#include <string>
#include <string_view>
#include <utility>
#include <variant>
#include <vector>

namespace quantape::markets {
/**
 * @file PillarVariants.h
 * @brief Family variants of the bootstrap instruments and the config factories
 *
 * Each family binds the concrete instruments a bootstrap path may consume into
 * one `Ladder` alternative set: discount curves use the money-market
 * instruments plus the OIS tag adapter, forecast curves add the swap
 * instruments, and the cross-currency ladder mixes FX swaps with xccy basis
 * swaps. The factories map a parsed `PillarSpec` to its family variant; their
 * definitions live with the config loader so the parse-time validation
 * messages stay in one translation unit.
 */

struct CurveStackSpec;
struct PillarSpec;
struct XccyCurveSpec;

namespace detail {

/// Adapter that presents one OIS pillar as a plain bootstrap instrument: risk
/// maturity and target from the pillar, the par rate from its prepared coupon
/// times through the shared money-market algebra. The raw pillar fields are
/// kept so a family variant can be re-adapted to the tagged pillar for the
/// `filledPillars` shim.
class OisPillarInstrument {
public:
    OisPillarInstrument() = default;

    OisPillarInstrument(datetime::Date date, double target,
                        std::vector<pricing::OisCouponTimes> coupons)
        : m_date(date), m_target(target), m_coupons(std::move(coupons)) {}

    /// Risk date: calendar-adjusted quoted termination.
    datetime::Date date() const { return m_date; }

    /// Quoted par rate.
    double target() const { return m_target; }

    /// Short kind label.
    static constexpr std::string_view kindName() noexcept { return "OisSwap"; }

    /// Model par rate over a discount curve set.
    template <typename ScalarT, typename CurveSetT>
    ScalarT impliedQuote(const CurveSetT& curves) const {
        return pricing::impliedOisParRate<ScalarT>(m_coupons, curves);
    }

    datetime::Date maturity{}; ///< Quoted maturity
    datetime::Date start{};    ///< Effective date (unset means the reference date)
    datetime::DayCounter quoteDayCounter{datetime::DayCount::Actual360};
    datetime::Calendar calendar{};
    datetime::Period fixedTenor{1, datetime::TimeUnit::Years};
    int paymentLag = 0;
    bool firstCouponFixed = false;
    double firstCouponRate = 0.0;
    datetime::BusinessDayConvention businessDayConvention =
        datetime::BusinessDayConvention::ModifiedFollowing;

private:
    datetime::Date m_date;
    double m_target = 0.0;
    std::vector<pricing::OisCouponTimes> m_coupons;
};

/// Forecast-curve family, kept under its historical `detail` spelling for the
/// existing vocabulary tests.
using ForecastInstrument = Ladder<instruments::Deposit, instruments::Fra, instruments::Future,
                                  instruments::Irs, instruments::BasisSwap>;

} // namespace detail

/// Discount-curve bootstrap family: the four money-market families as
/// plain-data instruments plus the OIS tag adapter.
using DiscountInstrument = Ladder<instruments::Deposit, instruments::Repo, instruments::Fra,
                                  instruments::Future, detail::OisPillarInstrument>;

/// Forecast-curve bootstrap family: money-market instruments plus the par IRS
/// and basis-swap instruments.
using ForecastInstrument = detail::ForecastInstrument;

/// Cross-currency bootstrap family: FX forward points at the short end and
/// cross-currency basis swaps at the long end. `Ladder` supplies `date()` and
/// `target()`; the model quote dispatches per alternative because the FX and
/// xccy alternatives read different curve sets.
using XccyInstrument = Ladder<instruments::FxSwap, instruments::XccyBasisSwap>;

/// Single FX instrument, the short-end alternative of `XccyInstrument`.
using FxInstrument = instruments::FxSwap;

/// Short kind label of a stored discount-curve instrument.
inline std::string_view instrumentKindName(const DiscountInstrument& instrument) {
    return std::visit([](const auto& alternative) { return alternative.kindName(); },
                      instrument.instrument());
}

/// Short kind label of a stored forecast-curve instrument.
inline std::string_view instrumentKindName(const ForecastInstrument& instrument) {
    return std::visit([](const auto& alternative) { return alternative.kindName(); },
                      instrument.instrument());
}

/// Map a parsed pillar to a discount-curve family instrument, preparing its
/// quote times with the supplied zero clock.
DiscountInstrument toDiscountInstrument(const PillarSpec& pillar,
                                        const datetime::Date& referenceDate,
                                        const datetime::DayCounter& zeroDayCounter);

/// Map a parsed pillar to a forecast-curve family instrument, preparing its
/// quote times with the supplied zero clock.
ForecastInstrument toForecastInstrument(const PillarSpec& pillar,
                                        const datetime::Date& referenceDate,
                                        const datetime::DayCounter& zeroDayCounter);

/// Map a parsed FxSwap pillar to an FX instrument: the config-side validations
/// run before `makeFxSwap`, so only the config messages can surface.
FxInstrument toFxInstrument(const CurveStackSpec& stack, const PillarSpec& pillar,
                            const XccyCurveSpec& xccy, const std::string& where);

/// Map a parsed XccySwap pillar to the one-value cross-currency ladder.
XccyInstrument toXccyInstrument(const PillarSpec& pillar, const XccyCurveSpec& xccy);

} // namespace quantape::markets
