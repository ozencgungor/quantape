/**
 * @file test_fx_instruments.cpp
 * @brief Dated FX instruments: settlement lags, cashflows, CIP and PV gates
 *
 * Settlement-lag factories are checked on EURUSD T+2 and a T+0 pair, with
 * weekend and joint-holiday rolls; cashflow payment dates are compared with
 * the factory/quote-derived dates; the date-aware PV overloads are checked
 * against the formula-level pricers and against the discounted materialized
 * cashflows; the CIP helpers are checked against the legacy pillar pricers;
 * and the `var`/`fvar<var>` instantiations are gated against central and
 * analytic derivatives.
 */

#include "quantape/math/StanMath.h"

#include "quantape/datetime/Calendar.h"
#include "quantape/instruments/FxInstruments.h"
#include "quantape/log/Log.h"
#include "quantape/markets/Curves/BootstrapInstrument.h"
#include "quantape/markets/Curves/DiscountCurve.h"
#include "quantape/markets/Curves/FxSwapBuilder.h"
#include "quantape/pricing/Fx.h"
#include "quantape/util/Check.h"

#include <cmath>
#include <cstddef>
#include <string>
#include <vector>

using namespace quantape;
using stan::math::fvar;
using stan::math::var;

namespace {

namespace dt = quantape::datetime;
namespace inst = quantape::instruments;
namespace mk = quantape::markets;

using mk::DiscountCurve;

const dt::Date kReference(2026, 9, 29);
const dt::DayCounter kZeroDc(dt::DayCount::Actual365Fixed);
const dt::Calendar kWeekends = dt::Calendar::weekendsOnly();
const mk::FXDescriptor kEurUsd("EUR", "USD");

DiscountCurve<double> buildCurve(const dt::Date& reference, const dt::DayCounter& zeroDc,
                                 const std::vector<dt::Date>& dates, double base, double slope) {
    std::vector<double> zeros;
    zeros.reserve(dates.size());
    for (const dt::Date& date : dates) {
        zeros.push_back(base + slope * dt::yearFraction(reference, date, zeroDc));
    }
    return DiscountCurve<double>(reference, dates, zeroDc, zeros,
                                 mk::InterpolationSpace::LogDiscount,
                                 mk::InterpolationScheme::Linear);
}

std::string codeOf(const inst::Currency& currency) {
    return std::string(currency.code.data(), currency.code.size());
}

/// Nested-AD functor of the instrument forward value over the spot and the 2Y
/// base zero node, with the quote node fixed. At one year the log-discount
/// interpolation makes the base discount `exp(-z)` and the quote discount
/// `exp(-0.04)`.
struct ForwardHessianInstrument {
    inst::FxForward forward;
    dt::Date reference;
    dt::DayCounter zeroDayCounter;
    double notional = 0.0;

    template <typename ScalarT>
    ScalarT operator()(const Eigen::Matrix<ScalarT, Eigen::Dynamic, 1>& x) const {
        const std::vector<double> times{0.0, 2.0};
        const std::vector<ScalarT> baseZeros{ScalarT(0.0), x[1]};
        const std::vector<ScalarT> quoteZeros{ScalarT(0.0), ScalarT(0.04)};
        const DiscountCurve<ScalarT> baseCurve(
            times, baseZeros, mk::InterpolationSpace::LogDiscount, mk::InterpolationScheme::Linear);
        const DiscountCurve<ScalarT> quoteCurve(times, quoteZeros,
                                                mk::InterpolationSpace::LogDiscount,
                                                mk::InterpolationScheme::Linear);
        const mk::FxSet<DiscountCurve<ScalarT>, DiscountCurve<ScalarT>> curves{baseCurve,
                                                                               quoteCurve};
        return mk::fxForwardPv(forward, curves, reference, zeroDayCounter, x[0], ScalarT(notional));
    }
};

/// Settlement lags: EURUSD T+2 rolls over weekends on the joint calendar, a
/// T+0 pair settles on the trade date, and the joint calendar takes the larger
/// lag and both holiday sets. Forward and swap dates advance the spot date by
/// their tenor.
void testSettlementLagDates() {
    const dt::Date thursday(2026, 10, 8);
    const dt::Date friday(2026, 10, 9);
    const dt::Date saturday(2026, 10, 10);
    const dt::Date monday(2026, 10, 12);
    const dt::Calendar noHolidays = dt::Calendar::noHolidays();

    // EURUSD T+2: Thursday's spot date is Monday after the weekend roll.
    const inst::FxSpot eurSpot =
        inst::makeFxSpot(kEurUsd, thursday, 1.10, 1.0e6, kWeekends, kWeekends, 2, 2);
    CHECK(eurSpot.valueDate == monday);
    CHECK(eurSpot.date() == monday);
    CHECK(eurSpot.nearDate() == monday);
    CHECK(inst::fxSpotDate(thursday, kWeekends, kWeekends, 2, 2) == monday);
    CHECK(inst::fxSpotDate(friday, kWeekends, kWeekends, 2, 2) == monday);

    // T+0 pair: the trade date itself on a business day; a Saturday trade rolls
    // to Monday on the joint calendar.
    const inst::FxSpot gbpSpot = inst::makeFxSpot(mk::FXDescriptor("GBP", "USD"), thursday, 1.27,
                                                  1.0e6, kWeekends, kWeekends, 0, 0);
    CHECK(gbpSpot.valueDate == thursday);
    CHECK(inst::fxSpotDate(saturday, kWeekends, kWeekends, 0, 0) == monday);

    // Joint calendar: the larger lag wins and a holiday on either leg rolls
    // the settlement date.
    const dt::Calendar baseHoliday = kWeekends.withExtraHolidays({monday});
    CHECK(inst::fxSpotDate(thursday, baseHoliday, kWeekends, 2, 1) == thursday.plusDays(5));
    const inst::FxSpot jointSpot =
        inst::makeFxSpot(kEurUsd, thursday, 1.10, 1.0e6, baseHoliday, kWeekends, 2, 1);
    CHECK(jointSpot.valueDate == thursday.plusDays(5));

    // A calendar without holiday rules has no closures, so a no-holiday T+2
    // pair settles two calendar days out.
    CHECK(inst::fxSpotDate(thursday, noHolidays, noHolidays, 2, 2) == saturday);

    // Forward and swap far dates advance the spot date by the tenor.
    const inst::FxForward forward =
        inst::makeFxForward(kEurUsd, thursday, dt::Period(1, dt::TimeUnit::Months), 1.10, 1.12,
                            1.0e6, kWeekends, kWeekends, 2, 2);
    CHECK(forward.valueDate == monday.plusMonths(1));
    CHECK(forward.date() == forward.valueDate);
    CHECK(forward.tenor == dt::Period(1, dt::TimeUnit::Months));

    const inst::FxSwap swap =
        inst::makeFxSwap(kEurUsd, thursday, dt::Period(3, dt::TimeUnit::Months), 1.10, 15.0,
                         mk::QuoteConvention::Points, false, 1e-4, kWeekends, kWeekends);
    CHECK(swap.start == monday);
    CHECK(swap.spotDate == monday);
    CHECK(swap.maturity == inst::fxForwardDate(thursday, kWeekends, kWeekends, 2, 2,
                                               dt::Period(3, dt::TimeUnit::Months)));
    CHECK(swap.nearDate() == swap.start);
    CHECK(swap.farDate() == swap.maturity);
    util::checkClose("swap outright from points", swap.quotedOutright(), 1.10 + 15.0 * 1e-4, 1e-15);
}

/// Cashflow materialization: the four swap legs settle at the near and far
/// dates, the currencies follow the pair, the amounts follow the direction and
/// the cashflow dates agree with the factory-resolved dates.
void testCashflowDates() {
    const dt::Date reference(2026, 10, 8);
    const dt::Date spotDate = inst::fxSpotDate(reference, kWeekends, kWeekends, 2, 2);
    const dt::Date farDate = inst::fxForwardDate(reference, kWeekends, kWeekends, 2, 2,
                                                 dt::Period(6, dt::TimeUnit::Months));

    // An otherwise undated swap rolls its dates from the reference date.
    inst::FxSwap manual;
    manual.pair = kEurUsd;
    manual.tradeDate = reference;
    manual.spot = 1.10;
    manual.points = 12.0;
    manual.pointsScale = 1e-4;
    manual.baseNotional = 2.0e6;
    manual.convention = mk::QuoteConvention::Points;
    manual.baseSpotLag = 2;
    manual.quoteSpotLag = 2;
    manual.farTenor = dt::Period(6, dt::TimeUnit::Months);
    const std::vector<inst::Cashflow> flows = manual.cashflows(reference, kWeekends, kWeekends);
    CHECK(flows.size() == 4);
    CHECK(flows[0].payDate == spotDate);
    CHECK(flows[1].payDate == spotDate);
    CHECK(flows[2].payDate == farDate);
    CHECK(flows[3].payDate == farDate);
    CHECK(codeOf(flows[0].currency) == "EUR");
    CHECK(codeOf(flows[1].currency) == "USD");
    CHECK(codeOf(flows[2].currency) == "EUR");
    CHECK(codeOf(flows[3].currency) == "USD");
    util::checkClose("swap near base flow", flows[0].amount, -2.0e6, 1e-12);
    util::checkClose("swap near quote flow", flows[1].amount, 2.0e6 * 1.10, 1e-9);
    util::checkClose("swap far base flow", flows[2].amount, 2.0e6, 1e-12);
    util::checkClose("swap far quote flow", flows[3].amount, -2.0e6 * (1.10 + 12.0 * 1e-4), 1e-9);

    // Factory-resolved swap: stored dates and cashflow dates agree.
    const inst::FxSwap swap =
        inst::makeFxSwap(kEurUsd, reference, dt::Period(6, dt::TimeUnit::Months), 1.10, 12.0,
                         mk::QuoteConvention::Points, false, 1e-4, kWeekends, kWeekends);
    CHECK(swap.start == spotDate);
    CHECK(swap.maturity == farDate);
    const std::vector<inst::Cashflow> swapFlows = swap.cashflows(reference, kWeekends, kWeekends);
    CHECK(swapFlows[0].payDate == swap.start);
    CHECK(swapFlows[2].payDate == swap.maturity);

    // Forward and spot cashflows settle at their value dates.
    const inst::FxForward forward =
        inst::makeFxForward(kEurUsd, reference, dt::Period(3, dt::TimeUnit::Months), 1.10, 1.11,
                            1.0e6, kWeekends, kWeekends, 2, 2);
    const std::vector<inst::Cashflow> forwardFlows =
        forward.cashflows(reference, kWeekends, kWeekends);
    CHECK(forwardFlows.size() == 2);
    CHECK(forwardFlows[0].payDate == forward.valueDate);
    CHECK(forwardFlows[0].payDate == inst::fxForwardDate(reference, kWeekends, kWeekends, 2, 2,
                                                         dt::Period(3, dt::TimeUnit::Months)));
    util::checkClose("forward base flow", forwardFlows[0].amount, 1.0e6, 1e-12);
    util::checkClose("forward quote flow", forwardFlows[1].amount, -1.0e6 * 1.11, 1e-9);

    const inst::FxSpot spotTrade =
        inst::makeFxSpot(kEurUsd, reference, 1.10, 1.0e6, kWeekends, kWeekends, 2, 2);
    const std::vector<inst::Cashflow> spotFlows =
        spotTrade.cashflows(reference, kWeekends, kWeekends);
    CHECK(spotFlows.size() == 2);
    CHECK(spotFlows[0].payDate == spotTrade.valueDate);
    CHECK(spotFlows[0].payDate == spotDate);
    util::checkClose("spot base flow", spotFlows[0].amount, 1.0e6, 1e-12);
    util::checkClose("spot quote flow", spotFlows[1].amount, -1.0e6 * 1.10, 1e-9);
}

/// Date-aware PV: each overload equals its formula-level pricer with discounts
/// read at the actual settlement dates, and the discounted materialized
/// cashflows agree with the closed-form swap value.
void testPvParity() {
    const std::vector<dt::Date> dates{kReference.plusYears(1), kReference.plusYears(2),
                                      kReference.plusYears(3)};
    const DiscountCurve<double> baseCurve = buildCurve(kReference, kZeroDc, dates, 0.020, 0.0010);
    const DiscountCurve<double> quoteCurve = buildCurve(kReference, kZeroDc, dates, 0.045, 0.0005);
    const mk::FxSet<DiscountCurve<double>, DiscountCurve<double>> curves{baseCurve, quoteCurve};
    const double spot = 1.10;
    const double notional = 2.5e6;

    const inst::FxSpot spotTrade =
        inst::makeFxSpot(kEurUsd, kReference, spot, notional, kWeekends, kWeekends, 2, 2);
    const double spotTime = kZeroDc.yearFraction(kReference, spotTrade.valueDate);
    util::checkClose("spot pv parity",
                     mk::fxSpotPv(spotTrade, curves, kReference, kZeroDc, spot, notional),
                     mk::fxSpotPv(spot, notional) * baseCurve.discount(spotTime), 1e-12);

    const inst::FxForward forward =
        inst::makeFxForward(kEurUsd, kReference, dt::Period(1, dt::TimeUnit::Years), spot, 1.12,
                            notional, kWeekends, kWeekends, 2, 2);
    const double valueTime = kZeroDc.yearFraction(kReference, forward.valueDate);
    util::checkClose("forward pv parity",
                     mk::fxForwardPv(forward, curves, kReference, kZeroDc, spot, notional),
                     mk::fxForwardPv(spot, 1.12, baseCurve.discount(valueTime),
                                     quoteCurve.discount(valueTime), notional),
                     1e-12);

    const inst::FxSwap swap =
        inst::makeFxSwap(kEurUsd, kReference, dt::Period(1, dt::TimeUnit::Years), spot, 25.0,
                         mk::QuoteConvention::Points, false, 1e-4, kWeekends, kWeekends);
    const double nearTime = kZeroDc.yearFraction(kReference, swap.start);
    const double farTime = kZeroDc.yearFraction(kReference, swap.maturity);
    const double farStrike = swap.quotedOutright();
    const double closedForm = mk::fxSwapPv(
        spot, spot, baseCurve.discount(nearTime), quoteCurve.discount(nearTime), farStrike,
        baseCurve.discount(farTime), quoteCurve.discount(farTime), notional);
    util::checkClose("swap pv parity",
                     mk::fxSwapPv(swap, curves, kReference, kZeroDc, spot, notional), closedForm,
                     1e-12);

    const std::vector<inst::CashflowT<double>> swapFlows =
        mk::fxCashflows(swap, notional, kReference, kWeekends, kWeekends);
    util::checkClose("swap cashflow pv parity",
                     mk::fxCashflowPv(kEurUsd, swapFlows, spot, kReference, kZeroDc, curves),
                     closedForm, 1e-9);

    const std::vector<inst::CashflowT<double>> forwardFlows =
        mk::fxCashflows(forward, notional, kReference, kWeekends, kWeekends);
    util::checkClose("forward cashflow pv parity",
                     mk::fxCashflowPv(kEurUsd, forwardFlows, spot, kReference, kZeroDc, curves),
                     mk::fxForwardPv(forward, curves, kReference, kZeroDc, spot, notional), 1e-9);
}

/// CIP helpers against the legacy pillar pricers for both collateral
/// directions, and the model quote against the quoted target.
void testCipAgainstLegacy() {
    const std::vector<dt::Date> dates{kReference.plusYears(1), kReference.plusYears(2),
                                      kReference.plusYears(3)};
    const DiscountCurve<double> baseCurve = buildCurve(kReference, kZeroDc, dates, 0.020, 0.0010);
    const DiscountCurve<double> quoteCurve = buildCurve(kReference, kZeroDc, dates, 0.045, 0.0005);
    const mk::FxSet<DiscountCurve<double>, DiscountCurve<double>> curves{baseCurve, quoteCurve};
    const double spot = 1.10;
    const dt::Date spotDate = inst::fxSpotDate(kReference, kWeekends, kWeekends, 2, 2);
    const dt::Date farDate = kReference.plusYears(2);

    // Quote collateral: the unknown leg is the base curve.
    const inst::FxSwap swap =
        inst::makeFxSwap(kEurUsd, kReference, spotDate, farDate, spot, 0.0,
                         mk::QuoteConvention::Points, false, 1.0, kWeekends, kWeekends);
    mk::FxSwapPillar legacy;
    legacy.start = swap.start;
    legacy.maturity = swap.maturity;
    legacy.spot = spot;
    legacy.isFxBaseCollateral = false;
    legacy.points =
        mk::impliedFxForwardPoints(baseCurve, quoteCurve, spot, legacy, kReference, kZeroDc);
    const double legacyOutright =
        mk::impliedFxOutright(baseCurve, quoteCurve, spot, legacy, kReference, kZeroDc);
    util::checkClose("cip outright vs legacy",
                     mk::impliedFxOutright(swap, curves, kReference, kZeroDc), legacyOutright,
                     1e-12);
    util::checkClose("cip points vs legacy",
                     mk::impliedFxForwardPoints(swap, curves, kReference, kZeroDc), legacy.points,
                     1e-12);

    // Base collateral: the unknown leg is the quote curve.
    const inst::FxSwap baseSwap =
        inst::makeFxSwap(kEurUsd, kReference, spotDate, farDate, spot, 0.0,
                         mk::QuoteConvention::Outright, true, 1.0, kWeekends, kWeekends);
    mk::FxSwapPillar legacyBase;
    legacyBase.start = baseSwap.start;
    legacyBase.maturity = baseSwap.maturity;
    legacyBase.spot = spot;
    legacyBase.isFxBaseCollateral = true;
    legacyBase.convention = mk::QuoteConvention::Outright;
    legacyBase.outright =
        mk::impliedFxOutright(quoteCurve, baseCurve, spot, legacyBase, kReference, kZeroDc);
    util::checkClose("cip base collateral vs legacy",
                     mk::impliedFxOutright(baseSwap, curves, kReference, kZeroDc),
                     legacyBase.outright, 1e-12);

    // The forward CIP lands on the same outright at the same date.
    const inst::FxForward forward = inst::makeFxForward(kEurUsd, kReference, farDate, spot, 0.0,
                                                        1.0e6, kWeekends, kWeekends, 2, 2, false);
    util::checkClose("forward cip vs legacy",
                     mk::impliedFxOutright(forward, curves, kReference, kZeroDc), legacyOutright,
                     1e-12);

    // A par instrument reprices its own quoted target.
    inst::FxSwap par = swap;
    par.points = legacy.points;
    par.zeroDayCounter = kZeroDc;
    util::checkClose("swap implied quote reprice", par.impliedQuote<double>(curves), par.target(),
                     1e-12);
}

/// The plain-data instruments satisfy the bootstrap-instrument concept for
/// `double`, `var` and `fvar<var>`.
void testBootstrapConcept() {
    using DoubleSet = mk::FxSet<DiscountCurve<double>, DiscountCurve<double>>;
    using VarSet = mk::FxSet<DiscountCurve<var>, DiscountCurve<var>>;
    using NestedSet = mk::FxSet<DiscountCurve<fvar<var>>, DiscountCurve<fvar<var>>>;
    static_assert(mk::BootstrapInstrument<inst::FxSpot, DoubleSet, double>);
    static_assert(mk::BootstrapInstrument<inst::FxForward, DoubleSet, double>);
    static_assert(mk::BootstrapInstrument<inst::FxSwap, DoubleSet, double>);
    static_assert(mk::BootstrapInstrument<inst::FxSwap, VarSet, var>);
    static_assert(mk::BootstrapInstrument<inst::FxSwap, NestedSet, fvar<var>>);
    static_assert(mk::BootstrapInstrument<inst::FxForward, VarSet, var>);

    const std::vector<double> times{0.0, 2.0};
    const std::vector<double> zeros{0.0, 0.04};
    const DiscountCurve<double> baseCurve(times, zeros, mk::InterpolationSpace::LogDiscount,
                                          mk::InterpolationScheme::Linear);
    const DiscountCurve<double> quoteCurve(times, zeros, mk::InterpolationSpace::LogDiscount,
                                           mk::InterpolationScheme::Linear);
    const DoubleSet curves{baseCurve, quoteCurve};
    const inst::FxSwap swap =
        inst::makeFxSwap(kEurUsd, kReference, dt::Period(2, dt::TimeUnit::Years), 1.10, 0.0,
                         mk::QuoteConvention::Outright, false, 1.0, kWeekends, kWeekends);
    const inst::FxSwap par = [&] {
        inst::FxSwap out = swap;
        out.convention = mk::QuoteConvention::Outright;
        out.outright = mk::impliedFxOutright(swap, curves, kReference, kZeroDc);
        return out;
    }();
    util::checkClose("concept implied quote", par.impliedQuote<double>(curves), par.target(),
                     1e-12);
    CHECK(par.date() == par.maturity);
}

/// Reverse- and forward-mode gates: `var` spot and curve adjoints against
/// central differences, and the nested `fvar<var>` Hessian of the instrument
/// forward against its analytic second-order values.
void testAdVsFiniteDifference() {
    const std::vector<dt::Date> dates{kReference.plusYears(2)};
    const DiscountCurve<double> baseDouble = buildCurve(kReference, kZeroDc, dates, 0.020, 0.0010);
    const DiscountCurve<double> quoteDouble = buildCurve(kReference, kZeroDc, dates, 0.045, 0.0005);
    const mk::FxSet<DiscountCurve<double>, DiscountCurve<double>> curves{baseDouble, quoteDouble};
    const double spot = 1.10;
    const double notional = 2.5e6;
    const dt::Date spotDate = inst::fxSpotDate(kReference, kWeekends, kWeekends, 2, 2);
    const dt::Date farDate = kReference.plusYears(2);
    const inst::FxSwap swap =
        inst::makeFxSwap(kEurUsd, kReference, spotDate, farDate, spot, 12.0,
                         mk::QuoteConvention::Points, false, 1e-4, kWeekends, kWeekends);
    const inst::FxForward forward = inst::makeFxForward(
        kEurUsd, kReference, farDate, spot, 1.12, notional, kWeekends, kWeekends, 2, 2, false);

    // Spot adjoint of the CIP model against a central difference.
    stan::math::recover_memory();
    {
        var spotVar(spot);
        var value = mk::impliedFxOutright(swap, curves, spotVar, kReference, kZeroDc);
        value.grad();
        const double step = 1e-4;
        const double fd = (mk::impliedFxOutright(swap, curves, spot + step, kReference, kZeroDc) -
                           mk::impliedFxOutright(swap, curves, spot - step, kReference, kZeroDc)) /
                          (2.0 * step);
        util::checkClose("cip spot adjoint vs fd", spotVar.adj(), fd, 1e-8);
    }
    stan::math::recover_memory();

    // Curve node adjoints of the CIP model against central differences.
    stan::math::recover_memory();
    {
        const double nodeTime = kZeroDc.yearFraction(kReference, dates[0]);
        std::vector<var> baseZeros{var(0.020 + 0.0010 * nodeTime)};
        std::vector<var> quoteZeros{var(0.045 + 0.0005 * nodeTime)};
        const DiscountCurve<var> baseVar(kReference, dates, kZeroDc, baseZeros,
                                         mk::InterpolationSpace::LogDiscount,
                                         mk::InterpolationScheme::Linear);
        const DiscountCurve<var> quoteVar(kReference, dates, kZeroDc, quoteZeros,
                                          mk::InterpolationSpace::LogDiscount,
                                          mk::InterpolationScheme::Linear);
        const mk::FxSet<DiscountCurve<var>, DiscountCurve<var>> curvesVar{baseVar, quoteVar};
        var value = mk::impliedFxOutright(swap, curvesVar, var(spot), kReference, kZeroDc);
        value.grad();

        const double step = 1e-6;
        const auto valueAt = [&](std::size_t node, double bump) {
            std::vector<double> shifted{baseZeros[0].val() + (node == 0 ? bump : 0.0)};
            const DiscountCurve<double> basePert(kReference, dates, kZeroDc, shifted,
                                                 mk::InterpolationSpace::LogDiscount,
                                                 mk::InterpolationScheme::Linear);
            const mk::FxSet<DiscountCurve<double>, DiscountCurve<double>> perturbed{basePert,
                                                                                    quoteDouble};
            return mk::impliedFxOutright(swap, perturbed, spot, kReference, kZeroDc);
        };
        const double baseFd = (valueAt(0, step) - valueAt(0, -step)) / (2.0 * step);
        util::checkClose("cip base node adjoint vs fd", baseZeros[0].adj(), baseFd, 1e-7);

        std::vector<double> quoteZerosDouble{quoteZeros[0].val()};
        const auto quoteValueAt = [&](double bump) {
            const DiscountCurve<double> quotePert(
                kReference, dates, kZeroDc, std::vector<double>{quoteZerosDouble[0] + bump},
                mk::InterpolationSpace::LogDiscount, mk::InterpolationScheme::Linear);
            const mk::FxSet<DiscountCurve<double>, DiscountCurve<double>> perturbed{baseDouble,
                                                                                    quotePert};
            return mk::impliedFxOutright(swap, perturbed, spot, kReference, kZeroDc);
        };
        const double quoteFd = (quoteValueAt(step) - quoteValueAt(-step)) / (2.0 * step);
        util::checkClose("cip quote node adjoint vs fd", quoteZeros[0].adj(), quoteFd, 1e-7);
    }
    stan::math::recover_memory();

    // Spot and notional adjoints of the date-aware forward PV.
    stan::math::recover_memory();
    {
        var spotVar(spot);
        var notionalVar(notional);
        var value = mk::fxForwardPv(forward, curves, kReference, kZeroDc, spotVar, notionalVar);
        value.grad();
        const double baseDiscount = baseDouble.discount(kZeroDc.yearFraction(kReference, farDate));
        const double quoteDiscount =
            quoteDouble.discount(kZeroDc.yearFraction(kReference, farDate));
        util::checkClose("forward pv spot adjoint", spotVar.adj() / (notional * baseDiscount), 1.0,
                         1e-12);
        util::checkClose("forward pv notional adjoint",
                         notionalVar.adj() / (spot * baseDiscount - 1.12 * quoteDiscount), 1.0,
                         1e-12);
        const double step = 1e-4;
        const double fd =
            (mk::fxForwardPv(forward, curves, kReference, kZeroDc, spot + step, notional) -
             mk::fxForwardPv(forward, curves, kReference, kZeroDc, spot - step, notional)) /
            (2.0 * step);
        util::checkClose("forward pv spot fd", fd / (notional * baseDiscount), 1.0, 1e-9);
    }
    stan::math::recover_memory();

    // Swap spot adjoint and, in a fresh tape, the cashflow-PV spot adjoint.
    const double nearDiscount = baseDouble.discount(kZeroDc.yearFraction(kReference, spotDate));
    const double farDiscount = baseDouble.discount(kZeroDc.yearFraction(kReference, farDate));
    const double swapSpotDelta = notional * (farDiscount - nearDiscount);
    stan::math::recover_memory();
    {
        var spotVar(spot);
        var value = mk::fxSwapPv(swap, curves, kReference, kZeroDc, spotVar, var(notional));
        value.grad();
        util::checkClose("swap pv spot adjoint", spotVar.adj() / swapSpotDelta, 1.0, 1e-12);
    }
    stan::math::recover_memory();
    {
        var spotVar(spot);
        const std::vector<inst::CashflowT<var>> flows =
            mk::fxCashflows(swap, var(notional), kReference, kWeekends, kWeekends);
        var cashflowValue = mk::fxCashflowPv(kEurUsd, flows, spotVar, kReference, kZeroDc, curves);
        cashflowValue.grad();
        util::checkClose("cashflow pv spot adjoint", spotVar.adj() / swapSpotDelta, 1.0, 1e-12);
    }
    stan::math::recover_memory();

    // Nested forward-over-reverse: the value is linear in the spot, so the
    // spot gamma is zero and the spot/base-node cross is -N D_base; the base
    // zero node feeds the curve and the quote node stays fixed.
    stan::math::recover_memory();
    {
        const inst::FxForward datedForward =
            inst::makeFxForward(kEurUsd, kReference, kReference.plusYears(1), spot, 1.12, notional,
                                kWeekends, kWeekends, 2, 2, false);
        Eigen::VectorXd point(2);
        point << spot, 0.02;
        Eigen::MatrixXd hessian;
        Eigen::VectorXd gradient;
        double value = 0.0;
        const ForwardHessianInstrument functor{datedForward, kReference, kZeroDc, notional};
        stan::math::hessian(functor, point, value, gradient, hessian);
        const double baseDiscount = std::exp(-0.02);
        const double quoteDiscount = std::exp(-0.04);
        util::checkClose("instrument hessian value",
                         value / (notional * (spot * baseDiscount - 1.12 * quoteDiscount)), 1.0,
                         1e-12);
        util::checkClose("instrument hessian spot delta", gradient[0] / (notional * baseDiscount),
                         1.0, 1e-12);
        util::checkClose("instrument hessian base node delta",
                         gradient[1] / (-notional * spot * baseDiscount), 1.0, 1e-12);
        util::checkClose("instrument hessian spot gamma", hessian(0, 0), 0.0, 1e-9);
        util::checkClose("instrument hessian spot-base cross",
                         hessian(0, 1) / (-notional * baseDiscount), 1.0, 1e-10);
        util::checkClose("instrument hessian base-spot cross",
                         hessian(1, 0) / (-notional * baseDiscount), 1.0, 1e-10);
        util::checkClose("instrument hessian base node gamma",
                         hessian(1, 1) / (notional * spot * baseDiscount), 1.0, 1e-10);
    }
    stan::math::recover_memory();
    QTA_LOG_INFO("test", "fx instrument AD gates passed");
}

} // namespace

int main() {
    testSettlementLagDates();
    testCashflowDates();
    testPvParity();
    testCipAgainstLegacy();
    testBootstrapConcept();
    testAdVsFiniteDifference();
    QTA_LOG_INFO("test", "test_fx_instruments: ok");
    return 0;
}
