/**
 * @file test_fx_pricing.cpp
 * @brief FX pricing, reset-aware cross-currency rows and FX stack risk gates
 *
 * The pricers are checked on spot/forward/swap parities, their reverse-mode
 * gradients against central differences, and their nested `fvar<var>` path
 * against analytic second-order values. The resetting-notional cross-currency
 * rows are checked against the finite-difference reference for both reset-leg
 * choices, the FX spot factor block against a finite difference of the value,
 * and the FX gamma and FXxIR cross-gamma against the mixed-coordinate Hessian
 * transform and a finite difference of the solved quote deltas.
 */

#include "quantape/math/StanMath.h"

#include "quantape/log/Log.h"
#include "quantape/markets/Curves/CurveBuilder.h"
#include "quantape/markets/Curves/SpreadCurve.h"
#include "quantape/markets/Curves/StackRisk.h"
#include "quantape/markets/Curves/XccyBasisBuilder.h"
#include "quantape/markets/Curves/XccyRisk.h"
#include "quantape/markets/Data/FXRate.h"
#include "quantape/markets/Data/FxQuote.h"
#include "quantape/math/LinearAlgebra/DenseSolve.h"
#include "quantape/pricing/Fx.h"
#include "quantape/util/Check.h"

#include <Eigen/Dense>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <memory>
#include <string>
#include <vector>

using namespace quantape;
using markets::CurvePillar;
using markets::DiscountCurve;
using markets::InterpolationScheme;
using markets::InterpolationSpace;
using markets::PillarKind;
using stan::math::fvar;
using stan::math::var;

namespace {

DiscountCurve<double> buildCurve(const datetime::Date& reference,
                                 const datetime::DayCounter& zeroDc,
                                 const std::vector<datetime::Date>& dates, double base,
                                 double slope) {
    std::vector<double> zeros;
    zeros.reserve(dates.size());
    for (const datetime::Date& date : dates) {
        zeros.push_back(base + slope * datetime::yearFraction(reference, date, zeroDc));
    }
    return DiscountCurve<double>(reference, dates, zeroDc, zeros, InterpolationSpace::LogDiscount,
                                 InterpolationScheme::Linear);
}

std::vector<CurvePillar> makeOisPillars(const datetime::Date& reference,
                                        const datetime::DayCounter& zeroDc,
                                        const datetime::Calendar& calendar,
                                        const std::vector<datetime::Date>& dates,
                                        const DiscountCurve<double>& target) {
    std::vector<CurvePillar> pillars;
    pillars.reserve(dates.size());
    for (const datetime::Date& date : dates) {
        CurvePillar pillar;
        pillar.maturity = date;
        pillar.kind = PillarKind::OisSwap;
        pillar.quoteDayCounter = zeroDc;
        pillar.calendar = calendar;
        pillar.quote = markets::impliedQuote(pillar, reference, target);
        pillars.push_back(pillar);
    }
    return pillars;
}

/// Nested-AD functor of the forward value over the spot and the base
/// discount factor (the strike is fixed).
struct ForwardHessian {
    double strike = 0.0;
    double quoteDiscount = 0.0;
    double notional = 0.0;
    template <typename ScalarT>
    ScalarT operator()(const Eigen::Matrix<ScalarT, Eigen::Dynamic, 1>& x) const {
        return markets::fxForwardPv(x[0], ScalarT(strike), x[1], ScalarT(quoteDiscount),
                                    ScalarT(notional));
    }
};

/// Nested-AD functor of the swap value over the spot and the far base
/// discount factor.
struct SwapHessian {
    double nearBase = 0.0;
    double nearQuote = 0.0;
    double farStrike = 0.0;
    double farQuote = 0.0;
    double notional = 0.0;
    template <typename ScalarT>
    ScalarT operator()(const Eigen::Matrix<ScalarT, Eigen::Dynamic, 1>& x) const {
        return markets::fxSwapPv(x[0], x[0], ScalarT(nearBase), ScalarT(nearQuote),
                                 ScalarT(farStrike), x[1], ScalarT(farQuote), ScalarT(notional));
    }
};

/// Root discount curve of the stack-risk gates: one par OIS pillar per year.
struct RootFixture {
    datetime::Date reference{2026, 9, 29};
    datetime::DayCounter zeroDc{datetime::DayCount::Actual365Fixed};
    datetime::Calendar calendar = datetime::Calendar::noHolidays();
    std::vector<datetime::Date> dates{reference.plusYears(1), reference.plusYears(2),
                                      reference.plusYears(3)};
    DiscountCurve<double> root;
    std::vector<CurvePillar> pillars;

    RootFixture()
        : root(buildCurve(reference, zeroDc, dates, 0.030, 0.0004)),
          pillars(makeOisPillars(reference, zeroDc, calendar, dates, root)) {}
};

/// Spot and forward/swap PV: the affine spot map, the covered-interest-parity
/// zero, the forward/spot relation, and the AD paths against central and
/// analytic derivatives.
void testFxSpotAndForwardPricing() {
    const datetime::Date reference(2026, 9, 29);
    const datetime::DayCounter zeroDc(datetime::DayCount::Actual365Fixed);
    const std::vector<datetime::Date> dates{reference.plusYears(1), reference.plusYears(2),
                                            reference.plusYears(3)};
    const DiscountCurve<double> baseCurve = buildCurve(reference, zeroDc, dates, 0.020, 0.0010);
    const DiscountCurve<double> quoteCurve = buildCurve(reference, zeroDc, dates, 0.045, 0.0005);
    const double spot = 1.10;
    const double strike = 1.12;
    const double notional = 2.5e6;
    const double maturity = datetime::yearFraction(reference, dates[1], zeroDc);
    const double baseDiscount = baseCurve.discount(maturity);
    const double quoteDiscount = quoteCurve.discount(maturity);

    const double spotPv = markets::fxSpotPv(spot, notional);
    util::checkClose("fx spot pv", spotPv, spot * notional, 1e-12);
    util::checkClose("fx spot pv via zero strike",
                     markets::fxForwardPv(spot, 0.0, 1.0, 1.0, notional), spotPv, 1e-12);

    const double forwardPv =
        markets::fxForwardPv(spot, strike, baseDiscount, quoteDiscount, notional);
    util::checkClose("fx forward pv", forwardPv,
                     notional * (spot * baseDiscount - strike * quoteDiscount), 1e-12);

    const double cipForward = spot * baseDiscount / quoteDiscount;
    util::checkClose("fx forward cip zero",
                     markets::fxForwardPv(spot, cipForward, baseDiscount, quoteDiscount, notional) /
                         notional,
                     0.0, 1e-12);
    const markets::FXDescriptor eurusd("EUR", "USD");
    const markets::FXRate<double> rate(spot, baseCurve, quoteCurve, eurusd);
    // Relative tolerance: the two overloads agree mathematically, but at
    // notional-scale values (~1e5) a 1e-12 absolute tolerance is below one ULP.
    util::checkCloseRel("fx forward rate overload",
                        markets::fxForwardPv(rate, maturity, strike, notional), forwardPv, 1e-12,
                        1e-12);

    // Reverse mode: the spot delta must equal the discounted base notional and
    // the curve node adjoints the analytic curve weights.
    stan::math::recover_memory();
    {
        std::vector<var> baseZeros;
        std::vector<var> quoteZeros;
        for (const datetime::Date& date : dates) {
            const double t = datetime::yearFraction(reference, date, zeroDc);
            baseZeros.push_back(var(0.020 + 0.0010 * t));
            quoteZeros.push_back(var(0.045 + 0.0005 * t));
        }
        const DiscountCurve<var> baseVar(reference, dates, zeroDc, baseZeros,
                                         InterpolationSpace::LogDiscount,
                                         InterpolationScheme::Linear);
        const DiscountCurve<var> quoteVar(reference, dates, zeroDc, quoteZeros,
                                          InterpolationSpace::LogDiscount,
                                          InterpolationScheme::Linear);
        var spotVar(spot);
        var strikeVar(strike);
        var notionalVar(notional);
        const markets::FXRate<var> rateVar(spotVar, baseVar, quoteVar, eurusd);
        var value = markets::fxForwardPv(rateVar, maturity, strikeVar, notionalVar);
        value.grad();
        util::checkClose("fx forward spot adjoint", spotVar.adj(), notional * baseDiscount, 1e-9);
        util::checkClose("fx forward strike adjoint", strikeVar.adj(), -notional * quoteDiscount,
                         1e-9);
        util::checkClose("fx forward notional adjoint", notionalVar.adj(),
                         (spot * baseDiscount - strike * quoteDiscount) * 1.0, 1e-9);
        std::vector<double> baseWeights;
        baseCurve.zeroNodeWeights(maturity, baseWeights);
        for (std::size_t i = 1; i < baseZeros.size() + 1; ++i) {
            const double expected = notional * spot * baseDiscount * (-maturity * baseWeights[i]);
            util::checkClose("fx forward base node adjoint", baseZeros[i - 1].adj(), expected,
                             1e-8);
        }
        std::vector<double> quoteWeights;
        quoteCurve.zeroNodeWeights(maturity, quoteWeights);
        for (std::size_t i = 1; i < quoteZeros.size() + 1; ++i) {
            const double expected =
                -notional * strike * quoteDiscount * (-maturity * quoteWeights[i]);
            util::checkClose("fx forward quote node adjoint", quoteZeros[i - 1].adj(), expected,
                             1e-8);
        }
    }
    stan::math::recover_memory();

    // Central difference of the spot value against the analytic delta.
    const double step = 1e-5;
    const double fdDelta =
        (markets::fxForwardPv(spot + step, strike, baseDiscount, quoteDiscount, notional) -
         markets::fxForwardPv(spot - step, strike, baseDiscount, quoteDiscount, notional)) /
        (2.0 * step);
    util::checkClose("fx forward delta vs fd", fdDelta / (notional * baseDiscount), 1.0, 1e-9);

    // Nested forward-over-reverse: the forward value is affine in the spot, so
    // the spot gamma is zero and the spot/base-discount cross is N. The functor
    // exercises the `fvar<var>` instantiation.
    stan::math::recover_memory();
    {
        Eigen::VectorXd point(2);
        point << spot, baseDiscount;
        Eigen::MatrixXd hessian;
        Eigen::VectorXd gradient;
        double value = 0.0;
        stan::math::hessian(ForwardHessian{strike, quoteDiscount, notional}, point, value, gradient,
                            hessian);
        util::checkClose("fx forward hessian value", value, forwardPv, 1e-10);
        util::checkClose("fx forward hessian delta", gradient[0], notional * baseDiscount, 1e-9);
        util::checkClose("fx forward hessian base discount delta", gradient[1], notional * spot,
                         1e-9);
        util::checkClose("fx forward hessian spot gamma", hessian(0, 0), 0.0, 1e-9);
        util::checkClose("fx forward hessian spot-base discount", hessian(0, 1), notional, 1e-8);
        util::checkClose("fx forward hessian base discount-spot", hessian(1, 0), notional, 1e-8);
    }
    stan::math::recover_memory();
}

/// Swap parities: the swap is the far minus the near forward, a CIP far strike
/// closes the far leg, and the quoted schedule overload agrees with the scalar
/// formula. AD gradients match the analytic spot delta.
void testFxSwapPricing() {
    const double spot = 1.10;
    const double nearTime = 0.5;
    const double farTime = 1.0;
    const double nearBase = 0.975;
    const double nearQuote = 0.973;
    const double farBase = 0.951;
    const double farQuote = 0.947;
    const double notional = 4.0e6;
    const double farStrike = spot * farBase / farQuote;

    const double nearPv = markets::fxForwardPv(spot, spot, nearBase, nearQuote, notional);
    const double farPv = markets::fxForwardPv(spot, farStrike, farBase, farQuote, notional);
    util::checkClose("fx swap far leg cip", farPv / notional, 0.0, 1e-12);
    const double swapPv =
        markets::fxSwapPv(spot, spot, nearBase, nearQuote, farStrike, farBase, farQuote, notional);
    util::checkClose("fx swap far minus near", swapPv, farPv - nearPv, 1e-8);
    util::checkClose("fx swap outer legs", swapPv, -nearPv, 1e-8);

    // Spot gradient: N (D_base_far - D_base_near).
    stan::math::recover_memory();
    {
        var spotVar(spot);
        var value = markets::fxSwapPv(spotVar, var(spot), var(nearBase), var(nearQuote),
                                      var(farStrike), var(farBase), var(farQuote), var(notional));
        value.grad();
        util::checkClose("fx swap spot adjoint", spotVar.adj(), notional * (farBase - nearBase),
                         1e-9);
    }
    stan::math::recover_memory();

    // Nested AD: zero spot gamma and the N spot/base-discount cross on the far
    // leg.
    stan::math::recover_memory();
    {
        Eigen::VectorXd point(2);
        point << spot, farBase;
        Eigen::MatrixXd hessian;
        Eigen::VectorXd gradient;
        double value = 0.0;
        stan::math::hessian(SwapHessian{nearBase, nearQuote, farStrike, farQuote, notional}, point,
                            value, gradient, hessian);
        util::checkClose("fx swap hessian value", value, swapPv, 1e-9);
        util::checkClose("fx swap hessian spot gamma", hessian(0, 0), 0.0, 1e-9);
        util::checkClose("fx swap hessian spot-far-base", hessian(0, 1), notional, 1e-8);
        util::checkClose("fx swap hessian far-base-spot", hessian(1, 0), notional, 1e-8);
    }
    stan::math::recover_memory();

    // Quoted schedule overload: near leg at spot, far leg at the quoted
    // outright, both discounted on the rate's curves.
    const datetime::Date reference(2026, 9, 29);
    const datetime::DayCounter zeroDc(datetime::DayCount::Actual365Fixed);
    const std::vector<datetime::Date> dates{reference.plusMonths(6), reference.plusYears(1),
                                            reference.plusYears(2)};
    const DiscountCurve<double> baseCurve = buildCurve(reference, zeroDc, dates, 0.021, 0.0009);
    const DiscountCurve<double> quoteCurve = buildCurve(reference, zeroDc, dates, 0.044, 0.0005);
    const markets::FXDescriptor eurusd("EUR", "USD");
    const markets::FXRate<double> rate(spot, baseCurve, quoteCurve, eurusd);
    const datetime::Date spotDate = reference.plusDays(2);
    const datetime::Date maturityDate = reference.plusMonths(6);
    const double nearTimeQuoted = zeroDc.yearFraction(reference, spotDate);
    const double farTimeQuoted = zeroDc.yearFraction(reference, maturityDate);
    const double outright =
        spot * baseCurve.discount(farTimeQuoted) / quoteCurve.discount(farTimeQuoted);
    const double points = (outright - spot) / 1e-4;
    const markets::FxSwapQuote quote(
        markets::FxQuote("eurusd.6m", "test", points - 0.5, points + 0.5), eurusd, spot, spotDate,
        spotDate, maturityDate, 1e-4, markets::QuoteConvention::Points);
    const double quotedPv = markets::fxSwapPv(rate, quote, reference, zeroDc, notional);
    const double scalarPv = markets::fxSwapPv(
        spot, spot, baseCurve.discount(nearTimeQuoted), quoteCurve.discount(nearTimeQuoted),
        outright, baseCurve.discount(farTimeQuoted), quoteCurve.discount(farTimeQuoted), notional);
    util::checkClose("fx swap quoted schedule", quotedPv, scalarPv, 1e-9);
    QTA_LOG_INFO("test", "fx quoted swap pv {} from outright {}", quotedPv, outright);
}

/// Resetting-notional cross-currency rows against the finite-difference
/// reference, for both reset legs and both spread sides, on a depth-2 forecast
/// chain whose domestic discount also parents the domestic forecast.
void testMtMRowsAgainstFiniteDifference() {
    const datetime::Date reference(2026, 9, 29);
    const datetime::DayCounter zeroDc(datetime::DayCount::Actual365Fixed);
    const datetime::Calendar calendar = datetime::Calendar::noHolidays();
    const auto build = [&](double base, double slope) {
        const std::vector<datetime::Date> dates{reference.plusYears(1), reference.plusYears(2),
                                                reference.plusYears(3), reference.plusYears(4)};
        return buildCurve(reference, zeroDc, dates, base, slope);
    };
    const DiscountCurve<double> foreignDiscount = build(0.030, 0.0006);
    const DiscountCurve<double> domesticDiscount = build(0.040, 0.0005);
    // The domestic forecast parent is a distinct object with the same values,
    // so the resetting domestic leg's own discount rows stay observable
    // instead of cancelling against an ancestor on the same object.
    auto domesticParent = std::make_shared<DiscountCurve<double>>(build(0.040, 0.0005));
    const std::vector<double> nodeTimes{0.0, 1.0, 2.0, 3.0, 4.0};
    const markets::SpreadCurve<double> domesticForecast(
        domesticParent, nodeTimes, std::vector<double>{0.0, 0.0003, 0.0004, 0.0005, 0.0006},
        InterpolationScheme::Linear);
    auto foreignBase = std::make_shared<DiscountCurve<double>>(build(0.025, 0.0007));
    auto foreignParent = std::make_shared<markets::SpreadCurve<double>>(
        foreignBase, nodeTimes, std::vector<double>{0.0, 0.0004, 0.0005, 0.0006, 0.0007},
        InterpolationScheme::Linear);
    const markets::SpreadCurve<double, markets::SpreadCurve<double>> foreignForecast(
        foreignParent, nodeTimes, std::vector<double>{0.0, 0.0002, 0.0003, 0.0004, 0.0005},
        InterpolationScheme::Linear);

    const markets::StackCurveView::Ptr foreignDiscountView =
        markets::StackCurveView::make(foreignDiscount);
    const markets::StackCurveView::Ptr foreignForecastView =
        markets::StackCurveView::make(foreignForecast);
    const markets::StackCurveView::Ptr domesticDiscountView =
        markets::StackCurveView::make(domesticDiscount);
    const markets::StackCurveView::Ptr domesticForecastView =
        markets::StackCurveView::make(domesticForecast);

    for (const bool resetForeign : {true, false}) {
        for (const bool spreadOnForeign : {true, false}) {
            markets::XccyPillar pillar;
            pillar.maturity = reference.plusYears(3);
            pillar.foreignTenor = datetime::Period(7, datetime::TimeUnit::Months);
            pillar.domesticTenor = datetime::Period(4, datetime::TimeUnit::Months);
            pillar.foreignPaymentLag = 2;
            pillar.domesticPaymentLag = 1;
            pillar.foreignBusinessDayConvention = datetime::BusinessDayConvention::Following;
            pillar.domesticBusinessDayConvention =
                datetime::BusinessDayConvention::ModifiedFollowing;
            pillar.foreignCalendar = calendar;
            pillar.domesticCalendar = calendar;
            pillar.foreignDayCounter = datetime::DayCounter(datetime::DayCount::Actual360);
            pillar.domesticDayCounter = datetime::DayCounter(datetime::DayCount::Actual360);
            pillar.notional = markets::XccyNotionalMode::MtM;
            pillar.resetForeignLeg = resetForeign;
            pillar.spreadOnForeignLeg = spreadOnForeign;

            const double viewSpread = markets::xccyBasisSpreadView(
                *foreignDiscountView, *foreignForecastView, *domesticDiscountView,
                *domesticForecastView, pillar, reference, zeroDc);
            const double concreteSpread =
                markets::impliedXccyBasisSpread(foreignDiscount, foreignForecast, domesticDiscount,
                                                domesticForecast, pillar, reference, zeroDc);
            util::checkClose("mtm view spread vs builder", viewSpread, concreteSpread, 1e-12);

            const std::vector<markets::XccyRowBlock> analytic = markets::xccySwapJacobianRowsView(
                *foreignDiscountView, *foreignForecastView, *domesticDiscountView,
                *domesticForecastView, pillar, reference, zeroDc);
            const std::vector<markets::XccyRowBlock> finiteDifference =
                markets::xccySwapJacobianRowsViewFiniteDifference(
                    *foreignDiscountView, *foreignForecastView, *domesticDiscountView,
                    *domesticForecastView, pillar, reference, zeroDc);
            CHECK(analytic.size() == finiteDifference.size());
            double foreignRisk = 0.0;
            double domesticRisk = 0.0;
            for (const markets::XccyRowBlock& block : analytic) {
                const markets::XccyRowBlock* referenceBlock = nullptr;
                for (const markets::XccyRowBlock& candidate : finiteDifference) {
                    if (candidate.curve->identity() == block.curve->identity()) {
                        referenceBlock = &candidate;
                        break;
                    }
                }
                CHECK(referenceBlock != nullptr);
                CHECK(block.row.size() == referenceBlock->row.size());
                for (std::size_t i = 0; i < block.row.size(); ++i) {
                    util::checkClose("mtm analytic row vs fd", block.row[i], referenceBlock->row[i],
                                     1e-6);
                    if (block.curve->identity() == foreignDiscountView->identity()) {
                        foreignRisk += std::abs(block.row[i]);
                    }
                    if (block.curve->identity() == domesticDiscountView->identity()) {
                        domesticRisk += std::abs(block.row[i]);
                    }
                }
            }
            for (const markets::XccyRowBlock& block : finiteDifference) {
                const bool matched =
                    std::any_of(analytic.begin(), analytic.end(), [&](const auto& candidate) {
                        return candidate.curve->identity() == block.curve->identity();
                    });
                CHECK(matched);
            }
            // The reset adjustment ties the two discount curves, so both legs
            // must carry nonzero discount risk for both reset choices.
            CHECK(foreignRisk > 1e-8);
            CHECK(domesticRisk > 1e-8);
            QTA_LOG_INFO("test",
                         "mtm rows resetForeign={} spreadOnForeign={}: {} blocks, risk {} / {}",
                         resetForeign, spreadOnForeign, analytic.size(), foreignRisk, domesticRisk);
        }
    }
}

/// FX spot as a one-node identity factor block: the solved quote delta must
/// equal the supplied dV/dS, and that input must match a central difference of
/// the discounted forward value.
void testFxFactorBlockAndCashflowGreeks() {
    RootFixture fixture;
    const double spot = 1.10;
    const double strike = 1.13;
    const double notional = 3.0e6;
    const double maturity =
        datetime::yearFraction(fixture.reference, fixture.dates[1], fixture.zeroDc);
    const DiscountCurve<double> baseCurve =
        buildCurve(fixture.reference, fixture.zeroDc, fixture.dates, 0.020, 0.0010);
    const double baseDiscount = baseCurve.discount(maturity);
    const double quoteDiscount = fixture.root.discount(maturity);
    const double dVdSpot = notional * baseDiscount;

    const auto valueAt = [&](double s) {
        return markets::fxForwardPv(s, strike, baseDiscount, quoteDiscount, notional);
    };
    const double step = 1e-5;
    const double fdDelta = (valueAt(spot + step) - valueAt(spot - step)) / (2.0 * step);
    util::checkClose("fx factor dVdS vs fd", fdDelta / dVdSpot, 1.0, 1e-9);

    std::vector<markets::StackCurveInput> inputs(2);
    inputs[0].curve = markets::StackCurveView::make(fixture.root);
    inputs[0].role = markets::CurveRole::Discount;
    inputs[0].discountPillars = fixture.pillars;
    inputs[0].dVdNodes.assign(fixture.root.size(), 0.0);
    inputs[1] =
        markets::makeFxSpotFactorInput(markets::makeFxFactorAnchorView(), dVdSpot, "EURUSD");
    CHECK(markets::stackFactorNodeCount(inputs[1]) == 1);

    const std::vector<markets::StackRiskEntry> entries =
        markets::stackQuoteRisk(inputs, fixture.reference);
    CHECK(entries.size() == 2);
    CHECK(entries[1].role == markets::CurveRole::FxSpot);
    CHECK(entries[1].points.size() == 1);
    CHECK(entries[1].points[0].label == "FxSpot EURUSD");
    CHECK(entries[1].points[0].bucket == "FxSpot");
    CHECK(entries[1].points[0].year == 0);
    util::checkClose("fx factor block delta", entries[1].points[0].delta, dVdSpot, 1e-12);

    const std::vector<markets::FxRiskPoint> table = markets::fxRiskTable(entries);
    CHECK(table.size() == 1);
    CHECK(table[0].pair == "EURUSD");
    CHECK(table[0].role == markets::CurveRole::FxSpot);
    util::checkClose("fx side table delta", table[0].delta, dVdSpot, 1e-12);

    // Cashflow greeks: the analytic base-notional delta, its reverse-mode
    // adjoint and the zero gamma, with the finite-difference fallback.
    const std::vector<double> baseNotionals{1.5e5, 2.5e5};
    const std::vector<double> baseDiscounts{0.98, 0.95};
    const std::vector<double> quoteCashflows{-4.0e4};
    const std::vector<double> quoteDiscounts{0.99};
    const double analyticDelta = 1.5e5 * 0.98 + 2.5e5 * 0.95;
    const markets::FxGreeks<double> greeks =
        markets::fxCashflowGreeks(baseNotionals, baseDiscounts);
    util::checkClose("fx cashflow delta", greeks.delta, analyticDelta, 1e-9);
    util::checkClose("fx cashflow gamma", greeks.gamma, 0.0, 0.0);
    const double cashflowPv =
        markets::fxCashflowPv(spot, baseNotionals, baseDiscounts, quoteCashflows, quoteDiscounts);
    util::checkClose("fx cashflow pv", cashflowPv,
                     spot * analyticDelta + quoteCashflows[0] * quoteDiscounts[0], 1e-9);
    const auto cashflowValue = [&](double s) {
        return markets::fxCashflowPv(s, baseNotionals, baseDiscounts, quoteCashflows,
                                     quoteDiscounts);
    };
    const markets::FxGreeks<double> fallback =
        markets::fxGreeksFiniteDifference(cashflowValue, spot);
    util::checkClose("fx cashflow fd delta", fallback.delta / analyticDelta, 1.0, 1e-9);

    // The finite-difference fallback recovers a convex spot gamma on a
    // quadratic evaluator with no cashflow representation.
    const double convexity = 3.0;
    const auto convexValue = [&](double s) {
        return 0.5 * convexity * (s - 1.0) * (s - 1.0) + 2.0 * s;
    };
    const markets::FxGreeks<double> convexGreeks =
        markets::fxGreeksFiniteDifference(convexValue, spot, 1e-3);
    util::checkClose("fx fallback convex delta", convexGreeks.delta, convexity * (spot - 1.0) + 2.0,
                     1e-6);
    util::checkClose("fx fallback convex gamma", convexGreeks.gamma, convexity, 1e-6);

    stan::math::recover_memory();
    {
        var spotVar(spot);
        var value =
            markets::fxCashflowPv(spotVar, std::vector<var>{var(1.5e5), var(2.5e5)},
                                  std::vector<var>{var(0.98), var(0.95)},
                                  std::vector<var>{var(-4.0e4)}, std::vector<var>{var(0.99)});
        value.grad();
        util::checkClose("fx cashflow spot adjoint", spotVar.adj(), analyticDelta, 1e-9);
    }
    stan::math::recover_memory();
}

/// FX gamma and FXxIR cross-gamma through the stack engine: the diagonal FX
/// entry is the mixed-coordinate Hessian input, the cross entries match the
/// dense transform J^T H J, and the whole matrix agrees with a central
/// difference of the solved quote deltas while the spot moves.
void testFxGammaAndCrossGamma() {
    RootFixture fixture;
    const std::size_t rootNodes = fixture.root.size() - 1;
    const std::size_t dim = rootNodes + 1;
    const std::size_t fxNode = rootNodes;
    std::vector<markets::StackCurveInput> inputs(2);
    inputs[0].curve = markets::StackCurveView::make(fixture.root);
    inputs[0].role = markets::CurveRole::Discount;
    inputs[0].discountPillars = fixture.pillars;
    inputs[0].dVdNodes.assign(fixture.root.size(), 0.0);
    inputs[1] = markets::makeFxSpotFactorInput(markets::makeFxFactorAnchorView(), 0.02, "EURUSD");

    // Mixed-coordinate Hessian: spot gamma, spot/root cross, root curvature.
    std::vector<double> hZeta(dim * dim, 0.0);
    hZeta[fxNode * dim + fxNode] = 2.5;
    hZeta[fxNode * dim + 0] = -0.4;
    hZeta[0 * dim + fxNode] = -0.4;
    hZeta[0 * dim + 0] = 1.2;
    hZeta[1 * dim + 1] = 0.8;
    hZeta[2 * dim + 2] = 0.6;
    const std::vector<double> g0{0.001, -0.002, 0.0015, 0.02};
    for (std::size_t i = 0; i < rootNodes; ++i) {
        inputs[0].dVdNodes[i + 1] = g0[i];
    }

    markets::StackQuoteGamma gamma = markets::stackQuoteGamma(inputs, hZeta, fixture.reference);
    CHECK(gamma.dim == dim);
    std::size_t fxIndex = dim;
    for (std::size_t i = 0; i < gamma.points.size(); ++i) {
        if (gamma.points[i].role == markets::CurveRole::FxSpot) {
            fxIndex = i;
        }
    }
    CHECK(fxIndex < dim);
    CHECK(gamma.points[fxIndex].label == "FxSpot EURUSD");
    util::checkClose("fx gamma diagonal", gamma.points[fxIndex].delta, 2.5, 1e-10);

    // Dense transform reference: J = F^{-1} and H_r = J^T (H - M) J with the
    // M correction zero on the FX row and column (identity factor rows never
    // re-solve), checked entry by entry against J^T H J.
    const markets::StackQuoteSystem system =
        markets::assembleStackQuoteSystem(inputs, fixture.reference);
    std::vector<double> jacobian(dim * dim, 0.0);
    std::vector<double> unit(dim, 0.0);
    for (std::size_t column = 0; column < dim; ++column) {
        std::fill(unit.begin(), unit.end(), 0.0);
        unit[column] = 1.0;
        const std::vector<double> solution = math::solveDense(system.jacobian, dim, unit);
        for (std::size_t k = 0; k < dim; ++k) {
            jacobian[k * dim + column] = solution[k];
        }
    }
    for (std::size_t i = 0; i < dim; ++i) {
        for (std::size_t j = 0; j < dim; ++j) {
            util::checkClose("fx gamma symmetry", gamma.at(i, j), gamma.at(j, i), 1e-12);
        }
    }
    for (std::size_t i = 0; i < dim; ++i) {
        double expected = 0.0;
        for (std::size_t b = 0; b < dim; ++b) {
            expected += hZeta[fxNode * dim + b] * jacobian[b * dim + i];
        }
        util::checkClose("fx cross gamma transform", gamma.at(fxIndex, i), expected, 1e-9);
    }

    // Central difference of the solved quote deltas while the spot node moves:
    // the node gradient is g(zeta) = g0 + H zeta with zeta_fx = spot.
    const auto solveAt = [&](double spotValue) {
        std::vector<double> zeta(dim, 0.0);
        for (std::size_t i = 0; i < rootNodes; ++i) {
            zeta[i] = fixture.root.zeros()[i + 1];
        }
        zeta[fxNode] = spotValue;
        std::vector<double> gradient(dim, 0.0);
        for (std::size_t a = 0; a < dim; ++a) {
            double sum = g0[a];
            for (std::size_t b = 0; b < dim; ++b) {
                sum += hZeta[a * dim + b] * zeta[b];
            }
            gradient[a] = sum;
        }
        std::vector<markets::StackCurveInput> moved = inputs;
        for (std::size_t i = 0; i < rootNodes; ++i) {
            moved[0].dVdNodes[i + 1] = gradient[i];
        }
        moved[1].dVdNodes[1] = gradient[fxNode];
        const std::vector<markets::StackRiskEntry> entries =
            markets::stackQuoteRisk(moved, fixture.reference);
        std::vector<double> solved;
        for (const markets::StackRiskEntry& entry : entries) {
            for (const markets::QuotePoint& point : entry.points) {
                solved.push_back(point.delta);
            }
        }
        return solved;
    };
    const double step = 1e-5;
    const std::vector<double> plus = solveAt(0.02 + step);
    const std::vector<double> minus = solveAt(0.02 - step);
    CHECK(plus.size() == dim);
    for (std::size_t a = 0; a < dim; ++a) {
        const double fd = (plus[a] - minus[a]) / (2.0 * step);
        util::checkClose("fx cross gamma vs fd", gamma.at(fxIndex, a), fd, 1e-7);
    }
    QTA_LOG_INFO("test", "fx gamma diag {} cross vs fd max {}", gamma.points[fxIndex].delta,
                 std::abs(gamma.at(fxIndex, 0) - (plus[0] - minus[0]) / (2.0 * step)));
}

/// Role names, role buckets, the maturity ladder and the FX greeks side table
/// all carry the new FX roles with their fixed labels.
void testFxRoleBucketsAndSideTable() {
    CHECK(markets::curveRoleName(markets::CurveRole::FxSpot) == "FxSpot");
    CHECK(markets::curveRoleName(markets::CurveRole::FxVol) == "FxVol");
    CHECK(markets::isFxRole(markets::CurveRole::FxSpot));
    CHECK(markets::isFxRole(markets::CurveRole::FxVol));
    CHECK(!markets::isFxRole(markets::CurveRole::Discount));

    std::vector<markets::StackRiskEntry> entries(2);
    entries[0].role = markets::CurveRole::Discount;
    entries[0].points.push_back(
        markets::QuotePoint{"OisSwap 1Y", "1Y", 1, markets::CurveRole::Discount, 100.0});
    entries[1].role = markets::CurveRole::FxSpot;
    entries[1].points.push_back(
        markets::QuotePoint{"FxSpot EURUSD", "FxSpot", 0, markets::CurveRole::FxSpot, 25.0});
    entries[1].points.push_back(
        markets::QuotePoint{"FxVol EURUSD 1Y", "FxVol", 1, markets::CurveRole::FxVol, -10.0});

    const std::vector<markets::RiskBucket> roles = markets::stackRoleBuckets(entries);
    const auto findBucket = [](const std::vector<markets::RiskBucket>& buckets,
                               const std::string& label) {
        for (const markets::RiskBucket& bucket : buckets) {
            if (bucket.label == label) {
                return bucket.delta;
            }
        }
        return 0.0;
    };
    util::checkClose("fx spot role bucket", findBucket(roles, "FxSpot"), 25.0, 1e-12);
    util::checkClose("fx vol role bucket", findBucket(roles, "FxVol"), -10.0, 1e-12);
    util::checkClose("discount role bucket", findBucket(roles, "Discount"), 100.0, 1e-12);

    const std::vector<markets::RiskBucket> ladder = markets::stackYearLadder(entries);
    util::checkClose("fx spot ladder bucket", findBucket(ladder, "FxSpot"), 25.0, 1e-12);
    util::checkClose("fx vol ladder bucket", findBucket(ladder, "FxVol"), -10.0, 1e-12);
    util::checkClose("year ladder discount bucket", findBucket(ladder, "1Y"), 100.0, 1e-12);
    double ladderTotal = 0.0;
    for (const markets::RiskBucket& bucket : ladder) {
        ladderTotal += bucket.delta;
    }
    util::checkClose("year ladder total", ladderTotal, 115.0, 1e-12);

    const std::vector<markets::FxRiskPoint> table = markets::fxRiskTable(entries);
    CHECK(table.size() == 2);
    CHECK(table[0].pair == "EURUSD");
    CHECK(table[0].role == markets::CurveRole::FxSpot);
    util::checkClose("fx table spot delta", table[0].delta, 25.0, 1e-12);
    CHECK(table[1].pair == "EURUSD");
    CHECK(table[1].role == markets::CurveRole::FxVol);
    util::checkClose("fx table vol delta", table[1].delta, -10.0, 1e-12);

    markets::StackQuoteGamma gamma;
    gamma.dim = 3;
    gamma.hessian = {0.0, 0.0, 0.0, 0.0, 0.5, 0.0, 0.0, 0.0, -0.25};
    const std::vector<markets::FxRiskPoint> withGamma = markets::fxRiskTable(entries, &gamma);
    CHECK(withGamma.size() == 2);
    util::checkClose("fx table spot gamma", withGamma[0].gamma, 0.5, 1e-12);
    util::checkClose("fx table vol gamma", withGamma[1].gamma, -0.25, 1e-12);
}

} // namespace

int main() {
    testFxSpotAndForwardPricing();
    testFxSwapPricing();
    testMtMRowsAgainstFiniteDifference();
    testFxFactorBlockAndCashflowGreeks();
    testFxGammaAndCrossGamma();
    testFxRoleBucketsAndSideTable();
    QTA_LOG_INFO("test", "fx pricing and risk gates passed");
    return 0;
}
