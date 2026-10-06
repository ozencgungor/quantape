/**
 * @file test_forecast_pillars.cpp
 * @brief Forecast-curve instruments: par IRS pillars, IRBS conventions, overlap
 *
 * Covers fixed-vs-float par IRS swaps with exogenous discounting bootstrapped
 * alongside basis swaps, the two IRBS quoting conventions and their
 * annuity-ratio relation Delta_two = Delta_single * A_y / A_fixed, the
 * priority filter for overlapping futures/deposits/FRAs, and forecast-curve
 * rate futures with the discount-curve style, convexity and fixing-grid
 * semantics.
 */

#include "quantape/markets/Curves/CurveBuilder.h"
#include "quantape/markets/Curves/StackRisk.h"

#include <cmath>
#include <cstddef>
#include <memory>
#include <string>
#include <vector>

#include "support/GtestSupport.h"

using namespace quantape;

namespace {

using markets::BasisPillar;
using markets::CurvePillar;
using markets::DiscountCurve;
using markets::ForecastPillar;
using markets::InterpolationScheme;
using markets::InterpolationSpace;
using markets::IrsPillar;
using markets::PillarKind;
using markets::SpreadCurve;

const datetime::Date kReference(2026, 9, 29);
const datetime::DayCounter kZeroDc(datetime::DayCount::Actual365Fixed);
const datetime::Calendar kCalendar = datetime::Calendar::noHolidays();

std::shared_ptr<DiscountCurve<double>> makeParent(int years, double base) {
    std::vector<datetime::Date> dates;
    std::vector<double> zeros;
    for (int year = 1; year <= years; ++year) {
        dates.push_back(kReference.plusYears(year));
        zeros.push_back(base + 0.0005 * static_cast<double>(year));
    }
    return std::make_shared<DiscountCurve<double>>(kReference, dates, kZeroDc, zeros,
                                                   InterpolationSpace::LogDiscount,
                                                   InterpolationScheme::Linear);
}

std::shared_ptr<SpreadCurve<double>> makeChild(const std::shared_ptr<DiscountCurve<double>>& parent,
                                               int years, double spreadBase, double spreadSlope) {
    std::vector<double> times{0.0};
    std::vector<double> spreads{0.0};
    for (int year = 1; year <= years; ++year) {
        const datetime::Date date = kReference.plusYears(year);
        times.push_back(datetime::yearFraction(kReference, date, kZeroDc));
        spreads.push_back(spreadBase + spreadSlope * static_cast<double>(year));
    }
    return std::make_shared<SpreadCurve<double>>(parent, times, spreads,
                                                 InterpolationScheme::Linear);
}

BasisPillar basisPillar(const datetime::Date& maturity) {
    BasisPillar pillar;
    pillar.maturity = maturity;
    pillar.floatTenor = datetime::Period(3, datetime::TimeUnit::Months);
    pillar.quoteDayCounter = datetime::DayCounter(datetime::DayCount::Actual360);
    pillar.calendar = kCalendar;
    return pillar;
}

IrsPillar irsPillar(const datetime::Date& maturity) {
    IrsPillar pillar;
    pillar.maturity = maturity;
    pillar.floatTenor = datetime::Period(3, datetime::TimeUnit::Months);
    pillar.floatCalendar = kCalendar;
    pillar.floatDayCounter = datetime::DayCounter(datetime::DayCount::Actual360);
    pillar.fixedTenor = datetime::Period(1, datetime::TimeUnit::Years);
    pillar.fixedCalendar = kCalendar;
    pillar.fixedDayCounter = datetime::DayCounter(datetime::DayCount::Thirty360BondBasis);
    return pillar;
}

ForecastPillar futurePillar(const datetime::Date& start, const datetime::Date& maturity,
                            markets::FutureStyle style, markets::AveragingStyle averaging,
                            double convexity) {
    ForecastPillar pillar;
    pillar.kind = ForecastPillar::Kind::Future;
    pillar.start = start;
    pillar.maturity = maturity;
    pillar.futureStyle = style;
    pillar.averagingStyle = averaging;
    pillar.convexityAdjustment = convexity;
    pillar.quoteDayCounter = datetime::DayCounter(datetime::DayCount::Actual360);
    pillar.calendar = kCalendar;
    return pillar;
}

struct FutureStyleCase {
    markets::FutureStyle style;
    markets::AveragingStyle averaging;
    const char* tag;
};

const FutureStyleCase kFutureStyleCases[] = {
    {markets::FutureStyle::Simple, markets::AveragingStyle::Arithmetic, "future simple"},
    {markets::FutureStyle::Compounded, markets::AveragingStyle::Arithmetic, "future compounded"},
    {markets::FutureStyle::Averaged, markets::AveragingStyle::Arithmetic,
     "future averaged arithmetic"},
    {markets::FutureStyle::Averaged, markets::AveragingStyle::Compounded,
     "future averaged compounded"},
};

} // namespace

TEST(ForecastIrs, irsParRateSingleCurve) {
    // A zero-spread child over the parent, with the same fixed and floating
    // schedule, must price at the parent's own par rate (cashflows telescope).
    const auto parent = makeParent(5, 0.03);
    const auto child = makeChild(parent, 5, 0.0, 0.0);
    const IrsPillar pillar = irsPillar(kReference.plusYears(3));
    const double rate = markets::impliedIrsRate(*child, *parent, pillar, kReference, kZeroDc);
    const double t = datetime::yearFraction(kReference, pillar.maturity, kZeroDc);
    // The floating leg telescopes to 1 - D(T); the par denominator is the
    // fixed annuity.
    CHECK_CLOSE("irs zero-spread identity", rate,
                (1.0 - parent->discount(t)) /
                    markets::fixedAnnuity(*parent, pillar, kReference, kZeroDc),
                1e-12);
}

TEST(ForecastIrs, irsBootstrapMixed) {
    const auto parent = makeParent(12, 0.032);
    const auto target = makeChild(parent, 12, 0.0008, 0.00008);

    std::vector<ForecastPillar> instruments;
    for (int year : {1, 2}) {
        ForecastPillar out;
        out.kind = ForecastPillar::Kind::BasisSwap;
        out.basis = basisPillar(kReference.plusYears(year));
        out.basis.spread = markets::impliedBasisSpread(*target, out.basis, kReference, kZeroDc);
        instruments.push_back(out);
    }
    for (int year : {3, 5, 7, 10}) {
        ForecastPillar out;
        out.kind = ForecastPillar::Kind::Irs;
        out.irs = irsPillar(kReference.plusYears(year));
        out.irs.quote = markets::impliedIrsRate(*target, *parent, out.irs, kReference, kZeroDc);
        instruments.push_back(out);
    }
    const SpreadCurve<double> child = markets::bootstrapForecastCurve(
        parent, nullptr, kReference, kZeroDc, InterpolationScheme::Linear, instruments);
    for (const ForecastPillar& instrument : instruments) {
        CHECK_CLOSE("mixed forecast reprice",
                    markets::impliedForecastQuote(child, *parent, instrument, kReference, kZeroDc),
                    instrument.kind == ForecastPillar::Kind::Irs ? instrument.irs.quote
                                                                 : instrument.basis.spread,
                    1e-9);
    }
    // Node-aligned cashflows under Linear interpolation recover the target
    // spread at the instrument maturities (adjusted dates differ from the
    // target grid by at most one calendar day at a weekend).
    for (std::size_t i = 0; i < instruments.size(); ++i) {
        const datetime::Date maturity = instruments[i].kind == ForecastPillar::Kind::Irs
                                            ? instruments[i].irs.maturity
                                            : instruments[i].basis.maturity;
        const double t = datetime::yearFraction(kReference, maturity, kZeroDc);
        CHECK_CLOSE("mixed forecast node recovery", child.spreadNodes().zeros()[i + 1],
                    target->spread(t), 1e-6);
    }
}

TEST(ForecastIrs, irsExogenousDiscounting) {
    const auto parent = makeParent(12, 0.032);
    const auto discountCurve = makeParent(12, 0.036); // different collateral curve
    const auto target = makeChild(parent, 12, 0.0008, 0.00008);

    std::vector<ForecastPillar> instruments;
    for (int year : {2, 5, 10}) {
        ForecastPillar out;
        out.kind = ForecastPillar::Kind::Irs;
        out.irs = irsPillar(kReference.plusYears(year));
        out.irs.quote =
            markets::impliedIrsRate(*target, *discountCurve, out.irs, kReference, kZeroDc);
        instruments.push_back(out);
    }
    const SpreadCurve<double> child = markets::bootstrapForecastCurve(
        parent, discountCurve.get(), kReference, kZeroDc, InterpolationScheme::Linear, instruments);
    for (const ForecastPillar& instrument : instruments) {
        CHECK_CLOSE(
            "exogenous discount reprice",
            markets::impliedForecastQuote(child, *discountCurve, instrument, kReference, kZeroDc),
            instrument.irs.quote, 1e-9);
    }
}

TEST(ForecastIrs, irbsConventions) {
    const auto parent = makeParent(5, 0.03);
    const auto childX = makeChild(parent, 5, 0.0010, 0.0002);
    const auto childY = makeChild(parent, 5, 0.0004, 0.0001);

    const datetime::Date maturity = kReference.plusYears(3);
    const BasisPillar basis = basisPillar(maturity);
    const IrsPillar irs = irsPillar(maturity);

    // Single-IRS convention: float-vs-float spread on the y leg.
    const double single = markets::impliedBasisSpread(*childX, *childY, basis, kReference, kZeroDc);
    // Two-IRS convention: difference of the two par rates, R_x - R_y.
    const double rateX = markets::impliedIrsRate(*childX, *parent, irs, kReference, kZeroDc);
    const double rateY = markets::impliedIrsRate(*childY, *parent, irs, kReference, kZeroDc);
    const double two = markets::basisSpreadAsIrsRateDifference(rateX, rateY);

    // Convention switch: Delta_two = Delta_single * A_y / A_fixed.
    const double annuityY = markets::legAnnuity(
        *parent, kReference, kReference, maturity, basis.floatTenor, basis.calendar,
        basis.quoteDayCounter, basis.businessDayConvention, 0, kZeroDc);
    const double annuityFixed = markets::fixedAnnuity(*parent, irs, kReference, kZeroDc);
    CHECK_CLOSE("irbs convention switch", two,
                markets::basisSpreadConventionSwitch(single, annuityY, annuityFixed), 1e-12);

    EXPECT_THROW((void)markets::basisSpreadConventionSwitch(single, annuityY, 0.0),
                 std::invalid_argument);
}

TEST(ForecastOverlap, overlapFilter) {
    std::vector<CurvePillar> pillars;
    CurvePillar deposit;
    deposit.kind = PillarKind::Deposit;
    deposit.maturity = datetime::Date(2027, 3, 29);
    pillars.push_back(deposit);
    CurvePillar future;
    future.kind = PillarKind::Future;
    future.start = datetime::Date(2027, 3, 30);
    future.maturity = datetime::Date(2027, 3, 30);
    pillars.push_back(future);
    CurvePillar fra;
    fra.kind = PillarKind::Fra;
    fra.start = datetime::Date(2027, 3, 29);
    fra.maturity = datetime::Date(2027, 6, 29);
    pillars.push_back(fra);
    CurvePillar swap;
    swap.kind = PillarKind::OisSwap;
    swap.maturity = datetime::Date(2027, 6, 29);
    pillars.push_back(swap);
    CurvePillar far;
    far.kind = PillarKind::OisSwap;
    far.maturity = datetime::Date(2028, 3, 29);
    pillars.push_back(far);

    const std::vector<CurvePillar> filtered = markets::filterOverlappingPillars(pillars, 2);
    EXPECT_TRUE(filtered.size() == 3);
    EXPECT_TRUE(filtered[0].kind == PillarKind::Future); // beats the overlapping deposit
    EXPECT_TRUE(filtered[1].kind == PillarKind::Fra);    // beats the same-maturity swap
    EXPECT_TRUE(filtered[2].kind == PillarKind::OisSwap);
    EXPECT_TRUE(filtered[2].maturity == datetime::Date(2028, 3, 29));

    EXPECT_THROW((void)markets::filterOverlappingPillars(pillars, -1), std::invalid_argument);
    // Clusters chain: maturities 0, 2 and 3 with a two-day window collapse to one.
    std::vector<CurvePillar> chained;
    for (int days : {0, 2, 3}) {
        CurvePillar pillar;
        pillar.kind = PillarKind::OisSwap;
        pillar.maturity = datetime::Date(2027, 6, 28).plusDays(days);
        chained.push_back(pillar);
    }
    EXPECT_TRUE(markets::filterOverlappingPillars(chained, 2).size() == 1);
}

TEST(ForecastRisk, irsRiskRowsVsFiniteDifference) {
    const auto parent = makeParent(5, 0.03);
    const auto child = makeChild(parent, 5, 0.0008, 0.0001);
    const IrsPillar pillar = irsPillar(kReference.plusYears(3));
    std::vector<double> fRow;
    std::vector<double> parentRow;
    std::vector<double> discountRow;
    markets::irsSwapJacobianRows(*child, *parent, *parent, pillar, kReference, kZeroDc, fRow,
                                 parentRow, discountRow);

    const double epsilon = 1e-6;
    const auto& spreadNodes = child->spreadNodes();
    for (std::size_t k = 1; k < child->spreadNodes().size(); ++k) {
        std::vector<double> plus = spreadNodes.zeros();
        std::vector<double> minus = spreadNodes.zeros();
        plus[k] += epsilon;
        minus[k] -= epsilon;
        const SpreadCurve<double> childPlus(parent, spreadNodes.times(), plus, spreadNodes.scheme(),
                                            spreadNodes.tension());
        const SpreadCurve<double> childMinus(parent, spreadNodes.times(), minus,
                                             spreadNodes.scheme(), spreadNodes.tension());
        const double fd =
            (markets::impliedIrsRate(childPlus, *parent, pillar, kReference, kZeroDc) -
             markets::impliedIrsRate(childMinus, *parent, pillar, kReference, kZeroDc)) /
            (2.0 * epsilon);
        CHECK_CLOSE("irs own-row vs FD", fRow[k - 1], fd, 1e-6);
    }
    // With one object acting as both forecast parent and discount curve, the
    // total cross row is the parent row (forward-dependence of the child
    // curve) plus the discount row (annuity and coupon discounting).
    for (std::size_t i = 1; i < parent->size(); ++i) {
        const auto bumpedParent = [&](double delta) {
            std::vector<double> zeros = parent->zeros();
            zeros[i] += delta;
            return std::make_shared<DiscountCurve<double>>(parent->times(), zeros, parent->space(),
                                                           parent->scheme(), parent->tension(),
                                                           parent->switchIndex());
        };
        const SpreadCurve<double> childPlus(bumpedParent(epsilon), spreadNodes.times(),
                                            spreadNodes.zeros(), spreadNodes.scheme(),
                                            spreadNodes.tension());
        const SpreadCurve<double> childMinus(bumpedParent(-epsilon), spreadNodes.times(),
                                             spreadNodes.zeros(), spreadNodes.scheme(),
                                             spreadNodes.tension());
        const double fd = (markets::impliedIrsRate(childPlus, *bumpedParent(epsilon), pillar,
                                                   kReference, kZeroDc) -
                           markets::impliedIrsRate(childMinus, *bumpedParent(-epsilon), pillar,
                                                   kReference, kZeroDc)) /
                          (2.0 * epsilon);
        CHECK_CLOSE("irs cross-row vs FD", parentRow[i - 1] + discountRow[i - 1], fd, 1e-6);
    }
    // Split the discounting off as an independent equal curve so a forecast
    // parent bump leaves discounting frozen: the parent row alone then prices
    // the child-forward response, and the discount row prices the annuity and
    // coupon discounting response.
    const auto discountCopy = std::make_shared<DiscountCurve<double>>(*parent);
    std::vector<double> splitParentRow;
    std::vector<double> splitDiscountRow;
    markets::irsSwapJacobianRows(*child, *parent, *discountCopy, pillar, kReference, kZeroDc, fRow,
                                 splitParentRow, splitDiscountRow);
    for (std::size_t i = 1; i < parent->size(); ++i) {
        const auto bumpedParent = [&](double delta) {
            std::vector<double> zeros = parent->zeros();
            zeros[i] += delta;
            return std::make_shared<DiscountCurve<double>>(parent->times(), zeros, parent->space(),
                                                           parent->scheme(), parent->tension(),
                                                           parent->switchIndex());
        };
        const SpreadCurve<double> childPlus(bumpedParent(epsilon), spreadNodes.times(),
                                            spreadNodes.zeros(), spreadNodes.scheme(),
                                            spreadNodes.tension());
        const SpreadCurve<double> childMinus(bumpedParent(-epsilon), spreadNodes.times(),
                                             spreadNodes.zeros(), spreadNodes.scheme(),
                                             spreadNodes.tension());
        const double fdParent =
            (markets::impliedIrsRate(childPlus, *discountCopy, pillar, kReference, kZeroDc) -
             markets::impliedIrsRate(childMinus, *discountCopy, pillar, kReference, kZeroDc)) /
            (2.0 * epsilon);
        CHECK_CLOSE("irs parent-row vs FD", splitParentRow[i - 1], fdParent, 1e-6);

        std::vector<double> upZeros = parent->zeros();
        std::vector<double> downZeros = parent->zeros();
        upZeros[i] += epsilon;
        downZeros[i] -= epsilon;
        const DiscountCurve<double> discountUp(parent->times(), upZeros, parent->space(),
                                               parent->scheme(), parent->tension(),
                                               parent->switchIndex());
        const DiscountCurve<double> discountDown(parent->times(), downZeros, parent->space(),
                                                 parent->scheme(), parent->tension(),
                                                 parent->switchIndex());
        const double fdDiscount =
            (markets::impliedIrsRate(*child, discountUp, pillar, kReference, kZeroDc) -
             markets::impliedIrsRate(*child, discountDown, pillar, kReference, kZeroDc)) /
            (2.0 * epsilon);
        CHECK_CLOSE("irs discount-row vs FD", splitDiscountRow[i - 1], fdDiscount, 1e-6);

        CHECK_CLOSE("irs parent-row stable", splitParentRow[i - 1], parentRow[i - 1], 1e-15);
        CHECK_CLOSE("irs discount-row stable", splitDiscountRow[i - 1], discountRow[i - 1], 1e-15);
    }
}

TEST(ForecastRisk, simpleForecastRiskRowsVsFiniteDifference) {
    const auto parent = makeParent(5, 0.03);
    const auto child = makeChild(parent, 5, 0.0008, 0.0001);
    ForecastPillar deposit;
    deposit.kind = ForecastPillar::Kind::Deposit;
    deposit.start = kReference.plusMonths(3);
    deposit.maturity = kReference.plusMonths(15);
    deposit.quoteDayCounter = datetime::DayCounter(datetime::DayCount::Actual360);
    deposit.calendar = kCalendar;
    std::vector<double> fRow;
    std::vector<double> parentRow;
    std::vector<double> discountRow;
    markets::forecastPillarJacobianRows(*child, deposit, kReference, nullptr, fRow, parentRow,
                                        discountRow);
    EXPECT_TRUE(discountRow.size() == parent->size() - 1);
    for (const double value : discountRow) {
        CHECK_CLOSE("deposit zero discount row", value, 0.0, 1e-18);
    }
    // An exogenous discount curve sizes the zero discount row from its own
    // nodes, not from the forecast parent.
    const auto wideDiscount = makeParent(7, 0.031);
    markets::forecastPillarJacobianRows(*child, deposit, kReference, wideDiscount.get(), fRow,
                                        parentRow, discountRow);
    EXPECT_TRUE(discountRow.size() == wideDiscount->size() - 1);

    const double epsilon = 1e-6;
    const auto& spreadNodes = child->spreadNodes();
    for (std::size_t k = 1; k < spreadNodes.size(); ++k) {
        std::vector<double> plus = spreadNodes.zeros();
        std::vector<double> minus = spreadNodes.zeros();
        plus[k] += epsilon;
        minus[k] -= epsilon;
        const SpreadCurve<double> childPlus(parent, spreadNodes.times(), plus, spreadNodes.scheme(),
                                            spreadNodes.tension());
        const SpreadCurve<double> childMinus(parent, spreadNodes.times(), minus,
                                             spreadNodes.scheme(), spreadNodes.tension());
        const double fd =
            (markets::impliedForecastQuote(childPlus, *parent, deposit, kReference, kZeroDc) -
             markets::impliedForecastQuote(childMinus, *parent, deposit, kReference, kZeroDc)) /
            (2.0 * epsilon);
        CHECK_CLOSE("deposit own-row vs FD", fRow[k - 1], fd, 1e-6);
    }
    for (std::size_t i = 1; i < parent->size(); ++i) {
        const auto bumpedParent = [&](double delta) {
            std::vector<double> zeros = parent->zeros();
            zeros[i] += delta;
            return std::make_shared<DiscountCurve<double>>(parent->times(), zeros, parent->space(),
                                                           parent->scheme(), parent->tension(),
                                                           parent->switchIndex());
        };
        const SpreadCurve<double> childPlus(bumpedParent(epsilon), spreadNodes.times(),
                                            spreadNodes.zeros(), spreadNodes.scheme(),
                                            spreadNodes.tension());
        const SpreadCurve<double> childMinus(bumpedParent(-epsilon), spreadNodes.times(),
                                             spreadNodes.zeros(), spreadNodes.scheme(),
                                             spreadNodes.tension());
        const double fd = (markets::impliedForecastQuote(childPlus, *bumpedParent(epsilon), deposit,
                                                         kReference, kZeroDc) -
                           markets::impliedForecastQuote(childMinus, *bumpedParent(-epsilon),
                                                         deposit, kReference, kZeroDc)) /
                          (2.0 * epsilon);
        CHECK_CLOSE("deposit parent-row vs FD", parentRow[i - 1], fd, 1e-6);
    }
}

TEST(ForecastRisk, simpleForecastStartGuard) {
    // A deposit/FRA start before the reference date has no forward and must be
    // rejected by the risk rows exactly as pricing rejects it.
    const auto parent = makeParent(5, 0.03);
    const auto child = makeChild(parent, 5, 0.0008, 0.0001);
    ForecastPillar deposit;
    deposit.kind = ForecastPillar::Kind::Deposit;
    deposit.start = kReference - 1;
    deposit.maturity = kReference.plusMonths(12);
    deposit.quoteDayCounter = datetime::DayCounter(datetime::DayCount::Actual360);
    deposit.calendar = kCalendar;
    std::vector<double> fRow;
    std::vector<double> parentRow;
    std::vector<double> discountRow;
    EXPECT_THROW((void)markets::forecastPillarJacobianRows(*child, deposit, kReference, nullptr,
                                                           fRow, parentRow, discountRow),
                 std::invalid_argument);

    EXPECT_THROW(
        {
            const auto childView = markets::StackCurveView::make(*child);
            const auto parentView = markets::StackCurveView::make(*parent);
            (void)markets::forecastSimpleJacobianRowsView(*childView, *parentView, *parentView,
                                                          deposit, kReference, fRow, parentRow,
                                                          discountRow);
        },
        std::invalid_argument);
}

TEST(ForecastRisk, riskLabelsUseAdjustedDates) {
    // Labels and buckets follow the date each forecast pillar actually prices,
    // including the calendar roll of the schedule termination.
    const auto parent = makeParent(5, 0.03);
    const datetime::Calendar calendar =
        datetime::Calendar::weekendsOnly().withExtraHolidays({datetime::Date(2027, 1, 1)});
    ForecastPillar deposit;
    deposit.kind = ForecastPillar::Kind::Deposit;
    deposit.start = kReference;
    deposit.maturity = datetime::Date(2027, 1, 1);
    deposit.calendar = calendar;
    deposit.businessDayConvention = datetime::BusinessDayConvention::ModifiedFollowing;
    deposit.quoteDayCounter = datetime::DayCounter(datetime::DayCount::Actual360);
    const SpreadCurve<double> child = markets::bootstrapForecastCurve(
        parent, nullptr, kReference, kZeroDc, InterpolationScheme::Linear,
        std::vector<ForecastPillar>{deposit});
    markets::StackCurveInput input;
    input.curve = markets::StackCurveView::make(child);
    input.role = markets::CurveRole::Forecast;
    input.forecastPillars = {deposit};
    std::vector<markets::QuotePoint> points;
    markets::appendStackQuoteMetadata(input, kReference, points);
    EXPECT_TRUE(points.size() == 1);
    EXPECT_TRUE(points[0].label == "Deposit 04Jan27");
    EXPECT_TRUE(points[0].bucket == "04Jan27");
    EXPECT_TRUE(points[0].role == markets::CurveRole::Forecast);

    ForecastPillar irs;
    irs.kind = ForecastPillar::Kind::Irs;
    irs.irs = irsPillar(datetime::Date(2027, 1, 1));
    irs.irs.fixedCalendar = calendar;
    input.forecastPillars = {irs};
    points.clear();
    markets::appendStackQuoteMetadata(input, kReference, points);
    EXPECT_TRUE(points.size() == 1);
    EXPECT_TRUE(points[0].label == "Irs 04Jan27");

    ForecastPillar basis;
    basis.kind = ForecastPillar::Kind::BasisSwap;
    basis.basis = basisPillar(datetime::Date(2027, 1, 1));
    basis.basis.calendar = calendar;
    input.forecastPillars = {basis};
    points.clear();
    markets::appendStackQuoteMetadata(input, kReference, points);
    EXPECT_TRUE(points.size() == 1);
    EXPECT_TRUE(points[0].label == "Basis 04Jan27");

    // A turn pillar without an explicit start prices from and labels with the
    // reference date.
    ForecastPillar turn;
    turn.kind = ForecastPillar::Kind::Deposit;
    turn.turnPillar = true;
    turn.maturity = kReference.plusMonths(1);
    turn.calendar = kCalendar;
    turn.quoteDayCounter = datetime::DayCounter(datetime::DayCount::Actual360);
    input.forecastPillars = {turn};
    points.clear();
    markets::appendStackQuoteMetadata(input, kReference, points);
    EXPECT_TRUE(points.size() == 1);
    EXPECT_TRUE(points[0].label == "Turn 2026-09-29");
    EXPECT_TRUE(points[0].bucket == "Turn 2026-09-29");
    EXPECT_TRUE(points[0].role == markets::CurveRole::TurnOverlay);
}

TEST(ForecastChaining, chainedForecastCurves) {
    static_assert(markets::CurveNodeProvider<DiscountCurve<double>>);
    static_assert(markets::CurveNodeProvider<SpreadCurve<double>>);
    static_assert(markets::CurveNodeProvider<SpreadCurve<double, SpreadCurve<double>>>);

    std::vector<CurvePillar> deposits;
    for (int months : {3, 6, 12}) {
        CurvePillar deposit;
        deposit.kind = PillarKind::Deposit;
        deposit.maturity = kReference.plusMonths(months);
        deposit.quote = 0.03 + 0.0002 * static_cast<double>(months / 3);
        deposit.quoteDayCounter = datetime::DayCounter(datetime::DayCount::Actual360);
        deposit.calendar = kCalendar;
        deposits.push_back(deposit);
    }
    auto base = std::make_shared<DiscountCurve<double>>(
        markets::bootstrapDiscountCurve(kReference, kZeroDc, InterpolationSpace::LogDiscount,
                                        InterpolationScheme::Linear, deposits));

    // Depth-1 child: a 3M IBOR-vs-OIS spread curve over the base.
    const auto target = makeChild(base, 2, 0.0005, 0.0002);
    std::vector<ForecastPillar> childPillars;
    for (int year : {1, 2}) {
        ForecastPillar out;
        out.kind = ForecastPillar::Kind::BasisSwap;
        out.basis = basisPillar(kReference.plusYears(year));
        out.basis.spread = markets::impliedBasisSpread(*target, out.basis, kReference, kZeroDc);
        childPillars.push_back(out);
    }
    const auto child = std::make_shared<SpreadCurve<double>>(markets::bootstrapForecastCurve(
        base, nullptr, kReference, kZeroDc, InterpolationScheme::Linear, childPillars));
    for (const ForecastPillar& pillar : childPillars) {
        CHECK_CLOSE("chained child reprice",
                    markets::impliedForecastQuote(*child, *base, pillar, kReference, kZeroDc),
                    pillar.basis.spread, 1e-9);
    }

    // Depth-2 child: a second spread curve over the spread curve, priced and
    // bootstrapped with the depth-1 child as both parent and discount curve.
    const IrsPillar irs = irsPillar(kReference.plusYears(1));
    std::vector<double> grandTimes{0.0};
    std::vector<double> grandSpreads{0.0};
    for (int year : {1, 2}) {
        grandTimes.push_back(
            datetime::yearFraction(kReference, kReference.plusYears(year), kZeroDc));
        grandSpreads.push_back(0.0008 + 0.0003 * static_cast<double>(year));
    }
    const auto grandTarget = std::make_shared<SpreadCurve<double, SpreadCurve<double>>>(
        child, grandTimes, grandSpreads, InterpolationScheme::Linear);
    ForecastPillar grandPillar;
    grandPillar.kind = ForecastPillar::Kind::Irs;
    grandPillar.irs = irs;
    grandPillar.irs.quote =
        markets::impliedForecastQuote(*grandTarget, *child, grandPillar, kReference, kZeroDc);
    const SpreadCurve<double, SpreadCurve<double>> grandchild = markets::bootstrapForecastCurve(
        child, nullptr, kReference, kZeroDc, InterpolationScheme::Linear,
        std::vector<ForecastPillar>{grandPillar});
    CHECK_CLOSE("chained grandchild reprice",
                markets::impliedForecastQuote(grandchild, *child, grandPillar, kReference, kZeroDc),
                grandPillar.irs.quote, 1e-9);
    CHECK_CLOSE("chained grandchild node", grandchild.spreadNodes().zeros()[1], grandSpreads[1],
                1e-9);
    EXPECT_TRUE(grandchild.parentPointer().get() == child.get());
    EXPECT_TRUE(grandchild.parentPointer()->parentPointer().get() == base.get());
    for (const double t : {0.25, 0.5, 1.0, 1.5, 2.0}) {
        CHECK_CLOSE("chained child composition", child->zero(t), base->zero(t) + child->spread(t),
                    1e-15);
        CHECK_CLOSE("chained composition", grandchild.zero(t),
                    child->zero(t) + grandchild.spread(t), 1e-15);
        CHECK_CLOSE("chained composition to base", grandchild.zero(t),
                    base->zero(t) + child->spread(t) + grandchild.spread(t), 1e-15);
        CHECK_CLOSE("chained discount", grandchild.discount(t),
                    std::exp(-(base->zero(t) + child->spread(t) + grandchild.spread(t)) * t),
                    1e-14);
    }
}

TEST(ForecastFuture, futureStylesReprice) {
    const auto parent = makeParent(5, 0.03);
    const datetime::Date starts[] = {datetime::Date(2027, 3, 17), datetime::Date(2027, 6, 16),
                                     datetime::Date(2027, 9, 15)};
    const datetime::Date maturities[] = {datetime::Date(2027, 6, 16), datetime::Date(2027, 9, 15),
                                         datetime::Date(2027, 12, 15)};
    std::vector<double> times{0.0};
    std::vector<double> spreads{0.0};
    for (int i = 0; i < 3; ++i) {
        times.push_back(datetime::yearFraction(kReference, maturities[i], kZeroDc));
        spreads.push_back(0.0008 + 0.0002 * static_cast<double>(i));
    }
    const auto target =
        std::make_shared<SpreadCurve<double>>(parent, times, spreads, InterpolationScheme::Linear);
    for (const FutureStyleCase& styleCase : kFutureStyleCases) {
        std::vector<ForecastPillar> instruments;
        for (int i = 0; i < 3; ++i) {
            ForecastPillar pillar =
                futurePillar(starts[i], maturities[i], styleCase.style, styleCase.averaging,
                             1e-5 * static_cast<double>(i + 1));
            pillar.quote =
                markets::impliedForecastQuote(*target, *parent, pillar, kReference, kZeroDc);
            instruments.push_back(pillar);
        }
        const SpreadCurve<double> child = markets::bootstrapForecastCurve(
            parent, nullptr, kReference, kZeroDc, InterpolationScheme::Linear, instruments);
        for (const ForecastPillar& instrument : instruments) {
            CHECK_CLOSE(
                styleCase.tag,
                markets::impliedForecastQuote(child, *parent, instrument, kReference, kZeroDc),
                instrument.quote, 1e-10);
        }
    }
}

TEST(ForecastFuture, futureRiskRowsVsFiniteDifference) {
    const auto parent = makeParent(5, 0.03);
    const auto child = makeChild(parent, 5, 0.0008, 0.0001);
    const double epsilon = 1e-6;
    const auto& spreadNodes = child->spreadNodes();
    for (const FutureStyleCase& styleCase : kFutureStyleCases) {
        const ForecastPillar pillar =
            futurePillar(kReference.plusMonths(4), kReference.plusMonths(16), styleCase.style,
                         styleCase.averaging, 1.5e-4);
        std::vector<double> fRow;
        std::vector<double> parentRow;
        std::vector<double> discountRow;
        markets::forecastPillarJacobianRows(*child, pillar, kReference, nullptr, fRow, parentRow,
                                            discountRow);
        EXPECT_TRUE(discountRow.size() == parent->size() - 1);
        for (const double value : discountRow) {
            CHECK_CLOSE(styleCase.tag, value, 0.0, 1e-18);
        }
        for (std::size_t k = 1; k < spreadNodes.size(); ++k) {
            std::vector<double> plus = spreadNodes.zeros();
            std::vector<double> minus = spreadNodes.zeros();
            plus[k] += epsilon;
            minus[k] -= epsilon;
            const SpreadCurve<double> childPlus(parent, spreadNodes.times(), plus,
                                                spreadNodes.scheme(), spreadNodes.tension());
            const SpreadCurve<double> childMinus(parent, spreadNodes.times(), minus,
                                                 spreadNodes.scheme(), spreadNodes.tension());
            const double fd =
                (markets::impliedForecastQuote(childPlus, *parent, pillar, kReference, kZeroDc) -
                 markets::impliedForecastQuote(childMinus, *parent, pillar, kReference, kZeroDc)) /
                (2.0 * epsilon);
            CHECK_CLOSE(styleCase.tag, fRow[k - 1], fd, 1e-6);
        }
        for (std::size_t i = 1; i < parent->size(); ++i) {
            const auto bumpedParent = [&](double delta) {
                std::vector<double> zeros = parent->zeros();
                zeros[i] += delta;
                return std::make_shared<DiscountCurve<double>>(
                    parent->times(), zeros, parent->space(), parent->scheme(), parent->tension(),
                    parent->switchIndex());
            };
            const SpreadCurve<double> childPlus(bumpedParent(epsilon), spreadNodes.times(),
                                                spreadNodes.zeros(), spreadNodes.scheme(),
                                                spreadNodes.tension());
            const SpreadCurve<double> childMinus(bumpedParent(-epsilon), spreadNodes.times(),
                                                 spreadNodes.zeros(), spreadNodes.scheme(),
                                                 spreadNodes.tension());
            const double fd = (markets::impliedForecastQuote(childPlus, *bumpedParent(epsilon),
                                                             pillar, kReference, kZeroDc) -
                               markets::impliedForecastQuote(childMinus, *bumpedParent(-epsilon),
                                                             pillar, kReference, kZeroDc)) /
                              (2.0 * epsilon);
            CHECK_CLOSE(styleCase.tag, parentRow[i - 1], fd, 1e-6);
        }
        // The view-based rows are the same formulas over the type-erased views.
        const auto childView = markets::StackCurveView::make(*child);
        const auto parentView = markets::StackCurveView::make(*parent);
        std::vector<double> viewOwn;
        std::vector<double> viewParent;
        std::vector<double> viewDiscount;
        markets::forecastFutureJacobianRowsView(*childView, *parentView, *parentView, pillar,
                                                kReference, viewOwn, viewParent, viewDiscount);
        CHECK_CLOSE_SEQ(styleCase.tag, viewOwn, fRow, 1e-15);
        CHECK_CLOSE_SEQ(styleCase.tag, viewParent, parentRow, 1e-15);
        EXPECT_TRUE(viewDiscount.size() == parent->size() - 1);
    }
}

TEST(ForecastFuture, futureAssembledRows) {
    const auto target = makeParent(5, 0.03);
    std::vector<CurvePillar> rootPillars;
    for (int year = 1; year <= 5; ++year) {
        CurvePillar deposit;
        deposit.kind = PillarKind::Deposit;
        deposit.maturity = kReference.plusYears(year);
        deposit.quoteDayCounter = kZeroDc;
        deposit.calendar = kCalendar;
        deposit.quote = markets::impliedQuote(deposit, kReference, *target);
        rootPillars.push_back(deposit);
    }
    const DiscountCurve<double> root =
        markets::bootstrapDiscountCurve(kReference, kZeroDc, InterpolationSpace::LogDiscount,
                                        InterpolationScheme::Linear, rootPillars);
    const auto rootPtr = std::make_shared<DiscountCurve<double>>(root);

    const datetime::Date start = kReference.plusMonths(4);
    const datetime::Date maturity = kReference.plusMonths(16);
    std::vector<double> times{0.0, datetime::yearFraction(kReference, maturity, kZeroDc)};
    std::vector<double> spreads{0.0, 0.0009};
    const auto targetChild =
        std::make_shared<SpreadCurve<double>>(rootPtr, times, spreads, InterpolationScheme::Linear);
    ForecastPillar pillar = futurePillar(start, maturity, markets::FutureStyle::Averaged,
                                         markets::AveragingStyle::Arithmetic, 1.5e-4);
    pillar.quote =
        markets::impliedForecastQuote(*targetChild, *rootPtr, pillar, kReference, kZeroDc);
    const SpreadCurve<double> child = markets::bootstrapForecastCurve(
        rootPtr, nullptr, kReference, kZeroDc, InterpolationScheme::Linear,
        std::vector<ForecastPillar>{pillar});

    std::vector<double> ownRow;
    std::vector<double> parentRow;
    std::vector<double> discountRow;
    markets::forecastPillarJacobianRows(child, pillar, kReference, &root, ownRow, parentRow,
                                        discountRow);

    markets::StackCurveInput rootInput;
    rootInput.curve = markets::StackCurveView::make(*rootPtr);
    rootInput.role = markets::CurveRole::Discount;
    rootInput.discountPillars = rootPillars;
    rootInput.dVdNodes.assign(root.size(), 0.0);
    markets::StackCurveInput childInput;
    childInput.curve = markets::StackCurveView::make(child);
    childInput.role = markets::CurveRole::Forecast;
    childInput.forecastPillars = {pillar};
    childInput.dVdNodes.assign(child.size(), 0.0);
    const markets::StackQuoteSystem system =
        markets::assembleStackQuoteSystem({rootInput, childInput}, kReference);
    const std::size_t rootNodes = root.size() - 1;
    EXPECT_TRUE(system.dim == rootNodes + 1);
    const std::size_t row = rootNodes;
    for (std::size_t i = 0; i < ownRow.size(); ++i) {
        CHECK_CLOSE("future assembled own row", system.jacobian[row * system.dim + rootNodes + i],
                    ownRow[i], 1e-15);
    }
    for (std::size_t i = 0; i < parentRow.size(); ++i) {
        CHECK_CLOSE("future assembled cross row", system.jacobian[row * system.dim + i],
                    parentRow[i], 1e-15);
    }
}

TEST(ForecastPillarValidation, pillarTargetValidation) {
    const auto parent = makeParent(5, 0.03);
    // A swap-kind pillar filled only with top-level fields must not bootstrap
    // against the zero-initialized kind fields.
    ForecastPillar irs;
    irs.kind = ForecastPillar::Kind::Irs;
    irs.maturity = kReference.plusYears(3);
    irs.quote = 0.03;
    EXPECT_THROW((void)markets::forecastPillarTarget(irs), std::invalid_argument);
    EXPECT_THROW((void)markets::forecastPillarQuotedMaturity(irs), std::invalid_argument);
    EXPECT_THROW((void)markets::forecastPillarRiskMaturity(irs), std::invalid_argument);
    EXPECT_THROW((void)markets::bootstrapForecastCurve(parent, nullptr, kReference, kZeroDc,
                                                       InterpolationScheme::Linear,
                                                       std::vector<ForecastPillar>{irs}),
                 std::invalid_argument);

    // The kind target is set, but the quote landed on the ignored top-level
    // field: a zero-target bootstrap must be rejected.
    ForecastPillar misplacedQuote;
    misplacedQuote.kind = ForecastPillar::Kind::Irs;
    misplacedQuote.irs = irsPillar(kReference.plusYears(3));
    misplacedQuote.quote = 0.03;
    EXPECT_THROW((void)markets::forecastPillarTarget(misplacedQuote), std::invalid_argument);

    // Zero leg tenor is rejected for swap kinds.
    ForecastPillar noFixedTenor;
    noFixedTenor.kind = ForecastPillar::Kind::Irs;
    noFixedTenor.irs = irsPillar(kReference.plusYears(3));
    noFixedTenor.irs.quote = 0.03;
    noFixedTenor.irs.fixedTenor = datetime::Period(0, datetime::TimeUnit::Months);
    EXPECT_THROW((void)markets::forecastPillarTarget(noFixedTenor), std::invalid_argument);

    ForecastPillar topLevelBasis;
    topLevelBasis.kind = ForecastPillar::Kind::BasisSwap;
    topLevelBasis.maturity = kReference.plusYears(2);
    EXPECT_THROW((void)markets::forecastPillarTarget(topLevelBasis), std::invalid_argument);

    ForecastPillar noFloatTenor;
    noFloatTenor.kind = ForecastPillar::Kind::BasisSwap;
    noFloatTenor.basis = basisPillar(kReference.plusYears(2));
    noFloatTenor.basis.floatTenor = datetime::Period(0, datetime::TimeUnit::Months);
    EXPECT_THROW((void)markets::forecastPillarTarget(noFloatTenor), std::invalid_argument);

    // A money-market pillar without a maturity is rejected by the node helper.
    ForecastPillar missingMaturity;
    missingMaturity.kind = ForecastPillar::Kind::Deposit;
    EXPECT_THROW((void)markets::forecastPillarQuotedMaturity(missingMaturity),
                 std::invalid_argument);
}

TEST(ForecastPillarValidation, futureStartGuards) {
    const auto parent = makeParent(5, 0.03);
    const auto child = makeChild(parent, 5, 0.0008, 0.0001);
    // The averaged arithmetic path must reject a fixing before the reference
    // date on both the forecast and the discount curve.
    ForecastPillar forecastFuture =
        futurePillar(kReference - 1, kReference.plusMonths(3), markets::FutureStyle::Averaged,
                     markets::AveragingStyle::Arithmetic, 0.0);
    EXPECT_THROW((void)markets::impliedForecastFuture(*child, forecastFuture, kReference, kZeroDc),
                 std::invalid_argument);

    CurvePillar discountFuture;
    discountFuture.kind = PillarKind::Future;
    discountFuture.start = kReference - 1;
    discountFuture.maturity = kReference.plusMonths(3);
    discountFuture.futureStyle = markets::FutureStyle::Averaged;
    discountFuture.averagingStyle = markets::AveragingStyle::Arithmetic;
    discountFuture.quoteDayCounter = datetime::DayCounter(datetime::DayCount::Actual360);
    discountFuture.calendar = kCalendar;
    EXPECT_THROW((void)markets::impliedQuote(discountFuture, kReference, *parent),
                 std::invalid_argument);
}
