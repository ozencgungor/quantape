/**
 * @file test_instrument_options.cpp
 * @brief Instrument options: FRA convexity and seasoned (already-fixed) coupons
 *
 * The FRA convexity follows the shifted-lognormal model: the market rate is
 * `R = ((1 + f tau) exp(C) - 1) / tau` with `C` from the index and discount
 * volatilities; the risk rows scale by `exp(C)`. Seasoned instruments give
 * the first floating coupon an already-known fixing.
 */

#include "quantape/log/Log.h"
#include "quantape/markets/Curves/CurveBuilder.h"
#include "quantape/markets/Curves/CurveConfig.h"
#include "quantape/markets/Curves/CurveRisk.h"
#include "quantape/markets/Curves/FraConvexity.h"
#include "quantape/markets/Curves/StackRisk.h"
#include "quantape/util/Check.h"

#include <cmath>
#include <cstddef>
#include <string>
#include <vector>

using namespace quantape;

namespace {

using markets::CurvePillar;
using markets::DiscountCurve;
using markets::InterpolationScheme;
using markets::InterpolationSpace;
using markets::PillarKind;

const datetime::Date kReference(2026, 9, 29);
const datetime::DayCounter kZeroDc(datetime::DayCount::Actual365Fixed);
const datetime::DayCounter kIndexDc(datetime::DayCount::Actual360);
const datetime::Calendar kCalendar = datetime::Calendar::noHolidays();

DiscountCurve<double> makeTarget(const std::vector<datetime::Date>& dates, double base,
                                 double slope) {
    std::vector<double> zeros;
    for (const datetime::Date& date : dates) {
        zeros.push_back(base + slope * datetime::yearFraction(kReference, date, kZeroDc));
    }
    return DiscountCurve<double>(kReference, dates, kZeroDc, zeros, InterpolationSpace::LogDiscount,
                                 InterpolationScheme::Linear);
}

std::vector<CurvePillar> convexityPillars(const DiscountCurve<double>& target, double exponent) {
    std::vector<CurvePillar> pillars;
    CurvePillar deposit;
    deposit.maturity = kReference.plusYears(1);
    deposit.kind = PillarKind::Deposit;
    deposit.quoteDayCounter = kZeroDc;
    deposit.calendar = kCalendar;
    deposit.quote = markets::impliedQuote(deposit, kReference, target);
    pillars.push_back(deposit);
    CurvePillar fra;
    fra.kind = PillarKind::Fra;
    fra.start = kReference.plusYears(1);
    fra.maturity = datetime::Period(3, datetime::TimeUnit::Months).advance(fra.start);
    fra.quoteDayCounter = kIndexDc;
    fra.calendar = kCalendar;
    fra.fraConvexityExponent = exponent;
    fra.quote = markets::impliedQuote(fra, kReference, target);
    pillars.push_back(fra);
    return pillars;
}

void testFraConvexityPricing() {
    std::vector<datetime::Date> dates;
    for (int year = 1; year <= 4; ++year) {
        dates.push_back(kReference.plusYears(year));
    }
    const DiscountCurve<double> target = makeTarget(dates, 0.03, 0.0005);
    const double exponent = 4e-5;
    std::vector<CurvePillar> pillars = convexityPillars(target, exponent);
    const DiscountCurve<double> curve = markets::bootstrapDiscountCurve(
        kReference, kZeroDc, InterpolationSpace::LogDiscount, InterpolationScheme::Linear, pillars);
    for (const CurvePillar& pillar : pillars) {
        util::checkClose("fra convexity reprice", markets::impliedQuote(pillar, kReference, curve),
                         pillar.quote, 1e-9);
    }
    const CurvePillar& fra = pillars.back();
    const double t1 = datetime::yearFraction(kReference, fra.start, kZeroDc);
    const double t2 = datetime::yearFraction(kReference, fra.maturity, kZeroDc);
    const double tau = datetime::yearFraction(fra.start, fra.maturity, kIndexDc);
    const double forward = (curve.discount(t1) / curve.discount(t2) - 1.0) / tau;
    const double expectedForward = ((1.0 + fra.quote * tau) * std::exp(-exponent) - 1.0) / tau;
    util::checkClose("fra convexity forward", forward, expectedForward, 1e-10);

    CurvePillar plain = fra;
    plain.fraConvexityExponent = 0.0;
    util::checkClose("fra zero exponent identity", markets::impliedQuote(plain, kReference, curve),
                     (curve.discount(t1) / curve.discount(t2) - 1.0) / tau, 1e-15);
}

void testFraConvexityRiskRow() {
    std::vector<datetime::Date> dates;
    for (int year = 1; year <= 4; ++year) {
        dates.push_back(kReference.plusYears(year));
    }
    const DiscountCurve<double> target = makeTarget(dates, 0.03, 0.0005);
    const double exponent = 5e-5;
    std::vector<CurvePillar> pillars = convexityPillars(target, exponent);
    const DiscountCurve<double> curve = markets::bootstrapDiscountCurve(
        kReference, kZeroDc, InterpolationSpace::LogDiscount, InterpolationScheme::Linear, pillars);
    const CurvePillar& fra = pillars.back();
    std::vector<double> row;
    CHECK(markets::pillarJacobianRow(fra, kReference, curve, row));
    CurvePillar plain = fra;
    plain.fraConvexityExponent = 0.0;
    std::vector<double> plainRow;
    CHECK(markets::pillarJacobianRow(plain, kReference, curve, plainRow));
    for (std::size_t i = 0; i < row.size(); ++i) {
        util::checkClose("fra convexity row scaling", row[i], plainRow[i] * std::exp(exponent),
                         1e-12);
    }
    for (std::size_t i = 1; i < curve.size(); ++i) {
        std::vector<double> zeros = curve.zeros();
        zeros[i] += 1e-8;
        const DiscountCurve<double> bumped(curve.times(), zeros, InterpolationSpace::LogDiscount,
                                           InterpolationScheme::Linear);
        const double fd = (markets::impliedQuote(fra, kReference, bumped) -
                           markets::impliedQuote(fra, kReference, curve)) /
                          1e-8;
        util::checkClose("fra convexity row vs FD", row[i - 1], fd, 1e-5);
    }
}

void testFraConvexityModel() {
    util::checkClose("fra exponent at fixing", markets::fraConvexityExponent(0.01, 0.008, 0.5, 0.0),
                     0.0, 1e-18);
    const double exponent = markets::fraConvexityExponent(0.01, 0.008, 0.5, 2.0);
    util::checkClose("fra exponent value", exponent, (0.01 * 0.01 - 0.01 * 0.008 * 0.5) * 2.0,
                     1e-18);
    CHECK(markets::fraConvexityExponent(0.01, 0.008, -0.5, 2.0) > exponent);
    util::checkClose("fra market rate", markets::impliedFraMarketRate(0.03, 4e-5, 0.25),
                     ((1.0 + 0.03 * 0.25) * std::exp(4e-5) - 1.0) / 0.25, 1e-15);
}

void testFraConvexityConfig() {
    const std::string json = R"({
        "asOf": "2026-09-29",
        "curves": [{
            "key": {"currency": "USD", "role": "Discount", "collateral": "USD"},
            "zeroDayCounter": "ACT/365F",
            "interpolation": {"space": "LogDiscount", "scheme": "Linear"},
            "bootstrap": {"method": "IterativeSequential", "accuracy": 1e-14},
            "fraConvexity": {"model": "ShiftedLognormal", "sigmaIndex": 0.01,
                             "sigmaDiscount": 0.008, "correlation": 0.5},
            "pillars": [
                {"maturity": "2027-09-29", "kind": "Deposit", "quote": 0.031,
                 "quoteDayCounter": "ACT/365F", "calendar": "NoHolidays"},
                {"start": "2028-09-29", "maturity": "2028-12-29", "kind": "Fra",
                 "quote": 0.032, "quoteDayCounter": "ACT/360", "calendar": "NoHolidays"}
            ]
        }]
    })";
    const markets::CurveStackSpec stack = markets::parseCurveStackSpec(json);
    CHECK(stack.curves.front().fraConvexity.enabled);
    const DiscountCurve<double> curve = markets::buildCurve(stack, stack.curves.front());
    const markets::PillarSpec& fraSpec = stack.curves.front().pillars.back();
    CurvePillar fra;
    fra.kind = PillarKind::Fra;
    fra.start = fraSpec.start;
    fra.maturity = fraSpec.maturity;
    fra.quoteDayCounter = fraSpec.quoteDayCounter;
    fra.calendar = fraSpec.calendar;
    const double timeToFixing =
        datetime::yearFraction(stack.asOf, fraSpec.start, stack.curves.front().zeroDayCounter);
    fra.fraConvexityExponent = markets::fraConvexityExponent(0.01, 0.008, 0.5, timeToFixing);
    util::checkClose("fra convexity config reprice", markets::impliedQuote(fra, stack.asOf, curve),
                     fraSpec.quote, 1e-9);

    const std::string invalid = R"({
        "asOf": "2026-09-29",
        "curves": [{
            "key": {"currency": "USD", "role": "Discount", "collateral": "USD"},
            "zeroDayCounter": "ACT/365F",
            "interpolation": {"space": "LogDiscount", "scheme": "Linear"},
            "bootstrap": {"method": "IterativeSequential", "accuracy": 1e-14},
            "fraConvexity": {"model": "ShiftedLognormal", "sigmaIndex": 0.01,
                             "sigmaDiscount": 0.008, "correlation": 2.0},
            "pillars": [
                {"maturity": "2027-09-29", "kind": "Deposit", "quote": 0.031,
                 "quoteDayCounter": "ACT/365F", "calendar": "NoHolidays"}
            ]
        }]
    })";
    bool threw = false;
    try {
        (void)markets::parseCurveStackSpec(invalid);
    } catch (const std::invalid_argument&) {
        threw = true;
    }
    CHECK(threw);
}

void testSeasonedOisPricingAndRisk() {
    std::vector<datetime::Date> dates;
    for (int year = 1; year <= 4; ++year) {
        dates.push_back(kReference.plusYears(year));
    }
    const DiscountCurve<double> target = makeTarget(dates, 0.03, 0.0005);

    std::vector<CurvePillar> pillars;
    CurvePillar deposit;
    deposit.maturity = datetime::Period(6, datetime::TimeUnit::Months).advance(kReference);
    deposit.kind = PillarKind::Deposit;
    deposit.quoteDayCounter = kZeroDc;
    deposit.calendar = kCalendar;
    deposit.quote = markets::impliedQuote(deposit, kReference, target);
    pillars.push_back(deposit);
    CurvePillar seasoned;
    seasoned.kind = PillarKind::OisSwap;
    seasoned.start = kReference.plusMonths(-6); // seasoned: effective six months ago
    seasoned.maturity = datetime::Period(18, datetime::TimeUnit::Months).advance(kReference);
    seasoned.fixedTenor = datetime::Period(1, datetime::TimeUnit::Years);
    seasoned.quoteDayCounter = kZeroDc;
    seasoned.calendar = kCalendar;
    seasoned.firstCouponFixed = true;
    seasoned.firstCouponRate = 0.0315;
    seasoned.quote = markets::impliedQuote(seasoned, kReference, target);
    pillars.push_back(seasoned);

    const DiscountCurve<double> curve = markets::bootstrapDiscountCurve(
        kReference, kZeroDc, InterpolationSpace::LogDiscount, InterpolationScheme::Linear, pillars);
    for (const CurvePillar& pillar : pillars) {
        util::checkClose("seasoned ois reprice", markets::impliedQuote(pillar, kReference, curve),
                         pillar.quote, 1e-9);
    }
    // A different fixing produces a different quote for the same curve.
    CurvePillar other = seasoned;
    other.firstCouponRate = 0.0415;
    CHECK(std::abs(markets::impliedQuote(other, kReference, curve) - seasoned.quote) > 1e-4);
    // Risk row against FD, including the known first coupon.
    std::vector<double> row;
    CHECK(markets::pillarJacobianRow(seasoned, kReference, curve, row));
    for (std::size_t i = 1; i < curve.size(); ++i) {
        std::vector<double> zeros = curve.zeros();
        zeros[i] += 1e-8;
        const DiscountCurve<double> bumped(curve.times(), zeros, InterpolationSpace::LogDiscount,
                                           InterpolationScheme::Linear);
        const double fd = (markets::impliedQuote(seasoned, kReference, bumped) -
                           markets::impliedQuote(seasoned, kReference, curve)) /
                          1e-8;
        util::checkClose("seasoned ois jacobian vs FD", row[i - 1], fd, 1e-5);
    }
}

void testSeasonedIrsPricingAndRows() {
    std::vector<datetime::Date> dates;
    for (int year = 1; year <= 5; ++year) {
        dates.push_back(kReference.plusYears(year));
    }
    const DiscountCurve<double> parent = makeTarget(dates, 0.032, 0.0004);
    auto parentPtr = std::make_shared<DiscountCurve<double>>(parent);
    std::vector<double> times{0.0};
    std::vector<double> spreads{0.0};
    for (int year = 1; year <= 5; ++year) {
        times.push_back(datetime::yearFraction(kReference, kReference.plusYears(year), kZeroDc));
        spreads.push_back(0.0008 + 0.0001 * static_cast<double>(year));
    }
    auto target = std::make_shared<markets::SpreadCurve<double>>(parentPtr, times, spreads,
                                                                 InterpolationScheme::Linear);

    markets::IrsPillar irs;
    irs.start = kReference.plusMonths(-3); // one past (fixed) coupon
    irs.maturity = datetime::Period(30, datetime::TimeUnit::Months).advance(kReference);
    irs.floatTenor = datetime::Period(3, datetime::TimeUnit::Months);
    irs.floatCalendar = kCalendar;
    irs.floatDayCounter = kIndexDc;
    irs.fixedTenor = datetime::Period(1, datetime::TimeUnit::Years);
    irs.fixedCalendar = kCalendar;
    irs.firstCouponFixed = true;
    irs.firstCouponRate = 0.034;
    irs.quote = markets::impliedIrsRate(*target, parent, irs, kReference, kZeroDc);

    std::vector<markets::ForecastPillar> instruments;
    markets::ForecastPillar out;
    out.kind = markets::ForecastPillar::Kind::Irs;
    out.irs = irs;
    instruments.push_back(out);
    const markets::SpreadCurve<double> child = markets::bootstrapForecastCurve(
        parentPtr, nullptr, kReference, kZeroDc, InterpolationScheme::Linear, instruments);
    util::checkClose("seasoned irs reprice",
                     markets::impliedForecastQuote(child, parent, out, kReference, kZeroDc),
                     irs.quote, 1e-9);

    std::vector<double> fRow;
    std::vector<double> parentRow;
    std::vector<double> discountRow;
    markets::irsSwapJacobianRows(child, parent, parent, irs, kReference, kZeroDc, fRow, parentRow,
                                 discountRow);
    const double epsilon = 1e-6;
    const auto& spreadNodes = child.spreadNodes();
    for (std::size_t k = 1; k < spreadNodes.zeros().size(); ++k) {
        std::vector<double> plus = spreadNodes.zeros();
        std::vector<double> minus = spreadNodes.zeros();
        plus[k] += epsilon;
        minus[k] -= epsilon;
        const markets::SpreadCurve<double> cp(parentPtr, spreadNodes.times(), plus,
                                              spreadNodes.scheme(), spreadNodes.tension());
        const markets::SpreadCurve<double> cm(parentPtr, spreadNodes.times(), minus,
                                              spreadNodes.scheme(), spreadNodes.tension());
        const double fd = (markets::impliedIrsRate(cp, parent, irs, kReference, kZeroDc) -
                           markets::impliedIrsRate(cm, parent, irs, kReference, kZeroDc)) /
                          (2.0 * epsilon);
        util::checkClose("seasoned irs own-row vs FD", fRow[k - 1], fd, 1e-6);
    }
    // Total cross row when the forecast parent is also the discount curve:
    // parent row plus discount row.
    for (std::size_t i = 1; i < parent.size(); ++i) {
        const auto bumpedParent = [&](double delta) {
            std::vector<double> zeros = parent.zeros();
            zeros[i] += delta;
            return std::make_shared<DiscountCurve<double>>(parent.times(), zeros, parent.space(),
                                                           parent.scheme(), parent.tension(),
                                                           parent.switchIndex());
        };
        const markets::SpreadCurve<double> cp(bumpedParent(epsilon), spreadNodes.times(),
                                              spreadNodes.zeros(), spreadNodes.scheme(),
                                              spreadNodes.tension());
        const markets::SpreadCurve<double> cm(bumpedParent(-epsilon), spreadNodes.times(),
                                              spreadNodes.zeros(), spreadNodes.scheme(),
                                              spreadNodes.tension());
        const double fd =
            (markets::impliedIrsRate(cp, *bumpedParent(epsilon), irs, kReference, kZeroDc) -
             markets::impliedIrsRate(cm, *bumpedParent(-epsilon), irs, kReference, kZeroDc)) /
            (2.0 * epsilon);
        util::checkClose("seasoned irs cross-row vs FD", parentRow[i - 1] + discountRow[i - 1], fd,
                         1e-6);
    }
    // With discounting split into an equal copy, the forecast parent row sees
    // only the child forwards (the known first coupon contributes nothing)
    // while the discount row carries the annuity and every coupon's payment
    // discount factor.
    const auto discountCopy = std::make_shared<DiscountCurve<double>>(parent);
    std::vector<double> splitParentRow;
    std::vector<double> splitDiscountRow;
    markets::irsSwapJacobianRows(child, parent, *discountCopy, irs, kReference, kZeroDc, fRow,
                                 splitParentRow, splitDiscountRow);
    for (std::size_t i = 1; i < parent.size(); ++i) {
        const auto bumpedParent = [&](double delta) {
            std::vector<double> zeros = parent.zeros();
            zeros[i] += delta;
            return std::make_shared<DiscountCurve<double>>(parent.times(), zeros, parent.space(),
                                                           parent.scheme(), parent.tension(),
                                                           parent.switchIndex());
        };
        const markets::SpreadCurve<double> cp(bumpedParent(epsilon), spreadNodes.times(),
                                              spreadNodes.zeros(), spreadNodes.scheme(),
                                              spreadNodes.tension());
        const markets::SpreadCurve<double> cm(bumpedParent(-epsilon), spreadNodes.times(),
                                              spreadNodes.zeros(), spreadNodes.scheme(),
                                              spreadNodes.tension());
        const double fdParent =
            (markets::impliedIrsRate(cp, *discountCopy, irs, kReference, kZeroDc) -
             markets::impliedIrsRate(cm, *discountCopy, irs, kReference, kZeroDc)) /
            (2.0 * epsilon);
        util::checkClose("seasoned irs parent-row vs FD", splitParentRow[i - 1], fdParent, 1e-6);

        std::vector<double> upZeros = parent.zeros();
        std::vector<double> downZeros = parent.zeros();
        upZeros[i] += epsilon;
        downZeros[i] -= epsilon;
        const DiscountCurve<double> discountUp(parent.times(), upZeros, parent.space(),
                                               parent.scheme(), parent.tension(),
                                               parent.switchIndex());
        const DiscountCurve<double> discountDown(parent.times(), downZeros, parent.space(),
                                                 parent.scheme(), parent.tension(),
                                                 parent.switchIndex());
        const double fdDiscount =
            (markets::impliedIrsRate(child, discountUp, irs, kReference, kZeroDc) -
             markets::impliedIrsRate(child, discountDown, irs, kReference, kZeroDc)) /
            (2.0 * epsilon);
        util::checkClose("seasoned irs discount-row vs FD", splitDiscountRow[i - 1], fdDiscount,
                         1e-6);
    }
}

void testSeasonedConfig() {
    const std::string json = R"({
        "asOf": "2026-09-29",
        "curves": [{
            "key": {"currency": "USD", "role": "Discount", "collateral": "USD"},
            "zeroDayCounter": "ACT/365F",
            "interpolation": {"space": "LogDiscount", "scheme": "Linear"},
            "bootstrap": {"method": "IterativeSequential", "accuracy": 1e-14},
            "pillars": [
                {"start": "2026-03-29", "maturity": "2028-03-29", "kind": "OisSwap",
                 "quote": 0.032, "quoteDayCounter": "ACT/365F", "calendar": "NoHolidays",
                 "fixedTenor": {"length": 1, "unit": "Years"}, "firstFixing": 0.0315}
            ]
        }]
    })";
    const markets::CurveStackSpec stack = markets::parseCurveStackSpec(json);
    const markets::PillarSpec& spec = stack.curves.front().pillars.front();
    CHECK(spec.firstCouponFixed);
    util::checkClose("seasoned config fixing", spec.firstCouponRate, 0.0315, 1e-15);
    const DiscountCurve<double> curve = markets::buildCurve(stack, stack.curves.front());
    CurvePillar out;
    out.kind = PillarKind::OisSwap;
    out.start = spec.start;
    out.maturity = spec.maturity;
    out.fixedTenor = spec.fixedTenor;
    out.quoteDayCounter = spec.quoteDayCounter;
    out.calendar = spec.calendar;
    out.firstCouponFixed = spec.firstCouponFixed;
    out.firstCouponRate = spec.firstCouponRate;
    util::checkClose("seasoned config reprice", markets::impliedQuote(out, stack.asOf, curve),
                     spec.quote, 1e-9);

    const std::string invalid = R"({
        "asOf": "2026-09-29",
        "curves": [{
            "key": {"currency": "USD", "role": "Discount", "collateral": "USD"},
            "zeroDayCounter": "ACT/365F",
            "interpolation": {"space": "LogDiscount", "scheme": "Linear"},
            "bootstrap": {"method": "IterativeSequential", "accuracy": 1e-14},
            "pillars": [
                {"maturity": "2027-03-29", "kind": "Deposit", "quote": 0.043,
                 "quoteDayCounter": "ACT/360", "calendar": "NoHolidays",
                 "firstFixing": 0.04}
            ]
        }]
    })";
    bool threw = false;
    try {
        (void)markets::parseCurveStackSpec(invalid);
    } catch (const std::invalid_argument&) {
        threw = true;
    }
    CHECK(threw);
}

} // namespace

int main() {
    testFraConvexityPricing();
    testFraConvexityRiskRow();
    testFraConvexityModel();
    testFraConvexityConfig();
    testSeasonedOisPricingAndRisk();
    testSeasonedIrsPricingAndRows();
    testSeasonedConfig();
    QTA_LOG_INFO("test", "test_instrument_options: ok");
    return 0;
}
