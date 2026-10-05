/**
 * @file test_futures.cpp
 * @brief Exchange-traded rate futures: pricing, convexity, bootstrap, risk
 *
 * A future is an FRA on the quoted exchange period (fixing to accrual end)
 * whose model quote is the fitted forward rate plus a convexity adjustment:
 * `R_fut = FRA + C`, mirroring `FuturesRateHelper::impliedQuote`. The strip
 * here uses real IMM dates and the Hull-White adjustment.
 */

#include "quantape/datetime/Imm.h"
#include "quantape/log/Log.h"
#include "quantape/markets/Curves/CurveBuilder.h"
#include "quantape/markets/Curves/CurveConfig.h"
#include "quantape/markets/Curves/CurveRisk.h"
#include "quantape/markets/Curves/HullWhiteConvexity.h"
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

struct FutureStrip {
    std::vector<datetime::Date> starts;
    std::vector<datetime::Date> ends;
};

FutureStrip immStrip(std::size_t contracts) {
    FutureStrip strip;
    datetime::Date start = datetime::nextIMMDate(kReference);
    for (std::size_t i = 0; i < contracts; ++i) {
        const datetime::Date end = datetime::nextIMMDate(start);
        strip.starts.push_back(start);
        strip.ends.push_back(end);
        start = end;
    }
    return strip;
}

DiscountCurve<double> makeTarget(const std::vector<datetime::Date>& dates, double base,
                                 double slope) {
    std::vector<double> zeros;
    zeros.reserve(dates.size());
    for (const datetime::Date& date : dates) {
        const double t = datetime::yearFraction(kReference, date, kZeroDc);
        zeros.push_back(base + slope * t);
    }
    return DiscountCurve<double>(kReference, dates, kZeroDc, zeros, InterpolationSpace::LogDiscount,
                                 InterpolationScheme::Linear);
}

std::vector<CurvePillar> futurePillars(const FutureStrip& strip, double convexity) {
    std::vector<CurvePillar> pillars;
    CurvePillar deposit;
    deposit.maturity = datetime::Period(3, datetime::TimeUnit::Months).advance(kReference);
    deposit.kind = PillarKind::Deposit;
    deposit.quoteDayCounter = kIndexDc;
    pillars.push_back(deposit);
    for (std::size_t i = 0; i < strip.starts.size(); ++i) {
        CurvePillar future;
        future.kind = PillarKind::Future;
        future.start = strip.starts[i];
        future.maturity = strip.ends[i];
        future.quoteDayCounter = kIndexDc;
        future.convexityAdjustment = convexity;
        pillars.push_back(future);
    }
    return pillars;
}

void testHullWhiteAdjustment() {
    // Zero volatility and zero mean reversion handle their limits.
    util::checkClose("hw zero vol", markets::hullWhiteFuturesAdjustment(0.0, 0.05, 1.0, 0.25, 1.0),
                     0.0, 1e-18);
    const double series = markets::hullWhiteFuturesAdjustment(0.01, 0.0, 1.0, 0.25, 1.0);
    const double nearlySeries = markets::hullWhiteFuturesAdjustment(0.01, 1e-9, 1.0, 0.25, 1.0);
    util::checkClose("hw series continuity", nearlySeries, series, 1e-12);
    const double seriesExponent = 0.01 * 0.01 * 0.25 * (0.5 * 1.0 * 1.0 + 1.0 * 0.25);
    // (P1/P2) (exp(D) - 1) / accrual, evaluated with the same expm1 the helper uses
    const double expected = std::expm1(seriesExponent) / 0.25;
    util::checkClose("hw series value", series, expected, 1e-15);
    util::checkClose("hw branch continuity",
                     markets::hullWhiteFuturesAdjustment(0.01, 1.1e-8, 1.0, 0.25, 1.0), series,
                     1e-6 * series);

    // Positive convexity, monotone in volatility, quadratic for small sigma.
    const double small = markets::hullWhiteFuturesAdjustment(0.001, 0.0, 1.0, 0.25, 1.0);
    const double doubleSmall = markets::hullWhiteFuturesAdjustment(0.002, 0.0, 1.0, 0.25, 1.0);
    CHECK(small > 0.0);
    const double smallExponent = 0.001 * 0.001 * 0.25 * (0.5 + 0.25);
    util::checkClose("hw sigma squared scaling", doubleSmall / small,
                     std::expm1(4.0 * smallExponent) / std::expm1(smallExponent), 1e-12);
    const double stronger = markets::hullWhiteFuturesAdjustment(0.02, 0.05, 1.0, 0.25, 1.0);
    CHECK(stronger > markets::hullWhiteFuturesAdjustment(0.01, 0.05, 1.0, 0.25, 1.0));
}

void testImmFutureBootstrap() {
    const FutureStrip strip = immStrip(8);
    for (const datetime::Date& start : strip.starts) {
        CHECK(datetime::isIMMDate(start));
        CHECK(start.weekday() == datetime::Weekday::Wednesday);
    }
    std::vector<datetime::Date> targetDates{
        datetime::Period(3, datetime::TimeUnit::Months).advance(kReference)};
    targetDates.insert(targetDates.end(), strip.ends.begin(), strip.ends.end());
    const DiscountCurve<double> target = makeTarget(targetDates, 0.035, 0.001);

    const double convexity = markets::hullWhiteFuturesAdjustment(0.01, 0.05, 1.0, 0.25, 1.0);
    std::vector<CurvePillar> pillars = futurePillars(strip, convexity);
    for (CurvePillar& pillar : pillars) {
        pillar.calendar = datetime::Calendar::noHolidays();
        pillar.quote = markets::impliedQuote(pillar, kReference, target);
    }
    const DiscountCurve<double> curve = markets::bootstrapDiscountCurve(
        kReference, kZeroDc, InterpolationSpace::LogDiscount, InterpolationScheme::Linear, pillars);
    for (const CurvePillar& pillar : pillars) {
        util::checkClose("future reprice", markets::impliedQuote(pillar, kReference, curve),
                         pillar.quote, 1e-9);
    }
    // IMM dates are curve nodes, so the target is recovered exactly.
    for (std::size_t i = 0; i < strip.ends.size(); ++i) {
        const double t = datetime::yearFraction(kReference, strip.ends[i], kZeroDc);
        util::checkClose("future node recovery", curve.zero(t), target.zero(t), 1e-10);
    }
    // The fitted forward over each period equals the futures quote minus C.
    for (std::size_t i = 0; i < strip.starts.size(); ++i) {
        const double t1 = datetime::yearFraction(kReference, strip.starts[i], kZeroDc);
        const double t2 = datetime::yearFraction(kReference, strip.ends[i], kZeroDc);
        const double tau = datetime::yearFraction(strip.starts[i], strip.ends[i], kIndexDc);
        const double forward = (curve.discount(t1) / curve.discount(t2) - 1.0) / tau;
        util::checkClose("future implies forward", forward, pillars[i + 1].quote - convexity,
                         1e-10);
    }
}

void testConvexityMovesForward() {
    const FutureStrip strip = immStrip(4);
    std::vector<datetime::Date> targetDates{
        datetime::Period(3, datetime::TimeUnit::Months).advance(kReference)};
    targetDates.insert(targetDates.end(), strip.ends.begin(), strip.ends.end());
    const DiscountCurve<double> target = makeTarget(targetDates, 0.03, 0.0);

    std::vector<CurvePillar> noConvexity = futurePillars(strip, 0.0);
    std::vector<CurvePillar> withConvexity = futurePillars(strip, 0.0005);
    for (CurvePillar& pillar : noConvexity) {
        pillar.calendar = datetime::Calendar::noHolidays();
        pillar.quote = markets::impliedQuote(pillar, kReference, target);
    }
    for (CurvePillar& pillar : withConvexity) {
        pillar.calendar = datetime::Calendar::noHolidays();
        pillar.quote = markets::impliedQuote(pillar, kReference, target);
    }
    const DiscountCurve<double> curve =
        markets::bootstrapDiscountCurve(kReference, kZeroDc, InterpolationSpace::LogDiscount,
                                        InterpolationScheme::Linear, withConvexity);
    for (std::size_t i = 0; i < strip.starts.size(); ++i) {
        const datetime::Date& start = strip.starts[i];
        const datetime::Date& end = strip.ends[i];
        const double t1 = datetime::yearFraction(kReference, start, kZeroDc);
        const double t2 = datetime::yearFraction(kReference, end, kZeroDc);
        const double tau = datetime::yearFraction(start, end, kIndexDc);
        // The curve forward is the futures-implied rate minus the adjustment.
        util::checkClose("convexity lowers forward",
                         (curve.discount(t1) / curve.discount(t2) - 1.0) / tau,
                         withConvexity[i + 1].quote - 0.0005, 1e-10);
    }
}

void testFutureRiskRowFiniteDifference() {
    const FutureStrip strip = immStrip(4);
    std::vector<datetime::Date> targetDates{
        datetime::Period(3, datetime::TimeUnit::Months).advance(kReference)};
    targetDates.insert(targetDates.end(), strip.ends.begin(), strip.ends.end());
    const DiscountCurve<double> target = makeTarget(targetDates, 0.032, 0.0005);
    std::vector<CurvePillar> pillars = futurePillars(strip, 0.0002);
    for (CurvePillar& pillar : pillars) {
        pillar.calendar = datetime::Calendar::noHolidays();
        pillar.quote = markets::impliedQuote(pillar, kReference, target);
    }
    const DiscountCurve<double> curve = markets::bootstrapDiscountCurve(
        kReference, kZeroDc, InterpolationSpace::LogDiscount, InterpolationScheme::Linear, pillars);
    const CurvePillar& future = pillars[2];
    std::vector<double> row;
    CHECK(markets::pillarJacobianRow(future, kReference, curve, row));
    for (std::size_t i = 1; i < curve.size(); ++i) {
        std::vector<double> zeros = curve.zeros();
        zeros[i] += 1e-8;
        const DiscountCurve<double> bumped(curve.times(), zeros, InterpolationSpace::LogDiscount,
                                           InterpolationScheme::Linear);
        const double fd = (markets::impliedQuote(future, kReference, bumped) -
                           markets::impliedQuote(future, kReference, curve)) /
                          1e-8;
        util::checkClose("future jacobian vs FD", row[i - 1], fd, 1e-5);
    }
}

void testFutureConfigRoundTrip() {
    const std::string json = R"({
        "asOf": "2026-09-29",
        "curves": [{
            "key": {"currency": "USD", "role": "Discount", "collateral": "USD"},
            "zeroDayCounter": "ACT/365F",
            "interpolation": {"space": "LogDiscount", "scheme": "Linear"},
            "bootstrap": {"method": "IterativeSequential", "accuracy": 1e-14},
            "pillars": [
                {"maturity": "2027-03-29", "kind": "Deposit", "quote": 0.043,
                 "quoteDayCounter": "ACT/360", "calendar": "NoHolidays"},
                {"start": "2027-06-16", "maturity": "2027-09-15", "kind": "Future",
                 "quote": 0.041, "convexityAdjustment": 0.0002,
                 "quoteDayCounter": "ACT/360", "calendar": "NoHolidays"}
            ]
        }]
    })";
    const markets::CurveStackSpec stack = markets::parseCurveStackSpec(json);
    CHECK(stack.curves.size() == 1);
    const markets::CurveSpec& spec = stack.curves.front();
    CHECK(spec.pillars.size() == 2);
    CHECK(spec.pillars[1].kind == markets::PillarSpec::Kind::Future);
    util::checkClose("config convexity", spec.pillars[1].convexityAdjustment, 0.0002, 1e-15);
    std::vector<CurvePillar> filled;
    const DiscountCurve<double> curve = markets::buildCurve(stack, spec, {}, &filled);
    CHECK(filled.size() == spec.pillars.size());
    for (const CurvePillar& pillar : filled) {
        util::checkClose("future config reprice", markets::impliedQuote(pillar, stack.asOf, curve),
                         pillar.quote, 1e-9);
    }
}

void testCurveLevelConvexity() {
    const std::string json = R"({
        "asOf": "2026-09-29",
        "curves": [{
            "key": {"currency": "USD", "role": "Discount", "collateral": "USD"},
            "zeroDayCounter": "ACT/365F",
            "interpolation": {"space": "LogDiscount", "scheme": "Linear"},
            "bootstrap": {"method": "IterativeSequential", "accuracy": 1e-14},
            "convexity": {"model": "HullWhite", "sigma": 0.01, "meanReversion": 0.05},
            "pillars": [
                {"maturity": "2027-03-29", "kind": "Deposit", "quote": 0.043,
                 "quoteDayCounter": "ACT/360", "calendar": "NoHolidays"},
                {"start": "2027-06-16", "maturity": "2027-09-15", "kind": "Future",
                 "quote": 0.041, "quoteDayCounter": "ACT/360", "calendar": "NoHolidays"}
            ]
        }]
    })";
    const markets::CurveStackSpec stack = markets::parseCurveStackSpec(json);
    CHECK(stack.curves.size() == 1);
    const markets::CurveSpec& spec = stack.curves.front();
    CHECK(spec.convexity.enabled);
    CHECK(!spec.convexity.hasReference);
    const DiscountCurve<double> curve = markets::buildCurve(stack, spec);

    const markets::PillarSpec& futureSpec = spec.pillars.back();
    const double t1 = datetime::yearFraction(stack.asOf, futureSpec.start, spec.zeroDayCounter);
    const double t2 = datetime::yearFraction(stack.asOf, futureSpec.maturity, spec.zeroDayCounter);
    const double accrual =
        datetime::yearFraction(futureSpec.start, futureSpec.maturity, futureSpec.quoteDayCounter);
    const double ratio = curve.discount(t1) / curve.discount(t2);
    const double adjustment = markets::hullWhiteFuturesAdjustment(0.01, 0.05, t1, accrual, ratio);
    const double forward = (ratio - 1.0) / accrual;
    util::checkClose("self-referential convexity", forward + adjustment, futureSpec.quote, 1e-9);

    // A Future pillar without either source of convexity is a config error.
    const std::string missing = R"({
        "asOf": "2026-09-29",
        "curves": [{
            "key": {"currency": "USD", "role": "Discount", "collateral": "USD"},
            "zeroDayCounter": "ACT/365F",
            "interpolation": {"space": "LogDiscount", "scheme": "Linear"},
            "bootstrap": {"method": "IterativeSequential", "accuracy": 1e-14},
            "pillars": [
                {"start": "2027-06-16", "maturity": "2027-09-15", "kind": "Future",
                 "quote": 0.041, "quoteDayCounter": "ACT/360", "calendar": "NoHolidays"}
            ]
        }]
    })";
    const markets::CurveStackSpec missingStack = markets::parseCurveStackSpec(missing);
    bool threw = false;
    try {
        (void)markets::buildCurve(missingStack, missingStack.curves.front());
    } catch (const std::invalid_argument&) {
        threw = true;
    }
    CHECK(threw);
}

void testReferenceCurveConvexity() {
    const std::string json = R"({
        "asOf": "2026-09-29",
        "curves": [
            {
                "key": {"currency": "USD", "role": "Discount", "collateral": "USD"},
                "zeroDayCounter": "ACT/365F",
                "interpolation": {"space": "LogDiscount", "scheme": "Linear"},
                "bootstrap": {"method": "IterativeSequential", "accuracy": 1e-14},
                "pillars": [
                    {"maturity": "2027-03-29", "kind": "Deposit", "quote": 0.043,
                     "quoteDayCounter": "ACT/360", "calendar": "NoHolidays"},
                    {"maturity": "2028-03-29", "kind": "OisSwap", "quote": 0.041,
                     "quoteDayCounter": "ACT/365F", "calendar": "NoHolidays"}
                ]
            },
            {
                "key": {"currency": "EUR", "role": "Discount", "indexTenor": {"length": 3, "unit": "Months"},
                        "collateral": "EUR"},
                "zeroDayCounter": "ACT/365F",
                "interpolation": {"space": "LogDiscount", "scheme": "Linear"},
                "bootstrap": {"method": "IterativeSequential", "accuracy": 1e-14},
                "convexity": {"model": "HullWhite", "sigma": 0.01, "meanReversion": 0.05,
                              "referenceCurve": {"currency": "USD", "role": "Discount",
                                                 "collateral": "USD"}},
                "pillars": [
                    {"maturity": "2027-03-29", "kind": "Deposit", "quote": 0.021,
                     "quoteDayCounter": "ACT/360", "calendar": "NoHolidays"},
                    {"start": "2027-06-16", "maturity": "2027-09-15", "kind": "Future",
                     "quote": 0.022, "quoteDayCounter": "ACT/360", "calendar": "NoHolidays"}
                ]
            }
        ]
    })";
    const markets::CurveStackSpec stack = markets::parseCurveStackSpec(json);
    CHECK(stack.curves.size() == 2);
    const markets::CurveSpec& discountSpec = stack.curves[0];
    const markets::CurveSpec& forecastSpec = stack.curves[1];
    CHECK(forecastSpec.convexity.hasReference);
    const DiscountCurve<double> discount = markets::buildCurve(stack, discountSpec);
    const DiscountCurve<double> forecast =
        markets::buildCurve(stack, forecastSpec, {{discountSpec.key, &discount}});

    const markets::PillarSpec& futureSpec = forecastSpec.pillars.back();
    const double t1 =
        datetime::yearFraction(stack.asOf, futureSpec.start, forecastSpec.zeroDayCounter);
    const double t2 =
        datetime::yearFraction(stack.asOf, futureSpec.maturity, forecastSpec.zeroDayCounter);
    const double accrual =
        datetime::yearFraction(futureSpec.start, futureSpec.maturity, futureSpec.quoteDayCounter);
    const double forward = (forecast.discount(t1) / forecast.discount(t2) - 1.0) / accrual;
    const double referenceAdjustment = markets::hullWhiteFuturesAdjustment(
        0.01, 0.05, t1, accrual, discount.discount(t1) / discount.discount(t2));
    util::checkClose("reference convexity", forward + referenceAdjustment, futureSpec.quote, 1e-9);
    // The EUR self-referential adjustment differs: the reference curve was used.
    const double selfAdjustment = markets::hullWhiteFuturesAdjustment(
        0.01, 0.05, t1, accrual, forecast.discount(t1) / forecast.discount(t2));
    CHECK(std::abs(selfAdjustment - referenceAdjustment) > 1e-10);
    CHECK(std::abs(forward + selfAdjustment - futureSpec.quote) > 1e-10);

    bool threw = false;
    try {
        (void)markets::buildCurve(stack, forecastSpec, {});
    } catch (const std::invalid_argument&) {
        threw = true;
    }
    CHECK(threw);
}

void testHullWhiteParameterFit() {
    const double trueSigma = 0.011;
    const double trueMeanReversion = 0.045;
    const double tau = 0.25;
    const double ratio = std::exp(0.04 * tau);
    std::vector<markets::HullWhiteObservation> observations;
    for (int i = 0; i < 9; ++i) {
        markets::HullWhiteObservation observation;
        observation.expiryTime = 0.5 + 0.25 * static_cast<double>(i);
        observation.accrual = tau;
        observation.startOverEndDiscount = ratio;
        observation.impliedAdjustment = markets::hullWhiteFuturesAdjustment(
            trueSigma, trueMeanReversion, observation.expiryTime, tau, ratio);
        observations.push_back(observation);
    }
    markets::HullWhiteFitOptions sigmaOptions;
    sigmaOptions.pinnedMeanReversion = trueMeanReversion;
    sigmaOptions.initialValue = 0.008;
    const markets::HullWhiteFitResult sigmaFit =
        markets::fitHullWhiteSigma(observations, sigmaOptions);
    util::checkClose("hw fit sigma", sigmaFit.sigma, trueSigma, 1e-9);
    CHECK(sigmaFit.rmsResidual < 1e-10);

    markets::HullWhiteFitOptions meanReversionOptions;
    meanReversionOptions.pinnedSigma = trueSigma;
    meanReversionOptions.initialValue = 0.03;
    const markets::HullWhiteFitResult meanReversionFit =
        markets::fitHullWhiteMeanReversion(observations, meanReversionOptions);
    util::checkClose("hw fit mean reversion", meanReversionFit.meanReversion, trueMeanReversion,
                     1e-7);
    CHECK(meanReversionFit.rmsResidual < 1e-10);

    bool threw = false;
    try {
        (void)markets::fitHullWhiteSigma({});
    } catch (const std::invalid_argument&) {
        threw = true;
    }
    CHECK(threw);
}

void testRfrFutureStyles() {
    const FutureStrip strip = immStrip(3);
    std::vector<datetime::Date> targetDates{
        datetime::Period(3, datetime::TimeUnit::Months).advance(kReference)};
    targetDates.insert(targetDates.end(), strip.ends.begin(), strip.ends.end());
    const DiscountCurve<double> target = makeTarget(targetDates, 0.03, 0.0005);

    std::vector<CurvePillar> pillars = futurePillars(strip, 0.0002);
    pillars[1].futureStyle = markets::FutureStyle::Compounded;
    pillars[2].futureStyle = markets::FutureStyle::Simple;
    for (CurvePillar& pillar : pillars) {
        pillar.calendar = datetime::Calendar::noHolidays();
        pillar.quote = markets::impliedQuote(pillar, kReference, target);
    }
    CurvePillar compounded = pillars[1];
    CurvePillar simple = pillars[1];
    simple.futureStyle = markets::FutureStyle::Simple;
    util::checkClose("compounded telescopes to simple",
                     markets::impliedQuote(compounded, kReference, target),
                     markets::impliedQuote(simple, kReference, target), 1e-15);
    const DiscountCurve<double> curve = markets::bootstrapDiscountCurve(
        kReference, kZeroDc, InterpolationSpace::LogDiscount, InterpolationScheme::Linear, pillars);
    for (const CurvePillar& pillar : pillars) {
        util::checkClose("rfr future reprice", markets::impliedQuote(pillar, kReference, curve),
                         pillar.quote, 1e-9);
    }
}

void testAveragedRfrFuture() {
    const FutureStrip strip = immStrip(3);
    const datetime::Date periodEnd =
        datetime::Period(1, datetime::TimeUnit::Months).advance(strip.starts.front());
    std::vector<datetime::Date> targetDates{
        datetime::Period(3, datetime::TimeUnit::Months).advance(kReference), periodEnd};
    targetDates.insert(targetDates.end(), strip.ends.begin(), strip.ends.end());
    const DiscountCurve<double> target = makeTarget(targetDates, 0.03, 0.0005);

    CurvePillar averaged;
    averaged.kind = PillarKind::Future;
    averaged.futureStyle = markets::FutureStyle::Averaged;
    averaged.start = strip.starts.front();
    averaged.maturity = periodEnd;
    averaged.quoteDayCounter = kIndexDc;
    averaged.calendar = datetime::Calendar::noHolidays();
    averaged.quote = markets::impliedQuote(averaged, kReference, target);

    const double t1 = datetime::yearFraction(kReference, averaged.start, kZeroDc);
    const double t2 = datetime::yearFraction(kReference, averaged.maturity, kZeroDc);
    const double tau = datetime::yearFraction(averaged.start, averaged.maturity, kIndexDc);
    const double simpleForward = (target.discount(t1) / target.discount(t2) - 1.0) / tau;
    CHECK(std::abs(averaged.quote - simpleForward) > 1e-7);

    std::vector<CurvePillar> pillars = futurePillars(strip, 0.0002);
    pillars[1] = averaged;
    for (CurvePillar& pillar : pillars) {
        pillar.calendar = datetime::Calendar::noHolidays();
        pillar.quote = markets::impliedQuote(pillar, kReference, target);
    }
    const DiscountCurve<double> curve = markets::bootstrapDiscountCurve(
        kReference, kZeroDc, InterpolationSpace::LogDiscount, InterpolationScheme::Linear, pillars);
    for (const CurvePillar& pillar : pillars) {
        util::checkClose("averaged future reprice",
                         markets::impliedQuote(pillar, kReference, curve), pillar.quote, 1e-9);
    }

    std::vector<double> row;
    CHECK(markets::pillarJacobianRow(averaged, kReference, curve, row));
    for (std::size_t i = 1; i < curve.size(); ++i) {
        std::vector<double> zeros = curve.zeros();
        zeros[i] += 1e-8;
        const DiscountCurve<double> bumped(curve.times(), zeros, InterpolationSpace::LogDiscount,
                                           InterpolationScheme::Linear);
        const double fd = (markets::impliedQuote(averaged, kReference, bumped) -
                           markets::impliedQuote(averaged, kReference, curve)) /
                          1e-8;
        util::checkClose("averaged jacobian vs FD", row[i - 1], fd, 1e-5);
    }
}

void testAveragedCompoundedFuture() {
    const FutureStrip strip = immStrip(3);
    const datetime::Date periodEnd =
        datetime::Period(1, datetime::TimeUnit::Months).advance(strip.starts.front());
    std::vector<datetime::Date> targetDates{
        datetime::Period(3, datetime::TimeUnit::Months).advance(kReference), periodEnd};
    targetDates.insert(targetDates.end(), strip.ends.begin(), strip.ends.end());
    const double flatZero = 0.03;
    const DiscountCurve<double> target = makeTarget(targetDates, flatZero, 0.0);

    const double convexity = 0.0004;
    CurvePillar averaged;
    averaged.kind = PillarKind::Future;
    averaged.futureStyle = markets::FutureStyle::Averaged;
    averaged.averagingStyle = markets::AveragingStyle::Compounded;
    averaged.start = strip.starts.front();
    averaged.maturity = periodEnd;
    averaged.quoteDayCounter = kIndexDc;
    averaged.calendar = datetime::Calendar::noHolidays();
    averaged.convexityAdjustment = convexity;
    averaged.quote = markets::impliedQuote(averaged, kReference, target);

    const double t1 = datetime::yearFraction(kReference, averaged.start, kZeroDc);
    const double t2 = datetime::yearFraction(kReference, averaged.maturity, kZeroDc);
    const double tau = datetime::yearFraction(averaged.start, averaged.maturity, kIndexDc);
    // Flat zero curve under LogDiscount/Linear: D(t) = exp(-z t), so the
    // compounded average reprices to expm1(z (t2 - t1)) / tau plus convexity.
    util::checkClose("compounded averaged hand check", averaged.quote,
                     std::expm1(flatZero * (t2 - t1)) / tau + convexity, 1e-14);

    // The explicit fixing-grid product is the telescoped period ratio, and the
    // quoted rate is its annualization over the reference period.
    const std::vector<datetime::Date> fixings =
        markets::businessDayFixings(averaged.calendar, averaged.start, averaged.maturity);
    CHECK(fixings.back() == averaged.maturity);
    double gridProduct = 1.0;
    for (std::size_t k = 1; k < fixings.size(); ++k) {
        const double previous = datetime::yearFraction(kReference, fixings[k - 1], kZeroDc);
        const double current = datetime::yearFraction(kReference, fixings[k], kZeroDc);
        gridProduct *= target.discount(previous) / target.discount(current);
    }
    util::checkClose("compounded averaged grid ratio", gridProduct,
                     target.compoundingFactor(t1, t2), 1e-14);
    util::checkClose("compounded averaged quote from grid", averaged.quote - convexity,
                     (gridProduct - 1.0) / tau, 1e-14);

    std::vector<CurvePillar> pillars = futurePillars(strip, 0.0002);
    pillars[1] = averaged;
    for (CurvePillar& pillar : pillars) {
        pillar.calendar = datetime::Calendar::noHolidays();
        pillar.quote = markets::impliedQuote(pillar, kReference, target);
    }
    const DiscountCurve<double> curve = markets::bootstrapDiscountCurve(
        kReference, kZeroDc, InterpolationSpace::LogDiscount, InterpolationScheme::Linear, pillars);
    for (const CurvePillar& pillar : pillars) {
        util::checkClose("compounded averaged reprice",
                         markets::impliedQuote(pillar, kReference, curve), pillar.quote, 1e-9);
    }

    std::vector<double> row;
    CHECK(markets::pillarJacobianRow(averaged, kReference, curve, row));
    const double step = 1e-7;
    for (std::size_t i = 1; i < curve.size(); ++i) {
        std::vector<double> plus = curve.zeros();
        std::vector<double> minus = curve.zeros();
        plus[i] += step;
        minus[i] -= step;
        const DiscountCurve<double> bumpedPlus(curve.times(), plus, InterpolationSpace::LogDiscount,
                                               InterpolationScheme::Linear);
        const DiscountCurve<double> bumpedMinus(
            curve.times(), minus, InterpolationSpace::LogDiscount, InterpolationScheme::Linear);
        const double fd = (markets::impliedQuote(averaged, kReference, bumpedPlus) -
                           markets::impliedQuote(averaged, kReference, bumpedMinus)) /
                          (2.0 * step);
        util::checkClose("compounded averaged jacobian vs FD", row[i - 1], fd, 1e-6);
    }
}

void testFutureStyleConfig() {
    const std::string json = R"({
        "asOf": "2026-09-29",
        "curves": [{
            "key": {"currency": "USD", "role": "Discount", "collateral": "USD"},
            "zeroDayCounter": "ACT/365F",
            "interpolation": {"space": "LogDiscount", "scheme": "Linear"},
            "bootstrap": {"method": "IterativeSequential", "accuracy": 1e-14},
            "pillars": [
                {"start": "2027-06-16", "maturity": "2027-09-15", "kind": "Future",
                 "style": "Averaged", "quote": 0.041, "convexityAdjustment": 0.0,
                 "quoteDayCounter": "ACT/360", "calendar": "NoHolidays"}
            ]
        }]
    })";
    const markets::CurveStackSpec stack = markets::parseCurveStackSpec(json);
    CHECK(stack.curves.front().pillars.front().futureStyle == markets::FutureStyle::Averaged);
    CHECK(stack.curves.front().pillars.front().averagingStyle ==
          markets::AveragingStyle::Arithmetic);

    const std::string invalid = R"({
        "asOf": "2026-09-29",
        "curves": [{
            "key": {"currency": "USD", "role": "Discount", "collateral": "USD"},
            "zeroDayCounter": "ACT/365F",
            "interpolation": {"space": "LogDiscount", "scheme": "Linear"},
            "bootstrap": {"method": "IterativeSequential", "accuracy": 1e-14},
            "pillars": [
                {"maturity": "2027-03-29", "kind": "Deposit", "style": "Averaged",
                 "quote": 0.043, "quoteDayCounter": "ACT/360", "calendar": "NoHolidays"}
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

void testAveragedWeekendFixings() {
    const datetime::Calendar weekends = datetime::Calendar::weekendsOnly();
    const datetime::Date start = kReference.plusMonths(1);
    const datetime::Date maturity = datetime::Period(1, datetime::TimeUnit::Months).advance(start);
    std::vector<datetime::Date> targetDates{start, maturity};
    const DiscountCurve<double> target = makeTarget(targetDates, 0.03, 0.0005);

    CurvePillar averaged;
    averaged.kind = PillarKind::Future;
    averaged.futureStyle = markets::FutureStyle::Averaged;
    averaged.start = start;
    averaged.maturity = maturity;
    averaged.quoteDayCounter = kIndexDc;
    averaged.calendar = weekends;
    averaged.quote = markets::impliedQuote(averaged, kReference, target);

    std::vector<CurvePillar> pillars;
    CurvePillar deposit;
    deposit.maturity = start;
    deposit.kind = PillarKind::Deposit;
    deposit.quoteDayCounter = kIndexDc;
    deposit.calendar = weekends;
    deposit.quote = markets::impliedQuote(deposit, kReference, target);
    pillars.push_back(deposit);
    pillars.push_back(averaged);
    const DiscountCurve<double> curve = markets::bootstrapDiscountCurve(
        kReference, kZeroDc, InterpolationSpace::LogDiscount, InterpolationScheme::Linear, pillars);
    util::checkClose("averaged weekend reprice", markets::impliedQuote(averaged, kReference, curve),
                     averaged.quote, 1e-9);
    const auto fixings = markets::businessDayFixings(weekends, start, maturity);
    CHECK(fixings.size() >= 20);
    CHECK(fixings.back() == maturity);
    for (std::size_t k = 1; k < fixings.size(); ++k) {
        CHECK(fixings[k] > fixings[k - 1]);
        // Only the final endpoint may be a non-business day: it closes the
        // quoted reference period.
        if (k + 1 < fixings.size()) {
            CHECK(weekends.isBusinessDay(fixings[k]));
        }
    }
}

} // namespace

int main() {
    testHullWhiteAdjustment();
    testImmFutureBootstrap();
    testConvexityMovesForward();
    testFutureRiskRowFiniteDifference();
    testFutureConfigRoundTrip();
    testCurveLevelConvexity();
    testReferenceCurveConvexity();
    testHullWhiteParameterFit();
    testRfrFutureStyles();
    testAveragedRfrFuture();
    testAveragedCompoundedFuture();
    testFutureStyleConfig();
    testAveragedWeekendFixings();
    QTA_LOG_INFO("test", "test_futures: ok");
    return 0;
}
