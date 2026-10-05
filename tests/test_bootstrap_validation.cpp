/**
 * @file test_bootstrap_validation.cpp
 * @brief Cross-scheme bootstrap validation
 *
 * Invariants checked across every interpolation space/scheme:
 *  - exact fit: every pillar reprices its quote on the returned curve,
 *  - node recovery when all instrument cashflows land on curve nodes,
 *  - off-node cashflows (semi-annual fixed legs, quarterly floats) and
 *    payment lags reach the whole-grid fixed point,
 *  - curves are finite, positive and arbitrage-free for positive inputs,
 *  - stubs, negative rates and basis curves behave consistently,
 *  - malformed inputs are rejected with clear errors.
 */

#include "quantape/markets/Curves/CurveBuilder.h"
#include "quantape/markets/Curves/SpreadCurve.h"

#include "quantape/log/Log.h"
#include "quantape/util/Check.h"

#include <cmath>
#include <cstddef>
#include <memory>
#include <string>
#include <vector>

using namespace quantape;

namespace {

using markets::BasisPillar;
using markets::CurvePillar;
using markets::DiscountCurve;
using markets::InterpolationScheme;
using markets::InterpolationSpace;
using markets::PillarKind;

const datetime::Date kReference(2026, 9, 29);
const datetime::DayCounter kZeroDc(datetime::DayCount::Actual365Fixed);

struct SchemeCase {
    InterpolationSpace space;
    InterpolationScheme scheme;
    double tension;
};

const std::vector<SchemeCase>& schemeMatrix() {
    static const std::vector<SchemeCase> matrix{
        {InterpolationSpace::LogDiscount, InterpolationScheme::Linear, 0.0},
        {InterpolationSpace::Zero, InterpolationScheme::Linear, 0.0},
        {InterpolationSpace::LogDiscount, InterpolationScheme::Akima, 0.0},
        {InterpolationSpace::LogDiscount, InterpolationScheme::TensionSpline, 8.0},
        {InterpolationSpace::LogDiscount, InterpolationScheme::MonotoneCubic, 0.0},
        {InterpolationSpace::LogDiscount, InterpolationScheme::MixedLinearCubic, 0.0},
    };
    return matrix;
}

DiscountCurve<double> makeTarget(const SchemeCase& item,
                                 const std::vector<datetime::Date>& dates, double base,
                                 double slope, double curvature) {
    std::vector<double> zeros;
    zeros.reserve(dates.size());
    for (const datetime::Date& date : dates) {
        const double t = datetime::yearFraction(kReference, date, kZeroDc);
        zeros.push_back(base + slope * t + curvature * t * t);
    }
    return DiscountCurve<double>(kReference, dates, kZeroDc, zeros, item.space, item.scheme,
                                 item.tension);
}

void checkExactFit(const std::string& label, const DiscountCurve<double>& curve,
                   const std::vector<CurvePillar>& pillars) {
    for (std::size_t i = 0; i < pillars.size(); ++i) {
        util::checkClose((label + " reprice").c_str(),
                         markets::impliedQuote(pillars[i], kReference, curve), pillars[i].quote,
                         1e-9);
    }
}

void checkCurveSanity(const DiscountCurve<double>& curve, bool expectDecreasing) {
    CHECK(curve.discount(0.0) == 1.0);
    double previous = curve.discount(0.0);
    for (double t = 0.25; t <= 10.0; t += 0.25) {
        const double df = curve.discount(t);
        CHECK(std::isfinite(df));
        CHECK(df > 0.0);
        if (expectDecreasing) {
            CHECK(df < previous);
        }
        previous = df;
    }
}

std::vector<CurvePillar> annualOisPillars(const DiscountCurve<double>& target,
                                          const std::vector<datetime::Date>& dates,
                                          const datetime::Calendar& calendar,
                                          const datetime::Period& fixedTenor, int paymentLag) {
    std::vector<CurvePillar> pillars;
    CurvePillar deposit;
    deposit.maturity = dates.front();
    deposit.kind = PillarKind::Deposit;
    deposit.quoteDayCounter = datetime::DayCounter(datetime::DayCount::Actual360);
    deposit.calendar = calendar;
    deposit.quote = markets::impliedQuote(deposit, kReference, target);
    pillars.push_back(deposit);
    for (std::size_t i = 1; i < dates.size(); ++i) {
        CurvePillar swap;
        swap.maturity = dates[i];
        swap.kind = PillarKind::OisSwap;
        swap.fixedTenor = fixedTenor;
        swap.paymentLag = paymentLag;
        swap.quoteDayCounter = kZeroDc;
        swap.calendar = calendar;
        swap.quote = markets::impliedQuote(swap, kReference, target);
        pillars.push_back(swap);
    }
    return pillars;
}

void testNodeAlignedRecovery() {
    const datetime::Calendar calendar = datetime::Calendar::noHolidays();
    std::vector<datetime::Date> dates{
        datetime::Period(6, datetime::TimeUnit::Months).advance(kReference)};
    for (int year = 1; year <= 10; ++year) {
        dates.push_back(kReference.plusYears(year));
    }
    for (const SchemeCase& item : schemeMatrix()) {
        const DiscountCurve<double> target =
            makeTarget(item, dates, 0.025, 0.0015, -0.00008);
        const std::vector<CurvePillar> pillars = annualOisPillars(
            target, dates, calendar, datetime::Period(1, datetime::TimeUnit::Years), 0);
        const DiscountCurve<double> curve = markets::bootstrapDiscountCurve(
            kReference, kZeroDc, item.space, item.scheme, pillars, 1e-14, item.tension);
        checkExactFit("node-aligned", curve, pillars);
        checkCurveSanity(curve, true);
        for (const datetime::Date& date : dates) {
            const double t = datetime::yearFraction(kReference, date, kZeroDc);
            util::checkClose("node recovery", curve.zero(t), target.zero(t), 1e-10);
        }
    }
}

void testOffNodeCashflowsAndLags() {
    const datetime::Calendar calendar = datetime::Calendar::noHolidays();
    std::vector<datetime::Date> dates;
    for (int year = 1; year <= 6; ++year) {
        dates.push_back(kReference.plusYears(year));
    }
    for (const SchemeCase& item : schemeMatrix()) {
        const DiscountCurve<double> target = makeTarget(item, dates, 0.03, -0.001, 0.0001);
        // Semi-annual fixed coupons on annual nodes evaluate inside interpolation
        // segments; Akima/mixed/tension stencils then need nodes beyond the own
        // pillar, which is exactly the whole-grid fixed point.
        const std::vector<CurvePillar> pillars =
            annualOisPillars(target, dates, calendar,
                             datetime::Period(6, datetime::TimeUnit::Months), 2);
        const DiscountCurve<double> curve = markets::bootstrapDiscountCurve(
            kReference, kZeroDc, item.space, item.scheme, pillars, 1e-14, item.tension);
        checkExactFit("off-node/lag", curve, pillars);
        checkCurveSanity(curve, true);
    }
}

void testStubsAndNegativeRates() {
    const datetime::Calendar calendar = datetime::Calendar::noHolidays();
    std::vector<datetime::Date> dates{
        datetime::Period(4, datetime::TimeUnit::Months).advance(kReference),
        datetime::Period(15, datetime::TimeUnit::Months).advance(kReference)};
    for (int year = 2; year <= 5; ++year) {
        dates.push_back(kReference.plusYears(year));
    }
    for (const SchemeCase& item : schemeMatrix()) {
        const DiscountCurve<double> target = makeTarget(item, dates, -0.002, 0.0, 0.0);
        const std::vector<CurvePillar> pillars =
            annualOisPillars(target, dates, calendar,
                             datetime::Period(1, datetime::TimeUnit::Years), 0);
        const DiscountCurve<double> curve = markets::bootstrapDiscountCurve(
            kReference, kZeroDc, item.space, item.scheme, pillars, 1e-14, item.tension);
        checkExactFit("stub/negative", curve, pillars);
        checkCurveSanity(curve, false);
    }
}

void testBasisCurveValidation() {
    const datetime::Calendar calendar = datetime::Calendar::noHolidays();
    std::vector<datetime::Date> rootDates;
    for (int year = 1; year <= 5; ++year) {
        rootDates.push_back(kReference.plusYears(year));
    }
    const SchemeCase rootCase{InterpolationSpace::LogDiscount, InterpolationScheme::Linear, 0.0};
    const DiscountCurve<double> rootTarget = makeTarget(rootCase, rootDates, 0.03, 0.0, 0.0);
    auto parent = std::make_shared<DiscountCurve<double>>(markets::bootstrapDiscountCurve(
        kReference, kZeroDc, rootCase.space, rootCase.scheme,
        annualOisPillars(rootTarget, rootDates, calendar,
                         datetime::Period(1, datetime::TimeUnit::Years), 2)));

    for (const InterpolationScheme spreadScheme :
         {InterpolationScheme::Linear, InterpolationScheme::Akima}) {
        for (const bool spreadOnParentLeg : {true, false}) {
            std::vector<BasisPillar> basisPillars;
            for (std::size_t i = 0; i < rootDates.size(); ++i) {
                BasisPillar pillar;
                pillar.maturity = rootDates[i];
                pillar.floatTenor = datetime::Period(3, datetime::TimeUnit::Months);
                pillar.quoteDayCounter = datetime::DayCounter(datetime::DayCount::Actual360);
                pillar.calendar = calendar;
                pillar.spreadOnParentLeg = spreadOnParentLeg;
                const double level = 0.0005 + 0.0001 * static_cast<double>(i);
                pillar.spread = spreadOnParentLeg ? level : -level;
                basisPillars.push_back(pillar);
            }
            const markets::SpreadCurve<double> child =
                markets::bootstrapSpreadCurve(parent, kReference, kZeroDc, spreadScheme,
                                              basisPillars, 1e-14, 0.0, nullptr);
            for (const BasisPillar& pillar : basisPillars) {
                util::checkClose("basis reprice",
                                 markets::impliedBasisSpread(child, pillar, kReference, kZeroDc),
                                 pillar.spread, 1e-9);
            }
            for (const datetime::Date& date : rootDates) {
                const double t = datetime::yearFraction(kReference, date, kZeroDc);
                CHECK(std::isfinite(child.discount(t)));
                CHECK(child.discount(t) > 0.0);
            }
        }
    }
}

void testErrorPaths() {
    const datetime::Calendar calendar = datetime::Calendar::noHolidays();
    CurvePillar first;
    first.maturity = kReference.plusYears(2);
    first.kind = PillarKind::Deposit;
    first.quoteDayCounter = kZeroDc;
    first.calendar = calendar;
    first.quote = 0.03;
    CurvePillar second;
    second.maturity = kReference.plusYears(1);
    second.kind = PillarKind::Deposit;
    second.quoteDayCounter = kZeroDc;
    second.calendar = calendar;
    second.quote = 0.03;
    bool threw = false;
    try {
        (void)markets::bootstrapDiscountCurve(kReference, kZeroDc, InterpolationSpace::LogDiscount,
                                              InterpolationScheme::Linear, {first, second});
    } catch (const std::invalid_argument&) {
        threw = true;
    }
    CHECK(threw);

    threw = false;
    try {
        (void)datetime::Schedule(kReference, kReference.plusYears(1),
                                 datetime::Period(0, datetime::TimeUnit::Months), calendar);
    } catch (const std::invalid_argument&) {
        threw = true;
    }
    CHECK(threw);
}

}  // namespace

int main() {
    testNodeAlignedRecovery();
    testOffNodeCashflowsAndLags();
    testStubsAndNegativeRates();
    testBasisCurveValidation();
    testErrorPaths();
    QTA_LOG_INFO("test", "test_bootstrap_validation: ok");
    return 0;
}
