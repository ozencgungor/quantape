#include "quantape/log/Log.h"
#include "quantape/markets/Curves/CurveRisk.h"
#include "quantape/markets/Curves/SpreadCurve.h"
#include "quantape/markets/Curves/StackRisk.h"
#include "quantape/markets/Curves/XccyBasisBuilder.h"
#include "quantape/math/LinearAlgebra/DenseSolve.h"
#include "quantape/util/Check.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <memory>
#include <stdexcept>
#include <variant>
#include <vector>

using namespace quantape;
using markets::DiscountCurve;
using markets::InterpolationScheme;
using markets::InterpolationSpace;

namespace {

void testXccyConstNotionalBootstrap() {
    const datetime::Date reference(2026, 9, 29);
    const datetime::DayCounter zeroDc(datetime::DayCount::Actual365Fixed);
    const datetime::Calendar calendar = datetime::Calendar::noHolidays();
    std::vector<datetime::Date> dates{reference};
    for (int years = 1; years <= 5; ++years) {
        dates.push_back(reference.plusYears(years));
    }
    const auto buildCurve = [&](double base, double slope) {
        std::vector<double> zeros;
        for (std::size_t i = 1; i < dates.size(); ++i) {
            const double t = datetime::yearFraction(reference, dates[i], zeroDc);
            zeros.push_back(base + slope * t);
        }
        return DiscountCurve<double>(
            reference, std::vector<datetime::Date>(dates.begin() + 1, dates.end()), zeroDc, zeros,
            InterpolationSpace::LogDiscount, InterpolationScheme::Linear);
    };
    const DiscountCurve<double> domesticDiscount = buildCurve(0.040, 0.0005);
    const DiscountCurve<double> domesticForecast = buildCurve(0.043, 0.0004);
    const DiscountCurve<double> foreignForecast = buildCurve(0.025, 0.0008);
    const DiscountCurve<double> foreignTarget = buildCurve(0.028, 0.0006);

    std::vector<markets::XccyPillar> pillars;
    for (std::size_t i = 1; i < dates.size(); ++i) {
        markets::XccyPillar pillar;
        pillar.maturity = dates[i];
        pillar.foreignTenor = datetime::Period(3, datetime::TimeUnit::Months);
        pillar.foreignCalendar = calendar;
        pillar.foreignDayCounter = datetime::DayCounter(datetime::DayCount::Actual360);
        pillar.domesticTenor = datetime::Period(3, datetime::TimeUnit::Months);
        pillar.domesticCalendar = calendar;
        pillar.domesticDayCounter = datetime::DayCounter(datetime::DayCount::Actual360);
        pillar.spread =
            markets::impliedXccyBasisSpread(foreignTarget, foreignForecast, domesticDiscount,
                                            domesticForecast, pillar, reference, zeroDc);
        CHECK(std::isfinite(pillar.spread));
        CHECK(std::abs(pillar.spread) < 0.05);
        pillars.push_back(pillar);
    }

    const DiscountCurve<double> curve = markets::bootstrapXccyDiscountCurve(
        domesticDiscount, domesticForecast, foreignForecast, reference, zeroDc,
        InterpolationSpace::LogDiscount, InterpolationScheme::Linear, pillars);
    CHECK(curve.size() == pillars.size() + 1);
    for (std::size_t i = 0; i < pillars.size(); ++i) {
        util::checkClose("xccy recovered zero", curve.zeros()[i + 1], foreignTarget.zeros()[i + 1],
                         1e-10);
        util::checkClose("xccy reprice",
                         markets::impliedXccyBasisSpread(curve, foreignForecast, domesticDiscount,
                                                         domesticForecast, pillars[i], reference,
                                                         zeroDc),
                         pillars[i].spread, 1e-10);
    }

    bool threw = false;
    try {
        std::vector<markets::XccyPillar> unsorted{pillars[2], pillars[1]};
        (void)markets::bootstrapXccyDiscountCurve(
            domesticDiscount, domesticForecast, foreignForecast, reference, zeroDc,
            InterpolationSpace::LogDiscount, InterpolationScheme::Linear, unsorted);
    } catch (const std::invalid_argument&) {
        threw = true;
    }
    CHECK(threw);
}

void testXccySpreadForecastsAndSpreadSide() {
    const datetime::Date reference(2026, 9, 29);
    const datetime::DayCounter zeroDc(datetime::DayCount::Actual365Fixed);
    const datetime::Calendar calendar = datetime::Calendar::noHolidays();
    std::vector<datetime::Date> dates{reference};
    for (int years = 1; years <= 5; ++years) {
        dates.push_back(reference.plusYears(years));
    }
    const auto buildCurve = [&](double base, double slope) {
        std::vector<double> zeros;
        for (std::size_t i = 1; i < dates.size(); ++i) {
            zeros.push_back(base + slope * datetime::yearFraction(reference, dates[i], zeroDc));
        }
        return DiscountCurve<double>(
            reference, std::vector<datetime::Date>(dates.begin() + 1, dates.end()), zeroDc, zeros,
            InterpolationSpace::LogDiscount, InterpolationScheme::Linear);
    };
    const DiscountCurve<double> domesticDiscount = buildCurve(0.040, 0.0005);
    const DiscountCurve<double> foreignTarget = buildCurve(0.028, 0.0006);
    auto domesticBase = std::make_shared<DiscountCurve<double>>(buildCurve(0.041, 0.0004));
    auto foreignBase = std::make_shared<DiscountCurve<double>>(buildCurve(0.024, 0.0007));
    std::vector<double> spreadTimes{0.0};
    std::vector<double> domesticSpreads{0.0};
    std::vector<double> foreignSpreads{0.0};
    for (std::size_t i = 1; i < dates.size(); ++i) {
        spreadTimes.push_back(datetime::yearFraction(reference, dates[i], zeroDc));
        domesticSpreads.push_back(0.0005 + 0.0001 * static_cast<double>(i));
        foreignSpreads.push_back(0.0002 + 0.0001 * static_cast<double>(i));
    }
    const markets::SpreadCurve<double> domesticForecast(domesticBase, spreadTimes, domesticSpreads);
    const markets::SpreadCurve<double> foreignForecast(foreignBase, spreadTimes, foreignSpreads);

    // Bench vs spread and spread vs spread forecasts; spread on either leg.
    for (const bool spreadOnForeign : {true, false}) {
        std::vector<markets::XccyPillar> pillars;
        for (std::size_t i = 1; i < dates.size(); ++i) {
            markets::XccyPillar pillar;
            pillar.maturity = dates[i];
            pillar.foreignTenor = datetime::Period(7, datetime::TimeUnit::Months);
            pillar.foreignPaymentLag = 2;
            pillar.domesticPaymentLag = 1;
            pillar.foreignBusinessDayConvention = datetime::BusinessDayConvention::Following;
            pillar.foreignCalendar = calendar;
            pillar.foreignDayCounter = datetime::DayCounter(datetime::DayCount::Actual360);
            pillar.domesticTenor = datetime::Period(3, datetime::TimeUnit::Months);
            pillar.domesticCalendar = calendar;
            pillar.domesticDayCounter = datetime::DayCounter(datetime::DayCount::Actual360);
            pillar.spreadOnForeignLeg = spreadOnForeign;
            pillar.spread =
                markets::impliedXccyBasisSpread(foreignTarget, foreignForecast, domesticDiscount,
                                                domesticForecast, pillar, reference, zeroDc);
            CHECK(std::isfinite(pillar.spread));
            pillars.push_back(pillar);
        }
        const DiscountCurve<double> curve = markets::bootstrapXccyDiscountCurve(
            domesticDiscount, domesticForecast, foreignForecast, reference, zeroDc,
            InterpolationSpace::LogDiscount, InterpolationScheme::Linear, pillars);
        for (std::size_t i = 0; i < pillars.size(); ++i) {
            util::checkClose("xccy spread-forecast recovered", curve.zeros()[i + 1],
                             foreignTarget.zeros()[i + 1], 1e-10);
            util::checkClose("xccy spread-forecast reprice",
                             markets::impliedXccyBasisSpread(curve, foreignForecast,
                                                             domesticDiscount, domesticForecast,
                                                             pillars[i], reference, zeroDc),
                             pillars[i].spread, 1e-10);
        }
    }
}

void testXccyCoupledBootstrap() {
    const datetime::Date reference(2026, 9, 29);
    const datetime::DayCounter zeroDayCounter(datetime::DayCount::Actual365Fixed);
    const datetime::Calendar calendar = datetime::Calendar::noHolidays();
    std::vector<datetime::Date> dates{reference};
    for (int years = 1; years <= 5; ++years) {
        dates.push_back(reference.plusYears(years));
    }
    const auto buildCurve = [&](double base, double slope, int maxYears) {
        std::vector<datetime::Date> pillarDates;
        std::vector<double> zeros;
        for (int years = 1; years <= maxYears; ++years) {
            const datetime::Date date = reference.plusYears(years);
            pillarDates.push_back(date);
            zeros.push_back(base + slope * datetime::yearFraction(reference, date, zeroDayCounter));
        }
        return DiscountCurve<double>(reference, pillarDates, zeroDayCounter, zeros,
                                     InterpolationSpace::LogDiscount, InterpolationScheme::Linear);
    };
    const DiscountCurve<double> domesticDiscount = buildCurve(0.040, 0.0005, 5);
    const DiscountCurve<double> domesticForecast = domesticDiscount;
    const DiscountCurve<double> xccyTarget = buildCurve(0.030, 0.0005, 3);
    auto foreignBase = std::make_shared<DiscountCurve<double>>(buildCurve(0.025, 0.0005, 3));
    const std::vector<double> spreadTimes{0.0, 1.0, 2.0, 3.0};
    const std::vector<double> spreadNodes{0.0, 0.0010, 0.0013, 0.0016};
    const markets::SpreadCurve<double> spreadTarget(foreignBase, spreadTimes, spreadNodes,
                                                    InterpolationScheme::Linear);

    std::vector<markets::XccyCoupledPillar> pillars;
    for (int years = 1; years <= 3; ++years) {
        const datetime::Date maturity = reference.plusYears(years);
        markets::XccyCoupledPillar coupled;
        coupled.xccy.maturity = maturity;
        coupled.xccy.foreignCalendar = calendar;
        coupled.xccy.domesticCalendar = calendar;
        coupled.xccy.spread = markets::impliedXccyBasisSpread(
            xccyTarget, spreadTarget, domesticDiscount, domesticForecast, coupled.xccy, reference,
            zeroDayCounter);
        coupled.basis.maturity = maturity;
        coupled.basis.calendar = calendar;
        coupled.basis.spread = markets::impliedBasisSpread(spreadTarget, coupled.basis, reference,
                                                           zeroDayCounter, &xccyTarget);
        pillars.push_back(coupled);
    }

    const markets::XccyCoupledResult coupled = markets::bootstrapXccyCoupled(
        domesticDiscount, domesticForecast, foreignBase, reference, zeroDayCounter,
        InterpolationSpace::LogDiscount, InterpolationScheme::Linear, pillars);
    CHECK(coupled.converged);
    CHECK(coupled.passes >= 1);
    CHECK(!coupled.usedJointFallback);
    for (std::size_t i = 0; i < pillars.size(); ++i) {
        util::checkClose("coupled xccy node", coupled.foreignDiscount.zeros()[i + 1],
                         xccyTarget.zeros()[i + 1], 1e-9);
        util::checkClose("coupled xccy reprice",
                         markets::impliedXccyBasisSpread(
                             coupled.foreignDiscount, coupled.foreignSpread, domesticDiscount,
                             domesticForecast, pillars[i].xccy, reference, zeroDayCounter),
                         pillars[i].xccy.spread, 1e-9);
        util::checkClose("coupled basis reprice",
                         markets::impliedBasisSpread(coupled.foreignSpread, pillars[i].basis,
                                                     reference, zeroDayCounter,
                                                     &coupled.foreignDiscount),
                         pillars[i].basis.spread, 1e-9);
    }
}

void testForecastNodeDateConversion() {
    // Native node times are inverted on the zero clock, so an ACT/360 node is
    // not misread on the ACT/365F offset (and leap-year ACT/365F nodes stay
    // exact).
    const datetime::Date reference(2026, 9, 29);
    const datetime::DayCounter act360(datetime::DayCount::Actual360);
    const datetime::DayCounter act365(datetime::DayCount::Actual365Fixed);
    CHECK(markets::forecastNodeDate(act360, reference, 30.0 / 360.0) == reference.plusDays(30));
    CHECK(markets::forecastNodeDate(act365, reference, 366.0 / 365.0) == reference.plusDays(366));
    CHECK(markets::forecastNodeDate(act365, reference, 0.0) == reference);
}

void testXccyJointFallbackAndRisk() {
    const datetime::Date reference(2026, 9, 29);
    const datetime::DayCounter zeroDayCounter(datetime::DayCount::Actual365Fixed);
    const datetime::Calendar calendar = datetime::Calendar::noHolidays();
    const auto buildCurve = [&](double base, double slope, int maxYears) {
        std::vector<datetime::Date> pillarDates;
        std::vector<double> zeros;
        for (int years = 1; years <= maxYears; ++years) {
            const datetime::Date date = reference.plusYears(years);
            pillarDates.push_back(date);
            zeros.push_back(base + slope * datetime::yearFraction(reference, date, zeroDayCounter));
        }
        return DiscountCurve<double>(reference, pillarDates, zeroDayCounter, zeros,
                                     InterpolationSpace::LogDiscount, InterpolationScheme::Linear);
    };
    const DiscountCurve<double> domesticDiscount = buildCurve(0.040, 0.0005, 5);
    const DiscountCurve<double> xccyTarget = buildCurve(0.030, 0.0005, 3);
    const DiscountCurve<double> foreignForecast = buildCurve(0.024, 0.0007, 3);
    const DiscountCurve<double> domesticForecast = buildCurve(0.043, 0.0004, 5);
    auto foreignBase = std::make_shared<DiscountCurve<double>>(buildCurve(0.025, 0.0005, 3));
    const markets::SpreadCurve<double> spreadTarget(
        foreignBase, std::vector<double>{0.0, 1.0, 2.0, 3.0},
        std::vector<double>{0.0, 0.0010, 0.0013, 0.0016}, InterpolationScheme::Linear);

    std::vector<markets::XccyCoupledPillar> coupledPillars;
    std::vector<markets::XccyPillar> xccyPillars;
    for (int years = 1; years <= 3; ++years) {
        const datetime::Date maturity = reference.plusYears(years);
        markets::XccyCoupledPillar cp;
        cp.xccy.maturity = maturity;
        cp.xccy.foreignCalendar = calendar;
        cp.xccy.domesticCalendar = calendar;
        cp.basis.maturity = maturity;
        cp.basis.calendar = calendar;
        cp.xccy.spread =
            markets::impliedXccyBasisSpread(xccyTarget, spreadTarget, domesticDiscount,
                                            domesticForecast, cp.xccy, reference, zeroDayCounter);
        cp.basis.spread = markets::impliedBasisSpread(spreadTarget, cp.basis, reference,
                                                      zeroDayCounter, &xccyTarget);
        coupledPillars.push_back(cp);
        xccyPillars.push_back(cp.xccy);
    }

    // Joint-LM fallback: one fixed-point pass is not enough, so LM takes over.
    math::FixedPointOptions options;
    options.maxPasses = 1;
    const markets::XccyCoupledResult fallback = markets::bootstrapXccyCoupled(
        domesticDiscount, domesticForecast, foreignBase, reference, zeroDayCounter,
        InterpolationSpace::LogDiscount, InterpolationScheme::Linear, coupledPillars, 1e-14, 0.0, 1,
        options);
    CHECK(fallback.usedJointFallback);
    CHECK(!fallback.converged);
    CHECK(fallback.passes == 1);
    for (std::size_t i = 0; i < xccyPillars.size(); ++i) {
        util::checkClose("joint fallback xccy node", fallback.foreignDiscount.zeros()[i + 1],
                         xccyTarget.zeros()[i + 1], 1e-8);
    }

    // Jacobian rows vs central FD.
    std::vector<double> f;
    std::vector<double> c;
    markets::assembleXccyJacobian(xccyTarget, spreadTarget, domesticDiscount, domesticForecast,
                                  xccyPillars, reference, f, c);
    const std::size_t n = xccyPillars.size();
    const std::size_t m = domesticDiscount.size() - 1;
    const double h = 1e-6;
    const auto foreignBumped = [&](std::size_t node, double delta) {
        std::vector<double> z = xccyTarget.zeros();
        z[node] += delta;
        return DiscountCurve<double>(xccyTarget.times(), z, xccyTarget.space(), xccyTarget.scheme(),
                                     xccyTarget.tension(), xccyTarget.switchIndex());
    };
    const auto domesticBumped = [&](std::size_t node, double delta) {
        std::vector<double> z = domesticDiscount.zeros();
        z[node] += delta;
        return DiscountCurve<double>(domesticDiscount.times(), z, domesticDiscount.space(),
                                     domesticDiscount.scheme(), domesticDiscount.tension(),
                                     domesticDiscount.switchIndex());
    };
    const auto quote = [&](const DiscountCurve<double>& fc, const DiscountCurve<double>& dc) {
        return markets::impliedXccyBasisSpread(fc, spreadTarget, dc, domesticForecast,
                                               xccyPillars[0], reference, zeroDayCounter);
    };
    util::checkClose("xccy foreign row FD", f[0],
                     (quote(foreignBumped(1, h), domesticDiscount) -
                      quote(foreignBumped(1, -h), domesticDiscount)) /
                         (2.0 * h),
                     1e-8);
    util::checkClose(
        "xccy domestic row FD", c[0],
        (quote(xccyTarget, domesticBumped(1, h)) - quote(xccyTarget, domesticBumped(1, -h))) /
            (2.0 * h),
        1e-8);

    // Chain-aware root rows and forecast factor rows.
    auto domesticBase = std::make_shared<DiscountCurve<double>>(domesticDiscount);
    const markets::SpreadCurve<double> domesticForecastSpread(
        domesticBase, std::vector<double>{0.0, 1.0, 2.0, 3.0, 4.0, 5.0},
        std::vector<double>{0.0, 0.0005, 0.0005, 0.0005, 0.0005, 0.0005},
        InterpolationScheme::Linear);
    std::vector<double> fFull;
    std::vector<double> cFull;
    std::vector<double> gFull;
    std::vector<double> hFull;
    markets::xccySwapJacobianRows(xccyTarget, spreadTarget, domesticDiscount,
                                  domesticForecastSpread, xccyPillars[0], reference, zeroDayCounter,
                                  fFull, cFull, gFull, hFull);
    const auto quoteChain = [&](const DiscountCurve<double>& rootCurve) {
        auto parent = std::make_shared<DiscountCurve<double>>(rootCurve);
        const auto& nodes = domesticForecastSpread.spreadNodes();
        const markets::SpreadCurve<double> forecast(parent, nodes.times(), nodes.zeros(),
                                                    nodes.scheme(), nodes.tension());
        return markets::impliedXccyBasisSpread(xccyTarget, spreadTarget, rootCurve, forecast,
                                               xccyPillars[0], reference, zeroDayCounter);
    };
    util::checkClose(
        "xccy chain root row FD", cFull[0],
        (quoteChain(domesticBumped(1, h)) - quoteChain(domesticBumped(1, -h))) / (2.0 * h), 1e-8);
    const auto quoteForward = [&](std::size_t node, double delta) {
        const auto& nodes = spreadTarget.spreadNodes();
        std::vector<double> spreads = nodes.zeros();
        spreads[node] += delta;
        const markets::SpreadCurve<double> forecast(spreadTarget.parentPointer(), nodes.times(),
                                                    spreads, nodes.scheme(), nodes.tension());
        return markets::impliedXccyBasisSpread(xccyTarget, forecast, domesticDiscount,
                                               domesticDiscount, xccyPillars[0], reference,
                                               zeroDayCounter);
    };
    util::checkClose("xccy foreign forecast row FD", gFull[0],
                     (quoteForward(1, h) - quoteForward(1, -h)) / (2.0 * h), 1e-8);

    // Implicit cross block dz/dp = -F^{-1} C vs a refit with a bumped root node.
    std::vector<double> rhs(n);
    for (std::size_t i = 0; i < n; ++i) {
        rhs[i] = -c[i * m];
    }
    const std::vector<double> crossColumn = quantape::math::solveDense(f, n, rhs);
    const auto refit = [&](const DiscountCurve<double>& root) {
        return markets::bootstrapXccyDiscountCurve(root, domesticForecast, spreadTarget, reference,
                                                   zeroDayCounter, InterpolationSpace::LogDiscount,
                                                   InterpolationScheme::Linear, xccyPillars);
    };
    const DiscountCurve<double> zPlus = refit(domesticBumped(1, h));
    const DiscountCurve<double> zMinus = refit(domesticBumped(1, -h));
    for (std::size_t i = 0; i < n; ++i) {
        util::checkClose("xccy cross implicit", crossColumn[i],
                         (zPlus.zeros()[i + 1] - zMinus.zeros()[i + 1]) / (2.0 * h), 1e-6);
    }

    // Stack tree pass smoke test.
    std::vector<markets::CurvePillar> rootPillars;
    for (int years = 1; years <= 5; ++years) {
        markets::CurvePillar pillar;
        pillar.maturity = reference.plusYears(years);
        pillar.kind = markets::PillarKind::OisSwap;
        pillar.quoteDayCounter = zeroDayCounter;
        pillar.calendar = calendar;
        pillar.quote = markets::impliedQuote(pillar, reference, domesticDiscount);
        rootPillars.push_back(pillar);
    }
    markets::XccyChildInput<DiscountCurve<double>, DiscountCurve<double>> child;
    child.foreignForecastRole = markets::CurveRole::TenorBasis;
    child.domesticForecastRole = markets::CurveRole::IborOisBasis;
    child.foreignDiscount = &fallback.foreignDiscount;
    child.pillars = xccyPillars;
    child.dVdForeignZeros.assign(fallback.foreignDiscount.size(), 0.0);
    child.dVdForeignZeros[1] = 1.0;
    child.foreignForecast = &foreignForecast;
    child.domesticForecast = &domesticForecast;
    std::vector<std::variant<decltype(child)>> children{child};
    std::vector<double> dVdRoot(domesticDiscount.size(), 0.0);
    dVdRoot[1] = 1.0;
    const std::vector<markets::StackRiskEntry> entries =
        markets::stackQuoteRiskXccy(domesticDiscount, rootPillars, dVdRoot, children, reference);
    CHECK(entries.size() == 4);
    CHECK(entries[0].points.size() == rootPillars.size());
    CHECK(entries[1].points.size() == n);
    CHECK(entries[2].points.size() == 3); // funder: foreign forecast spread nodes
    CHECK(entries[3].points.size() == domesticDiscount.size() - 1);
    for (const markets::StackRiskEntry& entry : entries) {
        entry.validate();
    }

    // Every xccy path reports under its own role, with maturity labels and
    // buckets in pillar order; the role buckets add back to the full table.
    CHECK(entries[0].points[0].role == markets::CurveRole::Discount);
    CHECK(entries[1].points[0].role == markets::CurveRole::XccyBasis);
    CHECK(entries[2].points[0].role == markets::CurveRole::TenorBasis);
    CHECK(entries[3].points[0].role == markets::CurveRole::IborOisBasis);
    for (const markets::QuotePoint& point : entries[1].points) {
        CHECK(point.role == markets::CurveRole::XccyBasis);
    }
    for (const markets::QuotePoint& point : entries[2].points) {
        CHECK(point.role == markets::CurveRole::TenorBasis);
    }
    for (const markets::QuotePoint& point : entries[3].points) {
        CHECK(point.role == markets::CurveRole::IborOisBasis);
    }
    CHECK(entries[1].points[0].bucket == "1Y");
    CHECK(entries[1].points[0].label == "Xccy 1Y");
    CHECK(entries[2].points[0].bucket == "1Y");
    CHECK(entries[2].points[0].label == "XccyFwd 1Y");
    CHECK(entries[3].points[0].bucket == "1Y");
    CHECK(entries[3].points[0].label == "XccyDom 1Y");
    double tableTotal = 0.0;
    for (const markets::StackRiskEntry& entry : entries) {
        for (const markets::QuotePoint& point : entry.points) {
            tableTotal += point.delta;
        }
    }
    double roleTotal = 0.0;
    bool sawDiscount = false;
    bool sawXccyBasis = false;
    bool sawTenorBasis = false;
    bool sawIborOisBasis = false;
    for (const markets::RiskBucket& bucket : markets::stackRoleBuckets(entries)) {
        roleTotal += bucket.delta;
        sawDiscount = sawDiscount || bucket.label == "Discount";
        sawXccyBasis = sawXccyBasis || bucket.label == "XccyBasis";
        sawTenorBasis = sawTenorBasis || bucket.label == "TenorBasis";
        sawIborOisBasis = sawIborOisBasis || bucket.label == "IborOisBasis";
    }
    util::checkClose("xccy role bucket total", roleTotal, tableTotal, 1e-12);
    CHECK(sawDiscount);
    CHECK(sawXccyBasis);
    CHECK(sawTenorBasis);
    CHECK(sawIborOisBasis);

    // The wrapper is the general engine: a hand-built unified input list (the
    // root, the xccy child, and the two forecast factor blocks) must produce
    // the same entry metadata and deltas to solver precision.
    {
        const markets::StackCurveView::Ptr rootView =
            markets::StackCurveView::make(domesticDiscount);
        const markets::StackCurveView::Ptr foreignForecastView =
            markets::StackCurveView::make(*child.foreignForecast);
        const markets::StackCurveView::Ptr domesticForecastView =
            markets::StackCurveView::make(*child.domesticForecast);
        std::vector<markets::StackCurveInput> unified(4);
        unified[0].curve = rootView;
        unified[0].role = markets::CurveRole::Discount;
        unified[0].discountPillars = rootPillars;
        unified[0].dVdNodes = dVdRoot;
        unified[1].curve = markets::StackCurveView::make(*child.foreignDiscount);
        unified[1].role = child.role;
        unified[1].xccyPillars = child.pillars;
        unified[1].xccy =
            markets::XccyRowInput{foreignForecastView, domesticForecastView, rootView};
        unified[1].dVdNodes = child.dVdForeignZeros;
        unified[2].curve = foreignForecastView;
        unified[2].role = child.foreignForecastRole;
        unified[2].factor = markets::StackFactorInput{
            "XccyFwd", markets::forecastNodeDayCounter(*child.foreignForecast)};
        unified[2].dVdNodes.assign(foreignForecastView->size(), 0.0);
        unified[3].curve = domesticForecastView;
        unified[3].role = child.domesticForecastRole;
        unified[3].factor = markets::StackFactorInput{
            "XccyDom", markets::forecastNodeDayCounter(*child.domesticForecast)};
        unified[3].dVdNodes.assign(domesticForecastView->size(), 0.0);
        const std::vector<markets::StackRiskEntry> general =
            markets::stackQuoteRisk(unified, reference);
        CHECK(general.size() == entries.size());
        for (std::size_t k = 0; k < general.size(); ++k) {
            CHECK(general[k].role == entries[k].role);
            CHECK(general[k].points.size() == entries[k].points.size());
            for (std::size_t j = 0; j < general[k].points.size(); ++j) {
                CHECK(general[k].points[j].label == entries[k].points[j].label);
                CHECK(general[k].points[j].bucket == entries[k].points[j].bucket);
                CHECK(general[k].points[j].year == entries[k].points[j].year);
                CHECK(general[k].points[j].role == entries[k].points[j].role);
                util::checkClose("xccy wrapper vs general engine", general[k].points[j].delta,
                                 entries[k].points[j].delta, 1e-10);
            }
        }

        // Cross-currency gamma now runs on the analytic rows: the general
        // curvature path differentiates the assembled Jacobian once. A zero
        // HZeta still leaves the bootstrap-curvature correction, and the
        // result must be a finite, symmetric matrix.
        const std::size_t unifiedDim = unified[0].curve->size() + unified[1].curve->size() +
                                       unified[2].curve->size() + unified[3].curve->size() - 4;
        const std::vector<double> zeroHessian(unifiedDim * unifiedDim, 0.0);
        const markets::StackQuoteGamma xccyGamma =
            markets::stackQuoteGamma(unified, zeroHessian, reference);
        CHECK(xccyGamma.dim == unifiedDim);
        for (std::size_t a = 0; a < unifiedDim; ++a) {
            for (std::size_t b = a + 1; b < unifiedDim; ++b) {
                util::checkClose("xccy gamma symmetry", xccyGamma.at(a, b), xccyGamma.at(b, a),
                                 1e-12);
            }
        }

        // An xccy row reference that is not a block of the input list is
        // rejected instead of silently binding to an equal-valued curve.
        {
            std::vector<markets::StackCurveInput> detached = unified;
            detached[1].xccy->foreignForecast =
                markets::StackCurveView::make(fallback.foreignSpread);
            bool referenceThrew = false;
            try {
                (void)markets::stackQuoteRisk(detached, reference);
            } catch (const std::invalid_argument&) {
                referenceThrew = true;
            }
            CHECK(referenceThrew);
        }

        // Delta regression: the block-elimination order the wrapper used
        // historically (child solve, root absorption, forecast residuals)
        // recovers the same root, child and factor deltas.
        const std::size_t nRef = xccyPillars.size();
        const std::size_t mRef = rootPillars.size();
        std::vector<double> fRootRef;
        std::vector<double> scratchRef;
        markets::assembleQuoteJacobian(domesticDiscount, rootPillars, reference, fRootRef,
                                       scratchRef);
        std::vector<double> fRef;
        std::vector<double> crossRef;
        std::vector<double> fwdRef;
        std::vector<double> domRef;
        markets::assembleXccyJacobianFull(*child.foreignDiscount, *child.foreignForecast,
                                          domesticDiscount, *child.domesticForecast, xccyPillars,
                                          reference, fRef, crossRef, fwdRef, domRef);
        std::vector<double> fChildTranspose(nRef * nRef);
        for (std::size_t j = 0; j < nRef; ++j) {
            for (std::size_t i = 0; i < nRef; ++i) {
                fChildTranspose[i * nRef + j] = fRef[j * nRef + i];
            }
        }
        std::vector<double> childG(nRef);
        for (std::size_t i = 0; i < nRef; ++i) {
            childG[i] = child.dVdForeignZeros[i + 1];
        }
        const std::vector<double> xChildRef = math::solveDense(fChildTranspose, nRef, childG);
        std::vector<double> rootG(mRef);
        for (std::size_t j = 0; j < mRef; ++j) {
            rootG[j] = dVdRoot[j + 1];
        }
        for (std::size_t i = 0; i < nRef; ++i) {
            for (std::size_t j = 0; j < mRef; ++j) {
                rootG[j] -= crossRef[i * mRef + j] * xChildRef[i];
            }
        }
        std::vector<double> fRootTranspose(mRef * mRef);
        for (std::size_t j = 0; j < mRef; ++j) {
            for (std::size_t i = 0; i < mRef; ++i) {
                fRootTranspose[i * mRef + j] = fRootRef[j * mRef + i];
            }
        }
        const std::vector<double> rootRef = math::solveDense(fRootTranspose, mRef, rootG);
        for (std::size_t j = 0; j < mRef; ++j) {
            util::checkClose("xccy root delta regression", entries[0].points[j].delta, rootRef[j],
                             1e-10);
        }
        for (std::size_t i = 0; i < nRef; ++i) {
            util::checkClose("xccy child delta regression", entries[1].points[i].delta,
                             xChildRef[i], 1e-10);
        }
        for (std::size_t k = 0; k < entries[2].points.size(); ++k) {
            double reference = 0.0;
            for (std::size_t i = 0; i < nRef; ++i) {
                reference -= fwdRef[i * entries[2].points.size() + k] * xChildRef[i];
            }
            util::checkClose("xccy fwd factor delta regression", entries[2].points[k].delta,
                             reference, 1e-10);
        }
        for (std::size_t k = 0; k < entries[3].points.size(); ++k) {
            double reference = 0.0;
            for (std::size_t i = 0; i < nRef; ++i) {
                reference -= domRef[i * entries[3].points.size() + k] * xChildRef[i];
            }
            util::checkClose("xccy dom factor delta regression", entries[3].points[k].delta,
                             reference, 1e-10);
        }
    }

    // Wrapper validation keeps the general engine's input contract: a wrong
    // root sensitivity size, a root pillar count that misses the nodes, or a
    // child without a foreign discount curve are all rejected.
    {
        const auto throwsInvalid = [](const auto& call) {
            try {
                call();
            } catch (const std::invalid_argument&) {
                return true;
            }
            return false;
        };
        const std::vector<std::variant<decltype(child)>> single{child};
        std::vector<double> wrongRoot(dVdRoot.size() + 1, 0.0);
        CHECK(throwsInvalid([&] {
            (void)markets::stackQuoteRiskXccy(domesticDiscount, rootPillars, wrongRoot, single,
                                              reference);
        }));
        const std::vector<markets::CurvePillar> shortPillars(rootPillars.begin(),
                                                             rootPillars.end() - 1);
        CHECK(throwsInvalid([&] {
            (void)markets::stackQuoteRiskXccy(domesticDiscount, shortPillars, dVdRoot, single,
                                              reference);
        }));
        auto broken = child;
        broken.foreignDiscount = nullptr;
        const std::vector<std::variant<decltype(child)>> brokenChildren{broken};
        CHECK(throwsInvalid([&] {
            (void)markets::stackQuoteRiskXccy(domesticDiscount, rootPillars, dVdRoot,
                                              brokenChildren, reference);
        }));
    }
}

/// Analytic cross-currency rows against the finite-difference reference, for
/// every referenced block: the leg discount curves, the forecast native nodes,
/// a depth-2 foreign forecast chain and a domestic forecast parented on the
/// domestic discount itself, with per-leg payment lags and stub schedules.
void testXccyAnalyticRowsAgainstFiniteDifference() {
    const datetime::Date reference(2026, 9, 29);
    const datetime::DayCounter zeroDc(datetime::DayCount::Actual365Fixed);
    const datetime::Calendar calendar = datetime::Calendar::noHolidays();
    const auto buildCurve = [&](double base, double slope) {
        const std::vector<datetime::Date> dates{reference.plusYears(1), reference.plusYears(2),
                                                reference.plusYears(3), reference.plusYears(4)};
        std::vector<double> zeros;
        for (const datetime::Date& date : dates) {
            zeros.push_back(base + slope * datetime::yearFraction(reference, date, zeroDc));
        }
        return DiscountCurve<double>(reference, dates, zeroDc, zeros,
                                     InterpolationSpace::LogDiscount, InterpolationScheme::Linear);
    };
    const DiscountCurve<double> foreignDiscount = buildCurve(0.030, 0.0006);
    // The domestic discount object also parents the domestic forecast, so its
    // block row carries both the discount and the forecast-parent sensitivity.
    auto domesticParent = std::make_shared<DiscountCurve<double>>(buildCurve(0.040, 0.0005));
    const std::vector<double> nodeTimes{0.0, 1.0, 2.0, 3.0, 4.0};
    const markets::SpreadCurve<double> domesticForecast(
        domesticParent, nodeTimes, std::vector<double>{0.0, 0.0003, 0.0004, 0.0005, 0.0006},
        InterpolationScheme::Linear);
    auto foreignBase = std::make_shared<DiscountCurve<double>>(buildCurve(0.025, 0.0007));
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
        markets::StackCurveView::make(*domesticParent);
    const markets::StackCurveView::Ptr domesticForecastView =
        markets::StackCurveView::make(domesticForecast);

    for (const bool spreadOnForeign : {true, false}) {
        markets::XccyPillar pillar;
        pillar.maturity = reference.plusYears(3);
        pillar.foreignTenor = datetime::Period(7, datetime::TimeUnit::Months);
        pillar.domesticTenor = datetime::Period(4, datetime::TimeUnit::Months);
        pillar.foreignPaymentLag = 2;
        pillar.domesticPaymentLag = 1;
        pillar.foreignBusinessDayConvention = datetime::BusinessDayConvention::Following;
        pillar.domesticBusinessDayConvention = datetime::BusinessDayConvention::ModifiedFollowing;
        pillar.foreignCalendar = calendar;
        pillar.domesticCalendar = calendar;
        pillar.foreignDayCounter = datetime::DayCounter(datetime::DayCount::Actual360);
        pillar.domesticDayCounter = datetime::DayCounter(datetime::DayCount::Actual360);
        pillar.spreadOnForeignLeg = spreadOnForeign;
        const std::vector<markets::XccyRowBlock> analytic = markets::xccySwapJacobianRowsView(
            *foreignDiscountView, *foreignForecastView, *domesticDiscountView,
            *domesticForecastView, pillar, reference, zeroDc);
        const std::vector<markets::XccyRowBlock> finiteDifference =
            markets::xccySwapJacobianRowsViewFiniteDifference(
                *foreignDiscountView, *foreignForecastView, *domesticDiscountView,
                *domesticForecastView, pillar, reference, zeroDc);
        // Foreign discount, foreign forecast, two foreign ancestors, domestic
        // discount (merged with the domestic forecast parent) and domestic
        // forecast.
        CHECK(analytic.size() == 6);
        CHECK(finiteDifference.size() == analytic.size());
        double maxRowDeviation = 0.0;
        for (std::size_t b = 0; b < analytic.size(); ++b) {
            CHECK(analytic[b].curve->identity() == finiteDifference[b].curve->identity());
            CHECK(analytic[b].row.size() == finiteDifference[b].row.size());
            for (std::size_t i = 0; i < analytic[b].row.size(); ++i) {
                maxRowDeviation = std::max(
                    maxRowDeviation, std::abs(analytic[b].row[i] - finiteDifference[b].row[i]));
                util::checkClose("xccy analytic row vs FD", analytic[b].row[i],
                                 finiteDifference[b].row[i], 1e-6);
            }
        }
        QTA_LOG_INFO("test", "xccy analytic rows vs FD ({} blocks, spread-on-foreign={}): max {}",
                     analytic.size(), spreadOnForeign, maxRowDeviation);
        double foreignBaseRisk = 0.0;
        double domesticDiscountRisk = 0.0;
        for (const markets::XccyRowBlock& block : analytic) {
            if (block.curve->identity() == foreignBase.get()) {
                for (const double value : block.row) {
                    foreignBaseRisk += std::abs(value);
                }
            }
            if (block.curve->identity() == domesticDiscountView->identity()) {
                for (const double value : block.row) {
                    domesticDiscountRisk += std::abs(value);
                }
            }
        }
        CHECK(foreignBaseRisk > 1e-8);
        CHECK(domesticDiscountRisk > 1e-8);
    }
}

/// Xccy gamma against an independent finite difference of the deltas: the
/// quote-space Hessian is compared with the central difference of
/// `stackQuoteRisk` after bumping each root and xccy quote and re-bootstrapping
/// the curve, with the quadratic value's node gradient re-evaluated at the
/// rebuilt nodes.
void testXccyGammaFiniteDifference() {
    const datetime::Date reference(2026, 9, 29);
    const datetime::DayCounter zeroDc(datetime::DayCount::Actual365Fixed);
    const datetime::Calendar calendar = datetime::Calendar::noHolidays();
    const auto buildCurve = [&](double base, double slope) {
        const std::vector<datetime::Date> dates{reference.plusYears(1), reference.plusYears(2)};
        std::vector<double> zeros;
        for (const datetime::Date& date : dates) {
            zeros.push_back(base + slope * datetime::yearFraction(reference, date, zeroDc));
        }
        return DiscountCurve<double>(reference, dates, zeroDc, zeros,
                                     InterpolationSpace::LogDiscount, InterpolationScheme::Linear);
    };
    const DiscountCurve<double> rootTarget = buildCurve(0.030, 0.0004);
    std::vector<markets::CurvePillar> rootPillars;
    for (int year = 1; year <= 2; ++year) {
        markets::CurvePillar pillar;
        pillar.maturity = reference.plusYears(year);
        pillar.kind = markets::PillarKind::OisSwap;
        pillar.quoteDayCounter = zeroDc;
        pillar.calendar = calendar;
        pillar.quote = markets::impliedQuote(pillar, reference, rootTarget);
        rootPillars.push_back(pillar);
    }
    const auto bootstrapRoot = [&](const std::vector<markets::CurvePillar>& quotes) {
        return std::make_shared<DiscountCurve<double>>(
            markets::bootstrapDiscountCurve(reference, zeroDc, InterpolationSpace::LogDiscount,
                                            InterpolationScheme::Linear, quotes));
    };
    std::shared_ptr<DiscountCurve<double>> rootPtr = bootstrapRoot(rootPillars);
    const std::vector<double> forecastTimes{0.0, 1.0, 2.0};
    const std::vector<double> forecastSpreads{0.0, 0.0008, 0.0011};
    const auto makeForecast = [&](const std::shared_ptr<DiscountCurve<double>>& parent) {
        return markets::SpreadCurve<double>(parent, forecastTimes, forecastSpreads,
                                            InterpolationScheme::Linear);
    };
    // The foreign forecast is parented on the root object, so a root bump
    // reaches the xccy rows through the forecast parent chain.
    const markets::SpreadCurve<double> foreignForecast = makeForecast(rootPtr);
    const DiscountCurve<double> domesticForecast = buildCurve(0.041, 0.0003);
    const DiscountCurve<double> foreignTarget = buildCurve(0.028, 0.0005);
    std::vector<markets::XccyPillar> xccyPillars;
    for (int year = 1; year <= 2; ++year) {
        markets::XccyPillar pillar;
        pillar.maturity = reference.plusYears(year);
        pillar.foreignTenor = datetime::Period(3, datetime::TimeUnit::Months);
        pillar.domesticTenor = datetime::Period(3, datetime::TimeUnit::Months);
        pillar.foreignCalendar = calendar;
        pillar.domesticCalendar = calendar;
        pillar.foreignDayCounter = datetime::DayCounter(datetime::DayCount::Actual360);
        pillar.domesticDayCounter = datetime::DayCounter(datetime::DayCount::Actual360);
        pillar.spread = markets::impliedXccyBasisSpread(
            foreignTarget, foreignForecast, *rootPtr, domesticForecast, pillar, reference, zeroDc);
        xccyPillars.push_back(pillar);
    }
    const auto bootstrapChild = [&](const DiscountCurve<double>& rootCurve,
                                    const markets::SpreadCurve<double>& forecast,
                                    const std::vector<markets::XccyPillar>& quotes) {
        return markets::bootstrapXccyDiscountCurve(rootCurve, domesticForecast, forecast, reference,
                                                   zeroDc, InterpolationSpace::LogDiscount,
                                                   InterpolationScheme::Linear, quotes);
    };
    const DiscountCurve<double> childBase = bootstrapChild(*rootPtr, foreignForecast, xccyPillars);

    const std::size_t m = rootPillars.size();
    const std::size_t n = xccyPillars.size();
    const std::size_t p = foreignForecast.size() - 1;
    const std::size_t q = domesticForecast.size() - 1;
    const std::size_t dim = m + n + p + q;
    std::vector<double> g(dim, 0.0);
    for (std::size_t i = 0; i < dim; ++i) {
        g[i] = 0.01 + 0.003 * static_cast<double>(i);
    }
    std::vector<double> HZeta(dim * dim, 0.0);
    for (std::size_t i = 0; i < dim; ++i) {
        for (std::size_t j = 0; j < dim; ++j) {
            HZeta[i * dim + j] =
                0.02 * static_cast<double>(std::min(i, j) + 1) + (i == j ? 0.05 : 0.0);
        }
    }
    const auto zetaAt =
        [&](const DiscountCurve<double>& rootCurve, const DiscountCurve<double>& childCurve,
            const markets::SpreadCurve<double>& fwdCurve, const DiscountCurve<double>& domCurve) {
            std::vector<double> zeta;
            zeta.reserve(dim);
            for (std::size_t i = 1; i <= m; ++i) {
                zeta.push_back(rootCurve.zeros()[i]);
            }
            for (std::size_t i = 1; i <= n; ++i) {
                zeta.push_back(childCurve.zeros()[i]);
            }
            for (std::size_t i = 1; i <= p; ++i) {
                zeta.push_back(fwdCurve.spreadNodes().zeros()[i]);
            }
            for (std::size_t i = 1; i <= q; ++i) {
                zeta.push_back(domCurve.zeros()[i]);
            }
            return zeta;
        };
    const auto gradientAt = [&](const std::vector<double>& zeta) {
        std::vector<double> nodeGradient(dim, 0.0);
        for (std::size_t i = 0; i < dim; ++i) {
            double sum = g[i];
            for (std::size_t j = 0; j < dim; ++j) {
                sum += HZeta[i * dim + j] * zeta[j];
            }
            nodeGradient[i] = sum;
        }
        return nodeGradient;
    };
    const auto buildInputs = [&](const std::shared_ptr<DiscountCurve<double>>& rootCurve,
                                 const DiscountCurve<double>& childCurve,
                                 const markets::SpreadCurve<double>& fwdCurve,
                                 const DiscountCurve<double>& domCurve,
                                 const std::vector<double>& nodeGradient) {
        const markets::StackCurveView::Ptr rootView = markets::StackCurveView::make(*rootCurve);
        const markets::StackCurveView::Ptr childView = markets::StackCurveView::make(childCurve);
        const markets::StackCurveView::Ptr fwdView = markets::StackCurveView::make(fwdCurve);
        const markets::StackCurveView::Ptr domView = markets::StackCurveView::make(domCurve);
        std::vector<markets::StackCurveInput> inputs(4);
        inputs[0].curve = rootView;
        inputs[0].role = markets::CurveRole::Discount;
        inputs[0].discountPillars = rootPillars;
        inputs[0].dVdNodes.assign(rootView->size(), 0.0);
        inputs[1].curve = childView;
        inputs[1].role = markets::CurveRole::XccyBasis;
        inputs[1].xccyPillars = xccyPillars;
        inputs[1].xccy = markets::XccyRowInput{fwdView, domView, rootView};
        inputs[1].dVdNodes.assign(childView->size(), 0.0);
        inputs[2].curve = fwdView;
        inputs[2].role = markets::CurveRole::Forecast;
        inputs[2].factor =
            markets::StackFactorInput{"XccyFwd", markets::forecastNodeDayCounter(fwdCurve)};
        inputs[2].dVdNodes.assign(fwdView->size(), 0.0);
        inputs[3].curve = domView;
        inputs[3].role = markets::CurveRole::Forecast;
        inputs[3].factor =
            markets::StackFactorInput{"XccyDom", markets::forecastNodeDayCounter(domCurve)};
        inputs[3].dVdNodes.assign(domView->size(), 0.0);
        for (std::size_t i = 0; i < m; ++i) {
            inputs[0].dVdNodes[i + 1] = nodeGradient[i];
        }
        for (std::size_t i = 0; i < n; ++i) {
            inputs[1].dVdNodes[i + 1] = nodeGradient[m + i];
        }
        for (std::size_t i = 0; i < p; ++i) {
            inputs[2].dVdNodes[i + 1] = nodeGradient[m + n + i];
        }
        for (std::size_t i = 0; i < q; ++i) {
            inputs[3].dVdNodes[i + 1] = nodeGradient[m + n + p + i];
        }
        return inputs;
    };
    const auto flatten = [](const std::vector<markets::StackRiskEntry>& entries) {
        std::vector<double> flat;
        for (const markets::StackRiskEntry& entry : entries) {
            for (const markets::QuotePoint& point : entry.points) {
                flat.push_back(point.delta);
            }
        }
        return flat;
    };
    const std::vector<double> baseZeta =
        zetaAt(*rootPtr, childBase, foreignForecast, domesticForecast);
    const std::vector<double> baseGradient = gradientAt(baseZeta);
    const markets::StackQuoteGamma gamma = markets::stackQuoteGamma(
        buildInputs(rootPtr, childBase, foreignForecast, domesticForecast, baseGradient), HZeta,
        reference);
    CHECK(gamma.dim == dim);
    for (std::size_t i = 0; i < dim; ++i) {
        for (std::size_t j = i + 1; j < dim; ++j) {
            util::checkClose("xccy gamma symmetry", gamma.at(i, j), gamma.at(j, i), 1e-12);
        }
    }
    const double epsilon = 1e-6;
    double maxGammaDeviation = 0.0;
    for (std::size_t r = 0; r < m; ++r) {
        std::vector<markets::CurvePillar> up = rootPillars;
        std::vector<markets::CurvePillar> down = rootPillars;
        up[r].quote += epsilon;
        down[r].quote -= epsilon;
        const std::shared_ptr<DiscountCurve<double>> rootUp = bootstrapRoot(up);
        const std::shared_ptr<DiscountCurve<double>> rootDown = bootstrapRoot(down);
        const markets::SpreadCurve<double> fwdUp = makeForecast(rootUp);
        const markets::SpreadCurve<double> fwdDown = makeForecast(rootDown);
        const DiscountCurve<double> childUp = bootstrapChild(*rootUp, fwdUp, xccyPillars);
        const DiscountCurve<double> childDown = bootstrapChild(*rootDown, fwdDown, xccyPillars);
        const std::vector<double> plus = flatten(markets::stackQuoteRisk(
            buildInputs(rootUp, childUp, fwdUp, domesticForecast,
                        gradientAt(zetaAt(*rootUp, childUp, fwdUp, domesticForecast))),
            reference));
        const std::vector<double> minus = flatten(markets::stackQuoteRisk(
            buildInputs(rootDown, childDown, fwdDown, domesticForecast,
                        gradientAt(zetaAt(*rootDown, childDown, fwdDown, domesticForecast))),
            reference));
        for (std::size_t a = 0; a < dim; ++a) {
            const double fd = (plus[a] - minus[a]) / (2.0 * epsilon);
            maxGammaDeviation =
                std::max(maxGammaDeviation, std::abs(gamma.hessian[a * dim + r] - fd));
            util::checkClose("xccy gamma root quote FD", gamma.hessian[a * dim + r], fd, 1e-6);
        }
    }
    for (std::size_t r = 0; r < n; ++r) {
        std::vector<markets::XccyPillar> up = xccyPillars;
        std::vector<markets::XccyPillar> down = xccyPillars;
        up[r].spread += epsilon;
        down[r].spread -= epsilon;
        const DiscountCurve<double> childUp = bootstrapChild(*rootPtr, foreignForecast, up);
        const DiscountCurve<double> childDown = bootstrapChild(*rootPtr, foreignForecast, down);
        const std::vector<double> plus = flatten(markets::stackQuoteRisk(
            buildInputs(rootPtr, childUp, foreignForecast, domesticForecast,
                        gradientAt(zetaAt(*rootPtr, childUp, foreignForecast, domesticForecast))),
            reference));
        const std::vector<double> minus = flatten(markets::stackQuoteRisk(
            buildInputs(rootPtr, childDown, foreignForecast, domesticForecast,
                        gradientAt(zetaAt(*rootPtr, childDown, foreignForecast, domesticForecast))),
            reference));
        for (std::size_t a = 0; a < dim; ++a) {
            const double fd = (plus[a] - minus[a]) / (2.0 * epsilon);
            maxGammaDeviation =
                std::max(maxGammaDeviation, std::abs(gamma.hessian[a * dim + m + r] - fd));
            util::checkClose("xccy gamma xccy quote FD", gamma.hessian[a * dim + m + r], fd, 1e-6);
        }
    }
    QTA_LOG_INFO("test", "xccy gamma vs delta FD over {} quotes: max deviation {}", m + n,
                 maxGammaDeviation);
}

/// Two children sharing one forecast object keep the shared factor block but
/// emit one `XccyFwd`/`XccyDom` entry each; the per-child residuals are
/// non-zero and sum to the shared block total.
void testXccySharedForecastSplit() {
    const datetime::Date reference(2026, 9, 29);
    const datetime::DayCounter zeroDc(datetime::DayCount::Actual365Fixed);
    const datetime::Calendar calendar = datetime::Calendar::noHolidays();
    const auto buildCurve = [&](double base, double slope) {
        const std::vector<datetime::Date> dates{reference.plusYears(1), reference.plusYears(2),
                                                reference.plusYears(3)};
        std::vector<double> zeros;
        for (const datetime::Date& date : dates) {
            zeros.push_back(base + slope * datetime::yearFraction(reference, date, zeroDc));
        }
        return DiscountCurve<double>(reference, dates, zeroDc, zeros,
                                     InterpolationSpace::LogDiscount, InterpolationScheme::Linear);
    };
    const DiscountCurve<double> rootTarget = buildCurve(0.030, 0.0004);
    std::vector<markets::CurvePillar> rootPillars;
    for (int year = 1; year <= 3; ++year) {
        markets::CurvePillar pillar;
        pillar.maturity = reference.plusYears(year);
        pillar.kind = markets::PillarKind::OisSwap;
        pillar.quoteDayCounter = zeroDc;
        pillar.calendar = calendar;
        pillar.quote = markets::impliedQuote(pillar, reference, rootTarget);
        rootPillars.push_back(pillar);
    }
    const auto rootPtr = std::make_shared<DiscountCurve<double>>(
        markets::bootstrapDiscountCurve(reference, zeroDc, InterpolationSpace::LogDiscount,
                                        InterpolationScheme::Linear, rootPillars));
    const DiscountCurve<double> domesticForecast = buildCurve(0.041, 0.0003);
    // The shared forecast is parented on the root object itself, so its parent
    // chain block is the root already present in the stack.
    const std::vector<double> forecastTimes{0.0, 1.0, 2.0, 3.0};
    const markets::SpreadCurve<double> foreignForecast(
        rootPtr, forecastTimes, std::vector<double>{0.0, 0.0005, 0.0007, 0.0009},
        InterpolationScheme::Linear);
    const DiscountCurve<double> targetA = buildCurve(0.028, 0.0005);
    const DiscountCurve<double> targetB = buildCurve(0.032, 0.0004);
    const auto makePillars = [&](const DiscountCurve<double>& target) {
        std::vector<markets::XccyPillar> pillars;
        for (int year = 1; year <= 3; ++year) {
            markets::XccyPillar pillar;
            pillar.maturity = reference.plusYears(year);
            pillar.foreignTenor = datetime::Period(3, datetime::TimeUnit::Months);
            pillar.domesticTenor = datetime::Period(3, datetime::TimeUnit::Months);
            pillar.foreignCalendar = calendar;
            pillar.domesticCalendar = calendar;
            pillar.foreignDayCounter = datetime::DayCounter(datetime::DayCount::Actual360);
            pillar.domesticDayCounter = datetime::DayCounter(datetime::DayCount::Actual360);
            pillar.spread = markets::impliedXccyBasisSpread(
                target, foreignForecast, *rootPtr, domesticForecast, pillar, reference, zeroDc);
            pillars.push_back(pillar);
        }
        return pillars;
    };
    const std::vector<markets::XccyPillar> pillarsA = makePillars(targetA);
    const std::vector<markets::XccyPillar> pillarsB = makePillars(targetB);
    const DiscountCurve<double> discountA = markets::bootstrapXccyDiscountCurve(
        *rootPtr, domesticForecast, foreignForecast, reference, zeroDc,
        InterpolationSpace::LogDiscount, InterpolationScheme::Linear, pillarsA);
    const DiscountCurve<double> discountB = markets::bootstrapXccyDiscountCurve(
        *rootPtr, domesticForecast, foreignForecast, reference, zeroDc,
        InterpolationSpace::LogDiscount, InterpolationScheme::Linear, pillarsB);

    using ChildT = markets::XccyChildInput<markets::SpreadCurve<double>, DiscountCurve<double>>;
    ChildT childA;
    childA.foreignForecastRole = markets::CurveRole::TenorBasis;
    childA.domesticForecastRole = markets::CurveRole::Forecast;
    childA.foreignDiscount = &discountA;
    childA.pillars = pillarsA;
    childA.dVdForeignZeros.assign(discountA.size(), 0.0);
    childA.dVdForeignZeros[1] = 1.0;
    childA.foreignForecast = &foreignForecast;
    childA.domesticForecast = &domesticForecast;
    childA.dVdForeignForecast = std::vector<double>(foreignForecast.size(), 0.0);
    (*childA.dVdForeignForecast)[1] = 0.25;
    childA.dVdDomesticForecast = std::vector<double>(domesticForecast.size(), 0.0);
    (*childA.dVdDomesticForecast)[1] = 0.10;
    ChildT childB = childA;
    childB.foreignForecastRole = markets::CurveRole::IborOisBasis;
    childB.domesticForecastRole = markets::CurveRole::TenorBasis;
    childB.foreignDiscount = &discountB;
    childB.pillars = pillarsB;
    childB.dVdForeignZeros.assign(discountB.size(), 0.0);
    childB.dVdForeignZeros[2] = 1.0;
    (*childB.dVdForeignForecast)[1] = 0.40;
    (*childB.dVdDomesticForecast)[1] = 0.20;

    std::vector<double> dVdRoot(rootPtr->size(), 0.0);
    dVdRoot[1] = 1.0;
    std::vector<std::variant<ChildT>> children;
    children.emplace_back(childA);
    children.emplace_back(childB);
    const std::vector<markets::StackRiskEntry> entries =
        markets::stackQuoteRiskXccy(*rootPtr, rootPillars, dVdRoot, children, reference);
    // Root, then each child's quote, foreign forecast and domestic forecast.
    CHECK(entries.size() == 7);
    CHECK(entries[0].points.size() == rootPillars.size());
    CHECK(entries[1].points.size() == pillarsA.size());
    CHECK(entries[2].points.size() == foreignForecast.size() - 1);
    CHECK(entries[3].points.size() == domesticForecast.size() - 1);
    CHECK(entries[4].points.size() == pillarsB.size());
    CHECK(entries[5].points.size() == foreignForecast.size() - 1);
    CHECK(entries[6].points.size() == domesticForecast.size() - 1);
    CHECK(entries[2].role == markets::CurveRole::TenorBasis);
    CHECK(entries[5].role == markets::CurveRole::IborOisBasis);
    CHECK(entries[3].role == markets::CurveRole::Forecast);
    CHECK(entries[6].role == markets::CurveRole::TenorBasis);

    // Shared-block reference: one unified input list with a single foreign and
    // domestic factor block holding the summed direct sensitivities.
    const markets::StackCurveView::Ptr rootView = markets::StackCurveView::make(*rootPtr);
    const markets::StackCurveView::Ptr fwdView = markets::StackCurveView::make(foreignForecast);
    const markets::StackCurveView::Ptr domView = markets::StackCurveView::make(domesticForecast);
    std::vector<markets::StackCurveInput> unified(5);
    unified[0].curve = rootView;
    unified[0].role = markets::CurveRole::Discount;
    unified[0].discountPillars = rootPillars;
    unified[0].dVdNodes = dVdRoot;
    for (std::size_t c = 0; c < 2; ++c) {
        const ChildT& child = c == 0 ? childA : childB;
        unified[1 + c].curve = markets::StackCurveView::make(*child.foreignDiscount);
        unified[1 + c].role = child.role;
        unified[1 + c].xccyPillars = child.pillars;
        unified[1 + c].xccy = markets::XccyRowInput{fwdView, domView, rootView};
        unified[1 + c].dVdNodes = child.dVdForeignZeros;
    }
    unified[3].curve = fwdView;
    unified[3].role = markets::CurveRole::Forecast;
    unified[3].factor =
        markets::StackFactorInput{"XccyFwd", markets::forecastNodeDayCounter(foreignForecast)};
    unified[3].dVdNodes.assign(fwdView->size(), 0.0);
    unified[4].curve = domView;
    unified[4].role = markets::CurveRole::Forecast;
    unified[4].factor =
        markets::StackFactorInput{"XccyDom", markets::forecastNodeDayCounter(domesticForecast)};
    unified[4].dVdNodes.assign(domView->size(), 0.0);
    for (std::size_t i = 1; i < childA.dVdForeignForecast->size(); ++i) {
        unified[3].dVdNodes[i] += (*childA.dVdForeignForecast)[i] + (*childB.dVdForeignForecast)[i];
    }
    for (std::size_t i = 1; i < childA.dVdDomesticForecast->size(); ++i) {
        unified[4].dVdNodes[i] +=
            (*childA.dVdDomesticForecast)[i] + (*childB.dVdDomesticForecast)[i];
    }
    const std::vector<markets::StackRiskEntry> shared = markets::stackQuoteRisk(unified, reference);
    CHECK(shared.size() == 5);
    double childAForward = 0.0;
    double childBForward = 0.0;
    double childADomestic = 0.0;
    double childBDomestic = 0.0;
    for (std::size_t k = 0; k < entries[2].points.size(); ++k) {
        childAForward += std::abs(entries[2].points[k].delta);
        childBForward += std::abs(entries[5].points[k].delta);
        util::checkClose("shared forecast split",
                         entries[2].points[k].delta + entries[5].points[k].delta,
                         shared[3].points[k].delta, 1e-10);
    }
    for (std::size_t k = 0; k < entries[3].points.size(); ++k) {
        childADomestic += std::abs(entries[3].points[k].delta);
        childBDomestic += std::abs(entries[6].points[k].delta);
        util::checkClose("shared domestic split",
                         entries[3].points[k].delta + entries[6].points[k].delta,
                         shared[4].points[k].delta, 1e-10);
    }
    CHECK(childAForward > 1e-8);
    CHECK(childBForward > 1e-8);
    CHECK(childADomestic > 1e-8);
    CHECK(childBDomestic > 1e-8);
    QTA_LOG_INFO("test",
                 "shared forecast split: |A| fwd={} dom={}, |B| fwd={} dom={}, per-node sums match "
                 "the shared block",
                 childAForward, childADomestic, childBForward, childBDomestic);
}

/// Three-coupon resetting-notional leg with distinct curves: with the
/// resetting leg's forecast equal to its own discount curve the telescoped
/// value `sum_k adjN_k (D(t_k)(1 + f_k tau_k) - D(t_{k-1}))` is zero, so the
/// implied spread reduces to the opposite constant-notional leg value over the
/// quoted leg's annuity. Golden values from an independent closed-form
/// reference.
void testXccyMtMTelescoping() {
    const datetime::Date reference(2026, 9, 29);
    const datetime::DayCounter zeroDc(datetime::DayCount::Actual365Fixed);
    const datetime::Calendar calendar = datetime::Calendar::noHolidays();
    const std::vector<datetime::Date> dates{reference.plusMonths(3), reference.plusMonths(6),
                                            reference.plusMonths(9)};
    const auto buildCurve = [&](double base, double slope) {
        std::vector<double> zeros;
        for (const datetime::Date& date : dates) {
            zeros.push_back(base + slope * datetime::yearFraction(reference, date, zeroDc));
        }
        return DiscountCurve<double>(reference, dates, zeroDc, zeros,
                                     InterpolationSpace::LogDiscount, InterpolationScheme::Linear);
    };
    const DiscountCurve<double> domesticDiscount = buildCurve(0.040, 0.0005);
    const DiscountCurve<double> domesticForecast = buildCurve(0.043, 0.0004);
    const DiscountCurve<double> foreignTarget = buildCurve(0.030, 0.0006);
    const DiscountCurve<double> foreignForecast = buildCurve(0.025, 0.0008);

    const auto makePillar = [&](bool resetForeign, bool spreadOnForeign) {
        markets::XccyPillar pillar;
        pillar.maturity = dates.back();
        pillar.foreignCalendar = calendar;
        pillar.domesticCalendar = calendar;
        pillar.foreignDayCounter = datetime::DayCounter(datetime::DayCount::Actual360);
        pillar.domesticDayCounter = datetime::DayCounter(datetime::DayCount::Actual360);
        pillar.notional = markets::XccyNotionalMode::MtM;
        pillar.resetForeignLeg = resetForeign;
        pillar.spreadOnForeignLeg = spreadOnForeign;
        return pillar;
    };

    // Foreign leg resets and forecasts on itself.
    for (const bool spreadOnForeign : {true, false}) {
        const double expected = spreadOnForeign ? 0.08254738125917123 : -0.08295639871603087;
        util::checkClose("mtm telescoping foreign reset",
                         markets::impliedXccyBasisSpread(
                             foreignTarget, foreignTarget, domesticDiscount, domesticForecast,
                             makePillar(true, spreadOnForeign), reference, zeroDc),
                         expected, 1e-12);
    }
    // Domestic leg resets and forecasts on itself.
    for (const bool spreadOnForeign : {true, false}) {
        const double expected = spreadOnForeign ? -0.055470884346166476 : 0.055745739340945656;
        util::checkClose("mtm telescoping domestic reset",
                         markets::impliedXccyBasisSpread(
                             foreignTarget, foreignForecast, domesticDiscount, domesticDiscount,
                             makePillar(false, spreadOnForeign), reference, zeroDc),
                         expected, 1e-12);
    }
}

/// Golden spreads of a three-coupon resetting-notional pillar with distinct
/// domestic/foreign curves, covering both reset-leg choices and both
/// `spreadOnForeignLeg` values. Values are produced by an independent
/// reference implementation of the telescoped reset and constant-notional
/// formulas.
void testXccyMtMGoldenSpreads() {
    const datetime::Date reference(2026, 9, 29);
    const datetime::DayCounter zeroDc(datetime::DayCount::Actual365Fixed);
    const datetime::Calendar calendar = datetime::Calendar::noHolidays();
    const std::vector<datetime::Date> dates{reference.plusMonths(3), reference.plusMonths(6),
                                            reference.plusMonths(9)};
    const auto buildCurve = [&](double base, double slope) {
        std::vector<double> zeros;
        for (const datetime::Date& date : dates) {
            zeros.push_back(base + slope * datetime::yearFraction(reference, date, zeroDc));
        }
        return DiscountCurve<double>(reference, dates, zeroDc, zeros,
                                     InterpolationSpace::LogDiscount, InterpolationScheme::Linear);
    };
    const DiscountCurve<double> domesticDiscount = buildCurve(0.040, 0.0005);
    const DiscountCurve<double> domesticForecast = buildCurve(0.043, 0.0004);
    const DiscountCurve<double> foreignTarget = buildCurve(0.030, 0.0006);
    const DiscountCurve<double> foreignForecast = buildCurve(0.025, 0.0008);

    struct GoldenCase {
        bool resetForeign;
        bool spreadOnForeign;
        double spread;
    };
    const std::vector<GoldenCase> cases{
        {true, true, 0.08735367316126888},
        {true, false, -0.087786505514019036},
        {false, true, -0.052562454453813891},
        {false, false, 0.052822898348205888},
    };
    for (const GoldenCase& golden : cases) {
        markets::XccyPillar pillar;
        pillar.maturity = dates.back();
        pillar.foreignCalendar = calendar;
        pillar.domesticCalendar = calendar;
        pillar.foreignDayCounter = datetime::DayCounter(datetime::DayCount::Actual360);
        pillar.domesticDayCounter = datetime::DayCounter(datetime::DayCount::Actual360);
        pillar.notional = markets::XccyNotionalMode::MtM;
        pillar.resetForeignLeg = golden.resetForeign;
        pillar.spreadOnForeignLeg = golden.spreadOnForeign;
        const double spread =
            markets::impliedXccyBasisSpread(foreignTarget, foreignForecast, domesticDiscount,
                                            domesticForecast, pillar, reference, zeroDc);
        util::checkClose("mtm golden spread", spread, golden.spread, 1e-12);
    }
}

/// A bootstrap ladder mixing one resetting-notional and one
/// constant-notional pillar recovers the synthetic foreign curve at both nodes
/// and reprices both quotes.
void testXccyMixedNotionalBootstrap() {
    const datetime::Date reference(2026, 9, 29);
    const datetime::DayCounter zeroDc(datetime::DayCount::Actual365Fixed);
    const datetime::Calendar calendar = datetime::Calendar::noHolidays();
    const auto buildCurve = [&](double base, double slope, int maxYears) {
        std::vector<datetime::Date> pillarDates;
        std::vector<double> zeros;
        for (int years = 1; years <= maxYears; ++years) {
            pillarDates.push_back(reference.plusYears(years));
            zeros.push_back(base +
                            slope * datetime::yearFraction(reference, pillarDates.back(), zeroDc));
        }
        return DiscountCurve<double>(reference, pillarDates, zeroDc, zeros,
                                     InterpolationSpace::LogDiscount, InterpolationScheme::Linear);
    };
    const DiscountCurve<double> domesticDiscount = buildCurve(0.040, 0.0005, 2);
    const DiscountCurve<double> domesticForecast = buildCurve(0.043, 0.0004, 2);
    const DiscountCurve<double> foreignForecast = buildCurve(0.025, 0.0008, 2);
    const DiscountCurve<double> foreignTarget = buildCurve(0.028, 0.0006, 2);

    std::vector<markets::XccyPillar> pillars;
    for (int years = 1; years <= 2; ++years) {
        markets::XccyPillar pillar;
        pillar.maturity = reference.plusYears(years);
        pillar.foreignCalendar = calendar;
        pillar.domesticCalendar = calendar;
        pillar.foreignTenor = datetime::Period(3, datetime::TimeUnit::Months);
        pillar.domesticTenor = datetime::Period(3, datetime::TimeUnit::Months);
        pillar.foreignDayCounter = datetime::DayCounter(datetime::DayCount::Actual360);
        pillar.domesticDayCounter = datetime::DayCounter(datetime::DayCount::Actual360);
        pillar.notional =
            years == 1 ? markets::XccyNotionalMode::MtM : markets::XccyNotionalMode::Const;
        pillar.spread =
            markets::impliedXccyBasisSpread(foreignTarget, foreignForecast, domesticDiscount,
                                            domesticForecast, pillar, reference, zeroDc);
        pillars.push_back(pillar);
    }

    const DiscountCurve<double> curve = markets::bootstrapXccyDiscountCurve(
        domesticDiscount, domesticForecast, foreignForecast, reference, zeroDc,
        InterpolationSpace::LogDiscount, InterpolationScheme::Linear, pillars);
    for (std::size_t i = 0; i < pillars.size(); ++i) {
        util::checkClose("mixed notionals recovered zero", curve.zeros()[i + 1],
                         foreignTarget.zeros()[i + 1], 1e-10);
        util::checkClose("mixed notionals reprice",
                         markets::impliedXccyBasisSpread(curve, foreignForecast, domesticDiscount,
                                                         domesticForecast, pillars[i], reference,
                                                         zeroDc),
                         pillars[i].spread, 1e-10);
    }
}

/// Coupled bootstrap with resetting-notional xccy pillars: the reset enters
/// only the discount-curve residual, so the fixed point must still recover the
/// synthetic discount and spread curves.
void testXccyCoupledMtM() {
    const datetime::Date reference(2026, 9, 29);
    const datetime::DayCounter zeroDc(datetime::DayCount::Actual365Fixed);
    const datetime::Calendar calendar = datetime::Calendar::noHolidays();
    const auto buildCurve = [&](double base, double slope, int maxYears) {
        std::vector<datetime::Date> pillarDates;
        std::vector<double> zeros;
        for (int years = 1; years <= maxYears; ++years) {
            const datetime::Date date = reference.plusYears(years);
            pillarDates.push_back(date);
            zeros.push_back(base + slope * datetime::yearFraction(reference, date, zeroDc));
        }
        return DiscountCurve<double>(reference, pillarDates, zeroDc, zeros,
                                     InterpolationSpace::LogDiscount, InterpolationScheme::Linear);
    };
    const DiscountCurve<double> domesticDiscount = buildCurve(0.040, 0.0005, 5);
    const DiscountCurve<double> domesticForecast = domesticDiscount;
    const DiscountCurve<double> xccyTarget = buildCurve(0.030, 0.0005, 3);
    auto foreignBase = std::make_shared<DiscountCurve<double>>(buildCurve(0.025, 0.0005, 3));
    const std::vector<double> spreadTimes{0.0, 1.0, 2.0, 3.0};
    const std::vector<double> spreadNodes{0.0, 0.0010, 0.0013, 0.0016};
    const markets::SpreadCurve<double> spreadTarget(foreignBase, spreadTimes, spreadNodes,
                                                    InterpolationScheme::Linear);

    std::vector<markets::XccyCoupledPillar> pillars;
    for (int years = 1; years <= 3; ++years) {
        const datetime::Date maturity = reference.plusYears(years);
        markets::XccyCoupledPillar coupled;
        coupled.xccy.maturity = maturity;
        coupled.xccy.foreignCalendar = calendar;
        coupled.xccy.domesticCalendar = calendar;
        coupled.xccy.notional = markets::XccyNotionalMode::MtM;
        coupled.xccy.spread =
            markets::impliedXccyBasisSpread(xccyTarget, spreadTarget, domesticDiscount,
                                            domesticForecast, coupled.xccy, reference, zeroDc);
        coupled.basis.maturity = maturity;
        coupled.basis.calendar = calendar;
        coupled.basis.spread = markets::impliedBasisSpread(spreadTarget, coupled.basis, reference,
                                                           zeroDc, &xccyTarget);
        pillars.push_back(coupled);
    }

    const markets::XccyCoupledResult coupled = markets::bootstrapXccyCoupled(
        domesticDiscount, domesticForecast, foreignBase, reference, zeroDc,
        InterpolationSpace::LogDiscount, InterpolationScheme::Linear, pillars);
    CHECK(coupled.converged);
    CHECK(!coupled.usedJointFallback);
    for (std::size_t i = 0; i < pillars.size(); ++i) {
        util::checkClose("coupled mtm xccy node", coupled.foreignDiscount.zeros()[i + 1],
                         xccyTarget.zeros()[i + 1], 1e-9);
        util::checkClose("coupled mtm xccy reprice",
                         markets::impliedXccyBasisSpread(
                             coupled.foreignDiscount, coupled.foreignSpread, domesticDiscount,
                             domesticForecast, pillars[i].xccy, reference, zeroDc),
                         pillars[i].xccy.spread, 1e-9);
        util::checkClose("coupled mtm basis reprice",
                         markets::impliedBasisSpread(coupled.foreignSpread, pillars[i].basis,
                                                     reference, zeroDc, &coupled.foreignDiscount),
                         pillars[i].basis.spread, 1e-9);
    }
}

} // namespace

int main() {
    testXccyConstNotionalBootstrap();
    testXccySpreadForecastsAndSpreadSide();
    testXccyCoupledBootstrap();
    testForecastNodeDateConversion();
    testXccyJointFallbackAndRisk();
    testXccyAnalyticRowsAgainstFiniteDifference();
    testXccyGammaFiniteDifference();
    testXccySharedForecastSplit();
    testXccyMtMTelescoping();
    testXccyMtMGoldenSpreads();
    testXccyMixedNotionalBootstrap();
    testXccyCoupledMtM();
    QTA_LOG_INFO("test", "test_xccy: ok");
    return 0;
}
