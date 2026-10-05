#include "quantape/log/Log.h"
#include "quantape/markets/Curves/SpreadCurve.h"
#include "quantape/markets/Curves/StackRisk.h"
#include "quantape/markets/Curves/XccyBasisBuilder.h"
#include "quantape/math/LinearAlgebra/DenseSolve.h"
#include "quantape/util/Check.h"

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
                                            domesticDiscount, cp.xccy, reference, zeroDayCounter);
        cp.basis.spread = markets::impliedBasisSpread(spreadTarget, cp.basis, reference,
                                                      zeroDayCounter, &xccyTarget);
        coupledPillars.push_back(cp);
        xccyPillars.push_back(cp.xccy);
    }

    // Joint-LM fallback: one fixed-point pass is not enough, so LM takes over.
    math::FixedPointOptions options;
    options.maxPasses = 1;
    const markets::XccyCoupledResult fallback = markets::bootstrapXccyCoupled(
        domesticDiscount, domesticDiscount, foreignBase, reference, zeroDayCounter,
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
    markets::assembleXccyJacobian(xccyTarget, spreadTarget, domesticDiscount, domesticDiscount,
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
        return markets::impliedXccyBasisSpread(fc, spreadTarget, dc, domesticDiscount,
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
        return markets::bootstrapXccyDiscountCurve(root, domesticDiscount, spreadTarget, reference,
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
    markets::XccyChildInput<markets::SpreadCurve<double>, DiscountCurve<double>> child;
    child.foreignForecastRole = markets::CurveRole::TenorBasis;
    child.domesticForecastRole = markets::CurveRole::IborOisBasis;
    child.foreignDiscount = &fallback.foreignDiscount;
    child.pillars = xccyPillars;
    child.dVdForeignZeros.assign(fallback.foreignDiscount.size(), 0.0);
    child.dVdForeignZeros[1] = 1.0;
    child.foreignForecast.emplace(spreadTarget);
    child.domesticForecast.emplace(domesticDiscount);
    std::vector<std::variant<decltype(child)>> children{child};
    std::vector<double> dVdRoot(domesticDiscount.size(), 0.0);
    dVdRoot[1] = 1.0;
    const std::vector<markets::StackRiskEntry> entries =
        markets::stackQuoteRiskXccy(domesticDiscount, rootPillars, dVdRoot, children, reference);
    CHECK(entries.size() == 4);
    CHECK(entries[0].quoteDeltas.size() == rootPillars.size());
    CHECK(entries[1].quoteDeltas.size() == n);
    CHECK(entries[2].quoteDeltas.size() == 3); // funder: foreign forecast spread nodes
    CHECK(entries[3].quoteDeltas.size() == domesticDiscount.size() - 1);
    for (const markets::StackRiskEntry& entry : entries) {
        for (const double value : entry.quoteDeltas) {
            CHECK(std::isfinite(value));
        }
    }

    // Every xccy path fills per-quote roles, buckets and labels, and the role
    // buckets add back to the full table.
    CHECK(entries[0].quoteRoles.size() == entries[0].quoteDeltas.size());
    CHECK(entries[1].quoteRoles.size() == entries[1].quoteDeltas.size());
    CHECK(entries[2].quoteRoles.size() == entries[2].quoteDeltas.size());
    CHECK(entries[3].quoteRoles.size() == entries[3].quoteDeltas.size());
    for (const markets::StackRiskEntry& entry : entries) {
        CHECK(entry.quoteBuckets.size() == entry.quoteDeltas.size());
        CHECK(entry.quoteLabels.size() == entry.quoteDeltas.size());
    }
    CHECK(entries[0].quoteRoles[0] == markets::CurveRole::Discount);
    CHECK(entries[1].quoteRoles[0] == markets::CurveRole::XccyBasis);
    CHECK(entries[2].quoteRoles[0] == markets::CurveRole::TenorBasis);
    CHECK(entries[3].quoteRoles[0] == markets::CurveRole::IborOisBasis);
    for (const markets::CurveRole role : entries[1].quoteRoles) {
        CHECK(role == markets::CurveRole::XccyBasis);
    }
    for (const markets::CurveRole role : entries[2].quoteRoles) {
        CHECK(role == markets::CurveRole::TenorBasis);
    }
    for (const markets::CurveRole role : entries[3].quoteRoles) {
        CHECK(role == markets::CurveRole::IborOisBasis);
    }
    CHECK(entries[1].quoteBuckets[0] == "1Y");
    CHECK(entries[1].quoteLabels[0] == "Xccy 1Y");
    CHECK(entries[2].quoteBuckets[0] == "1Y");
    CHECK(entries[2].quoteLabels[0] == "XccyFwd 1Y");
    CHECK(entries[3].quoteBuckets[0] == "1Y");
    CHECK(entries[3].quoteLabels[0] == "XccyDom 1Y");
    double tableTotal = 0.0;
    for (const markets::StackRiskEntry& entry : entries) {
        for (const double delta : entry.quoteDeltas) {
            tableTotal += delta;
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
}

} // namespace

int main() {
    testXccyConstNotionalBootstrap();
    testXccySpreadForecastsAndSpreadSide();
    testXccyCoupledBootstrap();
    testForecastNodeDateConversion();
    testXccyJointFallbackAndRisk();
    QTA_LOG_INFO("test", "test_xccy: ok");
    return 0;
}
