#include "quantape/log/Log.h"
#include "quantape/util/Check.h"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <memory>
#include <string>
#include <vector>

#include "paper_eur_curves_fixture.h"

namespace {

using namespace quantape::tests::paper_eur;

namespace dt = quantape::datetime;
namespace mk = quantape::markets;

void testEoniaShortEnd() {
    const mk::DiscountCurve<double> ois = buildOisEcb();
    double minimum = 1.0;
    double minimumTime = 0.0;
    for (double t = 0.25; t < 1.0; t += 1.0 / 365.0) {
        const double forward = ois.forward(t, t + 1.0 / 365.0);
        if (forward < minimum) {
            minimum = forward;
            minimumTime = t;
        }
    }
    const int dayOffset = static_cast<int>(minimumTime * 365.0 + 0.5);
    const dt::Date minimumDate = kSpot.plusDays(dayOffset);
    CHECK(minimum < 0.0);
    CHECK(minimum > -0.001);
    // The negative short-rate section bottoms out in the second week of
    // August 2013.
    CHECK(minimumDate >= dt::Date(2013, 8, 5));
    CHECK(minimumDate <= dt::Date(2013, 8, 20));
    QTA_LOG_INFO("test", "EUR OIS 1D forward minimum {:.2f} bp on {}-{}-{}", minimum * 1e4,
                 minimumDate.year(), minimumDate.month(), minimumDate.dayOfMonth());
}

void testEndogenousVsExogenous() {
    const mk::DiscountCurve<double> ois = buildOisEcb();
    const auto parent = std::make_shared<mk::DiscountCurve<double>>(ois);
    const std::vector<mk::ForecastPillar> pillars = euribor3mPillars();
    const mk::SpreadCurve<double> exogenous = mk::bootstrapForecastCurve(
        parent, &ois, kReference, kZeroDc, mk::InterpolationScheme::Linear, pillars);
    const mk::DiscountCurve<double> endogenous =
        bootstrapEndogenous3m(pillars, mk::InterpolationScheme::Linear);
    double maximum = 0.0;
    double maximumTime = 0.0;
    for (double t = 1.0; t < 30.0; t += 0.25) {
        const double difference =
            simpleThreeMonthForward(exogenous, t) - simpleThreeMonthForward(endogenous, t);
        if (std::abs(difference) > std::abs(maximum)) {
            maximum = difference;
            maximumTime = t;
        }
    }
    // With the negative-rate Eonia path of 2013 the annuity discount effect is
    // larger than the couple of basis points seen with the source's richer
    // short-end instrument set; the difference stays below 5 bp and local.
    CHECK(std::abs(maximum) < 0.0005);
    QTA_LOG_INFO("test", "3M exogenous vs endogenous: max {:.2f} bp at {:.2f}Y", maximum * 1e4,
                 maximumTime);
}

void testInterpolationSchemes() {
    // A single-curve 3M bootstrap on log-discounts, piecewise linear versus
    // monotone cubic and the Hyman-filtered natural cubic spline, read through
    // the smoothness of the 3M FRA curve.
    const std::vector<mk::ForecastPillar> pillars = euribor3mPillars();
    const mk::DiscountCurve<double> linear =
        bootstrapEndogenous3m(pillars, mk::InterpolationScheme::Linear);
    const mk::DiscountCurve<double> monotone =
        bootstrapEndogenous3m(pillars, mk::InterpolationScheme::MonotoneCubic);
    const mk::DiscountCurve<double> hyman =
        bootstrapEndogenous3m(pillars, mk::InterpolationScheme::HymanSpline);
    for (const mk::ForecastPillar& pillar : pillars) {
        quantape::util::checkClose("interpolation linear par",
                                   endogenousParRate(linear, pillar.irs), pillar.irs.quote, 1e-10);
        quantape::util::checkClose("interpolation monotone par",
                                   endogenousParRate(monotone, pillar.irs), pillar.irs.quote,
                                   1e-10);
        quantape::util::checkClose("interpolation hyman par", endogenousParRate(hyman, pillar.irs),
                                   pillar.irs.quote, 1e-10);
    }
    // Smoothness diagnostic on quarterly 3M forwards: the mean absolute
    // quarter-on-quarter change and the maximum jump over the sparse long end
    // (12Y-30Y), where piecewise-constant linear forwards step at pillars. The
    // samples run through the 30Y pillar so its step is included. The Hyman
    // filter keeps the natural spline monotone in the log-discount values, so
    // its forwards vary smoothly between pillars.
    double linearMean = 0.0;
    double monotoneMean = 0.0;
    double hymanMean = 0.0;
    double linearLongEnd = 0.0;
    double monotoneLongEnd = 0.0;
    double hymanLongEnd = 0.0;
    double minimumLinear = 1.0;
    double minimumMonotone = 1.0;
    double minimumHyman = 1.0;
    double previousLinear = 0.0;
    double previousMonotone = 0.0;
    double previousHyman = 0.0;
    std::size_t count = 0;
    for (double t = 1.0; t <= 30.0; t += 0.25) {
        const double linearForward = simpleThreeMonthForward(linear, t);
        const double monotoneForward = simpleThreeMonthForward(monotone, t);
        const double hymanForward = simpleThreeMonthForward(hyman, t);
        minimumLinear = std::min(minimumLinear, linearForward);
        minimumMonotone = std::min(minimumMonotone, monotoneForward);
        minimumHyman = std::min(minimumHyman, hymanForward);
        if (t > 1.0) {
            const double linearJump = std::abs(linearForward - previousLinear);
            const double monotoneJump = std::abs(monotoneForward - previousMonotone);
            const double hymanJump = std::abs(hymanForward - previousHyman);
            linearMean += linearJump;
            monotoneMean += monotoneJump;
            hymanMean += hymanJump;
            if (t > 12.0) {
                linearLongEnd = std::max(linearLongEnd, linearJump);
                monotoneLongEnd = std::max(monotoneLongEnd, monotoneJump);
                hymanLongEnd = std::max(hymanLongEnd, hymanJump);
            }
            ++count;
        }
        previousLinear = linearForward;
        previousMonotone = monotoneForward;
        previousHyman = hymanForward;
    }
    linearMean /= static_cast<double>(count);
    monotoneMean /= static_cast<double>(count);
    hymanMean /= static_cast<double>(count);
    QTA_LOG_INFO("test",
                 "3M interpolation: mean quarterly jump linear {:.2f} bp, monotone {:.2f} "
                 "bp, hyman {:.2f} bp; 12Y-30Y max jump linear {:.2f} bp, monotone {:.2f} "
                 "bp, hyman {:.2f} bp",
                 linearMean * 1e4, monotoneMean * 1e4, hymanMean * 1e4, linearLongEnd * 1e4,
                 monotoneLongEnd * 1e4, hymanLongEnd * 1e4);
    // All schemes reprice and keep forwards positive; the Hyman spline is
    // globally C1 and damps both smoothness statistics below the linear curve.
    CHECK(minimumLinear > 0.0);
    CHECK(minimumMonotone > 0.0);
    CHECK(minimumHyman > 0.0);
    CHECK(hymanMean < linearMean);
    CHECK(hymanLongEnd < linearLongEnd);
}

void testOisSpotCurve() {
    const std::vector<mk::CurvePillar> pillars = oisSpotPillars();
    CHECK(pillars.size() == 33);
    const mk::DiscountCurve<double> ois =
        mk::bootstrapDiscountCurve(kReference, kZeroDc, mk::InterpolationSpace::LogDiscount,
                                   mk::InterpolationScheme::Linear, pillars);
    reprice(pillars, ois, "ois spot reprice");
    const double minimumForward = minimumMonthlyForward(ois, 0.25, 1.0);
    CHECK(minimumForward < 0.0);
    CHECK(minimumForward > -0.001);
    QTA_LOG_INFO("test", "EUR OIS spot: z(1Y) {:.6f}, z(5Y) {:.6f}, min [3M,12M] forward {:.6f}",
                 ois.zero(1.0), ois.zero(5.0), minimumForward);
}

void testOisEcbForwardCurve() {
    const std::vector<mk::CurvePillar> pillars = oisEcbPillars();
    CHECK(pillars.size() == 34);
    const mk::DiscountCurve<double> ois =
        mk::bootstrapDiscountCurve(kReference, kZeroDc, mk::InterpolationSpace::LogDiscount,
                                   mk::InterpolationScheme::Linear, pillars);
    reprice(pillars, ois, "ois ecb reprice");
    // The negative forward-start quotes create negative short forwards in the
    // March-July 2013 window, with discount factors turning non-monotonic.
    const double minimumForward = minimumMonthlyForward(ois, 0.25, 1.0);
    CHECK(minimumForward < 0.0);
    CHECK(minimumForward > -0.001);
    QTA_LOG_INFO("test", "EUR OIS with ECB forwards: min [3M,12M] forward {:.6f}, z(1Y) {:.6f}",
                 minimumForward, ois.zero(1.0));
}

void testEuriborCurves() {
    const std::vector<mk::CurvePillar> oisPillars = oisEcbPillars();
    const mk::DiscountCurve<double> ois =
        mk::bootstrapDiscountCurve(kReference, kZeroDc, mk::InterpolationSpace::LogDiscount,
                                   mk::InterpolationScheme::Linear, oisPillars);
    const auto parent = std::make_shared<mk::DiscountCurve<double>>(ois);

    const std::vector<mk::ForecastPillar> pillars3m = euribor3mPillars();
    const std::vector<mk::ForecastPillar> pillars1m = euribor1mPillars();
    const mk::SpreadCurve<double> curve3m = mk::bootstrapForecastCurve(
        parent, &ois, kReference, kZeroDc, mk::InterpolationScheme::Linear, pillars3m);
    const mk::SpreadCurve<double> curve1m = mk::bootstrapForecastCurve(
        parent, &ois, kReference, kZeroDc, mk::InterpolationScheme::Linear, pillars1m);
    reprice(pillars3m, curve3m, ois, "euribor 3M reprice");
    reprice(pillars1m, curve1m, ois, "euribor 1M reprice");

    // Positive FRA-OIS basis at the 5Y point: 18.4 bp, consistent with the
    // quoted par spread (16.7 bp) plus the forward-section credit slope.
    const double basis5y = curve3m.forward(4.5, 5.0) - ois.forward(4.5, 5.0);
    CHECK(basis5y > 0.0015);
    CHECK(basis5y < 0.0022);

    // Long end: the 30Y-50Y section is upward sloping and stays positive.
    for (double t = 1.0; t < 50.0; t += 1.0) {
        CHECK(curve3m.forward(t, t + 1.0) > 0.0);
    }
    const double forward50y = curve3m.forward(49.0, 50.0);
    CHECK(forward50y > 0.0290);
    CHECK(forward50y < 0.0310);

    // Tenor-dependent credit: the 3M curve sits above the 1M curve at 1Y
    // (quoted par rates 0.141% versus 0.063%).
    const double spread1y = curve3m.forward(0.5, 1.0) - curve1m.forward(0.5, 1.0);
    CHECK(spread1y > 0.0011);
    CHECK(spread1y < 0.0017);

    QTA_LOG_INFO("test", "Euribor 3M: FRA-OIS 5Y {:.2f} bp, 50Y forward {:.6f}", basis5y * 1e4,
                 forward50y);
    QTA_LOG_INFO("test", "Euribor 3M vs 1M 1Y forward spread {:.2f} bp", spread1y * 1e4);
}

/// Short-end-augmented curves: the 1M, 3M and 6M strip each carry the source's
/// selected short-end deposits (and, for 3M, the tomorrow FRA and the eight
/// convexity-corrected futures) before their long IRS pillars; the 12M curve
/// combines the selected deposits, the six 12M FRAs and synthetic 12M IRS
/// quotes from the 6M strip plus the 6M-vs-12M basis.
void testEuriborShortEndCurves() {
    const mk::DiscountCurve<double> ois = buildOisEcb(mk::InterpolationScheme::HymanSpline);
    const auto parent = std::make_shared<mk::DiscountCurve<double>>(ois);

    const std::vector<mk::ForecastPillar> pillars1m = euribor1mSyntheticPillars();
    const std::vector<mk::ForecastPillar> pillars3m = euribor3mSyntheticPillars();
    const std::vector<mk::ForecastPillar> pillars6m = euribor6mPillars();
    const std::vector<mk::ForecastPillar> pillars12m = euribor12mPillars();
    CHECK(pillars1m.size() == 35);
    CHECK(pillars3m.size() == 29);
    CHECK(pillars6m.size() == 60);
    CHECK(pillars12m.size() == 25);

    const mk::SpreadCurve<double> curve1m = mk::bootstrapForecastCurve(
        parent, &ois, kReference, kZeroDc, mk::InterpolationScheme::Linear, pillars1m);
    const mk::SpreadCurve<double> curve3m = mk::bootstrapForecastCurve(
        parent, &ois, kReference, kZeroDc, mk::InterpolationScheme::Linear, pillars3m);
    const mk::SpreadCurve<double> curve6m = mk::bootstrapForecastCurve(
        parent, &ois, kReference, kZeroDc, mk::InterpolationScheme::Linear, pillars6m);
    const mk::SpreadCurve<double> curve12m = mk::bootstrapForecastCurve(
        parent, &ois, kReference, kZeroDc, mk::InterpolationScheme::Linear, pillars12m);
    reprice(pillars1m, curve1m, ois, "euribor 1M short-end reprice");
    reprice(pillars3m, curve3m, ois, "euribor 3M short-end reprice");
    reprice(pillars6m, curve6m, ois, "euribor 6M reprice");
    reprice(pillars12m, curve12m, ois, "euribor 12M reprice");
    QTA_LOG_INFO("test",
                 "Euribor short-end worst quote residuals: 1M {:.2e}, 3M {:.2e}, "
                 "6M {:.2e}, 12M {:.2e}",
                 worstForecastResidual(pillars1m, curve1m, ois),
                 worstForecastResidual(pillars3m, curve3m, ois),
                 worstForecastResidual(pillars6m, curve6m, ois),
                 worstForecastResidual(pillars12m, curve12m, ois));

    // The 1M long end is the 6M strip less the 1M-vs-6M basis: the published
    // selection stores the completed 1M par rates and flat-extrapolates the
    // 30Y basis over 35Y-60Y. Reconstruct every long-end quote from the 6M
    // strip and the basis strip to pin the sign convention.
    const auto sixMonthQuote = [](const dt::Date& maturity) {
        for (const QuoteRow& row : kEuribor6mIrsRows) {
            if (date(row.year, row.month, row.day) == maturity) {
                return row.percent / 100.0;
            }
        }
        throw std::runtime_error("no 6M IRS quote at 1M long-end maturity");
    };
    int longEndCount = 0;
    for (const mk::ForecastPillar& pillar : pillars1m) {
        if (pillar.kind != mk::ForecastPillar::Kind::Irs ||
            mk::forecastPillarQuotedMaturity(pillar) < date(2014, 12, 15)) {
            continue;
        }
        const dt::Date maturity = mk::forecastPillarQuotedMaturity(pillar);
        double basisBp = 16.30;
        for (const BasisQuoteRow& row : kEuribor1m6mBasisRows) {
            if (date(row.year, row.month, row.day) == maturity) {
                basisBp = row.bp;
                break;
            }
        }
        quantape::util::checkClose("1M long-end quote from 6M strip less basis", pillar.irs.quote,
                                   sixMonthQuote(maturity) - basisBp * 1e-4, 1e-9);
        ++longEndCount;
    }
    CHECK(longEndCount == 19);

    // Curve-level sign check at the maturities where the 6M strip itself has a
    // pillar: the 1M par rate is the 6M par rate less the quoted basis.
    const mk::IrsPillar sixMonthAt3y =
        irsPillar(date(2015, 12, 14), 0.424, dt::Period(6, dt::TimeUnit::Months)).irs;
    const mk::IrsPillar sixMonthAt5y =
        irsPillar(date(2017, 12, 13), 0.762, dt::Period(6, dt::TimeUnit::Months)).irs;
    const mk::IrsPillar oneMonthAt3y =
        irsPillar(date(2015, 12, 14), 0.186, dt::Period(1, dt::TimeUnit::Months)).irs;
    const mk::IrsPillar oneMonthAt5y =
        irsPillar(date(2017, 12, 13), 0.512, dt::Period(1, dt::TimeUnit::Months)).irs;
    const double par6m3y = mk::impliedIrsRate(curve6m, ois, sixMonthAt3y, kReference, kZeroDc);
    const double par6m5y = mk::impliedIrsRate(curve6m, ois, sixMonthAt5y, kReference, kZeroDc);
    const double par1m3y = mk::impliedIrsRate(curve1m, ois, oneMonthAt3y, kReference, kZeroDc);
    const double par1m5y = mk::impliedIrsRate(curve1m, ois, oneMonthAt5y, kReference, kZeroDc);
    quantape::util::checkClose("1M 3Y par from 6M less basis", par1m3y, par6m3y - 23.80e-4, 1e-10);
    quantape::util::checkClose("1M 5Y par from 6M less basis", par1m5y, par6m5y - 25.00e-4, 1e-10);
    QTA_LOG_INFO("test",
                 "1M long-end curve: 3Y par {:.2f} bp = 6M {:.2f} bp less 23.80 bp, 5Y par "
                 "{:.2f} bp = 6M {:.2f} bp less 25.00 bp",
                 par1m3y * 1e4, par6m3y * 1e4, par1m5y * 1e4, par6m5y * 1e4);
    // At 2Y the 6M strip is an FRA-implied point, not a swap pillar: the
    // published 2Y 1M quote (9.80 bp) still sits 22.60 bp under the implied 6M
    // par within the interpolation noise of the FRA strip.
    const mk::IrsPillar sixMonthAt2y =
        irsPillar(date(2014, 12, 15), 0.324, dt::Period(6, dt::TimeUnit::Months)).irs;
    const mk::IrsPillar oneMonthAt2y =
        irsPillar(date(2014, 12, 15), 0.098, dt::Period(1, dt::TimeUnit::Months)).irs;
    const double par6m2y = mk::impliedIrsRate(curve6m, ois, sixMonthAt2y, kReference, kZeroDc);
    const double par1m2y = mk::impliedIrsRate(curve1m, ois, oneMonthAt2y, kReference, kZeroDc);
    QTA_LOG_INFO("test", "1M 2Y par {:.2f} bp versus 6M implied {:.2f} bp less 22.60 bp = "
                         "{:.2f} bp",
                 par1m2y * 1e4, par6m2y * 1e4, (par6m2y - 22.60e-4) * 1e4);
    CHECK(std::abs(par1m2y - (par6m2y - 22.60e-4)) < 0.5e-4);

    // The synthetic 12M IRS quotes aggregate the 6M strip and the 6M-vs-12M
    // basis (3Y 0.424% + 17.90 bp, 30Y 2.256% + 6.60 bp), and the money-market
    // pillars win any date collision.
    double quote3y = 0.0;
    double quote30y = 0.0;
    int fra12x24Count = 0;
    int irsAt2yCount = 0;
    int deposit1yCount = 0;
    int irsAt1yCount = 0;
    for (const mk::ForecastPillar& pillar : pillars12m) {
        const dt::Date maturity = mk::forecastPillarQuotedMaturity(pillar);
        if (pillar.kind == mk::ForecastPillar::Kind::Irs && maturity == date(2015, 12, 14)) {
            quote3y = pillar.irs.quote;
        }
        if (pillar.kind == mk::ForecastPillar::Kind::Irs && maturity == date(2042, 12, 15)) {
            quote30y = pillar.irs.quote;
        }
        if (maturity == date(2013, 12, 13)) {
            if (pillar.kind == mk::ForecastPillar::Kind::Deposit) {
                ++deposit1yCount;
                quantape::util::checkClose("12M market deposit quote", pillar.quote, 0.005400,
                                           1e-12);
            }
            if (pillar.kind == mk::ForecastPillar::Kind::Irs) {
                ++irsAt1yCount;
            }
        }
        if (maturity == date(2014, 12, 15)) {
            if (pillar.kind == mk::ForecastPillar::Kind::Fra) {
                ++fra12x24Count;
                quantape::util::checkClose("12M market FRA quote", pillar.quote, 0.005070, 1e-12);
            }
            if (pillar.kind == mk::ForecastPillar::Kind::Irs) {
                ++irsAt2yCount;
            }
        }
    }
    quantape::util::checkClose("12M synthetic 3Y IRS quote", quote3y, 0.006030, 1e-12);
    quantape::util::checkClose("12M synthetic 30Y IRS quote", quote30y, 0.023220, 1e-12);
    CHECK(fra12x24Count == 1);
    CHECK(irsAt2yCount == 0);
    CHECK(deposit1yCount == 1);
    CHECK(irsAt1yCount == 0);

    // Tenor-dependent credit across the first year: the money-market curves
    // are ordered 1M < 3M < 6M < 12M in the 11M-12M forward.
    const double forward1m = curve1m.forward(0.9, 1.0);
    const double forward3m = curve3m.forward(0.9, 1.0);
    const double forward6m = curve6m.forward(0.9, 1.0);
    const double forward12m = curve12m.forward(0.9, 1.0);
    CHECK(forward1m < forward3m);
    CHECK(forward3m < forward6m);
    CHECK(forward6m < forward12m);
    QTA_LOG_INFO("test",
                 "Euribor 11M-12M forwards: 1M {:.2f} bp, 3M {:.2f} bp, 6M {:.2f} bp, "
                 "12M {:.2f} bp",
                 forward1m * 1e4, forward3m * 1e4, forward6m * 1e4, forward12m * 1e4);
}

/// The source's selected short-end instruments: the native futures quote the
/// exchange mid price and carry the listed convexity adjustment as an additive
/// forward adjustment, so the fitted forward is the published FRA-equivalent
/// futures rate.
void testSelectedShortEndInstruments() {
    const mk::DiscountCurve<double> ois = buildOisEcb();
    const auto parent = std::make_shared<mk::DiscountCurve<double>>(ois);
    const mk::SpreadCurve<double> curve3m = mk::bootstrapForecastCurve(
        parent, &ois, kReference, kZeroDc, mk::InterpolationScheme::Linear,
        euribor3mSyntheticPillars());
    const double listedPercent[8] = {0.1775, 0.1274, 0.1222, 0.1269,
                                     0.1565, 0.1961, 0.2556, 0.3101};
    double worstForward = 0.0;
    for (std::size_t i = 0; i < sizeof(kEuribor3mFutureRows) / sizeof(kEuribor3mFutureRows[0]);
         ++i) {
        const FutureQuoteRow& row = kEuribor3mFutureRows[i];
        const mk::ForecastPillar pillar = futurePillar(row);
        CHECK(pillar.kind == mk::ForecastPillar::Kind::Future);
        CHECK(pillar.futureStyle == mk::FutureStyle::Simple);
        quantape::util::checkClose("futures exchange quote", pillar.quote * 100.0,
                                   100.0 - row.midPrice, 1e-12);
        const dt::Date start = date(row.startYear, row.startMonth, row.startDay);
        const dt::Date maturity = date(row.year, row.month, row.day);
        const double t1 = dt::yearFraction(kReference, start, kZeroDc);
        const double t2 = dt::yearFraction(kReference, maturity, kZeroDc);
        const double tau = dt::yearFraction(start, maturity, kAct360);
        const double forward = (curve3m.discount(t1) / curve3m.discount(t2) - 1.0) / tau;
        // The fitted forward is the exchange rate less the listed adjustment,
        // i.e. the FRA-equivalent rate published for the strip.
        quantape::util::checkClose("futures convexity-adjusted forward", forward * 100.0,
                                   listedPercent[i], 1.1e-4);
        worstForward = std::max(worstForward, std::abs(forward * 100.0 - listedPercent[i]));
        if (i == 0) {
            QTA_LOG_INFO("test",
                         "3M futures sign: mid {:.4f}, quote {:.4f}%, convexity {:.4f}%, "
                         "fitted forward {:.4f}%",
                         row.midPrice, pillar.quote * 100.0, pillar.convexityAdjustment * 100.0,
                         forward * 100.0);
        }
    }
    QTA_LOG_INFO("test", "3M futures forward vs published FRA-equivalent: worst {:.2e} bp",
                 worstForward * 100.0);
    // The selected 1M end uses the SND/SWD/2WD/3WD synthetic deposits with
    // the 1MD market quote, not the earlier example table.
    quantape::util::checkClose("1M SND quote", kEuribor1mShortEndRows[0].percent, 0.0661, 1e-12);
    quantape::util::checkClose("1M SWD quote", kEuribor1mShortEndRows[1].percent, 0.0980, 1e-12);
    quantape::util::checkClose("1M 2WD quote", kEuribor1mShortEndRows[2].percent, 0.0993, 1e-12);
    quantape::util::checkClose("1M 3WD quote", kEuribor1mShortEndRows[3].percent, 0.1105, 1e-12);
    quantape::util::checkClose("1M 1MD quote", kEuribor1mShortEndRows[4].percent, 0.1100, 1e-12);
}

/// The year-end overlay leaves the curve untouched before the funding window,
/// raises the overnight forward across it by the full amplitude, and produces
/// the measured one-month and three-month FRA bumps when it wraps the
/// forecast curves.
void testYearEndOverlay() {
    const mk::DiscountCurve<double> ois = buildOisEcb(mk::InterpolationScheme::HymanSpline);
    const auto oisShared = std::make_shared<mk::DiscountCurve<double>>(ois);
    const mk::TurnOverlay<double> onOverlay(oisShared, {},
                                            {{kTurnBeginTime, kTurnEndTime, kTurnBumpAmplitude}});
    const double day = 1.0 / 365.0;
    quantape::util::checkClose("year-end overlay inert before window",
                               onOverlay.discount(kTurnBeginTime - 0.1),
                               ois.discount(kTurnBeginTime - 0.1), 1e-15);
    const double insideStart = kTurnBeginTime + 3.0 * day;
    const double bumpOn = onOverlay.forward(insideStart, insideStart + day) -
                          ois.forward(insideStart, insideStart + day);
    quantape::util::checkClose("year-end overlay overnight jump", bumpOn, 8.5e-4, 1e-9);
    const double afterStart = kTurnEndTime + day;
    quantape::util::checkClose("year-end overlay overnight reversion",
                               onOverlay.forward(afterStart, afterStart + day) -
                                   ois.forward(afterStart, afterStart + day),
                               0.0, 1e-12);

    const auto parent = std::make_shared<mk::DiscountCurve<double>>(ois);
    const mk::SpreadCurve<double> curve1m = mk::bootstrapForecastCurve(
        parent, &ois, kReference, kZeroDc, mk::InterpolationScheme::Linear, euribor1mPillars());
    const mk::SpreadCurve<double> curve3m = mk::bootstrapForecastCurve(
        parent, &ois, kReference, kZeroDc, mk::InterpolationScheme::HymanSpline,
        euribor3mSyntheticPillars());
    const auto curve1mShared = std::make_shared<mk::SpreadCurve<double>>(curve1m);
    const auto curve3mShared = std::make_shared<mk::SpreadCurve<double>>(curve3m);
    const mk::TurnOverlay<double, mk::SpreadCurve<double>> overlay1m(
        curve1mShared, {}, {{kTurnBeginTime, kTurnEndTime, kTurnBumpAmplitude}});
    const mk::TurnOverlay<double, mk::SpreadCurve<double>> overlay3m(
        curve3mShared, {}, {{kTurnBeginTime, kTurnEndTime, kTurnBumpAmplitude}});

    const auto simpleForward = [](const auto& curve, double t1, double t2, double tau) {
        return (curve.discount(t1) / curve.discount(t2) - 1.0) / tau;
    };
    const dt::Date oneMonthStart(2013, 12, 2);
    const dt::Date turnMaturity(2014, 1, 2);
    const dt::Date threeMonthStart(2013, 10, 1);
    const double t1m = dt::yearFraction(kReference, oneMonthStart, kZeroDc);
    const double t1mEnd = dt::yearFraction(kReference, turnMaturity, kZeroDc);
    const double tau1m = dt::yearFraction(oneMonthStart, turnMaturity, kAct360);
    const double t3m = dt::yearFraction(kReference, threeMonthStart, kZeroDc);
    const double t3mEnd = dt::yearFraction(kReference, turnMaturity, kZeroDc);
    const double tau3m = dt::yearFraction(threeMonthStart, turnMaturity, kAct360);
    const double bump1m =
        simpleForward(overlay1m, t1m, t1mEnd, tau1m) - simpleForward(curve1m, t1m, t1mEnd, tau1m);
    const double bump3m =
        simpleForward(overlay3m, t3m, t3mEnd, tau3m) - simpleForward(curve3m, t3m, t3mEnd, tau3m);
    quantape::util::checkClose("year-end overlay one-month jump", bump1m, 1.6e-4, 1e-5);
    quantape::util::checkClose("year-end overlay three-month jump", bump3m, 0.54e-4, 1e-5);
    QTA_LOG_INFO("test", "Year-end overlay bumps: ON {:.2f} bp, 1M {:.2f} bp, 3M {:.2f} bp",
                 bumpOn * 1e4, bump1m * 1e4, bump3m * 1e4);
}

/// Direct turn bootstrap: the funding-window jump is carried by the
/// bootstrap's own curve nodes instead of an overlay. The turn-spanning quote
/// is anchored on the smooth curve outside the window, the window itself adds
/// the measured amplitude at the explicit turn knots, and the crossing
/// instrument closes the section. No `TurnOverlay` is involved.
void testDirectTurnBootstrap() {
    const mk::DiscountCurve<double> ois = buildOisEcb(mk::InterpolationScheme::HymanSpline);
    const auto parent = std::make_shared<mk::DiscountCurve<double>>(ois);
    const mk::SpreadCurve<double> curve1m = mk::bootstrapForecastCurve(
        parent, &ois, kReference, kZeroDc, mk::InterpolationScheme::Linear, euribor1mPillars());
    const mk::SpreadCurve<double> curve3m =
        mk::bootstrapForecastCurve(parent, &ois, kReference, kZeroDc,
                                   mk::InterpolationScheme::Linear, euribor3mSyntheticPillars());

    // The smooth 1M curve now carries the long end from 2Y on, so the post-turn
    // closing FRA merges between the turn knots and the 2Y pillar; the 3M curve
    // keeps its smooth pillars from the crossing future on.
    const dt::Date postTurnEnd(2014, 3, 13);
    const std::vector<mk::ForecastPillar> direct1m = directTurnPillars(
        euribor1mPillars(), curve1m, kTurnOneMonthAmplitude,
        {exactFraPillar(kTurnEnd, postTurnEnd, simpleForwardOver(curve1m, kTurnEnd, postTurnEnd))});
    const std::vector<mk::ForecastPillar> direct3m =
        directTurnPillars(euribor3mSyntheticPillars(), curve3m, kTurnThreeMonthAmplitude, {});
    CHECK(direct1m.size() == 33);
    CHECK(direct3m.size() == 31);

    const mk::SpreadCurve<double> curve1mDirect = mk::bootstrapForecastCurve(
        parent, &ois, kReference, kZeroDc, mk::InterpolationScheme::Linear, direct1m);
    const mk::SpreadCurve<double> curve3mDirect = mk::bootstrapForecastCurve(
        parent, &ois, kReference, kZeroDc, mk::InterpolationScheme::Linear, direct3m);
    reprice(direct1m, curve1mDirect, ois, "euribor 1M direct turn reprice");
    reprice(direct3m, curve3mDirect, ois, "euribor 3M direct turn reprice");

    // The turn-spanning FRA knots reprice to the smooth forward plus the
    // measured window amplitude, and the pre-turn knot leaves the section
    // before the window untouched.
    quantape::util::checkClose("1M direct turn window quote",
                               simpleForwardOver(curve1mDirect, kTurnBegin, kTurnEnd) -
                                   simpleForwardOver(curve1m, kTurnBegin, kTurnEnd),
                               kTurnOneMonthAmplitude, 1e-10);
    quantape::util::checkClose("3M direct turn window quote",
                               simpleForwardOver(curve3mDirect, kTurnBegin, kTurnEnd) -
                                   simpleForwardOver(curve3m, kTurnBegin, kTurnEnd),
                               kTurnThreeMonthAmplitude, 1e-10);
    quantape::util::checkClose("1M direct turn pre-window anchor",
                               simpleForwardOver(curve1mDirect, date(2013, 12, 13), kTurnBegin),
                               simpleForwardOver(curve1m, date(2013, 12, 13), kTurnBegin), 1e-12);
    quantape::util::checkClose("3M direct turn pre-window anchor",
                               simpleForwardOver(curve3mDirect, date(2013, 12, 13), kTurnBegin),
                               simpleForwardOver(curve3m, date(2013, 12, 13), kTurnBegin), 1e-12);

    // Sampled turn-spanning FRA bumps over the sampling grid used by the
    // digitized comparison: the direct curves reproduce the source spikes
    // with no overlay wrapping them.
    const auto sampledBump = [](const auto& direct, const auto& plain, const dt::Date& start,
                                int months) {
        const dt::Date end = dt::Period(months, dt::TimeUnit::Months).advance(start);
        const double t = dt::yearFraction(kReference, start, kZeroDc);
        const double tEnd = dt::yearFraction(kReference, end, kZeroDc);
        const double tau = dt::yearFraction(start, end, kAct360);
        const auto fra = [&](const auto& curve) {
            return (curve.discount(t) / curve.discount(tEnd) - 1.0) / tau;
        };
        return fra(direct) - fra(plain);
    };
    double maximumOneMonthBump = 0.0;
    double maximumThreeMonthBump = 0.0;
    for (dt::Date start(2013, 10, 1); start <= dt::Date(2014, 3, 1);
         start = dt::Period(7, dt::TimeUnit::Days).advance(start)) {
        maximumOneMonthBump =
            std::max(maximumOneMonthBump, sampledBump(curve1mDirect, curve1m, start, 1));
        maximumThreeMonthBump =
            std::max(maximumThreeMonthBump, sampledBump(curve3mDirect, curve3m, start, 3));
    }
    CHECK(maximumOneMonthBump > 0.5e-4);
    CHECK(maximumOneMonthBump < 2.0e-4);
    CHECK(maximumThreeMonthBump > 0.1e-4);
    CHECK(maximumThreeMonthBump < 0.8e-4);
    QTA_LOG_INFO("test", "Direct turn knots: max sampled 1M bump {:.2f} bp, 3M bump {:.2f} bp",
                 maximumOneMonthBump * 1e4, maximumThreeMonthBump * 1e4);

    // Cross-check with the tabulated futures strip: interpolate the no-jump
    // crossing rate between the two non-crossing futures at their reference
    // period midpoints and rescale the crossing future's excess to the funding
    // window.
    const double u3Quote = kTurnCrossCheckFutureRows[0].percent / 100.0;
    const double z3Quote = kTurnCrossCheckFutureRows[1].percent / 100.0;
    const double h4Quote = kTurnCrossCheckFutureRows[2].percent / 100.0;
    const auto midpoint = [](const dt::Date& start, const dt::Date& end) {
        return 0.5 * (dt::yearFraction(kReference, start, kZeroDc) +
                      dt::yearFraction(kReference, end, kZeroDc));
    };
    const double u3Mid = midpoint(
        date(kTurnCrossCheckFutureRows[0].startYear, kTurnCrossCheckFutureRows[0].startMonth,
             kTurnCrossCheckFutureRows[0].startDay),
        date(kTurnCrossCheckFutureRows[0].year, kTurnCrossCheckFutureRows[0].month,
             kTurnCrossCheckFutureRows[0].day));
    const double z3Mid = midpoint(
        date(kTurnCrossCheckFutureRows[1].startYear, kTurnCrossCheckFutureRows[1].startMonth,
             kTurnCrossCheckFutureRows[1].startDay),
        date(kTurnCrossCheckFutureRows[1].year, kTurnCrossCheckFutureRows[1].month,
             kTurnCrossCheckFutureRows[1].day));
    const double h4Mid = midpoint(
        date(kTurnCrossCheckFutureRows[2].startYear, kTurnCrossCheckFutureRows[2].startMonth,
             kTurnCrossCheckFutureRows[2].startDay),
        date(kTurnCrossCheckFutureRows[2].year, kTurnCrossCheckFutureRows[2].month,
             kTurnCrossCheckFutureRows[2].day));
    const double noJump = u3Quote + (h4Quote - u3Quote) * (z3Mid - u3Mid) / (h4Mid - u3Mid);
    const double tauZ3 = dt::yearFraction(
        date(kTurnCrossCheckFutureRows[1].startYear, kTurnCrossCheckFutureRows[1].startMonth,
             kTurnCrossCheckFutureRows[1].startDay),
        date(kTurnCrossCheckFutureRows[1].year, kTurnCrossCheckFutureRows[1].month,
             kTurnCrossCheckFutureRows[1].day),
        kAct360);
    const double tauTurn = dt::yearFraction(kTurnBegin, kTurnEnd, kAct360);
    const double futuresAmplitude = (z3Quote - noJump) * tauZ3 / tauTurn;
    QTA_LOG_INFO("test",
                 "3M futures turn cross-check: no-jump {:.2f} bp, crossing {:.2f} bp, "
                 "rescaled window amplitude {:.2f} bp",
                 noJump * 1e4, z3Quote * 1e4, futuresAmplitude * 1e4);
    // The futures strip disagrees with the digitized FRA bump: the crossing
    // future sits below the level interpolated from its two neighbours, so the
    // rescaled amplitude is negative (-4.72 bp), opposite in sign to the
    // +4.0 bp direct-knot constant measured from the source FRA curves. The
    // futures reading is asserted as a documented out-of-sample cross-check
    // only and is not used to calibrate the turn amplitude.
    CHECK(futuresAmplitude < 0.0);
    quantape::util::checkClose("futures turn cross-check amplitude", futuresAmplitude, -4.72e-4,
                               1e-6);

    // The tabulated crossing 1M instrument (the synthetic 2Y 1M IRS) is also
    // the direct curve's 2Y long-end pillar quote, so its par rate checks that
    // the turn knots leave the long-end repricing intact.
    const mk::ForecastPillar synthetic2y = irsPillar(
        kSynthetic2y1mMaturity, kSynthetic2y1mQuote * 100.0, dt::Period(1, dt::TimeUnit::Months));
    const double model2y =
        mk::impliedIrsRate(curve1mDirect, ois, synthetic2y.irs, kReference, kZeroDc);
    QTA_LOG_INFO("test",
                 "1M crossing 2Y IRS out of sample: quote {:.2f} bp, model {:.2f} bp, "
                 "difference {:+.2f} bp",
                 kSynthetic2y1mQuote * 1e4, model2y * 1e4, (model2y - kSynthetic2y1mQuote) * 1e4);
    // Sub-basis-point agreement of the direct 1M curve with the 2Y pillar.
    CHECK(std::abs(model2y - kSynthetic2y1mQuote) < 0.5e-4);

    // The IMM row's tabulated start crosses the turn. The quoted two-year
    // maturity and the one-year reading of the same quote are both checked,
    // since the market strip pairs the quote with a one-year swap.
    const dt::Date immStart = date(kCrossingImmIrsRow.startYear, kCrossingImmIrsRow.startMonth,
                                   kCrossingImmIrsRow.startDay);
    mk::ForecastPillar crossingIrs =
        irsPillar(date(kCrossingImmIrsRow.year, kCrossingImmIrsRow.month, kCrossingImmIrsRow.day),
                  kCrossingImmIrsRow.percent, dt::Period(3, dt::TimeUnit::Months));
    crossingIrs.irs.start = immStart;
    mk::ForecastPillar crossingIrsOneYear = crossingIrs;
    crossingIrsOneYear.irs.maturity = dt::Period(1, dt::TimeUnit::Years).advance(immStart);
    const double modelImm =
        mk::impliedIrsRate(curve3mDirect, ois, crossingIrs.irs, kReference, kZeroDc);
    const double modelImmOneYear =
        mk::impliedIrsRate(curve3mDirect, ois, crossingIrsOneYear.irs, kReference, kZeroDc);
    QTA_LOG_INFO("test",
                 "3M crossing IMM IRS out of sample: quote {:.2f} bp, two-year model "
                 "{:.2f} bp ({:+.2f}), one-year model {:.2f} bp ({:+.2f})",
                 kCrossingImmIrsRow.percent * 100.0, modelImm * 1e4,
                 (modelImm - kCrossingImmIrsRow.percent / 100.0) * 1e4, modelImmOneYear * 1e4,
                 (modelImmOneYear - kCrossingImmIrsRow.percent / 100.0) * 1e4);
    // The source's curve prices both readings of the crossing quote above the
    // quote; the bounds are deliberately wider than the fitting residuals
    // because this is a genuinely out-of-sample instrument across the turn.
    const double immQuote = kCrossingImmIrsRow.percent / 100.0;
    CHECK(modelImmOneYear > immQuote);
    CHECK(modelImmOneYear - immQuote < 0.0008);
    CHECK(modelImm > immQuote);
    CHECK(modelImm - immQuote < 0.0025);
}

} // namespace

int main() {
    testOisSpotCurve();
    testOisEcbForwardCurve();
    testEoniaShortEnd();
    testEuriborCurves();
    testEuriborShortEndCurves();
    testSelectedShortEndInstruments();
    testEndogenousVsExogenous();
    testInterpolationSchemes();
    testYearEndOverlay();
    testDirectTurnBootstrap();
    if (const char* dumpDirectory = std::getenv("QTA_PAPER_EUR_DUMP")) {
        dumpPaperCurves(dumpDirectory);
    }
    QTA_LOG_INFO("test", "test_paper_eur_curves: ok");
    return 0;
}
