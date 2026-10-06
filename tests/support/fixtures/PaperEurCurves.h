#ifndef QUANTAPE_TESTS_SUPPORT_FIXTURES_PAPER_EUR_CURVES_H
#define QUANTAPE_TESTS_SUPPORT_FIXTURES_PAPER_EUR_CURVES_H

/// Shared EUR curve fixture for the curve replication suite: reference data,
/// bootstrap instrument sets, curve builders, portfolio helpers and the CSV
/// dump. Header-only and double-only, so the curve test and the AD curve-risk
/// test share one source of truth without either depending on the other.

#include "quantape/log/Log.h"
#include "quantape/markets/Curves/CurveBuilder.h"
#include "quantape/markets/Curves/SpreadCurve.h"
#include "quantape/markets/Curves/TurnOverlay.h"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace quantape::tests::paper_eur {

namespace dt = quantape::datetime;
namespace mk = quantape::markets;

inline const dt::Date kReference(2012, 12, 11); // Tuesday
inline const dt::Date kSpot(2012, 12, 13);      // spot = reference + 2 business days
inline const dt::DayCounter kZeroDc(dt::DayCount::Actual365Fixed);
inline const dt::DayCounter kAct360(dt::DayCount::Actual360);
inline const dt::DayCounter kThirty360(dt::DayCount::Thirty360BondBasis);
inline const dt::Calendar kCalendar = dt::Calendar::target();

// Year-end overlay as of 11 Dec 2012. The first two turns in the source are
// end-2012 and end-2013; the end-2012 jump is already carried by the quoted
// short-end pillars, so only the end-2013 turn is overlaid. A single flat
// 8.5 bp forward bump over the 27 Dec 2013 - 5 Jan 2014 funding window
// reproduces the measured FRA bumps: 8.5 bp overnight, 1.6 bp one-month over
// 31/360 and 0.54 bp three-month over 93/360.
inline const dt::Date kTurnBegin(2013, 12, 27);
inline const dt::Date kTurnEnd(2014, 1, 5);
inline const double kTurnBeginTime = dt::yearFraction(kReference, kTurnBegin, kZeroDc);
inline const double kTurnEndTime = dt::yearFraction(kReference, kTurnEnd, kZeroDc);
inline const double kTurnBumpAmplitude = 8.5e-4;

inline const dt::Date kSynthetic2y1mMaturity(2014, 12, 15);
inline const double kSynthetic2y1mQuote = 0.000980;

// Funding-window amplitudes measured from the source FRA curves with the
// crossing-strip estimator: the turn-spanning quotes less the smooth replica
// forward over the same accrual, rescaled to the funding window. The overnight
// strip gives the 8.5 bp carried by the overlay; the one- and three-month
// strips give about 4 bp, so a single overnight-calibrated amplitude
// overstates the longer tenors.
inline const double kTurnOneMonthAmplitude = 4.0e-4;
inline const double kTurnThreeMonthAmplitude = 4.0e-4;

struct QuoteRow {
    int year;
    int month;
    int day;
    double percent;
};

struct ForwardQuoteRow {
    int startYear;
    int startMonth;
    int startDay;
    int year;
    int month;
    int day;
    double percent;
};

struct BasisQuoteRow {
    int year;
    int month;
    int day;
    double bp;
};

/// Exchange-traded future: quoted start/maturity, the exchange mid price
/// (percent of par) and the listed convexity adjustment (percent). The
/// bootstrap pillar quotes `(100 - price)/100` and adds `C` to the fitted
/// forward, so the forward is the FRA-equivalent rate `(100 - price)/100 - C`.
struct FutureQuoteRow {
    int startYear;
    int startMonth;
    int startDay;
    int year;
    int month;
    int day;
    double midPrice;
    double convexityPercent;
};

// Turn-spanning quotes excluded from the smooth strips. The 3M futures strip
// (18 Sep - 18 Dec 2013, 18 Dec 2013 - 18 Mar 2014, 19 Mar - 19 Jun 2014) is
// tabulated: the middle future straddles the turn and the two neighbours give
// the no-jump level to interpolate against. The 1M strip has no tabulated
// crossing pillar because the last market 1M IRS matures 13 Dec 2013; the
// synthetic 2Y 1M IRS (2Y 6M IRS less the 2Y 1M-vs-6M basis, 0.324% - 22.6 bp)
// is used as a cross-check instrument beyond the turn.
inline const ForwardQuoteRow kTurnCrossCheckFutureRows[] = {
    {2013, 9, 18, 2013, 12, 18, 0.1269},
    {2013, 12, 18, 2014, 3, 18, 0.1565},
    {2014, 3, 19, 2014, 6, 19, 0.1961},
};

// IMM 3M IRS whose first quarterly coupon straddles the turn (start 18 Dec
// 2013, maturity 18 Dec 2015, 0.183%), an out-of-sample cross-check for the
// direct 3M curve.
inline const ForwardQuoteRow kCrossingImmIrsRow = {2013, 12, 18, 2015, 12, 18, 0.183};

inline dt::Date date(int year, int month, int day) {
    return dt::Date(year, month, day);
}

inline mk::CurvePillar oisPillar(const dt::Date& start, const dt::Date& maturity, double percent) {
    mk::CurvePillar pillar;
    pillar.kind = mk::PillarKind::OisSwap;
    pillar.start = start;
    pillar.maturity = maturity;
    pillar.quote = percent / 100.0;
    pillar.fixedTenor = dt::Period(1, dt::TimeUnit::Years);
    pillar.calendar = kCalendar;
    pillar.businessDayConvention = dt::BusinessDayConvention::ModifiedFollowing;
    pillar.quoteDayCounter = kThirty360;
    return pillar;
}

inline mk::ForecastPillar irsPillar(const dt::Date& maturity, double percent,
                                    const dt::Period& floatTenor) {
    mk::ForecastPillar pillar;
    pillar.kind = mk::ForecastPillar::Kind::Irs;
    pillar.irs.start = kSpot;
    pillar.irs.maturity = maturity;
    pillar.irs.quote = percent / 100.0;
    pillar.irs.floatTenor = floatTenor;
    pillar.irs.fixedTenor = dt::Period(1, dt::TimeUnit::Years);
    pillar.irs.floatCalendar = kCalendar;
    pillar.irs.fixedCalendar = kCalendar;
    pillar.irs.floatDayCounter = kAct360;
    pillar.irs.fixedDayCounter = kThirty360;
    pillar.irs.businessDayConvention = dt::BusinessDayConvention::ModifiedFollowing;
    return pillar;
}

inline mk::ForecastPillar syntheticDepositPillar(const dt::Date& maturity, double percent) {
    mk::ForecastPillar pillar;
    pillar.kind = mk::ForecastPillar::Kind::Deposit;
    pillar.start = kSpot;
    pillar.maturity = maturity;
    pillar.quote = percent / 100.0;
    pillar.quoteDayCounter = kAct360;
    pillar.calendar = kCalendar;
    pillar.businessDayConvention = dt::BusinessDayConvention::ModifiedFollowing;
    return pillar;
}

/// FRA pillar on the exact quoted dates (no business-day roll), so exchange
/// fixings and funding-window knots stay on the quoted dates even across a
/// weekend.
inline mk::ForecastPillar exactFraPillar(const dt::Date& start, const dt::Date& maturity,
                                         double decimalRate) {
    mk::ForecastPillar pillar;
    pillar.kind = mk::ForecastPillar::Kind::Fra;
    pillar.start = start;
    pillar.maturity = maturity;
    pillar.quote = decimalRate;
    pillar.quoteDayCounter = kAct360;
    pillar.calendar = kCalendar;
    pillar.businessDayConvention = dt::BusinessDayConvention::Unadjusted;
    return pillar;
}

inline mk::ForecastPillar syntheticFraPillar(const dt::Date& start, const dt::Date& maturity,
                                             double percent) {
    return exactFraPillar(start, maturity, percent / 100.0);
}

/// One listed future as a native forecast curve pillar: the exchange mid
/// price is the quoted rate, the listed convexity adjustment is added to the
/// fitted forward, and the exchange start/maturity dates are used unadjusted.
/// The fitted forward is therefore the quoted rate less the convexity
/// adjustment, i.e. the published FRA-equivalent rate.
inline mk::ForecastPillar futurePillar(const FutureQuoteRow& row) {
    mk::ForecastPillar pillar;
    pillar.kind = mk::ForecastPillar::Kind::Future;
    pillar.start = date(row.startYear, row.startMonth, row.startDay);
    pillar.maturity = date(row.year, row.month, row.day);
    pillar.quote = (100.0 - row.midPrice) / 100.0;
    pillar.convexityAdjustment = row.convexityPercent / 100.0;
    pillar.futureStyle = mk::FutureStyle::Simple;
    pillar.quoteDayCounter = kAct360;
    pillar.calendar = kCalendar;
    pillar.businessDayConvention = dt::BusinessDayConvention::Unadjusted;
    return pillar;
}

inline const QuoteRow kOisSpotRows[] = {
    {2012, 12, 20, 0.070}, {2012, 12, 27, 0.069}, {2013, 1, 3, 0.078},   {2013, 1, 14, 0.074},
    {2013, 2, 13, 0.061},  {2013, 3, 13, 0.047},  {2013, 4, 15, 0.033},  {2013, 5, 13, 0.024},
    {2013, 6, 13, 0.018},  {2013, 7, 15, 0.013},  {2013, 8, 13, 0.009},  {2013, 9, 13, 0.005},
    {2013, 10, 14, 0.003}, {2013, 11, 13, 0.001}, {2013, 12, 13, 0.000}, {2014, 3, 13, 0.002},
    {2014, 6, 13, 0.008},  {2014, 9, 15, 0.021},  {2014, 12, 15, 0.036}, {2015, 12, 14, 0.127},
    {2016, 12, 13, 0.274}, {2017, 12, 13, 0.456}, {2018, 12, 13, 0.647}, {2019, 12, 13, 0.827},
    {2020, 12, 14, 0.996}, {2021, 12, 13, 1.147}, {2022, 12, 13, 1.280}, {2023, 12, 13, 1.404},
    {2024, 12, 13, 1.516}, {2027, 12, 13, 1.764}, {2032, 12, 13, 1.939}, {2037, 12, 14, 2.003},
    {2042, 12, 15, 2.038},
};

inline const ForwardQuoteRow kEcbForwardRows[] = {
    {2013, 1, 16, 2013, 2, 13, 0.046},  {2013, 2, 13, 2013, 3, 13, 0.016},
    {2013, 3, 13, 2013, 4, 10, -0.007}, {2013, 4, 10, 2013, 5, 8, -0.013},
    {2013, 5, 8, 2013, 6, 12, -0.014},  {2013, 6, 12, 2013, 7, 10, -0.016},
};

inline std::vector<mk::CurvePillar> oisSpotPillars() {
    std::vector<mk::CurvePillar> pillars;
    pillars.reserve(sizeof(kOisSpotRows) / sizeof(kOisSpotRows[0]));
    for (const QuoteRow& row : kOisSpotRows) {
        pillars.push_back(oisPillar(kSpot, date(row.year, row.month, row.day), row.percent));
    }
    return pillars;
}

inline std::vector<mk::CurvePillar> oisEcbPillars() {
    std::vector<mk::CurvePillar> pillars;
    // Spot OIS up to one month, then the forward-start OIS on ECB dates
    // (including negative quotes), then the spot strip from seven months on.
    for (std::size_t i = 0; i < 4; ++i) {
        const QuoteRow& row = kOisSpotRows[i];
        pillars.push_back(oisPillar(kSpot, date(row.year, row.month, row.day), row.percent));
    }
    for (const ForwardQuoteRow& row : kEcbForwardRows) {
        pillars.push_back(oisPillar(date(row.startYear, row.startMonth, row.startDay),
                                    date(row.year, row.month, row.day), row.percent));
    }
    for (std::size_t i = 9; i < sizeof(kOisSpotRows) / sizeof(kOisSpotRows[0]); ++i) {
        const QuoteRow& row = kOisSpotRows[i];
        pillars.push_back(oisPillar(kSpot, date(row.year, row.month, row.day), row.percent));
    }
    return pillars;
}

inline std::vector<mk::ForecastPillar> euribor3mPillars() {
    const QuoteRow rows[] = {
        {2013, 12, 13, 0.141}, {2014, 3, 13, 0.144},  {2014, 6, 13, 0.153},  {2014, 9, 15, 0.168},
        {2014, 12, 15, 0.186}, {2015, 12, 14, 0.285}, {2016, 12, 13, 0.437}, {2017, 12, 13, 0.623},
        {2018, 12, 13, 0.817}, {2019, 12, 13, 1.000}, {2020, 12, 14, 1.171}, {2021, 12, 13, 1.324},
        {2022, 12, 13, 1.459}, {2023, 12, 13, 1.582}, {2024, 12, 13, 1.692}, {2027, 12, 13, 1.933},
        {2032, 12, 13, 2.099}, {2037, 12, 14, 2.156}, {2042, 12, 15, 2.186}, {2052, 12, 13, 2.288},
        {2062, 12, 13, 2.367},
    };
    std::vector<mk::ForecastPillar> pillars;
    pillars.reserve(sizeof(rows) / sizeof(rows[0]));
    for (const QuoteRow& row : rows) {
        pillars.push_back(irsPillar(date(row.year, row.month, row.day), row.percent,
                                    dt::Period(3, dt::TimeUnit::Months)));
    }
    return pillars;
}

// 1M-vs-6M Euribor IRBS mids by maturity: maturity date and mid spread in
// basis points. The spread is deducted from the 6M strip (the 1M leg pays the
// lower rate) to complete the 1M long end.
inline const BasisQuoteRow kEuribor1m6mBasisRows[] = {
    {2013, 12, 13, 22.20}, {2014, 12, 15, 22.60}, {2015, 12, 14, 23.80}, {2016, 12, 13, 24.60},
    {2017, 12, 13, 25.00}, {2018, 12, 13, 25.00}, {2019, 12, 13, 24.80}, {2020, 12, 14, 24.50},
    {2021, 12, 13, 24.10}, {2022, 12, 13, 23.70}, {2023, 12, 13, 23.30}, {2024, 12, 13, 22.80},
    {2027, 12, 13, 21.10}, {2032, 12, 13, 18.90}, {2037, 12, 14, 17.50}, {2042, 12, 15, 16.30},
};

// Published 1M long-end selection from 2Y to 60Y: the completed 1M par rates
// obtained from the 6M IRS strip less the 1M-vs-6M basis above, with the 30Y
// basis flat-extrapolated over 35Y-60Y as selected in the source.
inline const QuoteRow kEuribor1mLongEndRows[] = {
    {2014, 12, 15, 0.0980}, {2015, 12, 14, 0.1860}, {2016, 12, 13, 0.3300}, {2017, 12, 13, 0.5120},
    {2018, 12, 13, 0.7040}, {2019, 12, 13, 0.8870}, {2020, 12, 14, 1.0580}, {2021, 12, 13, 1.2110},
    {2022, 12, 13, 1.3470}, {2023, 12, 13, 1.4700}, {2024, 12, 13, 1.5810}, {2027, 12, 13, 1.8260},
    {2032, 12, 13, 1.9980}, {2037, 12, 14, 2.0590}, {2042, 12, 15, 2.0930}, {2047, 12, 13, 2.1320},
    {2052, 12, 13, 2.1850}, {2062, 12, 13, 2.2580}, {2072, 12, 13, 2.3000},
};

/// 1M long end from 2Y on: the published completed 1M par rates as 1M IRS.
inline std::vector<mk::ForecastPillar> euribor1mLongEndPillars() {
    std::vector<mk::ForecastPillar> pillars;
    pillars.reserve(sizeof(kEuribor1mLongEndRows) / sizeof(kEuribor1mLongEndRows[0]));
    for (const QuoteRow& row : kEuribor1mLongEndRows) {
        pillars.push_back(irsPillar(date(row.year, row.month, row.day), row.percent,
                                    dt::Period(1, dt::TimeUnit::Months)));
    }
    return pillars;
}

inline std::vector<mk::ForecastPillar> euribor1mPillars() {
    const QuoteRow rows[] = {
        {2013, 2, 13, 0.106},  {2013, 3, 13, 0.096},  {2013, 4, 15, 0.085},  {2013, 5, 13, 0.079},
        {2013, 6, 13, 0.075},  {2013, 7, 15, 0.071},  {2013, 8, 13, 0.069},  {2013, 9, 13, 0.066},
        {2013, 10, 14, 0.065}, {2013, 11, 13, 0.064}, {2013, 12, 13, 0.063},
    };
    std::vector<mk::ForecastPillar> pillars;
    pillars.reserve(sizeof(rows) / sizeof(rows[0]) +
                    sizeof(kEuribor1mLongEndRows) / sizeof(kEuribor1mLongEndRows[0]));
    for (const QuoteRow& row : rows) {
        pillars.push_back(irsPillar(date(row.year, row.month, row.day), row.percent,
                                    dt::Period(1, dt::TimeUnit::Months)));
    }
    for (const mk::ForecastPillar& pillar : euribor1mLongEndPillars()) {
        pillars.push_back(pillar);
    }
    return pillars;
}

// 1M short end selected in the source: four synthetic deposits (SND to 3WD)
// and the market 1M deposit, all spot-starting.
inline const QuoteRow kEuribor1mShortEndRows[] = {
    {2012, 12, 14, 0.0661}, {2012, 12, 20, 0.0980}, {2012, 12, 27, 0.0993},
    {2013, 1, 3, 0.1105},   {2013, 1, 14, 0.1100},
};

// 3M short end selected in the source: four synthetic deposits (2WD to 2MD)
// plus one tomorrow FRA.
inline const QuoteRow kEuribor3mShortEndRows[] = {
    {2012, 12, 27, 0.1865},
    {2013, 1, 3, 0.1969},
    {2013, 1, 14, 0.1951},
    {2013, 2, 13, 0.1874},
};

inline const ForwardQuoteRow kEuribor3mFraRows[] = {
    {2012, 12, 14, 2013, 3, 14, 0.1790},
};

// Eight quarterly futures selected in the source, with the exchange mid
// prices and the convexity adjustments listed with the futures strip.
inline const FutureQuoteRow kEuribor3mFutureRows[] = {
    {2012, 12, 19, 2013, 3, 19, 99.8225, 0.0000}, {2013, 3, 20, 2013, 6, 20, 99.8725, 0.0001},
    {2013, 6, 19, 2013, 9, 19, 99.8775, 0.0003},  {2013, 9, 18, 2013, 12, 18, 99.8725, 0.0006},
    {2013, 12, 18, 2014, 3, 18, 99.8425, 0.0009}, {2014, 3, 19, 2014, 6, 19, 99.8025, 0.0013},
    {2014, 6, 18, 2014, 9, 18, 99.7425, 0.0018},  {2014, 9, 17, 2014, 12, 17, 99.6875, 0.0024},
};

// 6M short end selected in the source: nine synthetic deposits (SND to 5MD).
inline const QuoteRow kEuribor6mShortEndRows[] = {
    {2012, 12, 14, 0.3565}, {2012, 12, 20, 0.3858}, {2012, 12, 27, 0.3840},
    {2013, 1, 3, 0.3922},   {2013, 1, 14, 0.3869},  {2013, 2, 13, 0.3698},
    {2013, 3, 13, 0.3527},  {2013, 4, 15, 0.3342},  {2013, 5, 13, 0.3225},
};

// Nineteen market 6M FRAs from tomorrow to 18x24, carrying the structure to
// Dec 2014.
inline const ForwardQuoteRow kEuribor6mFraRows[] = {
    {2012, 12, 14, 2013, 6, 14, 0.3120}, {2013, 1, 14, 2013, 7, 15, 0.2930},
    {2013, 2, 13, 2013, 8, 13, 0.2720},  {2013, 3, 13, 2013, 9, 13, 0.2600},
    {2013, 4, 15, 2013, 10, 15, 0.2560}, {2013, 5, 13, 2013, 11, 13, 0.2520},
    {2013, 6, 13, 2013, 12, 13, 0.2480}, {2013, 7, 15, 2014, 1, 15, 0.2540},
    {2013, 8, 13, 2014, 2, 13, 0.2610},  {2013, 9, 13, 2014, 3, 13, 0.2670},
    {2013, 10, 14, 2014, 4, 14, 0.2790}, {2013, 11, 13, 2014, 5, 13, 0.2910},
    {2013, 12, 13, 2014, 6, 13, 0.3030}, {2014, 1, 13, 2014, 7, 14, 0.3180},
    {2014, 2, 13, 2014, 8, 13, 0.3350},  {2014, 3, 13, 2014, 9, 15, 0.3520},
    {2014, 4, 14, 2014, 10, 14, 0.3710}, {2014, 5, 13, 2014, 11, 13, 0.3890},
    {2014, 6, 13, 2014, 12, 15, 0.4090},
};

// Par IRS strip on 6M Euribor: annual fixed leg, semi-annual floating leg.
inline const QuoteRow kEuribor6mIrsRows[] = {
    {2013, 12, 13, 0.286}, {2014, 3, 13, 0.250},  {2014, 6, 13, 0.293},  {2014, 9, 15, 0.282},
    {2014, 12, 15, 0.324}, {2015, 12, 14, 0.424}, {2016, 12, 13, 0.576}, {2017, 12, 13, 0.762},
    {2018, 12, 13, 0.954}, {2019, 12, 13, 1.135}, {2020, 12, 14, 1.303}, {2021, 12, 13, 1.452},
    {2022, 12, 13, 1.584}, {2023, 12, 13, 1.703}, {2024, 12, 13, 1.809}, {2025, 12, 15, 1.901},
    {2026, 12, 14, 1.976}, {2027, 12, 13, 2.037}, {2028, 12, 13, 2.086}, {2029, 12, 13, 2.123},
    {2030, 12, 13, 2.150}, {2031, 12, 15, 2.171}, {2032, 12, 13, 2.187}, {2033, 12, 13, 2.200},
    {2034, 12, 13, 2.211}, {2035, 12, 13, 2.220}, {2036, 12, 15, 2.228}, {2037, 12, 14, 2.234},
    {2038, 12, 13, 2.239}, {2039, 12, 13, 2.243}, {2040, 12, 13, 2.247}, {2041, 12, 13, 2.251},
    {2042, 12, 15, 2.256}, {2047, 12, 13, 2.295}, {2052, 12, 13, 2.348}, {2062, 12, 13, 2.421},
    {2072, 12, 13, 2.463},
};

// 6M-vs-12M Euribor IRBS basis: maturity date and mid spread in basis points.
inline const BasisQuoteRow kEuribor6m12mBasisRows[] = {
    {2013, 12, 13, 26.20}, {2014, 12, 15, 20.70}, {2015, 12, 14, 17.90}, {2016, 12, 13, 16.40},
    {2017, 12, 13, 15.10}, {2018, 12, 13, 13.90}, {2019, 12, 13, 13.00}, {2020, 12, 14, 12.30},
    {2021, 12, 13, 11.80}, {2022, 12, 13, 11.30}, {2023, 12, 13, 10.90}, {2024, 12, 13, 10.60},
    {2027, 12, 13, 9.30},  {2032, 12, 13, 8.00},  {2037, 12, 14, 7.20},  {2042, 12, 15, 6.60},
};

// 12M short end selected in the source: four synthetic deposits (1M to 9M)
// and the market 12M deposit.
inline const QuoteRow kEuribor12mDepositRows[] = {
    {2013, 1, 14, 0.6537}, {2013, 3, 13, 0.6187},  {2013, 6, 13, 0.5772},
    {2013, 9, 13, 0.5563}, {2013, 12, 13, 0.5400},
};

// Six 12M FRAs from 3x15 to 18x30; only 12x24 is a market quote, the other
// five are synthetic.
inline const ForwardQuoteRow kEuribor12mFraRows[] = {
    {2013, 3, 13, 2014, 3, 13, 0.4974}, {2013, 6, 13, 2014, 6, 13, 0.4783},
    {2013, 9, 13, 2014, 9, 15, 0.4822}, {2013, 12, 13, 2014, 12, 15, 0.5070},
    {2014, 3, 13, 2015, 3, 13, 0.5481}, {2014, 6, 13, 2015, 6, 15, 0.6025},
};

inline std::vector<mk::ForecastPillar> euribor6mIrsPillars() {
    std::vector<mk::ForecastPillar> pillars;
    pillars.reserve(sizeof(kEuribor6mIrsRows) / sizeof(kEuribor6mIrsRows[0]));
    for (const QuoteRow& row : kEuribor6mIrsRows) {
        pillars.push_back(irsPillar(date(row.year, row.month, row.day), row.percent,
                                    dt::Period(6, dt::TimeUnit::Months)));
    }
    return pillars;
}

/// 6M Euribor curve: nine synthetic 6M deposits, the nineteen market 6M FRAs
/// and the medium/long 6M IRS strip from 3Y on.
inline std::vector<mk::ForecastPillar> euribor6mPillars() {
    std::vector<mk::ForecastPillar> pillars;
    for (const QuoteRow& row : kEuribor6mShortEndRows) {
        pillars.push_back(syntheticDepositPillar(date(row.year, row.month, row.day), row.percent));
    }
    for (const ForwardQuoteRow& row : kEuribor6mFraRows) {
        pillars.push_back(exactFraPillar(date(row.startYear, row.startMonth, row.startDay),
                                         date(row.year, row.month, row.day), row.percent / 100.0));
    }
    for (const mk::ForecastPillar& pillar : euribor6mIrsPillars()) {
        if (mk::forecastPillarQuotedMaturity(pillar) >= date(2015, 12, 14)) {
            pillars.push_back(pillar);
        }
    }
    return pillars;
}

/// 1M Euribor curve: the selected 1M short-end deposits, then the market 1M
/// IRS strip with its 2Y-60Y long end.
inline std::vector<mk::ForecastPillar> euribor1mSyntheticPillars() {
    std::vector<mk::ForecastPillar> pillars;
    for (const QuoteRow& row : kEuribor1mShortEndRows) {
        pillars.push_back(syntheticDepositPillar(date(row.year, row.month, row.day), row.percent));
    }
    for (const mk::ForecastPillar& pillar : euribor1mPillars()) {
        pillars.push_back(pillar);
    }
    return pillars;
}

/// 3M Euribor curve: the selected 3M short-end deposits, the tomorrow FRA and
/// the eight convexity-corrected native futures, then the 3M IRS strip from
/// 3Y on.
inline std::vector<mk::ForecastPillar> euribor3mSyntheticPillars() {
    std::vector<mk::ForecastPillar> pillars;
    for (const QuoteRow& row : kEuribor3mShortEndRows) {
        pillars.push_back(syntheticDepositPillar(date(row.year, row.month, row.day), row.percent));
    }
    for (const ForwardQuoteRow& row : kEuribor3mFraRows) {
        pillars.push_back(exactFraPillar(date(row.startYear, row.startMonth, row.startDay),
                                         date(row.year, row.month, row.day), row.percent / 100.0));
    }
    for (const FutureQuoteRow& row : kEuribor3mFutureRows) {
        pillars.push_back(futurePillar(row));
    }
    for (const mk::ForecastPillar& pillar : euribor3mPillars()) {
        if (mk::forecastPillarQuotedMaturity(pillar) >= date(2015, 12, 14)) {
            pillars.push_back(pillar);
        }
    }
    return pillars;
}

/// 12M Euribor curve: the selected 12M deposits and FRAs, then synthetic 12M
/// IRS quotes built from the 6M IRS strip plus the 6M-vs-12M basis from 3Y on.
/// Where a synthetic IRS maturity coincides with a deposit or FRA maturity the
/// money-market pillar stays, so the quoted dates are unique.
inline std::vector<mk::ForecastPillar> euribor12mPillars() {
    std::vector<mk::ForecastPillar> pillars;
    for (const QuoteRow& row : kEuribor12mDepositRows) {
        pillars.push_back(syntheticDepositPillar(date(row.year, row.month, row.day), row.percent));
    }
    for (const ForwardQuoteRow& row : kEuribor12mFraRows) {
        pillars.push_back(exactFraPillar(date(row.startYear, row.startMonth, row.startDay),
                                         date(row.year, row.month, row.day), row.percent / 100.0));
    }
    for (const mk::ForecastPillar& base : euribor6mIrsPillars()) {
        if (mk::forecastPillarQuotedMaturity(base) < date(2015, 12, 14)) {
            continue;
        }
        for (const BasisQuoteRow& basis : kEuribor6m12mBasisRows) {
            if (base.irs.maturity != date(basis.year, basis.month, basis.day)) {
                continue;
            }
            mk::ForecastPillar out;
            out.kind = mk::ForecastPillar::Kind::Irs;
            out.irs = base.irs;
            out.irs.quote = base.irs.quote + basis.bp * 1e-4;
            out.irs.floatTenor = dt::Period(12, dt::TimeUnit::Months);
            pillars.push_back(out);
            break;
        }
    }
    std::stable_sort(pillars.begin(), pillars.end(),
                     [](const mk::ForecastPillar& left, const mk::ForecastPillar& right) {
                         return mk::forecastPillarQuotedMaturity(left) <
                                mk::forecastPillarQuotedMaturity(right);
                     });
    pillars.erase(std::unique(pillars.begin(), pillars.end(),
                              [](const mk::ForecastPillar& left, const mk::ForecastPillar& right) {
                                  return mk::forecastPillarQuotedMaturity(left) ==
                                         mk::forecastPillarQuotedMaturity(right);
                              }),
                  pillars.end());
    return pillars;
}

/// Node date the forecast bootstrap assigns to a pillar, matching the
/// calendar roll applied inside `bootstrapForecastCurve`.
inline dt::Date adjustedForecastMaturity(const mk::ForecastPillar& pillar) {
    switch (pillar.kind) {
        case mk::ForecastPillar::Kind::Irs:
            return pillar.irs.fixedCalendar.adjust(pillar.irs.maturity,
                                                   pillar.irs.businessDayConvention);
        case mk::ForecastPillar::Kind::Deposit:
        case mk::ForecastPillar::Kind::Fra:
            return pillar.calendar.adjust(pillar.maturity, pillar.businessDayConvention);
        case mk::ForecastPillar::Kind::Future:
            return pillar.maturity;
        case mk::ForecastPillar::Kind::BasisSwap:
            return pillar.basis.calendar.adjust(pillar.basis.maturity,
                                                pillar.basis.businessDayConvention);
    }
    return pillar.maturity;
}

template <typename ParentT>
inline double simpleForwardOver(const mk::SpreadCurve<double, ParentT>& curve,
                                const dt::Date& start, const dt::Date& end) {
    const double t1 = dt::yearFraction(kReference, start, kZeroDc);
    const double t2 = dt::yearFraction(kReference, end, kZeroDc);
    const double tau = dt::yearFraction(start, end, kAct360);
    return (curve.discount(t1) / curve.discount(t2) - 1.0) / tau;
}

/// Direct turn bootstrap instrument set: the smooth pillars with the interval
/// that contains the turn split by two explicit FRA knots at the funding-window
/// boundaries. The pre-turn knot is anchored on the smooth curve's own forward;
/// the turn-spanning knot carries the measured funding-window amplitude on top
/// of the same forward, so the bootstrap itself produces the jump. The first
/// smooth pillar beyond the window (or the closing instruments) resolves the
/// rest of the section.
template <typename ParentT>
inline std::vector<mk::ForecastPillar>
directTurnPillars(const std::vector<mk::ForecastPillar>& smoothPillars,
                  const mk::SpreadCurve<double, ParentT>& smooth, double turnAmplitude,
                  const std::vector<mk::ForecastPillar>& closingPillars) {
    std::size_t split = 0;
    while (split < smoothPillars.size() &&
           adjustedForecastMaturity(smoothPillars[split]) < kTurnBegin) {
        ++split;
    }
    if (split == 0) {
        throw std::runtime_error("directTurnPillars: no pillar before the turn");
    }
    const dt::Date anchor = adjustedForecastMaturity(smoothPillars[split - 1]);
    const double tAnchor = dt::yearFraction(kReference, anchor, kZeroDc);
    const double tBegin = dt::yearFraction(kReference, kTurnBegin, kZeroDc);
    const double tEnd = dt::yearFraction(kReference, kTurnEnd, kZeroDc);
    const double preTau = dt::yearFraction(anchor, kTurnBegin, kAct360);
    const double turnTau = dt::yearFraction(kTurnBegin, kTurnEnd, kAct360);
    const double preQuote = (smooth.discount(tAnchor) / smooth.discount(tBegin) - 1.0) / preTau;
    const double turnQuote =
        (smooth.discount(tBegin) / smooth.discount(tEnd) - 1.0) / turnTau + turnAmplitude;
    std::vector<mk::ForecastPillar> out(smoothPillars.begin(), smoothPillars.begin() + split);
    out.push_back(exactFraPillar(anchor, kTurnBegin, preQuote));
    out.push_back(exactFraPillar(kTurnBegin, kTurnEnd, turnQuote));
    out.insert(out.end(), smoothPillars.begin() + split, smoothPillars.end());
    out.insert(out.end(), closingPillars.begin(), closingPillars.end());
    // Closing instruments may fall inside a long smooth strip; the bootstrap
    // needs strictly increasing node dates, so merge by maturity.
    std::stable_sort(out.begin(), out.end(),
                     [](const mk::ForecastPillar& left, const mk::ForecastPillar& right) {
                         return adjustedForecastMaturity(left) < adjustedForecastMaturity(right);
                     });
    return out;
}

inline double worstForecastResidual(const std::vector<mk::ForecastPillar>& pillars,
                                    const mk::SpreadCurve<double>& curve,
                                    const mk::DiscountCurve<double>& discount) {
    double worst = 0.0;
    for (const mk::ForecastPillar& pillar : pillars) {
        worst = std::max(
            worst, std::abs(mk::impliedForecastQuote(curve, discount, pillar, kReference, kZeroDc) -
                            forecastPillarTarget(pillar)));
    }
    return worst;
}

inline double minimumMonthlyForward(const mk::DiscountCurve<double>& curve, double begin,
                                    double end) {
    double minimum = 1.0;
    const double step = 1.0 / 12.0;
    for (double t = begin; t < end; t += step) {
        minimum = std::min(minimum, curve.forward(t, t + step));
    }
    return minimum;
}

/// Worst absolute reprice error over all pillars; the caller owns the
/// assertion so the helper never invokes the retired fatal test harness.
inline double worstRepriceError(const std::vector<mk::CurvePillar>& pillars,
                                const mk::DiscountCurve<double>& curve) {
    double worst = 0.0;
    for (const mk::CurvePillar& pillar : pillars) {
        worst =
            std::max(worst, std::abs(mk::impliedQuote(pillar, kReference, curve) - pillar.quote));
    }
    return worst;
}

inline double worstRepriceError(const std::vector<mk::ForecastPillar>& pillars,
                                const mk::SpreadCurve<double>& curve,
                                const mk::DiscountCurve<double>& discount) {
    double worst = 0.0;
    for (const mk::ForecastPillar& pillar : pillars) {
        worst = std::max(
            worst, std::abs(mk::impliedForecastQuote(curve, discount, pillar, kReference, kZeroDc) -
                            forecastPillarTarget(pillar)));
    }
    return worst;
}

template <typename CurveT>
inline double simpleThreeMonthForward(const CurveT& curve, double t) {
    constexpr double tau = 0.25;
    return (curve.discount(t) / curve.discount(t + tau) - 1.0) / tau;
}

/// Classical single-curve (endogenous) bootstrap of the 3M strip: each pillar
/// solves `R A = D(spot) - D(T_i)` with every discount factor, including the
/// fixed annuity, taken from the curve being built.
inline mk::DiscountCurve<double>
bootstrapEndogenous3m(const std::vector<mk::ForecastPillar>& pillars,
                      mk::InterpolationScheme scheme) {
    const std::size_t count = pillars.size();
    std::vector<double> times(count, 0.0);
    for (std::size_t i = 0; i < count; ++i) {
        const mk::IrsPillar& irs = pillars[i].irs;
        times[i] = dt::yearFraction(
            kReference, irs.fixedCalendar.adjust(irs.maturity, irs.businessDayConvention), kZeroDc);
    }
    const quantape::math::BrentSolver<double> solver;
    std::vector<double> nodeTimes{0.0};
    for (std::size_t j = 0; j < count; ++j) {
        nodeTimes.push_back(times[j]);
    }
    std::vector<double> zeros(count, 0.0);
    for (int pass = 0; pass < 50; ++pass) {
        const std::vector<double> previous = zeros;
        double lastMove = 0.0;
        for (std::size_t i = 0; i < count; ++i) {
            const mk::IrsPillar& irs = pillars[i].irs;
            const dt::Schedule fixed(irs.start, irs.maturity, irs.fixedTenor, irs.fixedCalendar,
                                     irs.businessDayConvention, dt::DateGeneration::Forward, false,
                                     dt::BusinessDayConvention::Unadjusted);
            const auto residual = [&](double trial) {
                std::vector<double> trialZeros{0.0};
                for (std::size_t j = 0; j < count; ++j) {
                    trialZeros.push_back(j == i ? trial : zeros[j]);
                }
                const mk::DiscountCurve<double> curve(nodeTimes, trialZeros,
                                                      mk::InterpolationSpace::LogDiscount, scheme);
                double annuity = 0.0;
                for (std::size_t k = 1; k < fixed.dates().size(); ++k) {
                    const double tau = dt::yearFraction(fixed.dates()[k - 1], fixed.dates()[k],
                                                        irs.fixedDayCounter);
                    const dt::Date payDate = irs.fixedCalendar.advance(
                        fixed.dates()[k], dt::Period(irs.paymentLag, dt::TimeUnit::Days),
                        irs.businessDayConvention);
                    annuity += tau * curve.discount(dt::yearFraction(kReference, payDate, kZeroDc));
                }
                const double startDiscount =
                    curve.discount(dt::yearFraction(kReference, irs.start, kZeroDc));
                const double endDiscount = curve.discount(times[i]);
                return (startDiscount - endDiscount) / annuity - irs.quote;
            };
            const double guess = pass == 0 ? (i == 0 ? 0.0 : zeros[i - 1]) : previous[i];
            double lower = guess - 0.5;
            double upper = guess + 0.5;
            double fLower = residual(lower);
            double fUpper = residual(upper);
            int widen = 0;
            while (fLower * fUpper > 0.0 && widen < 20) {
                lower -= 0.5;
                upper += 0.5;
                fLower = residual(lower);
                fUpper = residual(upper);
                ++widen;
            }
            if (!(fLower * fUpper <= 0.0)) {
                throw std::runtime_error("bootstrapEndogenous3m: failed to bracket pillar " +
                                         std::to_string(i));
            }
            const double root = solver.solve(residual, 1e-14, guess, lower, upper);
            lastMove = std::max(lastMove, std::abs(root - previous[i]));
            zeros[i] = root;
        }
        if (lastMove < 1e-15) {
            break;
        }
    }
    std::vector<double> nodeZeros{0.0};
    for (std::size_t i = 0; i < count; ++i) {
        nodeZeros.push_back(zeros[i]);
    }
    return mk::DiscountCurve<double>(nodeTimes, nodeZeros, mk::InterpolationSpace::LogDiscount,
                                     scheme);
}

/// Par rate implied by the endogenous curve for one pillar, used to verify the
/// bootstrap residual independently of the solver.
inline double endogenousParRate(const mk::DiscountCurve<double>& curve, const mk::IrsPillar& irs) {
    const dt::Schedule fixed(irs.start, irs.maturity, irs.fixedTenor, irs.fixedCalendar,
                             irs.businessDayConvention, dt::DateGeneration::Forward, false,
                             dt::BusinessDayConvention::Unadjusted);
    double annuity = 0.0;
    for (std::size_t k = 1; k < fixed.dates().size(); ++k) {
        const double tau =
            dt::yearFraction(fixed.dates()[k - 1], fixed.dates()[k], irs.fixedDayCounter);
        const dt::Date payDate = irs.fixedCalendar.advance(
            fixed.dates()[k], dt::Period(irs.paymentLag, dt::TimeUnit::Days),
            irs.businessDayConvention);
        annuity += tau * curve.discount(dt::yearFraction(kReference, payDate, kZeroDc));
    }
    const double startDiscount = curve.discount(dt::yearFraction(kReference, irs.start, kZeroDc));
    const double endDiscount =
        curve.discount(dt::yearFraction(kReference, fixed.endDate(), kZeroDc));
    return (startDiscount - endDiscount) / annuity;
}

inline mk::DiscountCurve<double>
buildOisEcb(mk::InterpolationScheme scheme = mk::InterpolationScheme::Linear) {
    const std::vector<mk::CurvePillar> pillars = oisEcbPillars();
    return mk::bootstrapDiscountCurve(kReference, kZeroDc, mk::InterpolationSpace::LogDiscount,
                                      scheme, pillars);
}

/// One illustrative portfolio swap: a fixed-vs-float IRS valued on a forecast
/// curve with OIS discounting. `curve` indexes the stack inputs (0 = OIS root,
/// 1..4 = the 1M/3M/6M/12M forecast children).
struct RiskSwap {
    std::size_t curve = 0;
    mk::IrsPillar irs;
    double fixedRate = 0.0;
    double notional = 1.0;
};

/// Node-space gradient of one swap PV, `dV/d zeta`, over the forecast curve
/// nodes (spread nodes for a child, zero nodes for the root) and the discount
/// curve nodes. The bootstrap holds the quotes fixed, so the stack transform
/// consumes these frozen-curve partials.
template <typename ForecastT, typename DiscountT>
inline void accumulateSwapNodeSensitivity(const ForecastT& forecast, const DiscountT& discount,
                                          const mk::IrsPillar& irs, double fixedRate,
                                          double notional, std::vector<double>& dVdForecast,
                                          std::vector<double>& dVdDiscount) {
    const dt::Date effective = irs.start.serial() != 0 ? irs.start : kReference;
    const dt::Schedule floatSchedule(effective, irs.maturity, irs.floatTenor, irs.floatCalendar,
                                     irs.businessDayConvention, dt::DateGeneration::Forward, false,
                                     dt::BusinessDayConvention::Unadjusted);
    const dt::Schedule fixedSchedule(effective, irs.maturity, irs.fixedTenor, irs.fixedCalendar,
                                     irs.businessDayConvention, dt::DateGeneration::Forward, false,
                                     dt::BusinessDayConvention::Unadjusted);
    std::vector<double> weightsPrevious;
    std::vector<double> weightsCurrent;
    std::vector<double> weightsPay;
    const auto accumulatePayment = [&](const dt::Date& payDate, double cashflow) {
        const double tPay = dt::yearFraction(kReference, payDate, kZeroDc);
        const double payDiscount = discount.discount(tPay);
        discount.zeroNodeWeights(tPay, weightsPay);
        for (std::size_t i = 0; i < dVdDiscount.size(); ++i) {
            dVdDiscount[i] += notional * cashflow * (-tPay * payDiscount * weightsPay[i]);
        }
    };
    const std::vector<dt::Date>& floatDates = floatSchedule.dates();
    for (std::size_t k = 1; k < floatDates.size(); ++k) {
        const double tau = dt::yearFraction(floatDates[k - 1], floatDates[k], irs.floatDayCounter);
        const double tPrevious = dt::yearFraction(kReference, floatDates[k - 1], kZeroDc);
        const double tCurrent = dt::yearFraction(kReference, floatDates[k], kZeroDc);
        const double ratio = forecast.discount(tPrevious) / forecast.discount(tCurrent);
        const double forward = (ratio - 1.0) / tau;
        const dt::Date payDate =
            irs.floatCalendar.advance(floatDates[k], dt::Period(irs.paymentLag, dt::TimeUnit::Days),
                                      irs.businessDayConvention);
        const double payDiscount =
            discount.discount(dt::yearFraction(kReference, payDate, kZeroDc));
        forecast.zeroNodeWeights(tPrevious, weightsPrevious);
        forecast.zeroNodeWeights(tCurrent, weightsCurrent);
        for (std::size_t i = 0; i < dVdForecast.size(); ++i) {
            dVdForecast[i] += notional * payDiscount * ratio *
                              (-tPrevious * weightsPrevious[i] + tCurrent * weightsCurrent[i]);
        }
        accumulatePayment(payDate, tau * forward);
    }
    const std::vector<dt::Date>& fixedDates = fixedSchedule.dates();
    for (std::size_t j = 1; j < fixedDates.size(); ++j) {
        const double tau = dt::yearFraction(fixedDates[j - 1], fixedDates[j], irs.fixedDayCounter);
        const dt::Date payDate =
            irs.fixedCalendar.advance(fixedDates[j], dt::Period(irs.paymentLag, dt::TimeUnit::Days),
                                      irs.businessDayConvention);
        accumulatePayment(payDate, -fixedRate * tau);
    }
}

/// Annuity-weighted bump weight of one swap against the funding-window
/// amplitude: for every floating coupon the payment discount times the
/// coupon's window overlap, `sum_k (tau_k D_k) * (overlap_k / tau_k)`, scaled
/// by the coupon's simple-forward level and the overlay's own exponential at
/// the stored amplitude. This is the exact `dV/d(amplitude)` of the flat
/// forward bump.
template <typename ForecastT>
inline double turnWindowWeight(const ForecastT& forecast, const mk::DiscountCurve<double>& discount,
                               const mk::IrsPillar& irs, double amplitude, double notional) {
    const dt::Date effective = irs.start.serial() != 0 ? irs.start : kReference;
    const dt::Schedule floatSchedule(effective, irs.maturity, irs.floatTenor, irs.floatCalendar,
                                     irs.businessDayConvention, dt::DateGeneration::Forward, false,
                                     dt::BusinessDayConvention::Unadjusted);
    double weight = 0.0;
    const std::vector<dt::Date>& dates = floatSchedule.dates();
    for (std::size_t k = 1; k < dates.size(); ++k) {
        const double tPrevious = dt::yearFraction(kReference, dates[k - 1], kZeroDc);
        const double tCurrent = dt::yearFraction(kReference, dates[k], kZeroDc);
        const double widthPrevious =
            std::clamp(tPrevious, kTurnBeginTime, kTurnEndTime) - kTurnBeginTime;
        const double widthCurrent =
            std::clamp(tCurrent, kTurnBeginTime, kTurnEndTime) - kTurnBeginTime;
        const double overlap = widthCurrent - widthPrevious;
        if (!(overlap > 0.0)) {
            continue;
        }
        const dt::Date payDate = irs.floatCalendar.advance(
            dates[k], dt::Period(irs.paymentLag, dt::TimeUnit::Days), irs.businessDayConvention);
        const double baseRatio = forecast.discount(tPrevious) / forecast.discount(tCurrent);
        weight += notional * discount.discount(dt::yearFraction(kReference, payDate, kZeroDc)) *
                  baseRatio * overlap * std::exp(amplitude * overlap);
    }
    return weight;
}

/// Risk-table output path: honour an explicit override, otherwise place the
/// file under the nearest checkout's fit-artifact directory so the CSV lands
/// next to the other fit outputs regardless of the run directory.
inline std::string riskTablePath() {
    if (const char* overridePath = std::getenv("QTA_PAPER_EUR_RISK")) {
        return overridePath;
    }
    const std::filesystem::path relative("internal_docs/paper_fit/risk_table.csv");
    std::filesystem::path base = std::filesystem::current_path();
    for (int depth = 0; depth < 6; ++depth) {
        if (std::filesystem::exists(base / "internal_docs/paper_fit")) {
            return (base / relative).string();
        }
        if (!base.has_parent_path() || base.parent_path() == base) {
            break;
        }
        base = base.parent_path();
    }
    return relative.string();
}

/// Present value of one fixed-vs-float IRS: the floating coupons fix on
/// `forecast`, the fixed rate and every discount factor come from `discount`.
/// Scalar generic, so the same value drives the double node gradients and the
/// reverse-mode overlay turn-risk functor.
template <typename ForecastT, typename DiscountT>
inline auto swapValue(const ForecastT& forecast, const DiscountT& discount,
                      const mk::IrsPillar& irs, double fixedRate, double notional) {
    using Scalar = std::decay_t<decltype(forecast.discount(0.0))>;
    const dt::Date effective = irs.start.serial() != 0 ? irs.start : kReference;
    const dt::Schedule floatSchedule(effective, irs.maturity, irs.floatTenor, irs.floatCalendar,
                                     irs.businessDayConvention, dt::DateGeneration::Forward, false,
                                     dt::BusinessDayConvention::Unadjusted);
    const dt::Schedule fixedSchedule(effective, irs.maturity, irs.fixedTenor, irs.fixedCalendar,
                                     irs.businessDayConvention, dt::DateGeneration::Forward, false,
                                     dt::BusinessDayConvention::Unadjusted);
    Scalar value = 0;
    const std::vector<dt::Date>& floatDates = floatSchedule.dates();
    for (std::size_t k = 1; k < floatDates.size(); ++k) {
        const double tau = dt::yearFraction(floatDates[k - 1], floatDates[k], irs.floatDayCounter);
        const double tPrevious = dt::yearFraction(kReference, floatDates[k - 1], kZeroDc);
        const double tCurrent = dt::yearFraction(kReference, floatDates[k], kZeroDc);
        const Scalar forward =
            (forecast.discount(tPrevious) / forecast.discount(tCurrent) - 1.0) / tau;
        const dt::Date payDate =
            irs.floatCalendar.advance(floatDates[k], dt::Period(irs.paymentLag, dt::TimeUnit::Days),
                                      irs.businessDayConvention);
        value += notional * tau *
                 discount.discount(dt::yearFraction(kReference, payDate, kZeroDc)) *
                 (forward - fixedRate);
    }
    const std::vector<dt::Date>& fixedDates = fixedSchedule.dates();
    for (std::size_t j = 1; j < fixedDates.size(); ++j) {
        const double tau = dt::yearFraction(fixedDates[j - 1], fixedDates[j], irs.fixedDayCounter);
        const dt::Date payDate =
            irs.fixedCalendar.advance(fixedDates[j], dt::Period(irs.paymentLag, dt::TimeUnit::Days),
                                      irs.businessDayConvention);
        value -= notional * fixedRate * tau *
                 discount.discount(dt::yearFraction(kReference, payDate, kZeroDc));
    }
    return value;
}

inline mk::IrsPillar riskIrsPillar(const dt::Date& maturity, const dt::Period& floatTenor) {
    mk::IrsPillar irs;
    irs.start = kSpot;
    irs.maturity = maturity;
    irs.floatTenor = floatTenor;
    irs.fixedTenor = dt::Period(1, dt::TimeUnit::Years);
    irs.floatCalendar = kCalendar;
    irs.fixedCalendar = kCalendar;
    irs.floatDayCounter = kAct360;
    irs.fixedDayCounter = kThirty360;
    irs.businessDayConvention = dt::BusinessDayConvention::ModifiedFollowing;
    return irs;
}

/// Dump the constructed curves to CSV for offline plotting and comparison
/// (`QTA_PAPER_EUR_DUMP=<directory>`). Includes an out-of-sample comparison
/// against the quoted IMM forward-starting IRS, which are not bootstrap
/// instruments.
inline void dumpPaperCurves(const std::string& directory) {
    const mk::DiscountCurve<double> ois = buildOisEcb(mk::InterpolationScheme::HymanSpline);
    const auto parent = std::make_shared<mk::DiscountCurve<double>>(ois);
    const std::vector<mk::ForecastPillar> pillars3m = euribor3mPillars();
    const std::vector<mk::ForecastPillar> pillars3mShort = euribor3mSyntheticPillars();
    const std::vector<mk::ForecastPillar> pillars1mBaseline = euribor1mPillars();
    const std::vector<mk::ForecastPillar> pillars1m = euribor1mSyntheticPillars();
    const mk::SpreadCurve<double> curve3mBaseline = mk::bootstrapForecastCurve(
        parent, &ois, kReference, kZeroDc, mk::InterpolationScheme::Linear, pillars3m);
    const mk::SpreadCurve<double> curve3m = mk::bootstrapForecastCurve(
        parent, &ois, kReference, kZeroDc, mk::InterpolationScheme::Linear, pillars3mShort);
    const mk::SpreadCurve<double> curve1mBaseline = mk::bootstrapForecastCurve(
        parent, &ois, kReference, kZeroDc, mk::InterpolationScheme::Linear, pillars1mBaseline);
    const mk::SpreadCurve<double> curve1m = mk::bootstrapForecastCurve(
        parent, &ois, kReference, kZeroDc, mk::InterpolationScheme::Linear, pillars1m);
    const mk::SpreadCurve<double> curve3mHyman = mk::bootstrapForecastCurve(
        parent, &ois, kReference, kZeroDc, mk::InterpolationScheme::HymanSpline, pillars3mShort);
    const dt::Date postTurnEnd(2014, 3, 13);
    const std::vector<mk::ForecastPillar> pillars1mDirect = directTurnPillars(
        pillars1m, curve1m, kTurnOneMonthAmplitude,
        {exactFraPillar(kTurnEnd, postTurnEnd, simpleForwardOver(curve1m, kTurnEnd, postTurnEnd))});
    const std::vector<mk::ForecastPillar> pillars3mDirect =
        directTurnPillars(pillars3mShort, curve3m, kTurnThreeMonthAmplitude, {});
    const std::vector<mk::ForecastPillar> pillars3mDirectHyman =
        directTurnPillars(pillars3mShort, curve3mHyman, kTurnThreeMonthAmplitude, {});
    const mk::SpreadCurve<double> curve1mDirect = mk::bootstrapForecastCurve(
        parent, &ois, kReference, kZeroDc, mk::InterpolationScheme::Linear, pillars1mDirect);
    const mk::SpreadCurve<double> curve3mDirect = mk::bootstrapForecastCurve(
        parent, &ois, kReference, kZeroDc, mk::InterpolationScheme::Linear, pillars3mDirect);
    const mk::SpreadCurve<double> curve3mDirectHyman =
        mk::bootstrapForecastCurve(parent, &ois, kReference, kZeroDc,
                                   mk::InterpolationScheme::HymanSpline, pillars3mDirectHyman);
    const auto oisShared = std::make_shared<mk::DiscountCurve<double>>(ois);
    const mk::TurnOverlay<double> onOverlay(oisShared, {},
                                            {{kTurnBeginTime, kTurnEndTime, kTurnBumpAmplitude}});
    const auto curve1mShared = std::make_shared<mk::SpreadCurve<double>>(curve1m);
    const auto curve3mHymanShared = std::make_shared<mk::SpreadCurve<double>>(curve3mHyman);
    const mk::TurnOverlay<double, mk::SpreadCurve<double>> overlay1m(
        curve1mShared, {}, {{kTurnBeginTime, kTurnEndTime, kTurnBumpAmplitude}});
    const mk::TurnOverlay<double, mk::SpreadCurve<double>> overlay3m(
        curve3mHymanShared, {}, {{kTurnBeginTime, kTurnEndTime, kTurnBumpAmplitude}});

    std::vector<double> shortGrid;
    for (double t = 1.0 / 365.0; t < 2.0; t += 1.0 / 365.0) {
        shortGrid.push_back(t);
    }
    for (double t = 2.0; t < 5.0; t += 7.0 / 365.0) {
        shortGrid.push_back(t);
    }
    for (double t = 5.0; t <= 30.0; t += 1.0 / 12.0) {
        shortGrid.push_back(t);
    }

    {
        std::ofstream out(directory + "/ois.csv");
        out << "t,zero,fwd1d,fwd3m,fwd1d_overlay\n";
        for (const double t : shortGrid) {
            const double fwd1d = ois.forward(t, t + 1.0 / 365.0);
            const double fwd1dOverlay = onOverlay.forward(t, t + 1.0 / 365.0);
            const double fwd3m = (ois.discount(t) / ois.discount(t + 0.25) - 1.0) / 0.25;
            out << t << ',' << ois.zero(t) << ',' << fwd1d << ',' << fwd3m << ',' << fwd1dOverlay
                << '\n';
        }
    }
    {
        std::ofstream out(directory + "/fra.csv");
        out << "t,fra1m,fra3m,fra3m_hyman,ois3m,basis3m,fra1m_overlay,fra3m_overlay,"
               "fra1m_direct,fra3m_direct,fra3m_direct_hyman,fra1m_baseline\n";
        const auto emitFra = [&](const dt::Date& start) {
            const dt::Date end1 = dt::Period(1, dt::TimeUnit::Months).advance(start);
            const dt::Date end3 = dt::Period(3, dt::TimeUnit::Months).advance(start);
            const double t = dt::yearFraction(kReference, start, kZeroDc);
            const double t1 = dt::yearFraction(kReference, end1, kZeroDc);
            const double t3 = dt::yearFraction(kReference, end3, kZeroDc);
            const double tau1 = dt::yearFraction(start, end1, kAct360);
            const double tau3 = dt::yearFraction(start, end3, kAct360);
            const double fra1m = (curve1m.discount(t) / curve1m.discount(t1) - 1.0) / tau1;
            const double fra1mBaseline =
                (curve1mBaseline.discount(t) / curve1mBaseline.discount(t1) - 1.0) / tau1;
            const double fra1mOverlay =
                (overlay1m.discount(t) / overlay1m.discount(t1) - 1.0) / tau1;
            const double fra3m = (curve3m.discount(t) / curve3m.discount(t3) - 1.0) / tau3;
            const double fra3mHyman =
                (curve3mHyman.discount(t) / curve3mHyman.discount(t3) - 1.0) / tau3;
            const double fra3mOverlay =
                (overlay3m.discount(t) / overlay3m.discount(t3) - 1.0) / tau3;
            const double fra1mDirect =
                (curve1mDirect.discount(t) / curve1mDirect.discount(t1) - 1.0) / tau1;
            const double fra3mDirect =
                (curve3mDirect.discount(t) / curve3mDirect.discount(t3) - 1.0) / tau3;
            const double fra3mDirectHyman =
                (curve3mDirectHyman.discount(t) / curve3mDirectHyman.discount(t3) - 1.0) / tau3;
            const double ois3m = (ois.discount(t) / ois.discount(t3) - 1.0) / tau3;
            out << t << ',' << fra1m << ',' << fra3m << ',' << fra3mHyman << ',' << ois3m << ','
                << fra3m - ois3m << ',' << fra1mOverlay << ',' << fra3mOverlay << ',' << fra1mDirect
                << ',' << fra3mDirect << ',' << fra3mDirectHyman << ',' << fra1mBaseline << '\n';
        };
        const dt::Date twoYears = dt::Period(2, dt::TimeUnit::Years).advance(kSpot);
        const dt::Date endDate = dt::Period(50, dt::TimeUnit::Years).advance(kSpot);
        dt::Date cursor = kSpot;
        while (cursor <= endDate) {
            emitFra(cursor);
            cursor = dt::Period(cursor < twoYears ? 7 : 30, dt::TimeUnit::Days).advance(cursor);
        }
    }
    {
        const mk::DiscountCurve<double> endogenous =
            bootstrapEndogenous3m(pillars3m, mk::InterpolationScheme::Linear);
        std::ofstream out(directory + "/exo_endog.csv");
        out << "t,exogenous,endogenous,difference_bp\n";
        for (double t = 0.5; t <= 29.75; t += 0.25) {
            const double exogenous = simpleThreeMonthForward(curve3mBaseline, t);
            const double classical = simpleThreeMonthForward(endogenous, t);
            out << t << ',' << exogenous << ',' << classical << ',' << (exogenous - classical) * 1e4
                << '\n';
        }
    }
    {
        const mk::DiscountCurve<double> linear =
            bootstrapEndogenous3m(pillars3m, mk::InterpolationScheme::Linear);
        const mk::DiscountCurve<double> monotone =
            bootstrapEndogenous3m(pillars3m, mk::InterpolationScheme::MonotoneCubic);
        const mk::DiscountCurve<double> hyman =
            bootstrapEndogenous3m(pillars3m, mk::InterpolationScheme::HymanSpline);
        std::ofstream out(directory + "/interpolation.csv");
        out << "t,linear,monotone,hyman\n";
        for (double t = 1.0; t <= 30.0; t += 0.25) {
            out << t << ',' << simpleThreeMonthForward(linear, t) << ','
                << simpleThreeMonthForward(monotone, t) << ',' << simpleThreeMonthForward(hyman, t)
                << '\n';
        }
    }
    {
        struct ImmQuote {
            int startYear;
            int startMonth;
            int startDay;
            int year;
            int month;
            int day;
            double percent;
        };
        const ImmQuote quotes[] = {
            {2012, 12, 19, 2013, 12, 19, 0.138}, {2013, 3, 20, 2014, 3, 20, 0.134},
            {2013, 6, 19, 2014, 6, 19, 0.151},   {2013, 9, 18, 2014, 9, 18, 0.183},
            {2013, 12, 18, 2015, 12, 18, 0.183}, {2014, 3, 19, 2016, 3, 21, 0.208},
            {2014, 12, 17, 2017, 12, 18, 0.283},
        };
        std::ofstream out(directory + "/imm_out_of_sample.csv");
        out << "maturity,quote_bp,model_bp,difference_bp\n";
        std::ofstream outOneYear(directory + "/imm_one_year_out_of_sample.csv");
        outOneYear << "start,quote_bp,model_bp,difference_bp\n";
        for (const ImmQuote& quote : quotes) {
            mk::IrsPillar irs = pillars3m.front().irs;
            irs.start = date(quote.startYear, quote.startMonth, quote.startDay);
            irs.maturity = date(quote.year, quote.month, quote.day);
            irs.quote = quote.percent / 100.0;
            const double model = mk::impliedIrsRate(curve3m, ois, irs, kReference, kZeroDc);
            const dt::Date maturity =
                irs.fixedCalendar.adjust(irs.maturity, irs.businessDayConvention);
            out << maturity.year() << '-' << maturity.month() << '-' << maturity.dayOfMonth() << ','
                << quote.percent * 100.0 << ',' << model * 1e4 << ','
                << (model * 1e4 - quote.percent * 100.0) << '\n';
            // The last quoted IMM rows price as one-year swaps from their
            // start dates, not the extracted maturities.
            mk::IrsPillar oneYear = irs;
            oneYear.maturity = dt::Period(1, dt::TimeUnit::Years).advance(irs.start);
            const double modelOneYear =
                mk::impliedIrsRate(curve3m, ois, oneYear, kReference, kZeroDc);
            outOneYear << quote.startYear << '-' << quote.startMonth << '-' << quote.startDay << ','
                       << quote.percent * 100.0 << ',' << modelOneYear * 1e4 << ','
                       << (modelOneYear * 1e4 - quote.percent * 100.0) << '\n';
        }
    }
    {
        std::ofstream out(directory + "/pillars_3m.csv");
        out << "maturity,t,quote_bp\n";
        for (const mk::ForecastPillar& pillar : pillars3m) {
            const dt::Date maturity = pillar.irs.fixedCalendar.adjust(
                pillar.irs.maturity, pillar.irs.businessDayConvention);
            out << maturity.year() << '-' << maturity.month() << '-' << maturity.dayOfMonth() << ','
                << dt::yearFraction(kReference, maturity, kZeroDc) << ',' << pillar.irs.quote * 1e4
                << '\n';
        }
    }
    {
        // Turn diagnostics: instrument reprices, the sampled one- and
        // three-month turn bumps and the forward smoothness away from the
        // funding window, for the plain, overlay and direct-knot variants.
        const auto worstReprice = [&](const auto& pillars, const auto& curve) {
            double worst = 0.0;
            for (const mk::ForecastPillar& pillar : pillars) {
                worst = std::max(worst, std::abs(mk::impliedForecastQuote(curve, ois, pillar,
                                                                          kReference, kZeroDc) -
                                                 forecastPillarTarget(pillar)));
            }
            return worst;
        };
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
        const auto smoothness = [](const auto& curve) {
            double mean = 0.0;
            double maximum = 0.0;
            std::size_t count = 0;
            const auto accumulate = [&](double begin, double end) {
                double previous = simpleThreeMonthForward(curve, begin);
                for (double t = begin + 0.25; t <= end; t += 0.25) {
                    const double current = simpleThreeMonthForward(curve, t);
                    const double jump = std::abs(current - previous);
                    mean += jump;
                    maximum = std::max(maximum, jump);
                    previous = current;
                    ++count;
                }
            };
            accumulate(0.25, 0.90);
            accumulate(1.20, 2.00);
            return std::pair<double, double>(mean / static_cast<double>(count), maximum);
        };
        std::ofstream out(directory + "/turn_diagnostics.csv");
        out << "variant,nodes,worst_reprice,one_month_bump_bp,three_month_bump_bp,"
               "mean_quarterly_jump_bp,max_quarterly_jump_bp\n";
        const auto row = [&](const char* name, std::size_t nodes, double reprice, double bump1m,
                             double bump3m, const auto& curve) {
            const auto stats = smoothness(curve);
            out << name << ',' << nodes << ',' << reprice << ',' << bump1m * 1e4 << ','
                << bump3m * 1e4 << ',' << stats.first * 1e4 << ',' << stats.second * 1e4 << '\n';
        };
        row("plain", curve3mHyman.size(), worstReprice(pillars3mShort, curve3mHyman), 0.0, 0.0,
            curve3mHyman);
        row("overlay", curve3mHyman.size(), worstReprice(pillars3mShort, curve3mHyman),
            sampledBump(overlay1m, curve1m, date(2013, 12, 2), 1),
            sampledBump(overlay3m, curve3mHyman, date(2013, 10, 1), 3), overlay3m);
        row("direct", curve3mDirectHyman.size(),
            worstReprice(pillars3mDirectHyman, curve3mDirectHyman),
            sampledBump(curve1mDirect, curve1m, date(2013, 12, 2), 1),
            sampledBump(curve3mDirectHyman, curve3mHyman, date(2013, 10, 1), 3),
            curve3mDirectHyman);
        QTA_LOG_INFO("test", "Turn diagnostics written: plain/overlay/direct reprices, bumps and "
                             "smoothness");
    }
}

} // namespace quantape::tests::paper_eur

#endif
