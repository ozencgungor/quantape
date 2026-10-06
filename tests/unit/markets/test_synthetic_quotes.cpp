/**
 * @file test_synthetic_quotes.cpp
 * @brief Synthetic index FRA/deposit quotes from the OIS curve plus basis
 *
 * The synthetic quote uses the exact form
 * `1 + R_x tau_x = (1 + R_on tau_on) * exp(Delta)` with the integrated basis
 * `Delta` taken from the index-versus-overnight spread curve. The tests check
 * exact recovery of the index curve's own forward, the short-end bootstrap, and
 * the error paths.
 */

#include "quantape/markets/Curves/CurveBuilder.h"
#include "quantape/markets/Curves/SyntheticQuotes.h"

#include <cmath>
#include <cstddef>
#include <memory>
#include <vector>

#include "support/GtestSupport.h"

using namespace quantape;

namespace {

using markets::CurvePillar;
using markets::DiscountCurve;
using markets::InterpolationScheme;
using markets::InterpolationSpace;
using markets::PillarKind;
using markets::SpreadCurve;

const datetime::Date kReference(2026, 9, 29);
const datetime::DayCounter kZeroDc(datetime::DayCount::Actual365Fixed);
const datetime::DayCounter kOvernightDc(datetime::DayCount::Actual360);
const datetime::DayCounter kIndexDc(datetime::DayCount::Actual360);

DiscountCurve<double> makeShiftedCurve(int years, double shift) {
    std::vector<datetime::Date> dates;
    std::vector<double> zeros;
    for (int year = 1; year <= years; ++year) {
        const datetime::Date date = kReference.plusYears(year);
        dates.push_back(date);
        zeros.push_back(0.03 + 0.0005 * datetime::yearFraction(kReference, date, kZeroDc) + shift);
    }
    return DiscountCurve<double>(kReference, dates, kZeroDc, zeros, InterpolationSpace::LogDiscount,
                                 InterpolationScheme::Linear);
}

DiscountCurve<double> makeOvernightCurve(int years) {
    std::vector<datetime::Date> dates;
    std::vector<double> zeros;
    for (int year = 1; year <= years; ++year) {
        const datetime::Date date = kReference.plusYears(year);
        dates.push_back(date);
        zeros.push_back(0.03 + 0.0005 * datetime::yearFraction(kReference, date, kZeroDc));
    }
    return DiscountCurve<double>(kReference, dates, kZeroDc, zeros, InterpolationSpace::LogDiscount,
                                 InterpolationScheme::Linear);
}

} // namespace

TEST(SyntheticQuotes, integratedBasisRecoversIndexForward) {
    const DiscountCurve<double> overnight = makeOvernightCurve(5);
    auto parent = std::make_shared<DiscountCurve<double>>(overnight);
    const std::vector<double> times{0.0, 1.0, 2.0, 3.0, 4.0, 5.0};
    const std::vector<double> spreads{0.0, 0.001, 0.001, 0.001, 0.001, 0.001};
    const SpreadCurve<double> index(parent, times, spreads, InterpolationScheme::Linear);

    const datetime::Date start = kReference.plusYears(1);
    const datetime::Date maturity = datetime::Period(3, datetime::TimeUnit::Months).advance(start);
    const double t1 = datetime::yearFraction(kReference, start, kZeroDc);
    const double t2 = datetime::yearFraction(kReference, maturity, kZeroDc);
    const double tau = datetime::yearFraction(start, maturity, kIndexDc);

    const double synthetic = markets::syntheticFraQuote(index, overnight, kReference, start,
                                                        maturity, kOvernightDc, kIndexDc, kZeroDc);
    const double exact = (index.discount(t1) / index.discount(t2) - 1.0) / tau;
    const double delta = markets::integratedBasis(index, overnight, t1, t2);
    EXPECT_TRUE(delta > 0.0);
    // Exact identity: the integrated basis recovers the index curve forward.
    CHECK_CLOSE("synthetic FRA vs index forward", synthetic, exact, 1e-12);
    EXPECT_TRUE(synthetic > 0.0);

    const datetime::Date depositMaturity =
        datetime::Period(3, datetime::TimeUnit::Months).advance(kReference);
    const double syntheticDeposit = markets::syntheticDepositQuote(
        index, overnight, kReference, depositMaturity, kOvernightDc, kIndexDc, kZeroDc);
    const double tDeposit = datetime::yearFraction(kReference, depositMaturity, kZeroDc);
    const double tauDeposit = datetime::yearFraction(kReference, depositMaturity, kIndexDc);
    const double exactDeposit = (1.0 / index.discount(tDeposit) - 1.0) / tauDeposit;
    CHECK_CLOSE("synthetic deposit vs index rate", syntheticDeposit, exactDeposit, 1e-12);
}

TEST(SyntheticQuotes, syntheticDepositPinsShortEnd) {
    const DiscountCurve<double> overnight = makeOvernightCurve(5);
    auto parent = std::make_shared<DiscountCurve<double>>(overnight);
    const std::vector<double> times{0.0, 1.0, 2.0, 3.0, 4.0, 5.0};
    const std::vector<double> spreads{0.0, 0.001, 0.001, 0.001, 0.001, 0.001};
    const SpreadCurve<double> index(parent, times, spreads, InterpolationScheme::Linear);
    const DiscountCurve<double> indexTruth = makeShiftedCurve(5, 0.001);

    // Short end pinned by a synthetic 3M deposit, longer pillars from the
    // index curve itself.
    std::vector<CurvePillar> pillars;
    CurvePillar deposit;
    deposit.maturity = datetime::Period(3, datetime::TimeUnit::Months).advance(kReference);
    deposit.kind = PillarKind::Deposit;
    deposit.quoteDayCounter = kIndexDc;
    deposit.quote = markets::syntheticDepositQuote(index, overnight, kReference, deposit.maturity,
                                                   kOvernightDc, kIndexDc, kZeroDc);
    pillars.push_back(deposit);
    for (int year = 1; year <= 3; ++year) {
        CurvePillar swap;
        swap.maturity = kReference.plusYears(year);
        swap.kind = PillarKind::OisSwap;
        swap.quoteDayCounter = kZeroDc;
        swap.quote = markets::impliedQuote(swap, kReference, indexTruth);
        pillars.push_back(swap);
    }
    const DiscountCurve<double> curve = markets::bootstrapDiscountCurve(
        kReference, kZeroDc, InterpolationSpace::LogDiscount, InterpolationScheme::Linear, pillars);
    for (const CurvePillar& pillar : pillars) {
        CHECK_CLOSE("synthetic short-end reprice", markets::impliedQuote(pillar, kReference, curve),
                    pillar.quote, 1e-9);
    }
    const double tDeposit = datetime::yearFraction(kReference, deposit.maturity, kZeroDc);
    const double tauDeposit = datetime::yearFraction(kReference, deposit.maturity, kIndexDc);
    // The deposit node solves its own synthetic quote exactly.
    CHECK_CLOSE("synthetic deposit node", curve.zero(tDeposit),
                std::log(1.0 + deposit.quote * tauDeposit) / tDeposit, 1e-12);
    // The 1Y pillar is repriced from the index curve, so its node matches.
    const datetime::Date oneYear = kReference.plusYears(1);
    const double tOneYear = datetime::yearFraction(kReference, oneYear, kZeroDc);
    CHECK_CLOSE("1y node vs index curve", curve.zero(tOneYear), indexTruth.zero(tOneYear), 1e-9);
}

TEST(SyntheticQuotes, quoteExtrapolation) {
    const std::vector<double> times{1.0, 2.0, 3.0};
    const std::vector<double> quotes{0.01, 0.02, 0.03};
    CHECK_CLOSE("flat quote extrapolation", markets::extrapolatedQuote(times, quotes, 5.0), 0.03,
                1e-15);
    CHECK_CLOSE("linear quote extrapolation",
                markets::extrapolatedQuote(times, quotes, 5.0, markets::QuoteExtrapolation::Linear),
                0.05, 1e-15);
    EXPECT_THROW((void)markets::extrapolatedQuote(times, quotes, 2.0), std::invalid_argument);
}

TEST(SyntheticQuotes, rejectsInvalidWindows) {
    const DiscountCurve<double> overnight = makeOvernightCurve(3);
    auto parent = std::make_shared<DiscountCurve<double>>(overnight);
    const std::vector<double> times{0.0, 1.0, 2.0, 3.0};
    const std::vector<double> spreads{0.0, 0.001, 0.001, 0.001};
    const SpreadCurve<double> index(parent, times, spreads, InterpolationScheme::Linear);

    EXPECT_THROW((void)markets::integratedBasis(index, overnight, 2.0, 1.0), std::invalid_argument);
    EXPECT_THROW((void)markets::syntheticFraQuote(index, overnight, kReference,
                                                  kReference.plusYears(1), kReference.plusYears(1),
                                                  kOvernightDc, kIndexDc, kZeroDc),
                 std::invalid_argument);
}
