/**
 * @file test_curve_risk_report.cpp
 * @brief Hedge QR conditioning and risk-maturity label convention
 *
 * The hedge solve is checked on the augmented Householder QR path: a
 * near-collinear system whose normal equations are numerically singular, an
 * exact square solve, a rank-deficient system tamed by a ridge, and the input
 * validation gates. The report labels, maturity-tag buckets and horizon are
 * checked to follow the maturity each pillar actually prices (calendar-adjusted
 * for a deposit, a FRA and an OIS swap alike).
 */

#include "quantape/markets/Curves/CurveBuilder.h"
#include "quantape/markets/Curves/CurveRiskReport.h"
#include "quantape/markets/Curves/DiscountCurve.h"

#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <string>
#include <vector>

#include "support/GtestSupport.h"

using namespace quantape;

namespace {

using markets::CurvePillar;
using markets::CurveRiskReport;
using markets::DiscountCurve;
using markets::InterpolationScheme;
using markets::InterpolationSpace;
using markets::PillarKind;

const datetime::DayCounter kZeroDc(datetime::DayCount::Actual365Fixed);

template <typename Callable>
void checkInvalidArgument(const char* label, Callable&& call) {
    SCOPED_TRACE(label);
    EXPECT_THROW(call(), std::invalid_argument);
}

void testHedgeNearCollinear() {
    // Columns are nearly parallel: singular values sqrt(2) and 1e-7, so J^T J
    // has condition ~1e14 and the normal-equation solve loses the answer.
    const std::vector<double> jacobian{1.0, 1.0, 1e-7, 0.0, 0.0, 1e-7};
    const std::vector<double> delta{1.0, 0.5, -0.25};
    const std::vector<double> hedge = markets::solveHedge(jacobian, 3, 2, delta, 0.0);
    EXPECT_EQ(hedge.size(), 2u);
    EXPECT_TRUE(std::isfinite(hedge[0]));
    EXPECT_TRUE(std::isfinite(hedge[1]));
    // High-accuracy (50-digit) solution of J^T J h = -J^T delta.
    CHECK_CLOSE("near-collinear hedge 0", hedge[0], -3750000.50000000625, 1e-6);
    CHECK_CLOSE("near-collinear hedge 1", hedge[1], 3749999.49999999375, 1e-6);
    // Normal-equation residual stationarity: J^T (J h + delta) ~ 0.
    double normal0 = 0.0;
    double normal1 = 0.0;
    for (std::size_t i = 0; i < 3; ++i) {
        const double residual =
            jacobian[i * 2] * hedge[0] + jacobian[i * 2 + 1] * hedge[1] + delta[i];
        normal0 += jacobian[i * 2] * residual;
        normal1 += jacobian[i * 2 + 1] * residual;
    }
    EXPECT_LT(std::abs(normal0), 1e-6);
    EXPECT_LT(std::abs(normal1), 1e-6);
}

void testHedgeExactSquare() {
    const std::vector<double> jacobian{2.0, 1.0, 1.0, 3.0};
    const std::vector<double> delta{1.0, -2.0};
    const std::vector<double> hedge = markets::solveHedge(jacobian, 2, 2, delta, 0.0);
    EXPECT_EQ(hedge.size(), 2u);
    CHECK_CLOSE("square hedge 0", hedge[0], -1.0, 1e-12);
    CHECK_CLOSE("square hedge 1", hedge[1], 1.0, 1e-12);
    // Balance gates: J h + delta == 0.
    for (std::size_t i = 0; i < 2; ++i) {
        CHECK_CLOSE("square hedge residual",
                    jacobian[i * 2] * hedge[0] + jacobian[i * 2 + 1] * hedge[1] + delta[i], 0.0,
                    1e-12);
    }
}

void testHedgeRankDeficientRidge() {
    // Rank-1 J (all columns equal). The ridge solution is the minimum-norm
    // direction: x = -sum(delta) / (sum(J^2) + ridge) on both coordinates.
    const std::vector<double> jacobian{1.0, 1.0, 1.0, 1.0, 1.0, 1.0};
    const std::vector<double> delta{1.0, 2.0, 3.0};
    const double sum = 6.0;
    for (const double ridge : {1e-6, 0.5, 1.0}) {
        SCOPED_TRACE(::testing::Message() << "ridge=" << ridge);
        const std::vector<double> hedge = markets::solveHedge(jacobian, 3, 2, delta, ridge);
        const double expected = -sum / (6.0 + ridge);
        CHECK_CLOSE("rank-deficient ridge hedge 0", hedge[0], expected, 1e-9);
        CHECK_CLOSE("rank-deficient ridge hedge 1", hedge[1], expected, 1e-9);
    }
    // A tiny ridge tracks the unregularized minimum-norm solution -sum / 6.
    const std::vector<double> gentle = markets::solveHedge(jacobian, 3, 2, delta, 1e-6);
    CHECK_CLOSE("ridge approaches minimum norm", gentle[0], -sum / 6.0, 1e-6);
    CHECK_CLOSE("ridge approaches minimum norm symmetry", gentle[1], gentle[0], 1e-9);

    // Exactly singular J with no ridge exercises the rank fallback: the
    // minimum-norm solution is recovered instead of dividing by a zero pivot.
    const std::vector<double> singular{1.0, 1.0, 1.0, 1.0};
    const std::vector<double> inRange{-1.0, -1.0};
    const std::vector<double> fallback = markets::solveHedge(singular, 2, 2, inRange, 0.0);
    EXPECT_TRUE(std::isfinite(fallback[0]));
    EXPECT_TRUE(std::isfinite(fallback[1]));
    CHECK_CLOSE("singular fallback hedge 0", fallback[0], 0.5, 1e-9);
    CHECK_CLOSE("singular fallback hedge 1", fallback[1], 0.5, 1e-9);
}

void testHedgeValidation() {
    const std::vector<double> jacobian{1.0, 2.0, 3.0, 4.0};
    const std::vector<double> delta{1.0, 2.0};
    EXPECT_EQ(markets::solveHedge(jacobian, 2, 2, delta, 0.0).size(), 2u);
    checkInvalidArgument("m < h", [&] { markets::solveHedge({1.0, 2.0}, 1, 2, {1.0}, 0.0); });
    checkInvalidArgument("jacobian size",
                         [&] { markets::solveHedge({1.0, 2.0, 3.0}, 2, 1, delta, 0.0); });
    checkInvalidArgument("delta size", [&] { markets::solveHedge(jacobian, 2, 2, {1.0}, 0.0); });
    checkInvalidArgument("h = 0", [&] { markets::solveHedge({}, 0, 0, {}, 0.0); });
    checkInvalidArgument("negative ridge",
                         [&] { markets::solveHedge(jacobian, 2, 2, delta, -1.0); });
    checkInvalidArgument("nan ridge",
                         [&] { markets::solveHedge(jacobian, 2, 2, delta, std::nan("")); });
}

void testRiskMaturityLabels() {
    const datetime::Date reference(2026, 1, 1);
    const datetime::Date quoted(2026, 7, 2);
    const datetime::Calendar calendar =
        datetime::Calendar::weekendsOnly().withExtraHolidays({quoted, quoted.plusDays(1)});
    const datetime::Date adjusted =
        calendar.adjust(quoted, datetime::BusinessDayConvention::ModifiedFollowing);
    // 186 days / 365 = 0.5096: below one year, so the label is the compact date
    // of the adjusted maturity (Monday 2026-07-06) the deposit actually prices.
    EXPECT_EQ(std::lround(datetime::yearFraction(reference, quoted, kZeroDc)), 0);
    EXPECT_EQ(std::lround(datetime::yearFraction(reference, adjusted, kZeroDc)), 1);

    CurvePillar deposit;
    deposit.kind = PillarKind::Deposit;
    deposit.maturity = quoted;
    deposit.quote = 0.03;
    deposit.quoteDayCounter = datetime::DayCounter(datetime::DayCount::Actual360);
    deposit.calendar = calendar;
    deposit.businessDayConvention = datetime::BusinessDayConvention::ModifiedFollowing;
    const std::vector<CurvePillar> depositPillars{deposit};
    const DiscountCurve<double> depositCurve =
        markets::bootstrapDiscountCurve(reference, kZeroDc, InterpolationSpace::LogDiscount,
                                        InterpolationScheme::Linear, depositPillars);
    std::vector<double> depositSensitivity(depositCurve.size(), 0.0);
    depositSensitivity.back() = 1.0;
    const CurveRiskReport depositReport =
        markets::curveRiskReport(depositCurve, depositPillars, reference, depositSensitivity);
    EXPECT_EQ(depositReport.quoteLabels.size(), 1u);
    EXPECT_EQ(depositReport.quoteLabels[0], "Deposit 06Jul26");
    EXPECT_EQ(depositReport.byYear.size(), 1u);
    EXPECT_EQ(depositReport.byYear[0].label, "06Jul26");
    CHECK_CLOSE("deposit horizon uses adjusted maturity", depositReport.horizonYears,
                datetime::yearFraction(reference, adjusted, kZeroDc), 1e-12);
    CHECK_CLOSE("deposit bucket total", depositReport.byYear[0].delta, depositReport.totalDelta(),
                1e-12);

    CurvePillar ois;
    ois.kind = PillarKind::OisSwap;
    ois.maturity = quoted;
    ois.calendar = calendar;
    ois.quoteDayCounter = kZeroDc;
    ois.quote = 0.03;
    ois.fixedTenor = datetime::Period(1, datetime::TimeUnit::Years);
    ois.businessDayConvention = datetime::BusinessDayConvention::ModifiedFollowing;
    const std::vector<CurvePillar> oisPillars{ois};
    const DiscountCurve<double> oisCurve =
        markets::bootstrapDiscountCurve(reference, kZeroDc, InterpolationSpace::LogDiscount,
                                        InterpolationScheme::Linear, oisPillars);
    std::vector<double> oisSensitivity(oisCurve.size(), 0.0);
    oisSensitivity.back() = 1.0;
    const CurveRiskReport oisReport =
        markets::curveRiskReport(oisCurve, oisPillars, reference, oisSensitivity);
    EXPECT_EQ(oisReport.quoteLabels.size(), 1u);
    EXPECT_EQ(oisReport.quoteLabels[0], "OisSwap 06Jul26");
    EXPECT_EQ(oisReport.byYear.size(), 1u);
    EXPECT_EQ(oisReport.byYear[0].label, "06Jul26");
    CHECK_CLOSE("ois horizon uses adjusted maturity", oisReport.horizonYears,
                datetime::yearFraction(reference, adjusted, kZeroDc), 1e-12);
    // The schedule terminates on the adjusted date (the trailing Unadjusted
    // argument is the effective convention), so the OIS pillar reports the date
    // the swap actually terminates on.
    EXPECT_TRUE(markets::adjustedMaturity(ois) == adjusted);
}

void testSubYearBucketSeparation() {
    // Two distinct sub-year maturities must land in different buckets instead
    // of collapsing into a single rounded year, and the buckets must carry the
    // calendar-adjusted date the deposits actually price: 2026-03-02 is a
    // holiday (rolled to 03Mar26) and 2026-09-05 is a weekend (rolled to
    // 07Sep26).
    const datetime::Date reference(2026, 1, 1);
    const datetime::Calendar calendar =
        datetime::Calendar::weekendsOnly().withExtraHolidays({datetime::Date(2026, 3, 2)});
    const auto makeDeposit = [&](const datetime::Date& maturity) {
        CurvePillar pillar;
        pillar.kind = PillarKind::Deposit;
        pillar.maturity = maturity;
        pillar.quote = 0.03;
        pillar.quoteDayCounter = datetime::DayCounter(datetime::DayCount::Actual360);
        pillar.calendar = calendar;
        pillar.businessDayConvention = datetime::BusinessDayConvention::ModifiedFollowing;
        return pillar;
    };
    const std::vector<CurvePillar> pillars{makeDeposit(datetime::Date(2026, 3, 2)),
                                           makeDeposit(datetime::Date(2026, 9, 5))};
    const DiscountCurve<double> curve = markets::bootstrapDiscountCurve(
        reference, kZeroDc, InterpolationSpace::LogDiscount, InterpolationScheme::Linear, pillars);
    std::vector<double> sensitivity(curve.size(), 0.0);
    sensitivity[1] = 1.0;
    sensitivity[2] = 1.0;
    const CurveRiskReport report = markets::curveRiskReport(curve, pillars, reference, sensitivity);
    EXPECT_EQ(report.quoteLabels.size(), 2u);
    EXPECT_EQ(report.quoteLabels[0], "Deposit 03Mar26");
    EXPECT_EQ(report.quoteLabels[1], "Deposit 07Sep26");
    EXPECT_EQ(report.byYear.size(), 2u);
    EXPECT_EQ(report.byYear[0].label, "03Mar26");
    EXPECT_EQ(report.byYear[1].label, "07Sep26");
    CHECK_CLOSE("sub-year bucket total", report.byYear[0].delta + report.byYear[1].delta,
                report.totalDelta(), 1e-12);
}

void testGammaInputValidation() {
    const datetime::Date reference(2026, 1, 1);
    const datetime::Date maturity(2026, 7, 2);
    CurvePillar deposit;
    deposit.kind = PillarKind::Deposit;
    deposit.maturity = maturity;
    deposit.quote = 0.03;
    deposit.quoteDayCounter = datetime::DayCounter(datetime::DayCount::Actual360);
    deposit.calendar = datetime::Calendar::noHolidays();
    const std::vector<CurvePillar> pillars{deposit};
    const DiscountCurve<double> curve = markets::bootstrapDiscountCurve(
        reference, kZeroDc, InterpolationSpace::LogDiscount, InterpolationScheme::Linear, pillars);
    std::vector<double> sensitivity(curve.size(), 0.0);
    sensitivity.back() = 1.0;
    const std::size_t n = curve.size();
    const std::vector<double> tooSmall(n * n - 1, 0.0);
    checkInvalidArgument("HVdZeros size", [&] {
        (void)markets::curveRiskReport(curve, pillars, reference, sensitivity, {}, &tooSmall);
    });
    const std::vector<double> hessian(n * n, 0.0);
    const CurveRiskReport report =
        markets::curveRiskReport(curve, pillars, reference, sensitivity, {}, &hessian);
    EXPECT_EQ(report.gammaDiagonal.size(), pillars.size());
    EXPECT_EQ(report.gammaDiagByYear.size(), 1u);
    EXPECT_EQ(report.gammaCrossByYear.size(), 1u);
    EXPECT_EQ(report.gammaDiagByYear[0].label, "02Jul26");
}

} // namespace

class CurveRiskReportTest : public ::testing::Test {};

TEST_F(CurveRiskReportTest, hedgeNearCollinearAndExactSquare) {
    SCOPED_TRACE("near-collinear");
    testHedgeNearCollinear();
    SCOPED_TRACE("exact square");
    testHedgeExactSquare();
}

TEST_F(CurveRiskReportTest, hedgeRankDeficientRidge) {
    testHedgeRankDeficientRidge();
}

TEST_F(CurveRiskReportTest, hedgeValidation) {
    testHedgeValidation();
}

TEST_F(CurveRiskReportTest, riskMaturityLabels) {
    testRiskMaturityLabels();
}

TEST_F(CurveRiskReportTest, subYearBucketSeparation) {
    testSubYearBucketSeparation();
}

TEST_F(CurveRiskReportTest, gammaInputValidation) {
    testGammaInputValidation();
}
