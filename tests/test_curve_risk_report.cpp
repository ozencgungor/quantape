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

#include "quantape/log/Log.h"
#include "quantape/markets/Curves/CurveBuilder.h"
#include "quantape/markets/Curves/CurveRiskReport.h"
#include "quantape/markets/Curves/DiscountCurve.h"
#include "quantape/util/Check.h"

#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <string>
#include <vector>

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
    bool threw = false;
    try {
        call();
    } catch (const std::invalid_argument&) {
        threw = true;
    }
    CHECK(threw);
    QTA_LOG_INFO("test", "solveHedge validation {} threw", label);
}

void testHedgeNearCollinear() {
    // Columns are nearly parallel: singular values sqrt(2) and 1e-7, so J^T J
    // has condition ~1e14 and the normal-equation solve loses the answer.
    const std::vector<double> jacobian{1.0, 1.0, 1e-7, 0.0, 0.0, 1e-7};
    const std::vector<double> delta{1.0, 0.5, -0.25};
    const std::vector<double> hedge = markets::solveHedge(jacobian, 3, 2, delta, 0.0);
    CHECK(hedge.size() == 2);
    CHECK(std::isfinite(hedge[0]));
    CHECK(std::isfinite(hedge[1]));
    // High-accuracy (50-digit) solution of J^T J h = -J^T delta.
    util::checkClose("near-collinear hedge 0", hedge[0], -3750000.50000000625, 1e-6);
    util::checkClose("near-collinear hedge 1", hedge[1], 3749999.49999999375, 1e-6);
    // Normal-equation residual stationarity: J^T (J h + delta) ~ 0.
    double normal0 = 0.0;
    double normal1 = 0.0;
    for (std::size_t i = 0; i < 3; ++i) {
        const double residual =
            jacobian[i * 2] * hedge[0] + jacobian[i * 2 + 1] * hedge[1] + delta[i];
        normal0 += jacobian[i * 2] * residual;
        normal1 += jacobian[i * 2 + 1] * residual;
    }
    CHECK(std::abs(normal0) < 1e-6);
    CHECK(std::abs(normal1) < 1e-6);
    QTA_LOG_INFO("test", "near-collinear hedge=({}, {}) normal residual=({}, {})", hedge[0],
                 hedge[1], normal0, normal1);
}

void testHedgeExactSquare() {
    const std::vector<double> jacobian{2.0, 1.0, 1.0, 3.0};
    const std::vector<double> delta{1.0, -2.0};
    const std::vector<double> hedge = markets::solveHedge(jacobian, 2, 2, delta, 0.0);
    CHECK(hedge.size() == 2);
    util::checkClose("square hedge 0", hedge[0], -1.0, 1e-12);
    util::checkClose("square hedge 1", hedge[1], 1.0, 1e-12);
    // Balance gates: J h + delta == 0.
    for (std::size_t i = 0; i < 2; ++i) {
        util::checkClose("square hedge residual",
                         jacobian[i * 2] * hedge[0] + jacobian[i * 2 + 1] * hedge[1] + delta[i],
                         0.0, 1e-12);
    }
    QTA_LOG_INFO("test", "square hedge=({}, {})", hedge[0], hedge[1]);
}

void testHedgeRankDeficientRidge() {
    // Rank-1 J (all columns equal). The ridge solution is the minimum-norm
    // direction: x = -sum(delta) / (sum(J^2) + ridge) on both coordinates.
    const std::vector<double> jacobian{1.0, 1.0, 1.0, 1.0, 1.0, 1.0};
    const std::vector<double> delta{1.0, 2.0, 3.0};
    const double sum = 6.0;
    for (const double ridge : {1e-6, 0.5, 1.0}) {
        const std::vector<double> hedge = markets::solveHedge(jacobian, 3, 2, delta, ridge);
        const double expected = -sum / (6.0 + ridge);
        util::checkClose("rank-deficient ridge hedge 0", hedge[0], expected, 1e-9);
        util::checkClose("rank-deficient ridge hedge 1", hedge[1], expected, 1e-9);
        QTA_LOG_INFO("test", "rank-deficient ridge={} hedge=({}, {}) expected={}", ridge, hedge[0],
                     hedge[1], expected);
    }
    // A tiny ridge tracks the unregularized minimum-norm solution -sum / 6.
    const std::vector<double> gentle = markets::solveHedge(jacobian, 3, 2, delta, 1e-6);
    util::checkClose("ridge approaches minimum norm", gentle[0], -sum / 6.0, 1e-6);
    util::checkClose("ridge approaches minimum norm symmetry", gentle[1], gentle[0], 1e-9);

    // Exactly singular J with no ridge exercises the rank fallback: the
    // minimum-norm solution is recovered instead of dividing by a zero pivot.
    const std::vector<double> singular{1.0, 1.0, 1.0, 1.0};
    const std::vector<double> inRange{-1.0, -1.0};
    const std::vector<double> fallback = markets::solveHedge(singular, 2, 2, inRange, 0.0);
    CHECK(std::isfinite(fallback[0]));
    CHECK(std::isfinite(fallback[1]));
    util::checkClose("singular fallback hedge 0", fallback[0], 0.5, 1e-9);
    util::checkClose("singular fallback hedge 1", fallback[1], 0.5, 1e-9);
    QTA_LOG_INFO("test", "singular fallback hedge=({}, {})", fallback[0], fallback[1]);
}

void testHedgeValidation() {
    const std::vector<double> jacobian{1.0, 2.0, 3.0, 4.0};
    const std::vector<double> delta{1.0, 2.0};
    CHECK(markets::solveHedge(jacobian, 2, 2, delta, 0.0).size() == 2);
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
    CHECK(std::lround(datetime::yearFraction(reference, quoted, kZeroDc)) == 0);
    CHECK(std::lround(datetime::yearFraction(reference, adjusted, kZeroDc)) == 1);

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
    CHECK(depositReport.quoteLabels.size() == 1);
    CHECK(depositReport.quoteLabels[0] == "Deposit 06Jul26");
    CHECK(depositReport.byYear.size() == 1);
    CHECK(depositReport.byYear[0].label == "06Jul26");
    util::checkClose("deposit horizon uses adjusted maturity", depositReport.horizonYears,
                     datetime::yearFraction(reference, adjusted, kZeroDc), 1e-12);
    util::checkClose("deposit bucket total", depositReport.byYear[0].delta,
                     depositReport.totalDelta(), 1e-12);

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
    CHECK(oisReport.quoteLabels.size() == 1);
    CHECK(oisReport.quoteLabels[0] == "OisSwap 06Jul26");
    CHECK(oisReport.byYear.size() == 1);
    CHECK(oisReport.byYear[0].label == "06Jul26");
    util::checkClose("ois horizon uses adjusted maturity", oisReport.horizonYears,
                     datetime::yearFraction(reference, adjusted, kZeroDc), 1e-12);
    // The schedule terminates on the adjusted date (the trailing Unadjusted
    // argument is the effective convention), so the OIS pillar reports the date
    // the swap actually terminates on.
    CHECK(markets::adjustedMaturity(ois) == adjusted);
    QTA_LOG_INFO("test", "risk maturity deposit={} ois={}", depositReport.quoteLabels[0],
                 oisReport.quoteLabels[0]);
}

void testSubYearBucketSeparation() {
    // Two distinct sub-year maturities must land in different buckets instead
    // of collapsing into a single rounded year, and the buckets must carry the
    // calendar-adjusted date the deposits actually price: 2026-03-02 is a
    // holiday (rolled to 03Mar26) and 2026-09-05 is a weekend (rolled to
    // 07Sep26).
    const datetime::Date reference(2026, 1, 1);
    const datetime::Calendar calendar = datetime::Calendar::weekendsOnly().withExtraHolidays(
        {datetime::Date(2026, 3, 2)});
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
    CHECK(report.quoteLabels.size() == 2);
    CHECK(report.quoteLabels[0] == "Deposit 03Mar26");
    CHECK(report.quoteLabels[1] == "Deposit 07Sep26");
    CHECK(report.byYear.size() == 2);
    CHECK(report.byYear[0].label == "03Mar26");
    CHECK(report.byYear[1].label == "07Sep26");
    util::checkClose("sub-year bucket total", report.byYear[0].delta + report.byYear[1].delta,
                     report.totalDelta(), 1e-12);
    QTA_LOG_INFO("test", "sub-year buckets {} and {}", report.byYear[0].label,
                 report.byYear[1].label);
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
    checkInvalidArgument("HVdZeros size",
                         [&] {
                             (void)markets::curveRiskReport(curve, pillars, reference, sensitivity,
                                                            {}, &tooSmall);
                         });
    const std::vector<double> hessian(n * n, 0.0);
    const CurveRiskReport report =
        markets::curveRiskReport(curve, pillars, reference, sensitivity, {}, &hessian);
    CHECK(report.gammaDiagonal.size() == pillars.size());
    CHECK(report.gammaDiagByYear.size() == 1);
    CHECK(report.gammaCrossByYear.size() == 1);
    CHECK(report.gammaDiagByYear[0].label == "02Jul26");
}

} // namespace

int main() {
    testHedgeNearCollinear();
    testHedgeExactSquare();
    testHedgeRankDeficientRidge();
    testHedgeValidation();
    testRiskMaturityLabels();
    testSubYearBucketSeparation();
    testGammaInputValidation();
    QTA_LOG_INFO("test", "test_curve_risk_report: ok");
    return 0;
}
