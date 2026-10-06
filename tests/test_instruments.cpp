/**
 * @file test_instruments.cpp
 * @brief Instruments vocabulary and the bitwise discount-bootstrap gate
 *
 * Phase-0 gate for the plain-data instruments layer:
 *  - `Cashflow` layout and currency round-trip,
 *  - `DiscountCurveSet`/`BootstrapInstrument` concept checks including a
 *    `std::variant` `Ladder`, for `double`, `var` and `fvar<var>`,
 *  - bitwise equality of `bootstrapDiscountCurve` against a frozen copy of
 *    the pre-migration solver on the bootstrap-validation, futures and EUR
 *    fixture strips (node times, node zeros, every pillar quote and discount
 *    samples at 1Y/5Y/10Y/30Y).
 */

#include "quantape/math/StanMath.h"

#include "quantape/datetime/Imm.h"
#include "quantape/instruments/Cashflow.h"
#include "quantape/log/Log.h"
#include "quantape/markets/Curves/BootstrapInstrument.h"
#include "quantape/markets/Curves/CurveBuilder.h"
#include "quantape/markets/Curves/HullWhiteConvexity.h"
#include "quantape/math/Solvers/BrentSolver.h"
#include "quantape/util/Check.h"

#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>
#include <variant>
#include <vector>

#include "paper_eur_curves_fixture.h"

using namespace quantape;

namespace {

namespace dt = quantape::datetime;
namespace mk = quantape::markets;
namespace paper = quantape::tests::paper_eur;

const dt::Date kReference(2026, 9, 29);
const dt::DayCounter kZeroDc(dt::DayCount::Actual365Fixed);

/// Frozen copy of the pre-migration discount-curve solver, used as the
/// old-vs-new bitwise oracle until the owning phase migrates the evaluators.
mk::DiscountCurve<double>
legacyBootstrapDiscountCurve(const dt::Date& referenceDate, const dt::DayCounter& zeroDayCounter,
                             mk::InterpolationSpace space, mk::InterpolationScheme scheme,
                             const std::vector<mk::CurvePillar>& pillars, double accuracy = 1e-14,
                             double tension = 0.0, int switchIndex = 1) {
    if (pillars.empty()) {
        throw std::invalid_argument("bootstrapDiscountCurve: no pillars");
    }
    const std::size_t count = pillars.size();
    std::vector<double> nodeTimes(count);
    std::vector<dt::Date> maturityDates(count);
    for (std::size_t i = 0; i < count; ++i) {
        const dt::Date maturity = mk::pillarRiskMaturity(pillars[i]);
        nodeTimes[i] = dt::yearFraction(referenceDate, maturity, zeroDayCounter);
        if (!(nodeTimes[i] > (i == 0 ? 0.0 : nodeTimes[i - 1]))) {
            throw std::invalid_argument(
                "bootstrapDiscountCurve: maturities must be strictly increasing");
        }
        maturityDates[i] = maturity;
    }
    const dt::DayCounter trialZeroDayCounter(dt::DayCount::Actual365Fixed);
    std::vector<mk::detail::PillarQuoteTimes> solveQuoteTimes(count);
    std::vector<mk::detail::PillarQuoteTimes> checkQuoteTimes(count);
    for (std::size_t i = 0; i < count; ++i) {
        solveQuoteTimes[i] =
            mk::detail::makePillarQuoteTimes(pillars[i], referenceDate, trialZeroDayCounter);
        checkQuoteTimes[i] =
            mk::detail::makePillarQuoteTimes(pillars[i], referenceDate, zeroDayCounter);
    }

    const quantape::math::BrentSolver<double> solver;
    std::vector<double> zeros(count, 0.0);
    std::vector<double> trialTimes;
    std::vector<double> trialZeros;
    const auto solveNodes = [&](bool multiPass) {
        for (int pass = 0; pass < (multiPass ? 50 : 1); ++pass) {
            const std::vector<double> previous = zeros;
            double lastMove = 0.0;
            for (std::size_t i = 0; i < count; ++i) {
                const mk::CurvePillar& pillar = pillars[i];
                const std::size_t lastNode = multiPass && pass == 0 ? i : count - 1;
                bool cacheValid = false;
                double cachedX = 0.0;
                double cachedF = 0.0;
                std::optional<mk::DiscountCurve<double>> trialCurve;
                const auto residual = [&](double trialZero) {
                    if (cacheValid && trialZero == cachedX) {
                        return cachedF;
                    }
                    if (!trialCurve.has_value()) {
                        trialTimes.assign(1, 0.0);
                        trialZeros.assign(1, 0.0);
                        if (trialTimes.capacity() < lastNode + 2) {
                            trialTimes.reserve(lastNode + 2);
                            trialZeros.reserve(lastNode + 2);
                        }
                        for (std::size_t j = 0; j <= lastNode; ++j) {
                            trialTimes.push_back(nodeTimes[j]);
                            trialZeros.push_back(zeros[j]);
                        }
                        trialCurve.emplace(trialTimes, trialZeros, space, scheme, tension,
                                           switchIndex);
                    }
                    mk::detail::CurveTrialUpdater::setNode(*trialCurve, i + 1, trialZero);
                    cachedF = mk::detail::evaluatePillarQuote(solveQuoteTimes[i], *trialCurve) -
                              pillar.quote;
                    cachedX = trialZero;
                    cacheValid = true;
                    return cachedF;
                };

                const double guess = i == 0 ? 0.0 : zeros[i - 1];
                double lower = guess - 0.5;
                double upper = guess + 0.5;
                double fLower = residual(lower);
                double fUpper = residual(upper);
                int widen = 0;
                while (fLower * fUpper > 0.0 && widen < 12) {
                    lower -= 0.5;
                    upper += 0.5;
                    fLower = residual(lower);
                    fUpper = residual(upper);
                    ++widen;
                }
                if (!(fLower * fUpper <= 0.0) || !std::isfinite(fLower) || !std::isfinite(fUpper)) {
                    throw std::runtime_error("bootstrapDiscountCurve: failed to bracket pillar " +
                                             std::to_string(i) + " (" +
                                             std::string(mk::pillarKindName(pillar.kind)) + ")");
                }
                const double root = solver.solve(residual, accuracy, guess, lower, upper);
                const double move = std::abs(root - previous[i]);
                if (move > lastMove) {
                    lastMove = move;
                }
                zeros[i] = root;
            }
            if (!multiPass || lastMove < 1e-15) {
                break;
            }
        }
    };
    const auto worstResidual = [&]() {
        const mk::DiscountCurve<double> curve(referenceDate, maturityDates, zeroDayCounter, zeros,
                                              space, scheme, tension, switchIndex);
        double worst = 0.0;
        for (std::size_t i = 0; i < count; ++i) {
            const double check =
                mk::detail::evaluatePillarQuote(checkQuoteTimes[i], curve) - pillars[i].quote;
            if (!std::isfinite(check)) {
                worst = 1e300;
            } else if (std::abs(check) > worst) {
                worst = std::abs(check);
            }
        }
        return worst;
    };
    solveNodes(false);
    if (!(worstResidual() < 1e-9)) {
        solveNodes(true);
    }
    if (!(worstResidual() < 1e-9)) {
        throw std::runtime_error("bootstrapDiscountCurve: fixed point did not converge");
    }
    return mk::DiscountCurve<double>(referenceDate, maturityDates, zeroDayCounter, zeros, space,
                                     scheme, tension, switchIndex);
}

bool sameBits(double left, double right) {
    return std::bit_cast<std::uint64_t>(left) == std::bit_cast<std::uint64_t>(right);
}

/// Bitwise old-vs-new gate: node grid, all pillar model quotes and discount
/// samples at the standard tenors.
void checkBootstrapSameBits(const std::vector<mk::CurvePillar>& pillars,
                            const mk::DiscountCurve<double>& legacy,
                            const mk::DiscountCurve<double>& current) {
    CHECK(legacy.times().size() == current.times().size());
    CHECK(legacy.zeros().size() == current.zeros().size());
    for (std::size_t i = 0; i < legacy.times().size(); ++i) {
        CHECK(sameBits(legacy.times()[i], current.times()[i]));
    }
    for (std::size_t i = 0; i < legacy.zeros().size(); ++i) {
        CHECK(sameBits(legacy.zeros()[i], current.zeros()[i]));
    }
    for (const mk::CurvePillar& pillar : pillars) {
        CHECK(sameBits(mk::impliedQuote(pillar, legacy.referenceDate(), legacy),
                       mk::impliedQuote(pillar, current.referenceDate(), current)));
    }
    for (const double t : {1.0, 5.0, 10.0, 30.0}) {
        CHECK(sameBits(legacy.discount(t), current.discount(t)));
    }
}

struct SchemeCase {
    mk::InterpolationSpace space;
    mk::InterpolationScheme scheme;
    double tension;
};

const std::vector<SchemeCase>& schemeMatrix() {
    static const std::vector<SchemeCase> matrix{
        {mk::InterpolationSpace::LogDiscount, mk::InterpolationScheme::Linear, 0.0},
        {mk::InterpolationSpace::Zero, mk::InterpolationScheme::Linear, 0.0},
        {mk::InterpolationSpace::LogDiscount, mk::InterpolationScheme::Akima, 0.0},
        {mk::InterpolationSpace::LogDiscount, mk::InterpolationScheme::TensionSpline, 8.0},
        {mk::InterpolationSpace::LogDiscount, mk::InterpolationScheme::MonotoneCubic, 0.0},
        {mk::InterpolationSpace::LogDiscount, mk::InterpolationScheme::MixedLinearCubic, 0.0},
    };
    return matrix;
}

mk::DiscountCurve<double> makeTarget(const SchemeCase& item, const std::vector<dt::Date>& dates,
                                     double base, double slope, double curvature) {
    std::vector<double> zeros;
    zeros.reserve(dates.size());
    for (const dt::Date& date : dates) {
        const double t = dt::yearFraction(kReference, date, kZeroDc);
        zeros.push_back(base + slope * t + curvature * t * t);
    }
    return mk::DiscountCurve<double>(kReference, dates, kZeroDc, zeros, item.space, item.scheme,
                                     item.tension);
}

std::vector<mk::CurvePillar> annualOisPillars(const mk::DiscountCurve<double>& target,
                                              const std::vector<dt::Date>& dates,
                                              const dt::Calendar& calendar,
                                              const dt::Period& fixedTenor, int paymentLag) {
    std::vector<mk::CurvePillar> pillars;
    mk::CurvePillar deposit;
    deposit.maturity = dates.front();
    deposit.kind = mk::PillarKind::Deposit;
    deposit.quoteDayCounter = dt::DayCounter(dt::DayCount::Actual360);
    deposit.calendar = calendar;
    deposit.quote = mk::impliedQuote(deposit, kReference, target);
    pillars.push_back(deposit);
    for (std::size_t i = 1; i < dates.size(); ++i) {
        mk::CurvePillar swap;
        swap.maturity = dates[i];
        swap.kind = mk::PillarKind::OisSwap;
        swap.fixedTenor = fixedTenor;
        swap.paymentLag = paymentLag;
        swap.quoteDayCounter = kZeroDc;
        swap.calendar = calendar;
        swap.quote = mk::impliedQuote(swap, kReference, target);
        pillars.push_back(swap);
    }
    return pillars;
}

void checkBootstrapPair(const SchemeCase& item, const std::vector<mk::CurvePillar>& pillars) {
    const mk::DiscountCurve<double> legacy = legacyBootstrapDiscountCurve(
        kReference, kZeroDc, item.space, item.scheme, pillars, 1e-14, item.tension);
    const mk::DiscountCurve<double> current = mk::bootstrapDiscountCurve(
        kReference, kZeroDc, item.space, item.scheme, pillars, 1e-14, item.tension);
    checkBootstrapSameBits(pillars, legacy, current);
}

void testValidationFixtureBits() {
    const dt::Calendar calendar = dt::Calendar::noHolidays();
    std::vector<dt::Date> nodeDates{dt::Period(6, dt::TimeUnit::Months).advance(kReference)};
    for (int year = 1; year <= 10; ++year) {
        nodeDates.push_back(kReference.plusYears(year));
    }
    for (const SchemeCase& item : schemeMatrix()) {
        const mk::DiscountCurve<double> target =
            makeTarget(item, nodeDates, 0.025, 0.0015, -0.00008);
        checkBootstrapPair(item, annualOisPillars(target, nodeDates, calendar,
                                                  dt::Period(1, dt::TimeUnit::Years), 0));
    }

    std::vector<dt::Date> annualDates;
    for (int year = 1; year <= 6; ++year) {
        annualDates.push_back(kReference.plusYears(year));
    }
    for (const SchemeCase& item : schemeMatrix()) {
        const mk::DiscountCurve<double> target =
            makeTarget(item, annualDates, 0.03, -0.001, 0.0001);
        checkBootstrapPair(item, annualOisPillars(target, annualDates, calendar,
                                                  dt::Period(6, dt::TimeUnit::Months), 2));
    }

    std::vector<dt::Date> stubDates{dt::Period(4, dt::TimeUnit::Months).advance(kReference),
                                    dt::Period(15, dt::TimeUnit::Months).advance(kReference)};
    for (int year = 2; year <= 5; ++year) {
        stubDates.push_back(kReference.plusYears(year));
    }
    for (const SchemeCase& item : schemeMatrix()) {
        const mk::DiscountCurve<double> target = makeTarget(item, stubDates, -0.002, 0.0, 0.0);
        checkBootstrapPair(item, annualOisPillars(target, stubDates, calendar,
                                                  dt::Period(1, dt::TimeUnit::Years), 0));
    }
}

struct FutureStrip {
    std::vector<dt::Date> starts;
    std::vector<dt::Date> ends;
};

FutureStrip immStrip(std::size_t contracts) {
    FutureStrip strip;
    dt::Date start = dt::nextIMMDate(kReference);
    for (std::size_t i = 0; i < contracts; ++i) {
        const dt::Date end = dt::nextIMMDate(start);
        strip.starts.push_back(start);
        strip.ends.push_back(end);
        start = end;
    }
    return strip;
}

std::vector<mk::CurvePillar> futurePillars(const FutureStrip& strip, double convexity) {
    std::vector<mk::CurvePillar> pillars;
    mk::CurvePillar deposit;
    deposit.maturity = dt::Period(3, dt::TimeUnit::Months).advance(kReference);
    deposit.kind = mk::PillarKind::Deposit;
    deposit.quoteDayCounter = dt::DayCounter(dt::DayCount::Actual360);
    pillars.push_back(deposit);
    for (std::size_t i = 0; i < strip.starts.size(); ++i) {
        mk::CurvePillar future;
        future.kind = mk::PillarKind::Future;
        future.start = strip.starts[i];
        future.maturity = strip.ends[i];
        future.quoteDayCounter = dt::DayCounter(dt::DayCount::Actual360);
        future.convexityAdjustment = convexity;
        pillars.push_back(future);
    }
    return pillars;
}

void testFuturesFixtureBits() {
    const FutureStrip strip = immStrip(8);
    std::vector<dt::Date> targetDates{dt::Period(3, dt::TimeUnit::Months).advance(kReference)};
    targetDates.insert(targetDates.end(), strip.ends.begin(), strip.ends.end());
    std::vector<double> zeros;
    zeros.reserve(targetDates.size());
    for (const dt::Date& date : targetDates) {
        zeros.push_back(0.035 + 0.001 * dt::yearFraction(kReference, date, kZeroDc));
    }
    const mk::DiscountCurve<double> target(kReference, targetDates, kZeroDc, zeros,
                                           mk::InterpolationSpace::LogDiscount,
                                           mk::InterpolationScheme::Linear);

    const double convexity = mk::hullWhiteFuturesAdjustment(0.01, 0.05, 1.0, 0.25, 1.0);
    std::vector<mk::CurvePillar> pillars = futurePillars(strip, convexity);
    for (mk::CurvePillar& pillar : pillars) {
        pillar.calendar = dt::Calendar::noHolidays();
        pillar.quote = mk::impliedQuote(pillar, kReference, target);
    }
    const mk::DiscountCurve<double> legacy =
        legacyBootstrapDiscountCurve(kReference, kZeroDc, mk::InterpolationSpace::LogDiscount,
                                     mk::InterpolationScheme::Linear, pillars);
    const mk::DiscountCurve<double> current =
        mk::bootstrapDiscountCurve(kReference, kZeroDc, mk::InterpolationSpace::LogDiscount,
                                   mk::InterpolationScheme::Linear, pillars);
    checkBootstrapSameBits(pillars, legacy, current);
}

void testEurFixtureBits() {
    {
        const std::vector<mk::CurvePillar> pillars = paper::oisSpotPillars();
        const mk::DiscountCurve<double> legacy = legacyBootstrapDiscountCurve(
            paper::kReference, paper::kZeroDc, mk::InterpolationSpace::LogDiscount,
            mk::InterpolationScheme::Linear, pillars);
        const mk::DiscountCurve<double> current = mk::bootstrapDiscountCurve(
            paper::kReference, paper::kZeroDc, mk::InterpolationSpace::LogDiscount,
            mk::InterpolationScheme::Linear, pillars);
        checkBootstrapSameBits(pillars, legacy, current);
    }
    for (const mk::InterpolationScheme scheme :
         {mk::InterpolationScheme::Linear, mk::InterpolationScheme::HymanSpline}) {
        const std::vector<mk::CurvePillar> pillars = paper::oisEcbPillars();
        const mk::DiscountCurve<double> legacy =
            legacyBootstrapDiscountCurve(paper::kReference, paper::kZeroDc,
                                         mk::InterpolationSpace::LogDiscount, scheme, pillars);
        const mk::DiscountCurve<double> current =
            mk::bootstrapDiscountCurve(paper::kReference, paper::kZeroDc,
                                       mk::InterpolationSpace::LogDiscount, scheme, pillars);
        checkBootstrapSameBits(pillars, legacy, current);
    }
}

/// Minimal discount instrument used only to close a second `Ladder`
/// alternative and exercise the concept on a plain data type.
class DepositInstrument {
public:
    DepositInstrument(dt::Date maturity, double quote) : m_maturity(maturity), m_quote(quote) {}

    dt::Date date() const { return m_maturity; }
    double target() const { return m_quote; }

    template <typename ScalarT>
    ScalarT impliedQuote(const mk::DiscountSet<mk::DiscountCurve<ScalarT>>& curves) const {
        return (1.0 / curves.discount(0.5) - 1.0) / 0.5;
    }

private:
    dt::Date m_maturity;
    double m_quote = 0.0;
};

using DiscountPillarInstrument = mk::detail::DiscountPillarInstrument;
using DoubleSet = mk::DiscountSet<mk::DiscountCurve<double>>;
using VarSet = mk::DiscountSet<mk::DiscountCurve<stan::math::var>>;
using NestedSet = mk::DiscountSet<mk::DiscountCurve<stan::math::fvar<stan::math::var>>>;
using InstrumentLadder = mk::Ladder<DepositInstrument, DiscountPillarInstrument>;

static_assert(mk::DiscountCurveSet<DoubleSet, double>);
static_assert(mk::DiscountCurveSet<VarSet, stan::math::var>);
static_assert(mk::DiscountCurveSet<NestedSet, stan::math::fvar<stan::math::var>>);
static_assert(mk::BootstrapInstrument<DiscountPillarInstrument, DoubleSet, double>);
static_assert(mk::BootstrapInstrument<DiscountPillarInstrument, VarSet, stan::math::var>);
static_assert(mk::BootstrapInstrument<DiscountPillarInstrument, NestedSet,
                                      stan::math::fvar<stan::math::var>>);
static_assert(mk::BootstrapInstrument<InstrumentLadder, DoubleSet, double>);
static_assert(mk::BootstrapInstrument<InstrumentLadder, VarSet, stan::math::var>);
static_assert(
    mk::BootstrapInstrument<InstrumentLadder, NestedSet, stan::math::fvar<stan::math::var>>);

template <typename ScalarT>
double scalarValue(const ScalarT& value) {
    if constexpr (std::is_same_v<ScalarT, double>) {
        return value;
    } else if constexpr (std::is_same_v<ScalarT, stan::math::var>) {
        return value.val();
    } else {
        return value.val().val();
    }
}

template <typename ScalarT>
void checkConceptScalar() {
    const std::vector<double> times{0.0, 1.0};
    const std::vector<ScalarT> zeros{0.0, 0.02};
    const mk::DiscountCurve<ScalarT> curve(times, zeros, mk::InterpolationSpace::LogDiscount,
                                           mk::InterpolationScheme::Linear);
    const mk::DiscountSet<mk::DiscountCurve<ScalarT>> curves{curve};

    mk::CurvePillar pillar;
    pillar.kind = mk::PillarKind::Deposit;
    pillar.maturity = kReference.plusYears(1);
    pillar.quoteDayCounter = kZeroDc;
    pillar.calendar = dt::Calendar::noHolidays();
    pillar.quote = 0.02;
    const DiscountPillarInstrument instrument(
        pillar, mk::detail::makePillarQuoteTimes(pillar, kReference, kZeroDc));

    const ScalarT adapterQuote = instrument.impliedQuote<ScalarT>(curves);
    const InstrumentLadder ladder{InstrumentLadder::Instrument{instrument}};
    const ScalarT ladderQuote = ladder.impliedQuote<ScalarT>(curves);
    const double expected = 1.0 / std::exp(-0.02) - 1.0;
    util::checkClose("concept adapter quote", scalarValue(adapterQuote), expected, 1e-12);
    util::checkClose("concept ladder quote", scalarValue(ladderQuote), expected, 1e-12);
    CHECK(ladder.date() == kReference.plusYears(1));
    CHECK(ladder.target() == 0.02);
}

void testConceptQuotes() {
    checkConceptScalar<double>();
    checkConceptScalar<stan::math::var>();
    checkConceptScalar<stan::math::fvar<stan::math::var>>();
}

void testVariantLadder() {
    const std::vector<double> times{0.0, 1.0};
    const std::vector<double> zeros{0.0, 0.02};
    const mk::DiscountCurve<double> curve(times, zeros, mk::InterpolationSpace::LogDiscount,
                                          mk::InterpolationScheme::Linear);
    const DoubleSet curves{curve};

    const DepositInstrument deposit(kReference.plusYears(2), 0.03);
    const InstrumentLadder ladder{InstrumentLadder::Instrument{deposit}};
    const double quote = ladder.impliedQuote<double>(curves);
    util::checkClose("variant ladder deposit quote", quote, (1.0 / curve.discount(0.5) - 1.0) / 0.5,
                     1e-14);
    CHECK(ladder.date() == kReference.plusYears(2));
    CHECK(ladder.target() == 0.03);
    CHECK(std::holds_alternative<DepositInstrument>(ladder.instrument()));
}

void testCashflowLayout() {
    static_assert(std::is_trivially_copyable_v<instruments::Currency>);
    static_assert(std::is_trivially_copyable_v<instruments::Cashflow>);
    static_assert(sizeof(instruments::Currency) == 3);
    static_assert(offsetof(instruments::Cashflow, payDate) <
                  offsetof(instruments::Cashflow, currency));
    static_assert(offsetof(instruments::Cashflow, currency) <
                  offsetof(instruments::Cashflow, amount));
    static_assert(std::is_same_v<decltype(std::declval<instruments::Cashflow>().amount), double>);

    instruments::Cashflow cashflow;
    cashflow.payDate = kReference.plusYears(2);
    cashflow.currency = instruments::Currency{{'E', 'U', 'R'}};
    cashflow.amount = 1234.5;

    const instruments::Cashflow copy = cashflow;
    CHECK(copy.payDate == cashflow.payDate);
    CHECK(copy.currency.code == cashflow.currency.code);
    CHECK(copy.amount == cashflow.amount);
    const std::string_view code(copy.currency.code.data(), copy.currency.code.size());
    CHECK(code == "EUR");
}

} // namespace

int main() {
    testCashflowLayout();
    testConceptQuotes();
    testVariantLadder();
    testValidationFixtureBits();
    testFuturesFixtureBits();
    testEurFixtureBits();
    QTA_LOG_INFO("test", "test_instruments: ok");
    return 0;
}
