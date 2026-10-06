/**
 * @file test_ir_instruments.cpp
 * @brief Money-market instrument factories, priority and AD gates
 *
 * Phase-2 gates for the plain-data money-market instruments:
 *  - factory validation errors for deposits, repos, FRAs and futures,
 *  - `date()`/`target()`/`kindName()`/`priority()` metadata and the
 *    Future > Fra > Deposit > Repo overlap ordering with a stable tie-break,
 *  - `var` adjoints of the member `impliedQuote` against central differences
 *    and the `fvar<var>` value/derivative path, for every future style.
 */

#include "quantape/math/StanMath.h"

#include "quantape/datetime/Date.h"
#include "quantape/datetime/DayCounter.h"
#include "quantape/instruments/IrInstruments.h"
#include "quantape/log/Log.h"
#include "quantape/markets/Curves/CurveBuilder.h"
#include "quantape/markets/Curves/DiscountCurve.h"
#include "quantape/pricing/Ir.h"
#include "quantape/pricing/IrMath.h"
#include "quantape/util/Check.h"

#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <string>
#include <vector>

using namespace quantape;

namespace {

namespace dt = quantape::datetime;
namespace inst = quantape::instruments;
namespace mk = quantape::markets;
namespace pr = quantape::pricing;

const dt::Date kReference(2026, 9, 29);
const dt::DayCounter kZeroDc(dt::DayCount::Actual365Fixed);
const dt::DayCounter kIndexDc(dt::DayCount::Actual360);
const dt::Calendar kCalendar = dt::Calendar::noHolidays();
const std::vector<double> kTimes{0.0, 1.0, 2.0};

/// Builds a double curve on the shared node grid.
mk::DiscountCurve<double> makeCurve(const std::vector<double>& zeros) {
    return mk::DiscountCurve<double>(kTimes, zeros, mk::InterpolationSpace::LogDiscount,
                                     mk::InterpolationScheme::Linear);
}

mk::DiscountCurve<double> makeShiftedCurve(const std::vector<double>& zeros, std::size_t node,
                                           double bump) {
    std::vector<double> shifted = zeros;
    shifted[node] += bump;
    return makeCurve(shifted);
}

/// `var` adjoints of one instrument against central differences.
template <typename InstrumentT>
void checkVarAdjoints(const InstrumentT& instrument, const char* tag) {
    using stan::math::var;
    const std::vector<double> baseZeros{0.0, 0.02, 0.025};
    const mk::DiscountCurve<double> primalCurve = makeCurve(baseZeros);
    const double expected = instrument.template impliedQuote<double>(
        mk::DiscountSet<mk::DiscountCurve<double>>{primalCurve});

    stan::math::recover_memory();
    {
        std::vector<var> zeros(baseZeros.begin(), baseZeros.end());
        const mk::DiscountCurve<var> curve(kTimes, zeros, mk::InterpolationSpace::LogDiscount,
                                           mk::InterpolationScheme::Linear);
        var value =
            instrument.template impliedQuote<var>(mk::DiscountSet<mk::DiscountCurve<var>>{curve});
        value.grad();
        util::checkClose((std::string(tag) + " var value").c_str(), value.val(), expected, 1e-15);
        const double step = 1e-6;
        for (std::size_t j = 1; j < baseZeros.size(); ++j) {
            const double fd = (instrument.template impliedQuote<double>(
                                   mk::DiscountSet<mk::DiscountCurve<double>>{
                                       makeShiftedCurve(baseZeros, j, step)}) -
                               instrument.template impliedQuote<double>(
                                   mk::DiscountSet<mk::DiscountCurve<double>>{
                                       makeShiftedCurve(baseZeros, j, -step)})) /
                              (2.0 * step);
            util::checkClose((std::string(tag) + " var adjoint").c_str(), zeros[j].adj(), fd, 1e-6);
        }
    }
    stan::math::recover_memory();
}

/// `fvar<var>` value and directional derivative on a single solved node.
template <typename InstrumentT>
void checkFvarDirection(const InstrumentT& instrument, const char* tag) {
    using stan::math::fvar;
    using stan::math::var;
    const std::vector<double> baseZeros{0.0, 0.02};
    const std::vector<double> times{0.0, 1.0};
    const mk::DiscountCurve<double> primalCurve(
        times, baseZeros, mk::InterpolationSpace::LogDiscount, mk::InterpolationScheme::Linear);
    const double expected = instrument.template impliedQuote<double>(
        mk::DiscountSet<mk::DiscountCurve<double>>{primalCurve});
    const double step = 1e-6;
    const double fd =
        (instrument.template impliedQuote<double>(mk::DiscountSet<mk::DiscountCurve<double>>{
             mk::DiscountCurve<double>(times, std::vector<double>{0.0, 0.02 + step},
                                       mk::InterpolationSpace::LogDiscount,
                                       mk::InterpolationScheme::Linear)}) -
         instrument.template impliedQuote<double>(mk::DiscountSet<mk::DiscountCurve<double>>{
             mk::DiscountCurve<double>(times, std::vector<double>{0.0, 0.02 - step},
                                       mk::InterpolationSpace::LogDiscount,
                                       mk::InterpolationScheme::Linear)})) /
        (2.0 * step);

    stan::math::recover_memory();
    std::vector<fvar<var>> nestedZeros{fvar<var>(var(0.0), 0.0), fvar<var>(var(0.02), 1.0)};
    const mk::DiscountCurve<fvar<var>> nestedCurve(
        times, nestedZeros, mk::InterpolationSpace::LogDiscount, mk::InterpolationScheme::Linear);
    const fvar<var> value = instrument.template impliedQuote<fvar<var>>(
        mk::DiscountSet<mk::DiscountCurve<fvar<var>>>{nestedCurve});
    util::checkClose((std::string(tag) + " fvar value").c_str(), value.val_.val(), expected, 1e-15);
    util::checkClose((std::string(tag) + " fvar derivative").c_str(), value.d_.val(), fd, 1e-6);
    stan::math::recover_memory();
}

void testFactoryValidation() {
    // Deposit/repo: the quoted maturity is mandatory, and a maturity on or
    // before the reference date fails the accrual check.
    for (const bool repo : {false, true}) {
        bool threw = false;
        try {
            if (repo) {
                (void)inst::makeRepo(dt::Date{}, 0.02, kCalendar,
                                     dt::BusinessDayConvention::ModifiedFollowing, kIndexDc,
                                     kReference, kZeroDc);
            } else {
                (void)inst::makeDeposit(dt::Date{}, 0.02, kCalendar,
                                        dt::BusinessDayConvention::ModifiedFollowing, kIndexDc,
                                        kReference, kZeroDc);
            }
        } catch (const std::invalid_argument& error) {
            threw = true;
            const std::string message = error.what();
            CHECK(message.find(repo ? "makeRepo: maturity is required"
                                    : "makeDeposit: maturity is required") != std::string::npos);
        }
        CHECK(threw);
    }

    bool threwAccrual = false;
    try {
        (void)inst::makeDeposit(kReference.plusDays(-1), 0.02, kCalendar,
                                dt::BusinessDayConvention::ModifiedFollowing, kIndexDc, kReference,
                                kZeroDc);
    } catch (const std::invalid_argument& error) {
        threwAccrual = true;
        CHECK(std::string(error.what()).find("impliedQuote: non-positive deposit accrual") !=
              std::string::npos);
    }
    CHECK(threwAccrual);

    // FRA: both accrual ends are mandatory and the start must not precede the
    // reference date.
    bool threwFra = false;
    try {
        (void)inst::makeFra(dt::Date{}, kReference.plusMonths(3), 0.02, kCalendar,
                            dt::BusinessDayConvention::ModifiedFollowing, kIndexDc, kReference,
                            kZeroDc);
    } catch (const std::invalid_argument& error) {
        threwFra = true;
        CHECK(std::string(error.what()).find("makeFra: start and maturity are required") !=
              std::string::npos);
    }
    CHECK(threwFra);

    bool threwFraStart = false;
    try {
        (void)inst::makeFra(kReference.plusMonths(1), kReference.plusMonths(3), 0.02, kCalendar,
                            dt::BusinessDayConvention::ModifiedFollowing, kIndexDc,
                            kReference.plusMonths(2), kZeroDc);
    } catch (const std::invalid_argument& error) {
        threwFraStart = true;
        CHECK(std::string(error.what()).find("impliedQuote: FRA start before the reference date") !=
              std::string::npos);
    }
    CHECK(threwFraStart);

    // Future: the quoted reference period must be non-empty and ordered.
    bool threwFuture = false;
    try {
        (void)inst::makeFuture(kReference.plusMonths(3), kReference.plusMonths(3), 0.04,
                               pr::FutureStyle::Simple, pr::AveragingStyle::Arithmetic, 0.0,
                               kCalendar, kIndexDc, kReference, kZeroDc);
    } catch (const std::invalid_argument& error) {
        threwFuture = true;
        CHECK(std::string(error.what()).find("makeFuture: maturity must be after the start") !=
              std::string::npos);
    }
    CHECK(threwFuture);

    bool threwMissing = false;
    try {
        (void)inst::makeFuture(dt::Date{}, kReference.plusMonths(3), 0.04, pr::FutureStyle::Simple,
                               pr::AveragingStyle::Arithmetic, 0.0, kCalendar, kIndexDc, kReference,
                               kZeroDc);
    } catch (const std::invalid_argument& error) {
        threwMissing = true;
        CHECK(std::string(error.what()).find("makeFuture: start and maturity are required") !=
              std::string::npos);
    }
    CHECK(threwMissing);
}

void testPriorityOrdering() {
    CHECK(inst::Future::priority() < inst::Fra::priority());
    CHECK(inst::Fra::priority() < inst::Deposit::priority());
    CHECK(inst::Deposit::priority() < inst::Repo::priority());

    // Overlapping clusters keep the highest-priority instrument and, on a
    // strict tie, the first-listed one.
    std::vector<mk::CurvePillar> pillars(4);
    pillars[0].kind = mk::PillarKind::Deposit;
    pillars[0].maturity = kReference.plusYears(1);
    pillars[0].quote = 0.011;
    pillars[1].kind = mk::PillarKind::Future;
    pillars[1].maturity = kReference.plusYears(1);
    pillars[1].start = kReference.plusMonths(9);
    pillars[1].quote = 0.033;
    pillars[2].kind = mk::PillarKind::Repo;
    pillars[2].maturity = kReference.plusYears(2);
    pillars[2].quote = 0.022;
    pillars[3].kind = mk::PillarKind::Deposit;
    pillars[3].maturity = kReference.plusYears(2);
    pillars[3].quote = 0.021;

    const std::vector<mk::CurvePillar> filtered = mk::filterOverlappingPillars(pillars, 2);
    CHECK(filtered.size() == 2);
    CHECK(filtered[0].kind == mk::PillarKind::Future);
    CHECK(filtered[1].kind == mk::PillarKind::Deposit);
    CHECK(filtered[1].quote == 0.021);

    // Same-kind ties keep the earlier-listed pillar.
    std::vector<mk::CurvePillar> tied(2);
    tied[0].kind = mk::PillarKind::Repo;
    tied[0].maturity = kReference.plusYears(3);
    tied[0].quote = 0.031;
    tied[1].kind = mk::PillarKind::Repo;
    tied[1].maturity = kReference.plusYears(3);
    tied[1].quote = 0.032;
    const std::vector<mk::CurvePillar> tiedFiltered = mk::filterOverlappingPillars(tied, 2);
    CHECK(tiedFiltered.size() == 1);
    CHECK(tiedFiltered[0].quote == 0.031);
}

void testMetadata() {
    CHECK(inst::Deposit::kindName() == "Deposit");
    CHECK(inst::Repo::kindName() == "Repo");
    CHECK(inst::Fra::kindName() == "Fra");
    CHECK(inst::Future::kindName() == "Future");

    const inst::Deposit deposit = inst::makeDeposit(kReference.plusYears(1), 0.02, kCalendar,
                                                    dt::BusinessDayConvention::ModifiedFollowing,
                                                    kIndexDc, kReference, kZeroDc);
    CHECK(deposit.date() == kReference.plusYears(1));
    CHECK(deposit.target() == 0.02);

    const inst::Repo repo =
        inst::makeRepo(kReference.plusMonths(6), 0.021, kCalendar,
                       dt::BusinessDayConvention::ModifiedFollowing, kIndexDc, kReference, kZeroDc);
    CHECK(repo.date() == kReference.plusMonths(6));
    CHECK(repo.target() == 0.021);

    const inst::Fra fra =
        inst::makeFra(kReference.plusMonths(3), kReference.plusMonths(9), 0.022, kCalendar,
                      dt::BusinessDayConvention::ModifiedFollowing, kIndexDc, kReference, kZeroDc);
    CHECK(fra.date() == kReference.plusMonths(9));
    CHECK(fra.target() == 0.022);

    const inst::Future future = inst::makeFuture(
        kReference.plusMonths(3), kReference.plusMonths(6), 0.041, pr::FutureStyle::Averaged,
        pr::AveragingStyle::Arithmetic, 1e-4, kCalendar, kIndexDc, kReference, kZeroDc);
    CHECK(future.date() == kReference.plusMonths(6));
    CHECK(future.target() == 0.041);
}

void testDiscountAd() {
    const inst::Deposit deposit = inst::makeDeposit(kReference.plusYears(1), 0.02, kCalendar,
                                                    dt::BusinessDayConvention::ModifiedFollowing,
                                                    kIndexDc, kReference, kZeroDc);
    const inst::Repo repo =
        inst::makeRepo(kReference.plusYears(1), 0.02, kCalendar,
                       dt::BusinessDayConvention::ModifiedFollowing, kIndexDc, kReference, kZeroDc);
    const inst::Fra fra =
        inst::makeFra(kReference.plusMonths(6), kReference.plusYears(1), 0.02, kCalendar,
                      dt::BusinessDayConvention::ModifiedFollowing, kIndexDc, kReference, kZeroDc);
    const inst::Future simple = inst::makeFuture(
        kReference.plusMonths(6), kReference.plusYears(1), 0.02, pr::FutureStyle::Simple,
        pr::AveragingStyle::Arithmetic, 0.0, kCalendar, kIndexDc, kReference, kZeroDc);
    const inst::Future compounded = inst::makeFuture(
        kReference.plusMonths(6), kReference.plusYears(1), 0.02, pr::FutureStyle::Compounded,
        pr::AveragingStyle::Arithmetic, 0.0, kCalendar, kIndexDc, kReference, kZeroDc);
    const inst::Future averagedArithmetic = inst::makeFuture(
        kReference.plusMonths(6), kReference.plusYears(1), 0.02, pr::FutureStyle::Averaged,
        pr::AveragingStyle::Arithmetic, 1e-4, kCalendar, kIndexDc, kReference, kZeroDc);
    const inst::Future averagedCompounded = inst::makeFuture(
        kReference.plusMonths(6), kReference.plusYears(1), 0.02, pr::FutureStyle::Averaged,
        pr::AveragingStyle::Compounded, 1e-4, kCalendar, kIndexDc, kReference, kZeroDc);

    checkVarAdjoints(deposit, "deposit");
    checkVarAdjoints(repo, "repo");
    checkVarAdjoints(fra, "fra");
    checkVarAdjoints(simple, "future simple");
    checkVarAdjoints(compounded, "future compounded");
    checkVarAdjoints(averagedArithmetic, "future averaged arithmetic");
    checkVarAdjoints(averagedCompounded, "future averaged compounded");

    checkFvarDirection(deposit, "deposit");
    checkFvarDirection(fra, "fra");
    checkFvarDirection(averagedCompounded, "future averaged compounded");
}

/// Nested reverse-over-forward Hessian of the deposit quote over the solved
/// node, through the instrument member `impliedQuote`.
struct DepositQuoteHessian {
    inst::Deposit deposit;

    template <typename ScalarT>
    ScalarT operator()(const Eigen::Matrix<ScalarT, Eigen::Dynamic, 1>& x) const {
        const std::vector<double> times{0.0, 1.0};
        const std::vector<ScalarT> zeros{ScalarT(0.0), x[0]};
        const mk::DiscountCurve<ScalarT> curve(times, zeros, mk::InterpolationSpace::LogDiscount,
                                               mk::InterpolationScheme::Linear);
        return deposit.template impliedQuote<ScalarT>(
            mk::DiscountSet<mk::DiscountCurve<ScalarT>>{curve});
    }
};

/// The deposit quote has an analytic second derivative: for `r(z) =
/// (exp(z t) - 1) / tau`, `d2r/dz2 = t^2 exp(z t) / tau`.
void testNestedSecondOrder() {
    const inst::Deposit deposit = inst::makeDeposit(kReference.plusYears(1), 0.02, kCalendar,
                                                    dt::BusinessDayConvention::ModifiedFollowing,
                                                    kIndexDc, kReference, kZeroDc);

    Eigen::VectorXd point(1);
    point << 0.02;
    Eigen::VectorXd gradient;
    Eigen::MatrixXd hessian;
    double value = 0.0;
    stan::math::hessian(DepositQuoteHessian{deposit}, point, value, gradient, hessian);
    const double t = kZeroDc.yearFraction(kReference, deposit.date());
    const double tau = kIndexDc.yearFraction(kReference, deposit.date());
    util::checkClose("deposit hessian value", value, (std::exp(0.02 * t) - 1.0) / tau, 1e-12);
    util::checkClose("deposit hessian gradient", gradient[0], t * std::exp(0.02 * t) / tau, 1e-10);
    util::checkClose("deposit hessian second derivative", hessian(0, 0),
                     t * t * std::exp(0.02 * t) / tau, 1e-8);
}

/// Cashflow materialization uses the instrument currency POD and the known
/// fixed flows.
void testCashflows() {
    inst::Deposit deposit = inst::makeDeposit(kReference.plusYears(1), 0.02, kCalendar,
                                              dt::BusinessDayConvention::ModifiedFollowing,
                                              kIndexDc, kReference, kZeroDc);
    deposit.currency = inst::currencyFromCode("EUR");
    const std::vector<inst::Cashflow> flows = pr::irCashflows(deposit, kReference, 1.0e6);
    CHECK(flows.size() == 1);
    CHECK(flows[0].payDate == deposit.date());
    CHECK(inst::currencyCode(flows[0].currency) == "EUR");
    const double tau = kIndexDc.yearFraction(kReference, deposit.date());
    util::checkClose("deposit cashflow amount", flows[0].amount, 1.0e6 * (1.0 + 0.02 * tau), 1e-9);

    const inst::Future future = inst::makeFuture(
        kReference.plusMonths(3), kReference.plusMonths(6), 0.04, pr::FutureStyle::Simple,
        pr::AveragingStyle::Arithmetic, 0.0, kCalendar, kIndexDc, kReference, kZeroDc);
    const std::vector<inst::Cashflow> futureFlows = pr::irCashflows(future, kReference, 1.0e6);
    CHECK(futureFlows.size() == 1);
    const double futureTau = kIndexDc.yearFraction(future.start, future.maturity);
    util::checkClose("future cashflow amount", futureFlows[0].amount, 1.0e6 * 0.04 * futureTau,
                     1e-9);
}

} // namespace

int main() {
    testFactoryValidation();
    testPriorityOrdering();
    testMetadata();
    testDiscountAd();
    testNestedSecondOrder();
    testCashflows();
    QTA_LOG_INFO("test", "test_ir_instruments: ok");
    return 0;
}
