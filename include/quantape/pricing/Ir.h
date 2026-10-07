#pragma once

#include "quantape/datetime/Date.h"
#include "quantape/datetime/Period.h"
#include "quantape/datetime/Schedule.h"
#include "quantape/datetime/TimeConversion.h"
#include "quantape/instruments/Cashflow.h"
#include "quantape/instruments/IrInstruments.h"
#include "quantape/pricing/IrMath.h"

#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <vector>

namespace quantape::pricing {
/**
 * @file Ir.h
 * @brief Money-market instruments plus the scalar quote algebra
 *
 * Composes the plain-data instruments in `quantape/instruments/IrInstruments.h`
 * with the prepared-times algebra in `quantape/pricing/IrMath.h`: the
 * instrument overloads forward to the member `impliedQuote<ScalarT>`, and the
 * cashflow helpers materialize the known fixed flows into `CashflowT` records.
 * The scalar math stays free of instrument and view dependencies, so risk rows
 * can later be templated on the view type without pulling it into pricing.
 */

/// Model quote of a deposit instrument over a discount or forecast curve set.
template <typename ScalarT, typename CurveSetT>
ScalarT impliedDepositQuote(const instruments::Deposit& deposit, const CurveSetT& curves) {
    return deposit.template impliedQuote<ScalarT>(curves);
}

/// Model quote of a repo instrument over a discount or forecast curve set.
template <typename ScalarT, typename CurveSetT>
ScalarT impliedRepoQuote(const instruments::Repo& repo, const CurveSetT& curves) {
    return repo.template impliedQuote<ScalarT>(curves);
}

/// Model quote of a FRA instrument over a discount or forecast curve set.
template <typename ScalarT, typename CurveSetT>
ScalarT impliedFraQuote(const instruments::Fra& fra, const CurveSetT& curves) {
    return fra.template impliedQuote<ScalarT>(curves);
}

/// Model quote of a rate-future instrument over a discount or forecast curve
/// set.
template <typename ScalarT, typename CurveSetT>
ScalarT impliedFutureQuote(const instruments::Future& future, const CurveSetT& curves) {
    return future.template impliedQuote<ScalarT>(curves);
}

/// Model par rate of an IRS instrument over a forecast curve set.
template <typename ScalarT, typename CurveSetT>
ScalarT impliedIrsParRate(const instruments::Irs& irs, const CurveSetT& curves) {
    return irs.template impliedQuote<ScalarT>(curves);
}

/// Model spread of a basis-swap instrument over a forecast curve set.
template <typename ScalarT, typename CurveSetT>
ScalarT impliedBasisSpread(const instruments::BasisSwap& basis, const CurveSetT& curves) {
    return basis.template impliedQuote<ScalarT>(curves);
}

/// Materializes a deposit's known redemption flow: notional plus simple
/// interest at the quoted rate, paid at the maturity date.
template <typename ScalarT>
std::vector<instruments::CashflowT<ScalarT>> irCashflowsT(const instruments::Deposit& deposit,
                                                          const datetime::Date& referenceDate,
                                                          const ScalarT& notional) {
    const datetime::Date payDate = deposit.date();
    const double tau = datetime::yearFraction(referenceDate, payDate, deposit.quoteDayCounter);
    const ScalarT amount = notional * (1.0 + ScalarT(deposit.quote) * tau);
    return {instruments::CashflowT<ScalarT>{payDate, deposit.currency, amount}};
}

/// Materializes a repo's known redemption flow: notional plus simple interest
/// at the quoted rate, paid at the maturity date.
template <typename ScalarT>
std::vector<instruments::CashflowT<ScalarT>> irCashflowsT(const instruments::Repo& repo,
                                                          const datetime::Date& referenceDate,
                                                          const ScalarT& notional) {
    const datetime::Date payDate = repo.date();
    const double tau = datetime::yearFraction(referenceDate, payDate, repo.quoteDayCounter);
    const ScalarT amount = notional * (1.0 + ScalarT(repo.quote) * tau);
    return {instruments::CashflowT<ScalarT>{payDate, repo.currency, amount}};
}

/// Materializes a FRA's known fixed flow: the quoted rate over the accrual
/// period, settled at the maturity date.
template <typename ScalarT>
std::vector<instruments::CashflowT<ScalarT>> irCashflowsT(const instruments::Fra& fra,
                                                          const datetime::Date& /*referenceDate*/,
                                                          const ScalarT& notional) {
    const datetime::Date payDate = fra.date();
    const double tau = datetime::yearFraction(fra.start, payDate, fra.quoteDayCounter);
    const ScalarT amount = notional * ScalarT(fra.quote) * tau;
    return {instruments::CashflowT<ScalarT>{payDate, fra.currency, amount}};
}

/// Materializes a future's known fixed flow: the quoted rate over the quoted
/// reference period, settled at the maturity date.
template <typename ScalarT>
std::vector<instruments::CashflowT<ScalarT>> irCashflowsT(const instruments::Future& future,
                                                          const datetime::Date& /*referenceDate*/,
                                                          const ScalarT& notional) {
    const double tau =
        datetime::yearFraction(future.start, future.maturity, future.quoteDayCounter);
    const ScalarT amount = notional * ScalarT(future.quote) * tau;
    return {instruments::CashflowT<ScalarT>{future.maturity, future.currency, amount}};
}

/// Materializes an IRS's known fixed-leg flows: notional times the quoted
/// fixed rate times each coupon's year fraction, paid on the fixed schedule.
template <typename ScalarT>
std::vector<instruments::CashflowT<ScalarT>>
irCashflowsT(const instruments::Irs& irs, const datetime::Date& referenceDate,
             const ScalarT& notional, const instruments::Currency& currency) {
    const datetime::Date effective = irs.start.serial() != 0 ? irs.start : referenceDate;
    const datetime::Schedule schedule(effective, irs.maturity, irs.fixedTenor, irs.fixedCalendar,
                                      irs.businessDayConvention, datetime::DateGeneration::Forward,
                                      false, datetime::BusinessDayConvention::Unadjusted);
    const std::vector<datetime::Date>& dates = schedule.dates();
    std::vector<instruments::CashflowT<ScalarT>> flows;
    flows.reserve(dates.size() - 1);
    for (std::size_t k = 1; k < dates.size(); ++k) {
        const double tau = datetime::yearFraction(dates[k - 1], dates[k], irs.fixedDayCounter);
        const datetime::Date payDate = irs.fixedCalendar.advance(
            dates[k], datetime::Period(irs.paymentLag, datetime::TimeUnit::Days),
            irs.businessDayConvention);
        flows.push_back(instruments::CashflowT<ScalarT>{payDate, currency,
                                                        notional * ScalarT(irs.quote) * tau});
    }
    return flows;
}

/// Deterministic materialization overloads.
inline std::vector<instruments::Cashflow> irCashflows(const instruments::Deposit& deposit,
                                                      const datetime::Date& referenceDate,
                                                      double notional) {
    return irCashflowsT<double>(deposit, referenceDate, notional);
}

inline std::vector<instruments::Cashflow>
irCashflows(const instruments::Repo& repo, const datetime::Date& referenceDate, double notional) {
    return irCashflowsT<double>(repo, referenceDate, notional);
}

inline std::vector<instruments::Cashflow>
irCashflows(const instruments::Fra& fra, const datetime::Date& referenceDate, double notional) {
    return irCashflowsT<double>(fra, referenceDate, notional);
}

inline std::vector<instruments::Cashflow> irCashflows(const instruments::Future& future,
                                                      const datetime::Date& referenceDate,
                                                      double notional) {
    return irCashflowsT<double>(future, referenceDate, notional);
}

inline std::vector<instruments::Cashflow> irCashflows(const instruments::Irs& irs,
                                                      const datetime::Date& referenceDate,
                                                      double notional,
                                                      const instruments::Currency& currency) {
    return irCashflowsT<double>(irs, referenceDate, notional, currency);
}

namespace detail {

/// Per-thread scratch vectors of the single-curve pillar rows, so repeated
/// rows reuse capacity instead of re-allocating the same working vectors.
struct DiscountRowScratch {
    std::vector<double> weights;
    std::vector<double> weights2;
    std::vector<double> weightsStart;
    std::vector<double> weightsEnd;
    std::vector<double> payTimes;
    std::vector<double> startTimes;
    std::vector<double> endTimes;
    std::vector<double> taus;
    std::vector<double> payDf;
    std::vector<double> startDf;
    std::vector<double> endDf;
    std::vector<double> rowFloating;
    std::vector<double> rowAnnuity;
    ScheduleCache schedules;
};

inline DiscountRowScratch& discountRowScratch() {
    static thread_local DiscountRowScratch scratch;
    return scratch;
}

/// Per-thread scratch of the basis row builder.
struct BasisRowScratch {
    std::vector<double> times;
    std::vector<double> taus;
    std::vector<double> discountDf;
    std::vector<double> childDf;
    std::vector<double> parentDf;
    std::vector<double> childForward;
    std::vector<double> parentForward;
    std::vector<double> weightsPrevious;
    std::vector<double> weightsCurrent;
    std::vector<double> numeratorDiscount;
    std::vector<double> annuityDiscount;
    ScheduleCache schedules;
};

inline BasisRowScratch& basisRowScratch() {
    static thread_local BasisRowScratch scratch;
    return scratch;
}

/// Per-thread scratch of the IRS row builder.
struct IrsRowScratch {
    std::vector<double> weights;
    std::vector<double> annuityRow;
    std::vector<double> discountNumerator;
    std::vector<double> childPrev;
    std::vector<double> childCur;
    std::vector<double> parentPrev;
    std::vector<double> parentCur;
    std::vector<double> payWeights;
    ScheduleCache schedules;
};

inline IrsRowScratch& irsRowScratch() {
    static thread_local IrsRowScratch scratch;
    return scratch;
}

/// Per-thread scratch of the simple money-market forecast row builder.
struct SimpleRowScratch {
    std::vector<double> previous;
    std::vector<double> current;
    std::vector<double> previous2;
    std::vector<double> current2;
};

inline SimpleRowScratch& simpleRowScratch() {
    static thread_local SimpleRowScratch scratch;
    return scratch;
}

} // namespace detail

/// Fields of an OIS par swap needed by its analytic row: the schedule
/// conventions plus the optional known first coupon. The row rebuilds the
/// schedule on the curve's own zero clock, so no coupon times are required.
struct OisRowSpec {
    datetime::Date start{};
    datetime::Date maturity{};
    datetime::Period fixedTenor{1, datetime::TimeUnit::Years};
    datetime::Calendar calendar{};
    datetime::BusinessDayConvention businessDayConvention =
        datetime::BusinessDayConvention::ModifiedFollowing;
    int paymentLag = 0;
    datetime::DayCounter quoteDayCounter{datetime::DayCount::Actual360};
    bool firstCouponFixed = false;
    double firstCouponRate = 0.0;
};

/// Analytic discount-curve row of a deposit: `r = (1/D(T) - 1)/tau`, so
/// `dR/dD(T) = -1/(tau D^2)` and each node enters through `dD(T) = -T D w_i`.
template <class ViewT>
bool jacobianRow(const instruments::Deposit& deposit, const ViewT& curve,
                 const datetime::Date& referenceDate, std::vector<double>& row) {
    const std::size_t n = curve.size();
    row.assign(n - 1, 0.0);
    const datetime::Date maturity = deposit.date();
    const double tau = datetime::yearFraction(referenceDate, maturity, deposit.quoteDayCounter);
    const double t = datetime::yearFraction(referenceDate, maturity, curve.zeroDayCounter());
    const double df = curve.discount(t);
    if (t <= 0.0) {
        return true; // D(0) = 1 is node-independent
    }
    std::vector<double>& weights = detail::discountRowScratch().weights;
    curve.zeroNodeWeights(t, weights);
    const double factor = (-1.0 / (tau * df * df)) * (-t * df);
    for (std::size_t i = 1; i < n; ++i) {
        row[i - 1] += factor * weights[i];
    }
    return true;
}

/// Analytic discount-curve row of a repo: the deposit algebra under a
/// different reporting bucket.
template <class ViewT>
bool jacobianRow(const instruments::Repo& repo, const ViewT& curve,
                 const datetime::Date& referenceDate, std::vector<double>& row) {
    const std::size_t n = curve.size();
    row.assign(n - 1, 0.0);
    const datetime::Date maturity = repo.date();
    const double tau = datetime::yearFraction(referenceDate, maturity, repo.quoteDayCounter);
    const double t = datetime::yearFraction(referenceDate, maturity, curve.zeroDayCounter());
    const double df = curve.discount(t);
    if (t <= 0.0) {
        return true;
    }
    std::vector<double>& weights = detail::discountRowScratch().weights;
    curve.zeroNodeWeights(t, weights);
    const double factor = (-1.0 / (tau * df * df)) * (-t * df);
    for (std::size_t i = 1; i < n; ++i) {
        row[i - 1] += factor * weights[i];
    }
    return true;
}

/// Analytic discount-curve row of a FRA with the shifted-lognormal convexity
/// convention. Returns false when the accrual or end discount is degenerate.
template <class ViewT>
bool jacobianRow(const instruments::Fra& fra, const ViewT& curve,
                 const datetime::Date& referenceDate, std::vector<double>& row) {
    const std::size_t n = curve.size();
    row.assign(n - 1, 0.0);
    const datetime::Date start = fra.calendar.adjust(fra.start, fra.businessDayConvention);
    const datetime::Date maturity = fra.date();
    const double t1 = datetime::yearFraction(referenceDate, start, curve.zeroDayCounter());
    const double t2 = datetime::yearFraction(referenceDate, maturity, curve.zeroDayCounter());
    const double tau = datetime::yearFraction(start, maturity, fra.quoteDayCounter);
    const double d1 = curve.discount(t1);
    const double d2 = curve.discount(t2);
    if (!(tau > 0.0) || !(d2 > 0.0)) {
        return false;
    }
    detail::DiscountRowScratch& scratch = detail::discountRowScratch();
    curve.zeroNodeWeights(t1, scratch.weights);
    curve.zeroNodeWeights(t2, scratch.weights2);
    // dR/df = exp(C) for the shifted-lognormal convexity convention.
    const double scale = std::exp(fra.fraConvexityExponent) * d1 / (tau * d2);
    for (std::size_t i = 1; i < n; ++i) {
        row[i - 1] += scale * (-t1 * scratch.weights[i] + t2 * scratch.weights2[i]);
    }
    return true;
}

/// Analytic discount-curve row of a rate future: the period forward or the
/// arithmetic business-day-grid average, plus the convexity adjustment, which
/// is additive and drops out of the row. Returns false when degenerate.
template <class ViewT>
bool jacobianRow(const instruments::Future& future, const ViewT& curve,
                 const datetime::Date& referenceDate, std::vector<double>& row) {
    const std::size_t n = curve.size();
    row.assign(n - 1, 0.0);
    if (future.futureStyle == FutureStyle::Averaged &&
        future.averagingStyle == AveragingStyle::Arithmetic) {
        const std::vector<datetime::Date> fixings =
            businessDayFixings(future.calendar, future.start, future.maturity);
        const std::size_t periods = fixings.size() - 1;
        if (periods == 0) {
            return false;
        }
        detail::DiscountRowScratch& scratch = detail::discountRowScratch();
        std::vector<double>& weightsStart = scratch.weightsStart;
        std::vector<double>& weightsEnd = scratch.weightsEnd;
        for (std::size_t k = 0; k < periods; ++k) {
            const double tau =
                datetime::yearFraction(fixings[k], fixings[k + 1], future.quoteDayCounter);
            const double t1 =
                datetime::yearFraction(referenceDate, fixings[k], curve.zeroDayCounter());
            const double t2 =
                datetime::yearFraction(referenceDate, fixings[k + 1], curve.zeroDayCounter());
            const double d1 = curve.discount(t1);
            const double d2 = curve.discount(t2);
            if (!(tau > 0.0) || !(d2 > 0.0)) {
                return false;
            }
            curve.zeroNodeWeights(t1, weightsStart);
            curve.zeroNodeWeights(t2, weightsEnd);
            const double ratio = d1 / d2;
            const double scale = 1.0 / (tau * d2);
            for (std::size_t i = 1; i < n; ++i) {
                row[i - 1] +=
                    scale * (-t1 * d1 * weightsStart[i] + ratio * t2 * d2 * weightsEnd[i]);
            }
        }
        for (std::size_t i = 0; i < n - 1; ++i) {
            row[i] /= static_cast<double>(periods);
        }
        return true;
    }
    // Averaged+Compounded falls through here: the daily compounding product
    // telescopes to the period discount ratio, so the simple-rate row is exact.
    const double t1 = datetime::yearFraction(referenceDate, future.start, curve.zeroDayCounter());
    const double t2 =
        datetime::yearFraction(referenceDate, future.maturity, curve.zeroDayCounter());
    const double tau =
        datetime::yearFraction(future.start, future.maturity, future.quoteDayCounter);
    const double d1 = curve.discount(t1);
    const double d2 = curve.discount(t2);
    if (!(tau > 0.0) || !(d2 > 0.0)) {
        return false;
    }
    detail::DiscountRowScratch& scratch = detail::discountRowScratch();
    curve.zeroNodeWeights(t1, scratch.weights);
    curve.zeroNodeWeights(t2, scratch.weights2);
    const double scale = d1 / (tau * d2);
    for (std::size_t i = 1; i < n; ++i) {
        row[i - 1] += scale * (-t1 * scratch.weights[i] + t2 * scratch.weights2[i]);
    }
    return true;
}

/// Analytic discount-curve row of a par OIS: the compounded floating leg
/// telescopes, so `R = floating / annuity` and every discount factor enters
/// through the payment dates and the notional term. Returns false when the
/// annuity is degenerate.
template <class ViewT>
bool jacobianRow(const OisRowSpec& ois, const ViewT& curve, const datetime::Date& referenceDate,
                 std::vector<double>& row) {
    const std::size_t n = curve.size();
    row.assign(n - 1, 0.0);
    detail::DiscountRowScratch& scratch = detail::discountRowScratch();
    const datetime::Date effective = ois.start.serial() != 0 ? ois.start : referenceDate;
    const datetime::Schedule& schedule = scratch.schedules.get(
        effective, ois.maturity, ois.fixedTenor, ois.calendar, ois.businessDayConvention);
    const std::vector<datetime::Date>& dates = schedule.dates();
    const std::size_t payments = dates.size() - 1;
    std::vector<double>& payTimes = scratch.payTimes;
    std::vector<double>& startTimes = scratch.startTimes;
    std::vector<double>& endTimes = scratch.endTimes;
    std::vector<double>& taus = scratch.taus;
    std::vector<double>& payDf = scratch.payDf;
    std::vector<double>& startDf = scratch.startDf;
    std::vector<double>& endDf = scratch.endDf;
    payTimes.resize(payments);
    startTimes.resize(payments);
    endTimes.resize(payments);
    taus.resize(payments);
    payDf.resize(payments);
    startDf.resize(payments);
    endDf.resize(payments);
    double annuity = 0.0;
    double floating = 0.0;
    for (std::size_t k = 0; k < payments; ++k) {
        const datetime::Date payDate = ois.calendar.advance(
            dates[k + 1], datetime::Period(ois.paymentLag, datetime::TimeUnit::Days),
            ois.businessDayConvention);
        payTimes[k] = datetime::yearFraction(referenceDate, payDate, curve.zeroDayCounter());
        startTimes[k] = datetime::yearFraction(referenceDate, dates[k], curve.zeroDayCounter());
        endTimes[k] = datetime::yearFraction(referenceDate, dates[k + 1], curve.zeroDayCounter());
        taus[k] = datetime::yearFraction(dates[k], dates[k + 1], ois.quoteDayCounter);
        payDf[k] = curve.discount(payTimes[k]);
        startDf[k] = curve.discount(startTimes[k]);
        endDf[k] = curve.discount(endTimes[k]);
        annuity += taus[k] * payDf[k];
        if (k == 0 && ois.firstCouponFixed) {
            floating += payDf[k] * taus[k] * ois.firstCouponRate;
        } else {
            floating += payDf[k] * (startDf[k] / endDf[k] - 1.0);
        }
    }
    if (!(annuity > 0.0)) {
        return false;
    }
    std::vector<double>& rowFloating = scratch.rowFloating;
    std::vector<double>& rowAnnuity = scratch.rowAnnuity;
    rowFloating.assign(n - 1, 0.0);
    rowAnnuity.assign(n - 1, 0.0);
    const auto accumulateInto = [&](std::vector<double>& target, double dRdD, double t, double df) {
        if (t <= 0.0) {
            return;
        }
        curve.zeroNodeWeights(t, scratch.weights);
        const double factor = dRdD * (-t * df);
        for (std::size_t i = 1; i < n; ++i) {
            target[i - 1] += factor * scratch.weights[i];
        }
    };
    for (std::size_t k = 0; k < payments; ++k) {
        if (k == 0 && ois.firstCouponFixed) {
            accumulateInto(rowFloating, taus[k] * ois.firstCouponRate, payTimes[k], payDf[k]);
        } else {
            accumulateInto(rowFloating, startDf[k] / endDf[k] - 1.0, payTimes[k], payDf[k]);
            accumulateInto(rowFloating, payDf[k] / endDf[k], startTimes[k], startDf[k]);
            accumulateInto(rowFloating, -payDf[k] * startDf[k] / (endDf[k] * endDf[k]), endTimes[k],
                           endDf[k]);
        }
        accumulateInto(rowAnnuity, taus[k], payTimes[k], payDf[k]);
    }
    const double denominator = annuity * annuity;
    for (std::size_t i = 0; i < n - 1; ++i) {
        row[i] += (rowFloating[i] * annuity - floating * rowAnnuity[i]) / denominator;
    }
    return true;
}

/// Forecast-curve simple money-market rows of a deposit, repo or FRA:
/// `r = (D_f(t1) / D_f(t2) - 1) / tau` measured on the child's zero clock.
/// `ownRow` is over the child spread nodes, `parentRow` over the forecast
/// parent (an additive spread curve shifts the child forwards one-for-one with
/// its parent) and `discountRow` is structurally zero; the optional weight
/// sources express the parent partial on an ancestor curve's node grid.
template <class ViewT>
void simpleForwardJacobianRow(const datetime::Date& startDate, const datetime::Date& maturity,
                              const datetime::DayCounter& quoteDayCounter, const ViewT& child,
                              const ViewT& parent, const ViewT& discount,
                              const datetime::Date& referenceDate, std::vector<double>& ownRow,
                              std::vector<double>& parentRow, std::vector<double>& discountRow,
                              const ViewT* parentWeights, const ViewT* discountWeights) {
    const ViewT& parentWeightCurve = parentWeights != nullptr ? *parentWeights : parent;
    const ViewT& discountWeightCurve = discountWeights != nullptr ? *discountWeights : discount;
    const datetime::Date start = startDate.serial() != 0 ? startDate : referenceDate;
    const datetime::Date maturityDate = maturity;
    const datetime::DayCounter& zeroDayCounter = child.zeroDayCounter();
    const double t1 = datetime::yearFraction(referenceDate, start, zeroDayCounter);
    const double t2 = datetime::yearFraction(referenceDate, maturityDate, zeroDayCounter);
    const double tau = datetime::yearFraction(start, maturityDate, quoteDayCounter);
    if (!(tau > 0.0)) {
        throw std::invalid_argument("forecastSimpleJacobianRowsView: non-positive accrual");
    }
    if (!(t1 >= 0.0)) {
        throw std::invalid_argument(
            "forecastSimpleJacobianRowsView: start before the reference date");
    }
    const double ratio = child.discount(t1) / child.discount(t2);
    ownRow.assign(child.size() - 1, 0.0);
    parentRow.assign(parentWeightCurve.size() - 1, 0.0);
    discountRow.assign(discountWeightCurve.size() - 1, 0.0);
    const auto weightsAt = [](const ViewT& curve, double time, std::vector<double>& out) {
        // D(0) = 1 is node-independent; sensitivities at t = 0 are zero.
        if (time <= 0.0) {
            out.assign(curve.size(), 0.0);
            return;
        }
        curve.zeroNodeWeights(time, out);
    };
    detail::SimpleRowScratch& scratch = detail::simpleRowScratch();
    std::vector<double>& previous = scratch.previous;
    std::vector<double>& current = scratch.current;
    const double factor = ratio / tau;
    weightsAt(child, t1, previous);
    weightsAt(child, t2, current);
    for (std::size_t j = 1; j < child.size(); ++j) {
        ownRow[j - 1] = factor * (-t1 * previous[j] + t2 * current[j]);
    }
    weightsAt(parentWeightCurve, t1, previous);
    weightsAt(parentWeightCurve, t2, current);
    for (std::size_t j = 1; j < parentWeightCurve.size(); ++j) {
        parentRow[j - 1] = factor * (-t1 * previous[j] + t2 * current[j]);
    }
}

/// Forecast-curve rows of a synthetic deposit (spot or forward starting).
template <class ViewT>
void jacobianRow(const instruments::Deposit& deposit, const ViewT& child, const ViewT& parent,
                 const ViewT& discount, const datetime::Date& referenceDate,
                 std::vector<double>& ownRow, std::vector<double>& parentRow,
                 std::vector<double>& discountRow, const ViewT* parentWeights = nullptr,
                 const ViewT* discountWeights = nullptr) {
    simpleForwardJacobianRow(deposit.start, deposit.date(), deposit.quoteDayCounter, child, parent,
                             discount, referenceDate, ownRow, parentRow, discountRow, parentWeights,
                             discountWeights);
}

/// Forecast-curve rows of a synthetic repo quote.
template <class ViewT>
void jacobianRow(const instruments::Repo& repo, const ViewT& child, const ViewT& parent,
                 const ViewT& discount, const datetime::Date& referenceDate,
                 std::vector<double>& ownRow, std::vector<double>& parentRow,
                 std::vector<double>& discountRow, const ViewT* parentWeights = nullptr,
                 const ViewT* discountWeights = nullptr) {
    simpleForwardJacobianRow(repo.start, repo.date(), repo.quoteDayCounter, child, parent, discount,
                             referenceDate, ownRow, parentRow, discountRow, parentWeights,
                             discountWeights);
}

/// Forecast-curve rows of a synthetic FRA quote. Forecast-curve FRAs are
/// quoted as plain simple forwards, so the convexity exponent is not part of
/// this row (the discount-curve overload applies it).
template <class ViewT>
void jacobianRow(const instruments::Fra& fra, const ViewT& child, const ViewT& parent,
                 const ViewT& discount, const datetime::Date& referenceDate,
                 std::vector<double>& ownRow, std::vector<double>& parentRow,
                 std::vector<double>& discountRow, const ViewT* parentWeights = nullptr,
                 const ViewT* discountWeights = nullptr) {
    simpleForwardJacobianRow(fra.start, fra.date(), fra.quoteDayCounter, child, parent, discount,
                             referenceDate, ownRow, parentRow, discountRow, parentWeights,
                             discountWeights);
}

/// Core own/parent rows of a forecast-curve rate future: the period forward or
/// the business-day-grid averaged rate on `child` plus the stored convexity
/// adjustment. Convexity is additive and drops out of the row. `ownRow` picks
/// up the child spread-node weights; `parentRow` the `parent` weights, whose
/// nodes shift the child zero curve one-for-one.
template <class ViewT>
void jacobianRow(const instruments::Future& future, const ViewT& child, const ViewT& parent,
                 const datetime::Date& referenceDate, std::vector<double>& ownRow,
                 std::vector<double>& parentRow) {
    const datetime::DayCounter& zeroDayCounter = child.zeroDayCounter();
    ownRow.assign(child.size() - 1, 0.0);
    parentRow.assign(parent.size() - 1, 0.0);
    const auto weightsAt = [](const auto& curve, double time, std::vector<double>& out) {
        // D(0) = 1 is node-independent; sensitivities at t = 0 are zero.
        if (time <= 0.0) {
            out.assign(curve.size(), 0.0);
            return;
        }
        curve.zeroNodeWeights(time, out);
    };
    const double startTime = datetime::yearFraction(referenceDate, future.start, zeroDayCounter);
    if (!(startTime >= 0.0)) {
        throw std::invalid_argument(
            "forecastFutureRowsCore: futures fixing before the reference date");
    }
    detail::SimpleRowScratch& scratch = detail::simpleRowScratch();
    std::vector<double>& weightsStart = scratch.previous2;
    std::vector<double>& weightsEnd = scratch.current2;
    if (future.futureStyle == FutureStyle::Averaged &&
        future.averagingStyle == AveragingStyle::Arithmetic) {
        const std::vector<datetime::Date> fixings =
            businessDayFixings(future.calendar, future.start, future.maturity);
        const std::size_t periods = fixings.size() - 1;
        if (periods == 0) {
            throw std::invalid_argument(
                "forecastFutureRowsCore: empty averaged futures reference period");
        }
        for (std::size_t k = 0; k < periods; ++k) {
            const double tau =
                datetime::yearFraction(fixings[k], fixings[k + 1], future.quoteDayCounter);
            if (!(tau > 0.0)) {
                throw std::invalid_argument(
                    "forecastFutureRowsCore: non-positive averaged futures accrual");
            }
            const double t1 = datetime::yearFraction(referenceDate, fixings[k], zeroDayCounter);
            const double t2 = datetime::yearFraction(referenceDate, fixings[k + 1], zeroDayCounter);
            const double d1 = child.discount(t1);
            const double d2 = child.discount(t2);
            const double ratio = d1 / d2;
            const double scale = 1.0 / (tau * d2);
            const auto accumulate = [&](const auto& weightsCurve, std::vector<double>& row) {
                weightsAt(weightsCurve, t1, weightsStart);
                weightsAt(weightsCurve, t2, weightsEnd);
                for (std::size_t j = 1; j < row.size() + 1; ++j) {
                    row[j - 1] +=
                        scale * (-t1 * d1 * weightsStart[j] + ratio * t2 * d2 * weightsEnd[j]);
                }
            };
            accumulate(child, ownRow);
            accumulate(parent, parentRow);
        }
        for (double& value : ownRow) {
            value /= static_cast<double>(periods);
        }
        for (double& value : parentRow) {
            value /= static_cast<double>(periods);
        }
        return;
    }
    // Simple, compounded and compounded-average futures: the daily compounded
    // products telescope to the period forward, so the simple-rate row is
    // exact.
    const double tau =
        datetime::yearFraction(future.start, future.maturity, future.quoteDayCounter);
    if (!(tau > 0.0)) {
        throw std::invalid_argument("forecastFutureRowsCore: non-positive futures accrual");
    }
    const double endTime = datetime::yearFraction(referenceDate, future.maturity, zeroDayCounter);
    const double d1 = child.discount(startTime);
    const double d2 = child.discount(endTime);
    const double factor = d1 / (tau * d2);
    weightsAt(child, startTime, weightsStart);
    weightsAt(child, endTime, weightsEnd);
    for (std::size_t j = 1; j < ownRow.size() + 1; ++j) {
        ownRow[j - 1] = factor * (-startTime * weightsStart[j] + endTime * weightsEnd[j]);
    }
    weightsAt(parent, startTime, weightsStart);
    weightsAt(parent, endTime, weightsEnd);
    for (std::size_t j = 1; j < parentRow.size() + 1; ++j) {
        parentRow[j - 1] = factor * (-startTime * weightsStart[j] + endTime * weightsEnd[j]);
    }
}

/// View rows of a forecast-curve rate future: the own and forecast-parent
/// rows on the optional ancestor weight grids plus a structurally zero
/// discount row sized on the discount grid.
template <class ViewT>
void jacobianRow(const instruments::Future& future, const ViewT& child, const ViewT& parent,
                 const ViewT& discount, const datetime::Date& referenceDate,
                 std::vector<double>& ownRow, std::vector<double>& parentRow,
                 std::vector<double>& discountRow, const ViewT* parentWeights = nullptr,
                 const ViewT* discountWeights = nullptr) {
    const ViewT& parentWeightCurve = parentWeights != nullptr ? *parentWeights : parent;
    const ViewT& discountWeightCurve = discountWeights != nullptr ? *discountWeights : discount;
    jacobianRow(future, child, parentWeightCurve, referenceDate, ownRow, parentRow);
    discountRow.assign(discountWeightCurve.size() - 1, 0.0);
}

/// View-based basis rows for an arbitrary forecast parent and discount curve:
/// `ownRow` over the child spread nodes, `parentRow` over the forecast-parent
/// nodes (both forwards move with the parent, the annuity is frozen), and
/// `discountRow` over the discount nodes (discount factors only). The optional
/// weight sources let the same curve-space partial be expressed on an ancestor
/// curve's node grid, which is how a depth-2 chain's rows reach the root block.
template <class ViewT>
void jacobianRow(const instruments::BasisSwap& basis, const ViewT& child, const ViewT& parent,
                 const ViewT& discount, const datetime::Date& referenceDate,
                 std::vector<double>& ownRow, std::vector<double>& parentRow,
                 std::vector<double>& discountRow, const ViewT* parentWeights = nullptr,
                 const ViewT* discountWeights = nullptr) {
    detail::BasisRowScratch& scratch = detail::basisRowScratch();
    const datetime::Schedule& schedule =
        scratch.schedules.get(referenceDate, basis.maturity, basis.floatTenor, basis.calendar,
                              basis.businessDayConvention);
    const std::vector<datetime::Date>& dates = schedule.dates();
    const std::size_t periods = dates.size() - 1;
    const std::size_t nChild = child.size();
    const ViewT& parentWeightCurve = parentWeights != nullptr ? *parentWeights : parent;
    const ViewT& discountWeightCurve = discountWeights != nullptr ? *discountWeights : discount;
    const std::size_t nParent = parentWeightCurve.size();
    const std::size_t nDiscount = discountWeightCurve.size();
    const datetime::DayCounter& zeroDayCounter = discount.zeroDayCounter();
    std::vector<double>& times = scratch.times;
    std::vector<double>& taus = scratch.taus;
    std::vector<double>& discountDf = scratch.discountDf;
    std::vector<double>& childDf = scratch.childDf;
    std::vector<double>& parentDf = scratch.parentDf;
    std::vector<double>& childForward = scratch.childForward;
    std::vector<double>& parentForward = scratch.parentForward;
    times.resize(periods + 1);
    taus.resize(periods);
    discountDf.resize(periods + 1);
    childDf.resize(periods + 1);
    parentDf.resize(periods + 1);
    childForward.resize(periods);
    parentForward.resize(periods);
    times[0] = datetime::yearFraction(referenceDate, dates[0], zeroDayCounter);
    discountDf[0] = discount.discount(times[0]);
    childDf[0] = child.discount(times[0]);
    parentDf[0] = parent.discount(times[0]);
    double annuity = 0.0;
    double numerator = 0.0;
    for (std::size_t k = 0; k < periods; ++k) {
        times[k + 1] = datetime::yearFraction(referenceDate, dates[k + 1], zeroDayCounter);
        taus[k] = datetime::yearFraction(dates[k], dates[k + 1], basis.quoteDayCounter);
        discountDf[k + 1] = discount.discount(times[k + 1]);
        childDf[k + 1] = child.discount(times[k + 1]);
        parentDf[k + 1] = parent.discount(times[k + 1]);
        childForward[k] = (childDf[k] / childDf[k + 1] - 1.0) / taus[k];
        parentForward[k] = (parentDf[k] / parentDf[k + 1] - 1.0) / taus[k];
        annuity += taus[k] * discountDf[k + 1];
        numerator += taus[k] * discountDf[k + 1] * (childForward[k] - parentForward[k]);
    }
    if (!(annuity > 0.0)) {
        throw std::invalid_argument("basisSwapJacobianRowsView: non-positive annuity");
    }
    ownRow.assign(nChild - 1, 0.0);
    parentRow.assign(nParent - 1, 0.0);
    discountRow.assign(nDiscount - 1, 0.0);
    const auto weightsAt = [](const ViewT& curve, double time, std::vector<double>& out) {
        // D(0) = 1 is node-independent; sensitivities at t = 0 are zero.
        if (time <= 0.0) {
            out.assign(curve.size(), 0.0);
            return;
        }
        curve.zeroNodeWeights(time, out);
    };
    std::vector<double>& weightsPrevious = scratch.weightsPrevious;
    std::vector<double>& weightsCurrent = scratch.weightsCurrent;
    for (std::size_t k = 0; k < periods; ++k) {
        weightsAt(child, times[k], weightsPrevious);
        weightsAt(child, times[k + 1], weightsCurrent);
        const double childFactor = (1.0 + taus[k] * childForward[k]) / taus[k];
        const double weight = taus[k] * discountDf[k + 1] / annuity;
        for (std::size_t j = 1; j < nChild; ++j) {
            ownRow[j - 1] += weight * childFactor *
                             (-times[k] * weightsPrevious[j] + times[k + 1] * weightsCurrent[j]);
        }
        weightsAt(parentWeightCurve, times[k], weightsPrevious);
        weightsAt(parentWeightCurve, times[k + 1], weightsCurrent);
        const double forwardSpread = childForward[k] - parentForward[k];
        for (std::size_t j = 1; j < nParent; ++j) {
            parentRow[j - 1] += weight * forwardSpread *
                                (-times[k] * weightsPrevious[j] + times[k + 1] * weightsCurrent[j]);
        }
    }
    std::vector<double>& numeratorDiscount = scratch.numeratorDiscount;
    std::vector<double>& annuityDiscount = scratch.annuityDiscount;
    numeratorDiscount.assign(nDiscount - 1, 0.0);
    annuityDiscount.assign(nDiscount - 1, 0.0);
    for (std::size_t k = 0; k < periods; ++k) {
        weightsAt(discountWeightCurve, times[k + 1], weightsCurrent);
        const double forwardSpread = childForward[k] - parentForward[k];
        for (std::size_t j = 1; j < nDiscount; ++j) {
            const double dDiscount = -times[k + 1] * discountDf[k + 1] * weightsCurrent[j];
            numeratorDiscount[j - 1] += taus[k] * dDiscount * forwardSpread;
            annuityDiscount[j - 1] += taus[k] * dDiscount;
        }
    }
    const double denominator = annuity * annuity;
    for (std::size_t j = 0; j < nDiscount - 1; ++j) {
        discountRow[j] =
            (numeratorDiscount[j] * annuity - numerator * annuityDiscount[j]) / denominator;
    }
    if (!basis.spreadOnParentLeg) {
        for (double& value : ownRow) {
            value = -value;
        }
        for (double& value : parentRow) {
            value = -value;
        }
        for (double& value : discountRow) {
            value = -value;
        }
    }
}

/// Core IRS rows for `R = floatPv / fixedAnnuity` with the floating forwards
/// from the child spread curve over `forecastParent` and every discount factor
/// from `discount`, all year fractions measured on `zeroDayCounter`. The
/// optional weight sources express the same partials on an ancestor curve's
/// node grid for depth-2 chains.
template <class ViewT>
void jacobianRow(const instruments::Irs& irs, const ViewT& child, const ViewT& forecastParent,
                 const ViewT& discount, const datetime::Date& referenceDate,
                 const datetime::DayCounter& zeroDayCounter, std::vector<double>& fRow,
                 std::vector<double>& parentRow, std::vector<double>& discountRow,
                 const ViewT* parentWeights = nullptr, const ViewT* discountWeights = nullptr) {
    const datetime::Date effective = irs.start.serial() != 0 ? irs.start : referenceDate;
    detail::IrsRowScratch& scratch = detail::irsRowScratch();
    const datetime::Schedule& floatSchedule = scratch.schedules.get(
        effective, irs.maturity, irs.floatTenor, irs.floatCalendar, irs.businessDayConvention);
    const datetime::Schedule& fixedSchedule = scratch.schedules.get(
        effective, irs.maturity, irs.fixedTenor, irs.fixedCalendar, irs.businessDayConvention);
    const std::vector<datetime::Date>& floatDates = floatSchedule.dates();
    const std::vector<datetime::Date>& fixedDates = fixedSchedule.dates();
    const std::size_t nChild = child.size();
    const ViewT& parentWeightCurve = parentWeights != nullptr ? *parentWeights : forecastParent;
    const ViewT& discountWeightCurve = discountWeights != nullptr ? *discountWeights : discount;
    const std::size_t nParent = parentWeightCurve.size();
    const std::size_t nDiscount = discountWeightCurve.size();
    fRow.assign(nChild - 1, 0.0);
    parentRow.assign(nParent - 1, 0.0);
    discountRow.assign(nDiscount - 1, 0.0);
    const auto weightsAt = [](const ViewT& curve, double time, std::vector<double>& out) {
        if (time <= 0.0) {
            out.assign(curve.size(), 0.0);
            return;
        }
        curve.zeroNodeWeights(time, out);
    };
    std::vector<double>& weights = scratch.weights;
    std::vector<double>& annuityRow = scratch.annuityRow;
    annuityRow.assign(nDiscount - 1, 0.0);
    double annuity = 0.0;
    for (std::size_t j = 1; j < fixedDates.size(); ++j) {
        const double tau =
            datetime::yearFraction(fixedDates[j - 1], fixedDates[j], irs.fixedDayCounter);
        const datetime::Date payDate = irs.fixedCalendar.advance(
            fixedDates[j], datetime::Period(irs.paymentLag, datetime::TimeUnit::Days),
            irs.businessDayConvention);
        const double tPay = datetime::yearFraction(referenceDate, payDate, zeroDayCounter);
        const double discountPay = discount.discount(tPay);
        annuity += tau * discountPay;
        weightsAt(discountWeightCurve, tPay, weights);
        for (std::size_t i = 1; i < nDiscount; ++i) {
            annuityRow[i - 1] += -tau * tPay * discountPay * weights[i];
        }
    }
    if (!(annuity > 0.0)) {
        throw std::invalid_argument("irsSwapJacobianRows: non-positive fixed annuity");
    }
    std::vector<double>& discountNumerator = scratch.discountNumerator;
    std::vector<double>& childPrev = scratch.childPrev;
    std::vector<double>& childCur = scratch.childCur;
    std::vector<double>& parentPrev = scratch.parentPrev;
    std::vector<double>& parentCur = scratch.parentCur;
    std::vector<double>& payWeights = scratch.payWeights;
    discountNumerator.assign(nDiscount - 1, 0.0);
    double floatPv = 0.0;
    for (std::size_t k = 1; k < floatDates.size(); ++k) {
        const double tau =
            datetime::yearFraction(floatDates[k - 1], floatDates[k], irs.floatDayCounter);
        const double tPrev =
            datetime::yearFraction(referenceDate, floatDates[k - 1], zeroDayCounter);
        const double tCur = datetime::yearFraction(referenceDate, floatDates[k], zeroDayCounter);
        const datetime::Date payDate = irs.floatCalendar.advance(
            floatDates[k], datetime::Period(irs.paymentLag, datetime::TimeUnit::Days),
            irs.businessDayConvention);
        const double tPay = datetime::yearFraction(referenceDate, payDate, zeroDayCounter);
        const double discountPay = discount.discount(tPay);
        if (k == 1 && irs.firstCouponFixed) {
            const double couponPv = tau * discountPay * irs.firstCouponRate;
            floatPv += couponPv;
            weightsAt(discountWeightCurve, tPay, payWeights);
            for (std::size_t i = 1; i < nDiscount; ++i) {
                discountNumerator[i - 1] += -tPay * couponPv * payWeights[i];
            }
            continue;
        }
        const double childPrevious = child.discount(tPrev);
        const double childCurrent = child.discount(tCur);
        const double forward = (childPrevious / childCurrent - 1.0) / tau;
        floatPv += tau * discountPay * forward;
        const double ratio = childPrevious / childCurrent;
        weightsAt(child, tPrev, childPrev);
        weightsAt(child, tCur, childCur);
        weightsAt(parentWeightCurve, tPrev, parentPrev);
        weightsAt(parentWeightCurve, tCur, parentCur);
        weightsAt(discountWeightCurve, tPay, payWeights);
        for (std::size_t i = 1; i < nChild; ++i) {
            fRow[i - 1] +=
                (discountPay * ratio / annuity) * (-tPrev * childPrev[i] + tCur * childCur[i]);
        }
        for (std::size_t i = 1; i < nParent; ++i) {
            parentRow[i - 1] +=
                (discountPay * ratio / annuity) * (-tPrev * parentPrev[i] + tCur * parentCur[i]);
        }
        for (std::size_t i = 1; i < nDiscount; ++i) {
            discountNumerator[i - 1] += -tPay * tau * discountPay * forward * payWeights[i];
        }
    }
    const double denominator = annuity * annuity;
    for (std::size_t i = 1; i < nDiscount; ++i) {
        discountRow[i - 1] =
            (discountNumerator[i - 1] * annuity - floatPv * annuityRow[i - 1]) / denominator;
    }
}

} // namespace quantape::pricing
