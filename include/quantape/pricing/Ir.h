#pragma once

#include "quantape/datetime/Date.h"
#include "quantape/datetime/TimeConversion.h"
#include "quantape/instruments/Cashflow.h"
#include "quantape/instruments/IrInstruments.h"
#include "quantape/pricing/IrMath.h"

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

} // namespace quantape::pricing
