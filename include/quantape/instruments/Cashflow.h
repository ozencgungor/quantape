#pragma once

#include "quantape/datetime/Date.h"

#include <array>

namespace quantape::instruments {
/**
 * @file Cashflow.h
 * @brief Scalar-templated cashflow vocabulary for materialized payoffs
 *
 * A cashflow is plain data: a payment date, an inline three-character
 * currency and a scalar amount. The amount is templated so the same record
 * serves `double`, reverse-mode and forward-mode evaluation without erasing
 * the instrument type; the fixed-size currency keeps it trivially copyable
 * and allocation-free.
 */

/// ISO-style three-character currency code held inline.
struct Currency {
    std::array<char, 3> code{};
};

/// One cashflow: payment date, currency and amount in that currency.
/// @tparam ScalarT Numeric type (`double` or an AD scalar).
template <typename ScalarT>
struct CashflowT {
    datetime::Date payDate{};
    Currency currency{};
    ScalarT amount{};
};

/// Deterministic cashflow record.
using Cashflow = CashflowT<double>;

} // namespace quantape::instruments
