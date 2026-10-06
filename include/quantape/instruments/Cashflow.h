#pragma once

#include "quantape/datetime/Date.h"

#include <array>
#include <cstddef>
#include <string>
#include <string_view>

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
 *
 * Currency codes cross into string-keyed vocabulary (`CurveKey`,
 * `FXDescriptor`) only at the configuration and materialization boundaries,
 * through `currencyFromCode` and `currencyCode`; pricing and bootstrap code
 * carries the POD.
 */

/// ISO-style three-character currency code held inline.
struct Currency {
    std::array<char, 3> code{};
};

/// Converts a string-keyed currency code to the inline POD, padding or
/// truncating to three characters.
inline Currency currencyFromCode(std::string_view code) {
    Currency out;
    for (std::size_t i = 0; i < out.code.size(); ++i) {
        out.code[i] = i < code.size() ? code[i] : ' ';
    }
    return out;
}

/// Converts the inline currency POD back to a string-keyed code.
inline std::string currencyCode(const Currency& currency) {
    return std::string(currency.code.data(), currency.code.size());
}

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
