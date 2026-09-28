#ifndef QUANTAPE_MATH_INTEGRALS_EXPONENTIAL_FITTING_LAGUERRE_TABLE_H
#define QUANTAPE_MATH_INTEGRALS_EXPONENTIAL_FITTING_LAGUERRE_TABLE_H

#include <cstddef>

namespace quantape::math::detail {
/**
 * @file ExponentialFittingLaguerreTable.h
 * @brief Generated EFGL table accessor (rows: omega | nodes | weights)
 *
 * Flat row-major storage: `efglTable()[row * efglColumnCount() + col]`,
 * column 0 = omega, columns 1..64 = nodes (descending), columns
 * 65..128 = raw weights (weightless integration). See the .cpp for
 * provenance and the generator reference.
 */
const double* efglTable();
std::size_t efglRowCount();
std::size_t efglColumnCount();
} // namespace quantape::math::detail

#endif // QUANTAPE_MATH_INTEGRALS_EXPONENTIAL_FITTING_LAGUERRE_TABLE_H
