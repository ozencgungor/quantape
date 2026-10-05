#ifndef QUANTAPE_MATH_TRIANGULAR_SOLVE_H
#define QUANTAPE_MATH_TRIANGULAR_SOLVE_H

#include <cstddef>
#include <span>
#include <stdexcept>
#include <vector>

namespace quantape::math {
/**
 * @file TriangularSolve.h
 * @brief Dense triangular solves for the curve risk back-substitution
 *
 * Row-major storage; coefficients are `double` (interpolation weights and
 * instrument Jacobians are always passive), right-hand sides are `DoubleT`
 * so the same solve works on AD adjoints.
 */

/// Solve `L x = b` for a row-major lower-triangular `n x n` matrix.
template <typename DoubleT>
std::vector<DoubleT> solveLowerTriangular(std::span<const double> l, std::size_t n,
                                          const std::vector<DoubleT>& b,
                                          bool unitDiagonal = false) {
    if (l.size() != n * n || b.size() != n) {
        throw std::invalid_argument("solveLowerTriangular: size mismatch");
    }
    std::vector<DoubleT> x(n);
    for (std::size_t i = 0; i < n; ++i) {
        DoubleT sum = b[i];
        for (std::size_t k = 0; k < i; ++k) {
            sum -= l[i * n + k] * x[k];
        }
        const double diagonal = l[i * n + i];
        if (!unitDiagonal) {
            if (diagonal == 0.0) {
                throw std::invalid_argument("solveLowerTriangular: zero pivot");
            }
            sum /= diagonal;
        }
        x[i] = sum;
    }
    return x;
}

/// Solve `L^T x = b` for a row-major lower-triangular `n x n` matrix.
template <typename DoubleT>
std::vector<DoubleT> solveLowerTranspose(std::span<const double> l, std::size_t n,
                                         const std::vector<DoubleT>& b,
                                         bool unitDiagonal = false) {
    if (l.size() != n * n || b.size() != n) {
        throw std::invalid_argument("solveLowerTranspose: size mismatch");
    }
    std::vector<DoubleT> x(n);
    for (std::size_t ii = n; ii-- > 0;) {
        DoubleT sum = b[ii];
        for (std::size_t k = ii + 1; k < n; ++k) {
            sum -= l[k * n + ii] * x[k];
        }
        const double diagonal = l[ii * n + ii];
        if (!unitDiagonal) {
            if (diagonal == 0.0) {
                throw std::invalid_argument("solveLowerTranspose: zero pivot");
            }
            sum /= diagonal;
        }
        x[ii] = sum;
    }
    return x;
}

} // namespace quantape::math

#endif // QUANTAPE_MATH_TRIANGULAR_SOLVE_H
