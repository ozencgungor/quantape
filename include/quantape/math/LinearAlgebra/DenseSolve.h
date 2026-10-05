#ifndef QUANTAPE_MATH_DENSE_SOLVE_H
#define QUANTAPE_MATH_DENSE_SOLVE_H

#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <utility>
#include <vector>

namespace quantape::math {
/**
 * @file DenseSolve.h
 * @brief Small dense linear solve (row-major, partial pivoting)
 *
 * Used by the curve risk transform where the instrument Jacobian can carry
 * stencil spillover (Akima/tension splines) and is therefore not strictly
 * triangular. The matrix is copied; `n <= ~100` in this use.
 */

/// Solve `A x = b` for a row-major `n x n` matrix (throws when singular).
inline std::vector<double> solveDense(std::vector<double> a, std::size_t n,
                                      const std::vector<double>& b) {
    if (a.size() != n * n || b.size() != n) {
        throw std::invalid_argument("solveDense: size mismatch");
    }
    std::vector<double> x = b;
    for (std::size_t k = 0; k < n; ++k) {
        std::size_t pivot = k;
        double best = std::abs(a[k * n + k]);
        for (std::size_t i = k + 1; i < n; ++i) {
            const double candidate = std::abs(a[i * n + k]);
            if (candidate > best) {
                best = candidate;
                pivot = i;
            }
        }
        if (!(best > 0.0)) {
            throw std::invalid_argument("solveDense: singular matrix");
        }
        if (pivot != k) {
            for (std::size_t j = k; j < n; ++j) {
                std::swap(a[k * n + j], a[pivot * n + j]);
            }
            std::swap(x[k], x[pivot]);
        }
        const double diagonal = a[k * n + k];
        for (std::size_t i = k + 1; i < n; ++i) {
            const double factor = a[i * n + k] / diagonal;
            if (factor == 0.0) {
                continue;
            }
            for (std::size_t j = k; j < n; ++j) {
                a[i * n + j] -= factor * a[k * n + j];
            }
            x[i] -= factor * x[k];
        }
    }
    for (std::size_t ii = n; ii-- > 0;) {
        double sum = x[ii];
        for (std::size_t j = ii + 1; j < n; ++j) {
            sum -= a[ii * n + j] * x[j];
        }
        x[ii] = sum / a[ii * n + ii];
    }
    return x;
}

} // namespace quantape::math

#endif // QUANTAPE_MATH_DENSE_SOLVE_H
