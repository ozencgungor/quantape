#ifndef TRIDIAGONAL_SOLVER_H
#define TRIDIAGONAL_SOLVER_H

#include <cstddef>
#include <stdexcept>
#include <vector>

#include "SolverPrimitives.h"

namespace Math {
/**
 * @brief Thomas algorithm for a tridiagonal linear system
 *
 * Solves M x = d where M has sub-diagonal a, diagonal b and super-diagonal c:
 *
 *   | b0 c0  0 ... | | x0 |   | d0 |
 *   | a1 b1 c1 ... | | x1 | = | d1 |
 *   |  0 a2 b2 ... | | x2 |   | d2 |
 *
 * The first element of a and the last element of c are unused (set them to
 * any value, conventionally 0). Works for double and AD scalars, so it can be
 * used inside AAD kernels (the forward elimination carries the AD graph).
 *
 * @tparam DoubleT Numeric type (double or an AD scalar exposing val())
 */
template <typename DoubleT>
    requires SolverScalar<DoubleT>
class TridiagonalSolver {
public:
    /**
     * @param a Sub-diagonal (size n)
     * @param b Diagonal (size n)
     * @param c Super-diagonal (size n)
     * @param d Right-hand side (size n)
     * @return Solution vector x (size n)
     */
    static std::vector<DoubleT> solve(const std::vector<DoubleT>& a, const std::vector<DoubleT>& b,
                                      const std::vector<DoubleT>& c,
                                      const std::vector<DoubleT>& d) {
        const std::size_t n = d.size();
        if (n == 0) {
            return std::vector<DoubleT>();
        }
        if (a.size() != n || b.size() != n || c.size() != n) {
            throw std::runtime_error("Invalid input sizes for tridiagonal solver");
        }

        std::vector<DoubleT> c_prime(n);
        std::vector<DoubleT> d_prime(n);
        std::vector<DoubleT> x(n);

        c_prime[0] = c[0] / b[0];
        d_prime[0] = d[0] / b[0];

        for (std::size_t i = 1; i < n; ++i) {
            const DoubleT denominator = b[i] - a[i] * c_prime[i - 1];
            if (detail::primalValue(denominator) == 0.0) {
                throw std::runtime_error("TridiagonalSolver: zero pivot");
            }
            const DoubleT m = DoubleT(1.0) / denominator;
            c_prime[i] = c[i] * m;
            d_prime[i] = (d[i] - a[i] * d_prime[i - 1]) * m;
        }

        x[n - 1] = d_prime[n - 1];
        for (std::size_t i = n - 1; i-- > 0;) {
            x[i] = d_prime[i] - c_prime[i] * x[i + 1];
        }

        return x;
    }
};
} // namespace Math

#endif // TRIDIAGONAL_SOLVER_H
