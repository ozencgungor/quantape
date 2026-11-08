#ifndef TRIDIAGONAL_SOLVER_H
#define TRIDIAGONAL_SOLVER_H

#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

#include "SolverPrimitives.h"

namespace quantape::math {
namespace detail {
/// Relative pivot floor of the Thomas algorithm.
///
/// The solver does NOT pivot, so a pivot (including the first one, b[0]) whose
/// magnitude is at or below this fraction of the row entries that formed it is
/// indistinguishable from an exact zero in double precision: the forward
/// elimination's division would either produce NaN/Inf or amplify round-off
/// without bound. Such systems are rejected (documented failure mode) instead
/// of returning a silently wrong solution. The test is relative to the row, so
/// uniformly scaled (well-conditioned) systems still solve unchanged.
inline constexpr double kTridiagonalPivotTolerance = 1e-12;
} // namespace detail

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
 * No pivoting is performed. Every pivot is validated before it is inverted
 * (including b[0], which the classic formulation divides by unchecked): a
 * pivot that is non-finite or negligible against its row throws
 * std::runtime_error, so the documented failure mode for a singular or
 * near-singular system is an exception, never NaN/Inf or an amplified
 * solution.
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

        // First pivot: b[0]. For n > 1 the row that forms it is [b[0], c[0]],
        // so the row scale is |b[0]| + |c[0]|; for a 1x1 system only b[0]
        // matters (c[0] is unused, so it must not influence the test and a
        // tiny-but-correct 1x1 division stays allowed).
        const double scale0 = std::fabs(quantape::math::detail::primalValue(b[0])) +
                              (n > 1 ? std::fabs(quantape::math::detail::primalValue(c[0])) : 0.0);
        checkPivot(quantape::math::detail::primalValue(b[0]), scale0, 0);

        c_prime[0] = c[0] / b[0];
        d_prime[0] = d[0] / b[0];

        for (std::size_t i = 1; i < n; ++i) {
            const DoubleT denominator = b[i] - a[i] * c_prime[i - 1];
            // Row scale of the elimination step: the magnitudes of the two
            // terms whose difference is the pivot. A pivot far below it means
            // catastrophic cancellation in this (unpivoted) elimination.
            const double scale = std::fabs(quantape::math::detail::primalValue(b[i])) +
                                 std::fabs(quantape::math::detail::primalValue(a[i])) *
                                     std::fabs(quantape::math::detail::primalValue(c_prime[i - 1]));
            checkPivot(quantape::math::detail::primalValue(denominator), scale, i);
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

    /**
     * @brief Solve with passive double coefficients and an AD right-hand side
     *
     * Convenience overload for callers whose diagonals are grid data (pure
     * double) while the right-hand side carries the AD leaves, e.g. the
     * tension spline node values. The diagonals are promoted to DoubleT
     * internally; the all-DoubleT overload above keeps working for fully AD
     * coefficient vectors. Constrained out for DoubleT = double, where it
     * would duplicate the signature above.
     */
    static std::vector<DoubleT> solve(const std::vector<double>& a, const std::vector<double>& b,
                                      const std::vector<double>& c, const std::vector<DoubleT>& d)
        requires(!std::is_same_v<DoubleT, double>)
    {
        return solve(std::vector<DoubleT>(a.begin(), a.end()),
                     std::vector<DoubleT>(b.begin(), b.end()),
                     std::vector<DoubleT>(c.begin(), c.end()), d);
    }

private:
    /// Reject a pivot that cannot be trusted in double precision.
    ///
    /// `pivot` is its primal value and `scale` the magnitude of the row terms
    /// that formed it. Non-finite pivots (NaN/Inf) and pivots at or below
    /// kTridiagonalPivotTolerance * scale are unrecoverable without pivoting.
    static void checkPivot(double pivot, double scale, std::size_t index) {
        if (!std::isfinite(pivot) || !std::isfinite(scale) ||
            !(std::fabs(pivot) > quantape::math::detail::kTridiagonalPivotTolerance * scale)) {
            throw std::runtime_error(
                "TridiagonalSolver: singular or ill-conditioned system: pivot " +
                std::to_string(index) +
                " is not finite or is negligible against its row (no pivoting is performed)");
        }
    }
};
} // namespace quantape::math

#endif // TRIDIAGONAL_SOLVER_H
