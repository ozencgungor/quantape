#ifndef QP_SOLVER_H
#define QP_SOLVER_H

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <vector>

namespace quantape::math {
/**
 * @file QpSolver.h
 * @brief Small dense active-set QP solver for SLSQP/AUGLAG (Stan-free)
 *
 * Solves
 *   minimize   0.5 d' B d + g' d
 *   subject to A d + b <= 0   (inequality rows)
 *              A d + b  = 0   (rows flagged as equalities)
 *
 * B is symmetric positive definite (SLSQP keeps it so with a damped BFGS
 * update). The method is the classic primal active-set scheme: the working
 * set starts with the equalities and the inequalities violated at d = 0,
 * each iteration solves the equality-constrained KKT system over the working
 * set, steps toward that solution until a new constraint blocks, adds it,
 * and drops working inequalities with a negative multiplier. Dense Gaussian
 * elimination is used for the KKT systems; sizes are small (n + active).
 */

namespace detail {
/// Solve A x = b in place (A row-major n x n, b becomes x); false if singular
inline bool solveDenseInPlace(std::vector<double>& A, std::vector<double>& b, std::size_t n) {
    constexpr double kPivotTol = 1e-14;
    for (std::size_t col = 0; col < n; ++col) {
        std::size_t pivot = col;
        double best = std::fabs(A[col * n + col]);
        for (std::size_t r = col + 1; r < n; ++r) {
            const double candidate = std::fabs(A[r * n + col]);
            if (candidate > best) {
                best = candidate;
                pivot = r;
            }
        }
        if (best <= kPivotTol) {
            return false;
        }
        if (pivot != col) {
            for (std::size_t c = 0; c < n; ++c) {
                std::swap(A[col * n + c], A[pivot * n + c]);
            }
            std::swap(b[col], b[pivot]);
        }
        const double inv_pivot = 1.0 / A[col * n + col];
        for (std::size_t r = col + 1; r < n; ++r) {
            const double factor = A[r * n + col] * inv_pivot;
            if (factor == 0.0) {
                continue;
            }
            for (std::size_t c = col; c < n; ++c) {
                A[r * n + c] -= factor * A[col * n + c];
            }
            b[r] -= factor * b[col];
        }
    }
    for (std::size_t i = n; i-- > 0;) {
        double sum = b[i];
        for (std::size_t c = i + 1; c < n; ++c) {
            sum -= A[i * n + c] * b[c];
        }
        if (std::fabs(A[i * n + i]) <= kPivotTol) {
            return false;
        }
        b[i] = sum / A[i * n + i];
    }
    return true;
}

} // namespace detail

/// QP data (dense; n = variable count, m = constraint rows).
/// A is FLAT row-major m x n: one contiguous buffer so the solver never
/// allocates per-row storage and the caller can reuse it across iterations.
struct QpProblem {
    std::vector<double> g;      ///< linear term, size n
    std::vector<double> B;      ///< Hessian, n x n row-major, SPD
    std::vector<double> A;      ///< constraint matrix, m x n row-major
    std::vector<double> b;      ///< constraint offsets, size m
    std::vector<char> equality; ///< per-row equality flag

    std::size_t dimension() const { return g.size(); }
    std::size_t rows() const { return dimension() ? A.size() / dimension() : 0; }
};

struct QpResult {
    std::vector<double> d;      ///< primal solution
    std::vector<double> lambda; ///< multipliers (>= 0 for inequalities)
    bool success = false;
    int iterations = 0;
};

/// Active-set QP solve; see the file comment for the algorithm
inline QpResult solveActiveSetQp(const QpProblem& qp) {
    const std::size_t n = qp.dimension();
    const std::size_t m = qp.rows();

    QpResult result;
    result.d.assign(n, 0.0);
    result.lambda.assign(m, 0.0);
    if (n == 0) {
        result.success = true;
        return result;
    }

    // Unconstrained: solve B d = -g directly
    if (m == 0) {
        std::vector<double> M = qp.B;
        std::vector<double> rhs(n);
        for (std::size_t j = 0; j < n; ++j) {
            rhs[j] = -qp.g[j];
        }
        if (quantape::math::detail::solveDenseInPlace(M, rhs, n)) {
            result.d = std::move(rhs);
            result.success = true;
        }
        result.iterations = 1;
        return result;
    }

    // Working set: equalities plus inequalities violated at d = 0
    std::vector<char> in_work(m, 0);
    for (std::size_t i = 0; i < m; ++i) {
        if (qp.equality[i] || qp.b[i] > 1e-12 * (1.0 + std::fabs(qp.b[i]))) {
            in_work[i] = 1;
        }
    }

    const int max_iterations = static_cast<int>(20 * (n + m + 1));
    std::vector<double> d(0.0); // placeholder; replaced below
    d.assign(n, 0.0);

    for (int iter = 0; iter < max_iterations; ++iter) {
        // Assemble the working set
        std::vector<std::size_t> active;
        active.reserve(m);
        for (std::size_t i = 0; i < m; ++i) {
            if (in_work[i]) {
                active.push_back(i);
            }
        }

        // Build and solve the KKT system over a given working set;
        // `regularize` adds a small diagonal to the B block as a last resort.
        const auto build_and_solve = [&](const std::vector<std::size_t>& wset,
                                         std::vector<double>& sol, bool regularize = false) {
            const std::size_t wk = wset.size();
            const std::size_t wdim = n + wk;
            std::vector<double> M(wdim * wdim, 0.0);
            std::vector<double> rhs2(wdim, 0.0);
            for (std::size_t i = 0; i < n; ++i) {
                for (std::size_t j = 0; j < n; ++j) {
                    M[i * wdim + j] = qp.B[i * n + j];
                }
                if (regularize) {
                    M[i * wdim + i] += 1e-10;
                }
                rhs2[i] = -qp.g[i];
            }
            for (std::size_t wi = 0; wi < wk; ++wi) {
                const std::size_t ii = wset[wi];
                for (std::size_t j = 0; j < n; ++j) {
                    const double a = qp.A[ii * n + j];
                    M[(n + wi) * wdim + j] = a;
                    M[j * wdim + (n + wi)] = a;
                }
                rhs2[n + wi] = -qp.b[ii];
            }
            if (!quantape::math::detail::solveDenseInPlace(M, rhs2, wdim)) {
                return false;
            }
            sol = std::move(rhs2);
            return true;
        };

        std::vector<double> solution;
        if (!build_and_solve(active, solution)) {
            // Linearly dependent active rows (monotone + butterfly rows on a
            // flat arbitrage curve are dependent): drop dependent rows one at
            // a time, inequalities first, equalities only as a last resort.
            bool resolved = false;
            for (int pass = 0; pass < 2 && !resolved; ++pass) {
                const bool want_equality = (pass == 1);
                for (std::size_t wi = 0; wi < active.size() && !resolved; ++wi) {
                    const std::size_t ii = active[wi];
                    if ((qp.equality[ii] != 0) != want_equality) {
                        continue;
                    }
                    std::vector<std::size_t> reduced;
                    reduced.reserve(active.size() - 1);
                    for (std::size_t wj = 0; wj < active.size(); ++wj) {
                        if (wj != wi) {
                            reduced.push_back(active[wj]);
                        }
                    }
                    if (build_and_solve(reduced, solution)) {
                        in_work[ii] = 0; // keep the dependent row dropped
                        active = std::move(reduced);
                        resolved = true;
                    }
                }
            }
            if (!resolved && !build_and_solve(active, solution, /*regularize=*/true)) {
                result.success = false;
                return result;
            }
        }

        const std::size_t k = active.size();
        std::vector<double> d_new(solution.begin(),
                                  solution.begin() + static_cast<std::ptrdiff_t>(n));
        std::vector<double> lam_w(solution.begin() + static_cast<std::ptrdiff_t>(n),
                                  solution.end());

        // Distance from the current point to the EQP solution; when it is zero
        // the point is optimal for the current working set.
        std::vector<double> p(n);
        double step_norm = 0.0;
        for (std::size_t j = 0; j < n; ++j) {
            p[j] = d_new[j] - d[j];
            step_norm = std::max(step_norm, std::fabs(p[j]));
        }

        if (step_norm <= 1e-12) {
            // Candidate optimum: drop the most negative inequality multiplier
            std::size_t worst = m;
            double worst_value = 0.0;
            for (std::size_t wi = 0; wi < k; ++wi) {
                const std::size_t i = active[wi];
                if (!qp.equality[i] && lam_w[wi] < worst_value) {
                    worst = i;
                    worst_value = lam_w[wi];
                }
            }
            if (worst == m) {
                result.d = d_new;
                for (std::size_t wi = 0; wi < k; ++wi) {
                    result.lambda[active[wi]] = lam_w[wi];
                }
                result.success = true;
                result.iterations = iter + 1;
                return result;
            }
            in_work[worst] = 0;
            continue;
        }

        // Step from the current point toward the EQP solution; find blockers.
        // Rows that are already binding (zero remaining slack) all enter the
        // working set at once: adding them one at a time turns N identical
        // bounds into N cubic KKT solves.
        const auto row_dot = [&](std::size_t i, const std::vector<double>& v) {
            double sum = 0.0;
            for (std::size_t j = 0; j < n; ++j) {
                sum += qp.A[i * n + j] * v[j];
            }
            return sum;
        };

        double alpha = 1.0;
        std::size_t blocking = m;
        std::vector<std::size_t> binding;
        for (std::size_t i = 0; i < m; ++i) {
            if (in_work[i]) {
                continue;
            }
            const double ap = row_dot(i, p);
            if (ap <= 1e-14) {
                continue;
            }
            const double slack = qp.b[i] + row_dot(i, d);
            const double alpha_i = -slack / ap;
            if (alpha_i <= 1e-14 * (1.0 + std::fabs(alpha_i))) {
                binding.push_back(i);
                continue;
            }
            if (alpha_i < alpha) {
                alpha = std::max(0.0, alpha_i);
                blocking = i;
            }
        }

        if (!binding.empty()) {
            for (std::size_t i : binding) {
                in_work[i] = 1;
            }
            continue;
        }
        if (blocking != m) {
            for (std::size_t j = 0; j < n; ++j) {
                d[j] += alpha * p[j];
            }
            in_work[blocking] = 1;
            continue;
        }

        d = std::move(d_new);
    }

    result.success = false;
    return result;
}

} // namespace quantape::math

#endif // QP_SOLVER_H
