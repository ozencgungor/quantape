#ifndef QP_SOLVER_H
#define QP_SOLVER_H

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <utility>
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
 * update).  The solver is a textbook two-phase primal active-set method:
 *
 *   Phase I   Finds a feasible point, or proves that there is none, by
 *             solving the L1-violation LP
 *                 min sum_i a_i   s.t.  A_i d + b_i <= a_i,  a_i >= 0
 *             (equality rows get both signs) with a dense Bland-rule simplex.
 *   Phase II  Runs the primal active-set method from that feasible point: the
 *             working set is always a maximal linearly independent set of
 *             active rows, every iteration solves the equality-constrained QP
 *             over the working set, follows the step until the first blocking
 *             constraint (Bland tie-break), and drops working inequalities
 *             whose multiplier is negative (Bland: smallest row index).
 *
 * Starting Phase II from a feasible point is what makes the blocking rule
 * sound: a non-working row has non-negative slack, so `alpha_i = -slack/ap`
 * with `ap > 0` is the exact fraction of the step at which that row reaches
 * zero, and never increases an existing violation.  (The previous version
 * seeded the working set with rows violated at d = 0, which made the initial
 * KKT system singular, could drop a violated row during the rank recovery and
 * then never re-add it because the blocking rule assumed feasibility.)
 *
 * `success` is only reported after the returned point and multipliers have
 * passed an explicit primal-feasibility, dual-feasibility and stationarity
 * check, so an infeasible QP always reports failure.
 *
 * Multiplier convention (unchanged): L = f + sum_i lambda_i (A_i d + b_i),
 * with lambda_i >= 0 for inequality rows and lambda_i free for equality rows.
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

/// Largest absolute entry (0 for an empty vector)
inline double qpMaxAbs(const std::vector<double>& v) {
    double m = 0.0;
    for (double x : v) {
        m = std::max(m, std::fabs(x));
    }
    return m;
}

/// Euclidean inner product of two equal-length dense vectors.  Four partial
/// sums break the serial accumulator dependency; the (harmless) reassociation
/// roughly triples the throughput of the working-set orthogonalization.
inline double qpDot(const std::vector<double>& a, const std::vector<double>& b) {
    const std::size_t n = a.size();
    double s0 = 0.0;
    double s1 = 0.0;
    double s2 = 0.0;
    double s3 = 0.0;
    std::size_t i = 0;
    for (; i + 4 <= n; i += 4) {
        s0 += a[i] * b[i];
        s1 += a[i + 1] * b[i + 1];
        s2 += a[i + 2] * b[i + 2];
        s3 += a[i + 3] * b[i + 3];
    }
    for (; i < n; ++i) {
        s0 += a[i] * b[i];
    }
    return (s0 + s1) + (s2 + s3);
}

/// An orthonormal basis of a subset of constraint rows (R^n).  Used to keep
/// the working set a basis: a candidate row is accepted only when its
/// component orthogonal to the span is not numerically negligible.
struct QpRowSpace {
    std::vector<std::vector<double>> rows;

    void clear() { rows.clear(); }

    /// Append `a` when it is independent of the current rows.
    bool tryAdd(const std::vector<double>& a, double rel_tol = 1e-10) {
        const auto norm2 = [](const std::vector<double>& v) {
            double sum = 0.0;
            for (double x : v) {
                sum += x * x;
            }
            return sum;
        };
        const double a_norm2 = norm2(a);
        if (!(a_norm2 > 0.0)) {
            return false;
        }
        std::vector<double> v = a;
        // One modified Gram-Schmidt pass is enough unless the candidate is
        // close to the span, where cancellation needs a second pass ("twice is
        // enough").  Re-orthogonalizing only in that regime keeps the common
        // independent-row case cheap.
        for (const auto& q : rows) {
            const double coefficient = qpDot(q, v);
            if (coefficient == 0.0) {
                continue;
            }
            for (std::size_t j = 0; j < v.size(); ++j) {
                v[j] -= coefficient * q[j];
            }
        }
        double v_norm2 = norm2(v);
        if (v_norm2 <= 0.25 * a_norm2) {
            for (const auto& q : rows) {
                const double coefficient = qpDot(q, v);
                if (coefficient == 0.0) {
                    continue;
                }
                for (std::size_t j = 0; j < v.size(); ++j) {
                    v[j] -= coefficient * q[j];
                }
            }
            v_norm2 = norm2(v);
        }
        if (v_norm2 <= rel_tol * rel_tol * a_norm2) {
            return false;
        }
        const double v_norm = std::sqrt(v_norm2);
        for (double& x : v) {
            x /= v_norm;
        }
        rows.push_back(std::move(v));
        return true;
    }
};

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

namespace detail {

/// Residual r_i = A_i d + b_i of one constraint row.
inline double qpRowResidual(const QpProblem& qp, std::size_t i, const std::vector<double>& d) {
    const std::size_t n = qp.dimension();
    double r = qp.b[i];
    for (std::size_t j = 0; j < n; ++j) {
        r += qp.A[i * n + j] * d[j];
    }
    return r;
}

/// Relative row scale used by the feasibility tests:
///   1 + |b_i| + sum_j |A_ij| |d_j|
/// so the tolerance tracks the size of the terms that make up the residual.
inline double qpRowScale(const QpProblem& qp, std::size_t i, const std::vector<double>& d) {
    const std::size_t n = qp.dimension();
    double scale = 1.0 + std::fabs(qp.b[i]);
    for (std::size_t j = 0; j < n; ++j) {
        scale += std::fabs(qp.A[i * n + j]) * std::fabs(d[j]);
    }
    return scale;
}

/// Primal feasibility check: A d + b <= tol for inequalities, |A d + b| <= tol
/// for equalities, with a per-row relative tolerance.  `max_violation` is the
/// largest residual in units of the row tolerance (<= 1 means feasible).
inline bool qpFeasible(const QpProblem& qp, const std::vector<double>& d, double rel_tol,
                       double& max_violation) {
    const std::size_t m = qp.rows();
    bool feasible = true;
    max_violation = 0.0;
    for (std::size_t i = 0; i < m; ++i) {
        const double r = qpRowResidual(qp, i, d);
        const double tol = rel_tol * qpRowScale(qp, i, d);
        const double negative_part = qp.equality[i] ? std::fabs(r) : std::max(0.0, r);
        max_violation = std::max(max_violation, negative_part / tol);
        if (negative_part > tol) {
            feasible = false;
        }
    }
    return feasible;
}

/// Phase-I LP: find d with A d + b <= 0 (inequalities) and A d + b = 0
/// (equalities).  Solves the L1-violation LP
///
///     min sum_i a_i   s.t.  A_i d + b_i <= a_i,  a_i >= 0
///
/// (equality rows contribute A_i d + b_i <= a_i and its negation) with a
/// dense simplex.  Returns true and writes a feasible `d` when the minimum is
/// zero; returns false (leaving `d` untouched) when the constraints conflict.
inline bool qpPhaseIFeasible(const QpProblem& qp, std::vector<double>& d) {
    const std::size_t n = qp.dimension();
    const std::size_t m = qp.rows();

    // Structural columns: [u(n) | v(n) | a(m) | slacks] with d = u - v and one
    // slack per converted row; then one artificial per row for the basis.
    std::size_t row_count = 0;
    for (std::size_t i = 0; i < m; ++i) {
        row_count += qp.equality[i] ? 2u : 1u;
    }
    if (row_count == 0) {
        d.assign(n, 0.0);
        return true;
    }
    const std::size_t structural = 2 * n + m + row_count;
    const std::size_t total = structural + row_count;

    std::vector<double> matrix(row_count * structural, 0.0);
    std::vector<double> rhs(row_count, 0.0);
    std::size_t r = 0;
    // One converted row: sum_j a_j d_j - a_i + slack = `rhs_value`, with
    // d = u - v and the direction flipped by `negate`.
    const auto add_row = [&](const double* a, std::size_t i, bool negate, double rhs_value) {
        for (std::size_t j = 0; j < n; ++j) {
            matrix[r * structural + j] = negate ? -a[j] : a[j];
            matrix[r * structural + n + j] = negate ? a[j] : -a[j];
        }
        matrix[r * structural + 2 * n + i] = -1.0; // a_i
        matrix[r * structural + 2 * n + m + r] = 1.0;
        rhs[r] = rhs_value;
        ++r;
    };
    for (std::size_t i = 0; i < m; ++i) {
        const double* a = qp.A.data() + i * n;
        if (!qp.equality[i]) {
            add_row(a, i, false, -qp.b[i]);
        } else {
            add_row(a, i, false, -qp.b[i]);
            add_row(a, i, true, qp.b[i]);
        }
    }

    // Orient every row so its right-hand side is non-negative; the artificial
    // column (added below) always supplies the initial +1 unit column.
    for (std::size_t i = 0; i < row_count; ++i) {
        if (rhs[i] < 0.0) {
            rhs[i] = -rhs[i];
            for (std::size_t c = 0; c < structural; ++c) {
                matrix[i * structural + c] = -matrix[i * structural + c];
            }
        }
    }

    // Tableau [matrix | artificial identity | rhs], basis = artificials.
    const std::size_t columns = structural + row_count + 1;
    std::vector<double> tableau(row_count * columns, 0.0);
    for (std::size_t i = 0; i < row_count; ++i) {
        for (std::size_t c = 0; c < structural; ++c) {
            tableau[i * columns + c] = matrix[i * structural + c];
        }
        tableau[i * columns + structural + i] = 1.0;
        tableau[i * columns + columns - 1] = rhs[i];
    }
    std::vector<std::size_t> basis(row_count);
    for (std::size_t i = 0; i < row_count; ++i) {
        basis[i] = structural + i;
    }
    const auto is_artificial = [&](std::size_t col) { return col >= structural; };
    // Phase-I objective: sum of the epigraph variables a_i.  The artificial
    // columns only provide the initial basis and vanish at the optimum.
    const auto column_cost = [&](std::size_t col) {
        return (col >= 2 * n && col < 2 * n + m) || is_artificial(col) ? 1.0 : 0.0;
    };

    // Dantzig's rule (most negative reduced cost) is much faster in practice;
    // after this many iterations we switch to Bland's rule, whose smallest-index
    // choice rules out cycling and guarantees finite termination.  The Phase-I
    // LP is bounded below by zero, and a basis with zero objective is already a
    // feasible point, so that case exits before optimality is reached.
    const double kReducedCostTol = 1e-12;
    const double kPivotTol = 1e-14;
    const double kRatioTie = 1e-15;
    const int bland_after = std::max(1000, 20 * static_cast<int>(row_count + total));
    const double objective_tol = 1e-9 * (1.0 + qpMaxAbs(qp.b));
    const int max_iterations = static_cast<int>(50 * (row_count + total) + 1000);
    bool done = false;
    for (int iter = 0; iter < max_iterations; ++iter) {
        double objective_now = 0.0;
        for (std::size_t i = 0; i < row_count; ++i) {
            objective_now += column_cost(basis[i]) * tableau[i * columns + columns - 1];
        }
        if (objective_now <= objective_tol) {
            done = true; // feasible point found; the LP optimum is not needed
            break;
        }

        // Reduced costs c_j - c_B' T[:,j]; only artificial basis columns cost 1.
        std::size_t entering = total;
        if (iter < bland_after) {
            double best_reduced = -kReducedCostTol;
            for (std::size_t c = 0; c < total; ++c) {
                double reduced = column_cost(c);
                for (std::size_t i = 0; i < row_count; ++i) {
                    reduced -= column_cost(basis[i]) * tableau[i * columns + c];
                }
                if (reduced < best_reduced) {
                    best_reduced = reduced;
                    entering = c;
                }
            }
        } else {
            for (std::size_t c = 0; c < total; ++c) {
                double reduced = column_cost(c);
                for (std::size_t i = 0; i < row_count; ++i) {
                    reduced -= column_cost(basis[i]) * tableau[i * columns + c];
                }
                if (reduced < -kReducedCostTol) {
                    entering = c;
                    break;
                }
            }
        }
        if (entering == total) {
            done = true;
            break;
        }

        // Ratio test with Bland tie-break on the leaving basic variable.
        double min_ratio = std::numeric_limits<double>::infinity();
        std::size_t leaving = row_count;
        for (std::size_t i = 0; i < row_count; ++i) {
            const double pivot = tableau[i * columns + entering];
            if (pivot <= kPivotTol) {
                continue;
            }
            const double ratio = tableau[i * columns + columns - 1] / pivot;
            if (leaving == row_count || ratio < min_ratio - kRatioTie ||
                (ratio <= min_ratio + kRatioTie && basis[i] < basis[leaving])) {
                min_ratio = ratio;
                leaving = i;
            }
        }
        if (leaving == row_count) {
            return false; // unbounded Phase-I LP: cannot happen, defensive
        }

        const double pivot = tableau[leaving * columns + entering];
        for (std::size_t c = 0; c < columns; ++c) {
            tableau[leaving * columns + c] /= pivot;
        }
        for (std::size_t i = 0; i < row_count; ++i) {
            if (i == leaving) {
                continue;
            }
            const double factor = tableau[i * columns + entering];
            if (factor == 0.0) {
                continue;
            }
            for (std::size_t c = 0; c < columns; ++c) {
                tableau[i * columns + c] -= factor * tableau[leaving * columns + c];
            }
        }
        basis[leaving] = entering;
    }

    if (!done) {
        return false; // Bland's rule guarantees termination; defensive only
    }
    // Optimal Phase-I objective = sum a_i (basic a and artificial values).
    double objective = 0.0;
    for (std::size_t i = 0; i < row_count; ++i) {
        objective += column_cost(basis[i]) * tableau[i * columns + columns - 1];
    }
    if (!(objective <= 1e-9 * (1.0 + qpMaxAbs(qp.b)))) {
        return false;
    }

    // Read the primal point out of the final basis.
    std::vector<double> values(total, 0.0);
    for (std::size_t i = 0; i < row_count; ++i) {
        values[basis[i]] = tableau[i * columns + columns - 1];
    }
    d.assign(n, 0.0);
    for (std::size_t j = 0; j < n; ++j) {
        d[j] = values[j] - values[n + j];
    }
    return true;
}

/// Solve the equality-constrained QP over the working set `ws`
///
///     min_p 0.5 p' B p + (B d + g)' p   s.t.  A_W p = -(A_W d + b)
///
/// through the KKT system  [B A_W'; A_W 0] [p; mu] = [-(B d + g); -r_W].
/// The multipliers `mu` follow the file convention L = f + sum mu (A d + b).
inline bool qpSolveEqp(const QpProblem& qp, const std::vector<double>& d,
                       const std::vector<std::size_t>& ws, std::vector<double>& p,
                       std::vector<double>& mu) {
    const std::size_t n = qp.dimension();
    const std::size_t k = ws.size();
    const std::size_t dim = n + k;
    std::vector<double> matrix(dim * dim, 0.0);
    std::vector<double> rhs(dim, 0.0);
    for (std::size_t i = 0; i < n; ++i) {
        double gradient = qp.g[i];
        for (std::size_t j = 0; j < n; ++j) {
            matrix[i * dim + j] = qp.B[i * n + j];
            gradient += qp.B[i * n + j] * d[j];
        }
        rhs[i] = -gradient;
    }
    for (std::size_t w = 0; w < k; ++w) {
        const std::size_t i = ws[w];
        const double residual = qpRowResidual(qp, i, d);
        for (std::size_t j = 0; j < n; ++j) {
            const double a = qp.A[i * n + j];
            matrix[(n + w) * dim + j] = a;
            matrix[j * dim + (n + w)] = a;
        }
        rhs[n + w] = -residual;
    }
    if (!solveDenseInPlace(matrix, rhs, dim)) {
        return false;
    }
    p.assign(rhs.begin(), rhs.begin() + static_cast<std::ptrdiff_t>(n));
    mu.assign(rhs.begin() + static_cast<std::ptrdiff_t>(n), rhs.end());
    return true;
}

/// Final KKT gate: primal feasibility, dual feasibility and stationarity.
inline bool qpVerifyKkt(const QpProblem& qp, const std::vector<double>& d,
                        const std::vector<double>& lambda, double rel_tol) {
    const std::size_t n = qp.dimension();
    const std::size_t m = qp.rows();
    double max_violation = 0.0;
    if (!qpFeasible(qp, d, rel_tol, max_violation)) {
        return false;
    }
    // Dual feasibility: only inequality rows are sign-constrained.
    const double lambda_scale = 1.0 + qpMaxAbs(lambda);
    for (std::size_t i = 0; i < m; ++i) {
        if (!qp.equality[i] && lambda[i] < -rel_tol * lambda_scale) {
            return false;
        }
    }
    // Stationarity: g + B d + A' lambda = 0.
    std::vector<double> stationarity = qp.g;
    double max_bd = 0.0;
    double max_atl = 0.0;
    for (std::size_t j = 0; j < n; ++j) {
        for (std::size_t i = 0; i < n; ++i) {
            const double term = qp.B[j * n + i] * d[i];
            stationarity[j] += term;
            max_bd = std::max(max_bd, std::fabs(term));
        }
    }
    for (std::size_t i = 0; i < m; ++i) {
        if (lambda[i] == 0.0) {
            continue;
        }
        for (std::size_t j = 0; j < n; ++j) {
            const double term = lambda[i] * qp.A[i * n + j];
            stationarity[j] += term;
            max_atl = std::max(max_atl, std::fabs(term));
        }
    }
    const double scale = 1.0 + qpMaxAbs(qp.g) + max_bd + max_atl;
    return qpMaxAbs(stationarity) <= rel_tol * scale;
}

} // namespace detail

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

    // Tolerances: the feasibility gate is relative to each row's own scale.
    constexpr double kFeasTol = 1e-9;
    constexpr double kKktTol = 1e-8;

    // ── Phase I: a feasible starting point ───────────────────────────────
    std::vector<double> d(n, 0.0);
    double max_violation = 0.0;
    if (!detail::qpFeasible(qp, d, kFeasTol, max_violation)) {
        if (!detail::qpPhaseIFeasible(qp, d)) {
            result.success = false;
            return result;
        }
        if (!detail::qpFeasible(qp, d, kFeasTol, max_violation)) {
            result.success = false;
            return result;
        }
    }

    // ── Phase II: primal active-set iterations from the feasible point ───
    //
    // The working set is a maximal linearly independent subset of the rows
    // that are active at d (all equalities plus active inequalities).  A row
    // is appended to the working set only when its normal is independent of
    // the current ones, so the KKT system below stays non-singular.
    std::vector<std::size_t> working;
    detail::QpRowSpace row_space;
    // Rebuild the working set as a maximal independent subset (index order).
    const auto rebuild_working_set = [&]() {
        std::vector<std::size_t> kept;
        row_space.clear();
        for (std::size_t i : working) {
            std::vector<double> row(qp.A.begin() + static_cast<std::ptrdiff_t>(i * n),
                                    qp.A.begin() + static_cast<std::ptrdiff_t>((i + 1) * n));
            if (row_space.tryAdd(row)) {
                kept.push_back(i);
            }
        }
        working = std::move(kept);
    };
    for (std::size_t pass = 0; pass < 2; ++pass) {
        for (std::size_t i = 0; i < m; ++i) {
            const bool want_equality = (pass == 0);
            if ((qp.equality[i] != 0) != want_equality) {
                continue;
            }
            const double residual = detail::qpRowResidual(qp, i, d);
            // A row is active when it sits on its boundary (small residual of
            // either sign); strictly satisfied rows must stay out of the
            // working set or the subproblem would be over-restricted.
            const bool active =
                want_equality || std::fabs(residual) <= kFeasTol * detail::qpRowScale(qp, i, d);
            if (!active) {
                continue;
            }
            std::vector<double> row(qp.A.begin() + static_cast<std::ptrdiff_t>(i * n),
                                    qp.A.begin() + static_cast<std::ptrdiff_t>((i + 1) * n));
            if (row_space.tryAdd(row)) {
                working.push_back(i);
            }
        }
    }

    const int max_iterations = static_cast<int>(50 * (n + m + 1));
    std::vector<double> p;
    std::vector<double> mu;
    for (int iter = 0; iter < max_iterations; ++iter) {
        if (!detail::qpSolveEqp(qp, d, working, p, mu)) {
            // Numerically dependent working set: re-select a basis.  This can
            // only happen through roundoff (tryAdd enforces independence), and
            // dropping an active row cannot lose feasibility because Phase II
            // only ever drops rows that are active at the current point.
            const std::size_t before = working.size();
            rebuild_working_set();
            if (working.size() == before) {
                result.success = false;
                return result;
            }
            continue;
        }

        // Optimal over the working set: check the multipliers and either leave
        // at a KKT point or drop a working inequality (Bland: smallest index).
        const double step_norm = detail::qpMaxAbs(p);
        if (step_norm <= 1e-11 * (1.0 + detail::qpMaxAbs(d))) {
            const double mu_scale = 1.0 + detail::qpMaxAbs(mu);
            std::size_t leaving = m;
            for (std::size_t w = 0; w < working.size(); ++w) {
                const std::size_t i = working[w];
                if (qp.equality[i] || mu[w] >= -1e-10 * mu_scale) {
                    continue;
                }
                if (leaving == m || i < leaving) {
                    leaving = i;
                }
            }
            if (leaving == m) {
                // Candidate KKT point: accept only after explicit verification.
                result.d = d;
                result.lambda.assign(m, 0.0);
                for (std::size_t w = 0; w < working.size(); ++w) {
                    result.lambda[working[w]] = mu[w];
                }
                result.success = detail::qpVerifyKkt(qp, result.d, result.lambda, kKktTol);
                result.iterations = iter + 1;
                return result;
            }
            working.erase(std::remove(working.begin(), working.end(), leaving), working.end());
            rebuild_working_set();
            continue;
        }

        // Ratio test: the first non-working row that blocks the step.  Because
        // d is feasible, slack >= 0 and the step fraction cannot move any row
        // into violation.
        double alpha = 1.0;
        std::size_t blocking = m;
        std::vector<std::size_t> binding;
        for (std::size_t i = 0; i < m; ++i) {
            if (std::find(working.begin(), working.end(), i) != working.end()) {
                continue;
            }
            double ap = 0.0;
            double row_scale = 0.0;
            for (std::size_t j = 0; j < n; ++j) {
                ap += qp.A[i * n + j] * p[j];
                row_scale += std::fabs(qp.A[i * n + j]);
            }
            if (ap <= 1e-13 * (1.0 + row_scale * step_norm)) {
                continue;
            }
            const double slack = detail::qpRowResidual(qp, i, d);
            const double alpha_i = -slack / ap;
            if (alpha_i <= 1e-12) {
                binding.push_back(i); // already (numerically) active
                continue;
            }
            // A row only blocks when it is reached before the full step to the
            // equality-constrained minimizer (alpha = 1).
            if (alpha_i >= 1.0 - 1e-13) {
                continue;
            }
            if (blocking == m || alpha_i < alpha - 1e-14) {
                alpha = alpha_i;
                blocking = i;
            } else if (alpha_i <= alpha + 1e-14 && i < blocking) {
                blocking = i;
            }
        }

        // Add every currently-active blocker at once (a zero step that only
        // tightens the working set); rows whose normal is dependent are left
        // out -- they are implied by the working set and stay satisfied.
        std::vector<std::size_t> accepted;
        for (std::size_t i : binding) {
            std::vector<double> row(qp.A.begin() + static_cast<std::ptrdiff_t>(i * n),
                                    qp.A.begin() + static_cast<std::ptrdiff_t>((i + 1) * n));
            if (row_space.tryAdd(row)) {
                working.push_back(i);
                accepted.push_back(i);
            }
        }
        if (!accepted.empty()) {
            continue;
        }
        if (blocking != m) {
            std::vector<double> row(qp.A.begin() + static_cast<std::ptrdiff_t>(blocking * n),
                                    qp.A.begin() + static_cast<std::ptrdiff_t>((blocking + 1) * n));
            for (std::size_t j = 0; j < n; ++j) {
                d[j] += alpha * p[j];
            }
            if (row_space.tryAdd(row)) {
                working.push_back(blocking);
            }
            continue;
        }

        // Unblocked: move to the equality-constrained minimizer.
        for (std::size_t j = 0; j < n; ++j) {
            d[j] += p[j];
        }
    }

    result.success = false;
    result.d = std::move(d);
    return result;
}

} // namespace quantape::math

#endif // QP_SOLVER_H
