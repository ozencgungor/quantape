#ifndef LIGHTAPE_EIGEN_MATRIX_BLOCKS_H
#define LIGHTAPE_EIGEN_MATRIX_BLOCKS_H

#include <Eigen/Core>

#include <cstddef>
#include <cstdint>
#include <type_traits>
#include <vector>

#include "lightape/eigen/LeanReverseEigen.h"
#include "lightape/ops/Callbacks.h"

namespace lightape {
/**
 * @file MatrixBlocks.h
 * @brief Matrix-block tier for the lean reverse tape: one `External` node per
 *        Eigen block instead of one node per scalar
 *
 * The functions here flatten an Eigen expression of `RevScalar` (possibly
 * mixed with plain `double` coefficients) into a `std::vector<RevScalar>`
 * snapshot and record **one** `External` tape node (`RevTape::pushExternal`)
 * over that argument vector. The tape's external-argument cache stores the
 * `{value, slot}` pair of every coefficient, so the reverse sweep never
 * touches Eigen and never re-reads the caller's matrix. The node count per
 * block is 1 (or 0 when every coefficient is a plain constant,
 * `RevTape::kNone`, because `makeExternal` collapses constant-only blocks).
 *
 * The primal value is computed from a scratch `double` buffer with the plain
 * Eigen double kernels, so recording a block never appends scalar nodes.
 *
 * ## Exact first and second order
 *
 * `dot`, `sum` and `matrixVectorProduct` (value = `(A*x).sum()`) are
 * multilinear in their coefficients:
 * - `dot(x, y)` is bilinear: `dF/dx_i = y_i`, `dF/dy_i = x_i`, and the mixed
 *   second derivative is the delta pattern `d2F/dx_i dy_j = delta_ij`
 *   (same-vector pairs are 0). Implemented as the delta-pattern `vhp` (the
 *   `bArgs` mechanism of `RevTape::hessianVector`), so the replay HVP is
 *   exact (decision #26: second order is never approximated).
 * - `sum(x)` is linear: unit `vjp`, `vhp == 0` exactly.
 * - `matrixVectorProduct(A, x)` = `sum_ij A_ij x_j` is bilinear:
 *   `dF/dA_ij = x_j`, `dF/dx_j = sum_i A_ij`; the cross Hessian is the
 *   delta-pattern encoded in `matvecBlockVhp`.
 * Every block records `vjp`, `svp` and `vhp`, so `hessianVector` never stops
 * at these nodes (a missing `vhp` would fail loudly by design).
 *
 * ## Cost and the wide-block route
 *
 * Recording is O(k) `ExtArg` entries (16 B each: value + slot) in the tape's
 * external-argument cache plus one 48 B node, independent of the flop count:
 * a matrix-vector contraction collapses its whole reduction tree into that
 * one node. For k beyond a few hundred coefficients the O(k) argument cache
 * is the dominant cost of this tier. The wide/batched route is the
 * `BatchTape<W>` SoA sweep over one shared structure (decisions #9/#14,
 * `codegen_simd.md` section 3(e)) and the plane-backed block variant
 * (`matrix_vjp_jvp.md` section 2.2, B2p); this header is the semantics bridge
 * and the differential oracle for those layers, not the asymptotically
 * optimal form.
 *
 * ## Lifetime (arena_matrix / to_ref discipline, decision #16)
 *
 * A block result is a `RevScalar` = (value, tape slot) handle. The `External`
 * node references only slots plus tape-owned argument/payload copies, so the
 * Eigen arguments may die immediately after the call. Conversely, never keep
 * an Eigen expression built from AD scalars across a recording scope: nested
 * expressions hold `RevScalar` slots that dangle after `clear()`,
 * `resetTo()` or `releaseCurrentThread()` and would silently alias reused
 * slots. Materialize into a concrete `Eigen::Matrix<RevScalar>` before the
 * scope ends, exactly as Stan's `arena_matrix`/`to_ref` discipline requires
 * for cheap temporaries, and re-record rather than reusing an expression. For
 * the same Eigen-5 readiness reason `.select()` is not used anywhere in this
 * layer: its eager both-branch evaluation records orphan nodes
 * (`eigen_first_class.md`, decision #16).
 *
 * No Eigen internals: only public `MatrixBase`/`Map`/`coeff` access and the
 * plain double kernels for the primal are used.
 */

namespace matrix_blocks_detail {

/// Shape payload copied into the tape: `rows == k` for `dot`, `rows x cols`
/// for `matrixVectorProduct` (`cols == 0` marks the flat dot layout), `rows ==
/// k` for `sum`. Argument order in the tape's External arg cache is always
/// logical row-major over each operand, operands concatenated in call order.
struct MatrixBlockShape {
    std::uint32_t rows = 0;
    std::uint32_t cols = 0;
};

/// A coefficient type this layer can snapshot: `RevScalar` (active or
/// constant slot) or a plain arithmetic scalar (constant slot).
template <typename T>
inline constexpr bool isBlockScalar =
    std::is_same_v<std::decay_t<T>, RevScalar> || std::is_arithmetic_v<std::decay_t<T>>;

/// Primal value of a coefficient (plain cast for non-AD scalars).
template <typename T>
double blockValue(const T& x) noexcept {
    if constexpr (std::is_same_v<std::decay_t<T>, RevScalar>) {
        return x.value;
    } else {
        return static_cast<double>(x);
    }
}

/// Tape slot of a coefficient (`RevTape::kNone` for non-AD scalars).
template <typename T>
std::uint32_t blockSlot(const T& x) noexcept {
    if constexpr (std::is_same_v<std::decay_t<T>, RevScalar>) {
        return x.node;
    } else {
        return RevTape::kNone;
    }
}

/// Append the coefficients of `x` to `values` (primal) and `args`
/// (`{value, slot}` snapshots for the tape's External arg cache) in logical
/// row-major order. Two-index `coeff` keeps arbitrary MatrixBase expressions
/// (cwise/broadcast results) supported, not just materialized matrices.
template <typename Derived>
void flattenBlock(const Eigen::MatrixBase<Derived>& x, std::vector<double>& values,
                  std::vector<RevScalar>& args) {
    const Eigen::Index n = x.size();
    if (n == 0) {
        return;
    }
    eigen_assert(values.size() == args.size());
    const Eigen::Index cols = x.cols();
    const std::size_t base = values.size();
    values.resize(base + static_cast<std::size_t>(n));
    args.resize(base + static_cast<std::size_t>(n));
    for (Eigen::Index i = 0; i < n; ++i) {
        const auto c = x.derived().coeff(i / cols, i % cols);
        const double v = blockValue(c);
        values[base + static_cast<std::size_t>(i)] = v;
        args[base + static_cast<std::size_t>(i)] = RevScalar(v, blockSlot(c));
    }
}

/// double primal kernels (the docs' rule: blocks call Eigen's double kernels).
inline double dotValues(const double* v, std::size_t n) {
    Eigen::Map<const Eigen::VectorXd> x(v, static_cast<Eigen::Index>(n));
    Eigen::Map<const Eigen::VectorXd> y(v + n, static_cast<Eigen::Index>(n));
    return x.dot(y);
}

inline double sumValues(const std::vector<double>& v) {
    return Eigen::Map<const Eigen::VectorXd>(v.data(), static_cast<Eigen::Index>(v.size())).sum();
}

inline double matvecSumValues(std::uint32_t rows, std::uint32_t cols, const double* v) {
    // flattenBlock emits logical row-major coefficients.
    Eigen::Map<const Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> A(
        v, static_cast<Eigen::Index>(rows), static_cast<Eigen::Index>(cols));
    Eigen::Map<const Eigen::VectorXd> x(v + static_cast<std::size_t>(rows) * cols,
                                        static_cast<Eigen::Index>(cols));
    return (A * x).sum();
}

// ── External callbacks (captureless, noexcept, convertible to the ABI) ──
//
// All callbacks read the primal operands from the tape's ExtArg cache, so
// they need no Eigen and no allocation. Payload = MatrixBlockShape.

inline void dotBlockVjp(const void* payload, double seed, const ExtArg* a, std::uint32_t /*nArgs*/,
                        RevTape& tape) noexcept {
    const std::uint32_t n = static_cast<const MatrixBlockShape*>(payload)->rows;
    for (std::uint32_t i = 0; i < n; ++i) {
        tape.addAdjoint(a[i].node, seed * a[n + i].value);
        tape.addAdjoint(a[n + i].node, seed * a[i].value);
    }
}

inline double dotBlockSvp(const void* payload, const ExtArg* a, const double* s,
                          std::uint32_t /*nArgs*/) noexcept {
    const std::uint32_t n = static_cast<const MatrixBlockShape*>(payload)->rows;
    double out = 0.0;
    for (std::uint32_t i = 0; i < n; ++i) {
        out += s[i] * a[n + i].value + a[i].value * s[n + i];
    }
    return out;
}

inline void dotBlockVhp(const void* payload, double adjoint, const ExtArg* /*a*/, const double* s,
                        std::uint32_t /*nArgs*/, double* b) noexcept {
    const std::uint32_t n = static_cast<const MatrixBlockShape*>(payload)->rows;
    for (std::uint32_t i = 0; i < n; ++i) {
        b[i] += adjoint * s[n + i];
        b[n + i] += adjoint * s[i];
    }
}

inline void sumBlockVjp(const void* /*payload*/, double seed, const ExtArg* a, std::uint32_t nArgs,
                        RevTape& tape) noexcept {
    for (std::uint32_t i = 0; i < nArgs; ++i) {
        tape.addAdjoint(a[i].node, seed);
    }
}

inline double sumBlockSvp(const void* /*payload*/, const ExtArg* /*a*/, const double* s,
                          std::uint32_t nArgs) noexcept {
    double out = 0.0;
    for (std::uint32_t i = 0; i < nArgs; ++i) {
        out += s[i];
    }
    return out;
}

/// `sum` is linear: every entry of the contracted Hessian is exactly zero.
inline void sumBlockVhp(const void* /*payload*/, double /*adjoint*/, const ExtArg* /*a*/,
                        const double* /*s*/, std::uint32_t /*nArgs*/, double* /*b*/) noexcept {}

inline void matvecBlockVjp(const void* payload, double seed, const ExtArg* a,
                           std::uint32_t /*nArgs*/, RevTape& tape) noexcept {
    const MatrixBlockShape shape = *static_cast<const MatrixBlockShape*>(payload);
    const std::uint32_t rows = shape.rows;
    const std::uint32_t cols = shape.cols;
    const ExtArg* xa = a + static_cast<std::size_t>(rows) * cols;
    for (std::uint32_t j = 0; j < cols; ++j) {
        double colSum = 0.0;
        for (std::uint32_t i = 0; i < rows; ++i) {
            colSum += a[static_cast<std::size_t>(i) * cols + j].value;
        }
        for (std::uint32_t i = 0; i < rows; ++i) {
            tape.addAdjoint(a[static_cast<std::size_t>(i) * cols + j].node, seed * xa[j].value);
        }
        tape.addAdjoint(xa[j].node, seed * colSum);
    }
}

inline double matvecBlockSvp(const void* payload, const ExtArg* a, const double* s,
                             std::uint32_t /*nArgs*/) noexcept {
    const MatrixBlockShape shape = *static_cast<const MatrixBlockShape*>(payload);
    const std::uint32_t rows = shape.rows;
    const std::uint32_t cols = shape.cols;
    const ExtArg* xa = a + static_cast<std::size_t>(rows) * cols;
    const double* sx = s + static_cast<std::size_t>(rows) * cols;
    double out = 0.0;
    for (std::uint32_t j = 0; j < cols; ++j) {
        double colSumA = 0.0;
        double colSumS = 0.0;
        for (std::uint32_t i = 0; i < rows; ++i) {
            const std::size_t p = static_cast<std::size_t>(i) * cols + j;
            colSumA += a[p].value;
            colSumS += s[p];
        }
        out += xa[j].value * colSumS + sx[j] * colSumA;
    }
    return out;
}

inline void matvecBlockVhp(const void* payload, double adjoint, const ExtArg* /*a*/,
                           const double* s, std::uint32_t /*nArgs*/, double* b) noexcept {
    const MatrixBlockShape shape = *static_cast<const MatrixBlockShape*>(payload);
    const std::uint32_t rows = shape.rows;
    const std::uint32_t cols = shape.cols;
    const double* sx = s + static_cast<std::size_t>(rows) * cols;
    double* bx = b + static_cast<std::size_t>(rows) * cols;
    for (std::uint32_t j = 0; j < cols; ++j) {
        double colSumS = 0.0;
        for (std::uint32_t i = 0; i < rows; ++i) {
            const std::size_t p = static_cast<std::size_t>(i) * cols + j;
            colSumS += s[p];
            b[p] += adjoint * sx[j];
        }
        bx[j] += adjoint * colSumS;
    }
}

} // namespace matrix_blocks_detail

/// `x.dot(y)`: one `External` node over the 2k flattened coefficients.
/// Values are computed with the Eigen double kernel; `vjp` scales by the
/// opposite operand, `svp` is the bilinear directional derivative and `vhp`
/// scatters the delta-pattern cross Hessian. Constant-only blocks collapse.
template <typename XDerived, typename YDerived>
RevScalar dot(const Eigen::MatrixBase<XDerived>& x, const Eigen::MatrixBase<YDerived>& y) {
    static_assert(matrix_blocks_detail::isBlockScalar<typename XDerived::Scalar>,
                  "lightape::dot: x must be a double/RevScalar Eigen expression");
    static_assert(matrix_blocks_detail::isBlockScalar<typename YDerived::Scalar>,
                  "lightape::dot: y must be a double/RevScalar Eigen expression");
    eigen_assert(x.size() == y.size());
    const std::size_t n = static_cast<std::size_t>(x.size());
    std::vector<double> values;
    std::vector<RevScalar> args;
    matrix_blocks_detail::flattenBlock(x, values, args);
    matrix_blocks_detail::flattenBlock(y, values, args);
    const double value = (n == 0) ? 0.0 : matrix_blocks_detail::dotValues(values.data(), n);
    const matrix_blocks_detail::MatrixBlockShape shape{static_cast<std::uint32_t>(n), 0u};
    return makeVhpFvar(value, args.data(), static_cast<std::uint32_t>(args.size()),
                       matrix_blocks_detail::dotBlockVjp, matrix_blocks_detail::dotBlockSvp,
                       matrix_blocks_detail::dotBlockVhp, shape);
}

/// `x.sum()`: one `External` node, unit `vjp`, exactly zero Hessian.
template <typename Derived>
RevScalar sum(const Eigen::MatrixBase<Derived>& x) {
    static_assert(matrix_blocks_detail::isBlockScalar<typename Derived::Scalar>,
                  "lightape::sum: x must be a double/RevScalar Eigen expression");
    std::vector<double> values;
    std::vector<RevScalar> args;
    matrix_blocks_detail::flattenBlock(x, values, args);
    const double value = values.empty() ? 0.0 : matrix_blocks_detail::sumValues(values);
    const matrix_blocks_detail::MatrixBlockShape shape{static_cast<std::uint32_t>(args.size()), 0u};
    return makeVhpFvar(value, args.data(), static_cast<std::uint32_t>(args.size()),
                       matrix_blocks_detail::sumBlockVjp, matrix_blocks_detail::sumBlockSvp,
                       matrix_blocks_detail::sumBlockVhp, shape);
}

/// `sum(A*x)` (`x` a k-vector, `A` m x k): one `External` node.
/// `vjp`: `dF/dA_ij = x_j`, `dF/dx_j = sum_i A_ij`; the cross Hessian
/// `d2F/dA_ij dx_j = 1` is the delta pattern in `matvecBlockVhp`.
template <typename ADerived, typename XDerived>
RevScalar matrixVectorProduct(const Eigen::MatrixBase<ADerived>& A,
                              const Eigen::MatrixBase<XDerived>& x) {
    static_assert(matrix_blocks_detail::isBlockScalar<typename ADerived::Scalar>,
                  "lightape::matrixVectorProduct: A must be a double/RevScalar Eigen expression");
    static_assert(matrix_blocks_detail::isBlockScalar<typename XDerived::Scalar>,
                  "lightape::matrixVectorProduct: x must be a double/RevScalar Eigen expression");
    eigen_assert(x.rows() == 1 || x.cols() == 1);
    eigen_assert(x.size() == A.cols());
    const std::uint32_t rows = static_cast<std::uint32_t>(A.rows());
    const std::uint32_t cols = static_cast<std::uint32_t>(A.cols());
    std::vector<double> values;
    std::vector<RevScalar> args;
    matrix_blocks_detail::flattenBlock(A, values, args);
    matrix_blocks_detail::flattenBlock(x, values, args);
    const double value = (rows == 0 || cols == 0)
                             ? 0.0
                             : matrix_blocks_detail::matvecSumValues(rows, cols, values.data());
    const matrix_blocks_detail::MatrixBlockShape shape{rows, cols};
    return makeVhpFvar(value, args.data(), static_cast<std::uint32_t>(args.size()),
                       matrix_blocks_detail::matvecBlockVjp, matrix_blocks_detail::matvecBlockSvp,
                       matrix_blocks_detail::matvecBlockVhp, shape);
}

} // namespace lightape

#endif // LIGHTAPE_EIGEN_MATRIX_BLOCKS_H
