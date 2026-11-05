#ifndef LIGHTAPE_EIGEN_SELECT_H
#define LIGHTAPE_EIGEN_SELECT_H

#include <Eigen/Core>

#include <type_traits>

#include "lightape/eigen/ForwardScalarEigen.h"
#include "lightape/eigen/LeanReverseEigen.h"

namespace lightape {
/**
 * @file Select.h
 * @brief AD-safe lazy `select` for Eigen expressions (decision #16)
 *
 * ## NEVER call Eigen's `.select()` on AD scalars — use `lightape::select`
 *
 * Eigen 5 implements `select` as a `CwiseTernaryOp` over `(then, else, cond)`
 * whose `ternary_evaluator::coeff` computes the then-coefficient, the
 * else-coefficient and the condition coefficient *before* invoking the
 * ternary functor (argument evaluation happens before the call). For plain
 * arithmetic scalars the discarded branch is dead arithmetic; for
 * `RevScalar`/`Tangent` evaluating a coefficient is a side effect — it
 * appends tape nodes or performs AD bookkeeping. The discarded ("orphan")
 * nodes are still swept during the reverse pass, and a chain rule that
 * multiplies a zero adjoint by a primal-derived value can inject
 * `0 * inf -> NaN` into a live adjoint (`exp` at `+inf`, `log`, division...).
 * Probability code relies on `select` precisely to guard such numerically
 * dangerous branches, so eager selection is a correctness bug, not just
 * wasted work. Stan main (5.0.1, `prim/meta/lazy_select_evaluator.hpp`) added
 * an evaluator override for exactly this; this header is the lightape
 * equivalent: never depend on the accidental laziness of Eigen 3.4's
 * dedicated `Select` expression, because the version pin can move to Eigen 5
 * with no edits outside the Eigen adapters (decision #16,
 * `internal_docs/xad_stan_research/deep/eigen_first_class.md`).
 *
 * ## Semantics
 *
 * `lightape::select(cond, thenExpr, elseExpr)` walks the coefficients once:
 * - the condition coefficient is read from the primal only: `bool` arrays are
 *   used directly, integer/floating arrays are nonzero tests, and an array of
 *   AD scalars compares `.value` recursively (so nested `Tangent` levels
 *   work) — no node is created or read for the condition;
 * - only the winning branch's coefficient is evaluated, so only the selected
 *   branch records nodes/has AD side effects;
 * - the result is materialised into a plain evaluated array (`.eval()`-style
 *   construction), of the common scalar type of the two branches, so the
 *   branch expressions may die immediately and the surviving `RevScalar`
 *   nodes stay valid until the tape is reset.
 *
 * The conditions, then-branch and else-branch must have the same shape
 * (Eigen's own `select` also asserts this; no broadcasting support).
 *
 * ## Eigen-5 readiness (decision #16)
 *
 * The implementation uses only public Eigen API — `Derived::Scalar`,
 * `RowsAtCompileTime`/`ColsAtCompileTime`, `rows`/`cols`/`size`,
 * `coeff`/`coeffRef`, `resize` — with no Eigen-3-only internals, no functor
 * traits and no evaluator overrides, and no `EIGEN_VERSION_AT_LEAST`-gated
 * branches. It compiles and behaves identically against Eigen 3.4 and
 * Eigen 5.
 *
 * Arrays in, arrays out (the primary overload below); matrices in, matrices
 * out (the `MatrixBase` overload that follows). Pass Eigen expressions (for
 * example `x.array().exp()`) as branches — they are consumed coefficient by
 * coefficient, so only the winners are evaluated.
 */

namespace select_detail {

/// Common scalar of the two branches: the mixed-scalar specializations the
/// sibling adapters already provide (`ScalarBinaryOpTraits<RevScalar,
/// double, BinaryOp>`, `<Tangent<S,N>, S, BinaryOp>`, ...) resolve
/// AD/plain pairs; two arithmetic scalars fall back to `std::common_type`.
template <typename A, typename B, typename Enable = void>
struct CommonScalar {
    using type = Eigen::ScalarBinaryOpTraits<A, B>::ReturnType;
};

template <typename A, typename B>
struct CommonScalar<A, B, std::enable_if_t<std::is_arithmetic_v<A> && std::is_arithmetic_v<B>>> {
    using type = std::common_type_t<A, B>;
};

/// Primal truth of one condition coefficient. `bool` is used directly,
/// arithmetic scalars test nonzero, and an AD scalar is reduced to its primal
/// `value` recursively (nested `Tangent` levels included): the condition is a
/// derivative-free predicate, so no AD node takes part in the decision.
template <typename T>
bool conditionTrue(const T& x) {
    using U = std::remove_cv_t<std::remove_reference_t<T>>;
    if constexpr (std::is_same_v<U, bool>) {
        return x;
    } else if constexpr (std::is_arithmetic_v<U>) {
        return x != U(0);
    } else {
        return conditionTrue(x.value);
    }
}

/// Shared coefficient walk for every shape/kind of the public overloads.
/// `Result` is the already-typed (array or matrix) plain destination; this
/// function supplies the values. No Eigen internals are touched: two-index
/// `coeff` keeps arbitrary non-linear-access expressions (cwise/broadcast
/// results) supported, and `coeffRef` writes the plain destination.
template <typename CondDerived, typename ThenDerived, typename ElseDerived, typename Result>
void selectInto(const Eigen::DenseBase<CondDerived>& cond,
                const Eigen::DenseBase<ThenDerived>& thenExpr,
                const Eigen::DenseBase<ElseDerived>& elseExpr, Result& result) {
    const auto& condEval = cond.derived();
    const auto& thenEval = thenExpr.derived();
    const auto& elseEval = elseExpr.derived();
    const Eigen::Index rows = thenEval.rows();
    const Eigen::Index cols = thenEval.cols();
    eigen_assert(condEval.rows() == rows && condEval.cols() == cols);
    eigen_assert(elseEval.rows() == rows && elseEval.cols() == cols);
    result.resize(rows, cols);
    for (Eigen::Index i = 0; i < thenEval.size(); ++i) {
        const Eigen::Index r = i / cols;
        const Eigen::Index c = i % cols;
        // Only the selected coefficient is read: the losing branch is never
        // evaluated, so it records no tape nodes (decision #16).
        if (conditionTrue(condEval.coeff(r, c))) {
            result.coeffRef(r, c) = thenEval.coeff(r, c);
        } else {
            result.coeffRef(r, c) = elseEval.coeff(r, c);
        }
    }
}

} // namespace select_detail

/// Lazy, AD-safe coefficient-wise select for array expressions: the result is
/// a plain `Eigen::Array` of the common scalar type. `cond` may be a
/// `bool`/integer/floating array or an array of AD scalars (primal compare).
/// The branch expressions must have the same shape as `cond`. Never call
/// `.select()` on AD scalars instead of this function.
template <typename CondDerived, typename ThenDerived, typename ElseDerived>
auto select(const Eigen::ArrayBase<CondDerived>& cond,
            const Eigen::ArrayBase<ThenDerived>& thenExpr,
            const Eigen::ArrayBase<ElseDerived>& elseExpr) {
    using Common = select_detail::CommonScalar<typename ThenDerived::Scalar,
                                               typename ElseDerived::Scalar>::type;
    // Shape from the condition, exactly like Eigen's own `Select` traits
    // (`RowsAtCompileTime = ConditionMatrixType::RowsAtCompileTime`); the
    // runtime asserts in `selectInto` require the branches to match.
    Eigen::Array<Common, CondDerived::RowsAtCompileTime, CondDerived::ColsAtCompileTime> result;
    select_detail::selectInto(cond, thenExpr, elseExpr, result);
    return result;
}

/// Matrix/vector counterpart: matrix in, matrix out, with the same lazy
/// coefficient walk underneath (the winning branch only).
template <typename CondDerived, typename ThenDerived, typename ElseDerived>
auto select(const Eigen::MatrixBase<CondDerived>& cond,
            const Eigen::MatrixBase<ThenDerived>& thenExpr,
            const Eigen::MatrixBase<ElseDerived>& elseExpr) {
    using Common = select_detail::CommonScalar<typename ThenDerived::Scalar,
                                               typename ElseDerived::Scalar>::type;
    Eigen::Matrix<Common, CondDerived::RowsAtCompileTime, CondDerived::ColsAtCompileTime> result;
    select_detail::selectInto(cond, thenExpr, elseExpr, result);
    return result;
}

} // namespace lightape

#endif // LIGHTAPE_EIGEN_SELECT_H
