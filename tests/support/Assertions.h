#ifndef QUANTAPE_TESTS_SUPPORT_ASSERTIONS_H
#define QUANTAPE_TESTS_SUPPORT_ASSERTIONS_H

// Assertion helpers built on quantape::util::isCloseAbs / isClose.
//
// Semantics match the retired legacy close-gate family exactly: NaN never
// passes, +/-Inf only equals itself, and the failure text formats values with
// quantape::util::num(got, 12) / num(expected, 12) / num(tol, 3) and echoes the
// got/expected expressions. The macros wrap a SCOPED_TRACE(label), so a failing
// loop iteration reports its label.

#include "quantape/util/Check.h"
#include "quantape/util/FloatingPoint.h"

#include <cstddef>
#include <iterator>

#include <gtest/gtest.h>

namespace quantape::tests {

/// Scalar absolute-tolerance result; the predicate is util::isCloseAbs.
inline ::testing::AssertionResult closeAbsResult(const char* gotExpr, const char* expExpr,
                                                 double got, double expected, double tol) {
    if (quantape::util::isCloseAbs(got, expected, tol)) {
        return ::testing::AssertionSuccess();
    }
    return ::testing::AssertionFailure()
           << "got " << quantape::util::num(got, 12) << ", expected "
           << quantape::util::num(expected, 12) << ", tol " << quantape::util::num(tol, 3)
           << "\n  got-expr: " << gotExpr << "\n  exp-expr: " << expExpr;
}

/// Scalar relative-tolerance result; the predicate is util::isClose.
inline ::testing::AssertionResult closeRelResult(const char* gotExpr, const char* expExpr,
                                                 double got, double expected, double rtol,
                                                 double atol = 0.0) {
    if (quantape::util::isClose(got, expected, rtol, atol)) {
        return ::testing::AssertionSuccess();
    }
    return ::testing::AssertionFailure()
           << "got " << quantape::util::num(got, 12) << ", expected "
           << quantape::util::num(expected, 12) << ", rtol " << quantape::util::num(rtol, 3)
           << ", atol " << quantape::util::num(atol, 3) << "\n  got-expr: " << gotExpr
           << "\n  exp-expr: " << expExpr;
}

/// Element-wise result for sized sequences (std containers, C arrays,
/// initializer_list, Eigen dense and `.eval()`-ed expressions). Reports a size
/// mismatch and the first mismatching index, like the retired legacy gate did.
template <quantape::util::detail::ScalarSequence R1, quantape::util::detail::ScalarSequence R2>
::testing::AssertionResult closeSeqResult(const char* gotExpr, const char* expExpr, const R1& got,
                                          const R2& expected, double tol) {
    const std::size_t gotSize = static_cast<std::size_t>(std::size(got));
    const std::size_t expectedSize = static_cast<std::size_t>(std::size(expected));
    if (gotSize != expectedSize) {
        return ::testing::AssertionFailure()
               << "size mismatch got=" << gotSize << " expected=" << expectedSize
               << "\n  got-expr: " << gotExpr << "\n  exp-expr: " << expExpr;
    }
    for (std::size_t i = 0; i < gotSize; ++i) {
        const double g = quantape::util::detail::sequenceAt(got, i);
        const double e = quantape::util::detail::sequenceAt(expected, i);
        if (!quantape::util::isCloseAbs(g, e, tol)) {
            return ::testing::AssertionFailure()
                   << "first mismatch at index " << i << ": got " << quantape::util::num(g, 12)
                   << ", expected " << quantape::util::num(e, 12) << ", tol "
                   << quantape::util::num(tol, 3) << "\n  got-expr: " << gotExpr
                   << "\n  exp-expr: " << expExpr;
        }
    }
    return ::testing::AssertionSuccess();
}

/// AD-scalar result: checks the value AND the adjoint, like the retired
/// legacy AD overload. Accepts anything exposing val()/adj().
template <quantape::util::detail::ValAdjScalar T, quantape::util::detail::ValAdjScalar U>
::testing::AssertionResult adCloseResult(const char* gotExpr, const char* expExpr, const T& got,
                                         const U& expected, double tol) {
    const auto value = closeAbsResult(gotExpr, expExpr, static_cast<double>(got.val()),
                                      static_cast<double>(expected.val()), tol);
    if (!value) {
        return ::testing::AssertionFailure() << "value part: " << value.message();
    }
    const auto adjoint = closeAbsResult(gotExpr, expExpr, static_cast<double>(got.adj()),
                                        static_cast<double>(expected.adj()), tol);
    if (!adjoint) {
        return ::testing::AssertionFailure() << "adjoint part: " << adjoint.message();
    }
    return ::testing::AssertionSuccess();
}

} // namespace quantape::tests

#define CHECK_CLOSE(label, got, expected, tol)                                                     \
    do {                                                                                           \
        SCOPED_TRACE(label);                                                                       \
        EXPECT_TRUE(::quantape::tests::closeAbsResult(#got, #expected, (got), (expected), (tol))); \
    } while (0)

#define REQUIRE_CLOSE(label, got, expected, tol)                                                   \
    do {                                                                                           \
        SCOPED_TRACE(label);                                                                       \
        ASSERT_TRUE(::quantape::tests::closeAbsResult(#got, #expected, (got), (expected), (tol))); \
    } while (0)

#define CHECK_CLOSE_SEQ(label, got, expected, tol)                                                 \
    do {                                                                                           \
        SCOPED_TRACE(label);                                                                       \
        EXPECT_TRUE(::quantape::tests::closeSeqResult(#got, #expected, (got), (expected), (tol))); \
    } while (0)

#define CHECK_AD_CLOSE(label, got, expected, tol)                                                  \
    do {                                                                                           \
        SCOPED_TRACE(label);                                                                       \
        EXPECT_TRUE(::quantape::tests::adCloseResult(#got, #expected, (got), (expected), (tol)));  \
    } while (0)

#define CHECK_CLOSE_REL(label, got, expected, ...)                                                 \
    do {                                                                                           \
        SCOPED_TRACE(label);                                                                       \
        EXPECT_TRUE(                                                                               \
            ::quantape::tests::closeRelResult(#got, #expected, (got), (expected), __VA_ARGS__));   \
    } while (0)

#endif // QUANTAPE_TESTS_SUPPORT_ASSERTIONS_H
