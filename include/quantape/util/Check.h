#ifndef QUANTAPE_UTIL_CHECK_H
#define QUANTAPE_UTIL_CHECK_H

#include "quantape/util/FloatingPoint.h"

#include <charconv>
#include <concepts>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <iterator>
#include <string>

namespace quantape::util {
/**
 * @file Check.h
 * @brief Test-support helpers (std-only) shared by the test executables
 *
 * Lives in `util` so tests depend on library headers directly instead of a
 * private test-support header: `CHECK`, `checkClose`, and the diagnostic `num`
 * formatters. Everything here is std-only (the `util` self-containment rule
 * holds), header-only, and non-throwing. Failures print to stderr and call
 * `std::_Exit(1)`: skipping static destruction is deliberate — failing while
 * nested AD tapes are still live must not unwind through exit-time
 * destructors.
 *
 * `checkClose` is generic through templates: one scalar overload and one
 * element-wise overload for any sized range of scalar-convertible values.
 * Eigen vectors, matrices and dense expressions satisfy the range requirements
 * (`.size()`/`.begin()`/`.end()`), so no Eigen dependency is needed here —
 * std containers, C arrays, Eigen types, and any iterable all work.
 */

/// Shortest round-trip text for a double (status and diagnostic messages).
inline std::string num(double value) {
    char buffer[32];
    const auto res = std::to_chars(buffer, buffer + sizeof(buffer), value);
    return std::string(buffer, res.ptr);
}

/// `%.*g`-style text for approximate statistics (printf-compatible).
inline std::string num(double value, int precision) {
    char buffer[64];
    const auto res = std::to_chars(buffer, buffer + sizeof(buffer), value,
                                   std::chars_format::general, precision);
    return std::string(buffer, res.ptr);
}

/// Report a failed check and terminate without unwinding.
[[noreturn]] inline void fail(const char* file, int line, const char* expr) {
    std::fprintf(stderr, "CHECK failed at %s:%d: %s\n", file, line, expr);
    std::fflush(nullptr);
    std::_Exit(1);
}

/// Boolean gate behind the `CHECK` macro.
inline void check(bool ok, const char* file, int line, const char* expr) {
    if (!ok) {
        fail(file, line, expr);
    }
}

namespace detail {

/// Contiguous sized sequence with linear `data()` access (std::vector,
/// std::array, Eigen vectors and matrices, anything vector-like). Preferred
/// when available: order is exact and no iterator type is needed. Eigen
/// matrices only reach their coefficients this way — `begin()` is vector-only
/// and libc++'s `ranges::begin` additionally rejects Eigen through its
/// ADL/deleted-fallback machinery.
template <class R>
concept ContiguousSequence = requires(const R& r) {
    std::size(r);
    { r.data()[0] } -> std::convertible_to<double>;
};

/// Iterator-accessible sized sequence used only when no linear `data()`
/// exists (`std::initializer_list`, C arrays). The negation is required: for
/// Eigen matrices `begin()` is *declared* (and static-asserts only on
/// instantiation), so a plain `begin()` check would accept them here.
template <class R>
concept IteratorSequence = requires(const R& r) {
    std::size(r);
    requires(!ContiguousSequence<R>);
    std::begin(r);
    { *std::begin(r) } -> std::convertible_to<double>;
};

/// Anything checkClose can compare element-wise.
template <class R>
concept ScalarSequence = ContiguousSequence<R> || IteratorSequence<R>;

/// Indexed read for either access path (contiguous data preferred).
template <class R>
inline double sequenceAt(const R& r, std::size_t i) {
    if constexpr (ContiguousSequence<R>) {
        return static_cast<double>(r.data()[i]);
    } else {
        return static_cast<double>(*(std::begin(r) + static_cast<std::ptrdiff_t>(i)));
    }
}

/// Duck-typed AD scalar: exposes `val()`/`adj()` (Stan `var` today; future
/// in-house chains can match this shape or specialize a trait). No Stan/Eigen
/// include is needed here — the template is only instantiated in AD-aware test
/// translation units.
template <class T>
concept ValAdjScalar = requires(const T& x) {
    { x.val() } -> std::convertible_to<double>;
    { x.adj() } -> std::convertible_to<double>;
};

/// Common element/vector mismatch reporter (prints and terminates).
[[noreturn]] inline void failClose(const char* label, std::size_t index, double got,
                                   double expected, double tol) {
    std::fprintf(stderr, "FAIL: %s[%zu] got=%s expected=%s tol=%s\n", label, index,
                 num(got, 12).c_str(), num(expected, 12).c_str(), num(tol, 3).c_str());
    std::fflush(nullptr);
    std::_Exit(1);
}

/// AD-scalar mismatch reporter (value vs adjoint).
[[noreturn]] inline void failAdPart(const char* label, const char* part, double got, double expected,
                                    double tol) {
    std::fprintf(stderr, "FAIL: %s %s got=%s expected=%s tol=%s\n", label, part, num(got, 12).c_str(),
                 num(expected, 12).c_str(), num(tol, 3).c_str());
    std::fflush(nullptr);
    std::_Exit(1);
}

} // namespace detail

/// Scalar tolerance gate: prints label/got/expected/tol and exits on mismatch.
/// `FloatingPoint.h` semantics: NaN never passes, ±Inf only equals itself.
inline void checkClose(const char* label, double got, double expected, double tol) {
    if (!isCloseAbs(got, expected, tol)) {
        detail::failClose(label, 0, got, expected, tol);
    }
}

/**
 * @brief Element-wise tolerance gate for any sized sequence of scalars.
 *
 * Works for `std::vector`, `std::array`, C arrays, `std::initializer_list`,
 * Eigen vectors, Eigen matrices and `.eval()`-ed dense expressions, and
 * anything else with `size()` plus iterators or linear `data()`, with elements
 * convertible to `double`. Sizes must match; the first mismatching element
 * prints as `label[index]` and exits. Lazy Eigen expressions must be evaluated
 * first (`.eval()`), since they have neither iterators nor `data()`.
 */
template <detail::ScalarSequence R1, detail::ScalarSequence R2>
inline void checkClose(const char* label, const R1& got, const R2& expected, double tol) {
    const std::size_t gotSize = static_cast<std::size_t>(std::size(got));
    const std::size_t expectedSize = static_cast<std::size_t>(std::size(expected));
    if (gotSize != expectedSize) {
        std::fprintf(stderr, "FAIL: %s size mismatch got=%zu expected=%zu\n", label, gotSize,
                     expectedSize);
        std::fflush(nullptr);
        std::_Exit(1);
    }
    for (std::size_t i = 0; i < gotSize; ++i) {
        const double g = detail::sequenceAt(got, i);
        const double e = detail::sequenceAt(expected, i);
        if (!isCloseAbs(g, e, tol)) {
            detail::failClose(label, i, g, e, tol);
        }
    }
}

/**
 * @brief AD-scalar tolerance gate: checks **value and adjoint**.
 *
 * Matches any type exposing `val()`/`adj()` convertible to `double` (Stan
 * `var` in the tests today, future chains by the same shape). Both parts must
 * be close within `tol`; mismatches report `label value` / `label adjoint`.
 * Use the `double` overload (or `.val()`) when only the primal should be
 * compared.
 */
template <detail::ValAdjScalar T, detail::ValAdjScalar U>
inline void checkClose(const char* label, const T& got, const U& expected, double tol) {
    const double gotValue = static_cast<double>(got.val());
    const double expectedValue = static_cast<double>(expected.val());
    if (!isCloseAbs(gotValue, expectedValue, tol)) {
        detail::failAdPart(label, "value", gotValue, expectedValue, tol);
    }
    const double gotAdjoint = static_cast<double>(got.adj());
    const double expectedAdjoint = static_cast<double>(expected.adj());
    if (!isCloseAbs(gotAdjoint, expectedAdjoint, tol)) {
        detail::failAdPart(label, "adjoint", gotAdjoint, expectedAdjoint, tol);
    }
}

} // namespace quantape::util

#define CHECK(cond) ::quantape::util::check(static_cast<bool>(cond), __FILE__, __LINE__, #cond)

#endif // QUANTAPE_UTIL_CHECK_H
