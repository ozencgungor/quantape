#ifndef QUANTAPE_UTIL_CHECK_H
#define QUANTAPE_UTIL_CHECK_H

#include <charconv>
#include <concepts>
#include <cstddef>
#include <iterator>
#include <string>

namespace quantape::util {
/**
 * @file Check.h
 * @brief std-only diagnostic and formatting support for the test suite.
 *
 * Lives in `util` so tests depend on the library headers directly instead of a
 * private test-support header: `num` formats doubles for assertion messages,
 * `RecordProperty` diagnostics and benchmark output, and the `detail`
 * concepts back the sequence/AD close assertions in
 * `tests/support/Assertions.h`. Everything here is std-only (the `util`
 * self-containment rule holds), header-only, and non-throwing.
 *
 * This header is no longer a test harness: the legacy `CHECK` macro family
 * and its fatal `std::_Exit` failure path were retired with the googletest
 * migration, and no gtest header is included here.
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

/// Anything the sequence close assertions can compare element-wise.
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

} // namespace detail

} // namespace quantape::util

#endif // QUANTAPE_UTIL_CHECK_H
