#ifndef QUANTAPE_UTIL_FP_H
#define QUANTAPE_UTIL_FP_H

#include <bit>
#include <cstdint>

namespace quantape::util {
/**
 * @file FloatingPoint.h
 * @brief Floating-point predicates and tolerance comparisons, libm-free
 *
 * Every predicate reads the IEEE-754 bit pattern (`std::bit_cast`), so it is
 * `constexpr`, allocation-free, and independent of any compiler fast-math mode.
 * The project bans `-ffast-math`, but these helpers remain the sanctioned way
 * to ask "finite / NaN / Inf" at numeric boundaries and sentinel checks where
 * `std::isfinite` classification or quieting is undesirable.
 *
 * This is the lowest include layer: std-only, no other quantape module may be
 * included here, and every module may include this header.
 */

/// True for every finite double (exponent bits not all ones). NaN and ±Inf are
/// both non-finite; `-ffast-math`-style sentinels (`±max`) are finite.
[[nodiscard]] constexpr bool isFiniteBitwise(double x) noexcept {
    return ((std::bit_cast<std::uint64_t>(x) >> 52) & 0x7FFULL) != 0x7FFULL;
}

/// True for any NaN, quiet or signalling, payload-agnostic.
[[nodiscard]] constexpr bool isNanBitwise(double x) noexcept {
    const std::uint64_t bits = std::bit_cast<std::uint64_t>(x);
    return ((bits >> 52) & 0x7FFULL) == 0x7FFULL && (bits & 0x000FFFFFFFFFFFFFULL) != 0ULL;
}

/// True for +Inf and -Inf.
[[nodiscard]] constexpr bool isInfBitwise(double x) noexcept {
    const std::uint64_t bits = std::bit_cast<std::uint64_t>(x);
    return ((bits >> 52) & 0x7FFULL) == 0x7FFULL && (bits & 0x000FFFFFFFFFFFFFULL) == 0ULL;
}

/**
 * @brief NumPy-style all-close test: `|a - b| <= atol + rtol * |b|`.
 *
 * As in `numpy.isclose`, tolerance scales with `|b|` (the reference), so the
 * test is not symmetric; `atol` covers comparisons near zero. NaN is never
 * close unless `equalNan` is set (then NaN-to-NaN only); ±Inf is close only to
 * the same infinity. The defaults are double-precision near-equality
 * (1e-12 relative, 1e-15 absolute), not NumPy's float32-legacy defaults.
 */
[[nodiscard]] constexpr bool isClose(double a, double b, double rtol = 1e-12, double atol = 1e-15,
                                     bool equalNan = false) noexcept {
    if (isNanBitwise(a) || isNanBitwise(b)) {
        return equalNan && isNanBitwise(a) && isNanBitwise(b);
    }
    if (isInfBitwise(a) || isInfBitwise(b)) {
        return a == b;
    }
    const double diff = a > b ? a - b : b - a;
    const double scale = b >= 0.0 ? b : -b;
    return diff <= atol + rtol * scale;
}

/// Absolute-only tolerance test: `|a - b| <= tol` (NaN never close, ±Inf only
/// to itself). Saturating differences (`a - b` overflowing to Inf) fail, which
/// is the desired behaviour for tolerance checks.
[[nodiscard]] constexpr bool isCloseAbs(double a, double b, double tol) noexcept {
    if (isNanBitwise(a) || isNanBitwise(b)) {
        return false;
    }
    if (isInfBitwise(a) || isInfBitwise(b)) {
        return a == b;
    }
    const double diff = a > b ? a - b : b - a;
    return diff <= tol;
}

} // namespace quantape::util

#endif // QUANTAPE_UTIL_FP_H
