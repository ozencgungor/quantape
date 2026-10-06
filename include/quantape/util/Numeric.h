#ifndef QUANTAPE_UTIL_NUMERIC_H
#define QUANTAPE_UTIL_NUMERIC_H

#include <cstddef>

namespace quantape::util {
/**
 * @file numeric.h
 * @brief Small numeric helpers shared across modules (std-only, AD-friendly)
 *
 * The scalar templates operate on any type supporting the corresponding
 * operators, so the same call compiles for `double`, `RevScalar`,
 * `Tangent<...>`, and the planned AD chains. Nothing here allocates, depends
 * on type payloads, or includes another quantape module: `util` is the lowest
 * include layer. Integer helpers (`ceilDiv`, `alignUp`, `isPowerOfTwo`) exist
 * because chunked tape pools, SIMD batches, and block loops all need them.
 */

/// Clamp to [lo, hi]. Returns by value: returning a reference could dangle
/// when callers pass temporaries (the std::clamp pitfall).
template <class T>
constexpr T clamp(const T& value, const T& lo, const T& hi) noexcept {
    return value < lo ? lo : (hi < value ? hi : value);
}

/// Clamp to [0, 1].
template <class T>
constexpr T clamp01(const T& value) noexcept {
    return clamp(value, T(0), T(1));
}

/// Exact mathematical sign: -1, 0, +1 (NaN yields 0, ±0 yield 0).
template <class T>
constexpr int signum(const T& x) noexcept {
    return (x > T(0)) - (x < T(0));
}

/// ±1 with +1 at zero: the AD-friendly sign (derivative of |x|, sign of a
/// clamp direction) where `signum` would kill the gradient.
template <class T>
constexpr T unitSign(const T& x) noexcept {
    return x >= T(0) ? T(1) : T(-1);
}

/// `x * x` without the intermediate temporary of `std::pow`.
template <class T>
constexpr T sqr(const T& x) noexcept {
    return x * x;
}

/// `x * x * x`.
template <class T>
constexpr T cube(const T& x) noexcept {
    return x * x * x;
}

/// `a / b`, or `fallback` when `b` is exactly zero (no tolerance).
template <class T>
constexpr T safeDiv(const T& a, const T& b, const T& fallback = T(0)) noexcept {
    return b == T(0) ? fallback : a / b;
}

/// Linear interpolation `a + t (b - a)` (t is a passive fraction).
template <class T>
constexpr T lerp(const T& a, const T& b, double t) noexcept {
    return a + (b - a) * T(t);
}

/// Ceiling division for non-negative integers: `ceil(n / d)`, `d > 0`.
constexpr std::size_t ceilDiv(std::size_t n, std::size_t d) noexcept {
    return (n + d - 1) / d;
}

/// True when `n` is a power of two (zero is not).
constexpr bool isPowerOfTwo(std::size_t n) noexcept {
    return n != 0 && (n & (n - 1)) == 0;
}

/// Round `n` up to the next multiple of `align` (a power of two).
constexpr std::size_t alignUp(std::size_t n, std::size_t align) noexcept {
    return (n + align - 1) & ~(align - 1);
}

} // namespace quantape::util

#endif // QUANTAPE_UTIL_NUMERIC_H
