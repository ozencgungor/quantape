#ifndef QUANTAPE_UTIL_CONSTANTS_H
#define QUANTAPE_UTIL_CONSTANTS_H

namespace quantape::util {
/**
 * @file constants.h
 * @brief Mathematical constants with a double-double ready hi/lo split
 *
 * Each constant stores the correctly rounded double `hi` plus the exact
 * residual `lo = value - hi`; `hi + lo` reproduces the constant to ~106 bits,
 * enough to seed an extended-precision type (`quantape::math::DoubleDouble`
 * or any hi/lo pair) without loss. The struct implicitly converts to `hi`, so
 * plain-double code reads `util::kPi` unchanged while extended-precision code
 * consumes `kPi.hi` / `kPi.lo`.
 *
 * Values were generated at 120 decimal digits and split with correct double
 * rounding. `lo` may be negative (e.g. `kSqrt2`); that is the true residual,
 * not an error. Day-count and market conventions live in `quantape/datetime`,
 * never here.
 *
 * std-only, lowest include layer.
 */
struct HiLo {
    double hi = 0.0;
    double lo = 0.0;
    constexpr operator double() const noexcept { return hi; }
};

inline constexpr HiLo kPi{3.141592653589793, 1.2246467991473532e-16};
inline constexpr HiLo kTwoPi{6.283185307179586, 2.4492935982947064e-16};
inline constexpr HiLo kHalfPi{1.5707963267948966, 6.123233995736766e-17};
inline constexpr HiLo kQuarterPi{0.7853981633974483, 3.061616997868383e-17};
inline constexpr HiLo kInvPi{0.3183098861837907, -1.9678676675182486e-17};
inline constexpr HiLo kSqrt2{1.4142135623730951, -9.667293313452913e-17};
inline constexpr HiLo kInvSqrt2{0.7071067811865476, -4.833646656726457e-17};
inline constexpr HiLo kSqrt2Pi{2.5066282746310007, -1.8328579980459167e-16};
inline constexpr HiLo kInvSqrt2Pi{0.3989422804014327, -2.49232720227773e-17};
inline constexpr HiLo kLn2{0.6931471805599453, 2.3190468138462996e-17};
inline constexpr HiLo kLn10{2.302585092994046, -2.1707562233822494e-16};
inline constexpr HiLo kE{2.718281828459045, 1.4456468917292502e-16};
inline constexpr HiLo kEulerGamma{0.5772156649015329, -4.942915152430645e-18};

} // namespace quantape::util

#endif // QUANTAPE_UTIL_CONSTANTS_H
