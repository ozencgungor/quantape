#pragma once

#include <charconv>
#include <cstddef>
#include <string>

namespace quantape::format {

/// Large enough for every double format (shortest 34, scientific 25,
/// fixed 329, hex 16) plus slack.
inline constexpr std::size_t kMaxDoubleChars = 384;

enum class FloatFormat { Shortest, Scientific, Fixed, General, Hex };

/// Writes `value` into [out, out + size). Returns a pointer past the last
/// character written; on buffer exhaustion returns `out` with `*out = '\0'`.
/// `precision < 0` means shortest form; otherwise printf-style precision
/// (default 6 when a format is given without precision).
char* write(double value, char* out, std::size_t size, FloatFormat format = FloatFormat::Shortest,
            int precision = -1);

/// Convenience allocating overload for tools and messages.
std::string toString(double value, FloatFormat format = FloatFormat::Shortest, int precision = -1);

/// std::to_chars-style adapter for generic code.
std::to_chars_result to_chars(char* first, char* last, double value,
                              FloatFormat format = FloatFormat::Shortest, int precision = -1);

} // namespace quantape::format
