#include "quantape/format/Number.h"

#include "zmij-to-chars.h"

namespace quantape::format {
namespace {

zmij::chars_format toZmij(FloatFormat format) {
    switch (format) {
        case FloatFormat::Shortest:
            return zmij::chars_format::general;
        case FloatFormat::Scientific:
            return zmij::chars_format::scientific;
        case FloatFormat::Fixed:
            return zmij::chars_format::fixed;
        case FloatFormat::General:
            return zmij::chars_format::general;
        case FloatFormat::Hex:
            return zmij::chars_format::hex;
    }
    return zmij::chars_format::general;
}

} // namespace

char* write(double value, char* out, std::size_t size, FloatFormat format, int precision) {
    if (size == 0) {
        return out;
    }
    zmij::to_chars_result result;
    if (format == FloatFormat::Shortest) {
        result = zmij::to_chars(out, out + size, value);
    } else if (precision < 0) {
        result = zmij::to_chars(out, out + size, value, toZmij(format));
    } else {
        result = zmij::to_chars(out, out + size, value, toZmij(format), precision);
    }
    if (result.ec != std::errc{}) {
        *out = '\0';
        return out;
    }
    return result.ptr;
}

std::string toString(double value, FloatFormat format, int precision) {
    char buffer[kMaxDoubleChars];
    char* end = write(value, buffer, sizeof(buffer), format, precision);
    return std::string(buffer, end);
}

std::to_chars_result to_chars(char* first, char* last, double value, FloatFormat format,
                              int precision) {
    char* end = write(value, first, static_cast<std::size_t>(last - first), format, precision);
    if (end == first && first != last) {
        return {last, std::errc::value_too_large};
    }
    return {end, std::errc{}};
}

} // namespace quantape::format
