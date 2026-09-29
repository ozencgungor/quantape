#pragma once

// fmt formatters (quill's bundled fmt, namespace fmtquill) for the
// format::Num and format::Matrix wrappers. Included by the logging facade, so
// QTA_LOG_* statements can log the wrappers directly:
//
//   QTA_LOG_INFO("test", "mid={} H00={}", format::num(mid), format::num(H00, 6));
//   QTA_LOG_INFO("test", "hessian\n{}", format::matrix(H));
//
// Only available when logging is enabled (quill provides fmt).

#include "quantape/format/Matrix.h"
#include "quantape/format/Number.h"

#include "quill/DeferredFormatCodec.h"
#include "quill/bundled/fmt/format.h"

#include <string>
#include <string_view>

namespace fmtquill {

template <>
struct formatter<quantape::format::Num, char> {
    constexpr auto parse(format_parse_context& ctx) { return ctx.begin(); }

    template <typename FormatContext>
    auto format(const quantape::format::Num& value, FormatContext& ctx) const {
        char buffer[quantape::format::kMaxDoubleChars];
        char* end = quantape::format::write(
            value.value, buffer, sizeof(buffer),
            value.precision < 0 ? quantape::format::FloatFormat::Shortest
                                : quantape::format::FloatFormat::General,
            value.precision);
        return fmtquill::format_to(ctx.out(), "{}", std::string_view(buffer, end - buffer));
    }
};

template <>
struct formatter<quantape::format::Matrix, char> {
    constexpr auto parse(format_parse_context& ctx) { return ctx.begin(); }

    template <typename FormatContext>
    auto format(const quantape::format::Matrix& value, FormatContext& ctx) const {
        if (!value.data) {
            return fmtquill::format_to(ctx.out(), "[]");
        }
        const std::string text =
            quantape::format::detail::formatMatrix(*value.data, value.options);
        return fmtquill::format_to(ctx.out(), "{}", std::string_view(text));
    }
};

}  // namespace fmtquill

// Queue codecs: both wrappers own their data (Num by value, Matrix by shared
// snapshot), so deferred formatting on the backend is safe.
template <>
struct quill::Codec<quantape::format::Num> : quill::DeferredFormatCodec<quantape::format::Num> {};

template <>
struct quill::Codec<quantape::format::Matrix>
    : quill::DeferredFormatCodec<quantape::format::Matrix> {};
