#pragma once

// Implementation header for the logging macros. Only included when logging is
// enabled (see Log.h); pulls in quill's macro API and the component registry.

#include "quantape/format/FmtFormatters.h" // format::num / format::matrix in logs

#include <string_view>

#include "quill/LogMacros.h"
#include "quill/Logger.h"
#include "quill/backend/ThreadUtilities.h" // inline get_thread_id/get_thread_name

namespace quantape::log::detail {

/// Returns the logger for a component (e.g. "quantape.time"), creating it and
/// the logging backend on first use. Thread-safe.
quill::Logger* logger_for(std::string_view component);

} // namespace quantape::log::detail

#define QTA_LOG_IMPL_(quill_macro, component, ...)                                                 \
    do {                                                                                           \
        ::quill::Logger* const _qta_logger = ::quantape::log::detail::logger_for(component);       \
        quill_macro(_qta_logger, __VA_ARGS__);                                                     \
    } while (0)

#define QTA_LOG_TRACE(component, ...)    QTA_LOG_IMPL_(QUILL_LOG_TRACE_L1, component, __VA_ARGS__)
#define QTA_LOG_DEBUG(component, ...)    QTA_LOG_IMPL_(QUILL_LOG_DEBUG, component, __VA_ARGS__)
#define QTA_LOG_INFO(component, ...)     QTA_LOG_IMPL_(QUILL_LOG_INFO, component, __VA_ARGS__)
#define QTA_LOG_WARN(component, ...)     QTA_LOG_IMPL_(QUILL_LOG_WARNING, component, __VA_ARGS__)
#define QTA_LOG_ERROR(component, ...)    QTA_LOG_IMPL_(QUILL_LOG_ERROR, component, __VA_ARGS__)
#define QTA_LOG_CRITICAL(component, ...) QTA_LOG_IMPL_(QUILL_LOG_CRITICAL, component, __VA_ARGS__)
