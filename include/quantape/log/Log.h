#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <string_view>

namespace quantape::log {

enum class Level : std::uint8_t { Trace, Debug, Info, Warn, Error, Critical, Off };

struct Config {
    Level level = Level::Info;
    bool console = true;
    bool colored = true;
    std::string file; // empty = no file sink
    std::size_t maxFileBytes = 64ull * 1024 * 1024;
    std::uint32_t maxFiles = 5;
    std::string pattern; // empty = library default pattern
};

/// Configures logging. Idempotent: subsequent calls only update the level.
/// Called implicitly with defaults on the first log statement.
void initialize(const Config& config);

/// Flushes and stops the background worker. Called automatically at exit.
void shutdown();

void setLevel(Level level);
void setLevel(std::string_view component, Level level);

/// True when a statement at `level` would be emitted; use to guard expensive
/// argument computation in hot paths.
bool active(Level level);

Config config();

} // namespace quantape::log

#if defined(QUANTAPE_DISABLE_LOGGING)
#define QTA_LOG_TRACE(component, ...)    ((void)0)
#define QTA_LOG_DEBUG(component, ...)    ((void)0)
#define QTA_LOG_INFO(component, ...)     ((void)0)
#define QTA_LOG_WARN(component, ...)     ((void)0)
#define QTA_LOG_ERROR(component, ...)    ((void)0)
#define QTA_LOG_CRITICAL(component, ...) ((void)0)
#else
#include "quantape/log/LogImpl.h"
#endif
