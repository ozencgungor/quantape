#include "quantape/log/Log.h"

#if !defined(QUANTAPE_DISABLE_LOGGING)

#include <atomic>
#include <memory>
#include <mutex>
#include <unordered_map>
#include <vector>

#include "quill/Backend.h"
#include "quill/Frontend.h"
#include "quill/Logger.h"
#include "quill/core/PatternFormatterOptions.h"
#include "quill/sinks/ConsoleSink.h"
#include "quill/sinks/RotatingFileSink.h"

namespace quantape::log {
namespace {

std::mutex g_mutex;
Config g_config{};
bool g_initialized = false;
std::atomic<Level> g_level{Level::Info};
std::vector<std::shared_ptr<quill::Sink>> g_sinks;
std::unordered_map<std::string, quill::Logger*> g_loggers;

quill::LogLevel toQuill(Level level) {
    switch (level) {
        case Level::Trace:
            return quill::LogLevel::TraceL1;
        case Level::Debug:
            return quill::LogLevel::Debug;
        case Level::Info:
            return quill::LogLevel::Info;
        case Level::Warn:
            return quill::LogLevel::Warning;
        case Level::Error:
            return quill::LogLevel::Error;
        case Level::Critical:
            return quill::LogLevel::Critical;
        case Level::Off:
            return quill::LogLevel::None;
    }
    return quill::LogLevel::Info;
}

quill::Logger* makeLoggerLocked(const std::string& name) {
    quill::PatternFormatterOptions options;
    if (!g_config.pattern.empty()) {
        options = quill::PatternFormatterOptions(g_config.pattern);
    }
    quill::Logger* logger = quill::Frontend::create_or_get_logger(name, g_sinks, options);
    logger->set_log_level(toQuill(g_level.load(std::memory_order_relaxed)));
    return logger;
}

void initializeLocked(const Config& config) {
    g_config = config;
    g_level.store(config.level, std::memory_order_relaxed);

    quill::Backend::start();

    if (config.console) {
        quill::ConsoleSinkConfig console_config;
        console_config.set_stream("stderr");
        console_config.set_colour_mode(config.colored
                                           ? quill::ConsoleSinkConfig::ColourMode::Always
                                           : quill::ConsoleSinkConfig::ColourMode::Never);
        g_sinks.push_back(quill::Frontend::create_or_get_sink<quill::ConsoleSink>(
            "quantape_console", console_config));
    }
    if (!config.file.empty()) {
        quill::RotatingFileSinkConfig file_config;
        file_config.set_rotation_max_file_size(config.maxFileBytes);
        file_config.set_max_backup_files(config.maxFiles);
        // For file sinks the sink name doubles as the file path.
        g_sinks.push_back(
            quill::Frontend::create_or_get_sink<quill::RotatingFileSink>(config.file, file_config));
    }
    if (g_sinks.empty()) {
        g_sinks.push_back(
            quill::Frontend::create_or_get_sink<quill::ConsoleSink>("quantape_console_fallback"));
    }

    g_initialized = true;
}

} // namespace

void initialize(const Config& config) {
    std::lock_guard lock(g_mutex);
    if (g_initialized) {
        g_level.store(config.level, std::memory_order_relaxed);
        for (auto& [name, logger] : g_loggers) {
            logger->set_log_level(toQuill(config.level));
        }
        return;
    }
    initializeLocked(config);
}

void shutdown() {
    std::lock_guard lock(g_mutex);
    if (!g_initialized) {
        return;
    }
    quill::Backend::stop();
    g_loggers.clear();
    g_sinks.clear();
    g_initialized = false;
}

void setLevel(Level level) {
    std::lock_guard lock(g_mutex);
    g_level.store(level, std::memory_order_relaxed);
    if (!g_initialized) {
        return;
    }
    for (auto& [name, logger] : g_loggers) {
        logger->set_log_level(toQuill(level));
    }
}

void setLevel(std::string_view component, Level level) {
    std::lock_guard lock(g_mutex);
    auto it = g_loggers.find(std::string(component));
    if (it != g_loggers.end()) {
        it->second->set_log_level(toQuill(level));
    }
}

bool active(Level level) {
    if (level == Level::Off) {
        return false;
    }
    return static_cast<int>(level) >= static_cast<int>(g_level.load(std::memory_order_relaxed));
}

Config config() {
    std::lock_guard lock(g_mutex);
    return g_config;
}

namespace detail {

quill::Logger* logger_for(std::string_view component) {
    std::lock_guard lock(g_mutex);
    if (!g_initialized) {
        initializeLocked(Config{});
    }
    std::string name(component);
    auto it = g_loggers.find(name);
    if (it != g_loggers.end()) {
        return it->second;
    }
    quill::Logger* logger = makeLoggerLocked(name);
    g_loggers.emplace(std::move(name), logger);
    return logger;
}

} // namespace detail

} // namespace quantape::log

#endif // QUANTAPE_DISABLE_LOGGING
