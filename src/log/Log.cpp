#include "quantape/log/Log.h"

#if !defined(QUANTAPE_DISABLE_LOGGING)

#include <atomic>
#include <cstdlib>
#include <memory>
#include <mutex>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

#include "quill/Backend.h"
#include "quill/Frontend.h"
#include "quill/Logger.h"
#include "quill/core/PatternFormatterOptions.h"
#include "quill/filters/Filter.h"
#include "quill/sinks/ConsoleSink.h"
#include "quill/sinks/RotatingFileSink.h"
#include "quill/sinks/Sink.h"

namespace quantape::log {
namespace {

constexpr char kTimestampPattern[] = "%H:%M:%S.%Qms";

std::mutex g_mutex;
Config g_config{};
bool g_initialized = false;
std::atomic<Level> g_level{Level::Info};
std::vector<std::shared_ptr<quill::Sink>> g_sinks_standard;  // info/debug + warn+
std::vector<std::shared_ptr<quill::Sink>> g_sinks_test;      // terse test/bench pair
std::unordered_map<std::string, quill::Logger*> g_loggers;

/// Test/benchmark components (names starting with test/bench) use the terse
/// header: no timestamp or component on info/debug, no component on warn+.
bool isTestComponent(std::string_view component) {
    return component.starts_with("test") || component.starts_with("bench");
}

/// Per-sink level range filter: the terse console sink keeps info/debug, the
/// diagnostic sink keeps warn+. Filters are disjoint so every message is
/// rendered exactly once with the pattern for its severity.
class LevelRangeFilter final : public quill::Filter {
public:
    LevelRangeFilter(const char* name, quill::LogLevel minLevel, quill::LogLevel maxLevel)
        : quill::Filter(name), min_(minLevel), max_(maxLevel) {}

    bool filter(quill::MacroMetadata const*, uint64_t, std::string_view, std::string_view,
                std::string_view, quill::LogLevel level, std::string_view,
                std::string_view) noexcept override {
        const int value = static_cast<int>(level);
        return value >= static_cast<int>(min_) && value <= static_cast<int>(max_);
    }

private:
    quill::LogLevel min_;
    quill::LogLevel max_;
};

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

std::string buildPattern(bool diagnostic, bool showTime, bool showComponent) {
    std::string pattern;
    if (showTime) {
        pattern += "%(time) ";
    }
    pattern += "[%(log_level_short_code)]";
    if (showComponent) {
        pattern += " %(logger)";
    }
    if (diagnostic) {
        pattern += " %(short_source_location)";
    }
    pattern += " %(message)";
    return pattern;
}

bool showTimeInfo(const Config& config, bool testLike) {
    switch (config.timestamps) {
        case TimestampDisplay::On:
            return true;
        case TimestampDisplay::Off:
            return false;
        case TimestampDisplay::Auto:
            return !testLike;
    }
    return true;
}

bool showTimeDiagnostic(const Config& config) {
    return config.timestamps != TimestampDisplay::Off;
}

bool showComponent(const Config& config, bool testLike) {
    switch (config.component) {
        case ComponentDisplay::Always:
            return true;
        case ComponentDisplay::Never:
            return false;
        case ComponentDisplay::Auto:
            return !testLike;
    }
    return true;
}

std::shared_ptr<quill::Sink> makeConsoleSink(const char* id, const Config& config,
                                             std::string pattern, quill::LogLevel minLevel,
                                             quill::LogLevel maxLevel) {
    quill::ConsoleSinkConfig sink_config;
    sink_config.set_stream("stderr");
    sink_config.set_colour_mode(config.colored ? quill::ConsoleSinkConfig::ColourMode::Always
                                               : quill::ConsoleSinkConfig::ColourMode::Never);
    sink_config.set_override_pattern_formatter_options(
        quill::PatternFormatterOptions{std::move(pattern), kTimestampPattern});
    auto sink = quill::Frontend::create_or_get_sink<quill::ConsoleSink>(id, sink_config);
    sink->add_filter(std::make_unique<LevelRangeFilter>(id, minLevel, maxLevel));
    return sink;
}

/// Environment overrides: QUANTAPE_LOG_PATTERN, QUANTAPE_LOG_DIAGNOSTIC_PATTERN,
/// QUANTAPE_LOG_TIMESTAMP (auto/on/off), QUANTAPE_LOG_COMPONENT (auto/on/off).
void applyEnvironmentOverrides(Config& config) {
    if (const char* value = std::getenv("QUANTAPE_LOG_PATTERN")) {
        config.pattern = value;
    }
    if (const char* value = std::getenv("QUANTAPE_LOG_DIAGNOSTIC_PATTERN")) {
        config.diagnosticPattern = value;
    }
    if (const char* value = std::getenv("QUANTAPE_LOG_TIMESTAMP")) {
        const std::string_view mode{value};
        if (mode == "on" || mode == "always") {
            config.timestamps = TimestampDisplay::On;
        } else if (mode == "off" || mode == "none") {
            config.timestamps = TimestampDisplay::Off;
        } else {
            config.timestamps = TimestampDisplay::Auto;
        }
    }
    if (const char* value = std::getenv("QUANTAPE_LOG_COMPONENT")) {
        const std::string_view mode{value};
        if (mode == "on" || mode == "always") {
            config.component = ComponentDisplay::Always;
        } else if (mode == "off" || mode == "never") {
            config.component = ComponentDisplay::Never;
        } else {
            config.component = ComponentDisplay::Auto;
        }
    }
}

quill::Logger* makeLoggerLocked(const std::string& name) {
    const auto& sinks = isTestComponent(name) ? g_sinks_test : g_sinks_standard;
    quill::Logger* logger =
        quill::Frontend::create_or_get_logger(name, sinks, quill::PatternFormatterOptions{});
    logger->set_log_level(toQuill(g_level.load(std::memory_order_relaxed)));
    return logger;
}

void initializeLocked(const Config& config) {
    g_config = config;
    applyEnvironmentOverrides(g_config);
    g_level.store(g_config.level, std::memory_order_relaxed);

    quill::Backend::start();

    const bool overridePatterns = !g_config.pattern.empty() || !g_config.diagnosticPattern.empty();
    std::string standard_info;
    std::string standard_diag;
    if (overridePatterns) {
        standard_info = g_config.pattern.empty()
                            ? buildPattern(false, showTimeInfo(g_config, false),
                                           showComponent(g_config, false))
                            : g_config.pattern;
        standard_diag = g_config.diagnosticPattern.empty()
                            ? buildPattern(true, showTimeDiagnostic(g_config),
                                           showComponent(g_config, false))
                            : g_config.diagnosticPattern;
    } else {
        standard_info = buildPattern(false, showTimeInfo(g_config, false),
                                     showComponent(g_config, false));
        standard_diag = buildPattern(true, showTimeDiagnostic(g_config),
                                     showComponent(g_config, false));
    }
    const std::string test_info = g_config.pattern.empty()
                                      ? buildPattern(false, showTimeInfo(g_config, true),
                                                     showComponent(g_config, true))
                                      : g_config.pattern;
    const std::string test_diag = g_config.diagnosticPattern.empty()
                                      ? buildPattern(true, showTimeDiagnostic(g_config),
                                                     showComponent(g_config, true))
                                      : g_config.diagnosticPattern;

    if (g_config.console) {
        g_sinks_standard.push_back(makeConsoleSink("quantape_console_info", g_config,
                                                   standard_info, quill::LogLevel::TraceL3,
                                                   quill::LogLevel::Notice));
        g_sinks_standard.push_back(makeConsoleSink("quantape_console_diag", g_config,
                                                   standard_diag, quill::LogLevel::Warning,
                                                   quill::LogLevel::Critical));
        g_sinks_test.push_back(makeConsoleSink("quantape_test_info", g_config, test_info,
                                               quill::LogLevel::TraceL3,
                                               quill::LogLevel::Notice));
        g_sinks_test.push_back(makeConsoleSink("quantape_test_diag", g_config, test_diag,
                                               quill::LogLevel::Warning,
                                               quill::LogLevel::Critical));
    }
    if (!g_config.file.empty()) {
        quill::RotatingFileSinkConfig file_config;
        file_config.set_rotation_max_file_size(g_config.maxFileBytes);
        file_config.set_max_backup_files(g_config.maxFiles);
        // For file sinks the sink name doubles as the file path. The file keeps
        // everything with the full diagnostic header so warnings are traceable.
        auto sink =
            quill::Frontend::create_or_get_sink<quill::RotatingFileSink>(g_config.file, file_config);
        g_sinks_standard.push_back(sink);
        g_sinks_test.push_back(std::move(sink));
    }
    if (g_sinks_standard.empty()) {
        auto fallback = makeConsoleSink("quantape_console_fallback", g_config, standard_info,
                                        quill::LogLevel::TraceL3, quill::LogLevel::Critical);
        g_sinks_standard.push_back(fallback);
        g_sinks_test.push_back(std::move(fallback));
    }

    g_initialized = true;
}

}  // namespace

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
    g_sinks_standard.clear();
    g_sinks_test.clear();
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
    return static_cast<int>(level) >=
           static_cast<int>(g_level.load(std::memory_order_relaxed));
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

}  // namespace detail

}  // namespace quantape::log

#endif  // QUANTAPE_DISABLE_LOGGING
