// test_logging.cpp — facade levels, file sink capture, runtime level changes
#include "quantape/log/Log.h"

#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <string>

#include "quantape/log/Log.h"
#include "quantape/util/Check.h"

int main() {
    namespace fs = std::filesystem;
    const fs::path log_path = fs::temp_directory_path() / "quantape_test_logging.log";
    std::error_code ec;
    fs::remove(log_path, ec);

    quantape::log::Config config;
    config.console = false;
    config.level = quantape::log::Level::Info;
    config.file = log_path.string();
    quantape::log::initialize(config);

    CHECK(quantape::log::active(quantape::log::Level::Info));
    CHECK(quantape::log::active(quantape::log::Level::Error));
    CHECK(!quantape::log::active(quantape::log::Level::Debug));

    QTA_LOG_DEBUG("test", "debug hidden {}", 1);
    QTA_LOG_INFO("test", "info visible {}", 2);
    QTA_LOG_WARN("test", "warn visible {}", 3.5);
    QTA_LOG_ERROR("test", "error visible {}", 4);

    quantape::log::setLevel(quantape::log::Level::Debug);
    CHECK(quantape::log::active(quantape::log::Level::Debug));
    QTA_LOG_DEBUG("test", "debug now visible {}", 5);

    quantape::log::shutdown();

    std::ifstream input(log_path);
    CHECK(input.good());
    const std::string contents((std::istreambuf_iterator<char>(input)),
                               std::istreambuf_iterator<char>());

    CHECK(contents.find("info visible 2") != std::string::npos);
    CHECK(contents.find("warn visible 3.5") != std::string::npos);
    CHECK(contents.find("error visible 4") != std::string::npos);
    CHECK(contents.find("debug now visible 5") != std::string::npos);
    CHECK(contents.find("debug hidden 1") == std::string::npos);
    CHECK(contents.find("test") != std::string::npos);

    fs::remove(log_path, ec);
    QTA_LOG_INFO("test", "test_logging: ok");
    return 0;
}
