// test_logging.cpp — facade levels, file sink capture, runtime level changes
#include "quantape/log/Log.h"

#include <filesystem>
#include <fstream>
#include <iterator>
#include <string>

#include "support/GtestSupport.h"

namespace {

/// Process-scoped log sink: quill starts its backend with std::call_once and
/// cannot restart after shutdown(), so the first TempDirTest directory owns the
/// log file for the whole process (later cases and --gtest_repeat reuse it).
/// A static destructor removes the directory at process exit; the fixture still
/// owns initialize() and resets the level before every case.
class LoggingTest : public TempDirTest {
protected:
    void SetUp() override {
        TempDirTest::SetUp();
        if (s_logPath.empty()) {
            s_logPath = dir() / "quantape_test_logging.log";
            quantape::log::Config config;
            config.console = false;
            config.level = quantape::log::Level::Info;
            config.file = s_logPath.string();
            quantape::log::initialize(config);

            static const bool cleanup = [] {
                struct RemoveDir {
                    ~RemoveDir() {
                        std::error_code error;
                        std::filesystem::remove_all(s_logPath.parent_path(), error);
                    }
                };
                static const RemoveDir remover;
                return true;
            }();
            (void)cleanup;
        } else {
            quantape::log::setLevel(quantape::log::Level::Info);
        }
    }

    void TearDown() override {
        if (dir() != logPath().parent_path()) {
            TempDirTest::TearDown();
        }
    }

    const std::filesystem::path& logPath() const { return s_logPath; }

    /// Blocks until the backend has written every "test" message to the sink.
    static void flush() { quantape::log::detail::logger_for("test")->flush_log(); }

private:
    static std::filesystem::path s_logPath;
};

std::filesystem::path LoggingTest::s_logPath;

} // namespace

TEST_F(LoggingTest, levelActivationAndRuntimeChange) {
    EXPECT_TRUE(quantape::log::active(quantape::log::Level::Info));
    EXPECT_TRUE(quantape::log::active(quantape::log::Level::Error));
    EXPECT_FALSE(quantape::log::active(quantape::log::Level::Debug));

    quantape::log::setLevel(quantape::log::Level::Debug);
    EXPECT_TRUE(quantape::log::active(quantape::log::Level::Debug));
}

TEST_F(LoggingTest, fileSinkContent) {
    QTA_LOG_DEBUG("test", "debug hidden {}", 1);
    QTA_LOG_INFO("test", "info visible {}", 2);
    QTA_LOG_WARN("test", "warn visible {}", 3.5);
    QTA_LOG_ERROR("test", "error visible {}", 4);

    quantape::log::setLevel(quantape::log::Level::Debug);
    QTA_LOG_DEBUG("test", "debug now visible {}", 5);
    flush();

    std::ifstream input(logPath());
    EXPECT_TRUE(input.good());
    const std::string contents((std::istreambuf_iterator<char>(input)),
                               std::istreambuf_iterator<char>());

    EXPECT_NE(contents.find("info visible 2"), std::string::npos);
    EXPECT_NE(contents.find("warn visible 3.5"), std::string::npos);
    EXPECT_NE(contents.find("error visible 4"), std::string::npos);
    EXPECT_NE(contents.find("debug now visible 5"), std::string::npos);
    EXPECT_EQ(contents.find("debug hidden 1"), std::string::npos);
    EXPECT_NE(contents.find("test"), std::string::npos);
}
