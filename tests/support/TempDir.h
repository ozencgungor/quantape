#ifndef QUANTAPE_TESTS_SUPPORT_TEMP_DIR_H
#define QUANTAPE_TESTS_SUPPORT_TEMP_DIR_H

// TempDirTest: one pid/counter/case-unique scratch directory per test case.
//
// Never use fixed /tmp names in tests (they race under `ctest -j`); derive
// every output path from dir(). The directory is created in SetUp() and
// removed recursively in TearDown(), so cases stay independent and shuffle
// safe.

#include <atomic>
#include <filesystem>
#include <string>

#include <gtest/gtest.h>
#include <unistd.h>

namespace quantape::tests::detail {

inline std::string sanitizeTempComponent(const std::string& text) {
    std::string out = text;
    for (char& c : out) {
        const bool safe =
            (c >= 'A' && c <= 'Z') || (c >= 'a' && c <= 'z') || (c >= '0' && c <= '9') || c == '_';
        if (!safe) {
            c = '_';
        }
    }
    return out.empty() ? std::string("case") : out;
}

} // namespace quantape::tests::detail

class TempDirTest : public ::testing::Test {
protected:
    void SetUp() override {
        static std::atomic<unsigned> counter{0};
        const ::testing::TestInfo* info = ::testing::UnitTest::GetInstance()->current_test_info();
        const std::string caseName =
            info == nullptr ? "case" : quantape::tests::detail::sanitizeTempComponent(info->name());
        const std::string leaf = "quantape_test_" + std::to_string(::getpid()) + "_" +
                                 std::to_string(counter.fetch_add(1)) + "_" + caseName;
        m_dir = std::filesystem::path(::testing::TempDir()) / leaf;
        std::error_code error;
        std::filesystem::remove_all(m_dir, error);
        std::filesystem::create_directories(m_dir, error);
    }

    void TearDown() override {
        std::error_code error;
        std::filesystem::remove_all(m_dir, error);
    }

    /// Per-test scratch directory; create files under it, never in a fixed path.
    const std::filesystem::path& dir() const { return m_dir; }

private:
    std::filesystem::path m_dir;
};

#endif // QUANTAPE_TESTS_SUPPORT_TEMP_DIR_H
