#ifndef QUANTAPE_TESTS_SUPPORT_DATA_FILES_H
#define QUANTAPE_TESTS_SUPPORT_DATA_FILES_H

// Helpers for tracked test fixtures (tests/data) and optional external assets
// (e.g. the Sobol direction-number table). QTA_TEST_DATA_DIR is injected by
// quantape_add_gtest; the helper adds tests/ to the target include path, so
// support headers are included as "support/...".

#include <filesystem>
#include <string>

#include <gtest/gtest.h>

/// Directory holding the tracked test data files.
inline std::filesystem::path testDataDir() {
#ifdef QTA_TEST_DATA_DIR
    return std::filesystem::path(QTA_TEST_DATA_DIR);
#else
    return std::filesystem::path("tests/data");
#endif
}

/// Resolves `path` against testDataDir() (absolute paths pass through) and
/// records a non-fatal failure with the resolved location when the file is
/// missing. Returns the resolved path so callers can still log it.
inline std::filesystem::path RequireDataFile(const std::filesystem::path& path) {
    const std::filesystem::path full = path.is_absolute() ? path : testDataDir() / path;
    std::error_code error;
    if (!std::filesystem::is_regular_file(full, error)) {
        ADD_FAILURE() << "required test data file missing: " << full.string()
                      << " (QTA_TEST_DATA_DIR=" << testDataDir().string() << ")";
    }
    return full;
}

/// Marks the test skipped when `present` is false. GTEST_SKIP() only returns
/// from the helper it is called in, so test bodies must use SKIP_UNLESS_ASSET;
/// this function is for SetUp()/fixture hooks where returning is enough.
inline void SkipUnlessAsset(bool present,
                            const std::string& reason = "required test asset unavailable") {
    if (!present) {
        GTEST_SKIP() << reason;
    }
}

/// Skips the enclosing test (returns from the test body) when `present` is
/// false: `SKIP_UNLESS_ASSET(SobolGenerator::hasDefaultTable(), "no table");`.
#define SKIP_UNLESS_ASSET(present, reason)                                                         \
    do {                                                                                           \
        if (!(present)) {                                                                          \
            GTEST_SKIP() << reason;                                                                \
        }                                                                                          \
    } while (false)

#endif // QUANTAPE_TESTS_SUPPORT_DATA_FILES_H
