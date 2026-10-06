// test_sobol_tools.cpp — integration tests for extend_sobol / refine_sobol / verify_sobol
//
// Gates:
//   - extend: output structure, prefix identity, net exactness (via verify),
//     byte-for-byte determinism, resume from a partial table
//   - refine: output valid, two-sided objective non-increasing, determinism,
//     dry-run writes nothing and never touches the input, refuses in-place
//   - verify: negative cases fail (even m, non-primitive/duplicate polynomial,
//     dimension gap, changed prefix) and random valid tables pass
//
// Tool paths are injected by CMake (EXTEND_BIN / REFINE_BIN / VERIFY_BIN).
#include "quantape/math/Random/Sobol/DirectionNumbers.h"
#include "quantape/math/Random/Sobol/GF2.h"

#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <random>
#include <string>
#include <vector>

#include "support/GtestSupport.h"

#ifndef EXTEND_BIN
#define EXTEND_BIN ""
#endif
#ifndef REFINE_BIN
#define REFINE_BIN ""
#endif
#ifndef VERIFY_BIN
#define VERIFY_BIN ""
#endif

using namespace quantape::math::mc;
using namespace quantape::math::mc::sobol;

namespace {

std::vector<Entry> makeFixture(int nDims, std::uint64_t seed) {
    std::mt19937_64 rng(seed);
    std::vector<Entry> out;
    std::uint32_t dim = 2;
    for (int degree = 1; degree <= 8 && static_cast<int>(out.size()) < nDims; ++degree) {
        const auto polys = gf2::enumerate_primitive(degree);
        for (const std::uint64_t poly : polys) {
            if (static_cast<int>(out.size()) >= nDims) {
                break;
            }
            Entry e;
            e.dim = dim++;
            e.s = static_cast<std::uint32_t>(degree);
            e.a = gf2::encode_a(poly, degree);
            e.m.resize(static_cast<std::size_t>(degree));
            for (int k = 1; k <= degree; ++k) {
                e.m[static_cast<std::size_t>(k - 1)] = ((rng() % (1ULL << (k - 1))) << 1) | 1;
            }
            out.push_back(std::move(e));
        }
    }
    return out;
}

void writeTable(const std::filesystem::path& path, const std::vector<Entry>& entries) {
    sobol::save_joe_kuo(path.string(), entries, false);
}

std::uint64_t fileHash(const std::filesystem::path& path) {
    std::ifstream in(path, std::ios::binary);
    std::uint64_t h = 1469598103934665603ULL;
    char c;
    while (in.get(c)) {
        h ^= static_cast<unsigned char>(c);
        h *= 1099511628211ULL;
    }
    return h;
}

/// Parse the max= of the verifier's criterion line (two-sided mode).
double maxTwoSided(const std::filesystem::path& path, int window) {
    const RunResult result =
        runTool(VERIFY_BIN, {"--input=" + path.string(), "--window=" + std::to_string(window),
                             "--all", "--twosided", "--m=12"});
    const std::size_t pos = result.out.rfind("max=");
    if (pos == std::string::npos) {
        ADD_FAILURE() << "no max= in verify output for " << path << "\nstdout:\n"
                      << result.out << "\nstderr:\n"
                      << result.err;
        return 0.0;
    }
    return std::strtod(result.out.c_str() + pos + 4, nullptr);
}

} // namespace

class SobolToolsTest : public TempDirTest {
protected:
    void SetUp() override {
        const bool configured = !std::string(EXTEND_BIN).empty() &&
                                !std::string(REFINE_BIN).empty() &&
                                !std::string(VERIFY_BIN).empty();
        SkipUnlessAsset(configured, "EXTEND_BIN/REFINE_BIN/VERIFY_BIN not configured");
        TempDirTest::SetUp();
    }
};

class SobolVerifyTest : public TempDirTest {
protected:
    void SetUp() override {
        SkipUnlessAsset(!std::string(VERIFY_BIN).empty(), "VERIFY_BIN not configured");
        TempDirTest::SetUp();
    }
};

TEST_F(SobolToolsTest, extendStructureNetExactnessDeterminismAndResume) {
    const auto basePath = dir() / "base.txt";
    const auto extPath = dir() / "ext.txt";
    const auto ext2Path = dir() / "ext2.txt";
    const auto resumedPath = dir() / "resumed.txt";

    const auto base = makeFixture(30, 1);
    writeTable(basePath, base);

    // ── extend: structure, verify, determinism, resume ──
    const std::vector<std::string> extendArgs{"--local",
                                              "--target=40",
                                              "--level=1",
                                              "--threads=1",
                                              "--window=8",
                                              "--candidates=8",
                                              "--input=" + basePath.string(),
                                              "--output=" + extPath.string()};
    const RunResult extend = runTool(EXTEND_BIN, extendArgs);
    EXPECT_EQ(extend.status, 0) << "stderr:\n" << extend.err;
    const auto extended = load_joe_kuo(extPath.string());
    EXPECT_EQ(extended.size(), 39u);
    EXPECT_EQ(extended.back().dim, 40u);

    const RunResult verify =
        runTool(VERIFY_BIN, {"--input=" + extPath.string(), "--base=" + basePath.string(),
                             "--sample=4", "--m=12"});
    EXPECT_EQ(verify.status, 0) << "stderr:\n" << verify.err;

    const RunResult rerun = runTool(EXTEND_BIN, extendArgs);
    EXPECT_EQ(rerun.status, 0) << "stderr:\n" << rerun.err;
    const RunResult extend2 =
        runTool(EXTEND_BIN, {"--local", "--target=40", "--level=1", "--threads=1", "--window=8",
                             "--candidates=8", "--input=" + basePath.string(),
                             "--output=" + ext2Path.string()});
    EXPECT_EQ(extend2.status, 0) << "stderr:\n" << extend2.err;
    EXPECT_EQ(fileHash(extPath), fileHash(ext2Path));

    // resume from the extended table to a larger target
    const RunResult resume =
        runTool(EXTEND_BIN, {"--local", "--target=44", "--level=1", "--threads=1", "--window=8",
                             "--candidates=8", "--input=" + extPath.string(),
                             "--output=" + resumedPath.string()});
    EXPECT_EQ(resume.status, 0) << "stderr:\n" << resume.err;
    const auto resumed = load_joe_kuo(resumedPath.string());
    EXPECT_EQ(resumed.size(), 43u);
    EXPECT_EQ(resumed.back().dim, 44u);
    const RunResult verifyResumed =
        runTool(VERIFY_BIN, {"--input=" + resumedPath.string(), "--base=" + basePath.string(),
                             "--sample=4", "--m=12"});
    EXPECT_EQ(verifyResumed.status, 0) << "stderr:\n" << verifyResumed.err;
}

TEST_F(SobolToolsTest, refineObjectiveNonIncreasingAndDeterministic) {
    const auto extPath = dir() / "ext.txt";
    const auto badPath = dir() / "bad.txt";
    const auto refPath = dir() / "ref.txt";
    const auto ref2Path = dir() / "ref2.txt";
    const auto basePath = dir() / "base.txt";
    writeTable(basePath, makeFixture(30, 1));
    {
        const RunResult extend =
            runTool(EXTEND_BIN, {"--local", "--target=40", "--level=1", "--threads=1", "--window=8",
                                 "--candidates=8", "--input=" + basePath.string(),
                                 "--output=" + extPath.string()});
        EXPECT_EQ(extend.status, 0) << "stderr:\n" << extend.err;
    }

    // ── refine: objective non-increasing, valid, deterministic ──
    auto bad = load_joe_kuo(extPath.string());
    // Inject a poor candidate: all m_i = 1 for one extension dimension.
    auto& victim = bad[34]; // dim 36
    for (auto& mk : victim.m) {
        mk = 1;
    }
    writeTable(badPath, bad);
    const double beforeMax = maxTwoSided(badPath, 8);

    const std::vector<std::string> refineArgs{"--input=" + badPath.string(),
                                              "--output=" + refPath.string(),
                                              "--refine-from=32",
                                              "--window=8",
                                              "--candidates=32",
                                              "--epochs=3",
                                              "--fraction=1.0",
                                              "--threads=2",
                                              "--seed=42"};
    const RunResult refine = runTool(REFINE_BIN, refineArgs);
    EXPECT_EQ(refine.status, 0) << "stderr:\n" << refine.err;
    EXPECT_TRUE(std::filesystem::exists(refPath));
    const double afterMax = maxTwoSided(refPath, 8);
    EXPECT_LE(afterMax, beforeMax * (1.0 + 1e-12));
    EXPECT_LT(afterMax, beforeMax); // the injected bad dimension must improve
    const RunResult verify =
        runTool(VERIFY_BIN, {"--input=" + refPath.string(), "--base=" + basePath.string(),
                             "--sample=4", "--m=12"});
    EXPECT_EQ(verify.status, 0) << "stderr:\n" << verify.err;

    // determinism
    const std::vector<std::string> refine2Args{"--input=" + badPath.string(),
                                               "--output=" + ref2Path.string(),
                                               "--refine-from=32",
                                               "--window=8",
                                               "--candidates=32",
                                               "--epochs=3",
                                               "--fraction=1.0",
                                               "--threads=2",
                                               "--seed=42"};
    const RunResult refine2 = runTool(REFINE_BIN, refine2Args);
    EXPECT_EQ(refine2.status, 0) << "stderr:\n" << refine2.err;
    EXPECT_EQ(fileHash(refPath), fileHash(ref2Path));
    ::testing::Test::RecordProperty("refine_before_max", quantape::util::num(beforeMax, 4));
    ::testing::Test::RecordProperty("refine_after_max", quantape::util::num(afterMax, 4));
}

TEST_F(SobolToolsTest, refineDryRunWritesNothingAndInPlaceRefused) {
    const auto extPath = dir() / "ext.txt";
    const auto badPath = dir() / "bad.txt";
    const auto dryPath = dir() / "dry.txt";
    const auto basePath = dir() / "base.txt";
    writeTable(basePath, makeFixture(30, 1));
    {
        const RunResult extend =
            runTool(EXTEND_BIN, {"--local", "--target=40", "--level=1", "--threads=1", "--window=8",
                                 "--candidates=8", "--input=" + basePath.string(),
                                 "--output=" + extPath.string()});
        EXPECT_EQ(extend.status, 0) << "stderr:\n" << extend.err;
    }
    auto bad = load_joe_kuo(extPath.string());
    auto& victim = bad[34]; // dim 36
    for (auto& mk : victim.m) {
        mk = 1;
    }
    writeTable(badPath, bad);
    const std::uint64_t beforeHash = fileHash(badPath);

    // dry-run: input untouched, no output written
    const RunResult dry =
        runTool(REFINE_BIN, {"--input=" + badPath.string(), "--output=" + dryPath.string(),
                             "--refine-from=32", "--window=8", "--candidates=16", "--epochs=1",
                             "--fraction=1.0", "--threads=2", "--dry-run"});
    EXPECT_EQ(dry.status, 0) << "stderr:\n" << dry.err;
    EXPECT_EQ(fileHash(badPath), beforeHash);
    EXPECT_FALSE(std::filesystem::exists(dryPath));

    // refuses in-place refinement
    const RunResult inplace =
        runTool(REFINE_BIN,
                {"--input=" + badPath.string(), "--output=" + badPath.string(), "--refine-from=32",
                 "--window=8", "--candidates=8", "--epochs=1", "--threads=1"});
    EXPECT_NE(inplace.status, 0) << "stdout:\n" << inplace.out << "\nstderr:\n" << inplace.err;
    EXPECT_EQ(fileHash(badPath), beforeHash);
}

TEST_F(SobolToolsTest, refineDryRunFuzzSeeds) {
    // Multi-seed fuzz: refine random tables in dry-run, no crashes, no writes.
    for (std::uint64_t seed = 1; seed <= 5; ++seed) {
        const auto path = dir() / ("fuzz_" + std::to_string(seed) + ".txt");
        writeTable(path, makeFixture(40, 1000 + seed));
        const std::uint64_t h = fileHash(path);
        const RunResult fuzz =
            runTool(REFINE_BIN, {"--input=" + path.string(), "--refine-from=20", "--window=8",
                                 "--candidates=16", "--epochs=1", "--fraction=1.0", "--threads=2",
                                 "--dry-run", "--seed=" + std::to_string(seed)});
        EXPECT_EQ(fuzz.status, 0) << "stderr:\n" << fuzz.err;
        EXPECT_EQ(fileHash(path), h);
    }
}

TEST_F(SobolVerifyTest, negativeCasesFail) {
    const auto bad = [&](const std::vector<Entry>& entries, const std::string& name) {
        const auto path = dir() / name;
        writeTable(path, entries);
        const RunResult result =
            runTool(VERIFY_BIN, {"--input=" + path.string(), "--sample=2", "--m=8"});
        return result.status != 0;
    };
    auto evenM = makeFixture(20, 7);
    evenM[0].m[0] = 2; // m_1 must be odd
    EXPECT_TRUE(bad(evenM, "even.txt"));

    auto nonPrim = makeFixture(20, 8);
    nonPrim[0].s = 3;
    nonPrim[0].a = 0; // x^3 + 1, reducible
    nonPrim[0].m = {1, 1, 1};
    EXPECT_TRUE(bad(nonPrim, "nonprim.txt"));

    auto dup = makeFixture(20, 9);
    dup[1] = dup[0];
    EXPECT_TRUE(bad(dup, "dup.txt"));

    auto gap = makeFixture(20, 10);
    gap.erase(gap.begin() + 1);
    EXPECT_TRUE(bad(gap, "gap.txt"));

    // changed prefix (valid structure, wrong vs --base)
    auto changed = makeFixture(20, 11);
    changed[3].m[1] = (changed[3].m[1] == 1) ? 3 : 1;
    const auto changedPath = dir() / "changed.txt";
    const auto base20Path = dir() / "base20.txt";
    writeTable(base20Path, makeFixture(20, 11));
    writeTable(changedPath, changed);
    const RunResult changedResult =
        runTool(VERIFY_BIN, {"--input=" + changedPath.string(), "--base=" + base20Path.string(),
                             "--sample=2", "--m=8"});
    EXPECT_NE(changedResult.status, 0) << "stdout:\n"
                                       << changedResult.out << "\nstderr:\n"
                                       << changedResult.err;
}

TEST_F(SobolVerifyTest, randomValidTablesPass) {
    for (std::uint64_t seed = 1; seed <= 10; ++seed) {
        const auto path = dir() / ("valid_" + std::to_string(seed) + ".txt");
        writeTable(path, makeFixture(40, seed * 7919));
        const RunResult result =
            runTool(VERIFY_BIN, {"--input=" + path.string(), "--sample=3", "--m=10"});
        EXPECT_EQ(result.status, 0) << "seed " << seed << " stderr:\n" << result.err;
    }
}
