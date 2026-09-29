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

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <random>
#include <string>
#include <vector>

#include "TestSupport.h"
#include <unistd.h>

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

int run(const std::string& cmd) {
    const int rc = std::system(cmd.c_str());
    return rc;
}

/// Parse the max= of the verifier's criterion line (two-sided mode).
double maxTwoSided(const std::filesystem::path& path, int window) {
    const std::string cmd = std::string(VERIFY_BIN) + " --input=\"" + path.string() +
                            "\" --window=" + std::to_string(window) +
                            " --all --twosided --m=12 2>/dev/null";
    FILE* pipe = popen(cmd.c_str(), "r");
    CHECK(pipe != nullptr);
    std::string output;
    char buf[4096];
    while (std::fgets(buf, sizeof(buf), pipe) != nullptr) {
        output += buf;
    }
    pclose(pipe);
    const auto pos = output.rfind("max=");
    CHECK(pos != std::string::npos);
    return std::strtod(output.c_str() + pos + 4, nullptr);
}

} // namespace

int main() {
    CHECK(std::string(EXTEND_BIN).size() > 0);
    CHECK(std::string(REFINE_BIN).size() > 0);
    CHECK(std::string(VERIFY_BIN).size() > 0);

    const auto tmp =
        std::filesystem::temp_directory_path() / ("sobol_tools_test_" + std::to_string(::getpid()));
    std::filesystem::remove_all(tmp);
    std::filesystem::create_directories(tmp);

    const auto basePath = tmp / "base.txt";
    const auto extPath = tmp / "ext.txt";
    const auto ext2Path = tmp / "ext2.txt";
    const auto resumedPath = tmp / "resumed.txt";
    const auto badPath = tmp / "bad.txt";
    const auto refPath = tmp / "ref.txt";
    const auto ref2Path = tmp / "ref2.txt";
    const auto dryPath = tmp / "dry.txt";

    const auto base = makeFixture(30, 1);
    writeTable(basePath, base);

    // ── extend: structure, verify, determinism, resume ──
    {
        const std::string cmd = std::string(EXTEND_BIN) +
                                " --local --target=40 --level=1 "
                                "--threads=1 --window=8 --candidates=8 \
--input=\"" + basePath.string() +
                                "\" --output=\"" + extPath.string() + "\" 2>/dev/null";
        CHECK(run(cmd) == 0);
        const auto extended = load_joe_kuo(extPath.string());
        CHECK(extended.size() == 39 && extended.back().dim == 40);

        const std::string verify = std::string(VERIFY_BIN) + " --input=\"" + extPath.string() +
                                   "\" --base=\"" + basePath.string() +
                                   "\" --sample=4 --m=12 2>/dev/null";
        CHECK(run(verify) == 0);

        CHECK(run(cmd) == 0); // ext2 is not written; rerun to a second file
        const std::string cmd2 = std::string(EXTEND_BIN) +
                                 " --local --target=40 --level=1 "
                                 "--threads=1 --window=8 --candidates=8 \
--input=\"" + basePath.string() + "\" --output=\"" +
                                 ext2Path.string() + "\" 2>/dev/null";
        CHECK(run(cmd2) == 0);
        CHECK(fileHash(extPath) == fileHash(ext2Path));

        // resume from the extended table to a larger target
        const std::string cmd3 = std::string(EXTEND_BIN) +
                                 " --local --target=44 --level=1 "
                                 "--threads=1 --window=8 --candidates=8 \
--input=\"" + extPath.string() + "\" --output=\"" +
                                 resumedPath.string() + "\" 2>/dev/null";
        CHECK(run(cmd3) == 0);
        const auto resumed = load_joe_kuo(resumedPath.string());
        CHECK(resumed.size() == 43 && resumed.back().dim == 44);
        CHECK(run(std::string(VERIFY_BIN) + " --input=\"" + resumedPath.string() + "\" --base=\"" +
                  basePath.string() + "\" --sample=4 --m=12 2>/dev/null") == 0);
        QTA_LOG_INFO("quantape.test",
                     "  [ok] extend: structure, net exactness, determinism, resume");
    }

    // ── refine: objective non-increasing, valid, deterministic, dry-run ──
    {
        auto bad = load_joe_kuo(extPath.string());
        // Inject a poor candidate: all m_i = 1 for one extension dimension.
        auto& victim = bad[34]; // dim 36
        for (auto& mk : victim.m) {
            mk = 1;
        }
        writeTable(badPath, bad);
        const std::uint64_t beforeHash = fileHash(badPath);
        const double beforeMax = maxTwoSided(badPath, 8);

        const std::string refine = std::string(REFINE_BIN) + " --input=\"" + badPath.string() +
                                   "\" --output=\"" + refPath.string() +
                                   "\" --refine-from=32 --window=8 --candidates=32 --epochs=3 "
                                   "--fraction=1.0 --threads=2 --seed=42 2>/dev/null";
        CHECK(run(refine) == 0);
        CHECK(std::filesystem::exists(refPath));
        const double afterMax = maxTwoSided(refPath, 8);
        CHECK(afterMax <= beforeMax * (1.0 + 1e-12));
        CHECK(afterMax < beforeMax); // the injected bad dimension must improve
        CHECK(run(std::string(VERIFY_BIN) + " --input=\"" + refPath.string() + "\" --base=\"" +
                  basePath.string() + "\" --sample=4 --m=12 2>/dev/null") == 0);

        // determinism
        const std::string refine2 = std::string(REFINE_BIN) + " --input=\"" + badPath.string() +
                                    "\" --output=\"" + ref2Path.string() +
                                    "\" --refine-from=32 --window=8 --candidates=32 --epochs=3 "
                                    "--fraction=1.0 --threads=2 --seed=42 2>/dev/null";
        CHECK(run(refine2) == 0);
        CHECK(fileHash(refPath) == fileHash(ref2Path));

        // dry-run: input untouched, no output written
        const std::string dry = std::string(REFINE_BIN) + " --input=\"" + badPath.string() +
                                "\" --output=\"" + dryPath.string() +
                                "\" --refine-from=32 --window=8 --candidates=16 --epochs=1 "
                                "--fraction=1.0 --threads=2 --dry-run 2>/dev/null";
        CHECK(run(dry) == 0);
        CHECK(fileHash(badPath) == beforeHash);
        CHECK(!std::filesystem::exists(dryPath));

        // refuses in-place refinement
        const std::string inplace = std::string(REFINE_BIN) + " --input=\"" + badPath.string() +
                                    "\" --output=\"" + badPath.string() +
                                    "\" --refine-from=32 --window=8 --candidates=8 --epochs=1 "
                                    "--threads=1 2>/dev/null";
        CHECK(run(inplace) != 0);
        CHECK(fileHash(badPath) == beforeHash);
        QTA_LOG_INFO("quantape.test",
                     "  [ok] refine: objective {} -> {}, valid, deterministic, dry-run safe",
                     quantape_test::num(beforeMax, 4), quantape_test::num(afterMax, 4));

        // Multi-seed fuzz: refine random tables in dry-run, no crashes, no writes.
        for (std::uint64_t seed = 1; seed <= 5; ++seed) {
            const auto path = tmp / ("fuzz_" + std::to_string(seed) + ".txt");
            writeTable(path, makeFixture(40, 1000 + seed));
            const std::uint64_t h = fileHash(path);
            const std::string fuzz = std::string(REFINE_BIN) + " --input=\"" + path.string() +
                                     "\" --refine-from=20 --window=8 --candidates=16 --epochs=1 "
                                     "--fraction=1.0 --threads=2 --dry-run --seed=" +
                                     std::to_string(seed) + " 2>/dev/null";
            CHECK(run(fuzz) == 0);
            CHECK(fileHash(path) == h);
        }
        QTA_LOG_INFO("quantape.test", "  [ok] refine fuzz: 5 seeds, dry-run, no writes");
    }

    // ── verify negative cases ──
    {
        const auto bad = [&](const std::vector<Entry>& entries, const std::string& name) {
            const auto path = tmp / name;
            writeTable(path, entries);
            return run(std::string(VERIFY_BIN) + " --input=\"" + path.string() +
                       "\" --sample=2 --m=8 2>/dev/null") != 0;
        };
        auto evenM = makeFixture(20, 7);
        evenM[0].m[0] = 2; // m_1 must be odd
        CHECK(bad(evenM, "even.txt"));

        auto nonPrim = makeFixture(20, 8);
        nonPrim[0].s = 3;
        nonPrim[0].a = 0; // x^3 + 1, reducible
        nonPrim[0].m = {1, 1, 1};
        CHECK(bad(nonPrim, "nonprim.txt"));

        auto dup = makeFixture(20, 9);
        dup[1] = dup[0];
        CHECK(bad(dup, "dup.txt"));

        auto gap = makeFixture(20, 10);
        gap.erase(gap.begin() + 1);
        CHECK(bad(gap, "gap.txt"));

        // changed prefix (valid structure, wrong vs --base)
        auto changed = makeFixture(20, 11);
        changed[3].m[1] = (changed[3].m[1] == 1) ? 3 : 1;
        const auto changedPath = tmp / "changed.txt";
        const auto base20Path = tmp / "base20.txt";
        writeTable(base20Path, makeFixture(20, 11));
        writeTable(changedPath, changed);
        CHECK(run(std::string(VERIFY_BIN) + " --input=\"" + changedPath.string() + "\" --base=\"" +
                  base20Path.string() + "\" --sample=2 --m=8 2>/dev/null") != 0);

        // random valid tables pass
        for (std::uint64_t seed = 1; seed <= 10; ++seed) {
            const auto path = tmp / ("valid_" + std::to_string(seed) + ".txt");
            writeTable(path, makeFixture(40, seed * 7919));
            CHECK(run(std::string(VERIFY_BIN) + " --input=\"" + path.string() +
                      "\" --sample=3 --m=10 2>/dev/null") == 0);
        }
        QTA_LOG_INFO("quantape.test",
                     "  [ok] verify: 5 negative cases fail, 10 random valid tables pass");
    }

    std::filesystem::remove_all(tmp);
    QTA_LOG_INFO("quantape.test", "ALL SOBOL TOOL TESTS PASSED");
    return 0;
}
