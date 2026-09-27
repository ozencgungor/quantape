// test_sobol_quality.cpp — Joe-Kuo 2D t-values, Property A, search criterion
//
// Gates:
//   - fast tValue2D == brute-force rank definition (all splits, all m)
//   - published values: Table 2.1 of Joe-Kuo (2008), t(j, d; 12), d <= 8
//   - Property A (det(V_d) == 1 mod 2) against an independent determinant
//   - search: finite score, reproducibility for a fixed seed, and the level-1
//     pick is at least as good as every candidate (small exhaustive check)
//
// Stan/Eigen-free: standard library + the Sobol headers only.
#include "quantape/math/Random/Sobol/CBCSearch.h"

#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>

#define CHECK(cond)                                                                                \
    do {                                                                                           \
        if (!(cond)) {                                                                             \
            std::fprintf(stderr, "CHECK failed at %s:%d: %s\n", __FILE__, __LINE__, #cond);        \
            std::exit(1);                                                                          \
        }                                                                                          \
    } while (0)

using namespace quantape::math::mc;
using namespace quantape::math::mc::sobol;

namespace {

// First dimensions of the Joe-Kuo (2003) table [9]; Table 2.1 reference values.
std::vector<Entry> fixtures() {
    return {
        {2, 1, 0, {1}},          // d=2
        {3, 2, 1, {1, 1}},       // d=3
        {4, 3, 1, {1, 3, 7}},    // d=4
        {5, 3, 2, {1, 1, 5}},    // d=5
        {6, 4, 1, {1, 3, 1, 1}}, // d=6
        {7, 4, 4, {1, 1, 3, 7}}, // d=7
        {8, 5, 2, {1, 3, 3, 9, 9}},
    };
}

// High-degree entries from the D(7) table (dim 21199..21201); arbitrary
// direction numbers, so they are not part of the Property A fixture chain.
std::vector<Entry> highDegreeFixtures() {
    return {
        {21199,
         18,
         131008,
         {1, 1, 7, 13, 29, 5, 67, 119, 223, 307, 723, 2413, 6557, 6327, 307, 42517, 90863, 140327}},
        {21200,
         18,
         131020,
         {1, 1, 5, 5, 25, 35, 109, 171, 61, 165, 437, 1951, 4233, 5093, 2935, 55061, 117685,
          242593}},
        {21201,
         18,
         131059,
         {1, 3, 7, 5, 5, 21, 17, 197, 427, 425, 525, 2625, 1917, 403, 25733, 33363, 69927, 235799}},
    };
}

bool bruteFullRank(const std::uint32_t* rows, int n) {
    std::uint32_t pivots[32] = {};
    int rank = 0;
    for (int i = 0; i < n; ++i) {
        std::uint32_t v = rows[i];
        while (v != 0) {
            const int b = 31 - __builtin_clz(v);
            if (pivots[b] == 0) {
                pivots[b] = v;
                ++rank;
                break;
            }
            v ^= pivots[b];
        }
    }
    return rank == n;
}

/// Reference implementation straight from the paper's pseudocode (Figure 2.1).
int bruteTValue2D(const DirectionMatrix& first, const DirectionMatrix& second, int m) {
    const std::uint32_t mask = (m >= 32) ? 0xFFFFFFFFu : ((1u << m) - 1u);
    for (int t = 0; t <= m; ++t) {
        bool ok = true;
        for (int rj = 0; rj <= m - t && ok; ++rj) {
            const int rd = m - t - rj;
            std::uint32_t rows[32];
            int n = 0;
            for (int r = 0; r < rj; ++r) {
                rows[n++] = first.rows[static_cast<std::size_t>(r)] & mask;
            }
            for (int r = 0; r < rd; ++r) {
                rows[n++] = second.rows[static_cast<std::size_t>(r)] & mask;
            }
            ok = bruteFullRank(rows, n);
        }
        if (ok) {
            return t;
        }
    }
    return m;
}

/// Independent Property A check: rank of the d x d first-bit matrix V_d.
bool brutePropertyA(const std::vector<Entry>& entries, int d) {
    std::vector<std::uint32_t> rows(static_cast<std::size_t>(d), 0);
    std::vector<int> f;
    if (d >= 1) {
        rows[0] = 1u; // row of dimension 1: (1, 0, ..., 0)
    }
    for (int i = 2; i <= d; ++i) {
        f = firstBitSequence(entries[static_cast<std::size_t>(i - 2)], d);
        std::uint32_t row = 0;
        for (int k = 1; k <= d; ++k) {
            if (f[static_cast<std::size_t>(k)] != 0) {
                row |= (1u << (k - 1));
            }
        }
        rows[static_cast<std::size_t>(i - 1)] = row;
    }
    return bruteFullRank(rows.data(), d);
}

} // namespace

int main() {
    const auto entries = fixtures();

    // Direction matrices: dimension 1 is the identity.
    std::vector<DirectionMatrix> matrices(9);
    matrices[0] = identityMatrix();
    for (std::size_t i = 0; i < entries.size(); ++i) {
        matrices[i + 1] = directionMatrix(entries[i]);
    }
    for (int r = 0; r < 32; ++r) {
        CHECK(matrices[0].rows[static_cast<std::size_t>(r)] == (1u << r));
    }

    // Fast t-value == brute force (every pair, m = 1..12), and the published
    // Table 2.1 values at m = 12.
    const int table21[9][9] = {
        {},
        {},
        {0},
        {1, 1},
        {2, 2, 2},
        {2, 1, 2, 2},
        {2, 1, 2, 2, 2},
        {3, 2, 1, 2, 2, 1},
        {2, 3, 3, 2, 3, 2, 3},
    };
    int checked = 0;
    for (int d = 2; d <= 8; ++d) {
        for (int j = 1; j < d; ++j) {
            for (int m = 1; m <= 12; ++m) {
                const int fast = tValue2D(matrices[j - 1], matrices[d - 1], m);
                const int brute = bruteTValue2D(matrices[j - 1], matrices[d - 1], m);
                CHECK(fast == brute);
                ++checked;
            }
            CHECK(tValue2D(matrices[j - 1], matrices[d - 1], 12) == table21[d][j - 1]);
        }
    }
    std::printf("  [ok] tValue2D == brute force (%d checks), Table 2.1 values match\n", checked);

    // Property A: incremental checker agrees with an independent determinant.
    PropertyAChecker checker(64);
    CHECK(checker.add(entries[0])); // dimension 2
    for (std::size_t i = 1; i < entries.size(); ++i) {
        CHECK(checker.accepts(entries[i]));
        CHECK(checker.add(entries[i]));
    }
    for (int d = 1; d <= static_cast<int>(entries.size()) + 1; ++d) {
        CHECK(brutePropertyA(entries, d));
    }
    std::printf("  [ok] Property A checker matches independent determinant\n");

    // Search: finite deterministic score; the pick is optimal among the
    // evaluated candidates (seed pass plus pruned pass with a tiny budget).
    SearchContext context;
    context.criterion.window = 64;
    context.previous.push_back(identityMatrix());
    for (std::size_t i = 0; i < entries.size(); ++i) {
        context.previous.push_back(matrices[i + 1]);
    }
    context.weights.resize(context.previous.size());
    for (std::size_t i = 0; i < context.weights.size(); ++i) {
        context.weights[i] = std::pow(context.criterion.weightBase, static_cast<double>(i));
    }
    context.candidates = 16;

    const auto polys = gf2::enumerate_primitive(5);
    CHECK(!polys.empty());
    const std::uint64_t poly = polys[0];
    std::vector<WorkItem> items{{9, poly, 5, SearchLevel::WINDOWED, 0xabcULL}};

    auto ctxA = context;
    auto resA = process_batch(items, ctxA, 1);
    auto ctxB = context;
    auto resB = process_batch(items, ctxB, 1);
    CHECK(resA.size() == 1 && resB.size() == 1);
    CHECK(resA[0].score >= 0.0 && resA[0].score < kNoScore);
    CHECK(resA[0].m == resB[0].m);
    CHECK(resA[0].score == resB[0].score);

    // The chosen candidate must be at least as good as every candidate
    // regenerated from the same deterministic stream.
    {
        std::uint64_t state = 0xabcULL;
        const DirectionMatrix chosen = directionMatrixFor(poly, 5, resA[0].m);
        CHECK(criterionD(chosen, context.previous, context.weights, context.criterion) ==
              resA[0].score);
        for (int i = 0; i < 16; ++i) {
            const auto m = detail::randomDirectionSet(state, 5);
            const DirectionMatrix dm = directionMatrixFor(poly, 5, m);
            const double score =
                criterionD(dm, context.previous, context.weights, context.criterion);
            CHECK(resA[0].score <= score);
        }
    }
    std::printf("  [ok] search: deterministic, finite score, optimal among candidates\n");

    // Fuzz: fast t-value == brute force and symmetric for random valid direction
    // sets (degrees 1..12) and the high-degree fixtures, all m = 1..31.
    {
        std::mt19937_64 rng(0xC0FFEEULL);
        std::vector<DirectionMatrix> pool;
        for (int deg = 1; deg <= 12; ++deg) {
            const auto polys = gf2::enumerate_primitive(deg);
            CHECK(!polys.empty());
            for (int t = 0; t < 8; ++t) {
                Entry e;
                e.dim = 0;
                e.s = static_cast<uint32_t>(deg);
                e.a = gf2::encode_a(polys[rng() % polys.size()], deg);
                e.m.resize(static_cast<size_t>(deg));
                for (int k = 1; k <= deg; ++k) {
                    e.m[static_cast<size_t>(k - 1)] = ((rng() % (1ULL << (k - 1))) << 1) | 1;
                }
                pool.push_back(directionMatrix(e));
            }
        }
        const auto high = highDegreeFixtures();
        for (const auto& h : high) {
            pool.push_back(directionMatrix(h));
        }
        long checks = 0;
        for (int p = 0; p < 400; ++p) {
            const DirectionMatrix& a = pool[rng() % pool.size()];
            const DirectionMatrix& b = pool[rng() % pool.size()];
            for (int m = 1; m <= 31; ++m) {
                const int ab = tValue2D(a, b, m);
                CHECK(ab == tValue2D(b, a, m));
                CHECK(ab == bruteTValue2D(a, b, m));
                ++checks;
            }
        }
        std::printf("  [ok] fuzz: %ld checks fast==brute and symmetric (degrees 1-18)\n", checks);
    }

    std::printf("ALL SOBOL QUALITY TESTS PASSED\n");
    return 0;
}
