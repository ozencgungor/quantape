/**
 * @file verify_sobol.cpp
 * @brief Statistical and structural verification of an extended Joe-Kuo table.
 *
 * Standalone (standard library + the Sobol headers). Checks:
 *
 *   1. Structure: dimensions continuous from 2, m-count == degree, m_k odd and
 *      < 2^k, polynomials primitive and distinct, prefix unchanged vs --base.
 *   2. Criterion: for sampled dimensions d, D(q)(d) against the preceding
 *      --window dimensions (Joe-Kuo 2008), reported as a distribution.
 *   3. Net exactness (the meaningful uniformity test for digital nets): for
 *      pairs (j, d) with computed t-value t at level m, every cell of a
 *      2^{b1} x 2^{b2} grid with b1 + b2 = m - t must contain exactly 2^t
 *      points of the first 2^m points. Also 1D exact counts, moments and bit
 *      balance. (A chi-square null test is inappropriate here: Sobol points
 *      are deterministic nets, structured at resolutions beyond the guarantee.)
 *   4. Cross-check: if --sobol-cc is given, Kuo's reference generator is run on
 *      the same file and its points are compared column-by-column with ours.
 *
 * Exit code 1 on hard failures.
 *
 * Usage:
 *   verify_sobol --input joe-kuo-extended-65536-w256.txt \
 *       [--base internal_docs/new-joe-kuo-7.21201] \
 *       [--window 256] [--sample 16] [--m 16] \
 *       [--sobol-cc build/sobol_ref] [--all]
 */
#include "quantape/log/Log.h"
#include "quantape/math/Random/Sobol/DirectionNumbers.h"
#include "quantape/math/Random/Sobol/GF2.h"
#include "quantape/math/Random/Sobol/SobolQuality.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <string>
#include <unordered_set>
#include <vector>

using namespace quantape::math::mc;
using namespace quantape::math::mc::sobol;

namespace {

// ── 32-bit gray-code generation (reference convention of sobol.cc) ──

struct GrayGenerator {
    /// V[i] = m_i << (32 - i); L = ceil(log2(N)).
    GrayGenerator(const Entry& e, std::uint32_t N) : N_(N) {
        L_ = 1;
        while ((1u << L_) < N_) {
            ++L_;
        }
        const auto m = integerDirectionNumbers(e, L_);
        V_.assign(static_cast<std::size_t>(L_) + 1, 0);
        for (int i = 1; i <= L_; ++i) {
            V_[static_cast<std::size_t>(i)] =
                static_cast<std::uint32_t>(m[static_cast<std::size_t>(i)] << (32 - i));
        }
        initC();
    }

    /// Dimension 1 is special: all direction numbers are 1/2^k.
    GrayGenerator(std::uint32_t N, bool /*dimensionOne*/) : N_(N) {
        L_ = 1;
        while ((1u << L_) < N_) {
            ++L_;
        }
        V_.assign(static_cast<std::size_t>(L_) + 1, 0);
        for (int i = 1; i <= L_; ++i) {
            V_[static_cast<std::size_t>(i)] = 1u << (32 - i);
        }
        initC();
    }

    void initC() {
        C_.assign(N_, 0);
        C_[0] = 1;
        for (std::uint32_t i = 1; i < N_; ++i) {
            std::uint32_t value = i;
            std::uint32_t c = 1;
            while (value & 1u) {
                value >>= 1;
                ++c;
            }
            C_[i] = c;
        }
    }

    /// X values (/2^32) of the first N gray-code points.
    std::vector<std::uint32_t> run() const {
        std::vector<std::uint32_t> x(N_, 0);
        for (std::uint32_t i = 1; i < N_; ++i) {
            x[i] = x[i - 1] ^ V_[C_[i - 1]];
        }
        return x;
    }

    std::uint32_t N_;
    int L_ = 1;
    std::vector<std::uint32_t> V_;
    std::vector<std::uint32_t> C_;
};

/// Count occupancies of a 2^b1 x 2^b2 grid; returns true if every cell has
/// exactly `expected` hits (digital-net exactness).
bool exactGrid(const std::vector<std::uint32_t>& a, const std::vector<std::uint32_t>& b, int b1,
               int b2, std::uint32_t expected, std::uint32_t* worstSeen) {
    const std::size_t cells = std::size_t(1) << (b1 + b2);
    std::vector<std::uint32_t> counts(cells, 0);
    for (std::size_t i = 0; i < a.size(); ++i) {
        const std::uint32_t x = (b1 == 0) ? 0u : (a[i] >> (32 - b1));
        const std::uint32_t y = (b2 == 0) ? 0u : (b[i] >> (32 - b2));
        counts[(static_cast<std::size_t>(x) << b2) | y]++;
    }
    bool ok = true;
    std::uint32_t worst = 0;
    for (std::uint32_t c : counts) {
        if (c != expected)
            ok = false;
        worst = std::max(worst, c);
    }
    if (worstSeen)
        *worstSeen = worst;
    return ok;
}

} // namespace

int main(int argc, char** argv) {
    std::string input, base, sobolCc;
    int window = 256;
    int sample = 16;
    int mLevel = 16;             // 2^m points for net tests
    std::uint32_t crossLog2 = 6; // sobol.cc cross-check points (2^k)
    bool all = false;
    bool twoSided = false; // score against both sides with distance weights
    double weightBase = 0.9999;

    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        const auto value = [&]() -> std::string {
            const auto pos = arg.find('=');
            return pos == std::string::npos ? std::string() : arg.substr(pos + 1);
        };
        if (arg.rfind("--input=", 0) == 0)
            input = value();
        else if (arg.rfind("--base=", 0) == 0)
            base = value();
        else if (arg.rfind("--sobol-cc=", 0) == 0)
            sobolCc = value();
        else if (arg.rfind("--window=", 0) == 0)
            window = std::stoi(value());
        else if (arg.rfind("--sample=", 0) == 0)
            sample = std::stoi(value());
        else if (arg.rfind("--m=", 0) == 0)
            mLevel = std::stoi(value());
        else if (arg.rfind("--cross-points-log2=", 0) == 0)
            crossLog2 = static_cast<std::uint32_t>(std::stoi(value()));
        else if (arg == "--all")
            all = true;
        else if (arg == "--twosided")
            twoSided = true;
        else if (arg.rfind("--weight=", 0) == 0)
            weightBase = std::stod(value());
        else if (arg == "--help" || arg == "-h") {
            std::printf("usage: verify_sobol --input=FILE [--base=FILE] [--window=N] [--sample=K]\n"
                        "                    [--m=N] [--cross-points-log2=k] [--sobol-cc=BIN] "
                        "[--all]\n");
            return 0;
        } else {
            QTA_LOG_ERROR("quantape.tools", "unknown option: {}", arg);
            return 1;
        }
    }
    if (input.empty()) {
        QTA_LOG_ERROR("quantape.tools", "--input required");
        return 1;
    }

    int failures = 0;
    const auto entries = load_joe_kuo(input);
    if (entries.empty()) {
        QTA_LOG_ERROR("quantape.tools", "could not load {}", input);
        return 1;
    }
    std::printf("loaded %zu entries (dims 2..%u)\n", entries.size(), entries.back().dim);

    // ── 1. Structure ──
    {
        int bad = 0;
        std::unordered_set<std::uint64_t> polys;
        for (std::size_t i = 0; i < entries.size(); ++i) {
            const Entry& e = entries[i];
            if (e.dim != i + 2) {
                if (bad++ < 5)
                    QTA_LOG_WARN("quantape.tools", "dim discontinuity at index {}: dim={}", i,
                                 e.dim);
                continue;
            }
            if (e.m.size() != e.s) {
                if (bad++ < 5)
                    QTA_LOG_WARN("quantape.tools", "dim {}: m count {} != degree {}", e.dim,
                                 e.m.size(), e.s);
                continue;
            }
            for (std::uint32_t k = 1; k <= e.s; ++k) {
                const std::uint64_t mk = e.m[k - 1];
                if ((mk & 1u) == 0 || mk >= (1ULL << k)) {
                    if (bad++ < 5)
                        QTA_LOG_WARN("quantape.tools", "dim {}: m_{}={} invalid", e.dim, k, mk);
                    break;
                }
            }
            const std::uint64_t poly = gf2::decode_poly(static_cast<int>(e.s), e.a);
            if (!polys.insert(poly).second) {
                if (bad++ < 5)
                    QTA_LOG_WARN("quantape.tools", "duplicate polynomial at dim {}", e.dim);
            }
            if (!gf2::is_primitive(poly, static_cast<int>(e.s))) {
                if (bad++ < 5)
                    QTA_LOG_WARN("quantape.tools", "non-primitive polynomial at dim {}", e.dim);
            }
        }
        std::printf("structure: %s (%d violations)\n", bad ? "FAIL" : "ok", bad);
        failures += bad != 0;
    }

    // ── 1b. Prefix unchanged ──
    std::uint32_t firstNew = 2;
    if (!base.empty()) {
        const auto prefix = load_joe_kuo(base);
        int changed = 0;
        for (std::size_t i = 0; i < prefix.size() && i < entries.size(); ++i) {
            if (entries[i].dim != prefix[i].dim || entries[i].s != prefix[i].s ||
                entries[i].a != prefix[i].a || entries[i].m != prefix[i].m)
                ++changed;
        }
        firstNew = static_cast<std::uint32_t>(prefix.size() + 2);
        std::printf("prefix (dims <= %zu): %s (%d changed)\n", prefix.size(),
                    changed ? "FAIL" : "identical", changed);
        failures += changed != 0;
    }
    const std::uint32_t lastDim = entries.back().dim;
    const std::uint32_t nNew = lastDim >= firstNew ? lastDim - firstNew + 1 : 0;

    // Direction matrices for all dimensions (index i = dimension i+1).
    std::vector<DirectionMatrix> matrices(entries.size() + 2);
    matrices[0] = identityMatrix();
    for (std::size_t i = 0; i < entries.size(); ++i) {
        matrices[i + 1] = directionMatrix(entries[i]);
    }

    // ── 2. Criterion scores for sampled dimensions ──
    if (nNew == 0) {
        std::printf("criterion: no extension dimensions (input equals/under the base prefix)\n");
    } else {
        const int k = all ? static_cast<int>(nNew) : std::min<int>(sample, static_cast<int>(nNew));
        std::vector<double> scores;
        std::vector<double> weights(matrices.size());
        for (std::size_t i = 0; i < weights.size(); ++i) {
            weights[i] = std::pow(0.9999, static_cast<double>(i));
        }
        CriterionOptions options;
        options.window = window;
        for (int s = 0; s < k; ++s) {
            const std::uint32_t idx =
                all ? firstNew + static_cast<std::uint32_t>(s)
                    : firstNew +
                          static_cast<std::uint32_t>((static_cast<std::uint64_t>(s) * nNew) / k);
            if (twoSided) {
                // Symmetric windowed D(q) with distance weights (refine objective).
                const int centre = static_cast<int>(idx - 1);
                const int lo = std::max(0, centre - window);
                const int hi = std::min(static_cast<int>(matrices.size()) - 2, centre + window);
                double best = 0.0;
                for (int m = options.mMin; m <= options.mMax; ++m) {
                    double worst = 0.0;
                    for (int i = lo; i <= hi; ++i) {
                        if (i == centre)
                            continue;
                        const double w = std::pow(weightBase, std::abs(i - centre));
                        const double bound =
                            std::min(static_cast<double>(m),
                                     static_cast<double>(
                                         matrices[static_cast<std::size_t>(i)].degree +
                                         matrices[static_cast<std::size_t>(centre)].degree - 2)) *
                            w;
                        if (bound <= worst)
                            continue;
                        const int t = tValue2D(matrices[static_cast<std::size_t>(i)],
                                               matrices[static_cast<std::size_t>(centre)], m);
                        if (t == 0)
                            continue;
                        worst = std::max(worst, static_cast<double>(t) * w);
                    }
                    best = std::max(best, std::pow(worst, options.exponent) /
                                              (static_cast<double>(m) - worst + 1.0));
                }
                scores.push_back(best);
            } else {
                const std::vector<DirectionMatrix> prefix(matrices.begin(),
                                                          matrices.begin() + (idx - 1));
                scores.push_back(criterionD(matrices[idx - 1], prefix, weights, options));
            }
        }
        std::sort(scores.begin(), scores.end());
        int zeros = 0;
        for (double sc : scores) {
            zeros += (sc == 0.0);
        }
        std::printf("criterion D(6) window=%d%s over %d sampled dims: 0=%d min=%.4g median=%.4g "
                    "p90=%.4g p99=%.4g max=%.4g\n",
                    window, twoSided ? " (two-sided)" : "", k, zeros, scores.front(),
                    scores[scores.size() / 2],
                    scores[static_cast<std::size_t>(0.9 * scores.size())],
                    scores[static_cast<std::size_t>(0.99 * scores.size())], scores.back());
    }

    const std::uint32_t N = 1u << mLevel;

    // ── 3. Net exactness, moments, bit balance ──
    {
        const int k = std::min<int>(sample, static_cast<int>(nNew));
        std::vector<std::uint32_t> dims;
        for (int s = 0; s < k; ++s) {
            dims.push_back(firstNew +
                           static_cast<std::uint32_t>((static_cast<std::uint64_t>(s) * nNew) / k));
        }
        if (!dims.empty()) {
            dims.front() = firstNew;
        }
        int pairsTested = 0, pairsFailed = 0, dimsFailed = 0;
        GrayGenerator ones(N, true);
        const auto xOne = ones.run();

        for (std::uint32_t d : dims) {
            GrayGenerator gen(entries[d - 2], N);
            const auto x = gen.run();

            // Exactness of the 1D net at m bits: every 2^b bin has 2^{m-b} hits.
            {
                const int b = std::min(mLevel - 1, 12);
                const std::uint32_t expected = N >> b;
                std::uint32_t worst = 0;
                const bool ok = exactGrid(x, x, b, 0, expected, &worst);
                if (!ok)
                    ++dimsFailed;
            }

            // Moments.
            double sum = 0.0, sum2 = 0.0;
            for (std::uint32_t v : x) {
                const double u = static_cast<double>(v) / 4294967296.0;
                sum += u;
                sum2 += u * u;
            }
            const double mean = sum / N;
            const double var = sum2 / N - mean * mean;
            if (std::fabs(mean - 0.5) > 5e-3 || std::fabs(var - 1.0 / 12.0) > 5e-3)
                ++dimsFailed;

            // 2D net exactness against the t-value guarantee.
            const std::uint32_t partners[] = {1, d >= 2 ? d - 1 : 1, d >= 9 ? d - 8 : 1,
                                              d >= 65 ? d - 64 : 1};
            for (std::uint32_t dj : partners) {
                if (dj == d || dj < 1 || dj > lastDim)
                    continue;
                const DirectionMatrix& other = (dj == 1) ? matrices[0] : matrices[dj - 1];
                const int t = tValue2D(other, matrices[d - 1], mLevel);
                const int budget = mLevel - t;
                if (budget < 2)
                    continue;
                std::vector<std::uint32_t> y;
                if (dj == 1) {
                    y = xOne;
                } else {
                    GrayGenerator gj(entries[dj - 2], N);
                    y = gj.run();
                }
                const int b1 = budget / 2;
                const int b2 = budget - b1;
                std::uint32_t expected = 1u << t;
                std::uint32_t worst = 0;
                ++pairsTested;
                if (!exactGrid(x, y, b1, b2, expected, &worst)) {
                    ++pairsFailed;
                    if (pairsFailed <= 5)
                        QTA_LOG_WARN("quantape.tools",
                                     "net exactness FAIL (d={}, j={}, t={}, {}x{} grid, "
                                     "expected {}, worst cell {})",
                                     d, dj, t, 1 << b1, 1 << b2, expected, worst);
                }
            }
        }
        std::printf("net exactness: %d pairs tested, %d failed -> %s\n", pairsTested, pairsFailed,
                    pairsFailed ? "FAIL" : "ok");
        std::printf("1D bins/moments: %d of %d sampled dims failed -> %s\n", dimsFailed, k,
                    dimsFailed ? "FAIL" : "ok");
        failures += (pairsFailed != 0);
        failures += (dimsFailed != 0);
    }

    // ── 4. Cross-check vs the reference generator ──
    if (!sobolCc.empty() && crossLog2 <= 10) {
        const std::uint32_t NC = 1u << crossLog2;
        const std::string out = "/tmp/verify_sobol_cc.txt";
        const std::string cmd = sobolCc + " " + std::to_string(NC) + " " + std::to_string(lastDim) +
                                " " + input + " > " + out + " 2>/dev/null";
        std::printf("running reference generator: %s\n", cmd.c_str());
        if (std::system(cmd.c_str()) != 0) {
            QTA_LOG_ERROR("quantape.tools", "reference generator failed");
            ++failures;
        } else {
            std::ifstream in(out);
            std::vector<std::vector<double>> ref(
                static_cast<std::size_t>(NC),
                std::vector<double>(static_cast<std::size_t>(lastDim), 0.0));
            for (std::uint32_t i = 0; i < NC; ++i) {
                for (std::uint32_t j = 0; j < lastDim; ++j) {
                    in >> ref[i][j];
                }
            }
            int compared = 0, mismatches = 0;
            const std::uint32_t samples[] = {1, 2, firstNew,
                                             static_cast<std::uint32_t>(lastDim / 2), lastDim};
            for (std::uint32_t dim : samples) {
                if (dim < 1 || dim > lastDim)
                    continue;
                GrayGenerator gen =
                    (dim == 1) ? GrayGenerator(NC, true) : GrayGenerator(entries[dim - 2], NC);
                const auto x = gen.run();
                for (std::uint32_t i = 0; i < NC; ++i) {
                    const double ours = static_cast<double>(x[i]) / 4294967296.0;
                    ++compared;
                    if (std::fabs(ours - ref[i][dim - 1]) > 1e-15)
                        ++mismatches;
                }
            }
            std::printf("cross-check vs sobol.cc: %d values compared, %d mismatches -> %s\n",
                        compared, mismatches, mismatches ? "FAIL" : "ok");
            failures += mismatches != 0;
        }
    } else if (!sobolCc.empty()) {
        std::printf("cross-check skipped: use --cross-points-log2<=10\n");
    }

    std::printf(failures ? "VERIFY: %d failure(s)\n" : "VERIFY: all checks passed\n", failures);
    return failures ? 1 : 0;
}
