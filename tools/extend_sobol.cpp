/**
 * @file extend_sobol.cpp
 * @brief Extend Joe-Kuo Sobol direction numbers beyond 21201 dimensions.
 *
 * Standalone tool: only the C++ standard library and the headers under
 * include/quantape/math/Random/Sobol/ are needed (no Stan, Eigen, TBB or the
 * quantape library). Build:
 *
 *   g++ -O3 -std=c++20 -pthread -I include tools/extend_sobol.cpp -o extend_sobol
 *
 * Three modes of operation:
 *
 *   --local     Run everything in-process with a thread pool. This is the
 *               supported path for criterion search: dimensions are chosen
 *               sequentially (component-by-component), candidates of each
 *               dimension are evaluated in parallel.
 *
 *   --dispatch  Generate work batches as files (RANDOM level only in worker
 *               mode; criterion search needs the chosen prefix).
 *
 *   --worker    Read work items from stdin, write results to stdout.
 *
 * Search levels:
 *   0 (random)    random valid direction numbers; seconds for any target.
 *   1 (windowed)  minimise the weighted 2D-projection criterion D(q) against
 *                 the last --window dimensions (default 128). Recommended.
 *   2 (full)      criterion against every previous dimension (Joe-Kuo style;
 *                 cost grows linearly with dimension).
 *
 * Examples:
 *   # Random valid direction numbers: 300k dimensions in seconds
 *   ./extend_sobol --local --target=300000 --level=0 \
 *       --input=new-joe-kuo-6.21201 --output=joe-kuo-300k.txt
 *
 *   # Weighted 2D search with a 128-dimension window
 *   ./extend_sobol --local --target=150000 --level=1 --threads=16 --window=128 \
 *       --input=new-joe-kuo-6.21201 --output=joe-kuo-150k.txt
 */
#include "quantape/log/Log.h"
#include "quantape/math/Random/Sobol/CBCSearch.h"
#include "quantape/math/Random/Sobol/DirectionNumbers.h"
#include "quantape/math/Random/Sobol/GF2.h"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <iterator>
#include <sstream>
#include <string>
#include <unordered_set>
#include <vector>

using namespace quantape::math::mc;

namespace {

struct Options {
    int target = 100000;
    sobol::SearchLevel level = sobol::SearchLevel::WINDOWED;
    int threads = 0;
    int window = 128;
    int candidates = 0; ///< <= 0: max(64, 2000000 / dim) as in Joe-Kuo
    int mMin = 1;
    int mMax = 31;
    double exponent = 6.0;
    double weightBase = 0.9999;
};

struct PolyInfo {
    uint64_t poly;
    int degree;
};

} // namespace

// ═══════════════════════════════════════════════════════════════════════════
// Mode: --local
// ═══════════════════════════════════════════════════════════════════════════

void run_local(const std::string& input_file, const std::string& output_file,
               const Options& options) {
    // 1. Load existing Joe-Kuo file
    QTA_LOG_INFO("quantape.tools", "loading existing direction numbers from: {}", input_file);
    auto existing = sobol::load_joe_kuo(input_file);
    int start_dim = (int)existing.size() + 2; // +2 because dim 1 has no entry
    QTA_LOG_INFO("quantape.tools", "loaded {} entries (dims 2-{})", existing.size(), start_dim - 1);

    if (start_dim > options.target) {
        QTA_LOG_INFO("quantape.tools", "already have {} dims, target is {}; nothing to do",
                     start_dim - 1, options.target);
        return;
    }

    // 2. Find the starting polynomial degree and the already used polynomials
    int max_existing_degree = 0;
    std::vector<uint64_t> used_polys;
    used_polys.reserve(existing.size());
    for (auto& e : existing) {
        if ((int)e.s > max_existing_degree)
            max_existing_degree = e.s;
        used_polys.push_back(gf2::decode_poly((int)e.s, e.a));
    }
    const std::unordered_set<uint64_t> used(used_polys.begin(), used_polys.end());

    QTA_LOG_INFO("quantape.tools", "enumerating new primitive polynomials...");
    std::vector<PolyInfo> new_polys;
    const int needed = options.target - start_dim + 1;
    for (int deg = max_existing_degree; (int)new_polys.size() < needed; ++deg) {
        auto t0 = std::chrono::steady_clock::now();
        auto candidates = gf2::enumerate_primitive(deg);
        auto t1 = std::chrono::steady_clock::now();
        const double secs = std::chrono::duration<double>(t1 - t0).count();

        int added = 0;
        for (uint64_t p : candidates) {
            if ((int)new_polys.size() >= needed)
                break;
            if (used.count(p) == 0) {
                new_polys.push_back({p, deg});
                ++added;
            }
        }
        QTA_LOG_INFO("quantape.tools", "degree {}: {} candidates, added {} ({:.1f}s)", deg,
                     candidates.size(), added, secs);
    }

    // 3. Build work items
    QTA_LOG_INFO("quantape.tools", "building {} work items (level={})", new_polys.size(),
                 (int)options.level);

    std::vector<sobol::WorkItem> work(new_polys.size());
    for (size_t i = 0; i < new_polys.size(); ++i) {
        work[i].dim = start_dim + (uint32_t)i;
        work[i].polynomial = new_polys[i].poly;
        work[i].degree = new_polys[i].degree;
        work[i].level = options.level;
        work[i].seed = 0xcafe0000ULL + i; // reproducible seed per dim
    }

    // 4. Search context: dimensions 1..start_dim-1 (index i = dimension i+1)
    sobol::SearchContext context;
    context.criterion.window = (options.level == sobol::SearchLevel::FULL) ? 0 : options.window;
    context.criterion.mMin = options.mMin;
    context.criterion.mMax = options.mMax;
    context.criterion.exponent = options.exponent;
    context.criterion.weightBase = options.weightBase;
    context.candidates = options.candidates;
    if (options.level != sobol::SearchLevel::RANDOM) {
        context.previous.reserve(existing.size() + 1);
        context.previous.push_back(sobol::identityMatrix());
        for (auto& e : existing)
            context.previous.push_back(sobol::directionMatrix(e));
        if (start_dim <= 1111) {
            context.enforcePropertyA = true;
            context.propertyA = sobol::PropertyAChecker(1111);
            for (auto& e : existing)
                context.propertyA.add(e);
        }
    }

    // 5. Prepare output and run search (CBC results are checkpointed to the
    // output file every 100 dimensions, so an interrupted run can resume with
    // the partial file as --input).
    const std::string out = output_file.empty() ? "joe-kuo-extended.txt" : output_file;
    if (!input_file.empty() && input_file != out) {
        std::filesystem::copy_file(input_file, out,
                                   std::filesystem::copy_options::overwrite_existing);
    }
    QTA_LOG_INFO("quantape.tools", "searching for direction numbers (threads={}, level={})",
                 options.threads, (int)options.level);
    if (options.level != sobol::SearchLevel::RANDOM) {
        QTA_LOG_INFO("quantape.tools",
                     "criterion D({:.0f}): window={} m=[{},{}] weight={:.4g} candidates={}",
                     options.exponent, options.window, options.mMin, options.mMax,
                     options.weightBase, options.candidates > 0 ? "fixed" : "auto");
    }
    const auto t0 = std::chrono::steady_clock::now();
    size_t flushed = 0;
    std::vector<sobol::Entry> pending;
    const auto onResult = [&](std::size_t, const sobol::SearchResult& r) {
        sobol::Entry e{r.dim, r.degree, r.a_encoded, r.m};
        pending.push_back(std::move(e));
        if (pending.size() >= 100) {
            sobol::save_joe_kuo(out, pending, true);
            flushed += pending.size();
            pending.clear();
        }
    };

    auto results = sobol::process_batch(
        work, context, options.threads,
        [&](uint32_t done, uint32_t total) {
            if (done % 100 != 0 && done != total)
                return;
            const auto now = std::chrono::steady_clock::now();
            const double elapsed = std::chrono::duration<double>(now - t0).count();
            const double rate = done / std::max(elapsed, 1e-9);
            const double eta = (total - done) / std::max(rate, 1e-9);
            QTA_LOG_INFO("quantape.tools", "{}/{} ({:.1f}%) {:.1f} dims/s ETA {:.0f}s", done, total,
                         100.0 * done / total, rate, eta);
        },
        options.level != sobol::SearchLevel::RANDOM ? onResult : sobol::ResultCallback{});
    if (!pending.empty()) {
        sobol::save_joe_kuo(out, pending, true);
        flushed += pending.size();
        pending.clear();
    }

    const auto t1 = std::chrono::steady_clock::now();
    const double secs = std::chrono::duration<double>(t1 - t0).count();
    QTA_LOG_INFO("quantape.tools", "done in {:.1f}s ({:.1f} dims/sec)", secs,
                 results.size() / secs);

    // 6. Quality summary
    if (options.level != sobol::SearchLevel::RANDOM) {
        double minScore = sobol::kNoScore;
        double maxScore = 0.0;
        double sum = 0.0;
        int scored = 0;
        int saturated = 0;
        for (auto& r : results) {
            if (r.score >= sobol::kNoScore) {
                ++saturated;
                continue;
            }
            minScore = std::min(minScore, r.score);
            maxScore = std::max(maxScore, r.score);
            sum += r.score;
            ++scored;
        }
        QTA_LOG_INFO("quantape.tools", "quality D({:.0f}) score: min={:.4g} mean={:.4g} max={:.4g}",
                     options.exponent, minScore, scored > 0 ? sum / scored : 0.0, maxScore);
        if (saturated > 0)
            QTA_LOG_WARN("quantape.tools", "{} dimensions found no Property-A candidate",
                         saturated);
    }

    // 7. Write output (RANDOM level only; CBC entries are already checkpointed)
    if (options.level == sobol::SearchLevel::RANDOM) {
        std::vector<sobol::Entry> new_entries(results.size());
        for (size_t i = 0; i < results.size(); ++i) {
            new_entries[i].dim = results[i].dim;
            new_entries[i].s = results[i].degree;
            new_entries[i].a = results[i].a_encoded;
            new_entries[i].m = results[i].m;
        }
        QTA_LOG_INFO("quantape.tools", "writing {} new entries to: {}", new_entries.size(), out);
        sobol::save_joe_kuo(out, new_entries, true);
    } else {
        QTA_LOG_INFO("quantape.tools", "checkpointed {} new entries to: {}", flushed, out);
    }

    QTA_LOG_INFO("quantape.tools", "total dimensions: {}",
                 (int)(existing.size() + 1 + results.size()));
}

// ═══════════════════════════════════════════════════════════════════════════
// Mode: --dispatch  (generate work batches for container execution)
// ═══════════════════════════════════════════════════════════════════════════

void run_dispatch(const std::string& input_file, const std::string& outdir, const Options& options,
                  int n_batches) {
    auto existing = sobol::load_joe_kuo(input_file);
    int start_dim = (int)existing.size() + 2;

    int max_deg = 0;
    std::vector<uint64_t> used;
    for (auto& e : existing) {
        if ((int)e.s > max_deg)
            max_deg = e.s;
        used.push_back(gf2::decode_poly((int)e.s, e.a));
    }
    const std::unordered_set<uint64_t> used_set(used.begin(), used.end());

    std::vector<PolyInfo> polys;
    const int needed = options.target - start_dim + 1;
    for (int deg = max_deg; (int)polys.size() < needed; ++deg) {
        for (uint64_t p : gf2::enumerate_primitive(deg)) {
            if ((int)polys.size() >= needed)
                break;
            if (used_set.count(p) == 0)
                polys.push_back({p, deg});
        }
    }

    std::filesystem::create_directories(outdir);

    const int per_batch = ((int)polys.size() + n_batches - 1) / n_batches;
    for (int b = 0; b < n_batches; ++b) {
        const int lo = b * per_batch;
        const int hi = std::min(lo + per_batch, (int)polys.size());
        if (lo >= hi)
            break;

        std::vector<sobol::WorkItem> batch;
        for (int i = lo; i < hi; ++i) {
            sobol::WorkItem wi{};
            wi.dim = start_dim + i;
            wi.polynomial = polys[i].poly;
            wi.degree = polys[i].degree;
            wi.level = sobol::SearchLevel::RANDOM;
            wi.seed = 0xcafe0000ULL + i;
            batch.push_back(wi);
        }

        char fname[256];
        snprintf(fname, sizeof(fname), "%s/batch_%04d.txt", outdir.c_str(), b);
        std::ofstream out(fname);
        out << sobol::serialize_work(batch);
        QTA_LOG_INFO("quantape.tools", "wrote {} ({} items, dims {}-{})", fname, (int)batch.size(),
                     batch.front().dim, batch.back().dim);
    }

    QTA_LOG_INFO("quantape.tools", "to process each batch (RANDOM level) in a container:");
    QTA_LOG_INFO("quantape.tools", "  docker run sobol-worker < batch_XXXX.txt > results_XXXX.txt");
}

// ═══════════════════════════════════════════════════════════════════════════
// Mode: --worker  (read work from stdin, write results to stdout)
// ═══════════════════════════════════════════════════════════════════════════

void run_worker(int num_threads) {
    std::string input((std::istreambuf_iterator<char>(std::cin)), std::istreambuf_iterator<char>());

    auto items = sobol::deserialize_work(input);
    QTA_LOG_INFO("quantape.tools", "worker: received {} items", items.size());

    sobol::SearchContext context; // empty: worker supports RANDOM level
    auto results =
        sobol::process_batch(items, context, num_threads, [](uint32_t done, uint32_t total) {
            QTA_LOG_DEBUG("quantape.tools", "worker progress: {}/{}", done, total);
        });

    for (auto& r : results) {
        std::cout << r.dim << "\t" << r.degree << "\t" << r.a_encoded;
        for (auto mi : r.m)
            std::cout << "\t" << mi;
        std::cout << "\n";
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// CLI
// ═══════════════════════════════════════════════════════════════════════════

void usage(const char* prog) {
    fprintf(stderr,
            "Usage:\n"
            "  %s --local    [options]   Run search locally with a thread pool\n"
            "  %s --dispatch [options]   Generate work batches for containers\n"
            "  %s --worker   [options]   Process work from stdin (container mode)\n"
            "\n"
            "Options:\n"
            "  --input=FILE      Joe-Kuo input file (21201 dims)\n"
            "  --output=FILE     Output file (--local mode)\n"
            "  --outdir=DIR      Output directory for batches (--dispatch mode)\n"
            "  --target=N        Target number of dimensions (default: 100000)\n"
            "  --level=L         0=random, 1=windowed D(6) search, 2=full D(6) search\n"
            "  --threads=N       Thread count (default: hardware concurrency)\n"
            "  --window=N        Previous dimensions in the level-1 criterion (default: 128)\n"
            "  --candidates=N    Candidate sets per dimension (default: max(64, 2e6/dim))\n"
            "  --mmin=N          First m in the criterion range (default: 1)\n"
            "  --mmax=N          Last m in the criterion range (default: 31)\n"
            "  --exponent=X      q in the criterion D(q) (default: 6)\n"
            "  --weight=X        Weight base 0.9999^{j-1} (default: 0.9999)\n"
            "  --batches=N       Number of batches (--dispatch mode, default: 32)\n",
            prog, prog, prog);
}

int main(int argc, char** argv) {
    enum Mode { NONE, LOCAL, DISPATCH, WORKER } mode = NONE;
    std::string input_file, output_file, outdir = "./sobol_work";
    Options options;
    int batches = 32;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--local")
            mode = LOCAL;
        else if (arg == "--dispatch")
            mode = DISPATCH;
        else if (arg == "--worker")
            mode = WORKER;
        else if (arg == "--help" || arg == "-h") {
            usage(argv[0]);
            return 0;
        } else if (arg.rfind("--input=", 0) == 0)
            input_file = arg.substr(8);
        else if (arg.rfind("--output=", 0) == 0)
            output_file = arg.substr(9);
        else if (arg.rfind("--outdir=", 0) == 0)
            outdir = arg.substr(9);
        else if (arg.rfind("--target=", 0) == 0)
            options.target = std::stoi(arg.substr(9));
        else if (arg.rfind("--level=", 0) == 0)
            options.level = static_cast<sobol::SearchLevel>(std::stoi(arg.substr(8)));
        else if (arg.rfind("--threads=", 0) == 0)
            options.threads = std::stoi(arg.substr(10));
        else if (arg.rfind("--window=", 0) == 0)
            options.window = std::stoi(arg.substr(9));
        else if (arg.rfind("--candidates=", 0) == 0)
            options.candidates = std::stoi(arg.substr(13));
        else if (arg.rfind("--mmin=", 0) == 0)
            options.mMin = std::stoi(arg.substr(7));
        else if (arg.rfind("--mmax=", 0) == 0)
            options.mMax = std::stoi(arg.substr(7));
        else if (arg.rfind("--exponent=", 0) == 0)
            options.exponent = std::stod(arg.substr(11));
        else if (arg.rfind("--weight=", 0) == 0)
            options.weightBase = std::stod(arg.substr(9));
        else if (arg.rfind("--batches=", 0) == 0)
            batches = std::stoi(arg.substr(10));
        else {
            fprintf(stderr, "Unknown option: %s\n", arg.c_str());
            return 1;
        }
    }

    switch (mode) {
        case LOCAL:
            if (input_file.empty()) {
                fprintf(stderr, "Error: --input required\n");
                return 1;
            }
            run_local(input_file, output_file, options);
            break;
        case DISPATCH:
            if (input_file.empty()) {
                fprintf(stderr, "Error: --input required\n");
                return 1;
            }
            run_dispatch(input_file, outdir, options, batches);
            break;
        case WORKER:
            run_worker(options.threads);
            break;
        default:
            usage(argv[0]);
            return 1;
    }

    return 0;
}
