/**
 * @file extend_sobol.cpp
 * @brief Extend Joe-Kuo Sobol direction numbers from 21,201 to 100,000+ dimensions.
 *
 * Three modes of operation:
 *
 *   --local     Run everything in-process with thread pool.
 *               Good for extending to ~50K with CBC, or 100K+ with RANDOM.
 *
 *   --dispatch  Generate work batches as files for container execution.
 *               Each batch is a text file consumed by --worker mode.
 *
 *   --worker    Read work items from stdin, write results to stdout.
 *               Run inside Docker containers for distributed CBC search.
 *
 * Examples:
 *   # Extend to 100K locally, CBC search, 16 threads
 *   ./extend_sobol --local --target=100000 --level=1 --threads=16 \
 *       --input=new-joe-kuo-6.21201 --output=joe-kuo-100k.txt
 *
 *   # Generate work batches for 32 containers
 *   ./extend_sobol --dispatch --target=100000 --level=1 --batches=32 \
 *       --input=new-joe-kuo-6.21201 --outdir=./work/
 *
 *   # Worker mode (inside container)
 *   ./extend_sobol --worker < batch_007.txt > results_007.txt
 *
 *   # Random mode — instant, good enough for high dimensions
 *   ./extend_sobol --local --target=100000 --level=0 --input=new-joe-kuo-6.21201
 */

#include <chrono>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include "../Math/Sobol/CBCSearch.h"
#include "../Math/Sobol/DirectionNumbers.h"
#include "../Math/Sobol/GF2.h"

using namespace mc;

// ═══════════════════════════════════════════════════════════════════════════
// Polynomial collection: gather enough primitive polys to reach target dims
// ═══════════════════════════════════════════════════════════════════════════

struct PolyInfo {
    uint64_t poly;
    int degree;
};

std::vector<PolyInfo> collect_polynomials(int start_dim, int target_dim, int start_degree) {
    std::vector<PolyInfo> polys;
    int needed = target_dim - start_dim;
    int deg = start_degree;

    fprintf(stderr, "  Collecting %d primitive polynomials (degrees %d+)...\n", needed, deg);

    while ((int)polys.size() < needed) {
        auto t0 = std::chrono::steady_clock::now();
        uint64_t expected = gf2::count_primitive(deg);
        fprintf(stderr, "    degree %d: expecting ~%llu polys... ", deg,
                (unsigned long long)expected);

        auto found = gf2::enumerate_primitive(deg);

        auto t1 = std::chrono::steady_clock::now();
        double secs = std::chrono::duration<double>(t1 - t0).count();
        fprintf(stderr, "found %zu in %.1fs\n", found.size(), secs);

        for (auto p : found) {
            if ((int)polys.size() >= needed)
                break;
            polys.push_back({p, deg});
        }
        ++deg;
    }

    fprintf(stderr, "  Collected %zu polynomials (degrees %d-%d)\n", polys.size(), start_degree,
            deg - 1);
    return polys;
}

// ═══════════════════════════════════════════════════════════════════════════
// Mode: --local
// ═══════════════════════════════════════════════════════════════════════════

void run_local(const std::string& input_file, const std::string& output_file, int target_dim,
               sobol::SearchLevel level, int num_threads, int n_proj_check) {
    // 1. Load existing Joe-Kuo file
    fprintf(stderr, "Loading existing direction numbers from: %s\n", input_file.c_str());
    auto existing = sobol::load_joe_kuo(input_file);
    int start_dim = (int)existing.size() + 2; // +2 because dim 1 has no entry
    fprintf(stderr, "  Loaded %zu entries (dims 2-%d)\n", existing.size(), start_dim - 1);

    if (start_dim > target_dim) {
        fprintf(stderr, "  Already have %d dims, target is %d. Nothing to do.\n", start_dim - 1,
                target_dim);
        return;
    }

    // 2. Precompute direction numbers for existing dims (for 2D projection quality)
    std::vector<std::vector<uint64_t>> existing_v;
    if (level == sobol::SearchLevel::PROJ_2D) {
        fprintf(stderr, "  Precomputing existing direction numbers for 2D checks...\n");
        existing_v.reserve(existing.size());
        for (auto& e : existing)
            existing_v.push_back(sobol::entry_to_v(e));
    }

    // 3. Find the starting polynomial degree
    int max_existing_degree = 0;
    for (auto& e : existing)
        if ((int)e.s > max_existing_degree)
            max_existing_degree = e.s;
    // New polynomials start at the next degree after the last one used
    // (or same degree if not all polys of that degree are used)

    // Actually: we need to find polys NOT already used. The simplest approach:
    // enumerate from degree max_existing_degree upward, skip polys already in the file.
    // For Joe-Kuo, dims are ordered by degree, so all polys of degree ≤ max_existing_degree
    // through the last entry's polynomial are used. We can just start enumerating from
    // the next polynomial of that degree.
    //
    // Simpler: enumerate all primitive polys of each degree from max_existing_degree up,
    // skip those already in the existing set.

    // Build set of used polynomials
    std::vector<uint64_t> used_polys;
    for (auto& e : existing)
        used_polys.push_back(gf2::decode_poly(e.s, e.a));

    fprintf(stderr, "\nEnumerating new primitive polynomials...\n");

    std::vector<PolyInfo> new_polys;
    int needed = target_dim - start_dim + 1;
    for (int deg = max_existing_degree; (int)new_polys.size() < needed; ++deg) {
        auto t0 = std::chrono::steady_clock::now();
        auto candidates = gf2::enumerate_primitive(deg);
        auto t1 = std::chrono::steady_clock::now();
        double secs = std::chrono::duration<double>(t1 - t0).count();

        int added = 0;
        for (auto p : candidates) {
            if ((int)new_polys.size() >= needed)
                break;
            // Skip if already used
            bool used = false;
            for (auto u : used_polys) {
                if (u == p) {
                    used = true;
                    break;
                }
            }
            if (!used) {
                new_polys.push_back({p, deg});
                ++added;
            }
        }
        fprintf(stderr, "  degree %d: %zu candidates, added %d (%.1fs)\n", deg, candidates.size(),
                added, secs);
    }

    // 4. Build work items
    fprintf(stderr, "\nBuilding %d work items (level=%d)...\n", (int)new_polys.size(), (int)level);

    std::vector<sobol::WorkItem> work(new_polys.size());
    for (size_t i = 0; i < new_polys.size(); ++i) {
        work[i].dim = start_dim + (uint32_t)i;
        work[i].polynomial = new_polys[i].poly;
        work[i].degree = new_polys[i].degree;
        work[i].level = level;
        work[i].seed = 0xcafe0000ULL + i; // reproducible seed per dim
    }

    // 5. Run search
    fprintf(stderr, "\nSearching for direction numbers (%d threads)...\n", num_threads);
    auto t0 = std::chrono::steady_clock::now();

    auto results = sobol::process_batch(
        work, existing_v, n_proj_check, num_threads, [](uint32_t done, uint32_t total) {
            fprintf(stderr, "\r  %u / %u (%.1f%%)", done, total, 100.0 * done / total);
        });

    auto t1 = std::chrono::steady_clock::now();
    double secs = std::chrono::duration<double>(t1 - t0).count();
    fprintf(stderr, "\n  Done in %.1fs (%.1f dims/sec)\n", secs, results.size() / secs);

    // 6. Quality summary
    int t0_count = 0, max_t = 0;
    for (auto& r : results) {
        if (r.t_val == 0)
            ++t0_count;
        if (r.t_val > max_t)
            max_t = r.t_val;
    }
    fprintf(stderr, "\n  Quality: %d/%zu with t=0 (Property A), max t=%d\n", t0_count,
            results.size(), max_t);

    // 7. Write output
    std::vector<sobol::Entry> new_entries(results.size());
    for (size_t i = 0; i < results.size(); ++i) {
        new_entries[i].dim = results[i].dim;
        new_entries[i].s = results[i].degree;
        new_entries[i].a = results[i].a_encoded;
        new_entries[i].m = results[i].m;
    }

    std::string out = output_file.empty() ? "joe-kuo-extended.txt" : output_file;
    fprintf(stderr, "\n  Writing %zu new entries to: %s\n", new_entries.size(), out.c_str());

    // Copy existing file and append new entries
    if (!input_file.empty() && input_file != out) {
        std::filesystem::copy_file(input_file, out,
                                   std::filesystem::copy_options::overwrite_existing);
    }
    sobol::save_joe_kuo(out, new_entries, true);

    fprintf(stderr, "  Total dimensions: %d\n\n", (int)(existing.size() + 1 + new_entries.size()));
}

// ═══════════════════════════════════════════════════════════════════════════
// Mode: --dispatch  (generate work batches for container execution)
// ═══════════════════════════════════════════════════════════════════════════

void run_dispatch(const std::string& input_file, const std::string& outdir, int target_dim,
                  sobol::SearchLevel level, int n_batches) {
    auto existing = sobol::load_joe_kuo(input_file);
    int start_dim = (int)existing.size() + 2;

    int max_deg = 0;
    std::vector<uint64_t> used;
    for (auto& e : existing) {
        if ((int)e.s > max_deg)
            max_deg = e.s;
        used.push_back(gf2::decode_poly(e.s, e.a));
    }

    // Collect polynomials
    std::vector<PolyInfo> polys;
    int needed = target_dim - start_dim + 1;
    for (int deg = max_deg; (int)polys.size() < needed; ++deg) {
        for (auto p : gf2::enumerate_primitive(deg)) {
            if ((int)polys.size() >= needed)
                break;
            bool skip = false;
            for (auto u : used)
                if (u == p) {
                    skip = true;
                    break;
                }
            if (!skip)
                polys.push_back({p, deg});
        }
    }

    // Create output directory
    std::filesystem::create_directories(outdir);

    // Split into batches
    int per_batch = ((int)polys.size() + n_batches - 1) / n_batches;
    for (int b = 0; b < n_batches; ++b) {
        int lo = b * per_batch;
        int hi = std::min(lo + per_batch, (int)polys.size());
        if (lo >= hi)
            break;

        std::vector<sobol::WorkItem> batch;
        for (int i = lo; i < hi; ++i) {
            sobol::WorkItem wi{};
            wi.dim = start_dim + i;
            wi.polynomial = polys[i].poly;
            wi.degree = polys[i].degree;
            wi.level = level;
            wi.seed = 0xcafe0000ULL + i;
            batch.push_back(wi);
        }

        char fname[256];
        snprintf(fname, sizeof(fname), "%s/batch_%04d.txt", outdir.c_str(), b);
        std::ofstream out(fname);
        out << sobol::serialize_work(batch);
        fprintf(stderr, "  Wrote %s (%d items, dims %u-%u)\n", fname, (int)batch.size(),
                batch.front().dim, batch.back().dim);
    }

    fprintf(stderr, "\nTo process each batch in a container:\n");
    fprintf(stderr, "  docker run sobol-worker < batch_XXXX.txt > results_XXXX.txt\n\n");
}

// ═══════════════════════════════════════════════════════════════════════════
// Mode: --worker  (read work from stdin, write results to stdout)
// ═══════════════════════════════════════════════════════════════════════════

void run_worker(int num_threads) {
    // Read all of stdin
    std::string input((std::istreambuf_iterator<char>(std::cin)), std::istreambuf_iterator<char>());

    auto items = sobol::deserialize_work(input);
    fprintf(stderr, "  Worker: received %zu items\n", items.size());

    auto results =
        sobol::process_batch(items, {}, 0, num_threads, [](uint32_t done, uint32_t total) {
            fprintf(stderr, "\r  Worker: %u / %u", done, total);
        });
    fprintf(stderr, "\n");

    // Write results as Joe-Kuo format to stdout
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
            "  %s --local    [options]   Run search locally with thread pool\n"
            "  %s --dispatch [options]   Generate work batches for containers\n"
            "  %s --worker   [options]   Process work from stdin (container mode)\n"
            "\n"
            "Options:\n"
            "  --input=FILE     Joe-Kuo input file (21201 dims)\n"
            "  --output=FILE    Output file (--local mode)\n"
            "  --outdir=DIR     Output directory for batches (--dispatch mode)\n"
            "  --target=N       Target number of dimensions (default: 100000)\n"
            "  --level=L        Search level: 0=random, 1=property_a, 2=proj_2d (default: 1)\n"
            "  --threads=N      Thread count (default: hardware concurrency)\n"
            "  --batches=N      Number of batches (--dispatch mode, default: 32)\n"
            "  --proj-check=N   Existing dims to check for 2D quality (default: 100)\n"
            "\n",
            prog, prog, prog);
}

int main(int argc, char** argv) {
    enum Mode { NONE, LOCAL, DISPATCH, WORKER } mode = NONE;
    std::string input_file, output_file, outdir = "./sobol_work";
    int target = 100000, level = 1, threads = 0, batches = 32, proj_check = 100;

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
            target = std::stoi(arg.substr(9));
        else if (arg.rfind("--level=", 0) == 0)
            level = std::stoi(arg.substr(8));
        else if (arg.rfind("--threads=", 0) == 0)
            threads = std::stoi(arg.substr(10));
        else if (arg.rfind("--batches=", 0) == 0)
            batches = std::stoi(arg.substr(10));
        else if (arg.rfind("--proj-check=", 0) == 0)
            proj_check = std::stoi(arg.substr(13));
        else {
            fprintf(stderr, "Unknown option: %s\n", arg.c_str());
            return 1;
        }
    }

    auto slevel = static_cast<sobol::SearchLevel>(level);

    switch (mode) {
        case LOCAL:
            if (input_file.empty()) {
                fprintf(stderr, "Error: --input required\n");
                return 1;
            }
            run_local(input_file, output_file, target, slevel, threads, proj_check);
            break;
        case DISPATCH:
            if (input_file.empty()) {
                fprintf(stderr, "Error: --input required\n");
                return 1;
            }
            run_dispatch(input_file, outdir, target, slevel, batches);
            break;
        case WORKER:
            run_worker(threads);
            break;
        default:
            usage(argv[0]);
            return 1;
    }

    return 0;
}
