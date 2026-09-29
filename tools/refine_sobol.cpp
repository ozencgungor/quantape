/**
 * @file refine_sobol.cpp
 * @brief Iterative refinement (coordinate descent) of existing Sobol direction
 *        numbers, using the Joe-Kuo D(q) criterion scored against both
 *        neighbouring sides of each dimension.
 *
 * Generation picks each dimension against its earlier prefix only. Refinement
 * revisits a dimension with all others fixed and scores it against a window of
 * dimensions on BOTH sides, so pairs with later neighbours (never seen during
 * generation) can improve too. Accepted changes are monotone per dimension
 * (strictly lower windowed score); epochs repeat until no improvement.
 *
 * Standalone (standard library + the Sobol headers). By design it never
 * modifies the input file: the refined table is written to --output only,
 * unless --dry-run is given (report only).
 *
 * Usage:
 *   refine_sobol --input table.txt [--output refined.txt] [--dry-run]
 *       [--refine-from=21202] [--window=64] [--candidates=32] [--epochs=2]
 *       [--fraction=1.0] [--min-score=0] [--worst-first]
 *       [--mmin=1] [--mmax=31] [--q=6] [--weight=0.9999]
 *       [--threads=0] [--seed=...] [--changelog=changes.csv]
 *
 * Prints score statistics per epoch and ends with "REFINE COMPLETE".
 */
#include "quantape/log/Log.h"
#include "quantape/math/Random/Sobol/CBCSearch.h"
#include "quantape/math/Random/Sobol/DirectionNumbers.h"
#include "quantape/math/Random/Sobol/GF2.h"
#include "quantape/math/Random/Sobol/SobolQuality.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <string>
#include <utility>
#include <vector>

using namespace quantape::math::mc;
using namespace quantape::math::mc::sobol;

namespace {

struct Options {
    std::string input;
    std::string output;
    std::string changelog;
    bool dryRun = false;
    std::uint32_t refineFrom = 21202;
    int window = 64;
    int candidates = 32;
    int epochs = 2;
    double fraction = 1.0;
    double minScore = 0.0;
    bool worstFirst = true;
    int mMin = 1;
    int mMax = 31;
    double exponent = 6.0;
    double distanceWeight = 0.9999;
    int threads = 0;
    bool globalAccept = true;
    bool reportOnly = false;
    std::uint64_t seed = 0x51F3D9C4B2A7E115ULL;
};

struct RefineStats {
    std::size_t considered = 0;
    std::size_t improved = 0;
    double beforeMean = 0.0;
    double afterMean = 0.0;
    double beforeMax = 0.0;
    double afterMax = 0.0;
};

/// Distance weights pow(base, |i - idx|), index = distance.
std::vector<double> distanceWeights(const Options& o) {
    std::vector<double> w(static_cast<std::size_t>(o.window) + 1, 1.0);
    for (std::size_t d = 1; d < w.size(); ++d) {
        w[d] = w[d - 1] * o.distanceWeight;
    }
    return w;
}

/// Two-sided windowed D(q) score of a candidate, with early abort.
double twoSidedScore(const DirectionMatrix& candidate, const std::vector<DirectionMatrix>& all,
                     std::size_t index, const Options& o, const std::vector<double>& dw,
                     double abortAbove) {
    const int n = static_cast<int>(all.size());
    const int centre = static_cast<int>(index);
    const int lo = std::max(0, centre - o.window);
    const int hi = std::min(n - 1, centre + o.window);
    double best = 0.0;
    for (int m = o.mMin; m <= o.mMax; ++m) {
        double worst = 0.0;
        for (int i = lo; i <= hi; ++i) {
            if (i == centre) {
                continue;
            }
            const double w = dw[static_cast<std::size_t>(std::abs(i - centre))];
            const double bound =
                std::min(static_cast<double>(m),
                         static_cast<double>(all[static_cast<std::size_t>(i)].degree +
                                             candidate.degree - 2)) *
                w;
            if (bound <= worst) {
                continue;
            }
            const int t = tValue2D(all[static_cast<std::size_t>(i)], candidate, m);
            if (t == 0) {
                continue;
            }
            const double weighted = static_cast<double>(t) * w;
            if (weighted > worst) {
                worst = weighted;
                if (std::pow(worst, o.exponent) / (static_cast<double>(m) - worst + 1.0) >
                    abortAbove) {
                    return kNoScore;
                }
            }
        }
        const double value = std::pow(worst, o.exponent) / (static_cast<double>(m) - worst + 1.0);
        if (value > best) {
            best = value;
            if (best > abortAbove) {
                return kNoScore;
            }
        }
    }
    return best;
}

bool writeTable(const std::string& path, const std::vector<Entry>& entries) {
    const std::string tmp = path + ".tmp";
    try {
        sobol::save_joe_kuo(tmp, entries, false);
        std::filesystem::rename(tmp, path);
    } catch (...) {
        return false;
    }
    return true;
}

} // namespace

int main(int argc, char** argv) {
    std::setvbuf(stdout, nullptr, _IOLBF, 0); // stream progress into run.log
    Options o;
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        const auto value = [&]() -> std::string {
            const auto pos = arg.find('=');
            return pos == std::string::npos ? std::string() : arg.substr(pos + 1);
        };
        if (arg.rfind("--input=", 0) == 0)
            o.input = value();
        else if (arg.rfind("--output=", 0) == 0)
            o.output = value();
        else if (arg.rfind("--changelog=", 0) == 0)
            o.changelog = value();
        else if (arg == "--dry-run")
            o.dryRun = true;
        else if (arg == "--worst-first")
            o.worstFirst = true;
        else if (arg == "--sequential")
            o.worstFirst = false;
        else if (arg == "--no-global-accept")
            o.globalAccept = false;
        else if (arg == "--report-only")
            o.reportOnly = true;
        else if (arg.rfind("--refine-from=", 0) == 0)
            o.refineFrom = static_cast<std::uint32_t>(std::stoul(value()));
        else if (arg.rfind("--window=", 0) == 0)
            o.window = std::stoi(value());
        else if (arg.rfind("--candidates=", 0) == 0)
            o.candidates = std::stoi(value());
        else if (arg.rfind("--epochs=", 0) == 0)
            o.epochs = std::stoi(value());
        else if (arg.rfind("--fraction=", 0) == 0)
            o.fraction = std::stod(value());
        else if (arg.rfind("--min-score=", 0) == 0)
            o.minScore = std::stod(value());
        else if (arg.rfind("--mmin=", 0) == 0)
            o.mMin = std::stoi(value());
        else if (arg.rfind("--mmax=", 0) == 0)
            o.mMax = std::stoi(value());
        else if (arg.rfind("--q=", 0) == 0)
            o.exponent = std::stod(value());
        else if (arg.rfind("--weight=", 0) == 0)
            o.distanceWeight = std::stod(value());
        else if (arg.rfind("--threads=", 0) == 0)
            o.threads = std::stoi(value());
        else if (arg.rfind("--seed=", 0) == 0)
            o.seed = static_cast<std::uint64_t>(std::stoull(value()));
        else if (arg == "--help" || arg == "-h") {
            std::printf(
                "usage: refine_sobol --input=FILE [--output=FILE] [--dry-run] [--refine-from=N]\n"
                "                    [--window=N] [--candidates=N] [--epochs=N] [--fraction=X]\n"
                "                    [--min-score=X] [--worst-first|--sequential] [--mmin=N]\n"
                "                    [--mmax=N] [--q=X] [--weight=X] [--threads=N] [--seed=N]\n"
                "                    [--changelog=FILE] [--report-only]\n");
            return 0;
        } else {
            QTA_LOG_ERROR("quantape.tools", "unknown option: {}", arg);
            return 1;
        }
    }
    if (o.input.empty()) {
        QTA_LOG_ERROR("quantape.tools", "--input required");
        return 1;
    }
    if (o.input == o.output && !o.dryRun) {
        QTA_LOG_ERROR("quantape.tools",
                      "refusing to overwrite the input (use --dry-run or a different --output)");
        return 1;
    }
    if (o.threads <= 0) {
        o.threads = static_cast<int>(std::max(1u, std::thread::hardware_concurrency()));
    }

    auto entries = load_joe_kuo(o.input);
    if (entries.empty()) {
        QTA_LOG_ERROR("quantape.tools", "could not load {}", o.input);
        return 1;
    }
    std::printf("loaded %zu entries (dims 2..%u)\n", entries.size(), entries.back().dim);

    std::vector<DirectionMatrix> matrices(entries.size() + 2);
    matrices[0] = identityMatrix();
    for (std::size_t i = 0; i < entries.size(); ++i) {
        matrices[i + 1] = directionMatrix(entries[i]);
    }
    const std::size_t firstIdx = static_cast<std::size_t>(o.refineFrom) - 1; // matrices index
    if (firstIdx < 2 || matrices.size() < 2 || firstIdx >= matrices.size() - 1) {
        QTA_LOG_ERROR("quantape.tools", "--refine-from={} out of range (table has {} dims)",
                      o.refineFrom, entries.size() + 1);
        return 1;
    }
    // Real dimensions only: matrices index of the last dimension is size-2.
    const std::size_t refineCount = matrices.size() - 1 - firstIdx;

    std::ofstream changeLog;
    if (!o.changelog.empty() && !o.dryRun) {
        changeLog.open(o.changelog, std::ios::app);
    }

    const auto dw = distanceWeights(o);
    std::printf("refining dims %u..%zu, window=+-%d, candidates=%d, epochs=%d, fraction=%.2f%s\n",
                o.refineFrom, entries.size() + 1, o.window, o.candidates, o.epochs, o.fraction,
                o.dryRun ? " (dry run)" : "");

    if (o.reportOnly) {
        // Parallel full-distribution report of the two-sided windowed scores.
        std::vector<double> scores(matrices.size(), 0.0);
        sobol::detail::parallelFor(refineCount, o.threads, [&](std::size_t k) {
            const std::size_t idx = firstIdx + k;
            scores[idx] = twoSidedScore(matrices[idx], matrices, idx, o, dw, kNoScore);
        });
        std::vector<double> sorted(scores.begin() + static_cast<std::ptrdiff_t>(firstIdx),
                                   scores.begin() +
                                       static_cast<std::ptrdiff_t>(matrices.size() - 1));
        std::sort(sorted.begin(), sorted.end());
        double sum = 0.0;
        int zeros = 0;
        for (double v : sorted) {
            sum += v;
            zeros += (v == 0.0);
        }
        const auto q = [&](double p) {
            return sorted[static_cast<std::size_t>(p * (sorted.size() - 1))];
        };
        std::printf("refine report: dims=%zu window=+-%d min=%.4g p25=%.4g median=%.4g p90=%.4g "
                    "p99=%.4g max=%.4g mean=%.4g zeros=%d\n",
                    sorted.size(), o.window, sorted.front(), q(0.25), q(0.5), q(0.9), q(0.99),
                    sorted.back(), sum / sorted.size(), zeros);
        return 0;
    }

    const auto startTime = std::chrono::steady_clock::now();
    for (int epoch = 0; epoch < o.epochs; ++epoch) {
        RefineStats stats;

        // Score every refinable dimension (parallel, read-only snapshot).
        std::vector<double> scores(matrices.size(), 0.0);
        sobol::detail::parallelFor(refineCount, o.threads, [&](std::size_t k) {
            const std::size_t idx = firstIdx + k;
            scores[idx] = twoSidedScore(matrices[idx], matrices, idx, o, dw, kNoScore);
        });
        const std::vector<double> scoresBefore = scores;
        const std::vector<Entry> entriesBefore = o.globalAccept ? entries : std::vector<Entry>{};
        const std::vector<DirectionMatrix> matricesBefore =
            o.globalAccept ? matrices : std::vector<DirectionMatrix>{};

        // Selection: all, top fraction, or those above --min-score.
        std::vector<std::size_t> selected;
        for (std::size_t idx = firstIdx; idx < matrices.size() - 1; ++idx) {
            if (scores[idx] > o.minScore) {
                selected.push_back(idx);
            }
        }
        const std::size_t keep =
            std::max<std::size_t>(1, static_cast<std::size_t>(o.fraction * selected.size()));
        if (selected.size() > keep) {
            std::partial_sort(selected.begin(),
                              selected.begin() + static_cast<std::ptrdiff_t>(keep), selected.end(),
                              [&](std::size_t a, std::size_t b) { return scores[a] > scores[b]; });
            selected.resize(keep);
        } else if (o.worstFirst) {
            std::sort(selected.begin(), selected.end(),
                      [&](std::size_t a, std::size_t b) { return scores[a] > scores[b]; });
        } else {
            std::sort(selected.begin(), selected.end());
        }

        std::printf("epoch %d: %zu dims above score %.3g (of %zu refinable), processing %zu\n",
                    epoch, selected.size(), o.minScore, refineCount,
                    std::min(selected.size(), keep));

        for (std::size_t s = 0; s < selected.size(); ++s) {
            const std::size_t idx = selected[s];
            const double oldScore = scores[idx];
            if (oldScore == 0.0) {
                continue;
            }
            ++stats.considered;
            stats.beforeMean += oldScore;
            stats.beforeMax = std::max(stats.beforeMax, oldScore);

            // Candidate sets (deterministic per dim/epoch).
            std::uint64_t state = o.seed ^ (0x9E3779B97F4A7C15ULL * (idx + 1)) ^
                                  (0xC2B2AE3D27D4EB4FULL * static_cast<std::uint64_t>(epoch + 1));
            std::vector<std::vector<std::uint64_t>> candidates(
                static_cast<std::size_t>(o.candidates));
            for (auto& m : candidates) {
                m = sobol::detail::randomDirectionSet(state, static_cast<int>(entries[idx - 1].s));
            }
            const std::uint64_t poly =
                gf2::decode_poly(static_cast<int>(entries[idx - 1].s), entries[idx - 1].a);
            const int degree = static_cast<int>(entries[idx - 1].s);

            std::vector<double> candScores(candidates.size(), kNoScore);
            sobol::detail::parallelFor(candidates.size(), o.threads, [&](std::size_t c) {
                const DirectionMatrix dm = directionMatrixFor(poly, degree, candidates[c]);
                candScores[c] = twoSidedScore(dm, matrices, idx, o, dw, oldScore);
            });
            std::size_t best = candidates.size();
            for (std::size_t c = 0; c < candidates.size(); ++c) {
                if (candScores[c] < oldScore &&
                    (best == candidates.size() || candScores[c] < candScores[best])) {
                    best = c;
                }
            }
            if (best != candidates.size()) {
                ++stats.improved;
                stats.afterMean += candScores[best];
                stats.afterMax = std::max(stats.afterMax, candScores[best]);
                if (changeLog.is_open()) {
                    changeLog << entries[idx - 1].dim << ',' << oldScore << ',' << candScores[best]
                              << ',';
                    for (std::size_t k = 0; k < entries[idx - 1].m.size(); ++k) {
                        changeLog << (k ? " " : "") << entries[idx - 1].m[k];
                    }
                    changeLog << ',';
                    for (std::size_t k = 0; k < candidates[best].size(); ++k) {
                        changeLog << (k ? " " : "") << candidates[best][k];
                    }
                    changeLog << '\n';
                }
                entries[idx - 1].m = candidates[best];
                matrices[idx] = directionMatrixFor(poly, degree, candidates[best]);
                scores[idx] = candScores[best];
            } else {
                stats.afterMean += oldScore;
                stats.afterMax = std::max(stats.afterMax, oldScore);
            }
            if (!o.dryRun && o.output.size() > 0 && (s + 1) % 500 == 0) {
                if (!writeTable(o.output, entries)) {
                    QTA_LOG_ERROR("quantape.tools", "could not write {}", o.output);
                    return 1;
                }
            }
        }

        std::printf("epoch %d: improved %zu/%zu, mean %.4g -> %.4g, max %.4g -> %.4g\n", epoch,
                    stats.improved, stats.considered,
                    stats.considered ? stats.beforeMean / stats.considered : 0.0,
                    stats.considered ? stats.afterMean / stats.considered : 0.0, stats.beforeMax,
                    stats.afterMax);

        // Global acceptance: the whole-table (max, mean) objective must not
        // worsen, otherwise revert the epoch.
        if (o.globalAccept && stats.improved > 0) {
            std::vector<double> after(matrices.size(), 0.0);
            sobol::detail::parallelFor(refineCount, o.threads, [&](std::size_t k) {
                const std::size_t idx = firstIdx + k;
                after[idx] = twoSidedScore(matrices[idx], matrices, idx, o, dw, kNoScore);
            });
            double beforeMax = 0.0, beforeSum = 0.0, afterMax = 0.0, afterSum = 0.0;
            for (std::size_t idx = firstIdx; idx < matrices.size() - 1; ++idx) {
                beforeMax = std::max(beforeMax, scoresBefore[idx]);
                beforeSum += scoresBefore[idx];
                afterMax = std::max(afterMax, after[idx]);
                afterSum += after[idx];
            }
            const double n = static_cast<double>(refineCount);
            const bool better =
                (afterMax < beforeMax) || (afterMax == beforeMax && afterSum < beforeSum);
            std::printf("epoch %d global: max %.4g -> %.4g, mean %.4g -> %.4g -> %s\n", epoch,
                        beforeMax, afterMax, beforeSum / n, afterSum / n,
                        better ? "accepted" : "REVERTED");
            if (!better) {
                entries = entriesBefore;
                matrices = matricesBefore;
                if (!o.dryRun && !o.output.empty()) {
                    writeTable(o.output, entries);
                }
                break;
            }
        }
        if (stats.improved == 0) {
            std::printf("epoch %d: no improvement, stopping\n", epoch);
            break;
        }
    }

    if (!o.dryRun && !o.output.empty()) {
        if (!writeTable(o.output, entries)) {
            QTA_LOG_ERROR("quantape.tools", "could not write {}", o.output);
            return 1;
        }
        std::printf("wrote refined table: %s\n", o.output.c_str());
    }
    const auto elapsed =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - startTime).count();
    std::printf("elapsed %.1fs\n", elapsed);
    std::printf("REFINE COMPLETE\n");
    return 0;
}
