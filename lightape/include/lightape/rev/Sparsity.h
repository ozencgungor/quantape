#ifndef LIGHTAPE_REV_SPARSITY_H
#define LIGHTAPE_REV_SPARSITY_H

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <utility>
#include <vector>

#include "lightape/rev/LeanReverse.h"

namespace lightape {
/**
 * @file Sparsity.h
 * @brief Exact sparse Hessian assembly from colored Hessian-vector products
 *        (decision #21: colored **exact** HVP groups, never finite
 *        differences).
 *
 * `SparsityPattern` lists the (symmetric) nonzero structure of the Hessian,
 * `greedyColoring` computes a deterministic distance-2 (star) coloring of the
 * pattern graph, and `coloredHessian` assembles the dense matrix with one
 * exact `RevTape::hessianVector` call per color instead of one per column.
 *
 * **Why distance-2.** One HVP with direction `v = sum_{i in c} e_i` produces
 * `(H v)_k = sum_{i in c} H_{ki}`. If no two columns `i != j` of the same
 * color share a row `k` of the pattern, the sum collapses to the single
 * pattern entry, so the whole column of every `i in c` is read out exactly:
 * `H_ii = (H e_c)_i` and `H_ij = H_ji = (H e_c)_j` for every pattern entry
 * `(i,j)`. Adjacent columns must get different colors (otherwise `H e_c`
 * already mixes them); columns sharing a neighbor must get different colors
 * (otherwise the shared row mixes them). That is precisely a coloring in
 * which vertices at graph distance 1 or 2 differ, i.e. a star / distance-2
 * coloring of the pattern adjacency graph. Every pattern entry `(i,j)` is
 * then recovered from the pass of `color(i)` or `color(j)`, so a pattern with
 * `nnz` entries costs `#colors` HVPs, not `nnz` and not `n`.
 *
 * **Exactness.** The pattern must be a **superset** of the true Hessian
 * sparsity: a missing entry `(k,i)` hides a same-color conflict from the
 * coloring and can pollute the extraction. A superset only adds constraints
 * (more colors), never wrong values. Entries outside the pattern stay zero,
 * and `coloredHessian` fails loudly (`false`) on a malformed pattern rather
 * than approximating (policy #26).
 *
 * **Cross-pair identity (fallback path).** For any pair `i != j`, with
 * `v = e_i + e_j`:
 *     `(H v)_i = H_ii + H_ij`,   `(H v)_j = H_ji + H_jj`,
 * and for a symmetric Hessian `H_ji = H_ij`, so the polarization identity
 *     `H_ij = 0.5 * ( (H(e_i+e_j))_i + (H(e_i+e_j))_j - H_ii - H_jj )`
 * is exact with **no sparsity assumption at all**. The diagonals `H_ii` and
 * `H_jj` are already exact from the colored passes (`(H e_c)_i = H_ii` for
 * `i in c`). Note that the guess `(H(e_i+e_j))_i - (H e_i)_i - (H e_j)_i` is
 * wrong: `(H e_i)_i = H_ii` and `(H e_j)_i = H_ji = H_ij`, so it cancels to
 * zero. `coloredHessian` keeps the correct identity as a fallback for
 * pattern entries the colored scatter did not certify; with a correct
 * distance-2 coloring that set is empty, so normal runs cost exactly
 * `#colors` HVPs. The fallback is an exactness guard, never an approximation.
 */

/// Symmetric Hessian sparsity pattern over `n` variables; `entries` holds
/// ordered `(row, column)` pairs (duplicates and self-loops are tolerated).
struct SparsityPattern {
    std::size_t n = 0;
    std::vector<std::pair<std::size_t, std::size_t>> entries;

    /// Symmetrize in place: append every transpose, sort and deduplicate.
    /// One-sided patterns (upper or lower triangle only) are accepted as-is
    /// by `greedyColoring` and `coloredHessian`; this helper materializes
    /// both orientations when a caller wants them explicit.
    void symmetric() {
        const std::size_t count = entries.size();
        entries.reserve(2 * count);
        for (std::size_t t = 0; t < count; ++t) {
            entries.emplace_back(entries[t].second, entries[t].first);
        }
        std::sort(entries.begin(), entries.end());
        entries.erase(std::unique(entries.begin(), entries.end()), entries.end());
    }

    /// Symmetric band pattern with `|i - j| <= bandwidth`, diagonal always
    /// included: bandwidth 1 = tridiagonal, 2 = pentadiagonal,
    /// 3 = heptadiagonal (so a band of half-width `b` needs `2b + 1` colors
    /// under the distance-2 rule, for `n` large enough).
    static SparsityPattern banded(std::size_t n, std::size_t bandwidth) {
        SparsityPattern pattern;
        pattern.n = n;
        if (n == 0) {
            return pattern;
        }
        const std::size_t last = n - 1;
        for (std::size_t i = 0; i < n; ++i) {
            const std::size_t lo = i - std::min(i, bandwidth);
            const std::size_t hi = i + std::min(bandwidth, last - i);
            for (std::size_t j = lo; j <= hi; ++j) {
                pattern.entries.emplace_back(i, j);
            }
        }
        return pattern;
    }
};

/// Deterministic greedy distance-2 (star) coloring of the symmetric pattern
/// graph. Vertices are visited in natural order; each takes the smallest
/// color not used by an already-colored vertex at graph distance 1 or 2
/// (adjacent or sharing a neighbor). Self-loops and out-of-range entries are
/// ignored. After the sweep every distance-2 pair is re-checked; the function
/// returns the number of colors (colors are contiguous from 0, 0 for `n == 0`)
/// or 0 if that self-check ever finds a conflict, which a correct run cannot.
/// `colorOut` is sized to `n`.
inline std::size_t greedyColoring(const SparsityPattern& pattern,
                                  std::vector<std::size_t>& colorOut) {
    const std::size_t n = pattern.n;
    colorOut.assign(n, 0);
    if (n == 0) {
        return 0;
    }

    // Undirected adjacency: (i,j), i != j, is an edge in both directions.
    std::vector<std::vector<std::size_t>> adj(n);
    for (const std::pair<std::size_t, std::size_t>& e : pattern.entries) {
        const std::size_t i = e.first;
        const std::size_t j = e.second;
        if (i >= n || j >= n || i == j) {
            continue;
        }
        adj[i].push_back(j);
        adj[j].push_back(i);
    }
    for (std::vector<std::size_t>& neighbors : adj) {
        std::sort(neighbors.begin(), neighbors.end());
        neighbors.erase(std::unique(neighbors.begin(), neighbors.end()), neighbors.end());
    }

    // Smallest-color-first sweep: forbid the colors of every *earlier* vertex
    // at distance 1 or 2, then take the smallest color left.
    std::vector<unsigned char> forbidden;
    std::size_t used = 0;
    for (std::size_t i = 0; i < n; ++i) {
        forbidden.assign(used + 1, 0);
        for (const std::size_t k : adj[i]) {
            if (k < i) {
                forbidden[colorOut[k]] = 1;
            }
            for (const std::size_t j : adj[k]) {
                if (j < i) {
                    forbidden[colorOut[j]] = 1;
                }
            }
        }
        std::size_t c = 0;
        while (forbidden[c] != 0) {
            ++c;
        }
        colorOut[i] = c;
        if (c + 1 > used) {
            used = c + 1;
        }
    }

    // Self-check by recomputing every distance <= 2 pair. The local rule
    // cannot produce a conflict; returning 0 on one turns a corrupt coloring
    // into a loud failure instead of silently wrong Hessian entries.
    for (std::size_t i = 0; i < n; ++i) {
        for (const std::size_t k : adj[i]) {
            if (colorOut[i] == colorOut[k]) {
                return 0;
            }
            for (const std::size_t j : adj[k]) {
                if (j != i && colorOut[i] == colorOut[j]) {
                    return 0;
                }
            }
        }
    }
    return used;
}

/// Exact sparse Hessian by color groups. `inputs[0..nInputs)` are the tape
/// slots of the differentiation variables, `pattern` the Hessian sparsity
/// (must satisfy `pattern.n == nInputs`), and `outDense` the row-major
/// `nInputs x nInputs` result (`outDense[i * nInputs + j] = H_ij`), zeroed
/// first and then filled for every pattern entry and its transpose.
///
/// One `hessianVector` per color extracts all columns of that color; entries
/// the colored scatter did not certify fall back to the exact polarization
/// HVP (see the file comment). `hvpScratch`, when non-null, must point to
/// `2 * nInputs` doubles used as the direction and HVP buffers (avoids the
/// internal heap allocation); stack storage covers `nInputs <= 64`.
///
/// Returns false on an invalid pattern (`pattern.n != nInputs`, a pattern
/// entry out of range, or a corrupt coloring) and on the first failing
/// `hessianVector` (e.g. an `External` block without svp/vhp). On failure
/// `outDense` holds an unspecified partial assembly (it is zeroed on entry
/// and filled color by color); an empty pattern with valid `n` returns true
/// with the zero matrix and performs no HVP at all.
inline bool coloredHessian(RevTape& tape, std::uint32_t root, const std::uint32_t* inputs,
                           std::size_t nInputs, const SparsityPattern& pattern, double* outDense,
                           double* hvpScratch = nullptr) {
    if (pattern.n != nInputs) {
        return false;
    }
    for (const std::pair<std::size_t, std::size_t>& e : pattern.entries) {
        if (e.first >= nInputs || e.second >= nInputs) {
            return false;
        }
    }
    const std::size_t nSquared = nInputs * nInputs;
    for (std::size_t t = 0; t < nSquared; ++t) {
        outDense[t] = 0.0;
    }
    if (nInputs == 0 || pattern.entries.empty()) {
        return true; // nothing requested: the zero matrix is exact
    }

    std::vector<std::size_t> color;
    const std::size_t nColors = greedyColoring(pattern, color);
    if (nColors == 0) {
        return false; // self-check failed; no exact extraction is guaranteed
    }

    constexpr std::size_t kStackCap = 64;
    double stackScratch[2 * kStackCap];
    std::vector<double> heapScratch;
    double* dir = stackScratch;
    double* hvp = stackScratch + kStackCap;
    if (hvpScratch != nullptr) {
        dir = hvpScratch;
        hvp = hvpScratch + nInputs;
    } else if (nInputs > kStackCap) {
        heapScratch.resize(2 * nInputs);
        dir = heapScratch.data();
        hvp = heapScratch.data() + nInputs;
    }

    const std::size_t nEntries = pattern.entries.size();
    std::vector<unsigned char> covered(nEntries, 0);
    std::vector<double> diagonal(nInputs, 0.0);

    // One exact HVP per color: v = sum_{i in color c} e_i.
    for (std::size_t c = 0; c < nColors; ++c) {
        for (std::size_t i = 0; i < nInputs; ++i) {
            dir[i] = (color[i] == c) ? 1.0 : 0.0;
        }
        if (!tape.hessianVector(root, inputs, nInputs, dir, hvp)) {
            return false;
        }
        for (std::size_t i = 0; i < nInputs; ++i) {
            if (color[i] == c) {
                diagonal[i] = hvp[i]; // (H e_c)_i = H_ii: no same-color neighbor
            }
        }
        for (std::size_t e = 0; e < nEntries; ++e) {
            const std::size_t i = pattern.entries[e].first;
            const std::size_t j = pattern.entries[e].second;
            if (color[i] == c) {
                outDense[i * nInputs + j] = hvp[j];
                outDense[j * nInputs + i] = hvp[j];
                covered[e] = 1;
            } else if (color[j] == c) {
                outDense[i * nInputs + j] = hvp[i];
                outDense[j * nInputs + i] = hvp[i];
                covered[e] = 1;
            }
        }
    }

    // Fallback: exact pair extraction for entries the colored scatter did not
    // certify. Unreachable for a properly colored superset pattern; kept so a
    // defective coloring degrades to an exact O(#uncovered) HVP tail instead
    // of silently wrong entries.
    for (std::size_t e = 0; e < nEntries; ++e) {
        if (covered[e] != 0) {
            continue;
        }
        const std::size_t i = pattern.entries[e].first;
        const std::size_t j = pattern.entries[e].second;
        if (i == j) {
            for (std::size_t k = 0; k < nInputs; ++k) {
                dir[k] = (k == i) ? 1.0 : 0.0;
            }
            if (!tape.hessianVector(root, inputs, nInputs, dir, hvp)) {
                return false;
            }
            diagonal[i] = hvp[i];
            outDense[i * nInputs + i] = hvp[i];
            continue;
        }
        for (std::size_t k = 0; k < nInputs; ++k) {
            dir[k] = (k == i || k == j) ? 1.0 : 0.0;
        }
        if (!tape.hessianVector(root, inputs, nInputs, dir, hvp)) {
            return false;
        }
        const double hij = 0.5 * ((hvp[i] - diagonal[i]) + (hvp[j] - diagonal[j]));
        outDense[i * nInputs + j] = hij;
        outDense[j * nInputs + i] = hij;
    }
    return true;
}

} // namespace lightape

#endif // LIGHTAPE_REV_SPARSITY_H
