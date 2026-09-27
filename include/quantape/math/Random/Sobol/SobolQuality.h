#pragma once
/**
 * @file SobolQuality.h
 * @brief Joe-Kuo 2D projection t-values, weighted search criterion D(q), Property A.
 *
 * Implements the quality measures from
 *   S. Joe, F. Y. Kuo, "Constructing Sobol sequences with better two-dimensional
 *   projections", SIAM J. Sci. Comput. 30 (2008) 2635-2654.
 *
 * - tValue2D(j, d, m): t-value of the 2D projection of dimensions j and d for the
 *   first 2^m Sobol points, computed from the 32x32 corner matrices C_{32,j},
 *   C_{32,d} (Section 2.3, pseudocode Figure 2.1). Validated against Tables
 *   2.1 and 3.8 of the paper.
 * - weightedT(candidate, previous, m): T^(d; m) = max_j t(j, d; m) * 0.9999^{j-1}.
 * - criterionD(...): D(q)(d; mmin, mmax) = max_m T^(d;m)^q / (m - T^(d;m) + 1).
 * - PropertyAChecker: det(V_d) == 1 (mod 2) on the first-bit matrix (Section 2.4).
 *
 * Conventions:
 * - Rows of the direction matrices are 32-bit masks: bit (c-1) is column c, i.e.
 *   the digit r of direction number v_c, for c, r in 1..32.
 * - `previous` is indexed by absolute dimension: previous[i] is dimension i+1,
 *   with previous[0] the identity matrix of dimension 1.
 */
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <limits>
#include <vector>

#include "DirectionNumbers.h"
#include "GF2.h"

namespace quantape::math::mc {
namespace sobol {

/// Sentinel for "no finite score". Finite (not infinity) so it stays valid
/// under -ffast-math, where infinities are undefined behaviour.
inline constexpr double kNoScore = std::numeric_limits<double>::max();

// ═══════════════════════════════════════════════════════════════════════════
// Direction matrices (32 x 32 generator corners)
// ═══════════════════════════════════════════════════════════════════════════

struct DirectionMatrix {
    int degree = 0;
    std::array<std::uint32_t, 32> rows{};
};

/// Integer direction numbers m_1..m_count (m_k odd, m_k < 2^k) for an initial
/// set m_1..m_s of a degree-s polynomial. Values fit in 64 bits for count <= 64.
inline std::vector<std::uint64_t> extendDirectionNumbers(std::uint64_t polynomial, int degree,
                                                         const std::vector<std::uint64_t>& initial,
                                                         int count) {
    std::vector<std::uint64_t> m(static_cast<std::size_t>(count) + 1, 0);
    for (int k = 1; k <= degree && k <= count; ++k) {
        m[static_cast<std::size_t>(k)] = initial[static_cast<std::size_t>(k - 1)];
    }
    for (int k = degree + 1; k <= count; ++k) {
        std::uint64_t value = (m[static_cast<std::size_t>(k - degree)] << degree) ^
                              m[static_cast<std::size_t>(k - degree)];
        for (int j = 1; j < degree; ++j) {
            if ((polynomial >> (degree - j)) & 1) {
                value ^= (m[static_cast<std::size_t>(k - j)] << j);
            }
        }
        m[static_cast<std::size_t>(k)] = value;
    }
    return m;
}

/// Integer direction numbers m_1..m_count for a Joe-Kuo entry.
inline std::vector<std::uint64_t> integerDirectionNumbers(const Entry& e, int count) {
    return extendDirectionNumbers(gf2::decode_poly(static_cast<int>(e.s), e.a),
                                  static_cast<int>(e.s), e.m, count);
}

/// C32 matrix from a given initial direction number vector (may be shorter than
/// 32; the recurrence extends it), for a polynomial of the given degree.
inline DirectionMatrix directionMatrixFor(std::uint64_t polynomial, int degree,
                                          const std::vector<std::uint64_t>& initial) {
    const auto m = extendDirectionNumbers(polynomial, degree, initial, 32);
    DirectionMatrix dm;
    dm.degree = degree;
    for (int r = 1; r <= 32; ++r) {
        for (int c = r; c <= 32; ++c) {
            if ((m[static_cast<std::size_t>(c)] >> (c - r)) & 1) {
                dm.rows[static_cast<std::size_t>(r - 1)] |= (1u << (c - 1));
            }
        }
    }
    return dm;
}

/// C32 matrix for a Joe-Kuo entry.
inline DirectionMatrix directionMatrix(const Entry& e) {
    return directionMatrixFor(gf2::decode_poly(static_cast<int>(e.s), e.a), static_cast<int>(e.s),
                              e.m);
}

/// Dimension 1 (identity corner matrix).
inline DirectionMatrix identityMatrix() {
    DirectionMatrix dm;
    dm.degree = 1;
    for (int r = 1; r <= 32; ++r) {
        dm.rows[static_cast<std::size_t>(r - 1)] = (1u << (r - 1));
    }
    return dm;
}

// ═══════════════════════════════════════════════════════════════════════════
// t-value of a 2D projection
// ═══════════════════════════════════════════════════════════════════════════

namespace detail {

inline std::uint32_t lowMask(int m) {
    return m >= 32 ? 0xFFFFFFFFu : ((1u << m) - 1u);
}

/// Independence of the first rj rows of `first` and first rd rows of `second`,
/// restricted to the first m columns. Exploits the triangular structure: the
/// rows of `second` are reduced against the triangular basis of `first`; rows
/// of `second` beyond rj are untouched and take part in the rank check.
inline bool splitIndependent(const DirectionMatrix& first, const DirectionMatrix& second, int rj,
                             int rd, int m) {
    const std::uint32_t mask = lowMask(m);
    const int k = std::min(rj, rd);
    std::uint32_t reduced[32];
    int count = 0;
    for (int r = 1; r <= k; ++r) {
        std::uint32_t v = second.rows[static_cast<std::size_t>(r - 1)] & mask;
        for (int c = r; c <= rj; ++c) {
            if ((v >> (c - 1)) & 1) {
                v ^= (first.rows[static_cast<std::size_t>(c - 1)] & mask);
            }
        }
        reduced[count++] = v;
    }
    for (int r = k + 1; r <= rd; ++r) {
        reduced[count++] = second.rows[static_cast<std::size_t>(r - 1)] & mask;
    }
    // rank of rd vectors
    std::uint32_t pivots[32] = {};
    int rank = 0;
    for (int i = 0; i < count; ++i) {
        std::uint32_t v = reduced[i];
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
    return rank == count;
}

} // namespace detail

/// t(j, d; m): smallest t such that for every rj + rd = m - t the first rj rows
/// of `first` (dimension j) and first rd rows of `second` (dimension d) are
/// linearly independent (Joe-Kuo 2008, Figure 2.1).
inline int tValue2D(const DirectionMatrix& first, const DirectionMatrix& second, int m) {
    for (int t = 0; t <= m; ++t) {
        bool ok = true;
        for (int rj = 0; rj <= m - t && ok; ++rj) {
            ok = detail::splitIndependent(first, second, rj, m - t - rj, m);
        }
        if (ok) {
            return t;
        }
    }
    return m;
}

// ═══════════════════════════════════════════════════════════════════════════
// Search criterion
// ═══════════════════════════════════════════════════════════════════════════

struct CriterionOptions {
    int window = 128; ///< 0 or negative: all previous dimensions
    int mMin = 1;     ///< first m in the criterion range
    int mMax = 31;    ///< last m (31 keeps the matrix words at 32 bits)
    double weightBase = 0.9999;
    double exponent = 6.0; ///< q in D(q)
};

/// T^(d; m) = max_j t(j, d; m) * weights[j] over the (windowed) previous dims.
/// Pruning: contributions are skipped when the Joe-Kuo bound t <= min(m, s_j+s_d-2)
/// cannot exceed the current maximum, and the scan stops (returning infinity)
/// once this m alone already scores worse than `abortAbove`.
inline double weightedT(const DirectionMatrix& candidate,
                        const std::vector<DirectionMatrix>& previous,
                        const std::vector<double>& weights, int m, int window, double exponent,
                        double abortAbove) {
    const int n = static_cast<int>(previous.size());
    const int jBegin = (window <= 0) ? 0 : std::max(0, n - window);
    double worst = 0.0;
    for (int i = jBegin; i < n; ++i) {
        const double bound =
            std::min(static_cast<double>(m),
                     static_cast<double>(previous[static_cast<std::size_t>(i)].degree +
                                         candidate.degree - 2)) *
            weights[static_cast<std::size_t>(i)];
        if (bound <= worst) {
            continue;
        }
        const int t = tValue2D(previous[static_cast<std::size_t>(i)], candidate, m);
        if (t == 0) {
            continue;
        }
        const double weighted = static_cast<double>(t) * weights[static_cast<std::size_t>(i)];
        if (weighted > worst) {
            worst = weighted;
            const double value = std::pow(worst, exponent) / (static_cast<double>(m) - worst + 1.0);
            if (value > abortAbove) {
                return kNoScore;
            }
        }
    }
    return worst;
}

/// D(q)(d; mmin, mmax) = max_m T^(d;m)^q / (m - T^(d;m) + 1); lower is better.
/// Candidates whose partial score already exceeds `abortAbove` return infinity.
inline double criterionD(const DirectionMatrix& candidate,
                         const std::vector<DirectionMatrix>& previous,
                         const std::vector<double>& weights, const CriterionOptions& options,
                         double abortAbove = kNoScore) {
    double best = 0.0;
    for (int m = options.mMin; m <= options.mMax; ++m) {
        const double T = weightedT(candidate, previous, weights, m, options.window,
                                   options.exponent, abortAbove);
        if (T >= kNoScore) {
            return kNoScore;
        }
        const double denominator = static_cast<double>(m) - T + 1.0;
        const double value = std::pow(T, options.exponent) / denominator;
        if (value > best) {
            best = value;
            if (best > abortAbove) {
                return kNoScore;
            }
        }
    }
    return best;
}

// ═══════════════════════════════════════════════════════════════════════════
// Property A
// ═══════════════════════════════════════════════════════════════════════════

/// First bit (digit 1) of every direction number of a dimension, up to `count`.
inline std::vector<int> firstBitSequence(const Entry& e, int count) {
    std::vector<int> f(static_cast<std::size_t>(count) + 1, 0);
    const std::uint64_t poly = gf2::decode_poly(static_cast<int>(e.s), e.a);
    const auto m = integerDirectionNumbers(e, static_cast<int>(e.s));
    for (int k = 1; k <= static_cast<int>(e.s) && k <= count; ++k) {
        f[static_cast<std::size_t>(k)] =
            static_cast<int>((m[static_cast<std::size_t>(k)] >> (k - 1)) & 1);
    }
    for (int k = static_cast<int>(e.s) + 1; k <= count; ++k) {
        int value = f[static_cast<std::size_t>(k - e.s)];
        for (int j = 1; j < static_cast<int>(e.s); ++j) {
            if ((poly >> (e.s - j)) & 1) {
                value ^= f[static_cast<std::size_t>(k - j)];
            }
        }
        f[static_cast<std::size_t>(k)] = value;
    }
    return f;
}

/// Incremental det(V_d) == 1 (mod 2) checker, Joe-Kuo (2008) Section 2.4/3.5.
class PropertyAChecker {
public:
    explicit PropertyAChecker(int maxDimension)
        : maxDimension_(maxDimension), words_((maxDimension + 63) / 64),
          rows_(static_cast<std::size_t>(maxDimension) * words_, 0),
          pivots_(static_cast<std::size_t>(maxDimension), -1) {}

    /// Add the next dimension; returns false if det(V_d) == 0 (Property A lost).
    bool add(const Entry& e) {
        std::vector<std::uint64_t> row(words_, 0);
        if (nextDimension_ == 1) {
            row[0] = 1;
        } else {
            const auto f = firstBitSequence(e, maxDimension_);
            for (int k = 1; k <= maxDimension_; ++k) {
                if (f[static_cast<std::size_t>(k)] != 0) {
                    row[static_cast<std::size_t>((k - 1) / 64)] |= (1ULL << ((k - 1) % 64));
                }
            }
        }
        const int pivot = reduce(row);
        if (pivot < 0) {
            return false;
        }
        for (std::size_t w = 0; w < words_; ++w) {
            rows_[static_cast<std::size_t>(pivot) * words_ + w] = row[w];
        }
        pivots_[static_cast<std::size_t>(pivot)] = static_cast<int>(nextDimension_);
        ++nextDimension_;
        return true;
    }

    /// Whether the next dimension with this entry keeps Property A (state unchanged).
    bool accepts(const Entry& e) const {
        PropertyAChecker copy = *this;
        return copy.add(e);
    }

    int nextDimension() const { return nextDimension_; }

private:
    /// Row reduction over GF(2); returns the new pivot position or -1 if dependent.
    int reduce(std::vector<std::uint64_t>& row) const {
        for (int b = maxDimension_ - 1; b >= 0; --b) {
            if ((row[static_cast<std::size_t>(b / 64)] >> (b % 64)) & 1) {
                if (pivots_[static_cast<std::size_t>(b)] < 0) {
                    return b;
                }
                const std::size_t base = static_cast<std::size_t>(b) * words_;
                for (std::size_t w = 0; w < words_; ++w) {
                    row[w] ^= rows_[base + w];
                }
            }
        }
        return -1;
    }

    int maxDimension_;
    std::size_t words_;
    std::vector<std::uint64_t> rows_;
    std::vector<int> pivots_;
    int nextDimension_ = 1;
};

} // namespace sobol
} // namespace quantape::math::mc
