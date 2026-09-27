#pragma once
/**
 * @file DirectionNumbers.h
 * @brief Sobol direction number recurrence, quality evaluation, and I/O.
 *
 * Direction numbers v[i] are stored as W-bit left-justified integers:
 *   v[i] = m[i] << (W - i - 1)
 * where m[i] is the "raw" direction number (odd, m[i] < 2^(i+1)).
 */

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#include "GF2.h"

namespace quantape::math::mc {
namespace sobol {

constexpr int W = 64; // word size (bits)

// ═══════════════════════════════════════════════════════════════════════════
// Direction number recurrence
// ═══════════════════════════════════════════════════════════════════════════

// Given primitive polynomial of degree s and initial direction numbers m[0..s-1],
// compute the full W direction numbers as left-justified W-bit integers.
//
// Recurrence for i ≥ s:
//   v[i] = c_1·v[i-1] ⊕ c_2·v[i-2] ⊕ ... ⊕ c_{s-1}·v[i-s+1]
//          ⊕ v[i-s] ⊕ (v[i-s] >> s)
//
// where c_j = coefficient of x^{s-j} in poly (the j-th internal coefficient).
inline std::vector<uint64_t> compute_v(uint64_t poly, int s, const std::vector<uint64_t>& m) {
    std::vector<uint64_t> v(W);

    // Initial: v[i] = m[i] << (W - i - 1)
    for (int i = 0; i < s && i < W; ++i)
        v[i] = m[i] << (W - i - 1);

    // Recurrence
    for (int i = s; i < W; ++i) {
        uint64_t vi = v[i - s] ^ (v[i - s] >> s);
        for (int j = 1; j < s; ++j) {
            // c_j = bit (s-j) of poly (excluding leading x^s and trailing 1)
            if ((poly >> (s - j)) & 1)
                vi ^= v[i - j];
        }
        v[i] = vi;
    }
    return v;
}

// ═══════════════════════════════════════════════════════════════════════════
// Joe-Kuo file I/O
// ═══════════════════════════════════════════════════════════════════════════

struct Entry {
    uint32_t dim;
    uint32_t s;              // polynomial degree
    uint64_t a;              // encoded polynomial
    std::vector<uint64_t> m; // initial direction numbers m[0..s-1]
};

// Load Joe-Kuo format file (header line is skipped)
inline std::vector<Entry> load_joe_kuo(const std::string& path) {
    std::vector<Entry> entries;
    std::ifstream in(path);
    if (!in.is_open())
        return entries;

    std::string line;
    std::getline(in, line); // skip header

    while (std::getline(in, line)) {
        if (line.empty() || line[0] == '#')
            continue;
        std::istringstream iss(line);
        Entry e;
        iss >> e.dim >> e.s >> e.a;
        e.m.resize(e.s);
        for (uint32_t i = 0; i < e.s; ++i)
            iss >> e.m[i];
        if (iss)
            entries.push_back(std::move(e));
    }
    return entries;
}

// Save in Joe-Kuo format
inline void save_joe_kuo(const std::string& path, const std::vector<Entry>& entries,
                         bool append = false) {
    auto mode = append ? std::ios::app : std::ios::out;
    std::ofstream out(path, mode);
    if (!append)
        out << "d\ts\ta\tm_i\n";

    for (auto& e : entries) {
        out << e.dim << "\t" << e.s << "\t" << e.a;
        for (auto mi : e.m)
            out << "\t" << mi;
        out << "\n";
    }
}

// Convert Entry to full direction numbers
inline std::vector<uint64_t> entry_to_v(const Entry& e) {
    uint64_t poly = gf2::decode_poly(e.s, e.a);
    return compute_v(poly, e.s, e.m);
}

} // namespace sobol
} // namespace quantape::math::mc
