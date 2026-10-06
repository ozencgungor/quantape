/**
 * @file sobol_to_binary.cpp
 * @brief Convert a Joe-Kuo text table into the QSB1 binary asset used by
 *        SobolGenerator::fromBinary / fromDefaultTable (zero-copy mmap).
 *
 * Layout: "QSB1" | u32 dims | u32 maxBits | u32 wordBytes | u32 pad
 *         then (dims-1) x maxBits words, row-major by dimension (dim 2..dims).
 *
 * Usage: sobol_to_binary [--bits=32|64] [--dims=N] input.txt output.qsb
 *        --dims=N clamps the table to dimensions 2..N (default: all).
 */
#include "quantape/log/Log.h"
#include "quantape/math/Random/Sobol/DirectionNumbers.h"
#include "quantape/math/Random/Sobol/GF2.h"

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

using namespace quantape::math::mc;
using namespace quantape::math::mc::sobol;

int main(int argc, char** argv) {
    int bits = 32;
    std::uint32_t maxDims = 0; // 0 = all
    std::string input, output;
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg.rfind("--bits=", 0) == 0) {
            bits = std::atoi(arg.c_str() + 7);
        } else if (arg.rfind("--dims=", 0) == 0) {
            maxDims = static_cast<std::uint32_t>(std::atoi(arg.c_str() + 7));
        } else if (input.empty()) {
            input = arg;
        } else {
            output = arg;
        }
    }
    if (input.empty() || output.empty() || bits < 1 || bits > 64) {
        QTA_LOG_ERROR("quantape.tools",
                      "usage: sobol_to_binary [--bits=32|64] [--dims=N] input.txt output.qsb");
        return 1;
    }
    const auto entries = load_joe_kuo(input);
    if (entries.empty()) {
        QTA_LOG_ERROR("quantape.tools", "cannot load {}", input);
        return 1;
    }
    std::uint32_t dims = entries.back().dim;
    if (maxDims > 0) {
        dims = std::min(dims, std::max(maxDims, 2u));
    }
    const std::uint32_t wordBytes = (bits <= 32) ? 4 : 8;
    std::FILE* out = std::fopen(output.c_str(), "wb");
    if (out == nullptr) {
        QTA_LOG_ERROR("quantape.tools", "cannot write {}", output);
        return 1;
    }
    std::fwrite("QSB1", 1, 4, out);
    const std::uint32_t header[4] = {dims, static_cast<std::uint32_t>(bits), wordBytes, 0};
    std::fwrite(header, sizeof(std::uint32_t), 4, out);

    std::vector<std::uint32_t> w32(bits);
    std::vector<std::uint64_t> w64(bits);
    for (std::size_t i = 0; i + 2 <= dims; ++i) {
        const Entry& e = entries[i];
        const std::uint64_t poly = gf2::decode_poly(static_cast<int>(e.s), e.a);
        std::uint64_t m[65] = {0};
        for (std::uint32_t k = 1; k <= e.s && k <= static_cast<std::uint32_t>(bits); ++k) {
            m[k] = e.m[k - 1];
        }
        for (int k = static_cast<int>(e.s) + 1; k <= bits; ++k) {
            std::uint64_t value = (m[k - e.s] << e.s) ^ m[k - e.s];
            for (int j = 1; j < static_cast<int>(e.s); ++j) {
                if ((poly >> (e.s - j)) & 1) {
                    value ^= (m[k - j] << j);
                }
            }
            m[k] = value;
        }
        if (wordBytes == 4) {
            for (int k = 1; k <= bits; ++k) {
                w32[static_cast<std::size_t>(k - 1)] = static_cast<std::uint32_t>(m[k]) << (32 - k);
            }
            std::fwrite(w32.data(), sizeof(std::uint32_t), static_cast<std::size_t>(bits), out);
        } else {
            for (int k = 1; k <= bits; ++k) {
                w64[static_cast<std::size_t>(k - 1)] = m[k] << (64 - k);
            }
            std::fwrite(w64.data(), sizeof(std::uint64_t), static_cast<std::size_t>(bits), out);
        }
    }
    std::fclose(out);
    std::printf("wrote %s: %u dims, %d bits, %u-byte words\n", output.c_str(), dims, bits,
                wordBytes);
    return 0;
}
