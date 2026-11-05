#include <chrono>
#include <cstdint>
#include <cstdio>

#include "lightape/lightape.h"

namespace {

using lightape::RevScalar;
using lightape::RevTape;

constexpr int kIterations = 200000;
constexpr int kTerms = 32;

} // namespace

int main() {
    RevTape& tape = RevTape::active();
    double sink = 0.0;
    std::uint32_t firstNode = RevTape::kNone;

    const auto start = std::chrono::steady_clock::now();
    for (int it = 0; it < kIterations; ++it) {
        tape.clear();
        RevScalar acc(0.0);
        for (int i = 0; i < kTerms; ++i) {
            const double v = 1.0 + 0.001 * static_cast<double>(i);
            RevScalar x(v, tape.input(v));
            if (i == 0) {
                firstNode = x.node;
            }
            acc += x * x;
        }
        RevScalar f = sqrt(acc);
        tape.reverse(f.node);
        sink += tape.adjoint(firstNode);
    }
    const auto end = std::chrono::steady_clock::now();
    const double ns =
        static_cast<double>(
            std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count()) /
        static_cast<double>(kIterations);
    std::printf("lean reverse: %.1f ns/iter (%d terms, %d iterations, sink=%f)\n", ns, kTerms,
                kIterations, sink);
    return 0;
}
