#include <algorithm>
#include <array>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdio>

#include "lightape/batch/BatchTape.h"

namespace {

using lightape::BatchScalar;
using lightape::BatchTape;

constexpr int kIterations = 100000;
constexpr int kTerms = 16;
constexpr int kRounds = 5;

template <std::size_t W>
std::array<double, W> LaneValues(double base, double step) {
    std::array<double, W> v{};
    for (std::size_t l = 0; l < W; ++l) {
        v[l] = base + step * static_cast<double>(l);
    }
    return v;
}

/// The arithmetic composite: + - * / unary- sqrt exp log pow abs and the two
/// branch ops (lane-uniform by construction, no select nodes) per term.
template <std::size_t W>
double RunArithmetic(int iterations, double& sinkOut) {
    using Scalar = BatchScalar<W>;
    using Tape = BatchTape<W>;

    Tape& tape = Tape::active();
    double sink = 0.0;
    std::uint32_t firstNode = Tape::kNone;
    for (int it = 0; it < iterations; ++it) {
        tape.clear();
        Scalar acc(0.0);
        for (int i = 0; i < kTerms; ++i) {
            const double base = 1.2 + 0.001 * static_cast<double>(i);
            const Scalar x(tape.input(LaneValues<W>(base, 0.01)));
            if (i == 0) {
                firstNode = x.node;
            }
            acc = acc + x * x;
            acc = acc - x / (x + 2.0);
            acc = acc + sqrt(fabs(x - 1.0));
        }
        const Scalar f = exp(acc * 0.01) + log(acc + 1.0);
        const Scalar g = pow(fabs(f), 1.5) + fmin(f, f - 1.0) + fmax(f, f + 1.0);
        tape.reverse(g.node, 1.0);
        sink += tape.adjoint(firstNode)[0];
    }
    sinkOut = sink;
    return sink;
}

/// The math-heavy composite: normcdf(exp) + log + pow + erf + sin/cos + tanh
/// per term, closed by exp + log1p. Exercises the full Tier A op set through
/// the `VectorMath` seam.
template <std::size_t W>
double RunMath(int iterations, double& sinkOut) {
    using Scalar = BatchScalar<W>;
    using Tape = BatchTape<W>;

    Tape& tape = Tape::active();
    double sink = 0.0;
    std::uint32_t firstNode = Tape::kNone;
    for (int it = 0; it < iterations; ++it) {
        tape.clear();
        Scalar acc(0.0);
        for (int i = 0; i < kTerms; ++i) {
            const double base = 0.35 + 0.01 * static_cast<double>(i);
            const Scalar x(tape.input(LaneValues<W>(base, 0.005)));
            if (i == 0) {
                firstNode = x.node;
            }
            Scalar s(0.0);
            Scalar c(0.0);
            sincos(x, s, c);
            acc = acc + normcdf(exp(x)) + log(x + 1.0) + pow(x + 0.5, 2.5);
            acc = acc + erf(x) + s * s + c * 0.5 + tanh(x);
        }
        const Scalar f = exp(acc * 0.01) + log1p(acc);
        tape.reverse(f.node, 1.0);
        sink += tape.adjoint(firstNode)[0];
    }
    sinkOut = sink;
    return sink;
}

double TimeBatch(int iterations, double& sinkOut, double (*run)(int, double&)) {
    const auto start = std::chrono::steady_clock::now();
    [[maybe_unused]] const double sink = run(iterations, sinkOut);
    const auto end = std::chrono::steady_clock::now();
    const double ns = static_cast<double>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count());
    return ns / static_cast<double>(iterations);
}

double Median(std::array<double, kRounds> ns) {
    std::sort(ns.begin(), ns.end());
    return ns[kRounds / 2];
}

void Report(const char* name, const std::array<double, kRounds>& ns1,
            const std::array<double, kRounds>& ns4, const std::array<double, kRounds>& ns8,
            double sink) {
    const double med1 = Median(ns1);
    const double med4 = Median(ns4);
    const double med8 = Median(ns8);
    std::printf("%s composite (%d terms, %d iterations/round, %d rounds, sink=%.6g):\n", name,
                kTerms, kIterations, kRounds, sink);
    std::printf("  W=1: %8.1f ns/iter, %7.1f ns/path\n", med1, med1);
    std::printf("  W=4: %8.1f ns/iter, %7.1f ns/path (per-path %.2fx vs W=1)\n", med4, med4 / 4.0,
                med1 / (med4 / 4.0));
    std::printf("  W=8: %8.1f ns/iter, %7.1f ns/path (per-path %.2fx vs W=1)\n", med8, med8 / 8.0,
                med1 / (med8 / 8.0));
}

} // namespace

int main() {
#if defined(LIGHTAPE_VECTOR_MATH_SLEEF)
    std::printf("[bench] backend=sleef isa=%s vWidth=%zu\n",
                lightape::batch::VectorMath<1>::isaName(),
                lightape::batch::VectorMath<1>::vWidth());
#else
    std::printf("[bench] backend=libm-scalar isa=libm vWidth=1\n");
#endif
    std::array<double, kRounds> arithmetic1{};
    std::array<double, kRounds> arithmetic4{};
    std::array<double, kRounds> arithmetic8{};
    std::array<double, kRounds> math1{};
    std::array<double, kRounds> math4{};
    std::array<double, kRounds> math8{};
    double sinkArithmetic = 0.0;
    double sinkMath = 0.0;
    for (int r = 0; r < kRounds; ++r) {
        // Interleave every width of both composites round by round, so
        // frequency/thermal drift cannot favor one side.
        const auto i = static_cast<std::size_t>(r);
        arithmetic1[i] = TimeBatch(kIterations, sinkArithmetic, RunArithmetic<1>);
        arithmetic4[i] = TimeBatch(kIterations, sinkArithmetic, RunArithmetic<4>);
        arithmetic8[i] = TimeBatch(kIterations, sinkArithmetic, RunArithmetic<8>);
        math1[i] = TimeBatch(kIterations, sinkMath, RunMath<1>);
        math4[i] = TimeBatch(kIterations, sinkMath, RunMath<4>);
        math8[i] = TimeBatch(kIterations, sinkMath, RunMath<8>);
    }
    Report("arithmetic", arithmetic1, arithmetic4, arithmetic8, sinkArithmetic);
    Report("math", math1, math4, math8, sinkMath);
    return 0;
}
