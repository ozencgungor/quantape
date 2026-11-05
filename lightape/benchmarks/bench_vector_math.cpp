// SIMD vector-math bake-off harness (simd_vector_math_plan.md §7).
//
// Measures, per `v*` function and per compiled implementation (libm scalar,
// SLEEF explicit ISA plans, SLEEF generic scalar), the ns/value at W∈{1,4,8}
// plus the worst ULP against this host's scalar libm on a dense corpus, then
// prints a markdown table, a per-function winner line and a suggested
// `LIGHTAPE_MATH_IMPL_*` override map.
//
// Method: one 4096-value sweep per timed sample, warmup sweep before the
// rounds, R rounds with all implementations timed inside the same round
// (interleaved, so frequency/thermal drift hits every impl alike), median of
// the round samples. The wall-clock noise floor is ±4-5%, so only wins
// beyond that margin feed the suggested override map.
//
// Portability notes:
//   - The `libm` rows call `std::` scalar functions through the table, so
//     AOCL-LibM (or any libm replacement) can be bake-off'd without code
//     changes by LD_PRELOADing it on x86-64 Linux, e.g.
//     `LD_PRELOAD=/opt/aocl/lib/libaocl-libm.so ./lightape_vector_math_bench`.
//     If the preload also ships vector entry points, it can be wired later
//     as a candidate table behind the same `MathTable`.
//   - x86-64 AVX2/AVX-512 rows are only produced by running this binary on
//     the target CPUs (Zen 4/5, SPR/GNR); an Apple/ARM build cannot speak
//     for them. The per-ISA kernels live in compiled TUs with their own ISA
//     flags, so this binary runs the whole ladder on one host; the W=4 column
//     exercises the nearest-width AVX2/AVX2128 fallback of an AVX-512 part.
//   - macOS uses DYLD_INSERT_LIBRARIES for the same libm-preload trick.

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iterator>
#include <limits>
#include <string>
#include <vector>

#include "lightape/batch/VectorMath.h"

namespace {

using lightape::batch::autoChoice;
using lightape::batch::kMathFunctionNames;
using lightape::batch::MathFn;
using lightape::batch::MathImpl;
using lightape::batch::MathSelection;
using lightape::batch::MathTable;
using lightape::batch::VectorMath;

constexpr std::size_t kN = 4096;
constexpr int kRounds = 7;
constexpr std::size_t kWidths = 3;

double g_sink = 0.0;

/// NaN-vs-NaN counts as 0, Inf must match exactly, subnormal steps measured in
/// `denorm_min`: same convention as tests/test_vector_math.cpp.
double UlpDiff(double got, double ref) {
    if (std::isnan(ref)) {
        return std::isnan(got) ? 0.0 : std::numeric_limits<double>::infinity();
    }
    if (std::isnan(got)) {
        return std::numeric_limits<double>::infinity();
    }
    if (std::isinf(ref) || std::isinf(got)) {
        return got == ref ? 0.0 : std::numeric_limits<double>::infinity();
    }
    if (got == ref) {
        return 0.0;
    }
    const double mag = std::fabs(ref);
    double ulp = mag - std::nextafter(mag, 0.0);
    if (ulp == 0.0) {
        ulp = std::numeric_limits<double>::denorm_min();
    }
    return std::fabs(got - ref) / ulp;
}

// Inputs valid for the whole function set (positive nominal range; trig gets
// a wider sweep via the accuracy corpus, not the timing corpus).
std::vector<double> SpeedCorpusA() {
    std::vector<double> v(kN);
    for (std::size_t i = 0; i < kN; ++i) {
        v[i] = 0.3 + 0.001 * static_cast<double>((i * 37) % 1701); // [0.3, 2.0)
    }
    return v;
}

std::vector<double> SpeedCorpusB() {
    std::vector<double> v(kN);
    for (std::size_t i = 0; i < kN; ++i) {
        v[i] = 0.5 + 0.0005 * static_cast<double>((i * 53) % 1201); // [0.5, 1.1)
    }
    return v;
}

/// Dense accuracy corpus (domains, exponent sweep, specials), smaller than the
/// unit-test one but enough to catch tier violations.
std::vector<double> AccuracyCorpusA() {
    std::vector<double> c;
    const double ranges[][3] = {
        {-20.0, 20.0, 4001.0}, {-1.0, 1.0, 2001.0}, {0.0, 1e-6, 501.0},
        {-1e-6, 0.0, 501.0},   {1.0, 2.0, 1001.0},  {1e-300, 1e-290, 101.0},
    };
    for (const auto& r : ranges) {
        const std::size_t n = static_cast<std::size_t>(r[2]);
        for (std::size_t i = 0; i < n; ++i) {
            c.push_back(r[0] + (r[1] - r[0]) * static_cast<double>(i) / static_cast<double>(n - 1));
        }
    }
    for (int e = -1074; e <= 1023; e += 13) {
        const double v = std::ldexp(1.0, e);
        c.push_back(v);
        c.push_back(-v);
    }
    const double specials[] = {
        0.0,
        -0.0,
        std::numeric_limits<double>::denorm_min(),
        -std::numeric_limits<double>::denorm_min(),
        std::numeric_limits<double>::min(),
        -std::numeric_limits<double>::min(),
        std::numeric_limits<double>::max(),
        -std::numeric_limits<double>::max(),
        std::numeric_limits<double>::infinity(),
        -std::numeric_limits<double>::infinity(),
        std::numeric_limits<double>::quiet_NaN(),
        1.0,
        -1.0,
        0.5,
        -0.5,
        2.0,
        -2.0,
        1e16,
        -1e16,
    };
    for (double v : specials) {
        c.push_back(v);
    }
    return c;
}

std::vector<double> AccuracyCorpusB() {
    const std::vector<double>& a = AccuracyCorpusA();
    std::vector<double> b(a.size());
    for (std::size_t i = 0; i < a.size(); ++i) {
        b[i] = 0.5 * a[(i * 7919 + 13) % a.size()];
    }
    return b;
}

// ── evaluation runners: W lanes per `VectorMath<W>` call, n values total ──

using UnaryRun = void (*)(const double*, double*, std::size_t);
using BinaryRun = void (*)(const double*, const double*, double*, std::size_t);
using SinCosRun = void (*)(const double*, double*, double*, std::size_t);

// X(field, VectorMath method): `field` is the MathFn/std name, `Method` the
// `VectorMath<W>` member spelling.
#define LIGHTAPE_BENCH_UNARY_FUNCTIONS(X)                                                          \
    X(sqrt, vSqrt)                                                                                 \
    X(exp, vExp)                                                                                   \
    X(log, vLog)                                                                                   \
    X(expm1, vExpm1)                                                                               \
    X(log1p, vLog1p)                                                                               \
    X(sin, vSin)                                                                                   \
    X(cos, vCos)                                                                                   \
    X(tan, vTan)                                                                                   \
    X(asin, vAsin)                                                                                 \
    X(acos, vAcos)                                                                                 \
    X(atan, vAtan)                                                                                 \
    X(sinh, vSinh)                                                                                 \
    X(cosh, vCosh)                                                                                 \
    X(tanh, vTanh)                                                                                 \
    X(exp2, vExp2)                                                                                 \
    X(log2, vLog2)                                                                                 \
    X(log10, vLog10)                                                                               \
    X(erf, vErf)                                                                                   \
    X(erfc, vErfc)                                                                                 \
    X(cbrt, vCbrt)                                                                                 \
    X(normPdf, vNormPdf)                                                                           \
    X(normCdf, vNormCdf)
#define LIGHTAPE_BENCH_BINARY_FUNCTIONS(X) X(pow, vPow) X(atan2, vAtan2) X(hypot, vHypot)
#define LIGHTAPE_BENCH_PLAIN_UNARY_FUNCTIONS(X)                                                    \
    X(sqrt, vSqrt)                                                                                 \
    X(exp, vExp)                                                                                   \
    X(log, vLog)                                                                                   \
    X(expm1, vExpm1)                                                                               \
    X(log1p, vLog1p)                                                                               \
    X(sin, vSin)                                                                                   \
    X(cos, vCos)                                                                                   \
    X(tan, vTan)                                                                                   \
    X(asin, vAsin)                                                                                 \
    X(acos, vAcos)                                                                                 \
    X(atan, vAtan)                                                                                 \
    X(sinh, vSinh)                                                                                 \
    X(cosh, vCosh)                                                                                 \
    X(tanh, vTanh)                                                                                 \
    X(exp2, vExp2)                                                                                 \
    X(log2, vLog2)                                                                                 \
    X(log10, vLog10)                                                                               \
    X(erf, vErf)                                                                                   \
    X(erfc, vErfc)                                                                                 \
    X(cbrt, vCbrt)

template <std::size_t W>
struct WidthRuns {
// Trailing values overlap the last full W block (same convention as
// tests/test_vector_math.cpp), so no lane is left unwritten for n % W != 0.
#define LIGHTAPE_BENCH_DEF_UNARY(field, Method)                                                    \
    static void field(const double* in, double* out, std::size_t n) {                              \
        std::size_t i = 0;                                                                         \
        for (; i + W <= n; i += W) {                                                               \
            VectorMath<W>::Method(in + i, out + i);                                                \
        }                                                                                          \
        if (i < n && n >= W) {                                                                     \
            VectorMath<W>::Method(in + n - W, out + n - W);                                        \
        }                                                                                          \
    }
    LIGHTAPE_BENCH_UNARY_FUNCTIONS(LIGHTAPE_BENCH_DEF_UNARY)
#undef LIGHTAPE_BENCH_DEF_UNARY
#define LIGHTAPE_BENCH_DEF_BINARY(field, Method)                                                   \
    static void field(const double* a, const double* b, double* out, std::size_t n) {              \
        std::size_t i = 0;                                                                         \
        for (; i + W <= n; i += W) {                                                               \
            VectorMath<W>::Method(a + i, b + i, out + i);                                          \
        }                                                                                          \
        if (i < n && n >= W) {                                                                     \
            VectorMath<W>::Method(a + n - W, b + n - W, out + n - W);                              \
        }                                                                                          \
    }
    LIGHTAPE_BENCH_BINARY_FUNCTIONS(LIGHTAPE_BENCH_DEF_BINARY)
#undef LIGHTAPE_BENCH_DEF_BINARY
    static void sincos(const double* in, double* s, double* c, std::size_t n) {
        std::size_t i = 0;
        for (; i + W <= n; i += W) {
            VectorMath<W>::vSinCos(in + i, s + i, c + i);
        }
        if (i < n && n >= W) {
            VectorMath<W>::vSinCos(in + n - W, s + n - W, c + n - W);
        }
    }
};

/// Hand-written direct scalar loops (no dispatch table) for the
/// dispatch-overhead reference.
struct DirectRuns {
#define LIGHTAPE_BENCH_DEF_DIRECT_UNARY(field, Method)                                             \
    static void field(const double* in, double* out, std::size_t n) {                              \
        using std::field;                                                                          \
        for (std::size_t i = 0; i < n; ++i) {                                                      \
            out[i] = field(in[i]);                                                                 \
        }                                                                                          \
    }
    LIGHTAPE_BENCH_PLAIN_UNARY_FUNCTIONS(LIGHTAPE_BENCH_DEF_DIRECT_UNARY)
#undef LIGHTAPE_BENCH_DEF_DIRECT_UNARY
    static void normPdf(const double* in, double* out, std::size_t n) {
        using std::exp;
        for (std::size_t i = 0; i < n; ++i) {
            out[i] = lightape::batch::detail::kInvSqrt2Pi * exp(-0.5 * (in[i] * in[i]));
        }
    }
    static void normCdf(const double* in, double* out, std::size_t n) {
        using std::erfc;
        for (std::size_t i = 0; i < n; ++i) {
            out[i] = 0.5 * erfc(-lightape::batch::detail::kInvSqrt2 * in[i]);
        }
    }
#define LIGHTAPE_BENCH_DEF_DIRECT_BINARY(field, Method)                                            \
    static void field(const double* a, const double* b, double* out, std::size_t n) {              \
        using std::field;                                                                          \
        for (std::size_t i = 0; i < n; ++i) {                                                      \
            out[i] = field(a[i], b[i]);                                                            \
        }                                                                                          \
    }
    LIGHTAPE_BENCH_BINARY_FUNCTIONS(LIGHTAPE_BENCH_DEF_DIRECT_BINARY)
#undef LIGHTAPE_BENCH_DEF_DIRECT_BINARY
    static void sincos(const double* in, double* s, double* c, std::size_t n) {
        using std::cos;
        using std::sin;
        for (std::size_t i = 0; i < n; ++i) {
            s[i] = sin(in[i]);
            c[i] = cos(in[i]);
        }
    }
};

// ── case tables ──

struct UnaryCase {
    MathFn fn;
    UnaryRun run[kWidths];
    UnaryRun direct;
};

struct BinaryCase {
    MathFn fn;
    BinaryRun run[kWidths];
    BinaryRun direct;
};

constexpr const char* CaseName(MathFn fn) {
    return kMathFunctionNames[static_cast<std::size_t>(fn)];
}

#define LIGHTAPE_BENCH_UNARY_ROW(field)                                                            \
    {MathFn::field,                                                                                \
     {&WidthRuns<1>::field, &WidthRuns<4>::field, &WidthRuns<8>::field},                           \
     &DirectRuns::field}
#define LIGHTAPE_BENCH_BINARY_ROW(field)                                                           \
    {MathFn::field,                                                                                \
     {&WidthRuns<1>::field, &WidthRuns<4>::field, &WidthRuns<8>::field},                           \
     &DirectRuns::field}

constexpr UnaryCase kUnaryCases[] = {
    LIGHTAPE_BENCH_UNARY_ROW(sqrt),
    LIGHTAPE_BENCH_UNARY_ROW(exp),
    LIGHTAPE_BENCH_UNARY_ROW(log),
    LIGHTAPE_BENCH_UNARY_ROW(expm1),
    LIGHTAPE_BENCH_UNARY_ROW(log1p),
    LIGHTAPE_BENCH_UNARY_ROW(sin),
    LIGHTAPE_BENCH_UNARY_ROW(cos),
    LIGHTAPE_BENCH_UNARY_ROW(tan),
    LIGHTAPE_BENCH_UNARY_ROW(asin),
    LIGHTAPE_BENCH_UNARY_ROW(acos),
    LIGHTAPE_BENCH_UNARY_ROW(atan),
    LIGHTAPE_BENCH_UNARY_ROW(sinh),
    LIGHTAPE_BENCH_UNARY_ROW(cosh),
    LIGHTAPE_BENCH_UNARY_ROW(tanh),
    LIGHTAPE_BENCH_UNARY_ROW(exp2),
    LIGHTAPE_BENCH_UNARY_ROW(log2),
    LIGHTAPE_BENCH_UNARY_ROW(log10),
    LIGHTAPE_BENCH_UNARY_ROW(erf),
    LIGHTAPE_BENCH_UNARY_ROW(erfc),
    LIGHTAPE_BENCH_UNARY_ROW(cbrt),
    {MathFn::normPdf,
     {&WidthRuns<1>::normPdf, &WidthRuns<4>::normPdf, &WidthRuns<8>::normPdf},
     &DirectRuns::normPdf},
    {MathFn::normCdf,
     {&WidthRuns<1>::normCdf, &WidthRuns<4>::normCdf, &WidthRuns<8>::normCdf},
     &DirectRuns::normCdf},
};
constexpr BinaryCase kBinaryCases[] = {
    LIGHTAPE_BENCH_BINARY_ROW(pow),
    LIGHTAPE_BENCH_BINARY_ROW(atan2),
    LIGHTAPE_BENCH_BINARY_ROW(hypot),
};

const SinCosRun kSinCosRuns[kWidths] = {&WidthRuns<1>::sincos, &WidthRuns<4>::sincos,
                                        &WidthRuns<8>::sincos};

// ── implementation variants ──

struct Variant {
    std::string label; // e.g. "libm", "sleef:advsimd", "generic"
    MathTable table;
    bool selectable;   // reachable via LIGHTAPE_MATH_IMPL_<vFn>
    bool isDefaultIsa; // the plan `=sleef` resolves to

    const char* implValue() const {
        if (std::strcmp(table.sqrt.impl, "libm") == 0) {
            return "libm";
        }
        return std::strcmp(table.sqrt.isa, "generic") == 0 ? "generic" : "sleef";
    }
};

std::vector<Variant> BuildVariants() {
    std::vector<Variant> out;
    const auto push = [&](MathTable table, bool selectable, bool isDefaultIsa) {
        for (const Variant& v : out) {
            if (std::strcmp(v.table.sqrt.isa, table.sqrt.isa) == 0) {
                return; // dedupe by resolved ISA
            }
        }
        Variant v;
        if (std::strcmp(table.sqrt.isa, "libm") == 0) {
            v.label = "libm";
        } else {
            v.label = std::string("sleef:") + table.sqrt.isa;
        }
        v.table = table;
        v.selectable = selectable;
        v.isDefaultIsa = isDefaultIsa;
        out.push_back(std::move(v));
    };
    push(lightape::batch::resolveMathTable(MathSelection{.table = MathImpl::Libm}), true, false);
#ifdef LIGHTAPE_VECTOR_MATH_SLEEF
    // Default plan (whatever `sleef` resolves to on this CPU): selectable.
    push(lightape::batch::resolveMathTable(MathSelection{.table = MathImpl::Sleef}), true, true);
    // Explicit ISA plans, for the bake-off record; only `generic` is a
    // per-function override value of its own.
    const char* candidates[] = {"advsimd", "avx512f", "avx2", "avx2128", "generic"};
    for (const char* isa : candidates) {
        ::setenv("LIGHTAPE_VECTOR_ISA", isa, 1);
        MathTable table =
            lightape::batch::resolveMathTable(MathSelection{.table = MathImpl::Sleef});
        const bool generic = std::strcmp(table.sqrt.isa, "generic") == 0;
        push(std::move(table), generic, false);
    }
    ::unsetenv("LIGHTAPE_VECTOR_ISA");
#endif
    return out;
}

void Install(const Variant& variant) {
    lightape::batch::setMathTableForTesting(&variant.table);
}

double Median(std::array<double, kRounds> samples) {
    std::sort(samples.begin(), samples.end());
    return samples[kRounds / 2];
}

struct VariantTiming {
    std::array<double, kWidths> ns{}; // median ns/value per W
    double ulp = 0.0;                 // worst vs direct libm, W=8
};

// ── measurement ──

std::vector<std::vector<std::array<double, kWidths>>> g_unaryNs;
std::vector<std::vector<double>> g_unaryUlp;
std::vector<std::array<double, kWidths>> g_unaryDirectNs;
std::vector<std::array<double, kWidths>> g_binaryDirectNs;
double g_sincosDirectNs = 0.0;
std::vector<std::array<double, kWidths>> g_sincosNs;
std::vector<double> g_sincosUlp;
std::vector<std::vector<std::array<double, kWidths>>> g_binaryNs;
std::vector<std::vector<double>> g_binaryUlp;

void BenchUnary(const std::vector<Variant>& variants, const std::vector<double>& speed,
                const std::vector<double>& acc) {
    g_unaryNs.assign(std::size(kUnaryCases), {});
    g_unaryUlp.assign(std::size(kUnaryCases), {});
    g_unaryDirectNs.assign(std::size(kUnaryCases), {});
    std::vector<double> out(kN);
    std::vector<double> accOut(acc.size());
    std::vector<double> accRef(acc.size());

    for (std::size_t c = 0; c < std::size(kUnaryCases); ++c) {
        const UnaryCase& kase = kUnaryCases[c];
        // Warmup (discarded) for every variant/width.
        for (const Variant& v : variants) {
            Install(v);
            for (std::size_t w = 0; w < kWidths; ++w) {
                kase.run[w](speed.data(), out.data(), kN);
            }
        }
        std::vector<std::array<std::array<double, kRounds>, kWidths>> samples(variants.size());
        std::array<double, kRounds> directSamples{};
        for (int r = 0; r < kRounds; ++r) {
            for (std::size_t v = 0; v < variants.size(); ++v) {
                Install(variants[v]);
                for (std::size_t w = 0; w < kWidths; ++w) {
                    const auto t0 = std::chrono::steady_clock::now();
                    kase.run[w](speed.data(), out.data(), kN);
                    const auto t1 = std::chrono::steady_clock::now();
                    samples[v][w][r] =
                        std::chrono::duration<double, std::nano>(t1 - t0).count() / kN;
                }
            }
            // Dispatch reference: direct scalar loop (W-independent),
            // measured inside the same round so drift hits it too.
            const auto td0 = std::chrono::steady_clock::now();
            kase.direct(speed.data(), out.data(), kN);
            const auto td1 = std::chrono::steady_clock::now();
            directSamples[static_cast<std::size_t>(r)] =
                std::chrono::duration<double, std::nano>(td1 - td0).count() / kN;
        }
        std::vector<std::array<double, kWidths>> medians(variants.size());
        for (std::size_t v = 0; v < variants.size(); ++v) {
            for (std::size_t w = 0; w < kWidths; ++w) {
                medians[v][w] = Median(samples[v][w]);
            }
        }
        g_sink += out[c % kN];
        g_unaryDirectNs[c].fill(Median(directSamples));
        g_sink += out[0];

        // Accuracy (W=8) against the direct scalar reference.
        kase.direct(acc.data(), accRef.data(), acc.size());
        for (std::size_t v = 0; v < variants.size(); ++v) {
            Install(variants[v]);
            kase.run[2](acc.data(), accOut.data(), acc.size());
            double worst = 0.0;
            for (std::size_t i = 0; i < acc.size(); ++i) {
                worst = std::max(worst, UlpDiff(accOut[i], accRef[i]));
            }
            g_unaryUlp[c].push_back(worst);
        }
        g_unaryNs[c] = std::move(medians);
    }
}

void BenchBinary(const std::vector<Variant>& variants, const std::vector<double>& speedA,
                 const std::vector<double>& speedB, const std::vector<double>& accA,
                 const std::vector<double>& accB) {
    g_binaryNs.assign(std::size(kBinaryCases), {});
    g_binaryUlp.assign(std::size(kBinaryCases), {});
    g_binaryDirectNs.assign(std::size(kBinaryCases), {});
    std::vector<double> out(kN);
    std::vector<double> accOut(accA.size());
    std::vector<double> accRef(accA.size());

    for (std::size_t c = 0; c < std::size(kBinaryCases); ++c) {
        const BinaryCase& kase = kBinaryCases[c];
        for (const Variant& v : variants) {
            Install(v);
            for (std::size_t w = 0; w < kWidths; ++w) {
                kase.run[w](speedA.data(), speedB.data(), out.data(), kN);
            }
        }
        std::vector<std::array<std::array<double, kRounds>, kWidths>> samples(variants.size());
        std::array<double, kRounds> directSamples{};
        for (int r = 0; r < kRounds; ++r) {
            for (std::size_t v = 0; v < variants.size(); ++v) {
                Install(variants[v]);
                for (std::size_t w = 0; w < kWidths; ++w) {
                    const auto t0 = std::chrono::steady_clock::now();
                    kase.run[w](speedA.data(), speedB.data(), out.data(), kN);
                    const auto t1 = std::chrono::steady_clock::now();
                    samples[v][w][r] =
                        std::chrono::duration<double, std::nano>(t1 - t0).count() / kN;
                }
            }
            const auto td0 = std::chrono::steady_clock::now();
            kase.direct(speedA.data(), speedB.data(), out.data(), kN);
            const auto td1 = std::chrono::steady_clock::now();
            directSamples[static_cast<std::size_t>(r)] =
                std::chrono::duration<double, std::nano>(td1 - td0).count() / kN;
        }
        std::vector<std::array<double, kWidths>> medians(variants.size());
        for (std::size_t v = 0; v < variants.size(); ++v) {
            for (std::size_t w = 0; w < kWidths; ++w) {
                medians[v][w] = Median(samples[v][w]);
            }
        }
        g_binaryDirectNs[c].fill(Median(directSamples));
        g_sink += out[0];

        kase.direct(accA.data(), accB.data(), accRef.data(), accA.size());
        for (std::size_t v = 0; v < variants.size(); ++v) {
            Install(variants[v]);
            kase.run[2](accA.data(), accB.data(), accOut.data(), accA.size());
            double worst = 0.0;
            for (std::size_t i = 0; i < accA.size(); ++i) {
                worst = std::max(worst, UlpDiff(accOut[i], accRef[i]));
            }
            g_binaryUlp[c].push_back(worst);
        }
        g_binaryNs[c] = std::move(medians);
    }
}

void BenchSinCos(const std::vector<Variant>& variants, const std::vector<double>& speed,
                 const std::vector<double>& acc) {
    g_sincosNs.assign(variants.size(), {});
    g_sincosUlp.assign(variants.size(), 0.0);
    std::vector<double> s(kN);
    std::vector<double> c(kN);
    std::vector<double> accS(acc.size());
    std::vector<double> accC(acc.size());
    std::vector<double> refS(acc.size());
    std::vector<double> refC(acc.size());

    for (const Variant& v : variants) {
        Install(v);
        for (std::size_t w = 0; w < kWidths; ++w) {
            kSinCosRuns[w](speed.data(), s.data(), c.data(), kN);
        }
    }
    std::vector<std::array<std::array<double, kRounds>, kWidths>> samples(variants.size());
    std::array<double, kRounds> directSamples{};
    for (int r = 0; r < kRounds; ++r) {
        for (std::size_t v = 0; v < variants.size(); ++v) {
            Install(variants[v]);
            for (std::size_t w = 0; w < kWidths; ++w) {
                const auto t0 = std::chrono::steady_clock::now();
                kSinCosRuns[w](speed.data(), s.data(), c.data(), kN);
                const auto t1 = std::chrono::steady_clock::now();
                samples[v][w][r] = std::chrono::duration<double, std::nano>(t1 - t0).count() / kN;
            }
        }
        const auto td0 = std::chrono::steady_clock::now();
        DirectRuns::sincos(speed.data(), s.data(), c.data(), kN);
        const auto td1 = std::chrono::steady_clock::now();
        directSamples[static_cast<std::size_t>(r)] =
            std::chrono::duration<double, std::nano>(td1 - td0).count() / kN;
    }
    for (std::size_t v = 0; v < variants.size(); ++v) {
        for (std::size_t w = 0; w < kWidths; ++w) {
            g_sincosNs[v][w] = Median(samples[v][w]);
        }
    }
    g_sincosDirectNs = Median(directSamples);
    g_sink += s[0] + c[0];

    DirectRuns::sincos(acc.data(), refS.data(), refC.data(), acc.size());
    for (std::size_t v = 0; v < variants.size(); ++v) {
        Install(variants[v]);
        kSinCosRuns[2](acc.data(), accS.data(), accC.data(), acc.size());
        double worst = 0.0;
        for (std::size_t i = 0; i < acc.size(); ++i) {
            worst = std::max(worst, UlpDiff(accS[i], refS[i]));
            worst = std::max(worst, UlpDiff(accC[i], refC[i]));
        }
        g_sincosUlp[v] = worst;
    }
}

// ── reporting ──

void PrintHeader(const std::vector<Variant>& variants) {
    std::printf("# SIMD vector-math bake-off\n\n");
    std::printf("- host: %s (%s)\n",
#if defined(__aarch64__)
                "aarch64",
#elif defined(__x86_64__)
                "x86-64",
#else
                "unknown-arch",
#endif
                "compile time on this machine");
    std::printf("- build: %s\n",
#if defined(LIGHTAPE_VECTOR_MATH_SLEEF)
                "LIGHTAPE_VECTOR_MATH=sleef"
#else
                "LIGHTAPE_VECTOR_MATH=libm (SLEEF not compiled)"
#endif
    );
    std::printf("- samples: %zu values/sweep, %d interleaved rounds, median, warmup discarded\n",
                kN, kRounds);
    std::printf("- reference: direct scalar `std::` loop (libm); ULP column is worst vs it\n");
    std::printf("- variants:");
    for (const Variant& v : variants) {
        std::printf(" %s%s", v.label.c_str(), v.selectable ? "" : "(info)");
    }
    std::printf("\n\n");
}

void PrintUnaryTable(const std::vector<Variant>& variants) {
    for (std::size_t v = 0; v < variants.size(); ++v) {
        std::printf("### %s\n\n", variants[v].label.c_str());
        std::printf("| function | W=1 ns | W=4 ns | W=8 ns | worst ULP | direct ns |\n");
        std::printf("|---|---:|---:|---:|---:|---:|\n");
        for (std::size_t c = 0; c < std::size(kUnaryCases); ++c) {
            const auto& ns = g_unaryNs[c][v];
            std::printf("| %s | %.3f | %.3f | %.3f | %.2f | %.3f |\n", CaseName(kUnaryCases[c].fn),
                        ns[0], ns[1], ns[2], g_unaryUlp[c][v], g_unaryDirectNs[c][0]);
        }
        std::printf("\n");
    }
}

void PrintBinaryTable(const std::vector<Variant>& variants) {
    for (std::size_t v = 0; v < variants.size(); ++v) {
        std::printf("### %s (binary)\n\n", variants[v].label.c_str());
        std::printf("| function | W=1 ns | W=4 ns | W=8 ns | worst ULP | direct ns |\n");
        std::printf("|---|---:|---:|---:|---:|---:|\n");
        for (std::size_t c = 0; c < std::size(kBinaryCases); ++c) {
            const auto& ns = g_binaryNs[c][v];
            std::printf("| %s | %.3f | %.3f | %.3f | %.2f | %.3f |\n", CaseName(kBinaryCases[c].fn),
                        ns[0], ns[1], ns[2], g_binaryUlp[c][v], g_binaryDirectNs[c][0]);
        }
        std::printf("\n");
    }
}

void PrintSinCosTable(const std::vector<Variant>& variants) {
    std::printf("### vSinCos\n\n");
    std::printf("| impl | W=1 ns | W=4 ns | W=8 ns | worst ULP | direct ns |\n");
    std::printf("|---|---:|---:|---:|---:|---:|\n");
    for (std::size_t v = 0; v < variants.size(); ++v) {
        const auto& ns = g_sincosNs[v];
        std::printf("| %s | %.3f | %.3f | %.3f | %.2f | %.3f |\n", variants[v].label.c_str(), ns[0],
                    ns[1], ns[2], g_sincosUlp[v], g_sincosDirectNs);
    }
    std::printf("\n");
}

/// W=8 is the decision row: per-function overrides select an implementation,
/// not a width, and at W=1 a selected vector plan degrades to the scalar tail
/// (fast, but not evidence about the vector kernels). Comparing at W=8 keeps
/// the winner/suggestion honest for the deployment batch width.
constexpr std::size_t kDecision = 2;

const Variant* FindLibm(const std::vector<Variant>& variants) {
    for (const Variant& v : variants) {
        if (std::strcmp(v.table.sqrt.impl, "libm") == 0) {
            return &v;
        }
    }
    return nullptr;
}

const Variant* FindDefaultSleef(const std::vector<Variant>& variants) {
    for (const Variant& v : variants) {
        if (v.isDefaultIsa) {
            return &v;
        }
    }
    return nullptr;
}

void PrintWinner(const std::vector<Variant>& variants, MathFn fn,
                 const std::vector<std::array<double, kWidths>>& ns) {
    const Variant* best = nullptr;
    double bestNs = std::numeric_limits<double>::infinity();
    for (std::size_t v = 0; v < variants.size(); ++v) {
        if (ns[v][kDecision] < bestNs) {
            bestNs = ns[v][kDecision];
            best = &variants[v];
        }
    }
    if (best != nullptr) {
        std::printf("- %s: %s W=8 %.3f ns/value\n", CaseName(fn), best->label.c_str(), bestNs);
    }
}

/// Suggested per-function override map: where a selectable candidate beats
/// the platform `auto` default at W=8 by more than the ±4-5% noise floor.
void SuggestOne(const std::vector<Variant>& variants, MathFn fn,
                const std::vector<std::array<double, kWidths>>& ns, bool& any) {
    const Variant* libm = FindLibm(variants);
    const Variant* defaultSleef = FindDefaultSleef(variants);
    const MathImpl policy = autoChoice(fn);
    const Variant* autoVariant = policy == MathImpl::Libm ? libm : defaultSleef;
    if (autoVariant == nullptr) {
        return;
    }
    const std::size_t autoIdx = static_cast<std::size_t>(autoVariant - variants.data());
    const double autoNs = ns[autoIdx][kDecision];
    const Variant* best = nullptr;
    double bestNs = autoNs;
    for (std::size_t v = 0; v < variants.size(); ++v) {
        if (!variants[v].selectable) {
            continue;
        }
        if (ns[v][kDecision] < bestNs) {
            bestNs = ns[v][kDecision];
            best = &variants[v];
        }
    }
    if (best != nullptr && bestNs <= 0.95 * autoNs) {
        std::printf("LIGHTAPE_MATH_IMPL_%s=%s\n", CaseName(fn), best->implValue());
        any = true;
    }
}

void PrintWinners(const std::vector<Variant>& variants) {
    std::printf("### Winners at W=8 (all variants incl. informational ISAs)\n\n");
    for (std::size_t c = 0; c < std::size(kUnaryCases); ++c) {
        PrintWinner(variants, kUnaryCases[c].fn, g_unaryNs[c]);
    }
    for (std::size_t c = 0; c < std::size(kBinaryCases); ++c) {
        PrintWinner(variants, kBinaryCases[c].fn, g_binaryNs[c]);
    }
    PrintWinner(variants, MathFn::sincos, g_sincosNs);
    std::printf("\n");
}

void PrintSuggestions(const std::vector<Variant>& variants) {
    std::printf("### Suggested `LIGHTAPE_MATH_IMPL_*` overrides\n\n");
    std::printf("(only where a selectable impl beats the auto default at W=8 by >5%%)\n\n");
    bool any = false;
    for (std::size_t c = 0; c < std::size(kUnaryCases); ++c) {
        SuggestOne(variants, kUnaryCases[c].fn, g_unaryNs[c], any);
    }
    for (std::size_t c = 0; c < std::size(kBinaryCases); ++c) {
        SuggestOne(variants, kBinaryCases[c].fn, g_binaryNs[c], any);
    }
    SuggestOne(variants, MathFn::sincos, g_sincosNs, any);
    if (!any) {
        std::printf("(none: the auto default wins or is within the 5%% noise floor)\n");
    }
    std::printf("\n");
}

void PrintNotes() {
    std::printf("### Notes\n\n");
    std::printf("- x86-64 AVX2/AVX-512 rows must be produced on the target CPUs; an ARM build "
                "cannot speak for them.\n");
    std::printf(
        "- The libm rows call `std::` through the table, so AOCL-LibM can be bake-off'd "
        "without code changes: `LD_PRELOAD=/path/libaocl-libm.so ./lightape_vector_math_bench` "
        "(macOS: DYLD_INSERT_LIBRARIES).\n");
    std::printf("- The direct-ns column is the no-dispatch scalar loop; V1 measured the naive "
                "width-1 table dispatch at 17-26%% slower (V3 hoists/unrolls).\n");
    std::printf("- W=4 picks the widest SLEEF kernel that fits (avx2 on an AVX-512 part), so "
                "the W=4 column is a real vector width, not the scalar tail.\n");
    std::printf("- x86 `auto` follows the 2026-10 AWS bake-off: SLEEF for every function "
                "except vErfc, which stays libm; vNormCdf's SLEEF candidate is the fused "
                "erf-central kernel, not the slow u15 erfc (see VectorMath.h).\n");
    std::printf("- vNormCdf's worst-ULP column is not a kernel-accuracy number: the fused "
                "central form cancels toward ~1e-9 at |x| = 6, so ULP-vs-libm is huge while the "
                "gated error is absolute <= 1e-15 central / relative <= 1e-12 in the tails "
                "(tests/test_vector_math.cpp).\n");
    std::printf("- M2 Pro evidence: the V1-style raw probe shows Apple libm winning/tying "
                "every function (no V1 log win reproduced); this broader corpus measures a few "
                "AdvSIMD wins, and the fused vNormCdf is the one in the AArch64 auto list. "
                "Extend it only from a fresh run on the deployment workload (see "
                "VectorMath.h).\n");
}

} // namespace

int main() {
    const std::vector<Variant> variants = BuildVariants();
    const std::vector<double> speedA = SpeedCorpusA();
    const std::vector<double> speedB = SpeedCorpusB();
    const std::vector<double> accA = AccuracyCorpusA();
    const std::vector<double> accB = AccuracyCorpusB();

    BenchUnary(variants, speedA, accA);
    BenchBinary(variants, speedA, speedB, accA, accB);
    BenchSinCos(variants, speedA, accA);

    PrintHeader(variants);
    PrintUnaryTable(variants);
    PrintBinaryTable(variants);
    PrintSinCosTable(variants);
    PrintWinners(variants);
    PrintSuggestions(variants);
    PrintNotes();

    std::printf("sink=%.6g\n", g_sink);
    lightape::batch::setMathTableForTesting(nullptr);
    return 0;
}
