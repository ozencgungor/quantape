#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <string>
#include <utility>
#include <vector>

#include "lightape/batch/VectorMath.h"
#include "lightape/ops/Probability.h"
#include <gtest/gtest.h>

namespace {

using lightape::batch::kMathFunctionCount;
using lightape::batch::kMathFunctionNames;
using lightape::batch::MathBackend;
using lightape::batch::MathFn;
using lightape::batch::MathImpl;
using lightape::batch::MathSelection;
using lightape::batch::MathTable;
using lightape::batch::VectorMath;

/// Raw IEEE-754 bit pattern (plain `==` would hide signed-zero/NaN payloads).
std::uint64_t Bits(double v) {
    return std::bit_cast<std::uint64_t>(v);
}

/// `true` when the resolved entry is the LibmScalar reference, so the bitwise
/// gate applies; SLEEF/generic entries fall back to the documented ULP gate.
template <typename Entry>
bool IsLibmEntry(const Entry& entry) {
    return std::strcmp(entry.impl, "libm") == 0;
}

/// ULP distance used for the differential gate. NaN vs NaN counts as 0 (the
/// plan only forbids NaN/Inf *behaviour* divergence, not payload/sign), Inf
/// must match exactly, and near zero the step size is `denorm_min`, so
/// underflow to a subnormal is measured in representable doubles.
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

/// Dense evaluation corpus: intervals, a power-of-two sweep across the whole
/// exponent range (subnormals to max), one-ulp neighbours, and specials
/// (+-0, +-denorm_min, +-min/max, +-inf, NaN). Deliberately larger than any
/// `W` so the lane tails of every plan are exercised as well.
std::vector<double> BuildCorpus() {
    std::vector<double> c;
    const double ranges[][3] = {
        {-20.0, 20.0, 20001.0}, {-1.0, 1.0, 4001.0},      {0.0, 1e-6, 2001.0},
        {-1e-6, 0.0, 2001.0},   {1e-300, 1e-290, 501.0},  {-2.0, -1.0, 2001.0},
        {1.0, 2.0, 2001.0},     {1e6, 1e6 + 1.0, 2001.0}, {-1e6 - 1.0, -1e6, 2001.0},
    };
    for (const auto& r : ranges) {
        const std::size_t n = static_cast<std::size_t>(r[2]);
        for (std::size_t i = 0; i < n; ++i) {
            c.push_back(r[0] + (r[1] - r[0]) * static_cast<double>(i) / static_cast<double>(n - 1));
        }
    }
    for (int e = -1074; e <= 1023; e += 7) {
        const double v = std::ldexp(1.0, e);
        c.push_back(v);
        c.push_back(-v);
        c.push_back(std::nextafter(v, std::numeric_limits<double>::infinity()));
        c.push_back(-std::nextafter(v, std::numeric_limits<double>::infinity()));
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
        -std::numeric_limits<double>::quiet_NaN(),
        1.0,
        -1.0,
        0.5,
        -0.5,
        2.0,
        -2.0,
        3.0,
        -3.0,
        1e16,
        -1e16,
        1e300,
        -1e300,
    };
    for (double v : specials) {
        c.push_back(v);
    }
    return c;
}

const std::vector<double>& Corpus() {
    static const std::vector<double> c = BuildCorpus();
    return c;
}

/// Second operand of the binary kernels: a scrambled, halved copy of the
/// corpus, so domain edges (negative bases, zero exponents, huge/small)
/// recur in the pairs.
std::vector<double> BuildCorpusB() {
    const std::vector<double>& a = Corpus();
    std::vector<double> b(a.size());
    for (std::size_t i = 0; i < a.size(); ++i) {
        b[i] = 0.5 * a[(i * 7919 + 13) % a.size()];
    }
    return b;
}

const std::vector<double>& CorpusB() {
    static const std::vector<double> c = BuildCorpusB();
    return c;
}

/// Per-function record of the worst differential seen across all widths.
struct Stat {
    std::string name;
    double worstUlp = 0.0;
    double worstIn = 0.0;
    double worstGot = 0.0;
    double worstRef = 0.0;
};

std::vector<Stat>& Stats() {
    static std::vector<Stat> s;
    return s;
}

void NoteStat(const char* name, double worst, double in, double got, double ref) {
    for (Stat& s : Stats()) {
        if (s.name == name) {
            if (!(worst <= s.worstUlp)) {
                s.worstUlp = worst;
                s.worstIn = in;
                s.worstGot = got;
                s.worstRef = ref;
            }
            return;
        }
    }
    Stats().push_back(Stat{name, worst, in, got, ref});
}

// ── 64-bit FNV-1a over the bit patterns of every evaluated output ──

constexpr std::uint64_t kFnvOffset = 14695981039346656037ull;
constexpr std::uint64_t kFnvPrime = 1099511628211ull;

std::uint64_t g_fingerprint = kFnvOffset;

void MixBits(double v) {
    std::uint64_t b = Bits(v);
    for (int k = 0; k < 8; ++k) {
        g_fingerprint ^= b & 0xffu;
        g_fingerprint *= kFnvPrime;
        b >>= 8;
    }
}

/// The build flavor (`sleef` compiles the SLEEF candidate tables, `libm`
/// falls every non-libm request back to the scalar reference).
const char* BuildFlavor() {
#if defined(LIGHTAPE_VECTOR_MATH_SLEEF)
    return "sleef";
#else
    return "libm";
#endif
}

/// Summary name of the resolved table ("libm", an ISA, "generic" or "mixed").
const char* TableName() {
    return lightape::batch::mathTable().name;
}

std::size_t PlanWidth() {
    return lightape::batch::mathTable().width;
}

void ForEachWidth(auto&& f) {
    f.template operator()<1>();
    f.template operator()<2>();
    f.template operator()<3>();
    f.template operator()<4>();
    f.template operator()<5>();
    f.template operator()<8>();
    f.template operator()<16>();
}

/// The per-lane scalar evaluation of record for `vSinCos`. The target compiles
/// this file with `-fno-builtin` (see CMakeLists.txt), so the reference calls
/// the real `sin`/`cos` libm entries instead of letting LLVM fuse the adjacent
/// pair into Darwin's `__sincos` in one place and not the other; `noinline`
/// keeps the reference from being re-lowered differently from the kernel
/// instantiation.
#if defined(_MSC_VER)
#define LIGHTAPE_TEST_NOINLINE __declspec(noinline)
#else
#define LIGHTAPE_TEST_NOINLINE __attribute__((noinline))
#endif

LIGHTAPE_TEST_NOINLINE void ScalarSinCos(double x, double* s, double* c) {
    using std::cos;
    using std::sin;
    *s = sin(x);
    *c = cos(x);
}

template <std::size_t W, typename Kernel>
void EvalUnary(const std::vector<double>& in, std::vector<double>& out, Kernel& kernel) {
    out.resize(in.size());
    std::size_t i = 0;
    for (; i + W <= in.size(); i += W) {
        kernel.template operator()<W>(in.data() + i, out.data() + i);
    }
    if (i < in.size()) { // trailing lanes: overlap the last full W-block
        kernel.template operator()<W>(in.data() + in.size() - W, out.data() + in.size() - W);
    }
}

template <std::size_t W, typename Kernel>
void EvalBinary(const std::vector<double>& a, const std::vector<double>& b,
                std::vector<double>& out, Kernel& kernel) {
    out.resize(a.size());
    std::size_t i = 0;
    for (; i + W <= a.size(); i += W) {
        kernel.template operator()<W>(a.data() + i, b.data() + i, out.data() + i);
    }
    if (i < a.size()) {
        kernel.template operator()<W>(a.data() + a.size() - W, b.data() + b.size() - W,
                                      out.data() + a.size() - W);
    }
}

template <std::size_t W>
void EvalSinCos(const std::vector<double>& in, std::vector<double>& s, std::vector<double>& c) {
    s.resize(in.size());
    c.resize(in.size());
    std::size_t i = 0;
    for (; i + W <= in.size(); i += W) {
        VectorMath<W>::vSinCos(in.data() + i, s.data() + i, c.data() + i);
    }
    if (i < in.size()) {
        VectorMath<W>::vSinCos(in.data() + in.size() - W, s.data() + in.size() - W,
                               c.data() + in.size() - W);
    }
}

/// The differential gate for one (function, width): bitwise against the libm
/// reference build when `bitwise` (the table resolved the function to libm),
/// ULP-capped otherwise, plus the repeated-call bitwise stability check.
/// Every first-run output also feeds the FNV fingerprint.
template <std::size_t W>
void Compare(const char* name, const std::vector<double>& in, const std::vector<double>& got,
             const std::vector<double>& ref, double maxUlps, bool bitwise) {
    double worst = 0.0;
    std::size_t worstIdx = 0;
    std::size_t mismatches = 0;
    for (std::size_t i = 0; i < got.size(); ++i) {
        const double u = UlpDiff(got[i], ref[i]);
        if (!(u <= worst)) {
            worst = u;
            worstIdx = i;
        }
        if (Bits(got[i]) != Bits(ref[i])) {
            ++mismatches;
        }
        MixBits(got[i]);
    }
    NoteStat(name, worst, in[worstIdx], got[worstIdx], ref[worstIdx]);
    if (bitwise) {
        if (mismatches != 0) {
            for (std::size_t i = 0; i < got.size(); ++i) {
                if (Bits(got[i]) != Bits(ref[i])) {
                    EXPECT_EQ(Bits(got[i]), Bits(ref[i]))
                        << name << " W=" << W << " first mismatch at x=" << in[i]
                        << " got=" << got[i] << " ref=" << ref[i];
                    break;
                }
            }
        }
        EXPECT_EQ(mismatches, 0u) << name << " W=" << W << " must be bitwise libm";
    } else {
        EXPECT_LE(worst, maxUlps) << name << " W=" << W << " worst ULP " << worst
                                  << " at x=" << in[worstIdx] << " got=" << got[worstIdx]
                                  << " ref=" << ref[worstIdx];
    }
}

/// Bitwise stability of a repeated identical call.
void ExpectStable(const char* name, const std::vector<double>& first,
                  const std::vector<double>& second) {
    std::size_t mismatches = 0;
    for (std::size_t i = 0; i < first.size(); ++i) {
        mismatches += (Bits(first[i]) != Bits(second[i])) ? 1u : 0u;
    }
    EXPECT_EQ(mismatches, 0u) << name << " repeated-call bitwise stability";
}

template <typename Kernel, typename Ref>
void CheckUnaryCase(const char* name, Kernel kernel, Ref ref, double maxUlps, bool bitwise) {
    const std::vector<double>& in = Corpus();
    std::vector<double> refv(in.size());
    for (std::size_t i = 0; i < in.size(); ++i) {
        refv[i] = ref(in[i]);
    }
    ForEachWidth([&]<std::size_t W>() {
        std::vector<double> out;
        EvalUnary<W>(in, out, kernel);
        Compare<W>(name, in, out, refv, maxUlps, bitwise);
        std::vector<double> again;
        EvalUnary<W>(in, again, kernel);
        ExpectStable(name, out, again);
    });
}

template <typename Kernel, typename Ref>
void CheckBinaryCase(const char* name, Kernel kernel, Ref ref, double maxUlps, bool bitwise) {
    const std::vector<double>& a = Corpus();
    const std::vector<double>& b = CorpusB();
    std::vector<double> refv(a.size());
    for (std::size_t i = 0; i < a.size(); ++i) {
        refv[i] = ref(a[i], b[i]);
    }
    ForEachWidth([&]<std::size_t W>() {
        std::vector<double> out;
        EvalBinary<W>(a, b, out, kernel);
        Compare<W>(name, a, out, refv, maxUlps, bitwise);
        std::vector<double> again;
        EvalBinary<W>(a, b, again, kernel);
        ExpectStable(name, out, again);
    });
}

void CheckSinCosCase(double maxUlps, bool bitwise) {
    const std::vector<double>& in = Corpus();
    std::vector<double> refS(in.size());
    std::vector<double> refC(in.size());
    for (std::size_t i = 0; i < in.size(); ++i) {
        ScalarSinCos(in[i], refS.data() + i, refC.data() + i);
    }
    ForEachWidth([&]<std::size_t W>() {
        std::vector<double> s;
        std::vector<double> c;
        EvalSinCos<W>(in, s, c);
        Compare<W>("vSinCos.sin", in, s, refS, maxUlps, bitwise);
        Compare<W>("vSinCos.cos", in, c, refC, maxUlps, bitwise);
        std::vector<double> s2;
        std::vector<double> c2;
        EvalSinCos<W>(in, s2, c2);
        ExpectStable("vSinCos.sin", s, s2);
        ExpectStable("vSinCos.cos", c, c2);
    });
}

// ── fused vNormCdf gate (SLEEF-backed candidates) ──

/// Mirror of `detail::kNormCdfSplit` (VectorMath.h): the fused kernel's
/// central/tail boundary.
inline constexpr double kNormCdfSplit = 6.0;
/// Documented gates of the fused kernel: absolute in the central region
/// (the erf form loses relative accuracy but not absolute), relative in the
/// tails (the erfc form keeps it), and the seam gap at the split.
inline constexpr double kFusedNormCdfAbsCentral = 1e-15;
inline constexpr double kFusedNormCdfRelTail = 1e-12;
inline constexpr double kFusedNormCdfSeam = 1e-15;

/// Differential gate for the fused `vNormCdf`: |x| <= 6 -> absolute <= 1e-15
/// vs `lightape::normcdf`; |x| > 6 -> relative <= 1e-12; NaN passes through,
/// infinities are exact, an underflowed reference only requires the absolute
/// floor. Never bitwise against libm — the whole point of the fused kernel is
/// that the central expression differs from `0.5*erfc(...)`.
template <std::size_t W>
void CompareFusedNormCdf(const std::vector<double>& in, const std::vector<double>& got,
                         const std::vector<double>& ref) {
    double worstAbsCentral = 0.0;
    double worstRelTail = 0.0;
    double worstArgAbs = 0.0;
    double worstArgRel = 0.0;
    for (std::size_t i = 0; i < got.size(); ++i) {
        MixBits(got[i]);
        const double x = in[i];
        const double r = ref[i];
        if (std::isnan(r)) {
            EXPECT_TRUE(std::isnan(got[i])) << "vNormCdf W=" << W << " x=" << x;
            continue;
        }
        if (std::isinf(r)) {
            EXPECT_EQ(Bits(got[i]), Bits(r)) << "vNormCdf W=" << W << " x=" << x;
            continue;
        }
        if (std::fabs(x) <= kNormCdfSplit) {
            const double ad = std::fabs(got[i] - r);
            if (ad > worstAbsCentral) {
                worstAbsCentral = ad;
                worstArgAbs = x;
            }
        } else if (r != 0.0) {
            const double rel = std::fabs(got[i] - r) / std::fabs(r);
            if (rel > worstRelTail) {
                worstRelTail = rel;
                worstArgRel = x;
            }
        } else {
            EXPECT_LE(std::fabs(got[i]), kFusedNormCdfAbsCentral)
                << "vNormCdf W=" << W << " underflowed reference at x=" << x;
        }
    }
    EXPECT_LE(worstAbsCentral, kFusedNormCdfAbsCentral)
        << "vNormCdf W=" << W << " central absolute error at x=" << worstArgAbs;
    EXPECT_LE(worstRelTail, kFusedNormCdfRelTail)
        << "vNormCdf W=" << W << " tail relative error at x=" << worstArgRel;
}

/// Full fused gate: differential, repeated-call stability, and the seam at
/// +-6 (values from either side of the split must not step by more than the
/// central absolute gate).
void CheckFusedNormCdfCase() {
    const std::vector<double>& in = Corpus();
    std::vector<double> refv(in.size());
    for (std::size_t i = 0; i < in.size(); ++i) {
        refv[i] = lightape::normcdf(in[i]);
    }
    auto kernel = []<std::size_t VW>(const double* a, double* b) {
        VectorMath<VW>::vNormCdf(a, b);
    };
    ForEachWidth([&]<std::size_t W>() {
        std::vector<double> out;
        EvalUnary<W>(in, out, kernel);
        CompareFusedNormCdf<W>(in, out, refv);
        std::vector<double> again;
        EvalUnary<W>(in, again, kernel);
        ExpectStable("vNormCdf", out, again);

        // The true CDF moves by < 1e-24 across one ulp of 6; only the branch
        // switch could step, so the gap is the central formula's error.
        const auto evalOne = [&]<std::size_t VW>(double v) {
            std::array<double, VW> a{};
            std::array<double, VW> o{};
            a.fill(v);
            VectorMath<VW>::vNormCdf(a.data(), o.data());
            return o[0];
        };
        const double below = std::nextafter(kNormCdfSplit, 0.0);
        const double above = std::nextafter(kNormCdfSplit, 10.0);
        const double upperAt = evalOne.template operator()<W>(kNormCdfSplit);
        const double upperAbove = evalOne.template operator()<W>(above);
        const double upperBelow = evalOne.template operator()<W>(below);
        const double lowerAt = evalOne.template operator()<W>(-kNormCdfSplit);
        const double lowerAbove = evalOne.template operator()<W>(-below);
        const double lowerBelow = evalOne.template operator()<W>(-above);
        EXPECT_LE(std::fabs(upperBelow - upperAbove), kFusedNormCdfSeam)
            << "vNormCdf W=" << W << " seam below/above +6";
        EXPECT_LE(std::fabs(upperAt - upperAbove), kFusedNormCdfSeam)
            << "vNormCdf W=" << W << " seam at/above +6";
        EXPECT_LE(std::fabs(lowerBelow - lowerAbove), kFusedNormCdfSeam)
            << "vNormCdf W=" << W << " seam below/above -6";
        EXPECT_LE(std::fabs(lowerAt - lowerAbove), kFusedNormCdfSeam)
            << "vNormCdf W=" << W << " seam at/above -6";
    });
}

// Per-function ULP caps: max(2, measured W=8 worst + 1) on each ISA ladder,
// against the scalar libm reference on this file's dense corpus.
//
// x86-64 (2026-10 AWS bake-off, Intel SPR + AMD Zen 4, AVX-512 and AVX2):
// worst measured was 1 ULP for tan/erf and every function not listed below,
// 2 for tanh/normPdf (plus sinh/log10 on the denser unit corpus), and
// 3 for erfc/cbrt -> caps 2 / 3 / 4. These are deliberately tighter than the
// pre-bake-off guesses (which had tan/erf at 4, erfc at 6 and cbrt at 2).
//
// AArch64 (Apple M2 Pro, AdvSIMD d2, same corpus): tan 3, erf 3, erfc 5,
// everything else <= 2. The macOS libm reference is itself 2-5 ULP off at
// those witnesses (verified against a 60-digit MPFR evaluation in the V1
// report), so the AdvSIMD caps are 4/4/6 rather than the x86 values.
//
// `vNormCdf`'s SLEEF-backed candidates are fused kernels, not plain SLEEF
// entry points, so they never use these ULP caps: the libm-resolved entry is
// gated bitwise, every other entry through the dedicated absolute/relative
// gate below (the central erf form cancels to ~1e-9 at the split, where a ULP
// comparison against libm erfc is meaningless).
#if defined(__aarch64__)
inline constexpr double kCapTan = 4.0;
inline constexpr double kCapErf = 4.0;
inline constexpr double kCapErfc = 6.0;
inline constexpr double kCapCbrt = 2.0;
inline constexpr double kCapTanh = 2.0;
inline constexpr double kCapNormPdf = 3.0;
inline constexpr double kCapNormCdf = 6.0;
#else
inline constexpr double kCapTan = 2.0;
inline constexpr double kCapErf = 2.0;
inline constexpr double kCapErfc = 4.0;
inline constexpr double kCapCbrt = 4.0;
inline constexpr double kCapTanh = 3.0;
inline constexpr double kCapNormPdf = 3.0;
inline constexpr double kCapNormCdf = 3.0;
#endif

#define LIGHTAPE_VECTOR_MATH_UNARY_CASES(F)                                                        \
    F(vSqrt, sqrt, 2.0)                                                                            \
    F(vExp, exp, 2.0)                                                                              \
    F(vLog, log, 2.0)                                                                              \
    F(vExpm1, expm1, 2.0)                                                                          \
    F(vLog1p, log1p, 2.0)                                                                          \
    F(vSin, sin, 2.0)                                                                              \
    F(vCos, cos, 2.0)                                                                              \
    F(vTan, tan, kCapTan)                                                                          \
    F(vAsin, asin, 2.0)                                                                            \
    F(vAcos, acos, 2.0)                                                                            \
    F(vAtan, atan, 2.0)                                                                            \
    F(vSinh, sinh, 2.0)                                                                            \
    F(vCosh, cosh, 2.0)                                                                            \
    F(vTanh, tanh, kCapTanh)                                                                       \
    F(vExp2, exp2, 2.0)                                                                            \
    F(vLog2, log2, 2.0)                                                                            \
    F(vLog10, log10, 2.0)                                                                          \
    F(vErf, erf, kCapErf)                                                                          \
    F(vErfc, erfc, kCapErfc)                                                                       \
    F(vCbrt, cbrt, kCapCbrt)

#define LIGHTAPE_VECTOR_MATH_BINARY_CASES(F)                                                       \
    F(vPow, pow, 2.0)                                                                              \
    F(vAtan2, atan2, 2.0)                                                                          \
    F(vHypot, hypot, 2.0)

// The `fn` parameter doubles as the `MathTable` member name, so the macro can
// query the resolved implementation for this exact function.
#define LIGHTAPE_UNARY_CASE(name, fn, cap)                                                         \
    CheckUnaryCase(                                                                                \
        #name, []<std::size_t VW>(const double* a, double* b) { VectorMath<VW>::name(a, b); },     \
        [](double v) {                                                                             \
            using std::fn;                                                                         \
            return fn(v);                                                                          \
        },                                                                                         \
        cap, IsLibmEntry(lightape::batch::mathTable().fn));

#define LIGHTAPE_BINARY_CASE(name, fn, cap)                                                        \
    CheckBinaryCase(                                                                               \
        #name,                                                                                     \
        []<std::size_t VW>(const double* a, const double* b, double* o) {                          \
            VectorMath<VW>::name(a, b, o);                                                         \
        },                                                                                         \
        [](double a, double b) {                                                                   \
            using std::fn;                                                                         \
            return fn(a, b);                                                                       \
        },                                                                                         \
        cap, IsLibmEntry(lightape::batch::mathTable().fn));

/// Full differential sweep over the currently installed table. The bitwise
/// flag is derived per function from the resolved entry, so the same code
/// gates the auto table, the uniform candidates and the forced-libm table.
void RunDifferentialChecks() {
    LIGHTAPE_VECTOR_MATH_UNARY_CASES(LIGHTAPE_UNARY_CASE);
    LIGHTAPE_VECTOR_MATH_BINARY_CASES(LIGHTAPE_BINARY_CASE);
    CheckSinCosCase(2.0, IsLibmEntry(lightape::batch::mathTable().sincos));
    // Fused composites: reference is the exact libm expression they compose.
    CheckUnaryCase(
        "vNormPdf",
        []<std::size_t VW>(const double* a, double* b) { VectorMath<VW>::vNormPdf(a, b); },
        [](double v) { return lightape::normpdf(v); }, kCapNormPdf,
        IsLibmEntry(lightape::batch::mathTable().normPdf));
    // vNormCdf's libm entry is the bitwise reference (`0.5*erfc(...)`);
    // SLEEF-backed entries are the fused kernel and use the documented
    // absolute/relative gate instead of the ULP differential.
    if (IsLibmEntry(lightape::batch::mathTable().normCdf)) {
        CheckUnaryCase(
            "vNormCdf",
            []<std::size_t VW>(const double* a, double* b) { VectorMath<VW>::vNormCdf(a, b); },
            [](double v) { return lightape::normcdf(v); }, kCapNormCdf, true);
    } else {
        CheckFusedNormCdfCase();
    }
}

void PrintStats() {
    for (const Stat& s : Stats()) {
        std::printf("[vector-math] %-10s worst %8.3f ULP at x=%-24.17g got=%-24.17g ref=%-24.17g\n",
                    s.name.c_str(), s.worstUlp, s.worstIn, s.worstGot, s.worstRef);
    }
}

/// Test hook wrapper: installs a table for the dynamic extent of the object
/// and clears the slot afterwards (the next `mathTable()` re-resolves).
class ScopedMathTable {
public:
    explicit ScopedMathTable(const MathTable* table) {
        lightape::batch::setMathTableForTesting(table);
    }
    ~ScopedMathTable() { lightape::batch::setMathTableForTesting(nullptr); }
    ScopedMathTable(const ScopedMathTable&) = delete;
    ScopedMathTable& operator=(const ScopedMathTable&) = delete;
};

/// Unset every knob this header reads, so a test observes the default policy
/// regardless of how ctest/the shell was invoked.
void ClearMathEnv() {
    ::unsetenv("LIGHTAPE_VECTOR_MATH_FORCE_LIBM");
    ::unsetenv("LIGHTAPE_MATH_TABLE");
    ::unsetenv("LIGHTAPE_VECTOR_ISA");
    for (std::size_t i = 0; i < kMathFunctionCount; ++i) {
        char var[64];
        std::snprintf(var, sizeof(var), "LIGHTAPE_MATH_IMPL_%s", kMathFunctionNames[i]);
        ::unsetenv(var);
    }
}

TEST(VectorMathTest, MatchesReferenceAndIsStable) {
    g_fingerprint = kFnvOffset;
    RunDifferentialChecks();

    std::printf("[vector-math] build=%s table=%s width=%zu fingerprint=0x%016llx\n", BuildFlavor(),
                TableName(), PlanWidth(), static_cast<unsigned long long>(g_fingerprint));
    PrintStats();
#if defined(LIGHTAPE_FINGERPRINT_GOLDEN)
    EXPECT_EQ(g_fingerprint, static_cast<std::uint64_t>(LIGHTAPE_FINGERPRINT_GOLDEN))
        << "recorded fingerprint changed; per-table fingerprints are not stable across builds";
#else
    std::printf("[vector-math] (golden fingerprint assert disabled: define "
                "LIGHTAPE_FINGERPRINT_GOLDEN=<u64> to pin)\n");
#endif
}

/// One fingerprint per distinct resolved candidate table: the record of what
/// this build/ISA produces (plan §6.3). The libm candidate also exercises the
/// forced-libm bitwise-reference guarantee end to end, and the TU-split
/// invariant: the libm candidate must be bit-identical to the raw all-libm
/// table (the reference TU is compiled without ISA flags, so the SLEEF build's
/// libm fingerprint equals the portable libm build's; the bake-off script
/// asserts the cross-build equality).
TEST(VectorMathTest, FingerprintPerResolvedTable) {
    std::vector<std::pair<std::string, MathTable>> candidates;
    const auto add = [&](MathImpl choice) {
        MathTable table = lightape::batch::resolveMathTable(MathSelection{.table = choice});
        const std::string name = table.name;
        for (const auto& existing : candidates) {
            if (existing.first == name) {
                return;
            }
        }
        candidates.emplace_back(name, table);
    };
    add(MathImpl::Auto);
    add(MathImpl::Libm);
    add(MathImpl::Sleef);
    add(MathImpl::Generic);

    std::uint64_t libmFingerprint = 0;
    bool sawLibm = false;
    for (const auto& [name, table] : candidates) {
        ScopedMathTable guard(&table);
        g_fingerprint = kFnvOffset;
        RunDifferentialChecks();
        std::printf("[vector-math] table=%-8s fingerprint=0x%016llx\n", name.c_str(),
                    static_cast<unsigned long long>(g_fingerprint));
        if (name == "libm") {
            sawLibm = true;
            libmFingerprint = g_fingerprint;
            EXPECT_STREQ(table.sqrt.impl, "libm");
            EXPECT_STREQ(table.exp.impl, "libm");
            EXPECT_STREQ(table.erfc.impl, "libm");
            EXPECT_STREQ(table.sincos.impl, "libm");
        }
    }
    ASSERT_TRUE(sawLibm);

    ScopedMathTable guard(&lightape::batch::libmTable());
    g_fingerprint = kFnvOffset;
    RunDifferentialChecks();
    EXPECT_EQ(g_fingerprint, libmFingerprint)
        << "the libm reference must be bitwise identical to the raw libm table "
           "regardless of build flavor (no ISA flags in the reference TUs)";
}

/// The per-width resolution contract: every `v*` call must pick the widest
/// candidate that fits its W (W=4 on an AVX-512 machine runs the AVX2 d4
/// kernel, never the 8-wide one), the tail is the width-1 libm kernel, and a
/// requested ISA caps the ladder.
TEST(VectorMathTest, SelectsWidestKernelThatFits) {
#if defined(LIGHTAPE_VECTOR_MATH_SLEEF)
    ClearMathEnv();
    const MathTable sleef =
        lightape::batch::resolveMathTable(MathSelection{.table = MathImpl::Sleef});

    const lightape::batch::UnaryEntry* one = lightape::batch::selectEntry<1>(sleef.exp);
    ASSERT_NE(one, nullptr);
    EXPECT_EQ(one->width, 1u); // libm always ends the chain
    EXPECT_STREQ(one->impl, "libm");

    const lightape::batch::UnaryEntry* two = lightape::batch::selectEntry<2>(sleef.exp);
    ASSERT_NE(two, nullptr);
    EXPECT_LE(two->width, 2u);

#if defined(__aarch64__)
    EXPECT_EQ(two->width, 2u);
    EXPECT_STREQ(two->isa, "advsimd");
#elif defined(__x86_64__) && (defined(__GNUC__) || defined(__clang__))
    __builtin_cpu_init();
    const bool has512 = __builtin_cpu_supports("avx512f");
    const bool has256 = __builtin_cpu_supports("avx2") && __builtin_cpu_supports("fma");
    const lightape::batch::UnaryEntry* four = lightape::batch::selectEntry<4>(sleef.exp);
    const lightape::batch::UnaryEntry* eight = lightape::batch::selectEntry<8>(sleef.exp);
    ASSERT_NE(four, nullptr);
    ASSERT_NE(eight, nullptr);
    if (has512) {
        EXPECT_EQ(eight->width, 8u);
        EXPECT_STREQ(eight->isa, "avx512f");
        EXPECT_EQ(four->width, 4u);
        EXPECT_STREQ(four->isa, "avx2");
        EXPECT_NE(four->fn, eight->fn); // W=4 must not run the 8-wide kernel
        EXPECT_EQ(two->width, 2u);
        EXPECT_STREQ(two->isa, "avx2128");
    } else if (has256) {
        EXPECT_EQ(four->width, 4u); // capped: no AVX-512 on this part
        EXPECT_STREQ(four->isa, "avx2");
        EXPECT_EQ(eight->width, 4u);
        EXPECT_EQ(two->width, 2u);
        EXPECT_STREQ(two->isa, "avx2128");
    } else {
        EXPECT_EQ(eight->width, 1u); // no AVX2: the libm tail
    }

    if (has512) {
        // A requested ISA caps the ladder: `avx2` never resolves avx512f.
        ::setenv("LIGHTAPE_VECTOR_ISA", "avx2", 1);
        const MathTable capped =
            lightape::batch::resolveMathTable(MathSelection{.table = MathImpl::Sleef});
        ClearMathEnv();
        const lightape::batch::UnaryEntry* eightCapped =
            lightape::batch::selectEntry<8>(capped.exp);
        ASSERT_NE(eightCapped, nullptr);
        EXPECT_EQ(eightCapped->width, 4u);
        EXPECT_STREQ(eightCapped->isa, "avx2");
    }
#endif
#else
    const MathTable& table = lightape::batch::mathTable();
    const lightape::batch::UnaryEntry* one = lightape::batch::selectEntry<1>(table.exp);
    ASSERT_NE(one, nullptr);
    EXPECT_EQ(one->width, 1u);
    const lightape::batch::UnaryEntry* eight = lightape::batch::selectEntry<8>(table.exp);
    ASSERT_NE(eight, nullptr);
    EXPECT_EQ(eight->width, 1u);
#endif
}

/// The env path is parsed by `selectionFromEnv()` and resolved by
/// `resolveMathTable()`; both are exercised here without touching the cached
/// process table, plus the documented fallback when SLEEF is not compiled in
/// and the `FORCE_LIBM` precedence.
TEST(VectorMathTest, EnvOverrideSelectionResolvesPerFunction) {
    ClearMathEnv();
    ::setenv("LIGHTAPE_MATH_TABLE", "libm", 1);
    ::setenv("LIGHTAPE_MATH_IMPL_vLog", "generic", 1);
    ::setenv("LIGHTAPE_MATH_IMPL_vExp", "sleef", 1);
    ::setenv("LIGHTAPE_MATH_IMPL_vSin", "banana", 1); // unknown -> inherit

    const MathSelection sel = lightape::batch::selectionFromEnv();
    EXPECT_EQ(sel.table, MathImpl::Libm);
    EXPECT_FALSE(sel.forceLibm);
    EXPECT_EQ(sel.perFunction[static_cast<std::size_t>(MathFn::log)], MathImpl::Generic);
    EXPECT_EQ(sel.perFunction[static_cast<std::size_t>(MathFn::exp)], MathImpl::Sleef);
    EXPECT_EQ(sel.perFunction[static_cast<std::size_t>(MathFn::sin)], MathImpl::Auto);

    const MathTable table = lightape::batch::resolveMathTable(sel);
#if defined(LIGHTAPE_VECTOR_MATH_SLEEF)
    EXPECT_STREQ(table.log.impl, "generic");
    EXPECT_EQ(table.log.width, 1u);
#if defined(__x86_64__) && (defined(__GNUC__) || defined(__clang__))
    __builtin_cpu_init();
    if (__builtin_cpu_supports("avx2") && __builtin_cpu_supports("fma")) {
        EXPECT_STRNE(table.exp.impl, "libm"); // a vector rung exists
    } else {
        EXPECT_STREQ(table.exp.impl, "libm"); // x86 `sleef` -> libm without AVX2
    }
#else
    EXPECT_STRNE(table.exp.impl, "libm");
#endif
    EXPECT_STREQ(table.sin.impl, "libm"); // unknown value inherited the table
#else
    // Documented fallback: no SLEEF compiled, non-libm requests resolve libm.
    EXPECT_STREQ(table.log.impl, "libm");
    EXPECT_STREQ(table.exp.impl, "libm");
    EXPECT_STREQ(table.sin.impl, "libm");
#endif

    // FORCE_LIBM wins over the table and the per-function overrides.
    ::setenv("LIGHTAPE_VECTOR_MATH_FORCE_LIBM", "1", 1);
    const MathSelection forced = lightape::batch::selectionFromEnv();
    EXPECT_TRUE(forced.forceLibm);
    const MathTable forcedTable = lightape::batch::resolveMathTable(forced);
    EXPECT_STREQ(forcedTable.log.impl, "libm");
    EXPECT_STREQ(forcedTable.exp.impl, "libm");
    EXPECT_STREQ(forcedTable.name, "libm");
    EXPECT_EQ(forcedTable.width, 1u);

    // Default auto policy (all knobs cleared).
    ClearMathEnv();
    const MathTable autoTable = lightape::batch::resolveMathTable(MathSelection{});
#if defined(__aarch64__)
    // M2 evidence is corpus-dependent (VectorMath.h), so the conservative
    // auto list holds only the one measured-in function: the fused vNormCdf
    // (>5% vs the libm table candidate at W=8 in the harness's decision row).
    // A machine that reproduces another SLEEF win can force it per function
    // (`LIGHTAPE_MATH_IMPL_vLog=sleef`, ...) with no code change. The libm
    // build has no SLEEF candidate, so the same request resolves to libm.
    EXPECT_STREQ(autoTable.log.impl, "libm");
    EXPECT_STREQ(autoTable.exp.impl, "libm");
#if defined(LIGHTAPE_VECTOR_MATH_SLEEF)
    EXPECT_STRNE(autoTable.normCdf.impl, "libm");
#else
    EXPECT_STREQ(autoTable.normCdf.impl, "libm");
#endif
#elif defined(__x86_64__)
    // x86 auto defaults to SLEEF for every function except the measured erfc
    // exception (VectorMath.h); the fused vNormCdf is the SLEEF candidate and
    // is therefore in `auto`. Without an AVX2+FMA CPU there is no vector rung
    // and `sleef` honestly resolves to libm; the exceptions and the
    // override/table requests are checked only where a vector plan exists.
#if defined(LIGHTAPE_VECTOR_MATH_SLEEF)
    __builtin_cpu_init();
    const bool hasVectorPlan = __builtin_cpu_supports("avx2") && __builtin_cpu_supports("fma");
    if (hasVectorPlan) {
        EXPECT_STRNE(autoTable.exp.impl, "libm");
        EXPECT_STREQ(autoTable.erfc.impl, "libm");
        EXPECT_STRNE(autoTable.normCdf.impl, "libm");
        // ... but the per-function override and the table-wide `sleef`
        // request still reach SLEEF for the exceptions.
        ::setenv("LIGHTAPE_MATH_IMPL_vErfc", "sleef", 1);
        const MathTable erfcSleef =
            lightape::batch::resolveMathTable(lightape::batch::selectionFromEnv());
        ClearMathEnv();
        EXPECT_STRNE(erfcSleef.erfc.impl, "libm");
        // Explicit (env-free) selection path as a backstop.
        MathSelection explicitErfc;
        explicitErfc.perFunction[static_cast<std::size_t>(MathFn::erfc)] = MathImpl::Sleef;
        EXPECT_STRNE(lightape::batch::resolveMathTable(explicitErfc).erfc.impl, "libm");
        const MathTable allSleef =
            lightape::batch::resolveMathTable(MathSelection{.table = MathImpl::Sleef});
        EXPECT_STRNE(allSleef.erfc.impl, "libm");
        EXPECT_STRNE(allSleef.normCdf.impl, "libm");
    } else {
        EXPECT_STREQ(autoTable.exp.impl, "libm");
        EXPECT_STREQ(autoTable.erfc.impl, "libm");
    }
#else
    EXPECT_STREQ(autoTable.exp.impl, "libm");
    EXPECT_STREQ(autoTable.erfc.impl, "libm");
    EXPECT_STREQ(autoTable.normCdf.impl, "libm");
#endif
#else
    EXPECT_STREQ(autoTable.exp.impl, "libm");
#endif
    ClearMathEnv();
}

TEST(VectorMathTest, BackendTagAndPlanName) {
#if defined(LIGHTAPE_VECTOR_MATH_SLEEF)
    EXPECT_EQ(VectorMath<1>::backend, MathBackend::Sleef);
    EXPECT_EQ(VectorMath<8>::backend, MathBackend::Sleef);
#else
    EXPECT_EQ(VectorMath<1>::backend, MathBackend::LibmScalar);
    EXPECT_EQ(VectorMath<8>::backend, MathBackend::LibmScalar);
#endif
    EXPECT_NE(VectorMath<1>::isaName(), nullptr);
    EXPECT_GE(VectorMath<1>::vWidth(), 1u);
}

/// Decision #22: no FTZ/DAZ, so underflow stays subnormal and denormal
/// inputs survive the vector path. The witnesses match the scalar libm path
/// (bitwise under libm, <= 1 ULP under SLEEF) and must not be flushed to 0.
TEST(VectorMathTest, PreservesDenormals) {
    constexpr std::size_t W = 4;
    const double witness = -745.0; // exp(witness) == +denorm_min here
    const double ref = std::exp(witness);
    EXPECT_EQ(std::fpclassify(ref), FP_SUBNORMAL);
    EXPECT_NE(ref, 0.0);

    std::array<double, W> in{};
    in.fill(witness);
    std::array<double, W> out{};
    VectorMath<W>::vExp(in.data(), out.data());
    for (std::size_t l = 0; l < W; ++l) {
        EXPECT_EQ(std::fpclassify(out[l]), FP_SUBNORMAL) << "lane " << l << " flushed to zero";
        EXPECT_LE(UlpDiff(out[l], ref), 1.0) << "lane " << l;
    }

    const double tiny = std::numeric_limits<double>::denorm_min();
    std::array<double, W> d{};
    d.fill(tiny);
    std::array<double, W> r{};
    VectorMath<W>::vSqrt(d.data(), r.data());
    for (std::size_t l = 0; l < W; ++l) {
        EXPECT_EQ(Bits(r[l]), Bits(std::sqrt(tiny))) << "lane " << l << " denormal input";
    }
}

} // namespace
