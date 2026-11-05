#ifndef LIGHTAPE_BATCH_VECTORMATH_H
#define LIGHTAPE_BATCH_VECTORMATH_H

#include <array>
#include <atomic>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#ifdef LIGHTAPE_VECTOR_MATH_SLEEF
#if defined(__aarch64__)
// The AdvSIMD (and generic scalar) kernels stay header-only on AArch64: NEON
// is baseline. On x86-64 the SLEEF calls live in compiled per-ISA TUs and this
// header carries only `extern "C"` declarations (below), so it never includes
// `sleef.h` or `<immintrin.h>` there.
#include <sleef.h>
#endif
#endif

namespace lightape::batch {
/**
 * @file VectorMath.h
 * @brief Math-backend seam for the batch lane kernels (SIMD vector-math plan
 * §4-7, decision #28): one `VectorMath<W>` per lane width and one process-wide
 * `MathTable` of resolved per-function kernels, so the vector-libm backend
 * can be selected **per function and per ISA** without touching `BatchTape`'s
 * op set or its sweep.
 *
 * Every kernel is a W-lane contiguous array map (`const double*` in,
 * `double*` out, lanes 0..W-1):
 *   - unary: vSqrt, vExp, vLog, vExpm1, vLog1p, vSin, vCos, vTan, vAsin,
 *     vAcos, vAtan, vSinh, vCosh, vTanh, vAsinh, vAcosh, vAtanh, vExp2,
 *     vLog2, vLog10, vErf, vErfc, vCbrt, vNormPdf, vNormCdf;
 *   - binary (both operands arrays): vPow, vAtan2, vHypot;
 *   - dual output: vSinCos.
 *
 * The three inverse-hyperbolic kernels are **libm-only** for now: their
 * `MathTable` entries carry the single width-1 libm candidate
 * (`impl == "libm"`, `next == nullptr`) and are deliberately kept out of the
 * per-ISA X-macro ladders, so no `lightape_sleef_*_{asinh,acosh,atanh}`
 * kernels are required. SLEEF can be added later by extending the X-macro
 * lists and the `src/batch/Sleef*.cpp` TUs; `BatchTape` needs no change then.
 * Consequently they do not participate in the table's uniform-name summary.
 *
 * Every function resolves once per process to a candidate chain in
 * `mathTable()` (atomic slot, no magic-static guard), ordered widest first.
 * `VectorMath<W>::v*` runs the widest candidate with `width <= W` (see
 * `selectEntry<W>`) and finishes the remaining lanes through the width-1
 * `libm1` kernel of the same function. `LibmScalar` is therefore the bitwise
 * reference of record (decision #22): under a table whose entries all resolve
 * to libm, every output is bitwise equal to the scalar `RevScalar` path
 * evaluating the same expression lane by lane, in lane order.
 *
 * ## Nearest-width kernel selection
 *
 * The x86-64 ladder is avx512f (8 lanes) -> avx2 (4) -> avx2128 (2) -> libm
 * (1); AArch64 is advsimd (2) -> libm (1). A `VectorMath<4>` call on an
 * AVX-512 machine therefore runs the 4-wide AVX2 kernel instead of falling
 * through the 8-wide entry, which the 2026-10 AWS bake-off measured collapsing
 * to scalar speed (the 8-wide kernel does not fit 4 lanes): vExp 6.18 ns/value
 * at W=4 vs 1.54 at W=8 on Intel, 5.28 vs 1.13 on AMD. Selection is a bounded
 * walk over the chain (no allocation, no env access in the hot path).
 *
 * ## Resolution order (first match wins)
 *
 *   1. `LIGHTAPE_VECTOR_MATH_FORCE_LIBM=1` — every function libm (reference
 *      mode; also what the pre-table `LIGHTAPE_VECTOR_ISA=libm` implies).
 *   2. `LIGHTAPE_MATH_IMPL_<vFn>=libm|sleef|generic` — per-function override,
 *      e.g. `LIGHTAPE_MATH_IMPL_vLog=sleef`. `<vFn>` is the exact function
 *      name from `kMathFunctionNames` (`vSqrt`, ..., `vSinCos`, `vNormPdf`).
 *      Unknown values inherit the table default.
 *   3. `LIGHTAPE_MATH_TABLE=libm|sleef|generic|auto` (default `auto`) — the
 *      table-wide default. `sleef`/`generic` are silently unavailable (and
 *      fall back to libm) in a build configured without SLEEF; the resolved
 *      entry honestly reports `impl == "libm"`.
 *   4. `auto` policy, evidence-based (see below).
 *
 * `LIGHTAPE_VECTOR_ISA=advsimd|avx512f|avx2|avx2128|generic|libm|auto` still
 * selects the concrete SLEEF ladder that `sleef` resolves against (`auto` or
 * unset = the widest ISA the CPU supports); it no longer forces the whole
 * table. A requested ISA caps the ladder: `avx2` on an AVX-512 machine keeps
 * the avx2/avx2128/libm candidates and never resolves to avx512f. On x86-64
 * without AVX2+FMA there is no SLEEF vector rung, so `sleef`/`auto` resolve to
 * libm (`LIGHTAPE_VECTOR_ISA=generic` still selects the portable scalar plan).
 *
 * ## Fused normal CDF (`vNormCdf`)
 *
 * The SLEEF-backed candidate for `vNormCdf` is not the plain `u15` erfc
 * composition — SLEEF's erfc is its slowest elementary kernel (0.5-0.7x of
 * libm on x86, ~0.3x on M2) — but a fused kernel with a two-region split at
 * `kNormCdfSplit = 6`:
 *   - central region `|x| <= 6`: `0.5*erf(x/sqrt(2)) + 0.5`, i.e. SLEEF's
 *     fast erf (2.7-3x faster than its erfc; 3.3x faster than scalar libm
 *     normcdf on AVX-512) scaled by `kInvSqrt2`. The form loses relative
 *     accuracy as the value cancels toward ~1e-9 at the split, but its
 *     absolute error stays <= 1e-15 — the documented central gate;
 *   - tails `|x| > 6`: `0.5*erfc(-x/sqrt(2))`. The per-ISA vector erfc is
 *     not reused (a whole-block erfc would recompute every central lane, and
 *     the M2 raw probe measured SLEEF's u15 erfc slower than libm per lane);
 *     the rare tail lanes call scalar libm erfc, whose expression is bitwise
 *     equal to `lightape::normcdf`. Tails keep relative error <= 1e-12.
 *
 * The AdvSIMD kernel computes the central expression in one `vfma` (single
 * rounding, bit-identical to the scalar `0.5*e + 0.5` because `0.5*e` is
 * exact); the x86-64 kernels compose the declared per-ISA `erf` entry through
 * `detail::fusedNormCdfK`, because the header cannot call `sleef.h` there.
 * Both branches agree at the split to ~1e-16, so the split adds no step.
 *
 * ## `auto` policy (measured, not assumed)
 *
 * Evidence on Apple M2 Pro (2026-10-08): the V1-style raw probe (direct
 * scalar `std::` vs raw SLEEF AdvSIMD kernels, inputs near 0.25) shows Apple
 * libm winning or tying every double function — `exp` a tie, `log` a ~1.6x
 * libm win, `sin`/`sincos`/`tan`/`tanh`/`erf`/`erfc`/`pow`/`cbrt` libm wins
 * — and does not reproduce the V1 summary's "log ~2x SLEEF win". The
 * bake-off harness (`benchmarks/bench_vector_math.cpp`) on its broader
 * [0.3, 2.0) timing corpus shows the same overall picture but a handful of
 * AdvSIMD wins (`vSin`, `vAsin`, `vExp2`, `vErf`, `vSinCos`), i.e. the
 * per-function winner is corpus-dependent. The conservative default follows:
 *   - AArch64: `libm` for everything except the functions listed in
 *     `kAutoSleefAarch64` — currently `{}`. Extend that list only from a
 *     fresh `benchmarks/bench_vector_math.cpp` run on the deployment part
 *     and workload, e.g. `LIGHTAPE_MATH_IMPL_vLog=sleef` if a machine
 *     reproduces V1, or `vErf` etc. if the workload corpus confirms it.
 *     The fused `vNormCdf` is the exception measured in on M2 (2026-10-08,
 *     two `lightape_vector_math_bench` runs): 9.43/9.48 ns/value at W=8 vs
 *     11.14/11.25 for the libm table candidate (>5% in the harness's
 *     decision row, which produces the `LIGHTAPE_MATH_IMPL_vNormCdf=sleef`
 *     suggestion) and 9.19/9.22 for the direct scalar libm expression. The
 *     win is over the auto *table* path; against raw libm it is a tie inside
 *     the noise floor, which is documented in `kAutoSleefAarch64`.
 *   - x86-64: `sleef` for every function except the one measured exception
 *     (W=8 ns/value, SLEEF AVX-512 vs the direct scalar libm reference;
 *     2026-10 AWS bake-off on Intel SPR and AMD Zen 4):
 *       - `vErfc`: Intel 9.45 vs 8.90 (and 9.17 vs 9.17 in the first run),
 *         AMD 7.85 vs 8.74 -> libm.
 *     `vNormCdf` was removed from the exception list when the fused kernel
 *     landed: the same bake-off measures SLEEF AVX-512 erf at 3.28 (Intel)
 *     / 2.87 (AMD) ns/value versus 5.88/5.69 for direct scalar libm normcdf
 *     and 7.01/7.05 for the libm table row, and the fused kernel is that erf
 *     entry plus a scale/blend and an unreachable tail branch. NOTE: the
 *     fused kernel itself was not run on the x86 hosts (the bake-off table
 *     predates it); the x86 win is extrapolated from its measured `vErf`
 *     component, not a direct measurement.
 *     Every other function wins 2.5-4.5x (vExp 1.5 vs 6.0, vLog 1.4-1.9 vs
 *     4.4-5.0, vNormPdf 1.3-1.4 vs 4.7-4.9, ...). The AVX2 W=4 rows agree
 *     (erfc 16.4 vs 9.2 on Intel), so the policy holds on AVX2-only and
 *     AVX-512 parts alike. `LIGHTAPE_MATH_IMPL_<vFn>` still overrides any of
 *     it, and `LIGHTAPE_MATH_TABLE=sleef` forces SLEEF for the exception.
 *   - anything else: `libm`.
 *
 * ## Per-ISA translation units
 *
 * On x86-64 every SLEEF call lives in a compiled TU
 * (`src/batch/SleefGeneric.cpp`, `SleefAvx2.cpp` with the 4-wide AVX2 and
 * 2-wide AVX2128 kernels, `SleefAvx512.cpp` with the 8-wide AVX-512F kernels),
 * built with per-source ISA flags and `SLEEF_STATIC_LIBS` and linked only into
 * the SLEEF test/bench executables. This header itself includes no `sleef.h`
 * and no intrinsics on x86-64, so the libm reference TUs are compiled without
 * ISA flags; the libm candidate fingerprint of a SLEEF build is therefore
 * bit-identical to the portable libm build's. AArch64 keeps the AdvSIMD and
 * generic kernels header-only (NEON is baseline), so the TU split is x86-only.
 *
 * ## Dispatch cost
 *
 * V1 measured the naive table form (one indirect call per lane through a
 * width-1 libm entry) at 17-26% slower than the direct inlined scalar loop on
 * the M2 composite bench at W=1/W=8. That cost is the price of per-function
 * selectability and is accepted for V2; V3 will hoist the resolution out of
 * the tape sweep and unroll 2-4 vectors, which removes most of it. The
 * bake-off harness reports the direct-vs-table gap per function.
 *
 * ## Determinism / fingerprints
 *
 * Results are bitwise repeatable within one resolved table. A 64-bit FNV
 * fingerprint of a fixed evaluation corpus (`tests/test_vector_math.cpp`) is
 * the per-table record; cross-table bitwise equality is **not** promised
 * (FMA use and evaluation order differ per ISA, decision #22 / plan §6.3).
 * The libm-resolved table remains the cross-machine reference; a SLEEF table
 * is differential ≤2 ULP against it on the accuracy gates.
 */

/// Build-level backend tag (the concrete per-function table is resolved at
/// runtime; `VectorMath<W>::isaName()` reports the resolved summary).
enum class MathBackend { LibmScalar, Sleef };

namespace detail {

using UnaryKernel = void (*)(const double*, double*);
using BinaryKernel = void (*)(const double*, const double*, double*);
using SinCosKernel = void (*)(const double*, double*, double*);

// ── X-macro function lists: the single source of truth for the per-ISA
// entry-point declarations, the candidate-table construction and the
// nearest-width fallback wiring. Defined before the first use and #undef'ed at
// the end of the header. ──
#define LIGHTAPE_MATH_UNARY_FUNCTIONS(X)                                                           \
    X(sqrt)                                                                                        \
    X(exp)                                                                                         \
    X(log)                                                                                         \
    X(expm1)                                                                                       \
    X(log1p)                                                                                       \
    X(sin)                                                                                         \
    X(cos)                                                                                         \
    X(tan)                                                                                         \
    X(asin)                                                                                        \
    X(acos)                                                                                        \
    X(atan)                                                                                        \
    X(sinh)                                                                                        \
    X(cosh)                                                                                        \
    X(tanh)                                                                                        \
    X(exp2)                                                                                        \
    X(log2)                                                                                        \
    X(log10)                                                                                       \
    X(erf)                                                                                         \
    X(erfc)                                                                                        \
    X(cbrt)                                                                                        \
    X(normPdf)                                                                                     \
    X(normCdf)
/// `LIGHTAPE_MATH_UNARY_FUNCTIONS` minus `normCdf`: the x86-64 alias macros
/// bind the compiled-TU entry points for the plain kernels, while the fused
/// `normCdfK` is defined in the header after the aliases (it composes the
/// per-ISA `erfK` with the normcdf split/tail logic, see `kNormCdfSplit`) and
/// must not be overwritten by an alias. Every other consumer uses the full
/// list, so the two stay in sync by construction.
#define LIGHTAPE_MATH_UNARY_FUNCTIONS_ALIASABLE(X)                                                 \
    X(sqrt)                                                                                        \
    X(exp)                                                                                         \
    X(log)                                                                                         \
    X(expm1)                                                                                       \
    X(log1p)                                                                                       \
    X(sin)                                                                                         \
    X(cos)                                                                                         \
    X(tan)                                                                                         \
    X(asin)                                                                                        \
    X(acos)                                                                                        \
    X(atan)                                                                                        \
    X(sinh)                                                                                        \
    X(cosh)                                                                                        \
    X(tanh)                                                                                        \
    X(exp2)                                                                                        \
    X(log2)                                                                                        \
    X(log10)                                                                                       \
    X(erf)                                                                                         \
    X(erfc)                                                                                        \
    X(cbrt)                                                                                        \
    X(normPdf)

#define LIGHTAPE_MATH_BINARY_FUNCTIONS(X) X(pow) X(atan2) X(hypot)
#define LIGHTAPE_MATH_SINCOS_FUNCTIONS(X) X(sincos)

/// 1/sqrt(2*pi) and 1/sqrt(2), bit-identical to `lightape::kInvSqrt2Pi` /
/// `lightape::kInvSqrt2` (ops/Probability.h), so the fused normal kernels
/// below are bitwise equal to `lightape::normpdf` / `lightape::normcdf`.
inline constexpr double kInvSqrt2Pi = 0.39894228040143267794;
inline constexpr double kInvSqrt2 = 0.70710678118654752440;

/// Split of the fused `vNormCdf` kernel: the central region (|x| <= 6)
/// computes `0.5*erf(x/sqrt(2)) + 0.5` — SLEEF's erf is 2.7-3x faster than its
/// slow u15 erfc — and the tails (|x| > 6) switch to `0.5*erfc(-x/sqrt(2))`,
/// which keeps full relative accuracy where the central form cancels. The
/// central form's absolute error against the true CDF stays <= 1e-15 even at
/// the split (where the value is ~1e-9); the tails' relative error is
/// <= 1e-12. `6.0` is exactly representable and both branches agree at the
/// split to ~1e-16, so the kernel stays continuous (no step at the seam).
inline constexpr double kNormCdfSplit = 6.0;

// ── LibmScalar kernels: width 1, one `std::` call per lane. Always compiled
// (both build flavors): this is the bitwise reference of record and the
// scalar tail of every wider plan. Each function-local `using std::...`
// keeps the call a direct libm entry. ──
namespace libm1 {

inline void sqrtK(const double* in, double* out) {
    using std::sqrt;
    *out = sqrt(*in);
}
inline void expK(const double* in, double* out) {
    using std::exp;
    *out = exp(*in);
}
inline void logK(const double* in, double* out) {
    using std::log;
    *out = log(*in);
}
inline void expm1K(const double* in, double* out) {
    using std::expm1;
    *out = expm1(*in);
}
inline void log1pK(const double* in, double* out) {
    using std::log1p;
    *out = log1p(*in);
}
inline void sinK(const double* in, double* out) {
    using std::sin;
    *out = sin(*in);
}
inline void cosK(const double* in, double* out) {
    using std::cos;
    *out = cos(*in);
}
inline void tanK(const double* in, double* out) {
    using std::tan;
    *out = tan(*in);
}
inline void asinK(const double* in, double* out) {
    using std::asin;
    *out = asin(*in);
}
inline void acosK(const double* in, double* out) {
    using std::acos;
    *out = acos(*in);
}
inline void atanK(const double* in, double* out) {
    using std::atan;
    *out = atan(*in);
}
inline void sinhK(const double* in, double* out) {
    using std::sinh;
    *out = sinh(*in);
}
inline void coshK(const double* in, double* out) {
    using std::cosh;
    *out = cosh(*in);
}
inline void tanhK(const double* in, double* out) {
    using std::tanh;
    *out = tanh(*in);
}
inline void asinhK(const double* in, double* out) {
    using std::asinh;
    *out = asinh(*in);
}
inline void acoshK(const double* in, double* out) {
    using std::acosh;
    *out = acosh(*in);
}
inline void atanhK(const double* in, double* out) {
    using std::atanh;
    *out = atanh(*in);
}
inline void exp2K(const double* in, double* out) {
    using std::exp2;
    *out = exp2(*in);
}
inline void log2K(const double* in, double* out) {
    using std::log2;
    *out = log2(*in);
}
inline void log10K(const double* in, double* out) {
    using std::log10;
    *out = log10(*in);
}
inline void erfK(const double* in, double* out) {
    using std::erf;
    *out = erf(*in);
}
inline void erfcK(const double* in, double* out) {
    using std::erfc;
    *out = erfc(*in);
}
inline void cbrtK(const double* in, double* out) {
    using std::cbrt;
    *out = cbrt(*in);
}
inline void powK(const double* a, const double* b, double* out) {
    using std::pow;
    *out = pow(*a, *b);
}
inline void atan2K(const double* y, const double* x, double* out) {
    using std::atan2;
    *out = atan2(*y, *x);
}
inline void hypotK(const double* a, const double* b, double* out) {
    using std::hypot;
    *out = hypot(*a, *b);
}
inline void sincosK(const double* in, double* s, double* c) {
    using std::cos;
    using std::sin;
    *s = sin(*in);
    *c = cos(*in);
}
inline void normPdfK(const double* in, double* out) {
    using std::exp;
    *out = kInvSqrt2Pi * exp(-0.5 * (*in * *in));
}
inline void normCdfK(const double* in, double* out) {
    using std::erfc;
    *out = 0.5 * erfc(-kInvSqrt2 * *in);
}

} // namespace libm1

#ifdef LIGHTAPE_VECTOR_MATH_SLEEF

#if defined(__x86_64__) && (defined(__GNUC__) || defined(__clang__))

// ── x86-64: every SLEEF call lives in a compiled per-ISA TU
// (`src/batch/SleefGeneric.cpp`; `SleefAvx2.cpp` hosts the 4-wide AVX2 and the
// 2-wide AVX2128 kernels; `SleefAvx512.cpp` the 8-wide AVX-512F kernels). The
// per-source ISA flags and `SLEEF_STATIC_LIBS` live in lightape/CMakeLists.txt
// and only the SLEEF test/bench executables link them. This header therefore
// never includes `sleef.h` or `<immintrin.h>` on x86-64: it only declares the
// plain `extern "C"` entry points and aliases them into the candidate
// namespaces used by the tables below. ──

/// Fused normal CDF (see `kNormCdfSplit`) built from an already-bound
/// width-`W` `erf` entry: the TU holds the SLEEF call, this header only
/// composes it. The rare tail lanes fall back to scalar libm `erfc` per lane:
/// the 2026-10 AWS bake-off measured no per-lane vector-erfc win (AVX-512
/// erfc is a tie at best in ns/value) and a block erfc would recompute every
/// central lane, while the tail branch is unreachable on the timing corpus.
/// The central expression is `0.5*erf(x/sqrt(2)) + 0.5` (absolute error
/// <= 1e-15), the tail expression is the libm `0.5*erfc(-x/sqrt(2))` of the
/// reference (relative error <= 1e-12, and bitwise equal to `lightape::normcdf`
/// on tail lanes).
template <std::size_t W, UnaryKernel Erf>
void fusedNormCdfK(const double* in, double* out) {
    double scaled[W];
    for (std::size_t l = 0; l < W; ++l) {
        scaled[l] = kInvSqrt2 * in[l];
    }
    Erf(scaled, out);
    bool tail = false;
    for (std::size_t l = 0; l < W; ++l) {
        out[l] = 0.5 * out[l] + 0.5;
        tail = tail || std::fabs(in[l]) > kNormCdfSplit;
    }
    if (tail) [[unlikely]] {
        using std::erfc;
        for (std::size_t l = 0; l < W; ++l) {
            if (std::fabs(in[l]) > kNormCdfSplit) {
                out[l] = 0.5 * erfc(-kInvSqrt2 * in[l]);
            }
        }
    }
}

extern "C" {

#define LIGHTAPE_DECLARE_GENERIC_UNARY(field)                                                      \
    void lightape_sleef_generic_##field(const double*, double*);
LIGHTAPE_MATH_UNARY_FUNCTIONS(LIGHTAPE_DECLARE_GENERIC_UNARY)
#undef LIGHTAPE_DECLARE_GENERIC_UNARY
#define LIGHTAPE_DECLARE_GENERIC_BINARY(field)                                                     \
    void lightape_sleef_generic_##field(const double*, const double*, double*);
LIGHTAPE_MATH_BINARY_FUNCTIONS(LIGHTAPE_DECLARE_GENERIC_BINARY)
#undef LIGHTAPE_DECLARE_GENERIC_BINARY
#define LIGHTAPE_DECLARE_GENERIC_SINCOS(field)                                                     \
    void lightape_sleef_generic_##field(const double*, double*, double*);
LIGHTAPE_MATH_SINCOS_FUNCTIONS(LIGHTAPE_DECLARE_GENERIC_SINCOS)
#undef LIGHTAPE_DECLARE_GENERIC_SINCOS

#define LIGHTAPE_DECLARE_AVX2128_UNARY(field)                                                      \
    void lightape_sleef_avx2128_##field(const double*, double*);
LIGHTAPE_MATH_UNARY_FUNCTIONS(LIGHTAPE_DECLARE_AVX2128_UNARY)
#undef LIGHTAPE_DECLARE_AVX2128_UNARY
#define LIGHTAPE_DECLARE_AVX2128_BINARY(field)                                                     \
    void lightape_sleef_avx2128_##field(const double*, const double*, double*);
LIGHTAPE_MATH_BINARY_FUNCTIONS(LIGHTAPE_DECLARE_AVX2128_BINARY)
#undef LIGHTAPE_DECLARE_AVX2128_BINARY
#define LIGHTAPE_DECLARE_AVX2128_SINCOS(field)                                                     \
    void lightape_sleef_avx2128_##field(const double*, double*, double*);
LIGHTAPE_MATH_SINCOS_FUNCTIONS(LIGHTAPE_DECLARE_AVX2128_SINCOS)
#undef LIGHTAPE_DECLARE_AVX2128_SINCOS

#define LIGHTAPE_DECLARE_AVX2_UNARY(field) void lightape_sleef_avx2_##field(const double*, double*);
LIGHTAPE_MATH_UNARY_FUNCTIONS(LIGHTAPE_DECLARE_AVX2_UNARY)
#undef LIGHTAPE_DECLARE_AVX2_UNARY
#define LIGHTAPE_DECLARE_AVX2_BINARY(field)                                                        \
    void lightape_sleef_avx2_##field(const double*, const double*, double*);
LIGHTAPE_MATH_BINARY_FUNCTIONS(LIGHTAPE_DECLARE_AVX2_BINARY)
#undef LIGHTAPE_DECLARE_AVX2_BINARY
#define LIGHTAPE_DECLARE_AVX2_SINCOS(field)                                                        \
    void lightape_sleef_avx2_##field(const double*, double*, double*);
LIGHTAPE_MATH_SINCOS_FUNCTIONS(LIGHTAPE_DECLARE_AVX2_SINCOS)
#undef LIGHTAPE_DECLARE_AVX2_SINCOS

#define LIGHTAPE_DECLARE_AVX512F_UNARY(field)                                                      \
    void lightape_sleef_avx512f_##field(const double*, double*);
LIGHTAPE_MATH_UNARY_FUNCTIONS(LIGHTAPE_DECLARE_AVX512F_UNARY)
#undef LIGHTAPE_DECLARE_AVX512F_UNARY
#define LIGHTAPE_DECLARE_AVX512F_BINARY(field)                                                     \
    void lightape_sleef_avx512f_##field(const double*, const double*, double*);
LIGHTAPE_MATH_BINARY_FUNCTIONS(LIGHTAPE_DECLARE_AVX512F_BINARY)
#undef LIGHTAPE_DECLARE_AVX512F_BINARY
#define LIGHTAPE_DECLARE_AVX512F_SINCOS(field)                                                     \
    void lightape_sleef_avx512f_##field(const double*, double*, double*);
LIGHTAPE_MATH_SINCOS_FUNCTIONS(LIGHTAPE_DECLARE_AVX512F_SINCOS)
#undef LIGHTAPE_DECLARE_AVX512F_SINCOS

} // extern "C"

namespace generic {
#define LIGHTAPE_ALIAS_GENERIC_UNARY(field)                                                        \
    inline constexpr UnaryKernel field##K = &lightape_sleef_generic_##field;
LIGHTAPE_MATH_UNARY_FUNCTIONS_ALIASABLE(LIGHTAPE_ALIAS_GENERIC_UNARY)
#undef LIGHTAPE_ALIAS_GENERIC_UNARY
#define LIGHTAPE_ALIAS_GENERIC_BINARY(field)                                                       \
    inline constexpr BinaryKernel field##K = &lightape_sleef_generic_##field;
LIGHTAPE_MATH_BINARY_FUNCTIONS(LIGHTAPE_ALIAS_GENERIC_BINARY)
#undef LIGHTAPE_ALIAS_GENERIC_BINARY
#define LIGHTAPE_ALIAS_GENERIC_SINCOS(field)                                                       \
    inline constexpr SinCosKernel field##K = &lightape_sleef_generic_##field;
LIGHTAPE_MATH_SINCOS_FUNCTIONS(LIGHTAPE_ALIAS_GENERIC_SINCOS)
#undef LIGHTAPE_ALIAS_GENERIC_SINCOS
inline void normCdfK(const double* in, double* out) {
    fusedNormCdfK<1, erfK>(in, out);
}
} // namespace generic

namespace avx2128 {
#define LIGHTAPE_ALIAS_AVX2128_UNARY(field)                                                        \
    inline constexpr UnaryKernel field##K = &lightape_sleef_avx2128_##field;
LIGHTAPE_MATH_UNARY_FUNCTIONS_ALIASABLE(LIGHTAPE_ALIAS_AVX2128_UNARY)
#undef LIGHTAPE_ALIAS_AVX2128_UNARY
#define LIGHTAPE_ALIAS_AVX2128_BINARY(field)                                                       \
    inline constexpr BinaryKernel field##K = &lightape_sleef_avx2128_##field;
LIGHTAPE_MATH_BINARY_FUNCTIONS(LIGHTAPE_ALIAS_AVX2128_BINARY)
#undef LIGHTAPE_ALIAS_AVX2128_BINARY
#define LIGHTAPE_ALIAS_AVX2128_SINCOS(field)                                                       \
    inline constexpr SinCosKernel field##K = &lightape_sleef_avx2128_##field;
LIGHTAPE_MATH_SINCOS_FUNCTIONS(LIGHTAPE_ALIAS_AVX2128_SINCOS)
#undef LIGHTAPE_ALIAS_AVX2128_SINCOS
inline void normCdfK(const double* in, double* out) {
    fusedNormCdfK<2, erfK>(in, out);
}
} // namespace avx2128

namespace avx2 {
#define LIGHTAPE_ALIAS_AVX2_UNARY(field)                                                           \
    inline constexpr UnaryKernel field##K = &lightape_sleef_avx2_##field;
LIGHTAPE_MATH_UNARY_FUNCTIONS_ALIASABLE(LIGHTAPE_ALIAS_AVX2_UNARY)
#undef LIGHTAPE_ALIAS_AVX2_UNARY
#define LIGHTAPE_ALIAS_AVX2_BINARY(field)                                                          \
    inline constexpr BinaryKernel field##K = &lightape_sleef_avx2_##field;
LIGHTAPE_MATH_BINARY_FUNCTIONS(LIGHTAPE_ALIAS_AVX2_BINARY)
#undef LIGHTAPE_ALIAS_AVX2_BINARY
#define LIGHTAPE_ALIAS_AVX2_SINCOS(field)                                                          \
    inline constexpr SinCosKernel field##K = &lightape_sleef_avx2_##field;
LIGHTAPE_MATH_SINCOS_FUNCTIONS(LIGHTAPE_ALIAS_AVX2_SINCOS)
#undef LIGHTAPE_ALIAS_AVX2_SINCOS
inline void normCdfK(const double* in, double* out) {
    fusedNormCdfK<4, erfK>(in, out);
}
} // namespace avx2

namespace avx512f {
#define LIGHTAPE_ALIAS_AVX512F_UNARY(field)                                                        \
    inline constexpr UnaryKernel field##K = &lightape_sleef_avx512f_##field;
LIGHTAPE_MATH_UNARY_FUNCTIONS_ALIASABLE(LIGHTAPE_ALIAS_AVX512F_UNARY)
#undef LIGHTAPE_ALIAS_AVX512F_UNARY
#define LIGHTAPE_ALIAS_AVX512F_BINARY(field)                                                       \
    inline constexpr BinaryKernel field##K = &lightape_sleef_avx512f_##field;
LIGHTAPE_MATH_BINARY_FUNCTIONS(LIGHTAPE_ALIAS_AVX512F_BINARY)
#undef LIGHTAPE_ALIAS_AVX512F_BINARY
#define LIGHTAPE_ALIAS_AVX512F_SINCOS(field)                                                       \
    inline constexpr SinCosKernel field##K = &lightape_sleef_avx512f_##field;
LIGHTAPE_MATH_SINCOS_FUNCTIONS(LIGHTAPE_ALIAS_AVX512F_SINCOS)
#undef LIGHTAPE_ALIAS_AVX512F_SINCOS
inline void normCdfK(const double* in, double* out) {
    fusedNormCdfK<8, erfK>(in, out);
}
} // namespace avx512f

#else // !(x86-64 GCC/Clang)

// ── SLEEF scalar plan (`generic`): width 1, portable `Sleef_*_u10/u05/u15`
// entry points from libsleef, for bisection against libm without an explicit
// ISA. Header-only on AArch64 (NEON baseline) and other SLEEF platforms. ──
namespace generic {

inline void sqrtK(const double* in, double* out) {
    *out = Sleef_sqrt_u05(*in);
}
inline void expK(const double* in, double* out) {
    *out = Sleef_exp_u10(*in);
}
inline void logK(const double* in, double* out) {
    *out = Sleef_log_u10(*in);
}
inline void expm1K(const double* in, double* out) {
    *out = Sleef_expm1_u10(*in);
}
inline void log1pK(const double* in, double* out) {
    *out = Sleef_log1p_u10(*in);
}
inline void sinK(const double* in, double* out) {
    *out = Sleef_sin_u10(*in);
}
inline void cosK(const double* in, double* out) {
    *out = Sleef_cos_u10(*in);
}
inline void tanK(const double* in, double* out) {
    *out = Sleef_tan_u10(*in);
}
inline void asinK(const double* in, double* out) {
    *out = Sleef_asin_u10(*in);
}
inline void acosK(const double* in, double* out) {
    *out = Sleef_acos_u10(*in);
}
inline void atanK(const double* in, double* out) {
    *out = Sleef_atan_u10(*in);
}
inline void sinhK(const double* in, double* out) {
    *out = Sleef_sinh_u10(*in);
}
inline void coshK(const double* in, double* out) {
    *out = Sleef_cosh_u10(*in);
}
inline void tanhK(const double* in, double* out) {
    *out = Sleef_tanh_u10(*in);
}
inline void exp2K(const double* in, double* out) {
    *out = Sleef_exp2_u10(*in);
}
inline void log2K(const double* in, double* out) {
    *out = Sleef_log2_u10(*in);
}
inline void log10K(const double* in, double* out) {
    *out = Sleef_log10_u10(*in);
}
inline void erfK(const double* in, double* out) {
    *out = Sleef_erf_u10(*in);
}
inline void erfcK(const double* in, double* out) {
    *out = Sleef_erfc_u15(*in);
}
inline void cbrtK(const double* in, double* out) {
    *out = Sleef_cbrt_u10(*in);
}
inline void powK(const double* a, const double* b, double* out) {
    *out = Sleef_pow_u10(*a, *b);
}
inline void atan2K(const double* y, const double* x, double* out) {
    *out = Sleef_atan2_u10(*y, *x);
}
inline void hypotK(const double* a, const double* b, double* out) {
    *out = Sleef_hypot_u05(*a, *b);
}
inline void sincosK(const double* in, double* s, double* c) {
    const Sleef_double2 r = Sleef_sincos_u10(*in);
    *s = r.x;
    *c = r.y;
}
inline void normPdfK(const double* in, double* out) {
    *out = kInvSqrt2Pi * Sleef_exp_u10(-0.5 * (*in * *in));
}
/// Fused normal CDF (see `kNormCdfSplit`): SLEEF scalar erf in the central
/// region, scalar libm `erfc` in the tails. On Apple Silicon the M2 bake-off
/// measured SLEEF's u15 scalar erfc ~8x slower than libm's while the libm tail
/// expression is bitwise equal to `lightape::normcdf`, so the rare tail lanes
/// stay on libm (same choice as the x86 `fusedNormCdfK`).
inline void normCdfK(const double* in, double* out) {
    const double x = *in;
    if (std::fabs(x) > kNormCdfSplit) [[unlikely]] {
        using std::erfc;
        *out = 0.5 * erfc(-kInvSqrt2 * x);
    } else {
        *out = 0.5 * Sleef_erf_u10(kInvSqrt2 * x) + 0.5;
    }
}

} // namespace generic

#endif // x86-64 GCC/Clang

// ── SLEEF explicit-ISA plan: AArch64 AdvSIMD, 2 lanes/step. Apple Silicon
// and every AArch64 part have AdvSIMD in the baseline, so no runtime probe. ──
#if defined(__aarch64__)

namespace advsimd {

inline void sqrtK(const double* in, double* out) {
    vst1q_f64(out, Sleef_sqrtd2_u05advsimd(vld1q_f64(in)));
}
inline void expK(const double* in, double* out) {
    vst1q_f64(out, Sleef_expd2_u10advsimd(vld1q_f64(in)));
}
inline void logK(const double* in, double* out) {
    vst1q_f64(out, Sleef_logd2_u10advsimd(vld1q_f64(in)));
}
inline void expm1K(const double* in, double* out) {
    vst1q_f64(out, Sleef_expm1d2_u10advsimd(vld1q_f64(in)));
}
inline void log1pK(const double* in, double* out) {
    vst1q_f64(out, Sleef_log1pd2_u10advsimd(vld1q_f64(in)));
}
inline void sinK(const double* in, double* out) {
    vst1q_f64(out, Sleef_sind2_u10advsimd(vld1q_f64(in)));
}
inline void cosK(const double* in, double* out) {
    vst1q_f64(out, Sleef_cosd2_u10advsimd(vld1q_f64(in)));
}
inline void tanK(const double* in, double* out) {
    vst1q_f64(out, Sleef_tand2_u10advsimd(vld1q_f64(in)));
}
inline void asinK(const double* in, double* out) {
    vst1q_f64(out, Sleef_asind2_u10advsimd(vld1q_f64(in)));
}
inline void acosK(const double* in, double* out) {
    vst1q_f64(out, Sleef_acosd2_u10advsimd(vld1q_f64(in)));
}
inline void atanK(const double* in, double* out) {
    vst1q_f64(out, Sleef_atand2_u10advsimd(vld1q_f64(in)));
}
inline void sinhK(const double* in, double* out) {
    vst1q_f64(out, Sleef_sinhd2_u10advsimd(vld1q_f64(in)));
}
inline void coshK(const double* in, double* out) {
    vst1q_f64(out, Sleef_coshd2_u10advsimd(vld1q_f64(in)));
}
inline void tanhK(const double* in, double* out) {
    vst1q_f64(out, Sleef_tanhd2_u10advsimd(vld1q_f64(in)));
}
inline void exp2K(const double* in, double* out) {
    vst1q_f64(out, Sleef_exp2d2_u10advsimd(vld1q_f64(in)));
}
inline void log2K(const double* in, double* out) {
    vst1q_f64(out, Sleef_log2d2_u10advsimd(vld1q_f64(in)));
}
inline void log10K(const double* in, double* out) {
    vst1q_f64(out, Sleef_log10d2_u10advsimd(vld1q_f64(in)));
}
inline void erfK(const double* in, double* out) {
    vst1q_f64(out, Sleef_erfd2_u10advsimd(vld1q_f64(in)));
}
inline void erfcK(const double* in, double* out) {
    vst1q_f64(out, Sleef_erfcd2_u15advsimd(vld1q_f64(in)));
}
inline void cbrtK(const double* in, double* out) {
    vst1q_f64(out, Sleef_cbrtd2_u10advsimd(vld1q_f64(in)));
}
inline void powK(const double* a, const double* b, double* out) {
    vst1q_f64(out, Sleef_powd2_u10advsimd(vld1q_f64(a), vld1q_f64(b)));
}
inline void atan2K(const double* y, const double* x, double* out) {
    vst1q_f64(out, Sleef_atan2d2_u10advsimd(vld1q_f64(y), vld1q_f64(x)));
}
inline void hypotK(const double* a, const double* b, double* out) {
    vst1q_f64(out, Sleef_hypotd2_u05advsimd(vld1q_f64(a), vld1q_f64(b)));
}
inline void sincosK(const double* in, double* s, double* c) {
    const Sleef_float64x2_t_2 r = Sleef_sincosd2_u10advsimd(vld1q_f64(in));
    vst1q_f64(s, r.x);
    vst1q_f64(c, r.y);
}
inline void normPdfK(const double* in, double* out) {
    const float64x2_t x = vld1q_f64(in);
    const float64x2_t arg = vmulq_n_f64(vmulq_f64(x, x), -0.5);
    vst1q_f64(out, vmulq_n_f64(Sleef_expd2_u10advsimd(arg), kInvSqrt2Pi));
}
/// Fused normal CDF (see `kNormCdfSplit`): AdvSIMD erf in the central region
/// (`vfma` keeps the expression's single rounding) and scalar libm `erfc` in
/// the tails. The M2 bake-off measured AdvSIMD erf well ahead of the old
/// u15-erfc composition, while the wide AdvSIMD erfc is ~3x slower than one
/// scalar libm call and the tail branch is rare (unreachable on the timing
/// corpus), so the tail lanes go to libm; the libm tail expression is bitwise
/// equal to `lightape::normcdf`.
inline void normCdfK(const double* in, double* out) {
    const float64x2_t x = vld1q_f64(in);
    const float64x2_t central =
        vfmaq_n_f64(vdupq_n_f64(0.5), Sleef_erfd2_u10advsimd(vmulq_n_f64(x, kInvSqrt2)), 0.5);
    vst1q_f64(out, central);
    if (std::fabs(in[0]) > kNormCdfSplit || std::fabs(in[1]) > kNormCdfSplit) [[unlikely]] {
        using std::erfc;
        for (std::size_t l = 0; l < 2; ++l) {
            if (std::fabs(in[l]) > kNormCdfSplit) {
                out[l] = 0.5 * erfc(-kInvSqrt2 * in[l]);
            }
        }
    }
}

} // namespace advsimd

#endif // __aarch64__

#endif // LIGHTAPE_VECTOR_MATH_SLEEF

} // namespace detail

/// Concrete implementation that supplied a resolved table entry. `Auto` is
/// only a request value (inherit), never resolved.
enum class MathImpl { Auto, Libm, Sleef, Generic };

/// Every selectable `v*` function, in table order. Enumerator names match the
/// `MathTable` member names so the resolver macros can index by `MathFn::name`.
enum class MathFn : std::size_t {
    sqrt,
    exp,
    log,
    expm1,
    log1p,
    sin,
    cos,
    tan,
    asin,
    acos,
    atan,
    sinh,
    cosh,
    tanh,
    exp2,
    log2,
    log10,
    erf,
    erfc,
    cbrt,
    pow,
    atan2,
    hypot,
    sincos,
    normPdf,
    normCdf,
    Count
};

inline constexpr std::size_t kMathFunctionCount = static_cast<std::size_t>(MathFn::Count);

/// Environment names, aligned with `MathFn` (`LIGHTAPE_MATH_IMPL_<name>`).
inline constexpr const char* kMathFunctionNames[kMathFunctionCount] = {
    "vSqrt", "vExp",  "vLog",  "vExpm1", "vLog1p", "vSin",    "vCos",     "vTan",    "vAsin",
    "vAcos", "vAtan", "vSinh", "vCosh",  "vTanh",  "vExp2",   "vLog2",    "vLog10",  "vErf",
    "vErfc", "vCbrt", "vPow",  "vAtan2", "vHypot", "vSinCos", "vNormPdf", "vNormCdf"};

/// A resolved unary function: `impl` ∈ {"libm","sleef","generic"},
/// `isa` ∈ {"libm","generic","advsimd","avx2128","avx2","avx512f"}, one `fn`
/// call consumes exactly `width` lanes. `next` is the next-narrower candidate
/// for `VectorMath<W>` with `W < width` (the x86 ladder is
/// avx512f -> avx2 -> avx2128 -> libm; AArch64 is advsimd -> libm); it is
/// `nullptr` for the width-1 libm/generic end of the chain.
struct UnaryEntry {
    const char* impl;
    const char* isa;
    std::size_t width;
    detail::UnaryKernel fn;
    const UnaryEntry* next = nullptr;
};

/// A resolved binary function (both operands arrays); see `UnaryEntry`.
struct BinaryEntry {
    const char* impl;
    const char* isa;
    std::size_t width;
    detail::BinaryKernel fn;
    const BinaryEntry* next = nullptr;
};

/// A resolved dual-output sin/cos entry; see `UnaryEntry`.
struct SinCosEntry {
    const char* impl;
    const char* isa;
    std::size_t width;
    detail::SinCosKernel fn;
    const SinCosEntry* next = nullptr;
};

/// The nearest-width selector of the `VectorMath<W>` kernels: walk the
/// candidate chain from widest to narrowest and return the first entry whose
/// `width` fits `W` (`W >= 1`, so a well-formed chain always terminates at the
/// width-1 libm entry). Bounded walk, no allocation, no environment access.
/// Exposed for the resolution tests and the bake-off, e.g.
/// `selectEntry<4>(mathTable().exp)` reports the avx2 entry on an AVX-512
/// machine.
template <std::size_t W, typename Entry>
const Entry* selectEntry(const Entry& entry) {
    const Entry* e = &entry;
    while (e != nullptr && e->width > W) {
        e = e->next;
    }
    return e;
}

/// The process-wide per-function dispatch table. Every entry is
/// default-initialized to the `LibmScalar` kernel, so a default-constructed
/// table is the bitwise reference; `resolveMathTable()` rewires the entries
/// the selection asks for. `name`/`width` are reporting summaries only (a
/// single impl/ISA name when every entry resolved alike, else "mixed";
/// `width` is the widest entry step).
struct MathTable {
    const char* name = "libm";
    std::size_t width = 1;

    UnaryEntry sqrt{"libm", "libm", 1, detail::libm1::sqrtK};
    UnaryEntry exp{"libm", "libm", 1, detail::libm1::expK};
    UnaryEntry log{"libm", "libm", 1, detail::libm1::logK};
    UnaryEntry expm1{"libm", "libm", 1, detail::libm1::expm1K};
    UnaryEntry log1p{"libm", "libm", 1, detail::libm1::log1pK};
    UnaryEntry sin{"libm", "libm", 1, detail::libm1::sinK};
    UnaryEntry cos{"libm", "libm", 1, detail::libm1::cosK};
    UnaryEntry tan{"libm", "libm", 1, detail::libm1::tanK};
    UnaryEntry asin{"libm", "libm", 1, detail::libm1::asinK};
    UnaryEntry acos{"libm", "libm", 1, detail::libm1::acosK};
    UnaryEntry atan{"libm", "libm", 1, detail::libm1::atanK};
    UnaryEntry sinh{"libm", "libm", 1, detail::libm1::sinhK};
    UnaryEntry cosh{"libm", "libm", 1, detail::libm1::coshK};
    UnaryEntry tanh{"libm", "libm", 1, detail::libm1::tanhK};
    // Inverse hyperbolic: libm-only ladder rung for now (no SLEEF candidate;
    // see the file comment). The resolver never rewires these entries, so they
    // honestly report `impl == "libm"` in every candidate table.
    UnaryEntry asinh{"libm", "libm", 1, detail::libm1::asinhK};
    UnaryEntry acosh{"libm", "libm", 1, detail::libm1::acoshK};
    UnaryEntry atanh{"libm", "libm", 1, detail::libm1::atanhK};
    UnaryEntry exp2{"libm", "libm", 1, detail::libm1::exp2K};
    UnaryEntry log2{"libm", "libm", 1, detail::libm1::log2K};
    UnaryEntry log10{"libm", "libm", 1, detail::libm1::log10K};
    UnaryEntry erf{"libm", "libm", 1, detail::libm1::erfK};
    UnaryEntry erfc{"libm", "libm", 1, detail::libm1::erfcK};
    UnaryEntry cbrt{"libm", "libm", 1, detail::libm1::cbrtK};
    UnaryEntry normPdf{"libm", "libm", 1, detail::libm1::normPdfK};
    UnaryEntry normCdf{"libm", "libm", 1, detail::libm1::normCdfK};
    BinaryEntry pow{"libm", "libm", 1, detail::libm1::powK};
    BinaryEntry atan2{"libm", "libm", 1, detail::libm1::atan2K};
    BinaryEntry hypot{"libm", "libm", 1, detail::libm1::hypotK};
    SinCosEntry sincos{"libm", "libm", 1, detail::libm1::sincosK};
};

/// The all-libm candidate table (bitwise reference).
inline const MathTable& libmTable() {
    static const MathTable kTable; // default member initializers: LibmScalar
    return kTable;
}

// ── candidate-table construction: each implementation namespace maps to a
// table whose entries differ only in the kernel pointer; the X-macro lists at
// the top of `detail` are the single source of truth. ──
#define LIGHTAPE_SET_UNARY_ENTRY(field) t.field = UnaryEntry{implStr, isaStr, width, ns::field##K};
#define LIGHTAPE_SET_BINARY_ENTRY(field)                                                           \
    t.field = BinaryEntry{implStr, isaStr, width, ns::field##K};
#define LIGHTAPE_SET_SINCOS_ENTRY(field)                                                           \
    t.sincos = SinCosEntry{implStr, isaStr, width, ns::sincosK};

#define LIGHTAPE_SET_ALL_ENTRIES                                                                   \
    LIGHTAPE_MATH_UNARY_FUNCTIONS(LIGHTAPE_SET_UNARY_ENTRY)                                        \
    LIGHTAPE_MATH_BINARY_FUNCTIONS(LIGHTAPE_SET_BINARY_ENTRY)                                      \
    LIGHTAPE_MATH_SINCOS_FUNCTIONS(LIGHTAPE_SET_SINCOS_ENTRY)

#ifdef LIGHTAPE_VECTOR_MATH_SLEEF

// ── nearest-width fallback wiring: each candidate table links every entry to
// the same function's entry in the next-narrower table, so one process-wide
// resolution holds the whole ladder. x86-64: avx512f -> avx2 -> avx2128 ->
// libm; AArch64: advsimd -> libm. ──
#define LIGHTAPE_LINK_UNARY(field)  t.field.next = &lower.field;
#define LIGHTAPE_LINK_BINARY(field) t.field.next = &lower.field;
#define LIGHTAPE_LINK_SINCOS(field) t.sincos.next = &lower.sincos;
#define LIGHTAPE_LINK_FALLBACKS(lower)                                                             \
    LIGHTAPE_MATH_UNARY_FUNCTIONS(LIGHTAPE_LINK_UNARY)                                             \
    LIGHTAPE_MATH_BINARY_FUNCTIONS(LIGHTAPE_LINK_BINARY)                                           \
    LIGHTAPE_MATH_SINCOS_FUNCTIONS(LIGHTAPE_LINK_SINCOS)

/// SLEEF's portable scalar plan, width 1 (u10/u05/u15 tiers). Explicitly
/// selectable (`generic`); it needs no fallback because width 1 fits every W.
inline const MathTable& genericTable() {
    static const MathTable kTable = [] {
        constexpr const char* implStr = "generic";
        constexpr const char* isaStr = "generic";
        constexpr std::size_t width = 1;
        namespace ns = detail::generic;
        MathTable t;
        t.name = isaStr;
        t.width = width;
        LIGHTAPE_SET_ALL_ENTRIES
        return t;
    }();
    return kTable;
}

#if defined(__aarch64__)

/// Apple Silicon / AArch64 AdvSIMD plan, width 2 (baseline ISA, no probe).
inline const MathTable& advsimdTable() {
    static const MathTable kTable = [] {
        constexpr const char* implStr = "sleef";
        constexpr const char* isaStr = "advsimd";
        constexpr std::size_t width = 2;
        namespace ns = detail::advsimd;
        MathTable t;
        t.name = isaStr;
        t.width = width;
        LIGHTAPE_SET_ALL_ENTRIES
        const MathTable& lower = libmTable();
        LIGHTAPE_LINK_FALLBACKS(lower)
        return t;
    }();
    return kTable;
}

#endif // __aarch64__

#if defined(__x86_64__) && (defined(__GNUC__) || defined(__clang__))

/// x86-64 AVX2128 (128-bit FMA) plan, width 2: the bottom SLEEF rung.
inline const MathTable& avx2128Table() {
    static const MathTable kTable = [] {
        constexpr const char* implStr = "sleef";
        constexpr const char* isaStr = "avx2128";
        constexpr std::size_t width = 2;
        namespace ns = detail::avx2128;
        MathTable t;
        t.name = isaStr;
        t.width = width;
        LIGHTAPE_SET_ALL_ENTRIES
        const MathTable& lower = libmTable();
        LIGHTAPE_LINK_FALLBACKS(lower)
        return t;
    }();
    return kTable;
}

/// x86-64 AVX2 + FMA plan, width 4.
inline const MathTable& avx2Table() {
    static const MathTable kTable = [] {
        constexpr const char* implStr = "sleef";
        constexpr const char* isaStr = "avx2";
        constexpr std::size_t width = 4;
        namespace ns = detail::avx2;
        MathTable t;
        t.name = isaStr;
        t.width = width;
        LIGHTAPE_SET_ALL_ENTRIES
        const MathTable& lower = avx2128Table();
        LIGHTAPE_LINK_FALLBACKS(lower)
        return t;
    }();
    return kTable;
}

/// x86-64 AVX-512F plan, width 8.
inline const MathTable& avx512fTable() {
    static const MathTable kTable = [] {
        constexpr const char* implStr = "sleef";
        constexpr const char* isaStr = "avx512f";
        constexpr std::size_t width = 8;
        namespace ns = detail::avx512f;
        MathTable t;
        t.name = isaStr;
        t.width = width;
        LIGHTAPE_SET_ALL_ENTRIES
        const MathTable& lower = avx2Table();
        LIGHTAPE_LINK_FALLBACKS(lower)
        return t;
    }();
    return kTable;
}

#endif // x86-64 GCC/Clang

/// The concrete SLEEF ladder selected by `LIGHTAPE_VECTOR_ISA`: the requested
/// ISA when the CPU supports it, `auto`/unset = the widest supported ISA, an
/// unknown request = the portable scalar plan (`libm` stays spellable). The
/// returned table's entries carry the `next` chain down to the width-1 libm
/// tail, so `selectEntry<W>` never needs another lookup.
inline const MathTable& isaTable() {
    const char* isa = std::getenv("LIGHTAPE_VECTOR_ISA");
    if (isa != nullptr && std::strcmp(isa, "auto") == 0) {
        isa = nullptr;
    }
    if (isa != nullptr) {
#if defined(__aarch64__)
        if (std::strcmp(isa, "advsimd") == 0) {
            return advsimdTable();
        }
#endif
#if defined(__x86_64__) && (defined(__GNUC__) || defined(__clang__))
        __builtin_cpu_init();
        const bool has_fma = __builtin_cpu_supports("fma");
        if (std::strcmp(isa, "avx512f") == 0 && __builtin_cpu_supports("avx512f")) {
            return avx512fTable();
        }
        if (std::strcmp(isa, "avx2") == 0 && __builtin_cpu_supports("avx2") && has_fma) {
            return avx2Table();
        }
        if (std::strcmp(isa, "avx2128") == 0 && __builtin_cpu_supports("avx2") && has_fma) {
            return avx2128Table();
        }
#endif
        if (std::strcmp(isa, "libm") == 0) {
            return libmTable();
        }
        return genericTable();
    }
#if defined(__x86_64__) && (defined(__GNUC__) || defined(__clang__))
    __builtin_cpu_init();
    if (__builtin_cpu_supports("avx512f")) {
        return avx512fTable();
    }
    if (__builtin_cpu_supports("avx2") && __builtin_cpu_supports("fma")) {
        return avx2Table();
    }
    // No vector rung fits: the `sleef`/`auto` request honestly resolves to
    // libm. The portable SLEEF scalar plan stays selectable via the explicit
    // `generic` value (LIGHTAPE_VECTOR_ISA=generic / LIGHTAPE_MATH_TABLE).
    return libmTable();
#elif defined(__aarch64__)
    return advsimdTable();
#else
    return genericTable();
#endif
}

#else // !LIGHTAPE_VECTOR_MATH_SLEEF

/// Without SLEEF compiled in, every non-libm request falls back to the libm
/// candidate (documented behavior; the resolved entries report `impl ==
/// "libm"`).
inline const MathTable& genericTable() {
    return libmTable();
}
inline const MathTable& isaTable() {
    return libmTable();
}

#endif // LIGHTAPE_VECTOR_MATH_SLEEF

#undef LIGHTAPE_LINK_FALLBACKS
#undef LIGHTAPE_LINK_UNARY
#undef LIGHTAPE_LINK_BINARY
#undef LIGHTAPE_LINK_SINCOS
#undef LIGHTAPE_SET_ALL_ENTRIES
#undef LIGHTAPE_SET_UNARY_ENTRY
#undef LIGHTAPE_SET_BINARY_ENTRY
#undef LIGHTAPE_SET_SINCOS_ENTRY

/// Parse a `LIGHTAPE_MATH_*` value; `nullptr`/unknown/`auto` -> `Auto`.
inline MathImpl parseMathImpl(const char* value) {
    if (value == nullptr) {
        return MathImpl::Auto;
    }
    if (std::strcmp(value, "libm") == 0) {
        return MathImpl::Libm;
    }
    if (std::strcmp(value, "sleef") == 0) {
        return MathImpl::Sleef;
    }
    if (std::strcmp(value, "generic") == 0) {
        return MathImpl::Generic;
    }
    return MathImpl::Auto;
}

/// Spelling of a resolved implementation (reporting).
inline const char* mathImplName(MathImpl impl) {
    switch (impl) {
        case MathImpl::Libm:
            return "libm";
        case MathImpl::Sleef:
            return "sleef";
        case MathImpl::Generic:
            return "generic";
        case MathImpl::Auto:
            break;
    }
    return "auto";
}

/// Env-free request: table-wide choice + per-function overrides + force flag.
struct MathSelection {
    /// `LIGHTAPE_MATH_TABLE`; `Auto` follows the per-platform policy.
    MathImpl table = MathImpl::Auto;
    /// `LIGHTAPE_MATH_IMPL_<vFn>`; `Auto` inherits `table` (`MathFn` index).
    std::array<MathImpl, kMathFunctionCount> perFunction{};
    /// `LIGHTAPE_VECTOR_MATH_FORCE_LIBM=1`: everything libm (reference mode).
    bool forceLibm = false;
};

#if defined(__aarch64__)

/// The AArch64 `auto` SLEEF set: empty by default. The M2 evidence is
/// corpus-dependent (the V1-style raw probe shows libm winning/tying
/// everywhere and does not reproduce the V1 `log` claim; the broader
/// bake-off corpus shows a few AdvSIMD wins), so the conservative default
/// stays libm until `benchmarks/bench_vector_math.cpp` is run on the
/// deployment workload and the list (e.g. `MathFn::log`, `MathFn::erf`) is
/// filled from measured wins. The fused `vNormCdf` is the one entry measured
/// in: two 2026-10-08 `lightape_vector_math_bench` runs put it at 9.43/9.48
/// ns/value at W=8 versus 11.14/11.25 for the auto libm candidate (>5% in the
/// harness's decision row, which is what files the suggested override), so it
/// joins this list. Caveat recorded: against the *direct* scalar libm
/// expression (9.19/9.22, no table dispatch) the fused kernel is a tie inside
/// the noise floor; the win is over the auto table path as measured.
inline constexpr std::array<MathFn, 1> kAutoSleefAarch64{MathFn::normCdf};

/// Evidence-based per-platform `auto` choice for one function.
inline MathImpl autoChoice(MathFn fn) {
    for (MathFn candidate : kAutoSleefAarch64) {
        if (candidate == fn) {
            return MathImpl::Sleef;
        }
    }
    return MathImpl::Libm;
}

#elif defined(__x86_64__)

/// The x86-64 `auto` libm exceptions, from the 2026-10 AWS bake-off (W=8
/// ns/value, SLEEF AVX-512 vs the direct scalar libm reference):
///   - `vErfc`: Intel 9.45 vs 8.90, AMD 7.85 vs 8.74 -> libm (SLEEF's u15
///     erfc is a tie at best and usually slower).
/// `vNormCdf` left this list when the fused kernel landed: the same bake-off
/// measured SLEEF AVX-512 erf at 3.28 (Intel) / 2.87 (AMD) ns/value versus
/// 5.88/5.69 direct scalar libm normcdf, and the fused kernel is exactly that
/// erf entry plus a scale/blend and a tail branch unreachable on the timing
/// corpus. The fused kernel itself has not been timed on x86 (the bake-off
/// predates it), so the x86 win is extrapolated from its `vErf` component,
/// not directly measured here.
/// Every other function resolves to SLEEF (2.5-4.5x wins). Per-function
/// `LIGHTAPE_MATH_IMPL_<vFn>` overrides still win over this list, and
/// `LIGHTAPE_MATH_TABLE=sleef` forces SLEEF for the exception too.
inline constexpr std::array<MathFn, 1> kAutoLibmX86{MathFn::erfc};

/// Evidence-based per-platform `auto` choice for one function.
inline MathImpl autoChoice(MathFn fn) {
    for (MathFn candidate : kAutoLibmX86) {
        if (candidate == fn) {
            return MathImpl::Libm;
        }
    }
    return MathImpl::Sleef;
}

#else

inline MathImpl autoChoice(MathFn) {
    return MathImpl::Libm;
}

#endif

/// Resolution order (documented at the top of this file): force-libm, then
/// the per-function override, then the table-wide choice, then `auto`.
inline MathImpl effectiveChoice(const MathSelection& sel, MathFn fn) {
    if (sel.forceLibm) {
        return MathImpl::Libm;
    }
    if (const MathImpl per = sel.perFunction[static_cast<std::size_t>(fn)]; per != MathImpl::Auto) {
        return per;
    }
    switch (sel.table) {
        case MathImpl::Libm:
        case MathImpl::Sleef:
        case MathImpl::Generic:
            return sel.table;
        case MathImpl::Auto:
            break;
    }
    return autoChoice(fn);
}

/// Parse every `LIGHTAPE_MATH_*` variable once; env-free resolution happens in
/// `resolveMathTable()` so tests and the bake-off can drive explicit requests.
inline MathSelection selectionFromEnv() {
    MathSelection sel;
    if (const char* force = std::getenv("LIGHTAPE_VECTOR_MATH_FORCE_LIBM")) {
        sel.forceLibm = std::strcmp(force, "1") == 0;
    }
    sel.table = parseMathImpl(std::getenv("LIGHTAPE_MATH_TABLE"));
    for (std::size_t i = 0; i < kMathFunctionCount; ++i) {
        char var[64];
        std::snprintf(var, sizeof(var), "LIGHTAPE_MATH_IMPL_%s", kMathFunctionNames[i]);
        sel.perFunction[i] = parseMathImpl(std::getenv(var));
    }
    return sel;
}

namespace detail {

/// Reporting helper: does `entry` come from the same candidate as `impl`/`isa`?
template <typename Entry>
bool sameShapeAs(const Entry& entry, const char* impl, const char* isa) {
    return std::strcmp(entry.impl, impl) == 0 && std::strcmp(entry.isa, isa) == 0;
}

} // namespace detail

/// Build a concrete table from a selection: each entry is copied from the
/// libm, SLEEF-ISA or SLEEF-generic candidate. Pure (no env access), so tests
/// and the bake-off can resolve arbitrary requests.
inline MathTable resolveMathTable(const MathSelection& sel) {
    const MathTable& libm = libmTable();
    const MathTable& isa = isaTable();
    const MathTable& gen = genericTable();
    const auto pickUnary = [](MathImpl c, const UnaryEntry& l, const UnaryEntry& s,
                              const UnaryEntry& g) -> UnaryEntry {
        switch (c) {
            case MathImpl::Sleef:
                return s;
            case MathImpl::Generic:
                return g;
            case MathImpl::Libm:
            case MathImpl::Auto:
                break;
        }
        return l;
    };
    const auto pickBinary = [](MathImpl c, const BinaryEntry& l, const BinaryEntry& s,
                               const BinaryEntry& g) -> BinaryEntry {
        switch (c) {
            case MathImpl::Sleef:
                return s;
            case MathImpl::Generic:
                return g;
            case MathImpl::Libm:
            case MathImpl::Auto:
                break;
        }
        return l;
    };
    const auto pickSinCos = [](MathImpl c, const SinCosEntry& l, const SinCosEntry& s,
                               const SinCosEntry& g) -> SinCosEntry {
        switch (c) {
            case MathImpl::Sleef:
                return s;
            case MathImpl::Generic:
                return g;
            case MathImpl::Libm:
            case MathImpl::Auto:
                break;
        }
        return l;
    };

    MathTable out = libm;
    std::size_t width = 1;
#define LIGHTAPE_RESOLVE_UNARY(field)                                                              \
    do {                                                                                           \
        out.field =                                                                                \
            pickUnary(effectiveChoice(sel, MathFn::field), libm.field, isa.field, gen.field);      \
        if (out.field.width > width) {                                                             \
            width = out.field.width;                                                               \
        }                                                                                          \
    } while (false);
#define LIGHTAPE_RESOLVE_BINARY(field)                                                             \
    do {                                                                                           \
        out.field =                                                                                \
            pickBinary(effectiveChoice(sel, MathFn::field), libm.field, isa.field, gen.field);     \
        if (out.field.width > width) {                                                             \
            width = out.field.width;                                                               \
        }                                                                                          \
    } while (false);
#define LIGHTAPE_RESOLVE_SINCOS(field)                                                             \
    do {                                                                                           \
        out.sincos =                                                                               \
            pickSinCos(effectiveChoice(sel, MathFn::field), libm.sincos, isa.sincos, gen.sincos);  \
        if (out.sincos.width > width) {                                                            \
            width = out.sincos.width;                                                              \
        }                                                                                          \
    } while (false);
    LIGHTAPE_MATH_UNARY_FUNCTIONS(LIGHTAPE_RESOLVE_UNARY)
    LIGHTAPE_MATH_BINARY_FUNCTIONS(LIGHTAPE_RESOLVE_BINARY)
    LIGHTAPE_MATH_SINCOS_FUNCTIONS(LIGHTAPE_RESOLVE_SINCOS)
#undef LIGHTAPE_RESOLVE_UNARY
#undef LIGHTAPE_RESOLVE_BINARY
#undef LIGHTAPE_RESOLVE_SINCOS

    out.width = width;
    bool uniform = true;
#define LIGHTAPE_UNIFORM_UNARY(field)                                                              \
    uniform = uniform && detail::sameShapeAs(out.field, out.sqrt.impl, out.sqrt.isa);
#define LIGHTAPE_UNIFORM_BINARY(field)                                                             \
    uniform = uniform && detail::sameShapeAs(out.field, out.sqrt.impl, out.sqrt.isa);
#define LIGHTAPE_UNIFORM_SINCOS(field)                                                             \
    uniform = uniform && detail::sameShapeAs(out.sincos, out.sqrt.impl, out.sqrt.isa);
    LIGHTAPE_MATH_UNARY_FUNCTIONS(LIGHTAPE_UNIFORM_UNARY)
    LIGHTAPE_MATH_BINARY_FUNCTIONS(LIGHTAPE_UNIFORM_BINARY)
    LIGHTAPE_MATH_SINCOS_FUNCTIONS(LIGHTAPE_UNIFORM_SINCOS)
#undef LIGHTAPE_UNIFORM_UNARY
#undef LIGHTAPE_UNIFORM_BINARY
#undef LIGHTAPE_UNIFORM_SINCOS
    if (!uniform) {
        out.name = "mixed";
    } else if (std::strcmp(out.sqrt.impl, "libm") == 0) {
        out.name = "libm";
    } else if (std::strcmp(out.sqrt.impl, "sleef") == 0) {
        out.name = out.sqrt.isa;
    } else {
        out.name = "generic";
    }
    return out;
}

#undef LIGHTAPE_MATH_UNARY_FUNCTIONS
#undef LIGHTAPE_MATH_UNARY_FUNCTIONS_ALIASABLE
#undef LIGHTAPE_MATH_BINARY_FUNCTIONS
#undef LIGHTAPE_MATH_SINCOS_FUNCTIONS

namespace detail {

/// Process-lifetime table slot: the hot path is one acquire load, and a
/// benign resolve race stores equivalent tables (the loser is leaked, like
/// the winner). Never null after the first `mathTable()` call.
inline std::atomic<const MathTable*> g_mathTable{nullptr};

} // namespace detail

/// Resolve the environment selection once per process.
inline const MathTable& mathTable() {
    const MathTable* table = detail::g_mathTable.load(std::memory_order_acquire);
    if (table == nullptr) [[unlikely]] {
        // Intentionally leaked: one process-lifetime table.
        table = new MathTable(resolveMathTable(selectionFromEnv()));
        detail::g_mathTable.store(table, std::memory_order_release);
    }
    return *table;
}

#ifdef LIGHTAPE_VECTOR_MATH_TESTING
/// Test/bake-off hook: install `table` as the process table (`nullptr` clears
/// the slot so the next call re-resolves from the environment). Only compiled
/// when the consumer defines `LIGHTAPE_VECTOR_MATH_TESTING`.
inline void setMathTableForTesting(const MathTable* table) {
    detail::g_mathTable.store(table, std::memory_order_release);
}
#endif

template <std::size_t W>
struct VectorMath {
    static_assert(W >= 1, "VectorMath lane width must be positive");

    /// Compile-time build tag; the concrete per-function table resolves at
    /// runtime (`isaName()` reports the summary).
    static constexpr MathBackend backend =
#ifdef LIGHTAPE_VECTOR_MATH_SLEEF
        MathBackend::Sleef;
#else
        MathBackend::LibmScalar;
#endif

    /// Widest resolved entry step across the table (per-function widths vary;
    /// inspect `mathTable()` for the exact per-function resolution).
    static std::size_t vWidth() { return mathTable().width; }

    /// Resolved table summary: one ISA/impl name when every entry resolved
    /// alike, else "mixed". Cross-table bitwise equality is not promised.
    static const char* isaName() { return mathTable().name; }

    // ── unary lane kernels ──

    static void vSqrt(const double* in, double* out) {
        applyUnary<&MathTable::sqrt, detail::libm1::sqrtK>(in, out);
    }

    static void vExp(const double* in, double* out) {
        applyUnary<&MathTable::exp, detail::libm1::expK>(in, out);
    }

    static void vLog(const double* in, double* out) {
        applyUnary<&MathTable::log, detail::libm1::logK>(in, out);
    }

    static void vExpm1(const double* in, double* out) {
        applyUnary<&MathTable::expm1, detail::libm1::expm1K>(in, out);
    }

    static void vLog1p(const double* in, double* out) {
        applyUnary<&MathTable::log1p, detail::libm1::log1pK>(in, out);
    }

    static void vSin(const double* in, double* out) {
        applyUnary<&MathTable::sin, detail::libm1::sinK>(in, out);
    }

    static void vCos(const double* in, double* out) {
        applyUnary<&MathTable::cos, detail::libm1::cosK>(in, out);
    }

    /// sin and cos in one call. Vector steps evaluate both through the
    /// nearest-width `sincos` candidate; the scalar tail keeps the libm
    /// reference order (sin before cos, per lane).
    static void vSinCos(const double* in, double* s, double* c) {
        const SinCosEntry* e = selectEntry<W>(mathTable().sincos);
        std::size_t l = 0;
        if (e != nullptr) {
            for (; l + e->width <= W; l += e->width) {
                e->fn(in + l, s + l, c + l);
            }
        }
        for (; l < W; ++l) {
            detail::libm1::sincosK(in + l, s + l, c + l);
        }
    }

    static void vTan(const double* in, double* out) {
        applyUnary<&MathTable::tan, detail::libm1::tanK>(in, out);
    }

    static void vAsin(const double* in, double* out) {
        applyUnary<&MathTable::asin, detail::libm1::asinK>(in, out);
    }

    static void vAcos(const double* in, double* out) {
        applyUnary<&MathTable::acos, detail::libm1::acosK>(in, out);
    }

    static void vAtan(const double* in, double* out) {
        applyUnary<&MathTable::atan, detail::libm1::atanK>(in, out);
    }

    static void vSinh(const double* in, double* out) {
        applyUnary<&MathTable::sinh, detail::libm1::sinhK>(in, out);
    }

    static void vCosh(const double* in, double* out) {
        applyUnary<&MathTable::cosh, detail::libm1::coshK>(in, out);
    }

    static void vTanh(const double* in, double* out) {
        applyUnary<&MathTable::tanh, detail::libm1::tanhK>(in, out);
    }

    /// Inverse hyperbolic lane values. The per-function entry currently
    /// resolves to the single width-1 libm rung (no SLEEF kernel yet), so
    /// every lane is one `std::asinh`/`std::acosh`/`std::atanh` call.
    static void vAsinh(const double* in, double* out) {
        applyUnary<&MathTable::asinh, detail::libm1::asinhK>(in, out);
    }

    static void vAcosh(const double* in, double* out) {
        applyUnary<&MathTable::acosh, detail::libm1::acoshK>(in, out);
    }

    static void vAtanh(const double* in, double* out) {
        applyUnary<&MathTable::atanh, detail::libm1::atanhK>(in, out);
    }

    static void vExp2(const double* in, double* out) {
        applyUnary<&MathTable::exp2, detail::libm1::exp2K>(in, out);
    }

    static void vLog2(const double* in, double* out) {
        applyUnary<&MathTable::log2, detail::libm1::log2K>(in, out);
    }

    static void vLog10(const double* in, double* out) {
        applyUnary<&MathTable::log10, detail::libm1::log10K>(in, out);
    }

    static void vErf(const double* in, double* out) {
        applyUnary<&MathTable::erf, detail::libm1::erfK>(in, out);
    }

    static void vErfc(const double* in, double* out) {
        applyUnary<&MathTable::erfc, detail::libm1::erfcK>(in, out);
    }

    static void vCbrt(const double* in, double* out) {
        applyUnary<&MathTable::cbrt, detail::libm1::cbrtK>(in, out);
    }

    // ── binary lane kernels (both operands arrays) ──

    static void vPow(const double* a, const double* b, double* out) {
        applyBinary<&MathTable::pow, detail::libm1::powK>(a, b, out);
    }

    /// atan2(y, x) with `y` = first array, `x` = second, like `std::atan2`.
    static void vAtan2(const double* y, const double* x, double* out) {
        applyBinary<&MathTable::atan2, detail::libm1::atan2K>(y, x, out);
    }

    static void vHypot(const double* a, const double* b, double* out) {
        applyBinary<&MathTable::hypot, detail::libm1::hypotK>(a, b, out);
    }

    // ── fused normal kernels (Tier B: same expression as ops/Probability.h;
    // the per-function entry carries the whole composition, so `vNormPdf`
    // follows `LIGHTAPE_MATH_IMPL_vNormPdf` and not the `exp` entry) ──

    /// phi(x) = exp(-x^2/2)/sqrt(2*pi), the exact expression of
    /// `lightape::normpdf`.
    static void vNormPdf(const double* in, double* out) {
        applyUnary<&MathTable::normPdf, detail::libm1::normPdfK>(in, out);
    }

    /// Phi(x) = erfc(-x/sqrt(2))/2, the exact expression of
    /// `lightape::normcdf`.
    static void vNormCdf(const double* in, double* out) {
        applyUnary<&MathTable::normCdf, detail::libm1::normCdfK>(in, out);
    }

private:
    /// W lanes through the resolved entry in `width`-sized steps, then the
    /// scalar tail via the compile-time libm kernel (the reference per-lane
    /// call, in lane order).
    template <const UnaryEntry MathTable::* Entry, detail::UnaryKernel Scalar>
    static void applyUnary(const double* in, double* out) {
        const UnaryEntry* e = selectEntry<W>(mathTable().*Entry);
        std::size_t l = 0;
        if (e != nullptr) {
            for (; l + e->width <= W; l += e->width) {
                e->fn(in + l, out + l);
            }
        }
        for (; l < W; ++l) {
            Scalar(in + l, out + l);
        }
    }

    template <const BinaryEntry MathTable::* Entry, detail::BinaryKernel Scalar>
    static void applyBinary(const double* a, const double* b, double* out) {
        const BinaryEntry* e = selectEntry<W>(mathTable().*Entry);
        std::size_t l = 0;
        if (e != nullptr) {
            for (; l + e->width <= W; l += e->width) {
                e->fn(a + l, b + l, out + l);
            }
        }
        for (; l < W; ++l) {
            Scalar(a + l, b + l, out + l);
        }
    }
};

} // namespace lightape::batch

#endif // LIGHTAPE_BATCH_VECTORMATH_H
