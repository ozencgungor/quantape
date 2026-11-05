// x86-64 SLEEF kernels: 4-wide AVX2 and 2-wide AVX2128.
// Both rungs use the same -mavx2 -mfma source flags; AVX2128 only needs the
// 128-bit subset of those instructions.
//
// Compiled TU boundary for the x86-64 VectorMath backend: the portable
// `lightape/batch/VectorMath.h` header only declares these `extern "C"` entry
// points there; the per-source ISA flags and SLEEF linkage live in
// lightape/CMakeLists.txt, so no TU that includes the header (in particular
// the libm reference) ever sees sleef.h or ISA flags.

#include "lightape/batch/VectorMath.h"
#include <sleef.h>

namespace {
using lightape::batch::detail::kInvSqrt2;
using lightape::batch::detail::kInvSqrt2Pi;
} // namespace

extern "C" void lightape_sleef_avx2128_sqrt(const double* in, double* out) {
    _mm_storeu_pd(out, Sleef_sqrtd2_u05avx2128(_mm_loadu_pd(in)));
}
extern "C" void lightape_sleef_avx2128_exp(const double* in, double* out) {
    _mm_storeu_pd(out, Sleef_expd2_u10avx2128(_mm_loadu_pd(in)));
}
extern "C" void lightape_sleef_avx2128_log(const double* in, double* out) {
    _mm_storeu_pd(out, Sleef_logd2_u10avx2128(_mm_loadu_pd(in)));
}
extern "C" void lightape_sleef_avx2128_expm1(const double* in, double* out) {
    _mm_storeu_pd(out, Sleef_expm1d2_u10avx2128(_mm_loadu_pd(in)));
}
extern "C" void lightape_sleef_avx2128_log1p(const double* in, double* out) {
    _mm_storeu_pd(out, Sleef_log1pd2_u10avx2128(_mm_loadu_pd(in)));
}
extern "C" void lightape_sleef_avx2128_sin(const double* in, double* out) {
    _mm_storeu_pd(out, Sleef_sind2_u10avx2128(_mm_loadu_pd(in)));
}
extern "C" void lightape_sleef_avx2128_cos(const double* in, double* out) {
    _mm_storeu_pd(out, Sleef_cosd2_u10avx2128(_mm_loadu_pd(in)));
}
extern "C" void lightape_sleef_avx2128_tan(const double* in, double* out) {
    _mm_storeu_pd(out, Sleef_tand2_u10avx2128(_mm_loadu_pd(in)));
}
extern "C" void lightape_sleef_avx2128_asin(const double* in, double* out) {
    _mm_storeu_pd(out, Sleef_asind2_u10avx2128(_mm_loadu_pd(in)));
}
extern "C" void lightape_sleef_avx2128_acos(const double* in, double* out) {
    _mm_storeu_pd(out, Sleef_acosd2_u10avx2128(_mm_loadu_pd(in)));
}
extern "C" void lightape_sleef_avx2128_atan(const double* in, double* out) {
    _mm_storeu_pd(out, Sleef_atand2_u10avx2128(_mm_loadu_pd(in)));
}
extern "C" void lightape_sleef_avx2128_sinh(const double* in, double* out) {
    _mm_storeu_pd(out, Sleef_sinhd2_u10avx2128(_mm_loadu_pd(in)));
}
extern "C" void lightape_sleef_avx2128_cosh(const double* in, double* out) {
    _mm_storeu_pd(out, Sleef_coshd2_u10avx2128(_mm_loadu_pd(in)));
}
extern "C" void lightape_sleef_avx2128_tanh(const double* in, double* out) {
    _mm_storeu_pd(out, Sleef_tanhd2_u10avx2128(_mm_loadu_pd(in)));
}
extern "C" void lightape_sleef_avx2128_exp2(const double* in, double* out) {
    _mm_storeu_pd(out, Sleef_exp2d2_u10avx2128(_mm_loadu_pd(in)));
}
extern "C" void lightape_sleef_avx2128_log2(const double* in, double* out) {
    _mm_storeu_pd(out, Sleef_log2d2_u10avx2128(_mm_loadu_pd(in)));
}
extern "C" void lightape_sleef_avx2128_log10(const double* in, double* out) {
    _mm_storeu_pd(out, Sleef_log10d2_u10avx2128(_mm_loadu_pd(in)));
}
extern "C" void lightape_sleef_avx2128_erf(const double* in, double* out) {
    _mm_storeu_pd(out, Sleef_erfd2_u10avx2128(_mm_loadu_pd(in)));
}
extern "C" void lightape_sleef_avx2128_erfc(const double* in, double* out) {
    _mm_storeu_pd(out, Sleef_erfcd2_u15avx2128(_mm_loadu_pd(in)));
}
extern "C" void lightape_sleef_avx2128_cbrt(const double* in, double* out) {
    _mm_storeu_pd(out, Sleef_cbrtd2_u10avx2128(_mm_loadu_pd(in)));
}
extern "C" void lightape_sleef_avx2128_normPdf(const double* in, double* out) {
    const __m128d x = _mm_loadu_pd(in);
    const __m128d arg = _mm_mul_pd(_mm_mul_pd(x, x), _mm_set1_pd(-0.5));
    _mm_storeu_pd(out, _mm_mul_pd(Sleef_expd2_u10avx2128(arg), _mm_set1_pd(kInvSqrt2Pi)));
}
extern "C" void lightape_sleef_avx2128_normCdf(const double* in, double* out) {
    const __m128d arg = _mm_mul_pd(_mm_loadu_pd(in), _mm_set1_pd(-kInvSqrt2));
    _mm_storeu_pd(out, _mm_mul_pd(Sleef_erfcd2_u15avx2128(arg), _mm_set1_pd(0.5)));
}
extern "C" void lightape_sleef_avx2128_pow(const double* a, const double* b, double* out) {
    _mm_storeu_pd(out, Sleef_powd2_u10avx2128(_mm_loadu_pd(a), _mm_loadu_pd(b)));
}
extern "C" void lightape_sleef_avx2128_atan2(const double* y, const double* x, double* out) {
    _mm_storeu_pd(out, Sleef_atan2d2_u10avx2128(_mm_loadu_pd(y), _mm_loadu_pd(x)));
}
extern "C" void lightape_sleef_avx2128_hypot(const double* a, const double* b, double* out) {
    _mm_storeu_pd(out, Sleef_hypotd2_u05avx2128(_mm_loadu_pd(a), _mm_loadu_pd(b)));
}
extern "C" void lightape_sleef_avx2128_sincos(const double* in, double* s, double* c) {
    const Sleef___m128d_2 r = Sleef_sincosd2_u10avx2128(_mm_loadu_pd(in));
    _mm_storeu_pd(s, r.x);
    _mm_storeu_pd(c, r.y);
}

extern "C" void lightape_sleef_avx2_sqrt(const double* in, double* out) {
    _mm256_storeu_pd(out, Sleef_sqrtd4_u05avx2(_mm256_loadu_pd(in)));
}
extern "C" void lightape_sleef_avx2_exp(const double* in, double* out) {
    _mm256_storeu_pd(out, Sleef_expd4_u10avx2(_mm256_loadu_pd(in)));
}
extern "C" void lightape_sleef_avx2_log(const double* in, double* out) {
    _mm256_storeu_pd(out, Sleef_logd4_u10avx2(_mm256_loadu_pd(in)));
}
extern "C" void lightape_sleef_avx2_expm1(const double* in, double* out) {
    _mm256_storeu_pd(out, Sleef_expm1d4_u10avx2(_mm256_loadu_pd(in)));
}
extern "C" void lightape_sleef_avx2_log1p(const double* in, double* out) {
    _mm256_storeu_pd(out, Sleef_log1pd4_u10avx2(_mm256_loadu_pd(in)));
}
extern "C" void lightape_sleef_avx2_sin(const double* in, double* out) {
    _mm256_storeu_pd(out, Sleef_sind4_u10avx2(_mm256_loadu_pd(in)));
}
extern "C" void lightape_sleef_avx2_cos(const double* in, double* out) {
    _mm256_storeu_pd(out, Sleef_cosd4_u10avx2(_mm256_loadu_pd(in)));
}
extern "C" void lightape_sleef_avx2_tan(const double* in, double* out) {
    _mm256_storeu_pd(out, Sleef_tand4_u10avx2(_mm256_loadu_pd(in)));
}
extern "C" void lightape_sleef_avx2_asin(const double* in, double* out) {
    _mm256_storeu_pd(out, Sleef_asind4_u10avx2(_mm256_loadu_pd(in)));
}
extern "C" void lightape_sleef_avx2_acos(const double* in, double* out) {
    _mm256_storeu_pd(out, Sleef_acosd4_u10avx2(_mm256_loadu_pd(in)));
}
extern "C" void lightape_sleef_avx2_atan(const double* in, double* out) {
    _mm256_storeu_pd(out, Sleef_atand4_u10avx2(_mm256_loadu_pd(in)));
}
extern "C" void lightape_sleef_avx2_sinh(const double* in, double* out) {
    _mm256_storeu_pd(out, Sleef_sinhd4_u10avx2(_mm256_loadu_pd(in)));
}
extern "C" void lightape_sleef_avx2_cosh(const double* in, double* out) {
    _mm256_storeu_pd(out, Sleef_coshd4_u10avx2(_mm256_loadu_pd(in)));
}
extern "C" void lightape_sleef_avx2_tanh(const double* in, double* out) {
    _mm256_storeu_pd(out, Sleef_tanhd4_u10avx2(_mm256_loadu_pd(in)));
}
extern "C" void lightape_sleef_avx2_exp2(const double* in, double* out) {
    _mm256_storeu_pd(out, Sleef_exp2d4_u10avx2(_mm256_loadu_pd(in)));
}
extern "C" void lightape_sleef_avx2_log2(const double* in, double* out) {
    _mm256_storeu_pd(out, Sleef_log2d4_u10avx2(_mm256_loadu_pd(in)));
}
extern "C" void lightape_sleef_avx2_log10(const double* in, double* out) {
    _mm256_storeu_pd(out, Sleef_log10d4_u10avx2(_mm256_loadu_pd(in)));
}
extern "C" void lightape_sleef_avx2_erf(const double* in, double* out) {
    _mm256_storeu_pd(out, Sleef_erfd4_u10avx2(_mm256_loadu_pd(in)));
}
extern "C" void lightape_sleef_avx2_erfc(const double* in, double* out) {
    _mm256_storeu_pd(out, Sleef_erfcd4_u15avx2(_mm256_loadu_pd(in)));
}
extern "C" void lightape_sleef_avx2_cbrt(const double* in, double* out) {
    _mm256_storeu_pd(out, Sleef_cbrtd4_u10avx2(_mm256_loadu_pd(in)));
}
extern "C" void lightape_sleef_avx2_normPdf(const double* in, double* out) {
    const __m256d x = _mm256_loadu_pd(in);
    const __m256d arg = _mm256_mul_pd(_mm256_mul_pd(x, x), _mm256_set1_pd(-0.5));
    _mm256_storeu_pd(out, _mm256_mul_pd(Sleef_expd4_u10avx2(arg), _mm256_set1_pd(kInvSqrt2Pi)));
}
extern "C" void lightape_sleef_avx2_normCdf(const double* in, double* out) {
    const __m256d arg = _mm256_mul_pd(_mm256_loadu_pd(in), _mm256_set1_pd(-kInvSqrt2));
    _mm256_storeu_pd(out, _mm256_mul_pd(Sleef_erfcd4_u15avx2(arg), _mm256_set1_pd(0.5)));
}
extern "C" void lightape_sleef_avx2_pow(const double* a, const double* b, double* out) {
    _mm256_storeu_pd(out, Sleef_powd4_u10avx2(_mm256_loadu_pd(a), _mm256_loadu_pd(b)));
}
extern "C" void lightape_sleef_avx2_atan2(const double* y, const double* x, double* out) {
    _mm256_storeu_pd(out, Sleef_atan2d4_u10avx2(_mm256_loadu_pd(y), _mm256_loadu_pd(x)));
}
extern "C" void lightape_sleef_avx2_hypot(const double* a, const double* b, double* out) {
    _mm256_storeu_pd(out, Sleef_hypotd4_u05avx2(_mm256_loadu_pd(a), _mm256_loadu_pd(b)));
}
extern "C" void lightape_sleef_avx2_sincos(const double* in, double* s, double* c) {
    const Sleef___m256d_2 r = Sleef_sincosd4_u10avx2(_mm256_loadu_pd(in));
    _mm256_storeu_pd(s, r.x);
    _mm256_storeu_pd(c, r.y);
}
