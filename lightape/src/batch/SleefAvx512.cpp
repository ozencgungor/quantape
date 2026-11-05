// x86-64 SLEEF kernels: 8-wide AVX-512F.
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

extern "C" void lightape_sleef_avx512f_sqrt(const double* in, double* out) {
    _mm512_storeu_pd(out, Sleef_sqrtd8_u05avx512f(_mm512_loadu_pd(in)));
}
extern "C" void lightape_sleef_avx512f_exp(const double* in, double* out) {
    _mm512_storeu_pd(out, Sleef_expd8_u10avx512f(_mm512_loadu_pd(in)));
}
extern "C" void lightape_sleef_avx512f_log(const double* in, double* out) {
    _mm512_storeu_pd(out, Sleef_logd8_u10avx512f(_mm512_loadu_pd(in)));
}
extern "C" void lightape_sleef_avx512f_expm1(const double* in, double* out) {
    _mm512_storeu_pd(out, Sleef_expm1d8_u10avx512f(_mm512_loadu_pd(in)));
}
extern "C" void lightape_sleef_avx512f_log1p(const double* in, double* out) {
    _mm512_storeu_pd(out, Sleef_log1pd8_u10avx512f(_mm512_loadu_pd(in)));
}
extern "C" void lightape_sleef_avx512f_sin(const double* in, double* out) {
    _mm512_storeu_pd(out, Sleef_sind8_u10avx512f(_mm512_loadu_pd(in)));
}
extern "C" void lightape_sleef_avx512f_cos(const double* in, double* out) {
    _mm512_storeu_pd(out, Sleef_cosd8_u10avx512f(_mm512_loadu_pd(in)));
}
extern "C" void lightape_sleef_avx512f_tan(const double* in, double* out) {
    _mm512_storeu_pd(out, Sleef_tand8_u10avx512f(_mm512_loadu_pd(in)));
}
extern "C" void lightape_sleef_avx512f_asin(const double* in, double* out) {
    _mm512_storeu_pd(out, Sleef_asind8_u10avx512f(_mm512_loadu_pd(in)));
}
extern "C" void lightape_sleef_avx512f_acos(const double* in, double* out) {
    _mm512_storeu_pd(out, Sleef_acosd8_u10avx512f(_mm512_loadu_pd(in)));
}
extern "C" void lightape_sleef_avx512f_atan(const double* in, double* out) {
    _mm512_storeu_pd(out, Sleef_atand8_u10avx512f(_mm512_loadu_pd(in)));
}
extern "C" void lightape_sleef_avx512f_sinh(const double* in, double* out) {
    _mm512_storeu_pd(out, Sleef_sinhd8_u10avx512f(_mm512_loadu_pd(in)));
}
extern "C" void lightape_sleef_avx512f_cosh(const double* in, double* out) {
    _mm512_storeu_pd(out, Sleef_coshd8_u10avx512f(_mm512_loadu_pd(in)));
}
extern "C" void lightape_sleef_avx512f_tanh(const double* in, double* out) {
    _mm512_storeu_pd(out, Sleef_tanhd8_u10avx512f(_mm512_loadu_pd(in)));
}
extern "C" void lightape_sleef_avx512f_exp2(const double* in, double* out) {
    _mm512_storeu_pd(out, Sleef_exp2d8_u10avx512f(_mm512_loadu_pd(in)));
}
extern "C" void lightape_sleef_avx512f_log2(const double* in, double* out) {
    _mm512_storeu_pd(out, Sleef_log2d8_u10avx512f(_mm512_loadu_pd(in)));
}
extern "C" void lightape_sleef_avx512f_log10(const double* in, double* out) {
    _mm512_storeu_pd(out, Sleef_log10d8_u10avx512f(_mm512_loadu_pd(in)));
}
extern "C" void lightape_sleef_avx512f_erf(const double* in, double* out) {
    _mm512_storeu_pd(out, Sleef_erfd8_u10avx512f(_mm512_loadu_pd(in)));
}
extern "C" void lightape_sleef_avx512f_erfc(const double* in, double* out) {
    _mm512_storeu_pd(out, Sleef_erfcd8_u15avx512f(_mm512_loadu_pd(in)));
}
extern "C" void lightape_sleef_avx512f_cbrt(const double* in, double* out) {
    _mm512_storeu_pd(out, Sleef_cbrtd8_u10avx512f(_mm512_loadu_pd(in)));
}
extern "C" void lightape_sleef_avx512f_normPdf(const double* in, double* out) {
    const __m512d x = _mm512_loadu_pd(in);
    const __m512d arg = _mm512_mul_pd(_mm512_mul_pd(x, x), _mm512_set1_pd(-0.5));
    _mm512_storeu_pd(out, _mm512_mul_pd(Sleef_expd8_u10avx512f(arg), _mm512_set1_pd(kInvSqrt2Pi)));
}
extern "C" void lightape_sleef_avx512f_normCdf(const double* in, double* out) {
    const __m512d arg = _mm512_mul_pd(_mm512_loadu_pd(in), _mm512_set1_pd(-kInvSqrt2));
    _mm512_storeu_pd(out, _mm512_mul_pd(Sleef_erfcd8_u15avx512f(arg), _mm512_set1_pd(0.5)));
}
extern "C" void lightape_sleef_avx512f_pow(const double* a, const double* b, double* out) {
    _mm512_storeu_pd(out, Sleef_powd8_u10avx512f(_mm512_loadu_pd(a), _mm512_loadu_pd(b)));
}
extern "C" void lightape_sleef_avx512f_atan2(const double* y, const double* x, double* out) {
    _mm512_storeu_pd(out, Sleef_atan2d8_u10avx512f(_mm512_loadu_pd(y), _mm512_loadu_pd(x)));
}
extern "C" void lightape_sleef_avx512f_hypot(const double* a, const double* b, double* out) {
    _mm512_storeu_pd(out, Sleef_hypotd8_u05avx512f(_mm512_loadu_pd(a), _mm512_loadu_pd(b)));
}
extern "C" void lightape_sleef_avx512f_sincos(const double* in, double* s, double* c) {
    const Sleef___m512d_2 r = Sleef_sincosd8_u10avx512f(_mm512_loadu_pd(in));
    _mm512_storeu_pd(s, r.x);
    _mm512_storeu_pd(c, r.y);
}
