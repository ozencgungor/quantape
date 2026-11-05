// Portable SLEEF scalar kernels (`generic`) for the VectorMath seam.
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

extern "C" void lightape_sleef_generic_sqrt(const double* in, double* out) {
    *out = Sleef_sqrt_u05(*in);
}
extern "C" void lightape_sleef_generic_exp(const double* in, double* out) {
    *out = Sleef_exp_u10(*in);
}
extern "C" void lightape_sleef_generic_log(const double* in, double* out) {
    *out = Sleef_log_u10(*in);
}
extern "C" void lightape_sleef_generic_expm1(const double* in, double* out) {
    *out = Sleef_expm1_u10(*in);
}
extern "C" void lightape_sleef_generic_log1p(const double* in, double* out) {
    *out = Sleef_log1p_u10(*in);
}
extern "C" void lightape_sleef_generic_sin(const double* in, double* out) {
    *out = Sleef_sin_u10(*in);
}
extern "C" void lightape_sleef_generic_cos(const double* in, double* out) {
    *out = Sleef_cos_u10(*in);
}
extern "C" void lightape_sleef_generic_tan(const double* in, double* out) {
    *out = Sleef_tan_u10(*in);
}
extern "C" void lightape_sleef_generic_asin(const double* in, double* out) {
    *out = Sleef_asin_u10(*in);
}
extern "C" void lightape_sleef_generic_acos(const double* in, double* out) {
    *out = Sleef_acos_u10(*in);
}
extern "C" void lightape_sleef_generic_atan(const double* in, double* out) {
    *out = Sleef_atan_u10(*in);
}
extern "C" void lightape_sleef_generic_sinh(const double* in, double* out) {
    *out = Sleef_sinh_u10(*in);
}
extern "C" void lightape_sleef_generic_cosh(const double* in, double* out) {
    *out = Sleef_cosh_u10(*in);
}
extern "C" void lightape_sleef_generic_tanh(const double* in, double* out) {
    *out = Sleef_tanh_u10(*in);
}
extern "C" void lightape_sleef_generic_exp2(const double* in, double* out) {
    *out = Sleef_exp2_u10(*in);
}
extern "C" void lightape_sleef_generic_log2(const double* in, double* out) {
    *out = Sleef_log2_u10(*in);
}
extern "C" void lightape_sleef_generic_log10(const double* in, double* out) {
    *out = Sleef_log10_u10(*in);
}
extern "C" void lightape_sleef_generic_erf(const double* in, double* out) {
    *out = Sleef_erf_u10(*in);
}
extern "C" void lightape_sleef_generic_erfc(const double* in, double* out) {
    *out = Sleef_erfc_u15(*in);
}
extern "C" void lightape_sleef_generic_cbrt(const double* in, double* out) {
    *out = Sleef_cbrt_u10(*in);
}
extern "C" void lightape_sleef_generic_normPdf(const double* in, double* out) {
    *out = kInvSqrt2Pi * Sleef_exp_u10(-0.5 * (*in * *in));
}
extern "C" void lightape_sleef_generic_normCdf(const double* in, double* out) {
    *out = 0.5 * Sleef_erfc_u15(-kInvSqrt2 * *in);
}
extern "C" void lightape_sleef_generic_pow(const double* a, const double* b, double* out) {
    *out = Sleef_pow_u10(*a, *b);
}
extern "C" void lightape_sleef_generic_atan2(const double* y, const double* x, double* out) {
    *out = Sleef_atan2_u10(*y, *x);
}
extern "C" void lightape_sleef_generic_hypot(const double* a, const double* b, double* out) {
    *out = Sleef_hypot_u05(*a, *b);
}
extern "C" void lightape_sleef_generic_sincos(const double* in, double* s, double* c) {
    const Sleef_double2 r = Sleef_sincos_u10(*in);
    *s = r.x;
    *c = r.y;
}
