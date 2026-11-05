// Direct-compile test for the small math gaps (CMake wiring is separate):
// inverse hyperbolic functions on RevScalar/Tangent/BatchTape, the signbit
// predicate, and the ops/Bessel.h logBesselK wrapper.
//
// Build (cached gtest tree):
//   c++ -std=c++20 -O2 -Wall -Wextra -ffp-contract=off \
//     -I lightape/include \
//     -I <gtest-src>/googletest/include test_math_gaps.cpp \
//     <build>/lib/libgtest_main.a <build>/lib/libgtest.a -pthread
#include <array>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <string>

#include "lightape/batch/BatchTape.h"
#include "lightape/ops/Bessel.h"
#include "lightape/rev/LeanReverse.h"
#include <gtest/gtest.h>

namespace {

using lightape::BatchScalar;
using lightape::BatchTape;
using lightape::RevScalar;
using lightape::RevTape;
using lightape::Tangent;

/// Raw IEEE-754 bits: the per-lane bitwise contract needs this (plain `==`
/// hides signed-zero and NaN-payload differences).
std::uint64_t Bits(double v) {
    return std::bit_cast<std::uint64_t>(v);
}

/// `base + step * lane`, so W > 1 lanes carry different values.
template <std::size_t W>
std::array<double, W> Lane(double base, double step) {
    std::array<double, W> v{};
    for (std::size_t l = 0; l < W; ++l) {
        v[l] = base + step * static_cast<double>(l);
    }
    return v;
}

/// Distinct per-lane reverse seeds.
template <std::size_t W>
std::array<double, W> Seeds() {
    std::array<double, W> s{};
    for (std::size_t l = 0; l < W; ++l) {
        s[l] = 0.5 + 0.25 * static_cast<double>(l);
    }
    return s;
}

/// 1e-15 gate with a 1e-15 absolute floor: assembling log K in log space and
/// taking the log of the rounded primal can differ by a few ulp of the log
/// value (e.g. the half-integer closed form), so a pure relative 1e-15 is too
/// strict at values near zero.
void expectClose(double got, double expected, double tol, const char* what) {
    SCOPED_TRACE(what);
    EXPECT_NEAR(got, expected, tol * (1.0 + std::fabs(expected)));
}

// ── asinh / acosh / atanh ──

TEST(MathGapsTest, InverseHyperbolicValuesMatchStd) {
    for (double x : {0.5, -1.25, 3.0}) {
        EXPECT_NEAR(lightape::asinh(RevScalar(x)).value, std::asinh(x), 1e-15) << "asinh " << x;
    }
    // acosh domain edge: acosh(1) = 0, plus interior points.
    for (double x : {1.0, 1.5, 10.0}) {
        EXPECT_NEAR(lightape::acosh(RevScalar(x)).value, std::acosh(x), 1e-15) << "acosh " << x;
    }
    for (double x : {0.25, -0.75, 0.9}) {
        EXPECT_NEAR(lightape::atanh(RevScalar(x)).value, std::atanh(x), 1e-15) << "atanh " << x;
    }
}

TEST(MathGapsTest, InverseHyperbolicGradientsMatchAnalytic) {
    RevTape& tape = RevTape::active();
    for (double x : {0.5, -1.25, 3.0}) {
        tape.clear();
        const RevScalar v(x, tape.input(x));
        const RevScalar y = lightape::asinh(v);
        tape.reverse(y.node);
        EXPECT_NEAR(tape.adjoint(v.node), 1.0 / std::sqrt(1.0 + x * x), 1e-13) << "asinh " << x;
    }
    for (double x : {1.5, 2.0, 10.0}) {
        tape.clear();
        const RevScalar v(x, tape.input(x));
        const RevScalar y = lightape::acosh(v);
        tape.reverse(y.node);
        EXPECT_NEAR(tape.adjoint(v.node), 1.0 / std::sqrt(x * x - 1.0), 1e-13) << "acosh " << x;
    }
    for (double x : {0.25, -0.75, 0.9}) {
        tape.clear();
        const RevScalar v(x, tape.input(x));
        const RevScalar y = lightape::atanh(v);
        tape.reverse(y.node);
        EXPECT_NEAR(tape.adjoint(v.node), 1.0 / (1.0 - x * x), 1e-13) << "atanh " << x;
    }
    tape.clear();
}

TEST(MathGapsTest, InverseHyperbolicHessianVectorMatchesAnalytic) {
    RevTape& tape = RevTape::active();
    {
        const double x = 0.5;
        tape.clear();
        const RevScalar v(x, tape.input(x));
        const RevScalar y = lightape::asinh(v);
        const std::uint32_t in[1] = {v.node};
        const double dir[1] = {1.0};
        double out[1] = {0.0};
        ASSERT_TRUE(tape.hessianVector(y.node, in, 1, dir, out));
        const double w = 1.0 + x * x;
        EXPECT_NEAR(out[0], -x / (w * std::sqrt(w)), 1e-12); // -x (1+x^2)^-3/2
    }
    {
        const double x = 2.0;
        tape.clear();
        const RevScalar v(x, tape.input(x));
        const RevScalar y = lightape::acosh(v);
        const std::uint32_t in[1] = {v.node};
        const double dir[1] = {1.0};
        double out[1] = {0.0};
        ASSERT_TRUE(tape.hessianVector(y.node, in, 1, dir, out));
        const double w = x * x - 1.0;
        EXPECT_NEAR(out[0], -x / (w * std::sqrt(w)), 1e-12); // -x (x^2-1)^-3/2
    }
    {
        const double x = 0.25;
        tape.clear();
        const RevScalar v(x, tape.input(x));
        const RevScalar y = lightape::atanh(v);
        const std::uint32_t in[1] = {v.node};
        const double dir[1] = {1.0};
        double out[1] = {0.0};
        ASSERT_TRUE(tape.hessianVector(y.node, in, 1, dir, out));
        const double w = 1.0 - x * x;
        EXPECT_NEAR(out[0], 2.0 * x / (w * w), 1e-12); // 2x (1-x^2)^-2
    }
    tape.clear();
}

TEST(MathGapsTest, InverseHyperbolicTangentDirectionalDerivatives) {
    using D1 = Tangent<double, 1>;
    {
        const D1 x(0.5, {1.3});
        const D1 y = lightape::asinh(x);
        EXPECT_NEAR(y.value, std::asinh(0.5), 1e-15);
        EXPECT_NEAR(y.d[0], 1.3 / std::sqrt(1.0 + 0.25), 1e-13);
    }
    {
        const D1 x(2.0, {-0.4});
        const D1 y = lightape::acosh(x);
        EXPECT_NEAR(y.value, std::acosh(2.0), 1e-15);
        EXPECT_NEAR(y.d[0], -0.4 / std::sqrt(3.0), 1e-13);
    }
    {
        const D1 x(-0.25, {0.7});
        const D1 y = lightape::atanh(x);
        EXPECT_NEAR(y.value, std::atanh(-0.25), 1e-15);
        EXPECT_NEAR(y.d[0], 0.7 / (1.0 - 0.0625), 1e-13);
    }
}

// ── signbit ──

TEST(MathGapsTest, SignbitIsPredicateOnPrimal) {
    using D1 = Tangent<double, 1>;
    EXPECT_FALSE(lightape::signbit(RevScalar(3.5)));
    EXPECT_TRUE(lightape::signbit(RevScalar(-3.5)));
    EXPECT_FALSE(lightape::signbit(RevScalar(0.0)));
    EXPECT_TRUE(lightape::signbit(RevScalar(-0.0)));
    // Tangent levels (primal only, tangents ignored).
    EXPECT_FALSE(lightape::signbit(D1(3.5, {1.0})));
    EXPECT_TRUE(lightape::signbit(D1(-3.5, {1.0})));
    EXPECT_FALSE(lightape::signbit(D1(0.0, {1.0})));
    EXPECT_TRUE(lightape::signbit(D1(-0.0, {1.0})));
}

// ── logBesselK ──

TEST(MathGapsTest, LogBesselKValuesMatchLogOfK) {
    const double pts[][2] = {{0.5, 0.5}, {1.7, 2.3}, {5.0, 10.0}, {0.5, 20.0}};
    for (const auto& p : pts) {
        SCOPED_TRACE(std::to_string(p[0]) + " " + std::to_string(p[1]));
        const double ref = std::log(lightape::besselK(p[0], p[1]));
        expectClose(lightape::logBesselK(p[0], p[1]), ref, 1e-15, "double trampoline");
        expectClose(lightape::logBesselK(RevScalar(p[0]), RevScalar(p[1])).value, ref, 1e-15,
                    "RevScalar wrapper");
    }
}

TEST(MathGapsTest, LogBesselKGradientMatchesBaseKernels) {
    RevTape& tape = RevTape::active();
    const double nu = 1.7;
    const double z = 2.3;
    tape.clear();
    const RevScalar n(nu, tape.input(nu));
    const RevScalar zz(z, tape.input(z));
    const RevScalar f = lightape::logBesselK(n, zz);
    expectClose(f.value, std::log(lightape::besselK(nu, z)), 1e-15, "value");
    tape.reverse(f.node);
    const double k = lightape::besselK(nu, z);
    EXPECT_NEAR(tape.adjoint(n.node), lightape::detail::besselKNuLogDeriv(nu, z), 1e-12);
    EXPECT_NEAR(tape.adjoint(zz.node), lightape::detail::besselKPrime(nu, z) / k, 1e-12);
    tape.clear();
}

TEST(MathGapsTest, LogBesselKHessianVectorMatchesChainRule) {
    RevTape& tape = RevTape::active();
    const double nu = 1.7;
    const double z = 2.3;
    tape.clear();
    const RevScalar n(nu, tape.input(nu));
    const RevScalar zz(z, tape.input(z));
    const RevScalar f = lightape::logBesselK(n, zz);

    const double k = lightape::besselK(nu, z);
    const double kn = lightape::detail::besselKNuDeriv(nu, z);
    const double kz = lightape::detail::besselKPrime(nu, z);
    const double knn = lightape::detail::besselKNuDeriv2(nu, z);
    const double knz = lightape::detail::besselKNuZDeriv(nu, z);
    const double kzz = lightape::detail::besselKZZDeriv(nu, z);
    ASSERT_TRUE(std::isfinite(knn) && std::isfinite(knz) && std::isfinite(kzz));
    const double lnn = knn / k - (kn / k) * (kn / k);
    const double lnz = knz / k - kn * kz / (k * k);
    const double lzz = kzz / k - (kz / k) * (kz / k);

    const std::uint32_t in[2] = {n.node, zz.node};
    const double dir[2] = {0.6, 0.5};
    double out[2] = {-99.0, -99.0};
    ASSERT_TRUE(tape.hessianVector(f.node, in, 2, dir, out));
    EXPECT_NEAR(out[0], lnn * dir[0] + lnz * dir[1], 1e-10);
    EXPECT_NEAR(out[1], lnz * dir[0] + lzz * dir[1], 1e-10);
    tape.clear();
}

/// Overflow domain: `logBesselK` stays finite where `besselK` overflows, but
/// the K Hessian pack (and at (90, 0.01) also the first-order K ratios) goes
/// non-finite, so the wrapper must record first order only and `hessianVector`
/// must fail loudly without writing outputs.
TEST(MathGapsTest, LogBesselKOverflowDomainDegradesLoudly) {
    RevTape& tape = RevTape::active();
    // (90, 0.01): besselK overflows; the pack is fully non-finite.
    {
        tape.clear();
        const RevScalar n(90.0, tape.input(90.0));
        const RevScalar zz(0.01, tape.input(0.01));
        const RevScalar f = lightape::logBesselK(n, zz);
        EXPECT_TRUE(std::isfinite(f.value));
        const std::uint32_t in[2] = {n.node, zz.node};
        const double dir[2] = {0.6, 0.5};
        double out[2] = {-99.0, -99.0};
        EXPECT_FALSE(tape.hessianVector(f.node, in, 2, dir, out));
        EXPECT_DOUBLE_EQ(out[0], -99.0);
        EXPECT_DOUBLE_EQ(out[1], -99.0);
    }
    // (90, 0.1): K stays finite (e^583), the first-order factors stay finite,
    // but the order quadratures overflow -> second order must still fail loud.
    {
        tape.clear();
        const RevScalar n(90.0, tape.input(90.0));
        const RevScalar zz(0.1, tape.input(0.1));
        const RevScalar f = lightape::logBesselK(n, zz);
        EXPECT_TRUE(std::isfinite(f.value));
        tape.reverse(f.node);
        EXPECT_TRUE(std::isfinite(tape.adjoint(n.node)));
        EXPECT_TRUE(std::isfinite(tape.adjoint(zz.node)));
        const std::uint32_t in[2] = {n.node, zz.node};
        const double dir[2] = {0.6, 0.5};
        double out[2] = {-99.0, -99.0};
        EXPECT_FALSE(tape.hessianVector(f.node, in, 2, dir, out));
        EXPECT_DOUBLE_EQ(out[0], -99.0);
        EXPECT_DOUBLE_EQ(out[1], -99.0);
    }
    tape.clear();
}

TEST(MathGapsTest, LogBesselKTangentDirectionalDerivative) {
    using D1 = Tangent<double, 1>;
    const D1 nu(1.7, {0.6});
    const D1 z(2.3, {0.5});
    const D1 f = lightape::logBesselK(nu, z);
    expectClose(f.value, std::log(lightape::besselK(1.7, 2.3)), 1e-15, "value");
    const double gNu = lightape::detail::besselKNuLogDeriv(1.7, 2.3);
    const double gZ = lightape::detail::besselKPrime(1.7, 2.3) / lightape::besselK(1.7, 2.3);
    EXPECT_NEAR(f.d[0], gNu * 0.6 + gZ * 0.5, 1e-12);
}

// ── batch lanes: bitwise against the scalar RevScalar program ──

template <std::size_t W, typename BatchFn, typename ScalarFn>
void CheckUnaryBatchBitwise(const char* name, const std::array<double, W>& in,
                            const std::array<double, W>& seed, BatchFn bfn, ScalarFn sfn) {
    BatchTape<W>& tape = BatchTape<W>::active();
    tape.clear();
    const BatchScalar<W> bx = tape.input(in);
    const BatchScalar<W> by = bfn(bx);
    tape.reverse(by.node, seed);
    const std::array<double, W> bvalue = by.value;
    const std::array<double, W> badj = tape.adjoint(bx.node);

    RevTape& scalar = RevTape::active();
    for (std::size_t l = 0; l < W; ++l) {
        scalar.clear();
        const RevScalar sx(in[l], scalar.input(in[l]));
        const RevScalar sy = sfn(sx);
        scalar.reverse(sy.node, seed[l]);
        EXPECT_EQ(Bits(bvalue[l]), Bits(sy.value)) << name << " lane " << l;
        EXPECT_EQ(Bits(badj[l]), Bits(scalar.adjoint(sx.node))) << name << " lane " << l;
    }
    tape.clear();
    scalar.clear();
}

template <std::size_t W>
void CheckInverseHyperbolicBatchBitwise() {
    const std::array<double, W> xa = Lane<W>(0.5, 0.3);   // asinh, any real
    const std::array<double, W> xc = Lane<W>(2.0, 0.5);   // acosh, x >= 1
    const std::array<double, W> xt = Lane<W>(-0.9, 0.22); // atanh, |x| < 1
    const std::array<double, W> seed = Seeds<W>();
    CheckUnaryBatchBitwise<W>(
        "asinh", xa, seed, [](const BatchScalar<W>& x) { return lightape::asinh(x); },
        [](const RevScalar& x) { return lightape::asinh(x); });
    CheckUnaryBatchBitwise<W>(
        "acosh", xc, seed, [](const BatchScalar<W>& x) { return lightape::acosh(x); },
        [](const RevScalar& x) { return lightape::acosh(x); });
    CheckUnaryBatchBitwise<W>(
        "atanh", xt, seed, [](const BatchScalar<W>& x) { return lightape::atanh(x); },
        [](const RevScalar& x) { return lightape::atanh(x); });
}

TEST(MathGapsTest, InverseHyperbolicBatchMatchesScalarBitwise) {
    CheckInverseHyperbolicBatchBitwise<1>();
    CheckInverseHyperbolicBatchBitwise<2>();
    CheckInverseHyperbolicBatchBitwise<4>();
    CheckInverseHyperbolicBatchBitwise<8>();
}

} // namespace
