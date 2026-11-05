#include <array>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <string>

#include "lightape/batch/BatchTape.h"
#include "lightape/lightape.h"
#include <gtest/gtest.h>

// Bitwise-determinism regression suite for the reverse engine.
//
// Policy (internal_docs/xad_stan_research/deep/decisions.md #22): no
// fast-math, explicit `std::fma` where fusion is wanted, fixed accumulation
// order, bitwise gates.  Every comparison below is raw IEEE-754 equality:
// `std::bit_cast<std::uint64_t>` (NaN compares equal to NaN, +0.0 and -0.0
// stay distinct).  The suite is compiled manually twice during verification,
// once with `-ffp-contract=off` and once with `-ffp-contract=fast`; it must
// pass under both because every gate is an intra-binary comparison.  Whether
// the recorded primals/adjoints themselves differ between the two flag
// settings is a separate finding, surfaced through the recorded gtest
// properties at the top of the repeatability case.
//
// Covered:
//  * 50x record/reverse/HVP repeatability on a reused tape after `clear()`;
//  * chunk-size invariance across {64, 4096, 65536} (cross-chunk accumulation);
//  * CSE off/on invariance for exact duplicate-free and repeated statements,
//    plus the documented inexact-adjoint caveat;
//  * BatchTape<W> per-lane bitwise equality with the scalar RevScalar tape;
//  * `resetTo`/`clear` lifecycle reuse.

namespace {

using lightape::BatchScalar;
using lightape::BatchTape;
using lightape::RevScalar;
using lightape::RevTape;

/// Raw IEEE-754 bit pattern; `==` on doubles would hide +0.0/-0.0.
std::uint64_t Bits(double v) {
    return std::bit_cast<std::uint64_t>(v);
}

/// Bitwise equality with NaN handled (ANY NaN equals any NaN, payload
/// differences are not a determinism failure).
bool SameBits(double a, double b) {
    if (std::isnan(a) || std::isnan(b)) {
        return std::isnan(a) && std::isnan(b);
    }
    return Bits(a) == Bits(b);
}

void ExpectBits(double got, double want, const std::string& label) {
    EXPECT_TRUE(SameBits(got, want)) << label << ": got " << Bits(got) << " want " << Bits(want);
}

// Inputs and directional seed shared by every scalar-tape scenario.
constexpr std::array<double, 4> kX = {0.7, 0.9, 1.4, 0.6};
constexpr std::array<double, 4> kV = {0.8, -0.6, 0.3, 0.5};

/// Chain length for the long variant: ~2.5k composites put the tape well past
/// 65536 nodes, so every configured chunk size sees cross-chunk accumulation.
constexpr std::size_t kChain = 2500;

struct Eval {
    double value = 0.0;
    std::array<double, 4> adj{};
    std::array<double, 4> hvp{};
    bool hvpOk = false;
    std::size_t nodes = 0;
};

/// Featured composite: fma, logSumExp, softmax, normcdf, pow and trig.
RevScalar RevComposite(const RevScalar& x1, const RevScalar& x2, const RevScalar& x3,
                       const RevScalar& x4) {
    const RevScalar a = exp(x1) + sin(x2) - log(x3);
    const RevScalar b = sqrt(fabs(a)) + x3 * x4;
    const RevScalar c = pow(b, 1.5) / (x1 + 2.0);
    const RevScalar d = normcdf(c);
    const RevScalar e = fma(a, b, d);
    const RevScalar xs[3] = {a, b, e};
    const RevScalar lse = logSumExp(xs, 3);
    RevScalar p[3];
    softmax(xs, 3, p);
    const RevScalar g = tan(x2) * 0.1;
    return p[0] * e + p[1] * lse + p[2] * g + fmin(d, e);
}

std::array<RevScalar, 4> MakeInputs() {
    RevTape& tape = RevTape::active();
    std::array<RevScalar, 4> x{};
    for (std::size_t i = 0; i < x.size(); ++i) {
        x[i] = RevScalar(kX[i], tape.input(kX[i]));
    }
    return x;
}

/// Record the composite on top of an existing (possibly non-empty) tape,
/// sweep it once for value + input adjoints, then run the HVP.
Eval RecordSweep(const std::array<RevScalar, 4>& x) {
    RevTape& tape = RevTape::active();
    const RevScalar out = RevComposite(x[0], x[1], x[2], x[3]);
    Eval e;
    e.value = out.value;
    e.nodes = tape.size();
    tape.reverse(out.node, 1.0);
    for (std::size_t i = 0; i < x.size(); ++i) {
        e.adj[i] = tape.adjoint(x[i].node);
    }
    const std::uint32_t inputs[4] = {x[0].node, x[1].node, x[2].node, x[3].node};
    e.hvpOk = tape.hessianVector(out.node, inputs, 4, kV.data(), e.hvp.data());
    return e;
}

Eval RunShortComposite() {
    RevTape& tape = RevTape::active();
    tape.clear();
    tape.useCse(false);
    const std::array<RevScalar, 4> x = MakeInputs();
    return RecordSweep(x);
}

Eval RunLongComposite() {
    RevTape& tape = RevTape::active();
    const std::array<RevScalar, 4> x = MakeInputs();
    RevScalar acc = RevComposite(x[0], x[1], x[2], x[3]);
    for (std::size_t i = 0; i < kChain; ++i) {
        acc = fma(acc, RevScalar(0.5), RevComposite(x[0], x[1], x[2], x[3]));
    }
    Eval e;
    e.value = acc.value;
    e.nodes = tape.size();
    tape.reverse(acc.node, 1.0);
    for (std::size_t i = 0; i < x.size(); ++i) {
        e.adj[i] = tape.adjoint(x[i].node);
    }
    const std::uint32_t inputs[4] = {x[0].node, x[1].node, x[2].node, x[3].node};
    e.hvpOk = tape.hessianVector(acc.node, inputs, 4, kV.data(), e.hvp.data());
    return e;
}

void ExpectEvalBitwise(const Eval& got, const Eval& want, const std::string& label) {
    EXPECT_TRUE(got.hvpOk) << label << ": hessianVector failed";
    ExpectBits(got.value, want.value, label + ".value");
    for (std::size_t i = 0; i < got.adj.size(); ++i) {
        ExpectBits(got.adj[i], want.adj[i], label + ".adj[" + std::to_string(i) + "]");
        ExpectBits(got.hvp[i], want.hvp[i], label + ".hvp[" + std::to_string(i) + "]");
    }
}

class DeterminismTest : public ::testing::Test {
protected:
    // A pristine tape per test pins chunk size back to the default and CSE off.
    void SetUp() override {
        RevTape::releaseCurrentThread();
        RevTape::active();
    }
    void TearDown() override {
        RevTape& tape = RevTape::active();
        tape.useCse(false);
        tape.clear();
    }
};

TEST_F(DeterminismTest, RepeatabilityFiftyRunsOnReusedTape) {
    const Eval first = RunShortComposite();
    ASSERT_TRUE(first.hvpOk);
    // Stable across both `-ffp-contract` settings?  These properties let the
    // verification diff the two builds without weakening the bitwise gates.
    RecordProperty("composite_value_bits", std::to_string(Bits(first.value)));
    RecordProperty("composite_adj0_bits", std::to_string(Bits(first.adj[0])));
    RecordProperty("composite_hvp0_bits", std::to_string(Bits(first.hvp[0])));

    for (int run = 1; run < 50; ++run) {
        SCOPED_TRACE("run " + std::to_string(run + 1));
        const Eval e = RunShortComposite();
        ExpectEvalBitwise(e, first, "repeat");
    }
}

TEST_F(DeterminismTest, ChunkSizeInvarianceBitwise) {
    const std::size_t configs[3] = {64, 4096, 65536};
    Eval ref;
    for (std::size_t c = 0; c < 3; ++c) {
        RevTape& tape = RevTape::active();
        tape.clear();
        tape.useCse(false);
        tape.configureChunkSize(configs[c]);
        ASSERT_EQ(tape.chunkSize(), configs[c]);
        const Eval e = RunLongComposite();
        ASSERT_TRUE(e.hvpOk);
        if (c == 0) {
            ref = e;
            // Every configured chunk size must observe cross-chunk parents.
            EXPECT_GT(e.nodes, std::size_t{65536});
            // Cross-build evidence: the long-chain first-order result is
            // identical under -ffp-contract=off/fast on this platform, while
            // the second-order replay entries differ in the last ulp because
            // hessianVector's `bi * fa + ai * dfa` shape is implicitly fused
            // under contract=fast.  That is exactly why decision #22 mandates
            // -ffp-contract=off plus explicit std::fma.  Recording the bits
            // lets the verification diff the two builds.
            RecordProperty("long_value_bits", std::to_string(Bits(ref.value)));
            RecordProperty("long_adj0_bits", std::to_string(Bits(ref.adj[0])));
            for (std::size_t i = 0; i < ref.hvp.size(); ++i) {
                RecordProperty("long_hvp" + std::to_string(i) + "_bits",
                               std::to_string(Bits(ref.hvp[i])));
            }
        } else {
            SCOPED_TRACE("chunk " + std::to_string(configs[c]));
            ExpectEvalBitwise(e, ref, "chunk");
        }
    }
}

struct CseEval {
    double value = 0.0;
    std::array<double, 4> adj{};
    std::size_t nodes = 0;
};

void ExpectCseBitwise(const CseEval& on, const CseEval& off) {
    ExpectBits(on.value, off.value, "cse value");
    for (std::size_t i = 0; i < on.adj.size(); ++i) {
        ExpectBits(on.adj[i], off.adj[i], "cse adj[" + std::to_string(i) + "]");
    }
}

/// Duplicate-free, exactly representable program: CSE must be a no-op.
CseEval RunDuplicateFreeExact(bool cse) {
    RevTape& tape = RevTape::active();
    tape.clear();
    tape.useCse(cse);
    const RevScalar x1(0.5, tape.input(0.5));
    const RevScalar x2(0.25, tape.input(0.25));
    const RevScalar x3(2.0, tape.input(2.0));
    const RevScalar x4(4.0, tape.input(4.0));
    const RevScalar f = (x1 * x2 + x3) * (x4 - x3) + x3 / x1;
    CseEval r;
    r.value = f.value;
    r.nodes = tape.size();
    tape.reverse(f.node, 1.0);
    r.adj = {tape.adjoint(x1.node), tape.adjoint(x2.node), tape.adjoint(x3.node),
             tape.adjoint(x4.node)};
    return r;
}

TEST_F(DeterminismTest, CseInvarianceDuplicateFreeExact) {
    const CseEval off = RunDuplicateFreeExact(false);
    const CseEval on = RunDuplicateFreeExact(true);
    EXPECT_EQ(on.nodes, off.nodes); // no repeated statement, so nothing merges
    ExpectCseBitwise(on, off);
}

/// Repeated subexpression, exactly representable: with the cache on the
/// shared statement records once and its two consumer adjoints accumulate
/// before the single pullback; with the cache off they pull back separately.
/// Both groupings are exact here, so values AND adjoints must be bitwise equal.
CseEval RunRepeatedExact(bool cse) {
    RevTape& tape = RevTape::active();
    tape.clear();
    tape.useCse(cse);
    const RevScalar x1(0.5, tape.input(0.5));
    const RevScalar x2(0.25, tape.input(0.25));
    const RevScalar x3(2.0, tape.input(2.0));
    const RevScalar x4(4.0, tape.input(4.0));
    const RevScalar s1 = x1 * x2;
    const RevScalar s2 = x1 * x2; // immediate duplicate -> guaranteed cache hit
    const RevScalar t1 = x3 + x3;
    const RevScalar f = (s1 + s2) * (t1 + x4) + x4;
    CseEval r;
    r.value = f.value;
    r.nodes = tape.size();
    tape.reverse(f.node, 1.0);
    r.adj = {tape.adjoint(x1.node), tape.adjoint(x2.node), tape.adjoint(x3.node),
             tape.adjoint(x4.node)};
    return r;
}

TEST_F(DeterminismTest, CseInvarianceRepeatedExact) {
    const CseEval off = RunRepeatedExact(false);
    const CseEval on = RunRepeatedExact(true);
    EXPECT_LT(on.nodes, off.nodes); // the shared statement records once
    ExpectCseBitwise(on, off);
}

struct InexactEval {
    double value = 0.0;
    double adj = 0.0;
};

InexactEval RunInexactRepeated(bool cse) {
    RevTape& tape = RevTape::active();
    tape.clear();
    tape.useCse(cse);
    const RevScalar x(1.2, tape.input(1.2));
    const RevScalar e1 = exp(x);
    const RevScalar e2 = exp(x); // immediate duplicate -> guaranteed cache hit
    const RevScalar f = e1 * 0.1 + e2 * 0.2;
    InexactEval r;
    r.value = f.value;
    tape.reverse(f.node, 1.0);
    r.adj = tape.adjoint(x.node);
    return r;
}

TEST_F(DeterminismTest, CseInexactRepeatedAdjointsMayDiffer) {
    const InexactEval off = RunInexactRepeated(false);
    const InexactEval on = RunInexactRepeated(true);
    // The primal stays bitwise identical: a cache hit requires the recomputed
    // primal to be bitwise equal to the cached node's value.
    ExpectBits(on.value, off.value, "inexact cse value");
    // Adjoints are deliberately NOT bitwise-gated for inexact arithmetic.
    // Sharing moves the accumulation: the cached exp node receives
    // 0.2 + 0.1 = 0.30000000000000004 before the single pullback by `e`,
    // while the unshared tape ends with 0.2 * e + 0.1 * e.  For x = 1.2 on
    // binary64 those differ in the last ulp; that is by design (decision #22
    // pins bitwise gates to exactly representable cases).  Documented via the
    // property below rather than asserted, so the gate stays portable.
    EXPECT_NEAR(on.adj, off.adj, 1e-13);
    RecordProperty("inexact_repeated_adjoints_same_bits",
                   SameBits(on.adj, off.adj) ? "true" : "false");
}

template <std::size_t W>
std::array<double, W> Lane(double base, double step) {
    std::array<double, W> v{};
    for (std::size_t l = 0; l < W; ++l) {
        v[l] = base + step * static_cast<double>(l);
    }
    return v;
}

/// Scalar-generic program over the BatchTape opcode set (pattern reused from
/// tests/test_batch_tape.cpp).  The two branch ops are lane-uniform by
/// construction, so the batch and the scalar tape record the same statements.
template <typename S>
S LaneComposite(const S& x1, const S& x2, const S& x3, const S& x4) {
    const S one(1.0);
    const S a = exp(x1) - log(x2);
    const S b = sqrt(abs(a)) + x3 * x4;
    const S c = -b / (x1 + 2.0);
    const S d = pow(b, c);
    const S e = fmin(d, d + one);
    const S f = fmax(e, e - one);
    const S g = pow(abs(x4), 1.5);
    return f * 0.5 + g * 0.25 + a / (x2 + one);
}

template <std::size_t W>
void CheckBatchLaneInvariance() {
    const std::array<std::array<double, W>, 4> in = {Lane<W>(0.35, 0.05), Lane<W>(1.20, 0.08),
                                                     Lane<W>(0.80, 0.03), Lane<W>(1.60, 0.10)};
    std::array<double, W> seed{};
    for (std::size_t l = 0; l < W; ++l) {
        seed[l] = 0.7 + 0.1 * static_cast<double>(l);
    }

    BatchTape<W>& tape = BatchTape<W>::active();
    tape.clear();
    std::array<BatchScalar<W>, 4> x{};
    for (std::size_t i = 0; i < 4; ++i) {
        x[i] = tape.input(in[i]);
    }
    const BatchScalar<W> out = LaneComposite(x[0], x[1], x[2], x[3]);
    tape.reverse(out.node, seed);

    RevTape& scalar = RevTape::active();
    std::array<std::array<double, W>, 4> refAdj{};
    std::array<double, W> refValue{};
    for (std::size_t l = 0; l < W; ++l) {
        scalar.clear();
        std::array<RevScalar, 4> sx{};
        for (std::size_t i = 0; i < 4; ++i) {
            sx[i] = RevScalar(in[i][l], scalar.input(in[i][l]));
        }
        const RevScalar sout = LaneComposite(sx[0], sx[1], sx[2], sx[3]);
        scalar.reverse(sout.node, seed[l]);
        refValue[l] = sout.value;
        for (std::size_t i = 0; i < 4; ++i) {
            refAdj[i][l] = scalar.adjoint(sx[i].node);
        }
    }

    for (std::size_t l = 0; l < W; ++l) {
        SCOPED_TRACE("W=" + std::to_string(W) + " lane " + std::to_string(l));
        ExpectBits(out.value[l], refValue[l], "batch value");
        for (std::size_t i = 0; i < 4; ++i) {
            ExpectBits(tape.adjoint(x[i].node)[l], refAdj[i][l], "batch adjoint");
        }
    }
}

TEST_F(DeterminismTest, BatchTapeLaneInvarianceBitwise) {
    CheckBatchLaneInvariance<1>();
    CheckBatchLaneInvariance<2>();
    CheckBatchLaneInvariance<4>();
    CheckBatchLaneInvariance<8>();
}

TEST_F(DeterminismTest, ResetToAndClearReuseBitwise) {
    RevTape& tape = RevTape::active();
    tape.clear();
    tape.useCse(false);
    std::array<RevScalar, 4> x = MakeInputs();
    const std::size_t mark = tape.position();
    const Eval first = RecordSweep(x);
    ASSERT_TRUE(first.hvpOk);

    // Truncate the composite, keep the inputs, drop the stale leaf adjoints
    // (resetTo preserves them by contract), then re-record and re-sweep.
    tape.resetTo(mark);
    tape.resetAdjoints();
    EXPECT_EQ(tape.position(), mark);
    const Eval second = RecordSweep(x);
    ExpectEvalBitwise(second, first, "resetTo re-record");

    tape.resetTo(mark);
    tape.resetAdjoints();
    const Eval third = RecordSweep(x);
    ExpectEvalBitwise(third, first, "second resetTo re-record");

    // Full clear + rebuild: the fresh recording reuses the same indices.
    tape.clear();
    tape.useCse(false);
    std::array<RevScalar, 4> y = MakeInputs();
    EXPECT_EQ(y[0].node, x[0].node);
    const Eval fourth = RecordSweep(y);
    ExpectEvalBitwise(fourth, first, "clear re-record");
}

} // namespace
