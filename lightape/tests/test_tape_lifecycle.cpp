#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <vector>

#include "lightape/lightape.h"
#include <gtest/gtest.h>

namespace {

using lightape::RevScalar;
using lightape::RevTape;
using lightape::TapeScope;

class TapeLifecycleTest : public ::testing::Test {
protected:
    // A pristine tape per test also pins the chunk size back to the default.
    void SetUp() override {
        RevTape::releaseCurrentThread();
        RevTape::active();
    }
    void TearDown() override { RevTape::active().clear(); }
};

TEST_F(TapeLifecycleTest, ScopeRestoresPosition) {
    RevTape& tape = RevTape::active();
    RevScalar x(2.0, tape.input(2.0));
    const std::size_t base = tape.position();
    {
        TapeScope scope(tape);
        RevScalar acc = x;
        for (std::size_t i = 0; i < 50; ++i) {
            acc = acc + 1.0;
        }
        EXPECT_EQ(tape.position(), base + 50);
    }
    EXPECT_EQ(tape.position(), base);
    EXPECT_EQ(tape.size(), base - 1);

    RevScalar f = x * 3.0;
    tape.reverse(f.node);
    EXPECT_DOUBLE_EQ(tape.adjoint(x.node), 3.0);
}

TEST_F(TapeLifecycleTest, ScopeDismissKeepsRecording) {
    RevTape& tape = RevTape::active();
    RevScalar x(3.0, tape.input(3.0));
    double grad = 0.0;
    {
        TapeScope scope(tape);
        RevScalar y = x * x;
        tape.reverse(y.node);
        grad = tape.adjoint(x.node);
        scope.dismiss();
    }
    EXPECT_DOUBLE_EQ(grad, 6.0);
    EXPECT_EQ(tape.size(), 2u); // input + mul survive the dismissed scope

    RevScalar z(1.0, tape.input(1.0));
    EXPECT_EQ(z.node, 3u);
}

TEST_F(TapeLifecycleTest, NestedScopesRestoreInOrder) {
    RevTape& tape = RevTape::active();
    const std::size_t base = tape.position();
    {
        TapeScope outer(tape);
        RevScalar a(1.0, tape.input(1.0));
        EXPECT_EQ(a.node, 1u);
        const std::size_t innerPos = tape.position();
        {
            TapeScope inner(tape);
            RevScalar b(2.0, tape.input(2.0));
            EXPECT_EQ(tape.position(), innerPos + 1);
        }
        EXPECT_EQ(tape.position(), innerPos);
    }
    EXPECT_EQ(tape.position(), base);
}

TEST_F(TapeLifecycleTest, RunTruncatesOnReturn) {
    RevTape& tape = RevTape::active();
    const std::size_t base = tape.position();
    int calls = 0;
    RevTape::run([&] { ++calls; });
    EXPECT_EQ(calls, 1);

    const double grad = RevTape::run([&] {
        RevScalar x(2.0, tape.input(2.0));
        RevScalar y(3.0, tape.input(3.0));
        RevScalar f = x * x * y;
        tape.reverse(f.node);
        return tape.adjoint(x.node);
    });
    EXPECT_DOUBLE_EQ(grad, 12.0);
    EXPECT_EQ(tape.position(), base);
}

TEST_F(TapeLifecycleTest, ReleaseCurrentThreadGivesFreshTape) {
    RevTape& tape = RevTape::active();
    RevScalar x(1.0, tape.input(1.0));
    EXPECT_EQ(x.node, 1u);
    EXPECT_GT(tape.size(), 0u);

    RevTape::releaseCurrentThread();

    RevTape& fresh = RevTape::active();
    EXPECT_EQ(fresh.size(), 0u);
    EXPECT_EQ(fresh.position(), 1u);
    RevScalar y(4.0, fresh.input(4.0));
    RevScalar g = sqrt(y);
    fresh.reverse(g.node);
    EXPECT_DOUBLE_EQ(fresh.adjoint(y.node), 0.25);
}

TEST_F(TapeLifecycleTest, DefaultChunkSizeIsStaticDefault) {
    EXPECT_EQ(RevTape::kChunkNodes, 4096u);
    EXPECT_EQ(RevTape::active().chunkSize(), RevTape::kChunkNodes);
}

TEST_F(TapeLifecycleTest, ConfigureChunkSizeRoundsToPowerOfTwo) {
    RevTape& tape = RevTape::active();
    tape.configureChunkSize(100);
    EXPECT_EQ(tape.chunkSize(), 128u);
    tape.configureChunkSize(64);
    EXPECT_EQ(tape.chunkSize(), 64u);
    tape.configureChunkSize(1);
    EXPECT_EQ(tape.chunkSize(), 64u);
    tape.configureChunkSize(5000);
    EXPECT_EQ(tape.chunkSize(), 8192u);
    tape.configureChunkSize(1024);
    EXPECT_EQ(tape.chunkSize(), 1024u);
}

TEST_F(TapeLifecycleTest, SmallChunksSweepAcrossBoundaries) {
    RevTape& tape = RevTape::active();
    tape.configureChunkSize(128);
    EXPECT_EQ(tape.chunkSize(), 128u);

    constexpr std::size_t n = 320;
    std::vector<RevScalar> xs;
    xs.reserve(n);
    RevScalar acc(0.0);
    for (std::size_t i = 0; i < n; ++i) {
        const double v = 0.25 + 0.001 * static_cast<double>(i);
        xs.emplace_back(v, tape.input(v));
        acc = acc + xs[i] * 1.5;
    }
    EXPECT_GT(tape.size(), 2u * tape.chunkSize());
    tape.reverse(acc.node);
    for (std::size_t i = 0; i < n; ++i) {
        EXPECT_DOUBLE_EQ(tape.adjoint(xs[i].node), 1.5) << "i=" << i;
    }
}

TEST_F(TapeLifecycleTest, ScopeTruncatesRuntimeChunks) {
    RevTape& tape = RevTape::active();
    tape.configureChunkSize(128);
    const std::size_t base = tape.position();
    {
        TapeScope scope(tape);
        RevScalar acc(0.0, tape.input(0.0));
        for (std::size_t i = 0; i < 600; ++i) {
            acc = acc + 1.0;
        }
        EXPECT_GT(tape.size(), 3u * tape.chunkSize());
    }
    EXPECT_EQ(tape.position(), base);
}

TEST_F(TapeLifecycleTest, SincosReverseValueAndGradient) {
    RevTape& tape = RevTape::active();
    const double a = 1.25;
    RevScalar x(a, tape.input(a));
    const std::size_t before = tape.position();
    RevScalar s;
    RevScalar c;
    lightape::sincos(x, s, c);
    EXPECT_EQ(tape.position(), before + 2); // one Sin node + one Cos node
    EXPECT_DOUBLE_EQ(s.value, std::sin(a));
    EXPECT_DOUBLE_EQ(c.value, std::cos(a));

    RevScalar f = s + c * 2.0;
    tape.reverse(f.node);
    EXPECT_NEAR(tape.adjoint(x.node), std::cos(a) - 2.0 * std::sin(a), 1e-15);

    RevScalar zero(0.5);
    RevScalar zs;
    RevScalar zc;
    lightape::sincos(zero, zs, zc);
    EXPECT_EQ(zs.node, RevTape::kNone);
    EXPECT_EQ(zc.node, RevTape::kNone);
}

TEST_F(TapeLifecycleTest, SincosTangentDirectional) {
    using Tangent2 = lightape::Tangent<double, 2>;
    const double a = 0.6;
    Tangent2 in(a, std::array<double, 2>{1.0, -0.5});
    Tangent2 s;
    Tangent2 c;
    lightape::sincos(in, s, c);
    EXPECT_DOUBLE_EQ(s.value, std::sin(a));
    EXPECT_DOUBLE_EQ(c.value, std::cos(a));
    EXPECT_DOUBLE_EQ(s.d[0], std::cos(a));
    EXPECT_DOUBLE_EQ(s.d[1], -0.5 * std::cos(a));
    EXPECT_DOUBLE_EQ(c.d[0], -std::sin(a));
    EXPECT_DOUBLE_EQ(c.d[1], 0.5 * std::sin(a));
}

TEST_F(TapeLifecycleTest, SincosDoubleOverload) {
    double s = 0.0;
    double c = 0.0;
    lightape::sincos(0.75, s, c);
    EXPECT_DOUBLE_EQ(s, std::sin(0.75));
    EXPECT_DOUBLE_EQ(c, std::cos(0.75));
}

} // namespace
