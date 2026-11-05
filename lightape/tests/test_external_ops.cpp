#include <cmath>
#include <cstddef>
#include <cstdint>
#include <vector>

#include "lightape/lightape.h"
#include <gtest/gtest.h>

namespace {

using lightape::ExtArg;
using lightape::RevScalar;
using lightape::RevTape;

class ExternalOpsTest : public ::testing::Test {
protected:
    void SetUp() override { RevTape::active().clear(); }
    void TearDown() override { RevTape::active().clear(); }
};

struct Payload {
    double a = 0.0;
    double b = 0.0;
};

void payloadVjp(const void* payload, double seed, const ExtArg* args, std::uint32_t /*n*/,
                RevTape& tape) noexcept {
    const auto* p = static_cast<const Payload*>(payload);
    tape.addAdjoint(args[0].node, seed * p->a);
    tape.addAdjoint(args[1].node, seed * p->b);
}

TEST_F(ExternalOpsTest, FmaValueAndGradients) {
    RevTape& tape = RevTape::active();
    RevScalar x(1.5, tape.input(1.5));
    RevScalar y(2.5, tape.input(2.5));
    RevScalar z(0.75, tape.input(0.75));
    const std::size_t before = tape.size();
    RevScalar f = fma(x, y, z);
    EXPECT_DOUBLE_EQ(f.value, std::fma(1.5, 2.5, 0.75));
    EXPECT_EQ(tape.size(), before + 1u);
    tape.reverse(f.node);
    EXPECT_DOUBLE_EQ(tape.adjoint(x.node), 2.5);
    EXPECT_DOUBLE_EQ(tape.adjoint(y.node), 1.5);
    EXPECT_DOUBLE_EQ(tape.adjoint(z.node), 1.0);
}

TEST_F(ExternalOpsTest, FmaCollapsesConstants) {
    const RevScalar a(2.0);
    const RevScalar b(3.0);
    const RevScalar c(4.0);
    const RevScalar f = fma(a, b, c);
    EXPECT_EQ(f.node, RevTape::kNone);
    EXPECT_DOUBLE_EQ(f.value, 10.0);
    EXPECT_EQ(RevTape::active().size(), 0u);
}

TEST_F(ExternalOpsTest, LogSumExpValueAndGradients) {
    RevTape& tape = RevTape::active();
    constexpr std::size_t n = 5;
    const double values[n] = {-1.0, 0.5, 2.0, -0.25, 1.0};
    RevScalar xs[n] = {
        RevScalar(values[0], tape.input(values[0])), RevScalar(values[1], tape.input(values[1])),
        RevScalar(values[2], tape.input(values[2])), RevScalar(values[3], tape.input(values[3])),
        RevScalar(values[4], tape.input(values[4]))};
    const std::size_t before = tape.size();
    const RevScalar f = lightape::logSumExp(xs, n);
    EXPECT_EQ(tape.size(), before + 1u);

    double m = values[0];
    for (std::size_t i = 1; i < n; ++i) {
        if (values[i] > m) {
            m = values[i];
        }
    }
    double s = 0.0;
    for (std::size_t i = 0; i < n; ++i) {
        s += std::exp(values[i] - m);
    }
    EXPECT_NEAR(f.value, m + std::log(s), 1e-15);
    tape.reverse(f.node);
    for (std::size_t i = 0; i < n; ++i) {
        EXPECT_NEAR(tape.adjoint(xs[i].node), std::exp(values[i] - m) / s, 1e-15) << "i=" << i;
    }
}

TEST_F(ExternalOpsTest, LogSumExpCollapsesConstants) {
    const RevScalar xs[3] = {RevScalar(0.0), RevScalar(1.0), RevScalar(2.0)};
    const RevScalar f = lightape::logSumExp(xs, 3);
    EXPECT_EQ(f.node, RevTape::kNone);
    EXPECT_DOUBLE_EQ(f.value, std::log(std::exp(0.0) + std::exp(1.0) + std::exp(2.0)));
    EXPECT_EQ(RevTape::active().size(), 0u);
}

TEST_F(ExternalOpsTest, PayloadCopySurvivesSourceScope) {
    RevTape& tape = RevTape::active();
    RevScalar x(2.0, tape.input(2.0));
    RevScalar y(3.0, tape.input(3.0));
    const std::size_t before = tape.size();
    RevScalar f = [&] {
        const Payload payload{5.0, 7.0};
        const RevScalar args[2] = {x, y};
        return lightape::detail::makeExternal(5.0 * 2.0 + 7.0 * 3.0, args, 2, payloadVjp, payload);
    }();
    EXPECT_EQ(tape.size(), before + 1u);
    tape.reverse(f.node);
    EXPECT_DOUBLE_EQ(tape.adjoint(x.node), 5.0);
    EXPECT_DOUBLE_EQ(tape.adjoint(y.node), 7.0);
}

TEST_F(ExternalOpsTest, ResetToTruncatesExternalRecords) {
    RevTape& tape = RevTape::active();
    RevScalar x(1.5, tape.input(1.5));
    RevScalar y(2.0, tape.input(2.0));
    const std::size_t pos = tape.position();
    const RevScalar dropped = fma(x, y, RevScalar(1.0));
    EXPECT_EQ(tape.size() + 1u, tape.position());
    tape.resetTo(pos);
    EXPECT_EQ(tape.position(), pos);
    const RevScalar z(4.0, tape.input(4.0));
    const RevScalar f = fma(x, y, z);
    tape.reverse(f.node);
    EXPECT_DOUBLE_EQ(tape.adjoint(x.node), 2.0);
    EXPECT_DOUBLE_EQ(tape.adjoint(y.node), 1.5);
    EXPECT_DOUBLE_EQ(tape.adjoint(z.node), 1.0);
    EXPECT_DOUBLE_EQ(dropped.value, std::fma(1.5, 2.0, 1.0));
}

TEST_F(ExternalOpsTest, ClearReleasesExternalState) {
    RevTape& tape = RevTape::active();
    RevScalar x(1.0, tape.input(1.0));
    RevScalar y(2.0, tape.input(2.0));
    const RevScalar f = fma(x, y, RevScalar(0.0));
    EXPECT_EQ(tape.size() + 1u, tape.position());
    tape.clear();
    EXPECT_EQ(tape.size(), 0u);
    RevScalar a(1.0, tape.input(1.0));
    RevScalar b(3.0, tape.input(3.0));
    const RevScalar g = fma(a, b, RevScalar(0.5));
    tape.reverse(g.node);
    EXPECT_DOUBLE_EQ(tape.adjoint(a.node), 3.0);
    EXPECT_DOUBLE_EQ(tape.adjoint(b.node), 1.0);
    EXPECT_DOUBLE_EQ(f.value, 2.0);
    EXPECT_EQ(tape.size(), tape.position() - 1);
}

} // namespace
