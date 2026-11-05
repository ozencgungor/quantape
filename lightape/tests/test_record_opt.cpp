#include <cmath>
#include <cstddef>
#include <cstdint>

#include "lightape/lightape.h"
#include <gtest/gtest.h>

namespace {

using lightape::RevScalar;
using lightape::RevTape;

class RecordOptTest : public ::testing::Test {
protected:
    void SetUp() override {
        RevTape::releaseCurrentThread();
        RevTape::active();
    }
    void TearDown() override { RevTape::active().clear(); }
};

struct SquareEval {
    double value = 0.0;
    double grad = 0.0;
    double hvp = 0.0;
    std::size_t size = 0;
    bool hvpOk = false;
};

/// f(x) = (x*x) + (x*x), recorded from the same source line in either mode.
SquareEval evalSquareSum(bool cse) {
    RevTape& tape = RevTape::active();
    tape.useCse(cse);
    RevScalar x(2.0, tape.input(2.0));
    RevScalar f = (x * x) + (x * x);
    SquareEval out;
    out.value = f.value;
    out.size = tape.size();
    const std::uint32_t inputs[1] = {x.node};
    const double dir[1] = {1.0};
    out.hvpOk = tape.hessianVector(f.node, inputs, 1, dir, &out.hvp);
    lightape::gradient(tape, f.node, inputs, 1, &out.grad);
    return out;
}

struct ExtEval {
    double value = 0.0;
    double gradX = 0.0;
    double gradY = 0.0;
    double hvpX = 0.0;
    double hvpY = 0.0;
    std::size_t size = 0;
    bool hvpOk = false;
};

/// f(x, y) = fma(x, y, p) + p with p = (x*x) + (x*x): a repeated statement
/// feeding one External block and one plain add.
ExtEval evalWithExternal(bool cse) {
    RevTape& tape = RevTape::active();
    tape.useCse(cse);
    RevScalar x(1.5, tape.input(1.5));
    RevScalar y(-0.75, tape.input(-0.75));
    RevScalar p = (x * x) + (x * x);
    RevScalar e = lightape::fma(x, y, p);
    RevScalar f = e + p;
    ExtEval out;
    out.value = f.value;
    out.size = tape.size();
    const std::uint32_t inputs[2] = {x.node, y.node};
    const double dx[2] = {1.0, 0.0};
    const double dy[2] = {0.0, 1.0};
    double hx[2] = {0.0, 0.0};
    double hy[2] = {0.0, 0.0};
    out.hvpOk = tape.hessianVector(f.node, inputs, 2, dx, hx);
    out.hvpOk = tape.hessianVector(f.node, inputs, 2, dy, hy) && out.hvpOk;
    out.hvpX = hx[0];
    out.hvpY = hy[1];
    double g[2] = {0.0, 0.0};
    lightape::gradient(tape, f.node, inputs, 2, g);
    out.gradX = g[0];
    out.gradY = g[1];
    return out;
}

TEST_F(RecordOptTest, CseDisabledByDefaultRecordsDistinctNodes) {
    RevTape& tape = RevTape::active();
    EXPECT_FALSE(tape.cseEnabled());

    RevScalar x(2.0, tape.input(2.0));
    const std::size_t base = tape.position();
    RevScalar a = x * x;
    RevScalar b = x * x;
    EXPECT_NE(a.node, b.node);
    EXPECT_EQ(tape.position(), base + 2);
    EXPECT_DOUBLE_EQ(a.value, 4.0);
    EXPECT_DOUBLE_EQ(b.value, 4.0);
}

TEST_F(RecordOptTest, ConstantOperandIsNodeLess) {
    RevTape& tape = RevTape::active();
    RevScalar x(3.0, tape.input(3.0));
    const std::size_t base = tape.position();
    RevScalar y = x * 2.0;
    EXPECT_EQ(tape.position(), base + 1); // one Mul node, no constant leaf
    EXPECT_DOUBLE_EQ(y.value, 6.0);
    tape.reverse(y.node);
    EXPECT_DOUBLE_EQ(tape.adjoint(x.node), 2.0);
}

TEST_F(RecordOptTest, CseReusesNodeForRepeatedBinaryStatement) {
    RevTape& tape = RevTape::active();
    tape.useCse(true);

    RevScalar x(2.0, tape.input(2.0));
    RevScalar a = x * x;
    RevScalar b = x * x;
    EXPECT_EQ(a.node, b.node);
    EXPECT_DOUBLE_EQ(b.value, 4.0);
    EXPECT_EQ(tape.position(), 3u); // sink + input + one shared Mul

    RevScalar f = a + b;
    tape.reverse(f.node);
    EXPECT_DOUBLE_EQ(tape.adjoint(x.node), 8.0);
}

TEST_F(RecordOptTest, CseMergesRepeatedUnary) {
    RevTape& tape = RevTape::active();
    tape.useCse(true);

    RevScalar x(4.0, tape.input(4.0));
    RevScalar a = lightape::sqrt(x);
    RevScalar b = lightape::sqrt(x);
    EXPECT_EQ(a.node, b.node);
    EXPECT_DOUBLE_EQ(b.value, 2.0);

    RevScalar f = a + b; // two adjoint contributions into the shared node
    tape.reverse(f.node);
    EXPECT_DOUBLE_EQ(tape.adjoint(x.node), 0.5);
}

TEST_F(RecordOptTest, CseMatchesPlainRecording) {
    const SquareEval plain = evalSquareSum(false);
    RevTape::releaseCurrentThread();
    const SquareEval cse = evalSquareSum(true);

    EXPECT_TRUE(cse.hvpOk);
    EXPECT_EQ(cse.size, plain.size - 1); // the duplicate Mul is merged
    EXPECT_DOUBLE_EQ(cse.value, plain.value);
    EXPECT_DOUBLE_EQ(cse.grad, plain.grad);
    EXPECT_DOUBLE_EQ(cse.hvp, plain.hvp);
}

TEST_F(RecordOptTest, CseValueGuardRejectsMismatchedPrimal) {
    RevTape& tape = RevTape::active();
    tape.useCse(true);

    RevScalar x(2.0, tape.input(2.0));
    RevScalar y = x * x; // caches (Mul, x, x) -> node 2, primal 4
    EXPECT_EQ(y.node, 2u);

    // Same key, different recomputed primal: the bitwise guard must record a
    // fresh node instead of returning the cached slot.
    RevScalar stale = x;
    stale.value = 3.0;
    RevScalar z = stale * stale;
    EXPECT_NE(z.node, y.node);
    EXPECT_EQ(tape.position(), 4u);
    EXPECT_DOUBLE_EQ(z.value, 9.0);
}

TEST_F(RecordOptTest, CseWithExternalBlocksMatches) {
    const ExtEval plain = evalWithExternal(false);
    RevTape::releaseCurrentThread();
    const ExtEval cse = evalWithExternal(true);

    EXPECT_TRUE(cse.hvpOk);
    EXPECT_LT(cse.size, plain.size); // repeated (x*x) merged, External kept
    EXPECT_DOUBLE_EQ(cse.value, plain.value);
    EXPECT_DOUBLE_EQ(cse.gradX, plain.gradX);
    EXPECT_DOUBLE_EQ(cse.gradY, plain.gradY);
    EXPECT_DOUBLE_EQ(cse.hvpX, plain.hvpX);
    EXPECT_DOUBLE_EQ(cse.hvpY, plain.hvpY);
}

TEST_F(RecordOptTest, CseCacheDroppedWithRecording) {
    RevTape& tape = RevTape::active();
    tape.useCse(true);

    RevScalar x1(2.0, tape.input(2.0));
    RevScalar s1 = x1 * x1; // populates the cache
    EXPECT_DOUBLE_EQ(s1.value, 4.0);
    EXPECT_EQ(tape.position(), 3u);

    // The same key and the same primal after a truncation must not alias the
    // dead slot: resetTo() drops the cache with the recording.
    tape.resetTo(1);
    RevScalar x2(2.0, tape.input(2.0));
    RevScalar s2 = x2 * x2;
    EXPECT_EQ(tape.position(), 3u);
    EXPECT_EQ(s2.node, 2u);
    tape.reverse(s2.node);
    EXPECT_DOUBLE_EQ(tape.adjoint(x2.node), 4.0);

    // clear() keeps the useCse setting but drops the cache as well.
    tape.clear();
    EXPECT_TRUE(tape.cseEnabled());
    RevScalar x3(3.0, tape.input(3.0));
    RevScalar t1 = x3 * x3;
    RevScalar t2 = x3 * x3;
    EXPECT_EQ(t1.node, t2.node);
    EXPECT_EQ(tape.position(), 3u);
    EXPECT_DOUBLE_EQ(t2.value, 9.0);
}

} // namespace
