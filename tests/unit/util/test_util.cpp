// test_util.cpp — util/FloatingPoint.h, util/Constants.h, util/Numeric.h gates
#include "quantape/util/Constants.h"
#include "quantape/util/FloatingPoint.h"
#include "quantape/util/Numeric.h"

#include <Eigen/Dense>

#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <random>
#include <vector>

#include "support/GtestSupport.h"

using quantape::util::alignUp;
using quantape::util::ceilDiv;
using quantape::util::clamp;
using quantape::util::clamp01;
using quantape::util::cube;
using quantape::util::HiLo;
using quantape::util::isClose;
using quantape::util::isCloseAbs;
using quantape::util::isFiniteBitwise;
using quantape::util::isInfBitwise;
using quantape::util::isNanBitwise;
using quantape::util::isPowerOfTwo;
using quantape::util::lerp;
using quantape::util::safeDiv;
using quantape::util::signum;
using quantape::util::sqr;
using quantape::util::unitSign;

namespace {

const double kInf = std::numeric_limits<double>::infinity();
const double kNaN = std::numeric_limits<double>::quiet_NaN();

} // namespace

TEST(FpPredicates, edgesMatchLibmClassification) {
    EXPECT_TRUE(isFiniteBitwise(0.0));
    EXPECT_TRUE(isFiniteBitwise(-0.0));
    EXPECT_TRUE(isFiniteBitwise(-1.0e308));
    EXPECT_TRUE(isFiniteBitwise(std::numeric_limits<double>::denorm_min()));
    EXPECT_TRUE(isFiniteBitwise(std::numeric_limits<double>::max()));
    EXPECT_FALSE(isFiniteBitwise(kInf));
    EXPECT_FALSE(isFiniteBitwise(-kInf));
    EXPECT_FALSE(isFiniteBitwise(kNaN));
    EXPECT_TRUE(isInfBitwise(kInf));
    EXPECT_TRUE(isInfBitwise(-kInf));
    EXPECT_FALSE(isInfBitwise(std::numeric_limits<double>::max()));
    EXPECT_FALSE(isInfBitwise(kNaN));
    EXPECT_TRUE(isNanBitwise(kNaN));
    EXPECT_FALSE(isNanBitwise(kInf));
    EXPECT_FALSE(isNanBitwise(0.0));
}

TEST(FpPredicates, randomBitPatternsMatchLibm) {
    std::mt19937_64 rng(20260929);
    for (int i = 0; i < 200000; ++i) {
        const std::uint64_t bits = rng();
        double x = 0.0;
        std::memcpy(&x, &bits, sizeof(x));
        EXPECT_TRUE(isFiniteBitwise(x) == std::isfinite(x));
        EXPECT_TRUE(isNanBitwise(x) == std::isnan(x));
        EXPECT_TRUE(isInfBitwise(x) == std::isinf(x));
    }
}

// isClose: NumPy formula |a-b| <= atol + rtol|b| and the absolute floor.
TEST(IsClose, numpyFormulaAndAbsoluteFloor) {
    EXPECT_TRUE(isClose(1.0, 1.0));
    EXPECT_TRUE(isClose(1.0, 1.0 + 1e-13));
    EXPECT_FALSE(isClose(1.0, 1.0 + 1e-10));
    EXPECT_TRUE(isClose(-1.0, -1.0 - 1e-13));
    EXPECT_TRUE(isClose(0.0, 1e-16)); // absolute floor
    EXPECT_FALSE(isClose(0.0, 1e-3));
    EXPECT_TRUE(isClose(100.0, 100.1, 1e-3, 0.0)); // rtol only: 0.1 <= 1e-3 * 100.1
    EXPECT_FALSE(isClose(100.0, 100.2, 1e-3, 0.0));
    EXPECT_TRUE(isCloseAbs(2.0, 2.0 + 1e-16, 1e-15));
    EXPECT_FALSE(isCloseAbs(2.0, 2.0 + 1e-13, 1e-15));
}

TEST(IsClose, nanAndInfinityRules) {
    EXPECT_TRUE(isClose(kNaN, kNaN, 1e-12, 1e-15, true));
    EXPECT_FALSE(isClose(kNaN, kNaN));
    EXPECT_FALSE(isClose(kNaN, 1.0, 1e-12, 1e-15, true));
    EXPECT_TRUE(isClose(kInf, kInf));
    EXPECT_FALSE(isClose(kInf, -kInf));
    EXPECT_FALSE(isClose(kInf, 1.0));
    EXPECT_FALSE(isCloseAbs(kNaN, kNaN, 1.0));
    EXPECT_TRUE(isCloseAbs(kInf, kInf, 0.0));
}

// Constants: HiLo split is correctly rounded and d-d ready.
TEST(Constants, hiloSplitsAndExactRelations) {
    const double eps = std::numeric_limits<double>::epsilon();
    const quantape::util::HiLo all[] = {
        quantape::util::kPi,        quantape::util::kTwoPi,   quantape::util::kHalfPi,
        quantape::util::kQuarterPi, quantape::util::kInvPi,   quantape::util::kSqrt2,
        quantape::util::kInvSqrt2,  quantape::util::kSqrt2Pi, quantape::util::kInvSqrt2Pi,
        quantape::util::kLn2,       quantape::util::kLn10,    quantape::util::kE,
        quantape::util::kEulerGamma};
    for (const HiLo& c : all) {
        EXPECT_GT(c.hi, 0.0);
        EXPECT_LE(std::fabs(c.lo), eps * c.hi); // |residual| <= one ulp
    }
    EXPECT_GT(quantape::util::kPi.lo, 0.0);    // pi rounds down
    EXPECT_LT(quantape::util::kSqrt2.lo, 0.0); // sqrt(2) rounds up
    EXPECT_LT(quantape::util::kInvSqrt2.lo, 0.0);
    EXPECT_LT(quantape::util::kInvPi.lo, 0.0);
    EXPECT_GT(quantape::util::kLn2.lo, 0.0);
    EXPECT_LT(quantape::util::kLn10.lo, 0.0);
    EXPECT_GT(quantape::util::kE.lo, 0.0);

    // Exact power-of-two scaling relations (rounding commutes with ×2, ÷2).
    EXPECT_EQ(quantape::util::kTwoPi.hi, 2.0 * quantape::util::kPi.hi);
    EXPECT_EQ(quantape::util::kTwoPi.lo, 2.0 * quantape::util::kPi.lo);
    EXPECT_EQ(quantape::util::kHalfPi.hi, quantape::util::kPi.hi / 2.0);
    EXPECT_EQ(quantape::util::kHalfPi.lo, quantape::util::kPi.lo / 2.0);
    EXPECT_EQ(quantape::util::kQuarterPi.hi, quantape::util::kPi.hi / 4.0);

    // Implicit conversion yields the rounded double.
    const double pi = quantape::util::kPi;
    EXPECT_EQ(pi, quantape::util::kPi.hi);

    // Cross-consistency of independent constants (≈1 ulp of the expressions).
    EXPECT_TRUE(
        isClose(quantape::util::kSqrt2.hi * quantape::util::kInvSqrt2.hi, 1.0, 1e-15, 1e-15));
    EXPECT_TRUE(
        isClose(quantape::util::kSqrt2Pi.hi * quantape::util::kInvSqrt2Pi.hi, 1.0, 1e-15, 1e-15));
    EXPECT_TRUE(isClose(quantape::util::kPi.hi * quantape::util::kInvPi.hi, 1.0, 1e-15, 1e-15));
    EXPECT_TRUE(isClose(quantape::util::kSqrt2.hi * quantape::util::kSqrt2.hi, 2.0, 1e-15, 1e-15));
    EXPECT_TRUE(isClose(quantape::util::kE.hi, std::exp(1.0), 1e-15, 1e-15));
    EXPECT_TRUE(isClose(quantape::util::kLn2.hi, std::log(2.0), 1e-15, 1e-15));
    EXPECT_TRUE(isClose(quantape::util::kLn10.hi, std::log(10.0), 1e-15, 1e-15));
    EXPECT_TRUE(isClose(quantape::util::kHalfPi.hi, std::asin(1.0), 1e-15, 1e-15));
}

TEST(NumericHelpers, clampSignPowersAndLerp) {
    EXPECT_EQ(clamp(5.0, 0.0, 1.0), 1.0);
    EXPECT_EQ(clamp(-5.0, 0.0, 1.0), 0.0);
    EXPECT_EQ(clamp(0.25, 0.0, 1.0), 0.25);
    EXPECT_EQ(clamp01(-2.0), 0.0);
    EXPECT_EQ(clamp01(3.0), 1.0);
    EXPECT_EQ(signum(-3.0), -1);
    EXPECT_EQ(signum(0.0), 0);
    EXPECT_EQ(signum(-0.0), 0);
    EXPECT_EQ(signum(2.5), 1);
    EXPECT_EQ(signum(kNaN), 0);
    EXPECT_EQ(unitSign(0.0), 1.0);
    EXPECT_EQ(unitSign(-0.0), 1.0);
    EXPECT_EQ(unitSign(-2.5), -1.0);
    EXPECT_EQ(sqr(-3.0), 9.0);
    EXPECT_EQ(cube(-2.0), -8.0);
    EXPECT_EQ(safeDiv(1.0, 2.0), 0.5);
    EXPECT_EQ(safeDiv(1.0, 0.0), 0.0);
    EXPECT_EQ(safeDiv(1.0, 0.0, 42.0), 42.0);
    EXPECT_EQ(lerp(0.0, 10.0, 0.25), 2.5);
    EXPECT_EQ(lerp(-1.0, 1.0, 0.5), 0.0);
    EXPECT_EQ(ceilDiv(10, 3), 4);
    EXPECT_EQ(ceilDiv(9, 3), 3);
    EXPECT_EQ(ceilDiv(0, 5), 0);
    EXPECT_TRUE(isPowerOfTwo(1));
    EXPECT_TRUE(isPowerOfTwo(64));
    EXPECT_FALSE(isPowerOfTwo(0));
    EXPECT_FALSE(isPowerOfTwo(63));
    EXPECT_EQ(alignUp(5, 8), 8);
    EXPECT_EQ(alignUp(8, 8), 8);
    EXPECT_EQ(alignUp(0, 16), 0);
    EXPECT_EQ(alignUp(17, 16), 32);
}

// Generic checkClose success paths: scalar, iterable, matrix (via the support
// macros with the same semantics).
TEST(CheckCloseSuccess, scalarAndSequencePaths) {
    CHECK_CLOSE("scalar", 1.0, 1.0 + 1e-15, 1e-14);
    const std::vector<double> vec{1.0, 2.0, 3.0};
    CHECK_CLOSE_SEQ("vector", vec, vec, 0.0);
    const std::array<double, 3> arr{1.0, 2.0, 3.0};
    CHECK_CLOSE_SEQ("vector vs array", vec, arr, 1e-15);
    const std::array<float, 3> floats{1.0f, 2.0f, 3.0f};
    CHECK_CLOSE_SEQ("float array vs double vector", floats, vec, 1e-7);
    Eigen::VectorXd ev(3);
    ev << 1.0, 2.0, 3.0;
    CHECK_CLOSE_SEQ("eigen vector vs std vector", ev, vec, 0.0);
    Eigen::MatrixXd em(2, 2);
    em << 1.0, 2.0, 3.0, 4.0;
    CHECK_CLOSE_SEQ("eigen matrix", em, em, 0.0);
    CHECK_CLOSE_SEQ("eigen expression", (em * 2.0).eval(), (2.0 * em).eval(), 0.0);
}
