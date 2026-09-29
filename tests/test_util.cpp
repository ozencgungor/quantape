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

#include "quantape/log/Log.h"
#include "quantape/util/Check.h"

using quantape::util::alignUp;
using quantape::util::ceilDiv;
using quantape::util::checkClose;
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

int main() {
    // ── fp predicates: edges and libm cross-check on random bit patterns ──
    CHECK(isFiniteBitwise(0.0));
    CHECK(isFiniteBitwise(-0.0));
    CHECK(isFiniteBitwise(-1.0e308));
    CHECK(isFiniteBitwise(std::numeric_limits<double>::denorm_min()));
    CHECK(isFiniteBitwise(std::numeric_limits<double>::max()));
    CHECK(!isFiniteBitwise(kInf));
    CHECK(!isFiniteBitwise(-kInf));
    CHECK(!isFiniteBitwise(kNaN));
    CHECK(isInfBitwise(kInf));
    CHECK(isInfBitwise(-kInf));
    CHECK(!isInfBitwise(std::numeric_limits<double>::max()));
    CHECK(!isInfBitwise(kNaN));
    CHECK(isNanBitwise(kNaN));
    CHECK(!isNanBitwise(kInf));
    CHECK(!isNanBitwise(0.0));

    std::mt19937_64 rng(20260929);
    for (int i = 0; i < 200000; ++i) {
        const std::uint64_t bits = rng();
        double x = 0.0;
        std::memcpy(&x, &bits, sizeof(x));
        CHECK(isFiniteBitwise(x) == std::isfinite(x));
        CHECK(isNanBitwise(x) == std::isnan(x));
        CHECK(isInfBitwise(x) == std::isinf(x));
    }

    // ── isClose: NumPy formula |a-b| <= atol + rtol|b| and NaN/Inf rules ──
    CHECK(isClose(1.0, 1.0));
    CHECK(isClose(1.0, 1.0 + 1e-13));
    CHECK(!isClose(1.0, 1.0 + 1e-10));
    CHECK(isClose(-1.0, -1.0 - 1e-13));
    CHECK(isClose(0.0, 1e-16));  // absolute floor
    CHECK(!isClose(0.0, 1e-3));
    CHECK(isClose(100.0, 100.1, 1e-3, 0.0));  // rtol only: 0.1 <= 1e-3 * 100.1
    CHECK(!isClose(100.0, 100.2, 1e-3, 0.0));
    CHECK(isClose(kNaN, kNaN, 1e-12, 1e-15, true));
    CHECK(!isClose(kNaN, kNaN));
    CHECK(!isClose(kNaN, 1.0, 1e-12, 1e-15, true));
    CHECK(isClose(kInf, kInf));
    CHECK(!isClose(kInf, -kInf));
    CHECK(!isClose(kInf, 1.0));
    CHECK(isCloseAbs(2.0, 2.0 + 1e-16, 1e-15));
    CHECK(!isCloseAbs(2.0, 2.0 + 1e-13, 1e-15));
    CHECK(!isCloseAbs(kNaN, kNaN, 1.0));
    CHECK(isCloseAbs(kInf, kInf, 0.0));

    // ── constants: HiLo split is correctly rounded and d-d ready ──
    const double eps = std::numeric_limits<double>::epsilon();
    const quantape::util::HiLo all[] = {quantape::util::kPi,
                                        quantape::util::kTwoPi,
                                        quantape::util::kHalfPi,
                                        quantape::util::kQuarterPi,
                                        quantape::util::kInvPi,
                                        quantape::util::kSqrt2,
                                        quantape::util::kInvSqrt2,
                                        quantape::util::kSqrt2Pi,
                                        quantape::util::kInvSqrt2Pi,
                                        quantape::util::kLn2,
                                        quantape::util::kLn10,
                                        quantape::util::kE,
                                        quantape::util::kEulerGamma};
    for (const HiLo& c : all) {
        CHECK(c.hi > 0.0);
        CHECK(std::fabs(c.lo) <= eps * c.hi); // |residual| <= one ulp
    }
    CHECK(quantape::util::kPi.lo > 0.0);      // pi rounds down
    CHECK(quantape::util::kSqrt2.lo < 0.0);   // sqrt(2) rounds up
    CHECK(quantape::util::kInvSqrt2.lo < 0.0);
    CHECK(quantape::util::kInvPi.lo < 0.0);
    CHECK(quantape::util::kLn2.lo > 0.0);
    CHECK(quantape::util::kLn10.lo < 0.0);
    CHECK(quantape::util::kE.lo > 0.0);

    // Exact power-of-two scaling relations (rounding commutes with ×2, ÷2).
    CHECK(quantape::util::kTwoPi.hi == 2.0 * quantape::util::kPi.hi);
    CHECK(quantape::util::kTwoPi.lo == 2.0 * quantape::util::kPi.lo);
    CHECK(quantape::util::kHalfPi.hi == quantape::util::kPi.hi / 2.0);
    CHECK(quantape::util::kHalfPi.lo == quantape::util::kPi.lo / 2.0);
    CHECK(quantape::util::kQuarterPi.hi == quantape::util::kPi.hi / 4.0);

    // Implicit conversion yields the rounded double.
    const double pi = quantape::util::kPi;
    CHECK(pi == quantape::util::kPi.hi);

    // Cross-consistency of independent constants (≈1 ulp of the expressions).
    CHECK(isClose(quantape::util::kSqrt2.hi * quantape::util::kInvSqrt2.hi, 1.0, 1e-15, 1e-15));
    CHECK(isClose(quantape::util::kSqrt2Pi.hi * quantape::util::kInvSqrt2Pi.hi, 1.0, 1e-15, 1e-15));
    CHECK(isClose(quantape::util::kPi.hi * quantape::util::kInvPi.hi, 1.0, 1e-15, 1e-15));
    CHECK(isClose(quantape::util::kSqrt2.hi * quantape::util::kSqrt2.hi, 2.0, 1e-15, 1e-15));
    CHECK(isClose(quantape::util::kE.hi, std::exp(1.0), 1e-15, 1e-15));
    CHECK(isClose(quantape::util::kLn2.hi, std::log(2.0), 1e-15, 1e-15));
    CHECK(isClose(quantape::util::kLn10.hi, std::log(10.0), 1e-15, 1e-15));
    CHECK(isClose(quantape::util::kHalfPi.hi, std::asin(1.0), 1e-15, 1e-15));

    // ── numeric helpers ──
    CHECK(clamp(5.0, 0.0, 1.0) == 1.0);
    CHECK(clamp(-5.0, 0.0, 1.0) == 0.0);
    CHECK(clamp(0.25, 0.0, 1.0) == 0.25);
    CHECK(clamp01(-2.0) == 0.0);
    CHECK(clamp01(3.0) == 1.0);
    CHECK(signum(-3.0) == -1);
    CHECK(signum(0.0) == 0);
    CHECK(signum(-0.0) == 0);
    CHECK(signum(2.5) == 1);
    CHECK(signum(kNaN) == 0);
    CHECK(unitSign(0.0) == 1.0);
    CHECK(unitSign(-0.0) == 1.0);
    CHECK(unitSign(-2.5) == -1.0);
    CHECK(sqr(-3.0) == 9.0);
    CHECK(cube(-2.0) == -8.0);
    CHECK(safeDiv(1.0, 2.0) == 0.5);
    CHECK(safeDiv(1.0, 0.0) == 0.0);
    CHECK(safeDiv(1.0, 0.0, 42.0) == 42.0);
    CHECK(lerp(0.0, 10.0, 0.25) == 2.5);
    CHECK(lerp(-1.0, 1.0, 0.5) == 0.0);
    CHECK(ceilDiv(10, 3) == 4);
    CHECK(ceilDiv(9, 3) == 3);
    CHECK(ceilDiv(0, 5) == 0);
    CHECK(isPowerOfTwo(1));
    CHECK(isPowerOfTwo(64));
    CHECK(!isPowerOfTwo(0));
    CHECK(!isPowerOfTwo(63));
    CHECK(alignUp(5, 8) == 8);
    CHECK(alignUp(8, 8) == 8);
    CHECK(alignUp(0, 16) == 0);
    CHECK(alignUp(17, 16) == 32);

    // ── generic checkClose: scalar, iterable, matrix (success paths) ──
    checkClose("scalar", 1.0, 1.0 + 1e-15, 1e-14);
    const std::vector<double> vec{1.0, 2.0, 3.0};
    checkClose("vector", vec, vec, 0.0);
    checkClose("vector vs array", vec, std::array<double, 3>{1.0, 2.0, 3.0}, 1e-15);
    const std::array<float, 3> floats{1.0f, 2.0f, 3.0f};
    checkClose("float array vs double vector", floats, vec, 1e-7);
    Eigen::VectorXd ev(3);
    ev << 1.0, 2.0, 3.0;
    checkClose("eigen vector vs std vector", ev, vec, 0.0);
    Eigen::MatrixXd em(2, 2);
    em << 1.0, 2.0, 3.0, 4.0;
    checkClose("eigen matrix", em, em, 0.0);
    checkClose("eigen expression", (em * 2.0).eval(), (2.0 * em).eval(), 0.0);

    QTA_LOG_INFO("test", "test_util: ok");
    return 0;
}
