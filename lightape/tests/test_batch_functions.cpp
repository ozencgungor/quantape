#include <array>
#include <bit>
#include <cstddef>
#include <cstdint>

#include "lightape/batch/BatchFunctions.h"
#include "lightape/lightape.h"
#include <gtest/gtest.h>

namespace {

using lightape::BatchScalar;
using lightape::BatchTape;
using lightape::RevScalar;
using lightape::RevTape;
using lightape::batch::logSumExp;
using lightape::batch::smoothMax;
using lightape::batch::softmax;
using lightape::batch::softplus;

/// Raw IEEE-754 bit pattern: the comparison the bitwise contract needs (plain
/// double `==` would hide a signed-zero or NaN-payload difference).
std::uint64_t Bits(double v) {
    return std::bit_cast<std::uint64_t>(v);
}

/// `base + step * lane`, so W>1 lanes carry different values.
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

/// Scalar counterpart of `lightape::batch::softplus` recorded with the same
/// statement order. The batch tape has no `Log1p` opcode, so both sides
/// compose `log1p(u)` as `log(1 + u)` and cap the `exp` argument at the guard.
template <typename S>
S ScalarSoftplus(const S& x) {
    using std::exp;
    using std::fmin;
    using std::log;
    const S capped = fmin(x, S(30.0));
    const S y = log(S(1.0) + exp(capped));
    return x > S(30.0) ? x : y;
}

/// Scalar counterpart of `lightape::batch::smoothMax` recorded with the same
/// statement order (`log1p` composed as `log(1 + u)`).
template <typename S>
S ScalarSmoothMax(const S& a, const S& b, double k) {
    using std::exp;
    using std::fabs;
    using std::fmax;
    using std::log;
    const S kk(k);
    const S m = fmax(a, b);
    return m + log(S(1.0) + exp(-fabs(a - b) * kk)) / kk;
}

template <std::size_t W, std::size_t N>
void CheckLogSumExpBitwiseFor(const std::array<std::array<double, W>, N>& in) {
    const std::array<double, W> seed = Seeds<W>();

    BatchTape<W>& tape = BatchTape<W>::active();
    tape.clear();
    std::array<BatchScalar<W>, N> bx{};
    for (std::size_t i = 0; i < N; ++i) {
        bx[i] = tape.input(in[i]);
    }
    const BatchScalar<W> blse = logSumExp(bx.data(), N);
    tape.reverse(blse.node, seed);
    const std::array<double, W> bvalue = blse.value;
    std::array<std::array<double, W>, N> badj{};
    for (std::size_t i = 0; i < N; ++i) {
        badj[i] = tape.adjoint(bx[i].node);
    }

    RevTape& scalar = RevTape::active();
    for (std::size_t l = 0; l < W; ++l) {
        scalar.clear();
        std::array<RevScalar, N> sx{};
        for (std::size_t i = 0; i < N; ++i) {
            sx[i] = RevScalar(in[i][l], scalar.input(in[i][l]));
        }
        const RevScalar slse = lightape::logSumExp(sx.data(), N);
        scalar.reverse(slse.node, seed[l]);
        EXPECT_EQ(Bits(bvalue[l]), Bits(slse.value)) << "lane " << l;
        for (std::size_t i = 0; i < N; ++i) {
            EXPECT_EQ(Bits(badj[i][l]), Bits(scalar.adjoint(sx[i].node)))
                << "lane " << l << " component " << i;
        }
    }
    tape.clear();
    scalar.clear();
}

template <std::size_t W>
void CheckLogSumExpBitwise() {
    constexpr std::size_t N = 5;
    std::array<std::array<double, W>, N> in{};
    for (std::size_t i = 0; i < N; ++i) {
        in[i] = Lane<W>(0.4 * static_cast<double>(i) - 0.6, 0.07 * static_cast<double>(i + 1));
    }
    CheckLogSumExpBitwiseFor<W, N>(in);
}

TEST(BatchFunctionsTest, LogSumExpMatchesScalarBitwise) {
    CheckLogSumExpBitwise<1>();
    CheckLogSumExpBitwise<2>();
    CheckLogSumExpBitwise<4>();
    CheckLogSumExpBitwise<8>();
}

TEST(BatchFunctionsTest, LogSumExpExtremeShiftsMatchScalarBitwise) {
    constexpr std::size_t W = 8;
    std::array<std::array<double, W>, 4> large{};
    for (std::size_t i = 0; i < large.size(); ++i) {
        large[i] =
            Lane<W>(700.0 + 300.0 * static_cast<double>(i), 0.5 * static_cast<double>(i + 1));
    }
    CheckLogSumExpBitwiseFor<W, large.size()>(large);

    const std::array<std::array<double, W>, 3> huge = {Lane<W>(1e300, 1e299),
                                                       Lane<W>(-1e300, 2e299), Lane<W>(0.5, -3.0)};
    CheckLogSumExpBitwiseFor<W, huge.size()>(huge);
}

template <std::size_t W>
void CheckSoftmaxBitwise() {
    constexpr std::size_t N = 5;
    std::array<std::array<double, W>, N> in{};
    std::array<std::array<double, W>, N> weight{};
    for (std::size_t i = 0; i < N; ++i) {
        in[i] = Lane<W>(1000.0 + 0.25 * static_cast<double>(i), 0.13 * static_cast<double>(i + 1));
        weight[i] = Lane<W>(0.75 - 0.2 * static_cast<double>(i), 0.05 * static_cast<double>(i + 1));
    }
    const std::array<double, W> seed = Seeds<W>();

    BatchTape<W>& tape = BatchTape<W>::active();
    tape.clear();
    std::array<BatchScalar<W>, N> bx{};
    for (std::size_t i = 0; i < N; ++i) {
        bx[i] = tape.input(in[i]);
    }
    std::array<BatchScalar<W>, N> bout{};
    softmax(bx.data(), N, bout.data());
    BatchScalar<W> bf(0.0);
    for (std::size_t i = 0; i < N; ++i) {
        bf = bf + bout[i] * BatchScalar<W>(weight[i]);
    }
    tape.reverse(bf.node, seed);
    const std::array<double, W> bvalue = bf.value;
    std::array<std::array<double, W>, N> badj{};
    std::array<std::array<double, W>, N> boutValue{};
    for (std::size_t i = 0; i < N; ++i) {
        badj[i] = tape.adjoint(bx[i].node);
        boutValue[i] = bout[i].value;
    }

    RevTape& scalar = RevTape::active();
    for (std::size_t l = 0; l < W; ++l) {
        scalar.clear();
        std::array<RevScalar, N> sx{};
        for (std::size_t i = 0; i < N; ++i) {
            sx[i] = RevScalar(in[i][l], scalar.input(in[i][l]));
        }
        std::array<RevScalar, N> sout{};
        lightape::softmax(sx.data(), N, sout.data());
        RevScalar sf(0.0);
        for (std::size_t i = 0; i < N; ++i) {
            sf = sf + sout[i] * RevScalar(weight[i][l]);
        }
        scalar.reverse(sf.node, seed[l]);
        EXPECT_EQ(Bits(bvalue[l]), Bits(sf.value)) << "lane " << l;
        for (std::size_t i = 0; i < N; ++i) {
            EXPECT_EQ(Bits(boutValue[i][l]), Bits(sout[i].value))
                << "lane " << l << " component " << i;
            EXPECT_EQ(Bits(badj[i][l]), Bits(scalar.adjoint(sx[i].node)))
                << "lane " << l << " component " << i;
        }
    }
    tape.clear();
    scalar.clear();
}

TEST(BatchFunctionsTest, SoftmaxMatchesScalarBitwise) {
    CheckSoftmaxBitwise<1>();
    CheckSoftmaxBitwise<2>();
    CheckSoftmaxBitwise<4>();
    CheckSoftmaxBitwise<8>();
}

template <std::size_t W>
void CheckSoftplusBitwise(const std::array<double, W>& xv) {
    const std::array<double, W> seed = Seeds<W>();
    BatchTape<W>& tape = BatchTape<W>::active();
    tape.clear();
    const BatchScalar<W> bx = tape.input(xv);
    const BatchScalar<W> bf = softplus(bx);
    tape.reverse(bf.node, seed);
    const std::array<double, W> bvalue = bf.value;
    const std::array<double, W> badj = tape.adjoint(bx.node);

    RevTape& scalar = RevTape::active();
    for (std::size_t l = 0; l < W; ++l) {
        scalar.clear();
        const RevScalar sx(xv[l], scalar.input(xv[l]));
        const RevScalar sf = ScalarSoftplus(sx);
        scalar.reverse(sf.node, seed[l]);
        EXPECT_EQ(Bits(bvalue[l]), Bits(sf.value)) << "lane " << l << " value";
        EXPECT_EQ(Bits(badj[l]), Bits(scalar.adjoint(sx.node))) << "lane " << l << " adjoint";
    }
    tape.clear();
    scalar.clear();
}

TEST(BatchFunctionsTest, SoftplusMatchesScalarBitwise) {
    CheckSoftplusBitwise<1>(Lane<1>(0.5, 0.0));
    CheckSoftplusBitwise<1>(Lane<1>(40.0, 0.0));
    CheckSoftplusBitwise<2>(Lane<2>(26.0, 2.0)); // all <= 30
    CheckSoftplusBitwise<4>(Lane<4>(26.0, 2.0)); // straddles the guard
    CheckSoftplusBitwise<8>(Lane<8>(26.0, 2.0)); // straddles the guard
    CheckSoftplusBitwise<8>(Lane<8>(33.0, 1.0)); // all > 30
    CheckSoftplusBitwise<8>(Lane<8>(-30.0, 0.5));
    // Mixed lanes with values that would overflow an unguarded `exp`.
    const std::array<double, 4> huge = {-3.0, 7.5, 1e300, 3e299};
    CheckSoftplusBitwise<4>(huge);
}

template <std::size_t W>
void CheckSmoothMaxBitwise(const std::array<double, W>& av, const std::array<double, W>& bv,
                           double k) {
    const std::array<double, W> seed = Seeds<W>();
    BatchTape<W>& tape = BatchTape<W>::active();
    tape.clear();
    const BatchScalar<W> ba = tape.input(av);
    const BatchScalar<W> bb = tape.input(bv);
    const BatchScalar<W> bf = smoothMax(ba, bb, k);
    tape.reverse(bf.node, seed);
    const std::array<double, W> bvalue = bf.value;
    const std::array<double, W> badjA = tape.adjoint(ba.node);
    const std::array<double, W> badjB = tape.adjoint(bb.node);

    RevTape& scalar = RevTape::active();
    for (std::size_t l = 0; l < W; ++l) {
        scalar.clear();
        const RevScalar sa(av[l], scalar.input(av[l]));
        const RevScalar sb(bv[l], scalar.input(bv[l]));
        const RevScalar sf = ScalarSmoothMax(sa, sb, k);
        scalar.reverse(sf.node, seed[l]);
        EXPECT_EQ(Bits(bvalue[l]), Bits(sf.value)) << "lane " << l << " value";
        EXPECT_EQ(Bits(badjA[l]), Bits(scalar.adjoint(sa.node))) << "lane " << l << " a";
        EXPECT_EQ(Bits(badjB[l]), Bits(scalar.adjoint(sb.node))) << "lane " << l << " b";
    }
    tape.clear();
    scalar.clear();
}

TEST(BatchFunctionsTest, SmoothMaxMatchesScalarBitwise) {
    CheckSmoothMaxBitwise<1>(Lane<1>(0.5, 0.0), Lane<1>(-0.5, 0.0), 1.0);
    CheckSmoothMaxBitwise<2>(Lane<2>(0.5, 0.4), Lane<2>(-0.5, 0.3), 1.0);
    CheckSmoothMaxBitwise<4>(Lane<4>(0.5, 0.4), Lane<4>(-0.5, 0.3), 2.5);
    CheckSmoothMaxBitwise<8>(Lane<8>(-1.0, 0.5), Lane<8>(1.0, -0.5), 1.25); // mixed winners
    CheckSmoothMaxBitwise<8>(Lane<8>(1.5, 0.0), Lane<8>(1.5, 0.0), 3.0);    // tie
}

/// n = 1: the reductions are the identity/lse-of-one, verified lane-wise
/// against the scalar tapes.
template <std::size_t W>
void CheckSingleInputBitwise() {
    const std::array<double, W> xv = Lane<W>(0.75, 0.5);
    const std::array<double, W> seed = Seeds<W>();

    BatchTape<W>& tape = BatchTape<W>::active();
    tape.clear();
    const BatchScalar<W> bx = tape.input(xv);
    const BatchScalar<W> blse = logSumExp(&bx, 1);
    tape.reverse(blse.node, seed);
    const std::array<double, W> bvalue = blse.value;
    const std::array<double, W> badj = tape.adjoint(bx.node);

    tape.clear();
    const BatchScalar<W> bx2 = tape.input(xv);
    std::array<BatchScalar<W>, 1> bout{};
    softmax(&bx2, 1, bout.data());
    const BatchScalar<W> bf = bout[0] * BatchScalar<W>(1.5);
    tape.reverse(bf.node, seed);
    const std::array<double, W> bsoft = bout[0].value;
    const std::array<double, W> bsoftAdj = tape.adjoint(bx2.node);

    RevTape& scalar = RevTape::active();
    for (std::size_t l = 0; l < W; ++l) {
        scalar.clear();
        const RevScalar sx(xv[l], scalar.input(xv[l]));
        const RevScalar slse = lightape::logSumExp(&sx, 1);
        scalar.reverse(slse.node, seed[l]);
        EXPECT_EQ(Bits(bvalue[l]), Bits(slse.value)) << "lane " << l << " lse value";
        EXPECT_EQ(Bits(badj[l]), Bits(scalar.adjoint(sx.node))) << "lane " << l << " lse adjoint";

        scalar.clear();
        const RevScalar sx2(xv[l], scalar.input(xv[l]));
        std::array<RevScalar, 1> sout{};
        lightape::softmax(&sx2, 1, sout.data());
        const RevScalar sf = sout[0] * RevScalar(1.5);
        scalar.reverse(sf.node, seed[l]);
        EXPECT_EQ(Bits(bsoft[l]), Bits(sout[0].value)) << "lane " << l << " softmax value";
        EXPECT_EQ(Bits(bsoftAdj[l]), Bits(scalar.adjoint(sx2.node)))
            << "lane " << l << " softmax adjoint";
    }
    tape.clear();
    scalar.clear();
}

TEST(BatchFunctionsTest, SingleInputMatchesScalarBitwise) {
    CheckSingleInputBitwise<1>();
    CheckSingleInputBitwise<2>();
    CheckSingleInputBitwise<4>();
    CheckSingleInputBitwise<8>();
}

TEST(BatchFunctionsTest, EmptyRangeIsZeroConstant) {
    BatchTape<4>& tape = BatchTape<4>::active();
    tape.clear();
    const BatchScalar<4> r = logSumExp<4>(nullptr, 0);
    EXPECT_EQ(r.node, BatchTape<4>::kNone);
    for (std::size_t l = 0; l < 4; ++l) {
        EXPECT_EQ(Bits(r.value[l]), Bits(0.0)) << "lane " << l;
    }
    std::array<BatchScalar<4>, 1> out{};
    softmax<4>(nullptr, 0, out.data());
    EXPECT_EQ(tape.size(), 0u);
    tape.clear();
}

/// Node-less per-lane constants: every function must collapse to a node-less
/// constant (no tape traffic) with per-lane values equal to the scalar path.
template <std::size_t W>
void CheckConstantLanes() {
    constexpr std::size_t N = 4;
    std::array<std::array<double, W>, N> in{};
    for (std::size_t i = 0; i < N; ++i) {
        in[i] = Lane<W>(0.25 + 0.5 * static_cast<double>(i), 0.1 * static_cast<double>(i + 1));
    }
    std::array<BatchScalar<W>, N> cx{};
    for (std::size_t i = 0; i < N; ++i) {
        cx[i] = BatchScalar<W>(in[i]);
    }

    BatchTape<W>& tape = BatchTape<W>::active();
    tape.clear();
    const BatchScalar<W> blse = logSumExp(cx.data(), N);
    std::array<BatchScalar<W>, N> bout{};
    softmax(cx.data(), N, bout.data());
    const BatchScalar<W> bsoft = softplus(cx[0]);
    const BatchScalar<W> bsmooth = smoothMax(cx[0], cx[1], 1.5);
    EXPECT_EQ(blse.node, BatchTape<W>::kNone);
    EXPECT_EQ(bsoft.node, BatchTape<W>::kNone);
    EXPECT_EQ(bsmooth.node, BatchTape<W>::kNone);
    for (std::size_t i = 0; i < N; ++i) {
        EXPECT_EQ(bout[i].node, BatchTape<W>::kNone);
    }
    EXPECT_EQ(tape.size(), 0u);

    for (std::size_t l = 0; l < W; ++l) {
        std::array<RevScalar, N> sx{};
        for (std::size_t i = 0; i < N; ++i) {
            sx[i] = RevScalar(in[i][l]);
        }
        const RevScalar slse = lightape::logSumExp(sx.data(), N);
        EXPECT_EQ(Bits(blse.value[l]), Bits(slse.value)) << "lane " << l << " lse";
        std::array<RevScalar, N> sout{};
        lightape::softmax(sx.data(), N, sout.data());
        for (std::size_t i = 0; i < N; ++i) {
            EXPECT_EQ(Bits(bout[i].value[l]), Bits(sout[i].value))
                << "lane " << l << " component " << i;
        }
        EXPECT_EQ(Bits(bsoft.value[l]), Bits(ScalarSoftplus(RevScalar(in[0][l])).value))
            << "lane " << l << " softplus";
        EXPECT_EQ(Bits(bsmooth.value[l]),
                  Bits(ScalarSmoothMax(RevScalar(in[0][l]), RevScalar(in[1][l]), 1.5).value))
            << "lane " << l << " smoothMax";
    }

    // All lanes above the softplus guard: the constant select returns x.
    const std::array<double, W> big = Lane<W>(34.0, 0.5);
    const BatchScalar<W> bsoftBig = softplus(BatchScalar<W>(big));
    EXPECT_EQ(bsoftBig.node, BatchTape<W>::kNone);
    EXPECT_EQ(tape.size(), 0u);
    for (std::size_t l = 0; l < W; ++l) {
        EXPECT_EQ(Bits(bsoftBig.value[l]), Bits(big[l])) << "lane " << l;
    }
    tape.clear();
}

TEST(BatchFunctionsTest, ConstantLanesCollapse) {
    CheckConstantLanes<1>();
    CheckConstantLanes<2>();
    CheckConstantLanes<4>();
    CheckConstantLanes<8>();
}

/// Perturbing one lane must leave every other lane's value and adjoint
/// bitwise unchanged (lanes share the instruction stream, never data).
template <std::size_t W>
void CheckCrossLaneIndependence() {
    constexpr std::size_t N = 4;
    const std::array<double, W> seed = Seeds<W>();

    std::array<std::array<double, W>, N> inA{};
    for (std::size_t i = 0; i < N; ++i) {
        inA[i] = Lane<W>(0.3 * static_cast<double>(i) - 0.4, 0.05 * static_cast<double>(i + 1));
    }
    std::array<std::array<double, W>, N> inB = inA;
    for (std::size_t i = 0; i < N; ++i) {
        inB[i][0] += 7.25;
    }

    BatchTape<W>& tape = BatchTape<W>::active();
    tape.clear();
    std::array<BatchScalar<W>, N> ax{};
    for (std::size_t i = 0; i < N; ++i) {
        ax[i] = tape.input(inA[i]);
    }
    const BatchScalar<W> alf = logSumExp(ax.data(), N);
    tape.reverse(alf.node, seed);
    const std::array<double, W> aValue = alf.value;
    std::array<std::array<double, W>, N> aAdj{};
    for (std::size_t i = 0; i < N; ++i) {
        aAdj[i] = tape.adjoint(ax[i].node);
    }

    tape.clear();
    std::array<BatchScalar<W>, N> bx{};
    for (std::size_t i = 0; i < N; ++i) {
        bx[i] = tape.input(inB[i]);
    }
    const BatchScalar<W> blf = logSumExp(bx.data(), N);
    tape.reverse(blf.node, seed);
    const std::array<double, W> bValue = blf.value;
    std::array<std::array<double, W>, N> bAdj{};
    for (std::size_t i = 0; i < N; ++i) {
        bAdj[i] = tape.adjoint(bx[i].node);
    }

    for (std::size_t l = 1; l < W; ++l) {
        EXPECT_EQ(Bits(aValue[l]), Bits(bValue[l])) << "lse lane " << l;
        for (std::size_t i = 0; i < N; ++i) {
            EXPECT_EQ(Bits(aAdj[i][l]), Bits(bAdj[i][l])) << "lse lane " << l << " comp " << i;
        }
    }

    // softplus: lane 0 flips from the `x` branch to the composite branch.
    const std::array<double, W> xA = Lane<W>(31.0, 1.25);
    std::array<double, W> xB = xA;
    xB[0] = -4.0;

    tape.clear();
    const BatchScalar<W> spaIn = tape.input(xA);
    const BatchScalar<W> spa = softplus(spaIn);
    tape.reverse(spa.node, seed);
    const std::array<double, W> spaValue = spa.value;
    const std::array<double, W> spaAdj = tape.adjoint(spaIn.node);

    tape.clear();
    const BatchScalar<W> spbIn = tape.input(xB);
    const BatchScalar<W> spb = softplus(spbIn);
    tape.reverse(spb.node, seed);
    const std::array<double, W> spbValue = spb.value;
    const std::array<double, W> spbAdj = tape.adjoint(spbIn.node);

    for (std::size_t l = 1; l < W; ++l) {
        EXPECT_EQ(Bits(spaValue[l]), Bits(spbValue[l])) << "softplus lane " << l;
        EXPECT_EQ(Bits(spaAdj[l]), Bits(spbAdj[l])) << "softplus lane " << l;
    }
    tape.clear();
}

TEST(BatchFunctionsTest, CrossLaneIndependence) {
    CheckCrossLaneIndependence<2>();
    CheckCrossLaneIndependence<4>();
    CheckCrossLaneIndependence<8>();
}

} // namespace
