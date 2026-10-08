// test_sde_evaluate.cpp — block-batched SoA evaluation API (double)
//
// Gates:
//   - vectorized batch evaluator == per-path scalar evaluator (values)
//   - per-path lift == hand-written column loop (bitwise)
//   - multi-output blocks (PV + indicator rows) and shape validation
//   - output blocks are caller-owned: repeated calls reuse the same storage
//
// Stan-free: includes only mc/ headers + Eigen.
#include "quantape/mc/Evaluate.h"

#include <Eigen/Dense>

#include <cmath>
#include <cstddef>
#include <span>
#include <stdexcept>
#include <vector>

#include "support/GtestSupport.h"

using quantape::mc::BatchEvaluator;
using quantape::mc::evaluateBlock;
using quantape::mc::perPath;
using quantape::mc::PerPathEvaluator;
using quantape::mc::StateMatrix;

namespace {

// Two outputs: a quadratic "PV" and a call-like indicator row.
struct ScalarEvaluator {
    double strike = 0.5;
    template <typename Scalar>
    void operator()(std::span<const Scalar> state, std::span<Scalar> out) const {
        out[0] = state[0] * state[0] + Scalar(0.25) * state[1];
        out[1] = state[0] > Scalar(strike) ? state[0] - Scalar(strike) : Scalar(0.0);
    }
};

// Same map, vectorized across the block (Eigen expressions).
struct VectorizedEvaluator {
    double strike = 0.5;
    template <typename Scalar>
    void operator()(const StateMatrix<Scalar>& states, StateMatrix<Scalar>& out) const {
        out.resize(2, states.cols());
        out.row(0) =
            (states.row(0).array().square() + Scalar(0.25) * states.row(1).array()).matrix();
        out.row(1) = (states.row(0).array() - Scalar(strike)).max(Scalar(0.0)).matrix();
    }
};

static_assert(BatchEvaluator<VectorizedEvaluator, double>);
static_assert(BatchEvaluator<PerPathEvaluator<double, ScalarEvaluator>, double>);

StateMatrix<double> referenceEvaluate(const StateMatrix<double>& states,
                                      const ScalarEvaluator& evaluator) {
    StateMatrix<double> out(2, states.cols());
    const std::size_t nDims = static_cast<std::size_t>(states.rows());
    for (Eigen::Index p = 0; p < states.cols(); ++p) {
        evaluator(std::span<const double>(states.col(p).data(), nDims),
                  std::span<double>(out.col(p).data(), 2));
    }
    return out;
}

} // namespace

TEST(SdeEvaluate, vectorizedBatchMatchesPerPath) {
    const ScalarEvaluator scalar;
    StateMatrix<double> states(2, 37);
    for (Eigen::Index p = 0; p < states.cols(); ++p) {
        states(0, p) = 0.4 + 0.03 * static_cast<double>(p);
        states(1, p) = 0.7 - 0.01 * static_cast<double>(p);
    }

    StateMatrix<double> outBatch(2, states.cols());
    evaluateBlock<double>(VectorizedEvaluator{}, states, outBatch);
    const StateMatrix<double> reference = referenceEvaluate(states, scalar);

    ASSERT_EQ(outBatch.rows(), reference.rows());
    ASSERT_EQ(outBatch.cols(), reference.cols());
    for (Eigen::Index i = 0; i < outBatch.rows(); ++i) {
        for (Eigen::Index p = 0; p < outBatch.cols(); ++p) {
            CHECK_CLOSE("vectorized vs per-path", outBatch(i, p), reference(i, p), 1e-13);
        }
    }
}

TEST(SdeEvaluate, perPathLiftMatchesColumnLoopBitwise) {
    const ScalarEvaluator scalar;
    StateMatrix<double> states(2, 101);
    for (Eigen::Index p = 0; p < states.cols(); ++p) {
        states(0, p) = 0.5 + 0.01 * static_cast<double>(p);
        states(1, p) = 1.0 - 0.02 * static_cast<double>(p);
    }

    StateMatrix<double> outLift(2, states.cols());
    evaluateBlock<double>(perPath<double>(scalar), states, outLift);
    const StateMatrix<double> reference = referenceEvaluate(states, scalar);
    EXPECT_TRUE((outLift.array() == reference.array()).all());
}

TEST(SdeEvaluate, multiOutputRowsAndCallerOwnedReuse) {
    const ScalarEvaluator scalar;
    StateMatrix<double> states(2, 16);
    for (Eigen::Index p = 0; p < states.cols(); ++p) {
        states(0, p) = 0.1 * static_cast<double>(p);
        states(1, p) = 0.5;
    }

    StateMatrix<double> out(2, states.cols());
    const double* dataBefore = out.data();
    for (int repeat = 0; repeat < 3; ++repeat) {
        evaluateBlock(perPath<double>(scalar), states, out);
        EXPECT_EQ(out.data(), dataBefore); // no reallocation across calls
    }
    // Row 1 is the call-like indicator: strictly below/above the strike.
    for (Eigen::Index p = 0; p < states.cols(); ++p) {
        const double expected = std::max(states(0, p) - scalar.strike, 0.0);
        CHECK_CLOSE("indicator row", out(1, p), expected, 0.0);
    }
}

TEST(SdeEvaluate, shapeMismatchThrows) {
    const ScalarEvaluator scalar;
    StateMatrix<double> states(2, 5);
    StateMatrix<double> out(2, 4);
    EXPECT_THROW(evaluateBlock(perPath<double>(scalar), states, out), std::invalid_argument);
}
