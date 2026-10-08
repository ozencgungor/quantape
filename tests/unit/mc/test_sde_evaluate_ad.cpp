// test_sde_evaluate_ad.cpp — block-batched SoA evaluation under AD
//
// Gates:
//   - batch (block-at-a-time) and per-path lift agree with a scalar reference
//   - values and adjoints agree for `var`
//   - values and directional derivatives agree for `fvar<var>`
//
// Uses Stan AD; parallel-safe under STAN_THREADS.
#include "quantape/math/StanMath.h"

#include "quantape/mc/Evaluate.h"

#include <Eigen/Dense>

#include <cmath>
#include <cstddef>
#include <span>
#include <vector>

#include "support/GtestSupport.h"
#include "support/StanTapeFixture.h"

using quantape::mc::evaluateBlock;
using quantape::mc::perPath;
using quantape::mc::StateMatrix;

namespace {

// Two-output scalar-generic map: a quadratic PV and an affine row.
struct ScalarEvaluator {
    template <typename Scalar>
    void operator()(std::span<const Scalar> state, std::span<Scalar> out) const {
        out[0] = state[0] * state[0] + Scalar(0.25) * state[1];
        out[1] = state[0] * state[1] - Scalar(0.75);
    }
};

// The same map evaluated block-at-a-time in one call.
struct BlockEvaluator {
    template <typename Scalar>
    void operator()(const StateMatrix<Scalar>& states, StateMatrix<Scalar>& out) const {
        out.resize(2, states.cols());
        for (Eigen::Index p = 0; p < states.cols(); ++p) {
            out(0, p) = states(0, p) * states(0, p) + Scalar(0.25) * states(1, p);
            out(1, p) = states(0, p) * states(1, p) - Scalar(0.75);
        }
    }
};

} // namespace

class SdeEvaluateAdTest : public StanTapeTest {};

TEST_F(SdeEvaluateAdTest, blockAndLiftMatchReferenceWithVarGradients) {
    using stan::math::var;
    constexpr Eigen::Index nPaths = 5;
    StateMatrix<var> states(2, nPaths);
    StateMatrix<var> outBlock(2, nPaths);
    StateMatrix<var> outLift(2, nPaths);
    StateMatrix<var> outRef(2, nPaths);
    for (Eigen::Index p = 0; p < nPaths; ++p) {
        states(0, p) = 0.3 + 0.1 * static_cast<double>(p);
        states(1, p) = -0.4 + 0.05 * static_cast<double>(p);
    }

    evaluateBlock<var>(BlockEvaluator{}, states, outBlock);
    evaluateBlock<var>(perPath<var>(ScalarEvaluator{}), states, outLift);
    for (Eigen::Index p = 0; p < nPaths; ++p) {
        ScalarEvaluator{}(std::span<const var>(states.col(p).data(), 2),
                          std::span<var>(outRef.col(p).data(), 2));
    }

    for (Eigen::Index i = 0; i < 2; ++i) {
        for (Eigen::Index p = 0; p < nPaths; ++p) {
            CHECK_CLOSE("block value", outBlock(i, p).val(), outRef(i, p).val(), 1e-14);
            CHECK_CLOSE("lift value", outLift(i, p).val(), outRef(i, p).val(), 1e-14);
        }
    }

    // Gradients: one single-root pass per branch on a cleared tape, then
    // compare against the analytic map and against each other.
    var blockSum = outBlock.sum();
    var liftSum = outLift.sum();
    var refSum = outRef.sum();
    Eigen::MatrixXd gradBlock(2, nPaths);
    Eigen::MatrixXd gradLift(2, nPaths);
    Eigen::MatrixXd gradRef(2, nPaths);
    const auto collect = [&](var& root, Eigen::MatrixXd& target) {
        stan::math::set_zero_all_adjoints();
        root.grad();
        for (Eigen::Index i = 0; i < 2; ++i) {
            for (Eigen::Index p = 0; p < nPaths; ++p) {
                target(i, p) = states(i, p).adj();
            }
        }
    };
    collect(blockSum, gradBlock);
    collect(liftSum, gradLift);
    collect(refSum, gradRef);
    for (Eigen::Index p = 0; p < nPaths; ++p) {
        const double x0 = states(0, p).val();
        const double x1 = states(1, p).val();
        CHECK_CLOSE("analytic d/dx0", gradBlock(0, p), 2.0 * x0 + x1, 1e-12);
        CHECK_CLOSE("analytic d/dx1", gradBlock(1, p), 0.25 + x0, 1e-12);
        for (Eigen::Index i = 0; i < 2; ++i) {
            CHECK_CLOSE("lift gradient", gradLift(i, p), gradBlock(i, p), 1e-12);
            CHECK_CLOSE("reference gradient", gradRef(i, p), gradBlock(i, p), 1e-12);
        }
    }
}

TEST_F(SdeEvaluateAdTest, blockAndLiftMatchForFvarVar) {
    using stan::math::fvar;
    using stan::math::var;
    using dual = fvar<var>;
    constexpr Eigen::Index nPaths = 4;
    StateMatrix<dual> states(2, nPaths);
    for (Eigen::Index p = 0; p < nPaths; ++p) {
        states(0, p) = dual(var(0.3 + 0.1 * static_cast<double>(p)));
        states(0, p).d_ = 1.0;
        states(1, p) = dual(var(-0.4 + 0.05 * static_cast<double>(p)));
        states(1, p).d_ = 0.5;
    }

    StateMatrix<dual> outBlock(2, nPaths);
    StateMatrix<dual> outLift(2, nPaths);
    StateMatrix<dual> outRef(2, nPaths);
    evaluateBlock<dual>(BlockEvaluator{}, states, outBlock);
    evaluateBlock<dual>(perPath<dual>(ScalarEvaluator{}), states, outLift);
    for (Eigen::Index p = 0; p < nPaths; ++p) {
        ScalarEvaluator{}(std::span<const dual>(states.col(p).data(), 2),
                          std::span<dual>(outRef.col(p).data(), 2));
    }

    for (Eigen::Index i = 0; i < 2; ++i) {
        for (Eigen::Index p = 0; p < nPaths; ++p) {
            CHECK_CLOSE("fvar value", outBlock(i, p).val_.val(), outRef(i, p).val_.val(), 1e-14);
            CHECK_CLOSE("fvar value (lift)", outLift(i, p).val_.val(), outRef(i, p).val_.val(),
                        1e-14);
            CHECK_CLOSE("fvar derivative", outBlock(i, p).d_.val(), outRef(i, p).d_.val(), 1e-14);
            CHECK_CLOSE("fvar derivative (lift)", outLift(i, p).d_.val(), outRef(i, p).d_.val(),
                        1e-14);
        }
    }
}
