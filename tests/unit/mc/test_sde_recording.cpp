// test_sde_recording.cpp — recordTimes storage/sampling contract
//
// Gates:
//   - full explicit index set == default full retention, bitwise
//   - arbitrary index set stores exactly those states (bitwise subset)
//   - empty index set stores nothing while the full grid is still stepped
//   - recorded path == recorded block path, bitwise; parallel == sequential
//   - invalid index sets are rejected
//
// Stan-free: includes only mc/ headers + Eigen.
#include "quantape/mc/Estimator.h"
#include "quantape/mc/RandomSource.h"
#include "quantape/mc/Schemes.h"
#include "quantape/mc/SdeSimulator.h"
#include "quantape/mc/TimeGrid.h"

#include <Eigen/Dense>

#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <vector>

#include "support/GtestSupport.h"

using quantape::mc::diffusionOf;
using quantape::mc::driftOf;
using quantape::mc::Estimate;
using quantape::mc::IidGaussianSource;
using quantape::mc::PathBlock;
using quantape::mc::Schedule;
using quantape::mc::SdeSimulator;
using quantape::mc::StateMatrix;
using quantape::mc::TimeGrid;

namespace {

constexpr std::size_t kSteps = 8;

// dX = kappa (level - X) dt + sigma dW ; theta = {kappa, level, sigma}
struct OuModel {
    template <typename Scalar>
    void drift(const StateMatrix<Scalar>& x, double, const std::vector<Scalar>& th,
               StateMatrix<Scalar>& out) const {
        out = (th[0] * (th[1] - x.array())).matrix();
    }

    template <typename Scalar>
    void diffusion(const StateMatrix<Scalar>& x, double, const std::vector<Scalar>& th, std::size_t,
                   StateMatrix<Scalar>& out) const {
        out.resize(x.rows(), x.cols());
        out.setConstant(th[2]);
    }
};

struct TerminalCall {
    double strike = 1.0;
    void operator()(const PathBlock<double>& block, Eigen::VectorXd& out) const {
        const auto& x = block.states.back();
        out.resize(static_cast<Eigen::Index>(block.nPaths));
        for (Eigen::Index p = 0; p < x.cols(); ++p) {
            out(p) = std::max(x(0, p) - strike, 0.0);
        }
    }
};

bool sameStates(const std::vector<StateMatrix<double>>& a,
                const std::vector<StateMatrix<double>>& b) {
    if (a.size() != b.size()) {
        return false;
    }
    for (std::size_t k = 0; k < a.size(); ++k) {
        if (a[k].rows() != b[k].rows() || a[k].cols() != b[k].cols() ||
            !(a[k].array() == b[k].array()).all()) {
            return false;
        }
    }
    return true;
}

std::vector<std::vector<double>> thetaSteps(std::size_t nSteps) {
    return std::vector<std::vector<double>>(nSteps, {1.5, 0.8, 0.4});
}

} // namespace

TEST(SdeRecording, fullExplicitSetMatchesDefaultRetention) {
    const TimeGrid grid(1.0, kSteps);
    const SdeSimulator<double> simulator(grid, thetaSteps(kSteps));
    const IidGaussianSource<> source(1, 4242);
    const auto x0 = Eigen::VectorXd::Constant(1, 2.0);

    const auto full = simulator.simulate(x0, driftOf(OuModel{}), diffusionOf(OuModel{}), source, 64,
                                         16, Schedule::Sequential);
    std::vector<std::size_t> allIndices;
    for (std::size_t k = 0; k <= kSteps; ++k) {
        allIndices.push_back(k);
    }
    const auto explicitAll =
        simulator.simulateRecorded(x0, driftOf(OuModel{}), diffusionOf(OuModel{}), source, 64, 16,
                                   allIndices, Schedule::Sequential);

    ASSERT_EQ(full.size(), explicitAll.size());
    for (std::size_t bi = 0; bi < full.size(); ++bi) {
        EXPECT_TRUE(sameStates(full[bi].states, explicitAll[bi].states));
        EXPECT_TRUE(explicitAll[bi].storesAll());
        EXPECT_EQ(explicitAll[bi].nRecorded(), kSteps + 1);
        EXPECT_EQ(explicitAll[bi].nSteps(), kSteps);
    }
}

TEST(SdeRecording, arbitrarySetStoresExactSubsetBitwise) {
    const TimeGrid grid(1.0, kSteps);
    const SdeSimulator<double> simulator(grid, thetaSteps(kSteps));
    const IidGaussianSource<> source(1, 4242);
    const auto x0 = Eigen::VectorXd::Constant(1, 2.0);

    const auto full = simulator.simulate(x0, driftOf(OuModel{}), diffusionOf(OuModel{}), source, 64,
                                         16, Schedule::Sequential);
    const std::vector<std::size_t> exposure = {0, 3, 7, 8};
    const auto recorded =
        simulator.simulateRecorded(x0, driftOf(OuModel{}), diffusionOf(OuModel{}), source, 64, 16,
                                   exposure, Schedule::Sequential);

    ASSERT_EQ(recorded.size(), full.size());
    for (std::size_t bi = 0; bi < full.size(); ++bi) {
        const PathBlock<double>& block = recorded[bi];
        EXPECT_FALSE(block.storesAll());
        EXPECT_FALSE(block.storesNone());
        EXPECT_EQ(block.nRecorded(), exposure.size());
        EXPECT_EQ(block.nSteps(), kSteps);
        EXPECT_EQ(block.nDims, 1u);
        EXPECT_EQ(block.nPaths, 16u);
        ASSERT_EQ(block.states.size(), exposure.size());
        for (std::size_t r = 0; r < exposure.size(); ++r) {
            EXPECT_EQ(block.recordGridIndex(r), exposure[r]);
            EXPECT_TRUE((block.states[r].array() == full[bi].states[exposure[r]].array()).all());
        }
    }
}

TEST(SdeRecording, emptySetStoresNothingButStepsTheGrid) {
    const TimeGrid grid(1.0, kSteps);
    const SdeSimulator<double> simulator(grid, thetaSteps(kSteps));
    const IidGaussianSource<> source(1, 4242);
    const auto x0 = Eigen::VectorXd::Constant(1, 2.0);

    std::size_t blocksSeen = 0;
    std::size_t pathsSeen = 0;
    simulator.simulateBlocksRecorded(
        x0, driftOf(OuModel{}), diffusionOf(OuModel{}), source, 50, 16, {},
        [&](const PathBlock<double>& block, std::size_t) {
            ++blocksSeen;
            pathsSeen += block.nPaths;
            EXPECT_TRUE(block.storesNone());
            EXPECT_TRUE(block.states.empty());
            EXPECT_EQ(block.nRecorded(), 0u);
            EXPECT_EQ(block.nSteps(), kSteps);
        },
        Schedule::Sequential);
    EXPECT_EQ(blocksSeen, 4u);
    EXPECT_EQ(pathsSeen, 50u);
}

TEST(SdeRecording, recordedPathMatchesRecordedBlock) {
    const TimeGrid grid(1.0, kSteps);
    const SdeSimulator<double> simulator(grid, thetaSteps(kSteps));
    const IidGaussianSource<> source(1, 4242);
    const auto x0 = Eigen::VectorXd::Constant(1, 2.0);
    const std::vector<std::size_t> exposure = {0, 2, 6, 8};

    const auto blocks = simulator.simulateRecorded(x0, driftOf(OuModel{}), diffusionOf(OuModel{}),
                                                   source, 32, 8, exposure);
    for (std::size_t p = 0; p < 32; ++p) {
        const auto path = simulator.simulatePathRecorded(
            x0, driftOf(OuModel{}), diffusionOf(OuModel{}), source, p, exposure);
        ASSERT_EQ(path.nRecorded(), exposure.size());
        const std::size_t bi = p / 8;
        const std::size_t pi = p % 8;
        for (std::size_t r = 0; r < exposure.size(); ++r) {
            EXPECT_EQ(path.states[r](0, 0), blocks[bi].states[r](0, pi));
        }
    }
}

TEST(SdeRecording, parallelScheduleIsBitwise) {
    const TimeGrid grid(1.0, kSteps);
    const SdeSimulator<double> simulator(grid, thetaSteps(kSteps));
    const auto x0 = Eigen::VectorXd::Constant(1, 2.0);
    const std::vector<std::size_t> exposure = {0, 2, 5, 8};

    const IidGaussianSource<> seqSource(1, 99);
    const auto seq =
        simulator.simulateRecorded(x0, driftOf(OuModel{}), diffusionOf(OuModel{}), seqSource, 2000,
                                   256, exposure, Schedule::Sequential);
    const IidGaussianSource<> parSource(1, 99);
    const auto par = simulator.simulateRecorded(x0, driftOf(OuModel{}), diffusionOf(OuModel{}),
                                                parSource, 2000, 256, exposure, Schedule::Parallel);
    ASSERT_EQ(seq.size(), par.size());
    for (std::size_t bi = 0; bi < seq.size(); ++bi) {
        EXPECT_TRUE(sameStates(seq[bi].states, par[bi].states));
    }
}

TEST(SdeRecording, invalidIndexSetsAreRejected) {
    const TimeGrid grid(1.0, kSteps);
    const SdeSimulator<double> simulator(grid, thetaSteps(kSteps));
    const IidGaussianSource<> source(1, 7);
    const auto x0 = Eigen::VectorXd::Constant(1, 2.0);

    EXPECT_THROW(simulator.simulateRecorded(x0, driftOf(OuModel{}), diffusionOf(OuModel{}), source,
                                            8, 4, {kSteps + 1}),
                 std::invalid_argument);
    EXPECT_THROW(simulator.simulateRecorded(x0, driftOf(OuModel{}), diffusionOf(OuModel{}), source,
                                            8, 4, {3, 2}),
                 std::invalid_argument);
    EXPECT_THROW(simulator.simulateRecorded(x0, driftOf(OuModel{}), diffusionOf(OuModel{}), source,
                                            8, 4, {2, 2}),
                 std::invalid_argument);

    PathBlock<double> block;
    EXPECT_THROW(block.resizeRecorded(1, kSteps, 1, {9}), std::invalid_argument);
    block.resizeRecorded(1, kSteps, 4, {1, 5});
    EXPECT_EQ(block.nSteps(), kSteps);
    EXPECT_EQ(block.nRecorded(), 2u);
    EXPECT_EQ(block.recordGridIndex(1), 5u);
}

TEST(SdeRecording, estimatorUsesTerminalRecord) {
    const TimeGrid grid(1.0, kSteps);
    const SdeSimulator<double> simulator(grid, thetaSteps(kSteps));
    const IidGaussianSource<> source(1, 123);
    const auto x0 = Eigen::VectorXd::Constant(1, 2.0);

    const auto full = simulator.simulate(x0, driftOf(OuModel{}), diffusionOf(OuModel{}), source,
                                         4096, 512, Schedule::Sequential);
    const auto recorded =
        simulator.simulateRecorded(x0, driftOf(OuModel{}), diffusionOf(OuModel{}), source, 4096,
                                   512, {0, kSteps}, Schedule::Sequential);

    const Estimate fullEstimate = quantape::mc::estimate(full, TerminalCall{2.0});
    const Estimate recordedEstimate = quantape::mc::estimate(recorded, TerminalCall{2.0});
    EXPECT_EQ(fullEstimate.nPaths, recordedEstimate.nPaths);
    CHECK_CLOSE("recorded estimator mean", recordedEstimate.mean, fullEstimate.mean, 0.0);
    CHECK_CLOSE("recorded estimator variance", recordedEstimate.variance, fullEstimate.variance,
                0.0);
    CHECK_CLOSE("recorded estimator SE", recordedEstimate.stdError, fullEstimate.stdError, 0.0);
}
