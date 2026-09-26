// bench_sde.cpp — timing harness for the SDE simulation engine
//
// Measures per-draw / per-path-step costs (streamed, so large path tensors
// are not allocated) plus parallel-schedule scaling; the sampling-profiler
// target for optimization work:
//   ./build/bench_sde [reps]
//   /usr/bin/sample bench_sde 5 -file /tmp/prof.txt
#include "quantape/mc/Estimator.h"
#include "quantape/mc/MomentMatching.h"
#include "quantape/mc/Parallel.h"
#include "quantape/mc/RandomSource.h"
#include "quantape/mc/Schemes.h"
#include "quantape/mc/SdePrimitives.h"
#include "quantape/mc/SdeSimulator.h"
#include "quantape/mc/TimeGrid.h"
#include "quantape/processes/HestonQeProcess.h"
#include "quantape/processes/SdeProcesses.h"

#include <Eigen/Dense>

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <functional>
#include <type_traits>
#include <vector>

using quantape::mc::IidGaussianSource;
using quantape::mc::Schedule;
using quantape::mc::SdeSimulator;
using quantape::mc::TimeGrid;
using quantape::processes::GbmProcess;
using quantape::processes::HestonProcess;
using quantape::processes::HestonQeProcess;

namespace {

double bench_us(const char* name, double per_unit, int reps, const std::function<double()>& fn) {
    fn(); // warmup
    const auto t0 = std::chrono::steady_clock::now();
    double sink = 0.0;
    for (int i = 0; i < reps; ++i) {
        sink += fn();
    }
    const auto t1 = std::chrono::steady_clock::now();
    const double us = std::chrono::duration<double, std::micro>(t1 - t0).count() / reps;
    std::printf("%-42s %10.1f us/run   %7.2f ns/unit   (sink %.3e)\n", name, us,
                per_unit > 0.0 ? us * 1000.0 / per_unit : 0.0, sink);
    std::fflush(stdout);
    return us;
}

struct LinearModel {
    Eigen::MatrixXd driftMatrix;                    // out = driftMatrix * x
    std::vector<Eigen::MatrixXd> diffusionMatrices; // q matrices

    template <typename Scalar>
    void drift(const quantape::mc::StateMatrix<Scalar>& x, double, const std::vector<Scalar>&,
               quantape::mc::StateMatrix<Scalar>& out) const {
        if constexpr (std::is_same_v<Scalar, double>) {
            out.noalias() = driftMatrix * x;
        } else {
            out.noalias() = driftMatrix.template cast<Scalar>() * x;
        }
    }
    template <typename Scalar>
    void diffusion(const quantape::mc::StateMatrix<Scalar>& x, double, const std::vector<Scalar>&,
                   std::size_t factor, quantape::mc::StateMatrix<Scalar>& out) const {
        if constexpr (std::is_same_v<Scalar, double>) {
            out.noalias() = diffusionMatrices[factor] * x;
        } else {
            out.noalias() = diffusionMatrices[factor].template cast<Scalar>() * x;
        }
    }
};

} // namespace

int main(int argc, char** argv) {
    const int reps = argc > 1 ? std::atoi(argv[1]) : 20;
    std::printf("SDE engine benchmark, reps=%d\n", reps);

    // ── 1. Source: keyed draws ──
    {
        const std::size_t nPaths = 65536;
        IidGaussianSource<> source(2, 1234);
        Eigen::MatrixXd z;
        bench_us("source fill q=2, 65536 paths", static_cast<double>(nPaths * 2), reps, [&] {
            source.fill(0, 0, nPaths, z);
            return z(0, 0) + z(1, static_cast<Eigen::Index>(nPaths - 1));
        });
        bench_us("source fillUniform 65536 paths", static_cast<double>(nPaths), reps, [&] {
            source.fillUniform(0, 0, nPaths, 0, 1, z);
            return z(0, 0) + z(0, static_cast<Eigen::Index>(nPaths - 1));
        });
    }

    // ── 2. Euler GBM streamed (true engine cost) + block sizes ──
    {
        const GbmProcess model{0.05, 0.2};
        const std::size_t nPaths = 100000;
        const std::size_t nSteps = 252;
        const TimeGrid grid(1.0, nSteps);
        std::vector<std::vector<double>> theta(nSteps);
        const auto x0 = Eigen::VectorXd::Constant(1, 100.0);
        for (std::size_t blockSize : {std::size_t(1024), std::size_t(4096), std::size_t(16384)}) {
            const SdeSimulator<double> simulator(grid, theta);
            const IidGaussianSource<> source(1, 7);
            char name[96];
            std::snprintf(name, sizeof(name), "euler gbm stream 100k x 252 (block %zu)",
                          static_cast<std::size_t>(blockSize));
            bench_us(name, static_cast<double>(nPaths * nSteps), reps, [&] {
                double checksum = 0.0;
                simulator.simulateBlocks(
                    x0, quantape::mc::driftOf(model), quantape::mc::diffusionOf(model), source,
                    nPaths, blockSize,
                    [&checksum](const quantape::mc::PathBlock<double>& block, std::size_t) {
                        checksum += block.states.back()(0, 0);
                    });
                return checksum;
            });
        }
        {
            const SdeSimulator<double> simulator(grid, theta);
            const IidGaussianSource<> source(1, 7);
            const std::size_t blockSize = 1024;
            const std::size_t nBlocks = (nPaths + blockSize - 1) / blockSize;
            std::vector<double> partials(nBlocks, 0.0);
            bench_us("euler gbm stream 100k x 252 (parallel)", static_cast<double>(nPaths * nSteps),
                     reps, [&] {
                         simulator.simulateBlocks(
                             x0, quantape::mc::driftOf(model), quantape::mc::diffusionOf(model),
                             source, nPaths, blockSize,
                             [&partials](const quantape::mc::PathBlock<double>& block,
                                         std::size_t blockIndex) {
                                 partials[blockIndex] = block.states.back()(0, 0);
                             },
                             Schedule::Parallel);
                         double checksum = 0.0;
                         for (double v : partials) {
                             checksum += v;
                         }
                         return checksum;
                     });
            bench_us("heston qe stream 50k x 252 (parallel)", static_cast<double>(nPaths * nSteps),
                     reps, [&] {
                         const std::size_t blockSize = 4096;
                         const std::size_t nBlocks = (nPaths + blockSize - 1) / blockSize;
                         std::vector<double> partials(nBlocks, 0.0);
                         simulator.simulateBlocks(
                             x0, quantape::mc::driftOf(model), quantape::mc::diffusionOf(model),
                             source, nPaths, blockSize,
                             [&partials](const quantape::mc::PathBlock<double>& block,
                                         std::size_t blockIndex) {
                                 partials[blockIndex] = block.states.back()(1, 0);
                             },
                             Schedule::Parallel);
                         double checksum = 0.0;
                         for (double v : partials) {
                             checksum += v;
                         }
                         return checksum;
                     });
        }
    }

    // ── 3. Parallel schedule scaling (collecting, small enough to fit) ──
    {
        const GbmProcess model{0.05, 0.2};
        const std::size_t nPaths = 20000;
        const std::size_t nSteps = 252;
        const TimeGrid grid(1.0, nSteps);
        std::vector<std::vector<double>> theta(nSteps);
        const auto x0 = Eigen::VectorXd::Constant(1, 100.0);
        const SdeSimulator<double> simulator(grid, theta);
        for (Schedule schedule : {Schedule::Sequential, Schedule::Parallel}) {
            const IidGaussianSource<> source(1, 7);
            bench_us(schedule == Schedule::Parallel ? "euler gbm 20k x 252 (parallel)"
                                                    : "euler gbm 20k x 252 (sequential)",
                     static_cast<double>(nPaths * nSteps), reps, [&] {
                         const auto blocks = simulator.simulate(x0, quantape::mc::driftOf(model),
                                                                quantape::mc::diffusionOf(model),
                                                                source, nPaths, 4096, schedule);
                         return blocks.front().states.back()(0, 0);
                     });
        }
    }

    // ── 4. Heston QE (2 factors + uniform), streamed + parallel ──
    {
        const HestonProcess model{0.02, 2.0, 0.04, 0.3, -0.7};
        const auto x0 = (Eigen::Vector2d() << std::log(100.0), 0.04).finished();
        const std::size_t nSteps = 252;
        const TimeGrid grid(1.0, nSteps);
        std::vector<std::vector<double>> theta(nSteps);
        {
            const std::size_t nPaths = 50000;
            const SdeSimulator<double, HestonQeProcess<HestonProcess>> simulator(
                grid, theta, HestonQeProcess<HestonProcess>{model});
            const IidGaussianSource<> source(2, 9);
            bench_us("heston qe stream 50k x 252", static_cast<double>(nPaths * nSteps), reps, [&] {
                double checksum = 0.0;
                simulator.simulateBlocks(
                    x0, quantape::mc::driftOf(model), quantape::mc::diffusionOf(model), source,
                    nPaths, 4096,
                    [&checksum](const quantape::mc::PathBlock<double>& block, std::size_t) {
                        checksum += block.states.back()(1, 0);
                    });
                return checksum;
            });
        }
        {
            const std::size_t nPaths = 10000;
            const SdeSimulator<double, HestonQeProcess<HestonProcess>> simulator(
                grid, theta, HestonQeProcess<HestonProcess>{model});
            const IidGaussianSource<> source(2, 9);
            bench_us(
                "heston qe 10k x 252 (parallel)", static_cast<double>(nPaths * nSteps), reps, [&] {
                    const auto blocks = simulator.simulate(x0, quantape::mc::driftOf(model),
                                                           quantape::mc::diffusionOf(model), source,
                                                           nPaths, 2048, Schedule::Parallel);
                    return blocks.front().states.back()(1, 0);
                });
        }
    }

    // ── 5. Estimator (sequential vs parallel over blocks) ──
    {
        const GbmProcess model{0.0, 0.2};
        const std::size_t nPaths = 200000;
        const std::size_t nSteps = 64;
        const TimeGrid grid(1.0, nSteps);
        std::vector<std::vector<double>> theta(nSteps);
        const SdeSimulator<double> simulator(grid, theta);
        const IidGaussianSource<> source(1, 11);
        const auto blocks =
            simulator.simulate(Eigen::VectorXd::Constant(1, 100.0), quantape::mc::driftOf(model),
                               quantape::mc::diffusionOf(model), source, nPaths, 8192);
        const auto payoff = [](const quantape::mc::PathBlock<double>& b, Eigen::VectorXd& out) {
            const auto& x = b.states.back();
            out = (x.row(0).array() - 100.0).max(0.0).matrix().transpose();
        };
        bench_us("estimator 200k (sequential)", static_cast<double>(nPaths), reps, [&] {
            const auto e = quantape::mc::estimate(blocks, payoff, Schedule::Sequential);
            return e.mean + e.stdError;
        });
        bench_us("estimator 200k (parallel)", static_cast<double>(nPaths), reps, [&] {
            const auto e = quantape::mc::estimate(blocks, payoff, Schedule::Parallel);
            return e.mean + e.stdError;
        });
    }

    // ── 6. Multi-dimensional linear model ──
    {
        const std::size_t dims = 64;
        const std::size_t factors = 8;
        const std::size_t nPaths = 2048;
        const std::size_t nSteps = 32;
        LinearModel model;
        model.driftMatrix = Eigen::MatrixXd::Random(static_cast<Eigen::Index>(dims),
                                                    static_cast<Eigen::Index>(dims)) *
                            0.1;
        model.diffusionMatrices.resize(factors);
        for (auto& m : model.diffusionMatrices) {
            m = Eigen::MatrixXd::Random(static_cast<Eigen::Index>(dims),
                                        static_cast<Eigen::Index>(dims)) *
                0.1;
        }
        const TimeGrid grid(1.0, nSteps);
        std::vector<std::vector<double>> theta(nSteps);
        const SdeSimulator<double> simulator(grid, theta);
        const IidGaussianSource<> source(factors, 13);
        const auto x0 = Eigen::VectorXd::Constant(static_cast<Eigen::Index>(dims), 1.0);
        bench_us("linear model d=64 q=8 stream 2048 x 32", static_cast<double>(nPaths * nSteps),
                 reps, [&] {
                     double checksum = 0.0;
                     simulator.simulateBlocks(
                         x0, quantape::mc::driftOf(model), quantape::mc::diffusionOf(model), source,
                         nPaths, 512,
                         [&checksum](const quantape::mc::PathBlock<double>& block, std::size_t) {
                             checksum += block.states.back()(0, 0);
                         });
                     return checksum;
                 });
    }

    return 0;
}
