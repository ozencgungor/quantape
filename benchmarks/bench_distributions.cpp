// bench_distributions — scalar kernel timings for the distributions module.
//
// Reports median ns/evaluation over a fixed input grid. Build in the release
// tree (benchmarks are ON there) and compare across revisions with interleaved
// A/B binaries per the repo noise-floor policy (~+/-4-5%).
#include "quantape/math/Distributions.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <random>
#include <string>
#include <vector>

namespace {

volatile double g_sink = 0.0;

using namespace quantape::math;

/// Median ns/eval of `f` over `xs`, `reps` repetitions (grid reused).
template <class F>
double medianNs(const std::vector<double>& xs, int reps, F&& f) {
    std::vector<double> samples;
    samples.reserve(static_cast<std::size_t>(reps));
    for (int r = 0; r < reps; ++r) {
        const auto start = std::chrono::steady_clock::now();
        double sink = 0.0;
        for (double x : xs) {
            sink += f(x);
        }
        const auto stop = std::chrono::steady_clock::now();
        g_sink = sink;
        samples.push_back(std::chrono::duration<double, std::nano>(stop - start).count() /
                          static_cast<double>(xs.size()));
    }
    std::sort(samples.begin(), samples.end());
    return samples[samples.size() / 2];
}

void report(const std::string& name, double ns) {
    std::printf("%-34s %10.1f ns/eval\n", name.c_str(), ns);
}

} // namespace

int main() {
    std::mt19937_64 rng(42);
    std::normal_distribution<double> z(0.0, 1.0);
    std::vector<double> xs(4096);
    for (double& v : xs) {
        v = 0.4 * z(rng);
    }
    std::vector<double> uniforms(4096);
    uniforms[0] = 1e-6;
    uniforms[1] = 1e-3;
    for (std::size_t i = 2; i < uniforms.size(); ++i) {
        uniforms[i] = 0.5 + 0.4 * z(rng) / 4.0;
    }
    std::vector<double> positives(4096);
    for (double& v : positives) {
        v = std::abs(v) + 0.05;
    }

    const int reps = 7;
    const Normal<double> normal{0.3, 1.4};

    report("normal pdf", medianNs(xs, reps, [&](double x) { return normal.pdf(x); }));
    report("normal cdf", medianNs(xs, reps, [&](double x) { return normal.cdf(x); }));
    report("normal logCdf", medianNs(xs, reps, [&](double x) { return normal.logCdf(x); }));
    report("normal inv", medianNs(uniforms, reps, [&](double p) { return normal.inv(p); }));

    const BivariateNormal<double> bvnLow{-0.5};
    const BivariateNormal<double> bvnMid{0.3};
    const BivariateNormal<double> bvnHigh{0.95};
    // Distinct consecutive grid values so the first argument spans the tail.
    std::vector<double> ab(xs.begin(), xs.end() - 1);
    const auto bvnRun = [&](const BivariateNormal<double>& d) {
        std::vector<double> samples;
        samples.reserve(static_cast<std::size_t>(reps));
        for (int r = 0; r < reps; ++r) {
            const auto start = std::chrono::steady_clock::now();
            double sink = 0.0;
            for (std::size_t i = 0; i + 1 < ab.size(); ++i) {
                sink += d.cdf(ab[i], ab[i + 1]);
            }
            const auto stop = std::chrono::steady_clock::now();
            g_sink = sink;
            samples.push_back(std::chrono::duration<double, std::nano>(stop - start).count() /
                              static_cast<double>(ab.size() - 1));
        }
        std::sort(samples.begin(), samples.end());
        return samples[samples.size() / 2];
    };
    report("bvn cdf (rho=-0.5)", bvnRun(bvnLow));
    report("bvn cdf (rho=+0.3)", bvnRun(bvnMid));
    report("bvn cdf (rho=+0.95)", bvnRun(bvnHigh));

    const Gamma<double> gamma{2.3, 1.7};
    report("gamma pdf", medianNs(positives, reps, [&](double x) { return gamma.pdf(x); }));
    report("gamma cdf", medianNs(positives, reps, [&](double x) { return gamma.cdf(x); }));

    const StudentT<double> studentT{4.0};
    report("student-t cdf", medianNs(xs, reps, [&](double x) { return studentT.cdf(x); }));

    const Beta<double> beta{2.0, 3.5};
    report("beta cdf", medianNs(uniforms, reps, [&](double p) { return beta.cdf(p); }));

    const NoncentralChiSquare<double> nccs{3.0, 4.0};
    report("noncentral chi2 cdf", medianNs(positives, reps, [&](double x) { return nccs.cdf(x); }));
    report("noncentral chi2 cdfGradHess",
           medianNs(positives, reps, [&](double x) { return nccs.cdfGradHess(x).value; }));
    report("noncentral chi2 pdfGradHess",
           medianNs(positives, reps, [&](double x) { return nccs.pdfGradHess(x).value; }));

    const Poisson<double> poisson{3.2};
    report("poisson cdf", medianNs(positives, reps, [&](double x) {
               return poisson.cdf(static_cast<int>(std::floor(4.0 * x)));
           }));

    const Lognormal<double> lognormal{0.3, 0.7};
    report("lognormal cdf", medianNs(positives, reps, [&](double x) { return lognormal.cdf(x); }));

    const InverseGaussian<double> invGauss{1.0, 2.0};
    report("inverse Gaussian cdf",
           medianNs(positives, reps, [&](double x) { return invGauss.cdf(x); }));

    return g_sink == 0.0 ? 1 : 0;
}
