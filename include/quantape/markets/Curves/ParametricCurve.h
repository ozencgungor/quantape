#pragma once

#include "quantape/math/Optimization/LevenbergMarquardt.h"

#include <array>
#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <string_view>
#include <vector>

namespace quantape::markets {
/**
 * @file ParametricCurve.h
 * @brief Best-fit parametric zero curves (Nelson-Siegel, Svensson)
 *
 * These are *best-fit* curves, not exact-fit bootstrap curves: they are
 * fitted by weighted least squares through the LM solver and are therefore
 * flagged as non-exact-fit in the stack (they cannot appear as intermediate
 * parents in a bootstrap chain; risk flows through the calibration IFT
 * layer).
 *
 * Zero rate convention: continuously compounded. Parameter layout
 * `{b0, b1, b2, tau1, b3, tau2}`; `b3`/`tau2` are unused for Nelson-Siegel.
 */

enum class ParametricForm { NelsonSiegel, Svensson };

constexpr std::string_view parametricFormName(ParametricForm form) noexcept {
    switch (form) {
        case ParametricForm::NelsonSiegel: return "NelsonSiegel";
        case ParametricForm::Svensson: return "Svensson";
    }
    return "Unknown";
}

struct ParametricCurve {
    ParametricForm form = ParametricForm::NelsonSiegel;
    std::array<double, 6> parameters{}; ///< b0, b1, b2, tau1, b3, tau2

    /// Continuously compounded zero rate at `t` (t > 0).
    double zero(double t) const {
        if (!(t > 0.0)) {
            throw std::invalid_argument("ParametricCurve::zero: t must be positive");
        }
        const double b0 = parameters[0];
        const double b1 = parameters[1];
        const double b2 = parameters[2];
        const double tau1 = parameters[3];
        const double x1 = t / tau1;
        const double decay1 = (1.0 - std::exp(-x1)) / x1;
        double value = b0 + b1 * decay1 + b2 * (decay1 - std::exp(-x1));
        if (form == ParametricForm::Svensson) {
            const double b3 = parameters[4];
            const double tau2 = parameters[5];
            const double x2 = t / tau2;
            const double decay2 = (1.0 - std::exp(-x2)) / x2;
            value += b3 * (decay2 - std::exp(-x2));
        }
        return value;
    }

    double discount(double t) const {
        if (t <= 0.0) {
            return 1.0;
        }
        return std::exp(-zero(t) * t);
    }

    /// Weighted least-squares fit of `zeros` at `times` (LM, taus in log space).
    static ParametricCurve fit(ParametricForm form, const std::vector<double>& times,
                               const std::vector<double>& zeros,
                               const std::vector<double>& weights = {}) {
        if (times.size() != zeros.size() || times.size() < 2) {
            throw std::invalid_argument("ParametricCurve::fit: size mismatch or too few points");
        }
        const std::size_t n = times.size();
        std::vector<double> w(n, 1.0);
        if (!weights.empty()) {
            if (weights.size() != n) {
                throw std::invalid_argument("ParametricCurve::fit: weights size mismatch");
            }
            w = weights;
        }
        for (double& value : w) {
            value = std::sqrt(value);
        }
        // Init: long rate from the last point, level from the first, curvature
        // from the sample nearest the initial tau (avoids the standard NS
        // local minimum when starting away from the data).
        const double longRate = zeros.back();
        const double shortRate = zeros.front();
        const double tauInit = 2.0;
        const double b0Init = longRate;
        const double b1Init = shortRate - longRate;
        double b2Init = 0.0;
        std::size_t mid = 0;
        double bestDistance = std::abs(times.front() - tauInit);
        for (std::size_t i = 1; i < n; ++i) {
            const double distance = std::abs(times[i] - tauInit);
            if (distance < bestDistance) {
                bestDistance = distance;
                mid = i;
            }
        }
        {
            const double scaled = times[mid] / tauInit;
            const double decay = (1.0 - std::exp(-scaled)) / scaled;
            const double denominator = decay - std::exp(-scaled);
            if (std::abs(denominator) > 1e-12) {
                b2Init = (zeros[mid] - b0Init - b1Init * decay) / denominator;
            }
        }
        std::vector<double> x{b0Init, b1Init, b2Init, std::log(tauInit), 0.0,
                              std::log(5.0)};
        const auto residual = [&](const std::vector<double>& point, std::vector<double>& out) {
            ParametricCurve trial;
            trial.form = form;
            trial.parameters = {point[0], point[1], point[2], std::exp(point[3]), point[4],
                                std::exp(point[5])};
            out.clear();
            out.reserve(n);
            for (std::size_t i = 0; i < n; ++i) {
                out.push_back(w[i] * (trial.zero(times[i]) - zeros[i]));
            }
        };
        math::LevenbergMarquardtOptions options;
        options.maxIterations = 500;
        const math::LevenbergMarquardtResult result = math::levenbergMarquardt(residual, x, options);
        if (result.status == math::OptimizeResult::Failure) {
            throw std::runtime_error("ParametricCurve::fit: LM failed");
        }
        ParametricCurve curve;
        curve.form = form;
        curve.parameters = {x[0], x[1], x[2], std::exp(x[3]), x[4], std::exp(x[5])};
        return curve;
    }
};

} // namespace quantape::markets
