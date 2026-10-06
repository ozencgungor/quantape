#pragma once

#include "quantape/instruments/BootstrapInstrument.h"
#include "quantape/markets/Curves/DiscountCurve.h"
#include "quantape/math/Solvers/BrentSolver.h"

#include <cmath>
#include <cstddef>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace quantape::markets {
/**
 * @file BootstrapInstrument.h
 * @brief Curve-set vocabulary (re-export) and the generic sequential node solver
 *
 * The plain-data instrument vocabulary — curve sets, the `BootstrapInstrument`
 * concept and the heterogeneous `Ladder` — lives in
 * `quantape/instruments/BootstrapInstrument.h` and is included here for the
 * existing curve-side consumers. This header keeps the concrete-curve solver
 * that needs `DiscountCurve` plus the Brent root finder, so pricing headers
 * can pull the vocabulary alone.
 */

namespace detail {

/// Sequential exact-fit node solver shared by the FX-swap and cross-currency
/// bootstraps: one Brent solve per node against a trial curve moved in place
/// from the current nodes, then a triangular Gauss-Seidel re-pass when the
/// first pass leaves residual error. `residual(i, curve)` returns the model
/// quote minus its target. Node 0 (t = 0) is pinned to zero. One trial curve
/// is built per node solve, so the Brent steps only move the solved node
/// through `CurveTrialUpdater` instead of rebuilding the grid each time.
template <typename Residual>
std::vector<double>
bootstrapNodesByBrent(InterpolationSpace space, InterpolationScheme scheme, double tension,
                      int switchIndex, double accuracy, const std::vector<double>& nodeTimes,
                      const Residual& residual, std::string_view context = "bootstrap") {
    const std::size_t count = nodeTimes.size();
    const quantape::math::BrentSolver<double> solver;
    std::vector<double> zeros(count, 0.0);
    std::vector<double> trialTimes;
    std::vector<double> trialZeros;
    const auto worstResidual = [&]() {
        trialTimes.assign(1, 0.0);
        trialZeros.assign(1, 0.0);
        if (trialTimes.capacity() < count + 1) {
            trialTimes.reserve(count + 1);
            trialZeros.reserve(count + 1);
        }
        for (std::size_t j = 0; j < count; ++j) {
            trialTimes.push_back(nodeTimes[j]);
            trialZeros.push_back(zeros[j]);
        }
        const DiscountCurve<double> curve(trialTimes, trialZeros, space, scheme, tension,
                                          switchIndex);
        double worst = 0.0;
        for (std::size_t i = 0; i < count; ++i) {
            const double check = residual(i, curve);
            if (!std::isfinite(check)) {
                worst = 1e300;
            } else if (std::abs(check) > worst) {
                worst = std::abs(check);
            }
        }
        return worst;
    };
    const auto solveNodes = [&](bool multiPass) {
        for (int pass = 0; pass < (multiPass ? 50 : 1); ++pass) {
            const std::vector<double> previous = zeros;
            double lastMove = 0.0;
            for (std::size_t i = 0; i < count; ++i) {
                const std::size_t lastNode = multiPass && pass == 0 ? i : count - 1;
                // The trial curve is rebuilt once per node solve and then only
                // the solved node moves; the scheme state stays valid because
                // the node grid is fixed for the whole solve.
                std::optional<DiscountCurve<double>> trialCurve;
                bool cacheValid = false;
                double cachedZero = 0.0;
                double cachedResidual = 0.0;
                const auto objective = [&](double trialZero) {
                    if (cacheValid && trialZero == cachedZero) {
                        return cachedResidual;
                    }
                    if (!trialCurve.has_value()) {
                        trialTimes.assign(1, 0.0);
                        trialZeros.assign(1, 0.0);
                        if (trialTimes.capacity() < lastNode + 2) {
                            trialTimes.reserve(lastNode + 2);
                            trialZeros.reserve(lastNode + 2);
                        }
                        for (std::size_t j = 0; j <= lastNode; ++j) {
                            trialTimes.push_back(nodeTimes[j]);
                            trialZeros.push_back(zeros[j]);
                        }
                        trialCurve.emplace(trialTimes, trialZeros, space, scheme, tension,
                                           switchIndex);
                    }
                    CurveTrialUpdater::setNode(*trialCurve, i + 1, trialZero);
                    cachedResidual = residual(i, *trialCurve);
                    cachedZero = trialZero;
                    cacheValid = true;
                    return cachedResidual;
                };
                const double guess = i == 0 ? 0.0 : zeros[i - 1];
                double lower = guess - 0.5;
                double upper = guess + 0.5;
                double fLower = objective(lower);
                double fUpper = objective(upper);
                int widen = 0;
                while (fLower * fUpper > 0.0 && widen < 12) {
                    lower -= 0.5;
                    upper += 0.5;
                    fLower = objective(lower);
                    fUpper = objective(upper);
                    ++widen;
                }
                if (!(fLower * fUpper <= 0.0) || !std::isfinite(fLower) || !std::isfinite(fUpper)) {
                    throw std::runtime_error(std::string(context) + ": failed to bracket pillar " +
                                             std::to_string(i));
                }
                const double root = solver.solve(objective, accuracy, guess, lower, upper);
                const double move = std::abs(root - previous[i]);
                if (move > lastMove) {
                    lastMove = move;
                }
                zeros[i] = root;
            }
            if (!multiPass || lastMove < 1e-15) {
                break;
            }
        }
    };
    solveNodes(false);
    if (!(worstResidual() < 1e-9)) {
        solveNodes(true);
    }
    if (!(worstResidual() < 1e-9)) {
        throw std::runtime_error(std::string(context) + ": fixed point did not converge");
    }
    return zeros;
}

} // namespace detail

} // namespace quantape::markets
