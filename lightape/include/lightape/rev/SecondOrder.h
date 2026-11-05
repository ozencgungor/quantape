#ifndef LIGHTAPE_REV_SECOND_ORDER_H
#define LIGHTAPE_REV_SECOND_ORDER_H

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <vector>

#include "lightape/rev/LeanReverse.h"

namespace lightape {
/**
 * @file SecondOrder.h
 * @brief Convenience surface over the lean reverse tape's second order.
 *
 * `gradient` is the classic reverse sweep + leaf readout; `hessian` fills a
 * dense matrix column by column by replaying the recorded tape once per
 * basis direction (via `RevTape::hessianVector`), so it stays exact and
 * needs no second tape.
 */

/// Gradient of `root` w.r.t. the `nInputs` tape slots `inputs`, written into
/// `out[0..nInputs)`. Zeroes every adjoint, seeds `root` with 1 and runs one
/// reverse sweep; leaf (`Input`) adjoints survive the sweep for readout.
inline void gradient(RevTape& tape, std::uint32_t root, const std::uint32_t* inputs,
                     std::size_t nInputs, double* out) {
    tape.resetAdjoints();
    tape.reverse(root, 1.0);
    for (std::size_t k = 0; k < nInputs; ++k) {
        out[k] = tape.adjoint(inputs[k]);
    }
}

/// Dense Hessian, row-major: `outRowMajor[i * nInputs + k] = (H e_k)_i`.
/// Replays the tape once per unit direction, so nInputs sweeps total. The
/// direction and HVP buffers live on the stack for `nInputs <= 64`; larger
/// problems fall back to heap vectors. Returns false on the first column
/// whose `hessianVector` fails (e.g. an External block without svp/vhp);
/// `outRowMajor` then holds unspecified/partial values.
inline bool hessian(RevTape& tape, std::uint32_t root, const std::uint32_t* inputs,
                    std::size_t nInputs, double* outRowMajor) {
    constexpr std::size_t kStackCap = 64;
    double stackDir[kStackCap];
    double stackHvp[kStackCap];
    std::vector<double> heapDir;
    std::vector<double> heapHvp;
    double* dir = stackDir;
    double* hvp = stackHvp;
    if (nInputs > kStackCap) {
        heapDir.resize(nInputs);
        heapHvp.resize(nInputs);
        dir = heapDir.data();
        hvp = heapHvp.data();
    }
    for (std::size_t k = 0; k < nInputs; ++k) {
        for (std::size_t i = 0; i < nInputs; ++i) {
            dir[i] = (i == k) ? 1.0 : 0.0;
        }
        if (!tape.hessianVector(root, inputs, nInputs, dir, hvp)) {
            return false;
        }
        for (std::size_t i = 0; i < nInputs; ++i) {
            outRowMajor[i * nInputs + k] = hvp[i];
        }
    }
    return true;
}

/// Member-forwarding convenience: same contract as `RevTape::hessianVector`.
inline bool hessianVector(RevTape& tape, std::uint32_t root, const std::uint32_t* inputs,
                          std::size_t nInputs, const double* direction, double* out) {
    return tape.hessianVector(root, inputs, nInputs, direction, out);
}

/// Runs `hessian` and checks symmetry entry-wise:
/// `|H_ij - H_ji| <= tol * max(1, |H_ij|)`. Returns the check result
/// (false also when the Hessian computation itself fails). Does not
/// symmetrize `out`: asymmetric entries are reported, never averaged.
inline bool hessianSymmetric(RevTape& tape, std::uint32_t root, const std::uint32_t* inputs,
                             std::size_t n, double* out, double tol = 1e-12) {
    if (!hessian(tape, root, inputs, n, out)) {
        return false;
    }
    for (std::size_t i = 0; i < n; ++i) {
        for (std::size_t j = i + 1; j < n; ++j) {
            const double hij = out[i * n + j];
            const double hji = out[j * n + i];
            if (std::fabs(hij - hji) > tol * std::max(1.0, std::fabs(hij))) {
                return false;
            }
        }
    }
    return true;
}

} // namespace lightape

#endif // LIGHTAPE_REV_SECOND_ORDER_H
