#include "quantape/markets/Curves/CurveRisk.h"

#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <utility>
#include <vector>

namespace quantape::markets {
namespace detail {

DenseLu factorDenseLu(std::vector<double> matrix, std::size_t dim) {
    if (matrix.size() != dim * dim) {
        throw std::invalid_argument("factorDenseLu: size mismatch");
    }
    DenseLu factors;
    factors.dim = dim;
    factors.pivots.resize(dim);
    for (std::size_t k = 0; k < dim; ++k) {
        std::size_t pivot = k;
        double best = std::abs(matrix[k * dim + k]);
        for (std::size_t i = k + 1; i < dim; ++i) {
            const double candidate = std::abs(matrix[i * dim + k]);
            if (candidate > best) {
                best = candidate;
                pivot = i;
            }
        }
        if (!(best > 0.0)) {
            throw std::invalid_argument("factorDenseLu: singular matrix");
        }
        factors.pivots[k] = pivot;
        if (pivot != k) {
            for (std::size_t j = k; j < dim; ++j) {
                std::swap(matrix[k * dim + j], matrix[pivot * dim + j]);
            }
        }
        const double diagonal = matrix[k * dim + k];
        for (std::size_t i = k + 1; i < dim; ++i) {
            const double factor = matrix[i * dim + k] / diagonal;
            matrix[i * dim + k] = factor;
            if (factor == 0.0) {
                continue;
            }
            for (std::size_t j = k + 1; j < dim; ++j) {
                matrix[i * dim + j] -= factor * matrix[k * dim + j];
            }
        }
    }
    factors.lu = std::move(matrix);
    return factors;
}

void solveDenseLuInto(const DenseLu& factors, const std::vector<double>& rhs,
                      std::vector<double>& x) {
    const std::size_t dim = factors.dim;
    if (rhs.size() != dim || factors.lu.size() != dim * dim) {
        throw std::invalid_argument("solveDenseLuInto: size mismatch");
    }
    x = rhs;
    for (std::size_t k = 0; k < dim; ++k) {
        const std::size_t pivot = factors.pivots[k];
        if (pivot != k) {
            std::swap(x[k], x[pivot]);
        }
        const double lead = x[k];
        for (std::size_t i = k + 1; i < dim; ++i) {
            const double lower = factors.lu[i * dim + k];
            if (lower != 0.0) {
                x[i] -= lower * lead;
            }
        }
    }
    for (std::size_t ii = dim; ii-- > 0;) {
        double sum = x[ii];
        for (std::size_t j = ii + 1; j < dim; ++j) {
            sum -= factors.lu[ii * dim + j] * x[j];
        }
        x[ii] = sum / factors.lu[ii * dim + ii];
    }
}

std::vector<double> solveDenseLu(const DenseLu& factors, const std::vector<double>& rhs) {
    std::vector<double> x;
    solveDenseLuInto(factors, rhs, x);
    return x;
}

} // namespace detail
} // namespace quantape::markets
