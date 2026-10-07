#include "quantape/markets/Curves/CurveRisk.h"

#include <algorithm>
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

QuoteRisk transformQuoteRiskFromJacobian(const std::vector<double>& f, std::size_t m,
                                         const std::vector<double>& dVdZeros) {
    if (f.size() != m * m || dVdZeros.size() != m + 1) {
        throw std::invalid_argument("transformQuoteRisk: size mismatch");
    }
    std::vector<double> fTranspose(m * m);
    for (std::size_t j = 0; j < m; ++j) {
        for (std::size_t i = 0; i < m; ++i) {
            fTranspose[i * m + j] = f[j * m + i];
        }
    }
    std::vector<double> g(m);
    for (std::size_t i = 0; i < m; ++i) {
        g[i] = dVdZeros[i + 1];
    }
    QuoteRisk result;
    result.quoteDeltas = quantape::math::solveDense(std::move(fTranspose), m, g);
    return result;
}

QuoteGamma transformQuoteGammaFromJacobian(std::vector<double> f, std::size_t m,
                                           const std::vector<double>& HVdZeros) {
    const std::size_t n = m + 1;
    if (f.size() != m * m || HVdZeros.size() != n * n) {
        throw std::invalid_argument("transformQuoteGamma: size mismatch");
    }
    // J = F^{-1} by one partial-pivoted LU factorization plus m
    // back-substitutions, instead of m independent dense solves.
    const DenseLu factors = factorDenseLu(std::move(f), m);
    std::vector<double> jacobianInverse(m * m, 0.0);
    std::vector<double> unit(m, 0.0);
    std::vector<double> solution;
    for (std::size_t column = 0; column < m; ++column) {
        std::fill(unit.begin(), unit.end(), 0.0);
        unit[column] = 1.0;
        solveDenseLuInto(factors, unit, solution);
        for (std::size_t k = 0; k < m; ++k) {
            jacobianInverse[k * m + column] = solution[k];
        }
    }
    // H_r = J^T H_zeta J, restricting H_zeta to the solved nodes.
    std::vector<double> tmp(m * m, 0.0);
    for (std::size_t i = 0; i < m; ++i) {
        for (std::size_t b = 0; b < m; ++b) {
            double sum = 0.0;
            for (std::size_t a = 0; a < m; ++a) {
                sum += HVdZeros[(i + 1) * n + (a + 1)] * jacobianInverse[a * m + b];
            }
            tmp[i * m + b] = sum;
        }
    }
    QuoteGamma result;
    result.dim = m;
    result.hessian.assign(m * m, 0.0);
    for (std::size_t i = 0; i < m; ++i) {
        for (std::size_t j = 0; j < m; ++j) {
            double sum = 0.0;
            for (std::size_t a = 0; a < m; ++a) {
                sum += jacobianInverse[a * m + i] * tmp[a * m + j];
            }
            result.hessian[i * m + j] = sum;
        }
    }
    // Symmetrize: round-off in the two matrix products can leave an
    // asymmetric residual even though the analytic transform is symmetric.
    for (std::size_t i = 0; i < m; ++i) {
        for (std::size_t j = i + 1; j < m; ++j) {
            const double symmetric = 0.5 * (result.hessian[i * m + j] + result.hessian[j * m + i]);
            result.hessian[i * m + j] = symmetric;
            result.hessian[j * m + i] = symmetric;
        }
    }
    return result;
}

} // namespace detail
} // namespace quantape::markets
