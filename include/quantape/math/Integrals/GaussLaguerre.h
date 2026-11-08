#ifndef QUANTAPE_MATH_INTEGRALS_GAUSS_LAGUERRE_H
#define QUANTAPE_MATH_INTEGRALS_GAUSS_LAGUERRE_H

#include "quantape/math/Precision/DoubleDouble.h"

#include <Eigen/Dense>

#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <vector>

namespace quantape::math {
/**
 * @file GaussLaguerre.h
 * @brief Arbitrary-order Gauss–Laguerre rule: `int_0^inf e^{-x} f(x) dx`
 *
 * Nodes come from the Golub–Welsch construction (eigen-decomposition of the
 * Laguerre Jacobi matrix: diagonal `2i-1`, off-diagonal `i`), so any order is
 * available — unlike the tabulated orders of `GaussLegendreQuadrature`.
 *
 * Weights use the derivative form `w_i = 1/(x_i [L'_n(x_i)]^2)`. (The
 * squared first eigenvector component equals `w_i` only up to its absolute
 * roundoff, ~eps, which exceeds the tiny extreme weights: w_i ~ 1e-101 at
 * n=64.) The eigenvalues carry an absolute error ~eps*||J|| that the
 * derivative form amplifies at the extremes, so each root is polished with
 * Newton steps and `L_n`, `L'_n` are evaluated by their three-term
 * recurrences in double-double (DoubleDouble.h). Every weight is then
 * relatively accurate to ~1e-15, and `sum_i w_i x_i^k = k!` holds for
 * `k <= 2n-1` across n = 8..128.
 *
 * The rule is a plain weighted sum, so the evaluator is scalar-generic
 * (double, complex, AD scalars): nodes and weights are fixed doubles.
 *
 * Uses are the EFGL seed rule (`mu = 0`), an independent reference rule
 * for tests, and moment gates: the n-point rule integrates
 * `x^k e^{-x}` exactly for `k <= 2n-1`.
 *
 * Construction cost is one dense symmetric eigen-solve (`O(n^3)`) plus the
 * double-double recurrence (`O(n^2)`); build the rule once per order and
 * reuse (the model objects do).
 */
class GaussLaguerre {
public:
    explicit GaussLaguerre(std::size_t order) : m_order(order) {
        if (order < 1) {
            throw std::invalid_argument("GaussLaguerre: order must be positive");
        }
        build();
    }

    std::size_t order() const { return m_order; }
    const std::vector<double>& nodes() const { return m_nodes; }
    const std::vector<double>& weights() const { return m_weights; }

    /// Fixed-rule sum `sum_i w_i f(x_i)`; f is called with the scalar type.
    template <typename Scalar, typename F>
    Scalar evaluate(const F& f) const {
        Scalar sum = Scalar(0.0);
        for (std::size_t i = 0; i < m_nodes.size(); ++i) {
            sum += Scalar(m_weights[i]) * f(Scalar(m_nodes[i]));
        }
        return sum;
    }

private:
    /// `L_n(x)` and `L'_n(x)` in double-double from the three-term
    /// recurrences
    ///   (k+1) L_{k+1}     = (2k+1-x) L_k     - k L_{k-1}
    ///   (k+1) L'_{k+1}    = -L_k + (2k+1-x) L'_k - k L'_{k-1}
    /// evaluated forwards from L_0 = 1, L_1 = 1-x. The recurrences are stable
    /// here only in the relative sense the weight formula needs: at a root of
    /// L_n the intermediate terms can be ~1e130 for n=128, so double would
    /// lose the derivative entirely; double-double leaves ~1e-30.
    static void laguerreAt(std::size_t order, const DoubleDouble& x, DoubleDouble& ln,
                           DoubleDouble& lpn) {
        DoubleDouble lm2(1.0);     // L_0
        DoubleDouble lm1(1.0 - x); // L_1
        DoubleDouble dm2(0.0);     // L'_0
        DoubleDouble dm1(-1.0);    // L'_1
        for (std::size_t k = 1; k < order; ++k) {
            const DoubleDouble c(static_cast<double>(2 * k + 1));
            const DoubleDouble kk(static_cast<double>(k));
            const DoubleDouble k1(static_cast<double>(k + 1));
            const DoubleDouble lk1 = (c - x) * lm1 - kk * lm2;
            const DoubleDouble dk1 = -lm1 + (c - x) * dm1 - kk * dm2;
            lm2 = lm1;
            dm2 = dm1;
            lm1 = lk1 / k1;
            dm1 = dk1 / k1;
        }
        ln = lm1;
        lpn = dm1;
    }

    void build() {
        const Eigen::Index n = static_cast<Eigen::Index>(m_order);
        Eigen::MatrixXd jacobi = Eigen::MatrixXd::Zero(n, n);
        for (Eigen::Index i = 1; i <= n; ++i) {
            jacobi(i - 1, i - 1) = static_cast<double>(2 * i - 1);
        }
        for (Eigen::Index i = 1; i < n; ++i) {
            const double b = static_cast<double>(i);
            jacobi(i - 1, i) = b;
            jacobi(i, i - 1) = b;
        }
        const Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> solver(jacobi);
        m_nodes.resize(m_order);
        m_weights.resize(m_order);
        for (Eigen::Index i = 0; i < n; ++i) {
            // The eigenvalue error is absolute (~eps*||J||), so the relative
            // error is largest for the small nodes and the weight formula
            // amplifies it most at the extreme ones. Three Newton steps take
            // the roots to double-double accuracy (quadratic convergence from
            // ~1e-12); the extra final evaluation is the one the weight uses.
            DoubleDouble x(solver.eigenvalues()(i));
            for (int iteration = 0; iteration < 3; ++iteration) {
                DoubleDouble ln, lpn;
                laguerreAt(m_order, x, ln, lpn);
                if (lpn.hi == 0.0 && lpn.lo == 0.0) {
                    break;
                }
                x = x - ln / lpn;
            }
            DoubleDouble ln, lpn;
            laguerreAt(m_order, x, ln, lpn);
            // w_i = 1 / (x_i [L'_n(x_i)]^2): the Gauss-Laguerre weight
            // identity, evaluated with the derivative rather than the
            // eigenvector component sqrt(w_i).
            m_nodes[static_cast<std::size_t>(i)] = x.value();
            m_weights[static_cast<std::size_t>(i)] = (DoubleDouble(1.0) / (x * lpn * lpn)).value();
        }
    }

    std::size_t m_order;
    std::vector<double> m_nodes;
    std::vector<double> m_weights;
};

} // namespace quantape::math

#endif // QUANTAPE_MATH_INTEGRALS_GAUSS_LAGUERRE_H
