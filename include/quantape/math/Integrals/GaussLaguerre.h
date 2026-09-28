#ifndef QUANTAPE_MATH_INTEGRALS_GAUSS_LAGUERRE_H
#define QUANTAPE_MATH_INTEGRALS_GAUSS_LAGUERRE_H

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
 * Nodes/weights from the Golub–Welsch construction (eigen-decomposition of
 * the Laguerre Jacobi matrix: diagonal `2i-1`, off-diagonal `i`), so any
 * order is available — unlike the tabulated orders of
 * `GaussLegendreQuadrature`. The rule is a plain weighted sum, so the
 * evaluator is scalar-generic (double, complex, AD scalars): nodes and
 * weights are fixed doubles.
 *
 * Uses are the EFGL seed rule (`mu = 0`), an independent reference rule
 * for tests, and moment gates: the n-point rule integrates
 * `x^k e^{-x}` exactly for `k <= 2n-1`.
 *
 * Construction cost is one dense symmetric eigen-solve (`O(n^3)`); build
 * the rule once per order and reuse (the model objects do).
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
            m_nodes[static_cast<std::size_t>(i)] = solver.eigenvalues()(i);
            const double first = solver.eigenvectors()(0, i);
            m_weights[static_cast<std::size_t>(i)] = first * first;
        }
    }

    std::size_t m_order;
    std::vector<double> m_nodes;
    std::vector<double> m_weights;
};

} // namespace quantape::math

#endif // QUANTAPE_MATH_INTEGRALS_GAUSS_LAGUERRE_H
