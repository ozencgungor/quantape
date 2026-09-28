#ifndef QUANTAPE_MATH_INTEGRALS_GAUSS_LEGENDRE_H
#define QUANTAPE_MATH_INTEGRALS_GAUSS_LEGENDRE_H

#include <Eigen/Dense>

#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <vector>

namespace quantape::math {
/**
 * @file GaussLegendre.h
 * @brief Arbitrary-order Gauss–Legendre rule on `[-1, 1]`
 *
 * Complement to `GaussianQuadrature.h`'s tabulated
 * `GaussLegendreQuadrature` (orders 2..20): nodes/weights for any order via
 * the Golub–Welsch construction (Legendre Jacobi matrix: zero diagonal,
 * off-diagonal `i / sqrt(4 i^2 - 1)`). The rule is a plain weighted sum
 * with fixed double nodes/weights, so `integrate` is scalar-generic
 * (double, complex, AD scalars) — the property needed by the complex-step
 * Heston Jacobians.
 *
 * Primary use here: finite-interval truncation of the Lewis integral,
 * `int_0^{u_max} A(u) du = u_max * sum_i w_i A(u_max x_i)` with
 * `{x_i, w_i}` the rule mapped to `[0, 1]` (see `nodes01()`/`weights01()`).
 */
class GaussLegendre {
public:
    explicit GaussLegendre(std::size_t order) : order_(order) {
        if (order < 1) {
            throw std::invalid_argument("GaussLegendre: order must be positive");
        }
        build();
    }

    std::size_t order() const { return order_; }
    /// Nodes on [-1, 1], ascending.
    const std::vector<double>& nodes() const { return nodes_; }
    const std::vector<double>& weights() const { return weights_; }

    /// Fixed-rule integral over [a, b] with a scalar-generic integrand.
    template <typename Scalar, typename F>
    Scalar integrate(const F& f, const Scalar& a, const Scalar& b) const {
        const Scalar scale = (b - a) / Scalar(2.0);
        const Scalar shift = (b + a) / Scalar(2.0);
        Scalar sum = Scalar(0.0);
        for (std::size_t i = 0; i < nodes_.size(); ++i) {
            sum += Scalar(weights_[i]) * f(scale * Scalar(nodes_[i]) + shift);
        }
        return scale * sum;
    }

private:
    void build() {
        const Eigen::Index n = static_cast<Eigen::Index>(order_);
        Eigen::MatrixXd jacobi = Eigen::MatrixXd::Zero(n, n);
        for (Eigen::Index i = 1; i < n; ++i) {
            const double b =
                static_cast<double>(i) / std::sqrt(4.0 * static_cast<double>(i * i) - 1.0);
            jacobi(i - 1, i) = b;
            jacobi(i, i - 1) = b;
        }
        const Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> solver(jacobi);
        nodes_.resize(order_);
        weights_.resize(order_);
        for (Eigen::Index i = 0; i < n; ++i) {
            nodes_[static_cast<std::size_t>(i)] = solver.eigenvalues()(i);
            const double first = solver.eigenvectors()(0, i);
            weights_[static_cast<std::size_t>(i)] = 2.0 * first * first;
        }
    }

    std::size_t order_;
    std::vector<double> nodes_;
    std::vector<double> weights_;
};

} // namespace quantape::math

#endif // QUANTAPE_MATH_INTEGRALS_GAUSS_LEGENDRE_H
