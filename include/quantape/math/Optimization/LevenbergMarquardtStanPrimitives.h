#ifndef QUANTAPE_MATH_LEVENBERG_MARQUARDT_STAN_PRIMITIVES_H
#define QUANTAPE_MATH_LEVENBERG_MARQUARDT_STAN_PRIMITIVES_H

#include "quantape/math/StanMath.h"

#include "quantape/math/Optimization/LevenbergMarquardt.h"

#include <cstddef>
#include <stdexcept>
#include <vector>

namespace quantape::math {
/**
 * @file LevenbergMarquardtStanPrimitives.h
 * @brief Exact-AD-Jacobian entry point for the LM solver
 *
 * The optimizer still iterates in `double`; only the Jacobian is computed with
 * Stan reverse mode (`stan::math::jacobian`), so the fitted optimum is a plain
 * double point. Risk flows through the standard IFT path
 * (`calibrationIft`/`kktCalibrationJacobian`), never by differentiating
 * through the solver.
 *
 * The residual must be scalar-generic:
 * `void r(const std::vector<S>& x, std::vector<S>& out)` for `S = double` and
 * `S = stan::math::var`. The template parameter is deduced by the caller and
 * may be shared with other data: the residual is invoked with the parameter
 * vector as `S` and any fixed data (e.g. the market vector) as `double`.
 */
template <typename Residual>
LevenbergMarquardtResult levenbergMarquardtAd(const Residual& residuals, std::vector<double>& x,
                                              const LevenbergMarquardtOptions& options = {}) {
    const std::size_t n = x.size();
    if (n == 0) {
        throw std::invalid_argument("levenbergMarquardtAd: empty parameter vector");
    }
    std::vector<double> residual;
    residuals(x, residual);
    const std::size_t m = residual.size();
    if (m == 0) {
        throw std::invalid_argument("levenbergMarquardtAd: empty residual vector");
    }

    const auto jacobian = [&](const std::vector<double>& point, std::vector<double>& flat) {
        if (point.size() != n || flat.size() != m * n) {
            throw std::invalid_argument("levenbergMarquardtAd: size mismatch");
        }
        const Eigen::Matrix<double, Eigen::Dynamic, 1> pointEigen =
            Eigen::Map<const Eigen::Matrix<double, Eigen::Dynamic, 1>>(point.data(),
                                                                       static_cast<Eigen::Index>(n));
        Eigen::Matrix<double, Eigen::Dynamic, 1> valueEigen;
        Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> jacobianEigen;
        stan::math::jacobian(
            [&](const Eigen::Matrix<stan::math::var, Eigen::Dynamic, 1>& varPoint) {
                std::vector<stan::math::var> input(varPoint.data(),
                                                   varPoint.data() + varPoint.size());
                std::vector<stan::math::var> output;
                residuals(input, output);
                Eigen::Matrix<stan::math::var, Eigen::Dynamic, 1> result(
                    static_cast<Eigen::Index>(output.size()));
                for (std::size_t i = 0; i < output.size(); ++i) {
                    result(static_cast<Eigen::Index>(i)) = output[i];
                }
                return result;
            },
            pointEigen, valueEigen, jacobianEigen);
        if (static_cast<std::size_t>(jacobianEigen.rows()) != m ||
            static_cast<std::size_t>(jacobianEigen.cols()) != n) {
            throw std::invalid_argument(
                "levenbergMarquardtAd: residual size changed between evaluations");
        }
        for (std::size_t i = 0; i < m; ++i) {
            for (std::size_t j = 0; j < n; ++j) {
                flat[i * n + j] = jacobianEigen(static_cast<Eigen::Index>(i),
                                                static_cast<Eigen::Index>(j));
            }
        }
    };

    const auto residualDouble = [&](const std::vector<double>& point, std::vector<double>& out) {
        residuals(point, out);
        if (out.size() != m) {
            throw std::invalid_argument(
                "levenbergMarquardtAd: residual size changed between evaluations");
        }
    };
    return detail::levenbergMarquardtCore(residualDouble, jacobian, x, options);
}

} // namespace quantape::math

#endif // QUANTAPE_MATH_LEVENBERG_MARQUARDT_STAN_PRIMITIVES_H
