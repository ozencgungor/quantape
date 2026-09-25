// test_cubic_weights.cpp — validates the AD dispatch of CubicInterpolation:
//
//   1. value match: double path vs var path (all methods, smooth on/off)
//   2. gradient: stan::math::gradient vs finite differences of the double
//      implementation (away from kinks for the adaptive methods)
//   3. Hessian: exactly zero for Spline/Parabolic (linear in y); zero away
//      from kinks for the adaptive methods (piecewise-linear in y)
//   4. x-derivative: derivativeImpl vs finite difference in x
//   5. dispatch: usesWeightMatrix() true iff Spline/Parabolic
//
// Run: ./test_cubic_weights
#include "quantape/math/Interpolations/InterpolationStanPrimitives.h"
#include "quantape/math/StanMath.h"

#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <vector>

#define CHECK(cond)                                                                                \
    do {                                                                                           \
        if (!(cond)) {                                                                             \
            std::cerr << "FAIL: " << #cond << " (line " << __LINE__ << ")\n";                      \
            std::exit(1);                                                                          \
        }                                                                                          \
    } while (0)

using quantape::math::CubicInterpolation;

namespace {

const std::vector<double> g_x = {0.0, 0.5, 1.1, 2.0, 3.5, 5.0};
const std::vector<double> g_y = {1.3, 0.9, 2.4, 2.1, 0.7, 1.8};
// evaluation points: segment midpoints, away from adaptive-method kinks
const std::vector<double> g_eval = {0.25, 0.8, 1.55, 2.75, 4.25};

double evalDouble(const std::vector<double>& y, double x,
                  CubicInterpolation<double>::DerivativeApprox da, bool smooth) {
    const CubicInterpolation<double> interp(g_x, y, da, smooth);
    return interp(x);
}

std::vector<double> fdGradient(const std::vector<double>& y, double x,
                               CubicInterpolation<double>::DerivativeApprox da, bool smooth) {
    const double h = 1e-6;
    std::vector<double> grad(y.size(), 0.0);
    for (size_t j = 0; j < y.size(); ++j) {
        auto yp = y;
        auto ym = y;
        yp[j] += h;
        ym[j] -= h;
        grad[j] = (evalDouble(yp, x, da, smooth) - evalDouble(ym, x, da, smooth)) / (2.0 * h);
    }
    return grad;
}

/// Second-order finite differences of the double interpolant: validates the
/// true (active-branch) y-Hessian of the AD coefficient path.
std::vector<std::vector<double>>
fdHessian(double x, CubicInterpolation<double>::DerivativeApprox da, bool smooth) {
    const double h = 1e-4;
    const size_t n = g_y.size();
    std::vector<std::vector<double>> H(n, std::vector<double>(n, 0.0));
    for (size_t i = 0; i < n; ++i) {
        auto yp = g_y;
        auto ym = g_y;
        yp[i] += h;
        ym[i] -= h;
        const auto gp = fdGradient(yp, x, da, smooth);
        const auto gm = fdGradient(ym, x, da, smooth);
        for (size_t j = 0; j < n; ++j)
            H[j][i] = (gp[j] - gm[j]) / (2.0 * h);
    }
    return H;
}

std::vector<double> adGradient(double x, CubicInterpolation<double>::DerivativeApprox da,
                               bool smooth) {
    const Eigen::VectorXd xv = Eigen::Map<const Eigen::VectorXd>(g_y.data(), g_y.size());
    double fx = 0.0;
    Eigen::VectorXd grad;
    stan::math::gradient(
        [&](const auto& xx) {
            using Scalar = typename std::decay_t<decltype(xx)>::Scalar;
            if constexpr (std::is_same_v<Scalar, double>) {
                // Stan's empty-argument instantiation — must return double
                std::vector<double> yv(xx.data(), xx.data() + xx.size());
                const CubicInterpolation<double> interp(g_x, yv, da, smooth);
                return interp(x);
            } else {
                std::vector<Scalar> yv(xx.data(), xx.data() + xx.size());
                const CubicInterpolation<Scalar> interp(g_x, yv, da, smooth);
                return interp(Scalar(x));
            }
        },
        xv, fx, grad);
    return std::vector<double>(grad.data(), grad.data() + grad.size());
}

Eigen::Matrix<double, -1, -1> adHessian(double x, CubicInterpolation<double>::DerivativeApprox da,
                                        bool smooth) {
    const Eigen::VectorXd xv = Eigen::Map<const Eigen::VectorXd>(g_y.data(), g_y.size());
    double fx = 0.0;
    Eigen::VectorXd grad;
    Eigen::Matrix<double, -1, -1> H;
    stan::math::hessian(
        [&](const auto& xx) {
            using Scalar = typename std::decay_t<decltype(xx)>::Scalar;
            if constexpr (std::is_same_v<Scalar, double>) {
                std::vector<double> yv(xx.data(), xx.data() + xx.size());
                const CubicInterpolation<double> interp(g_x, yv, da, smooth);
                return interp(x);
            } else {
                std::vector<Scalar> yv(xx.data(), xx.data() + xx.size());
                const CubicInterpolation<Scalar> interp(g_x, yv, da, smooth);
                return interp(Scalar(x, 0.0)); // fvar: tangent 0
            }
        },
        xv, fx, grad, H);
    return H;
}

double maxHessian(double x, CubicInterpolation<double>::DerivativeApprox da, bool smooth) {
    const auto H = adHessian(x, da, smooth);
    double m = 0.0;
    for (int i = 0; i < H.rows(); ++i)
        for (int j = 0; j < H.cols(); ++j)
            m = std::max(m, std::abs(H(i, j)));
    return m;
}

const char* methodName(CubicInterpolation<double>::DerivativeApprox da) {
    switch (da) {
        case quantape::math::CubicDerivativeApprox::Spline:
            return "Spline";
        case quantape::math::CubicDerivativeApprox::Parabolic:
            return "Parabolic";
        case quantape::math::CubicDerivativeApprox::Akima:
            return "Akima";
        case quantape::math::CubicDerivativeApprox::Kruger:
            return "Kruger";
        case quantape::math::CubicDerivativeApprox::Harmonic:
            return "Harmonic";
    }
    return "?";
}

} // namespace

int main() {
    std::cout << std::setprecision(6);
    const std::vector<CubicInterpolation<double>::DerivativeApprox> methods = {
        quantape::math::CubicDerivativeApprox::Spline,
        quantape::math::CubicDerivativeApprox::Parabolic,
        quantape::math::CubicDerivativeApprox::Akima,
        quantape::math::CubicDerivativeApprox::Kruger,
        quantape::math::CubicDerivativeApprox::Harmonic,
    };

    for (auto da : methods) {
        for (bool smooth : {false, true}) {
            // 1) value match, var vs double
            for (double x : g_eval) {
                const double ref = evalDouble(g_y, x, da, smooth);
                std::vector<stan::math::var> yv(g_y.begin(), g_y.end());
                const CubicInterpolation<stan::math::var> interp(g_x, yv, da, smooth);
                const stan::math::var v = interp(stan::math::var(x));
                const double diff = std::abs(v.val() - ref);
                if (diff > 1e-12 * (1.0 + std::abs(ref))) {
                    std::cerr << "value mismatch: " << methodName(da) << " smooth=" << smooth
                              << " x=" << x << " ref=" << std::setprecision(17) << ref
                              << " var=" << v.val() << " diff=" << diff << "\n";
                    CHECK(false);
                }
            }

            // 2) gradient vs finite differences (away from kinks)
            for (double x : g_eval) {
                const auto fd = fdGradient(g_y, x, da, smooth);
                const auto ad = adGradient(x, da, smooth);
                for (size_t j = 0; j < g_y.size(); ++j) {
                    const double tol = smooth ? 1e-4 : 1e-5;
                    if (std::abs(fd[j] - ad[j]) > tol * (1.0 + std::abs(fd[j]))) {
                        std::cerr << "gradient mismatch: " << methodName(da) << " smooth=" << smooth
                                  << " x=" << x << " j=" << j << " fd=" << fd[j] << " ad=" << ad[j]
                                  << "\n";
                        CHECK(false);
                    }
                }
            }

            // 3) Hessian: identically zero for the exactly-linear methods
            //    (Spline/Parabolic); for the adaptive methods the default AD
            //    path now exposes the true active-branch curvature, verified
            //    against second-order finite differences (g_eval avoids kinks)
            if (!smooth) {
                const bool linearInY = (da == quantape::math::CubicDerivativeApprox::Spline ||
                                        da == quantape::math::CubicDerivativeApprox::Parabolic);
                for (double x : g_eval) {
                    const auto H = adHessian(x, da, smooth);
                    if (linearInY) {
                        if (maxHessian(x, da, smooth) > 1e-8) {
                            std::cerr << "hessian not zero: " << methodName(da) << " x=" << x
                                      << " max|H|=" << maxHessian(x, da, smooth) << "\n";
                            CHECK(false);
                        }
                    } else {
                        const auto fdH = fdHessian(x, da, smooth);
                        double maxdiff = 0.0, maxref = 0.0;
                        for (int i = 0; i < H.rows(); ++i) {
                            for (int j = 0; j < H.cols(); ++j) {
                                maxdiff = std::max(maxdiff, std::abs(H(i, j) - fdH[i][j]));
                                maxref = std::max(maxref, std::abs(fdH[i][j]));
                            }
                        }
                        if (maxdiff > 5e-3 * (1.0 + maxref)) {
                            std::cerr << "hessian mismatch: " << methodName(da) << " x=" << x
                                      << " maxdiff=" << maxdiff << " maxref=" << maxref << "\n";
                            CHECK(false);
                        }
                    }
                }
            }

            // 4) x-derivative vs finite difference
            for (double x : g_eval) {
                const double h = 1e-6;
                const double fd =
                    (evalDouble(g_y, x + h, da, smooth) - evalDouble(g_y, x - h, da, smooth)) /
                    (2.0 * h);
                std::vector<stan::math::var> yv(g_y.begin(), g_y.end());
                const CubicInterpolation<stan::math::var> interp(g_x, yv, da, smooth);
                const double ad = interp.derivative(stan::math::var(x)).val();
                const double tol = smooth ? 1e-4 : 1e-5;
                if (std::abs(fd - ad) > tol * (1.0 + std::abs(fd))) {
                    std::cerr << "x-derivative mismatch: " << methodName(da) << " smooth=" << smooth
                              << " x=" << x << " fd=" << fd << " ad=" << ad << "\n";
                    CHECK(false);
                }
            }
        }

        // 5) dispatch flag
        {
            std::vector<stan::math::var> yv(g_y.begin(), g_y.end());
            const CubicInterpolation<stan::math::var> interp(g_x, yv, da);
            const bool expectWeights = (da == quantape::math::CubicDerivativeApprox::Spline ||
                                        da == quantape::math::CubicDerivativeApprox::Parabolic);
            CHECK(interp.usesWeightMatrix() == expectWeights);
        }

        std::cout << methodName(da) << ": value/gradient/hessian/x-derivative all pass\n";
    }

    std::cout << "test_cubic_weights: all invariants hold\n";
    return 0;
}
