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
#include "Math/Interpolations/InterpolationStanPrimitives.h"

#include <stan/math.hpp>
#include <stan/math/mix.hpp>

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

using Math::CubicInterpolation;

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

std::vector<double> fdGradient(double x, CubicInterpolation<double>::DerivativeApprox da,
                               bool smooth) {
    const double h = 1e-6;
    std::vector<double> grad(g_y.size(), 0.0);
    for (size_t j = 0; j < g_y.size(); ++j) {
        auto yp = g_y;
        auto ym = g_y;
        yp[j] += h;
        ym[j] -= h;
        grad[j] = (evalDouble(yp, x, da, smooth) - evalDouble(ym, x, da, smooth)) / (2.0 * h);
    }
    return grad;
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

double maxHessian(double x, CubicInterpolation<double>::DerivativeApprox da, bool smooth) {
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
    double m = 0.0;
    for (int i = 0; i < H.rows(); ++i)
        for (int j = 0; j < H.cols(); ++j)
            m = std::max(m, std::abs(H(i, j)));
    return m;
}

const char* methodName(CubicInterpolation<double>::DerivativeApprox da) {
    switch (da) {
        case Math::CubicDerivativeApprox::Spline:
            return "Spline";
        case Math::CubicDerivativeApprox::Parabolic:
            return "Parabolic";
        case Math::CubicDerivativeApprox::Akima:
            return "Akima";
        case Math::CubicDerivativeApprox::Kruger:
            return "Kruger";
        case Math::CubicDerivativeApprox::Harmonic:
            return "Harmonic";
    }
    return "?";
}

} // namespace

int main() {
    std::cout << std::setprecision(6);
    const std::vector<CubicInterpolation<double>::DerivativeApprox> methods = {
        Math::CubicDerivativeApprox::Spline,   Math::CubicDerivativeApprox::Parabolic,
        Math::CubicDerivativeApprox::Akima,    Math::CubicDerivativeApprox::Kruger,
        Math::CubicDerivativeApprox::Harmonic,
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
                const auto fd = fdGradient(x, da, smooth);
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

            // 3) Hessian: zero for linear methods; zero away from kinks for
            //    adaptive ones (smoothing deliberately breaks this)
            if (!smooth) {
                for (double x : g_eval) {
                    const double h = maxHessian(x, da, smooth);
                    if (h > 1e-8) {
                        std::cerr << "hessian not zero: " << methodName(da) << " x=" << x
                                  << " max|H|=" << h << "\n";
                        CHECK(false);
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
            const bool expectWeights = (da == Math::CubicDerivativeApprox::Spline ||
                                        da == Math::CubicDerivativeApprox::Parabolic);
            CHECK(interp.usesWeightMatrix() == expectWeights);
        }

        std::cout << methodName(da) << ": value/gradient/hessian/x-derivative all pass\n";
    }

    std::cout << "test_cubic_weights: all invariants hold\n";
    return 0;
}
