// test_bicubic_weights.cpp — validates the AD dispatch of BicubicInterpolation:
//
//   1. value match: double path vs var path (all methods, smooth on/off)
//   2. gradient over all z entries: stan::math::gradient vs finite
//      differences of the double implementation (away from kinks for the
//      adaptive methods)
//   3. Hessian (in z): exactly zero for Spline/Parabolic (the bicubic is
//      linear in the node values); zero away from kinks for the adaptive
//      methods in non-smooth mode
//   4. dispatch: usesWeightMatrix() true iff Spline/Parabolic with AD
//
// Run: ./test_bicubic_weights
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

using quantape::math::BicubicInterpolation;
using quantape::math::CubicDerivativeApprox;

namespace {

const std::vector<double> g_x = {0.0, 0.5, 1.1, 2.0, 3.5, 5.0};
const std::vector<double> g_y = {0.0, 0.7, 1.9, 3.2};

std::vector<std::vector<double>> baseZ() {
    std::vector<std::vector<double>> z(g_y.size(), std::vector<double>(g_x.size()));
    for (size_t j = 0; j < g_y.size(); ++j)
        for (size_t i = 0; i < g_x.size(); ++i)
            z[j][i] = std::sin(g_x[i]) + std::cos(g_y[j]) + 0.1 * g_x[i] * g_y[j] + 0.5;
    return z;
}

const auto g_z = baseZ();

// evaluation points: (x, y) pairs inside the grid
const std::vector<std::pair<double, double>> g_eval = {
    {0.25, 0.30}, {0.80, 1.00}, {1.55, 2.10}, {2.75, 3.00}, {4.20, 0.55}};

double evalDouble(const std::vector<std::vector<double>>& z, double x, double y,
                  CubicDerivativeApprox method, bool smooth) {
    const BicubicInterpolation<double> interp(g_x, g_y, z, method, smooth);
    return interp(x, y);
}

std::vector<double> fdGradient(const std::vector<std::vector<double>>& z, double x, double y,
                               CubicDerivativeApprox method, bool smooth) {
    const double h = 1e-6;
    std::vector<double> grad;
    for (size_t j = 0; j < g_y.size(); ++j) {
        for (size_t i = 0; i < g_x.size(); ++i) {
            auto zp = z;
            auto zm = z;
            zp[j][i] += h;
            zm[j][i] -= h;
            grad.push_back(
                (evalDouble(zp, x, y, method, smooth) - evalDouble(zm, x, y, method, smooth)) /
                (2.0 * h));
        }
    }
    return grad;
}

/// Second-order finite differences of the double interpolant: validates the
/// true (active-branch) z-Hessian of the AD coefficient path.
std::vector<std::vector<double>> fdHessian(double x, double y, CubicDerivativeApprox method,
                                           bool smooth) {
    const double h = 1e-4;
    const size_t nz = g_x.size() * g_y.size();
    std::vector<std::vector<double>> H(nz, std::vector<double>(nz, 0.0));
    size_t col = 0;
    for (size_t j = 0; j < g_y.size(); ++j) {
        for (size_t k = 0; k < g_x.size(); ++k, ++col) {
            auto zp = g_z;
            auto zm = g_z;
            zp[j][k] += h;
            zm[j][k] -= h;
            const auto gp = fdGradient(zp, x, y, method, smooth);
            const auto gm = fdGradient(zm, x, y, method, smooth);
            for (size_t m = 0; m < nz; ++m)
                H[m][col] = (gp[m] - gm[m]) / (2.0 * h);
        }
    }
    return H;
}

std::vector<double> adGradient(double x, double y, CubicDerivativeApprox method, bool smooth) {
    const size_t nz = g_x.size() * g_y.size();
    Eigen::VectorXd flat(nz);
    size_t k = 0;
    for (size_t j = 0; j < g_y.size(); ++j)
        for (size_t i = 0; i < g_x.size(); ++i)
            flat(k++) = g_z[j][i];

    double fx = 0.0;
    Eigen::VectorXd grad;
    stan::math::gradient(
        [&](const auto& xx) {
            using Scalar = typename std::decay_t<decltype(xx)>::Scalar;
            std::vector<std::vector<Scalar>> z(g_y.size(), std::vector<Scalar>(g_x.size()));
            size_t m = 0;
            for (size_t j = 0; j < g_y.size(); ++j)
                for (size_t i = 0; i < g_x.size(); ++i)
                    z[j][i] = xx(m++);
            if constexpr (std::is_same_v<Scalar, double>) {
                const BicubicInterpolation<double> interp(g_x, g_y, z, method, smooth);
                return interp(x, y);
            } else {
                const BicubicInterpolation<Scalar> interp(g_x, g_y, z, method, smooth);
                return interp(Scalar(x), Scalar(y));
            }
        },
        flat, fx, grad);
    return std::vector<double>(grad.data(), grad.data() + grad.size());
}

Eigen::Matrix<double, -1, -1> adHessian(double x, double y, CubicDerivativeApprox method,
                                        bool smooth) {
    const size_t nz = g_x.size() * g_y.size();
    Eigen::VectorXd flat(nz);
    size_t k = 0;
    for (size_t j = 0; j < g_y.size(); ++j)
        for (size_t i = 0; i < g_x.size(); ++i)
            flat(k++) = g_z[j][i];

    double fx = 0.0;
    Eigen::VectorXd grad;
    Eigen::Matrix<double, -1, -1> H;
    stan::math::hessian(
        [&](const auto& xx) {
            using Scalar = typename std::decay_t<decltype(xx)>::Scalar;
            std::vector<std::vector<Scalar>> z(g_y.size(), std::vector<Scalar>(g_x.size()));
            size_t m = 0;
            for (size_t j = 0; j < g_y.size(); ++j)
                for (size_t i = 0; i < g_x.size(); ++i)
                    z[j][i] = xx(m++);
            if constexpr (std::is_same_v<Scalar, double>) {
                const BicubicInterpolation<double> interp(g_x, g_y, z, method, smooth);
                return interp(x, y);
            } else {
                const BicubicInterpolation<Scalar> interp(g_x, g_y, z, method, smooth);
                return interp(Scalar(x, 0.0), Scalar(y, 0.0));
            }
        },
        flat, fx, grad, H);
    return H;
}

double maxHessian(double x, double y, CubicDerivativeApprox method, bool smooth) {
    const auto H = adHessian(x, y, method, smooth);
    double m = 0.0;
    for (int i = 0; i < H.rows(); ++i)
        for (int j = 0; j < H.cols(); ++j)
            m = std::max(m, std::abs(H(i, j)));
    return m;
}

const char* methodName(CubicDerivativeApprox da) {
    switch (da) {
        case CubicDerivativeApprox::Spline:
            return "Spline";
        case CubicDerivativeApprox::Parabolic:
            return "Parabolic";
        case CubicDerivativeApprox::Akima:
            return "Akima";
        case CubicDerivativeApprox::Kruger:
            return "Kruger";
        case CubicDerivativeApprox::Harmonic:
            return "Harmonic";
    }
    return "?";
}

} // namespace

int main() {
    std::cout << std::setprecision(6);
    const std::vector<CubicDerivativeApprox> methods = {
        CubicDerivativeApprox::Spline,   CubicDerivativeApprox::Parabolic,
        CubicDerivativeApprox::Akima,    CubicDerivativeApprox::Kruger,
        CubicDerivativeApprox::Harmonic,
    };

    for (auto method : methods) {
        for (bool smooth : {false, true}) {
            // 1) value match, var vs double
            for (const auto& [x, y] : g_eval) {
                const double ref = evalDouble(g_z, x, y, method, smooth);
                std::vector<stan::math::var> vx; // placeholder to avoid
                (void)vx;                        // unused warnings
                const BicubicInterpolation<stan::math::var> interp(g_x, g_y, g_z, method, smooth);
                const stan::math::var v = interp(stan::math::var(x), stan::math::var(y));
                const double diff = std::abs(v.val() - ref);
                if (diff > 1e-12 * (1.0 + std::abs(ref))) {
                    std::cerr << "value mismatch: " << methodName(method) << " smooth=" << smooth
                              << " (x,y)=(" << x << "," << y << ") ref=" << std::setprecision(17)
                              << ref << " var=" << v.val() << " diff=" << diff << "\n";
                    CHECK(false);
                }
            }

            // 2) gradient vs finite differences (away from kinks)
            for (const auto& [x, y] : g_eval) {
                const auto fd = fdGradient(g_z, x, y, method, smooth);
                const auto ad = adGradient(x, y, method, smooth);
                for (size_t k = 0; k < fd.size(); ++k) {
                    const double tol = smooth ? 1e-4 : 1e-5;
                    if (std::abs(fd[k] - ad[k]) > tol * (1.0 + std::abs(fd[k]))) {
                        std::cerr << "gradient mismatch: " << methodName(method)
                                  << " smooth=" << smooth << " (x,y)=(" << x << "," << y
                                  << ") k=" << k << " fd=" << fd[k] << " ad=" << ad[k] << "\n";
                        CHECK(false);
                    }
                }
            }

            // 3) Hessian: identically zero for the exactly-linear methods
            //    (Spline/Parabolic); for the adaptive methods the default AD
            //    path exposes the true active-branch curvature, verified
            //    against second-order finite differences (g_eval avoids kinks)
            if (!smooth) {
                const bool linearInZ = (method == CubicDerivativeApprox::Spline ||
                                        method == CubicDerivativeApprox::Parabolic);
                for (const auto& [x, y] : g_eval) {
                    if (linearInZ) {
                        const double h = maxHessian(x, y, method, smooth);
                        if (h > 1e-7) {
                            std::cerr << "hessian not zero: " << methodName(method) << " (x,y)=("
                                      << x << "," << y << ") max|H|=" << h << "\n";
                            CHECK(false);
                        }
                    } else {
                        const auto H = adHessian(x, y, method, smooth);
                        const auto fdH = fdHessian(x, y, method, smooth);
                        double maxdiff = 0.0, maxref = 0.0;
                        for (int i = 0; i < H.rows(); ++i) {
                            for (int j = 0; j < H.cols(); ++j) {
                                maxdiff = std::max(maxdiff, std::abs(H(i, j) - fdH[i][j]));
                                maxref = std::max(maxref, std::abs(fdH[i][j]));
                            }
                        }
                        if (maxdiff > 5e-3 * (1.0 + maxref)) {
                            std::cerr << "hessian mismatch: " << methodName(method) << " (x,y)=("
                                      << x << "," << y << ") maxdiff=" << maxdiff
                                      << " maxref=" << maxref << "\n";
                            CHECK(false);
                        }
                    }
                }
            }
        }

        // 4) dispatch flag
        {
            const BicubicInterpolation<stan::math::var> interp(g_x, g_y, g_z, method);
            const bool expectWeights = (method == CubicDerivativeApprox::Spline ||
                                        method == CubicDerivativeApprox::Parabolic);
            CHECK(interp.usesWeightMatrix() == expectWeights);
        }

        std::cout << methodName(method) << ": value/gradient/hessian all pass\n";
    }

    std::cout << "test_bicubic_weights: all invariants hold\n";
    return 0;
}
