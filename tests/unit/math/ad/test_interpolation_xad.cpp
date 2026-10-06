// test_interpolation_xad.cpp — validates evaluation-point AD of the
// interpolators (Phase 1): the default operator()/derivative() put the query
// coordinate on the tape.
//
//   1. linear:        dI/dx = segment slope; mixed d2I/dxdy = ±inv_dx
//   2. log-linear:    analytic dI/dx and mixed entries
//   3. bilinear:      analytic dI/dx, dI/dy, mixed d2I/dxdy and d2I/dxdz
//   4. cubic (Spline): dI/dx = derivative(); mixed d2I/dxdy vs FD of the
//                     spline derivative in y
//   5. evaluateFixed: x adjoint stays zero (passive-abscissa policy)
#include "quantape/math/StanMath.h"

#include "quantape/math/Interpolations/InterpolationStanPrimitives.h"

#include <cmath>
#include <cstdio>
#include <vector>

#include "support/GtestSupport.h"
#include "support/StanTapeFixture.h"

using quantape::math::BilinearInterpolation;
using quantape::math::CubicDerivativeApprox;
using quantape::math::CubicInterpolation;
using quantape::math::LinearInterpolation;
using quantape::math::LogLinearInterpolation;

class InterpolationXadTest : public StanTapeTest {};

namespace {

const std::vector<double> g_x{0.0, 1.0, 2.0, 3.0};
const std::vector<double> g_y{0.0, 1.0, 4.0, 9.0};

/// Full Hessian of a 1D interpolation evaluated at the query point, with the
/// flat parameter vector [x, y_0..y_{n-1}].
template <typename Build>
Eigen::Matrix<double, -1, -1> hessian1D(double x, const std::vector<double>& y, Build build) {
    const size_t n = y.size();
    Eigen::VectorXd flat(n + 1);
    flat(0) = x;
    for (size_t i = 0; i < n; ++i)
        flat(static_cast<Eigen::Index>(i + 1)) = y[i];

    double fx = 0.0;
    Eigen::VectorXd grad;
    Eigen::Matrix<double, -1, -1> H;
    stan::math::hessian(
        [&](const auto& v) {
            using S = typename std::decay_t<decltype(v)>::Scalar;
            std::vector<S> yv(n);
            for (size_t i = 0; i < n; ++i)
                yv[i] = v(static_cast<Eigen::Index>(i + 1));
            return build(v(0), yv);
        },
        flat, fx, grad, H);
    return H;
}

/// Same for the 2D grid: [x, y, z_00..z_{ny-1,nx-1}].
template <typename Build>
Eigen::Matrix<double, -1, -1> hessian2D(double x, double y,
                                        const std::vector<std::vector<double>>& z, Build build) {
    const size_t nx = z[0].size();
    const size_t nz = nx * z.size();
    Eigen::VectorXd flat(nz + 2);
    flat(0) = x;
    flat(1) = y;
    size_t k = 2;
    for (const auto& row : z)
        for (double zij : row)
            flat(static_cast<Eigen::Index>(k++)) = zij;

    double fx = 0.0;
    Eigen::VectorXd grad;
    Eigen::Matrix<double, -1, -1> H;
    stan::math::hessian(
        [&](const auto& v) {
            using S = typename std::decay_t<decltype(v)>::Scalar;
            std::vector<std::vector<S>> zv(z.size(), std::vector<S>(nx));
            size_t m = 2;
            for (size_t j = 0; j < z.size(); ++j)
                for (size_t i = 0; i < nx; ++i)
                    zv[j][i] = v(static_cast<Eigen::Index>(m++));
            return build(v(0), v(1), zv);
        },
        flat, fx, grad, H);
    return H;
}

} // namespace

TEST_F(InterpolationXadTest, linearHessian) {
    const double x = 1.5;

    const auto H =
        hessian1D(x, g_y, [](const auto& xq, const std::vector<std::decay_t<decltype(xq)>>& yv) {
            using S = std::decay_t<decltype(xq)>;
            const LinearInterpolation<S> interp(g_x, yv);
            return interp(xq);
        });

    // value = 2.5 (y[1]=1, y[2]=4), slope = 3, mixed ±inv_dx = ±1
    CHECK_CLOSE("linear hessian d2I/dx dy1", H(0, 2), -1.0, 1e-12);
    CHECK_CLOSE("linear hessian d2I/dx dy2", H(0, 3), 1.0, 1e-12);
    CHECK_CLOSE("linear hessian d2I/dx2", H(0, 0), 0.0, 1e-12);
    CHECK_CLOSE("linear hessian d2I/dy1dy2", H(2, 3), 0.0, 1e-12);
}

TEST_F(InterpolationXadTest, logLinearHessian) {
    const double x = 1.5;
    const double t = 0.5, dx = 1.0;
    const double L1 = std::log(1.0), L2 = std::log(4.0);
    const double f = std::exp((1.0 - t) * L1 + t * L2);

    const auto H =
        hessian1D(x, g_y, [](const auto& xq, const std::vector<std::decay_t<decltype(xq)>>& yv) {
            using S = std::decay_t<decltype(xq)>;
            const LogLinearInterpolation<S> interp(g_x, yv);
            return interp(xq);
        });

    CHECK_CLOSE("loglin hessian d2I/dx dy1", H(0, 2),
                f * ((1.0 - t) * (L2 - L1) - 1.0) / (g_y[1] * dx), 1e-10);
    CHECK_CLOSE("loglin hessian d2I/dx dy2", H(0, 3), f * (t * (L2 - L1) + 1.0) / (g_y[2] * dx),
                1e-10);
    CHECK_CLOSE("loglin hessian d2I/dx2", H(0, 0), f * (L2 - L1) * (L2 - L1) / (dx * dx), 1e-10);
}

TEST_F(InterpolationXadTest, bilinearHessian) {
    const std::vector<double> bx{0.0, 1.0, 2.0};
    const std::vector<double> by{0.0, 1.0};
    const std::vector<std::vector<double>> z{{1.0, 2.0, 3.0}, {4.0, 6.0, 8.0}};
    const double x = 0.5, y = 0.5;

    const auto H = hessian2D(x, y, z,
                             [&](const auto& xq, const auto& yq,
                                 const std::vector<std::vector<std::decay_t<decltype(xq)>>>& zv) {
                                 using S = std::decay_t<decltype(xq)>;
                                 const BilinearInterpolation<S> interp(bx, by, zv);
                                 return interp(xq, yq);
                             });

    // flat order: 0=x, 1=y, 2=z00, 3=z10, 4=z20, 5=z01, 6=z11, 7=z21
    CHECK_CLOSE("bilinear d2I/dx dz00", H(0, 2), -0.5, 1e-12);
    CHECK_CLOSE("bilinear d2I/dx dz10", H(0, 3), 0.5, 1e-12);
    CHECK_CLOSE("bilinear d2I/dy dz00", H(1, 2), -0.5, 1e-12);
    CHECK_CLOSE("bilinear d2I/dx dy", H(0, 1), 1.0, 1e-12); // (z11-z01)-(z10-z00)
}

TEST_F(InterpolationXadTest, cubicSplineMixedHessian) {
    const double x = 1.55;

    const auto H =
        hessian1D(x, g_y, [](const auto& xq, const std::vector<std::decay_t<decltype(xq)>>& yv) {
            using S = std::decay_t<decltype(xq)>;
            const CubicInterpolation<S> interp(g_x, yv, CubicDerivativeApprox::Spline);
            return interp(xq);
        });

    // dI/dx must equal the spline derivative; mixed entries validate against
    // FD of the double spline derivative in y
    const double h = 1e-6;
    for (size_t j = 0; j < g_y.size(); ++j) {
        auto yp = g_y, ym = g_y;
        yp[j] += h;
        ym[j] -= h;
        const CubicInterpolation<double> ip(g_x, yp, CubicDerivativeApprox::Spline);
        const CubicInterpolation<double> im(g_x, ym, CubicDerivativeApprox::Spline);
        const double fd = (ip.derivative(x) - im.derivative(x)) / (2.0 * h);
        char label[64];
        std::snprintf(label, sizeof(label), "cubic d2I/dx dy%zu", j);
        CHECK_CLOSE(label, H(0, static_cast<Eigen::Index>(j + 1)), fd,
                    1e-5 * (1.0 + std::fabs(fd)));
    }
}

TEST_F(InterpolationXadTest, evaluateFixedAndDefaultAbscissaAdjoints) {
    {
        stan::math::var x = 1.5;
        std::vector<stan::math::var> y{0.0, 1.0, 4.0, 9.0};
        const LinearInterpolation<stan::math::var> interp(g_x, y);

        stan::math::var fixed = interp.evaluateFixed(x);
        fixed.grad();
        CHECK_CLOSE("evaluateFixed x adjoint", x.adj(), 0.0, 1e-12);
        CHECK_CLOSE("evaluateFixed y2 adjoint", y[2].adj(), 0.5, 1e-12);
    }
    {
        stan::math::var x = 1.5;
        std::vector<stan::math::var> y{0.0, 1.0, 4.0, 9.0};
        const LinearInterpolation<stan::math::var> interp(g_x, y);

        stan::math::var ad = interp(x);
        ad.grad();
        CHECK_CLOSE("default x adjoint", x.adj(), 3.0, 1e-12);
        CHECK_CLOSE("default y2 adjoint", y[2].adj(), 0.5, 1e-12);
    }
}
