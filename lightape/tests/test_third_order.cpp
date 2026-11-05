#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>

#include "lightape/lightape.h"
#include "lightape/mix/ThirdOrder.h"
#include <gtest/gtest.h>

namespace {

using lightape::FwdFwdRevScalar;
using lightape::RevScalar;
using lightape::RevTape;
using lightape::Tangent;

class ThirdOrderTest : public ::testing::Test {
protected:
    void SetUp() override { RevTape::active().clear(); }
    void TearDown() override { RevTape::active().clear(); }
};

/// Seed one input on the tape for the two independent forward direction sets:
/// the value-level tangents get `inner[l]` (the `B` set), the outer tangent
/// values get `outer[k]` (the `A` set), and the cross slots are zero (the
/// straight-line parameterization `x(eps, eta) = x0 + sum_k eps_k A_k +
/// sum_l eta_l B_l`). `value` becomes the tape leaf.
template <std::size_t N>
FwdFwdRevScalar<N> SeedInput(RevTape& tape, double value, const double* outer,
                             const double* inner) {
    using Level1 = Tangent<RevScalar, N>;
    using Level2 = Tangent<Level1, N>;
    std::array<RevScalar, N> innerTangents{};
    std::array<Level1, N> outerTangents{};
    for (std::size_t k = 0; k < N; ++k) {
        innerTangents[k] = RevScalar(inner[k]);
        const std::array<RevScalar, N> cross{};
        outerTangents[k] = Level1(RevScalar(outer[k]), cross);
    }
    return Level2(Level1(RevScalar(value, tape.input(value)), innerTangents), outerTangents);
}

/// Third derivatives of f = exp(x y) + log(x) - 3 sin(y) at (x, y): the four
/// distinct entries of the symmetric basis tensor.
struct ExpLogSinThird {
    double xxx;
    double xxy;
    double xyy;
    double yyy;
};

ExpLogSinThird ExpLogSinDerivs(double x, double y) {
    const double e = std::exp(x * y);
    return {y * y * y * e + 2.0 / (x * x * x), y * (2.0 + x * y) * e, x * (2.0 + x * y) * e,
            x * x * x * e + 3.0 * std::cos(y)};
}

/// Pick the symmetric entry by the number of y slots (indices equal 1).
double Pick(const ExpLogSinThird& d, int a, int b, int c) {
    const int yCount = a + b + c;
    if (yCount == 0) {
        return d.xxx;
    }
    if (yCount == 1) {
        return d.xxy;
    }
    if (yCount == 2) {
        return d.xyy;
    }
    return d.yyy;
}

/// f = exp(x y) + log(x) - 3 sin(y) at (0.7, 1.3) with basis seeds: the
/// `N x N x n` output is the dense third derivative tensor, checked against
/// the analytic formulas at 1e-12. This is the three-way contraction with
/// A_k = B_k = e_k.
TEST_F(ThirdOrderTest, ExpLogSinTensorMatchesAnalytic) {
    constexpr std::size_t n = 2;
    RevTape& tape = RevTape::active();
    const double xv = 0.7;
    const double yv = 1.3;
    const double basis[2][2] = {{1.0, 0.0}, {0.0, 1.0}};

    const FwdFwdRevScalar<n> x = SeedInput<n>(tape, xv, basis[0], basis[0]);
    const FwdFwdRevScalar<n> y = SeedInput<n>(tape, yv, basis[1], basis[1]);
    const FwdFwdRevScalar<n> f = exp(x * y) + log(x) - 3.0 * sin(y);

    const std::uint32_t inputs[n] = {x.value.value.node, y.value.value.node};
    const std::size_t recorded = tape.size();
    double out[n * n * n] = {};
    lightape::reverseThird(f, n, inputs, out);
    EXPECT_EQ(tape.size(), recorded); // the sweeps append no nodes

    const ExpLogSinThird d = ExpLogSinDerivs(xv, yv);
    for (std::size_t k = 0; k < n; ++k) {
        for (std::size_t l = 0; l < n; ++l) {
            for (std::size_t p = 0; p < n; ++p) {
                EXPECT_NEAR(out[(k * n + l) * n + p],
                            Pick(d, static_cast<int>(k), static_cast<int>(l), static_cast<int>(p)),
                            1e-12)
                    << "k=" << k << " l=" << l << " p=" << p;
            }
        }
    }
}

/// Arbitrary (non-basis) outer/inner direction pairs: every output row is the
/// basis tensor contracted with the two seeded directions,
/// `out[(k n + l) n + p] = sum_{a,b} A_k[a] B_l[b] f_{a b p}`.
TEST_F(ThirdOrderTest, ArbitraryDirectionPairsContractBasisTensor) {
    constexpr std::size_t n = 2;
    RevTape& tape = RevTape::active();
    const double xv = 0.7;
    const double yv = 1.3;
    const double dirs[n][2] = {{0.8, -0.6}, {0.3, 0.9}};

    double outerX[n] = {};
    double innerX[n] = {};
    double outerY[n] = {};
    double innerY[n] = {};
    for (std::size_t k = 0; k < n; ++k) {
        outerX[k] = innerX[k] = dirs[k][0];
        outerY[k] = innerY[k] = dirs[k][1];
    }
    const FwdFwdRevScalar<n> x = SeedInput<n>(tape, xv, outerX, innerX);
    const FwdFwdRevScalar<n> y = SeedInput<n>(tape, yv, outerY, innerY);
    const FwdFwdRevScalar<n> f = exp(x * y) + log(x) - 3.0 * sin(y);

    const std::uint32_t inputs[n] = {x.value.value.node, y.value.value.node};
    double out[n * n * n] = {};
    lightape::reverseThird(f, n, inputs, out);

    const ExpLogSinThird d = ExpLogSinDerivs(xv, yv);
    for (std::size_t k = 0; k < n; ++k) {
        for (std::size_t l = 0; l < n; ++l) {
            for (std::size_t p = 0; p < n; ++p) {
                double expected = 0.0;
                for (int a = 0; a < 2; ++a) {
                    for (int b = 0; b < 2; ++b) {
                        expected += dirs[k][static_cast<std::size_t>(a)] *
                                    dirs[l][static_cast<std::size_t>(b)] *
                                    Pick(d, a, b, static_cast<int>(p));
                    }
                }
                EXPECT_NEAR(out[(k * n + l) * n + p], expected, 1e-12)
                    << "k=" << k << " l=" << l << " p=" << p;
            }
        }
    }
}

/// The pure `d3/deps3` along v from the nested reverse recording must agree
/// with the tape-free triple-nested forward jet at the same 1e-12, and with
/// the analytic contraction.
TEST_F(ThirdOrderTest, DirectionalMatchesTripleNestedForward) {
    const double xv = 0.7;
    const double yv = 1.3;
    const double v[2] = {0.8, -0.6};

    // Tape-free oracle: x(t) = x0 + t v, nested three levels deep.
    using D1 = Tangent<double, 1>;
    using D2 = Tangent<D1, 1>;
    using D3 = Tangent<D2, 1>;
    const D3 fx(D2(D1(xv, {v[0]}), {D1(v[0], {0.0})}), {D2(D1(v[0], {0.0}), {D1(0.0, {0.0})})});
    const D3 fy(D2(D1(yv, {v[1]}), {D1(v[1], {0.0})}), {D2(D1(v[1], {0.0}), {D1(0.0, {0.0})})});
    const D3 fj = exp(fx * fy) + log(fx) - 3.0 * sin(fy);
    const double forward = fj.d[0].d[0].d[0];
    const ExpLogSinThird d = ExpLogSinDerivs(xv, yv);
    const double analytic =
        Pick(d, 0, 0, 0) * v[0] * v[0] * v[0] + 3.0 * Pick(d, 0, 0, 1) * v[0] * v[0] * v[1] +
        3.0 * Pick(d, 0, 1, 1) * v[0] * v[1] * v[1] + Pick(d, 1, 1, 1) * v[1] * v[1] * v[1];
    EXPECT_NEAR(forward, analytic, 1e-12);

    // Same path, recorded with both forward levels seeded to v.
    RevTape& tape = RevTape::active();
    const double ax[1] = {v[0]};
    const double ay[1] = {v[1]};
    const FwdFwdRevScalar<1> x = SeedInput<1>(tape, xv, ax, ax);
    const FwdFwdRevScalar<1> y = SeedInput<1>(tape, yv, ay, ay);
    const FwdFwdRevScalar<1> f = exp(x * y) + log(x) - 3.0 * sin(y);
    const std::uint32_t inputs[2] = {x.value.value.node, y.value.value.node};
    EXPECT_NEAR(lightape::thirdDirectional(f, inputs, 2, v), forward, 1e-12);

    // The [0][0] row of the tensor form is the same gradient row.
    double out[1 * 1 * 2] = {};
    lightape::reverseThird(f, 2, inputs, out);
    EXPECT_NEAR(lightape::thirdDirectional(f, inputs, 2, v), out[0] * v[0] + out[1] * v[1], 1e-14);
}

/// f = x^3 + x^2 y + y^3: the third tensor is constant (6, 2, 0, 6) and the
/// nested reverse chain reproduces it exactly.
TEST_F(ThirdOrderTest, CubicPolynomialTensorIsExact) {
    constexpr std::size_t n = 2;
    RevTape& tape = RevTape::active();
    const double basis[2][2] = {{1.0, 0.0}, {0.0, 1.0}};
    const FwdFwdRevScalar<n> x = SeedInput<n>(tape, 0.7, basis[0], basis[0]);
    const FwdFwdRevScalar<n> y = SeedInput<n>(tape, 1.3, basis[1], basis[1]);
    const FwdFwdRevScalar<n> f = x * x * x + x * x * y + y * y * y;

    const std::uint32_t inputs[n] = {x.value.value.node, y.value.value.node};
    double out[n * n * n] = {};
    lightape::reverseThird(f, n, inputs, out);

    // Basis entries: f_xxx = 6, f_xxy = 2, f_xyy = 0, f_yyy = 6.
    const double expected[n][n][n] = {
        {{6.0, 2.0}, {2.0, 0.0}},
        {{2.0, 0.0}, {0.0, 6.0}},
    };
    for (std::size_t k = 0; k < n; ++k) {
        for (std::size_t l = 0; l < n; ++l) {
            for (std::size_t p = 0; p < n; ++p) {
                EXPECT_DOUBLE_EQ(out[(k * n + l) * n + p], expected[k][l][p])
                    << "k=" << k << " l=" << l << " p=" << p;
            }
        }
    }
}

/// A linear function has a node-less mixed component: the tensor helper writes
/// zero rows without sweeping and the directional helper returns exactly zero.
TEST_F(ThirdOrderTest, LinearFunctionWritesZeroRows) {
    constexpr std::size_t n = 2;
    RevTape& tape = RevTape::active();
    const double basis[2][2] = {{1.0, 0.0}, {0.0, 1.0}};
    const FwdFwdRevScalar<n> x = SeedInput<n>(tape, 1.5, basis[0], basis[0]);
    const FwdFwdRevScalar<n> y = SeedInput<n>(tape, -0.5, basis[1], basis[1]);
    const FwdFwdRevScalar<n> f = 2.0 * x + 3.0 * y;

    const std::uint32_t inputs[n] = {x.value.value.node, y.value.value.node};
    double out[n * n * n];
    for (double& value : out) {
        value = -1.0;
    }
    lightape::reverseThird(f, n, inputs, out);
    for (std::size_t k = 0; k < n; ++k) {
        for (std::size_t l = 0; l < n; ++l) {
            for (std::size_t p = 0; p < n; ++p) {
                EXPECT_DOUBLE_EQ(out[(k * n + l) * n + p], 0.0);
            }
        }
    }

    const double v[2] = {0.8, -0.6};
    EXPECT_DOUBLE_EQ(lightape::thirdDirectional(f, inputs, 2, v), 0.0);
}

/// Gamma family: the polygamma chain reaches tetragamma in the mixed nodes,
/// so d3/dx3 lgamma(x) = psi''(x) = tetragamma(x).
TEST_F(ThirdOrderTest, LgammaDirectionalMatchesTetragamma) {
    constexpr std::size_t n = 1;
    RevTape& tape = RevTape::active();
    const double xv = 1.7;
    const double v[1] = {1.3};
    const FwdFwdRevScalar<n> x = SeedInput<n>(tape, xv, v, v);
    const FwdFwdRevScalar<n> f = lgamma(x);

    const std::uint32_t inputs[1] = {x.value.value.node};
    const double got = lightape::thirdDirectional(f, inputs, n, v);
    EXPECT_NEAR(got, v[0] * v[0] * v[0] * lightape::tetragamma(xv), 1e-12);
}

/// Gamma family, mixed argument: f = lgamma(x y) with z = x y has the symmetric
/// basis tensor
///   f_xxx = psi''(z) y^3
///   f_xxy = psi''(z) x y^2 + 2 psi'(z) y
///   f_xyy = psi''(z) x^2 y + 2 psi'(z) x
///   f_yyy = psi''(z) x^3.
TEST_F(ThirdOrderTest, LgammaMixedArgumentTensorMatchesPolygamma) {
    constexpr std::size_t n = 2;
    RevTape& tape = RevTape::active();
    const double xv = 1.4;
    const double yv = 0.9;
    const double basis[2][2] = {{1.0, 0.0}, {0.0, 1.0}};
    const FwdFwdRevScalar<n> x = SeedInput<n>(tape, xv, basis[0], basis[0]);
    const FwdFwdRevScalar<n> y = SeedInput<n>(tape, yv, basis[1], basis[1]);
    const FwdFwdRevScalar<n> f = lgamma(x * y);

    const std::uint32_t inputs[n] = {x.value.value.node, y.value.value.node};
    double out[n * n * n] = {};
    lightape::reverseThird(f, n, inputs, out);

    const double z = xv * yv;
    const double psi1 = lightape::trigamma(z);
    const double psi2 = lightape::tetragamma(z);
    const double fxxx = psi2 * yv * yv * yv;
    const double fxxy = psi2 * xv * yv * yv + 2.0 * psi1 * yv;
    const double fxyy = psi2 * xv * xv * yv + 2.0 * psi1 * xv;
    const double fyyy = psi2 * xv * xv * xv;
    const double expected[n][n][n] = {
        {{fxxx, fxxy}, {fxxy, fxyy}},
        {{fxxy, fxyy}, {fxyy, fyyy}},
    };
    for (std::size_t k = 0; k < n; ++k) {
        for (std::size_t l = 0; l < n; ++l) {
            for (std::size_t p = 0; p < n; ++p) {
                EXPECT_NEAR(out[(k * n + l) * n + p], expected[k][l][p], 1e-12)
                    << "k=" << k << " l=" << l << " p=" << p;
            }
        }
    }
}

} // namespace
