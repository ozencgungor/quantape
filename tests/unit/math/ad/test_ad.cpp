/**
 * @file test_ad.cpp
 * @brief Automatic differentiation walkthrough with Stan Math: interpolation,
 *        integration, and a financial PV-sensitivity example.
 *
 * The former print-only demo is promoted to gates (module_plan_numerics
 * §3.13) with the values recorded from the legacy run; tolerances are the
 * plan's suggested bands.
 */

#include "quantape/math/StanMath.h"

#include "quantape/math/Integrals/IntegratorStanPrimitives.h"
#include "quantape/math/Interpolations.h"
#include "quantape/math/Interpolations/InterpolationStanPrimitives.h"
#include "quantape/math/NumericalMethods.h"
#include "quantape/util/Constants.h"
using ::quantape::util::kPi;

#include <cmath>
#include <string>
#include <utility>
#include <vector>

#include "support/GtestSupport.h"
#include "support/StanTapeFixture.h"

using ADVariableT = stan::math::var;
using namespace quantape::math;

class AdInterpolationTest : public StanTapeTest {};
class AdIntegrationTest : public StanTapeTest {};
class AdFinancialTest : public StanTapeTest {};

TEST_F(AdInterpolationTest, linearValueSlopeAndKnotAdjoints) {
    // Create interpolation data with AD types
    std::vector<ADVariableT> x = {0.0, 1.0, 2.0, 3.0};
    std::vector<ADVariableT> y = {0.0, 1.0, 4.0, 9.0};

    LinearInterpolation<ADVariableT> interp(x, y);

    // Interpolate at x=1.5: value 2.5, slope 3 (segment [1,2])
    ADVariableT x_eval = 1.5;
    ADVariableT y_interp = interp(x_eval);
    CHECK_CLOSE("linear value at 1.5", y_interp.val(), 2.5, 1e-12);

    stan::math::grad(y_interp.vi_);
    CHECK_CLOSE("linear dI/dx", x_eval.adj(), 3.0, 1e-12);

    // Sensitivity to knot values: 0.5 at each side of the active segment.
    std::vector<ADVariableT> x2 = {0.0, 1.0, 2.0, 3.0};
    std::vector<ADVariableT> y2 = {0.0, 1.0, 4.0, 9.0};
    LinearInterpolation<ADVariableT> interp2(x2, y2);

    ADVariableT result = interp2(1.5);
    stan::math::grad(result.vi_);
    CHECK_CLOSE("linear d result/d y[1]", y2[1].adj(), 0.5, 1e-12);
    CHECK_CLOSE("linear d result/d y[2]", y2[2].adj(), 0.5, 1e-12);
}

TEST_F(AdInterpolationTest, cubicSplineValueAndSlope) {
    // Data for x^2 function; natural-BC spline value at 1.5 is 2.2.
    std::vector<ADVariableT> x = {0.0, 1.0, 2.0, 3.0};
    std::vector<ADVariableT> y = {0.0, 1.0, 4.0, 9.0};

    CubicInterpolation<ADVariableT> spline(x, y, CubicDerivativeApprox::Spline);

    ADVariableT x_eval = 1.5;
    ADVariableT y_interp = spline(x_eval);
    CHECK_CLOSE("cubic spline value at 1.5", y_interp.val(), 2.25, 0.1);

    stan::math::grad(y_interp.vi_);
    CHECK_CLOSE("cubic spline dI/dx at 1.5", x_eval.adj(), 3.0, 1e-9);
}

TEST_F(AdInterpolationTest, cubicMethodsMatchSinAtQuarterPi) {
    // Smooth function sin(x) sampled on [0, pi].
    const double expected_value = std::sin(kPi / 4.0);
    const double expected_deriv = std::cos(kPi / 4.0);

    std::vector<std::pair<std::string, typename CubicInterpolation<ADVariableT>::DerivativeApprox>>
        methods = {{"Spline", CubicDerivativeApprox::Spline},
                   {"Parabolic", CubicDerivativeApprox::Parabolic},
                   {"Akima", CubicDerivativeApprox::Akima},
                   {"Kruger", CubicDerivativeApprox::Kruger},
                   {"Harmonic", CubicDerivativeApprox::Harmonic}};

    for (const auto& [name, method] : methods) {
        SCOPED_TRACE(name);

        // Recreate vectors with fresh AD variables for each method.
        std::vector<ADVariableT> x_copy;
        std::vector<ADVariableT> y_copy;
        for (int i = 0; i <= 10; ++i) {
            double xi = i * kPi / 10.0;
            x_copy.push_back(ADVariableT(xi));
            y_copy.push_back(ADVariableT(std::sin(xi)));
        }

        CubicInterpolation<ADVariableT> interp(x_copy, y_copy, method);
        ADVariableT x_test = kPi / 4.0;
        ADVariableT result = interp(x_test);

        stan::math::grad(result.vi_);
        CHECK_CLOSE("cubic method value at pi/4", result.val(), expected_value, 1e-3);
        CHECK_CLOSE("cubic method slope at pi/4", x_test.adj(), expected_deriv, 2e-2);
    }
}

TEST_F(AdInterpolationTest, bilinearPartials) {
    // 2D grid: f(x,y) = x*y
    std::vector<ADVariableT> x = {0.0, 1.0, 2.0};
    std::vector<ADVariableT> y = {0.0, 1.0, 2.0};
    std::vector<std::vector<ADVariableT>> z(3, std::vector<ADVariableT>(3));

    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            z[i][j] = x[j] * y[i];
        }
    }

    BilinearInterpolation<ADVariableT> interp(x, y, z);

    ADVariableT x_eval = 1.5;
    ADVariableT y_eval = 1.5;
    ADVariableT result = interp(x_eval, y_eval);
    CHECK_CLOSE("bilinear value at (1.5,1.5)", result.val(), 2.25, 1e-12);

    stan::math::grad(result.vi_);
    CHECK_CLOSE("bilinear dI/dx", x_eval.adj(), 1.5, 1e-12);
    CHECK_CLOSE("bilinear dI/dy", y_eval.adj(), 1.5, 1e-12);
}

TEST_F(AdIntegrationTest, movingBoundAndParametricIntegrand) {
    // I(theta) = int_0^theta x^2 dx = theta^3/3, dI/dtheta = theta^2 at 2.
    ADVariableT theta = 2.0;
    auto f = [](ADVariableT x) { return x * x; };

    TrapezoidIntegratorDefault<ADVariableT> integrator(1e-6, 1000);
    ADVariableT integral = integrator(f, ADVariableT(0.0), theta);
    CHECK_CLOSE("moving-bound integral", integral.val(), 8.0 / 3.0, 5e-7);

    stan::math::grad(integral.vi_);
    CHECK_CLOSE("moving-bound dI/dtheta", theta.adj(), 4.0, 5e-7);

    // Parametric integral: I(theta) = int_0^1 theta x^2 dx = theta/3.
    ADVariableT param = 3.0;
    auto f_param = [param](ADVariableT x) { return param * x * x; };

    TrapezoidIntegratorDefault<ADVariableT> integrator2(1e-6, 1000);
    ADVariableT integral2 = integrator2(f_param, ADVariableT(0.0), ADVariableT(1.0));
    CHECK_CLOSE("parametric integral", integral2.val(), 1.0, 5e-7);

    stan::math::grad(integral2.vi_);
    CHECK_CLOSE("parametric dI/dtheta", param.adj(), 1.0 / 3.0, 4e-8);
}

TEST_F(AdIntegrationTest, integrateInterpolatedExponential) {
    // Interpolated exp(x) on [0, 2] (11 knots, spacing 0.2).
    std::vector<ADVariableT> x_data;
    std::vector<ADVariableT> y_data;
    for (int i = 0; i <= 10; ++i) {
        double xi = i * 0.2;
        x_data.push_back(ADVariableT(xi));
        y_data.push_back(ADVariableT(std::exp(xi)));
    }

    CubicInterpolation<ADVariableT> spline(x_data, y_data, CubicDerivativeApprox::Spline);

    ADVariableT upper_limit = 1.0;
    auto interpolated_func = [&spline](ADVariableT x) {
        return spline(x, true); // Allow extrapolation
    };

    SimpsonIntegrator<ADVariableT> integrator(1e-6, 1000);
    ADVariableT integral = integrator(interpolated_func, ADVariableT(0.0), upper_limit);

    CHECK_CLOSE("integral of interpolated exp", integral.val(), std::exp(1.0) - 1.0, 3e-4);

    stan::math::grad(integral.vi_);
    CHECK_CLOSE("d/d upper limit of interpolated exp", upper_limit.adj(), std::exp(1.0), 1e-6);
}

TEST_F(AdFinancialTest, discountCurvePvSensitivities) {
    // Discount factors at different maturities (time in years).
    std::vector<ADVariableT> maturities = {0.0, 0.5, 1.0, 2.0, 3.0, 5.0};
    std::vector<ADVariableT> discount_factors = {1.0, 0.98, 0.96, 0.92, 0.88, 0.80};

    CubicInterpolation<ADVariableT> discount_curve(maturities, discount_factors,
                                                   CubicDerivativeApprox::Spline);

    // Cash flow: 100 at t=1.5 years.
    ADVariableT cash_flow_time = 1.5;
    ADVariableT cash_flow_amount = 100.0;

    ADVariableT discount_factor = discount_curve(cash_flow_time);
    ADVariableT present_value = cash_flow_amount * discount_factor;

    CHECK_CLOSE("discount factor at t=1.5", discount_factor.val(), 0.94, 1e-12);
    CHECK_CLOSE("present value", present_value.val(), 94.0, 1e-12);

    stan::math::grad(present_value.vi_);
    // Recorded from the legacy run; the weights sum to 1 (PV = 100 * sum w_i DF_i).
    const double expected_sensitivities[] = {2.608695652173913, -33.91304347826087,
                                             86.84782608695653, 53.26086956521739,
                                             -8.80434782608696, 0.0};
    for (size_t i = 0; i < discount_factors.size(); ++i) {
        CHECK_CLOSE("PV sensitivity to curve knot", discount_factors[i].adj(),
                    expected_sensitivities[i], 1e-12);
    }
}
