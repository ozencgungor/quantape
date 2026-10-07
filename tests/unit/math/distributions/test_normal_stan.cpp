// test_normal_stan.cpp — generic scalar path (Stan var) conformance: the same
// DoubleT-templated distribution types must produce exact gradients.
#include "quantape/math/StanMath.h" // must be first (Eigen plugin ordering)

#include "quantape/math/Distributions/Beta.h"
#include "quantape/math/Distributions/BivariateNormal.h"
#include "quantape/math/Distributions/ChiSquare.h"
#include "quantape/math/Distributions/DistributionsStanPrimitives.h"
#include "quantape/math/Distributions/Exponential.h"
#include "quantape/math/Distributions/Gamma.h"
#include "quantape/math/Distributions/InverseGaussian.h"
#include "quantape/math/Distributions/Lognormal.h"
#include "quantape/math/Distributions/NoncentralBeta.h"
#include "quantape/math/Distributions/NoncentralChiSquare.h"
#include "quantape/math/Distributions/Normal.h"
#include "quantape/math/Distributions/Poisson.h"
#include "quantape/math/Distributions/StudentT.h"

#include <cmath>

#include "support/GtestSupport.h"
#include "support/StanTapeFixture.h"

using stan::math::var;

namespace {

quantape::math::Normal<double> normal64() {
    return quantape::math::Normal<double>{0.3, 1.4};
}

} // namespace

class DistributionAdTest : public StanTapeTest {};

TEST_F(DistributionAdTest, normalCdfGradientMatchesDensity) {
    var mu = 0.3;
    var sigma = 1.4;
    var x = 0.8;
    const auto n = quantape::math::Normal<var>{mu, sigma};
    var y = n.cdf(x);
    y.grad();

    const double pdf = normal64().pdf(0.8);
    CHECK_CLOSE("cdf value", y.val(), normal64().cdf(0.8), 1e-14);
    CHECK_CLOSE_REL("d/dx", x.adj(), pdf, 1e-12);
    CHECK_CLOSE_REL("d/dmu", mu.adj(), -pdf, 1e-12);
    CHECK_CLOSE_REL("d/dsigma", sigma.adj(), -pdf * ((0.8 - 0.3) / 1.4), 1e-12);
}

TEST_F(DistributionAdTest, normalPdfAndLogCdfGradient) {
    var mu = 0.0;
    var sigma = 1.0;
    var x = -1.3;
    const auto n = quantape::math::Normal<var>{mu, sigma};
    var y = n.logCdf(x);
    y.grad();
    const quantape::math::Normal<double> stdNormal{0.0, 1.0};
    const double cdf = stdNormal.cdf(-1.3);
    const double pdf = stdNormal.pdf(-1.3);
    CHECK_CLOSE_REL("logcdf value", y.val(), std::log(cdf), 1e-13);
    CHECK_CLOSE_REL("dlogcdf/dx", x.adj(), pdf / cdf, 1e-10);
}

TEST_F(DistributionAdTest, gammaCdfGradientMatchesDensity) {
    var shape = 2.3;
    var rate = 1.7;
    var x = 1.1;
    // Generic scalar kernel directly: `Gamma<var>::cdf` has an optimized
    // callback specialization now, this keeps the generic path covered.
    var y = quantape::math::detail::gammaCdfT<var>(shape, rate, x);
    y.grad();

    const quantape::math::Gamma<double> g64{2.3, 1.7};
    const double pdf = g64.pdf(1.1);
    CHECK_CLOSE("gamma cdf value", y.val(), g64.cdf(1.1), 1e-13);
    CHECK_CLOSE_REL("gamma d/dx", x.adj(), pdf, 1e-9);
    // cdf = P(shape, rate*x) => d/drate = x * pdf(x) / rate
    CHECK_CLOSE_REL("gamma d/drate", rate.adj(), 1.1 * pdf / 1.7, 1e-9);
}

TEST_F(DistributionAdTest, studentTCdfGradientMatchesDensity) {
    var nu = 4.0;
    var x = 0.9;
    // Generic scalar kernel directly (see the Gamma case above).
    var y = quantape::math::detail::studentTCdf<var>(nu, x);
    y.grad();

    const quantape::math::StudentT<double> t64{4.0};
    const double pdf = t64.pdf(0.9);
    CHECK_CLOSE("student-t cdf value", y.val(), t64.cdf(0.9), 1e-13);
    CHECK_CLOSE_REL("student-t d/dx", x.adj(), pdf, 1e-9);
}

TEST_F(DistributionAdTest, betaCdfGradientMatchesDensity) {
    var a = 2.0;
    var b = 3.5;
    var x = 0.4;
    // Generic scalar kernel directly (see the Gamma case above).
    var y = quantape::math::detail::incBeta<var>(a, b, x);
    y.grad();

    const quantape::math::Beta<double> beta64{2.0, 3.5};
    const double pdf = beta64.pdf(0.4);
    CHECK_CLOSE("beta cdf value", y.val(), beta64.cdf(0.4), 1e-13);
    CHECK_CLOSE_REL("beta d/dx", x.adj(), pdf, 1e-9);
}

TEST_F(DistributionAdTest, noncentralChiSquareCdfGenericPath) {
    // Valid parameters take the optimized pack callback now (the generic tapped
    // mixture is the fallback for the x = 0/invalid boundaries); the value and
    // d/dx conformance against the double kernel is unchanged.
    var nu = 3.0;
    var lambda = 2.0;
    var x = 2.5;
    const auto nc = quantape::math::NoncentralChiSquare<var>{nu, lambda};
    var y = nc.cdf(x);
    y.grad();

    const quantape::math::NoncentralChiSquare<double> nc64{3.0, 2.0};
    CHECK_CLOSE_REL("noncentral chi2 value", y.val(), nc64.cdf(2.5), 1e-12);
    CHECK_CLOSE_REL("noncentral chi2 d/dx", x.adj(), nc64.pdf(2.5), 1e-8);
}

TEST_F(DistributionAdTest, poissonCdfGradientInLambda) {
    var lambda = 3.2;
    const auto p = quantape::math::Poisson<var>{lambda};
    var y = p.cdf(4);
    y.grad();

    const quantape::math::Poisson<double> p64{3.2};
    CHECK_CLOSE("poisson cdf value", y.val(), p64.cdf(4), 1e-14);
    CHECK_CLOSE_REL("poisson d/dlambda", lambda.adj(), -p64.pmf(4), 1e-12);
}

// ============================================================================
// make_callback_var specializations (DistributionsStanPrimitives.h): value and
// every adjoint against the double analytic packs.
// ============================================================================

TEST_F(DistributionAdTest, normalCdfCallbackMatchesAnalyticPack) {
    var mu = 0.3;
    var sigma = 1.4;
    var x = 0.8;
    const quantape::math::Normal<var> n{mu, sigma};
    var y = n.cdf(x);
    y.grad();

    const auto pack = normal64().cdfGradHess(0.8);
    CHECK_CLOSE("callback cdf value", y.val(), pack.value, 0.0);
    CHECK_CLOSE_REL("callback cdf d/dx", x.adj(), pack.gx, 1e-12);
    CHECK_CLOSE_REL("callback cdf d/dmu", mu.adj(), pack.gmu, 1e-12);
    CHECK_CLOSE_REL("callback cdf d/dsigma", sigma.adj(), pack.gsig, 1e-12);
    CHECK_CLOSE_REL("vs stan normal_cdf", y.val(), stan::math::normal_cdf(0.8, 0.3, 1.4), 1e-14);
}

TEST_F(DistributionAdTest, normalPdfCallbackMatchesAnalyticPartials) {
    var mu = 0.3;
    var sigma = 1.4;
    var x = 0.8;
    const quantape::math::Normal<var> n{mu, sigma};
    var y = n.pdf(x);
    y.grad();

    const double pdf = normal64().pdf(0.8);
    const double z = (0.8 - 0.3) / 1.4;
    CHECK_CLOSE("callback pdf value", y.val(), pdf, 0.0);
    CHECK_CLOSE_REL("callback pdf d/dx", x.adj(), -z * pdf / 1.4, 1e-12);
    CHECK_CLOSE_REL("callback pdf d/dmu", mu.adj(), z * pdf / 1.4, 1e-12);
    CHECK_CLOSE_REL("callback pdf d/dsigma", sigma.adj(), pdf * (z * z - 1.0) / 1.4, 1e-12);

    // Independent finite-difference verification of the two parameter formulas.
    const double h = 1e-6;
    CHECK_CLOSE_REL("pdf d/dmu vs FD", mu.adj(),
                    (quantape::math::Normal<double>{0.3 + h, 1.4}.pdf(0.8) -
                     quantape::math::Normal<double>{0.3 - h, 1.4}.pdf(0.8)) /
                        (2.0 * h),
                    1e-7);
    CHECK_CLOSE_REL("pdf d/dsigma vs FD", sigma.adj(),
                    (quantape::math::Normal<double>{0.3, 1.4 + h}.pdf(0.8) -
                     quantape::math::Normal<double>{0.3, 1.4 - h}.pdf(0.8)) /
                        (2.0 * h),
                    1e-7);
}

TEST_F(DistributionAdTest, normalLogCdfCallbackMatchesAnalyticRatios) {
    var mu = 0.3;
    var sigma = 1.4;
    var x = 0.8;
    const quantape::math::Normal<var> n{mu, sigma};
    var y = n.logCdf(x);
    y.grad();

    const double cdf = normal64().cdf(0.8);
    const double pdf = normal64().pdf(0.8);
    const double z = (0.8 - 0.3) / 1.4;
    const double gx = pdf / cdf;
    CHECK_CLOSE("callback logcdf value", y.val(), std::log(cdf), 0.0);
    CHECK_CLOSE_REL("callback logcdf d/dx", x.adj(), gx, 1e-12);
    CHECK_CLOSE_REL("callback logcdf d/dmu", mu.adj(), -gx, 1e-12);
    CHECK_CLOSE_REL("callback logcdf d/dsigma", sigma.adj(), -gx * z, 1e-12);
    CHECK_CLOSE_REL("vs stan normal_lcdf", y.val(), stan::math::normal_lcdf(0.8, 0.3, 1.4), 1e-13);
}

TEST_F(DistributionAdTest, normalCcdfCallbackSignFlipsCdfPack) {
    var mu = 0.3;
    var sigma = 1.4;
    var x = 0.8;
    const quantape::math::Normal<var> n{mu, sigma};
    var y = n.ccdf(x);
    y.grad();

    const auto pack = normal64().cdfGradHess(0.8);
    CHECK_CLOSE("callback ccdf value", y.val(), normal64().ccdf(0.8), 0.0);
    CHECK_CLOSE_REL("callback ccdf d/dx", x.adj(), -pack.gx, 1e-12);
    CHECK_CLOSE_REL("callback ccdf d/dmu", mu.adj(), -pack.gmu, 1e-12);
    CHECK_CLOSE_REL("callback ccdf d/dsigma", sigma.adj(), -pack.gsig, 1e-12);
}

TEST_F(DistributionAdTest, normalLogCcdfCallbackMatchesAnalyticRatios) {
    var mu = 0.3;
    var sigma = 1.4;
    var x = 0.8;
    const quantape::math::Normal<var> n{mu, sigma};
    auto* stack = stan::math::ChainableStack::instance_;
    const std::size_t before = stack->var_stack_.size();
    var y = n.logCcdf(x);
    const std::size_t nodes = stack->var_stack_.size() - before;
    y.grad();

    const double ccdf = normal64().ccdf(0.8);
    const double pdf = normal64().pdf(0.8);
    const double z = (0.8 - 0.3) / 1.4;
    const double gx = -pdf / ccdf; // d/dx log(ccdf)
    CHECK_CLOSE("callback logccdf value", y.val(), std::log(ccdf), 0.0);
    CHECK_CLOSE_REL("callback logccdf d/dx", x.adj(), gx, 1e-12);
    CHECK_CLOSE_REL("callback logccdf d/dmu", mu.adj(), -gx, 1e-12);
    CHECK_CLOSE_REL("callback logccdf d/dsigma", sigma.adj(), -z * gx, 1e-12);
    CHECK_CLOSE_REL("vs stan normal_lccdf", y.val(), stan::math::normal_lccdf(0.8, 0.3, 1.4),
                    1e-13);
    EXPECT_EQ(nodes, std::size_t(1));
    RecordProperty("var_normal_logccdf_nodes", static_cast<int>(nodes));
}

TEST_F(DistributionAdTest, bivariateNormalCdfCallbackMatchesPack) {
    for (double rho0 : {-0.4, 0.0, 0.5}) {
        SCOPED_TRACE(rho0);
        var a = 0.4;
        var b = -0.7;
        var rho = rho0;
        const quantape::math::BivariateNormal<var> bvn{rho};
        var y = bvn.cdf(a, b);
        y.grad();

        const auto pack = quantape::math::detail::bvnCdfGradHess(0.4, -0.7, rho0);
        CHECK_CLOSE("callback bvn value", y.val(), pack.value, 0.0);
        CHECK_CLOSE_REL("callback bvn d/da", a.adj(), pack.ga, 1e-12);
        CHECK_CLOSE_REL("callback bvn d/db", b.adj(), pack.gb, 1e-12);
        CHECK_CLOSE_REL("callback bvn d/drho", rho.adj(), pack.gr, 1e-12);
    }
}

TEST_F(DistributionAdTest, poissonCallbackPmfMatchesLambdaPack) {
    var lambda = 3.2;
    const quantape::math::Poisson<var> p{lambda};
    var y = p.pmf(4);
    y.grad();

    const auto pack = quantape::math::detail::poissonPmfD1D2(3.2, 4);
    CHECK_CLOSE("callback pmf value", y.val(), pack.value, 0.0);
    CHECK_CLOSE_REL("callback pmf d/dlambda", lambda.adj(), pack.d1, 1e-12);
    CHECK_CLOSE_REL("vs stan poisson pmf", y.val(), std::exp(stan::math::poisson_lpmf(4, 3.2)),
                    1e-13);
}

TEST_F(DistributionAdTest, poissonCallbackCdfMatchesLambdaPack) {
    var lambda = 3.2;
    const quantape::math::Poisson<var> p{lambda};
    var y = p.cdf(4);
    y.grad();

    const auto pack = quantape::math::detail::poissonCdfD1D2(3.2, 4);
    CHECK_CLOSE("callback cdf value", y.val(), pack.value, 0.0);
    CHECK_CLOSE_REL("callback cdf d/dlambda", lambda.adj(), pack.d1, 1e-12);
    CHECK_CLOSE_REL("vs stan poisson cdf", y.val(), stan::math::poisson_cdf(4, 3.2), 1e-13);
}

// Node-count evidence: each callback specialization allocates exactly one
// `vari` on the chainable stack; the generic scalar path taps the kernel
// internals. The stack API matches the repo benchmarks
// (`ChainableStack::instance_->var_stack_`).
TEST_F(DistributionAdTest, callbackSpecializationsAddExactlyOneTapeNode) {
    auto* stack = stan::math::ChainableStack::instance_;
    var x = 0.8;
    var mu = 0.3;
    var sigma = 1.4;
    const quantape::math::Normal<var> n{mu, sigma};

    const std::size_t beforeNormal = stack->var_stack_.size();
    var yCdf = n.cdf(x);
    const std::size_t cdfNodes = stack->var_stack_.size() - beforeNormal;
    var yPdf = n.pdf(x);
    const std::size_t pdfNodes = stack->var_stack_.size() - beforeNormal - cdfNodes;
    var yLogCdf = n.logCdf(x);
    const std::size_t logCdfNodes = stack->var_stack_.size() - beforeNormal - cdfNodes - pdfNodes;
    var yCcdf = n.ccdf(x);
    const std::size_t ccdfNodes =
        stack->var_stack_.size() - beforeNormal - cdfNodes - pdfNodes - logCdfNodes;
    EXPECT_EQ(cdfNodes, std::size_t(1));
    EXPECT_EQ(pdfNodes, std::size_t(1));
    EXPECT_EQ(logCdfNodes, std::size_t(1));
    EXPECT_EQ(ccdfNodes, std::size_t(1));

    var a = 0.4;
    var b = -0.7;
    var rho = 0.5;
    const quantape::math::BivariateNormal<var> bvn{rho};
    const std::size_t beforeBvn = stack->var_stack_.size();
    var yBvn = bvn.cdf(a, b);
    EXPECT_EQ(stack->var_stack_.size() - beforeBvn, std::size_t(1));

    var lambda = 3.2;
    const quantape::math::Poisson<var> p{lambda};
    const std::size_t beforePmf = stack->var_stack_.size();
    var yPmf = p.pmf(4);
    const std::size_t pmfNodes = stack->var_stack_.size() - beforePmf;
    var yPcdf = p.cdf(4);
    const std::size_t pcdfNodes = stack->var_stack_.size() - beforePmf - pmfNodes;
    EXPECT_EQ(pmfNodes, std::size_t(1));
    EXPECT_EQ(pcdfNodes, std::size_t(1));

    // Generic path the specialization replaces: many kernel nodes.
    const std::size_t beforeGeneric = stack->var_stack_.size();
    var yGeneric = quantape::math::detail::normalCdfT<var>(x, mu, sigma);
    const std::size_t genericNodes = stack->var_stack_.size() - beforeGeneric;
    EXPECT_GT(genericNodes, std::size_t(1));
    RecordProperty("generic_normal_cdf_nodes", static_cast<int>(genericNodes));

    const std::size_t beforeGenericPmf = stack->var_stack_.size();
    var yGenericPmf = quantape::math::detail::poissonPmfT<var>(lambda, 4);
    const std::size_t genericPmfNodes = stack->var_stack_.size() - beforeGenericPmf;
    EXPECT_GT(genericPmfNodes, std::size_t(1));
    RecordProperty("generic_poisson_pmf_nodes", static_cast<int>(genericPmfNodes));

    const std::size_t beforeGenericBvn = stack->var_stack_.size();
    var yGenericBvn = quantape::math::detail::bvnCdf(a, b, rho);
    const std::size_t genericBvnNodes = stack->var_stack_.size() - beforeGenericBvn;
    EXPECT_GT(genericBvnNodes, std::size_t(1));
    RecordProperty("generic_bvn_cdf_nodes", static_cast<int>(genericBvnNodes));

    (void)yCdf;
    (void)yPdf;
    (void)yLogCdf;
    (void)yCcdf;
    (void)yBvn;
    (void)yPmf;
    (void)yPcdf;
    (void)yGeneric;
    (void)yGenericPmf;
    (void)yGenericBvn;
}

// The callbacks capture the participating `var` handles by value, never
// `this`: a temporary distribution object may die before `.grad()` runs (the
// old `[this]` form would read a reused stack slot and push stale partials).
TEST_F(DistributionAdTest, callbacksSurviveTemporaryDistributionObjects) {
    var x = 0.8;
    var mu = 0.3;
    var sigma = 1.4;
    var yNormal = quantape::math::Normal<var>{mu, sigma}.logCdf(x);
    var a = 0.4;
    var b = -0.7;
    var rho = 0.5;
    var yBvn = quantape::math::BivariateNormal<var>{rho}.cdf(a, b);
    var lambda = 3.2;
    var yPoisson = quantape::math::Poisson<var>{lambda}.pmf(4);

    yNormal.grad();
    const double cdf = normal64().cdf(0.8);
    const double pdf = normal64().pdf(0.8);
    CHECK_CLOSE_REL("temporary normal logcdf d/dx", x.adj(), pdf / cdf, 1e-12);

    yBvn.grad();
    const auto bvnPack = quantape::math::detail::bvnCdfGradHess(0.4, -0.7, 0.5);
    CHECK_CLOSE_REL("temporary bvn d/drho", rho.adj(), bvnPack.gr, 1e-12);

    yPoisson.grad();
    const auto pmfPack = quantape::math::detail::poissonPmfD1D2(3.2, 4);
    CHECK_CLOSE_REL("temporary poisson d/dlambda", lambda.adj(), pmfPack.d1, 1e-12);
}

// Edge semantics: the fallback branches must reproduce the generic path.
TEST_F(DistributionAdTest, edgeCasesFallBackToGenericPath) {
    // Poisson(lambda = 0) is the point mass at 0: generic path returns the
    // exact constant and no lambda sensitivity (the double pack reports the
    // one-sided limits instead).
    var lambda0 = 0.0;
    const quantape::math::Poisson<var> p0{lambda0};
    var y0 = p0.cdf(3);
    y0.grad();
    CHECK_CLOSE("poisson point mass value", y0.val(), 1.0, 0.0);
    CHECK_CLOSE("poisson zero lambda adjoint", lambda0.adj(), 0.0, 0.0);

    // BivariateNormal(rho = 1) is the exact min(a, b) limit: generic path
    // keeps its branch value and one-sided adjoints.
    var a = 0.4;
    var b = -0.7;
    var rho1 = 1.0;
    const quantape::math::BivariateNormal<var> bvn{rho1};
    var y1 = bvn.cdf(a, b);
    y1.grad();
    const quantape::math::Normal<double> stdNormal{0.0, 1.0};
    CHECK_CLOSE("bvn rho=1 value", y1.val(), stdNormal.cdf(-0.7), 0.0);
    CHECK_CLOSE("bvn rho=1 no a sensitivity", a.adj(), 0.0, 0.0);
    CHECK_CLOSE_REL("bvn rho=1 d/db", b.adj(), stdNormal.pdf(-0.7), 1e-14);
    CHECK_CLOSE("bvn rho=1 no rho sensitivity", rho1.adj(), 0.0, 0.0);
}

// ============================================================================
// Second-order (`fvar<var>`) routes: stan::math::hessian over the rebuilt
// distribution must reproduce the analytic Hessian packs. The members and
// arguments are fvar<var>; the value callback mirrors the var route and the
// tangent callback pushes H . w, so the kernels stay off the tape.
// ============================================================================

namespace {

enum class NormalFvarMethod { cdf, pdf, logCdf, ccdf, logCcdf };

struct NormalFvarHessian {
    NormalFvarMethod method;
    template <class T>
    T operator()(const Eigen::Matrix<T, Eigen::Dynamic, 1>& v) const {
        const quantape::math::Normal<T> n{v(1), v(2)};
        switch (method) {
            case NormalFvarMethod::cdf:
                return n.cdf(v(0));
            case NormalFvarMethod::pdf:
                return n.pdf(v(0));
            case NormalFvarMethod::logCdf:
                return n.logCdf(v(0));
            case NormalFvarMethod::ccdf:
                return n.ccdf(v(0));
            case NormalFvarMethod::logCcdf:
                return n.logCcdf(v(0));
        }
        return n.cdf(v(0));
    }
};

/// Expected pack for the specialized route, mirroring the doc'd formulas: the
/// cdf/pdf packs directly (pdf is reshaped to the same 10-field layout), the
/// ccdf pack sign-flipped and the log variants by the quotient rule.
quantape::math::detail::NormalCdfGradHess expectedNormalFvarPack(NormalFvarMethod method, double x0,
                                                                 double mu0, double sigma0) {
    using quantape::math::detail::NormalCdfGradHess;
    using quantape::math::detail::NormalPdfGradHess;
    const NormalCdfGradHess cdf = quantape::math::detail::normalCdfGradHess(x0, mu0, sigma0);
    if (method == NormalFvarMethod::cdf) {
        return cdf;
    }
    if (method == NormalFvarMethod::pdf) {
        const NormalPdfGradHess p = quantape::math::detail::normalPdfGradHess(x0, mu0, sigma0);
        return {p.value, p.gx, p.gmu, p.gsig, p.hxx, p.hxmu, p.hxsig, p.hmumu, p.hmusig, p.hsigsig};
    }
    if (method == NormalFvarMethod::ccdf) {
        NormalCdfGradHess out = cdf;
        out.value = quantape::math::detail::normalCcdfDouble(x0, mu0, sigma0);
        out.gx = -cdf.gx;
        out.gmu = -cdf.gmu;
        out.gsig = -cdf.gsig;
        out.hxx = -cdf.hxx;
        out.hxmu = -cdf.hxmu;
        out.hxsig = -cdf.hxsig;
        out.hmumu = -cdf.hmumu;
        out.hmusig = -cdf.hmusig;
        out.hsigsig = -cdf.hsigsig;
        return out;
    }
    if (method == NormalFvarMethod::logCdf) {
        const double invF = 1.0 / cdf.value;
        NormalCdfGradHess out{};
        out.value = std::log(cdf.value);
        out.gx = cdf.gx * invF;
        out.gmu = cdf.gmu * invF;
        out.gsig = cdf.gsig * invF;
        out.hxx = cdf.hxx * invF - out.gx * out.gx;
        out.hxmu = cdf.hxmu * invF - out.gx * out.gmu;
        out.hxsig = cdf.hxsig * invF - out.gx * out.gsig;
        out.hmumu = cdf.hmumu * invF - out.gmu * out.gmu;
        out.hmusig = cdf.hmusig * invF - out.gmu * out.gsig;
        out.hsigsig = cdf.hsigsig * invF - out.gsig * out.gsig;
        return out;
    }
    const double ccdf = quantape::math::detail::normalCcdfDouble(x0, mu0, sigma0);
    const double invG = 1.0 / ccdf;
    NormalCdfGradHess out{};
    out.value = std::log(ccdf);
    out.gx = -cdf.gx * invG;
    out.gmu = -cdf.gmu * invG;
    out.gsig = -cdf.gsig * invG;
    out.hxx = -cdf.hxx * invG - out.gx * out.gx;
    out.hxmu = -cdf.hxmu * invG - out.gx * out.gmu;
    out.hxsig = -cdf.hxsig * invG - out.gx * out.gsig;
    out.hmumu = -cdf.hmumu * invG - out.gmu * out.gmu;
    out.hmusig = -cdf.hmusig * invG - out.gmu * out.gsig;
    out.hsigsig = -cdf.hsigsig * invG - out.gsig * out.gsig;
    return out;
}

/// Compares a hessian() result (grad and symmetric Hessian) against a pack.
void checkNormalFvarHessian(const char* methodName, const Eigen::VectorXd& grad,
                            const Eigen::MatrixXd& hessian,
                            const quantape::math::detail::NormalCdfGradHess& pack) {
    const double expectedGrad[3] = {pack.gx, pack.gmu, pack.gsig};
    const double expectedHess[3][3] = {{pack.hxx, pack.hxmu, pack.hxsig},
                                       {pack.hxmu, pack.hmumu, pack.hmusig},
                                       {pack.hxsig, pack.hmusig, pack.hsigsig}};
    for (int i = 0; i < 3; ++i) {
        CHECK_CLOSE_REL(std::string(methodName) + " grad " + std::to_string(i), grad(i),
                        expectedGrad[i], 1e-9, 1e-12);
        for (int j = 0; j < 3; ++j) {
            CHECK_CLOSE_REL(std::string(methodName) + " hessian " + std::to_string(i) +
                                std::to_string(j),
                            hessian(i, j), expectedHess[i][j], 1e-9, 1e-12);
        }
    }
}

struct BvnFvarHessian {
    template <class T>
    T operator()(const Eigen::Matrix<T, Eigen::Dynamic, 1>& v) const {
        const quantape::math::BivariateNormal<T> bvn{v(2)};
        return bvn.cdf(v(0), v(1));
    }
};

struct PoissonFvarHessian {
    bool useCdf;
    int k;
    template <class T>
    T operator()(const Eigen::Matrix<T, Eigen::Dynamic, 1>& v) const {
        const quantape::math::Poisson<T> p{v(0)};
        return useCdf ? p.cdf(k) : p.pmf(k);
    }
};

} // namespace

TEST_F(DistributionAdTest, normalFvarHessianMatchesAnalyticPacks) {
    const double points[3][3] = {{0.8, 0.3, 1.4}, {-1.7, -0.4, 2.1}, {2.3, 0.0, 0.7}};
    const NormalFvarMethod methods[5] = {NormalFvarMethod::cdf, NormalFvarMethod::pdf,
                                         NormalFvarMethod::logCdf, NormalFvarMethod::ccdf,
                                         NormalFvarMethod::logCcdf};
    for (const auto& p : points) {
        SCOPED_TRACE(p[0]);
        for (NormalFvarMethod method : methods) {
            Eigen::VectorXd point(3);
            point << p[0], p[1], p[2];
            double fx = 0.0;
            Eigen::VectorXd grad(3);
            Eigen::MatrixXd hessian(3, 3);
            stan::math::hessian(NormalFvarHessian{method}, point, fx, grad, hessian);

            const auto pack = expectedNormalFvarPack(method, p[0], p[1], p[2]);
            CHECK_CLOSE("normal fvar value", fx, pack.value, 1e-14);
            checkNormalFvarHessian("normal fvar", grad, hessian, pack);
        }
    }
}

TEST_F(DistributionAdTest, bivariateNormalFvarHessianMatchesPack) {
    const double points[2][2] = {{0.4, -0.7}, {-1.2, 0.6}};
    for (double rho0 : {-0.4, 0.0, 0.5}) {
        for (const auto& p : points) {
            SCOPED_TRACE(rho0);
            Eigen::VectorXd point(3);
            point << p[0], p[1], rho0;
            double fx = 0.0;
            Eigen::VectorXd grad(3);
            Eigen::MatrixXd hessian(3, 3);
            stan::math::hessian(BvnFvarHessian{}, point, fx, grad, hessian);

            const auto pack = quantape::math::detail::bvnCdfGradHess(p[0], p[1], rho0);
            CHECK_CLOSE("bvn fvar value", fx, pack.value, 1e-14);
            const double expectedGrad[3] = {pack.ga, pack.gb, pack.gr};
            const double expectedHess[3][3] = {{pack.haa, pack.hab, pack.har},
                                               {pack.hab, pack.hbb, pack.hbr},
                                               {pack.har, pack.hbr, pack.hrr}};
            for (int i = 0; i < 3; ++i) {
                CHECK_CLOSE_REL(std::string("bvn fvar grad ") + std::to_string(i), grad(i),
                                expectedGrad[i], 1e-9, 1e-12);
                for (int j = 0; j < 3; ++j) {
                    CHECK_CLOSE_REL(std::string("bvn fvar hessian ") + std::to_string(i) +
                                        std::to_string(j),
                                    hessian(i, j), expectedHess[i][j], 1e-9, 1e-12);
                }
            }
        }
    }
}

TEST_F(DistributionAdTest, poissonFvarHessianMatchesLambdaPack) {
    for (double lambda0 : {0.7, 3.2}) {
        for (int k : {0, 4}) {
            for (bool useCdf : {false, true}) {
                SCOPED_TRACE(lambda0);
                Eigen::VectorXd point(1);
                point << lambda0;
                double fx = 0.0;
                Eigen::VectorXd grad(1);
                Eigen::MatrixXd hessian(1, 1);
                stan::math::hessian(PoissonFvarHessian{useCdf, k}, point, fx, grad, hessian);

                const auto pack = useCdf ? quantape::math::detail::poissonCdfD1D2(lambda0, k)
                                         : quantape::math::detail::poissonPmfD1D2(lambda0, k);
                CHECK_CLOSE("poisson fvar value", fx, pack.value, 1e-14);
                CHECK_CLOSE_REL("poisson fvar d/dlambda", grad(0), pack.d1, 1e-9, 1e-12);
                CHECK_CLOSE_REL("poisson fvar d2/dlambda2", hessian(0, 0), pack.d2, 1e-9, 1e-12);
            }
        }
    }
}

// Node-count evidence: each `fvar<var>` specialization allocates exactly two
// callback nodes (value + tangent), never taping the kernel (compare with the
// var-tier one-node counts above and the generic multi-node path).
TEST_F(DistributionAdTest, fvarCallbacksAddExactlyTwoTapeNodes) {
    using fvar = stan::math::fvar<var>;
    auto* stack = stan::math::ChainableStack::instance_;

    fvar x(0.8, 1.0), mu(0.3, 0.0), sigma(1.4, 0.0);
    const quantape::math::Normal<fvar> n{mu, sigma};

    std::size_t before = stack->var_stack_.size();
    fvar yCdf = n.cdf(x);
    const std::size_t cdfNodes = stack->var_stack_.size() - before;
    before = stack->var_stack_.size();
    fvar yPdf = n.pdf(x);
    const std::size_t pdfNodes = stack->var_stack_.size() - before;
    before = stack->var_stack_.size();
    fvar yLogCdf = n.logCdf(x);
    const std::size_t logCdfNodes = stack->var_stack_.size() - before;
    before = stack->var_stack_.size();
    fvar yCcdf = n.ccdf(x);
    const std::size_t ccdfNodes = stack->var_stack_.size() - before;
    before = stack->var_stack_.size();
    fvar yLogCcdf = n.logCcdf(x);
    const std::size_t logCcdfNodes = stack->var_stack_.size() - before;
    EXPECT_EQ(cdfNodes, std::size_t(2));
    EXPECT_EQ(pdfNodes, std::size_t(2));
    EXPECT_EQ(logCdfNodes, std::size_t(2));
    EXPECT_EQ(ccdfNodes, std::size_t(2));
    EXPECT_EQ(logCcdfNodes, std::size_t(2));
    RecordProperty("fvar_normal_cdf_nodes", static_cast<int>(cdfNodes));
    RecordProperty("fvar_normal_pdf_nodes", static_cast<int>(pdfNodes));
    RecordProperty("fvar_normal_logcdf_nodes", static_cast<int>(logCdfNodes));
    RecordProperty("fvar_normal_ccdf_nodes", static_cast<int>(ccdfNodes));
    RecordProperty("fvar_normal_logccdf_nodes", static_cast<int>(logCcdfNodes));

    fvar a(0.4, 1.0), b(-0.7, 0.0), rho(0.5, 0.0);
    const quantape::math::BivariateNormal<fvar> bvn{rho};
    before = stack->var_stack_.size();
    fvar yBvn = bvn.cdf(a, b);
    const std::size_t bvnNodes = stack->var_stack_.size() - before;
    EXPECT_EQ(bvnNodes, std::size_t(2));
    RecordProperty("fvar_bvn_cdf_nodes", static_cast<int>(bvnNodes));

    fvar lambda(3.2, 1.0);
    const quantape::math::Poisson<fvar> p{lambda};
    before = stack->var_stack_.size();
    fvar yPmf = p.pmf(4);
    const std::size_t pmfNodes = stack->var_stack_.size() - before;
    before = stack->var_stack_.size();
    fvar yPcdf = p.cdf(4);
    const std::size_t pcdfNodes = stack->var_stack_.size() - before;
    EXPECT_EQ(pmfNodes, std::size_t(2));
    EXPECT_EQ(pcdfNodes, std::size_t(2));
    RecordProperty("fvar_poisson_pmf_nodes", static_cast<int>(pmfNodes));
    RecordProperty("fvar_poisson_cdf_nodes", static_cast<int>(pcdfNodes));

    (void)yCdf;
    (void)yPdf;
    (void)yLogCdf;
    (void)yCcdf;
    (void)yLogCcdf;
    (void)yBvn;
    (void)yPmf;
    (void)yPcdf;
}

// ============================================================================
// Family callback routes (DistributionsStanPrimitives.h): the 1-node `var`
// and 2-node `fvar<var>` specializations for Gamma, InverseGamma, ChiSquare,
// Beta, StudentT, Lognormal, Exponential and InverseGaussian. Values and every
// adjoint are checked against the double analytic packs; the second-order
// routes are checked through `stan::math::hessian`; node counts are recorded.
// ============================================================================

namespace {

/// Value + adjoints of a `var` route against an analytic pack.
void checkVarRoute(const char* label, double value, std::initializer_list<double> adjoints,
                   double packValue, std::initializer_list<double> expected) {
    CHECK_CLOSE(std::string(label) + " value", value, packValue, 0.0);
    ASSERT_EQ(adjoints.size(), expected.size());
    auto got = adjoints.begin();
    auto want = expected.begin();
    std::size_t i = 0;
    for (; got != adjoints.end(); ++got, ++want, ++i) {
        CHECK_CLOSE_REL(std::string(label) + " adj " + std::to_string(i), *got, *want, 1e-12,
                        1e-14);
    }
}

/// Value + gradient + symmetric Hessian of a `hessian()` result against an
/// analytic pack (row-major `h`, `g.size()` active leaves).
void checkPackRoute(const char* label, double value, const Eigen::VectorXd& grad,
                    const Eigen::MatrixXd& hess, double packValue, std::initializer_list<double> g,
                    std::initializer_list<double> h) {
    CHECK_CLOSE(std::string(label) + " value", value, packValue, 0.0);
    const std::size_t n = g.size();
    ASSERT_EQ(h.size(), n * n);
    std::size_t i = 0;
    for (double gi : g) {
        CHECK_CLOSE_REL(std::string(label) + " grad " + std::to_string(i),
                        grad(static_cast<Eigen::Index>(i)), gi, 1e-9, 1e-12);
        ++i;
    }
    i = 0;
    for (double hij : h) {
        CHECK_CLOSE_REL(std::string(label) + " hessian " + std::to_string(i),
                        hess(static_cast<Eigen::Index>(i / n), static_cast<Eigen::Index>(i % n)),
                        hij, 1e-9, 1e-12);
        ++i;
    }
}

struct GammaFvarHessian {
    bool useCdf;
    template <class T>
    T operator()(const Eigen::Matrix<T, Eigen::Dynamic, 1>& v) const {
        const quantape::math::Gamma<T> g{v(0), v(1)};
        return useCdf ? g.cdf(v(2)) : g.pdf(v(2));
    }
};

struct InverseGammaFvarHessian {
    bool useCdf;
    template <class T>
    T operator()(const Eigen::Matrix<T, Eigen::Dynamic, 1>& v) const {
        const quantape::math::InverseGamma<T> g{v(0), v(1)};
        return useCdf ? g.cdf(v(2)) : g.pdf(v(2));
    }
};

struct ChiSquareFvarHessian {
    bool useCdf;
    template <class T>
    T operator()(const Eigen::Matrix<T, Eigen::Dynamic, 1>& v) const {
        const quantape::math::ChiSquare<T> g{v(0)};
        return useCdf ? g.cdf(v(1)) : g.pdf(v(1));
    }
};

struct BetaFvarHessian {
    bool useCdf;
    template <class T>
    T operator()(const Eigen::Matrix<T, Eigen::Dynamic, 1>& v) const {
        const quantape::math::Beta<T> g{v(0), v(1)};
        return useCdf ? g.cdf(v(2)) : g.pdf(v(2));
    }
};

struct StudentTFvarHessian {
    bool useCdf;
    template <class T>
    T operator()(const Eigen::Matrix<T, Eigen::Dynamic, 1>& v) const {
        const quantape::math::StudentT<T> g{v(0)};
        return useCdf ? g.cdf(v(1)) : g.pdf(v(1));
    }
};

struct LognormalFvarHessian {
    bool useCdf;
    template <class T>
    T operator()(const Eigen::Matrix<T, Eigen::Dynamic, 1>& v) const {
        const quantape::math::Lognormal<T> g{v(0), v(1)};
        return useCdf ? g.cdf(v(2)) : g.pdf(v(2));
    }
};

struct ExponentialFvarHessian {
    bool useCdf;
    template <class T>
    T operator()(const Eigen::Matrix<T, Eigen::Dynamic, 1>& v) const {
        const quantape::math::Exponential<T> g{v(0)};
        return useCdf ? g.cdf(v(1)) : g.pdf(v(1));
    }
};

struct InverseGaussianFvarHessian {
    bool useCdf;
    template <class T>
    T operator()(const Eigen::Matrix<T, Eigen::Dynamic, 1>& v) const {
        const quantape::math::InverseGaussian<T> g{v(0), v(1)};
        return useCdf ? g.cdf(v(2)) : g.pdf(v(2));
    }
};

} // namespace

TEST_F(DistributionAdTest, gammaCallbackVarMatchesAnalyticPack) {
    {
        var shape = 2.3;
        var rate = 1.7;
        var x = 1.1;
        const quantape::math::Gamma<var> g{shape, rate};
        auto* stack = stan::math::ChainableStack::instance_;
        const std::size_t before = stack->var_stack_.size();
        var y = g.cdf(x);
        const std::size_t nodes = stack->var_stack_.size() - before;
        y.grad();
        const auto pack = quantape::math::detail::gammaCdfGradHess(2.3, 1.7, 1.1);
        checkVarRoute("gamma cdf", y.val(), {shape.adj(), rate.adj(), x.adj()}, pack.value,
                      {pack.gShape, pack.gRate, pack.gx});
        EXPECT_EQ(nodes, std::size_t(1));
        RecordProperty("var_gamma_cdf_nodes", static_cast<int>(nodes));
    }
    {
        var shape = 2.3;
        var rate = 1.7;
        var x = 1.1;
        const quantape::math::Gamma<var> g{shape, rate};
        auto* stack = stan::math::ChainableStack::instance_;
        const std::size_t before = stack->var_stack_.size();
        var y = g.pdf(x);
        const std::size_t nodes = stack->var_stack_.size() - before;
        y.grad();
        const auto pack = quantape::math::detail::gammaPdfGradHess(2.3, 1.7, 1.1);
        checkVarRoute("gamma pdf", y.val(), {shape.adj(), rate.adj(), x.adj()}, pack.value,
                      {pack.gShape, pack.gRate, pack.gx});
        EXPECT_EQ(nodes, std::size_t(1));
        RecordProperty("var_gamma_pdf_nodes", static_cast<int>(nodes));
    }
}

TEST_F(DistributionAdTest, gammaCallbackFvarHessianMatchesAnalyticPack) {
    using fvar = stan::math::fvar<stan::math::var>;
    const double shape0 = 2.3, rate0 = 1.7, x0 = 1.1;
    Eigen::VectorXd point(3);
    point << shape0, rate0, x0;
    for (bool useCdf : {true, false}) {
        SCOPED_TRACE(useCdf);
        double fx = 0.0;
        Eigen::VectorXd grad(3);
        Eigen::MatrixXd hess(3, 3);
        stan::math::hessian(GammaFvarHessian{useCdf}, point, fx, grad, hess);
        if (useCdf) {
            const auto pack = quantape::math::detail::gammaCdfGradHess(shape0, rate0, x0);
            checkPackRoute("gamma fvar cdf", fx, grad, hess, pack.value,
                           {pack.gShape, pack.gRate, pack.gx},
                           {pack.hShapeShape, pack.hShapeRate, pack.hShapeX, pack.hShapeRate,
                            pack.hRateRate, pack.hRateX, pack.hShapeX, pack.hRateX, pack.hxx});
        } else {
            const auto pack = quantape::math::detail::gammaPdfGradHess(shape0, rate0, x0);
            checkPackRoute("gamma fvar pdf", fx, grad, hess, pack.value,
                           {pack.gShape, pack.gRate, pack.gx},
                           {pack.hShapeShape, pack.hShapeRate, pack.hShapeX, pack.hShapeRate,
                            pack.hRateRate, pack.hRateX, pack.hShapeX, pack.hRateX, pack.hxx});
        }
    }
    {
        fvar shape(shape0, 1.0), rate(rate0, 0.0), x(x0, 0.0);
        const quantape::math::Gamma<fvar> g{shape, rate};
        auto* stack = stan::math::ChainableStack::instance_;
        const std::size_t before = stack->var_stack_.size();
        fvar y = g.cdf(x);
        const std::size_t nodes = stack->var_stack_.size() - before;
        EXPECT_EQ(nodes, std::size_t(2));
        RecordProperty("fvar_gamma_cdf_nodes", static_cast<int>(nodes));
        const std::size_t beforePdf = stack->var_stack_.size();
        fvar yPdf = g.pdf(x);
        const std::size_t pdfNodes = stack->var_stack_.size() - beforePdf;
        EXPECT_EQ(pdfNodes, std::size_t(2));
        RecordProperty("fvar_gamma_pdf_nodes", static_cast<int>(pdfNodes));
        (void)y;
        (void)yPdf;
    }
}

TEST_F(DistributionAdTest, inverseGammaCallbackVarMatchesAnalyticPack) {
    {
        var shape = 2.3;
        var scale = 1.7;
        var x = 1.4;
        const quantape::math::InverseGamma<var> g{shape, scale};
        auto* stack = stan::math::ChainableStack::instance_;
        const std::size_t before = stack->var_stack_.size();
        var y = g.cdf(x);
        const std::size_t nodes = stack->var_stack_.size() - before;
        y.grad();
        const auto pack = quantape::math::detail::inverseGammaCdfGradHess(2.3, 1.7, 1.4);
        checkVarRoute("invgamma cdf", y.val(), {shape.adj(), scale.adj(), x.adj()}, pack.value,
                      {pack.gShape, pack.gScale, pack.gx});
        EXPECT_EQ(nodes, std::size_t(1));
        RecordProperty("var_invgamma_cdf_nodes", static_cast<int>(nodes));
    }
    {
        var shape = 2.3;
        var scale = 1.7;
        var x = 1.4;
        const quantape::math::InverseGamma<var> g{shape, scale};
        auto* stack = stan::math::ChainableStack::instance_;
        const std::size_t before = stack->var_stack_.size();
        var y = g.pdf(x);
        const std::size_t nodes = stack->var_stack_.size() - before;
        y.grad();
        const auto pack = quantape::math::detail::inverseGammaPdfGradHess(2.3, 1.7, 1.4);
        checkVarRoute("invgamma pdf", y.val(), {shape.adj(), scale.adj(), x.adj()}, pack.value,
                      {pack.gShape, pack.gScale, pack.gx});
        EXPECT_EQ(nodes, std::size_t(1));
        RecordProperty("var_invgamma_pdf_nodes", static_cast<int>(nodes));
    }
}

TEST_F(DistributionAdTest, inverseGammaCallbackFvarHessianMatchesAnalyticPack) {
    using fvar = stan::math::fvar<stan::math::var>;
    const double shape0 = 2.3, scale0 = 1.7, x0 = 1.4;
    Eigen::VectorXd point(3);
    point << shape0, scale0, x0;
    for (bool useCdf : {true, false}) {
        SCOPED_TRACE(useCdf);
        double fx = 0.0;
        Eigen::VectorXd grad(3);
        Eigen::MatrixXd hess(3, 3);
        stan::math::hessian(InverseGammaFvarHessian{useCdf}, point, fx, grad, hess);
        if (useCdf) {
            const auto pack = quantape::math::detail::inverseGammaCdfGradHess(shape0, scale0, x0);
            checkPackRoute("invgamma fvar cdf", fx, grad, hess, pack.value,
                           {pack.gShape, pack.gScale, pack.gx},
                           {pack.hShapeShape, pack.hShapeScale, pack.hShapeX, pack.hShapeScale,
                            pack.hScaleScale, pack.hScaleX, pack.hShapeX, pack.hScaleX, pack.hxx});
        } else {
            const auto pack = quantape::math::detail::inverseGammaPdfGradHess(shape0, scale0, x0);
            checkPackRoute("invgamma fvar pdf", fx, grad, hess, pack.value,
                           {pack.gShape, pack.gScale, pack.gx},
                           {pack.hShapeShape, pack.hShapeScale, pack.hShapeX, pack.hShapeScale,
                            pack.hScaleScale, pack.hScaleX, pack.hShapeX, pack.hScaleX, pack.hxx});
        }
    }
    {
        fvar shape(shape0, 1.0), scale(scale0, 0.0), x(x0, 0.0);
        const quantape::math::InverseGamma<fvar> g{shape, scale};
        auto* stack = stan::math::ChainableStack::instance_;
        const std::size_t before = stack->var_stack_.size();
        fvar y = g.cdf(x);
        const std::size_t nodes = stack->var_stack_.size() - before;
        EXPECT_EQ(nodes, std::size_t(2));
        RecordProperty("fvar_invgamma_cdf_nodes", static_cast<int>(nodes));
        const std::size_t beforePdf = stack->var_stack_.size();
        fvar yPdf = g.pdf(x);
        const std::size_t pdfNodes = stack->var_stack_.size() - beforePdf;
        EXPECT_EQ(pdfNodes, std::size_t(2));
        RecordProperty("fvar_invgamma_pdf_nodes", static_cast<int>(pdfNodes));
        (void)y;
        (void)yPdf;
    }
}

TEST_F(DistributionAdTest, chiSquareCallbackVarMatchesAnalyticPack) {
    {
        var df = 4.5;
        var x = 2.7;
        const quantape::math::ChiSquare<var> g{df};
        auto* stack = stan::math::ChainableStack::instance_;
        const std::size_t before = stack->var_stack_.size();
        var y = g.cdf(x);
        const std::size_t nodes = stack->var_stack_.size() - before;
        y.grad();
        const auto pack = quantape::math::detail::chiSquareCdfGradHess(4.5, 2.7);
        checkVarRoute("chi2 cdf", y.val(), {df.adj(), x.adj()}, pack.value, {pack.gDf, pack.gx});
        EXPECT_EQ(nodes, std::size_t(1));
        RecordProperty("var_chi2_cdf_nodes", static_cast<int>(nodes));
    }
    {
        var df = 4.5;
        var x = 2.7;
        const quantape::math::ChiSquare<var> g{df};
        auto* stack = stan::math::ChainableStack::instance_;
        const std::size_t before = stack->var_stack_.size();
        var y = g.pdf(x);
        const std::size_t nodes = stack->var_stack_.size() - before;
        y.grad();
        const auto pack = quantape::math::detail::chiSquarePdfGradHess(4.5, 2.7);
        checkVarRoute("chi2 pdf", y.val(), {df.adj(), x.adj()}, pack.value, {pack.gDf, pack.gx});
        EXPECT_EQ(nodes, std::size_t(1));
        RecordProperty("var_chi2_pdf_nodes", static_cast<int>(nodes));
    }
}

TEST_F(DistributionAdTest, chiSquareCallbackFvarHessianMatchesAnalyticPack) {
    using fvar = stan::math::fvar<stan::math::var>;
    const double df0 = 4.5, x0 = 2.7;
    Eigen::VectorXd point(2);
    point << df0, x0;
    for (bool useCdf : {true, false}) {
        SCOPED_TRACE(useCdf);
        double fx = 0.0;
        Eigen::VectorXd grad(2);
        Eigen::MatrixXd hess(2, 2);
        stan::math::hessian(ChiSquareFvarHessian{useCdf}, point, fx, grad, hess);
        if (useCdf) {
            const auto pack = quantape::math::detail::chiSquareCdfGradHess(df0, x0);
            checkPackRoute("chi2 fvar cdf", fx, grad, hess, pack.value, {pack.gDf, pack.gx},
                           {pack.hDfDf, pack.hDfX, pack.hDfX, pack.hxx});
        } else {
            const auto pack = quantape::math::detail::chiSquarePdfGradHess(df0, x0);
            checkPackRoute("chi2 fvar pdf", fx, grad, hess, pack.value, {pack.gDf, pack.gx},
                           {pack.hDfDf, pack.hDfX, pack.hDfX, pack.hxx});
        }
    }
    {
        fvar df(df0, 1.0), x(x0, 0.0);
        const quantape::math::ChiSquare<fvar> g{df};
        auto* stack = stan::math::ChainableStack::instance_;
        const std::size_t before = stack->var_stack_.size();
        fvar y = g.cdf(x);
        const std::size_t nodes = stack->var_stack_.size() - before;
        EXPECT_EQ(nodes, std::size_t(2));
        RecordProperty("fvar_chi2_cdf_nodes", static_cast<int>(nodes));
        const std::size_t beforePdf = stack->var_stack_.size();
        fvar yPdf = g.pdf(x);
        const std::size_t pdfNodes = stack->var_stack_.size() - beforePdf;
        EXPECT_EQ(pdfNodes, std::size_t(2));
        RecordProperty("fvar_chi2_pdf_nodes", static_cast<int>(pdfNodes));
        (void)y;
        (void)yPdf;
    }
}

TEST_F(DistributionAdTest, betaCallbackVarMatchesAnalyticPack) {
    {
        var alpha = 2.0;
        var beta = 3.5;
        var x = 0.4;
        const quantape::math::Beta<var> g{alpha, beta};
        auto* stack = stan::math::ChainableStack::instance_;
        const std::size_t before = stack->var_stack_.size();
        var y = g.cdf(x);
        const std::size_t nodes = stack->var_stack_.size() - before;
        y.grad();
        const auto pack = quantape::math::detail::betaCdfGradHess(2.0, 3.5, 0.4);
        checkVarRoute("beta cdf", y.val(), {alpha.adj(), beta.adj(), x.adj()}, pack.value,
                      {pack.gAlpha, pack.gBeta, pack.gx});
        EXPECT_EQ(nodes, std::size_t(1));
        RecordProperty("var_beta_cdf_nodes", static_cast<int>(nodes));
    }
    {
        var alpha = 2.0;
        var beta = 3.5;
        var x = 0.4;
        const quantape::math::Beta<var> g{alpha, beta};
        auto* stack = stan::math::ChainableStack::instance_;
        const std::size_t before = stack->var_stack_.size();
        var y = g.pdf(x);
        const std::size_t nodes = stack->var_stack_.size() - before;
        y.grad();
        const auto pack = quantape::math::detail::betaPdfGradHess(2.0, 3.5, 0.4);
        checkVarRoute("beta pdf", y.val(), {alpha.adj(), beta.adj(), x.adj()}, pack.value,
                      {pack.gAlpha, pack.gBeta, pack.gx});
        EXPECT_EQ(nodes, std::size_t(1));
        RecordProperty("var_beta_pdf_nodes", static_cast<int>(nodes));
    }
}

TEST_F(DistributionAdTest, betaCallbackFvarHessianMatchesAnalyticPack) {
    using fvar = stan::math::fvar<stan::math::var>;
    const double alpha0 = 2.0, beta0 = 3.5, x0 = 0.4;
    Eigen::VectorXd point(3);
    point << alpha0, beta0, x0;
    for (bool useCdf : {true, false}) {
        SCOPED_TRACE(useCdf);
        double fx = 0.0;
        Eigen::VectorXd grad(3);
        Eigen::MatrixXd hess(3, 3);
        stan::math::hessian(BetaFvarHessian{useCdf}, point, fx, grad, hess);
        if (useCdf) {
            const auto pack = quantape::math::detail::betaCdfGradHess(alpha0, beta0, x0);
            checkPackRoute("beta fvar cdf", fx, grad, hess, pack.value,
                           {pack.gAlpha, pack.gBeta, pack.gx},
                           {pack.hAA, pack.hAB, pack.hAx, pack.hAB, pack.hBB, pack.hBx, pack.hAx,
                            pack.hBx, pack.hxx});
        } else {
            const auto pack = quantape::math::detail::betaPdfGradHess(alpha0, beta0, x0);
            checkPackRoute("beta fvar pdf", fx, grad, hess, pack.value,
                           {pack.gAlpha, pack.gBeta, pack.gx},
                           {pack.hAA, pack.hAB, pack.hAx, pack.hAB, pack.hBB, pack.hBx, pack.hAx,
                            pack.hBx, pack.hxx});
        }
    }
    {
        fvar alpha(alpha0, 1.0), beta(beta0, 0.0), x(x0, 0.0);
        const quantape::math::Beta<fvar> g{alpha, beta};
        auto* stack = stan::math::ChainableStack::instance_;
        const std::size_t before = stack->var_stack_.size();
        fvar y = g.cdf(x);
        const std::size_t nodes = stack->var_stack_.size() - before;
        EXPECT_EQ(nodes, std::size_t(2));
        RecordProperty("fvar_beta_cdf_nodes", static_cast<int>(nodes));
        const std::size_t beforePdf = stack->var_stack_.size();
        fvar yPdf = g.pdf(x);
        const std::size_t pdfNodes = stack->var_stack_.size() - beforePdf;
        EXPECT_EQ(pdfNodes, std::size_t(2));
        RecordProperty("fvar_beta_pdf_nodes", static_cast<int>(pdfNodes));
        (void)y;
        (void)yPdf;
    }
}

TEST_F(DistributionAdTest, studentTCallbackVarMatchesAnalyticPack) {
    {
        var nu = 4.0;
        var x = 0.9;
        const quantape::math::StudentT<var> g{nu};
        auto* stack = stan::math::ChainableStack::instance_;
        const std::size_t before = stack->var_stack_.size();
        var y = g.cdf(x);
        const std::size_t nodes = stack->var_stack_.size() - before;
        y.grad();
        const auto pack = quantape::math::detail::studentTCdfGradHess(4.0, 0.9);
        checkVarRoute("student-t cdf", y.val(), {nu.adj(), x.adj()}, pack.value,
                      {pack.gNu, pack.gx});
        EXPECT_EQ(nodes, std::size_t(1));
        RecordProperty("var_studentt_cdf_nodes", static_cast<int>(nodes));
    }
    {
        var nu = 4.0;
        var x = 0.9;
        const quantape::math::StudentT<var> g{nu};
        auto* stack = stan::math::ChainableStack::instance_;
        const std::size_t before = stack->var_stack_.size();
        var y = g.pdf(x);
        const std::size_t nodes = stack->var_stack_.size() - before;
        y.grad();
        const auto pack = quantape::math::detail::studentTPdfGradHess(4.0, 0.9);
        checkVarRoute("student-t pdf", y.val(), {nu.adj(), x.adj()}, pack.value,
                      {pack.gNu, pack.gx});
        EXPECT_EQ(nodes, std::size_t(1));
        RecordProperty("var_studentt_pdf_nodes", static_cast<int>(nodes));
    }
}

TEST_F(DistributionAdTest, studentTCallbackFvarHessianMatchesAnalyticPack) {
    using fvar = stan::math::fvar<stan::math::var>;
    const double nu0 = 4.0, x0 = 0.9;
    Eigen::VectorXd point(2);
    point << nu0, x0;
    for (bool useCdf : {true, false}) {
        SCOPED_TRACE(useCdf);
        double fx = 0.0;
        Eigen::VectorXd grad(2);
        Eigen::MatrixXd hess(2, 2);
        stan::math::hessian(StudentTFvarHessian{useCdf}, point, fx, grad, hess);
        if (useCdf) {
            const auto pack = quantape::math::detail::studentTCdfGradHess(nu0, x0);
            checkPackRoute("student-t fvar cdf", fx, grad, hess, pack.value, {pack.gNu, pack.gx},
                           {pack.hNuNu, pack.hNux, pack.hNux, pack.hxx});
        } else {
            const auto pack = quantape::math::detail::studentTPdfGradHess(nu0, x0);
            checkPackRoute("student-t fvar pdf", fx, grad, hess, pack.value, {pack.gNu, pack.gx},
                           {pack.hNuNu, pack.hNux, pack.hNux, pack.hxx});
        }
    }
    {
        fvar nu(nu0, 1.0), x(x0, 0.0);
        const quantape::math::StudentT<fvar> g{nu};
        auto* stack = stan::math::ChainableStack::instance_;
        const std::size_t before = stack->var_stack_.size();
        fvar y = g.cdf(x);
        const std::size_t nodes = stack->var_stack_.size() - before;
        EXPECT_EQ(nodes, std::size_t(2));
        RecordProperty("fvar_studentt_cdf_nodes", static_cast<int>(nodes));
        const std::size_t beforePdf = stack->var_stack_.size();
        fvar yPdf = g.pdf(x);
        const std::size_t pdfNodes = stack->var_stack_.size() - beforePdf;
        EXPECT_EQ(pdfNodes, std::size_t(2));
        RecordProperty("fvar_studentt_pdf_nodes", static_cast<int>(pdfNodes));
        (void)y;
        (void)yPdf;
    }
}

TEST_F(DistributionAdTest, lognormalCallbackVarMatchesAnalyticPack) {
    {
        var mu = 0.2;
        var sigma = 0.7;
        var x = 1.3;
        const quantape::math::Lognormal<var> g{mu, sigma};
        auto* stack = stan::math::ChainableStack::instance_;
        const std::size_t before = stack->var_stack_.size();
        var y = g.cdf(x);
        const std::size_t nodes = stack->var_stack_.size() - before;
        y.grad();
        const auto pack = quantape::math::detail::lognormalCdfGradHess(0.2, 0.7, 1.3);
        checkVarRoute("lognormal cdf", y.val(), {mu.adj(), sigma.adj(), x.adj()}, pack.value,
                      {pack.gmu, pack.gsig, pack.gx});
        EXPECT_EQ(nodes, std::size_t(1));
        RecordProperty("var_lognormal_cdf_nodes", static_cast<int>(nodes));
    }
    {
        var mu = 0.2;
        var sigma = 0.7;
        var x = 1.3;
        const quantape::math::Lognormal<var> g{mu, sigma};
        auto* stack = stan::math::ChainableStack::instance_;
        const std::size_t before = stack->var_stack_.size();
        var y = g.pdf(x);
        const std::size_t nodes = stack->var_stack_.size() - before;
        y.grad();
        const auto pack = quantape::math::detail::lognormalPdfGradHess(0.2, 0.7, 1.3);
        checkVarRoute("lognormal pdf", y.val(), {mu.adj(), sigma.adj(), x.adj()}, pack.value,
                      {pack.gmu, pack.gsig, pack.gx});
        EXPECT_EQ(nodes, std::size_t(1));
        RecordProperty("var_lognormal_pdf_nodes", static_cast<int>(nodes));
    }
}

TEST_F(DistributionAdTest, lognormalCallbackFvarHessianMatchesAnalyticPack) {
    using fvar = stan::math::fvar<stan::math::var>;
    const double mu0 = 0.2, sigma0 = 0.7, x0 = 1.3;
    Eigen::VectorXd point(3);
    point << mu0, sigma0, x0;
    for (bool useCdf : {true, false}) {
        SCOPED_TRACE(useCdf);
        double fx = 0.0;
        Eigen::VectorXd grad(3);
        Eigen::MatrixXd hess(3, 3);
        stan::math::hessian(LognormalFvarHessian{useCdf}, point, fx, grad, hess);
        if (useCdf) {
            const auto pack = quantape::math::detail::lognormalCdfGradHess(mu0, sigma0, x0);
            checkPackRoute("lognormal fvar cdf", fx, grad, hess, pack.value,
                           {pack.gmu, pack.gsig, pack.gx},
                           {pack.hmumu, pack.hmusig, pack.hmux, pack.hmusig, pack.hsigsig,
                            pack.hsigx, pack.hmux, pack.hsigx, pack.hxx});
        } else {
            const auto pack = quantape::math::detail::lognormalPdfGradHess(mu0, sigma0, x0);
            checkPackRoute("lognormal fvar pdf", fx, grad, hess, pack.value,
                           {pack.gmu, pack.gsig, pack.gx},
                           {pack.hmumu, pack.hmusig, pack.hmux, pack.hmusig, pack.hsigsig,
                            pack.hsigx, pack.hmux, pack.hsigx, pack.hxx});
        }
    }
    {
        fvar mu(mu0, 1.0), sigma(sigma0, 0.0), x(x0, 0.0);
        const quantape::math::Lognormal<fvar> g{mu, sigma};
        auto* stack = stan::math::ChainableStack::instance_;
        const std::size_t before = stack->var_stack_.size();
        fvar y = g.cdf(x);
        const std::size_t nodes = stack->var_stack_.size() - before;
        EXPECT_EQ(nodes, std::size_t(2));
        RecordProperty("fvar_lognormal_cdf_nodes", static_cast<int>(nodes));
        const std::size_t beforePdf = stack->var_stack_.size();
        fvar yPdf = g.pdf(x);
        const std::size_t pdfNodes = stack->var_stack_.size() - beforePdf;
        EXPECT_EQ(pdfNodes, std::size_t(2));
        RecordProperty("fvar_lognormal_pdf_nodes", static_cast<int>(pdfNodes));
        (void)y;
        (void)yPdf;
    }
}

TEST_F(DistributionAdTest, exponentialCallbackVarMatchesAnalyticPack) {
    {
        var lambda = 1.7;
        var x = 0.8;
        const quantape::math::Exponential<var> g{lambda};
        auto* stack = stan::math::ChainableStack::instance_;
        const std::size_t before = stack->var_stack_.size();
        var y = g.cdf(x);
        const std::size_t nodes = stack->var_stack_.size() - before;
        y.grad();
        const auto pack = quantape::math::detail::exponentialCdfGradHess(1.7, 0.8);
        checkVarRoute("exponential cdf", y.val(), {lambda.adj(), x.adj()}, pack.value,
                      {pack.gLambda, pack.gx});
        EXPECT_EQ(nodes, std::size_t(1));
        RecordProperty("var_exponential_cdf_nodes", static_cast<int>(nodes));
    }
    {
        var lambda = 1.7;
        var x = 0.8;
        const quantape::math::Exponential<var> g{lambda};
        auto* stack = stan::math::ChainableStack::instance_;
        const std::size_t before = stack->var_stack_.size();
        var y = g.pdf(x);
        const std::size_t nodes = stack->var_stack_.size() - before;
        y.grad();
        const auto pack = quantape::math::detail::exponentialPdfGradHess(1.7, 0.8);
        checkVarRoute("exponential pdf", y.val(), {lambda.adj(), x.adj()}, pack.value,
                      {pack.gLambda, pack.gx});
        EXPECT_EQ(nodes, std::size_t(1));
        RecordProperty("var_exponential_pdf_nodes", static_cast<int>(nodes));
    }
}

TEST_F(DistributionAdTest, exponentialCallbackFvarHessianMatchesAnalyticPack) {
    using fvar = stan::math::fvar<stan::math::var>;
    const double lambda0 = 1.7, x0 = 0.8;
    Eigen::VectorXd point(2);
    point << lambda0, x0;
    for (bool useCdf : {true, false}) {
        SCOPED_TRACE(useCdf);
        double fx = 0.0;
        Eigen::VectorXd grad(2);
        Eigen::MatrixXd hess(2, 2);
        stan::math::hessian(ExponentialFvarHessian{useCdf}, point, fx, grad, hess);
        if (useCdf) {
            const auto pack = quantape::math::detail::exponentialCdfGradHess(lambda0, x0);
            checkPackRoute("exponential fvar cdf", fx, grad, hess, pack.value,
                           {pack.gLambda, pack.gx}, {pack.hLL, pack.hLx, pack.hLx, pack.hxx});
        } else {
            const auto pack = quantape::math::detail::exponentialPdfGradHess(lambda0, x0);
            checkPackRoute("exponential fvar pdf", fx, grad, hess, pack.value,
                           {pack.gLambda, pack.gx}, {pack.hLL, pack.hLx, pack.hLx, pack.hxx});
        }
    }
    {
        fvar lambda(lambda0, 1.0), x(x0, 0.0);
        const quantape::math::Exponential<fvar> g{lambda};
        auto* stack = stan::math::ChainableStack::instance_;
        const std::size_t before = stack->var_stack_.size();
        fvar y = g.cdf(x);
        const std::size_t nodes = stack->var_stack_.size() - before;
        EXPECT_EQ(nodes, std::size_t(2));
        RecordProperty("fvar_exponential_cdf_nodes", static_cast<int>(nodes));
        const std::size_t beforePdf = stack->var_stack_.size();
        fvar yPdf = g.pdf(x);
        const std::size_t pdfNodes = stack->var_stack_.size() - beforePdf;
        EXPECT_EQ(pdfNodes, std::size_t(2));
        RecordProperty("fvar_exponential_pdf_nodes", static_cast<int>(pdfNodes));
        (void)y;
        (void)yPdf;
    }
}

TEST_F(DistributionAdTest, inverseGaussianCallbackVarMatchesAnalyticPack) {
    {
        var mu = 1.5;
        var lambda = 2.3;
        var x = 1.2;
        const quantape::math::InverseGaussian<var> g{mu, lambda};
        auto* stack = stan::math::ChainableStack::instance_;
        const std::size_t before = stack->var_stack_.size();
        var y = g.cdf(x);
        const std::size_t nodes = stack->var_stack_.size() - before;
        y.grad();
        const auto pack = quantape::math::detail::inverseGaussianCdfGradHess(1.5, 2.3, 1.2);
        checkVarRoute("inverse-gaussian cdf", y.val(), {mu.adj(), lambda.adj(), x.adj()},
                      pack.value, {pack.gmu, pack.gLambda, pack.gx});
        EXPECT_EQ(nodes, std::size_t(1));
        RecordProperty("var_invgaussian_cdf_nodes", static_cast<int>(nodes));
    }
    {
        var mu = 1.5;
        var lambda = 2.3;
        var x = 1.2;
        const quantape::math::InverseGaussian<var> g{mu, lambda};
        auto* stack = stan::math::ChainableStack::instance_;
        const std::size_t before = stack->var_stack_.size();
        var y = g.pdf(x);
        const std::size_t nodes = stack->var_stack_.size() - before;
        y.grad();
        const auto pack = quantape::math::detail::inverseGaussianPdfGradHess(1.5, 2.3, 1.2);
        checkVarRoute("inverse-gaussian pdf", y.val(), {mu.adj(), lambda.adj(), x.adj()},
                      pack.value, {pack.gmu, pack.gLambda, pack.gx});
        EXPECT_EQ(nodes, std::size_t(1));
        RecordProperty("var_invgaussian_pdf_nodes", static_cast<int>(nodes));
    }
}

TEST_F(DistributionAdTest, inverseGaussianCallbackFvarHessianMatchesAnalyticPack) {
    using fvar = stan::math::fvar<stan::math::var>;
    const double mu0 = 1.5, lambda0 = 2.3, x0 = 1.2;
    Eigen::VectorXd point(3);
    point << mu0, lambda0, x0;
    for (bool useCdf : {true, false}) {
        SCOPED_TRACE(useCdf);
        double fx = 0.0;
        Eigen::VectorXd grad(3);
        Eigen::MatrixXd hess(3, 3);
        stan::math::hessian(InverseGaussianFvarHessian{useCdf}, point, fx, grad, hess);
        if (useCdf) {
            const auto pack = quantape::math::detail::inverseGaussianCdfGradHess(mu0, lambda0, x0);
            checkPackRoute("inverse-gaussian fvar cdf", fx, grad, hess, pack.value,
                           {pack.gmu, pack.gLambda, pack.gx},
                           {pack.hmumu, pack.hmuLambda, pack.hmux, pack.hmuLambda, pack.hLL,
                            pack.hLx, pack.hmux, pack.hLx, pack.hxx});
        } else {
            const auto pack = quantape::math::detail::inverseGaussianPdfGradHess(mu0, lambda0, x0);
            checkPackRoute("inverse-gaussian fvar pdf", fx, grad, hess, pack.value,
                           {pack.gmu, pack.gLambda, pack.gx},
                           {pack.hmumu, pack.hmuLambda, pack.hmux, pack.hmuLambda, pack.hLL,
                            pack.hLx, pack.hmux, pack.hLx, pack.hxx});
        }
    }
    {
        fvar mu(mu0, 1.0), lambda(lambda0, 0.0), x(x0, 0.0);
        const quantape::math::InverseGaussian<fvar> g{mu, lambda};
        auto* stack = stan::math::ChainableStack::instance_;
        const std::size_t before = stack->var_stack_.size();
        fvar y = g.cdf(x);
        const std::size_t nodes = stack->var_stack_.size() - before;
        EXPECT_EQ(nodes, std::size_t(2));
        RecordProperty("fvar_invgaussian_cdf_nodes", static_cast<int>(nodes));
        const std::size_t beforePdf = stack->var_stack_.size();
        fvar yPdf = g.pdf(x);
        const std::size_t pdfNodes = stack->var_stack_.size() - beforePdf;
        EXPECT_EQ(pdfNodes, std::size_t(2));
        RecordProperty("fvar_invgaussian_pdf_nodes", static_cast<int>(pdfNodes));
        (void)y;
        (void)yPdf;
    }
}

// Fallback boundaries of the family routes: the guards must reproduce the
// generic kernel constants/derivatives (the packs are NaN or one-sided there).
TEST_F(DistributionAdTest, familyEdgeBoundariesFallBackToGenericPath) {
    {
        var shape = 0.8;
        var rate = 1.7;
        var x0 = 0.0;
        const quantape::math::Gamma<var> g{shape, rate};
        var y = g.cdf(x0);
        y.grad();
        CHECK_CLOSE("gamma cdf(0) value", y.val(), 0.0, 0.0);
        CHECK_CLOSE("gamma cdf(0) d/dx", x0.adj(), 0.0, 0.0);
        CHECK_CLOSE("gamma cdf(0) d/dshape", shape.adj(), 0.0, 0.0);
    }
    {
        var alpha = 2.0;
        var beta = 3.5;
        var x1 = 1.0;
        const quantape::math::Beta<var> b{alpha, beta};
        var y = b.cdf(x1);
        y.grad();
        CHECK_CLOSE("beta cdf(1) value", y.val(), 1.0, 0.0);
        CHECK_CLOSE("beta cdf(1) d/dx", x1.adj(), 0.0, 0.0);
    }
    {
        var nu = 4.0;
        var x0 = 0.0;
        const quantape::math::StudentT<var> t{nu};
        var y = t.cdf(x0);
        y.grad();
        CHECK_CLOSE("student-t cdf(0) value", y.val(), 0.5, 0.0);
        CHECK_CLOSE("student-t cdf(0) d/dx", x0.adj(), 0.0, 0.0);
    }
    {
        var df = 4.5;
        var x0 = 0.0;
        const quantape::math::ChiSquare<var> c{df};
        var y = c.cdf(x0);
        y.grad();
        CHECK_CLOSE("chi2 cdf(0) value", y.val(), 0.0, 0.0);
        CHECK_CLOSE("chi2 cdf(0) d/dx", x0.adj(), 0.0, 0.0);
    }
}

// ============================================================================
// Noncentral family callback routes: the 1-node `var` and 2-node `fvar<var>`
// specializations for NoncentralChiSquare::cdf (the header provides no nu partial
// for the density, so pdf stays double-only), NoncentralBeta::{cdf,pdf} and
// NoncentralF::{cdf,pdf}. Values and every adjoint are checked against the
// double analytic packs; the second-order routes run `stan::math::hessian`
// over all active leaves. The evaluation points sit in the packs'
// mild-conditioning regime (nu/ncp <= ~12, moderate alpha/beta), far inside the
// documented reference-grid envelopes, so the shared 1e-9 gate is not loosened.
// ============================================================================

namespace {

struct NoncentralChiSquareFvarHessian {
    template <class T>
    T operator()(const Eigen::Matrix<T, Eigen::Dynamic, 1>& v) const {
        const quantape::math::NoncentralChiSquare<T> nc{v(0), v(1)};
        return nc.cdf(v(2));
    }
};

struct NoncentralBetaFvarHessian {
    bool useCdf;
    template <class T>
    T operator()(const Eigen::Matrix<T, Eigen::Dynamic, 1>& v) const {
        const quantape::math::NoncentralBeta<T> nb{v(0), v(1), v(2)};
        return useCdf ? nb.cdf(v(3)) : nb.pdf(v(3));
    }
};

struct NoncentralFFvarHessian {
    bool useCdf;
    template <class T>
    T operator()(const Eigen::Matrix<T, Eigen::Dynamic, 1>& v) const {
        const quantape::math::NoncentralF<T> nf{v(0), v(1), v(2)};
        return useCdf ? nf.cdf(v(3)) : nf.pdf(v(3));
    }
};

} // namespace

TEST_F(DistributionAdTest, noncentralChiSquareCallbackVarMatchesAnalyticPack) {
    {
        var df = 3.0;
        var ncp = 2.0;
        var x = 2.5;
        const quantape::math::NoncentralChiSquare<var> nc{df, ncp};
        auto* stack = stan::math::ChainableStack::instance_;
        const std::size_t before = stack->var_stack_.size();
        var y = nc.cdf(x);
        const std::size_t nodes = stack->var_stack_.size() - before;
        y.grad();
        const auto pack = quantape::math::detail::noncentralChiSquareCdfGradHess(3.0, 2.0, 2.5);
        checkVarRoute("nccs cdf", y.val(), {df.adj(), ncp.adj(), x.adj()}, pack.value,
                      {pack.gNu, pack.gLambda, pack.gx});
        EXPECT_EQ(nodes, std::size_t(1));
        RecordProperty("var_nccs_cdf_nodes", static_cast<int>(nodes));
    }
    {
        // ncp = 0 is the pack's central chi-square reduction (no fallback).
        var df = 4.0;
        var ncp = 0.0;
        var x = 1.7;
        const quantape::math::NoncentralChiSquare<var> nc{df, ncp};
        var y = nc.cdf(x);
        y.grad();
        const auto pack = quantape::math::detail::noncentralChiSquareCdfGradHess(4.0, 0.0, 1.7);
        checkVarRoute("nccs cdf ncp=0", y.val(), {df.adj(), ncp.adj(), x.adj()}, pack.value,
                      {pack.gNu, pack.gLambda, pack.gx});
    }
    {
        // x = 0 is the all-NaN pack boundary: the route falls back to the
        // generic tapped mixture (value 0 with zero adjoints).
        var df = 3.0;
        var ncp = 2.0;
        var x0 = 0.0;
        const quantape::math::NoncentralChiSquare<var> nc{df, ncp};
        var y = nc.cdf(x0);
        y.grad();
        CHECK_CLOSE("nccs cdf(0) value", y.val(), 0.0, 0.0);
        CHECK_CLOSE("nccs cdf(0) d/ddf", df.adj(), 0.0, 0.0);
        CHECK_CLOSE("nccs cdf(0) d/dncp", ncp.adj(), 0.0, 0.0);
        CHECK_CLOSE("nccs cdf(0) d/dx", x0.adj(), 0.0, 0.0);
    }
    {
        // x < 0 takes the pack's zero branch (generic constant zero).
        var df = 3.0;
        var ncp = 2.0;
        var xNeg = -1.0;
        const quantape::math::NoncentralChiSquare<var> nc{df, ncp};
        var y = nc.cdf(xNeg);
        y.grad();
        CHECK_CLOSE("nccs cdf(-1) value", y.val(), 0.0, 0.0);
        CHECK_CLOSE("nccs cdf(-1) d/ddf", df.adj(), 0.0, 0.0);
        CHECK_CLOSE("nccs cdf(-1) d/dncp", ncp.adj(), 0.0, 0.0);
        CHECK_CLOSE("nccs cdf(-1) d/dx", xNeg.adj(), 0.0, 0.0);
    }
}

TEST_F(DistributionAdTest, noncentralChiSquareCallbackFvarHessianMatchesAnalyticPack) {
    using fvar = stan::math::fvar<stan::math::var>;
    const double df0 = 3.0, ncp0 = 2.0, x0 = 2.5;
    Eigen::VectorXd point(3);
    point << df0, ncp0, x0;
    double fx = 0.0;
    Eigen::VectorXd grad(3);
    Eigen::MatrixXd hess(3, 3);
    stan::math::hessian(NoncentralChiSquareFvarHessian{}, point, fx, grad, hess);

    const auto pack = quantape::math::detail::noncentralChiSquareCdfGradHess(df0, ncp0, x0);
    checkPackRoute("nccs fvar cdf", fx, grad, hess, pack.value, {pack.gNu, pack.gLambda, pack.gx},
                   {pack.hNuNu, pack.hNuLambda, pack.hNuX, pack.hNuLambda, pack.hLambdaLambda,
                    pack.hLambdaX, pack.hNuX, pack.hLambdaX, pack.hxx});

    fvar df(df0, 1.0), ncp(ncp0, 0.0), x(x0, 0.0);
    const quantape::math::NoncentralChiSquare<fvar> nc{df, ncp};
    auto* stack = stan::math::ChainableStack::instance_;
    const std::size_t before = stack->var_stack_.size();
    fvar y = nc.cdf(x);
    const std::size_t nodes = stack->var_stack_.size() - before;
    EXPECT_EQ(nodes, std::size_t(2));
    RecordProperty("fvar_nccs_cdf_nodes", static_cast<int>(nodes));
    (void)y;
}

TEST_F(DistributionAdTest, noncentralBetaCallbackVarMatchesAnalyticPack) {
    {
        var alpha = 2.0;
        var beta = 3.5;
        var ncp = 1.5;
        var x = 0.4;
        const quantape::math::NoncentralBeta<var> nb{alpha, beta, ncp};
        auto* stack = stan::math::ChainableStack::instance_;
        const std::size_t before = stack->var_stack_.size();
        var y = nb.cdf(x);
        const std::size_t nodes = stack->var_stack_.size() - before;
        y.grad();
        const auto pack = quantape::math::detail::noncentralBetaCdfGradHess(2.0, 3.5, 1.5, 0.4);
        checkVarRoute("ncb cdf", y.val(), {alpha.adj(), beta.adj(), ncp.adj(), x.adj()}, pack.value,
                      {pack.gAlpha, pack.gBeta, pack.gNcp, pack.gx});
        EXPECT_EQ(nodes, std::size_t(1));
        RecordProperty("var_ncb_cdf_nodes", static_cast<int>(nodes));
    }
    {
        var alpha = 2.0;
        var beta = 3.5;
        var ncp = 1.5;
        var x = 0.4;
        const quantape::math::NoncentralBeta<var> nb{alpha, beta, ncp};
        auto* stack = stan::math::ChainableStack::instance_;
        const std::size_t before = stack->var_stack_.size();
        var y = nb.pdf(x);
        const std::size_t nodes = stack->var_stack_.size() - before;
        y.grad();
        const auto pack = quantape::math::detail::noncentralBetaPdfGradHess(2.0, 3.5, 1.5, 0.4);
        checkVarRoute("ncb pdf", y.val(), {alpha.adj(), beta.adj(), ncp.adj(), x.adj()}, pack.value,
                      {pack.gAlpha, pack.gBeta, pack.gNcp, pack.gx});
        EXPECT_EQ(nodes, std::size_t(1));
        RecordProperty("var_ncb_pdf_nodes", static_cast<int>(nodes));
    }
    {
        // ncp = 0 reduces exactly to the central-beta pack entries.
        var alpha = 2.0;
        var beta = 3.5;
        var ncp = 0.0;
        var x = 0.4;
        const quantape::math::NoncentralBeta<var> nb{alpha, beta, ncp};
        var y = nb.cdf(x);
        y.grad();
        const auto pack = quantape::math::detail::noncentralBetaCdfGradHess(2.0, 3.5, 0.0, 0.4);
        checkVarRoute("ncb cdf ncp=0", y.val(), {alpha.adj(), beta.adj(), ncp.adj(), x.adj()},
                      pack.value, {pack.gAlpha, pack.gBeta, pack.gNcp, pack.gx});
    }
}

TEST_F(DistributionAdTest, noncentralBetaCallbackFvarHessianMatchesAnalyticPack) {
    using fvar = stan::math::fvar<stan::math::var>;
    const double alpha0 = 2.0, beta0 = 3.5, ncp0 = 1.5, x0 = 0.4;
    Eigen::VectorXd point(4);
    point << alpha0, beta0, ncp0, x0;
    for (bool useCdf : {true, false}) {
        SCOPED_TRACE(useCdf);
        double fx = 0.0;
        Eigen::VectorXd grad(4);
        Eigen::MatrixXd hess(4, 4);
        stan::math::hessian(NoncentralBetaFvarHessian{useCdf}, point, fx, grad, hess);
        if (useCdf) {
            const auto pack =
                quantape::math::detail::noncentralBetaCdfGradHess(alpha0, beta0, ncp0, x0);
            checkPackRoute("ncb fvar cdf", fx, grad, hess, pack.value,
                           {pack.gAlpha, pack.gBeta, pack.gNcp, pack.gx},
                           {pack.hAA, pack.hAB, pack.hANcp, pack.hAx, pack.hAB, pack.hBB,
                            pack.hBNcp, pack.hBx, pack.hANcp, pack.hBNcp, pack.hNcpNcp, pack.hNcpX,
                            pack.hAx, pack.hBx, pack.hNcpX, pack.hxx});
        } else {
            const auto pack =
                quantape::math::detail::noncentralBetaPdfGradHess(alpha0, beta0, ncp0, x0);
            checkPackRoute("ncb fvar pdf", fx, grad, hess, pack.value,
                           {pack.gAlpha, pack.gBeta, pack.gNcp, pack.gx},
                           {pack.hAA, pack.hAB, pack.hANcp, pack.hAx, pack.hAB, pack.hBB,
                            pack.hBNcp, pack.hBx, pack.hANcp, pack.hBNcp, pack.hNcpNcp, pack.hNcpX,
                            pack.hAx, pack.hBx, pack.hNcpX, pack.hxx});
        }
    }
    {
        fvar alpha(alpha0, 1.0), beta(beta0, 0.0), ncp(ncp0, 0.0), x(x0, 0.0);
        const quantape::math::NoncentralBeta<fvar> nb{alpha, beta, ncp};
        auto* stack = stan::math::ChainableStack::instance_;
        const std::size_t before = stack->var_stack_.size();
        fvar y = nb.cdf(x);
        const std::size_t nodes = stack->var_stack_.size() - before;
        EXPECT_EQ(nodes, std::size_t(2));
        RecordProperty("fvar_ncb_cdf_nodes", static_cast<int>(nodes));
        const std::size_t beforePdf = stack->var_stack_.size();
        fvar yPdf = nb.pdf(x);
        const std::size_t pdfNodes = stack->var_stack_.size() - beforePdf;
        EXPECT_EQ(pdfNodes, std::size_t(2));
        RecordProperty("fvar_ncb_pdf_nodes", static_cast<int>(pdfNodes));
        (void)y;
        (void)yPdf;
    }
}

TEST_F(DistributionAdTest, noncentralFCallbackVarMatchesAnalyticPack) {
    {
        var nu1 = 5.0;
        var nu2 = 7.0;
        var ncp = 1.5;
        var y0 = 1.2;
        const quantape::math::NoncentralF<var> nf{nu1, nu2, ncp};
        auto* stack = stan::math::ChainableStack::instance_;
        const std::size_t before = stack->var_stack_.size();
        var y = nf.cdf(y0);
        const std::size_t nodes = stack->var_stack_.size() - before;
        y.grad();
        const auto pack = quantape::math::detail::noncentralFCdfGradHess(5.0, 7.0, 1.5, 1.2);
        checkVarRoute("ncf cdf", y.val(), {nu1.adj(), nu2.adj(), ncp.adj(), y0.adj()}, pack.value,
                      {pack.gNu1, pack.gNu2, pack.gNcp, pack.gy});
        EXPECT_EQ(nodes, std::size_t(1));
        RecordProperty("var_ncf_cdf_nodes", static_cast<int>(nodes));
    }
    {
        var nu1 = 5.0;
        var nu2 = 7.0;
        var ncp = 1.5;
        var y0 = 1.2;
        const quantape::math::NoncentralF<var> nf{nu1, nu2, ncp};
        auto* stack = stan::math::ChainableStack::instance_;
        const std::size_t before = stack->var_stack_.size();
        var y = nf.pdf(y0);
        const std::size_t nodes = stack->var_stack_.size() - before;
        y.grad();
        const auto pack = quantape::math::detail::noncentralFPdfGradHess(5.0, 7.0, 1.5, 1.2);
        checkVarRoute("ncf pdf", y.val(), {nu1.adj(), nu2.adj(), ncp.adj(), y0.adj()}, pack.value,
                      {pack.gNu1, pack.gNu2, pack.gNcp, pack.gy});
        EXPECT_EQ(nodes, std::size_t(1));
        RecordProperty("var_ncf_pdf_nodes", static_cast<int>(nodes));
    }
}

TEST_F(DistributionAdTest, noncentralFCallbackFvarHessianMatchesAnalyticPack) {
    using fvar = stan::math::fvar<stan::math::var>;
    const double nu1_0 = 5.0, nu2_0 = 7.0, ncp0 = 1.5, y0 = 1.2;
    Eigen::VectorXd point(4);
    point << nu1_0, nu2_0, ncp0, y0;
    for (bool useCdf : {true, false}) {
        SCOPED_TRACE(useCdf);
        double fx = 0.0;
        Eigen::VectorXd grad(4);
        Eigen::MatrixXd hess(4, 4);
        stan::math::hessian(NoncentralFFvarHessian{useCdf}, point, fx, grad, hess);
        if (useCdf) {
            const auto pack =
                quantape::math::detail::noncentralFCdfGradHess(nu1_0, nu2_0, ncp0, y0);
            checkPackRoute("ncf fvar cdf", fx, grad, hess, pack.value,
                           {pack.gNu1, pack.gNu2, pack.gNcp, pack.gy},
                           {pack.hNu1Nu1, pack.hNu1Nu2, pack.hNu1Ncp, pack.hNu1Y, pack.hNu1Nu2,
                            pack.hNu2Nu2, pack.hNu2Ncp, pack.hNu2Y, pack.hNu1Ncp, pack.hNu2Ncp,
                            pack.hNcpNcp, pack.hNcpY, pack.hNu1Y, pack.hNu2Y, pack.hNcpY,
                            pack.hyy});
        } else {
            const auto pack =
                quantape::math::detail::noncentralFPdfGradHess(nu1_0, nu2_0, ncp0, y0);
            checkPackRoute("ncf fvar pdf", fx, grad, hess, pack.value,
                           {pack.gNu1, pack.gNu2, pack.gNcp, pack.gy},
                           {pack.hNu1Nu1, pack.hNu1Nu2, pack.hNu1Ncp, pack.hNu1Y, pack.hNu1Nu2,
                            pack.hNu2Nu2, pack.hNu2Ncp, pack.hNu2Y, pack.hNu1Ncp, pack.hNu2Ncp,
                            pack.hNcpNcp, pack.hNcpY, pack.hNu1Y, pack.hNu2Y, pack.hNcpY,
                            pack.hyy});
        }
    }
    {
        fvar nu1(nu1_0, 1.0), nu2(nu2_0, 0.0), ncp(ncp0, 0.0), y(y0, 0.0);
        const quantape::math::NoncentralF<fvar> nf{nu1, nu2, ncp};
        auto* stack = stan::math::ChainableStack::instance_;
        const std::size_t before = stack->var_stack_.size();
        fvar f = nf.cdf(y);
        const std::size_t nodes = stack->var_stack_.size() - before;
        EXPECT_EQ(nodes, std::size_t(2));
        RecordProperty("fvar_ncf_cdf_nodes", static_cast<int>(nodes));
        const std::size_t beforePdf = stack->var_stack_.size();
        fvar fPdf = nf.pdf(y);
        const std::size_t pdfNodes = stack->var_stack_.size() - beforePdf;
        EXPECT_EQ(pdfNodes, std::size_t(2));
        RecordProperty("fvar_ncf_pdf_nodes", static_cast<int>(pdfNodes));
        (void)f;
        (void)fPdf;
    }
}

// ============================================================================
// Noncentral tail/log callback routes (workstream NL): NoncentralChiSquare::
// {ccdf,logCcdf} and NoncentralBeta/NoncentralF::{ccdf,logPdf}. The node value
// is the matching double accessor (the log-space accessors stay exact where the
// double probability underflows into the quotient) and every push comes from
// the family pack: ccdf negates the cdf pack; logCcdf is the quotient rule on
// the negated pack; logPdf is f_i/f and f_ij/f - f_i f_j/f^2 on the density
// pack, zeroed when the double density is exactly zero. Values and adjoints are
// checked against the double packs (second-order entries at 1e-9 rel / 1e-12
// abs); the fallback boundaries are checked against the documented branches.
// ============================================================================

namespace {

/// fvar<var> functor for the noncentral chi-square tail routes: `useLogCcdf`
/// selects `logCcdf`, otherwise `ccdf`.
struct NoncentralChiSquareTailFvarHessian {
    bool useLogCcdf;
    template <class T>
    T operator()(const Eigen::Matrix<T, Eigen::Dynamic, 1>& v) const {
        const quantape::math::NoncentralChiSquare<T> nc{v(0), v(1)};
        return useLogCcdf ? nc.logCcdf(v(2)) : nc.ccdf(v(2));
    }
};

/// fvar<var> functor for the noncentral-beta tail/log routes.
struct NoncentralBetaTailFvarHessian {
    bool useLogPdf;
    template <class T>
    T operator()(const Eigen::Matrix<T, Eigen::Dynamic, 1>& v) const {
        const quantape::math::NoncentralBeta<T> nb{v(0), v(1), v(2)};
        return useLogPdf ? nb.logPdf(v(3)) : nb.ccdf(v(3));
    }
};

/// fvar<var> functor for the noncentral-F tail/log routes.
struct NoncentralFTailFvarHessian {
    bool useLogPdf;
    template <class T>
    T operator()(const Eigen::Matrix<T, Eigen::Dynamic, 1>& v) const {
        const quantape::math::NoncentralF<T> nf{v(0), v(1), v(2)};
        return useLogPdf ? nf.logPdf(v(3)) : nf.ccdf(v(3));
    }
};

} // namespace

TEST_F(DistributionAdTest, noncentralChiSquareTailCallbackVarMatchesAnalyticPack) {
    const quantape::math::NoncentralChiSquare<double> nc64{3.0, 2.0};
    const auto pack = quantape::math::detail::noncentralChiSquareCdfGradHess(3.0, 2.0, 2.5);
    const double invG = 1.0 / nc64.ccdf(2.5);
    {
        var df = 3.0;
        var ncp = 2.0;
        var x = 2.5;
        const quantape::math::NoncentralChiSquare<var> nc{df, ncp};
        auto* stack = stan::math::ChainableStack::instance_;
        const std::size_t before = stack->var_stack_.size();
        var y = nc.ccdf(x);
        const std::size_t nodes = stack->var_stack_.size() - before;
        y.grad();
        checkVarRoute("nccs ccdf", y.val(), {df.adj(), ncp.adj(), x.adj()}, nc64.ccdf(2.5),
                      {-pack.gNu, -pack.gLambda, -pack.gx});
        EXPECT_EQ(nodes, std::size_t(1));
        RecordProperty("var_nccs_ccdf_nodes", static_cast<int>(nodes));
    }
    {
        var df = 3.0;
        var ncp = 2.0;
        var x = 2.5;
        const quantape::math::NoncentralChiSquare<var> nc{df, ncp};
        auto* stack = stan::math::ChainableStack::instance_;
        const std::size_t before = stack->var_stack_.size();
        var y = nc.logCcdf(x);
        const std::size_t nodes = stack->var_stack_.size() - before;
        y.grad();
        checkVarRoute("nccs logccdf", y.val(), {df.adj(), ncp.adj(), x.adj()}, nc64.logCcdf(2.5),
                      {-pack.gNu * invG, -pack.gLambda * invG, -pack.gx * invG});
        EXPECT_EQ(nodes, std::size_t(1));
        RecordProperty("var_nccs_logccdf_nodes", static_cast<int>(nodes));
    }
    {
        // ncp = 0 is the pack's central chi-square reduction (no fallback).
        var df = 4.0;
        var ncp = 0.0;
        var x = 1.7;
        const quantape::math::NoncentralChiSquare<var> nc{df, ncp};
        var y = nc.ccdf(x);
        y.grad();
        const auto central = quantape::math::detail::noncentralChiSquareCdfGradHess(4.0, 0.0, 1.7);
        const quantape::math::NoncentralChiSquare<double> c64{4.0, 0.0};
        checkVarRoute("nccs ccdf ncp=0", y.val(), {df.adj(), ncp.adj(), x.adj()}, c64.ccdf(1.7),
                      {-central.gNu, -central.gLambda, -central.gx});
    }
    {
        // x = 0: the pack is all-NaN; ccdf falls back to the generic tapped
        // mixture (~1, zero adjoints) and logCcdf takes its documented constant
        // branch (0 with zero pushes).
        var df = 3.0;
        var ncp = 2.0;
        var x0 = 0.0;
        const quantape::math::NoncentralChiSquare<var> nc{df, ncp};
        var y = nc.ccdf(x0);
        y.grad();
        CHECK_CLOSE("nccs ccdf(0) value", y.val(), 1.0, 1e-15);
        CHECK_CLOSE("nccs ccdf(0) d/ddf", df.adj(), 0.0, 0.0);
        CHECK_CLOSE("nccs ccdf(0) d/dncp", ncp.adj(), 0.0, 0.0);
        CHECK_CLOSE("nccs ccdf(0) d/dx", x0.adj(), 0.0, 0.0);

        var dfLog = 3.0;
        var ncpLog = 2.0;
        var xLog = 0.0;
        const quantape::math::NoncentralChiSquare<var> ncLog{dfLog, ncpLog};
        var yLog = ncLog.logCcdf(xLog);
        yLog.grad();
        CHECK_CLOSE("nccs logccdf(0) value", yLog.val(), 0.0, 0.0);
        CHECK_CLOSE("nccs logccdf(0) d/ddf", dfLog.adj(), 0.0, 0.0);
        CHECK_CLOSE("nccs logccdf(0) d/dncp", ncpLog.adj(), 0.0, 0.0);
        CHECK_CLOSE("nccs logccdf(0) d/dx", xLog.adj(), 0.0, 0.0);
    }
    {
        // x < 0: zero cdf pack -> ccdf = 1 / logCcdf = 0 with zero adjoints.
        var df = 3.0;
        var ncp = 2.0;
        var xNeg = -1.0;
        const quantape::math::NoncentralChiSquare<var> nc{df, ncp};
        var y = nc.ccdf(xNeg);
        y.grad();
        CHECK_CLOSE("nccs ccdf(-1) value", y.val(), 1.0, 0.0);
        CHECK_CLOSE("nccs ccdf(-1) d/ddf", df.adj(), 0.0, 0.0);
        CHECK_CLOSE("nccs ccdf(-1) d/dncp", ncp.adj(), 0.0, 0.0);
        CHECK_CLOSE("nccs ccdf(-1) d/dx", xNeg.adj(), 0.0, 0.0);

        var xLog = -1.0;
        var yLog = quantape::math::NoncentralChiSquare<var>{df, ncp}.logCcdf(xLog);
        yLog.grad();
        CHECK_CLOSE("nccs logccdf(-1) value", yLog.val(), 0.0, 0.0);
        CHECK_CLOSE("nccs logccdf(-1) d/dx", xLog.adj(), 0.0, 0.0);
    }
    {
        // Deep upper tail: the double ccdf underflows to 0 while the log-space
        // accessor stays finite; the quotient cannot be formed, so the route
        // documents the zero-push branch and keeps the accessor's value.
        const quantape::math::NoncentralChiSquare<double> deep{3.0, 500.0};
        CHECK_CLOSE("nccs deep ccdf underflow", deep.ccdf(5000.0), 0.0, 0.0);
        var df = 3.0;
        var ncp = 500.0;
        var x = 5000.0;
        const quantape::math::NoncentralChiSquare<var> nc{df, ncp};
        var y = nc.logCcdf(x);
        y.grad();
        CHECK_CLOSE("nccs logccdf deep value", y.val(), deep.logCcdf(5000.0), 0.0);
        CHECK_CLOSE("nccs logccdf deep d/ddf", df.adj(), 0.0, 0.0);
        CHECK_CLOSE("nccs logccdf deep d/dncp", ncp.adj(), 0.0, 0.0);
        CHECK_CLOSE("nccs logccdf deep d/dx", x.adj(), 0.0, 0.0);
    }
    {
        // Invalid parameters: NaN accessor values; ccdf keeps the generic
        // mixture's NaN, logCcdf pushes nothing (fresh leaves: the ccdf grad
        // above leaves NaN adjoints behind).
        var df = -1.0;
        var ncp = 2.0;
        var x = 2.5;
        const quantape::math::NoncentralChiSquare<var> nc{df, ncp};
        var yCcdf = nc.ccdf(x);
        yCcdf.grad();
        EXPECT_TRUE(std::isnan(yCcdf.val()));

        var dfLog = -1.0;
        var ncpLog = 2.0;
        var xLog = 2.5;
        const quantape::math::NoncentralChiSquare<var> ncLog{dfLog, ncpLog};
        var yLog = ncLog.logCcdf(xLog);
        yLog.grad();
        EXPECT_TRUE(std::isnan(yLog.val()));
        CHECK_CLOSE("nccs logccdf invalid d/ddf", dfLog.adj(), 0.0, 0.0);
        CHECK_CLOSE("nccs logccdf invalid d/dncp", ncpLog.adj(), 0.0, 0.0);
        CHECK_CLOSE("nccs logccdf invalid d/dx", xLog.adj(), 0.0, 0.0);
    }
}

TEST_F(DistributionAdTest, noncentralChiSquareTailFvarHessianMatchesAnalyticPack) {
    using fvar = stan::math::fvar<stan::math::var>;
    const double df0 = 3.0, ncp0 = 2.0, x0 = 2.5;
    const auto pack = quantape::math::detail::noncentralChiSquareCdfGradHess(df0, ncp0, x0);
    const quantape::math::NoncentralChiSquare<double> nc64{df0, ncp0};
    const double invG = 1.0 / nc64.ccdf(x0);
    const double gDf = -pack.gNu * invG;
    const double gNcp = -pack.gLambda * invG;
    const double gx = -pack.gx * invG;
    for (bool useLogCcdf : {false, true}) {
        SCOPED_TRACE(useLogCcdf);
        Eigen::VectorXd point(3);
        point << df0, ncp0, x0;
        double fx = 0.0;
        Eigen::VectorXd grad(3);
        Eigen::MatrixXd hess(3, 3);
        stan::math::hessian(NoncentralChiSquareTailFvarHessian{useLogCcdf}, point, fx, grad, hess);
        if (!useLogCcdf) {
            checkPackRoute("nccs fvar ccdf", fx, grad, hess, nc64.ccdf(x0),
                           {-pack.gNu, -pack.gLambda, -pack.gx},
                           {-pack.hNuNu, -pack.hNuLambda, -pack.hNuX, -pack.hNuLambda,
                            -pack.hLambdaLambda, -pack.hLambdaX, -pack.hNuX, -pack.hLambdaX,
                            -pack.hxx});
        } else {
            checkPackRoute("nccs fvar logccdf", fx, grad, hess, nc64.logCcdf(x0), {gDf, gNcp, gx},
                           {-pack.hNuNu * invG - gDf * gDf, -pack.hNuLambda * invG - gDf * gNcp,
                            -pack.hNuX * invG - gDf * gx, -pack.hNuLambda * invG - gNcp * gDf,
                            -pack.hLambdaLambda * invG - gNcp * gNcp,
                            -pack.hLambdaX * invG - gNcp * gx, -pack.hNuX * invG - gx * gDf,
                            -pack.hLambdaX * invG - gx * gNcp, -pack.hxx * invG - gx * gx});
        }
    }
    {
        // Deep-tail fallback through hessian(): value exact, zero gradient and
        // Hessian (the zero-push branch).
        const quantape::math::NoncentralChiSquare<double> deep{3.0, 500.0};
        Eigen::VectorXd point(3);
        point << 3.0, 500.0, 5000.0;
        double fx = 0.0;
        Eigen::VectorXd grad(3);
        Eigen::MatrixXd hess(3, 3);
        stan::math::hessian(NoncentralChiSquareTailFvarHessian{true}, point, fx, grad, hess);
        CHECK_CLOSE("nccs fvar logccdf deep value", fx, deep.logCcdf(5000.0), 0.0);
        for (int i = 0; i < 3; ++i) {
            CHECK_CLOSE("nccs fvar logccdf deep grad", grad(i), 0.0, 0.0);
            for (int j = 0; j < 3; ++j) {
                CHECK_CLOSE("nccs fvar logccdf deep hess", hess(i, j), 0.0, 0.0);
            }
        }
    }
    {
        fvar df(df0, 1.0), ncp(ncp0, 0.0), x(x0, 0.0);
        const quantape::math::NoncentralChiSquare<fvar> nc{df, ncp};
        auto* stack = stan::math::ChainableStack::instance_;
        const std::size_t before = stack->var_stack_.size();
        fvar y = nc.ccdf(x);
        const std::size_t nodes = stack->var_stack_.size() - before;
        EXPECT_EQ(nodes, std::size_t(2));
        RecordProperty("fvar_nccs_ccdf_nodes", static_cast<int>(nodes));
        const std::size_t beforeLog = stack->var_stack_.size();
        fvar yLog = nc.logCcdf(x);
        const std::size_t logNodes = stack->var_stack_.size() - beforeLog;
        EXPECT_EQ(logNodes, std::size_t(2));
        RecordProperty("fvar_nccs_logccdf_nodes", static_cast<int>(logNodes));
        (void)y;
        (void)yLog;
    }
}

TEST_F(DistributionAdTest, noncentralBetaTailCallbackVarMatchesAnalyticPack) {
    const quantape::math::NoncentralBeta<double> nb64{2.0, 3.5, 1.5};
    const auto cdfPack = quantape::math::detail::noncentralBetaCdfGradHess(2.0, 3.5, 1.5, 0.4);
    const auto pdfPack = quantape::math::detail::noncentralBetaPdfGradHess(2.0, 3.5, 1.5, 0.4);
    const double invF = 1.0 / pdfPack.value;
    {
        var alpha = 2.0;
        var beta = 3.5;
        var ncp = 1.5;
        var x = 0.4;
        const quantape::math::NoncentralBeta<var> nb{alpha, beta, ncp};
        auto* stack = stan::math::ChainableStack::instance_;
        const std::size_t before = stack->var_stack_.size();
        var y = nb.ccdf(x);
        const std::size_t nodes = stack->var_stack_.size() - before;
        y.grad();
        checkVarRoute("ncb ccdf", y.val(), {alpha.adj(), beta.adj(), ncp.adj(), x.adj()},
                      nb64.ccdf(0.4),
                      {-cdfPack.gAlpha, -cdfPack.gBeta, -cdfPack.gNcp, -cdfPack.gx});
        EXPECT_EQ(nodes, std::size_t(1));
        RecordProperty("var_ncb_ccdf_nodes", static_cast<int>(nodes));
    }
    {
        var alpha = 2.0;
        var beta = 3.5;
        var ncp = 1.5;
        var x = 0.4;
        const quantape::math::NoncentralBeta<var> nb{alpha, beta, ncp};
        auto* stack = stan::math::ChainableStack::instance_;
        const std::size_t before = stack->var_stack_.size();
        var y = nb.logPdf(x);
        const std::size_t nodes = stack->var_stack_.size() - before;
        y.grad();
        checkVarRoute(
            "ncb logpdf", y.val(), {alpha.adj(), beta.adj(), ncp.adj(), x.adj()}, nb64.logPdf(0.4),
            {pdfPack.gAlpha * invF, pdfPack.gBeta * invF, pdfPack.gNcp * invF, pdfPack.gx * invF});
        EXPECT_EQ(nodes, std::size_t(1));
        RecordProperty("var_ncb_logpdf_nodes", static_cast<int>(nodes));
    }
    {
        // ncp = 0 reduces exactly to the central-beta pack entries.
        var alpha = 2.0;
        var beta = 3.5;
        var ncp = 0.0;
        var x = 0.4;
        const quantape::math::NoncentralBeta<var> nb{alpha, beta, ncp};
        var y = nb.logPdf(x);
        y.grad();
        const auto pack = quantape::math::detail::noncentralBetaPdfGradHess(2.0, 3.5, 0.0, 0.4);
        const double inv = 1.0 / pack.value;
        const quantape::math::NoncentralBeta<double> nb64Zero{2.0, 3.5, 0.0};
        checkVarRoute("ncb logpdf ncp=0", y.val(), {alpha.adj(), beta.adj(), ncp.adj(), x.adj()},
                      nb64Zero.logPdf(0.4),
                      {pack.gAlpha * inv, pack.gBeta * inv, pack.gNcp * inv, pack.gx * inv});
    }
    {
        // Outside [0, 1]: ccdf constant 1 / 0, logPdf -Inf; zero pushes.
        const quantape::math::NoncentralBeta<double> nbLo{2.0, 3.5, 1.5};
        var alpha = 2.0;
        var beta = 3.5;
        var ncp = 1.5;
        var xLo = -0.5;
        const quantape::math::NoncentralBeta<var> nb{alpha, beta, ncp};
        var yLo = nb.ccdf(xLo);
        yLo.grad();
        CHECK_CLOSE("ncb ccdf(-0.5) value", yLo.val(), nbLo.ccdf(-0.5), 0.0);
        CHECK_CLOSE("ncb ccdf(-0.5) d/dalpha", alpha.adj(), 0.0, 0.0);
        CHECK_CLOSE("ncb ccdf(-0.5) d/dbeta", beta.adj(), 0.0, 0.0);
        CHECK_CLOSE("ncb ccdf(-0.5) d/dncp", ncp.adj(), 0.0, 0.0);
        CHECK_CLOSE("ncb ccdf(-0.5) d/dx", xLo.adj(), 0.0, 0.0);

        var xLog = -0.5;
        const quantape::math::NoncentralBeta<var> nbLog{alpha, beta, ncp};
        var yLog = nbLog.logPdf(xLog);
        yLog.grad();
        CHECK_CLOSE("ncb logpdf(-0.5) value", yLog.val(), nbLo.logPdf(-0.5), 0.0);
        CHECK_CLOSE("ncb logpdf(-0.5) d/dx", xLog.adj(), 0.0, 0.0);

        var xHi = 1.5;
        const quantape::math::NoncentralBeta<var> nbHi{alpha, beta, ncp};
        var yHi = nbHi.ccdf(xHi);
        yHi.grad();
        CHECK_CLOSE("ncb ccdf(1.5) value", yHi.val(), nbLo.ccdf(1.5), 0.0);
        CHECK_CLOSE("ncb ccdf(1.5) d/dx", xHi.adj(), 0.0, 0.0);
    }
    {
        // x = 0 with alpha > 1: the density (and pack value) is exactly zero
        // while the log-space accessor is -Inf -> zero pushes documented.
        const quantape::math::NoncentralBeta<double> nb64Edge{2.0, 3.5, 1.5};
        var alpha = 2.0;
        var beta = 3.5;
        var ncp = 1.5;
        var x0 = 0.0;
        const quantape::math::NoncentralBeta<var> nb{alpha, beta, ncp};
        var y = nb.logPdf(x0);
        y.grad();
        CHECK_CLOSE("ncb logpdf(0) value", y.val(), nb64Edge.logPdf(0.0), 0.0);
        EXPECT_TRUE(std::isinf(y.val()) && y.val() < 0.0);
        CHECK_CLOSE("ncb logpdf(0) d/dalpha", alpha.adj(), 0.0, 0.0);
        CHECK_CLOSE("ncb logpdf(0) d/dbeta", beta.adj(), 0.0, 0.0);
        CHECK_CLOSE("ncb logpdf(0) d/dncp", ncp.adj(), 0.0, 0.0);
        CHECK_CLOSE("ncb logpdf(0) d/dx", x0.adj(), 0.0, 0.0);
    }
    {
        // Invalid parameters: NaN accessors, no NaN leaks from the zero-density
        // guard only where the pack is genuinely NaN (logPdf pushes NaN there).
        var alpha = -1.0;
        var beta = 3.5;
        var ncp = 1.5;
        var x = 0.4;
        const quantape::math::NoncentralBeta<var> nb{alpha, beta, ncp};
        var yCcdf = nb.ccdf(x);
        yCcdf.grad();
        EXPECT_TRUE(std::isnan(yCcdf.val()));
        var xLog = 0.4;
        const quantape::math::NoncentralBeta<var> nbLog{alpha, beta, ncp};
        var yLog = nbLog.logPdf(xLog);
        yLog.grad();
        EXPECT_TRUE(std::isnan(yLog.val()));
    }
}

TEST_F(DistributionAdTest, noncentralBetaTailFvarHessianMatchesAnalyticPack) {
    const double alpha0 = 2.0, beta0 = 3.5, ncp0 = 1.5, x0 = 0.4;
    const quantape::math::NoncentralBeta<double> nb64{alpha0, beta0, ncp0};
    const auto cdfPack = quantape::math::detail::noncentralBetaCdfGradHess(alpha0, beta0, ncp0, x0);
    const auto pdfPack = quantape::math::detail::noncentralBetaPdfGradHess(alpha0, beta0, ncp0, x0);
    const double invF = 1.0 / pdfPack.value;
    const double gA = pdfPack.gAlpha * invF;
    const double gB = pdfPack.gBeta * invF;
    const double gN = pdfPack.gNcp * invF;
    const double gx = pdfPack.gx * invF;
    for (bool useLogPdf : {false, true}) {
        SCOPED_TRACE(useLogPdf);
        Eigen::VectorXd point(4);
        point << alpha0, beta0, ncp0, x0;
        double fx = 0.0;
        Eigen::VectorXd grad(4);
        Eigen::MatrixXd hess(4, 4);
        stan::math::hessian(NoncentralBetaTailFvarHessian{useLogPdf}, point, fx, grad, hess);
        if (!useLogPdf) {
            checkPackRoute("ncb fvar ccdf", fx, grad, hess, nb64.ccdf(x0),
                           {-cdfPack.gAlpha, -cdfPack.gBeta, -cdfPack.gNcp, -cdfPack.gx},
                           {-cdfPack.hAA, -cdfPack.hAB, -cdfPack.hANcp, -cdfPack.hAx, -cdfPack.hAB,
                            -cdfPack.hBB, -cdfPack.hBNcp, -cdfPack.hBx, -cdfPack.hANcp,
                            -cdfPack.hBNcp, -cdfPack.hNcpNcp, -cdfPack.hNcpX, -cdfPack.hAx,
                            -cdfPack.hBx, -cdfPack.hNcpX, -cdfPack.hxx});
        } else {
            checkPackRoute("ncb fvar logpdf", fx, grad, hess, nb64.logPdf(x0), {gA, gB, gN, gx},
                           {pdfPack.hAA * invF - gA * gA, pdfPack.hAB * invF - gA * gB,
                            pdfPack.hANcp * invF - gA * gN, pdfPack.hAx * invF - gA * gx,
                            pdfPack.hAB * invF - gB * gA, pdfPack.hBB * invF - gB * gB,
                            pdfPack.hBNcp * invF - gB * gN, pdfPack.hBx * invF - gB * gx,
                            pdfPack.hANcp * invF - gN * gA, pdfPack.hBNcp * invF - gN * gB,
                            pdfPack.hNcpNcp * invF - gN * gN, pdfPack.hNcpX * invF - gN * gx,
                            pdfPack.hAx * invF - gx * gA, pdfPack.hBx * invF - gx * gB,
                            pdfPack.hNcpX * invF - gx * gN, pdfPack.hxx * invF - gx * gx});
        }
    }
    {
        stan::math::fvar<var> alpha(alpha0, 1.0), beta(beta0, 0.0), ncp(ncp0, 0.0), x(x0, 0.0);
        const quantape::math::NoncentralBeta<stan::math::fvar<var>> nb{alpha, beta, ncp};
        auto* stack = stan::math::ChainableStack::instance_;
        const std::size_t before = stack->var_stack_.size();
        auto y = nb.ccdf(x);
        const std::size_t nodes = stack->var_stack_.size() - before;
        EXPECT_EQ(nodes, std::size_t(2));
        RecordProperty("fvar_ncb_ccdf_nodes", static_cast<int>(nodes));
        const std::size_t beforeLog = stack->var_stack_.size();
        auto yLog = nb.logPdf(x);
        const std::size_t logNodes = stack->var_stack_.size() - beforeLog;
        EXPECT_EQ(logNodes, std::size_t(2));
        RecordProperty("fvar_ncb_logpdf_nodes", static_cast<int>(logNodes));
        (void)y;
        (void)yLog;
    }
}

TEST_F(DistributionAdTest, noncentralFTailCallbackVarMatchesAnalyticPack) {
    const quantape::math::NoncentralF<double> nf64{5.0, 7.0, 1.5};
    const auto cdfPack = quantape::math::detail::noncentralFCdfGradHess(5.0, 7.0, 1.5, 1.2);
    const auto pdfPack = quantape::math::detail::noncentralFPdfGradHess(5.0, 7.0, 1.5, 1.2);
    const double invF = 1.0 / pdfPack.value;
    {
        var nu1 = 5.0;
        var nu2 = 7.0;
        var ncp = 1.5;
        var y0 = 1.2;
        const quantape::math::NoncentralF<var> nf{nu1, nu2, ncp};
        auto* stack = stan::math::ChainableStack::instance_;
        const std::size_t before = stack->var_stack_.size();
        var y = nf.ccdf(y0);
        const std::size_t nodes = stack->var_stack_.size() - before;
        y.grad();
        checkVarRoute("ncf ccdf", y.val(), {nu1.adj(), nu2.adj(), ncp.adj(), y0.adj()},
                      nf64.ccdf(1.2), {-cdfPack.gNu1, -cdfPack.gNu2, -cdfPack.gNcp, -cdfPack.gy});
        EXPECT_EQ(nodes, std::size_t(1));
        RecordProperty("var_ncf_ccdf_nodes", static_cast<int>(nodes));
    }
    {
        var nu1 = 5.0;
        var nu2 = 7.0;
        var ncp = 1.5;
        var y0 = 1.2;
        const quantape::math::NoncentralF<var> nf{nu1, nu2, ncp};
        auto* stack = stan::math::ChainableStack::instance_;
        const std::size_t before = stack->var_stack_.size();
        var y = nf.logPdf(y0);
        const std::size_t nodes = stack->var_stack_.size() - before;
        y.grad();
        checkVarRoute(
            "ncf logpdf", y.val(), {nu1.adj(), nu2.adj(), ncp.adj(), y0.adj()}, nf64.logPdf(1.2),
            {pdfPack.gNu1 * invF, pdfPack.gNu2 * invF, pdfPack.gNcp * invF, pdfPack.gy * invF});
        EXPECT_EQ(nodes, std::size_t(1));
        RecordProperty("var_ncf_logpdf_nodes", static_cast<int>(nodes));
    }
    {
        // ncp = 0 reduction and the y < 0 / y = +Inf zero branches.
        var nu1 = 5.0;
        var nu2 = 7.0;
        var ncp = 0.0;
        var y0 = 1.2;
        const quantape::math::NoncentralF<var> nf{nu1, nu2, ncp};
        var y = nf.ccdf(y0);
        y.grad();
        const auto pack = quantape::math::detail::noncentralFCdfGradHess(5.0, 7.0, 0.0, 1.2);
        const quantape::math::NoncentralF<double> f64Zero{5.0, 7.0, 0.0};
        checkVarRoute("ncf ccdf ncp=0", y.val(), {nu1.adj(), nu2.adj(), ncp.adj(), y0.adj()},
                      f64Zero.ccdf(1.2), {-pack.gNu1, -pack.gNu2, -pack.gNcp, -pack.gy});

        var nu1Lo = 5.0;
        var nu2Lo = 7.0;
        var ncpLo = 1.5;
        var yLo = -0.5;
        const quantape::math::NoncentralF<var> nfLo{nu1Lo, nu2Lo, ncpLo};
        var yCcdf = nfLo.ccdf(yLo);
        yCcdf.grad();
        CHECK_CLOSE("ncf ccdf(-0.5) value", yCcdf.val(), nf64.ccdf(-0.5), 0.0);
        CHECK_CLOSE("ncf ccdf(-0.5) d/dnu1", nu1Lo.adj(), 0.0, 0.0);
        CHECK_CLOSE("ncf ccdf(-0.5) d/dy", yLo.adj(), 0.0, 0.0);

        var yLog = -0.5;
        const quantape::math::NoncentralF<var> nfLog{nu1Lo, nu2Lo, ncpLo};
        var yLp = nfLog.logPdf(yLog);
        yLp.grad();
        CHECK_CLOSE("ncf logpdf(-0.5) value", yLp.val(), nf64.logPdf(-0.5), 0.0);
        CHECK_CLOSE("ncf logpdf(-0.5) d/dy", yLog.adj(), 0.0, 0.0);

        var yInf = std::numeric_limits<double>::infinity();
        const quantape::math::NoncentralF<var> nfInf{nu1Lo, nu2Lo, ncpLo};
        var yInfOut = nfInf.logPdf(yInf);
        yInfOut.grad();
        CHECK_CLOSE("ncf logpdf(+Inf) value", yInfOut.val(),
                    nf64.logPdf(std::numeric_limits<double>::infinity()), 0.0);
        CHECK_CLOSE("ncf logpdf(+Inf) d/dy", yInf.adj(), 0.0, 0.0);
    }
    {
        // y = 0 with nu1 > 2: the beta density (and F pack value) is exactly
        // zero while the log-space accessor is -Inf -> documented zero pushes.
        const quantape::math::NoncentralF<double> f64Edge{5.0, 7.0, 1.5};
        var nu1 = 5.0;
        var nu2 = 7.0;
        var ncp = 1.5;
        var y0 = 0.0;
        const quantape::math::NoncentralF<var> nf{nu1, nu2, ncp};
        var y = nf.logPdf(y0);
        y.grad();
        CHECK_CLOSE("ncf logpdf(0) value", y.val(), f64Edge.logPdf(0.0), 0.0);
        EXPECT_TRUE(std::isinf(y.val()) && y.val() < 0.0);
        CHECK_CLOSE("ncf logpdf(0) d/dnu1", nu1.adj(), 0.0, 0.0);
        CHECK_CLOSE("ncf logpdf(0) d/dnu2", nu2.adj(), 0.0, 0.0);
        CHECK_CLOSE("ncf logpdf(0) d/dncp", ncp.adj(), 0.0, 0.0);
        CHECK_CLOSE("ncf logpdf(0) d/dy", y0.adj(), 0.0, 0.0);
    }
    {
        // Invalid parameters: NaN accessor values.
        var nu1 = -1.0;
        var nu2 = 7.0;
        var ncp = 1.5;
        var y0 = 1.2;
        const quantape::math::NoncentralF<var> nf{nu1, nu2, ncp};
        var yCcdf = nf.ccdf(y0);
        yCcdf.grad();
        EXPECT_TRUE(std::isnan(yCcdf.val()));
        var yLog = 1.2;
        const quantape::math::NoncentralF<var> nfLog{nu1, nu2, ncp};
        var yLp = nfLog.logPdf(yLog);
        yLp.grad();
        EXPECT_TRUE(std::isnan(yLp.val()));
    }
}

TEST_F(DistributionAdTest, noncentralFTailFvarHessianMatchesAnalyticPack) {
    const double nu1_0 = 5.0, nu2_0 = 7.0, ncp0 = 1.5, y0 = 1.2;
    const quantape::math::NoncentralF<double> nf64{nu1_0, nu2_0, ncp0};
    const auto cdfPack = quantape::math::detail::noncentralFCdfGradHess(nu1_0, nu2_0, ncp0, y0);
    const auto pdfPack = quantape::math::detail::noncentralFPdfGradHess(nu1_0, nu2_0, ncp0, y0);
    const double invF = 1.0 / pdfPack.value;
    const double gNu1 = pdfPack.gNu1 * invF;
    const double gNu2 = pdfPack.gNu2 * invF;
    const double gNcp = pdfPack.gNcp * invF;
    const double gy = pdfPack.gy * invF;
    for (bool useLogPdf : {false, true}) {
        SCOPED_TRACE(useLogPdf);
        Eigen::VectorXd point(4);
        point << nu1_0, nu2_0, ncp0, y0;
        double fx = 0.0;
        Eigen::VectorXd grad(4);
        Eigen::MatrixXd hess(4, 4);
        stan::math::hessian(NoncentralFTailFvarHessian{useLogPdf}, point, fx, grad, hess);
        if (!useLogPdf) {
            checkPackRoute("ncf fvar ccdf", fx, grad, hess, nf64.ccdf(y0),
                           {-cdfPack.gNu1, -cdfPack.gNu2, -cdfPack.gNcp, -cdfPack.gy},
                           {-cdfPack.hNu1Nu1, -cdfPack.hNu1Nu2, -cdfPack.hNu1Ncp, -cdfPack.hNu1Y,
                            -cdfPack.hNu1Nu2, -cdfPack.hNu2Nu2, -cdfPack.hNu2Ncp, -cdfPack.hNu2Y,
                            -cdfPack.hNu1Ncp, -cdfPack.hNu2Ncp, -cdfPack.hNcpNcp, -cdfPack.hNcpY,
                            -cdfPack.hNu1Y, -cdfPack.hNu2Y, -cdfPack.hNcpY, -cdfPack.hyy});
        } else {
            checkPackRoute(
                "ncf fvar logpdf", fx, grad, hess, nf64.logPdf(y0), {gNu1, gNu2, gNcp, gy},
                {pdfPack.hNu1Nu1 * invF - gNu1 * gNu1, pdfPack.hNu1Nu2 * invF - gNu1 * gNu2,
                 pdfPack.hNu1Ncp * invF - gNu1 * gNcp, pdfPack.hNu1Y * invF - gNu1 * gy,
                 pdfPack.hNu1Nu2 * invF - gNu2 * gNu1, pdfPack.hNu2Nu2 * invF - gNu2 * gNu2,
                 pdfPack.hNu2Ncp * invF - gNu2 * gNcp, pdfPack.hNu2Y * invF - gNu2 * gy,
                 pdfPack.hNu1Ncp * invF - gNcp * gNu1, pdfPack.hNu2Ncp * invF - gNcp * gNu2,
                 pdfPack.hNcpNcp * invF - gNcp * gNcp, pdfPack.hNcpY * invF - gNcp * gy,
                 pdfPack.hNu1Y * invF - gy * gNu1, pdfPack.hNu2Y * invF - gy * gNu2,
                 pdfPack.hNcpY * invF - gy * gNcp, pdfPack.hyy * invF - gy * gy});
        }
    }
    {
        stan::math::fvar<var> nu1(nu1_0, 1.0), nu2(nu2_0, 0.0), ncp(ncp0, 0.0), y(y0, 0.0);
        const quantape::math::NoncentralF<stan::math::fvar<var>> nf{nu1, nu2, ncp};
        auto* stack = stan::math::ChainableStack::instance_;
        const std::size_t before = stack->var_stack_.size();
        auto f = nf.ccdf(y);
        const std::size_t nodes = stack->var_stack_.size() - before;
        EXPECT_EQ(nodes, std::size_t(2));
        RecordProperty("fvar_ncf_ccdf_nodes", static_cast<int>(nodes));
        const std::size_t beforeLog = stack->var_stack_.size();
        auto fLog = nf.logPdf(y);
        const std::size_t logNodes = stack->var_stack_.size() - beforeLog;
        EXPECT_EQ(logNodes, std::size_t(2));
        RecordProperty("fvar_ncf_logpdf_nodes", static_cast<int>(logNodes));
        (void)f;
        (void)fLog;
    }
}

// ---------------------------------------------------------------------------
// NoncentralChiSquare density routes: pdf / logPdf for var and fvar<var>
// (mixture density pack; the generic accessors are double-only).
// ---------------------------------------------------------------------------

namespace {

struct NccsPdfFvarHessian {
    bool logForm;
    template <class T>
    T operator()(const Eigen::Matrix<T, Eigen::Dynamic, 1>& v) const {
        const quantape::math::NoncentralChiSquare<T> nc{v(0), v(1)};
        return logForm ? nc.logPdf(v(2)) : nc.pdf(v(2));
    }
};

} // namespace

TEST_F(DistributionAdTest, noncentralChiSquarePdfHessianMatchesPack) {
    Eigen::VectorXd point(3);
    point << 3.0, 2.0, 2.5;
    const auto pack = quantape::math::NoncentralChiSquare<double>{3.0, 2.0}.pdfGradHess(2.5);
    for (bool logForm : {false, true}) {
        double fx = 0.0;
        Eigen::VectorXd grad(3);
        Eigen::MatrixXd hessian(3, 3);
        stan::math::hessian(NccsPdfFvarHessian{logForm}, point, fx, grad, hessian);
        if (!logForm) {
            CHECK_CLOSE_REL("pdf value", fx, pack.value, 1e-9);
            CHECK_CLOSE_REL("pdf gNu", grad(0), pack.gNu, 1e-8);
            CHECK_CLOSE_REL("pdf gLambda", grad(1), pack.gLambda, 1e-8);
            CHECK_CLOSE_REL("pdf gx", grad(2), pack.gx, 1e-8);
            CHECK_CLOSE_REL("pdf hNuNu", hessian(0, 0), pack.hNuNu, 1e-7);
            CHECK_CLOSE_REL("pdf hNuLambda", hessian(0, 1), pack.hNuLambda, 1e-7);
            CHECK_CLOSE_REL("pdf hNuX", hessian(0, 2), pack.hNuX, 1e-7);
            CHECK_CLOSE_REL("pdf hLambdaLambda", hessian(1, 1), pack.hLambdaLambda, 1e-7);
            CHECK_CLOSE_REL("pdf hLambdaX", hessian(1, 2), pack.hLambdaX, 1e-7);
            CHECK_CLOSE_REL("pdf hxx", hessian(2, 2), pack.hxx, 1e-7);
        } else {
            const double inv = 1.0 / pack.value;
            const double g[3] = {pack.gNu * inv, pack.gLambda * inv, pack.gx * inv};
            const double hv[3][3] = {{pack.hNuNu, pack.hNuLambda, pack.hNuX},
                                     {pack.hNuLambda, pack.hLambdaLambda, pack.hLambdaX},
                                     {pack.hNuX, pack.hLambdaX, pack.hxx}};
            CHECK_CLOSE_REL("logpdf value", fx, std::log(pack.value), 1e-13);
            for (int i = 0; i < 3; ++i) {
                CHECK_CLOSE_REL("logpdf grad", grad(i), g[i], 1e-8);
            }
            for (int i = 0; i < 3; ++i) {
                for (int j = 0; j < 3; ++j) {
                    CHECK_CLOSE_REL("logpdf hess", hessian(i, j), hv[i][j] * inv - g[i] * g[j],
                                    1e-7);
                }
            }
        }
    }
}

TEST_F(DistributionAdTest, noncentralChiSquarePdfNodeCounts) {
    auto* stack = stan::math::ChainableStack::instance_;
    using stan::math::fvar;
    using var = stan::math::var;
    {
        const std::size_t before = stack->var_stack_.size();
        var df = 3.0;
        var ncp = 2.0;
        var x = 2.5;
        var y = quantape::math::NoncentralChiSquare<var>{df, ncp}.pdf(x);
        const std::size_t nodes = stack->var_stack_.size() - before;
        y.grad();
        EXPECT_EQ(nodes, std::size_t(1));
        RecordProperty("var_nccs_pdf_nodes", static_cast<int>(nodes));
    }
    {
        const std::size_t before = stack->var_stack_.size();
        fvar<var> df = 3.0;
        fvar<var> ncp = 2.0;
        fvar<var> x = 2.5;
        fvar<var> y = quantape::math::NoncentralChiSquare<fvar<var>>{df, ncp}.pdf(x);
        const std::size_t nodes = stack->var_stack_.size() - before;
        y.d_.grad();
        EXPECT_EQ(nodes, std::size_t(2));
        RecordProperty("fvar_nccs_pdf_nodes", static_cast<int>(nodes));
    }
    {
        const std::size_t before = stack->var_stack_.size();
        var df = 3.0;
        var ncp = 2.0;
        var x = 2.5;
        var y = quantape::math::NoncentralChiSquare<var>{df, ncp}.logPdf(x);
        const std::size_t nodes = stack->var_stack_.size() - before;
        y.grad();
        EXPECT_EQ(nodes, std::size_t(1));
        RecordProperty("var_nccs_logpdf_nodes", static_cast<int>(nodes));
    }
    {
        const std::size_t before = stack->var_stack_.size();
        fvar<var> df = 3.0;
        fvar<var> ncp = 2.0;
        fvar<var> x = 2.5;
        fvar<var> y = quantape::math::NoncentralChiSquare<fvar<var>>{df, ncp}.logPdf(x);
        const std::size_t nodes = stack->var_stack_.size() - before;
        y.d_.grad();
        EXPECT_EQ(nodes, std::size_t(2));
        RecordProperty("fvar_nccs_logpdf_nodes", static_cast<int>(nodes));
    }
}
