// test_numeraire.cpp — numeraire view: deterministic limit, collateral
// deflation (Z = V/N = V*D, N = 1/D), zero-vol martingality, rejection rules
// and AD state gradients
#include "quantape/math/StanMath.h" // must be first (Eigen plugin ordering)

#include "quantape/markets/Curves/Curve.h"
#include "quantape/markets/Curves/DiscountCurve.h"
#include "quantape/markets/Curves/SpreadCurve.h"
#include "quantape/sde/Measure.h"
#include "quantape/sde/MeasureAdjustment.h"
#include "quantape/sde/Numeraire.h"
#include "quantape/sde/StateCurveMap.h"

#include <array>
#include <cmath>
#include <cstddef>
#include <initializer_list>
#include <memory>
#include <span>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "support/GtestSupport.h"
#include "support/StanTapeFixture.h"

using namespace quantape;
using stan::math::fvar;
using stan::math::var;

namespace {

using markets::DiscountCurve;
using markets::InterpolationScheme;
using markets::InterpolationSpace;
using markets::SpreadCurve;

constexpr std::size_t kNObs = 4;
constexpr std::size_t kNPay = 8;
constexpr double kStep = 1e-6;
constexpr double kTol = 1e-12;

DiscountCurve<double> makeOisCurve() {
    return DiscountCurve<double>({0.0, 1.0, 2.0, 3.0, 5.0}, {0.0, 0.02, 0.025, 0.03, 0.035},
                                 InterpolationSpace::LogDiscount, InterpolationScheme::Linear);
}

SpreadCurve<double> makeSpreadCurve(std::shared_ptr<const DiscountCurve<double>> parent) {
    return SpreadCurve<double>(std::move(parent), {0.0, 1.0, 2.0, 3.0, 5.0},
                               {0.0, 0.001, 0.0012, 0.0015, 0.002});
}

std::vector<double> obsTimes() {
    return {0.0, 0.5, 1.0, 2.0};
}

std::vector<double> payTimes() {
    return {0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0};
}

/// Two-factor loading block on the observation/payment grid:
/// B_0 = 0.04 * tau, B_1 = 0.015 * tau^2, rho = 0.35.
sde::GaussianStateCurve::Loadings makeLoadings() {
    sde::GaussianStateCurve::Loadings loadings;
    loadings.factor.assign(2, std::vector<double>(kNObs * kNPay, 0.0));
    loadings.correlation = {1.0, 0.35, 0.35, 1.0};
    const std::vector<double> obs = obsTimes();
    const std::vector<double> pay = payTimes();
    for (std::size_t k = 0; k < kNObs; ++k) {
        for (std::size_t j = 0; j < kNPay; ++j) {
            const double tau = pay[j] - obs[k];
            loadings.factor[0][k * kNPay + j] = 0.04 * tau;
            loadings.factor[1][k * kNPay + j] = 0.015 * tau * tau;
        }
    }
    return loadings;
}

sde::GaussianStateCurve makeMap() {
    return sde::GaussianStateCurve::fromCurve(makeOisCurve(), obsTimes(), payTimes(),
                                              makeLoadings());
}

/// Zero-vol (deterministic loading block) map: every bond is the base curve
/// ratio P0(T_j)/P0(t_k) exactly, so a zero state is a deterministic model.
sde::GaussianStateCurve makeDeterministicMap() {
    sde::GaussianStateCurve::Loadings loadings;
    loadings.factor.assign(2, std::vector<double>(kNObs * kNPay, 0.0));
    loadings.correlation = {1.0, 0.0, 0.0, 1.0};
    return sde::GaussianStateCurve::fromCurve(makeOisCurve(), obsTimes(), payTimes(),
                                              std::move(loadings));
}

sde::MeasureSpec riskNeutralSpec() {
    return sde::MeasureSpec{};
}

sde::MeasureSpec tForwardSpec(double forwardTime) {
    sde::MeasureSpec spec;
    spec.kind = sde::MeasureKind::TForward;
    spec.forwardTime = forwardTime;
    return spec;
}

sde::MeasureSpec annuitySpec(std::vector<double> weights) {
    sde::MeasureSpec spec;
    spec.kind = sde::MeasureKind::Annuity;
    spec.annuityWeights = std::move(weights);
    return spec;
}

sde::MeasureSpec collateralCurrencySpec() {
    sde::MeasureSpec spec;
    spec.kind = sde::MeasureKind::CollateralCurrency;
    return spec;
}

/// Deterministic stand-in for a canonical model: every measure except the
/// collateral-currency one is representable and a deterministic state carries
/// no drift correction.
class DummyMeasureAdjustment {
public:
    bool supports(const sde::MeasureSpec& measure) const noexcept {
        return measure.kind != sde::MeasureKind::CollateralCurrency;
    }

    template <typename Scalar>
    Scalar measureAdjustment(const sde::MeasureSpec&, double, std::span<const Scalar>) const {
        return Scalar(0.0);
    }
};

} // namespace

static_assert(sde::MeasureAdjustmentModel<DummyMeasureAdjustment, double>);
static_assert(sde::MeasureAdjustmentModel<DummyMeasureAdjustment, var>);
static_assert(sde::MeasureAdjustmentModel<DummyMeasureAdjustment, fvar<var>>);

TEST(Numeraire, zeroStateReproducesDeterministicCurves) {
    const std::span<const double> zeroState;
    const DiscountCurve<double> ois = makeOisCurve();
    const sde::Numeraire<DiscountCurve<double>> discountNumeraire(ois);
    // The numeraire is the collateral account N = 1/D, not the discount.
    for (const double t : {0.5, 1.0, 2.0, 3.5}) {
        CHECK_CLOSE("numeraire value is the inverse curve discount, bit-exact",
                    discountNumeraire.numeraireValue(t, zeroState), 1.0 / ois.discount(t), 0.0);
    }

    const auto parent = std::make_shared<const DiscountCurve<double>>(makeOisCurve());
    const SpreadCurve<double> spread = makeSpreadCurve(parent);
    const sde::Numeraire<SpreadCurve<double>> spreadNumeraire(spread);
    // rollRatio(t1,t2) = N(t1)/N(t2) = D(t2)/D(t1): the forward discount.
    CHECK_CLOSE("roll ratio is the forward discount D(t2)/D(t1), bit-exact",
                spreadNumeraire.rollRatio(1.0, 3.0, zeroState),
                spread.discount(3.0) / spread.discount(1.0), 0.0);
}

TEST(Numeraire, deflatedTimesNumeraireRecoversValue) {
    const std::span<const double> zeroState;
    const DiscountCurve<double> ois = makeOisCurve();
    const sde::Numeraire<DiscountCurve<double>> numeraire(ois);
    const double value = 250.0;
    const double t = 2.5;
    CHECK_CLOSE("deflated value times the account numeraire recovers the value",
                numeraire.deflated(value, t, zeroState) * numeraire.numeraireValue(t, zeroState),
                value, kTol);
}

TEST(Numeraire, crossMeasureDeterministicStateAgrees) {
    const DummyMeasureAdjustment model;
    const sde::GaussianStateCurve map = makeMap();
    const sde::Numeraire<sde::GaussianStateCurve> numeraire(map);
    const std::array<double, 2> stateValues{0.12, -0.07};
    const std::span<const double> state(stateValues);
    const double t = 3.0;
    const double payoff = 125.0;

    const std::array<sde::MeasureSpec, 3> measures{riskNeutralSpec(), tForwardSpec(t),
                                                   annuitySpec({0.25, 0.25, 0.25, 0.25})};
    const double reference = numeraire.deflated(payoff, t, state);
    for (const sde::MeasureSpec& measure : measures) {
        sde::requireMeasureSupport(model, measure);
        CHECK_CLOSE("deterministic state carries no measure adjustment",
                    model.measureAdjustment(measure, t, state), 0.0, 0.0);
        CHECK_CLOSE("deflated payoff is measure-independent for a deterministic state",
                    numeraire.deflated(payoff, t, state), reference, 0.0);
        CHECK_CLOSE("PV recovers the raw payoff under every measure",
                    reference * numeraire.numeraireValue(t, state), payoff, kTol);
    }
}

TEST(Numeraire, crossCurrencyCollateralRoundTripsWithDeterministicFx) {
    const DiscountCurve<double> eurOis = makeOisCurve();
    const DiscountCurve<double> usdOis =
        DiscountCurve<double>({0.0, 1.0, 2.0, 3.0, 5.0}, {0.0, 0.03, 0.032, 0.034, 0.036},
                              InterpolationSpace::LogDiscount, InterpolationScheme::Linear);
    const sde::Numeraire<DiscountCurve<double>> eurNumeraire(eurOis);
    const sde::Numeraire<DiscountCurve<double>> usdNumeraire(usdOis);
    const std::span<const double> state;
    const double t = 3.0;
    const double fxSpot = 1.087; // deterministic USD per EUR
    const double payoffEur = 100.0;

    sde::CollateralSpec collateral;
    collateral.key.currency = "EUR";
    collateral.key.role = markets::CurveRole::Discount;
    collateral.key.collateral = "USD";
    collateral.hasFxPair = true;
    collateral.fxPair = "EURUSD";
    ASSERT_TRUE(collateral.hasFxPair);

    // Covered interest parity with deterministic curves: the forward FX quotes
    // the ratio of the collateral-currency zero bonds, spot * D_usd/D_eur,
    // which is spot * N_eur/N_usd with the account numeraire N = 1/D.
    const double fxForward =
        fxSpot * eurNumeraire.numeraireValue(t, state) / usdNumeraire.numeraireValue(t, state);
    CHECK_CLOSE_REL("CIP forward FX is spot * D_usd(0,t)/D_eur(0,t)", fxForward,
                    fxSpot * usdOis.discount(t) / eurOis.discount(t), 1e-15);
    CHECK_CLOSE_REL("forward FX times the USD collateral numeraire equals spot times EUR",
                    fxForward * usdNumeraire.numeraireValue(t, state),
                    fxSpot * eurNumeraire.numeraireValue(t, state), 1e-15);

    const double valueUsd = payoffEur * fxForward;
    const double deflatedUsd = usdNumeraire.deflated(valueUsd, t, state);
    CHECK_CLOSE("USD collateral deflation recovers the USD value",
                deflatedUsd * usdNumeraire.numeraireValue(t, state), valueUsd, kTol);
    CHECK_CLOSE("collateral deflation and FX conversion round-trip to EUR",
                deflatedUsd * usdNumeraire.numeraireValue(t, state) / fxForward, payoffEur, kTol);
    CHECK_CLOSE("EUR numeraire deflation recovers the raw payoff",
                eurNumeraire.deflated(payoffEur, t, state) * eurNumeraire.numeraireValue(t, state),
                payoffEur, kTol);
}

TEST(Numeraire, deflationAppliesExactlyOneDiscountFactor) {
    const std::span<const double> state;
    const DiscountCurve<double> ois = makeOisCurve();
    const sde::Numeraire<DiscountCurve<double>> numeraire(ois);
    const double payoff = 250.0;
    const double t = 2.5;

    CHECK_CLOSE("N(0) is one, so the final PV needs no extra scaling",
                numeraire.numeraireValue(0.0, state), 1.0, 0.0);
    // Z(t) = V(t)/N(t) = V(t)*D(0,t): exactly one discount factor.
    CHECK_CLOSE("deflated payoff applies exactly one discount factor",
                numeraire.deflated(payoff, t, state), payoff * ois.discount(t), 0.0);
    CHECK_CLOSE("numeraire value is the reciprocal discount", numeraire.numeraireValue(t, state),
                1.0 / ois.discount(t), 0.0);
}

TEST(Numeraire, unitCashflowPricesToDiscount) {
    const std::span<const double> state;
    const DiscountCurve<double> ois = makeOisCurve();
    const sde::Numeraire<DiscountCurve<double>> numeraire(ois);
    // A unit cashflow at T has time-T value 1; its T-deflated value is the
    // risk-neutral PV D(0,T) (not 1/D(0,T), the inverted-units defect).
    for (const double t : {0.5, 1.0, 2.0, 2.5, 3.5, 5.0}) {
        CHECK_CLOSE("unit cashflow at T deflates to D(0,T), bit-exact",
                    numeraire.deflated(1.0, t, state), ois.discount(t), 0.0);
    }
}

// The gate the old convention failed: with the collateral account as the
// numeraire, a zero-vol model's deflated value is a martingale and prices to
// the trade's own discount curve. Every path carries the same state, so
// E[Z_k] = Z_k and the identity is exact.
TEST(Numeraire, zeroVolDeflatedValueIsMartingale) {
    const sde::GaussianStateCurve map = makeDeterministicMap();
    const sde::Numeraire<sde::GaussianStateCurve> numeraire(map);
    const std::array<double, 2> zeroStateValues{0.0, 0.0};
    const std::span<const double> state(zeroStateValues);
    const std::span<const double> obs = map.obsTimes();
    const std::span<const double> pay = map.payTimes();
    const std::size_t jT = 3; // T = 2.0, on both grids
    const double T = pay[jT];
    const double pv = makeOisCurve().discount(T);

    // Z(0) = V(0) = P(0,T) is the PV and needs no extra N(0) scaling.
    const double z0 = numeraire.deflated(map.bond(0, jT, state), 0.0, state);
    CHECK_CLOSE("Z(0) is the PV D(0,T), bit-exact", z0, pv, 0.0);

    for (std::size_t k = 0; k < obs.size(); ++k) {
        const double valueAtK = map.bond(k, jT, state); // P(t_k, T); 1 at T
        const double zk = numeraire.deflated(valueAtK, obs[k], state);
        CHECK_CLOSE("zero-vol deflated value is a martingale: Z_k = Z_0", zk, z0, 0.0);
        CHECK_CLOSE("zero-vol deflated value matches the discount curve", zk, pv, 0.0);
    }
    // The terminal point of the same path: P(T,T) = 1 deflates to D(0,T).
    CHECK_CLOSE("terminal P(T,T)=1 prices to D(0,T)",
                numeraire.deflated(map.bond(obs.size() - 1, jT, state), T, state), pv, 0.0);
}

TEST(MeasureAdjustment, requireMeasureSupportPassesForSupportedMeasure) {
    const DummyMeasureAdjustment model;
    EXPECT_NO_THROW(sde::requireMeasureSupport(model, riskNeutralSpec()));
    EXPECT_NO_THROW(sde::requireMeasureSupport(model, tForwardSpec(2.0)));
    EXPECT_NO_THROW(sde::requireMeasureSupport(model, annuitySpec({1.0})));
}

TEST(MeasureAdjustment, requireMeasureSupportThrowsForUnsupportedMeasure) {
    const DummyMeasureAdjustment model;
    const sde::MeasureSpec unsupported = collateralCurrencySpec();
    EXPECT_THROW(sde::requireMeasureSupport(model, unsupported), std::invalid_argument);
    try {
        sde::requireMeasureSupport(model, unsupported);
        ADD_FAILURE() << "unsupported measure was accepted";
    } catch (const std::invalid_argument& error) {
        EXPECT_NE(std::string(error.what()).find("CollateralCurrency"), std::string::npos);
    }
}

TEST(Numeraire, nullOrUnsetViewThrows) {
    using Ois = DiscountCurve<double>;
    EXPECT_THROW(sde::Numeraire<Ois>(static_cast<const Ois*>(nullptr)), std::invalid_argument);

    const sde::Numeraire<Ois> unset;
    const std::span<const double> zeroState;
    EXPECT_THROW(unset.numeraireValue(1.0, zeroState), std::invalid_argument);
    EXPECT_THROW(unset.numeraireValue(0.0, zeroState), std::invalid_argument);
    EXPECT_THROW(unset.deflated(1.0, 1.0, zeroState), std::invalid_argument);
    EXPECT_THROW(unset.deflated(1.0, 0.0, zeroState), std::invalid_argument);
    EXPECT_THROW(unset.rollRatio(1.0, 2.0, zeroState), std::invalid_argument);
    EXPECT_THROW(unset.rollRatio(0.0, 0.0, zeroState), std::invalid_argument);
}

class NumeraireAdTest : public StanTapeTest {};

TEST_F(NumeraireAdTest, varStateGradientMatchesCentralDifference) {
    const sde::GaussianStateCurve map = makeMap();
    const sde::Numeraire<sde::GaussianStateCurve> numeraire(map);
    const double t = 3.0;
    const double x0 = 0.12;
    const double x1 = -0.07;
    std::array<var, 2> state{var(x0), var(x1)};

    var y = numeraire.numeraireValue(t, std::span<const var>(state));
    y.grad();
    const double ad0 = state[0].adj();
    const double ad1 = state[1].adj();

    const auto eval = [&](double a, double b) {
        const std::array<double, 2> s{a, b};
        return numeraire.numeraireValue(t, std::span<const double>(s));
    };
    const double fd0 = (eval(x0 + kStep, x1) - eval(x0 - kStep, x1)) / (2.0 * kStep);
    const double fd1 = (eval(x0, x1 + kStep) - eval(x0, x1 - kStep)) / (2.0 * kStep);
    CHECK_CLOSE("numeraire state adjoint x0", ad0, fd0, 1e-8);
    CHECK_CLOSE("numeraire state adjoint x1", ad1, fd1, 1e-8);
}

TEST_F(NumeraireAdTest, fvarVarEvaluationIsFinite) {
    const sde::GaussianStateCurve map = makeMap();
    const sde::Numeraire<sde::GaussianStateCurve> numeraire(map);
    const double t = 3.0;
    const double x0 = 0.12;
    const double x1 = -0.07;
    const double v0 = 0.7;
    const double v1 = -0.4;
    std::array<fvar<var>, 2> state{fvar<var>(var(x0), v0), fvar<var>(var(x1), v1)};

    const fvar<var> y = numeraire.numeraireValue(t, std::span<const fvar<var>>(state));
    EXPECT_TRUE(std::isfinite(y.val_.val()));
    EXPECT_TRUE(std::isfinite(y.d_.val()));

    const std::array<double, 2> plain{x0, x1};
    CHECK_CLOSE("fvar value matches the double evaluation", y.val_.val(),
                numeraire.numeraireValue(t, std::span<const double>(plain)), 1e-15);
}

TEST_F(NumeraireAdTest, measureAdjustmentIsAdTemplated) {
    const DummyMeasureAdjustment model;
    const sde::MeasureSpec measure = tForwardSpec(2.0);
    std::array<var, 2> state{var(0.3), var(-0.2)};
    var correction = model.measureAdjustment(measure, 2.0, std::span<const var>(state));
    correction.grad();
    CHECK_CLOSE("deterministic adjustment value", correction.val(), 0.0, 0.0);
    CHECK_CLOSE("deterministic adjustment adjoint w.r.t. state", state[0].adj(), 0.0, 0.0);
}
