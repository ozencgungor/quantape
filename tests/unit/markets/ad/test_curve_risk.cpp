/**
 * @file test_curve_risk.cpp
 * @brief Quote-risk, stack-risk and gamma transforms for the curve stack
 *
 * Double-only gates: node-weight partitions and finite differences, front-end
 * structural zeros, quote risk against bump-and-rebootstrap, the dense LU
 * replay, Gauss-Newton quote gamma, the risk report buckets, the depth-1 and
 * depth-2 stack transforms (delta and gamma), rebuild/clock preservation and
 * the validation rejects.
 */

#include "quantape/markets/Curves/CurveBuilder.h"
#include "quantape/markets/Curves/CurveRisk.h"
#include "quantape/markets/Curves/CurveRiskReport.h"
#include "quantape/markets/Curves/DiscountCurve.h"
#include "quantape/markets/Curves/StackRisk.h"
#include "quantape/markets/Curves/StackRiskRows.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

#include "support/GtestSupport.h"

using namespace quantape;

namespace {

using markets::CurvePillar;
using markets::DiscountCurve;
using markets::InterpolationScheme;
using markets::InterpolationSpace;
using markets::PillarKind;

struct Fixture {
    datetime::Date reference{2026, 9, 29};
    std::vector<datetime::Date> dates;
    std::vector<double> targetZeros;
    std::vector<CurvePillar> pillars;
    std::optional<DiscountCurve<double>> curve;

    Fixture(InterpolationSpace space, InterpolationScheme scheme, double tension = 0.0) {
        const datetime::DayCounter zeroDayCounter(datetime::DayCount::Actual365Fixed);
        const datetime::Calendar calendar = datetime::Calendar::noHolidays();
        dates.push_back(datetime::Period(6, datetime::TimeUnit::Months).advance(reference));
        for (int years = 1; years <= 5; ++years) {
            dates.push_back(reference.plusYears(years));
        }
        for (const auto& date : dates) {
            const double t = datetime::yearFraction(reference, date, zeroDayCounter);
            targetZeros.push_back(0.04 - 0.002 * std::exp(-0.5 * t));
        }
        const DiscountCurve<double> target(reference, dates, zeroDayCounter, targetZeros, space,
                                           scheme, tension);
        CurvePillar deposit;
        deposit.maturity = dates.front();
        deposit.kind = PillarKind::Deposit;
        deposit.quoteDayCounter = datetime::DayCounter(datetime::DayCount::Actual360);
        deposit.calendar = calendar;
        deposit.quote = markets::impliedQuote(deposit, reference, target);
        pillars.push_back(deposit);
        for (std::size_t i = 1; i < dates.size(); ++i) {
            CurvePillar swap;
            swap.maturity = dates[i];
            swap.kind = PillarKind::OisSwap;
            swap.quoteDayCounter = zeroDayCounter;
            swap.calendar = calendar;
            swap.quote = markets::impliedQuote(swap, reference, target);
            pillars.push_back(swap);
        }
        curve = markets::bootstrapDiscountCurve(reference, zeroDayCounter, space, scheme, pillars,
                                                1e-14, tension);
    }
};

std::vector<markets::ForecastPillar>
asForecastPillars(const std::vector<markets::BasisPillar>& basisPillars) {
    std::vector<markets::ForecastPillar> pillars;
    pillars.reserve(basisPillars.size());
    for (const markets::BasisPillar& basis : basisPillars) {
        markets::ForecastPillar pillar;
        pillar.kind = markets::ForecastPillar::Kind::BasisSwap;
        pillar.basis = basis;
        pillars.push_back(pillar);
    }
    return pillars;
}

/// Root OIS curve with an IRS forecast child bootstrapped over it. The root
/// has one par OIS pillar per year; the child has one par IRS pillar per
/// requested year with quarterly float and annual fixed schedules.
struct IrsStackFixture {
    datetime::Date reference{2026, 9, 29};
    datetime::DayCounter zeroDc{datetime::DayCount::Actual365Fixed};
    datetime::Calendar calendar = datetime::Calendar::noHolidays();
    std::shared_ptr<DiscountCurve<double>> root;
    std::vector<CurvePillar> rootPillars;
    std::shared_ptr<markets::SpreadCurve<double>> child;
    std::vector<markets::ForecastPillar> childPillars;

    IrsStackFixture(int rootYears, const std::vector<int>& irsYears,
                    datetime::DayCounter zeroDayCounter =
                        datetime::DayCounter(datetime::DayCount::Actual365Fixed))
        : zeroDc(std::move(zeroDayCounter)) {
        std::vector<datetime::Date> dates;
        std::vector<double> zeros;
        for (int year = 1; year <= rootYears; ++year) {
            dates.push_back(reference.plusYears(year));
            zeros.push_back(0.03 + 0.0004 * static_cast<double>(year));
        }
        const DiscountCurve<double> target(reference, dates, zeroDc, zeros,
                                           InterpolationSpace::LogDiscount,
                                           InterpolationScheme::Linear);
        for (std::size_t i = 0; i < dates.size(); ++i) {
            CurvePillar swap;
            swap.maturity = dates[i];
            swap.kind = PillarKind::OisSwap;
            swap.quoteDayCounter = zeroDc;
            swap.calendar = calendar;
            swap.quote = markets::impliedQuote(swap, reference, target);
            rootPillars.push_back(swap);
        }
        root = std::make_shared<DiscountCurve<double>>(
            markets::bootstrapDiscountCurve(reference, zeroDc, InterpolationSpace::LogDiscount,
                                            InterpolationScheme::Linear, rootPillars));
        for (const int year : irsYears) {
            markets::ForecastPillar instrument;
            instrument.kind = markets::ForecastPillar::Kind::Irs;
            instrument.irs.maturity = reference.plusYears(year);
            instrument.irs.floatTenor = datetime::Period(3, datetime::TimeUnit::Months);
            instrument.irs.floatCalendar = calendar;
            instrument.irs.floatDayCounter = datetime::DayCounter(datetime::DayCount::Actual360);
            instrument.irs.fixedTenor = datetime::Period(1, datetime::TimeUnit::Years);
            instrument.irs.fixedCalendar = calendar;
            instrument.irs.fixedDayCounter =
                datetime::DayCounter(datetime::DayCount::Thirty360BondBasis);
            childPillars.push_back(instrument);
        }
        child = std::make_shared<markets::SpreadCurve<double>>(markets::bootstrapForecastCurve(
            root, nullptr, reference, zeroDc, InterpolationScheme::Linear, childPillars));
    }
};

/// Root plus one forecast child after re-bootstrapping from quote vectors.
struct Depth1Stack {
    std::shared_ptr<DiscountCurve<double>> root;
    std::shared_ptr<markets::SpreadCurve<double>> child;
};

Depth1Stack rebuildDepth1Stack(const IrsStackFixture& fixture,
                               const std::vector<CurvePillar>& rootPillars,
                               const std::vector<markets::ForecastPillar>& childPillars) {
    Depth1Stack stack;
    stack.root = std::make_shared<DiscountCurve<double>>(markets::bootstrapDiscountCurve(
        fixture.reference, fixture.zeroDc, InterpolationSpace::LogDiscount,
        InterpolationScheme::Linear, rootPillars));
    stack.child = std::make_shared<markets::SpreadCurve<double>>(
        markets::bootstrapForecastCurve(stack.root, nullptr, fixture.reference, fixture.zeroDc,
                                        InterpolationScheme::Linear, childPillars));
    return stack;
}

/// Central-difference the exact quote gradient of the quadratic node value
/// `V = g^T zeta + 0.5 zeta^T HZeta zeta` by re-bootstrapping the depth-1
/// stack from bumped quotes, and compare every analytic gamma column with it.
/// Returns the largest absolute deviation from the Gauss-Newton transform
/// `J^T HZeta J`.
double checkStackGammaFiniteDifference(const IrsStackFixture& fixture, const std::vector<double>& g,
                                       const std::vector<double>& HZeta,
                                       const markets::StackQuoteGamma& gamma, double tolerance) {
    const std::size_t mRoot = fixture.rootPillars.size();
    const std::size_t mChild = fixture.childPillars.size();
    const std::size_t dim = mRoot + mChild;
    const auto quoteGradient = [&](const std::vector<CurvePillar>& rootQuotes,
                                   const std::vector<markets::ForecastPillar>& childQuotes) {
        const Depth1Stack stack = rebuildDepth1Stack(fixture, rootQuotes, childQuotes);
        std::vector<double> zeta(dim, 0.0);
        const std::vector<double> rootNodes = stack.root->zeros();
        for (std::size_t i = 0; i < mRoot; ++i) {
            zeta[i] = rootNodes[i + 1];
        }
        const std::vector<double> childNodes = stack.child->spreadNodes().zeros();
        for (std::size_t i = 0; i < mChild; ++i) {
            zeta[mRoot + i] = childNodes[i + 1];
        }
        std::vector<double> nodeGradient(dim, 0.0);
        for (std::size_t i = 0; i < dim; ++i) {
            double sum = g[i];
            for (std::size_t j = 0; j < dim; ++j) {
                sum += HZeta[i * dim + j] * zeta[j];
            }
            nodeGradient[i] = sum;
        }
        std::vector<double> rootGradient(stack.root->size(), 0.0);
        for (std::size_t i = 0; i < mRoot; ++i) {
            rootGradient[i + 1] = nodeGradient[i];
        }
        std::vector<double> childGradient(stack.child->size(), 0.0);
        for (std::size_t i = 0; i < mChild; ++i) {
            childGradient[i + 1] = nodeGradient[mRoot + i];
        }
        std::vector<markets::StackChildInput> fdChildren;
        fdChildren.push_back(markets::StackChildInput{
            markets::CurveRole::Forecast, stack.child.get(), childQuotes, childGradient});
        const std::vector<markets::StackRiskEntry> entries = markets::stackQuoteRisk(
            *stack.root, rootQuotes, rootGradient, fdChildren, fixture.reference);
        std::vector<double> gradient(dim, 0.0);
        std::size_t index = 0;
        for (const markets::StackRiskEntry& entry : entries) {
            for (const markets::QuotePoint& point : entry.points) {
                gradient[index++] = point.delta;
            }
        }
        return gradient;
    };

    const double epsilon = 1e-5;
    for (std::size_t r = 0; r < dim; ++r) {
        std::vector<CurvePillar> rootUp = fixture.rootPillars;
        std::vector<CurvePillar> rootDown = fixture.rootPillars;
        std::vector<markets::ForecastPillar> childUp = fixture.childPillars;
        std::vector<markets::ForecastPillar> childDown = fixture.childPillars;
        if (r < mRoot) {
            rootUp[r].quote += epsilon;
            rootDown[r].quote -= epsilon;
        } else {
            childUp[r - mRoot].irs.quote += epsilon;
            childDown[r - mRoot].irs.quote -= epsilon;
        }
        const std::vector<double> plus = quoteGradient(rootUp, childUp);
        const std::vector<double> minus = quoteGradient(rootDown, childDown);
        for (std::size_t a = 0; a < dim; ++a) {
            const double fd = (plus[a] - minus[a]) / (2.0 * epsilon);
            CHECK_CLOSE("exact stack gamma vs FD", gamma.at(a, r), fd, tolerance);
        }
    }

    // Gauss-Newton transform for comparison: J^T HZeta J with J = F^{-1}.
    std::vector<markets::StackCurveInput> inputs(2);
    inputs[0].curve = markets::StackCurveView::make(*fixture.root);
    inputs[0].role = markets::CurveRole::Discount;
    inputs[0].discountPillars = fixture.rootPillars;
    inputs[0].dVdNodes.assign(fixture.root->size(), 0.0);
    inputs[1].curve = markets::StackCurveView::make(*fixture.child);
    inputs[1].role = markets::CurveRole::Forecast;
    inputs[1].forecastPillars = fixture.childPillars;
    inputs[1].dVdNodes.assign(fixture.child->size(), 0.0);
    const markets::StackQuoteSystem system =
        markets::assembleStackQuoteSystem(inputs, fixture.reference);
    std::vector<double> jacobian(dim * dim, 0.0);
    std::vector<double> unit(dim, 0.0);
    for (std::size_t column = 0; column < dim; ++column) {
        std::fill(unit.begin(), unit.end(), 0.0);
        unit[column] = 1.0;
        std::vector<double> fCopy = system.jacobian;
        const std::vector<double> solution = math::solveDense(std::move(fCopy), dim, unit);
        for (std::size_t k = 0; k < dim; ++k) {
            jacobian[k * dim + column] = solution[k];
        }
    }
    double maxDifference = 0.0;
    for (std::size_t i = 0; i < dim; ++i) {
        for (std::size_t j = 0; j < dim; ++j) {
            double gaussNewton = 0.0;
            for (std::size_t a = 0; a < dim; ++a) {
                for (std::size_t b = 0; b < dim; ++b) {
                    gaussNewton +=
                        jacobian[a * dim + i] * HZeta[a * dim + b] * jacobian[b * dim + j];
                }
            }
            maxDifference = std::max(maxDifference, std::abs(gamma.at(i, j) - gaussNewton));
        }
    }
    return maxDifference;
}

/// Root discount curve with a depth-2 forecast chain: a basis child over the
/// root and an IRS grandchild over the child. With `rootDiscount` the
/// grandchild discounts exogenously on the root instead of on its parent.
struct Depth2StackFixture {
    datetime::Date reference{2026, 9, 29};
    datetime::DayCounter zeroDc{datetime::DayCount::Actual365Fixed};
    datetime::Calendar calendar = datetime::Calendar::noHolidays();
    std::shared_ptr<DiscountCurve<double>> root;
    std::vector<CurvePillar> rootPillars;
    std::shared_ptr<markets::SpreadCurve<double>> child;
    std::vector<markets::ForecastPillar> childPillars;
    std::shared_ptr<markets::SpreadCurve<double, markets::SpreadCurve<double>>> grandchild;
    std::vector<markets::ForecastPillar> grandPillars;
    bool rootDiscount = false;

    explicit Depth2StackFixture(bool exogenousRootDiscount) : rootDiscount(exogenousRootDiscount) {
        Fixture fx(InterpolationSpace::LogDiscount, InterpolationScheme::Linear);
        reference = fx.reference;
        rootPillars = fx.pillars;
        root = std::make_shared<DiscountCurve<double>>(*fx.curve);
        std::vector<markets::BasisPillar> basisPillars;
        for (int year = 1; year <= 3; ++year) {
            markets::BasisPillar pillar;
            pillar.maturity = reference.plusYears(year);
            pillar.floatTenor = datetime::Period(3, datetime::TimeUnit::Months);
            pillar.quoteDayCounter = zeroDc;
            pillar.calendar = calendar;
            pillar.spread = 0.0005 + 0.0002 * static_cast<double>(year);
            basisPillars.push_back(pillar);
        }
        childPillars = asForecastPillars(basisPillars);
        child = std::make_shared<markets::SpreadCurve<double>>(markets::bootstrapForecastCurve(
            root, nullptr, reference, zeroDc, InterpolationScheme::Linear, childPillars));
        for (int year = 1; year <= 2; ++year) {
            markets::ForecastPillar instrument;
            instrument.kind = markets::ForecastPillar::Kind::Irs;
            instrument.irs.maturity = reference.plusYears(year);
            instrument.irs.floatTenor = datetime::Period(3, datetime::TimeUnit::Months);
            instrument.irs.floatCalendar = calendar;
            instrument.irs.floatDayCounter = datetime::DayCounter(datetime::DayCount::Actual360);
            instrument.irs.fixedTenor = datetime::Period(1, datetime::TimeUnit::Years);
            instrument.irs.fixedCalendar = calendar;
            instrument.irs.fixedDayCounter =
                datetime::DayCounter(datetime::DayCount::Thirty360BondBasis);
            instrument.irs.quote = 0.02 + 0.0015 * static_cast<double>(year);
            grandPillars.push_back(instrument);
        }
        const DiscountCurve<double>* discount = rootDiscount ? root.get() : nullptr;
        grandchild = std::make_shared<markets::SpreadCurve<double, markets::SpreadCurve<double>>>(
            markets::bootstrapForecastCurve(child, discount, reference, zeroDc,
                                            InterpolationScheme::Linear, grandPillars));
    }
};

/// The three curves of a depth-2 stack after a re-bootstrap.
struct Depth2Curves {
    std::shared_ptr<DiscountCurve<double>> root;
    std::shared_ptr<markets::SpreadCurve<double>> child;
    std::shared_ptr<markets::SpreadCurve<double, markets::SpreadCurve<double>>> grandchild;
};

Depth2Curves rebuildDepth2Stack(const Depth2StackFixture& fixture,
                                const std::vector<CurvePillar>& rootPillars,
                                const std::vector<markets::ForecastPillar>& childPillars,
                                const std::vector<markets::ForecastPillar>& grandPillars) {
    Depth2Curves curves;
    curves.root = std::make_shared<DiscountCurve<double>>(markets::bootstrapDiscountCurve(
        fixture.reference, fixture.zeroDc, InterpolationSpace::LogDiscount,
        InterpolationScheme::Linear, rootPillars));
    curves.child = std::make_shared<markets::SpreadCurve<double>>(
        markets::bootstrapForecastCurve(curves.root, nullptr, fixture.reference, fixture.zeroDc,
                                        InterpolationScheme::Linear, childPillars));
    const DiscountCurve<double>* discount = fixture.rootDiscount ? curves.root.get() : nullptr;
    curves.grandchild =
        std::make_shared<markets::SpreadCurve<double, markets::SpreadCurve<double>>>(
            markets::bootstrapForecastCurve(curves.child, discount, fixture.reference,
                                            fixture.zeroDc, InterpolationScheme::Linear,
                                            grandPillars));
    return curves;
}

std::vector<markets::StackCurveInput> depth2StackInputs(const Depth2StackFixture& fixture,
                                                        const std::vector<double>& dVdRoot,
                                                        const std::vector<double>& dVdChild,
                                                        const std::vector<double>& dVdGrand) {
    std::vector<markets::StackCurveInput> inputs(3);
    inputs[0].curve = markets::StackCurveView::make(*fixture.root);
    inputs[0].role = markets::CurveRole::Discount;
    inputs[0].discountPillars = fixture.rootPillars;
    inputs[0].dVdNodes = dVdRoot;
    inputs[1].curve = markets::StackCurveView::make(*fixture.child);
    inputs[1].role = markets::CurveRole::Forecast;
    inputs[1].forecastPillars = fixture.childPillars;
    inputs[1].dVdNodes = dVdChild;
    inputs[2].curve = markets::StackCurveView::make(*fixture.grandchild);
    inputs[2].role = markets::CurveRole::Forecast;
    inputs[2].forecastPillars = fixture.grandPillars;
    inputs[2].dVdNodes = dVdGrand;
    if (fixture.rootDiscount) {
        inputs[2].discount = markets::StackCurveView::make(*fixture.root);
    }
    return inputs;
}

/// Arbitrary linear value gradients over the three stack node vectors.
struct Depth2Gradients {
    std::vector<double> root;
    std::vector<double> child;
    std::vector<double> grand;
};

Depth2Gradients depth2Gradients(const Depth2StackFixture& fixture) {
    Depth2Gradients gradients;
    gradients.root.assign(fixture.root->size(), 0.0);
    gradients.root[1] = 0.7;
    gradients.root[3] = -0.4;
    gradients.root[5] = 0.25;
    gradients.child.assign(fixture.child->size(), 0.0);
    gradients.child[1] = -0.6;
    gradients.child[3] = 0.5;
    gradients.grand.assign(fixture.grandchild->size(), 0.0);
    gradients.grand[1] = 0.9;
    gradients.grand[2] = -0.3;
    return gradients;
}

double depth2Value(const Depth2Curves& curves, const Depth2Gradients& gradients) {
    double value = 0.0;
    const std::vector<double> rootNodes = curves.root->zeros();
    for (std::size_t i = 0; i < gradients.root.size(); ++i) {
        value += gradients.root[i] * rootNodes[i];
    }
    const std::vector<double> childNodes = curves.child->spreadNodes().zeros();
    for (std::size_t i = 0; i < gradients.child.size(); ++i) {
        value += gradients.child[i] * childNodes[i];
    }
    const std::vector<double> grandNodes = curves.grandchild->spreadNodes().zeros();
    for (std::size_t i = 0; i < gradients.grand.size(); ++i) {
        value += gradients.grand[i] * grandNodes[i];
    }
    return value;
}

std::vector<double> discountZerosSensitivity(const DiscountCurve<double>& curve, double t) {
    std::vector<double> weights;
    curve.zeroNodeWeights(t, weights);
    const double df = curve.discount(t);
    std::vector<double> out(weights.size(), 0.0);
    for (std::size_t i = 0; i < weights.size(); ++i) {
        out[i] = -t * df * weights[i];
    }
    return out;
}

std::vector<std::string> pointLabels(const std::vector<markets::QuotePoint>& points) {
    std::vector<std::string> labels;
    labels.reserve(points.size());
    for (const markets::QuotePoint& point : points) {
        labels.push_back(point.label);
    }
    return labels;
}

std::vector<double> pointDeltas(const std::vector<markets::QuotePoint>& points) {
    std::vector<double> deltas;
    deltas.reserve(points.size());
    for (const markets::QuotePoint& point : points) {
        deltas.push_back(point.delta);
    }
    return deltas;
}

bool samePointMetadata(const std::vector<markets::QuotePoint>& left,
                       const std::vector<markets::QuotePoint>& right) {
    if (left.size() != right.size()) {
        return false;
    }
    for (std::size_t i = 0; i < left.size(); ++i) {
        if (left[i].label != right[i].label || left[i].bucket != right[i].bucket ||
            left[i].year != right[i].year || left[i].role != right[i].role) {
            return false;
        }
    }
    return true;
}

void testBasisRiskRepresentation() {
    Fixture fx(InterpolationSpace::LogDiscount, InterpolationScheme::Linear);
    auto parent = std::make_shared<DiscountCurve<double>>(*fx.curve);
    const datetime::DayCounter zeroDc(datetime::DayCount::Actual365Fixed);
    std::vector<markets::BasisPillar> basisPillars;
    for (std::size_t i = 1; i < fx.dates.size(); ++i) {
        markets::BasisPillar pillar;
        pillar.maturity = fx.dates[i];
        pillar.floatTenor = datetime::Period(3, datetime::TimeUnit::Months);
        pillar.quoteDayCounter = zeroDc;
        pillar.calendar = datetime::Calendar::noHolidays();
        pillar.spread = 0.0005 + 0.0001 * static_cast<double>(i);
        basisPillars.push_back(pillar);
    }
    const markets::SpreadCurve<double> child = markets::bootstrapSpreadCurve(
        parent, fx.reference, zeroDc, InterpolationScheme::Linear, basisPillars);
    const std::size_t m = basisPillars.size();
    const std::size_t mp = parent->size() - 1;
    std::vector<double> f;
    std::vector<double> c;
    markets::assembleBasisJacobian(child, basisPillars, fx.reference, f, c);

    const double epsilon = 1e-6;
    const auto& spreadNodes = child.spreadNodes();
    for (std::size_t j = 0; j < m; ++j) {
        for (std::size_t k = 1; k <= m; ++k) {
            std::vector<double> plus = spreadNodes.zeros();
            std::vector<double> minus = spreadNodes.zeros();
            plus[k] += epsilon;
            minus[k] -= epsilon;
            const markets::SpreadCurve<double> childPlus(
                parent, spreadNodes.times(), plus, spreadNodes.scheme(), spreadNodes.tension());
            const markets::SpreadCurve<double> childMinus(
                parent, spreadNodes.times(), minus, spreadNodes.scheme(), spreadNodes.tension());
            const double fd =
                (markets::impliedBasisSpread(childPlus, basisPillars[j], fx.reference, zeroDc) -
                 markets::impliedBasisSpread(childMinus, basisPillars[j], fx.reference, zeroDc)) /
                (2.0 * epsilon);
            CHECK_CLOSE("basis own-row vs FD", f[j * m + (k - 1)], fd, 1e-6);
        }
    }
    for (std::size_t j = 0; j < m; ++j) {
        for (std::size_t i = 1; i <= mp; ++i) {
            const auto bumpedParent = [&](double delta) {
                std::vector<double> zeros = parent->zeros();
                zeros[i] += delta;
                return std::make_shared<DiscountCurve<double>>(
                    parent->times(), zeros, parent->space(), parent->scheme(), parent->tension(),
                    parent->switchIndex());
            };
            const markets::SpreadCurve<double> childPlus(bumpedParent(epsilon), spreadNodes.times(),
                                                         spreadNodes.zeros(), spreadNodes.scheme(),
                                                         spreadNodes.tension());
            const markets::SpreadCurve<double> childMinus(
                bumpedParent(-epsilon), spreadNodes.times(), spreadNodes.zeros(),
                spreadNodes.scheme(), spreadNodes.tension());
            const double fd =
                (markets::impliedBasisSpread(childPlus, basisPillars[j], fx.reference, zeroDc) -
                 markets::impliedBasisSpread(childMinus, basisPillars[j], fx.reference, zeroDc)) /
                (2.0 * epsilon);
            CHECK_CLOSE("basis cross-row vs FD", c[j * mp + (i - 1)], fd, 1e-6);
        }
    }

    const double t = 4.5;
    const double value = child.discount(t);
    std::vector<double> childWeights;
    child.zeroNodeWeights(t, childWeights);
    std::vector<double> dVdSpread(child.size(), 0.0);
    for (std::size_t i = 0; i < childWeights.size(); ++i) {
        dVdSpread[i] = -t * value * childWeights[i];
    }
    std::vector<double> rootWeights;
    parent->zeroNodeWeights(t, rootWeights);
    std::vector<double> dVdRoot(rootWeights.size());
    for (std::size_t i = 0; i < rootWeights.size(); ++i) {
        dVdRoot[i] = -t * value * rootWeights[i];
    }
    std::vector<markets::StackChildInput> children;
    children.push_back(markets::StackChildInput{markets::CurveRole::Forecast, &child,
                                                asForecastPillars(basisPillars), dVdSpread});
    (void)markets::stackQuoteRisk(*parent, fx.pillars, dVdRoot, children, fx.reference);
}

void testFrontEndRiskCompleteness() {
    // A valuation at 0.25y touches only the fixed node 0 and the 6M deposit
    // node. In the singleton bootstrap the deposit quote only moves its own
    // node while every later quote holds its own node fixed, so the quote
    // risk must be concentrated in the deposit pillar and be exactly zero
    // elsewhere. This pins the structural zeros the front-end reports show
    // (they are not dropped risk).
    Fixture fixture(InterpolationSpace::LogDiscount, InterpolationScheme::Linear);
    const double t = 0.25;
    std::vector<double> dVdZeros = discountZerosSensitivity(*fixture.curve, t);
    const markets::QuoteRisk risk =
        markets::transformQuoteRisk(*fixture.curve, fixture.pillars, fixture.reference, dVdZeros);
    const double epsilon = 1e-6;
    for (std::size_t j = 0; j < fixture.pillars.size(); ++j) {
        std::vector<CurvePillar> up = fixture.pillars;
        std::vector<CurvePillar> down = fixture.pillars;
        up[j].quote += epsilon;
        down[j].quote -= epsilon;
        const datetime::DayCounter zeroDc(datetime::DayCount::Actual365Fixed);
        const double fd = (markets::bootstrapDiscountCurve(fixture.reference, zeroDc,
                                                           InterpolationSpace::LogDiscount,
                                                           InterpolationScheme::Linear, up, 1e-14)
                               .discount(t) -
                           markets::bootstrapDiscountCurve(fixture.reference, zeroDc,
                                                           InterpolationSpace::LogDiscount,
                                                           InterpolationScheme::Linear, down, 1e-14)
                               .discount(t)) /
                          (2.0 * epsilon);
        CHECK_CLOSE("front-end quote risk vs FD", risk.quoteDeltas[j], fd, 1e-8);
    }
    EXPECT_GT(std::abs(risk.quoteDeltas.front()), 1e-3);
    for (std::size_t j = 1; j < risk.quoteDeltas.size(); ++j) {
        EXPECT_EQ(risk.quoteDeltas[j], 0.0);
    }
}

void testZeroTimeNodeWeights() {
    const std::vector<double> times{0.0, 1.0, 2.0};
    const std::vector<double> zeros{0.0, 0.03, 0.035};
    const DiscountCurve<double> curve(times, zeros, InterpolationSpace::LogDiscount,
                                      InterpolationScheme::Linear);
    std::vector<double> weights;
    curve.zeroNodeWeights(0.0, weights);
    for (const double weight : weights) {
        EXPECT_EQ(weight, 0.0);
    }
}

void testExtrapolatedNodeWeights() {
    const std::vector<double> times{0.0, 1.0, 2.0, 3.0};
    const std::vector<double> zeros{0.0, 0.03, 0.035, 0.04};
    const double t = 3.0 + 2.0 / 365.0; // first pay date past the last node
    const auto check = [&](InterpolationSpace space, InterpolationScheme scheme, double tension) {
        const DiscountCurve<double> curve(times, zeros, space, scheme, tension);
        std::vector<double> weights;
        curve.zeroNodeWeights(t, weights);
        const double epsilon = 1e-6;
        for (std::size_t j = 1; j < zeros.size(); ++j) {
            std::vector<double> bumped = zeros;
            bumped[j] += epsilon;
            const DiscountCurve<double> up(times, bumped, space, scheme, tension);
            const double fd = (up.zero(t) - curve.zero(t)) / epsilon;
            CHECK_CLOSE("extrapolated node weight vs FD", weights[j], fd, 1e-5);
        }
    };
    check(InterpolationSpace::Zero, InterpolationScheme::Linear, 0.0);
    check(InterpolationSpace::LogDiscount, InterpolationScheme::Linear, 0.0);
    check(InterpolationSpace::LogDiscount, InterpolationScheme::Akima, 0.0);
    check(InterpolationSpace::LogDiscount, InterpolationScheme::MonotoneCubic, 0.0);
    check(InterpolationSpace::LogDiscount, InterpolationScheme::TensionSpline, 8.0);
}

void testQuoteRiskFiniteDifference(InterpolationSpace space, InterpolationScheme scheme,
                                   double tension = 0.0, double tolerance = 1e-8) {
    Fixture fixture(space, scheme, tension);
    const double valuationTime = 4.5;
    const std::vector<double> dVdZeros = discountZerosSensitivity(*fixture.curve, valuationTime);
    const markets::QuoteRisk risk =
        markets::transformQuoteRisk(*fixture.curve, fixture.pillars, fixture.reference, dVdZeros);

    const double epsilon = 1e-6;
    for (std::size_t j = 0; j < fixture.pillars.size(); ++j) {
        std::vector<CurvePillar> up = fixture.pillars;
        std::vector<CurvePillar> down = fixture.pillars;
        up[j].quote += epsilon;
        down[j].quote -= epsilon;
        const datetime::DayCounter zeroDc(datetime::DayCount::Actual365Fixed);
        const double valueUp = markets::bootstrapDiscountCurve(fixture.reference, zeroDc, space,
                                                               scheme, up, 1e-14, tension)
                                   .discount(valuationTime);
        const double valueDown = markets::bootstrapDiscountCurve(fixture.reference, zeroDc, space,
                                                                 scheme, down, 1e-14, tension)
                                     .discount(valuationTime);
        const double fd = (valueUp - valueDown) / (2.0 * epsilon);
        CHECK_CLOSE("quote risk vs FD", risk.quoteDeltas[j], fd, tolerance);
    }
}

void testNodeWeightsFiniteDifference(InterpolationSpace space, InterpolationScheme scheme,
                                     double tension = 0.0, double partitionTol = 1e-10) {
    const datetime::DayCounter zeroDc(datetime::DayCount::Actual365Fixed);
    const std::vector<double> times{0.0, 1.0, 2.0, 3.0, 4.0};
    const std::vector<double> zeros{0.0, 0.03, 0.035, 0.038, 0.04};
    const DiscountCurve<double> curve(times, zeros, space, scheme, tension);
    const double t = 1.7;
    // The interpolation operator reproduces constants: space-value weights sum
    // to one over all nodes (including the fixed t = 0 node).
    std::vector<double> spaceWeights;
    curve.spaceValueWeights(t, spaceWeights);
    double total = 0.0;
    for (const double w : spaceWeights) {
        total += w;
    }
    CHECK_CLOSE("node weights partition", total, 1.0, partitionTol);

    // Sensitivities to the solved nodes (t = 0 node fixed) match FD.
    std::vector<double> weights;
    curve.zeroNodeWeights(t, weights);

    const double epsilon = 1e-7;
    for (std::size_t j = 1; j < zeros.size(); ++j) {
        std::vector<double> bumped = zeros;
        bumped[j] += epsilon;
        const DiscountCurve<double> up(times, bumped, space, scheme, tension);
        const double fd =
            (static_cast<double>(up.zero(t)) - static_cast<double>(curve.zero(t))) / epsilon;
        CHECK_CLOSE("node weight vs FD", weights[j], fd, 1e-5);
    }
}

void testDenseLuMatchesSolveDense() {
    // A pivot-swapping matrix: one factorization must replay the same
    // interleaved elimination as an independent dense solve, both for the
    // combined right-hand side and for the individual identity columns.
    const std::vector<double> matrix{1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 10.0};
    const std::vector<double> rhs{1.0, -2.0, 0.5};
    const std::size_t n = 3;
    const std::vector<double> reference = math::solveDense(matrix, n, rhs);
    const markets::detail::DenseLu factors = markets::detail::factorDenseLu(matrix, n);
    const std::vector<double> unit0 = markets::detail::solveDenseLu(factors, {1.0, 0.0, 0.0});
    const std::vector<double> unit1 = markets::detail::solveDenseLu(factors, {0.0, 1.0, 0.0});
    const std::vector<double> unit2 = markets::detail::solveDenseLu(factors, {0.0, 0.0, 1.0});
    const std::vector<double> direct = markets::detail::solveDenseLu(factors, rhs);
    CHECK_CLOSE_SEQ("LU direct solve", direct, reference, 1e-14);
    for (std::size_t i = 0; i < n; ++i) {
        const double combined = unit0[i] * rhs[0] + unit1[i] * rhs[1] + unit2[i] * rhs[2];
        CHECK_CLOSE("LU column combination", combined, reference[i], 1e-14);
    }
}

void testQuoteGamma() {
    Fixture fixture(InterpolationSpace::LogDiscount, InterpolationScheme::Linear);
    const std::size_t n = fixture.curve->size();
    const std::size_t m = fixture.pillars.size();
    const double t = 4.5;
    std::vector<double> weights;
    fixture.curve->spaceValueWeights(t, weights);
    const double df = fixture.curve->discount(t);
    // dV/dz_i = -t D w_i (LogDiscount: z(t) = y(t)/t, space weights are on y;
    // zeroNodeWeights converts; use that conversion for consistency).
    std::vector<double> u;
    fixture.curve->zeroNodeWeights(t, u);
    std::vector<double> dVdZeros(n, 0.0);
    for (std::size_t i = 0; i < n; ++i) {
        dVdZeros[i] = -t * df * u[i];
    }
    // H_zeta ~= t^2 D u u^T (interpolation weights treated as constant).
    std::vector<double> H(n * n, 0.0);
    for (std::size_t i = 0; i < n; ++i) {
        for (std::size_t j = 0; j < n; ++j) {
            H[i * n + j] = t * t * df * u[i] * u[j];
        }
    }
    const markets::QuoteGamma gamma =
        markets::transformQuoteGamma(*fixture.curve, fixture.pillars, fixture.reference, H);
    EXPECT_EQ(gamma.dim, m);

    // The transform is Gauss-Newton: validate it against a numerical
    // Jacobian reference H_ref = J_num^T H_zeta J_num (total second
    // differences also contain the neglected bootstrap-curvature term).
    const double epsilon = 1e-6;
    std::vector<double> jacobian(m * m, 0.0);
    for (std::size_t j = 0; j < m; ++j) {
        std::vector<CurvePillar> up = fixture.pillars;
        std::vector<CurvePillar> down = fixture.pillars;
        up[j].quote += epsilon;
        down[j].quote -= epsilon;
        const datetime::DayCounter zeroDc(datetime::DayCount::Actual365Fixed);
        const auto curveUp = markets::bootstrapDiscountCurve(
            fixture.reference, zeroDc, InterpolationSpace::LogDiscount, InterpolationScheme::Linear,
            up, 1e-14);
        const auto curveDown = markets::bootstrapDiscountCurve(
            fixture.reference, zeroDc, InterpolationSpace::LogDiscount, InterpolationScheme::Linear,
            down, 1e-14);
        for (std::size_t k = 0; k < m; ++k) {
            jacobian[k * m + j] =
                (curveUp.zeros()[k + 1] - curveDown.zeros()[k + 1]) / (2.0 * epsilon);
        }
    }
    std::vector<double> restricted(m * m, 0.0);
    for (std::size_t a = 0; a < m; ++a) {
        for (std::size_t b = 0; b < m; ++b) {
            restricted[a * m + b] = H[(a + 1) * n + (b + 1)];
        }
    }
    std::vector<double> reference(m * m, 0.0);
    for (std::size_t i = 0; i < m; ++i) {
        for (std::size_t j = 0; j < m; ++j) {
            double sum = 0.0;
            for (std::size_t a = 0; a < m; ++a) {
                for (std::size_t b = 0; b < m; ++b) {
                    sum += jacobian[a * m + i] * restricted[a * m + b] * jacobian[b * m + j];
                }
            }
            reference[i * m + j] = sum;
        }
    }
    for (std::size_t i = 0; i < m; ++i) {
        CHECK_CLOSE("gamma diagonal vs reference", gamma.at(i, i), reference[i * m + i], 1e-8);
    }
    const std::size_t i0 = 1;
    const std::size_t j0 = 3;
    CHECK_CLOSE("gamma cross vs reference", gamma.at(i0, j0), reference[i0 * m + j0], 1e-8);
    CHECK_CLOSE("gamma symmetry", gamma.at(i0, j0), gamma.at(j0, i0), 1e-12);
}

void testFraRepoGranular() {
    const datetime::Date reference(2026, 9, 29);
    const datetime::DayCounter zeroDc(datetime::DayCount::Actual365Fixed);
    const datetime::Calendar calendar = datetime::Calendar::noHolidays();
    const datetime::Date sixM = datetime::Period(6, datetime::TimeUnit::Months).advance(reference);
    const datetime::Date oneY = reference.plusYears(1);
    const datetime::Date eighteenM =
        datetime::Period(18, datetime::TimeUnit::Months).advance(reference);
    std::vector<datetime::Date> dates{sixM, oneY, eighteenM};
    for (int years = 2; years <= 5; ++years) {
        dates.push_back(reference.plusYears(years));
    }
    std::vector<double> targetZeros;
    for (const auto& date : dates) {
        const double t = datetime::yearFraction(reference, date, zeroDc);
        targetZeros.push_back(0.04 - 0.002 * std::exp(-0.5 * t));
    }
    const DiscountCurve<double> target(reference, dates, zeroDc, targetZeros,
                                       InterpolationSpace::LogDiscount,
                                       InterpolationScheme::Linear);

    std::vector<CurvePillar> pillars;
    CurvePillar deposit;
    deposit.maturity = sixM;
    deposit.kind = PillarKind::Deposit;
    deposit.quoteDayCounter = datetime::DayCounter(datetime::DayCount::Actual360);
    deposit.calendar = calendar;
    deposit.quote = markets::impliedQuote(deposit, reference, target);
    pillars.push_back(deposit);
    CurvePillar fra;
    fra.start = sixM;
    fra.maturity = oneY;
    fra.kind = PillarKind::Fra;
    fra.quoteDayCounter = zeroDc;
    fra.calendar = calendar;
    fra.quote = markets::impliedQuote(fra, reference, target);
    pillars.push_back(fra);
    CurvePillar repo;
    repo.maturity = eighteenM;
    repo.kind = PillarKind::Repo;
    repo.quoteDayCounter = zeroDc;
    repo.calendar = calendar;
    repo.quote = markets::impliedQuote(repo, reference, target);
    pillars.push_back(repo);
    for (int years = 2; years <= 5; ++years) {
        CurvePillar swap;
        swap.maturity = reference.plusYears(years);
        swap.kind = PillarKind::OisSwap;
        swap.quoteDayCounter = zeroDc;
        swap.calendar = calendar;
        swap.quote = markets::impliedQuote(swap, reference, target);
        pillars.push_back(swap);
    }
    const DiscountCurve<double> curve = markets::bootstrapDiscountCurve(
        reference, zeroDc, InterpolationSpace::LogDiscount, InterpolationScheme::Linear, pillars);
    for (const CurvePillar& pillar : pillars) {
        CHECK_CLOSE("fra/repo reprice", markets::impliedQuote(pillar, reference, curve),
                    pillar.quote, 1e-11);
    }

    // Jacobian rows for the new kinds against finite differences of the
    // implied quote in the bootstrap node values.
    for (const CurvePillar& pillar : {pillars[1], pillars[2]}) {
        std::vector<double> row;
        EXPECT_TRUE(markets::pillarJacobianRow(pillar, reference, curve, row));
        const std::size_t n = curve.size();
        for (std::size_t i = 1; i < n; ++i) {
            std::vector<double> zeros = curve.zeros();
            zeros[i] += 1e-8;
            const DiscountCurve<double> bumped(
                curve.times(), zeros, InterpolationSpace::LogDiscount, InterpolationScheme::Linear);
            const double fd = (markets::impliedQuote(pillar, reference, bumped) -
                               markets::impliedQuote(pillar, reference, curve)) /
                              1e-8;
            CHECK_CLOSE("fra/repo jacobian vs FD", row[i - 1], fd, 1e-5);
        }
    }

    std::vector<double> u;
    curve.zeroNodeWeights(4.5, u);
    const double df = curve.discount(4.5);
    std::vector<double> dVdZeros(u.size());
    for (std::size_t i = 0; i < u.size(); ++i) {
        dVdZeros[i] = -4.5 * df * u[i];
    }
    const markets::CurveRiskReport report =
        markets::curveRiskReport(curve, pillars, reference, dVdZeros);
    bool sawFra = false;
    bool sawRepo = false;
    for (const std::string& label : report.quoteLabels) {
        sawFra = sawFra || label.rfind("Fra", 0) == 0;
        sawRepo = sawRepo || label.rfind("Repo", 0) == 0;
    }
    EXPECT_TRUE(sawFra);
    EXPECT_TRUE(sawRepo);

    const std::vector<std::pair<markets::CurveRole, markets::CurveRiskReport>> multi{
        {markets::CurveRole::Discount, report}, {markets::CurveRole::Forecast, report}};
    const std::vector<markets::RiskBucket> buckets = markets::aggregateByRoleAndKind(multi);
    double total = 0.0;
    for (const markets::RiskBucket& bucket : buckets) {
        total += bucket.delta;
    }
    CHECK_CLOSE("multi-curve granular total", total, 2.0 * report.totalDelta(), 1e-12);
}

void testStackRiskTreePass() {
    Fixture fx(InterpolationSpace::LogDiscount, InterpolationScheme::Linear);
    auto parent = std::make_shared<DiscountCurve<double>>(*fx.curve);
    const datetime::DayCounter zeroDc(datetime::DayCount::Actual365Fixed);
    std::vector<markets::BasisPillar> basisPillars;
    for (std::size_t i = 1; i < fx.dates.size(); ++i) {
        markets::BasisPillar pillar;
        pillar.maturity = fx.dates[i];
        pillar.floatTenor = datetime::Period(3, datetime::TimeUnit::Months);
        pillar.quoteDayCounter = zeroDc;
        pillar.calendar = datetime::Calendar::noHolidays();
        pillar.spread = 0.0005 + 0.0001 * static_cast<double>(i);
        basisPillars.push_back(pillar);
    }
    const auto rebuildChild = [&](const std::shared_ptr<const DiscountCurve<double>>& root,
                                  const std::vector<markets::BasisPillar>& pillars) {
        return markets::bootstrapSpreadCurve(root, fx.reference, zeroDc,
                                             InterpolationScheme::Linear, pillars);
    };
    const markets::SpreadCurve<double> child = rebuildChild(parent, basisPillars);

    const double t = 4.5;
    const double value = child.discount(t);
    std::vector<double> childWeights;
    child.zeroNodeWeights(t, childWeights);
    std::vector<double> dVdSpread(child.size(), 0.0);
    for (std::size_t i = 0; i < childWeights.size(); ++i) {
        dVdSpread[i] = -t * value * childWeights[i];
    }
    std::vector<double> rootWeights;
    parent->zeroNodeWeights(t, rootWeights);
    std::vector<double> dVdRoot(rootWeights.size());
    for (std::size_t i = 0; i < rootWeights.size(); ++i) {
        dVdRoot[i] = -t * value * rootWeights[i];
    }
    std::vector<markets::StackChildInput> children;
    children.push_back(markets::StackChildInput{markets::CurveRole::Forecast, &child,
                                                asForecastPillars(basisPillars), dVdSpread});
    const std::vector<markets::StackRiskEntry> stack =
        markets::stackQuoteRisk(*parent, fx.pillars, dVdRoot, children, fx.reference);
    EXPECT_EQ(stack.size(), 2u);

    const double epsilon = 1e-6;
    // Child totals: FD with the parent frozen.
    for (std::size_t j = 0; j < basisPillars.size(); ++j) {
        auto up = basisPillars;
        auto down = basisPillars;
        up[j].spread += epsilon;
        down[j].spread -= epsilon;
        const double fd =
            (rebuildChild(parent, up).discount(t) - rebuildChild(parent, down).discount(t)) /
            (2.0 * epsilon);
        CHECK_CLOSE("stack child total vs FD", stack[1].points[j].delta, fd, 1e-4);
    }
    // Root totals: FD with child quotes fixed (spreads re-solved).
    for (std::size_t j = 0; j < fx.pillars.size(); ++j) {
        std::vector<CurvePillar> up = fx.pillars;
        std::vector<CurvePillar> down = fx.pillars;
        up[j].quote += epsilon;
        down[j].quote -= epsilon;
        const auto rootUp = std::make_shared<DiscountCurve<double>>(
            markets::bootstrapDiscountCurve(fx.reference, zeroDc, InterpolationSpace::LogDiscount,
                                            InterpolationScheme::Linear, up));
        const auto rootDown = std::make_shared<DiscountCurve<double>>(
            markets::bootstrapDiscountCurve(fx.reference, zeroDc, InterpolationSpace::LogDiscount,
                                            InterpolationScheme::Linear, down));
        const double fd = (rebuildChild(rootUp, basisPillars).discount(t) -
                           rebuildChild(rootDown, basisPillars).discount(t)) /
                          (2.0 * epsilon);
        CHECK_CLOSE("stack root total vs FD", stack[0].points[j].delta, fd, 1e-4);
    }
}

void testStackIrsChildRiskAndGamma() {
    // One root pillar and one IRS child pillar: the stack Jacobian reduces to
    // scalar blocks F_p = [a], F_c = [c], C = [d], so J can be written by hand.
    IrsStackFixture fixture(1, {1});
    const std::size_t mRoot = fixture.rootPillars.size();
    const std::size_t mChild = fixture.childPillars.size();
    EXPECT_EQ(mRoot, 1u);
    EXPECT_EQ(mChild, 1u);
    std::vector<double> fRoot;
    std::vector<double> scratch;
    markets::assembleQuoteJacobian(*fixture.root, fixture.rootPillars, fixture.reference, fRoot,
                                   scratch);
    std::vector<double> fChild;
    std::vector<double> cross;
    markets::assembleForecastJacobian(*fixture.child, fixture.childPillars, fixture.reference,
                                      nullptr, fChild, cross);
    const double a = fRoot[0];
    const double c = fChild[0];
    const double d = cross[0];
    const double jacobian[2][2] = {{1.0 / a, 0.0}, {-d / (a * c), 1.0 / c}};

    // A quadratic value function V = g^T zeta + 0.5 zeta^T HZeta zeta. The
    // stack gamma consumes the node gradient at the base nodes,
    // dV/dzeta = g + HZeta zeta_0, and must match central differences of the
    // quote gradient x = J^T (g + HZeta zeta) re-bootstrapped from quotes.
    const std::vector<double> g{0.3, -0.7};
    const std::vector<double> HZeta{2.0, -0.5, -0.5, 1.5};
    const double zetaRoot = fixture.root->zeros()[1];
    const double zetaChild = fixture.child->spreadNodes().zeros()[1];
    std::vector<double> dVdRoot(fixture.root->size(), 0.0);
    dVdRoot[1] = g[0] + HZeta[0] * zetaRoot + HZeta[1] * zetaChild;
    std::vector<double> dVdSpread(fixture.child->size(), 0.0);
    dVdSpread[1] = g[1] + HZeta[2] * zetaRoot + HZeta[3] * zetaChild;
    std::vector<markets::StackChildInput> children;
    children.push_back(markets::StackChildInput{markets::CurveRole::Forecast, fixture.child.get(),
                                                fixture.childPillars, dVdSpread});
    const markets::StackQuoteGamma gamma = markets::stackQuoteGamma(
        *fixture.root, fixture.rootPillars, dVdRoot, HZeta, fixture.reference, children);
    EXPECT_EQ(gamma.dim, 2u);
    CHECK_CLOSE("stack gamma symmetry", gamma.at(0, 1), gamma.at(1, 0), 1e-14);
    const double curvature = checkStackGammaFiniteDifference(fixture, g, HZeta, gamma, 1e-5);
    EXPECT_GT(curvature, 1e-9);

    // The stack deltas are J^T (g + HZeta zeta_0) with the same node gradient.
    const std::vector<markets::StackRiskEntry> stack = markets::stackQuoteRisk(
        *fixture.root, fixture.rootPillars, dVdRoot, children, fixture.reference);
    const double expectedRoot = jacobian[0][0] * dVdRoot[1] + jacobian[1][0] * dVdSpread[1];
    const double expectedChild = jacobian[0][1] * dVdRoot[1] + jacobian[1][1] * dVdSpread[1];
    CHECK_CLOSE("stack delta root equals J^T g", stack[0].points[0].delta, expectedRoot, 1e-12);
    CHECK_CLOSE("stack delta child equals J^T g", stack[1].points[0].delta, expectedChild, 1e-12);

    // Labels, years and roles follow the stackQuoteRisk entry order.
    std::vector<std::string> labels;
    std::vector<int> years;
    std::vector<markets::CurveRole> roles;
    for (const markets::StackRiskEntry& entry : stack) {
        for (const markets::QuotePoint& point : entry.points) {
            labels.push_back(point.label);
            years.push_back(point.year);
            roles.push_back(point.role);
        }
    }
    std::vector<int> gammaYears;
    std::vector<markets::CurveRole> gammaRoles;
    for (const markets::QuotePoint& point : gamma.points) {
        gammaYears.push_back(point.year);
        gammaRoles.push_back(point.role);
    }
    EXPECT_TRUE(pointLabels(gamma.points) == labels);
    EXPECT_TRUE(gammaYears == years);
    EXPECT_TRUE(gammaRoles == roles);
    EXPECT_EQ(gamma.points.back().label, "Irs 1Y");

    // An equal discount curve as a different object is accepted and produces
    // the same gamma as the implicit forecast-parent discounting.
    const auto rootCopy = std::make_shared<DiscountCurve<double>>(*fixture.root);
    children[0].discountCurve = rootCopy.get();
    const markets::StackQuoteGamma gammaCopy = markets::stackQuoteGamma(
        *fixture.root, fixture.rootPillars, dVdRoot, HZeta, fixture.reference, children);
    for (std::size_t k = 0; k < gamma.hessian.size(); ++k) {
        CHECK_CLOSE("stack gamma equal discount copy", gammaCopy.hessian[k], gamma.hessian[k],
                    1e-15);
    }

    // A discount curve outside the root grid is rejected for IRS children.
    const std::vector<double> otherTimes{0.0, 0.5, 1.0};
    const std::vector<double> otherZeros{0.0, 0.02, 0.03};
    const DiscountCurve<double> otherDiscount(
        otherTimes, otherZeros, InterpolationSpace::LogDiscount, InterpolationScheme::Linear);
    children[0].discountCurve = &otherDiscount;
    EXPECT_THROW((void)(markets::stackQuoteRisk(*fixture.root, fixture.rootPillars, dVdRoot,
                                                children, fixture.reference)),
                 std::invalid_argument);
    EXPECT_THROW((void)(markets::stackQuoteGamma(*fixture.root, fixture.rootPillars, dVdRoot, HZeta,
                                                 fixture.reference, children)),
                 std::invalid_argument);
}

void testStackIrsGammaSymmetry() {
    IrsStackFixture fixture(5, {2, 5});
    const auto rebuildChild = [&](const std::shared_ptr<const DiscountCurve<double>>& rootCurve,
                                  const std::vector<markets::ForecastPillar>& pillars) {
        return markets::bootstrapForecastCurve(rootCurve, nullptr, fixture.reference,
                                               fixture.zeroDc, InterpolationScheme::Linear,
                                               pillars);
    };
    const double t = 3.0;
    const double value = fixture.child->discount(t);
    std::vector<double> childWeights;
    fixture.child->zeroNodeWeights(t, childWeights);
    std::vector<double> dVdSpread(fixture.child->size(), 0.0);
    for (std::size_t i = 0; i < childWeights.size(); ++i) {
        dVdSpread[i] = -t * value * childWeights[i];
    }
    std::vector<double> rootWeights;
    fixture.root->zeroNodeWeights(t, rootWeights);
    std::vector<double> dVdRoot(fixture.root->size(), 0.0);
    for (std::size_t i = 0; i < rootWeights.size(); ++i) {
        dVdRoot[i] = -t * value * rootWeights[i];
    }
    std::vector<markets::StackChildInput> children;
    children.push_back(markets::StackChildInput{markets::CurveRole::Forecast, fixture.child.get(),
                                                fixture.childPillars, dVdSpread});
    const std::vector<markets::StackRiskEntry> stack = markets::stackQuoteRisk(
        *fixture.root, fixture.rootPillars, dVdRoot, children, fixture.reference);
    EXPECT_EQ(stack.size(), 2u);
    EXPECT_EQ(stack[1].points.size(), fixture.childPillars.size());

    const double epsilon = 1e-6;
    // Child totals: re-bootstrap the child with the parent frozen.
    for (std::size_t j = 0; j < fixture.childPillars.size(); ++j) {
        auto up = fixture.childPillars;
        auto down = fixture.childPillars;
        up[j].irs.quote += epsilon;
        down[j].irs.quote -= epsilon;
        const double fd = (rebuildChild(fixture.root, up).discount(t) -
                           rebuildChild(fixture.root, down).discount(t)) /
                          (2.0 * epsilon);
        CHECK_CLOSE("irs stack child total vs FD", stack[1].points[j].delta, fd, 1e-4);
    }
    // Root totals: child quotes fixed (spreads re-solved) while the root moves.
    for (std::size_t j = 0; j < fixture.rootPillars.size(); ++j) {
        auto up = fixture.rootPillars;
        auto down = fixture.rootPillars;
        up[j].quote += epsilon;
        down[j].quote -= epsilon;
        const auto rootUp = std::make_shared<DiscountCurve<double>>(markets::bootstrapDiscountCurve(
            fixture.reference, fixture.zeroDc, InterpolationSpace::LogDiscount,
            InterpolationScheme::Linear, up));
        const auto rootDown =
            std::make_shared<DiscountCurve<double>>(markets::bootstrapDiscountCurve(
                fixture.reference, fixture.zeroDc, InterpolationSpace::LogDiscount,
                InterpolationScheme::Linear, down));
        const double fd = (rebuildChild(rootUp, fixture.childPillars).discount(t) -
                           rebuildChild(rootDown, fixture.childPillars).discount(t)) /
                          (2.0 * epsilon);
        CHECK_CLOSE("irs stack root total vs FD", stack[0].points[j].delta, fd, 1e-4);
    }

    // Stack gamma with an arbitrary symmetric portfolio Hessian: dimension,
    // full fill, exactness against re-bootstrapped finite differences,
    // symmetry and a nonzero bootstrap-curvature correction.
    const std::size_t dim = fixture.rootPillars.size() + fixture.childPillars.size();
    std::vector<double> HZeta(dim * dim, 0.0);
    for (std::size_t i = 0; i < dim; ++i) {
        for (std::size_t j = 0; j < dim; ++j) {
            HZeta[i * dim + j] =
                0.05 * static_cast<double>(std::min(i, j) + 1) + (i == j ? 0.1 : 0.0);
        }
    }
    // The gamma consumes the node gradient at the base nodes,
    // p = g + HZeta zeta_0, while the finite-difference reference
    // differentiates the quote gradient x = J^T (g + HZeta zeta) of the
    // quadratic V = g^T zeta + 0.5 zeta^T HZeta zeta.
    std::vector<double> g(dim, 0.0);
    for (std::size_t i = 0; i < fixture.rootPillars.size(); ++i) {
        g[i] = dVdRoot[i + 1];
    }
    for (std::size_t i = 0; i < fixture.childPillars.size(); ++i) {
        g[fixture.rootPillars.size() + i] = dVdSpread[i + 1];
    }
    std::vector<double> zeta(dim, 0.0);
    for (std::size_t i = 0; i < fixture.rootPillars.size(); ++i) {
        zeta[i] = fixture.root->zeros()[i + 1];
    }
    for (std::size_t i = 0; i < fixture.childPillars.size(); ++i) {
        zeta[fixture.rootPillars.size() + i] = fixture.child->spreadNodes().zeros()[i + 1];
    }
    std::vector<double> gammaRoot = dVdRoot;
    std::vector<double> gammaChild = dVdSpread;
    for (std::size_t i = 0; i < fixture.rootPillars.size(); ++i) {
        double p = g[i];
        for (std::size_t j = 0; j < dim; ++j) {
            p += HZeta[i * dim + j] * zeta[j];
        }
        gammaRoot[i + 1] = p;
    }
    for (std::size_t i = 0; i < fixture.childPillars.size(); ++i) {
        const std::size_t k = fixture.rootPillars.size() + i;
        double p = g[k];
        for (std::size_t j = 0; j < dim; ++j) {
            p += HZeta[k * dim + j] * zeta[j];
        }
        gammaChild[i + 1] = p;
    }
    children[0].dVdSpread = gammaChild;
    const markets::StackQuoteGamma gamma = markets::stackQuoteGamma(
        *fixture.root, fixture.rootPillars, gammaRoot, HZeta, fixture.reference, children);
    EXPECT_EQ(gamma.dim, dim);
    EXPECT_EQ(gamma.hessian.size(), dim * dim);
    for (std::size_t i = 0; i < dim; ++i) {
        for (std::size_t j = i + 1; j < dim; ++j) {
            CHECK_CLOSE("stack gamma full symmetry", gamma.at(i, j), gamma.at(j, i), 1e-12);
        }
    }
    const double curvature = checkStackGammaFiniteDifference(fixture, g, HZeta, gamma, 1e-5);
    EXPECT_GT(curvature, 1e-8);
    // A mismatched H_zeta is rejected.
    EXPECT_THROW((void)(markets::stackQuoteGamma(*fixture.root, fixture.rootPillars, gammaRoot,
                                                 std::vector<double>(dim, 0.0), fixture.reference,
                                                 children)),
                 std::invalid_argument);
}

void runDepth2FiniteDifference(const char* prefix, bool exogenousRootDiscount) {
    Depth2StackFixture fixture(exogenousRootDiscount);
    const Depth2Gradients gradients = depth2Gradients(fixture);
    const std::vector<markets::StackCurveInput> inputs =
        depth2StackInputs(fixture, gradients.root, gradients.child, gradients.grand);
    const std::vector<markets::StackRiskEntry> stack =
        markets::stackQuoteRisk(inputs, fixture.reference);
    EXPECT_EQ(stack.size(), 3u);

    const double epsilon = 1e-6;
    const double tolerance = 1e-6;
    const std::string rootLabel = std::string(prefix) + " root quote vs FD";
    for (std::size_t j = 0; j < fixture.rootPillars.size(); ++j) {
        std::vector<CurvePillar> up = fixture.rootPillars;
        std::vector<CurvePillar> down = fixture.rootPillars;
        up[j].quote += epsilon;
        down[j].quote -= epsilon;
        const double fd = (depth2Value(rebuildDepth2Stack(fixture, up, fixture.childPillars,
                                                          fixture.grandPillars),
                                       gradients) -
                           depth2Value(rebuildDepth2Stack(fixture, down, fixture.childPillars,
                                                          fixture.grandPillars),
                                       gradients)) /
                          (2.0 * epsilon);
        CHECK_CLOSE(rootLabel.c_str(), stack[0].points[j].delta, fd, tolerance);
    }
    const std::string childLabel = std::string(prefix) + " child quote vs FD";
    for (std::size_t j = 0; j < fixture.childPillars.size(); ++j) {
        std::vector<markets::ForecastPillar> up = fixture.childPillars;
        std::vector<markets::ForecastPillar> down = fixture.childPillars;
        up[j].basis.spread += epsilon;
        down[j].basis.spread -= epsilon;
        const double fd =
            (depth2Value(rebuildDepth2Stack(fixture, fixture.rootPillars, up, fixture.grandPillars),
                         gradients) -
             depth2Value(
                 rebuildDepth2Stack(fixture, fixture.rootPillars, down, fixture.grandPillars),
                 gradients)) /
            (2.0 * epsilon);
        CHECK_CLOSE(childLabel.c_str(), stack[1].points[j].delta, fd, tolerance);
    }
    const std::string grandLabel = std::string(prefix) + " grandchild quote vs FD";
    for (std::size_t j = 0; j < fixture.grandPillars.size(); ++j) {
        std::vector<markets::ForecastPillar> up = fixture.grandPillars;
        std::vector<markets::ForecastPillar> down = fixture.grandPillars;
        up[j].irs.quote += epsilon;
        down[j].irs.quote -= epsilon;
        const double fd =
            (depth2Value(rebuildDepth2Stack(fixture, fixture.rootPillars, fixture.childPillars, up),
                         gradients) -
             depth2Value(
                 rebuildDepth2Stack(fixture, fixture.rootPillars, fixture.childPillars, down),
                 gradients)) /
            (2.0 * epsilon);
        CHECK_CLOSE(grandLabel.c_str(), stack[2].points[j].delta, fd, tolerance);
    }
}

void runDepth2GammaFiniteDifference(bool exogenousRootDiscount) {
    Depth2StackFixture fixture(exogenousRootDiscount);
    const Depth2Gradients gradients = depth2Gradients(fixture);
    const std::size_t mRoot = fixture.rootPillars.size();
    const std::size_t mChild = fixture.childPillars.size();
    const std::size_t mGrand = fixture.grandPillars.size();
    const std::size_t dim = mRoot + mChild + mGrand;
    std::vector<double> HZeta(dim * dim, 0.0);
    for (std::size_t i = 0; i < dim; ++i) {
        for (std::size_t j = 0; j < dim; ++j) {
            HZeta[i * dim + j] =
                0.02 * static_cast<double>(std::min(i, j) + 1) + (i == j ? 0.05 : 0.0);
        }
    }
    std::vector<double> g(dim, 0.0);
    for (std::size_t i = 0; i < mRoot; ++i) {
        g[i] = gradients.root[i + 1];
    }
    for (std::size_t i = 0; i < mChild; ++i) {
        g[mRoot + i] = gradients.child[i + 1];
    }
    for (std::size_t i = 0; i < mGrand; ++i) {
        g[mRoot + mChild + i] = gradients.grand[i + 1];
    }
    const auto nodeGradientAt = [&](const Depth2Curves& curves) {
        std::vector<double> zeta(dim, 0.0);
        for (std::size_t i = 0; i < mRoot; ++i) {
            zeta[i] = curves.root->zeros()[i + 1];
        }
        for (std::size_t i = 0; i < mChild; ++i) {
            zeta[mRoot + i] = curves.child->spreadNodes().zeros()[i + 1];
        }
        for (std::size_t i = 0; i < mGrand; ++i) {
            zeta[mRoot + mChild + i] = curves.grandchild->spreadNodes().zeros()[i + 1];
        }
        std::vector<double> p(dim, 0.0);
        for (std::size_t i = 0; i < dim; ++i) {
            double sum = g[i];
            for (std::size_t j = 0; j < dim; ++j) {
                sum += HZeta[i * dim + j] * zeta[j];
            }
            p[i] = sum;
        }
        return p;
    };
    const std::vector<CurvePillar> baseRoot = fixture.rootPillars;
    const std::vector<markets::ForecastPillar> baseChild = fixture.childPillars;
    const std::vector<markets::ForecastPillar> baseGrand = fixture.grandPillars;
    const std::vector<double> baseP =
        nodeGradientAt(rebuildDepth2Stack(fixture, baseRoot, baseChild, baseGrand));
    std::vector<double> rootGradient(fixture.root->size(), 0.0);
    std::vector<double> childGradient(fixture.child->size(), 0.0);
    std::vector<double> grandGradient(fixture.grandchild->size(), 0.0);
    for (std::size_t i = 0; i < mRoot; ++i) {
        rootGradient[i + 1] = baseP[i];
    }
    for (std::size_t i = 0; i < mChild; ++i) {
        childGradient[i + 1] = baseP[mRoot + i];
    }
    for (std::size_t i = 0; i < mGrand; ++i) {
        grandGradient[i + 1] = baseP[mRoot + mChild + i];
    }
    const std::vector<markets::StackCurveInput> inputs =
        depth2StackInputs(fixture, rootGradient, childGradient, grandGradient);
    const markets::StackQuoteGamma gamma =
        markets::stackQuoteGamma(inputs, HZeta, fixture.reference);
    EXPECT_EQ(gamma.dim, dim);
    for (std::size_t i = 0; i < dim; ++i) {
        for (std::size_t j = i + 1; j < dim; ++j) {
            CHECK_CLOSE("depth-2 stack gamma symmetry", gamma.at(i, j), gamma.at(j, i), 1e-12);
        }
    }

    const auto quoteGradient = [&](const std::vector<CurvePillar>& rootQuotes,
                                   const std::vector<markets::ForecastPillar>& childQuotes,
                                   const std::vector<markets::ForecastPillar>& grandQuotes) {
        const Depth2Curves curves =
            rebuildDepth2Stack(fixture, rootQuotes, childQuotes, grandQuotes);
        const std::vector<double> p = nodeGradientAt(curves);
        std::vector<markets::StackCurveInput> fdInputs(3);
        fdInputs[0].curve = markets::StackCurveView::make(*curves.root);
        fdInputs[0].role = markets::CurveRole::Discount;
        fdInputs[0].discountPillars = rootQuotes;
        fdInputs[0].dVdNodes.assign(curves.root->size(), 0.0);
        for (std::size_t i = 0; i < mRoot; ++i) {
            fdInputs[0].dVdNodes[i + 1] = p[i];
        }
        fdInputs[1].curve = markets::StackCurveView::make(*curves.child);
        fdInputs[1].role = markets::CurveRole::Forecast;
        fdInputs[1].forecastPillars = childQuotes;
        fdInputs[1].dVdNodes.assign(curves.child->size(), 0.0);
        for (std::size_t i = 0; i < mChild; ++i) {
            fdInputs[1].dVdNodes[i + 1] = p[mRoot + i];
        }
        fdInputs[2].curve = markets::StackCurveView::make(*curves.grandchild);
        fdInputs[2].role = markets::CurveRole::Forecast;
        fdInputs[2].forecastPillars = grandQuotes;
        fdInputs[2].dVdNodes.assign(curves.grandchild->size(), 0.0);
        for (std::size_t i = 0; i < mGrand; ++i) {
            fdInputs[2].dVdNodes[i + 1] = p[mRoot + mChild + i];
        }
        if (exogenousRootDiscount) {
            fdInputs[2].discount = markets::StackCurveView::make(*curves.root);
        }
        const std::vector<markets::StackRiskEntry> entries =
            markets::stackQuoteRisk(fdInputs, fixture.reference);
        std::vector<double> gradient(dim, 0.0);
        std::size_t index = 0;
        for (const markets::StackRiskEntry& entry : entries) {
            for (const markets::QuotePoint& point : entry.points) {
                gradient[index++] = point.delta;
            }
        }
        return gradient;
    };
    const double epsilon = 1e-5;
    for (std::size_t r = 0; r < dim; ++r) {
        auto rootUp = baseRoot;
        auto rootDown = baseRoot;
        auto childUp = baseChild;
        auto childDown = baseChild;
        auto grandUp = baseGrand;
        auto grandDown = baseGrand;
        if (r < mRoot) {
            rootUp[r].quote += epsilon;
            rootDown[r].quote -= epsilon;
        } else if (r < mRoot + mChild) {
            childUp[r - mRoot].basis.spread += epsilon;
            childDown[r - mRoot].basis.spread -= epsilon;
        } else {
            grandUp[r - mRoot - mChild].irs.quote += epsilon;
            grandDown[r - mRoot - mChild].irs.quote -= epsilon;
        }
        const std::vector<double> plus = quoteGradient(rootUp, childUp, grandUp);
        const std::vector<double> minus = quoteGradient(rootDown, childDown, grandDown);
        for (std::size_t a = 0; a < dim; ++a) {
            const double fd = (plus[a] - minus[a]) / (2.0 * epsilon);
            CHECK_CLOSE("depth-2 exact stack gamma vs FD", gamma.at(a, r), fd, 1e-5);
        }
    }
}

void testStackDepth1Equivalence() {
    Depth2StackFixture fixture(false);
    const Depth2Gradients gradients = depth2Gradients(fixture);
    const std::vector<markets::StackCurveInput> inputs =
        depth2StackInputs(fixture, gradients.root, gradients.child, gradients.grand);
    const std::vector<markets::StackCurveInput> depthOneInputs(inputs.begin(), inputs.begin() + 2);
    const std::vector<markets::StackRiskEntry> general =
        markets::stackQuoteRisk(depthOneInputs, fixture.reference);

    std::vector<markets::StackChildInput> children;
    children.push_back(markets::StackChildInput{markets::CurveRole::Forecast, fixture.child.get(),
                                                fixture.childPillars, gradients.child});
    const std::vector<markets::StackRiskEntry> legacy = markets::stackQuoteRisk(
        *fixture.root, fixture.rootPillars, gradients.root, children, fixture.reference);

    EXPECT_EQ(general.size(), legacy.size());
    for (std::size_t k = 0; k < general.size(); ++k) {
        EXPECT_TRUE(general[k].role == legacy[k].role);
        EXPECT_TRUE(samePointMetadata(general[k].points, legacy[k].points));
        CHECK_CLOSE_SEQ("depth-1 equivalence deltas", pointDeltas(general[k].points),
                        pointDeltas(legacy[k].points), 1e-12);
    }

    // The general and legacy gamma overloads agree entry by entry.
    const std::size_t gammaDim = general[0].points.size() + general[1].points.size();
    std::vector<double> HZeta(gammaDim * gammaDim, 0.0);
    for (std::size_t i = 0; i < gammaDim; ++i) {
        for (std::size_t j = 0; j < gammaDim; ++j) {
            HZeta[i * gammaDim + j] =
                0.03 * static_cast<double>(std::min(i, j) + 1) + (i == j ? 0.07 : 0.0);
        }
    }
    const markets::StackQuoteGamma generalGamma =
        markets::stackQuoteGamma(depthOneInputs, HZeta, fixture.reference);
    const markets::StackQuoteGamma legacyGamma = markets::stackQuoteGamma(
        *fixture.root, fixture.rootPillars, gradients.root, HZeta, fixture.reference, children);
    EXPECT_EQ(generalGamma.dim, legacyGamma.dim);
    EXPECT_TRUE(samePointMetadata(generalGamma.points, legacyGamma.points));
    for (std::size_t k = 0; k < generalGamma.hessian.size(); ++k) {
        CHECK_CLOSE("depth-1 equivalence gamma", generalGamma.hessian[k], legacyGamma.hessian[k],
                    1e-12);
    }
}

void testStackRebuildWithNode() {
    Depth2StackFixture fixture(false);
    const markets::StackCurveView::Ptr rootView = markets::StackCurveView::make(*fixture.root);
    const markets::StackCurveView::Ptr childView = markets::StackCurveView::make(*fixture.child);
    const markets::StackCurveView::Ptr grandchildView =
        markets::StackCurveView::make(*fixture.grandchild);

    const double t = 2.5;
    const std::size_t rootNode = 3;
    const double delta = 0.002;
    const markets::StackCurveView::Ptr bumpedRoot =
        rootView->rebuildWithNode(rootNode, delta, nullptr);
    const markets::StackCurveView::Ptr rebuiltChild =
        childView->rebuildWithNode(0, 0.0, bumpedRoot);

    std::vector<double> zeros = fixture.root->zeros();
    zeros[rootNode] += delta;
    const DiscountCurve<double> expectedRoot(fixture.root->times(), zeros, fixture.root->space(),
                                             fixture.root->scheme(), fixture.root->tension(),
                                             fixture.root->switchIndex());
    const DiscountCurve<double>& spreadNodes = fixture.child->spreadNodes();
    const markets::SpreadCurve<double> expectedChild(
        std::make_shared<DiscountCurve<double>>(expectedRoot), spreadNodes.times(),
        spreadNodes.zeros(), spreadNodes.scheme(), spreadNodes.tension());
    CHECK_CLOSE("rebuildWithNode parent bump moves child discount", rebuiltChild->discount(t),
                expectedChild.discount(t), 1e-12);
    EXPECT_TRUE(rebuiltChild->discount(t) != childView->discount(t));

    const markets::StackCurveView::Ptr bumpedChild = childView->rebuildWithNode(1, delta, nullptr);
    CHECK_CLOSE("rebuildWithNode child bump keeps parent discount",
                bumpedChild->parentView()->discount(t), rootView->discount(t), 1e-15);

    EXPECT_THROW((void)(grandchildView->rebuildWithNode(1, delta, rootView)),
                 std::invalid_argument);
}

void testRebuiltViewIdentityCollision() {
    // Two rebuilds of the same base curve with different bumps must never
    // compare identity-equal. Rebuilt views own their copy, so the identity is
    // stable per rebuild instead of the address of a stack temporary, which
    // sequential rebuilds can reuse.
    IrsStackFixture fixture(2, {1});
    const markets::StackCurveView::Ptr base = markets::StackCurveView::make(*fixture.root);
    const markets::StackCurveView::Ptr up = base->rebuildWithNode(1, 1e-3, nullptr);
    const markets::StackCurveView::Ptr down = base->rebuildWithNode(1, -1e-3, nullptr);
    EXPECT_TRUE(up->identity() != down->identity());
    EXPECT_TRUE(up->discount(1.5) != down->discount(1.5));
    EXPECT_TRUE(!markets::sameCurveView(*up, *down));
    // Equal rebuilds still match by value even though each owns its copy.
    const markets::StackCurveView::Ptr upAgain = base->rebuildWithNode(1, 1e-3, nullptr);
    EXPECT_TRUE(up->identity() != upAgain->identity());
    EXPECT_TRUE(markets::sameCurveView(*up, *upAgain));
}

void testStackGammaValidation() {
    markets::StackQuoteGamma gamma;
    gamma.dim = 2;
    gamma.hessian = {1.0, 0.0, 0.0}; // too short for dim x dim
    EXPECT_THROW((void)gamma.at(0, 0), std::invalid_argument);
    gamma.hessian = {1.0, 0.0, 0.0, 1.0};
    CHECK_CLOSE("gamma at in-bounds", gamma.at(1, 1), 1.0, 0.0);
    EXPECT_THROW((void)gamma.at(2, 0), std::invalid_argument);
    markets::StackQuoteGamma empty;
    EXPECT_THROW((void)empty.at(0, 0), std::invalid_argument);

    // Point-metadata validation: a complete point passes, a partial one is
    // rejected, and the gamma gate ties the point count to `dim`.
    const std::vector<markets::QuotePoint> complete{
        markets::QuotePoint{"Irs 1Y", "1Y", 1, markets::CurveRole::Forecast, 0.5}};
    markets::validateQuotePoints(complete, 1);
    EXPECT_THROW((void)(markets::validateQuotePoints(
                     {markets::QuotePoint{"Irs 1Y", "", 1, markets::CurveRole::Forecast, 0.5}}, 1)),
                 std::invalid_argument);
    EXPECT_THROW((void)(markets::validateQuotePoints(complete, 2)), std::invalid_argument);
    markets::StackRiskEntry malformed;
    malformed.points.push_back(markets::QuotePoint{"", "1Y", 1, markets::CurveRole::Forecast, 0.5});
    EXPECT_THROW((void)malformed.validate(), std::invalid_argument);
    gamma.points = complete;
    gamma.dim = 1;
    gamma.hessian = {1.0};
    gamma.validate();
    gamma.points.clear();
    EXPECT_THROW((void)gamma.validate(), std::invalid_argument);
}

void testRebuildWithNodeMetadata() {
    // A non-Act365F date-based curve must keep its zero clock through
    // rebuildWithNode; the times constructor would silently switch the clock.
    const datetime::Date reference(2026, 9, 29);
    const datetime::DayCounter act360(datetime::DayCount::Actual360);
    const std::vector<datetime::Date> dates{reference.plusMonths(6), reference.plusYears(1),
                                            reference.plusYears(2)};
    const std::vector<double> pillars{0.03, 0.033, 0.036};
    const DiscountCurve<double> base(reference, dates, act360, pillars,
                                     InterpolationSpace::LogDiscount, InterpolationScheme::Linear);
    const markets::StackCurveView::Ptr view = markets::StackCurveView::make(base);
    const markets::StackCurveView::Ptr bumped = view->rebuildWithNode(1, 0.002, nullptr);
    EXPECT_TRUE(bumped->zeroDayCounter().convention() == datetime::DayCount::Actual360);
    std::vector<double> expectedZeros = pillars;
    expectedZeros[0] += 0.002;
    const DiscountCurve<double> expected(reference, dates, act360, expectedZeros,
                                         InterpolationSpace::LogDiscount,
                                         InterpolationScheme::Linear);
    EXPECT_TRUE(bumped->times() == expected.times());
    CHECK_CLOSE("rebuilt date-based discount", bumped->discount(1.5), expected.discount(1.5),
                1e-15);

    // The spread adapter keeps the clock of its re-bound parent.
    const auto parent = std::make_shared<DiscountCurve<double>>(base);
    const markets::SpreadCurve<double> child(parent, base.times(),
                                             std::vector<double>{0.0, 0.0005, 0.0007, 0.0009},
                                             InterpolationScheme::Linear);
    const markets::StackCurveView::Ptr childView = markets::StackCurveView::make(child);
    const markets::StackCurveView::Ptr childOver = childView->rebuildWithNode(0, 0.0, bumped);
    EXPECT_TRUE(childOver->zeroDayCounter().convention() == datetime::DayCount::Actual360);
    EXPECT_TRUE(childOver->parentView() != nullptr);
    EXPECT_TRUE(childOver->parentView()->zeroDayCounter().convention() ==
                datetime::DayCount::Actual360);
}

void testStackClockPreservedThroughBumps() {
    // Date-based ACT/360 root and spread child. The quote Jacobian is
    // assembled on the curves' own zero clock; a zero-delta rebuild through
    // bumpStackInputs (the same machinery stackQuoteGamma uses for its
    // plus/minus curvature) must reproduce that Jacobian, which fails when the
    // rebuild falls back to the default ACT/365F times constructor.
    const datetime::Date reference(2026, 9, 29);
    const datetime::DayCounter act360(datetime::DayCount::Actual360);
    const datetime::Calendar calendar = datetime::Calendar::noHolidays();
    const std::vector<datetime::Date> dates{reference.plusYears(1), reference.plusYears(2)};
    const std::vector<double> zeros{0.03, 0.035};
    const DiscountCurve<double> root(reference, dates, act360, zeros,
                                     InterpolationSpace::LogDiscount, InterpolationScheme::Linear);
    std::vector<CurvePillar> rootPillars;
    for (std::size_t i = 0; i < dates.size(); ++i) {
        CurvePillar pillar;
        pillar.kind = PillarKind::OisSwap;
        pillar.maturity = dates[i];
        pillar.quoteDayCounter = act360;
        pillar.calendar = calendar;
        pillar.quote = markets::impliedQuote(pillar, reference, root);
        rootPillars.push_back(pillar);
    }
    const auto parent = std::make_shared<DiscountCurve<double>>(root);
    const markets::SpreadCurve<double> child(
        parent, root.times(), std::vector<double>{0.0, 0.001, 0.0015}, InterpolationScheme::Linear);
    std::vector<markets::ForecastPillar> childPillars;
    for (int year = 1; year <= 2; ++year) {
        markets::ForecastPillar instrument;
        instrument.kind = markets::ForecastPillar::Kind::Irs;
        instrument.irs.maturity = reference.plusYears(year);
        instrument.irs.floatTenor = datetime::Period(3, datetime::TimeUnit::Months);
        instrument.irs.floatCalendar = calendar;
        instrument.irs.floatDayCounter = act360;
        instrument.irs.fixedTenor = datetime::Period(1, datetime::TimeUnit::Years);
        instrument.irs.fixedCalendar = calendar;
        instrument.irs.fixedDayCounter =
            datetime::DayCounter(datetime::DayCount::Thirty360BondBasis);
        childPillars.push_back(instrument);
    }
    std::vector<markets::StackCurveInput> inputs(2);
    inputs[0].curve = markets::StackCurveView::make(root);
    inputs[0].role = markets::CurveRole::Discount;
    inputs[0].discountPillars = rootPillars;
    inputs[0].dVdNodes.assign(root.size(), 0.0);
    inputs[1].curve = markets::StackCurveView::make(child);
    inputs[1].role = markets::CurveRole::Forecast;
    inputs[1].forecastPillars = childPillars;
    inputs[1].dVdNodes.assign(child.size(), 0.0);
    const markets::StackQuoteSystem base = markets::assembleStackQuoteSystem(inputs, reference);
    const markets::StackQuoteSystem rootBumped =
        markets::assembleStackQuoteSystem(markets::bumpStackInputs(inputs, 0, 1, 0.0), reference);
    const markets::StackQuoteSystem childBumped =
        markets::assembleStackQuoteSystem(markets::bumpStackInputs(inputs, 1, 1, 0.0), reference);
    EXPECT_EQ(base.dim, rootBumped.dim);
    EXPECT_EQ(base.dim, childBumped.dim);
    CHECK_CLOSE_SEQ("act360 zero-bump root jacobian", base.jacobian, rootBumped.jacobian, 1e-14);
    CHECK_CLOSE_SEQ("act360 zero-bump child jacobian", base.jacobian, childBumped.jacobian, 1e-14);
}

void testSameCurveSchemeFingerprint() {
    // Same node grid and zeros, different interpolation scheme: the discount
    // functions differ between nodes, so the curves are not interchangeable.
    IrsStackFixture fixture(5, {2, 5});
    const DiscountCurve<double> rootCopy(*fixture.root);
    EXPECT_TRUE(markets::sameCurveValues(rootCopy, *fixture.root));
    const DiscountCurve<double> tensionRoot(
        fixture.root->times(), fixture.root->zeros(), fixture.root->space(),
        InterpolationScheme::TensionSpline, 8.0, fixture.root->switchIndex());
    EXPECT_TRUE(fixture.root->discount(0.5) != tensionRoot.discount(0.5));
    EXPECT_TRUE(!markets::sameCurveValues(tensionRoot, *fixture.root));
    EXPECT_TRUE(!markets::sameCurveView(*markets::StackCurveView::make(tensionRoot),
                                        *markets::StackCurveView::make(*fixture.root)));

    std::vector<double> dVdRoot(fixture.root->size(), 0.0);
    dVdRoot[1] = 1.0;
    std::vector<double> dVdSpread(fixture.child->size(), 0.0);
    dVdSpread[1] = 1.0;
    std::vector<markets::StackChildInput> children;
    children.push_back(markets::StackChildInput{markets::CurveRole::Forecast, fixture.child.get(),
                                                fixture.childPillars, dVdSpread});
    children[0].discountCurve = &tensionRoot;
    EXPECT_THROW((void)(markets::stackQuoteRisk(*fixture.root, fixture.rootPillars, dVdRoot,
                                                children, fixture.reference)),
                 std::invalid_argument);

    // The off-parent discount shortcut must not fold the discount row into the
    // forecast-parent cross block when the scheme differs.
    const std::size_t mChild = fixture.childPillars.size();
    const std::size_t mParent = fixture.root->size() - 1;
    std::vector<double> expected(mChild * mParent, 0.0);
    std::vector<double> fRow;
    std::vector<double> parentRow;
    std::vector<double> discountRow;
    for (std::size_t j = 0; j < mChild; ++j) {
        markets::forecastPillarJacobianRows(*fixture.child, fixture.childPillars[j],
                                            fixture.reference, &tensionRoot, fRow, parentRow,
                                            discountRow);
        for (std::size_t i = 0; i < mParent; ++i) {
            expected[j * mParent + i] = parentRow[i];
        }
    }
    std::vector<double> f;
    std::vector<double> explicitCross;
    markets::assembleForecastJacobian(*fixture.child, fixture.childPillars, fixture.reference,
                                      &tensionRoot, f, explicitCross);
    EXPECT_TRUE(explicitCross == expected);
}

void testStackInputValidation() {
    Depth2StackFixture fixture(false);
    const Depth2Gradients gradients = depth2Gradients(fixture);

    {
        markets::StackCurveInput input;
        input.curve = markets::StackCurveView::make(*fixture.root);
        input.discountPillars = fixture.rootPillars;
        input.discountPillars.pop_back();
        input.dVdNodes = gradients.root;
        EXPECT_THROW((void)(markets::stackQuoteRisk({input}, fixture.reference)),
                     std::invalid_argument);
    }
    {
        const std::vector<double> otherTimes{0.0, 1.0, 2.0};
        const std::vector<double> otherZeros{0.0, 0.02, 0.03};
        const DiscountCurve<double> other(otherTimes, otherZeros, InterpolationSpace::LogDiscount,
                                          InterpolationScheme::Linear);
        std::vector<markets::StackCurveInput> inputs =
            depth2StackInputs(fixture, gradients.root, gradients.child, gradients.grand);
        inputs[1].discount = markets::StackCurveView::make(other);
        EXPECT_THROW((void)(markets::stackQuoteRisk(inputs, fixture.reference)),
                     std::invalid_argument);
    }
    {
        const std::vector<double> standaloneTimes{0.0, 1.0, 2.0, 3.0};
        const std::vector<double> standaloneZeros{0.0, 0.03, 0.03, 0.03};
        const DiscountCurve<double> standalone(standaloneTimes, standaloneZeros,
                                               InterpolationSpace::LogDiscount,
                                               InterpolationScheme::Linear);
        markets::StackCurveInput input;
        input.curve = markets::StackCurveView::make(standalone);
        input.role = markets::CurveRole::Forecast;
        input.forecastPillars = fixture.childPillars;
        input.dVdNodes.assign(standalone.size(), 0.0);
        EXPECT_THROW((void)(markets::stackQuoteRisk({input}, fixture.reference)),
                     std::invalid_argument);
    }
}

void testStackLadderHedgeDual() {
    Fixture fx(InterpolationSpace::LogDiscount, InterpolationScheme::Linear);
    auto parent = std::make_shared<DiscountCurve<double>>(*fx.curve);
    const datetime::DayCounter zeroDc(datetime::DayCount::Actual365Fixed);
    std::vector<markets::BasisPillar> basisPillars;
    for (std::size_t i = 1; i < fx.dates.size(); ++i) {
        markets::BasisPillar pillar;
        pillar.maturity = fx.dates[i];
        pillar.floatTenor = datetime::Period(3, datetime::TimeUnit::Months);
        pillar.quoteDayCounter = zeroDc;
        pillar.calendar = datetime::Calendar::noHolidays();
        pillar.spread = 0.0005 + 0.0001 * static_cast<double>(i);
        basisPillars.push_back(pillar);
    }
    const auto child = markets::bootstrapSpreadCurve(parent, fx.reference, zeroDc,
                                                     InterpolationScheme::Linear, basisPillars);
    const double t = 4.5;
    const double value = child.discount(t);
    std::vector<double> childWeights;
    child.zeroNodeWeights(t, childWeights);
    std::vector<double> dVdSpread(child.size(), 0.0);
    for (std::size_t i = 0; i < childWeights.size(); ++i) {
        dVdSpread[i] = -t * value * childWeights[i];
    }
    std::vector<double> rootWeights;
    parent->zeroNodeWeights(t, rootWeights);
    std::vector<double> dVdRoot(rootWeights.size());
    for (std::size_t i = 0; i < rootWeights.size(); ++i) {
        dVdRoot[i] = -t * value * rootWeights[i];
    }
    std::vector<markets::StackChildInput> children;
    children.push_back(markets::StackChildInput{markets::CurveRole::Forecast, &child,
                                                asForecastPillars(basisPillars), dVdSpread});
    const auto stack =
        markets::stackQuoteRisk(*parent, fx.pillars, dVdRoot, children, fx.reference);
    double total = 0.0;
    for (const auto& entry : stack) {
        for (const markets::QuotePoint& point : entry.points) {
            total += point.delta;
        }
    }
    double ladderTotal = 0.0;
    bool sawBasis = false;
    for (const markets::RiskBucket& bucket : markets::stackYearLadder(stack)) {
        ladderTotal += bucket.delta;
        sawBasis = sawBasis || bucket.label == "4Y" || bucket.label == "5Y";
    }
    CHECK_CLOSE("stack ladder total", ladderTotal, total, 1e-12);
    EXPECT_TRUE(sawBasis);
    EXPECT_TRUE(!stack[0].points.empty());
    EXPECT_TRUE(!stack[1].points.empty());

    // Hedge solve: 2x2 sensitivity matrix, exact recovery.
    const std::vector<double> jacobian{1.0, 0.5, 0.5, 1.0};
    const std::vector<double> delta{1.0, -0.5};
    const std::vector<double> hedge = markets::solveHedge(jacobian, 2, 2, delta, 0.0);
    CHECK_CLOSE("hedge bucket 0", jacobian[0] * hedge[0] + jacobian[1] * hedge[1], -delta[0],
                1e-12);
    CHECK_CLOSE("hedge bucket 1", jacobian[2] * hedge[0] + jacobian[3] * hedge[1], -delta[1],
                1e-12);

    // Dual pricing/risk scheme view: Linear pricing, Akima risk.
    const DiscountCurve<double> riskCurve =
        markets::bootstrapDiscountCurve(fx.reference, zeroDc, InterpolationSpace::LogDiscount,
                                        InterpolationScheme::Akima, fx.pillars);
    const markets::DualSchemeRisk dual =
        markets::dualSchemeQuoteRisk(*parent, riskCurve, fx.pillars, fx.reference, dVdRoot);
    double maxAudit = 0.0;
    double maxBasis = 0.0;
    for (std::size_t j = 0; j < dual.audit.size(); ++j) {
        CHECK_CLOSE("dual basis identity", dual.basis[j], dual.hedge[j] - dual.audit[j], 1e-15);
        maxAudit = std::max(maxAudit, std::abs(dual.audit[j]));
        maxBasis = std::max(maxBasis, std::abs(dual.basis[j]));
    }
    EXPECT_LE(maxBasis, 0.3 * maxAudit + 1e-6);

    // Off-node coupons: semi-annual fixed legs on annual nodes evaluate inside
    // interpolation segments, where the Linear and Akima Jacobians differ. With
    // annual legs every coupon lands exactly on a node, where every scheme is
    // nodal and the dual basis is structurally zero.
    std::vector<CurvePillar> semiPillars;
    semiPillars.push_back(fx.pillars.front());
    for (std::size_t i = 1; i < fx.pillars.size(); ++i) {
        CurvePillar pillar = fx.pillars[i];
        pillar.fixedTenor = datetime::Period(6, datetime::TimeUnit::Months);
        pillar.quote = markets::impliedQuote(pillar, fx.reference, *fx.curve);
        semiPillars.push_back(pillar);
    }
    const DiscountCurve<double> semiPricing =
        markets::bootstrapDiscountCurve(fx.reference, zeroDc, InterpolationSpace::LogDiscount,
                                        InterpolationScheme::Linear, semiPillars);
    const DiscountCurve<double> semiRisk =
        markets::bootstrapDiscountCurve(fx.reference, zeroDc, InterpolationSpace::LogDiscount,
                                        InterpolationScheme::Akima, semiPillars);
    std::vector<double> semiWeights;
    semiPricing.zeroNodeWeights(t, semiWeights);
    const double semiDf = semiPricing.discount(t);
    std::vector<double> dVdSemi(semiWeights.size());
    for (std::size_t i = 0; i < semiWeights.size(); ++i) {
        dVdSemi[i] = -t * semiDf * semiWeights[i];
    }
    const markets::DualSchemeRisk semiDual =
        markets::dualSchemeQuoteRisk(semiPricing, semiRisk, semiPillars, fx.reference, dVdSemi);
    double maxSemiBasis = 0.0;
    for (const double value : semiDual.basis) {
        maxSemiBasis = std::max(maxSemiBasis, std::abs(value));
    }
    EXPECT_GT(maxSemiBasis, 1e-8);
}

void testRiskReport() {
    Fixture fixture(InterpolationSpace::LogDiscount, InterpolationScheme::Linear);
    const double t = 4.5;
    std::vector<double> weights;
    fixture.curve->zeroNodeWeights(t, weights);
    const double df = fixture.curve->discount(t);
    std::vector<double> dVdZeros(weights.size());
    for (std::size_t i = 0; i < weights.size(); ++i) {
        dVdZeros[i] = -t * df * weights[i];
    }
    const markets::CurveRiskReport report =
        markets::curveRiskReport(*fixture.curve, fixture.pillars, fixture.reference, dVdZeros);
    EXPECT_EQ(report.quoteDeltas.size(), fixture.pillars.size());
    EXPECT_EQ(report.quoteLabels.size(), fixture.pillars.size());

    double kindSum = 0.0;
    for (const markets::RiskBucket& bucket : report.byKind) {
        kindSum += bucket.delta;
    }
    double endSum = 0.0;
    for (const markets::RiskBucket& bucket : report.byEnd) {
        endSum += bucket.delta;
    }
    double yearSum = 0.0;
    for (const markets::RiskBucket& bucket : report.byYear) {
        yearSum += bucket.delta;
    }
    CHECK_CLOSE("report kind total", kindSum, report.totalDelta(), 1e-12);
    CHECK_CLOSE("report end total", endSum, report.totalDelta(), 1e-12);
    CHECK_CLOSE("report year total", yearSum, report.totalDelta(), 1e-12);
    CHECK_CLOSE("report horizon", report.horizonYears, 5.0, 0.01);

    std::vector<double> u;
    fixture.curve->zeroNodeWeights(t, u);
    const double dfGamma = fixture.curve->discount(t);
    std::vector<double> H(u.size() * u.size(), 0.0);
    for (std::size_t i = 0; i < u.size(); ++i) {
        for (std::size_t j = 0; j < u.size(); ++j) {
            H[i * u.size() + j] = t * t * dfGamma * u[i] * u[j];
        }
    }
    const markets::CurveRiskReport withGamma = markets::curveRiskReport(
        *fixture.curve, fixture.pillars, fixture.reference,
        [&] {
            std::vector<double> g(u.size());
            for (std::size_t i = 0; i < u.size(); ++i) {
                g[i] = -t * df * u[i];
            }
            return g;
        }(),
        {}, &H);

    EXPECT_EQ(withGamma.gammaDiagonal.size(), fixture.pillars.size());
    EXPECT_TRUE(!withGamma.gammaCrossByYear.empty());
    double yearGammaSum = 0.0;
    for (const markets::RiskBucket& bucket : withGamma.gammaDiagByYear) {
        yearGammaSum += bucket.delta;
    }
    double quoteGammaSum = 0.0;
    for (const double value : withGamma.gammaDiagonal) {
        quoteGammaSum += value;
    }
    CHECK_CLOSE("gamma year total", yearGammaSum, quoteGammaSum, 1e-10);
}

struct NodeWeightCase {
    InterpolationSpace space;
    InterpolationScheme scheme;
    double tension;
    const char* label;
};

const NodeWeightCase kNodeWeightCases[] = {
    {InterpolationSpace::Zero, InterpolationScheme::Linear, 0.0, "zero linear"},
    {InterpolationSpace::LogDiscount, InterpolationScheme::Linear, 0.0, "log-discount linear"},
    {InterpolationSpace::Zero, InterpolationScheme::Akima, 0.0, "zero akima"},
    {InterpolationSpace::LogDiscount, InterpolationScheme::TensionSpline, 8.0,
     "log-discount tension 8"},
};

struct QuoteRiskCase {
    InterpolationSpace space;
    InterpolationScheme scheme;
    double tension;
    double tolerance;
    const char* label;
};

const QuoteRiskCase kQuoteRiskCases[] = {
    {InterpolationSpace::LogDiscount, InterpolationScheme::Linear, 0.0, 1e-8,
     "log-discount linear"},
    // Akima rows are primal-pinned subgradients (branch-adaptive slopes), so
    // FD agreement is at kink tolerance rather than machine precision.
    {InterpolationSpace::Zero, InterpolationScheme::Akima, 0.0, 2e-4, "zero akima"},
    {InterpolationSpace::LogDiscount, InterpolationScheme::TensionSpline, 8.0, 1e-8,
     "log-discount tension 8"},
};

} // namespace

class CurveRiskTest : public ::testing::Test {};

TEST_F(CurveRiskTest, nodeWeightsPartitionAndFiniteDifference) {
    for (const NodeWeightCase& param : kNodeWeightCases) {
        SCOPED_TRACE(param.label);
        testNodeWeightsFiniteDifference(param.space, param.scheme, param.tension);
    }
    SCOPED_TRACE("zero time weights");
    testZeroTimeNodeWeights();
}

TEST_F(CurveRiskTest, frontEndAndExtrapolatedNodeWeights) {
    SCOPED_TRACE("front-end completeness");
    testFrontEndRiskCompleteness();
    SCOPED_TRACE("extrapolated weights");
    testExtrapolatedNodeWeights();
}

TEST_F(CurveRiskTest, quoteRiskMatchesBumpRebootstrap) {
    for (const QuoteRiskCase& param : kQuoteRiskCases) {
        SCOPED_TRACE(param.label);
        testQuoteRiskFiniteDifference(param.space, param.scheme, param.tension, param.tolerance);
    }
}

TEST_F(CurveRiskTest, denseLuAndQuoteGamma) {
    SCOPED_TRACE("dense LU replay");
    testDenseLuMatchesSolveDense();
    SCOPED_TRACE("quote gamma");
    testQuoteGamma();
}

TEST_F(CurveRiskTest, riskReportBucketsAndGammaTotals) {
    testRiskReport();
}

TEST_F(CurveRiskTest, fraRepoGranularRisk) {
    testFraRepoGranular();
}

TEST_F(CurveRiskTest, basisStackAndLadderRisk) {
    SCOPED_TRACE("basis risk representation");
    testBasisRiskRepresentation();
    SCOPED_TRACE("stack risk tree pass");
    testStackRiskTreePass();
    SCOPED_TRACE("stack ladder hedge dual");
    testStackLadderHedgeDual();
}

TEST_F(CurveRiskTest, stackIrsChildRiskAndGammaSymmetry) {
    SCOPED_TRACE("child risk and gamma");
    testStackIrsChildRiskAndGamma();
    SCOPED_TRACE("gamma symmetry");
    testStackIrsGammaSymmetry();
}

TEST_F(CurveRiskTest, stackDepth2RiskAndGamma) {
    SCOPED_TRACE("depth-2 chain");
    runDepth2FiniteDifference("depth-2 chain", false);
    SCOPED_TRACE("depth-2 off-parent discount");
    runDepth2FiniteDifference("depth-2 off-parent discount", true);
    SCOPED_TRACE("depth-2 chain gamma");
    runDepth2GammaFiniteDifference(false);
    SCOPED_TRACE("depth-2 off-parent discount gamma");
    runDepth2GammaFiniteDifference(true);
}

TEST_F(CurveRiskTest, stackDepth1AndRebuildGates) {
    SCOPED_TRACE("depth-1 equivalence");
    testStackDepth1Equivalence();
    SCOPED_TRACE("rebuild with node");
    testStackRebuildWithNode();
    SCOPED_TRACE("rebuilt view identity");
    testRebuiltViewIdentityCollision();
    SCOPED_TRACE("rebuild metadata");
    testRebuildWithNodeMetadata();
    SCOPED_TRACE("clock preserved through bumps");
    testStackClockPreservedThroughBumps();
}

TEST_F(CurveRiskTest, stackValidationGates) {
    SCOPED_TRACE("gamma validation");
    testStackGammaValidation();
    SCOPED_TRACE("input validation");
    testStackInputValidation();
}

TEST_F(CurveRiskTest, sameCurveSchemeFingerprint) {
    testSameCurveSchemeFingerprint();
}
