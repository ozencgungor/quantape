#include "quantape/format/Number.h"
#include "quantape/log/Log.h"
#include "quantape/markets/Curves/CurveBuilder.h"
#include "quantape/markets/Curves/CurveRisk.h"
#include "quantape/markets/Curves/CurveRiskReport.h"
#include "quantape/markets/Curves/DiscountCurve.h"
#include "quantape/markets/Curves/StackRisk.h"
#include "quantape/util/Check.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <iostream>
#include <memory>
#include <optional>
#include <string>
#include <vector>

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
            for (const double delta : entry.quoteDeltas) {
                gradient[index++] = delta;
            }
        }
        return gradient;
    };

    const double epsilon = 1e-5;
    double maxDeviation = 0.0;
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
            maxDeviation = std::max(maxDeviation, std::abs(gamma.at(a, r) - fd));
            util::checkClose("exact stack gamma vs FD", gamma.at(a, r), fd, tolerance);
        }
    }
    QTA_LOG_INFO("test", "exact stack gamma max FD deviation: {}", maxDeviation);

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

std::string padRight(std::string text, std::size_t width) {
    if (text.size() < width) {
        text.append(width - text.size(), ' ');
    }
    return text;
}

std::string padLeft(std::string text, std::size_t width) {
    if (text.size() < width) {
        text.insert(0, width - text.size(), ' ');
    }
    return text;
}

std::string valueText(double value, int precision = 10) {
    return format::toString(value, format::FloatFormat::General, precision);
}

std::string maturityLabel(double years) {
    return format::toString(years, format::FloatFormat::Fixed, 2) + "y";
}

void logBuckets(const char* title, const std::vector<markets::RiskBucket>& buckets) {
    std::string out = std::string("  ") + title + "\n";
    for (const markets::RiskBucket& bucket : buckets) {
        out += "      " + padRight(bucket.label, 14) + " " + padLeft(valueText(bucket.delta), 18) +
               "\n";
    }
    if (!out.empty() && out.back() == '\n') {
        out.pop_back();
    }
    QTA_LOG_INFO("test", "{}", out);
}

void logDeltas(const char* title, const std::vector<std::string>& labels,
               const std::vector<double>& deltas) {
    std::string out = std::string("  ") + title + "\n";
    for (std::size_t i = 0; i < deltas.size(); ++i) {
        out += "      " + padRight(i < labels.size() ? labels[i] : std::string("?"), 14) + " " +
               padLeft(valueText(deltas[i]), 18) + "\n";
    }
    if (!out.empty() && out.back() == '\n') {
        out.pop_back();
    }
    QTA_LOG_INFO("test", "{}", out);
}

void logReport(const char* title, const markets::CurveRiskReport& report) {
    std::string out = std::string("=== ") + title + " ===\n";
    out += "  horizon " + format::toString(report.horizonYears, format::FloatFormat::Fixed, 3) +
           "y   total " + valueText(report.totalDelta()) + "\n";
    out += "  quote deltas:\n";
    for (std::size_t i = 0; i < report.quoteDeltas.size(); ++i) {
        out +=
            "      " +
            padRight(i < report.quoteLabels.size() ? report.quoteLabels[i] : std::string("?"), 14) +
            " " + padLeft(valueText(report.quoteDeltas[i]), 18) + "\n";
    }
    const auto appendBuckets = [&](const char* name,
                                   const std::vector<markets::RiskBucket>& buckets) {
        out += std::string("  ") + name + "\n";
        for (const markets::RiskBucket& bucket : buckets) {
            out += "      " + padRight(bucket.label, 14) + " " +
                   padLeft(valueText(bucket.delta), 18) + "\n";
        }
    };
    appendBuckets("by kind:", report.byKind);
    appendBuckets("by end:", report.byEnd);
    appendBuckets("by year:", report.byYear);
    if (!report.gammaDiagonal.empty()) {
        out += "  gamma diagonal:\n";
        for (std::size_t i = 0; i < report.gammaDiagonal.size(); ++i) {
            out +=
                "      " +
                padRight(i < report.quoteLabels.size() ? report.quoteLabels[i] : std::string("?"),
                         14) +
                " " + padLeft(valueText(report.gammaDiagonal[i]), 18) + "\n";
        }
        appendBuckets("gamma by year:", report.gammaDiagByYear);
        const std::size_t n = report.gammaBucketLabels.size();
        out += "  gamma cross by year (" + std::to_string(n) + "x" + std::to_string(n) + "):\n";
        out += "      " + padRight("year", 10);
        for (const std::string& label : report.gammaBucketLabels) {
            out += padLeft(label, 12);
        }
        out += "\n";
        for (std::size_t i = 0; i < n; ++i) {
            out += "      " + padRight(report.gammaBucketLabels[i], 10);
            for (std::size_t j = 0; j < n; ++j) {
                out += padLeft(valueText(report.gammaCrossByYear[i * n + j], 6), 12);
            }
            out += "\n";
        }
    }
    if (!out.empty() && out.back() == '\n') {
        out.pop_back();
    }
    QTA_LOG_INFO("test", "{}", out);
}

void logStack(const char* title, const std::vector<markets::StackRiskEntry>& entries) {
    std::string out = std::string("=== ") + title + " ===\n";
    for (const markets::StackRiskEntry& entry : entries) {
        out += "  role " + std::string(markets::curveRoleName(entry.role)) + "\n";
        for (std::size_t j = 0; j < entry.quoteDeltas.size(); ++j) {
            out += "      " +
                   padRight(j < entry.quoteLabels.size() ? entry.quoteLabels[j] : std::string("?"),
                            14) +
                   " " + padLeft(valueText(entry.quoteDeltas[j]), 18) + "\n";
        }
    }
    if (!out.empty() && out.back() == '\n') {
        out.pop_back();
    }
    QTA_LOG_INFO("test", "{}", out);
}

void logMatrix(const char* title, const std::vector<double>& matrix, std::size_t rows,
               std::size_t cols, const std::vector<std::string>& rowLabels,
               const std::vector<std::string>& colLabels) {
    std::string out = std::string("  ") + title + " (" + std::to_string(rows) + "x" +
                      std::to_string(cols) + ")\n";
    out += "        " + padRight("", 8);
    for (const std::string& label : colLabels) {
        out += padLeft(label, 10);
    }
    out += "\n";
    for (std::size_t i = 0; i < rows; ++i) {
        out += "        " + padRight(rowLabels[i], 8);
        for (std::size_t j = 0; j < cols; ++j) {
            out += padLeft(valueText(matrix[i * cols + j], 5), 10);
        }
        out += "\n";
    }
    if (!out.empty() && out.back() == '\n') {
        out.pop_back();
    }
    QTA_LOG_INFO("test", "{}", out);
}

std::vector<std::string> pillarLabels(const datetime::Date& reference,
                                      const std::vector<CurvePillar>& pillars) {
    const datetime::DayCounter zeroDc(datetime::DayCount::Actual365Fixed);
    std::vector<std::string> labels;
    for (const CurvePillar& pillar : pillars) {
        const datetime::Date maturity = markets::pillarRiskMaturity(pillar);
        const double t = datetime::yearFraction(reference, maturity, zeroDc);
        labels.push_back(std::string(markets::pillarKindName(pillar.kind)) + " " +
                         markets::riskMaturityTag(maturity, t));
    }
    return labels;
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

    std::vector<std::string> rowLabels;
    for (const markets::BasisPillar& pillar : basisPillars) {
        rowLabels.push_back(
            maturityLabel(datetime::yearFraction(fx.reference, pillar.maturity, zeroDc)));
    }
    std::vector<std::string> childLabels;
    for (std::size_t k = 1; k <= m; ++k) {
        childLabels.push_back(maturityLabel(child.spreadNodes().times()[k]));
    }
    std::vector<std::string> parentLabels;
    for (std::size_t i = 1; i <= mp; ++i) {
        parentLabels.push_back(maturityLabel(parent->times()[i]));
    }
    QTA_LOG_INFO("test", "=== basis risk representation ===");
    QTA_LOG_INFO("test",
                 "quoted spread on the {} leg; F = d b_j / d spread_k, C = d b_j / d z_parent_i",
                 basisPillars.front().spreadOnParentLeg ? "OIS parent" : "IBOR child");
    logMatrix("own-curve block F", f, m, m, rowLabels, childLabels);
    logMatrix("cross block C", c, m, mp, rowLabels, parentLabels);

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
            util::checkClose("basis own-row vs FD", f[j * m + (k - 1)], fd, 1e-6);
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
            util::checkClose("basis cross-row vs FD", c[j * mp + (i - 1)], fd, 1e-6);
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
    const std::vector<markets::StackRiskEntry> stack =
        markets::stackQuoteRisk(*parent, fx.pillars, dVdRoot, children, fx.reference);
    logStack("basis stack quote risk (root vs basis child)", stack);
    logBuckets("basis stack year ladder:", markets::stackYearLadder(stack));
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
        util::checkClose("front-end quote risk vs FD", risk.quoteDeltas[j], fd, 1e-8);
    }
    CHECK(std::abs(risk.quoteDeltas.front()) > 1e-3);
    for (std::size_t j = 1; j < risk.quoteDeltas.size(); ++j) {
        CHECK(risk.quoteDeltas[j] == 0.0);
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
        CHECK(weight == 0.0);
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
            util::checkClose("extrapolated node weight vs FD", weights[j], fd, 1e-5);
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
        util::checkClose("quote risk vs FD", risk.quoteDeltas[j], fd, tolerance);
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
    util::checkClose("node weights partition", total, 1.0, partitionTol);

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
        util::checkClose("node weight vs FD", weights[j], fd, 1e-5);
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
    util::checkClose("LU direct solve", direct, reference, 1e-14);
    for (std::size_t i = 0; i < n; ++i) {
        const double combined = unit0[i] * rhs[0] + unit1[i] * rhs[1] + unit2[i] * rhs[2];
        util::checkClose("LU column combination", combined, reference[i], 1e-14);
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
    CHECK(gamma.dim == m);

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
        util::checkClose("gamma diagonal vs reference", gamma.at(i, i), reference[i * m + i], 1e-8);
    }
    const std::size_t i0 = 1;
    const std::size_t j0 = 3;
    util::checkClose("gamma cross vs reference", gamma.at(i0, j0), reference[i0 * m + j0], 1e-8);
    util::checkClose("gamma symmetry", gamma.at(i0, j0), gamma.at(j0, i0), 1e-12);
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
        util::checkClose("fra/repo reprice", markets::impliedQuote(pillar, reference, curve),
                         pillar.quote, 1e-11);
    }

    // Jacobian rows for the new kinds against finite differences of the
    // implied quote in the bootstrap node values.
    for (const CurvePillar& pillar : {pillars[1], pillars[2]}) {
        std::vector<double> row;
        CHECK(markets::pillarJacobianRow(pillar, reference, curve, row));
        const std::size_t n = curve.size();
        for (std::size_t i = 1; i < n; ++i) {
            std::vector<double> zeros = curve.zeros();
            zeros[i] += 1e-8;
            const DiscountCurve<double> bumped(
                curve.times(), zeros, InterpolationSpace::LogDiscount, InterpolationScheme::Linear);
            const double fd = (markets::impliedQuote(pillar, reference, bumped) -
                               markets::impliedQuote(pillar, reference, curve)) /
                              1e-8;
            util::checkClose("fra/repo jacobian vs FD", row[i - 1], fd, 1e-5);
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
    CHECK(sawFra);
    CHECK(sawRepo);

    const std::vector<std::pair<markets::CurveRole, markets::CurveRiskReport>> multi{
        {markets::CurveRole::Discount, report}, {markets::CurveRole::Forecast, report}};
    const std::vector<markets::RiskBucket> buckets = markets::aggregateByRoleAndKind(multi);
    double total = 0.0;
    for (const markets::RiskBucket& bucket : buckets) {
        total += bucket.delta;
    }
    util::checkClose("multi-curve granular total", total, 2.0 * report.totalDelta(), 1e-12);
    logBuckets("multi-role granular buckets:", buckets);
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
    logStack("stack tree pass (root + basis child)", stack);
    CHECK(stack.size() == 2);

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
        util::checkClose("stack child total vs FD", stack[1].quoteDeltas[j], fd, 1e-4);
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
        util::checkClose("stack root total vs FD", stack[0].quoteDeltas[j], fd, 1e-4);
    }
}

void testStackIrsChildRiskAndGamma() {
    // One root pillar and one IRS child pillar: the stack Jacobian reduces to
    // scalar blocks F_p = [a], F_c = [c], C = [d], so J can be written by hand.
    IrsStackFixture fixture(1, {1});
    const std::size_t mRoot = fixture.rootPillars.size();
    const std::size_t mChild = fixture.childPillars.size();
    CHECK(mRoot == 1);
    CHECK(mChild == 1);
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
    CHECK(gamma.dim == 2);
    util::checkClose("stack gamma symmetry", gamma.at(0, 1), gamma.at(1, 0), 1e-14);
    logMatrix("exact stack gamma", gamma.hessian, 2, 2, gamma.quoteLabels, gamma.quoteLabels);
    const double curvature = checkStackGammaFiniteDifference(fixture, g, HZeta, gamma, 1e-5);
    CHECK(curvature > 1e-9);
    QTA_LOG_INFO("test", "stack gamma curvature vs Gauss-Newton: {}", curvature);

    // The stack deltas are J^T (g + HZeta zeta_0) with the same node gradient.
    const std::vector<markets::StackRiskEntry> stack = markets::stackQuoteRisk(
        *fixture.root, fixture.rootPillars, dVdRoot, children, fixture.reference);
    const double expectedRoot = jacobian[0][0] * dVdRoot[1] + jacobian[1][0] * dVdSpread[1];
    const double expectedChild = jacobian[0][1] * dVdRoot[1] + jacobian[1][1] * dVdSpread[1];
    util::checkClose("stack delta root equals J^T g", stack[0].quoteDeltas[0], expectedRoot, 1e-12);
    util::checkClose("stack delta child equals J^T g", stack[1].quoteDeltas[0], expectedChild,
                     1e-12);

    // Labels, years and roles follow the stackQuoteRisk entry order.
    std::vector<std::string> labels;
    std::vector<int> years;
    std::vector<markets::CurveRole> roles;
    for (const markets::StackRiskEntry& entry : stack) {
        for (std::size_t j = 0; j < entry.quoteDeltas.size(); ++j) {
            labels.push_back(entry.quoteLabels[j]);
            years.push_back(entry.quoteYears[j]);
            roles.push_back(entry.role);
        }
    }
    CHECK(gamma.quoteLabels == labels);
    CHECK(gamma.quoteYears == years);
    CHECK(gamma.roles == roles);
    CHECK(gamma.quoteLabels.back() == "Irs 1Y");

    // An equal discount curve as a different object is accepted and produces
    // the same gamma as the implicit forecast-parent discounting.
    const auto rootCopy = std::make_shared<DiscountCurve<double>>(*fixture.root);
    children[0].discountCurve = rootCopy.get();
    const markets::StackQuoteGamma gammaCopy = markets::stackQuoteGamma(
        *fixture.root, fixture.rootPillars, dVdRoot, HZeta, fixture.reference, children);
    for (std::size_t k = 0; k < gamma.hessian.size(); ++k) {
        util::checkClose("stack gamma equal discount copy", gammaCopy.hessian[k], gamma.hessian[k],
                         1e-15);
    }

    // A discount curve outside the root grid is rejected for IRS children.
    const std::vector<double> otherTimes{0.0, 0.5, 1.0};
    const std::vector<double> otherZeros{0.0, 0.02, 0.03};
    const DiscountCurve<double> otherDiscount(
        otherTimes, otherZeros, InterpolationSpace::LogDiscount, InterpolationScheme::Linear);
    children[0].discountCurve = &otherDiscount;
    bool threw = false;
    try {
        (void)markets::stackQuoteRisk(*fixture.root, fixture.rootPillars, dVdRoot, children,
                                      fixture.reference);
    } catch (const std::invalid_argument&) {
        threw = true;
    }
    CHECK(threw);
    threw = false;
    try {
        (void)markets::stackQuoteGamma(*fixture.root, fixture.rootPillars, dVdRoot, HZeta,
                                       fixture.reference, children);
    } catch (const std::invalid_argument&) {
        threw = true;
    }
    CHECK(threw);
    logStack("stack IRS child quote risk", stack);
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
    CHECK(stack.size() == 2);
    CHECK(stack[1].quoteDeltas.size() == fixture.childPillars.size());

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
        util::checkClose("irs stack child total vs FD", stack[1].quoteDeltas[j], fd, 1e-4);
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
        util::checkClose("irs stack root total vs FD", stack[0].quoteDeltas[j], fd, 1e-4);
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
    CHECK(gamma.dim == dim);
    CHECK(gamma.hessian.size() == dim * dim);
    for (std::size_t i = 0; i < dim; ++i) {
        for (std::size_t j = i + 1; j < dim; ++j) {
            util::checkClose("stack gamma full symmetry", gamma.at(i, j), gamma.at(j, i), 1e-12);
        }
    }
    const double curvature = checkStackGammaFiniteDifference(fixture, g, HZeta, gamma, 1e-5);
    CHECK(curvature > 1e-8);
    QTA_LOG_INFO("test", "stack gamma curvature vs Gauss-Newton: {}", curvature);
    // A mismatched H_zeta is rejected.
    bool threw = false;
    try {
        (void)markets::stackQuoteGamma(*fixture.root, fixture.rootPillars, gammaRoot,
                                       std::vector<double>(dim, 0.0), fixture.reference, children);
    } catch (const std::invalid_argument&) {
        threw = true;
    }
    CHECK(threw);
    logStack("stack IRS child risk (5y root)", stack);
}

void runDepth2FiniteDifference(const char* prefix, bool exogenousRootDiscount) {
    Depth2StackFixture fixture(exogenousRootDiscount);
    const Depth2Gradients gradients = depth2Gradients(fixture);
    const std::vector<markets::StackCurveInput> inputs =
        depth2StackInputs(fixture, gradients.root, gradients.child, gradients.grand);
    const std::vector<markets::StackRiskEntry> stack =
        markets::stackQuoteRisk(inputs, fixture.reference);
    CHECK(stack.size() == 3);
    logStack(prefix, stack);

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
        util::checkClose(rootLabel.c_str(), stack[0].quoteDeltas[j], fd, tolerance);
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
        util::checkClose(childLabel.c_str(), stack[1].quoteDeltas[j], fd, tolerance);
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
        util::checkClose(grandLabel.c_str(), stack[2].quoteDeltas[j], fd, tolerance);
    }
}

void testStackDepth2ChainFiniteDifference() {
    runDepth2FiniteDifference("depth-2 chain", false);
}

void testStackDepth2ExogenousDiscountFiniteDifference() {
    runDepth2FiniteDifference("depth-2 off-parent discount", true);
}

void runDepth2GammaFiniteDifference(const char* prefix, bool exogenousRootDiscount) {
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
    CHECK(gamma.dim == dim);
    for (std::size_t i = 0; i < dim; ++i) {
        for (std::size_t j = i + 1; j < dim; ++j) {
            util::checkClose("depth-2 stack gamma symmetry", gamma.at(i, j), gamma.at(j, i), 1e-12);
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
            for (const double delta : entry.quoteDeltas) {
                gradient[index++] = delta;
            }
        }
        return gradient;
    };
    const double epsilon = 1e-5;
    double maxDeviation = 0.0;
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
            maxDeviation = std::max(maxDeviation, std::abs(gamma.at(a, r) - fd));
            util::checkClose("depth-2 exact stack gamma vs FD", gamma.at(a, r), fd, 1e-5);
        }
    }
    QTA_LOG_INFO("test", "{} max depth-2 gamma FD deviation: {}", prefix, maxDeviation);
    logMatrix(prefix, gamma.hessian, dim, dim, gamma.quoteLabels, gamma.quoteLabels);
}

void testStackDepth2GammaFiniteDifference() {
    runDepth2GammaFiniteDifference("depth-2 chain gamma", false);
}

void testStackDepth2ExogenousGammaFiniteDifference() {
    runDepth2GammaFiniteDifference("depth-2 off-parent discount gamma", true);
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

    CHECK(general.size() == legacy.size());
    for (std::size_t k = 0; k < general.size(); ++k) {
        CHECK(general[k].role == legacy[k].role);
        CHECK(general[k].quoteLabels == legacy[k].quoteLabels);
        CHECK(general[k].quoteYears == legacy[k].quoteYears);
        util::checkClose("depth-1 equivalence deltas", general[k].quoteDeltas,
                         legacy[k].quoteDeltas, 1e-12);
    }
    logStack("depth-1 general engine", general);

    // The general and legacy gamma overloads agree entry by entry.
    const std::size_t gammaDim = general[0].quoteDeltas.size() + general[1].quoteDeltas.size();
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
    CHECK(generalGamma.dim == legacyGamma.dim);
    CHECK(generalGamma.quoteLabels == legacyGamma.quoteLabels);
    CHECK(generalGamma.quoteYears == legacyGamma.quoteYears);
    CHECK(generalGamma.roles == legacyGamma.roles);
    for (std::size_t k = 0; k < generalGamma.hessian.size(); ++k) {
        util::checkClose("depth-1 equivalence gamma", generalGamma.hessian[k],
                         legacyGamma.hessian[k], 1e-12);
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
    util::checkClose("rebuildWithNode parent bump moves child discount", rebuiltChild->discount(t),
                     expectedChild.discount(t), 1e-12);
    CHECK(rebuiltChild->discount(t) != childView->discount(t));

    const markets::StackCurveView::Ptr bumpedChild = childView->rebuildWithNode(1, delta, nullptr);
    util::checkClose("rebuildWithNode child bump keeps parent discount",
                     bumpedChild->parentView()->discount(t), rootView->discount(t), 1e-15);

    bool threw = false;
    try {
        (void)grandchildView->rebuildWithNode(1, delta, rootView);
    } catch (const std::invalid_argument&) {
        threw = true;
    }
    CHECK(threw);
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
    CHECK(up->identity() != down->identity());
    CHECK(up->discount(1.5) != down->discount(1.5));
    CHECK(!markets::sameCurveView(*up, *down));
    // Equal rebuilds still match by value even though each owns its copy.
    const markets::StackCurveView::Ptr upAgain = base->rebuildWithNode(1, 1e-3, nullptr);
    CHECK(up->identity() != upAgain->identity());
    CHECK(markets::sameCurveView(*up, *upAgain));
    QTA_LOG_INFO("test", "rebuilt view identity: D(1.5) up={} down={}",
                 up->discount(1.5), down->discount(1.5));
}

void testStackGammaValidation() {
    markets::StackQuoteGamma gamma;
    gamma.dim = 2;
    gamma.hessian = {1.0, 0.0, 0.0}; // too short for dim x dim
    bool threw = false;
    try {
        (void)gamma.at(0, 0);
    } catch (const std::invalid_argument&) {
        threw = true;
    }
    CHECK(threw);
    gamma.hessian = {1.0, 0.0, 0.0, 1.0};
    util::checkClose("gamma at in-bounds", gamma.at(1, 1), 1.0, 0.0);
    threw = false;
    try {
        (void)gamma.at(2, 0);
    } catch (const std::invalid_argument&) {
        threw = true;
    }
    CHECK(threw);
    markets::StackQuoteGamma empty;
    threw = false;
    try {
        (void)empty.at(0, 0);
    } catch (const std::invalid_argument&) {
        threw = true;
    }
    CHECK(threw);
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
                                     InterpolationSpace::LogDiscount,
                                     InterpolationScheme::Linear);
    const markets::StackCurveView::Ptr view = markets::StackCurveView::make(base);
    const markets::StackCurveView::Ptr bumped = view->rebuildWithNode(1, 0.002, nullptr);
    CHECK(bumped->zeroDayCounter().convention() == datetime::DayCount::Actual360);
    std::vector<double> expectedZeros = pillars;
    expectedZeros[0] += 0.002;
    const DiscountCurve<double> expected(reference, dates, act360, expectedZeros,
                                         InterpolationSpace::LogDiscount,
                                         InterpolationScheme::Linear);
    CHECK(bumped->times() == expected.times());
    util::checkClose("rebuilt date-based discount", bumped->discount(1.5), expected.discount(1.5),
                     1e-15);

    // The spread adapter keeps the clock of its re-bound parent.
    const auto parent = std::make_shared<DiscountCurve<double>>(base);
    const markets::SpreadCurve<double> child(parent, base.times(),
                                             std::vector<double>{0.0, 0.0005, 0.0007, 0.0009},
                                             InterpolationScheme::Linear);
    const markets::StackCurveView::Ptr childView = markets::StackCurveView::make(child);
    const markets::StackCurveView::Ptr childOver = childView->rebuildWithNode(0, 0.0, bumped);
    CHECK(childOver->zeroDayCounter().convention() == datetime::DayCount::Actual360);
    CHECK(childOver->parentView() != nullptr);
    CHECK(childOver->parentView()->zeroDayCounter().convention() ==
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
                                     InterpolationSpace::LogDiscount,
                                     InterpolationScheme::Linear);
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
    const markets::SpreadCurve<double> child(parent, root.times(),
                                             std::vector<double>{0.0, 0.001, 0.0015},
                                             InterpolationScheme::Linear);
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
    CHECK(base.dim == rootBumped.dim);
    CHECK(base.dim == childBumped.dim);
    util::checkClose("act360 zero-bump root jacobian", base.jacobian, rootBumped.jacobian, 1e-14);
    util::checkClose("act360 zero-bump child jacobian", base.jacobian, childBumped.jacobian, 1e-14);
}

void testSameCurveSchemeFingerprint() {
    // Same node grid and zeros, different interpolation scheme: the discount
    // functions differ between nodes, so the curves are not interchangeable.
    IrsStackFixture fixture(5, {2, 5});
    const DiscountCurve<double> rootCopy(*fixture.root);
    CHECK(markets::sameCurveValues(rootCopy, *fixture.root));
    const DiscountCurve<double> tensionRoot(
        fixture.root->times(), fixture.root->zeros(), fixture.root->space(),
        InterpolationScheme::TensionSpline, 8.0, fixture.root->switchIndex());
    CHECK(fixture.root->discount(0.5) != tensionRoot.discount(0.5));
    CHECK(!markets::sameCurveValues(tensionRoot, *fixture.root));
    CHECK(!markets::sameCurveView(*markets::StackCurveView::make(tensionRoot),
                                  *markets::StackCurveView::make(*fixture.root)));

    std::vector<double> dVdRoot(fixture.root->size(), 0.0);
    dVdRoot[1] = 1.0;
    std::vector<double> dVdSpread(fixture.child->size(), 0.0);
    dVdSpread[1] = 1.0;
    std::vector<markets::StackChildInput> children;
    children.push_back(markets::StackChildInput{markets::CurveRole::Forecast, fixture.child.get(),
                                                fixture.childPillars, dVdSpread});
    children[0].discountCurve = &tensionRoot;
    bool threw = false;
    try {
        (void)markets::stackQuoteRisk(*fixture.root, fixture.rootPillars, dVdRoot, children,
                                      fixture.reference);
    } catch (const std::invalid_argument&) {
        threw = true;
    }
    CHECK(threw);

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
    CHECK(explicitCross == expected);
}

void testStackInputValidation() {
    Depth2StackFixture fixture(false);
    const Depth2Gradients gradients = depth2Gradients(fixture);
    const auto throwsInvalid = [](const auto& call) {
        try {
            call();
        } catch (const std::invalid_argument&) {
            return true;
        }
        return false;
    };

    {
        markets::StackCurveInput input;
        input.curve = markets::StackCurveView::make(*fixture.root);
        input.discountPillars = fixture.rootPillars;
        input.discountPillars.pop_back();
        input.dVdNodes = gradients.root;
        CHECK(throwsInvalid([&] { (void)markets::stackQuoteRisk({input}, fixture.reference); }));
    }
    {
        const std::vector<double> otherTimes{0.0, 1.0, 2.0};
        const std::vector<double> otherZeros{0.0, 0.02, 0.03};
        const DiscountCurve<double> other(otherTimes, otherZeros, InterpolationSpace::LogDiscount,
                                          InterpolationScheme::Linear);
        std::vector<markets::StackCurveInput> inputs =
            depth2StackInputs(fixture, gradients.root, gradients.child, gradients.grand);
        inputs[1].discount = markets::StackCurveView::make(other);
        CHECK(throwsInvalid([&] { (void)markets::stackQuoteRisk(inputs, fixture.reference); }));
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
        CHECK(throwsInvalid([&] { (void)markets::stackQuoteRisk({input}, fixture.reference); }));
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
        for (const double delta : entry.quoteDeltas) {
            total += delta;
        }
    }
    double ladderTotal = 0.0;
    bool sawBasis = false;
    for (const markets::RiskBucket& bucket : markets::stackYearLadder(stack)) {
        ladderTotal += bucket.delta;
        sawBasis = sawBasis || bucket.label == "4Y" || bucket.label == "5Y";
    }
    logStack("stack ladder/hedge/dual quote risk", stack);
    logBuckets("stack year ladder:", markets::stackYearLadder(stack));
    util::checkClose("stack ladder total", ladderTotal, total, 1e-12);
    CHECK(sawBasis);
    CHECK(!stack[0].quoteLabels.empty());
    CHECK(!stack[1].quoteLabels.empty());

    // Hedge solve: 2x2 sensitivity matrix, exact recovery.
    const std::vector<double> jacobian{1.0, 0.5, 0.5, 1.0};
    const std::vector<double> delta{1.0, -0.5};
    const std::vector<double> hedge = markets::solveHedge(jacobian, 2, 2, delta, 0.0);
    logDeltas("hedge solve (2 buckets, A x = -delta):", {"bucket 0", "bucket 1"}, hedge);
    util::checkClose("hedge bucket 0", jacobian[0] * hedge[0] + jacobian[1] * hedge[1], -delta[0],
                     1e-12);
    util::checkClose("hedge bucket 1", jacobian[2] * hedge[0] + jacobian[3] * hedge[1], -delta[1],
                     1e-12);

    // Dual pricing/risk scheme view: Linear pricing, Akima risk.
    const DiscountCurve<double> riskCurve =
        markets::bootstrapDiscountCurve(fx.reference, zeroDc, InterpolationSpace::LogDiscount,
                                        InterpolationScheme::Akima, fx.pillars);
    const markets::DualSchemeRisk dual =
        markets::dualSchemeQuoteRisk(*parent, riskCurve, fx.pillars, fx.reference, dVdRoot);
    const std::vector<std::string> labels = pillarLabels(fx.reference, fx.pillars);
    logDeltas("dual audit (Linear pricing):", labels, dual.audit);
    logDeltas("dual hedge (Akima risk):", labels, dual.hedge);
    logDeltas("dual basis (hedge - audit):", labels, dual.basis);
    double maxAudit = 0.0;
    double maxBasis = 0.0;
    for (std::size_t j = 0; j < dual.audit.size(); ++j) {
        util::checkClose("dual basis identity", dual.basis[j], dual.hedge[j] - dual.audit[j],
                         1e-15);
        maxAudit = std::max(maxAudit, std::abs(dual.audit[j]));
        maxBasis = std::max(maxBasis, std::abs(dual.basis[j]));
    }
    CHECK(maxBasis <= 0.3 * maxAudit + 1e-6);

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
    const std::vector<std::string> semiLabels = pillarLabels(fx.reference, semiPillars);
    logDeltas("dual audit (Linear pricing, semi-annual legs):", semiLabels, semiDual.audit);
    logDeltas("dual hedge (Akima risk, semi-annual legs):", semiLabels, semiDual.hedge);
    logDeltas("dual basis (hedge - audit, semi-annual legs):", semiLabels, semiDual.basis);
    double maxSemiBasis = 0.0;
    for (const double value : semiDual.basis) {
        maxSemiBasis = std::max(maxSemiBasis, std::abs(value));
    }
    CHECK(maxSemiBasis > 1e-8);
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
    CHECK(report.quoteDeltas.size() == fixture.pillars.size());
    CHECK(report.quoteLabels.size() == fixture.pillars.size());

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
    util::checkClose("report kind total", kindSum, report.totalDelta(), 1e-12);
    util::checkClose("report end total", endSum, report.totalDelta(), 1e-12);
    util::checkClose("report year total", yearSum, report.totalDelta(), 1e-12);
    util::checkClose("report horizon", report.horizonYears, 5.0, 0.01);
    logReport("curve risk report (deltas)", report);

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

    logReport("curve risk report (deltas + gamma)", withGamma);
    CHECK(withGamma.gammaDiagonal.size() == fixture.pillars.size());
    CHECK(!withGamma.gammaCrossByYear.empty());
    double yearGammaSum = 0.0;
    for (const markets::RiskBucket& bucket : withGamma.gammaDiagByYear) {
        yearGammaSum += bucket.delta;
    }
    double quoteGammaSum = 0.0;
    for (const double value : withGamma.gammaDiagonal) {
        quoteGammaSum += value;
    }
    util::checkClose("gamma year total", yearGammaSum, quoteGammaSum, 1e-10);
}

} // namespace

int main() {
    testNodeWeightsFiniteDifference(InterpolationSpace::Zero, InterpolationScheme::Linear);
    testNodeWeightsFiniteDifference(InterpolationSpace::LogDiscount, InterpolationScheme::Linear);
    testNodeWeightsFiniteDifference(InterpolationSpace::Zero, InterpolationScheme::Akima);
    testNodeWeightsFiniteDifference(InterpolationSpace::LogDiscount,
                                    InterpolationScheme::TensionSpline, 8.0);
    testFrontEndRiskCompleteness();
    testZeroTimeNodeWeights();
    testExtrapolatedNodeWeights();
    testQuoteRiskFiniteDifference(InterpolationSpace::LogDiscount, InterpolationScheme::Linear);
    // Akima rows are primal-pinned subgradients (branch-adaptive slopes), so
    // FD agreement is at kink tolerance rather than machine precision.
    testQuoteRiskFiniteDifference(InterpolationSpace::Zero, InterpolationScheme::Akima, 0.0, 2e-4);
    testQuoteRiskFiniteDifference(InterpolationSpace::LogDiscount,
                                  InterpolationScheme::TensionSpline, 8.0);
    testDenseLuMatchesSolveDense();
    testQuoteGamma();
    testRiskReport();
    testFraRepoGranular();
    testStackRiskTreePass();
    testStackIrsChildRiskAndGamma();
    testStackIrsGammaSymmetry();
    testStackDepth2ChainFiniteDifference();
    testStackDepth2ExogenousDiscountFiniteDifference();
    testStackDepth2GammaFiniteDifference();
    testStackDepth2ExogenousGammaFiniteDifference();
    testStackDepth1Equivalence();
    testStackRebuildWithNode();
    testRebuiltViewIdentityCollision();
    testRebuildWithNodeMetadata();
    testStackGammaValidation();
    testStackClockPreservedThroughBumps();
    testStackInputValidation();
    testSameCurveSchemeFingerprint();
    testStackLadderHedgeDual();
    testBasisRiskRepresentation();
    QTA_LOG_INFO("test", "test_curve_risk: ok");
    return 0;
}
