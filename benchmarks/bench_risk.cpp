// bench_risk.cpp — timing and allocation harness for the curve risk layer
//
// Measures the quote-risk transforms, the stack delta/gamma engine, the
// curve-risk report, the bucket aggregations and the cross-currency rows.
// Each case runs `reps` times after a warmup and reports p50/p99 wall time;
// the allocation-tracking build (bench_risk_alloc) additionally reports the
// allocation count and bytes of one run:
//   ./build/release/benchmarks/bench_risk [reps] [name-filter]
//   ./build/release/benchmarks/bench_risk_alloc [reps] [name-filter]
#include "quantape/format/Number.h"
#include "quantape/log/Log.h"
#include "quantape/markets/Curves/CurveBuilder.h"
#include "quantape/markets/Curves/CurveRisk.h"
#include "quantape/markets/Curves/CurveRiskReport.h"
#include "quantape/markets/Curves/DiscountCurve.h"
#include "quantape/markets/Curves/SpreadCurve.h"
#include "quantape/markets/Curves/StackRisk.h"
#include "quantape/markets/Curves/StackRiskRows.h"
#include "quantape/markets/Curves/TurnOverlay.h"
#include "quantape/markets/Curves/XccyRisk.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <functional>
#include <memory>
#include <new>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <variant>
#include <vector>

using namespace quantape;

namespace {

// ── allocation counters (compile-time opt-in, see bench_risk_alloc) ──
//
// The stack gamma rebuilds curve views per finite-difference bump and the row
// builders allocate scratch; counting every allocation needs the malloc/free
// symbols replaced. The counters and overrides therefore live behind
// QUANTAPE_RISK_ALLOC_COUNTS and are built into the separate bench_risk_alloc
// target, whose allocation column is the metric and whose timing column is not
// comparable to bench_risk.
#if defined(QUANTAPE_RISK_ALLOC_COUNTS)
std::atomic<std::uint64_t> g_allocations{0};
std::atomic<std::uint64_t> g_allocatedBytes{0};

void countAllocation(std::size_t bytes) {
    g_allocations.fetch_add(1, std::memory_order_relaxed);
    g_allocatedBytes.fetch_add(bytes, std::memory_order_relaxed);
}
#endif

volatile double g_sink = 0.0;

} // namespace

#if defined(QUANTAPE_RISK_ALLOC_COUNTS) && defined(__APPLE__)
#include <malloc/malloc.h>

extern "C" void* malloc(std::size_t size) {
    countAllocation(size);
    return malloc_zone_malloc(malloc_default_zone(), size);
}
extern "C" void free(void* pointer) {
    if (pointer != nullptr) {
        malloc_zone_free(malloc_default_zone(), pointer);
    }
}
#endif

#if defined(QUANTAPE_RISK_ALLOC_COUNTS)
void* operator new(std::size_t size) {
    countAllocation(size);
#if defined(__APPLE__)
    void* p = malloc_zone_malloc(malloc_default_zone(), size);
#else
    void* p = std::malloc(size);
#endif
    if (p != nullptr) {
        return p;
    }
    throw std::bad_alloc();
}
void* operator new[](std::size_t size) {
    countAllocation(size);
#if defined(__APPLE__)
    void* p = malloc_zone_malloc(malloc_default_zone(), size);
#else
    void* p = std::malloc(size);
#endif
    if (p != nullptr) {
        return p;
    }
    throw std::bad_alloc();
}
void* operator new(std::size_t size, std::align_val_t alignment) {
    countAllocation(size);
    void* p = nullptr;
    if (posix_memalign(&p, static_cast<std::size_t>(alignment), size) == 0) {
        return p;
    }
    throw std::bad_alloc();
}
void* operator new[](std::size_t size, std::align_val_t alignment) {
    return ::operator new(size, alignment);
}
void* operator new(std::size_t size, const std::nothrow_t&) noexcept {
    countAllocation(size);
#if defined(__APPLE__)
    return malloc_zone_malloc(malloc_default_zone(), size);
#else
    return std::malloc(size);
#endif
}
void* operator new[](std::size_t size, const std::nothrow_t&) noexcept {
    return ::operator new(size, std::nothrow);
}
void operator delete(void* p) noexcept {
#if defined(__APPLE__)
    if (p != nullptr) {
        malloc_zone_free(malloc_default_zone(), p);
    }
#else
    std::free(p);
#endif
}
void operator delete[](void* p) noexcept {
    ::operator delete(p);
}
void operator delete(void* p, std::size_t) noexcept {
    ::operator delete(p);
}
void operator delete[](void* p, std::size_t) noexcept {
    ::operator delete(p);
}
void operator delete(void* p, std::align_val_t) noexcept {
#if defined(__APPLE__)
    if (p != nullptr) {
        malloc_zone_free(malloc_default_zone(), p);
    }
#else
    std::free(p);
#endif
}
void operator delete[](void* p, std::align_val_t) noexcept {
    ::operator delete(p);
}
void operator delete(void* p, std::size_t, std::align_val_t) noexcept {
    ::operator delete(p);
}
void operator delete[](void* p, std::size_t, std::align_val_t) noexcept {
    ::operator delete(p);
}
void operator delete(void* p, const std::nothrow_t&) noexcept {
    ::operator delete(p);
}
void operator delete[](void* p, const std::nothrow_t&) noexcept {
    ::operator delete(p);
}
#endif // QUANTAPE_RISK_ALLOC_COUNTS

namespace {

using markets::CurvePillar;
using markets::CurveRole;
using markets::DiscountCurve;
using markets::ForecastPillar;
using markets::InterpolationScheme;
using markets::InterpolationSpace;

std::string g_filter;

bool selected(std::string_view name) {
    return g_filter.empty() || name.find(g_filter) != std::string_view::npos;
}

struct Timing {
    double p50Us = 0.0;
    double p99Us = 0.0;
    std::uint64_t allocations = 0;
    std::uint64_t allocatedBytes = 0;
};

/// Warm up once, then time `reps` runs and reduce to p50/p99. On the
/// allocation build the reported count is the smallest seen, which drops the
/// one-off allocations from returning the result into the caller's sink.
void bench(const std::string& name, int reps, const std::function<double()>& runOnce) {
    if (!selected(name)) {
        return;
    }
    g_sink = runOnce();
    std::vector<double> samples;
    samples.reserve(static_cast<std::size_t>(reps));
    Timing timing;
#if defined(QUANTAPE_RISK_ALLOC_COUNTS)
    timing.allocations = UINT64_MAX;
#endif
    for (int r = 0; r < reps; ++r) {
#if defined(QUANTAPE_RISK_ALLOC_COUNTS)
        g_allocations.store(0, std::memory_order_relaxed);
        g_allocatedBytes.store(0, std::memory_order_relaxed);
#endif
        const auto start = std::chrono::steady_clock::now();
        g_sink = runOnce();
        const auto stop = std::chrono::steady_clock::now();
        samples.push_back(std::chrono::duration<double, std::micro>(stop - start).count());
#if defined(QUANTAPE_RISK_ALLOC_COUNTS)
        const std::uint64_t allocations = g_allocations.load(std::memory_order_relaxed);
        if (allocations < timing.allocations) {
            timing.allocations = allocations;
            timing.allocatedBytes = g_allocatedBytes.load(std::memory_order_relaxed);
        }
#endif
    }
    std::sort(samples.begin(), samples.end());
    const auto quantile = [&](double q) {
        const std::size_t index =
            static_cast<std::size_t>(std::ceil(q * static_cast<double>(samples.size()))) - 1;
        return samples[std::min(index, samples.size() - 1)];
    };
    timing.p50Us = quantile(0.5);
    timing.p99Us = quantile(0.99);
    QTA_LOG_INFO("bench", "{}  p50 {} us  p99 {} us  allocs {}  {} KB", name,
                 format::num(timing.p50Us, 3), format::num(timing.p99Us, 3), timing.allocations,
                 format::num(static_cast<double>(timing.allocatedBytes) / 1024.0, 1));
}

// ── single-curve fixtures ────────────────────────────────────────────────────

struct DiscountFixture {
    datetime::Date reference{2026, 9, 29};
    datetime::DayCounter zeroDc{datetime::DayCount::Actual365Fixed};
    datetime::Calendar calendar = datetime::Calendar::noHolidays();
    std::vector<CurvePillar> pillars;
    std::shared_ptr<DiscountCurve<double>> curve;

    explicit DiscountFixture(int pillarCount) {
        std::vector<datetime::Date> pillarDates;
        pillarDates.push_back(datetime::Period(6, datetime::TimeUnit::Months).advance(reference));
        for (int years = 1; years <= pillarCount - 1; ++years) {
            pillarDates.push_back(reference.plusYears(years));
        }
        std::vector<double> targetZeros;
        targetZeros.reserve(pillarDates.size());
        for (const auto& date : pillarDates) {
            const double t = datetime::yearFraction(reference, date, zeroDc);
            targetZeros.push_back(0.04 - 0.002 * std::exp(-0.5 * t));
        }
        const DiscountCurve<double> target(reference, pillarDates, zeroDc, targetZeros,
                                           InterpolationSpace::LogDiscount,
                                           InterpolationScheme::Linear);
        CurvePillar deposit;
        deposit.maturity = pillarDates.front();
        deposit.kind = markets::PillarKind::Deposit;
        deposit.quoteDayCounter = datetime::DayCounter(datetime::DayCount::Actual360);
        deposit.calendar = calendar;
        deposit.quote = markets::impliedQuote(deposit, reference, target);
        pillars.push_back(deposit);
        for (std::size_t i = 1; i < pillarDates.size(); ++i) {
            CurvePillar swap;
            swap.maturity = pillarDates[i];
            swap.kind = markets::PillarKind::OisSwap;
            swap.quoteDayCounter = zeroDc;
            swap.calendar = calendar;
            swap.quote = markets::impliedQuote(swap, reference, target);
            pillars.push_back(swap);
        }
        curve = std::make_shared<DiscountCurve<double>>(
            markets::bootstrapDiscountCurve(reference, zeroDc, InterpolationSpace::LogDiscount,
                                            InterpolationScheme::Linear, pillars));
    }

    std::vector<double> dVdZerosAt(double t) const {
        std::vector<double> weights;
        curve->zeroNodeWeights(t, weights);
        const double df = curve->discount(t);
        std::vector<double> out(weights.size(), 0.0);
        for (std::size_t i = 0; i < weights.size(); ++i) {
            out[i] = -t * df * weights[i];
        }
        return out;
    }

    std::vector<double> nodeHessianAt(double t) const {
        std::vector<double> weights;
        curve->zeroNodeWeights(t, weights);
        const double df = curve->discount(t);
        const std::size_t n = weights.size();
        std::vector<double> out(n * n, 0.0);
        for (std::size_t i = 0; i < n; ++i) {
            for (std::size_t j = 0; j < n; ++j) {
                out[i * n + j] = t * t * df * weights[i] * weights[j];
            }
        }
        return out;
    }
};

// ── stack fixtures ───────────────────────────────────────────────────────────

struct StackFixture {
    datetime::Date reference{2026, 9, 29};
    datetime::DayCounter zeroDc{datetime::DayCount::Actual365Fixed};
    datetime::Calendar calendar = datetime::Calendar::noHolidays();
    std::shared_ptr<DiscountCurve<double>> root;
    std::vector<CurvePillar> rootPillars;
    std::shared_ptr<markets::SpreadCurve<double>> child;
    std::vector<ForecastPillar> childPillars;
    std::vector<markets::StackCurveInput> inputs;
    std::vector<double> HZeta;
    std::size_t dim = 0;

    StackFixture(int rootYears, int childYears) {
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
            swap.kind = markets::PillarKind::OisSwap;
            swap.quoteDayCounter = zeroDc;
            swap.calendar = calendar;
            swap.quote = markets::impliedQuote(swap, reference, target);
            rootPillars.push_back(swap);
        }
        root = std::make_shared<DiscountCurve<double>>(
            markets::bootstrapDiscountCurve(reference, zeroDc, InterpolationSpace::LogDiscount,
                                            InterpolationScheme::Linear, rootPillars));
        for (int year = 1; year <= childYears; ++year) {
            ForecastPillar instrument;
            instrument.kind = ForecastPillar::Kind::Irs;
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

        const double t = 4.5;
        std::vector<double> rootWeights;
        root->zeroNodeWeights(t, rootWeights);
        std::vector<double> rootSensitivity(root->size(), 0.0);
        for (std::size_t i = 0; i < rootWeights.size(); ++i) {
            rootSensitivity[i] = -t * root->discount(t) * rootWeights[i];
        }
        std::vector<double> childWeights;
        child->zeroNodeWeights(t, childWeights);
        std::vector<double> childSensitivity(child->size(), 0.0);
        for (std::size_t i = 0; i < childWeights.size(); ++i) {
            childSensitivity[i] = -t * child->discount(t) * childWeights[i];
        }

        inputs.resize(2);
        inputs[0].curve = markets::StackCurveView::make(*root);
        inputs[0].role = CurveRole::Discount;
        inputs[0].discountPillars = rootPillars;
        inputs[0].dVdNodes = std::move(rootSensitivity);
        inputs[1].curve = markets::StackCurveView::make(*child);
        inputs[1].role = CurveRole::Forecast;
        inputs[1].forecastPillars = childPillars;
        inputs[1].dVdNodes = std::move(childSensitivity);

        dim = rootPillars.size() + childPillars.size();
        HZeta.assign(dim * dim, 0.0);
        for (std::size_t i = 0; i < dim; ++i) {
            for (std::size_t j = 0; j < dim; ++j) {
                HZeta[i * dim + j] =
                    i == j ? 1.0e-2
                           : 1.0e-4 /
                                 (1.0 + std::abs(static_cast<double>(i) - static_cast<double>(j)));
            }
        }
    }
};

void benchStack(const StackFixture& fixture, int reps) {
    const std::string suffix = std::to_string(fixture.dim) + " dim";
    bench("stackQuoteRisk " + suffix, reps, [&] {
        const std::vector<markets::StackRiskEntry> entries =
            markets::stackQuoteRisk(fixture.inputs, fixture.reference);
        return entries.back().points.back().delta;
    });
    bench("stackQuoteGamma " + suffix, reps, [&] {
        const markets::StackQuoteGamma gamma =
            markets::stackQuoteGamma(fixture.inputs, fixture.HZeta, fixture.reference);
        return gamma.hessian.back();
    });
    bench("assembleStackQuoteSystem " + suffix, reps, [&] {
        const markets::StackQuoteSystem system =
            markets::assembleStackQuoteSystem(fixture.inputs, fixture.reference);
        return system.jacobian.back();
    });
    bench("bumpStackInputs " + suffix, reps, [&] {
        const std::vector<markets::StackCurveInput> bumped =
            markets::bumpStackInputs(fixture.inputs, 0, 1, 1e-6);
        return bumped[0].curve->discount(1.0);
    });
}

// ── cross-currency fixture ───────────────────────────────────────────────────

struct XccyFixture {
    datetime::Date reference{2026, 9, 29};
    datetime::DayCounter zeroDc{datetime::DayCount::Actual365Fixed};
    datetime::Calendar calendar = datetime::Calendar::noHolidays();
    std::optional<DiscountCurve<double>> domesticDiscount;
    std::optional<DiscountCurve<double>> foreignDiscount;
    std::shared_ptr<DiscountCurve<double>> domesticBase;
    std::shared_ptr<DiscountCurve<double>> foreignBase;
    std::shared_ptr<markets::SpreadCurve<double>> domesticForecast;
    std::shared_ptr<markets::SpreadCurve<double>> foreignForecast;
    std::vector<markets::XccyPillar> pillars;

    XccyFixture(int pillarCount) {
        std::vector<datetime::Date> dates{reference};
        for (int years = 1; years <= pillarCount; ++years) {
            dates.push_back(reference.plusYears(years));
        }
        const auto build = [&](double base, double slope) {
            std::vector<double> zeros;
            zeros.reserve(dates.size() - 1);
            for (std::size_t i = 1; i < dates.size(); ++i) {
                zeros.push_back(base + slope * datetime::yearFraction(reference, dates[i], zeroDc));
            }
            return DiscountCurve<double>(
                reference, std::vector<datetime::Date>(dates.begin() + 1, dates.end()), zeroDc,
                zeros, InterpolationSpace::LogDiscount, InterpolationScheme::Linear);
        };
        domesticDiscount = build(0.040, 0.0005);
        foreignDiscount = build(0.028, 0.0006);
        domesticBase = std::make_shared<DiscountCurve<double>>(build(0.041, 0.0004));
        foreignBase = std::make_shared<DiscountCurve<double>>(build(0.024, 0.0007));
        std::vector<double> spreadTimes{0.0};
        std::vector<double> domesticSpreads{0.0};
        std::vector<double> foreignSpreads{0.0};
        for (std::size_t i = 1; i < dates.size(); ++i) {
            spreadTimes.push_back(datetime::yearFraction(reference, dates[i], zeroDc));
            domesticSpreads.push_back(0.0005 + 0.0001 * static_cast<double>(i));
            foreignSpreads.push_back(0.0002 + 0.0001 * static_cast<double>(i));
        }
        domesticForecast = std::make_shared<markets::SpreadCurve<double>>(domesticBase, spreadTimes,
                                                                          domesticSpreads);
        foreignForecast = std::make_shared<markets::SpreadCurve<double>>(foreignBase, spreadTimes,
                                                                         foreignSpreads);
        for (std::size_t i = 1; i < dates.size(); ++i) {
            markets::XccyPillar pillar;
            pillar.maturity = dates[i];
            pillar.foreignTenor = datetime::Period(3, datetime::TimeUnit::Months);
            pillar.foreignPaymentLag = 2;
            pillar.domesticPaymentLag = 1;
            pillar.foreignBusinessDayConvention = datetime::BusinessDayConvention::Following;
            pillar.foreignCalendar = calendar;
            pillar.foreignDayCounter = datetime::DayCounter(datetime::DayCount::Actual360);
            pillar.domesticTenor = datetime::Period(3, datetime::TimeUnit::Months);
            pillar.domesticCalendar = calendar;
            pillar.domesticDayCounter = datetime::DayCounter(datetime::DayCount::Actual360);
            pillars.push_back(pillar);
        }
    }
};

void benchXccy(const XccyFixture& fixture, int reps) {
    const std::size_t pillarCount = fixture.pillars.size();
    const std::string suffix = std::to_string(pillarCount) + " pillars";
    bench("xccy rows view " + suffix, reps, [&] {
        const markets::StackCurveView::Ptr foreignDiscountView =
            markets::StackCurveView::make(*fixture.foreignDiscount);
        const markets::StackCurveView::Ptr foreignForecastView =
            markets::StackCurveView::make(*fixture.foreignForecast);
        const markets::StackCurveView::Ptr domesticDiscountView =
            markets::StackCurveView::make(*fixture.domesticDiscount);
        const markets::StackCurveView::Ptr domesticForecastView =
            markets::StackCurveView::make(*fixture.domesticForecast);
        double sink = 0.0;
        for (const markets::XccyPillar& pillar : fixture.pillars) {
            const std::vector<markets::XccyRowBlock> blocks = markets::xccySwapJacobianRowsView(
                *foreignDiscountView, *foreignForecastView, *domesticDiscountView,
                *domesticForecastView, pillar, fixture.reference, fixture.zeroDc);
            sink += blocks.back().row.back();
        }
        return sink;
    });
    bench("xccy assemble full " + suffix, reps, [&] {
        std::vector<double> f;
        std::vector<double> c;
        std::vector<double> g;
        std::vector<double> h;
        markets::assembleXccyJacobianFull(*fixture.foreignDiscount, *fixture.foreignForecast,
                                          *fixture.domesticDiscount, *fixture.domesticForecast,
                                          fixture.pillars, fixture.reference, f, c, g, h);
        return f.back() + c.back() + g.back() + h.back();
    });
}

void benchXccyStack(int reps) {
    constexpr int kRootYears = 4;
    constexpr int kPillarCount = 4;
    StackFixture stackFixture(kRootYears, 1);
    XccyFixture xccyFixture(kPillarCount);
    // The forecast spreads are parented on the stack root itself, so their
    // parent-chain row lands on the root block already present in the stack.
    const std::vector<double>& rootTimes = stackFixture.root->times();
    std::vector<double> foreignSpreads{0.0};
    std::vector<double> domesticSpreads{0.0};
    for (std::size_t i = 1; i < rootTimes.size(); ++i) {
        foreignSpreads.push_back(0.0002 + 0.0001 * static_cast<double>(i));
        domesticSpreads.push_back(0.0005 + 0.0001 * static_cast<double>(i));
    }
    const markets::SpreadCurve<double> foreignForecast(stackFixture.root, rootTimes, foreignSpreads,
                                                       InterpolationScheme::Linear);
    const markets::SpreadCurve<double> domesticForecast(
        stackFixture.root, rootTimes, domesticSpreads, InterpolationScheme::Linear);
    std::vector<double> dVdForeignZeros(xccyFixture.foreignDiscount->size(), 0.0);
    const double t = 4.5;
    std::vector<double> weights;
    xccyFixture.foreignDiscount->zeroNodeWeights(t, weights);
    const double df = xccyFixture.foreignDiscount->discount(t);
    for (std::size_t i = 0; i < weights.size(); ++i) {
        dVdForeignZeros[i] = -t * df * weights[i];
    }
    using Child =
        markets::XccyChildInput<markets::SpreadCurve<double>, markets::SpreadCurve<double>>;
    Child child;
    child.foreignDiscount = &*xccyFixture.foreignDiscount;
    child.pillars = xccyFixture.pillars;
    child.dVdForeignZeros = dVdForeignZeros;
    child.foreignForecast = &foreignForecast;
    child.domesticForecast = &domesticForecast;
    const std::vector<std::variant<Child>> children{child};
    const std::string suffix = std::to_string(kRootYears + kPillarCount) + " dim";
    bench("stackQuoteRiskXccy " + suffix, reps, [&] {
        const std::vector<markets::StackRiskEntry> entries = markets::stackQuoteRiskXccy(
            *stackFixture.root, stackFixture.rootPillars, stackFixture.inputs[0].dVdNodes, children,
            stackFixture.reference);
        return entries.back().points.back().delta;
    });
}

void benchSingleCurve(int reps) {
    constexpr int kPillars = 21;
    const DiscountFixture fixture(kPillars);
    const std::size_t dim = fixture.pillars.size();
    const std::vector<double> dVdZeros = fixture.dVdZerosAt(10.0);
    const std::vector<double> HZeta = fixture.nodeHessianAt(10.0);
    const std::string suffix = std::to_string(dim) + " pillars";
    bench("transformQuoteRisk " + suffix, reps, [&] {
        const markets::QuoteRisk risk = markets::transformQuoteRisk(*fixture.curve, fixture.pillars,
                                                                    fixture.reference, dVdZeros);
        return risk.quoteDeltas.back();
    });
    bench("transformQuoteGamma " + suffix, reps, [&] {
        const markets::QuoteGamma gamma =
            markets::transformQuoteGamma(*fixture.curve, fixture.pillars, fixture.reference, HZeta);
        return gamma.hessian.back();
    });
    bench("curveRiskReport " + suffix, reps, [&] {
        const markets::CurveRiskReport report =
            markets::curveRiskReport(*fixture.curve, fixture.pillars, fixture.reference, dVdZeros);
        return report.totalDelta();
    });
    bench("curveRiskReport+hessian " + suffix, reps, [&] {
        const markets::CurveRiskReport report = markets::curveRiskReport(
            *fixture.curve, fixture.pillars, fixture.reference, dVdZeros, {}, &HZeta);
        return report.gammaDiagonal.back();
    });
}

void benchAggregations(int reps) {
    StackFixture fixture(6, 6);
    const std::vector<markets::StackRiskEntry> entries =
        markets::stackQuoteRisk(fixture.inputs, fixture.reference);
    bench("stackRoleBuckets " + std::to_string(fixture.dim) + " dim", reps, [&] {
        const std::vector<markets::RiskBucket> buckets = markets::stackRoleBuckets(entries);
        return buckets.back().delta;
    });
    bench("stackYearLadder " + std::to_string(fixture.dim) + " dim", reps, [&] {
        const std::vector<markets::RiskBucket> buckets = markets::stackYearLadder(entries);
        return buckets.back().delta;
    });
}

} // namespace

int main(int argc, char** argv) {
    int reps = 50;
    if (argc > 1) {
        reps = std::max(1, std::atoi(argv[1]));
    }
    if (argc > 2) {
        g_filter = argv[2];
    }
    for (const auto& item : {std::pair<int, int>{4, 3}, {6, 6}, {10, 10}}) {
        const StackFixture fixture(item.first, item.second);
        benchStack(fixture, reps);
    }
    benchSingleCurve(reps);
    benchAggregations(reps);
    const XccyFixture xccyFixture(5);
    benchXccy(xccyFixture, reps);
    benchXccyStack(reps);
    QTA_LOG_INFO("bench", "done");
    return 0;
}
